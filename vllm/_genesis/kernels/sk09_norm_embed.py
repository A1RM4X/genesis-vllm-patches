# SPDX-License-Identifier: Apache-2.0
"""SK-09 — NORM_EMBED — RMSNorm + quant fusionado y embed passthrough, en GPU.

RMSNorm + quant per-token INT8 en **una sola pasada**: la fila entra una vez a
registros y se reusa para la varianza, el ``amax`` y la cuantización. Es el
productor natural de los GEMM INT8 (SK-01/03/05/07/10): entrega ``(q, s)`` sin
que ninguno de ellos tenga que recalcular estadísticas de fila.

    y = x * rsqrt(mean(x^2) + eps) * ((w + GAMMA_OFFSET) * s_pow2)
    s = amax(|y|) / 127
    q = clamp(round(y / s), -127, 127)

Corrección respecto de la versión anterior
------------------------------------------
La versión anterior fijaba ``IS_GEMMA=1`` como ``constexpr`` en el wrapper, sin
ningún parámetro para desactivarlo, de modo que el kernel siempre calculaba
``w_eff = 1 + w`` (semántica Gemma). Qwen3.5 usa RMSNorm plana (``w``, sin el
``+1``): medido contra una RMSNorm estándar, la diferencia llegaba a **101
niveles INT8 de 127**. Ahora la semántica es el parámetro ``gamma_offset``
(0.0 plana, 1.0 Gemma) y por defecto es **0.0**.

También se fue todo el andamiaje que no aportaba: el memo de lanzamiento con
seguimiento de alineación de ``data_ptr``, el buffer único con la escala
empotrada a 16 B, el camino ``_wide`` con ``tl.dot(..., input_precision="ieee")``
para reproducir bit a bit el orden de reducción de ATen (una reducción de suma
de cuadrados no necesita Tensor Cores), y el fallback torch. El kernel resultante
es una sola pasada por fila.

``s_pow2`` es el vector SmoothQuant restringido a potencias de dos absorbido en
la norma. Cuando no hay, el llamador no pasa nada y el puntero apunta a un
escalar 1.0 con stride 0: broadcast gratis, sin rama.

Los kernels son branchless y no validan nada: se asume todo comprobado.
"""

from __future__ import annotations
import ctypes
import os
import subprocess
import tempfile
import threading

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Plomeria PTX. DUPLICADA A PROPOSITO en cada archivo de kernel: este modulo no
# importa plomeria compartida de ningun lado. Si hay que arreglar lo mismo en
# 18 archivos, se arregla en 18 archivos; es la decision explicita del
# proyecto -- codigo repetido antes que codigo compartido.
# ---------------------------------------------------------------------------

_ARCH = 86
_OPT = 3
# 48 KB es el maximo de shared DINAMICA por defecto en sm_86. Los tiles de
# prefill piden 73728 B (72 KB), asi que hay que pedirlo explicito con
# cuFuncSetAttribute o el launch falla con CUDA_ERROR_INVALID_VALUE (1).
_TECHO_SHARED = 49152
_ATRIB_MAX_SHARED = 8   # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES

_cerrojo = threading.Lock()
_libcuda_cache = None


def _libcuda():
    """libcuda cruda. El contexto lo inicializa torch en su primer op de GPU."""
    global _libcuda_cache
    if _libcuda_cache is None:
        _libcuda_cache = ctypes.CDLL("libcuda.so.1")
    return _libcuda_cache


def _ruta_ptxas() -> str:
    """El ptxas que trae Triton, no el del sistema.

    Triton emite PTX con una version de ISA concreta; el ptxas del CUDA
    instalado puede ser mas viejo o mas nuevo y rechazarlo.
    """
    base = os.path.dirname(__import__("triton").__file__)
    cand = os.path.join(base, "backends", "nvidia", "bin", "ptxas")
    return cand if os.path.exists(cand) else "ptxas"


def _ensamblar(ptx: str) -> bytes:
    """PTX -> cubin. Levanta RuntimeError con el stderr completo si falla."""
    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, "k.ptx")
        fc = os.path.join(d, "k.cubin")
        with open(fp, "w") as f:
            f.write(ptx)
        r = subprocess.run([_ruta_ptxas(), "-arch=sm_%d" % _ARCH, "-O%d" % _OPT,
                            fp, "-o", fc], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("ptxas fallo:\n" + r.stderr)
        with open(fc, "rb") as f:
            return f.read()


class _Nativo:
    """Una variante de PTX embebida: ensambla bajo demanda y lanza por libcuda.

    :param caso: id de diagnostico.
    :param ptx: el asm completo.
    :param entry: nombre de la funcion dentro del modulo.
    :param warps: warps por bloque (blockDim.x = warps * 32).
    :param shared: bytes de shared dinamica que pide el cubin.
    :param abi: indices, sobre la lista completa de args del kernel original,
        de los params que SI viajan en el ABI, en orden. Salen de
        ``handle.src.constants``: son los que NO quedaron especializados.
    :param horneado: ``posicion -> valor`` de los params horneados en el cubin.
        Se verifican en cada lanzamiento: si llega otro valor el resultado
        seria incorrecto EN SILENCIO, asi que se levanta ValueError.
    :param div16: posiciones cuyo valor tiene que seguir siendo multiplo de 16.
        Triton lo asume al compilar (``tt.divisibility``) y lo usa para
        vectorizar los accesos. Con un valor que no cumple sale mal en
        silencio: el JIT recompilaria, el PTX embebido no puede.
    """

    def __init__(self, caso, ptx, entry, warps, shared, abi, horneado, div16):
        self.caso = caso
        self.ptx = ptx
        self.entry = entry
        self.warps = warps
        self.shared = shared
        self.abi = abi
        self.horneado = horneado
        self.div16 = div16
        self._fn = None
        self._mod = None

    def _cargar(self):
        """Camino rapido: si ya esta cargado no hace nada.

        El ensamblado vive aparte y marcado con ``torch.compiler.disable``
        porque usa un lock, y dynamo no sabe entrar a un context manager de
        `lock`: falla con "Unsupported context manager". El lanzamiento cae
        DENTRO de la region que vLLM compila (el forward del modelo), asi que
        con el `with` en el camino trazado el arranque se muere en
        ``profile_run`` -> ``_dummy_run``. Medido: ensamblar y cargar cuesta
        ~25 ms por variante y pasa una sola vez, asi que el corte de grafo del
        camino lento no se paga en regimen.
        """
        if self._fn is None:
            self._ensamblar_y_cargar()

    @torch.compiler.disable
    def _ensamblar_y_cargar(self):
        """Ensambla con ptxas y carga el modulo. Una sola vez, thread-safe."""
        with _cerrojo:
            if self._fn is not None:
                return
            cubin = _ensamblar(self.ptx)
            img = (ctypes.c_char * len(cubin)).from_buffer_copy(cubin)
            mod = ctypes.c_void_p()
            res = _libcuda().cuModuleLoadData(ctypes.byref(mod), img)
            if res != 0:
                raise RuntimeError("cuModuleLoadData(%s): %d" % (self.caso, res))
            fn = ctypes.c_void_p()
            res = _libcuda().cuModuleGetFunction(ctypes.byref(fn), mod,
                                                 self.entry.encode())
            if res != 0:
                raise RuntimeError("cuModuleGetFunction(%s): %d" % (self.caso, res))
            if self.shared > _TECHO_SHARED:
                res = _libcuda().cuFuncSetAttribute(
                    fn, _ATRIB_MAX_SHARED, ctypes.c_int(self.shared))
                if res != 0:
                    raise RuntimeError("cuFuncSetAttribute(%s, %dB): %d"
                                       % (self.caso, self.shared, res))
            self._mod, self._fn = mod, fn

    def __call__(self, grid, *args):
        """Lanza. ``args`` es la lista COMPLETA en orden de declaracion del
        kernel original; se consumen las posiciones de ``abi``."""
        self._cargar()
        for pos, val in self.horneado.items():
            if args[pos] != val:
                raise ValueError(
                    "%s: el cubin esta horneado con param[%d]=%r y llego %r; "
                    "este asm no aplica a estos inputs"
                    % (self.caso, pos, val, args[pos]))
        for pos in self.div16:
            if args[pos] % 16:
                raise ValueError(
                    "%s: param[%d]=%r tiene que ser multiplo de 16 (Triton lo "
                    "asumio al compilar para vectorizar)"
                    % (self.caso, pos, args[pos]))
        vals = []
        for i in self.abi:
            v = args[i]
            if isinstance(v, torch.Tensor):
                vals.append(ctypes.c_uint64(v.data_ptr()))
            elif isinstance(v, float):
                vals.append(ctypes.c_float(v))
            else:
                vals.append(ctypes.c_int32(int(v)))
        # ABI de Triton: params del kernel + global_scratch + profile_scratch.
        # global_scratch_size es 0 en estos kernels, asi que NULL es seguro.
        vals.append(ctypes.c_uint64(0))
        vals.append(ctypes.c_uint64(0))
        arr = (ctypes.c_void_p * len(vals))(
            *[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in vals])
        gx, gy, gz = (list(grid) + [1, 1])[:3]
        res = _libcuda().cuLaunchKernel(
            self._fn, gx, gy, gz, self.warps * 32, 1, 1,
            ctypes.c_uint(self.shared),
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
            arr, None)
        if res != 0:
            raise RuntimeError("cuLaunchKernel(%s): %d" % (self.caso, res))





SK_ID = "SK-09"
SK_NAME = "NORM_EMBED"
BLOCK_MAX: int = 16384

_ONE: dict[torch.device, torch.Tensor] = {}


def _one(device: torch.device) -> torch.Tensor:
    """Escalar 1.0 por device, usado como ``s_pow2`` neutro con stride 0."""
    o = _ONE.get(device)
    if o is None:
        o = torch.ones((), dtype=torch.float32, device=device)
        _ONE[device] = o
    return o


@triton.jit
def _sk09_rmsnorm_quant_kernel(
    x_ptr, w_ptr, s_pow2_ptr, q_ptr, s_ptr,
    K, stride_xm, stride_qm, stride_sp,
    BLOCK: tl.constexpr, EPS: tl.constexpr, GAMMA_OFFSET: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < K
    x = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32) + GAMMA_OFFSET
    w = w * tl.load(s_pow2_ptr + offs * stride_sp, mask=mask, other=1.0).to(tl.float32)
    y = x * tl.rsqrt(tl.sum(x * x, axis=0) / K + EPS) * w
    amax = tl.maximum(tl.max(tl.abs(y), axis=0), 1e-30)
    yq = y * (127.0 / amax)
    q = (yq + tl.where(yq >= 0.0, 0.5, -0.5)).to(tl.int32)
    tl.store(q_ptr + row * stride_qm + offs, tl.minimum(tl.maximum(q, -127), 127).to(tl.int8), mask=mask)
    tl.store(s_ptr + row, amax * (1.0 / 127.0))


@triton.jit
def _sk09_quant_kernel(x_ptr, q_ptr, s_ptr, K, stride_xm, stride_qm, BLOCK: tl.constexpr):
    """Quant per-token amax/127 sin norma delante, una fila por programa."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < K
    x = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-30)
    xq = x * (127.0 / amax)
    q = (xq + tl.where(xq >= 0.0, 0.5, -0.5)).to(tl.int32)
    tl.store(q_ptr + row * stride_qm + offs, tl.minimum(tl.maximum(q, -127), 127).to(tl.int8), mask=mask)
    tl.store(s_ptr + row, amax * (1.0 / 127.0))


@triton.jit
def _sk09_rmsnorm_kernel(
    x_ptr, w_ptr, out_ptr, K, stride_xm, stride_om,
    BLOCK: tl.constexpr, EPS: tl.constexpr, GAMMA_OFFSET: tl.constexpr,
):
    """RMSNorm sola, salida bf16, para las normas que no alimentan un GEMM INT8."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < K
    x = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32) + GAMMA_OFFSET
    y = x * tl.rsqrt(tl.sum(x * x, axis=0) / K + EPS) * w
    tl.store(out_ptr + row * stride_om + offs, y.to(tl.bfloat16), mask=mask)





def _sk09_rmsnorm_quant_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    s_pow2: torch.Tensor | None = None,
    eps: float = 1e-6,
    gamma_offset: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Referencia para tests: lanza el kernel Triton original (JIT)."""
    k = x.shape[-1]
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    s = torch.empty((m,), dtype=torch.float32, device=x.device)
    sp = _one(x.device) if s_pow2 is None else s_pow2
    _sk09_rmsnorm_quant_kernel[(m,)](
        x2, weight, sp, q, s,
        k, x2.stride(0), q.stride(0), 0 if s_pow2 is None else sp.stride(0),
        BLOCK=triton.next_power_of_2(k), EPS=eps, GAMMA_OFFSET=gamma_offset,
        num_warps=8, num_stages=1,
    )
    return q, s


def rmsnorm_quant_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    s_pow2: torch.Tensor | None = None,
    eps: float = 1e-6,
    gamma_offset: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm + quant per-token -> ``(q [M,K] int8, s [M] fp32)``.

    ``gamma_offset``: 0.0 RMSNorm plana (Qwen3.5), 1.0 semántica Gemma.

    Producción: PTX embebido vía libcuda cruda, sin JIT de Triton. Con una
    geometría/semántica fuera del cubin horneado (``BLOCK != 8192``,
    """
    k = x.shape[-1]
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    s = torch.empty((m,), dtype=torch.float32, device=x.device)
    sp = _one(x.device) if s_pow2 is None else s_pow2
    _lanzar_quant1(
        (m,), x2, weight, sp, q, s,
        k, x2.stride(0), q.stride(0), 0 if s_pow2 is None else sp.stride(0),
        triton.next_power_of_2(k), eps, gamma_offset,
    )
    return q, s
    return _sk09_rmsnorm_quant_triton(x, weight, s_pow2, eps, gamma_offset)





def _sk09_quant_triton(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Referencia para tests: lanza el kernel Triton original (JIT)."""
    k = x.shape[-1]
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    s = torch.empty((m,), dtype=torch.float32, device=x.device)
    _sk09_quant_kernel[(m,)](
        x2, q, s, k, x2.stride(0), q.stride(0),
        BLOCK=triton.next_power_of_2(k), num_warps=8, num_stages=1,
    )
    return q.view(x.shape), s.view(*x.shape[:-1], 1)


def quant_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quant per-token INT8 -> ``(q [..., K] int8, s [..., 1] fp32)``.

    Sin tope de K: ``BLOCK = next_power_of_2(K)``. ``fused_quant_triton`` tenía
    ``BLOCK_MAX = 8192`` y un ``assert n <= BLOCK_MAX``, pero ``down_proj`` de
    Qwen3.5-27B tiene **K = 8704** por rank (TP=2), así que ese assert hacía
    reventar el ``apply`` de PN110 en las 65 capas ``down_proj`` — y sin
    fallback, porque el peso Marlin ya había sido liberado.

    Producción: PTX embebido vía libcuda cruda, sin JIT de Triton. Con una
    geometría fuera del cubin horneado (``BLOCK != 16384``) o con el
    """
    k = x.shape[-1]
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    s = torch.empty((m,), dtype=torch.float32, device=x.device)
    _lanzar_quant0(
        (m,), x2, q, s, k, x2.stride(0), q.stride(0),
        triton.next_power_of_2(k),
    )
    return q.view(x.shape), s.view(*x.shape[:-1], 1)
    return _sk09_quant_triton(x)


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    gamma_offset: float = 0.0,
) -> torch.Tensor:
    """RMSNorm sola -> bf16, misma forma que ``x``."""
    k = x.shape[-1]
    x2 = x.reshape(-1, k)
    out = torch.empty_like(x2, dtype=torch.bfloat16)
    _lanzar_quant2((x2.shape[0],),
            x2, weight, out, k, x2.stride(0), out.stride(0), triton.next_power_of_2(k), eps)
    return out.view(x.shape)


def embed_tokens_bf16_passthrough(embed_weight: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Embed lookup. Es un gather puro sobre bf16: ``index_select`` ya es óptimo."""
    return embed_weight.index_select(0, input_ids.reshape(-1)).view(*input_ids.shape, embed_weight.shape[1])


fused_rmsnorm_quant = rmsnorm_quant_fused
sk09_rmsnorm_quant = rmsnorm_quant_fused

__all__ = [
    "SK_ID", "SK_NAME", "BLOCK_MAX",
    "rmsnorm_quant_fused", "fused_rmsnorm_quant", "sk09_rmsnorm_quant",
    "rmsnorm", "quant_per_token", "embed_tokens_bf16_passthrough",
]


# --- quant _sk09_quant_kernel: 4 variante(s) de PTX embebido ---

_QPTX0_0 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk09_quant_kernel      // -- Begin function _sk09_quant_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk09_quant_kernel
.visible .entry _sk09_quant_kernel(
	.param .u64 .ptr .global .align 1 _sk09_quant_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk09_quant_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk09_quant_kernel_param_2,
	.param .u32 _sk09_quant_kernel_param_3,
	.param .u32 _sk09_quant_kernel_param_4,
	.param .u32 _sk09_quant_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk09_quant_kernel_param_6,
	.param .u64 .ptr .global .align 1 _sk09_quant_kernel_param_7
)
.reqntid 256
{
	.reg .pred 	%p<15>;
	.reg .b16 	%rs<9>;
	.reg .b32 	%r<125>;
	.reg .b64 	%rd<11>;
	.loc	1 84 0                          // sk09_norm_embed.py:84:0
$L__func_begin0:
	.loc	1 84 0                          // sk09_norm_embed.py:84:0

// %bb.0:
	ld.param.b64 	%rd4, [_sk09_quant_kernel_param_0];
	ld.param.b64 	%rd5, [_sk09_quant_kernel_param_1];
$L__tmp0:
	.loc	1 86 24                         // sk09_norm_embed.py:86:24
	mov.u32 	%r14, %ctaid.x;
	ld.param.b64 	%rd6, [_sk09_quant_kernel_param_2];
	.loc	1 87 24                         // sk09_norm_embed.py:87:24
	mov.u32 	%r15, %tid.x;
	and.b32 	%r16, %r15, 255;
	ld.param.b32 	%r17, [_sk09_quant_kernel_param_3];
	and.b32 	%r18, %r15, 31;
	ld.param.b32 	%r19, [_sk09_quant_kernel_param_4];
	shr.u32 	%r20, %r15, 5;
	ld.param.b32 	%r21, [_sk09_quant_kernel_param_5];
	shl.b32 	%r22, %r15, 3;
	and.b32 	%r23, %r22, 2040;
	.loc	1 88 18                         // sk09_norm_embed.py:88:18
	setp.lt.s32 	%p1, %r23, %r17;
	.loc	1 89 30                         // sk09_norm_embed.py:89:30
	mul.lo.s32 	%r24, %r19, %r14;
	.loc	1 89 24                         // sk09_norm_embed.py:89:24
	mad.wide.s32 	%rd7, %r24, 2, %rd4;
	.loc	1 89 42                         // sk09_norm_embed.py:89:42
	cvt.u64.u32 	%rd8, %r23;
	mad.wide.u32 	%rd1, %r23, 2, %rd7;
	mov.b32 	%r5, 0;
	.loc	1 89 16                         // sk09_norm_embed.py:89:16
	// begin inline asm
	mov.u32 %r1, %r5;
	mov.u32 %r2, %r5;
	mov.u32 %r3, %r5;
	mov.u32 %r4, %r5;
	@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
	// end inline asm
$L__tmp1:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	setp.eq.b32 	%p2, %r18, 0;
	shr.u32 	%r25, %r15, 3;
	and.b32 	%r26, %r25, 28;
	mov.b32 	%r27, global_smem;
	add.s32 	%r6, %r27, %r26;
	setp.lt.u32 	%p3, %r16, 8;
	shl.b32 	%r28, %r16, 2;
	add.s32 	%r9, %r27, %r28;
	and.b32 	%r29, %r15, 7;
	setp.eq.b32 	%p6, %r29, 0;
	and.pred 	%p4, %p3, %p6;
$L__tmp2:
	.loc	1 93 27                         // sk09_norm_embed.py:93:27
	mul.lo.s32 	%r30, %r21, %r14;
	.loc	1 93 21                         // sk09_norm_embed.py:93:21
	cvt.s64.s32 	%rd9, %r30;
	add.s64 	%rd10, %rd5, %rd9;
	.loc	1 93 39                         // sk09_norm_embed.py:93:39
	add.s64 	%rd2, %rd10, %rd8;
	.loc	1 89 73                         // sk09_norm_embed.py:89:73
	mov.b32 	{%rs1, %rs2}, %r4;
	cvt.f32.bf16 	%r31, %rs2;
	cvt.f32.bf16 	%r32, %rs1;
	mov.b32 	{%rs3, %rs4}, %r3;
	cvt.f32.bf16 	%r33, %rs4;
	cvt.f32.bf16 	%r34, %rs3;
	.loc	1 90 36                         // sk09_norm_embed.py:90:36
	abs.f32 	%r35, %r34;
	abs.f32 	%r36, %r33;
	abs.f32 	%r37, %r32;
	abs.f32 	%r38, %r31;
	.loc	1 89 73                         // sk09_norm_embed.py:89:73
	mov.b32 	{%rs5, %rs6}, %r2;
	cvt.f32.bf16 	%r39, %rs6;
	cvt.f32.bf16 	%r40, %rs5;
	mov.b32 	{%rs7, %rs8}, %r1;
	cvt.f32.bf16 	%r41, %rs8;
	cvt.f32.bf16 	%r42, %rs7;
	.loc	1 90 36                         // sk09_norm_embed.py:90:36
	abs.f32 	%r43, %r42;
	abs.f32 	%r44, %r41;
	abs.f32 	%r45, %r40;
	abs.f32 	%r46, %r39;
$L__tmp3:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r47, %r43, %r44;
	max.f32 	%r48, %r47, %r45;
	max.f32 	%r49, %r48, %r46;
	max.f32 	%r50, %r49, %r35;
	max.f32 	%r51, %r50, %r36;
	max.f32 	%r52, %r51, %r37;
	max.f32 	%r53, %r52, %r38;
$L__tmp4:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	shfl.sync.bfly.b32 	%r54, %r53, 16, 31, -1;
$L__tmp5:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r55, %r53, %r54;
$L__tmp6:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	shfl.sync.bfly.b32 	%r56, %r55, 8, 31, -1;
$L__tmp7:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r57, %r55, %r56;
$L__tmp8:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	shfl.sync.bfly.b32 	%r58, %r57, 4, 31, -1;
$L__tmp9:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r59, %r57, %r58;
$L__tmp10:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	shfl.sync.bfly.b32 	%r60, %r59, 2, 31, -1;
$L__tmp11:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r61, %r59, %r60;
$L__tmp12:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	shfl.sync.bfly.b32 	%r62, %r61, 1, 31, -1;
$L__tmp13:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r7, %r61, %r62;
$L__tmp14:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	// begin inline asm
	@%p2 st.shared.b32 [ %r6 + 0 ], %r7;
	// end inline asm
	bar.sync 	0;
	// begin inline asm
	@%p3 ld.shared.b32 %r8, [ %r9 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r63, %r8, 4, 31, -1;
$L__tmp15:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r64, %r8, %r63;
$L__tmp16:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	shfl.sync.bfly.b32 	%r65, %r64, 2, 31, -1;
$L__tmp17:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r66, %r64, %r65;
$L__tmp18:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	shfl.sync.bfly.b32 	%r67, %r66, 1, 31, -1;
$L__tmp19:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r10, %r66, %r67;
$L__tmp20:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	// begin inline asm
	@%p4 st.shared.b32 [ %r9 + 0 ], %r10;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r68, [global_smem];
$L__tmp21:
	.loc	1 90 49                         // sk09_norm_embed.py:90:49
	max.f32 	%r69, %r68, 0f0DA24260;
	mov.b32 	%r70, 0f42FE0000;
	.loc	1 91 22                         // sk09_norm_embed.py:91:22
	div.full.f32 	%r71, %r70, %r69;
	.loc	1 91 14                         // sk09_norm_embed.py:91:14
	mul.f32 	%r72, %r71, %r41;
	mul.f32 	%r73, %r71, %r42;
	mul.f32 	%r74, %r71, %r39;
	mul.f32 	%r75, %r71, %r40;
	mul.f32 	%r76, %r71, %r33;
	mul.f32 	%r77, %r71, %r34;
	mul.f32 	%r78, %r71, %r31;
	mul.f32 	%r79, %r71, %r32;
	.loc	1 92 29                         // sk09_norm_embed.py:92:29
	.loc	1 92 39                         // sk09_norm_embed.py:92:39
	lop3.b32 	%r80, 0x3f000000, %r72, 0x80000000, 0xF8;
	lop3.b32 	%r81, 0x3f000000, %r73, 0x80000000, 0xF8;
	lop3.b32 	%r82, 0x3f000000, %r74, 0x80000000, 0xF8;
	lop3.b32 	%r83, 0x3f000000, %r75, 0x80000000, 0xF8;
	lop3.b32 	%r84, 0x3f000000, %r76, 0x80000000, 0xF8;
	lop3.b32 	%r85, 0x3f000000, %r77, 0x80000000, 0xF8;
	lop3.b32 	%r86, 0x3f000000, %r78, 0x80000000, 0xF8;
	lop3.b32 	%r87, 0x3f000000, %r79, 0x80000000, 0xF8;
	.loc	1 92 14                         // sk09_norm_embed.py:92:14
	fma.rn.f32 	%r88, %r71, %r40, %r83;
	fma.rn.f32 	%r89, %r71, %r39, %r82;
	fma.rn.f32 	%r90, %r71, %r42, %r81;
	fma.rn.f32 	%r91, %r71, %r41, %r80;
	fma.rn.f32 	%r92, %r71, %r32, %r87;
	fma.rn.f32 	%r93, %r71, %r31, %r86;
	fma.rn.f32 	%r94, %r71, %r34, %r85;
	fma.rn.f32 	%r95, %r71, %r33, %r84;
	.loc	1 92 49                         // sk09_norm_embed.py:92:49
	cvt.rzi.s32.f32 	%r96, %r91;
	cvt.rzi.s32.f32 	%r97, %r90;
	cvt.rzi.s32.f32 	%r98, %r89;
	cvt.rzi.s32.f32 	%r99, %r88;
	cvt.rzi.s32.f32 	%r100, %r95;
	cvt.rzi.s32.f32 	%r101, %r94;
	cvt.rzi.s32.f32 	%r102, %r93;
	cvt.rzi.s32.f32 	%r103, %r92;
	.loc	1 93 70                         // sk09_norm_embed.py:93:70
	max.s32 	%r104, %r99, -127;
	max.s32 	%r105, %r98, -127;
	max.s32 	%r106, %r97, -127;
	max.s32 	%r107, %r96, -127;
	max.s32 	%r108, %r103, -127;
	max.s32 	%r109, %r102, -127;
	max.s32 	%r110, %r101, -127;
	max.s32 	%r111, %r100, -127;
	.loc	1 93 77                         // sk09_norm_embed.py:93:77
	min.s32 	%r112, %r107, 127;
	min.s32 	%r113, %r106, 127;
	min.s32 	%r114, %r105, 127;
	min.s32 	%r115, %r104, 127;
	min.s32 	%r116, %r111, 127;
	min.s32 	%r117, %r110, 127;
	min.s32 	%r118, %r109, 127;
	min.s32 	%r119, %r108, 127;
	.loc	1 93 85                         // sk09_norm_embed.py:93:85
	prmt.b32 	%r120, %r115, %r114, 0x3340U;
	prmt.b32 	%r121, %r113, %r112, 0x3340U;
	prmt.b32 	%r11, %r121, %r120, 0x5410U;
	prmt.b32 	%r122, %r119, %r118, 0x3340U;
	prmt.b32 	%r123, %r117, %r116, 0x3340U;
	prmt.b32 	%r12, %r123, %r122, 0x5410U;
	.loc	1 93 45                         // sk09_norm_embed.py:93:45
	// begin inline asm
	@%p1 st.global.v2.b32 [ %rd2 + 0 ], { %r11, %r12 };
	// end inline asm
	.loc	1 94 21                         // sk09_norm_embed.py:94:21
	mad.wide.u32 	%rd3, %r14, 4, %rd6;
	.loc	1 94 34                         // sk09_norm_embed.py:94:34
	mul.f32 	%r13, %r69, 0f3C010204;
	.loc	1 94 26                         // sk09_norm_embed.py:94:26
	or.b32 	%r124, %r18, %r20;
	setp.eq.b32 	%p5, %r124, 0;
	// begin inline asm
	@%p5 st.global.b32 [ %rd3 + 0 ], { %r13 };
	// end inline asm
	.loc	1 94 4                          // sk09_norm_embed.py:94:4
	ret;
$L__tmp22:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk09_norm_embed.py"
	.file	2 "/usr/local/lib/python3.12/dist-packages/triton/language/standard.py"
	.section	.debug_abbrev
	{
.b8 1                                   // Abbreviation Code
.b8 17                                  // DW_TAG_compile_unit
.b8 1                                   // DW_CHILDREN_yes
.b8 37                                  // DW_AT_producer
.b8 8                                   // DW_FORM_string
.b8 19                                  // DW_AT_language
.b8 5                                   // DW_FORM_data2
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 16                                  // DW_AT_stmt_list
.b8 6                                   // DW_FORM_data4
.b8 27                                  // DW_AT_comp_dir
.b8 8                                   // DW_FORM_string
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 2                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 0                                   // DW_CHILDREN_no
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 32                                  // DW_AT_inline
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 3                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 1                                   // DW_CHILDREN_yes
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 4                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 1                                   // DW_CHILDREN_yes
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 11                                  // DW_FORM_data1
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 5                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 0                                   // DW_CHILDREN_no
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 11                                  // DW_FORM_data1
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 161                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x9a DW_TAG_compile_unit
.b8 116                                 // DW_AT_producer
.b8 114
.b8 105
.b8 116
.b8 111
.b8 110
.b8 0
.b8 2                                   // DW_AT_language
.b8 0
.b8 115                                 // DW_AT_name
.b8 107
.b8 48
.b8 57
.b8 95
.b8 110
.b8 111
.b8 114
.b8 109
.b8 95
.b8 101
.b8 109
.b8 98
.b8 101
.b8 100
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 114
.b8 101
.b8 112
.b8 111
.b8 47
.b8 118
.b8 108
.b8 108
.b8 109
.b8 47
.b8 95
.b8 103
.b8 101
.b8 110
.b8 101
.b8 115
.b8 105
.b8 115
.b8 47
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 115
.b8 0
.b8 2                                   // Abbrev [2] 0x48:0x15 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 57
.b8 95
.b8 113
.b8 117
.b8 97
.b8 110
.b8 116
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x5d:0x47 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 72                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x72:0x31 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp21                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 90                                  // DW_AT_call_line
.b8 29                                  // DW_AT_call_column
.b8 5                                   // Abbrev [5] 0x8a:0x18 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp3                           // DW_AT_low_pc
.b64 $L__tmp20                          // DW_AT_high_pc
.b8 2                                   // DW_AT_call_file
.b8 191                                 // DW_AT_call_line
.b8 40                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_QVAR0_0 = _Nativo(
    "sk09_norm_embed/_sk09_quant_kernel/blk2048",
    _QPTX0_0, "_sk09_quant_kernel",
    warps=8, shared=32,
    abi=[0, 1, 2, 3, 4, 5],
    horneado={6: 2048},
    div16=[3, 4, 5],
)
# E1=8 lop3, E2=0 mul

_QPTX0_1 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk09_quant_kernel      // -- Begin function _sk09_quant_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk09_quant_kernel
.visible .entry _sk09_quant_kernel(
	.param .u64 .ptr .global .align 1 _sk09_quant_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk09_quant_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk09_quant_kernel_param_2,
	.param .u32 _sk09_quant_kernel_param_3,
	.param .u32 _sk09_quant_kernel_param_4,
	.param .u32 _sk09_quant_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk09_quant_kernel_param_6,
	.param .u64 .ptr .global .align 1 _sk09_quant_kernel_param_7
)
.reqntid 256
{
	.reg .pred 	%p<23>;
	.reg .b16 	%rs<17>;
	.reg .b32 	%r<207>;
	.reg .b64 	%rd<12>;
	.loc	1 84 0                          // sk09_norm_embed.py:84:0
$L__func_begin0:
	.loc	1 84 0                          // sk09_norm_embed.py:84:0

// %bb.0:
	ld.param.b64 	%rd5, [_sk09_quant_kernel_param_0];
	ld.param.b64 	%rd6, [_sk09_quant_kernel_param_1];
$L__tmp0:
	.loc	1 86 24                         // sk09_norm_embed.py:86:24
	mov.u32 	%r20, %ctaid.x;
	ld.param.b64 	%rd7, [_sk09_quant_kernel_param_2];
	.loc	1 87 24                         // sk09_norm_embed.py:87:24
	mov.u32 	%r21, %tid.x;
	and.b32 	%r22, %r21, 255;
	ld.param.b32 	%r23, [_sk09_quant_kernel_param_3];
	and.b32 	%r24, %r21, 31;
	ld.param.b32 	%r25, [_sk09_quant_kernel_param_4];
	shr.u32 	%r26, %r21, 5;
	ld.param.b32 	%r27, [_sk09_quant_kernel_param_5];
	shl.b32 	%r28, %r21, 4;
	and.b32 	%r29, %r28, 4080;
	.loc	1 88 18                         // sk09_norm_embed.py:88:18
	setp.lt.s32 	%p1, %r29, %r23;
	.loc	1 89 30                         // sk09_norm_embed.py:89:30
	mul.lo.s32 	%r30, %r25, %r20;
	.loc	1 89 24                         // sk09_norm_embed.py:89:24
	mad.wide.s32 	%rd8, %r30, 2, %rd5;
	.loc	1 89 42                         // sk09_norm_embed.py:89:42
	cvt.u64.u32 	%rd9, %r29;
	mad.wide.u32 	%rd1, %r29, 2, %rd8;
	add.s64 	%rd2, %rd1, 16;
	mov.b32 	%r5, 0;
	.loc	1 89 16                         // sk09_norm_embed.py:89:16
	// begin inline asm
	mov.u32 %r1, %r5;
	mov.u32 %r2, %r5;
	mov.u32 %r3, %r5;
	mov.u32 %r4, %r5;
	@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r6, %r5;
	mov.u32 %r7, %r5;
	mov.u32 %r8, %r5;
	mov.u32 %r9, %r5;
	@%p1 ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd2 + 0 ];
	// end inline asm
$L__tmp1:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	setp.eq.b32 	%p2, %r24, 0;
	shr.u32 	%r31, %r21, 3;
	and.b32 	%r32, %r31, 28;
	mov.b32 	%r33, global_smem;
	add.s32 	%r10, %r33, %r32;
	setp.lt.u32 	%p3, %r22, 8;
	shl.b32 	%r34, %r22, 2;
	add.s32 	%r13, %r33, %r34;
	and.b32 	%r35, %r21, 7;
	setp.eq.b32 	%p6, %r35, 0;
	and.pred 	%p4, %p3, %p6;
$L__tmp2:
	.loc	1 93 27                         // sk09_norm_embed.py:93:27
	mul.lo.s32 	%r36, %r27, %r20;
	.loc	1 93 21                         // sk09_norm_embed.py:93:21
	cvt.s64.s32 	%rd10, %r36;
	add.s64 	%rd11, %rd6, %rd10;
	.loc	1 93 39                         // sk09_norm_embed.py:93:39
	add.s64 	%rd3, %rd11, %rd9;
	.loc	1 89 73                         // sk09_norm_embed.py:89:73
	mov.b32 	{%rs1, %rs2}, %r9;
	cvt.f32.bf16 	%r37, %rs2;
	cvt.f32.bf16 	%r38, %rs1;
	mov.b32 	{%rs3, %rs4}, %r8;
	cvt.f32.bf16 	%r39, %rs4;
	cvt.f32.bf16 	%r40, %rs3;
	.loc	1 90 36                         // sk09_norm_embed.py:90:36
	abs.f32 	%r41, %r40;
	abs.f32 	%r42, %r39;
	abs.f32 	%r43, %r38;
	abs.f32 	%r44, %r37;
	.loc	1 89 73                         // sk09_norm_embed.py:89:73
	mov.b32 	{%rs5, %rs6}, %r7;
	cvt.f32.bf16 	%r45, %rs6;
	cvt.f32.bf16 	%r46, %rs5;
	mov.b32 	{%rs7, %rs8}, %r6;
	cvt.f32.bf16 	%r47, %rs8;
	cvt.f32.bf16 	%r48, %rs7;
	.loc	1 90 36                         // sk09_norm_embed.py:90:36
	abs.f32 	%r49, %r48;
	abs.f32 	%r50, %r47;
	abs.f32 	%r51, %r46;
	abs.f32 	%r52, %r45;
	.loc	1 89 73                         // sk09_norm_embed.py:89:73
	mov.b32 	{%rs9, %rs10}, %r4;
	cvt.f32.bf16 	%r53, %rs10;
	cvt.f32.bf16 	%r54, %rs9;
	mov.b32 	{%rs11, %rs12}, %r3;
	cvt.f32.bf16 	%r55, %rs12;
	cvt.f32.bf16 	%r56, %rs11;
	.loc	1 90 36                         // sk09_norm_embed.py:90:36
	abs.f32 	%r57, %r56;
	abs.f32 	%r58, %r55;
	abs.f32 	%r59, %r54;
	abs.f32 	%r60, %r53;
	.loc	1 89 73                         // sk09_norm_embed.py:89:73
	mov.b32 	{%rs13, %rs14}, %r2;
	cvt.f32.bf16 	%r61, %rs14;
	cvt.f32.bf16 	%r62, %rs13;
	mov.b32 	{%rs15, %rs16}, %r1;
	cvt.f32.bf16 	%r63, %rs16;
	cvt.f32.bf16 	%r64, %rs15;
	.loc	1 90 36                         // sk09_norm_embed.py:90:36
	abs.f32 	%r65, %r64;
	abs.f32 	%r66, %r63;
	abs.f32 	%r67, %r62;
	abs.f32 	%r68, %r61;
$L__tmp3:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r69, %r65, %r66;
	max.f32 	%r70, %r69, %r67;
	max.f32 	%r71, %r70, %r68;
	max.f32 	%r72, %r71, %r57;
	max.f32 	%r73, %r72, %r58;
	max.f32 	%r74, %r73, %r59;
	max.f32 	%r75, %r74, %r60;
	max.f32 	%r76, %r75, %r49;
	max.f32 	%r77, %r76, %r50;
	max.f32 	%r78, %r77, %r51;
	max.f32 	%r79, %r78, %r52;
	max.f32 	%r80, %r79, %r41;
	max.f32 	%r81, %r80, %r42;
	max.f32 	%r82, %r81, %r43;
	max.f32 	%r83, %r82, %r44;
$L__tmp4:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	shfl.sync.bfly.b32 	%r84, %r83, 16, 31, -1;
$L__tmp5:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r85, %r83, %r84;
$L__tmp6:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	shfl.sync.bfly.b32 	%r86, %r85, 8, 31, -1;
$L__tmp7:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r87, %r85, %r86;
$L__tmp8:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	shfl.sync.bfly.b32 	%r88, %r87, 4, 31, -1;
$L__tmp9:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r89, %r87, %r88;
$L__tmp10:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	shfl.sync.bfly.b32 	%r90, %r89, 2, 31, -1;
$L__tmp11:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r91, %r89, %r90;
$L__tmp12:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	shfl.sync.bfly.b32 	%r92, %r91, 1, 31, -1;
$L__tmp13:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r11, %r91, %r92;
$L__tmp14:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	// begin inline asm
	@%p2 st.shared.b32 [ %r10 + 0 ], %r11;
	// end inline asm
	bar.sync 	0;
	// begin inline asm
	@%p3 ld.shared.b32 %r12, [ %r13 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r93, %r12, 4, 31, -1;
$L__tmp15:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r94, %r12, %r93;
$L__tmp16:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	shfl.sync.bfly.b32 	%r95, %r94, 2, 31, -1;
$L__tmp17:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r96, %r94, %r95;
$L__tmp18:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	shfl.sync.bfly.b32 	%r97, %r96, 1, 31, -1;
$L__tmp19:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r14, %r96, %r97;
$L__tmp20:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	// begin inline asm
	@%p4 st.shared.b32 [ %r13 + 0 ], %r14;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r98, [global_smem];
$L__tmp21:
	.loc	1 90 49                         // sk09_norm_embed.py:90:49
	max.f32 	%r99, %r98, 0f0DA24260;
	mov.b32 	%r100, 0f42FE0000;
	.loc	1 91 22                         // sk09_norm_embed.py:91:22
	div.full.f32 	%r101, %r100, %r99;
	.loc	1 91 14                         // sk09_norm_embed.py:91:14
	mul.f32 	%r102, %r101, %r63;
	mul.f32 	%r103, %r101, %r64;
	mul.f32 	%r104, %r101, %r61;
	mul.f32 	%r105, %r101, %r62;
	mul.f32 	%r106, %r101, %r55;
	mul.f32 	%r107, %r101, %r56;
	mul.f32 	%r108, %r101, %r53;
	mul.f32 	%r109, %r101, %r54;
	mul.f32 	%r110, %r101, %r47;
	mul.f32 	%r111, %r101, %r48;
	mul.f32 	%r112, %r101, %r45;
	mul.f32 	%r113, %r101, %r46;
	mul.f32 	%r114, %r101, %r39;
	mul.f32 	%r115, %r101, %r40;
	mul.f32 	%r116, %r101, %r37;
	mul.f32 	%r117, %r101, %r38;
	.loc	1 92 29                         // sk09_norm_embed.py:92:29
	.loc	1 92 39                         // sk09_norm_embed.py:92:39
	lop3.b32 	%r118, 0x3f000000, %r102, 0x80000000, 0xF8;
	lop3.b32 	%r119, 0x3f000000, %r103, 0x80000000, 0xF8;
	lop3.b32 	%r120, 0x3f000000, %r104, 0x80000000, 0xF8;
	lop3.b32 	%r121, 0x3f000000, %r105, 0x80000000, 0xF8;
	lop3.b32 	%r122, 0x3f000000, %r106, 0x80000000, 0xF8;
	lop3.b32 	%r123, 0x3f000000, %r107, 0x80000000, 0xF8;
	lop3.b32 	%r124, 0x3f000000, %r108, 0x80000000, 0xF8;
	lop3.b32 	%r125, 0x3f000000, %r109, 0x80000000, 0xF8;
	lop3.b32 	%r126, 0x3f000000, %r110, 0x80000000, 0xF8;
	lop3.b32 	%r127, 0x3f000000, %r111, 0x80000000, 0xF8;
	lop3.b32 	%r128, 0x3f000000, %r112, 0x80000000, 0xF8;
	lop3.b32 	%r129, 0x3f000000, %r113, 0x80000000, 0xF8;
	lop3.b32 	%r130, 0x3f000000, %r114, 0x80000000, 0xF8;
	lop3.b32 	%r131, 0x3f000000, %r115, 0x80000000, 0xF8;
	lop3.b32 	%r132, 0x3f000000, %r116, 0x80000000, 0xF8;
	lop3.b32 	%r133, 0x3f000000, %r117, 0x80000000, 0xF8;
	.loc	1 92 14                         // sk09_norm_embed.py:92:14
	fma.rn.f32 	%r134, %r101, %r62, %r121;
	fma.rn.f32 	%r135, %r101, %r61, %r120;
	fma.rn.f32 	%r136, %r101, %r64, %r119;
	fma.rn.f32 	%r137, %r101, %r63, %r118;
	fma.rn.f32 	%r138, %r101, %r54, %r125;
	fma.rn.f32 	%r139, %r101, %r53, %r124;
	fma.rn.f32 	%r140, %r101, %r56, %r123;
	fma.rn.f32 	%r141, %r101, %r55, %r122;
	fma.rn.f32 	%r142, %r101, %r46, %r129;
	fma.rn.f32 	%r143, %r101, %r45, %r128;
	fma.rn.f32 	%r144, %r101, %r48, %r127;
	fma.rn.f32 	%r145, %r101, %r47, %r126;
	fma.rn.f32 	%r146, %r101, %r38, %r133;
	fma.rn.f32 	%r147, %r101, %r37, %r132;
	fma.rn.f32 	%r148, %r101, %r40, %r131;
	fma.rn.f32 	%r149, %r101, %r39, %r130;
	.loc	1 92 49                         // sk09_norm_embed.py:92:49
	cvt.rzi.s32.f32 	%r150, %r137;
	cvt.rzi.s32.f32 	%r151, %r136;
	cvt.rzi.s32.f32 	%r152, %r135;
	cvt.rzi.s32.f32 	%r153, %r134;
	cvt.rzi.s32.f32 	%r154, %r141;
	cvt.rzi.s32.f32 	%r155, %r140;
	cvt.rzi.s32.f32 	%r156, %r139;
	cvt.rzi.s32.f32 	%r157, %r138;
	cvt.rzi.s32.f32 	%r158, %r145;
	cvt.rzi.s32.f32 	%r159, %r144;
	cvt.rzi.s32.f32 	%r160, %r143;
	cvt.rzi.s32.f32 	%r161, %r142;
	cvt.rzi.s32.f32 	%r162, %r149;
	cvt.rzi.s32.f32 	%r163, %r148;
	cvt.rzi.s32.f32 	%r164, %r147;
	cvt.rzi.s32.f32 	%r165, %r146;
	.loc	1 93 70                         // sk09_norm_embed.py:93:70
	max.s32 	%r166, %r153, -127;
	max.s32 	%r167, %r152, -127;
	max.s32 	%r168, %r151, -127;
	max.s32 	%r169, %r150, -127;
	max.s32 	%r170, %r157, -127;
	max.s32 	%r171, %r156, -127;
	max.s32 	%r172, %r155, -127;
	max.s32 	%r173, %r154, -127;
	max.s32 	%r174, %r161, -127;
	max.s32 	%r175, %r160, -127;
	max.s32 	%r176, %r159, -127;
	max.s32 	%r177, %r158, -127;
	max.s32 	%r178, %r165, -127;
	max.s32 	%r179, %r164, -127;
	max.s32 	%r180, %r163, -127;
	max.s32 	%r181, %r162, -127;
	.loc	1 93 77                         // sk09_norm_embed.py:93:77
	min.s32 	%r182, %r169, 127;
	min.s32 	%r183, %r168, 127;
	min.s32 	%r184, %r167, 127;
	min.s32 	%r185, %r166, 127;
	min.s32 	%r186, %r173, 127;
	min.s32 	%r187, %r172, 127;
	min.s32 	%r188, %r171, 127;
	min.s32 	%r189, %r170, 127;
	min.s32 	%r190, %r177, 127;
	min.s32 	%r191, %r176, 127;
	min.s32 	%r192, %r175, 127;
	min.s32 	%r193, %r174, 127;
	min.s32 	%r194, %r181, 127;
	min.s32 	%r195, %r180, 127;
	min.s32 	%r196, %r179, 127;
	min.s32 	%r197, %r178, 127;
	.loc	1 93 85                         // sk09_norm_embed.py:93:85
	prmt.b32 	%r198, %r185, %r184, 0x3340U;
	prmt.b32 	%r199, %r183, %r182, 0x3340U;
	prmt.b32 	%r15, %r199, %r198, 0x5410U;
	prmt.b32 	%r200, %r189, %r188, 0x3340U;
	prmt.b32 	%r201, %r187, %r186, 0x3340U;
	prmt.b32 	%r16, %r201, %r200, 0x5410U;
	prmt.b32 	%r202, %r193, %r192, 0x3340U;
	prmt.b32 	%r203, %r191, %r190, 0x3340U;
	prmt.b32 	%r17, %r203, %r202, 0x5410U;
	prmt.b32 	%r204, %r197, %r196, 0x3340U;
	prmt.b32 	%r205, %r195, %r194, 0x3340U;
	prmt.b32 	%r18, %r205, %r204, 0x5410U;
	.loc	1 93 45                         // sk09_norm_embed.py:93:45
	// begin inline asm
	@%p1 st.global.v4.b32 [ %rd3 + 0 ], { %r15, %r16, %r17, %r18 };
	// end inline asm
	.loc	1 94 21                         // sk09_norm_embed.py:94:21
	mad.wide.u32 	%rd4, %r20, 4, %rd7;
	.loc	1 94 34                         // sk09_norm_embed.py:94:34
	mul.f32 	%r19, %r99, 0f3C010204;
	.loc	1 94 26                         // sk09_norm_embed.py:94:26
	or.b32 	%r206, %r24, %r26;
	setp.eq.b32 	%p5, %r206, 0;
	// begin inline asm
	@%p5 st.global.b32 [ %rd4 + 0 ], { %r19 };
	// end inline asm
	.loc	1 94 4                          // sk09_norm_embed.py:94:4
	ret;
$L__tmp22:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk09_norm_embed.py"
	.file	2 "/usr/local/lib/python3.12/dist-packages/triton/language/standard.py"
	.section	.debug_abbrev
	{
.b8 1                                   // Abbreviation Code
.b8 17                                  // DW_TAG_compile_unit
.b8 1                                   // DW_CHILDREN_yes
.b8 37                                  // DW_AT_producer
.b8 8                                   // DW_FORM_string
.b8 19                                  // DW_AT_language
.b8 5                                   // DW_FORM_data2
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 16                                  // DW_AT_stmt_list
.b8 6                                   // DW_FORM_data4
.b8 27                                  // DW_AT_comp_dir
.b8 8                                   // DW_FORM_string
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 2                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 0                                   // DW_CHILDREN_no
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 32                                  // DW_AT_inline
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 3                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 1                                   // DW_CHILDREN_yes
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 4                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 1                                   // DW_CHILDREN_yes
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 11                                  // DW_FORM_data1
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 5                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 0                                   // DW_CHILDREN_no
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 11                                  // DW_FORM_data1
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 161                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x9a DW_TAG_compile_unit
.b8 116                                 // DW_AT_producer
.b8 114
.b8 105
.b8 116
.b8 111
.b8 110
.b8 0
.b8 2                                   // DW_AT_language
.b8 0
.b8 115                                 // DW_AT_name
.b8 107
.b8 48
.b8 57
.b8 95
.b8 110
.b8 111
.b8 114
.b8 109
.b8 95
.b8 101
.b8 109
.b8 98
.b8 101
.b8 100
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 114
.b8 101
.b8 112
.b8 111
.b8 47
.b8 118
.b8 108
.b8 108
.b8 109
.b8 47
.b8 95
.b8 103
.b8 101
.b8 110
.b8 101
.b8 115
.b8 105
.b8 115
.b8 47
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 115
.b8 0
.b8 2                                   // Abbrev [2] 0x48:0x15 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 57
.b8 95
.b8 113
.b8 117
.b8 97
.b8 110
.b8 116
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x5d:0x47 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 72                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x72:0x31 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp21                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 90                                  // DW_AT_call_line
.b8 29                                  // DW_AT_call_column
.b8 5                                   // Abbrev [5] 0x8a:0x18 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp3                           // DW_AT_low_pc
.b64 $L__tmp20                          // DW_AT_high_pc
.b8 2                                   // DW_AT_call_file
.b8 191                                 // DW_AT_call_line
.b8 40                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_QVAR0_1 = _Nativo(
    "sk09_norm_embed/_sk09_quant_kernel/blk4096",
    _QPTX0_1, "_sk09_quant_kernel",
    warps=8, shared=32,
    abi=[0, 1, 2, 3, 4, 5],
    horneado={6: 4096},
    div16=[3, 4, 5],
)
# E1=16 lop3, E2=0 mul

_QPTX0_2 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk09_quant_kernel      // -- Begin function _sk09_quant_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk09_quant_kernel
.visible .entry _sk09_quant_kernel(
	.param .u64 .ptr .global .align 1 _sk09_quant_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk09_quant_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk09_quant_kernel_param_2,
	.param .u32 _sk09_quant_kernel_param_3,
	.param .u32 _sk09_quant_kernel_param_4,
	.param .u32 _sk09_quant_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk09_quant_kernel_param_6,
	.param .u64 .ptr .global .align 1 _sk09_quant_kernel_param_7
)
.reqntid 256
{
	.reg .pred 	%p<40>;
	.reg .b16 	%rs<33>;
	.reg .b32 	%r<372>;
	.reg .b64 	%rd<15>;
	.loc	1 84 0                          // sk09_norm_embed.py:84:0
$L__func_begin0:
	.loc	1 84 0                          // sk09_norm_embed.py:84:0

// %bb.0:
	ld.param.b64 	%rd8, [_sk09_quant_kernel_param_0];
	ld.param.b64 	%rd9, [_sk09_quant_kernel_param_1];
$L__tmp0:
	.loc	1 86 24                         // sk09_norm_embed.py:86:24
	mov.u32 	%r32, %ctaid.x;
	ld.param.b64 	%rd10, [_sk09_quant_kernel_param_2];
	.loc	1 87 24                         // sk09_norm_embed.py:87:24
	mov.u32 	%r33, %tid.x;
	and.b32 	%r34, %r33, 255;
	ld.param.b32 	%r35, [_sk09_quant_kernel_param_3];
	and.b32 	%r36, %r33, 31;
	ld.param.b32 	%r37, [_sk09_quant_kernel_param_4];
	shr.u32 	%r38, %r33, 5;
	ld.param.b32 	%r39, [_sk09_quant_kernel_param_5];
	shl.b32 	%r40, %r33, 4;
	and.b32 	%r41, %r40, 4080;
	or.b32 	%r42, %r41, 4096;
	.loc	1 88 18                         // sk09_norm_embed.py:88:18
	setp.lt.s32 	%p1, %r41, %r35;
	setp.lt.s32 	%p2, %r42, %r35;
	.loc	1 89 30                         // sk09_norm_embed.py:89:30
	mul.lo.s32 	%r43, %r37, %r32;
	.loc	1 89 24                         // sk09_norm_embed.py:89:24
	mad.wide.s32 	%rd11, %r43, 2, %rd8;
	.loc	1 89 42                         // sk09_norm_embed.py:89:42
	cvt.u64.u32 	%rd12, %r41;
	mad.wide.u32 	%rd1, %r41, 2, %rd11;
	add.s64 	%rd2, %rd1, 16;
	add.s64 	%rd3, %rd1, 8192;
	add.s64 	%rd4, %rd1, 8208;
	mov.b32 	%r5, 0;
	.loc	1 89 16                         // sk09_norm_embed.py:89:16
	// begin inline asm
	mov.u32 %r1, %r5;
	mov.u32 %r2, %r5;
	mov.u32 %r3, %r5;
	mov.u32 %r4, %r5;
	@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r6, %r5;
	mov.u32 %r7, %r5;
	mov.u32 %r8, %r5;
	mov.u32 %r9, %r5;
	@%p1 ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd2 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r10, %r5;
	mov.u32 %r11, %r5;
	mov.u32 %r12, %r5;
	mov.u32 %r13, %r5;
	@%p2 ld.global.v4.b32 { %r10, %r11, %r12, %r13 }, [ %rd3 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r14, %r5;
	mov.u32 %r15, %r5;
	mov.u32 %r16, %r5;
	mov.u32 %r17, %r5;
	@%p2 ld.global.v4.b32 { %r14, %r15, %r16, %r17 }, [ %rd4 + 0 ];
	// end inline asm
$L__tmp1:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	setp.eq.b32 	%p3, %r36, 0;
	shr.u32 	%r44, %r33, 3;
	and.b32 	%r45, %r44, 28;
	mov.b32 	%r46, global_smem;
	add.s32 	%r18, %r46, %r45;
	setp.lt.u32 	%p4, %r34, 8;
	shl.b32 	%r47, %r34, 2;
	add.s32 	%r21, %r46, %r47;
	and.b32 	%r48, %r33, 7;
	setp.eq.b32 	%p7, %r48, 0;
	and.pred 	%p5, %p4, %p7;
$L__tmp2:
	.loc	1 93 27                         // sk09_norm_embed.py:93:27
	mul.lo.s32 	%r49, %r39, %r32;
	.loc	1 93 21                         // sk09_norm_embed.py:93:21
	cvt.s64.s32 	%rd13, %r49;
	add.s64 	%rd14, %rd9, %rd13;
	.loc	1 93 39                         // sk09_norm_embed.py:93:39
	add.s64 	%rd5, %rd14, %rd12;
	add.s64 	%rd6, %rd5, 4096;
	.loc	1 89 73                         // sk09_norm_embed.py:89:73
	mov.b32 	{%rs1, %rs2}, %r9;
	cvt.f32.bf16 	%r50, %rs2;
	cvt.f32.bf16 	%r51, %rs1;
	mov.b32 	{%rs3, %rs4}, %r8;
	cvt.f32.bf16 	%r52, %rs4;
	cvt.f32.bf16 	%r53, %rs3;
	.loc	1 90 36                         // sk09_norm_embed.py:90:36
	abs.f32 	%r54, %r53;
	abs.f32 	%r55, %r52;
	abs.f32 	%r56, %r51;
	abs.f32 	%r57, %r50;
	.loc	1 89 73                         // sk09_norm_embed.py:89:73
	mov.b32 	{%rs5, %rs6}, %r7;
	cvt.f32.bf16 	%r58, %rs6;
	cvt.f32.bf16 	%r59, %rs5;
	mov.b32 	{%rs7, %rs8}, %r6;
	cvt.f32.bf16 	%r60, %rs8;
	cvt.f32.bf16 	%r61, %rs7;
	.loc	1 90 36                         // sk09_norm_embed.py:90:36
	abs.f32 	%r62, %r61;
	abs.f32 	%r63, %r60;
	abs.f32 	%r64, %r59;
	abs.f32 	%r65, %r58;
	.loc	1 89 73                         // sk09_norm_embed.py:89:73
	mov.b32 	{%rs9, %rs10}, %r4;
	cvt.f32.bf16 	%r66, %rs10;
	cvt.f32.bf16 	%r67, %rs9;
	mov.b32 	{%rs11, %rs12}, %r3;
	cvt.f32.bf16 	%r68, %rs12;
	cvt.f32.bf16 	%r69, %rs11;
	.loc	1 90 36                         // sk09_norm_embed.py:90:36
	abs.f32 	%r70, %r69;
	abs.f32 	%r71, %r68;
	abs.f32 	%r72, %r67;
	abs.f32 	%r73, %r66;
	.loc	1 89 73                         // sk09_norm_embed.py:89:73
	mov.b32 	{%rs13, %rs14}, %r2;
	cvt.f32.bf16 	%r74, %rs14;
	cvt.f32.bf16 	%r75, %rs13;
	mov.b32 	{%rs15, %rs16}, %r1;
	cvt.f32.bf16 	%r76, %rs16;
	cvt.f32.bf16 	%r77, %rs15;
	.loc	1 90 36                         // sk09_norm_embed.py:90:36
	abs.f32 	%r78, %r77;
	abs.f32 	%r79, %r76;
	abs.f32 	%r80, %r75;
	abs.f32 	%r81, %r74;
$L__tmp3:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r82, %r78, %r79;
	max.f32 	%r83, %r82, %r80;
	max.f32 	%r84, %r83, %r81;
	max.f32 	%r85, %r84, %r70;
	max.f32 	%r86, %r85, %r71;
	max.f32 	%r87, %r86, %r72;
	max.f32 	%r88, %r87, %r73;
	max.f32 	%r89, %r88, %r62;
	max.f32 	%r90, %r89, %r63;
	max.f32 	%r91, %r90, %r64;
	max.f32 	%r92, %r91, %r65;
	max.f32 	%r93, %r92, %r54;
	max.f32 	%r94, %r93, %r55;
	max.f32 	%r95, %r94, %r56;
	max.f32 	%r96, %r95, %r57;
$L__tmp4:
	.loc	1 89 73                         // sk09_norm_embed.py:89:73
	mov.b32 	{%rs17, %rs18}, %r17;
	cvt.f32.bf16 	%r97, %rs18;
	cvt.f32.bf16 	%r98, %rs17;
	mov.b32 	{%rs19, %rs20}, %r16;
	cvt.f32.bf16 	%r99, %rs20;
	cvt.f32.bf16 	%r100, %rs19;
	.loc	1 90 36                         // sk09_norm_embed.py:90:36
	abs.f32 	%r101, %r100;
	abs.f32 	%r102, %r99;
	abs.f32 	%r103, %r98;
	abs.f32 	%r104, %r97;
	.loc	1 89 73                         // sk09_norm_embed.py:89:73
	mov.b32 	{%rs21, %rs22}, %r15;
	cvt.f32.bf16 	%r105, %rs22;
	cvt.f32.bf16 	%r106, %rs21;
	mov.b32 	{%rs23, %rs24}, %r14;
	cvt.f32.bf16 	%r107, %rs24;
	cvt.f32.bf16 	%r108, %rs23;
	.loc	1 90 36                         // sk09_norm_embed.py:90:36
	abs.f32 	%r109, %r108;
	abs.f32 	%r110, %r107;
	abs.f32 	%r111, %r106;
	abs.f32 	%r112, %r105;
	.loc	1 89 73                         // sk09_norm_embed.py:89:73
	mov.b32 	{%rs25, %rs26}, %r13;
	cvt.f32.bf16 	%r113, %rs26;
	cvt.f32.bf16 	%r114, %rs25;
	mov.b32 	{%rs27, %rs28}, %r12;
	cvt.f32.bf16 	%r115, %rs28;
	cvt.f32.bf16 	%r116, %rs27;
	.loc	1 90 36                         // sk09_norm_embed.py:90:36
	abs.f32 	%r117, %r116;
	abs.f32 	%r118, %r115;
	abs.f32 	%r119, %r114;
	abs.f32 	%r120, %r113;
	.loc	1 89 73                         // sk09_norm_embed.py:89:73
	mov.b32 	{%rs29, %rs30}, %r11;
	cvt.f32.bf16 	%r121, %rs30;
	cvt.f32.bf16 	%r122, %rs29;
	mov.b32 	{%rs31, %rs32}, %r10;
	cvt.f32.bf16 	%r123, %rs32;
	cvt.f32.bf16 	%r124, %rs31;
	.loc	1 90 36                         // sk09_norm_embed.py:90:36
	abs.f32 	%r125, %r124;
	abs.f32 	%r126, %r123;
	abs.f32 	%r127, %r122;
	abs.f32 	%r128, %r121;
$L__tmp5:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r129, %r96, %r125;
	max.f32 	%r130, %r129, %r126;
	max.f32 	%r131, %r130, %r127;
	max.f32 	%r132, %r131, %r128;
	max.f32 	%r133, %r132, %r117;
	max.f32 	%r134, %r133, %r118;
	max.f32 	%r135, %r134, %r119;
	max.f32 	%r136, %r135, %r120;
	max.f32 	%r137, %r136, %r109;
	max.f32 	%r138, %r137, %r110;
	max.f32 	%r139, %r138, %r111;
	max.f32 	%r140, %r139, %r112;
	max.f32 	%r141, %r140, %r101;
	max.f32 	%r142, %r141, %r102;
	max.f32 	%r143, %r142, %r103;
	max.f32 	%r144, %r143, %r104;
$L__tmp6:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	shfl.sync.bfly.b32 	%r145, %r144, 16, 31, -1;
$L__tmp7:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r146, %r144, %r145;
$L__tmp8:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	shfl.sync.bfly.b32 	%r147, %r146, 8, 31, -1;
$L__tmp9:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r148, %r146, %r147;
$L__tmp10:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	shfl.sync.bfly.b32 	%r149, %r148, 4, 31, -1;
$L__tmp11:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r150, %r148, %r149;
$L__tmp12:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	shfl.sync.bfly.b32 	%r151, %r150, 2, 31, -1;
$L__tmp13:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r152, %r150, %r151;
$L__tmp14:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	shfl.sync.bfly.b32 	%r153, %r152, 1, 31, -1;
$L__tmp15:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r19, %r152, %r153;
$L__tmp16:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	// begin inline asm
	@%p3 st.shared.b32 [ %r18 + 0 ], %r19;
	// end inline asm
	bar.sync 	0;
	// begin inline asm
	@%p4 ld.shared.b32 %r20, [ %r21 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r154, %r20, 4, 31, -1;
$L__tmp17:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r155, %r20, %r154;
$L__tmp18:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	shfl.sync.bfly.b32 	%r156, %r155, 2, 31, -1;
$L__tmp19:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r157, %r155, %r156;
$L__tmp20:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	shfl.sync.bfly.b32 	%r158, %r157, 1, 31, -1;
$L__tmp21:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r22, %r157, %r158;
$L__tmp22:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	// begin inline asm
	@%p5 st.shared.b32 [ %r21 + 0 ], %r22;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r159, [global_smem];
$L__tmp23:
	.loc	1 90 49                         // sk09_norm_embed.py:90:49
	max.f32 	%r160, %r159, 0f0DA24260;
	mov.b32 	%r161, 0f42FE0000;
	.loc	1 91 22                         // sk09_norm_embed.py:91:22
	div.full.f32 	%r162, %r161, %r160;
	.loc	1 91 14                         // sk09_norm_embed.py:91:14
	mul.f32 	%r163, %r162, %r76;
	mul.f32 	%r164, %r162, %r77;
	mul.f32 	%r165, %r162, %r74;
	mul.f32 	%r166, %r162, %r75;
	mul.f32 	%r167, %r162, %r68;
	mul.f32 	%r168, %r162, %r69;
	mul.f32 	%r169, %r162, %r66;
	mul.f32 	%r170, %r162, %r67;
	mul.f32 	%r171, %r162, %r60;
	mul.f32 	%r172, %r162, %r61;
	mul.f32 	%r173, %r162, %r58;
	mul.f32 	%r174, %r162, %r59;
	mul.f32 	%r175, %r162, %r52;
	mul.f32 	%r176, %r162, %r53;
	mul.f32 	%r177, %r162, %r50;
	mul.f32 	%r178, %r162, %r51;
	mul.f32 	%r179, %r162, %r123;
	mul.f32 	%r180, %r162, %r124;
	mul.f32 	%r181, %r162, %r121;
	mul.f32 	%r182, %r162, %r122;
	mul.f32 	%r183, %r162, %r115;
	mul.f32 	%r184, %r162, %r116;
	mul.f32 	%r185, %r162, %r113;
	mul.f32 	%r186, %r162, %r114;
	mul.f32 	%r187, %r162, %r107;
	mul.f32 	%r188, %r162, %r108;
	mul.f32 	%r189, %r162, %r105;
	mul.f32 	%r190, %r162, %r106;
	mul.f32 	%r191, %r162, %r99;
	mul.f32 	%r192, %r162, %r100;
	mul.f32 	%r193, %r162, %r97;
	mul.f32 	%r194, %r162, %r98;
	.loc	1 92 29                         // sk09_norm_embed.py:92:29
	.loc	1 92 39                         // sk09_norm_embed.py:92:39
	lop3.b32 	%r195, 0x3f000000, %r163, 0x80000000, 0xF8;
	lop3.b32 	%r196, 0x3f000000, %r164, 0x80000000, 0xF8;
	lop3.b32 	%r197, 0x3f000000, %r165, 0x80000000, 0xF8;
	lop3.b32 	%r198, 0x3f000000, %r166, 0x80000000, 0xF8;
	lop3.b32 	%r199, 0x3f000000, %r167, 0x80000000, 0xF8;
	lop3.b32 	%r200, 0x3f000000, %r168, 0x80000000, 0xF8;
	lop3.b32 	%r201, 0x3f000000, %r169, 0x80000000, 0xF8;
	lop3.b32 	%r202, 0x3f000000, %r170, 0x80000000, 0xF8;
	lop3.b32 	%r203, 0x3f000000, %r171, 0x80000000, 0xF8;
	lop3.b32 	%r204, 0x3f000000, %r172, 0x80000000, 0xF8;
	lop3.b32 	%r205, 0x3f000000, %r173, 0x80000000, 0xF8;
	lop3.b32 	%r206, 0x3f000000, %r174, 0x80000000, 0xF8;
	lop3.b32 	%r207, 0x3f000000, %r175, 0x80000000, 0xF8;
	lop3.b32 	%r208, 0x3f000000, %r176, 0x80000000, 0xF8;
	lop3.b32 	%r209, 0x3f000000, %r177, 0x80000000, 0xF8;
	lop3.b32 	%r210, 0x3f000000, %r178, 0x80000000, 0xF8;
	lop3.b32 	%r211, 0x3f000000, %r179, 0x80000000, 0xF8;
	lop3.b32 	%r212, 0x3f000000, %r180, 0x80000000, 0xF8;
	lop3.b32 	%r213, 0x3f000000, %r181, 0x80000000, 0xF8;
	lop3.b32 	%r214, 0x3f000000, %r182, 0x80000000, 0xF8;
	lop3.b32 	%r215, 0x3f000000, %r183, 0x80000000, 0xF8;
	lop3.b32 	%r216, 0x3f000000, %r184, 0x80000000, 0xF8;
	lop3.b32 	%r217, 0x3f000000, %r185, 0x80000000, 0xF8;
	lop3.b32 	%r218, 0x3f000000, %r186, 0x80000000, 0xF8;
	lop3.b32 	%r219, 0x3f000000, %r187, 0x80000000, 0xF8;
	lop3.b32 	%r220, 0x3f000000, %r188, 0x80000000, 0xF8;
	lop3.b32 	%r221, 0x3f000000, %r189, 0x80000000, 0xF8;
	lop3.b32 	%r222, 0x3f000000, %r190, 0x80000000, 0xF8;
	lop3.b32 	%r223, 0x3f000000, %r191, 0x80000000, 0xF8;
	lop3.b32 	%r224, 0x3f000000, %r192, 0x80000000, 0xF8;
	lop3.b32 	%r225, 0x3f000000, %r193, 0x80000000, 0xF8;
	lop3.b32 	%r226, 0x3f000000, %r194, 0x80000000, 0xF8;
	.loc	1 92 14                         // sk09_norm_embed.py:92:14
	fma.rn.f32 	%r227, %r162, %r75, %r198;
	fma.rn.f32 	%r228, %r162, %r74, %r197;
	fma.rn.f32 	%r229, %r162, %r77, %r196;
	fma.rn.f32 	%r230, %r162, %r76, %r195;
	fma.rn.f32 	%r231, %r162, %r67, %r202;
	fma.rn.f32 	%r232, %r162, %r66, %r201;
	fma.rn.f32 	%r233, %r162, %r69, %r200;
	fma.rn.f32 	%r234, %r162, %r68, %r199;
	fma.rn.f32 	%r235, %r162, %r59, %r206;
	fma.rn.f32 	%r236, %r162, %r58, %r205;
	fma.rn.f32 	%r237, %r162, %r61, %r204;
	fma.rn.f32 	%r238, %r162, %r60, %r203;
	fma.rn.f32 	%r239, %r162, %r51, %r210;
	fma.rn.f32 	%r240, %r162, %r50, %r209;
	fma.rn.f32 	%r241, %r162, %r53, %r208;
	fma.rn.f32 	%r242, %r162, %r52, %r207;
	fma.rn.f32 	%r243, %r162, %r122, %r214;
	fma.rn.f32 	%r244, %r162, %r121, %r213;
	fma.rn.f32 	%r245, %r162, %r124, %r212;
	fma.rn.f32 	%r246, %r162, %r123, %r211;
	fma.rn.f32 	%r247, %r162, %r114, %r218;
	fma.rn.f32 	%r248, %r162, %r113, %r217;
	fma.rn.f32 	%r249, %r162, %r116, %r216;
	fma.rn.f32 	%r250, %r162, %r115, %r215;
	fma.rn.f32 	%r251, %r162, %r106, %r222;
	fma.rn.f32 	%r252, %r162, %r105, %r221;
	fma.rn.f32 	%r253, %r162, %r108, %r220;
	fma.rn.f32 	%r254, %r162, %r107, %r219;
	fma.rn.f32 	%r255, %r162, %r98, %r226;
	fma.rn.f32 	%r256, %r162, %r97, %r225;
	fma.rn.f32 	%r257, %r162, %r100, %r224;
	fma.rn.f32 	%r258, %r162, %r99, %r223;
	.loc	1 92 49                         // sk09_norm_embed.py:92:49
	cvt.rzi.s32.f32 	%r259, %r230;
	cvt.rzi.s32.f32 	%r260, %r229;
	cvt.rzi.s32.f32 	%r261, %r228;
	cvt.rzi.s32.f32 	%r262, %r227;
	cvt.rzi.s32.f32 	%r263, %r234;
	cvt.rzi.s32.f32 	%r264, %r233;
	cvt.rzi.s32.f32 	%r265, %r232;
	cvt.rzi.s32.f32 	%r266, %r231;
	cvt.rzi.s32.f32 	%r267, %r238;
	cvt.rzi.s32.f32 	%r268, %r237;
	cvt.rzi.s32.f32 	%r269, %r236;
	cvt.rzi.s32.f32 	%r270, %r235;
	cvt.rzi.s32.f32 	%r271, %r242;
	cvt.rzi.s32.f32 	%r272, %r241;
	cvt.rzi.s32.f32 	%r273, %r240;
	cvt.rzi.s32.f32 	%r274, %r239;
	cvt.rzi.s32.f32 	%r275, %r246;
	cvt.rzi.s32.f32 	%r276, %r245;
	cvt.rzi.s32.f32 	%r277, %r244;
	cvt.rzi.s32.f32 	%r278, %r243;
	cvt.rzi.s32.f32 	%r279, %r250;
	cvt.rzi.s32.f32 	%r280, %r249;
	cvt.rzi.s32.f32 	%r281, %r248;
	cvt.rzi.s32.f32 	%r282, %r247;
	cvt.rzi.s32.f32 	%r283, %r254;
	cvt.rzi.s32.f32 	%r284, %r253;
	cvt.rzi.s32.f32 	%r285, %r252;
	cvt.rzi.s32.f32 	%r286, %r251;
	cvt.rzi.s32.f32 	%r287, %r258;
	cvt.rzi.s32.f32 	%r288, %r257;
	cvt.rzi.s32.f32 	%r289, %r256;
	cvt.rzi.s32.f32 	%r290, %r255;
	.loc	1 93 70                         // sk09_norm_embed.py:93:70
	max.s32 	%r291, %r262, -127;
	max.s32 	%r292, %r261, -127;
	max.s32 	%r293, %r260, -127;
	max.s32 	%r294, %r259, -127;
	max.s32 	%r295, %r266, -127;
	max.s32 	%r296, %r265, -127;
	max.s32 	%r297, %r264, -127;
	max.s32 	%r298, %r263, -127;
	max.s32 	%r299, %r270, -127;
	max.s32 	%r300, %r269, -127;
	max.s32 	%r301, %r268, -127;
	max.s32 	%r302, %r267, -127;
	max.s32 	%r303, %r274, -127;
	max.s32 	%r304, %r273, -127;
	max.s32 	%r305, %r272, -127;
	max.s32 	%r306, %r271, -127;
	max.s32 	%r307, %r278, -127;
	max.s32 	%r308, %r277, -127;
	max.s32 	%r309, %r276, -127;
	max.s32 	%r310, %r275, -127;
	max.s32 	%r311, %r282, -127;
	max.s32 	%r312, %r281, -127;
	max.s32 	%r313, %r280, -127;
	max.s32 	%r314, %r279, -127;
	max.s32 	%r315, %r286, -127;
	max.s32 	%r316, %r285, -127;
	max.s32 	%r317, %r284, -127;
	max.s32 	%r318, %r283, -127;
	max.s32 	%r319, %r290, -127;
	max.s32 	%r320, %r289, -127;
	max.s32 	%r321, %r288, -127;
	max.s32 	%r322, %r287, -127;
	.loc	1 93 77                         // sk09_norm_embed.py:93:77
	min.s32 	%r323, %r294, 127;
	min.s32 	%r324, %r293, 127;
	min.s32 	%r325, %r292, 127;
	min.s32 	%r326, %r291, 127;
	min.s32 	%r327, %r298, 127;
	min.s32 	%r328, %r297, 127;
	min.s32 	%r329, %r296, 127;
	min.s32 	%r330, %r295, 127;
	min.s32 	%r331, %r302, 127;
	min.s32 	%r332, %r301, 127;
	min.s32 	%r333, %r300, 127;
	min.s32 	%r334, %r299, 127;
	min.s32 	%r335, %r306, 127;
	min.s32 	%r336, %r305, 127;
	min.s32 	%r337, %r304, 127;
	min.s32 	%r338, %r303, 127;
	min.s32 	%r339, %r310, 127;
	min.s32 	%r340, %r309, 127;
	min.s32 	%r341, %r308, 127;
	min.s32 	%r342, %r307, 127;
	min.s32 	%r343, %r314, 127;
	min.s32 	%r344, %r313, 127;
	min.s32 	%r345, %r312, 127;
	min.s32 	%r346, %r311, 127;
	min.s32 	%r347, %r318, 127;
	min.s32 	%r348, %r317, 127;
	min.s32 	%r349, %r316, 127;
	min.s32 	%r350, %r315, 127;
	min.s32 	%r351, %r322, 127;
	min.s32 	%r352, %r321, 127;
	min.s32 	%r353, %r320, 127;
	min.s32 	%r354, %r319, 127;
	.loc	1 93 85                         // sk09_norm_embed.py:93:85
	prmt.b32 	%r355, %r326, %r325, 0x3340U;
	prmt.b32 	%r356, %r324, %r323, 0x3340U;
	prmt.b32 	%r23, %r356, %r355, 0x5410U;
	prmt.b32 	%r357, %r330, %r329, 0x3340U;
	prmt.b32 	%r358, %r328, %r327, 0x3340U;
	prmt.b32 	%r24, %r358, %r357, 0x5410U;
	prmt.b32 	%r359, %r334, %r333, 0x3340U;
	prmt.b32 	%r360, %r332, %r331, 0x3340U;
	prmt.b32 	%r25, %r360, %r359, 0x5410U;
	prmt.b32 	%r361, %r338, %r337, 0x3340U;
	prmt.b32 	%r362, %r336, %r335, 0x3340U;
	prmt.b32 	%r26, %r362, %r361, 0x5410U;
	prmt.b32 	%r363, %r342, %r341, 0x3340U;
	prmt.b32 	%r364, %r340, %r339, 0x3340U;
	prmt.b32 	%r27, %r364, %r363, 0x5410U;
	prmt.b32 	%r365, %r346, %r345, 0x3340U;
	prmt.b32 	%r366, %r344, %r343, 0x3340U;
	prmt.b32 	%r28, %r366, %r365, 0x5410U;
	prmt.b32 	%r367, %r350, %r349, 0x3340U;
	prmt.b32 	%r368, %r348, %r347, 0x3340U;
	prmt.b32 	%r29, %r368, %r367, 0x5410U;
	prmt.b32 	%r369, %r354, %r353, 0x3340U;
	prmt.b32 	%r370, %r352, %r351, 0x3340U;
	prmt.b32 	%r30, %r370, %r369, 0x5410U;
	.loc	1 93 45                         // sk09_norm_embed.py:93:45
	// begin inline asm
	@%p1 st.global.v4.b32 [ %rd5 + 0 ], { %r23, %r24, %r25, %r26 };
	// end inline asm
	// begin inline asm
	@%p2 st.global.v4.b32 [ %rd6 + 0 ], { %r27, %r28, %r29, %r30 };
	// end inline asm
	.loc	1 94 21                         // sk09_norm_embed.py:94:21
	mad.wide.u32 	%rd7, %r32, 4, %rd10;
	.loc	1 94 34                         // sk09_norm_embed.py:94:34
	mul.f32 	%r31, %r160, 0f3C010204;
	.loc	1 94 26                         // sk09_norm_embed.py:94:26
	or.b32 	%r371, %r36, %r38;
	setp.eq.b32 	%p6, %r371, 0;
	// begin inline asm
	@%p6 st.global.b32 [ %rd7 + 0 ], { %r31 };
	// end inline asm
	.loc	1 94 4                          // sk09_norm_embed.py:94:4
	ret;
$L__tmp24:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk09_norm_embed.py"
	.file	2 "/usr/local/lib/python3.12/dist-packages/triton/language/standard.py"
	.section	.debug_abbrev
	{
.b8 1                                   // Abbreviation Code
.b8 17                                  // DW_TAG_compile_unit
.b8 1                                   // DW_CHILDREN_yes
.b8 37                                  // DW_AT_producer
.b8 8                                   // DW_FORM_string
.b8 19                                  // DW_AT_language
.b8 5                                   // DW_FORM_data2
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 16                                  // DW_AT_stmt_list
.b8 6                                   // DW_FORM_data4
.b8 27                                  // DW_AT_comp_dir
.b8 8                                   // DW_FORM_string
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 2                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 0                                   // DW_CHILDREN_no
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 32                                  // DW_AT_inline
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 3                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 1                                   // DW_CHILDREN_yes
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 4                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 1                                   // DW_CHILDREN_yes
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 11                                  // DW_FORM_data1
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 5                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 0                                   // DW_CHILDREN_no
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 11                                  // DW_FORM_data1
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 161                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x9a DW_TAG_compile_unit
.b8 116                                 // DW_AT_producer
.b8 114
.b8 105
.b8 116
.b8 111
.b8 110
.b8 0
.b8 2                                   // DW_AT_language
.b8 0
.b8 115                                 // DW_AT_name
.b8 107
.b8 48
.b8 57
.b8 95
.b8 110
.b8 111
.b8 114
.b8 109
.b8 95
.b8 101
.b8 109
.b8 98
.b8 101
.b8 100
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 114
.b8 101
.b8 112
.b8 111
.b8 47
.b8 118
.b8 108
.b8 108
.b8 109
.b8 47
.b8 95
.b8 103
.b8 101
.b8 110
.b8 101
.b8 115
.b8 105
.b8 115
.b8 47
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 115
.b8 0
.b8 2                                   // Abbrev [2] 0x48:0x15 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 57
.b8 95
.b8 113
.b8 117
.b8 97
.b8 110
.b8 116
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x5d:0x47 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 72                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x72:0x31 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp23                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 90                                  // DW_AT_call_line
.b8 29                                  // DW_AT_call_column
.b8 5                                   // Abbrev [5] 0x8a:0x18 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp3                           // DW_AT_low_pc
.b64 $L__tmp22                          // DW_AT_high_pc
.b8 2                                   // DW_AT_call_file
.b8 191                                 // DW_AT_call_line
.b8 40                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_QVAR0_2 = _Nativo(
    "sk09_norm_embed/_sk09_quant_kernel/blk8192",
    _QPTX0_2, "_sk09_quant_kernel",
    warps=8, shared=32,
    abi=[0, 1, 2, 3, 4, 5],
    horneado={6: 8192},
    div16=[3, 4, 5],
)
# E1=32 lop3, E2=0 mul

_QPTX0_3 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk09_quant_kernel      // -- Begin function _sk09_quant_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk09_quant_kernel
.visible .entry _sk09_quant_kernel(
	.param .u64 .ptr .global .align 1 _sk09_quant_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk09_quant_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk09_quant_kernel_param_2,
	.param .u32 _sk09_quant_kernel_param_3,
	.param .u32 _sk09_quant_kernel_param_4,
	.param .u32 _sk09_quant_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk09_quant_kernel_param_6,
	.param .u64 .ptr .global .align 1 _sk09_quant_kernel_param_7
)
.reqntid 256
{
	.reg .pred 	%p<74>;
	.reg .b16 	%rs<65>;
	.reg .b32 	%r<703>;
	.reg .b64 	%rd<22>;
	.loc	1 84 0                          // sk09_norm_embed.py:84:0
$L__func_begin0:
	.loc	1 84 0                          // sk09_norm_embed.py:84:0

// %bb.0:
	ld.param.b64 	%rd14, [_sk09_quant_kernel_param_0];
	ld.param.b64 	%rd15, [_sk09_quant_kernel_param_1];
$L__tmp0:
	.loc	1 86 24                         // sk09_norm_embed.py:86:24
	mov.u32 	%r56, %ctaid.x;
	ld.param.b64 	%rd16, [_sk09_quant_kernel_param_2];
	.loc	1 87 24                         // sk09_norm_embed.py:87:24
	mov.u32 	%r57, %tid.x;
	and.b32 	%r58, %r57, 255;
	ld.param.b32 	%r59, [_sk09_quant_kernel_param_3];
	and.b32 	%r60, %r57, 31;
	ld.param.b32 	%r61, [_sk09_quant_kernel_param_4];
	shr.u32 	%r62, %r57, 5;
	ld.param.b32 	%r63, [_sk09_quant_kernel_param_5];
	shl.b32 	%r64, %r57, 4;
	and.b32 	%r65, %r64, 4080;
	or.b32 	%r66, %r65, 4096;
	or.b32 	%r67, %r65, 8192;
	or.b32 	%r68, %r64, 12288;
	or.b32 	%r69, %r64, 12296;
	.loc	1 88 18                         // sk09_norm_embed.py:88:18
	setp.lt.s32 	%p1, %r65, %r59;
	setp.lt.s32 	%p2, %r66, %r59;
	setp.lt.s32 	%p3, %r67, %r59;
	setp.lt.s32 	%p4, %r68, %r59;
	.loc	1 89 30                         // sk09_norm_embed.py:89:30
	mul.lo.s32 	%r70, %r61, %r56;
	.loc	1 89 24                         // sk09_norm_embed.py:89:24
	mad.wide.s32 	%rd17, %r70, 2, %rd14;
	.loc	1 89 42                         // sk09_norm_embed.py:89:42
	cvt.u64.u32 	%rd18, %r65;
	mad.wide.u32 	%rd1, %r65, 2, %rd17;
	add.s64 	%rd2, %rd1, 16;
	add.s64 	%rd3, %rd1, 8192;
	add.s64 	%rd4, %rd1, 8208;
	add.s64 	%rd5, %rd1, 16384;
	add.s64 	%rd6, %rd1, 16400;
	cvt.u64.u32 	%rd19, %r68;
	mad.wide.u32 	%rd7, %r68, 2, %rd17;
	mad.wide.u32 	%rd8, %r69, 2, %rd17;
	mov.b32 	%r5, 0;
	.loc	1 89 16                         // sk09_norm_embed.py:89:16
	// begin inline asm
	mov.u32 %r1, %r5;
	mov.u32 %r2, %r5;
	mov.u32 %r3, %r5;
	mov.u32 %r4, %r5;
	@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r6, %r5;
	mov.u32 %r7, %r5;
	mov.u32 %r8, %r5;
	mov.u32 %r9, %r5;
	@%p1 ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd2 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r10, %r5;
	mov.u32 %r11, %r5;
	mov.u32 %r12, %r5;
	mov.u32 %r13, %r5;
	@%p2 ld.global.v4.b32 { %r10, %r11, %r12, %r13 }, [ %rd3 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r14, %r5;
	mov.u32 %r15, %r5;
	mov.u32 %r16, %r5;
	mov.u32 %r17, %r5;
	@%p2 ld.global.v4.b32 { %r14, %r15, %r16, %r17 }, [ %rd4 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r18, %r5;
	mov.u32 %r19, %r5;
	mov.u32 %r20, %r5;
	mov.u32 %r21, %r5;
	@%p3 ld.global.v4.b32 { %r18, %r19, %r20, %r21 }, [ %rd5 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r22, %r5;
	mov.u32 %r23, %r5;
	mov.u32 %r24, %r5;
	mov.u32 %r25, %r5;
	@%p3 ld.global.v4.b32 { %r22, %r23, %r24, %r25 }, [ %rd6 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r26, %r5;
	mov.u32 %r27, %r5;
	mov.u32 %r28, %r5;
	mov.u32 %r29, %r5;
	@%p4 ld.global.v4.b32 { %r26, %r27, %r28, %r29 }, [ %rd7 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r30, %r5;
	mov.u32 %r31, %r5;
	mov.u32 %r32, %r5;
	mov.u32 %r33, %r5;
	@%p4 ld.global.v4.b32 { %r30, %r31, %r32, %r33 }, [ %rd8 + 0 ];
	// end inline asm
$L__tmp1:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	setp.eq.b32 	%p5, %r60, 0;
	shr.u32 	%r71, %r57, 3;
	and.b32 	%r72, %r71, 28;
	mov.b32 	%r73, global_smem;
	add.s32 	%r34, %r73, %r72;
	setp.lt.u32 	%p6, %r58, 8;
	shl.b32 	%r74, %r58, 2;
	add.s32 	%r37, %r73, %r74;
	and.b32 	%r75, %r57, 7;
	setp.eq.b32 	%p9, %r75, 0;
	and.pred 	%p7, %p6, %p9;
$L__tmp2:
	.loc	1 93 27                         // sk09_norm_embed.py:93:27
	mul.lo.s32 	%r76, %r63, %r56;
	.loc	1 93 21                         // sk09_norm_embed.py:93:21
	cvt.s64.s32 	%rd20, %r76;
	add.s64 	%rd21, %rd15, %rd20;
	.loc	1 93 39                         // sk09_norm_embed.py:93:39
	add.s64 	%rd9, %rd21, %rd18;
	add.s64 	%rd10, %rd9, 4096;
	add.s64 	%rd11, %rd9, 8192;
	add.s64 	%rd12, %rd21, %rd19;
	.loc	1 89 73                         // sk09_norm_embed.py:89:73
	mov.b32 	{%rs1, %rs2}, %r9;
	cvt.f32.bf16 	%r77, %rs2;
	cvt.f32.bf16 	%r78, %rs1;
	mov.b32 	{%rs3, %rs4}, %r8;
	cvt.f32.bf16 	%r79, %rs4;
	cvt.f32.bf16 	%r80, %rs3;
	.loc	1 90 36                         // sk09_norm_embed.py:90:36
	abs.f32 	%r81, %r80;
	abs.f32 	%r82, %r79;
	abs.f32 	%r83, %r78;
	abs.f32 	%r84, %r77;
	.loc	1 89 73                         // sk09_norm_embed.py:89:73
	mov.b32 	{%rs5, %rs6}, %r7;
	cvt.f32.bf16 	%r85, %rs6;
	cvt.f32.bf16 	%r86, %rs5;
	mov.b32 	{%rs7, %rs8}, %r6;
	cvt.f32.bf16 	%r87, %rs8;
	cvt.f32.bf16 	%r88, %rs7;
	.loc	1 90 36                         // sk09_norm_embed.py:90:36
	abs.f32 	%r89, %r88;
	abs.f32 	%r90, %r87;
	abs.f32 	%r91, %r86;
	abs.f32 	%r92, %r85;
	.loc	1 89 73                         // sk09_norm_embed.py:89:73
	mov.b32 	{%rs9, %rs10}, %r4;
	cvt.f32.bf16 	%r93, %rs10;
	cvt.f32.bf16 	%r94, %rs9;
	mov.b32 	{%rs11, %rs12}, %r3;
	cvt.f32.bf16 	%r95, %rs12;
	cvt.f32.bf16 	%r96, %rs11;
	.loc	1 90 36                         // sk09_norm_embed.py:90:36
	abs.f32 	%r97, %r96;
	abs.f32 	%r98, %r95;
	abs.f32 	%r99, %r94;
	abs.f32 	%r100, %r93;
	.loc	1 89 73                         // sk09_norm_embed.py:89:73
	mov.b32 	{%rs13, %rs14}, %r2;
	cvt.f32.bf16 	%r101, %rs14;
	cvt.f32.bf16 	%r102, %rs13;
	mov.b32 	{%rs15, %rs16}, %r1;
	cvt.f32.bf16 	%r103, %rs16;
	cvt.f32.bf16 	%r104, %rs15;
	.loc	1 90 36                         // sk09_norm_embed.py:90:36
	abs.f32 	%r105, %r104;
	abs.f32 	%r106, %r103;
	abs.f32 	%r107, %r102;
	abs.f32 	%r108, %r101;
$L__tmp3:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r109, %r105, %r106;
	max.f32 	%r110, %r109, %r107;
	max.f32 	%r111, %r110, %r108;
	max.f32 	%r112, %r111, %r97;
	max.f32 	%r113, %r112, %r98;
	max.f32 	%r114, %r113, %r99;
	max.f32 	%r115, %r114, %r100;
	max.f32 	%r116, %r115, %r89;
	max.f32 	%r117, %r116, %r90;
	max.f32 	%r118, %r117, %r91;
	max.f32 	%r119, %r118, %r92;
	max.f32 	%r120, %r119, %r81;
	max.f32 	%r121, %r120, %r82;
	max.f32 	%r122, %r121, %r83;
	max.f32 	%r123, %r122, %r84;
$L__tmp4:
	.loc	1 89 73                         // sk09_norm_embed.py:89:73
	mov.b32 	{%rs17, %rs18}, %r17;
	cvt.f32.bf16 	%r124, %rs18;
	cvt.f32.bf16 	%r125, %rs17;
	mov.b32 	{%rs19, %rs20}, %r16;
	cvt.f32.bf16 	%r126, %rs20;
	cvt.f32.bf16 	%r127, %rs19;
	.loc	1 90 36                         // sk09_norm_embed.py:90:36
	abs.f32 	%r128, %r127;
	abs.f32 	%r129, %r126;
	abs.f32 	%r130, %r125;
	abs.f32 	%r131, %r124;
	.loc	1 89 73                         // sk09_norm_embed.py:89:73
	mov.b32 	{%rs21, %rs22}, %r15;
	cvt.f32.bf16 	%r132, %rs22;
	cvt.f32.bf16 	%r133, %rs21;
	mov.b32 	{%rs23, %rs24}, %r14;
	cvt.f32.bf16 	%r134, %rs24;
	cvt.f32.bf16 	%r135, %rs23;
	.loc	1 90 36                         // sk09_norm_embed.py:90:36
	abs.f32 	%r136, %r135;
	abs.f32 	%r137, %r134;
	abs.f32 	%r138, %r133;
	abs.f32 	%r139, %r132;
	.loc	1 89 73                         // sk09_norm_embed.py:89:73
	mov.b32 	{%rs25, %rs26}, %r13;
	cvt.f32.bf16 	%r140, %rs26;
	cvt.f32.bf16 	%r141, %rs25;
	mov.b32 	{%rs27, %rs28}, %r12;
	cvt.f32.bf16 	%r142, %rs28;
	cvt.f32.bf16 	%r143, %rs27;
	.loc	1 90 36                         // sk09_norm_embed.py:90:36
	abs.f32 	%r144, %r143;
	abs.f32 	%r145, %r142;
	abs.f32 	%r146, %r141;
	abs.f32 	%r147, %r140;
	.loc	1 89 73                         // sk09_norm_embed.py:89:73
	mov.b32 	{%rs29, %rs30}, %r11;
	cvt.f32.bf16 	%r148, %rs30;
	cvt.f32.bf16 	%r149, %rs29;
	mov.b32 	{%rs31, %rs32}, %r10;
	cvt.f32.bf16 	%r150, %rs32;
	cvt.f32.bf16 	%r151, %rs31;
	.loc	1 90 36                         // sk09_norm_embed.py:90:36
	abs.f32 	%r152, %r151;
	abs.f32 	%r153, %r150;
	abs.f32 	%r154, %r149;
	abs.f32 	%r155, %r148;
$L__tmp5:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r156, %r123, %r152;
	max.f32 	%r157, %r156, %r153;
	max.f32 	%r158, %r157, %r154;
	max.f32 	%r159, %r158, %r155;
	max.f32 	%r160, %r159, %r144;
	max.f32 	%r161, %r160, %r145;
	max.f32 	%r162, %r161, %r146;
	max.f32 	%r163, %r162, %r147;
	max.f32 	%r164, %r163, %r136;
	max.f32 	%r165, %r164, %r137;
	max.f32 	%r166, %r165, %r138;
	max.f32 	%r167, %r166, %r139;
	max.f32 	%r168, %r167, %r128;
	max.f32 	%r169, %r168, %r129;
	max.f32 	%r170, %r169, %r130;
	max.f32 	%r171, %r170, %r131;
$L__tmp6:
	.loc	1 89 73                         // sk09_norm_embed.py:89:73
	mov.b32 	{%rs33, %rs34}, %r25;
	cvt.f32.bf16 	%r172, %rs34;
	cvt.f32.bf16 	%r173, %rs33;
	mov.b32 	{%rs35, %rs36}, %r24;
	cvt.f32.bf16 	%r174, %rs36;
	cvt.f32.bf16 	%r175, %rs35;
	.loc	1 90 36                         // sk09_norm_embed.py:90:36
	abs.f32 	%r176, %r175;
	abs.f32 	%r177, %r174;
	abs.f32 	%r178, %r173;
	abs.f32 	%r179, %r172;
	.loc	1 89 73                         // sk09_norm_embed.py:89:73
	mov.b32 	{%rs37, %rs38}, %r23;
	cvt.f32.bf16 	%r180, %rs38;
	cvt.f32.bf16 	%r181, %rs37;
	mov.b32 	{%rs39, %rs40}, %r22;
	cvt.f32.bf16 	%r182, %rs40;
	cvt.f32.bf16 	%r183, %rs39;
	.loc	1 90 36                         // sk09_norm_embed.py:90:36
	abs.f32 	%r184, %r183;
	abs.f32 	%r185, %r182;
	abs.f32 	%r186, %r181;
	abs.f32 	%r187, %r180;
	.loc	1 89 73                         // sk09_norm_embed.py:89:73
	mov.b32 	{%rs41, %rs42}, %r21;
	cvt.f32.bf16 	%r188, %rs42;
	cvt.f32.bf16 	%r189, %rs41;
	mov.b32 	{%rs43, %rs44}, %r20;
	cvt.f32.bf16 	%r190, %rs44;
	cvt.f32.bf16 	%r191, %rs43;
	.loc	1 90 36                         // sk09_norm_embed.py:90:36
	abs.f32 	%r192, %r191;
	abs.f32 	%r193, %r190;
	abs.f32 	%r194, %r189;
	abs.f32 	%r195, %r188;
	.loc	1 89 73                         // sk09_norm_embed.py:89:73
	mov.b32 	{%rs45, %rs46}, %r19;
	cvt.f32.bf16 	%r196, %rs46;
	cvt.f32.bf16 	%r197, %rs45;
	mov.b32 	{%rs47, %rs48}, %r18;
	cvt.f32.bf16 	%r198, %rs48;
	cvt.f32.bf16 	%r199, %rs47;
	.loc	1 90 36                         // sk09_norm_embed.py:90:36
	abs.f32 	%r200, %r199;
	abs.f32 	%r201, %r198;
	abs.f32 	%r202, %r197;
	abs.f32 	%r203, %r196;
$L__tmp7:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r204, %r171, %r200;
	max.f32 	%r205, %r204, %r201;
	max.f32 	%r206, %r205, %r202;
	max.f32 	%r207, %r206, %r203;
	max.f32 	%r208, %r207, %r192;
	max.f32 	%r209, %r208, %r193;
	max.f32 	%r210, %r209, %r194;
	max.f32 	%r211, %r210, %r195;
	max.f32 	%r212, %r211, %r184;
	max.f32 	%r213, %r212, %r185;
	max.f32 	%r214, %r213, %r186;
	max.f32 	%r215, %r214, %r187;
	max.f32 	%r216, %r215, %r176;
	max.f32 	%r217, %r216, %r177;
	max.f32 	%r218, %r217, %r178;
	max.f32 	%r219, %r218, %r179;
$L__tmp8:
	.loc	1 89 73                         // sk09_norm_embed.py:89:73
	mov.b32 	{%rs49, %rs50}, %r33;
	cvt.f32.bf16 	%r220, %rs50;
	cvt.f32.bf16 	%r221, %rs49;
	mov.b32 	{%rs51, %rs52}, %r32;
	cvt.f32.bf16 	%r222, %rs52;
	cvt.f32.bf16 	%r223, %rs51;
	.loc	1 90 36                         // sk09_norm_embed.py:90:36
	abs.f32 	%r224, %r223;
	abs.f32 	%r225, %r222;
	abs.f32 	%r226, %r221;
	abs.f32 	%r227, %r220;
	.loc	1 89 73                         // sk09_norm_embed.py:89:73
	mov.b32 	{%rs53, %rs54}, %r31;
	cvt.f32.bf16 	%r228, %rs54;
	cvt.f32.bf16 	%r229, %rs53;
	mov.b32 	{%rs55, %rs56}, %r30;
	cvt.f32.bf16 	%r230, %rs56;
	cvt.f32.bf16 	%r231, %rs55;
	.loc	1 90 36                         // sk09_norm_embed.py:90:36
	abs.f32 	%r232, %r231;
	abs.f32 	%r233, %r230;
	abs.f32 	%r234, %r229;
	abs.f32 	%r235, %r228;
	.loc	1 89 73                         // sk09_norm_embed.py:89:73
	mov.b32 	{%rs57, %rs58}, %r29;
	cvt.f32.bf16 	%r236, %rs58;
	cvt.f32.bf16 	%r237, %rs57;
	mov.b32 	{%rs59, %rs60}, %r28;
	cvt.f32.bf16 	%r238, %rs60;
	cvt.f32.bf16 	%r239, %rs59;
	.loc	1 90 36                         // sk09_norm_embed.py:90:36
	abs.f32 	%r240, %r239;
	abs.f32 	%r241, %r238;
	abs.f32 	%r242, %r237;
	abs.f32 	%r243, %r236;
	.loc	1 89 73                         // sk09_norm_embed.py:89:73
	mov.b32 	{%rs61, %rs62}, %r27;
	cvt.f32.bf16 	%r244, %rs62;
	cvt.f32.bf16 	%r245, %rs61;
	mov.b32 	{%rs63, %rs64}, %r26;
	cvt.f32.bf16 	%r246, %rs64;
	cvt.f32.bf16 	%r247, %rs63;
	.loc	1 90 36                         // sk09_norm_embed.py:90:36
	abs.f32 	%r248, %r247;
	abs.f32 	%r249, %r246;
	abs.f32 	%r250, %r245;
	abs.f32 	%r251, %r244;
$L__tmp9:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r252, %r219, %r248;
	max.f32 	%r253, %r252, %r249;
	max.f32 	%r254, %r253, %r250;
	max.f32 	%r255, %r254, %r251;
	max.f32 	%r256, %r255, %r240;
	max.f32 	%r257, %r256, %r241;
	max.f32 	%r258, %r257, %r242;
	max.f32 	%r259, %r258, %r243;
	max.f32 	%r260, %r259, %r232;
	max.f32 	%r261, %r260, %r233;
	max.f32 	%r262, %r261, %r234;
	max.f32 	%r263, %r262, %r235;
	max.f32 	%r264, %r263, %r224;
	max.f32 	%r265, %r264, %r225;
	max.f32 	%r266, %r265, %r226;
	max.f32 	%r267, %r266, %r227;
$L__tmp10:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	shfl.sync.bfly.b32 	%r268, %r267, 16, 31, -1;
$L__tmp11:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r269, %r267, %r268;
$L__tmp12:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	shfl.sync.bfly.b32 	%r270, %r269, 8, 31, -1;
$L__tmp13:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r271, %r269, %r270;
$L__tmp14:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	shfl.sync.bfly.b32 	%r272, %r271, 4, 31, -1;
$L__tmp15:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r273, %r271, %r272;
$L__tmp16:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	shfl.sync.bfly.b32 	%r274, %r273, 2, 31, -1;
$L__tmp17:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r275, %r273, %r274;
$L__tmp18:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	shfl.sync.bfly.b32 	%r276, %r275, 1, 31, -1;
$L__tmp19:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r35, %r275, %r276;
$L__tmp20:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	// begin inline asm
	@%p5 st.shared.b32 [ %r34 + 0 ], %r35;
	// end inline asm
	bar.sync 	0;
	// begin inline asm
	@%p6 ld.shared.b32 %r36, [ %r37 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r277, %r36, 4, 31, -1;
$L__tmp21:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r278, %r36, %r277;
$L__tmp22:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	shfl.sync.bfly.b32 	%r279, %r278, 2, 31, -1;
$L__tmp23:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r280, %r278, %r279;
$L__tmp24:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	shfl.sync.bfly.b32 	%r281, %r280, 1, 31, -1;
$L__tmp25:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:90:29 ] ]
	max.f32 	%r38, %r280, %r281;
$L__tmp26:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:90:29 ]
	// begin inline asm
	@%p7 st.shared.b32 [ %r37 + 0 ], %r38;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r282, [global_smem];
$L__tmp27:
	.loc	1 90 49                         // sk09_norm_embed.py:90:49
	max.f32 	%r283, %r282, 0f0DA24260;
	mov.b32 	%r284, 0f42FE0000;
	.loc	1 91 22                         // sk09_norm_embed.py:91:22
	div.full.f32 	%r285, %r284, %r283;
	.loc	1 91 14                         // sk09_norm_embed.py:91:14
	mul.f32 	%r286, %r285, %r103;
	mul.f32 	%r287, %r285, %r104;
	mul.f32 	%r288, %r285, %r101;
	mul.f32 	%r289, %r285, %r102;
	mul.f32 	%r290, %r285, %r95;
	mul.f32 	%r291, %r285, %r96;
	mul.f32 	%r292, %r285, %r93;
	mul.f32 	%r293, %r285, %r94;
	mul.f32 	%r294, %r285, %r87;
	mul.f32 	%r295, %r285, %r88;
	mul.f32 	%r296, %r285, %r85;
	mul.f32 	%r297, %r285, %r86;
	mul.f32 	%r298, %r285, %r79;
	mul.f32 	%r299, %r285, %r80;
	mul.f32 	%r300, %r285, %r77;
	mul.f32 	%r301, %r285, %r78;
	mul.f32 	%r302, %r285, %r150;
	mul.f32 	%r303, %r285, %r151;
	mul.f32 	%r304, %r285, %r148;
	mul.f32 	%r305, %r285, %r149;
	mul.f32 	%r306, %r285, %r142;
	mul.f32 	%r307, %r285, %r143;
	mul.f32 	%r308, %r285, %r140;
	mul.f32 	%r309, %r285, %r141;
	mul.f32 	%r310, %r285, %r134;
	mul.f32 	%r311, %r285, %r135;
	mul.f32 	%r312, %r285, %r132;
	mul.f32 	%r313, %r285, %r133;
	mul.f32 	%r314, %r285, %r126;
	mul.f32 	%r315, %r285, %r127;
	mul.f32 	%r316, %r285, %r124;
	mul.f32 	%r317, %r285, %r125;
	mul.f32 	%r318, %r285, %r198;
	mul.f32 	%r319, %r285, %r199;
	mul.f32 	%r320, %r285, %r196;
	mul.f32 	%r321, %r285, %r197;
	mul.f32 	%r322, %r285, %r190;
	mul.f32 	%r323, %r285, %r191;
	mul.f32 	%r324, %r285, %r188;
	mul.f32 	%r325, %r285, %r189;
	mul.f32 	%r326, %r285, %r182;
	mul.f32 	%r327, %r285, %r183;
	mul.f32 	%r328, %r285, %r180;
	mul.f32 	%r329, %r285, %r181;
	mul.f32 	%r330, %r285, %r174;
	mul.f32 	%r331, %r285, %r175;
	mul.f32 	%r332, %r285, %r172;
	mul.f32 	%r333, %r285, %r173;
	mul.f32 	%r334, %r285, %r246;
	mul.f32 	%r335, %r285, %r247;
	mul.f32 	%r336, %r285, %r244;
	mul.f32 	%r337, %r285, %r245;
	mul.f32 	%r338, %r285, %r238;
	mul.f32 	%r339, %r285, %r239;
	mul.f32 	%r340, %r285, %r236;
	mul.f32 	%r341, %r285, %r237;
	mul.f32 	%r342, %r285, %r230;
	mul.f32 	%r343, %r285, %r231;
	mul.f32 	%r344, %r285, %r228;
	mul.f32 	%r345, %r285, %r229;
	mul.f32 	%r346, %r285, %r222;
	mul.f32 	%r347, %r285, %r223;
	mul.f32 	%r348, %r285, %r220;
	mul.f32 	%r349, %r285, %r221;
	.loc	1 92 29                         // sk09_norm_embed.py:92:29
	.loc	1 92 39                         // sk09_norm_embed.py:92:39
	lop3.b32 	%r350, 0x3f000000, %r286, 0x80000000, 0xF8;
	lop3.b32 	%r351, 0x3f000000, %r287, 0x80000000, 0xF8;
	lop3.b32 	%r352, 0x3f000000, %r288, 0x80000000, 0xF8;
	lop3.b32 	%r353, 0x3f000000, %r289, 0x80000000, 0xF8;
	lop3.b32 	%r354, 0x3f000000, %r290, 0x80000000, 0xF8;
	lop3.b32 	%r355, 0x3f000000, %r291, 0x80000000, 0xF8;
	lop3.b32 	%r356, 0x3f000000, %r292, 0x80000000, 0xF8;
	lop3.b32 	%r357, 0x3f000000, %r293, 0x80000000, 0xF8;
	lop3.b32 	%r358, 0x3f000000, %r294, 0x80000000, 0xF8;
	lop3.b32 	%r359, 0x3f000000, %r295, 0x80000000, 0xF8;
	lop3.b32 	%r360, 0x3f000000, %r296, 0x80000000, 0xF8;
	lop3.b32 	%r361, 0x3f000000, %r297, 0x80000000, 0xF8;
	lop3.b32 	%r362, 0x3f000000, %r298, 0x80000000, 0xF8;
	lop3.b32 	%r363, 0x3f000000, %r299, 0x80000000, 0xF8;
	lop3.b32 	%r364, 0x3f000000, %r300, 0x80000000, 0xF8;
	lop3.b32 	%r365, 0x3f000000, %r301, 0x80000000, 0xF8;
	lop3.b32 	%r366, 0x3f000000, %r302, 0x80000000, 0xF8;
	lop3.b32 	%r367, 0x3f000000, %r303, 0x80000000, 0xF8;
	lop3.b32 	%r368, 0x3f000000, %r304, 0x80000000, 0xF8;
	lop3.b32 	%r369, 0x3f000000, %r305, 0x80000000, 0xF8;
	lop3.b32 	%r370, 0x3f000000, %r306, 0x80000000, 0xF8;
	lop3.b32 	%r371, 0x3f000000, %r307, 0x80000000, 0xF8;
	lop3.b32 	%r372, 0x3f000000, %r308, 0x80000000, 0xF8;
	lop3.b32 	%r373, 0x3f000000, %r309, 0x80000000, 0xF8;
	lop3.b32 	%r374, 0x3f000000, %r310, 0x80000000, 0xF8;
	lop3.b32 	%r375, 0x3f000000, %r311, 0x80000000, 0xF8;
	lop3.b32 	%r376, 0x3f000000, %r312, 0x80000000, 0xF8;
	lop3.b32 	%r377, 0x3f000000, %r313, 0x80000000, 0xF8;
	lop3.b32 	%r378, 0x3f000000, %r314, 0x80000000, 0xF8;
	lop3.b32 	%r379, 0x3f000000, %r315, 0x80000000, 0xF8;
	lop3.b32 	%r380, 0x3f000000, %r316, 0x80000000, 0xF8;
	lop3.b32 	%r381, 0x3f000000, %r317, 0x80000000, 0xF8;
	lop3.b32 	%r382, 0x3f000000, %r318, 0x80000000, 0xF8;
	lop3.b32 	%r383, 0x3f000000, %r319, 0x80000000, 0xF8;
	lop3.b32 	%r384, 0x3f000000, %r320, 0x80000000, 0xF8;
	lop3.b32 	%r385, 0x3f000000, %r321, 0x80000000, 0xF8;
	lop3.b32 	%r386, 0x3f000000, %r322, 0x80000000, 0xF8;
	lop3.b32 	%r387, 0x3f000000, %r323, 0x80000000, 0xF8;
	lop3.b32 	%r388, 0x3f000000, %r324, 0x80000000, 0xF8;
	lop3.b32 	%r389, 0x3f000000, %r325, 0x80000000, 0xF8;
	lop3.b32 	%r390, 0x3f000000, %r326, 0x80000000, 0xF8;
	lop3.b32 	%r391, 0x3f000000, %r327, 0x80000000, 0xF8;
	lop3.b32 	%r392, 0x3f000000, %r328, 0x80000000, 0xF8;
	lop3.b32 	%r393, 0x3f000000, %r329, 0x80000000, 0xF8;
	lop3.b32 	%r394, 0x3f000000, %r330, 0x80000000, 0xF8;
	lop3.b32 	%r395, 0x3f000000, %r331, 0x80000000, 0xF8;
	lop3.b32 	%r396, 0x3f000000, %r332, 0x80000000, 0xF8;
	lop3.b32 	%r397, 0x3f000000, %r333, 0x80000000, 0xF8;
	lop3.b32 	%r398, 0x3f000000, %r334, 0x80000000, 0xF8;
	lop3.b32 	%r399, 0x3f000000, %r335, 0x80000000, 0xF8;
	lop3.b32 	%r400, 0x3f000000, %r336, 0x80000000, 0xF8;
	lop3.b32 	%r401, 0x3f000000, %r337, 0x80000000, 0xF8;
	lop3.b32 	%r402, 0x3f000000, %r338, 0x80000000, 0xF8;
	lop3.b32 	%r403, 0x3f000000, %r339, 0x80000000, 0xF8;
	lop3.b32 	%r404, 0x3f000000, %r340, 0x80000000, 0xF8;
	lop3.b32 	%r405, 0x3f000000, %r341, 0x80000000, 0xF8;
	lop3.b32 	%r406, 0x3f000000, %r342, 0x80000000, 0xF8;
	lop3.b32 	%r407, 0x3f000000, %r343, 0x80000000, 0xF8;
	lop3.b32 	%r408, 0x3f000000, %r344, 0x80000000, 0xF8;
	lop3.b32 	%r409, 0x3f000000, %r345, 0x80000000, 0xF8;
	lop3.b32 	%r410, 0x3f000000, %r346, 0x80000000, 0xF8;
	lop3.b32 	%r411, 0x3f000000, %r347, 0x80000000, 0xF8;
	lop3.b32 	%r412, 0x3f000000, %r348, 0x80000000, 0xF8;
	lop3.b32 	%r413, 0x3f000000, %r349, 0x80000000, 0xF8;
	.loc	1 92 14                         // sk09_norm_embed.py:92:14
	fma.rn.f32 	%r414, %r285, %r102, %r353;
	fma.rn.f32 	%r415, %r285, %r101, %r352;
	fma.rn.f32 	%r416, %r285, %r104, %r351;
	fma.rn.f32 	%r417, %r285, %r103, %r350;
	fma.rn.f32 	%r418, %r285, %r94, %r357;
	fma.rn.f32 	%r419, %r285, %r93, %r356;
	fma.rn.f32 	%r420, %r285, %r96, %r355;
	fma.rn.f32 	%r421, %r285, %r95, %r354;
	fma.rn.f32 	%r422, %r285, %r86, %r361;
	fma.rn.f32 	%r423, %r285, %r85, %r360;
	fma.rn.f32 	%r424, %r285, %r88, %r359;
	fma.rn.f32 	%r425, %r285, %r87, %r358;
	fma.rn.f32 	%r426, %r285, %r78, %r365;
	fma.rn.f32 	%r427, %r285, %r77, %r364;
	fma.rn.f32 	%r428, %r285, %r80, %r363;
	fma.rn.f32 	%r429, %r285, %r79, %r362;
	fma.rn.f32 	%r430, %r285, %r149, %r369;
	fma.rn.f32 	%r431, %r285, %r148, %r368;
	fma.rn.f32 	%r432, %r285, %r151, %r367;
	fma.rn.f32 	%r433, %r285, %r150, %r366;
	fma.rn.f32 	%r434, %r285, %r141, %r373;
	fma.rn.f32 	%r435, %r285, %r140, %r372;
	fma.rn.f32 	%r436, %r285, %r143, %r371;
	fma.rn.f32 	%r437, %r285, %r142, %r370;
	fma.rn.f32 	%r438, %r285, %r133, %r377;
	fma.rn.f32 	%r439, %r285, %r132, %r376;
	fma.rn.f32 	%r440, %r285, %r135, %r375;
	fma.rn.f32 	%r441, %r285, %r134, %r374;
	fma.rn.f32 	%r442, %r285, %r125, %r381;
	fma.rn.f32 	%r443, %r285, %r124, %r380;
	fma.rn.f32 	%r444, %r285, %r127, %r379;
	fma.rn.f32 	%r445, %r285, %r126, %r378;
	fma.rn.f32 	%r446, %r285, %r197, %r385;
	fma.rn.f32 	%r447, %r285, %r196, %r384;
	fma.rn.f32 	%r448, %r285, %r199, %r383;
	fma.rn.f32 	%r449, %r285, %r198, %r382;
	fma.rn.f32 	%r450, %r285, %r189, %r389;
	fma.rn.f32 	%r451, %r285, %r188, %r388;
	fma.rn.f32 	%r452, %r285, %r191, %r387;
	fma.rn.f32 	%r453, %r285, %r190, %r386;
	fma.rn.f32 	%r454, %r285, %r181, %r393;
	fma.rn.f32 	%r455, %r285, %r180, %r392;
	fma.rn.f32 	%r456, %r285, %r183, %r391;
	fma.rn.f32 	%r457, %r285, %r182, %r390;
	fma.rn.f32 	%r458, %r285, %r173, %r397;
	fma.rn.f32 	%r459, %r285, %r172, %r396;
	fma.rn.f32 	%r460, %r285, %r175, %r395;
	fma.rn.f32 	%r461, %r285, %r174, %r394;
	fma.rn.f32 	%r462, %r285, %r245, %r401;
	fma.rn.f32 	%r463, %r285, %r244, %r400;
	fma.rn.f32 	%r464, %r285, %r247, %r399;
	fma.rn.f32 	%r465, %r285, %r246, %r398;
	fma.rn.f32 	%r466, %r285, %r237, %r405;
	fma.rn.f32 	%r467, %r285, %r236, %r404;
	fma.rn.f32 	%r468, %r285, %r239, %r403;
	fma.rn.f32 	%r469, %r285, %r238, %r402;
	fma.rn.f32 	%r470, %r285, %r229, %r409;
	fma.rn.f32 	%r471, %r285, %r228, %r408;
	fma.rn.f32 	%r472, %r285, %r231, %r407;
	fma.rn.f32 	%r473, %r285, %r230, %r406;
	fma.rn.f32 	%r474, %r285, %r221, %r413;
	fma.rn.f32 	%r475, %r285, %r220, %r412;
	fma.rn.f32 	%r476, %r285, %r223, %r411;
	fma.rn.f32 	%r477, %r285, %r222, %r410;
	.loc	1 92 49                         // sk09_norm_embed.py:92:49
	cvt.rzi.s32.f32 	%r478, %r417;
	cvt.rzi.s32.f32 	%r479, %r416;
	cvt.rzi.s32.f32 	%r480, %r415;
	cvt.rzi.s32.f32 	%r481, %r414;
	cvt.rzi.s32.f32 	%r482, %r421;
	cvt.rzi.s32.f32 	%r483, %r420;
	cvt.rzi.s32.f32 	%r484, %r419;
	cvt.rzi.s32.f32 	%r485, %r418;
	cvt.rzi.s32.f32 	%r486, %r425;
	cvt.rzi.s32.f32 	%r487, %r424;
	cvt.rzi.s32.f32 	%r488, %r423;
	cvt.rzi.s32.f32 	%r489, %r422;
	cvt.rzi.s32.f32 	%r490, %r429;
	cvt.rzi.s32.f32 	%r491, %r428;
	cvt.rzi.s32.f32 	%r492, %r427;
	cvt.rzi.s32.f32 	%r493, %r426;
	cvt.rzi.s32.f32 	%r494, %r433;
	cvt.rzi.s32.f32 	%r495, %r432;
	cvt.rzi.s32.f32 	%r496, %r431;
	cvt.rzi.s32.f32 	%r497, %r430;
	cvt.rzi.s32.f32 	%r498, %r437;
	cvt.rzi.s32.f32 	%r499, %r436;
	cvt.rzi.s32.f32 	%r500, %r435;
	cvt.rzi.s32.f32 	%r501, %r434;
	cvt.rzi.s32.f32 	%r502, %r441;
	cvt.rzi.s32.f32 	%r503, %r440;
	cvt.rzi.s32.f32 	%r504, %r439;
	cvt.rzi.s32.f32 	%r505, %r438;
	cvt.rzi.s32.f32 	%r506, %r445;
	cvt.rzi.s32.f32 	%r507, %r444;
	cvt.rzi.s32.f32 	%r508, %r443;
	cvt.rzi.s32.f32 	%r509, %r442;
	cvt.rzi.s32.f32 	%r510, %r449;
	cvt.rzi.s32.f32 	%r511, %r448;
	cvt.rzi.s32.f32 	%r512, %r447;
	cvt.rzi.s32.f32 	%r513, %r446;
	cvt.rzi.s32.f32 	%r514, %r453;
	cvt.rzi.s32.f32 	%r515, %r452;
	cvt.rzi.s32.f32 	%r516, %r451;
	cvt.rzi.s32.f32 	%r517, %r450;
	cvt.rzi.s32.f32 	%r518, %r457;
	cvt.rzi.s32.f32 	%r519, %r456;
	cvt.rzi.s32.f32 	%r520, %r455;
	cvt.rzi.s32.f32 	%r521, %r454;
	cvt.rzi.s32.f32 	%r522, %r461;
	cvt.rzi.s32.f32 	%r523, %r460;
	cvt.rzi.s32.f32 	%r524, %r459;
	cvt.rzi.s32.f32 	%r525, %r458;
	cvt.rzi.s32.f32 	%r526, %r465;
	cvt.rzi.s32.f32 	%r527, %r464;
	cvt.rzi.s32.f32 	%r528, %r463;
	cvt.rzi.s32.f32 	%r529, %r462;
	cvt.rzi.s32.f32 	%r530, %r469;
	cvt.rzi.s32.f32 	%r531, %r468;
	cvt.rzi.s32.f32 	%r532, %r467;
	cvt.rzi.s32.f32 	%r533, %r466;
	cvt.rzi.s32.f32 	%r534, %r473;
	cvt.rzi.s32.f32 	%r535, %r472;
	cvt.rzi.s32.f32 	%r536, %r471;
	cvt.rzi.s32.f32 	%r537, %r470;
	cvt.rzi.s32.f32 	%r538, %r477;
	cvt.rzi.s32.f32 	%r539, %r476;
	cvt.rzi.s32.f32 	%r540, %r475;
	cvt.rzi.s32.f32 	%r541, %r474;
	.loc	1 93 70                         // sk09_norm_embed.py:93:70
	max.s32 	%r542, %r481, -127;
	max.s32 	%r543, %r480, -127;
	max.s32 	%r544, %r479, -127;
	max.s32 	%r545, %r478, -127;
	max.s32 	%r546, %r485, -127;
	max.s32 	%r547, %r484, -127;
	max.s32 	%r548, %r483, -127;
	max.s32 	%r549, %r482, -127;
	max.s32 	%r550, %r489, -127;
	max.s32 	%r551, %r488, -127;
	max.s32 	%r552, %r487, -127;
	max.s32 	%r553, %r486, -127;
	max.s32 	%r554, %r493, -127;
	max.s32 	%r555, %r492, -127;
	max.s32 	%r556, %r491, -127;
	max.s32 	%r557, %r490, -127;
	max.s32 	%r558, %r497, -127;
	max.s32 	%r559, %r496, -127;
	max.s32 	%r560, %r495, -127;
	max.s32 	%r561, %r494, -127;
	max.s32 	%r562, %r501, -127;
	max.s32 	%r563, %r500, -127;
	max.s32 	%r564, %r499, -127;
	max.s32 	%r565, %r498, -127;
	max.s32 	%r566, %r505, -127;
	max.s32 	%r567, %r504, -127;
	max.s32 	%r568, %r503, -127;
	max.s32 	%r569, %r502, -127;
	max.s32 	%r570, %r509, -127;
	max.s32 	%r571, %r508, -127;
	max.s32 	%r572, %r507, -127;
	max.s32 	%r573, %r506, -127;
	max.s32 	%r574, %r513, -127;
	max.s32 	%r575, %r512, -127;
	max.s32 	%r576, %r511, -127;
	max.s32 	%r577, %r510, -127;
	max.s32 	%r578, %r517, -127;
	max.s32 	%r579, %r516, -127;
	max.s32 	%r580, %r515, -127;
	max.s32 	%r581, %r514, -127;
	max.s32 	%r582, %r521, -127;
	max.s32 	%r583, %r520, -127;
	max.s32 	%r584, %r519, -127;
	max.s32 	%r585, %r518, -127;
	max.s32 	%r586, %r525, -127;
	max.s32 	%r587, %r524, -127;
	max.s32 	%r588, %r523, -127;
	max.s32 	%r589, %r522, -127;
	max.s32 	%r590, %r529, -127;
	max.s32 	%r591, %r528, -127;
	max.s32 	%r592, %r527, -127;
	max.s32 	%r593, %r526, -127;
	max.s32 	%r594, %r533, -127;
	max.s32 	%r595, %r532, -127;
	max.s32 	%r596, %r531, -127;
	max.s32 	%r597, %r530, -127;
	max.s32 	%r598, %r537, -127;
	max.s32 	%r599, %r536, -127;
	max.s32 	%r600, %r535, -127;
	max.s32 	%r601, %r534, -127;
	max.s32 	%r602, %r541, -127;
	max.s32 	%r603, %r540, -127;
	max.s32 	%r604, %r539, -127;
	max.s32 	%r605, %r538, -127;
	.loc	1 93 77                         // sk09_norm_embed.py:93:77
	min.s32 	%r606, %r545, 127;
	min.s32 	%r607, %r544, 127;
	min.s32 	%r608, %r543, 127;
	min.s32 	%r609, %r542, 127;
	min.s32 	%r610, %r549, 127;
	min.s32 	%r611, %r548, 127;
	min.s32 	%r612, %r547, 127;
	min.s32 	%r613, %r546, 127;
	min.s32 	%r614, %r553, 127;
	min.s32 	%r615, %r552, 127;
	min.s32 	%r616, %r551, 127;
	min.s32 	%r617, %r550, 127;
	min.s32 	%r618, %r557, 127;
	min.s32 	%r619, %r556, 127;
	min.s32 	%r620, %r555, 127;
	min.s32 	%r621, %r554, 127;
	min.s32 	%r622, %r561, 127;
	min.s32 	%r623, %r560, 127;
	min.s32 	%r624, %r559, 127;
	min.s32 	%r625, %r558, 127;
	min.s32 	%r626, %r565, 127;
	min.s32 	%r627, %r564, 127;
	min.s32 	%r628, %r563, 127;
	min.s32 	%r629, %r562, 127;
	min.s32 	%r630, %r569, 127;
	min.s32 	%r631, %r568, 127;
	min.s32 	%r632, %r567, 127;
	min.s32 	%r633, %r566, 127;
	min.s32 	%r634, %r573, 127;
	min.s32 	%r635, %r572, 127;
	min.s32 	%r636, %r571, 127;
	min.s32 	%r637, %r570, 127;
	min.s32 	%r638, %r577, 127;
	min.s32 	%r639, %r576, 127;
	min.s32 	%r640, %r575, 127;
	min.s32 	%r641, %r574, 127;
	min.s32 	%r642, %r581, 127;
	min.s32 	%r643, %r580, 127;
	min.s32 	%r644, %r579, 127;
	min.s32 	%r645, %r578, 127;
	min.s32 	%r646, %r585, 127;
	min.s32 	%r647, %r584, 127;
	min.s32 	%r648, %r583, 127;
	min.s32 	%r649, %r582, 127;
	min.s32 	%r650, %r589, 127;
	min.s32 	%r651, %r588, 127;
	min.s32 	%r652, %r587, 127;
	min.s32 	%r653, %r586, 127;
	min.s32 	%r654, %r593, 127;
	min.s32 	%r655, %r592, 127;
	min.s32 	%r656, %r591, 127;
	min.s32 	%r657, %r590, 127;
	min.s32 	%r658, %r597, 127;
	min.s32 	%r659, %r596, 127;
	min.s32 	%r660, %r595, 127;
	min.s32 	%r661, %r594, 127;
	min.s32 	%r662, %r601, 127;
	min.s32 	%r663, %r600, 127;
	min.s32 	%r664, %r599, 127;
	min.s32 	%r665, %r598, 127;
	min.s32 	%r666, %r605, 127;
	min.s32 	%r667, %r604, 127;
	min.s32 	%r668, %r603, 127;
	min.s32 	%r669, %r602, 127;
	.loc	1 93 85                         // sk09_norm_embed.py:93:85
	prmt.b32 	%r670, %r609, %r608, 0x3340U;
	prmt.b32 	%r671, %r607, %r606, 0x3340U;
	prmt.b32 	%r39, %r671, %r670, 0x5410U;
	prmt.b32 	%r672, %r613, %r612, 0x3340U;
	prmt.b32 	%r673, %r611, %r610, 0x3340U;
	prmt.b32 	%r40, %r673, %r672, 0x5410U;
	prmt.b32 	%r674, %r617, %r616, 0x3340U;
	prmt.b32 	%r675, %r615, %r614, 0x3340U;
	prmt.b32 	%r41, %r675, %r674, 0x5410U;
	prmt.b32 	%r676, %r621, %r620, 0x3340U;
	prmt.b32 	%r677, %r619, %r618, 0x3340U;
	prmt.b32 	%r42, %r677, %r676, 0x5410U;
	prmt.b32 	%r678, %r625, %r624, 0x3340U;
	prmt.b32 	%r679, %r623, %r622, 0x3340U;
	prmt.b32 	%r43, %r679, %r678, 0x5410U;
	prmt.b32 	%r680, %r629, %r628, 0x3340U;
	prmt.b32 	%r681, %r627, %r626, 0x3340U;
	prmt.b32 	%r44, %r681, %r680, 0x5410U;
	prmt.b32 	%r682, %r633, %r632, 0x3340U;
	prmt.b32 	%r683, %r631, %r630, 0x3340U;
	prmt.b32 	%r45, %r683, %r682, 0x5410U;
	prmt.b32 	%r684, %r637, %r636, 0x3340U;
	prmt.b32 	%r685, %r635, %r634, 0x3340U;
	prmt.b32 	%r46, %r685, %r684, 0x5410U;
	prmt.b32 	%r686, %r641, %r640, 0x3340U;
	prmt.b32 	%r687, %r639, %r638, 0x3340U;
	prmt.b32 	%r47, %r687, %r686, 0x5410U;
	prmt.b32 	%r688, %r645, %r644, 0x3340U;
	prmt.b32 	%r689, %r643, %r642, 0x3340U;
	prmt.b32 	%r48, %r689, %r688, 0x5410U;
	prmt.b32 	%r690, %r649, %r648, 0x3340U;
	prmt.b32 	%r691, %r647, %r646, 0x3340U;
	prmt.b32 	%r49, %r691, %r690, 0x5410U;
	prmt.b32 	%r692, %r653, %r652, 0x3340U;
	prmt.b32 	%r693, %r651, %r650, 0x3340U;
	prmt.b32 	%r50, %r693, %r692, 0x5410U;
	prmt.b32 	%r694, %r657, %r656, 0x3340U;
	prmt.b32 	%r695, %r655, %r654, 0x3340U;
	prmt.b32 	%r51, %r695, %r694, 0x5410U;
	prmt.b32 	%r696, %r661, %r660, 0x3340U;
	prmt.b32 	%r697, %r659, %r658, 0x3340U;
	prmt.b32 	%r52, %r697, %r696, 0x5410U;
	prmt.b32 	%r698, %r665, %r664, 0x3340U;
	prmt.b32 	%r699, %r663, %r662, 0x3340U;
	prmt.b32 	%r53, %r699, %r698, 0x5410U;
	prmt.b32 	%r700, %r669, %r668, 0x3340U;
	prmt.b32 	%r701, %r667, %r666, 0x3340U;
	prmt.b32 	%r54, %r701, %r700, 0x5410U;
	.loc	1 93 45                         // sk09_norm_embed.py:93:45
	// begin inline asm
	@%p1 st.global.v4.b32 [ %rd9 + 0 ], { %r39, %r40, %r41, %r42 };
	// end inline asm
	// begin inline asm
	@%p2 st.global.v4.b32 [ %rd10 + 0 ], { %r43, %r44, %r45, %r46 };
	// end inline asm
	// begin inline asm
	@%p3 st.global.v4.b32 [ %rd11 + 0 ], { %r47, %r48, %r49, %r50 };
	// end inline asm
	// begin inline asm
	@%p4 st.global.v4.b32 [ %rd12 + 0 ], { %r51, %r52, %r53, %r54 };
	// end inline asm
	.loc	1 94 21                         // sk09_norm_embed.py:94:21
	mad.wide.u32 	%rd13, %r56, 4, %rd16;
	.loc	1 94 34                         // sk09_norm_embed.py:94:34
	mul.f32 	%r55, %r283, 0f3C010204;
	.loc	1 94 26                         // sk09_norm_embed.py:94:26
	or.b32 	%r702, %r60, %r62;
	setp.eq.b32 	%p8, %r702, 0;
	// begin inline asm
	@%p8 st.global.b32 [ %rd13 + 0 ], { %r55 };
	// end inline asm
	.loc	1 94 4                          // sk09_norm_embed.py:94:4
	ret;
$L__tmp28:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk09_norm_embed.py"
	.file	2 "/usr/local/lib/python3.12/dist-packages/triton/language/standard.py"
	.section	.debug_abbrev
	{
.b8 1                                   // Abbreviation Code
.b8 17                                  // DW_TAG_compile_unit
.b8 1                                   // DW_CHILDREN_yes
.b8 37                                  // DW_AT_producer
.b8 8                                   // DW_FORM_string
.b8 19                                  // DW_AT_language
.b8 5                                   // DW_FORM_data2
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 16                                  // DW_AT_stmt_list
.b8 6                                   // DW_FORM_data4
.b8 27                                  // DW_AT_comp_dir
.b8 8                                   // DW_FORM_string
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 2                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 0                                   // DW_CHILDREN_no
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 32                                  // DW_AT_inline
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 3                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 1                                   // DW_CHILDREN_yes
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 4                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 1                                   // DW_CHILDREN_yes
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 11                                  // DW_FORM_data1
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 5                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 0                                   // DW_CHILDREN_no
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 11                                  // DW_FORM_data1
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 161                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x9a DW_TAG_compile_unit
.b8 116                                 // DW_AT_producer
.b8 114
.b8 105
.b8 116
.b8 111
.b8 110
.b8 0
.b8 2                                   // DW_AT_language
.b8 0
.b8 115                                 // DW_AT_name
.b8 107
.b8 48
.b8 57
.b8 95
.b8 110
.b8 111
.b8 114
.b8 109
.b8 95
.b8 101
.b8 109
.b8 98
.b8 101
.b8 100
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 114
.b8 101
.b8 112
.b8 111
.b8 47
.b8 118
.b8 108
.b8 108
.b8 109
.b8 47
.b8 95
.b8 103
.b8 101
.b8 110
.b8 101
.b8 115
.b8 105
.b8 115
.b8 47
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 115
.b8 0
.b8 2                                   // Abbrev [2] 0x48:0x15 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 57
.b8 95
.b8 113
.b8 117
.b8 97
.b8 110
.b8 116
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x5d:0x47 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 72                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x72:0x31 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp27                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 90                                  // DW_AT_call_line
.b8 29                                  // DW_AT_call_column
.b8 5                                   // Abbrev [5] 0x8a:0x18 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp3                           // DW_AT_low_pc
.b64 $L__tmp26                          // DW_AT_high_pc
.b8 2                                   // DW_AT_call_file
.b8 191                                 // DW_AT_call_line
.b8 40                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_QVAR0_3 = _Nativo(
    "sk09_norm_embed/_sk09_quant_kernel/blk16384",
    _QPTX0_3, "_sk09_quant_kernel",
    warps=8, shared=32,
    abi=[0, 1, 2, 3, 4, 5],
    horneado={6: 16384},
    div16=[3, 4, 5],
)
# E1=64 lop3, E2=0 mul

_POR_Q0 = [_QVAR0_0, _QVAR0_1, _QVAR0_2, _QVAR0_3]


def _q0_impl(grid: list[int], x_ptr: torch.Tensor, q_ptr: torch.Tensor, s_ptr: torch.Tensor, K: int, stride_xm: int, stride_qm: int, BLOCK: int) -> None:
    """Cuerpo del custom op: elige la variante y lanza el PTX."""
    args = (x_ptr, q_ptr, s_ptr, K, stride_xm, stride_qm, BLOCK)
    for _v in _POR_Q0:
        if all(args[_p] == _x for _p, _x in _v.horneado.items()):
            return _v(tuple(grid), *args)
    raise ValueError(
        "genesis_sk09_norm_embed_q0: no hay PTX embebido para estos constexpr; "
        "horneados disponibles: %r" % [_v.horneado for _v in _POR_Q0])


def _q0_fake(grid: list[int], x_ptr: torch.Tensor, q_ptr: torch.Tensor, s_ptr: torch.Tensor, K: int, stride_xm: int, stride_qm: int, BLOCK: int) -> None:
    """Meta impl: no toca la GPU; las salidas se mutan in-place."""
    return None


# Registro AL IMPORTAR, no en el primer uso: direct_register_custom_op
# llama a torch._library.infer_schema, que dynamo se niega a trazar
# ("Attempted to call function marked as skipped"). El hasattr evita el
# choque cuando el modulo se importa dos veces con nombres distintos,
# como hace el gate de tools/monolitizar.py.
if not hasattr(torch.ops.vllm, "genesis_sk09_norm_embed_q0"):
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op(
        op_name="genesis_sk09_norm_embed_q0",
        op_func=_q0_impl,
        mutates_args=['q_ptr', 's_ptr'],
        fake_impl=_q0_fake,
    )


def _lanzar_quant0(grid, *args):
    """Lanza el PTX embebido. Es el UNICO camino ejecutable.

    No hay fallback a Triton ni kill-switch: el kernel Triton de este
    archivo es privado y solo lo llaman los tests.
    """
    g = [grid] if isinstance(grid, int) else list(grid)
    return torch.ops.vllm.genesis_sk09_norm_embed_q0(g, *args)


# --- quant _sk09_rmsnorm_quant_kernel: 4 variante(s) de PTX embebido ---

_QPTX1_0 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk09_rmsnorm_quant_kernel // -- Begin function _sk09_rmsnorm_quant_kernel
.extern .shared .align 16 .b8 global_smem[];
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
                                        // @_sk09_rmsnorm_quant_kernel
.visible .entry _sk09_rmsnorm_quant_kernel(
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_quant_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_quant_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_quant_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_quant_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_quant_kernel_param_4,
	.param .u32 _sk09_rmsnorm_quant_kernel_param_5,
	.param .u32 _sk09_rmsnorm_quant_kernel_param_6,
	.param .u32 _sk09_rmsnorm_quant_kernel_param_7,
	.param .u32 _sk09_rmsnorm_quant_kernel_param_8,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_quant_kernel_param_9,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_quant_kernel_param_10
)
.reqntid 256
{
	.reg .pred 	%p<15>;
	.reg .b16 	%rs<17>;
	.reg .b32 	%r<218>;
	.reg .b64 	%rd<23>;
	.loc	1 64 0                          // sk09_norm_embed.py:64:0
$L__func_begin0:
	.loc	1 64 0                          // sk09_norm_embed.py:64:0

// %bb.0:                               // %__nv_rsqrtf.exit
	ld.param.b64 	%rd13, [_sk09_rmsnorm_quant_kernel_param_0];
	ld.param.b64 	%rd14, [_sk09_rmsnorm_quant_kernel_param_1];
$L__tmp0:
	.loc	1 69 24                         // sk09_norm_embed.py:69:24
	mov.u32 	%r30, %ctaid.x;
	ld.param.b64 	%rd15, [_sk09_rmsnorm_quant_kernel_param_2];
	.loc	1 70 24                         // sk09_norm_embed.py:70:24
	mov.u32 	%r31, %tid.x;
	and.b32 	%r32, %r31, 255;
	ld.param.b64 	%rd16, [_sk09_rmsnorm_quant_kernel_param_3];
	and.b32 	%r33, %r31, 31;
	ld.param.b64 	%rd17, [_sk09_rmsnorm_quant_kernel_param_4];
	shr.u32 	%r34, %r31, 5;
	ld.param.b32 	%r35, [_sk09_rmsnorm_quant_kernel_param_5];
	shl.b32 	%r36, %r31, 3;
	ld.param.b32 	%r37, [_sk09_rmsnorm_quant_kernel_param_6];
	and.b32 	%r38, %r36, 2040;
	ld.param.b32 	%r39, [_sk09_rmsnorm_quant_kernel_param_7];
	.loc	1 71 18                         // sk09_norm_embed.py:71:18
	setp.lt.s32 	%p1, %r38, %r35;
	ld.param.b32 	%r40, [_sk09_rmsnorm_quant_kernel_param_8];
	.loc	1 72 30                         // sk09_norm_embed.py:72:30
	mul.lo.s32 	%r41, %r37, %r30;
	.loc	1 72 24                         // sk09_norm_embed.py:72:24
	mad.wide.s32 	%rd18, %r41, 2, %rd13;
	.loc	1 72 42                         // sk09_norm_embed.py:72:42
	cvt.u64.u32 	%rd19, %r38;
	mul.wide.u32 	%rd20, %r38, 2;
	add.s64 	%rd1, %rd18, %rd20;
	mov.b32 	%r5, 0;
	.loc	1 72 16                         // sk09_norm_embed.py:72:16
	// begin inline asm
	mov.u32 %r1, %r5;
	mov.u32 %r2, %r5;
	mov.u32 %r3, %r5;
	mov.u32 %r4, %r5;
	@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
	// end inline asm
	.loc	1 72 73                         // sk09_norm_embed.py:72:73
	mov.b32 	{%rs1, %rs2}, %r2;
	cvt.f32.bf16 	%r42, %rs2;
	cvt.f32.bf16 	%r43, %rs1;
	mov.b32 	{%rs3, %rs4}, %r1;
	cvt.f32.bf16 	%r44, %rs3;
	cvt.f32.bf16 	%r45, %rs4;
	mov.b32 	{%rs5, %rs6}, %r4;
	cvt.f32.bf16 	%r46, %rs6;
	cvt.f32.bf16 	%r47, %rs5;
	mov.b32 	{%rs7, %rs8}, %r3;
	cvt.f32.bf16 	%r48, %rs8;
	cvt.f32.bf16 	%r49, %rs7;
	.loc	1 73 24                         // sk09_norm_embed.py:73:24
	add.s64 	%rd2, %rd14, %rd20;
	.loc	1 73 16                         // sk09_norm_embed.py:73:16
	// begin inline asm
	mov.u32 %r6, %r5;
	mov.u32 %r7, %r5;
	mov.u32 %r8, %r5;
	mov.u32 %r9, %r5;
	@%p1 ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd2 + 0 ];
	// end inline asm
	.loc	1 74 40                         // sk09_norm_embed.py:74:40
	mul.lo.s32 	%r50, %r40, %r38;
	shl.b32 	%r51, %r40, 1;
	add.s32 	%r52, %r50, %r51;
	mad.lo.s32 	%r53, %r40, 3, %r50;
	shl.b32 	%r54, %r40, 2;
	add.s32 	%r55, %r50, %r54;
	mad.lo.s32 	%r56, %r40, 5, %r50;
	mad.lo.s32 	%r57, %r40, 6, %r50;
	mad.lo.s32 	%r58, %r40, 7, %r50;
	.loc	1 74 33                         // sk09_norm_embed.py:74:33
	mad.wide.s32 	%rd3, %r50, 4, %rd15;
	mad.wide.s32 	%rd4, %r40, 4, %rd3;
	mad.wide.s32 	%rd5, %r52, 4, %rd15;
	mad.wide.s32 	%rd6, %r53, 4, %rd15;
	mad.wide.s32 	%rd7, %r55, 4, %rd15;
	mad.wide.s32 	%rd8, %r56, 4, %rd15;
	mad.wide.s32 	%rd9, %r57, 4, %rd15;
	mad.wide.s32 	%rd10, %r58, 4, %rd15;
	mov.b32 	%r11, 1065353216;
	.loc	1 74 20                         // sk09_norm_embed.py:74:20
	// begin inline asm
	mov.u32 %r10, %r11;
	@%p1 ld.global.b32 { %r10 }, [ %rd3 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r12, %r11;
	@%p1 ld.global.b32 { %r12 }, [ %rd4 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r13, %r11;
	@%p1 ld.global.b32 { %r13 }, [ %rd5 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r14, %r11;
	@%p1 ld.global.b32 { %r14 }, [ %rd6 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r15, %r11;
	@%p1 ld.global.b32 { %r15 }, [ %rd7 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r16, %r11;
	@%p1 ld.global.b32 { %r16 }, [ %rd8 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r17, %r11;
	@%p1 ld.global.b32 { %r17 }, [ %rd9 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r18, %r11;
	@%p1 ld.global.b32 { %r18 }, [ %rd10 + 0 ];
	// end inline asm
	.loc	1 75 32                         // sk09_norm_embed.py:75:32
	mul.f32 	%r59, %r45, %r45;
$L__tmp1:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	fma.rn.f32 	%r60, %r44, %r44, %r59;
	fma.rn.f32 	%r61, %r43, %r43, %r60;
	fma.rn.f32 	%r62, %r42, %r42, %r61;
	fma.rn.f32 	%r63, %r49, %r49, %r62;
	fma.rn.f32 	%r64, %r48, %r48, %r63;
	fma.rn.f32 	%r65, %r47, %r47, %r64;
	fma.rn.f32 	%r66, %r46, %r46, %r65;
$L__tmp2:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	shfl.sync.bfly.b32 	%r67, %r66, 16, 31, -1;
$L__tmp3:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	add.f32 	%r68, %r66, %r67;
$L__tmp4:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	shfl.sync.bfly.b32 	%r69, %r68, 8, 31, -1;
$L__tmp5:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	add.f32 	%r70, %r68, %r69;
$L__tmp6:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	shfl.sync.bfly.b32 	%r71, %r70, 4, 31, -1;
$L__tmp7:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	add.f32 	%r72, %r70, %r71;
$L__tmp8:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	shfl.sync.bfly.b32 	%r73, %r72, 2, 31, -1;
$L__tmp9:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	add.f32 	%r74, %r72, %r73;
$L__tmp10:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	shfl.sync.bfly.b32 	%r75, %r74, 1, 31, -1;
$L__tmp11:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	add.f32 	%r20, %r74, %r75;
$L__tmp12:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	setp.eq.b32 	%p2, %r33, 0;
	shr.u32 	%r76, %r31, 3;
	and.b32 	%r77, %r76, 28;
	mov.b32 	%r78, global_smem;
	add.s32 	%r19, %r78, %r77;
	// begin inline asm
	@%p2 st.shared.b32 [ %r19 + 0 ], %r20;
	// end inline asm
	bar.sync 	0;
	setp.lt.u32 	%p3, %r32, 8;
	shl.b32 	%r79, %r32, 2;
	add.s32 	%r22, %r78, %r79;
	// begin inline asm
	@%p3 ld.shared.b32 %r21, [ %r22 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r80, %r21, 4, 31, -1;
$L__tmp13:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	add.f32 	%r81, %r21, %r80;
$L__tmp14:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	shfl.sync.bfly.b32 	%r82, %r81, 2, 31, -1;
$L__tmp15:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	add.f32 	%r83, %r81, %r82;
$L__tmp16:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	shfl.sync.bfly.b32 	%r84, %r83, 1, 31, -1;
$L__tmp17:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	add.f32 	%r23, %r83, %r84;
$L__tmp18:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	and.b32 	%r85, %r31, 7;
	setp.eq.b32 	%p6, %r85, 0;
	and.pred 	%p4, %p3, %p6;
	// begin inline asm
	@%p4 st.shared.b32 [ %r22 + 0 ], %r23;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r86, [global_smem];
$L__tmp19:
	.loc	1 75 45                         // sk09_norm_embed.py:75:45
	cvt.rn.f32.s32 	%r87, %r35;
	div.full.f32 	%r88, %r86, %r87;
	.loc	1 75 49                         // sk09_norm_embed.py:75:49
	add.f32 	%r89, %r88, 0f358637BD;
	.loc	1 75 21                         // sk09_norm_embed.py:75:21
	rsqrt.approx.ftz.f32 	%r90, %r89;
	.loc	1 75 12                         // sk09_norm_embed.py:75:12
	mul.f32 	%r91, %r90, %r44;
	mul.f32 	%r92, %r90, %r45;
	mul.f32 	%r93, %r90, %r43;
	mul.f32 	%r94, %r90, %r42;
	mul.f32 	%r95, %r90, %r49;
	mul.f32 	%r96, %r90, %r48;
	mul.f32 	%r97, %r90, %r47;
	mul.f32 	%r98, %r90, %r46;
$L__tmp20:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	bar.sync 	0;
$L__tmp21:
	.loc	1 79 27                         // sk09_norm_embed.py:79:27
	mul.lo.s32 	%r99, %r39, %r30;
	.loc	1 79 21                         // sk09_norm_embed.py:79:21
	cvt.s64.s32 	%rd21, %r99;
	add.s64 	%rd22, %rd16, %rd21;
	.loc	1 79 39                         // sk09_norm_embed.py:79:39
	add.s64 	%rd11, %rd22, %rd19;
	.loc	1 73 55                         // sk09_norm_embed.py:73:55
	mov.b32 	{%rs9, %rs10}, %r8;
	cvt.f32.bf16 	%r100, %rs9;
	cvt.f32.bf16 	%r101, %rs10;
	mov.b32 	{%rs11, %rs12}, %r9;
	cvt.f32.bf16 	%r102, %rs11;
	cvt.f32.bf16 	%r103, %rs12;
	.loc	1 73 69                         // sk09_norm_embed.py:73:69
	add.f32 	%r104, %r103, 0f00000000;
	add.f32 	%r105, %r102, 0f00000000;
	add.f32 	%r106, %r101, 0f00000000;
	add.f32 	%r107, %r100, 0f00000000;
	.loc	1 74 12                         // sk09_norm_embed.py:74:12
	mul.f32 	%r108, %r107, %r15;
	mul.f32 	%r109, %r106, %r16;
	mul.f32 	%r110, %r105, %r17;
	mul.f32 	%r111, %r104, %r18;
	.loc	1 75 56                         // sk09_norm_embed.py:75:56
	mul.f32 	%r112, %r111, %r98;
	mul.f32 	%r113, %r110, %r97;
	mul.f32 	%r114, %r109, %r96;
	mul.f32 	%r115, %r108, %r95;
	.loc	1 76 36                         // sk09_norm_embed.py:76:36
	abs.f32 	%r116, %r115;
	abs.f32 	%r117, %r114;
	abs.f32 	%r118, %r113;
	abs.f32 	%r119, %r112;
	.loc	1 73 55                         // sk09_norm_embed.py:73:55
	mov.b32 	{%rs13, %rs14}, %r6;
	cvt.f32.bf16 	%r120, %rs13;
	cvt.f32.bf16 	%r121, %rs14;
	mov.b32 	{%rs15, %rs16}, %r7;
	cvt.f32.bf16 	%r122, %rs15;
	cvt.f32.bf16 	%r123, %rs16;
	.loc	1 73 69                         // sk09_norm_embed.py:73:69
	add.f32 	%r124, %r123, 0f00000000;
	add.f32 	%r125, %r122, 0f00000000;
	add.f32 	%r126, %r121, 0f00000000;
	add.f32 	%r127, %r120, 0f00000000;
	.loc	1 74 12                         // sk09_norm_embed.py:74:12
	mul.f32 	%r128, %r127, %r10;
	mul.f32 	%r129, %r126, %r12;
	mul.f32 	%r130, %r125, %r13;
	mul.f32 	%r131, %r124, %r14;
	.loc	1 75 56                         // sk09_norm_embed.py:75:56
	mul.f32 	%r132, %r131, %r94;
	mul.f32 	%r133, %r130, %r93;
	mul.f32 	%r134, %r129, %r92;
	mul.f32 	%r135, %r128, %r91;
	.loc	1 76 36                         // sk09_norm_embed.py:76:36
	abs.f32 	%r136, %r135;
	abs.f32 	%r137, %r134;
	abs.f32 	%r138, %r133;
	abs.f32 	%r139, %r132;
$L__tmp22:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r140, %r136, %r137;
	max.f32 	%r141, %r140, %r138;
	max.f32 	%r142, %r141, %r139;
	max.f32 	%r143, %r142, %r116;
	max.f32 	%r144, %r143, %r117;
	max.f32 	%r145, %r144, %r118;
	max.f32 	%r146, %r145, %r119;
$L__tmp23:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	shfl.sync.bfly.b32 	%r147, %r146, 16, 31, -1;
$L__tmp24:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r148, %r146, %r147;
$L__tmp25:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	shfl.sync.bfly.b32 	%r149, %r148, 8, 31, -1;
$L__tmp26:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r150, %r148, %r149;
$L__tmp27:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	shfl.sync.bfly.b32 	%r151, %r150, 4, 31, -1;
$L__tmp28:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r152, %r150, %r151;
$L__tmp29:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	shfl.sync.bfly.b32 	%r153, %r152, 2, 31, -1;
$L__tmp30:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r154, %r152, %r153;
$L__tmp31:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	shfl.sync.bfly.b32 	%r155, %r154, 1, 31, -1;
$L__tmp32:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r24, %r154, %r155;
$L__tmp33:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	// begin inline asm
	@%p2 st.shared.b32 [ %r19 + 0 ], %r24;
	// end inline asm
	bar.sync 	0;
	// begin inline asm
	@%p3 ld.shared.b32 %r25, [ %r22 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r156, %r25, 4, 31, -1;
$L__tmp34:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r157, %r25, %r156;
$L__tmp35:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	shfl.sync.bfly.b32 	%r158, %r157, 2, 31, -1;
$L__tmp36:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r159, %r157, %r158;
$L__tmp37:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	shfl.sync.bfly.b32 	%r160, %r159, 1, 31, -1;
$L__tmp38:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r26, %r159, %r160;
$L__tmp39:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	// begin inline asm
	@%p4 st.shared.b32 [ %r22 + 0 ], %r26;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r161, [global_smem];
$L__tmp40:
	.loc	1 76 49                         // sk09_norm_embed.py:76:49
	max.f32 	%r162, %r161, 0f0DA24260;
	mov.b32 	%r163, 0f42FE0000;
	.loc	1 77 22                         // sk09_norm_embed.py:77:22
	div.full.f32 	%r164, %r163, %r162;
	.loc	1 77 14                         // sk09_norm_embed.py:77:14
	mul.f32 	%r165, %r134, %r164;
	mul.f32 	%r166, %r135, %r164;
	mul.f32 	%r167, %r132, %r164;
	mul.f32 	%r168, %r133, %r164;
	mul.f32 	%r169, %r114, %r164;
	mul.f32 	%r170, %r115, %r164;
	mul.f32 	%r171, %r112, %r164;
	mul.f32 	%r172, %r113, %r164;
	.loc	1 78 29                         // sk09_norm_embed.py:78:29
	.loc	1 78 39                         // sk09_norm_embed.py:78:39
	lop3.b32 	%r173, 0x3f000000, %r165, 0x80000000, 0xF8;
	lop3.b32 	%r174, 0x3f000000, %r166, 0x80000000, 0xF8;
	lop3.b32 	%r175, 0x3f000000, %r167, 0x80000000, 0xF8;
	lop3.b32 	%r176, 0x3f000000, %r168, 0x80000000, 0xF8;
	lop3.b32 	%r177, 0x3f000000, %r169, 0x80000000, 0xF8;
	lop3.b32 	%r178, 0x3f000000, %r170, 0x80000000, 0xF8;
	lop3.b32 	%r179, 0x3f000000, %r171, 0x80000000, 0xF8;
	lop3.b32 	%r180, 0x3f000000, %r172, 0x80000000, 0xF8;
	.loc	1 78 14                         // sk09_norm_embed.py:78:14
	fma.rn.f32 	%r181, %r133, %r164, %r176;
	fma.rn.f32 	%r182, %r132, %r164, %r175;
	fma.rn.f32 	%r183, %r135, %r164, %r174;
	fma.rn.f32 	%r184, %r134, %r164, %r173;
	fma.rn.f32 	%r185, %r113, %r164, %r180;
	fma.rn.f32 	%r186, %r112, %r164, %r179;
	fma.rn.f32 	%r187, %r115, %r164, %r178;
	fma.rn.f32 	%r188, %r114, %r164, %r177;
	.loc	1 78 49                         // sk09_norm_embed.py:78:49
	cvt.rzi.s32.f32 	%r189, %r184;
	cvt.rzi.s32.f32 	%r190, %r183;
	cvt.rzi.s32.f32 	%r191, %r182;
	cvt.rzi.s32.f32 	%r192, %r181;
	cvt.rzi.s32.f32 	%r193, %r188;
	cvt.rzi.s32.f32 	%r194, %r187;
	cvt.rzi.s32.f32 	%r195, %r186;
	cvt.rzi.s32.f32 	%r196, %r185;
	.loc	1 79 70                         // sk09_norm_embed.py:79:70
	max.s32 	%r197, %r192, -127;
	max.s32 	%r198, %r191, -127;
	max.s32 	%r199, %r190, -127;
	max.s32 	%r200, %r189, -127;
	max.s32 	%r201, %r196, -127;
	max.s32 	%r202, %r195, -127;
	max.s32 	%r203, %r194, -127;
	max.s32 	%r204, %r193, -127;
	.loc	1 79 77                         // sk09_norm_embed.py:79:77
	min.s32 	%r205, %r200, 127;
	min.s32 	%r206, %r199, 127;
	min.s32 	%r207, %r198, 127;
	min.s32 	%r208, %r197, 127;
	min.s32 	%r209, %r204, 127;
	min.s32 	%r210, %r203, 127;
	min.s32 	%r211, %r202, 127;
	min.s32 	%r212, %r201, 127;
	.loc	1 79 85                         // sk09_norm_embed.py:79:85
	prmt.b32 	%r213, %r208, %r207, 0x3340U;
	prmt.b32 	%r214, %r206, %r205, 0x3340U;
	prmt.b32 	%r27, %r214, %r213, 0x5410U;
	prmt.b32 	%r215, %r212, %r211, 0x3340U;
	prmt.b32 	%r216, %r210, %r209, 0x3340U;
	prmt.b32 	%r28, %r216, %r215, 0x5410U;
	.loc	1 79 45                         // sk09_norm_embed.py:79:45
	// begin inline asm
	@%p1 st.global.v2.b32 [ %rd11 + 0 ], { %r27, %r28 };
	// end inline asm
	.loc	1 80 21                         // sk09_norm_embed.py:80:21
	mad.wide.u32 	%rd12, %r30, 4, %rd17;
	.loc	1 80 34                         // sk09_norm_embed.py:80:34
	mul.f32 	%r29, %r162, 0f3C010204;
	.loc	1 80 26                         // sk09_norm_embed.py:80:26
	or.b32 	%r217, %r33, %r34;
	setp.eq.b32 	%p5, %r217, 0;
	// begin inline asm
	@%p5 st.global.b32 [ %rd12 + 0 ], { %r29 };
	// end inline asm
	.loc	1 80 4                          // sk09_norm_embed.py:80:4
	ret;
$L__tmp41:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk09_norm_embed.py"
	.file	2 "/usr/local/lib/python3.12/dist-packages/triton/language/standard.py"
	.section	.debug_abbrev
	{
.b8 1                                   // Abbreviation Code
.b8 17                                  // DW_TAG_compile_unit
.b8 1                                   // DW_CHILDREN_yes
.b8 37                                  // DW_AT_producer
.b8 8                                   // DW_FORM_string
.b8 19                                  // DW_AT_language
.b8 5                                   // DW_FORM_data2
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 16                                  // DW_AT_stmt_list
.b8 6                                   // DW_FORM_data4
.b8 27                                  // DW_AT_comp_dir
.b8 8                                   // DW_FORM_string
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 2                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 0                                   // DW_CHILDREN_no
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 32                                  // DW_AT_inline
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 3                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 1                                   // DW_CHILDREN_yes
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 4                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 1                                   // DW_CHILDREN_yes
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 11                                  // DW_FORM_data1
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 5                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 0                                   // DW_CHILDREN_no
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 5                                   // DW_FORM_data2
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 6                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 0                                   // DW_CHILDREN_no
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 11                                  // DW_FORM_data1
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 219                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0xd4 DW_TAG_compile_unit
.b8 116                                 // DW_AT_producer
.b8 114
.b8 105
.b8 116
.b8 111
.b8 110
.b8 0
.b8 2                                   // DW_AT_language
.b8 0
.b8 115                                 // DW_AT_name
.b8 107
.b8 48
.b8 57
.b8 95
.b8 110
.b8 111
.b8 114
.b8 109
.b8 95
.b8 101
.b8 109
.b8 98
.b8 101
.b8 100
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 114
.b8 101
.b8 112
.b8 111
.b8 47
.b8 118
.b8 108
.b8 108
.b8 109
.b8 47
.b8 95
.b8 103
.b8 101
.b8 110
.b8 101
.b8 115
.b8 105
.b8 115
.b8 47
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 115
.b8 0
.b8 2                                   // Abbrev [2] 0x48:0x1d DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 57
.b8 95
.b8 114
.b8 109
.b8 115
.b8 110
.b8 111
.b8 114
.b8 109
.b8 95
.b8 113
.b8 117
.b8 97
.b8 110
.b8 116
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x65:0x79 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 72                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x7a:0x32 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp19                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 75                                  // DW_AT_call_line
.b8 28                                  // DW_AT_call_column
.b8 5                                   // Abbrev [5] 0x92:0x19 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp18                          // DW_AT_high_pc
.b8 2                                   // DW_AT_call_file
.b8 37                                  // DW_AT_call_line
.b8 1
.b8 36                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 4                                   // Abbrev [4] 0xac:0x31 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp20                          // DW_AT_low_pc
.b64 $L__tmp40                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 76                                  // DW_AT_call_line
.b8 29                                  // DW_AT_call_column
.b8 6                                   // Abbrev [6] 0xc4:0x18 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp22                          // DW_AT_low_pc
.b64 $L__tmp39                          // DW_AT_high_pc
.b8 2                                   // DW_AT_call_file
.b8 191                                 // DW_AT_call_line
.b8 40                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_QVAR1_0 = _Nativo(
    "sk09_norm_embed/_sk09_rmsnorm_quant_kernel/blk2048",
    _QPTX1_0, "_sk09_rmsnorm_quant_kernel",
    warps=8, shared=32,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    horneado={9: 2048, 10: 1e-06, 11: 0.0},
    div16=[5, 6, 7, 8],
)
# E1=8 lop3, E2=0 mul

_QPTX1_1 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk09_rmsnorm_quant_kernel // -- Begin function _sk09_rmsnorm_quant_kernel
.extern .shared .align 16 .b8 global_smem[];
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
                                        // @_sk09_rmsnorm_quant_kernel
.visible .entry _sk09_rmsnorm_quant_kernel(
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_quant_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_quant_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_quant_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_quant_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_quant_kernel_param_4,
	.param .u32 _sk09_rmsnorm_quant_kernel_param_5,
	.param .u32 _sk09_rmsnorm_quant_kernel_param_6,
	.param .u32 _sk09_rmsnorm_quant_kernel_param_7,
	.param .u32 _sk09_rmsnorm_quant_kernel_param_8,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_quant_kernel_param_9,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_quant_kernel_param_10
)
.reqntid 256
{
	.reg .pred 	%p<23>;
	.reg .b16 	%rs<33>;
	.reg .b32 	%r<369>;
	.reg .b64 	%rd<33>;
	.loc	1 64 0                          // sk09_norm_embed.py:64:0
$L__func_begin0:
	.loc	1 64 0                          // sk09_norm_embed.py:64:0

// %bb.0:                               // %__nv_rsqrtf.exit
	ld.param.b64 	%rd23, [_sk09_rmsnorm_quant_kernel_param_0];
	ld.param.b64 	%rd24, [_sk09_rmsnorm_quant_kernel_param_1];
$L__tmp0:
	.loc	1 69 24                         // sk09_norm_embed.py:69:24
	mov.u32 	%r48, %ctaid.x;
	ld.param.b64 	%rd25, [_sk09_rmsnorm_quant_kernel_param_2];
	.loc	1 70 24                         // sk09_norm_embed.py:70:24
	mov.u32 	%r49, %tid.x;
	and.b32 	%r50, %r49, 255;
	ld.param.b64 	%rd26, [_sk09_rmsnorm_quant_kernel_param_3];
	and.b32 	%r51, %r49, 31;
	ld.param.b64 	%rd27, [_sk09_rmsnorm_quant_kernel_param_4];
	shr.u32 	%r52, %r49, 5;
	ld.param.b32 	%r53, [_sk09_rmsnorm_quant_kernel_param_5];
	shl.b32 	%r54, %r49, 4;
	ld.param.b32 	%r55, [_sk09_rmsnorm_quant_kernel_param_6];
	and.b32 	%r56, %r54, 4080;
	ld.param.b32 	%r57, [_sk09_rmsnorm_quant_kernel_param_7];
	.loc	1 71 18                         // sk09_norm_embed.py:71:18
	setp.lt.s32 	%p1, %r56, %r53;
	ld.param.b32 	%r58, [_sk09_rmsnorm_quant_kernel_param_8];
	.loc	1 72 30                         // sk09_norm_embed.py:72:30
	mul.lo.s32 	%r59, %r55, %r48;
	.loc	1 72 24                         // sk09_norm_embed.py:72:24
	mad.wide.s32 	%rd28, %r59, 2, %rd23;
	.loc	1 72 42                         // sk09_norm_embed.py:72:42
	cvt.u64.u32 	%rd29, %r56;
	mul.wide.u32 	%rd30, %r56, 2;
	add.s64 	%rd1, %rd28, %rd30;
	add.s64 	%rd2, %rd1, 16;
	mov.b32 	%r5, 0;
	.loc	1 72 16                         // sk09_norm_embed.py:72:16
	// begin inline asm
	mov.u32 %r1, %r5;
	mov.u32 %r2, %r5;
	mov.u32 %r3, %r5;
	mov.u32 %r4, %r5;
	@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r6, %r5;
	mov.u32 %r7, %r5;
	mov.u32 %r8, %r5;
	mov.u32 %r9, %r5;
	@%p1 ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd2 + 0 ];
	// end inline asm
	.loc	1 72 73                         // sk09_norm_embed.py:72:73
	mov.b32 	{%rs1, %rs2}, %r2;
	cvt.f32.bf16 	%r60, %rs2;
	cvt.f32.bf16 	%r61, %rs1;
	mov.b32 	{%rs3, %rs4}, %r1;
	cvt.f32.bf16 	%r62, %rs3;
	cvt.f32.bf16 	%r63, %rs4;
	mov.b32 	{%rs5, %rs6}, %r4;
	cvt.f32.bf16 	%r64, %rs6;
	cvt.f32.bf16 	%r65, %rs5;
	mov.b32 	{%rs7, %rs8}, %r3;
	cvt.f32.bf16 	%r66, %rs8;
	cvt.f32.bf16 	%r67, %rs7;
	mov.b32 	{%rs9, %rs10}, %r7;
	cvt.f32.bf16 	%r68, %rs10;
	cvt.f32.bf16 	%r69, %rs9;
	mov.b32 	{%rs11, %rs12}, %r6;
	cvt.f32.bf16 	%r70, %rs12;
	cvt.f32.bf16 	%r71, %rs11;
	mov.b32 	{%rs13, %rs14}, %r9;
	cvt.f32.bf16 	%r72, %rs14;
	cvt.f32.bf16 	%r73, %rs13;
	mov.b32 	{%rs15, %rs16}, %r8;
	cvt.f32.bf16 	%r74, %rs16;
	cvt.f32.bf16 	%r75, %rs15;
	.loc	1 73 24                         // sk09_norm_embed.py:73:24
	add.s64 	%rd3, %rd24, %rd30;
	add.s64 	%rd4, %rd3, 16;
	.loc	1 73 16                         // sk09_norm_embed.py:73:16
	// begin inline asm
	mov.u32 %r10, %r5;
	mov.u32 %r11, %r5;
	mov.u32 %r12, %r5;
	mov.u32 %r13, %r5;
	@%p1 ld.global.v4.b32 { %r10, %r11, %r12, %r13 }, [ %rd3 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r14, %r5;
	mov.u32 %r15, %r5;
	mov.u32 %r16, %r5;
	mov.u32 %r17, %r5;
	@%p1 ld.global.v4.b32 { %r14, %r15, %r16, %r17 }, [ %rd4 + 0 ];
	// end inline asm
	.loc	1 74 40                         // sk09_norm_embed.py:74:40
	mul.lo.s32 	%r76, %r58, %r56;
	shl.b32 	%r77, %r58, 1;
	add.s32 	%r78, %r76, %r77;
	mad.lo.s32 	%r79, %r58, 3, %r76;
	shl.b32 	%r80, %r58, 2;
	add.s32 	%r81, %r76, %r80;
	mad.lo.s32 	%r82, %r58, 5, %r76;
	mad.lo.s32 	%r83, %r58, 6, %r76;
	mad.lo.s32 	%r84, %r58, 7, %r76;
	shl.b32 	%r85, %r58, 3;
	add.s32 	%r86, %r76, %r85;
	mad.lo.s32 	%r87, %r58, 9, %r76;
	mad.lo.s32 	%r88, %r58, 10, %r76;
	mad.lo.s32 	%r89, %r58, 11, %r76;
	mad.lo.s32 	%r90, %r58, 12, %r76;
	mad.lo.s32 	%r91, %r58, 13, %r76;
	mad.lo.s32 	%r92, %r58, 14, %r76;
	mad.lo.s32 	%r93, %r58, 15, %r76;
	.loc	1 74 33                         // sk09_norm_embed.py:74:33
	mad.wide.s32 	%rd5, %r76, 4, %rd25;
	mad.wide.s32 	%rd6, %r58, 4, %rd5;
	mad.wide.s32 	%rd7, %r78, 4, %rd25;
	mad.wide.s32 	%rd8, %r79, 4, %rd25;
	mad.wide.s32 	%rd9, %r81, 4, %rd25;
	mad.wide.s32 	%rd10, %r82, 4, %rd25;
	mad.wide.s32 	%rd11, %r83, 4, %rd25;
	mad.wide.s32 	%rd12, %r84, 4, %rd25;
	mad.wide.s32 	%rd13, %r86, 4, %rd25;
	mad.wide.s32 	%rd14, %r87, 4, %rd25;
	mad.wide.s32 	%rd15, %r88, 4, %rd25;
	mad.wide.s32 	%rd16, %r89, 4, %rd25;
	mad.wide.s32 	%rd17, %r90, 4, %rd25;
	mad.wide.s32 	%rd18, %r91, 4, %rd25;
	mad.wide.s32 	%rd19, %r92, 4, %rd25;
	mad.wide.s32 	%rd20, %r93, 4, %rd25;
	mov.b32 	%r19, 1065353216;
	.loc	1 74 20                         // sk09_norm_embed.py:74:20
	// begin inline asm
	mov.u32 %r18, %r19;
	@%p1 ld.global.b32 { %r18 }, [ %rd5 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r20, %r19;
	@%p1 ld.global.b32 { %r20 }, [ %rd6 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r21, %r19;
	@%p1 ld.global.b32 { %r21 }, [ %rd7 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r22, %r19;
	@%p1 ld.global.b32 { %r22 }, [ %rd8 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r23, %r19;
	@%p1 ld.global.b32 { %r23 }, [ %rd9 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r24, %r19;
	@%p1 ld.global.b32 { %r24 }, [ %rd10 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r25, %r19;
	@%p1 ld.global.b32 { %r25 }, [ %rd11 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r26, %r19;
	@%p1 ld.global.b32 { %r26 }, [ %rd12 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r27, %r19;
	@%p1 ld.global.b32 { %r27 }, [ %rd13 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r28, %r19;
	@%p1 ld.global.b32 { %r28 }, [ %rd14 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r29, %r19;
	@%p1 ld.global.b32 { %r29 }, [ %rd15 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r30, %r19;
	@%p1 ld.global.b32 { %r30 }, [ %rd16 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r31, %r19;
	@%p1 ld.global.b32 { %r31 }, [ %rd17 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r32, %r19;
	@%p1 ld.global.b32 { %r32 }, [ %rd18 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r33, %r19;
	@%p1 ld.global.b32 { %r33 }, [ %rd19 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r34, %r19;
	@%p1 ld.global.b32 { %r34 }, [ %rd20 + 0 ];
	// end inline asm
	.loc	1 75 32                         // sk09_norm_embed.py:75:32
	mul.f32 	%r94, %r63, %r63;
$L__tmp1:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	fma.rn.f32 	%r95, %r62, %r62, %r94;
	fma.rn.f32 	%r96, %r61, %r61, %r95;
	fma.rn.f32 	%r97, %r60, %r60, %r96;
	fma.rn.f32 	%r98, %r67, %r67, %r97;
	fma.rn.f32 	%r99, %r66, %r66, %r98;
	fma.rn.f32 	%r100, %r65, %r65, %r99;
	fma.rn.f32 	%r101, %r64, %r64, %r100;
	fma.rn.f32 	%r102, %r71, %r71, %r101;
	fma.rn.f32 	%r103, %r70, %r70, %r102;
	fma.rn.f32 	%r104, %r69, %r69, %r103;
	fma.rn.f32 	%r105, %r68, %r68, %r104;
	fma.rn.f32 	%r106, %r75, %r75, %r105;
	fma.rn.f32 	%r107, %r74, %r74, %r106;
	fma.rn.f32 	%r108, %r73, %r73, %r107;
	fma.rn.f32 	%r109, %r72, %r72, %r108;
$L__tmp2:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	shfl.sync.bfly.b32 	%r110, %r109, 16, 31, -1;
$L__tmp3:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	add.f32 	%r111, %r109, %r110;
$L__tmp4:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	shfl.sync.bfly.b32 	%r112, %r111, 8, 31, -1;
$L__tmp5:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	add.f32 	%r113, %r111, %r112;
$L__tmp6:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	shfl.sync.bfly.b32 	%r114, %r113, 4, 31, -1;
$L__tmp7:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	add.f32 	%r115, %r113, %r114;
$L__tmp8:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	shfl.sync.bfly.b32 	%r116, %r115, 2, 31, -1;
$L__tmp9:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	add.f32 	%r117, %r115, %r116;
$L__tmp10:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	shfl.sync.bfly.b32 	%r118, %r117, 1, 31, -1;
$L__tmp11:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	add.f32 	%r36, %r117, %r118;
$L__tmp12:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	setp.eq.b32 	%p2, %r51, 0;
	shr.u32 	%r119, %r49, 3;
	and.b32 	%r120, %r119, 28;
	mov.b32 	%r121, global_smem;
	add.s32 	%r35, %r121, %r120;
	// begin inline asm
	@%p2 st.shared.b32 [ %r35 + 0 ], %r36;
	// end inline asm
	bar.sync 	0;
	setp.lt.u32 	%p3, %r50, 8;
	shl.b32 	%r122, %r50, 2;
	add.s32 	%r38, %r121, %r122;
	// begin inline asm
	@%p3 ld.shared.b32 %r37, [ %r38 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r123, %r37, 4, 31, -1;
$L__tmp13:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	add.f32 	%r124, %r37, %r123;
$L__tmp14:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	shfl.sync.bfly.b32 	%r125, %r124, 2, 31, -1;
$L__tmp15:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	add.f32 	%r126, %r124, %r125;
$L__tmp16:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	shfl.sync.bfly.b32 	%r127, %r126, 1, 31, -1;
$L__tmp17:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	add.f32 	%r39, %r126, %r127;
$L__tmp18:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	and.b32 	%r128, %r49, 7;
	setp.eq.b32 	%p6, %r128, 0;
	and.pred 	%p4, %p3, %p6;
	// begin inline asm
	@%p4 st.shared.b32 [ %r38 + 0 ], %r39;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r129, [global_smem];
$L__tmp19:
	.loc	1 75 45                         // sk09_norm_embed.py:75:45
	cvt.rn.f32.s32 	%r130, %r53;
	div.full.f32 	%r131, %r129, %r130;
	.loc	1 75 49                         // sk09_norm_embed.py:75:49
	add.f32 	%r132, %r131, 0f358637BD;
	.loc	1 75 21                         // sk09_norm_embed.py:75:21
	rsqrt.approx.ftz.f32 	%r133, %r132;
	.loc	1 75 12                         // sk09_norm_embed.py:75:12
	mul.f32 	%r134, %r133, %r62;
	mul.f32 	%r135, %r133, %r63;
	mul.f32 	%r136, %r133, %r61;
	mul.f32 	%r137, %r133, %r60;
	mul.f32 	%r138, %r133, %r67;
	mul.f32 	%r139, %r133, %r66;
	mul.f32 	%r140, %r133, %r65;
	mul.f32 	%r141, %r133, %r64;
	mul.f32 	%r142, %r133, %r71;
	mul.f32 	%r143, %r133, %r70;
	mul.f32 	%r144, %r133, %r69;
	mul.f32 	%r145, %r133, %r68;
	mul.f32 	%r146, %r133, %r75;
	mul.f32 	%r147, %r133, %r74;
	mul.f32 	%r148, %r133, %r73;
	mul.f32 	%r149, %r133, %r72;
$L__tmp20:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	bar.sync 	0;
$L__tmp21:
	.loc	1 79 27                         // sk09_norm_embed.py:79:27
	mul.lo.s32 	%r150, %r57, %r48;
	.loc	1 79 21                         // sk09_norm_embed.py:79:21
	cvt.s64.s32 	%rd31, %r150;
	add.s64 	%rd32, %rd26, %rd31;
	.loc	1 79 39                         // sk09_norm_embed.py:79:39
	add.s64 	%rd21, %rd32, %rd29;
	.loc	1 73 55                         // sk09_norm_embed.py:73:55
	mov.b32 	{%rs17, %rs18}, %r16;
	cvt.f32.bf16 	%r151, %rs17;
	cvt.f32.bf16 	%r152, %rs18;
	mov.b32 	{%rs19, %rs20}, %r17;
	cvt.f32.bf16 	%r153, %rs19;
	cvt.f32.bf16 	%r154, %rs20;
	.loc	1 73 69                         // sk09_norm_embed.py:73:69
	add.f32 	%r155, %r154, 0f00000000;
	add.f32 	%r156, %r153, 0f00000000;
	add.f32 	%r157, %r152, 0f00000000;
	add.f32 	%r158, %r151, 0f00000000;
	.loc	1 74 12                         // sk09_norm_embed.py:74:12
	mul.f32 	%r159, %r158, %r31;
	mul.f32 	%r160, %r157, %r32;
	mul.f32 	%r161, %r156, %r33;
	mul.f32 	%r162, %r155, %r34;
	.loc	1 75 56                         // sk09_norm_embed.py:75:56
	mul.f32 	%r163, %r162, %r149;
	mul.f32 	%r164, %r161, %r148;
	mul.f32 	%r165, %r160, %r147;
	mul.f32 	%r166, %r159, %r146;
	.loc	1 76 36                         // sk09_norm_embed.py:76:36
	abs.f32 	%r167, %r166;
	abs.f32 	%r168, %r165;
	abs.f32 	%r169, %r164;
	abs.f32 	%r170, %r163;
	.loc	1 73 55                         // sk09_norm_embed.py:73:55
	mov.b32 	{%rs21, %rs22}, %r14;
	cvt.f32.bf16 	%r171, %rs21;
	cvt.f32.bf16 	%r172, %rs22;
	mov.b32 	{%rs23, %rs24}, %r15;
	cvt.f32.bf16 	%r173, %rs23;
	cvt.f32.bf16 	%r174, %rs24;
	.loc	1 73 69                         // sk09_norm_embed.py:73:69
	add.f32 	%r175, %r174, 0f00000000;
	add.f32 	%r176, %r173, 0f00000000;
	add.f32 	%r177, %r172, 0f00000000;
	add.f32 	%r178, %r171, 0f00000000;
	.loc	1 74 12                         // sk09_norm_embed.py:74:12
	mul.f32 	%r179, %r178, %r27;
	mul.f32 	%r180, %r177, %r28;
	mul.f32 	%r181, %r176, %r29;
	mul.f32 	%r182, %r175, %r30;
	.loc	1 75 56                         // sk09_norm_embed.py:75:56
	mul.f32 	%r183, %r182, %r145;
	mul.f32 	%r184, %r181, %r144;
	mul.f32 	%r185, %r180, %r143;
	mul.f32 	%r186, %r179, %r142;
	.loc	1 76 36                         // sk09_norm_embed.py:76:36
	abs.f32 	%r187, %r186;
	abs.f32 	%r188, %r185;
	abs.f32 	%r189, %r184;
	abs.f32 	%r190, %r183;
	.loc	1 73 55                         // sk09_norm_embed.py:73:55
	mov.b32 	{%rs25, %rs26}, %r12;
	cvt.f32.bf16 	%r191, %rs25;
	cvt.f32.bf16 	%r192, %rs26;
	mov.b32 	{%rs27, %rs28}, %r13;
	cvt.f32.bf16 	%r193, %rs27;
	cvt.f32.bf16 	%r194, %rs28;
	.loc	1 73 69                         // sk09_norm_embed.py:73:69
	add.f32 	%r195, %r194, 0f00000000;
	add.f32 	%r196, %r193, 0f00000000;
	add.f32 	%r197, %r192, 0f00000000;
	add.f32 	%r198, %r191, 0f00000000;
	.loc	1 74 12                         // sk09_norm_embed.py:74:12
	mul.f32 	%r199, %r198, %r23;
	mul.f32 	%r200, %r197, %r24;
	mul.f32 	%r201, %r196, %r25;
	mul.f32 	%r202, %r195, %r26;
	.loc	1 75 56                         // sk09_norm_embed.py:75:56
	mul.f32 	%r203, %r202, %r141;
	mul.f32 	%r204, %r201, %r140;
	mul.f32 	%r205, %r200, %r139;
	mul.f32 	%r206, %r199, %r138;
	.loc	1 76 36                         // sk09_norm_embed.py:76:36
	abs.f32 	%r207, %r206;
	abs.f32 	%r208, %r205;
	abs.f32 	%r209, %r204;
	abs.f32 	%r210, %r203;
	.loc	1 73 55                         // sk09_norm_embed.py:73:55
	mov.b32 	{%rs29, %rs30}, %r10;
	cvt.f32.bf16 	%r211, %rs29;
	cvt.f32.bf16 	%r212, %rs30;
	mov.b32 	{%rs31, %rs32}, %r11;
	cvt.f32.bf16 	%r213, %rs31;
	cvt.f32.bf16 	%r214, %rs32;
	.loc	1 73 69                         // sk09_norm_embed.py:73:69
	add.f32 	%r215, %r214, 0f00000000;
	add.f32 	%r216, %r213, 0f00000000;
	add.f32 	%r217, %r212, 0f00000000;
	add.f32 	%r218, %r211, 0f00000000;
	.loc	1 74 12                         // sk09_norm_embed.py:74:12
	mul.f32 	%r219, %r218, %r18;
	mul.f32 	%r220, %r217, %r20;
	mul.f32 	%r221, %r216, %r21;
	mul.f32 	%r222, %r215, %r22;
	.loc	1 75 56                         // sk09_norm_embed.py:75:56
	mul.f32 	%r223, %r222, %r137;
	mul.f32 	%r224, %r221, %r136;
	mul.f32 	%r225, %r220, %r135;
	mul.f32 	%r226, %r219, %r134;
	.loc	1 76 36                         // sk09_norm_embed.py:76:36
	abs.f32 	%r227, %r226;
	abs.f32 	%r228, %r225;
	abs.f32 	%r229, %r224;
	abs.f32 	%r230, %r223;
$L__tmp22:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r231, %r227, %r228;
	max.f32 	%r232, %r231, %r229;
	max.f32 	%r233, %r232, %r230;
	max.f32 	%r234, %r233, %r207;
	max.f32 	%r235, %r234, %r208;
	max.f32 	%r236, %r235, %r209;
	max.f32 	%r237, %r236, %r210;
	max.f32 	%r238, %r237, %r187;
	max.f32 	%r239, %r238, %r188;
	max.f32 	%r240, %r239, %r189;
	max.f32 	%r241, %r240, %r190;
	max.f32 	%r242, %r241, %r167;
	max.f32 	%r243, %r242, %r168;
	max.f32 	%r244, %r243, %r169;
	max.f32 	%r245, %r244, %r170;
$L__tmp23:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	shfl.sync.bfly.b32 	%r246, %r245, 16, 31, -1;
$L__tmp24:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r247, %r245, %r246;
$L__tmp25:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	shfl.sync.bfly.b32 	%r248, %r247, 8, 31, -1;
$L__tmp26:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r249, %r247, %r248;
$L__tmp27:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	shfl.sync.bfly.b32 	%r250, %r249, 4, 31, -1;
$L__tmp28:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r251, %r249, %r250;
$L__tmp29:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	shfl.sync.bfly.b32 	%r252, %r251, 2, 31, -1;
$L__tmp30:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r253, %r251, %r252;
$L__tmp31:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	shfl.sync.bfly.b32 	%r254, %r253, 1, 31, -1;
$L__tmp32:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r40, %r253, %r254;
$L__tmp33:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	// begin inline asm
	@%p2 st.shared.b32 [ %r35 + 0 ], %r40;
	// end inline asm
	bar.sync 	0;
	// begin inline asm
	@%p3 ld.shared.b32 %r41, [ %r38 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r255, %r41, 4, 31, -1;
$L__tmp34:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r256, %r41, %r255;
$L__tmp35:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	shfl.sync.bfly.b32 	%r257, %r256, 2, 31, -1;
$L__tmp36:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r258, %r256, %r257;
$L__tmp37:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	shfl.sync.bfly.b32 	%r259, %r258, 1, 31, -1;
$L__tmp38:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r42, %r258, %r259;
$L__tmp39:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	// begin inline asm
	@%p4 st.shared.b32 [ %r38 + 0 ], %r42;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r260, [global_smem];
$L__tmp40:
	.loc	1 76 49                         // sk09_norm_embed.py:76:49
	max.f32 	%r261, %r260, 0f0DA24260;
	mov.b32 	%r262, 0f42FE0000;
	.loc	1 77 22                         // sk09_norm_embed.py:77:22
	div.full.f32 	%r263, %r262, %r261;
	.loc	1 77 14                         // sk09_norm_embed.py:77:14
	mul.f32 	%r264, %r225, %r263;
	mul.f32 	%r265, %r226, %r263;
	mul.f32 	%r266, %r223, %r263;
	mul.f32 	%r267, %r224, %r263;
	mul.f32 	%r268, %r205, %r263;
	mul.f32 	%r269, %r206, %r263;
	mul.f32 	%r270, %r203, %r263;
	mul.f32 	%r271, %r204, %r263;
	mul.f32 	%r272, %r185, %r263;
	mul.f32 	%r273, %r186, %r263;
	mul.f32 	%r274, %r183, %r263;
	mul.f32 	%r275, %r184, %r263;
	mul.f32 	%r276, %r165, %r263;
	mul.f32 	%r277, %r166, %r263;
	mul.f32 	%r278, %r163, %r263;
	mul.f32 	%r279, %r164, %r263;
	.loc	1 78 29                         // sk09_norm_embed.py:78:29
	.loc	1 78 39                         // sk09_norm_embed.py:78:39
	lop3.b32 	%r280, 0x3f000000, %r264, 0x80000000, 0xF8;
	lop3.b32 	%r281, 0x3f000000, %r265, 0x80000000, 0xF8;
	lop3.b32 	%r282, 0x3f000000, %r266, 0x80000000, 0xF8;
	lop3.b32 	%r283, 0x3f000000, %r267, 0x80000000, 0xF8;
	lop3.b32 	%r284, 0x3f000000, %r268, 0x80000000, 0xF8;
	lop3.b32 	%r285, 0x3f000000, %r269, 0x80000000, 0xF8;
	lop3.b32 	%r286, 0x3f000000, %r270, 0x80000000, 0xF8;
	lop3.b32 	%r287, 0x3f000000, %r271, 0x80000000, 0xF8;
	lop3.b32 	%r288, 0x3f000000, %r272, 0x80000000, 0xF8;
	lop3.b32 	%r289, 0x3f000000, %r273, 0x80000000, 0xF8;
	lop3.b32 	%r290, 0x3f000000, %r274, 0x80000000, 0xF8;
	lop3.b32 	%r291, 0x3f000000, %r275, 0x80000000, 0xF8;
	lop3.b32 	%r292, 0x3f000000, %r276, 0x80000000, 0xF8;
	lop3.b32 	%r293, 0x3f000000, %r277, 0x80000000, 0xF8;
	lop3.b32 	%r294, 0x3f000000, %r278, 0x80000000, 0xF8;
	lop3.b32 	%r295, 0x3f000000, %r279, 0x80000000, 0xF8;
	.loc	1 78 14                         // sk09_norm_embed.py:78:14
	fma.rn.f32 	%r296, %r224, %r263, %r283;
	fma.rn.f32 	%r297, %r223, %r263, %r282;
	fma.rn.f32 	%r298, %r226, %r263, %r281;
	fma.rn.f32 	%r299, %r225, %r263, %r280;
	fma.rn.f32 	%r300, %r204, %r263, %r287;
	fma.rn.f32 	%r301, %r203, %r263, %r286;
	fma.rn.f32 	%r302, %r206, %r263, %r285;
	fma.rn.f32 	%r303, %r205, %r263, %r284;
	fma.rn.f32 	%r304, %r184, %r263, %r291;
	fma.rn.f32 	%r305, %r183, %r263, %r290;
	fma.rn.f32 	%r306, %r186, %r263, %r289;
	fma.rn.f32 	%r307, %r185, %r263, %r288;
	fma.rn.f32 	%r308, %r164, %r263, %r295;
	fma.rn.f32 	%r309, %r163, %r263, %r294;
	fma.rn.f32 	%r310, %r166, %r263, %r293;
	fma.rn.f32 	%r311, %r165, %r263, %r292;
	.loc	1 78 49                         // sk09_norm_embed.py:78:49
	cvt.rzi.s32.f32 	%r312, %r299;
	cvt.rzi.s32.f32 	%r313, %r298;
	cvt.rzi.s32.f32 	%r314, %r297;
	cvt.rzi.s32.f32 	%r315, %r296;
	cvt.rzi.s32.f32 	%r316, %r303;
	cvt.rzi.s32.f32 	%r317, %r302;
	cvt.rzi.s32.f32 	%r318, %r301;
	cvt.rzi.s32.f32 	%r319, %r300;
	cvt.rzi.s32.f32 	%r320, %r307;
	cvt.rzi.s32.f32 	%r321, %r306;
	cvt.rzi.s32.f32 	%r322, %r305;
	cvt.rzi.s32.f32 	%r323, %r304;
	cvt.rzi.s32.f32 	%r324, %r311;
	cvt.rzi.s32.f32 	%r325, %r310;
	cvt.rzi.s32.f32 	%r326, %r309;
	cvt.rzi.s32.f32 	%r327, %r308;
	.loc	1 79 70                         // sk09_norm_embed.py:79:70
	max.s32 	%r328, %r315, -127;
	max.s32 	%r329, %r314, -127;
	max.s32 	%r330, %r313, -127;
	max.s32 	%r331, %r312, -127;
	max.s32 	%r332, %r319, -127;
	max.s32 	%r333, %r318, -127;
	max.s32 	%r334, %r317, -127;
	max.s32 	%r335, %r316, -127;
	max.s32 	%r336, %r323, -127;
	max.s32 	%r337, %r322, -127;
	max.s32 	%r338, %r321, -127;
	max.s32 	%r339, %r320, -127;
	max.s32 	%r340, %r327, -127;
	max.s32 	%r341, %r326, -127;
	max.s32 	%r342, %r325, -127;
	max.s32 	%r343, %r324, -127;
	.loc	1 79 77                         // sk09_norm_embed.py:79:77
	min.s32 	%r344, %r331, 127;
	min.s32 	%r345, %r330, 127;
	min.s32 	%r346, %r329, 127;
	min.s32 	%r347, %r328, 127;
	min.s32 	%r348, %r335, 127;
	min.s32 	%r349, %r334, 127;
	min.s32 	%r350, %r333, 127;
	min.s32 	%r351, %r332, 127;
	min.s32 	%r352, %r339, 127;
	min.s32 	%r353, %r338, 127;
	min.s32 	%r354, %r337, 127;
	min.s32 	%r355, %r336, 127;
	min.s32 	%r356, %r343, 127;
	min.s32 	%r357, %r342, 127;
	min.s32 	%r358, %r341, 127;
	min.s32 	%r359, %r340, 127;
	.loc	1 79 85                         // sk09_norm_embed.py:79:85
	prmt.b32 	%r360, %r347, %r346, 0x3340U;
	prmt.b32 	%r361, %r345, %r344, 0x3340U;
	prmt.b32 	%r43, %r361, %r360, 0x5410U;
	prmt.b32 	%r362, %r351, %r350, 0x3340U;
	prmt.b32 	%r363, %r349, %r348, 0x3340U;
	prmt.b32 	%r44, %r363, %r362, 0x5410U;
	prmt.b32 	%r364, %r355, %r354, 0x3340U;
	prmt.b32 	%r365, %r353, %r352, 0x3340U;
	prmt.b32 	%r45, %r365, %r364, 0x5410U;
	prmt.b32 	%r366, %r359, %r358, 0x3340U;
	prmt.b32 	%r367, %r357, %r356, 0x3340U;
	prmt.b32 	%r46, %r367, %r366, 0x5410U;
	.loc	1 79 45                         // sk09_norm_embed.py:79:45
	// begin inline asm
	@%p1 st.global.v4.b32 [ %rd21 + 0 ], { %r43, %r44, %r45, %r46 };
	// end inline asm
	.loc	1 80 21                         // sk09_norm_embed.py:80:21
	mad.wide.u32 	%rd22, %r48, 4, %rd27;
	.loc	1 80 34                         // sk09_norm_embed.py:80:34
	mul.f32 	%r47, %r261, 0f3C010204;
	.loc	1 80 26                         // sk09_norm_embed.py:80:26
	or.b32 	%r368, %r51, %r52;
	setp.eq.b32 	%p5, %r368, 0;
	// begin inline asm
	@%p5 st.global.b32 [ %rd22 + 0 ], { %r47 };
	// end inline asm
	.loc	1 80 4                          // sk09_norm_embed.py:80:4
	ret;
$L__tmp41:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk09_norm_embed.py"
	.file	2 "/usr/local/lib/python3.12/dist-packages/triton/language/standard.py"
	.section	.debug_abbrev
	{
.b8 1                                   // Abbreviation Code
.b8 17                                  // DW_TAG_compile_unit
.b8 1                                   // DW_CHILDREN_yes
.b8 37                                  // DW_AT_producer
.b8 8                                   // DW_FORM_string
.b8 19                                  // DW_AT_language
.b8 5                                   // DW_FORM_data2
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 16                                  // DW_AT_stmt_list
.b8 6                                   // DW_FORM_data4
.b8 27                                  // DW_AT_comp_dir
.b8 8                                   // DW_FORM_string
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 2                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 0                                   // DW_CHILDREN_no
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 32                                  // DW_AT_inline
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 3                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 1                                   // DW_CHILDREN_yes
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 4                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 1                                   // DW_CHILDREN_yes
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 11                                  // DW_FORM_data1
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 5                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 0                                   // DW_CHILDREN_no
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 5                                   // DW_FORM_data2
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 6                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 0                                   // DW_CHILDREN_no
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 11                                  // DW_FORM_data1
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 219                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0xd4 DW_TAG_compile_unit
.b8 116                                 // DW_AT_producer
.b8 114
.b8 105
.b8 116
.b8 111
.b8 110
.b8 0
.b8 2                                   // DW_AT_language
.b8 0
.b8 115                                 // DW_AT_name
.b8 107
.b8 48
.b8 57
.b8 95
.b8 110
.b8 111
.b8 114
.b8 109
.b8 95
.b8 101
.b8 109
.b8 98
.b8 101
.b8 100
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 114
.b8 101
.b8 112
.b8 111
.b8 47
.b8 118
.b8 108
.b8 108
.b8 109
.b8 47
.b8 95
.b8 103
.b8 101
.b8 110
.b8 101
.b8 115
.b8 105
.b8 115
.b8 47
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 115
.b8 0
.b8 2                                   // Abbrev [2] 0x48:0x1d DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 57
.b8 95
.b8 114
.b8 109
.b8 115
.b8 110
.b8 111
.b8 114
.b8 109
.b8 95
.b8 113
.b8 117
.b8 97
.b8 110
.b8 116
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x65:0x79 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 72                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x7a:0x32 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp19                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 75                                  // DW_AT_call_line
.b8 28                                  // DW_AT_call_column
.b8 5                                   // Abbrev [5] 0x92:0x19 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp18                          // DW_AT_high_pc
.b8 2                                   // DW_AT_call_file
.b8 37                                  // DW_AT_call_line
.b8 1
.b8 36                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 4                                   // Abbrev [4] 0xac:0x31 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp20                          // DW_AT_low_pc
.b64 $L__tmp40                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 76                                  // DW_AT_call_line
.b8 29                                  // DW_AT_call_column
.b8 6                                   // Abbrev [6] 0xc4:0x18 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp22                          // DW_AT_low_pc
.b64 $L__tmp39                          // DW_AT_high_pc
.b8 2                                   // DW_AT_call_file
.b8 191                                 // DW_AT_call_line
.b8 40                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_QVAR1_1 = _Nativo(
    "sk09_norm_embed/_sk09_rmsnorm_quant_kernel/blk4096",
    _QPTX1_1, "_sk09_rmsnorm_quant_kernel",
    warps=8, shared=32,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    horneado={9: 4096, 10: 1e-06, 11: 0.0},
    div16=[5, 6, 7, 8],
)
# E1=16 lop3, E2=0 mul

_QPTX1_2 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk09_rmsnorm_quant_kernel // -- Begin function _sk09_rmsnorm_quant_kernel
.extern .shared .align 16 .b8 global_smem[];
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
                                        // @_sk09_rmsnorm_quant_kernel
.visible .entry _sk09_rmsnorm_quant_kernel(
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_quant_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_quant_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_quant_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_quant_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_quant_kernel_param_4,
	.param .u32 _sk09_rmsnorm_quant_kernel_param_5,
	.param .u32 _sk09_rmsnorm_quant_kernel_param_6,
	.param .u32 _sk09_rmsnorm_quant_kernel_param_7,
	.param .u32 _sk09_rmsnorm_quant_kernel_param_8,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_quant_kernel_param_9,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_quant_kernel_param_10
)
.reqntid 256
{
	.reg .pred 	%p<40>;
	.reg .b16 	%rs<65>;
	.reg .b32 	%r<671>;
	.reg .b64 	%rd<54>;
	.loc	1 64 0                          // sk09_norm_embed.py:64:0
$L__func_begin0:
	.loc	1 64 0                          // sk09_norm_embed.py:64:0

// %bb.0:                               // %__nv_rsqrtf.exit
	ld.param.b64 	%rd44, [_sk09_rmsnorm_quant_kernel_param_0];
	ld.param.b64 	%rd45, [_sk09_rmsnorm_quant_kernel_param_1];
$L__tmp0:
	.loc	1 69 24                         // sk09_norm_embed.py:69:24
	mov.u32 	%r84, %ctaid.x;
	ld.param.b64 	%rd46, [_sk09_rmsnorm_quant_kernel_param_2];
	.loc	1 70 24                         // sk09_norm_embed.py:70:24
	mov.u32 	%r85, %tid.x;
	and.b32 	%r86, %r85, 255;
	ld.param.b64 	%rd47, [_sk09_rmsnorm_quant_kernel_param_3];
	and.b32 	%r87, %r85, 31;
	ld.param.b64 	%rd48, [_sk09_rmsnorm_quant_kernel_param_4];
	shr.u32 	%r88, %r85, 5;
	ld.param.b32 	%r89, [_sk09_rmsnorm_quant_kernel_param_5];
	shl.b32 	%r90, %r85, 4;
	ld.param.b32 	%r91, [_sk09_rmsnorm_quant_kernel_param_6];
	and.b32 	%r92, %r90, 4080;
	ld.param.b32 	%r93, [_sk09_rmsnorm_quant_kernel_param_7];
	or.b32 	%r94, %r92, 4096;
	ld.param.b32 	%r95, [_sk09_rmsnorm_quant_kernel_param_8];
	.loc	1 71 18                         // sk09_norm_embed.py:71:18
	setp.lt.s32 	%p1, %r92, %r89;
	setp.lt.s32 	%p2, %r94, %r89;
	.loc	1 72 30                         // sk09_norm_embed.py:72:30
	mul.lo.s32 	%r96, %r91, %r84;
	.loc	1 72 24                         // sk09_norm_embed.py:72:24
	mad.wide.s32 	%rd49, %r96, 2, %rd44;
	.loc	1 72 42                         // sk09_norm_embed.py:72:42
	cvt.u64.u32 	%rd50, %r92;
	mul.wide.u32 	%rd51, %r92, 2;
	add.s64 	%rd1, %rd49, %rd51;
	add.s64 	%rd2, %rd1, 16;
	add.s64 	%rd3, %rd1, 8192;
	add.s64 	%rd4, %rd1, 8208;
	mov.b32 	%r5, 0;
	.loc	1 72 16                         // sk09_norm_embed.py:72:16
	// begin inline asm
	mov.u32 %r1, %r5;
	mov.u32 %r2, %r5;
	mov.u32 %r3, %r5;
	mov.u32 %r4, %r5;
	@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r6, %r5;
	mov.u32 %r7, %r5;
	mov.u32 %r8, %r5;
	mov.u32 %r9, %r5;
	@%p1 ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd2 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r10, %r5;
	mov.u32 %r11, %r5;
	mov.u32 %r12, %r5;
	mov.u32 %r13, %r5;
	@%p2 ld.global.v4.b32 { %r10, %r11, %r12, %r13 }, [ %rd3 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r14, %r5;
	mov.u32 %r15, %r5;
	mov.u32 %r16, %r5;
	mov.u32 %r17, %r5;
	@%p2 ld.global.v4.b32 { %r14, %r15, %r16, %r17 }, [ %rd4 + 0 ];
	// end inline asm
	.loc	1 72 73                         // sk09_norm_embed.py:72:73
	mov.b32 	{%rs1, %rs2}, %r2;
	cvt.f32.bf16 	%r97, %rs2;
	cvt.f32.bf16 	%r98, %rs1;
	mov.b32 	{%rs3, %rs4}, %r1;
	cvt.f32.bf16 	%r99, %rs3;
	cvt.f32.bf16 	%r100, %rs4;
	mov.b32 	{%rs5, %rs6}, %r4;
	cvt.f32.bf16 	%r101, %rs6;
	cvt.f32.bf16 	%r102, %rs5;
	mov.b32 	{%rs7, %rs8}, %r3;
	cvt.f32.bf16 	%r103, %rs8;
	cvt.f32.bf16 	%r104, %rs7;
	mov.b32 	{%rs9, %rs10}, %r7;
	cvt.f32.bf16 	%r105, %rs10;
	cvt.f32.bf16 	%r106, %rs9;
	mov.b32 	{%rs11, %rs12}, %r6;
	cvt.f32.bf16 	%r107, %rs12;
	cvt.f32.bf16 	%r108, %rs11;
	mov.b32 	{%rs13, %rs14}, %r9;
	cvt.f32.bf16 	%r109, %rs14;
	cvt.f32.bf16 	%r110, %rs13;
	mov.b32 	{%rs15, %rs16}, %r8;
	cvt.f32.bf16 	%r111, %rs16;
	cvt.f32.bf16 	%r112, %rs15;
	mov.b32 	{%rs17, %rs18}, %r11;
	cvt.f32.bf16 	%r113, %rs18;
	cvt.f32.bf16 	%r114, %rs17;
	mov.b32 	{%rs19, %rs20}, %r10;
	cvt.f32.bf16 	%r115, %rs20;
	cvt.f32.bf16 	%r116, %rs19;
	mov.b32 	{%rs21, %rs22}, %r13;
	cvt.f32.bf16 	%r117, %rs22;
	cvt.f32.bf16 	%r118, %rs21;
	mov.b32 	{%rs23, %rs24}, %r12;
	cvt.f32.bf16 	%r119, %rs24;
	cvt.f32.bf16 	%r120, %rs23;
	mov.b32 	{%rs25, %rs26}, %r15;
	cvt.f32.bf16 	%r121, %rs26;
	cvt.f32.bf16 	%r122, %rs25;
	mov.b32 	{%rs27, %rs28}, %r14;
	cvt.f32.bf16 	%r123, %rs28;
	cvt.f32.bf16 	%r124, %rs27;
	mov.b32 	{%rs29, %rs30}, %r17;
	cvt.f32.bf16 	%r125, %rs30;
	cvt.f32.bf16 	%r126, %rs29;
	mov.b32 	{%rs31, %rs32}, %r16;
	cvt.f32.bf16 	%r127, %rs32;
	cvt.f32.bf16 	%r128, %rs31;
	.loc	1 73 24                         // sk09_norm_embed.py:73:24
	add.s64 	%rd5, %rd45, %rd51;
	add.s64 	%rd6, %rd5, 16;
	add.s64 	%rd7, %rd5, 8192;
	add.s64 	%rd8, %rd5, 8208;
	.loc	1 73 16                         // sk09_norm_embed.py:73:16
	// begin inline asm
	mov.u32 %r18, %r5;
	mov.u32 %r19, %r5;
	mov.u32 %r20, %r5;
	mov.u32 %r21, %r5;
	@%p1 ld.global.v4.b32 { %r18, %r19, %r20, %r21 }, [ %rd5 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r22, %r5;
	mov.u32 %r23, %r5;
	mov.u32 %r24, %r5;
	mov.u32 %r25, %r5;
	@%p1 ld.global.v4.b32 { %r22, %r23, %r24, %r25 }, [ %rd6 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r26, %r5;
	mov.u32 %r27, %r5;
	mov.u32 %r28, %r5;
	mov.u32 %r29, %r5;
	@%p2 ld.global.v4.b32 { %r26, %r27, %r28, %r29 }, [ %rd7 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r30, %r5;
	mov.u32 %r31, %r5;
	mov.u32 %r32, %r5;
	mov.u32 %r33, %r5;
	@%p2 ld.global.v4.b32 { %r30, %r31, %r32, %r33 }, [ %rd8 + 0 ];
	// end inline asm
	.loc	1 74 40                         // sk09_norm_embed.py:74:40
	mul.lo.s32 	%r129, %r95, %r92;
	shl.b32 	%r130, %r95, 1;
	add.s32 	%r131, %r129, %r130;
	mad.lo.s32 	%r132, %r95, 3, %r129;
	shl.b32 	%r133, %r95, 2;
	add.s32 	%r134, %r129, %r133;
	mad.lo.s32 	%r135, %r95, 5, %r129;
	mad.lo.s32 	%r136, %r95, 6, %r129;
	mad.lo.s32 	%r137, %r95, 7, %r129;
	shl.b32 	%r138, %r95, 3;
	add.s32 	%r139, %r129, %r138;
	mad.lo.s32 	%r140, %r95, 9, %r129;
	mad.lo.s32 	%r141, %r95, 10, %r129;
	mad.lo.s32 	%r142, %r95, 11, %r129;
	mad.lo.s32 	%r143, %r95, 12, %r129;
	mad.lo.s32 	%r144, %r95, 13, %r129;
	mad.lo.s32 	%r145, %r95, 14, %r129;
	mad.lo.s32 	%r146, %r95, 15, %r129;
	shl.b32 	%r147, %r95, 12;
	add.s32 	%r148, %r129, %r147;
	mad.lo.s32 	%r149, %r95, 4097, %r129;
	mad.lo.s32 	%r150, %r95, 4098, %r129;
	mad.lo.s32 	%r151, %r95, 4099, %r129;
	mad.lo.s32 	%r152, %r95, 4100, %r129;
	mad.lo.s32 	%r153, %r95, 4101, %r129;
	mad.lo.s32 	%r154, %r95, 4102, %r129;
	mad.lo.s32 	%r155, %r95, 4103, %r129;
	mad.lo.s32 	%r156, %r95, 4104, %r129;
	mad.lo.s32 	%r157, %r95, 4105, %r129;
	mad.lo.s32 	%r158, %r95, 4106, %r129;
	mad.lo.s32 	%r159, %r95, 4107, %r129;
	mad.lo.s32 	%r160, %r95, 4108, %r129;
	mad.lo.s32 	%r161, %r95, 4109, %r129;
	mad.lo.s32 	%r162, %r95, 4110, %r129;
	mad.lo.s32 	%r163, %r95, 4111, %r129;
	.loc	1 74 33                         // sk09_norm_embed.py:74:33
	mad.wide.s32 	%rd9, %r129, 4, %rd46;
	mad.wide.s32 	%rd10, %r95, 4, %rd9;
	mad.wide.s32 	%rd11, %r131, 4, %rd46;
	mad.wide.s32 	%rd12, %r132, 4, %rd46;
	mad.wide.s32 	%rd13, %r134, 4, %rd46;
	mad.wide.s32 	%rd14, %r135, 4, %rd46;
	mad.wide.s32 	%rd15, %r136, 4, %rd46;
	mad.wide.s32 	%rd16, %r137, 4, %rd46;
	mad.wide.s32 	%rd17, %r139, 4, %rd46;
	mad.wide.s32 	%rd18, %r140, 4, %rd46;
	mad.wide.s32 	%rd19, %r141, 4, %rd46;
	mad.wide.s32 	%rd20, %r142, 4, %rd46;
	mad.wide.s32 	%rd21, %r143, 4, %rd46;
	mad.wide.s32 	%rd22, %r144, 4, %rd46;
	mad.wide.s32 	%rd23, %r145, 4, %rd46;
	mad.wide.s32 	%rd24, %r146, 4, %rd46;
	mad.wide.s32 	%rd25, %r148, 4, %rd46;
	mad.wide.s32 	%rd26, %r149, 4, %rd46;
	mad.wide.s32 	%rd27, %r150, 4, %rd46;
	mad.wide.s32 	%rd28, %r151, 4, %rd46;
	mad.wide.s32 	%rd29, %r152, 4, %rd46;
	mad.wide.s32 	%rd30, %r153, 4, %rd46;
	mad.wide.s32 	%rd31, %r154, 4, %rd46;
	mad.wide.s32 	%rd32, %r155, 4, %rd46;
	mad.wide.s32 	%rd33, %r156, 4, %rd46;
	mad.wide.s32 	%rd34, %r157, 4, %rd46;
	mad.wide.s32 	%rd35, %r158, 4, %rd46;
	mad.wide.s32 	%rd36, %r159, 4, %rd46;
	mad.wide.s32 	%rd37, %r160, 4, %rd46;
	mad.wide.s32 	%rd38, %r161, 4, %rd46;
	mad.wide.s32 	%rd39, %r162, 4, %rd46;
	mad.wide.s32 	%rd40, %r163, 4, %rd46;
	mov.b32 	%r35, 1065353216;
	.loc	1 74 20                         // sk09_norm_embed.py:74:20
	// begin inline asm
	mov.u32 %r34, %r35;
	@%p1 ld.global.b32 { %r34 }, [ %rd9 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r36, %r35;
	@%p1 ld.global.b32 { %r36 }, [ %rd10 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r37, %r35;
	@%p1 ld.global.b32 { %r37 }, [ %rd11 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r38, %r35;
	@%p1 ld.global.b32 { %r38 }, [ %rd12 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r39, %r35;
	@%p1 ld.global.b32 { %r39 }, [ %rd13 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r40, %r35;
	@%p1 ld.global.b32 { %r40 }, [ %rd14 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r41, %r35;
	@%p1 ld.global.b32 { %r41 }, [ %rd15 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r42, %r35;
	@%p1 ld.global.b32 { %r42 }, [ %rd16 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r43, %r35;
	@%p1 ld.global.b32 { %r43 }, [ %rd17 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r44, %r35;
	@%p1 ld.global.b32 { %r44 }, [ %rd18 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r45, %r35;
	@%p1 ld.global.b32 { %r45 }, [ %rd19 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r46, %r35;
	@%p1 ld.global.b32 { %r46 }, [ %rd20 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r47, %r35;
	@%p1 ld.global.b32 { %r47 }, [ %rd21 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r48, %r35;
	@%p1 ld.global.b32 { %r48 }, [ %rd22 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r49, %r35;
	@%p1 ld.global.b32 { %r49 }, [ %rd23 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r50, %r35;
	@%p1 ld.global.b32 { %r50 }, [ %rd24 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r51, %r35;
	@%p2 ld.global.b32 { %r51 }, [ %rd25 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r52, %r35;
	@%p2 ld.global.b32 { %r52 }, [ %rd26 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r53, %r35;
	@%p2 ld.global.b32 { %r53 }, [ %rd27 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r54, %r35;
	@%p2 ld.global.b32 { %r54 }, [ %rd28 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r55, %r35;
	@%p2 ld.global.b32 { %r55 }, [ %rd29 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r56, %r35;
	@%p2 ld.global.b32 { %r56 }, [ %rd30 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r57, %r35;
	@%p2 ld.global.b32 { %r57 }, [ %rd31 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r58, %r35;
	@%p2 ld.global.b32 { %r58 }, [ %rd32 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r59, %r35;
	@%p2 ld.global.b32 { %r59 }, [ %rd33 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r60, %r35;
	@%p2 ld.global.b32 { %r60 }, [ %rd34 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r61, %r35;
	@%p2 ld.global.b32 { %r61 }, [ %rd35 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r62, %r35;
	@%p2 ld.global.b32 { %r62 }, [ %rd36 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r63, %r35;
	@%p2 ld.global.b32 { %r63 }, [ %rd37 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r64, %r35;
	@%p2 ld.global.b32 { %r64 }, [ %rd38 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r65, %r35;
	@%p2 ld.global.b32 { %r65 }, [ %rd39 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r66, %r35;
	@%p2 ld.global.b32 { %r66 }, [ %rd40 + 0 ];
	// end inline asm
	.loc	1 75 32                         // sk09_norm_embed.py:75:32
	mul.f32 	%r164, %r100, %r100;
$L__tmp1:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	fma.rn.f32 	%r165, %r99, %r99, %r164;
	fma.rn.f32 	%r166, %r98, %r98, %r165;
	fma.rn.f32 	%r167, %r97, %r97, %r166;
	fma.rn.f32 	%r168, %r104, %r104, %r167;
	fma.rn.f32 	%r169, %r103, %r103, %r168;
	fma.rn.f32 	%r170, %r102, %r102, %r169;
	fma.rn.f32 	%r171, %r101, %r101, %r170;
	fma.rn.f32 	%r172, %r108, %r108, %r171;
	fma.rn.f32 	%r173, %r107, %r107, %r172;
	fma.rn.f32 	%r174, %r106, %r106, %r173;
	fma.rn.f32 	%r175, %r105, %r105, %r174;
	fma.rn.f32 	%r176, %r112, %r112, %r175;
	fma.rn.f32 	%r177, %r111, %r111, %r176;
	fma.rn.f32 	%r178, %r110, %r110, %r177;
	fma.rn.f32 	%r179, %r109, %r109, %r178;
	fma.rn.f32 	%r180, %r116, %r116, %r179;
	fma.rn.f32 	%r181, %r115, %r115, %r180;
	fma.rn.f32 	%r182, %r114, %r114, %r181;
	fma.rn.f32 	%r183, %r113, %r113, %r182;
	fma.rn.f32 	%r184, %r120, %r120, %r183;
	fma.rn.f32 	%r185, %r119, %r119, %r184;
	fma.rn.f32 	%r186, %r118, %r118, %r185;
	fma.rn.f32 	%r187, %r117, %r117, %r186;
	fma.rn.f32 	%r188, %r124, %r124, %r187;
	fma.rn.f32 	%r189, %r123, %r123, %r188;
	fma.rn.f32 	%r190, %r122, %r122, %r189;
	fma.rn.f32 	%r191, %r121, %r121, %r190;
	fma.rn.f32 	%r192, %r128, %r128, %r191;
	fma.rn.f32 	%r193, %r127, %r127, %r192;
	fma.rn.f32 	%r194, %r126, %r126, %r193;
	fma.rn.f32 	%r195, %r125, %r125, %r194;
$L__tmp2:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	shfl.sync.bfly.b32 	%r196, %r195, 16, 31, -1;
$L__tmp3:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	add.f32 	%r197, %r195, %r196;
$L__tmp4:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	shfl.sync.bfly.b32 	%r198, %r197, 8, 31, -1;
$L__tmp5:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	add.f32 	%r199, %r197, %r198;
$L__tmp6:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	shfl.sync.bfly.b32 	%r200, %r199, 4, 31, -1;
$L__tmp7:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	add.f32 	%r201, %r199, %r200;
$L__tmp8:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	shfl.sync.bfly.b32 	%r202, %r201, 2, 31, -1;
$L__tmp9:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	add.f32 	%r203, %r201, %r202;
$L__tmp10:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	shfl.sync.bfly.b32 	%r204, %r203, 1, 31, -1;
$L__tmp11:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	add.f32 	%r68, %r203, %r204;
$L__tmp12:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	setp.eq.b32 	%p3, %r87, 0;
	shr.u32 	%r205, %r85, 3;
	and.b32 	%r206, %r205, 28;
	mov.b32 	%r207, global_smem;
	add.s32 	%r67, %r207, %r206;
	// begin inline asm
	@%p3 st.shared.b32 [ %r67 + 0 ], %r68;
	// end inline asm
	bar.sync 	0;
	setp.lt.u32 	%p4, %r86, 8;
	shl.b32 	%r208, %r86, 2;
	add.s32 	%r70, %r207, %r208;
	// begin inline asm
	@%p4 ld.shared.b32 %r69, [ %r70 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r209, %r69, 4, 31, -1;
$L__tmp13:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	add.f32 	%r210, %r69, %r209;
$L__tmp14:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	shfl.sync.bfly.b32 	%r211, %r210, 2, 31, -1;
$L__tmp15:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	add.f32 	%r212, %r210, %r211;
$L__tmp16:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	shfl.sync.bfly.b32 	%r213, %r212, 1, 31, -1;
$L__tmp17:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	add.f32 	%r71, %r212, %r213;
$L__tmp18:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	and.b32 	%r214, %r85, 7;
	setp.eq.b32 	%p7, %r214, 0;
	and.pred 	%p5, %p4, %p7;
	// begin inline asm
	@%p5 st.shared.b32 [ %r70 + 0 ], %r71;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r215, [global_smem];
$L__tmp19:
	.loc	1 75 45                         // sk09_norm_embed.py:75:45
	cvt.rn.f32.s32 	%r216, %r89;
	div.full.f32 	%r217, %r215, %r216;
	.loc	1 75 49                         // sk09_norm_embed.py:75:49
	add.f32 	%r218, %r217, 0f358637BD;
	.loc	1 75 21                         // sk09_norm_embed.py:75:21
	rsqrt.approx.ftz.f32 	%r219, %r218;
	.loc	1 75 12                         // sk09_norm_embed.py:75:12
	mul.f32 	%r220, %r219, %r99;
	mul.f32 	%r221, %r219, %r100;
	mul.f32 	%r222, %r219, %r98;
	mul.f32 	%r223, %r219, %r97;
	mul.f32 	%r224, %r219, %r104;
	mul.f32 	%r225, %r219, %r103;
	mul.f32 	%r226, %r219, %r102;
	mul.f32 	%r227, %r219, %r101;
	mul.f32 	%r228, %r219, %r108;
	mul.f32 	%r229, %r219, %r107;
	mul.f32 	%r230, %r219, %r106;
	mul.f32 	%r231, %r219, %r105;
	mul.f32 	%r232, %r219, %r112;
	mul.f32 	%r233, %r219, %r111;
	mul.f32 	%r234, %r219, %r110;
	mul.f32 	%r235, %r219, %r109;
	mul.f32 	%r236, %r219, %r116;
	mul.f32 	%r237, %r219, %r115;
	mul.f32 	%r238, %r219, %r114;
	mul.f32 	%r239, %r219, %r113;
	mul.f32 	%r240, %r219, %r120;
	mul.f32 	%r241, %r219, %r119;
	mul.f32 	%r242, %r219, %r118;
	mul.f32 	%r243, %r219, %r117;
	mul.f32 	%r244, %r219, %r124;
	mul.f32 	%r245, %r219, %r123;
	mul.f32 	%r246, %r219, %r122;
	mul.f32 	%r247, %r219, %r121;
	mul.f32 	%r248, %r219, %r128;
	mul.f32 	%r249, %r219, %r127;
	mul.f32 	%r250, %r219, %r126;
	mul.f32 	%r251, %r219, %r125;
$L__tmp20:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	bar.sync 	0;
$L__tmp21:
	.loc	1 79 27                         // sk09_norm_embed.py:79:27
	mul.lo.s32 	%r252, %r93, %r84;
	.loc	1 79 21                         // sk09_norm_embed.py:79:21
	cvt.s64.s32 	%rd52, %r252;
	add.s64 	%rd53, %rd47, %rd52;
	.loc	1 79 39                         // sk09_norm_embed.py:79:39
	add.s64 	%rd41, %rd53, %rd50;
	add.s64 	%rd42, %rd41, 4096;
	.loc	1 73 55                         // sk09_norm_embed.py:73:55
	mov.b32 	{%rs33, %rs34}, %r24;
	cvt.f32.bf16 	%r253, %rs33;
	cvt.f32.bf16 	%r254, %rs34;
	mov.b32 	{%rs35, %rs36}, %r25;
	cvt.f32.bf16 	%r255, %rs35;
	cvt.f32.bf16 	%r256, %rs36;
	.loc	1 73 69                         // sk09_norm_embed.py:73:69
	add.f32 	%r257, %r256, 0f00000000;
	add.f32 	%r258, %r255, 0f00000000;
	add.f32 	%r259, %r254, 0f00000000;
	add.f32 	%r260, %r253, 0f00000000;
	.loc	1 74 12                         // sk09_norm_embed.py:74:12
	mul.f32 	%r261, %r260, %r47;
	mul.f32 	%r262, %r259, %r48;
	mul.f32 	%r263, %r258, %r49;
	mul.f32 	%r264, %r257, %r50;
	.loc	1 75 56                         // sk09_norm_embed.py:75:56
	mul.f32 	%r265, %r264, %r235;
	mul.f32 	%r266, %r263, %r234;
	mul.f32 	%r267, %r262, %r233;
	mul.f32 	%r268, %r261, %r232;
	.loc	1 76 36                         // sk09_norm_embed.py:76:36
	abs.f32 	%r269, %r268;
	abs.f32 	%r270, %r267;
	abs.f32 	%r271, %r266;
	abs.f32 	%r272, %r265;
	.loc	1 73 55                         // sk09_norm_embed.py:73:55
	mov.b32 	{%rs37, %rs38}, %r22;
	cvt.f32.bf16 	%r273, %rs37;
	cvt.f32.bf16 	%r274, %rs38;
	mov.b32 	{%rs39, %rs40}, %r23;
	cvt.f32.bf16 	%r275, %rs39;
	cvt.f32.bf16 	%r276, %rs40;
	.loc	1 73 69                         // sk09_norm_embed.py:73:69
	add.f32 	%r277, %r276, 0f00000000;
	add.f32 	%r278, %r275, 0f00000000;
	add.f32 	%r279, %r274, 0f00000000;
	add.f32 	%r280, %r273, 0f00000000;
	.loc	1 74 12                         // sk09_norm_embed.py:74:12
	mul.f32 	%r281, %r280, %r43;
	mul.f32 	%r282, %r279, %r44;
	mul.f32 	%r283, %r278, %r45;
	mul.f32 	%r284, %r277, %r46;
	.loc	1 75 56                         // sk09_norm_embed.py:75:56
	mul.f32 	%r285, %r284, %r231;
	mul.f32 	%r286, %r283, %r230;
	mul.f32 	%r287, %r282, %r229;
	mul.f32 	%r288, %r281, %r228;
	.loc	1 76 36                         // sk09_norm_embed.py:76:36
	abs.f32 	%r289, %r288;
	abs.f32 	%r290, %r287;
	abs.f32 	%r291, %r286;
	abs.f32 	%r292, %r285;
	.loc	1 73 55                         // sk09_norm_embed.py:73:55
	mov.b32 	{%rs41, %rs42}, %r20;
	cvt.f32.bf16 	%r293, %rs41;
	cvt.f32.bf16 	%r294, %rs42;
	mov.b32 	{%rs43, %rs44}, %r21;
	cvt.f32.bf16 	%r295, %rs43;
	cvt.f32.bf16 	%r296, %rs44;
	.loc	1 73 69                         // sk09_norm_embed.py:73:69
	add.f32 	%r297, %r296, 0f00000000;
	add.f32 	%r298, %r295, 0f00000000;
	add.f32 	%r299, %r294, 0f00000000;
	add.f32 	%r300, %r293, 0f00000000;
	.loc	1 74 12                         // sk09_norm_embed.py:74:12
	mul.f32 	%r301, %r300, %r39;
	mul.f32 	%r302, %r299, %r40;
	mul.f32 	%r303, %r298, %r41;
	mul.f32 	%r304, %r297, %r42;
	.loc	1 75 56                         // sk09_norm_embed.py:75:56
	mul.f32 	%r305, %r304, %r227;
	mul.f32 	%r306, %r303, %r226;
	mul.f32 	%r307, %r302, %r225;
	mul.f32 	%r308, %r301, %r224;
	.loc	1 76 36                         // sk09_norm_embed.py:76:36
	abs.f32 	%r309, %r308;
	abs.f32 	%r310, %r307;
	abs.f32 	%r311, %r306;
	abs.f32 	%r312, %r305;
	.loc	1 73 55                         // sk09_norm_embed.py:73:55
	mov.b32 	{%rs45, %rs46}, %r18;
	cvt.f32.bf16 	%r313, %rs45;
	cvt.f32.bf16 	%r314, %rs46;
	mov.b32 	{%rs47, %rs48}, %r19;
	cvt.f32.bf16 	%r315, %rs47;
	cvt.f32.bf16 	%r316, %rs48;
	.loc	1 73 69                         // sk09_norm_embed.py:73:69
	add.f32 	%r317, %r316, 0f00000000;
	add.f32 	%r318, %r315, 0f00000000;
	add.f32 	%r319, %r314, 0f00000000;
	add.f32 	%r320, %r313, 0f00000000;
	.loc	1 74 12                         // sk09_norm_embed.py:74:12
	mul.f32 	%r321, %r320, %r34;
	mul.f32 	%r322, %r319, %r36;
	mul.f32 	%r323, %r318, %r37;
	mul.f32 	%r324, %r317, %r38;
	.loc	1 75 56                         // sk09_norm_embed.py:75:56
	mul.f32 	%r325, %r324, %r223;
	mul.f32 	%r326, %r323, %r222;
	mul.f32 	%r327, %r322, %r221;
	mul.f32 	%r328, %r321, %r220;
	.loc	1 76 36                         // sk09_norm_embed.py:76:36
	abs.f32 	%r329, %r328;
	abs.f32 	%r330, %r327;
	abs.f32 	%r331, %r326;
	abs.f32 	%r332, %r325;
$L__tmp22:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r333, %r329, %r330;
	max.f32 	%r334, %r333, %r331;
	max.f32 	%r335, %r334, %r332;
	max.f32 	%r336, %r335, %r309;
	max.f32 	%r337, %r336, %r310;
	max.f32 	%r338, %r337, %r311;
	max.f32 	%r339, %r338, %r312;
	max.f32 	%r340, %r339, %r289;
	max.f32 	%r341, %r340, %r290;
	max.f32 	%r342, %r341, %r291;
	max.f32 	%r343, %r342, %r292;
	max.f32 	%r344, %r343, %r269;
	max.f32 	%r345, %r344, %r270;
	max.f32 	%r346, %r345, %r271;
	max.f32 	%r347, %r346, %r272;
$L__tmp23:
	.loc	1 73 55                         // sk09_norm_embed.py:73:55
	mov.b32 	{%rs49, %rs50}, %r32;
	cvt.f32.bf16 	%r348, %rs49;
	cvt.f32.bf16 	%r349, %rs50;
	mov.b32 	{%rs51, %rs52}, %r33;
	cvt.f32.bf16 	%r350, %rs51;
	cvt.f32.bf16 	%r351, %rs52;
	.loc	1 73 69                         // sk09_norm_embed.py:73:69
	add.f32 	%r352, %r351, 0f00000000;
	add.f32 	%r353, %r350, 0f00000000;
	add.f32 	%r354, %r349, 0f00000000;
	add.f32 	%r355, %r348, 0f00000000;
	.loc	1 74 12                         // sk09_norm_embed.py:74:12
	mul.f32 	%r356, %r355, %r63;
	mul.f32 	%r357, %r354, %r64;
	mul.f32 	%r358, %r353, %r65;
	mul.f32 	%r359, %r352, %r66;
	.loc	1 75 56                         // sk09_norm_embed.py:75:56
	mul.f32 	%r360, %r359, %r251;
	mul.f32 	%r361, %r358, %r250;
	mul.f32 	%r362, %r357, %r249;
	mul.f32 	%r363, %r356, %r248;
	.loc	1 76 36                         // sk09_norm_embed.py:76:36
	abs.f32 	%r364, %r363;
	abs.f32 	%r365, %r362;
	abs.f32 	%r366, %r361;
	abs.f32 	%r367, %r360;
	.loc	1 73 55                         // sk09_norm_embed.py:73:55
	mov.b32 	{%rs53, %rs54}, %r30;
	cvt.f32.bf16 	%r368, %rs53;
	cvt.f32.bf16 	%r369, %rs54;
	mov.b32 	{%rs55, %rs56}, %r31;
	cvt.f32.bf16 	%r370, %rs55;
	cvt.f32.bf16 	%r371, %rs56;
	.loc	1 73 69                         // sk09_norm_embed.py:73:69
	add.f32 	%r372, %r371, 0f00000000;
	add.f32 	%r373, %r370, 0f00000000;
	add.f32 	%r374, %r369, 0f00000000;
	add.f32 	%r375, %r368, 0f00000000;
	.loc	1 74 12                         // sk09_norm_embed.py:74:12
	mul.f32 	%r376, %r375, %r59;
	mul.f32 	%r377, %r374, %r60;
	mul.f32 	%r378, %r373, %r61;
	mul.f32 	%r379, %r372, %r62;
	.loc	1 75 56                         // sk09_norm_embed.py:75:56
	mul.f32 	%r380, %r379, %r247;
	mul.f32 	%r381, %r378, %r246;
	mul.f32 	%r382, %r377, %r245;
	mul.f32 	%r383, %r376, %r244;
	.loc	1 76 36                         // sk09_norm_embed.py:76:36
	abs.f32 	%r384, %r383;
	abs.f32 	%r385, %r382;
	abs.f32 	%r386, %r381;
	abs.f32 	%r387, %r380;
	.loc	1 73 55                         // sk09_norm_embed.py:73:55
	mov.b32 	{%rs57, %rs58}, %r28;
	cvt.f32.bf16 	%r388, %rs57;
	cvt.f32.bf16 	%r389, %rs58;
	mov.b32 	{%rs59, %rs60}, %r29;
	cvt.f32.bf16 	%r390, %rs59;
	cvt.f32.bf16 	%r391, %rs60;
	.loc	1 73 69                         // sk09_norm_embed.py:73:69
	add.f32 	%r392, %r391, 0f00000000;
	add.f32 	%r393, %r390, 0f00000000;
	add.f32 	%r394, %r389, 0f00000000;
	add.f32 	%r395, %r388, 0f00000000;
	.loc	1 74 12                         // sk09_norm_embed.py:74:12
	mul.f32 	%r396, %r395, %r55;
	mul.f32 	%r397, %r394, %r56;
	mul.f32 	%r398, %r393, %r57;
	mul.f32 	%r399, %r392, %r58;
	.loc	1 75 56                         // sk09_norm_embed.py:75:56
	mul.f32 	%r400, %r399, %r243;
	mul.f32 	%r401, %r398, %r242;
	mul.f32 	%r402, %r397, %r241;
	mul.f32 	%r403, %r396, %r240;
	.loc	1 76 36                         // sk09_norm_embed.py:76:36
	abs.f32 	%r404, %r403;
	abs.f32 	%r405, %r402;
	abs.f32 	%r406, %r401;
	abs.f32 	%r407, %r400;
	.loc	1 73 55                         // sk09_norm_embed.py:73:55
	mov.b32 	{%rs61, %rs62}, %r26;
	cvt.f32.bf16 	%r408, %rs61;
	cvt.f32.bf16 	%r409, %rs62;
	mov.b32 	{%rs63, %rs64}, %r27;
	cvt.f32.bf16 	%r410, %rs63;
	cvt.f32.bf16 	%r411, %rs64;
	.loc	1 73 69                         // sk09_norm_embed.py:73:69
	add.f32 	%r412, %r411, 0f00000000;
	add.f32 	%r413, %r410, 0f00000000;
	add.f32 	%r414, %r409, 0f00000000;
	add.f32 	%r415, %r408, 0f00000000;
	.loc	1 74 12                         // sk09_norm_embed.py:74:12
	mul.f32 	%r416, %r415, %r51;
	mul.f32 	%r417, %r414, %r52;
	mul.f32 	%r418, %r413, %r53;
	mul.f32 	%r419, %r412, %r54;
	.loc	1 75 56                         // sk09_norm_embed.py:75:56
	mul.f32 	%r420, %r419, %r239;
	mul.f32 	%r421, %r418, %r238;
	mul.f32 	%r422, %r417, %r237;
	mul.f32 	%r423, %r416, %r236;
	.loc	1 76 36                         // sk09_norm_embed.py:76:36
	abs.f32 	%r424, %r423;
	abs.f32 	%r425, %r422;
	abs.f32 	%r426, %r421;
	abs.f32 	%r427, %r420;
$L__tmp24:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r428, %r347, %r424;
	max.f32 	%r429, %r428, %r425;
	max.f32 	%r430, %r429, %r426;
	max.f32 	%r431, %r430, %r427;
	max.f32 	%r432, %r431, %r404;
	max.f32 	%r433, %r432, %r405;
	max.f32 	%r434, %r433, %r406;
	max.f32 	%r435, %r434, %r407;
	max.f32 	%r436, %r435, %r384;
	max.f32 	%r437, %r436, %r385;
	max.f32 	%r438, %r437, %r386;
	max.f32 	%r439, %r438, %r387;
	max.f32 	%r440, %r439, %r364;
	max.f32 	%r441, %r440, %r365;
	max.f32 	%r442, %r441, %r366;
	max.f32 	%r443, %r442, %r367;
$L__tmp25:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	shfl.sync.bfly.b32 	%r444, %r443, 16, 31, -1;
$L__tmp26:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r445, %r443, %r444;
$L__tmp27:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	shfl.sync.bfly.b32 	%r446, %r445, 8, 31, -1;
$L__tmp28:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r447, %r445, %r446;
$L__tmp29:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	shfl.sync.bfly.b32 	%r448, %r447, 4, 31, -1;
$L__tmp30:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r449, %r447, %r448;
$L__tmp31:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	shfl.sync.bfly.b32 	%r450, %r449, 2, 31, -1;
$L__tmp32:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r451, %r449, %r450;
$L__tmp33:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	shfl.sync.bfly.b32 	%r452, %r451, 1, 31, -1;
$L__tmp34:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r72, %r451, %r452;
$L__tmp35:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	// begin inline asm
	@%p3 st.shared.b32 [ %r67 + 0 ], %r72;
	// end inline asm
	bar.sync 	0;
	// begin inline asm
	@%p4 ld.shared.b32 %r73, [ %r70 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r453, %r73, 4, 31, -1;
$L__tmp36:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r454, %r73, %r453;
$L__tmp37:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	shfl.sync.bfly.b32 	%r455, %r454, 2, 31, -1;
$L__tmp38:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r456, %r454, %r455;
$L__tmp39:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	shfl.sync.bfly.b32 	%r457, %r456, 1, 31, -1;
$L__tmp40:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r74, %r456, %r457;
$L__tmp41:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	// begin inline asm
	@%p5 st.shared.b32 [ %r70 + 0 ], %r74;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r458, [global_smem];
$L__tmp42:
	.loc	1 76 49                         // sk09_norm_embed.py:76:49
	max.f32 	%r459, %r458, 0f0DA24260;
	mov.b32 	%r460, 0f42FE0000;
	.loc	1 77 22                         // sk09_norm_embed.py:77:22
	div.full.f32 	%r461, %r460, %r459;
	.loc	1 77 14                         // sk09_norm_embed.py:77:14
	mul.f32 	%r462, %r327, %r461;
	mul.f32 	%r463, %r328, %r461;
	mul.f32 	%r464, %r325, %r461;
	mul.f32 	%r465, %r326, %r461;
	mul.f32 	%r466, %r307, %r461;
	mul.f32 	%r467, %r308, %r461;
	mul.f32 	%r468, %r305, %r461;
	mul.f32 	%r469, %r306, %r461;
	mul.f32 	%r470, %r287, %r461;
	mul.f32 	%r471, %r288, %r461;
	mul.f32 	%r472, %r285, %r461;
	mul.f32 	%r473, %r286, %r461;
	mul.f32 	%r474, %r267, %r461;
	mul.f32 	%r475, %r268, %r461;
	mul.f32 	%r476, %r265, %r461;
	mul.f32 	%r477, %r266, %r461;
	mul.f32 	%r478, %r422, %r461;
	mul.f32 	%r479, %r423, %r461;
	mul.f32 	%r480, %r420, %r461;
	mul.f32 	%r481, %r421, %r461;
	mul.f32 	%r482, %r402, %r461;
	mul.f32 	%r483, %r403, %r461;
	mul.f32 	%r484, %r400, %r461;
	mul.f32 	%r485, %r401, %r461;
	mul.f32 	%r486, %r382, %r461;
	mul.f32 	%r487, %r383, %r461;
	mul.f32 	%r488, %r380, %r461;
	mul.f32 	%r489, %r381, %r461;
	mul.f32 	%r490, %r362, %r461;
	mul.f32 	%r491, %r363, %r461;
	mul.f32 	%r492, %r360, %r461;
	mul.f32 	%r493, %r361, %r461;
	.loc	1 78 29                         // sk09_norm_embed.py:78:29
	.loc	1 78 39                         // sk09_norm_embed.py:78:39
	lop3.b32 	%r494, 0x3f000000, %r462, 0x80000000, 0xF8;
	lop3.b32 	%r495, 0x3f000000, %r463, 0x80000000, 0xF8;
	lop3.b32 	%r496, 0x3f000000, %r464, 0x80000000, 0xF8;
	lop3.b32 	%r497, 0x3f000000, %r465, 0x80000000, 0xF8;
	lop3.b32 	%r498, 0x3f000000, %r466, 0x80000000, 0xF8;
	lop3.b32 	%r499, 0x3f000000, %r467, 0x80000000, 0xF8;
	lop3.b32 	%r500, 0x3f000000, %r468, 0x80000000, 0xF8;
	lop3.b32 	%r501, 0x3f000000, %r469, 0x80000000, 0xF8;
	lop3.b32 	%r502, 0x3f000000, %r470, 0x80000000, 0xF8;
	lop3.b32 	%r503, 0x3f000000, %r471, 0x80000000, 0xF8;
	lop3.b32 	%r504, 0x3f000000, %r472, 0x80000000, 0xF8;
	lop3.b32 	%r505, 0x3f000000, %r473, 0x80000000, 0xF8;
	lop3.b32 	%r506, 0x3f000000, %r474, 0x80000000, 0xF8;
	lop3.b32 	%r507, 0x3f000000, %r475, 0x80000000, 0xF8;
	lop3.b32 	%r508, 0x3f000000, %r476, 0x80000000, 0xF8;
	lop3.b32 	%r509, 0x3f000000, %r477, 0x80000000, 0xF8;
	lop3.b32 	%r510, 0x3f000000, %r478, 0x80000000, 0xF8;
	lop3.b32 	%r511, 0x3f000000, %r479, 0x80000000, 0xF8;
	lop3.b32 	%r512, 0x3f000000, %r480, 0x80000000, 0xF8;
	lop3.b32 	%r513, 0x3f000000, %r481, 0x80000000, 0xF8;
	lop3.b32 	%r514, 0x3f000000, %r482, 0x80000000, 0xF8;
	lop3.b32 	%r515, 0x3f000000, %r483, 0x80000000, 0xF8;
	lop3.b32 	%r516, 0x3f000000, %r484, 0x80000000, 0xF8;
	lop3.b32 	%r517, 0x3f000000, %r485, 0x80000000, 0xF8;
	lop3.b32 	%r518, 0x3f000000, %r486, 0x80000000, 0xF8;
	lop3.b32 	%r519, 0x3f000000, %r487, 0x80000000, 0xF8;
	lop3.b32 	%r520, 0x3f000000, %r488, 0x80000000, 0xF8;
	lop3.b32 	%r521, 0x3f000000, %r489, 0x80000000, 0xF8;
	lop3.b32 	%r522, 0x3f000000, %r490, 0x80000000, 0xF8;
	lop3.b32 	%r523, 0x3f000000, %r491, 0x80000000, 0xF8;
	lop3.b32 	%r524, 0x3f000000, %r492, 0x80000000, 0xF8;
	lop3.b32 	%r525, 0x3f000000, %r493, 0x80000000, 0xF8;
	.loc	1 78 14                         // sk09_norm_embed.py:78:14
	fma.rn.f32 	%r526, %r326, %r461, %r497;
	fma.rn.f32 	%r527, %r325, %r461, %r496;
	fma.rn.f32 	%r528, %r328, %r461, %r495;
	fma.rn.f32 	%r529, %r327, %r461, %r494;
	fma.rn.f32 	%r530, %r306, %r461, %r501;
	fma.rn.f32 	%r531, %r305, %r461, %r500;
	fma.rn.f32 	%r532, %r308, %r461, %r499;
	fma.rn.f32 	%r533, %r307, %r461, %r498;
	fma.rn.f32 	%r534, %r286, %r461, %r505;
	fma.rn.f32 	%r535, %r285, %r461, %r504;
	fma.rn.f32 	%r536, %r288, %r461, %r503;
	fma.rn.f32 	%r537, %r287, %r461, %r502;
	fma.rn.f32 	%r538, %r266, %r461, %r509;
	fma.rn.f32 	%r539, %r265, %r461, %r508;
	fma.rn.f32 	%r540, %r268, %r461, %r507;
	fma.rn.f32 	%r541, %r267, %r461, %r506;
	fma.rn.f32 	%r542, %r421, %r461, %r513;
	fma.rn.f32 	%r543, %r420, %r461, %r512;
	fma.rn.f32 	%r544, %r423, %r461, %r511;
	fma.rn.f32 	%r545, %r422, %r461, %r510;
	fma.rn.f32 	%r546, %r401, %r461, %r517;
	fma.rn.f32 	%r547, %r400, %r461, %r516;
	fma.rn.f32 	%r548, %r403, %r461, %r515;
	fma.rn.f32 	%r549, %r402, %r461, %r514;
	fma.rn.f32 	%r550, %r381, %r461, %r521;
	fma.rn.f32 	%r551, %r380, %r461, %r520;
	fma.rn.f32 	%r552, %r383, %r461, %r519;
	fma.rn.f32 	%r553, %r382, %r461, %r518;
	fma.rn.f32 	%r554, %r361, %r461, %r525;
	fma.rn.f32 	%r555, %r360, %r461, %r524;
	fma.rn.f32 	%r556, %r363, %r461, %r523;
	fma.rn.f32 	%r557, %r362, %r461, %r522;
	.loc	1 78 49                         // sk09_norm_embed.py:78:49
	cvt.rzi.s32.f32 	%r558, %r529;
	cvt.rzi.s32.f32 	%r559, %r528;
	cvt.rzi.s32.f32 	%r560, %r527;
	cvt.rzi.s32.f32 	%r561, %r526;
	cvt.rzi.s32.f32 	%r562, %r533;
	cvt.rzi.s32.f32 	%r563, %r532;
	cvt.rzi.s32.f32 	%r564, %r531;
	cvt.rzi.s32.f32 	%r565, %r530;
	cvt.rzi.s32.f32 	%r566, %r537;
	cvt.rzi.s32.f32 	%r567, %r536;
	cvt.rzi.s32.f32 	%r568, %r535;
	cvt.rzi.s32.f32 	%r569, %r534;
	cvt.rzi.s32.f32 	%r570, %r541;
	cvt.rzi.s32.f32 	%r571, %r540;
	cvt.rzi.s32.f32 	%r572, %r539;
	cvt.rzi.s32.f32 	%r573, %r538;
	cvt.rzi.s32.f32 	%r574, %r545;
	cvt.rzi.s32.f32 	%r575, %r544;
	cvt.rzi.s32.f32 	%r576, %r543;
	cvt.rzi.s32.f32 	%r577, %r542;
	cvt.rzi.s32.f32 	%r578, %r549;
	cvt.rzi.s32.f32 	%r579, %r548;
	cvt.rzi.s32.f32 	%r580, %r547;
	cvt.rzi.s32.f32 	%r581, %r546;
	cvt.rzi.s32.f32 	%r582, %r553;
	cvt.rzi.s32.f32 	%r583, %r552;
	cvt.rzi.s32.f32 	%r584, %r551;
	cvt.rzi.s32.f32 	%r585, %r550;
	cvt.rzi.s32.f32 	%r586, %r557;
	cvt.rzi.s32.f32 	%r587, %r556;
	cvt.rzi.s32.f32 	%r588, %r555;
	cvt.rzi.s32.f32 	%r589, %r554;
	.loc	1 79 70                         // sk09_norm_embed.py:79:70
	max.s32 	%r590, %r561, -127;
	max.s32 	%r591, %r560, -127;
	max.s32 	%r592, %r559, -127;
	max.s32 	%r593, %r558, -127;
	max.s32 	%r594, %r565, -127;
	max.s32 	%r595, %r564, -127;
	max.s32 	%r596, %r563, -127;
	max.s32 	%r597, %r562, -127;
	max.s32 	%r598, %r569, -127;
	max.s32 	%r599, %r568, -127;
	max.s32 	%r600, %r567, -127;
	max.s32 	%r601, %r566, -127;
	max.s32 	%r602, %r573, -127;
	max.s32 	%r603, %r572, -127;
	max.s32 	%r604, %r571, -127;
	max.s32 	%r605, %r570, -127;
	max.s32 	%r606, %r577, -127;
	max.s32 	%r607, %r576, -127;
	max.s32 	%r608, %r575, -127;
	max.s32 	%r609, %r574, -127;
	max.s32 	%r610, %r581, -127;
	max.s32 	%r611, %r580, -127;
	max.s32 	%r612, %r579, -127;
	max.s32 	%r613, %r578, -127;
	max.s32 	%r614, %r585, -127;
	max.s32 	%r615, %r584, -127;
	max.s32 	%r616, %r583, -127;
	max.s32 	%r617, %r582, -127;
	max.s32 	%r618, %r589, -127;
	max.s32 	%r619, %r588, -127;
	max.s32 	%r620, %r587, -127;
	max.s32 	%r621, %r586, -127;
	.loc	1 79 77                         // sk09_norm_embed.py:79:77
	min.s32 	%r622, %r593, 127;
	min.s32 	%r623, %r592, 127;
	min.s32 	%r624, %r591, 127;
	min.s32 	%r625, %r590, 127;
	min.s32 	%r626, %r597, 127;
	min.s32 	%r627, %r596, 127;
	min.s32 	%r628, %r595, 127;
	min.s32 	%r629, %r594, 127;
	min.s32 	%r630, %r601, 127;
	min.s32 	%r631, %r600, 127;
	min.s32 	%r632, %r599, 127;
	min.s32 	%r633, %r598, 127;
	min.s32 	%r634, %r605, 127;
	min.s32 	%r635, %r604, 127;
	min.s32 	%r636, %r603, 127;
	min.s32 	%r637, %r602, 127;
	min.s32 	%r638, %r609, 127;
	min.s32 	%r639, %r608, 127;
	min.s32 	%r640, %r607, 127;
	min.s32 	%r641, %r606, 127;
	min.s32 	%r642, %r613, 127;
	min.s32 	%r643, %r612, 127;
	min.s32 	%r644, %r611, 127;
	min.s32 	%r645, %r610, 127;
	min.s32 	%r646, %r617, 127;
	min.s32 	%r647, %r616, 127;
	min.s32 	%r648, %r615, 127;
	min.s32 	%r649, %r614, 127;
	min.s32 	%r650, %r621, 127;
	min.s32 	%r651, %r620, 127;
	min.s32 	%r652, %r619, 127;
	min.s32 	%r653, %r618, 127;
	.loc	1 79 85                         // sk09_norm_embed.py:79:85
	prmt.b32 	%r654, %r625, %r624, 0x3340U;
	prmt.b32 	%r655, %r623, %r622, 0x3340U;
	prmt.b32 	%r75, %r655, %r654, 0x5410U;
	prmt.b32 	%r656, %r629, %r628, 0x3340U;
	prmt.b32 	%r657, %r627, %r626, 0x3340U;
	prmt.b32 	%r76, %r657, %r656, 0x5410U;
	prmt.b32 	%r658, %r633, %r632, 0x3340U;
	prmt.b32 	%r659, %r631, %r630, 0x3340U;
	prmt.b32 	%r77, %r659, %r658, 0x5410U;
	prmt.b32 	%r660, %r637, %r636, 0x3340U;
	prmt.b32 	%r661, %r635, %r634, 0x3340U;
	prmt.b32 	%r78, %r661, %r660, 0x5410U;
	prmt.b32 	%r662, %r641, %r640, 0x3340U;
	prmt.b32 	%r663, %r639, %r638, 0x3340U;
	prmt.b32 	%r79, %r663, %r662, 0x5410U;
	prmt.b32 	%r664, %r645, %r644, 0x3340U;
	prmt.b32 	%r665, %r643, %r642, 0x3340U;
	prmt.b32 	%r80, %r665, %r664, 0x5410U;
	prmt.b32 	%r666, %r649, %r648, 0x3340U;
	prmt.b32 	%r667, %r647, %r646, 0x3340U;
	prmt.b32 	%r81, %r667, %r666, 0x5410U;
	prmt.b32 	%r668, %r653, %r652, 0x3340U;
	prmt.b32 	%r669, %r651, %r650, 0x3340U;
	prmt.b32 	%r82, %r669, %r668, 0x5410U;
	.loc	1 79 45                         // sk09_norm_embed.py:79:45
	// begin inline asm
	@%p1 st.global.v4.b32 [ %rd41 + 0 ], { %r75, %r76, %r77, %r78 };
	// end inline asm
	// begin inline asm
	@%p2 st.global.v4.b32 [ %rd42 + 0 ], { %r79, %r80, %r81, %r82 };
	// end inline asm
	.loc	1 80 21                         // sk09_norm_embed.py:80:21
	mad.wide.u32 	%rd43, %r84, 4, %rd48;
	.loc	1 80 34                         // sk09_norm_embed.py:80:34
	mul.f32 	%r83, %r459, 0f3C010204;
	.loc	1 80 26                         // sk09_norm_embed.py:80:26
	or.b32 	%r670, %r87, %r88;
	setp.eq.b32 	%p6, %r670, 0;
	// begin inline asm
	@%p6 st.global.b32 [ %rd43 + 0 ], { %r83 };
	// end inline asm
	.loc	1 80 4                          // sk09_norm_embed.py:80:4
	ret;
$L__tmp43:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk09_norm_embed.py"
	.file	2 "/usr/local/lib/python3.12/dist-packages/triton/language/standard.py"
	.section	.debug_abbrev
	{
.b8 1                                   // Abbreviation Code
.b8 17                                  // DW_TAG_compile_unit
.b8 1                                   // DW_CHILDREN_yes
.b8 37                                  // DW_AT_producer
.b8 8                                   // DW_FORM_string
.b8 19                                  // DW_AT_language
.b8 5                                   // DW_FORM_data2
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 16                                  // DW_AT_stmt_list
.b8 6                                   // DW_FORM_data4
.b8 27                                  // DW_AT_comp_dir
.b8 8                                   // DW_FORM_string
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 2                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 0                                   // DW_CHILDREN_no
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 32                                  // DW_AT_inline
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 3                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 1                                   // DW_CHILDREN_yes
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 4                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 1                                   // DW_CHILDREN_yes
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 11                                  // DW_FORM_data1
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 5                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 0                                   // DW_CHILDREN_no
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 5                                   // DW_FORM_data2
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 6                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 0                                   // DW_CHILDREN_no
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 11                                  // DW_FORM_data1
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 219                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0xd4 DW_TAG_compile_unit
.b8 116                                 // DW_AT_producer
.b8 114
.b8 105
.b8 116
.b8 111
.b8 110
.b8 0
.b8 2                                   // DW_AT_language
.b8 0
.b8 115                                 // DW_AT_name
.b8 107
.b8 48
.b8 57
.b8 95
.b8 110
.b8 111
.b8 114
.b8 109
.b8 95
.b8 101
.b8 109
.b8 98
.b8 101
.b8 100
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 114
.b8 101
.b8 112
.b8 111
.b8 47
.b8 118
.b8 108
.b8 108
.b8 109
.b8 47
.b8 95
.b8 103
.b8 101
.b8 110
.b8 101
.b8 115
.b8 105
.b8 115
.b8 47
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 115
.b8 0
.b8 2                                   // Abbrev [2] 0x48:0x1d DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 57
.b8 95
.b8 114
.b8 109
.b8 115
.b8 110
.b8 111
.b8 114
.b8 109
.b8 95
.b8 113
.b8 117
.b8 97
.b8 110
.b8 116
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x65:0x79 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 72                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x7a:0x32 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp19                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 75                                  // DW_AT_call_line
.b8 28                                  // DW_AT_call_column
.b8 5                                   // Abbrev [5] 0x92:0x19 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp18                          // DW_AT_high_pc
.b8 2                                   // DW_AT_call_file
.b8 37                                  // DW_AT_call_line
.b8 1
.b8 36                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 4                                   // Abbrev [4] 0xac:0x31 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp20                          // DW_AT_low_pc
.b64 $L__tmp42                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 76                                  // DW_AT_call_line
.b8 29                                  // DW_AT_call_column
.b8 6                                   // Abbrev [6] 0xc4:0x18 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp22                          // DW_AT_low_pc
.b64 $L__tmp41                          // DW_AT_high_pc
.b8 2                                   // DW_AT_call_file
.b8 191                                 // DW_AT_call_line
.b8 40                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_QVAR1_2 = _Nativo(
    "sk09_norm_embed/_sk09_rmsnorm_quant_kernel/blk8192",
    _QPTX1_2, "_sk09_rmsnorm_quant_kernel",
    warps=8, shared=32,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    horneado={9: 8192, 10: 1e-06, 11: 0.0},
    div16=[5, 6, 7, 8],
)
# E1=32 lop3, E2=0 mul

_QPTX1_3 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk09_rmsnorm_quant_kernel // -- Begin function _sk09_rmsnorm_quant_kernel
.extern .shared .align 16 .b8 global_smem[];
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
                                        // @_sk09_rmsnorm_quant_kernel
.visible .entry _sk09_rmsnorm_quant_kernel(
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_quant_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_quant_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_quant_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_quant_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_quant_kernel_param_4,
	.param .u32 _sk09_rmsnorm_quant_kernel_param_5,
	.param .u32 _sk09_rmsnorm_quant_kernel_param_6,
	.param .u32 _sk09_rmsnorm_quant_kernel_param_7,
	.param .u32 _sk09_rmsnorm_quant_kernel_param_8,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_quant_kernel_param_9,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_quant_kernel_param_10
)
.reqntid 256
{
	.reg .pred 	%p<74>;
	.reg .b16 	%rs<129>;
	.reg .b32 	%r<1289>;
	.reg .b64 	%rd<99>;
	.loc	1 64 0                          // sk09_norm_embed.py:64:0
$L__func_begin0:
	.loc	1 64 0                          // sk09_norm_embed.py:64:0

// %bb.0:                               // %__nv_rsqrtf.exit
	ld.param.b64 	%rd86, [_sk09_rmsnorm_quant_kernel_param_0];
	ld.param.b64 	%rd87, [_sk09_rmsnorm_quant_kernel_param_1];
$L__tmp0:
	.loc	1 69 24                         // sk09_norm_embed.py:69:24
	mov.u32 	%r156, %ctaid.x;
	ld.param.b64 	%rd88, [_sk09_rmsnorm_quant_kernel_param_2];
	.loc	1 70 24                         // sk09_norm_embed.py:70:24
	mov.u32 	%r157, %tid.x;
	and.b32 	%r158, %r157, 255;
	ld.param.b64 	%rd89, [_sk09_rmsnorm_quant_kernel_param_3];
	and.b32 	%r159, %r157, 31;
	ld.param.b64 	%rd90, [_sk09_rmsnorm_quant_kernel_param_4];
	shr.u32 	%r160, %r157, 5;
	ld.param.b32 	%r161, [_sk09_rmsnorm_quant_kernel_param_5];
	shl.b32 	%r162, %r157, 4;
	ld.param.b32 	%r163, [_sk09_rmsnorm_quant_kernel_param_6];
	and.b32 	%r164, %r162, 4080;
	ld.param.b32 	%r165, [_sk09_rmsnorm_quant_kernel_param_7];
	or.b32 	%r166, %r164, 4096;
	ld.param.b32 	%r167, [_sk09_rmsnorm_quant_kernel_param_8];
	or.b32 	%r168, %r164, 8192;
	or.b32 	%r169, %r162, 12288;
	or.b32 	%r170, %r162, 12289;
	or.b32 	%r171, %r162, 12290;
	or.b32 	%r172, %r162, 12291;
	or.b32 	%r173, %r162, 12292;
	or.b32 	%r174, %r162, 12293;
	or.b32 	%r175, %r162, 12294;
	or.b32 	%r176, %r162, 12295;
	or.b32 	%r177, %r162, 12296;
	or.b32 	%r178, %r162, 12297;
	or.b32 	%r179, %r162, 12298;
	or.b32 	%r180, %r162, 12299;
	or.b32 	%r181, %r162, 12300;
	or.b32 	%r182, %r162, 12301;
	or.b32 	%r183, %r162, 12302;
	or.b32 	%r184, %r162, 12303;
	.loc	1 71 18                         // sk09_norm_embed.py:71:18
	setp.lt.s32 	%p1, %r164, %r161;
	setp.lt.s32 	%p2, %r166, %r161;
	setp.lt.s32 	%p3, %r168, %r161;
	setp.lt.s32 	%p4, %r169, %r161;
	.loc	1 72 30                         // sk09_norm_embed.py:72:30
	mul.lo.s32 	%r185, %r163, %r156;
	.loc	1 72 24                         // sk09_norm_embed.py:72:24
	mad.wide.s32 	%rd91, %r185, 2, %rd86;
	.loc	1 72 42                         // sk09_norm_embed.py:72:42
	cvt.u64.u32 	%rd92, %r164;
	mul.wide.u32 	%rd93, %r164, 2;
	add.s64 	%rd1, %rd91, %rd93;
	add.s64 	%rd2, %rd1, 16;
	add.s64 	%rd3, %rd1, 8192;
	add.s64 	%rd4, %rd1, 8208;
	add.s64 	%rd5, %rd1, 16384;
	add.s64 	%rd6, %rd1, 16400;
	cvt.u64.u32 	%rd94, %r169;
	mul.wide.u32 	%rd95, %r169, 2;
	add.s64 	%rd7, %rd91, %rd95;
	mul.wide.u32 	%rd96, %r177, 2;
	add.s64 	%rd8, %rd91, %rd96;
	mov.b32 	%r5, 0;
	.loc	1 72 16                         // sk09_norm_embed.py:72:16
	// begin inline asm
	mov.u32 %r1, %r5;
	mov.u32 %r2, %r5;
	mov.u32 %r3, %r5;
	mov.u32 %r4, %r5;
	@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r6, %r5;
	mov.u32 %r7, %r5;
	mov.u32 %r8, %r5;
	mov.u32 %r9, %r5;
	@%p1 ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd2 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r10, %r5;
	mov.u32 %r11, %r5;
	mov.u32 %r12, %r5;
	mov.u32 %r13, %r5;
	@%p2 ld.global.v4.b32 { %r10, %r11, %r12, %r13 }, [ %rd3 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r14, %r5;
	mov.u32 %r15, %r5;
	mov.u32 %r16, %r5;
	mov.u32 %r17, %r5;
	@%p2 ld.global.v4.b32 { %r14, %r15, %r16, %r17 }, [ %rd4 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r18, %r5;
	mov.u32 %r19, %r5;
	mov.u32 %r20, %r5;
	mov.u32 %r21, %r5;
	@%p3 ld.global.v4.b32 { %r18, %r19, %r20, %r21 }, [ %rd5 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r22, %r5;
	mov.u32 %r23, %r5;
	mov.u32 %r24, %r5;
	mov.u32 %r25, %r5;
	@%p3 ld.global.v4.b32 { %r22, %r23, %r24, %r25 }, [ %rd6 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r26, %r5;
	mov.u32 %r27, %r5;
	mov.u32 %r28, %r5;
	mov.u32 %r29, %r5;
	@%p4 ld.global.v4.b32 { %r26, %r27, %r28, %r29 }, [ %rd7 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r30, %r5;
	mov.u32 %r31, %r5;
	mov.u32 %r32, %r5;
	mov.u32 %r33, %r5;
	@%p4 ld.global.v4.b32 { %r30, %r31, %r32, %r33 }, [ %rd8 + 0 ];
	// end inline asm
	.loc	1 72 73                         // sk09_norm_embed.py:72:73
	mov.b32 	{%rs1, %rs2}, %r2;
	cvt.f32.bf16 	%r186, %rs2;
	cvt.f32.bf16 	%r187, %rs1;
	mov.b32 	{%rs3, %rs4}, %r1;
	cvt.f32.bf16 	%r188, %rs3;
	cvt.f32.bf16 	%r189, %rs4;
	mov.b32 	{%rs5, %rs6}, %r4;
	cvt.f32.bf16 	%r190, %rs6;
	cvt.f32.bf16 	%r191, %rs5;
	mov.b32 	{%rs7, %rs8}, %r3;
	cvt.f32.bf16 	%r192, %rs8;
	cvt.f32.bf16 	%r193, %rs7;
	mov.b32 	{%rs9, %rs10}, %r7;
	cvt.f32.bf16 	%r194, %rs10;
	cvt.f32.bf16 	%r195, %rs9;
	mov.b32 	{%rs11, %rs12}, %r6;
	cvt.f32.bf16 	%r196, %rs12;
	cvt.f32.bf16 	%r197, %rs11;
	mov.b32 	{%rs13, %rs14}, %r9;
	cvt.f32.bf16 	%r198, %rs14;
	cvt.f32.bf16 	%r199, %rs13;
	mov.b32 	{%rs15, %rs16}, %r8;
	cvt.f32.bf16 	%r200, %rs16;
	cvt.f32.bf16 	%r201, %rs15;
	mov.b32 	{%rs17, %rs18}, %r11;
	cvt.f32.bf16 	%r202, %rs18;
	cvt.f32.bf16 	%r203, %rs17;
	mov.b32 	{%rs19, %rs20}, %r10;
	cvt.f32.bf16 	%r204, %rs20;
	cvt.f32.bf16 	%r205, %rs19;
	mov.b32 	{%rs21, %rs22}, %r13;
	cvt.f32.bf16 	%r206, %rs22;
	cvt.f32.bf16 	%r207, %rs21;
	mov.b32 	{%rs23, %rs24}, %r12;
	cvt.f32.bf16 	%r208, %rs24;
	cvt.f32.bf16 	%r209, %rs23;
	mov.b32 	{%rs25, %rs26}, %r15;
	cvt.f32.bf16 	%r210, %rs26;
	cvt.f32.bf16 	%r211, %rs25;
	mov.b32 	{%rs27, %rs28}, %r14;
	cvt.f32.bf16 	%r212, %rs28;
	cvt.f32.bf16 	%r213, %rs27;
	mov.b32 	{%rs29, %rs30}, %r17;
	cvt.f32.bf16 	%r214, %rs30;
	cvt.f32.bf16 	%r215, %rs29;
	mov.b32 	{%rs31, %rs32}, %r16;
	cvt.f32.bf16 	%r216, %rs32;
	cvt.f32.bf16 	%r217, %rs31;
	mov.b32 	{%rs33, %rs34}, %r19;
	cvt.f32.bf16 	%r218, %rs34;
	cvt.f32.bf16 	%r219, %rs33;
	mov.b32 	{%rs35, %rs36}, %r18;
	cvt.f32.bf16 	%r220, %rs36;
	cvt.f32.bf16 	%r221, %rs35;
	mov.b32 	{%rs37, %rs38}, %r21;
	cvt.f32.bf16 	%r222, %rs38;
	cvt.f32.bf16 	%r223, %rs37;
	mov.b32 	{%rs39, %rs40}, %r20;
	cvt.f32.bf16 	%r224, %rs40;
	cvt.f32.bf16 	%r225, %rs39;
	mov.b32 	{%rs41, %rs42}, %r23;
	cvt.f32.bf16 	%r226, %rs42;
	cvt.f32.bf16 	%r227, %rs41;
	mov.b32 	{%rs43, %rs44}, %r22;
	cvt.f32.bf16 	%r228, %rs44;
	cvt.f32.bf16 	%r229, %rs43;
	mov.b32 	{%rs45, %rs46}, %r25;
	cvt.f32.bf16 	%r230, %rs46;
	cvt.f32.bf16 	%r231, %rs45;
	mov.b32 	{%rs47, %rs48}, %r24;
	cvt.f32.bf16 	%r232, %rs48;
	cvt.f32.bf16 	%r233, %rs47;
	mov.b32 	{%rs49, %rs50}, %r27;
	cvt.f32.bf16 	%r234, %rs50;
	cvt.f32.bf16 	%r235, %rs49;
	mov.b32 	{%rs51, %rs52}, %r26;
	cvt.f32.bf16 	%r236, %rs52;
	cvt.f32.bf16 	%r237, %rs51;
	mov.b32 	{%rs53, %rs54}, %r29;
	cvt.f32.bf16 	%r238, %rs54;
	cvt.f32.bf16 	%r239, %rs53;
	mov.b32 	{%rs55, %rs56}, %r28;
	cvt.f32.bf16 	%r240, %rs56;
	cvt.f32.bf16 	%r241, %rs55;
	mov.b32 	{%rs57, %rs58}, %r31;
	cvt.f32.bf16 	%r242, %rs58;
	cvt.f32.bf16 	%r243, %rs57;
	mov.b32 	{%rs59, %rs60}, %r30;
	cvt.f32.bf16 	%r244, %rs60;
	cvt.f32.bf16 	%r245, %rs59;
	mov.b32 	{%rs61, %rs62}, %r33;
	cvt.f32.bf16 	%r246, %rs62;
	cvt.f32.bf16 	%r247, %rs61;
	mov.b32 	{%rs63, %rs64}, %r32;
	cvt.f32.bf16 	%r248, %rs64;
	cvt.f32.bf16 	%r249, %rs63;
	.loc	1 73 24                         // sk09_norm_embed.py:73:24
	add.s64 	%rd9, %rd87, %rd93;
	add.s64 	%rd10, %rd9, 16;
	add.s64 	%rd11, %rd9, 8192;
	add.s64 	%rd12, %rd9, 8208;
	add.s64 	%rd13, %rd9, 16384;
	add.s64 	%rd14, %rd9, 16400;
	add.s64 	%rd15, %rd87, %rd95;
	add.s64 	%rd16, %rd87, %rd96;
	.loc	1 73 16                         // sk09_norm_embed.py:73:16
	// begin inline asm
	mov.u32 %r34, %r5;
	mov.u32 %r35, %r5;
	mov.u32 %r36, %r5;
	mov.u32 %r37, %r5;
	@%p1 ld.global.v4.b32 { %r34, %r35, %r36, %r37 }, [ %rd9 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r38, %r5;
	mov.u32 %r39, %r5;
	mov.u32 %r40, %r5;
	mov.u32 %r41, %r5;
	@%p1 ld.global.v4.b32 { %r38, %r39, %r40, %r41 }, [ %rd10 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r42, %r5;
	mov.u32 %r43, %r5;
	mov.u32 %r44, %r5;
	mov.u32 %r45, %r5;
	@%p2 ld.global.v4.b32 { %r42, %r43, %r44, %r45 }, [ %rd11 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r46, %r5;
	mov.u32 %r47, %r5;
	mov.u32 %r48, %r5;
	mov.u32 %r49, %r5;
	@%p2 ld.global.v4.b32 { %r46, %r47, %r48, %r49 }, [ %rd12 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r50, %r5;
	mov.u32 %r51, %r5;
	mov.u32 %r52, %r5;
	mov.u32 %r53, %r5;
	@%p3 ld.global.v4.b32 { %r50, %r51, %r52, %r53 }, [ %rd13 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r54, %r5;
	mov.u32 %r55, %r5;
	mov.u32 %r56, %r5;
	mov.u32 %r57, %r5;
	@%p3 ld.global.v4.b32 { %r54, %r55, %r56, %r57 }, [ %rd14 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r58, %r5;
	mov.u32 %r59, %r5;
	mov.u32 %r60, %r5;
	mov.u32 %r61, %r5;
	@%p4 ld.global.v4.b32 { %r58, %r59, %r60, %r61 }, [ %rd15 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r62, %r5;
	mov.u32 %r63, %r5;
	mov.u32 %r64, %r5;
	mov.u32 %r65, %r5;
	@%p4 ld.global.v4.b32 { %r62, %r63, %r64, %r65 }, [ %rd16 + 0 ];
	// end inline asm
	.loc	1 74 40                         // sk09_norm_embed.py:74:40
	mul.lo.s32 	%r250, %r167, %r164;
	shl.b32 	%r251, %r167, 1;
	add.s32 	%r252, %r250, %r251;
	mad.lo.s32 	%r253, %r167, 3, %r250;
	shl.b32 	%r254, %r167, 2;
	add.s32 	%r255, %r250, %r254;
	mad.lo.s32 	%r256, %r167, 5, %r250;
	mad.lo.s32 	%r257, %r167, 6, %r250;
	mad.lo.s32 	%r258, %r167, 7, %r250;
	shl.b32 	%r259, %r167, 3;
	add.s32 	%r260, %r250, %r259;
	mad.lo.s32 	%r261, %r167, 9, %r250;
	mad.lo.s32 	%r262, %r167, 10, %r250;
	mad.lo.s32 	%r263, %r167, 11, %r250;
	mad.lo.s32 	%r264, %r167, 12, %r250;
	mad.lo.s32 	%r265, %r167, 13, %r250;
	mad.lo.s32 	%r266, %r167, 14, %r250;
	mad.lo.s32 	%r267, %r167, 15, %r250;
	shl.b32 	%r268, %r167, 12;
	add.s32 	%r269, %r250, %r268;
	mad.lo.s32 	%r270, %r167, 4097, %r250;
	mad.lo.s32 	%r271, %r167, 4098, %r250;
	mad.lo.s32 	%r272, %r167, 4099, %r250;
	mad.lo.s32 	%r273, %r167, 4100, %r250;
	mad.lo.s32 	%r274, %r167, 4101, %r250;
	mad.lo.s32 	%r275, %r167, 4102, %r250;
	mad.lo.s32 	%r276, %r167, 4103, %r250;
	mad.lo.s32 	%r277, %r167, 4104, %r250;
	mad.lo.s32 	%r278, %r167, 4105, %r250;
	mad.lo.s32 	%r279, %r167, 4106, %r250;
	mad.lo.s32 	%r280, %r167, 4107, %r250;
	mad.lo.s32 	%r281, %r167, 4108, %r250;
	mad.lo.s32 	%r282, %r167, 4109, %r250;
	mad.lo.s32 	%r283, %r167, 4110, %r250;
	mad.lo.s32 	%r284, %r167, 4111, %r250;
	shl.b32 	%r285, %r167, 13;
	add.s32 	%r286, %r250, %r285;
	mad.lo.s32 	%r287, %r167, 8193, %r250;
	mad.lo.s32 	%r288, %r167, 8194, %r250;
	mad.lo.s32 	%r289, %r167, 8195, %r250;
	mad.lo.s32 	%r290, %r167, 8196, %r250;
	mad.lo.s32 	%r291, %r167, 8197, %r250;
	mad.lo.s32 	%r292, %r167, 8198, %r250;
	mad.lo.s32 	%r293, %r167, 8199, %r250;
	mad.lo.s32 	%r294, %r167, 8200, %r250;
	mad.lo.s32 	%r295, %r167, 8201, %r250;
	mad.lo.s32 	%r296, %r167, 8202, %r250;
	mad.lo.s32 	%r297, %r167, 8203, %r250;
	mad.lo.s32 	%r298, %r167, 8204, %r250;
	mad.lo.s32 	%r299, %r167, 8205, %r250;
	mad.lo.s32 	%r300, %r167, 8206, %r250;
	mad.lo.s32 	%r301, %r167, 8207, %r250;
	mul.lo.s32 	%r302, %r167, %r169;
	mul.lo.s32 	%r303, %r167, %r170;
	mul.lo.s32 	%r304, %r167, %r171;
	mul.lo.s32 	%r305, %r167, %r172;
	mul.lo.s32 	%r306, %r167, %r173;
	mul.lo.s32 	%r307, %r167, %r174;
	mul.lo.s32 	%r308, %r167, %r175;
	mul.lo.s32 	%r309, %r167, %r176;
	mul.lo.s32 	%r310, %r167, %r177;
	mul.lo.s32 	%r311, %r167, %r178;
	mul.lo.s32 	%r312, %r167, %r179;
	mul.lo.s32 	%r313, %r167, %r180;
	mul.lo.s32 	%r314, %r167, %r181;
	mul.lo.s32 	%r315, %r167, %r182;
	mul.lo.s32 	%r316, %r167, %r183;
	mul.lo.s32 	%r317, %r167, %r184;
	.loc	1 74 33                         // sk09_norm_embed.py:74:33
	mad.wide.s32 	%rd17, %r250, 4, %rd88;
	mad.wide.s32 	%rd18, %r167, 4, %rd17;
	mad.wide.s32 	%rd19, %r252, 4, %rd88;
	mad.wide.s32 	%rd20, %r253, 4, %rd88;
	mad.wide.s32 	%rd21, %r255, 4, %rd88;
	mad.wide.s32 	%rd22, %r256, 4, %rd88;
	mad.wide.s32 	%rd23, %r257, 4, %rd88;
	mad.wide.s32 	%rd24, %r258, 4, %rd88;
	mad.wide.s32 	%rd25, %r260, 4, %rd88;
	mad.wide.s32 	%rd26, %r261, 4, %rd88;
	mad.wide.s32 	%rd27, %r262, 4, %rd88;
	mad.wide.s32 	%rd28, %r263, 4, %rd88;
	mad.wide.s32 	%rd29, %r264, 4, %rd88;
	mad.wide.s32 	%rd30, %r265, 4, %rd88;
	mad.wide.s32 	%rd31, %r266, 4, %rd88;
	mad.wide.s32 	%rd32, %r267, 4, %rd88;
	mad.wide.s32 	%rd33, %r269, 4, %rd88;
	mad.wide.s32 	%rd34, %r270, 4, %rd88;
	mad.wide.s32 	%rd35, %r271, 4, %rd88;
	mad.wide.s32 	%rd36, %r272, 4, %rd88;
	mad.wide.s32 	%rd37, %r273, 4, %rd88;
	mad.wide.s32 	%rd38, %r274, 4, %rd88;
	mad.wide.s32 	%rd39, %r275, 4, %rd88;
	mad.wide.s32 	%rd40, %r276, 4, %rd88;
	mad.wide.s32 	%rd41, %r277, 4, %rd88;
	mad.wide.s32 	%rd42, %r278, 4, %rd88;
	mad.wide.s32 	%rd43, %r279, 4, %rd88;
	mad.wide.s32 	%rd44, %r280, 4, %rd88;
	mad.wide.s32 	%rd45, %r281, 4, %rd88;
	mad.wide.s32 	%rd46, %r282, 4, %rd88;
	mad.wide.s32 	%rd47, %r283, 4, %rd88;
	mad.wide.s32 	%rd48, %r284, 4, %rd88;
	mad.wide.s32 	%rd49, %r286, 4, %rd88;
	mad.wide.s32 	%rd50, %r287, 4, %rd88;
	mad.wide.s32 	%rd51, %r288, 4, %rd88;
	mad.wide.s32 	%rd52, %r289, 4, %rd88;
	mad.wide.s32 	%rd53, %r290, 4, %rd88;
	mad.wide.s32 	%rd54, %r291, 4, %rd88;
	mad.wide.s32 	%rd55, %r292, 4, %rd88;
	mad.wide.s32 	%rd56, %r293, 4, %rd88;
	mad.wide.s32 	%rd57, %r294, 4, %rd88;
	mad.wide.s32 	%rd58, %r295, 4, %rd88;
	mad.wide.s32 	%rd59, %r296, 4, %rd88;
	mad.wide.s32 	%rd60, %r297, 4, %rd88;
	mad.wide.s32 	%rd61, %r298, 4, %rd88;
	mad.wide.s32 	%rd62, %r299, 4, %rd88;
	mad.wide.s32 	%rd63, %r300, 4, %rd88;
	mad.wide.s32 	%rd64, %r301, 4, %rd88;
	mad.wide.s32 	%rd65, %r302, 4, %rd88;
	mad.wide.s32 	%rd66, %r303, 4, %rd88;
	mad.wide.s32 	%rd67, %r304, 4, %rd88;
	mad.wide.s32 	%rd68, %r305, 4, %rd88;
	mad.wide.s32 	%rd69, %r306, 4, %rd88;
	mad.wide.s32 	%rd70, %r307, 4, %rd88;
	mad.wide.s32 	%rd71, %r308, 4, %rd88;
	mad.wide.s32 	%rd72, %r309, 4, %rd88;
	mad.wide.s32 	%rd73, %r310, 4, %rd88;
	mad.wide.s32 	%rd74, %r311, 4, %rd88;
	mad.wide.s32 	%rd75, %r312, 4, %rd88;
	mad.wide.s32 	%rd76, %r313, 4, %rd88;
	mad.wide.s32 	%rd77, %r314, 4, %rd88;
	mad.wide.s32 	%rd78, %r315, 4, %rd88;
	mad.wide.s32 	%rd79, %r316, 4, %rd88;
	mad.wide.s32 	%rd80, %r317, 4, %rd88;
	mov.b32 	%r67, 1065353216;
	.loc	1 74 20                         // sk09_norm_embed.py:74:20
	// begin inline asm
	mov.u32 %r66, %r67;
	@%p1 ld.global.b32 { %r66 }, [ %rd17 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r68, %r67;
	@%p1 ld.global.b32 { %r68 }, [ %rd18 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r69, %r67;
	@%p1 ld.global.b32 { %r69 }, [ %rd19 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r70, %r67;
	@%p1 ld.global.b32 { %r70 }, [ %rd20 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r71, %r67;
	@%p1 ld.global.b32 { %r71 }, [ %rd21 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r72, %r67;
	@%p1 ld.global.b32 { %r72 }, [ %rd22 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r73, %r67;
	@%p1 ld.global.b32 { %r73 }, [ %rd23 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r74, %r67;
	@%p1 ld.global.b32 { %r74 }, [ %rd24 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r75, %r67;
	@%p1 ld.global.b32 { %r75 }, [ %rd25 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r76, %r67;
	@%p1 ld.global.b32 { %r76 }, [ %rd26 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r77, %r67;
	@%p1 ld.global.b32 { %r77 }, [ %rd27 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r78, %r67;
	@%p1 ld.global.b32 { %r78 }, [ %rd28 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r79, %r67;
	@%p1 ld.global.b32 { %r79 }, [ %rd29 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r80, %r67;
	@%p1 ld.global.b32 { %r80 }, [ %rd30 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r81, %r67;
	@%p1 ld.global.b32 { %r81 }, [ %rd31 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r82, %r67;
	@%p1 ld.global.b32 { %r82 }, [ %rd32 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r83, %r67;
	@%p2 ld.global.b32 { %r83 }, [ %rd33 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r84, %r67;
	@%p2 ld.global.b32 { %r84 }, [ %rd34 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r85, %r67;
	@%p2 ld.global.b32 { %r85 }, [ %rd35 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r86, %r67;
	@%p2 ld.global.b32 { %r86 }, [ %rd36 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r87, %r67;
	@%p2 ld.global.b32 { %r87 }, [ %rd37 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r88, %r67;
	@%p2 ld.global.b32 { %r88 }, [ %rd38 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r89, %r67;
	@%p2 ld.global.b32 { %r89 }, [ %rd39 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r90, %r67;
	@%p2 ld.global.b32 { %r90 }, [ %rd40 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r91, %r67;
	@%p2 ld.global.b32 { %r91 }, [ %rd41 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r92, %r67;
	@%p2 ld.global.b32 { %r92 }, [ %rd42 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r93, %r67;
	@%p2 ld.global.b32 { %r93 }, [ %rd43 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r94, %r67;
	@%p2 ld.global.b32 { %r94 }, [ %rd44 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r95, %r67;
	@%p2 ld.global.b32 { %r95 }, [ %rd45 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r96, %r67;
	@%p2 ld.global.b32 { %r96 }, [ %rd46 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r97, %r67;
	@%p2 ld.global.b32 { %r97 }, [ %rd47 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r98, %r67;
	@%p2 ld.global.b32 { %r98 }, [ %rd48 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r99, %r67;
	@%p3 ld.global.b32 { %r99 }, [ %rd49 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r100, %r67;
	@%p3 ld.global.b32 { %r100 }, [ %rd50 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r101, %r67;
	@%p3 ld.global.b32 { %r101 }, [ %rd51 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r102, %r67;
	@%p3 ld.global.b32 { %r102 }, [ %rd52 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r103, %r67;
	@%p3 ld.global.b32 { %r103 }, [ %rd53 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r104, %r67;
	@%p3 ld.global.b32 { %r104 }, [ %rd54 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r105, %r67;
	@%p3 ld.global.b32 { %r105 }, [ %rd55 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r106, %r67;
	@%p3 ld.global.b32 { %r106 }, [ %rd56 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r107, %r67;
	@%p3 ld.global.b32 { %r107 }, [ %rd57 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r108, %r67;
	@%p3 ld.global.b32 { %r108 }, [ %rd58 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r109, %r67;
	@%p3 ld.global.b32 { %r109 }, [ %rd59 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r110, %r67;
	@%p3 ld.global.b32 { %r110 }, [ %rd60 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r111, %r67;
	@%p3 ld.global.b32 { %r111 }, [ %rd61 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r112, %r67;
	@%p3 ld.global.b32 { %r112 }, [ %rd62 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r113, %r67;
	@%p3 ld.global.b32 { %r113 }, [ %rd63 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r114, %r67;
	@%p3 ld.global.b32 { %r114 }, [ %rd64 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r115, %r67;
	@%p4 ld.global.b32 { %r115 }, [ %rd65 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r116, %r67;
	@%p4 ld.global.b32 { %r116 }, [ %rd66 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r117, %r67;
	@%p4 ld.global.b32 { %r117 }, [ %rd67 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r118, %r67;
	@%p4 ld.global.b32 { %r118 }, [ %rd68 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r119, %r67;
	@%p4 ld.global.b32 { %r119 }, [ %rd69 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r120, %r67;
	@%p4 ld.global.b32 { %r120 }, [ %rd70 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r121, %r67;
	@%p4 ld.global.b32 { %r121 }, [ %rd71 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r122, %r67;
	@%p4 ld.global.b32 { %r122 }, [ %rd72 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r123, %r67;
	@%p4 ld.global.b32 { %r123 }, [ %rd73 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r124, %r67;
	@%p4 ld.global.b32 { %r124 }, [ %rd74 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r125, %r67;
	@%p4 ld.global.b32 { %r125 }, [ %rd75 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r126, %r67;
	@%p4 ld.global.b32 { %r126 }, [ %rd76 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r127, %r67;
	@%p4 ld.global.b32 { %r127 }, [ %rd77 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r128, %r67;
	@%p4 ld.global.b32 { %r128 }, [ %rd78 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r129, %r67;
	@%p4 ld.global.b32 { %r129 }, [ %rd79 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r130, %r67;
	@%p4 ld.global.b32 { %r130 }, [ %rd80 + 0 ];
	// end inline asm
	.loc	1 75 32                         // sk09_norm_embed.py:75:32
	mul.f32 	%r318, %r189, %r189;
$L__tmp1:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	fma.rn.f32 	%r319, %r188, %r188, %r318;
	fma.rn.f32 	%r320, %r187, %r187, %r319;
	fma.rn.f32 	%r321, %r186, %r186, %r320;
	fma.rn.f32 	%r322, %r193, %r193, %r321;
	fma.rn.f32 	%r323, %r192, %r192, %r322;
	fma.rn.f32 	%r324, %r191, %r191, %r323;
	fma.rn.f32 	%r325, %r190, %r190, %r324;
	fma.rn.f32 	%r326, %r197, %r197, %r325;
	fma.rn.f32 	%r327, %r196, %r196, %r326;
	fma.rn.f32 	%r328, %r195, %r195, %r327;
	fma.rn.f32 	%r329, %r194, %r194, %r328;
	fma.rn.f32 	%r330, %r201, %r201, %r329;
	fma.rn.f32 	%r331, %r200, %r200, %r330;
	fma.rn.f32 	%r332, %r199, %r199, %r331;
	fma.rn.f32 	%r333, %r198, %r198, %r332;
	fma.rn.f32 	%r334, %r205, %r205, %r333;
	fma.rn.f32 	%r335, %r204, %r204, %r334;
	fma.rn.f32 	%r336, %r203, %r203, %r335;
	fma.rn.f32 	%r337, %r202, %r202, %r336;
	fma.rn.f32 	%r338, %r209, %r209, %r337;
	fma.rn.f32 	%r339, %r208, %r208, %r338;
	fma.rn.f32 	%r340, %r207, %r207, %r339;
	fma.rn.f32 	%r341, %r206, %r206, %r340;
	fma.rn.f32 	%r342, %r213, %r213, %r341;
	fma.rn.f32 	%r343, %r212, %r212, %r342;
	fma.rn.f32 	%r344, %r211, %r211, %r343;
	fma.rn.f32 	%r345, %r210, %r210, %r344;
	fma.rn.f32 	%r346, %r217, %r217, %r345;
	fma.rn.f32 	%r347, %r216, %r216, %r346;
	fma.rn.f32 	%r348, %r215, %r215, %r347;
	fma.rn.f32 	%r349, %r214, %r214, %r348;
	fma.rn.f32 	%r350, %r221, %r221, %r349;
	fma.rn.f32 	%r351, %r220, %r220, %r350;
	fma.rn.f32 	%r352, %r219, %r219, %r351;
	fma.rn.f32 	%r353, %r218, %r218, %r352;
	fma.rn.f32 	%r354, %r225, %r225, %r353;
	fma.rn.f32 	%r355, %r224, %r224, %r354;
	fma.rn.f32 	%r356, %r223, %r223, %r355;
	fma.rn.f32 	%r357, %r222, %r222, %r356;
	fma.rn.f32 	%r358, %r229, %r229, %r357;
	fma.rn.f32 	%r359, %r228, %r228, %r358;
	fma.rn.f32 	%r360, %r227, %r227, %r359;
	fma.rn.f32 	%r361, %r226, %r226, %r360;
	fma.rn.f32 	%r362, %r233, %r233, %r361;
	fma.rn.f32 	%r363, %r232, %r232, %r362;
	fma.rn.f32 	%r364, %r231, %r231, %r363;
	fma.rn.f32 	%r365, %r230, %r230, %r364;
	fma.rn.f32 	%r366, %r237, %r237, %r365;
	fma.rn.f32 	%r367, %r236, %r236, %r366;
	fma.rn.f32 	%r368, %r235, %r235, %r367;
	fma.rn.f32 	%r369, %r234, %r234, %r368;
	fma.rn.f32 	%r370, %r241, %r241, %r369;
	fma.rn.f32 	%r371, %r240, %r240, %r370;
	fma.rn.f32 	%r372, %r239, %r239, %r371;
	fma.rn.f32 	%r373, %r238, %r238, %r372;
	fma.rn.f32 	%r374, %r245, %r245, %r373;
	fma.rn.f32 	%r375, %r244, %r244, %r374;
	fma.rn.f32 	%r376, %r243, %r243, %r375;
	fma.rn.f32 	%r377, %r242, %r242, %r376;
	fma.rn.f32 	%r378, %r249, %r249, %r377;
	fma.rn.f32 	%r379, %r248, %r248, %r378;
	fma.rn.f32 	%r380, %r247, %r247, %r379;
	fma.rn.f32 	%r381, %r246, %r246, %r380;
$L__tmp2:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	shfl.sync.bfly.b32 	%r382, %r381, 16, 31, -1;
$L__tmp3:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	add.f32 	%r383, %r381, %r382;
$L__tmp4:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	shfl.sync.bfly.b32 	%r384, %r383, 8, 31, -1;
$L__tmp5:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	add.f32 	%r385, %r383, %r384;
$L__tmp6:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	shfl.sync.bfly.b32 	%r386, %r385, 4, 31, -1;
$L__tmp7:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	add.f32 	%r387, %r385, %r386;
$L__tmp8:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	shfl.sync.bfly.b32 	%r388, %r387, 2, 31, -1;
$L__tmp9:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	add.f32 	%r389, %r387, %r388;
$L__tmp10:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	shfl.sync.bfly.b32 	%r390, %r389, 1, 31, -1;
$L__tmp11:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	add.f32 	%r132, %r389, %r390;
$L__tmp12:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	setp.eq.b32 	%p5, %r159, 0;
	shr.u32 	%r391, %r157, 3;
	and.b32 	%r392, %r391, 28;
	mov.b32 	%r393, global_smem;
	add.s32 	%r131, %r393, %r392;
	// begin inline asm
	@%p5 st.shared.b32 [ %r131 + 0 ], %r132;
	// end inline asm
	bar.sync 	0;
	setp.lt.u32 	%p6, %r158, 8;
	shl.b32 	%r394, %r158, 2;
	add.s32 	%r134, %r393, %r394;
	// begin inline asm
	@%p6 ld.shared.b32 %r133, [ %r134 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r395, %r133, 4, 31, -1;
$L__tmp13:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	add.f32 	%r396, %r133, %r395;
$L__tmp14:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	shfl.sync.bfly.b32 	%r397, %r396, 2, 31, -1;
$L__tmp15:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	add.f32 	%r398, %r396, %r397;
$L__tmp16:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	shfl.sync.bfly.b32 	%r399, %r398, 1, 31, -1;
$L__tmp17:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:75:28 ] ]
	add.f32 	%r135, %r398, %r399;
$L__tmp18:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:75:28 ]
	and.b32 	%r400, %r157, 7;
	setp.eq.b32 	%p9, %r400, 0;
	and.pred 	%p7, %p6, %p9;
	// begin inline asm
	@%p7 st.shared.b32 [ %r134 + 0 ], %r135;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r401, [global_smem];
$L__tmp19:
	.loc	1 75 45                         // sk09_norm_embed.py:75:45
	cvt.rn.f32.s32 	%r402, %r161;
	div.full.f32 	%r403, %r401, %r402;
	.loc	1 75 49                         // sk09_norm_embed.py:75:49
	add.f32 	%r404, %r403, 0f358637BD;
	.loc	1 75 21                         // sk09_norm_embed.py:75:21
	rsqrt.approx.ftz.f32 	%r405, %r404;
	.loc	1 75 12                         // sk09_norm_embed.py:75:12
	mul.f32 	%r406, %r405, %r188;
	mul.f32 	%r407, %r405, %r189;
	mul.f32 	%r408, %r405, %r187;
	mul.f32 	%r409, %r405, %r186;
	mul.f32 	%r410, %r405, %r193;
	mul.f32 	%r411, %r405, %r192;
	mul.f32 	%r412, %r405, %r191;
	mul.f32 	%r413, %r405, %r190;
	mul.f32 	%r414, %r405, %r197;
	mul.f32 	%r415, %r405, %r196;
	mul.f32 	%r416, %r405, %r195;
	mul.f32 	%r417, %r405, %r194;
	mul.f32 	%r418, %r405, %r201;
	mul.f32 	%r419, %r405, %r200;
	mul.f32 	%r420, %r405, %r199;
	mul.f32 	%r421, %r405, %r198;
	mul.f32 	%r422, %r405, %r205;
	mul.f32 	%r423, %r405, %r204;
	mul.f32 	%r424, %r405, %r203;
	mul.f32 	%r425, %r405, %r202;
	mul.f32 	%r426, %r405, %r209;
	mul.f32 	%r427, %r405, %r208;
	mul.f32 	%r428, %r405, %r207;
	mul.f32 	%r429, %r405, %r206;
	mul.f32 	%r430, %r405, %r213;
	mul.f32 	%r431, %r405, %r212;
	mul.f32 	%r432, %r405, %r211;
	mul.f32 	%r433, %r405, %r210;
	mul.f32 	%r434, %r405, %r217;
	mul.f32 	%r435, %r405, %r216;
	mul.f32 	%r436, %r405, %r215;
	mul.f32 	%r437, %r405, %r214;
	mul.f32 	%r438, %r405, %r221;
	mul.f32 	%r439, %r405, %r220;
	mul.f32 	%r440, %r405, %r219;
	mul.f32 	%r441, %r405, %r218;
	mul.f32 	%r442, %r405, %r225;
	mul.f32 	%r443, %r405, %r224;
	mul.f32 	%r444, %r405, %r223;
	mul.f32 	%r445, %r405, %r222;
	mul.f32 	%r446, %r405, %r229;
	mul.f32 	%r447, %r405, %r228;
	mul.f32 	%r448, %r405, %r227;
	mul.f32 	%r449, %r405, %r226;
	mul.f32 	%r450, %r405, %r233;
	mul.f32 	%r451, %r405, %r232;
	mul.f32 	%r452, %r405, %r231;
	mul.f32 	%r453, %r405, %r230;
	mul.f32 	%r454, %r405, %r237;
	mul.f32 	%r455, %r405, %r236;
	mul.f32 	%r456, %r405, %r235;
	mul.f32 	%r457, %r405, %r234;
	mul.f32 	%r458, %r405, %r241;
	mul.f32 	%r459, %r405, %r240;
	mul.f32 	%r460, %r405, %r239;
	mul.f32 	%r461, %r405, %r238;
	mul.f32 	%r462, %r405, %r245;
	mul.f32 	%r463, %r405, %r244;
	mul.f32 	%r464, %r405, %r243;
	mul.f32 	%r465, %r405, %r242;
	mul.f32 	%r466, %r405, %r249;
	mul.f32 	%r467, %r405, %r248;
	mul.f32 	%r468, %r405, %r247;
	mul.f32 	%r469, %r405, %r246;
$L__tmp20:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	bar.sync 	0;
$L__tmp21:
	.loc	1 79 27                         // sk09_norm_embed.py:79:27
	mul.lo.s32 	%r470, %r165, %r156;
	.loc	1 79 21                         // sk09_norm_embed.py:79:21
	cvt.s64.s32 	%rd97, %r470;
	add.s64 	%rd98, %rd89, %rd97;
	.loc	1 79 39                         // sk09_norm_embed.py:79:39
	add.s64 	%rd81, %rd98, %rd92;
	add.s64 	%rd82, %rd81, 4096;
	add.s64 	%rd83, %rd81, 8192;
	add.s64 	%rd84, %rd98, %rd94;
	.loc	1 73 55                         // sk09_norm_embed.py:73:55
	mov.b32 	{%rs65, %rs66}, %r40;
	cvt.f32.bf16 	%r471, %rs65;
	cvt.f32.bf16 	%r472, %rs66;
	mov.b32 	{%rs67, %rs68}, %r41;
	cvt.f32.bf16 	%r473, %rs67;
	cvt.f32.bf16 	%r474, %rs68;
	.loc	1 73 69                         // sk09_norm_embed.py:73:69
	add.f32 	%r475, %r474, 0f00000000;
	add.f32 	%r476, %r473, 0f00000000;
	add.f32 	%r477, %r472, 0f00000000;
	add.f32 	%r478, %r471, 0f00000000;
	.loc	1 74 12                         // sk09_norm_embed.py:74:12
	mul.f32 	%r479, %r478, %r79;
	mul.f32 	%r480, %r477, %r80;
	mul.f32 	%r481, %r476, %r81;
	mul.f32 	%r482, %r475, %r82;
	.loc	1 75 56                         // sk09_norm_embed.py:75:56
	mul.f32 	%r483, %r482, %r421;
	mul.f32 	%r484, %r481, %r420;
	mul.f32 	%r485, %r480, %r419;
	mul.f32 	%r486, %r479, %r418;
	.loc	1 76 36                         // sk09_norm_embed.py:76:36
	abs.f32 	%r487, %r486;
	abs.f32 	%r488, %r485;
	abs.f32 	%r489, %r484;
	abs.f32 	%r490, %r483;
	.loc	1 73 55                         // sk09_norm_embed.py:73:55
	mov.b32 	{%rs69, %rs70}, %r38;
	cvt.f32.bf16 	%r491, %rs69;
	cvt.f32.bf16 	%r492, %rs70;
	mov.b32 	{%rs71, %rs72}, %r39;
	cvt.f32.bf16 	%r493, %rs71;
	cvt.f32.bf16 	%r494, %rs72;
	.loc	1 73 69                         // sk09_norm_embed.py:73:69
	add.f32 	%r495, %r494, 0f00000000;
	add.f32 	%r496, %r493, 0f00000000;
	add.f32 	%r497, %r492, 0f00000000;
	add.f32 	%r498, %r491, 0f00000000;
	.loc	1 74 12                         // sk09_norm_embed.py:74:12
	mul.f32 	%r499, %r498, %r75;
	mul.f32 	%r500, %r497, %r76;
	mul.f32 	%r501, %r496, %r77;
	mul.f32 	%r502, %r495, %r78;
	.loc	1 75 56                         // sk09_norm_embed.py:75:56
	mul.f32 	%r503, %r502, %r417;
	mul.f32 	%r504, %r501, %r416;
	mul.f32 	%r505, %r500, %r415;
	mul.f32 	%r506, %r499, %r414;
	.loc	1 76 36                         // sk09_norm_embed.py:76:36
	abs.f32 	%r507, %r506;
	abs.f32 	%r508, %r505;
	abs.f32 	%r509, %r504;
	abs.f32 	%r510, %r503;
	.loc	1 73 55                         // sk09_norm_embed.py:73:55
	mov.b32 	{%rs73, %rs74}, %r36;
	cvt.f32.bf16 	%r511, %rs73;
	cvt.f32.bf16 	%r512, %rs74;
	mov.b32 	{%rs75, %rs76}, %r37;
	cvt.f32.bf16 	%r513, %rs75;
	cvt.f32.bf16 	%r514, %rs76;
	.loc	1 73 69                         // sk09_norm_embed.py:73:69
	add.f32 	%r515, %r514, 0f00000000;
	add.f32 	%r516, %r513, 0f00000000;
	add.f32 	%r517, %r512, 0f00000000;
	add.f32 	%r518, %r511, 0f00000000;
	.loc	1 74 12                         // sk09_norm_embed.py:74:12
	mul.f32 	%r519, %r518, %r71;
	mul.f32 	%r520, %r517, %r72;
	mul.f32 	%r521, %r516, %r73;
	mul.f32 	%r522, %r515, %r74;
	.loc	1 75 56                         // sk09_norm_embed.py:75:56
	mul.f32 	%r523, %r522, %r413;
	mul.f32 	%r524, %r521, %r412;
	mul.f32 	%r525, %r520, %r411;
	mul.f32 	%r526, %r519, %r410;
	.loc	1 76 36                         // sk09_norm_embed.py:76:36
	abs.f32 	%r527, %r526;
	abs.f32 	%r528, %r525;
	abs.f32 	%r529, %r524;
	abs.f32 	%r530, %r523;
	.loc	1 73 55                         // sk09_norm_embed.py:73:55
	mov.b32 	{%rs77, %rs78}, %r34;
	cvt.f32.bf16 	%r531, %rs77;
	cvt.f32.bf16 	%r532, %rs78;
	mov.b32 	{%rs79, %rs80}, %r35;
	cvt.f32.bf16 	%r533, %rs79;
	cvt.f32.bf16 	%r534, %rs80;
	.loc	1 73 69                         // sk09_norm_embed.py:73:69
	add.f32 	%r535, %r534, 0f00000000;
	add.f32 	%r536, %r533, 0f00000000;
	add.f32 	%r537, %r532, 0f00000000;
	add.f32 	%r538, %r531, 0f00000000;
	.loc	1 74 12                         // sk09_norm_embed.py:74:12
	mul.f32 	%r539, %r538, %r66;
	mul.f32 	%r540, %r537, %r68;
	mul.f32 	%r541, %r536, %r69;
	mul.f32 	%r542, %r535, %r70;
	.loc	1 75 56                         // sk09_norm_embed.py:75:56
	mul.f32 	%r543, %r542, %r409;
	mul.f32 	%r544, %r541, %r408;
	mul.f32 	%r545, %r540, %r407;
	mul.f32 	%r546, %r539, %r406;
	.loc	1 76 36                         // sk09_norm_embed.py:76:36
	abs.f32 	%r547, %r546;
	abs.f32 	%r548, %r545;
	abs.f32 	%r549, %r544;
	abs.f32 	%r550, %r543;
$L__tmp22:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r551, %r547, %r548;
	max.f32 	%r552, %r551, %r549;
	max.f32 	%r553, %r552, %r550;
	max.f32 	%r554, %r553, %r527;
	max.f32 	%r555, %r554, %r528;
	max.f32 	%r556, %r555, %r529;
	max.f32 	%r557, %r556, %r530;
	max.f32 	%r558, %r557, %r507;
	max.f32 	%r559, %r558, %r508;
	max.f32 	%r560, %r559, %r509;
	max.f32 	%r561, %r560, %r510;
	max.f32 	%r562, %r561, %r487;
	max.f32 	%r563, %r562, %r488;
	max.f32 	%r564, %r563, %r489;
	max.f32 	%r565, %r564, %r490;
$L__tmp23:
	.loc	1 73 55                         // sk09_norm_embed.py:73:55
	mov.b32 	{%rs81, %rs82}, %r48;
	cvt.f32.bf16 	%r566, %rs81;
	cvt.f32.bf16 	%r567, %rs82;
	mov.b32 	{%rs83, %rs84}, %r49;
	cvt.f32.bf16 	%r568, %rs83;
	cvt.f32.bf16 	%r569, %rs84;
	.loc	1 73 69                         // sk09_norm_embed.py:73:69
	add.f32 	%r570, %r569, 0f00000000;
	add.f32 	%r571, %r568, 0f00000000;
	add.f32 	%r572, %r567, 0f00000000;
	add.f32 	%r573, %r566, 0f00000000;
	.loc	1 74 12                         // sk09_norm_embed.py:74:12
	mul.f32 	%r574, %r573, %r95;
	mul.f32 	%r575, %r572, %r96;
	mul.f32 	%r576, %r571, %r97;
	mul.f32 	%r577, %r570, %r98;
	.loc	1 75 56                         // sk09_norm_embed.py:75:56
	mul.f32 	%r578, %r577, %r437;
	mul.f32 	%r579, %r576, %r436;
	mul.f32 	%r580, %r575, %r435;
	mul.f32 	%r581, %r574, %r434;
	.loc	1 76 36                         // sk09_norm_embed.py:76:36
	abs.f32 	%r582, %r581;
	abs.f32 	%r583, %r580;
	abs.f32 	%r584, %r579;
	abs.f32 	%r585, %r578;
	.loc	1 73 55                         // sk09_norm_embed.py:73:55
	mov.b32 	{%rs85, %rs86}, %r46;
	cvt.f32.bf16 	%r586, %rs85;
	cvt.f32.bf16 	%r587, %rs86;
	mov.b32 	{%rs87, %rs88}, %r47;
	cvt.f32.bf16 	%r588, %rs87;
	cvt.f32.bf16 	%r589, %rs88;
	.loc	1 73 69                         // sk09_norm_embed.py:73:69
	add.f32 	%r590, %r589, 0f00000000;
	add.f32 	%r591, %r588, 0f00000000;
	add.f32 	%r592, %r587, 0f00000000;
	add.f32 	%r593, %r586, 0f00000000;
	.loc	1 74 12                         // sk09_norm_embed.py:74:12
	mul.f32 	%r594, %r593, %r91;
	mul.f32 	%r595, %r592, %r92;
	mul.f32 	%r596, %r591, %r93;
	mul.f32 	%r597, %r590, %r94;
	.loc	1 75 56                         // sk09_norm_embed.py:75:56
	mul.f32 	%r598, %r597, %r433;
	mul.f32 	%r599, %r596, %r432;
	mul.f32 	%r600, %r595, %r431;
	mul.f32 	%r601, %r594, %r430;
	.loc	1 76 36                         // sk09_norm_embed.py:76:36
	abs.f32 	%r602, %r601;
	abs.f32 	%r603, %r600;
	abs.f32 	%r604, %r599;
	abs.f32 	%r605, %r598;
	.loc	1 73 55                         // sk09_norm_embed.py:73:55
	mov.b32 	{%rs89, %rs90}, %r44;
	cvt.f32.bf16 	%r606, %rs89;
	cvt.f32.bf16 	%r607, %rs90;
	mov.b32 	{%rs91, %rs92}, %r45;
	cvt.f32.bf16 	%r608, %rs91;
	cvt.f32.bf16 	%r609, %rs92;
	.loc	1 73 69                         // sk09_norm_embed.py:73:69
	add.f32 	%r610, %r609, 0f00000000;
	add.f32 	%r611, %r608, 0f00000000;
	add.f32 	%r612, %r607, 0f00000000;
	add.f32 	%r613, %r606, 0f00000000;
	.loc	1 74 12                         // sk09_norm_embed.py:74:12
	mul.f32 	%r614, %r613, %r87;
	mul.f32 	%r615, %r612, %r88;
	mul.f32 	%r616, %r611, %r89;
	mul.f32 	%r617, %r610, %r90;
	.loc	1 75 56                         // sk09_norm_embed.py:75:56
	mul.f32 	%r618, %r617, %r429;
	mul.f32 	%r619, %r616, %r428;
	mul.f32 	%r620, %r615, %r427;
	mul.f32 	%r621, %r614, %r426;
	.loc	1 76 36                         // sk09_norm_embed.py:76:36
	abs.f32 	%r622, %r621;
	abs.f32 	%r623, %r620;
	abs.f32 	%r624, %r619;
	abs.f32 	%r625, %r618;
	.loc	1 73 55                         // sk09_norm_embed.py:73:55
	mov.b32 	{%rs93, %rs94}, %r42;
	cvt.f32.bf16 	%r626, %rs93;
	cvt.f32.bf16 	%r627, %rs94;
	mov.b32 	{%rs95, %rs96}, %r43;
	cvt.f32.bf16 	%r628, %rs95;
	cvt.f32.bf16 	%r629, %rs96;
	.loc	1 73 69                         // sk09_norm_embed.py:73:69
	add.f32 	%r630, %r629, 0f00000000;
	add.f32 	%r631, %r628, 0f00000000;
	add.f32 	%r632, %r627, 0f00000000;
	add.f32 	%r633, %r626, 0f00000000;
	.loc	1 74 12                         // sk09_norm_embed.py:74:12
	mul.f32 	%r634, %r633, %r83;
	mul.f32 	%r635, %r632, %r84;
	mul.f32 	%r636, %r631, %r85;
	mul.f32 	%r637, %r630, %r86;
	.loc	1 75 56                         // sk09_norm_embed.py:75:56
	mul.f32 	%r638, %r637, %r425;
	mul.f32 	%r639, %r636, %r424;
	mul.f32 	%r640, %r635, %r423;
	mul.f32 	%r641, %r634, %r422;
	.loc	1 76 36                         // sk09_norm_embed.py:76:36
	abs.f32 	%r642, %r641;
	abs.f32 	%r643, %r640;
	abs.f32 	%r644, %r639;
	abs.f32 	%r645, %r638;
$L__tmp24:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r646, %r565, %r642;
	max.f32 	%r647, %r646, %r643;
	max.f32 	%r648, %r647, %r644;
	max.f32 	%r649, %r648, %r645;
	max.f32 	%r650, %r649, %r622;
	max.f32 	%r651, %r650, %r623;
	max.f32 	%r652, %r651, %r624;
	max.f32 	%r653, %r652, %r625;
	max.f32 	%r654, %r653, %r602;
	max.f32 	%r655, %r654, %r603;
	max.f32 	%r656, %r655, %r604;
	max.f32 	%r657, %r656, %r605;
	max.f32 	%r658, %r657, %r582;
	max.f32 	%r659, %r658, %r583;
	max.f32 	%r660, %r659, %r584;
	max.f32 	%r661, %r660, %r585;
$L__tmp25:
	.loc	1 73 55                         // sk09_norm_embed.py:73:55
	mov.b32 	{%rs97, %rs98}, %r56;
	cvt.f32.bf16 	%r662, %rs97;
	cvt.f32.bf16 	%r663, %rs98;
	mov.b32 	{%rs99, %rs100}, %r57;
	cvt.f32.bf16 	%r664, %rs99;
	cvt.f32.bf16 	%r665, %rs100;
	.loc	1 73 69                         // sk09_norm_embed.py:73:69
	add.f32 	%r666, %r665, 0f00000000;
	add.f32 	%r667, %r664, 0f00000000;
	add.f32 	%r668, %r663, 0f00000000;
	add.f32 	%r669, %r662, 0f00000000;
	.loc	1 74 12                         // sk09_norm_embed.py:74:12
	mul.f32 	%r670, %r669, %r111;
	mul.f32 	%r671, %r668, %r112;
	mul.f32 	%r672, %r667, %r113;
	mul.f32 	%r673, %r666, %r114;
	.loc	1 75 56                         // sk09_norm_embed.py:75:56
	mul.f32 	%r674, %r673, %r453;
	mul.f32 	%r675, %r672, %r452;
	mul.f32 	%r676, %r671, %r451;
	mul.f32 	%r677, %r670, %r450;
	.loc	1 76 36                         // sk09_norm_embed.py:76:36
	abs.f32 	%r678, %r677;
	abs.f32 	%r679, %r676;
	abs.f32 	%r680, %r675;
	abs.f32 	%r681, %r674;
	.loc	1 73 55                         // sk09_norm_embed.py:73:55
	mov.b32 	{%rs101, %rs102}, %r54;
	cvt.f32.bf16 	%r682, %rs101;
	cvt.f32.bf16 	%r683, %rs102;
	mov.b32 	{%rs103, %rs104}, %r55;
	cvt.f32.bf16 	%r684, %rs103;
	cvt.f32.bf16 	%r685, %rs104;
	.loc	1 73 69                         // sk09_norm_embed.py:73:69
	add.f32 	%r686, %r685, 0f00000000;
	add.f32 	%r687, %r684, 0f00000000;
	add.f32 	%r688, %r683, 0f00000000;
	add.f32 	%r689, %r682, 0f00000000;
	.loc	1 74 12                         // sk09_norm_embed.py:74:12
	mul.f32 	%r690, %r689, %r107;
	mul.f32 	%r691, %r688, %r108;
	mul.f32 	%r692, %r687, %r109;
	mul.f32 	%r693, %r686, %r110;
	.loc	1 75 56                         // sk09_norm_embed.py:75:56
	mul.f32 	%r694, %r693, %r449;
	mul.f32 	%r695, %r692, %r448;
	mul.f32 	%r696, %r691, %r447;
	mul.f32 	%r697, %r690, %r446;
	.loc	1 76 36                         // sk09_norm_embed.py:76:36
	abs.f32 	%r698, %r697;
	abs.f32 	%r699, %r696;
	abs.f32 	%r700, %r695;
	abs.f32 	%r701, %r694;
	.loc	1 73 55                         // sk09_norm_embed.py:73:55
	mov.b32 	{%rs105, %rs106}, %r52;
	cvt.f32.bf16 	%r702, %rs105;
	cvt.f32.bf16 	%r703, %rs106;
	mov.b32 	{%rs107, %rs108}, %r53;
	cvt.f32.bf16 	%r704, %rs107;
	cvt.f32.bf16 	%r705, %rs108;
	.loc	1 73 69                         // sk09_norm_embed.py:73:69
	add.f32 	%r706, %r705, 0f00000000;
	add.f32 	%r707, %r704, 0f00000000;
	add.f32 	%r708, %r703, 0f00000000;
	add.f32 	%r709, %r702, 0f00000000;
	.loc	1 74 12                         // sk09_norm_embed.py:74:12
	mul.f32 	%r710, %r709, %r103;
	mul.f32 	%r711, %r708, %r104;
	mul.f32 	%r712, %r707, %r105;
	mul.f32 	%r713, %r706, %r106;
	.loc	1 75 56                         // sk09_norm_embed.py:75:56
	mul.f32 	%r714, %r713, %r445;
	mul.f32 	%r715, %r712, %r444;
	mul.f32 	%r716, %r711, %r443;
	mul.f32 	%r717, %r710, %r442;
	.loc	1 76 36                         // sk09_norm_embed.py:76:36
	abs.f32 	%r718, %r717;
	abs.f32 	%r719, %r716;
	abs.f32 	%r720, %r715;
	abs.f32 	%r721, %r714;
	.loc	1 73 55                         // sk09_norm_embed.py:73:55
	mov.b32 	{%rs109, %rs110}, %r50;
	cvt.f32.bf16 	%r722, %rs109;
	cvt.f32.bf16 	%r723, %rs110;
	mov.b32 	{%rs111, %rs112}, %r51;
	cvt.f32.bf16 	%r724, %rs111;
	cvt.f32.bf16 	%r725, %rs112;
	.loc	1 73 69                         // sk09_norm_embed.py:73:69
	add.f32 	%r726, %r725, 0f00000000;
	add.f32 	%r727, %r724, 0f00000000;
	add.f32 	%r728, %r723, 0f00000000;
	add.f32 	%r729, %r722, 0f00000000;
	.loc	1 74 12                         // sk09_norm_embed.py:74:12
	mul.f32 	%r730, %r729, %r99;
	mul.f32 	%r731, %r728, %r100;
	mul.f32 	%r732, %r727, %r101;
	mul.f32 	%r733, %r726, %r102;
	.loc	1 75 56                         // sk09_norm_embed.py:75:56
	mul.f32 	%r734, %r733, %r441;
	mul.f32 	%r735, %r732, %r440;
	mul.f32 	%r736, %r731, %r439;
	mul.f32 	%r737, %r730, %r438;
	.loc	1 76 36                         // sk09_norm_embed.py:76:36
	abs.f32 	%r738, %r737;
	abs.f32 	%r739, %r736;
	abs.f32 	%r740, %r735;
	abs.f32 	%r741, %r734;
$L__tmp26:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r742, %r661, %r738;
	max.f32 	%r743, %r742, %r739;
	max.f32 	%r744, %r743, %r740;
	max.f32 	%r745, %r744, %r741;
	max.f32 	%r746, %r745, %r718;
	max.f32 	%r747, %r746, %r719;
	max.f32 	%r748, %r747, %r720;
	max.f32 	%r749, %r748, %r721;
	max.f32 	%r750, %r749, %r698;
	max.f32 	%r751, %r750, %r699;
	max.f32 	%r752, %r751, %r700;
	max.f32 	%r753, %r752, %r701;
	max.f32 	%r754, %r753, %r678;
	max.f32 	%r755, %r754, %r679;
	max.f32 	%r756, %r755, %r680;
	max.f32 	%r757, %r756, %r681;
$L__tmp27:
	.loc	1 73 55                         // sk09_norm_embed.py:73:55
	mov.b32 	{%rs113, %rs114}, %r64;
	cvt.f32.bf16 	%r758, %rs113;
	cvt.f32.bf16 	%r759, %rs114;
	mov.b32 	{%rs115, %rs116}, %r65;
	cvt.f32.bf16 	%r760, %rs115;
	cvt.f32.bf16 	%r761, %rs116;
	.loc	1 73 69                         // sk09_norm_embed.py:73:69
	add.f32 	%r762, %r761, 0f00000000;
	add.f32 	%r763, %r760, 0f00000000;
	add.f32 	%r764, %r759, 0f00000000;
	add.f32 	%r765, %r758, 0f00000000;
	.loc	1 74 12                         // sk09_norm_embed.py:74:12
	mul.f32 	%r766, %r765, %r127;
	mul.f32 	%r767, %r764, %r128;
	mul.f32 	%r768, %r763, %r129;
	mul.f32 	%r769, %r762, %r130;
	.loc	1 75 56                         // sk09_norm_embed.py:75:56
	mul.f32 	%r770, %r769, %r469;
	mul.f32 	%r771, %r768, %r468;
	mul.f32 	%r772, %r767, %r467;
	mul.f32 	%r773, %r766, %r466;
	.loc	1 76 36                         // sk09_norm_embed.py:76:36
	abs.f32 	%r774, %r773;
	abs.f32 	%r775, %r772;
	abs.f32 	%r776, %r771;
	abs.f32 	%r777, %r770;
	.loc	1 73 55                         // sk09_norm_embed.py:73:55
	mov.b32 	{%rs117, %rs118}, %r62;
	cvt.f32.bf16 	%r778, %rs117;
	cvt.f32.bf16 	%r779, %rs118;
	mov.b32 	{%rs119, %rs120}, %r63;
	cvt.f32.bf16 	%r780, %rs119;
	cvt.f32.bf16 	%r781, %rs120;
	.loc	1 73 69                         // sk09_norm_embed.py:73:69
	add.f32 	%r782, %r781, 0f00000000;
	add.f32 	%r783, %r780, 0f00000000;
	add.f32 	%r784, %r779, 0f00000000;
	add.f32 	%r785, %r778, 0f00000000;
	.loc	1 74 12                         // sk09_norm_embed.py:74:12
	mul.f32 	%r786, %r785, %r123;
	mul.f32 	%r787, %r784, %r124;
	mul.f32 	%r788, %r783, %r125;
	mul.f32 	%r789, %r782, %r126;
	.loc	1 75 56                         // sk09_norm_embed.py:75:56
	mul.f32 	%r790, %r789, %r465;
	mul.f32 	%r791, %r788, %r464;
	mul.f32 	%r792, %r787, %r463;
	mul.f32 	%r793, %r786, %r462;
	.loc	1 76 36                         // sk09_norm_embed.py:76:36
	abs.f32 	%r794, %r793;
	abs.f32 	%r795, %r792;
	abs.f32 	%r796, %r791;
	abs.f32 	%r797, %r790;
	.loc	1 73 55                         // sk09_norm_embed.py:73:55
	mov.b32 	{%rs121, %rs122}, %r60;
	cvt.f32.bf16 	%r798, %rs121;
	cvt.f32.bf16 	%r799, %rs122;
	mov.b32 	{%rs123, %rs124}, %r61;
	cvt.f32.bf16 	%r800, %rs123;
	cvt.f32.bf16 	%r801, %rs124;
	.loc	1 73 69                         // sk09_norm_embed.py:73:69
	add.f32 	%r802, %r801, 0f00000000;
	add.f32 	%r803, %r800, 0f00000000;
	add.f32 	%r804, %r799, 0f00000000;
	add.f32 	%r805, %r798, 0f00000000;
	.loc	1 74 12                         // sk09_norm_embed.py:74:12
	mul.f32 	%r806, %r805, %r119;
	mul.f32 	%r807, %r804, %r120;
	mul.f32 	%r808, %r803, %r121;
	mul.f32 	%r809, %r802, %r122;
	.loc	1 75 56                         // sk09_norm_embed.py:75:56
	mul.f32 	%r810, %r809, %r461;
	mul.f32 	%r811, %r808, %r460;
	mul.f32 	%r812, %r807, %r459;
	mul.f32 	%r813, %r806, %r458;
	.loc	1 76 36                         // sk09_norm_embed.py:76:36
	abs.f32 	%r814, %r813;
	abs.f32 	%r815, %r812;
	abs.f32 	%r816, %r811;
	abs.f32 	%r817, %r810;
	.loc	1 73 55                         // sk09_norm_embed.py:73:55
	mov.b32 	{%rs125, %rs126}, %r58;
	cvt.f32.bf16 	%r818, %rs125;
	cvt.f32.bf16 	%r819, %rs126;
	mov.b32 	{%rs127, %rs128}, %r59;
	cvt.f32.bf16 	%r820, %rs127;
	cvt.f32.bf16 	%r821, %rs128;
	.loc	1 73 69                         // sk09_norm_embed.py:73:69
	add.f32 	%r822, %r821, 0f00000000;
	add.f32 	%r823, %r820, 0f00000000;
	add.f32 	%r824, %r819, 0f00000000;
	add.f32 	%r825, %r818, 0f00000000;
	.loc	1 74 12                         // sk09_norm_embed.py:74:12
	mul.f32 	%r826, %r825, %r115;
	mul.f32 	%r827, %r824, %r116;
	mul.f32 	%r828, %r823, %r117;
	mul.f32 	%r829, %r822, %r118;
	.loc	1 75 56                         // sk09_norm_embed.py:75:56
	mul.f32 	%r830, %r829, %r457;
	mul.f32 	%r831, %r828, %r456;
	mul.f32 	%r832, %r827, %r455;
	mul.f32 	%r833, %r826, %r454;
	.loc	1 76 36                         // sk09_norm_embed.py:76:36
	abs.f32 	%r834, %r833;
	abs.f32 	%r835, %r832;
	abs.f32 	%r836, %r831;
	abs.f32 	%r837, %r830;
$L__tmp28:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r838, %r757, %r834;
	max.f32 	%r839, %r838, %r835;
	max.f32 	%r840, %r839, %r836;
	max.f32 	%r841, %r840, %r837;
	max.f32 	%r842, %r841, %r814;
	max.f32 	%r843, %r842, %r815;
	max.f32 	%r844, %r843, %r816;
	max.f32 	%r845, %r844, %r817;
	max.f32 	%r846, %r845, %r794;
	max.f32 	%r847, %r846, %r795;
	max.f32 	%r848, %r847, %r796;
	max.f32 	%r849, %r848, %r797;
	max.f32 	%r850, %r849, %r774;
	max.f32 	%r851, %r850, %r775;
	max.f32 	%r852, %r851, %r776;
	max.f32 	%r853, %r852, %r777;
$L__tmp29:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	shfl.sync.bfly.b32 	%r854, %r853, 16, 31, -1;
$L__tmp30:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r855, %r853, %r854;
$L__tmp31:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	shfl.sync.bfly.b32 	%r856, %r855, 8, 31, -1;
$L__tmp32:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r857, %r855, %r856;
$L__tmp33:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	shfl.sync.bfly.b32 	%r858, %r857, 4, 31, -1;
$L__tmp34:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r859, %r857, %r858;
$L__tmp35:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	shfl.sync.bfly.b32 	%r860, %r859, 2, 31, -1;
$L__tmp36:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r861, %r859, %r860;
$L__tmp37:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	shfl.sync.bfly.b32 	%r862, %r861, 1, 31, -1;
$L__tmp38:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r136, %r861, %r862;
$L__tmp39:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	// begin inline asm
	@%p5 st.shared.b32 [ %r131 + 0 ], %r136;
	// end inline asm
	bar.sync 	0;
	// begin inline asm
	@%p6 ld.shared.b32 %r137, [ %r134 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r863, %r137, 4, 31, -1;
$L__tmp40:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r864, %r137, %r863;
$L__tmp41:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	shfl.sync.bfly.b32 	%r865, %r864, 2, 31, -1;
$L__tmp42:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r866, %r864, %r865;
$L__tmp43:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	shfl.sync.bfly.b32 	%r867, %r866, 1, 31, -1;
$L__tmp44:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:76:29 ] ]
	max.f32 	%r138, %r866, %r867;
$L__tmp45:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:76:29 ]
	// begin inline asm
	@%p7 st.shared.b32 [ %r134 + 0 ], %r138;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r868, [global_smem];
$L__tmp46:
	.loc	1 76 49                         // sk09_norm_embed.py:76:49
	max.f32 	%r869, %r868, 0f0DA24260;
	mov.b32 	%r870, 0f42FE0000;
	.loc	1 77 22                         // sk09_norm_embed.py:77:22
	div.full.f32 	%r871, %r870, %r869;
	.loc	1 77 14                         // sk09_norm_embed.py:77:14
	mul.f32 	%r872, %r545, %r871;
	mul.f32 	%r873, %r546, %r871;
	mul.f32 	%r874, %r543, %r871;
	mul.f32 	%r875, %r544, %r871;
	mul.f32 	%r876, %r525, %r871;
	mul.f32 	%r877, %r526, %r871;
	mul.f32 	%r878, %r523, %r871;
	mul.f32 	%r879, %r524, %r871;
	mul.f32 	%r880, %r505, %r871;
	mul.f32 	%r881, %r506, %r871;
	mul.f32 	%r882, %r503, %r871;
	mul.f32 	%r883, %r504, %r871;
	mul.f32 	%r884, %r485, %r871;
	mul.f32 	%r885, %r486, %r871;
	mul.f32 	%r886, %r483, %r871;
	mul.f32 	%r887, %r484, %r871;
	mul.f32 	%r888, %r640, %r871;
	mul.f32 	%r889, %r641, %r871;
	mul.f32 	%r890, %r638, %r871;
	mul.f32 	%r891, %r639, %r871;
	mul.f32 	%r892, %r620, %r871;
	mul.f32 	%r893, %r621, %r871;
	mul.f32 	%r894, %r618, %r871;
	mul.f32 	%r895, %r619, %r871;
	mul.f32 	%r896, %r600, %r871;
	mul.f32 	%r897, %r601, %r871;
	mul.f32 	%r898, %r598, %r871;
	mul.f32 	%r899, %r599, %r871;
	mul.f32 	%r900, %r580, %r871;
	mul.f32 	%r901, %r581, %r871;
	mul.f32 	%r902, %r578, %r871;
	mul.f32 	%r903, %r579, %r871;
	mul.f32 	%r904, %r736, %r871;
	mul.f32 	%r905, %r737, %r871;
	mul.f32 	%r906, %r734, %r871;
	mul.f32 	%r907, %r735, %r871;
	mul.f32 	%r908, %r716, %r871;
	mul.f32 	%r909, %r717, %r871;
	mul.f32 	%r910, %r714, %r871;
	mul.f32 	%r911, %r715, %r871;
	mul.f32 	%r912, %r696, %r871;
	mul.f32 	%r913, %r697, %r871;
	mul.f32 	%r914, %r694, %r871;
	mul.f32 	%r915, %r695, %r871;
	mul.f32 	%r916, %r676, %r871;
	mul.f32 	%r917, %r677, %r871;
	mul.f32 	%r918, %r674, %r871;
	mul.f32 	%r919, %r675, %r871;
	mul.f32 	%r920, %r832, %r871;
	mul.f32 	%r921, %r833, %r871;
	mul.f32 	%r922, %r830, %r871;
	mul.f32 	%r923, %r831, %r871;
	mul.f32 	%r924, %r812, %r871;
	mul.f32 	%r925, %r813, %r871;
	mul.f32 	%r926, %r810, %r871;
	mul.f32 	%r927, %r811, %r871;
	mul.f32 	%r928, %r792, %r871;
	mul.f32 	%r929, %r793, %r871;
	mul.f32 	%r930, %r790, %r871;
	mul.f32 	%r931, %r791, %r871;
	mul.f32 	%r932, %r772, %r871;
	mul.f32 	%r933, %r773, %r871;
	mul.f32 	%r934, %r770, %r871;
	mul.f32 	%r935, %r771, %r871;
	.loc	1 78 29                         // sk09_norm_embed.py:78:29
	.loc	1 78 39                         // sk09_norm_embed.py:78:39
	lop3.b32 	%r936, 0x3f000000, %r872, 0x80000000, 0xF8;
	lop3.b32 	%r937, 0x3f000000, %r873, 0x80000000, 0xF8;
	lop3.b32 	%r938, 0x3f000000, %r874, 0x80000000, 0xF8;
	lop3.b32 	%r939, 0x3f000000, %r875, 0x80000000, 0xF8;
	lop3.b32 	%r940, 0x3f000000, %r876, 0x80000000, 0xF8;
	lop3.b32 	%r941, 0x3f000000, %r877, 0x80000000, 0xF8;
	lop3.b32 	%r942, 0x3f000000, %r878, 0x80000000, 0xF8;
	lop3.b32 	%r943, 0x3f000000, %r879, 0x80000000, 0xF8;
	lop3.b32 	%r944, 0x3f000000, %r880, 0x80000000, 0xF8;
	lop3.b32 	%r945, 0x3f000000, %r881, 0x80000000, 0xF8;
	lop3.b32 	%r946, 0x3f000000, %r882, 0x80000000, 0xF8;
	lop3.b32 	%r947, 0x3f000000, %r883, 0x80000000, 0xF8;
	lop3.b32 	%r948, 0x3f000000, %r884, 0x80000000, 0xF8;
	lop3.b32 	%r949, 0x3f000000, %r885, 0x80000000, 0xF8;
	lop3.b32 	%r950, 0x3f000000, %r886, 0x80000000, 0xF8;
	lop3.b32 	%r951, 0x3f000000, %r887, 0x80000000, 0xF8;
	lop3.b32 	%r952, 0x3f000000, %r888, 0x80000000, 0xF8;
	lop3.b32 	%r953, 0x3f000000, %r889, 0x80000000, 0xF8;
	lop3.b32 	%r954, 0x3f000000, %r890, 0x80000000, 0xF8;
	lop3.b32 	%r955, 0x3f000000, %r891, 0x80000000, 0xF8;
	lop3.b32 	%r956, 0x3f000000, %r892, 0x80000000, 0xF8;
	lop3.b32 	%r957, 0x3f000000, %r893, 0x80000000, 0xF8;
	lop3.b32 	%r958, 0x3f000000, %r894, 0x80000000, 0xF8;
	lop3.b32 	%r959, 0x3f000000, %r895, 0x80000000, 0xF8;
	lop3.b32 	%r960, 0x3f000000, %r896, 0x80000000, 0xF8;
	lop3.b32 	%r961, 0x3f000000, %r897, 0x80000000, 0xF8;
	lop3.b32 	%r962, 0x3f000000, %r898, 0x80000000, 0xF8;
	lop3.b32 	%r963, 0x3f000000, %r899, 0x80000000, 0xF8;
	lop3.b32 	%r964, 0x3f000000, %r900, 0x80000000, 0xF8;
	lop3.b32 	%r965, 0x3f000000, %r901, 0x80000000, 0xF8;
	lop3.b32 	%r966, 0x3f000000, %r902, 0x80000000, 0xF8;
	lop3.b32 	%r967, 0x3f000000, %r903, 0x80000000, 0xF8;
	lop3.b32 	%r968, 0x3f000000, %r904, 0x80000000, 0xF8;
	lop3.b32 	%r969, 0x3f000000, %r905, 0x80000000, 0xF8;
	lop3.b32 	%r970, 0x3f000000, %r906, 0x80000000, 0xF8;
	lop3.b32 	%r971, 0x3f000000, %r907, 0x80000000, 0xF8;
	lop3.b32 	%r972, 0x3f000000, %r908, 0x80000000, 0xF8;
	lop3.b32 	%r973, 0x3f000000, %r909, 0x80000000, 0xF8;
	lop3.b32 	%r974, 0x3f000000, %r910, 0x80000000, 0xF8;
	lop3.b32 	%r975, 0x3f000000, %r911, 0x80000000, 0xF8;
	lop3.b32 	%r976, 0x3f000000, %r912, 0x80000000, 0xF8;
	lop3.b32 	%r977, 0x3f000000, %r913, 0x80000000, 0xF8;
	lop3.b32 	%r978, 0x3f000000, %r914, 0x80000000, 0xF8;
	lop3.b32 	%r979, 0x3f000000, %r915, 0x80000000, 0xF8;
	lop3.b32 	%r980, 0x3f000000, %r916, 0x80000000, 0xF8;
	lop3.b32 	%r981, 0x3f000000, %r917, 0x80000000, 0xF8;
	lop3.b32 	%r982, 0x3f000000, %r918, 0x80000000, 0xF8;
	lop3.b32 	%r983, 0x3f000000, %r919, 0x80000000, 0xF8;
	lop3.b32 	%r984, 0x3f000000, %r920, 0x80000000, 0xF8;
	lop3.b32 	%r985, 0x3f000000, %r921, 0x80000000, 0xF8;
	lop3.b32 	%r986, 0x3f000000, %r922, 0x80000000, 0xF8;
	lop3.b32 	%r987, 0x3f000000, %r923, 0x80000000, 0xF8;
	lop3.b32 	%r988, 0x3f000000, %r924, 0x80000000, 0xF8;
	lop3.b32 	%r989, 0x3f000000, %r925, 0x80000000, 0xF8;
	lop3.b32 	%r990, 0x3f000000, %r926, 0x80000000, 0xF8;
	lop3.b32 	%r991, 0x3f000000, %r927, 0x80000000, 0xF8;
	lop3.b32 	%r992, 0x3f000000, %r928, 0x80000000, 0xF8;
	lop3.b32 	%r993, 0x3f000000, %r929, 0x80000000, 0xF8;
	lop3.b32 	%r994, 0x3f000000, %r930, 0x80000000, 0xF8;
	lop3.b32 	%r995, 0x3f000000, %r931, 0x80000000, 0xF8;
	lop3.b32 	%r996, 0x3f000000, %r932, 0x80000000, 0xF8;
	lop3.b32 	%r997, 0x3f000000, %r933, 0x80000000, 0xF8;
	lop3.b32 	%r998, 0x3f000000, %r934, 0x80000000, 0xF8;
	lop3.b32 	%r999, 0x3f000000, %r935, 0x80000000, 0xF8;
	.loc	1 78 14                         // sk09_norm_embed.py:78:14
	fma.rn.f32 	%r1000, %r544, %r871, %r939;
	fma.rn.f32 	%r1001, %r543, %r871, %r938;
	fma.rn.f32 	%r1002, %r546, %r871, %r937;
	fma.rn.f32 	%r1003, %r545, %r871, %r936;
	fma.rn.f32 	%r1004, %r524, %r871, %r943;
	fma.rn.f32 	%r1005, %r523, %r871, %r942;
	fma.rn.f32 	%r1006, %r526, %r871, %r941;
	fma.rn.f32 	%r1007, %r525, %r871, %r940;
	fma.rn.f32 	%r1008, %r504, %r871, %r947;
	fma.rn.f32 	%r1009, %r503, %r871, %r946;
	fma.rn.f32 	%r1010, %r506, %r871, %r945;
	fma.rn.f32 	%r1011, %r505, %r871, %r944;
	fma.rn.f32 	%r1012, %r484, %r871, %r951;
	fma.rn.f32 	%r1013, %r483, %r871, %r950;
	fma.rn.f32 	%r1014, %r486, %r871, %r949;
	fma.rn.f32 	%r1015, %r485, %r871, %r948;
	fma.rn.f32 	%r1016, %r639, %r871, %r955;
	fma.rn.f32 	%r1017, %r638, %r871, %r954;
	fma.rn.f32 	%r1018, %r641, %r871, %r953;
	fma.rn.f32 	%r1019, %r640, %r871, %r952;
	fma.rn.f32 	%r1020, %r619, %r871, %r959;
	fma.rn.f32 	%r1021, %r618, %r871, %r958;
	fma.rn.f32 	%r1022, %r621, %r871, %r957;
	fma.rn.f32 	%r1023, %r620, %r871, %r956;
	fma.rn.f32 	%r1024, %r599, %r871, %r963;
	fma.rn.f32 	%r1025, %r598, %r871, %r962;
	fma.rn.f32 	%r1026, %r601, %r871, %r961;
	fma.rn.f32 	%r1027, %r600, %r871, %r960;
	fma.rn.f32 	%r1028, %r579, %r871, %r967;
	fma.rn.f32 	%r1029, %r578, %r871, %r966;
	fma.rn.f32 	%r1030, %r581, %r871, %r965;
	fma.rn.f32 	%r1031, %r580, %r871, %r964;
	fma.rn.f32 	%r1032, %r735, %r871, %r971;
	fma.rn.f32 	%r1033, %r734, %r871, %r970;
	fma.rn.f32 	%r1034, %r737, %r871, %r969;
	fma.rn.f32 	%r1035, %r736, %r871, %r968;
	fma.rn.f32 	%r1036, %r715, %r871, %r975;
	fma.rn.f32 	%r1037, %r714, %r871, %r974;
	fma.rn.f32 	%r1038, %r717, %r871, %r973;
	fma.rn.f32 	%r1039, %r716, %r871, %r972;
	fma.rn.f32 	%r1040, %r695, %r871, %r979;
	fma.rn.f32 	%r1041, %r694, %r871, %r978;
	fma.rn.f32 	%r1042, %r697, %r871, %r977;
	fma.rn.f32 	%r1043, %r696, %r871, %r976;
	fma.rn.f32 	%r1044, %r675, %r871, %r983;
	fma.rn.f32 	%r1045, %r674, %r871, %r982;
	fma.rn.f32 	%r1046, %r677, %r871, %r981;
	fma.rn.f32 	%r1047, %r676, %r871, %r980;
	fma.rn.f32 	%r1048, %r831, %r871, %r987;
	fma.rn.f32 	%r1049, %r830, %r871, %r986;
	fma.rn.f32 	%r1050, %r833, %r871, %r985;
	fma.rn.f32 	%r1051, %r832, %r871, %r984;
	fma.rn.f32 	%r1052, %r811, %r871, %r991;
	fma.rn.f32 	%r1053, %r810, %r871, %r990;
	fma.rn.f32 	%r1054, %r813, %r871, %r989;
	fma.rn.f32 	%r1055, %r812, %r871, %r988;
	fma.rn.f32 	%r1056, %r791, %r871, %r995;
	fma.rn.f32 	%r1057, %r790, %r871, %r994;
	fma.rn.f32 	%r1058, %r793, %r871, %r993;
	fma.rn.f32 	%r1059, %r792, %r871, %r992;
	fma.rn.f32 	%r1060, %r771, %r871, %r999;
	fma.rn.f32 	%r1061, %r770, %r871, %r998;
	fma.rn.f32 	%r1062, %r773, %r871, %r997;
	fma.rn.f32 	%r1063, %r772, %r871, %r996;
	.loc	1 78 49                         // sk09_norm_embed.py:78:49
	cvt.rzi.s32.f32 	%r1064, %r1003;
	cvt.rzi.s32.f32 	%r1065, %r1002;
	cvt.rzi.s32.f32 	%r1066, %r1001;
	cvt.rzi.s32.f32 	%r1067, %r1000;
	cvt.rzi.s32.f32 	%r1068, %r1007;
	cvt.rzi.s32.f32 	%r1069, %r1006;
	cvt.rzi.s32.f32 	%r1070, %r1005;
	cvt.rzi.s32.f32 	%r1071, %r1004;
	cvt.rzi.s32.f32 	%r1072, %r1011;
	cvt.rzi.s32.f32 	%r1073, %r1010;
	cvt.rzi.s32.f32 	%r1074, %r1009;
	cvt.rzi.s32.f32 	%r1075, %r1008;
	cvt.rzi.s32.f32 	%r1076, %r1015;
	cvt.rzi.s32.f32 	%r1077, %r1014;
	cvt.rzi.s32.f32 	%r1078, %r1013;
	cvt.rzi.s32.f32 	%r1079, %r1012;
	cvt.rzi.s32.f32 	%r1080, %r1019;
	cvt.rzi.s32.f32 	%r1081, %r1018;
	cvt.rzi.s32.f32 	%r1082, %r1017;
	cvt.rzi.s32.f32 	%r1083, %r1016;
	cvt.rzi.s32.f32 	%r1084, %r1023;
	cvt.rzi.s32.f32 	%r1085, %r1022;
	cvt.rzi.s32.f32 	%r1086, %r1021;
	cvt.rzi.s32.f32 	%r1087, %r1020;
	cvt.rzi.s32.f32 	%r1088, %r1027;
	cvt.rzi.s32.f32 	%r1089, %r1026;
	cvt.rzi.s32.f32 	%r1090, %r1025;
	cvt.rzi.s32.f32 	%r1091, %r1024;
	cvt.rzi.s32.f32 	%r1092, %r1031;
	cvt.rzi.s32.f32 	%r1093, %r1030;
	cvt.rzi.s32.f32 	%r1094, %r1029;
	cvt.rzi.s32.f32 	%r1095, %r1028;
	cvt.rzi.s32.f32 	%r1096, %r1035;
	cvt.rzi.s32.f32 	%r1097, %r1034;
	cvt.rzi.s32.f32 	%r1098, %r1033;
	cvt.rzi.s32.f32 	%r1099, %r1032;
	cvt.rzi.s32.f32 	%r1100, %r1039;
	cvt.rzi.s32.f32 	%r1101, %r1038;
	cvt.rzi.s32.f32 	%r1102, %r1037;
	cvt.rzi.s32.f32 	%r1103, %r1036;
	cvt.rzi.s32.f32 	%r1104, %r1043;
	cvt.rzi.s32.f32 	%r1105, %r1042;
	cvt.rzi.s32.f32 	%r1106, %r1041;
	cvt.rzi.s32.f32 	%r1107, %r1040;
	cvt.rzi.s32.f32 	%r1108, %r1047;
	cvt.rzi.s32.f32 	%r1109, %r1046;
	cvt.rzi.s32.f32 	%r1110, %r1045;
	cvt.rzi.s32.f32 	%r1111, %r1044;
	cvt.rzi.s32.f32 	%r1112, %r1051;
	cvt.rzi.s32.f32 	%r1113, %r1050;
	cvt.rzi.s32.f32 	%r1114, %r1049;
	cvt.rzi.s32.f32 	%r1115, %r1048;
	cvt.rzi.s32.f32 	%r1116, %r1055;
	cvt.rzi.s32.f32 	%r1117, %r1054;
	cvt.rzi.s32.f32 	%r1118, %r1053;
	cvt.rzi.s32.f32 	%r1119, %r1052;
	cvt.rzi.s32.f32 	%r1120, %r1059;
	cvt.rzi.s32.f32 	%r1121, %r1058;
	cvt.rzi.s32.f32 	%r1122, %r1057;
	cvt.rzi.s32.f32 	%r1123, %r1056;
	cvt.rzi.s32.f32 	%r1124, %r1063;
	cvt.rzi.s32.f32 	%r1125, %r1062;
	cvt.rzi.s32.f32 	%r1126, %r1061;
	cvt.rzi.s32.f32 	%r1127, %r1060;
	.loc	1 79 70                         // sk09_norm_embed.py:79:70
	max.s32 	%r1128, %r1067, -127;
	max.s32 	%r1129, %r1066, -127;
	max.s32 	%r1130, %r1065, -127;
	max.s32 	%r1131, %r1064, -127;
	max.s32 	%r1132, %r1071, -127;
	max.s32 	%r1133, %r1070, -127;
	max.s32 	%r1134, %r1069, -127;
	max.s32 	%r1135, %r1068, -127;
	max.s32 	%r1136, %r1075, -127;
	max.s32 	%r1137, %r1074, -127;
	max.s32 	%r1138, %r1073, -127;
	max.s32 	%r1139, %r1072, -127;
	max.s32 	%r1140, %r1079, -127;
	max.s32 	%r1141, %r1078, -127;
	max.s32 	%r1142, %r1077, -127;
	max.s32 	%r1143, %r1076, -127;
	max.s32 	%r1144, %r1083, -127;
	max.s32 	%r1145, %r1082, -127;
	max.s32 	%r1146, %r1081, -127;
	max.s32 	%r1147, %r1080, -127;
	max.s32 	%r1148, %r1087, -127;
	max.s32 	%r1149, %r1086, -127;
	max.s32 	%r1150, %r1085, -127;
	max.s32 	%r1151, %r1084, -127;
	max.s32 	%r1152, %r1091, -127;
	max.s32 	%r1153, %r1090, -127;
	max.s32 	%r1154, %r1089, -127;
	max.s32 	%r1155, %r1088, -127;
	max.s32 	%r1156, %r1095, -127;
	max.s32 	%r1157, %r1094, -127;
	max.s32 	%r1158, %r1093, -127;
	max.s32 	%r1159, %r1092, -127;
	max.s32 	%r1160, %r1099, -127;
	max.s32 	%r1161, %r1098, -127;
	max.s32 	%r1162, %r1097, -127;
	max.s32 	%r1163, %r1096, -127;
	max.s32 	%r1164, %r1103, -127;
	max.s32 	%r1165, %r1102, -127;
	max.s32 	%r1166, %r1101, -127;
	max.s32 	%r1167, %r1100, -127;
	max.s32 	%r1168, %r1107, -127;
	max.s32 	%r1169, %r1106, -127;
	max.s32 	%r1170, %r1105, -127;
	max.s32 	%r1171, %r1104, -127;
	max.s32 	%r1172, %r1111, -127;
	max.s32 	%r1173, %r1110, -127;
	max.s32 	%r1174, %r1109, -127;
	max.s32 	%r1175, %r1108, -127;
	max.s32 	%r1176, %r1115, -127;
	max.s32 	%r1177, %r1114, -127;
	max.s32 	%r1178, %r1113, -127;
	max.s32 	%r1179, %r1112, -127;
	max.s32 	%r1180, %r1119, -127;
	max.s32 	%r1181, %r1118, -127;
	max.s32 	%r1182, %r1117, -127;
	max.s32 	%r1183, %r1116, -127;
	max.s32 	%r1184, %r1123, -127;
	max.s32 	%r1185, %r1122, -127;
	max.s32 	%r1186, %r1121, -127;
	max.s32 	%r1187, %r1120, -127;
	max.s32 	%r1188, %r1127, -127;
	max.s32 	%r1189, %r1126, -127;
	max.s32 	%r1190, %r1125, -127;
	max.s32 	%r1191, %r1124, -127;
	.loc	1 79 77                         // sk09_norm_embed.py:79:77
	min.s32 	%r1192, %r1131, 127;
	min.s32 	%r1193, %r1130, 127;
	min.s32 	%r1194, %r1129, 127;
	min.s32 	%r1195, %r1128, 127;
	min.s32 	%r1196, %r1135, 127;
	min.s32 	%r1197, %r1134, 127;
	min.s32 	%r1198, %r1133, 127;
	min.s32 	%r1199, %r1132, 127;
	min.s32 	%r1200, %r1139, 127;
	min.s32 	%r1201, %r1138, 127;
	min.s32 	%r1202, %r1137, 127;
	min.s32 	%r1203, %r1136, 127;
	min.s32 	%r1204, %r1143, 127;
	min.s32 	%r1205, %r1142, 127;
	min.s32 	%r1206, %r1141, 127;
	min.s32 	%r1207, %r1140, 127;
	min.s32 	%r1208, %r1147, 127;
	min.s32 	%r1209, %r1146, 127;
	min.s32 	%r1210, %r1145, 127;
	min.s32 	%r1211, %r1144, 127;
	min.s32 	%r1212, %r1151, 127;
	min.s32 	%r1213, %r1150, 127;
	min.s32 	%r1214, %r1149, 127;
	min.s32 	%r1215, %r1148, 127;
	min.s32 	%r1216, %r1155, 127;
	min.s32 	%r1217, %r1154, 127;
	min.s32 	%r1218, %r1153, 127;
	min.s32 	%r1219, %r1152, 127;
	min.s32 	%r1220, %r1159, 127;
	min.s32 	%r1221, %r1158, 127;
	min.s32 	%r1222, %r1157, 127;
	min.s32 	%r1223, %r1156, 127;
	min.s32 	%r1224, %r1163, 127;
	min.s32 	%r1225, %r1162, 127;
	min.s32 	%r1226, %r1161, 127;
	min.s32 	%r1227, %r1160, 127;
	min.s32 	%r1228, %r1167, 127;
	min.s32 	%r1229, %r1166, 127;
	min.s32 	%r1230, %r1165, 127;
	min.s32 	%r1231, %r1164, 127;
	min.s32 	%r1232, %r1171, 127;
	min.s32 	%r1233, %r1170, 127;
	min.s32 	%r1234, %r1169, 127;
	min.s32 	%r1235, %r1168, 127;
	min.s32 	%r1236, %r1175, 127;
	min.s32 	%r1237, %r1174, 127;
	min.s32 	%r1238, %r1173, 127;
	min.s32 	%r1239, %r1172, 127;
	min.s32 	%r1240, %r1179, 127;
	min.s32 	%r1241, %r1178, 127;
	min.s32 	%r1242, %r1177, 127;
	min.s32 	%r1243, %r1176, 127;
	min.s32 	%r1244, %r1183, 127;
	min.s32 	%r1245, %r1182, 127;
	min.s32 	%r1246, %r1181, 127;
	min.s32 	%r1247, %r1180, 127;
	min.s32 	%r1248, %r1187, 127;
	min.s32 	%r1249, %r1186, 127;
	min.s32 	%r1250, %r1185, 127;
	min.s32 	%r1251, %r1184, 127;
	min.s32 	%r1252, %r1191, 127;
	min.s32 	%r1253, %r1190, 127;
	min.s32 	%r1254, %r1189, 127;
	min.s32 	%r1255, %r1188, 127;
	.loc	1 79 85                         // sk09_norm_embed.py:79:85
	prmt.b32 	%r1256, %r1195, %r1194, 0x3340U;
	prmt.b32 	%r1257, %r1193, %r1192, 0x3340U;
	prmt.b32 	%r139, %r1257, %r1256, 0x5410U;
	prmt.b32 	%r1258, %r1199, %r1198, 0x3340U;
	prmt.b32 	%r1259, %r1197, %r1196, 0x3340U;
	prmt.b32 	%r140, %r1259, %r1258, 0x5410U;
	prmt.b32 	%r1260, %r1203, %r1202, 0x3340U;
	prmt.b32 	%r1261, %r1201, %r1200, 0x3340U;
	prmt.b32 	%r141, %r1261, %r1260, 0x5410U;
	prmt.b32 	%r1262, %r1207, %r1206, 0x3340U;
	prmt.b32 	%r1263, %r1205, %r1204, 0x3340U;
	prmt.b32 	%r142, %r1263, %r1262, 0x5410U;
	prmt.b32 	%r1264, %r1211, %r1210, 0x3340U;
	prmt.b32 	%r1265, %r1209, %r1208, 0x3340U;
	prmt.b32 	%r143, %r1265, %r1264, 0x5410U;
	prmt.b32 	%r1266, %r1215, %r1214, 0x3340U;
	prmt.b32 	%r1267, %r1213, %r1212, 0x3340U;
	prmt.b32 	%r144, %r1267, %r1266, 0x5410U;
	prmt.b32 	%r1268, %r1219, %r1218, 0x3340U;
	prmt.b32 	%r1269, %r1217, %r1216, 0x3340U;
	prmt.b32 	%r145, %r1269, %r1268, 0x5410U;
	prmt.b32 	%r1270, %r1223, %r1222, 0x3340U;
	prmt.b32 	%r1271, %r1221, %r1220, 0x3340U;
	prmt.b32 	%r146, %r1271, %r1270, 0x5410U;
	prmt.b32 	%r1272, %r1227, %r1226, 0x3340U;
	prmt.b32 	%r1273, %r1225, %r1224, 0x3340U;
	prmt.b32 	%r147, %r1273, %r1272, 0x5410U;
	prmt.b32 	%r1274, %r1231, %r1230, 0x3340U;
	prmt.b32 	%r1275, %r1229, %r1228, 0x3340U;
	prmt.b32 	%r148, %r1275, %r1274, 0x5410U;
	prmt.b32 	%r1276, %r1235, %r1234, 0x3340U;
	prmt.b32 	%r1277, %r1233, %r1232, 0x3340U;
	prmt.b32 	%r149, %r1277, %r1276, 0x5410U;
	prmt.b32 	%r1278, %r1239, %r1238, 0x3340U;
	prmt.b32 	%r1279, %r1237, %r1236, 0x3340U;
	prmt.b32 	%r150, %r1279, %r1278, 0x5410U;
	prmt.b32 	%r1280, %r1243, %r1242, 0x3340U;
	prmt.b32 	%r1281, %r1241, %r1240, 0x3340U;
	prmt.b32 	%r151, %r1281, %r1280, 0x5410U;
	prmt.b32 	%r1282, %r1247, %r1246, 0x3340U;
	prmt.b32 	%r1283, %r1245, %r1244, 0x3340U;
	prmt.b32 	%r152, %r1283, %r1282, 0x5410U;
	prmt.b32 	%r1284, %r1251, %r1250, 0x3340U;
	prmt.b32 	%r1285, %r1249, %r1248, 0x3340U;
	prmt.b32 	%r153, %r1285, %r1284, 0x5410U;
	prmt.b32 	%r1286, %r1255, %r1254, 0x3340U;
	prmt.b32 	%r1287, %r1253, %r1252, 0x3340U;
	prmt.b32 	%r154, %r1287, %r1286, 0x5410U;
	.loc	1 79 45                         // sk09_norm_embed.py:79:45
	// begin inline asm
	@%p1 st.global.v4.b32 [ %rd81 + 0 ], { %r139, %r140, %r141, %r142 };
	// end inline asm
	// begin inline asm
	@%p2 st.global.v4.b32 [ %rd82 + 0 ], { %r143, %r144, %r145, %r146 };
	// end inline asm
	// begin inline asm
	@%p3 st.global.v4.b32 [ %rd83 + 0 ], { %r147, %r148, %r149, %r150 };
	// end inline asm
	// begin inline asm
	@%p4 st.global.v4.b32 [ %rd84 + 0 ], { %r151, %r152, %r153, %r154 };
	// end inline asm
	.loc	1 80 21                         // sk09_norm_embed.py:80:21
	mad.wide.u32 	%rd85, %r156, 4, %rd90;
	.loc	1 80 34                         // sk09_norm_embed.py:80:34
	mul.f32 	%r155, %r869, 0f3C010204;
	.loc	1 80 26                         // sk09_norm_embed.py:80:26
	or.b32 	%r1288, %r159, %r160;
	setp.eq.b32 	%p8, %r1288, 0;
	// begin inline asm
	@%p8 st.global.b32 [ %rd85 + 0 ], { %r155 };
	// end inline asm
	.loc	1 80 4                          // sk09_norm_embed.py:80:4
	ret;
$L__tmp47:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk09_norm_embed.py"
	.file	2 "/usr/local/lib/python3.12/dist-packages/triton/language/standard.py"
	.section	.debug_abbrev
	{
.b8 1                                   // Abbreviation Code
.b8 17                                  // DW_TAG_compile_unit
.b8 1                                   // DW_CHILDREN_yes
.b8 37                                  // DW_AT_producer
.b8 8                                   // DW_FORM_string
.b8 19                                  // DW_AT_language
.b8 5                                   // DW_FORM_data2
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 16                                  // DW_AT_stmt_list
.b8 6                                   // DW_FORM_data4
.b8 27                                  // DW_AT_comp_dir
.b8 8                                   // DW_FORM_string
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 2                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 0                                   // DW_CHILDREN_no
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 32                                  // DW_AT_inline
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 3                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 1                                   // DW_CHILDREN_yes
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 4                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 1                                   // DW_CHILDREN_yes
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 11                                  // DW_FORM_data1
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 5                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 0                                   // DW_CHILDREN_no
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 5                                   // DW_FORM_data2
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 6                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 0                                   // DW_CHILDREN_no
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 11                                  // DW_FORM_data1
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 219                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0xd4 DW_TAG_compile_unit
.b8 116                                 // DW_AT_producer
.b8 114
.b8 105
.b8 116
.b8 111
.b8 110
.b8 0
.b8 2                                   // DW_AT_language
.b8 0
.b8 115                                 // DW_AT_name
.b8 107
.b8 48
.b8 57
.b8 95
.b8 110
.b8 111
.b8 114
.b8 109
.b8 95
.b8 101
.b8 109
.b8 98
.b8 101
.b8 100
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 114
.b8 101
.b8 112
.b8 111
.b8 47
.b8 118
.b8 108
.b8 108
.b8 109
.b8 47
.b8 95
.b8 103
.b8 101
.b8 110
.b8 101
.b8 115
.b8 105
.b8 115
.b8 47
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 115
.b8 0
.b8 2                                   // Abbrev [2] 0x48:0x1d DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 57
.b8 95
.b8 114
.b8 109
.b8 115
.b8 110
.b8 111
.b8 114
.b8 109
.b8 95
.b8 113
.b8 117
.b8 97
.b8 110
.b8 116
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x65:0x79 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 72                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x7a:0x32 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp19                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 75                                  // DW_AT_call_line
.b8 28                                  // DW_AT_call_column
.b8 5                                   // Abbrev [5] 0x92:0x19 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp18                          // DW_AT_high_pc
.b8 2                                   // DW_AT_call_file
.b8 37                                  // DW_AT_call_line
.b8 1
.b8 36                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 4                                   // Abbrev [4] 0xac:0x31 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp20                          // DW_AT_low_pc
.b64 $L__tmp46                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 76                                  // DW_AT_call_line
.b8 29                                  // DW_AT_call_column
.b8 6                                   // Abbrev [6] 0xc4:0x18 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp22                          // DW_AT_low_pc
.b64 $L__tmp45                          // DW_AT_high_pc
.b8 2                                   // DW_AT_call_file
.b8 191                                 // DW_AT_call_line
.b8 40                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_QVAR1_3 = _Nativo(
    "sk09_norm_embed/_sk09_rmsnorm_quant_kernel/blk16384",
    _QPTX1_3, "_sk09_rmsnorm_quant_kernel",
    warps=8, shared=32,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    horneado={9: 16384, 10: 1e-06, 11: 0.0},
    div16=[5, 6, 7, 8],
)
# E1=64 lop3, E2=0 mul

_POR_Q1 = [_QVAR1_0, _QVAR1_1, _QVAR1_2, _QVAR1_3]


def _q1_impl(grid: list[int], x_ptr: torch.Tensor, w_ptr: torch.Tensor, s_pow2_ptr: torch.Tensor, q_ptr: torch.Tensor, s_ptr: torch.Tensor, K: int, stride_xm: int, stride_qm: int, stride_sp: int, BLOCK: int, EPS: float, GAMMA_OFFSET: float) -> None:
    """Cuerpo del custom op: elige la variante y lanza el PTX."""
    args = (x_ptr, w_ptr, s_pow2_ptr, q_ptr, s_ptr, K, stride_xm, stride_qm, stride_sp, BLOCK, EPS, GAMMA_OFFSET)
    for _v in _POR_Q1:
        if all(args[_p] == _x for _p, _x in _v.horneado.items()):
            return _v(tuple(grid), *args)
    raise ValueError(
        "genesis_sk09_norm_embed_q1: no hay PTX embebido para estos constexpr; "
        "horneados disponibles: %r" % [_v.horneado for _v in _POR_Q1])


def _q1_fake(grid: list[int], x_ptr: torch.Tensor, w_ptr: torch.Tensor, s_pow2_ptr: torch.Tensor, q_ptr: torch.Tensor, s_ptr: torch.Tensor, K: int, stride_xm: int, stride_qm: int, stride_sp: int, BLOCK: int, EPS: float, GAMMA_OFFSET: float) -> None:
    """Meta impl: no toca la GPU; las salidas se mutan in-place."""
    return None


# Registro AL IMPORTAR, no en el primer uso: direct_register_custom_op
# llama a torch._library.infer_schema, que dynamo se niega a trazar
# ("Attempted to call function marked as skipped"). El hasattr evita el
# choque cuando el modulo se importa dos veces con nombres distintos,
# como hace el gate de tools/monolitizar.py.
if not hasattr(torch.ops.vllm, "genesis_sk09_norm_embed_q1"):
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op(
        op_name="genesis_sk09_norm_embed_q1",
        op_func=_q1_impl,
        mutates_args=['q_ptr', 's_ptr'],
        fake_impl=_q1_fake,
    )


def _lanzar_quant1(grid, *args):
    """Lanza el PTX embebido. Es el UNICO camino ejecutable.

    No hay fallback a Triton ni kill-switch: el kernel Triton de este
    archivo es privado y solo lo llaman los tests.
    """
    g = [grid] if isinstance(grid, int) else list(grid)
    return torch.ops.vllm.genesis_sk09_norm_embed_q1(g, *args)


# --- quant _sk09_rmsnorm_kernel: 4 variante(s) de PTX embebido ---

_QPTX2_0 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk09_rmsnorm_kernel    // -- Begin function _sk09_rmsnorm_kernel
.extern .shared .align 16 .b8 global_smem[];
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
                                        // @_sk09_rmsnorm_kernel
.visible .entry _sk09_rmsnorm_kernel(
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_kernel_param_2,
	.param .u32 _sk09_rmsnorm_kernel_param_3,
	.param .u32 _sk09_rmsnorm_kernel_param_4,
	.param .u32 _sk09_rmsnorm_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_kernel_param_6,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_kernel_param_7
)
.reqntid 256
{
	.reg .pred 	%p<6>;
	.reg .b16 	%rs<17>;
	.reg .b32 	%r<102>;
	.reg .b64 	%rd<10>;
	.loc	1 98 0                          // sk09_norm_embed.py:98:0
$L__func_begin0:
	.loc	1 98 0                          // sk09_norm_embed.py:98:0

// %bb.0:                               // %__nv_rsqrtf.exit
	ld.param.b64 	%rd4, [_sk09_rmsnorm_kernel_param_0];
	ld.param.b64 	%rd5, [_sk09_rmsnorm_kernel_param_1];
$L__tmp0:
	.loc	1 103 24                        // sk09_norm_embed.py:103:24
	mov.u32 	%r19, %ctaid.x;
	ld.param.b64 	%rd6, [_sk09_rmsnorm_kernel_param_2];
	.loc	1 104 24                        // sk09_norm_embed.py:104:24
	mov.u32 	%r20, %tid.x;
	and.b32 	%r21, %r20, 255;
	ld.param.b32 	%r22, [_sk09_rmsnorm_kernel_param_3];
	and.b32 	%r23, %r20, 31;
	ld.param.b32 	%r24, [_sk09_rmsnorm_kernel_param_4];
	ld.param.b32 	%r25, [_sk09_rmsnorm_kernel_param_5];
	shl.b32 	%r26, %r20, 3;
	and.b32 	%r27, %r26, 2040;
	.loc	1 105 18                        // sk09_norm_embed.py:105:18
	setp.lt.s32 	%p1, %r27, %r22;
	.loc	1 106 30                        // sk09_norm_embed.py:106:30
	mul.lo.s32 	%r28, %r24, %r19;
	.loc	1 106 24                        // sk09_norm_embed.py:106:24
	mad.wide.s32 	%rd7, %r28, 2, %rd4;
	.loc	1 106 42                        // sk09_norm_embed.py:106:42
	mul.wide.u32 	%rd8, %r27, 2;
	add.s64 	%rd1, %rd7, %rd8;
	mov.b32 	%r5, 0;
	.loc	1 106 16                        // sk09_norm_embed.py:106:16
	// begin inline asm
	mov.u32 %r1, %r5;
	mov.u32 %r2, %r5;
	mov.u32 %r3, %r5;
	mov.u32 %r4, %r5;
	@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
	// end inline asm
	.loc	1 106 73                        // sk09_norm_embed.py:106:73
	mov.b32 	{%rs1, %rs2}, %r1;
	cvt.f32.bf16 	%r29, %rs1;
	cvt.f32.bf16 	%r30, %rs2;
	mov.b32 	{%rs3, %rs4}, %r2;
	cvt.f32.bf16 	%r31, %rs4;
	cvt.f32.bf16 	%r32, %rs3;
	mov.b32 	{%rs5, %rs6}, %r3;
	cvt.f32.bf16 	%r33, %rs6;
	cvt.f32.bf16 	%r34, %rs5;
	mov.b32 	{%rs7, %rs8}, %r4;
	cvt.f32.bf16 	%r35, %rs8;
	cvt.f32.bf16 	%r36, %rs7;
	.loc	1 107 24                        // sk09_norm_embed.py:107:24
	add.s64 	%rd2, %rd5, %rd8;
	.loc	1 107 16                        // sk09_norm_embed.py:107:16
	// begin inline asm
	mov.u32 %r6, %r5;
	mov.u32 %r7, %r5;
	mov.u32 %r8, %r5;
	mov.u32 %r9, %r5;
	@%p1 ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd2 + 0 ];
	// end inline asm
	.loc	1 108 32                        // sk09_norm_embed.py:108:32
	mul.f32 	%r37, %r30, %r30;
$L__tmp1:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	fma.rn.f32 	%r38, %r29, %r29, %r37;
	fma.rn.f32 	%r39, %r32, %r32, %r38;
	fma.rn.f32 	%r40, %r31, %r31, %r39;
	fma.rn.f32 	%r41, %r34, %r34, %r40;
	fma.rn.f32 	%r42, %r33, %r33, %r41;
	fma.rn.f32 	%r43, %r36, %r36, %r42;
	fma.rn.f32 	%r44, %r35, %r35, %r43;
$L__tmp2:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	shfl.sync.bfly.b32 	%r45, %r44, 16, 31, -1;
$L__tmp3:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	add.f32 	%r46, %r44, %r45;
$L__tmp4:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	shfl.sync.bfly.b32 	%r47, %r46, 8, 31, -1;
$L__tmp5:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	add.f32 	%r48, %r46, %r47;
$L__tmp6:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	shfl.sync.bfly.b32 	%r49, %r48, 4, 31, -1;
$L__tmp7:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	add.f32 	%r50, %r48, %r49;
$L__tmp8:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	shfl.sync.bfly.b32 	%r51, %r50, 2, 31, -1;
$L__tmp9:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	add.f32 	%r52, %r50, %r51;
$L__tmp10:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	shfl.sync.bfly.b32 	%r53, %r52, 1, 31, -1;
$L__tmp11:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	add.f32 	%r11, %r52, %r53;
$L__tmp12:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	setp.eq.b32 	%p2, %r23, 0;
	shr.u32 	%r54, %r20, 3;
	and.b32 	%r55, %r54, 28;
	mov.b32 	%r56, global_smem;
	add.s32 	%r10, %r56, %r55;
	// begin inline asm
	@%p2 st.shared.b32 [ %r10 + 0 ], %r11;
	// end inline asm
	bar.sync 	0;
	setp.lt.u32 	%p3, %r21, 8;
	shl.b32 	%r57, %r21, 2;
	add.s32 	%r13, %r56, %r57;
	// begin inline asm
	@%p3 ld.shared.b32 %r12, [ %r13 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r58, %r12, 4, 31, -1;
$L__tmp13:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	add.f32 	%r59, %r12, %r58;
$L__tmp14:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	shfl.sync.bfly.b32 	%r60, %r59, 2, 31, -1;
$L__tmp15:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	add.f32 	%r61, %r59, %r60;
$L__tmp16:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	shfl.sync.bfly.b32 	%r62, %r61, 1, 31, -1;
$L__tmp17:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	add.f32 	%r14, %r61, %r62;
$L__tmp18:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	and.b32 	%r63, %r20, 7;
	setp.eq.b32 	%p5, %r63, 0;
	and.pred 	%p4, %p3, %p5;
	// begin inline asm
	@%p4 st.shared.b32 [ %r13 + 0 ], %r14;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r64, [global_smem];
$L__tmp19:
	.loc	1 108 45                        // sk09_norm_embed.py:108:45
	cvt.rn.f32.s32 	%r65, %r22;
	div.full.f32 	%r66, %r64, %r65;
	.loc	1 108 49                        // sk09_norm_embed.py:108:49
	add.f32 	%r67, %r66, 0f358637BD;
	.loc	1 108 21                        // sk09_norm_embed.py:108:21
	rsqrt.approx.ftz.f32 	%r68, %r67;
	.loc	1 108 12                        // sk09_norm_embed.py:108:12
	mul.f32 	%r69, %r68, %r30;
	mul.f32 	%r70, %r68, %r29;
	mul.f32 	%r71, %r68, %r31;
	mul.f32 	%r72, %r68, %r32;
	mul.f32 	%r73, %r68, %r33;
	mul.f32 	%r74, %r68, %r34;
	mul.f32 	%r75, %r68, %r35;
	mul.f32 	%r76, %r68, %r36;
	.loc	1 109 29                        // sk09_norm_embed.py:109:29
	mul.lo.s32 	%r77, %r25, %r19;
	.loc	1 109 23                        // sk09_norm_embed.py:109:23
	mad.wide.s32 	%rd9, %r77, 2, %rd6;
	.loc	1 109 41                        // sk09_norm_embed.py:109:41
	add.s64 	%rd3, %rd9, %rd8;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs9, %rs10}, %r6;
	cvt.f32.bf16 	%r78, %rs9;
	cvt.f32.bf16 	%r79, %rs10;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r80, %r79, 0f00000000;
	add.f32 	%r81, %r78, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r82, %r81, %r70;
	mul.f32 	%r83, %r80, %r69;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r15, %r83, %r82;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs11, %rs12}, %r7;
	cvt.f32.bf16 	%r84, %rs11;
	cvt.f32.bf16 	%r85, %rs12;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r86, %r85, 0f00000000;
	add.f32 	%r87, %r84, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r88, %r87, %r72;
	mul.f32 	%r89, %r86, %r71;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r16, %r89, %r88;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs13, %rs14}, %r8;
	cvt.f32.bf16 	%r90, %rs13;
	cvt.f32.bf16 	%r91, %rs14;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r92, %r91, 0f00000000;
	add.f32 	%r93, %r90, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r94, %r93, %r74;
	mul.f32 	%r95, %r92, %r73;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r17, %r95, %r94;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs15, %rs16}, %r9;
	cvt.f32.bf16 	%r96, %rs15;
	cvt.f32.bf16 	%r97, %rs16;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r98, %r97, 0f00000000;
	add.f32 	%r99, %r96, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r100, %r99, %r76;
	mul.f32 	%r101, %r98, %r75;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r18, %r101, %r100;
	.loc	1 109 47                        // sk09_norm_embed.py:109:47
	// begin inline asm
	@%p1 st.global.v4.b32 [ %rd3 + 0 ], { %r15, %r16, %r17, %r18 };
	// end inline asm
	.loc	1 109 4                         // sk09_norm_embed.py:109:4
	ret;
$L__tmp20:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk09_norm_embed.py"
	.file	2 "/usr/local/lib/python3.12/dist-packages/triton/language/standard.py"
	.section	.debug_abbrev
	{
.b8 1                                   // Abbreviation Code
.b8 17                                  // DW_TAG_compile_unit
.b8 1                                   // DW_CHILDREN_yes
.b8 37                                  // DW_AT_producer
.b8 8                                   // DW_FORM_string
.b8 19                                  // DW_AT_language
.b8 5                                   // DW_FORM_data2
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 16                                  // DW_AT_stmt_list
.b8 6                                   // DW_FORM_data4
.b8 27                                  // DW_AT_comp_dir
.b8 8                                   // DW_FORM_string
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 2                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 0                                   // DW_CHILDREN_no
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 32                                  // DW_AT_inline
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 3                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 1                                   // DW_CHILDREN_yes
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 4                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 1                                   // DW_CHILDREN_yes
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 11                                  // DW_FORM_data1
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 5                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 0                                   // DW_CHILDREN_no
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 5                                   // DW_FORM_data2
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 164                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x9d DW_TAG_compile_unit
.b8 116                                 // DW_AT_producer
.b8 114
.b8 105
.b8 116
.b8 111
.b8 110
.b8 0
.b8 2                                   // DW_AT_language
.b8 0
.b8 115                                 // DW_AT_name
.b8 107
.b8 48
.b8 57
.b8 95
.b8 110
.b8 111
.b8 114
.b8 109
.b8 95
.b8 101
.b8 109
.b8 98
.b8 101
.b8 100
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 114
.b8 101
.b8 112
.b8 111
.b8 47
.b8 118
.b8 108
.b8 108
.b8 109
.b8 47
.b8 95
.b8 103
.b8 101
.b8 110
.b8 101
.b8 115
.b8 105
.b8 115
.b8 47
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 115
.b8 0
.b8 2                                   // Abbrev [2] 0x48:0x17 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 57
.b8 95
.b8 114
.b8 109
.b8 115
.b8 110
.b8 111
.b8 114
.b8 109
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x5f:0x48 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 72                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x74:0x32 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp19                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 108                                 // DW_AT_call_line
.b8 28                                  // DW_AT_call_column
.b8 5                                   // Abbrev [5] 0x8c:0x19 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp18                          // DW_AT_high_pc
.b8 2                                   // DW_AT_call_file
.b8 37                                  // DW_AT_call_line
.b8 1
.b8 36                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_QVAR2_0 = _Nativo(
    "sk09_norm_embed/_sk09_rmsnorm_kernel/blk2048",
    _QPTX2_0, "_sk09_rmsnorm_kernel",
    warps=8, shared=32,
    abi=[0, 1, 2, 3, 4, 5],
    horneado={6: 2048, 7: 1e-06, 8: 0.0},
    div16=[3, 4, 5],
)
# E1=0 lop3, E2=0 mul

_QPTX2_1 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk09_rmsnorm_kernel    // -- Begin function _sk09_rmsnorm_kernel
.extern .shared .align 16 .b8 global_smem[];
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
                                        // @_sk09_rmsnorm_kernel
.visible .entry _sk09_rmsnorm_kernel(
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_kernel_param_2,
	.param .u32 _sk09_rmsnorm_kernel_param_3,
	.param .u32 _sk09_rmsnorm_kernel_param_4,
	.param .u32 _sk09_rmsnorm_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_kernel_param_6,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_kernel_param_7
)
.reqntid 256
{
	.reg .pred 	%p<7>;
	.reg .b16 	%rs<33>;
	.reg .b32 	%r<163>;
	.reg .b64 	%rd<13>;
	.loc	1 98 0                          // sk09_norm_embed.py:98:0
$L__func_begin0:
	.loc	1 98 0                          // sk09_norm_embed.py:98:0

// %bb.0:                               // %__nv_rsqrtf.exit
	ld.param.b64 	%rd7, [_sk09_rmsnorm_kernel_param_0];
	ld.param.b64 	%rd8, [_sk09_rmsnorm_kernel_param_1];
$L__tmp0:
	.loc	1 103 24                        // sk09_norm_embed.py:103:24
	mov.u32 	%r31, %ctaid.x;
	ld.param.b64 	%rd9, [_sk09_rmsnorm_kernel_param_2];
	.loc	1 104 24                        // sk09_norm_embed.py:104:24
	mov.u32 	%r32, %tid.x;
	and.b32 	%r33, %r32, 255;
	ld.param.b32 	%r34, [_sk09_rmsnorm_kernel_param_3];
	and.b32 	%r35, %r32, 31;
	ld.param.b32 	%r36, [_sk09_rmsnorm_kernel_param_4];
	ld.param.b32 	%r37, [_sk09_rmsnorm_kernel_param_5];
	shl.b32 	%r38, %r32, 3;
	and.b32 	%r39, %r38, 2040;
	or.b32 	%r40, %r39, 2048;
	.loc	1 105 18                        // sk09_norm_embed.py:105:18
	setp.lt.s32 	%p1, %r39, %r34;
	setp.lt.s32 	%p2, %r40, %r34;
	.loc	1 106 30                        // sk09_norm_embed.py:106:30
	mul.lo.s32 	%r41, %r36, %r31;
	.loc	1 106 24                        // sk09_norm_embed.py:106:24
	mad.wide.s32 	%rd10, %r41, 2, %rd7;
	.loc	1 106 42                        // sk09_norm_embed.py:106:42
	mul.wide.u32 	%rd11, %r39, 2;
	add.s64 	%rd1, %rd10, %rd11;
	add.s64 	%rd2, %rd1, 4096;
	mov.b32 	%r5, 0;
	.loc	1 106 16                        // sk09_norm_embed.py:106:16
	// begin inline asm
	mov.u32 %r1, %r5;
	mov.u32 %r2, %r5;
	mov.u32 %r3, %r5;
	mov.u32 %r4, %r5;
	@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r6, %r5;
	mov.u32 %r7, %r5;
	mov.u32 %r8, %r5;
	mov.u32 %r9, %r5;
	@%p2 ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd2 + 0 ];
	// end inline asm
	.loc	1 106 73                        // sk09_norm_embed.py:106:73
	mov.b32 	{%rs1, %rs2}, %r1;
	cvt.f32.bf16 	%r42, %rs1;
	cvt.f32.bf16 	%r43, %rs2;
	mov.b32 	{%rs3, %rs4}, %r2;
	cvt.f32.bf16 	%r44, %rs4;
	cvt.f32.bf16 	%r45, %rs3;
	mov.b32 	{%rs5, %rs6}, %r3;
	cvt.f32.bf16 	%r46, %rs6;
	cvt.f32.bf16 	%r47, %rs5;
	mov.b32 	{%rs7, %rs8}, %r4;
	cvt.f32.bf16 	%r48, %rs8;
	cvt.f32.bf16 	%r49, %rs7;
	mov.b32 	{%rs9, %rs10}, %r6;
	cvt.f32.bf16 	%r50, %rs10;
	cvt.f32.bf16 	%r51, %rs9;
	mov.b32 	{%rs11, %rs12}, %r7;
	cvt.f32.bf16 	%r52, %rs12;
	cvt.f32.bf16 	%r53, %rs11;
	mov.b32 	{%rs13, %rs14}, %r8;
	cvt.f32.bf16 	%r54, %rs14;
	cvt.f32.bf16 	%r55, %rs13;
	mov.b32 	{%rs15, %rs16}, %r9;
	cvt.f32.bf16 	%r56, %rs16;
	cvt.f32.bf16 	%r57, %rs15;
	.loc	1 107 24                        // sk09_norm_embed.py:107:24
	add.s64 	%rd3, %rd8, %rd11;
	add.s64 	%rd4, %rd3, 4096;
	.loc	1 107 16                        // sk09_norm_embed.py:107:16
	// begin inline asm
	mov.u32 %r10, %r5;
	mov.u32 %r11, %r5;
	mov.u32 %r12, %r5;
	mov.u32 %r13, %r5;
	@%p1 ld.global.v4.b32 { %r10, %r11, %r12, %r13 }, [ %rd3 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r14, %r5;
	mov.u32 %r15, %r5;
	mov.u32 %r16, %r5;
	mov.u32 %r17, %r5;
	@%p2 ld.global.v4.b32 { %r14, %r15, %r16, %r17 }, [ %rd4 + 0 ];
	// end inline asm
	.loc	1 108 32                        // sk09_norm_embed.py:108:32
	mul.f32 	%r58, %r43, %r43;
$L__tmp1:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	fma.rn.f32 	%r59, %r42, %r42, %r58;
	fma.rn.f32 	%r60, %r45, %r45, %r59;
	fma.rn.f32 	%r61, %r44, %r44, %r60;
	fma.rn.f32 	%r62, %r47, %r47, %r61;
	fma.rn.f32 	%r63, %r46, %r46, %r62;
	fma.rn.f32 	%r64, %r49, %r49, %r63;
	fma.rn.f32 	%r65, %r48, %r48, %r64;
	fma.rn.f32 	%r66, %r51, %r51, %r65;
	fma.rn.f32 	%r67, %r50, %r50, %r66;
	fma.rn.f32 	%r68, %r53, %r53, %r67;
	fma.rn.f32 	%r69, %r52, %r52, %r68;
	fma.rn.f32 	%r70, %r55, %r55, %r69;
	fma.rn.f32 	%r71, %r54, %r54, %r70;
	fma.rn.f32 	%r72, %r57, %r57, %r71;
	fma.rn.f32 	%r73, %r56, %r56, %r72;
$L__tmp2:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	shfl.sync.bfly.b32 	%r74, %r73, 16, 31, -1;
$L__tmp3:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	add.f32 	%r75, %r73, %r74;
$L__tmp4:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	shfl.sync.bfly.b32 	%r76, %r75, 8, 31, -1;
$L__tmp5:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	add.f32 	%r77, %r75, %r76;
$L__tmp6:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	shfl.sync.bfly.b32 	%r78, %r77, 4, 31, -1;
$L__tmp7:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	add.f32 	%r79, %r77, %r78;
$L__tmp8:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	shfl.sync.bfly.b32 	%r80, %r79, 2, 31, -1;
$L__tmp9:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	add.f32 	%r81, %r79, %r80;
$L__tmp10:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	shfl.sync.bfly.b32 	%r82, %r81, 1, 31, -1;
$L__tmp11:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	add.f32 	%r19, %r81, %r82;
$L__tmp12:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	setp.eq.b32 	%p3, %r35, 0;
	shr.u32 	%r83, %r32, 3;
	and.b32 	%r84, %r83, 28;
	mov.b32 	%r85, global_smem;
	add.s32 	%r18, %r85, %r84;
	// begin inline asm
	@%p3 st.shared.b32 [ %r18 + 0 ], %r19;
	// end inline asm
	bar.sync 	0;
	setp.lt.u32 	%p4, %r33, 8;
	shl.b32 	%r86, %r33, 2;
	add.s32 	%r21, %r85, %r86;
	// begin inline asm
	@%p4 ld.shared.b32 %r20, [ %r21 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r87, %r20, 4, 31, -1;
$L__tmp13:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	add.f32 	%r88, %r20, %r87;
$L__tmp14:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	shfl.sync.bfly.b32 	%r89, %r88, 2, 31, -1;
$L__tmp15:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	add.f32 	%r90, %r88, %r89;
$L__tmp16:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	shfl.sync.bfly.b32 	%r91, %r90, 1, 31, -1;
$L__tmp17:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	add.f32 	%r22, %r90, %r91;
$L__tmp18:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	and.b32 	%r92, %r32, 7;
	setp.eq.b32 	%p6, %r92, 0;
	and.pred 	%p5, %p4, %p6;
	// begin inline asm
	@%p5 st.shared.b32 [ %r21 + 0 ], %r22;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r93, [global_smem];
$L__tmp19:
	.loc	1 108 45                        // sk09_norm_embed.py:108:45
	cvt.rn.f32.s32 	%r94, %r34;
	div.full.f32 	%r95, %r93, %r94;
	.loc	1 108 49                        // sk09_norm_embed.py:108:49
	add.f32 	%r96, %r95, 0f358637BD;
	.loc	1 108 21                        // sk09_norm_embed.py:108:21
	rsqrt.approx.ftz.f32 	%r97, %r96;
	.loc	1 108 12                        // sk09_norm_embed.py:108:12
	mul.f32 	%r98, %r97, %r43;
	mul.f32 	%r99, %r97, %r42;
	mul.f32 	%r100, %r97, %r44;
	mul.f32 	%r101, %r97, %r45;
	mul.f32 	%r102, %r97, %r46;
	mul.f32 	%r103, %r97, %r47;
	mul.f32 	%r104, %r97, %r48;
	mul.f32 	%r105, %r97, %r49;
	mul.f32 	%r106, %r97, %r50;
	mul.f32 	%r107, %r97, %r51;
	mul.f32 	%r108, %r97, %r52;
	mul.f32 	%r109, %r97, %r53;
	mul.f32 	%r110, %r97, %r54;
	mul.f32 	%r111, %r97, %r55;
	mul.f32 	%r112, %r97, %r56;
	mul.f32 	%r113, %r97, %r57;
	.loc	1 109 29                        // sk09_norm_embed.py:109:29
	mul.lo.s32 	%r114, %r37, %r31;
	.loc	1 109 23                        // sk09_norm_embed.py:109:23
	mad.wide.s32 	%rd12, %r114, 2, %rd9;
	.loc	1 109 41                        // sk09_norm_embed.py:109:41
	add.s64 	%rd5, %rd12, %rd11;
	add.s64 	%rd6, %rd5, 4096;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs17, %rs18}, %r10;
	cvt.f32.bf16 	%r115, %rs17;
	cvt.f32.bf16 	%r116, %rs18;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r117, %r116, 0f00000000;
	add.f32 	%r118, %r115, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r119, %r118, %r99;
	mul.f32 	%r120, %r117, %r98;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r23, %r120, %r119;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs19, %rs20}, %r11;
	cvt.f32.bf16 	%r121, %rs19;
	cvt.f32.bf16 	%r122, %rs20;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r123, %r122, 0f00000000;
	add.f32 	%r124, %r121, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r125, %r124, %r101;
	mul.f32 	%r126, %r123, %r100;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r24, %r126, %r125;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs21, %rs22}, %r12;
	cvt.f32.bf16 	%r127, %rs21;
	cvt.f32.bf16 	%r128, %rs22;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r129, %r128, 0f00000000;
	add.f32 	%r130, %r127, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r131, %r130, %r103;
	mul.f32 	%r132, %r129, %r102;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r25, %r132, %r131;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs23, %rs24}, %r13;
	cvt.f32.bf16 	%r133, %rs23;
	cvt.f32.bf16 	%r134, %rs24;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r135, %r134, 0f00000000;
	add.f32 	%r136, %r133, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r137, %r136, %r105;
	mul.f32 	%r138, %r135, %r104;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r26, %r138, %r137;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs25, %rs26}, %r14;
	cvt.f32.bf16 	%r139, %rs25;
	cvt.f32.bf16 	%r140, %rs26;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r141, %r140, 0f00000000;
	add.f32 	%r142, %r139, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r143, %r142, %r107;
	mul.f32 	%r144, %r141, %r106;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r27, %r144, %r143;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs27, %rs28}, %r15;
	cvt.f32.bf16 	%r145, %rs27;
	cvt.f32.bf16 	%r146, %rs28;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r147, %r146, 0f00000000;
	add.f32 	%r148, %r145, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r149, %r148, %r109;
	mul.f32 	%r150, %r147, %r108;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r28, %r150, %r149;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs29, %rs30}, %r16;
	cvt.f32.bf16 	%r151, %rs29;
	cvt.f32.bf16 	%r152, %rs30;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r153, %r152, 0f00000000;
	add.f32 	%r154, %r151, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r155, %r154, %r111;
	mul.f32 	%r156, %r153, %r110;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r29, %r156, %r155;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs31, %rs32}, %r17;
	cvt.f32.bf16 	%r157, %rs31;
	cvt.f32.bf16 	%r158, %rs32;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r159, %r158, 0f00000000;
	add.f32 	%r160, %r157, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r161, %r160, %r113;
	mul.f32 	%r162, %r159, %r112;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r30, %r162, %r161;
	.loc	1 109 47                        // sk09_norm_embed.py:109:47
	// begin inline asm
	@%p1 st.global.v4.b32 [ %rd5 + 0 ], { %r23, %r24, %r25, %r26 };
	// end inline asm
	// begin inline asm
	@%p2 st.global.v4.b32 [ %rd6 + 0 ], { %r27, %r28, %r29, %r30 };
	// end inline asm
	.loc	1 109 4                         // sk09_norm_embed.py:109:4
	ret;
$L__tmp20:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk09_norm_embed.py"
	.file	2 "/usr/local/lib/python3.12/dist-packages/triton/language/standard.py"
	.section	.debug_abbrev
	{
.b8 1                                   // Abbreviation Code
.b8 17                                  // DW_TAG_compile_unit
.b8 1                                   // DW_CHILDREN_yes
.b8 37                                  // DW_AT_producer
.b8 8                                   // DW_FORM_string
.b8 19                                  // DW_AT_language
.b8 5                                   // DW_FORM_data2
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 16                                  // DW_AT_stmt_list
.b8 6                                   // DW_FORM_data4
.b8 27                                  // DW_AT_comp_dir
.b8 8                                   // DW_FORM_string
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 2                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 0                                   // DW_CHILDREN_no
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 32                                  // DW_AT_inline
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 3                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 1                                   // DW_CHILDREN_yes
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 4                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 1                                   // DW_CHILDREN_yes
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 11                                  // DW_FORM_data1
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 5                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 0                                   // DW_CHILDREN_no
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 5                                   // DW_FORM_data2
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 164                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x9d DW_TAG_compile_unit
.b8 116                                 // DW_AT_producer
.b8 114
.b8 105
.b8 116
.b8 111
.b8 110
.b8 0
.b8 2                                   // DW_AT_language
.b8 0
.b8 115                                 // DW_AT_name
.b8 107
.b8 48
.b8 57
.b8 95
.b8 110
.b8 111
.b8 114
.b8 109
.b8 95
.b8 101
.b8 109
.b8 98
.b8 101
.b8 100
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 114
.b8 101
.b8 112
.b8 111
.b8 47
.b8 118
.b8 108
.b8 108
.b8 109
.b8 47
.b8 95
.b8 103
.b8 101
.b8 110
.b8 101
.b8 115
.b8 105
.b8 115
.b8 47
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 115
.b8 0
.b8 2                                   // Abbrev [2] 0x48:0x17 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 57
.b8 95
.b8 114
.b8 109
.b8 115
.b8 110
.b8 111
.b8 114
.b8 109
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x5f:0x48 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 72                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x74:0x32 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp19                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 108                                 // DW_AT_call_line
.b8 28                                  // DW_AT_call_column
.b8 5                                   // Abbrev [5] 0x8c:0x19 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp18                          // DW_AT_high_pc
.b8 2                                   // DW_AT_call_file
.b8 37                                  // DW_AT_call_line
.b8 1
.b8 36                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_QVAR2_1 = _Nativo(
    "sk09_norm_embed/_sk09_rmsnorm_kernel/blk4096",
    _QPTX2_1, "_sk09_rmsnorm_kernel",
    warps=8, shared=32,
    abi=[0, 1, 2, 3, 4, 5],
    horneado={6: 4096, 7: 1e-06, 8: 0.0},
    div16=[3, 4, 5],
)
# E1=0 lop3, E2=0 mul

_QPTX2_2 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk09_rmsnorm_kernel    // -- Begin function _sk09_rmsnorm_kernel
.extern .shared .align 16 .b8 global_smem[];
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
                                        // @_sk09_rmsnorm_kernel
.visible .entry _sk09_rmsnorm_kernel(
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_kernel_param_2,
	.param .u32 _sk09_rmsnorm_kernel_param_3,
	.param .u32 _sk09_rmsnorm_kernel_param_4,
	.param .u32 _sk09_rmsnorm_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_kernel_param_6,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_kernel_param_7
)
.reqntid 256
{
	.reg .pred 	%p<9>;
	.reg .b16 	%rs<65>;
	.reg .b32 	%r<285>;
	.reg .b64 	%rd<20>;
	.loc	1 98 0                          // sk09_norm_embed.py:98:0
$L__func_begin0:
	.loc	1 98 0                          // sk09_norm_embed.py:98:0

// %bb.0:                               // %__nv_rsqrtf.exit
	ld.param.b64 	%rd13, [_sk09_rmsnorm_kernel_param_0];
	ld.param.b64 	%rd14, [_sk09_rmsnorm_kernel_param_1];
$L__tmp0:
	.loc	1 103 24                        // sk09_norm_embed.py:103:24
	mov.u32 	%r55, %ctaid.x;
	ld.param.b64 	%rd15, [_sk09_rmsnorm_kernel_param_2];
	.loc	1 104 24                        // sk09_norm_embed.py:104:24
	mov.u32 	%r56, %tid.x;
	and.b32 	%r57, %r56, 255;
	ld.param.b32 	%r58, [_sk09_rmsnorm_kernel_param_3];
	and.b32 	%r59, %r56, 31;
	ld.param.b32 	%r60, [_sk09_rmsnorm_kernel_param_4];
	ld.param.b32 	%r61, [_sk09_rmsnorm_kernel_param_5];
	shl.b32 	%r62, %r56, 3;
	and.b32 	%r63, %r62, 2040;
	or.b32 	%r64, %r63, 2048;
	or.b32 	%r65, %r63, 4096;
	or.b32 	%r66, %r62, 6144;
	.loc	1 105 18                        // sk09_norm_embed.py:105:18
	setp.lt.s32 	%p1, %r63, %r58;
	setp.lt.s32 	%p2, %r64, %r58;
	setp.lt.s32 	%p3, %r65, %r58;
	setp.lt.s32 	%p4, %r66, %r58;
	.loc	1 106 30                        // sk09_norm_embed.py:106:30
	mul.lo.s32 	%r67, %r60, %r55;
	.loc	1 106 24                        // sk09_norm_embed.py:106:24
	mad.wide.s32 	%rd16, %r67, 2, %rd13;
	.loc	1 106 42                        // sk09_norm_embed.py:106:42
	mul.wide.u32 	%rd17, %r63, 2;
	add.s64 	%rd1, %rd16, %rd17;
	add.s64 	%rd2, %rd1, 4096;
	add.s64 	%rd3, %rd1, 8192;
	mul.wide.u32 	%rd18, %r66, 2;
	add.s64 	%rd4, %rd16, %rd18;
	mov.b32 	%r5, 0;
	.loc	1 106 16                        // sk09_norm_embed.py:106:16
	// begin inline asm
	mov.u32 %r1, %r5;
	mov.u32 %r2, %r5;
	mov.u32 %r3, %r5;
	mov.u32 %r4, %r5;
	@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r6, %r5;
	mov.u32 %r7, %r5;
	mov.u32 %r8, %r5;
	mov.u32 %r9, %r5;
	@%p2 ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd2 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r10, %r5;
	mov.u32 %r11, %r5;
	mov.u32 %r12, %r5;
	mov.u32 %r13, %r5;
	@%p3 ld.global.v4.b32 { %r10, %r11, %r12, %r13 }, [ %rd3 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r14, %r5;
	mov.u32 %r15, %r5;
	mov.u32 %r16, %r5;
	mov.u32 %r17, %r5;
	@%p4 ld.global.v4.b32 { %r14, %r15, %r16, %r17 }, [ %rd4 + 0 ];
	// end inline asm
	.loc	1 106 73                        // sk09_norm_embed.py:106:73
	mov.b32 	{%rs1, %rs2}, %r1;
	cvt.f32.bf16 	%r68, %rs1;
	cvt.f32.bf16 	%r69, %rs2;
	mov.b32 	{%rs3, %rs4}, %r2;
	cvt.f32.bf16 	%r70, %rs4;
	cvt.f32.bf16 	%r71, %rs3;
	mov.b32 	{%rs5, %rs6}, %r3;
	cvt.f32.bf16 	%r72, %rs6;
	cvt.f32.bf16 	%r73, %rs5;
	mov.b32 	{%rs7, %rs8}, %r4;
	cvt.f32.bf16 	%r74, %rs8;
	cvt.f32.bf16 	%r75, %rs7;
	mov.b32 	{%rs9, %rs10}, %r6;
	cvt.f32.bf16 	%r76, %rs10;
	cvt.f32.bf16 	%r77, %rs9;
	mov.b32 	{%rs11, %rs12}, %r7;
	cvt.f32.bf16 	%r78, %rs12;
	cvt.f32.bf16 	%r79, %rs11;
	mov.b32 	{%rs13, %rs14}, %r8;
	cvt.f32.bf16 	%r80, %rs14;
	cvt.f32.bf16 	%r81, %rs13;
	mov.b32 	{%rs15, %rs16}, %r9;
	cvt.f32.bf16 	%r82, %rs16;
	cvt.f32.bf16 	%r83, %rs15;
	mov.b32 	{%rs17, %rs18}, %r10;
	cvt.f32.bf16 	%r84, %rs18;
	cvt.f32.bf16 	%r85, %rs17;
	mov.b32 	{%rs19, %rs20}, %r11;
	cvt.f32.bf16 	%r86, %rs20;
	cvt.f32.bf16 	%r87, %rs19;
	mov.b32 	{%rs21, %rs22}, %r12;
	cvt.f32.bf16 	%r88, %rs22;
	cvt.f32.bf16 	%r89, %rs21;
	mov.b32 	{%rs23, %rs24}, %r13;
	cvt.f32.bf16 	%r90, %rs24;
	cvt.f32.bf16 	%r91, %rs23;
	mov.b32 	{%rs25, %rs26}, %r14;
	cvt.f32.bf16 	%r92, %rs26;
	cvt.f32.bf16 	%r93, %rs25;
	mov.b32 	{%rs27, %rs28}, %r15;
	cvt.f32.bf16 	%r94, %rs28;
	cvt.f32.bf16 	%r95, %rs27;
	mov.b32 	{%rs29, %rs30}, %r16;
	cvt.f32.bf16 	%r96, %rs30;
	cvt.f32.bf16 	%r97, %rs29;
	mov.b32 	{%rs31, %rs32}, %r17;
	cvt.f32.bf16 	%r98, %rs32;
	cvt.f32.bf16 	%r99, %rs31;
	.loc	1 107 24                        // sk09_norm_embed.py:107:24
	add.s64 	%rd5, %rd14, %rd17;
	add.s64 	%rd6, %rd5, 4096;
	add.s64 	%rd7, %rd5, 8192;
	add.s64 	%rd8, %rd14, %rd18;
	.loc	1 107 16                        // sk09_norm_embed.py:107:16
	// begin inline asm
	mov.u32 %r18, %r5;
	mov.u32 %r19, %r5;
	mov.u32 %r20, %r5;
	mov.u32 %r21, %r5;
	@%p1 ld.global.v4.b32 { %r18, %r19, %r20, %r21 }, [ %rd5 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r22, %r5;
	mov.u32 %r23, %r5;
	mov.u32 %r24, %r5;
	mov.u32 %r25, %r5;
	@%p2 ld.global.v4.b32 { %r22, %r23, %r24, %r25 }, [ %rd6 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r26, %r5;
	mov.u32 %r27, %r5;
	mov.u32 %r28, %r5;
	mov.u32 %r29, %r5;
	@%p3 ld.global.v4.b32 { %r26, %r27, %r28, %r29 }, [ %rd7 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r30, %r5;
	mov.u32 %r31, %r5;
	mov.u32 %r32, %r5;
	mov.u32 %r33, %r5;
	@%p4 ld.global.v4.b32 { %r30, %r31, %r32, %r33 }, [ %rd8 + 0 ];
	// end inline asm
	.loc	1 108 32                        // sk09_norm_embed.py:108:32
	mul.f32 	%r100, %r69, %r69;
$L__tmp1:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	fma.rn.f32 	%r101, %r68, %r68, %r100;
	fma.rn.f32 	%r102, %r71, %r71, %r101;
	fma.rn.f32 	%r103, %r70, %r70, %r102;
	fma.rn.f32 	%r104, %r73, %r73, %r103;
	fma.rn.f32 	%r105, %r72, %r72, %r104;
	fma.rn.f32 	%r106, %r75, %r75, %r105;
	fma.rn.f32 	%r107, %r74, %r74, %r106;
	fma.rn.f32 	%r108, %r77, %r77, %r107;
	fma.rn.f32 	%r109, %r76, %r76, %r108;
	fma.rn.f32 	%r110, %r79, %r79, %r109;
	fma.rn.f32 	%r111, %r78, %r78, %r110;
	fma.rn.f32 	%r112, %r81, %r81, %r111;
	fma.rn.f32 	%r113, %r80, %r80, %r112;
	fma.rn.f32 	%r114, %r83, %r83, %r113;
	fma.rn.f32 	%r115, %r82, %r82, %r114;
	fma.rn.f32 	%r116, %r85, %r85, %r115;
	fma.rn.f32 	%r117, %r84, %r84, %r116;
	fma.rn.f32 	%r118, %r87, %r87, %r117;
	fma.rn.f32 	%r119, %r86, %r86, %r118;
	fma.rn.f32 	%r120, %r89, %r89, %r119;
	fma.rn.f32 	%r121, %r88, %r88, %r120;
	fma.rn.f32 	%r122, %r91, %r91, %r121;
	fma.rn.f32 	%r123, %r90, %r90, %r122;
	fma.rn.f32 	%r124, %r93, %r93, %r123;
	fma.rn.f32 	%r125, %r92, %r92, %r124;
	fma.rn.f32 	%r126, %r95, %r95, %r125;
	fma.rn.f32 	%r127, %r94, %r94, %r126;
	fma.rn.f32 	%r128, %r97, %r97, %r127;
	fma.rn.f32 	%r129, %r96, %r96, %r128;
	fma.rn.f32 	%r130, %r99, %r99, %r129;
	fma.rn.f32 	%r131, %r98, %r98, %r130;
$L__tmp2:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	shfl.sync.bfly.b32 	%r132, %r131, 16, 31, -1;
$L__tmp3:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	add.f32 	%r133, %r131, %r132;
$L__tmp4:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	shfl.sync.bfly.b32 	%r134, %r133, 8, 31, -1;
$L__tmp5:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	add.f32 	%r135, %r133, %r134;
$L__tmp6:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	shfl.sync.bfly.b32 	%r136, %r135, 4, 31, -1;
$L__tmp7:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	add.f32 	%r137, %r135, %r136;
$L__tmp8:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	shfl.sync.bfly.b32 	%r138, %r137, 2, 31, -1;
$L__tmp9:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	add.f32 	%r139, %r137, %r138;
$L__tmp10:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	shfl.sync.bfly.b32 	%r140, %r139, 1, 31, -1;
$L__tmp11:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	add.f32 	%r35, %r139, %r140;
$L__tmp12:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	setp.eq.b32 	%p5, %r59, 0;
	shr.u32 	%r141, %r56, 3;
	and.b32 	%r142, %r141, 28;
	mov.b32 	%r143, global_smem;
	add.s32 	%r34, %r143, %r142;
	// begin inline asm
	@%p5 st.shared.b32 [ %r34 + 0 ], %r35;
	// end inline asm
	bar.sync 	0;
	setp.lt.u32 	%p6, %r57, 8;
	shl.b32 	%r144, %r57, 2;
	add.s32 	%r37, %r143, %r144;
	// begin inline asm
	@%p6 ld.shared.b32 %r36, [ %r37 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r145, %r36, 4, 31, -1;
$L__tmp13:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	add.f32 	%r146, %r36, %r145;
$L__tmp14:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	shfl.sync.bfly.b32 	%r147, %r146, 2, 31, -1;
$L__tmp15:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	add.f32 	%r148, %r146, %r147;
$L__tmp16:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	shfl.sync.bfly.b32 	%r149, %r148, 1, 31, -1;
$L__tmp17:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	add.f32 	%r38, %r148, %r149;
$L__tmp18:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	and.b32 	%r150, %r56, 7;
	setp.eq.b32 	%p8, %r150, 0;
	and.pred 	%p7, %p6, %p8;
	// begin inline asm
	@%p7 st.shared.b32 [ %r37 + 0 ], %r38;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r151, [global_smem];
$L__tmp19:
	.loc	1 108 45                        // sk09_norm_embed.py:108:45
	cvt.rn.f32.s32 	%r152, %r58;
	div.full.f32 	%r153, %r151, %r152;
	.loc	1 108 49                        // sk09_norm_embed.py:108:49
	add.f32 	%r154, %r153, 0f358637BD;
	.loc	1 108 21                        // sk09_norm_embed.py:108:21
	rsqrt.approx.ftz.f32 	%r155, %r154;
	.loc	1 108 12                        // sk09_norm_embed.py:108:12
	mul.f32 	%r156, %r155, %r69;
	mul.f32 	%r157, %r155, %r68;
	mul.f32 	%r158, %r155, %r70;
	mul.f32 	%r159, %r155, %r71;
	mul.f32 	%r160, %r155, %r72;
	mul.f32 	%r161, %r155, %r73;
	mul.f32 	%r162, %r155, %r74;
	mul.f32 	%r163, %r155, %r75;
	mul.f32 	%r164, %r155, %r76;
	mul.f32 	%r165, %r155, %r77;
	mul.f32 	%r166, %r155, %r78;
	mul.f32 	%r167, %r155, %r79;
	mul.f32 	%r168, %r155, %r80;
	mul.f32 	%r169, %r155, %r81;
	mul.f32 	%r170, %r155, %r82;
	mul.f32 	%r171, %r155, %r83;
	mul.f32 	%r172, %r155, %r84;
	mul.f32 	%r173, %r155, %r85;
	mul.f32 	%r174, %r155, %r86;
	mul.f32 	%r175, %r155, %r87;
	mul.f32 	%r176, %r155, %r88;
	mul.f32 	%r177, %r155, %r89;
	mul.f32 	%r178, %r155, %r90;
	mul.f32 	%r179, %r155, %r91;
	mul.f32 	%r180, %r155, %r92;
	mul.f32 	%r181, %r155, %r93;
	mul.f32 	%r182, %r155, %r94;
	mul.f32 	%r183, %r155, %r95;
	mul.f32 	%r184, %r155, %r96;
	mul.f32 	%r185, %r155, %r97;
	mul.f32 	%r186, %r155, %r98;
	mul.f32 	%r187, %r155, %r99;
	.loc	1 109 29                        // sk09_norm_embed.py:109:29
	mul.lo.s32 	%r188, %r61, %r55;
	.loc	1 109 23                        // sk09_norm_embed.py:109:23
	mad.wide.s32 	%rd19, %r188, 2, %rd15;
	.loc	1 109 41                        // sk09_norm_embed.py:109:41
	add.s64 	%rd9, %rd19, %rd17;
	add.s64 	%rd10, %rd9, 4096;
	add.s64 	%rd11, %rd9, 8192;
	add.s64 	%rd12, %rd19, %rd18;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs33, %rs34}, %r18;
	cvt.f32.bf16 	%r189, %rs33;
	cvt.f32.bf16 	%r190, %rs34;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r191, %r190, 0f00000000;
	add.f32 	%r192, %r189, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r193, %r192, %r157;
	mul.f32 	%r194, %r191, %r156;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r39, %r194, %r193;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs35, %rs36}, %r19;
	cvt.f32.bf16 	%r195, %rs35;
	cvt.f32.bf16 	%r196, %rs36;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r197, %r196, 0f00000000;
	add.f32 	%r198, %r195, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r199, %r198, %r159;
	mul.f32 	%r200, %r197, %r158;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r40, %r200, %r199;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs37, %rs38}, %r20;
	cvt.f32.bf16 	%r201, %rs37;
	cvt.f32.bf16 	%r202, %rs38;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r203, %r202, 0f00000000;
	add.f32 	%r204, %r201, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r205, %r204, %r161;
	mul.f32 	%r206, %r203, %r160;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r41, %r206, %r205;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs39, %rs40}, %r21;
	cvt.f32.bf16 	%r207, %rs39;
	cvt.f32.bf16 	%r208, %rs40;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r209, %r208, 0f00000000;
	add.f32 	%r210, %r207, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r211, %r210, %r163;
	mul.f32 	%r212, %r209, %r162;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r42, %r212, %r211;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs41, %rs42}, %r22;
	cvt.f32.bf16 	%r213, %rs41;
	cvt.f32.bf16 	%r214, %rs42;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r215, %r214, 0f00000000;
	add.f32 	%r216, %r213, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r217, %r216, %r165;
	mul.f32 	%r218, %r215, %r164;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r43, %r218, %r217;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs43, %rs44}, %r23;
	cvt.f32.bf16 	%r219, %rs43;
	cvt.f32.bf16 	%r220, %rs44;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r221, %r220, 0f00000000;
	add.f32 	%r222, %r219, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r223, %r222, %r167;
	mul.f32 	%r224, %r221, %r166;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r44, %r224, %r223;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs45, %rs46}, %r24;
	cvt.f32.bf16 	%r225, %rs45;
	cvt.f32.bf16 	%r226, %rs46;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r227, %r226, 0f00000000;
	add.f32 	%r228, %r225, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r229, %r228, %r169;
	mul.f32 	%r230, %r227, %r168;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r45, %r230, %r229;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs47, %rs48}, %r25;
	cvt.f32.bf16 	%r231, %rs47;
	cvt.f32.bf16 	%r232, %rs48;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r233, %r232, 0f00000000;
	add.f32 	%r234, %r231, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r235, %r234, %r171;
	mul.f32 	%r236, %r233, %r170;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r46, %r236, %r235;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs49, %rs50}, %r26;
	cvt.f32.bf16 	%r237, %rs49;
	cvt.f32.bf16 	%r238, %rs50;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r239, %r238, 0f00000000;
	add.f32 	%r240, %r237, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r241, %r240, %r173;
	mul.f32 	%r242, %r239, %r172;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r47, %r242, %r241;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs51, %rs52}, %r27;
	cvt.f32.bf16 	%r243, %rs51;
	cvt.f32.bf16 	%r244, %rs52;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r245, %r244, 0f00000000;
	add.f32 	%r246, %r243, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r247, %r246, %r175;
	mul.f32 	%r248, %r245, %r174;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r48, %r248, %r247;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs53, %rs54}, %r28;
	cvt.f32.bf16 	%r249, %rs53;
	cvt.f32.bf16 	%r250, %rs54;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r251, %r250, 0f00000000;
	add.f32 	%r252, %r249, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r253, %r252, %r177;
	mul.f32 	%r254, %r251, %r176;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r49, %r254, %r253;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs55, %rs56}, %r29;
	cvt.f32.bf16 	%r255, %rs55;
	cvt.f32.bf16 	%r256, %rs56;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r257, %r256, 0f00000000;
	add.f32 	%r258, %r255, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r259, %r258, %r179;
	mul.f32 	%r260, %r257, %r178;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r50, %r260, %r259;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs57, %rs58}, %r30;
	cvt.f32.bf16 	%r261, %rs57;
	cvt.f32.bf16 	%r262, %rs58;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r263, %r262, 0f00000000;
	add.f32 	%r264, %r261, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r265, %r264, %r181;
	mul.f32 	%r266, %r263, %r180;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r51, %r266, %r265;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs59, %rs60}, %r31;
	cvt.f32.bf16 	%r267, %rs59;
	cvt.f32.bf16 	%r268, %rs60;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r269, %r268, 0f00000000;
	add.f32 	%r270, %r267, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r271, %r270, %r183;
	mul.f32 	%r272, %r269, %r182;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r52, %r272, %r271;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs61, %rs62}, %r32;
	cvt.f32.bf16 	%r273, %rs61;
	cvt.f32.bf16 	%r274, %rs62;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r275, %r274, 0f00000000;
	add.f32 	%r276, %r273, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r277, %r276, %r185;
	mul.f32 	%r278, %r275, %r184;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r53, %r278, %r277;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs63, %rs64}, %r33;
	cvt.f32.bf16 	%r279, %rs63;
	cvt.f32.bf16 	%r280, %rs64;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r281, %r280, 0f00000000;
	add.f32 	%r282, %r279, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r283, %r282, %r187;
	mul.f32 	%r284, %r281, %r186;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r54, %r284, %r283;
	.loc	1 109 47                        // sk09_norm_embed.py:109:47
	// begin inline asm
	@%p1 st.global.v4.b32 [ %rd9 + 0 ], { %r39, %r40, %r41, %r42 };
	// end inline asm
	// begin inline asm
	@%p2 st.global.v4.b32 [ %rd10 + 0 ], { %r43, %r44, %r45, %r46 };
	// end inline asm
	// begin inline asm
	@%p3 st.global.v4.b32 [ %rd11 + 0 ], { %r47, %r48, %r49, %r50 };
	// end inline asm
	// begin inline asm
	@%p4 st.global.v4.b32 [ %rd12 + 0 ], { %r51, %r52, %r53, %r54 };
	// end inline asm
	.loc	1 109 4                         // sk09_norm_embed.py:109:4
	ret;
$L__tmp20:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk09_norm_embed.py"
	.file	2 "/usr/local/lib/python3.12/dist-packages/triton/language/standard.py"
	.section	.debug_abbrev
	{
.b8 1                                   // Abbreviation Code
.b8 17                                  // DW_TAG_compile_unit
.b8 1                                   // DW_CHILDREN_yes
.b8 37                                  // DW_AT_producer
.b8 8                                   // DW_FORM_string
.b8 19                                  // DW_AT_language
.b8 5                                   // DW_FORM_data2
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 16                                  // DW_AT_stmt_list
.b8 6                                   // DW_FORM_data4
.b8 27                                  // DW_AT_comp_dir
.b8 8                                   // DW_FORM_string
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 2                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 0                                   // DW_CHILDREN_no
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 32                                  // DW_AT_inline
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 3                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 1                                   // DW_CHILDREN_yes
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 4                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 1                                   // DW_CHILDREN_yes
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 11                                  // DW_FORM_data1
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 5                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 0                                   // DW_CHILDREN_no
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 5                                   // DW_FORM_data2
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 164                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x9d DW_TAG_compile_unit
.b8 116                                 // DW_AT_producer
.b8 114
.b8 105
.b8 116
.b8 111
.b8 110
.b8 0
.b8 2                                   // DW_AT_language
.b8 0
.b8 115                                 // DW_AT_name
.b8 107
.b8 48
.b8 57
.b8 95
.b8 110
.b8 111
.b8 114
.b8 109
.b8 95
.b8 101
.b8 109
.b8 98
.b8 101
.b8 100
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 114
.b8 101
.b8 112
.b8 111
.b8 47
.b8 118
.b8 108
.b8 108
.b8 109
.b8 47
.b8 95
.b8 103
.b8 101
.b8 110
.b8 101
.b8 115
.b8 105
.b8 115
.b8 47
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 115
.b8 0
.b8 2                                   // Abbrev [2] 0x48:0x17 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 57
.b8 95
.b8 114
.b8 109
.b8 115
.b8 110
.b8 111
.b8 114
.b8 109
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x5f:0x48 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 72                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x74:0x32 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp19                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 108                                 // DW_AT_call_line
.b8 28                                  // DW_AT_call_column
.b8 5                                   // Abbrev [5] 0x8c:0x19 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp18                          // DW_AT_high_pc
.b8 2                                   // DW_AT_call_file
.b8 37                                  // DW_AT_call_line
.b8 1
.b8 36                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_QVAR2_2 = _Nativo(
    "sk09_norm_embed/_sk09_rmsnorm_kernel/blk8192",
    _QPTX2_2, "_sk09_rmsnorm_kernel",
    warps=8, shared=32,
    abi=[0, 1, 2, 3, 4, 5],
    horneado={6: 8192, 7: 1e-06, 8: 0.0},
    div16=[3, 4, 5],
)
# E1=0 lop3, E2=0 mul

_QPTX2_3 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk09_rmsnorm_kernel    // -- Begin function _sk09_rmsnorm_kernel
.extern .shared .align 16 .b8 global_smem[];
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
                                        // @_sk09_rmsnorm_kernel
.visible .entry _sk09_rmsnorm_kernel(
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_kernel_param_2,
	.param .u32 _sk09_rmsnorm_kernel_param_3,
	.param .u32 _sk09_rmsnorm_kernel_param_4,
	.param .u32 _sk09_rmsnorm_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_kernel_param_6,
	.param .u64 .ptr .global .align 1 _sk09_rmsnorm_kernel_param_7
)
.reqntid 256
{
	.reg .pred 	%p<13>;
	.reg .b16 	%rs<129>;
	.reg .b32 	%r<529>;
	.reg .b64 	%rd<33>;
	.loc	1 98 0                          // sk09_norm_embed.py:98:0
$L__func_begin0:
	.loc	1 98 0                          // sk09_norm_embed.py:98:0

// %bb.0:                               // %__nv_rsqrtf.exit
	ld.param.b64 	%rd25, [_sk09_rmsnorm_kernel_param_0];
	ld.param.b64 	%rd26, [_sk09_rmsnorm_kernel_param_1];
$L__tmp0:
	.loc	1 103 24                        // sk09_norm_embed.py:103:24
	mov.u32 	%r103, %ctaid.x;
	ld.param.b64 	%rd27, [_sk09_rmsnorm_kernel_param_2];
	.loc	1 104 24                        // sk09_norm_embed.py:104:24
	mov.u32 	%r104, %tid.x;
	and.b32 	%r105, %r104, 255;
	ld.param.b32 	%r106, [_sk09_rmsnorm_kernel_param_3];
	and.b32 	%r107, %r104, 31;
	ld.param.b32 	%r108, [_sk09_rmsnorm_kernel_param_4];
	ld.param.b32 	%r109, [_sk09_rmsnorm_kernel_param_5];
	shl.b32 	%r110, %r104, 3;
	and.b32 	%r111, %r110, 2040;
	or.b32 	%r112, %r111, 2048;
	or.b32 	%r113, %r111, 4096;
	or.b32 	%r114, %r110, 6144;
	or.b32 	%r115, %r111, 8192;
	or.b32 	%r116, %r111, 10240;
	or.b32 	%r117, %r111, 12288;
	or.b32 	%r118, %r110, 14336;
	.loc	1 105 18                        // sk09_norm_embed.py:105:18
	setp.lt.s32 	%p1, %r111, %r106;
	setp.lt.s32 	%p2, %r112, %r106;
	setp.lt.s32 	%p3, %r113, %r106;
	setp.lt.s32 	%p4, %r114, %r106;
	setp.lt.s32 	%p5, %r115, %r106;
	setp.lt.s32 	%p6, %r116, %r106;
	setp.lt.s32 	%p7, %r117, %r106;
	setp.lt.s32 	%p8, %r118, %r106;
	.loc	1 106 30                        // sk09_norm_embed.py:106:30
	mul.lo.s32 	%r119, %r108, %r103;
	.loc	1 106 24                        // sk09_norm_embed.py:106:24
	mad.wide.s32 	%rd28, %r119, 2, %rd25;
	.loc	1 106 42                        // sk09_norm_embed.py:106:42
	mul.wide.u32 	%rd29, %r111, 2;
	add.s64 	%rd1, %rd28, %rd29;
	add.s64 	%rd2, %rd1, 4096;
	add.s64 	%rd3, %rd1, 8192;
	mul.wide.u32 	%rd30, %r114, 2;
	add.s64 	%rd4, %rd28, %rd30;
	add.s64 	%rd5, %rd1, 16384;
	add.s64 	%rd6, %rd1, 20480;
	add.s64 	%rd7, %rd1, 24576;
	mul.wide.u32 	%rd31, %r118, 2;
	add.s64 	%rd8, %rd28, %rd31;
	mov.b32 	%r5, 0;
	.loc	1 106 16                        // sk09_norm_embed.py:106:16
	// begin inline asm
	mov.u32 %r1, %r5;
	mov.u32 %r2, %r5;
	mov.u32 %r3, %r5;
	mov.u32 %r4, %r5;
	@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r6, %r5;
	mov.u32 %r7, %r5;
	mov.u32 %r8, %r5;
	mov.u32 %r9, %r5;
	@%p2 ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd2 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r10, %r5;
	mov.u32 %r11, %r5;
	mov.u32 %r12, %r5;
	mov.u32 %r13, %r5;
	@%p3 ld.global.v4.b32 { %r10, %r11, %r12, %r13 }, [ %rd3 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r14, %r5;
	mov.u32 %r15, %r5;
	mov.u32 %r16, %r5;
	mov.u32 %r17, %r5;
	@%p4 ld.global.v4.b32 { %r14, %r15, %r16, %r17 }, [ %rd4 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r18, %r5;
	mov.u32 %r19, %r5;
	mov.u32 %r20, %r5;
	mov.u32 %r21, %r5;
	@%p5 ld.global.v4.b32 { %r18, %r19, %r20, %r21 }, [ %rd5 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r22, %r5;
	mov.u32 %r23, %r5;
	mov.u32 %r24, %r5;
	mov.u32 %r25, %r5;
	@%p6 ld.global.v4.b32 { %r22, %r23, %r24, %r25 }, [ %rd6 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r26, %r5;
	mov.u32 %r27, %r5;
	mov.u32 %r28, %r5;
	mov.u32 %r29, %r5;
	@%p7 ld.global.v4.b32 { %r26, %r27, %r28, %r29 }, [ %rd7 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r30, %r5;
	mov.u32 %r31, %r5;
	mov.u32 %r32, %r5;
	mov.u32 %r33, %r5;
	@%p8 ld.global.v4.b32 { %r30, %r31, %r32, %r33 }, [ %rd8 + 0 ];
	// end inline asm
	.loc	1 106 73                        // sk09_norm_embed.py:106:73
	mov.b32 	{%rs1, %rs2}, %r1;
	cvt.f32.bf16 	%r120, %rs1;
	cvt.f32.bf16 	%r121, %rs2;
	mov.b32 	{%rs3, %rs4}, %r2;
	cvt.f32.bf16 	%r122, %rs4;
	cvt.f32.bf16 	%r123, %rs3;
	mov.b32 	{%rs5, %rs6}, %r3;
	cvt.f32.bf16 	%r124, %rs6;
	cvt.f32.bf16 	%r125, %rs5;
	mov.b32 	{%rs7, %rs8}, %r4;
	cvt.f32.bf16 	%r126, %rs8;
	cvt.f32.bf16 	%r127, %rs7;
	mov.b32 	{%rs9, %rs10}, %r6;
	cvt.f32.bf16 	%r128, %rs10;
	cvt.f32.bf16 	%r129, %rs9;
	mov.b32 	{%rs11, %rs12}, %r7;
	cvt.f32.bf16 	%r130, %rs12;
	cvt.f32.bf16 	%r131, %rs11;
	mov.b32 	{%rs13, %rs14}, %r8;
	cvt.f32.bf16 	%r132, %rs14;
	cvt.f32.bf16 	%r133, %rs13;
	mov.b32 	{%rs15, %rs16}, %r9;
	cvt.f32.bf16 	%r134, %rs16;
	cvt.f32.bf16 	%r135, %rs15;
	mov.b32 	{%rs17, %rs18}, %r10;
	cvt.f32.bf16 	%r136, %rs18;
	cvt.f32.bf16 	%r137, %rs17;
	mov.b32 	{%rs19, %rs20}, %r11;
	cvt.f32.bf16 	%r138, %rs20;
	cvt.f32.bf16 	%r139, %rs19;
	mov.b32 	{%rs21, %rs22}, %r12;
	cvt.f32.bf16 	%r140, %rs22;
	cvt.f32.bf16 	%r141, %rs21;
	mov.b32 	{%rs23, %rs24}, %r13;
	cvt.f32.bf16 	%r142, %rs24;
	cvt.f32.bf16 	%r143, %rs23;
	mov.b32 	{%rs25, %rs26}, %r14;
	cvt.f32.bf16 	%r144, %rs26;
	cvt.f32.bf16 	%r145, %rs25;
	mov.b32 	{%rs27, %rs28}, %r15;
	cvt.f32.bf16 	%r146, %rs28;
	cvt.f32.bf16 	%r147, %rs27;
	mov.b32 	{%rs29, %rs30}, %r16;
	cvt.f32.bf16 	%r148, %rs30;
	cvt.f32.bf16 	%r149, %rs29;
	mov.b32 	{%rs31, %rs32}, %r17;
	cvt.f32.bf16 	%r150, %rs32;
	cvt.f32.bf16 	%r151, %rs31;
	mov.b32 	{%rs33, %rs34}, %r18;
	cvt.f32.bf16 	%r152, %rs34;
	cvt.f32.bf16 	%r153, %rs33;
	mov.b32 	{%rs35, %rs36}, %r19;
	cvt.f32.bf16 	%r154, %rs36;
	cvt.f32.bf16 	%r155, %rs35;
	mov.b32 	{%rs37, %rs38}, %r20;
	cvt.f32.bf16 	%r156, %rs38;
	cvt.f32.bf16 	%r157, %rs37;
	mov.b32 	{%rs39, %rs40}, %r21;
	cvt.f32.bf16 	%r158, %rs40;
	cvt.f32.bf16 	%r159, %rs39;
	mov.b32 	{%rs41, %rs42}, %r22;
	cvt.f32.bf16 	%r160, %rs42;
	cvt.f32.bf16 	%r161, %rs41;
	mov.b32 	{%rs43, %rs44}, %r23;
	cvt.f32.bf16 	%r162, %rs44;
	cvt.f32.bf16 	%r163, %rs43;
	mov.b32 	{%rs45, %rs46}, %r24;
	cvt.f32.bf16 	%r164, %rs46;
	cvt.f32.bf16 	%r165, %rs45;
	mov.b32 	{%rs47, %rs48}, %r25;
	cvt.f32.bf16 	%r166, %rs48;
	cvt.f32.bf16 	%r167, %rs47;
	mov.b32 	{%rs49, %rs50}, %r26;
	cvt.f32.bf16 	%r168, %rs50;
	cvt.f32.bf16 	%r169, %rs49;
	mov.b32 	{%rs51, %rs52}, %r27;
	cvt.f32.bf16 	%r170, %rs52;
	cvt.f32.bf16 	%r171, %rs51;
	mov.b32 	{%rs53, %rs54}, %r28;
	cvt.f32.bf16 	%r172, %rs54;
	cvt.f32.bf16 	%r173, %rs53;
	mov.b32 	{%rs55, %rs56}, %r29;
	cvt.f32.bf16 	%r174, %rs56;
	cvt.f32.bf16 	%r175, %rs55;
	mov.b32 	{%rs57, %rs58}, %r30;
	cvt.f32.bf16 	%r176, %rs58;
	cvt.f32.bf16 	%r177, %rs57;
	mov.b32 	{%rs59, %rs60}, %r31;
	cvt.f32.bf16 	%r178, %rs60;
	cvt.f32.bf16 	%r179, %rs59;
	mov.b32 	{%rs61, %rs62}, %r32;
	cvt.f32.bf16 	%r180, %rs62;
	cvt.f32.bf16 	%r181, %rs61;
	mov.b32 	{%rs63, %rs64}, %r33;
	cvt.f32.bf16 	%r182, %rs64;
	cvt.f32.bf16 	%r183, %rs63;
	.loc	1 107 24                        // sk09_norm_embed.py:107:24
	add.s64 	%rd9, %rd26, %rd29;
	add.s64 	%rd10, %rd9, 4096;
	add.s64 	%rd11, %rd9, 8192;
	add.s64 	%rd12, %rd26, %rd30;
	add.s64 	%rd13, %rd9, 16384;
	add.s64 	%rd14, %rd9, 20480;
	add.s64 	%rd15, %rd9, 24576;
	add.s64 	%rd16, %rd26, %rd31;
	.loc	1 107 16                        // sk09_norm_embed.py:107:16
	// begin inline asm
	mov.u32 %r34, %r5;
	mov.u32 %r35, %r5;
	mov.u32 %r36, %r5;
	mov.u32 %r37, %r5;
	@%p1 ld.global.v4.b32 { %r34, %r35, %r36, %r37 }, [ %rd9 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r38, %r5;
	mov.u32 %r39, %r5;
	mov.u32 %r40, %r5;
	mov.u32 %r41, %r5;
	@%p2 ld.global.v4.b32 { %r38, %r39, %r40, %r41 }, [ %rd10 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r42, %r5;
	mov.u32 %r43, %r5;
	mov.u32 %r44, %r5;
	mov.u32 %r45, %r5;
	@%p3 ld.global.v4.b32 { %r42, %r43, %r44, %r45 }, [ %rd11 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r46, %r5;
	mov.u32 %r47, %r5;
	mov.u32 %r48, %r5;
	mov.u32 %r49, %r5;
	@%p4 ld.global.v4.b32 { %r46, %r47, %r48, %r49 }, [ %rd12 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r50, %r5;
	mov.u32 %r51, %r5;
	mov.u32 %r52, %r5;
	mov.u32 %r53, %r5;
	@%p5 ld.global.v4.b32 { %r50, %r51, %r52, %r53 }, [ %rd13 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r54, %r5;
	mov.u32 %r55, %r5;
	mov.u32 %r56, %r5;
	mov.u32 %r57, %r5;
	@%p6 ld.global.v4.b32 { %r54, %r55, %r56, %r57 }, [ %rd14 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r58, %r5;
	mov.u32 %r59, %r5;
	mov.u32 %r60, %r5;
	mov.u32 %r61, %r5;
	@%p7 ld.global.v4.b32 { %r58, %r59, %r60, %r61 }, [ %rd15 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r62, %r5;
	mov.u32 %r63, %r5;
	mov.u32 %r64, %r5;
	mov.u32 %r65, %r5;
	@%p8 ld.global.v4.b32 { %r62, %r63, %r64, %r65 }, [ %rd16 + 0 ];
	// end inline asm
	.loc	1 108 32                        // sk09_norm_embed.py:108:32
	mul.f32 	%r184, %r121, %r121;
$L__tmp1:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	fma.rn.f32 	%r185, %r120, %r120, %r184;
	fma.rn.f32 	%r186, %r123, %r123, %r185;
	fma.rn.f32 	%r187, %r122, %r122, %r186;
	fma.rn.f32 	%r188, %r125, %r125, %r187;
	fma.rn.f32 	%r189, %r124, %r124, %r188;
	fma.rn.f32 	%r190, %r127, %r127, %r189;
	fma.rn.f32 	%r191, %r126, %r126, %r190;
	fma.rn.f32 	%r192, %r129, %r129, %r191;
	fma.rn.f32 	%r193, %r128, %r128, %r192;
	fma.rn.f32 	%r194, %r131, %r131, %r193;
	fma.rn.f32 	%r195, %r130, %r130, %r194;
	fma.rn.f32 	%r196, %r133, %r133, %r195;
	fma.rn.f32 	%r197, %r132, %r132, %r196;
	fma.rn.f32 	%r198, %r135, %r135, %r197;
	fma.rn.f32 	%r199, %r134, %r134, %r198;
	fma.rn.f32 	%r200, %r137, %r137, %r199;
	fma.rn.f32 	%r201, %r136, %r136, %r200;
	fma.rn.f32 	%r202, %r139, %r139, %r201;
	fma.rn.f32 	%r203, %r138, %r138, %r202;
	fma.rn.f32 	%r204, %r141, %r141, %r203;
	fma.rn.f32 	%r205, %r140, %r140, %r204;
	fma.rn.f32 	%r206, %r143, %r143, %r205;
	fma.rn.f32 	%r207, %r142, %r142, %r206;
	fma.rn.f32 	%r208, %r145, %r145, %r207;
	fma.rn.f32 	%r209, %r144, %r144, %r208;
	fma.rn.f32 	%r210, %r147, %r147, %r209;
	fma.rn.f32 	%r211, %r146, %r146, %r210;
	fma.rn.f32 	%r212, %r149, %r149, %r211;
	fma.rn.f32 	%r213, %r148, %r148, %r212;
	fma.rn.f32 	%r214, %r151, %r151, %r213;
	fma.rn.f32 	%r215, %r150, %r150, %r214;
	fma.rn.f32 	%r216, %r153, %r153, %r215;
	fma.rn.f32 	%r217, %r152, %r152, %r216;
	fma.rn.f32 	%r218, %r155, %r155, %r217;
	fma.rn.f32 	%r219, %r154, %r154, %r218;
	fma.rn.f32 	%r220, %r157, %r157, %r219;
	fma.rn.f32 	%r221, %r156, %r156, %r220;
	fma.rn.f32 	%r222, %r159, %r159, %r221;
	fma.rn.f32 	%r223, %r158, %r158, %r222;
	fma.rn.f32 	%r224, %r161, %r161, %r223;
	fma.rn.f32 	%r225, %r160, %r160, %r224;
	fma.rn.f32 	%r226, %r163, %r163, %r225;
	fma.rn.f32 	%r227, %r162, %r162, %r226;
	fma.rn.f32 	%r228, %r165, %r165, %r227;
	fma.rn.f32 	%r229, %r164, %r164, %r228;
	fma.rn.f32 	%r230, %r167, %r167, %r229;
	fma.rn.f32 	%r231, %r166, %r166, %r230;
	fma.rn.f32 	%r232, %r169, %r169, %r231;
	fma.rn.f32 	%r233, %r168, %r168, %r232;
	fma.rn.f32 	%r234, %r171, %r171, %r233;
	fma.rn.f32 	%r235, %r170, %r170, %r234;
	fma.rn.f32 	%r236, %r173, %r173, %r235;
	fma.rn.f32 	%r237, %r172, %r172, %r236;
	fma.rn.f32 	%r238, %r175, %r175, %r237;
	fma.rn.f32 	%r239, %r174, %r174, %r238;
	fma.rn.f32 	%r240, %r177, %r177, %r239;
	fma.rn.f32 	%r241, %r176, %r176, %r240;
	fma.rn.f32 	%r242, %r179, %r179, %r241;
	fma.rn.f32 	%r243, %r178, %r178, %r242;
	fma.rn.f32 	%r244, %r181, %r181, %r243;
	fma.rn.f32 	%r245, %r180, %r180, %r244;
	fma.rn.f32 	%r246, %r183, %r183, %r245;
	fma.rn.f32 	%r247, %r182, %r182, %r246;
$L__tmp2:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	shfl.sync.bfly.b32 	%r248, %r247, 16, 31, -1;
$L__tmp3:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	add.f32 	%r249, %r247, %r248;
$L__tmp4:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	shfl.sync.bfly.b32 	%r250, %r249, 8, 31, -1;
$L__tmp5:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	add.f32 	%r251, %r249, %r250;
$L__tmp6:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	shfl.sync.bfly.b32 	%r252, %r251, 4, 31, -1;
$L__tmp7:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	add.f32 	%r253, %r251, %r252;
$L__tmp8:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	shfl.sync.bfly.b32 	%r254, %r253, 2, 31, -1;
$L__tmp9:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	add.f32 	%r255, %r253, %r254;
$L__tmp10:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	shfl.sync.bfly.b32 	%r256, %r255, 1, 31, -1;
$L__tmp11:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	add.f32 	%r67, %r255, %r256;
$L__tmp12:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	setp.eq.b32 	%p9, %r107, 0;
	shr.u32 	%r257, %r104, 3;
	and.b32 	%r258, %r257, 28;
	mov.b32 	%r259, global_smem;
	add.s32 	%r66, %r259, %r258;
	// begin inline asm
	@%p9 st.shared.b32 [ %r66 + 0 ], %r67;
	// end inline asm
	bar.sync 	0;
	setp.lt.u32 	%p10, %r105, 8;
	shl.b32 	%r260, %r105, 2;
	add.s32 	%r69, %r259, %r260;
	// begin inline asm
	@%p10 ld.shared.b32 %r68, [ %r69 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r261, %r68, 4, 31, -1;
$L__tmp13:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	add.f32 	%r262, %r68, %r261;
$L__tmp14:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	shfl.sync.bfly.b32 	%r263, %r262, 2, 31, -1;
$L__tmp15:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	add.f32 	%r264, %r262, %r263;
$L__tmp16:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	shfl.sync.bfly.b32 	%r265, %r264, 1, 31, -1;
$L__tmp17:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:108:28 ] ]
	add.f32 	%r70, %r264, %r265;
$L__tmp18:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:108:28 ]
	and.b32 	%r266, %r104, 7;
	setp.eq.b32 	%p12, %r266, 0;
	and.pred 	%p11, %p10, %p12;
	// begin inline asm
	@%p11 st.shared.b32 [ %r69 + 0 ], %r70;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r267, [global_smem];
$L__tmp19:
	.loc	1 108 45                        // sk09_norm_embed.py:108:45
	cvt.rn.f32.s32 	%r268, %r106;
	div.full.f32 	%r269, %r267, %r268;
	.loc	1 108 49                        // sk09_norm_embed.py:108:49
	add.f32 	%r270, %r269, 0f358637BD;
	.loc	1 108 21                        // sk09_norm_embed.py:108:21
	rsqrt.approx.ftz.f32 	%r271, %r270;
	.loc	1 108 12                        // sk09_norm_embed.py:108:12
	mul.f32 	%r272, %r271, %r121;
	mul.f32 	%r273, %r271, %r120;
	mul.f32 	%r274, %r271, %r122;
	mul.f32 	%r275, %r271, %r123;
	mul.f32 	%r276, %r271, %r124;
	mul.f32 	%r277, %r271, %r125;
	mul.f32 	%r278, %r271, %r126;
	mul.f32 	%r279, %r271, %r127;
	mul.f32 	%r280, %r271, %r128;
	mul.f32 	%r281, %r271, %r129;
	mul.f32 	%r282, %r271, %r130;
	mul.f32 	%r283, %r271, %r131;
	mul.f32 	%r284, %r271, %r132;
	mul.f32 	%r285, %r271, %r133;
	mul.f32 	%r286, %r271, %r134;
	mul.f32 	%r287, %r271, %r135;
	mul.f32 	%r288, %r271, %r136;
	mul.f32 	%r289, %r271, %r137;
	mul.f32 	%r290, %r271, %r138;
	mul.f32 	%r291, %r271, %r139;
	mul.f32 	%r292, %r271, %r140;
	mul.f32 	%r293, %r271, %r141;
	mul.f32 	%r294, %r271, %r142;
	mul.f32 	%r295, %r271, %r143;
	mul.f32 	%r296, %r271, %r144;
	mul.f32 	%r297, %r271, %r145;
	mul.f32 	%r298, %r271, %r146;
	mul.f32 	%r299, %r271, %r147;
	mul.f32 	%r300, %r271, %r148;
	mul.f32 	%r301, %r271, %r149;
	mul.f32 	%r302, %r271, %r150;
	mul.f32 	%r303, %r271, %r151;
	mul.f32 	%r304, %r271, %r152;
	mul.f32 	%r305, %r271, %r153;
	mul.f32 	%r306, %r271, %r154;
	mul.f32 	%r307, %r271, %r155;
	mul.f32 	%r308, %r271, %r156;
	mul.f32 	%r309, %r271, %r157;
	mul.f32 	%r310, %r271, %r158;
	mul.f32 	%r311, %r271, %r159;
	mul.f32 	%r312, %r271, %r160;
	mul.f32 	%r313, %r271, %r161;
	mul.f32 	%r314, %r271, %r162;
	mul.f32 	%r315, %r271, %r163;
	mul.f32 	%r316, %r271, %r164;
	mul.f32 	%r317, %r271, %r165;
	mul.f32 	%r318, %r271, %r166;
	mul.f32 	%r319, %r271, %r167;
	mul.f32 	%r320, %r271, %r168;
	mul.f32 	%r321, %r271, %r169;
	mul.f32 	%r322, %r271, %r170;
	mul.f32 	%r323, %r271, %r171;
	mul.f32 	%r324, %r271, %r172;
	mul.f32 	%r325, %r271, %r173;
	mul.f32 	%r326, %r271, %r174;
	mul.f32 	%r327, %r271, %r175;
	mul.f32 	%r328, %r271, %r176;
	mul.f32 	%r329, %r271, %r177;
	mul.f32 	%r330, %r271, %r178;
	mul.f32 	%r331, %r271, %r179;
	mul.f32 	%r332, %r271, %r180;
	mul.f32 	%r333, %r271, %r181;
	mul.f32 	%r334, %r271, %r182;
	mul.f32 	%r335, %r271, %r183;
	.loc	1 109 29                        // sk09_norm_embed.py:109:29
	mul.lo.s32 	%r336, %r109, %r103;
	.loc	1 109 23                        // sk09_norm_embed.py:109:23
	mad.wide.s32 	%rd32, %r336, 2, %rd27;
	.loc	1 109 41                        // sk09_norm_embed.py:109:41
	add.s64 	%rd17, %rd32, %rd29;
	add.s64 	%rd18, %rd17, 4096;
	add.s64 	%rd19, %rd17, 8192;
	add.s64 	%rd20, %rd32, %rd30;
	add.s64 	%rd21, %rd17, 16384;
	add.s64 	%rd22, %rd17, 20480;
	add.s64 	%rd23, %rd17, 24576;
	add.s64 	%rd24, %rd32, %rd31;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs65, %rs66}, %r34;
	cvt.f32.bf16 	%r337, %rs65;
	cvt.f32.bf16 	%r338, %rs66;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r339, %r338, 0f00000000;
	add.f32 	%r340, %r337, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r341, %r340, %r273;
	mul.f32 	%r342, %r339, %r272;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r71, %r342, %r341;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs67, %rs68}, %r35;
	cvt.f32.bf16 	%r343, %rs67;
	cvt.f32.bf16 	%r344, %rs68;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r345, %r344, 0f00000000;
	add.f32 	%r346, %r343, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r347, %r346, %r275;
	mul.f32 	%r348, %r345, %r274;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r72, %r348, %r347;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs69, %rs70}, %r36;
	cvt.f32.bf16 	%r349, %rs69;
	cvt.f32.bf16 	%r350, %rs70;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r351, %r350, 0f00000000;
	add.f32 	%r352, %r349, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r353, %r352, %r277;
	mul.f32 	%r354, %r351, %r276;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r73, %r354, %r353;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs71, %rs72}, %r37;
	cvt.f32.bf16 	%r355, %rs71;
	cvt.f32.bf16 	%r356, %rs72;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r357, %r356, 0f00000000;
	add.f32 	%r358, %r355, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r359, %r358, %r279;
	mul.f32 	%r360, %r357, %r278;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r74, %r360, %r359;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs73, %rs74}, %r38;
	cvt.f32.bf16 	%r361, %rs73;
	cvt.f32.bf16 	%r362, %rs74;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r363, %r362, 0f00000000;
	add.f32 	%r364, %r361, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r365, %r364, %r281;
	mul.f32 	%r366, %r363, %r280;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r75, %r366, %r365;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs75, %rs76}, %r39;
	cvt.f32.bf16 	%r367, %rs75;
	cvt.f32.bf16 	%r368, %rs76;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r369, %r368, 0f00000000;
	add.f32 	%r370, %r367, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r371, %r370, %r283;
	mul.f32 	%r372, %r369, %r282;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r76, %r372, %r371;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs77, %rs78}, %r40;
	cvt.f32.bf16 	%r373, %rs77;
	cvt.f32.bf16 	%r374, %rs78;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r375, %r374, 0f00000000;
	add.f32 	%r376, %r373, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r377, %r376, %r285;
	mul.f32 	%r378, %r375, %r284;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r77, %r378, %r377;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs79, %rs80}, %r41;
	cvt.f32.bf16 	%r379, %rs79;
	cvt.f32.bf16 	%r380, %rs80;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r381, %r380, 0f00000000;
	add.f32 	%r382, %r379, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r383, %r382, %r287;
	mul.f32 	%r384, %r381, %r286;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r78, %r384, %r383;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs81, %rs82}, %r42;
	cvt.f32.bf16 	%r385, %rs81;
	cvt.f32.bf16 	%r386, %rs82;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r387, %r386, 0f00000000;
	add.f32 	%r388, %r385, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r389, %r388, %r289;
	mul.f32 	%r390, %r387, %r288;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r79, %r390, %r389;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs83, %rs84}, %r43;
	cvt.f32.bf16 	%r391, %rs83;
	cvt.f32.bf16 	%r392, %rs84;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r393, %r392, 0f00000000;
	add.f32 	%r394, %r391, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r395, %r394, %r291;
	mul.f32 	%r396, %r393, %r290;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r80, %r396, %r395;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs85, %rs86}, %r44;
	cvt.f32.bf16 	%r397, %rs85;
	cvt.f32.bf16 	%r398, %rs86;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r399, %r398, 0f00000000;
	add.f32 	%r400, %r397, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r401, %r400, %r293;
	mul.f32 	%r402, %r399, %r292;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r81, %r402, %r401;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs87, %rs88}, %r45;
	cvt.f32.bf16 	%r403, %rs87;
	cvt.f32.bf16 	%r404, %rs88;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r405, %r404, 0f00000000;
	add.f32 	%r406, %r403, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r407, %r406, %r295;
	mul.f32 	%r408, %r405, %r294;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r82, %r408, %r407;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs89, %rs90}, %r46;
	cvt.f32.bf16 	%r409, %rs89;
	cvt.f32.bf16 	%r410, %rs90;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r411, %r410, 0f00000000;
	add.f32 	%r412, %r409, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r413, %r412, %r297;
	mul.f32 	%r414, %r411, %r296;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r83, %r414, %r413;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs91, %rs92}, %r47;
	cvt.f32.bf16 	%r415, %rs91;
	cvt.f32.bf16 	%r416, %rs92;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r417, %r416, 0f00000000;
	add.f32 	%r418, %r415, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r419, %r418, %r299;
	mul.f32 	%r420, %r417, %r298;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r84, %r420, %r419;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs93, %rs94}, %r48;
	cvt.f32.bf16 	%r421, %rs93;
	cvt.f32.bf16 	%r422, %rs94;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r423, %r422, 0f00000000;
	add.f32 	%r424, %r421, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r425, %r424, %r301;
	mul.f32 	%r426, %r423, %r300;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r85, %r426, %r425;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs95, %rs96}, %r49;
	cvt.f32.bf16 	%r427, %rs95;
	cvt.f32.bf16 	%r428, %rs96;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r429, %r428, 0f00000000;
	add.f32 	%r430, %r427, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r431, %r430, %r303;
	mul.f32 	%r432, %r429, %r302;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r86, %r432, %r431;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs97, %rs98}, %r50;
	cvt.f32.bf16 	%r433, %rs97;
	cvt.f32.bf16 	%r434, %rs98;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r435, %r434, 0f00000000;
	add.f32 	%r436, %r433, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r437, %r436, %r305;
	mul.f32 	%r438, %r435, %r304;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r87, %r438, %r437;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs99, %rs100}, %r51;
	cvt.f32.bf16 	%r439, %rs99;
	cvt.f32.bf16 	%r440, %rs100;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r441, %r440, 0f00000000;
	add.f32 	%r442, %r439, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r443, %r442, %r307;
	mul.f32 	%r444, %r441, %r306;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r88, %r444, %r443;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs101, %rs102}, %r52;
	cvt.f32.bf16 	%r445, %rs101;
	cvt.f32.bf16 	%r446, %rs102;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r447, %r446, 0f00000000;
	add.f32 	%r448, %r445, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r449, %r448, %r309;
	mul.f32 	%r450, %r447, %r308;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r89, %r450, %r449;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs103, %rs104}, %r53;
	cvt.f32.bf16 	%r451, %rs103;
	cvt.f32.bf16 	%r452, %rs104;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r453, %r452, 0f00000000;
	add.f32 	%r454, %r451, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r455, %r454, %r311;
	mul.f32 	%r456, %r453, %r310;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r90, %r456, %r455;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs105, %rs106}, %r54;
	cvt.f32.bf16 	%r457, %rs105;
	cvt.f32.bf16 	%r458, %rs106;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r459, %r458, 0f00000000;
	add.f32 	%r460, %r457, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r461, %r460, %r313;
	mul.f32 	%r462, %r459, %r312;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r91, %r462, %r461;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs107, %rs108}, %r55;
	cvt.f32.bf16 	%r463, %rs107;
	cvt.f32.bf16 	%r464, %rs108;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r465, %r464, 0f00000000;
	add.f32 	%r466, %r463, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r467, %r466, %r315;
	mul.f32 	%r468, %r465, %r314;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r92, %r468, %r467;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs109, %rs110}, %r56;
	cvt.f32.bf16 	%r469, %rs109;
	cvt.f32.bf16 	%r470, %rs110;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r471, %r470, 0f00000000;
	add.f32 	%r472, %r469, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r473, %r472, %r317;
	mul.f32 	%r474, %r471, %r316;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r93, %r474, %r473;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs111, %rs112}, %r57;
	cvt.f32.bf16 	%r475, %rs111;
	cvt.f32.bf16 	%r476, %rs112;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r477, %r476, 0f00000000;
	add.f32 	%r478, %r475, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r479, %r478, %r319;
	mul.f32 	%r480, %r477, %r318;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r94, %r480, %r479;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs113, %rs114}, %r58;
	cvt.f32.bf16 	%r481, %rs113;
	cvt.f32.bf16 	%r482, %rs114;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r483, %r482, 0f00000000;
	add.f32 	%r484, %r481, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r485, %r484, %r321;
	mul.f32 	%r486, %r483, %r320;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r95, %r486, %r485;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs115, %rs116}, %r59;
	cvt.f32.bf16 	%r487, %rs115;
	cvt.f32.bf16 	%r488, %rs116;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r489, %r488, 0f00000000;
	add.f32 	%r490, %r487, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r491, %r490, %r323;
	mul.f32 	%r492, %r489, %r322;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r96, %r492, %r491;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs117, %rs118}, %r60;
	cvt.f32.bf16 	%r493, %rs117;
	cvt.f32.bf16 	%r494, %rs118;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r495, %r494, 0f00000000;
	add.f32 	%r496, %r493, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r497, %r496, %r325;
	mul.f32 	%r498, %r495, %r324;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r97, %r498, %r497;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs119, %rs120}, %r61;
	cvt.f32.bf16 	%r499, %rs119;
	cvt.f32.bf16 	%r500, %rs120;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r501, %r500, 0f00000000;
	add.f32 	%r502, %r499, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r503, %r502, %r327;
	mul.f32 	%r504, %r501, %r326;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r98, %r504, %r503;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs121, %rs122}, %r62;
	cvt.f32.bf16 	%r505, %rs121;
	cvt.f32.bf16 	%r506, %rs122;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r507, %r506, 0f00000000;
	add.f32 	%r508, %r505, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r509, %r508, %r329;
	mul.f32 	%r510, %r507, %r328;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r99, %r510, %r509;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs123, %rs124}, %r63;
	cvt.f32.bf16 	%r511, %rs123;
	cvt.f32.bf16 	%r512, %rs124;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r513, %r512, 0f00000000;
	add.f32 	%r514, %r511, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r515, %r514, %r331;
	mul.f32 	%r516, %r513, %r330;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r100, %r516, %r515;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs125, %rs126}, %r64;
	cvt.f32.bf16 	%r517, %rs125;
	cvt.f32.bf16 	%r518, %rs126;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r519, %r518, 0f00000000;
	add.f32 	%r520, %r517, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r521, %r520, %r333;
	mul.f32 	%r522, %r519, %r332;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r101, %r522, %r521;
	.loc	1 107 55                        // sk09_norm_embed.py:107:55
	mov.b32 	{%rs127, %rs128}, %r65;
	cvt.f32.bf16 	%r523, %rs127;
	cvt.f32.bf16 	%r524, %rs128;
	.loc	1 107 69                        // sk09_norm_embed.py:107:69
	add.f32 	%r525, %r524, 0f00000000;
	add.f32 	%r526, %r523, 0f00000000;
	.loc	1 108 56                        // sk09_norm_embed.py:108:56
	mul.f32 	%r527, %r526, %r335;
	mul.f32 	%r528, %r525, %r334;
	.loc	1 109 52                        // sk09_norm_embed.py:109:52
	cvt.rn.bf16x2.f32 	%r102, %r528, %r527;
	.loc	1 109 47                        // sk09_norm_embed.py:109:47
	// begin inline asm
	@%p1 st.global.v4.b32 [ %rd17 + 0 ], { %r71, %r72, %r73, %r74 };
	// end inline asm
	// begin inline asm
	@%p2 st.global.v4.b32 [ %rd18 + 0 ], { %r75, %r76, %r77, %r78 };
	// end inline asm
	// begin inline asm
	@%p3 st.global.v4.b32 [ %rd19 + 0 ], { %r79, %r80, %r81, %r82 };
	// end inline asm
	// begin inline asm
	@%p4 st.global.v4.b32 [ %rd20 + 0 ], { %r83, %r84, %r85, %r86 };
	// end inline asm
	// begin inline asm
	@%p5 st.global.v4.b32 [ %rd21 + 0 ], { %r87, %r88, %r89, %r90 };
	// end inline asm
	// begin inline asm
	@%p6 st.global.v4.b32 [ %rd22 + 0 ], { %r91, %r92, %r93, %r94 };
	// end inline asm
	// begin inline asm
	@%p7 st.global.v4.b32 [ %rd23 + 0 ], { %r95, %r96, %r97, %r98 };
	// end inline asm
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd24 + 0 ], { %r99, %r100, %r101, %r102 };
	// end inline asm
	.loc	1 109 4                         // sk09_norm_embed.py:109:4
	ret;
$L__tmp20:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk09_norm_embed.py"
	.file	2 "/usr/local/lib/python3.12/dist-packages/triton/language/standard.py"
	.section	.debug_abbrev
	{
.b8 1                                   // Abbreviation Code
.b8 17                                  // DW_TAG_compile_unit
.b8 1                                   // DW_CHILDREN_yes
.b8 37                                  // DW_AT_producer
.b8 8                                   // DW_FORM_string
.b8 19                                  // DW_AT_language
.b8 5                                   // DW_FORM_data2
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 16                                  // DW_AT_stmt_list
.b8 6                                   // DW_FORM_data4
.b8 27                                  // DW_AT_comp_dir
.b8 8                                   // DW_FORM_string
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 2                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 0                                   // DW_CHILDREN_no
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 32                                  // DW_AT_inline
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 3                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 1                                   // DW_CHILDREN_yes
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 4                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 1                                   // DW_CHILDREN_yes
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 11                                  // DW_FORM_data1
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 5                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 0                                   // DW_CHILDREN_no
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 5                                   // DW_FORM_data2
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 164                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x9d DW_TAG_compile_unit
.b8 116                                 // DW_AT_producer
.b8 114
.b8 105
.b8 116
.b8 111
.b8 110
.b8 0
.b8 2                                   // DW_AT_language
.b8 0
.b8 115                                 // DW_AT_name
.b8 107
.b8 48
.b8 57
.b8 95
.b8 110
.b8 111
.b8 114
.b8 109
.b8 95
.b8 101
.b8 109
.b8 98
.b8 101
.b8 100
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 114
.b8 101
.b8 112
.b8 111
.b8 47
.b8 118
.b8 108
.b8 108
.b8 109
.b8 47
.b8 95
.b8 103
.b8 101
.b8 110
.b8 101
.b8 115
.b8 105
.b8 115
.b8 47
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 115
.b8 0
.b8 2                                   // Abbrev [2] 0x48:0x17 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 57
.b8 95
.b8 114
.b8 109
.b8 115
.b8 110
.b8 111
.b8 114
.b8 109
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x5f:0x48 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 72                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x74:0x32 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp19                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 108                                 // DW_AT_call_line
.b8 28                                  // DW_AT_call_column
.b8 5                                   // Abbrev [5] 0x8c:0x19 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp18                          // DW_AT_high_pc
.b8 2                                   // DW_AT_call_file
.b8 37                                  // DW_AT_call_line
.b8 1
.b8 36                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_QVAR2_3 = _Nativo(
    "sk09_norm_embed/_sk09_rmsnorm_kernel/blk16384",
    _QPTX2_3, "_sk09_rmsnorm_kernel",
    warps=8, shared=32,
    abi=[0, 1, 2, 3, 4, 5],
    horneado={6: 16384, 7: 1e-06, 8: 0.0},
    div16=[3, 4, 5],
)
# E1=0 lop3, E2=0 mul

_POR_Q2 = [_QVAR2_0, _QVAR2_1, _QVAR2_2, _QVAR2_3]


def _q2_impl(grid: list[int], x_ptr: torch.Tensor, w_ptr: torch.Tensor, out_ptr: torch.Tensor, K: int, stride_xm: int, stride_om: int, BLOCK: int, EPS: float, GAMMA_OFFSET: float) -> None:
    """Cuerpo del custom op: elige la variante y lanza el PTX."""
    args = (x_ptr, w_ptr, out_ptr, K, stride_xm, stride_om, BLOCK, EPS, GAMMA_OFFSET)
    for _v in _POR_Q2:
        if all(args[_p] == _x for _p, _x in _v.horneado.items()):
            return _v(tuple(grid), *args)
    raise ValueError(
        "genesis_sk09_norm_embed_q2: no hay PTX embebido para estos constexpr; "
        "horneados disponibles: %r" % [_v.horneado for _v in _POR_Q2])


def _q2_fake(grid: list[int], x_ptr: torch.Tensor, w_ptr: torch.Tensor, out_ptr: torch.Tensor, K: int, stride_xm: int, stride_om: int, BLOCK: int, EPS: float, GAMMA_OFFSET: float) -> None:
    """Meta impl: no toca la GPU; las salidas se mutan in-place."""
    return None


# Registro AL IMPORTAR, no en el primer uso: direct_register_custom_op
# llama a torch._library.infer_schema, que dynamo se niega a trazar
# ("Attempted to call function marked as skipped"). El hasattr evita el
# choque cuando el modulo se importa dos veces con nombres distintos,
# como hace el gate de tools/monolitizar.py.
if not hasattr(torch.ops.vllm, "genesis_sk09_norm_embed_q2"):
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op(
        op_name="genesis_sk09_norm_embed_q2",
        op_func=_q2_impl,
        mutates_args=['out_ptr'],
        fake_impl=_q2_fake,
    )


def _lanzar_quant2(grid, *args):
    """Lanza el PTX embebido. Es el UNICO camino ejecutable.

    No hay fallback a Triton ni kill-switch: el kernel Triton de este
    archivo es privado y solo lo llaman los tests.
    """
    g = [grid] if isinstance(grid, int) else list(grid)
    return torch.ops.vllm.genesis_sk09_norm_embed_q2(g, *args)
