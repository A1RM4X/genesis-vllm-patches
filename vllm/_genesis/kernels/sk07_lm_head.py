# SPDX-License-Identifier: Apache-2.0
"""SK-07 — LM_HEAD_VOCAB — capa completa lm_head con gather de vocabulario en GPU.

Capa completa (Qwen3.5-27B, TP=2):
    hidden bf16 [M, 5120]
      -> quant per-token amax/127                      (kernel 1)
      -> GEMM INT8 diádico sobre las columnas pedidas   (kernel 2)
      -> logits [M, S]

Geometría: vocab global 248320, per-rank TP=2 V=124160; K=5120.

La ganancia algorítmica de SK-07 es el **gather**: en verificación MTP y en
sampling top-k sólo hacen falta S columnas del vocabulario, no las 124160. El
kernel indexa B por ``sampled`` en vez de recorrer V entero, así que el coste
baja de ``M*K*V`` a ``M*K*S``. Con el peso en layout column-major ``[K, V]``
(``stride_bk == 1``) cada columna gatherada es un tramo contiguo de K bytes, de
modo que el acceso sigue siendo coalescido.

El camino de vocabulario completo usa **el mismo kernel**, pasándole un
``arange(V)`` cacheado como índice: no hace falta una segunda especialización
ni ninguna rama.

Diseño (sm_86, mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32):
  * ``BLOCK_K == SHIFT_BLOCK == 128`` -> un ``tl.dot`` por bloque diádico.
  * Shift por columna como ``acc += int_acc.to(f32) * exp2(shift)``.
  * Acumulador fp32 y escalas fuera del bucle k. La versión anterior acumulaba
    el epílogo en bf16 y recalculaba ``amax`` dos veces por cada bloque de S.
  * Punteros que avanzan, grid 1-D con swizzle L2.

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




SK_ID = "SK-07"
SK_NAME = "LM_HEAD_VOCAB"
VOCAB_GLOBAL: int = 248320
VOCAB_PER_RANK: int = 124160
LM_HEAD_K: int = 5120

SHIFT_BLOCK: int = 128

_CFG: tuple[tuple[int, int, int, int, int, int], ...] = (
    # (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages) por bucket de M.
    # Medido en RTX 3090 con relojes fijos a 1500 MHz, CUDA events, salida
    # prealocada y configs intercaladas round-robin (sin eso el ruido es +-25%
    # y la busqueda devuelve resultados incoherentes).
    (16, 128, 128, 8, 8, 3),    # M <=   32   decode
    (128, 128, 128, 8, 8, 3),   # M <=  128
    (256, 128, 64, 8, 8, 4),    # M <= 1024
    (256, 128, 64, 8, 8, 4),    # M  > 1024   prefill
)
# Fraccion de ola por debajo de la cual conviene bajar BLOCK_M aunque se pierda
# intensidad aritmetica.
SM_COUNT: int = 82          # GA102 (RTX 3090)
_CTA_MIN: int = 49          # ~0.6 olas
_CFG_POCOS_CTA: tuple[int, int, int, int, int, int] = (64, 128, 128, 8, 4, 4)

BLOCK_M: int = 128
BLOCK_N: int = 128
BLOCK_S: int = 128
BLOCK_K: int = 128

_ARANGE: dict[tuple[torch.device, int], torch.Tensor] = {}


def _arange(device: torch.device, v: int) -> torch.Tensor:
    """``arange(V)`` cacheado: índice neutro para el camino de vocab completo."""
    key = (device, v)
    idx = _ARANGE.get(key)
    if idx is None:
        idx = torch.arange(v, dtype=torch.int32, device=device)
        _ARANGE[key] = idx
    return idx

_ZERO: dict[torch.device, torch.Tensor] = {}


def _zero(device: torch.device) -> torch.Tensor:
    """Escalar cero por device, usado como epílogo neutro con strides 0."""
    z = _ZERO.get(device)
    if z is None:
        z = torch.zeros((), dtype=torch.bfloat16, device=device)
        _ZERO[device] = z
    return z


def _cfg(m: int, n: int) -> tuple[int, int, int, int, int, int]:
    """Tile por bucket de M, con correccion cuando faltan CTAs.

    Se probo elegir el tile puramente por conteo de CTAs y salio PEOR que la
    tabla (mediana 0.93x, y hasta 0.35x a M=128): con BLOCK_M=16 el mma
    m16n8k32 queda al minimo y B se re-lee 8 veces. El conteo de CTAs solo
    manda cuando el tile de la tabla no llega ni a media ola, que es lo que
    pasa con N=5120 a M=128 (40 CTAs para 82 SM): ahi bajar a BLOCK_M=64 los
    duplica y da 1.41-1.45x.
    """
    c = _CFG[(m > 32) + (m > 128) + (m > 1024)]
    if m > 32 and -(-m // c[0]) * -(-n // c[1]) < _CTA_MIN:
        return _CFG_POCOS_CTA
    return c


@triton.jit
def _sk07_quant_kernel(x_ptr, q_ptr, s_ptr, K, stride_xm, stride_qm, BLOCK: tl.constexpr):
    """Quant per-token amax/127, una fila por programa."""
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
def _sk07_lm_head_kernel(
    a_ptr, b_ptr, out_ptr, idx_ptr, resid_ptr, a_scale_ptr, b_scale_ptr, shifts_ptr,
    M, S, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_out_m, stride_out_s, stride_res_m, stride_res_s,
    stride_shift_k, stride_shift_n,
    BLOCK_M: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, SHIFT_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_s = tl.cdiv(S, BLOCK_S)
    pid_in_group = GROUP_M * num_pid_s
    first_m = (pid // pid_in_group) * GROUP_M
    group_m = min(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + ((pid % pid_in_group) % group_m)
    pid_s = (pid % pid_in_group) // group_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_s = (pid_s * BLOCK_S + tl.arange(0, BLOCK_S)) % S
    offs_k = tl.arange(0, BLOCK_K)

    cols = tl.load(idx_ptr + offs_s).to(tl.int32)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + cols[None, :] * stride_bn
    sh_ptrs = shifts_ptr + (cols // SHIFT_BLOCK) * stride_shift_n

    acc = tl.zeros((BLOCK_M, BLOCK_S), dtype=tl.float32)
    for kb in range(0, K // BLOCK_K):
        # `kb * stride_shift_k` indexaba FUERA DE RANGO: el shift es por
        # bloque de SHIFT_BLOCK=128 filas de K, no por iteracion, y con
        # BLOCK_K=64 hay dos iteraciones por bloque diadico. Con K=3072
        # son 48 iteraciones sobre una tabla de 24 filas. Coincidian solo
        # con BLOCK_K=128, asi que el fallo aparecia recien en el tile de
        # prefill: CUDA_ERROR_ILLEGAL_ADDRESS al lanzar.
        p2 = tl.exp2(tl.load(sh_ptrs + (kb * BLOCK_K // SHIFT_BLOCK)
                             * stride_shift_k).to(tl.float32))
        acc += tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), out_dtype=tl.int32).to(tl.float32) * p2[None, :]
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    acc = acc * tl.load(a_scale_ptr + offs_m).to(tl.float32)[:, None]
    acc = acc * tl.load(b_scale_ptr + cols).to(tl.float32)[None, :]
    acc += tl.load(resid_ptr + offs_m[:, None] * stride_res_m + cols[None, :] * stride_res_s).to(tl.float32)

    out_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    tl.store(
        out_ptr + out_m[:, None] * stride_out_m + out_s[None, :] * stride_out_s,
        acc.to(out_ptr.dtype.element_ty),
        mask=(out_m[:, None] < M) & (out_s[None, :] < S),
    )


def sk07_quant(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quant per-token INT8 -> ``(q [M,K] int8, s [M] fp32)``."""
    M, K = hidden.shape
    q = torch.empty((M, K), dtype=torch.int8, device=hidden.device)
    s = torch.empty((M,), dtype=torch.float32, device=hidden.device)
    _lanzar_quant0((M,),
            hidden, q, s, K, hidden.stride(0), q.stride(0), triton.next_power_of_2(K))
    return q, s


def lm_head_gemm(
    a: torch.Tensor,
    weight: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    shifts: torch.Tensor,
    sampled_ids: torch.Tensor,
    residual: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """GEMM INT8 diádico sobre las columnas ``sampled_ids`` del vocabulario."""
    M, K = a.shape
    S = sampled_ids.shape[0]
    res = _zero(a.device) if residual is None else residual
    rm, rs = (0, 0) if residual is None else (res.stride(0), res.stride(1))
    bm, bs_, bk, gm, warps, stages = _cfg(M, S)
    out = torch.empty((M, S), dtype=out_dtype, device=a.device)
    grid = (triton.cdiv(M, bm) * triton.cdiv(S, bs_),)
    _lanzar(grid, (bm, bs_, bk, gm, warps, stages), False,
            a,
            weight,
            out,
            sampled_ids,
            res,
            a_scales,
            b_scales,
            shifts,
            M,
            S,
            K,
            a.stride(0),
            a.stride(1),
            weight.stride(0),
            weight.stride(1),
            out.stride(0),
            out.stride(1),
            rm,
            rs,
            shifts.stride(0),
            shifts.stride(1),
            bm,
            bs_,
            bk,
            gm,
            SHIFT_BLOCK)
    return out


def lm_head_forward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    shifts: torch.Tensor,
    sampled_ids: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Capa completa lm_head: quant -> GEMM diádico con gather -> logits.

    ``sampled_ids`` None recorre el vocabulario completo usando un ``arange``
    cacheado, por el mismo kernel y sin ninguna rama en el camino caliente.
    """
    a, a_scales = sk07_quant(hidden)
    idx = _arange(hidden.device, weight.shape[1]) if sampled_ids is None else sampled_ids
    return lm_head_gemm(a, weight, a_scales, weight_scale, shifts, idx, None, out_dtype)


lm_head_fused_sampled = lm_head_forward
sk07_lm_head = lm_head_forward

__all__ = [
    "SK_ID", "SK_NAME", "VOCAB_GLOBAL", "VOCAB_PER_RANK", "LM_HEAD_K",
    "SHIFT_BLOCK", "BLOCK_M", "BLOCK_N", "BLOCK_S", "BLOCK_K",
    "sk07_quant", "lm_head_gemm", "lm_head_forward", "lm_head_fused_sampled", "sk07_lm_head",
]


# --- variantes PTX embebidas ---

_PTX_0 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk07_lm_head_kernel    // -- Begin function _sk07_lm_head_kernel
.extern .shared .align 16 .b8 global_smem[];
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
                                        // @_sk07_lm_head_kernel
.visible .entry _sk07_lm_head_kernel(
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_6,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_7,
	.param .u32 _sk07_lm_head_kernel_param_8,
	.param .u32 _sk07_lm_head_kernel_param_9,
	.param .u32 _sk07_lm_head_kernel_param_10,
	.param .u32 _sk07_lm_head_kernel_param_11,
	.param .u32 _sk07_lm_head_kernel_param_12,
	.param .u32 _sk07_lm_head_kernel_param_13,
	.param .u32 _sk07_lm_head_kernel_param_14,
	.param .u32 _sk07_lm_head_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_16,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_17
)
.reqntid 256
{
	.reg .pred 	%p<11>;
	.reg .b16 	%rs<33>;
	.reg .b32 	%r<327>;
	.reg .b64 	%rd<90>;
	.loc	1 123 0                         // sk07_lm_head.py:123:0
$L__func_begin0:
	.loc	1 123 0                         // sk07_lm_head.py:123:0

// %bb.0:
	ld.param.b32 	%r22, [_sk07_lm_head_kernel_param_14];
	ld.param.b32 	%r21, [_sk07_lm_head_kernel_param_13];
	ld.param.b32 	%r20, [_sk07_lm_head_kernel_param_10];
	ld.param.b32 	%r19, [_sk07_lm_head_kernel_param_9];
	ld.param.b32 	%r18, [_sk07_lm_head_kernel_param_8];
	ld.param.b64 	%rd24, [_sk07_lm_head_kernel_param_6];
	ld.param.b64 	%rd23, [_sk07_lm_head_kernel_param_5];
	ld.param.b64 	%rd22, [_sk07_lm_head_kernel_param_4];
	ld.param.b64 	%rd21, [_sk07_lm_head_kernel_param_2];
	ld.param.b64 	%rd20, [_sk07_lm_head_kernel_param_1];
	ld.param.b64 	%rd19, [_sk07_lm_head_kernel_param_0];
$L__tmp0:
	.loc	1 132 24                        // sk07_lm_head.py:132:24
	mov.u32 	%r54, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk07_lm_head.py:133:27 ]
	add.s32 	%r55, %r18, 15;
	.loc	2 43 30                         // standard.py:43:30 @[ sk07_lm_head.py:133:27 ]
	shr.s32 	%r56, %r55, 31;
	shr.u32 	%r57, %r56, 28;
	add.s32 	%r58, %r55, %r57;
	shr.s32 	%r59, %r58, 4;
	ld.param.b64 	%rd43, [_sk07_lm_head_kernel_param_3];
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk07_lm_head.py:134:27 ]
	add.s32 	%r60, %r19, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk07_lm_head.py:134:27 ]
	shr.s32 	%r61, %r60, 31;
	shr.u32 	%r62, %r61, 25;
	add.s32 	%r63, %r60, %r62;
	shr.s32 	%r64, %r63, 7;
$L__tmp3:
	.loc	1 135 29                        // sk07_lm_head.py:135:29
	shl.b32 	%r65, %r64, 3;
	.loc	1 136 22                        // sk07_lm_head.py:136:22
	div.s32 	%r66, %r54, %r65;
	.loc	1 136 38                        // sk07_lm_head.py:136:38
	shl.b32 	%r67, %r66, 3;
	.loc	1 137 30                        // sk07_lm_head.py:137:30
	sub.s32 	%r68, %r59, %r67;
	.loc	1 137 39                        // sk07_lm_head.py:137:39
	min.s32 	%r69, %r68, 8;
	ld.param.b32 	%r70, [_sk07_lm_head_kernel_param_11];
	.loc	1 138 30                        // sk07_lm_head.py:138:30
	mul.lo.s32 	%r71, %r66, %r65;
	ld.param.b32 	%r72, [_sk07_lm_head_kernel_param_12];
	sub.s32 	%r73, %r54, %r71;
	.loc	1 139 36                        // sk07_lm_head.py:139:36
	div.s32 	%r74, %r73, %r69;
	.loc	1 138 46                        // sk07_lm_head.py:138:46
	mul.lo.s32 	%r75, %r74, %r69;
	sub.s32 	%r76, %r73, %r75;
	.loc	1 138 23                        // sk07_lm_head.py:138:23
	add.s32 	%r77, %r76, %r67;
	.loc	1 141 22                        // sk07_lm_head.py:141:22
	shl.b32 	%r1, %r77, 4;
	.loc	1 141 45                        // sk07_lm_head.py:141:45
	mov.u32 	%r2, %tid.x;
	and.b32 	%r3, %r2, 240;
	bfe.u32 	%r78, %r2, 4, 4;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r4, %r1, %r78;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r5, %r4, %r18;
	.loc	1 142 22                        // sk07_lm_head.py:142:22
	shl.b32 	%r79, %r74, 7;
	.loc	1 142 45                        // sk07_lm_head.py:142:45
	shr.u32 	%r80, %r2, 3;
	bfe.u32 	%r81, %r2, 3, 5;
	shl.b32 	%r6, %r2, 1;
	and.b32 	%r82, %r6, 6;
	and.b32 	%r7, %r2, 224;
	shr.u32 	%r83, %r7, 2;
	or.b32 	%r84, %r82, %r83;
	and.b32 	%r8, %r2, 15;
	shl.b32 	%r85, %r8, 3;
	.loc	1 142 32                        // sk07_lm_head.py:142:32
	or.b32 	%r86, %r79, %r81;
	or.b32 	%r87, %r86, 32;
	or.b32 	%r88, %r86, 64;
	or.b32 	%r89, %r80, %r79;
	or.b32 	%r90, %r89, 96;
	or.b32 	%r91, %r79, %r84;
	or.b32 	%r92, %r91, 64;
	or.b32 	%r9, %r79, %r85;
	or.b32 	%r93, %r9, 4;
	.loc	1 142 57                        // sk07_lm_head.py:142:57
	rem.s32 	%r94, %r86, %r19;
	rem.s32 	%r95, %r87, %r19;
	rem.s32 	%r96, %r88, %r19;
	rem.s32 	%r97, %r90, %r19;
	rem.s32 	%r98, %r91, %r19;
	rem.s32 	%r99, %r92, %r19;
	rem.s32 	%r100, %r9, %r19;
	rem.s32 	%r101, %r93, %r19;
	.loc	1 145 29                        // sk07_lm_head.py:145:29
	mad.wide.s32 	%rd25, %r94, 4, %rd43;
	mad.wide.s32 	%rd26, %r95, 4, %rd43;
	mad.wide.s32 	%rd27, %r96, 4, %rd43;
	mad.wide.s32 	%rd28, %r97, 4, %rd43;
	mad.wide.s32 	%rd29, %r98, 4, %rd43;
	mad.wide.s32 	%rd30, %r99, 4, %rd43;
	mad.wide.s32 	%rd31, %r100, 4, %rd43;
	mad.wide.s32 	%rd32, %r101, 4, %rd43;
	.loc	1 145 19                        // sk07_lm_head.py:145:19
	// begin inline asm
	mov.u32 %r24, 0x0;
	ld.global.b32 { %r24 }, [ %rd25 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r25, 0x0;
	ld.global.b32 { %r25 }, [ %rd26 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r26, 0x0;
	ld.global.b32 { %r26 }, [ %rd27 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r27, 0x0;
	ld.global.b32 { %r27 }, [ %rd28 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r28, 0x0;
	mov.u32 %r29, 0x0;
	ld.global.v2.b32 { %r28, %r29 }, [ %rd29 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r30, 0x0;
	mov.u32 %r31, 0x0;
	ld.global.v2.b32 { %r30, %r31 }, [ %rd30 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r32, 0x0;
	mov.u32 %r33, 0x0;
	mov.u32 %r34, 0x0;
	mov.u32 %r35, 0x0;
	ld.global.v4.b32 { %r32, %r33, %r34, %r35 }, [ %rd31 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r36, 0x0;
	mov.u32 %r37, 0x0;
	mov.u32 %r38, 0x0;
	mov.u32 %r39, 0x0;
	ld.global.v4.b32 { %r36, %r37, %r38, %r39 }, [ %rd32 + 0 ];
	// end inline asm
	.loc	1 146 39                        // sk07_lm_head.py:146:39
	mul.lo.s32 	%r102, %r5, %r70;
	.loc	1 146 21                        // sk07_lm_head.py:146:21
	cvt.s64.s32 	%rd1, %r102;
	add.s64 	%rd45, %rd19, %rd1;
	.loc	1 146 51                        // sk07_lm_head.py:146:51
	cvt.u64.u32 	%rd2, %r85;
	add.s64 	%rd33, %rd45, %rd2;
	.loc	1 147 28                        // sk07_lm_head.py:147:28
	and.b32 	%r10, %r2, 7;
	shl.b32 	%r103, %r10, 4;
	.loc	1 147 21                        // sk07_lm_head.py:147:21
	cvt.u64.u32 	%rd3, %r103;
	add.s64 	%rd46, %rd20, %rd3;
	.loc	1 147 67                        // sk07_lm_head.py:147:67
	mul.lo.s32 	%r104, %r24, %r72;
	mul.lo.s32 	%r105, %r25, %r72;
	mul.lo.s32 	%r106, %r26, %r72;
	mul.lo.s32 	%r107, %r27, %r72;
	.loc	1 147 51                        // sk07_lm_head.py:147:51
	cvt.s64.s32 	%rd4, %r104;
	add.s64 	%rd34, %rd46, %rd4;
	cvt.s64.s32 	%rd5, %r105;
	add.s64 	%rd35, %rd46, %rd5;
	cvt.s64.s32 	%rd6, %r106;
	add.s64 	%rd36, %rd46, %rd6;
	cvt.s64.s32 	%rd7, %r107;
	add.s64 	%rd37, %rd46, %rd7;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	setp.lt.s32 	%p1, %r20, 128;
	setp.gt.s32 	%p2, %r20, 127;
	.loc	1 160 30                        // sk07_lm_head.py:160:30
	and.b32 	%r124, %r2, 255;
	shl.b32 	%r125, %r124, 3;
	and.b32 	%r126, %r2, 112;
	xor.b32 	%r127, %r125, %r126;
	mov.b32 	%r128, global_smem;
	add.s32 	%r11, %r128, %r127;
	add.s32 	%r40, %r11, 32768;
	selp.b32 	%r41, 8, 0, %p2;
	// begin inline asm
	cp.async.ca.shared.global [ %r40 + 0 ], [ %rd33 + 0 ], 0x8, %r41;
	// end inline asm
	cp.async.commit_group;
	.loc	1 160 47                        // sk07_lm_head.py:160:47
	shl.b32 	%r129, %r124, 4;
	and.b32 	%r130, %r6, 112;
	xor.b32 	%r131, %r129, %r130;
	add.s32 	%r42, %r128, %r131;
	selp.b32 	%r43, 16, 0, %p2;
	// begin inline asm
	cp.async.cg.shared.global [ %r42 + 0 ], [ %rd34 + 0 ], 0x10, %r43;
	// end inline asm
	add.s32 	%r44, %r42, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r44 + 0 ], [ %rd35 + 0 ], 0x10, %r43;
	// end inline asm
	add.s32 	%r45, %r42, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r45 + 0 ], [ %rd36 + 0 ], 0x10, %r43;
	// end inline asm
	add.s32 	%r46, %r42, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r46 + 0 ], [ %rd37 + 0 ], 0x10, %r43;
	// end inline asm
	cp.async.commit_group;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	setp.gt.s32 	%p3, %r20, 255;
	.loc	1 161 18                        // sk07_lm_head.py:161:18
	add.s64 	%rd38, %rd33, 128;
	.loc	1 162 18                        // sk07_lm_head.py:162:18
	add.s64 	%rd39, %rd34, 128;
	add.s64 	%rd40, %rd35, 128;
	add.s64 	%rd41, %rd36, 128;
	add.s64 	%rd42, %rd37, 128;
	.loc	1 160 30                        // sk07_lm_head.py:160:30
	bar.sync 	0;
	add.s32 	%r47, %r11, 34816;
	selp.b32 	%r48, 8, 0, %p3;
	// begin inline asm
	cp.async.ca.shared.global [ %r47 + 0 ], [ %rd38 + 0 ], 0x8, %r48;
	// end inline asm
	cp.async.commit_group;
	.loc	1 160 47                        // sk07_lm_head.py:160:47
	add.s32 	%r49, %r42, 16384;
	selp.b32 	%r50, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r49 + 0 ], [ %rd39 + 0 ], 0x10, %r50;
	// end inline asm
	add.s32 	%r51, %r42, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r51 + 0 ], [ %rd40 + 0 ], 0x10, %r50;
	// end inline asm
	add.s32 	%r52, %r42, 24576;
	// begin inline asm
	cp.async.cg.shared.global [ %r52 + 0 ], [ %rd41 + 0 ], 0x10, %r50;
	// end inline asm
	add.s32 	%r53, %r42, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r53 + 0 ], [ %rd42 + 0 ], 0x10, %r50;
	// end inline asm
	cp.async.commit_group;
	mov.b32 	%r319, 0f00000000;
	cvt.u32.u64 	%r315, %rd3;
	mov.b32 	%r320, %r319;
	mov.b32 	%r321, %r319;
	mov.b32 	%r322, %r319;
	mov.b32 	%r323, %r319;
	mov.b32 	%r324, %r319;
	mov.b32 	%r325, %r319;
	mov.b32 	%r326, %r319;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	@%p1 bra 	$L__BB0_3;
// %bb.1:                               // %.lr.ph
	.loc	1 0 23                          // sk07_lm_head.py:0:23
	ld.param.b32 	%r23, [_sk07_lm_head_kernel_param_15];
	ld.param.b64 	%rd44, [_sk07_lm_head_kernel_param_7];
	shr.s32 	%r108, %r28, 31;
	shr.u32 	%r109, %r108, 25;
	add.s32 	%r110, %r28, %r109;
	shr.s32 	%r111, %r110, 7;
	shr.s32 	%r112, %r29, 31;
	shr.u32 	%r113, %r112, 25;
	add.s32 	%r114, %r29, %r113;
	shr.s32 	%r115, %r114, 7;
	shr.s32 	%r116, %r30, 31;
	shr.u32 	%r117, %r116, 25;
	add.s32 	%r118, %r30, %r117;
	shr.s32 	%r119, %r118, 7;
	shr.s32 	%r120, %r31, 31;
	shr.u32 	%r121, %r120, 25;
	add.s32 	%r122, %r31, %r121;
	shr.s32 	%r123, %r122, 7;
	cvt.s64.s32 	%rd47, %r111;
	add.s64 	%rd8, %rd44, %rd47;
	cvt.s64.s32 	%rd48, %r115;
	add.s64 	%rd9, %rd44, %rd48;
	cvt.s64.s32 	%rd49, %r119;
	add.s64 	%rd10, %rd44, %rd49;
	cvt.s64.s32 	%rd50, %r123;
	add.s64 	%rd11, %rd44, %rd50;
	.loc	1 151 28                        // sk07_lm_head.py:151:28
	shr.u32 	%r132, %r20, 7;
	add.s32 	%r133, %r132, -2;
	shl.b32 	%r134, %r8, 7;
	and.b32 	%r135, %r2, 16;
	xor.b32 	%r136, %r315, %r135;
	or.b32 	%r12, %r136, %r134;
	xor.b32 	%r13, %r12, 32;
	xor.b32 	%r14, %r12, 64;
	xor.b32 	%r15, %r12, 96;
	shl.b32 	%r137, %r10, 7;
	shl.b32 	%r138, %r7, 5;
	and.b32 	%r139, %r6, 48;
	or.b32 	%r140, %r137, %r138;
	xor.b32 	%r141, %r315, %r139;
	or.b32 	%r16, %r140, %r141;
	xor.b32 	%r17, %r16, 64;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	cvt.s64.s32 	%rd12, %r133;
	and.b32 	%r142, %r20, -128;
	cvt.u64.u32 	%rd13, %r142;
	add.s64 	%rd51, %rd3, %rd7;
	add.s64 	%rd52, %rd51, %rd20;
	add.s64 	%rd14, %rd52, 256;
	add.s64 	%rd53, %rd3, %rd6;
	add.s64 	%rd54, %rd53, %rd20;
	add.s64 	%rd15, %rd54, 256;
	add.s64 	%rd55, %rd3, %rd5;
	add.s64 	%rd56, %rd55, %rd20;
	add.s64 	%rd16, %rd56, 256;
	add.s64 	%rd57, %rd3, %rd4;
	add.s64 	%rd58, %rd57, %rd20;
	add.s64 	%rd17, %rd58, 256;
	add.s64 	%rd59, %rd2, %rd1;
	add.s64 	%rd60, %rd59, %rd19;
	add.s64 	%rd18, %rd60, 256;
	mov.b32 	%r319, 0f00000000;
	mov.b32 	%r318, 1;
	mov.b32 	%r317, -1;
	mov.b64 	%rd88, 0;
	mov.b32 	%r143, 0;
	mov.b32 	%r316, %r143;
	mov.b64 	%rd89, %rd88;
	mov.b32 	%r320, %r319;
	mov.b32 	%r321, %r319;
	mov.b32 	%r322, %r319;
	mov.b32 	%r323, %r319;
	mov.b32 	%r324, %r319;
	mov.b32 	%r325, %r319;
	mov.b32 	%r326, %r319;
$L__BB0_2:                              // %__nv_exp2f.exit
                                        // =>This Inner Loop Header: Depth=1
	setp.lt.s64 	%p4, %rd89, %rd12;
	add.s32 	%r191, %r317, 1;
	setp.gt.s32 	%p5, %r191, 1;
	selp.b32 	%r317, 0, %r191, %p5;
	.loc	1 158 39                        // sk07_lm_head.py:158:39
	cvt.s64.s32 	%rd70, %r316;
	add.s64 	%rd61, %rd8, %rd70;
	add.s64 	%rd62, %rd9, %rd70;
	add.s64 	%rd63, %rd10, %rd70;
	add.s64 	%rd64, %rd11, %rd70;
	.loc	1 158 29                        // sk07_lm_head.py:158:29
	// begin inline asm
	mov.u16 %rs1, 0x0;
	ld.global.b8 { %rs1 }, [ %rd61 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs5, %rs1;
	// begin inline asm
	mov.u16 %rs2, 0x0;
	ld.global.b8 { %rs2 }, [ %rd62 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs6, %rs2;
	// begin inline asm
	mov.u16 %rs3, 0x0;
	ld.global.b8 { %rs3 }, [ %rd63 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs7, %rs3;
	// begin inline asm
	mov.u16 %rs4, 0x0;
	ld.global.b8 { %rs4 }, [ %rd64 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs8, %rs4;
	.loc	1 159 50                        // sk07_lm_head.py:159:50
	cvt.rn.f32.s16 	%r192, %rs5;
	cvt.rn.f32.s16 	%r193, %rs6;
	cvt.rn.f32.s16 	%r194, %rs7;
	cvt.rn.f32.s16 	%r195, %rs8;
	.loc	1 158 21                        // sk07_lm_head.py:158:21
	ex2.approx.ftz.f32 	%r196, %r192;
	ex2.approx.ftz.f32 	%r197, %r193;
	ex2.approx.ftz.f32 	%r198, %r194;
	ex2.approx.ftz.f32 	%r199, %r195;
	.loc	1 160 30                        // sk07_lm_head.py:160:30
	cp.async.wait_group 	2;
	bar.sync 	0;
	shl.b32 	%r200, %r317, 11;
	add.s32 	%r201, %r128, %r200;
	add.s32 	%r202, %r201, %r12;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r144, %r145, %r146, %r147}, [%r202+32768];
	add.s32 	%r203, %r201, %r13;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r156, %r157, %r158, %r159}, [%r203+32768];
	add.s32 	%r204, %r201, %r14;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r168, %r169, %r170, %r171}, [%r204+32768];
	add.s32 	%r205, %r201, %r15;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r176, %r177, %r178, %r179}, [%r205+32768];
	.loc	1 160 47                        // sk07_lm_head.py:160:47
	shl.b32 	%r206, %r317, 14;
	add.s32 	%r207, %r128, %r206;
	add.s32 	%r208, %r207, %r16;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r148, %r149, %r160, %r161}, [%r208];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r150, %r151, %r166, %r167}, [%r208+8192];
	add.s32 	%r209, %r207, %r17;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r172, %r173, %r180, %r181}, [%r209];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r174, %r175, %r182, %r183}, [%r209+8192];
	.loc	1 160 39                        // sk07_lm_head.py:160:39
	mov.b32 	%r152, %r143;
	mov.b32 	%r153, %r143;
	mov.b32 	%r154, %r143;
	mov.b32 	%r155, %r143;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r152, %r153, %r154, %r155 }, { %r144, %r145, %r146, %r147 }, { %r148, %r149 }, { %r152, %r153, %r154, %r155 };
	// end inline asm
	mov.b32 	%r162, %r143;
	mov.b32 	%r163, %r143;
	mov.b32 	%r164, %r143;
	mov.b32 	%r165, %r143;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r162, %r163, %r164, %r165 }, { %r144, %r145, %r146, %r147 }, { %r150, %r151 }, { %r162, %r163, %r164, %r165 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r152, %r153, %r154, %r155 }, { %r156, %r157, %r158, %r159 }, { %r160, %r161 }, { %r152, %r153, %r154, %r155 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r162, %r163, %r164, %r165 }, { %r156, %r157, %r158, %r159 }, { %r166, %r167 }, { %r162, %r163, %r164, %r165 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r152, %r153, %r154, %r155 }, { %r168, %r169, %r170, %r171 }, { %r172, %r173 }, { %r152, %r153, %r154, %r155 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r162, %r163, %r164, %r165 }, { %r168, %r169, %r170, %r171 }, { %r174, %r175 }, { %r162, %r163, %r164, %r165 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r152, %r153, %r154, %r155 }, { %r176, %r177, %r178, %r179 }, { %r180, %r181 }, { %r152, %r153, %r154, %r155 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r162, %r163, %r164, %r165 }, { %r176, %r177, %r178, %r179 }, { %r182, %r183 }, { %r162, %r163, %r164, %r165 };
	// end inline asm
	.loc	1 160 79                        // sk07_lm_head.py:160:79
	cvt.rn.f32.s32 	%r210, %r162;
	cvt.rn.f32.s32 	%r211, %r163;
	cvt.rn.f32.s32 	%r212, %r164;
	cvt.rn.f32.s32 	%r213, %r165;
	cvt.rn.f32.s32 	%r214, %r152;
	cvt.rn.f32.s32 	%r215, %r153;
	cvt.rn.f32.s32 	%r216, %r154;
	cvt.rn.f32.s32 	%r217, %r155;
	.loc	1 160 15                        // sk07_lm_head.py:160:15
	fma.rn.f32 	%r322, %r197, %r217, %r322;
	fma.rn.f32 	%r321, %r196, %r216, %r321;
	fma.rn.f32 	%r320, %r197, %r215, %r320;
	fma.rn.f32 	%r319, %r196, %r214, %r319;
	fma.rn.f32 	%r326, %r199, %r213, %r326;
	fma.rn.f32 	%r325, %r198, %r212, %r325;
	fma.rn.f32 	%r324, %r199, %r211, %r324;
	fma.rn.f32 	%r323, %r198, %r210, %r323;
	.loc	1 162 18                        // sk07_lm_head.py:162:18
	add.s64 	%rd65, %rd18, %rd88;
	add.s64 	%rd66, %rd17, %rd88;
	add.s64 	%rd67, %rd16, %rd88;
	add.s64 	%rd68, %rd15, %rd88;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	add.s64 	%rd69, %rd14, %rd88;
	add.s32 	%r218, %r318, 1;
	setp.gt.s32 	%p6, %r218, 1;
	selp.b32 	%r318, 0, %r218, %p6;
	.loc	1 160 30                        // sk07_lm_head.py:160:30
	shl.b32 	%r219, %r318, 11;
	bar.sync 	0;
	add.s32 	%r220, %r11, %r219;
	add.s32 	%r184, %r220, 32768;
	selp.b32 	%r185, 8, 0, %p4;
	// begin inline asm
	cp.async.ca.shared.global [ %r184 + 0 ], [ %rd65 + 0 ], 0x8, %r185;
	// end inline asm
	cp.async.commit_group;
	.loc	1 160 47                        // sk07_lm_head.py:160:47
	shl.b32 	%r221, %r318, 14;
	add.s32 	%r186, %r42, %r221;
	selp.b32 	%r187, 16, 0, %p4;
	// begin inline asm
	cp.async.cg.shared.global [ %r186 + 0 ], [ %rd66 + 0 ], 0x10, %r187;
	// end inline asm
	add.s32 	%r188, %r186, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r188 + 0 ], [ %rd67 + 0 ], 0x10, %r187;
	// end inline asm
	add.s32 	%r189, %r186, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r189 + 0 ], [ %rd68 + 0 ], 0x10, %r187;
	// end inline asm
	add.s32 	%r190, %r186, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r190 + 0 ], [ %rd69 + 0 ], 0x10, %r187;
	// end inline asm
	cp.async.commit_group;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	add.s64 	%rd89, %rd89, 1;
	add.s64 	%rd88, %rd88, 128;
	add.s32 	%r316, %r316, %r23;
	setp.ne.b64 	%p7, %rd13, %rd88;
	@%p7 bra 	$L__BB0_2;
$L__BB0_3:                              // %._crit_edge
	.loc	1 141 45                        // sk07_lm_head.py:141:45
	and.b32 	%r241, %r2, 28;
	bfe.u32 	%r242, %r2, 2, 3;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r243, %r242, %r1;
	or.b32 	%r244, %r243, 8;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r245, %r244, %r18;
	rem.s32 	%r246, %r243, %r18;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 164 38                        // sk07_lm_head.py:164:38
	mad.wide.s32 	%rd71, %r246, 4, %rd23;
	mad.wide.s32 	%rd72, %r245, 4, %rd23;
	.loc	1 164 24                        // sk07_lm_head.py:164:24
	// begin inline asm
	mov.u32 %r222, 0x0;
	ld.global.b32 { %r222 }, [ %rd71 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r223, 0x0;
	ld.global.b32 { %r223 }, [ %rd72 + 0 ];
	// end inline asm
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r247, %r319, %r222;
	mul.f32 	%r248, %r320, %r222;
	mul.f32 	%r249, %r321, %r223;
	mul.f32 	%r250, %r322, %r223;
	mul.f32 	%r251, %r323, %r222;
	mul.f32 	%r252, %r324, %r222;
	mul.f32 	%r253, %r325, %r223;
	mul.f32 	%r254, %r326, %r223;
	.loc	1 165 38                        // sk07_lm_head.py:165:38
	mad.wide.s32 	%rd73, %r28, 4, %rd24;
	mad.wide.s32 	%rd74, %r29, 4, %rd24;
	mad.wide.s32 	%rd75, %r30, 4, %rd24;
	mad.wide.s32 	%rd76, %r31, 4, %rd24;
	.loc	1 165 24                        // sk07_lm_head.py:165:24
	// begin inline asm
	mov.u32 %r224, 0x0;
	ld.global.b32 { %r224 }, [ %rd73 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r225, 0x0;
	ld.global.b32 { %r225 }, [ %rd74 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r226, 0x0;
	ld.global.b32 { %r226 }, [ %rd75 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r227, 0x0;
	ld.global.b32 { %r227 }, [ %rd76 + 0 ];
	// end inline asm
	.loc	1 166 49                        // sk07_lm_head.py:166:49
	mul.lo.s32 	%r255, %r5, %r22;
	.loc	1 166 31                        // sk07_lm_head.py:166:31
	mad.wide.s32 	%rd86, %r255, 2, %rd22;
	.loc	1 166 64                        // sk07_lm_head.py:166:64
	mad.wide.s32 	%rd77, %r32, 2, %rd86;
	mad.wide.s32 	%rd78, %r33, 2, %rd86;
	mad.wide.s32 	%rd79, %r34, 2, %rd86;
	mad.wide.s32 	%rd80, %r35, 2, %rd86;
	mad.wide.s32 	%rd81, %r36, 2, %rd86;
	mad.wide.s32 	%rd82, %r37, 2, %rd86;
	mad.wide.s32 	%rd83, %r38, 2, %rd86;
	mad.wide.s32 	%rd84, %r39, 2, %rd86;
	.loc	1 166 19                        // sk07_lm_head.py:166:19
	// begin inline asm
	mov.u16 %rs9, 0x0;
	ld.global.b16 { %rs9 }, [ %rd77 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs10, 0x0;
	ld.global.b16 { %rs10 }, [ %rd78 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs11, 0x0;
	ld.global.b16 { %rs11 }, [ %rd79 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs12, 0x0;
	ld.global.b16 { %rs12 }, [ %rd80 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs13, 0x0;
	ld.global.b16 { %rs13 }, [ %rd81 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs14, 0x0;
	ld.global.b16 { %rs14 }, [ %rd82 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs15, 0x0;
	ld.global.b16 { %rs15 }, [ %rd83 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs16, 0x0;
	ld.global.b16 { %rs16 }, [ %rd84 + 0 ];
	// end inline asm
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	and.b32 	%r256, %r2, 120;
	shl.b32 	%r257, %r256, 5;
	or.b32 	%r258, %r257, %r315;
	xor.b32 	%r259, %r258, %r3;
	add.s32 	%r228, %r128, %r259;
	mov.b32 	%r229, {%rs9, %rs10};
	mov.b32 	%r230, {%rs11, %rs12};
	mov.b32 	%r231, {%rs13, %rs14};
	mov.b32 	%r232, {%rs15, %rs16};
	// begin inline asm
	st.shared.v4.b32 [ %r228 + 0 ], { %r229, %r230, %r231, %r232 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r260, %r10, 9;
	shl.b32 	%r261, %r2, 4;
	and.b32 	%r262, %r261, 496;
	shr.u32 	%r263, %r7, 1;
	xor.b32 	%r264, %r262, %r263;
	add.s32 	%r265, %r128, %r260;
	add.s32 	%r266, %r265, %r264;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r267, %r268, %r269, %r270}, [%r266];
	mov.b32 	{%rs25, %rs26}, %r267;
	mov.b32 	{%rs27, %rs28}, %r268;
	mov.b32 	{%rs29, %rs30}, %r269;
	mov.b32 	{%rs31, %rs32}, %r270;
	cvt.f32.bf16 	%r271, %rs25;
	cvt.f32.bf16 	%r272, %rs26;
	cvt.f32.bf16 	%r273, %rs27;
	cvt.f32.bf16 	%r274, %rs28;
	cvt.f32.bf16 	%r275, %rs29;
	cvt.f32.bf16 	%r276, %rs30;
	cvt.f32.bf16 	%r277, %rs31;
	cvt.f32.bf16 	%r278, %rs32;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r279, %r247, %r224, %r271;
	fma.rn.f32 	%r280, %r248, %r225, %r272;
	fma.rn.f32 	%r281, %r249, %r224, %r273;
	fma.rn.f32 	%r282, %r250, %r225, %r274;
	fma.rn.f32 	%r283, %r251, %r226, %r275;
	fma.rn.f32 	%r284, %r252, %r227, %r276;
	fma.rn.f32 	%r285, %r253, %r226, %r277;
	fma.rn.f32 	%r286, %r254, %r227, %r278;
	.loc	1 173 31                        // sk07_lm_head.py:173:31
	setp.lt.s32 	%p9, %r4, %r18;
	.loc	1 173 54                        // sk07_lm_head.py:173:54
	setp.lt.s32 	%p10, %r9, %r19;
	.loc	1 173 37                        // sk07_lm_head.py:173:37
	and.pred 	%p8, %p9, %p10;
	.loc	1 171 35                        // sk07_lm_head.py:171:35
	mul.lo.s32 	%r287, %r4, %r21;
	.loc	1 171 18                        // sk07_lm_head.py:171:18
	mad.wide.s32 	%rd87, %r287, 2, %rd21;
	.loc	1 171 50                        // sk07_lm_head.py:171:50
	mad.wide.s32 	%rd85, %r9, 2, %rd87;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16.f32 	%rs17, %r279;
	cvt.rn.bf16.f32 	%rs18, %r280;
	cvt.rn.bf16.f32 	%rs19, %r281;
	cvt.rn.bf16.f32 	%rs20, %r282;
	cvt.rn.bf16.f32 	%rs21, %r283;
	cvt.rn.bf16.f32 	%rs22, %r284;
	cvt.rn.bf16.f32 	%rs23, %r285;
	cvt.rn.bf16.f32 	%rs24, %r286;
	bar.sync 	0;
	shl.b32 	%r288, %r2, 5;
	and.b32 	%r289, %r288, 768;
	shl.b32 	%r290, %r241, 1;
	and.b32 	%r291, %r2, 1;
	neg.s32 	%r292, %r291;
	and.b32 	%r293, %r292, 1088;
	bfe.s32 	%r294, %r2, 1, 1;
	and.b32 	%r295, %r294, 2052;
	or.b32 	%r296, %r289, %r290;
	or.b32 	%r297, %r293, %r296;
	xor.b32 	%r298, %r297, %r263;
	or.b32 	%r299, %r298, %r295;
	add.s32 	%r233, %r128, %r299;
	// begin inline asm
	st.shared.v2.b16 [ %r233 + 0 ], { %rs17, %rs18 };
	// end inline asm
	add.s32 	%r234, %r233, 128;
	// begin inline asm
	st.shared.v2.b16 [ %r234 + 0 ], { %rs19, %rs20 };
	// end inline asm
	xor.b32 	%r300, %r299, 4;
	add.s32 	%r235, %r128, %r300;
	// begin inline asm
	st.shared.v2.b16 [ %r235 + 0 ], { %rs21, %rs22 };
	// end inline asm
	add.s32 	%r236, %r235, 128;
	// begin inline asm
	st.shared.v2.b16 [ %r236 + 0 ], { %rs23, %rs24 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r301, %r2, 3;
	and.b32 	%r302, %r301, 768;
	shr.u32 	%r303, %r256, 1;
	and.b32 	%r304, %r2, 128;
	or.b32 	%r305, %r315, %r302;
	xor.b32 	%r306, %r305, %r303;
	or.b32 	%r307, %r306, %r304;
	add.s32 	%r308, %r128, %r307;
	ld.shared.b32 	%r237, [%r308];
	xor.b32 	%r309, %r307, 64;
	add.s32 	%r310, %r128, %r309;
	ld.shared.b32 	%r238, [%r310+1024];
	xor.b32 	%r311, %r307, 4;
	add.s32 	%r312, %r128, %r311;
	ld.shared.b32 	%r239, [%r312+2048];
	xor.b32 	%r313, %r307, 68;
	add.s32 	%r314, %r128, %r313;
	ld.shared.b32 	%r240, [%r314+3072];
	.loc	1 172 8                         // sk07_lm_head.py:172:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd85 + 0 ], { %r237, %r238, %r239, %r240 };
	// end inline asm
	.loc	1 170 4                         // sk07_lm_head.py:170:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk07_lm_head.py"
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
.b32 159                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x98 DW_TAG_compile_unit
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
.b8 55
.b8 95
.b8 108
.b8 109
.b8 95
.b8 104
.b8 101
.b8 97
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
.b8 2                                   // Abbrev [2] 0x45:0x17 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 55
.b8 95
.b8 108
.b8 109
.b8 95
.b8 104
.b8 101
.b8 97
.b8 100
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x5c:0x46 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 69                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x71:0x18 DW_TAG_inlined_subroutine
.b32 69                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 133                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x89:0x18 DW_TAG_inlined_subroutine
.b32 69                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 134                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_0 = _Nativo(
    "sk07_lm_head/tile16x128x128_shift0_abi16",
    _PTX_0, "_sk07_lm_head_kernel",
    warps=8, shared=36864,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 17, 19],
    horneado={12: 1, 13: 1, 16: 1, 18: 1, 20: 1, 21: 16, 22: 128, 23: 128, 24: 8, 25: 128},
    div16=[9, 10, 11, 14, 15, 17],
)

_PTX_1 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk07_lm_head_kernel    // -- Begin function _sk07_lm_head_kernel
.extern .shared .align 16 .b8 global_smem[];
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
                                        // @_sk07_lm_head_kernel
.visible .entry _sk07_lm_head_kernel(
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_6,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_7,
	.param .u32 _sk07_lm_head_kernel_param_8,
	.param .u32 _sk07_lm_head_kernel_param_9,
	.param .u32 _sk07_lm_head_kernel_param_10,
	.param .u32 _sk07_lm_head_kernel_param_11,
	.param .u32 _sk07_lm_head_kernel_param_12,
	.param .u32 _sk07_lm_head_kernel_param_13,
	.param .u32 _sk07_lm_head_kernel_param_14,
	.param .u32 _sk07_lm_head_kernel_param_15,
	.param .u32 _sk07_lm_head_kernel_param_16,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_17,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_18
)
.reqntid 256
{
	.reg .pred 	%p<11>;
	.reg .b16 	%rs<33>;
	.reg .b32 	%r<336>;
	.reg .b64 	%rd<90>;
	.loc	1 123 0                         // sk07_lm_head.py:123:0
$L__func_begin0:
	.loc	1 123 0                         // sk07_lm_head.py:123:0

// %bb.0:
	ld.param.b32 	%r23, [_sk07_lm_head_kernel_param_15];
	ld.param.b32 	%r22, [_sk07_lm_head_kernel_param_14];
	ld.param.b32 	%r21, [_sk07_lm_head_kernel_param_13];
	ld.param.b32 	%r20, [_sk07_lm_head_kernel_param_10];
	ld.param.b32 	%r19, [_sk07_lm_head_kernel_param_9];
	ld.param.b32 	%r18, [_sk07_lm_head_kernel_param_8];
	ld.param.b64 	%rd24, [_sk07_lm_head_kernel_param_6];
	ld.param.b64 	%rd23, [_sk07_lm_head_kernel_param_5];
	ld.param.b64 	%rd22, [_sk07_lm_head_kernel_param_4];
	ld.param.b64 	%rd21, [_sk07_lm_head_kernel_param_2];
	ld.param.b64 	%rd20, [_sk07_lm_head_kernel_param_1];
	ld.param.b64 	%rd19, [_sk07_lm_head_kernel_param_0];
$L__tmp0:
	.loc	1 132 24                        // sk07_lm_head.py:132:24
	mov.u32 	%r55, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk07_lm_head.py:133:27 ]
	add.s32 	%r56, %r18, 15;
	.loc	2 43 30                         // standard.py:43:30 @[ sk07_lm_head.py:133:27 ]
	shr.s32 	%r57, %r56, 31;
	shr.u32 	%r58, %r57, 28;
	add.s32 	%r59, %r56, %r58;
	shr.s32 	%r60, %r59, 4;
	ld.param.b64 	%rd43, [_sk07_lm_head_kernel_param_3];
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk07_lm_head.py:134:27 ]
	add.s32 	%r61, %r19, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk07_lm_head.py:134:27 ]
	shr.s32 	%r62, %r61, 31;
	shr.u32 	%r63, %r62, 25;
	add.s32 	%r64, %r61, %r63;
	shr.s32 	%r65, %r64, 7;
$L__tmp3:
	.loc	1 135 29                        // sk07_lm_head.py:135:29
	shl.b32 	%r66, %r65, 3;
	.loc	1 136 22                        // sk07_lm_head.py:136:22
	div.s32 	%r67, %r55, %r66;
	.loc	1 136 38                        // sk07_lm_head.py:136:38
	shl.b32 	%r68, %r67, 3;
	.loc	1 137 30                        // sk07_lm_head.py:137:30
	sub.s32 	%r69, %r60, %r68;
	.loc	1 137 39                        // sk07_lm_head.py:137:39
	min.s32 	%r70, %r69, 8;
	ld.param.b32 	%r71, [_sk07_lm_head_kernel_param_11];
	.loc	1 138 30                        // sk07_lm_head.py:138:30
	mul.lo.s32 	%r72, %r67, %r66;
	ld.param.b32 	%r73, [_sk07_lm_head_kernel_param_12];
	sub.s32 	%r74, %r55, %r72;
	.loc	1 139 36                        // sk07_lm_head.py:139:36
	div.s32 	%r75, %r74, %r70;
	.loc	1 138 46                        // sk07_lm_head.py:138:46
	mul.lo.s32 	%r76, %r75, %r70;
	sub.s32 	%r77, %r74, %r76;
	.loc	1 138 23                        // sk07_lm_head.py:138:23
	add.s32 	%r78, %r77, %r68;
	.loc	1 141 22                        // sk07_lm_head.py:141:22
	shl.b32 	%r1, %r78, 4;
	.loc	1 141 45                        // sk07_lm_head.py:141:45
	mov.u32 	%r2, %tid.x;
	and.b32 	%r3, %r2, 240;
	bfe.u32 	%r79, %r2, 4, 4;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r4, %r1, %r79;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r5, %r4, %r18;
	.loc	1 142 22                        // sk07_lm_head.py:142:22
	shl.b32 	%r80, %r75, 7;
	.loc	1 142 45                        // sk07_lm_head.py:142:45
	shr.u32 	%r81, %r2, 3;
	bfe.u32 	%r82, %r2, 3, 5;
	shl.b32 	%r6, %r2, 1;
	and.b32 	%r83, %r6, 6;
	and.b32 	%r7, %r2, 224;
	shr.u32 	%r84, %r7, 2;
	or.b32 	%r85, %r83, %r84;
	and.b32 	%r8, %r2, 15;
	shl.b32 	%r86, %r8, 3;
	.loc	1 142 32                        // sk07_lm_head.py:142:32
	or.b32 	%r87, %r80, %r82;
	or.b32 	%r88, %r87, 32;
	or.b32 	%r89, %r87, 64;
	or.b32 	%r90, %r81, %r80;
	or.b32 	%r91, %r90, 96;
	or.b32 	%r92, %r80, %r85;
	or.b32 	%r93, %r92, 64;
	or.b32 	%r9, %r80, %r86;
	or.b32 	%r94, %r9, 4;
	.loc	1 142 57                        // sk07_lm_head.py:142:57
	rem.s32 	%r95, %r87, %r19;
	rem.s32 	%r96, %r88, %r19;
	rem.s32 	%r97, %r89, %r19;
	rem.s32 	%r98, %r91, %r19;
	rem.s32 	%r99, %r92, %r19;
	rem.s32 	%r100, %r93, %r19;
	rem.s32 	%r101, %r9, %r19;
	rem.s32 	%r102, %r94, %r19;
	.loc	1 145 29                        // sk07_lm_head.py:145:29
	mad.wide.s32 	%rd25, %r95, 4, %rd43;
	mad.wide.s32 	%rd26, %r96, 4, %rd43;
	mad.wide.s32 	%rd27, %r97, 4, %rd43;
	mad.wide.s32 	%rd28, %r98, 4, %rd43;
	mad.wide.s32 	%rd29, %r99, 4, %rd43;
	mad.wide.s32 	%rd30, %r100, 4, %rd43;
	mad.wide.s32 	%rd31, %r101, 4, %rd43;
	mad.wide.s32 	%rd32, %r102, 4, %rd43;
	.loc	1 145 19                        // sk07_lm_head.py:145:19
	// begin inline asm
	mov.u32 %r25, 0x0;
	ld.global.b32 { %r25 }, [ %rd25 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r26, 0x0;
	ld.global.b32 { %r26 }, [ %rd26 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r27, 0x0;
	ld.global.b32 { %r27 }, [ %rd27 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r28, 0x0;
	ld.global.b32 { %r28 }, [ %rd28 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r29, 0x0;
	mov.u32 %r30, 0x0;
	ld.global.v2.b32 { %r29, %r30 }, [ %rd29 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r31, 0x0;
	mov.u32 %r32, 0x0;
	ld.global.v2.b32 { %r31, %r32 }, [ %rd30 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r33, 0x0;
	mov.u32 %r34, 0x0;
	mov.u32 %r35, 0x0;
	mov.u32 %r36, 0x0;
	ld.global.v4.b32 { %r33, %r34, %r35, %r36 }, [ %rd31 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r37, 0x0;
	mov.u32 %r38, 0x0;
	mov.u32 %r39, 0x0;
	mov.u32 %r40, 0x0;
	ld.global.v4.b32 { %r37, %r38, %r39, %r40 }, [ %rd32 + 0 ];
	// end inline asm
	.loc	1 146 39                        // sk07_lm_head.py:146:39
	mul.lo.s32 	%r103, %r5, %r71;
	.loc	1 146 21                        // sk07_lm_head.py:146:21
	cvt.s64.s32 	%rd1, %r103;
	add.s64 	%rd45, %rd19, %rd1;
	.loc	1 146 51                        // sk07_lm_head.py:146:51
	cvt.u64.u32 	%rd2, %r86;
	add.s64 	%rd33, %rd45, %rd2;
	.loc	1 147 28                        // sk07_lm_head.py:147:28
	and.b32 	%r10, %r2, 7;
	shl.b32 	%r104, %r10, 4;
	.loc	1 147 21                        // sk07_lm_head.py:147:21
	cvt.u64.u32 	%rd3, %r104;
	add.s64 	%rd46, %rd20, %rd3;
	.loc	1 147 67                        // sk07_lm_head.py:147:67
	mul.lo.s32 	%r105, %r25, %r73;
	mul.lo.s32 	%r106, %r26, %r73;
	mul.lo.s32 	%r107, %r27, %r73;
	mul.lo.s32 	%r108, %r28, %r73;
	.loc	1 147 51                        // sk07_lm_head.py:147:51
	cvt.s64.s32 	%rd4, %r105;
	add.s64 	%rd34, %rd46, %rd4;
	cvt.s64.s32 	%rd5, %r106;
	add.s64 	%rd35, %rd46, %rd5;
	cvt.s64.s32 	%rd6, %r107;
	add.s64 	%rd36, %rd46, %rd6;
	cvt.s64.s32 	%rd7, %r108;
	add.s64 	%rd37, %rd46, %rd7;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	setp.lt.s32 	%p1, %r20, 128;
	setp.gt.s32 	%p2, %r20, 127;
	.loc	1 160 30                        // sk07_lm_head.py:160:30
	and.b32 	%r125, %r2, 255;
	shl.b32 	%r126, %r125, 3;
	and.b32 	%r127, %r2, 112;
	xor.b32 	%r128, %r126, %r127;
	mov.b32 	%r129, global_smem;
	add.s32 	%r11, %r129, %r128;
	add.s32 	%r41, %r11, 32768;
	selp.b32 	%r42, 8, 0, %p2;
	// begin inline asm
	cp.async.ca.shared.global [ %r41 + 0 ], [ %rd33 + 0 ], 0x8, %r42;
	// end inline asm
	cp.async.commit_group;
	.loc	1 160 47                        // sk07_lm_head.py:160:47
	shl.b32 	%r130, %r125, 4;
	and.b32 	%r131, %r6, 112;
	xor.b32 	%r132, %r130, %r131;
	add.s32 	%r43, %r129, %r132;
	selp.b32 	%r44, 16, 0, %p2;
	// begin inline asm
	cp.async.cg.shared.global [ %r43 + 0 ], [ %rd34 + 0 ], 0x10, %r44;
	// end inline asm
	add.s32 	%r45, %r43, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r45 + 0 ], [ %rd35 + 0 ], 0x10, %r44;
	// end inline asm
	add.s32 	%r46, %r43, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r46 + 0 ], [ %rd36 + 0 ], 0x10, %r44;
	// end inline asm
	add.s32 	%r47, %r43, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r47 + 0 ], [ %rd37 + 0 ], 0x10, %r44;
	// end inline asm
	cp.async.commit_group;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	setp.gt.s32 	%p3, %r20, 255;
	.loc	1 161 18                        // sk07_lm_head.py:161:18
	add.s64 	%rd38, %rd33, 128;
	.loc	1 162 18                        // sk07_lm_head.py:162:18
	add.s64 	%rd39, %rd34, 128;
	add.s64 	%rd40, %rd35, 128;
	add.s64 	%rd41, %rd36, 128;
	add.s64 	%rd42, %rd37, 128;
	.loc	1 160 30                        // sk07_lm_head.py:160:30
	bar.sync 	0;
	add.s32 	%r48, %r11, 34816;
	selp.b32 	%r49, 8, 0, %p3;
	// begin inline asm
	cp.async.ca.shared.global [ %r48 + 0 ], [ %rd38 + 0 ], 0x8, %r49;
	// end inline asm
	cp.async.commit_group;
	.loc	1 160 47                        // sk07_lm_head.py:160:47
	add.s32 	%r50, %r43, 16384;
	selp.b32 	%r51, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r50 + 0 ], [ %rd39 + 0 ], 0x10, %r51;
	// end inline asm
	add.s32 	%r52, %r43, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r52 + 0 ], [ %rd40 + 0 ], 0x10, %r51;
	// end inline asm
	add.s32 	%r53, %r43, 24576;
	// begin inline asm
	cp.async.cg.shared.global [ %r53 + 0 ], [ %rd41 + 0 ], 0x10, %r51;
	// end inline asm
	add.s32 	%r54, %r43, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r54 + 0 ], [ %rd42 + 0 ], 0x10, %r51;
	// end inline asm
	cp.async.commit_group;
	mov.b32 	%r328, 0f00000000;
	cvt.u32.u64 	%r324, %rd3;
	mov.b32 	%r329, %r328;
	mov.b32 	%r330, %r328;
	mov.b32 	%r331, %r328;
	mov.b32 	%r332, %r328;
	mov.b32 	%r333, %r328;
	mov.b32 	%r334, %r328;
	mov.b32 	%r335, %r328;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	@%p1 bra 	$L__BB0_3;
// %bb.1:                               // %.lr.ph
	.loc	1 0 23                          // sk07_lm_head.py:0:23
	ld.param.b32 	%r24, [_sk07_lm_head_kernel_param_16];
	ld.param.b64 	%rd44, [_sk07_lm_head_kernel_param_7];
	shr.s32 	%r109, %r29, 31;
	shr.u32 	%r110, %r109, 25;
	add.s32 	%r111, %r29, %r110;
	shr.s32 	%r112, %r111, 7;
	shr.s32 	%r113, %r30, 31;
	shr.u32 	%r114, %r113, 25;
	add.s32 	%r115, %r30, %r114;
	shr.s32 	%r116, %r115, 7;
	shr.s32 	%r117, %r31, 31;
	shr.u32 	%r118, %r117, 25;
	add.s32 	%r119, %r31, %r118;
	shr.s32 	%r120, %r119, 7;
	shr.s32 	%r121, %r32, 31;
	shr.u32 	%r122, %r121, 25;
	add.s32 	%r123, %r32, %r122;
	shr.s32 	%r124, %r123, 7;
	cvt.s64.s32 	%rd47, %r112;
	add.s64 	%rd8, %rd44, %rd47;
	cvt.s64.s32 	%rd48, %r116;
	add.s64 	%rd9, %rd44, %rd48;
	cvt.s64.s32 	%rd49, %r120;
	add.s64 	%rd10, %rd44, %rd49;
	cvt.s64.s32 	%rd50, %r124;
	add.s64 	%rd11, %rd44, %rd50;
	.loc	1 151 28                        // sk07_lm_head.py:151:28
	shr.u32 	%r133, %r20, 7;
	add.s32 	%r134, %r133, -2;
	shl.b32 	%r135, %r8, 7;
	and.b32 	%r136, %r2, 16;
	xor.b32 	%r137, %r324, %r136;
	or.b32 	%r12, %r137, %r135;
	xor.b32 	%r13, %r12, 32;
	xor.b32 	%r14, %r12, 64;
	xor.b32 	%r15, %r12, 96;
	shl.b32 	%r138, %r10, 7;
	shl.b32 	%r139, %r7, 5;
	and.b32 	%r140, %r6, 48;
	or.b32 	%r141, %r138, %r139;
	xor.b32 	%r142, %r324, %r140;
	or.b32 	%r16, %r141, %r142;
	xor.b32 	%r17, %r16, 64;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	cvt.s64.s32 	%rd12, %r134;
	and.b32 	%r143, %r20, -128;
	cvt.u64.u32 	%rd13, %r143;
	add.s64 	%rd51, %rd3, %rd7;
	add.s64 	%rd52, %rd51, %rd20;
	add.s64 	%rd14, %rd52, 256;
	add.s64 	%rd53, %rd3, %rd6;
	add.s64 	%rd54, %rd53, %rd20;
	add.s64 	%rd15, %rd54, 256;
	add.s64 	%rd55, %rd3, %rd5;
	add.s64 	%rd56, %rd55, %rd20;
	add.s64 	%rd16, %rd56, 256;
	add.s64 	%rd57, %rd3, %rd4;
	add.s64 	%rd58, %rd57, %rd20;
	add.s64 	%rd17, %rd58, 256;
	add.s64 	%rd59, %rd2, %rd1;
	add.s64 	%rd60, %rd59, %rd19;
	add.s64 	%rd18, %rd60, 256;
	mov.b32 	%r328, 0f00000000;
	mov.b32 	%r327, 1;
	mov.b32 	%r326, -1;
	mov.b64 	%rd88, 0;
	mov.b32 	%r144, 0;
	mov.b32 	%r325, %r144;
	mov.b64 	%rd89, %rd88;
	mov.b32 	%r329, %r328;
	mov.b32 	%r330, %r328;
	mov.b32 	%r331, %r328;
	mov.b32 	%r332, %r328;
	mov.b32 	%r333, %r328;
	mov.b32 	%r334, %r328;
	mov.b32 	%r335, %r328;
$L__BB0_2:                              // %__nv_exp2f.exit
                                        // =>This Inner Loop Header: Depth=1
	setp.lt.s64 	%p4, %rd89, %rd12;
	add.s32 	%r192, %r326, 1;
	setp.gt.s32 	%p5, %r192, 1;
	selp.b32 	%r326, 0, %r192, %p5;
	.loc	1 158 39                        // sk07_lm_head.py:158:39
	cvt.s64.s32 	%rd70, %r325;
	add.s64 	%rd61, %rd8, %rd70;
	add.s64 	%rd62, %rd9, %rd70;
	add.s64 	%rd63, %rd10, %rd70;
	add.s64 	%rd64, %rd11, %rd70;
	.loc	1 158 29                        // sk07_lm_head.py:158:29
	// begin inline asm
	mov.u16 %rs1, 0x0;
	ld.global.b8 { %rs1 }, [ %rd61 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs5, %rs1;
	// begin inline asm
	mov.u16 %rs2, 0x0;
	ld.global.b8 { %rs2 }, [ %rd62 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs6, %rs2;
	// begin inline asm
	mov.u16 %rs3, 0x0;
	ld.global.b8 { %rs3 }, [ %rd63 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs7, %rs3;
	// begin inline asm
	mov.u16 %rs4, 0x0;
	ld.global.b8 { %rs4 }, [ %rd64 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs8, %rs4;
	.loc	1 159 50                        // sk07_lm_head.py:159:50
	cvt.rn.f32.s16 	%r193, %rs5;
	cvt.rn.f32.s16 	%r194, %rs6;
	cvt.rn.f32.s16 	%r195, %rs7;
	cvt.rn.f32.s16 	%r196, %rs8;
	.loc	1 158 21                        // sk07_lm_head.py:158:21
	ex2.approx.ftz.f32 	%r197, %r193;
	ex2.approx.ftz.f32 	%r198, %r194;
	ex2.approx.ftz.f32 	%r199, %r195;
	ex2.approx.ftz.f32 	%r200, %r196;
	.loc	1 160 30                        // sk07_lm_head.py:160:30
	cp.async.wait_group 	2;
	bar.sync 	0;
	shl.b32 	%r201, %r326, 11;
	add.s32 	%r202, %r129, %r201;
	add.s32 	%r203, %r202, %r12;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r145, %r146, %r147, %r148}, [%r203+32768];
	add.s32 	%r204, %r202, %r13;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r157, %r158, %r159, %r160}, [%r204+32768];
	add.s32 	%r205, %r202, %r14;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r169, %r170, %r171, %r172}, [%r205+32768];
	add.s32 	%r206, %r202, %r15;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r177, %r178, %r179, %r180}, [%r206+32768];
	.loc	1 160 47                        // sk07_lm_head.py:160:47
	shl.b32 	%r207, %r326, 14;
	add.s32 	%r208, %r129, %r207;
	add.s32 	%r209, %r208, %r16;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r149, %r150, %r161, %r162}, [%r209];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r151, %r152, %r167, %r168}, [%r209+8192];
	add.s32 	%r210, %r208, %r17;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r173, %r174, %r181, %r182}, [%r210];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r175, %r176, %r183, %r184}, [%r210+8192];
	.loc	1 160 39                        // sk07_lm_head.py:160:39
	mov.b32 	%r153, %r144;
	mov.b32 	%r154, %r144;
	mov.b32 	%r155, %r144;
	mov.b32 	%r156, %r144;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r153, %r154, %r155, %r156 }, { %r145, %r146, %r147, %r148 }, { %r149, %r150 }, { %r153, %r154, %r155, %r156 };
	// end inline asm
	mov.b32 	%r166, %r144;
	mov.b32 	%r163, %r144;
	mov.b32 	%r164, %r144;
	mov.b32 	%r165, %r144;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r163, %r164, %r165, %r166 }, { %r145, %r146, %r147, %r148 }, { %r151, %r152 }, { %r163, %r164, %r165, %r166 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r153, %r154, %r155, %r156 }, { %r157, %r158, %r159, %r160 }, { %r161, %r162 }, { %r153, %r154, %r155, %r156 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r163, %r164, %r165, %r166 }, { %r157, %r158, %r159, %r160 }, { %r167, %r168 }, { %r163, %r164, %r165, %r166 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r153, %r154, %r155, %r156 }, { %r169, %r170, %r171, %r172 }, { %r173, %r174 }, { %r153, %r154, %r155, %r156 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r163, %r164, %r165, %r166 }, { %r169, %r170, %r171, %r172 }, { %r175, %r176 }, { %r163, %r164, %r165, %r166 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r153, %r154, %r155, %r156 }, { %r177, %r178, %r179, %r180 }, { %r181, %r182 }, { %r153, %r154, %r155, %r156 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r163, %r164, %r165, %r166 }, { %r177, %r178, %r179, %r180 }, { %r183, %r184 }, { %r163, %r164, %r165, %r166 };
	// end inline asm
	.loc	1 160 79                        // sk07_lm_head.py:160:79
	cvt.rn.f32.s32 	%r211, %r163;
	cvt.rn.f32.s32 	%r212, %r164;
	cvt.rn.f32.s32 	%r213, %r165;
	cvt.rn.f32.s32 	%r214, %r166;
	cvt.rn.f32.s32 	%r215, %r153;
	cvt.rn.f32.s32 	%r216, %r154;
	cvt.rn.f32.s32 	%r217, %r155;
	cvt.rn.f32.s32 	%r218, %r156;
	.loc	1 160 15                        // sk07_lm_head.py:160:15
	fma.rn.f32 	%r331, %r198, %r218, %r331;
	fma.rn.f32 	%r330, %r197, %r217, %r330;
	fma.rn.f32 	%r329, %r198, %r216, %r329;
	fma.rn.f32 	%r328, %r197, %r215, %r328;
	fma.rn.f32 	%r335, %r200, %r214, %r335;
	fma.rn.f32 	%r334, %r199, %r213, %r334;
	fma.rn.f32 	%r333, %r200, %r212, %r333;
	fma.rn.f32 	%r332, %r199, %r211, %r332;
	.loc	1 162 18                        // sk07_lm_head.py:162:18
	add.s64 	%rd65, %rd18, %rd88;
	add.s64 	%rd66, %rd17, %rd88;
	add.s64 	%rd67, %rd16, %rd88;
	add.s64 	%rd68, %rd15, %rd88;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	add.s64 	%rd69, %rd14, %rd88;
	add.s32 	%r219, %r327, 1;
	setp.gt.s32 	%p6, %r219, 1;
	selp.b32 	%r327, 0, %r219, %p6;
	.loc	1 160 30                        // sk07_lm_head.py:160:30
	shl.b32 	%r220, %r327, 11;
	bar.sync 	0;
	add.s32 	%r221, %r11, %r220;
	add.s32 	%r185, %r221, 32768;
	selp.b32 	%r186, 8, 0, %p4;
	// begin inline asm
	cp.async.ca.shared.global [ %r185 + 0 ], [ %rd65 + 0 ], 0x8, %r186;
	// end inline asm
	cp.async.commit_group;
	.loc	1 160 47                        // sk07_lm_head.py:160:47
	shl.b32 	%r222, %r327, 14;
	add.s32 	%r187, %r43, %r222;
	selp.b32 	%r188, 16, 0, %p4;
	// begin inline asm
	cp.async.cg.shared.global [ %r187 + 0 ], [ %rd66 + 0 ], 0x10, %r188;
	// end inline asm
	add.s32 	%r189, %r187, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r189 + 0 ], [ %rd67 + 0 ], 0x10, %r188;
	// end inline asm
	add.s32 	%r190, %r187, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r190 + 0 ], [ %rd68 + 0 ], 0x10, %r188;
	// end inline asm
	add.s32 	%r191, %r187, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r191 + 0 ], [ %rd69 + 0 ], 0x10, %r188;
	// end inline asm
	cp.async.commit_group;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	add.s64 	%rd89, %rd89, 1;
	add.s64 	%rd88, %rd88, 128;
	add.s32 	%r325, %r325, %r24;
	setp.ne.b64 	%p7, %rd13, %rd88;
	@%p7 bra 	$L__BB0_2;
$L__BB0_3:                              // %._crit_edge
	.loc	1 141 45                        // sk07_lm_head.py:141:45
	and.b32 	%r242, %r2, 28;
	bfe.u32 	%r243, %r2, 2, 3;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r244, %r243, %r1;
	or.b32 	%r245, %r244, 8;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r246, %r245, %r18;
	rem.s32 	%r247, %r244, %r18;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 164 38                        // sk07_lm_head.py:164:38
	mad.wide.s32 	%rd71, %r247, 4, %rd23;
	mad.wide.s32 	%rd72, %r246, 4, %rd23;
	.loc	1 164 24                        // sk07_lm_head.py:164:24
	// begin inline asm
	mov.u32 %r223, 0x0;
	ld.global.b32 { %r223 }, [ %rd71 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r224, 0x0;
	ld.global.b32 { %r224 }, [ %rd72 + 0 ];
	// end inline asm
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r248, %r328, %r223;
	mul.f32 	%r249, %r329, %r223;
	mul.f32 	%r250, %r330, %r224;
	mul.f32 	%r251, %r331, %r224;
	mul.f32 	%r252, %r332, %r223;
	mul.f32 	%r253, %r333, %r223;
	mul.f32 	%r254, %r334, %r224;
	mul.f32 	%r255, %r335, %r224;
	.loc	1 165 38                        // sk07_lm_head.py:165:38
	mad.wide.s32 	%rd73, %r29, 4, %rd24;
	mad.wide.s32 	%rd74, %r30, 4, %rd24;
	mad.wide.s32 	%rd75, %r31, 4, %rd24;
	mad.wide.s32 	%rd76, %r32, 4, %rd24;
	.loc	1 165 24                        // sk07_lm_head.py:165:24
	// begin inline asm
	mov.u32 %r225, 0x0;
	ld.global.b32 { %r225 }, [ %rd73 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r226, 0x0;
	ld.global.b32 { %r226 }, [ %rd74 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r227, 0x0;
	ld.global.b32 { %r227 }, [ %rd75 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r228, 0x0;
	ld.global.b32 { %r228 }, [ %rd76 + 0 ];
	// end inline asm
	.loc	1 166 49                        // sk07_lm_head.py:166:49
	mul.lo.s32 	%r256, %r5, %r22;
	.loc	1 166 31                        // sk07_lm_head.py:166:31
	mad.wide.s32 	%rd86, %r256, 2, %rd22;
	.loc	1 166 80                        // sk07_lm_head.py:166:80
	mul.lo.s32 	%r257, %r33, %r23;
	mul.lo.s32 	%r258, %r34, %r23;
	mul.lo.s32 	%r259, %r35, %r23;
	mul.lo.s32 	%r260, %r36, %r23;
	mul.lo.s32 	%r261, %r37, %r23;
	mul.lo.s32 	%r262, %r38, %r23;
	mul.lo.s32 	%r263, %r39, %r23;
	mul.lo.s32 	%r264, %r40, %r23;
	.loc	1 166 64                        // sk07_lm_head.py:166:64
	mad.wide.s32 	%rd77, %r257, 2, %rd86;
	mad.wide.s32 	%rd78, %r258, 2, %rd86;
	mad.wide.s32 	%rd79, %r259, 2, %rd86;
	mad.wide.s32 	%rd80, %r260, 2, %rd86;
	mad.wide.s32 	%rd81, %r261, 2, %rd86;
	mad.wide.s32 	%rd82, %r262, 2, %rd86;
	mad.wide.s32 	%rd83, %r263, 2, %rd86;
	mad.wide.s32 	%rd84, %r264, 2, %rd86;
	.loc	1 166 19                        // sk07_lm_head.py:166:19
	// begin inline asm
	mov.u16 %rs9, 0x0;
	ld.global.b16 { %rs9 }, [ %rd77 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs10, 0x0;
	ld.global.b16 { %rs10 }, [ %rd78 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs11, 0x0;
	ld.global.b16 { %rs11 }, [ %rd79 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs12, 0x0;
	ld.global.b16 { %rs12 }, [ %rd80 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs13, 0x0;
	ld.global.b16 { %rs13 }, [ %rd81 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs14, 0x0;
	ld.global.b16 { %rs14 }, [ %rd82 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs15, 0x0;
	ld.global.b16 { %rs15 }, [ %rd83 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs16, 0x0;
	ld.global.b16 { %rs16 }, [ %rd84 + 0 ];
	// end inline asm
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	and.b32 	%r265, %r2, 120;
	shl.b32 	%r266, %r265, 5;
	or.b32 	%r267, %r266, %r324;
	xor.b32 	%r268, %r267, %r3;
	add.s32 	%r229, %r129, %r268;
	mov.b32 	%r230, {%rs9, %rs10};
	mov.b32 	%r231, {%rs11, %rs12};
	mov.b32 	%r232, {%rs13, %rs14};
	mov.b32 	%r233, {%rs15, %rs16};
	// begin inline asm
	st.shared.v4.b32 [ %r229 + 0 ], { %r230, %r231, %r232, %r233 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r269, %r10, 9;
	shl.b32 	%r270, %r2, 4;
	and.b32 	%r271, %r270, 496;
	shr.u32 	%r272, %r7, 1;
	xor.b32 	%r273, %r271, %r272;
	add.s32 	%r274, %r129, %r269;
	add.s32 	%r275, %r274, %r273;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r276, %r277, %r278, %r279}, [%r275];
	mov.b32 	{%rs25, %rs26}, %r276;
	mov.b32 	{%rs27, %rs28}, %r277;
	mov.b32 	{%rs29, %rs30}, %r278;
	mov.b32 	{%rs31, %rs32}, %r279;
	cvt.f32.bf16 	%r280, %rs25;
	cvt.f32.bf16 	%r281, %rs26;
	cvt.f32.bf16 	%r282, %rs27;
	cvt.f32.bf16 	%r283, %rs28;
	cvt.f32.bf16 	%r284, %rs29;
	cvt.f32.bf16 	%r285, %rs30;
	cvt.f32.bf16 	%r286, %rs31;
	cvt.f32.bf16 	%r287, %rs32;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r288, %r248, %r225, %r280;
	fma.rn.f32 	%r289, %r249, %r226, %r281;
	fma.rn.f32 	%r290, %r250, %r225, %r282;
	fma.rn.f32 	%r291, %r251, %r226, %r283;
	fma.rn.f32 	%r292, %r252, %r227, %r284;
	fma.rn.f32 	%r293, %r253, %r228, %r285;
	fma.rn.f32 	%r294, %r254, %r227, %r286;
	fma.rn.f32 	%r295, %r255, %r228, %r287;
	.loc	1 173 31                        // sk07_lm_head.py:173:31
	setp.lt.s32 	%p9, %r4, %r18;
	.loc	1 173 54                        // sk07_lm_head.py:173:54
	setp.lt.s32 	%p10, %r9, %r19;
	.loc	1 173 37                        // sk07_lm_head.py:173:37
	and.pred 	%p8, %p9, %p10;
	.loc	1 171 35                        // sk07_lm_head.py:171:35
	mul.lo.s32 	%r296, %r4, %r21;
	.loc	1 171 18                        // sk07_lm_head.py:171:18
	mad.wide.s32 	%rd87, %r296, 2, %rd21;
	.loc	1 171 50                        // sk07_lm_head.py:171:50
	mad.wide.s32 	%rd85, %r9, 2, %rd87;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16.f32 	%rs17, %r288;
	cvt.rn.bf16.f32 	%rs18, %r289;
	cvt.rn.bf16.f32 	%rs19, %r290;
	cvt.rn.bf16.f32 	%rs20, %r291;
	cvt.rn.bf16.f32 	%rs21, %r292;
	cvt.rn.bf16.f32 	%rs22, %r293;
	cvt.rn.bf16.f32 	%rs23, %r294;
	cvt.rn.bf16.f32 	%rs24, %r295;
	bar.sync 	0;
	shl.b32 	%r297, %r2, 5;
	and.b32 	%r298, %r297, 768;
	shl.b32 	%r299, %r242, 1;
	and.b32 	%r300, %r2, 1;
	neg.s32 	%r301, %r300;
	and.b32 	%r302, %r301, 1088;
	bfe.s32 	%r303, %r2, 1, 1;
	and.b32 	%r304, %r303, 2052;
	or.b32 	%r305, %r298, %r299;
	or.b32 	%r306, %r302, %r305;
	xor.b32 	%r307, %r306, %r272;
	or.b32 	%r308, %r307, %r304;
	add.s32 	%r234, %r129, %r308;
	// begin inline asm
	st.shared.v2.b16 [ %r234 + 0 ], { %rs17, %rs18 };
	// end inline asm
	add.s32 	%r235, %r234, 128;
	// begin inline asm
	st.shared.v2.b16 [ %r235 + 0 ], { %rs19, %rs20 };
	// end inline asm
	xor.b32 	%r309, %r308, 4;
	add.s32 	%r236, %r129, %r309;
	// begin inline asm
	st.shared.v2.b16 [ %r236 + 0 ], { %rs21, %rs22 };
	// end inline asm
	add.s32 	%r237, %r236, 128;
	// begin inline asm
	st.shared.v2.b16 [ %r237 + 0 ], { %rs23, %rs24 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r310, %r2, 3;
	and.b32 	%r311, %r310, 768;
	shr.u32 	%r312, %r265, 1;
	and.b32 	%r313, %r2, 128;
	or.b32 	%r314, %r324, %r311;
	xor.b32 	%r315, %r314, %r312;
	or.b32 	%r316, %r315, %r313;
	add.s32 	%r317, %r129, %r316;
	ld.shared.b32 	%r238, [%r317];
	xor.b32 	%r318, %r316, 64;
	add.s32 	%r319, %r129, %r318;
	ld.shared.b32 	%r239, [%r319+1024];
	xor.b32 	%r320, %r316, 4;
	add.s32 	%r321, %r129, %r320;
	ld.shared.b32 	%r240, [%r321+2048];
	xor.b32 	%r322, %r316, 68;
	add.s32 	%r323, %r129, %r322;
	ld.shared.b32 	%r241, [%r323+3072];
	.loc	1 172 8                         // sk07_lm_head.py:172:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd85 + 0 ], { %r238, %r239, %r240, %r241 };
	// end inline asm
	.loc	1 170 4                         // sk07_lm_head.py:170:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk07_lm_head.py"
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
.b32 159                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x98 DW_TAG_compile_unit
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
.b8 55
.b8 95
.b8 108
.b8 109
.b8 95
.b8 104
.b8 101
.b8 97
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
.b8 2                                   // Abbrev [2] 0x45:0x17 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 55
.b8 95
.b8 108
.b8 109
.b8 95
.b8 104
.b8 101
.b8 97
.b8 100
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x5c:0x46 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 69                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x71:0x18 DW_TAG_inlined_subroutine
.b32 69                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 133                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x89:0x18 DW_TAG_inlined_subroutine
.b32 69                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 134                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_1 = _Nativo(
    "sk07_lm_head/tile16x128x128_shift0_abi17",
    _PTX_1, "_sk07_lm_head_kernel",
    warps=8, shared=36864,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 17, 18, 19],
    horneado={12: 1, 13: 1, 16: 1, 20: 1, 21: 16, 22: 128, 23: 128, 24: 8, 25: 128},
    div16=[9, 10, 11, 14, 15, 17, 18],
)

_PTX_2 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk07_lm_head_kernel    // -- Begin function _sk07_lm_head_kernel
.extern .shared .align 16 .b8 global_smem[];
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
                                        // @_sk07_lm_head_kernel
.visible .entry _sk07_lm_head_kernel(
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_6,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_7,
	.param .u32 _sk07_lm_head_kernel_param_8,
	.param .u32 _sk07_lm_head_kernel_param_9,
	.param .u32 _sk07_lm_head_kernel_param_10,
	.param .u32 _sk07_lm_head_kernel_param_11,
	.param .u32 _sk07_lm_head_kernel_param_12,
	.param .u32 _sk07_lm_head_kernel_param_13,
	.param .u32 _sk07_lm_head_kernel_param_14,
	.param .u32 _sk07_lm_head_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_16,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_17
)
.reqntid 256
{
	.reg .pred 	%p<25>;
	.reg .b16 	%rs<145>;
	.reg .b32 	%r<953>;
	.reg .b64 	%rd<229>;
	.loc	1 123 0                         // sk07_lm_head.py:123:0
$L__func_begin0:
	.loc	1 123 0                         // sk07_lm_head.py:123:0

// %bb.0:
	ld.param.b32 	%r25, [_sk07_lm_head_kernel_param_14];
	ld.param.b32 	%r24, [_sk07_lm_head_kernel_param_13];
	ld.param.b32 	%r23, [_sk07_lm_head_kernel_param_10];
	ld.param.b32 	%r22, [_sk07_lm_head_kernel_param_9];
	ld.param.b32 	%r21, [_sk07_lm_head_kernel_param_8];
	ld.param.b64 	%rd33, [_sk07_lm_head_kernel_param_6];
	ld.param.b64 	%rd32, [_sk07_lm_head_kernel_param_5];
	ld.param.b64 	%rd31, [_sk07_lm_head_kernel_param_4];
	ld.param.b64 	%rd30, [_sk07_lm_head_kernel_param_2];
	ld.param.b64 	%rd29, [_sk07_lm_head_kernel_param_1];
	ld.param.b64 	%rd28, [_sk07_lm_head_kernel_param_0];
$L__tmp0:
	.loc	1 132 24                        // sk07_lm_head.py:132:24
	mov.u32 	%r73, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk07_lm_head.py:133:27 ]
	add.s32 	%r74, %r21, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk07_lm_head.py:133:27 ]
	shr.s32 	%r75, %r74, 31;
	shr.u32 	%r76, %r75, 25;
	add.s32 	%r77, %r74, %r76;
	shr.s32 	%r78, %r77, 7;
	ld.param.b64 	%rd62, [_sk07_lm_head_kernel_param_3];
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk07_lm_head.py:134:27 ]
	add.s32 	%r79, %r22, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk07_lm_head.py:134:27 ]
	shr.s32 	%r80, %r79, 31;
	shr.u32 	%r81, %r80, 25;
	add.s32 	%r82, %r79, %r81;
	shr.s32 	%r83, %r82, 7;
$L__tmp3:
	.loc	1 135 29                        // sk07_lm_head.py:135:29
	shl.b32 	%r84, %r83, 3;
	.loc	1 136 22                        // sk07_lm_head.py:136:22
	div.s32 	%r85, %r73, %r84;
	.loc	1 136 38                        // sk07_lm_head.py:136:38
	shl.b32 	%r86, %r85, 3;
	.loc	1 137 30                        // sk07_lm_head.py:137:30
	sub.s32 	%r87, %r78, %r86;
	.loc	1 137 39                        // sk07_lm_head.py:137:39
	min.s32 	%r88, %r87, 8;
	ld.param.b32 	%r89, [_sk07_lm_head_kernel_param_11];
	.loc	1 138 30                        // sk07_lm_head.py:138:30
	mul.lo.s32 	%r90, %r85, %r84;
	ld.param.b32 	%r91, [_sk07_lm_head_kernel_param_12];
	sub.s32 	%r92, %r73, %r90;
	.loc	1 139 36                        // sk07_lm_head.py:139:36
	div.s32 	%r93, %r92, %r88;
	.loc	1 138 46                        // sk07_lm_head.py:138:46
	mul.lo.s32 	%r94, %r93, %r88;
	sub.s32 	%r95, %r92, %r94;
	.loc	1 138 23                        // sk07_lm_head.py:138:23
	add.s32 	%r96, %r95, %r86;
	.loc	1 141 22                        // sk07_lm_head.py:141:22
	shl.b32 	%r1, %r96, 7;
	.loc	1 141 45                        // sk07_lm_head.py:141:45
	mov.u32 	%r2, %tid.x;
	and.b32 	%r3, %r2, 248;
	bfe.u32 	%r97, %r2, 3, 5;
	or.b32 	%r98, %r97, 32;
	or.b32 	%r99, %r97, 64;
	or.b32 	%r100, %r97, 96;
	and.b32 	%r4, %r2, 3;
	shl.b32 	%r101, %r4, 1;
	and.b32 	%r5, %r2, 96;
	shr.u32 	%r102, %r5, 2;
	or.b32 	%r103, %r101, %r102;
	and.b32 	%r6, %r2, 7;
	shl.b32 	%r104, %r6, 4;
	and.b32 	%r7, %r2, 15;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r105, %r1, %r97;
	or.b32 	%r106, %r1, %r98;
	or.b32 	%r107, %r1, %r99;
	or.b32 	%r108, %r1, %r100;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r8, %r105, %r21;
	rem.s32 	%r9, %r106, %r21;
	rem.s32 	%r10, %r107, %r21;
	rem.s32 	%r11, %r108, %r21;
	.loc	1 142 22                        // sk07_lm_head.py:142:22
	shl.b32 	%r12, %r93, 7;
	.loc	1 142 32                        // sk07_lm_head.py:142:32
	or.b32 	%r109, %r12, %r97;
	or.b32 	%r110, %r12, %r98;
	or.b32 	%r111, %r12, %r99;
	or.b32 	%r112, %r12, %r100;
	or.b32 	%r113, %r12, %r103;
	or.b32 	%r114, %r113, 32;
	or.b32 	%r115, %r113, 64;
	or.b32 	%r116, %r113, 96;
	or.b32 	%r117, %r12, %r104;
	or.b32 	%r118, %r117, 4;
	or.b32 	%r119, %r117, 8;
	or.b32 	%r120, %r117, 12;
	.loc	1 142 57                        // sk07_lm_head.py:142:57
	rem.s32 	%r121, %r109, %r22;
	rem.s32 	%r122, %r110, %r22;
	rem.s32 	%r123, %r111, %r22;
	rem.s32 	%r124, %r112, %r22;
	rem.s32 	%r125, %r113, %r22;
	rem.s32 	%r126, %r114, %r22;
	rem.s32 	%r127, %r115, %r22;
	rem.s32 	%r128, %r116, %r22;
	rem.s32 	%r129, %r117, %r22;
	rem.s32 	%r130, %r118, %r22;
	rem.s32 	%r131, %r119, %r22;
	rem.s32 	%r132, %r120, %r22;
	.loc	1 145 29                        // sk07_lm_head.py:145:29
	mad.wide.s32 	%rd34, %r121, 4, %rd62;
	mad.wide.s32 	%rd35, %r122, 4, %rd62;
	mad.wide.s32 	%rd36, %r123, 4, %rd62;
	mad.wide.s32 	%rd37, %r124, 4, %rd62;
	mad.wide.s32 	%rd38, %r125, 4, %rd62;
	mad.wide.s32 	%rd39, %r126, 4, %rd62;
	mad.wide.s32 	%rd40, %r127, 4, %rd62;
	mad.wide.s32 	%rd41, %r128, 4, %rd62;
	mad.wide.s32 	%rd42, %r129, 4, %rd62;
	mad.wide.s32 	%rd43, %r130, 4, %rd62;
	mad.wide.s32 	%rd44, %r131, 4, %rd62;
	mad.wide.s32 	%rd45, %r132, 4, %rd62;
	.loc	1 145 19                        // sk07_lm_head.py:145:19
	// begin inline asm
	mov.u32 %r27, 0x0;
	ld.global.b32 { %r27 }, [ %rd34 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r28, 0x0;
	ld.global.b32 { %r28 }, [ %rd35 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r29, 0x0;
	ld.global.b32 { %r29 }, [ %rd36 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r30, 0x0;
	ld.global.b32 { %r30 }, [ %rd37 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r31, 0x0;
	mov.u32 %r32, 0x0;
	ld.global.v2.b32 { %r31, %r32 }, [ %rd38 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r33, 0x0;
	mov.u32 %r34, 0x0;
	ld.global.v2.b32 { %r33, %r34 }, [ %rd39 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r35, 0x0;
	mov.u32 %r36, 0x0;
	ld.global.v2.b32 { %r35, %r36 }, [ %rd40 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r37, 0x0;
	mov.u32 %r38, 0x0;
	ld.global.v2.b32 { %r37, %r38 }, [ %rd41 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r39, 0x0;
	mov.u32 %r40, 0x0;
	mov.u32 %r41, 0x0;
	mov.u32 %r42, 0x0;
	ld.global.v4.b32 { %r39, %r40, %r41, %r42 }, [ %rd42 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r43, 0x0;
	mov.u32 %r44, 0x0;
	mov.u32 %r45, 0x0;
	mov.u32 %r46, 0x0;
	ld.global.v4.b32 { %r43, %r44, %r45, %r46 }, [ %rd43 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r47, 0x0;
	mov.u32 %r48, 0x0;
	mov.u32 %r49, 0x0;
	mov.u32 %r50, 0x0;
	ld.global.v4.b32 { %r47, %r48, %r49, %r50 }, [ %rd44 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r51, 0x0;
	mov.u32 %r52, 0x0;
	mov.u32 %r53, 0x0;
	mov.u32 %r54, 0x0;
	ld.global.v4.b32 { %r51, %r52, %r53, %r54 }, [ %rd45 + 0 ];
	// end inline asm
	.loc	1 146 39                        // sk07_lm_head.py:146:39
	mul.lo.s32 	%r133, %r8, %r89;
	mul.lo.s32 	%r134, %r9, %r89;
	mul.lo.s32 	%r135, %r10, %r89;
	mul.lo.s32 	%r136, %r11, %r89;
	.loc	1 146 21                        // sk07_lm_head.py:146:21
	cvt.s64.s32 	%rd1, %r133;
	add.s64 	%rd64, %rd28, %rd1;
	cvt.s64.s32 	%rd2, %r134;
	add.s64 	%rd65, %rd28, %rd2;
	cvt.s64.s32 	%rd3, %r135;
	add.s64 	%rd66, %rd28, %rd3;
	cvt.s64.s32 	%rd4, %r136;
	add.s64 	%rd67, %rd28, %rd4;
	.loc	1 146 51                        // sk07_lm_head.py:146:51
	cvt.u64.u32 	%rd5, %r104;
	add.s64 	%rd46, %rd64, %rd5;
	add.s64 	%rd47, %rd65, %rd5;
	add.s64 	%rd48, %rd66, %rd5;
	add.s64 	%rd49, %rd67, %rd5;
	.loc	1 147 21                        // sk07_lm_head.py:147:21
	add.s64 	%rd68, %rd29, %rd5;
	.loc	1 147 67                        // sk07_lm_head.py:147:67
	mul.lo.s32 	%r137, %r27, %r91;
	mul.lo.s32 	%r138, %r28, %r91;
	mul.lo.s32 	%r139, %r29, %r91;
	mul.lo.s32 	%r140, %r30, %r91;
	.loc	1 147 51                        // sk07_lm_head.py:147:51
	cvt.s64.s32 	%rd6, %r137;
	add.s64 	%rd50, %rd68, %rd6;
	cvt.s64.s32 	%rd7, %r138;
	add.s64 	%rd51, %rd68, %rd7;
	cvt.s64.s32 	%rd8, %r139;
	add.s64 	%rd52, %rd68, %rd8;
	cvt.s64.s32 	%rd9, %r140;
	add.s64 	%rd53, %rd68, %rd9;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	setp.gt.s32 	%p1, %r23, 127;
	.loc	1 160 30                        // sk07_lm_head.py:160:30
	shl.b32 	%r13, %r2, 4;
	and.b32 	%r173, %r13, 4080;
	and.b32 	%r14, %r2, 56;
	shl.b32 	%r174, %r14, 1;
	xor.b32 	%r175, %r173, %r174;
	mov.b32 	%r176, global_smem;
	add.s32 	%r55, %r176, %r175;
	selp.b32 	%r56, 16, 0, %p1;
	// begin inline asm
	cp.async.cg.shared.global [ %r55 + 0 ], [ %rd46 + 0 ], 0x10, %r56;
	// end inline asm
	add.s32 	%r57, %r55, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r57 + 0 ], [ %rd47 + 0 ], 0x10, %r56;
	// end inline asm
	add.s32 	%r58, %r55, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r58 + 0 ], [ %rd48 + 0 ], 0x10, %r56;
	// end inline asm
	add.s32 	%r59, %r55, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r59 + 0 ], [ %rd49 + 0 ], 0x10, %r56;
	// end inline asm
	cp.async.commit_group;
	.loc	1 160 47                        // sk07_lm_head.py:160:47
	add.s32 	%r60, %r55, 32768;
	// begin inline asm
	cp.async.cg.shared.global [ %r60 + 0 ], [ %rd50 + 0 ], 0x10, %r56;
	// end inline asm
	add.s32 	%r61, %r55, 36864;
	// begin inline asm
	cp.async.cg.shared.global [ %r61 + 0 ], [ %rd51 + 0 ], 0x10, %r56;
	// end inline asm
	add.s32 	%r62, %r55, 40960;
	// begin inline asm
	cp.async.cg.shared.global [ %r62 + 0 ], [ %rd52 + 0 ], 0x10, %r56;
	// end inline asm
	add.s32 	%r63, %r55, 45056;
	// begin inline asm
	cp.async.cg.shared.global [ %r63 + 0 ], [ %rd53 + 0 ], 0x10, %r56;
	// end inline asm
	cp.async.commit_group;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	setp.gt.s32 	%p2, %r23, 255;
	.loc	1 161 18                        // sk07_lm_head.py:161:18
	add.s64 	%rd54, %rd46, 128;
	add.s64 	%rd55, %rd47, 128;
	add.s64 	%rd56, %rd48, 128;
	add.s64 	%rd57, %rd49, 128;
	.loc	1 162 18                        // sk07_lm_head.py:162:18
	add.s64 	%rd58, %rd50, 128;
	add.s64 	%rd59, %rd51, 128;
	add.s64 	%rd60, %rd52, 128;
	add.s64 	%rd61, %rd53, 128;
	.loc	1 160 30                        // sk07_lm_head.py:160:30
	bar.sync 	0;
	add.s32 	%r64, %r55, 16384;
	selp.b32 	%r65, 16, 0, %p2;
	// begin inline asm
	cp.async.cg.shared.global [ %r64 + 0 ], [ %rd54 + 0 ], 0x10, %r65;
	// end inline asm
	add.s32 	%r66, %r55, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r66 + 0 ], [ %rd55 + 0 ], 0x10, %r65;
	// end inline asm
	add.s32 	%r67, %r55, 24576;
	// begin inline asm
	cp.async.cg.shared.global [ %r67 + 0 ], [ %rd56 + 0 ], 0x10, %r65;
	// end inline asm
	add.s32 	%r68, %r55, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r68 + 0 ], [ %rd57 + 0 ], 0x10, %r65;
	// end inline asm
	cp.async.commit_group;
	.loc	1 160 47                        // sk07_lm_head.py:160:47
	add.s32 	%r69, %r55, 49152;
	// begin inline asm
	cp.async.cg.shared.global [ %r69 + 0 ], [ %rd58 + 0 ], 0x10, %r65;
	// end inline asm
	add.s32 	%r70, %r55, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r70 + 0 ], [ %rd59 + 0 ], 0x10, %r65;
	// end inline asm
	add.s32 	%r71, %r55, 57344;
	// begin inline asm
	cp.async.cg.shared.global [ %r71 + 0 ], [ %rd60 + 0 ], 0x10, %r65;
	// end inline asm
	add.s32 	%r72, %r55, 61440;
	// begin inline asm
	cp.async.cg.shared.global [ %r72 + 0 ], [ %rd61 + 0 ], 0x10, %r65;
	// end inline asm
	cp.async.commit_group;
	cvt.u32.u64 	%r883, %rd5;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	@%p1 bra 	$L__BB0_2;
	bra.uni 	$L__BB0_1;
$L__BB0_2:                              // %.lr.ph
	.loc	1 0 23                          // sk07_lm_head.py:0:23
	ld.param.b32 	%r26, [_sk07_lm_head_kernel_param_15];
	ld.param.b64 	%rd63, [_sk07_lm_head_kernel_param_7];
	shr.s32 	%r141, %r31, 31;
	shr.u32 	%r142, %r141, 25;
	add.s32 	%r143, %r31, %r142;
	shr.s32 	%r144, %r143, 7;
	shr.s32 	%r145, %r32, 31;
	shr.u32 	%r146, %r145, 25;
	add.s32 	%r147, %r32, %r146;
	shr.s32 	%r148, %r147, 7;
	shr.s32 	%r149, %r33, 31;
	shr.u32 	%r150, %r149, 25;
	add.s32 	%r151, %r33, %r150;
	shr.s32 	%r152, %r151, 7;
	shr.s32 	%r153, %r34, 31;
	shr.u32 	%r154, %r153, 25;
	add.s32 	%r155, %r34, %r154;
	shr.s32 	%r156, %r155, 7;
	shr.s32 	%r157, %r35, 31;
	shr.u32 	%r158, %r157, 25;
	add.s32 	%r159, %r35, %r158;
	shr.s32 	%r160, %r159, 7;
	shr.s32 	%r161, %r36, 31;
	shr.u32 	%r162, %r161, 25;
	add.s32 	%r163, %r36, %r162;
	shr.s32 	%r164, %r163, 7;
	shr.s32 	%r165, %r37, 31;
	shr.u32 	%r166, %r165, 25;
	add.s32 	%r167, %r37, %r166;
	shr.s32 	%r168, %r167, 7;
	shr.s32 	%r169, %r38, 31;
	shr.u32 	%r170, %r169, 25;
	add.s32 	%r171, %r38, %r170;
	shr.s32 	%r172, %r171, 7;
	cvt.s64.s32 	%rd69, %r144;
	add.s64 	%rd10, %rd63, %rd69;
	cvt.s64.s32 	%rd70, %r148;
	add.s64 	%rd11, %rd63, %rd70;
	cvt.s64.s32 	%rd71, %r152;
	add.s64 	%rd12, %rd63, %rd71;
	cvt.s64.s32 	%rd72, %r156;
	add.s64 	%rd13, %rd63, %rd72;
	cvt.s64.s32 	%rd73, %r160;
	add.s64 	%rd14, %rd63, %rd73;
	cvt.s64.s32 	%rd74, %r164;
	add.s64 	%rd15, %rd63, %rd74;
	cvt.s64.s32 	%rd75, %r168;
	add.s64 	%rd16, %rd63, %rd75;
	cvt.s64.s32 	%rd76, %r172;
	add.s64 	%rd17, %rd63, %rd76;
	.loc	1 151 28                        // sk07_lm_head.py:151:28
	shr.u32 	%r177, %r23, 7;
	add.s32 	%r178, %r177, -2;
	shl.b32 	%r179, %r7, 7;
	and.b32 	%r180, %r13, 2160;
	and.b32 	%r887, %r2, 16;
	or.b32 	%r181, %r179, %r180;
	xor.b32 	%r15, %r181, %r887;
	xor.b32 	%r16, %r15, 32;
	xor.b32 	%r17, %r15, 64;
	xor.b32 	%r18, %r15, 96;
	shl.b32 	%r182, %r6, 7;
	shl.b32 	%r183, %r5, 5;
	shl.b32 	%r888, %r2, 1;
	and.b32 	%r184, %r888, 48;
	or.b32 	%r185, %r182, %r183;
	xor.b32 	%r186, %r883, %r184;
	or.b32 	%r19, %r185, %r186;
	xor.b32 	%r20, %r19, 64;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	cvt.s64.s32 	%rd18, %r178;
	and.b32 	%r187, %r23, -128;
	cvt.u64.u32 	%rd19, %r187;
	add.s64 	%rd77, %rd5, %rd9;
	add.s64 	%rd78, %rd77, %rd29;
	add.s64 	%rd20, %rd78, 256;
	add.s64 	%rd79, %rd5, %rd8;
	add.s64 	%rd80, %rd79, %rd29;
	add.s64 	%rd21, %rd80, 256;
	add.s64 	%rd81, %rd5, %rd7;
	add.s64 	%rd82, %rd81, %rd29;
	add.s64 	%rd22, %rd82, 256;
	add.s64 	%rd83, %rd5, %rd6;
	add.s64 	%rd84, %rd83, %rd29;
	add.s64 	%rd23, %rd84, 256;
	add.s64 	%rd85, %rd5, %rd4;
	add.s64 	%rd86, %rd85, %rd28;
	add.s64 	%rd24, %rd86, 256;
	add.s64 	%rd87, %rd5, %rd3;
	add.s64 	%rd88, %rd87, %rd28;
	add.s64 	%rd25, %rd88, 256;
	add.s64 	%rd89, %rd5, %rd2;
	add.s64 	%rd90, %rd89, %rd28;
	add.s64 	%rd26, %rd90, 256;
	add.s64 	%rd91, %rd5, %rd1;
	add.s64 	%rd92, %rd91, %rd28;
	add.s64 	%rd27, %rd92, 256;
	mov.b32 	%r889, 0f00000000;
	mov.b32 	%r886, 1;
	mov.b32 	%r885, -1;
	mov.b64 	%rd227, 0;
	mov.b32 	%r188, 0;
	mov.b32 	%r884, %r188;
	mov.b64 	%rd228, %rd227;
	mov.b32 	%r890, %r889;
	mov.b32 	%r891, %r889;
	mov.b32 	%r892, %r889;
	mov.b32 	%r893, %r889;
	mov.b32 	%r894, %r889;
	mov.b32 	%r895, %r889;
	mov.b32 	%r896, %r889;
	mov.b32 	%r897, %r889;
	mov.b32 	%r898, %r889;
	mov.b32 	%r899, %r889;
	mov.b32 	%r900, %r889;
	mov.b32 	%r901, %r889;
	mov.b32 	%r902, %r889;
	mov.b32 	%r903, %r889;
	mov.b32 	%r904, %r889;
	mov.b32 	%r905, %r889;
	mov.b32 	%r906, %r889;
	mov.b32 	%r907, %r889;
	mov.b32 	%r908, %r889;
	mov.b32 	%r909, %r889;
	mov.b32 	%r910, %r889;
	mov.b32 	%r911, %r889;
	mov.b32 	%r912, %r889;
	mov.b32 	%r913, %r889;
	mov.b32 	%r914, %r889;
	mov.b32 	%r915, %r889;
	mov.b32 	%r916, %r889;
	mov.b32 	%r917, %r889;
	mov.b32 	%r918, %r889;
	mov.b32 	%r919, %r889;
	mov.b32 	%r920, %r889;
	mov.b32 	%r921, %r889;
	mov.b32 	%r922, %r889;
	mov.b32 	%r923, %r889;
	mov.b32 	%r924, %r889;
	mov.b32 	%r925, %r889;
	mov.b32 	%r926, %r889;
	mov.b32 	%r927, %r889;
	mov.b32 	%r928, %r889;
	mov.b32 	%r929, %r889;
	mov.b32 	%r930, %r889;
	mov.b32 	%r931, %r889;
	mov.b32 	%r932, %r889;
	mov.b32 	%r933, %r889;
	mov.b32 	%r934, %r889;
	mov.b32 	%r935, %r889;
	mov.b32 	%r936, %r889;
	mov.b32 	%r937, %r889;
	mov.b32 	%r938, %r889;
	mov.b32 	%r939, %r889;
	mov.b32 	%r940, %r889;
	mov.b32 	%r941, %r889;
	mov.b32 	%r942, %r889;
	mov.b32 	%r943, %r889;
	mov.b32 	%r944, %r889;
	mov.b32 	%r945, %r889;
	mov.b32 	%r946, %r889;
	mov.b32 	%r947, %r889;
	mov.b32 	%r948, %r889;
	mov.b32 	%r949, %r889;
	mov.b32 	%r950, %r889;
	mov.b32 	%r951, %r889;
	mov.b32 	%r952, %r889;
$L__BB0_3:                              // %__nv_exp2f.exit
                                        // =>This Inner Loop Header: Depth=1
	setp.lt.s64 	%p3, %rd228, %rd18;
	add.s32 	%r358, %r885, 1;
	setp.gt.s32 	%p4, %r358, 1;
	selp.b32 	%r885, 0, %r358, %p4;
	.loc	1 158 39                        // sk07_lm_head.py:158:39
	cvt.s64.s32 	%rd109, %r884;
	add.s64 	%rd93, %rd10, %rd109;
	add.s64 	%rd94, %rd11, %rd109;
	add.s64 	%rd95, %rd12, %rd109;
	add.s64 	%rd96, %rd13, %rd109;
	add.s64 	%rd97, %rd14, %rd109;
	add.s64 	%rd98, %rd15, %rd109;
	add.s64 	%rd99, %rd16, %rd109;
	add.s64 	%rd100, %rd17, %rd109;
	.loc	1 158 29                        // sk07_lm_head.py:158:29
	// begin inline asm
	mov.u16 %rs1, 0x0;
	ld.global.b8 { %rs1 }, [ %rd93 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs9, %rs1;
	// begin inline asm
	mov.u16 %rs2, 0x0;
	ld.global.b8 { %rs2 }, [ %rd94 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs10, %rs2;
	// begin inline asm
	mov.u16 %rs3, 0x0;
	ld.global.b8 { %rs3 }, [ %rd95 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs11, %rs3;
	// begin inline asm
	mov.u16 %rs4, 0x0;
	ld.global.b8 { %rs4 }, [ %rd96 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs12, %rs4;
	// begin inline asm
	mov.u16 %rs5, 0x0;
	ld.global.b8 { %rs5 }, [ %rd97 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs13, %rs5;
	// begin inline asm
	mov.u16 %rs6, 0x0;
	ld.global.b8 { %rs6 }, [ %rd98 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs14, %rs6;
	// begin inline asm
	mov.u16 %rs7, 0x0;
	ld.global.b8 { %rs7 }, [ %rd99 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs15, %rs7;
	// begin inline asm
	mov.u16 %rs8, 0x0;
	ld.global.b8 { %rs8 }, [ %rd100 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs16, %rs8;
	.loc	1 159 50                        // sk07_lm_head.py:159:50
	cvt.rn.f32.s16 	%r359, %rs9;
	cvt.rn.f32.s16 	%r360, %rs10;
	cvt.rn.f32.s16 	%r361, %rs11;
	cvt.rn.f32.s16 	%r362, %rs12;
	cvt.rn.f32.s16 	%r363, %rs13;
	cvt.rn.f32.s16 	%r364, %rs14;
	cvt.rn.f32.s16 	%r365, %rs15;
	cvt.rn.f32.s16 	%r366, %rs16;
	.loc	1 158 21                        // sk07_lm_head.py:158:21
	ex2.approx.ftz.f32 	%r367, %r359;
	ex2.approx.ftz.f32 	%r368, %r360;
	ex2.approx.ftz.f32 	%r369, %r361;
	ex2.approx.ftz.f32 	%r370, %r362;
	ex2.approx.ftz.f32 	%r371, %r363;
	ex2.approx.ftz.f32 	%r372, %r364;
	ex2.approx.ftz.f32 	%r373, %r365;
	ex2.approx.ftz.f32 	%r374, %r366;
	.loc	1 160 30                        // sk07_lm_head.py:160:30
	cp.async.wait_group 	2;
	bar.sync 	0;
	shl.b32 	%r375, %r885, 14;
	add.s32 	%r376, %r176, %r375;
	add.s32 	%r377, %r376, %r15;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r189, %r190, %r191, %r192}, [%r377];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r201, %r202, %r203, %r204}, [%r377+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r205, %r206, %r207, %r208}, [%r377+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r209, %r210, %r211, %r212}, [%r377+12288];
	add.s32 	%r378, %r376, %r16;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r217, %r218, %r219, %r220}, [%r378];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r245, %r246, %r247, %r248}, [%r378+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r265, %r266, %r267, %r268}, [%r378+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r285, %r286, %r287, %r288}, [%r378+12288];
	add.s32 	%r379, %r376, %r17;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r301, %r302, %r303, %r304}, [%r379];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r313, %r314, %r315, %r316}, [%r379+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r317, %r318, %r319, %r320}, [%r379+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r321, %r322, %r323, %r324}, [%r379+12288];
	add.s32 	%r380, %r376, %r18;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r325, %r326, %r327, %r328}, [%r380];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r337, %r338, %r339, %r340}, [%r380+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r341, %r342, %r343, %r344}, [%r380+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r345, %r346, %r347, %r348}, [%r380+12288];
	.loc	1 160 47                        // sk07_lm_head.py:160:47
	add.s32 	%r381, %r376, %r19;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r193, %r194, %r221, %r222}, [%r381+32768];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r195, %r196, %r227, %r228}, [%r381+36864];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r197, %r198, %r233, %r234}, [%r381+40960];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r199, %r200, %r239, %r240}, [%r381+45056];
	add.s32 	%r382, %r376, %r20;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r305, %r306, %r329, %r330}, [%r382+32768];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r307, %r308, %r331, %r332}, [%r382+36864];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r309, %r310, %r333, %r334}, [%r382+40960];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r311, %r312, %r335, %r336}, [%r382+45056];
	.loc	1 160 39                        // sk07_lm_head.py:160:39
	mov.b32 	%r213, %r188;
	mov.b32 	%r214, %r188;
	mov.b32 	%r215, %r188;
	mov.b32 	%r216, %r188;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r213, %r214, %r215, %r216 }, { %r189, %r190, %r191, %r192 }, { %r193, %r194 }, { %r213, %r214, %r215, %r216 };
	// end inline asm
	mov.b32 	%r223, %r188;
	mov.b32 	%r224, %r188;
	mov.b32 	%r225, %r188;
	mov.b32 	%r226, %r188;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r223, %r224, %r225, %r226 }, { %r189, %r190, %r191, %r192 }, { %r195, %r196 }, { %r223, %r224, %r225, %r226 };
	// end inline asm
	mov.b32 	%r229, %r188;
	mov.b32 	%r230, %r188;
	mov.b32 	%r231, %r188;
	mov.b32 	%r232, %r188;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r229, %r230, %r231, %r232 }, { %r189, %r190, %r191, %r192 }, { %r197, %r198 }, { %r229, %r230, %r231, %r232 };
	// end inline asm
	mov.b32 	%r235, %r188;
	mov.b32 	%r236, %r188;
	mov.b32 	%r237, %r188;
	mov.b32 	%r238, %r188;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r235, %r236, %r237, %r238 }, { %r189, %r190, %r191, %r192 }, { %r199, %r200 }, { %r235, %r236, %r237, %r238 };
	// end inline asm
	mov.b32 	%r241, %r188;
	mov.b32 	%r242, %r188;
	mov.b32 	%r243, %r188;
	mov.b32 	%r244, %r188;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r241, %r242, %r243, %r244 }, { %r201, %r202, %r203, %r204 }, { %r193, %r194 }, { %r241, %r242, %r243, %r244 };
	// end inline asm
	mov.b32 	%r249, %r188;
	mov.b32 	%r250, %r188;
	mov.b32 	%r251, %r188;
	mov.b32 	%r252, %r188;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r249, %r250, %r251, %r252 }, { %r201, %r202, %r203, %r204 }, { %r195, %r196 }, { %r249, %r250, %r251, %r252 };
	// end inline asm
	mov.b32 	%r253, %r188;
	mov.b32 	%r254, %r188;
	mov.b32 	%r255, %r188;
	mov.b32 	%r256, %r188;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r253, %r254, %r255, %r256 }, { %r201, %r202, %r203, %r204 }, { %r197, %r198 }, { %r253, %r254, %r255, %r256 };
	// end inline asm
	mov.b32 	%r257, %r188;
	mov.b32 	%r258, %r188;
	mov.b32 	%r259, %r188;
	mov.b32 	%r260, %r188;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r257, %r258, %r259, %r260 }, { %r201, %r202, %r203, %r204 }, { %r199, %r200 }, { %r257, %r258, %r259, %r260 };
	// end inline asm
	mov.b32 	%r261, %r188;
	mov.b32 	%r262, %r188;
	mov.b32 	%r263, %r188;
	mov.b32 	%r264, %r188;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r261, %r262, %r263, %r264 }, { %r205, %r206, %r207, %r208 }, { %r193, %r194 }, { %r261, %r262, %r263, %r264 };
	// end inline asm
	mov.b32 	%r269, %r188;
	mov.b32 	%r270, %r188;
	mov.b32 	%r271, %r188;
	mov.b32 	%r272, %r188;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r269, %r270, %r271, %r272 }, { %r205, %r206, %r207, %r208 }, { %r195, %r196 }, { %r269, %r270, %r271, %r272 };
	// end inline asm
	mov.b32 	%r273, %r188;
	mov.b32 	%r274, %r188;
	mov.b32 	%r275, %r188;
	mov.b32 	%r276, %r188;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r273, %r274, %r275, %r276 }, { %r205, %r206, %r207, %r208 }, { %r197, %r198 }, { %r273, %r274, %r275, %r276 };
	// end inline asm
	mov.b32 	%r277, %r188;
	mov.b32 	%r278, %r188;
	mov.b32 	%r279, %r188;
	mov.b32 	%r280, %r188;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r277, %r278, %r279, %r280 }, { %r205, %r206, %r207, %r208 }, { %r199, %r200 }, { %r277, %r278, %r279, %r280 };
	// end inline asm
	mov.b32 	%r281, %r188;
	mov.b32 	%r282, %r188;
	mov.b32 	%r283, %r188;
	mov.b32 	%r284, %r188;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r281, %r282, %r283, %r284 }, { %r209, %r210, %r211, %r212 }, { %r193, %r194 }, { %r281, %r282, %r283, %r284 };
	// end inline asm
	mov.b32 	%r289, %r188;
	mov.b32 	%r290, %r188;
	mov.b32 	%r291, %r188;
	mov.b32 	%r292, %r188;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r289, %r290, %r291, %r292 }, { %r209, %r210, %r211, %r212 }, { %r195, %r196 }, { %r289, %r290, %r291, %r292 };
	// end inline asm
	mov.b32 	%r293, %r188;
	mov.b32 	%r294, %r188;
	mov.b32 	%r295, %r188;
	mov.b32 	%r296, %r188;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r293, %r294, %r295, %r296 }, { %r209, %r210, %r211, %r212 }, { %r197, %r198 }, { %r293, %r294, %r295, %r296 };
	// end inline asm
	mov.b32 	%r300, %r188;
	mov.b32 	%r297, %r188;
	mov.b32 	%r298, %r188;
	mov.b32 	%r299, %r188;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r297, %r298, %r299, %r300 }, { %r209, %r210, %r211, %r212 }, { %r199, %r200 }, { %r297, %r298, %r299, %r300 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r213, %r214, %r215, %r216 }, { %r217, %r218, %r219, %r220 }, { %r221, %r222 }, { %r213, %r214, %r215, %r216 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r223, %r224, %r225, %r226 }, { %r217, %r218, %r219, %r220 }, { %r227, %r228 }, { %r223, %r224, %r225, %r226 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r229, %r230, %r231, %r232 }, { %r217, %r218, %r219, %r220 }, { %r233, %r234 }, { %r229, %r230, %r231, %r232 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r235, %r236, %r237, %r238 }, { %r217, %r218, %r219, %r220 }, { %r239, %r240 }, { %r235, %r236, %r237, %r238 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r241, %r242, %r243, %r244 }, { %r245, %r246, %r247, %r248 }, { %r221, %r222 }, { %r241, %r242, %r243, %r244 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r249, %r250, %r251, %r252 }, { %r245, %r246, %r247, %r248 }, { %r227, %r228 }, { %r249, %r250, %r251, %r252 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r253, %r254, %r255, %r256 }, { %r245, %r246, %r247, %r248 }, { %r233, %r234 }, { %r253, %r254, %r255, %r256 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r257, %r258, %r259, %r260 }, { %r245, %r246, %r247, %r248 }, { %r239, %r240 }, { %r257, %r258, %r259, %r260 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r261, %r262, %r263, %r264 }, { %r265, %r266, %r267, %r268 }, { %r221, %r222 }, { %r261, %r262, %r263, %r264 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r269, %r270, %r271, %r272 }, { %r265, %r266, %r267, %r268 }, { %r227, %r228 }, { %r269, %r270, %r271, %r272 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r273, %r274, %r275, %r276 }, { %r265, %r266, %r267, %r268 }, { %r233, %r234 }, { %r273, %r274, %r275, %r276 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r277, %r278, %r279, %r280 }, { %r265, %r266, %r267, %r268 }, { %r239, %r240 }, { %r277, %r278, %r279, %r280 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r281, %r282, %r283, %r284 }, { %r285, %r286, %r287, %r288 }, { %r221, %r222 }, { %r281, %r282, %r283, %r284 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r289, %r290, %r291, %r292 }, { %r285, %r286, %r287, %r288 }, { %r227, %r228 }, { %r289, %r290, %r291, %r292 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r293, %r294, %r295, %r296 }, { %r285, %r286, %r287, %r288 }, { %r233, %r234 }, { %r293, %r294, %r295, %r296 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r297, %r298, %r299, %r300 }, { %r285, %r286, %r287, %r288 }, { %r239, %r240 }, { %r297, %r298, %r299, %r300 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r213, %r214, %r215, %r216 }, { %r301, %r302, %r303, %r304 }, { %r305, %r306 }, { %r213, %r214, %r215, %r216 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r223, %r224, %r225, %r226 }, { %r301, %r302, %r303, %r304 }, { %r307, %r308 }, { %r223, %r224, %r225, %r226 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r229, %r230, %r231, %r232 }, { %r301, %r302, %r303, %r304 }, { %r309, %r310 }, { %r229, %r230, %r231, %r232 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r235, %r236, %r237, %r238 }, { %r301, %r302, %r303, %r304 }, { %r311, %r312 }, { %r235, %r236, %r237, %r238 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r241, %r242, %r243, %r244 }, { %r313, %r314, %r315, %r316 }, { %r305, %r306 }, { %r241, %r242, %r243, %r244 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r249, %r250, %r251, %r252 }, { %r313, %r314, %r315, %r316 }, { %r307, %r308 }, { %r249, %r250, %r251, %r252 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r253, %r254, %r255, %r256 }, { %r313, %r314, %r315, %r316 }, { %r309, %r310 }, { %r253, %r254, %r255, %r256 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r257, %r258, %r259, %r260 }, { %r313, %r314, %r315, %r316 }, { %r311, %r312 }, { %r257, %r258, %r259, %r260 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r261, %r262, %r263, %r264 }, { %r317, %r318, %r319, %r320 }, { %r305, %r306 }, { %r261, %r262, %r263, %r264 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r269, %r270, %r271, %r272 }, { %r317, %r318, %r319, %r320 }, { %r307, %r308 }, { %r269, %r270, %r271, %r272 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r273, %r274, %r275, %r276 }, { %r317, %r318, %r319, %r320 }, { %r309, %r310 }, { %r273, %r274, %r275, %r276 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r277, %r278, %r279, %r280 }, { %r317, %r318, %r319, %r320 }, { %r311, %r312 }, { %r277, %r278, %r279, %r280 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r281, %r282, %r283, %r284 }, { %r321, %r322, %r323, %r324 }, { %r305, %r306 }, { %r281, %r282, %r283, %r284 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r289, %r290, %r291, %r292 }, { %r321, %r322, %r323, %r324 }, { %r307, %r308 }, { %r289, %r290, %r291, %r292 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r293, %r294, %r295, %r296 }, { %r321, %r322, %r323, %r324 }, { %r309, %r310 }, { %r293, %r294, %r295, %r296 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r297, %r298, %r299, %r300 }, { %r321, %r322, %r323, %r324 }, { %r311, %r312 }, { %r297, %r298, %r299, %r300 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r213, %r214, %r215, %r216 }, { %r325, %r326, %r327, %r328 }, { %r329, %r330 }, { %r213, %r214, %r215, %r216 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r223, %r224, %r225, %r226 }, { %r325, %r326, %r327, %r328 }, { %r331, %r332 }, { %r223, %r224, %r225, %r226 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r229, %r230, %r231, %r232 }, { %r325, %r326, %r327, %r328 }, { %r333, %r334 }, { %r229, %r230, %r231, %r232 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r235, %r236, %r237, %r238 }, { %r325, %r326, %r327, %r328 }, { %r335, %r336 }, { %r235, %r236, %r237, %r238 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r241, %r242, %r243, %r244 }, { %r337, %r338, %r339, %r340 }, { %r329, %r330 }, { %r241, %r242, %r243, %r244 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r249, %r250, %r251, %r252 }, { %r337, %r338, %r339, %r340 }, { %r331, %r332 }, { %r249, %r250, %r251, %r252 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r253, %r254, %r255, %r256 }, { %r337, %r338, %r339, %r340 }, { %r333, %r334 }, { %r253, %r254, %r255, %r256 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r257, %r258, %r259, %r260 }, { %r337, %r338, %r339, %r340 }, { %r335, %r336 }, { %r257, %r258, %r259, %r260 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r261, %r262, %r263, %r264 }, { %r341, %r342, %r343, %r344 }, { %r329, %r330 }, { %r261, %r262, %r263, %r264 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r269, %r270, %r271, %r272 }, { %r341, %r342, %r343, %r344 }, { %r331, %r332 }, { %r269, %r270, %r271, %r272 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r273, %r274, %r275, %r276 }, { %r341, %r342, %r343, %r344 }, { %r333, %r334 }, { %r273, %r274, %r275, %r276 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r277, %r278, %r279, %r280 }, { %r341, %r342, %r343, %r344 }, { %r335, %r336 }, { %r277, %r278, %r279, %r280 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r281, %r282, %r283, %r284 }, { %r345, %r346, %r347, %r348 }, { %r329, %r330 }, { %r281, %r282, %r283, %r284 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r289, %r290, %r291, %r292 }, { %r345, %r346, %r347, %r348 }, { %r331, %r332 }, { %r289, %r290, %r291, %r292 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r293, %r294, %r295, %r296 }, { %r345, %r346, %r347, %r348 }, { %r333, %r334 }, { %r293, %r294, %r295, %r296 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r297, %r298, %r299, %r300 }, { %r345, %r346, %r347, %r348 }, { %r335, %r336 }, { %r297, %r298, %r299, %r300 };
	// end inline asm
	.loc	1 160 79                        // sk07_lm_head.py:160:79
	cvt.rn.f32.s32 	%r383, %r297;
	cvt.rn.f32.s32 	%r384, %r298;
	cvt.rn.f32.s32 	%r385, %r299;
	cvt.rn.f32.s32 	%r386, %r300;
	cvt.rn.f32.s32 	%r387, %r293;
	cvt.rn.f32.s32 	%r388, %r294;
	cvt.rn.f32.s32 	%r389, %r295;
	cvt.rn.f32.s32 	%r390, %r296;
	cvt.rn.f32.s32 	%r391, %r289;
	cvt.rn.f32.s32 	%r392, %r290;
	cvt.rn.f32.s32 	%r393, %r291;
	cvt.rn.f32.s32 	%r394, %r292;
	cvt.rn.f32.s32 	%r395, %r281;
	cvt.rn.f32.s32 	%r396, %r282;
	cvt.rn.f32.s32 	%r397, %r283;
	cvt.rn.f32.s32 	%r398, %r284;
	cvt.rn.f32.s32 	%r399, %r277;
	cvt.rn.f32.s32 	%r400, %r278;
	cvt.rn.f32.s32 	%r401, %r279;
	cvt.rn.f32.s32 	%r402, %r280;
	cvt.rn.f32.s32 	%r403, %r273;
	cvt.rn.f32.s32 	%r404, %r274;
	cvt.rn.f32.s32 	%r405, %r275;
	cvt.rn.f32.s32 	%r406, %r276;
	cvt.rn.f32.s32 	%r407, %r269;
	cvt.rn.f32.s32 	%r408, %r270;
	cvt.rn.f32.s32 	%r409, %r271;
	cvt.rn.f32.s32 	%r410, %r272;
	cvt.rn.f32.s32 	%r411, %r261;
	cvt.rn.f32.s32 	%r412, %r262;
	cvt.rn.f32.s32 	%r413, %r263;
	cvt.rn.f32.s32 	%r414, %r264;
	cvt.rn.f32.s32 	%r415, %r257;
	cvt.rn.f32.s32 	%r416, %r258;
	cvt.rn.f32.s32 	%r417, %r259;
	cvt.rn.f32.s32 	%r418, %r260;
	cvt.rn.f32.s32 	%r419, %r253;
	cvt.rn.f32.s32 	%r420, %r254;
	cvt.rn.f32.s32 	%r421, %r255;
	cvt.rn.f32.s32 	%r422, %r256;
	cvt.rn.f32.s32 	%r423, %r249;
	cvt.rn.f32.s32 	%r424, %r250;
	cvt.rn.f32.s32 	%r425, %r251;
	cvt.rn.f32.s32 	%r426, %r252;
	cvt.rn.f32.s32 	%r427, %r241;
	cvt.rn.f32.s32 	%r428, %r242;
	cvt.rn.f32.s32 	%r429, %r243;
	cvt.rn.f32.s32 	%r430, %r244;
	cvt.rn.f32.s32 	%r431, %r235;
	cvt.rn.f32.s32 	%r432, %r236;
	cvt.rn.f32.s32 	%r433, %r237;
	cvt.rn.f32.s32 	%r434, %r238;
	cvt.rn.f32.s32 	%r435, %r229;
	cvt.rn.f32.s32 	%r436, %r230;
	cvt.rn.f32.s32 	%r437, %r231;
	cvt.rn.f32.s32 	%r438, %r232;
	cvt.rn.f32.s32 	%r439, %r223;
	cvt.rn.f32.s32 	%r440, %r224;
	cvt.rn.f32.s32 	%r441, %r225;
	cvt.rn.f32.s32 	%r442, %r226;
	cvt.rn.f32.s32 	%r443, %r213;
	cvt.rn.f32.s32 	%r444, %r214;
	cvt.rn.f32.s32 	%r445, %r215;
	cvt.rn.f32.s32 	%r446, %r216;
	.loc	1 160 15                        // sk07_lm_head.py:160:15
	fma.rn.f32 	%r892, %r368, %r446, %r892;
	fma.rn.f32 	%r891, %r367, %r445, %r891;
	fma.rn.f32 	%r890, %r368, %r444, %r890;
	fma.rn.f32 	%r889, %r367, %r443, %r889;
	fma.rn.f32 	%r896, %r370, %r442, %r896;
	fma.rn.f32 	%r895, %r369, %r441, %r895;
	fma.rn.f32 	%r894, %r370, %r440, %r894;
	fma.rn.f32 	%r893, %r369, %r439, %r893;
	fma.rn.f32 	%r900, %r372, %r438, %r900;
	fma.rn.f32 	%r899, %r371, %r437, %r899;
	fma.rn.f32 	%r898, %r372, %r436, %r898;
	fma.rn.f32 	%r897, %r371, %r435, %r897;
	fma.rn.f32 	%r904, %r374, %r434, %r904;
	fma.rn.f32 	%r903, %r373, %r433, %r903;
	fma.rn.f32 	%r902, %r374, %r432, %r902;
	fma.rn.f32 	%r901, %r373, %r431, %r901;
	fma.rn.f32 	%r908, %r368, %r430, %r908;
	fma.rn.f32 	%r907, %r367, %r429, %r907;
	fma.rn.f32 	%r906, %r368, %r428, %r906;
	fma.rn.f32 	%r905, %r367, %r427, %r905;
	fma.rn.f32 	%r912, %r370, %r426, %r912;
	fma.rn.f32 	%r911, %r369, %r425, %r911;
	fma.rn.f32 	%r910, %r370, %r424, %r910;
	fma.rn.f32 	%r909, %r369, %r423, %r909;
	fma.rn.f32 	%r916, %r372, %r422, %r916;
	fma.rn.f32 	%r915, %r371, %r421, %r915;
	fma.rn.f32 	%r914, %r372, %r420, %r914;
	fma.rn.f32 	%r913, %r371, %r419, %r913;
	fma.rn.f32 	%r920, %r374, %r418, %r920;
	fma.rn.f32 	%r919, %r373, %r417, %r919;
	fma.rn.f32 	%r918, %r374, %r416, %r918;
	fma.rn.f32 	%r917, %r373, %r415, %r917;
	fma.rn.f32 	%r924, %r368, %r414, %r924;
	fma.rn.f32 	%r923, %r367, %r413, %r923;
	fma.rn.f32 	%r922, %r368, %r412, %r922;
	fma.rn.f32 	%r921, %r367, %r411, %r921;
	fma.rn.f32 	%r928, %r370, %r410, %r928;
	fma.rn.f32 	%r927, %r369, %r409, %r927;
	fma.rn.f32 	%r926, %r370, %r408, %r926;
	fma.rn.f32 	%r925, %r369, %r407, %r925;
	fma.rn.f32 	%r932, %r372, %r406, %r932;
	fma.rn.f32 	%r931, %r371, %r405, %r931;
	fma.rn.f32 	%r930, %r372, %r404, %r930;
	fma.rn.f32 	%r929, %r371, %r403, %r929;
	fma.rn.f32 	%r936, %r374, %r402, %r936;
	fma.rn.f32 	%r935, %r373, %r401, %r935;
	fma.rn.f32 	%r934, %r374, %r400, %r934;
	fma.rn.f32 	%r933, %r373, %r399, %r933;
	fma.rn.f32 	%r940, %r368, %r398, %r940;
	fma.rn.f32 	%r939, %r367, %r397, %r939;
	fma.rn.f32 	%r938, %r368, %r396, %r938;
	fma.rn.f32 	%r937, %r367, %r395, %r937;
	fma.rn.f32 	%r944, %r370, %r394, %r944;
	fma.rn.f32 	%r943, %r369, %r393, %r943;
	fma.rn.f32 	%r942, %r370, %r392, %r942;
	fma.rn.f32 	%r941, %r369, %r391, %r941;
	fma.rn.f32 	%r948, %r372, %r390, %r948;
	fma.rn.f32 	%r947, %r371, %r389, %r947;
	fma.rn.f32 	%r946, %r372, %r388, %r946;
	fma.rn.f32 	%r945, %r371, %r387, %r945;
	fma.rn.f32 	%r952, %r374, %r386, %r952;
	fma.rn.f32 	%r951, %r373, %r385, %r951;
	fma.rn.f32 	%r950, %r374, %r384, %r950;
	fma.rn.f32 	%r949, %r373, %r383, %r949;
	.loc	1 161 18                        // sk07_lm_head.py:161:18
	add.s64 	%rd101, %rd27, %rd227;
	add.s64 	%rd102, %rd26, %rd227;
	add.s64 	%rd103, %rd25, %rd227;
	.loc	1 162 18                        // sk07_lm_head.py:162:18
	add.s64 	%rd104, %rd24, %rd227;
	add.s64 	%rd105, %rd23, %rd227;
	add.s64 	%rd106, %rd22, %rd227;
	add.s64 	%rd107, %rd21, %rd227;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	add.s64 	%rd108, %rd20, %rd227;
	add.s32 	%r447, %r886, 1;
	setp.gt.s32 	%p5, %r447, 1;
	selp.b32 	%r886, 0, %r447, %p5;
	.loc	1 160 30                        // sk07_lm_head.py:160:30
	shl.b32 	%r448, %r886, 14;
	bar.sync 	0;
	add.s32 	%r349, %r55, %r448;
	selp.b32 	%r350, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r349 + 0 ], [ %rd101 + 0 ], 0x10, %r350;
	// end inline asm
	add.s32 	%r351, %r349, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r351 + 0 ], [ %rd102 + 0 ], 0x10, %r350;
	// end inline asm
	add.s32 	%r352, %r349, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r352 + 0 ], [ %rd103 + 0 ], 0x10, %r350;
	// end inline asm
	add.s32 	%r353, %r349, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r353 + 0 ], [ %rd104 + 0 ], 0x10, %r350;
	// end inline asm
	cp.async.commit_group;
	.loc	1 160 47                        // sk07_lm_head.py:160:47
	add.s32 	%r354, %r349, 32768;
	// begin inline asm
	cp.async.cg.shared.global [ %r354 + 0 ], [ %rd105 + 0 ], 0x10, %r350;
	// end inline asm
	add.s32 	%r355, %r349, 36864;
	// begin inline asm
	cp.async.cg.shared.global [ %r355 + 0 ], [ %rd106 + 0 ], 0x10, %r350;
	// end inline asm
	add.s32 	%r356, %r349, 40960;
	// begin inline asm
	cp.async.cg.shared.global [ %r356 + 0 ], [ %rd107 + 0 ], 0x10, %r350;
	// end inline asm
	add.s32 	%r357, %r349, 45056;
	// begin inline asm
	cp.async.cg.shared.global [ %r357 + 0 ], [ %rd108 + 0 ], 0x10, %r350;
	// end inline asm
	cp.async.commit_group;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	add.s64 	%rd228, %rd228, 1;
	add.s64 	%rd227, %rd227, 128;
	add.s32 	%r884, %r884, %r26;
	setp.ne.b64 	%p6, %rd19, %rd227;
	@%p6 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_4;
$L__BB0_1:                              // %.._crit_edge_crit_edge
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	shl.b32 	%r888, %r2, 1;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	and.b32 	%r887, %r2, 16;
	mov.b32 	%r889, 0f00000000;
	mov.b32 	%r890, %r889;
	mov.b32 	%r891, %r889;
	mov.b32 	%r892, %r889;
	mov.b32 	%r893, %r889;
	mov.b32 	%r894, %r889;
	mov.b32 	%r895, %r889;
	mov.b32 	%r896, %r889;
	mov.b32 	%r897, %r889;
	mov.b32 	%r898, %r889;
	mov.b32 	%r899, %r889;
	mov.b32 	%r900, %r889;
	mov.b32 	%r901, %r889;
	mov.b32 	%r902, %r889;
	mov.b32 	%r903, %r889;
	mov.b32 	%r904, %r889;
	mov.b32 	%r905, %r889;
	mov.b32 	%r906, %r889;
	mov.b32 	%r907, %r889;
	mov.b32 	%r908, %r889;
	mov.b32 	%r909, %r889;
	mov.b32 	%r910, %r889;
	mov.b32 	%r911, %r889;
	mov.b32 	%r912, %r889;
	mov.b32 	%r913, %r889;
	mov.b32 	%r914, %r889;
	mov.b32 	%r915, %r889;
	mov.b32 	%r916, %r889;
	mov.b32 	%r917, %r889;
	mov.b32 	%r918, %r889;
	mov.b32 	%r919, %r889;
	mov.b32 	%r920, %r889;
	mov.b32 	%r921, %r889;
	mov.b32 	%r922, %r889;
	mov.b32 	%r923, %r889;
	mov.b32 	%r924, %r889;
	mov.b32 	%r925, %r889;
	mov.b32 	%r926, %r889;
	mov.b32 	%r927, %r889;
	mov.b32 	%r928, %r889;
	mov.b32 	%r929, %r889;
	mov.b32 	%r930, %r889;
	mov.b32 	%r931, %r889;
	mov.b32 	%r932, %r889;
	mov.b32 	%r933, %r889;
	mov.b32 	%r934, %r889;
	mov.b32 	%r935, %r889;
	mov.b32 	%r936, %r889;
	mov.b32 	%r937, %r889;
	mov.b32 	%r938, %r889;
	mov.b32 	%r939, %r889;
	mov.b32 	%r940, %r889;
	mov.b32 	%r941, %r889;
	mov.b32 	%r942, %r889;
	mov.b32 	%r943, %r889;
	mov.b32 	%r944, %r889;
	mov.b32 	%r945, %r889;
	mov.b32 	%r946, %r889;
	mov.b32 	%r947, %r889;
	mov.b32 	%r948, %r889;
	mov.b32 	%r949, %r889;
	mov.b32 	%r950, %r889;
	mov.b32 	%r951, %r889;
	mov.b32 	%r952, %r889;
$L__BB0_4:                              // %._crit_edge
	.loc	1 141 45                        // sk07_lm_head.py:141:45
	shl.b32 	%r571, %r7, 3;
	.loc	1 142 32                        // sk07_lm_head.py:142:32
	or.b32 	%r572, %r12, %r571;
	.loc	1 141 45                        // sk07_lm_head.py:141:45
	and.b32 	%r573, %r2, 128;
	shr.u32 	%r574, %r573, 3;
	shr.u32 	%r575, %r2, 2;
	bfe.u32 	%r576, %r2, 2, 3;
	or.b32 	%r577, %r574, %r576;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r578, %r577, %r1;
	or.b32 	%r579, %r578, 104;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r580, %r579, %r21;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r581, %r578, 96;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r582, %r581, %r21;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r583, %r578, 72;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r584, %r583, %r21;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r585, %r578, 64;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r586, %r585, %r21;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r587, %r578, 40;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r588, %r587, %r21;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r589, %r578, 32;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r590, %r589, %r21;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r591, %r578, 8;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r592, %r591, %r21;
	rem.s32 	%r593, %r578, %r21;
	.loc	1 141 45                        // sk07_lm_head.py:141:45
	shr.u32 	%r594, %r2, 4;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r595, %r594, %r1;
	or.b32 	%r596, %r595, 112;
	.loc	1 141 45                        // sk07_lm_head.py:141:45
	bfe.u32 	%r597, %r2, 4, 4;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r598, %r597, %r1;
	or.b32 	%r599, %r598, 96;
	or.b32 	%r600, %r598, 80;
	or.b32 	%r601, %r598, 64;
	or.b32 	%r602, %r595, 48;
	or.b32 	%r603, %r598, 32;
	or.b32 	%r604, %r598, 16;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 164 38                        // sk07_lm_head.py:164:38
	mad.wide.s32 	%rd110, %r593, 4, %rd32;
	mad.wide.s32 	%rd111, %r592, 4, %rd32;
	mad.wide.s32 	%rd112, %r590, 4, %rd32;
	mad.wide.s32 	%rd113, %r588, 4, %rd32;
	mad.wide.s32 	%rd114, %r586, 4, %rd32;
	mad.wide.s32 	%rd115, %r584, 4, %rd32;
	mad.wide.s32 	%rd116, %r582, 4, %rd32;
	mad.wide.s32 	%rd117, %r580, 4, %rd32;
	.loc	1 164 24                        // sk07_lm_head.py:164:24
	// begin inline asm
	mov.u32 %r449, 0x0;
	ld.global.b32 { %r449 }, [ %rd110 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r450, 0x0;
	ld.global.b32 { %r450 }, [ %rd111 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r451, 0x0;
	ld.global.b32 { %r451 }, [ %rd112 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r452, 0x0;
	ld.global.b32 { %r452 }, [ %rd113 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r453, 0x0;
	ld.global.b32 { %r453 }, [ %rd114 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r454, 0x0;
	ld.global.b32 { %r454 }, [ %rd115 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r455, 0x0;
	ld.global.b32 { %r455 }, [ %rd116 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r456, 0x0;
	ld.global.b32 { %r456 }, [ %rd117 + 0 ];
	// end inline asm
	.loc	1 165 38                        // sk07_lm_head.py:165:38
	mad.wide.s32 	%rd118, %r31, 4, %rd33;
	mad.wide.s32 	%rd119, %r32, 4, %rd33;
	mad.wide.s32 	%rd120, %r33, 4, %rd33;
	mad.wide.s32 	%rd121, %r34, 4, %rd33;
	mad.wide.s32 	%rd122, %r35, 4, %rd33;
	mad.wide.s32 	%rd123, %r36, 4, %rd33;
	mad.wide.s32 	%rd124, %r37, 4, %rd33;
	mad.wide.s32 	%rd125, %r38, 4, %rd33;
	.loc	1 165 24                        // sk07_lm_head.py:165:24
	// begin inline asm
	mov.u32 %r457, 0x0;
	ld.global.b32 { %r457 }, [ %rd118 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r458, 0x0;
	ld.global.b32 { %r458 }, [ %rd119 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r459, 0x0;
	ld.global.b32 { %r459 }, [ %rd120 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r460, 0x0;
	ld.global.b32 { %r460 }, [ %rd121 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r461, 0x0;
	ld.global.b32 { %r461 }, [ %rd122 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r462, 0x0;
	ld.global.b32 { %r462 }, [ %rd123 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r463, 0x0;
	ld.global.b32 { %r463 }, [ %rd124 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r464, 0x0;
	ld.global.b32 { %r464 }, [ %rd125 + 0 ];
	// end inline asm
	.loc	1 166 49                        // sk07_lm_head.py:166:49
	mul.lo.s32 	%r605, %r8, %r25;
	mul.lo.s32 	%r606, %r9, %r25;
	mul.lo.s32 	%r607, %r10, %r25;
	mul.lo.s32 	%r608, %r11, %r25;
	.loc	1 166 31                        // sk07_lm_head.py:166:31
	mad.wide.s32 	%rd198, %r605, 2, %rd31;
	mad.wide.s32 	%rd199, %r606, 2, %rd31;
	mad.wide.s32 	%rd200, %r607, 2, %rd31;
	mad.wide.s32 	%rd201, %r608, 2, %rd31;
	.loc	1 166 64                        // sk07_lm_head.py:166:64
	mul.wide.s32 	%rd202, %r39, 2;
	add.s64 	%rd126, %rd198, %rd202;
	mul.wide.s32 	%rd203, %r40, 2;
	add.s64 	%rd127, %rd198, %rd203;
	mul.wide.s32 	%rd204, %r41, 2;
	add.s64 	%rd128, %rd198, %rd204;
	mul.wide.s32 	%rd205, %r42, 2;
	add.s64 	%rd129, %rd198, %rd205;
	mul.wide.s32 	%rd206, %r43, 2;
	add.s64 	%rd130, %rd198, %rd206;
	mul.wide.s32 	%rd207, %r44, 2;
	add.s64 	%rd131, %rd198, %rd207;
	mul.wide.s32 	%rd208, %r45, 2;
	add.s64 	%rd132, %rd198, %rd208;
	mul.wide.s32 	%rd209, %r46, 2;
	add.s64 	%rd133, %rd198, %rd209;
	mul.wide.s32 	%rd210, %r47, 2;
	add.s64 	%rd134, %rd198, %rd210;
	mul.wide.s32 	%rd211, %r48, 2;
	add.s64 	%rd135, %rd198, %rd211;
	mul.wide.s32 	%rd212, %r49, 2;
	add.s64 	%rd136, %rd198, %rd212;
	mul.wide.s32 	%rd213, %r50, 2;
	add.s64 	%rd137, %rd198, %rd213;
	mul.wide.s32 	%rd214, %r51, 2;
	add.s64 	%rd138, %rd198, %rd214;
	mul.wide.s32 	%rd215, %r52, 2;
	add.s64 	%rd139, %rd198, %rd215;
	mul.wide.s32 	%rd216, %r53, 2;
	add.s64 	%rd140, %rd198, %rd216;
	mul.wide.s32 	%rd217, %r54, 2;
	add.s64 	%rd141, %rd198, %rd217;
	add.s64 	%rd142, %rd199, %rd202;
	add.s64 	%rd143, %rd199, %rd203;
	add.s64 	%rd144, %rd199, %rd204;
	add.s64 	%rd145, %rd199, %rd205;
	add.s64 	%rd146, %rd199, %rd206;
	add.s64 	%rd147, %rd199, %rd207;
	add.s64 	%rd148, %rd199, %rd208;
	add.s64 	%rd149, %rd199, %rd209;
	add.s64 	%rd150, %rd199, %rd210;
	add.s64 	%rd151, %rd199, %rd211;
	add.s64 	%rd152, %rd199, %rd212;
	add.s64 	%rd153, %rd199, %rd213;
	add.s64 	%rd154, %rd199, %rd214;
	add.s64 	%rd155, %rd199, %rd215;
	add.s64 	%rd156, %rd199, %rd216;
	add.s64 	%rd157, %rd199, %rd217;
	add.s64 	%rd158, %rd200, %rd202;
	add.s64 	%rd159, %rd200, %rd203;
	add.s64 	%rd160, %rd200, %rd204;
	add.s64 	%rd161, %rd200, %rd205;
	add.s64 	%rd162, %rd200, %rd206;
	add.s64 	%rd163, %rd200, %rd207;
	add.s64 	%rd164, %rd200, %rd208;
	add.s64 	%rd165, %rd200, %rd209;
	add.s64 	%rd166, %rd200, %rd210;
	add.s64 	%rd167, %rd200, %rd211;
	add.s64 	%rd168, %rd200, %rd212;
	add.s64 	%rd169, %rd200, %rd213;
	add.s64 	%rd170, %rd200, %rd214;
	add.s64 	%rd171, %rd200, %rd215;
	add.s64 	%rd172, %rd200, %rd216;
	add.s64 	%rd173, %rd200, %rd217;
	add.s64 	%rd174, %rd201, %rd202;
	add.s64 	%rd175, %rd201, %rd203;
	add.s64 	%rd176, %rd201, %rd204;
	add.s64 	%rd177, %rd201, %rd205;
	add.s64 	%rd178, %rd201, %rd206;
	add.s64 	%rd179, %rd201, %rd207;
	add.s64 	%rd180, %rd201, %rd208;
	add.s64 	%rd181, %rd201, %rd209;
	add.s64 	%rd182, %rd201, %rd210;
	add.s64 	%rd183, %rd201, %rd211;
	add.s64 	%rd184, %rd201, %rd212;
	add.s64 	%rd185, %rd201, %rd213;
	add.s64 	%rd186, %rd201, %rd214;
	add.s64 	%rd187, %rd201, %rd215;
	add.s64 	%rd188, %rd201, %rd216;
	add.s64 	%rd189, %rd201, %rd217;
	.loc	1 166 19                        // sk07_lm_head.py:166:19
	// begin inline asm
	mov.u16 %rs17, 0x0;
	ld.global.b16 { %rs17 }, [ %rd126 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs18, 0x0;
	ld.global.b16 { %rs18 }, [ %rd127 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs19, 0x0;
	ld.global.b16 { %rs19 }, [ %rd128 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs20, 0x0;
	ld.global.b16 { %rs20 }, [ %rd129 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs21, 0x0;
	ld.global.b16 { %rs21 }, [ %rd130 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs22, 0x0;
	ld.global.b16 { %rs22 }, [ %rd131 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs23, 0x0;
	ld.global.b16 { %rs23 }, [ %rd132 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs24, 0x0;
	ld.global.b16 { %rs24 }, [ %rd133 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs25, 0x0;
	ld.global.b16 { %rs25 }, [ %rd134 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs26, 0x0;
	ld.global.b16 { %rs26 }, [ %rd135 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs27, 0x0;
	ld.global.b16 { %rs27 }, [ %rd136 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs28, 0x0;
	ld.global.b16 { %rs28 }, [ %rd137 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs29, 0x0;
	ld.global.b16 { %rs29 }, [ %rd138 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs30, 0x0;
	ld.global.b16 { %rs30 }, [ %rd139 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs31, 0x0;
	ld.global.b16 { %rs31 }, [ %rd140 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs32, 0x0;
	ld.global.b16 { %rs32 }, [ %rd141 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs33, 0x0;
	ld.global.b16 { %rs33 }, [ %rd142 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs34, 0x0;
	ld.global.b16 { %rs34 }, [ %rd143 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs35, 0x0;
	ld.global.b16 { %rs35 }, [ %rd144 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs36, 0x0;
	ld.global.b16 { %rs36 }, [ %rd145 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs37, 0x0;
	ld.global.b16 { %rs37 }, [ %rd146 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs38, 0x0;
	ld.global.b16 { %rs38 }, [ %rd147 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs39, 0x0;
	ld.global.b16 { %rs39 }, [ %rd148 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs40, 0x0;
	ld.global.b16 { %rs40 }, [ %rd149 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs41, 0x0;
	ld.global.b16 { %rs41 }, [ %rd150 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs42, 0x0;
	ld.global.b16 { %rs42 }, [ %rd151 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs43, 0x0;
	ld.global.b16 { %rs43 }, [ %rd152 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs44, 0x0;
	ld.global.b16 { %rs44 }, [ %rd153 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs45, 0x0;
	ld.global.b16 { %rs45 }, [ %rd154 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs46, 0x0;
	ld.global.b16 { %rs46 }, [ %rd155 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs47, 0x0;
	ld.global.b16 { %rs47 }, [ %rd156 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs48, 0x0;
	ld.global.b16 { %rs48 }, [ %rd157 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs49, 0x0;
	ld.global.b16 { %rs49 }, [ %rd158 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs50, 0x0;
	ld.global.b16 { %rs50 }, [ %rd159 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs51, 0x0;
	ld.global.b16 { %rs51 }, [ %rd160 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs52, 0x0;
	ld.global.b16 { %rs52 }, [ %rd161 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs53, 0x0;
	ld.global.b16 { %rs53 }, [ %rd162 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs54, 0x0;
	ld.global.b16 { %rs54 }, [ %rd163 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs55, 0x0;
	ld.global.b16 { %rs55 }, [ %rd164 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs56, 0x0;
	ld.global.b16 { %rs56 }, [ %rd165 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs57, 0x0;
	ld.global.b16 { %rs57 }, [ %rd166 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs58, 0x0;
	ld.global.b16 { %rs58 }, [ %rd167 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs59, 0x0;
	ld.global.b16 { %rs59 }, [ %rd168 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs60, 0x0;
	ld.global.b16 { %rs60 }, [ %rd169 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs61, 0x0;
	ld.global.b16 { %rs61 }, [ %rd170 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs62, 0x0;
	ld.global.b16 { %rs62 }, [ %rd171 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs63, 0x0;
	ld.global.b16 { %rs63 }, [ %rd172 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs64, 0x0;
	ld.global.b16 { %rs64 }, [ %rd173 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs65, 0x0;
	ld.global.b16 { %rs65 }, [ %rd174 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs66, 0x0;
	ld.global.b16 { %rs66 }, [ %rd175 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs67, 0x0;
	ld.global.b16 { %rs67 }, [ %rd176 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs68, 0x0;
	ld.global.b16 { %rs68 }, [ %rd177 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs69, 0x0;
	ld.global.b16 { %rs69 }, [ %rd178 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs70, 0x0;
	ld.global.b16 { %rs70 }, [ %rd179 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs71, 0x0;
	ld.global.b16 { %rs71 }, [ %rd180 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs72, 0x0;
	ld.global.b16 { %rs72 }, [ %rd181 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs73, 0x0;
	ld.global.b16 { %rs73 }, [ %rd182 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs74, 0x0;
	ld.global.b16 { %rs74 }, [ %rd183 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs75, 0x0;
	ld.global.b16 { %rs75 }, [ %rd184 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs76, 0x0;
	ld.global.b16 { %rs76 }, [ %rd185 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs77, 0x0;
	ld.global.b16 { %rs77 }, [ %rd186 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs78, 0x0;
	ld.global.b16 { %rs78 }, [ %rd187 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs79, 0x0;
	ld.global.b16 { %rs79 }, [ %rd188 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs80, 0x0;
	ld.global.b16 { %rs80 }, [ %rd189 + 0 ];
	// end inline asm
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	shl.b32 	%r609, %r14, 7;
	shl.b32 	%r610, %r3, 1;
	or.b32 	%r611, %r609, %r883;
	xor.b32 	%r612, %r611, %r610;
	add.s32 	%r465, %r176, %r612;
	mov.b32 	%r466, {%rs17, %rs18};
	mov.b32 	%r467, {%rs19, %rs20};
	mov.b32 	%r468, {%rs21, %rs22};
	mov.b32 	%r469, {%rs23, %rs24};
	// begin inline asm
	st.shared.v4.b32 [ %r465 + 0 ], { %r466, %r467, %r468, %r469 };
	// end inline asm
	add.s32 	%r470, %r465, 512;
	mov.b32 	%r471, {%rs25, %rs26};
	mov.b32 	%r472, {%rs27, %rs28};
	mov.b32 	%r473, {%rs29, %rs30};
	mov.b32 	%r474, {%rs31, %rs32};
	// begin inline asm
	st.shared.v4.b32 [ %r470 + 0 ], { %r471, %r472, %r473, %r474 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r613, %r6, 10;
	and.b32 	%r614, %r13, 752;
	and.b32 	%r615, %r888, 288;
	and.b32 	%r616, %r575, 16;
	xor.b32 	%r617, %r614, %r615;
	xor.b32 	%r618, %r617, %r616;
	or.b32 	%r619, %r618, %r613;
	add.s32 	%r620, %r176, %r619;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r621, %r622, %r623, %r624}, [%r620];
	xor.b32 	%r625, %r619, 64;
	add.s32 	%r626, %r176, %r625;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r627, %r628, %r629, %r630}, [%r626];
	bar.sync 	0;
	mov.b32 	%r475, {%rs33, %rs34};
	mov.b32 	%r476, {%rs35, %rs36};
	mov.b32 	%r477, {%rs37, %rs38};
	mov.b32 	%r478, {%rs39, %rs40};
	// begin inline asm
	st.shared.v4.b32 [ %r465 + 0 ], { %r475, %r476, %r477, %r478 };
	// end inline asm
	mov.b32 	%r479, {%rs41, %rs42};
	mov.b32 	%r480, {%rs43, %rs44};
	mov.b32 	%r481, {%rs45, %rs46};
	mov.b32 	%r482, {%rs47, %rs48};
	// begin inline asm
	st.shared.v4.b32 [ %r470 + 0 ], { %r479, %r480, %r481, %r482 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r631, %r632, %r633, %r634}, [%r620];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r635, %r636, %r637, %r638}, [%r626];
	bar.sync 	0;
	mov.b32 	%r483, {%rs49, %rs50};
	mov.b32 	%r484, {%rs51, %rs52};
	mov.b32 	%r485, {%rs53, %rs54};
	mov.b32 	%r486, {%rs55, %rs56};
	// begin inline asm
	st.shared.v4.b32 [ %r465 + 0 ], { %r483, %r484, %r485, %r486 };
	// end inline asm
	mov.b32 	%r487, {%rs57, %rs58};
	mov.b32 	%r488, {%rs59, %rs60};
	mov.b32 	%r489, {%rs61, %rs62};
	mov.b32 	%r490, {%rs63, %rs64};
	// begin inline asm
	st.shared.v4.b32 [ %r470 + 0 ], { %r487, %r488, %r489, %r490 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r639, %r640, %r641, %r642}, [%r620];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r643, %r644, %r645, %r646}, [%r626];
	bar.sync 	0;
	mov.b32 	%r491, {%rs65, %rs66};
	mov.b32 	%r492, {%rs67, %rs68};
	mov.b32 	%r493, {%rs69, %rs70};
	mov.b32 	%r494, {%rs71, %rs72};
	// begin inline asm
	st.shared.v4.b32 [ %r465 + 0 ], { %r491, %r492, %r493, %r494 };
	// end inline asm
	mov.b32 	%r495, {%rs73, %rs74};
	mov.b32 	%r496, {%rs75, %rs76};
	mov.b32 	%r497, {%rs77, %rs78};
	mov.b32 	%r498, {%rs79, %rs80};
	// begin inline asm
	st.shared.v4.b32 [ %r470 + 0 ], { %r495, %r496, %r497, %r498 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r647, %r648, %r649, %r650}, [%r620];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r651, %r652, %r653, %r654}, [%r626];
	.loc	1 173 31                        // sk07_lm_head.py:173:31
	setp.lt.s32 	%p15, %r598, %r21;
	setp.lt.s32 	%p16, %r604, %r21;
	setp.lt.s32 	%p17, %r603, %r21;
	setp.lt.s32 	%p18, %r602, %r21;
	setp.lt.s32 	%p19, %r601, %r21;
	setp.lt.s32 	%p20, %r600, %r21;
	setp.lt.s32 	%p21, %r599, %r21;
	setp.lt.s32 	%p22, %r596, %r21;
	.loc	1 173 54                        // sk07_lm_head.py:173:54
	setp.lt.s32 	%p23, %r572, %r22;
	.loc	1 173 37                        // sk07_lm_head.py:173:37
	and.pred 	%p7, %p15, %p23;
	and.pred 	%p8, %p16, %p23;
	and.pred 	%p9, %p17, %p23;
	and.pred 	%p10, %p18, %p23;
	and.pred 	%p11, %p19, %p23;
	and.pred 	%p12, %p20, %p23;
	and.pred 	%p13, %p21, %p23;
	and.pred 	%p14, %p22, %p23;
	.loc	1 171 35                        // sk07_lm_head.py:171:35
	mul.lo.s32 	%r655, %r598, %r24;
	mul.lo.s32 	%r656, %r604, %r24;
	mul.lo.s32 	%r657, %r603, %r24;
	mul.lo.s32 	%r658, %r602, %r24;
	mul.lo.s32 	%r659, %r601, %r24;
	mul.lo.s32 	%r660, %r600, %r24;
	mul.lo.s32 	%r661, %r599, %r24;
	mul.lo.s32 	%r662, %r596, %r24;
	.loc	1 171 18                        // sk07_lm_head.py:171:18
	mad.wide.s32 	%rd218, %r655, 2, %rd30;
	mad.wide.s32 	%rd219, %r656, 2, %rd30;
	mad.wide.s32 	%rd220, %r657, 2, %rd30;
	mad.wide.s32 	%rd221, %r658, 2, %rd30;
	mad.wide.s32 	%rd222, %r659, 2, %rd30;
	mad.wide.s32 	%rd223, %r660, 2, %rd30;
	mad.wide.s32 	%rd224, %r661, 2, %rd30;
	mad.wide.s32 	%rd225, %r662, 2, %rd30;
	.loc	1 171 50                        // sk07_lm_head.py:171:50
	mul.wide.s32 	%rd226, %r572, 2;
	add.s64 	%rd190, %rd218, %rd226;
	add.s64 	%rd191, %rd219, %rd226;
	add.s64 	%rd192, %rd220, %rd226;
	add.s64 	%rd193, %rd221, %rd226;
	add.s64 	%rd194, %rd222, %rd226;
	add.s64 	%rd195, %rd223, %rd226;
	add.s64 	%rd196, %rd224, %rd226;
	add.s64 	%rd197, %rd225, %rd226;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r663, %r938, %r455;
	mul.f32 	%r664, %r937, %r455;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs81, %rs82}, %r647;
	cvt.f32.bf16 	%r665, %rs82;
	cvt.f32.bf16 	%r666, %rs81;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r667, %r664, %r457, %r666;
	fma.rn.f32 	%r668, %r663, %r458, %r665;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r669, %r890, %r449;
	mul.f32 	%r670, %r889, %r449;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs83, %rs84}, %r621;
	cvt.f32.bf16 	%r671, %rs84;
	cvt.f32.bf16 	%r672, %rs83;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r673, %r670, %r457, %r672;
	fma.rn.f32 	%r674, %r669, %r458, %r671;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r500, %r674, %r673;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r675, %r892, %r450;
	mul.f32 	%r676, %r891, %r450;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs85, %rs86}, %r622;
	cvt.f32.bf16 	%r677, %rs86;
	cvt.f32.bf16 	%r678, %rs85;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r679, %r676, %r457, %r678;
	fma.rn.f32 	%r680, %r675, %r458, %r677;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r505, %r680, %r679;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r681, %r906, %r451;
	mul.f32 	%r682, %r905, %r451;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs87, %rs88}, %r631;
	cvt.f32.bf16 	%r683, %rs88;
	cvt.f32.bf16 	%r684, %rs87;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r685, %r682, %r457, %r684;
	fma.rn.f32 	%r686, %r681, %r458, %r683;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r501, %r686, %r685;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r687, %r908, %r452;
	mul.f32 	%r688, %r907, %r452;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs89, %rs90}, %r632;
	cvt.f32.bf16 	%r689, %rs90;
	cvt.f32.bf16 	%r690, %rs89;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r691, %r688, %r457, %r690;
	fma.rn.f32 	%r692, %r687, %r458, %r689;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r506, %r692, %r691;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r693, %r922, %r453;
	mul.f32 	%r694, %r921, %r453;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs91, %rs92}, %r639;
	cvt.f32.bf16 	%r695, %rs92;
	cvt.f32.bf16 	%r696, %rs91;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r697, %r694, %r457, %r696;
	fma.rn.f32 	%r698, %r693, %r458, %r695;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r502, %r698, %r697;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r699, %r924, %r454;
	mul.f32 	%r700, %r923, %r454;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs93, %rs94}, %r640;
	cvt.f32.bf16 	%r701, %rs94;
	cvt.f32.bf16 	%r702, %rs93;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r703, %r700, %r457, %r702;
	fma.rn.f32 	%r704, %r699, %r458, %r701;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r507, %r704, %r703;
	cvt.rn.bf16x2.f32 	%r503, %r668, %r667;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r705, %r940, %r456;
	mul.f32 	%r706, %r939, %r456;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs95, %rs96}, %r648;
	cvt.f32.bf16 	%r707, %rs96;
	cvt.f32.bf16 	%r708, %rs95;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r709, %r706, %r457, %r708;
	fma.rn.f32 	%r710, %r705, %r458, %r707;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r508, %r710, %r709;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r711, %r942, %r455;
	mul.f32 	%r712, %r941, %r455;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs97, %rs98}, %r649;
	cvt.f32.bf16 	%r713, %rs98;
	cvt.f32.bf16 	%r714, %rs97;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r715, %r712, %r459, %r714;
	fma.rn.f32 	%r716, %r711, %r460, %r713;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r717, %r894, %r449;
	mul.f32 	%r718, %r893, %r449;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs99, %rs100}, %r623;
	cvt.f32.bf16 	%r719, %rs100;
	cvt.f32.bf16 	%r720, %rs99;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r721, %r718, %r459, %r720;
	fma.rn.f32 	%r722, %r717, %r460, %r719;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r520, %r722, %r721;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r723, %r896, %r450;
	mul.f32 	%r724, %r895, %r450;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs101, %rs102}, %r624;
	cvt.f32.bf16 	%r725, %rs102;
	cvt.f32.bf16 	%r726, %rs101;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r727, %r724, %r459, %r726;
	fma.rn.f32 	%r728, %r723, %r460, %r725;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r525, %r728, %r727;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r729, %r910, %r451;
	mul.f32 	%r730, %r909, %r451;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs103, %rs104}, %r633;
	cvt.f32.bf16 	%r731, %rs104;
	cvt.f32.bf16 	%r732, %rs103;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r733, %r730, %r459, %r732;
	fma.rn.f32 	%r734, %r729, %r460, %r731;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r521, %r734, %r733;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r735, %r912, %r452;
	mul.f32 	%r736, %r911, %r452;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs105, %rs106}, %r634;
	cvt.f32.bf16 	%r737, %rs106;
	cvt.f32.bf16 	%r738, %rs105;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r739, %r736, %r459, %r738;
	fma.rn.f32 	%r740, %r735, %r460, %r737;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r526, %r740, %r739;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r741, %r926, %r453;
	mul.f32 	%r742, %r925, %r453;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs107, %rs108}, %r641;
	cvt.f32.bf16 	%r743, %rs108;
	cvt.f32.bf16 	%r744, %rs107;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r745, %r742, %r459, %r744;
	fma.rn.f32 	%r746, %r741, %r460, %r743;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r522, %r746, %r745;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r747, %r928, %r454;
	mul.f32 	%r748, %r927, %r454;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs109, %rs110}, %r642;
	cvt.f32.bf16 	%r749, %rs110;
	cvt.f32.bf16 	%r750, %rs109;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r751, %r748, %r459, %r750;
	fma.rn.f32 	%r752, %r747, %r460, %r749;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r527, %r752, %r751;
	cvt.rn.bf16x2.f32 	%r523, %r716, %r715;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r753, %r944, %r456;
	mul.f32 	%r754, %r943, %r456;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs111, %rs112}, %r650;
	cvt.f32.bf16 	%r755, %rs112;
	cvt.f32.bf16 	%r756, %rs111;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r757, %r754, %r459, %r756;
	fma.rn.f32 	%r758, %r753, %r460, %r755;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r528, %r758, %r757;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r759, %r946, %r455;
	mul.f32 	%r760, %r945, %r455;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs113, %rs114}, %r651;
	cvt.f32.bf16 	%r761, %rs114;
	cvt.f32.bf16 	%r762, %rs113;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r763, %r760, %r461, %r762;
	fma.rn.f32 	%r764, %r759, %r462, %r761;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r765, %r898, %r449;
	mul.f32 	%r766, %r897, %r449;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs115, %rs116}, %r627;
	cvt.f32.bf16 	%r767, %rs116;
	cvt.f32.bf16 	%r768, %rs115;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r769, %r766, %r461, %r768;
	fma.rn.f32 	%r770, %r765, %r462, %r767;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r510, %r770, %r769;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r771, %r900, %r450;
	mul.f32 	%r772, %r899, %r450;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs117, %rs118}, %r628;
	cvt.f32.bf16 	%r773, %rs118;
	cvt.f32.bf16 	%r774, %rs117;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r775, %r772, %r461, %r774;
	fma.rn.f32 	%r776, %r771, %r462, %r773;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r515, %r776, %r775;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r777, %r914, %r451;
	mul.f32 	%r778, %r913, %r451;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs119, %rs120}, %r635;
	cvt.f32.bf16 	%r779, %rs120;
	cvt.f32.bf16 	%r780, %rs119;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r781, %r778, %r461, %r780;
	fma.rn.f32 	%r782, %r777, %r462, %r779;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r511, %r782, %r781;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r783, %r916, %r452;
	mul.f32 	%r784, %r915, %r452;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs121, %rs122}, %r636;
	cvt.f32.bf16 	%r785, %rs122;
	cvt.f32.bf16 	%r786, %rs121;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r787, %r784, %r461, %r786;
	fma.rn.f32 	%r788, %r783, %r462, %r785;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r516, %r788, %r787;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r789, %r930, %r453;
	mul.f32 	%r790, %r929, %r453;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs123, %rs124}, %r643;
	cvt.f32.bf16 	%r791, %rs124;
	cvt.f32.bf16 	%r792, %rs123;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r793, %r790, %r461, %r792;
	fma.rn.f32 	%r794, %r789, %r462, %r791;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r512, %r794, %r793;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r795, %r932, %r454;
	mul.f32 	%r796, %r931, %r454;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs125, %rs126}, %r644;
	cvt.f32.bf16 	%r797, %rs126;
	cvt.f32.bf16 	%r798, %rs125;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r799, %r796, %r461, %r798;
	fma.rn.f32 	%r800, %r795, %r462, %r797;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r517, %r800, %r799;
	cvt.rn.bf16x2.f32 	%r513, %r764, %r763;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r801, %r948, %r456;
	mul.f32 	%r802, %r947, %r456;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs127, %rs128}, %r652;
	cvt.f32.bf16 	%r803, %rs128;
	cvt.f32.bf16 	%r804, %rs127;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r805, %r802, %r461, %r804;
	fma.rn.f32 	%r806, %r801, %r462, %r803;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r518, %r806, %r805;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r807, %r950, %r455;
	mul.f32 	%r808, %r949, %r455;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs129, %rs130}, %r653;
	cvt.f32.bf16 	%r809, %rs130;
	cvt.f32.bf16 	%r810, %rs129;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r811, %r808, %r463, %r810;
	fma.rn.f32 	%r812, %r807, %r464, %r809;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r813, %r902, %r449;
	mul.f32 	%r814, %r901, %r449;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs131, %rs132}, %r629;
	cvt.f32.bf16 	%r815, %rs132;
	cvt.f32.bf16 	%r816, %rs131;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r817, %r814, %r463, %r816;
	fma.rn.f32 	%r818, %r813, %r464, %r815;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r530, %r818, %r817;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r819, %r904, %r450;
	mul.f32 	%r820, %r903, %r450;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs133, %rs134}, %r630;
	cvt.f32.bf16 	%r821, %rs134;
	cvt.f32.bf16 	%r822, %rs133;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r823, %r820, %r463, %r822;
	fma.rn.f32 	%r824, %r819, %r464, %r821;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r535, %r824, %r823;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r825, %r918, %r451;
	mul.f32 	%r826, %r917, %r451;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs135, %rs136}, %r637;
	cvt.f32.bf16 	%r827, %rs136;
	cvt.f32.bf16 	%r828, %rs135;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r829, %r826, %r463, %r828;
	fma.rn.f32 	%r830, %r825, %r464, %r827;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r531, %r830, %r829;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r831, %r920, %r452;
	mul.f32 	%r832, %r919, %r452;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs137, %rs138}, %r638;
	cvt.f32.bf16 	%r833, %rs138;
	cvt.f32.bf16 	%r834, %rs137;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r835, %r832, %r463, %r834;
	fma.rn.f32 	%r836, %r831, %r464, %r833;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r536, %r836, %r835;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r837, %r934, %r453;
	mul.f32 	%r838, %r933, %r453;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs139, %rs140}, %r645;
	cvt.f32.bf16 	%r839, %rs140;
	cvt.f32.bf16 	%r840, %rs139;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r841, %r838, %r463, %r840;
	fma.rn.f32 	%r842, %r837, %r464, %r839;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r532, %r842, %r841;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r843, %r936, %r454;
	mul.f32 	%r844, %r935, %r454;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs141, %rs142}, %r646;
	cvt.f32.bf16 	%r845, %rs142;
	cvt.f32.bf16 	%r846, %rs141;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r847, %r844, %r463, %r846;
	fma.rn.f32 	%r848, %r843, %r464, %r845;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r537, %r848, %r847;
	cvt.rn.bf16x2.f32 	%r533, %r812, %r811;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r849, %r952, %r456;
	mul.f32 	%r850, %r951, %r456;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs143, %rs144}, %r654;
	cvt.f32.bf16 	%r851, %rs144;
	cvt.f32.bf16 	%r852, %rs143;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r853, %r850, %r463, %r852;
	fma.rn.f32 	%r854, %r849, %r464, %r851;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r538, %r854, %r853;
	bar.sync 	0;
	shl.b32 	%r855, %r4, 13;
	shl.b32 	%r856, %r4, 5;
	and.b32 	%r857, %r13, 384;
	shr.u32 	%r858, %r5, 1;
	bfe.s32 	%r859, %r2, 2, 1;
	and.b32 	%r860, %r859, 4112;
	shl.b32 	%r861, %r573, 3;
	or.b32 	%r862, %r855, %r861;
	or.b32 	%r863, %r856, %r857;
	xor.b32 	%r864, %r860, %r858;
	xor.b32 	%r865, %r864, %r863;
	or.b32 	%r866, %r865, %r862;
	add.s32 	%r499, %r176, %r866;
	// begin inline asm
	st.shared.v4.b32 [ %r499 + 0 ], { %r500, %r501, %r502, %r503 };
	// end inline asm
	add.s32 	%r504, %r499, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r504 + 0 ], { %r505, %r506, %r507, %r508 };
	// end inline asm
	add.s32 	%r509, %r499, 2048;
	// begin inline asm
	st.shared.v4.b32 [ %r509 + 0 ], { %r510, %r511, %r512, %r513 };
	// end inline asm
	add.s32 	%r514, %r499, 2560;
	// begin inline asm
	st.shared.v4.b32 [ %r514 + 0 ], { %r515, %r516, %r517, %r518 };
	// end inline asm
	xor.b32 	%r867, %r866, 64;
	add.s32 	%r519, %r176, %r867;
	// begin inline asm
	st.shared.v4.b32 [ %r519 + 0 ], { %r520, %r521, %r522, %r523 };
	// end inline asm
	add.s32 	%r524, %r519, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r524 + 0 ], { %r525, %r526, %r527, %r528 };
	// end inline asm
	add.s32 	%r529, %r519, 2048;
	// begin inline asm
	st.shared.v4.b32 [ %r529 + 0 ], { %r530, %r531, %r532, %r533 };
	// end inline asm
	add.s32 	%r534, %r519, 2560;
	// begin inline asm
	st.shared.v4.b32 [ %r534 + 0 ], { %r535, %r536, %r537, %r538 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r868, %r2, 2;
	and.b32 	%r869, %r868, 896;
	shl.b32 	%r870, %r2, 8;
	and.b32 	%r871, %r870, 2048;
	setp.eq.b32 	%p24, %r887, 0;
	selp.b32 	%r872, 0, 4112, %p24;
	or.b32 	%r873, %r883, %r869;
	xor.b32 	%r874, %r873, %r872;
	or.b32 	%r875, %r874, %r871;
	add.s32 	%r876, %r176, %r875;
	ld.shared.v4.b32 	{%r539, %r547, %r555, %r563}, [%r876];
	ld.shared.v4.b32 	{%r543, %r551, %r559, %r567}, [%r876+1024];
	xor.b32 	%r877, %r875, 32;
	add.s32 	%r878, %r176, %r877;
	ld.shared.v4.b32 	{%r540, %r548, %r556, %r564}, [%r878+8192];
	ld.shared.v4.b32 	{%r544, %r552, %r560, %r568}, [%r878+9216];
	xor.b32 	%r879, %r875, 64;
	add.s32 	%r880, %r176, %r879;
	ld.shared.v4.b32 	{%r541, %r549, %r557, %r565}, [%r880+16384];
	ld.shared.v4.b32 	{%r545, %r553, %r561, %r569}, [%r880+17408];
	xor.b32 	%r881, %r875, 96;
	add.s32 	%r882, %r176, %r881;
	ld.shared.v4.b32 	{%r542, %r550, %r558, %r566}, [%r882+24576];
	ld.shared.v4.b32 	{%r546, %r554, %r562, %r570}, [%r882+25600];
	.loc	1 172 8                         // sk07_lm_head.py:172:8
	// begin inline asm
	@%p7 st.global.v4.b32 [ %rd190 + 0 ], { %r539, %r540, %r541, %r542 };
	// end inline asm
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd191 + 0 ], { %r543, %r544, %r545, %r546 };
	// end inline asm
	// begin inline asm
	@%p9 st.global.v4.b32 [ %rd192 + 0 ], { %r547, %r548, %r549, %r550 };
	// end inline asm
	// begin inline asm
	@%p10 st.global.v4.b32 [ %rd193 + 0 ], { %r551, %r552, %r553, %r554 };
	// end inline asm
	// begin inline asm
	@%p11 st.global.v4.b32 [ %rd194 + 0 ], { %r555, %r556, %r557, %r558 };
	// end inline asm
	// begin inline asm
	@%p12 st.global.v4.b32 [ %rd195 + 0 ], { %r559, %r560, %r561, %r562 };
	// end inline asm
	// begin inline asm
	@%p13 st.global.v4.b32 [ %rd196 + 0 ], { %r563, %r564, %r565, %r566 };
	// end inline asm
	// begin inline asm
	@%p14 st.global.v4.b32 [ %rd197 + 0 ], { %r567, %r568, %r569, %r570 };
	// end inline asm
	.loc	1 170 4                         // sk07_lm_head.py:170:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk07_lm_head.py"
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
.b32 159                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x98 DW_TAG_compile_unit
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
.b8 55
.b8 95
.b8 108
.b8 109
.b8 95
.b8 104
.b8 101
.b8 97
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
.b8 2                                   // Abbrev [2] 0x45:0x17 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 55
.b8 95
.b8 108
.b8 109
.b8 95
.b8 104
.b8 101
.b8 97
.b8 100
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x5c:0x46 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 69                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x71:0x18 DW_TAG_inlined_subroutine
.b32 69                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 133                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x89:0x18 DW_TAG_inlined_subroutine
.b32 69                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 134                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_2 = _Nativo(
    "sk07_lm_head/tile128x128x128_shift0_abi16",
    _PTX_2, "_sk07_lm_head_kernel",
    warps=8, shared=65536,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 17, 19],
    horneado={12: 1, 13: 1, 16: 1, 18: 1, 20: 1, 21: 128, 22: 128, 23: 128, 24: 8, 25: 128},
    div16=[9, 10, 11, 14, 15, 17],
)

_PTX_3 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk07_lm_head_kernel    // -- Begin function _sk07_lm_head_kernel
.extern .shared .align 16 .b8 global_smem[];
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
                                        // @_sk07_lm_head_kernel
.visible .entry _sk07_lm_head_kernel(
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_6,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_7,
	.param .u32 _sk07_lm_head_kernel_param_8,
	.param .u32 _sk07_lm_head_kernel_param_9,
	.param .u32 _sk07_lm_head_kernel_param_10,
	.param .u32 _sk07_lm_head_kernel_param_11,
	.param .u32 _sk07_lm_head_kernel_param_12,
	.param .u32 _sk07_lm_head_kernel_param_13,
	.param .u32 _sk07_lm_head_kernel_param_14,
	.param .u32 _sk07_lm_head_kernel_param_15,
	.param .u32 _sk07_lm_head_kernel_param_16,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_17,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_18
)
.reqntid 256
{
	.reg .pred 	%p<25>;
	.reg .b16 	%rs<145>;
	.reg .b32 	%r<970>;
	.reg .b64 	%rd<229>;
	.loc	1 123 0                         // sk07_lm_head.py:123:0
$L__func_begin0:
	.loc	1 123 0                         // sk07_lm_head.py:123:0

// %bb.0:
	ld.param.b32 	%r26, [_sk07_lm_head_kernel_param_15];
	ld.param.b32 	%r25, [_sk07_lm_head_kernel_param_14];
	ld.param.b32 	%r24, [_sk07_lm_head_kernel_param_13];
	ld.param.b32 	%r23, [_sk07_lm_head_kernel_param_10];
	ld.param.b32 	%r22, [_sk07_lm_head_kernel_param_9];
	ld.param.b32 	%r21, [_sk07_lm_head_kernel_param_8];
	ld.param.b64 	%rd33, [_sk07_lm_head_kernel_param_6];
	ld.param.b64 	%rd32, [_sk07_lm_head_kernel_param_5];
	ld.param.b64 	%rd31, [_sk07_lm_head_kernel_param_4];
	ld.param.b64 	%rd30, [_sk07_lm_head_kernel_param_2];
	ld.param.b64 	%rd29, [_sk07_lm_head_kernel_param_1];
	ld.param.b64 	%rd28, [_sk07_lm_head_kernel_param_0];
$L__tmp0:
	.loc	1 132 24                        // sk07_lm_head.py:132:24
	mov.u32 	%r74, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk07_lm_head.py:133:27 ]
	add.s32 	%r75, %r21, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk07_lm_head.py:133:27 ]
	shr.s32 	%r76, %r75, 31;
	shr.u32 	%r77, %r76, 25;
	add.s32 	%r78, %r75, %r77;
	shr.s32 	%r79, %r78, 7;
	ld.param.b64 	%rd62, [_sk07_lm_head_kernel_param_3];
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk07_lm_head.py:134:27 ]
	add.s32 	%r80, %r22, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk07_lm_head.py:134:27 ]
	shr.s32 	%r81, %r80, 31;
	shr.u32 	%r82, %r81, 25;
	add.s32 	%r83, %r80, %r82;
	shr.s32 	%r84, %r83, 7;
$L__tmp3:
	.loc	1 135 29                        // sk07_lm_head.py:135:29
	shl.b32 	%r85, %r84, 3;
	.loc	1 136 22                        // sk07_lm_head.py:136:22
	div.s32 	%r86, %r74, %r85;
	.loc	1 136 38                        // sk07_lm_head.py:136:38
	shl.b32 	%r87, %r86, 3;
	.loc	1 137 30                        // sk07_lm_head.py:137:30
	sub.s32 	%r88, %r79, %r87;
	.loc	1 137 39                        // sk07_lm_head.py:137:39
	min.s32 	%r89, %r88, 8;
	ld.param.b32 	%r90, [_sk07_lm_head_kernel_param_11];
	.loc	1 138 30                        // sk07_lm_head.py:138:30
	mul.lo.s32 	%r91, %r86, %r85;
	ld.param.b32 	%r92, [_sk07_lm_head_kernel_param_12];
	sub.s32 	%r93, %r74, %r91;
	.loc	1 139 36                        // sk07_lm_head.py:139:36
	div.s32 	%r94, %r93, %r89;
	.loc	1 138 46                        // sk07_lm_head.py:138:46
	mul.lo.s32 	%r95, %r94, %r89;
	sub.s32 	%r96, %r93, %r95;
	.loc	1 138 23                        // sk07_lm_head.py:138:23
	add.s32 	%r97, %r96, %r87;
	.loc	1 141 22                        // sk07_lm_head.py:141:22
	shl.b32 	%r1, %r97, 7;
	.loc	1 141 45                        // sk07_lm_head.py:141:45
	mov.u32 	%r2, %tid.x;
	and.b32 	%r3, %r2, 248;
	bfe.u32 	%r98, %r2, 3, 5;
	or.b32 	%r99, %r98, 32;
	or.b32 	%r100, %r98, 64;
	or.b32 	%r101, %r98, 96;
	and.b32 	%r4, %r2, 3;
	shl.b32 	%r102, %r4, 1;
	and.b32 	%r5, %r2, 96;
	shr.u32 	%r103, %r5, 2;
	or.b32 	%r104, %r102, %r103;
	and.b32 	%r6, %r2, 7;
	shl.b32 	%r105, %r6, 4;
	and.b32 	%r7, %r2, 15;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r106, %r1, %r98;
	or.b32 	%r107, %r1, %r99;
	or.b32 	%r108, %r1, %r100;
	or.b32 	%r109, %r1, %r101;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r8, %r106, %r21;
	rem.s32 	%r9, %r107, %r21;
	rem.s32 	%r10, %r108, %r21;
	rem.s32 	%r11, %r109, %r21;
	.loc	1 142 22                        // sk07_lm_head.py:142:22
	shl.b32 	%r12, %r94, 7;
	.loc	1 142 32                        // sk07_lm_head.py:142:32
	or.b32 	%r110, %r12, %r98;
	or.b32 	%r111, %r12, %r99;
	or.b32 	%r112, %r12, %r100;
	or.b32 	%r113, %r12, %r101;
	or.b32 	%r114, %r12, %r104;
	or.b32 	%r115, %r114, 32;
	or.b32 	%r116, %r114, 64;
	or.b32 	%r117, %r114, 96;
	or.b32 	%r118, %r12, %r105;
	or.b32 	%r119, %r118, 4;
	or.b32 	%r120, %r118, 8;
	or.b32 	%r121, %r118, 12;
	.loc	1 142 57                        // sk07_lm_head.py:142:57
	rem.s32 	%r122, %r110, %r22;
	rem.s32 	%r123, %r111, %r22;
	rem.s32 	%r124, %r112, %r22;
	rem.s32 	%r125, %r113, %r22;
	rem.s32 	%r126, %r114, %r22;
	rem.s32 	%r127, %r115, %r22;
	rem.s32 	%r128, %r116, %r22;
	rem.s32 	%r129, %r117, %r22;
	rem.s32 	%r130, %r118, %r22;
	rem.s32 	%r131, %r119, %r22;
	rem.s32 	%r132, %r120, %r22;
	rem.s32 	%r133, %r121, %r22;
	.loc	1 145 29                        // sk07_lm_head.py:145:29
	mad.wide.s32 	%rd34, %r122, 4, %rd62;
	mad.wide.s32 	%rd35, %r123, 4, %rd62;
	mad.wide.s32 	%rd36, %r124, 4, %rd62;
	mad.wide.s32 	%rd37, %r125, 4, %rd62;
	mad.wide.s32 	%rd38, %r126, 4, %rd62;
	mad.wide.s32 	%rd39, %r127, 4, %rd62;
	mad.wide.s32 	%rd40, %r128, 4, %rd62;
	mad.wide.s32 	%rd41, %r129, 4, %rd62;
	mad.wide.s32 	%rd42, %r130, 4, %rd62;
	mad.wide.s32 	%rd43, %r131, 4, %rd62;
	mad.wide.s32 	%rd44, %r132, 4, %rd62;
	mad.wide.s32 	%rd45, %r133, 4, %rd62;
	.loc	1 145 19                        // sk07_lm_head.py:145:19
	// begin inline asm
	mov.u32 %r28, 0x0;
	ld.global.b32 { %r28 }, [ %rd34 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r29, 0x0;
	ld.global.b32 { %r29 }, [ %rd35 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r30, 0x0;
	ld.global.b32 { %r30 }, [ %rd36 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r31, 0x0;
	ld.global.b32 { %r31 }, [ %rd37 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r32, 0x0;
	mov.u32 %r33, 0x0;
	ld.global.v2.b32 { %r32, %r33 }, [ %rd38 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r34, 0x0;
	mov.u32 %r35, 0x0;
	ld.global.v2.b32 { %r34, %r35 }, [ %rd39 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r36, 0x0;
	mov.u32 %r37, 0x0;
	ld.global.v2.b32 { %r36, %r37 }, [ %rd40 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r38, 0x0;
	mov.u32 %r39, 0x0;
	ld.global.v2.b32 { %r38, %r39 }, [ %rd41 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r40, 0x0;
	mov.u32 %r41, 0x0;
	mov.u32 %r42, 0x0;
	mov.u32 %r43, 0x0;
	ld.global.v4.b32 { %r40, %r41, %r42, %r43 }, [ %rd42 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r44, 0x0;
	mov.u32 %r45, 0x0;
	mov.u32 %r46, 0x0;
	mov.u32 %r47, 0x0;
	ld.global.v4.b32 { %r44, %r45, %r46, %r47 }, [ %rd43 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r48, 0x0;
	mov.u32 %r49, 0x0;
	mov.u32 %r50, 0x0;
	mov.u32 %r51, 0x0;
	ld.global.v4.b32 { %r48, %r49, %r50, %r51 }, [ %rd44 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r52, 0x0;
	mov.u32 %r53, 0x0;
	mov.u32 %r54, 0x0;
	mov.u32 %r55, 0x0;
	ld.global.v4.b32 { %r52, %r53, %r54, %r55 }, [ %rd45 + 0 ];
	// end inline asm
	.loc	1 146 39                        // sk07_lm_head.py:146:39
	mul.lo.s32 	%r134, %r8, %r90;
	mul.lo.s32 	%r135, %r9, %r90;
	mul.lo.s32 	%r136, %r10, %r90;
	mul.lo.s32 	%r137, %r11, %r90;
	.loc	1 146 21                        // sk07_lm_head.py:146:21
	cvt.s64.s32 	%rd1, %r134;
	add.s64 	%rd64, %rd28, %rd1;
	cvt.s64.s32 	%rd2, %r135;
	add.s64 	%rd65, %rd28, %rd2;
	cvt.s64.s32 	%rd3, %r136;
	add.s64 	%rd66, %rd28, %rd3;
	cvt.s64.s32 	%rd4, %r137;
	add.s64 	%rd67, %rd28, %rd4;
	.loc	1 146 51                        // sk07_lm_head.py:146:51
	cvt.u64.u32 	%rd5, %r105;
	add.s64 	%rd46, %rd64, %rd5;
	add.s64 	%rd47, %rd65, %rd5;
	add.s64 	%rd48, %rd66, %rd5;
	add.s64 	%rd49, %rd67, %rd5;
	.loc	1 147 21                        // sk07_lm_head.py:147:21
	add.s64 	%rd68, %rd29, %rd5;
	.loc	1 147 67                        // sk07_lm_head.py:147:67
	mul.lo.s32 	%r138, %r28, %r92;
	mul.lo.s32 	%r139, %r29, %r92;
	mul.lo.s32 	%r140, %r30, %r92;
	mul.lo.s32 	%r141, %r31, %r92;
	.loc	1 147 51                        // sk07_lm_head.py:147:51
	cvt.s64.s32 	%rd6, %r138;
	add.s64 	%rd50, %rd68, %rd6;
	cvt.s64.s32 	%rd7, %r139;
	add.s64 	%rd51, %rd68, %rd7;
	cvt.s64.s32 	%rd8, %r140;
	add.s64 	%rd52, %rd68, %rd8;
	cvt.s64.s32 	%rd9, %r141;
	add.s64 	%rd53, %rd68, %rd9;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	setp.gt.s32 	%p1, %r23, 127;
	.loc	1 160 30                        // sk07_lm_head.py:160:30
	shl.b32 	%r13, %r2, 4;
	and.b32 	%r174, %r13, 4080;
	and.b32 	%r14, %r2, 56;
	shl.b32 	%r175, %r14, 1;
	xor.b32 	%r176, %r174, %r175;
	mov.b32 	%r177, global_smem;
	add.s32 	%r56, %r177, %r176;
	selp.b32 	%r57, 16, 0, %p1;
	// begin inline asm
	cp.async.cg.shared.global [ %r56 + 0 ], [ %rd46 + 0 ], 0x10, %r57;
	// end inline asm
	add.s32 	%r58, %r56, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r58 + 0 ], [ %rd47 + 0 ], 0x10, %r57;
	// end inline asm
	add.s32 	%r59, %r56, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r59 + 0 ], [ %rd48 + 0 ], 0x10, %r57;
	// end inline asm
	add.s32 	%r60, %r56, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r60 + 0 ], [ %rd49 + 0 ], 0x10, %r57;
	// end inline asm
	cp.async.commit_group;
	.loc	1 160 47                        // sk07_lm_head.py:160:47
	add.s32 	%r61, %r56, 32768;
	// begin inline asm
	cp.async.cg.shared.global [ %r61 + 0 ], [ %rd50 + 0 ], 0x10, %r57;
	// end inline asm
	add.s32 	%r62, %r56, 36864;
	// begin inline asm
	cp.async.cg.shared.global [ %r62 + 0 ], [ %rd51 + 0 ], 0x10, %r57;
	// end inline asm
	add.s32 	%r63, %r56, 40960;
	// begin inline asm
	cp.async.cg.shared.global [ %r63 + 0 ], [ %rd52 + 0 ], 0x10, %r57;
	// end inline asm
	add.s32 	%r64, %r56, 45056;
	// begin inline asm
	cp.async.cg.shared.global [ %r64 + 0 ], [ %rd53 + 0 ], 0x10, %r57;
	// end inline asm
	cp.async.commit_group;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	setp.gt.s32 	%p2, %r23, 255;
	.loc	1 161 18                        // sk07_lm_head.py:161:18
	add.s64 	%rd54, %rd46, 128;
	add.s64 	%rd55, %rd47, 128;
	add.s64 	%rd56, %rd48, 128;
	add.s64 	%rd57, %rd49, 128;
	.loc	1 162 18                        // sk07_lm_head.py:162:18
	add.s64 	%rd58, %rd50, 128;
	add.s64 	%rd59, %rd51, 128;
	add.s64 	%rd60, %rd52, 128;
	add.s64 	%rd61, %rd53, 128;
	.loc	1 160 30                        // sk07_lm_head.py:160:30
	bar.sync 	0;
	add.s32 	%r65, %r56, 16384;
	selp.b32 	%r66, 16, 0, %p2;
	// begin inline asm
	cp.async.cg.shared.global [ %r65 + 0 ], [ %rd54 + 0 ], 0x10, %r66;
	// end inline asm
	add.s32 	%r67, %r56, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r67 + 0 ], [ %rd55 + 0 ], 0x10, %r66;
	// end inline asm
	add.s32 	%r68, %r56, 24576;
	// begin inline asm
	cp.async.cg.shared.global [ %r68 + 0 ], [ %rd56 + 0 ], 0x10, %r66;
	// end inline asm
	add.s32 	%r69, %r56, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r69 + 0 ], [ %rd57 + 0 ], 0x10, %r66;
	// end inline asm
	cp.async.commit_group;
	.loc	1 160 47                        // sk07_lm_head.py:160:47
	add.s32 	%r70, %r56, 49152;
	// begin inline asm
	cp.async.cg.shared.global [ %r70 + 0 ], [ %rd58 + 0 ], 0x10, %r66;
	// end inline asm
	add.s32 	%r71, %r56, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r71 + 0 ], [ %rd59 + 0 ], 0x10, %r66;
	// end inline asm
	add.s32 	%r72, %r56, 57344;
	// begin inline asm
	cp.async.cg.shared.global [ %r72 + 0 ], [ %rd60 + 0 ], 0x10, %r66;
	// end inline asm
	add.s32 	%r73, %r56, 61440;
	// begin inline asm
	cp.async.cg.shared.global [ %r73 + 0 ], [ %rd61 + 0 ], 0x10, %r66;
	// end inline asm
	cp.async.commit_group;
	cvt.u32.u64 	%r900, %rd5;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	@%p1 bra 	$L__BB0_2;
	bra.uni 	$L__BB0_1;
$L__BB0_2:                              // %.lr.ph
	.loc	1 0 23                          // sk07_lm_head.py:0:23
	ld.param.b32 	%r27, [_sk07_lm_head_kernel_param_16];
	ld.param.b64 	%rd63, [_sk07_lm_head_kernel_param_7];
	shr.s32 	%r142, %r32, 31;
	shr.u32 	%r143, %r142, 25;
	add.s32 	%r144, %r32, %r143;
	shr.s32 	%r145, %r144, 7;
	shr.s32 	%r146, %r33, 31;
	shr.u32 	%r147, %r146, 25;
	add.s32 	%r148, %r33, %r147;
	shr.s32 	%r149, %r148, 7;
	shr.s32 	%r150, %r34, 31;
	shr.u32 	%r151, %r150, 25;
	add.s32 	%r152, %r34, %r151;
	shr.s32 	%r153, %r152, 7;
	shr.s32 	%r154, %r35, 31;
	shr.u32 	%r155, %r154, 25;
	add.s32 	%r156, %r35, %r155;
	shr.s32 	%r157, %r156, 7;
	shr.s32 	%r158, %r36, 31;
	shr.u32 	%r159, %r158, 25;
	add.s32 	%r160, %r36, %r159;
	shr.s32 	%r161, %r160, 7;
	shr.s32 	%r162, %r37, 31;
	shr.u32 	%r163, %r162, 25;
	add.s32 	%r164, %r37, %r163;
	shr.s32 	%r165, %r164, 7;
	shr.s32 	%r166, %r38, 31;
	shr.u32 	%r167, %r166, 25;
	add.s32 	%r168, %r38, %r167;
	shr.s32 	%r169, %r168, 7;
	shr.s32 	%r170, %r39, 31;
	shr.u32 	%r171, %r170, 25;
	add.s32 	%r172, %r39, %r171;
	shr.s32 	%r173, %r172, 7;
	cvt.s64.s32 	%rd69, %r145;
	add.s64 	%rd10, %rd63, %rd69;
	cvt.s64.s32 	%rd70, %r149;
	add.s64 	%rd11, %rd63, %rd70;
	cvt.s64.s32 	%rd71, %r153;
	add.s64 	%rd12, %rd63, %rd71;
	cvt.s64.s32 	%rd72, %r157;
	add.s64 	%rd13, %rd63, %rd72;
	cvt.s64.s32 	%rd73, %r161;
	add.s64 	%rd14, %rd63, %rd73;
	cvt.s64.s32 	%rd74, %r165;
	add.s64 	%rd15, %rd63, %rd74;
	cvt.s64.s32 	%rd75, %r169;
	add.s64 	%rd16, %rd63, %rd75;
	cvt.s64.s32 	%rd76, %r173;
	add.s64 	%rd17, %rd63, %rd76;
	.loc	1 151 28                        // sk07_lm_head.py:151:28
	shr.u32 	%r178, %r23, 7;
	add.s32 	%r179, %r178, -2;
	shl.b32 	%r180, %r7, 7;
	and.b32 	%r181, %r13, 2160;
	and.b32 	%r904, %r2, 16;
	or.b32 	%r182, %r180, %r181;
	xor.b32 	%r15, %r182, %r904;
	xor.b32 	%r16, %r15, 32;
	xor.b32 	%r17, %r15, 64;
	xor.b32 	%r18, %r15, 96;
	shl.b32 	%r183, %r6, 7;
	shl.b32 	%r184, %r5, 5;
	shl.b32 	%r905, %r2, 1;
	and.b32 	%r185, %r905, 48;
	or.b32 	%r186, %r183, %r184;
	xor.b32 	%r187, %r900, %r185;
	or.b32 	%r19, %r186, %r187;
	xor.b32 	%r20, %r19, 64;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	cvt.s64.s32 	%rd18, %r179;
	and.b32 	%r188, %r23, -128;
	cvt.u64.u32 	%rd19, %r188;
	add.s64 	%rd77, %rd5, %rd9;
	add.s64 	%rd78, %rd77, %rd29;
	add.s64 	%rd20, %rd78, 256;
	add.s64 	%rd79, %rd5, %rd8;
	add.s64 	%rd80, %rd79, %rd29;
	add.s64 	%rd21, %rd80, 256;
	add.s64 	%rd81, %rd5, %rd7;
	add.s64 	%rd82, %rd81, %rd29;
	add.s64 	%rd22, %rd82, 256;
	add.s64 	%rd83, %rd5, %rd6;
	add.s64 	%rd84, %rd83, %rd29;
	add.s64 	%rd23, %rd84, 256;
	add.s64 	%rd85, %rd5, %rd4;
	add.s64 	%rd86, %rd85, %rd28;
	add.s64 	%rd24, %rd86, 256;
	add.s64 	%rd87, %rd5, %rd3;
	add.s64 	%rd88, %rd87, %rd28;
	add.s64 	%rd25, %rd88, 256;
	add.s64 	%rd89, %rd5, %rd2;
	add.s64 	%rd90, %rd89, %rd28;
	add.s64 	%rd26, %rd90, 256;
	add.s64 	%rd91, %rd5, %rd1;
	add.s64 	%rd92, %rd91, %rd28;
	add.s64 	%rd27, %rd92, 256;
	mov.b32 	%r906, 0f00000000;
	mov.b32 	%r903, 1;
	mov.b32 	%r902, -1;
	mov.b64 	%rd227, 0;
	mov.b32 	%r189, 0;
	mov.b32 	%r901, %r189;
	mov.b64 	%rd228, %rd227;
	mov.b32 	%r907, %r906;
	mov.b32 	%r908, %r906;
	mov.b32 	%r909, %r906;
	mov.b32 	%r910, %r906;
	mov.b32 	%r911, %r906;
	mov.b32 	%r912, %r906;
	mov.b32 	%r913, %r906;
	mov.b32 	%r914, %r906;
	mov.b32 	%r915, %r906;
	mov.b32 	%r916, %r906;
	mov.b32 	%r917, %r906;
	mov.b32 	%r918, %r906;
	mov.b32 	%r919, %r906;
	mov.b32 	%r920, %r906;
	mov.b32 	%r921, %r906;
	mov.b32 	%r922, %r906;
	mov.b32 	%r923, %r906;
	mov.b32 	%r924, %r906;
	mov.b32 	%r925, %r906;
	mov.b32 	%r926, %r906;
	mov.b32 	%r927, %r906;
	mov.b32 	%r928, %r906;
	mov.b32 	%r929, %r906;
	mov.b32 	%r930, %r906;
	mov.b32 	%r931, %r906;
	mov.b32 	%r932, %r906;
	mov.b32 	%r933, %r906;
	mov.b32 	%r934, %r906;
	mov.b32 	%r935, %r906;
	mov.b32 	%r936, %r906;
	mov.b32 	%r937, %r906;
	mov.b32 	%r938, %r906;
	mov.b32 	%r939, %r906;
	mov.b32 	%r940, %r906;
	mov.b32 	%r941, %r906;
	mov.b32 	%r942, %r906;
	mov.b32 	%r943, %r906;
	mov.b32 	%r944, %r906;
	mov.b32 	%r945, %r906;
	mov.b32 	%r946, %r906;
	mov.b32 	%r947, %r906;
	mov.b32 	%r948, %r906;
	mov.b32 	%r949, %r906;
	mov.b32 	%r950, %r906;
	mov.b32 	%r951, %r906;
	mov.b32 	%r952, %r906;
	mov.b32 	%r953, %r906;
	mov.b32 	%r954, %r906;
	mov.b32 	%r955, %r906;
	mov.b32 	%r956, %r906;
	mov.b32 	%r957, %r906;
	mov.b32 	%r958, %r906;
	mov.b32 	%r959, %r906;
	mov.b32 	%r960, %r906;
	mov.b32 	%r961, %r906;
	mov.b32 	%r962, %r906;
	mov.b32 	%r963, %r906;
	mov.b32 	%r964, %r906;
	mov.b32 	%r965, %r906;
	mov.b32 	%r966, %r906;
	mov.b32 	%r967, %r906;
	mov.b32 	%r968, %r906;
	mov.b32 	%r969, %r906;
$L__BB0_3:                              // %__nv_exp2f.exit
                                        // =>This Inner Loop Header: Depth=1
	setp.lt.s64 	%p3, %rd228, %rd18;
	add.s32 	%r359, %r902, 1;
	setp.gt.s32 	%p4, %r359, 1;
	selp.b32 	%r902, 0, %r359, %p4;
	.loc	1 158 39                        // sk07_lm_head.py:158:39
	cvt.s64.s32 	%rd109, %r901;
	add.s64 	%rd93, %rd10, %rd109;
	add.s64 	%rd94, %rd11, %rd109;
	add.s64 	%rd95, %rd12, %rd109;
	add.s64 	%rd96, %rd13, %rd109;
	add.s64 	%rd97, %rd14, %rd109;
	add.s64 	%rd98, %rd15, %rd109;
	add.s64 	%rd99, %rd16, %rd109;
	add.s64 	%rd100, %rd17, %rd109;
	.loc	1 158 29                        // sk07_lm_head.py:158:29
	// begin inline asm
	mov.u16 %rs1, 0x0;
	ld.global.b8 { %rs1 }, [ %rd93 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs9, %rs1;
	// begin inline asm
	mov.u16 %rs2, 0x0;
	ld.global.b8 { %rs2 }, [ %rd94 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs10, %rs2;
	// begin inline asm
	mov.u16 %rs3, 0x0;
	ld.global.b8 { %rs3 }, [ %rd95 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs11, %rs3;
	// begin inline asm
	mov.u16 %rs4, 0x0;
	ld.global.b8 { %rs4 }, [ %rd96 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs12, %rs4;
	// begin inline asm
	mov.u16 %rs5, 0x0;
	ld.global.b8 { %rs5 }, [ %rd97 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs13, %rs5;
	// begin inline asm
	mov.u16 %rs6, 0x0;
	ld.global.b8 { %rs6 }, [ %rd98 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs14, %rs6;
	// begin inline asm
	mov.u16 %rs7, 0x0;
	ld.global.b8 { %rs7 }, [ %rd99 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs15, %rs7;
	// begin inline asm
	mov.u16 %rs8, 0x0;
	ld.global.b8 { %rs8 }, [ %rd100 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs16, %rs8;
	.loc	1 159 50                        // sk07_lm_head.py:159:50
	cvt.rn.f32.s16 	%r360, %rs9;
	cvt.rn.f32.s16 	%r361, %rs10;
	cvt.rn.f32.s16 	%r362, %rs11;
	cvt.rn.f32.s16 	%r363, %rs12;
	cvt.rn.f32.s16 	%r364, %rs13;
	cvt.rn.f32.s16 	%r365, %rs14;
	cvt.rn.f32.s16 	%r366, %rs15;
	cvt.rn.f32.s16 	%r367, %rs16;
	.loc	1 158 21                        // sk07_lm_head.py:158:21
	ex2.approx.ftz.f32 	%r368, %r360;
	ex2.approx.ftz.f32 	%r369, %r361;
	ex2.approx.ftz.f32 	%r370, %r362;
	ex2.approx.ftz.f32 	%r371, %r363;
	ex2.approx.ftz.f32 	%r372, %r364;
	ex2.approx.ftz.f32 	%r373, %r365;
	ex2.approx.ftz.f32 	%r374, %r366;
	ex2.approx.ftz.f32 	%r375, %r367;
	.loc	1 160 30                        // sk07_lm_head.py:160:30
	cp.async.wait_group 	2;
	bar.sync 	0;
	shl.b32 	%r376, %r902, 14;
	add.s32 	%r377, %r177, %r376;
	add.s32 	%r378, %r377, %r15;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r190, %r191, %r192, %r193}, [%r378];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r202, %r203, %r204, %r205}, [%r378+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r206, %r207, %r208, %r209}, [%r378+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r210, %r211, %r212, %r213}, [%r378+12288];
	add.s32 	%r379, %r377, %r16;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r218, %r219, %r220, %r221}, [%r379];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r246, %r247, %r248, %r249}, [%r379+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r266, %r267, %r268, %r269}, [%r379+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r286, %r287, %r288, %r289}, [%r379+12288];
	add.s32 	%r380, %r377, %r17;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r302, %r303, %r304, %r305}, [%r380];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r314, %r315, %r316, %r317}, [%r380+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r318, %r319, %r320, %r321}, [%r380+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r322, %r323, %r324, %r325}, [%r380+12288];
	add.s32 	%r381, %r377, %r18;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r326, %r327, %r328, %r329}, [%r381];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r338, %r339, %r340, %r341}, [%r381+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r342, %r343, %r344, %r345}, [%r381+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r346, %r347, %r348, %r349}, [%r381+12288];
	.loc	1 160 47                        // sk07_lm_head.py:160:47
	add.s32 	%r382, %r377, %r19;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r194, %r195, %r222, %r223}, [%r382+32768];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r196, %r197, %r228, %r229}, [%r382+36864];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r198, %r199, %r234, %r235}, [%r382+40960];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r200, %r201, %r240, %r241}, [%r382+45056];
	add.s32 	%r383, %r377, %r20;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r306, %r307, %r330, %r331}, [%r383+32768];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r308, %r309, %r332, %r333}, [%r383+36864];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r310, %r311, %r334, %r335}, [%r383+40960];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r312, %r313, %r336, %r337}, [%r383+45056];
	.loc	1 160 39                        // sk07_lm_head.py:160:39
	mov.b32 	%r214, %r189;
	mov.b32 	%r215, %r189;
	mov.b32 	%r216, %r189;
	mov.b32 	%r217, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r214, %r215, %r216, %r217 }, { %r190, %r191, %r192, %r193 }, { %r194, %r195 }, { %r214, %r215, %r216, %r217 };
	// end inline asm
	mov.b32 	%r224, %r189;
	mov.b32 	%r225, %r189;
	mov.b32 	%r226, %r189;
	mov.b32 	%r227, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r224, %r225, %r226, %r227 }, { %r190, %r191, %r192, %r193 }, { %r196, %r197 }, { %r224, %r225, %r226, %r227 };
	// end inline asm
	mov.b32 	%r230, %r189;
	mov.b32 	%r231, %r189;
	mov.b32 	%r232, %r189;
	mov.b32 	%r233, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r230, %r231, %r232, %r233 }, { %r190, %r191, %r192, %r193 }, { %r198, %r199 }, { %r230, %r231, %r232, %r233 };
	// end inline asm
	mov.b32 	%r236, %r189;
	mov.b32 	%r237, %r189;
	mov.b32 	%r238, %r189;
	mov.b32 	%r239, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r236, %r237, %r238, %r239 }, { %r190, %r191, %r192, %r193 }, { %r200, %r201 }, { %r236, %r237, %r238, %r239 };
	// end inline asm
	mov.b32 	%r242, %r189;
	mov.b32 	%r243, %r189;
	mov.b32 	%r244, %r189;
	mov.b32 	%r245, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r242, %r243, %r244, %r245 }, { %r202, %r203, %r204, %r205 }, { %r194, %r195 }, { %r242, %r243, %r244, %r245 };
	// end inline asm
	mov.b32 	%r250, %r189;
	mov.b32 	%r251, %r189;
	mov.b32 	%r252, %r189;
	mov.b32 	%r253, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r250, %r251, %r252, %r253 }, { %r202, %r203, %r204, %r205 }, { %r196, %r197 }, { %r250, %r251, %r252, %r253 };
	// end inline asm
	mov.b32 	%r254, %r189;
	mov.b32 	%r255, %r189;
	mov.b32 	%r256, %r189;
	mov.b32 	%r257, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r254, %r255, %r256, %r257 }, { %r202, %r203, %r204, %r205 }, { %r198, %r199 }, { %r254, %r255, %r256, %r257 };
	// end inline asm
	mov.b32 	%r258, %r189;
	mov.b32 	%r259, %r189;
	mov.b32 	%r260, %r189;
	mov.b32 	%r261, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r258, %r259, %r260, %r261 }, { %r202, %r203, %r204, %r205 }, { %r200, %r201 }, { %r258, %r259, %r260, %r261 };
	// end inline asm
	mov.b32 	%r262, %r189;
	mov.b32 	%r263, %r189;
	mov.b32 	%r264, %r189;
	mov.b32 	%r265, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r262, %r263, %r264, %r265 }, { %r206, %r207, %r208, %r209 }, { %r194, %r195 }, { %r262, %r263, %r264, %r265 };
	// end inline asm
	mov.b32 	%r270, %r189;
	mov.b32 	%r271, %r189;
	mov.b32 	%r272, %r189;
	mov.b32 	%r273, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r270, %r271, %r272, %r273 }, { %r206, %r207, %r208, %r209 }, { %r196, %r197 }, { %r270, %r271, %r272, %r273 };
	// end inline asm
	mov.b32 	%r274, %r189;
	mov.b32 	%r275, %r189;
	mov.b32 	%r276, %r189;
	mov.b32 	%r277, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r274, %r275, %r276, %r277 }, { %r206, %r207, %r208, %r209 }, { %r198, %r199 }, { %r274, %r275, %r276, %r277 };
	// end inline asm
	mov.b32 	%r278, %r189;
	mov.b32 	%r279, %r189;
	mov.b32 	%r280, %r189;
	mov.b32 	%r281, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r278, %r279, %r280, %r281 }, { %r206, %r207, %r208, %r209 }, { %r200, %r201 }, { %r278, %r279, %r280, %r281 };
	// end inline asm
	mov.b32 	%r282, %r189;
	mov.b32 	%r283, %r189;
	mov.b32 	%r284, %r189;
	mov.b32 	%r285, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r282, %r283, %r284, %r285 }, { %r210, %r211, %r212, %r213 }, { %r194, %r195 }, { %r282, %r283, %r284, %r285 };
	// end inline asm
	mov.b32 	%r290, %r189;
	mov.b32 	%r291, %r189;
	mov.b32 	%r292, %r189;
	mov.b32 	%r293, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r290, %r291, %r292, %r293 }, { %r210, %r211, %r212, %r213 }, { %r196, %r197 }, { %r290, %r291, %r292, %r293 };
	// end inline asm
	mov.b32 	%r294, %r189;
	mov.b32 	%r295, %r189;
	mov.b32 	%r296, %r189;
	mov.b32 	%r297, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r294, %r295, %r296, %r297 }, { %r210, %r211, %r212, %r213 }, { %r198, %r199 }, { %r294, %r295, %r296, %r297 };
	// end inline asm
	mov.b32 	%r298, %r189;
	mov.b32 	%r299, %r189;
	mov.b32 	%r300, %r189;
	mov.b32 	%r301, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r298, %r299, %r300, %r301 }, { %r210, %r211, %r212, %r213 }, { %r200, %r201 }, { %r298, %r299, %r300, %r301 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r214, %r215, %r216, %r217 }, { %r218, %r219, %r220, %r221 }, { %r222, %r223 }, { %r214, %r215, %r216, %r217 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r224, %r225, %r226, %r227 }, { %r218, %r219, %r220, %r221 }, { %r228, %r229 }, { %r224, %r225, %r226, %r227 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r230, %r231, %r232, %r233 }, { %r218, %r219, %r220, %r221 }, { %r234, %r235 }, { %r230, %r231, %r232, %r233 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r236, %r237, %r238, %r239 }, { %r218, %r219, %r220, %r221 }, { %r240, %r241 }, { %r236, %r237, %r238, %r239 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r242, %r243, %r244, %r245 }, { %r246, %r247, %r248, %r249 }, { %r222, %r223 }, { %r242, %r243, %r244, %r245 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r250, %r251, %r252, %r253 }, { %r246, %r247, %r248, %r249 }, { %r228, %r229 }, { %r250, %r251, %r252, %r253 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r254, %r255, %r256, %r257 }, { %r246, %r247, %r248, %r249 }, { %r234, %r235 }, { %r254, %r255, %r256, %r257 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r258, %r259, %r260, %r261 }, { %r246, %r247, %r248, %r249 }, { %r240, %r241 }, { %r258, %r259, %r260, %r261 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r262, %r263, %r264, %r265 }, { %r266, %r267, %r268, %r269 }, { %r222, %r223 }, { %r262, %r263, %r264, %r265 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r270, %r271, %r272, %r273 }, { %r266, %r267, %r268, %r269 }, { %r228, %r229 }, { %r270, %r271, %r272, %r273 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r274, %r275, %r276, %r277 }, { %r266, %r267, %r268, %r269 }, { %r234, %r235 }, { %r274, %r275, %r276, %r277 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r278, %r279, %r280, %r281 }, { %r266, %r267, %r268, %r269 }, { %r240, %r241 }, { %r278, %r279, %r280, %r281 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r282, %r283, %r284, %r285 }, { %r286, %r287, %r288, %r289 }, { %r222, %r223 }, { %r282, %r283, %r284, %r285 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r290, %r291, %r292, %r293 }, { %r286, %r287, %r288, %r289 }, { %r228, %r229 }, { %r290, %r291, %r292, %r293 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r294, %r295, %r296, %r297 }, { %r286, %r287, %r288, %r289 }, { %r234, %r235 }, { %r294, %r295, %r296, %r297 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r298, %r299, %r300, %r301 }, { %r286, %r287, %r288, %r289 }, { %r240, %r241 }, { %r298, %r299, %r300, %r301 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r214, %r215, %r216, %r217 }, { %r302, %r303, %r304, %r305 }, { %r306, %r307 }, { %r214, %r215, %r216, %r217 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r224, %r225, %r226, %r227 }, { %r302, %r303, %r304, %r305 }, { %r308, %r309 }, { %r224, %r225, %r226, %r227 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r230, %r231, %r232, %r233 }, { %r302, %r303, %r304, %r305 }, { %r310, %r311 }, { %r230, %r231, %r232, %r233 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r236, %r237, %r238, %r239 }, { %r302, %r303, %r304, %r305 }, { %r312, %r313 }, { %r236, %r237, %r238, %r239 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r242, %r243, %r244, %r245 }, { %r314, %r315, %r316, %r317 }, { %r306, %r307 }, { %r242, %r243, %r244, %r245 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r250, %r251, %r252, %r253 }, { %r314, %r315, %r316, %r317 }, { %r308, %r309 }, { %r250, %r251, %r252, %r253 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r254, %r255, %r256, %r257 }, { %r314, %r315, %r316, %r317 }, { %r310, %r311 }, { %r254, %r255, %r256, %r257 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r258, %r259, %r260, %r261 }, { %r314, %r315, %r316, %r317 }, { %r312, %r313 }, { %r258, %r259, %r260, %r261 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r262, %r263, %r264, %r265 }, { %r318, %r319, %r320, %r321 }, { %r306, %r307 }, { %r262, %r263, %r264, %r265 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r270, %r271, %r272, %r273 }, { %r318, %r319, %r320, %r321 }, { %r308, %r309 }, { %r270, %r271, %r272, %r273 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r274, %r275, %r276, %r277 }, { %r318, %r319, %r320, %r321 }, { %r310, %r311 }, { %r274, %r275, %r276, %r277 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r278, %r279, %r280, %r281 }, { %r318, %r319, %r320, %r321 }, { %r312, %r313 }, { %r278, %r279, %r280, %r281 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r282, %r283, %r284, %r285 }, { %r322, %r323, %r324, %r325 }, { %r306, %r307 }, { %r282, %r283, %r284, %r285 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r290, %r291, %r292, %r293 }, { %r322, %r323, %r324, %r325 }, { %r308, %r309 }, { %r290, %r291, %r292, %r293 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r294, %r295, %r296, %r297 }, { %r322, %r323, %r324, %r325 }, { %r310, %r311 }, { %r294, %r295, %r296, %r297 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r298, %r299, %r300, %r301 }, { %r322, %r323, %r324, %r325 }, { %r312, %r313 }, { %r298, %r299, %r300, %r301 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r214, %r215, %r216, %r217 }, { %r326, %r327, %r328, %r329 }, { %r330, %r331 }, { %r214, %r215, %r216, %r217 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r224, %r225, %r226, %r227 }, { %r326, %r327, %r328, %r329 }, { %r332, %r333 }, { %r224, %r225, %r226, %r227 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r230, %r231, %r232, %r233 }, { %r326, %r327, %r328, %r329 }, { %r334, %r335 }, { %r230, %r231, %r232, %r233 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r236, %r237, %r238, %r239 }, { %r326, %r327, %r328, %r329 }, { %r336, %r337 }, { %r236, %r237, %r238, %r239 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r242, %r243, %r244, %r245 }, { %r338, %r339, %r340, %r341 }, { %r330, %r331 }, { %r242, %r243, %r244, %r245 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r250, %r251, %r252, %r253 }, { %r338, %r339, %r340, %r341 }, { %r332, %r333 }, { %r250, %r251, %r252, %r253 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r254, %r255, %r256, %r257 }, { %r338, %r339, %r340, %r341 }, { %r334, %r335 }, { %r254, %r255, %r256, %r257 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r258, %r259, %r260, %r261 }, { %r338, %r339, %r340, %r341 }, { %r336, %r337 }, { %r258, %r259, %r260, %r261 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r262, %r263, %r264, %r265 }, { %r342, %r343, %r344, %r345 }, { %r330, %r331 }, { %r262, %r263, %r264, %r265 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r270, %r271, %r272, %r273 }, { %r342, %r343, %r344, %r345 }, { %r332, %r333 }, { %r270, %r271, %r272, %r273 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r274, %r275, %r276, %r277 }, { %r342, %r343, %r344, %r345 }, { %r334, %r335 }, { %r274, %r275, %r276, %r277 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r278, %r279, %r280, %r281 }, { %r342, %r343, %r344, %r345 }, { %r336, %r337 }, { %r278, %r279, %r280, %r281 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r282, %r283, %r284, %r285 }, { %r346, %r347, %r348, %r349 }, { %r330, %r331 }, { %r282, %r283, %r284, %r285 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r290, %r291, %r292, %r293 }, { %r346, %r347, %r348, %r349 }, { %r332, %r333 }, { %r290, %r291, %r292, %r293 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r294, %r295, %r296, %r297 }, { %r346, %r347, %r348, %r349 }, { %r334, %r335 }, { %r294, %r295, %r296, %r297 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r298, %r299, %r300, %r301 }, { %r346, %r347, %r348, %r349 }, { %r336, %r337 }, { %r298, %r299, %r300, %r301 };
	// end inline asm
	.loc	1 160 79                        // sk07_lm_head.py:160:79
	cvt.rn.f32.s32 	%r384, %r298;
	cvt.rn.f32.s32 	%r385, %r299;
	cvt.rn.f32.s32 	%r386, %r300;
	cvt.rn.f32.s32 	%r387, %r301;
	cvt.rn.f32.s32 	%r388, %r294;
	cvt.rn.f32.s32 	%r389, %r295;
	cvt.rn.f32.s32 	%r390, %r296;
	cvt.rn.f32.s32 	%r391, %r297;
	cvt.rn.f32.s32 	%r392, %r290;
	cvt.rn.f32.s32 	%r393, %r291;
	cvt.rn.f32.s32 	%r394, %r292;
	cvt.rn.f32.s32 	%r395, %r293;
	cvt.rn.f32.s32 	%r396, %r282;
	cvt.rn.f32.s32 	%r397, %r283;
	cvt.rn.f32.s32 	%r398, %r284;
	cvt.rn.f32.s32 	%r399, %r285;
	cvt.rn.f32.s32 	%r400, %r278;
	cvt.rn.f32.s32 	%r401, %r279;
	cvt.rn.f32.s32 	%r402, %r280;
	cvt.rn.f32.s32 	%r403, %r281;
	cvt.rn.f32.s32 	%r404, %r274;
	cvt.rn.f32.s32 	%r405, %r275;
	cvt.rn.f32.s32 	%r406, %r276;
	cvt.rn.f32.s32 	%r407, %r277;
	cvt.rn.f32.s32 	%r408, %r270;
	cvt.rn.f32.s32 	%r409, %r271;
	cvt.rn.f32.s32 	%r410, %r272;
	cvt.rn.f32.s32 	%r411, %r273;
	cvt.rn.f32.s32 	%r412, %r262;
	cvt.rn.f32.s32 	%r413, %r263;
	cvt.rn.f32.s32 	%r414, %r264;
	cvt.rn.f32.s32 	%r415, %r265;
	cvt.rn.f32.s32 	%r416, %r258;
	cvt.rn.f32.s32 	%r417, %r259;
	cvt.rn.f32.s32 	%r418, %r260;
	cvt.rn.f32.s32 	%r419, %r261;
	cvt.rn.f32.s32 	%r420, %r254;
	cvt.rn.f32.s32 	%r421, %r255;
	cvt.rn.f32.s32 	%r422, %r256;
	cvt.rn.f32.s32 	%r423, %r257;
	cvt.rn.f32.s32 	%r424, %r250;
	cvt.rn.f32.s32 	%r425, %r251;
	cvt.rn.f32.s32 	%r426, %r252;
	cvt.rn.f32.s32 	%r427, %r253;
	cvt.rn.f32.s32 	%r428, %r242;
	cvt.rn.f32.s32 	%r429, %r243;
	cvt.rn.f32.s32 	%r430, %r244;
	cvt.rn.f32.s32 	%r431, %r245;
	cvt.rn.f32.s32 	%r432, %r236;
	cvt.rn.f32.s32 	%r433, %r237;
	cvt.rn.f32.s32 	%r434, %r238;
	cvt.rn.f32.s32 	%r435, %r239;
	cvt.rn.f32.s32 	%r436, %r230;
	cvt.rn.f32.s32 	%r437, %r231;
	cvt.rn.f32.s32 	%r438, %r232;
	cvt.rn.f32.s32 	%r439, %r233;
	cvt.rn.f32.s32 	%r440, %r224;
	cvt.rn.f32.s32 	%r441, %r225;
	cvt.rn.f32.s32 	%r442, %r226;
	cvt.rn.f32.s32 	%r443, %r227;
	cvt.rn.f32.s32 	%r444, %r214;
	cvt.rn.f32.s32 	%r445, %r215;
	cvt.rn.f32.s32 	%r446, %r216;
	cvt.rn.f32.s32 	%r447, %r217;
	.loc	1 160 15                        // sk07_lm_head.py:160:15
	fma.rn.f32 	%r909, %r369, %r447, %r909;
	fma.rn.f32 	%r908, %r368, %r446, %r908;
	fma.rn.f32 	%r907, %r369, %r445, %r907;
	fma.rn.f32 	%r906, %r368, %r444, %r906;
	fma.rn.f32 	%r913, %r371, %r443, %r913;
	fma.rn.f32 	%r912, %r370, %r442, %r912;
	fma.rn.f32 	%r911, %r371, %r441, %r911;
	fma.rn.f32 	%r910, %r370, %r440, %r910;
	fma.rn.f32 	%r917, %r373, %r439, %r917;
	fma.rn.f32 	%r916, %r372, %r438, %r916;
	fma.rn.f32 	%r915, %r373, %r437, %r915;
	fma.rn.f32 	%r914, %r372, %r436, %r914;
	fma.rn.f32 	%r921, %r375, %r435, %r921;
	fma.rn.f32 	%r920, %r374, %r434, %r920;
	fma.rn.f32 	%r919, %r375, %r433, %r919;
	fma.rn.f32 	%r918, %r374, %r432, %r918;
	fma.rn.f32 	%r925, %r369, %r431, %r925;
	fma.rn.f32 	%r924, %r368, %r430, %r924;
	fma.rn.f32 	%r923, %r369, %r429, %r923;
	fma.rn.f32 	%r922, %r368, %r428, %r922;
	fma.rn.f32 	%r929, %r371, %r427, %r929;
	fma.rn.f32 	%r928, %r370, %r426, %r928;
	fma.rn.f32 	%r927, %r371, %r425, %r927;
	fma.rn.f32 	%r926, %r370, %r424, %r926;
	fma.rn.f32 	%r933, %r373, %r423, %r933;
	fma.rn.f32 	%r932, %r372, %r422, %r932;
	fma.rn.f32 	%r931, %r373, %r421, %r931;
	fma.rn.f32 	%r930, %r372, %r420, %r930;
	fma.rn.f32 	%r937, %r375, %r419, %r937;
	fma.rn.f32 	%r936, %r374, %r418, %r936;
	fma.rn.f32 	%r935, %r375, %r417, %r935;
	fma.rn.f32 	%r934, %r374, %r416, %r934;
	fma.rn.f32 	%r941, %r369, %r415, %r941;
	fma.rn.f32 	%r940, %r368, %r414, %r940;
	fma.rn.f32 	%r939, %r369, %r413, %r939;
	fma.rn.f32 	%r938, %r368, %r412, %r938;
	fma.rn.f32 	%r945, %r371, %r411, %r945;
	fma.rn.f32 	%r944, %r370, %r410, %r944;
	fma.rn.f32 	%r943, %r371, %r409, %r943;
	fma.rn.f32 	%r942, %r370, %r408, %r942;
	fma.rn.f32 	%r949, %r373, %r407, %r949;
	fma.rn.f32 	%r948, %r372, %r406, %r948;
	fma.rn.f32 	%r947, %r373, %r405, %r947;
	fma.rn.f32 	%r946, %r372, %r404, %r946;
	fma.rn.f32 	%r953, %r375, %r403, %r953;
	fma.rn.f32 	%r952, %r374, %r402, %r952;
	fma.rn.f32 	%r951, %r375, %r401, %r951;
	fma.rn.f32 	%r950, %r374, %r400, %r950;
	fma.rn.f32 	%r957, %r369, %r399, %r957;
	fma.rn.f32 	%r956, %r368, %r398, %r956;
	fma.rn.f32 	%r955, %r369, %r397, %r955;
	fma.rn.f32 	%r954, %r368, %r396, %r954;
	fma.rn.f32 	%r961, %r371, %r395, %r961;
	fma.rn.f32 	%r960, %r370, %r394, %r960;
	fma.rn.f32 	%r959, %r371, %r393, %r959;
	fma.rn.f32 	%r958, %r370, %r392, %r958;
	fma.rn.f32 	%r965, %r373, %r391, %r965;
	fma.rn.f32 	%r964, %r372, %r390, %r964;
	fma.rn.f32 	%r963, %r373, %r389, %r963;
	fma.rn.f32 	%r962, %r372, %r388, %r962;
	fma.rn.f32 	%r969, %r375, %r387, %r969;
	fma.rn.f32 	%r968, %r374, %r386, %r968;
	fma.rn.f32 	%r967, %r375, %r385, %r967;
	fma.rn.f32 	%r966, %r374, %r384, %r966;
	.loc	1 161 18                        // sk07_lm_head.py:161:18
	add.s64 	%rd101, %rd27, %rd227;
	add.s64 	%rd102, %rd26, %rd227;
	add.s64 	%rd103, %rd25, %rd227;
	.loc	1 162 18                        // sk07_lm_head.py:162:18
	add.s64 	%rd104, %rd24, %rd227;
	add.s64 	%rd105, %rd23, %rd227;
	add.s64 	%rd106, %rd22, %rd227;
	add.s64 	%rd107, %rd21, %rd227;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	add.s64 	%rd108, %rd20, %rd227;
	add.s32 	%r448, %r903, 1;
	setp.gt.s32 	%p5, %r448, 1;
	selp.b32 	%r903, 0, %r448, %p5;
	.loc	1 160 30                        // sk07_lm_head.py:160:30
	shl.b32 	%r449, %r903, 14;
	bar.sync 	0;
	add.s32 	%r350, %r56, %r449;
	selp.b32 	%r351, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r350 + 0 ], [ %rd101 + 0 ], 0x10, %r351;
	// end inline asm
	add.s32 	%r352, %r350, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r352 + 0 ], [ %rd102 + 0 ], 0x10, %r351;
	// end inline asm
	add.s32 	%r353, %r350, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r353 + 0 ], [ %rd103 + 0 ], 0x10, %r351;
	// end inline asm
	add.s32 	%r354, %r350, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r354 + 0 ], [ %rd104 + 0 ], 0x10, %r351;
	// end inline asm
	cp.async.commit_group;
	.loc	1 160 47                        // sk07_lm_head.py:160:47
	add.s32 	%r355, %r350, 32768;
	// begin inline asm
	cp.async.cg.shared.global [ %r355 + 0 ], [ %rd105 + 0 ], 0x10, %r351;
	// end inline asm
	add.s32 	%r356, %r350, 36864;
	// begin inline asm
	cp.async.cg.shared.global [ %r356 + 0 ], [ %rd106 + 0 ], 0x10, %r351;
	// end inline asm
	add.s32 	%r357, %r350, 40960;
	// begin inline asm
	cp.async.cg.shared.global [ %r357 + 0 ], [ %rd107 + 0 ], 0x10, %r351;
	// end inline asm
	add.s32 	%r358, %r350, 45056;
	// begin inline asm
	cp.async.cg.shared.global [ %r358 + 0 ], [ %rd108 + 0 ], 0x10, %r351;
	// end inline asm
	cp.async.commit_group;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	add.s64 	%rd228, %rd228, 1;
	add.s64 	%rd227, %rd227, 128;
	add.s32 	%r901, %r901, %r27;
	setp.ne.b64 	%p6, %rd19, %rd227;
	@%p6 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_4;
$L__BB0_1:                              // %.._crit_edge_crit_edge
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	shl.b32 	%r905, %r2, 1;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	and.b32 	%r904, %r2, 16;
	mov.b32 	%r906, 0f00000000;
	mov.b32 	%r907, %r906;
	mov.b32 	%r908, %r906;
	mov.b32 	%r909, %r906;
	mov.b32 	%r910, %r906;
	mov.b32 	%r911, %r906;
	mov.b32 	%r912, %r906;
	mov.b32 	%r913, %r906;
	mov.b32 	%r914, %r906;
	mov.b32 	%r915, %r906;
	mov.b32 	%r916, %r906;
	mov.b32 	%r917, %r906;
	mov.b32 	%r918, %r906;
	mov.b32 	%r919, %r906;
	mov.b32 	%r920, %r906;
	mov.b32 	%r921, %r906;
	mov.b32 	%r922, %r906;
	mov.b32 	%r923, %r906;
	mov.b32 	%r924, %r906;
	mov.b32 	%r925, %r906;
	mov.b32 	%r926, %r906;
	mov.b32 	%r927, %r906;
	mov.b32 	%r928, %r906;
	mov.b32 	%r929, %r906;
	mov.b32 	%r930, %r906;
	mov.b32 	%r931, %r906;
	mov.b32 	%r932, %r906;
	mov.b32 	%r933, %r906;
	mov.b32 	%r934, %r906;
	mov.b32 	%r935, %r906;
	mov.b32 	%r936, %r906;
	mov.b32 	%r937, %r906;
	mov.b32 	%r938, %r906;
	mov.b32 	%r939, %r906;
	mov.b32 	%r940, %r906;
	mov.b32 	%r941, %r906;
	mov.b32 	%r942, %r906;
	mov.b32 	%r943, %r906;
	mov.b32 	%r944, %r906;
	mov.b32 	%r945, %r906;
	mov.b32 	%r946, %r906;
	mov.b32 	%r947, %r906;
	mov.b32 	%r948, %r906;
	mov.b32 	%r949, %r906;
	mov.b32 	%r950, %r906;
	mov.b32 	%r951, %r906;
	mov.b32 	%r952, %r906;
	mov.b32 	%r953, %r906;
	mov.b32 	%r954, %r906;
	mov.b32 	%r955, %r906;
	mov.b32 	%r956, %r906;
	mov.b32 	%r957, %r906;
	mov.b32 	%r958, %r906;
	mov.b32 	%r959, %r906;
	mov.b32 	%r960, %r906;
	mov.b32 	%r961, %r906;
	mov.b32 	%r962, %r906;
	mov.b32 	%r963, %r906;
	mov.b32 	%r964, %r906;
	mov.b32 	%r965, %r906;
	mov.b32 	%r966, %r906;
	mov.b32 	%r967, %r906;
	mov.b32 	%r968, %r906;
	mov.b32 	%r969, %r906;
$L__BB0_4:                              // %._crit_edge
	.loc	1 141 45                        // sk07_lm_head.py:141:45
	shl.b32 	%r572, %r7, 3;
	.loc	1 142 32                        // sk07_lm_head.py:142:32
	or.b32 	%r573, %r12, %r572;
	.loc	1 141 45                        // sk07_lm_head.py:141:45
	and.b32 	%r574, %r2, 128;
	shr.u32 	%r575, %r574, 3;
	shr.u32 	%r576, %r2, 2;
	bfe.u32 	%r577, %r2, 2, 3;
	or.b32 	%r578, %r575, %r577;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r579, %r578, %r1;
	or.b32 	%r580, %r579, 104;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r581, %r580, %r21;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r582, %r579, 96;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r583, %r582, %r21;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r584, %r579, 72;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r585, %r584, %r21;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r586, %r579, 64;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r587, %r586, %r21;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r588, %r579, 40;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r589, %r588, %r21;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r590, %r579, 32;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r591, %r590, %r21;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r592, %r579, 8;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r593, %r592, %r21;
	rem.s32 	%r594, %r579, %r21;
	.loc	1 141 45                        // sk07_lm_head.py:141:45
	shr.u32 	%r595, %r2, 4;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r596, %r595, %r1;
	or.b32 	%r597, %r596, 112;
	.loc	1 141 45                        // sk07_lm_head.py:141:45
	bfe.u32 	%r598, %r2, 4, 4;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r599, %r598, %r1;
	or.b32 	%r600, %r599, 96;
	or.b32 	%r601, %r599, 80;
	or.b32 	%r602, %r599, 64;
	or.b32 	%r603, %r596, 48;
	or.b32 	%r604, %r599, 32;
	or.b32 	%r605, %r599, 16;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 164 38                        // sk07_lm_head.py:164:38
	mad.wide.s32 	%rd110, %r594, 4, %rd32;
	mad.wide.s32 	%rd111, %r593, 4, %rd32;
	mad.wide.s32 	%rd112, %r591, 4, %rd32;
	mad.wide.s32 	%rd113, %r589, 4, %rd32;
	mad.wide.s32 	%rd114, %r587, 4, %rd32;
	mad.wide.s32 	%rd115, %r585, 4, %rd32;
	mad.wide.s32 	%rd116, %r583, 4, %rd32;
	mad.wide.s32 	%rd117, %r581, 4, %rd32;
	.loc	1 164 24                        // sk07_lm_head.py:164:24
	// begin inline asm
	mov.u32 %r450, 0x0;
	ld.global.b32 { %r450 }, [ %rd110 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r451, 0x0;
	ld.global.b32 { %r451 }, [ %rd111 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r452, 0x0;
	ld.global.b32 { %r452 }, [ %rd112 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r453, 0x0;
	ld.global.b32 { %r453 }, [ %rd113 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r454, 0x0;
	ld.global.b32 { %r454 }, [ %rd114 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r455, 0x0;
	ld.global.b32 { %r455 }, [ %rd115 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r456, 0x0;
	ld.global.b32 { %r456 }, [ %rd116 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r457, 0x0;
	ld.global.b32 { %r457 }, [ %rd117 + 0 ];
	// end inline asm
	.loc	1 165 38                        // sk07_lm_head.py:165:38
	mad.wide.s32 	%rd118, %r32, 4, %rd33;
	mad.wide.s32 	%rd119, %r33, 4, %rd33;
	mad.wide.s32 	%rd120, %r34, 4, %rd33;
	mad.wide.s32 	%rd121, %r35, 4, %rd33;
	mad.wide.s32 	%rd122, %r36, 4, %rd33;
	mad.wide.s32 	%rd123, %r37, 4, %rd33;
	mad.wide.s32 	%rd124, %r38, 4, %rd33;
	mad.wide.s32 	%rd125, %r39, 4, %rd33;
	.loc	1 165 24                        // sk07_lm_head.py:165:24
	// begin inline asm
	mov.u32 %r458, 0x0;
	ld.global.b32 { %r458 }, [ %rd118 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r459, 0x0;
	ld.global.b32 { %r459 }, [ %rd119 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r460, 0x0;
	ld.global.b32 { %r460 }, [ %rd120 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r461, 0x0;
	ld.global.b32 { %r461 }, [ %rd121 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r462, 0x0;
	ld.global.b32 { %r462 }, [ %rd122 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r463, 0x0;
	ld.global.b32 { %r463 }, [ %rd123 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r464, 0x0;
	ld.global.b32 { %r464 }, [ %rd124 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r465, 0x0;
	ld.global.b32 { %r465 }, [ %rd125 + 0 ];
	// end inline asm
	.loc	1 166 49                        // sk07_lm_head.py:166:49
	mul.lo.s32 	%r606, %r8, %r25;
	mul.lo.s32 	%r607, %r9, %r25;
	mul.lo.s32 	%r608, %r10, %r25;
	mul.lo.s32 	%r609, %r11, %r25;
	.loc	1 166 31                        // sk07_lm_head.py:166:31
	mad.wide.s32 	%rd198, %r606, 2, %rd31;
	mad.wide.s32 	%rd199, %r607, 2, %rd31;
	mad.wide.s32 	%rd200, %r608, 2, %rd31;
	mad.wide.s32 	%rd201, %r609, 2, %rd31;
	.loc	1 166 80                        // sk07_lm_head.py:166:80
	mul.lo.s32 	%r610, %r40, %r26;
	mul.lo.s32 	%r611, %r41, %r26;
	mul.lo.s32 	%r612, %r42, %r26;
	mul.lo.s32 	%r613, %r43, %r26;
	mul.lo.s32 	%r614, %r44, %r26;
	mul.lo.s32 	%r615, %r45, %r26;
	mul.lo.s32 	%r616, %r46, %r26;
	mul.lo.s32 	%r617, %r47, %r26;
	mul.lo.s32 	%r618, %r48, %r26;
	mul.lo.s32 	%r619, %r49, %r26;
	mul.lo.s32 	%r620, %r50, %r26;
	mul.lo.s32 	%r621, %r51, %r26;
	mul.lo.s32 	%r622, %r52, %r26;
	mul.lo.s32 	%r623, %r53, %r26;
	mul.lo.s32 	%r624, %r54, %r26;
	mul.lo.s32 	%r625, %r55, %r26;
	.loc	1 166 64                        // sk07_lm_head.py:166:64
	mul.wide.s32 	%rd202, %r610, 2;
	add.s64 	%rd126, %rd198, %rd202;
	mul.wide.s32 	%rd203, %r611, 2;
	add.s64 	%rd127, %rd198, %rd203;
	mul.wide.s32 	%rd204, %r612, 2;
	add.s64 	%rd128, %rd198, %rd204;
	mul.wide.s32 	%rd205, %r613, 2;
	add.s64 	%rd129, %rd198, %rd205;
	mul.wide.s32 	%rd206, %r614, 2;
	add.s64 	%rd130, %rd198, %rd206;
	mul.wide.s32 	%rd207, %r615, 2;
	add.s64 	%rd131, %rd198, %rd207;
	mul.wide.s32 	%rd208, %r616, 2;
	add.s64 	%rd132, %rd198, %rd208;
	mul.wide.s32 	%rd209, %r617, 2;
	add.s64 	%rd133, %rd198, %rd209;
	mul.wide.s32 	%rd210, %r618, 2;
	add.s64 	%rd134, %rd198, %rd210;
	mul.wide.s32 	%rd211, %r619, 2;
	add.s64 	%rd135, %rd198, %rd211;
	mul.wide.s32 	%rd212, %r620, 2;
	add.s64 	%rd136, %rd198, %rd212;
	mul.wide.s32 	%rd213, %r621, 2;
	add.s64 	%rd137, %rd198, %rd213;
	mul.wide.s32 	%rd214, %r622, 2;
	add.s64 	%rd138, %rd198, %rd214;
	mul.wide.s32 	%rd215, %r623, 2;
	add.s64 	%rd139, %rd198, %rd215;
	mul.wide.s32 	%rd216, %r624, 2;
	add.s64 	%rd140, %rd198, %rd216;
	mul.wide.s32 	%rd217, %r625, 2;
	add.s64 	%rd141, %rd198, %rd217;
	add.s64 	%rd142, %rd199, %rd202;
	add.s64 	%rd143, %rd199, %rd203;
	add.s64 	%rd144, %rd199, %rd204;
	add.s64 	%rd145, %rd199, %rd205;
	add.s64 	%rd146, %rd199, %rd206;
	add.s64 	%rd147, %rd199, %rd207;
	add.s64 	%rd148, %rd199, %rd208;
	add.s64 	%rd149, %rd199, %rd209;
	add.s64 	%rd150, %rd199, %rd210;
	add.s64 	%rd151, %rd199, %rd211;
	add.s64 	%rd152, %rd199, %rd212;
	add.s64 	%rd153, %rd199, %rd213;
	add.s64 	%rd154, %rd199, %rd214;
	add.s64 	%rd155, %rd199, %rd215;
	add.s64 	%rd156, %rd199, %rd216;
	add.s64 	%rd157, %rd199, %rd217;
	add.s64 	%rd158, %rd200, %rd202;
	add.s64 	%rd159, %rd200, %rd203;
	add.s64 	%rd160, %rd200, %rd204;
	add.s64 	%rd161, %rd200, %rd205;
	add.s64 	%rd162, %rd200, %rd206;
	add.s64 	%rd163, %rd200, %rd207;
	add.s64 	%rd164, %rd200, %rd208;
	add.s64 	%rd165, %rd200, %rd209;
	add.s64 	%rd166, %rd200, %rd210;
	add.s64 	%rd167, %rd200, %rd211;
	add.s64 	%rd168, %rd200, %rd212;
	add.s64 	%rd169, %rd200, %rd213;
	add.s64 	%rd170, %rd200, %rd214;
	add.s64 	%rd171, %rd200, %rd215;
	add.s64 	%rd172, %rd200, %rd216;
	add.s64 	%rd173, %rd200, %rd217;
	add.s64 	%rd174, %rd201, %rd202;
	add.s64 	%rd175, %rd201, %rd203;
	add.s64 	%rd176, %rd201, %rd204;
	add.s64 	%rd177, %rd201, %rd205;
	add.s64 	%rd178, %rd201, %rd206;
	add.s64 	%rd179, %rd201, %rd207;
	add.s64 	%rd180, %rd201, %rd208;
	add.s64 	%rd181, %rd201, %rd209;
	add.s64 	%rd182, %rd201, %rd210;
	add.s64 	%rd183, %rd201, %rd211;
	add.s64 	%rd184, %rd201, %rd212;
	add.s64 	%rd185, %rd201, %rd213;
	add.s64 	%rd186, %rd201, %rd214;
	add.s64 	%rd187, %rd201, %rd215;
	add.s64 	%rd188, %rd201, %rd216;
	add.s64 	%rd189, %rd201, %rd217;
	.loc	1 166 19                        // sk07_lm_head.py:166:19
	// begin inline asm
	mov.u16 %rs17, 0x0;
	ld.global.b16 { %rs17 }, [ %rd126 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs18, 0x0;
	ld.global.b16 { %rs18 }, [ %rd127 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs19, 0x0;
	ld.global.b16 { %rs19 }, [ %rd128 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs20, 0x0;
	ld.global.b16 { %rs20 }, [ %rd129 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs21, 0x0;
	ld.global.b16 { %rs21 }, [ %rd130 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs22, 0x0;
	ld.global.b16 { %rs22 }, [ %rd131 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs23, 0x0;
	ld.global.b16 { %rs23 }, [ %rd132 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs24, 0x0;
	ld.global.b16 { %rs24 }, [ %rd133 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs25, 0x0;
	ld.global.b16 { %rs25 }, [ %rd134 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs26, 0x0;
	ld.global.b16 { %rs26 }, [ %rd135 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs27, 0x0;
	ld.global.b16 { %rs27 }, [ %rd136 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs28, 0x0;
	ld.global.b16 { %rs28 }, [ %rd137 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs29, 0x0;
	ld.global.b16 { %rs29 }, [ %rd138 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs30, 0x0;
	ld.global.b16 { %rs30 }, [ %rd139 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs31, 0x0;
	ld.global.b16 { %rs31 }, [ %rd140 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs32, 0x0;
	ld.global.b16 { %rs32 }, [ %rd141 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs33, 0x0;
	ld.global.b16 { %rs33 }, [ %rd142 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs34, 0x0;
	ld.global.b16 { %rs34 }, [ %rd143 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs35, 0x0;
	ld.global.b16 { %rs35 }, [ %rd144 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs36, 0x0;
	ld.global.b16 { %rs36 }, [ %rd145 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs37, 0x0;
	ld.global.b16 { %rs37 }, [ %rd146 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs38, 0x0;
	ld.global.b16 { %rs38 }, [ %rd147 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs39, 0x0;
	ld.global.b16 { %rs39 }, [ %rd148 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs40, 0x0;
	ld.global.b16 { %rs40 }, [ %rd149 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs41, 0x0;
	ld.global.b16 { %rs41 }, [ %rd150 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs42, 0x0;
	ld.global.b16 { %rs42 }, [ %rd151 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs43, 0x0;
	ld.global.b16 { %rs43 }, [ %rd152 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs44, 0x0;
	ld.global.b16 { %rs44 }, [ %rd153 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs45, 0x0;
	ld.global.b16 { %rs45 }, [ %rd154 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs46, 0x0;
	ld.global.b16 { %rs46 }, [ %rd155 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs47, 0x0;
	ld.global.b16 { %rs47 }, [ %rd156 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs48, 0x0;
	ld.global.b16 { %rs48 }, [ %rd157 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs49, 0x0;
	ld.global.b16 { %rs49 }, [ %rd158 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs50, 0x0;
	ld.global.b16 { %rs50 }, [ %rd159 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs51, 0x0;
	ld.global.b16 { %rs51 }, [ %rd160 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs52, 0x0;
	ld.global.b16 { %rs52 }, [ %rd161 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs53, 0x0;
	ld.global.b16 { %rs53 }, [ %rd162 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs54, 0x0;
	ld.global.b16 { %rs54 }, [ %rd163 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs55, 0x0;
	ld.global.b16 { %rs55 }, [ %rd164 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs56, 0x0;
	ld.global.b16 { %rs56 }, [ %rd165 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs57, 0x0;
	ld.global.b16 { %rs57 }, [ %rd166 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs58, 0x0;
	ld.global.b16 { %rs58 }, [ %rd167 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs59, 0x0;
	ld.global.b16 { %rs59 }, [ %rd168 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs60, 0x0;
	ld.global.b16 { %rs60 }, [ %rd169 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs61, 0x0;
	ld.global.b16 { %rs61 }, [ %rd170 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs62, 0x0;
	ld.global.b16 { %rs62 }, [ %rd171 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs63, 0x0;
	ld.global.b16 { %rs63 }, [ %rd172 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs64, 0x0;
	ld.global.b16 { %rs64 }, [ %rd173 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs65, 0x0;
	ld.global.b16 { %rs65 }, [ %rd174 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs66, 0x0;
	ld.global.b16 { %rs66 }, [ %rd175 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs67, 0x0;
	ld.global.b16 { %rs67 }, [ %rd176 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs68, 0x0;
	ld.global.b16 { %rs68 }, [ %rd177 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs69, 0x0;
	ld.global.b16 { %rs69 }, [ %rd178 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs70, 0x0;
	ld.global.b16 { %rs70 }, [ %rd179 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs71, 0x0;
	ld.global.b16 { %rs71 }, [ %rd180 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs72, 0x0;
	ld.global.b16 { %rs72 }, [ %rd181 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs73, 0x0;
	ld.global.b16 { %rs73 }, [ %rd182 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs74, 0x0;
	ld.global.b16 { %rs74 }, [ %rd183 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs75, 0x0;
	ld.global.b16 { %rs75 }, [ %rd184 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs76, 0x0;
	ld.global.b16 { %rs76 }, [ %rd185 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs77, 0x0;
	ld.global.b16 { %rs77 }, [ %rd186 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs78, 0x0;
	ld.global.b16 { %rs78 }, [ %rd187 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs79, 0x0;
	ld.global.b16 { %rs79 }, [ %rd188 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs80, 0x0;
	ld.global.b16 { %rs80 }, [ %rd189 + 0 ];
	// end inline asm
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	shl.b32 	%r626, %r14, 7;
	shl.b32 	%r627, %r3, 1;
	or.b32 	%r628, %r626, %r900;
	xor.b32 	%r629, %r628, %r627;
	add.s32 	%r466, %r177, %r629;
	mov.b32 	%r467, {%rs17, %rs18};
	mov.b32 	%r468, {%rs19, %rs20};
	mov.b32 	%r469, {%rs21, %rs22};
	mov.b32 	%r470, {%rs23, %rs24};
	// begin inline asm
	st.shared.v4.b32 [ %r466 + 0 ], { %r467, %r468, %r469, %r470 };
	// end inline asm
	add.s32 	%r471, %r466, 512;
	mov.b32 	%r472, {%rs25, %rs26};
	mov.b32 	%r473, {%rs27, %rs28};
	mov.b32 	%r474, {%rs29, %rs30};
	mov.b32 	%r475, {%rs31, %rs32};
	// begin inline asm
	st.shared.v4.b32 [ %r471 + 0 ], { %r472, %r473, %r474, %r475 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r630, %r6, 10;
	and.b32 	%r631, %r13, 752;
	and.b32 	%r632, %r905, 288;
	and.b32 	%r633, %r576, 16;
	xor.b32 	%r634, %r631, %r632;
	xor.b32 	%r635, %r634, %r633;
	or.b32 	%r636, %r635, %r630;
	add.s32 	%r637, %r177, %r636;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r638, %r639, %r640, %r641}, [%r637];
	xor.b32 	%r642, %r636, 64;
	add.s32 	%r643, %r177, %r642;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r644, %r645, %r646, %r647}, [%r643];
	bar.sync 	0;
	mov.b32 	%r476, {%rs33, %rs34};
	mov.b32 	%r477, {%rs35, %rs36};
	mov.b32 	%r478, {%rs37, %rs38};
	mov.b32 	%r479, {%rs39, %rs40};
	// begin inline asm
	st.shared.v4.b32 [ %r466 + 0 ], { %r476, %r477, %r478, %r479 };
	// end inline asm
	mov.b32 	%r480, {%rs41, %rs42};
	mov.b32 	%r481, {%rs43, %rs44};
	mov.b32 	%r482, {%rs45, %rs46};
	mov.b32 	%r483, {%rs47, %rs48};
	// begin inline asm
	st.shared.v4.b32 [ %r471 + 0 ], { %r480, %r481, %r482, %r483 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r648, %r649, %r650, %r651}, [%r637];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r652, %r653, %r654, %r655}, [%r643];
	bar.sync 	0;
	mov.b32 	%r484, {%rs49, %rs50};
	mov.b32 	%r485, {%rs51, %rs52};
	mov.b32 	%r486, {%rs53, %rs54};
	mov.b32 	%r487, {%rs55, %rs56};
	// begin inline asm
	st.shared.v4.b32 [ %r466 + 0 ], { %r484, %r485, %r486, %r487 };
	// end inline asm
	mov.b32 	%r488, {%rs57, %rs58};
	mov.b32 	%r489, {%rs59, %rs60};
	mov.b32 	%r490, {%rs61, %rs62};
	mov.b32 	%r491, {%rs63, %rs64};
	// begin inline asm
	st.shared.v4.b32 [ %r471 + 0 ], { %r488, %r489, %r490, %r491 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r656, %r657, %r658, %r659}, [%r637];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r660, %r661, %r662, %r663}, [%r643];
	bar.sync 	0;
	mov.b32 	%r492, {%rs65, %rs66};
	mov.b32 	%r493, {%rs67, %rs68};
	mov.b32 	%r494, {%rs69, %rs70};
	mov.b32 	%r495, {%rs71, %rs72};
	// begin inline asm
	st.shared.v4.b32 [ %r466 + 0 ], { %r492, %r493, %r494, %r495 };
	// end inline asm
	mov.b32 	%r496, {%rs73, %rs74};
	mov.b32 	%r497, {%rs75, %rs76};
	mov.b32 	%r498, {%rs77, %rs78};
	mov.b32 	%r499, {%rs79, %rs80};
	// begin inline asm
	st.shared.v4.b32 [ %r471 + 0 ], { %r496, %r497, %r498, %r499 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r664, %r665, %r666, %r667}, [%r637];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r668, %r669, %r670, %r671}, [%r643];
	.loc	1 173 31                        // sk07_lm_head.py:173:31
	setp.lt.s32 	%p15, %r599, %r21;
	setp.lt.s32 	%p16, %r605, %r21;
	setp.lt.s32 	%p17, %r604, %r21;
	setp.lt.s32 	%p18, %r603, %r21;
	setp.lt.s32 	%p19, %r602, %r21;
	setp.lt.s32 	%p20, %r601, %r21;
	setp.lt.s32 	%p21, %r600, %r21;
	setp.lt.s32 	%p22, %r597, %r21;
	.loc	1 173 54                        // sk07_lm_head.py:173:54
	setp.lt.s32 	%p23, %r573, %r22;
	.loc	1 173 37                        // sk07_lm_head.py:173:37
	and.pred 	%p7, %p15, %p23;
	and.pred 	%p8, %p16, %p23;
	and.pred 	%p9, %p17, %p23;
	and.pred 	%p10, %p18, %p23;
	and.pred 	%p11, %p19, %p23;
	and.pred 	%p12, %p20, %p23;
	and.pred 	%p13, %p21, %p23;
	and.pred 	%p14, %p22, %p23;
	.loc	1 171 35                        // sk07_lm_head.py:171:35
	mul.lo.s32 	%r672, %r599, %r24;
	mul.lo.s32 	%r673, %r605, %r24;
	mul.lo.s32 	%r674, %r604, %r24;
	mul.lo.s32 	%r675, %r603, %r24;
	mul.lo.s32 	%r676, %r602, %r24;
	mul.lo.s32 	%r677, %r601, %r24;
	mul.lo.s32 	%r678, %r600, %r24;
	mul.lo.s32 	%r679, %r597, %r24;
	.loc	1 171 18                        // sk07_lm_head.py:171:18
	mad.wide.s32 	%rd218, %r672, 2, %rd30;
	mad.wide.s32 	%rd219, %r673, 2, %rd30;
	mad.wide.s32 	%rd220, %r674, 2, %rd30;
	mad.wide.s32 	%rd221, %r675, 2, %rd30;
	mad.wide.s32 	%rd222, %r676, 2, %rd30;
	mad.wide.s32 	%rd223, %r677, 2, %rd30;
	mad.wide.s32 	%rd224, %r678, 2, %rd30;
	mad.wide.s32 	%rd225, %r679, 2, %rd30;
	.loc	1 171 50                        // sk07_lm_head.py:171:50
	mul.wide.s32 	%rd226, %r573, 2;
	add.s64 	%rd190, %rd218, %rd226;
	add.s64 	%rd191, %rd219, %rd226;
	add.s64 	%rd192, %rd220, %rd226;
	add.s64 	%rd193, %rd221, %rd226;
	add.s64 	%rd194, %rd222, %rd226;
	add.s64 	%rd195, %rd223, %rd226;
	add.s64 	%rd196, %rd224, %rd226;
	add.s64 	%rd197, %rd225, %rd226;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r680, %r955, %r456;
	mul.f32 	%r681, %r954, %r456;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs81, %rs82}, %r664;
	cvt.f32.bf16 	%r682, %rs82;
	cvt.f32.bf16 	%r683, %rs81;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r684, %r681, %r458, %r683;
	fma.rn.f32 	%r685, %r680, %r459, %r682;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r686, %r907, %r450;
	mul.f32 	%r687, %r906, %r450;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs83, %rs84}, %r638;
	cvt.f32.bf16 	%r688, %rs84;
	cvt.f32.bf16 	%r689, %rs83;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r690, %r687, %r458, %r689;
	fma.rn.f32 	%r691, %r686, %r459, %r688;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r501, %r691, %r690;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r692, %r909, %r451;
	mul.f32 	%r693, %r908, %r451;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs85, %rs86}, %r639;
	cvt.f32.bf16 	%r694, %rs86;
	cvt.f32.bf16 	%r695, %rs85;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r696, %r693, %r458, %r695;
	fma.rn.f32 	%r697, %r692, %r459, %r694;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r506, %r697, %r696;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r698, %r923, %r452;
	mul.f32 	%r699, %r922, %r452;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs87, %rs88}, %r648;
	cvt.f32.bf16 	%r700, %rs88;
	cvt.f32.bf16 	%r701, %rs87;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r702, %r699, %r458, %r701;
	fma.rn.f32 	%r703, %r698, %r459, %r700;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r502, %r703, %r702;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r704, %r925, %r453;
	mul.f32 	%r705, %r924, %r453;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs89, %rs90}, %r649;
	cvt.f32.bf16 	%r706, %rs90;
	cvt.f32.bf16 	%r707, %rs89;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r708, %r705, %r458, %r707;
	fma.rn.f32 	%r709, %r704, %r459, %r706;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r507, %r709, %r708;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r710, %r939, %r454;
	mul.f32 	%r711, %r938, %r454;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs91, %rs92}, %r656;
	cvt.f32.bf16 	%r712, %rs92;
	cvt.f32.bf16 	%r713, %rs91;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r714, %r711, %r458, %r713;
	fma.rn.f32 	%r715, %r710, %r459, %r712;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r503, %r715, %r714;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r716, %r941, %r455;
	mul.f32 	%r717, %r940, %r455;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs93, %rs94}, %r657;
	cvt.f32.bf16 	%r718, %rs94;
	cvt.f32.bf16 	%r719, %rs93;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r720, %r717, %r458, %r719;
	fma.rn.f32 	%r721, %r716, %r459, %r718;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r508, %r721, %r720;
	cvt.rn.bf16x2.f32 	%r504, %r685, %r684;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r722, %r957, %r457;
	mul.f32 	%r723, %r956, %r457;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs95, %rs96}, %r665;
	cvt.f32.bf16 	%r724, %rs96;
	cvt.f32.bf16 	%r725, %rs95;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r726, %r723, %r458, %r725;
	fma.rn.f32 	%r727, %r722, %r459, %r724;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r509, %r727, %r726;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r728, %r959, %r456;
	mul.f32 	%r729, %r958, %r456;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs97, %rs98}, %r666;
	cvt.f32.bf16 	%r730, %rs98;
	cvt.f32.bf16 	%r731, %rs97;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r732, %r729, %r460, %r731;
	fma.rn.f32 	%r733, %r728, %r461, %r730;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r734, %r911, %r450;
	mul.f32 	%r735, %r910, %r450;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs99, %rs100}, %r640;
	cvt.f32.bf16 	%r736, %rs100;
	cvt.f32.bf16 	%r737, %rs99;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r738, %r735, %r460, %r737;
	fma.rn.f32 	%r739, %r734, %r461, %r736;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r521, %r739, %r738;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r740, %r913, %r451;
	mul.f32 	%r741, %r912, %r451;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs101, %rs102}, %r641;
	cvt.f32.bf16 	%r742, %rs102;
	cvt.f32.bf16 	%r743, %rs101;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r744, %r741, %r460, %r743;
	fma.rn.f32 	%r745, %r740, %r461, %r742;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r526, %r745, %r744;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r746, %r927, %r452;
	mul.f32 	%r747, %r926, %r452;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs103, %rs104}, %r650;
	cvt.f32.bf16 	%r748, %rs104;
	cvt.f32.bf16 	%r749, %rs103;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r750, %r747, %r460, %r749;
	fma.rn.f32 	%r751, %r746, %r461, %r748;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r522, %r751, %r750;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r752, %r929, %r453;
	mul.f32 	%r753, %r928, %r453;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs105, %rs106}, %r651;
	cvt.f32.bf16 	%r754, %rs106;
	cvt.f32.bf16 	%r755, %rs105;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r756, %r753, %r460, %r755;
	fma.rn.f32 	%r757, %r752, %r461, %r754;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r527, %r757, %r756;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r758, %r943, %r454;
	mul.f32 	%r759, %r942, %r454;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs107, %rs108}, %r658;
	cvt.f32.bf16 	%r760, %rs108;
	cvt.f32.bf16 	%r761, %rs107;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r762, %r759, %r460, %r761;
	fma.rn.f32 	%r763, %r758, %r461, %r760;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r523, %r763, %r762;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r764, %r945, %r455;
	mul.f32 	%r765, %r944, %r455;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs109, %rs110}, %r659;
	cvt.f32.bf16 	%r766, %rs110;
	cvt.f32.bf16 	%r767, %rs109;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r768, %r765, %r460, %r767;
	fma.rn.f32 	%r769, %r764, %r461, %r766;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r528, %r769, %r768;
	cvt.rn.bf16x2.f32 	%r524, %r733, %r732;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r770, %r961, %r457;
	mul.f32 	%r771, %r960, %r457;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs111, %rs112}, %r667;
	cvt.f32.bf16 	%r772, %rs112;
	cvt.f32.bf16 	%r773, %rs111;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r774, %r771, %r460, %r773;
	fma.rn.f32 	%r775, %r770, %r461, %r772;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r529, %r775, %r774;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r776, %r963, %r456;
	mul.f32 	%r777, %r962, %r456;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs113, %rs114}, %r668;
	cvt.f32.bf16 	%r778, %rs114;
	cvt.f32.bf16 	%r779, %rs113;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r780, %r777, %r462, %r779;
	fma.rn.f32 	%r781, %r776, %r463, %r778;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r782, %r915, %r450;
	mul.f32 	%r783, %r914, %r450;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs115, %rs116}, %r644;
	cvt.f32.bf16 	%r784, %rs116;
	cvt.f32.bf16 	%r785, %rs115;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r786, %r783, %r462, %r785;
	fma.rn.f32 	%r787, %r782, %r463, %r784;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r511, %r787, %r786;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r788, %r917, %r451;
	mul.f32 	%r789, %r916, %r451;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs117, %rs118}, %r645;
	cvt.f32.bf16 	%r790, %rs118;
	cvt.f32.bf16 	%r791, %rs117;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r792, %r789, %r462, %r791;
	fma.rn.f32 	%r793, %r788, %r463, %r790;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r516, %r793, %r792;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r794, %r931, %r452;
	mul.f32 	%r795, %r930, %r452;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs119, %rs120}, %r652;
	cvt.f32.bf16 	%r796, %rs120;
	cvt.f32.bf16 	%r797, %rs119;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r798, %r795, %r462, %r797;
	fma.rn.f32 	%r799, %r794, %r463, %r796;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r512, %r799, %r798;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r800, %r933, %r453;
	mul.f32 	%r801, %r932, %r453;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs121, %rs122}, %r653;
	cvt.f32.bf16 	%r802, %rs122;
	cvt.f32.bf16 	%r803, %rs121;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r804, %r801, %r462, %r803;
	fma.rn.f32 	%r805, %r800, %r463, %r802;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r517, %r805, %r804;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r806, %r947, %r454;
	mul.f32 	%r807, %r946, %r454;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs123, %rs124}, %r660;
	cvt.f32.bf16 	%r808, %rs124;
	cvt.f32.bf16 	%r809, %rs123;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r810, %r807, %r462, %r809;
	fma.rn.f32 	%r811, %r806, %r463, %r808;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r513, %r811, %r810;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r812, %r949, %r455;
	mul.f32 	%r813, %r948, %r455;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs125, %rs126}, %r661;
	cvt.f32.bf16 	%r814, %rs126;
	cvt.f32.bf16 	%r815, %rs125;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r816, %r813, %r462, %r815;
	fma.rn.f32 	%r817, %r812, %r463, %r814;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r518, %r817, %r816;
	cvt.rn.bf16x2.f32 	%r514, %r781, %r780;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r818, %r965, %r457;
	mul.f32 	%r819, %r964, %r457;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs127, %rs128}, %r669;
	cvt.f32.bf16 	%r820, %rs128;
	cvt.f32.bf16 	%r821, %rs127;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r822, %r819, %r462, %r821;
	fma.rn.f32 	%r823, %r818, %r463, %r820;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r519, %r823, %r822;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r824, %r967, %r456;
	mul.f32 	%r825, %r966, %r456;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs129, %rs130}, %r670;
	cvt.f32.bf16 	%r826, %rs130;
	cvt.f32.bf16 	%r827, %rs129;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r828, %r825, %r464, %r827;
	fma.rn.f32 	%r829, %r824, %r465, %r826;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r830, %r919, %r450;
	mul.f32 	%r831, %r918, %r450;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs131, %rs132}, %r646;
	cvt.f32.bf16 	%r832, %rs132;
	cvt.f32.bf16 	%r833, %rs131;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r834, %r831, %r464, %r833;
	fma.rn.f32 	%r835, %r830, %r465, %r832;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r531, %r835, %r834;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r836, %r921, %r451;
	mul.f32 	%r837, %r920, %r451;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs133, %rs134}, %r647;
	cvt.f32.bf16 	%r838, %rs134;
	cvt.f32.bf16 	%r839, %rs133;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r840, %r837, %r464, %r839;
	fma.rn.f32 	%r841, %r836, %r465, %r838;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r536, %r841, %r840;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r842, %r935, %r452;
	mul.f32 	%r843, %r934, %r452;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs135, %rs136}, %r654;
	cvt.f32.bf16 	%r844, %rs136;
	cvt.f32.bf16 	%r845, %rs135;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r846, %r843, %r464, %r845;
	fma.rn.f32 	%r847, %r842, %r465, %r844;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r532, %r847, %r846;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r848, %r937, %r453;
	mul.f32 	%r849, %r936, %r453;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs137, %rs138}, %r655;
	cvt.f32.bf16 	%r850, %rs138;
	cvt.f32.bf16 	%r851, %rs137;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r852, %r849, %r464, %r851;
	fma.rn.f32 	%r853, %r848, %r465, %r850;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r537, %r853, %r852;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r854, %r951, %r454;
	mul.f32 	%r855, %r950, %r454;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs139, %rs140}, %r662;
	cvt.f32.bf16 	%r856, %rs140;
	cvt.f32.bf16 	%r857, %rs139;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r858, %r855, %r464, %r857;
	fma.rn.f32 	%r859, %r854, %r465, %r856;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r533, %r859, %r858;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r860, %r953, %r455;
	mul.f32 	%r861, %r952, %r455;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs141, %rs142}, %r663;
	cvt.f32.bf16 	%r862, %rs142;
	cvt.f32.bf16 	%r863, %rs141;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r864, %r861, %r464, %r863;
	fma.rn.f32 	%r865, %r860, %r465, %r862;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r538, %r865, %r864;
	cvt.rn.bf16x2.f32 	%r534, %r829, %r828;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r866, %r969, %r457;
	mul.f32 	%r867, %r968, %r457;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs143, %rs144}, %r671;
	cvt.f32.bf16 	%r868, %rs144;
	cvt.f32.bf16 	%r869, %rs143;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r870, %r867, %r464, %r869;
	fma.rn.f32 	%r871, %r866, %r465, %r868;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r539, %r871, %r870;
	bar.sync 	0;
	shl.b32 	%r872, %r4, 13;
	shl.b32 	%r873, %r4, 5;
	and.b32 	%r874, %r13, 384;
	shr.u32 	%r875, %r5, 1;
	bfe.s32 	%r876, %r2, 2, 1;
	and.b32 	%r877, %r876, 4112;
	shl.b32 	%r878, %r574, 3;
	or.b32 	%r879, %r872, %r878;
	or.b32 	%r880, %r873, %r874;
	xor.b32 	%r881, %r877, %r875;
	xor.b32 	%r882, %r881, %r880;
	or.b32 	%r883, %r882, %r879;
	add.s32 	%r500, %r177, %r883;
	// begin inline asm
	st.shared.v4.b32 [ %r500 + 0 ], { %r501, %r502, %r503, %r504 };
	// end inline asm
	add.s32 	%r505, %r500, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r505 + 0 ], { %r506, %r507, %r508, %r509 };
	// end inline asm
	add.s32 	%r510, %r500, 2048;
	// begin inline asm
	st.shared.v4.b32 [ %r510 + 0 ], { %r511, %r512, %r513, %r514 };
	// end inline asm
	add.s32 	%r515, %r500, 2560;
	// begin inline asm
	st.shared.v4.b32 [ %r515 + 0 ], { %r516, %r517, %r518, %r519 };
	// end inline asm
	xor.b32 	%r884, %r883, 64;
	add.s32 	%r520, %r177, %r884;
	// begin inline asm
	st.shared.v4.b32 [ %r520 + 0 ], { %r521, %r522, %r523, %r524 };
	// end inline asm
	add.s32 	%r525, %r520, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r525 + 0 ], { %r526, %r527, %r528, %r529 };
	// end inline asm
	add.s32 	%r530, %r520, 2048;
	// begin inline asm
	st.shared.v4.b32 [ %r530 + 0 ], { %r531, %r532, %r533, %r534 };
	// end inline asm
	add.s32 	%r535, %r520, 2560;
	// begin inline asm
	st.shared.v4.b32 [ %r535 + 0 ], { %r536, %r537, %r538, %r539 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r885, %r2, 2;
	and.b32 	%r886, %r885, 896;
	shl.b32 	%r887, %r2, 8;
	and.b32 	%r888, %r887, 2048;
	setp.eq.b32 	%p24, %r904, 0;
	selp.b32 	%r889, 0, 4112, %p24;
	or.b32 	%r890, %r900, %r886;
	xor.b32 	%r891, %r890, %r889;
	or.b32 	%r892, %r891, %r888;
	add.s32 	%r893, %r177, %r892;
	ld.shared.v4.b32 	{%r540, %r548, %r556, %r564}, [%r893];
	ld.shared.v4.b32 	{%r544, %r552, %r560, %r568}, [%r893+1024];
	xor.b32 	%r894, %r892, 32;
	add.s32 	%r895, %r177, %r894;
	ld.shared.v4.b32 	{%r541, %r549, %r557, %r565}, [%r895+8192];
	ld.shared.v4.b32 	{%r545, %r553, %r561, %r569}, [%r895+9216];
	xor.b32 	%r896, %r892, 64;
	add.s32 	%r897, %r177, %r896;
	ld.shared.v4.b32 	{%r542, %r550, %r558, %r566}, [%r897+16384];
	ld.shared.v4.b32 	{%r546, %r554, %r562, %r570}, [%r897+17408];
	xor.b32 	%r898, %r892, 96;
	add.s32 	%r899, %r177, %r898;
	ld.shared.v4.b32 	{%r543, %r551, %r559, %r567}, [%r899+24576];
	ld.shared.v4.b32 	{%r547, %r555, %r563, %r571}, [%r899+25600];
	.loc	1 172 8                         // sk07_lm_head.py:172:8
	// begin inline asm
	@%p7 st.global.v4.b32 [ %rd190 + 0 ], { %r540, %r541, %r542, %r543 };
	// end inline asm
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd191 + 0 ], { %r544, %r545, %r546, %r547 };
	// end inline asm
	// begin inline asm
	@%p9 st.global.v4.b32 [ %rd192 + 0 ], { %r548, %r549, %r550, %r551 };
	// end inline asm
	// begin inline asm
	@%p10 st.global.v4.b32 [ %rd193 + 0 ], { %r552, %r553, %r554, %r555 };
	// end inline asm
	// begin inline asm
	@%p11 st.global.v4.b32 [ %rd194 + 0 ], { %r556, %r557, %r558, %r559 };
	// end inline asm
	// begin inline asm
	@%p12 st.global.v4.b32 [ %rd195 + 0 ], { %r560, %r561, %r562, %r563 };
	// end inline asm
	// begin inline asm
	@%p13 st.global.v4.b32 [ %rd196 + 0 ], { %r564, %r565, %r566, %r567 };
	// end inline asm
	// begin inline asm
	@%p14 st.global.v4.b32 [ %rd197 + 0 ], { %r568, %r569, %r570, %r571 };
	// end inline asm
	.loc	1 170 4                         // sk07_lm_head.py:170:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk07_lm_head.py"
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
.b32 159                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x98 DW_TAG_compile_unit
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
.b8 55
.b8 95
.b8 108
.b8 109
.b8 95
.b8 104
.b8 101
.b8 97
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
.b8 2                                   // Abbrev [2] 0x45:0x17 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 55
.b8 95
.b8 108
.b8 109
.b8 95
.b8 104
.b8 101
.b8 97
.b8 100
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x5c:0x46 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 69                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x71:0x18 DW_TAG_inlined_subroutine
.b32 69                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 133                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x89:0x18 DW_TAG_inlined_subroutine
.b32 69                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 134                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_3 = _Nativo(
    "sk07_lm_head/tile128x128x128_shift0_abi17",
    _PTX_3, "_sk07_lm_head_kernel",
    warps=8, shared=65536,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 17, 18, 19],
    horneado={12: 1, 13: 1, 16: 1, 20: 1, 21: 128, 22: 128, 23: 128, 24: 8, 25: 128},
    div16=[9, 10, 11, 14, 15, 17, 18],
)

_PTX_4 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk07_lm_head_kernel    // -- Begin function _sk07_lm_head_kernel
.extern .shared .align 16 .b8 global_smem[];
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
                                        // @_sk07_lm_head_kernel
.visible .entry _sk07_lm_head_kernel(
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_6,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_7,
	.param .u32 _sk07_lm_head_kernel_param_8,
	.param .u32 _sk07_lm_head_kernel_param_9,
	.param .u32 _sk07_lm_head_kernel_param_10,
	.param .u32 _sk07_lm_head_kernel_param_11,
	.param .u32 _sk07_lm_head_kernel_param_12,
	.param .u32 _sk07_lm_head_kernel_param_13,
	.param .u32 _sk07_lm_head_kernel_param_14,
	.param .u32 _sk07_lm_head_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_16,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_17
)
.reqntid 256
{
	.reg .pred 	%p<42>;
	.reg .b16 	%rs<289>;
	.reg .b32 	%r<1535>;
	.reg .b64 	%rd<320>;
	.loc	1 123 0                         // sk07_lm_head.py:123:0
$L__func_begin0:
	.loc	1 123 0                         // sk07_lm_head.py:123:0

// %bb.0:
	ld.param.b32 	%r18, [_sk07_lm_head_kernel_param_14];
	ld.param.b32 	%r17, [_sk07_lm_head_kernel_param_13];
	ld.param.b32 	%r16, [_sk07_lm_head_kernel_param_9];
	ld.param.b32 	%r15, [_sk07_lm_head_kernel_param_8];
	ld.param.b64 	%rd29, [_sk07_lm_head_kernel_param_6];
	ld.param.b64 	%rd28, [_sk07_lm_head_kernel_param_5];
	ld.param.b64 	%rd27, [_sk07_lm_head_kernel_param_4];
	ld.param.b64 	%rd26, [_sk07_lm_head_kernel_param_2];
	ld.param.b64 	%rd25, [_sk07_lm_head_kernel_param_1];
	ld.param.b64 	%rd24, [_sk07_lm_head_kernel_param_0];
$L__tmp0:
	.loc	1 132 24                        // sk07_lm_head.py:132:24
	mov.u32 	%r67, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk07_lm_head.py:133:27 ]
	add.s32 	%r68, %r15, 255;
	.loc	2 43 30                         // standard.py:43:30 @[ sk07_lm_head.py:133:27 ]
	shr.s32 	%r69, %r68, 31;
	shr.u32 	%r70, %r69, 24;
	add.s32 	%r71, %r68, %r70;
	shr.s32 	%r72, %r71, 8;
	ld.param.b64 	%rd60, [_sk07_lm_head_kernel_param_3];
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk07_lm_head.py:134:27 ]
	add.s32 	%r73, %r16, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk07_lm_head.py:134:27 ]
	shr.s32 	%r74, %r73, 31;
	shr.u32 	%r75, %r74, 25;
	add.s32 	%r76, %r73, %r75;
	shr.s32 	%r77, %r76, 7;
$L__tmp3:
	.loc	1 135 29                        // sk07_lm_head.py:135:29
	shl.b32 	%r78, %r77, 3;
	.loc	1 136 22                        // sk07_lm_head.py:136:22
	div.s32 	%r79, %r67, %r78;
	.loc	1 136 38                        // sk07_lm_head.py:136:38
	shl.b32 	%r80, %r79, 3;
	.loc	1 137 30                        // sk07_lm_head.py:137:30
	sub.s32 	%r81, %r72, %r80;
	ld.param.b32 	%r82, [_sk07_lm_head_kernel_param_10];
	.loc	1 137 39                        // sk07_lm_head.py:137:39
	min.s32 	%r83, %r81, 8;
	ld.param.b32 	%r84, [_sk07_lm_head_kernel_param_11];
	.loc	1 138 30                        // sk07_lm_head.py:138:30
	mul.lo.s32 	%r85, %r79, %r78;
	ld.param.b32 	%r86, [_sk07_lm_head_kernel_param_12];
	sub.s32 	%r87, %r67, %r85;
	.loc	1 139 36                        // sk07_lm_head.py:139:36
	div.s32 	%r88, %r87, %r83;
	.loc	1 138 46                        // sk07_lm_head.py:138:46
	mul.lo.s32 	%r89, %r88, %r83;
	sub.s32 	%r90, %r87, %r89;
	.loc	1 138 23                        // sk07_lm_head.py:138:23
	add.s32 	%r91, %r90, %r80;
	.loc	1 141 22                        // sk07_lm_head.py:141:22
	shl.b32 	%r1, %r91, 8;
	.loc	1 141 45                        // sk07_lm_head.py:141:45
	mov.u32 	%r2, %tid.x;
	shr.u32 	%r92, %r2, 2;
	bfe.u32 	%r93, %r2, 2, 6;
	or.b32 	%r94, %r93, 64;
	and.b32 	%r3, %r2, 255;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r95, %r1, %r93;
	or.b32 	%r96, %r1, %r94;
	or.b32 	%r97, %r95, 128;
	or.b32 	%r98, %r1, %r92;
	or.b32 	%r99, %r98, 192;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r100, %r95, %r15;
	rem.s32 	%r101, %r96, %r15;
	rem.s32 	%r102, %r97, %r15;
	rem.s32 	%r103, %r99, %r15;
	.loc	1 142 22                        // sk07_lm_head.py:142:22
	shl.b32 	%r104, %r88, 7;
	.loc	1 142 45                        // sk07_lm_head.py:142:45
	and.b32 	%r4, %r2, 3;
	shl.b32 	%r105, %r4, 1;
	and.b32 	%r5, %r2, 32;
	shr.u32 	%r106, %r5, 2;
	or.b32 	%r107, %r106, %r105;
	and.b32 	%r6, %r2, 15;
	shl.b32 	%r108, %r6, 3;
	.loc	1 142 32                        // sk07_lm_head.py:142:32
	or.b32 	%r109, %r104, %r93;
	or.b32 	%r110, %r104, %r94;
	or.b32 	%r111, %r104, %r107;
	or.b32 	%r112, %r111, 16;
	or.b32 	%r113, %r111, 32;
	or.b32 	%r114, %r111, 48;
	or.b32 	%r115, %r111, 64;
	or.b32 	%r116, %r111, 80;
	or.b32 	%r117, %r111, 96;
	or.b32 	%r118, %r111, 112;
	or.b32 	%r7, %r104, %r108;
	or.b32 	%r119, %r7, 4;
	.loc	1 142 57                        // sk07_lm_head.py:142:57
	rem.s32 	%r120, %r109, %r16;
	rem.s32 	%r121, %r110, %r16;
	rem.s32 	%r122, %r111, %r16;
	rem.s32 	%r123, %r112, %r16;
	rem.s32 	%r124, %r113, %r16;
	rem.s32 	%r125, %r114, %r16;
	rem.s32 	%r126, %r115, %r16;
	rem.s32 	%r127, %r116, %r16;
	rem.s32 	%r128, %r117, %r16;
	rem.s32 	%r129, %r118, %r16;
	rem.s32 	%r130, %r7, %r16;
	rem.s32 	%r131, %r119, %r16;
	.loc	1 145 29                        // sk07_lm_head.py:145:29
	mad.wide.s32 	%rd30, %r120, 4, %rd60;
	mad.wide.s32 	%rd31, %r121, 4, %rd60;
	mad.wide.s32 	%rd32, %r122, 4, %rd60;
	mad.wide.s32 	%rd33, %r123, 4, %rd60;
	mad.wide.s32 	%rd34, %r124, 4, %rd60;
	mad.wide.s32 	%rd35, %r125, 4, %rd60;
	mad.wide.s32 	%rd36, %r126, 4, %rd60;
	mad.wide.s32 	%rd37, %r127, 4, %rd60;
	mad.wide.s32 	%rd38, %r128, 4, %rd60;
	mad.wide.s32 	%rd39, %r129, 4, %rd60;
	mad.wide.s32 	%rd40, %r130, 4, %rd60;
	mad.wide.s32 	%rd41, %r131, 4, %rd60;
	.loc	1 145 19                        // sk07_lm_head.py:145:19
	// begin inline asm
	mov.u32 %r20, 0x0;
	ld.global.b32 { %r20 }, [ %rd30 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r21, 0x0;
	ld.global.b32 { %r21 }, [ %rd31 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r22, 0x0;
	mov.u32 %r23, 0x0;
	ld.global.v2.b32 { %r22, %r23 }, [ %rd32 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r24, 0x0;
	mov.u32 %r25, 0x0;
	ld.global.v2.b32 { %r24, %r25 }, [ %rd33 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r26, 0x0;
	mov.u32 %r27, 0x0;
	ld.global.v2.b32 { %r26, %r27 }, [ %rd34 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r28, 0x0;
	mov.u32 %r29, 0x0;
	ld.global.v2.b32 { %r28, %r29 }, [ %rd35 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r30, 0x0;
	mov.u32 %r31, 0x0;
	ld.global.v2.b32 { %r30, %r31 }, [ %rd36 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r32, 0x0;
	mov.u32 %r33, 0x0;
	ld.global.v2.b32 { %r32, %r33 }, [ %rd37 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r34, 0x0;
	mov.u32 %r35, 0x0;
	ld.global.v2.b32 { %r34, %r35 }, [ %rd38 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r36, 0x0;
	mov.u32 %r37, 0x0;
	ld.global.v2.b32 { %r36, %r37 }, [ %rd39 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r38, 0x0;
	mov.u32 %r39, 0x0;
	mov.u32 %r40, 0x0;
	mov.u32 %r41, 0x0;
	ld.global.v4.b32 { %r38, %r39, %r40, %r41 }, [ %rd40 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r42, 0x0;
	mov.u32 %r43, 0x0;
	mov.u32 %r44, 0x0;
	mov.u32 %r45, 0x0;
	ld.global.v4.b32 { %r42, %r43, %r44, %r45 }, [ %rd41 + 0 ];
	// end inline asm
	.loc	1 146 39                        // sk07_lm_head.py:146:39
	mul.lo.s32 	%r132, %r100, %r84;
	mul.lo.s32 	%r133, %r101, %r84;
	mul.lo.s32 	%r134, %r102, %r84;
	mul.lo.s32 	%r135, %r103, %r84;
	.loc	1 146 21                        // sk07_lm_head.py:146:21
	cvt.s64.s32 	%rd1, %r132;
	add.s64 	%rd62, %rd24, %rd1;
	cvt.s64.s32 	%rd2, %r133;
	add.s64 	%rd63, %rd24, %rd2;
	cvt.s64.s32 	%rd3, %r134;
	add.s64 	%rd64, %rd24, %rd3;
	cvt.s64.s32 	%rd4, %r135;
	add.s64 	%rd65, %rd24, %rd4;
	.loc	1 146 58                        // sk07_lm_head.py:146:58
	shl.b32 	%r136, %r4, 4;
	.loc	1 146 51                        // sk07_lm_head.py:146:51
	cvt.u64.u32 	%rd5, %r136;
	add.s64 	%rd42, %rd62, %rd5;
	add.s64 	%rd43, %rd63, %rd5;
	add.s64 	%rd44, %rd64, %rd5;
	add.s64 	%rd45, %rd65, %rd5;
	.loc	1 147 21                        // sk07_lm_head.py:147:21
	add.s64 	%rd66, %rd25, %rd5;
	.loc	1 147 67                        // sk07_lm_head.py:147:67
	mul.lo.s32 	%r137, %r20, %r86;
	mul.lo.s32 	%r138, %r21, %r86;
	.loc	1 147 51                        // sk07_lm_head.py:147:51
	cvt.s64.s32 	%rd6, %r137;
	add.s64 	%rd46, %rd66, %rd6;
	cvt.s64.s32 	%rd7, %r138;
	add.s64 	%rd47, %rd66, %rd7;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	setp.gt.s32 	%p1, %r82, 63;
	.loc	1 160 30                        // sk07_lm_head.py:160:30
	shl.b32 	%r206, %r3, 4;
	shl.b32 	%r9, %r2, 1;
	and.b32 	%r10, %r9, 48;
	xor.b32 	%r207, %r206, %r10;
	mov.b32 	%r208, global_smem;
	add.s32 	%r46, %r208, %r207;
	selp.b32 	%r47, 16, 0, %p1;
	// begin inline asm
	cp.async.cg.shared.global [ %r46 + 0 ], [ %rd42 + 0 ], 0x10, %r47;
	// end inline asm
	add.s32 	%r48, %r46, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r48 + 0 ], [ %rd43 + 0 ], 0x10, %r47;
	// end inline asm
	add.s32 	%r49, %r46, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r49 + 0 ], [ %rd44 + 0 ], 0x10, %r47;
	// end inline asm
	add.s32 	%r50, %r46, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r50 + 0 ], [ %rd45 + 0 ], 0x10, %r47;
	// end inline asm
	cp.async.commit_group;
	.loc	1 160 47                        // sk07_lm_head.py:160:47
	add.s32 	%r51, %r46, 49152;
	// begin inline asm
	cp.async.cg.shared.global [ %r51 + 0 ], [ %rd46 + 0 ], 0x10, %r47;
	// end inline asm
	add.s32 	%r52, %r46, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r52 + 0 ], [ %rd47 + 0 ], 0x10, %r47;
	// end inline asm
	cp.async.commit_group;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	setp.gt.s32 	%p2, %r82, 127;
	.loc	1 161 18                        // sk07_lm_head.py:161:18
	add.s64 	%rd48, %rd42, 64;
	add.s64 	%rd49, %rd43, 64;
	add.s64 	%rd50, %rd44, 64;
	add.s64 	%rd51, %rd45, 64;
	.loc	1 162 18                        // sk07_lm_head.py:162:18
	add.s64 	%rd52, %rd46, 64;
	add.s64 	%rd53, %rd47, 64;
	.loc	1 160 30                        // sk07_lm_head.py:160:30
	bar.sync 	0;
	add.s32 	%r53, %r46, 16384;
	selp.b32 	%r54, 16, 0, %p2;
	// begin inline asm
	cp.async.cg.shared.global [ %r53 + 0 ], [ %rd48 + 0 ], 0x10, %r54;
	// end inline asm
	add.s32 	%r55, %r46, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r55 + 0 ], [ %rd49 + 0 ], 0x10, %r54;
	// end inline asm
	add.s32 	%r56, %r46, 24576;
	// begin inline asm
	cp.async.cg.shared.global [ %r56 + 0 ], [ %rd50 + 0 ], 0x10, %r54;
	// end inline asm
	add.s32 	%r57, %r46, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r57 + 0 ], [ %rd51 + 0 ], 0x10, %r54;
	// end inline asm
	cp.async.commit_group;
	.loc	1 160 47                        // sk07_lm_head.py:160:47
	add.s32 	%r58, %r46, 57344;
	// begin inline asm
	cp.async.cg.shared.global [ %r58 + 0 ], [ %rd52 + 0 ], 0x10, %r54;
	// end inline asm
	add.s32 	%r59, %r46, 61440;
	// begin inline asm
	cp.async.cg.shared.global [ %r59 + 0 ], [ %rd53 + 0 ], 0x10, %r54;
	// end inline asm
	cp.async.commit_group;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	setp.gt.s32 	%p3, %r82, 191;
	.loc	1 161 18                        // sk07_lm_head.py:161:18
	add.s64 	%rd54, %rd42, 128;
	add.s64 	%rd55, %rd43, 128;
	add.s64 	%rd56, %rd44, 128;
	add.s64 	%rd57, %rd45, 128;
	.loc	1 162 18                        // sk07_lm_head.py:162:18
	add.s64 	%rd58, %rd46, 128;
	add.s64 	%rd59, %rd47, 128;
	.loc	1 160 30                        // sk07_lm_head.py:160:30
	bar.sync 	0;
	add.s32 	%r60, %r46, 32768;
	selp.b32 	%r61, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r60 + 0 ], [ %rd54 + 0 ], 0x10, %r61;
	// end inline asm
	add.s32 	%r62, %r46, 36864;
	// begin inline asm
	cp.async.cg.shared.global [ %r62 + 0 ], [ %rd55 + 0 ], 0x10, %r61;
	// end inline asm
	add.s32 	%r63, %r46, 40960;
	// begin inline asm
	cp.async.cg.shared.global [ %r63 + 0 ], [ %rd56 + 0 ], 0x10, %r61;
	// end inline asm
	add.s32 	%r64, %r46, 45056;
	// begin inline asm
	cp.async.cg.shared.global [ %r64 + 0 ], [ %rd57 + 0 ], 0x10, %r61;
	// end inline asm
	cp.async.commit_group;
	.loc	1 160 47                        // sk07_lm_head.py:160:47
	add.s32 	%r65, %r46, 65536;
	// begin inline asm
	cp.async.cg.shared.global [ %r65 + 0 ], [ %rd58 + 0 ], 0x10, %r61;
	// end inline asm
	add.s32 	%r66, %r46, 69632;
	// begin inline asm
	cp.async.cg.shared.global [ %r66 + 0 ], [ %rd59 + 0 ], 0x10, %r61;
	// end inline asm
	cp.async.commit_group;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	@%p1 bra 	$L__BB0_2;
	bra.uni 	$L__BB0_1;
$L__BB0_2:                              // %.lr.ph
	.loc	1 0 23                          // sk07_lm_head.py:0:23
	ld.param.b32 	%r19, [_sk07_lm_head_kernel_param_15];
	ld.param.b64 	%rd61, [_sk07_lm_head_kernel_param_7];
	shr.s32 	%r139, %r22, 31;
	shr.u32 	%r140, %r139, 25;
	add.s32 	%r141, %r22, %r140;
	shr.s32 	%r142, %r141, 7;
	shr.s32 	%r143, %r23, 31;
	shr.u32 	%r144, %r143, 25;
	add.s32 	%r145, %r23, %r144;
	shr.s32 	%r146, %r145, 7;
	shr.s32 	%r147, %r24, 31;
	shr.u32 	%r148, %r147, 25;
	add.s32 	%r149, %r24, %r148;
	shr.s32 	%r150, %r149, 7;
	shr.s32 	%r151, %r25, 31;
	shr.u32 	%r152, %r151, 25;
	add.s32 	%r153, %r25, %r152;
	shr.s32 	%r154, %r153, 7;
	shr.s32 	%r155, %r26, 31;
	shr.u32 	%r156, %r155, 25;
	add.s32 	%r157, %r26, %r156;
	shr.s32 	%r158, %r157, 7;
	shr.s32 	%r159, %r27, 31;
	shr.u32 	%r160, %r159, 25;
	add.s32 	%r161, %r27, %r160;
	shr.s32 	%r162, %r161, 7;
	shr.s32 	%r163, %r28, 31;
	shr.u32 	%r164, %r163, 25;
	add.s32 	%r165, %r28, %r164;
	shr.s32 	%r166, %r165, 7;
	shr.s32 	%r167, %r29, 31;
	shr.u32 	%r168, %r167, 25;
	add.s32 	%r169, %r29, %r168;
	shr.s32 	%r170, %r169, 7;
	shr.s32 	%r171, %r30, 31;
	shr.u32 	%r172, %r171, 25;
	add.s32 	%r173, %r30, %r172;
	shr.s32 	%r174, %r173, 7;
	shr.s32 	%r175, %r31, 31;
	shr.u32 	%r176, %r175, 25;
	add.s32 	%r177, %r31, %r176;
	shr.s32 	%r178, %r177, 7;
	shr.s32 	%r179, %r32, 31;
	shr.u32 	%r180, %r179, 25;
	add.s32 	%r181, %r32, %r180;
	shr.s32 	%r182, %r181, 7;
	shr.s32 	%r183, %r33, 31;
	shr.u32 	%r184, %r183, 25;
	add.s32 	%r185, %r33, %r184;
	shr.s32 	%r186, %r185, 7;
	shr.s32 	%r187, %r34, 31;
	shr.u32 	%r188, %r187, 25;
	add.s32 	%r189, %r34, %r188;
	shr.s32 	%r190, %r189, 7;
	shr.s32 	%r191, %r35, 31;
	shr.u32 	%r192, %r191, 25;
	add.s32 	%r193, %r35, %r192;
	shr.s32 	%r194, %r193, 7;
	shr.s32 	%r195, %r36, 31;
	shr.u32 	%r196, %r195, 25;
	add.s32 	%r197, %r36, %r196;
	shr.s32 	%r198, %r197, 7;
	shr.s32 	%r199, %r37, 31;
	shr.u32 	%r200, %r199, 25;
	add.s32 	%r201, %r37, %r200;
	shr.s32 	%r202, %r201, 7;
	cvt.s64.s32 	%rd67, %r142;
	add.s64 	%rd8, %rd61, %rd67;
	cvt.s64.s32 	%rd68, %r146;
	add.s64 	%rd9, %rd61, %rd68;
	cvt.s64.s32 	%rd69, %r150;
	add.s64 	%rd10, %rd61, %rd69;
	cvt.s64.s32 	%rd70, %r154;
	add.s64 	%rd11, %rd61, %rd70;
	cvt.s64.s32 	%rd71, %r158;
	add.s64 	%rd12, %rd61, %rd71;
	cvt.s64.s32 	%rd72, %r162;
	add.s64 	%rd13, %rd61, %rd72;
	cvt.s64.s32 	%rd73, %r166;
	add.s64 	%rd14, %rd61, %rd73;
	cvt.s64.s32 	%rd74, %r170;
	add.s64 	%rd15, %rd61, %rd74;
	cvt.s64.s32 	%rd75, %r174;
	add.s64 	%rd16, %rd61, %rd75;
	cvt.s64.s32 	%rd76, %r178;
	add.s64 	%rd17, %rd61, %rd76;
	cvt.s64.s32 	%rd77, %r182;
	add.s64 	%rd18, %rd61, %rd77;
	cvt.s64.s32 	%rd78, %r186;
	add.s64 	%rd19, %rd61, %rd78;
	cvt.s64.s32 	%rd79, %r190;
	add.s64 	%rd20, %rd61, %rd79;
	cvt.s64.s32 	%rd80, %r194;
	add.s64 	%rd21, %rd61, %rd80;
	cvt.s64.s32 	%rd81, %r198;
	add.s64 	%rd22, %rd61, %rd81;
	cvt.s64.s32 	%rd82, %r202;
	add.s64 	%rd23, %rd61, %rd82;
	shr.s32 	%r203, %r82, 31;
	shr.u32 	%r204, %r203, 26;
	add.s32 	%r205, %r82, %r204;
	shr.s32 	%r8, %r205, 6;
	add.s32 	%r11, %r8, -3;
	shl.b32 	%r209, %r6, 6;
	shl.b32 	%r1405, %r2, 4;
	and.b32 	%r210, %r1405, 3072;
	shl.b32 	%r211, %r2, 3;
	and.b32 	%r212, %r211, 48;
	and.b32 	%r1406, %r2, 16;
	or.b32 	%r213, %r209, %r210;
	xor.b32 	%r214, %r212, %r1406;
	or.b32 	%r12, %r213, %r214;
	xor.b32 	%r13, %r12, 32;
	shl.b32 	%r215, %r2, 6;
	and.b32 	%r216, %r215, 448;
	shl.b32 	%r217, %r5, 4;
	or.b32 	%r218, %r216, %r212;
	xor.b32 	%r219, %r218, %r10;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	add.s32 	%r220, %r208, %r217;
	add.s32 	%r14, %r220, %r219;
	add.s64 	%rd83, %rd7, %rd25;
	add.s64 	%rd319, %rd83, 192;
	add.s64 	%rd84, %rd6, %rd25;
	add.s64 	%rd318, %rd84, 192;
	add.s64 	%rd85, %rd4, %rd24;
	add.s64 	%rd317, %rd85, 192;
	add.s64 	%rd86, %rd3, %rd24;
	add.s64 	%rd316, %rd86, 192;
	add.s64 	%rd87, %rd2, %rd24;
	add.s64 	%rd315, %rd87, 192;
	add.s64 	%rd88, %rd1, %rd24;
	add.s64 	%rd314, %rd88, 192;
	mov.b32 	%r1407, 0f00000000;
	mov.b32 	%r221, 0;
	mov.b32 	%r1403, 2;
	mov.b32 	%r1402, -1;
	mov.b32 	%r1404, %r221;
	mov.b32 	%r1408, %r1407;
	mov.b32 	%r1409, %r1407;
	mov.b32 	%r1410, %r1407;
	mov.b32 	%r1411, %r1407;
	mov.b32 	%r1412, %r1407;
	mov.b32 	%r1413, %r1407;
	mov.b32 	%r1414, %r1407;
	mov.b32 	%r1415, %r1407;
	mov.b32 	%r1416, %r1407;
	mov.b32 	%r1417, %r1407;
	mov.b32 	%r1418, %r1407;
	mov.b32 	%r1419, %r1407;
	mov.b32 	%r1420, %r1407;
	mov.b32 	%r1421, %r1407;
	mov.b32 	%r1422, %r1407;
	mov.b32 	%r1423, %r1407;
	mov.b32 	%r1424, %r1407;
	mov.b32 	%r1425, %r1407;
	mov.b32 	%r1426, %r1407;
	mov.b32 	%r1427, %r1407;
	mov.b32 	%r1428, %r1407;
	mov.b32 	%r1429, %r1407;
	mov.b32 	%r1430, %r1407;
	mov.b32 	%r1431, %r1407;
	mov.b32 	%r1432, %r1407;
	mov.b32 	%r1433, %r1407;
	mov.b32 	%r1434, %r1407;
	mov.b32 	%r1435, %r1407;
	mov.b32 	%r1436, %r1407;
	mov.b32 	%r1437, %r1407;
	mov.b32 	%r1438, %r1407;
	mov.b32 	%r1439, %r1407;
	mov.b32 	%r1440, %r1407;
	mov.b32 	%r1441, %r1407;
	mov.b32 	%r1442, %r1407;
	mov.b32 	%r1443, %r1407;
	mov.b32 	%r1444, %r1407;
	mov.b32 	%r1445, %r1407;
	mov.b32 	%r1446, %r1407;
	mov.b32 	%r1447, %r1407;
	mov.b32 	%r1448, %r1407;
	mov.b32 	%r1449, %r1407;
	mov.b32 	%r1450, %r1407;
	mov.b32 	%r1451, %r1407;
	mov.b32 	%r1452, %r1407;
	mov.b32 	%r1453, %r1407;
	mov.b32 	%r1454, %r1407;
	mov.b32 	%r1455, %r1407;
	mov.b32 	%r1456, %r1407;
	mov.b32 	%r1457, %r1407;
	mov.b32 	%r1458, %r1407;
	mov.b32 	%r1459, %r1407;
	mov.b32 	%r1460, %r1407;
	mov.b32 	%r1461, %r1407;
	mov.b32 	%r1462, %r1407;
	mov.b32 	%r1463, %r1407;
	mov.b32 	%r1464, %r1407;
	mov.b32 	%r1465, %r1407;
	mov.b32 	%r1466, %r1407;
	mov.b32 	%r1467, %r1407;
	mov.b32 	%r1468, %r1407;
	mov.b32 	%r1469, %r1407;
	mov.b32 	%r1470, %r1407;
	mov.b32 	%r1471, %r1407;
	mov.b32 	%r1472, %r1407;
	mov.b32 	%r1473, %r1407;
	mov.b32 	%r1474, %r1407;
	mov.b32 	%r1475, %r1407;
	mov.b32 	%r1476, %r1407;
	mov.b32 	%r1477, %r1407;
	mov.b32 	%r1478, %r1407;
	mov.b32 	%r1479, %r1407;
	mov.b32 	%r1480, %r1407;
	mov.b32 	%r1481, %r1407;
	mov.b32 	%r1482, %r1407;
	mov.b32 	%r1483, %r1407;
	mov.b32 	%r1484, %r1407;
	mov.b32 	%r1485, %r1407;
	mov.b32 	%r1486, %r1407;
	mov.b32 	%r1487, %r1407;
	mov.b32 	%r1488, %r1407;
	mov.b32 	%r1489, %r1407;
	mov.b32 	%r1490, %r1407;
	mov.b32 	%r1491, %r1407;
	mov.b32 	%r1492, %r1407;
	mov.b32 	%r1493, %r1407;
	mov.b32 	%r1494, %r1407;
	mov.b32 	%r1495, %r1407;
	mov.b32 	%r1496, %r1407;
	mov.b32 	%r1497, %r1407;
	mov.b32 	%r1498, %r1407;
	mov.b32 	%r1499, %r1407;
	mov.b32 	%r1500, %r1407;
	mov.b32 	%r1501, %r1407;
	mov.b32 	%r1502, %r1407;
	mov.b32 	%r1503, %r1407;
	mov.b32 	%r1504, %r1407;
	mov.b32 	%r1505, %r1407;
	mov.b32 	%r1506, %r1407;
	mov.b32 	%r1507, %r1407;
	mov.b32 	%r1508, %r1407;
	mov.b32 	%r1509, %r1407;
	mov.b32 	%r1510, %r1407;
	mov.b32 	%r1511, %r1407;
	mov.b32 	%r1512, %r1407;
	mov.b32 	%r1513, %r1407;
	mov.b32 	%r1514, %r1407;
	mov.b32 	%r1515, %r1407;
	mov.b32 	%r1516, %r1407;
	mov.b32 	%r1517, %r1407;
	mov.b32 	%r1518, %r1407;
	mov.b32 	%r1519, %r1407;
	mov.b32 	%r1520, %r1407;
	mov.b32 	%r1521, %r1407;
	mov.b32 	%r1522, %r1407;
	mov.b32 	%r1523, %r1407;
	mov.b32 	%r1524, %r1407;
	mov.b32 	%r1525, %r1407;
	mov.b32 	%r1526, %r1407;
	mov.b32 	%r1527, %r1407;
	mov.b32 	%r1528, %r1407;
	mov.b32 	%r1529, %r1407;
	mov.b32 	%r1530, %r1407;
	mov.b32 	%r1531, %r1407;
	mov.b32 	%r1532, %r1407;
	mov.b32 	%r1533, %r1407;
	mov.b32 	%r1534, %r1407;
$L__BB0_3:                              // %__nv_exp2f.exit
                                        // =>This Inner Loop Header: Depth=1
	setp.lt.s32 	%p4, %r1404, %r11;
	add.s32 	%r421, %r1402, 1;
	setp.gt.s32 	%p5, %r421, 2;
	selp.b32 	%r1402, 0, %r421, %p5;
	.loc	1 158 56                        // sk07_lm_head.py:158:56
	bfe.u32 	%r422, %r1404, 1, 25;
	.loc	1 159 31                        // sk07_lm_head.py:159:31
	mul.lo.s32 	%r423, %r422, %r19;
	.loc	1 158 39                        // sk07_lm_head.py:158:39
	cvt.s64.s32 	%rd111, %r423;
	add.s64 	%rd89, %rd8, %rd111;
	add.s64 	%rd90, %rd9, %rd111;
	add.s64 	%rd91, %rd10, %rd111;
	add.s64 	%rd92, %rd11, %rd111;
	add.s64 	%rd93, %rd12, %rd111;
	add.s64 	%rd94, %rd13, %rd111;
	add.s64 	%rd95, %rd14, %rd111;
	add.s64 	%rd96, %rd15, %rd111;
	add.s64 	%rd97, %rd16, %rd111;
	add.s64 	%rd98, %rd17, %rd111;
	add.s64 	%rd99, %rd18, %rd111;
	add.s64 	%rd100, %rd19, %rd111;
	add.s64 	%rd101, %rd20, %rd111;
	add.s64 	%rd102, %rd21, %rd111;
	add.s64 	%rd103, %rd22, %rd111;
	add.s64 	%rd104, %rd23, %rd111;
	.loc	1 158 29                        // sk07_lm_head.py:158:29
	// begin inline asm
	mov.u16 %rs1, 0x0;
	ld.global.b8 { %rs1 }, [ %rd89 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs17, %rs1;
	// begin inline asm
	mov.u16 %rs2, 0x0;
	ld.global.b8 { %rs2 }, [ %rd90 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs18, %rs2;
	// begin inline asm
	mov.u16 %rs3, 0x0;
	ld.global.b8 { %rs3 }, [ %rd91 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs19, %rs3;
	// begin inline asm
	mov.u16 %rs4, 0x0;
	ld.global.b8 { %rs4 }, [ %rd92 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs20, %rs4;
	// begin inline asm
	mov.u16 %rs5, 0x0;
	ld.global.b8 { %rs5 }, [ %rd93 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs21, %rs5;
	// begin inline asm
	mov.u16 %rs6, 0x0;
	ld.global.b8 { %rs6 }, [ %rd94 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs22, %rs6;
	// begin inline asm
	mov.u16 %rs7, 0x0;
	ld.global.b8 { %rs7 }, [ %rd95 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs23, %rs7;
	// begin inline asm
	mov.u16 %rs8, 0x0;
	ld.global.b8 { %rs8 }, [ %rd96 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs24, %rs8;
	// begin inline asm
	mov.u16 %rs9, 0x0;
	ld.global.b8 { %rs9 }, [ %rd97 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs25, %rs9;
	// begin inline asm
	mov.u16 %rs10, 0x0;
	ld.global.b8 { %rs10 }, [ %rd98 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs26, %rs10;
	// begin inline asm
	mov.u16 %rs11, 0x0;
	ld.global.b8 { %rs11 }, [ %rd99 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs27, %rs11;
	// begin inline asm
	mov.u16 %rs12, 0x0;
	ld.global.b8 { %rs12 }, [ %rd100 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs28, %rs12;
	// begin inline asm
	mov.u16 %rs13, 0x0;
	ld.global.b8 { %rs13 }, [ %rd101 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs29, %rs13;
	// begin inline asm
	mov.u16 %rs14, 0x0;
	ld.global.b8 { %rs14 }, [ %rd102 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs30, %rs14;
	// begin inline asm
	mov.u16 %rs15, 0x0;
	ld.global.b8 { %rs15 }, [ %rd103 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs31, %rs15;
	// begin inline asm
	mov.u16 %rs16, 0x0;
	ld.global.b8 { %rs16 }, [ %rd104 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs32, %rs16;
	.loc	1 159 50                        // sk07_lm_head.py:159:50
	cvt.rn.f32.s16 	%r424, %rs17;
	cvt.rn.f32.s16 	%r425, %rs18;
	cvt.rn.f32.s16 	%r426, %rs19;
	cvt.rn.f32.s16 	%r427, %rs20;
	cvt.rn.f32.s16 	%r428, %rs21;
	cvt.rn.f32.s16 	%r429, %rs22;
	cvt.rn.f32.s16 	%r430, %rs23;
	cvt.rn.f32.s16 	%r431, %rs24;
	cvt.rn.f32.s16 	%r432, %rs25;
	cvt.rn.f32.s16 	%r433, %rs26;
	cvt.rn.f32.s16 	%r434, %rs27;
	cvt.rn.f32.s16 	%r435, %rs28;
	cvt.rn.f32.s16 	%r436, %rs29;
	cvt.rn.f32.s16 	%r437, %rs30;
	cvt.rn.f32.s16 	%r438, %rs31;
	cvt.rn.f32.s16 	%r439, %rs32;
	.loc	1 158 21                        // sk07_lm_head.py:158:21
	ex2.approx.ftz.f32 	%r440, %r424;
	ex2.approx.ftz.f32 	%r441, %r425;
	ex2.approx.ftz.f32 	%r442, %r426;
	ex2.approx.ftz.f32 	%r443, %r427;
	ex2.approx.ftz.f32 	%r444, %r428;
	ex2.approx.ftz.f32 	%r445, %r429;
	ex2.approx.ftz.f32 	%r446, %r430;
	ex2.approx.ftz.f32 	%r447, %r431;
	ex2.approx.ftz.f32 	%r448, %r432;
	ex2.approx.ftz.f32 	%r449, %r433;
	ex2.approx.ftz.f32 	%r450, %r434;
	ex2.approx.ftz.f32 	%r451, %r435;
	ex2.approx.ftz.f32 	%r452, %r436;
	ex2.approx.ftz.f32 	%r453, %r437;
	ex2.approx.ftz.f32 	%r454, %r438;
	ex2.approx.ftz.f32 	%r455, %r439;
	.loc	1 160 30                        // sk07_lm_head.py:160:30
	cp.async.wait_group 	4;
	bar.sync 	0;
	shl.b32 	%r456, %r1402, 14;
	add.s32 	%r457, %r208, %r456;
	add.s32 	%r458, %r457, %r12;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r222, %r223, %r224, %r225}, [%r458];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r242, %r243, %r244, %r245}, [%r458+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r246, %r247, %r248, %r249}, [%r458+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r250, %r251, %r252, %r253}, [%r458+12288];
	add.s32 	%r459, %r457, %r13;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r258, %r259, %r260, %r261}, [%r459];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r310, %r311, %r312, %r313}, [%r459+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r346, %r347, %r348, %r349}, [%r459+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r382, %r383, %r384, %r385}, [%r459+12288];
	.loc	1 160 47                        // sk07_lm_head.py:160:47
	shl.b32 	%r460, %r1402, 13;
	add.s32 	%r461, %r14, %r460;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r226, %r227, %r262, %r263}, [%r461+49152];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r228, %r229, %r268, %r269}, [%r461+50176];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r230, %r231, %r274, %r275}, [%r461+51200];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r232, %r233, %r280, %r281}, [%r461+52224];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r234, %r235, %r286, %r287}, [%r461+53248];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r236, %r237, %r292, %r293}, [%r461+54272];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r238, %r239, %r298, %r299}, [%r461+55296];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r240, %r241, %r304, %r305}, [%r461+56320];
	.loc	1 160 39                        // sk07_lm_head.py:160:39
	mov.b32 	%r254, %r221;
	mov.b32 	%r255, %r221;
	mov.b32 	%r256, %r221;
	mov.b32 	%r257, %r221;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r254, %r255, %r256, %r257 }, { %r222, %r223, %r224, %r225 }, { %r226, %r227 }, { %r254, %r255, %r256, %r257 };
	// end inline asm
	mov.b32 	%r264, %r221;
	mov.b32 	%r265, %r221;
	mov.b32 	%r266, %r221;
	mov.b32 	%r267, %r221;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r264, %r265, %r266, %r267 }, { %r222, %r223, %r224, %r225 }, { %r228, %r229 }, { %r264, %r265, %r266, %r267 };
	// end inline asm
	mov.b32 	%r270, %r221;
	mov.b32 	%r271, %r221;
	mov.b32 	%r272, %r221;
	mov.b32 	%r273, %r221;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r270, %r271, %r272, %r273 }, { %r222, %r223, %r224, %r225 }, { %r230, %r231 }, { %r270, %r271, %r272, %r273 };
	// end inline asm
	mov.b32 	%r276, %r221;
	mov.b32 	%r277, %r221;
	mov.b32 	%r278, %r221;
	mov.b32 	%r279, %r221;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r276, %r277, %r278, %r279 }, { %r222, %r223, %r224, %r225 }, { %r232, %r233 }, { %r276, %r277, %r278, %r279 };
	// end inline asm
	mov.b32 	%r282, %r221;
	mov.b32 	%r283, %r221;
	mov.b32 	%r284, %r221;
	mov.b32 	%r285, %r221;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r282, %r283, %r284, %r285 }, { %r222, %r223, %r224, %r225 }, { %r234, %r235 }, { %r282, %r283, %r284, %r285 };
	// end inline asm
	mov.b32 	%r288, %r221;
	mov.b32 	%r289, %r221;
	mov.b32 	%r290, %r221;
	mov.b32 	%r291, %r221;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r288, %r289, %r290, %r291 }, { %r222, %r223, %r224, %r225 }, { %r236, %r237 }, { %r288, %r289, %r290, %r291 };
	// end inline asm
	mov.b32 	%r294, %r221;
	mov.b32 	%r295, %r221;
	mov.b32 	%r296, %r221;
	mov.b32 	%r297, %r221;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r294, %r295, %r296, %r297 }, { %r222, %r223, %r224, %r225 }, { %r238, %r239 }, { %r294, %r295, %r296, %r297 };
	// end inline asm
	mov.b32 	%r300, %r221;
	mov.b32 	%r301, %r221;
	mov.b32 	%r302, %r221;
	mov.b32 	%r303, %r221;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r300, %r301, %r302, %r303 }, { %r222, %r223, %r224, %r225 }, { %r240, %r241 }, { %r300, %r301, %r302, %r303 };
	// end inline asm
	mov.b32 	%r306, %r221;
	mov.b32 	%r307, %r221;
	mov.b32 	%r308, %r221;
	mov.b32 	%r309, %r221;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r306, %r307, %r308, %r309 }, { %r242, %r243, %r244, %r245 }, { %r226, %r227 }, { %r306, %r307, %r308, %r309 };
	// end inline asm
	mov.b32 	%r314, %r221;
	mov.b32 	%r315, %r221;
	mov.b32 	%r316, %r221;
	mov.b32 	%r317, %r221;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r314, %r315, %r316, %r317 }, { %r242, %r243, %r244, %r245 }, { %r228, %r229 }, { %r314, %r315, %r316, %r317 };
	// end inline asm
	mov.b32 	%r318, %r221;
	mov.b32 	%r319, %r221;
	mov.b32 	%r320, %r221;
	mov.b32 	%r321, %r221;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r318, %r319, %r320, %r321 }, { %r242, %r243, %r244, %r245 }, { %r230, %r231 }, { %r318, %r319, %r320, %r321 };
	// end inline asm
	mov.b32 	%r322, %r221;
	mov.b32 	%r323, %r221;
	mov.b32 	%r324, %r221;
	mov.b32 	%r325, %r221;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r322, %r323, %r324, %r325 }, { %r242, %r243, %r244, %r245 }, { %r232, %r233 }, { %r322, %r323, %r324, %r325 };
	// end inline asm
	mov.b32 	%r326, %r221;
	mov.b32 	%r327, %r221;
	mov.b32 	%r328, %r221;
	mov.b32 	%r329, %r221;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r326, %r327, %r328, %r329 }, { %r242, %r243, %r244, %r245 }, { %r234, %r235 }, { %r326, %r327, %r328, %r329 };
	// end inline asm
	mov.b32 	%r330, %r221;
	mov.b32 	%r331, %r221;
	mov.b32 	%r332, %r221;
	mov.b32 	%r333, %r221;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r330, %r331, %r332, %r333 }, { %r242, %r243, %r244, %r245 }, { %r236, %r237 }, { %r330, %r331, %r332, %r333 };
	// end inline asm
	mov.b32 	%r334, %r221;
	mov.b32 	%r335, %r221;
	mov.b32 	%r336, %r221;
	mov.b32 	%r337, %r221;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r334, %r335, %r336, %r337 }, { %r242, %r243, %r244, %r245 }, { %r238, %r239 }, { %r334, %r335, %r336, %r337 };
	// end inline asm
	mov.b32 	%r338, %r221;
	mov.b32 	%r339, %r221;
	mov.b32 	%r340, %r221;
	mov.b32 	%r341, %r221;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r338, %r339, %r340, %r341 }, { %r242, %r243, %r244, %r245 }, { %r240, %r241 }, { %r338, %r339, %r340, %r341 };
	// end inline asm
	mov.b32 	%r342, %r221;
	mov.b32 	%r343, %r221;
	mov.b32 	%r344, %r221;
	mov.b32 	%r345, %r221;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r342, %r343, %r344, %r345 }, { %r246, %r247, %r248, %r249 }, { %r226, %r227 }, { %r342, %r343, %r344, %r345 };
	// end inline asm
	mov.b32 	%r350, %r221;
	mov.b32 	%r351, %r221;
	mov.b32 	%r352, %r221;
	mov.b32 	%r353, %r221;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r350, %r351, %r352, %r353 }, { %r246, %r247, %r248, %r249 }, { %r228, %r229 }, { %r350, %r351, %r352, %r353 };
	// end inline asm
	mov.b32 	%r354, %r221;
	mov.b32 	%r355, %r221;
	mov.b32 	%r356, %r221;
	mov.b32 	%r357, %r221;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r354, %r355, %r356, %r357 }, { %r246, %r247, %r248, %r249 }, { %r230, %r231 }, { %r354, %r355, %r356, %r357 };
	// end inline asm
	mov.b32 	%r358, %r221;
	mov.b32 	%r359, %r221;
	mov.b32 	%r360, %r221;
	mov.b32 	%r361, %r221;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r358, %r359, %r360, %r361 }, { %r246, %r247, %r248, %r249 }, { %r232, %r233 }, { %r358, %r359, %r360, %r361 };
	// end inline asm
	mov.b32 	%r362, %r221;
	mov.b32 	%r363, %r221;
	mov.b32 	%r364, %r221;
	mov.b32 	%r365, %r221;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r362, %r363, %r364, %r365 }, { %r246, %r247, %r248, %r249 }, { %r234, %r235 }, { %r362, %r363, %r364, %r365 };
	// end inline asm
	mov.b32 	%r366, %r221;
	mov.b32 	%r367, %r221;
	mov.b32 	%r368, %r221;
	mov.b32 	%r369, %r221;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r366, %r367, %r368, %r369 }, { %r246, %r247, %r248, %r249 }, { %r236, %r237 }, { %r366, %r367, %r368, %r369 };
	// end inline asm
	mov.b32 	%r370, %r221;
	mov.b32 	%r371, %r221;
	mov.b32 	%r372, %r221;
	mov.b32 	%r373, %r221;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r370, %r371, %r372, %r373 }, { %r246, %r247, %r248, %r249 }, { %r238, %r239 }, { %r370, %r371, %r372, %r373 };
	// end inline asm
	mov.b32 	%r374, %r221;
	mov.b32 	%r375, %r221;
	mov.b32 	%r376, %r221;
	mov.b32 	%r377, %r221;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r374, %r375, %r376, %r377 }, { %r246, %r247, %r248, %r249 }, { %r240, %r241 }, { %r374, %r375, %r376, %r377 };
	// end inline asm
	mov.b32 	%r378, %r221;
	mov.b32 	%r379, %r221;
	mov.b32 	%r380, %r221;
	mov.b32 	%r381, %r221;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r378, %r379, %r380, %r381 }, { %r250, %r251, %r252, %r253 }, { %r226, %r227 }, { %r378, %r379, %r380, %r381 };
	// end inline asm
	mov.b32 	%r386, %r221;
	mov.b32 	%r387, %r221;
	mov.b32 	%r388, %r221;
	mov.b32 	%r389, %r221;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r386, %r387, %r388, %r389 }, { %r250, %r251, %r252, %r253 }, { %r228, %r229 }, { %r386, %r387, %r388, %r389 };
	// end inline asm
	mov.b32 	%r390, %r221;
	mov.b32 	%r391, %r221;
	mov.b32 	%r392, %r221;
	mov.b32 	%r393, %r221;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r390, %r391, %r392, %r393 }, { %r250, %r251, %r252, %r253 }, { %r230, %r231 }, { %r390, %r391, %r392, %r393 };
	// end inline asm
	mov.b32 	%r394, %r221;
	mov.b32 	%r395, %r221;
	mov.b32 	%r396, %r221;
	mov.b32 	%r397, %r221;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r394, %r395, %r396, %r397 }, { %r250, %r251, %r252, %r253 }, { %r232, %r233 }, { %r394, %r395, %r396, %r397 };
	// end inline asm
	mov.b32 	%r398, %r221;
	mov.b32 	%r399, %r221;
	mov.b32 	%r400, %r221;
	mov.b32 	%r401, %r221;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r398, %r399, %r400, %r401 }, { %r250, %r251, %r252, %r253 }, { %r234, %r235 }, { %r398, %r399, %r400, %r401 };
	// end inline asm
	mov.b32 	%r402, %r221;
	mov.b32 	%r403, %r221;
	mov.b32 	%r404, %r221;
	mov.b32 	%r405, %r221;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r402, %r403, %r404, %r405 }, { %r250, %r251, %r252, %r253 }, { %r236, %r237 }, { %r402, %r403, %r404, %r405 };
	// end inline asm
	mov.b32 	%r406, %r221;
	mov.b32 	%r407, %r221;
	mov.b32 	%r408, %r221;
	mov.b32 	%r409, %r221;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r406, %r407, %r408, %r409 }, { %r250, %r251, %r252, %r253 }, { %r238, %r239 }, { %r406, %r407, %r408, %r409 };
	// end inline asm
	mov.b32 	%r413, %r221;
	mov.b32 	%r410, %r221;
	mov.b32 	%r411, %r221;
	mov.b32 	%r412, %r221;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r410, %r411, %r412, %r413 }, { %r250, %r251, %r252, %r253 }, { %r240, %r241 }, { %r410, %r411, %r412, %r413 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r254, %r255, %r256, %r257 }, { %r258, %r259, %r260, %r261 }, { %r262, %r263 }, { %r254, %r255, %r256, %r257 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r264, %r265, %r266, %r267 }, { %r258, %r259, %r260, %r261 }, { %r268, %r269 }, { %r264, %r265, %r266, %r267 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r270, %r271, %r272, %r273 }, { %r258, %r259, %r260, %r261 }, { %r274, %r275 }, { %r270, %r271, %r272, %r273 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r276, %r277, %r278, %r279 }, { %r258, %r259, %r260, %r261 }, { %r280, %r281 }, { %r276, %r277, %r278, %r279 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r282, %r283, %r284, %r285 }, { %r258, %r259, %r260, %r261 }, { %r286, %r287 }, { %r282, %r283, %r284, %r285 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r288, %r289, %r290, %r291 }, { %r258, %r259, %r260, %r261 }, { %r292, %r293 }, { %r288, %r289, %r290, %r291 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r294, %r295, %r296, %r297 }, { %r258, %r259, %r260, %r261 }, { %r298, %r299 }, { %r294, %r295, %r296, %r297 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r300, %r301, %r302, %r303 }, { %r258, %r259, %r260, %r261 }, { %r304, %r305 }, { %r300, %r301, %r302, %r303 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r306, %r307, %r308, %r309 }, { %r310, %r311, %r312, %r313 }, { %r262, %r263 }, { %r306, %r307, %r308, %r309 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r314, %r315, %r316, %r317 }, { %r310, %r311, %r312, %r313 }, { %r268, %r269 }, { %r314, %r315, %r316, %r317 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r318, %r319, %r320, %r321 }, { %r310, %r311, %r312, %r313 }, { %r274, %r275 }, { %r318, %r319, %r320, %r321 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r322, %r323, %r324, %r325 }, { %r310, %r311, %r312, %r313 }, { %r280, %r281 }, { %r322, %r323, %r324, %r325 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r326, %r327, %r328, %r329 }, { %r310, %r311, %r312, %r313 }, { %r286, %r287 }, { %r326, %r327, %r328, %r329 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r330, %r331, %r332, %r333 }, { %r310, %r311, %r312, %r313 }, { %r292, %r293 }, { %r330, %r331, %r332, %r333 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r334, %r335, %r336, %r337 }, { %r310, %r311, %r312, %r313 }, { %r298, %r299 }, { %r334, %r335, %r336, %r337 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r338, %r339, %r340, %r341 }, { %r310, %r311, %r312, %r313 }, { %r304, %r305 }, { %r338, %r339, %r340, %r341 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r342, %r343, %r344, %r345 }, { %r346, %r347, %r348, %r349 }, { %r262, %r263 }, { %r342, %r343, %r344, %r345 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r350, %r351, %r352, %r353 }, { %r346, %r347, %r348, %r349 }, { %r268, %r269 }, { %r350, %r351, %r352, %r353 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r354, %r355, %r356, %r357 }, { %r346, %r347, %r348, %r349 }, { %r274, %r275 }, { %r354, %r355, %r356, %r357 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r358, %r359, %r360, %r361 }, { %r346, %r347, %r348, %r349 }, { %r280, %r281 }, { %r358, %r359, %r360, %r361 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r362, %r363, %r364, %r365 }, { %r346, %r347, %r348, %r349 }, { %r286, %r287 }, { %r362, %r363, %r364, %r365 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r366, %r367, %r368, %r369 }, { %r346, %r347, %r348, %r349 }, { %r292, %r293 }, { %r366, %r367, %r368, %r369 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r370, %r371, %r372, %r373 }, { %r346, %r347, %r348, %r349 }, { %r298, %r299 }, { %r370, %r371, %r372, %r373 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r374, %r375, %r376, %r377 }, { %r346, %r347, %r348, %r349 }, { %r304, %r305 }, { %r374, %r375, %r376, %r377 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r378, %r379, %r380, %r381 }, { %r382, %r383, %r384, %r385 }, { %r262, %r263 }, { %r378, %r379, %r380, %r381 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r386, %r387, %r388, %r389 }, { %r382, %r383, %r384, %r385 }, { %r268, %r269 }, { %r386, %r387, %r388, %r389 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r390, %r391, %r392, %r393 }, { %r382, %r383, %r384, %r385 }, { %r274, %r275 }, { %r390, %r391, %r392, %r393 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r394, %r395, %r396, %r397 }, { %r382, %r383, %r384, %r385 }, { %r280, %r281 }, { %r394, %r395, %r396, %r397 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r398, %r399, %r400, %r401 }, { %r382, %r383, %r384, %r385 }, { %r286, %r287 }, { %r398, %r399, %r400, %r401 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r402, %r403, %r404, %r405 }, { %r382, %r383, %r384, %r385 }, { %r292, %r293 }, { %r402, %r403, %r404, %r405 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r406, %r407, %r408, %r409 }, { %r382, %r383, %r384, %r385 }, { %r298, %r299 }, { %r406, %r407, %r408, %r409 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r410, %r411, %r412, %r413 }, { %r382, %r383, %r384, %r385 }, { %r304, %r305 }, { %r410, %r411, %r412, %r413 };
	// end inline asm
	.loc	1 160 79                        // sk07_lm_head.py:160:79
	cvt.rn.f32.s32 	%r462, %r413;
	cvt.rn.f32.s32 	%r463, %r412;
	cvt.rn.f32.s32 	%r464, %r411;
	cvt.rn.f32.s32 	%r465, %r410;
	cvt.rn.f32.s32 	%r466, %r409;
	cvt.rn.f32.s32 	%r467, %r408;
	cvt.rn.f32.s32 	%r468, %r407;
	cvt.rn.f32.s32 	%r469, %r406;
	cvt.rn.f32.s32 	%r470, %r405;
	cvt.rn.f32.s32 	%r471, %r404;
	cvt.rn.f32.s32 	%r472, %r403;
	cvt.rn.f32.s32 	%r473, %r402;
	cvt.rn.f32.s32 	%r474, %r401;
	cvt.rn.f32.s32 	%r475, %r400;
	cvt.rn.f32.s32 	%r476, %r399;
	cvt.rn.f32.s32 	%r477, %r398;
	cvt.rn.f32.s32 	%r478, %r397;
	cvt.rn.f32.s32 	%r479, %r396;
	cvt.rn.f32.s32 	%r480, %r395;
	cvt.rn.f32.s32 	%r481, %r394;
	cvt.rn.f32.s32 	%r482, %r393;
	cvt.rn.f32.s32 	%r483, %r392;
	cvt.rn.f32.s32 	%r484, %r391;
	cvt.rn.f32.s32 	%r485, %r390;
	cvt.rn.f32.s32 	%r486, %r389;
	cvt.rn.f32.s32 	%r487, %r388;
	cvt.rn.f32.s32 	%r488, %r387;
	cvt.rn.f32.s32 	%r489, %r386;
	cvt.rn.f32.s32 	%r490, %r381;
	cvt.rn.f32.s32 	%r491, %r380;
	cvt.rn.f32.s32 	%r492, %r379;
	cvt.rn.f32.s32 	%r493, %r378;
	cvt.rn.f32.s32 	%r494, %r374;
	cvt.rn.f32.s32 	%r495, %r375;
	cvt.rn.f32.s32 	%r496, %r376;
	cvt.rn.f32.s32 	%r497, %r377;
	cvt.rn.f32.s32 	%r498, %r370;
	cvt.rn.f32.s32 	%r499, %r371;
	cvt.rn.f32.s32 	%r500, %r372;
	cvt.rn.f32.s32 	%r501, %r373;
	cvt.rn.f32.s32 	%r502, %r366;
	cvt.rn.f32.s32 	%r503, %r367;
	cvt.rn.f32.s32 	%r504, %r368;
	cvt.rn.f32.s32 	%r505, %r369;
	cvt.rn.f32.s32 	%r506, %r362;
	cvt.rn.f32.s32 	%r507, %r363;
	cvt.rn.f32.s32 	%r508, %r364;
	cvt.rn.f32.s32 	%r509, %r365;
	cvt.rn.f32.s32 	%r510, %r358;
	cvt.rn.f32.s32 	%r511, %r359;
	cvt.rn.f32.s32 	%r512, %r360;
	cvt.rn.f32.s32 	%r513, %r361;
	cvt.rn.f32.s32 	%r514, %r354;
	cvt.rn.f32.s32 	%r515, %r355;
	cvt.rn.f32.s32 	%r516, %r356;
	cvt.rn.f32.s32 	%r517, %r357;
	cvt.rn.f32.s32 	%r518, %r350;
	cvt.rn.f32.s32 	%r519, %r351;
	cvt.rn.f32.s32 	%r520, %r352;
	cvt.rn.f32.s32 	%r521, %r353;
	cvt.rn.f32.s32 	%r522, %r342;
	cvt.rn.f32.s32 	%r523, %r343;
	cvt.rn.f32.s32 	%r524, %r344;
	cvt.rn.f32.s32 	%r525, %r345;
	cvt.rn.f32.s32 	%r526, %r338;
	cvt.rn.f32.s32 	%r527, %r339;
	cvt.rn.f32.s32 	%r528, %r340;
	cvt.rn.f32.s32 	%r529, %r341;
	cvt.rn.f32.s32 	%r530, %r334;
	cvt.rn.f32.s32 	%r531, %r335;
	cvt.rn.f32.s32 	%r532, %r336;
	cvt.rn.f32.s32 	%r533, %r337;
	cvt.rn.f32.s32 	%r534, %r330;
	cvt.rn.f32.s32 	%r535, %r331;
	cvt.rn.f32.s32 	%r536, %r332;
	cvt.rn.f32.s32 	%r537, %r333;
	cvt.rn.f32.s32 	%r538, %r326;
	cvt.rn.f32.s32 	%r539, %r327;
	cvt.rn.f32.s32 	%r540, %r328;
	cvt.rn.f32.s32 	%r541, %r329;
	cvt.rn.f32.s32 	%r542, %r322;
	cvt.rn.f32.s32 	%r543, %r323;
	cvt.rn.f32.s32 	%r544, %r324;
	cvt.rn.f32.s32 	%r545, %r325;
	cvt.rn.f32.s32 	%r546, %r318;
	cvt.rn.f32.s32 	%r547, %r319;
	cvt.rn.f32.s32 	%r548, %r320;
	cvt.rn.f32.s32 	%r549, %r321;
	cvt.rn.f32.s32 	%r550, %r314;
	cvt.rn.f32.s32 	%r551, %r315;
	cvt.rn.f32.s32 	%r552, %r316;
	cvt.rn.f32.s32 	%r553, %r317;
	cvt.rn.f32.s32 	%r554, %r306;
	cvt.rn.f32.s32 	%r555, %r307;
	cvt.rn.f32.s32 	%r556, %r308;
	cvt.rn.f32.s32 	%r557, %r309;
	cvt.rn.f32.s32 	%r558, %r300;
	cvt.rn.f32.s32 	%r559, %r301;
	cvt.rn.f32.s32 	%r560, %r302;
	cvt.rn.f32.s32 	%r561, %r303;
	cvt.rn.f32.s32 	%r562, %r294;
	cvt.rn.f32.s32 	%r563, %r295;
	cvt.rn.f32.s32 	%r564, %r296;
	cvt.rn.f32.s32 	%r565, %r297;
	cvt.rn.f32.s32 	%r566, %r288;
	cvt.rn.f32.s32 	%r567, %r289;
	cvt.rn.f32.s32 	%r568, %r290;
	cvt.rn.f32.s32 	%r569, %r291;
	cvt.rn.f32.s32 	%r570, %r282;
	cvt.rn.f32.s32 	%r571, %r283;
	cvt.rn.f32.s32 	%r572, %r284;
	cvt.rn.f32.s32 	%r573, %r285;
	cvt.rn.f32.s32 	%r574, %r276;
	cvt.rn.f32.s32 	%r575, %r277;
	cvt.rn.f32.s32 	%r576, %r278;
	cvt.rn.f32.s32 	%r577, %r279;
	cvt.rn.f32.s32 	%r578, %r270;
	cvt.rn.f32.s32 	%r579, %r271;
	cvt.rn.f32.s32 	%r580, %r272;
	cvt.rn.f32.s32 	%r581, %r273;
	cvt.rn.f32.s32 	%r582, %r264;
	cvt.rn.f32.s32 	%r583, %r265;
	cvt.rn.f32.s32 	%r584, %r266;
	cvt.rn.f32.s32 	%r585, %r267;
	cvt.rn.f32.s32 	%r586, %r254;
	cvt.rn.f32.s32 	%r587, %r255;
	cvt.rn.f32.s32 	%r588, %r256;
	cvt.rn.f32.s32 	%r589, %r257;
	.loc	1 160 15                        // sk07_lm_head.py:160:15
	fma.rn.f32 	%r1410, %r441, %r589, %r1410;
	fma.rn.f32 	%r1409, %r440, %r588, %r1409;
	fma.rn.f32 	%r1408, %r441, %r587, %r1408;
	fma.rn.f32 	%r1407, %r440, %r586, %r1407;
	fma.rn.f32 	%r1414, %r443, %r585, %r1414;
	fma.rn.f32 	%r1413, %r442, %r584, %r1413;
	fma.rn.f32 	%r1412, %r443, %r583, %r1412;
	fma.rn.f32 	%r1411, %r442, %r582, %r1411;
	fma.rn.f32 	%r1418, %r445, %r581, %r1418;
	fma.rn.f32 	%r1417, %r444, %r580, %r1417;
	fma.rn.f32 	%r1416, %r445, %r579, %r1416;
	fma.rn.f32 	%r1415, %r444, %r578, %r1415;
	fma.rn.f32 	%r1422, %r447, %r577, %r1422;
	fma.rn.f32 	%r1421, %r446, %r576, %r1421;
	fma.rn.f32 	%r1420, %r447, %r575, %r1420;
	fma.rn.f32 	%r1419, %r446, %r574, %r1419;
	fma.rn.f32 	%r1426, %r449, %r573, %r1426;
	fma.rn.f32 	%r1425, %r448, %r572, %r1425;
	fma.rn.f32 	%r1424, %r449, %r571, %r1424;
	fma.rn.f32 	%r1423, %r448, %r570, %r1423;
	fma.rn.f32 	%r1430, %r451, %r569, %r1430;
	fma.rn.f32 	%r1429, %r450, %r568, %r1429;
	fma.rn.f32 	%r1428, %r451, %r567, %r1428;
	fma.rn.f32 	%r1427, %r450, %r566, %r1427;
	fma.rn.f32 	%r1434, %r453, %r565, %r1434;
	fma.rn.f32 	%r1433, %r452, %r564, %r1433;
	fma.rn.f32 	%r1432, %r453, %r563, %r1432;
	fma.rn.f32 	%r1431, %r452, %r562, %r1431;
	fma.rn.f32 	%r1438, %r455, %r561, %r1438;
	fma.rn.f32 	%r1437, %r454, %r560, %r1437;
	fma.rn.f32 	%r1436, %r455, %r559, %r1436;
	fma.rn.f32 	%r1435, %r454, %r558, %r1435;
	fma.rn.f32 	%r1442, %r441, %r557, %r1442;
	fma.rn.f32 	%r1441, %r440, %r556, %r1441;
	fma.rn.f32 	%r1440, %r441, %r555, %r1440;
	fma.rn.f32 	%r1439, %r440, %r554, %r1439;
	fma.rn.f32 	%r1446, %r443, %r553, %r1446;
	fma.rn.f32 	%r1445, %r442, %r552, %r1445;
	fma.rn.f32 	%r1444, %r443, %r551, %r1444;
	fma.rn.f32 	%r1443, %r442, %r550, %r1443;
	fma.rn.f32 	%r1450, %r445, %r549, %r1450;
	fma.rn.f32 	%r1449, %r444, %r548, %r1449;
	fma.rn.f32 	%r1448, %r445, %r547, %r1448;
	fma.rn.f32 	%r1447, %r444, %r546, %r1447;
	fma.rn.f32 	%r1454, %r447, %r545, %r1454;
	fma.rn.f32 	%r1453, %r446, %r544, %r1453;
	fma.rn.f32 	%r1452, %r447, %r543, %r1452;
	fma.rn.f32 	%r1451, %r446, %r542, %r1451;
	fma.rn.f32 	%r1458, %r449, %r541, %r1458;
	fma.rn.f32 	%r1457, %r448, %r540, %r1457;
	fma.rn.f32 	%r1456, %r449, %r539, %r1456;
	fma.rn.f32 	%r1455, %r448, %r538, %r1455;
	fma.rn.f32 	%r1462, %r451, %r537, %r1462;
	fma.rn.f32 	%r1461, %r450, %r536, %r1461;
	fma.rn.f32 	%r1460, %r451, %r535, %r1460;
	fma.rn.f32 	%r1459, %r450, %r534, %r1459;
	fma.rn.f32 	%r1466, %r453, %r533, %r1466;
	fma.rn.f32 	%r1465, %r452, %r532, %r1465;
	fma.rn.f32 	%r1464, %r453, %r531, %r1464;
	fma.rn.f32 	%r1463, %r452, %r530, %r1463;
	fma.rn.f32 	%r1470, %r455, %r529, %r1470;
	fma.rn.f32 	%r1469, %r454, %r528, %r1469;
	fma.rn.f32 	%r1468, %r455, %r527, %r1468;
	fma.rn.f32 	%r1467, %r454, %r526, %r1467;
	fma.rn.f32 	%r1474, %r441, %r525, %r1474;
	fma.rn.f32 	%r1473, %r440, %r524, %r1473;
	fma.rn.f32 	%r1472, %r441, %r523, %r1472;
	fma.rn.f32 	%r1471, %r440, %r522, %r1471;
	fma.rn.f32 	%r1478, %r443, %r521, %r1478;
	fma.rn.f32 	%r1477, %r442, %r520, %r1477;
	fma.rn.f32 	%r1476, %r443, %r519, %r1476;
	fma.rn.f32 	%r1475, %r442, %r518, %r1475;
	fma.rn.f32 	%r1482, %r445, %r517, %r1482;
	fma.rn.f32 	%r1481, %r444, %r516, %r1481;
	fma.rn.f32 	%r1480, %r445, %r515, %r1480;
	fma.rn.f32 	%r1479, %r444, %r514, %r1479;
	fma.rn.f32 	%r1486, %r447, %r513, %r1486;
	fma.rn.f32 	%r1485, %r446, %r512, %r1485;
	fma.rn.f32 	%r1484, %r447, %r511, %r1484;
	fma.rn.f32 	%r1483, %r446, %r510, %r1483;
	fma.rn.f32 	%r1490, %r449, %r509, %r1490;
	fma.rn.f32 	%r1489, %r448, %r508, %r1489;
	fma.rn.f32 	%r1488, %r449, %r507, %r1488;
	fma.rn.f32 	%r1487, %r448, %r506, %r1487;
	fma.rn.f32 	%r1494, %r451, %r505, %r1494;
	fma.rn.f32 	%r1493, %r450, %r504, %r1493;
	fma.rn.f32 	%r1492, %r451, %r503, %r1492;
	fma.rn.f32 	%r1491, %r450, %r502, %r1491;
	fma.rn.f32 	%r1498, %r453, %r501, %r1498;
	fma.rn.f32 	%r1497, %r452, %r500, %r1497;
	fma.rn.f32 	%r1496, %r453, %r499, %r1496;
	fma.rn.f32 	%r1495, %r452, %r498, %r1495;
	fma.rn.f32 	%r1502, %r455, %r497, %r1502;
	fma.rn.f32 	%r1501, %r454, %r496, %r1501;
	fma.rn.f32 	%r1500, %r455, %r495, %r1500;
	fma.rn.f32 	%r1499, %r454, %r494, %r1499;
	fma.rn.f32 	%r1503, %r440, %r493, %r1503;
	fma.rn.f32 	%r1504, %r441, %r492, %r1504;
	fma.rn.f32 	%r1505, %r440, %r491, %r1505;
	fma.rn.f32 	%r1506, %r441, %r490, %r1506;
	fma.rn.f32 	%r1507, %r442, %r489, %r1507;
	fma.rn.f32 	%r1508, %r443, %r488, %r1508;
	fma.rn.f32 	%r1509, %r442, %r487, %r1509;
	fma.rn.f32 	%r1510, %r443, %r486, %r1510;
	fma.rn.f32 	%r1511, %r444, %r485, %r1511;
	fma.rn.f32 	%r1512, %r445, %r484, %r1512;
	fma.rn.f32 	%r1513, %r444, %r483, %r1513;
	fma.rn.f32 	%r1514, %r445, %r482, %r1514;
	fma.rn.f32 	%r1515, %r446, %r481, %r1515;
	fma.rn.f32 	%r1516, %r447, %r480, %r1516;
	fma.rn.f32 	%r1517, %r446, %r479, %r1517;
	fma.rn.f32 	%r1518, %r447, %r478, %r1518;
	fma.rn.f32 	%r1519, %r448, %r477, %r1519;
	fma.rn.f32 	%r1520, %r449, %r476, %r1520;
	fma.rn.f32 	%r1521, %r448, %r475, %r1521;
	fma.rn.f32 	%r1522, %r449, %r474, %r1522;
	fma.rn.f32 	%r1523, %r450, %r473, %r1523;
	fma.rn.f32 	%r1524, %r451, %r472, %r1524;
	fma.rn.f32 	%r1525, %r450, %r471, %r1525;
	fma.rn.f32 	%r1526, %r451, %r470, %r1526;
	fma.rn.f32 	%r1527, %r452, %r469, %r1527;
	fma.rn.f32 	%r1528, %r453, %r468, %r1528;
	fma.rn.f32 	%r1529, %r452, %r467, %r1529;
	fma.rn.f32 	%r1530, %r453, %r466, %r1530;
	fma.rn.f32 	%r1531, %r454, %r465, %r1531;
	fma.rn.f32 	%r1532, %r455, %r464, %r1532;
	fma.rn.f32 	%r1533, %r454, %r463, %r1533;
	fma.rn.f32 	%r1534, %r455, %r462, %r1534;
	.loc	1 161 18                        // sk07_lm_head.py:161:18
	add.s64 	%rd105, %rd314, %rd5;
	add.s64 	%rd106, %rd315, %rd5;
	add.s64 	%rd107, %rd316, %rd5;
	.loc	1 162 18                        // sk07_lm_head.py:162:18
	add.s64 	%rd108, %rd317, %rd5;
	add.s64 	%rd109, %rd318, %rd5;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	add.s64 	%rd110, %rd319, %rd5;
	add.s32 	%r590, %r1403, 1;
	setp.gt.s32 	%p6, %r590, 2;
	selp.b32 	%r1403, 0, %r590, %p6;
	.loc	1 160 30                        // sk07_lm_head.py:160:30
	shl.b32 	%r591, %r1403, 14;
	bar.sync 	0;
	add.s32 	%r414, %r46, %r591;
	selp.b32 	%r415, 16, 0, %p4;
	// begin inline asm
	cp.async.cg.shared.global [ %r414 + 0 ], [ %rd105 + 0 ], 0x10, %r415;
	// end inline asm
	add.s32 	%r416, %r414, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r416 + 0 ], [ %rd106 + 0 ], 0x10, %r415;
	// end inline asm
	add.s32 	%r417, %r414, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r417 + 0 ], [ %rd107 + 0 ], 0x10, %r415;
	// end inline asm
	add.s32 	%r418, %r414, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r418 + 0 ], [ %rd108 + 0 ], 0x10, %r415;
	// end inline asm
	cp.async.commit_group;
	.loc	1 160 47                        // sk07_lm_head.py:160:47
	shl.b32 	%r592, %r1403, 13;
	add.s32 	%r593, %r46, %r592;
	add.s32 	%r419, %r593, 49152;
	// begin inline asm
	cp.async.cg.shared.global [ %r419 + 0 ], [ %rd109 + 0 ], 0x10, %r415;
	// end inline asm
	add.s32 	%r420, %r593, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r420 + 0 ], [ %rd110 + 0 ], 0x10, %r415;
	// end inline asm
	cp.async.commit_group;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	add.s32 	%r1404, %r1404, 1;
	add.s64 	%rd319, %rd319, 64;
	add.s64 	%rd318, %rd318, 64;
	add.s64 	%rd317, %rd317, 64;
	add.s64 	%rd316, %rd316, 64;
	add.s64 	%rd315, %rd315, 64;
	add.s64 	%rd314, %rd314, 64;
	setp.ne.b32 	%p7, %r8, %r1404;
	@%p7 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_4;
$L__BB0_1:                              // %.._crit_edge_crit_edge
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	and.b32 	%r1406, %r2, 16;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	shl.b32 	%r1405, %r2, 4;
	mov.b32 	%r1407, 0f00000000;
	mov.b32 	%r1408, %r1407;
	mov.b32 	%r1409, %r1407;
	mov.b32 	%r1410, %r1407;
	mov.b32 	%r1411, %r1407;
	mov.b32 	%r1412, %r1407;
	mov.b32 	%r1413, %r1407;
	mov.b32 	%r1414, %r1407;
	mov.b32 	%r1415, %r1407;
	mov.b32 	%r1416, %r1407;
	mov.b32 	%r1417, %r1407;
	mov.b32 	%r1418, %r1407;
	mov.b32 	%r1419, %r1407;
	mov.b32 	%r1420, %r1407;
	mov.b32 	%r1421, %r1407;
	mov.b32 	%r1422, %r1407;
	mov.b32 	%r1423, %r1407;
	mov.b32 	%r1424, %r1407;
	mov.b32 	%r1425, %r1407;
	mov.b32 	%r1426, %r1407;
	mov.b32 	%r1427, %r1407;
	mov.b32 	%r1428, %r1407;
	mov.b32 	%r1429, %r1407;
	mov.b32 	%r1430, %r1407;
	mov.b32 	%r1431, %r1407;
	mov.b32 	%r1432, %r1407;
	mov.b32 	%r1433, %r1407;
	mov.b32 	%r1434, %r1407;
	mov.b32 	%r1435, %r1407;
	mov.b32 	%r1436, %r1407;
	mov.b32 	%r1437, %r1407;
	mov.b32 	%r1438, %r1407;
	mov.b32 	%r1439, %r1407;
	mov.b32 	%r1440, %r1407;
	mov.b32 	%r1441, %r1407;
	mov.b32 	%r1442, %r1407;
	mov.b32 	%r1443, %r1407;
	mov.b32 	%r1444, %r1407;
	mov.b32 	%r1445, %r1407;
	mov.b32 	%r1446, %r1407;
	mov.b32 	%r1447, %r1407;
	mov.b32 	%r1448, %r1407;
	mov.b32 	%r1449, %r1407;
	mov.b32 	%r1450, %r1407;
	mov.b32 	%r1451, %r1407;
	mov.b32 	%r1452, %r1407;
	mov.b32 	%r1453, %r1407;
	mov.b32 	%r1454, %r1407;
	mov.b32 	%r1455, %r1407;
	mov.b32 	%r1456, %r1407;
	mov.b32 	%r1457, %r1407;
	mov.b32 	%r1458, %r1407;
	mov.b32 	%r1459, %r1407;
	mov.b32 	%r1460, %r1407;
	mov.b32 	%r1461, %r1407;
	mov.b32 	%r1462, %r1407;
	mov.b32 	%r1463, %r1407;
	mov.b32 	%r1464, %r1407;
	mov.b32 	%r1465, %r1407;
	mov.b32 	%r1466, %r1407;
	mov.b32 	%r1467, %r1407;
	mov.b32 	%r1468, %r1407;
	mov.b32 	%r1469, %r1407;
	mov.b32 	%r1470, %r1407;
	mov.b32 	%r1471, %r1407;
	mov.b32 	%r1472, %r1407;
	mov.b32 	%r1473, %r1407;
	mov.b32 	%r1474, %r1407;
	mov.b32 	%r1475, %r1407;
	mov.b32 	%r1476, %r1407;
	mov.b32 	%r1477, %r1407;
	mov.b32 	%r1478, %r1407;
	mov.b32 	%r1479, %r1407;
	mov.b32 	%r1480, %r1407;
	mov.b32 	%r1481, %r1407;
	mov.b32 	%r1482, %r1407;
	mov.b32 	%r1483, %r1407;
	mov.b32 	%r1484, %r1407;
	mov.b32 	%r1485, %r1407;
	mov.b32 	%r1486, %r1407;
	mov.b32 	%r1487, %r1407;
	mov.b32 	%r1488, %r1407;
	mov.b32 	%r1489, %r1407;
	mov.b32 	%r1490, %r1407;
	mov.b32 	%r1491, %r1407;
	mov.b32 	%r1492, %r1407;
	mov.b32 	%r1493, %r1407;
	mov.b32 	%r1494, %r1407;
	mov.b32 	%r1495, %r1407;
	mov.b32 	%r1496, %r1407;
	mov.b32 	%r1497, %r1407;
	mov.b32 	%r1498, %r1407;
	mov.b32 	%r1499, %r1407;
	mov.b32 	%r1500, %r1407;
	mov.b32 	%r1501, %r1407;
	mov.b32 	%r1502, %r1407;
	mov.b32 	%r1503, %r1407;
	mov.b32 	%r1504, %r1407;
	mov.b32 	%r1505, %r1407;
	mov.b32 	%r1506, %r1407;
	mov.b32 	%r1507, %r1407;
	mov.b32 	%r1508, %r1407;
	mov.b32 	%r1509, %r1407;
	mov.b32 	%r1510, %r1407;
	mov.b32 	%r1511, %r1407;
	mov.b32 	%r1512, %r1407;
	mov.b32 	%r1513, %r1407;
	mov.b32 	%r1514, %r1407;
	mov.b32 	%r1515, %r1407;
	mov.b32 	%r1516, %r1407;
	mov.b32 	%r1517, %r1407;
	mov.b32 	%r1518, %r1407;
	mov.b32 	%r1519, %r1407;
	mov.b32 	%r1520, %r1407;
	mov.b32 	%r1521, %r1407;
	mov.b32 	%r1522, %r1407;
	mov.b32 	%r1523, %r1407;
	mov.b32 	%r1524, %r1407;
	mov.b32 	%r1525, %r1407;
	mov.b32 	%r1526, %r1407;
	mov.b32 	%r1527, %r1407;
	mov.b32 	%r1528, %r1407;
	mov.b32 	%r1529, %r1407;
	mov.b32 	%r1530, %r1407;
	mov.b32 	%r1531, %r1407;
	mov.b32 	%r1532, %r1407;
	mov.b32 	%r1533, %r1407;
	mov.b32 	%r1534, %r1407;
$L__BB0_4:                              // %._crit_edge
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r824, %r1, %r3;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r825, %r824, %r15;
	.loc	1 141 45                        // sk07_lm_head.py:141:45
	and.b32 	%r826, %r2, 240;
	bfe.u32 	%r827, %r2, 4, 4;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r828, %r827, %r1;
	or.b32 	%r829, %r828, 240;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r830, %r829, %r15;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r831, %r828, 224;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r832, %r831, %r15;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r833, %r828, 208;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r834, %r833, %r15;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r835, %r828, 192;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r836, %r835, %r15;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r837, %r828, 176;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r838, %r837, %r15;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r839, %r828, 160;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r840, %r839, %r15;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r841, %r828, 144;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r842, %r841, %r15;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r843, %r828, 128;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r844, %r843, %r15;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r845, %r828, 112;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r846, %r845, %r15;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r847, %r828, 96;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r848, %r847, %r15;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r849, %r828, 80;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r850, %r849, %r15;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r851, %r828, 64;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r852, %r851, %r15;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r853, %r828, 48;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r854, %r853, %r15;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r855, %r828, 32;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r856, %r855, %r15;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r857, %r828, 16;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r858, %r857, %r15;
	rem.s32 	%r859, %r828, %r15;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 164 38                        // sk07_lm_head.py:164:38
	mad.wide.s32 	%rd112, %r825, 4, %rd28;
	.loc	1 164 24                        // sk07_lm_head.py:164:24
	// begin inline asm
	mov.u32 %r595, 0x0;
	ld.global.b32 { %r595 }, [ %rd112 + 0 ];
	// end inline asm
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	and.b32 	%r860, %r2, 7;
	shl.b32 	%r861, %r860, 3;
	shl.b32 	%r862, %r826, 2;
	and.b32 	%r863, %r2, 8;
	shr.u32 	%r864, %r863, 1;
	add.s32 	%r865, %r208, %r861;
	add.s32 	%r866, %r865, %r862;
	add.s32 	%r594, %r866, %r864;
	// begin inline asm
	st.shared.b32 [ %r594 + 0 ], %r595;
	// end inline asm
	bar.sync 	0;
	and.b32 	%r867, %r9, 56;
	and.b32 	%r868, %r2, 192;
	add.s32 	%r869, %r208, %r867;
	add.s32 	%r870, %r869, %r868;
	ld.shared.v2.b32 	{%r871, %r872}, [%r870];
	ld.shared.v2.b32 	{%r873, %r874}, [%r870+256];
	ld.shared.v2.b32 	{%r875, %r876}, [%r870+512];
	ld.shared.v2.b32 	{%r877, %r878}, [%r870+768];
	.loc	1 165 38                        // sk07_lm_head.py:165:38
	mad.wide.s32 	%rd113, %r22, 4, %rd29;
	mad.wide.s32 	%rd114, %r23, 4, %rd29;
	mad.wide.s32 	%rd115, %r24, 4, %rd29;
	mad.wide.s32 	%rd116, %r25, 4, %rd29;
	mad.wide.s32 	%rd117, %r26, 4, %rd29;
	mad.wide.s32 	%rd118, %r27, 4, %rd29;
	mad.wide.s32 	%rd119, %r28, 4, %rd29;
	mad.wide.s32 	%rd120, %r29, 4, %rd29;
	mad.wide.s32 	%rd121, %r30, 4, %rd29;
	mad.wide.s32 	%rd122, %r31, 4, %rd29;
	mad.wide.s32 	%rd123, %r32, 4, %rd29;
	mad.wide.s32 	%rd124, %r33, 4, %rd29;
	mad.wide.s32 	%rd125, %r34, 4, %rd29;
	mad.wide.s32 	%rd126, %r35, 4, %rd29;
	mad.wide.s32 	%rd127, %r36, 4, %rd29;
	mad.wide.s32 	%rd128, %r37, 4, %rd29;
	.loc	1 165 24                        // sk07_lm_head.py:165:24
	// begin inline asm
	mov.u32 %r596, 0x0;
	ld.global.b32 { %r596 }, [ %rd113 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r597, 0x0;
	ld.global.b32 { %r597 }, [ %rd114 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r598, 0x0;
	ld.global.b32 { %r598 }, [ %rd115 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r599, 0x0;
	ld.global.b32 { %r599 }, [ %rd116 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r600, 0x0;
	ld.global.b32 { %r600 }, [ %rd117 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r601, 0x0;
	ld.global.b32 { %r601 }, [ %rd118 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r602, 0x0;
	ld.global.b32 { %r602 }, [ %rd119 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r603, 0x0;
	ld.global.b32 { %r603 }, [ %rd120 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r604, 0x0;
	ld.global.b32 { %r604 }, [ %rd121 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r605, 0x0;
	ld.global.b32 { %r605 }, [ %rd122 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r606, 0x0;
	ld.global.b32 { %r606 }, [ %rd123 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r607, 0x0;
	ld.global.b32 { %r607 }, [ %rd124 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r608, 0x0;
	ld.global.b32 { %r608 }, [ %rd125 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r609, 0x0;
	ld.global.b32 { %r609 }, [ %rd126 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r610, 0x0;
	ld.global.b32 { %r610 }, [ %rd127 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r611, 0x0;
	ld.global.b32 { %r611 }, [ %rd128 + 0 ];
	// end inline asm
	.loc	1 166 49                        // sk07_lm_head.py:166:49
	mul.lo.s32 	%r879, %r859, %r18;
	mul.lo.s32 	%r880, %r858, %r18;
	mul.lo.s32 	%r881, %r856, %r18;
	mul.lo.s32 	%r882, %r854, %r18;
	mul.lo.s32 	%r883, %r852, %r18;
	mul.lo.s32 	%r884, %r850, %r18;
	mul.lo.s32 	%r885, %r848, %r18;
	mul.lo.s32 	%r886, %r846, %r18;
	mul.lo.s32 	%r887, %r844, %r18;
	mul.lo.s32 	%r888, %r842, %r18;
	mul.lo.s32 	%r889, %r840, %r18;
	mul.lo.s32 	%r890, %r838, %r18;
	mul.lo.s32 	%r891, %r836, %r18;
	mul.lo.s32 	%r892, %r834, %r18;
	mul.lo.s32 	%r893, %r832, %r18;
	mul.lo.s32 	%r894, %r830, %r18;
	.loc	1 166 31                        // sk07_lm_head.py:166:31
	mad.wide.s32 	%rd273, %r879, 2, %rd27;
	mad.wide.s32 	%rd274, %r880, 2, %rd27;
	mad.wide.s32 	%rd275, %r881, 2, %rd27;
	mad.wide.s32 	%rd276, %r882, 2, %rd27;
	mad.wide.s32 	%rd277, %r883, 2, %rd27;
	mad.wide.s32 	%rd278, %r884, 2, %rd27;
	mad.wide.s32 	%rd279, %r885, 2, %rd27;
	mad.wide.s32 	%rd280, %r886, 2, %rd27;
	mad.wide.s32 	%rd281, %r887, 2, %rd27;
	mad.wide.s32 	%rd282, %r888, 2, %rd27;
	mad.wide.s32 	%rd283, %r889, 2, %rd27;
	mad.wide.s32 	%rd284, %r890, 2, %rd27;
	mad.wide.s32 	%rd285, %r891, 2, %rd27;
	mad.wide.s32 	%rd286, %r892, 2, %rd27;
	mad.wide.s32 	%rd287, %r893, 2, %rd27;
	mad.wide.s32 	%rd288, %r894, 2, %rd27;
	.loc	1 166 64                        // sk07_lm_head.py:166:64
	mul.wide.s32 	%rd289, %r38, 2;
	add.s64 	%rd129, %rd273, %rd289;
	mul.wide.s32 	%rd290, %r39, 2;
	add.s64 	%rd130, %rd273, %rd290;
	mul.wide.s32 	%rd291, %r40, 2;
	add.s64 	%rd131, %rd273, %rd291;
	mul.wide.s32 	%rd292, %r41, 2;
	add.s64 	%rd132, %rd273, %rd292;
	mul.wide.s32 	%rd293, %r42, 2;
	add.s64 	%rd133, %rd273, %rd293;
	mul.wide.s32 	%rd294, %r43, 2;
	add.s64 	%rd134, %rd273, %rd294;
	mul.wide.s32 	%rd295, %r44, 2;
	add.s64 	%rd135, %rd273, %rd295;
	mul.wide.s32 	%rd296, %r45, 2;
	add.s64 	%rd136, %rd273, %rd296;
	add.s64 	%rd137, %rd274, %rd289;
	add.s64 	%rd138, %rd274, %rd290;
	add.s64 	%rd139, %rd274, %rd291;
	add.s64 	%rd140, %rd274, %rd292;
	add.s64 	%rd141, %rd274, %rd293;
	add.s64 	%rd142, %rd274, %rd294;
	add.s64 	%rd143, %rd274, %rd295;
	add.s64 	%rd144, %rd274, %rd296;
	add.s64 	%rd145, %rd275, %rd289;
	add.s64 	%rd146, %rd275, %rd290;
	add.s64 	%rd147, %rd275, %rd291;
	add.s64 	%rd148, %rd275, %rd292;
	add.s64 	%rd149, %rd275, %rd293;
	add.s64 	%rd150, %rd275, %rd294;
	add.s64 	%rd151, %rd275, %rd295;
	add.s64 	%rd152, %rd275, %rd296;
	add.s64 	%rd153, %rd276, %rd289;
	add.s64 	%rd154, %rd276, %rd290;
	add.s64 	%rd155, %rd276, %rd291;
	add.s64 	%rd156, %rd276, %rd292;
	add.s64 	%rd157, %rd276, %rd293;
	add.s64 	%rd158, %rd276, %rd294;
	add.s64 	%rd159, %rd276, %rd295;
	add.s64 	%rd160, %rd276, %rd296;
	add.s64 	%rd161, %rd277, %rd289;
	add.s64 	%rd162, %rd277, %rd290;
	add.s64 	%rd163, %rd277, %rd291;
	add.s64 	%rd164, %rd277, %rd292;
	add.s64 	%rd165, %rd277, %rd293;
	add.s64 	%rd166, %rd277, %rd294;
	add.s64 	%rd167, %rd277, %rd295;
	add.s64 	%rd168, %rd277, %rd296;
	add.s64 	%rd169, %rd278, %rd289;
	add.s64 	%rd170, %rd278, %rd290;
	add.s64 	%rd171, %rd278, %rd291;
	add.s64 	%rd172, %rd278, %rd292;
	add.s64 	%rd173, %rd278, %rd293;
	add.s64 	%rd174, %rd278, %rd294;
	add.s64 	%rd175, %rd278, %rd295;
	add.s64 	%rd176, %rd278, %rd296;
	add.s64 	%rd177, %rd279, %rd289;
	add.s64 	%rd178, %rd279, %rd290;
	add.s64 	%rd179, %rd279, %rd291;
	add.s64 	%rd180, %rd279, %rd292;
	add.s64 	%rd181, %rd279, %rd293;
	add.s64 	%rd182, %rd279, %rd294;
	add.s64 	%rd183, %rd279, %rd295;
	add.s64 	%rd184, %rd279, %rd296;
	add.s64 	%rd185, %rd280, %rd289;
	add.s64 	%rd186, %rd280, %rd290;
	add.s64 	%rd187, %rd280, %rd291;
	add.s64 	%rd188, %rd280, %rd292;
	add.s64 	%rd189, %rd280, %rd293;
	add.s64 	%rd190, %rd280, %rd294;
	add.s64 	%rd191, %rd280, %rd295;
	add.s64 	%rd192, %rd280, %rd296;
	add.s64 	%rd193, %rd281, %rd289;
	add.s64 	%rd194, %rd281, %rd290;
	add.s64 	%rd195, %rd281, %rd291;
	add.s64 	%rd196, %rd281, %rd292;
	add.s64 	%rd197, %rd281, %rd293;
	add.s64 	%rd198, %rd281, %rd294;
	add.s64 	%rd199, %rd281, %rd295;
	add.s64 	%rd200, %rd281, %rd296;
	add.s64 	%rd201, %rd282, %rd289;
	add.s64 	%rd202, %rd282, %rd290;
	add.s64 	%rd203, %rd282, %rd291;
	add.s64 	%rd204, %rd282, %rd292;
	add.s64 	%rd205, %rd282, %rd293;
	add.s64 	%rd206, %rd282, %rd294;
	add.s64 	%rd207, %rd282, %rd295;
	add.s64 	%rd208, %rd282, %rd296;
	add.s64 	%rd209, %rd283, %rd289;
	add.s64 	%rd210, %rd283, %rd290;
	add.s64 	%rd211, %rd283, %rd291;
	add.s64 	%rd212, %rd283, %rd292;
	add.s64 	%rd213, %rd283, %rd293;
	add.s64 	%rd214, %rd283, %rd294;
	add.s64 	%rd215, %rd283, %rd295;
	add.s64 	%rd216, %rd283, %rd296;
	add.s64 	%rd217, %rd284, %rd289;
	add.s64 	%rd218, %rd284, %rd290;
	add.s64 	%rd219, %rd284, %rd291;
	add.s64 	%rd220, %rd284, %rd292;
	add.s64 	%rd221, %rd284, %rd293;
	add.s64 	%rd222, %rd284, %rd294;
	add.s64 	%rd223, %rd284, %rd295;
	add.s64 	%rd224, %rd284, %rd296;
	add.s64 	%rd225, %rd285, %rd289;
	add.s64 	%rd226, %rd285, %rd290;
	add.s64 	%rd227, %rd285, %rd291;
	add.s64 	%rd228, %rd285, %rd292;
	add.s64 	%rd229, %rd285, %rd293;
	add.s64 	%rd230, %rd285, %rd294;
	add.s64 	%rd231, %rd285, %rd295;
	add.s64 	%rd232, %rd285, %rd296;
	add.s64 	%rd233, %rd286, %rd289;
	add.s64 	%rd234, %rd286, %rd290;
	add.s64 	%rd235, %rd286, %rd291;
	add.s64 	%rd236, %rd286, %rd292;
	add.s64 	%rd237, %rd286, %rd293;
	add.s64 	%rd238, %rd286, %rd294;
	add.s64 	%rd239, %rd286, %rd295;
	add.s64 	%rd240, %rd286, %rd296;
	add.s64 	%rd241, %rd287, %rd289;
	add.s64 	%rd242, %rd287, %rd290;
	add.s64 	%rd243, %rd287, %rd291;
	add.s64 	%rd244, %rd287, %rd292;
	add.s64 	%rd245, %rd287, %rd293;
	add.s64 	%rd246, %rd287, %rd294;
	add.s64 	%rd247, %rd287, %rd295;
	add.s64 	%rd248, %rd287, %rd296;
	add.s64 	%rd249, %rd288, %rd289;
	add.s64 	%rd250, %rd288, %rd290;
	add.s64 	%rd251, %rd288, %rd291;
	add.s64 	%rd252, %rd288, %rd292;
	add.s64 	%rd253, %rd288, %rd293;
	add.s64 	%rd254, %rd288, %rd294;
	add.s64 	%rd255, %rd288, %rd295;
	add.s64 	%rd256, %rd288, %rd296;
	.loc	1 166 19                        // sk07_lm_head.py:166:19
	// begin inline asm
	mov.u16 %rs33, 0x0;
	ld.global.b16 { %rs33 }, [ %rd129 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs34, 0x0;
	ld.global.b16 { %rs34 }, [ %rd130 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs35, 0x0;
	ld.global.b16 { %rs35 }, [ %rd131 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs36, 0x0;
	ld.global.b16 { %rs36 }, [ %rd132 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs37, 0x0;
	ld.global.b16 { %rs37 }, [ %rd133 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs38, 0x0;
	ld.global.b16 { %rs38 }, [ %rd134 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs39, 0x0;
	ld.global.b16 { %rs39 }, [ %rd135 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs40, 0x0;
	ld.global.b16 { %rs40 }, [ %rd136 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs41, 0x0;
	ld.global.b16 { %rs41 }, [ %rd137 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs42, 0x0;
	ld.global.b16 { %rs42 }, [ %rd138 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs43, 0x0;
	ld.global.b16 { %rs43 }, [ %rd139 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs44, 0x0;
	ld.global.b16 { %rs44 }, [ %rd140 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs45, 0x0;
	ld.global.b16 { %rs45 }, [ %rd141 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs46, 0x0;
	ld.global.b16 { %rs46 }, [ %rd142 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs47, 0x0;
	ld.global.b16 { %rs47 }, [ %rd143 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs48, 0x0;
	ld.global.b16 { %rs48 }, [ %rd144 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs49, 0x0;
	ld.global.b16 { %rs49 }, [ %rd145 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs50, 0x0;
	ld.global.b16 { %rs50 }, [ %rd146 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs51, 0x0;
	ld.global.b16 { %rs51 }, [ %rd147 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs52, 0x0;
	ld.global.b16 { %rs52 }, [ %rd148 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs53, 0x0;
	ld.global.b16 { %rs53 }, [ %rd149 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs54, 0x0;
	ld.global.b16 { %rs54 }, [ %rd150 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs55, 0x0;
	ld.global.b16 { %rs55 }, [ %rd151 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs56, 0x0;
	ld.global.b16 { %rs56 }, [ %rd152 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs57, 0x0;
	ld.global.b16 { %rs57 }, [ %rd153 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs58, 0x0;
	ld.global.b16 { %rs58 }, [ %rd154 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs59, 0x0;
	ld.global.b16 { %rs59 }, [ %rd155 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs60, 0x0;
	ld.global.b16 { %rs60 }, [ %rd156 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs61, 0x0;
	ld.global.b16 { %rs61 }, [ %rd157 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs62, 0x0;
	ld.global.b16 { %rs62 }, [ %rd158 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs63, 0x0;
	ld.global.b16 { %rs63 }, [ %rd159 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs64, 0x0;
	ld.global.b16 { %rs64 }, [ %rd160 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs65, 0x0;
	ld.global.b16 { %rs65 }, [ %rd161 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs66, 0x0;
	ld.global.b16 { %rs66 }, [ %rd162 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs67, 0x0;
	ld.global.b16 { %rs67 }, [ %rd163 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs68, 0x0;
	ld.global.b16 { %rs68 }, [ %rd164 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs69, 0x0;
	ld.global.b16 { %rs69 }, [ %rd165 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs70, 0x0;
	ld.global.b16 { %rs70 }, [ %rd166 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs71, 0x0;
	ld.global.b16 { %rs71 }, [ %rd167 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs72, 0x0;
	ld.global.b16 { %rs72 }, [ %rd168 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs73, 0x0;
	ld.global.b16 { %rs73 }, [ %rd169 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs74, 0x0;
	ld.global.b16 { %rs74 }, [ %rd170 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs75, 0x0;
	ld.global.b16 { %rs75 }, [ %rd171 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs76, 0x0;
	ld.global.b16 { %rs76 }, [ %rd172 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs77, 0x0;
	ld.global.b16 { %rs77 }, [ %rd173 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs78, 0x0;
	ld.global.b16 { %rs78 }, [ %rd174 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs79, 0x0;
	ld.global.b16 { %rs79 }, [ %rd175 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs80, 0x0;
	ld.global.b16 { %rs80 }, [ %rd176 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs81, 0x0;
	ld.global.b16 { %rs81 }, [ %rd177 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs82, 0x0;
	ld.global.b16 { %rs82 }, [ %rd178 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs83, 0x0;
	ld.global.b16 { %rs83 }, [ %rd179 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs84, 0x0;
	ld.global.b16 { %rs84 }, [ %rd180 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs85, 0x0;
	ld.global.b16 { %rs85 }, [ %rd181 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs86, 0x0;
	ld.global.b16 { %rs86 }, [ %rd182 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs87, 0x0;
	ld.global.b16 { %rs87 }, [ %rd183 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs88, 0x0;
	ld.global.b16 { %rs88 }, [ %rd184 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs89, 0x0;
	ld.global.b16 { %rs89 }, [ %rd185 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs90, 0x0;
	ld.global.b16 { %rs90 }, [ %rd186 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs91, 0x0;
	ld.global.b16 { %rs91 }, [ %rd187 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs92, 0x0;
	ld.global.b16 { %rs92 }, [ %rd188 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs93, 0x0;
	ld.global.b16 { %rs93 }, [ %rd189 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs94, 0x0;
	ld.global.b16 { %rs94 }, [ %rd190 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs95, 0x0;
	ld.global.b16 { %rs95 }, [ %rd191 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs96, 0x0;
	ld.global.b16 { %rs96 }, [ %rd192 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs97, 0x0;
	ld.global.b16 { %rs97 }, [ %rd193 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs98, 0x0;
	ld.global.b16 { %rs98 }, [ %rd194 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs99, 0x0;
	ld.global.b16 { %rs99 }, [ %rd195 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs100, 0x0;
	ld.global.b16 { %rs100 }, [ %rd196 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs101, 0x0;
	ld.global.b16 { %rs101 }, [ %rd197 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs102, 0x0;
	ld.global.b16 { %rs102 }, [ %rd198 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs103, 0x0;
	ld.global.b16 { %rs103 }, [ %rd199 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs104, 0x0;
	ld.global.b16 { %rs104 }, [ %rd200 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs105, 0x0;
	ld.global.b16 { %rs105 }, [ %rd201 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs106, 0x0;
	ld.global.b16 { %rs106 }, [ %rd202 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs107, 0x0;
	ld.global.b16 { %rs107 }, [ %rd203 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs108, 0x0;
	ld.global.b16 { %rs108 }, [ %rd204 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs109, 0x0;
	ld.global.b16 { %rs109 }, [ %rd205 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs110, 0x0;
	ld.global.b16 { %rs110 }, [ %rd206 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs111, 0x0;
	ld.global.b16 { %rs111 }, [ %rd207 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs112, 0x0;
	ld.global.b16 { %rs112 }, [ %rd208 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs113, 0x0;
	ld.global.b16 { %rs113 }, [ %rd209 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs114, 0x0;
	ld.global.b16 { %rs114 }, [ %rd210 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs115, 0x0;
	ld.global.b16 { %rs115 }, [ %rd211 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs116, 0x0;
	ld.global.b16 { %rs116 }, [ %rd212 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs117, 0x0;
	ld.global.b16 { %rs117 }, [ %rd213 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs118, 0x0;
	ld.global.b16 { %rs118 }, [ %rd214 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs119, 0x0;
	ld.global.b16 { %rs119 }, [ %rd215 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs120, 0x0;
	ld.global.b16 { %rs120 }, [ %rd216 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs121, 0x0;
	ld.global.b16 { %rs121 }, [ %rd217 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs122, 0x0;
	ld.global.b16 { %rs122 }, [ %rd218 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs123, 0x0;
	ld.global.b16 { %rs123 }, [ %rd219 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs124, 0x0;
	ld.global.b16 { %rs124 }, [ %rd220 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs125, 0x0;
	ld.global.b16 { %rs125 }, [ %rd221 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs126, 0x0;
	ld.global.b16 { %rs126 }, [ %rd222 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs127, 0x0;
	ld.global.b16 { %rs127 }, [ %rd223 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs128, 0x0;
	ld.global.b16 { %rs128 }, [ %rd224 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs129, 0x0;
	ld.global.b16 { %rs129 }, [ %rd225 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs130, 0x0;
	ld.global.b16 { %rs130 }, [ %rd226 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs131, 0x0;
	ld.global.b16 { %rs131 }, [ %rd227 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs132, 0x0;
	ld.global.b16 { %rs132 }, [ %rd228 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs133, 0x0;
	ld.global.b16 { %rs133 }, [ %rd229 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs134, 0x0;
	ld.global.b16 { %rs134 }, [ %rd230 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs135, 0x0;
	ld.global.b16 { %rs135 }, [ %rd231 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs136, 0x0;
	ld.global.b16 { %rs136 }, [ %rd232 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs137, 0x0;
	ld.global.b16 { %rs137 }, [ %rd233 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs138, 0x0;
	ld.global.b16 { %rs138 }, [ %rd234 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs139, 0x0;
	ld.global.b16 { %rs139 }, [ %rd235 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs140, 0x0;
	ld.global.b16 { %rs140 }, [ %rd236 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs141, 0x0;
	ld.global.b16 { %rs141 }, [ %rd237 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs142, 0x0;
	ld.global.b16 { %rs142 }, [ %rd238 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs143, 0x0;
	ld.global.b16 { %rs143 }, [ %rd239 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs144, 0x0;
	ld.global.b16 { %rs144 }, [ %rd240 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs145, 0x0;
	ld.global.b16 { %rs145 }, [ %rd241 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs146, 0x0;
	ld.global.b16 { %rs146 }, [ %rd242 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs147, 0x0;
	ld.global.b16 { %rs147 }, [ %rd243 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs148, 0x0;
	ld.global.b16 { %rs148 }, [ %rd244 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs149, 0x0;
	ld.global.b16 { %rs149 }, [ %rd245 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs150, 0x0;
	ld.global.b16 { %rs150 }, [ %rd246 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs151, 0x0;
	ld.global.b16 { %rs151 }, [ %rd247 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs152, 0x0;
	ld.global.b16 { %rs152 }, [ %rd248 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs153, 0x0;
	ld.global.b16 { %rs153 }, [ %rd249 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs154, 0x0;
	ld.global.b16 { %rs154 }, [ %rd250 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs155, 0x0;
	ld.global.b16 { %rs155 }, [ %rd251 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs156, 0x0;
	ld.global.b16 { %rs156 }, [ %rd252 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs157, 0x0;
	ld.global.b16 { %rs157 }, [ %rd253 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs158, 0x0;
	ld.global.b16 { %rs158 }, [ %rd254 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs159, 0x0;
	ld.global.b16 { %rs159 }, [ %rd255 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs160, 0x0;
	ld.global.b16 { %rs160 }, [ %rd256 + 0 ];
	// end inline asm
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	bar.sync 	0;
	shl.b32 	%r895, %r2, 7;
	and.b32 	%r896, %r895, 15360;
	shl.b32 	%r897, %r860, 4;
	or.b32 	%r898, %r896, %r897;
	xor.b32 	%r899, %r898, %r826;
	add.s32 	%r612, %r208, %r899;
	mov.b32 	%r613, {%rs33, %rs34};
	mov.b32 	%r614, {%rs35, %rs36};
	mov.b32 	%r615, {%rs37, %rs38};
	mov.b32 	%r616, {%rs39, %rs40};
	// begin inline asm
	st.shared.v4.b32 [ %r612 + 0 ], { %r613, %r614, %r615, %r616 };
	// end inline asm
	add.s32 	%r617, %r612, 256;
	mov.b32 	%r618, {%rs41, %rs42};
	mov.b32 	%r619, {%rs43, %rs44};
	mov.b32 	%r620, {%rs45, %rs46};
	mov.b32 	%r621, {%rs47, %rs48};
	// begin inline asm
	st.shared.v4.b32 [ %r617 + 0 ], { %r618, %r619, %r620, %r621 };
	// end inline asm
	add.s32 	%r622, %r612, 512;
	mov.b32 	%r623, {%rs49, %rs50};
	mov.b32 	%r624, {%rs51, %rs52};
	mov.b32 	%r625, {%rs53, %rs54};
	mov.b32 	%r626, {%rs55, %rs56};
	// begin inline asm
	st.shared.v4.b32 [ %r622 + 0 ], { %r623, %r624, %r625, %r626 };
	// end inline asm
	add.s32 	%r627, %r612, 768;
	mov.b32 	%r628, {%rs57, %rs58};
	mov.b32 	%r629, {%rs59, %rs60};
	mov.b32 	%r630, {%rs61, %rs62};
	mov.b32 	%r631, {%rs63, %rs64};
	// begin inline asm
	st.shared.v4.b32 [ %r627 + 0 ], { %r628, %r629, %r630, %r631 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r900, %r860, 11;
	shl.b32 	%r901, %r6, 4;
	shl.b32 	%r902, %r868, 2;
	setp.eq.b32 	%p24, %r1406, 0;
	shl.b32 	%r903, %r1406, 1;
	shr.u32 	%r904, %r5, 1;
	or.b32 	%r905, %r901, %r902;
	or.b32 	%r906, %r903, %r904;
	xor.b32 	%r907, %r905, %r906;
	or.b32 	%r908, %r907, %r900;
	add.s32 	%r909, %r208, %r908;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r910, %r911, %r912, %r913}, [%r909];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r914, %r915, %r916, %r917}, [%r909+1024];
	xor.b32 	%r918, %r908, 64;
	add.s32 	%r919, %r208, %r918;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r920, %r921, %r922, %r923}, [%r919];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r924, %r925, %r926, %r927}, [%r919+1024];
	bar.sync 	0;
	mov.b32 	%r632, {%rs65, %rs66};
	mov.b32 	%r633, {%rs67, %rs68};
	mov.b32 	%r634, {%rs69, %rs70};
	mov.b32 	%r635, {%rs71, %rs72};
	// begin inline asm
	st.shared.v4.b32 [ %r612 + 0 ], { %r632, %r633, %r634, %r635 };
	// end inline asm
	mov.b32 	%r636, {%rs73, %rs74};
	mov.b32 	%r637, {%rs75, %rs76};
	mov.b32 	%r638, {%rs77, %rs78};
	mov.b32 	%r639, {%rs79, %rs80};
	// begin inline asm
	st.shared.v4.b32 [ %r617 + 0 ], { %r636, %r637, %r638, %r639 };
	// end inline asm
	mov.b32 	%r640, {%rs81, %rs82};
	mov.b32 	%r641, {%rs83, %rs84};
	mov.b32 	%r642, {%rs85, %rs86};
	mov.b32 	%r643, {%rs87, %rs88};
	// begin inline asm
	st.shared.v4.b32 [ %r622 + 0 ], { %r640, %r641, %r642, %r643 };
	// end inline asm
	mov.b32 	%r644, {%rs89, %rs90};
	mov.b32 	%r645, {%rs91, %rs92};
	mov.b32 	%r646, {%rs93, %rs94};
	mov.b32 	%r647, {%rs95, %rs96};
	// begin inline asm
	st.shared.v4.b32 [ %r627 + 0 ], { %r644, %r645, %r646, %r647 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r928, %r929, %r930, %r931}, [%r909];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r932, %r933, %r934, %r935}, [%r909+1024];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r936, %r937, %r938, %r939}, [%r919];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r940, %r941, %r942, %r943}, [%r919+1024];
	bar.sync 	0;
	mov.b32 	%r648, {%rs97, %rs98};
	mov.b32 	%r649, {%rs99, %rs100};
	mov.b32 	%r650, {%rs101, %rs102};
	mov.b32 	%r651, {%rs103, %rs104};
	// begin inline asm
	st.shared.v4.b32 [ %r612 + 0 ], { %r648, %r649, %r650, %r651 };
	// end inline asm
	mov.b32 	%r652, {%rs105, %rs106};
	mov.b32 	%r653, {%rs107, %rs108};
	mov.b32 	%r654, {%rs109, %rs110};
	mov.b32 	%r655, {%rs111, %rs112};
	// begin inline asm
	st.shared.v4.b32 [ %r617 + 0 ], { %r652, %r653, %r654, %r655 };
	// end inline asm
	mov.b32 	%r656, {%rs113, %rs114};
	mov.b32 	%r657, {%rs115, %rs116};
	mov.b32 	%r658, {%rs117, %rs118};
	mov.b32 	%r659, {%rs119, %rs120};
	// begin inline asm
	st.shared.v4.b32 [ %r622 + 0 ], { %r656, %r657, %r658, %r659 };
	// end inline asm
	mov.b32 	%r660, {%rs121, %rs122};
	mov.b32 	%r661, {%rs123, %rs124};
	mov.b32 	%r662, {%rs125, %rs126};
	mov.b32 	%r663, {%rs127, %rs128};
	// begin inline asm
	st.shared.v4.b32 [ %r627 + 0 ], { %r660, %r661, %r662, %r663 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r944, %r945, %r946, %r947}, [%r909];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r948, %r949, %r950, %r951}, [%r909+1024];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r952, %r953, %r954, %r955}, [%r919];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r956, %r957, %r958, %r959}, [%r919+1024];
	bar.sync 	0;
	mov.b32 	%r664, {%rs129, %rs130};
	mov.b32 	%r665, {%rs131, %rs132};
	mov.b32 	%r666, {%rs133, %rs134};
	mov.b32 	%r667, {%rs135, %rs136};
	// begin inline asm
	st.shared.v4.b32 [ %r612 + 0 ], { %r664, %r665, %r666, %r667 };
	// end inline asm
	mov.b32 	%r668, {%rs137, %rs138};
	mov.b32 	%r669, {%rs139, %rs140};
	mov.b32 	%r670, {%rs141, %rs142};
	mov.b32 	%r671, {%rs143, %rs144};
	// begin inline asm
	st.shared.v4.b32 [ %r617 + 0 ], { %r668, %r669, %r670, %r671 };
	// end inline asm
	mov.b32 	%r672, {%rs145, %rs146};
	mov.b32 	%r673, {%rs147, %rs148};
	mov.b32 	%r674, {%rs149, %rs150};
	mov.b32 	%r675, {%rs151, %rs152};
	// begin inline asm
	st.shared.v4.b32 [ %r622 + 0 ], { %r672, %r673, %r674, %r675 };
	// end inline asm
	mov.b32 	%r676, {%rs153, %rs154};
	mov.b32 	%r677, {%rs155, %rs156};
	mov.b32 	%r678, {%rs157, %rs158};
	mov.b32 	%r679, {%rs159, %rs160};
	// begin inline asm
	st.shared.v4.b32 [ %r627 + 0 ], { %r676, %r677, %r678, %r679 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r960, %r961, %r962, %r963}, [%r909];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r964, %r965, %r966, %r967}, [%r909+1024];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r968, %r969, %r970, %r971}, [%r919];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r972, %r973, %r974, %r975}, [%r919+1024];
	.loc	1 173 31                        // sk07_lm_head.py:173:31
	setp.lt.s32 	%p25, %r828, %r15;
	setp.lt.s32 	%p26, %r857, %r15;
	setp.lt.s32 	%p27, %r855, %r15;
	setp.lt.s32 	%p28, %r853, %r15;
	setp.lt.s32 	%p29, %r851, %r15;
	setp.lt.s32 	%p30, %r849, %r15;
	setp.lt.s32 	%p31, %r847, %r15;
	setp.lt.s32 	%p32, %r845, %r15;
	setp.lt.s32 	%p33, %r843, %r15;
	setp.lt.s32 	%p34, %r841, %r15;
	setp.lt.s32 	%p35, %r839, %r15;
	setp.lt.s32 	%p36, %r837, %r15;
	setp.lt.s32 	%p37, %r835, %r15;
	setp.lt.s32 	%p38, %r833, %r15;
	setp.lt.s32 	%p39, %r831, %r15;
	setp.lt.s32 	%p40, %r829, %r15;
	.loc	1 173 54                        // sk07_lm_head.py:173:54
	setp.lt.s32 	%p41, %r7, %r16;
	.loc	1 173 37                        // sk07_lm_head.py:173:37
	and.pred 	%p8, %p25, %p41;
	and.pred 	%p9, %p26, %p41;
	and.pred 	%p10, %p27, %p41;
	and.pred 	%p11, %p28, %p41;
	and.pred 	%p12, %p29, %p41;
	and.pred 	%p13, %p30, %p41;
	and.pred 	%p14, %p31, %p41;
	and.pred 	%p15, %p32, %p41;
	and.pred 	%p16, %p33, %p41;
	and.pred 	%p17, %p34, %p41;
	and.pred 	%p18, %p35, %p41;
	and.pred 	%p19, %p36, %p41;
	and.pred 	%p20, %p37, %p41;
	and.pred 	%p21, %p38, %p41;
	and.pred 	%p22, %p39, %p41;
	and.pred 	%p23, %p40, %p41;
	.loc	1 171 35                        // sk07_lm_head.py:171:35
	mul.lo.s32 	%r976, %r828, %r17;
	mul.lo.s32 	%r977, %r857, %r17;
	mul.lo.s32 	%r978, %r855, %r17;
	mul.lo.s32 	%r979, %r853, %r17;
	mul.lo.s32 	%r980, %r851, %r17;
	mul.lo.s32 	%r981, %r849, %r17;
	mul.lo.s32 	%r982, %r847, %r17;
	mul.lo.s32 	%r983, %r845, %r17;
	mul.lo.s32 	%r984, %r843, %r17;
	mul.lo.s32 	%r985, %r841, %r17;
	mul.lo.s32 	%r986, %r839, %r17;
	mul.lo.s32 	%r987, %r837, %r17;
	mul.lo.s32 	%r988, %r835, %r17;
	mul.lo.s32 	%r989, %r833, %r17;
	mul.lo.s32 	%r990, %r831, %r17;
	mul.lo.s32 	%r991, %r829, %r17;
	.loc	1 171 18                        // sk07_lm_head.py:171:18
	mad.wide.s32 	%rd297, %r976, 2, %rd26;
	mad.wide.s32 	%rd298, %r977, 2, %rd26;
	mad.wide.s32 	%rd299, %r978, 2, %rd26;
	mad.wide.s32 	%rd300, %r979, 2, %rd26;
	mad.wide.s32 	%rd301, %r980, 2, %rd26;
	mad.wide.s32 	%rd302, %r981, 2, %rd26;
	mad.wide.s32 	%rd303, %r982, 2, %rd26;
	mad.wide.s32 	%rd304, %r983, 2, %rd26;
	mad.wide.s32 	%rd305, %r984, 2, %rd26;
	mad.wide.s32 	%rd306, %r985, 2, %rd26;
	mad.wide.s32 	%rd307, %r986, 2, %rd26;
	mad.wide.s32 	%rd308, %r987, 2, %rd26;
	mad.wide.s32 	%rd309, %r988, 2, %rd26;
	mad.wide.s32 	%rd310, %r989, 2, %rd26;
	mad.wide.s32 	%rd311, %r990, 2, %rd26;
	mad.wide.s32 	%rd312, %r991, 2, %rd26;
	.loc	1 171 50                        // sk07_lm_head.py:171:50
	mul.wide.s32 	%rd313, %r7, 2;
	add.s64 	%rd257, %rd297, %rd313;
	add.s64 	%rd258, %rd298, %rd313;
	add.s64 	%rd259, %rd299, %rd313;
	add.s64 	%rd260, %rd300, %rd313;
	add.s64 	%rd261, %rd301, %rd313;
	add.s64 	%rd262, %rd302, %rd313;
	add.s64 	%rd263, %rd303, %rd313;
	add.s64 	%rd264, %rd304, %rd313;
	add.s64 	%rd265, %rd305, %rd313;
	add.s64 	%rd266, %rd306, %rd313;
	add.s64 	%rd267, %rd307, %rd313;
	add.s64 	%rd268, %rd308, %rd313;
	add.s64 	%rd269, %rd309, %rd313;
	add.s64 	%rd270, %rd310, %rd313;
	add.s64 	%rd271, %rd311, %rd313;
	add.s64 	%rd272, %rd312, %rd313;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r992, %r1504, %r877;
	mul.f32 	%r993, %r1503, %r877;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs161, %rs162}, %r960;
	cvt.f32.bf16 	%r994, %rs162;
	cvt.f32.bf16 	%r995, %rs161;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r996, %r993, %r596, %r995;
	fma.rn.f32 	%r997, %r992, %r597, %r994;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r998, %r1408, %r871;
	mul.f32 	%r999, %r1407, %r871;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs163, %rs164}, %r910;
	cvt.f32.bf16 	%r1000, %rs164;
	cvt.f32.bf16 	%r1001, %rs163;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1002, %r999, %r596, %r1001;
	fma.rn.f32 	%r1003, %r998, %r597, %r1000;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r681, %r1003, %r1002;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1004, %r1410, %r872;
	mul.f32 	%r1005, %r1409, %r872;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs165, %rs166}, %r911;
	cvt.f32.bf16 	%r1006, %rs166;
	cvt.f32.bf16 	%r1007, %rs165;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1008, %r1005, %r596, %r1007;
	fma.rn.f32 	%r1009, %r1004, %r597, %r1006;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r686, %r1009, %r1008;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1010, %r1440, %r873;
	mul.f32 	%r1011, %r1439, %r873;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs167, %rs168}, %r928;
	cvt.f32.bf16 	%r1012, %rs168;
	cvt.f32.bf16 	%r1013, %rs167;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1014, %r1011, %r596, %r1013;
	fma.rn.f32 	%r1015, %r1010, %r597, %r1012;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r682, %r1015, %r1014;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1016, %r1442, %r874;
	mul.f32 	%r1017, %r1441, %r874;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs169, %rs170}, %r929;
	cvt.f32.bf16 	%r1018, %rs170;
	cvt.f32.bf16 	%r1019, %rs169;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1020, %r1017, %r596, %r1019;
	fma.rn.f32 	%r1021, %r1016, %r597, %r1018;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r687, %r1021, %r1020;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1022, %r1472, %r875;
	mul.f32 	%r1023, %r1471, %r875;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs171, %rs172}, %r944;
	cvt.f32.bf16 	%r1024, %rs172;
	cvt.f32.bf16 	%r1025, %rs171;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1026, %r1023, %r596, %r1025;
	fma.rn.f32 	%r1027, %r1022, %r597, %r1024;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r683, %r1027, %r1026;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1028, %r1474, %r876;
	mul.f32 	%r1029, %r1473, %r876;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs173, %rs174}, %r945;
	cvt.f32.bf16 	%r1030, %rs174;
	cvt.f32.bf16 	%r1031, %rs173;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1032, %r1029, %r596, %r1031;
	fma.rn.f32 	%r1033, %r1028, %r597, %r1030;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r688, %r1033, %r1032;
	cvt.rn.bf16x2.f32 	%r684, %r997, %r996;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1034, %r1506, %r878;
	mul.f32 	%r1035, %r1505, %r878;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs175, %rs176}, %r961;
	cvt.f32.bf16 	%r1036, %rs176;
	cvt.f32.bf16 	%r1037, %rs175;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1038, %r1035, %r596, %r1037;
	fma.rn.f32 	%r1039, %r1034, %r597, %r1036;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r689, %r1039, %r1038;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1040, %r1508, %r877;
	mul.f32 	%r1041, %r1507, %r877;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs177, %rs178}, %r962;
	cvt.f32.bf16 	%r1042, %rs178;
	cvt.f32.bf16 	%r1043, %rs177;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1044, %r1041, %r598, %r1043;
	fma.rn.f32 	%r1045, %r1040, %r599, %r1042;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1046, %r1412, %r871;
	mul.f32 	%r1047, %r1411, %r871;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs179, %rs180}, %r912;
	cvt.f32.bf16 	%r1048, %rs180;
	cvt.f32.bf16 	%r1049, %rs179;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1050, %r1047, %r598, %r1049;
	fma.rn.f32 	%r1051, %r1046, %r599, %r1048;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r701, %r1051, %r1050;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1052, %r1414, %r872;
	mul.f32 	%r1053, %r1413, %r872;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs181, %rs182}, %r913;
	cvt.f32.bf16 	%r1054, %rs182;
	cvt.f32.bf16 	%r1055, %rs181;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1056, %r1053, %r598, %r1055;
	fma.rn.f32 	%r1057, %r1052, %r599, %r1054;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r706, %r1057, %r1056;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1058, %r1444, %r873;
	mul.f32 	%r1059, %r1443, %r873;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs183, %rs184}, %r930;
	cvt.f32.bf16 	%r1060, %rs184;
	cvt.f32.bf16 	%r1061, %rs183;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1062, %r1059, %r598, %r1061;
	fma.rn.f32 	%r1063, %r1058, %r599, %r1060;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r702, %r1063, %r1062;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1064, %r1446, %r874;
	mul.f32 	%r1065, %r1445, %r874;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs185, %rs186}, %r931;
	cvt.f32.bf16 	%r1066, %rs186;
	cvt.f32.bf16 	%r1067, %rs185;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1068, %r1065, %r598, %r1067;
	fma.rn.f32 	%r1069, %r1064, %r599, %r1066;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r707, %r1069, %r1068;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1070, %r1476, %r875;
	mul.f32 	%r1071, %r1475, %r875;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs187, %rs188}, %r946;
	cvt.f32.bf16 	%r1072, %rs188;
	cvt.f32.bf16 	%r1073, %rs187;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1074, %r1071, %r598, %r1073;
	fma.rn.f32 	%r1075, %r1070, %r599, %r1072;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r703, %r1075, %r1074;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1076, %r1478, %r876;
	mul.f32 	%r1077, %r1477, %r876;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs189, %rs190}, %r947;
	cvt.f32.bf16 	%r1078, %rs190;
	cvt.f32.bf16 	%r1079, %rs189;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1080, %r1077, %r598, %r1079;
	fma.rn.f32 	%r1081, %r1076, %r599, %r1078;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r708, %r1081, %r1080;
	cvt.rn.bf16x2.f32 	%r704, %r1045, %r1044;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1082, %r1510, %r878;
	mul.f32 	%r1083, %r1509, %r878;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs191, %rs192}, %r963;
	cvt.f32.bf16 	%r1084, %rs192;
	cvt.f32.bf16 	%r1085, %rs191;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1086, %r1083, %r598, %r1085;
	fma.rn.f32 	%r1087, %r1082, %r599, %r1084;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r709, %r1087, %r1086;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1088, %r1512, %r877;
	mul.f32 	%r1089, %r1511, %r877;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs193, %rs194}, %r968;
	cvt.f32.bf16 	%r1090, %rs194;
	cvt.f32.bf16 	%r1091, %rs193;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1092, %r1089, %r600, %r1091;
	fma.rn.f32 	%r1093, %r1088, %r601, %r1090;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1094, %r1416, %r871;
	mul.f32 	%r1095, %r1415, %r871;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs195, %rs196}, %r920;
	cvt.f32.bf16 	%r1096, %rs196;
	cvt.f32.bf16 	%r1097, %rs195;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1098, %r1095, %r600, %r1097;
	fma.rn.f32 	%r1099, %r1094, %r601, %r1096;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r721, %r1099, %r1098;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1100, %r1418, %r872;
	mul.f32 	%r1101, %r1417, %r872;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs197, %rs198}, %r921;
	cvt.f32.bf16 	%r1102, %rs198;
	cvt.f32.bf16 	%r1103, %rs197;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1104, %r1101, %r600, %r1103;
	fma.rn.f32 	%r1105, %r1100, %r601, %r1102;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r726, %r1105, %r1104;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1106, %r1448, %r873;
	mul.f32 	%r1107, %r1447, %r873;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs199, %rs200}, %r936;
	cvt.f32.bf16 	%r1108, %rs200;
	cvt.f32.bf16 	%r1109, %rs199;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1110, %r1107, %r600, %r1109;
	fma.rn.f32 	%r1111, %r1106, %r601, %r1108;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r722, %r1111, %r1110;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1112, %r1450, %r874;
	mul.f32 	%r1113, %r1449, %r874;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs201, %rs202}, %r937;
	cvt.f32.bf16 	%r1114, %rs202;
	cvt.f32.bf16 	%r1115, %rs201;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1116, %r1113, %r600, %r1115;
	fma.rn.f32 	%r1117, %r1112, %r601, %r1114;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r727, %r1117, %r1116;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1118, %r1480, %r875;
	mul.f32 	%r1119, %r1479, %r875;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs203, %rs204}, %r952;
	cvt.f32.bf16 	%r1120, %rs204;
	cvt.f32.bf16 	%r1121, %rs203;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1122, %r1119, %r600, %r1121;
	fma.rn.f32 	%r1123, %r1118, %r601, %r1120;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r723, %r1123, %r1122;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1124, %r1482, %r876;
	mul.f32 	%r1125, %r1481, %r876;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs205, %rs206}, %r953;
	cvt.f32.bf16 	%r1126, %rs206;
	cvt.f32.bf16 	%r1127, %rs205;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1128, %r1125, %r600, %r1127;
	fma.rn.f32 	%r1129, %r1124, %r601, %r1126;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r728, %r1129, %r1128;
	cvt.rn.bf16x2.f32 	%r724, %r1093, %r1092;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1130, %r1514, %r878;
	mul.f32 	%r1131, %r1513, %r878;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs207, %rs208}, %r969;
	cvt.f32.bf16 	%r1132, %rs208;
	cvt.f32.bf16 	%r1133, %rs207;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1134, %r1131, %r600, %r1133;
	fma.rn.f32 	%r1135, %r1130, %r601, %r1132;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r729, %r1135, %r1134;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1136, %r1516, %r877;
	mul.f32 	%r1137, %r1515, %r877;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs209, %rs210}, %r970;
	cvt.f32.bf16 	%r1138, %rs210;
	cvt.f32.bf16 	%r1139, %rs209;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1140, %r1137, %r602, %r1139;
	fma.rn.f32 	%r1141, %r1136, %r603, %r1138;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1142, %r1420, %r871;
	mul.f32 	%r1143, %r1419, %r871;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs211, %rs212}, %r922;
	cvt.f32.bf16 	%r1144, %rs212;
	cvt.f32.bf16 	%r1145, %rs211;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1146, %r1143, %r602, %r1145;
	fma.rn.f32 	%r1147, %r1142, %r603, %r1144;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r741, %r1147, %r1146;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1148, %r1422, %r872;
	mul.f32 	%r1149, %r1421, %r872;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs213, %rs214}, %r923;
	cvt.f32.bf16 	%r1150, %rs214;
	cvt.f32.bf16 	%r1151, %rs213;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1152, %r1149, %r602, %r1151;
	fma.rn.f32 	%r1153, %r1148, %r603, %r1150;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r746, %r1153, %r1152;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1154, %r1452, %r873;
	mul.f32 	%r1155, %r1451, %r873;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs215, %rs216}, %r938;
	cvt.f32.bf16 	%r1156, %rs216;
	cvt.f32.bf16 	%r1157, %rs215;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1158, %r1155, %r602, %r1157;
	fma.rn.f32 	%r1159, %r1154, %r603, %r1156;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r742, %r1159, %r1158;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1160, %r1454, %r874;
	mul.f32 	%r1161, %r1453, %r874;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs217, %rs218}, %r939;
	cvt.f32.bf16 	%r1162, %rs218;
	cvt.f32.bf16 	%r1163, %rs217;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1164, %r1161, %r602, %r1163;
	fma.rn.f32 	%r1165, %r1160, %r603, %r1162;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r747, %r1165, %r1164;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1166, %r1484, %r875;
	mul.f32 	%r1167, %r1483, %r875;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs219, %rs220}, %r954;
	cvt.f32.bf16 	%r1168, %rs220;
	cvt.f32.bf16 	%r1169, %rs219;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1170, %r1167, %r602, %r1169;
	fma.rn.f32 	%r1171, %r1166, %r603, %r1168;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r743, %r1171, %r1170;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1172, %r1486, %r876;
	mul.f32 	%r1173, %r1485, %r876;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs221, %rs222}, %r955;
	cvt.f32.bf16 	%r1174, %rs222;
	cvt.f32.bf16 	%r1175, %rs221;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1176, %r1173, %r602, %r1175;
	fma.rn.f32 	%r1177, %r1172, %r603, %r1174;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r748, %r1177, %r1176;
	cvt.rn.bf16x2.f32 	%r744, %r1141, %r1140;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1178, %r1518, %r878;
	mul.f32 	%r1179, %r1517, %r878;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs223, %rs224}, %r971;
	cvt.f32.bf16 	%r1180, %rs224;
	cvt.f32.bf16 	%r1181, %rs223;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1182, %r1179, %r602, %r1181;
	fma.rn.f32 	%r1183, %r1178, %r603, %r1180;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r749, %r1183, %r1182;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1184, %r1520, %r877;
	mul.f32 	%r1185, %r1519, %r877;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs225, %rs226}, %r964;
	cvt.f32.bf16 	%r1186, %rs226;
	cvt.f32.bf16 	%r1187, %rs225;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1188, %r1185, %r604, %r1187;
	fma.rn.f32 	%r1189, %r1184, %r605, %r1186;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1190, %r1424, %r871;
	mul.f32 	%r1191, %r1423, %r871;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs227, %rs228}, %r914;
	cvt.f32.bf16 	%r1192, %rs228;
	cvt.f32.bf16 	%r1193, %rs227;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1194, %r1191, %r604, %r1193;
	fma.rn.f32 	%r1195, %r1190, %r605, %r1192;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r691, %r1195, %r1194;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1196, %r1426, %r872;
	mul.f32 	%r1197, %r1425, %r872;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs229, %rs230}, %r915;
	cvt.f32.bf16 	%r1198, %rs230;
	cvt.f32.bf16 	%r1199, %rs229;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1200, %r1197, %r604, %r1199;
	fma.rn.f32 	%r1201, %r1196, %r605, %r1198;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r696, %r1201, %r1200;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1202, %r1456, %r873;
	mul.f32 	%r1203, %r1455, %r873;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs231, %rs232}, %r932;
	cvt.f32.bf16 	%r1204, %rs232;
	cvt.f32.bf16 	%r1205, %rs231;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1206, %r1203, %r604, %r1205;
	fma.rn.f32 	%r1207, %r1202, %r605, %r1204;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r692, %r1207, %r1206;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1208, %r1458, %r874;
	mul.f32 	%r1209, %r1457, %r874;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs233, %rs234}, %r933;
	cvt.f32.bf16 	%r1210, %rs234;
	cvt.f32.bf16 	%r1211, %rs233;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1212, %r1209, %r604, %r1211;
	fma.rn.f32 	%r1213, %r1208, %r605, %r1210;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r697, %r1213, %r1212;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1214, %r1488, %r875;
	mul.f32 	%r1215, %r1487, %r875;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs235, %rs236}, %r948;
	cvt.f32.bf16 	%r1216, %rs236;
	cvt.f32.bf16 	%r1217, %rs235;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1218, %r1215, %r604, %r1217;
	fma.rn.f32 	%r1219, %r1214, %r605, %r1216;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r693, %r1219, %r1218;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1220, %r1490, %r876;
	mul.f32 	%r1221, %r1489, %r876;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs237, %rs238}, %r949;
	cvt.f32.bf16 	%r1222, %rs238;
	cvt.f32.bf16 	%r1223, %rs237;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1224, %r1221, %r604, %r1223;
	fma.rn.f32 	%r1225, %r1220, %r605, %r1222;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r698, %r1225, %r1224;
	cvt.rn.bf16x2.f32 	%r694, %r1189, %r1188;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1226, %r1522, %r878;
	mul.f32 	%r1227, %r1521, %r878;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs239, %rs240}, %r965;
	cvt.f32.bf16 	%r1228, %rs240;
	cvt.f32.bf16 	%r1229, %rs239;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1230, %r1227, %r604, %r1229;
	fma.rn.f32 	%r1231, %r1226, %r605, %r1228;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r699, %r1231, %r1230;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1232, %r1524, %r877;
	mul.f32 	%r1233, %r1523, %r877;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs241, %rs242}, %r966;
	cvt.f32.bf16 	%r1234, %rs242;
	cvt.f32.bf16 	%r1235, %rs241;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1236, %r1233, %r606, %r1235;
	fma.rn.f32 	%r1237, %r1232, %r607, %r1234;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1238, %r1428, %r871;
	mul.f32 	%r1239, %r1427, %r871;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs243, %rs244}, %r916;
	cvt.f32.bf16 	%r1240, %rs244;
	cvt.f32.bf16 	%r1241, %rs243;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1242, %r1239, %r606, %r1241;
	fma.rn.f32 	%r1243, %r1238, %r607, %r1240;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r711, %r1243, %r1242;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1244, %r1430, %r872;
	mul.f32 	%r1245, %r1429, %r872;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs245, %rs246}, %r917;
	cvt.f32.bf16 	%r1246, %rs246;
	cvt.f32.bf16 	%r1247, %rs245;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1248, %r1245, %r606, %r1247;
	fma.rn.f32 	%r1249, %r1244, %r607, %r1246;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r716, %r1249, %r1248;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1250, %r1460, %r873;
	mul.f32 	%r1251, %r1459, %r873;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs247, %rs248}, %r934;
	cvt.f32.bf16 	%r1252, %rs248;
	cvt.f32.bf16 	%r1253, %rs247;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1254, %r1251, %r606, %r1253;
	fma.rn.f32 	%r1255, %r1250, %r607, %r1252;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r712, %r1255, %r1254;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1256, %r1462, %r874;
	mul.f32 	%r1257, %r1461, %r874;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs249, %rs250}, %r935;
	cvt.f32.bf16 	%r1258, %rs250;
	cvt.f32.bf16 	%r1259, %rs249;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1260, %r1257, %r606, %r1259;
	fma.rn.f32 	%r1261, %r1256, %r607, %r1258;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r717, %r1261, %r1260;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1262, %r1492, %r875;
	mul.f32 	%r1263, %r1491, %r875;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs251, %rs252}, %r950;
	cvt.f32.bf16 	%r1264, %rs252;
	cvt.f32.bf16 	%r1265, %rs251;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1266, %r1263, %r606, %r1265;
	fma.rn.f32 	%r1267, %r1262, %r607, %r1264;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r713, %r1267, %r1266;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1268, %r1494, %r876;
	mul.f32 	%r1269, %r1493, %r876;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs253, %rs254}, %r951;
	cvt.f32.bf16 	%r1270, %rs254;
	cvt.f32.bf16 	%r1271, %rs253;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1272, %r1269, %r606, %r1271;
	fma.rn.f32 	%r1273, %r1268, %r607, %r1270;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r718, %r1273, %r1272;
	cvt.rn.bf16x2.f32 	%r714, %r1237, %r1236;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1274, %r1526, %r878;
	mul.f32 	%r1275, %r1525, %r878;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs255, %rs256}, %r967;
	cvt.f32.bf16 	%r1276, %rs256;
	cvt.f32.bf16 	%r1277, %rs255;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1278, %r1275, %r606, %r1277;
	fma.rn.f32 	%r1279, %r1274, %r607, %r1276;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r719, %r1279, %r1278;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1280, %r1528, %r877;
	mul.f32 	%r1281, %r1527, %r877;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs257, %rs258}, %r972;
	cvt.f32.bf16 	%r1282, %rs258;
	cvt.f32.bf16 	%r1283, %rs257;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1284, %r1281, %r608, %r1283;
	fma.rn.f32 	%r1285, %r1280, %r609, %r1282;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1286, %r1432, %r871;
	mul.f32 	%r1287, %r1431, %r871;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs259, %rs260}, %r924;
	cvt.f32.bf16 	%r1288, %rs260;
	cvt.f32.bf16 	%r1289, %rs259;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1290, %r1287, %r608, %r1289;
	fma.rn.f32 	%r1291, %r1286, %r609, %r1288;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r731, %r1291, %r1290;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1292, %r1434, %r872;
	mul.f32 	%r1293, %r1433, %r872;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs261, %rs262}, %r925;
	cvt.f32.bf16 	%r1294, %rs262;
	cvt.f32.bf16 	%r1295, %rs261;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1296, %r1293, %r608, %r1295;
	fma.rn.f32 	%r1297, %r1292, %r609, %r1294;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r736, %r1297, %r1296;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1298, %r1464, %r873;
	mul.f32 	%r1299, %r1463, %r873;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs263, %rs264}, %r940;
	cvt.f32.bf16 	%r1300, %rs264;
	cvt.f32.bf16 	%r1301, %rs263;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1302, %r1299, %r608, %r1301;
	fma.rn.f32 	%r1303, %r1298, %r609, %r1300;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r732, %r1303, %r1302;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1304, %r1466, %r874;
	mul.f32 	%r1305, %r1465, %r874;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs265, %rs266}, %r941;
	cvt.f32.bf16 	%r1306, %rs266;
	cvt.f32.bf16 	%r1307, %rs265;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1308, %r1305, %r608, %r1307;
	fma.rn.f32 	%r1309, %r1304, %r609, %r1306;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r737, %r1309, %r1308;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1310, %r1496, %r875;
	mul.f32 	%r1311, %r1495, %r875;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs267, %rs268}, %r956;
	cvt.f32.bf16 	%r1312, %rs268;
	cvt.f32.bf16 	%r1313, %rs267;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1314, %r1311, %r608, %r1313;
	fma.rn.f32 	%r1315, %r1310, %r609, %r1312;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r733, %r1315, %r1314;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1316, %r1498, %r876;
	mul.f32 	%r1317, %r1497, %r876;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs269, %rs270}, %r957;
	cvt.f32.bf16 	%r1318, %rs270;
	cvt.f32.bf16 	%r1319, %rs269;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1320, %r1317, %r608, %r1319;
	fma.rn.f32 	%r1321, %r1316, %r609, %r1318;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r738, %r1321, %r1320;
	cvt.rn.bf16x2.f32 	%r734, %r1285, %r1284;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1322, %r1530, %r878;
	mul.f32 	%r1323, %r1529, %r878;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs271, %rs272}, %r973;
	cvt.f32.bf16 	%r1324, %rs272;
	cvt.f32.bf16 	%r1325, %rs271;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1326, %r1323, %r608, %r1325;
	fma.rn.f32 	%r1327, %r1322, %r609, %r1324;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r739, %r1327, %r1326;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1328, %r1532, %r877;
	mul.f32 	%r1329, %r1531, %r877;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs273, %rs274}, %r974;
	cvt.f32.bf16 	%r1330, %rs274;
	cvt.f32.bf16 	%r1331, %rs273;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1332, %r1329, %r610, %r1331;
	fma.rn.f32 	%r1333, %r1328, %r611, %r1330;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1334, %r1436, %r871;
	mul.f32 	%r1335, %r1435, %r871;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs275, %rs276}, %r926;
	cvt.f32.bf16 	%r1336, %rs276;
	cvt.f32.bf16 	%r1337, %rs275;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1338, %r1335, %r610, %r1337;
	fma.rn.f32 	%r1339, %r1334, %r611, %r1336;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r751, %r1339, %r1338;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1340, %r1438, %r872;
	mul.f32 	%r1341, %r1437, %r872;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs277, %rs278}, %r927;
	cvt.f32.bf16 	%r1342, %rs278;
	cvt.f32.bf16 	%r1343, %rs277;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1344, %r1341, %r610, %r1343;
	fma.rn.f32 	%r1345, %r1340, %r611, %r1342;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r756, %r1345, %r1344;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1346, %r1468, %r873;
	mul.f32 	%r1347, %r1467, %r873;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs279, %rs280}, %r942;
	cvt.f32.bf16 	%r1348, %rs280;
	cvt.f32.bf16 	%r1349, %rs279;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1350, %r1347, %r610, %r1349;
	fma.rn.f32 	%r1351, %r1346, %r611, %r1348;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r752, %r1351, %r1350;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1352, %r1470, %r874;
	mul.f32 	%r1353, %r1469, %r874;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs281, %rs282}, %r943;
	cvt.f32.bf16 	%r1354, %rs282;
	cvt.f32.bf16 	%r1355, %rs281;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1356, %r1353, %r610, %r1355;
	fma.rn.f32 	%r1357, %r1352, %r611, %r1354;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r757, %r1357, %r1356;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1358, %r1500, %r875;
	mul.f32 	%r1359, %r1499, %r875;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs283, %rs284}, %r958;
	cvt.f32.bf16 	%r1360, %rs284;
	cvt.f32.bf16 	%r1361, %rs283;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1362, %r1359, %r610, %r1361;
	fma.rn.f32 	%r1363, %r1358, %r611, %r1360;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r753, %r1363, %r1362;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1364, %r1502, %r876;
	mul.f32 	%r1365, %r1501, %r876;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs285, %rs286}, %r959;
	cvt.f32.bf16 	%r1366, %rs286;
	cvt.f32.bf16 	%r1367, %rs285;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1368, %r1365, %r610, %r1367;
	fma.rn.f32 	%r1369, %r1364, %r611, %r1366;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r758, %r1369, %r1368;
	cvt.rn.bf16x2.f32 	%r754, %r1333, %r1332;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1370, %r1534, %r878;
	mul.f32 	%r1371, %r1533, %r878;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs287, %rs288}, %r975;
	cvt.f32.bf16 	%r1372, %rs288;
	cvt.f32.bf16 	%r1373, %rs287;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1374, %r1371, %r610, %r1373;
	fma.rn.f32 	%r1375, %r1370, %r611, %r1372;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r759, %r1375, %r1374;
	bar.sync 	0;
	shl.b32 	%r1376, %r4, 14;
	shl.b32 	%r1377, %r4, 5;
	and.b32 	%r1378, %r1405, 3456;
	bfe.s32 	%r1379, %r2, 2, 1;
	and.b32 	%r1380, %r1379, 8208;
	or.b32 	%r1381, %r1377, %r1378;
	xor.b32 	%r1382, %r1380, %r904;
	or.b32 	%r1383, %r1382, %r1381;
	or.b32 	%r1384, %r1383, %r1376;
	add.s32 	%r680, %r208, %r1384;
	// begin inline asm
	st.shared.v4.b32 [ %r680 + 0 ], { %r681, %r682, %r683, %r684 };
	// end inline asm
	add.s32 	%r685, %r680, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r685 + 0 ], { %r686, %r687, %r688, %r689 };
	// end inline asm
	add.s32 	%r690, %r680, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r690 + 0 ], { %r691, %r692, %r693, %r694 };
	// end inline asm
	add.s32 	%r695, %r680, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r695 + 0 ], { %r696, %r697, %r698, %r699 };
	// end inline asm
	xor.b32 	%r1385, %r1384, 32;
	add.s32 	%r700, %r208, %r1385;
	// begin inline asm
	st.shared.v4.b32 [ %r700 + 0 ], { %r701, %r702, %r703, %r704 };
	// end inline asm
	add.s32 	%r705, %r700, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r705 + 0 ], { %r706, %r707, %r708, %r709 };
	// end inline asm
	add.s32 	%r710, %r700, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r710 + 0 ], { %r711, %r712, %r713, %r714 };
	// end inline asm
	add.s32 	%r715, %r700, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r715 + 0 ], { %r716, %r717, %r718, %r719 };
	// end inline asm
	xor.b32 	%r1386, %r1384, 64;
	add.s32 	%r720, %r208, %r1386;
	// begin inline asm
	st.shared.v4.b32 [ %r720 + 0 ], { %r721, %r722, %r723, %r724 };
	// end inline asm
	add.s32 	%r725, %r720, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r725 + 0 ], { %r726, %r727, %r728, %r729 };
	// end inline asm
	add.s32 	%r730, %r720, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r730 + 0 ], { %r731, %r732, %r733, %r734 };
	// end inline asm
	add.s32 	%r735, %r720, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r735 + 0 ], { %r736, %r737, %r738, %r739 };
	// end inline asm
	xor.b32 	%r1387, %r1384, 96;
	add.s32 	%r740, %r208, %r1387;
	// begin inline asm
	st.shared.v4.b32 [ %r740 + 0 ], { %r741, %r742, %r743, %r744 };
	// end inline asm
	add.s32 	%r745, %r740, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r745 + 0 ], { %r746, %r747, %r748, %r749 };
	// end inline asm
	add.s32 	%r750, %r740, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r750 + 0 ], { %r751, %r752, %r753, %r754 };
	// end inline asm
	add.s32 	%r755, %r740, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r755 + 0 ], { %r756, %r757, %r758, %r759 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r1388, %r2, 2;
	and.b32 	%r1389, %r1388, 896;
	shl.b32 	%r1390, %r863, 9;
	selp.b32 	%r1391, 0, 8208, %p24;
	or.b32 	%r1392, %r897, %r1389;
	xor.b32 	%r1393, %r1392, %r1391;
	or.b32 	%r1394, %r1393, %r1390;
	add.s32 	%r1395, %r208, %r1394;
	ld.shared.v4.b32 	{%r760, %r776, %r792, %r808}, [%r1395];
	ld.shared.v4.b32 	{%r764, %r780, %r796, %r812}, [%r1395+1024];
	ld.shared.v4.b32 	{%r768, %r784, %r800, %r816}, [%r1395+2048];
	ld.shared.v4.b32 	{%r772, %r788, %r804, %r820}, [%r1395+3072];
	xor.b32 	%r1396, %r1394, 32;
	add.s32 	%r1397, %r208, %r1396;
	ld.shared.v4.b32 	{%r761, %r777, %r793, %r809}, [%r1397+16384];
	ld.shared.v4.b32 	{%r765, %r781, %r797, %r813}, [%r1397+17408];
	ld.shared.v4.b32 	{%r769, %r785, %r801, %r817}, [%r1397+18432];
	ld.shared.v4.b32 	{%r773, %r789, %r805, %r821}, [%r1397+19456];
	xor.b32 	%r1398, %r1394, 64;
	add.s32 	%r1399, %r208, %r1398;
	ld.shared.v4.b32 	{%r762, %r778, %r794, %r810}, [%r1399+32768];
	ld.shared.v4.b32 	{%r766, %r782, %r798, %r814}, [%r1399+33792];
	ld.shared.v4.b32 	{%r770, %r786, %r802, %r818}, [%r1399+34816];
	ld.shared.v4.b32 	{%r774, %r790, %r806, %r822}, [%r1399+35840];
	xor.b32 	%r1400, %r1394, 96;
	add.s32 	%r1401, %r208, %r1400;
	ld.shared.v4.b32 	{%r763, %r779, %r795, %r811}, [%r1401+49152];
	ld.shared.v4.b32 	{%r767, %r783, %r799, %r815}, [%r1401+50176];
	ld.shared.v4.b32 	{%r771, %r787, %r803, %r819}, [%r1401+51200];
	ld.shared.v4.b32 	{%r775, %r791, %r807, %r823}, [%r1401+52224];
	.loc	1 172 8                         // sk07_lm_head.py:172:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd257 + 0 ], { %r760, %r761, %r762, %r763 };
	// end inline asm
	// begin inline asm
	@%p9 st.global.v4.b32 [ %rd258 + 0 ], { %r764, %r765, %r766, %r767 };
	// end inline asm
	// begin inline asm
	@%p10 st.global.v4.b32 [ %rd259 + 0 ], { %r768, %r769, %r770, %r771 };
	// end inline asm
	// begin inline asm
	@%p11 st.global.v4.b32 [ %rd260 + 0 ], { %r772, %r773, %r774, %r775 };
	// end inline asm
	// begin inline asm
	@%p12 st.global.v4.b32 [ %rd261 + 0 ], { %r776, %r777, %r778, %r779 };
	// end inline asm
	// begin inline asm
	@%p13 st.global.v4.b32 [ %rd262 + 0 ], { %r780, %r781, %r782, %r783 };
	// end inline asm
	// begin inline asm
	@%p14 st.global.v4.b32 [ %rd263 + 0 ], { %r784, %r785, %r786, %r787 };
	// end inline asm
	// begin inline asm
	@%p15 st.global.v4.b32 [ %rd264 + 0 ], { %r788, %r789, %r790, %r791 };
	// end inline asm
	// begin inline asm
	@%p16 st.global.v4.b32 [ %rd265 + 0 ], { %r792, %r793, %r794, %r795 };
	// end inline asm
	// begin inline asm
	@%p17 st.global.v4.b32 [ %rd266 + 0 ], { %r796, %r797, %r798, %r799 };
	// end inline asm
	// begin inline asm
	@%p18 st.global.v4.b32 [ %rd267 + 0 ], { %r800, %r801, %r802, %r803 };
	// end inline asm
	// begin inline asm
	@%p19 st.global.v4.b32 [ %rd268 + 0 ], { %r804, %r805, %r806, %r807 };
	// end inline asm
	// begin inline asm
	@%p20 st.global.v4.b32 [ %rd269 + 0 ], { %r808, %r809, %r810, %r811 };
	// end inline asm
	// begin inline asm
	@%p21 st.global.v4.b32 [ %rd270 + 0 ], { %r812, %r813, %r814, %r815 };
	// end inline asm
	// begin inline asm
	@%p22 st.global.v4.b32 [ %rd271 + 0 ], { %r816, %r817, %r818, %r819 };
	// end inline asm
	// begin inline asm
	@%p23 st.global.v4.b32 [ %rd272 + 0 ], { %r820, %r821, %r822, %r823 };
	// end inline asm
	.loc	1 170 4                         // sk07_lm_head.py:170:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk07_lm_head.py"
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
.b32 159                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x98 DW_TAG_compile_unit
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
.b8 55
.b8 95
.b8 108
.b8 109
.b8 95
.b8 104
.b8 101
.b8 97
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
.b8 2                                   // Abbrev [2] 0x45:0x17 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 55
.b8 95
.b8 108
.b8 109
.b8 95
.b8 104
.b8 101
.b8 97
.b8 100
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x5c:0x46 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 69                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x71:0x18 DW_TAG_inlined_subroutine
.b32 69                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 133                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x89:0x18 DW_TAG_inlined_subroutine
.b32 69                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 134                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_4 = _Nativo(
    "sk07_lm_head/tile256x128x64_shift0_abi16",
    _PTX_4, "_sk07_lm_head_kernel",
    warps=8, shared=73728,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 17, 19],
    horneado={12: 1, 13: 1, 16: 1, 18: 1, 20: 1, 21: 256, 22: 128, 23: 64, 24: 8, 25: 128},
    div16=[9, 10, 11, 14, 15, 17],
)

_PTX_5 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk07_lm_head_kernel    // -- Begin function _sk07_lm_head_kernel
.extern .shared .align 16 .b8 global_smem[];
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
                                        // @_sk07_lm_head_kernel
.visible .entry _sk07_lm_head_kernel(
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_6,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_7,
	.param .u32 _sk07_lm_head_kernel_param_8,
	.param .u32 _sk07_lm_head_kernel_param_9,
	.param .u32 _sk07_lm_head_kernel_param_10,
	.param .u32 _sk07_lm_head_kernel_param_11,
	.param .u32 _sk07_lm_head_kernel_param_12,
	.param .u32 _sk07_lm_head_kernel_param_13,
	.param .u32 _sk07_lm_head_kernel_param_14,
	.param .u32 _sk07_lm_head_kernel_param_15,
	.param .u32 _sk07_lm_head_kernel_param_16,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_17,
	.param .u64 .ptr .global .align 1 _sk07_lm_head_kernel_param_18
)
.reqntid 256
{
	.reg .pred 	%p<42>;
	.reg .b16 	%rs<289>;
	.reg .b32 	%r<1544>;
	.reg .b64 	%rd<320>;
	.loc	1 123 0                         // sk07_lm_head.py:123:0
$L__func_begin0:
	.loc	1 123 0                         // sk07_lm_head.py:123:0

// %bb.0:
	ld.param.b32 	%r19, [_sk07_lm_head_kernel_param_15];
	ld.param.b32 	%r18, [_sk07_lm_head_kernel_param_14];
	ld.param.b32 	%r17, [_sk07_lm_head_kernel_param_13];
	ld.param.b32 	%r16, [_sk07_lm_head_kernel_param_9];
	ld.param.b32 	%r15, [_sk07_lm_head_kernel_param_8];
	ld.param.b64 	%rd29, [_sk07_lm_head_kernel_param_6];
	ld.param.b64 	%rd28, [_sk07_lm_head_kernel_param_5];
	ld.param.b64 	%rd27, [_sk07_lm_head_kernel_param_4];
	ld.param.b64 	%rd26, [_sk07_lm_head_kernel_param_2];
	ld.param.b64 	%rd25, [_sk07_lm_head_kernel_param_1];
	ld.param.b64 	%rd24, [_sk07_lm_head_kernel_param_0];
$L__tmp0:
	.loc	1 132 24                        // sk07_lm_head.py:132:24
	mov.u32 	%r68, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk07_lm_head.py:133:27 ]
	add.s32 	%r69, %r15, 255;
	.loc	2 43 30                         // standard.py:43:30 @[ sk07_lm_head.py:133:27 ]
	shr.s32 	%r70, %r69, 31;
	shr.u32 	%r71, %r70, 24;
	add.s32 	%r72, %r69, %r71;
	shr.s32 	%r73, %r72, 8;
	ld.param.b64 	%rd60, [_sk07_lm_head_kernel_param_3];
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk07_lm_head.py:134:27 ]
	add.s32 	%r74, %r16, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk07_lm_head.py:134:27 ]
	shr.s32 	%r75, %r74, 31;
	shr.u32 	%r76, %r75, 25;
	add.s32 	%r77, %r74, %r76;
	shr.s32 	%r78, %r77, 7;
$L__tmp3:
	.loc	1 135 29                        // sk07_lm_head.py:135:29
	shl.b32 	%r79, %r78, 3;
	.loc	1 136 22                        // sk07_lm_head.py:136:22
	div.s32 	%r80, %r68, %r79;
	.loc	1 136 38                        // sk07_lm_head.py:136:38
	shl.b32 	%r81, %r80, 3;
	.loc	1 137 30                        // sk07_lm_head.py:137:30
	sub.s32 	%r82, %r73, %r81;
	ld.param.b32 	%r83, [_sk07_lm_head_kernel_param_10];
	.loc	1 137 39                        // sk07_lm_head.py:137:39
	min.s32 	%r84, %r82, 8;
	ld.param.b32 	%r85, [_sk07_lm_head_kernel_param_11];
	.loc	1 138 30                        // sk07_lm_head.py:138:30
	mul.lo.s32 	%r86, %r80, %r79;
	ld.param.b32 	%r87, [_sk07_lm_head_kernel_param_12];
	sub.s32 	%r88, %r68, %r86;
	.loc	1 139 36                        // sk07_lm_head.py:139:36
	div.s32 	%r89, %r88, %r84;
	.loc	1 138 46                        // sk07_lm_head.py:138:46
	mul.lo.s32 	%r90, %r89, %r84;
	sub.s32 	%r91, %r88, %r90;
	.loc	1 138 23                        // sk07_lm_head.py:138:23
	add.s32 	%r92, %r91, %r81;
	.loc	1 141 22                        // sk07_lm_head.py:141:22
	shl.b32 	%r1, %r92, 8;
	.loc	1 141 45                        // sk07_lm_head.py:141:45
	mov.u32 	%r2, %tid.x;
	shr.u32 	%r93, %r2, 2;
	bfe.u32 	%r94, %r2, 2, 6;
	or.b32 	%r95, %r94, 64;
	and.b32 	%r3, %r2, 255;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r96, %r1, %r94;
	or.b32 	%r97, %r1, %r95;
	or.b32 	%r98, %r96, 128;
	or.b32 	%r99, %r1, %r93;
	or.b32 	%r100, %r99, 192;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r101, %r96, %r15;
	rem.s32 	%r102, %r97, %r15;
	rem.s32 	%r103, %r98, %r15;
	rem.s32 	%r104, %r100, %r15;
	.loc	1 142 22                        // sk07_lm_head.py:142:22
	shl.b32 	%r105, %r89, 7;
	.loc	1 142 45                        // sk07_lm_head.py:142:45
	and.b32 	%r4, %r2, 3;
	shl.b32 	%r106, %r4, 1;
	and.b32 	%r5, %r2, 32;
	shr.u32 	%r107, %r5, 2;
	or.b32 	%r108, %r107, %r106;
	and.b32 	%r6, %r2, 15;
	shl.b32 	%r109, %r6, 3;
	.loc	1 142 32                        // sk07_lm_head.py:142:32
	or.b32 	%r110, %r105, %r94;
	or.b32 	%r111, %r105, %r95;
	or.b32 	%r112, %r105, %r108;
	or.b32 	%r113, %r112, 16;
	or.b32 	%r114, %r112, 32;
	or.b32 	%r115, %r112, 48;
	or.b32 	%r116, %r112, 64;
	or.b32 	%r117, %r112, 80;
	or.b32 	%r118, %r112, 96;
	or.b32 	%r119, %r112, 112;
	or.b32 	%r7, %r105, %r109;
	or.b32 	%r120, %r7, 4;
	.loc	1 142 57                        // sk07_lm_head.py:142:57
	rem.s32 	%r121, %r110, %r16;
	rem.s32 	%r122, %r111, %r16;
	rem.s32 	%r123, %r112, %r16;
	rem.s32 	%r124, %r113, %r16;
	rem.s32 	%r125, %r114, %r16;
	rem.s32 	%r126, %r115, %r16;
	rem.s32 	%r127, %r116, %r16;
	rem.s32 	%r128, %r117, %r16;
	rem.s32 	%r129, %r118, %r16;
	rem.s32 	%r130, %r119, %r16;
	rem.s32 	%r131, %r7, %r16;
	rem.s32 	%r132, %r120, %r16;
	.loc	1 145 29                        // sk07_lm_head.py:145:29
	mad.wide.s32 	%rd30, %r121, 4, %rd60;
	mad.wide.s32 	%rd31, %r122, 4, %rd60;
	mad.wide.s32 	%rd32, %r123, 4, %rd60;
	mad.wide.s32 	%rd33, %r124, 4, %rd60;
	mad.wide.s32 	%rd34, %r125, 4, %rd60;
	mad.wide.s32 	%rd35, %r126, 4, %rd60;
	mad.wide.s32 	%rd36, %r127, 4, %rd60;
	mad.wide.s32 	%rd37, %r128, 4, %rd60;
	mad.wide.s32 	%rd38, %r129, 4, %rd60;
	mad.wide.s32 	%rd39, %r130, 4, %rd60;
	mad.wide.s32 	%rd40, %r131, 4, %rd60;
	mad.wide.s32 	%rd41, %r132, 4, %rd60;
	.loc	1 145 19                        // sk07_lm_head.py:145:19
	// begin inline asm
	mov.u32 %r21, 0x0;
	ld.global.b32 { %r21 }, [ %rd30 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r22, 0x0;
	ld.global.b32 { %r22 }, [ %rd31 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r23, 0x0;
	mov.u32 %r24, 0x0;
	ld.global.v2.b32 { %r23, %r24 }, [ %rd32 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r25, 0x0;
	mov.u32 %r26, 0x0;
	ld.global.v2.b32 { %r25, %r26 }, [ %rd33 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r27, 0x0;
	mov.u32 %r28, 0x0;
	ld.global.v2.b32 { %r27, %r28 }, [ %rd34 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r29, 0x0;
	mov.u32 %r30, 0x0;
	ld.global.v2.b32 { %r29, %r30 }, [ %rd35 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r31, 0x0;
	mov.u32 %r32, 0x0;
	ld.global.v2.b32 { %r31, %r32 }, [ %rd36 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r33, 0x0;
	mov.u32 %r34, 0x0;
	ld.global.v2.b32 { %r33, %r34 }, [ %rd37 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r35, 0x0;
	mov.u32 %r36, 0x0;
	ld.global.v2.b32 { %r35, %r36 }, [ %rd38 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r37, 0x0;
	mov.u32 %r38, 0x0;
	ld.global.v2.b32 { %r37, %r38 }, [ %rd39 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r39, 0x0;
	mov.u32 %r40, 0x0;
	mov.u32 %r41, 0x0;
	mov.u32 %r42, 0x0;
	ld.global.v4.b32 { %r39, %r40, %r41, %r42 }, [ %rd40 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r43, 0x0;
	mov.u32 %r44, 0x0;
	mov.u32 %r45, 0x0;
	mov.u32 %r46, 0x0;
	ld.global.v4.b32 { %r43, %r44, %r45, %r46 }, [ %rd41 + 0 ];
	// end inline asm
	.loc	1 146 39                        // sk07_lm_head.py:146:39
	mul.lo.s32 	%r133, %r101, %r85;
	mul.lo.s32 	%r134, %r102, %r85;
	mul.lo.s32 	%r135, %r103, %r85;
	mul.lo.s32 	%r136, %r104, %r85;
	.loc	1 146 21                        // sk07_lm_head.py:146:21
	cvt.s64.s32 	%rd1, %r133;
	add.s64 	%rd62, %rd24, %rd1;
	cvt.s64.s32 	%rd2, %r134;
	add.s64 	%rd63, %rd24, %rd2;
	cvt.s64.s32 	%rd3, %r135;
	add.s64 	%rd64, %rd24, %rd3;
	cvt.s64.s32 	%rd4, %r136;
	add.s64 	%rd65, %rd24, %rd4;
	.loc	1 146 58                        // sk07_lm_head.py:146:58
	shl.b32 	%r137, %r4, 4;
	.loc	1 146 51                        // sk07_lm_head.py:146:51
	cvt.u64.u32 	%rd5, %r137;
	add.s64 	%rd42, %rd62, %rd5;
	add.s64 	%rd43, %rd63, %rd5;
	add.s64 	%rd44, %rd64, %rd5;
	add.s64 	%rd45, %rd65, %rd5;
	.loc	1 147 21                        // sk07_lm_head.py:147:21
	add.s64 	%rd66, %rd25, %rd5;
	.loc	1 147 67                        // sk07_lm_head.py:147:67
	mul.lo.s32 	%r138, %r21, %r87;
	mul.lo.s32 	%r139, %r22, %r87;
	.loc	1 147 51                        // sk07_lm_head.py:147:51
	cvt.s64.s32 	%rd6, %r138;
	add.s64 	%rd46, %rd66, %rd6;
	cvt.s64.s32 	%rd7, %r139;
	add.s64 	%rd47, %rd66, %rd7;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	setp.gt.s32 	%p1, %r83, 63;
	.loc	1 160 30                        // sk07_lm_head.py:160:30
	shl.b32 	%r207, %r3, 4;
	shl.b32 	%r9, %r2, 1;
	and.b32 	%r10, %r9, 48;
	xor.b32 	%r208, %r207, %r10;
	mov.b32 	%r209, global_smem;
	add.s32 	%r47, %r209, %r208;
	selp.b32 	%r48, 16, 0, %p1;
	// begin inline asm
	cp.async.cg.shared.global [ %r47 + 0 ], [ %rd42 + 0 ], 0x10, %r48;
	// end inline asm
	add.s32 	%r49, %r47, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r49 + 0 ], [ %rd43 + 0 ], 0x10, %r48;
	// end inline asm
	add.s32 	%r50, %r47, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r50 + 0 ], [ %rd44 + 0 ], 0x10, %r48;
	// end inline asm
	add.s32 	%r51, %r47, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r51 + 0 ], [ %rd45 + 0 ], 0x10, %r48;
	// end inline asm
	cp.async.commit_group;
	.loc	1 160 47                        // sk07_lm_head.py:160:47
	add.s32 	%r52, %r47, 49152;
	// begin inline asm
	cp.async.cg.shared.global [ %r52 + 0 ], [ %rd46 + 0 ], 0x10, %r48;
	// end inline asm
	add.s32 	%r53, %r47, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r53 + 0 ], [ %rd47 + 0 ], 0x10, %r48;
	// end inline asm
	cp.async.commit_group;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	setp.gt.s32 	%p2, %r83, 127;
	.loc	1 161 18                        // sk07_lm_head.py:161:18
	add.s64 	%rd48, %rd42, 64;
	add.s64 	%rd49, %rd43, 64;
	add.s64 	%rd50, %rd44, 64;
	add.s64 	%rd51, %rd45, 64;
	.loc	1 162 18                        // sk07_lm_head.py:162:18
	add.s64 	%rd52, %rd46, 64;
	add.s64 	%rd53, %rd47, 64;
	.loc	1 160 30                        // sk07_lm_head.py:160:30
	bar.sync 	0;
	add.s32 	%r54, %r47, 16384;
	selp.b32 	%r55, 16, 0, %p2;
	// begin inline asm
	cp.async.cg.shared.global [ %r54 + 0 ], [ %rd48 + 0 ], 0x10, %r55;
	// end inline asm
	add.s32 	%r56, %r47, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r56 + 0 ], [ %rd49 + 0 ], 0x10, %r55;
	// end inline asm
	add.s32 	%r57, %r47, 24576;
	// begin inline asm
	cp.async.cg.shared.global [ %r57 + 0 ], [ %rd50 + 0 ], 0x10, %r55;
	// end inline asm
	add.s32 	%r58, %r47, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r58 + 0 ], [ %rd51 + 0 ], 0x10, %r55;
	// end inline asm
	cp.async.commit_group;
	.loc	1 160 47                        // sk07_lm_head.py:160:47
	add.s32 	%r59, %r47, 57344;
	// begin inline asm
	cp.async.cg.shared.global [ %r59 + 0 ], [ %rd52 + 0 ], 0x10, %r55;
	// end inline asm
	add.s32 	%r60, %r47, 61440;
	// begin inline asm
	cp.async.cg.shared.global [ %r60 + 0 ], [ %rd53 + 0 ], 0x10, %r55;
	// end inline asm
	cp.async.commit_group;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	setp.gt.s32 	%p3, %r83, 191;
	.loc	1 161 18                        // sk07_lm_head.py:161:18
	add.s64 	%rd54, %rd42, 128;
	add.s64 	%rd55, %rd43, 128;
	add.s64 	%rd56, %rd44, 128;
	add.s64 	%rd57, %rd45, 128;
	.loc	1 162 18                        // sk07_lm_head.py:162:18
	add.s64 	%rd58, %rd46, 128;
	add.s64 	%rd59, %rd47, 128;
	.loc	1 160 30                        // sk07_lm_head.py:160:30
	bar.sync 	0;
	add.s32 	%r61, %r47, 32768;
	selp.b32 	%r62, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r61 + 0 ], [ %rd54 + 0 ], 0x10, %r62;
	// end inline asm
	add.s32 	%r63, %r47, 36864;
	// begin inline asm
	cp.async.cg.shared.global [ %r63 + 0 ], [ %rd55 + 0 ], 0x10, %r62;
	// end inline asm
	add.s32 	%r64, %r47, 40960;
	// begin inline asm
	cp.async.cg.shared.global [ %r64 + 0 ], [ %rd56 + 0 ], 0x10, %r62;
	// end inline asm
	add.s32 	%r65, %r47, 45056;
	// begin inline asm
	cp.async.cg.shared.global [ %r65 + 0 ], [ %rd57 + 0 ], 0x10, %r62;
	// end inline asm
	cp.async.commit_group;
	.loc	1 160 47                        // sk07_lm_head.py:160:47
	add.s32 	%r66, %r47, 65536;
	// begin inline asm
	cp.async.cg.shared.global [ %r66 + 0 ], [ %rd58 + 0 ], 0x10, %r62;
	// end inline asm
	add.s32 	%r67, %r47, 69632;
	// begin inline asm
	cp.async.cg.shared.global [ %r67 + 0 ], [ %rd59 + 0 ], 0x10, %r62;
	// end inline asm
	cp.async.commit_group;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	@%p1 bra 	$L__BB0_2;
	bra.uni 	$L__BB0_1;
$L__BB0_2:                              // %.lr.ph
	.loc	1 0 23                          // sk07_lm_head.py:0:23
	ld.param.b32 	%r20, [_sk07_lm_head_kernel_param_16];
	ld.param.b64 	%rd61, [_sk07_lm_head_kernel_param_7];
	shr.s32 	%r140, %r23, 31;
	shr.u32 	%r141, %r140, 25;
	add.s32 	%r142, %r23, %r141;
	shr.s32 	%r143, %r142, 7;
	shr.s32 	%r144, %r24, 31;
	shr.u32 	%r145, %r144, 25;
	add.s32 	%r146, %r24, %r145;
	shr.s32 	%r147, %r146, 7;
	shr.s32 	%r148, %r25, 31;
	shr.u32 	%r149, %r148, 25;
	add.s32 	%r150, %r25, %r149;
	shr.s32 	%r151, %r150, 7;
	shr.s32 	%r152, %r26, 31;
	shr.u32 	%r153, %r152, 25;
	add.s32 	%r154, %r26, %r153;
	shr.s32 	%r155, %r154, 7;
	shr.s32 	%r156, %r27, 31;
	shr.u32 	%r157, %r156, 25;
	add.s32 	%r158, %r27, %r157;
	shr.s32 	%r159, %r158, 7;
	shr.s32 	%r160, %r28, 31;
	shr.u32 	%r161, %r160, 25;
	add.s32 	%r162, %r28, %r161;
	shr.s32 	%r163, %r162, 7;
	shr.s32 	%r164, %r29, 31;
	shr.u32 	%r165, %r164, 25;
	add.s32 	%r166, %r29, %r165;
	shr.s32 	%r167, %r166, 7;
	shr.s32 	%r168, %r30, 31;
	shr.u32 	%r169, %r168, 25;
	add.s32 	%r170, %r30, %r169;
	shr.s32 	%r171, %r170, 7;
	shr.s32 	%r172, %r31, 31;
	shr.u32 	%r173, %r172, 25;
	add.s32 	%r174, %r31, %r173;
	shr.s32 	%r175, %r174, 7;
	shr.s32 	%r176, %r32, 31;
	shr.u32 	%r177, %r176, 25;
	add.s32 	%r178, %r32, %r177;
	shr.s32 	%r179, %r178, 7;
	shr.s32 	%r180, %r33, 31;
	shr.u32 	%r181, %r180, 25;
	add.s32 	%r182, %r33, %r181;
	shr.s32 	%r183, %r182, 7;
	shr.s32 	%r184, %r34, 31;
	shr.u32 	%r185, %r184, 25;
	add.s32 	%r186, %r34, %r185;
	shr.s32 	%r187, %r186, 7;
	shr.s32 	%r188, %r35, 31;
	shr.u32 	%r189, %r188, 25;
	add.s32 	%r190, %r35, %r189;
	shr.s32 	%r191, %r190, 7;
	shr.s32 	%r192, %r36, 31;
	shr.u32 	%r193, %r192, 25;
	add.s32 	%r194, %r36, %r193;
	shr.s32 	%r195, %r194, 7;
	shr.s32 	%r196, %r37, 31;
	shr.u32 	%r197, %r196, 25;
	add.s32 	%r198, %r37, %r197;
	shr.s32 	%r199, %r198, 7;
	shr.s32 	%r200, %r38, 31;
	shr.u32 	%r201, %r200, 25;
	add.s32 	%r202, %r38, %r201;
	shr.s32 	%r203, %r202, 7;
	cvt.s64.s32 	%rd67, %r143;
	add.s64 	%rd8, %rd61, %rd67;
	cvt.s64.s32 	%rd68, %r147;
	add.s64 	%rd9, %rd61, %rd68;
	cvt.s64.s32 	%rd69, %r151;
	add.s64 	%rd10, %rd61, %rd69;
	cvt.s64.s32 	%rd70, %r155;
	add.s64 	%rd11, %rd61, %rd70;
	cvt.s64.s32 	%rd71, %r159;
	add.s64 	%rd12, %rd61, %rd71;
	cvt.s64.s32 	%rd72, %r163;
	add.s64 	%rd13, %rd61, %rd72;
	cvt.s64.s32 	%rd73, %r167;
	add.s64 	%rd14, %rd61, %rd73;
	cvt.s64.s32 	%rd74, %r171;
	add.s64 	%rd15, %rd61, %rd74;
	cvt.s64.s32 	%rd75, %r175;
	add.s64 	%rd16, %rd61, %rd75;
	cvt.s64.s32 	%rd76, %r179;
	add.s64 	%rd17, %rd61, %rd76;
	cvt.s64.s32 	%rd77, %r183;
	add.s64 	%rd18, %rd61, %rd77;
	cvt.s64.s32 	%rd78, %r187;
	add.s64 	%rd19, %rd61, %rd78;
	cvt.s64.s32 	%rd79, %r191;
	add.s64 	%rd20, %rd61, %rd79;
	cvt.s64.s32 	%rd80, %r195;
	add.s64 	%rd21, %rd61, %rd80;
	cvt.s64.s32 	%rd81, %r199;
	add.s64 	%rd22, %rd61, %rd81;
	cvt.s64.s32 	%rd82, %r203;
	add.s64 	%rd23, %rd61, %rd82;
	shr.s32 	%r204, %r83, 31;
	shr.u32 	%r205, %r204, 26;
	add.s32 	%r206, %r83, %r205;
	shr.s32 	%r8, %r206, 6;
	add.s32 	%r11, %r8, -3;
	shl.b32 	%r210, %r6, 6;
	shl.b32 	%r1414, %r2, 4;
	and.b32 	%r211, %r1414, 3072;
	shl.b32 	%r212, %r2, 3;
	and.b32 	%r213, %r212, 48;
	and.b32 	%r1415, %r2, 16;
	or.b32 	%r214, %r210, %r211;
	xor.b32 	%r215, %r213, %r1415;
	or.b32 	%r12, %r214, %r215;
	xor.b32 	%r13, %r12, 32;
	shl.b32 	%r216, %r2, 6;
	and.b32 	%r217, %r216, 448;
	shl.b32 	%r218, %r5, 4;
	or.b32 	%r219, %r217, %r213;
	xor.b32 	%r220, %r219, %r10;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	add.s32 	%r221, %r209, %r218;
	add.s32 	%r14, %r221, %r220;
	add.s64 	%rd83, %rd7, %rd25;
	add.s64 	%rd319, %rd83, 192;
	add.s64 	%rd84, %rd6, %rd25;
	add.s64 	%rd318, %rd84, 192;
	add.s64 	%rd85, %rd4, %rd24;
	add.s64 	%rd317, %rd85, 192;
	add.s64 	%rd86, %rd3, %rd24;
	add.s64 	%rd316, %rd86, 192;
	add.s64 	%rd87, %rd2, %rd24;
	add.s64 	%rd315, %rd87, 192;
	add.s64 	%rd88, %rd1, %rd24;
	add.s64 	%rd314, %rd88, 192;
	mov.b32 	%r1416, 0f00000000;
	mov.b32 	%r222, 0;
	mov.b32 	%r1412, 2;
	mov.b32 	%r1411, -1;
	mov.b32 	%r1413, %r222;
	mov.b32 	%r1417, %r1416;
	mov.b32 	%r1418, %r1416;
	mov.b32 	%r1419, %r1416;
	mov.b32 	%r1420, %r1416;
	mov.b32 	%r1421, %r1416;
	mov.b32 	%r1422, %r1416;
	mov.b32 	%r1423, %r1416;
	mov.b32 	%r1424, %r1416;
	mov.b32 	%r1425, %r1416;
	mov.b32 	%r1426, %r1416;
	mov.b32 	%r1427, %r1416;
	mov.b32 	%r1428, %r1416;
	mov.b32 	%r1429, %r1416;
	mov.b32 	%r1430, %r1416;
	mov.b32 	%r1431, %r1416;
	mov.b32 	%r1432, %r1416;
	mov.b32 	%r1433, %r1416;
	mov.b32 	%r1434, %r1416;
	mov.b32 	%r1435, %r1416;
	mov.b32 	%r1436, %r1416;
	mov.b32 	%r1437, %r1416;
	mov.b32 	%r1438, %r1416;
	mov.b32 	%r1439, %r1416;
	mov.b32 	%r1440, %r1416;
	mov.b32 	%r1441, %r1416;
	mov.b32 	%r1442, %r1416;
	mov.b32 	%r1443, %r1416;
	mov.b32 	%r1444, %r1416;
	mov.b32 	%r1445, %r1416;
	mov.b32 	%r1446, %r1416;
	mov.b32 	%r1447, %r1416;
	mov.b32 	%r1448, %r1416;
	mov.b32 	%r1449, %r1416;
	mov.b32 	%r1450, %r1416;
	mov.b32 	%r1451, %r1416;
	mov.b32 	%r1452, %r1416;
	mov.b32 	%r1453, %r1416;
	mov.b32 	%r1454, %r1416;
	mov.b32 	%r1455, %r1416;
	mov.b32 	%r1456, %r1416;
	mov.b32 	%r1457, %r1416;
	mov.b32 	%r1458, %r1416;
	mov.b32 	%r1459, %r1416;
	mov.b32 	%r1460, %r1416;
	mov.b32 	%r1461, %r1416;
	mov.b32 	%r1462, %r1416;
	mov.b32 	%r1463, %r1416;
	mov.b32 	%r1464, %r1416;
	mov.b32 	%r1465, %r1416;
	mov.b32 	%r1466, %r1416;
	mov.b32 	%r1467, %r1416;
	mov.b32 	%r1468, %r1416;
	mov.b32 	%r1469, %r1416;
	mov.b32 	%r1470, %r1416;
	mov.b32 	%r1471, %r1416;
	mov.b32 	%r1472, %r1416;
	mov.b32 	%r1473, %r1416;
	mov.b32 	%r1474, %r1416;
	mov.b32 	%r1475, %r1416;
	mov.b32 	%r1476, %r1416;
	mov.b32 	%r1477, %r1416;
	mov.b32 	%r1478, %r1416;
	mov.b32 	%r1479, %r1416;
	mov.b32 	%r1480, %r1416;
	mov.b32 	%r1481, %r1416;
	mov.b32 	%r1482, %r1416;
	mov.b32 	%r1483, %r1416;
	mov.b32 	%r1484, %r1416;
	mov.b32 	%r1485, %r1416;
	mov.b32 	%r1486, %r1416;
	mov.b32 	%r1487, %r1416;
	mov.b32 	%r1488, %r1416;
	mov.b32 	%r1489, %r1416;
	mov.b32 	%r1490, %r1416;
	mov.b32 	%r1491, %r1416;
	mov.b32 	%r1492, %r1416;
	mov.b32 	%r1493, %r1416;
	mov.b32 	%r1494, %r1416;
	mov.b32 	%r1495, %r1416;
	mov.b32 	%r1496, %r1416;
	mov.b32 	%r1497, %r1416;
	mov.b32 	%r1498, %r1416;
	mov.b32 	%r1499, %r1416;
	mov.b32 	%r1500, %r1416;
	mov.b32 	%r1501, %r1416;
	mov.b32 	%r1502, %r1416;
	mov.b32 	%r1503, %r1416;
	mov.b32 	%r1504, %r1416;
	mov.b32 	%r1505, %r1416;
	mov.b32 	%r1506, %r1416;
	mov.b32 	%r1507, %r1416;
	mov.b32 	%r1508, %r1416;
	mov.b32 	%r1509, %r1416;
	mov.b32 	%r1510, %r1416;
	mov.b32 	%r1511, %r1416;
	mov.b32 	%r1512, %r1416;
	mov.b32 	%r1513, %r1416;
	mov.b32 	%r1514, %r1416;
	mov.b32 	%r1515, %r1416;
	mov.b32 	%r1516, %r1416;
	mov.b32 	%r1517, %r1416;
	mov.b32 	%r1518, %r1416;
	mov.b32 	%r1519, %r1416;
	mov.b32 	%r1520, %r1416;
	mov.b32 	%r1521, %r1416;
	mov.b32 	%r1522, %r1416;
	mov.b32 	%r1523, %r1416;
	mov.b32 	%r1524, %r1416;
	mov.b32 	%r1525, %r1416;
	mov.b32 	%r1526, %r1416;
	mov.b32 	%r1527, %r1416;
	mov.b32 	%r1528, %r1416;
	mov.b32 	%r1529, %r1416;
	mov.b32 	%r1530, %r1416;
	mov.b32 	%r1531, %r1416;
	mov.b32 	%r1532, %r1416;
	mov.b32 	%r1533, %r1416;
	mov.b32 	%r1534, %r1416;
	mov.b32 	%r1535, %r1416;
	mov.b32 	%r1536, %r1416;
	mov.b32 	%r1537, %r1416;
	mov.b32 	%r1538, %r1416;
	mov.b32 	%r1539, %r1416;
	mov.b32 	%r1540, %r1416;
	mov.b32 	%r1541, %r1416;
	mov.b32 	%r1542, %r1416;
	mov.b32 	%r1543, %r1416;
$L__BB0_3:                              // %__nv_exp2f.exit
                                        // =>This Inner Loop Header: Depth=1
	setp.lt.s32 	%p4, %r1413, %r11;
	add.s32 	%r422, %r1411, 1;
	setp.gt.s32 	%p5, %r422, 2;
	selp.b32 	%r1411, 0, %r422, %p5;
	.loc	1 158 56                        // sk07_lm_head.py:158:56
	bfe.u32 	%r423, %r1413, 1, 25;
	.loc	1 159 31                        // sk07_lm_head.py:159:31
	mul.lo.s32 	%r424, %r423, %r20;
	.loc	1 158 39                        // sk07_lm_head.py:158:39
	cvt.s64.s32 	%rd111, %r424;
	add.s64 	%rd89, %rd8, %rd111;
	add.s64 	%rd90, %rd9, %rd111;
	add.s64 	%rd91, %rd10, %rd111;
	add.s64 	%rd92, %rd11, %rd111;
	add.s64 	%rd93, %rd12, %rd111;
	add.s64 	%rd94, %rd13, %rd111;
	add.s64 	%rd95, %rd14, %rd111;
	add.s64 	%rd96, %rd15, %rd111;
	add.s64 	%rd97, %rd16, %rd111;
	add.s64 	%rd98, %rd17, %rd111;
	add.s64 	%rd99, %rd18, %rd111;
	add.s64 	%rd100, %rd19, %rd111;
	add.s64 	%rd101, %rd20, %rd111;
	add.s64 	%rd102, %rd21, %rd111;
	add.s64 	%rd103, %rd22, %rd111;
	add.s64 	%rd104, %rd23, %rd111;
	.loc	1 158 29                        // sk07_lm_head.py:158:29
	// begin inline asm
	mov.u16 %rs1, 0x0;
	ld.global.b8 { %rs1 }, [ %rd89 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs17, %rs1;
	// begin inline asm
	mov.u16 %rs2, 0x0;
	ld.global.b8 { %rs2 }, [ %rd90 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs18, %rs2;
	// begin inline asm
	mov.u16 %rs3, 0x0;
	ld.global.b8 { %rs3 }, [ %rd91 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs19, %rs3;
	// begin inline asm
	mov.u16 %rs4, 0x0;
	ld.global.b8 { %rs4 }, [ %rd92 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs20, %rs4;
	// begin inline asm
	mov.u16 %rs5, 0x0;
	ld.global.b8 { %rs5 }, [ %rd93 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs21, %rs5;
	// begin inline asm
	mov.u16 %rs6, 0x0;
	ld.global.b8 { %rs6 }, [ %rd94 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs22, %rs6;
	// begin inline asm
	mov.u16 %rs7, 0x0;
	ld.global.b8 { %rs7 }, [ %rd95 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs23, %rs7;
	// begin inline asm
	mov.u16 %rs8, 0x0;
	ld.global.b8 { %rs8 }, [ %rd96 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs24, %rs8;
	// begin inline asm
	mov.u16 %rs9, 0x0;
	ld.global.b8 { %rs9 }, [ %rd97 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs25, %rs9;
	// begin inline asm
	mov.u16 %rs10, 0x0;
	ld.global.b8 { %rs10 }, [ %rd98 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs26, %rs10;
	// begin inline asm
	mov.u16 %rs11, 0x0;
	ld.global.b8 { %rs11 }, [ %rd99 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs27, %rs11;
	// begin inline asm
	mov.u16 %rs12, 0x0;
	ld.global.b8 { %rs12 }, [ %rd100 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs28, %rs12;
	// begin inline asm
	mov.u16 %rs13, 0x0;
	ld.global.b8 { %rs13 }, [ %rd101 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs29, %rs13;
	// begin inline asm
	mov.u16 %rs14, 0x0;
	ld.global.b8 { %rs14 }, [ %rd102 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs30, %rs14;
	// begin inline asm
	mov.u16 %rs15, 0x0;
	ld.global.b8 { %rs15 }, [ %rd103 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs31, %rs15;
	// begin inline asm
	mov.u16 %rs16, 0x0;
	ld.global.b8 { %rs16 }, [ %rd104 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs32, %rs16;
	.loc	1 159 50                        // sk07_lm_head.py:159:50
	cvt.rn.f32.s16 	%r425, %rs17;
	cvt.rn.f32.s16 	%r426, %rs18;
	cvt.rn.f32.s16 	%r427, %rs19;
	cvt.rn.f32.s16 	%r428, %rs20;
	cvt.rn.f32.s16 	%r429, %rs21;
	cvt.rn.f32.s16 	%r430, %rs22;
	cvt.rn.f32.s16 	%r431, %rs23;
	cvt.rn.f32.s16 	%r432, %rs24;
	cvt.rn.f32.s16 	%r433, %rs25;
	cvt.rn.f32.s16 	%r434, %rs26;
	cvt.rn.f32.s16 	%r435, %rs27;
	cvt.rn.f32.s16 	%r436, %rs28;
	cvt.rn.f32.s16 	%r437, %rs29;
	cvt.rn.f32.s16 	%r438, %rs30;
	cvt.rn.f32.s16 	%r439, %rs31;
	cvt.rn.f32.s16 	%r440, %rs32;
	.loc	1 158 21                        // sk07_lm_head.py:158:21
	ex2.approx.ftz.f32 	%r441, %r425;
	ex2.approx.ftz.f32 	%r442, %r426;
	ex2.approx.ftz.f32 	%r443, %r427;
	ex2.approx.ftz.f32 	%r444, %r428;
	ex2.approx.ftz.f32 	%r445, %r429;
	ex2.approx.ftz.f32 	%r446, %r430;
	ex2.approx.ftz.f32 	%r447, %r431;
	ex2.approx.ftz.f32 	%r448, %r432;
	ex2.approx.ftz.f32 	%r449, %r433;
	ex2.approx.ftz.f32 	%r450, %r434;
	ex2.approx.ftz.f32 	%r451, %r435;
	ex2.approx.ftz.f32 	%r452, %r436;
	ex2.approx.ftz.f32 	%r453, %r437;
	ex2.approx.ftz.f32 	%r454, %r438;
	ex2.approx.ftz.f32 	%r455, %r439;
	ex2.approx.ftz.f32 	%r456, %r440;
	.loc	1 160 30                        // sk07_lm_head.py:160:30
	cp.async.wait_group 	4;
	bar.sync 	0;
	shl.b32 	%r457, %r1411, 14;
	add.s32 	%r458, %r209, %r457;
	add.s32 	%r459, %r458, %r12;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r223, %r224, %r225, %r226}, [%r459];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r243, %r244, %r245, %r246}, [%r459+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r247, %r248, %r249, %r250}, [%r459+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r251, %r252, %r253, %r254}, [%r459+12288];
	add.s32 	%r460, %r458, %r13;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r259, %r260, %r261, %r262}, [%r460];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r311, %r312, %r313, %r314}, [%r460+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r347, %r348, %r349, %r350}, [%r460+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r383, %r384, %r385, %r386}, [%r460+12288];
	.loc	1 160 47                        // sk07_lm_head.py:160:47
	shl.b32 	%r461, %r1411, 13;
	add.s32 	%r462, %r14, %r461;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r227, %r228, %r263, %r264}, [%r462+49152];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r229, %r230, %r269, %r270}, [%r462+50176];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r231, %r232, %r275, %r276}, [%r462+51200];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r233, %r234, %r281, %r282}, [%r462+52224];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r235, %r236, %r287, %r288}, [%r462+53248];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r237, %r238, %r293, %r294}, [%r462+54272];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r239, %r240, %r299, %r300}, [%r462+55296];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r241, %r242, %r305, %r306}, [%r462+56320];
	.loc	1 160 39                        // sk07_lm_head.py:160:39
	mov.b32 	%r255, %r222;
	mov.b32 	%r256, %r222;
	mov.b32 	%r257, %r222;
	mov.b32 	%r258, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r255, %r256, %r257, %r258 }, { %r223, %r224, %r225, %r226 }, { %r227, %r228 }, { %r255, %r256, %r257, %r258 };
	// end inline asm
	mov.b32 	%r265, %r222;
	mov.b32 	%r266, %r222;
	mov.b32 	%r267, %r222;
	mov.b32 	%r268, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r265, %r266, %r267, %r268 }, { %r223, %r224, %r225, %r226 }, { %r229, %r230 }, { %r265, %r266, %r267, %r268 };
	// end inline asm
	mov.b32 	%r271, %r222;
	mov.b32 	%r272, %r222;
	mov.b32 	%r273, %r222;
	mov.b32 	%r274, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r271, %r272, %r273, %r274 }, { %r223, %r224, %r225, %r226 }, { %r231, %r232 }, { %r271, %r272, %r273, %r274 };
	// end inline asm
	mov.b32 	%r277, %r222;
	mov.b32 	%r278, %r222;
	mov.b32 	%r279, %r222;
	mov.b32 	%r280, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r277, %r278, %r279, %r280 }, { %r223, %r224, %r225, %r226 }, { %r233, %r234 }, { %r277, %r278, %r279, %r280 };
	// end inline asm
	mov.b32 	%r283, %r222;
	mov.b32 	%r284, %r222;
	mov.b32 	%r285, %r222;
	mov.b32 	%r286, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r283, %r284, %r285, %r286 }, { %r223, %r224, %r225, %r226 }, { %r235, %r236 }, { %r283, %r284, %r285, %r286 };
	// end inline asm
	mov.b32 	%r289, %r222;
	mov.b32 	%r290, %r222;
	mov.b32 	%r291, %r222;
	mov.b32 	%r292, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r289, %r290, %r291, %r292 }, { %r223, %r224, %r225, %r226 }, { %r237, %r238 }, { %r289, %r290, %r291, %r292 };
	// end inline asm
	mov.b32 	%r295, %r222;
	mov.b32 	%r296, %r222;
	mov.b32 	%r297, %r222;
	mov.b32 	%r298, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r295, %r296, %r297, %r298 }, { %r223, %r224, %r225, %r226 }, { %r239, %r240 }, { %r295, %r296, %r297, %r298 };
	// end inline asm
	mov.b32 	%r301, %r222;
	mov.b32 	%r302, %r222;
	mov.b32 	%r303, %r222;
	mov.b32 	%r304, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r301, %r302, %r303, %r304 }, { %r223, %r224, %r225, %r226 }, { %r241, %r242 }, { %r301, %r302, %r303, %r304 };
	// end inline asm
	mov.b32 	%r307, %r222;
	mov.b32 	%r308, %r222;
	mov.b32 	%r309, %r222;
	mov.b32 	%r310, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r307, %r308, %r309, %r310 }, { %r243, %r244, %r245, %r246 }, { %r227, %r228 }, { %r307, %r308, %r309, %r310 };
	// end inline asm
	mov.b32 	%r315, %r222;
	mov.b32 	%r316, %r222;
	mov.b32 	%r317, %r222;
	mov.b32 	%r318, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r315, %r316, %r317, %r318 }, { %r243, %r244, %r245, %r246 }, { %r229, %r230 }, { %r315, %r316, %r317, %r318 };
	// end inline asm
	mov.b32 	%r319, %r222;
	mov.b32 	%r320, %r222;
	mov.b32 	%r321, %r222;
	mov.b32 	%r322, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r319, %r320, %r321, %r322 }, { %r243, %r244, %r245, %r246 }, { %r231, %r232 }, { %r319, %r320, %r321, %r322 };
	// end inline asm
	mov.b32 	%r323, %r222;
	mov.b32 	%r324, %r222;
	mov.b32 	%r325, %r222;
	mov.b32 	%r326, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r323, %r324, %r325, %r326 }, { %r243, %r244, %r245, %r246 }, { %r233, %r234 }, { %r323, %r324, %r325, %r326 };
	// end inline asm
	mov.b32 	%r327, %r222;
	mov.b32 	%r328, %r222;
	mov.b32 	%r329, %r222;
	mov.b32 	%r330, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r327, %r328, %r329, %r330 }, { %r243, %r244, %r245, %r246 }, { %r235, %r236 }, { %r327, %r328, %r329, %r330 };
	// end inline asm
	mov.b32 	%r331, %r222;
	mov.b32 	%r332, %r222;
	mov.b32 	%r333, %r222;
	mov.b32 	%r334, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r331, %r332, %r333, %r334 }, { %r243, %r244, %r245, %r246 }, { %r237, %r238 }, { %r331, %r332, %r333, %r334 };
	// end inline asm
	mov.b32 	%r335, %r222;
	mov.b32 	%r336, %r222;
	mov.b32 	%r337, %r222;
	mov.b32 	%r338, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r335, %r336, %r337, %r338 }, { %r243, %r244, %r245, %r246 }, { %r239, %r240 }, { %r335, %r336, %r337, %r338 };
	// end inline asm
	mov.b32 	%r339, %r222;
	mov.b32 	%r340, %r222;
	mov.b32 	%r341, %r222;
	mov.b32 	%r342, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r339, %r340, %r341, %r342 }, { %r243, %r244, %r245, %r246 }, { %r241, %r242 }, { %r339, %r340, %r341, %r342 };
	// end inline asm
	mov.b32 	%r343, %r222;
	mov.b32 	%r344, %r222;
	mov.b32 	%r345, %r222;
	mov.b32 	%r346, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r343, %r344, %r345, %r346 }, { %r247, %r248, %r249, %r250 }, { %r227, %r228 }, { %r343, %r344, %r345, %r346 };
	// end inline asm
	mov.b32 	%r351, %r222;
	mov.b32 	%r352, %r222;
	mov.b32 	%r353, %r222;
	mov.b32 	%r354, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r351, %r352, %r353, %r354 }, { %r247, %r248, %r249, %r250 }, { %r229, %r230 }, { %r351, %r352, %r353, %r354 };
	// end inline asm
	mov.b32 	%r355, %r222;
	mov.b32 	%r356, %r222;
	mov.b32 	%r357, %r222;
	mov.b32 	%r358, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r355, %r356, %r357, %r358 }, { %r247, %r248, %r249, %r250 }, { %r231, %r232 }, { %r355, %r356, %r357, %r358 };
	// end inline asm
	mov.b32 	%r359, %r222;
	mov.b32 	%r360, %r222;
	mov.b32 	%r361, %r222;
	mov.b32 	%r362, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r359, %r360, %r361, %r362 }, { %r247, %r248, %r249, %r250 }, { %r233, %r234 }, { %r359, %r360, %r361, %r362 };
	// end inline asm
	mov.b32 	%r363, %r222;
	mov.b32 	%r364, %r222;
	mov.b32 	%r365, %r222;
	mov.b32 	%r366, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r363, %r364, %r365, %r366 }, { %r247, %r248, %r249, %r250 }, { %r235, %r236 }, { %r363, %r364, %r365, %r366 };
	// end inline asm
	mov.b32 	%r367, %r222;
	mov.b32 	%r368, %r222;
	mov.b32 	%r369, %r222;
	mov.b32 	%r370, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r367, %r368, %r369, %r370 }, { %r247, %r248, %r249, %r250 }, { %r237, %r238 }, { %r367, %r368, %r369, %r370 };
	// end inline asm
	mov.b32 	%r371, %r222;
	mov.b32 	%r372, %r222;
	mov.b32 	%r373, %r222;
	mov.b32 	%r374, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r371, %r372, %r373, %r374 }, { %r247, %r248, %r249, %r250 }, { %r239, %r240 }, { %r371, %r372, %r373, %r374 };
	// end inline asm
	mov.b32 	%r375, %r222;
	mov.b32 	%r376, %r222;
	mov.b32 	%r377, %r222;
	mov.b32 	%r378, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r375, %r376, %r377, %r378 }, { %r247, %r248, %r249, %r250 }, { %r241, %r242 }, { %r375, %r376, %r377, %r378 };
	// end inline asm
	mov.b32 	%r379, %r222;
	mov.b32 	%r380, %r222;
	mov.b32 	%r381, %r222;
	mov.b32 	%r382, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r379, %r380, %r381, %r382 }, { %r251, %r252, %r253, %r254 }, { %r227, %r228 }, { %r379, %r380, %r381, %r382 };
	// end inline asm
	mov.b32 	%r387, %r222;
	mov.b32 	%r388, %r222;
	mov.b32 	%r389, %r222;
	mov.b32 	%r390, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r387, %r388, %r389, %r390 }, { %r251, %r252, %r253, %r254 }, { %r229, %r230 }, { %r387, %r388, %r389, %r390 };
	// end inline asm
	mov.b32 	%r391, %r222;
	mov.b32 	%r392, %r222;
	mov.b32 	%r393, %r222;
	mov.b32 	%r394, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r391, %r392, %r393, %r394 }, { %r251, %r252, %r253, %r254 }, { %r231, %r232 }, { %r391, %r392, %r393, %r394 };
	// end inline asm
	mov.b32 	%r395, %r222;
	mov.b32 	%r396, %r222;
	mov.b32 	%r397, %r222;
	mov.b32 	%r398, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r395, %r396, %r397, %r398 }, { %r251, %r252, %r253, %r254 }, { %r233, %r234 }, { %r395, %r396, %r397, %r398 };
	// end inline asm
	mov.b32 	%r399, %r222;
	mov.b32 	%r400, %r222;
	mov.b32 	%r401, %r222;
	mov.b32 	%r402, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r399, %r400, %r401, %r402 }, { %r251, %r252, %r253, %r254 }, { %r235, %r236 }, { %r399, %r400, %r401, %r402 };
	// end inline asm
	mov.b32 	%r403, %r222;
	mov.b32 	%r404, %r222;
	mov.b32 	%r405, %r222;
	mov.b32 	%r406, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r403, %r404, %r405, %r406 }, { %r251, %r252, %r253, %r254 }, { %r237, %r238 }, { %r403, %r404, %r405, %r406 };
	// end inline asm
	mov.b32 	%r407, %r222;
	mov.b32 	%r408, %r222;
	mov.b32 	%r409, %r222;
	mov.b32 	%r410, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r407, %r408, %r409, %r410 }, { %r251, %r252, %r253, %r254 }, { %r239, %r240 }, { %r407, %r408, %r409, %r410 };
	// end inline asm
	mov.b32 	%r411, %r222;
	mov.b32 	%r412, %r222;
	mov.b32 	%r413, %r222;
	mov.b32 	%r414, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r411, %r412, %r413, %r414 }, { %r251, %r252, %r253, %r254 }, { %r241, %r242 }, { %r411, %r412, %r413, %r414 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r255, %r256, %r257, %r258 }, { %r259, %r260, %r261, %r262 }, { %r263, %r264 }, { %r255, %r256, %r257, %r258 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r265, %r266, %r267, %r268 }, { %r259, %r260, %r261, %r262 }, { %r269, %r270 }, { %r265, %r266, %r267, %r268 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r271, %r272, %r273, %r274 }, { %r259, %r260, %r261, %r262 }, { %r275, %r276 }, { %r271, %r272, %r273, %r274 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r277, %r278, %r279, %r280 }, { %r259, %r260, %r261, %r262 }, { %r281, %r282 }, { %r277, %r278, %r279, %r280 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r283, %r284, %r285, %r286 }, { %r259, %r260, %r261, %r262 }, { %r287, %r288 }, { %r283, %r284, %r285, %r286 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r289, %r290, %r291, %r292 }, { %r259, %r260, %r261, %r262 }, { %r293, %r294 }, { %r289, %r290, %r291, %r292 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r295, %r296, %r297, %r298 }, { %r259, %r260, %r261, %r262 }, { %r299, %r300 }, { %r295, %r296, %r297, %r298 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r301, %r302, %r303, %r304 }, { %r259, %r260, %r261, %r262 }, { %r305, %r306 }, { %r301, %r302, %r303, %r304 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r307, %r308, %r309, %r310 }, { %r311, %r312, %r313, %r314 }, { %r263, %r264 }, { %r307, %r308, %r309, %r310 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r315, %r316, %r317, %r318 }, { %r311, %r312, %r313, %r314 }, { %r269, %r270 }, { %r315, %r316, %r317, %r318 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r319, %r320, %r321, %r322 }, { %r311, %r312, %r313, %r314 }, { %r275, %r276 }, { %r319, %r320, %r321, %r322 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r323, %r324, %r325, %r326 }, { %r311, %r312, %r313, %r314 }, { %r281, %r282 }, { %r323, %r324, %r325, %r326 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r327, %r328, %r329, %r330 }, { %r311, %r312, %r313, %r314 }, { %r287, %r288 }, { %r327, %r328, %r329, %r330 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r331, %r332, %r333, %r334 }, { %r311, %r312, %r313, %r314 }, { %r293, %r294 }, { %r331, %r332, %r333, %r334 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r335, %r336, %r337, %r338 }, { %r311, %r312, %r313, %r314 }, { %r299, %r300 }, { %r335, %r336, %r337, %r338 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r339, %r340, %r341, %r342 }, { %r311, %r312, %r313, %r314 }, { %r305, %r306 }, { %r339, %r340, %r341, %r342 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r343, %r344, %r345, %r346 }, { %r347, %r348, %r349, %r350 }, { %r263, %r264 }, { %r343, %r344, %r345, %r346 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r351, %r352, %r353, %r354 }, { %r347, %r348, %r349, %r350 }, { %r269, %r270 }, { %r351, %r352, %r353, %r354 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r355, %r356, %r357, %r358 }, { %r347, %r348, %r349, %r350 }, { %r275, %r276 }, { %r355, %r356, %r357, %r358 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r359, %r360, %r361, %r362 }, { %r347, %r348, %r349, %r350 }, { %r281, %r282 }, { %r359, %r360, %r361, %r362 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r363, %r364, %r365, %r366 }, { %r347, %r348, %r349, %r350 }, { %r287, %r288 }, { %r363, %r364, %r365, %r366 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r367, %r368, %r369, %r370 }, { %r347, %r348, %r349, %r350 }, { %r293, %r294 }, { %r367, %r368, %r369, %r370 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r371, %r372, %r373, %r374 }, { %r347, %r348, %r349, %r350 }, { %r299, %r300 }, { %r371, %r372, %r373, %r374 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r375, %r376, %r377, %r378 }, { %r347, %r348, %r349, %r350 }, { %r305, %r306 }, { %r375, %r376, %r377, %r378 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r379, %r380, %r381, %r382 }, { %r383, %r384, %r385, %r386 }, { %r263, %r264 }, { %r379, %r380, %r381, %r382 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r387, %r388, %r389, %r390 }, { %r383, %r384, %r385, %r386 }, { %r269, %r270 }, { %r387, %r388, %r389, %r390 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r391, %r392, %r393, %r394 }, { %r383, %r384, %r385, %r386 }, { %r275, %r276 }, { %r391, %r392, %r393, %r394 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r395, %r396, %r397, %r398 }, { %r383, %r384, %r385, %r386 }, { %r281, %r282 }, { %r395, %r396, %r397, %r398 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r399, %r400, %r401, %r402 }, { %r383, %r384, %r385, %r386 }, { %r287, %r288 }, { %r399, %r400, %r401, %r402 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r403, %r404, %r405, %r406 }, { %r383, %r384, %r385, %r386 }, { %r293, %r294 }, { %r403, %r404, %r405, %r406 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r407, %r408, %r409, %r410 }, { %r383, %r384, %r385, %r386 }, { %r299, %r300 }, { %r407, %r408, %r409, %r410 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r411, %r412, %r413, %r414 }, { %r383, %r384, %r385, %r386 }, { %r305, %r306 }, { %r411, %r412, %r413, %r414 };
	// end inline asm
	.loc	1 160 79                        // sk07_lm_head.py:160:79
	cvt.rn.f32.s32 	%r463, %r414;
	cvt.rn.f32.s32 	%r464, %r413;
	cvt.rn.f32.s32 	%r465, %r412;
	cvt.rn.f32.s32 	%r466, %r411;
	cvt.rn.f32.s32 	%r467, %r410;
	cvt.rn.f32.s32 	%r468, %r409;
	cvt.rn.f32.s32 	%r469, %r408;
	cvt.rn.f32.s32 	%r470, %r407;
	cvt.rn.f32.s32 	%r471, %r406;
	cvt.rn.f32.s32 	%r472, %r405;
	cvt.rn.f32.s32 	%r473, %r404;
	cvt.rn.f32.s32 	%r474, %r403;
	cvt.rn.f32.s32 	%r475, %r402;
	cvt.rn.f32.s32 	%r476, %r401;
	cvt.rn.f32.s32 	%r477, %r400;
	cvt.rn.f32.s32 	%r478, %r399;
	cvt.rn.f32.s32 	%r479, %r398;
	cvt.rn.f32.s32 	%r480, %r397;
	cvt.rn.f32.s32 	%r481, %r396;
	cvt.rn.f32.s32 	%r482, %r395;
	cvt.rn.f32.s32 	%r483, %r394;
	cvt.rn.f32.s32 	%r484, %r393;
	cvt.rn.f32.s32 	%r485, %r392;
	cvt.rn.f32.s32 	%r486, %r391;
	cvt.rn.f32.s32 	%r487, %r390;
	cvt.rn.f32.s32 	%r488, %r389;
	cvt.rn.f32.s32 	%r489, %r388;
	cvt.rn.f32.s32 	%r490, %r387;
	cvt.rn.f32.s32 	%r491, %r382;
	cvt.rn.f32.s32 	%r492, %r381;
	cvt.rn.f32.s32 	%r493, %r380;
	cvt.rn.f32.s32 	%r494, %r379;
	cvt.rn.f32.s32 	%r495, %r375;
	cvt.rn.f32.s32 	%r496, %r376;
	cvt.rn.f32.s32 	%r497, %r377;
	cvt.rn.f32.s32 	%r498, %r378;
	cvt.rn.f32.s32 	%r499, %r371;
	cvt.rn.f32.s32 	%r500, %r372;
	cvt.rn.f32.s32 	%r501, %r373;
	cvt.rn.f32.s32 	%r502, %r374;
	cvt.rn.f32.s32 	%r503, %r367;
	cvt.rn.f32.s32 	%r504, %r368;
	cvt.rn.f32.s32 	%r505, %r369;
	cvt.rn.f32.s32 	%r506, %r370;
	cvt.rn.f32.s32 	%r507, %r363;
	cvt.rn.f32.s32 	%r508, %r364;
	cvt.rn.f32.s32 	%r509, %r365;
	cvt.rn.f32.s32 	%r510, %r366;
	cvt.rn.f32.s32 	%r511, %r359;
	cvt.rn.f32.s32 	%r512, %r360;
	cvt.rn.f32.s32 	%r513, %r361;
	cvt.rn.f32.s32 	%r514, %r362;
	cvt.rn.f32.s32 	%r515, %r355;
	cvt.rn.f32.s32 	%r516, %r356;
	cvt.rn.f32.s32 	%r517, %r357;
	cvt.rn.f32.s32 	%r518, %r358;
	cvt.rn.f32.s32 	%r519, %r351;
	cvt.rn.f32.s32 	%r520, %r352;
	cvt.rn.f32.s32 	%r521, %r353;
	cvt.rn.f32.s32 	%r522, %r354;
	cvt.rn.f32.s32 	%r523, %r343;
	cvt.rn.f32.s32 	%r524, %r344;
	cvt.rn.f32.s32 	%r525, %r345;
	cvt.rn.f32.s32 	%r526, %r346;
	cvt.rn.f32.s32 	%r527, %r339;
	cvt.rn.f32.s32 	%r528, %r340;
	cvt.rn.f32.s32 	%r529, %r341;
	cvt.rn.f32.s32 	%r530, %r342;
	cvt.rn.f32.s32 	%r531, %r335;
	cvt.rn.f32.s32 	%r532, %r336;
	cvt.rn.f32.s32 	%r533, %r337;
	cvt.rn.f32.s32 	%r534, %r338;
	cvt.rn.f32.s32 	%r535, %r331;
	cvt.rn.f32.s32 	%r536, %r332;
	cvt.rn.f32.s32 	%r537, %r333;
	cvt.rn.f32.s32 	%r538, %r334;
	cvt.rn.f32.s32 	%r539, %r327;
	cvt.rn.f32.s32 	%r540, %r328;
	cvt.rn.f32.s32 	%r541, %r329;
	cvt.rn.f32.s32 	%r542, %r330;
	cvt.rn.f32.s32 	%r543, %r323;
	cvt.rn.f32.s32 	%r544, %r324;
	cvt.rn.f32.s32 	%r545, %r325;
	cvt.rn.f32.s32 	%r546, %r326;
	cvt.rn.f32.s32 	%r547, %r319;
	cvt.rn.f32.s32 	%r548, %r320;
	cvt.rn.f32.s32 	%r549, %r321;
	cvt.rn.f32.s32 	%r550, %r322;
	cvt.rn.f32.s32 	%r551, %r315;
	cvt.rn.f32.s32 	%r552, %r316;
	cvt.rn.f32.s32 	%r553, %r317;
	cvt.rn.f32.s32 	%r554, %r318;
	cvt.rn.f32.s32 	%r555, %r307;
	cvt.rn.f32.s32 	%r556, %r308;
	cvt.rn.f32.s32 	%r557, %r309;
	cvt.rn.f32.s32 	%r558, %r310;
	cvt.rn.f32.s32 	%r559, %r301;
	cvt.rn.f32.s32 	%r560, %r302;
	cvt.rn.f32.s32 	%r561, %r303;
	cvt.rn.f32.s32 	%r562, %r304;
	cvt.rn.f32.s32 	%r563, %r295;
	cvt.rn.f32.s32 	%r564, %r296;
	cvt.rn.f32.s32 	%r565, %r297;
	cvt.rn.f32.s32 	%r566, %r298;
	cvt.rn.f32.s32 	%r567, %r289;
	cvt.rn.f32.s32 	%r568, %r290;
	cvt.rn.f32.s32 	%r569, %r291;
	cvt.rn.f32.s32 	%r570, %r292;
	cvt.rn.f32.s32 	%r571, %r283;
	cvt.rn.f32.s32 	%r572, %r284;
	cvt.rn.f32.s32 	%r573, %r285;
	cvt.rn.f32.s32 	%r574, %r286;
	cvt.rn.f32.s32 	%r575, %r277;
	cvt.rn.f32.s32 	%r576, %r278;
	cvt.rn.f32.s32 	%r577, %r279;
	cvt.rn.f32.s32 	%r578, %r280;
	cvt.rn.f32.s32 	%r579, %r271;
	cvt.rn.f32.s32 	%r580, %r272;
	cvt.rn.f32.s32 	%r581, %r273;
	cvt.rn.f32.s32 	%r582, %r274;
	cvt.rn.f32.s32 	%r583, %r265;
	cvt.rn.f32.s32 	%r584, %r266;
	cvt.rn.f32.s32 	%r585, %r267;
	cvt.rn.f32.s32 	%r586, %r268;
	cvt.rn.f32.s32 	%r587, %r255;
	cvt.rn.f32.s32 	%r588, %r256;
	cvt.rn.f32.s32 	%r589, %r257;
	cvt.rn.f32.s32 	%r590, %r258;
	.loc	1 160 15                        // sk07_lm_head.py:160:15
	fma.rn.f32 	%r1419, %r442, %r590, %r1419;
	fma.rn.f32 	%r1418, %r441, %r589, %r1418;
	fma.rn.f32 	%r1417, %r442, %r588, %r1417;
	fma.rn.f32 	%r1416, %r441, %r587, %r1416;
	fma.rn.f32 	%r1423, %r444, %r586, %r1423;
	fma.rn.f32 	%r1422, %r443, %r585, %r1422;
	fma.rn.f32 	%r1421, %r444, %r584, %r1421;
	fma.rn.f32 	%r1420, %r443, %r583, %r1420;
	fma.rn.f32 	%r1427, %r446, %r582, %r1427;
	fma.rn.f32 	%r1426, %r445, %r581, %r1426;
	fma.rn.f32 	%r1425, %r446, %r580, %r1425;
	fma.rn.f32 	%r1424, %r445, %r579, %r1424;
	fma.rn.f32 	%r1431, %r448, %r578, %r1431;
	fma.rn.f32 	%r1430, %r447, %r577, %r1430;
	fma.rn.f32 	%r1429, %r448, %r576, %r1429;
	fma.rn.f32 	%r1428, %r447, %r575, %r1428;
	fma.rn.f32 	%r1435, %r450, %r574, %r1435;
	fma.rn.f32 	%r1434, %r449, %r573, %r1434;
	fma.rn.f32 	%r1433, %r450, %r572, %r1433;
	fma.rn.f32 	%r1432, %r449, %r571, %r1432;
	fma.rn.f32 	%r1439, %r452, %r570, %r1439;
	fma.rn.f32 	%r1438, %r451, %r569, %r1438;
	fma.rn.f32 	%r1437, %r452, %r568, %r1437;
	fma.rn.f32 	%r1436, %r451, %r567, %r1436;
	fma.rn.f32 	%r1443, %r454, %r566, %r1443;
	fma.rn.f32 	%r1442, %r453, %r565, %r1442;
	fma.rn.f32 	%r1441, %r454, %r564, %r1441;
	fma.rn.f32 	%r1440, %r453, %r563, %r1440;
	fma.rn.f32 	%r1447, %r456, %r562, %r1447;
	fma.rn.f32 	%r1446, %r455, %r561, %r1446;
	fma.rn.f32 	%r1445, %r456, %r560, %r1445;
	fma.rn.f32 	%r1444, %r455, %r559, %r1444;
	fma.rn.f32 	%r1451, %r442, %r558, %r1451;
	fma.rn.f32 	%r1450, %r441, %r557, %r1450;
	fma.rn.f32 	%r1449, %r442, %r556, %r1449;
	fma.rn.f32 	%r1448, %r441, %r555, %r1448;
	fma.rn.f32 	%r1455, %r444, %r554, %r1455;
	fma.rn.f32 	%r1454, %r443, %r553, %r1454;
	fma.rn.f32 	%r1453, %r444, %r552, %r1453;
	fma.rn.f32 	%r1452, %r443, %r551, %r1452;
	fma.rn.f32 	%r1459, %r446, %r550, %r1459;
	fma.rn.f32 	%r1458, %r445, %r549, %r1458;
	fma.rn.f32 	%r1457, %r446, %r548, %r1457;
	fma.rn.f32 	%r1456, %r445, %r547, %r1456;
	fma.rn.f32 	%r1463, %r448, %r546, %r1463;
	fma.rn.f32 	%r1462, %r447, %r545, %r1462;
	fma.rn.f32 	%r1461, %r448, %r544, %r1461;
	fma.rn.f32 	%r1460, %r447, %r543, %r1460;
	fma.rn.f32 	%r1467, %r450, %r542, %r1467;
	fma.rn.f32 	%r1466, %r449, %r541, %r1466;
	fma.rn.f32 	%r1465, %r450, %r540, %r1465;
	fma.rn.f32 	%r1464, %r449, %r539, %r1464;
	fma.rn.f32 	%r1471, %r452, %r538, %r1471;
	fma.rn.f32 	%r1470, %r451, %r537, %r1470;
	fma.rn.f32 	%r1469, %r452, %r536, %r1469;
	fma.rn.f32 	%r1468, %r451, %r535, %r1468;
	fma.rn.f32 	%r1475, %r454, %r534, %r1475;
	fma.rn.f32 	%r1474, %r453, %r533, %r1474;
	fma.rn.f32 	%r1473, %r454, %r532, %r1473;
	fma.rn.f32 	%r1472, %r453, %r531, %r1472;
	fma.rn.f32 	%r1479, %r456, %r530, %r1479;
	fma.rn.f32 	%r1478, %r455, %r529, %r1478;
	fma.rn.f32 	%r1477, %r456, %r528, %r1477;
	fma.rn.f32 	%r1476, %r455, %r527, %r1476;
	fma.rn.f32 	%r1483, %r442, %r526, %r1483;
	fma.rn.f32 	%r1482, %r441, %r525, %r1482;
	fma.rn.f32 	%r1481, %r442, %r524, %r1481;
	fma.rn.f32 	%r1480, %r441, %r523, %r1480;
	fma.rn.f32 	%r1487, %r444, %r522, %r1487;
	fma.rn.f32 	%r1486, %r443, %r521, %r1486;
	fma.rn.f32 	%r1485, %r444, %r520, %r1485;
	fma.rn.f32 	%r1484, %r443, %r519, %r1484;
	fma.rn.f32 	%r1491, %r446, %r518, %r1491;
	fma.rn.f32 	%r1490, %r445, %r517, %r1490;
	fma.rn.f32 	%r1489, %r446, %r516, %r1489;
	fma.rn.f32 	%r1488, %r445, %r515, %r1488;
	fma.rn.f32 	%r1495, %r448, %r514, %r1495;
	fma.rn.f32 	%r1494, %r447, %r513, %r1494;
	fma.rn.f32 	%r1493, %r448, %r512, %r1493;
	fma.rn.f32 	%r1492, %r447, %r511, %r1492;
	fma.rn.f32 	%r1499, %r450, %r510, %r1499;
	fma.rn.f32 	%r1498, %r449, %r509, %r1498;
	fma.rn.f32 	%r1497, %r450, %r508, %r1497;
	fma.rn.f32 	%r1496, %r449, %r507, %r1496;
	fma.rn.f32 	%r1503, %r452, %r506, %r1503;
	fma.rn.f32 	%r1502, %r451, %r505, %r1502;
	fma.rn.f32 	%r1501, %r452, %r504, %r1501;
	fma.rn.f32 	%r1500, %r451, %r503, %r1500;
	fma.rn.f32 	%r1507, %r454, %r502, %r1507;
	fma.rn.f32 	%r1506, %r453, %r501, %r1506;
	fma.rn.f32 	%r1505, %r454, %r500, %r1505;
	fma.rn.f32 	%r1504, %r453, %r499, %r1504;
	fma.rn.f32 	%r1511, %r456, %r498, %r1511;
	fma.rn.f32 	%r1510, %r455, %r497, %r1510;
	fma.rn.f32 	%r1509, %r456, %r496, %r1509;
	fma.rn.f32 	%r1508, %r455, %r495, %r1508;
	fma.rn.f32 	%r1512, %r441, %r494, %r1512;
	fma.rn.f32 	%r1513, %r442, %r493, %r1513;
	fma.rn.f32 	%r1514, %r441, %r492, %r1514;
	fma.rn.f32 	%r1515, %r442, %r491, %r1515;
	fma.rn.f32 	%r1516, %r443, %r490, %r1516;
	fma.rn.f32 	%r1517, %r444, %r489, %r1517;
	fma.rn.f32 	%r1518, %r443, %r488, %r1518;
	fma.rn.f32 	%r1519, %r444, %r487, %r1519;
	fma.rn.f32 	%r1520, %r445, %r486, %r1520;
	fma.rn.f32 	%r1521, %r446, %r485, %r1521;
	fma.rn.f32 	%r1522, %r445, %r484, %r1522;
	fma.rn.f32 	%r1523, %r446, %r483, %r1523;
	fma.rn.f32 	%r1524, %r447, %r482, %r1524;
	fma.rn.f32 	%r1525, %r448, %r481, %r1525;
	fma.rn.f32 	%r1526, %r447, %r480, %r1526;
	fma.rn.f32 	%r1527, %r448, %r479, %r1527;
	fma.rn.f32 	%r1528, %r449, %r478, %r1528;
	fma.rn.f32 	%r1529, %r450, %r477, %r1529;
	fma.rn.f32 	%r1530, %r449, %r476, %r1530;
	fma.rn.f32 	%r1531, %r450, %r475, %r1531;
	fma.rn.f32 	%r1532, %r451, %r474, %r1532;
	fma.rn.f32 	%r1533, %r452, %r473, %r1533;
	fma.rn.f32 	%r1534, %r451, %r472, %r1534;
	fma.rn.f32 	%r1535, %r452, %r471, %r1535;
	fma.rn.f32 	%r1536, %r453, %r470, %r1536;
	fma.rn.f32 	%r1537, %r454, %r469, %r1537;
	fma.rn.f32 	%r1538, %r453, %r468, %r1538;
	fma.rn.f32 	%r1539, %r454, %r467, %r1539;
	fma.rn.f32 	%r1540, %r455, %r466, %r1540;
	fma.rn.f32 	%r1541, %r456, %r465, %r1541;
	fma.rn.f32 	%r1542, %r455, %r464, %r1542;
	fma.rn.f32 	%r1543, %r456, %r463, %r1543;
	.loc	1 161 18                        // sk07_lm_head.py:161:18
	add.s64 	%rd105, %rd314, %rd5;
	add.s64 	%rd106, %rd315, %rd5;
	add.s64 	%rd107, %rd316, %rd5;
	.loc	1 162 18                        // sk07_lm_head.py:162:18
	add.s64 	%rd108, %rd317, %rd5;
	add.s64 	%rd109, %rd318, %rd5;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	add.s64 	%rd110, %rd319, %rd5;
	add.s32 	%r591, %r1412, 1;
	setp.gt.s32 	%p6, %r591, 2;
	selp.b32 	%r1412, 0, %r591, %p6;
	.loc	1 160 30                        // sk07_lm_head.py:160:30
	shl.b32 	%r592, %r1412, 14;
	bar.sync 	0;
	add.s32 	%r415, %r47, %r592;
	selp.b32 	%r416, 16, 0, %p4;
	// begin inline asm
	cp.async.cg.shared.global [ %r415 + 0 ], [ %rd105 + 0 ], 0x10, %r416;
	// end inline asm
	add.s32 	%r417, %r415, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r417 + 0 ], [ %rd106 + 0 ], 0x10, %r416;
	// end inline asm
	add.s32 	%r418, %r415, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r418 + 0 ], [ %rd107 + 0 ], 0x10, %r416;
	// end inline asm
	add.s32 	%r419, %r415, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r419 + 0 ], [ %rd108 + 0 ], 0x10, %r416;
	// end inline asm
	cp.async.commit_group;
	.loc	1 160 47                        // sk07_lm_head.py:160:47
	shl.b32 	%r593, %r1412, 13;
	add.s32 	%r594, %r47, %r593;
	add.s32 	%r420, %r594, 49152;
	// begin inline asm
	cp.async.cg.shared.global [ %r420 + 0 ], [ %rd109 + 0 ], 0x10, %r416;
	// end inline asm
	add.s32 	%r421, %r594, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r421 + 0 ], [ %rd110 + 0 ], 0x10, %r416;
	// end inline asm
	cp.async.commit_group;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	add.s32 	%r1413, %r1413, 1;
	add.s64 	%rd319, %rd319, 64;
	add.s64 	%rd318, %rd318, 64;
	add.s64 	%rd317, %rd317, 64;
	add.s64 	%rd316, %rd316, 64;
	add.s64 	%rd315, %rd315, 64;
	add.s64 	%rd314, %rd314, 64;
	setp.ne.b32 	%p7, %r8, %r1413;
	@%p7 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_4;
$L__BB0_1:                              // %.._crit_edge_crit_edge
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	and.b32 	%r1415, %r2, 16;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	shl.b32 	%r1414, %r2, 4;
	mov.b32 	%r1416, 0f00000000;
	mov.b32 	%r1417, %r1416;
	mov.b32 	%r1418, %r1416;
	mov.b32 	%r1419, %r1416;
	mov.b32 	%r1420, %r1416;
	mov.b32 	%r1421, %r1416;
	mov.b32 	%r1422, %r1416;
	mov.b32 	%r1423, %r1416;
	mov.b32 	%r1424, %r1416;
	mov.b32 	%r1425, %r1416;
	mov.b32 	%r1426, %r1416;
	mov.b32 	%r1427, %r1416;
	mov.b32 	%r1428, %r1416;
	mov.b32 	%r1429, %r1416;
	mov.b32 	%r1430, %r1416;
	mov.b32 	%r1431, %r1416;
	mov.b32 	%r1432, %r1416;
	mov.b32 	%r1433, %r1416;
	mov.b32 	%r1434, %r1416;
	mov.b32 	%r1435, %r1416;
	mov.b32 	%r1436, %r1416;
	mov.b32 	%r1437, %r1416;
	mov.b32 	%r1438, %r1416;
	mov.b32 	%r1439, %r1416;
	mov.b32 	%r1440, %r1416;
	mov.b32 	%r1441, %r1416;
	mov.b32 	%r1442, %r1416;
	mov.b32 	%r1443, %r1416;
	mov.b32 	%r1444, %r1416;
	mov.b32 	%r1445, %r1416;
	mov.b32 	%r1446, %r1416;
	mov.b32 	%r1447, %r1416;
	mov.b32 	%r1448, %r1416;
	mov.b32 	%r1449, %r1416;
	mov.b32 	%r1450, %r1416;
	mov.b32 	%r1451, %r1416;
	mov.b32 	%r1452, %r1416;
	mov.b32 	%r1453, %r1416;
	mov.b32 	%r1454, %r1416;
	mov.b32 	%r1455, %r1416;
	mov.b32 	%r1456, %r1416;
	mov.b32 	%r1457, %r1416;
	mov.b32 	%r1458, %r1416;
	mov.b32 	%r1459, %r1416;
	mov.b32 	%r1460, %r1416;
	mov.b32 	%r1461, %r1416;
	mov.b32 	%r1462, %r1416;
	mov.b32 	%r1463, %r1416;
	mov.b32 	%r1464, %r1416;
	mov.b32 	%r1465, %r1416;
	mov.b32 	%r1466, %r1416;
	mov.b32 	%r1467, %r1416;
	mov.b32 	%r1468, %r1416;
	mov.b32 	%r1469, %r1416;
	mov.b32 	%r1470, %r1416;
	mov.b32 	%r1471, %r1416;
	mov.b32 	%r1472, %r1416;
	mov.b32 	%r1473, %r1416;
	mov.b32 	%r1474, %r1416;
	mov.b32 	%r1475, %r1416;
	mov.b32 	%r1476, %r1416;
	mov.b32 	%r1477, %r1416;
	mov.b32 	%r1478, %r1416;
	mov.b32 	%r1479, %r1416;
	mov.b32 	%r1480, %r1416;
	mov.b32 	%r1481, %r1416;
	mov.b32 	%r1482, %r1416;
	mov.b32 	%r1483, %r1416;
	mov.b32 	%r1484, %r1416;
	mov.b32 	%r1485, %r1416;
	mov.b32 	%r1486, %r1416;
	mov.b32 	%r1487, %r1416;
	mov.b32 	%r1488, %r1416;
	mov.b32 	%r1489, %r1416;
	mov.b32 	%r1490, %r1416;
	mov.b32 	%r1491, %r1416;
	mov.b32 	%r1492, %r1416;
	mov.b32 	%r1493, %r1416;
	mov.b32 	%r1494, %r1416;
	mov.b32 	%r1495, %r1416;
	mov.b32 	%r1496, %r1416;
	mov.b32 	%r1497, %r1416;
	mov.b32 	%r1498, %r1416;
	mov.b32 	%r1499, %r1416;
	mov.b32 	%r1500, %r1416;
	mov.b32 	%r1501, %r1416;
	mov.b32 	%r1502, %r1416;
	mov.b32 	%r1503, %r1416;
	mov.b32 	%r1504, %r1416;
	mov.b32 	%r1505, %r1416;
	mov.b32 	%r1506, %r1416;
	mov.b32 	%r1507, %r1416;
	mov.b32 	%r1508, %r1416;
	mov.b32 	%r1509, %r1416;
	mov.b32 	%r1510, %r1416;
	mov.b32 	%r1511, %r1416;
	mov.b32 	%r1512, %r1416;
	mov.b32 	%r1513, %r1416;
	mov.b32 	%r1514, %r1416;
	mov.b32 	%r1515, %r1416;
	mov.b32 	%r1516, %r1416;
	mov.b32 	%r1517, %r1416;
	mov.b32 	%r1518, %r1416;
	mov.b32 	%r1519, %r1416;
	mov.b32 	%r1520, %r1416;
	mov.b32 	%r1521, %r1416;
	mov.b32 	%r1522, %r1416;
	mov.b32 	%r1523, %r1416;
	mov.b32 	%r1524, %r1416;
	mov.b32 	%r1525, %r1416;
	mov.b32 	%r1526, %r1416;
	mov.b32 	%r1527, %r1416;
	mov.b32 	%r1528, %r1416;
	mov.b32 	%r1529, %r1416;
	mov.b32 	%r1530, %r1416;
	mov.b32 	%r1531, %r1416;
	mov.b32 	%r1532, %r1416;
	mov.b32 	%r1533, %r1416;
	mov.b32 	%r1534, %r1416;
	mov.b32 	%r1535, %r1416;
	mov.b32 	%r1536, %r1416;
	mov.b32 	%r1537, %r1416;
	mov.b32 	%r1538, %r1416;
	mov.b32 	%r1539, %r1416;
	mov.b32 	%r1540, %r1416;
	mov.b32 	%r1541, %r1416;
	mov.b32 	%r1542, %r1416;
	mov.b32 	%r1543, %r1416;
$L__BB0_4:                              // %._crit_edge
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r825, %r1, %r3;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r826, %r825, %r15;
	.loc	1 141 45                        // sk07_lm_head.py:141:45
	and.b32 	%r827, %r2, 240;
	bfe.u32 	%r828, %r2, 4, 4;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r829, %r828, %r1;
	or.b32 	%r830, %r829, 240;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r831, %r830, %r15;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r832, %r829, 224;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r833, %r832, %r15;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r834, %r829, 208;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r835, %r834, %r15;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r836, %r829, 192;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r837, %r836, %r15;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r838, %r829, 176;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r839, %r838, %r15;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r840, %r829, 160;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r841, %r840, %r15;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r842, %r829, 144;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r843, %r842, %r15;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r844, %r829, 128;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r845, %r844, %r15;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r846, %r829, 112;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r847, %r846, %r15;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r848, %r829, 96;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r849, %r848, %r15;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r850, %r829, 80;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r851, %r850, %r15;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r852, %r829, 64;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r853, %r852, %r15;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r854, %r829, 48;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r855, %r854, %r15;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r856, %r829, 32;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r857, %r856, %r15;
	.loc	1 141 32                        // sk07_lm_head.py:141:32
	or.b32 	%r858, %r829, 16;
	.loc	1 141 57                        // sk07_lm_head.py:141:57
	rem.s32 	%r859, %r858, %r15;
	rem.s32 	%r860, %r829, %r15;
	.loc	1 151 23                        // sk07_lm_head.py:151:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 164 38                        // sk07_lm_head.py:164:38
	mad.wide.s32 	%rd112, %r826, 4, %rd28;
	.loc	1 164 24                        // sk07_lm_head.py:164:24
	// begin inline asm
	mov.u32 %r596, 0x0;
	ld.global.b32 { %r596 }, [ %rd112 + 0 ];
	// end inline asm
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	and.b32 	%r861, %r2, 7;
	shl.b32 	%r862, %r861, 3;
	shl.b32 	%r863, %r827, 2;
	and.b32 	%r864, %r2, 8;
	shr.u32 	%r865, %r864, 1;
	add.s32 	%r866, %r209, %r862;
	add.s32 	%r867, %r866, %r863;
	add.s32 	%r595, %r867, %r865;
	// begin inline asm
	st.shared.b32 [ %r595 + 0 ], %r596;
	// end inline asm
	bar.sync 	0;
	and.b32 	%r868, %r9, 56;
	and.b32 	%r869, %r2, 192;
	add.s32 	%r870, %r209, %r868;
	add.s32 	%r871, %r870, %r869;
	ld.shared.v2.b32 	{%r872, %r873}, [%r871];
	ld.shared.v2.b32 	{%r874, %r875}, [%r871+256];
	ld.shared.v2.b32 	{%r876, %r877}, [%r871+512];
	ld.shared.v2.b32 	{%r878, %r879}, [%r871+768];
	.loc	1 165 38                        // sk07_lm_head.py:165:38
	mad.wide.s32 	%rd113, %r23, 4, %rd29;
	mad.wide.s32 	%rd114, %r24, 4, %rd29;
	mad.wide.s32 	%rd115, %r25, 4, %rd29;
	mad.wide.s32 	%rd116, %r26, 4, %rd29;
	mad.wide.s32 	%rd117, %r27, 4, %rd29;
	mad.wide.s32 	%rd118, %r28, 4, %rd29;
	mad.wide.s32 	%rd119, %r29, 4, %rd29;
	mad.wide.s32 	%rd120, %r30, 4, %rd29;
	mad.wide.s32 	%rd121, %r31, 4, %rd29;
	mad.wide.s32 	%rd122, %r32, 4, %rd29;
	mad.wide.s32 	%rd123, %r33, 4, %rd29;
	mad.wide.s32 	%rd124, %r34, 4, %rd29;
	mad.wide.s32 	%rd125, %r35, 4, %rd29;
	mad.wide.s32 	%rd126, %r36, 4, %rd29;
	mad.wide.s32 	%rd127, %r37, 4, %rd29;
	mad.wide.s32 	%rd128, %r38, 4, %rd29;
	.loc	1 165 24                        // sk07_lm_head.py:165:24
	// begin inline asm
	mov.u32 %r597, 0x0;
	ld.global.b32 { %r597 }, [ %rd113 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r598, 0x0;
	ld.global.b32 { %r598 }, [ %rd114 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r599, 0x0;
	ld.global.b32 { %r599 }, [ %rd115 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r600, 0x0;
	ld.global.b32 { %r600 }, [ %rd116 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r601, 0x0;
	ld.global.b32 { %r601 }, [ %rd117 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r602, 0x0;
	ld.global.b32 { %r602 }, [ %rd118 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r603, 0x0;
	ld.global.b32 { %r603 }, [ %rd119 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r604, 0x0;
	ld.global.b32 { %r604 }, [ %rd120 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r605, 0x0;
	ld.global.b32 { %r605 }, [ %rd121 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r606, 0x0;
	ld.global.b32 { %r606 }, [ %rd122 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r607, 0x0;
	ld.global.b32 { %r607 }, [ %rd123 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r608, 0x0;
	ld.global.b32 { %r608 }, [ %rd124 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r609, 0x0;
	ld.global.b32 { %r609 }, [ %rd125 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r610, 0x0;
	ld.global.b32 { %r610 }, [ %rd126 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r611, 0x0;
	ld.global.b32 { %r611 }, [ %rd127 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r612, 0x0;
	ld.global.b32 { %r612 }, [ %rd128 + 0 ];
	// end inline asm
	.loc	1 166 49                        // sk07_lm_head.py:166:49
	mul.lo.s32 	%r880, %r860, %r18;
	mul.lo.s32 	%r881, %r859, %r18;
	mul.lo.s32 	%r882, %r857, %r18;
	mul.lo.s32 	%r883, %r855, %r18;
	mul.lo.s32 	%r884, %r853, %r18;
	mul.lo.s32 	%r885, %r851, %r18;
	mul.lo.s32 	%r886, %r849, %r18;
	mul.lo.s32 	%r887, %r847, %r18;
	mul.lo.s32 	%r888, %r845, %r18;
	mul.lo.s32 	%r889, %r843, %r18;
	mul.lo.s32 	%r890, %r841, %r18;
	mul.lo.s32 	%r891, %r839, %r18;
	mul.lo.s32 	%r892, %r837, %r18;
	mul.lo.s32 	%r893, %r835, %r18;
	mul.lo.s32 	%r894, %r833, %r18;
	mul.lo.s32 	%r895, %r831, %r18;
	.loc	1 166 31                        // sk07_lm_head.py:166:31
	mad.wide.s32 	%rd273, %r880, 2, %rd27;
	mad.wide.s32 	%rd274, %r881, 2, %rd27;
	mad.wide.s32 	%rd275, %r882, 2, %rd27;
	mad.wide.s32 	%rd276, %r883, 2, %rd27;
	mad.wide.s32 	%rd277, %r884, 2, %rd27;
	mad.wide.s32 	%rd278, %r885, 2, %rd27;
	mad.wide.s32 	%rd279, %r886, 2, %rd27;
	mad.wide.s32 	%rd280, %r887, 2, %rd27;
	mad.wide.s32 	%rd281, %r888, 2, %rd27;
	mad.wide.s32 	%rd282, %r889, 2, %rd27;
	mad.wide.s32 	%rd283, %r890, 2, %rd27;
	mad.wide.s32 	%rd284, %r891, 2, %rd27;
	mad.wide.s32 	%rd285, %r892, 2, %rd27;
	mad.wide.s32 	%rd286, %r893, 2, %rd27;
	mad.wide.s32 	%rd287, %r894, 2, %rd27;
	mad.wide.s32 	%rd288, %r895, 2, %rd27;
	.loc	1 166 80                        // sk07_lm_head.py:166:80
	mul.lo.s32 	%r896, %r39, %r19;
	mul.lo.s32 	%r897, %r40, %r19;
	mul.lo.s32 	%r898, %r41, %r19;
	mul.lo.s32 	%r899, %r42, %r19;
	mul.lo.s32 	%r900, %r43, %r19;
	mul.lo.s32 	%r901, %r44, %r19;
	mul.lo.s32 	%r902, %r45, %r19;
	mul.lo.s32 	%r903, %r46, %r19;
	.loc	1 166 64                        // sk07_lm_head.py:166:64
	mul.wide.s32 	%rd289, %r896, 2;
	add.s64 	%rd129, %rd273, %rd289;
	mul.wide.s32 	%rd290, %r897, 2;
	add.s64 	%rd130, %rd273, %rd290;
	mul.wide.s32 	%rd291, %r898, 2;
	add.s64 	%rd131, %rd273, %rd291;
	mul.wide.s32 	%rd292, %r899, 2;
	add.s64 	%rd132, %rd273, %rd292;
	mul.wide.s32 	%rd293, %r900, 2;
	add.s64 	%rd133, %rd273, %rd293;
	mul.wide.s32 	%rd294, %r901, 2;
	add.s64 	%rd134, %rd273, %rd294;
	mul.wide.s32 	%rd295, %r902, 2;
	add.s64 	%rd135, %rd273, %rd295;
	mul.wide.s32 	%rd296, %r903, 2;
	add.s64 	%rd136, %rd273, %rd296;
	add.s64 	%rd137, %rd274, %rd289;
	add.s64 	%rd138, %rd274, %rd290;
	add.s64 	%rd139, %rd274, %rd291;
	add.s64 	%rd140, %rd274, %rd292;
	add.s64 	%rd141, %rd274, %rd293;
	add.s64 	%rd142, %rd274, %rd294;
	add.s64 	%rd143, %rd274, %rd295;
	add.s64 	%rd144, %rd274, %rd296;
	add.s64 	%rd145, %rd275, %rd289;
	add.s64 	%rd146, %rd275, %rd290;
	add.s64 	%rd147, %rd275, %rd291;
	add.s64 	%rd148, %rd275, %rd292;
	add.s64 	%rd149, %rd275, %rd293;
	add.s64 	%rd150, %rd275, %rd294;
	add.s64 	%rd151, %rd275, %rd295;
	add.s64 	%rd152, %rd275, %rd296;
	add.s64 	%rd153, %rd276, %rd289;
	add.s64 	%rd154, %rd276, %rd290;
	add.s64 	%rd155, %rd276, %rd291;
	add.s64 	%rd156, %rd276, %rd292;
	add.s64 	%rd157, %rd276, %rd293;
	add.s64 	%rd158, %rd276, %rd294;
	add.s64 	%rd159, %rd276, %rd295;
	add.s64 	%rd160, %rd276, %rd296;
	add.s64 	%rd161, %rd277, %rd289;
	add.s64 	%rd162, %rd277, %rd290;
	add.s64 	%rd163, %rd277, %rd291;
	add.s64 	%rd164, %rd277, %rd292;
	add.s64 	%rd165, %rd277, %rd293;
	add.s64 	%rd166, %rd277, %rd294;
	add.s64 	%rd167, %rd277, %rd295;
	add.s64 	%rd168, %rd277, %rd296;
	add.s64 	%rd169, %rd278, %rd289;
	add.s64 	%rd170, %rd278, %rd290;
	add.s64 	%rd171, %rd278, %rd291;
	add.s64 	%rd172, %rd278, %rd292;
	add.s64 	%rd173, %rd278, %rd293;
	add.s64 	%rd174, %rd278, %rd294;
	add.s64 	%rd175, %rd278, %rd295;
	add.s64 	%rd176, %rd278, %rd296;
	add.s64 	%rd177, %rd279, %rd289;
	add.s64 	%rd178, %rd279, %rd290;
	add.s64 	%rd179, %rd279, %rd291;
	add.s64 	%rd180, %rd279, %rd292;
	add.s64 	%rd181, %rd279, %rd293;
	add.s64 	%rd182, %rd279, %rd294;
	add.s64 	%rd183, %rd279, %rd295;
	add.s64 	%rd184, %rd279, %rd296;
	add.s64 	%rd185, %rd280, %rd289;
	add.s64 	%rd186, %rd280, %rd290;
	add.s64 	%rd187, %rd280, %rd291;
	add.s64 	%rd188, %rd280, %rd292;
	add.s64 	%rd189, %rd280, %rd293;
	add.s64 	%rd190, %rd280, %rd294;
	add.s64 	%rd191, %rd280, %rd295;
	add.s64 	%rd192, %rd280, %rd296;
	add.s64 	%rd193, %rd281, %rd289;
	add.s64 	%rd194, %rd281, %rd290;
	add.s64 	%rd195, %rd281, %rd291;
	add.s64 	%rd196, %rd281, %rd292;
	add.s64 	%rd197, %rd281, %rd293;
	add.s64 	%rd198, %rd281, %rd294;
	add.s64 	%rd199, %rd281, %rd295;
	add.s64 	%rd200, %rd281, %rd296;
	add.s64 	%rd201, %rd282, %rd289;
	add.s64 	%rd202, %rd282, %rd290;
	add.s64 	%rd203, %rd282, %rd291;
	add.s64 	%rd204, %rd282, %rd292;
	add.s64 	%rd205, %rd282, %rd293;
	add.s64 	%rd206, %rd282, %rd294;
	add.s64 	%rd207, %rd282, %rd295;
	add.s64 	%rd208, %rd282, %rd296;
	add.s64 	%rd209, %rd283, %rd289;
	add.s64 	%rd210, %rd283, %rd290;
	add.s64 	%rd211, %rd283, %rd291;
	add.s64 	%rd212, %rd283, %rd292;
	add.s64 	%rd213, %rd283, %rd293;
	add.s64 	%rd214, %rd283, %rd294;
	add.s64 	%rd215, %rd283, %rd295;
	add.s64 	%rd216, %rd283, %rd296;
	add.s64 	%rd217, %rd284, %rd289;
	add.s64 	%rd218, %rd284, %rd290;
	add.s64 	%rd219, %rd284, %rd291;
	add.s64 	%rd220, %rd284, %rd292;
	add.s64 	%rd221, %rd284, %rd293;
	add.s64 	%rd222, %rd284, %rd294;
	add.s64 	%rd223, %rd284, %rd295;
	add.s64 	%rd224, %rd284, %rd296;
	add.s64 	%rd225, %rd285, %rd289;
	add.s64 	%rd226, %rd285, %rd290;
	add.s64 	%rd227, %rd285, %rd291;
	add.s64 	%rd228, %rd285, %rd292;
	add.s64 	%rd229, %rd285, %rd293;
	add.s64 	%rd230, %rd285, %rd294;
	add.s64 	%rd231, %rd285, %rd295;
	add.s64 	%rd232, %rd285, %rd296;
	add.s64 	%rd233, %rd286, %rd289;
	add.s64 	%rd234, %rd286, %rd290;
	add.s64 	%rd235, %rd286, %rd291;
	add.s64 	%rd236, %rd286, %rd292;
	add.s64 	%rd237, %rd286, %rd293;
	add.s64 	%rd238, %rd286, %rd294;
	add.s64 	%rd239, %rd286, %rd295;
	add.s64 	%rd240, %rd286, %rd296;
	add.s64 	%rd241, %rd287, %rd289;
	add.s64 	%rd242, %rd287, %rd290;
	add.s64 	%rd243, %rd287, %rd291;
	add.s64 	%rd244, %rd287, %rd292;
	add.s64 	%rd245, %rd287, %rd293;
	add.s64 	%rd246, %rd287, %rd294;
	add.s64 	%rd247, %rd287, %rd295;
	add.s64 	%rd248, %rd287, %rd296;
	add.s64 	%rd249, %rd288, %rd289;
	add.s64 	%rd250, %rd288, %rd290;
	add.s64 	%rd251, %rd288, %rd291;
	add.s64 	%rd252, %rd288, %rd292;
	add.s64 	%rd253, %rd288, %rd293;
	add.s64 	%rd254, %rd288, %rd294;
	add.s64 	%rd255, %rd288, %rd295;
	add.s64 	%rd256, %rd288, %rd296;
	.loc	1 166 19                        // sk07_lm_head.py:166:19
	// begin inline asm
	mov.u16 %rs33, 0x0;
	ld.global.b16 { %rs33 }, [ %rd129 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs34, 0x0;
	ld.global.b16 { %rs34 }, [ %rd130 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs35, 0x0;
	ld.global.b16 { %rs35 }, [ %rd131 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs36, 0x0;
	ld.global.b16 { %rs36 }, [ %rd132 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs37, 0x0;
	ld.global.b16 { %rs37 }, [ %rd133 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs38, 0x0;
	ld.global.b16 { %rs38 }, [ %rd134 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs39, 0x0;
	ld.global.b16 { %rs39 }, [ %rd135 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs40, 0x0;
	ld.global.b16 { %rs40 }, [ %rd136 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs41, 0x0;
	ld.global.b16 { %rs41 }, [ %rd137 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs42, 0x0;
	ld.global.b16 { %rs42 }, [ %rd138 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs43, 0x0;
	ld.global.b16 { %rs43 }, [ %rd139 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs44, 0x0;
	ld.global.b16 { %rs44 }, [ %rd140 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs45, 0x0;
	ld.global.b16 { %rs45 }, [ %rd141 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs46, 0x0;
	ld.global.b16 { %rs46 }, [ %rd142 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs47, 0x0;
	ld.global.b16 { %rs47 }, [ %rd143 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs48, 0x0;
	ld.global.b16 { %rs48 }, [ %rd144 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs49, 0x0;
	ld.global.b16 { %rs49 }, [ %rd145 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs50, 0x0;
	ld.global.b16 { %rs50 }, [ %rd146 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs51, 0x0;
	ld.global.b16 { %rs51 }, [ %rd147 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs52, 0x0;
	ld.global.b16 { %rs52 }, [ %rd148 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs53, 0x0;
	ld.global.b16 { %rs53 }, [ %rd149 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs54, 0x0;
	ld.global.b16 { %rs54 }, [ %rd150 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs55, 0x0;
	ld.global.b16 { %rs55 }, [ %rd151 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs56, 0x0;
	ld.global.b16 { %rs56 }, [ %rd152 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs57, 0x0;
	ld.global.b16 { %rs57 }, [ %rd153 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs58, 0x0;
	ld.global.b16 { %rs58 }, [ %rd154 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs59, 0x0;
	ld.global.b16 { %rs59 }, [ %rd155 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs60, 0x0;
	ld.global.b16 { %rs60 }, [ %rd156 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs61, 0x0;
	ld.global.b16 { %rs61 }, [ %rd157 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs62, 0x0;
	ld.global.b16 { %rs62 }, [ %rd158 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs63, 0x0;
	ld.global.b16 { %rs63 }, [ %rd159 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs64, 0x0;
	ld.global.b16 { %rs64 }, [ %rd160 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs65, 0x0;
	ld.global.b16 { %rs65 }, [ %rd161 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs66, 0x0;
	ld.global.b16 { %rs66 }, [ %rd162 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs67, 0x0;
	ld.global.b16 { %rs67 }, [ %rd163 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs68, 0x0;
	ld.global.b16 { %rs68 }, [ %rd164 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs69, 0x0;
	ld.global.b16 { %rs69 }, [ %rd165 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs70, 0x0;
	ld.global.b16 { %rs70 }, [ %rd166 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs71, 0x0;
	ld.global.b16 { %rs71 }, [ %rd167 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs72, 0x0;
	ld.global.b16 { %rs72 }, [ %rd168 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs73, 0x0;
	ld.global.b16 { %rs73 }, [ %rd169 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs74, 0x0;
	ld.global.b16 { %rs74 }, [ %rd170 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs75, 0x0;
	ld.global.b16 { %rs75 }, [ %rd171 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs76, 0x0;
	ld.global.b16 { %rs76 }, [ %rd172 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs77, 0x0;
	ld.global.b16 { %rs77 }, [ %rd173 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs78, 0x0;
	ld.global.b16 { %rs78 }, [ %rd174 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs79, 0x0;
	ld.global.b16 { %rs79 }, [ %rd175 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs80, 0x0;
	ld.global.b16 { %rs80 }, [ %rd176 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs81, 0x0;
	ld.global.b16 { %rs81 }, [ %rd177 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs82, 0x0;
	ld.global.b16 { %rs82 }, [ %rd178 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs83, 0x0;
	ld.global.b16 { %rs83 }, [ %rd179 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs84, 0x0;
	ld.global.b16 { %rs84 }, [ %rd180 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs85, 0x0;
	ld.global.b16 { %rs85 }, [ %rd181 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs86, 0x0;
	ld.global.b16 { %rs86 }, [ %rd182 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs87, 0x0;
	ld.global.b16 { %rs87 }, [ %rd183 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs88, 0x0;
	ld.global.b16 { %rs88 }, [ %rd184 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs89, 0x0;
	ld.global.b16 { %rs89 }, [ %rd185 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs90, 0x0;
	ld.global.b16 { %rs90 }, [ %rd186 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs91, 0x0;
	ld.global.b16 { %rs91 }, [ %rd187 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs92, 0x0;
	ld.global.b16 { %rs92 }, [ %rd188 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs93, 0x0;
	ld.global.b16 { %rs93 }, [ %rd189 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs94, 0x0;
	ld.global.b16 { %rs94 }, [ %rd190 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs95, 0x0;
	ld.global.b16 { %rs95 }, [ %rd191 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs96, 0x0;
	ld.global.b16 { %rs96 }, [ %rd192 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs97, 0x0;
	ld.global.b16 { %rs97 }, [ %rd193 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs98, 0x0;
	ld.global.b16 { %rs98 }, [ %rd194 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs99, 0x0;
	ld.global.b16 { %rs99 }, [ %rd195 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs100, 0x0;
	ld.global.b16 { %rs100 }, [ %rd196 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs101, 0x0;
	ld.global.b16 { %rs101 }, [ %rd197 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs102, 0x0;
	ld.global.b16 { %rs102 }, [ %rd198 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs103, 0x0;
	ld.global.b16 { %rs103 }, [ %rd199 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs104, 0x0;
	ld.global.b16 { %rs104 }, [ %rd200 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs105, 0x0;
	ld.global.b16 { %rs105 }, [ %rd201 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs106, 0x0;
	ld.global.b16 { %rs106 }, [ %rd202 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs107, 0x0;
	ld.global.b16 { %rs107 }, [ %rd203 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs108, 0x0;
	ld.global.b16 { %rs108 }, [ %rd204 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs109, 0x0;
	ld.global.b16 { %rs109 }, [ %rd205 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs110, 0x0;
	ld.global.b16 { %rs110 }, [ %rd206 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs111, 0x0;
	ld.global.b16 { %rs111 }, [ %rd207 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs112, 0x0;
	ld.global.b16 { %rs112 }, [ %rd208 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs113, 0x0;
	ld.global.b16 { %rs113 }, [ %rd209 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs114, 0x0;
	ld.global.b16 { %rs114 }, [ %rd210 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs115, 0x0;
	ld.global.b16 { %rs115 }, [ %rd211 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs116, 0x0;
	ld.global.b16 { %rs116 }, [ %rd212 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs117, 0x0;
	ld.global.b16 { %rs117 }, [ %rd213 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs118, 0x0;
	ld.global.b16 { %rs118 }, [ %rd214 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs119, 0x0;
	ld.global.b16 { %rs119 }, [ %rd215 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs120, 0x0;
	ld.global.b16 { %rs120 }, [ %rd216 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs121, 0x0;
	ld.global.b16 { %rs121 }, [ %rd217 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs122, 0x0;
	ld.global.b16 { %rs122 }, [ %rd218 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs123, 0x0;
	ld.global.b16 { %rs123 }, [ %rd219 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs124, 0x0;
	ld.global.b16 { %rs124 }, [ %rd220 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs125, 0x0;
	ld.global.b16 { %rs125 }, [ %rd221 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs126, 0x0;
	ld.global.b16 { %rs126 }, [ %rd222 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs127, 0x0;
	ld.global.b16 { %rs127 }, [ %rd223 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs128, 0x0;
	ld.global.b16 { %rs128 }, [ %rd224 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs129, 0x0;
	ld.global.b16 { %rs129 }, [ %rd225 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs130, 0x0;
	ld.global.b16 { %rs130 }, [ %rd226 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs131, 0x0;
	ld.global.b16 { %rs131 }, [ %rd227 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs132, 0x0;
	ld.global.b16 { %rs132 }, [ %rd228 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs133, 0x0;
	ld.global.b16 { %rs133 }, [ %rd229 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs134, 0x0;
	ld.global.b16 { %rs134 }, [ %rd230 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs135, 0x0;
	ld.global.b16 { %rs135 }, [ %rd231 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs136, 0x0;
	ld.global.b16 { %rs136 }, [ %rd232 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs137, 0x0;
	ld.global.b16 { %rs137 }, [ %rd233 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs138, 0x0;
	ld.global.b16 { %rs138 }, [ %rd234 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs139, 0x0;
	ld.global.b16 { %rs139 }, [ %rd235 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs140, 0x0;
	ld.global.b16 { %rs140 }, [ %rd236 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs141, 0x0;
	ld.global.b16 { %rs141 }, [ %rd237 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs142, 0x0;
	ld.global.b16 { %rs142 }, [ %rd238 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs143, 0x0;
	ld.global.b16 { %rs143 }, [ %rd239 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs144, 0x0;
	ld.global.b16 { %rs144 }, [ %rd240 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs145, 0x0;
	ld.global.b16 { %rs145 }, [ %rd241 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs146, 0x0;
	ld.global.b16 { %rs146 }, [ %rd242 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs147, 0x0;
	ld.global.b16 { %rs147 }, [ %rd243 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs148, 0x0;
	ld.global.b16 { %rs148 }, [ %rd244 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs149, 0x0;
	ld.global.b16 { %rs149 }, [ %rd245 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs150, 0x0;
	ld.global.b16 { %rs150 }, [ %rd246 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs151, 0x0;
	ld.global.b16 { %rs151 }, [ %rd247 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs152, 0x0;
	ld.global.b16 { %rs152 }, [ %rd248 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs153, 0x0;
	ld.global.b16 { %rs153 }, [ %rd249 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs154, 0x0;
	ld.global.b16 { %rs154 }, [ %rd250 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs155, 0x0;
	ld.global.b16 { %rs155 }, [ %rd251 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs156, 0x0;
	ld.global.b16 { %rs156 }, [ %rd252 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs157, 0x0;
	ld.global.b16 { %rs157 }, [ %rd253 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs158, 0x0;
	ld.global.b16 { %rs158 }, [ %rd254 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs159, 0x0;
	ld.global.b16 { %rs159 }, [ %rd255 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs160, 0x0;
	ld.global.b16 { %rs160 }, [ %rd256 + 0 ];
	// end inline asm
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	bar.sync 	0;
	shl.b32 	%r904, %r2, 7;
	and.b32 	%r905, %r904, 15360;
	shl.b32 	%r906, %r861, 4;
	or.b32 	%r907, %r905, %r906;
	xor.b32 	%r908, %r907, %r827;
	add.s32 	%r613, %r209, %r908;
	mov.b32 	%r614, {%rs33, %rs34};
	mov.b32 	%r615, {%rs35, %rs36};
	mov.b32 	%r616, {%rs37, %rs38};
	mov.b32 	%r617, {%rs39, %rs40};
	// begin inline asm
	st.shared.v4.b32 [ %r613 + 0 ], { %r614, %r615, %r616, %r617 };
	// end inline asm
	add.s32 	%r618, %r613, 256;
	mov.b32 	%r619, {%rs41, %rs42};
	mov.b32 	%r620, {%rs43, %rs44};
	mov.b32 	%r621, {%rs45, %rs46};
	mov.b32 	%r622, {%rs47, %rs48};
	// begin inline asm
	st.shared.v4.b32 [ %r618 + 0 ], { %r619, %r620, %r621, %r622 };
	// end inline asm
	add.s32 	%r623, %r613, 512;
	mov.b32 	%r624, {%rs49, %rs50};
	mov.b32 	%r625, {%rs51, %rs52};
	mov.b32 	%r626, {%rs53, %rs54};
	mov.b32 	%r627, {%rs55, %rs56};
	// begin inline asm
	st.shared.v4.b32 [ %r623 + 0 ], { %r624, %r625, %r626, %r627 };
	// end inline asm
	add.s32 	%r628, %r613, 768;
	mov.b32 	%r629, {%rs57, %rs58};
	mov.b32 	%r630, {%rs59, %rs60};
	mov.b32 	%r631, {%rs61, %rs62};
	mov.b32 	%r632, {%rs63, %rs64};
	// begin inline asm
	st.shared.v4.b32 [ %r628 + 0 ], { %r629, %r630, %r631, %r632 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r909, %r861, 11;
	shl.b32 	%r910, %r6, 4;
	shl.b32 	%r911, %r869, 2;
	setp.eq.b32 	%p24, %r1415, 0;
	shl.b32 	%r912, %r1415, 1;
	shr.u32 	%r913, %r5, 1;
	or.b32 	%r914, %r910, %r911;
	or.b32 	%r915, %r912, %r913;
	xor.b32 	%r916, %r914, %r915;
	or.b32 	%r917, %r916, %r909;
	add.s32 	%r918, %r209, %r917;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r919, %r920, %r921, %r922}, [%r918];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r923, %r924, %r925, %r926}, [%r918+1024];
	xor.b32 	%r927, %r917, 64;
	add.s32 	%r928, %r209, %r927;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r929, %r930, %r931, %r932}, [%r928];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r933, %r934, %r935, %r936}, [%r928+1024];
	bar.sync 	0;
	mov.b32 	%r633, {%rs65, %rs66};
	mov.b32 	%r634, {%rs67, %rs68};
	mov.b32 	%r635, {%rs69, %rs70};
	mov.b32 	%r636, {%rs71, %rs72};
	// begin inline asm
	st.shared.v4.b32 [ %r613 + 0 ], { %r633, %r634, %r635, %r636 };
	// end inline asm
	mov.b32 	%r637, {%rs73, %rs74};
	mov.b32 	%r638, {%rs75, %rs76};
	mov.b32 	%r639, {%rs77, %rs78};
	mov.b32 	%r640, {%rs79, %rs80};
	// begin inline asm
	st.shared.v4.b32 [ %r618 + 0 ], { %r637, %r638, %r639, %r640 };
	// end inline asm
	mov.b32 	%r641, {%rs81, %rs82};
	mov.b32 	%r642, {%rs83, %rs84};
	mov.b32 	%r643, {%rs85, %rs86};
	mov.b32 	%r644, {%rs87, %rs88};
	// begin inline asm
	st.shared.v4.b32 [ %r623 + 0 ], { %r641, %r642, %r643, %r644 };
	// end inline asm
	mov.b32 	%r645, {%rs89, %rs90};
	mov.b32 	%r646, {%rs91, %rs92};
	mov.b32 	%r647, {%rs93, %rs94};
	mov.b32 	%r648, {%rs95, %rs96};
	// begin inline asm
	st.shared.v4.b32 [ %r628 + 0 ], { %r645, %r646, %r647, %r648 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r937, %r938, %r939, %r940}, [%r918];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r941, %r942, %r943, %r944}, [%r918+1024];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r945, %r946, %r947, %r948}, [%r928];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r949, %r950, %r951, %r952}, [%r928+1024];
	bar.sync 	0;
	mov.b32 	%r649, {%rs97, %rs98};
	mov.b32 	%r650, {%rs99, %rs100};
	mov.b32 	%r651, {%rs101, %rs102};
	mov.b32 	%r652, {%rs103, %rs104};
	// begin inline asm
	st.shared.v4.b32 [ %r613 + 0 ], { %r649, %r650, %r651, %r652 };
	// end inline asm
	mov.b32 	%r653, {%rs105, %rs106};
	mov.b32 	%r654, {%rs107, %rs108};
	mov.b32 	%r655, {%rs109, %rs110};
	mov.b32 	%r656, {%rs111, %rs112};
	// begin inline asm
	st.shared.v4.b32 [ %r618 + 0 ], { %r653, %r654, %r655, %r656 };
	// end inline asm
	mov.b32 	%r657, {%rs113, %rs114};
	mov.b32 	%r658, {%rs115, %rs116};
	mov.b32 	%r659, {%rs117, %rs118};
	mov.b32 	%r660, {%rs119, %rs120};
	// begin inline asm
	st.shared.v4.b32 [ %r623 + 0 ], { %r657, %r658, %r659, %r660 };
	// end inline asm
	mov.b32 	%r661, {%rs121, %rs122};
	mov.b32 	%r662, {%rs123, %rs124};
	mov.b32 	%r663, {%rs125, %rs126};
	mov.b32 	%r664, {%rs127, %rs128};
	// begin inline asm
	st.shared.v4.b32 [ %r628 + 0 ], { %r661, %r662, %r663, %r664 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r953, %r954, %r955, %r956}, [%r918];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r957, %r958, %r959, %r960}, [%r918+1024];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r961, %r962, %r963, %r964}, [%r928];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r965, %r966, %r967, %r968}, [%r928+1024];
	bar.sync 	0;
	mov.b32 	%r665, {%rs129, %rs130};
	mov.b32 	%r666, {%rs131, %rs132};
	mov.b32 	%r667, {%rs133, %rs134};
	mov.b32 	%r668, {%rs135, %rs136};
	// begin inline asm
	st.shared.v4.b32 [ %r613 + 0 ], { %r665, %r666, %r667, %r668 };
	// end inline asm
	mov.b32 	%r669, {%rs137, %rs138};
	mov.b32 	%r670, {%rs139, %rs140};
	mov.b32 	%r671, {%rs141, %rs142};
	mov.b32 	%r672, {%rs143, %rs144};
	// begin inline asm
	st.shared.v4.b32 [ %r618 + 0 ], { %r669, %r670, %r671, %r672 };
	// end inline asm
	mov.b32 	%r673, {%rs145, %rs146};
	mov.b32 	%r674, {%rs147, %rs148};
	mov.b32 	%r675, {%rs149, %rs150};
	mov.b32 	%r676, {%rs151, %rs152};
	// begin inline asm
	st.shared.v4.b32 [ %r623 + 0 ], { %r673, %r674, %r675, %r676 };
	// end inline asm
	mov.b32 	%r677, {%rs153, %rs154};
	mov.b32 	%r678, {%rs155, %rs156};
	mov.b32 	%r679, {%rs157, %rs158};
	mov.b32 	%r680, {%rs159, %rs160};
	// begin inline asm
	st.shared.v4.b32 [ %r628 + 0 ], { %r677, %r678, %r679, %r680 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r969, %r970, %r971, %r972}, [%r918];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r973, %r974, %r975, %r976}, [%r918+1024];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r977, %r978, %r979, %r980}, [%r928];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r981, %r982, %r983, %r984}, [%r928+1024];
	.loc	1 173 31                        // sk07_lm_head.py:173:31
	setp.lt.s32 	%p25, %r829, %r15;
	setp.lt.s32 	%p26, %r858, %r15;
	setp.lt.s32 	%p27, %r856, %r15;
	setp.lt.s32 	%p28, %r854, %r15;
	setp.lt.s32 	%p29, %r852, %r15;
	setp.lt.s32 	%p30, %r850, %r15;
	setp.lt.s32 	%p31, %r848, %r15;
	setp.lt.s32 	%p32, %r846, %r15;
	setp.lt.s32 	%p33, %r844, %r15;
	setp.lt.s32 	%p34, %r842, %r15;
	setp.lt.s32 	%p35, %r840, %r15;
	setp.lt.s32 	%p36, %r838, %r15;
	setp.lt.s32 	%p37, %r836, %r15;
	setp.lt.s32 	%p38, %r834, %r15;
	setp.lt.s32 	%p39, %r832, %r15;
	setp.lt.s32 	%p40, %r830, %r15;
	.loc	1 173 54                        // sk07_lm_head.py:173:54
	setp.lt.s32 	%p41, %r7, %r16;
	.loc	1 173 37                        // sk07_lm_head.py:173:37
	and.pred 	%p8, %p25, %p41;
	and.pred 	%p9, %p26, %p41;
	and.pred 	%p10, %p27, %p41;
	and.pred 	%p11, %p28, %p41;
	and.pred 	%p12, %p29, %p41;
	and.pred 	%p13, %p30, %p41;
	and.pred 	%p14, %p31, %p41;
	and.pred 	%p15, %p32, %p41;
	and.pred 	%p16, %p33, %p41;
	and.pred 	%p17, %p34, %p41;
	and.pred 	%p18, %p35, %p41;
	and.pred 	%p19, %p36, %p41;
	and.pred 	%p20, %p37, %p41;
	and.pred 	%p21, %p38, %p41;
	and.pred 	%p22, %p39, %p41;
	and.pred 	%p23, %p40, %p41;
	.loc	1 171 35                        // sk07_lm_head.py:171:35
	mul.lo.s32 	%r985, %r829, %r17;
	mul.lo.s32 	%r986, %r858, %r17;
	mul.lo.s32 	%r987, %r856, %r17;
	mul.lo.s32 	%r988, %r854, %r17;
	mul.lo.s32 	%r989, %r852, %r17;
	mul.lo.s32 	%r990, %r850, %r17;
	mul.lo.s32 	%r991, %r848, %r17;
	mul.lo.s32 	%r992, %r846, %r17;
	mul.lo.s32 	%r993, %r844, %r17;
	mul.lo.s32 	%r994, %r842, %r17;
	mul.lo.s32 	%r995, %r840, %r17;
	mul.lo.s32 	%r996, %r838, %r17;
	mul.lo.s32 	%r997, %r836, %r17;
	mul.lo.s32 	%r998, %r834, %r17;
	mul.lo.s32 	%r999, %r832, %r17;
	mul.lo.s32 	%r1000, %r830, %r17;
	.loc	1 171 18                        // sk07_lm_head.py:171:18
	mad.wide.s32 	%rd297, %r985, 2, %rd26;
	mad.wide.s32 	%rd298, %r986, 2, %rd26;
	mad.wide.s32 	%rd299, %r987, 2, %rd26;
	mad.wide.s32 	%rd300, %r988, 2, %rd26;
	mad.wide.s32 	%rd301, %r989, 2, %rd26;
	mad.wide.s32 	%rd302, %r990, 2, %rd26;
	mad.wide.s32 	%rd303, %r991, 2, %rd26;
	mad.wide.s32 	%rd304, %r992, 2, %rd26;
	mad.wide.s32 	%rd305, %r993, 2, %rd26;
	mad.wide.s32 	%rd306, %r994, 2, %rd26;
	mad.wide.s32 	%rd307, %r995, 2, %rd26;
	mad.wide.s32 	%rd308, %r996, 2, %rd26;
	mad.wide.s32 	%rd309, %r997, 2, %rd26;
	mad.wide.s32 	%rd310, %r998, 2, %rd26;
	mad.wide.s32 	%rd311, %r999, 2, %rd26;
	mad.wide.s32 	%rd312, %r1000, 2, %rd26;
	.loc	1 171 50                        // sk07_lm_head.py:171:50
	mul.wide.s32 	%rd313, %r7, 2;
	add.s64 	%rd257, %rd297, %rd313;
	add.s64 	%rd258, %rd298, %rd313;
	add.s64 	%rd259, %rd299, %rd313;
	add.s64 	%rd260, %rd300, %rd313;
	add.s64 	%rd261, %rd301, %rd313;
	add.s64 	%rd262, %rd302, %rd313;
	add.s64 	%rd263, %rd303, %rd313;
	add.s64 	%rd264, %rd304, %rd313;
	add.s64 	%rd265, %rd305, %rd313;
	add.s64 	%rd266, %rd306, %rd313;
	add.s64 	%rd267, %rd307, %rd313;
	add.s64 	%rd268, %rd308, %rd313;
	add.s64 	%rd269, %rd309, %rd313;
	add.s64 	%rd270, %rd310, %rd313;
	add.s64 	%rd271, %rd311, %rd313;
	add.s64 	%rd272, %rd312, %rd313;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1001, %r1513, %r878;
	mul.f32 	%r1002, %r1512, %r878;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs161, %rs162}, %r969;
	cvt.f32.bf16 	%r1003, %rs162;
	cvt.f32.bf16 	%r1004, %rs161;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1005, %r1002, %r597, %r1004;
	fma.rn.f32 	%r1006, %r1001, %r598, %r1003;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1007, %r1417, %r872;
	mul.f32 	%r1008, %r1416, %r872;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs163, %rs164}, %r919;
	cvt.f32.bf16 	%r1009, %rs164;
	cvt.f32.bf16 	%r1010, %rs163;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1011, %r1008, %r597, %r1010;
	fma.rn.f32 	%r1012, %r1007, %r598, %r1009;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r682, %r1012, %r1011;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1013, %r1419, %r873;
	mul.f32 	%r1014, %r1418, %r873;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs165, %rs166}, %r920;
	cvt.f32.bf16 	%r1015, %rs166;
	cvt.f32.bf16 	%r1016, %rs165;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1017, %r1014, %r597, %r1016;
	fma.rn.f32 	%r1018, %r1013, %r598, %r1015;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r687, %r1018, %r1017;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1019, %r1449, %r874;
	mul.f32 	%r1020, %r1448, %r874;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs167, %rs168}, %r937;
	cvt.f32.bf16 	%r1021, %rs168;
	cvt.f32.bf16 	%r1022, %rs167;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1023, %r1020, %r597, %r1022;
	fma.rn.f32 	%r1024, %r1019, %r598, %r1021;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r683, %r1024, %r1023;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1025, %r1451, %r875;
	mul.f32 	%r1026, %r1450, %r875;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs169, %rs170}, %r938;
	cvt.f32.bf16 	%r1027, %rs170;
	cvt.f32.bf16 	%r1028, %rs169;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1029, %r1026, %r597, %r1028;
	fma.rn.f32 	%r1030, %r1025, %r598, %r1027;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r688, %r1030, %r1029;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1031, %r1481, %r876;
	mul.f32 	%r1032, %r1480, %r876;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs171, %rs172}, %r953;
	cvt.f32.bf16 	%r1033, %rs172;
	cvt.f32.bf16 	%r1034, %rs171;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1035, %r1032, %r597, %r1034;
	fma.rn.f32 	%r1036, %r1031, %r598, %r1033;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r684, %r1036, %r1035;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1037, %r1483, %r877;
	mul.f32 	%r1038, %r1482, %r877;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs173, %rs174}, %r954;
	cvt.f32.bf16 	%r1039, %rs174;
	cvt.f32.bf16 	%r1040, %rs173;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1041, %r1038, %r597, %r1040;
	fma.rn.f32 	%r1042, %r1037, %r598, %r1039;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r689, %r1042, %r1041;
	cvt.rn.bf16x2.f32 	%r685, %r1006, %r1005;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1043, %r1515, %r879;
	mul.f32 	%r1044, %r1514, %r879;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs175, %rs176}, %r970;
	cvt.f32.bf16 	%r1045, %rs176;
	cvt.f32.bf16 	%r1046, %rs175;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1047, %r1044, %r597, %r1046;
	fma.rn.f32 	%r1048, %r1043, %r598, %r1045;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r690, %r1048, %r1047;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1049, %r1517, %r878;
	mul.f32 	%r1050, %r1516, %r878;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs177, %rs178}, %r971;
	cvt.f32.bf16 	%r1051, %rs178;
	cvt.f32.bf16 	%r1052, %rs177;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1053, %r1050, %r599, %r1052;
	fma.rn.f32 	%r1054, %r1049, %r600, %r1051;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1055, %r1421, %r872;
	mul.f32 	%r1056, %r1420, %r872;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs179, %rs180}, %r921;
	cvt.f32.bf16 	%r1057, %rs180;
	cvt.f32.bf16 	%r1058, %rs179;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1059, %r1056, %r599, %r1058;
	fma.rn.f32 	%r1060, %r1055, %r600, %r1057;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r702, %r1060, %r1059;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1061, %r1423, %r873;
	mul.f32 	%r1062, %r1422, %r873;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs181, %rs182}, %r922;
	cvt.f32.bf16 	%r1063, %rs182;
	cvt.f32.bf16 	%r1064, %rs181;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1065, %r1062, %r599, %r1064;
	fma.rn.f32 	%r1066, %r1061, %r600, %r1063;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r707, %r1066, %r1065;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1067, %r1453, %r874;
	mul.f32 	%r1068, %r1452, %r874;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs183, %rs184}, %r939;
	cvt.f32.bf16 	%r1069, %rs184;
	cvt.f32.bf16 	%r1070, %rs183;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1071, %r1068, %r599, %r1070;
	fma.rn.f32 	%r1072, %r1067, %r600, %r1069;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r703, %r1072, %r1071;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1073, %r1455, %r875;
	mul.f32 	%r1074, %r1454, %r875;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs185, %rs186}, %r940;
	cvt.f32.bf16 	%r1075, %rs186;
	cvt.f32.bf16 	%r1076, %rs185;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1077, %r1074, %r599, %r1076;
	fma.rn.f32 	%r1078, %r1073, %r600, %r1075;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r708, %r1078, %r1077;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1079, %r1485, %r876;
	mul.f32 	%r1080, %r1484, %r876;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs187, %rs188}, %r955;
	cvt.f32.bf16 	%r1081, %rs188;
	cvt.f32.bf16 	%r1082, %rs187;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1083, %r1080, %r599, %r1082;
	fma.rn.f32 	%r1084, %r1079, %r600, %r1081;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r704, %r1084, %r1083;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1085, %r1487, %r877;
	mul.f32 	%r1086, %r1486, %r877;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs189, %rs190}, %r956;
	cvt.f32.bf16 	%r1087, %rs190;
	cvt.f32.bf16 	%r1088, %rs189;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1089, %r1086, %r599, %r1088;
	fma.rn.f32 	%r1090, %r1085, %r600, %r1087;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r709, %r1090, %r1089;
	cvt.rn.bf16x2.f32 	%r705, %r1054, %r1053;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1091, %r1519, %r879;
	mul.f32 	%r1092, %r1518, %r879;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs191, %rs192}, %r972;
	cvt.f32.bf16 	%r1093, %rs192;
	cvt.f32.bf16 	%r1094, %rs191;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1095, %r1092, %r599, %r1094;
	fma.rn.f32 	%r1096, %r1091, %r600, %r1093;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r710, %r1096, %r1095;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1097, %r1521, %r878;
	mul.f32 	%r1098, %r1520, %r878;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs193, %rs194}, %r977;
	cvt.f32.bf16 	%r1099, %rs194;
	cvt.f32.bf16 	%r1100, %rs193;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1101, %r1098, %r601, %r1100;
	fma.rn.f32 	%r1102, %r1097, %r602, %r1099;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1103, %r1425, %r872;
	mul.f32 	%r1104, %r1424, %r872;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs195, %rs196}, %r929;
	cvt.f32.bf16 	%r1105, %rs196;
	cvt.f32.bf16 	%r1106, %rs195;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1107, %r1104, %r601, %r1106;
	fma.rn.f32 	%r1108, %r1103, %r602, %r1105;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r722, %r1108, %r1107;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1109, %r1427, %r873;
	mul.f32 	%r1110, %r1426, %r873;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs197, %rs198}, %r930;
	cvt.f32.bf16 	%r1111, %rs198;
	cvt.f32.bf16 	%r1112, %rs197;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1113, %r1110, %r601, %r1112;
	fma.rn.f32 	%r1114, %r1109, %r602, %r1111;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r727, %r1114, %r1113;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1115, %r1457, %r874;
	mul.f32 	%r1116, %r1456, %r874;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs199, %rs200}, %r945;
	cvt.f32.bf16 	%r1117, %rs200;
	cvt.f32.bf16 	%r1118, %rs199;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1119, %r1116, %r601, %r1118;
	fma.rn.f32 	%r1120, %r1115, %r602, %r1117;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r723, %r1120, %r1119;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1121, %r1459, %r875;
	mul.f32 	%r1122, %r1458, %r875;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs201, %rs202}, %r946;
	cvt.f32.bf16 	%r1123, %rs202;
	cvt.f32.bf16 	%r1124, %rs201;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1125, %r1122, %r601, %r1124;
	fma.rn.f32 	%r1126, %r1121, %r602, %r1123;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r728, %r1126, %r1125;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1127, %r1489, %r876;
	mul.f32 	%r1128, %r1488, %r876;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs203, %rs204}, %r961;
	cvt.f32.bf16 	%r1129, %rs204;
	cvt.f32.bf16 	%r1130, %rs203;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1131, %r1128, %r601, %r1130;
	fma.rn.f32 	%r1132, %r1127, %r602, %r1129;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r724, %r1132, %r1131;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1133, %r1491, %r877;
	mul.f32 	%r1134, %r1490, %r877;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs205, %rs206}, %r962;
	cvt.f32.bf16 	%r1135, %rs206;
	cvt.f32.bf16 	%r1136, %rs205;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1137, %r1134, %r601, %r1136;
	fma.rn.f32 	%r1138, %r1133, %r602, %r1135;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r729, %r1138, %r1137;
	cvt.rn.bf16x2.f32 	%r725, %r1102, %r1101;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1139, %r1523, %r879;
	mul.f32 	%r1140, %r1522, %r879;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs207, %rs208}, %r978;
	cvt.f32.bf16 	%r1141, %rs208;
	cvt.f32.bf16 	%r1142, %rs207;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1143, %r1140, %r601, %r1142;
	fma.rn.f32 	%r1144, %r1139, %r602, %r1141;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r730, %r1144, %r1143;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1145, %r1525, %r878;
	mul.f32 	%r1146, %r1524, %r878;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs209, %rs210}, %r979;
	cvt.f32.bf16 	%r1147, %rs210;
	cvt.f32.bf16 	%r1148, %rs209;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1149, %r1146, %r603, %r1148;
	fma.rn.f32 	%r1150, %r1145, %r604, %r1147;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1151, %r1429, %r872;
	mul.f32 	%r1152, %r1428, %r872;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs211, %rs212}, %r931;
	cvt.f32.bf16 	%r1153, %rs212;
	cvt.f32.bf16 	%r1154, %rs211;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1155, %r1152, %r603, %r1154;
	fma.rn.f32 	%r1156, %r1151, %r604, %r1153;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r742, %r1156, %r1155;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1157, %r1431, %r873;
	mul.f32 	%r1158, %r1430, %r873;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs213, %rs214}, %r932;
	cvt.f32.bf16 	%r1159, %rs214;
	cvt.f32.bf16 	%r1160, %rs213;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1161, %r1158, %r603, %r1160;
	fma.rn.f32 	%r1162, %r1157, %r604, %r1159;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r747, %r1162, %r1161;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1163, %r1461, %r874;
	mul.f32 	%r1164, %r1460, %r874;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs215, %rs216}, %r947;
	cvt.f32.bf16 	%r1165, %rs216;
	cvt.f32.bf16 	%r1166, %rs215;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1167, %r1164, %r603, %r1166;
	fma.rn.f32 	%r1168, %r1163, %r604, %r1165;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r743, %r1168, %r1167;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1169, %r1463, %r875;
	mul.f32 	%r1170, %r1462, %r875;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs217, %rs218}, %r948;
	cvt.f32.bf16 	%r1171, %rs218;
	cvt.f32.bf16 	%r1172, %rs217;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1173, %r1170, %r603, %r1172;
	fma.rn.f32 	%r1174, %r1169, %r604, %r1171;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r748, %r1174, %r1173;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1175, %r1493, %r876;
	mul.f32 	%r1176, %r1492, %r876;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs219, %rs220}, %r963;
	cvt.f32.bf16 	%r1177, %rs220;
	cvt.f32.bf16 	%r1178, %rs219;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1179, %r1176, %r603, %r1178;
	fma.rn.f32 	%r1180, %r1175, %r604, %r1177;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r744, %r1180, %r1179;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1181, %r1495, %r877;
	mul.f32 	%r1182, %r1494, %r877;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs221, %rs222}, %r964;
	cvt.f32.bf16 	%r1183, %rs222;
	cvt.f32.bf16 	%r1184, %rs221;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1185, %r1182, %r603, %r1184;
	fma.rn.f32 	%r1186, %r1181, %r604, %r1183;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r749, %r1186, %r1185;
	cvt.rn.bf16x2.f32 	%r745, %r1150, %r1149;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1187, %r1527, %r879;
	mul.f32 	%r1188, %r1526, %r879;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs223, %rs224}, %r980;
	cvt.f32.bf16 	%r1189, %rs224;
	cvt.f32.bf16 	%r1190, %rs223;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1191, %r1188, %r603, %r1190;
	fma.rn.f32 	%r1192, %r1187, %r604, %r1189;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r750, %r1192, %r1191;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1193, %r1529, %r878;
	mul.f32 	%r1194, %r1528, %r878;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs225, %rs226}, %r973;
	cvt.f32.bf16 	%r1195, %rs226;
	cvt.f32.bf16 	%r1196, %rs225;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1197, %r1194, %r605, %r1196;
	fma.rn.f32 	%r1198, %r1193, %r606, %r1195;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1199, %r1433, %r872;
	mul.f32 	%r1200, %r1432, %r872;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs227, %rs228}, %r923;
	cvt.f32.bf16 	%r1201, %rs228;
	cvt.f32.bf16 	%r1202, %rs227;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1203, %r1200, %r605, %r1202;
	fma.rn.f32 	%r1204, %r1199, %r606, %r1201;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r692, %r1204, %r1203;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1205, %r1435, %r873;
	mul.f32 	%r1206, %r1434, %r873;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs229, %rs230}, %r924;
	cvt.f32.bf16 	%r1207, %rs230;
	cvt.f32.bf16 	%r1208, %rs229;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1209, %r1206, %r605, %r1208;
	fma.rn.f32 	%r1210, %r1205, %r606, %r1207;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r697, %r1210, %r1209;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1211, %r1465, %r874;
	mul.f32 	%r1212, %r1464, %r874;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs231, %rs232}, %r941;
	cvt.f32.bf16 	%r1213, %rs232;
	cvt.f32.bf16 	%r1214, %rs231;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1215, %r1212, %r605, %r1214;
	fma.rn.f32 	%r1216, %r1211, %r606, %r1213;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r693, %r1216, %r1215;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1217, %r1467, %r875;
	mul.f32 	%r1218, %r1466, %r875;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs233, %rs234}, %r942;
	cvt.f32.bf16 	%r1219, %rs234;
	cvt.f32.bf16 	%r1220, %rs233;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1221, %r1218, %r605, %r1220;
	fma.rn.f32 	%r1222, %r1217, %r606, %r1219;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r698, %r1222, %r1221;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1223, %r1497, %r876;
	mul.f32 	%r1224, %r1496, %r876;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs235, %rs236}, %r957;
	cvt.f32.bf16 	%r1225, %rs236;
	cvt.f32.bf16 	%r1226, %rs235;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1227, %r1224, %r605, %r1226;
	fma.rn.f32 	%r1228, %r1223, %r606, %r1225;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r694, %r1228, %r1227;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1229, %r1499, %r877;
	mul.f32 	%r1230, %r1498, %r877;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs237, %rs238}, %r958;
	cvt.f32.bf16 	%r1231, %rs238;
	cvt.f32.bf16 	%r1232, %rs237;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1233, %r1230, %r605, %r1232;
	fma.rn.f32 	%r1234, %r1229, %r606, %r1231;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r699, %r1234, %r1233;
	cvt.rn.bf16x2.f32 	%r695, %r1198, %r1197;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1235, %r1531, %r879;
	mul.f32 	%r1236, %r1530, %r879;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs239, %rs240}, %r974;
	cvt.f32.bf16 	%r1237, %rs240;
	cvt.f32.bf16 	%r1238, %rs239;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1239, %r1236, %r605, %r1238;
	fma.rn.f32 	%r1240, %r1235, %r606, %r1237;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r700, %r1240, %r1239;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1241, %r1533, %r878;
	mul.f32 	%r1242, %r1532, %r878;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs241, %rs242}, %r975;
	cvt.f32.bf16 	%r1243, %rs242;
	cvt.f32.bf16 	%r1244, %rs241;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1245, %r1242, %r607, %r1244;
	fma.rn.f32 	%r1246, %r1241, %r608, %r1243;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1247, %r1437, %r872;
	mul.f32 	%r1248, %r1436, %r872;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs243, %rs244}, %r925;
	cvt.f32.bf16 	%r1249, %rs244;
	cvt.f32.bf16 	%r1250, %rs243;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1251, %r1248, %r607, %r1250;
	fma.rn.f32 	%r1252, %r1247, %r608, %r1249;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r712, %r1252, %r1251;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1253, %r1439, %r873;
	mul.f32 	%r1254, %r1438, %r873;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs245, %rs246}, %r926;
	cvt.f32.bf16 	%r1255, %rs246;
	cvt.f32.bf16 	%r1256, %rs245;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1257, %r1254, %r607, %r1256;
	fma.rn.f32 	%r1258, %r1253, %r608, %r1255;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r717, %r1258, %r1257;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1259, %r1469, %r874;
	mul.f32 	%r1260, %r1468, %r874;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs247, %rs248}, %r943;
	cvt.f32.bf16 	%r1261, %rs248;
	cvt.f32.bf16 	%r1262, %rs247;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1263, %r1260, %r607, %r1262;
	fma.rn.f32 	%r1264, %r1259, %r608, %r1261;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r713, %r1264, %r1263;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1265, %r1471, %r875;
	mul.f32 	%r1266, %r1470, %r875;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs249, %rs250}, %r944;
	cvt.f32.bf16 	%r1267, %rs250;
	cvt.f32.bf16 	%r1268, %rs249;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1269, %r1266, %r607, %r1268;
	fma.rn.f32 	%r1270, %r1265, %r608, %r1267;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r718, %r1270, %r1269;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1271, %r1501, %r876;
	mul.f32 	%r1272, %r1500, %r876;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs251, %rs252}, %r959;
	cvt.f32.bf16 	%r1273, %rs252;
	cvt.f32.bf16 	%r1274, %rs251;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1275, %r1272, %r607, %r1274;
	fma.rn.f32 	%r1276, %r1271, %r608, %r1273;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r714, %r1276, %r1275;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1277, %r1503, %r877;
	mul.f32 	%r1278, %r1502, %r877;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs253, %rs254}, %r960;
	cvt.f32.bf16 	%r1279, %rs254;
	cvt.f32.bf16 	%r1280, %rs253;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1281, %r1278, %r607, %r1280;
	fma.rn.f32 	%r1282, %r1277, %r608, %r1279;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r719, %r1282, %r1281;
	cvt.rn.bf16x2.f32 	%r715, %r1246, %r1245;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1283, %r1535, %r879;
	mul.f32 	%r1284, %r1534, %r879;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs255, %rs256}, %r976;
	cvt.f32.bf16 	%r1285, %rs256;
	cvt.f32.bf16 	%r1286, %rs255;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1287, %r1284, %r607, %r1286;
	fma.rn.f32 	%r1288, %r1283, %r608, %r1285;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r720, %r1288, %r1287;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1289, %r1537, %r878;
	mul.f32 	%r1290, %r1536, %r878;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs257, %rs258}, %r981;
	cvt.f32.bf16 	%r1291, %rs258;
	cvt.f32.bf16 	%r1292, %rs257;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1293, %r1290, %r609, %r1292;
	fma.rn.f32 	%r1294, %r1289, %r610, %r1291;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1295, %r1441, %r872;
	mul.f32 	%r1296, %r1440, %r872;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs259, %rs260}, %r933;
	cvt.f32.bf16 	%r1297, %rs260;
	cvt.f32.bf16 	%r1298, %rs259;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1299, %r1296, %r609, %r1298;
	fma.rn.f32 	%r1300, %r1295, %r610, %r1297;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r732, %r1300, %r1299;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1301, %r1443, %r873;
	mul.f32 	%r1302, %r1442, %r873;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs261, %rs262}, %r934;
	cvt.f32.bf16 	%r1303, %rs262;
	cvt.f32.bf16 	%r1304, %rs261;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1305, %r1302, %r609, %r1304;
	fma.rn.f32 	%r1306, %r1301, %r610, %r1303;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r737, %r1306, %r1305;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1307, %r1473, %r874;
	mul.f32 	%r1308, %r1472, %r874;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs263, %rs264}, %r949;
	cvt.f32.bf16 	%r1309, %rs264;
	cvt.f32.bf16 	%r1310, %rs263;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1311, %r1308, %r609, %r1310;
	fma.rn.f32 	%r1312, %r1307, %r610, %r1309;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r733, %r1312, %r1311;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1313, %r1475, %r875;
	mul.f32 	%r1314, %r1474, %r875;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs265, %rs266}, %r950;
	cvt.f32.bf16 	%r1315, %rs266;
	cvt.f32.bf16 	%r1316, %rs265;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1317, %r1314, %r609, %r1316;
	fma.rn.f32 	%r1318, %r1313, %r610, %r1315;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r738, %r1318, %r1317;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1319, %r1505, %r876;
	mul.f32 	%r1320, %r1504, %r876;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs267, %rs268}, %r965;
	cvt.f32.bf16 	%r1321, %rs268;
	cvt.f32.bf16 	%r1322, %rs267;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1323, %r1320, %r609, %r1322;
	fma.rn.f32 	%r1324, %r1319, %r610, %r1321;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r734, %r1324, %r1323;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1325, %r1507, %r877;
	mul.f32 	%r1326, %r1506, %r877;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs269, %rs270}, %r966;
	cvt.f32.bf16 	%r1327, %rs270;
	cvt.f32.bf16 	%r1328, %rs269;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1329, %r1326, %r609, %r1328;
	fma.rn.f32 	%r1330, %r1325, %r610, %r1327;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r739, %r1330, %r1329;
	cvt.rn.bf16x2.f32 	%r735, %r1294, %r1293;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1331, %r1539, %r879;
	mul.f32 	%r1332, %r1538, %r879;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs271, %rs272}, %r982;
	cvt.f32.bf16 	%r1333, %rs272;
	cvt.f32.bf16 	%r1334, %rs271;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1335, %r1332, %r609, %r1334;
	fma.rn.f32 	%r1336, %r1331, %r610, %r1333;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r740, %r1336, %r1335;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1337, %r1541, %r878;
	mul.f32 	%r1338, %r1540, %r878;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs273, %rs274}, %r983;
	cvt.f32.bf16 	%r1339, %rs274;
	cvt.f32.bf16 	%r1340, %rs273;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1341, %r1338, %r611, %r1340;
	fma.rn.f32 	%r1342, %r1337, %r612, %r1339;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1343, %r1445, %r872;
	mul.f32 	%r1344, %r1444, %r872;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs275, %rs276}, %r935;
	cvt.f32.bf16 	%r1345, %rs276;
	cvt.f32.bf16 	%r1346, %rs275;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1347, %r1344, %r611, %r1346;
	fma.rn.f32 	%r1348, %r1343, %r612, %r1345;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r752, %r1348, %r1347;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1349, %r1447, %r873;
	mul.f32 	%r1350, %r1446, %r873;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs277, %rs278}, %r936;
	cvt.f32.bf16 	%r1351, %rs278;
	cvt.f32.bf16 	%r1352, %rs277;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1353, %r1350, %r611, %r1352;
	fma.rn.f32 	%r1354, %r1349, %r612, %r1351;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r757, %r1354, %r1353;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1355, %r1477, %r874;
	mul.f32 	%r1356, %r1476, %r874;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs279, %rs280}, %r951;
	cvt.f32.bf16 	%r1357, %rs280;
	cvt.f32.bf16 	%r1358, %rs279;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1359, %r1356, %r611, %r1358;
	fma.rn.f32 	%r1360, %r1355, %r612, %r1357;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r753, %r1360, %r1359;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1361, %r1479, %r875;
	mul.f32 	%r1362, %r1478, %r875;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs281, %rs282}, %r952;
	cvt.f32.bf16 	%r1363, %rs282;
	cvt.f32.bf16 	%r1364, %rs281;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1365, %r1362, %r611, %r1364;
	fma.rn.f32 	%r1366, %r1361, %r612, %r1363;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r758, %r1366, %r1365;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1367, %r1509, %r876;
	mul.f32 	%r1368, %r1508, %r876;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs283, %rs284}, %r967;
	cvt.f32.bf16 	%r1369, %rs284;
	cvt.f32.bf16 	%r1370, %rs283;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1371, %r1368, %r611, %r1370;
	fma.rn.f32 	%r1372, %r1367, %r612, %r1369;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r754, %r1372, %r1371;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1373, %r1511, %r877;
	mul.f32 	%r1374, %r1510, %r877;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs285, %rs286}, %r968;
	cvt.f32.bf16 	%r1375, %rs286;
	cvt.f32.bf16 	%r1376, %rs285;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1377, %r1374, %r611, %r1376;
	fma.rn.f32 	%r1378, %r1373, %r612, %r1375;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r759, %r1378, %r1377;
	cvt.rn.bf16x2.f32 	%r755, %r1342, %r1341;
	.loc	1 164 16                        // sk07_lm_head.py:164:16
	mul.f32 	%r1379, %r1543, %r879;
	mul.f32 	%r1380, %r1542, %r879;
	.loc	1 166 97                        // sk07_lm_head.py:166:97
	mov.b32 	{%rs287, %rs288}, %r984;
	cvt.f32.bf16 	%r1381, %rs288;
	cvt.f32.bf16 	%r1382, %rs287;
	.loc	1 166 11                        // sk07_lm_head.py:166:11
	fma.rn.f32 	%r1383, %r1380, %r611, %r1382;
	fma.rn.f32 	%r1384, %r1379, %r612, %r1381;
	.loc	1 172 15                        // sk07_lm_head.py:172:15
	cvt.rn.bf16x2.f32 	%r760, %r1384, %r1383;
	bar.sync 	0;
	shl.b32 	%r1385, %r4, 14;
	shl.b32 	%r1386, %r4, 5;
	and.b32 	%r1387, %r1414, 3456;
	bfe.s32 	%r1388, %r2, 2, 1;
	and.b32 	%r1389, %r1388, 8208;
	or.b32 	%r1390, %r1386, %r1387;
	xor.b32 	%r1391, %r1389, %r913;
	or.b32 	%r1392, %r1391, %r1390;
	or.b32 	%r1393, %r1392, %r1385;
	add.s32 	%r681, %r209, %r1393;
	// begin inline asm
	st.shared.v4.b32 [ %r681 + 0 ], { %r682, %r683, %r684, %r685 };
	// end inline asm
	add.s32 	%r686, %r681, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r686 + 0 ], { %r687, %r688, %r689, %r690 };
	// end inline asm
	add.s32 	%r691, %r681, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r691 + 0 ], { %r692, %r693, %r694, %r695 };
	// end inline asm
	add.s32 	%r696, %r681, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r696 + 0 ], { %r697, %r698, %r699, %r700 };
	// end inline asm
	xor.b32 	%r1394, %r1393, 32;
	add.s32 	%r701, %r209, %r1394;
	// begin inline asm
	st.shared.v4.b32 [ %r701 + 0 ], { %r702, %r703, %r704, %r705 };
	// end inline asm
	add.s32 	%r706, %r701, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r706 + 0 ], { %r707, %r708, %r709, %r710 };
	// end inline asm
	add.s32 	%r711, %r701, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r711 + 0 ], { %r712, %r713, %r714, %r715 };
	// end inline asm
	add.s32 	%r716, %r701, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r716 + 0 ], { %r717, %r718, %r719, %r720 };
	// end inline asm
	xor.b32 	%r1395, %r1393, 64;
	add.s32 	%r721, %r209, %r1395;
	// begin inline asm
	st.shared.v4.b32 [ %r721 + 0 ], { %r722, %r723, %r724, %r725 };
	// end inline asm
	add.s32 	%r726, %r721, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r726 + 0 ], { %r727, %r728, %r729, %r730 };
	// end inline asm
	add.s32 	%r731, %r721, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r731 + 0 ], { %r732, %r733, %r734, %r735 };
	// end inline asm
	add.s32 	%r736, %r721, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r736 + 0 ], { %r737, %r738, %r739, %r740 };
	// end inline asm
	xor.b32 	%r1396, %r1393, 96;
	add.s32 	%r741, %r209, %r1396;
	// begin inline asm
	st.shared.v4.b32 [ %r741 + 0 ], { %r742, %r743, %r744, %r745 };
	// end inline asm
	add.s32 	%r746, %r741, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r746 + 0 ], { %r747, %r748, %r749, %r750 };
	// end inline asm
	add.s32 	%r751, %r741, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r751 + 0 ], { %r752, %r753, %r754, %r755 };
	// end inline asm
	add.s32 	%r756, %r741, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r756 + 0 ], { %r757, %r758, %r759, %r760 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r1397, %r2, 2;
	and.b32 	%r1398, %r1397, 896;
	shl.b32 	%r1399, %r864, 9;
	selp.b32 	%r1400, 0, 8208, %p24;
	or.b32 	%r1401, %r906, %r1398;
	xor.b32 	%r1402, %r1401, %r1400;
	or.b32 	%r1403, %r1402, %r1399;
	add.s32 	%r1404, %r209, %r1403;
	ld.shared.v4.b32 	{%r761, %r777, %r793, %r809}, [%r1404];
	ld.shared.v4.b32 	{%r765, %r781, %r797, %r813}, [%r1404+1024];
	ld.shared.v4.b32 	{%r769, %r785, %r801, %r817}, [%r1404+2048];
	ld.shared.v4.b32 	{%r773, %r789, %r805, %r821}, [%r1404+3072];
	xor.b32 	%r1405, %r1403, 32;
	add.s32 	%r1406, %r209, %r1405;
	ld.shared.v4.b32 	{%r762, %r778, %r794, %r810}, [%r1406+16384];
	ld.shared.v4.b32 	{%r766, %r782, %r798, %r814}, [%r1406+17408];
	ld.shared.v4.b32 	{%r770, %r786, %r802, %r818}, [%r1406+18432];
	ld.shared.v4.b32 	{%r774, %r790, %r806, %r822}, [%r1406+19456];
	xor.b32 	%r1407, %r1403, 64;
	add.s32 	%r1408, %r209, %r1407;
	ld.shared.v4.b32 	{%r763, %r779, %r795, %r811}, [%r1408+32768];
	ld.shared.v4.b32 	{%r767, %r783, %r799, %r815}, [%r1408+33792];
	ld.shared.v4.b32 	{%r771, %r787, %r803, %r819}, [%r1408+34816];
	ld.shared.v4.b32 	{%r775, %r791, %r807, %r823}, [%r1408+35840];
	xor.b32 	%r1409, %r1403, 96;
	add.s32 	%r1410, %r209, %r1409;
	ld.shared.v4.b32 	{%r764, %r780, %r796, %r812}, [%r1410+49152];
	ld.shared.v4.b32 	{%r768, %r784, %r800, %r816}, [%r1410+50176];
	ld.shared.v4.b32 	{%r772, %r788, %r804, %r820}, [%r1410+51200];
	ld.shared.v4.b32 	{%r776, %r792, %r808, %r824}, [%r1410+52224];
	.loc	1 172 8                         // sk07_lm_head.py:172:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd257 + 0 ], { %r761, %r762, %r763, %r764 };
	// end inline asm
	// begin inline asm
	@%p9 st.global.v4.b32 [ %rd258 + 0 ], { %r765, %r766, %r767, %r768 };
	// end inline asm
	// begin inline asm
	@%p10 st.global.v4.b32 [ %rd259 + 0 ], { %r769, %r770, %r771, %r772 };
	// end inline asm
	// begin inline asm
	@%p11 st.global.v4.b32 [ %rd260 + 0 ], { %r773, %r774, %r775, %r776 };
	// end inline asm
	// begin inline asm
	@%p12 st.global.v4.b32 [ %rd261 + 0 ], { %r777, %r778, %r779, %r780 };
	// end inline asm
	// begin inline asm
	@%p13 st.global.v4.b32 [ %rd262 + 0 ], { %r781, %r782, %r783, %r784 };
	// end inline asm
	// begin inline asm
	@%p14 st.global.v4.b32 [ %rd263 + 0 ], { %r785, %r786, %r787, %r788 };
	// end inline asm
	// begin inline asm
	@%p15 st.global.v4.b32 [ %rd264 + 0 ], { %r789, %r790, %r791, %r792 };
	// end inline asm
	// begin inline asm
	@%p16 st.global.v4.b32 [ %rd265 + 0 ], { %r793, %r794, %r795, %r796 };
	// end inline asm
	// begin inline asm
	@%p17 st.global.v4.b32 [ %rd266 + 0 ], { %r797, %r798, %r799, %r800 };
	// end inline asm
	// begin inline asm
	@%p18 st.global.v4.b32 [ %rd267 + 0 ], { %r801, %r802, %r803, %r804 };
	// end inline asm
	// begin inline asm
	@%p19 st.global.v4.b32 [ %rd268 + 0 ], { %r805, %r806, %r807, %r808 };
	// end inline asm
	// begin inline asm
	@%p20 st.global.v4.b32 [ %rd269 + 0 ], { %r809, %r810, %r811, %r812 };
	// end inline asm
	// begin inline asm
	@%p21 st.global.v4.b32 [ %rd270 + 0 ], { %r813, %r814, %r815, %r816 };
	// end inline asm
	// begin inline asm
	@%p22 st.global.v4.b32 [ %rd271 + 0 ], { %r817, %r818, %r819, %r820 };
	// end inline asm
	// begin inline asm
	@%p23 st.global.v4.b32 [ %rd272 + 0 ], { %r821, %r822, %r823, %r824 };
	// end inline asm
	.loc	1 170 4                         // sk07_lm_head.py:170:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk07_lm_head.py"
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
.b32 159                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x98 DW_TAG_compile_unit
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
.b8 55
.b8 95
.b8 108
.b8 109
.b8 95
.b8 104
.b8 101
.b8 97
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
.b8 2                                   // Abbrev [2] 0x45:0x17 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 55
.b8 95
.b8 108
.b8 109
.b8 95
.b8 104
.b8 101
.b8 97
.b8 100
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x5c:0x46 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 69                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x71:0x18 DW_TAG_inlined_subroutine
.b32 69                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 133                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x89:0x18 DW_TAG_inlined_subroutine
.b32 69                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 134                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_5 = _Nativo(
    "sk07_lm_head/tile256x128x64_shift0_abi17",
    _PTX_5, "_sk07_lm_head_kernel",
    warps=8, shared=73728,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 17, 18, 19],
    horneado={12: 1, 13: 1, 16: 1, 20: 1, 21: 256, 22: 128, 23: 64, 24: 8, 25: 128},
    div16=[9, 10, 11, 14, 15, 17, 18],
)


# (cfg, has_shift) -> variantes candidatas. La eleccion final
# entre ellas la hace _lanzar comparando los valores horneados:
# con residual y sin residual el ABI tiene distinta cantidad de
# params, porque stride_res_n solo se especializa cuando vale 1.
_POR_CFG = {}
_POR_CFG.setdefault(((16, 128, 128, 8, 8, 3), False), []).append(_VAR_0)
_POR_CFG.setdefault(((16, 128, 128, 8, 8, 3), False), []).append(_VAR_1)
_POR_CFG.setdefault(((128, 128, 128, 8, 8, 3), False), []).append(_VAR_2)
_POR_CFG.setdefault(((128, 128, 128, 8, 8, 3), False), []).append(_VAR_3)
_POR_CFG.setdefault(((256, 128, 64, 8, 8, 4), False), []).append(_VAR_4)
_POR_CFG.setdefault(((256, 128, 64, 8, 8, 4), False), []).append(_VAR_5)


def _lanzar(grid, cfg, has_shift, *args):
    """Elige la variante cuyos valores horneados coinciden con estos args."""
    cands = _POR_CFG.get((cfg, bool(has_shift)))
    if not cands:
        raise KeyError("no hay PTX embebido para cfg=%r has_shift=%r" % (cfg, has_shift))
    for v in cands:
        if all(args[p] == val for p, val in v.horneado.items()):
            return v((grid,) if isinstance(grid, int) else grid, *args)
    raise KeyError("no hay variante para cfg=%r has_shift=%r con estos strides" % (cfg, has_shift))


# --- quant _sk07_quant_kernel: 4 variante(s) de PTX embebido ---

_QPTX0_0 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk07_quant_kernel      // -- Begin function _sk07_quant_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk07_quant_kernel
.visible .entry _sk07_quant_kernel(
	.param .u64 .ptr .global .align 1 _sk07_quant_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk07_quant_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk07_quant_kernel_param_2,
	.param .u32 _sk07_quant_kernel_param_3,
	.param .u32 _sk07_quant_kernel_param_4,
	.param .u32 _sk07_quant_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk07_quant_kernel_param_6,
	.param .u64 .ptr .global .align 1 _sk07_quant_kernel_param_7
)
.reqntid 256
{
	.reg .pred 	%p<15>;
	.reg .b16 	%rs<9>;
	.reg .b32 	%r<125>;
	.reg .b64 	%rd<11>;
	.loc	1 109 0                         // sk07_lm_head.py:109:0
$L__func_begin0:
	.loc	1 109 0                         // sk07_lm_head.py:109:0

// %bb.0:
	ld.param.b64 	%rd4, [_sk07_quant_kernel_param_0];
	ld.param.b64 	%rd5, [_sk07_quant_kernel_param_1];
$L__tmp0:
	.loc	1 111 24                        // sk07_lm_head.py:111:24
	mov.u32 	%r14, %ctaid.x;
	ld.param.b64 	%rd6, [_sk07_quant_kernel_param_2];
	.loc	1 112 24                        // sk07_lm_head.py:112:24
	mov.u32 	%r15, %tid.x;
	and.b32 	%r16, %r15, 255;
	ld.param.b32 	%r17, [_sk07_quant_kernel_param_3];
	and.b32 	%r18, %r15, 31;
	ld.param.b32 	%r19, [_sk07_quant_kernel_param_4];
	shr.u32 	%r20, %r15, 5;
	ld.param.b32 	%r21, [_sk07_quant_kernel_param_5];
	shl.b32 	%r22, %r15, 3;
	and.b32 	%r23, %r22, 2040;
	.loc	1 113 18                        // sk07_lm_head.py:113:18
	setp.lt.s32 	%p1, %r23, %r17;
	.loc	1 114 30                        // sk07_lm_head.py:114:30
	mul.lo.s32 	%r24, %r19, %r14;
	.loc	1 114 24                        // sk07_lm_head.py:114:24
	mad.wide.s32 	%rd7, %r24, 2, %rd4;
	.loc	1 114 42                        // sk07_lm_head.py:114:42
	cvt.u64.u32 	%rd8, %r23;
	mad.wide.u32 	%rd1, %r23, 2, %rd7;
	mov.b32 	%r5, 0;
	.loc	1 114 16                        // sk07_lm_head.py:114:16
	// begin inline asm
	mov.u32 %r1, %r5;
	mov.u32 %r2, %r5;
	mov.u32 %r3, %r5;
	mov.u32 %r4, %r5;
	@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
	// end inline asm
$L__tmp1:
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
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
	.loc	1 118 27                        // sk07_lm_head.py:118:27
	mul.lo.s32 	%r30, %r21, %r14;
	.loc	1 118 21                        // sk07_lm_head.py:118:21
	cvt.s64.s32 	%rd9, %r30;
	add.s64 	%rd10, %rd5, %rd9;
	.loc	1 118 39                        // sk07_lm_head.py:118:39
	add.s64 	%rd2, %rd10, %rd8;
	.loc	1 114 73                        // sk07_lm_head.py:114:73
	mov.b32 	{%rs1, %rs2}, %r4;
	cvt.f32.bf16 	%r31, %rs2;
	cvt.f32.bf16 	%r32, %rs1;
	mov.b32 	{%rs3, %rs4}, %r3;
	cvt.f32.bf16 	%r33, %rs4;
	cvt.f32.bf16 	%r34, %rs3;
	.loc	1 115 36                        // sk07_lm_head.py:115:36
	abs.f32 	%r35, %r34;
	abs.f32 	%r36, %r33;
	abs.f32 	%r37, %r32;
	abs.f32 	%r38, %r31;
	.loc	1 114 73                        // sk07_lm_head.py:114:73
	mov.b32 	{%rs5, %rs6}, %r2;
	cvt.f32.bf16 	%r39, %rs6;
	cvt.f32.bf16 	%r40, %rs5;
	mov.b32 	{%rs7, %rs8}, %r1;
	cvt.f32.bf16 	%r41, %rs8;
	cvt.f32.bf16 	%r42, %rs7;
	.loc	1 115 36                        // sk07_lm_head.py:115:36
	abs.f32 	%r43, %r42;
	abs.f32 	%r44, %r41;
	abs.f32 	%r45, %r40;
	abs.f32 	%r46, %r39;
$L__tmp3:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
	max.f32 	%r47, %r43, %r44;
	max.f32 	%r48, %r47, %r45;
	max.f32 	%r49, %r48, %r46;
	max.f32 	%r50, %r49, %r35;
	max.f32 	%r51, %r50, %r36;
	max.f32 	%r52, %r51, %r37;
	max.f32 	%r53, %r52, %r38;
$L__tmp4:
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	shfl.sync.bfly.b32 	%r54, %r53, 16, 31, -1;
$L__tmp5:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
	max.f32 	%r55, %r53, %r54;
$L__tmp6:
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	shfl.sync.bfly.b32 	%r56, %r55, 8, 31, -1;
$L__tmp7:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
	max.f32 	%r57, %r55, %r56;
$L__tmp8:
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	shfl.sync.bfly.b32 	%r58, %r57, 4, 31, -1;
$L__tmp9:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
	max.f32 	%r59, %r57, %r58;
$L__tmp10:
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	shfl.sync.bfly.b32 	%r60, %r59, 2, 31, -1;
$L__tmp11:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
	max.f32 	%r61, %r59, %r60;
$L__tmp12:
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	shfl.sync.bfly.b32 	%r62, %r61, 1, 31, -1;
$L__tmp13:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
	max.f32 	%r7, %r61, %r62;
$L__tmp14:
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	// begin inline asm
	@%p2 st.shared.b32 [ %r6 + 0 ], %r7;
	// end inline asm
	bar.sync 	0;
	// begin inline asm
	@%p3 ld.shared.b32 %r8, [ %r9 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r63, %r8, 4, 31, -1;
$L__tmp15:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
	max.f32 	%r64, %r8, %r63;
$L__tmp16:
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	shfl.sync.bfly.b32 	%r65, %r64, 2, 31, -1;
$L__tmp17:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
	max.f32 	%r66, %r64, %r65;
$L__tmp18:
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	shfl.sync.bfly.b32 	%r67, %r66, 1, 31, -1;
$L__tmp19:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
	max.f32 	%r10, %r66, %r67;
$L__tmp20:
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	// begin inline asm
	@%p4 st.shared.b32 [ %r9 + 0 ], %r10;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r68, [global_smem];
$L__tmp21:
	.loc	1 115 49                        // sk07_lm_head.py:115:49
	max.f32 	%r69, %r68, 0f0DA24260;
	mov.b32 	%r70, 0f42FE0000;
	.loc	1 116 22                        // sk07_lm_head.py:116:22
	div.full.f32 	%r71, %r70, %r69;
	.loc	1 116 14                        // sk07_lm_head.py:116:14
	mul.f32 	%r72, %r71, %r41;
	mul.f32 	%r73, %r71, %r42;
	mul.f32 	%r74, %r71, %r39;
	mul.f32 	%r75, %r71, %r40;
	mul.f32 	%r76, %r71, %r33;
	mul.f32 	%r77, %r71, %r34;
	mul.f32 	%r78, %r71, %r31;
	mul.f32 	%r79, %r71, %r32;
	.loc	1 117 29                        // sk07_lm_head.py:117:29
	.loc	1 117 39                        // sk07_lm_head.py:117:39
	lop3.b32 	%r80, 0x3f000000, %r72, 0x80000000, 0xF8;
	lop3.b32 	%r81, 0x3f000000, %r73, 0x80000000, 0xF8;
	lop3.b32 	%r82, 0x3f000000, %r74, 0x80000000, 0xF8;
	lop3.b32 	%r83, 0x3f000000, %r75, 0x80000000, 0xF8;
	lop3.b32 	%r84, 0x3f000000, %r76, 0x80000000, 0xF8;
	lop3.b32 	%r85, 0x3f000000, %r77, 0x80000000, 0xF8;
	lop3.b32 	%r86, 0x3f000000, %r78, 0x80000000, 0xF8;
	lop3.b32 	%r87, 0x3f000000, %r79, 0x80000000, 0xF8;
	.loc	1 117 14                        // sk07_lm_head.py:117:14
	fma.rn.f32 	%r88, %r71, %r40, %r83;
	fma.rn.f32 	%r89, %r71, %r39, %r82;
	fma.rn.f32 	%r90, %r71, %r42, %r81;
	fma.rn.f32 	%r91, %r71, %r41, %r80;
	fma.rn.f32 	%r92, %r71, %r32, %r87;
	fma.rn.f32 	%r93, %r71, %r31, %r86;
	fma.rn.f32 	%r94, %r71, %r34, %r85;
	fma.rn.f32 	%r95, %r71, %r33, %r84;
	.loc	1 117 49                        // sk07_lm_head.py:117:49
	cvt.rzi.s32.f32 	%r96, %r91;
	cvt.rzi.s32.f32 	%r97, %r90;
	cvt.rzi.s32.f32 	%r98, %r89;
	cvt.rzi.s32.f32 	%r99, %r88;
	cvt.rzi.s32.f32 	%r100, %r95;
	cvt.rzi.s32.f32 	%r101, %r94;
	cvt.rzi.s32.f32 	%r102, %r93;
	cvt.rzi.s32.f32 	%r103, %r92;
	.loc	1 118 70                        // sk07_lm_head.py:118:70
	max.s32 	%r104, %r99, -127;
	max.s32 	%r105, %r98, -127;
	max.s32 	%r106, %r97, -127;
	max.s32 	%r107, %r96, -127;
	max.s32 	%r108, %r103, -127;
	max.s32 	%r109, %r102, -127;
	max.s32 	%r110, %r101, -127;
	max.s32 	%r111, %r100, -127;
	.loc	1 118 77                        // sk07_lm_head.py:118:77
	min.s32 	%r112, %r107, 127;
	min.s32 	%r113, %r106, 127;
	min.s32 	%r114, %r105, 127;
	min.s32 	%r115, %r104, 127;
	min.s32 	%r116, %r111, 127;
	min.s32 	%r117, %r110, 127;
	min.s32 	%r118, %r109, 127;
	min.s32 	%r119, %r108, 127;
	.loc	1 118 85                        // sk07_lm_head.py:118:85
	prmt.b32 	%r120, %r115, %r114, 0x3340U;
	prmt.b32 	%r121, %r113, %r112, 0x3340U;
	prmt.b32 	%r11, %r121, %r120, 0x5410U;
	prmt.b32 	%r122, %r119, %r118, 0x3340U;
	prmt.b32 	%r123, %r117, %r116, 0x3340U;
	prmt.b32 	%r12, %r123, %r122, 0x5410U;
	.loc	1 118 45                        // sk07_lm_head.py:118:45
	// begin inline asm
	@%p1 st.global.v2.b32 [ %rd2 + 0 ], { %r11, %r12 };
	// end inline asm
	.loc	1 119 21                        // sk07_lm_head.py:119:21
	mad.wide.u32 	%rd3, %r14, 4, %rd6;
	.loc	1 119 34                        // sk07_lm_head.py:119:34
	mul.f32 	%r13, %r69, 0f3C010204;
	.loc	1 119 26                        // sk07_lm_head.py:119:26
	or.b32 	%r124, %r18, %r20;
	setp.eq.b32 	%p5, %r124, 0;
	// begin inline asm
	@%p5 st.global.b32 [ %rd3 + 0 ], { %r13 };
	// end inline asm
	.loc	1 119 4                         // sk07_lm_head.py:119:4
	ret;
$L__tmp22:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk07_lm_head.py"
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
.b32 158                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x97 DW_TAG_compile_unit
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
.b8 55
.b8 95
.b8 108
.b8 109
.b8 95
.b8 104
.b8 101
.b8 97
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
.b8 2                                   // Abbrev [2] 0x45:0x15 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 55
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
.b8 3                                   // Abbrev [3] 0x5a:0x47 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 69                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x6f:0x31 DW_TAG_inlined_subroutine
.b32 69                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp21                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 115                                 // DW_AT_call_line
.b8 29                                  // DW_AT_call_column
.b8 5                                   // Abbrev [5] 0x87:0x18 DW_TAG_inlined_subroutine
.b32 69                                 // DW_AT_abstract_origin
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
    "sk07_lm_head/_sk07_quant_kernel/blk2048",
    _QPTX0_0, "_sk07_quant_kernel",
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

	// .globl	_sk07_quant_kernel      // -- Begin function _sk07_quant_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk07_quant_kernel
.visible .entry _sk07_quant_kernel(
	.param .u64 .ptr .global .align 1 _sk07_quant_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk07_quant_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk07_quant_kernel_param_2,
	.param .u32 _sk07_quant_kernel_param_3,
	.param .u32 _sk07_quant_kernel_param_4,
	.param .u32 _sk07_quant_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk07_quant_kernel_param_6,
	.param .u64 .ptr .global .align 1 _sk07_quant_kernel_param_7
)
.reqntid 256
{
	.reg .pred 	%p<23>;
	.reg .b16 	%rs<17>;
	.reg .b32 	%r<207>;
	.reg .b64 	%rd<12>;
	.loc	1 109 0                         // sk07_lm_head.py:109:0
$L__func_begin0:
	.loc	1 109 0                         // sk07_lm_head.py:109:0

// %bb.0:
	ld.param.b64 	%rd5, [_sk07_quant_kernel_param_0];
	ld.param.b64 	%rd6, [_sk07_quant_kernel_param_1];
$L__tmp0:
	.loc	1 111 24                        // sk07_lm_head.py:111:24
	mov.u32 	%r20, %ctaid.x;
	ld.param.b64 	%rd7, [_sk07_quant_kernel_param_2];
	.loc	1 112 24                        // sk07_lm_head.py:112:24
	mov.u32 	%r21, %tid.x;
	and.b32 	%r22, %r21, 255;
	ld.param.b32 	%r23, [_sk07_quant_kernel_param_3];
	and.b32 	%r24, %r21, 31;
	ld.param.b32 	%r25, [_sk07_quant_kernel_param_4];
	shr.u32 	%r26, %r21, 5;
	ld.param.b32 	%r27, [_sk07_quant_kernel_param_5];
	shl.b32 	%r28, %r21, 4;
	and.b32 	%r29, %r28, 4080;
	.loc	1 113 18                        // sk07_lm_head.py:113:18
	setp.lt.s32 	%p1, %r29, %r23;
	.loc	1 114 30                        // sk07_lm_head.py:114:30
	mul.lo.s32 	%r30, %r25, %r20;
	.loc	1 114 24                        // sk07_lm_head.py:114:24
	mad.wide.s32 	%rd8, %r30, 2, %rd5;
	.loc	1 114 42                        // sk07_lm_head.py:114:42
	cvt.u64.u32 	%rd9, %r29;
	mad.wide.u32 	%rd1, %r29, 2, %rd8;
	add.s64 	%rd2, %rd1, 16;
	mov.b32 	%r5, 0;
	.loc	1 114 16                        // sk07_lm_head.py:114:16
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
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
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
	.loc	1 118 27                        // sk07_lm_head.py:118:27
	mul.lo.s32 	%r36, %r27, %r20;
	.loc	1 118 21                        // sk07_lm_head.py:118:21
	cvt.s64.s32 	%rd10, %r36;
	add.s64 	%rd11, %rd6, %rd10;
	.loc	1 118 39                        // sk07_lm_head.py:118:39
	add.s64 	%rd3, %rd11, %rd9;
	.loc	1 114 73                        // sk07_lm_head.py:114:73
	mov.b32 	{%rs1, %rs2}, %r9;
	cvt.f32.bf16 	%r37, %rs2;
	cvt.f32.bf16 	%r38, %rs1;
	mov.b32 	{%rs3, %rs4}, %r8;
	cvt.f32.bf16 	%r39, %rs4;
	cvt.f32.bf16 	%r40, %rs3;
	.loc	1 115 36                        // sk07_lm_head.py:115:36
	abs.f32 	%r41, %r40;
	abs.f32 	%r42, %r39;
	abs.f32 	%r43, %r38;
	abs.f32 	%r44, %r37;
	.loc	1 114 73                        // sk07_lm_head.py:114:73
	mov.b32 	{%rs5, %rs6}, %r7;
	cvt.f32.bf16 	%r45, %rs6;
	cvt.f32.bf16 	%r46, %rs5;
	mov.b32 	{%rs7, %rs8}, %r6;
	cvt.f32.bf16 	%r47, %rs8;
	cvt.f32.bf16 	%r48, %rs7;
	.loc	1 115 36                        // sk07_lm_head.py:115:36
	abs.f32 	%r49, %r48;
	abs.f32 	%r50, %r47;
	abs.f32 	%r51, %r46;
	abs.f32 	%r52, %r45;
	.loc	1 114 73                        // sk07_lm_head.py:114:73
	mov.b32 	{%rs9, %rs10}, %r4;
	cvt.f32.bf16 	%r53, %rs10;
	cvt.f32.bf16 	%r54, %rs9;
	mov.b32 	{%rs11, %rs12}, %r3;
	cvt.f32.bf16 	%r55, %rs12;
	cvt.f32.bf16 	%r56, %rs11;
	.loc	1 115 36                        // sk07_lm_head.py:115:36
	abs.f32 	%r57, %r56;
	abs.f32 	%r58, %r55;
	abs.f32 	%r59, %r54;
	abs.f32 	%r60, %r53;
	.loc	1 114 73                        // sk07_lm_head.py:114:73
	mov.b32 	{%rs13, %rs14}, %r2;
	cvt.f32.bf16 	%r61, %rs14;
	cvt.f32.bf16 	%r62, %rs13;
	mov.b32 	{%rs15, %rs16}, %r1;
	cvt.f32.bf16 	%r63, %rs16;
	cvt.f32.bf16 	%r64, %rs15;
	.loc	1 115 36                        // sk07_lm_head.py:115:36
	abs.f32 	%r65, %r64;
	abs.f32 	%r66, %r63;
	abs.f32 	%r67, %r62;
	abs.f32 	%r68, %r61;
$L__tmp3:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
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
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	shfl.sync.bfly.b32 	%r84, %r83, 16, 31, -1;
$L__tmp5:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
	max.f32 	%r85, %r83, %r84;
$L__tmp6:
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	shfl.sync.bfly.b32 	%r86, %r85, 8, 31, -1;
$L__tmp7:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
	max.f32 	%r87, %r85, %r86;
$L__tmp8:
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	shfl.sync.bfly.b32 	%r88, %r87, 4, 31, -1;
$L__tmp9:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
	max.f32 	%r89, %r87, %r88;
$L__tmp10:
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	shfl.sync.bfly.b32 	%r90, %r89, 2, 31, -1;
$L__tmp11:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
	max.f32 	%r91, %r89, %r90;
$L__tmp12:
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	shfl.sync.bfly.b32 	%r92, %r91, 1, 31, -1;
$L__tmp13:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
	max.f32 	%r11, %r91, %r92;
$L__tmp14:
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	// begin inline asm
	@%p2 st.shared.b32 [ %r10 + 0 ], %r11;
	// end inline asm
	bar.sync 	0;
	// begin inline asm
	@%p3 ld.shared.b32 %r12, [ %r13 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r93, %r12, 4, 31, -1;
$L__tmp15:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
	max.f32 	%r94, %r12, %r93;
$L__tmp16:
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	shfl.sync.bfly.b32 	%r95, %r94, 2, 31, -1;
$L__tmp17:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
	max.f32 	%r96, %r94, %r95;
$L__tmp18:
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	shfl.sync.bfly.b32 	%r97, %r96, 1, 31, -1;
$L__tmp19:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
	max.f32 	%r14, %r96, %r97;
$L__tmp20:
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	// begin inline asm
	@%p4 st.shared.b32 [ %r13 + 0 ], %r14;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r98, [global_smem];
$L__tmp21:
	.loc	1 115 49                        // sk07_lm_head.py:115:49
	max.f32 	%r99, %r98, 0f0DA24260;
	mov.b32 	%r100, 0f42FE0000;
	.loc	1 116 22                        // sk07_lm_head.py:116:22
	div.full.f32 	%r101, %r100, %r99;
	.loc	1 116 14                        // sk07_lm_head.py:116:14
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
	.loc	1 117 29                        // sk07_lm_head.py:117:29
	.loc	1 117 39                        // sk07_lm_head.py:117:39
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
	.loc	1 117 14                        // sk07_lm_head.py:117:14
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
	.loc	1 117 49                        // sk07_lm_head.py:117:49
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
	.loc	1 118 70                        // sk07_lm_head.py:118:70
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
	.loc	1 118 77                        // sk07_lm_head.py:118:77
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
	.loc	1 118 85                        // sk07_lm_head.py:118:85
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
	.loc	1 118 45                        // sk07_lm_head.py:118:45
	// begin inline asm
	@%p1 st.global.v4.b32 [ %rd3 + 0 ], { %r15, %r16, %r17, %r18 };
	// end inline asm
	.loc	1 119 21                        // sk07_lm_head.py:119:21
	mad.wide.u32 	%rd4, %r20, 4, %rd7;
	.loc	1 119 34                        // sk07_lm_head.py:119:34
	mul.f32 	%r19, %r99, 0f3C010204;
	.loc	1 119 26                        // sk07_lm_head.py:119:26
	or.b32 	%r206, %r24, %r26;
	setp.eq.b32 	%p5, %r206, 0;
	// begin inline asm
	@%p5 st.global.b32 [ %rd4 + 0 ], { %r19 };
	// end inline asm
	.loc	1 119 4                         // sk07_lm_head.py:119:4
	ret;
$L__tmp22:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk07_lm_head.py"
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
.b32 158                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x97 DW_TAG_compile_unit
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
.b8 55
.b8 95
.b8 108
.b8 109
.b8 95
.b8 104
.b8 101
.b8 97
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
.b8 2                                   // Abbrev [2] 0x45:0x15 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 55
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
.b8 3                                   // Abbrev [3] 0x5a:0x47 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 69                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x6f:0x31 DW_TAG_inlined_subroutine
.b32 69                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp21                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 115                                 // DW_AT_call_line
.b8 29                                  // DW_AT_call_column
.b8 5                                   // Abbrev [5] 0x87:0x18 DW_TAG_inlined_subroutine
.b32 69                                 // DW_AT_abstract_origin
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
    "sk07_lm_head/_sk07_quant_kernel/blk4096",
    _QPTX0_1, "_sk07_quant_kernel",
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

	// .globl	_sk07_quant_kernel      // -- Begin function _sk07_quant_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk07_quant_kernel
.visible .entry _sk07_quant_kernel(
	.param .u64 .ptr .global .align 1 _sk07_quant_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk07_quant_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk07_quant_kernel_param_2,
	.param .u32 _sk07_quant_kernel_param_3,
	.param .u32 _sk07_quant_kernel_param_4,
	.param .u32 _sk07_quant_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk07_quant_kernel_param_6,
	.param .u64 .ptr .global .align 1 _sk07_quant_kernel_param_7
)
.reqntid 256
{
	.reg .pred 	%p<40>;
	.reg .b16 	%rs<33>;
	.reg .b32 	%r<372>;
	.reg .b64 	%rd<15>;
	.loc	1 109 0                         // sk07_lm_head.py:109:0
$L__func_begin0:
	.loc	1 109 0                         // sk07_lm_head.py:109:0

// %bb.0:
	ld.param.b64 	%rd8, [_sk07_quant_kernel_param_0];
	ld.param.b64 	%rd9, [_sk07_quant_kernel_param_1];
$L__tmp0:
	.loc	1 111 24                        // sk07_lm_head.py:111:24
	mov.u32 	%r32, %ctaid.x;
	ld.param.b64 	%rd10, [_sk07_quant_kernel_param_2];
	.loc	1 112 24                        // sk07_lm_head.py:112:24
	mov.u32 	%r33, %tid.x;
	and.b32 	%r34, %r33, 255;
	ld.param.b32 	%r35, [_sk07_quant_kernel_param_3];
	and.b32 	%r36, %r33, 31;
	ld.param.b32 	%r37, [_sk07_quant_kernel_param_4];
	shr.u32 	%r38, %r33, 5;
	ld.param.b32 	%r39, [_sk07_quant_kernel_param_5];
	shl.b32 	%r40, %r33, 4;
	and.b32 	%r41, %r40, 4080;
	or.b32 	%r42, %r41, 4096;
	.loc	1 113 18                        // sk07_lm_head.py:113:18
	setp.lt.s32 	%p1, %r41, %r35;
	setp.lt.s32 	%p2, %r42, %r35;
	.loc	1 114 30                        // sk07_lm_head.py:114:30
	mul.lo.s32 	%r43, %r37, %r32;
	.loc	1 114 24                        // sk07_lm_head.py:114:24
	mad.wide.s32 	%rd11, %r43, 2, %rd8;
	.loc	1 114 42                        // sk07_lm_head.py:114:42
	cvt.u64.u32 	%rd12, %r41;
	mad.wide.u32 	%rd1, %r41, 2, %rd11;
	add.s64 	%rd2, %rd1, 16;
	add.s64 	%rd3, %rd1, 8192;
	add.s64 	%rd4, %rd1, 8208;
	mov.b32 	%r5, 0;
	.loc	1 114 16                        // sk07_lm_head.py:114:16
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
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
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
	.loc	1 118 27                        // sk07_lm_head.py:118:27
	mul.lo.s32 	%r49, %r39, %r32;
	.loc	1 118 21                        // sk07_lm_head.py:118:21
	cvt.s64.s32 	%rd13, %r49;
	add.s64 	%rd14, %rd9, %rd13;
	.loc	1 118 39                        // sk07_lm_head.py:118:39
	add.s64 	%rd5, %rd14, %rd12;
	add.s64 	%rd6, %rd5, 4096;
	.loc	1 114 73                        // sk07_lm_head.py:114:73
	mov.b32 	{%rs1, %rs2}, %r9;
	cvt.f32.bf16 	%r50, %rs2;
	cvt.f32.bf16 	%r51, %rs1;
	mov.b32 	{%rs3, %rs4}, %r8;
	cvt.f32.bf16 	%r52, %rs4;
	cvt.f32.bf16 	%r53, %rs3;
	.loc	1 115 36                        // sk07_lm_head.py:115:36
	abs.f32 	%r54, %r53;
	abs.f32 	%r55, %r52;
	abs.f32 	%r56, %r51;
	abs.f32 	%r57, %r50;
	.loc	1 114 73                        // sk07_lm_head.py:114:73
	mov.b32 	{%rs5, %rs6}, %r7;
	cvt.f32.bf16 	%r58, %rs6;
	cvt.f32.bf16 	%r59, %rs5;
	mov.b32 	{%rs7, %rs8}, %r6;
	cvt.f32.bf16 	%r60, %rs8;
	cvt.f32.bf16 	%r61, %rs7;
	.loc	1 115 36                        // sk07_lm_head.py:115:36
	abs.f32 	%r62, %r61;
	abs.f32 	%r63, %r60;
	abs.f32 	%r64, %r59;
	abs.f32 	%r65, %r58;
	.loc	1 114 73                        // sk07_lm_head.py:114:73
	mov.b32 	{%rs9, %rs10}, %r4;
	cvt.f32.bf16 	%r66, %rs10;
	cvt.f32.bf16 	%r67, %rs9;
	mov.b32 	{%rs11, %rs12}, %r3;
	cvt.f32.bf16 	%r68, %rs12;
	cvt.f32.bf16 	%r69, %rs11;
	.loc	1 115 36                        // sk07_lm_head.py:115:36
	abs.f32 	%r70, %r69;
	abs.f32 	%r71, %r68;
	abs.f32 	%r72, %r67;
	abs.f32 	%r73, %r66;
	.loc	1 114 73                        // sk07_lm_head.py:114:73
	mov.b32 	{%rs13, %rs14}, %r2;
	cvt.f32.bf16 	%r74, %rs14;
	cvt.f32.bf16 	%r75, %rs13;
	mov.b32 	{%rs15, %rs16}, %r1;
	cvt.f32.bf16 	%r76, %rs16;
	cvt.f32.bf16 	%r77, %rs15;
	.loc	1 115 36                        // sk07_lm_head.py:115:36
	abs.f32 	%r78, %r77;
	abs.f32 	%r79, %r76;
	abs.f32 	%r80, %r75;
	abs.f32 	%r81, %r74;
$L__tmp3:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
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
	.loc	1 114 73                        // sk07_lm_head.py:114:73
	mov.b32 	{%rs17, %rs18}, %r17;
	cvt.f32.bf16 	%r97, %rs18;
	cvt.f32.bf16 	%r98, %rs17;
	mov.b32 	{%rs19, %rs20}, %r16;
	cvt.f32.bf16 	%r99, %rs20;
	cvt.f32.bf16 	%r100, %rs19;
	.loc	1 115 36                        // sk07_lm_head.py:115:36
	abs.f32 	%r101, %r100;
	abs.f32 	%r102, %r99;
	abs.f32 	%r103, %r98;
	abs.f32 	%r104, %r97;
	.loc	1 114 73                        // sk07_lm_head.py:114:73
	mov.b32 	{%rs21, %rs22}, %r15;
	cvt.f32.bf16 	%r105, %rs22;
	cvt.f32.bf16 	%r106, %rs21;
	mov.b32 	{%rs23, %rs24}, %r14;
	cvt.f32.bf16 	%r107, %rs24;
	cvt.f32.bf16 	%r108, %rs23;
	.loc	1 115 36                        // sk07_lm_head.py:115:36
	abs.f32 	%r109, %r108;
	abs.f32 	%r110, %r107;
	abs.f32 	%r111, %r106;
	abs.f32 	%r112, %r105;
	.loc	1 114 73                        // sk07_lm_head.py:114:73
	mov.b32 	{%rs25, %rs26}, %r13;
	cvt.f32.bf16 	%r113, %rs26;
	cvt.f32.bf16 	%r114, %rs25;
	mov.b32 	{%rs27, %rs28}, %r12;
	cvt.f32.bf16 	%r115, %rs28;
	cvt.f32.bf16 	%r116, %rs27;
	.loc	1 115 36                        // sk07_lm_head.py:115:36
	abs.f32 	%r117, %r116;
	abs.f32 	%r118, %r115;
	abs.f32 	%r119, %r114;
	abs.f32 	%r120, %r113;
	.loc	1 114 73                        // sk07_lm_head.py:114:73
	mov.b32 	{%rs29, %rs30}, %r11;
	cvt.f32.bf16 	%r121, %rs30;
	cvt.f32.bf16 	%r122, %rs29;
	mov.b32 	{%rs31, %rs32}, %r10;
	cvt.f32.bf16 	%r123, %rs32;
	cvt.f32.bf16 	%r124, %rs31;
	.loc	1 115 36                        // sk07_lm_head.py:115:36
	abs.f32 	%r125, %r124;
	abs.f32 	%r126, %r123;
	abs.f32 	%r127, %r122;
	abs.f32 	%r128, %r121;
$L__tmp5:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
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
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	shfl.sync.bfly.b32 	%r145, %r144, 16, 31, -1;
$L__tmp7:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
	max.f32 	%r146, %r144, %r145;
$L__tmp8:
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	shfl.sync.bfly.b32 	%r147, %r146, 8, 31, -1;
$L__tmp9:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
	max.f32 	%r148, %r146, %r147;
$L__tmp10:
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	shfl.sync.bfly.b32 	%r149, %r148, 4, 31, -1;
$L__tmp11:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
	max.f32 	%r150, %r148, %r149;
$L__tmp12:
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	shfl.sync.bfly.b32 	%r151, %r150, 2, 31, -1;
$L__tmp13:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
	max.f32 	%r152, %r150, %r151;
$L__tmp14:
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	shfl.sync.bfly.b32 	%r153, %r152, 1, 31, -1;
$L__tmp15:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
	max.f32 	%r19, %r152, %r153;
$L__tmp16:
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	// begin inline asm
	@%p3 st.shared.b32 [ %r18 + 0 ], %r19;
	// end inline asm
	bar.sync 	0;
	// begin inline asm
	@%p4 ld.shared.b32 %r20, [ %r21 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r154, %r20, 4, 31, -1;
$L__tmp17:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
	max.f32 	%r155, %r20, %r154;
$L__tmp18:
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	shfl.sync.bfly.b32 	%r156, %r155, 2, 31, -1;
$L__tmp19:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
	max.f32 	%r157, %r155, %r156;
$L__tmp20:
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	shfl.sync.bfly.b32 	%r158, %r157, 1, 31, -1;
$L__tmp21:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
	max.f32 	%r22, %r157, %r158;
$L__tmp22:
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	// begin inline asm
	@%p5 st.shared.b32 [ %r21 + 0 ], %r22;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r159, [global_smem];
$L__tmp23:
	.loc	1 115 49                        // sk07_lm_head.py:115:49
	max.f32 	%r160, %r159, 0f0DA24260;
	mov.b32 	%r161, 0f42FE0000;
	.loc	1 116 22                        // sk07_lm_head.py:116:22
	div.full.f32 	%r162, %r161, %r160;
	.loc	1 116 14                        // sk07_lm_head.py:116:14
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
	.loc	1 117 29                        // sk07_lm_head.py:117:29
	.loc	1 117 39                        // sk07_lm_head.py:117:39
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
	.loc	1 117 14                        // sk07_lm_head.py:117:14
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
	.loc	1 117 49                        // sk07_lm_head.py:117:49
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
	.loc	1 118 70                        // sk07_lm_head.py:118:70
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
	.loc	1 118 77                        // sk07_lm_head.py:118:77
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
	.loc	1 118 85                        // sk07_lm_head.py:118:85
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
	.loc	1 118 45                        // sk07_lm_head.py:118:45
	// begin inline asm
	@%p1 st.global.v4.b32 [ %rd5 + 0 ], { %r23, %r24, %r25, %r26 };
	// end inline asm
	// begin inline asm
	@%p2 st.global.v4.b32 [ %rd6 + 0 ], { %r27, %r28, %r29, %r30 };
	// end inline asm
	.loc	1 119 21                        // sk07_lm_head.py:119:21
	mad.wide.u32 	%rd7, %r32, 4, %rd10;
	.loc	1 119 34                        // sk07_lm_head.py:119:34
	mul.f32 	%r31, %r160, 0f3C010204;
	.loc	1 119 26                        // sk07_lm_head.py:119:26
	or.b32 	%r371, %r36, %r38;
	setp.eq.b32 	%p6, %r371, 0;
	// begin inline asm
	@%p6 st.global.b32 [ %rd7 + 0 ], { %r31 };
	// end inline asm
	.loc	1 119 4                         // sk07_lm_head.py:119:4
	ret;
$L__tmp24:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk07_lm_head.py"
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
.b32 158                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x97 DW_TAG_compile_unit
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
.b8 55
.b8 95
.b8 108
.b8 109
.b8 95
.b8 104
.b8 101
.b8 97
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
.b8 2                                   // Abbrev [2] 0x45:0x15 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 55
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
.b8 3                                   // Abbrev [3] 0x5a:0x47 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 69                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x6f:0x31 DW_TAG_inlined_subroutine
.b32 69                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp23                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 115                                 // DW_AT_call_line
.b8 29                                  // DW_AT_call_column
.b8 5                                   // Abbrev [5] 0x87:0x18 DW_TAG_inlined_subroutine
.b32 69                                 // DW_AT_abstract_origin
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
    "sk07_lm_head/_sk07_quant_kernel/blk8192",
    _QPTX0_2, "_sk07_quant_kernel",
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

	// .globl	_sk07_quant_kernel      // -- Begin function _sk07_quant_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk07_quant_kernel
.visible .entry _sk07_quant_kernel(
	.param .u64 .ptr .global .align 1 _sk07_quant_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk07_quant_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk07_quant_kernel_param_2,
	.param .u32 _sk07_quant_kernel_param_3,
	.param .u32 _sk07_quant_kernel_param_4,
	.param .u32 _sk07_quant_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk07_quant_kernel_param_6,
	.param .u64 .ptr .global .align 1 _sk07_quant_kernel_param_7
)
.reqntid 256
{
	.reg .pred 	%p<74>;
	.reg .b16 	%rs<65>;
	.reg .b32 	%r<703>;
	.reg .b64 	%rd<22>;
	.loc	1 109 0                         // sk07_lm_head.py:109:0
$L__func_begin0:
	.loc	1 109 0                         // sk07_lm_head.py:109:0

// %bb.0:
	ld.param.b64 	%rd14, [_sk07_quant_kernel_param_0];
	ld.param.b64 	%rd15, [_sk07_quant_kernel_param_1];
$L__tmp0:
	.loc	1 111 24                        // sk07_lm_head.py:111:24
	mov.u32 	%r56, %ctaid.x;
	ld.param.b64 	%rd16, [_sk07_quant_kernel_param_2];
	.loc	1 112 24                        // sk07_lm_head.py:112:24
	mov.u32 	%r57, %tid.x;
	and.b32 	%r58, %r57, 255;
	ld.param.b32 	%r59, [_sk07_quant_kernel_param_3];
	and.b32 	%r60, %r57, 31;
	ld.param.b32 	%r61, [_sk07_quant_kernel_param_4];
	shr.u32 	%r62, %r57, 5;
	ld.param.b32 	%r63, [_sk07_quant_kernel_param_5];
	shl.b32 	%r64, %r57, 4;
	and.b32 	%r65, %r64, 4080;
	or.b32 	%r66, %r65, 4096;
	or.b32 	%r67, %r65, 8192;
	or.b32 	%r68, %r64, 12288;
	or.b32 	%r69, %r64, 12296;
	.loc	1 113 18                        // sk07_lm_head.py:113:18
	setp.lt.s32 	%p1, %r65, %r59;
	setp.lt.s32 	%p2, %r66, %r59;
	setp.lt.s32 	%p3, %r67, %r59;
	setp.lt.s32 	%p4, %r68, %r59;
	.loc	1 114 30                        // sk07_lm_head.py:114:30
	mul.lo.s32 	%r70, %r61, %r56;
	.loc	1 114 24                        // sk07_lm_head.py:114:24
	mad.wide.s32 	%rd17, %r70, 2, %rd14;
	.loc	1 114 42                        // sk07_lm_head.py:114:42
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
	.loc	1 114 16                        // sk07_lm_head.py:114:16
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
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
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
	.loc	1 118 27                        // sk07_lm_head.py:118:27
	mul.lo.s32 	%r76, %r63, %r56;
	.loc	1 118 21                        // sk07_lm_head.py:118:21
	cvt.s64.s32 	%rd20, %r76;
	add.s64 	%rd21, %rd15, %rd20;
	.loc	1 118 39                        // sk07_lm_head.py:118:39
	add.s64 	%rd9, %rd21, %rd18;
	add.s64 	%rd10, %rd9, 4096;
	add.s64 	%rd11, %rd9, 8192;
	add.s64 	%rd12, %rd21, %rd19;
	.loc	1 114 73                        // sk07_lm_head.py:114:73
	mov.b32 	{%rs1, %rs2}, %r9;
	cvt.f32.bf16 	%r77, %rs2;
	cvt.f32.bf16 	%r78, %rs1;
	mov.b32 	{%rs3, %rs4}, %r8;
	cvt.f32.bf16 	%r79, %rs4;
	cvt.f32.bf16 	%r80, %rs3;
	.loc	1 115 36                        // sk07_lm_head.py:115:36
	abs.f32 	%r81, %r80;
	abs.f32 	%r82, %r79;
	abs.f32 	%r83, %r78;
	abs.f32 	%r84, %r77;
	.loc	1 114 73                        // sk07_lm_head.py:114:73
	mov.b32 	{%rs5, %rs6}, %r7;
	cvt.f32.bf16 	%r85, %rs6;
	cvt.f32.bf16 	%r86, %rs5;
	mov.b32 	{%rs7, %rs8}, %r6;
	cvt.f32.bf16 	%r87, %rs8;
	cvt.f32.bf16 	%r88, %rs7;
	.loc	1 115 36                        // sk07_lm_head.py:115:36
	abs.f32 	%r89, %r88;
	abs.f32 	%r90, %r87;
	abs.f32 	%r91, %r86;
	abs.f32 	%r92, %r85;
	.loc	1 114 73                        // sk07_lm_head.py:114:73
	mov.b32 	{%rs9, %rs10}, %r4;
	cvt.f32.bf16 	%r93, %rs10;
	cvt.f32.bf16 	%r94, %rs9;
	mov.b32 	{%rs11, %rs12}, %r3;
	cvt.f32.bf16 	%r95, %rs12;
	cvt.f32.bf16 	%r96, %rs11;
	.loc	1 115 36                        // sk07_lm_head.py:115:36
	abs.f32 	%r97, %r96;
	abs.f32 	%r98, %r95;
	abs.f32 	%r99, %r94;
	abs.f32 	%r100, %r93;
	.loc	1 114 73                        // sk07_lm_head.py:114:73
	mov.b32 	{%rs13, %rs14}, %r2;
	cvt.f32.bf16 	%r101, %rs14;
	cvt.f32.bf16 	%r102, %rs13;
	mov.b32 	{%rs15, %rs16}, %r1;
	cvt.f32.bf16 	%r103, %rs16;
	cvt.f32.bf16 	%r104, %rs15;
	.loc	1 115 36                        // sk07_lm_head.py:115:36
	abs.f32 	%r105, %r104;
	abs.f32 	%r106, %r103;
	abs.f32 	%r107, %r102;
	abs.f32 	%r108, %r101;
$L__tmp3:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
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
	.loc	1 114 73                        // sk07_lm_head.py:114:73
	mov.b32 	{%rs17, %rs18}, %r17;
	cvt.f32.bf16 	%r124, %rs18;
	cvt.f32.bf16 	%r125, %rs17;
	mov.b32 	{%rs19, %rs20}, %r16;
	cvt.f32.bf16 	%r126, %rs20;
	cvt.f32.bf16 	%r127, %rs19;
	.loc	1 115 36                        // sk07_lm_head.py:115:36
	abs.f32 	%r128, %r127;
	abs.f32 	%r129, %r126;
	abs.f32 	%r130, %r125;
	abs.f32 	%r131, %r124;
	.loc	1 114 73                        // sk07_lm_head.py:114:73
	mov.b32 	{%rs21, %rs22}, %r15;
	cvt.f32.bf16 	%r132, %rs22;
	cvt.f32.bf16 	%r133, %rs21;
	mov.b32 	{%rs23, %rs24}, %r14;
	cvt.f32.bf16 	%r134, %rs24;
	cvt.f32.bf16 	%r135, %rs23;
	.loc	1 115 36                        // sk07_lm_head.py:115:36
	abs.f32 	%r136, %r135;
	abs.f32 	%r137, %r134;
	abs.f32 	%r138, %r133;
	abs.f32 	%r139, %r132;
	.loc	1 114 73                        // sk07_lm_head.py:114:73
	mov.b32 	{%rs25, %rs26}, %r13;
	cvt.f32.bf16 	%r140, %rs26;
	cvt.f32.bf16 	%r141, %rs25;
	mov.b32 	{%rs27, %rs28}, %r12;
	cvt.f32.bf16 	%r142, %rs28;
	cvt.f32.bf16 	%r143, %rs27;
	.loc	1 115 36                        // sk07_lm_head.py:115:36
	abs.f32 	%r144, %r143;
	abs.f32 	%r145, %r142;
	abs.f32 	%r146, %r141;
	abs.f32 	%r147, %r140;
	.loc	1 114 73                        // sk07_lm_head.py:114:73
	mov.b32 	{%rs29, %rs30}, %r11;
	cvt.f32.bf16 	%r148, %rs30;
	cvt.f32.bf16 	%r149, %rs29;
	mov.b32 	{%rs31, %rs32}, %r10;
	cvt.f32.bf16 	%r150, %rs32;
	cvt.f32.bf16 	%r151, %rs31;
	.loc	1 115 36                        // sk07_lm_head.py:115:36
	abs.f32 	%r152, %r151;
	abs.f32 	%r153, %r150;
	abs.f32 	%r154, %r149;
	abs.f32 	%r155, %r148;
$L__tmp5:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
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
	.loc	1 114 73                        // sk07_lm_head.py:114:73
	mov.b32 	{%rs33, %rs34}, %r25;
	cvt.f32.bf16 	%r172, %rs34;
	cvt.f32.bf16 	%r173, %rs33;
	mov.b32 	{%rs35, %rs36}, %r24;
	cvt.f32.bf16 	%r174, %rs36;
	cvt.f32.bf16 	%r175, %rs35;
	.loc	1 115 36                        // sk07_lm_head.py:115:36
	abs.f32 	%r176, %r175;
	abs.f32 	%r177, %r174;
	abs.f32 	%r178, %r173;
	abs.f32 	%r179, %r172;
	.loc	1 114 73                        // sk07_lm_head.py:114:73
	mov.b32 	{%rs37, %rs38}, %r23;
	cvt.f32.bf16 	%r180, %rs38;
	cvt.f32.bf16 	%r181, %rs37;
	mov.b32 	{%rs39, %rs40}, %r22;
	cvt.f32.bf16 	%r182, %rs40;
	cvt.f32.bf16 	%r183, %rs39;
	.loc	1 115 36                        // sk07_lm_head.py:115:36
	abs.f32 	%r184, %r183;
	abs.f32 	%r185, %r182;
	abs.f32 	%r186, %r181;
	abs.f32 	%r187, %r180;
	.loc	1 114 73                        // sk07_lm_head.py:114:73
	mov.b32 	{%rs41, %rs42}, %r21;
	cvt.f32.bf16 	%r188, %rs42;
	cvt.f32.bf16 	%r189, %rs41;
	mov.b32 	{%rs43, %rs44}, %r20;
	cvt.f32.bf16 	%r190, %rs44;
	cvt.f32.bf16 	%r191, %rs43;
	.loc	1 115 36                        // sk07_lm_head.py:115:36
	abs.f32 	%r192, %r191;
	abs.f32 	%r193, %r190;
	abs.f32 	%r194, %r189;
	abs.f32 	%r195, %r188;
	.loc	1 114 73                        // sk07_lm_head.py:114:73
	mov.b32 	{%rs45, %rs46}, %r19;
	cvt.f32.bf16 	%r196, %rs46;
	cvt.f32.bf16 	%r197, %rs45;
	mov.b32 	{%rs47, %rs48}, %r18;
	cvt.f32.bf16 	%r198, %rs48;
	cvt.f32.bf16 	%r199, %rs47;
	.loc	1 115 36                        // sk07_lm_head.py:115:36
	abs.f32 	%r200, %r199;
	abs.f32 	%r201, %r198;
	abs.f32 	%r202, %r197;
	abs.f32 	%r203, %r196;
$L__tmp7:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
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
	.loc	1 114 73                        // sk07_lm_head.py:114:73
	mov.b32 	{%rs49, %rs50}, %r33;
	cvt.f32.bf16 	%r220, %rs50;
	cvt.f32.bf16 	%r221, %rs49;
	mov.b32 	{%rs51, %rs52}, %r32;
	cvt.f32.bf16 	%r222, %rs52;
	cvt.f32.bf16 	%r223, %rs51;
	.loc	1 115 36                        // sk07_lm_head.py:115:36
	abs.f32 	%r224, %r223;
	abs.f32 	%r225, %r222;
	abs.f32 	%r226, %r221;
	abs.f32 	%r227, %r220;
	.loc	1 114 73                        // sk07_lm_head.py:114:73
	mov.b32 	{%rs53, %rs54}, %r31;
	cvt.f32.bf16 	%r228, %rs54;
	cvt.f32.bf16 	%r229, %rs53;
	mov.b32 	{%rs55, %rs56}, %r30;
	cvt.f32.bf16 	%r230, %rs56;
	cvt.f32.bf16 	%r231, %rs55;
	.loc	1 115 36                        // sk07_lm_head.py:115:36
	abs.f32 	%r232, %r231;
	abs.f32 	%r233, %r230;
	abs.f32 	%r234, %r229;
	abs.f32 	%r235, %r228;
	.loc	1 114 73                        // sk07_lm_head.py:114:73
	mov.b32 	{%rs57, %rs58}, %r29;
	cvt.f32.bf16 	%r236, %rs58;
	cvt.f32.bf16 	%r237, %rs57;
	mov.b32 	{%rs59, %rs60}, %r28;
	cvt.f32.bf16 	%r238, %rs60;
	cvt.f32.bf16 	%r239, %rs59;
	.loc	1 115 36                        // sk07_lm_head.py:115:36
	abs.f32 	%r240, %r239;
	abs.f32 	%r241, %r238;
	abs.f32 	%r242, %r237;
	abs.f32 	%r243, %r236;
	.loc	1 114 73                        // sk07_lm_head.py:114:73
	mov.b32 	{%rs61, %rs62}, %r27;
	cvt.f32.bf16 	%r244, %rs62;
	cvt.f32.bf16 	%r245, %rs61;
	mov.b32 	{%rs63, %rs64}, %r26;
	cvt.f32.bf16 	%r246, %rs64;
	cvt.f32.bf16 	%r247, %rs63;
	.loc	1 115 36                        // sk07_lm_head.py:115:36
	abs.f32 	%r248, %r247;
	abs.f32 	%r249, %r246;
	abs.f32 	%r250, %r245;
	abs.f32 	%r251, %r244;
$L__tmp9:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
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
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	shfl.sync.bfly.b32 	%r268, %r267, 16, 31, -1;
$L__tmp11:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
	max.f32 	%r269, %r267, %r268;
$L__tmp12:
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	shfl.sync.bfly.b32 	%r270, %r269, 8, 31, -1;
$L__tmp13:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
	max.f32 	%r271, %r269, %r270;
$L__tmp14:
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	shfl.sync.bfly.b32 	%r272, %r271, 4, 31, -1;
$L__tmp15:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
	max.f32 	%r273, %r271, %r272;
$L__tmp16:
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	shfl.sync.bfly.b32 	%r274, %r273, 2, 31, -1;
$L__tmp17:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
	max.f32 	%r275, %r273, %r274;
$L__tmp18:
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	shfl.sync.bfly.b32 	%r276, %r275, 1, 31, -1;
$L__tmp19:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
	max.f32 	%r35, %r275, %r276;
$L__tmp20:
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	// begin inline asm
	@%p5 st.shared.b32 [ %r34 + 0 ], %r35;
	// end inline asm
	bar.sync 	0;
	// begin inline asm
	@%p6 ld.shared.b32 %r36, [ %r37 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r277, %r36, 4, 31, -1;
$L__tmp21:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
	max.f32 	%r278, %r36, %r277;
$L__tmp22:
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	shfl.sync.bfly.b32 	%r279, %r278, 2, 31, -1;
$L__tmp23:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
	max.f32 	%r280, %r278, %r279;
$L__tmp24:
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	shfl.sync.bfly.b32 	%r281, %r280, 1, 31, -1;
$L__tmp25:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk07_lm_head.py:115:29 ] ]
	max.f32 	%r38, %r280, %r281;
$L__tmp26:
	.loc	2 191 40                        // standard.py:191:40 @[ sk07_lm_head.py:115:29 ]
	// begin inline asm
	@%p7 st.shared.b32 [ %r37 + 0 ], %r38;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r282, [global_smem];
$L__tmp27:
	.loc	1 115 49                        // sk07_lm_head.py:115:49
	max.f32 	%r283, %r282, 0f0DA24260;
	mov.b32 	%r284, 0f42FE0000;
	.loc	1 116 22                        // sk07_lm_head.py:116:22
	div.full.f32 	%r285, %r284, %r283;
	.loc	1 116 14                        // sk07_lm_head.py:116:14
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
	.loc	1 117 29                        // sk07_lm_head.py:117:29
	.loc	1 117 39                        // sk07_lm_head.py:117:39
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
	.loc	1 117 14                        // sk07_lm_head.py:117:14
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
	.loc	1 117 49                        // sk07_lm_head.py:117:49
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
	.loc	1 118 70                        // sk07_lm_head.py:118:70
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
	.loc	1 118 77                        // sk07_lm_head.py:118:77
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
	.loc	1 118 85                        // sk07_lm_head.py:118:85
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
	.loc	1 118 45                        // sk07_lm_head.py:118:45
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
	.loc	1 119 21                        // sk07_lm_head.py:119:21
	mad.wide.u32 	%rd13, %r56, 4, %rd16;
	.loc	1 119 34                        // sk07_lm_head.py:119:34
	mul.f32 	%r55, %r283, 0f3C010204;
	.loc	1 119 26                        // sk07_lm_head.py:119:26
	or.b32 	%r702, %r60, %r62;
	setp.eq.b32 	%p8, %r702, 0;
	// begin inline asm
	@%p8 st.global.b32 [ %rd13 + 0 ], { %r55 };
	// end inline asm
	.loc	1 119 4                         // sk07_lm_head.py:119:4
	ret;
$L__tmp28:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk07_lm_head.py"
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
.b32 158                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x97 DW_TAG_compile_unit
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
.b8 55
.b8 95
.b8 108
.b8 109
.b8 95
.b8 104
.b8 101
.b8 97
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
.b8 2                                   // Abbrev [2] 0x45:0x15 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 55
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
.b8 3                                   // Abbrev [3] 0x5a:0x47 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 69                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x6f:0x31 DW_TAG_inlined_subroutine
.b32 69                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp27                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 115                                 // DW_AT_call_line
.b8 29                                  // DW_AT_call_column
.b8 5                                   // Abbrev [5] 0x87:0x18 DW_TAG_inlined_subroutine
.b32 69                                 // DW_AT_abstract_origin
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
    "sk07_lm_head/_sk07_quant_kernel/blk16384",
    _QPTX0_3, "_sk07_quant_kernel",
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
        "genesis_sk07_lm_head_q0: no hay PTX embebido para estos constexpr; "
        "horneados disponibles: %r" % [_v.horneado for _v in _POR_Q0])


def _q0_fake(grid: list[int], x_ptr: torch.Tensor, q_ptr: torch.Tensor, s_ptr: torch.Tensor, K: int, stride_xm: int, stride_qm: int, BLOCK: int) -> None:
    """Meta impl: no toca la GPU; las salidas se mutan in-place."""
    return None


# Registro AL IMPORTAR, no en el primer uso: direct_register_custom_op
# llama a torch._library.infer_schema, que dynamo se niega a trazar
# ("Attempted to call function marked as skipped"). El hasattr evita el
# choque cuando el modulo se importa dos veces con nombres distintos,
# como hace el gate de tools/monolitizar.py.
if not hasattr(torch.ops.vllm, "genesis_sk07_lm_head_q0"):
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op(
        op_name="genesis_sk07_lm_head_q0",
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
    return torch.ops.vllm.genesis_sk07_lm_head_q0(g, *args)
