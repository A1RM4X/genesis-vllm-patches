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


def habilitado() -> bool:
    """Kill-switch: ``GENESIS_PTQ_NATIVO=0`` desactiva el camino PTX.

    Definido ACA, no importado: este modulo no comparte plomeria con nadie.
    """
    return os.environ.get("GENESIS_PTQ_NATIVO", "1").strip().lower() \
        not in ("0", "false", "no", "off")




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


_SK09_RMSNORM_QUANT_PTX = r"""//
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
	.loc	1 59 0                          // sk09_norm_embed.py:59:0
$L__func_begin0:
	.loc	1 59 0                          // sk09_norm_embed.py:59:0

// %bb.0:                               // %__nv_rsqrtf.exit
	ld.param.b64 	%rd44, [_sk09_rmsnorm_quant_kernel_param_0];
	ld.param.b64 	%rd45, [_sk09_rmsnorm_quant_kernel_param_1];
$L__tmp0:
	.loc	1 64 24                         // sk09_norm_embed.py:64:24
	mov.u32 	%r84, %ctaid.x;
	ld.param.b64 	%rd46, [_sk09_rmsnorm_quant_kernel_param_2];
	.loc	1 65 24                         // sk09_norm_embed.py:65:24
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
	.loc	1 66 18                         // sk09_norm_embed.py:66:18
	setp.lt.s32 	%p1, %r92, %r89;
	setp.lt.s32 	%p2, %r94, %r89;
	.loc	1 67 30                         // sk09_norm_embed.py:67:30
	mul.lo.s32 	%r96, %r91, %r84;
	.loc	1 67 24                         // sk09_norm_embed.py:67:24
	mad.wide.s32 	%rd49, %r96, 2, %rd44;
	.loc	1 67 42                         // sk09_norm_embed.py:67:42
	cvt.u64.u32 	%rd50, %r92;
	mul.wide.u32 	%rd51, %r92, 2;
	add.s64 	%rd1, %rd49, %rd51;
	add.s64 	%rd2, %rd1, 16;
	add.s64 	%rd3, %rd1, 8192;
	add.s64 	%rd4, %rd1, 8208;
	mov.b32 	%r5, 0;
	.loc	1 67 16                         // sk09_norm_embed.py:67:16
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
	.loc	1 67 73                         // sk09_norm_embed.py:67:73
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
	.loc	1 68 24                         // sk09_norm_embed.py:68:24
	add.s64 	%rd5, %rd45, %rd51;
	add.s64 	%rd6, %rd5, 16;
	add.s64 	%rd7, %rd5, 8192;
	add.s64 	%rd8, %rd5, 8208;
	.loc	1 68 16                         // sk09_norm_embed.py:68:16
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
	.loc	1 69 40                         // sk09_norm_embed.py:69:40
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
	.loc	1 69 33                         // sk09_norm_embed.py:69:33
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
	.loc	1 69 20                         // sk09_norm_embed.py:69:20
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
	.loc	1 70 32                         // sk09_norm_embed.py:70:32
	mul.f32 	%r164, %r100, %r100;
$L__tmp1:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:70:28 ] ]
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
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:70:28 ]
	shfl.sync.bfly.b32 	%r196, %r195, 16, 31, -1;
$L__tmp3:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:70:28 ] ]
	add.f32 	%r197, %r195, %r196;
$L__tmp4:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:70:28 ]
	shfl.sync.bfly.b32 	%r198, %r197, 8, 31, -1;
$L__tmp5:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:70:28 ] ]
	add.f32 	%r199, %r197, %r198;
$L__tmp6:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:70:28 ]
	shfl.sync.bfly.b32 	%r200, %r199, 4, 31, -1;
$L__tmp7:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:70:28 ] ]
	add.f32 	%r201, %r199, %r200;
$L__tmp8:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:70:28 ]
	shfl.sync.bfly.b32 	%r202, %r201, 2, 31, -1;
$L__tmp9:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:70:28 ] ]
	add.f32 	%r203, %r201, %r202;
$L__tmp10:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:70:28 ]
	shfl.sync.bfly.b32 	%r204, %r203, 1, 31, -1;
$L__tmp11:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:70:28 ] ]
	add.f32 	%r68, %r203, %r204;
$L__tmp12:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:70:28 ]
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
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:70:28 ] ]
	add.f32 	%r210, %r69, %r209;
$L__tmp14:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:70:28 ]
	shfl.sync.bfly.b32 	%r211, %r210, 2, 31, -1;
$L__tmp15:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:70:28 ] ]
	add.f32 	%r212, %r210, %r211;
$L__tmp16:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:70:28 ]
	shfl.sync.bfly.b32 	%r213, %r212, 1, 31, -1;
$L__tmp17:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk09_norm_embed.py:70:28 ] ]
	add.f32 	%r71, %r212, %r213;
$L__tmp18:
	.loc	2 293 36                        // standard.py:293:36 @[ sk09_norm_embed.py:70:28 ]
	and.b32 	%r214, %r85, 7;
	setp.eq.b32 	%p7, %r214, 0;
	and.pred 	%p5, %p4, %p7;
	// begin inline asm
	@%p5 st.shared.b32 [ %r70 + 0 ], %r71;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r215, [global_smem];
$L__tmp19:
	.loc	1 70 45                         // sk09_norm_embed.py:70:45
	cvt.rn.f32.s32 	%r216, %r89;
	div.full.f32 	%r217, %r215, %r216;
	.loc	1 70 49                         // sk09_norm_embed.py:70:49
	add.f32 	%r218, %r217, 0f358637BD;
	.loc	1 70 21                         // sk09_norm_embed.py:70:21
	rsqrt.approx.ftz.f32 	%r219, %r218;
	.loc	1 70 12                         // sk09_norm_embed.py:70:12
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
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:71:29 ]
	bar.sync 	0;
$L__tmp21:
	.loc	1 74 27                         // sk09_norm_embed.py:74:27
	mul.lo.s32 	%r252, %r93, %r84;
	.loc	1 74 21                         // sk09_norm_embed.py:74:21
	cvt.s64.s32 	%rd52, %r252;
	add.s64 	%rd53, %rd47, %rd52;
	.loc	1 74 39                         // sk09_norm_embed.py:74:39
	add.s64 	%rd41, %rd53, %rd50;
	add.s64 	%rd42, %rd41, 4096;
	.loc	1 68 55                         // sk09_norm_embed.py:68:55
	mov.b32 	{%rs33, %rs34}, %r24;
	cvt.f32.bf16 	%r253, %rs33;
	cvt.f32.bf16 	%r254, %rs34;
	mov.b32 	{%rs35, %rs36}, %r25;
	cvt.f32.bf16 	%r255, %rs35;
	cvt.f32.bf16 	%r256, %rs36;
	.loc	1 68 69                         // sk09_norm_embed.py:68:69
	add.f32 	%r257, %r256, 0f00000000;
	add.f32 	%r258, %r255, 0f00000000;
	add.f32 	%r259, %r254, 0f00000000;
	add.f32 	%r260, %r253, 0f00000000;
	.loc	1 69 12                         // sk09_norm_embed.py:69:12
	mul.f32 	%r261, %r260, %r47;
	mul.f32 	%r262, %r259, %r48;
	mul.f32 	%r263, %r258, %r49;
	mul.f32 	%r264, %r257, %r50;
	.loc	1 70 56                         // sk09_norm_embed.py:70:56
	mul.f32 	%r265, %r264, %r235;
	mul.f32 	%r266, %r263, %r234;
	mul.f32 	%r267, %r262, %r233;
	mul.f32 	%r268, %r261, %r232;
	.loc	1 71 36                         // sk09_norm_embed.py:71:36
	abs.f32 	%r269, %r268;
	abs.f32 	%r270, %r267;
	abs.f32 	%r271, %r266;
	abs.f32 	%r272, %r265;
	.loc	1 68 55                         // sk09_norm_embed.py:68:55
	mov.b32 	{%rs37, %rs38}, %r22;
	cvt.f32.bf16 	%r273, %rs37;
	cvt.f32.bf16 	%r274, %rs38;
	mov.b32 	{%rs39, %rs40}, %r23;
	cvt.f32.bf16 	%r275, %rs39;
	cvt.f32.bf16 	%r276, %rs40;
	.loc	1 68 69                         // sk09_norm_embed.py:68:69
	add.f32 	%r277, %r276, 0f00000000;
	add.f32 	%r278, %r275, 0f00000000;
	add.f32 	%r279, %r274, 0f00000000;
	add.f32 	%r280, %r273, 0f00000000;
	.loc	1 69 12                         // sk09_norm_embed.py:69:12
	mul.f32 	%r281, %r280, %r43;
	mul.f32 	%r282, %r279, %r44;
	mul.f32 	%r283, %r278, %r45;
	mul.f32 	%r284, %r277, %r46;
	.loc	1 70 56                         // sk09_norm_embed.py:70:56
	mul.f32 	%r285, %r284, %r231;
	mul.f32 	%r286, %r283, %r230;
	mul.f32 	%r287, %r282, %r229;
	mul.f32 	%r288, %r281, %r228;
	.loc	1 71 36                         // sk09_norm_embed.py:71:36
	abs.f32 	%r289, %r288;
	abs.f32 	%r290, %r287;
	abs.f32 	%r291, %r286;
	abs.f32 	%r292, %r285;
	.loc	1 68 55                         // sk09_norm_embed.py:68:55
	mov.b32 	{%rs41, %rs42}, %r20;
	cvt.f32.bf16 	%r293, %rs41;
	cvt.f32.bf16 	%r294, %rs42;
	mov.b32 	{%rs43, %rs44}, %r21;
	cvt.f32.bf16 	%r295, %rs43;
	cvt.f32.bf16 	%r296, %rs44;
	.loc	1 68 69                         // sk09_norm_embed.py:68:69
	add.f32 	%r297, %r296, 0f00000000;
	add.f32 	%r298, %r295, 0f00000000;
	add.f32 	%r299, %r294, 0f00000000;
	add.f32 	%r300, %r293, 0f00000000;
	.loc	1 69 12                         // sk09_norm_embed.py:69:12
	mul.f32 	%r301, %r300, %r39;
	mul.f32 	%r302, %r299, %r40;
	mul.f32 	%r303, %r298, %r41;
	mul.f32 	%r304, %r297, %r42;
	.loc	1 70 56                         // sk09_norm_embed.py:70:56
	mul.f32 	%r305, %r304, %r227;
	mul.f32 	%r306, %r303, %r226;
	mul.f32 	%r307, %r302, %r225;
	mul.f32 	%r308, %r301, %r224;
	.loc	1 71 36                         // sk09_norm_embed.py:71:36
	abs.f32 	%r309, %r308;
	abs.f32 	%r310, %r307;
	abs.f32 	%r311, %r306;
	abs.f32 	%r312, %r305;
	.loc	1 68 55                         // sk09_norm_embed.py:68:55
	mov.b32 	{%rs45, %rs46}, %r18;
	cvt.f32.bf16 	%r313, %rs45;
	cvt.f32.bf16 	%r314, %rs46;
	mov.b32 	{%rs47, %rs48}, %r19;
	cvt.f32.bf16 	%r315, %rs47;
	cvt.f32.bf16 	%r316, %rs48;
	.loc	1 68 69                         // sk09_norm_embed.py:68:69
	add.f32 	%r317, %r316, 0f00000000;
	add.f32 	%r318, %r315, 0f00000000;
	add.f32 	%r319, %r314, 0f00000000;
	add.f32 	%r320, %r313, 0f00000000;
	.loc	1 69 12                         // sk09_norm_embed.py:69:12
	mul.f32 	%r321, %r320, %r34;
	mul.f32 	%r322, %r319, %r36;
	mul.f32 	%r323, %r318, %r37;
	mul.f32 	%r324, %r317, %r38;
	.loc	1 70 56                         // sk09_norm_embed.py:70:56
	mul.f32 	%r325, %r324, %r223;
	mul.f32 	%r326, %r323, %r222;
	mul.f32 	%r327, %r322, %r221;
	mul.f32 	%r328, %r321, %r220;
	.loc	1 71 36                         // sk09_norm_embed.py:71:36
	abs.f32 	%r329, %r328;
	abs.f32 	%r330, %r327;
	abs.f32 	%r331, %r326;
	abs.f32 	%r332, %r325;
$L__tmp22:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:71:29 ] ]
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
	.loc	1 68 55                         // sk09_norm_embed.py:68:55
	mov.b32 	{%rs49, %rs50}, %r32;
	cvt.f32.bf16 	%r348, %rs49;
	cvt.f32.bf16 	%r349, %rs50;
	mov.b32 	{%rs51, %rs52}, %r33;
	cvt.f32.bf16 	%r350, %rs51;
	cvt.f32.bf16 	%r351, %rs52;
	.loc	1 68 69                         // sk09_norm_embed.py:68:69
	add.f32 	%r352, %r351, 0f00000000;
	add.f32 	%r353, %r350, 0f00000000;
	add.f32 	%r354, %r349, 0f00000000;
	add.f32 	%r355, %r348, 0f00000000;
	.loc	1 69 12                         // sk09_norm_embed.py:69:12
	mul.f32 	%r356, %r355, %r63;
	mul.f32 	%r357, %r354, %r64;
	mul.f32 	%r358, %r353, %r65;
	mul.f32 	%r359, %r352, %r66;
	.loc	1 70 56                         // sk09_norm_embed.py:70:56
	mul.f32 	%r360, %r359, %r251;
	mul.f32 	%r361, %r358, %r250;
	mul.f32 	%r362, %r357, %r249;
	mul.f32 	%r363, %r356, %r248;
	.loc	1 71 36                         // sk09_norm_embed.py:71:36
	abs.f32 	%r364, %r363;
	abs.f32 	%r365, %r362;
	abs.f32 	%r366, %r361;
	abs.f32 	%r367, %r360;
	.loc	1 68 55                         // sk09_norm_embed.py:68:55
	mov.b32 	{%rs53, %rs54}, %r30;
	cvt.f32.bf16 	%r368, %rs53;
	cvt.f32.bf16 	%r369, %rs54;
	mov.b32 	{%rs55, %rs56}, %r31;
	cvt.f32.bf16 	%r370, %rs55;
	cvt.f32.bf16 	%r371, %rs56;
	.loc	1 68 69                         // sk09_norm_embed.py:68:69
	add.f32 	%r372, %r371, 0f00000000;
	add.f32 	%r373, %r370, 0f00000000;
	add.f32 	%r374, %r369, 0f00000000;
	add.f32 	%r375, %r368, 0f00000000;
	.loc	1 69 12                         // sk09_norm_embed.py:69:12
	mul.f32 	%r376, %r375, %r59;
	mul.f32 	%r377, %r374, %r60;
	mul.f32 	%r378, %r373, %r61;
	mul.f32 	%r379, %r372, %r62;
	.loc	1 70 56                         // sk09_norm_embed.py:70:56
	mul.f32 	%r380, %r379, %r247;
	mul.f32 	%r381, %r378, %r246;
	mul.f32 	%r382, %r377, %r245;
	mul.f32 	%r383, %r376, %r244;
	.loc	1 71 36                         // sk09_norm_embed.py:71:36
	abs.f32 	%r384, %r383;
	abs.f32 	%r385, %r382;
	abs.f32 	%r386, %r381;
	abs.f32 	%r387, %r380;
	.loc	1 68 55                         // sk09_norm_embed.py:68:55
	mov.b32 	{%rs57, %rs58}, %r28;
	cvt.f32.bf16 	%r388, %rs57;
	cvt.f32.bf16 	%r389, %rs58;
	mov.b32 	{%rs59, %rs60}, %r29;
	cvt.f32.bf16 	%r390, %rs59;
	cvt.f32.bf16 	%r391, %rs60;
	.loc	1 68 69                         // sk09_norm_embed.py:68:69
	add.f32 	%r392, %r391, 0f00000000;
	add.f32 	%r393, %r390, 0f00000000;
	add.f32 	%r394, %r389, 0f00000000;
	add.f32 	%r395, %r388, 0f00000000;
	.loc	1 69 12                         // sk09_norm_embed.py:69:12
	mul.f32 	%r396, %r395, %r55;
	mul.f32 	%r397, %r394, %r56;
	mul.f32 	%r398, %r393, %r57;
	mul.f32 	%r399, %r392, %r58;
	.loc	1 70 56                         // sk09_norm_embed.py:70:56
	mul.f32 	%r400, %r399, %r243;
	mul.f32 	%r401, %r398, %r242;
	mul.f32 	%r402, %r397, %r241;
	mul.f32 	%r403, %r396, %r240;
	.loc	1 71 36                         // sk09_norm_embed.py:71:36
	abs.f32 	%r404, %r403;
	abs.f32 	%r405, %r402;
	abs.f32 	%r406, %r401;
	abs.f32 	%r407, %r400;
	.loc	1 68 55                         // sk09_norm_embed.py:68:55
	mov.b32 	{%rs61, %rs62}, %r26;
	cvt.f32.bf16 	%r408, %rs61;
	cvt.f32.bf16 	%r409, %rs62;
	mov.b32 	{%rs63, %rs64}, %r27;
	cvt.f32.bf16 	%r410, %rs63;
	cvt.f32.bf16 	%r411, %rs64;
	.loc	1 68 69                         // sk09_norm_embed.py:68:69
	add.f32 	%r412, %r411, 0f00000000;
	add.f32 	%r413, %r410, 0f00000000;
	add.f32 	%r414, %r409, 0f00000000;
	add.f32 	%r415, %r408, 0f00000000;
	.loc	1 69 12                         // sk09_norm_embed.py:69:12
	mul.f32 	%r416, %r415, %r51;
	mul.f32 	%r417, %r414, %r52;
	mul.f32 	%r418, %r413, %r53;
	mul.f32 	%r419, %r412, %r54;
	.loc	1 70 56                         // sk09_norm_embed.py:70:56
	mul.f32 	%r420, %r419, %r239;
	mul.f32 	%r421, %r418, %r238;
	mul.f32 	%r422, %r417, %r237;
	mul.f32 	%r423, %r416, %r236;
	.loc	1 71 36                         // sk09_norm_embed.py:71:36
	abs.f32 	%r424, %r423;
	abs.f32 	%r425, %r422;
	abs.f32 	%r426, %r421;
	abs.f32 	%r427, %r420;
$L__tmp24:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:71:29 ] ]
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
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:71:29 ]
	shfl.sync.bfly.b32 	%r444, %r443, 16, 31, -1;
$L__tmp26:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:71:29 ] ]
	max.f32 	%r445, %r443, %r444;
$L__tmp27:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:71:29 ]
	shfl.sync.bfly.b32 	%r446, %r445, 8, 31, -1;
$L__tmp28:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:71:29 ] ]
	max.f32 	%r447, %r445, %r446;
$L__tmp29:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:71:29 ]
	shfl.sync.bfly.b32 	%r448, %r447, 4, 31, -1;
$L__tmp30:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:71:29 ] ]
	max.f32 	%r449, %r447, %r448;
$L__tmp31:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:71:29 ]
	shfl.sync.bfly.b32 	%r450, %r449, 2, 31, -1;
$L__tmp32:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:71:29 ] ]
	max.f32 	%r451, %r449, %r450;
$L__tmp33:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:71:29 ]
	shfl.sync.bfly.b32 	%r452, %r451, 1, 31, -1;
$L__tmp34:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:71:29 ] ]
	max.f32 	%r72, %r451, %r452;
$L__tmp35:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:71:29 ]
	// begin inline asm
	@%p3 st.shared.b32 [ %r67 + 0 ], %r72;
	// end inline asm
	bar.sync 	0;
	// begin inline asm
	@%p4 ld.shared.b32 %r73, [ %r70 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r453, %r73, 4, 31, -1;
$L__tmp36:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:71:29 ] ]
	max.f32 	%r454, %r73, %r453;
$L__tmp37:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:71:29 ]
	shfl.sync.bfly.b32 	%r455, %r454, 2, 31, -1;
$L__tmp38:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:71:29 ] ]
	max.f32 	%r456, %r454, %r455;
$L__tmp39:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:71:29 ]
	shfl.sync.bfly.b32 	%r457, %r456, 1, 31, -1;
$L__tmp40:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:71:29 ] ]
	max.f32 	%r74, %r456, %r457;
$L__tmp41:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:71:29 ]
	// begin inline asm
	@%p5 st.shared.b32 [ %r70 + 0 ], %r74;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r458, [global_smem];
$L__tmp42:
	.loc	1 71 49                         // sk09_norm_embed.py:71:49
	max.f32 	%r459, %r458, 0f0DA24260;
	mov.b32 	%r460, 0f42FE0000;
	.loc	1 72 22                         // sk09_norm_embed.py:72:22
	div.full.f32 	%r461, %r460, %r459;
	.loc	1 72 14                         // sk09_norm_embed.py:72:14
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
	.loc	1 73 29                         // sk09_norm_embed.py:73:29
	.loc	1 73 39                         // sk09_norm_embed.py:73:39
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
	.loc	1 73 14                         // sk09_norm_embed.py:73:14
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
	.loc	1 73 49                         // sk09_norm_embed.py:73:49
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
	.loc	1 74 70                         // sk09_norm_embed.py:74:70
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
	.loc	1 74 77                         // sk09_norm_embed.py:74:77
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
	.loc	1 74 85                         // sk09_norm_embed.py:74:85
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
	.loc	1 74 45                         // sk09_norm_embed.py:74:45
	// begin inline asm
	@%p1 st.global.v4.b32 [ %rd41 + 0 ], { %r75, %r76, %r77, %r78 };
	// end inline asm
	// begin inline asm
	@%p2 st.global.v4.b32 [ %rd42 + 0 ], { %r79, %r80, %r81, %r82 };
	// end inline asm
	.loc	1 75 21                         // sk09_norm_embed.py:75:21
	mad.wide.u32 	%rd43, %r84, 4, %rd48;
	.loc	1 75 34                         // sk09_norm_embed.py:75:34
	mul.f32 	%r83, %r459, 0f3C010204;
	.loc	1 75 26                         // sk09_norm_embed.py:75:26
	or.b32 	%r670, %r87, %r88;
	setp.eq.b32 	%p6, %r670, 0;
	// begin inline asm
	@%p6 st.global.b32 [ %rd43 + 0 ], { %r83 };
	// end inline asm
	.loc	1 75 4                          // sk09_norm_embed.py:75:4
	ret;
$L__tmp43:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/home/usuario/Proyectos/genesis-vllm-patches/vllm/_genesis/kernels/sk09_norm_embed.py"
	.file	2 "/home/usuario/Proyectos/genesis-vllm-patches/.venv/lib/python3.12/site-packages/triton/language/standard.py"
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
.b32 258                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0xfb DW_TAG_compile_unit
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
.b8 104
.b8 111
.b8 109
.b8 101
.b8 47
.b8 117
.b8 115
.b8 117
.b8 97
.b8 114
.b8 105
.b8 111
.b8 47
.b8 80
.b8 114
.b8 111
.b8 121
.b8 101
.b8 99
.b8 116
.b8 111
.b8 115
.b8 47
.b8 103
.b8 101
.b8 110
.b8 101
.b8 115
.b8 105
.b8 115
.b8 45
.b8 118
.b8 108
.b8 108
.b8 109
.b8 45
.b8 112
.b8 97
.b8 116
.b8 99
.b8 104
.b8 101
.b8 115
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
.b8 2                                   // Abbrev [2] 0x6f:0x1d DW_TAG_subprogram
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
.b8 3                                   // Abbrev [3] 0x8c:0x79 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 111                                // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0xa1:0x32 DW_TAG_inlined_subroutine
.b32 111                                // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp19                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 70                                  // DW_AT_call_line
.b8 28                                  // DW_AT_call_column
.b8 5                                   // Abbrev [5] 0xb9:0x19 DW_TAG_inlined_subroutine
.b32 111                                // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp18                          // DW_AT_high_pc
.b8 2                                   // DW_AT_call_file
.b8 37                                  // DW_AT_call_line
.b8 1
.b8 36                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 4                                   // Abbrev [4] 0xd3:0x31 DW_TAG_inlined_subroutine
.b32 111                                // DW_AT_abstract_origin
.b64 $L__tmp20                          // DW_AT_low_pc
.b64 $L__tmp42                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 71                                  // DW_AT_call_line
.b8 29                                  // DW_AT_call_column
.b8 6                                   // Abbrev [6] 0xeb:0x18 DW_TAG_inlined_subroutine
.b32 111                                // DW_AT_abstract_origin
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

_SK09_RMSNORM_QUANT_NATIVO = _Nativo(
    'B1/sk09_rmsnorm_quant',
    _SK09_RMSNORM_QUANT_PTX, '_sk09_rmsnorm_quant_kernel',
    warps=8, shared=32,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    horneado={9: 8192, 10: 1e-06, 11: 0.0},
    div16=[],
)


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
    _lanzar_quant1((m,),
            x2, weight, sp, q, s, k, x2.stride(0), q.stride(0), 0 if s_pow2 is None else sp.stride(0), triton.next_power_of_2(k), eps)
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
    ``eps != 1e-6``, ``gamma_offset != 0.0``) o con el kill-switch
    ``GENESIS_PTQ_NATIVO=0`` cae al kernel Triton de referencia.
    """
    if habilitado() and {9: triton.next_power_of_2(x.shape[-1]), 10: eps,
                         11: gamma_offset} \
            == _SK09_RMSNORM_QUANT_NATIVO.horneado:
        k = x.shape[-1]
        x2 = x.reshape(-1, k)
        m = x2.shape[0]
        q = torch.empty((m, k), dtype=torch.int8, device=x.device)
        s = torch.empty((m,), dtype=torch.float32, device=x.device)
        sp = _one(x.device) if s_pow2 is None else s_pow2
        _SK09_RMSNORM_QUANT_NATIVO(
            (m,), x2, weight, sp, q, s,
            k, x2.stride(0), q.stride(0), 0 if s_pow2 is None else sp.stride(0),
            triton.next_power_of_2(k), eps, gamma_offset,
        )
        return q, s
    return _sk09_rmsnorm_quant_triton(x, weight, s_pow2, eps, gamma_offset)


_SK09_QUANT_PTX = r"""//
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
	.loc	1 79 0                          // sk09_norm_embed.py:79:0
$L__func_begin0:
	.loc	1 79 0                          // sk09_norm_embed.py:79:0

// %bb.0:
	ld.param.b64 	%rd14, [_sk09_quant_kernel_param_0];
	ld.param.b64 	%rd15, [_sk09_quant_kernel_param_1];
$L__tmp0:
	.loc	1 81 24                         // sk09_norm_embed.py:81:24
	mov.u32 	%r56, %ctaid.x;
	ld.param.b64 	%rd16, [_sk09_quant_kernel_param_2];
	.loc	1 82 24                         // sk09_norm_embed.py:82:24
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
	.loc	1 83 18                         // sk09_norm_embed.py:83:18
	setp.lt.s32 	%p1, %r65, %r59;
	setp.lt.s32 	%p2, %r66, %r59;
	setp.lt.s32 	%p3, %r67, %r59;
	setp.lt.s32 	%p4, %r68, %r59;
	.loc	1 84 30                         // sk09_norm_embed.py:84:30
	mul.lo.s32 	%r70, %r61, %r56;
	.loc	1 84 24                         // sk09_norm_embed.py:84:24
	mad.wide.s32 	%rd17, %r70, 2, %rd14;
	.loc	1 84 42                         // sk09_norm_embed.py:84:42
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
	.loc	1 84 16                         // sk09_norm_embed.py:84:16
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
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:85:29 ]
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
	.loc	1 88 27                         // sk09_norm_embed.py:88:27
	mul.lo.s32 	%r76, %r63, %r56;
	.loc	1 88 21                         // sk09_norm_embed.py:88:21
	cvt.s64.s32 	%rd20, %r76;
	add.s64 	%rd21, %rd15, %rd20;
	.loc	1 88 39                         // sk09_norm_embed.py:88:39
	add.s64 	%rd9, %rd21, %rd18;
	add.s64 	%rd10, %rd9, 4096;
	add.s64 	%rd11, %rd9, 8192;
	add.s64 	%rd12, %rd21, %rd19;
	.loc	1 84 73                         // sk09_norm_embed.py:84:73
	mov.b32 	{%rs1, %rs2}, %r9;
	cvt.f32.bf16 	%r77, %rs2;
	cvt.f32.bf16 	%r78, %rs1;
	mov.b32 	{%rs3, %rs4}, %r8;
	cvt.f32.bf16 	%r79, %rs4;
	cvt.f32.bf16 	%r80, %rs3;
	.loc	1 85 36                         // sk09_norm_embed.py:85:36
	abs.f32 	%r81, %r80;
	abs.f32 	%r82, %r79;
	abs.f32 	%r83, %r78;
	abs.f32 	%r84, %r77;
	.loc	1 84 73                         // sk09_norm_embed.py:84:73
	mov.b32 	{%rs5, %rs6}, %r7;
	cvt.f32.bf16 	%r85, %rs6;
	cvt.f32.bf16 	%r86, %rs5;
	mov.b32 	{%rs7, %rs8}, %r6;
	cvt.f32.bf16 	%r87, %rs8;
	cvt.f32.bf16 	%r88, %rs7;
	.loc	1 85 36                         // sk09_norm_embed.py:85:36
	abs.f32 	%r89, %r88;
	abs.f32 	%r90, %r87;
	abs.f32 	%r91, %r86;
	abs.f32 	%r92, %r85;
	.loc	1 84 73                         // sk09_norm_embed.py:84:73
	mov.b32 	{%rs9, %rs10}, %r4;
	cvt.f32.bf16 	%r93, %rs10;
	cvt.f32.bf16 	%r94, %rs9;
	mov.b32 	{%rs11, %rs12}, %r3;
	cvt.f32.bf16 	%r95, %rs12;
	cvt.f32.bf16 	%r96, %rs11;
	.loc	1 85 36                         // sk09_norm_embed.py:85:36
	abs.f32 	%r97, %r96;
	abs.f32 	%r98, %r95;
	abs.f32 	%r99, %r94;
	abs.f32 	%r100, %r93;
	.loc	1 84 73                         // sk09_norm_embed.py:84:73
	mov.b32 	{%rs13, %rs14}, %r2;
	cvt.f32.bf16 	%r101, %rs14;
	cvt.f32.bf16 	%r102, %rs13;
	mov.b32 	{%rs15, %rs16}, %r1;
	cvt.f32.bf16 	%r103, %rs16;
	cvt.f32.bf16 	%r104, %rs15;
	.loc	1 85 36                         // sk09_norm_embed.py:85:36
	abs.f32 	%r105, %r104;
	abs.f32 	%r106, %r103;
	abs.f32 	%r107, %r102;
	abs.f32 	%r108, %r101;
$L__tmp3:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:85:29 ] ]
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
	.loc	1 84 73                         // sk09_norm_embed.py:84:73
	mov.b32 	{%rs17, %rs18}, %r17;
	cvt.f32.bf16 	%r124, %rs18;
	cvt.f32.bf16 	%r125, %rs17;
	mov.b32 	{%rs19, %rs20}, %r16;
	cvt.f32.bf16 	%r126, %rs20;
	cvt.f32.bf16 	%r127, %rs19;
	.loc	1 85 36                         // sk09_norm_embed.py:85:36
	abs.f32 	%r128, %r127;
	abs.f32 	%r129, %r126;
	abs.f32 	%r130, %r125;
	abs.f32 	%r131, %r124;
	.loc	1 84 73                         // sk09_norm_embed.py:84:73
	mov.b32 	{%rs21, %rs22}, %r15;
	cvt.f32.bf16 	%r132, %rs22;
	cvt.f32.bf16 	%r133, %rs21;
	mov.b32 	{%rs23, %rs24}, %r14;
	cvt.f32.bf16 	%r134, %rs24;
	cvt.f32.bf16 	%r135, %rs23;
	.loc	1 85 36                         // sk09_norm_embed.py:85:36
	abs.f32 	%r136, %r135;
	abs.f32 	%r137, %r134;
	abs.f32 	%r138, %r133;
	abs.f32 	%r139, %r132;
	.loc	1 84 73                         // sk09_norm_embed.py:84:73
	mov.b32 	{%rs25, %rs26}, %r13;
	cvt.f32.bf16 	%r140, %rs26;
	cvt.f32.bf16 	%r141, %rs25;
	mov.b32 	{%rs27, %rs28}, %r12;
	cvt.f32.bf16 	%r142, %rs28;
	cvt.f32.bf16 	%r143, %rs27;
	.loc	1 85 36                         // sk09_norm_embed.py:85:36
	abs.f32 	%r144, %r143;
	abs.f32 	%r145, %r142;
	abs.f32 	%r146, %r141;
	abs.f32 	%r147, %r140;
	.loc	1 84 73                         // sk09_norm_embed.py:84:73
	mov.b32 	{%rs29, %rs30}, %r11;
	cvt.f32.bf16 	%r148, %rs30;
	cvt.f32.bf16 	%r149, %rs29;
	mov.b32 	{%rs31, %rs32}, %r10;
	cvt.f32.bf16 	%r150, %rs32;
	cvt.f32.bf16 	%r151, %rs31;
	.loc	1 85 36                         // sk09_norm_embed.py:85:36
	abs.f32 	%r152, %r151;
	abs.f32 	%r153, %r150;
	abs.f32 	%r154, %r149;
	abs.f32 	%r155, %r148;
$L__tmp5:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:85:29 ] ]
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
	.loc	1 84 73                         // sk09_norm_embed.py:84:73
	mov.b32 	{%rs33, %rs34}, %r25;
	cvt.f32.bf16 	%r172, %rs34;
	cvt.f32.bf16 	%r173, %rs33;
	mov.b32 	{%rs35, %rs36}, %r24;
	cvt.f32.bf16 	%r174, %rs36;
	cvt.f32.bf16 	%r175, %rs35;
	.loc	1 85 36                         // sk09_norm_embed.py:85:36
	abs.f32 	%r176, %r175;
	abs.f32 	%r177, %r174;
	abs.f32 	%r178, %r173;
	abs.f32 	%r179, %r172;
	.loc	1 84 73                         // sk09_norm_embed.py:84:73
	mov.b32 	{%rs37, %rs38}, %r23;
	cvt.f32.bf16 	%r180, %rs38;
	cvt.f32.bf16 	%r181, %rs37;
	mov.b32 	{%rs39, %rs40}, %r22;
	cvt.f32.bf16 	%r182, %rs40;
	cvt.f32.bf16 	%r183, %rs39;
	.loc	1 85 36                         // sk09_norm_embed.py:85:36
	abs.f32 	%r184, %r183;
	abs.f32 	%r185, %r182;
	abs.f32 	%r186, %r181;
	abs.f32 	%r187, %r180;
	.loc	1 84 73                         // sk09_norm_embed.py:84:73
	mov.b32 	{%rs41, %rs42}, %r21;
	cvt.f32.bf16 	%r188, %rs42;
	cvt.f32.bf16 	%r189, %rs41;
	mov.b32 	{%rs43, %rs44}, %r20;
	cvt.f32.bf16 	%r190, %rs44;
	cvt.f32.bf16 	%r191, %rs43;
	.loc	1 85 36                         // sk09_norm_embed.py:85:36
	abs.f32 	%r192, %r191;
	abs.f32 	%r193, %r190;
	abs.f32 	%r194, %r189;
	abs.f32 	%r195, %r188;
	.loc	1 84 73                         // sk09_norm_embed.py:84:73
	mov.b32 	{%rs45, %rs46}, %r19;
	cvt.f32.bf16 	%r196, %rs46;
	cvt.f32.bf16 	%r197, %rs45;
	mov.b32 	{%rs47, %rs48}, %r18;
	cvt.f32.bf16 	%r198, %rs48;
	cvt.f32.bf16 	%r199, %rs47;
	.loc	1 85 36                         // sk09_norm_embed.py:85:36
	abs.f32 	%r200, %r199;
	abs.f32 	%r201, %r198;
	abs.f32 	%r202, %r197;
	abs.f32 	%r203, %r196;
$L__tmp7:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:85:29 ] ]
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
	.loc	1 84 73                         // sk09_norm_embed.py:84:73
	mov.b32 	{%rs49, %rs50}, %r33;
	cvt.f32.bf16 	%r220, %rs50;
	cvt.f32.bf16 	%r221, %rs49;
	mov.b32 	{%rs51, %rs52}, %r32;
	cvt.f32.bf16 	%r222, %rs52;
	cvt.f32.bf16 	%r223, %rs51;
	.loc	1 85 36                         // sk09_norm_embed.py:85:36
	abs.f32 	%r224, %r223;
	abs.f32 	%r225, %r222;
	abs.f32 	%r226, %r221;
	abs.f32 	%r227, %r220;
	.loc	1 84 73                         // sk09_norm_embed.py:84:73
	mov.b32 	{%rs53, %rs54}, %r31;
	cvt.f32.bf16 	%r228, %rs54;
	cvt.f32.bf16 	%r229, %rs53;
	mov.b32 	{%rs55, %rs56}, %r30;
	cvt.f32.bf16 	%r230, %rs56;
	cvt.f32.bf16 	%r231, %rs55;
	.loc	1 85 36                         // sk09_norm_embed.py:85:36
	abs.f32 	%r232, %r231;
	abs.f32 	%r233, %r230;
	abs.f32 	%r234, %r229;
	abs.f32 	%r235, %r228;
	.loc	1 84 73                         // sk09_norm_embed.py:84:73
	mov.b32 	{%rs57, %rs58}, %r29;
	cvt.f32.bf16 	%r236, %rs58;
	cvt.f32.bf16 	%r237, %rs57;
	mov.b32 	{%rs59, %rs60}, %r28;
	cvt.f32.bf16 	%r238, %rs60;
	cvt.f32.bf16 	%r239, %rs59;
	.loc	1 85 36                         // sk09_norm_embed.py:85:36
	abs.f32 	%r240, %r239;
	abs.f32 	%r241, %r238;
	abs.f32 	%r242, %r237;
	abs.f32 	%r243, %r236;
	.loc	1 84 73                         // sk09_norm_embed.py:84:73
	mov.b32 	{%rs61, %rs62}, %r27;
	cvt.f32.bf16 	%r244, %rs62;
	cvt.f32.bf16 	%r245, %rs61;
	mov.b32 	{%rs63, %rs64}, %r26;
	cvt.f32.bf16 	%r246, %rs64;
	cvt.f32.bf16 	%r247, %rs63;
	.loc	1 85 36                         // sk09_norm_embed.py:85:36
	abs.f32 	%r248, %r247;
	abs.f32 	%r249, %r246;
	abs.f32 	%r250, %r245;
	abs.f32 	%r251, %r244;
$L__tmp9:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:85:29 ] ]
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
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:85:29 ]
	shfl.sync.bfly.b32 	%r268, %r267, 16, 31, -1;
$L__tmp11:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:85:29 ] ]
	max.f32 	%r269, %r267, %r268;
$L__tmp12:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:85:29 ]
	shfl.sync.bfly.b32 	%r270, %r269, 8, 31, -1;
$L__tmp13:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:85:29 ] ]
	max.f32 	%r271, %r269, %r270;
$L__tmp14:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:85:29 ]
	shfl.sync.bfly.b32 	%r272, %r271, 4, 31, -1;
$L__tmp15:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:85:29 ] ]
	max.f32 	%r273, %r271, %r272;
$L__tmp16:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:85:29 ]
	shfl.sync.bfly.b32 	%r274, %r273, 2, 31, -1;
$L__tmp17:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:85:29 ] ]
	max.f32 	%r275, %r273, %r274;
$L__tmp18:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:85:29 ]
	shfl.sync.bfly.b32 	%r276, %r275, 1, 31, -1;
$L__tmp19:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:85:29 ] ]
	max.f32 	%r35, %r275, %r276;
$L__tmp20:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:85:29 ]
	// begin inline asm
	@%p5 st.shared.b32 [ %r34 + 0 ], %r35;
	// end inline asm
	bar.sync 	0;
	// begin inline asm
	@%p6 ld.shared.b32 %r36, [ %r37 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r277, %r36, 4, 31, -1;
$L__tmp21:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:85:29 ] ]
	max.f32 	%r278, %r36, %r277;
$L__tmp22:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:85:29 ]
	shfl.sync.bfly.b32 	%r279, %r278, 2, 31, -1;
$L__tmp23:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:85:29 ] ]
	max.f32 	%r280, %r278, %r279;
$L__tmp24:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:85:29 ]
	shfl.sync.bfly.b32 	%r281, %r280, 1, 31, -1;
$L__tmp25:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk09_norm_embed.py:85:29 ] ]
	max.f32 	%r38, %r280, %r281;
$L__tmp26:
	.loc	2 191 40                        // standard.py:191:40 @[ sk09_norm_embed.py:85:29 ]
	// begin inline asm
	@%p7 st.shared.b32 [ %r37 + 0 ], %r38;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r282, [global_smem];
$L__tmp27:
	.loc	1 85 49                         // sk09_norm_embed.py:85:49
	max.f32 	%r283, %r282, 0f0DA24260;
	mov.b32 	%r284, 0f42FE0000;
	.loc	1 86 22                         // sk09_norm_embed.py:86:22
	div.full.f32 	%r285, %r284, %r283;
	.loc	1 86 14                         // sk09_norm_embed.py:86:14
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
	.loc	1 87 29                         // sk09_norm_embed.py:87:29
	.loc	1 87 39                         // sk09_norm_embed.py:87:39
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
	.loc	1 87 14                         // sk09_norm_embed.py:87:14
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
	.loc	1 87 49                         // sk09_norm_embed.py:87:49
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
	.loc	1 88 70                         // sk09_norm_embed.py:88:70
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
	.loc	1 88 77                         // sk09_norm_embed.py:88:77
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
	.loc	1 88 85                         // sk09_norm_embed.py:88:85
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
	.loc	1 88 45                         // sk09_norm_embed.py:88:45
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
	.loc	1 89 21                         // sk09_norm_embed.py:89:21
	mad.wide.u32 	%rd13, %r56, 4, %rd16;
	.loc	1 89 34                         // sk09_norm_embed.py:89:34
	mul.f32 	%r55, %r283, 0f3C010204;
	.loc	1 89 26                         // sk09_norm_embed.py:89:26
	or.b32 	%r702, %r60, %r62;
	setp.eq.b32 	%p8, %r702, 0;
	// begin inline asm
	@%p8 st.global.b32 [ %rd13 + 0 ], { %r55 };
	// end inline asm
	.loc	1 89 4                          // sk09_norm_embed.py:89:4
	ret;
$L__tmp28:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/home/usuario/Proyectos/genesis-vllm-patches/vllm/_genesis/kernels/sk09_norm_embed.py"
	.file	2 "/home/usuario/Proyectos/genesis-vllm-patches/.venv/lib/python3.12/site-packages/triton/language/standard.py"
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
.b32 200                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0xc1 DW_TAG_compile_unit
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
.b8 104
.b8 111
.b8 109
.b8 101
.b8 47
.b8 117
.b8 115
.b8 117
.b8 97
.b8 114
.b8 105
.b8 111
.b8 47
.b8 80
.b8 114
.b8 111
.b8 121
.b8 101
.b8 99
.b8 116
.b8 111
.b8 115
.b8 47
.b8 103
.b8 101
.b8 110
.b8 101
.b8 115
.b8 105
.b8 115
.b8 45
.b8 118
.b8 108
.b8 108
.b8 109
.b8 45
.b8 112
.b8 97
.b8 116
.b8 99
.b8 104
.b8 101
.b8 115
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
.b8 2                                   // Abbrev [2] 0x6f:0x15 DW_TAG_subprogram
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
.b8 3                                   // Abbrev [3] 0x84:0x47 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 111                                // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x99:0x31 DW_TAG_inlined_subroutine
.b32 111                                // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp27                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 85                                  // DW_AT_call_line
.b8 29                                  // DW_AT_call_column
.b8 5                                   // Abbrev [5] 0xb1:0x18 DW_TAG_inlined_subroutine
.b32 111                                // DW_AT_abstract_origin
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

_SK09_QUANT_NATIVO = _Nativo(
    'B1/sk09_quant',
    _SK09_QUANT_PTX, '_sk09_quant_kernel',
    warps=8, shared=32,
    abi=[0, 1, 2, 3, 4, 5],
    horneado={6: 16384},
    div16=[],
)


def _sk09_quant_triton(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Referencia para tests: lanza el kernel Triton original (JIT)."""
    k = x.shape[-1]
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    s = torch.empty((m,), dtype=torch.float32, device=x.device)
    _lanzar_quant0((m,),
            x2, q, s, k, x2.stride(0), q.stride(0), triton.next_power_of_2(k))
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
    kill-switch ``GENESIS_PTQ_NATIVO=0`` cae al kernel Triton de referencia.
    """
    if habilitado() and {6: triton.next_power_of_2(x.shape[-1])} \
            == _SK09_QUANT_NATIVO.horneado:
        k = x.shape[-1]
        x2 = x.reshape(-1, k)
        m = x2.shape[0]
        q = torch.empty((m, k), dtype=torch.int8, device=x.device)
        s = torch.empty((m,), dtype=torch.float32, device=x.device)
        _SK09_QUANT_NATIVO(
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


# --- quant: PTX embebido (E1=32 lop3, E2=0 mul) ---

_QPTX0 = r"""//
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

_QVAR0 = _Nativo(
    "sk09_norm_embed/_sk09_quant_kernel",
    _QPTX0, "_sk09_quant_kernel",
    warps=8, shared=32,
    abi=[0, 1, 2, 3, 4, 5],
    horneado={6: 8192},
    div16=[3, 4, 5],
)


def _q0_impl(grid: list[int], x_ptr: torch.Tensor, q_ptr: torch.Tensor, s_ptr: torch.Tensor, K: int, stride_xm: int, stride_qm: int, BLOCK: int) -> None:
    """Cuerpo del custom op: lanza el PTX embebido."""
    _QVAR0(tuple(grid), x_ptr, q_ptr, s_ptr, K, stride_xm, stride_qm, BLOCK)


def _q0_fake(grid: list[int], x_ptr: torch.Tensor, q_ptr: torch.Tensor, s_ptr: torch.Tensor, K: int, stride_xm: int, stride_qm: int, BLOCK: int) -> None:
    """Meta impl: no toca la GPU; las salidas se mutan in-place."""
    return None


_q0_registrado = False


def _q0_registrar() -> None:
    """Registra el op una sola vez, fuera de la region compilada."""
    global _q0_registrado
    if _q0_registrado:
        return
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op(
        op_name="genesis_sk09_norm_embed_q0",
        op_func=_q0_impl,
        mutates_args=['q_ptr', 's_ptr'],
        fake_impl=_q0_fake,
    )
    _q0_registrado = True


def _lanzar_quant0(grid, *args):
    """PTX embebido via custom op; cae al Triton con el kill-switch.

    ``GENESIS_PTQ_NATIVO=0`` vuelve al kernel Triton, que queda en este
    archivo como referencia para los tests.
    """
    if habilitado():
        _q0_registrar()
        g = [grid] if isinstance(grid, int) else list(grid)
        return torch.ops.vllm.genesis_sk09_norm_embed_q0(g, *args)
    ce = ['BLOCK']
    nom = ['x_ptr', 'q_ptr', 's_ptr', 'K', 'stride_xm', 'stride_qm', 'BLOCK']
    return _sk09_quant_kernel[grid](*args[:len(nom) - len(ce)],
                    **{n: args[nom.index(n)] for n in ce},
                    num_warps=8, num_stages=1)


# --- quant: PTX embebido (E1=32 lop3, E2=0 mul) ---

_QPTX1 = r"""//
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

_QVAR1 = _Nativo(
    "sk09_norm_embed/_sk09_rmsnorm_quant_kernel",
    _QPTX1, "_sk09_rmsnorm_quant_kernel",
    warps=8, shared=32,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    horneado={9: 8192, 10: 1e-06, 11: 0.0},
    div16=[5, 6, 7, 8],
)


def _q1_impl(grid: list[int], x_ptr: torch.Tensor, w_ptr: torch.Tensor, s_pow2_ptr: torch.Tensor, q_ptr: torch.Tensor, s_ptr: torch.Tensor, K: int, stride_xm: int, stride_qm: int, stride_sp: int, BLOCK: int, EPS: float, GAMMA_OFFSET: float) -> None:
    """Cuerpo del custom op: lanza el PTX embebido."""
    _QVAR1(tuple(grid), x_ptr, w_ptr, s_pow2_ptr, q_ptr, s_ptr, K, stride_xm, stride_qm, stride_sp, BLOCK, EPS, GAMMA_OFFSET)


def _q1_fake(grid: list[int], x_ptr: torch.Tensor, w_ptr: torch.Tensor, s_pow2_ptr: torch.Tensor, q_ptr: torch.Tensor, s_ptr: torch.Tensor, K: int, stride_xm: int, stride_qm: int, stride_sp: int, BLOCK: int, EPS: float, GAMMA_OFFSET: float) -> None:
    """Meta impl: no toca la GPU; las salidas se mutan in-place."""
    return None


_q1_registrado = False


def _q1_registrar() -> None:
    """Registra el op una sola vez, fuera de la region compilada."""
    global _q1_registrado
    if _q1_registrado:
        return
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op(
        op_name="genesis_sk09_norm_embed_q1",
        op_func=_q1_impl,
        mutates_args=['q_ptr', 's_ptr'],
        fake_impl=_q1_fake,
    )
    _q1_registrado = True


def _lanzar_quant1(grid, *args):
    """PTX embebido via custom op; cae al Triton con el kill-switch.

    ``GENESIS_PTQ_NATIVO=0`` vuelve al kernel Triton, que queda en este
    archivo como referencia para los tests.
    """
    if habilitado():
        _q1_registrar()
        g = [grid] if isinstance(grid, int) else list(grid)
        return torch.ops.vllm.genesis_sk09_norm_embed_q1(g, *args)
    ce = ['BLOCK', 'EPS', 'GAMMA_OFFSET']
    nom = ['x_ptr', 'w_ptr', 's_pow2_ptr', 'q_ptr', 's_ptr', 'K', 'stride_xm', 'stride_qm', 'stride_sp', 'BLOCK', 'EPS', 'GAMMA_OFFSET']
    return _sk09_rmsnorm_quant_kernel[grid](*args[:len(nom) - len(ce)],
                    **{n: args[nom.index(n)] for n in ce},
                    num_warps=8, num_stages=1)


# --- quant: PTX embebido (E1=0 lop3, E2=0 mul) ---

_QPTX2 = r"""//
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

_QVAR2 = _Nativo(
    "sk09_norm_embed/_sk09_rmsnorm_kernel",
    _QPTX2, "_sk09_rmsnorm_kernel",
    warps=8, shared=32,
    abi=[0, 1, 2, 3, 4, 5],
    horneado={6: 8192, 7: 1e-06, 8: 0.0},
    div16=[3, 4, 5],
)


def _q2_impl(grid: list[int], x_ptr: torch.Tensor, w_ptr: torch.Tensor, out_ptr: torch.Tensor, K: int, stride_xm: int, stride_om: int, BLOCK: int, EPS: float, GAMMA_OFFSET: float) -> None:
    """Cuerpo del custom op: lanza el PTX embebido."""
    _QVAR2(tuple(grid), x_ptr, w_ptr, out_ptr, K, stride_xm, stride_om, BLOCK, EPS, GAMMA_OFFSET)


def _q2_fake(grid: list[int], x_ptr: torch.Tensor, w_ptr: torch.Tensor, out_ptr: torch.Tensor, K: int, stride_xm: int, stride_om: int, BLOCK: int, EPS: float, GAMMA_OFFSET: float) -> None:
    """Meta impl: no toca la GPU; las salidas se mutan in-place."""
    return None


_q2_registrado = False


def _q2_registrar() -> None:
    """Registra el op una sola vez, fuera de la region compilada."""
    global _q2_registrado
    if _q2_registrado:
        return
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op(
        op_name="genesis_sk09_norm_embed_q2",
        op_func=_q2_impl,
        mutates_args=['out_ptr'],
        fake_impl=_q2_fake,
    )
    _q2_registrado = True


def _lanzar_quant2(grid, *args):
    """PTX embebido via custom op; cae al Triton con el kill-switch.

    ``GENESIS_PTQ_NATIVO=0`` vuelve al kernel Triton, que queda en este
    archivo como referencia para los tests.
    """
    if habilitado():
        _q2_registrar()
        g = [grid] if isinstance(grid, int) else list(grid)
        return torch.ops.vllm.genesis_sk09_norm_embed_q2(g, *args)
    ce = ['BLOCK', 'EPS', 'GAMMA_OFFSET']
    nom = ['x_ptr', 'w_ptr', 'out_ptr', 'K', 'stride_xm', 'stride_om', 'BLOCK', 'EPS', 'GAMMA_OFFSET']
    return _sk09_rmsnorm_kernel[grid](*args[:len(nom) - len(ce)],
                    **{n: args[nom.index(n)] for n in ce},
                    num_warps=8, num_stages=1)
