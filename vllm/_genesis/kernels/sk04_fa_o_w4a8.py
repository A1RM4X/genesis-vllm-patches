# SPDX-License-Identifier: Apache-2.0
"""SK-04 W4A8 — FA_O_W4A8 — capa completa Full-Attention o_proj (RowParallel) en INT4/INT8.

Capa completa: quant per-token -> GEMM W4A8 con residual fusionado en el
epílogo -> all_reduce. Geometría per-rank TP=2: K=3072, N=5120.

Convención W4A8 (única para toda la familia SK-*_w4a8)
------------------------------------------------------
``w_packed`` ``[K/2, N]`` int8: cada byte lleva dos pesos de 4 bits, el de
``k=2i`` en el nibble bajo y el de ``k=2i+1`` en el alto, con zero-point 8.
``w_scales`` ``[K/128, N]`` fp32: una escala por grupo de 128 filas (GPTQ).
Activación INT8 per-token con su escala ``[M]`` fp32.

Qué cambió respecto de la versión anterior
------------------------------------------
El desempaquetado leía **cada byte dos veces** —una para el nibble par y otra
para el impar— y descartaba la mitad en cada lectura::

    packed_k = cur_k // 2
    is_even  = (cur_k % 2) == 0
    packed   = tl.load(w_packed_ptrs, ...)      # el byte entero
    w_q      = tl.where(is_even[:, None], packed & 15, (packed >> 4) & 15)

Eso da exactamente el mismo tráfico de memoria que INT8: **el ahorro de ancho
de banda de W4, que es el único motivo para hacer W4A8, desaparecía**. Ahora el
byte se lee una vez, salen los dos nibbles juntos, la activación se lee contigua
y se parte en pares/impares en registro con ``tl.split``, y el producto se cierra
con dos ``tl.dot`` sobre las mitades::

    packed = tl.load(w_ptrs)                    # una vez
    w_lo, w_hi = (packed & 15) - 8, ((packed >> 4) & 15) - 8
    a_even, a_odd = tl.split(tl.reshape(a, (BLOCK_M, BLOCK_K // 2, 2)))
    int_acc = tl.dot(a_even, w_lo) + tl.dot(a_odd, w_hi)

Además el acumulador pasó de bf16 (8 bits de mantisa, 96 parciales
encadenados) a INT32 por grupo con acumulación fp32, y el epílogo de escala
salió del bucle interno: antes se aplicaba una vez por cada ``BLOCK_K=32``.

Diseño (sm_86, mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32):
  * ``BLOCK_K == GROUP_SIZE == 128`` -> una escala de grupo por iteración.
  * Tiles grandes, punteros que avanzan, grid 1-D con swizzle L2, num_stages 3-4.
  * Residual fusionado en el epílogo (strides 0 sobre un escalar cero cuando
    no hay residual: broadcast gratis, sin rama).

RowParallel: con TP>1 el residual se suma después del all_reduce; con TP=1 va
fusionado en el epílogo.


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
        """Ensambla y carga el modulo una sola vez (thread-safe)."""
        if self._fn is not None:
            return
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




_SK04 = "vllm._genesis.kernels.sk04_fa_o_w4a8"

_PTX_SRC_SK04_QUANT = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk04_fa_o_w4a8_quant_kernel // -- Begin function _sk04_fa_o_w4a8_quant_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk04_fa_o_w4a8_quant_kernel
.visible .entry _sk04_fa_o_w4a8_quant_kernel(
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_quant_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_quant_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_quant_kernel_param_2,
	.param .u32 _sk04_fa_o_w4a8_quant_kernel_param_3,
	.param .u32 _sk04_fa_o_w4a8_quant_kernel_param_4,
	.param .u32 _sk04_fa_o_w4a8_quant_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_quant_kernel_param_6,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_quant_kernel_param_7
)
.reqntid 256
{
	.reg .pred 	%p<23>;
	.reg .b16 	%rs<17>;
	.reg .b32 	%r<207>;
	.reg .b64 	%rd<12>;
	.loc	1 113 0                         // sk04_fa_o_w4a8.py:113:0
$L__func_begin0:
	.loc	1 113 0                         // sk04_fa_o_w4a8.py:113:0

// %bb.0:
	ld.param.b64 	%rd5, [_sk04_fa_o_w4a8_quant_kernel_param_0];
	ld.param.b64 	%rd6, [_sk04_fa_o_w4a8_quant_kernel_param_1];
$L__tmp0:
	.loc	1 115 24                        // sk04_fa_o_w4a8.py:115:24
	mov.u32 	%r20, %ctaid.x;
	ld.param.b64 	%rd7, [_sk04_fa_o_w4a8_quant_kernel_param_2];
	.loc	1 116 24                        // sk04_fa_o_w4a8.py:116:24
	mov.u32 	%r21, %tid.x;
	and.b32 	%r22, %r21, 255;
	ld.param.b32 	%r23, [_sk04_fa_o_w4a8_quant_kernel_param_3];
	and.b32 	%r24, %r21, 31;
	ld.param.b32 	%r25, [_sk04_fa_o_w4a8_quant_kernel_param_4];
	shr.u32 	%r26, %r21, 5;
	ld.param.b32 	%r27, [_sk04_fa_o_w4a8_quant_kernel_param_5];
	shl.b32 	%r28, %r21, 4;
	and.b32 	%r29, %r28, 4080;
	.loc	1 117 18                        // sk04_fa_o_w4a8.py:117:18
	setp.lt.s32 	%p1, %r29, %r23;
	.loc	1 118 30                        // sk04_fa_o_w4a8.py:118:30
	mul.lo.s32 	%r30, %r25, %r20;
	.loc	1 118 24                        // sk04_fa_o_w4a8.py:118:24
	mad.wide.s32 	%rd8, %r30, 2, %rd5;
	.loc	1 118 42                        // sk04_fa_o_w4a8.py:118:42
	cvt.u64.u32 	%rd9, %r29;
	mad.wide.u32 	%rd1, %r29, 2, %rd8;
	add.s64 	%rd2, %rd1, 16;
	mov.b32 	%r5, 0;
	.loc	1 118 16                        // sk04_fa_o_w4a8.py:118:16
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
	.loc	2 191 40                        // standard.py:191:40 @[ sk04_fa_o_w4a8.py:119:29 ]
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
	.loc	1 122 27                        // sk04_fa_o_w4a8.py:122:27
	mul.lo.s32 	%r36, %r27, %r20;
	.loc	1 122 21                        // sk04_fa_o_w4a8.py:122:21
	cvt.s64.s32 	%rd10, %r36;
	add.s64 	%rd11, %rd6, %rd10;
	.loc	1 122 39                        // sk04_fa_o_w4a8.py:122:39
	add.s64 	%rd3, %rd11, %rd9;
	.loc	1 118 73                        // sk04_fa_o_w4a8.py:118:73
	mov.b32 	{%rs1, %rs2}, %r9;
	cvt.f32.bf16 	%r37, %rs2;
	cvt.f32.bf16 	%r38, %rs1;
	mov.b32 	{%rs3, %rs4}, %r8;
	cvt.f32.bf16 	%r39, %rs4;
	cvt.f32.bf16 	%r40, %rs3;
	.loc	1 119 36                        // sk04_fa_o_w4a8.py:119:36
	abs.f32 	%r41, %r40;
	abs.f32 	%r42, %r39;
	abs.f32 	%r43, %r38;
	abs.f32 	%r44, %r37;
	.loc	1 118 73                        // sk04_fa_o_w4a8.py:118:73
	mov.b32 	{%rs5, %rs6}, %r7;
	cvt.f32.bf16 	%r45, %rs6;
	cvt.f32.bf16 	%r46, %rs5;
	mov.b32 	{%rs7, %rs8}, %r6;
	cvt.f32.bf16 	%r47, %rs8;
	cvt.f32.bf16 	%r48, %rs7;
	.loc	1 119 36                        // sk04_fa_o_w4a8.py:119:36
	abs.f32 	%r49, %r48;
	abs.f32 	%r50, %r47;
	abs.f32 	%r51, %r46;
	abs.f32 	%r52, %r45;
	.loc	1 118 73                        // sk04_fa_o_w4a8.py:118:73
	mov.b32 	{%rs9, %rs10}, %r4;
	cvt.f32.bf16 	%r53, %rs10;
	cvt.f32.bf16 	%r54, %rs9;
	mov.b32 	{%rs11, %rs12}, %r3;
	cvt.f32.bf16 	%r55, %rs12;
	cvt.f32.bf16 	%r56, %rs11;
	.loc	1 119 36                        // sk04_fa_o_w4a8.py:119:36
	abs.f32 	%r57, %r56;
	abs.f32 	%r58, %r55;
	abs.f32 	%r59, %r54;
	abs.f32 	%r60, %r53;
	.loc	1 118 73                        // sk04_fa_o_w4a8.py:118:73
	mov.b32 	{%rs13, %rs14}, %r2;
	cvt.f32.bf16 	%r61, %rs14;
	cvt.f32.bf16 	%r62, %rs13;
	mov.b32 	{%rs15, %rs16}, %r1;
	cvt.f32.bf16 	%r63, %rs16;
	cvt.f32.bf16 	%r64, %rs15;
	.loc	1 119 36                        // sk04_fa_o_w4a8.py:119:36
	abs.f32 	%r65, %r64;
	abs.f32 	%r66, %r63;
	abs.f32 	%r67, %r62;
	abs.f32 	%r68, %r61;
$L__tmp3:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk04_fa_o_w4a8.py:119:29 ] ]
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
	.loc	2 191 40                        // standard.py:191:40 @[ sk04_fa_o_w4a8.py:119:29 ]
	shfl.sync.bfly.b32 	%r84, %r83, 16, 31, -1;
$L__tmp5:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk04_fa_o_w4a8.py:119:29 ] ]
	max.f32 	%r85, %r83, %r84;
$L__tmp6:
	.loc	2 191 40                        // standard.py:191:40 @[ sk04_fa_o_w4a8.py:119:29 ]
	shfl.sync.bfly.b32 	%r86, %r85, 8, 31, -1;
$L__tmp7:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk04_fa_o_w4a8.py:119:29 ] ]
	max.f32 	%r87, %r85, %r86;
$L__tmp8:
	.loc	2 191 40                        // standard.py:191:40 @[ sk04_fa_o_w4a8.py:119:29 ]
	shfl.sync.bfly.b32 	%r88, %r87, 4, 31, -1;
$L__tmp9:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk04_fa_o_w4a8.py:119:29 ] ]
	max.f32 	%r89, %r87, %r88;
$L__tmp10:
	.loc	2 191 40                        // standard.py:191:40 @[ sk04_fa_o_w4a8.py:119:29 ]
	shfl.sync.bfly.b32 	%r90, %r89, 2, 31, -1;
$L__tmp11:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk04_fa_o_w4a8.py:119:29 ] ]
	max.f32 	%r91, %r89, %r90;
$L__tmp12:
	.loc	2 191 40                        // standard.py:191:40 @[ sk04_fa_o_w4a8.py:119:29 ]
	shfl.sync.bfly.b32 	%r92, %r91, 1, 31, -1;
$L__tmp13:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk04_fa_o_w4a8.py:119:29 ] ]
	max.f32 	%r11, %r91, %r92;
$L__tmp14:
	.loc	2 191 40                        // standard.py:191:40 @[ sk04_fa_o_w4a8.py:119:29 ]
	// begin inline asm
	@%p2 st.shared.b32 [ %r10 + 0 ], %r11;
	// end inline asm
	bar.sync 	0;
	// begin inline asm
	@%p3 ld.shared.b32 %r12, [ %r13 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r93, %r12, 4, 31, -1;
$L__tmp15:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk04_fa_o_w4a8.py:119:29 ] ]
	max.f32 	%r94, %r12, %r93;
$L__tmp16:
	.loc	2 191 40                        // standard.py:191:40 @[ sk04_fa_o_w4a8.py:119:29 ]
	shfl.sync.bfly.b32 	%r95, %r94, 2, 31, -1;
$L__tmp17:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk04_fa_o_w4a8.py:119:29 ] ]
	max.f32 	%r96, %r94, %r95;
$L__tmp18:
	.loc	2 191 40                        // standard.py:191:40 @[ sk04_fa_o_w4a8.py:119:29 ]
	shfl.sync.bfly.b32 	%r97, %r96, 1, 31, -1;
$L__tmp19:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk04_fa_o_w4a8.py:119:29 ] ]
	max.f32 	%r14, %r96, %r97;
$L__tmp20:
	.loc	2 191 40                        // standard.py:191:40 @[ sk04_fa_o_w4a8.py:119:29 ]
	// begin inline asm
	@%p4 st.shared.b32 [ %r13 + 0 ], %r14;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r98, [global_smem];
$L__tmp21:
	.loc	1 119 49                        // sk04_fa_o_w4a8.py:119:49
	max.f32 	%r99, %r98, 0f0DA24260;
	mov.b32 	%r100, 0f42FE0000;
	.loc	1 120 22                        // sk04_fa_o_w4a8.py:120:22
	div.full.f32 	%r101, %r100, %r99;
	.loc	1 120 14                        // sk04_fa_o_w4a8.py:120:14
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
	.loc	1 121 29                        // sk04_fa_o_w4a8.py:121:29
	.loc	1 121 39                        // sk04_fa_o_w4a8.py:121:39
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
	.loc	1 121 14                        // sk04_fa_o_w4a8.py:121:14
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
	.loc	1 121 49                        // sk04_fa_o_w4a8.py:121:49
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
	.loc	1 122 70                        // sk04_fa_o_w4a8.py:122:70
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
	.loc	1 122 77                        // sk04_fa_o_w4a8.py:122:77
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
	.loc	1 122 85                        // sk04_fa_o_w4a8.py:122:85
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
	.loc	1 122 45                        // sk04_fa_o_w4a8.py:122:45
	// begin inline asm
	@%p1 st.global.v4.b32 [ %rd3 + 0 ], { %r15, %r16, %r17, %r18 };
	// end inline asm
	.loc	1 123 21                        // sk04_fa_o_w4a8.py:123:21
	mad.wide.u32 	%rd4, %r20, 4, %rd7;
	.loc	1 123 34                        // sk04_fa_o_w4a8.py:123:34
	mul.f32 	%r19, %r99, 0f3C010204;
	.loc	1 123 26                        // sk04_fa_o_w4a8.py:123:26
	or.b32 	%r206, %r24, %r26;
	setp.eq.b32 	%p5, %r206, 0;
	// begin inline asm
	@%p5 st.global.b32 [ %rd4 + 0 ], { %r19 };
	// end inline asm
	.loc	1 123 4                         // sk04_fa_o_w4a8.py:123:4
	ret;
$L__tmp22:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/home/usuario/Proyectos/genesis-vllm-patches/vllm/_genesis/kernels/sk04_fa_o_w4a8.py"
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
.b32 209                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0xca DW_TAG_compile_unit
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
.b8 52
.b8 95
.b8 102
.b8 97
.b8 95
.b8 111
.b8 95
.b8 119
.b8 52
.b8 97
.b8 56
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
.b8 2                                   // Abbrev [2] 0x6e:0x1f DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 52
.b8 95
.b8 102
.b8 97
.b8 95
.b8 111
.b8 95
.b8 119
.b8 52
.b8 97
.b8 56
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
.b8 3                                   // Abbrev [3] 0x8d:0x47 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 110                                // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0xa2:0x31 DW_TAG_inlined_subroutine
.b32 110                                // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp21                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 119                                 // DW_AT_call_line
.b8 29                                  // DW_AT_call_column
.b8 5                                   // Abbrev [5] 0xba:0x18 DW_TAG_inlined_subroutine
.b32 110                                // DW_AT_abstract_origin
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

_PTX_KERNEL_SK04_QUANT = _Nativo(
    'B2/sk04_fa_o_w4a8.py__quant',
    _PTX_SRC_SK04_QUANT, '_sk04_fa_o_w4a8_quant_kernel',
    warps=8, shared=32,
    abi=[0, 1, 2, 3, 4, 5],
    horneado={6: 4096},
    div16=[],
)

SK_ID = "SK-04-W4A8"
SK_NAME = "FA_O_W4A8"
GLOBAL_SHAPE = (5120, 6144)
RANK_SHAPE = (5120, 3072)
NUM_LAYERS = 17
ROW_PARALLEL = True

GROUP_SIZE: int = 128

# Split-K de decode: reparte K entre CTAs cuando hay pocos bloques de N.
SPLITK_MAX_M: int = 32
SPLITK_BLOCK_N: int = 128
SPLITK_BLOCK_K: int = 128
SPLIT_K: int = 4
# GA102 (RTX 3090) tiene 82 SM. El split-K sólo paga cuando N/BLOCK_N no llega
# a llenar una ola: con N=17408 ya hay 136 CTAs y repartir K sólo agrega
# atómicas y un buffer fp32 intermedio (medido: 0.081 -> 0.100 ms a M=1).
SM_COUNT: int = 82

# (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages) por bucket de M.
# BLOCK_K == GROUP_SIZE == 128 -> una escala de grupo por iteración.
_CFG: tuple[tuple[int, int, int, int, int, int], ...] = (
    # (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages) por bucket de M.
    # Barrido propio de W4A8 en RTX 3090 (K=5120, N=8192), mediana de 5 corridas.
    # Su perfil NO es el de W8A8: aca el kernel esta ALU-bound desempaquetando
    # nibbles, asi que gana con menos warps (4) y mas stages, no con tiles mas
    # grandes. Ganancia sobre la config anterior:
    #   M=32 1.18x   M=128 1.24x   M=512 1.26x   M=1664 1.20x   M=8000 1.23x
    (16, 128, 128, 8, 8, 3),    # M <=   16  decode
    (64, 128, 128, 8, 4, 4),    # M <=  128
    (64, 128, 128, 8, 4, 4),    # M <= 1024
    (128, 256, 64, 8, 8, 3),    # M  > 1024  prefill
)

BLOCK_M: int = 64
BLOCK_N: int = 128
BLOCK_K: int = 128

_ZERO: dict[torch.device, torch.Tensor] = {}


def _zero(device: torch.device) -> torch.Tensor:
    """Escalar cero por device, usado como residual neutro con strides 0."""
    z = _ZERO.get(device)
    if z is None:
        z = torch.zeros((), dtype=torch.bfloat16, device=device)
        _ZERO[device] = z
    return z


def _cfg(m: int) -> tuple[int, int, int, int, int, int]:
    return _CFG[(m > 16) + (m > 128) + (m > 1024)]


@triton.jit
def _sk04_fa_o_w4a8_quant_kernel(x_ptr, q_ptr, s_ptr, K, stride_xm, stride_qm, BLOCK: tl.constexpr):
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
def _sk04_fa_o_w4a8_splitk_kernel(
    a_ptr, w_ptr, out_ptr, a_scale_ptr, w_scale_ptr,
    M, N, K,
    stride_am, stride_ak, stride_wk, stride_wn,
    stride_out_m, stride_out_n,
    stride_ws_g, stride_ws_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """Variante split-K para decode: reparte K entre ``SPLIT_K`` CTAs.

    El grid es (M/BLOCK_M, N/BLOCK_N, SPLIT_K). La dimension de M es
    obligatoria: sin ella el kernel calculaba SOLO las primeras BLOCK_M=16
    filas y las de arriba quedaban sin escribir. Medido antes del fix: a M=17
    salian mal 3 de 17 filas, a M=32 salian mal 19 de 32.

    En decode el GEMM es bandwidth-bound sobre el peso, pero con ``BLOCK_N=128``
    una capa como ``down_proj`` (N=5120) sólo genera 40 CTAs para 82 SM: media
    GPU parada. Repartiendo K en 4 se llega a 160 y el ancho de banda sube de
    32% a 57% del techo de la 3090. Cada trozo aplica sus propias escalas de
    grupo y de fila, así que la reducción entre trozos es una suma directa
    (``tl.atomic_add`` sobre fp32).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    offs_kh = tl.arange(0, BLOCK_K // 2)

    groups = (K // BLOCK_K) // SPLIT_K
    g0 = pid_k * groups
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + (g0 * BLOCK_K + offs_k)[None, :] * stride_ak
    w_ptrs = w_ptr + (g0 * (BLOCK_K // 2) + offs_kh)[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for g in range(groups):
        packed = tl.load(w_ptrs).to(tl.int32)
        w_lo = ((packed & 15) - 8).to(tl.int8)
        w_hi = (((packed >> 4) & 15) - 8).to(tl.int8)
        a_even, a_odd = tl.split(tl.reshape(tl.load(a_ptrs), (BLOCK_M, BLOCK_K // 2, 2)))
        int_acc = tl.dot(a_even, w_lo, out_dtype=tl.int32) + tl.dot(a_odd, w_hi, out_dtype=tl.int32)
        w_scales = tl.load(w_scale_ptr + (g0 + g) * stride_ws_g + offs_n * stride_ws_n).to(tl.float32)
        acc += int_acc.to(tl.float32) * w_scales[None, :]
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += (BLOCK_K // 2) * stride_wk

    acc = acc * tl.load(a_scale_ptr + offs_m).to(tl.float32)[:, None]
    out_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.atomic_add(
        out_ptr + out_m[:, None] * stride_out_m + out_n[None, :] * stride_out_n,
        acc,
        mask=(out_m[:, None] < M) & (out_n[None, :] < N),
    )


@triton.jit
def _sk04_fa_o_w4a8_kernel(
    a_ptr, w_ptr, out_ptr, resid_ptr, a_scale_ptr, w_scale_ptr,
    M, N, K,
    stride_am, stride_ak, stride_wk, stride_wn,
    stride_out_m, stride_out_n, stride_res_m, stride_res_n,
    stride_ws_g, stride_ws_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """GEMM W4A8: activación INT8 per-token, peso INT4 empacado con escala por grupo."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_in_group = GROUP_M * num_pid_n
    first_m = (pid // pid_in_group) * GROUP_M
    group_m = min(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + ((pid % pid_in_group) % group_m)
    pid_n = (pid % pid_in_group) // group_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    offs_kh = tl.arange(0, BLOCK_K // 2)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = w_ptr + offs_kh[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for g in range(0, K // BLOCK_K):
        # Una sola lectura del byte empacado: los dos nibbles salen a la vez.
        packed = tl.load(w_ptrs).to(tl.int32)
        w_lo = ((packed & 15) - 8).to(tl.int8)
        w_hi = (((packed >> 4) & 15) - 8).to(tl.int8)
        # La activación se lee contigua y se parte en pares/impares en registro.
        a_even, a_odd = tl.split(tl.reshape(tl.load(a_ptrs), (BLOCK_M, BLOCK_K // 2, 2)))
        int_acc = tl.dot(a_even, w_lo, out_dtype=tl.int32) + tl.dot(a_odd, w_hi, out_dtype=tl.int32)
        w_scales = tl.load(w_scale_ptr + g * stride_ws_g + offs_n * stride_ws_n).to(tl.float32)
        acc += int_acc.to(tl.float32) * w_scales[None, :]
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += (BLOCK_K // 2) * stride_wk

    acc = acc * tl.load(a_scale_ptr + offs_m).to(tl.float32)[:, None]
    acc += tl.load(resid_ptr + offs_m[:, None] * stride_res_m + offs_n[None, :] * stride_res_n).to(tl.float32)

    out_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(
        out_ptr + out_m[:, None] * stride_out_m + out_n[None, :] * stride_out_n,
        acc.to(out_ptr.dtype.element_ty),
        mask=(out_m[:, None] < M) & (out_n[None, :] < N),
    )


def sk04_fa_o_w4a8_quant_triton(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quant per-token INT8 -> ``(q [M,K] int8, s [M] fp32)``. Camino Triton,
    referencia para tests (bit-exacto contra el PTX embebido)."""
    M, K = hidden.shape
    q = torch.empty((M, K), dtype=torch.int8, device=hidden.device)
    s = torch.empty((M,), dtype=torch.float32, device=hidden.device)
    _sk04_fa_o_w4a8_quant_kernel[(M,)](
        hidden, q, s, K, hidden.stride(0), q.stride(0),
        BLOCK=triton.next_power_of_2(K), num_warps=8, num_stages=1,
    )
    return q, s


def sk04_fa_o_w4a8_quant(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quant per-token INT8 -> ``(q [M,K] int8, s [M] fp32)``. Camino de
    producción: PTX nativo embebido (libcuda cruda, sin JIT de Triton)."""
    M, K = hidden.shape
    q = torch.empty((M, K), dtype=torch.int8, device=hidden.device)
    s = torch.empty((M,), dtype=torch.float32, device=hidden.device)
    if habilitado():
        _PTX_KERNEL_SK04_QUANT(
            (M,), hidden, q, s, K, hidden.stride(0), q.stride(0),
            triton.next_power_of_2(K),
        )
        return q, s
    return sk04_fa_o_w4a8_quant_triton(hidden)


def sk04_fa_o_w4a8_gemm(
    a: torch.Tensor,
    w_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_scales: torch.Tensor,
    residual: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """GEMM W4A8. ``a`` int8 [M,K], ``w_packed`` int8 [K/2,N], ``w_scales`` [K/128,N]."""
    M, K = a.shape
    N = w_packed.shape[1]
    res = _zero(a.device) if residual is None else residual
    rm, rn = (0, 0) if residual is None else (res.stride(0), res.stride(1))
    # Decode: split-K para no dejar SMs ociosos cuando N/BLOCK_N < 2 olas.
    if (
        M <= SPLITK_MAX_M
        and triton.cdiv(N, SPLITK_BLOCK_N) < SM_COUNT
        and (K // SPLITK_BLOCK_K) % SPLIT_K == 0
    ):
        acc = (
            torch.zeros((M, N), dtype=torch.float32, device=a.device)
            if residual is None
            else residual.to(torch.float32).expand(M, N).contiguous()
        )
        _sk04_fa_o_w4a8_splitk_kernel[(triton.cdiv(M, 16), triton.cdiv(N, SPLITK_BLOCK_N), SPLIT_K)](
            a, w_packed, acc, a_scales, w_scales,
            M, N, K,
            a.stride(0), a.stride(1), w_packed.stride(0), w_packed.stride(1),
            acc.stride(0), acc.stride(1),
            w_scales.stride(0), w_scales.stride(1),
            BLOCK_M=16, BLOCK_N=SPLITK_BLOCK_N, BLOCK_K=SPLITK_BLOCK_K, SPLIT_K=SPLIT_K,
            num_warps=4, num_stages=3,
        )
        return acc.to(out_dtype)
    bm, bn, bk, gm, warps, stages = _cfg(M)
    out = torch.empty((M, N), dtype=out_dtype, device=a.device)
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    _lanzar(grid, (bm, bn, bk, gm, warps, stages), False,
            a,
            w_packed,
            out,
            res,
            a_scales,
            w_scales,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            w_packed.stride(0),
            w_packed.stride(1),
            out.stride(0),
            out.stride(1),
            rm,
            rn,
            w_scales.stride(0),
            w_scales.stride(1),
            bm,
            bn,
            bk,
            gm)
    return out


def sk04_fa_o_w4a8_proj(
    attn_out: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    residual: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    do_allreduce: bool = True,
) -> torch.Tensor:
    """Capa completa FA o_proj W4A8: quant -> GEMM -> all_reduce -> residual."""
    a, a_scales = sk04_fa_o_w4a8_quant(attn_out)
    if not do_allreduce:
        return sk04_fa_o_w4a8_gemm(a, w_packed, a_scales, w_scales, residual, out_dtype)
    out = sk04_fa_o_w4a8_gemm(a, w_packed, a_scales, w_scales, None, out_dtype)
    torch.distributed.all_reduce(out)
    out.add_(residual)
    return out


__all__ = [
    "SK_ID", "SK_NAME", "GLOBAL_SHAPE", "RANK_SHAPE", "NUM_LAYERS", "ROW_PARALLEL",
    "GROUP_SIZE", "BLOCK_M", "BLOCK_N", "BLOCK_K", "SPLIT_K", "SPLITK_MAX_M", "SM_COUNT",
    "sk04_fa_o_w4a8_quant", "sk04_fa_o_w4a8_gemm", "sk04_fa_o_w4a8_proj",
]


# --- variantes PTX embebidas ---

_PTX_0 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk04_fa_o_w4a8_kernel  // -- Begin function _sk04_fa_o_w4a8_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk04_fa_o_w4a8_kernel
.visible .entry _sk04_fa_o_w4a8_kernel(
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_5,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_6,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_7,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_8,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_9,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_10,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_11,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_12,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_13,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_14,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_15
)
.reqntid 256
{
	.reg .pred 	%p<11>;
	.reg .b16 	%rs<177>;
	.reg .b32 	%r<478>;
	.reg .b64 	%rd<57>;
	.loc	1 797 0                         // sk04_fa_o_w4a8.py:797:0
$L__func_begin0:
	.loc	1 797 0                         // sk04_fa_o_w4a8.py:797:0

// %bb.0:
	ld.param.b32 	%r28, [_sk04_fa_o_w4a8_kernel_param_12];
	ld.param.b32 	%r27, [_sk04_fa_o_w4a8_kernel_param_11];
	ld.param.b32 	%r26, [_sk04_fa_o_w4a8_kernel_param_8];
	ld.param.b32 	%r25, [_sk04_fa_o_w4a8_kernel_param_7];
	ld.param.b32 	%r24, [_sk04_fa_o_w4a8_kernel_param_6];
	ld.param.b64 	%rd17, [_sk04_fa_o_w4a8_kernel_param_4];
	ld.param.b64 	%rd16, [_sk04_fa_o_w4a8_kernel_param_3];
	ld.param.b64 	%rd15, [_sk04_fa_o_w4a8_kernel_param_2];
	ld.param.b64 	%rd14, [_sk04_fa_o_w4a8_kernel_param_1];
	ld.param.b64 	%rd13, [_sk04_fa_o_w4a8_kernel_param_0];
$L__tmp0:
	.loc	1 807 24                        // sk04_fa_o_w4a8.py:807:24
	mov.u32 	%r42, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk04_fa_o_w4a8.py:808:27 ]
	add.s32 	%r43, %r24, 15;
	.loc	2 43 30                         // standard.py:43:30 @[ sk04_fa_o_w4a8.py:808:27 ]
	shr.s32 	%r44, %r43, 31;
	shr.u32 	%r45, %r44, 28;
	add.s32 	%r46, %r43, %r45;
	shr.s32 	%r47, %r46, 4;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk04_fa_o_w4a8.py:809:27 ]
	add.s32 	%r48, %r25, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk04_fa_o_w4a8.py:809:27 ]
	shr.s32 	%r49, %r48, 31;
	shr.u32 	%r50, %r49, 25;
	add.s32 	%r51, %r48, %r50;
	shr.s32 	%r52, %r51, 7;
$L__tmp3:
	.loc	1 810 29                        // sk04_fa_o_w4a8.py:810:29
	shl.b32 	%r53, %r52, 3;
	.loc	1 811 22                        // sk04_fa_o_w4a8.py:811:22
	div.s32 	%r54, %r42, %r53;
	.loc	1 811 38                        // sk04_fa_o_w4a8.py:811:38
	shl.b32 	%r55, %r54, 3;
	ld.param.b32 	%r56, [_sk04_fa_o_w4a8_kernel_param_9];
	.loc	1 812 30                        // sk04_fa_o_w4a8.py:812:30
	sub.s32 	%r57, %r47, %r55;
	ld.param.b32 	%r58, [_sk04_fa_o_w4a8_kernel_param_10];
	.loc	1 812 39                        // sk04_fa_o_w4a8.py:812:39
	min.s32 	%r59, %r57, 8;
	.loc	1 813 30                        // sk04_fa_o_w4a8.py:813:30
	mul.lo.s32 	%r60, %r54, %r53;
	sub.s32 	%r61, %r42, %r60;
	.loc	1 814 36                        // sk04_fa_o_w4a8.py:814:36
	div.s32 	%r62, %r61, %r59;
	.loc	1 813 46                        // sk04_fa_o_w4a8.py:813:46
	mul.lo.s32 	%r63, %r62, %r59;
	sub.s32 	%r64, %r61, %r63;
	.loc	1 813 23                        // sk04_fa_o_w4a8.py:813:23
	add.s32 	%r65, %r64, %r55;
	.loc	1 816 22                        // sk04_fa_o_w4a8.py:816:22
	shl.b32 	%r1, %r65, 4;
	.loc	1 816 45                        // sk04_fa_o_w4a8.py:816:45
	mov.u32 	%r2, %tid.x;
	and.b32 	%r3, %r2, 240;
	bfe.u32 	%r66, %r2, 4, 4;
	and.b32 	%r4, %r2, 28;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r5, %r1, %r66;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r6, %r5, %r24;
	.loc	1 817 22                        // sk04_fa_o_w4a8.py:817:22
	shl.b32 	%r7, %r62, 7;
	.loc	1 817 45                        // sk04_fa_o_w4a8.py:817:45
	shl.b32 	%r8, %r2, 3;
	and.b32 	%r67, %r8, 120;
	and.b32 	%r9, %r2, 224;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r10, %r7, %r67;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r68, %r10, %r25;
	.loc	1 821 39                        // sk04_fa_o_w4a8.py:821:39
	mul.lo.s32 	%r69, %r6, %r56;
	.loc	1 821 21                        // sk04_fa_o_w4a8.py:821:21
	cvt.s64.s32 	%rd1, %r69;
	add.s64 	%rd29, %rd13, %rd1;
	.loc	1 821 51                        // sk04_fa_o_w4a8.py:821:51
	cvt.u64.u32 	%rd2, %r67;
	add.s64 	%rd23, %rd29, %rd2;
	.loc	1 822 40                        // sk04_fa_o_w4a8.py:822:40
	mul.lo.s32 	%r70, %r58, %r66;
	shl.b32 	%r71, %r58, 4;
	add.s32 	%r72, %r70, %r71;
	shl.b32 	%r73, %r58, 5;
	add.s32 	%r74, %r70, %r73;
	mad.lo.s32 	%r75, %r58, 48, %r70;
	.loc	1 822 21                        // sk04_fa_o_w4a8.py:822:21
	cvt.s64.s32 	%rd3, %r70;
	add.s64 	%rd30, %rd14, %rd3;
	cvt.s64.s32 	%rd4, %r72;
	add.s64 	%rd31, %rd14, %rd4;
	cvt.s64.s32 	%rd5, %r74;
	add.s64 	%rd32, %rd14, %rd5;
	cvt.s64.s32 	%rd6, %r75;
	add.s64 	%rd33, %rd14, %rd6;
	.loc	1 822 52                        // sk04_fa_o_w4a8.py:822:52
	cvt.s64.s32 	%rd7, %r68;
	add.s64 	%rd19, %rd30, %rd7;
	add.s64 	%rd20, %rd31, %rd7;
	add.s64 	%rd21, %rd32, %rd7;
	add.s64 	%rd22, %rd33, %rd7;
	.loc	1 836 35                        // sk04_fa_o_w4a8.py:836:35
	shl.b32 	%r76, %r58, 6;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	setp.lt.s32 	%p1, %r26, 128;
	setp.gt.s32 	%p2, %r26, 127;
	.loc	1 827 25                        // sk04_fa_o_w4a8.py:827:25
	and.b32 	%r77, %r8, 2040;
	shr.u32 	%r78, %r3, 1;
	xor.b32 	%r79, %r77, %r78;
	mov.b32 	%r80, global_smem;
	add.s32 	%r30, %r80, %r79;
	selp.b32 	%r31, 8, 0, %p2;
	// begin inline asm
	cp.async.ca.shared.global [ %r30 + 0 ], [ %rd19 + 0 ], 0x8, %r31;
	// end inline asm
	add.s32 	%r32, %r30, 2048;
	// begin inline asm
	cp.async.ca.shared.global [ %r32 + 0 ], [ %rd20 + 0 ], 0x8, %r31;
	// end inline asm
	add.s32 	%r33, %r30, 4096;
	// begin inline asm
	cp.async.ca.shared.global [ %r33 + 0 ], [ %rd21 + 0 ], 0x8, %r31;
	// end inline asm
	add.s32 	%r34, %r30, 6144;
	// begin inline asm
	cp.async.ca.shared.global [ %r34 + 0 ], [ %rd22 + 0 ], 0x8, %r31;
	// end inline asm
	cp.async.commit_group;
	.loc	1 831 52                        // sk04_fa_o_w4a8.py:831:52
	shl.b32 	%r81, %r2, 1;
	and.b32 	%r82, %r81, 96;
	xor.b32 	%r83, %r77, %r82;
	add.s32 	%r11, %r80, %r83;
	add.s32 	%r35, %r11, 16384;
	// begin inline asm
	cp.async.ca.shared.global [ %r35 + 0 ], [ %rd23 + 0 ], 0x8, %r31;
	// end inline asm
	cp.async.commit_group;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	setp.gt.s32 	%p3, %r26, 255;
	.loc	1 835 18                        // sk04_fa_o_w4a8.py:835:18
	add.s64 	%rd28, %rd23, 128;
	.loc	1 836 18                        // sk04_fa_o_w4a8.py:836:18
	cvt.s64.s32 	%rd8, %r76;
	add.s64 	%rd24, %rd19, %rd8;
	add.s64 	%rd25, %rd20, %rd8;
	add.s64 	%rd26, %rd21, %rd8;
	add.s64 	%rd27, %rd22, %rd8;
	.loc	1 827 25                        // sk04_fa_o_w4a8.py:827:25
	bar.sync 	0;
	add.s32 	%r36, %r30, 8192;
	selp.b32 	%r37, 8, 0, %p3;
	// begin inline asm
	cp.async.ca.shared.global [ %r36 + 0 ], [ %rd24 + 0 ], 0x8, %r37;
	// end inline asm
	add.s32 	%r38, %r30, 10240;
	// begin inline asm
	cp.async.ca.shared.global [ %r38 + 0 ], [ %rd25 + 0 ], 0x8, %r37;
	// end inline asm
	add.s32 	%r39, %r30, 12288;
	// begin inline asm
	cp.async.ca.shared.global [ %r39 + 0 ], [ %rd26 + 0 ], 0x8, %r37;
	// end inline asm
	add.s32 	%r40, %r30, 14336;
	// begin inline asm
	cp.async.ca.shared.global [ %r40 + 0 ], [ %rd27 + 0 ], 0x8, %r37;
	// end inline asm
	cp.async.commit_group;
	.loc	1 831 52                        // sk04_fa_o_w4a8.py:831:52
	add.s32 	%r41, %r11, 18432;
	// begin inline asm
	cp.async.ca.shared.global [ %r41 + 0 ], [ %rd28 + 0 ], 0x8, %r37;
	// end inline asm
	cp.async.commit_group;
	mov.b32 	%r470, 0f00000000;
	mov.b32 	%r471, %r470;
	mov.b32 	%r472, %r470;
	mov.b32 	%r473, %r470;
	mov.b32 	%r474, %r470;
	mov.b32 	%r475, %r470;
	mov.b32 	%r476, %r470;
	mov.b32 	%r477, %r470;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	@%p1 bra 	$L__BB0_3;
// %bb.1:                               // %.lr.ph
	.loc	1 0 22                          // sk04_fa_o_w4a8.py:0:22
	ld.param.b32 	%r29, [_sk04_fa_o_w4a8_kernel_param_13];
	ld.param.b64 	%rd18, [_sk04_fa_o_w4a8_kernel_param_5];
	cvt.u32.u64 	%r84, %rd2;
	.loc	1 825 27                        // sk04_fa_o_w4a8.py:825:27
	shr.u32 	%r85, %r26, 7;
	.loc	1 817 45                        // sk04_fa_o_w4a8.py:817:45
	and.b32 	%r86, %r2, 3;
	shl.b32 	%r87, %r86, 1;
	shr.u32 	%r88, %r9, 2;
	or.b32 	%r89, %r87, %r88;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r90, %r89, %r7;
	or.b32 	%r91, %r90, 64;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r92, %r91, %r25;
	rem.s32 	%r93, %r90, %r25;
	add.s32 	%r94, %r85, -2;
	bfe.u32 	%r95, %r2, 2, 6;
	mul.lo.s32 	%r96, %r86, 544;
	xor.b32 	%r12, %r96, %r95;
	xor.b32 	%r13, %r12, 136;
	xor.b32 	%r14, %r12, 272;
	xor.b32 	%r15, %r12, 408;
	xor.b32 	%r16, %r12, 64;
	xor.b32 	%r17, %r12, 200;
	xor.b32 	%r18, %r12, 336;
	xor.b32 	%r19, %r12, 472;
	shl.b32 	%r97, %r4, 5;
	or.b32 	%r20, %r97, %r84;
	xor.b32 	%r21, %r20, 32;
	xor.b32 	%r22, %r20, 64;
	xor.b32 	%r23, %r20, 96;
	cvt.s64.s32 	%rd9, %r93;
	cvt.s64.s32 	%rd10, %r92;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	cvt.s64.s32 	%rd11, %r94;
	cvt.u64.u32 	%rd12, %r85;
	shl.b64 	%rd34, %rd8, 1;
	add.s64 	%rd35, %rd34, %rd7;
	add.s64 	%rd55, %rd14, %rd35;
	add.s64 	%rd36, %rd2, %rd1;
	add.s64 	%rd37, %rd36, %rd13;
	add.s64 	%rd54, %rd37, 256;
	mov.b32 	%r470, 0f00000000;
	mov.b32 	%r469, 1;
	mov.b32 	%r468, -1;
	mov.b64 	%rd56, 0;
	mov.b32 	%r98, 0;
	shl.b64 	%rd45, %rd9, 2;
	shl.b64 	%rd46, %rd10, 2;
	mov.b32 	%r467, %r98;
	mov.b32 	%r471, %r470;
	mov.b32 	%r472, %r470;
	mov.b32 	%r473, %r470;
	mov.b32 	%r474, %r470;
	mov.b32 	%r475, %r470;
	mov.b32 	%r476, %r470;
	mov.b32 	%r477, %r470;
$L__BB0_2:                              // =>This Inner Loop Header: Depth=1
	setp.lt.s64 	%p4, %rd56, %rd11;
	add.s32 	%r149, %r468, 1;
	setp.gt.s32 	%p5, %r149, 1;
	selp.b32 	%r468, 0, %r149, %p5;
	.loc	1 827 25                        // sk04_fa_o_w4a8.py:827:25
	cp.async.wait_group 	2;
	bar.sync 	0;
	shl.b32 	%r150, %r468, 13;
	add.s32 	%r151, %r80, %r150;
	.loc	1 828 38                        // sk04_fa_o_w4a8.py:828:38
	add.s32 	%r152, %r151, %r12;
	ld.shared.b8 	%rs1, [%r152];
	ld.shared.b8 	%rs2, [%r152+2048];
	ld.shared.b8 	%rs3, [%r152+4096];
	ld.shared.b8 	%rs4, [%r152+6144];
	add.s32 	%r153, %r151, %r13;
	ld.shared.b8 	%rs5, [%r153];
	ld.shared.b8 	%rs6, [%r153+2048];
	ld.shared.b8 	%rs7, [%r153+4096];
	ld.shared.b8 	%rs8, [%r153+6144];
	add.s32 	%r154, %r151, %r14;
	ld.shared.b8 	%rs9, [%r154];
	ld.shared.b8 	%rs10, [%r154+2048];
	ld.shared.b8 	%rs11, [%r154+4096];
	ld.shared.b8 	%rs12, [%r154+6144];
	add.s32 	%r155, %r151, %r15;
	ld.shared.b8 	%rs13, [%r155];
	ld.shared.b8 	%rs14, [%r155+2048];
	ld.shared.b8 	%rs15, [%r155+4096];
	ld.shared.b8 	%rs16, [%r155+6144];
	add.s32 	%r156, %r151, %r16;
	ld.shared.b8 	%rs17, [%r156];
	ld.shared.b8 	%rs18, [%r156+2048];
	ld.shared.b8 	%rs19, [%r156+4096];
	ld.shared.b8 	%rs20, [%r156+6144];
	add.s32 	%r157, %r151, %r17;
	ld.shared.b8 	%rs21, [%r157];
	ld.shared.b8 	%rs22, [%r157+2048];
	ld.shared.b8 	%rs23, [%r157+4096];
	ld.shared.b8 	%rs24, [%r157+6144];
	add.s32 	%r158, %r151, %r18;
	ld.shared.b8 	%rs25, [%r158];
	ld.shared.b8 	%rs26, [%r158+2048];
	ld.shared.b8 	%rs27, [%r158+4096];
	ld.shared.b8 	%rs28, [%r158+6144];
	add.s32 	%r159, %r151, %r19;
	ld.shared.b8 	%rs29, [%r159];
	ld.shared.b8 	%rs30, [%r159+2048];
	ld.shared.b8 	%rs31, [%r159+4096];
	ld.shared.b8 	%rs32, [%r159+6144];
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r160, %rs13;
	cvt.u32.u16 	%r161, %rs9;
	prmt.b32 	%r162, %r161, %r160, 0x3340U;
	cvt.u32.u16 	%r163, %rs5;
	cvt.u32.u16 	%r164, %rs1;
	prmt.b32 	%r165, %r164, %r163, 0x3340U;
	prmt.b32 	%r166, %r165, %r162, 0x5410U;
	and.b32 	%r167, %r166, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r168, %r167, 0, 0x7773U;
	cvt.u16.u32 	%rs33, %r168;
	add.s16 	%rs34, %rs33, -8;
	cvt.u32.u16 	%r169, %rs34;
	prmt.b32 	%r170, %r167, 0, 0x7772U;
	cvt.u16.u32 	%rs35, %r170;
	add.s16 	%rs36, %rs35, -8;
	cvt.u32.u16 	%r171, %rs36;
	prmt.b32 	%r172, %r171, %r169, 0x3340U;
	prmt.b32 	%r173, %r167, 0, 0x7771U;
	cvt.u16.u32 	%rs37, %r173;
	add.s16 	%rs38, %rs37, -8;
	cvt.u32.u16 	%r174, %rs38;
	prmt.b32 	%r175, %r167, 0, 0x7770U;
	cvt.u16.u32 	%rs39, %r175;
	add.s16 	%rs40, %rs39, -8;
	cvt.u32.u16 	%r176, %rs40;
	prmt.b32 	%r177, %r176, %r174, 0x3340U;
	prmt.b32 	%r99, %r177, %r172, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r178, %rs14;
	cvt.u32.u16 	%r179, %rs10;
	prmt.b32 	%r180, %r179, %r178, 0x3340U;
	cvt.u32.u16 	%r181, %rs6;
	cvt.u32.u16 	%r182, %rs2;
	prmt.b32 	%r183, %r182, %r181, 0x3340U;
	prmt.b32 	%r184, %r183, %r180, 0x5410U;
	and.b32 	%r185, %r184, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r186, %r185, 0, 0x7773U;
	cvt.u16.u32 	%rs41, %r186;
	add.s16 	%rs42, %rs41, -8;
	cvt.u32.u16 	%r187, %rs42;
	prmt.b32 	%r188, %r185, 0, 0x7772U;
	cvt.u16.u32 	%rs43, %r188;
	add.s16 	%rs44, %rs43, -8;
	cvt.u32.u16 	%r189, %rs44;
	prmt.b32 	%r190, %r189, %r187, 0x3340U;
	prmt.b32 	%r191, %r185, 0, 0x7771U;
	cvt.u16.u32 	%rs45, %r191;
	add.s16 	%rs46, %rs45, -8;
	cvt.u32.u16 	%r192, %rs46;
	prmt.b32 	%r193, %r185, 0, 0x7770U;
	cvt.u16.u32 	%rs47, %r193;
	add.s16 	%rs48, %rs47, -8;
	cvt.u32.u16 	%r194, %rs48;
	prmt.b32 	%r195, %r194, %r192, 0x3340U;
	prmt.b32 	%r100, %r195, %r190, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r196, %rs15;
	cvt.u32.u16 	%r197, %rs11;
	prmt.b32 	%r198, %r197, %r196, 0x3340U;
	cvt.u32.u16 	%r199, %rs7;
	cvt.u32.u16 	%r200, %rs3;
	prmt.b32 	%r201, %r200, %r199, 0x3340U;
	prmt.b32 	%r202, %r201, %r198, 0x5410U;
	and.b32 	%r203, %r202, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r204, %r203, 0, 0x7773U;
	cvt.u16.u32 	%rs49, %r204;
	add.s16 	%rs50, %rs49, -8;
	cvt.u32.u16 	%r205, %rs50;
	prmt.b32 	%r206, %r203, 0, 0x7772U;
	cvt.u16.u32 	%rs51, %r206;
	add.s16 	%rs52, %rs51, -8;
	cvt.u32.u16 	%r207, %rs52;
	prmt.b32 	%r208, %r207, %r205, 0x3340U;
	prmt.b32 	%r209, %r203, 0, 0x7771U;
	cvt.u16.u32 	%rs53, %r209;
	add.s16 	%rs54, %rs53, -8;
	cvt.u32.u16 	%r210, %rs54;
	prmt.b32 	%r211, %r203, 0, 0x7770U;
	cvt.u16.u32 	%rs55, %r211;
	add.s16 	%rs56, %rs55, -8;
	cvt.u32.u16 	%r212, %rs56;
	prmt.b32 	%r213, %r212, %r210, 0x3340U;
	prmt.b32 	%r111, %r213, %r208, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r214, %rs16;
	cvt.u32.u16 	%r215, %rs12;
	prmt.b32 	%r216, %r215, %r214, 0x3340U;
	cvt.u32.u16 	%r217, %rs8;
	cvt.u32.u16 	%r218, %rs4;
	prmt.b32 	%r219, %r218, %r217, 0x3340U;
	prmt.b32 	%r220, %r219, %r216, 0x5410U;
	and.b32 	%r221, %r220, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r222, %r221, 0, 0x7773U;
	cvt.u16.u32 	%rs57, %r222;
	add.s16 	%rs58, %rs57, -8;
	cvt.u32.u16 	%r223, %rs58;
	prmt.b32 	%r224, %r221, 0, 0x7772U;
	cvt.u16.u32 	%rs59, %r224;
	add.s16 	%rs60, %rs59, -8;
	cvt.u32.u16 	%r225, %rs60;
	prmt.b32 	%r226, %r225, %r223, 0x3340U;
	prmt.b32 	%r227, %r221, 0, 0x7771U;
	cvt.u16.u32 	%rs61, %r227;
	add.s16 	%rs62, %rs61, -8;
	cvt.u32.u16 	%r228, %rs62;
	prmt.b32 	%r229, %r221, 0, 0x7770U;
	cvt.u16.u32 	%rs63, %r229;
	add.s16 	%rs64, %rs63, -8;
	cvt.u32.u16 	%r230, %rs64;
	prmt.b32 	%r231, %r230, %r228, 0x3340U;
	prmt.b32 	%r112, %r231, %r226, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r232, %rs29;
	cvt.u32.u16 	%r233, %rs25;
	prmt.b32 	%r234, %r233, %r232, 0x3340U;
	cvt.u32.u16 	%r235, %rs21;
	cvt.u32.u16 	%r236, %rs17;
	prmt.b32 	%r237, %r236, %r235, 0x3340U;
	prmt.b32 	%r238, %r237, %r234, 0x5410U;
	and.b32 	%r239, %r238, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r240, %r239, 0, 0x7773U;
	cvt.u16.u32 	%rs65, %r240;
	add.s16 	%rs66, %rs65, -8;
	cvt.u32.u16 	%r241, %rs66;
	prmt.b32 	%r242, %r239, 0, 0x7772U;
	cvt.u16.u32 	%rs67, %r242;
	add.s16 	%rs68, %rs67, -8;
	cvt.u32.u16 	%r243, %rs68;
	prmt.b32 	%r244, %r243, %r241, 0x3340U;
	prmt.b32 	%r245, %r239, 0, 0x7771U;
	cvt.u16.u32 	%rs69, %r245;
	add.s16 	%rs70, %rs69, -8;
	cvt.u32.u16 	%r246, %rs70;
	prmt.b32 	%r247, %r239, 0, 0x7770U;
	cvt.u16.u32 	%rs71, %r247;
	add.s16 	%rs72, %rs71, -8;
	cvt.u32.u16 	%r248, %rs72;
	prmt.b32 	%r249, %r248, %r246, 0x3340U;
	prmt.b32 	%r105, %r249, %r244, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r250, %rs30;
	cvt.u32.u16 	%r251, %rs26;
	prmt.b32 	%r252, %r251, %r250, 0x3340U;
	cvt.u32.u16 	%r253, %rs22;
	cvt.u32.u16 	%r254, %rs18;
	prmt.b32 	%r255, %r254, %r253, 0x3340U;
	prmt.b32 	%r256, %r255, %r252, 0x5410U;
	and.b32 	%r257, %r256, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r258, %r257, 0, 0x7773U;
	cvt.u16.u32 	%rs73, %r258;
	add.s16 	%rs74, %rs73, -8;
	cvt.u32.u16 	%r259, %rs74;
	prmt.b32 	%r260, %r257, 0, 0x7772U;
	cvt.u16.u32 	%rs75, %r260;
	add.s16 	%rs76, %rs75, -8;
	cvt.u32.u16 	%r261, %rs76;
	prmt.b32 	%r262, %r261, %r259, 0x3340U;
	prmt.b32 	%r263, %r257, 0, 0x7771U;
	cvt.u16.u32 	%rs77, %r263;
	add.s16 	%rs78, %rs77, -8;
	cvt.u32.u16 	%r264, %rs78;
	prmt.b32 	%r265, %r257, 0, 0x7770U;
	cvt.u16.u32 	%rs79, %r265;
	add.s16 	%rs80, %rs79, -8;
	cvt.u32.u16 	%r266, %rs80;
	prmt.b32 	%r267, %r266, %r264, 0x3340U;
	prmt.b32 	%r106, %r267, %r262, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r268, %rs31;
	cvt.u32.u16 	%r269, %rs27;
	prmt.b32 	%r270, %r269, %r268, 0x3340U;
	cvt.u32.u16 	%r271, %rs23;
	cvt.u32.u16 	%r272, %rs19;
	prmt.b32 	%r273, %r272, %r271, 0x3340U;
	prmt.b32 	%r274, %r273, %r270, 0x5410U;
	and.b32 	%r275, %r274, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r276, %r275, 0, 0x7773U;
	cvt.u16.u32 	%rs81, %r276;
	add.s16 	%rs82, %rs81, -8;
	cvt.u32.u16 	%r277, %rs82;
	prmt.b32 	%r278, %r275, 0, 0x7772U;
	cvt.u16.u32 	%rs83, %r278;
	add.s16 	%rs84, %rs83, -8;
	cvt.u32.u16 	%r279, %rs84;
	prmt.b32 	%r280, %r279, %r277, 0x3340U;
	prmt.b32 	%r281, %r275, 0, 0x7771U;
	cvt.u16.u32 	%rs85, %r281;
	add.s16 	%rs86, %rs85, -8;
	cvt.u32.u16 	%r282, %rs86;
	prmt.b32 	%r283, %r275, 0, 0x7770U;
	cvt.u16.u32 	%rs87, %r283;
	add.s16 	%rs88, %rs87, -8;
	cvt.u32.u16 	%r284, %rs88;
	prmt.b32 	%r285, %r284, %r282, 0x3340U;
	prmt.b32 	%r121, %r285, %r280, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r286, %rs32;
	cvt.u32.u16 	%r287, %rs28;
	prmt.b32 	%r288, %r287, %r286, 0x3340U;
	cvt.u32.u16 	%r289, %rs24;
	cvt.u32.u16 	%r290, %rs20;
	prmt.b32 	%r291, %r290, %r289, 0x3340U;
	prmt.b32 	%r292, %r291, %r288, 0x5410U;
	and.b32 	%r293, %r292, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r294, %r293, 0, 0x7773U;
	cvt.u16.u32 	%rs89, %r294;
	add.s16 	%rs90, %rs89, -8;
	cvt.u32.u16 	%r295, %rs90;
	prmt.b32 	%r296, %r293, 0, 0x7772U;
	cvt.u16.u32 	%rs91, %r296;
	add.s16 	%rs92, %rs91, -8;
	cvt.u32.u16 	%r297, %rs92;
	prmt.b32 	%r298, %r297, %r295, 0x3340U;
	prmt.b32 	%r299, %r293, 0, 0x7771U;
	cvt.u16.u32 	%rs93, %r299;
	add.s16 	%rs94, %rs93, -8;
	cvt.u32.u16 	%r300, %rs94;
	prmt.b32 	%r301, %r293, 0, 0x7770U;
	cvt.u16.u32 	%rs95, %r301;
	add.s16 	%rs96, %rs95, -8;
	cvt.u32.u16 	%r302, %rs96;
	prmt.b32 	%r303, %r302, %r300, 0x3340U;
	prmt.b32 	%r122, %r303, %r298, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs97, %rs1, 4;
	shr.u16 	%rs98, %rs5, 4;
	shr.u16 	%rs99, %rs9, 4;
	shr.u16 	%rs100, %rs13, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs101, %rs100, -8;
	cvt.u32.u16 	%r304, %rs101;
	add.s16 	%rs102, %rs99, -8;
	cvt.u32.u16 	%r305, %rs102;
	prmt.b32 	%r306, %r305, %r304, 0x3340U;
	add.s16 	%rs103, %rs98, -8;
	cvt.u32.u16 	%r307, %rs103;
	add.s16 	%rs104, %rs97, -8;
	cvt.u32.u16 	%r308, %rs104;
	prmt.b32 	%r309, %r308, %r307, 0x3340U;
	prmt.b32 	%r123, %r309, %r306, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs105, %rs2, 4;
	shr.u16 	%rs106, %rs6, 4;
	shr.u16 	%rs107, %rs10, 4;
	shr.u16 	%rs108, %rs14, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs109, %rs108, -8;
	cvt.u32.u16 	%r310, %rs109;
	add.s16 	%rs110, %rs107, -8;
	cvt.u32.u16 	%r311, %rs110;
	prmt.b32 	%r312, %r311, %r310, 0x3340U;
	add.s16 	%rs111, %rs106, -8;
	cvt.u32.u16 	%r313, %rs111;
	add.s16 	%rs112, %rs105, -8;
	cvt.u32.u16 	%r314, %rs112;
	prmt.b32 	%r315, %r314, %r313, 0x3340U;
	prmt.b32 	%r124, %r315, %r312, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs113, %rs3, 4;
	shr.u16 	%rs114, %rs7, 4;
	shr.u16 	%rs115, %rs11, 4;
	shr.u16 	%rs116, %rs15, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs117, %rs116, -8;
	cvt.u32.u16 	%r316, %rs117;
	add.s16 	%rs118, %rs115, -8;
	cvt.u32.u16 	%r317, %rs118;
	prmt.b32 	%r318, %r317, %r316, 0x3340U;
	add.s16 	%rs119, %rs114, -8;
	cvt.u32.u16 	%r319, %rs119;
	add.s16 	%rs120, %rs113, -8;
	cvt.u32.u16 	%r320, %rs120;
	prmt.b32 	%r321, %r320, %r319, 0x3340U;
	prmt.b32 	%r131, %r321, %r318, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs121, %rs4, 4;
	shr.u16 	%rs122, %rs8, 4;
	shr.u16 	%rs123, %rs12, 4;
	shr.u16 	%rs124, %rs16, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs125, %rs124, -8;
	cvt.u32.u16 	%r322, %rs125;
	add.s16 	%rs126, %rs123, -8;
	cvt.u32.u16 	%r323, %rs126;
	prmt.b32 	%r324, %r323, %r322, 0x3340U;
	add.s16 	%rs127, %rs122, -8;
	cvt.u32.u16 	%r325, %rs127;
	add.s16 	%rs128, %rs121, -8;
	cvt.u32.u16 	%r326, %rs128;
	prmt.b32 	%r327, %r326, %r325, 0x3340U;
	prmt.b32 	%r132, %r327, %r324, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs129, %rs17, 4;
	shr.u16 	%rs130, %rs21, 4;
	shr.u16 	%rs131, %rs25, 4;
	shr.u16 	%rs132, %rs29, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs133, %rs132, -8;
	cvt.u32.u16 	%r328, %rs133;
	add.s16 	%rs134, %rs131, -8;
	cvt.u32.u16 	%r329, %rs134;
	prmt.b32 	%r330, %r329, %r328, 0x3340U;
	add.s16 	%rs135, %rs130, -8;
	cvt.u32.u16 	%r331, %rs135;
	add.s16 	%rs136, %rs129, -8;
	cvt.u32.u16 	%r332, %rs136;
	prmt.b32 	%r333, %r332, %r331, 0x3340U;
	prmt.b32 	%r129, %r333, %r330, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs137, %rs18, 4;
	shr.u16 	%rs138, %rs22, 4;
	shr.u16 	%rs139, %rs26, 4;
	shr.u16 	%rs140, %rs30, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs141, %rs140, -8;
	cvt.u32.u16 	%r334, %rs141;
	add.s16 	%rs142, %rs139, -8;
	cvt.u32.u16 	%r335, %rs142;
	prmt.b32 	%r336, %r335, %r334, 0x3340U;
	add.s16 	%rs143, %rs138, -8;
	cvt.u32.u16 	%r337, %rs143;
	add.s16 	%rs144, %rs137, -8;
	cvt.u32.u16 	%r338, %rs144;
	prmt.b32 	%r339, %r338, %r337, 0x3340U;
	prmt.b32 	%r130, %r339, %r336, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs145, %rs19, 4;
	shr.u16 	%rs146, %rs23, 4;
	shr.u16 	%rs147, %rs27, 4;
	shr.u16 	%rs148, %rs31, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs149, %rs148, -8;
	cvt.u32.u16 	%r340, %rs149;
	add.s16 	%rs150, %rs147, -8;
	cvt.u32.u16 	%r341, %rs150;
	prmt.b32 	%r342, %r341, %r340, 0x3340U;
	add.s16 	%rs151, %rs146, -8;
	cvt.u32.u16 	%r343, %rs151;
	add.s16 	%rs152, %rs145, -8;
	cvt.u32.u16 	%r344, %rs152;
	prmt.b32 	%r345, %r344, %r343, 0x3340U;
	prmt.b32 	%r137, %r345, %r342, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs153, %rs20, 4;
	shr.u16 	%rs154, %rs24, 4;
	shr.u16 	%rs155, %rs28, 4;
	shr.u16 	%rs156, %rs32, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs157, %rs156, -8;
	cvt.u32.u16 	%r346, %rs157;
	add.s16 	%rs158, %rs155, -8;
	cvt.u32.u16 	%r347, %rs158;
	prmt.b32 	%r348, %r347, %r346, 0x3340U;
	add.s16 	%rs159, %rs154, -8;
	cvt.u32.u16 	%r349, %rs159;
	add.s16 	%rs160, %rs153, -8;
	cvt.u32.u16 	%r350, %rs160;
	prmt.b32 	%r351, %r350, %r349, 0x3340U;
	prmt.b32 	%r138, %r351, %r348, 0x5410U;
	.loc	1 831 52                        // sk04_fa_o_w4a8.py:831:52
	shl.b32 	%r352, %r468, 11;
	add.s32 	%r353, %r80, %r352;
	.loc	1 831 33                        // sk04_fa_o_w4a8.py:831:33
	add.s32 	%r354, %r353, %r20;
	ld.shared.v2.b32 	{%r355, %r356}, [%r354+16384];
	ld.shared.v2.b32 	{%r357, %r358}, [%r354+17408];
	add.s32 	%r359, %r353, %r21;
	ld.shared.v2.b32 	{%r360, %r361}, [%r359+16384];
	ld.shared.v2.b32 	{%r362, %r363}, [%r359+17408];
	add.s32 	%r364, %r353, %r22;
	ld.shared.v2.b32 	{%r365, %r366}, [%r364+16384];
	ld.shared.v2.b32 	{%r367, %r368}, [%r364+17408];
	add.s32 	%r369, %r353, %r23;
	ld.shared.v2.b32 	{%r370, %r371}, [%r369+16384];
	ld.shared.v2.b32 	{%r372, %r373}, [%r369+17408];
	.loc	1 832 33                        // sk04_fa_o_w4a8.py:832:33
	prmt.b32 	%r101, %r355, %r356, 0x6420U;
	prmt.b32 	%r102, %r357, %r358, 0x6420U;
	prmt.b32 	%r103, %r360, %r361, 0x6420U;
	prmt.b32 	%r104, %r362, %r363, 0x6420U;
	prmt.b32 	%r117, %r365, %r366, 0x6420U;
	prmt.b32 	%r118, %r367, %r368, 0x6420U;
	prmt.b32 	%r119, %r370, %r371, 0x6420U;
	prmt.b32 	%r120, %r372, %r373, 0x6420U;
	mov.b32 	%r107, %r98;
	mov.b32 	%r108, %r98;
	mov.b32 	%r109, %r98;
	mov.b32 	%r110, %r98;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r107, %r108, %r109, %r110 }, { %r101, %r102, %r103, %r104 }, { %r99, %r100 }, { %r107, %r108, %r109, %r110 };
	// end inline asm
	mov.b32 	%r116, %r98;
	mov.b32 	%r113, %r98;
	mov.b32 	%r114, %r98;
	mov.b32 	%r115, %r98;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r113, %r114, %r115, %r116 }, { %r101, %r102, %r103, %r104 }, { %r105, %r106 }, { %r113, %r114, %r115, %r116 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r107, %r108, %r109, %r110 }, { %r117, %r118, %r119, %r120 }, { %r111, %r112 }, { %r107, %r108, %r109, %r110 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r113, %r114, %r115, %r116 }, { %r117, %r118, %r119, %r120 }, { %r121, %r122 }, { %r113, %r114, %r115, %r116 };
	// end inline asm
	.loc	1 832 75                        // sk04_fa_o_w4a8.py:832:75
	prmt.b32 	%r125, %r355, %r356, 0x7531U;
	prmt.b32 	%r126, %r357, %r358, 0x7531U;
	prmt.b32 	%r127, %r360, %r361, 0x7531U;
	prmt.b32 	%r128, %r362, %r363, 0x7531U;
	prmt.b32 	%r133, %r365, %r366, 0x7531U;
	prmt.b32 	%r134, %r367, %r368, 0x7531U;
	prmt.b32 	%r135, %r370, %r371, 0x7531U;
	prmt.b32 	%r136, %r372, %r373, 0x7531U;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r107, %r108, %r109, %r110 }, { %r125, %r126, %r127, %r128 }, { %r123, %r124 }, { %r107, %r108, %r109, %r110 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r113, %r114, %r115, %r116 }, { %r125, %r126, %r127, %r128 }, { %r129, %r130 }, { %r113, %r114, %r115, %r116 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r107, %r108, %r109, %r110 }, { %r133, %r134, %r135, %r136 }, { %r131, %r132 }, { %r107, %r108, %r109, %r110 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r113, %r114, %r115, %r116 }, { %r133, %r134, %r135, %r136 }, { %r137, %r138 }, { %r113, %r114, %r115, %r116 };
	// end inline asm
	.loc	1 833 41                        // sk04_fa_o_w4a8.py:833:41
	mad.wide.s32 	%rd44, %r467, 4, %rd18;
	.loc	1 833 59                        // sk04_fa_o_w4a8.py:833:59
	add.s64 	%rd38, %rd44, %rd45;
	add.s64 	%rd39, %rd44, %rd46;
	.loc	1 833 27                        // sk04_fa_o_w4a8.py:833:27
	// begin inline asm
	mov.u32 %r139, 0x0;
	mov.u32 %r140, 0x0;
	ld.global.v2.b32 { %r139, %r140 }, [ %rd38 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r141, 0x0;
	mov.u32 %r142, 0x0;
	ld.global.v2.b32 { %r141, %r142 }, [ %rd39 + 0 ];
	// end inline asm
	.loc	1 834 26                        // sk04_fa_o_w4a8.py:834:26
	cvt.rn.f32.s32 	%r374, %r113;
	cvt.rn.f32.s32 	%r375, %r114;
	cvt.rn.f32.s32 	%r376, %r115;
	cvt.rn.f32.s32 	%r377, %r116;
	cvt.rn.f32.s32 	%r378, %r107;
	cvt.rn.f32.s32 	%r379, %r108;
	cvt.rn.f32.s32 	%r380, %r109;
	cvt.rn.f32.s32 	%r381, %r110;
	.loc	1 834 15                        // sk04_fa_o_w4a8.py:834:15
	fma.rn.f32 	%r473, %r140, %r381, %r473;
	fma.rn.f32 	%r472, %r139, %r380, %r472;
	fma.rn.f32 	%r471, %r140, %r379, %r471;
	fma.rn.f32 	%r470, %r139, %r378, %r470;
	fma.rn.f32 	%r477, %r142, %r377, %r477;
	fma.rn.f32 	%r476, %r141, %r376, %r476;
	fma.rn.f32 	%r475, %r142, %r375, %r475;
	fma.rn.f32 	%r474, %r141, %r374, %r474;
	.loc	1 836 18                        // sk04_fa_o_w4a8.py:836:18
	add.s64 	%rd40, %rd55, %rd3;
	add.s64 	%rd41, %rd55, %rd4;
	add.s64 	%rd42, %rd55, %rd5;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	add.s64 	%rd43, %rd55, %rd6;
	add.s32 	%r382, %r469, 1;
	setp.gt.s32 	%p6, %r382, 1;
	selp.b32 	%r469, 0, %r382, %p6;
	.loc	1 827 25                        // sk04_fa_o_w4a8.py:827:25
	shl.b32 	%r383, %r469, 13;
	bar.sync 	0;
	add.s32 	%r143, %r30, %r383;
	selp.b32 	%r144, 8, 0, %p4;
	// begin inline asm
	cp.async.ca.shared.global [ %r143 + 0 ], [ %rd40 + 0 ], 0x8, %r144;
	// end inline asm
	add.s32 	%r145, %r143, 2048;
	// begin inline asm
	cp.async.ca.shared.global [ %r145 + 0 ], [ %rd41 + 0 ], 0x8, %r144;
	// end inline asm
	add.s32 	%r146, %r143, 4096;
	// begin inline asm
	cp.async.ca.shared.global [ %r146 + 0 ], [ %rd42 + 0 ], 0x8, %r144;
	// end inline asm
	add.s32 	%r147, %r143, 6144;
	// begin inline asm
	cp.async.ca.shared.global [ %r147 + 0 ], [ %rd43 + 0 ], 0x8, %r144;
	// end inline asm
	cp.async.commit_group;
	.loc	1 831 52                        // sk04_fa_o_w4a8.py:831:52
	shl.b32 	%r384, %r469, 11;
	add.s32 	%r385, %r11, %r384;
	add.s32 	%r148, %r385, 16384;
	// begin inline asm
	cp.async.ca.shared.global [ %r148 + 0 ], [ %rd54 + 0 ], 0x8, %r144;
	// end inline asm
	cp.async.commit_group;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	add.s64 	%rd56, %rd56, 1;
	add.s32 	%r467, %r467, %r29;
	add.s64 	%rd55, %rd55, %rd8;
	add.s64 	%rd54, %rd54, 128;
	setp.ne.b64 	%p7, %rd12, %rd56;
	@%p7 bra 	$L__BB0_2;
$L__BB0_3:                              // %._crit_edge
	.loc	1 816 45                        // sk04_fa_o_w4a8.py:816:45
	shr.u32 	%r401, %r4, 2;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r402, %r401, %r1;
	or.b32 	%r403, %r402, 8;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r404, %r403, %r24;
	rem.s32 	%r405, %r402, %r24;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 838 38                        // sk04_fa_o_w4a8.py:838:38
	mad.wide.s32 	%rd47, %r405, 4, %rd17;
	mad.wide.s32 	%rd48, %r404, 4, %rd17;
	.loc	1 838 24                        // sk04_fa_o_w4a8.py:838:24
	// begin inline asm
	mov.u32 %r386, 0x0;
	ld.global.b32 { %r386 }, [ %rd47 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r387, 0x0;
	ld.global.b32 { %r387 }, [ %rd48 + 0 ];
	// end inline asm
	.loc	1 839 49                        // sk04_fa_o_w4a8.py:839:49
	mul.lo.s32 	%r406, %r6, %r28;
	.loc	1 839 31                        // sk04_fa_o_w4a8.py:839:31
	mad.wide.s32 	%rd51, %r406, 2, %rd16;
	.loc	1 839 64                        // sk04_fa_o_w4a8.py:839:64
	shl.b64 	%rd52, %rd7, 1;
	add.s64 	%rd49, %rd51, %rd52;
	.loc	1 839 19                        // sk04_fa_o_w4a8.py:839:19
	// begin inline asm
	mov.u32 %r389, 0x0;
	mov.u32 %r390, 0x0;
	mov.u32 %r391, 0x0;
	mov.u32 %r392, 0x0;
	ld.global.v4.b32 { %r389, %r390, %r391, %r392 }, [ %rd49 + 0 ];
	// end inline asm
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	and.b32 	%r407, %r2, 120;
	shl.b32 	%r408, %r407, 5;
	and.b32 	%r409, %r2, 7;
	shl.b32 	%r410, %r409, 4;
	or.b32 	%r411, %r408, %r410;
	xor.b32 	%r412, %r411, %r3;
	add.s32 	%r388, %r80, %r412;
	// begin inline asm
	st.shared.v4.b32 [ %r388 + 0 ], { %r389, %r390, %r391, %r392 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r413, %r409, 9;
	shl.b32 	%r414, %r2, 4;
	and.b32 	%r415, %r414, 496;
	shr.u32 	%r416, %r9, 1;
	xor.b32 	%r417, %r415, %r416;
	add.s32 	%r418, %r80, %r413;
	add.s32 	%r419, %r418, %r417;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r420, %r421, %r422, %r423}, [%r419];
	mov.b32 	{%rs169, %rs170}, %r420;
	mov.b32 	{%rs171, %rs172}, %r421;
	mov.b32 	{%rs173, %rs174}, %r422;
	mov.b32 	{%rs175, %rs176}, %r423;
	cvt.f32.bf16 	%r424, %rs169;
	cvt.f32.bf16 	%r425, %rs170;
	cvt.f32.bf16 	%r426, %rs171;
	cvt.f32.bf16 	%r427, %rs172;
	cvt.f32.bf16 	%r428, %rs173;
	cvt.f32.bf16 	%r429, %rs174;
	cvt.f32.bf16 	%r430, %rs175;
	cvt.f32.bf16 	%r431, %rs176;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r432, %r470, %r386, %r424;
	fma.rn.f32 	%r433, %r471, %r386, %r425;
	fma.rn.f32 	%r434, %r472, %r387, %r426;
	fma.rn.f32 	%r435, %r473, %r387, %r427;
	fma.rn.f32 	%r436, %r474, %r386, %r428;
	fma.rn.f32 	%r437, %r475, %r386, %r429;
	fma.rn.f32 	%r438, %r476, %r387, %r430;
	fma.rn.f32 	%r439, %r477, %r387, %r431;
	.loc	1 846 31                        // sk04_fa_o_w4a8.py:846:31
	setp.lt.s32 	%p9, %r5, %r24;
	.loc	1 846 54                        // sk04_fa_o_w4a8.py:846:54
	setp.lt.s32 	%p10, %r10, %r25;
	.loc	1 846 37                        // sk04_fa_o_w4a8.py:846:37
	and.pred 	%p8, %p9, %p10;
	.loc	1 844 35                        // sk04_fa_o_w4a8.py:844:35
	mul.lo.s32 	%r440, %r5, %r27;
	.loc	1 844 18                        // sk04_fa_o_w4a8.py:844:18
	mad.wide.s32 	%rd53, %r440, 2, %rd15;
	.loc	1 844 50                        // sk04_fa_o_w4a8.py:844:50
	mad.wide.s32 	%rd50, %r10, 2, %rd53;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16.f32 	%rs161, %r432;
	cvt.rn.bf16.f32 	%rs162, %r433;
	cvt.rn.bf16.f32 	%rs163, %r434;
	cvt.rn.bf16.f32 	%rs164, %r435;
	cvt.rn.bf16.f32 	%rs165, %r436;
	cvt.rn.bf16.f32 	%rs166, %r437;
	cvt.rn.bf16.f32 	%rs167, %r438;
	cvt.rn.bf16.f32 	%rs168, %r439;
	bar.sync 	0;
	shl.b32 	%r441, %r2, 5;
	and.b32 	%r442, %r441, 768;
	shl.b32 	%r443, %r4, 1;
	and.b32 	%r444, %r2, 1;
	neg.s32 	%r445, %r444;
	and.b32 	%r446, %r445, 1088;
	bfe.s32 	%r447, %r2, 1, 1;
	and.b32 	%r448, %r447, 2052;
	or.b32 	%r449, %r442, %r443;
	or.b32 	%r450, %r446, %r449;
	xor.b32 	%r451, %r450, %r416;
	or.b32 	%r452, %r451, %r448;
	add.s32 	%r393, %r80, %r452;
	// begin inline asm
	st.shared.v2.b16 [ %r393 + 0 ], { %rs161, %rs162 };
	// end inline asm
	add.s32 	%r394, %r393, 128;
	// begin inline asm
	st.shared.v2.b16 [ %r394 + 0 ], { %rs163, %rs164 };
	// end inline asm
	xor.b32 	%r453, %r452, 4;
	add.s32 	%r395, %r80, %r453;
	// begin inline asm
	st.shared.v2.b16 [ %r395 + 0 ], { %rs165, %rs166 };
	// end inline asm
	add.s32 	%r396, %r395, 128;
	// begin inline asm
	st.shared.v2.b16 [ %r396 + 0 ], { %rs167, %rs168 };
	// end inline asm
	bar.sync 	0;
	and.b32 	%r454, %r8, 768;
	shr.u32 	%r455, %r407, 1;
	and.b32 	%r456, %r2, 128;
	or.b32 	%r457, %r410, %r454;
	xor.b32 	%r458, %r457, %r455;
	or.b32 	%r459, %r458, %r456;
	add.s32 	%r460, %r80, %r459;
	ld.shared.b32 	%r397, [%r460];
	xor.b32 	%r461, %r459, 64;
	add.s32 	%r462, %r80, %r461;
	ld.shared.b32 	%r398, [%r462+1024];
	xor.b32 	%r463, %r459, 4;
	add.s32 	%r464, %r80, %r463;
	ld.shared.b32 	%r399, [%r464+2048];
	xor.b32 	%r465, %r459, 68;
	add.s32 	%r466, %r80, %r465;
	ld.shared.b32 	%r400, [%r466+3072];
	.loc	1 845 8                         // sk04_fa_o_w4a8.py:845:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd50 + 0 ], { %r397, %r398, %r399, %r400 };
	// end inline asm
	.loc	1 843 4                         // sk04_fa_o_w4a8.py:843:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk04_fa_o_w4a8.py"
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
.b8 5                                   // DW_FORM_data2
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 165                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x9e DW_TAG_compile_unit
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
.b8 52
.b8 95
.b8 102
.b8 97
.b8 95
.b8 111
.b8 95
.b8 119
.b8 52
.b8 97
.b8 56
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
.b8 2                                   // Abbrev [2] 0x47:0x19 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 52
.b8 95
.b8 102
.b8 97
.b8 95
.b8 111
.b8 95
.b8 119
.b8 52
.b8 97
.b8 56
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x60:0x48 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 71                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x75:0x19 DW_TAG_inlined_subroutine
.b32 71                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 40                                  // DW_AT_call_line
.b8 3
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x8e:0x19 DW_TAG_inlined_subroutine
.b32 71                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 41                                  // DW_AT_call_line
.b8 3
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_0 = _Nativo(
    "sk04_fa_o_w4a8/tile16x128x128_shift0_abi14",
    _PTX_0, "_sk04_fa_o_w4a8_kernel",
    warps=8, shared=20480,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 17],
    horneado={10: 1, 12: 1, 14: 1, 16: 1, 18: 1, 19: 16, 20: 128, 21: 128, 22: 8},
    div16=[7, 8, 9, 11, 13, 15, 17],
)

_PTX_1 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk04_fa_o_w4a8_kernel  // -- Begin function _sk04_fa_o_w4a8_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk04_fa_o_w4a8_kernel
.visible .entry _sk04_fa_o_w4a8_kernel(
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_5,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_6,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_7,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_8,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_9,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_10,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_11,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_12,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_13,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_14,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_16
)
.reqntid 256
{
	.reg .pred 	%p<11>;
	.reg .b16 	%rs<185>;
	.reg .b32 	%r<502>;
	.reg .b64 	%rd<63>;
	.loc	1 797 0                         // sk04_fa_o_w4a8.py:797:0
$L__func_begin0:
	.loc	1 797 0                         // sk04_fa_o_w4a8.py:797:0

// %bb.0:
	ld.param.b32 	%r29, [_sk04_fa_o_w4a8_kernel_param_13];
	ld.param.b32 	%r28, [_sk04_fa_o_w4a8_kernel_param_12];
	ld.param.b32 	%r27, [_sk04_fa_o_w4a8_kernel_param_11];
	ld.param.b32 	%r26, [_sk04_fa_o_w4a8_kernel_param_8];
	ld.param.b32 	%r25, [_sk04_fa_o_w4a8_kernel_param_7];
	ld.param.b32 	%r24, [_sk04_fa_o_w4a8_kernel_param_6];
	ld.param.b64 	%rd17, [_sk04_fa_o_w4a8_kernel_param_4];
	ld.param.b64 	%rd16, [_sk04_fa_o_w4a8_kernel_param_3];
	ld.param.b64 	%rd15, [_sk04_fa_o_w4a8_kernel_param_2];
	ld.param.b64 	%rd14, [_sk04_fa_o_w4a8_kernel_param_1];
	ld.param.b64 	%rd13, [_sk04_fa_o_w4a8_kernel_param_0];
$L__tmp0:
	.loc	1 807 24                        // sk04_fa_o_w4a8.py:807:24
	mov.u32 	%r43, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk04_fa_o_w4a8.py:808:27 ]
	add.s32 	%r44, %r24, 15;
	.loc	2 43 30                         // standard.py:43:30 @[ sk04_fa_o_w4a8.py:808:27 ]
	shr.s32 	%r45, %r44, 31;
	shr.u32 	%r46, %r45, 28;
	add.s32 	%r47, %r44, %r46;
	shr.s32 	%r48, %r47, 4;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk04_fa_o_w4a8.py:809:27 ]
	add.s32 	%r49, %r25, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk04_fa_o_w4a8.py:809:27 ]
	shr.s32 	%r50, %r49, 31;
	shr.u32 	%r51, %r50, 25;
	add.s32 	%r52, %r49, %r51;
	shr.s32 	%r53, %r52, 7;
$L__tmp3:
	.loc	1 810 29                        // sk04_fa_o_w4a8.py:810:29
	shl.b32 	%r54, %r53, 3;
	.loc	1 811 22                        // sk04_fa_o_w4a8.py:811:22
	div.s32 	%r55, %r43, %r54;
	.loc	1 811 38                        // sk04_fa_o_w4a8.py:811:38
	shl.b32 	%r56, %r55, 3;
	ld.param.b32 	%r57, [_sk04_fa_o_w4a8_kernel_param_9];
	.loc	1 812 30                        // sk04_fa_o_w4a8.py:812:30
	sub.s32 	%r58, %r48, %r56;
	ld.param.b32 	%r59, [_sk04_fa_o_w4a8_kernel_param_10];
	.loc	1 812 39                        // sk04_fa_o_w4a8.py:812:39
	min.s32 	%r60, %r58, 8;
	.loc	1 813 30                        // sk04_fa_o_w4a8.py:813:30
	mul.lo.s32 	%r61, %r55, %r54;
	sub.s32 	%r62, %r43, %r61;
	.loc	1 814 36                        // sk04_fa_o_w4a8.py:814:36
	div.s32 	%r63, %r62, %r60;
	.loc	1 813 46                        // sk04_fa_o_w4a8.py:813:46
	mul.lo.s32 	%r64, %r63, %r60;
	sub.s32 	%r65, %r62, %r64;
	.loc	1 813 23                        // sk04_fa_o_w4a8.py:813:23
	add.s32 	%r66, %r65, %r56;
	.loc	1 816 22                        // sk04_fa_o_w4a8.py:816:22
	shl.b32 	%r1, %r66, 4;
	.loc	1 816 45                        // sk04_fa_o_w4a8.py:816:45
	mov.u32 	%r2, %tid.x;
	and.b32 	%r3, %r2, 240;
	bfe.u32 	%r67, %r2, 4, 4;
	and.b32 	%r4, %r2, 28;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r5, %r1, %r67;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r6, %r5, %r24;
	.loc	1 817 22                        // sk04_fa_o_w4a8.py:817:22
	shl.b32 	%r7, %r63, 7;
	.loc	1 817 45                        // sk04_fa_o_w4a8.py:817:45
	shl.b32 	%r8, %r2, 3;
	and.b32 	%r68, %r8, 120;
	and.b32 	%r9, %r2, 224;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r10, %r7, %r68;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r69, %r10, %r25;
	.loc	1 821 39                        // sk04_fa_o_w4a8.py:821:39
	mul.lo.s32 	%r70, %r6, %r57;
	.loc	1 821 21                        // sk04_fa_o_w4a8.py:821:21
	cvt.s64.s32 	%rd1, %r70;
	add.s64 	%rd29, %rd13, %rd1;
	.loc	1 821 51                        // sk04_fa_o_w4a8.py:821:51
	cvt.u64.u32 	%rd2, %r68;
	add.s64 	%rd23, %rd29, %rd2;
	.loc	1 822 40                        // sk04_fa_o_w4a8.py:822:40
	mul.lo.s32 	%r71, %r59, %r67;
	shl.b32 	%r72, %r59, 4;
	add.s32 	%r73, %r71, %r72;
	shl.b32 	%r74, %r59, 5;
	add.s32 	%r75, %r71, %r74;
	mad.lo.s32 	%r76, %r59, 48, %r71;
	.loc	1 822 21                        // sk04_fa_o_w4a8.py:822:21
	cvt.s64.s32 	%rd3, %r71;
	add.s64 	%rd30, %rd14, %rd3;
	cvt.s64.s32 	%rd4, %r73;
	add.s64 	%rd31, %rd14, %rd4;
	cvt.s64.s32 	%rd5, %r75;
	add.s64 	%rd32, %rd14, %rd5;
	cvt.s64.s32 	%rd6, %r76;
	add.s64 	%rd33, %rd14, %rd6;
	.loc	1 822 52                        // sk04_fa_o_w4a8.py:822:52
	cvt.s64.s32 	%rd7, %r69;
	add.s64 	%rd19, %rd30, %rd7;
	add.s64 	%rd20, %rd31, %rd7;
	add.s64 	%rd21, %rd32, %rd7;
	add.s64 	%rd22, %rd33, %rd7;
	.loc	1 836 35                        // sk04_fa_o_w4a8.py:836:35
	shl.b32 	%r77, %r59, 6;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	setp.lt.s32 	%p1, %r26, 128;
	setp.gt.s32 	%p2, %r26, 127;
	.loc	1 827 25                        // sk04_fa_o_w4a8.py:827:25
	and.b32 	%r78, %r8, 2040;
	shr.u32 	%r79, %r3, 1;
	xor.b32 	%r80, %r78, %r79;
	mov.b32 	%r81, global_smem;
	add.s32 	%r31, %r81, %r80;
	selp.b32 	%r32, 8, 0, %p2;
	// begin inline asm
	cp.async.ca.shared.global [ %r31 + 0 ], [ %rd19 + 0 ], 0x8, %r32;
	// end inline asm
	add.s32 	%r33, %r31, 2048;
	// begin inline asm
	cp.async.ca.shared.global [ %r33 + 0 ], [ %rd20 + 0 ], 0x8, %r32;
	// end inline asm
	add.s32 	%r34, %r31, 4096;
	// begin inline asm
	cp.async.ca.shared.global [ %r34 + 0 ], [ %rd21 + 0 ], 0x8, %r32;
	// end inline asm
	add.s32 	%r35, %r31, 6144;
	// begin inline asm
	cp.async.ca.shared.global [ %r35 + 0 ], [ %rd22 + 0 ], 0x8, %r32;
	// end inline asm
	cp.async.commit_group;
	.loc	1 831 52                        // sk04_fa_o_w4a8.py:831:52
	shl.b32 	%r82, %r2, 1;
	and.b32 	%r83, %r82, 96;
	xor.b32 	%r84, %r78, %r83;
	add.s32 	%r11, %r81, %r84;
	add.s32 	%r36, %r11, 16384;
	// begin inline asm
	cp.async.ca.shared.global [ %r36 + 0 ], [ %rd23 + 0 ], 0x8, %r32;
	// end inline asm
	cp.async.commit_group;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	setp.gt.s32 	%p3, %r26, 255;
	.loc	1 835 18                        // sk04_fa_o_w4a8.py:835:18
	add.s64 	%rd28, %rd23, 128;
	.loc	1 836 18                        // sk04_fa_o_w4a8.py:836:18
	cvt.s64.s32 	%rd8, %r77;
	add.s64 	%rd24, %rd19, %rd8;
	add.s64 	%rd25, %rd20, %rd8;
	add.s64 	%rd26, %rd21, %rd8;
	add.s64 	%rd27, %rd22, %rd8;
	.loc	1 827 25                        // sk04_fa_o_w4a8.py:827:25
	bar.sync 	0;
	add.s32 	%r37, %r31, 8192;
	selp.b32 	%r38, 8, 0, %p3;
	// begin inline asm
	cp.async.ca.shared.global [ %r37 + 0 ], [ %rd24 + 0 ], 0x8, %r38;
	// end inline asm
	add.s32 	%r39, %r31, 10240;
	// begin inline asm
	cp.async.ca.shared.global [ %r39 + 0 ], [ %rd25 + 0 ], 0x8, %r38;
	// end inline asm
	add.s32 	%r40, %r31, 12288;
	// begin inline asm
	cp.async.ca.shared.global [ %r40 + 0 ], [ %rd26 + 0 ], 0x8, %r38;
	// end inline asm
	add.s32 	%r41, %r31, 14336;
	// begin inline asm
	cp.async.ca.shared.global [ %r41 + 0 ], [ %rd27 + 0 ], 0x8, %r38;
	// end inline asm
	cp.async.commit_group;
	.loc	1 831 52                        // sk04_fa_o_w4a8.py:831:52
	add.s32 	%r42, %r11, 18432;
	// begin inline asm
	cp.async.ca.shared.global [ %r42 + 0 ], [ %rd28 + 0 ], 0x8, %r38;
	// end inline asm
	cp.async.commit_group;
	mov.b32 	%r494, 0f00000000;
	mov.b32 	%r495, %r494;
	mov.b32 	%r496, %r494;
	mov.b32 	%r497, %r494;
	mov.b32 	%r498, %r494;
	mov.b32 	%r499, %r494;
	mov.b32 	%r500, %r494;
	mov.b32 	%r501, %r494;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	@%p1 bra 	$L__BB0_3;
// %bb.1:                               // %.lr.ph
	.loc	1 0 22                          // sk04_fa_o_w4a8.py:0:22
	ld.param.b32 	%r30, [_sk04_fa_o_w4a8_kernel_param_14];
	ld.param.b64 	%rd18, [_sk04_fa_o_w4a8_kernel_param_5];
	cvt.u32.u64 	%r85, %rd2;
	.loc	1 825 27                        // sk04_fa_o_w4a8.py:825:27
	shr.u32 	%r86, %r26, 7;
	.loc	1 817 45                        // sk04_fa_o_w4a8.py:817:45
	and.b32 	%r87, %r2, 3;
	shl.b32 	%r88, %r87, 1;
	shr.u32 	%r89, %r9, 2;
	or.b32 	%r90, %r88, %r89;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r91, %r90, %r7;
	or.b32 	%r92, %r91, 64;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r93, %r92, %r25;
	rem.s32 	%r94, %r91, %r25;
	add.s32 	%r95, %r86, -2;
	bfe.u32 	%r96, %r2, 2, 6;
	mul.lo.s32 	%r97, %r87, 544;
	xor.b32 	%r12, %r97, %r96;
	xor.b32 	%r13, %r12, 136;
	xor.b32 	%r14, %r12, 272;
	xor.b32 	%r15, %r12, 408;
	xor.b32 	%r16, %r12, 64;
	xor.b32 	%r17, %r12, 200;
	xor.b32 	%r18, %r12, 336;
	xor.b32 	%r19, %r12, 472;
	shl.b32 	%r98, %r4, 5;
	or.b32 	%r20, %r98, %r85;
	xor.b32 	%r21, %r20, 32;
	xor.b32 	%r22, %r20, 64;
	xor.b32 	%r23, %r20, 96;
	cvt.s64.s32 	%rd9, %r94;
	cvt.s64.s32 	%rd10, %r93;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	cvt.s64.s32 	%rd11, %r95;
	cvt.u64.u32 	%rd12, %r86;
	shl.b64 	%rd34, %rd8, 1;
	add.s64 	%rd35, %rd34, %rd7;
	add.s64 	%rd61, %rd14, %rd35;
	add.s64 	%rd36, %rd2, %rd1;
	add.s64 	%rd37, %rd36, %rd13;
	add.s64 	%rd60, %rd37, 256;
	mov.b32 	%r494, 0f00000000;
	mov.b32 	%r493, 1;
	mov.b32 	%r492, -1;
	mov.b64 	%rd62, 0;
	mov.b32 	%r99, 0;
	shl.b64 	%rd45, %rd9, 2;
	shl.b64 	%rd46, %rd10, 2;
	mov.b32 	%r491, %r99;
	mov.b32 	%r495, %r494;
	mov.b32 	%r496, %r494;
	mov.b32 	%r497, %r494;
	mov.b32 	%r498, %r494;
	mov.b32 	%r499, %r494;
	mov.b32 	%r500, %r494;
	mov.b32 	%r501, %r494;
$L__BB0_2:                              // =>This Inner Loop Header: Depth=1
	setp.lt.s64 	%p4, %rd62, %rd11;
	add.s32 	%r150, %r492, 1;
	setp.gt.s32 	%p5, %r150, 1;
	selp.b32 	%r492, 0, %r150, %p5;
	.loc	1 827 25                        // sk04_fa_o_w4a8.py:827:25
	cp.async.wait_group 	2;
	bar.sync 	0;
	shl.b32 	%r151, %r492, 13;
	add.s32 	%r152, %r81, %r151;
	.loc	1 828 38                        // sk04_fa_o_w4a8.py:828:38
	add.s32 	%r153, %r152, %r12;
	ld.shared.b8 	%rs1, [%r153];
	ld.shared.b8 	%rs2, [%r153+2048];
	ld.shared.b8 	%rs3, [%r153+4096];
	ld.shared.b8 	%rs4, [%r153+6144];
	add.s32 	%r154, %r152, %r13;
	ld.shared.b8 	%rs5, [%r154];
	ld.shared.b8 	%rs6, [%r154+2048];
	ld.shared.b8 	%rs7, [%r154+4096];
	ld.shared.b8 	%rs8, [%r154+6144];
	add.s32 	%r155, %r152, %r14;
	ld.shared.b8 	%rs9, [%r155];
	ld.shared.b8 	%rs10, [%r155+2048];
	ld.shared.b8 	%rs11, [%r155+4096];
	ld.shared.b8 	%rs12, [%r155+6144];
	add.s32 	%r156, %r152, %r15;
	ld.shared.b8 	%rs13, [%r156];
	ld.shared.b8 	%rs14, [%r156+2048];
	ld.shared.b8 	%rs15, [%r156+4096];
	ld.shared.b8 	%rs16, [%r156+6144];
	add.s32 	%r157, %r152, %r16;
	ld.shared.b8 	%rs17, [%r157];
	ld.shared.b8 	%rs18, [%r157+2048];
	ld.shared.b8 	%rs19, [%r157+4096];
	ld.shared.b8 	%rs20, [%r157+6144];
	add.s32 	%r158, %r152, %r17;
	ld.shared.b8 	%rs21, [%r158];
	ld.shared.b8 	%rs22, [%r158+2048];
	ld.shared.b8 	%rs23, [%r158+4096];
	ld.shared.b8 	%rs24, [%r158+6144];
	add.s32 	%r159, %r152, %r18;
	ld.shared.b8 	%rs25, [%r159];
	ld.shared.b8 	%rs26, [%r159+2048];
	ld.shared.b8 	%rs27, [%r159+4096];
	ld.shared.b8 	%rs28, [%r159+6144];
	add.s32 	%r160, %r152, %r19;
	ld.shared.b8 	%rs29, [%r160];
	ld.shared.b8 	%rs30, [%r160+2048];
	ld.shared.b8 	%rs31, [%r160+4096];
	ld.shared.b8 	%rs32, [%r160+6144];
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r161, %rs13;
	cvt.u32.u16 	%r162, %rs9;
	prmt.b32 	%r163, %r162, %r161, 0x3340U;
	cvt.u32.u16 	%r164, %rs5;
	cvt.u32.u16 	%r165, %rs1;
	prmt.b32 	%r166, %r165, %r164, 0x3340U;
	prmt.b32 	%r167, %r166, %r163, 0x5410U;
	and.b32 	%r168, %r167, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r169, %r168, 0, 0x7773U;
	cvt.u16.u32 	%rs33, %r169;
	add.s16 	%rs34, %rs33, -8;
	cvt.u32.u16 	%r170, %rs34;
	prmt.b32 	%r171, %r168, 0, 0x7772U;
	cvt.u16.u32 	%rs35, %r171;
	add.s16 	%rs36, %rs35, -8;
	cvt.u32.u16 	%r172, %rs36;
	prmt.b32 	%r173, %r172, %r170, 0x3340U;
	prmt.b32 	%r174, %r168, 0, 0x7771U;
	cvt.u16.u32 	%rs37, %r174;
	add.s16 	%rs38, %rs37, -8;
	cvt.u32.u16 	%r175, %rs38;
	prmt.b32 	%r176, %r168, 0, 0x7770U;
	cvt.u16.u32 	%rs39, %r176;
	add.s16 	%rs40, %rs39, -8;
	cvt.u32.u16 	%r177, %rs40;
	prmt.b32 	%r178, %r177, %r175, 0x3340U;
	prmt.b32 	%r100, %r178, %r173, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r179, %rs14;
	cvt.u32.u16 	%r180, %rs10;
	prmt.b32 	%r181, %r180, %r179, 0x3340U;
	cvt.u32.u16 	%r182, %rs6;
	cvt.u32.u16 	%r183, %rs2;
	prmt.b32 	%r184, %r183, %r182, 0x3340U;
	prmt.b32 	%r185, %r184, %r181, 0x5410U;
	and.b32 	%r186, %r185, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r187, %r186, 0, 0x7773U;
	cvt.u16.u32 	%rs41, %r187;
	add.s16 	%rs42, %rs41, -8;
	cvt.u32.u16 	%r188, %rs42;
	prmt.b32 	%r189, %r186, 0, 0x7772U;
	cvt.u16.u32 	%rs43, %r189;
	add.s16 	%rs44, %rs43, -8;
	cvt.u32.u16 	%r190, %rs44;
	prmt.b32 	%r191, %r190, %r188, 0x3340U;
	prmt.b32 	%r192, %r186, 0, 0x7771U;
	cvt.u16.u32 	%rs45, %r192;
	add.s16 	%rs46, %rs45, -8;
	cvt.u32.u16 	%r193, %rs46;
	prmt.b32 	%r194, %r186, 0, 0x7770U;
	cvt.u16.u32 	%rs47, %r194;
	add.s16 	%rs48, %rs47, -8;
	cvt.u32.u16 	%r195, %rs48;
	prmt.b32 	%r196, %r195, %r193, 0x3340U;
	prmt.b32 	%r101, %r196, %r191, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r197, %rs15;
	cvt.u32.u16 	%r198, %rs11;
	prmt.b32 	%r199, %r198, %r197, 0x3340U;
	cvt.u32.u16 	%r200, %rs7;
	cvt.u32.u16 	%r201, %rs3;
	prmt.b32 	%r202, %r201, %r200, 0x3340U;
	prmt.b32 	%r203, %r202, %r199, 0x5410U;
	and.b32 	%r204, %r203, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r205, %r204, 0, 0x7773U;
	cvt.u16.u32 	%rs49, %r205;
	add.s16 	%rs50, %rs49, -8;
	cvt.u32.u16 	%r206, %rs50;
	prmt.b32 	%r207, %r204, 0, 0x7772U;
	cvt.u16.u32 	%rs51, %r207;
	add.s16 	%rs52, %rs51, -8;
	cvt.u32.u16 	%r208, %rs52;
	prmt.b32 	%r209, %r208, %r206, 0x3340U;
	prmt.b32 	%r210, %r204, 0, 0x7771U;
	cvt.u16.u32 	%rs53, %r210;
	add.s16 	%rs54, %rs53, -8;
	cvt.u32.u16 	%r211, %rs54;
	prmt.b32 	%r212, %r204, 0, 0x7770U;
	cvt.u16.u32 	%rs55, %r212;
	add.s16 	%rs56, %rs55, -8;
	cvt.u32.u16 	%r213, %rs56;
	prmt.b32 	%r214, %r213, %r211, 0x3340U;
	prmt.b32 	%r112, %r214, %r209, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r215, %rs16;
	cvt.u32.u16 	%r216, %rs12;
	prmt.b32 	%r217, %r216, %r215, 0x3340U;
	cvt.u32.u16 	%r218, %rs8;
	cvt.u32.u16 	%r219, %rs4;
	prmt.b32 	%r220, %r219, %r218, 0x3340U;
	prmt.b32 	%r221, %r220, %r217, 0x5410U;
	and.b32 	%r222, %r221, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r223, %r222, 0, 0x7773U;
	cvt.u16.u32 	%rs57, %r223;
	add.s16 	%rs58, %rs57, -8;
	cvt.u32.u16 	%r224, %rs58;
	prmt.b32 	%r225, %r222, 0, 0x7772U;
	cvt.u16.u32 	%rs59, %r225;
	add.s16 	%rs60, %rs59, -8;
	cvt.u32.u16 	%r226, %rs60;
	prmt.b32 	%r227, %r226, %r224, 0x3340U;
	prmt.b32 	%r228, %r222, 0, 0x7771U;
	cvt.u16.u32 	%rs61, %r228;
	add.s16 	%rs62, %rs61, -8;
	cvt.u32.u16 	%r229, %rs62;
	prmt.b32 	%r230, %r222, 0, 0x7770U;
	cvt.u16.u32 	%rs63, %r230;
	add.s16 	%rs64, %rs63, -8;
	cvt.u32.u16 	%r231, %rs64;
	prmt.b32 	%r232, %r231, %r229, 0x3340U;
	prmt.b32 	%r113, %r232, %r227, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r233, %rs29;
	cvt.u32.u16 	%r234, %rs25;
	prmt.b32 	%r235, %r234, %r233, 0x3340U;
	cvt.u32.u16 	%r236, %rs21;
	cvt.u32.u16 	%r237, %rs17;
	prmt.b32 	%r238, %r237, %r236, 0x3340U;
	prmt.b32 	%r239, %r238, %r235, 0x5410U;
	and.b32 	%r240, %r239, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r241, %r240, 0, 0x7773U;
	cvt.u16.u32 	%rs65, %r241;
	add.s16 	%rs66, %rs65, -8;
	cvt.u32.u16 	%r242, %rs66;
	prmt.b32 	%r243, %r240, 0, 0x7772U;
	cvt.u16.u32 	%rs67, %r243;
	add.s16 	%rs68, %rs67, -8;
	cvt.u32.u16 	%r244, %rs68;
	prmt.b32 	%r245, %r244, %r242, 0x3340U;
	prmt.b32 	%r246, %r240, 0, 0x7771U;
	cvt.u16.u32 	%rs69, %r246;
	add.s16 	%rs70, %rs69, -8;
	cvt.u32.u16 	%r247, %rs70;
	prmt.b32 	%r248, %r240, 0, 0x7770U;
	cvt.u16.u32 	%rs71, %r248;
	add.s16 	%rs72, %rs71, -8;
	cvt.u32.u16 	%r249, %rs72;
	prmt.b32 	%r250, %r249, %r247, 0x3340U;
	prmt.b32 	%r106, %r250, %r245, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r251, %rs30;
	cvt.u32.u16 	%r252, %rs26;
	prmt.b32 	%r253, %r252, %r251, 0x3340U;
	cvt.u32.u16 	%r254, %rs22;
	cvt.u32.u16 	%r255, %rs18;
	prmt.b32 	%r256, %r255, %r254, 0x3340U;
	prmt.b32 	%r257, %r256, %r253, 0x5410U;
	and.b32 	%r258, %r257, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r259, %r258, 0, 0x7773U;
	cvt.u16.u32 	%rs73, %r259;
	add.s16 	%rs74, %rs73, -8;
	cvt.u32.u16 	%r260, %rs74;
	prmt.b32 	%r261, %r258, 0, 0x7772U;
	cvt.u16.u32 	%rs75, %r261;
	add.s16 	%rs76, %rs75, -8;
	cvt.u32.u16 	%r262, %rs76;
	prmt.b32 	%r263, %r262, %r260, 0x3340U;
	prmt.b32 	%r264, %r258, 0, 0x7771U;
	cvt.u16.u32 	%rs77, %r264;
	add.s16 	%rs78, %rs77, -8;
	cvt.u32.u16 	%r265, %rs78;
	prmt.b32 	%r266, %r258, 0, 0x7770U;
	cvt.u16.u32 	%rs79, %r266;
	add.s16 	%rs80, %rs79, -8;
	cvt.u32.u16 	%r267, %rs80;
	prmt.b32 	%r268, %r267, %r265, 0x3340U;
	prmt.b32 	%r107, %r268, %r263, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r269, %rs31;
	cvt.u32.u16 	%r270, %rs27;
	prmt.b32 	%r271, %r270, %r269, 0x3340U;
	cvt.u32.u16 	%r272, %rs23;
	cvt.u32.u16 	%r273, %rs19;
	prmt.b32 	%r274, %r273, %r272, 0x3340U;
	prmt.b32 	%r275, %r274, %r271, 0x5410U;
	and.b32 	%r276, %r275, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r277, %r276, 0, 0x7773U;
	cvt.u16.u32 	%rs81, %r277;
	add.s16 	%rs82, %rs81, -8;
	cvt.u32.u16 	%r278, %rs82;
	prmt.b32 	%r279, %r276, 0, 0x7772U;
	cvt.u16.u32 	%rs83, %r279;
	add.s16 	%rs84, %rs83, -8;
	cvt.u32.u16 	%r280, %rs84;
	prmt.b32 	%r281, %r280, %r278, 0x3340U;
	prmt.b32 	%r282, %r276, 0, 0x7771U;
	cvt.u16.u32 	%rs85, %r282;
	add.s16 	%rs86, %rs85, -8;
	cvt.u32.u16 	%r283, %rs86;
	prmt.b32 	%r284, %r276, 0, 0x7770U;
	cvt.u16.u32 	%rs87, %r284;
	add.s16 	%rs88, %rs87, -8;
	cvt.u32.u16 	%r285, %rs88;
	prmt.b32 	%r286, %r285, %r283, 0x3340U;
	prmt.b32 	%r122, %r286, %r281, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r287, %rs32;
	cvt.u32.u16 	%r288, %rs28;
	prmt.b32 	%r289, %r288, %r287, 0x3340U;
	cvt.u32.u16 	%r290, %rs24;
	cvt.u32.u16 	%r291, %rs20;
	prmt.b32 	%r292, %r291, %r290, 0x3340U;
	prmt.b32 	%r293, %r292, %r289, 0x5410U;
	and.b32 	%r294, %r293, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r295, %r294, 0, 0x7773U;
	cvt.u16.u32 	%rs89, %r295;
	add.s16 	%rs90, %rs89, -8;
	cvt.u32.u16 	%r296, %rs90;
	prmt.b32 	%r297, %r294, 0, 0x7772U;
	cvt.u16.u32 	%rs91, %r297;
	add.s16 	%rs92, %rs91, -8;
	cvt.u32.u16 	%r298, %rs92;
	prmt.b32 	%r299, %r298, %r296, 0x3340U;
	prmt.b32 	%r300, %r294, 0, 0x7771U;
	cvt.u16.u32 	%rs93, %r300;
	add.s16 	%rs94, %rs93, -8;
	cvt.u32.u16 	%r301, %rs94;
	prmt.b32 	%r302, %r294, 0, 0x7770U;
	cvt.u16.u32 	%rs95, %r302;
	add.s16 	%rs96, %rs95, -8;
	cvt.u32.u16 	%r303, %rs96;
	prmt.b32 	%r304, %r303, %r301, 0x3340U;
	prmt.b32 	%r123, %r304, %r299, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs97, %rs1, 4;
	shr.u16 	%rs98, %rs5, 4;
	shr.u16 	%rs99, %rs9, 4;
	shr.u16 	%rs100, %rs13, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs101, %rs100, -8;
	cvt.u32.u16 	%r305, %rs101;
	add.s16 	%rs102, %rs99, -8;
	cvt.u32.u16 	%r306, %rs102;
	prmt.b32 	%r307, %r306, %r305, 0x3340U;
	add.s16 	%rs103, %rs98, -8;
	cvt.u32.u16 	%r308, %rs103;
	add.s16 	%rs104, %rs97, -8;
	cvt.u32.u16 	%r309, %rs104;
	prmt.b32 	%r310, %r309, %r308, 0x3340U;
	prmt.b32 	%r124, %r310, %r307, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs105, %rs2, 4;
	shr.u16 	%rs106, %rs6, 4;
	shr.u16 	%rs107, %rs10, 4;
	shr.u16 	%rs108, %rs14, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs109, %rs108, -8;
	cvt.u32.u16 	%r311, %rs109;
	add.s16 	%rs110, %rs107, -8;
	cvt.u32.u16 	%r312, %rs110;
	prmt.b32 	%r313, %r312, %r311, 0x3340U;
	add.s16 	%rs111, %rs106, -8;
	cvt.u32.u16 	%r314, %rs111;
	add.s16 	%rs112, %rs105, -8;
	cvt.u32.u16 	%r315, %rs112;
	prmt.b32 	%r316, %r315, %r314, 0x3340U;
	prmt.b32 	%r125, %r316, %r313, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs113, %rs3, 4;
	shr.u16 	%rs114, %rs7, 4;
	shr.u16 	%rs115, %rs11, 4;
	shr.u16 	%rs116, %rs15, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs117, %rs116, -8;
	cvt.u32.u16 	%r317, %rs117;
	add.s16 	%rs118, %rs115, -8;
	cvt.u32.u16 	%r318, %rs118;
	prmt.b32 	%r319, %r318, %r317, 0x3340U;
	add.s16 	%rs119, %rs114, -8;
	cvt.u32.u16 	%r320, %rs119;
	add.s16 	%rs120, %rs113, -8;
	cvt.u32.u16 	%r321, %rs120;
	prmt.b32 	%r322, %r321, %r320, 0x3340U;
	prmt.b32 	%r132, %r322, %r319, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs121, %rs4, 4;
	shr.u16 	%rs122, %rs8, 4;
	shr.u16 	%rs123, %rs12, 4;
	shr.u16 	%rs124, %rs16, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs125, %rs124, -8;
	cvt.u32.u16 	%r323, %rs125;
	add.s16 	%rs126, %rs123, -8;
	cvt.u32.u16 	%r324, %rs126;
	prmt.b32 	%r325, %r324, %r323, 0x3340U;
	add.s16 	%rs127, %rs122, -8;
	cvt.u32.u16 	%r326, %rs127;
	add.s16 	%rs128, %rs121, -8;
	cvt.u32.u16 	%r327, %rs128;
	prmt.b32 	%r328, %r327, %r326, 0x3340U;
	prmt.b32 	%r133, %r328, %r325, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs129, %rs17, 4;
	shr.u16 	%rs130, %rs21, 4;
	shr.u16 	%rs131, %rs25, 4;
	shr.u16 	%rs132, %rs29, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs133, %rs132, -8;
	cvt.u32.u16 	%r329, %rs133;
	add.s16 	%rs134, %rs131, -8;
	cvt.u32.u16 	%r330, %rs134;
	prmt.b32 	%r331, %r330, %r329, 0x3340U;
	add.s16 	%rs135, %rs130, -8;
	cvt.u32.u16 	%r332, %rs135;
	add.s16 	%rs136, %rs129, -8;
	cvt.u32.u16 	%r333, %rs136;
	prmt.b32 	%r334, %r333, %r332, 0x3340U;
	prmt.b32 	%r130, %r334, %r331, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs137, %rs18, 4;
	shr.u16 	%rs138, %rs22, 4;
	shr.u16 	%rs139, %rs26, 4;
	shr.u16 	%rs140, %rs30, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs141, %rs140, -8;
	cvt.u32.u16 	%r335, %rs141;
	add.s16 	%rs142, %rs139, -8;
	cvt.u32.u16 	%r336, %rs142;
	prmt.b32 	%r337, %r336, %r335, 0x3340U;
	add.s16 	%rs143, %rs138, -8;
	cvt.u32.u16 	%r338, %rs143;
	add.s16 	%rs144, %rs137, -8;
	cvt.u32.u16 	%r339, %rs144;
	prmt.b32 	%r340, %r339, %r338, 0x3340U;
	prmt.b32 	%r131, %r340, %r337, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs145, %rs19, 4;
	shr.u16 	%rs146, %rs23, 4;
	shr.u16 	%rs147, %rs27, 4;
	shr.u16 	%rs148, %rs31, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs149, %rs148, -8;
	cvt.u32.u16 	%r341, %rs149;
	add.s16 	%rs150, %rs147, -8;
	cvt.u32.u16 	%r342, %rs150;
	prmt.b32 	%r343, %r342, %r341, 0x3340U;
	add.s16 	%rs151, %rs146, -8;
	cvt.u32.u16 	%r344, %rs151;
	add.s16 	%rs152, %rs145, -8;
	cvt.u32.u16 	%r345, %rs152;
	prmt.b32 	%r346, %r345, %r344, 0x3340U;
	prmt.b32 	%r138, %r346, %r343, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs153, %rs20, 4;
	shr.u16 	%rs154, %rs24, 4;
	shr.u16 	%rs155, %rs28, 4;
	shr.u16 	%rs156, %rs32, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs157, %rs156, -8;
	cvt.u32.u16 	%r347, %rs157;
	add.s16 	%rs158, %rs155, -8;
	cvt.u32.u16 	%r348, %rs158;
	prmt.b32 	%r349, %r348, %r347, 0x3340U;
	add.s16 	%rs159, %rs154, -8;
	cvt.u32.u16 	%r350, %rs159;
	add.s16 	%rs160, %rs153, -8;
	cvt.u32.u16 	%r351, %rs160;
	prmt.b32 	%r352, %r351, %r350, 0x3340U;
	prmt.b32 	%r139, %r352, %r349, 0x5410U;
	.loc	1 831 52                        // sk04_fa_o_w4a8.py:831:52
	shl.b32 	%r353, %r492, 11;
	add.s32 	%r354, %r81, %r353;
	.loc	1 831 33                        // sk04_fa_o_w4a8.py:831:33
	add.s32 	%r355, %r354, %r20;
	ld.shared.v2.b32 	{%r356, %r357}, [%r355+16384];
	ld.shared.v2.b32 	{%r358, %r359}, [%r355+17408];
	add.s32 	%r360, %r354, %r21;
	ld.shared.v2.b32 	{%r361, %r362}, [%r360+16384];
	ld.shared.v2.b32 	{%r363, %r364}, [%r360+17408];
	add.s32 	%r365, %r354, %r22;
	ld.shared.v2.b32 	{%r366, %r367}, [%r365+16384];
	ld.shared.v2.b32 	{%r368, %r369}, [%r365+17408];
	add.s32 	%r370, %r354, %r23;
	ld.shared.v2.b32 	{%r371, %r372}, [%r370+16384];
	ld.shared.v2.b32 	{%r373, %r374}, [%r370+17408];
	.loc	1 832 33                        // sk04_fa_o_w4a8.py:832:33
	prmt.b32 	%r102, %r356, %r357, 0x6420U;
	prmt.b32 	%r103, %r358, %r359, 0x6420U;
	prmt.b32 	%r104, %r361, %r362, 0x6420U;
	prmt.b32 	%r105, %r363, %r364, 0x6420U;
	prmt.b32 	%r118, %r366, %r367, 0x6420U;
	prmt.b32 	%r119, %r368, %r369, 0x6420U;
	prmt.b32 	%r120, %r371, %r372, 0x6420U;
	prmt.b32 	%r121, %r373, %r374, 0x6420U;
	mov.b32 	%r108, %r99;
	mov.b32 	%r109, %r99;
	mov.b32 	%r110, %r99;
	mov.b32 	%r111, %r99;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r108, %r109, %r110, %r111 }, { %r102, %r103, %r104, %r105 }, { %r100, %r101 }, { %r108, %r109, %r110, %r111 };
	// end inline asm
	mov.b32 	%r117, %r99;
	mov.b32 	%r114, %r99;
	mov.b32 	%r115, %r99;
	mov.b32 	%r116, %r99;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r114, %r115, %r116, %r117 }, { %r102, %r103, %r104, %r105 }, { %r106, %r107 }, { %r114, %r115, %r116, %r117 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r108, %r109, %r110, %r111 }, { %r118, %r119, %r120, %r121 }, { %r112, %r113 }, { %r108, %r109, %r110, %r111 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r114, %r115, %r116, %r117 }, { %r118, %r119, %r120, %r121 }, { %r122, %r123 }, { %r114, %r115, %r116, %r117 };
	// end inline asm
	.loc	1 832 75                        // sk04_fa_o_w4a8.py:832:75
	prmt.b32 	%r126, %r356, %r357, 0x7531U;
	prmt.b32 	%r127, %r358, %r359, 0x7531U;
	prmt.b32 	%r128, %r361, %r362, 0x7531U;
	prmt.b32 	%r129, %r363, %r364, 0x7531U;
	prmt.b32 	%r134, %r366, %r367, 0x7531U;
	prmt.b32 	%r135, %r368, %r369, 0x7531U;
	prmt.b32 	%r136, %r371, %r372, 0x7531U;
	prmt.b32 	%r137, %r373, %r374, 0x7531U;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r108, %r109, %r110, %r111 }, { %r126, %r127, %r128, %r129 }, { %r124, %r125 }, { %r108, %r109, %r110, %r111 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r114, %r115, %r116, %r117 }, { %r126, %r127, %r128, %r129 }, { %r130, %r131 }, { %r114, %r115, %r116, %r117 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r108, %r109, %r110, %r111 }, { %r134, %r135, %r136, %r137 }, { %r132, %r133 }, { %r108, %r109, %r110, %r111 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r114, %r115, %r116, %r117 }, { %r134, %r135, %r136, %r137 }, { %r138, %r139 }, { %r114, %r115, %r116, %r117 };
	// end inline asm
	.loc	1 833 41                        // sk04_fa_o_w4a8.py:833:41
	mad.wide.s32 	%rd44, %r491, 4, %rd18;
	.loc	1 833 59                        // sk04_fa_o_w4a8.py:833:59
	add.s64 	%rd38, %rd44, %rd45;
	add.s64 	%rd39, %rd44, %rd46;
	.loc	1 833 27                        // sk04_fa_o_w4a8.py:833:27
	// begin inline asm
	mov.u32 %r140, 0x0;
	mov.u32 %r141, 0x0;
	ld.global.v2.b32 { %r140, %r141 }, [ %rd38 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r142, 0x0;
	mov.u32 %r143, 0x0;
	ld.global.v2.b32 { %r142, %r143 }, [ %rd39 + 0 ];
	// end inline asm
	.loc	1 834 26                        // sk04_fa_o_w4a8.py:834:26
	cvt.rn.f32.s32 	%r375, %r114;
	cvt.rn.f32.s32 	%r376, %r115;
	cvt.rn.f32.s32 	%r377, %r116;
	cvt.rn.f32.s32 	%r378, %r117;
	cvt.rn.f32.s32 	%r379, %r108;
	cvt.rn.f32.s32 	%r380, %r109;
	cvt.rn.f32.s32 	%r381, %r110;
	cvt.rn.f32.s32 	%r382, %r111;
	.loc	1 834 15                        // sk04_fa_o_w4a8.py:834:15
	fma.rn.f32 	%r497, %r141, %r382, %r497;
	fma.rn.f32 	%r496, %r140, %r381, %r496;
	fma.rn.f32 	%r495, %r141, %r380, %r495;
	fma.rn.f32 	%r494, %r140, %r379, %r494;
	fma.rn.f32 	%r501, %r143, %r378, %r501;
	fma.rn.f32 	%r500, %r142, %r377, %r500;
	fma.rn.f32 	%r499, %r143, %r376, %r499;
	fma.rn.f32 	%r498, %r142, %r375, %r498;
	.loc	1 836 18                        // sk04_fa_o_w4a8.py:836:18
	add.s64 	%rd40, %rd61, %rd3;
	add.s64 	%rd41, %rd61, %rd4;
	add.s64 	%rd42, %rd61, %rd5;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	add.s64 	%rd43, %rd61, %rd6;
	add.s32 	%r383, %r493, 1;
	setp.gt.s32 	%p6, %r383, 1;
	selp.b32 	%r493, 0, %r383, %p6;
	.loc	1 827 25                        // sk04_fa_o_w4a8.py:827:25
	shl.b32 	%r384, %r493, 13;
	bar.sync 	0;
	add.s32 	%r144, %r31, %r384;
	selp.b32 	%r145, 8, 0, %p4;
	// begin inline asm
	cp.async.ca.shared.global [ %r144 + 0 ], [ %rd40 + 0 ], 0x8, %r145;
	// end inline asm
	add.s32 	%r146, %r144, 2048;
	// begin inline asm
	cp.async.ca.shared.global [ %r146 + 0 ], [ %rd41 + 0 ], 0x8, %r145;
	// end inline asm
	add.s32 	%r147, %r144, 4096;
	// begin inline asm
	cp.async.ca.shared.global [ %r147 + 0 ], [ %rd42 + 0 ], 0x8, %r145;
	// end inline asm
	add.s32 	%r148, %r144, 6144;
	// begin inline asm
	cp.async.ca.shared.global [ %r148 + 0 ], [ %rd43 + 0 ], 0x8, %r145;
	// end inline asm
	cp.async.commit_group;
	.loc	1 831 52                        // sk04_fa_o_w4a8.py:831:52
	shl.b32 	%r385, %r493, 11;
	add.s32 	%r386, %r11, %r385;
	add.s32 	%r149, %r386, 16384;
	// begin inline asm
	cp.async.ca.shared.global [ %r149 + 0 ], [ %rd60 + 0 ], 0x8, %r145;
	// end inline asm
	cp.async.commit_group;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	add.s64 	%rd62, %rd62, 1;
	add.s32 	%r491, %r491, %r30;
	add.s64 	%rd61, %rd61, %rd8;
	add.s64 	%rd60, %rd60, 128;
	setp.ne.b64 	%p7, %rd12, %rd62;
	@%p7 bra 	$L__BB0_2;
$L__BB0_3:                              // %._crit_edge
	.loc	1 0 22                          // sk04_fa_o_w4a8.py:0:22
	cvt.u32.u64 	%r402, %rd7;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r403, %r10, 7;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r404, %r403, %r25;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r405, %r10, 6;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r406, %r405, %r25;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r407, %r10, 5;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r408, %r407, %r25;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r409, %r10, 4;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r410, %r409, %r25;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r411, %r10, 3;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r412, %r411, %r25;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r413, %r10, 2;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r414, %r413, %r25;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r415, %r10, 1;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r416, %r415, %r25;
	.loc	1 816 45                        // sk04_fa_o_w4a8.py:816:45
	shr.u32 	%r417, %r4, 2;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r418, %r417, %r1;
	or.b32 	%r419, %r418, 8;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r420, %r419, %r24;
	rem.s32 	%r421, %r418, %r24;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 838 38                        // sk04_fa_o_w4a8.py:838:38
	mad.wide.s32 	%rd47, %r421, 4, %rd17;
	mad.wide.s32 	%rd48, %r420, 4, %rd17;
	.loc	1 838 24                        // sk04_fa_o_w4a8.py:838:24
	// begin inline asm
	mov.u32 %r387, 0x0;
	ld.global.b32 { %r387 }, [ %rd47 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r388, 0x0;
	ld.global.b32 { %r388 }, [ %rd48 + 0 ];
	// end inline asm
	.loc	1 839 49                        // sk04_fa_o_w4a8.py:839:49
	mul.lo.s32 	%r422, %r6, %r28;
	.loc	1 839 31                        // sk04_fa_o_w4a8.py:839:31
	mad.wide.s32 	%rd58, %r422, 2, %rd16;
	.loc	1 839 82                        // sk04_fa_o_w4a8.py:839:82
	mul.lo.s32 	%r423, %r402, %r29;
	mul.lo.s32 	%r424, %r416, %r29;
	mul.lo.s32 	%r425, %r414, %r29;
	mul.lo.s32 	%r426, %r412, %r29;
	mul.lo.s32 	%r427, %r410, %r29;
	mul.lo.s32 	%r428, %r408, %r29;
	mul.lo.s32 	%r429, %r406, %r29;
	mul.lo.s32 	%r430, %r404, %r29;
	.loc	1 839 64                        // sk04_fa_o_w4a8.py:839:64
	mad.wide.s32 	%rd49, %r423, 2, %rd58;
	mad.wide.s32 	%rd50, %r424, 2, %rd58;
	mad.wide.s32 	%rd51, %r425, 2, %rd58;
	mad.wide.s32 	%rd52, %r426, 2, %rd58;
	mad.wide.s32 	%rd53, %r427, 2, %rd58;
	mad.wide.s32 	%rd54, %r428, 2, %rd58;
	mad.wide.s32 	%rd55, %r429, 2, %rd58;
	mad.wide.s32 	%rd56, %r430, 2, %rd58;
	.loc	1 839 19                        // sk04_fa_o_w4a8.py:839:19
	// begin inline asm
	mov.u16 %rs161, 0x0;
	ld.global.b16 { %rs161 }, [ %rd49 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs162, 0x0;
	ld.global.b16 { %rs162 }, [ %rd50 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs163, 0x0;
	ld.global.b16 { %rs163 }, [ %rd51 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs164, 0x0;
	ld.global.b16 { %rs164 }, [ %rd52 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs165, 0x0;
	ld.global.b16 { %rs165 }, [ %rd53 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs166, 0x0;
	ld.global.b16 { %rs166 }, [ %rd54 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs167, 0x0;
	ld.global.b16 { %rs167 }, [ %rd55 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs168, 0x0;
	ld.global.b16 { %rs168 }, [ %rd56 + 0 ];
	// end inline asm
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	and.b32 	%r431, %r2, 120;
	shl.b32 	%r432, %r431, 5;
	and.b32 	%r433, %r2, 7;
	shl.b32 	%r434, %r433, 4;
	or.b32 	%r435, %r432, %r434;
	xor.b32 	%r436, %r435, %r3;
	add.s32 	%r389, %r81, %r436;
	mov.b32 	%r390, {%rs161, %rs162};
	mov.b32 	%r391, {%rs163, %rs164};
	mov.b32 	%r392, {%rs165, %rs166};
	mov.b32 	%r393, {%rs167, %rs168};
	// begin inline asm
	st.shared.v4.b32 [ %r389 + 0 ], { %r390, %r391, %r392, %r393 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r437, %r433, 9;
	shl.b32 	%r438, %r2, 4;
	and.b32 	%r439, %r438, 496;
	shr.u32 	%r440, %r9, 1;
	xor.b32 	%r441, %r439, %r440;
	add.s32 	%r442, %r81, %r437;
	add.s32 	%r443, %r442, %r441;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r444, %r445, %r446, %r447}, [%r443];
	mov.b32 	{%rs177, %rs178}, %r444;
	mov.b32 	{%rs179, %rs180}, %r445;
	mov.b32 	{%rs181, %rs182}, %r446;
	mov.b32 	{%rs183, %rs184}, %r447;
	cvt.f32.bf16 	%r448, %rs177;
	cvt.f32.bf16 	%r449, %rs178;
	cvt.f32.bf16 	%r450, %rs179;
	cvt.f32.bf16 	%r451, %rs180;
	cvt.f32.bf16 	%r452, %rs181;
	cvt.f32.bf16 	%r453, %rs182;
	cvt.f32.bf16 	%r454, %rs183;
	cvt.f32.bf16 	%r455, %rs184;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r456, %r494, %r387, %r448;
	fma.rn.f32 	%r457, %r495, %r387, %r449;
	fma.rn.f32 	%r458, %r496, %r388, %r450;
	fma.rn.f32 	%r459, %r497, %r388, %r451;
	fma.rn.f32 	%r460, %r498, %r387, %r452;
	fma.rn.f32 	%r461, %r499, %r387, %r453;
	fma.rn.f32 	%r462, %r500, %r388, %r454;
	fma.rn.f32 	%r463, %r501, %r388, %r455;
	.loc	1 846 31                        // sk04_fa_o_w4a8.py:846:31
	setp.lt.s32 	%p9, %r5, %r24;
	.loc	1 846 54                        // sk04_fa_o_w4a8.py:846:54
	setp.lt.s32 	%p10, %r10, %r25;
	.loc	1 846 37                        // sk04_fa_o_w4a8.py:846:37
	and.pred 	%p8, %p9, %p10;
	.loc	1 844 35                        // sk04_fa_o_w4a8.py:844:35
	mul.lo.s32 	%r464, %r5, %r27;
	.loc	1 844 18                        // sk04_fa_o_w4a8.py:844:18
	mad.wide.s32 	%rd59, %r464, 2, %rd15;
	.loc	1 844 50                        // sk04_fa_o_w4a8.py:844:50
	mad.wide.s32 	%rd57, %r10, 2, %rd59;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16.f32 	%rs169, %r456;
	cvt.rn.bf16.f32 	%rs170, %r457;
	cvt.rn.bf16.f32 	%rs171, %r458;
	cvt.rn.bf16.f32 	%rs172, %r459;
	cvt.rn.bf16.f32 	%rs173, %r460;
	cvt.rn.bf16.f32 	%rs174, %r461;
	cvt.rn.bf16.f32 	%rs175, %r462;
	cvt.rn.bf16.f32 	%rs176, %r463;
	bar.sync 	0;
	shl.b32 	%r465, %r2, 5;
	and.b32 	%r466, %r465, 768;
	shl.b32 	%r467, %r4, 1;
	and.b32 	%r468, %r2, 1;
	neg.s32 	%r469, %r468;
	and.b32 	%r470, %r469, 1088;
	bfe.s32 	%r471, %r2, 1, 1;
	and.b32 	%r472, %r471, 2052;
	or.b32 	%r473, %r466, %r467;
	or.b32 	%r474, %r470, %r473;
	xor.b32 	%r475, %r474, %r440;
	or.b32 	%r476, %r475, %r472;
	add.s32 	%r394, %r81, %r476;
	// begin inline asm
	st.shared.v2.b16 [ %r394 + 0 ], { %rs169, %rs170 };
	// end inline asm
	add.s32 	%r395, %r394, 128;
	// begin inline asm
	st.shared.v2.b16 [ %r395 + 0 ], { %rs171, %rs172 };
	// end inline asm
	xor.b32 	%r477, %r476, 4;
	add.s32 	%r396, %r81, %r477;
	// begin inline asm
	st.shared.v2.b16 [ %r396 + 0 ], { %rs173, %rs174 };
	// end inline asm
	add.s32 	%r397, %r396, 128;
	// begin inline asm
	st.shared.v2.b16 [ %r397 + 0 ], { %rs175, %rs176 };
	// end inline asm
	bar.sync 	0;
	and.b32 	%r478, %r8, 768;
	shr.u32 	%r479, %r431, 1;
	and.b32 	%r480, %r2, 128;
	or.b32 	%r481, %r434, %r478;
	xor.b32 	%r482, %r481, %r479;
	or.b32 	%r483, %r482, %r480;
	add.s32 	%r484, %r81, %r483;
	ld.shared.b32 	%r398, [%r484];
	xor.b32 	%r485, %r483, 64;
	add.s32 	%r486, %r81, %r485;
	ld.shared.b32 	%r399, [%r486+1024];
	xor.b32 	%r487, %r483, 4;
	add.s32 	%r488, %r81, %r487;
	ld.shared.b32 	%r400, [%r488+2048];
	xor.b32 	%r489, %r483, 68;
	add.s32 	%r490, %r81, %r489;
	ld.shared.b32 	%r401, [%r490+3072];
	.loc	1 845 8                         // sk04_fa_o_w4a8.py:845:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd57 + 0 ], { %r398, %r399, %r400, %r401 };
	// end inline asm
	.loc	1 843 4                         // sk04_fa_o_w4a8.py:843:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk04_fa_o_w4a8.py"
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
.b8 5                                   // DW_FORM_data2
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 165                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x9e DW_TAG_compile_unit
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
.b8 52
.b8 95
.b8 102
.b8 97
.b8 95
.b8 111
.b8 95
.b8 119
.b8 52
.b8 97
.b8 56
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
.b8 2                                   // Abbrev [2] 0x47:0x19 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 52
.b8 95
.b8 102
.b8 97
.b8 95
.b8 111
.b8 95
.b8 119
.b8 52
.b8 97
.b8 56
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x60:0x48 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 71                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x75:0x19 DW_TAG_inlined_subroutine
.b32 71                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 40                                  // DW_AT_call_line
.b8 3
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x8e:0x19 DW_TAG_inlined_subroutine
.b32 71                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 41                                  // DW_AT_call_line
.b8 3
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_1 = _Nativo(
    "sk04_fa_o_w4a8/tile16x128x128_shift0_abi15",
    _PTX_1, "_sk04_fa_o_w4a8_kernel",
    warps=8, shared=20480,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 16, 17],
    horneado={10: 1, 12: 1, 14: 1, 18: 1, 19: 16, 20: 128, 21: 128, 22: 8},
    div16=[7, 8, 9, 11, 13, 15, 16, 17],
)

_PTX_2 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk04_fa_o_w4a8_kernel  // -- Begin function _sk04_fa_o_w4a8_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk04_fa_o_w4a8_kernel
.visible .entry _sk04_fa_o_w4a8_kernel(
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_5,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_6,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_7,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_8,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_9,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_10,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_11,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_12,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_13,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_14,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_15
)
.reqntid 128
{
	.reg .pred 	%p<25>;
	.reg .b16 	%rs<385>;
	.reg .b32 	%r<1314>;
	.reg .b64 	%rd<141>;
	.loc	1 797 0                         // sk04_fa_o_w4a8.py:797:0
$L__func_begin0:
	.loc	1 797 0                         // sk04_fa_o_w4a8.py:797:0

// %bb.0:
	ld.param.b32 	%r51, [_sk04_fa_o_w4a8_kernel_param_12];
	ld.param.b32 	%r50, [_sk04_fa_o_w4a8_kernel_param_11];
	ld.param.b32 	%r49, [_sk04_fa_o_w4a8_kernel_param_8];
	ld.param.b32 	%r48, [_sk04_fa_o_w4a8_kernel_param_7];
	ld.param.b32 	%r47, [_sk04_fa_o_w4a8_kernel_param_6];
	ld.param.b64 	%rd27, [_sk04_fa_o_w4a8_kernel_param_4];
	ld.param.b64 	%rd26, [_sk04_fa_o_w4a8_kernel_param_3];
	ld.param.b64 	%rd25, [_sk04_fa_o_w4a8_kernel_param_2];
	ld.param.b64 	%rd24, [_sk04_fa_o_w4a8_kernel_param_1];
	ld.param.b64 	%rd23, [_sk04_fa_o_w4a8_kernel_param_0];
$L__tmp0:
	.loc	1 807 24                        // sk04_fa_o_w4a8.py:807:24
	mov.u32 	%r95, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk04_fa_o_w4a8.py:808:27 ]
	add.s32 	%r96, %r47, 63;
	.loc	2 43 30                         // standard.py:43:30 @[ sk04_fa_o_w4a8.py:808:27 ]
	shr.s32 	%r97, %r96, 31;
	shr.u32 	%r98, %r97, 26;
	add.s32 	%r99, %r96, %r98;
	shr.s32 	%r100, %r99, 6;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk04_fa_o_w4a8.py:809:27 ]
	add.s32 	%r101, %r48, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk04_fa_o_w4a8.py:809:27 ]
	shr.s32 	%r102, %r101, 31;
	shr.u32 	%r103, %r102, 25;
	add.s32 	%r104, %r101, %r103;
	shr.s32 	%r105, %r104, 7;
$L__tmp3:
	.loc	1 810 29                        // sk04_fa_o_w4a8.py:810:29
	shl.b32 	%r106, %r105, 3;
	.loc	1 811 22                        // sk04_fa_o_w4a8.py:811:22
	div.s32 	%r107, %r95, %r106;
	.loc	1 811 38                        // sk04_fa_o_w4a8.py:811:38
	shl.b32 	%r108, %r107, 3;
	ld.param.b32 	%r109, [_sk04_fa_o_w4a8_kernel_param_9];
	.loc	1 812 30                        // sk04_fa_o_w4a8.py:812:30
	sub.s32 	%r110, %r100, %r108;
	ld.param.b32 	%r111, [_sk04_fa_o_w4a8_kernel_param_10];
	.loc	1 812 39                        // sk04_fa_o_w4a8.py:812:39
	min.s32 	%r112, %r110, 8;
	.loc	1 813 30                        // sk04_fa_o_w4a8.py:813:30
	mul.lo.s32 	%r113, %r107, %r106;
	sub.s32 	%r114, %r95, %r113;
	.loc	1 814 36                        // sk04_fa_o_w4a8.py:814:36
	div.s32 	%r115, %r114, %r112;
	.loc	1 813 46                        // sk04_fa_o_w4a8.py:813:46
	mul.lo.s32 	%r116, %r115, %r112;
	sub.s32 	%r117, %r114, %r116;
	.loc	1 813 23                        // sk04_fa_o_w4a8.py:813:23
	add.s32 	%r118, %r117, %r108;
	.loc	1 816 22                        // sk04_fa_o_w4a8.py:816:22
	shl.b32 	%r1, %r118, 6;
	.loc	1 816 45                        // sk04_fa_o_w4a8.py:816:45
	mov.u32 	%r2, %tid.x;
	and.b32 	%r119, %r2, 112;
	bfe.u32 	%r3, %r2, 4, 3;
	or.b32 	%r4, %r3, 8;
	or.b32 	%r5, %r3, 16;
	or.b32 	%r6, %r3, 24;
	or.b32 	%r7, %r3, 32;
	or.b32 	%r8, %r3, 40;
	or.b32 	%r9, %r3, 48;
	or.b32 	%r10, %r3, 56;
	and.b32 	%r11, %r2, 120;
	bfe.u32 	%r120, %r2, 3, 4;
	and.b32 	%r12, %r2, 28;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r121, %r1, %r120;
	or.b32 	%r122, %r121, 16;
	or.b32 	%r123, %r121, 32;
	or.b32 	%r124, %r121, 48;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r13, %r121, %r47;
	rem.s32 	%r14, %r122, %r47;
	rem.s32 	%r15, %r123, %r47;
	rem.s32 	%r16, %r124, %r47;
	.loc	1 817 22                        // sk04_fa_o_w4a8.py:817:22
	shl.b32 	%r17, %r115, 7;
	.loc	1 817 45                        // sk04_fa_o_w4a8.py:817:45
	and.b32 	%r18, %r2, 15;
	shl.b32 	%r19, %r18, 3;
	and.b32 	%r20, %r2, 7;
	shl.b32 	%r125, %r20, 4;
	and.b32 	%r21, %r2, 127;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r22, %r17, %r19;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r126, %r22, %r48;
	.loc	1 821 39                        // sk04_fa_o_w4a8.py:821:39
	mul.lo.s32 	%r127, %r13, %r109;
	mul.lo.s32 	%r128, %r14, %r109;
	mul.lo.s32 	%r129, %r15, %r109;
	mul.lo.s32 	%r130, %r16, %r109;
	.loc	1 821 21                        // sk04_fa_o_w4a8.py:821:21
	cvt.s64.s32 	%rd1, %r127;
	add.s64 	%rd65, %rd23, %rd1;
	cvt.s64.s32 	%rd2, %r128;
	add.s64 	%rd66, %rd23, %rd2;
	cvt.s64.s32 	%rd3, %r129;
	add.s64 	%rd67, %rd23, %rd3;
	cvt.s64.s32 	%rd4, %r130;
	add.s64 	%rd68, %rd23, %rd4;
	.loc	1 821 51                        // sk04_fa_o_w4a8.py:821:51
	cvt.u64.u32 	%rd5, %r125;
	add.s64 	%rd37, %rd65, %rd5;
	add.s64 	%rd38, %rd66, %rd5;
	add.s64 	%rd39, %rd67, %rd5;
	add.s64 	%rd40, %rd68, %rd5;
	.loc	1 822 40                        // sk04_fa_o_w4a8.py:822:40
	mul.lo.s32 	%r131, %r111, %r3;
	shl.b32 	%r132, %r111, 3;
	add.s32 	%r133, %r131, %r132;
	shl.b32 	%r134, %r111, 4;
	add.s32 	%r135, %r131, %r134;
	mad.lo.s32 	%r136, %r111, 24, %r131;
	shl.b32 	%r137, %r111, 5;
	add.s32 	%r138, %r131, %r137;
	mad.lo.s32 	%r139, %r111, 40, %r131;
	mad.lo.s32 	%r140, %r111, 48, %r131;
	mad.lo.s32 	%r141, %r111, 56, %r131;
	.loc	1 822 21                        // sk04_fa_o_w4a8.py:822:21
	cvt.s64.s32 	%rd6, %r131;
	add.s64 	%rd69, %rd24, %rd6;
	cvt.s64.s32 	%rd7, %r133;
	add.s64 	%rd70, %rd24, %rd7;
	cvt.s64.s32 	%rd8, %r135;
	add.s64 	%rd71, %rd24, %rd8;
	cvt.s64.s32 	%rd9, %r136;
	add.s64 	%rd72, %rd24, %rd9;
	cvt.s64.s32 	%rd10, %r138;
	add.s64 	%rd73, %rd24, %rd10;
	cvt.s64.s32 	%rd11, %r139;
	add.s64 	%rd74, %rd24, %rd11;
	cvt.s64.s32 	%rd12, %r140;
	add.s64 	%rd75, %rd24, %rd12;
	cvt.s64.s32 	%rd13, %r141;
	add.s64 	%rd76, %rd24, %rd13;
	.loc	1 822 52                        // sk04_fa_o_w4a8.py:822:52
	cvt.s64.s32 	%rd14, %r126;
	add.s64 	%rd29, %rd69, %rd14;
	add.s64 	%rd33, %rd70, %rd14;
	add.s64 	%rd30, %rd71, %rd14;
	add.s64 	%rd34, %rd72, %rd14;
	add.s64 	%rd31, %rd73, %rd14;
	add.s64 	%rd35, %rd74, %rd14;
	add.s64 	%rd32, %rd75, %rd14;
	add.s64 	%rd36, %rd76, %rd14;
	.loc	1 836 35                        // sk04_fa_o_w4a8.py:836:35
	shl.b32 	%r142, %r111, 6;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	setp.gt.s32 	%p1, %r49, 127;
	.loc	1 827 25                        // sk04_fa_o_w4a8.py:827:25
	shl.b32 	%r143, %r21, 3;
	shr.u32 	%r144, %r119, 1;
	xor.b32 	%r145, %r143, %r144;
	mov.b32 	%r146, global_smem;
	add.s32 	%r53, %r146, %r145;
	selp.b32 	%r54, 8, 0, %p1;
	// begin inline asm
	cp.async.ca.shared.global [ %r53 + 0 ], [ %rd29 + 0 ], 0x8, %r54;
	// end inline asm
	add.s32 	%r55, %r53, 2048;
	// begin inline asm
	cp.async.ca.shared.global [ %r55 + 0 ], [ %rd30 + 0 ], 0x8, %r54;
	// end inline asm
	add.s32 	%r56, %r53, 4096;
	// begin inline asm
	cp.async.ca.shared.global [ %r56 + 0 ], [ %rd31 + 0 ], 0x8, %r54;
	// end inline asm
	add.s32 	%r57, %r53, 6144;
	// begin inline asm
	cp.async.ca.shared.global [ %r57 + 0 ], [ %rd32 + 0 ], 0x8, %r54;
	// end inline asm
	xor.b32 	%r23, %r145, 64;
	add.s32 	%r147, %r146, %r23;
	add.s32 	%r58, %r147, 1024;
	// begin inline asm
	cp.async.ca.shared.global [ %r58 + 0 ], [ %rd33 + 0 ], 0x8, %r54;
	// end inline asm
	add.s32 	%r59, %r147, 3072;
	// begin inline asm
	cp.async.ca.shared.global [ %r59 + 0 ], [ %rd34 + 0 ], 0x8, %r54;
	// end inline asm
	add.s32 	%r60, %r147, 5120;
	// begin inline asm
	cp.async.ca.shared.global [ %r60 + 0 ], [ %rd35 + 0 ], 0x8, %r54;
	// end inline asm
	add.s32 	%r61, %r147, 7168;
	// begin inline asm
	cp.async.ca.shared.global [ %r61 + 0 ], [ %rd36 + 0 ], 0x8, %r54;
	// end inline asm
	cp.async.commit_group;
	.loc	1 831 52                        // sk04_fa_o_w4a8.py:831:52
	shl.b32 	%r148, %r21, 4;
	and.b32 	%r24, %r2, 24;
	shl.b32 	%r149, %r24, 2;
	xor.b32 	%r150, %r148, %r149;
	add.s32 	%r25, %r146, %r150;
	add.s32 	%r62, %r25, 24576;
	selp.b32 	%r63, 16, 0, %p1;
	// begin inline asm
	cp.async.cg.shared.global [ %r62 + 0 ], [ %rd37 + 0 ], 0x10, %r63;
	// end inline asm
	add.s32 	%r64, %r25, 26624;
	// begin inline asm
	cp.async.cg.shared.global [ %r64 + 0 ], [ %rd38 + 0 ], 0x10, %r63;
	// end inline asm
	add.s32 	%r65, %r25, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r65 + 0 ], [ %rd39 + 0 ], 0x10, %r63;
	// end inline asm
	add.s32 	%r66, %r25, 30720;
	// begin inline asm
	cp.async.cg.shared.global [ %r66 + 0 ], [ %rd40 + 0 ], 0x10, %r63;
	// end inline asm
	cp.async.commit_group;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	setp.gt.s32 	%p2, %r49, 255;
	.loc	1 835 18                        // sk04_fa_o_w4a8.py:835:18
	add.s64 	%rd49, %rd37, 128;
	add.s64 	%rd50, %rd38, 128;
	add.s64 	%rd51, %rd39, 128;
	add.s64 	%rd52, %rd40, 128;
	.loc	1 836 18                        // sk04_fa_o_w4a8.py:836:18
	cvt.s64.s32 	%rd15, %r142;
	add.s64 	%rd41, %rd29, %rd15;
	add.s64 	%rd45, %rd33, %rd15;
	add.s64 	%rd42, %rd30, %rd15;
	add.s64 	%rd46, %rd34, %rd15;
	add.s64 	%rd43, %rd31, %rd15;
	add.s64 	%rd47, %rd35, %rd15;
	add.s64 	%rd44, %rd32, %rd15;
	add.s64 	%rd48, %rd36, %rd15;
	.loc	1 827 25                        // sk04_fa_o_w4a8.py:827:25
	bar.sync 	0;
	add.s32 	%r67, %r53, 8192;
	selp.b32 	%r68, 8, 0, %p2;
	// begin inline asm
	cp.async.ca.shared.global [ %r67 + 0 ], [ %rd41 + 0 ], 0x8, %r68;
	// end inline asm
	add.s32 	%r69, %r53, 10240;
	// begin inline asm
	cp.async.ca.shared.global [ %r69 + 0 ], [ %rd42 + 0 ], 0x8, %r68;
	// end inline asm
	add.s32 	%r70, %r53, 12288;
	// begin inline asm
	cp.async.ca.shared.global [ %r70 + 0 ], [ %rd43 + 0 ], 0x8, %r68;
	// end inline asm
	add.s32 	%r71, %r53, 14336;
	// begin inline asm
	cp.async.ca.shared.global [ %r71 + 0 ], [ %rd44 + 0 ], 0x8, %r68;
	// end inline asm
	add.s32 	%r72, %r147, 9216;
	// begin inline asm
	cp.async.ca.shared.global [ %r72 + 0 ], [ %rd45 + 0 ], 0x8, %r68;
	// end inline asm
	add.s32 	%r73, %r147, 11264;
	// begin inline asm
	cp.async.ca.shared.global [ %r73 + 0 ], [ %rd46 + 0 ], 0x8, %r68;
	// end inline asm
	add.s32 	%r74, %r147, 13312;
	// begin inline asm
	cp.async.ca.shared.global [ %r74 + 0 ], [ %rd47 + 0 ], 0x8, %r68;
	// end inline asm
	add.s32 	%r75, %r147, 15360;
	// begin inline asm
	cp.async.ca.shared.global [ %r75 + 0 ], [ %rd48 + 0 ], 0x8, %r68;
	// end inline asm
	cp.async.commit_group;
	.loc	1 831 52                        // sk04_fa_o_w4a8.py:831:52
	add.s32 	%r76, %r25, 32768;
	selp.b32 	%r77, 16, 0, %p2;
	// begin inline asm
	cp.async.cg.shared.global [ %r76 + 0 ], [ %rd49 + 0 ], 0x10, %r77;
	// end inline asm
	add.s32 	%r78, %r25, 34816;
	// begin inline asm
	cp.async.cg.shared.global [ %r78 + 0 ], [ %rd50 + 0 ], 0x10, %r77;
	// end inline asm
	add.s32 	%r79, %r25, 36864;
	// begin inline asm
	cp.async.cg.shared.global [ %r79 + 0 ], [ %rd51 + 0 ], 0x10, %r77;
	// end inline asm
	add.s32 	%r80, %r25, 38912;
	// begin inline asm
	cp.async.cg.shared.global [ %r80 + 0 ], [ %rd52 + 0 ], 0x10, %r77;
	// end inline asm
	cp.async.commit_group;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	setp.gt.s32 	%p3, %r49, 383;
	.loc	1 835 18                        // sk04_fa_o_w4a8.py:835:18
	add.s64 	%rd61, %rd37, 256;
	add.s64 	%rd62, %rd38, 256;
	add.s64 	%rd63, %rd39, 256;
	add.s64 	%rd64, %rd40, 256;
	.loc	1 836 18                        // sk04_fa_o_w4a8.py:836:18
	add.s64 	%rd53, %rd41, %rd15;
	add.s64 	%rd57, %rd45, %rd15;
	add.s64 	%rd54, %rd42, %rd15;
	add.s64 	%rd58, %rd46, %rd15;
	add.s64 	%rd55, %rd43, %rd15;
	add.s64 	%rd59, %rd47, %rd15;
	add.s64 	%rd56, %rd44, %rd15;
	add.s64 	%rd60, %rd48, %rd15;
	.loc	1 827 25                        // sk04_fa_o_w4a8.py:827:25
	bar.sync 	0;
	add.s32 	%r81, %r53, 16384;
	selp.b32 	%r82, 8, 0, %p3;
	// begin inline asm
	cp.async.ca.shared.global [ %r81 + 0 ], [ %rd53 + 0 ], 0x8, %r82;
	// end inline asm
	add.s32 	%r83, %r53, 18432;
	// begin inline asm
	cp.async.ca.shared.global [ %r83 + 0 ], [ %rd54 + 0 ], 0x8, %r82;
	// end inline asm
	add.s32 	%r84, %r53, 20480;
	// begin inline asm
	cp.async.ca.shared.global [ %r84 + 0 ], [ %rd55 + 0 ], 0x8, %r82;
	// end inline asm
	add.s32 	%r85, %r53, 22528;
	// begin inline asm
	cp.async.ca.shared.global [ %r85 + 0 ], [ %rd56 + 0 ], 0x8, %r82;
	// end inline asm
	add.s32 	%r86, %r147, 17408;
	// begin inline asm
	cp.async.ca.shared.global [ %r86 + 0 ], [ %rd57 + 0 ], 0x8, %r82;
	// end inline asm
	add.s32 	%r87, %r147, 19456;
	// begin inline asm
	cp.async.ca.shared.global [ %r87 + 0 ], [ %rd58 + 0 ], 0x8, %r82;
	// end inline asm
	add.s32 	%r88, %r147, 21504;
	// begin inline asm
	cp.async.ca.shared.global [ %r88 + 0 ], [ %rd59 + 0 ], 0x8, %r82;
	// end inline asm
	add.s32 	%r89, %r147, 23552;
	// begin inline asm
	cp.async.ca.shared.global [ %r89 + 0 ], [ %rd60 + 0 ], 0x8, %r82;
	// end inline asm
	cp.async.commit_group;
	.loc	1 831 52                        // sk04_fa_o_w4a8.py:831:52
	add.s32 	%r90, %r25, 40960;
	selp.b32 	%r91, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r90 + 0 ], [ %rd61 + 0 ], 0x10, %r91;
	// end inline asm
	add.s32 	%r92, %r25, 43008;
	// begin inline asm
	cp.async.cg.shared.global [ %r92 + 0 ], [ %rd62 + 0 ], 0x10, %r91;
	// end inline asm
	add.s32 	%r93, %r25, 45056;
	// begin inline asm
	cp.async.cg.shared.global [ %r93 + 0 ], [ %rd63 + 0 ], 0x10, %r91;
	// end inline asm
	add.s32 	%r94, %r25, 47104;
	// begin inline asm
	cp.async.cg.shared.global [ %r94 + 0 ], [ %rd64 + 0 ], 0x10, %r91;
	// end inline asm
	cp.async.commit_group;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	@%p1 bra 	$L__BB0_2;
	bra.uni 	$L__BB0_1;
$L__BB0_2:                              // %.lr.ph
	.loc	1 0 22                          // sk04_fa_o_w4a8.py:0:22
	ld.param.b32 	%r52, [_sk04_fa_o_w4a8_kernel_param_13];
	ld.param.b64 	%rd28, [_sk04_fa_o_w4a8_kernel_param_5];
	.loc	1 825 27                        // sk04_fa_o_w4a8.py:825:27
	shr.u32 	%r151, %r49, 7;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r152, %r17, %r21;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r153, %r152, %r48;
	add.s32 	%r154, %r151, -3;
	and.b32 	%r1248, %r2, 3;
	shr.u32 	%r1249, %r2, 2;
	bfe.u32 	%r155, %r2, 2, 5;
	mul.lo.s32 	%r156, %r1248, 544;
	or.b32 	%r26, %r156, %r155;
	xor.b32 	%r27, %r26, 136;
	xor.b32 	%r28, %r26, 272;
	xor.b32 	%r29, %r26, 408;
	xor.b32 	%r30, %r26, 32;
	xor.b32 	%r31, %r26, 168;
	xor.b32 	%r32, %r26, 304;
	xor.b32 	%r33, %r26, 440;
	xor.b32 	%r34, %r26, 64;
	xor.b32 	%r35, %r26, 200;
	xor.b32 	%r36, %r26, 336;
	xor.b32 	%r37, %r26, 472;
	xor.b32 	%r38, %r26, 96;
	xor.b32 	%r39, %r26, 232;
	xor.b32 	%r40, %r26, 368;
	xor.b32 	%r41, %r26, 504;
	shl.b32 	%r157, %r12, 5;
	or.b32 	%r42, %r19, %r157;
	xor.b32 	%r43, %r42, 32;
	xor.b32 	%r44, %r42, 64;
	xor.b32 	%r45, %r42, 96;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	mad.wide.s32 	%rd16, %r153, 4, %rd28;
	shl.b32 	%r158, %r21, 2;
	add.s32 	%r159, %r146, %r158;
	add.s32 	%r325, %r159, 49152;
	shl.b32 	%r160, %r1248, 3;
	and.b32 	%r1247, %r2, 96;
	add.s32 	%r161, %r146, %r160;
	add.s32 	%r46, %r161, %r1247;
	cvt.s64.s32 	%rd17, %r154;
	and.b32 	%r162, %r49, -128;
	cvt.u64.u32 	%rd18, %r162;
	mad.lo.s64 	%rd77, %rd15, 3, %rd14;
	add.s64 	%rd138, %rd24, %rd77;
	add.s64 	%rd78, %rd5, %rd4;
	add.s64 	%rd79, %rd78, %rd23;
	add.s64 	%rd19, %rd79, 384;
	add.s64 	%rd80, %rd5, %rd3;
	add.s64 	%rd81, %rd80, %rd23;
	add.s64 	%rd20, %rd81, 384;
	add.s64 	%rd82, %rd5, %rd2;
	add.s64 	%rd83, %rd82, %rd23;
	add.s64 	%rd21, %rd83, 384;
	add.s64 	%rd84, %rd5, %rd1;
	add.s64 	%rd85, %rd84, %rd23;
	add.s64 	%rd22, %rd85, 384;
	mov.b32 	%r1250, 0f00000000;
	mov.b32 	%r1246, 2;
	mov.b32 	%r1245, -1;
	mov.b64 	%rd139, 0;
	mov.b32 	%r163, 0;
	mov.b32 	%r1244, %r163;
	mov.b64 	%rd140, %rd139;
	mov.b32 	%r1251, %r1250;
	mov.b32 	%r1252, %r1250;
	mov.b32 	%r1253, %r1250;
	mov.b32 	%r1254, %r1250;
	mov.b32 	%r1255, %r1250;
	mov.b32 	%r1256, %r1250;
	mov.b32 	%r1257, %r1250;
	mov.b32 	%r1258, %r1250;
	mov.b32 	%r1259, %r1250;
	mov.b32 	%r1260, %r1250;
	mov.b32 	%r1261, %r1250;
	mov.b32 	%r1262, %r1250;
	mov.b32 	%r1263, %r1250;
	mov.b32 	%r1264, %r1250;
	mov.b32 	%r1265, %r1250;
	mov.b32 	%r1266, %r1250;
	mov.b32 	%r1267, %r1250;
	mov.b32 	%r1268, %r1250;
	mov.b32 	%r1269, %r1250;
	mov.b32 	%r1270, %r1250;
	mov.b32 	%r1271, %r1250;
	mov.b32 	%r1272, %r1250;
	mov.b32 	%r1273, %r1250;
	mov.b32 	%r1274, %r1250;
	mov.b32 	%r1275, %r1250;
	mov.b32 	%r1276, %r1250;
	mov.b32 	%r1277, %r1250;
	mov.b32 	%r1278, %r1250;
	mov.b32 	%r1279, %r1250;
	mov.b32 	%r1280, %r1250;
	mov.b32 	%r1281, %r1250;
	mov.b32 	%r1282, %r1250;
	mov.b32 	%r1283, %r1250;
	mov.b32 	%r1284, %r1250;
	mov.b32 	%r1285, %r1250;
	mov.b32 	%r1286, %r1250;
	mov.b32 	%r1287, %r1250;
	mov.b32 	%r1288, %r1250;
	mov.b32 	%r1289, %r1250;
	mov.b32 	%r1290, %r1250;
	mov.b32 	%r1291, %r1250;
	mov.b32 	%r1292, %r1250;
	mov.b32 	%r1293, %r1250;
	mov.b32 	%r1294, %r1250;
	mov.b32 	%r1295, %r1250;
	mov.b32 	%r1296, %r1250;
	mov.b32 	%r1297, %r1250;
	mov.b32 	%r1298, %r1250;
	mov.b32 	%r1299, %r1250;
	mov.b32 	%r1300, %r1250;
	mov.b32 	%r1301, %r1250;
	mov.b32 	%r1302, %r1250;
	mov.b32 	%r1303, %r1250;
	mov.b32 	%r1304, %r1250;
	mov.b32 	%r1305, %r1250;
	mov.b32 	%r1306, %r1250;
	mov.b32 	%r1307, %r1250;
	mov.b32 	%r1308, %r1250;
	mov.b32 	%r1309, %r1250;
	mov.b32 	%r1310, %r1250;
	mov.b32 	%r1311, %r1250;
	mov.b32 	%r1312, %r1250;
	mov.b32 	%r1313, %r1250;
$L__BB0_3:                              // =>This Inner Loop Header: Depth=1
	setp.lt.s64 	%p4, %rd140, %rd17;
	add.s32 	%r340, %r1245, 1;
	setp.gt.s32 	%p5, %r340, 2;
	selp.b32 	%r1245, 0, %r340, %p5;
	.loc	1 827 25                        // sk04_fa_o_w4a8.py:827:25
	cp.async.wait_group 	4;
	bar.sync 	0;
	shl.b32 	%r341, %r1245, 13;
	add.s32 	%r342, %r146, %r341;
	.loc	1 828 38                        // sk04_fa_o_w4a8.py:828:38
	add.s32 	%r343, %r342, %r26;
	ld.shared.b8 	%rs1, [%r343];
	ld.shared.b8 	%rs2, [%r343+2048];
	ld.shared.b8 	%rs3, [%r343+4096];
	ld.shared.b8 	%rs4, [%r343+6144];
	add.s32 	%r344, %r342, %r27;
	ld.shared.b8 	%rs5, [%r344];
	ld.shared.b8 	%rs6, [%r344+2048];
	ld.shared.b8 	%rs7, [%r344+4096];
	ld.shared.b8 	%rs8, [%r344+6144];
	add.s32 	%r345, %r342, %r28;
	ld.shared.b8 	%rs9, [%r345];
	ld.shared.b8 	%rs10, [%r345+2048];
	ld.shared.b8 	%rs11, [%r345+4096];
	ld.shared.b8 	%rs12, [%r345+6144];
	add.s32 	%r346, %r342, %r29;
	ld.shared.b8 	%rs13, [%r346];
	ld.shared.b8 	%rs14, [%r346+2048];
	ld.shared.b8 	%rs15, [%r346+4096];
	ld.shared.b8 	%rs16, [%r346+6144];
	add.s32 	%r347, %r342, %r30;
	ld.shared.b8 	%rs17, [%r347];
	ld.shared.b8 	%rs18, [%r347+2048];
	ld.shared.b8 	%rs19, [%r347+4096];
	ld.shared.b8 	%rs20, [%r347+6144];
	add.s32 	%r348, %r342, %r31;
	ld.shared.b8 	%rs21, [%r348];
	ld.shared.b8 	%rs22, [%r348+2048];
	ld.shared.b8 	%rs23, [%r348+4096];
	ld.shared.b8 	%rs24, [%r348+6144];
	add.s32 	%r349, %r342, %r32;
	ld.shared.b8 	%rs25, [%r349];
	ld.shared.b8 	%rs26, [%r349+2048];
	ld.shared.b8 	%rs27, [%r349+4096];
	ld.shared.b8 	%rs28, [%r349+6144];
	add.s32 	%r350, %r342, %r33;
	ld.shared.b8 	%rs29, [%r350];
	ld.shared.b8 	%rs30, [%r350+2048];
	ld.shared.b8 	%rs31, [%r350+4096];
	ld.shared.b8 	%rs32, [%r350+6144];
	add.s32 	%r351, %r342, %r34;
	ld.shared.b8 	%rs33, [%r351];
	ld.shared.b8 	%rs34, [%r351+2048];
	ld.shared.b8 	%rs35, [%r351+4096];
	ld.shared.b8 	%rs36, [%r351+6144];
	add.s32 	%r352, %r342, %r35;
	ld.shared.b8 	%rs37, [%r352];
	ld.shared.b8 	%rs38, [%r352+2048];
	ld.shared.b8 	%rs39, [%r352+4096];
	ld.shared.b8 	%rs40, [%r352+6144];
	add.s32 	%r353, %r342, %r36;
	ld.shared.b8 	%rs41, [%r353];
	ld.shared.b8 	%rs42, [%r353+2048];
	ld.shared.b8 	%rs43, [%r353+4096];
	ld.shared.b8 	%rs44, [%r353+6144];
	add.s32 	%r354, %r342, %r37;
	ld.shared.b8 	%rs45, [%r354];
	ld.shared.b8 	%rs46, [%r354+2048];
	ld.shared.b8 	%rs47, [%r354+4096];
	ld.shared.b8 	%rs48, [%r354+6144];
	add.s32 	%r355, %r342, %r38;
	ld.shared.b8 	%rs49, [%r355];
	ld.shared.b8 	%rs50, [%r355+2048];
	ld.shared.b8 	%rs51, [%r355+4096];
	ld.shared.b8 	%rs52, [%r355+6144];
	add.s32 	%r356, %r342, %r39;
	ld.shared.b8 	%rs53, [%r356];
	ld.shared.b8 	%rs54, [%r356+2048];
	ld.shared.b8 	%rs55, [%r356+4096];
	ld.shared.b8 	%rs56, [%r356+6144];
	add.s32 	%r357, %r342, %r40;
	ld.shared.b8 	%rs57, [%r357];
	ld.shared.b8 	%rs58, [%r357+2048];
	ld.shared.b8 	%rs59, [%r357+4096];
	ld.shared.b8 	%rs60, [%r357+6144];
	add.s32 	%r358, %r342, %r41;
	ld.shared.b8 	%rs61, [%r358];
	ld.shared.b8 	%rs62, [%r358+2048];
	ld.shared.b8 	%rs63, [%r358+4096];
	ld.shared.b8 	%rs64, [%r358+6144];
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r359, %rs13;
	cvt.u32.u16 	%r360, %rs9;
	prmt.b32 	%r361, %r360, %r359, 0x3340U;
	cvt.u32.u16 	%r362, %rs5;
	cvt.u32.u16 	%r363, %rs1;
	prmt.b32 	%r364, %r363, %r362, 0x3340U;
	prmt.b32 	%r365, %r364, %r361, 0x5410U;
	and.b32 	%r366, %r365, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r367, %r366, 0, 0x7773U;
	cvt.u16.u32 	%rs65, %r367;
	add.s16 	%rs66, %rs65, -8;
	cvt.u32.u16 	%r368, %rs66;
	prmt.b32 	%r369, %r366, 0, 0x7772U;
	cvt.u16.u32 	%rs67, %r369;
	add.s16 	%rs68, %rs67, -8;
	cvt.u32.u16 	%r370, %rs68;
	prmt.b32 	%r371, %r370, %r368, 0x3340U;
	prmt.b32 	%r372, %r366, 0, 0x7771U;
	cvt.u16.u32 	%rs69, %r372;
	add.s16 	%rs70, %rs69, -8;
	cvt.u32.u16 	%r373, %rs70;
	prmt.b32 	%r374, %r366, 0, 0x7770U;
	cvt.u16.u32 	%rs71, %r374;
	add.s16 	%rs72, %rs71, -8;
	cvt.u32.u16 	%r375, %rs72;
	prmt.b32 	%r376, %r375, %r373, 0x3340U;
	prmt.b32 	%r168, %r376, %r371, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r377, %rs14;
	cvt.u32.u16 	%r378, %rs10;
	prmt.b32 	%r379, %r378, %r377, 0x3340U;
	cvt.u32.u16 	%r380, %rs6;
	cvt.u32.u16 	%r381, %rs2;
	prmt.b32 	%r382, %r381, %r380, 0x3340U;
	prmt.b32 	%r383, %r382, %r379, 0x5410U;
	and.b32 	%r384, %r383, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r385, %r384, 0, 0x7773U;
	cvt.u16.u32 	%rs73, %r385;
	add.s16 	%rs74, %rs73, -8;
	cvt.u32.u16 	%r386, %rs74;
	prmt.b32 	%r387, %r384, 0, 0x7772U;
	cvt.u16.u32 	%rs75, %r387;
	add.s16 	%rs76, %rs75, -8;
	cvt.u32.u16 	%r388, %rs76;
	prmt.b32 	%r389, %r388, %r386, 0x3340U;
	prmt.b32 	%r390, %r384, 0, 0x7771U;
	cvt.u16.u32 	%rs77, %r390;
	add.s16 	%rs78, %rs77, -8;
	cvt.u32.u16 	%r391, %rs78;
	prmt.b32 	%r392, %r384, 0, 0x7770U;
	cvt.u16.u32 	%rs79, %r392;
	add.s16 	%rs80, %rs79, -8;
	cvt.u32.u16 	%r393, %rs80;
	prmt.b32 	%r394, %r393, %r391, 0x3340U;
	prmt.b32 	%r169, %r394, %r389, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r395, %rs15;
	cvt.u32.u16 	%r396, %rs11;
	prmt.b32 	%r397, %r396, %r395, 0x3340U;
	cvt.u32.u16 	%r398, %rs7;
	cvt.u32.u16 	%r399, %rs3;
	prmt.b32 	%r400, %r399, %r398, 0x3340U;
	prmt.b32 	%r401, %r400, %r397, 0x5410U;
	and.b32 	%r402, %r401, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r403, %r402, 0, 0x7773U;
	cvt.u16.u32 	%rs81, %r403;
	add.s16 	%rs82, %rs81, -8;
	cvt.u32.u16 	%r404, %rs82;
	prmt.b32 	%r405, %r402, 0, 0x7772U;
	cvt.u16.u32 	%rs83, %r405;
	add.s16 	%rs84, %rs83, -8;
	cvt.u32.u16 	%r406, %rs84;
	prmt.b32 	%r407, %r406, %r404, 0x3340U;
	prmt.b32 	%r408, %r402, 0, 0x7771U;
	cvt.u16.u32 	%rs85, %r408;
	add.s16 	%rs86, %rs85, -8;
	cvt.u32.u16 	%r409, %rs86;
	prmt.b32 	%r410, %r402, 0, 0x7770U;
	cvt.u16.u32 	%rs87, %r410;
	add.s16 	%rs88, %rs87, -8;
	cvt.u32.u16 	%r411, %rs88;
	prmt.b32 	%r412, %r411, %r409, 0x3340U;
	prmt.b32 	%r212, %r412, %r407, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r413, %rs16;
	cvt.u32.u16 	%r414, %rs12;
	prmt.b32 	%r415, %r414, %r413, 0x3340U;
	cvt.u32.u16 	%r416, %rs8;
	cvt.u32.u16 	%r417, %rs4;
	prmt.b32 	%r418, %r417, %r416, 0x3340U;
	prmt.b32 	%r419, %r418, %r415, 0x5410U;
	and.b32 	%r420, %r419, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r421, %r420, 0, 0x7773U;
	cvt.u16.u32 	%rs89, %r421;
	add.s16 	%rs90, %rs89, -8;
	cvt.u32.u16 	%r422, %rs90;
	prmt.b32 	%r423, %r420, 0, 0x7772U;
	cvt.u16.u32 	%rs91, %r423;
	add.s16 	%rs92, %rs91, -8;
	cvt.u32.u16 	%r424, %rs92;
	prmt.b32 	%r425, %r424, %r422, 0x3340U;
	prmt.b32 	%r426, %r420, 0, 0x7771U;
	cvt.u16.u32 	%rs93, %r426;
	add.s16 	%rs94, %rs93, -8;
	cvt.u32.u16 	%r427, %rs94;
	prmt.b32 	%r428, %r420, 0, 0x7770U;
	cvt.u16.u32 	%rs95, %r428;
	add.s16 	%rs96, %rs95, -8;
	cvt.u32.u16 	%r429, %rs96;
	prmt.b32 	%r430, %r429, %r427, 0x3340U;
	prmt.b32 	%r213, %r430, %r425, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r431, %rs29;
	cvt.u32.u16 	%r432, %rs25;
	prmt.b32 	%r433, %r432, %r431, 0x3340U;
	cvt.u32.u16 	%r434, %rs21;
	cvt.u32.u16 	%r435, %rs17;
	prmt.b32 	%r436, %r435, %r434, 0x3340U;
	prmt.b32 	%r437, %r436, %r433, 0x5410U;
	and.b32 	%r438, %r437, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r439, %r438, 0, 0x7773U;
	cvt.u16.u32 	%rs97, %r439;
	add.s16 	%rs98, %rs97, -8;
	cvt.u32.u16 	%r440, %rs98;
	prmt.b32 	%r441, %r438, 0, 0x7772U;
	cvt.u16.u32 	%rs99, %r441;
	add.s16 	%rs100, %rs99, -8;
	cvt.u32.u16 	%r442, %rs100;
	prmt.b32 	%r443, %r442, %r440, 0x3340U;
	prmt.b32 	%r444, %r438, 0, 0x7771U;
	cvt.u16.u32 	%rs101, %r444;
	add.s16 	%rs102, %rs101, -8;
	cvt.u32.u16 	%r445, %rs102;
	prmt.b32 	%r446, %r438, 0, 0x7770U;
	cvt.u16.u32 	%rs103, %r446;
	add.s16 	%rs104, %rs103, -8;
	cvt.u32.u16 	%r447, %rs104;
	prmt.b32 	%r448, %r447, %r445, 0x3340U;
	prmt.b32 	%r174, %r448, %r443, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r449, %rs30;
	cvt.u32.u16 	%r450, %rs26;
	prmt.b32 	%r451, %r450, %r449, 0x3340U;
	cvt.u32.u16 	%r452, %rs22;
	cvt.u32.u16 	%r453, %rs18;
	prmt.b32 	%r454, %r453, %r452, 0x3340U;
	prmt.b32 	%r455, %r454, %r451, 0x5410U;
	and.b32 	%r456, %r455, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r457, %r456, 0, 0x7773U;
	cvt.u16.u32 	%rs105, %r457;
	add.s16 	%rs106, %rs105, -8;
	cvt.u32.u16 	%r458, %rs106;
	prmt.b32 	%r459, %r456, 0, 0x7772U;
	cvt.u16.u32 	%rs107, %r459;
	add.s16 	%rs108, %rs107, -8;
	cvt.u32.u16 	%r460, %rs108;
	prmt.b32 	%r461, %r460, %r458, 0x3340U;
	prmt.b32 	%r462, %r456, 0, 0x7771U;
	cvt.u16.u32 	%rs109, %r462;
	add.s16 	%rs110, %rs109, -8;
	cvt.u32.u16 	%r463, %rs110;
	prmt.b32 	%r464, %r456, 0, 0x7770U;
	cvt.u16.u32 	%rs111, %r464;
	add.s16 	%rs112, %rs111, -8;
	cvt.u32.u16 	%r465, %rs112;
	prmt.b32 	%r466, %r465, %r463, 0x3340U;
	prmt.b32 	%r175, %r466, %r461, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r467, %rs31;
	cvt.u32.u16 	%r468, %rs27;
	prmt.b32 	%r469, %r468, %r467, 0x3340U;
	cvt.u32.u16 	%r470, %rs23;
	cvt.u32.u16 	%r471, %rs19;
	prmt.b32 	%r472, %r471, %r470, 0x3340U;
	prmt.b32 	%r473, %r472, %r469, 0x5410U;
	and.b32 	%r474, %r473, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r475, %r474, 0, 0x7773U;
	cvt.u16.u32 	%rs113, %r475;
	add.s16 	%rs114, %rs113, -8;
	cvt.u32.u16 	%r476, %rs114;
	prmt.b32 	%r477, %r474, 0, 0x7772U;
	cvt.u16.u32 	%rs115, %r477;
	add.s16 	%rs116, %rs115, -8;
	cvt.u32.u16 	%r478, %rs116;
	prmt.b32 	%r479, %r478, %r476, 0x3340U;
	prmt.b32 	%r480, %r474, 0, 0x7771U;
	cvt.u16.u32 	%rs117, %r480;
	add.s16 	%rs118, %rs117, -8;
	cvt.u32.u16 	%r481, %rs118;
	prmt.b32 	%r482, %r474, 0, 0x7770U;
	cvt.u16.u32 	%rs119, %r482;
	add.s16 	%rs120, %rs119, -8;
	cvt.u32.u16 	%r483, %rs120;
	prmt.b32 	%r484, %r483, %r481, 0x3340U;
	prmt.b32 	%r222, %r484, %r479, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r485, %rs32;
	cvt.u32.u16 	%r486, %rs28;
	prmt.b32 	%r487, %r486, %r485, 0x3340U;
	cvt.u32.u16 	%r488, %rs24;
	cvt.u32.u16 	%r489, %rs20;
	prmt.b32 	%r490, %r489, %r488, 0x3340U;
	prmt.b32 	%r491, %r490, %r487, 0x5410U;
	and.b32 	%r492, %r491, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r493, %r492, 0, 0x7773U;
	cvt.u16.u32 	%rs121, %r493;
	add.s16 	%rs122, %rs121, -8;
	cvt.u32.u16 	%r494, %rs122;
	prmt.b32 	%r495, %r492, 0, 0x7772U;
	cvt.u16.u32 	%rs123, %r495;
	add.s16 	%rs124, %rs123, -8;
	cvt.u32.u16 	%r496, %rs124;
	prmt.b32 	%r497, %r496, %r494, 0x3340U;
	prmt.b32 	%r498, %r492, 0, 0x7771U;
	cvt.u16.u32 	%rs125, %r498;
	add.s16 	%rs126, %rs125, -8;
	cvt.u32.u16 	%r499, %rs126;
	prmt.b32 	%r500, %r492, 0, 0x7770U;
	cvt.u16.u32 	%rs127, %r500;
	add.s16 	%rs128, %rs127, -8;
	cvt.u32.u16 	%r501, %rs128;
	prmt.b32 	%r502, %r501, %r499, 0x3340U;
	prmt.b32 	%r223, %r502, %r497, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r503, %rs45;
	cvt.u32.u16 	%r504, %rs41;
	prmt.b32 	%r505, %r504, %r503, 0x3340U;
	cvt.u32.u16 	%r506, %rs37;
	cvt.u32.u16 	%r507, %rs33;
	prmt.b32 	%r508, %r507, %r506, 0x3340U;
	prmt.b32 	%r509, %r508, %r505, 0x5410U;
	and.b32 	%r510, %r509, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r511, %r510, 0, 0x7773U;
	cvt.u16.u32 	%rs129, %r511;
	add.s16 	%rs130, %rs129, -8;
	cvt.u32.u16 	%r512, %rs130;
	prmt.b32 	%r513, %r510, 0, 0x7772U;
	cvt.u16.u32 	%rs131, %r513;
	add.s16 	%rs132, %rs131, -8;
	cvt.u32.u16 	%r514, %rs132;
	prmt.b32 	%r515, %r514, %r512, 0x3340U;
	prmt.b32 	%r516, %r510, 0, 0x7771U;
	cvt.u16.u32 	%rs133, %r516;
	add.s16 	%rs134, %rs133, -8;
	cvt.u32.u16 	%r517, %rs134;
	prmt.b32 	%r518, %r510, 0, 0x7770U;
	cvt.u16.u32 	%rs135, %r518;
	add.s16 	%rs136, %rs135, -8;
	cvt.u32.u16 	%r519, %rs136;
	prmt.b32 	%r520, %r519, %r517, 0x3340U;
	prmt.b32 	%r176, %r520, %r515, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r521, %rs46;
	cvt.u32.u16 	%r522, %rs42;
	prmt.b32 	%r523, %r522, %r521, 0x3340U;
	cvt.u32.u16 	%r524, %rs38;
	cvt.u32.u16 	%r525, %rs34;
	prmt.b32 	%r526, %r525, %r524, 0x3340U;
	prmt.b32 	%r527, %r526, %r523, 0x5410U;
	and.b32 	%r528, %r527, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r529, %r528, 0, 0x7773U;
	cvt.u16.u32 	%rs137, %r529;
	add.s16 	%rs138, %rs137, -8;
	cvt.u32.u16 	%r530, %rs138;
	prmt.b32 	%r531, %r528, 0, 0x7772U;
	cvt.u16.u32 	%rs139, %r531;
	add.s16 	%rs140, %rs139, -8;
	cvt.u32.u16 	%r532, %rs140;
	prmt.b32 	%r533, %r532, %r530, 0x3340U;
	prmt.b32 	%r534, %r528, 0, 0x7771U;
	cvt.u16.u32 	%rs141, %r534;
	add.s16 	%rs142, %rs141, -8;
	cvt.u32.u16 	%r535, %rs142;
	prmt.b32 	%r536, %r528, 0, 0x7770U;
	cvt.u16.u32 	%rs143, %r536;
	add.s16 	%rs144, %rs143, -8;
	cvt.u32.u16 	%r537, %rs144;
	prmt.b32 	%r538, %r537, %r535, 0x3340U;
	prmt.b32 	%r177, %r538, %r533, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r539, %rs47;
	cvt.u32.u16 	%r540, %rs43;
	prmt.b32 	%r541, %r540, %r539, 0x3340U;
	cvt.u32.u16 	%r542, %rs39;
	cvt.u32.u16 	%r543, %rs35;
	prmt.b32 	%r544, %r543, %r542, 0x3340U;
	prmt.b32 	%r545, %r544, %r541, 0x5410U;
	and.b32 	%r546, %r545, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r547, %r546, 0, 0x7773U;
	cvt.u16.u32 	%rs145, %r547;
	add.s16 	%rs146, %rs145, -8;
	cvt.u32.u16 	%r548, %rs146;
	prmt.b32 	%r549, %r546, 0, 0x7772U;
	cvt.u16.u32 	%rs147, %r549;
	add.s16 	%rs148, %rs147, -8;
	cvt.u32.u16 	%r550, %rs148;
	prmt.b32 	%r551, %r550, %r548, 0x3340U;
	prmt.b32 	%r552, %r546, 0, 0x7771U;
	cvt.u16.u32 	%rs149, %r552;
	add.s16 	%rs150, %rs149, -8;
	cvt.u32.u16 	%r553, %rs150;
	prmt.b32 	%r554, %r546, 0, 0x7770U;
	cvt.u16.u32 	%rs151, %r554;
	add.s16 	%rs152, %rs151, -8;
	cvt.u32.u16 	%r555, %rs152;
	prmt.b32 	%r556, %r555, %r553, 0x3340U;
	prmt.b32 	%r228, %r556, %r551, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r557, %rs48;
	cvt.u32.u16 	%r558, %rs44;
	prmt.b32 	%r559, %r558, %r557, 0x3340U;
	cvt.u32.u16 	%r560, %rs40;
	cvt.u32.u16 	%r561, %rs36;
	prmt.b32 	%r562, %r561, %r560, 0x3340U;
	prmt.b32 	%r563, %r562, %r559, 0x5410U;
	and.b32 	%r564, %r563, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r565, %r564, 0, 0x7773U;
	cvt.u16.u32 	%rs153, %r565;
	add.s16 	%rs154, %rs153, -8;
	cvt.u32.u16 	%r566, %rs154;
	prmt.b32 	%r567, %r564, 0, 0x7772U;
	cvt.u16.u32 	%rs155, %r567;
	add.s16 	%rs156, %rs155, -8;
	cvt.u32.u16 	%r568, %rs156;
	prmt.b32 	%r569, %r568, %r566, 0x3340U;
	prmt.b32 	%r570, %r564, 0, 0x7771U;
	cvt.u16.u32 	%rs157, %r570;
	add.s16 	%rs158, %rs157, -8;
	cvt.u32.u16 	%r571, %rs158;
	prmt.b32 	%r572, %r564, 0, 0x7770U;
	cvt.u16.u32 	%rs159, %r572;
	add.s16 	%rs160, %rs159, -8;
	cvt.u32.u16 	%r573, %rs160;
	prmt.b32 	%r574, %r573, %r571, 0x3340U;
	prmt.b32 	%r229, %r574, %r569, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r575, %rs61;
	cvt.u32.u16 	%r576, %rs57;
	prmt.b32 	%r577, %r576, %r575, 0x3340U;
	cvt.u32.u16 	%r578, %rs53;
	cvt.u32.u16 	%r579, %rs49;
	prmt.b32 	%r580, %r579, %r578, 0x3340U;
	prmt.b32 	%r581, %r580, %r577, 0x5410U;
	and.b32 	%r582, %r581, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r583, %r582, 0, 0x7773U;
	cvt.u16.u32 	%rs161, %r583;
	add.s16 	%rs162, %rs161, -8;
	cvt.u32.u16 	%r584, %rs162;
	prmt.b32 	%r585, %r582, 0, 0x7772U;
	cvt.u16.u32 	%rs163, %r585;
	add.s16 	%rs164, %rs163, -8;
	cvt.u32.u16 	%r586, %rs164;
	prmt.b32 	%r587, %r586, %r584, 0x3340U;
	prmt.b32 	%r588, %r582, 0, 0x7771U;
	cvt.u16.u32 	%rs165, %r588;
	add.s16 	%rs166, %rs165, -8;
	cvt.u32.u16 	%r589, %rs166;
	prmt.b32 	%r590, %r582, 0, 0x7770U;
	cvt.u16.u32 	%rs167, %r590;
	add.s16 	%rs168, %rs167, -8;
	cvt.u32.u16 	%r591, %rs168;
	prmt.b32 	%r592, %r591, %r589, 0x3340U;
	prmt.b32 	%r178, %r592, %r587, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r593, %rs62;
	cvt.u32.u16 	%r594, %rs58;
	prmt.b32 	%r595, %r594, %r593, 0x3340U;
	cvt.u32.u16 	%r596, %rs54;
	cvt.u32.u16 	%r597, %rs50;
	prmt.b32 	%r598, %r597, %r596, 0x3340U;
	prmt.b32 	%r599, %r598, %r595, 0x5410U;
	and.b32 	%r600, %r599, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r601, %r600, 0, 0x7773U;
	cvt.u16.u32 	%rs169, %r601;
	add.s16 	%rs170, %rs169, -8;
	cvt.u32.u16 	%r602, %rs170;
	prmt.b32 	%r603, %r600, 0, 0x7772U;
	cvt.u16.u32 	%rs171, %r603;
	add.s16 	%rs172, %rs171, -8;
	cvt.u32.u16 	%r604, %rs172;
	prmt.b32 	%r605, %r604, %r602, 0x3340U;
	prmt.b32 	%r606, %r600, 0, 0x7771U;
	cvt.u16.u32 	%rs173, %r606;
	add.s16 	%rs174, %rs173, -8;
	cvt.u32.u16 	%r607, %rs174;
	prmt.b32 	%r608, %r600, 0, 0x7770U;
	cvt.u16.u32 	%rs175, %r608;
	add.s16 	%rs176, %rs175, -8;
	cvt.u32.u16 	%r609, %rs176;
	prmt.b32 	%r610, %r609, %r607, 0x3340U;
	prmt.b32 	%r179, %r610, %r605, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r611, %rs63;
	cvt.u32.u16 	%r612, %rs59;
	prmt.b32 	%r613, %r612, %r611, 0x3340U;
	cvt.u32.u16 	%r614, %rs55;
	cvt.u32.u16 	%r615, %rs51;
	prmt.b32 	%r616, %r615, %r614, 0x3340U;
	prmt.b32 	%r617, %r616, %r613, 0x5410U;
	and.b32 	%r618, %r617, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r619, %r618, 0, 0x7773U;
	cvt.u16.u32 	%rs177, %r619;
	add.s16 	%rs178, %rs177, -8;
	cvt.u32.u16 	%r620, %rs178;
	prmt.b32 	%r621, %r618, 0, 0x7772U;
	cvt.u16.u32 	%rs179, %r621;
	add.s16 	%rs180, %rs179, -8;
	cvt.u32.u16 	%r622, %rs180;
	prmt.b32 	%r623, %r622, %r620, 0x3340U;
	prmt.b32 	%r624, %r618, 0, 0x7771U;
	cvt.u16.u32 	%rs181, %r624;
	add.s16 	%rs182, %rs181, -8;
	cvt.u32.u16 	%r625, %rs182;
	prmt.b32 	%r626, %r618, 0, 0x7770U;
	cvt.u16.u32 	%rs183, %r626;
	add.s16 	%rs184, %rs183, -8;
	cvt.u32.u16 	%r627, %rs184;
	prmt.b32 	%r628, %r627, %r625, 0x3340U;
	prmt.b32 	%r234, %r628, %r623, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r629, %rs64;
	cvt.u32.u16 	%r630, %rs60;
	prmt.b32 	%r631, %r630, %r629, 0x3340U;
	cvt.u32.u16 	%r632, %rs56;
	cvt.u32.u16 	%r633, %rs52;
	prmt.b32 	%r634, %r633, %r632, 0x3340U;
	prmt.b32 	%r635, %r634, %r631, 0x5410U;
	and.b32 	%r636, %r635, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r637, %r636, 0, 0x7773U;
	cvt.u16.u32 	%rs185, %r637;
	add.s16 	%rs186, %rs185, -8;
	cvt.u32.u16 	%r638, %rs186;
	prmt.b32 	%r639, %r636, 0, 0x7772U;
	cvt.u16.u32 	%rs187, %r639;
	add.s16 	%rs188, %rs187, -8;
	cvt.u32.u16 	%r640, %rs188;
	prmt.b32 	%r641, %r640, %r638, 0x3340U;
	prmt.b32 	%r642, %r636, 0, 0x7771U;
	cvt.u16.u32 	%rs189, %r642;
	add.s16 	%rs190, %rs189, -8;
	cvt.u32.u16 	%r643, %rs190;
	prmt.b32 	%r644, %r636, 0, 0x7770U;
	cvt.u16.u32 	%rs191, %r644;
	add.s16 	%rs192, %rs191, -8;
	cvt.u32.u16 	%r645, %rs192;
	prmt.b32 	%r646, %r645, %r643, 0x3340U;
	prmt.b32 	%r235, %r646, %r641, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs193, %rs1, 4;
	shr.u16 	%rs194, %rs5, 4;
	shr.u16 	%rs195, %rs9, 4;
	shr.u16 	%rs196, %rs13, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs197, %rs196, -8;
	cvt.u32.u16 	%r647, %rs197;
	add.s16 	%rs198, %rs195, -8;
	cvt.u32.u16 	%r648, %rs198;
	prmt.b32 	%r649, %r648, %r647, 0x3340U;
	add.s16 	%rs199, %rs194, -8;
	cvt.u32.u16 	%r650, %rs199;
	add.s16 	%rs200, %rs193, -8;
	cvt.u32.u16 	%r651, %rs200;
	prmt.b32 	%r652, %r651, %r650, 0x3340U;
	prmt.b32 	%r280, %r652, %r649, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs201, %rs2, 4;
	shr.u16 	%rs202, %rs6, 4;
	shr.u16 	%rs203, %rs10, 4;
	shr.u16 	%rs204, %rs14, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs205, %rs204, -8;
	cvt.u32.u16 	%r653, %rs205;
	add.s16 	%rs206, %rs203, -8;
	cvt.u32.u16 	%r654, %rs206;
	prmt.b32 	%r655, %r654, %r653, 0x3340U;
	add.s16 	%rs207, %rs202, -8;
	cvt.u32.u16 	%r656, %rs207;
	add.s16 	%rs208, %rs201, -8;
	cvt.u32.u16 	%r657, %rs208;
	prmt.b32 	%r658, %r657, %r656, 0x3340U;
	prmt.b32 	%r281, %r658, %r655, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs209, %rs3, 4;
	shr.u16 	%rs210, %rs7, 4;
	shr.u16 	%rs211, %rs11, 4;
	shr.u16 	%rs212, %rs15, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs213, %rs212, -8;
	cvt.u32.u16 	%r659, %rs213;
	add.s16 	%rs214, %rs211, -8;
	cvt.u32.u16 	%r660, %rs214;
	prmt.b32 	%r661, %r660, %r659, 0x3340U;
	add.s16 	%rs215, %rs210, -8;
	cvt.u32.u16 	%r662, %rs215;
	add.s16 	%rs216, %rs209, -8;
	cvt.u32.u16 	%r663, %rs216;
	prmt.b32 	%r664, %r663, %r662, 0x3340U;
	prmt.b32 	%r304, %r664, %r661, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs217, %rs4, 4;
	shr.u16 	%rs218, %rs8, 4;
	shr.u16 	%rs219, %rs12, 4;
	shr.u16 	%rs220, %rs16, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs221, %rs220, -8;
	cvt.u32.u16 	%r665, %rs221;
	add.s16 	%rs222, %rs219, -8;
	cvt.u32.u16 	%r666, %rs222;
	prmt.b32 	%r667, %r666, %r665, 0x3340U;
	add.s16 	%rs223, %rs218, -8;
	cvt.u32.u16 	%r668, %rs223;
	add.s16 	%rs224, %rs217, -8;
	cvt.u32.u16 	%r669, %rs224;
	prmt.b32 	%r670, %r669, %r668, 0x3340U;
	prmt.b32 	%r305, %r670, %r667, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs225, %rs17, 4;
	shr.u16 	%rs226, %rs21, 4;
	shr.u16 	%rs227, %rs25, 4;
	shr.u16 	%rs228, %rs29, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs229, %rs228, -8;
	cvt.u32.u16 	%r671, %rs229;
	add.s16 	%rs230, %rs227, -8;
	cvt.u32.u16 	%r672, %rs230;
	prmt.b32 	%r673, %r672, %r671, 0x3340U;
	add.s16 	%rs231, %rs226, -8;
	cvt.u32.u16 	%r674, %rs231;
	add.s16 	%rs232, %rs225, -8;
	cvt.u32.u16 	%r675, %rs232;
	prmt.b32 	%r676, %r675, %r674, 0x3340U;
	prmt.b32 	%r286, %r676, %r673, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs233, %rs18, 4;
	shr.u16 	%rs234, %rs22, 4;
	shr.u16 	%rs235, %rs26, 4;
	shr.u16 	%rs236, %rs30, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs237, %rs236, -8;
	cvt.u32.u16 	%r677, %rs237;
	add.s16 	%rs238, %rs235, -8;
	cvt.u32.u16 	%r678, %rs238;
	prmt.b32 	%r679, %r678, %r677, 0x3340U;
	add.s16 	%rs239, %rs234, -8;
	cvt.u32.u16 	%r680, %rs239;
	add.s16 	%rs240, %rs233, -8;
	cvt.u32.u16 	%r681, %rs240;
	prmt.b32 	%r682, %r681, %r680, 0x3340U;
	prmt.b32 	%r287, %r682, %r679, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs241, %rs19, 4;
	shr.u16 	%rs242, %rs23, 4;
	shr.u16 	%rs243, %rs27, 4;
	shr.u16 	%rs244, %rs31, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs245, %rs244, -8;
	cvt.u32.u16 	%r683, %rs245;
	add.s16 	%rs246, %rs243, -8;
	cvt.u32.u16 	%r684, %rs246;
	prmt.b32 	%r685, %r684, %r683, 0x3340U;
	add.s16 	%rs247, %rs242, -8;
	cvt.u32.u16 	%r686, %rs247;
	add.s16 	%rs248, %rs241, -8;
	cvt.u32.u16 	%r687, %rs248;
	prmt.b32 	%r688, %r687, %r686, 0x3340U;
	prmt.b32 	%r310, %r688, %r685, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs249, %rs20, 4;
	shr.u16 	%rs250, %rs24, 4;
	shr.u16 	%rs251, %rs28, 4;
	shr.u16 	%rs252, %rs32, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs253, %rs252, -8;
	cvt.u32.u16 	%r689, %rs253;
	add.s16 	%rs254, %rs251, -8;
	cvt.u32.u16 	%r690, %rs254;
	prmt.b32 	%r691, %r690, %r689, 0x3340U;
	add.s16 	%rs255, %rs250, -8;
	cvt.u32.u16 	%r692, %rs255;
	add.s16 	%rs256, %rs249, -8;
	cvt.u32.u16 	%r693, %rs256;
	prmt.b32 	%r694, %r693, %r692, 0x3340U;
	prmt.b32 	%r311, %r694, %r691, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs257, %rs33, 4;
	shr.u16 	%rs258, %rs37, 4;
	shr.u16 	%rs259, %rs41, 4;
	shr.u16 	%rs260, %rs45, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs261, %rs260, -8;
	cvt.u32.u16 	%r695, %rs261;
	add.s16 	%rs262, %rs259, -8;
	cvt.u32.u16 	%r696, %rs262;
	prmt.b32 	%r697, %r696, %r695, 0x3340U;
	add.s16 	%rs263, %rs258, -8;
	cvt.u32.u16 	%r698, %rs263;
	add.s16 	%rs264, %rs257, -8;
	cvt.u32.u16 	%r699, %rs264;
	prmt.b32 	%r700, %r699, %r698, 0x3340U;
	prmt.b32 	%r288, %r700, %r697, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs265, %rs34, 4;
	shr.u16 	%rs266, %rs38, 4;
	shr.u16 	%rs267, %rs42, 4;
	shr.u16 	%rs268, %rs46, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs269, %rs268, -8;
	cvt.u32.u16 	%r701, %rs269;
	add.s16 	%rs270, %rs267, -8;
	cvt.u32.u16 	%r702, %rs270;
	prmt.b32 	%r703, %r702, %r701, 0x3340U;
	add.s16 	%rs271, %rs266, -8;
	cvt.u32.u16 	%r704, %rs271;
	add.s16 	%rs272, %rs265, -8;
	cvt.u32.u16 	%r705, %rs272;
	prmt.b32 	%r706, %r705, %r704, 0x3340U;
	prmt.b32 	%r289, %r706, %r703, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs273, %rs35, 4;
	shr.u16 	%rs274, %rs39, 4;
	shr.u16 	%rs275, %rs43, 4;
	shr.u16 	%rs276, %rs47, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs277, %rs276, -8;
	cvt.u32.u16 	%r707, %rs277;
	add.s16 	%rs278, %rs275, -8;
	cvt.u32.u16 	%r708, %rs278;
	prmt.b32 	%r709, %r708, %r707, 0x3340U;
	add.s16 	%rs279, %rs274, -8;
	cvt.u32.u16 	%r710, %rs279;
	add.s16 	%rs280, %rs273, -8;
	cvt.u32.u16 	%r711, %rs280;
	prmt.b32 	%r712, %r711, %r710, 0x3340U;
	prmt.b32 	%r312, %r712, %r709, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs281, %rs36, 4;
	shr.u16 	%rs282, %rs40, 4;
	shr.u16 	%rs283, %rs44, 4;
	shr.u16 	%rs284, %rs48, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs285, %rs284, -8;
	cvt.u32.u16 	%r713, %rs285;
	add.s16 	%rs286, %rs283, -8;
	cvt.u32.u16 	%r714, %rs286;
	prmt.b32 	%r715, %r714, %r713, 0x3340U;
	add.s16 	%rs287, %rs282, -8;
	cvt.u32.u16 	%r716, %rs287;
	add.s16 	%rs288, %rs281, -8;
	cvt.u32.u16 	%r717, %rs288;
	prmt.b32 	%r718, %r717, %r716, 0x3340U;
	prmt.b32 	%r313, %r718, %r715, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs289, %rs49, 4;
	shr.u16 	%rs290, %rs53, 4;
	shr.u16 	%rs291, %rs57, 4;
	shr.u16 	%rs292, %rs61, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs293, %rs292, -8;
	cvt.u32.u16 	%r719, %rs293;
	add.s16 	%rs294, %rs291, -8;
	cvt.u32.u16 	%r720, %rs294;
	prmt.b32 	%r721, %r720, %r719, 0x3340U;
	add.s16 	%rs295, %rs290, -8;
	cvt.u32.u16 	%r722, %rs295;
	add.s16 	%rs296, %rs289, -8;
	cvt.u32.u16 	%r723, %rs296;
	prmt.b32 	%r724, %r723, %r722, 0x3340U;
	prmt.b32 	%r290, %r724, %r721, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs297, %rs50, 4;
	shr.u16 	%rs298, %rs54, 4;
	shr.u16 	%rs299, %rs58, 4;
	shr.u16 	%rs300, %rs62, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs301, %rs300, -8;
	cvt.u32.u16 	%r725, %rs301;
	add.s16 	%rs302, %rs299, -8;
	cvt.u32.u16 	%r726, %rs302;
	prmt.b32 	%r727, %r726, %r725, 0x3340U;
	add.s16 	%rs303, %rs298, -8;
	cvt.u32.u16 	%r728, %rs303;
	add.s16 	%rs304, %rs297, -8;
	cvt.u32.u16 	%r729, %rs304;
	prmt.b32 	%r730, %r729, %r728, 0x3340U;
	prmt.b32 	%r291, %r730, %r727, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs305, %rs51, 4;
	shr.u16 	%rs306, %rs55, 4;
	shr.u16 	%rs307, %rs59, 4;
	shr.u16 	%rs308, %rs63, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs309, %rs308, -8;
	cvt.u32.u16 	%r731, %rs309;
	add.s16 	%rs310, %rs307, -8;
	cvt.u32.u16 	%r732, %rs310;
	prmt.b32 	%r733, %r732, %r731, 0x3340U;
	add.s16 	%rs311, %rs306, -8;
	cvt.u32.u16 	%r734, %rs311;
	add.s16 	%rs312, %rs305, -8;
	cvt.u32.u16 	%r735, %rs312;
	prmt.b32 	%r736, %r735, %r734, 0x3340U;
	prmt.b32 	%r314, %r736, %r733, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs313, %rs52, 4;
	shr.u16 	%rs314, %rs56, 4;
	shr.u16 	%rs315, %rs60, 4;
	shr.u16 	%rs316, %rs64, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs317, %rs316, -8;
	cvt.u32.u16 	%r737, %rs317;
	add.s16 	%rs318, %rs315, -8;
	cvt.u32.u16 	%r738, %rs318;
	prmt.b32 	%r739, %r738, %r737, 0x3340U;
	add.s16 	%rs319, %rs314, -8;
	cvt.u32.u16 	%r740, %rs319;
	add.s16 	%rs320, %rs313, -8;
	cvt.u32.u16 	%r741, %rs320;
	prmt.b32 	%r742, %r741, %r740, 0x3340U;
	prmt.b32 	%r315, %r742, %r739, 0x5410U;
	.loc	1 831 33                        // sk04_fa_o_w4a8.py:831:33
	add.s32 	%r743, %r342, %r42;
	ld.shared.v2.b32 	{%r744, %r745}, [%r743+24576];
	ld.shared.v2.b32 	{%r746, %r747}, [%r743+25600];
	ld.shared.v2.b32 	{%r748, %r749}, [%r743+26624];
	ld.shared.v2.b32 	{%r750, %r751}, [%r743+27648];
	ld.shared.v2.b32 	{%r752, %r753}, [%r743+28672];
	ld.shared.v2.b32 	{%r754, %r755}, [%r743+29696];
	ld.shared.v2.b32 	{%r756, %r757}, [%r743+30720];
	ld.shared.v2.b32 	{%r758, %r759}, [%r743+31744];
	add.s32 	%r760, %r342, %r43;
	ld.shared.v2.b32 	{%r761, %r762}, [%r760+24576];
	ld.shared.v2.b32 	{%r763, %r764}, [%r760+25600];
	ld.shared.v2.b32 	{%r765, %r766}, [%r760+26624];
	ld.shared.v2.b32 	{%r767, %r768}, [%r760+27648];
	ld.shared.v2.b32 	{%r769, %r770}, [%r760+28672];
	ld.shared.v2.b32 	{%r771, %r772}, [%r760+29696];
	ld.shared.v2.b32 	{%r773, %r774}, [%r760+30720];
	ld.shared.v2.b32 	{%r775, %r776}, [%r760+31744];
	add.s32 	%r777, %r342, %r44;
	ld.shared.v2.b32 	{%r778, %r779}, [%r777+24576];
	ld.shared.v2.b32 	{%r780, %r781}, [%r777+25600];
	ld.shared.v2.b32 	{%r782, %r783}, [%r777+26624];
	ld.shared.v2.b32 	{%r784, %r785}, [%r777+27648];
	ld.shared.v2.b32 	{%r786, %r787}, [%r777+28672];
	ld.shared.v2.b32 	{%r788, %r789}, [%r777+29696];
	ld.shared.v2.b32 	{%r790, %r791}, [%r777+30720];
	ld.shared.v2.b32 	{%r792, %r793}, [%r777+31744];
	add.s32 	%r794, %r342, %r45;
	ld.shared.v2.b32 	{%r795, %r796}, [%r794+24576];
	ld.shared.v2.b32 	{%r797, %r798}, [%r794+25600];
	ld.shared.v2.b32 	{%r799, %r800}, [%r794+26624];
	ld.shared.v2.b32 	{%r801, %r802}, [%r794+27648];
	ld.shared.v2.b32 	{%r803, %r804}, [%r794+28672];
	ld.shared.v2.b32 	{%r805, %r806}, [%r794+29696];
	ld.shared.v2.b32 	{%r807, %r808}, [%r794+30720];
	ld.shared.v2.b32 	{%r809, %r810}, [%r794+31744];
	.loc	1 832 33                        // sk04_fa_o_w4a8.py:832:33
	prmt.b32 	%r164, %r744, %r745, 0x6420U;
	prmt.b32 	%r165, %r746, %r747, 0x6420U;
	prmt.b32 	%r166, %r761, %r762, 0x6420U;
	prmt.b32 	%r167, %r763, %r764, 0x6420U;
	prmt.b32 	%r196, %r778, %r779, 0x6420U;
	prmt.b32 	%r197, %r780, %r781, 0x6420U;
	prmt.b32 	%r198, %r795, %r796, 0x6420U;
	prmt.b32 	%r199, %r797, %r798, 0x6420U;
	prmt.b32 	%r170, %r748, %r749, 0x6420U;
	prmt.b32 	%r171, %r750, %r751, 0x6420U;
	prmt.b32 	%r172, %r765, %r766, 0x6420U;
	prmt.b32 	%r173, %r767, %r768, 0x6420U;
	prmt.b32 	%r218, %r782, %r783, 0x6420U;
	prmt.b32 	%r219, %r784, %r785, 0x6420U;
	prmt.b32 	%r220, %r799, %r800, 0x6420U;
	prmt.b32 	%r221, %r801, %r802, 0x6420U;
	prmt.b32 	%r180, %r752, %r753, 0x6420U;
	prmt.b32 	%r181, %r754, %r755, 0x6420U;
	prmt.b32 	%r182, %r769, %r770, 0x6420U;
	prmt.b32 	%r183, %r771, %r772, 0x6420U;
	prmt.b32 	%r244, %r786, %r787, 0x6420U;
	prmt.b32 	%r245, %r788, %r789, 0x6420U;
	prmt.b32 	%r246, %r803, %r804, 0x6420U;
	prmt.b32 	%r247, %r805, %r806, 0x6420U;
	prmt.b32 	%r184, %r756, %r757, 0x6420U;
	prmt.b32 	%r185, %r758, %r759, 0x6420U;
	prmt.b32 	%r186, %r773, %r774, 0x6420U;
	prmt.b32 	%r187, %r775, %r776, 0x6420U;
	prmt.b32 	%r264, %r790, %r791, 0x6420U;
	prmt.b32 	%r265, %r792, %r793, 0x6420U;
	prmt.b32 	%r266, %r807, %r808, 0x6420U;
	prmt.b32 	%r267, %r809, %r810, 0x6420U;
	mov.b32 	%r188, %r163;
	mov.b32 	%r189, %r163;
	mov.b32 	%r190, %r163;
	mov.b32 	%r191, %r163;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r188, %r189, %r190, %r191 }, { %r164, %r165, %r166, %r167 }, { %r168, %r169 }, { %r188, %r189, %r190, %r191 };
	// end inline asm
	mov.b32 	%r192, %r163;
	mov.b32 	%r193, %r163;
	mov.b32 	%r194, %r163;
	mov.b32 	%r195, %r163;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r192, %r193, %r194, %r195 }, { %r164, %r165, %r166, %r167 }, { %r174, %r175 }, { %r192, %r193, %r194, %r195 };
	// end inline asm
	mov.b32 	%r200, %r163;
	mov.b32 	%r201, %r163;
	mov.b32 	%r202, %r163;
	mov.b32 	%r203, %r163;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r200, %r201, %r202, %r203 }, { %r164, %r165, %r166, %r167 }, { %r176, %r177 }, { %r200, %r201, %r202, %r203 };
	// end inline asm
	mov.b32 	%r204, %r163;
	mov.b32 	%r205, %r163;
	mov.b32 	%r206, %r163;
	mov.b32 	%r207, %r163;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r204, %r205, %r206, %r207 }, { %r164, %r165, %r166, %r167 }, { %r178, %r179 }, { %r204, %r205, %r206, %r207 };
	// end inline asm
	mov.b32 	%r208, %r163;
	mov.b32 	%r209, %r163;
	mov.b32 	%r210, %r163;
	mov.b32 	%r211, %r163;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r208, %r209, %r210, %r211 }, { %r170, %r171, %r172, %r173 }, { %r168, %r169 }, { %r208, %r209, %r210, %r211 };
	// end inline asm
	mov.b32 	%r214, %r163;
	mov.b32 	%r215, %r163;
	mov.b32 	%r216, %r163;
	mov.b32 	%r217, %r163;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r214, %r215, %r216, %r217 }, { %r170, %r171, %r172, %r173 }, { %r174, %r175 }, { %r214, %r215, %r216, %r217 };
	// end inline asm
	mov.b32 	%r224, %r163;
	mov.b32 	%r225, %r163;
	mov.b32 	%r226, %r163;
	mov.b32 	%r227, %r163;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r224, %r225, %r226, %r227 }, { %r170, %r171, %r172, %r173 }, { %r176, %r177 }, { %r224, %r225, %r226, %r227 };
	// end inline asm
	mov.b32 	%r230, %r163;
	mov.b32 	%r231, %r163;
	mov.b32 	%r232, %r163;
	mov.b32 	%r233, %r163;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r230, %r231, %r232, %r233 }, { %r170, %r171, %r172, %r173 }, { %r178, %r179 }, { %r230, %r231, %r232, %r233 };
	// end inline asm
	mov.b32 	%r236, %r163;
	mov.b32 	%r237, %r163;
	mov.b32 	%r238, %r163;
	mov.b32 	%r239, %r163;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r236, %r237, %r238, %r239 }, { %r180, %r181, %r182, %r183 }, { %r168, %r169 }, { %r236, %r237, %r238, %r239 };
	// end inline asm
	mov.b32 	%r240, %r163;
	mov.b32 	%r241, %r163;
	mov.b32 	%r242, %r163;
	mov.b32 	%r243, %r163;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r240, %r241, %r242, %r243 }, { %r180, %r181, %r182, %r183 }, { %r174, %r175 }, { %r240, %r241, %r242, %r243 };
	// end inline asm
	mov.b32 	%r248, %r163;
	mov.b32 	%r249, %r163;
	mov.b32 	%r250, %r163;
	mov.b32 	%r251, %r163;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r248, %r249, %r250, %r251 }, { %r180, %r181, %r182, %r183 }, { %r176, %r177 }, { %r248, %r249, %r250, %r251 };
	// end inline asm
	mov.b32 	%r252, %r163;
	mov.b32 	%r253, %r163;
	mov.b32 	%r254, %r163;
	mov.b32 	%r255, %r163;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r252, %r253, %r254, %r255 }, { %r180, %r181, %r182, %r183 }, { %r178, %r179 }, { %r252, %r253, %r254, %r255 };
	// end inline asm
	mov.b32 	%r256, %r163;
	mov.b32 	%r257, %r163;
	mov.b32 	%r258, %r163;
	mov.b32 	%r259, %r163;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r256, %r257, %r258, %r259 }, { %r184, %r185, %r186, %r187 }, { %r168, %r169 }, { %r256, %r257, %r258, %r259 };
	// end inline asm
	mov.b32 	%r260, %r163;
	mov.b32 	%r261, %r163;
	mov.b32 	%r262, %r163;
	mov.b32 	%r263, %r163;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r260, %r261, %r262, %r263 }, { %r184, %r185, %r186, %r187 }, { %r174, %r175 }, { %r260, %r261, %r262, %r263 };
	// end inline asm
	mov.b32 	%r268, %r163;
	mov.b32 	%r269, %r163;
	mov.b32 	%r270, %r163;
	mov.b32 	%r271, %r163;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r268, %r269, %r270, %r271 }, { %r184, %r185, %r186, %r187 }, { %r176, %r177 }, { %r268, %r269, %r270, %r271 };
	// end inline asm
	mov.b32 	%r275, %r163;
	mov.b32 	%r272, %r163;
	mov.b32 	%r273, %r163;
	mov.b32 	%r274, %r163;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r272, %r273, %r274, %r275 }, { %r184, %r185, %r186, %r187 }, { %r178, %r179 }, { %r272, %r273, %r274, %r275 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r188, %r189, %r190, %r191 }, { %r196, %r197, %r198, %r199 }, { %r212, %r213 }, { %r188, %r189, %r190, %r191 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r192, %r193, %r194, %r195 }, { %r196, %r197, %r198, %r199 }, { %r222, %r223 }, { %r192, %r193, %r194, %r195 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r200, %r201, %r202, %r203 }, { %r196, %r197, %r198, %r199 }, { %r228, %r229 }, { %r200, %r201, %r202, %r203 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r204, %r205, %r206, %r207 }, { %r196, %r197, %r198, %r199 }, { %r234, %r235 }, { %r204, %r205, %r206, %r207 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r208, %r209, %r210, %r211 }, { %r218, %r219, %r220, %r221 }, { %r212, %r213 }, { %r208, %r209, %r210, %r211 };
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
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r236, %r237, %r238, %r239 }, { %r244, %r245, %r246, %r247 }, { %r212, %r213 }, { %r236, %r237, %r238, %r239 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r240, %r241, %r242, %r243 }, { %r244, %r245, %r246, %r247 }, { %r222, %r223 }, { %r240, %r241, %r242, %r243 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r248, %r249, %r250, %r251 }, { %r244, %r245, %r246, %r247 }, { %r228, %r229 }, { %r248, %r249, %r250, %r251 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r252, %r253, %r254, %r255 }, { %r244, %r245, %r246, %r247 }, { %r234, %r235 }, { %r252, %r253, %r254, %r255 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r256, %r257, %r258, %r259 }, { %r264, %r265, %r266, %r267 }, { %r212, %r213 }, { %r256, %r257, %r258, %r259 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r260, %r261, %r262, %r263 }, { %r264, %r265, %r266, %r267 }, { %r222, %r223 }, { %r260, %r261, %r262, %r263 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r268, %r269, %r270, %r271 }, { %r264, %r265, %r266, %r267 }, { %r228, %r229 }, { %r268, %r269, %r270, %r271 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r272, %r273, %r274, %r275 }, { %r264, %r265, %r266, %r267 }, { %r234, %r235 }, { %r272, %r273, %r274, %r275 };
	// end inline asm
	.loc	1 832 75                        // sk04_fa_o_w4a8.py:832:75
	prmt.b32 	%r276, %r744, %r745, 0x7531U;
	prmt.b32 	%r277, %r746, %r747, 0x7531U;
	prmt.b32 	%r278, %r761, %r762, 0x7531U;
	prmt.b32 	%r279, %r763, %r764, 0x7531U;
	prmt.b32 	%r300, %r778, %r779, 0x7531U;
	prmt.b32 	%r301, %r780, %r781, 0x7531U;
	prmt.b32 	%r302, %r795, %r796, 0x7531U;
	prmt.b32 	%r303, %r797, %r798, 0x7531U;
	prmt.b32 	%r282, %r748, %r749, 0x7531U;
	prmt.b32 	%r283, %r750, %r751, 0x7531U;
	prmt.b32 	%r284, %r765, %r766, 0x7531U;
	prmt.b32 	%r285, %r767, %r768, 0x7531U;
	prmt.b32 	%r306, %r782, %r783, 0x7531U;
	prmt.b32 	%r307, %r784, %r785, 0x7531U;
	prmt.b32 	%r308, %r799, %r800, 0x7531U;
	prmt.b32 	%r309, %r801, %r802, 0x7531U;
	prmt.b32 	%r292, %r752, %r753, 0x7531U;
	prmt.b32 	%r293, %r754, %r755, 0x7531U;
	prmt.b32 	%r294, %r769, %r770, 0x7531U;
	prmt.b32 	%r295, %r771, %r772, 0x7531U;
	prmt.b32 	%r316, %r786, %r787, 0x7531U;
	prmt.b32 	%r317, %r788, %r789, 0x7531U;
	prmt.b32 	%r318, %r803, %r804, 0x7531U;
	prmt.b32 	%r319, %r805, %r806, 0x7531U;
	prmt.b32 	%r296, %r756, %r757, 0x7531U;
	prmt.b32 	%r297, %r758, %r759, 0x7531U;
	prmt.b32 	%r298, %r773, %r774, 0x7531U;
	prmt.b32 	%r299, %r775, %r776, 0x7531U;
	prmt.b32 	%r320, %r790, %r791, 0x7531U;
	prmt.b32 	%r321, %r792, %r793, 0x7531U;
	prmt.b32 	%r322, %r807, %r808, 0x7531U;
	prmt.b32 	%r323, %r809, %r810, 0x7531U;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r188, %r189, %r190, %r191 }, { %r276, %r277, %r278, %r279 }, { %r280, %r281 }, { %r188, %r189, %r190, %r191 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r192, %r193, %r194, %r195 }, { %r276, %r277, %r278, %r279 }, { %r286, %r287 }, { %r192, %r193, %r194, %r195 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r200, %r201, %r202, %r203 }, { %r276, %r277, %r278, %r279 }, { %r288, %r289 }, { %r200, %r201, %r202, %r203 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r204, %r205, %r206, %r207 }, { %r276, %r277, %r278, %r279 }, { %r290, %r291 }, { %r204, %r205, %r206, %r207 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r208, %r209, %r210, %r211 }, { %r282, %r283, %r284, %r285 }, { %r280, %r281 }, { %r208, %r209, %r210, %r211 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r214, %r215, %r216, %r217 }, { %r282, %r283, %r284, %r285 }, { %r286, %r287 }, { %r214, %r215, %r216, %r217 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r224, %r225, %r226, %r227 }, { %r282, %r283, %r284, %r285 }, { %r288, %r289 }, { %r224, %r225, %r226, %r227 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r230, %r231, %r232, %r233 }, { %r282, %r283, %r284, %r285 }, { %r290, %r291 }, { %r230, %r231, %r232, %r233 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r236, %r237, %r238, %r239 }, { %r292, %r293, %r294, %r295 }, { %r280, %r281 }, { %r236, %r237, %r238, %r239 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r240, %r241, %r242, %r243 }, { %r292, %r293, %r294, %r295 }, { %r286, %r287 }, { %r240, %r241, %r242, %r243 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r248, %r249, %r250, %r251 }, { %r292, %r293, %r294, %r295 }, { %r288, %r289 }, { %r248, %r249, %r250, %r251 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r252, %r253, %r254, %r255 }, { %r292, %r293, %r294, %r295 }, { %r290, %r291 }, { %r252, %r253, %r254, %r255 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r256, %r257, %r258, %r259 }, { %r296, %r297, %r298, %r299 }, { %r280, %r281 }, { %r256, %r257, %r258, %r259 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r260, %r261, %r262, %r263 }, { %r296, %r297, %r298, %r299 }, { %r286, %r287 }, { %r260, %r261, %r262, %r263 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r268, %r269, %r270, %r271 }, { %r296, %r297, %r298, %r299 }, { %r288, %r289 }, { %r268, %r269, %r270, %r271 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r272, %r273, %r274, %r275 }, { %r296, %r297, %r298, %r299 }, { %r290, %r291 }, { %r272, %r273, %r274, %r275 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r188, %r189, %r190, %r191 }, { %r300, %r301, %r302, %r303 }, { %r304, %r305 }, { %r188, %r189, %r190, %r191 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r192, %r193, %r194, %r195 }, { %r300, %r301, %r302, %r303 }, { %r310, %r311 }, { %r192, %r193, %r194, %r195 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r200, %r201, %r202, %r203 }, { %r300, %r301, %r302, %r303 }, { %r312, %r313 }, { %r200, %r201, %r202, %r203 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r204, %r205, %r206, %r207 }, { %r300, %r301, %r302, %r303 }, { %r314, %r315 }, { %r204, %r205, %r206, %r207 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r208, %r209, %r210, %r211 }, { %r306, %r307, %r308, %r309 }, { %r304, %r305 }, { %r208, %r209, %r210, %r211 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r214, %r215, %r216, %r217 }, { %r306, %r307, %r308, %r309 }, { %r310, %r311 }, { %r214, %r215, %r216, %r217 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r224, %r225, %r226, %r227 }, { %r306, %r307, %r308, %r309 }, { %r312, %r313 }, { %r224, %r225, %r226, %r227 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r230, %r231, %r232, %r233 }, { %r306, %r307, %r308, %r309 }, { %r314, %r315 }, { %r230, %r231, %r232, %r233 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r236, %r237, %r238, %r239 }, { %r316, %r317, %r318, %r319 }, { %r304, %r305 }, { %r236, %r237, %r238, %r239 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r240, %r241, %r242, %r243 }, { %r316, %r317, %r318, %r319 }, { %r310, %r311 }, { %r240, %r241, %r242, %r243 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r248, %r249, %r250, %r251 }, { %r316, %r317, %r318, %r319 }, { %r312, %r313 }, { %r248, %r249, %r250, %r251 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r252, %r253, %r254, %r255 }, { %r316, %r317, %r318, %r319 }, { %r314, %r315 }, { %r252, %r253, %r254, %r255 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r256, %r257, %r258, %r259 }, { %r320, %r321, %r322, %r323 }, { %r304, %r305 }, { %r256, %r257, %r258, %r259 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r260, %r261, %r262, %r263 }, { %r320, %r321, %r322, %r323 }, { %r310, %r311 }, { %r260, %r261, %r262, %r263 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r268, %r269, %r270, %r271 }, { %r320, %r321, %r322, %r323 }, { %r312, %r313 }, { %r268, %r269, %r270, %r271 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r272, %r273, %r274, %r275 }, { %r320, %r321, %r322, %r323 }, { %r314, %r315 }, { %r272, %r273, %r274, %r275 };
	// end inline asm
	.loc	1 833 59                        // sk04_fa_o_w4a8.py:833:59
	mad.wide.s32 	%rd86, %r1244, 4, %rd16;
	.loc	1 833 27                        // sk04_fa_o_w4a8.py:833:27
	// begin inline asm
	mov.u32 %r324, 0x0;
	ld.global.b32 { %r324 }, [ %rd86 + 0 ];
	// end inline asm
	.loc	1 834 40                        // sk04_fa_o_w4a8.py:834:40
	// begin inline asm
	st.shared.b32 [ %r325 + 0 ], %r324;
	// end inline asm
	bar.sync 	0;
	.loc	1 834 26                        // sk04_fa_o_w4a8.py:834:26
	cvt.rn.f32.s32 	%r811, %r188;
	cvt.rn.f32.s32 	%r812, %r189;
	cvt.rn.f32.s32 	%r813, %r190;
	cvt.rn.f32.s32 	%r814, %r191;
	cvt.rn.f32.s32 	%r815, %r192;
	cvt.rn.f32.s32 	%r816, %r193;
	cvt.rn.f32.s32 	%r817, %r194;
	cvt.rn.f32.s32 	%r818, %r195;
	cvt.rn.f32.s32 	%r819, %r200;
	cvt.rn.f32.s32 	%r820, %r201;
	cvt.rn.f32.s32 	%r821, %r202;
	cvt.rn.f32.s32 	%r822, %r203;
	cvt.rn.f32.s32 	%r823, %r204;
	cvt.rn.f32.s32 	%r824, %r205;
	cvt.rn.f32.s32 	%r825, %r206;
	cvt.rn.f32.s32 	%r826, %r207;
	cvt.rn.f32.s32 	%r827, %r208;
	cvt.rn.f32.s32 	%r828, %r209;
	cvt.rn.f32.s32 	%r829, %r210;
	cvt.rn.f32.s32 	%r830, %r211;
	cvt.rn.f32.s32 	%r831, %r214;
	cvt.rn.f32.s32 	%r832, %r215;
	cvt.rn.f32.s32 	%r833, %r216;
	cvt.rn.f32.s32 	%r834, %r217;
	cvt.rn.f32.s32 	%r835, %r224;
	cvt.rn.f32.s32 	%r836, %r225;
	cvt.rn.f32.s32 	%r837, %r226;
	cvt.rn.f32.s32 	%r838, %r227;
	cvt.rn.f32.s32 	%r839, %r230;
	cvt.rn.f32.s32 	%r840, %r231;
	cvt.rn.f32.s32 	%r841, %r232;
	cvt.rn.f32.s32 	%r842, %r233;
	cvt.rn.f32.s32 	%r843, %r236;
	cvt.rn.f32.s32 	%r844, %r237;
	cvt.rn.f32.s32 	%r845, %r238;
	cvt.rn.f32.s32 	%r846, %r239;
	cvt.rn.f32.s32 	%r847, %r240;
	cvt.rn.f32.s32 	%r848, %r241;
	cvt.rn.f32.s32 	%r849, %r242;
	cvt.rn.f32.s32 	%r850, %r243;
	cvt.rn.f32.s32 	%r851, %r248;
	cvt.rn.f32.s32 	%r852, %r249;
	cvt.rn.f32.s32 	%r853, %r250;
	cvt.rn.f32.s32 	%r854, %r251;
	cvt.rn.f32.s32 	%r855, %r252;
	cvt.rn.f32.s32 	%r856, %r253;
	cvt.rn.f32.s32 	%r857, %r254;
	cvt.rn.f32.s32 	%r858, %r255;
	cvt.rn.f32.s32 	%r859, %r256;
	cvt.rn.f32.s32 	%r860, %r257;
	cvt.rn.f32.s32 	%r861, %r258;
	cvt.rn.f32.s32 	%r862, %r259;
	cvt.rn.f32.s32 	%r863, %r260;
	cvt.rn.f32.s32 	%r864, %r261;
	cvt.rn.f32.s32 	%r865, %r262;
	cvt.rn.f32.s32 	%r866, %r263;
	cvt.rn.f32.s32 	%r867, %r268;
	cvt.rn.f32.s32 	%r868, %r269;
	cvt.rn.f32.s32 	%r869, %r270;
	cvt.rn.f32.s32 	%r870, %r271;
	cvt.rn.f32.s32 	%r871, %r272;
	cvt.rn.f32.s32 	%r872, %r273;
	cvt.rn.f32.s32 	%r873, %r274;
	cvt.rn.f32.s32 	%r874, %r275;
	.loc	1 834 40                        // sk04_fa_o_w4a8.py:834:40
	ld.shared.v2.b32 	{%r875, %r876}, [%r46+49152];
	ld.shared.v2.b32 	{%r877, %r878}, [%r46+49280];
	ld.shared.v2.b32 	{%r879, %r880}, [%r46+49408];
	ld.shared.v2.b32 	{%r881, %r882}, [%r46+49536];
	.loc	1 834 15                        // sk04_fa_o_w4a8.py:834:15
	fma.rn.f32 	%r1313, %r882, %r874, %r1313;
	fma.rn.f32 	%r1312, %r881, %r873, %r1312;
	fma.rn.f32 	%r1311, %r882, %r872, %r1311;
	fma.rn.f32 	%r1310, %r881, %r871, %r1310;
	fma.rn.f32 	%r1309, %r880, %r870, %r1309;
	fma.rn.f32 	%r1308, %r879, %r869, %r1308;
	fma.rn.f32 	%r1307, %r880, %r868, %r1307;
	fma.rn.f32 	%r1306, %r879, %r867, %r1306;
	fma.rn.f32 	%r1305, %r878, %r866, %r1305;
	fma.rn.f32 	%r1304, %r877, %r865, %r1304;
	fma.rn.f32 	%r1303, %r878, %r864, %r1303;
	fma.rn.f32 	%r1302, %r877, %r863, %r1302;
	fma.rn.f32 	%r1301, %r876, %r862, %r1301;
	fma.rn.f32 	%r1300, %r875, %r861, %r1300;
	fma.rn.f32 	%r1299, %r876, %r860, %r1299;
	fma.rn.f32 	%r1298, %r875, %r859, %r1298;
	fma.rn.f32 	%r1297, %r882, %r858, %r1297;
	fma.rn.f32 	%r1296, %r881, %r857, %r1296;
	fma.rn.f32 	%r1295, %r882, %r856, %r1295;
	fma.rn.f32 	%r1294, %r881, %r855, %r1294;
	fma.rn.f32 	%r1293, %r880, %r854, %r1293;
	fma.rn.f32 	%r1292, %r879, %r853, %r1292;
	fma.rn.f32 	%r1291, %r880, %r852, %r1291;
	fma.rn.f32 	%r1290, %r879, %r851, %r1290;
	fma.rn.f32 	%r1289, %r878, %r850, %r1289;
	fma.rn.f32 	%r1288, %r877, %r849, %r1288;
	fma.rn.f32 	%r1287, %r878, %r848, %r1287;
	fma.rn.f32 	%r1286, %r877, %r847, %r1286;
	fma.rn.f32 	%r1285, %r876, %r846, %r1285;
	fma.rn.f32 	%r1284, %r875, %r845, %r1284;
	fma.rn.f32 	%r1283, %r876, %r844, %r1283;
	fma.rn.f32 	%r1282, %r875, %r843, %r1282;
	fma.rn.f32 	%r1281, %r882, %r842, %r1281;
	fma.rn.f32 	%r1280, %r881, %r841, %r1280;
	fma.rn.f32 	%r1279, %r882, %r840, %r1279;
	fma.rn.f32 	%r1278, %r881, %r839, %r1278;
	fma.rn.f32 	%r1277, %r880, %r838, %r1277;
	fma.rn.f32 	%r1276, %r879, %r837, %r1276;
	fma.rn.f32 	%r1275, %r880, %r836, %r1275;
	fma.rn.f32 	%r1274, %r879, %r835, %r1274;
	fma.rn.f32 	%r1273, %r878, %r834, %r1273;
	fma.rn.f32 	%r1272, %r877, %r833, %r1272;
	fma.rn.f32 	%r1271, %r878, %r832, %r1271;
	fma.rn.f32 	%r1270, %r877, %r831, %r1270;
	fma.rn.f32 	%r1269, %r876, %r830, %r1269;
	fma.rn.f32 	%r1268, %r875, %r829, %r1268;
	fma.rn.f32 	%r1267, %r876, %r828, %r1267;
	fma.rn.f32 	%r1266, %r875, %r827, %r1266;
	fma.rn.f32 	%r1265, %r882, %r826, %r1265;
	fma.rn.f32 	%r1264, %r881, %r825, %r1264;
	fma.rn.f32 	%r1263, %r882, %r824, %r1263;
	fma.rn.f32 	%r1262, %r881, %r823, %r1262;
	fma.rn.f32 	%r1261, %r880, %r822, %r1261;
	fma.rn.f32 	%r1260, %r879, %r821, %r1260;
	fma.rn.f32 	%r1259, %r880, %r820, %r1259;
	fma.rn.f32 	%r1258, %r879, %r819, %r1258;
	fma.rn.f32 	%r1257, %r878, %r818, %r1257;
	fma.rn.f32 	%r1256, %r877, %r817, %r1256;
	fma.rn.f32 	%r1255, %r878, %r816, %r1255;
	fma.rn.f32 	%r1254, %r877, %r815, %r1254;
	fma.rn.f32 	%r1253, %r876, %r814, %r1253;
	fma.rn.f32 	%r1252, %r875, %r813, %r1252;
	fma.rn.f32 	%r1251, %r876, %r812, %r1251;
	fma.rn.f32 	%r1250, %r875, %r811, %r1250;
	.loc	1 835 18                        // sk04_fa_o_w4a8.py:835:18
	add.s64 	%rd95, %rd22, %rd139;
	add.s64 	%rd96, %rd21, %rd139;
	add.s64 	%rd97, %rd20, %rd139;
	.loc	1 836 18                        // sk04_fa_o_w4a8.py:836:18
	add.s64 	%rd98, %rd19, %rd139;
	add.s64 	%rd87, %rd138, %rd6;
	add.s64 	%rd91, %rd138, %rd7;
	add.s64 	%rd88, %rd138, %rd8;
	add.s64 	%rd92, %rd138, %rd9;
	add.s64 	%rd89, %rd138, %rd10;
	add.s64 	%rd93, %rd138, %rd11;
	add.s64 	%rd90, %rd138, %rd12;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	add.s64 	%rd94, %rd138, %rd13;
	add.s32 	%r883, %r1246, 1;
	setp.gt.s32 	%p6, %r883, 2;
	selp.b32 	%r1246, 0, %r883, %p6;
	.loc	1 827 25                        // sk04_fa_o_w4a8.py:827:25
	shl.b32 	%r884, %r1246, 13;
	add.s32 	%r885, %r146, %r884;
	add.s32 	%r326, %r53, %r884;
	selp.b32 	%r327, 8, 0, %p4;
	// begin inline asm
	cp.async.ca.shared.global [ %r326 + 0 ], [ %rd87 + 0 ], 0x8, %r327;
	// end inline asm
	add.s32 	%r328, %r326, 2048;
	// begin inline asm
	cp.async.ca.shared.global [ %r328 + 0 ], [ %rd88 + 0 ], 0x8, %r327;
	// end inline asm
	add.s32 	%r329, %r326, 4096;
	// begin inline asm
	cp.async.ca.shared.global [ %r329 + 0 ], [ %rd89 + 0 ], 0x8, %r327;
	// end inline asm
	add.s32 	%r330, %r326, 6144;
	// begin inline asm
	cp.async.ca.shared.global [ %r330 + 0 ], [ %rd90 + 0 ], 0x8, %r327;
	// end inline asm
	add.s32 	%r886, %r885, %r23;
	add.s32 	%r331, %r886, 1024;
	// begin inline asm
	cp.async.ca.shared.global [ %r331 + 0 ], [ %rd91 + 0 ], 0x8, %r327;
	// end inline asm
	add.s32 	%r332, %r886, 3072;
	// begin inline asm
	cp.async.ca.shared.global [ %r332 + 0 ], [ %rd92 + 0 ], 0x8, %r327;
	// end inline asm
	add.s32 	%r333, %r886, 5120;
	// begin inline asm
	cp.async.ca.shared.global [ %r333 + 0 ], [ %rd93 + 0 ], 0x8, %r327;
	// end inline asm
	add.s32 	%r334, %r886, 7168;
	// begin inline asm
	cp.async.ca.shared.global [ %r334 + 0 ], [ %rd94 + 0 ], 0x8, %r327;
	// end inline asm
	cp.async.commit_group;
	.loc	1 831 52                        // sk04_fa_o_w4a8.py:831:52
	add.s32 	%r887, %r25, %r884;
	add.s32 	%r335, %r887, 24576;
	selp.b32 	%r336, 16, 0, %p4;
	// begin inline asm
	cp.async.cg.shared.global [ %r335 + 0 ], [ %rd95 + 0 ], 0x10, %r336;
	// end inline asm
	add.s32 	%r337, %r887, 26624;
	// begin inline asm
	cp.async.cg.shared.global [ %r337 + 0 ], [ %rd96 + 0 ], 0x10, %r336;
	// end inline asm
	add.s32 	%r338, %r887, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r338 + 0 ], [ %rd97 + 0 ], 0x10, %r336;
	// end inline asm
	add.s32 	%r339, %r887, 30720;
	// begin inline asm
	cp.async.cg.shared.global [ %r339 + 0 ], [ %rd98 + 0 ], 0x10, %r336;
	// end inline asm
	cp.async.commit_group;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	add.s64 	%rd140, %rd140, 1;
	add.s64 	%rd139, %rd139, 128;
	add.s32 	%r1244, %r1244, %r52;
	add.s64 	%rd138, %rd138, %rd15;
	setp.ne.b64 	%p7, %rd18, %rd139;
	@%p7 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_4;
$L__BB0_1:                              // %.._crit_edge_crit_edge
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	shr.u32 	%r1249, %r2, 2;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	and.b32 	%r1248, %r2, 3;
	and.b32 	%r1247, %r2, 96;
	mov.b32 	%r1250, 0f00000000;
	mov.b32 	%r1251, %r1250;
	mov.b32 	%r1252, %r1250;
	mov.b32 	%r1253, %r1250;
	mov.b32 	%r1254, %r1250;
	mov.b32 	%r1255, %r1250;
	mov.b32 	%r1256, %r1250;
	mov.b32 	%r1257, %r1250;
	mov.b32 	%r1258, %r1250;
	mov.b32 	%r1259, %r1250;
	mov.b32 	%r1260, %r1250;
	mov.b32 	%r1261, %r1250;
	mov.b32 	%r1262, %r1250;
	mov.b32 	%r1263, %r1250;
	mov.b32 	%r1264, %r1250;
	mov.b32 	%r1265, %r1250;
	mov.b32 	%r1266, %r1250;
	mov.b32 	%r1267, %r1250;
	mov.b32 	%r1268, %r1250;
	mov.b32 	%r1269, %r1250;
	mov.b32 	%r1270, %r1250;
	mov.b32 	%r1271, %r1250;
	mov.b32 	%r1272, %r1250;
	mov.b32 	%r1273, %r1250;
	mov.b32 	%r1274, %r1250;
	mov.b32 	%r1275, %r1250;
	mov.b32 	%r1276, %r1250;
	mov.b32 	%r1277, %r1250;
	mov.b32 	%r1278, %r1250;
	mov.b32 	%r1279, %r1250;
	mov.b32 	%r1280, %r1250;
	mov.b32 	%r1281, %r1250;
	mov.b32 	%r1282, %r1250;
	mov.b32 	%r1283, %r1250;
	mov.b32 	%r1284, %r1250;
	mov.b32 	%r1285, %r1250;
	mov.b32 	%r1286, %r1250;
	mov.b32 	%r1287, %r1250;
	mov.b32 	%r1288, %r1250;
	mov.b32 	%r1289, %r1250;
	mov.b32 	%r1290, %r1250;
	mov.b32 	%r1291, %r1250;
	mov.b32 	%r1292, %r1250;
	mov.b32 	%r1293, %r1250;
	mov.b32 	%r1294, %r1250;
	mov.b32 	%r1295, %r1250;
	mov.b32 	%r1296, %r1250;
	mov.b32 	%r1297, %r1250;
	mov.b32 	%r1298, %r1250;
	mov.b32 	%r1299, %r1250;
	mov.b32 	%r1300, %r1250;
	mov.b32 	%r1301, %r1250;
	mov.b32 	%r1302, %r1250;
	mov.b32 	%r1303, %r1250;
	mov.b32 	%r1304, %r1250;
	mov.b32 	%r1305, %r1250;
	mov.b32 	%r1306, %r1250;
	mov.b32 	%r1307, %r1250;
	mov.b32 	%r1308, %r1250;
	mov.b32 	%r1309, %r1250;
	mov.b32 	%r1310, %r1250;
	mov.b32 	%r1311, %r1250;
	mov.b32 	%r1312, %r1250;
	mov.b32 	%r1313, %r1250;
$L__BB0_4:                              // %._crit_edge
	.loc	1 0 15                          // sk04_fa_o_w4a8.py:0:15
	cvt.u32.u64 	%r998, %rd5;
	.loc	1 817 45                        // sk04_fa_o_w4a8.py:817:45
	or.b32 	%r999, %r17, %r998;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r1000, %r999, 8;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r1001, %r1000, %r48;
	rem.s32 	%r1002, %r999, %r48;
	.loc	1 816 45                        // sk04_fa_o_w4a8.py:816:45
	shr.u32 	%r1003, %r12, 2;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1004, %r1003, %r1;
	or.b32 	%r1005, %r1004, 56;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1006, %r1005, %r47;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1007, %r1004, 48;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1008, %r1007, %r47;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1009, %r1004, 40;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1010, %r1009, %r47;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1011, %r1004, 32;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1012, %r1011, %r47;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1013, %r1004, 24;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1014, %r1013, %r47;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1015, %r1004, 16;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1016, %r1015, %r47;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1017, %r1004, 8;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1018, %r1017, %r47;
	rem.s32 	%r1019, %r1004, %r47;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1020, %r1, %r10;
	or.b32 	%r1021, %r1, %r9;
	or.b32 	%r1022, %r1, %r8;
	or.b32 	%r1023, %r1, %r7;
	or.b32 	%r1024, %r1, %r6;
	or.b32 	%r1025, %r1, %r5;
	or.b32 	%r1026, %r1, %r4;
	or.b32 	%r1027, %r1, %r3;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 838 38                        // sk04_fa_o_w4a8.py:838:38
	mad.wide.s32 	%rd99, %r1019, 4, %rd27;
	mad.wide.s32 	%rd100, %r1018, 4, %rd27;
	mad.wide.s32 	%rd101, %r1016, 4, %rd27;
	mad.wide.s32 	%rd102, %r1014, 4, %rd27;
	mad.wide.s32 	%rd103, %r1012, 4, %rd27;
	mad.wide.s32 	%rd104, %r1010, 4, %rd27;
	mad.wide.s32 	%rd105, %r1008, 4, %rd27;
	mad.wide.s32 	%rd106, %r1006, 4, %rd27;
	.loc	1 838 24                        // sk04_fa_o_w4a8.py:838:24
	// begin inline asm
	mov.u32 %r888, 0x0;
	ld.global.b32 { %r888 }, [ %rd99 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r889, 0x0;
	ld.global.b32 { %r889 }, [ %rd100 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r890, 0x0;
	ld.global.b32 { %r890 }, [ %rd101 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r891, 0x0;
	ld.global.b32 { %r891 }, [ %rd102 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r892, 0x0;
	ld.global.b32 { %r892 }, [ %rd103 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r893, 0x0;
	ld.global.b32 { %r893 }, [ %rd104 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r894, 0x0;
	ld.global.b32 { %r894 }, [ %rd105 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r895, 0x0;
	ld.global.b32 { %r895 }, [ %rd106 + 0 ];
	// end inline asm
	.loc	1 839 49                        // sk04_fa_o_w4a8.py:839:49
	mul.lo.s32 	%r1028, %r13, %r51;
	mul.lo.s32 	%r1029, %r14, %r51;
	mul.lo.s32 	%r1030, %r15, %r51;
	mul.lo.s32 	%r1031, %r16, %r51;
	.loc	1 839 31                        // sk04_fa_o_w4a8.py:839:31
	mad.wide.s32 	%rd123, %r1028, 2, %rd26;
	mad.wide.s32 	%rd124, %r1029, 2, %rd26;
	mad.wide.s32 	%rd125, %r1030, 2, %rd26;
	mad.wide.s32 	%rd126, %r1031, 2, %rd26;
	.loc	1 839 64                        // sk04_fa_o_w4a8.py:839:64
	mul.wide.s32 	%rd127, %r1002, 2;
	add.s64 	%rd107, %rd123, %rd127;
	mul.wide.s32 	%rd128, %r1001, 2;
	add.s64 	%rd108, %rd123, %rd128;
	add.s64 	%rd109, %rd124, %rd127;
	add.s64 	%rd110, %rd124, %rd128;
	add.s64 	%rd111, %rd125, %rd127;
	add.s64 	%rd112, %rd125, %rd128;
	add.s64 	%rd113, %rd126, %rd127;
	add.s64 	%rd114, %rd126, %rd128;
	.loc	1 839 19                        // sk04_fa_o_w4a8.py:839:19
	// begin inline asm
	mov.u32 %r897, 0x0;
	mov.u32 %r898, 0x0;
	mov.u32 %r899, 0x0;
	mov.u32 %r900, 0x0;
	ld.global.v4.b32 { %r897, %r898, %r899, %r900 }, [ %rd107 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r902, 0x0;
	mov.u32 %r903, 0x0;
	mov.u32 %r904, 0x0;
	mov.u32 %r905, 0x0;
	ld.global.v4.b32 { %r902, %r903, %r904, %r905 }, [ %rd108 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r906, 0x0;
	mov.u32 %r907, 0x0;
	mov.u32 %r908, 0x0;
	mov.u32 %r909, 0x0;
	ld.global.v4.b32 { %r906, %r907, %r908, %r909 }, [ %rd109 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r910, 0x0;
	mov.u32 %r911, 0x0;
	mov.u32 %r912, 0x0;
	mov.u32 %r913, 0x0;
	ld.global.v4.b32 { %r910, %r911, %r912, %r913 }, [ %rd110 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r914, 0x0;
	mov.u32 %r915, 0x0;
	mov.u32 %r916, 0x0;
	mov.u32 %r917, 0x0;
	ld.global.v4.b32 { %r914, %r915, %r916, %r917 }, [ %rd111 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r918, 0x0;
	mov.u32 %r919, 0x0;
	mov.u32 %r920, 0x0;
	mov.u32 %r921, 0x0;
	ld.global.v4.b32 { %r918, %r919, %r920, %r921 }, [ %rd112 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r922, 0x0;
	mov.u32 %r923, 0x0;
	mov.u32 %r924, 0x0;
	mov.u32 %r925, 0x0;
	ld.global.v4.b32 { %r922, %r923, %r924, %r925 }, [ %rd113 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r926, 0x0;
	mov.u32 %r927, 0x0;
	mov.u32 %r928, 0x0;
	mov.u32 %r929, 0x0;
	ld.global.v4.b32 { %r926, %r927, %r928, %r929 }, [ %rd114 + 0 ];
	// end inline asm
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	shl.b32 	%r1032, %r2, 6;
	and.b32 	%r1033, %r1032, 3584;
	shl.b32 	%r1034, %r11, 1;
	or.b32 	%r1035, %r1033, %r998;
	xor.b32 	%r1036, %r1035, %r1034;
	add.s32 	%r896, %r146, %r1036;
	// begin inline asm
	st.shared.v4.b32 [ %r896 + 0 ], { %r897, %r898, %r899, %r900 };
	// end inline asm
	add.s32 	%r901, %r896, 256;
	// begin inline asm
	st.shared.v4.b32 [ %r901 + 0 ], { %r902, %r903, %r904, %r905 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r1037, %r20, 9;
	shl.b32 	%r1038, %r18, 4;
	bfe.s32 	%r1039, %r2, 4, 1;
	and.b32 	%r1040, %r2, 16;
	shl.b32 	%r1041, %r1040, 1;
	shl.b32 	%r1042, %r2, 3;
	and.b32 	%r1043, %r1042, 256;
	and.b32 	%r1044, %r1249, 16;
	or.b32 	%r1045, %r1037, %r1043;
	xor.b32 	%r1046, %r1038, %r1041;
	xor.b32 	%r1047, %r1046, %r1044;
	or.b32 	%r1048, %r1047, %r1045;
	add.s32 	%r1049, %r146, %r1048;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1050, %r1051, %r1052, %r1053}, [%r1049];
	xor.b32 	%r1054, %r1048, 64;
	add.s32 	%r1055, %r146, %r1054;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1056, %r1057, %r1058, %r1059}, [%r1055];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r896 + 0 ], { %r906, %r907, %r908, %r909 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r901 + 0 ], { %r910, %r911, %r912, %r913 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1060, %r1061, %r1062, %r1063}, [%r1049];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1064, %r1065, %r1066, %r1067}, [%r1055];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r896 + 0 ], { %r914, %r915, %r916, %r917 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r901 + 0 ], { %r918, %r919, %r920, %r921 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1068, %r1069, %r1070, %r1071}, [%r1049];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1072, %r1073, %r1074, %r1075}, [%r1055];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r896 + 0 ], { %r922, %r923, %r924, %r925 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r901 + 0 ], { %r926, %r927, %r928, %r929 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1076, %r1077, %r1078, %r1079}, [%r1049];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1080, %r1081, %r1082, %r1083}, [%r1055];
	.loc	1 846 31                        // sk04_fa_o_w4a8.py:846:31
	setp.lt.s32 	%p16, %r1027, %r47;
	setp.lt.s32 	%p17, %r1026, %r47;
	setp.lt.s32 	%p18, %r1025, %r47;
	setp.lt.s32 	%p19, %r1024, %r47;
	setp.lt.s32 	%p20, %r1023, %r47;
	setp.lt.s32 	%p21, %r1022, %r47;
	setp.lt.s32 	%p22, %r1021, %r47;
	setp.lt.s32 	%p23, %r1020, %r47;
	.loc	1 846 54                        // sk04_fa_o_w4a8.py:846:54
	setp.lt.s32 	%p24, %r22, %r48;
	.loc	1 846 37                        // sk04_fa_o_w4a8.py:846:37
	and.pred 	%p8, %p16, %p24;
	and.pred 	%p9, %p17, %p24;
	and.pred 	%p10, %p18, %p24;
	and.pred 	%p11, %p19, %p24;
	and.pred 	%p12, %p20, %p24;
	and.pred 	%p13, %p21, %p24;
	and.pred 	%p14, %p22, %p24;
	and.pred 	%p15, %p23, %p24;
	.loc	1 844 35                        // sk04_fa_o_w4a8.py:844:35
	mul.lo.s32 	%r1084, %r1027, %r50;
	mul.lo.s32 	%r1085, %r1026, %r50;
	mul.lo.s32 	%r1086, %r1025, %r50;
	mul.lo.s32 	%r1087, %r1024, %r50;
	mul.lo.s32 	%r1088, %r1023, %r50;
	mul.lo.s32 	%r1089, %r1022, %r50;
	mul.lo.s32 	%r1090, %r1021, %r50;
	mul.lo.s32 	%r1091, %r1020, %r50;
	.loc	1 844 18                        // sk04_fa_o_w4a8.py:844:18
	mad.wide.s32 	%rd129, %r1084, 2, %rd25;
	mad.wide.s32 	%rd130, %r1085, 2, %rd25;
	mad.wide.s32 	%rd131, %r1086, 2, %rd25;
	mad.wide.s32 	%rd132, %r1087, 2, %rd25;
	mad.wide.s32 	%rd133, %r1088, 2, %rd25;
	mad.wide.s32 	%rd134, %r1089, 2, %rd25;
	mad.wide.s32 	%rd135, %r1090, 2, %rd25;
	mad.wide.s32 	%rd136, %r1091, 2, %rd25;
	.loc	1 844 50                        // sk04_fa_o_w4a8.py:844:50
	mul.wide.s32 	%rd137, %r22, 2;
	add.s64 	%rd115, %rd129, %rd137;
	add.s64 	%rd116, %rd130, %rd137;
	add.s64 	%rd117, %rd131, %rd137;
	add.s64 	%rd118, %rd132, %rd137;
	add.s64 	%rd119, %rd133, %rd137;
	add.s64 	%rd120, %rd134, %rd137;
	add.s64 	%rd121, %rd135, %rd137;
	add.s64 	%rd122, %rd136, %rd137;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs321, %rs322}, %r1050;
	cvt.f32.bf16 	%r1092, %rs322;
	cvt.f32.bf16 	%r1093, %rs321;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1094, %r1250, %r888, %r1093;
	fma.rn.f32 	%r1095, %r1251, %r888, %r1092;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r931, %r1095, %r1094;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs323, %rs324}, %r1051;
	cvt.f32.bf16 	%r1096, %rs324;
	cvt.f32.bf16 	%r1097, %rs323;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1098, %r1252, %r889, %r1097;
	fma.rn.f32 	%r1099, %r1253, %r889, %r1096;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r932, %r1099, %r1098;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs325, %rs326}, %r1052;
	cvt.f32.bf16 	%r1100, %rs326;
	cvt.f32.bf16 	%r1101, %rs325;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1102, %r1254, %r888, %r1101;
	fma.rn.f32 	%r1103, %r1255, %r888, %r1100;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r941, %r1103, %r1102;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs327, %rs328}, %r1053;
	cvt.f32.bf16 	%r1104, %rs328;
	cvt.f32.bf16 	%r1105, %rs327;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1106, %r1256, %r889, %r1105;
	fma.rn.f32 	%r1107, %r1257, %r889, %r1104;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r942, %r1107, %r1106;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs329, %rs330}, %r1056;
	cvt.f32.bf16 	%r1108, %rs330;
	cvt.f32.bf16 	%r1109, %rs329;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1110, %r1258, %r888, %r1109;
	fma.rn.f32 	%r1111, %r1259, %r888, %r1108;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r936, %r1111, %r1110;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs331, %rs332}, %r1057;
	cvt.f32.bf16 	%r1112, %rs332;
	cvt.f32.bf16 	%r1113, %rs331;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1114, %r1260, %r889, %r1113;
	fma.rn.f32 	%r1115, %r1261, %r889, %r1112;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r937, %r1115, %r1114;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs333, %rs334}, %r1058;
	cvt.f32.bf16 	%r1116, %rs334;
	cvt.f32.bf16 	%r1117, %rs333;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1118, %r1262, %r888, %r1117;
	fma.rn.f32 	%r1119, %r1263, %r888, %r1116;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r946, %r1119, %r1118;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs335, %rs336}, %r1059;
	cvt.f32.bf16 	%r1120, %rs336;
	cvt.f32.bf16 	%r1121, %rs335;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1122, %r1264, %r889, %r1121;
	fma.rn.f32 	%r1123, %r1265, %r889, %r1120;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r947, %r1123, %r1122;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs337, %rs338}, %r1060;
	cvt.f32.bf16 	%r1124, %rs338;
	cvt.f32.bf16 	%r1125, %rs337;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1126, %r1266, %r890, %r1125;
	fma.rn.f32 	%r1127, %r1267, %r890, %r1124;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r933, %r1127, %r1126;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs339, %rs340}, %r1061;
	cvt.f32.bf16 	%r1128, %rs340;
	cvt.f32.bf16 	%r1129, %rs339;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1130, %r1268, %r891, %r1129;
	fma.rn.f32 	%r1131, %r1269, %r891, %r1128;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r934, %r1131, %r1130;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs341, %rs342}, %r1062;
	cvt.f32.bf16 	%r1132, %rs342;
	cvt.f32.bf16 	%r1133, %rs341;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1134, %r1270, %r890, %r1133;
	fma.rn.f32 	%r1135, %r1271, %r890, %r1132;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r943, %r1135, %r1134;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs343, %rs344}, %r1063;
	cvt.f32.bf16 	%r1136, %rs344;
	cvt.f32.bf16 	%r1137, %rs343;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1138, %r1272, %r891, %r1137;
	fma.rn.f32 	%r1139, %r1273, %r891, %r1136;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r944, %r1139, %r1138;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs345, %rs346}, %r1064;
	cvt.f32.bf16 	%r1140, %rs346;
	cvt.f32.bf16 	%r1141, %rs345;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1142, %r1274, %r890, %r1141;
	fma.rn.f32 	%r1143, %r1275, %r890, %r1140;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r938, %r1143, %r1142;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs347, %rs348}, %r1065;
	cvt.f32.bf16 	%r1144, %rs348;
	cvt.f32.bf16 	%r1145, %rs347;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1146, %r1276, %r891, %r1145;
	fma.rn.f32 	%r1147, %r1277, %r891, %r1144;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r939, %r1147, %r1146;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs349, %rs350}, %r1066;
	cvt.f32.bf16 	%r1148, %rs350;
	cvt.f32.bf16 	%r1149, %rs349;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1150, %r1278, %r890, %r1149;
	fma.rn.f32 	%r1151, %r1279, %r890, %r1148;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r948, %r1151, %r1150;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs351, %rs352}, %r1067;
	cvt.f32.bf16 	%r1152, %rs352;
	cvt.f32.bf16 	%r1153, %rs351;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1154, %r1280, %r891, %r1153;
	fma.rn.f32 	%r1155, %r1281, %r891, %r1152;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r949, %r1155, %r1154;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs353, %rs354}, %r1068;
	cvt.f32.bf16 	%r1156, %rs354;
	cvt.f32.bf16 	%r1157, %rs353;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1158, %r1282, %r892, %r1157;
	fma.rn.f32 	%r1159, %r1283, %r892, %r1156;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r950, %r1159, %r1158;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs355, %rs356}, %r1069;
	cvt.f32.bf16 	%r1160, %rs356;
	cvt.f32.bf16 	%r1161, %rs355;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1162, %r1284, %r893, %r1161;
	fma.rn.f32 	%r1163, %r1285, %r893, %r1160;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r951, %r1163, %r1162;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs357, %rs358}, %r1070;
	cvt.f32.bf16 	%r1164, %rs358;
	cvt.f32.bf16 	%r1165, %rs357;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1166, %r1286, %r892, %r1165;
	fma.rn.f32 	%r1167, %r1287, %r892, %r1164;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r958, %r1167, %r1166;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs359, %rs360}, %r1071;
	cvt.f32.bf16 	%r1168, %rs360;
	cvt.f32.bf16 	%r1169, %rs359;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1170, %r1288, %r893, %r1169;
	fma.rn.f32 	%r1171, %r1289, %r893, %r1168;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r959, %r1171, %r1170;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs361, %rs362}, %r1072;
	cvt.f32.bf16 	%r1172, %rs362;
	cvt.f32.bf16 	%r1173, %rs361;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1174, %r1290, %r892, %r1173;
	fma.rn.f32 	%r1175, %r1291, %r892, %r1172;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r954, %r1175, %r1174;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs363, %rs364}, %r1073;
	cvt.f32.bf16 	%r1176, %rs364;
	cvt.f32.bf16 	%r1177, %rs363;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1178, %r1292, %r893, %r1177;
	fma.rn.f32 	%r1179, %r1293, %r893, %r1176;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r955, %r1179, %r1178;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs365, %rs366}, %r1074;
	cvt.f32.bf16 	%r1180, %rs366;
	cvt.f32.bf16 	%r1181, %rs365;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1182, %r1294, %r892, %r1181;
	fma.rn.f32 	%r1183, %r1295, %r892, %r1180;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r962, %r1183, %r1182;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs367, %rs368}, %r1075;
	cvt.f32.bf16 	%r1184, %rs368;
	cvt.f32.bf16 	%r1185, %rs367;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1186, %r1296, %r893, %r1185;
	fma.rn.f32 	%r1187, %r1297, %r893, %r1184;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r963, %r1187, %r1186;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs369, %rs370}, %r1076;
	cvt.f32.bf16 	%r1188, %rs370;
	cvt.f32.bf16 	%r1189, %rs369;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1190, %r1298, %r894, %r1189;
	fma.rn.f32 	%r1191, %r1299, %r894, %r1188;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r952, %r1191, %r1190;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs371, %rs372}, %r1077;
	cvt.f32.bf16 	%r1192, %rs372;
	cvt.f32.bf16 	%r1193, %rs371;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1194, %r1300, %r895, %r1193;
	fma.rn.f32 	%r1195, %r1301, %r895, %r1192;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r953, %r1195, %r1194;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs373, %rs374}, %r1078;
	cvt.f32.bf16 	%r1196, %rs374;
	cvt.f32.bf16 	%r1197, %rs373;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1198, %r1302, %r894, %r1197;
	fma.rn.f32 	%r1199, %r1303, %r894, %r1196;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r960, %r1199, %r1198;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs375, %rs376}, %r1079;
	cvt.f32.bf16 	%r1200, %rs376;
	cvt.f32.bf16 	%r1201, %rs375;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1202, %r1304, %r895, %r1201;
	fma.rn.f32 	%r1203, %r1305, %r895, %r1200;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r961, %r1203, %r1202;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs377, %rs378}, %r1080;
	cvt.f32.bf16 	%r1204, %rs378;
	cvt.f32.bf16 	%r1205, %rs377;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1206, %r1306, %r894, %r1205;
	fma.rn.f32 	%r1207, %r1307, %r894, %r1204;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r956, %r1207, %r1206;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs379, %rs380}, %r1081;
	cvt.f32.bf16 	%r1208, %rs380;
	cvt.f32.bf16 	%r1209, %rs379;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1210, %r1308, %r895, %r1209;
	fma.rn.f32 	%r1211, %r1309, %r895, %r1208;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r957, %r1211, %r1210;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs381, %rs382}, %r1082;
	cvt.f32.bf16 	%r1212, %rs382;
	cvt.f32.bf16 	%r1213, %rs381;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1214, %r1310, %r894, %r1213;
	fma.rn.f32 	%r1215, %r1311, %r894, %r1212;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r964, %r1215, %r1214;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs383, %rs384}, %r1083;
	cvt.f32.bf16 	%r1216, %rs384;
	cvt.f32.bf16 	%r1217, %rs383;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1218, %r1312, %r895, %r1217;
	fma.rn.f32 	%r1219, %r1313, %r895, %r1216;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r965, %r1219, %r1218;
	bar.sync 	0;
	shl.b32 	%r1220, %r1248, 11;
	shl.b32 	%r1221, %r1248, 5;
	shl.b32 	%r1222, %r24, 4;
	shr.u32 	%r1223, %r1247, 1;
	bfe.s32 	%r1224, %r2, 2, 1;
	and.b32 	%r1225, %r1224, 1040;
	or.b32 	%r1226, %r1221, %r1222;
	xor.b32 	%r1227, %r1225, %r1223;
	xor.b32 	%r1228, %r1227, %r1226;
	or.b32 	%r1229, %r1228, %r1220;
	add.s32 	%r930, %r146, %r1229;
	// begin inline asm
	st.shared.v4.b32 [ %r930 + 0 ], { %r931, %r932, %r933, %r934 };
	// end inline asm
	add.s32 	%r935, %r930, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r935 + 0 ], { %r936, %r937, %r938, %r939 };
	// end inline asm
	xor.b32 	%r1230, %r1229, 64;
	add.s32 	%r940, %r146, %r1230;
	// begin inline asm
	st.shared.v4.b32 [ %r940 + 0 ], { %r941, %r942, %r943, %r944 };
	// end inline asm
	add.s32 	%r945, %r940, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r945 + 0 ], { %r946, %r947, %r948, %r949 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r1231, %r1247, 2;
	and.b32 	%r1232, %r1032, 512;
	and.b32 	%r1233, %r1039, 1040;
	or.b32 	%r1234, %r998, %r1231;
	xor.b32 	%r1235, %r1234, %r1233;
	or.b32 	%r1236, %r1235, %r1232;
	add.s32 	%r1237, %r146, %r1236;
	ld.shared.v4.b32 	{%r966, %r970, %r974, %r978}, [%r1237];
	xor.b32 	%r1238, %r1236, 32;
	add.s32 	%r1239, %r146, %r1238;
	ld.shared.v4.b32 	{%r967, %r971, %r975, %r979}, [%r1239+2048];
	xor.b32 	%r1240, %r1236, 64;
	add.s32 	%r1241, %r146, %r1240;
	ld.shared.v4.b32 	{%r968, %r972, %r976, %r980}, [%r1241+4096];
	xor.b32 	%r1242, %r1236, 96;
	add.s32 	%r1243, %r146, %r1242;
	ld.shared.v4.b32 	{%r969, %r973, %r977, %r981}, [%r1243+6144];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r930 + 0 ], { %r950, %r951, %r952, %r953 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r935 + 0 ], { %r954, %r955, %r956, %r957 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r940 + 0 ], { %r958, %r959, %r960, %r961 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r945 + 0 ], { %r962, %r963, %r964, %r965 };
	// end inline asm
	bar.sync 	0;
	ld.shared.v4.b32 	{%r982, %r986, %r990, %r994}, [%r1237];
	ld.shared.v4.b32 	{%r983, %r987, %r991, %r995}, [%r1239+2048];
	ld.shared.v4.b32 	{%r984, %r988, %r992, %r996}, [%r1241+4096];
	ld.shared.v4.b32 	{%r985, %r989, %r993, %r997}, [%r1243+6144];
	.loc	1 845 8                         // sk04_fa_o_w4a8.py:845:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd115 + 0 ], { %r966, %r967, %r968, %r969 };
	// end inline asm
	// begin inline asm
	@%p9 st.global.v4.b32 [ %rd116 + 0 ], { %r970, %r971, %r972, %r973 };
	// end inline asm
	// begin inline asm
	@%p10 st.global.v4.b32 [ %rd117 + 0 ], { %r974, %r975, %r976, %r977 };
	// end inline asm
	// begin inline asm
	@%p11 st.global.v4.b32 [ %rd118 + 0 ], { %r978, %r979, %r980, %r981 };
	// end inline asm
	// begin inline asm
	@%p12 st.global.v4.b32 [ %rd119 + 0 ], { %r982, %r983, %r984, %r985 };
	// end inline asm
	// begin inline asm
	@%p13 st.global.v4.b32 [ %rd120 + 0 ], { %r986, %r987, %r988, %r989 };
	// end inline asm
	// begin inline asm
	@%p14 st.global.v4.b32 [ %rd121 + 0 ], { %r990, %r991, %r992, %r993 };
	// end inline asm
	// begin inline asm
	@%p15 st.global.v4.b32 [ %rd122 + 0 ], { %r994, %r995, %r996, %r997 };
	// end inline asm
	.loc	1 843 4                         // sk04_fa_o_w4a8.py:843:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk04_fa_o_w4a8.py"
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
.b8 5                                   // DW_FORM_data2
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 165                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x9e DW_TAG_compile_unit
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
.b8 52
.b8 95
.b8 102
.b8 97
.b8 95
.b8 111
.b8 95
.b8 119
.b8 52
.b8 97
.b8 56
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
.b8 2                                   // Abbrev [2] 0x47:0x19 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 52
.b8 95
.b8 102
.b8 97
.b8 95
.b8 111
.b8 95
.b8 119
.b8 52
.b8 97
.b8 56
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x60:0x48 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 71                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x75:0x19 DW_TAG_inlined_subroutine
.b32 71                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 40                                  // DW_AT_call_line
.b8 3
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x8e:0x19 DW_TAG_inlined_subroutine
.b32 71                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 41                                  // DW_AT_call_line
.b8 3
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_2 = _Nativo(
    "sk04_fa_o_w4a8/tile64x128x128_shift0_abi14",
    _PTX_2, "_sk04_fa_o_w4a8_kernel",
    warps=4, shared=49664,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 17],
    horneado={10: 1, 12: 1, 14: 1, 16: 1, 18: 1, 19: 64, 20: 128, 21: 128, 22: 8},
    div16=[7, 8, 9, 11, 13, 15, 17],
)

_PTX_3 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk04_fa_o_w4a8_kernel  // -- Begin function _sk04_fa_o_w4a8_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk04_fa_o_w4a8_kernel
.visible .entry _sk04_fa_o_w4a8_kernel(
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_5,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_6,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_7,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_8,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_9,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_10,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_11,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_12,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_13,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_14,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_16
)
.reqntid 128
{
	.reg .pred 	%p<25>;
	.reg .b16 	%rs<449>;
	.reg .b32 	%r<1359>;
	.reg .b64 	%rd<211>;
	.loc	1 797 0                         // sk04_fa_o_w4a8.py:797:0
$L__func_begin0:
	.loc	1 797 0                         // sk04_fa_o_w4a8.py:797:0

// %bb.0:
	ld.param.b32 	%r52, [_sk04_fa_o_w4a8_kernel_param_13];
	ld.param.b32 	%r51, [_sk04_fa_o_w4a8_kernel_param_12];
	ld.param.b32 	%r50, [_sk04_fa_o_w4a8_kernel_param_11];
	ld.param.b32 	%r49, [_sk04_fa_o_w4a8_kernel_param_8];
	ld.param.b32 	%r48, [_sk04_fa_o_w4a8_kernel_param_7];
	ld.param.b32 	%r47, [_sk04_fa_o_w4a8_kernel_param_6];
	ld.param.b64 	%rd27, [_sk04_fa_o_w4a8_kernel_param_4];
	ld.param.b64 	%rd26, [_sk04_fa_o_w4a8_kernel_param_3];
	ld.param.b64 	%rd25, [_sk04_fa_o_w4a8_kernel_param_2];
	ld.param.b64 	%rd24, [_sk04_fa_o_w4a8_kernel_param_1];
	ld.param.b64 	%rd23, [_sk04_fa_o_w4a8_kernel_param_0];
$L__tmp0:
	.loc	1 807 24                        // sk04_fa_o_w4a8.py:807:24
	mov.u32 	%r96, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk04_fa_o_w4a8.py:808:27 ]
	add.s32 	%r97, %r47, 63;
	.loc	2 43 30                         // standard.py:43:30 @[ sk04_fa_o_w4a8.py:808:27 ]
	shr.s32 	%r98, %r97, 31;
	shr.u32 	%r99, %r98, 26;
	add.s32 	%r100, %r97, %r99;
	shr.s32 	%r101, %r100, 6;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk04_fa_o_w4a8.py:809:27 ]
	add.s32 	%r102, %r48, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk04_fa_o_w4a8.py:809:27 ]
	shr.s32 	%r103, %r102, 31;
	shr.u32 	%r104, %r103, 25;
	add.s32 	%r105, %r102, %r104;
	shr.s32 	%r106, %r105, 7;
$L__tmp3:
	.loc	1 810 29                        // sk04_fa_o_w4a8.py:810:29
	shl.b32 	%r107, %r106, 3;
	.loc	1 811 22                        // sk04_fa_o_w4a8.py:811:22
	div.s32 	%r108, %r96, %r107;
	.loc	1 811 38                        // sk04_fa_o_w4a8.py:811:38
	shl.b32 	%r109, %r108, 3;
	ld.param.b32 	%r110, [_sk04_fa_o_w4a8_kernel_param_9];
	.loc	1 812 30                        // sk04_fa_o_w4a8.py:812:30
	sub.s32 	%r111, %r101, %r109;
	ld.param.b32 	%r112, [_sk04_fa_o_w4a8_kernel_param_10];
	.loc	1 812 39                        // sk04_fa_o_w4a8.py:812:39
	min.s32 	%r113, %r111, 8;
	.loc	1 813 30                        // sk04_fa_o_w4a8.py:813:30
	mul.lo.s32 	%r114, %r108, %r107;
	sub.s32 	%r115, %r96, %r114;
	.loc	1 814 36                        // sk04_fa_o_w4a8.py:814:36
	div.s32 	%r116, %r115, %r113;
	.loc	1 813 46                        // sk04_fa_o_w4a8.py:813:46
	mul.lo.s32 	%r117, %r116, %r113;
	sub.s32 	%r118, %r115, %r117;
	.loc	1 813 23                        // sk04_fa_o_w4a8.py:813:23
	add.s32 	%r119, %r118, %r109;
	.loc	1 816 22                        // sk04_fa_o_w4a8.py:816:22
	shl.b32 	%r1, %r119, 6;
	.loc	1 816 45                        // sk04_fa_o_w4a8.py:816:45
	mov.u32 	%r2, %tid.x;
	and.b32 	%r120, %r2, 112;
	bfe.u32 	%r3, %r2, 4, 3;
	or.b32 	%r4, %r3, 8;
	or.b32 	%r5, %r3, 16;
	or.b32 	%r6, %r3, 24;
	or.b32 	%r7, %r3, 32;
	or.b32 	%r8, %r3, 40;
	or.b32 	%r9, %r3, 48;
	or.b32 	%r10, %r3, 56;
	and.b32 	%r11, %r2, 120;
	bfe.u32 	%r121, %r2, 3, 4;
	and.b32 	%r12, %r2, 28;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r122, %r1, %r121;
	or.b32 	%r123, %r122, 16;
	or.b32 	%r124, %r122, 32;
	or.b32 	%r125, %r122, 48;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r13, %r122, %r47;
	rem.s32 	%r14, %r123, %r47;
	rem.s32 	%r15, %r124, %r47;
	rem.s32 	%r16, %r125, %r47;
	.loc	1 817 22                        // sk04_fa_o_w4a8.py:817:22
	shl.b32 	%r17, %r116, 7;
	.loc	1 817 45                        // sk04_fa_o_w4a8.py:817:45
	and.b32 	%r18, %r2, 15;
	shl.b32 	%r19, %r18, 3;
	and.b32 	%r20, %r2, 7;
	shl.b32 	%r126, %r20, 4;
	and.b32 	%r21, %r2, 127;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r22, %r17, %r19;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r127, %r22, %r48;
	.loc	1 821 39                        // sk04_fa_o_w4a8.py:821:39
	mul.lo.s32 	%r128, %r13, %r110;
	mul.lo.s32 	%r129, %r14, %r110;
	mul.lo.s32 	%r130, %r15, %r110;
	mul.lo.s32 	%r131, %r16, %r110;
	.loc	1 821 21                        // sk04_fa_o_w4a8.py:821:21
	cvt.s64.s32 	%rd1, %r128;
	add.s64 	%rd65, %rd23, %rd1;
	cvt.s64.s32 	%rd2, %r129;
	add.s64 	%rd66, %rd23, %rd2;
	cvt.s64.s32 	%rd3, %r130;
	add.s64 	%rd67, %rd23, %rd3;
	cvt.s64.s32 	%rd4, %r131;
	add.s64 	%rd68, %rd23, %rd4;
	.loc	1 821 51                        // sk04_fa_o_w4a8.py:821:51
	cvt.u64.u32 	%rd5, %r126;
	add.s64 	%rd37, %rd65, %rd5;
	add.s64 	%rd38, %rd66, %rd5;
	add.s64 	%rd39, %rd67, %rd5;
	add.s64 	%rd40, %rd68, %rd5;
	.loc	1 822 40                        // sk04_fa_o_w4a8.py:822:40
	mul.lo.s32 	%r132, %r112, %r3;
	shl.b32 	%r133, %r112, 3;
	add.s32 	%r134, %r132, %r133;
	shl.b32 	%r135, %r112, 4;
	add.s32 	%r136, %r132, %r135;
	mad.lo.s32 	%r137, %r112, 24, %r132;
	shl.b32 	%r138, %r112, 5;
	add.s32 	%r139, %r132, %r138;
	mad.lo.s32 	%r140, %r112, 40, %r132;
	mad.lo.s32 	%r141, %r112, 48, %r132;
	mad.lo.s32 	%r142, %r112, 56, %r132;
	.loc	1 822 21                        // sk04_fa_o_w4a8.py:822:21
	cvt.s64.s32 	%rd6, %r132;
	add.s64 	%rd69, %rd24, %rd6;
	cvt.s64.s32 	%rd7, %r134;
	add.s64 	%rd70, %rd24, %rd7;
	cvt.s64.s32 	%rd8, %r136;
	add.s64 	%rd71, %rd24, %rd8;
	cvt.s64.s32 	%rd9, %r137;
	add.s64 	%rd72, %rd24, %rd9;
	cvt.s64.s32 	%rd10, %r139;
	add.s64 	%rd73, %rd24, %rd10;
	cvt.s64.s32 	%rd11, %r140;
	add.s64 	%rd74, %rd24, %rd11;
	cvt.s64.s32 	%rd12, %r141;
	add.s64 	%rd75, %rd24, %rd12;
	cvt.s64.s32 	%rd13, %r142;
	add.s64 	%rd76, %rd24, %rd13;
	.loc	1 822 52                        // sk04_fa_o_w4a8.py:822:52
	cvt.s64.s32 	%rd14, %r127;
	add.s64 	%rd29, %rd69, %rd14;
	add.s64 	%rd33, %rd70, %rd14;
	add.s64 	%rd30, %rd71, %rd14;
	add.s64 	%rd34, %rd72, %rd14;
	add.s64 	%rd31, %rd73, %rd14;
	add.s64 	%rd35, %rd74, %rd14;
	add.s64 	%rd32, %rd75, %rd14;
	add.s64 	%rd36, %rd76, %rd14;
	.loc	1 836 35                        // sk04_fa_o_w4a8.py:836:35
	shl.b32 	%r143, %r112, 6;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	setp.gt.s32 	%p1, %r49, 127;
	.loc	1 827 25                        // sk04_fa_o_w4a8.py:827:25
	shl.b32 	%r144, %r21, 3;
	shr.u32 	%r145, %r120, 1;
	xor.b32 	%r146, %r144, %r145;
	mov.b32 	%r147, global_smem;
	add.s32 	%r54, %r147, %r146;
	selp.b32 	%r55, 8, 0, %p1;
	// begin inline asm
	cp.async.ca.shared.global [ %r54 + 0 ], [ %rd29 + 0 ], 0x8, %r55;
	// end inline asm
	add.s32 	%r56, %r54, 2048;
	// begin inline asm
	cp.async.ca.shared.global [ %r56 + 0 ], [ %rd30 + 0 ], 0x8, %r55;
	// end inline asm
	add.s32 	%r57, %r54, 4096;
	// begin inline asm
	cp.async.ca.shared.global [ %r57 + 0 ], [ %rd31 + 0 ], 0x8, %r55;
	// end inline asm
	add.s32 	%r58, %r54, 6144;
	// begin inline asm
	cp.async.ca.shared.global [ %r58 + 0 ], [ %rd32 + 0 ], 0x8, %r55;
	// end inline asm
	xor.b32 	%r23, %r146, 64;
	add.s32 	%r148, %r147, %r23;
	add.s32 	%r59, %r148, 1024;
	// begin inline asm
	cp.async.ca.shared.global [ %r59 + 0 ], [ %rd33 + 0 ], 0x8, %r55;
	// end inline asm
	add.s32 	%r60, %r148, 3072;
	// begin inline asm
	cp.async.ca.shared.global [ %r60 + 0 ], [ %rd34 + 0 ], 0x8, %r55;
	// end inline asm
	add.s32 	%r61, %r148, 5120;
	// begin inline asm
	cp.async.ca.shared.global [ %r61 + 0 ], [ %rd35 + 0 ], 0x8, %r55;
	// end inline asm
	add.s32 	%r62, %r148, 7168;
	// begin inline asm
	cp.async.ca.shared.global [ %r62 + 0 ], [ %rd36 + 0 ], 0x8, %r55;
	// end inline asm
	cp.async.commit_group;
	.loc	1 831 52                        // sk04_fa_o_w4a8.py:831:52
	shl.b32 	%r149, %r21, 4;
	and.b32 	%r24, %r2, 24;
	shl.b32 	%r150, %r24, 2;
	xor.b32 	%r151, %r149, %r150;
	add.s32 	%r25, %r147, %r151;
	add.s32 	%r63, %r25, 24576;
	selp.b32 	%r64, 16, 0, %p1;
	// begin inline asm
	cp.async.cg.shared.global [ %r63 + 0 ], [ %rd37 + 0 ], 0x10, %r64;
	// end inline asm
	add.s32 	%r65, %r25, 26624;
	// begin inline asm
	cp.async.cg.shared.global [ %r65 + 0 ], [ %rd38 + 0 ], 0x10, %r64;
	// end inline asm
	add.s32 	%r66, %r25, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r66 + 0 ], [ %rd39 + 0 ], 0x10, %r64;
	// end inline asm
	add.s32 	%r67, %r25, 30720;
	// begin inline asm
	cp.async.cg.shared.global [ %r67 + 0 ], [ %rd40 + 0 ], 0x10, %r64;
	// end inline asm
	cp.async.commit_group;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	setp.gt.s32 	%p2, %r49, 255;
	.loc	1 835 18                        // sk04_fa_o_w4a8.py:835:18
	add.s64 	%rd49, %rd37, 128;
	add.s64 	%rd50, %rd38, 128;
	add.s64 	%rd51, %rd39, 128;
	add.s64 	%rd52, %rd40, 128;
	.loc	1 836 18                        // sk04_fa_o_w4a8.py:836:18
	cvt.s64.s32 	%rd15, %r143;
	add.s64 	%rd41, %rd29, %rd15;
	add.s64 	%rd45, %rd33, %rd15;
	add.s64 	%rd42, %rd30, %rd15;
	add.s64 	%rd46, %rd34, %rd15;
	add.s64 	%rd43, %rd31, %rd15;
	add.s64 	%rd47, %rd35, %rd15;
	add.s64 	%rd44, %rd32, %rd15;
	add.s64 	%rd48, %rd36, %rd15;
	.loc	1 827 25                        // sk04_fa_o_w4a8.py:827:25
	bar.sync 	0;
	add.s32 	%r68, %r54, 8192;
	selp.b32 	%r69, 8, 0, %p2;
	// begin inline asm
	cp.async.ca.shared.global [ %r68 + 0 ], [ %rd41 + 0 ], 0x8, %r69;
	// end inline asm
	add.s32 	%r70, %r54, 10240;
	// begin inline asm
	cp.async.ca.shared.global [ %r70 + 0 ], [ %rd42 + 0 ], 0x8, %r69;
	// end inline asm
	add.s32 	%r71, %r54, 12288;
	// begin inline asm
	cp.async.ca.shared.global [ %r71 + 0 ], [ %rd43 + 0 ], 0x8, %r69;
	// end inline asm
	add.s32 	%r72, %r54, 14336;
	// begin inline asm
	cp.async.ca.shared.global [ %r72 + 0 ], [ %rd44 + 0 ], 0x8, %r69;
	// end inline asm
	add.s32 	%r73, %r148, 9216;
	// begin inline asm
	cp.async.ca.shared.global [ %r73 + 0 ], [ %rd45 + 0 ], 0x8, %r69;
	// end inline asm
	add.s32 	%r74, %r148, 11264;
	// begin inline asm
	cp.async.ca.shared.global [ %r74 + 0 ], [ %rd46 + 0 ], 0x8, %r69;
	// end inline asm
	add.s32 	%r75, %r148, 13312;
	// begin inline asm
	cp.async.ca.shared.global [ %r75 + 0 ], [ %rd47 + 0 ], 0x8, %r69;
	// end inline asm
	add.s32 	%r76, %r148, 15360;
	// begin inline asm
	cp.async.ca.shared.global [ %r76 + 0 ], [ %rd48 + 0 ], 0x8, %r69;
	// end inline asm
	cp.async.commit_group;
	.loc	1 831 52                        // sk04_fa_o_w4a8.py:831:52
	add.s32 	%r77, %r25, 32768;
	selp.b32 	%r78, 16, 0, %p2;
	// begin inline asm
	cp.async.cg.shared.global [ %r77 + 0 ], [ %rd49 + 0 ], 0x10, %r78;
	// end inline asm
	add.s32 	%r79, %r25, 34816;
	// begin inline asm
	cp.async.cg.shared.global [ %r79 + 0 ], [ %rd50 + 0 ], 0x10, %r78;
	// end inline asm
	add.s32 	%r80, %r25, 36864;
	// begin inline asm
	cp.async.cg.shared.global [ %r80 + 0 ], [ %rd51 + 0 ], 0x10, %r78;
	// end inline asm
	add.s32 	%r81, %r25, 38912;
	// begin inline asm
	cp.async.cg.shared.global [ %r81 + 0 ], [ %rd52 + 0 ], 0x10, %r78;
	// end inline asm
	cp.async.commit_group;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	setp.gt.s32 	%p3, %r49, 383;
	.loc	1 835 18                        // sk04_fa_o_w4a8.py:835:18
	add.s64 	%rd61, %rd37, 256;
	add.s64 	%rd62, %rd38, 256;
	add.s64 	%rd63, %rd39, 256;
	add.s64 	%rd64, %rd40, 256;
	.loc	1 836 18                        // sk04_fa_o_w4a8.py:836:18
	add.s64 	%rd53, %rd41, %rd15;
	add.s64 	%rd57, %rd45, %rd15;
	add.s64 	%rd54, %rd42, %rd15;
	add.s64 	%rd58, %rd46, %rd15;
	add.s64 	%rd55, %rd43, %rd15;
	add.s64 	%rd59, %rd47, %rd15;
	add.s64 	%rd56, %rd44, %rd15;
	add.s64 	%rd60, %rd48, %rd15;
	.loc	1 827 25                        // sk04_fa_o_w4a8.py:827:25
	bar.sync 	0;
	add.s32 	%r82, %r54, 16384;
	selp.b32 	%r83, 8, 0, %p3;
	// begin inline asm
	cp.async.ca.shared.global [ %r82 + 0 ], [ %rd53 + 0 ], 0x8, %r83;
	// end inline asm
	add.s32 	%r84, %r54, 18432;
	// begin inline asm
	cp.async.ca.shared.global [ %r84 + 0 ], [ %rd54 + 0 ], 0x8, %r83;
	// end inline asm
	add.s32 	%r85, %r54, 20480;
	// begin inline asm
	cp.async.ca.shared.global [ %r85 + 0 ], [ %rd55 + 0 ], 0x8, %r83;
	// end inline asm
	add.s32 	%r86, %r54, 22528;
	// begin inline asm
	cp.async.ca.shared.global [ %r86 + 0 ], [ %rd56 + 0 ], 0x8, %r83;
	// end inline asm
	add.s32 	%r87, %r148, 17408;
	// begin inline asm
	cp.async.ca.shared.global [ %r87 + 0 ], [ %rd57 + 0 ], 0x8, %r83;
	// end inline asm
	add.s32 	%r88, %r148, 19456;
	// begin inline asm
	cp.async.ca.shared.global [ %r88 + 0 ], [ %rd58 + 0 ], 0x8, %r83;
	// end inline asm
	add.s32 	%r89, %r148, 21504;
	// begin inline asm
	cp.async.ca.shared.global [ %r89 + 0 ], [ %rd59 + 0 ], 0x8, %r83;
	// end inline asm
	add.s32 	%r90, %r148, 23552;
	// begin inline asm
	cp.async.ca.shared.global [ %r90 + 0 ], [ %rd60 + 0 ], 0x8, %r83;
	// end inline asm
	cp.async.commit_group;
	.loc	1 831 52                        // sk04_fa_o_w4a8.py:831:52
	add.s32 	%r91, %r25, 40960;
	selp.b32 	%r92, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r91 + 0 ], [ %rd61 + 0 ], 0x10, %r92;
	// end inline asm
	add.s32 	%r93, %r25, 43008;
	// begin inline asm
	cp.async.cg.shared.global [ %r93 + 0 ], [ %rd62 + 0 ], 0x10, %r92;
	// end inline asm
	add.s32 	%r94, %r25, 45056;
	// begin inline asm
	cp.async.cg.shared.global [ %r94 + 0 ], [ %rd63 + 0 ], 0x10, %r92;
	// end inline asm
	add.s32 	%r95, %r25, 47104;
	// begin inline asm
	cp.async.cg.shared.global [ %r95 + 0 ], [ %rd64 + 0 ], 0x10, %r92;
	// end inline asm
	cp.async.commit_group;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	@%p1 bra 	$L__BB0_2;
	bra.uni 	$L__BB0_1;
$L__BB0_2:                              // %.lr.ph
	.loc	1 0 22                          // sk04_fa_o_w4a8.py:0:22
	ld.param.b32 	%r53, [_sk04_fa_o_w4a8_kernel_param_14];
	ld.param.b64 	%rd28, [_sk04_fa_o_w4a8_kernel_param_5];
	.loc	1 825 27                        // sk04_fa_o_w4a8.py:825:27
	shr.u32 	%r152, %r49, 7;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r153, %r17, %r21;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r154, %r153, %r48;
	add.s32 	%r155, %r152, -3;
	and.b32 	%r1293, %r2, 3;
	shr.u32 	%r1294, %r2, 2;
	bfe.u32 	%r156, %r2, 2, 5;
	mul.lo.s32 	%r157, %r1293, 544;
	or.b32 	%r26, %r157, %r156;
	xor.b32 	%r27, %r26, 136;
	xor.b32 	%r28, %r26, 272;
	xor.b32 	%r29, %r26, 408;
	xor.b32 	%r30, %r26, 32;
	xor.b32 	%r31, %r26, 168;
	xor.b32 	%r32, %r26, 304;
	xor.b32 	%r33, %r26, 440;
	xor.b32 	%r34, %r26, 64;
	xor.b32 	%r35, %r26, 200;
	xor.b32 	%r36, %r26, 336;
	xor.b32 	%r37, %r26, 472;
	xor.b32 	%r38, %r26, 96;
	xor.b32 	%r39, %r26, 232;
	xor.b32 	%r40, %r26, 368;
	xor.b32 	%r41, %r26, 504;
	shl.b32 	%r158, %r12, 5;
	or.b32 	%r42, %r19, %r158;
	xor.b32 	%r43, %r42, 32;
	xor.b32 	%r44, %r42, 64;
	xor.b32 	%r45, %r42, 96;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	mad.wide.s32 	%rd16, %r154, 4, %rd28;
	shl.b32 	%r159, %r21, 2;
	add.s32 	%r160, %r147, %r159;
	add.s32 	%r326, %r160, 49152;
	shl.b32 	%r161, %r1293, 3;
	and.b32 	%r1292, %r2, 96;
	add.s32 	%r162, %r147, %r161;
	add.s32 	%r46, %r162, %r1292;
	cvt.s64.s32 	%rd17, %r155;
	and.b32 	%r163, %r49, -128;
	cvt.u64.u32 	%rd18, %r163;
	mad.lo.s64 	%rd77, %rd15, 3, %rd14;
	add.s64 	%rd208, %rd24, %rd77;
	add.s64 	%rd78, %rd5, %rd4;
	add.s64 	%rd79, %rd78, %rd23;
	add.s64 	%rd19, %rd79, 384;
	add.s64 	%rd80, %rd5, %rd3;
	add.s64 	%rd81, %rd80, %rd23;
	add.s64 	%rd20, %rd81, 384;
	add.s64 	%rd82, %rd5, %rd2;
	add.s64 	%rd83, %rd82, %rd23;
	add.s64 	%rd21, %rd83, 384;
	add.s64 	%rd84, %rd5, %rd1;
	add.s64 	%rd85, %rd84, %rd23;
	add.s64 	%rd22, %rd85, 384;
	mov.b32 	%r1295, 0f00000000;
	mov.b32 	%r1291, 2;
	mov.b32 	%r1290, -1;
	mov.b64 	%rd209, 0;
	mov.b32 	%r164, 0;
	mov.b32 	%r1289, %r164;
	mov.b64 	%rd210, %rd209;
	mov.b32 	%r1296, %r1295;
	mov.b32 	%r1297, %r1295;
	mov.b32 	%r1298, %r1295;
	mov.b32 	%r1299, %r1295;
	mov.b32 	%r1300, %r1295;
	mov.b32 	%r1301, %r1295;
	mov.b32 	%r1302, %r1295;
	mov.b32 	%r1303, %r1295;
	mov.b32 	%r1304, %r1295;
	mov.b32 	%r1305, %r1295;
	mov.b32 	%r1306, %r1295;
	mov.b32 	%r1307, %r1295;
	mov.b32 	%r1308, %r1295;
	mov.b32 	%r1309, %r1295;
	mov.b32 	%r1310, %r1295;
	mov.b32 	%r1311, %r1295;
	mov.b32 	%r1312, %r1295;
	mov.b32 	%r1313, %r1295;
	mov.b32 	%r1314, %r1295;
	mov.b32 	%r1315, %r1295;
	mov.b32 	%r1316, %r1295;
	mov.b32 	%r1317, %r1295;
	mov.b32 	%r1318, %r1295;
	mov.b32 	%r1319, %r1295;
	mov.b32 	%r1320, %r1295;
	mov.b32 	%r1321, %r1295;
	mov.b32 	%r1322, %r1295;
	mov.b32 	%r1323, %r1295;
	mov.b32 	%r1324, %r1295;
	mov.b32 	%r1325, %r1295;
	mov.b32 	%r1326, %r1295;
	mov.b32 	%r1327, %r1295;
	mov.b32 	%r1328, %r1295;
	mov.b32 	%r1329, %r1295;
	mov.b32 	%r1330, %r1295;
	mov.b32 	%r1331, %r1295;
	mov.b32 	%r1332, %r1295;
	mov.b32 	%r1333, %r1295;
	mov.b32 	%r1334, %r1295;
	mov.b32 	%r1335, %r1295;
	mov.b32 	%r1336, %r1295;
	mov.b32 	%r1337, %r1295;
	mov.b32 	%r1338, %r1295;
	mov.b32 	%r1339, %r1295;
	mov.b32 	%r1340, %r1295;
	mov.b32 	%r1341, %r1295;
	mov.b32 	%r1342, %r1295;
	mov.b32 	%r1343, %r1295;
	mov.b32 	%r1344, %r1295;
	mov.b32 	%r1345, %r1295;
	mov.b32 	%r1346, %r1295;
	mov.b32 	%r1347, %r1295;
	mov.b32 	%r1348, %r1295;
	mov.b32 	%r1349, %r1295;
	mov.b32 	%r1350, %r1295;
	mov.b32 	%r1351, %r1295;
	mov.b32 	%r1352, %r1295;
	mov.b32 	%r1353, %r1295;
	mov.b32 	%r1354, %r1295;
	mov.b32 	%r1355, %r1295;
	mov.b32 	%r1356, %r1295;
	mov.b32 	%r1357, %r1295;
	mov.b32 	%r1358, %r1295;
$L__BB0_3:                              // =>This Inner Loop Header: Depth=1
	setp.lt.s64 	%p4, %rd210, %rd17;
	add.s32 	%r341, %r1290, 1;
	setp.gt.s32 	%p5, %r341, 2;
	selp.b32 	%r1290, 0, %r341, %p5;
	.loc	1 827 25                        // sk04_fa_o_w4a8.py:827:25
	cp.async.wait_group 	4;
	bar.sync 	0;
	shl.b32 	%r342, %r1290, 13;
	add.s32 	%r343, %r147, %r342;
	.loc	1 828 38                        // sk04_fa_o_w4a8.py:828:38
	add.s32 	%r344, %r343, %r26;
	ld.shared.b8 	%rs1, [%r344];
	ld.shared.b8 	%rs2, [%r344+2048];
	ld.shared.b8 	%rs3, [%r344+4096];
	ld.shared.b8 	%rs4, [%r344+6144];
	add.s32 	%r345, %r343, %r27;
	ld.shared.b8 	%rs5, [%r345];
	ld.shared.b8 	%rs6, [%r345+2048];
	ld.shared.b8 	%rs7, [%r345+4096];
	ld.shared.b8 	%rs8, [%r345+6144];
	add.s32 	%r346, %r343, %r28;
	ld.shared.b8 	%rs9, [%r346];
	ld.shared.b8 	%rs10, [%r346+2048];
	ld.shared.b8 	%rs11, [%r346+4096];
	ld.shared.b8 	%rs12, [%r346+6144];
	add.s32 	%r347, %r343, %r29;
	ld.shared.b8 	%rs13, [%r347];
	ld.shared.b8 	%rs14, [%r347+2048];
	ld.shared.b8 	%rs15, [%r347+4096];
	ld.shared.b8 	%rs16, [%r347+6144];
	add.s32 	%r348, %r343, %r30;
	ld.shared.b8 	%rs17, [%r348];
	ld.shared.b8 	%rs18, [%r348+2048];
	ld.shared.b8 	%rs19, [%r348+4096];
	ld.shared.b8 	%rs20, [%r348+6144];
	add.s32 	%r349, %r343, %r31;
	ld.shared.b8 	%rs21, [%r349];
	ld.shared.b8 	%rs22, [%r349+2048];
	ld.shared.b8 	%rs23, [%r349+4096];
	ld.shared.b8 	%rs24, [%r349+6144];
	add.s32 	%r350, %r343, %r32;
	ld.shared.b8 	%rs25, [%r350];
	ld.shared.b8 	%rs26, [%r350+2048];
	ld.shared.b8 	%rs27, [%r350+4096];
	ld.shared.b8 	%rs28, [%r350+6144];
	add.s32 	%r351, %r343, %r33;
	ld.shared.b8 	%rs29, [%r351];
	ld.shared.b8 	%rs30, [%r351+2048];
	ld.shared.b8 	%rs31, [%r351+4096];
	ld.shared.b8 	%rs32, [%r351+6144];
	add.s32 	%r352, %r343, %r34;
	ld.shared.b8 	%rs33, [%r352];
	ld.shared.b8 	%rs34, [%r352+2048];
	ld.shared.b8 	%rs35, [%r352+4096];
	ld.shared.b8 	%rs36, [%r352+6144];
	add.s32 	%r353, %r343, %r35;
	ld.shared.b8 	%rs37, [%r353];
	ld.shared.b8 	%rs38, [%r353+2048];
	ld.shared.b8 	%rs39, [%r353+4096];
	ld.shared.b8 	%rs40, [%r353+6144];
	add.s32 	%r354, %r343, %r36;
	ld.shared.b8 	%rs41, [%r354];
	ld.shared.b8 	%rs42, [%r354+2048];
	ld.shared.b8 	%rs43, [%r354+4096];
	ld.shared.b8 	%rs44, [%r354+6144];
	add.s32 	%r355, %r343, %r37;
	ld.shared.b8 	%rs45, [%r355];
	ld.shared.b8 	%rs46, [%r355+2048];
	ld.shared.b8 	%rs47, [%r355+4096];
	ld.shared.b8 	%rs48, [%r355+6144];
	add.s32 	%r356, %r343, %r38;
	ld.shared.b8 	%rs49, [%r356];
	ld.shared.b8 	%rs50, [%r356+2048];
	ld.shared.b8 	%rs51, [%r356+4096];
	ld.shared.b8 	%rs52, [%r356+6144];
	add.s32 	%r357, %r343, %r39;
	ld.shared.b8 	%rs53, [%r357];
	ld.shared.b8 	%rs54, [%r357+2048];
	ld.shared.b8 	%rs55, [%r357+4096];
	ld.shared.b8 	%rs56, [%r357+6144];
	add.s32 	%r358, %r343, %r40;
	ld.shared.b8 	%rs57, [%r358];
	ld.shared.b8 	%rs58, [%r358+2048];
	ld.shared.b8 	%rs59, [%r358+4096];
	ld.shared.b8 	%rs60, [%r358+6144];
	add.s32 	%r359, %r343, %r41;
	ld.shared.b8 	%rs61, [%r359];
	ld.shared.b8 	%rs62, [%r359+2048];
	ld.shared.b8 	%rs63, [%r359+4096];
	ld.shared.b8 	%rs64, [%r359+6144];
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r360, %rs13;
	cvt.u32.u16 	%r361, %rs9;
	prmt.b32 	%r362, %r361, %r360, 0x3340U;
	cvt.u32.u16 	%r363, %rs5;
	cvt.u32.u16 	%r364, %rs1;
	prmt.b32 	%r365, %r364, %r363, 0x3340U;
	prmt.b32 	%r366, %r365, %r362, 0x5410U;
	and.b32 	%r367, %r366, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r368, %r367, 0, 0x7773U;
	cvt.u16.u32 	%rs65, %r368;
	add.s16 	%rs66, %rs65, -8;
	cvt.u32.u16 	%r369, %rs66;
	prmt.b32 	%r370, %r367, 0, 0x7772U;
	cvt.u16.u32 	%rs67, %r370;
	add.s16 	%rs68, %rs67, -8;
	cvt.u32.u16 	%r371, %rs68;
	prmt.b32 	%r372, %r371, %r369, 0x3340U;
	prmt.b32 	%r373, %r367, 0, 0x7771U;
	cvt.u16.u32 	%rs69, %r373;
	add.s16 	%rs70, %rs69, -8;
	cvt.u32.u16 	%r374, %rs70;
	prmt.b32 	%r375, %r367, 0, 0x7770U;
	cvt.u16.u32 	%rs71, %r375;
	add.s16 	%rs72, %rs71, -8;
	cvt.u32.u16 	%r376, %rs72;
	prmt.b32 	%r377, %r376, %r374, 0x3340U;
	prmt.b32 	%r169, %r377, %r372, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r378, %rs14;
	cvt.u32.u16 	%r379, %rs10;
	prmt.b32 	%r380, %r379, %r378, 0x3340U;
	cvt.u32.u16 	%r381, %rs6;
	cvt.u32.u16 	%r382, %rs2;
	prmt.b32 	%r383, %r382, %r381, 0x3340U;
	prmt.b32 	%r384, %r383, %r380, 0x5410U;
	and.b32 	%r385, %r384, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r386, %r385, 0, 0x7773U;
	cvt.u16.u32 	%rs73, %r386;
	add.s16 	%rs74, %rs73, -8;
	cvt.u32.u16 	%r387, %rs74;
	prmt.b32 	%r388, %r385, 0, 0x7772U;
	cvt.u16.u32 	%rs75, %r388;
	add.s16 	%rs76, %rs75, -8;
	cvt.u32.u16 	%r389, %rs76;
	prmt.b32 	%r390, %r389, %r387, 0x3340U;
	prmt.b32 	%r391, %r385, 0, 0x7771U;
	cvt.u16.u32 	%rs77, %r391;
	add.s16 	%rs78, %rs77, -8;
	cvt.u32.u16 	%r392, %rs78;
	prmt.b32 	%r393, %r385, 0, 0x7770U;
	cvt.u16.u32 	%rs79, %r393;
	add.s16 	%rs80, %rs79, -8;
	cvt.u32.u16 	%r394, %rs80;
	prmt.b32 	%r395, %r394, %r392, 0x3340U;
	prmt.b32 	%r170, %r395, %r390, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r396, %rs15;
	cvt.u32.u16 	%r397, %rs11;
	prmt.b32 	%r398, %r397, %r396, 0x3340U;
	cvt.u32.u16 	%r399, %rs7;
	cvt.u32.u16 	%r400, %rs3;
	prmt.b32 	%r401, %r400, %r399, 0x3340U;
	prmt.b32 	%r402, %r401, %r398, 0x5410U;
	and.b32 	%r403, %r402, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r404, %r403, 0, 0x7773U;
	cvt.u16.u32 	%rs81, %r404;
	add.s16 	%rs82, %rs81, -8;
	cvt.u32.u16 	%r405, %rs82;
	prmt.b32 	%r406, %r403, 0, 0x7772U;
	cvt.u16.u32 	%rs83, %r406;
	add.s16 	%rs84, %rs83, -8;
	cvt.u32.u16 	%r407, %rs84;
	prmt.b32 	%r408, %r407, %r405, 0x3340U;
	prmt.b32 	%r409, %r403, 0, 0x7771U;
	cvt.u16.u32 	%rs85, %r409;
	add.s16 	%rs86, %rs85, -8;
	cvt.u32.u16 	%r410, %rs86;
	prmt.b32 	%r411, %r403, 0, 0x7770U;
	cvt.u16.u32 	%rs87, %r411;
	add.s16 	%rs88, %rs87, -8;
	cvt.u32.u16 	%r412, %rs88;
	prmt.b32 	%r413, %r412, %r410, 0x3340U;
	prmt.b32 	%r213, %r413, %r408, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r414, %rs16;
	cvt.u32.u16 	%r415, %rs12;
	prmt.b32 	%r416, %r415, %r414, 0x3340U;
	cvt.u32.u16 	%r417, %rs8;
	cvt.u32.u16 	%r418, %rs4;
	prmt.b32 	%r419, %r418, %r417, 0x3340U;
	prmt.b32 	%r420, %r419, %r416, 0x5410U;
	and.b32 	%r421, %r420, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r422, %r421, 0, 0x7773U;
	cvt.u16.u32 	%rs89, %r422;
	add.s16 	%rs90, %rs89, -8;
	cvt.u32.u16 	%r423, %rs90;
	prmt.b32 	%r424, %r421, 0, 0x7772U;
	cvt.u16.u32 	%rs91, %r424;
	add.s16 	%rs92, %rs91, -8;
	cvt.u32.u16 	%r425, %rs92;
	prmt.b32 	%r426, %r425, %r423, 0x3340U;
	prmt.b32 	%r427, %r421, 0, 0x7771U;
	cvt.u16.u32 	%rs93, %r427;
	add.s16 	%rs94, %rs93, -8;
	cvt.u32.u16 	%r428, %rs94;
	prmt.b32 	%r429, %r421, 0, 0x7770U;
	cvt.u16.u32 	%rs95, %r429;
	add.s16 	%rs96, %rs95, -8;
	cvt.u32.u16 	%r430, %rs96;
	prmt.b32 	%r431, %r430, %r428, 0x3340U;
	prmt.b32 	%r214, %r431, %r426, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r432, %rs29;
	cvt.u32.u16 	%r433, %rs25;
	prmt.b32 	%r434, %r433, %r432, 0x3340U;
	cvt.u32.u16 	%r435, %rs21;
	cvt.u32.u16 	%r436, %rs17;
	prmt.b32 	%r437, %r436, %r435, 0x3340U;
	prmt.b32 	%r438, %r437, %r434, 0x5410U;
	and.b32 	%r439, %r438, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r440, %r439, 0, 0x7773U;
	cvt.u16.u32 	%rs97, %r440;
	add.s16 	%rs98, %rs97, -8;
	cvt.u32.u16 	%r441, %rs98;
	prmt.b32 	%r442, %r439, 0, 0x7772U;
	cvt.u16.u32 	%rs99, %r442;
	add.s16 	%rs100, %rs99, -8;
	cvt.u32.u16 	%r443, %rs100;
	prmt.b32 	%r444, %r443, %r441, 0x3340U;
	prmt.b32 	%r445, %r439, 0, 0x7771U;
	cvt.u16.u32 	%rs101, %r445;
	add.s16 	%rs102, %rs101, -8;
	cvt.u32.u16 	%r446, %rs102;
	prmt.b32 	%r447, %r439, 0, 0x7770U;
	cvt.u16.u32 	%rs103, %r447;
	add.s16 	%rs104, %rs103, -8;
	cvt.u32.u16 	%r448, %rs104;
	prmt.b32 	%r449, %r448, %r446, 0x3340U;
	prmt.b32 	%r175, %r449, %r444, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r450, %rs30;
	cvt.u32.u16 	%r451, %rs26;
	prmt.b32 	%r452, %r451, %r450, 0x3340U;
	cvt.u32.u16 	%r453, %rs22;
	cvt.u32.u16 	%r454, %rs18;
	prmt.b32 	%r455, %r454, %r453, 0x3340U;
	prmt.b32 	%r456, %r455, %r452, 0x5410U;
	and.b32 	%r457, %r456, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r458, %r457, 0, 0x7773U;
	cvt.u16.u32 	%rs105, %r458;
	add.s16 	%rs106, %rs105, -8;
	cvt.u32.u16 	%r459, %rs106;
	prmt.b32 	%r460, %r457, 0, 0x7772U;
	cvt.u16.u32 	%rs107, %r460;
	add.s16 	%rs108, %rs107, -8;
	cvt.u32.u16 	%r461, %rs108;
	prmt.b32 	%r462, %r461, %r459, 0x3340U;
	prmt.b32 	%r463, %r457, 0, 0x7771U;
	cvt.u16.u32 	%rs109, %r463;
	add.s16 	%rs110, %rs109, -8;
	cvt.u32.u16 	%r464, %rs110;
	prmt.b32 	%r465, %r457, 0, 0x7770U;
	cvt.u16.u32 	%rs111, %r465;
	add.s16 	%rs112, %rs111, -8;
	cvt.u32.u16 	%r466, %rs112;
	prmt.b32 	%r467, %r466, %r464, 0x3340U;
	prmt.b32 	%r176, %r467, %r462, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r468, %rs31;
	cvt.u32.u16 	%r469, %rs27;
	prmt.b32 	%r470, %r469, %r468, 0x3340U;
	cvt.u32.u16 	%r471, %rs23;
	cvt.u32.u16 	%r472, %rs19;
	prmt.b32 	%r473, %r472, %r471, 0x3340U;
	prmt.b32 	%r474, %r473, %r470, 0x5410U;
	and.b32 	%r475, %r474, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r476, %r475, 0, 0x7773U;
	cvt.u16.u32 	%rs113, %r476;
	add.s16 	%rs114, %rs113, -8;
	cvt.u32.u16 	%r477, %rs114;
	prmt.b32 	%r478, %r475, 0, 0x7772U;
	cvt.u16.u32 	%rs115, %r478;
	add.s16 	%rs116, %rs115, -8;
	cvt.u32.u16 	%r479, %rs116;
	prmt.b32 	%r480, %r479, %r477, 0x3340U;
	prmt.b32 	%r481, %r475, 0, 0x7771U;
	cvt.u16.u32 	%rs117, %r481;
	add.s16 	%rs118, %rs117, -8;
	cvt.u32.u16 	%r482, %rs118;
	prmt.b32 	%r483, %r475, 0, 0x7770U;
	cvt.u16.u32 	%rs119, %r483;
	add.s16 	%rs120, %rs119, -8;
	cvt.u32.u16 	%r484, %rs120;
	prmt.b32 	%r485, %r484, %r482, 0x3340U;
	prmt.b32 	%r223, %r485, %r480, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r486, %rs32;
	cvt.u32.u16 	%r487, %rs28;
	prmt.b32 	%r488, %r487, %r486, 0x3340U;
	cvt.u32.u16 	%r489, %rs24;
	cvt.u32.u16 	%r490, %rs20;
	prmt.b32 	%r491, %r490, %r489, 0x3340U;
	prmt.b32 	%r492, %r491, %r488, 0x5410U;
	and.b32 	%r493, %r492, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r494, %r493, 0, 0x7773U;
	cvt.u16.u32 	%rs121, %r494;
	add.s16 	%rs122, %rs121, -8;
	cvt.u32.u16 	%r495, %rs122;
	prmt.b32 	%r496, %r493, 0, 0x7772U;
	cvt.u16.u32 	%rs123, %r496;
	add.s16 	%rs124, %rs123, -8;
	cvt.u32.u16 	%r497, %rs124;
	prmt.b32 	%r498, %r497, %r495, 0x3340U;
	prmt.b32 	%r499, %r493, 0, 0x7771U;
	cvt.u16.u32 	%rs125, %r499;
	add.s16 	%rs126, %rs125, -8;
	cvt.u32.u16 	%r500, %rs126;
	prmt.b32 	%r501, %r493, 0, 0x7770U;
	cvt.u16.u32 	%rs127, %r501;
	add.s16 	%rs128, %rs127, -8;
	cvt.u32.u16 	%r502, %rs128;
	prmt.b32 	%r503, %r502, %r500, 0x3340U;
	prmt.b32 	%r224, %r503, %r498, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r504, %rs45;
	cvt.u32.u16 	%r505, %rs41;
	prmt.b32 	%r506, %r505, %r504, 0x3340U;
	cvt.u32.u16 	%r507, %rs37;
	cvt.u32.u16 	%r508, %rs33;
	prmt.b32 	%r509, %r508, %r507, 0x3340U;
	prmt.b32 	%r510, %r509, %r506, 0x5410U;
	and.b32 	%r511, %r510, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r512, %r511, 0, 0x7773U;
	cvt.u16.u32 	%rs129, %r512;
	add.s16 	%rs130, %rs129, -8;
	cvt.u32.u16 	%r513, %rs130;
	prmt.b32 	%r514, %r511, 0, 0x7772U;
	cvt.u16.u32 	%rs131, %r514;
	add.s16 	%rs132, %rs131, -8;
	cvt.u32.u16 	%r515, %rs132;
	prmt.b32 	%r516, %r515, %r513, 0x3340U;
	prmt.b32 	%r517, %r511, 0, 0x7771U;
	cvt.u16.u32 	%rs133, %r517;
	add.s16 	%rs134, %rs133, -8;
	cvt.u32.u16 	%r518, %rs134;
	prmt.b32 	%r519, %r511, 0, 0x7770U;
	cvt.u16.u32 	%rs135, %r519;
	add.s16 	%rs136, %rs135, -8;
	cvt.u32.u16 	%r520, %rs136;
	prmt.b32 	%r521, %r520, %r518, 0x3340U;
	prmt.b32 	%r177, %r521, %r516, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r522, %rs46;
	cvt.u32.u16 	%r523, %rs42;
	prmt.b32 	%r524, %r523, %r522, 0x3340U;
	cvt.u32.u16 	%r525, %rs38;
	cvt.u32.u16 	%r526, %rs34;
	prmt.b32 	%r527, %r526, %r525, 0x3340U;
	prmt.b32 	%r528, %r527, %r524, 0x5410U;
	and.b32 	%r529, %r528, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r530, %r529, 0, 0x7773U;
	cvt.u16.u32 	%rs137, %r530;
	add.s16 	%rs138, %rs137, -8;
	cvt.u32.u16 	%r531, %rs138;
	prmt.b32 	%r532, %r529, 0, 0x7772U;
	cvt.u16.u32 	%rs139, %r532;
	add.s16 	%rs140, %rs139, -8;
	cvt.u32.u16 	%r533, %rs140;
	prmt.b32 	%r534, %r533, %r531, 0x3340U;
	prmt.b32 	%r535, %r529, 0, 0x7771U;
	cvt.u16.u32 	%rs141, %r535;
	add.s16 	%rs142, %rs141, -8;
	cvt.u32.u16 	%r536, %rs142;
	prmt.b32 	%r537, %r529, 0, 0x7770U;
	cvt.u16.u32 	%rs143, %r537;
	add.s16 	%rs144, %rs143, -8;
	cvt.u32.u16 	%r538, %rs144;
	prmt.b32 	%r539, %r538, %r536, 0x3340U;
	prmt.b32 	%r178, %r539, %r534, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r540, %rs47;
	cvt.u32.u16 	%r541, %rs43;
	prmt.b32 	%r542, %r541, %r540, 0x3340U;
	cvt.u32.u16 	%r543, %rs39;
	cvt.u32.u16 	%r544, %rs35;
	prmt.b32 	%r545, %r544, %r543, 0x3340U;
	prmt.b32 	%r546, %r545, %r542, 0x5410U;
	and.b32 	%r547, %r546, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r548, %r547, 0, 0x7773U;
	cvt.u16.u32 	%rs145, %r548;
	add.s16 	%rs146, %rs145, -8;
	cvt.u32.u16 	%r549, %rs146;
	prmt.b32 	%r550, %r547, 0, 0x7772U;
	cvt.u16.u32 	%rs147, %r550;
	add.s16 	%rs148, %rs147, -8;
	cvt.u32.u16 	%r551, %rs148;
	prmt.b32 	%r552, %r551, %r549, 0x3340U;
	prmt.b32 	%r553, %r547, 0, 0x7771U;
	cvt.u16.u32 	%rs149, %r553;
	add.s16 	%rs150, %rs149, -8;
	cvt.u32.u16 	%r554, %rs150;
	prmt.b32 	%r555, %r547, 0, 0x7770U;
	cvt.u16.u32 	%rs151, %r555;
	add.s16 	%rs152, %rs151, -8;
	cvt.u32.u16 	%r556, %rs152;
	prmt.b32 	%r557, %r556, %r554, 0x3340U;
	prmt.b32 	%r229, %r557, %r552, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r558, %rs48;
	cvt.u32.u16 	%r559, %rs44;
	prmt.b32 	%r560, %r559, %r558, 0x3340U;
	cvt.u32.u16 	%r561, %rs40;
	cvt.u32.u16 	%r562, %rs36;
	prmt.b32 	%r563, %r562, %r561, 0x3340U;
	prmt.b32 	%r564, %r563, %r560, 0x5410U;
	and.b32 	%r565, %r564, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r566, %r565, 0, 0x7773U;
	cvt.u16.u32 	%rs153, %r566;
	add.s16 	%rs154, %rs153, -8;
	cvt.u32.u16 	%r567, %rs154;
	prmt.b32 	%r568, %r565, 0, 0x7772U;
	cvt.u16.u32 	%rs155, %r568;
	add.s16 	%rs156, %rs155, -8;
	cvt.u32.u16 	%r569, %rs156;
	prmt.b32 	%r570, %r569, %r567, 0x3340U;
	prmt.b32 	%r571, %r565, 0, 0x7771U;
	cvt.u16.u32 	%rs157, %r571;
	add.s16 	%rs158, %rs157, -8;
	cvt.u32.u16 	%r572, %rs158;
	prmt.b32 	%r573, %r565, 0, 0x7770U;
	cvt.u16.u32 	%rs159, %r573;
	add.s16 	%rs160, %rs159, -8;
	cvt.u32.u16 	%r574, %rs160;
	prmt.b32 	%r575, %r574, %r572, 0x3340U;
	prmt.b32 	%r230, %r575, %r570, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r576, %rs61;
	cvt.u32.u16 	%r577, %rs57;
	prmt.b32 	%r578, %r577, %r576, 0x3340U;
	cvt.u32.u16 	%r579, %rs53;
	cvt.u32.u16 	%r580, %rs49;
	prmt.b32 	%r581, %r580, %r579, 0x3340U;
	prmt.b32 	%r582, %r581, %r578, 0x5410U;
	and.b32 	%r583, %r582, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r584, %r583, 0, 0x7773U;
	cvt.u16.u32 	%rs161, %r584;
	add.s16 	%rs162, %rs161, -8;
	cvt.u32.u16 	%r585, %rs162;
	prmt.b32 	%r586, %r583, 0, 0x7772U;
	cvt.u16.u32 	%rs163, %r586;
	add.s16 	%rs164, %rs163, -8;
	cvt.u32.u16 	%r587, %rs164;
	prmt.b32 	%r588, %r587, %r585, 0x3340U;
	prmt.b32 	%r589, %r583, 0, 0x7771U;
	cvt.u16.u32 	%rs165, %r589;
	add.s16 	%rs166, %rs165, -8;
	cvt.u32.u16 	%r590, %rs166;
	prmt.b32 	%r591, %r583, 0, 0x7770U;
	cvt.u16.u32 	%rs167, %r591;
	add.s16 	%rs168, %rs167, -8;
	cvt.u32.u16 	%r592, %rs168;
	prmt.b32 	%r593, %r592, %r590, 0x3340U;
	prmt.b32 	%r179, %r593, %r588, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r594, %rs62;
	cvt.u32.u16 	%r595, %rs58;
	prmt.b32 	%r596, %r595, %r594, 0x3340U;
	cvt.u32.u16 	%r597, %rs54;
	cvt.u32.u16 	%r598, %rs50;
	prmt.b32 	%r599, %r598, %r597, 0x3340U;
	prmt.b32 	%r600, %r599, %r596, 0x5410U;
	and.b32 	%r601, %r600, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r602, %r601, 0, 0x7773U;
	cvt.u16.u32 	%rs169, %r602;
	add.s16 	%rs170, %rs169, -8;
	cvt.u32.u16 	%r603, %rs170;
	prmt.b32 	%r604, %r601, 0, 0x7772U;
	cvt.u16.u32 	%rs171, %r604;
	add.s16 	%rs172, %rs171, -8;
	cvt.u32.u16 	%r605, %rs172;
	prmt.b32 	%r606, %r605, %r603, 0x3340U;
	prmt.b32 	%r607, %r601, 0, 0x7771U;
	cvt.u16.u32 	%rs173, %r607;
	add.s16 	%rs174, %rs173, -8;
	cvt.u32.u16 	%r608, %rs174;
	prmt.b32 	%r609, %r601, 0, 0x7770U;
	cvt.u16.u32 	%rs175, %r609;
	add.s16 	%rs176, %rs175, -8;
	cvt.u32.u16 	%r610, %rs176;
	prmt.b32 	%r611, %r610, %r608, 0x3340U;
	prmt.b32 	%r180, %r611, %r606, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r612, %rs63;
	cvt.u32.u16 	%r613, %rs59;
	prmt.b32 	%r614, %r613, %r612, 0x3340U;
	cvt.u32.u16 	%r615, %rs55;
	cvt.u32.u16 	%r616, %rs51;
	prmt.b32 	%r617, %r616, %r615, 0x3340U;
	prmt.b32 	%r618, %r617, %r614, 0x5410U;
	and.b32 	%r619, %r618, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r620, %r619, 0, 0x7773U;
	cvt.u16.u32 	%rs177, %r620;
	add.s16 	%rs178, %rs177, -8;
	cvt.u32.u16 	%r621, %rs178;
	prmt.b32 	%r622, %r619, 0, 0x7772U;
	cvt.u16.u32 	%rs179, %r622;
	add.s16 	%rs180, %rs179, -8;
	cvt.u32.u16 	%r623, %rs180;
	prmt.b32 	%r624, %r623, %r621, 0x3340U;
	prmt.b32 	%r625, %r619, 0, 0x7771U;
	cvt.u16.u32 	%rs181, %r625;
	add.s16 	%rs182, %rs181, -8;
	cvt.u32.u16 	%r626, %rs182;
	prmt.b32 	%r627, %r619, 0, 0x7770U;
	cvt.u16.u32 	%rs183, %r627;
	add.s16 	%rs184, %rs183, -8;
	cvt.u32.u16 	%r628, %rs184;
	prmt.b32 	%r629, %r628, %r626, 0x3340U;
	prmt.b32 	%r235, %r629, %r624, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r630, %rs64;
	cvt.u32.u16 	%r631, %rs60;
	prmt.b32 	%r632, %r631, %r630, 0x3340U;
	cvt.u32.u16 	%r633, %rs56;
	cvt.u32.u16 	%r634, %rs52;
	prmt.b32 	%r635, %r634, %r633, 0x3340U;
	prmt.b32 	%r636, %r635, %r632, 0x5410U;
	and.b32 	%r637, %r636, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r638, %r637, 0, 0x7773U;
	cvt.u16.u32 	%rs185, %r638;
	add.s16 	%rs186, %rs185, -8;
	cvt.u32.u16 	%r639, %rs186;
	prmt.b32 	%r640, %r637, 0, 0x7772U;
	cvt.u16.u32 	%rs187, %r640;
	add.s16 	%rs188, %rs187, -8;
	cvt.u32.u16 	%r641, %rs188;
	prmt.b32 	%r642, %r641, %r639, 0x3340U;
	prmt.b32 	%r643, %r637, 0, 0x7771U;
	cvt.u16.u32 	%rs189, %r643;
	add.s16 	%rs190, %rs189, -8;
	cvt.u32.u16 	%r644, %rs190;
	prmt.b32 	%r645, %r637, 0, 0x7770U;
	cvt.u16.u32 	%rs191, %r645;
	add.s16 	%rs192, %rs191, -8;
	cvt.u32.u16 	%r646, %rs192;
	prmt.b32 	%r647, %r646, %r644, 0x3340U;
	prmt.b32 	%r236, %r647, %r642, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs193, %rs1, 4;
	shr.u16 	%rs194, %rs5, 4;
	shr.u16 	%rs195, %rs9, 4;
	shr.u16 	%rs196, %rs13, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs197, %rs196, -8;
	cvt.u32.u16 	%r648, %rs197;
	add.s16 	%rs198, %rs195, -8;
	cvt.u32.u16 	%r649, %rs198;
	prmt.b32 	%r650, %r649, %r648, 0x3340U;
	add.s16 	%rs199, %rs194, -8;
	cvt.u32.u16 	%r651, %rs199;
	add.s16 	%rs200, %rs193, -8;
	cvt.u32.u16 	%r652, %rs200;
	prmt.b32 	%r653, %r652, %r651, 0x3340U;
	prmt.b32 	%r281, %r653, %r650, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs201, %rs2, 4;
	shr.u16 	%rs202, %rs6, 4;
	shr.u16 	%rs203, %rs10, 4;
	shr.u16 	%rs204, %rs14, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs205, %rs204, -8;
	cvt.u32.u16 	%r654, %rs205;
	add.s16 	%rs206, %rs203, -8;
	cvt.u32.u16 	%r655, %rs206;
	prmt.b32 	%r656, %r655, %r654, 0x3340U;
	add.s16 	%rs207, %rs202, -8;
	cvt.u32.u16 	%r657, %rs207;
	add.s16 	%rs208, %rs201, -8;
	cvt.u32.u16 	%r658, %rs208;
	prmt.b32 	%r659, %r658, %r657, 0x3340U;
	prmt.b32 	%r282, %r659, %r656, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs209, %rs3, 4;
	shr.u16 	%rs210, %rs7, 4;
	shr.u16 	%rs211, %rs11, 4;
	shr.u16 	%rs212, %rs15, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs213, %rs212, -8;
	cvt.u32.u16 	%r660, %rs213;
	add.s16 	%rs214, %rs211, -8;
	cvt.u32.u16 	%r661, %rs214;
	prmt.b32 	%r662, %r661, %r660, 0x3340U;
	add.s16 	%rs215, %rs210, -8;
	cvt.u32.u16 	%r663, %rs215;
	add.s16 	%rs216, %rs209, -8;
	cvt.u32.u16 	%r664, %rs216;
	prmt.b32 	%r665, %r664, %r663, 0x3340U;
	prmt.b32 	%r305, %r665, %r662, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs217, %rs4, 4;
	shr.u16 	%rs218, %rs8, 4;
	shr.u16 	%rs219, %rs12, 4;
	shr.u16 	%rs220, %rs16, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs221, %rs220, -8;
	cvt.u32.u16 	%r666, %rs221;
	add.s16 	%rs222, %rs219, -8;
	cvt.u32.u16 	%r667, %rs222;
	prmt.b32 	%r668, %r667, %r666, 0x3340U;
	add.s16 	%rs223, %rs218, -8;
	cvt.u32.u16 	%r669, %rs223;
	add.s16 	%rs224, %rs217, -8;
	cvt.u32.u16 	%r670, %rs224;
	prmt.b32 	%r671, %r670, %r669, 0x3340U;
	prmt.b32 	%r306, %r671, %r668, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs225, %rs17, 4;
	shr.u16 	%rs226, %rs21, 4;
	shr.u16 	%rs227, %rs25, 4;
	shr.u16 	%rs228, %rs29, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs229, %rs228, -8;
	cvt.u32.u16 	%r672, %rs229;
	add.s16 	%rs230, %rs227, -8;
	cvt.u32.u16 	%r673, %rs230;
	prmt.b32 	%r674, %r673, %r672, 0x3340U;
	add.s16 	%rs231, %rs226, -8;
	cvt.u32.u16 	%r675, %rs231;
	add.s16 	%rs232, %rs225, -8;
	cvt.u32.u16 	%r676, %rs232;
	prmt.b32 	%r677, %r676, %r675, 0x3340U;
	prmt.b32 	%r287, %r677, %r674, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs233, %rs18, 4;
	shr.u16 	%rs234, %rs22, 4;
	shr.u16 	%rs235, %rs26, 4;
	shr.u16 	%rs236, %rs30, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs237, %rs236, -8;
	cvt.u32.u16 	%r678, %rs237;
	add.s16 	%rs238, %rs235, -8;
	cvt.u32.u16 	%r679, %rs238;
	prmt.b32 	%r680, %r679, %r678, 0x3340U;
	add.s16 	%rs239, %rs234, -8;
	cvt.u32.u16 	%r681, %rs239;
	add.s16 	%rs240, %rs233, -8;
	cvt.u32.u16 	%r682, %rs240;
	prmt.b32 	%r683, %r682, %r681, 0x3340U;
	prmt.b32 	%r288, %r683, %r680, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs241, %rs19, 4;
	shr.u16 	%rs242, %rs23, 4;
	shr.u16 	%rs243, %rs27, 4;
	shr.u16 	%rs244, %rs31, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs245, %rs244, -8;
	cvt.u32.u16 	%r684, %rs245;
	add.s16 	%rs246, %rs243, -8;
	cvt.u32.u16 	%r685, %rs246;
	prmt.b32 	%r686, %r685, %r684, 0x3340U;
	add.s16 	%rs247, %rs242, -8;
	cvt.u32.u16 	%r687, %rs247;
	add.s16 	%rs248, %rs241, -8;
	cvt.u32.u16 	%r688, %rs248;
	prmt.b32 	%r689, %r688, %r687, 0x3340U;
	prmt.b32 	%r311, %r689, %r686, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs249, %rs20, 4;
	shr.u16 	%rs250, %rs24, 4;
	shr.u16 	%rs251, %rs28, 4;
	shr.u16 	%rs252, %rs32, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs253, %rs252, -8;
	cvt.u32.u16 	%r690, %rs253;
	add.s16 	%rs254, %rs251, -8;
	cvt.u32.u16 	%r691, %rs254;
	prmt.b32 	%r692, %r691, %r690, 0x3340U;
	add.s16 	%rs255, %rs250, -8;
	cvt.u32.u16 	%r693, %rs255;
	add.s16 	%rs256, %rs249, -8;
	cvt.u32.u16 	%r694, %rs256;
	prmt.b32 	%r695, %r694, %r693, 0x3340U;
	prmt.b32 	%r312, %r695, %r692, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs257, %rs33, 4;
	shr.u16 	%rs258, %rs37, 4;
	shr.u16 	%rs259, %rs41, 4;
	shr.u16 	%rs260, %rs45, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs261, %rs260, -8;
	cvt.u32.u16 	%r696, %rs261;
	add.s16 	%rs262, %rs259, -8;
	cvt.u32.u16 	%r697, %rs262;
	prmt.b32 	%r698, %r697, %r696, 0x3340U;
	add.s16 	%rs263, %rs258, -8;
	cvt.u32.u16 	%r699, %rs263;
	add.s16 	%rs264, %rs257, -8;
	cvt.u32.u16 	%r700, %rs264;
	prmt.b32 	%r701, %r700, %r699, 0x3340U;
	prmt.b32 	%r289, %r701, %r698, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs265, %rs34, 4;
	shr.u16 	%rs266, %rs38, 4;
	shr.u16 	%rs267, %rs42, 4;
	shr.u16 	%rs268, %rs46, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs269, %rs268, -8;
	cvt.u32.u16 	%r702, %rs269;
	add.s16 	%rs270, %rs267, -8;
	cvt.u32.u16 	%r703, %rs270;
	prmt.b32 	%r704, %r703, %r702, 0x3340U;
	add.s16 	%rs271, %rs266, -8;
	cvt.u32.u16 	%r705, %rs271;
	add.s16 	%rs272, %rs265, -8;
	cvt.u32.u16 	%r706, %rs272;
	prmt.b32 	%r707, %r706, %r705, 0x3340U;
	prmt.b32 	%r290, %r707, %r704, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs273, %rs35, 4;
	shr.u16 	%rs274, %rs39, 4;
	shr.u16 	%rs275, %rs43, 4;
	shr.u16 	%rs276, %rs47, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs277, %rs276, -8;
	cvt.u32.u16 	%r708, %rs277;
	add.s16 	%rs278, %rs275, -8;
	cvt.u32.u16 	%r709, %rs278;
	prmt.b32 	%r710, %r709, %r708, 0x3340U;
	add.s16 	%rs279, %rs274, -8;
	cvt.u32.u16 	%r711, %rs279;
	add.s16 	%rs280, %rs273, -8;
	cvt.u32.u16 	%r712, %rs280;
	prmt.b32 	%r713, %r712, %r711, 0x3340U;
	prmt.b32 	%r313, %r713, %r710, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs281, %rs36, 4;
	shr.u16 	%rs282, %rs40, 4;
	shr.u16 	%rs283, %rs44, 4;
	shr.u16 	%rs284, %rs48, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs285, %rs284, -8;
	cvt.u32.u16 	%r714, %rs285;
	add.s16 	%rs286, %rs283, -8;
	cvt.u32.u16 	%r715, %rs286;
	prmt.b32 	%r716, %r715, %r714, 0x3340U;
	add.s16 	%rs287, %rs282, -8;
	cvt.u32.u16 	%r717, %rs287;
	add.s16 	%rs288, %rs281, -8;
	cvt.u32.u16 	%r718, %rs288;
	prmt.b32 	%r719, %r718, %r717, 0x3340U;
	prmt.b32 	%r314, %r719, %r716, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs289, %rs49, 4;
	shr.u16 	%rs290, %rs53, 4;
	shr.u16 	%rs291, %rs57, 4;
	shr.u16 	%rs292, %rs61, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs293, %rs292, -8;
	cvt.u32.u16 	%r720, %rs293;
	add.s16 	%rs294, %rs291, -8;
	cvt.u32.u16 	%r721, %rs294;
	prmt.b32 	%r722, %r721, %r720, 0x3340U;
	add.s16 	%rs295, %rs290, -8;
	cvt.u32.u16 	%r723, %rs295;
	add.s16 	%rs296, %rs289, -8;
	cvt.u32.u16 	%r724, %rs296;
	prmt.b32 	%r725, %r724, %r723, 0x3340U;
	prmt.b32 	%r291, %r725, %r722, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs297, %rs50, 4;
	shr.u16 	%rs298, %rs54, 4;
	shr.u16 	%rs299, %rs58, 4;
	shr.u16 	%rs300, %rs62, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs301, %rs300, -8;
	cvt.u32.u16 	%r726, %rs301;
	add.s16 	%rs302, %rs299, -8;
	cvt.u32.u16 	%r727, %rs302;
	prmt.b32 	%r728, %r727, %r726, 0x3340U;
	add.s16 	%rs303, %rs298, -8;
	cvt.u32.u16 	%r729, %rs303;
	add.s16 	%rs304, %rs297, -8;
	cvt.u32.u16 	%r730, %rs304;
	prmt.b32 	%r731, %r730, %r729, 0x3340U;
	prmt.b32 	%r292, %r731, %r728, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs305, %rs51, 4;
	shr.u16 	%rs306, %rs55, 4;
	shr.u16 	%rs307, %rs59, 4;
	shr.u16 	%rs308, %rs63, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs309, %rs308, -8;
	cvt.u32.u16 	%r732, %rs309;
	add.s16 	%rs310, %rs307, -8;
	cvt.u32.u16 	%r733, %rs310;
	prmt.b32 	%r734, %r733, %r732, 0x3340U;
	add.s16 	%rs311, %rs306, -8;
	cvt.u32.u16 	%r735, %rs311;
	add.s16 	%rs312, %rs305, -8;
	cvt.u32.u16 	%r736, %rs312;
	prmt.b32 	%r737, %r736, %r735, 0x3340U;
	prmt.b32 	%r315, %r737, %r734, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs313, %rs52, 4;
	shr.u16 	%rs314, %rs56, 4;
	shr.u16 	%rs315, %rs60, 4;
	shr.u16 	%rs316, %rs64, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs317, %rs316, -8;
	cvt.u32.u16 	%r738, %rs317;
	add.s16 	%rs318, %rs315, -8;
	cvt.u32.u16 	%r739, %rs318;
	prmt.b32 	%r740, %r739, %r738, 0x3340U;
	add.s16 	%rs319, %rs314, -8;
	cvt.u32.u16 	%r741, %rs319;
	add.s16 	%rs320, %rs313, -8;
	cvt.u32.u16 	%r742, %rs320;
	prmt.b32 	%r743, %r742, %r741, 0x3340U;
	prmt.b32 	%r316, %r743, %r740, 0x5410U;
	.loc	1 831 33                        // sk04_fa_o_w4a8.py:831:33
	add.s32 	%r744, %r343, %r42;
	ld.shared.v2.b32 	{%r745, %r746}, [%r744+24576];
	ld.shared.v2.b32 	{%r747, %r748}, [%r744+25600];
	ld.shared.v2.b32 	{%r749, %r750}, [%r744+26624];
	ld.shared.v2.b32 	{%r751, %r752}, [%r744+27648];
	ld.shared.v2.b32 	{%r753, %r754}, [%r744+28672];
	ld.shared.v2.b32 	{%r755, %r756}, [%r744+29696];
	ld.shared.v2.b32 	{%r757, %r758}, [%r744+30720];
	ld.shared.v2.b32 	{%r759, %r760}, [%r744+31744];
	add.s32 	%r761, %r343, %r43;
	ld.shared.v2.b32 	{%r762, %r763}, [%r761+24576];
	ld.shared.v2.b32 	{%r764, %r765}, [%r761+25600];
	ld.shared.v2.b32 	{%r766, %r767}, [%r761+26624];
	ld.shared.v2.b32 	{%r768, %r769}, [%r761+27648];
	ld.shared.v2.b32 	{%r770, %r771}, [%r761+28672];
	ld.shared.v2.b32 	{%r772, %r773}, [%r761+29696];
	ld.shared.v2.b32 	{%r774, %r775}, [%r761+30720];
	ld.shared.v2.b32 	{%r776, %r777}, [%r761+31744];
	add.s32 	%r778, %r343, %r44;
	ld.shared.v2.b32 	{%r779, %r780}, [%r778+24576];
	ld.shared.v2.b32 	{%r781, %r782}, [%r778+25600];
	ld.shared.v2.b32 	{%r783, %r784}, [%r778+26624];
	ld.shared.v2.b32 	{%r785, %r786}, [%r778+27648];
	ld.shared.v2.b32 	{%r787, %r788}, [%r778+28672];
	ld.shared.v2.b32 	{%r789, %r790}, [%r778+29696];
	ld.shared.v2.b32 	{%r791, %r792}, [%r778+30720];
	ld.shared.v2.b32 	{%r793, %r794}, [%r778+31744];
	add.s32 	%r795, %r343, %r45;
	ld.shared.v2.b32 	{%r796, %r797}, [%r795+24576];
	ld.shared.v2.b32 	{%r798, %r799}, [%r795+25600];
	ld.shared.v2.b32 	{%r800, %r801}, [%r795+26624];
	ld.shared.v2.b32 	{%r802, %r803}, [%r795+27648];
	ld.shared.v2.b32 	{%r804, %r805}, [%r795+28672];
	ld.shared.v2.b32 	{%r806, %r807}, [%r795+29696];
	ld.shared.v2.b32 	{%r808, %r809}, [%r795+30720];
	ld.shared.v2.b32 	{%r810, %r811}, [%r795+31744];
	.loc	1 832 33                        // sk04_fa_o_w4a8.py:832:33
	prmt.b32 	%r165, %r745, %r746, 0x6420U;
	prmt.b32 	%r166, %r747, %r748, 0x6420U;
	prmt.b32 	%r167, %r762, %r763, 0x6420U;
	prmt.b32 	%r168, %r764, %r765, 0x6420U;
	prmt.b32 	%r197, %r779, %r780, 0x6420U;
	prmt.b32 	%r198, %r781, %r782, 0x6420U;
	prmt.b32 	%r199, %r796, %r797, 0x6420U;
	prmt.b32 	%r200, %r798, %r799, 0x6420U;
	prmt.b32 	%r171, %r749, %r750, 0x6420U;
	prmt.b32 	%r172, %r751, %r752, 0x6420U;
	prmt.b32 	%r173, %r766, %r767, 0x6420U;
	prmt.b32 	%r174, %r768, %r769, 0x6420U;
	prmt.b32 	%r219, %r783, %r784, 0x6420U;
	prmt.b32 	%r220, %r785, %r786, 0x6420U;
	prmt.b32 	%r221, %r800, %r801, 0x6420U;
	prmt.b32 	%r222, %r802, %r803, 0x6420U;
	prmt.b32 	%r181, %r753, %r754, 0x6420U;
	prmt.b32 	%r182, %r755, %r756, 0x6420U;
	prmt.b32 	%r183, %r770, %r771, 0x6420U;
	prmt.b32 	%r184, %r772, %r773, 0x6420U;
	prmt.b32 	%r245, %r787, %r788, 0x6420U;
	prmt.b32 	%r246, %r789, %r790, 0x6420U;
	prmt.b32 	%r247, %r804, %r805, 0x6420U;
	prmt.b32 	%r248, %r806, %r807, 0x6420U;
	prmt.b32 	%r185, %r757, %r758, 0x6420U;
	prmt.b32 	%r186, %r759, %r760, 0x6420U;
	prmt.b32 	%r187, %r774, %r775, 0x6420U;
	prmt.b32 	%r188, %r776, %r777, 0x6420U;
	prmt.b32 	%r265, %r791, %r792, 0x6420U;
	prmt.b32 	%r266, %r793, %r794, 0x6420U;
	prmt.b32 	%r267, %r808, %r809, 0x6420U;
	prmt.b32 	%r268, %r810, %r811, 0x6420U;
	mov.b32 	%r189, %r164;
	mov.b32 	%r190, %r164;
	mov.b32 	%r191, %r164;
	mov.b32 	%r192, %r164;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r189, %r190, %r191, %r192 }, { %r165, %r166, %r167, %r168 }, { %r169, %r170 }, { %r189, %r190, %r191, %r192 };
	// end inline asm
	mov.b32 	%r193, %r164;
	mov.b32 	%r194, %r164;
	mov.b32 	%r195, %r164;
	mov.b32 	%r196, %r164;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r193, %r194, %r195, %r196 }, { %r165, %r166, %r167, %r168 }, { %r175, %r176 }, { %r193, %r194, %r195, %r196 };
	// end inline asm
	mov.b32 	%r201, %r164;
	mov.b32 	%r202, %r164;
	mov.b32 	%r203, %r164;
	mov.b32 	%r204, %r164;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r201, %r202, %r203, %r204 }, { %r165, %r166, %r167, %r168 }, { %r177, %r178 }, { %r201, %r202, %r203, %r204 };
	// end inline asm
	mov.b32 	%r205, %r164;
	mov.b32 	%r206, %r164;
	mov.b32 	%r207, %r164;
	mov.b32 	%r208, %r164;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r205, %r206, %r207, %r208 }, { %r165, %r166, %r167, %r168 }, { %r179, %r180 }, { %r205, %r206, %r207, %r208 };
	// end inline asm
	mov.b32 	%r209, %r164;
	mov.b32 	%r210, %r164;
	mov.b32 	%r211, %r164;
	mov.b32 	%r212, %r164;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r209, %r210, %r211, %r212 }, { %r171, %r172, %r173, %r174 }, { %r169, %r170 }, { %r209, %r210, %r211, %r212 };
	// end inline asm
	mov.b32 	%r215, %r164;
	mov.b32 	%r216, %r164;
	mov.b32 	%r217, %r164;
	mov.b32 	%r218, %r164;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r215, %r216, %r217, %r218 }, { %r171, %r172, %r173, %r174 }, { %r175, %r176 }, { %r215, %r216, %r217, %r218 };
	// end inline asm
	mov.b32 	%r225, %r164;
	mov.b32 	%r226, %r164;
	mov.b32 	%r227, %r164;
	mov.b32 	%r228, %r164;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r225, %r226, %r227, %r228 }, { %r171, %r172, %r173, %r174 }, { %r177, %r178 }, { %r225, %r226, %r227, %r228 };
	// end inline asm
	mov.b32 	%r231, %r164;
	mov.b32 	%r232, %r164;
	mov.b32 	%r233, %r164;
	mov.b32 	%r234, %r164;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r231, %r232, %r233, %r234 }, { %r171, %r172, %r173, %r174 }, { %r179, %r180 }, { %r231, %r232, %r233, %r234 };
	// end inline asm
	mov.b32 	%r237, %r164;
	mov.b32 	%r238, %r164;
	mov.b32 	%r239, %r164;
	mov.b32 	%r240, %r164;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r237, %r238, %r239, %r240 }, { %r181, %r182, %r183, %r184 }, { %r169, %r170 }, { %r237, %r238, %r239, %r240 };
	// end inline asm
	mov.b32 	%r241, %r164;
	mov.b32 	%r242, %r164;
	mov.b32 	%r243, %r164;
	mov.b32 	%r244, %r164;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r241, %r242, %r243, %r244 }, { %r181, %r182, %r183, %r184 }, { %r175, %r176 }, { %r241, %r242, %r243, %r244 };
	// end inline asm
	mov.b32 	%r249, %r164;
	mov.b32 	%r250, %r164;
	mov.b32 	%r251, %r164;
	mov.b32 	%r252, %r164;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r249, %r250, %r251, %r252 }, { %r181, %r182, %r183, %r184 }, { %r177, %r178 }, { %r249, %r250, %r251, %r252 };
	// end inline asm
	mov.b32 	%r253, %r164;
	mov.b32 	%r254, %r164;
	mov.b32 	%r255, %r164;
	mov.b32 	%r256, %r164;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r253, %r254, %r255, %r256 }, { %r181, %r182, %r183, %r184 }, { %r179, %r180 }, { %r253, %r254, %r255, %r256 };
	// end inline asm
	mov.b32 	%r257, %r164;
	mov.b32 	%r258, %r164;
	mov.b32 	%r259, %r164;
	mov.b32 	%r260, %r164;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r257, %r258, %r259, %r260 }, { %r185, %r186, %r187, %r188 }, { %r169, %r170 }, { %r257, %r258, %r259, %r260 };
	// end inline asm
	mov.b32 	%r261, %r164;
	mov.b32 	%r262, %r164;
	mov.b32 	%r263, %r164;
	mov.b32 	%r264, %r164;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r261, %r262, %r263, %r264 }, { %r185, %r186, %r187, %r188 }, { %r175, %r176 }, { %r261, %r262, %r263, %r264 };
	// end inline asm
	mov.b32 	%r269, %r164;
	mov.b32 	%r270, %r164;
	mov.b32 	%r271, %r164;
	mov.b32 	%r272, %r164;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r269, %r270, %r271, %r272 }, { %r185, %r186, %r187, %r188 }, { %r177, %r178 }, { %r269, %r270, %r271, %r272 };
	// end inline asm
	mov.b32 	%r273, %r164;
	mov.b32 	%r274, %r164;
	mov.b32 	%r275, %r164;
	mov.b32 	%r276, %r164;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r273, %r274, %r275, %r276 }, { %r185, %r186, %r187, %r188 }, { %r179, %r180 }, { %r273, %r274, %r275, %r276 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r189, %r190, %r191, %r192 }, { %r197, %r198, %r199, %r200 }, { %r213, %r214 }, { %r189, %r190, %r191, %r192 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r193, %r194, %r195, %r196 }, { %r197, %r198, %r199, %r200 }, { %r223, %r224 }, { %r193, %r194, %r195, %r196 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r201, %r202, %r203, %r204 }, { %r197, %r198, %r199, %r200 }, { %r229, %r230 }, { %r201, %r202, %r203, %r204 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r205, %r206, %r207, %r208 }, { %r197, %r198, %r199, %r200 }, { %r235, %r236 }, { %r205, %r206, %r207, %r208 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r209, %r210, %r211, %r212 }, { %r219, %r220, %r221, %r222 }, { %r213, %r214 }, { %r209, %r210, %r211, %r212 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r215, %r216, %r217, %r218 }, { %r219, %r220, %r221, %r222 }, { %r223, %r224 }, { %r215, %r216, %r217, %r218 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r225, %r226, %r227, %r228 }, { %r219, %r220, %r221, %r222 }, { %r229, %r230 }, { %r225, %r226, %r227, %r228 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r231, %r232, %r233, %r234 }, { %r219, %r220, %r221, %r222 }, { %r235, %r236 }, { %r231, %r232, %r233, %r234 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r237, %r238, %r239, %r240 }, { %r245, %r246, %r247, %r248 }, { %r213, %r214 }, { %r237, %r238, %r239, %r240 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r241, %r242, %r243, %r244 }, { %r245, %r246, %r247, %r248 }, { %r223, %r224 }, { %r241, %r242, %r243, %r244 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r249, %r250, %r251, %r252 }, { %r245, %r246, %r247, %r248 }, { %r229, %r230 }, { %r249, %r250, %r251, %r252 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r253, %r254, %r255, %r256 }, { %r245, %r246, %r247, %r248 }, { %r235, %r236 }, { %r253, %r254, %r255, %r256 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r257, %r258, %r259, %r260 }, { %r265, %r266, %r267, %r268 }, { %r213, %r214 }, { %r257, %r258, %r259, %r260 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r261, %r262, %r263, %r264 }, { %r265, %r266, %r267, %r268 }, { %r223, %r224 }, { %r261, %r262, %r263, %r264 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r269, %r270, %r271, %r272 }, { %r265, %r266, %r267, %r268 }, { %r229, %r230 }, { %r269, %r270, %r271, %r272 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r273, %r274, %r275, %r276 }, { %r265, %r266, %r267, %r268 }, { %r235, %r236 }, { %r273, %r274, %r275, %r276 };
	// end inline asm
	.loc	1 832 75                        // sk04_fa_o_w4a8.py:832:75
	prmt.b32 	%r277, %r745, %r746, 0x7531U;
	prmt.b32 	%r278, %r747, %r748, 0x7531U;
	prmt.b32 	%r279, %r762, %r763, 0x7531U;
	prmt.b32 	%r280, %r764, %r765, 0x7531U;
	prmt.b32 	%r301, %r779, %r780, 0x7531U;
	prmt.b32 	%r302, %r781, %r782, 0x7531U;
	prmt.b32 	%r303, %r796, %r797, 0x7531U;
	prmt.b32 	%r304, %r798, %r799, 0x7531U;
	prmt.b32 	%r283, %r749, %r750, 0x7531U;
	prmt.b32 	%r284, %r751, %r752, 0x7531U;
	prmt.b32 	%r285, %r766, %r767, 0x7531U;
	prmt.b32 	%r286, %r768, %r769, 0x7531U;
	prmt.b32 	%r307, %r783, %r784, 0x7531U;
	prmt.b32 	%r308, %r785, %r786, 0x7531U;
	prmt.b32 	%r309, %r800, %r801, 0x7531U;
	prmt.b32 	%r310, %r802, %r803, 0x7531U;
	prmt.b32 	%r293, %r753, %r754, 0x7531U;
	prmt.b32 	%r294, %r755, %r756, 0x7531U;
	prmt.b32 	%r295, %r770, %r771, 0x7531U;
	prmt.b32 	%r296, %r772, %r773, 0x7531U;
	prmt.b32 	%r317, %r787, %r788, 0x7531U;
	prmt.b32 	%r318, %r789, %r790, 0x7531U;
	prmt.b32 	%r319, %r804, %r805, 0x7531U;
	prmt.b32 	%r320, %r806, %r807, 0x7531U;
	prmt.b32 	%r297, %r757, %r758, 0x7531U;
	prmt.b32 	%r298, %r759, %r760, 0x7531U;
	prmt.b32 	%r299, %r774, %r775, 0x7531U;
	prmt.b32 	%r300, %r776, %r777, 0x7531U;
	prmt.b32 	%r321, %r791, %r792, 0x7531U;
	prmt.b32 	%r322, %r793, %r794, 0x7531U;
	prmt.b32 	%r323, %r808, %r809, 0x7531U;
	prmt.b32 	%r324, %r810, %r811, 0x7531U;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r189, %r190, %r191, %r192 }, { %r277, %r278, %r279, %r280 }, { %r281, %r282 }, { %r189, %r190, %r191, %r192 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r193, %r194, %r195, %r196 }, { %r277, %r278, %r279, %r280 }, { %r287, %r288 }, { %r193, %r194, %r195, %r196 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r201, %r202, %r203, %r204 }, { %r277, %r278, %r279, %r280 }, { %r289, %r290 }, { %r201, %r202, %r203, %r204 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r205, %r206, %r207, %r208 }, { %r277, %r278, %r279, %r280 }, { %r291, %r292 }, { %r205, %r206, %r207, %r208 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r209, %r210, %r211, %r212 }, { %r283, %r284, %r285, %r286 }, { %r281, %r282 }, { %r209, %r210, %r211, %r212 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r215, %r216, %r217, %r218 }, { %r283, %r284, %r285, %r286 }, { %r287, %r288 }, { %r215, %r216, %r217, %r218 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r225, %r226, %r227, %r228 }, { %r283, %r284, %r285, %r286 }, { %r289, %r290 }, { %r225, %r226, %r227, %r228 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r231, %r232, %r233, %r234 }, { %r283, %r284, %r285, %r286 }, { %r291, %r292 }, { %r231, %r232, %r233, %r234 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r237, %r238, %r239, %r240 }, { %r293, %r294, %r295, %r296 }, { %r281, %r282 }, { %r237, %r238, %r239, %r240 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r241, %r242, %r243, %r244 }, { %r293, %r294, %r295, %r296 }, { %r287, %r288 }, { %r241, %r242, %r243, %r244 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r249, %r250, %r251, %r252 }, { %r293, %r294, %r295, %r296 }, { %r289, %r290 }, { %r249, %r250, %r251, %r252 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r253, %r254, %r255, %r256 }, { %r293, %r294, %r295, %r296 }, { %r291, %r292 }, { %r253, %r254, %r255, %r256 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r257, %r258, %r259, %r260 }, { %r297, %r298, %r299, %r300 }, { %r281, %r282 }, { %r257, %r258, %r259, %r260 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r261, %r262, %r263, %r264 }, { %r297, %r298, %r299, %r300 }, { %r287, %r288 }, { %r261, %r262, %r263, %r264 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r269, %r270, %r271, %r272 }, { %r297, %r298, %r299, %r300 }, { %r289, %r290 }, { %r269, %r270, %r271, %r272 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r273, %r274, %r275, %r276 }, { %r297, %r298, %r299, %r300 }, { %r291, %r292 }, { %r273, %r274, %r275, %r276 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r189, %r190, %r191, %r192 }, { %r301, %r302, %r303, %r304 }, { %r305, %r306 }, { %r189, %r190, %r191, %r192 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r193, %r194, %r195, %r196 }, { %r301, %r302, %r303, %r304 }, { %r311, %r312 }, { %r193, %r194, %r195, %r196 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r201, %r202, %r203, %r204 }, { %r301, %r302, %r303, %r304 }, { %r313, %r314 }, { %r201, %r202, %r203, %r204 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r205, %r206, %r207, %r208 }, { %r301, %r302, %r303, %r304 }, { %r315, %r316 }, { %r205, %r206, %r207, %r208 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r209, %r210, %r211, %r212 }, { %r307, %r308, %r309, %r310 }, { %r305, %r306 }, { %r209, %r210, %r211, %r212 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r215, %r216, %r217, %r218 }, { %r307, %r308, %r309, %r310 }, { %r311, %r312 }, { %r215, %r216, %r217, %r218 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r225, %r226, %r227, %r228 }, { %r307, %r308, %r309, %r310 }, { %r313, %r314 }, { %r225, %r226, %r227, %r228 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r231, %r232, %r233, %r234 }, { %r307, %r308, %r309, %r310 }, { %r315, %r316 }, { %r231, %r232, %r233, %r234 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r237, %r238, %r239, %r240 }, { %r317, %r318, %r319, %r320 }, { %r305, %r306 }, { %r237, %r238, %r239, %r240 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r241, %r242, %r243, %r244 }, { %r317, %r318, %r319, %r320 }, { %r311, %r312 }, { %r241, %r242, %r243, %r244 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r249, %r250, %r251, %r252 }, { %r317, %r318, %r319, %r320 }, { %r313, %r314 }, { %r249, %r250, %r251, %r252 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r253, %r254, %r255, %r256 }, { %r317, %r318, %r319, %r320 }, { %r315, %r316 }, { %r253, %r254, %r255, %r256 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r257, %r258, %r259, %r260 }, { %r321, %r322, %r323, %r324 }, { %r305, %r306 }, { %r257, %r258, %r259, %r260 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r261, %r262, %r263, %r264 }, { %r321, %r322, %r323, %r324 }, { %r311, %r312 }, { %r261, %r262, %r263, %r264 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r269, %r270, %r271, %r272 }, { %r321, %r322, %r323, %r324 }, { %r313, %r314 }, { %r269, %r270, %r271, %r272 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r273, %r274, %r275, %r276 }, { %r321, %r322, %r323, %r324 }, { %r315, %r316 }, { %r273, %r274, %r275, %r276 };
	// end inline asm
	.loc	1 833 59                        // sk04_fa_o_w4a8.py:833:59
	mad.wide.s32 	%rd86, %r1289, 4, %rd16;
	.loc	1 833 27                        // sk04_fa_o_w4a8.py:833:27
	// begin inline asm
	mov.u32 %r325, 0x0;
	ld.global.b32 { %r325 }, [ %rd86 + 0 ];
	// end inline asm
	.loc	1 834 40                        // sk04_fa_o_w4a8.py:834:40
	// begin inline asm
	st.shared.b32 [ %r326 + 0 ], %r325;
	// end inline asm
	bar.sync 	0;
	.loc	1 834 26                        // sk04_fa_o_w4a8.py:834:26
	cvt.rn.f32.s32 	%r812, %r189;
	cvt.rn.f32.s32 	%r813, %r190;
	cvt.rn.f32.s32 	%r814, %r191;
	cvt.rn.f32.s32 	%r815, %r192;
	cvt.rn.f32.s32 	%r816, %r193;
	cvt.rn.f32.s32 	%r817, %r194;
	cvt.rn.f32.s32 	%r818, %r195;
	cvt.rn.f32.s32 	%r819, %r196;
	cvt.rn.f32.s32 	%r820, %r201;
	cvt.rn.f32.s32 	%r821, %r202;
	cvt.rn.f32.s32 	%r822, %r203;
	cvt.rn.f32.s32 	%r823, %r204;
	cvt.rn.f32.s32 	%r824, %r205;
	cvt.rn.f32.s32 	%r825, %r206;
	cvt.rn.f32.s32 	%r826, %r207;
	cvt.rn.f32.s32 	%r827, %r208;
	cvt.rn.f32.s32 	%r828, %r209;
	cvt.rn.f32.s32 	%r829, %r210;
	cvt.rn.f32.s32 	%r830, %r211;
	cvt.rn.f32.s32 	%r831, %r212;
	cvt.rn.f32.s32 	%r832, %r215;
	cvt.rn.f32.s32 	%r833, %r216;
	cvt.rn.f32.s32 	%r834, %r217;
	cvt.rn.f32.s32 	%r835, %r218;
	cvt.rn.f32.s32 	%r836, %r225;
	cvt.rn.f32.s32 	%r837, %r226;
	cvt.rn.f32.s32 	%r838, %r227;
	cvt.rn.f32.s32 	%r839, %r228;
	cvt.rn.f32.s32 	%r840, %r231;
	cvt.rn.f32.s32 	%r841, %r232;
	cvt.rn.f32.s32 	%r842, %r233;
	cvt.rn.f32.s32 	%r843, %r234;
	cvt.rn.f32.s32 	%r844, %r237;
	cvt.rn.f32.s32 	%r845, %r238;
	cvt.rn.f32.s32 	%r846, %r239;
	cvt.rn.f32.s32 	%r847, %r240;
	cvt.rn.f32.s32 	%r848, %r241;
	cvt.rn.f32.s32 	%r849, %r242;
	cvt.rn.f32.s32 	%r850, %r243;
	cvt.rn.f32.s32 	%r851, %r244;
	cvt.rn.f32.s32 	%r852, %r249;
	cvt.rn.f32.s32 	%r853, %r250;
	cvt.rn.f32.s32 	%r854, %r251;
	cvt.rn.f32.s32 	%r855, %r252;
	cvt.rn.f32.s32 	%r856, %r253;
	cvt.rn.f32.s32 	%r857, %r254;
	cvt.rn.f32.s32 	%r858, %r255;
	cvt.rn.f32.s32 	%r859, %r256;
	cvt.rn.f32.s32 	%r860, %r257;
	cvt.rn.f32.s32 	%r861, %r258;
	cvt.rn.f32.s32 	%r862, %r259;
	cvt.rn.f32.s32 	%r863, %r260;
	cvt.rn.f32.s32 	%r864, %r261;
	cvt.rn.f32.s32 	%r865, %r262;
	cvt.rn.f32.s32 	%r866, %r263;
	cvt.rn.f32.s32 	%r867, %r264;
	cvt.rn.f32.s32 	%r868, %r269;
	cvt.rn.f32.s32 	%r869, %r270;
	cvt.rn.f32.s32 	%r870, %r271;
	cvt.rn.f32.s32 	%r871, %r272;
	cvt.rn.f32.s32 	%r872, %r273;
	cvt.rn.f32.s32 	%r873, %r274;
	cvt.rn.f32.s32 	%r874, %r275;
	cvt.rn.f32.s32 	%r875, %r276;
	.loc	1 834 40                        // sk04_fa_o_w4a8.py:834:40
	ld.shared.v2.b32 	{%r876, %r877}, [%r46+49152];
	ld.shared.v2.b32 	{%r878, %r879}, [%r46+49280];
	ld.shared.v2.b32 	{%r880, %r881}, [%r46+49408];
	ld.shared.v2.b32 	{%r882, %r883}, [%r46+49536];
	.loc	1 834 15                        // sk04_fa_o_w4a8.py:834:15
	fma.rn.f32 	%r1358, %r883, %r875, %r1358;
	fma.rn.f32 	%r1357, %r882, %r874, %r1357;
	fma.rn.f32 	%r1356, %r883, %r873, %r1356;
	fma.rn.f32 	%r1355, %r882, %r872, %r1355;
	fma.rn.f32 	%r1354, %r881, %r871, %r1354;
	fma.rn.f32 	%r1353, %r880, %r870, %r1353;
	fma.rn.f32 	%r1352, %r881, %r869, %r1352;
	fma.rn.f32 	%r1351, %r880, %r868, %r1351;
	fma.rn.f32 	%r1350, %r879, %r867, %r1350;
	fma.rn.f32 	%r1349, %r878, %r866, %r1349;
	fma.rn.f32 	%r1348, %r879, %r865, %r1348;
	fma.rn.f32 	%r1347, %r878, %r864, %r1347;
	fma.rn.f32 	%r1346, %r877, %r863, %r1346;
	fma.rn.f32 	%r1345, %r876, %r862, %r1345;
	fma.rn.f32 	%r1344, %r877, %r861, %r1344;
	fma.rn.f32 	%r1343, %r876, %r860, %r1343;
	fma.rn.f32 	%r1342, %r883, %r859, %r1342;
	fma.rn.f32 	%r1341, %r882, %r858, %r1341;
	fma.rn.f32 	%r1340, %r883, %r857, %r1340;
	fma.rn.f32 	%r1339, %r882, %r856, %r1339;
	fma.rn.f32 	%r1338, %r881, %r855, %r1338;
	fma.rn.f32 	%r1337, %r880, %r854, %r1337;
	fma.rn.f32 	%r1336, %r881, %r853, %r1336;
	fma.rn.f32 	%r1335, %r880, %r852, %r1335;
	fma.rn.f32 	%r1334, %r879, %r851, %r1334;
	fma.rn.f32 	%r1333, %r878, %r850, %r1333;
	fma.rn.f32 	%r1332, %r879, %r849, %r1332;
	fma.rn.f32 	%r1331, %r878, %r848, %r1331;
	fma.rn.f32 	%r1330, %r877, %r847, %r1330;
	fma.rn.f32 	%r1329, %r876, %r846, %r1329;
	fma.rn.f32 	%r1328, %r877, %r845, %r1328;
	fma.rn.f32 	%r1327, %r876, %r844, %r1327;
	fma.rn.f32 	%r1326, %r883, %r843, %r1326;
	fma.rn.f32 	%r1325, %r882, %r842, %r1325;
	fma.rn.f32 	%r1324, %r883, %r841, %r1324;
	fma.rn.f32 	%r1323, %r882, %r840, %r1323;
	fma.rn.f32 	%r1322, %r881, %r839, %r1322;
	fma.rn.f32 	%r1321, %r880, %r838, %r1321;
	fma.rn.f32 	%r1320, %r881, %r837, %r1320;
	fma.rn.f32 	%r1319, %r880, %r836, %r1319;
	fma.rn.f32 	%r1318, %r879, %r835, %r1318;
	fma.rn.f32 	%r1317, %r878, %r834, %r1317;
	fma.rn.f32 	%r1316, %r879, %r833, %r1316;
	fma.rn.f32 	%r1315, %r878, %r832, %r1315;
	fma.rn.f32 	%r1314, %r877, %r831, %r1314;
	fma.rn.f32 	%r1313, %r876, %r830, %r1313;
	fma.rn.f32 	%r1312, %r877, %r829, %r1312;
	fma.rn.f32 	%r1311, %r876, %r828, %r1311;
	fma.rn.f32 	%r1310, %r883, %r827, %r1310;
	fma.rn.f32 	%r1309, %r882, %r826, %r1309;
	fma.rn.f32 	%r1308, %r883, %r825, %r1308;
	fma.rn.f32 	%r1307, %r882, %r824, %r1307;
	fma.rn.f32 	%r1306, %r881, %r823, %r1306;
	fma.rn.f32 	%r1305, %r880, %r822, %r1305;
	fma.rn.f32 	%r1304, %r881, %r821, %r1304;
	fma.rn.f32 	%r1303, %r880, %r820, %r1303;
	fma.rn.f32 	%r1302, %r879, %r819, %r1302;
	fma.rn.f32 	%r1301, %r878, %r818, %r1301;
	fma.rn.f32 	%r1300, %r879, %r817, %r1300;
	fma.rn.f32 	%r1299, %r878, %r816, %r1299;
	fma.rn.f32 	%r1298, %r877, %r815, %r1298;
	fma.rn.f32 	%r1297, %r876, %r814, %r1297;
	fma.rn.f32 	%r1296, %r877, %r813, %r1296;
	fma.rn.f32 	%r1295, %r876, %r812, %r1295;
	.loc	1 835 18                        // sk04_fa_o_w4a8.py:835:18
	add.s64 	%rd95, %rd22, %rd209;
	add.s64 	%rd96, %rd21, %rd209;
	add.s64 	%rd97, %rd20, %rd209;
	.loc	1 836 18                        // sk04_fa_o_w4a8.py:836:18
	add.s64 	%rd98, %rd19, %rd209;
	add.s64 	%rd87, %rd208, %rd6;
	add.s64 	%rd91, %rd208, %rd7;
	add.s64 	%rd88, %rd208, %rd8;
	add.s64 	%rd92, %rd208, %rd9;
	add.s64 	%rd89, %rd208, %rd10;
	add.s64 	%rd93, %rd208, %rd11;
	add.s64 	%rd90, %rd208, %rd12;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	add.s64 	%rd94, %rd208, %rd13;
	add.s32 	%r884, %r1291, 1;
	setp.gt.s32 	%p6, %r884, 2;
	selp.b32 	%r1291, 0, %r884, %p6;
	.loc	1 827 25                        // sk04_fa_o_w4a8.py:827:25
	shl.b32 	%r885, %r1291, 13;
	add.s32 	%r886, %r147, %r885;
	add.s32 	%r327, %r54, %r885;
	selp.b32 	%r328, 8, 0, %p4;
	// begin inline asm
	cp.async.ca.shared.global [ %r327 + 0 ], [ %rd87 + 0 ], 0x8, %r328;
	// end inline asm
	add.s32 	%r329, %r327, 2048;
	// begin inline asm
	cp.async.ca.shared.global [ %r329 + 0 ], [ %rd88 + 0 ], 0x8, %r328;
	// end inline asm
	add.s32 	%r330, %r327, 4096;
	// begin inline asm
	cp.async.ca.shared.global [ %r330 + 0 ], [ %rd89 + 0 ], 0x8, %r328;
	// end inline asm
	add.s32 	%r331, %r327, 6144;
	// begin inline asm
	cp.async.ca.shared.global [ %r331 + 0 ], [ %rd90 + 0 ], 0x8, %r328;
	// end inline asm
	add.s32 	%r887, %r886, %r23;
	add.s32 	%r332, %r887, 1024;
	// begin inline asm
	cp.async.ca.shared.global [ %r332 + 0 ], [ %rd91 + 0 ], 0x8, %r328;
	// end inline asm
	add.s32 	%r333, %r887, 3072;
	// begin inline asm
	cp.async.ca.shared.global [ %r333 + 0 ], [ %rd92 + 0 ], 0x8, %r328;
	// end inline asm
	add.s32 	%r334, %r887, 5120;
	// begin inline asm
	cp.async.ca.shared.global [ %r334 + 0 ], [ %rd93 + 0 ], 0x8, %r328;
	// end inline asm
	add.s32 	%r335, %r887, 7168;
	// begin inline asm
	cp.async.ca.shared.global [ %r335 + 0 ], [ %rd94 + 0 ], 0x8, %r328;
	// end inline asm
	cp.async.commit_group;
	.loc	1 831 52                        // sk04_fa_o_w4a8.py:831:52
	add.s32 	%r888, %r25, %r885;
	add.s32 	%r336, %r888, 24576;
	selp.b32 	%r337, 16, 0, %p4;
	// begin inline asm
	cp.async.cg.shared.global [ %r336 + 0 ], [ %rd95 + 0 ], 0x10, %r337;
	// end inline asm
	add.s32 	%r338, %r888, 26624;
	// begin inline asm
	cp.async.cg.shared.global [ %r338 + 0 ], [ %rd96 + 0 ], 0x10, %r337;
	// end inline asm
	add.s32 	%r339, %r888, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r339 + 0 ], [ %rd97 + 0 ], 0x10, %r337;
	// end inline asm
	add.s32 	%r340, %r888, 30720;
	// begin inline asm
	cp.async.cg.shared.global [ %r340 + 0 ], [ %rd98 + 0 ], 0x10, %r337;
	// end inline asm
	cp.async.commit_group;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	add.s64 	%rd210, %rd210, 1;
	add.s64 	%rd209, %rd209, 128;
	add.s32 	%r1289, %r1289, %r53;
	add.s64 	%rd208, %rd208, %rd15;
	setp.ne.b64 	%p7, %rd18, %rd209;
	@%p7 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_4;
$L__BB0_1:                              // %.._crit_edge_crit_edge
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	shr.u32 	%r1294, %r2, 2;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	and.b32 	%r1293, %r2, 3;
	and.b32 	%r1292, %r2, 96;
	mov.b32 	%r1295, 0f00000000;
	mov.b32 	%r1296, %r1295;
	mov.b32 	%r1297, %r1295;
	mov.b32 	%r1298, %r1295;
	mov.b32 	%r1299, %r1295;
	mov.b32 	%r1300, %r1295;
	mov.b32 	%r1301, %r1295;
	mov.b32 	%r1302, %r1295;
	mov.b32 	%r1303, %r1295;
	mov.b32 	%r1304, %r1295;
	mov.b32 	%r1305, %r1295;
	mov.b32 	%r1306, %r1295;
	mov.b32 	%r1307, %r1295;
	mov.b32 	%r1308, %r1295;
	mov.b32 	%r1309, %r1295;
	mov.b32 	%r1310, %r1295;
	mov.b32 	%r1311, %r1295;
	mov.b32 	%r1312, %r1295;
	mov.b32 	%r1313, %r1295;
	mov.b32 	%r1314, %r1295;
	mov.b32 	%r1315, %r1295;
	mov.b32 	%r1316, %r1295;
	mov.b32 	%r1317, %r1295;
	mov.b32 	%r1318, %r1295;
	mov.b32 	%r1319, %r1295;
	mov.b32 	%r1320, %r1295;
	mov.b32 	%r1321, %r1295;
	mov.b32 	%r1322, %r1295;
	mov.b32 	%r1323, %r1295;
	mov.b32 	%r1324, %r1295;
	mov.b32 	%r1325, %r1295;
	mov.b32 	%r1326, %r1295;
	mov.b32 	%r1327, %r1295;
	mov.b32 	%r1328, %r1295;
	mov.b32 	%r1329, %r1295;
	mov.b32 	%r1330, %r1295;
	mov.b32 	%r1331, %r1295;
	mov.b32 	%r1332, %r1295;
	mov.b32 	%r1333, %r1295;
	mov.b32 	%r1334, %r1295;
	mov.b32 	%r1335, %r1295;
	mov.b32 	%r1336, %r1295;
	mov.b32 	%r1337, %r1295;
	mov.b32 	%r1338, %r1295;
	mov.b32 	%r1339, %r1295;
	mov.b32 	%r1340, %r1295;
	mov.b32 	%r1341, %r1295;
	mov.b32 	%r1342, %r1295;
	mov.b32 	%r1343, %r1295;
	mov.b32 	%r1344, %r1295;
	mov.b32 	%r1345, %r1295;
	mov.b32 	%r1346, %r1295;
	mov.b32 	%r1347, %r1295;
	mov.b32 	%r1348, %r1295;
	mov.b32 	%r1349, %r1295;
	mov.b32 	%r1350, %r1295;
	mov.b32 	%r1351, %r1295;
	mov.b32 	%r1352, %r1295;
	mov.b32 	%r1353, %r1295;
	mov.b32 	%r1354, %r1295;
	mov.b32 	%r1355, %r1295;
	mov.b32 	%r1356, %r1295;
	mov.b32 	%r1357, %r1295;
	mov.b32 	%r1358, %r1295;
$L__BB0_4:                              // %._crit_edge
	.loc	1 0 15                          // sk04_fa_o_w4a8.py:0:15
	cvt.u32.u64 	%r999, %rd5;
	.loc	1 817 45                        // sk04_fa_o_w4a8.py:817:45
	or.b32 	%r1000, %r17, %r999;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r1001, %r1000, 15;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r1002, %r1001, %r48;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r1003, %r1000, 14;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r1004, %r1003, %r48;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r1005, %r1000, 13;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r1006, %r1005, %r48;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r1007, %r1000, 12;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r1008, %r1007, %r48;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r1009, %r1000, 11;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r1010, %r1009, %r48;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r1011, %r1000, 10;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r1012, %r1011, %r48;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r1013, %r1000, 9;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r1014, %r1013, %r48;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r1015, %r1000, 8;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r1016, %r1015, %r48;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r1017, %r1000, 7;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r1018, %r1017, %r48;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r1019, %r1000, 6;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r1020, %r1019, %r48;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r1021, %r1000, 5;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r1022, %r1021, %r48;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r1023, %r1000, 4;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r1024, %r1023, %r48;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r1025, %r1000, 3;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r1026, %r1025, %r48;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r1027, %r1000, 2;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r1028, %r1027, %r48;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r1029, %r1000, 1;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r1030, %r1029, %r48;
	rem.s32 	%r1031, %r1000, %r48;
	.loc	1 816 45                        // sk04_fa_o_w4a8.py:816:45
	shr.u32 	%r1032, %r12, 2;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1033, %r1032, %r1;
	or.b32 	%r1034, %r1033, 56;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1035, %r1034, %r47;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1036, %r1033, 48;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1037, %r1036, %r47;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1038, %r1033, 40;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1039, %r1038, %r47;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1040, %r1033, 32;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1041, %r1040, %r47;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1042, %r1033, 24;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1043, %r1042, %r47;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1044, %r1033, 16;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1045, %r1044, %r47;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1046, %r1033, 8;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1047, %r1046, %r47;
	rem.s32 	%r1048, %r1033, %r47;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1049, %r1, %r10;
	or.b32 	%r1050, %r1, %r9;
	or.b32 	%r1051, %r1, %r8;
	or.b32 	%r1052, %r1, %r7;
	or.b32 	%r1053, %r1, %r6;
	or.b32 	%r1054, %r1, %r5;
	or.b32 	%r1055, %r1, %r4;
	or.b32 	%r1056, %r1, %r3;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 838 38                        // sk04_fa_o_w4a8.py:838:38
	mad.wide.s32 	%rd99, %r1048, 4, %rd27;
	mad.wide.s32 	%rd100, %r1047, 4, %rd27;
	mad.wide.s32 	%rd101, %r1045, 4, %rd27;
	mad.wide.s32 	%rd102, %r1043, 4, %rd27;
	mad.wide.s32 	%rd103, %r1041, 4, %rd27;
	mad.wide.s32 	%rd104, %r1039, 4, %rd27;
	mad.wide.s32 	%rd105, %r1037, 4, %rd27;
	mad.wide.s32 	%rd106, %r1035, 4, %rd27;
	.loc	1 838 24                        // sk04_fa_o_w4a8.py:838:24
	// begin inline asm
	mov.u32 %r889, 0x0;
	ld.global.b32 { %r889 }, [ %rd99 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r890, 0x0;
	ld.global.b32 { %r890 }, [ %rd100 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r891, 0x0;
	ld.global.b32 { %r891 }, [ %rd101 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r892, 0x0;
	ld.global.b32 { %r892 }, [ %rd102 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r893, 0x0;
	ld.global.b32 { %r893 }, [ %rd103 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r894, 0x0;
	ld.global.b32 { %r894 }, [ %rd104 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r895, 0x0;
	ld.global.b32 { %r895 }, [ %rd105 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r896, 0x0;
	ld.global.b32 { %r896 }, [ %rd106 + 0 ];
	// end inline asm
	.loc	1 839 49                        // sk04_fa_o_w4a8.py:839:49
	mul.lo.s32 	%r1057, %r13, %r51;
	mul.lo.s32 	%r1058, %r14, %r51;
	mul.lo.s32 	%r1059, %r15, %r51;
	mul.lo.s32 	%r1060, %r16, %r51;
	.loc	1 839 31                        // sk04_fa_o_w4a8.py:839:31
	mad.wide.s32 	%rd179, %r1057, 2, %rd26;
	mad.wide.s32 	%rd180, %r1058, 2, %rd26;
	mad.wide.s32 	%rd181, %r1059, 2, %rd26;
	mad.wide.s32 	%rd182, %r1060, 2, %rd26;
	.loc	1 839 82                        // sk04_fa_o_w4a8.py:839:82
	mul.lo.s32 	%r1061, %r1031, %r52;
	mul.lo.s32 	%r1062, %r1030, %r52;
	mul.lo.s32 	%r1063, %r1028, %r52;
	mul.lo.s32 	%r1064, %r1026, %r52;
	mul.lo.s32 	%r1065, %r1024, %r52;
	mul.lo.s32 	%r1066, %r1022, %r52;
	mul.lo.s32 	%r1067, %r1020, %r52;
	mul.lo.s32 	%r1068, %r1018, %r52;
	mul.lo.s32 	%r1069, %r1016, %r52;
	mul.lo.s32 	%r1070, %r1014, %r52;
	mul.lo.s32 	%r1071, %r1012, %r52;
	mul.lo.s32 	%r1072, %r1010, %r52;
	mul.lo.s32 	%r1073, %r1008, %r52;
	mul.lo.s32 	%r1074, %r1006, %r52;
	mul.lo.s32 	%r1075, %r1004, %r52;
	mul.lo.s32 	%r1076, %r1002, %r52;
	.loc	1 839 64                        // sk04_fa_o_w4a8.py:839:64
	mul.wide.s32 	%rd183, %r1061, 2;
	add.s64 	%rd107, %rd179, %rd183;
	mul.wide.s32 	%rd184, %r1062, 2;
	add.s64 	%rd108, %rd179, %rd184;
	mul.wide.s32 	%rd185, %r1063, 2;
	add.s64 	%rd109, %rd179, %rd185;
	mul.wide.s32 	%rd186, %r1064, 2;
	add.s64 	%rd110, %rd179, %rd186;
	mul.wide.s32 	%rd187, %r1065, 2;
	add.s64 	%rd111, %rd179, %rd187;
	mul.wide.s32 	%rd188, %r1066, 2;
	add.s64 	%rd112, %rd179, %rd188;
	mul.wide.s32 	%rd189, %r1067, 2;
	add.s64 	%rd113, %rd179, %rd189;
	mul.wide.s32 	%rd190, %r1068, 2;
	add.s64 	%rd114, %rd179, %rd190;
	mul.wide.s32 	%rd191, %r1069, 2;
	add.s64 	%rd115, %rd179, %rd191;
	mul.wide.s32 	%rd192, %r1070, 2;
	add.s64 	%rd116, %rd179, %rd192;
	mul.wide.s32 	%rd193, %r1071, 2;
	add.s64 	%rd117, %rd179, %rd193;
	mul.wide.s32 	%rd194, %r1072, 2;
	add.s64 	%rd118, %rd179, %rd194;
	mul.wide.s32 	%rd195, %r1073, 2;
	add.s64 	%rd119, %rd179, %rd195;
	mul.wide.s32 	%rd196, %r1074, 2;
	add.s64 	%rd120, %rd179, %rd196;
	mul.wide.s32 	%rd197, %r1075, 2;
	add.s64 	%rd121, %rd179, %rd197;
	mul.wide.s32 	%rd198, %r1076, 2;
	add.s64 	%rd122, %rd179, %rd198;
	add.s64 	%rd123, %rd180, %rd183;
	add.s64 	%rd124, %rd180, %rd184;
	add.s64 	%rd125, %rd180, %rd185;
	add.s64 	%rd126, %rd180, %rd186;
	add.s64 	%rd127, %rd180, %rd187;
	add.s64 	%rd128, %rd180, %rd188;
	add.s64 	%rd129, %rd180, %rd189;
	add.s64 	%rd130, %rd180, %rd190;
	add.s64 	%rd131, %rd180, %rd191;
	add.s64 	%rd132, %rd180, %rd192;
	add.s64 	%rd133, %rd180, %rd193;
	add.s64 	%rd134, %rd180, %rd194;
	add.s64 	%rd135, %rd180, %rd195;
	add.s64 	%rd136, %rd180, %rd196;
	add.s64 	%rd137, %rd180, %rd197;
	add.s64 	%rd138, %rd180, %rd198;
	add.s64 	%rd139, %rd181, %rd183;
	add.s64 	%rd140, %rd181, %rd184;
	add.s64 	%rd141, %rd181, %rd185;
	add.s64 	%rd142, %rd181, %rd186;
	add.s64 	%rd143, %rd181, %rd187;
	add.s64 	%rd144, %rd181, %rd188;
	add.s64 	%rd145, %rd181, %rd189;
	add.s64 	%rd146, %rd181, %rd190;
	add.s64 	%rd147, %rd181, %rd191;
	add.s64 	%rd148, %rd181, %rd192;
	add.s64 	%rd149, %rd181, %rd193;
	add.s64 	%rd150, %rd181, %rd194;
	add.s64 	%rd151, %rd181, %rd195;
	add.s64 	%rd152, %rd181, %rd196;
	add.s64 	%rd153, %rd181, %rd197;
	add.s64 	%rd154, %rd181, %rd198;
	add.s64 	%rd155, %rd182, %rd183;
	add.s64 	%rd156, %rd182, %rd184;
	add.s64 	%rd157, %rd182, %rd185;
	add.s64 	%rd158, %rd182, %rd186;
	add.s64 	%rd159, %rd182, %rd187;
	add.s64 	%rd160, %rd182, %rd188;
	add.s64 	%rd161, %rd182, %rd189;
	add.s64 	%rd162, %rd182, %rd190;
	add.s64 	%rd163, %rd182, %rd191;
	add.s64 	%rd164, %rd182, %rd192;
	add.s64 	%rd165, %rd182, %rd193;
	add.s64 	%rd166, %rd182, %rd194;
	add.s64 	%rd167, %rd182, %rd195;
	add.s64 	%rd168, %rd182, %rd196;
	add.s64 	%rd169, %rd182, %rd197;
	add.s64 	%rd170, %rd182, %rd198;
	.loc	1 839 19                        // sk04_fa_o_w4a8.py:839:19
	// begin inline asm
	mov.u16 %rs321, 0x0;
	ld.global.b16 { %rs321 }, [ %rd107 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs322, 0x0;
	ld.global.b16 { %rs322 }, [ %rd108 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs323, 0x0;
	ld.global.b16 { %rs323 }, [ %rd109 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs324, 0x0;
	ld.global.b16 { %rs324 }, [ %rd110 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs325, 0x0;
	ld.global.b16 { %rs325 }, [ %rd111 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs326, 0x0;
	ld.global.b16 { %rs326 }, [ %rd112 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs327, 0x0;
	ld.global.b16 { %rs327 }, [ %rd113 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs328, 0x0;
	ld.global.b16 { %rs328 }, [ %rd114 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs329, 0x0;
	ld.global.b16 { %rs329 }, [ %rd115 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs330, 0x0;
	ld.global.b16 { %rs330 }, [ %rd116 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs331, 0x0;
	ld.global.b16 { %rs331 }, [ %rd117 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs332, 0x0;
	ld.global.b16 { %rs332 }, [ %rd118 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs333, 0x0;
	ld.global.b16 { %rs333 }, [ %rd119 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs334, 0x0;
	ld.global.b16 { %rs334 }, [ %rd120 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs335, 0x0;
	ld.global.b16 { %rs335 }, [ %rd121 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs336, 0x0;
	ld.global.b16 { %rs336 }, [ %rd122 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs337, 0x0;
	ld.global.b16 { %rs337 }, [ %rd123 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs338, 0x0;
	ld.global.b16 { %rs338 }, [ %rd124 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs339, 0x0;
	ld.global.b16 { %rs339 }, [ %rd125 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs340, 0x0;
	ld.global.b16 { %rs340 }, [ %rd126 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs341, 0x0;
	ld.global.b16 { %rs341 }, [ %rd127 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs342, 0x0;
	ld.global.b16 { %rs342 }, [ %rd128 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs343, 0x0;
	ld.global.b16 { %rs343 }, [ %rd129 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs344, 0x0;
	ld.global.b16 { %rs344 }, [ %rd130 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs345, 0x0;
	ld.global.b16 { %rs345 }, [ %rd131 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs346, 0x0;
	ld.global.b16 { %rs346 }, [ %rd132 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs347, 0x0;
	ld.global.b16 { %rs347 }, [ %rd133 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs348, 0x0;
	ld.global.b16 { %rs348 }, [ %rd134 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs349, 0x0;
	ld.global.b16 { %rs349 }, [ %rd135 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs350, 0x0;
	ld.global.b16 { %rs350 }, [ %rd136 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs351, 0x0;
	ld.global.b16 { %rs351 }, [ %rd137 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs352, 0x0;
	ld.global.b16 { %rs352 }, [ %rd138 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs353, 0x0;
	ld.global.b16 { %rs353 }, [ %rd139 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs354, 0x0;
	ld.global.b16 { %rs354 }, [ %rd140 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs355, 0x0;
	ld.global.b16 { %rs355 }, [ %rd141 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs356, 0x0;
	ld.global.b16 { %rs356 }, [ %rd142 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs357, 0x0;
	ld.global.b16 { %rs357 }, [ %rd143 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs358, 0x0;
	ld.global.b16 { %rs358 }, [ %rd144 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs359, 0x0;
	ld.global.b16 { %rs359 }, [ %rd145 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs360, 0x0;
	ld.global.b16 { %rs360 }, [ %rd146 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs361, 0x0;
	ld.global.b16 { %rs361 }, [ %rd147 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs362, 0x0;
	ld.global.b16 { %rs362 }, [ %rd148 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs363, 0x0;
	ld.global.b16 { %rs363 }, [ %rd149 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs364, 0x0;
	ld.global.b16 { %rs364 }, [ %rd150 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs365, 0x0;
	ld.global.b16 { %rs365 }, [ %rd151 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs366, 0x0;
	ld.global.b16 { %rs366 }, [ %rd152 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs367, 0x0;
	ld.global.b16 { %rs367 }, [ %rd153 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs368, 0x0;
	ld.global.b16 { %rs368 }, [ %rd154 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs369, 0x0;
	ld.global.b16 { %rs369 }, [ %rd155 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs370, 0x0;
	ld.global.b16 { %rs370 }, [ %rd156 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs371, 0x0;
	ld.global.b16 { %rs371 }, [ %rd157 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs372, 0x0;
	ld.global.b16 { %rs372 }, [ %rd158 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs373, 0x0;
	ld.global.b16 { %rs373 }, [ %rd159 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs374, 0x0;
	ld.global.b16 { %rs374 }, [ %rd160 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs375, 0x0;
	ld.global.b16 { %rs375 }, [ %rd161 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs376, 0x0;
	ld.global.b16 { %rs376 }, [ %rd162 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs377, 0x0;
	ld.global.b16 { %rs377 }, [ %rd163 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs378, 0x0;
	ld.global.b16 { %rs378 }, [ %rd164 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs379, 0x0;
	ld.global.b16 { %rs379 }, [ %rd165 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs380, 0x0;
	ld.global.b16 { %rs380 }, [ %rd166 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs381, 0x0;
	ld.global.b16 { %rs381 }, [ %rd167 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs382, 0x0;
	ld.global.b16 { %rs382 }, [ %rd168 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs383, 0x0;
	ld.global.b16 { %rs383 }, [ %rd169 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs384, 0x0;
	ld.global.b16 { %rs384 }, [ %rd170 + 0 ];
	// end inline asm
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	shl.b32 	%r1077, %r2, 6;
	and.b32 	%r1078, %r1077, 3584;
	shl.b32 	%r1079, %r11, 1;
	or.b32 	%r1080, %r1078, %r999;
	xor.b32 	%r1081, %r1080, %r1079;
	add.s32 	%r897, %r147, %r1081;
	mov.b32 	%r898, {%rs321, %rs322};
	mov.b32 	%r899, {%rs323, %rs324};
	mov.b32 	%r900, {%rs325, %rs326};
	mov.b32 	%r901, {%rs327, %rs328};
	// begin inline asm
	st.shared.v4.b32 [ %r897 + 0 ], { %r898, %r899, %r900, %r901 };
	// end inline asm
	add.s32 	%r902, %r897, 256;
	mov.b32 	%r903, {%rs329, %rs330};
	mov.b32 	%r904, {%rs331, %rs332};
	mov.b32 	%r905, {%rs333, %rs334};
	mov.b32 	%r906, {%rs335, %rs336};
	// begin inline asm
	st.shared.v4.b32 [ %r902 + 0 ], { %r903, %r904, %r905, %r906 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r1082, %r20, 9;
	shl.b32 	%r1083, %r18, 4;
	bfe.s32 	%r1084, %r2, 4, 1;
	and.b32 	%r1085, %r2, 16;
	shl.b32 	%r1086, %r1085, 1;
	shl.b32 	%r1087, %r2, 3;
	and.b32 	%r1088, %r1087, 256;
	and.b32 	%r1089, %r1294, 16;
	or.b32 	%r1090, %r1082, %r1088;
	xor.b32 	%r1091, %r1083, %r1086;
	xor.b32 	%r1092, %r1091, %r1089;
	or.b32 	%r1093, %r1092, %r1090;
	add.s32 	%r1094, %r147, %r1093;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1095, %r1096, %r1097, %r1098}, [%r1094];
	xor.b32 	%r1099, %r1093, 64;
	add.s32 	%r1100, %r147, %r1099;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1101, %r1102, %r1103, %r1104}, [%r1100];
	bar.sync 	0;
	mov.b32 	%r907, {%rs337, %rs338};
	mov.b32 	%r908, {%rs339, %rs340};
	mov.b32 	%r909, {%rs341, %rs342};
	mov.b32 	%r910, {%rs343, %rs344};
	// begin inline asm
	st.shared.v4.b32 [ %r897 + 0 ], { %r907, %r908, %r909, %r910 };
	// end inline asm
	mov.b32 	%r911, {%rs345, %rs346};
	mov.b32 	%r912, {%rs347, %rs348};
	mov.b32 	%r913, {%rs349, %rs350};
	mov.b32 	%r914, {%rs351, %rs352};
	// begin inline asm
	st.shared.v4.b32 [ %r902 + 0 ], { %r911, %r912, %r913, %r914 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1105, %r1106, %r1107, %r1108}, [%r1094];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1109, %r1110, %r1111, %r1112}, [%r1100];
	bar.sync 	0;
	mov.b32 	%r915, {%rs353, %rs354};
	mov.b32 	%r916, {%rs355, %rs356};
	mov.b32 	%r917, {%rs357, %rs358};
	mov.b32 	%r918, {%rs359, %rs360};
	// begin inline asm
	st.shared.v4.b32 [ %r897 + 0 ], { %r915, %r916, %r917, %r918 };
	// end inline asm
	mov.b32 	%r919, {%rs361, %rs362};
	mov.b32 	%r920, {%rs363, %rs364};
	mov.b32 	%r921, {%rs365, %rs366};
	mov.b32 	%r922, {%rs367, %rs368};
	// begin inline asm
	st.shared.v4.b32 [ %r902 + 0 ], { %r919, %r920, %r921, %r922 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1113, %r1114, %r1115, %r1116}, [%r1094];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1117, %r1118, %r1119, %r1120}, [%r1100];
	bar.sync 	0;
	mov.b32 	%r923, {%rs369, %rs370};
	mov.b32 	%r924, {%rs371, %rs372};
	mov.b32 	%r925, {%rs373, %rs374};
	mov.b32 	%r926, {%rs375, %rs376};
	// begin inline asm
	st.shared.v4.b32 [ %r897 + 0 ], { %r923, %r924, %r925, %r926 };
	// end inline asm
	mov.b32 	%r927, {%rs377, %rs378};
	mov.b32 	%r928, {%rs379, %rs380};
	mov.b32 	%r929, {%rs381, %rs382};
	mov.b32 	%r930, {%rs383, %rs384};
	// begin inline asm
	st.shared.v4.b32 [ %r902 + 0 ], { %r927, %r928, %r929, %r930 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1121, %r1122, %r1123, %r1124}, [%r1094];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1125, %r1126, %r1127, %r1128}, [%r1100];
	.loc	1 846 31                        // sk04_fa_o_w4a8.py:846:31
	setp.lt.s32 	%p16, %r1056, %r47;
	setp.lt.s32 	%p17, %r1055, %r47;
	setp.lt.s32 	%p18, %r1054, %r47;
	setp.lt.s32 	%p19, %r1053, %r47;
	setp.lt.s32 	%p20, %r1052, %r47;
	setp.lt.s32 	%p21, %r1051, %r47;
	setp.lt.s32 	%p22, %r1050, %r47;
	setp.lt.s32 	%p23, %r1049, %r47;
	.loc	1 846 54                        // sk04_fa_o_w4a8.py:846:54
	setp.lt.s32 	%p24, %r22, %r48;
	.loc	1 846 37                        // sk04_fa_o_w4a8.py:846:37
	and.pred 	%p8, %p16, %p24;
	and.pred 	%p9, %p17, %p24;
	and.pred 	%p10, %p18, %p24;
	and.pred 	%p11, %p19, %p24;
	and.pred 	%p12, %p20, %p24;
	and.pred 	%p13, %p21, %p24;
	and.pred 	%p14, %p22, %p24;
	and.pred 	%p15, %p23, %p24;
	.loc	1 844 35                        // sk04_fa_o_w4a8.py:844:35
	mul.lo.s32 	%r1129, %r1056, %r50;
	mul.lo.s32 	%r1130, %r1055, %r50;
	mul.lo.s32 	%r1131, %r1054, %r50;
	mul.lo.s32 	%r1132, %r1053, %r50;
	mul.lo.s32 	%r1133, %r1052, %r50;
	mul.lo.s32 	%r1134, %r1051, %r50;
	mul.lo.s32 	%r1135, %r1050, %r50;
	mul.lo.s32 	%r1136, %r1049, %r50;
	.loc	1 844 18                        // sk04_fa_o_w4a8.py:844:18
	mad.wide.s32 	%rd199, %r1129, 2, %rd25;
	mad.wide.s32 	%rd200, %r1130, 2, %rd25;
	mad.wide.s32 	%rd201, %r1131, 2, %rd25;
	mad.wide.s32 	%rd202, %r1132, 2, %rd25;
	mad.wide.s32 	%rd203, %r1133, 2, %rd25;
	mad.wide.s32 	%rd204, %r1134, 2, %rd25;
	mad.wide.s32 	%rd205, %r1135, 2, %rd25;
	mad.wide.s32 	%rd206, %r1136, 2, %rd25;
	.loc	1 844 50                        // sk04_fa_o_w4a8.py:844:50
	mul.wide.s32 	%rd207, %r22, 2;
	add.s64 	%rd171, %rd199, %rd207;
	add.s64 	%rd172, %rd200, %rd207;
	add.s64 	%rd173, %rd201, %rd207;
	add.s64 	%rd174, %rd202, %rd207;
	add.s64 	%rd175, %rd203, %rd207;
	add.s64 	%rd176, %rd204, %rd207;
	add.s64 	%rd177, %rd205, %rd207;
	add.s64 	%rd178, %rd206, %rd207;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs385, %rs386}, %r1095;
	cvt.f32.bf16 	%r1137, %rs386;
	cvt.f32.bf16 	%r1138, %rs385;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1139, %r1295, %r889, %r1138;
	fma.rn.f32 	%r1140, %r1296, %r889, %r1137;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r932, %r1140, %r1139;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs387, %rs388}, %r1096;
	cvt.f32.bf16 	%r1141, %rs388;
	cvt.f32.bf16 	%r1142, %rs387;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1143, %r1297, %r890, %r1142;
	fma.rn.f32 	%r1144, %r1298, %r890, %r1141;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r933, %r1144, %r1143;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs389, %rs390}, %r1097;
	cvt.f32.bf16 	%r1145, %rs390;
	cvt.f32.bf16 	%r1146, %rs389;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1147, %r1299, %r889, %r1146;
	fma.rn.f32 	%r1148, %r1300, %r889, %r1145;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r942, %r1148, %r1147;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs391, %rs392}, %r1098;
	cvt.f32.bf16 	%r1149, %rs392;
	cvt.f32.bf16 	%r1150, %rs391;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1151, %r1301, %r890, %r1150;
	fma.rn.f32 	%r1152, %r1302, %r890, %r1149;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r943, %r1152, %r1151;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs393, %rs394}, %r1101;
	cvt.f32.bf16 	%r1153, %rs394;
	cvt.f32.bf16 	%r1154, %rs393;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1155, %r1303, %r889, %r1154;
	fma.rn.f32 	%r1156, %r1304, %r889, %r1153;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r937, %r1156, %r1155;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs395, %rs396}, %r1102;
	cvt.f32.bf16 	%r1157, %rs396;
	cvt.f32.bf16 	%r1158, %rs395;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1159, %r1305, %r890, %r1158;
	fma.rn.f32 	%r1160, %r1306, %r890, %r1157;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r938, %r1160, %r1159;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs397, %rs398}, %r1103;
	cvt.f32.bf16 	%r1161, %rs398;
	cvt.f32.bf16 	%r1162, %rs397;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1163, %r1307, %r889, %r1162;
	fma.rn.f32 	%r1164, %r1308, %r889, %r1161;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r947, %r1164, %r1163;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs399, %rs400}, %r1104;
	cvt.f32.bf16 	%r1165, %rs400;
	cvt.f32.bf16 	%r1166, %rs399;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1167, %r1309, %r890, %r1166;
	fma.rn.f32 	%r1168, %r1310, %r890, %r1165;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r948, %r1168, %r1167;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs401, %rs402}, %r1105;
	cvt.f32.bf16 	%r1169, %rs402;
	cvt.f32.bf16 	%r1170, %rs401;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1171, %r1311, %r891, %r1170;
	fma.rn.f32 	%r1172, %r1312, %r891, %r1169;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r934, %r1172, %r1171;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs403, %rs404}, %r1106;
	cvt.f32.bf16 	%r1173, %rs404;
	cvt.f32.bf16 	%r1174, %rs403;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1175, %r1313, %r892, %r1174;
	fma.rn.f32 	%r1176, %r1314, %r892, %r1173;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r935, %r1176, %r1175;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs405, %rs406}, %r1107;
	cvt.f32.bf16 	%r1177, %rs406;
	cvt.f32.bf16 	%r1178, %rs405;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1179, %r1315, %r891, %r1178;
	fma.rn.f32 	%r1180, %r1316, %r891, %r1177;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r944, %r1180, %r1179;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs407, %rs408}, %r1108;
	cvt.f32.bf16 	%r1181, %rs408;
	cvt.f32.bf16 	%r1182, %rs407;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1183, %r1317, %r892, %r1182;
	fma.rn.f32 	%r1184, %r1318, %r892, %r1181;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r945, %r1184, %r1183;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs409, %rs410}, %r1109;
	cvt.f32.bf16 	%r1185, %rs410;
	cvt.f32.bf16 	%r1186, %rs409;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1187, %r1319, %r891, %r1186;
	fma.rn.f32 	%r1188, %r1320, %r891, %r1185;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r939, %r1188, %r1187;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs411, %rs412}, %r1110;
	cvt.f32.bf16 	%r1189, %rs412;
	cvt.f32.bf16 	%r1190, %rs411;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1191, %r1321, %r892, %r1190;
	fma.rn.f32 	%r1192, %r1322, %r892, %r1189;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r940, %r1192, %r1191;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs413, %rs414}, %r1111;
	cvt.f32.bf16 	%r1193, %rs414;
	cvt.f32.bf16 	%r1194, %rs413;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1195, %r1323, %r891, %r1194;
	fma.rn.f32 	%r1196, %r1324, %r891, %r1193;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r949, %r1196, %r1195;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs415, %rs416}, %r1112;
	cvt.f32.bf16 	%r1197, %rs416;
	cvt.f32.bf16 	%r1198, %rs415;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1199, %r1325, %r892, %r1198;
	fma.rn.f32 	%r1200, %r1326, %r892, %r1197;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r950, %r1200, %r1199;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs417, %rs418}, %r1113;
	cvt.f32.bf16 	%r1201, %rs418;
	cvt.f32.bf16 	%r1202, %rs417;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1203, %r1327, %r893, %r1202;
	fma.rn.f32 	%r1204, %r1328, %r893, %r1201;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r951, %r1204, %r1203;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs419, %rs420}, %r1114;
	cvt.f32.bf16 	%r1205, %rs420;
	cvt.f32.bf16 	%r1206, %rs419;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1207, %r1329, %r894, %r1206;
	fma.rn.f32 	%r1208, %r1330, %r894, %r1205;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r952, %r1208, %r1207;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs421, %rs422}, %r1115;
	cvt.f32.bf16 	%r1209, %rs422;
	cvt.f32.bf16 	%r1210, %rs421;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1211, %r1331, %r893, %r1210;
	fma.rn.f32 	%r1212, %r1332, %r893, %r1209;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r959, %r1212, %r1211;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs423, %rs424}, %r1116;
	cvt.f32.bf16 	%r1213, %rs424;
	cvt.f32.bf16 	%r1214, %rs423;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1215, %r1333, %r894, %r1214;
	fma.rn.f32 	%r1216, %r1334, %r894, %r1213;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r960, %r1216, %r1215;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs425, %rs426}, %r1117;
	cvt.f32.bf16 	%r1217, %rs426;
	cvt.f32.bf16 	%r1218, %rs425;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1219, %r1335, %r893, %r1218;
	fma.rn.f32 	%r1220, %r1336, %r893, %r1217;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r955, %r1220, %r1219;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs427, %rs428}, %r1118;
	cvt.f32.bf16 	%r1221, %rs428;
	cvt.f32.bf16 	%r1222, %rs427;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1223, %r1337, %r894, %r1222;
	fma.rn.f32 	%r1224, %r1338, %r894, %r1221;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r956, %r1224, %r1223;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs429, %rs430}, %r1119;
	cvt.f32.bf16 	%r1225, %rs430;
	cvt.f32.bf16 	%r1226, %rs429;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1227, %r1339, %r893, %r1226;
	fma.rn.f32 	%r1228, %r1340, %r893, %r1225;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r963, %r1228, %r1227;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs431, %rs432}, %r1120;
	cvt.f32.bf16 	%r1229, %rs432;
	cvt.f32.bf16 	%r1230, %rs431;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1231, %r1341, %r894, %r1230;
	fma.rn.f32 	%r1232, %r1342, %r894, %r1229;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r964, %r1232, %r1231;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs433, %rs434}, %r1121;
	cvt.f32.bf16 	%r1233, %rs434;
	cvt.f32.bf16 	%r1234, %rs433;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1235, %r1343, %r895, %r1234;
	fma.rn.f32 	%r1236, %r1344, %r895, %r1233;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r953, %r1236, %r1235;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs435, %rs436}, %r1122;
	cvt.f32.bf16 	%r1237, %rs436;
	cvt.f32.bf16 	%r1238, %rs435;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1239, %r1345, %r896, %r1238;
	fma.rn.f32 	%r1240, %r1346, %r896, %r1237;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r954, %r1240, %r1239;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs437, %rs438}, %r1123;
	cvt.f32.bf16 	%r1241, %rs438;
	cvt.f32.bf16 	%r1242, %rs437;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1243, %r1347, %r895, %r1242;
	fma.rn.f32 	%r1244, %r1348, %r895, %r1241;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r961, %r1244, %r1243;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs439, %rs440}, %r1124;
	cvt.f32.bf16 	%r1245, %rs440;
	cvt.f32.bf16 	%r1246, %rs439;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1247, %r1349, %r896, %r1246;
	fma.rn.f32 	%r1248, %r1350, %r896, %r1245;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r962, %r1248, %r1247;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs441, %rs442}, %r1125;
	cvt.f32.bf16 	%r1249, %rs442;
	cvt.f32.bf16 	%r1250, %rs441;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1251, %r1351, %r895, %r1250;
	fma.rn.f32 	%r1252, %r1352, %r895, %r1249;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r957, %r1252, %r1251;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs443, %rs444}, %r1126;
	cvt.f32.bf16 	%r1253, %rs444;
	cvt.f32.bf16 	%r1254, %rs443;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1255, %r1353, %r896, %r1254;
	fma.rn.f32 	%r1256, %r1354, %r896, %r1253;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r958, %r1256, %r1255;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs445, %rs446}, %r1127;
	cvt.f32.bf16 	%r1257, %rs446;
	cvt.f32.bf16 	%r1258, %rs445;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1259, %r1355, %r895, %r1258;
	fma.rn.f32 	%r1260, %r1356, %r895, %r1257;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r965, %r1260, %r1259;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs447, %rs448}, %r1128;
	cvt.f32.bf16 	%r1261, %rs448;
	cvt.f32.bf16 	%r1262, %rs447;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1263, %r1357, %r896, %r1262;
	fma.rn.f32 	%r1264, %r1358, %r896, %r1261;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r966, %r1264, %r1263;
	bar.sync 	0;
	shl.b32 	%r1265, %r1293, 11;
	shl.b32 	%r1266, %r1293, 5;
	shl.b32 	%r1267, %r24, 4;
	shr.u32 	%r1268, %r1292, 1;
	bfe.s32 	%r1269, %r2, 2, 1;
	and.b32 	%r1270, %r1269, 1040;
	or.b32 	%r1271, %r1266, %r1267;
	xor.b32 	%r1272, %r1270, %r1268;
	xor.b32 	%r1273, %r1272, %r1271;
	or.b32 	%r1274, %r1273, %r1265;
	add.s32 	%r931, %r147, %r1274;
	// begin inline asm
	st.shared.v4.b32 [ %r931 + 0 ], { %r932, %r933, %r934, %r935 };
	// end inline asm
	add.s32 	%r936, %r931, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r936 + 0 ], { %r937, %r938, %r939, %r940 };
	// end inline asm
	xor.b32 	%r1275, %r1274, 64;
	add.s32 	%r941, %r147, %r1275;
	// begin inline asm
	st.shared.v4.b32 [ %r941 + 0 ], { %r942, %r943, %r944, %r945 };
	// end inline asm
	add.s32 	%r946, %r941, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r946 + 0 ], { %r947, %r948, %r949, %r950 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r1276, %r1292, 2;
	and.b32 	%r1277, %r1077, 512;
	and.b32 	%r1278, %r1084, 1040;
	or.b32 	%r1279, %r999, %r1276;
	xor.b32 	%r1280, %r1279, %r1278;
	or.b32 	%r1281, %r1280, %r1277;
	add.s32 	%r1282, %r147, %r1281;
	ld.shared.v4.b32 	{%r967, %r971, %r975, %r979}, [%r1282];
	xor.b32 	%r1283, %r1281, 32;
	add.s32 	%r1284, %r147, %r1283;
	ld.shared.v4.b32 	{%r968, %r972, %r976, %r980}, [%r1284+2048];
	xor.b32 	%r1285, %r1281, 64;
	add.s32 	%r1286, %r147, %r1285;
	ld.shared.v4.b32 	{%r969, %r973, %r977, %r981}, [%r1286+4096];
	xor.b32 	%r1287, %r1281, 96;
	add.s32 	%r1288, %r147, %r1287;
	ld.shared.v4.b32 	{%r970, %r974, %r978, %r982}, [%r1288+6144];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r931 + 0 ], { %r951, %r952, %r953, %r954 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r936 + 0 ], { %r955, %r956, %r957, %r958 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r941 + 0 ], { %r959, %r960, %r961, %r962 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r946 + 0 ], { %r963, %r964, %r965, %r966 };
	// end inline asm
	bar.sync 	0;
	ld.shared.v4.b32 	{%r983, %r987, %r991, %r995}, [%r1282];
	ld.shared.v4.b32 	{%r984, %r988, %r992, %r996}, [%r1284+2048];
	ld.shared.v4.b32 	{%r985, %r989, %r993, %r997}, [%r1286+4096];
	ld.shared.v4.b32 	{%r986, %r990, %r994, %r998}, [%r1288+6144];
	.loc	1 845 8                         // sk04_fa_o_w4a8.py:845:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd171 + 0 ], { %r967, %r968, %r969, %r970 };
	// end inline asm
	// begin inline asm
	@%p9 st.global.v4.b32 [ %rd172 + 0 ], { %r971, %r972, %r973, %r974 };
	// end inline asm
	// begin inline asm
	@%p10 st.global.v4.b32 [ %rd173 + 0 ], { %r975, %r976, %r977, %r978 };
	// end inline asm
	// begin inline asm
	@%p11 st.global.v4.b32 [ %rd174 + 0 ], { %r979, %r980, %r981, %r982 };
	// end inline asm
	// begin inline asm
	@%p12 st.global.v4.b32 [ %rd175 + 0 ], { %r983, %r984, %r985, %r986 };
	// end inline asm
	// begin inline asm
	@%p13 st.global.v4.b32 [ %rd176 + 0 ], { %r987, %r988, %r989, %r990 };
	// end inline asm
	// begin inline asm
	@%p14 st.global.v4.b32 [ %rd177 + 0 ], { %r991, %r992, %r993, %r994 };
	// end inline asm
	// begin inline asm
	@%p15 st.global.v4.b32 [ %rd178 + 0 ], { %r995, %r996, %r997, %r998 };
	// end inline asm
	.loc	1 843 4                         // sk04_fa_o_w4a8.py:843:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk04_fa_o_w4a8.py"
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
.b8 5                                   // DW_FORM_data2
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 165                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x9e DW_TAG_compile_unit
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
.b8 52
.b8 95
.b8 102
.b8 97
.b8 95
.b8 111
.b8 95
.b8 119
.b8 52
.b8 97
.b8 56
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
.b8 2                                   // Abbrev [2] 0x47:0x19 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 52
.b8 95
.b8 102
.b8 97
.b8 95
.b8 111
.b8 95
.b8 119
.b8 52
.b8 97
.b8 56
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x60:0x48 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 71                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x75:0x19 DW_TAG_inlined_subroutine
.b32 71                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 40                                  // DW_AT_call_line
.b8 3
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x8e:0x19 DW_TAG_inlined_subroutine
.b32 71                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 41                                  // DW_AT_call_line
.b8 3
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_3 = _Nativo(
    "sk04_fa_o_w4a8/tile64x128x128_shift0_abi15",
    _PTX_3, "_sk04_fa_o_w4a8_kernel",
    warps=4, shared=49664,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 16, 17],
    horneado={10: 1, 12: 1, 14: 1, 18: 1, 19: 64, 20: 128, 21: 128, 22: 8},
    div16=[7, 8, 9, 11, 13, 15, 16, 17],
)

_PTX_4 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk04_fa_o_w4a8_kernel  // -- Begin function _sk04_fa_o_w4a8_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk04_fa_o_w4a8_kernel
.visible .entry _sk04_fa_o_w4a8_kernel(
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_5,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_6,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_7,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_8,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_9,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_10,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_11,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_12,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_13,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_14,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_15
)
.reqntid 256
{
	.reg .pred 	%p<41>;
	.reg .b16 	%rs<289>;
	.reg .b32 	%r<1536>;
	.reg .b64 	%rd<134>;
	.loc	1 797 0                         // sk04_fa_o_w4a8.py:797:0
$L__func_begin0:
	.loc	1 797 0                         // sk04_fa_o_w4a8.py:797:0

// %bb.0:
	ld.param.b32 	%r34, [_sk04_fa_o_w4a8_kernel_param_12];
	ld.param.b32 	%r33, [_sk04_fa_o_w4a8_kernel_param_11];
	ld.param.b32 	%r32, [_sk04_fa_o_w4a8_kernel_param_8];
	ld.param.b32 	%r31, [_sk04_fa_o_w4a8_kernel_param_7];
	ld.param.b32 	%r30, [_sk04_fa_o_w4a8_kernel_param_6];
	ld.param.b64 	%rd17, [_sk04_fa_o_w4a8_kernel_param_4];
	ld.param.b64 	%rd16, [_sk04_fa_o_w4a8_kernel_param_3];
	ld.param.b64 	%rd15, [_sk04_fa_o_w4a8_kernel_param_2];
	ld.param.b64 	%rd14, [_sk04_fa_o_w4a8_kernel_param_1];
	ld.param.b64 	%rd13, [_sk04_fa_o_w4a8_kernel_param_0];
$L__tmp0:
	.loc	1 807 24                        // sk04_fa_o_w4a8.py:807:24
	mov.u32 	%r52, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk04_fa_o_w4a8.py:808:27 ]
	add.s32 	%r53, %r30, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk04_fa_o_w4a8.py:808:27 ]
	shr.s32 	%r54, %r53, 31;
	shr.u32 	%r55, %r54, 25;
	add.s32 	%r56, %r53, %r55;
	shr.s32 	%r57, %r56, 7;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk04_fa_o_w4a8.py:809:27 ]
	add.s32 	%r58, %r31, 255;
	.loc	2 43 30                         // standard.py:43:30 @[ sk04_fa_o_w4a8.py:809:27 ]
	shr.s32 	%r59, %r58, 31;
	shr.u32 	%r60, %r59, 24;
	add.s32 	%r61, %r58, %r60;
	shr.s32 	%r62, %r61, 8;
$L__tmp3:
	.loc	1 810 29                        // sk04_fa_o_w4a8.py:810:29
	shl.b32 	%r63, %r62, 3;
	.loc	1 811 22                        // sk04_fa_o_w4a8.py:811:22
	div.s32 	%r64, %r52, %r63;
	.loc	1 811 38                        // sk04_fa_o_w4a8.py:811:38
	shl.b32 	%r65, %r64, 3;
	ld.param.b32 	%r66, [_sk04_fa_o_w4a8_kernel_param_9];
	.loc	1 812 30                        // sk04_fa_o_w4a8.py:812:30
	sub.s32 	%r67, %r57, %r65;
	ld.param.b32 	%r68, [_sk04_fa_o_w4a8_kernel_param_10];
	.loc	1 812 39                        // sk04_fa_o_w4a8.py:812:39
	min.s32 	%r69, %r67, 8;
	.loc	1 813 30                        // sk04_fa_o_w4a8.py:813:30
	mul.lo.s32 	%r70, %r64, %r63;
	sub.s32 	%r71, %r52, %r70;
	.loc	1 814 36                        // sk04_fa_o_w4a8.py:814:36
	div.s32 	%r72, %r71, %r69;
	.loc	1 813 46                        // sk04_fa_o_w4a8.py:813:46
	mul.lo.s32 	%r73, %r72, %r69;
	sub.s32 	%r74, %r71, %r73;
	.loc	1 813 23                        // sk04_fa_o_w4a8.py:813:23
	add.s32 	%r75, %r74, %r65;
	.loc	1 816 22                        // sk04_fa_o_w4a8.py:816:22
	shl.b32 	%r1, %r75, 7;
	.loc	1 816 45                        // sk04_fa_o_w4a8.py:816:45
	mov.u32 	%r2, %tid.x;
	shr.u32 	%r3, %r2, 2;
	bfe.u32 	%r4, %r2, 2, 6;
	and.b32 	%r5, %r2, 224;
	bfe.u32 	%r6, %r2, 5, 3;
	or.b32 	%r7, %r6, 8;
	or.b32 	%r8, %r6, 16;
	or.b32 	%r9, %r6, 24;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r76, %r1, %r4;
	or.b32 	%r77, %r76, 64;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r78, %r76, %r30;
	rem.s32 	%r79, %r77, %r30;
	.loc	1 817 22                        // sk04_fa_o_w4a8.py:817:22
	shl.b32 	%r10, %r72, 8;
	.loc	1 817 45                        // sk04_fa_o_w4a8.py:817:45
	and.b32 	%r11, %r2, 31;
	shl.b32 	%r80, %r11, 3;
	and.b32 	%r12, %r2, 255;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r13, %r10, %r80;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r81, %r13, %r31;
	.loc	1 821 39                        // sk04_fa_o_w4a8.py:821:39
	mul.lo.s32 	%r82, %r78, %r66;
	mul.lo.s32 	%r83, %r79, %r66;
	.loc	1 821 21                        // sk04_fa_o_w4a8.py:821:21
	cvt.s64.s32 	%rd1, %r82;
	add.s64 	%rd31, %rd13, %rd1;
	cvt.s64.s32 	%rd2, %r83;
	add.s64 	%rd32, %rd13, %rd2;
	.loc	1 821 58                        // sk04_fa_o_w4a8.py:821:58
	and.b32 	%r14, %r2, 3;
	shl.b32 	%r84, %r14, 4;
	.loc	1 821 51                        // sk04_fa_o_w4a8.py:821:51
	cvt.u64.u32 	%rd3, %r84;
	add.s64 	%rd23, %rd31, %rd3;
	add.s64 	%rd24, %rd32, %rd3;
	.loc	1 822 40                        // sk04_fa_o_w4a8.py:822:40
	mul.lo.s32 	%r85, %r68, %r6;
	shl.b32 	%r86, %r68, 3;
	add.s32 	%r87, %r85, %r86;
	shl.b32 	%r88, %r68, 4;
	add.s32 	%r89, %r85, %r88;
	mad.lo.s32 	%r90, %r68, 24, %r85;
	.loc	1 822 21                        // sk04_fa_o_w4a8.py:822:21
	cvt.s64.s32 	%rd4, %r85;
	add.s64 	%rd33, %rd14, %rd4;
	cvt.s64.s32 	%rd5, %r87;
	add.s64 	%rd34, %rd14, %rd5;
	cvt.s64.s32 	%rd6, %r89;
	add.s64 	%rd35, %rd14, %rd6;
	cvt.s64.s32 	%rd7, %r90;
	add.s64 	%rd36, %rd14, %rd7;
	.loc	1 822 52                        // sk04_fa_o_w4a8.py:822:52
	cvt.s64.s32 	%rd8, %r81;
	add.s64 	%rd19, %rd33, %rd8;
	add.s64 	%rd21, %rd34, %rd8;
	add.s64 	%rd20, %rd35, %rd8;
	add.s64 	%rd22, %rd36, %rd8;
	.loc	1 836 35                        // sk04_fa_o_w4a8.py:836:35
	shl.b32 	%r91, %r68, 5;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	setp.lt.s32 	%p1, %r32, 64;
	setp.gt.s32 	%p2, %r32, 63;
	.loc	1 827 25                        // sk04_fa_o_w4a8.py:827:25
	shl.b32 	%r92, %r12, 3;
	shr.u32 	%r93, %r5, 2;
	xor.b32 	%r94, %r92, %r93;
	mov.b32 	%r95, global_smem;
	add.s32 	%r36, %r95, %r94;
	selp.b32 	%r37, 8, 0, %p2;
	// begin inline asm
	cp.async.ca.shared.global [ %r36 + 0 ], [ %rd19 + 0 ], 0x8, %r37;
	// end inline asm
	add.s32 	%r38, %r36, 4096;
	// begin inline asm
	cp.async.ca.shared.global [ %r38 + 0 ], [ %rd20 + 0 ], 0x8, %r37;
	// end inline asm
	xor.b32 	%r15, %r94, 64;
	add.s32 	%r96, %r95, %r15;
	add.s32 	%r39, %r96, 2048;
	// begin inline asm
	cp.async.ca.shared.global [ %r39 + 0 ], [ %rd21 + 0 ], 0x8, %r37;
	// end inline asm
	add.s32 	%r40, %r96, 6144;
	// begin inline asm
	cp.async.ca.shared.global [ %r40 + 0 ], [ %rd22 + 0 ], 0x8, %r37;
	// end inline asm
	cp.async.commit_group;
	.loc	1 831 52                        // sk04_fa_o_w4a8.py:831:52
	shl.b32 	%r16, %r2, 4;
	and.b32 	%r97, %r16, 3952;
	bfe.s32 	%r98, %r2, 3, 1;
	and.b32 	%r17, %r98, 160;
	xor.b32 	%r99, %r17, %r97;
	add.s32 	%r18, %r95, %r99;
	add.s32 	%r41, %r18, 16384;
	selp.b32 	%r42, 16, 0, %p2;
	// begin inline asm
	cp.async.cg.shared.global [ %r41 + 0 ], [ %rd23 + 0 ], 0x10, %r42;
	// end inline asm
	add.s32 	%r43, %r18, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r43 + 0 ], [ %rd24 + 0 ], 0x10, %r42;
	// end inline asm
	cp.async.commit_group;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	setp.gt.s32 	%p3, %r32, 127;
	.loc	1 835 18                        // sk04_fa_o_w4a8.py:835:18
	add.s64 	%rd29, %rd23, 64;
	add.s64 	%rd30, %rd24, 64;
	.loc	1 836 18                        // sk04_fa_o_w4a8.py:836:18
	cvt.s64.s32 	%rd9, %r91;
	add.s64 	%rd25, %rd19, %rd9;
	add.s64 	%rd27, %rd21, %rd9;
	add.s64 	%rd26, %rd20, %rd9;
	add.s64 	%rd28, %rd22, %rd9;
	.loc	1 827 25                        // sk04_fa_o_w4a8.py:827:25
	bar.sync 	0;
	add.s32 	%r44, %r36, 8192;
	selp.b32 	%r45, 8, 0, %p3;
	// begin inline asm
	cp.async.ca.shared.global [ %r44 + 0 ], [ %rd25 + 0 ], 0x8, %r45;
	// end inline asm
	add.s32 	%r46, %r36, 12288;
	// begin inline asm
	cp.async.ca.shared.global [ %r46 + 0 ], [ %rd26 + 0 ], 0x8, %r45;
	// end inline asm
	add.s32 	%r47, %r96, 10240;
	// begin inline asm
	cp.async.ca.shared.global [ %r47 + 0 ], [ %rd27 + 0 ], 0x8, %r45;
	// end inline asm
	add.s32 	%r48, %r96, 14336;
	// begin inline asm
	cp.async.ca.shared.global [ %r48 + 0 ], [ %rd28 + 0 ], 0x8, %r45;
	// end inline asm
	cp.async.commit_group;
	.loc	1 831 52                        // sk04_fa_o_w4a8.py:831:52
	add.s32 	%r49, %r18, 24576;
	selp.b32 	%r50, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r49 + 0 ], [ %rd29 + 0 ], 0x10, %r50;
	// end inline asm
	add.s32 	%r51, %r18, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r51 + 0 ], [ %rd30 + 0 ], 0x10, %r50;
	// end inline asm
	cp.async.commit_group;
	mov.b32 	%r1408, 0f00000000;
	mov.b32 	%r1409, %r1408;
	mov.b32 	%r1410, %r1408;
	mov.b32 	%r1411, %r1408;
	mov.b32 	%r1412, %r1408;
	mov.b32 	%r1413, %r1408;
	mov.b32 	%r1414, %r1408;
	mov.b32 	%r1415, %r1408;
	mov.b32 	%r1416, %r1408;
	mov.b32 	%r1417, %r1408;
	mov.b32 	%r1418, %r1408;
	mov.b32 	%r1419, %r1408;
	mov.b32 	%r1420, %r1408;
	mov.b32 	%r1421, %r1408;
	mov.b32 	%r1422, %r1408;
	mov.b32 	%r1423, %r1408;
	mov.b32 	%r1424, %r1408;
	mov.b32 	%r1425, %r1408;
	mov.b32 	%r1426, %r1408;
	mov.b32 	%r1427, %r1408;
	mov.b32 	%r1428, %r1408;
	mov.b32 	%r1429, %r1408;
	mov.b32 	%r1430, %r1408;
	mov.b32 	%r1431, %r1408;
	mov.b32 	%r1432, %r1408;
	mov.b32 	%r1433, %r1408;
	mov.b32 	%r1434, %r1408;
	mov.b32 	%r1435, %r1408;
	mov.b32 	%r1436, %r1408;
	mov.b32 	%r1437, %r1408;
	mov.b32 	%r1438, %r1408;
	mov.b32 	%r1439, %r1408;
	mov.b32 	%r1440, %r1408;
	mov.b32 	%r1441, %r1408;
	mov.b32 	%r1442, %r1408;
	mov.b32 	%r1443, %r1408;
	mov.b32 	%r1444, %r1408;
	mov.b32 	%r1445, %r1408;
	mov.b32 	%r1446, %r1408;
	mov.b32 	%r1447, %r1408;
	mov.b32 	%r1448, %r1408;
	mov.b32 	%r1449, %r1408;
	mov.b32 	%r1450, %r1408;
	mov.b32 	%r1451, %r1408;
	mov.b32 	%r1452, %r1408;
	mov.b32 	%r1453, %r1408;
	mov.b32 	%r1454, %r1408;
	mov.b32 	%r1455, %r1408;
	mov.b32 	%r1456, %r1408;
	mov.b32 	%r1457, %r1408;
	mov.b32 	%r1458, %r1408;
	mov.b32 	%r1459, %r1408;
	mov.b32 	%r1460, %r1408;
	mov.b32 	%r1461, %r1408;
	mov.b32 	%r1462, %r1408;
	mov.b32 	%r1463, %r1408;
	mov.b32 	%r1464, %r1408;
	mov.b32 	%r1465, %r1408;
	mov.b32 	%r1466, %r1408;
	mov.b32 	%r1467, %r1408;
	mov.b32 	%r1468, %r1408;
	mov.b32 	%r1469, %r1408;
	mov.b32 	%r1470, %r1408;
	mov.b32 	%r1471, %r1408;
	mov.b32 	%r1472, %r1408;
	mov.b32 	%r1473, %r1408;
	mov.b32 	%r1474, %r1408;
	mov.b32 	%r1475, %r1408;
	mov.b32 	%r1476, %r1408;
	mov.b32 	%r1477, %r1408;
	mov.b32 	%r1478, %r1408;
	mov.b32 	%r1479, %r1408;
	mov.b32 	%r1480, %r1408;
	mov.b32 	%r1481, %r1408;
	mov.b32 	%r1482, %r1408;
	mov.b32 	%r1483, %r1408;
	mov.b32 	%r1484, %r1408;
	mov.b32 	%r1485, %r1408;
	mov.b32 	%r1486, %r1408;
	mov.b32 	%r1487, %r1408;
	mov.b32 	%r1488, %r1408;
	mov.b32 	%r1489, %r1408;
	mov.b32 	%r1490, %r1408;
	mov.b32 	%r1491, %r1408;
	mov.b32 	%r1492, %r1408;
	mov.b32 	%r1493, %r1408;
	mov.b32 	%r1494, %r1408;
	mov.b32 	%r1495, %r1408;
	mov.b32 	%r1496, %r1408;
	mov.b32 	%r1497, %r1408;
	mov.b32 	%r1498, %r1408;
	mov.b32 	%r1499, %r1408;
	mov.b32 	%r1500, %r1408;
	mov.b32 	%r1501, %r1408;
	mov.b32 	%r1502, %r1408;
	mov.b32 	%r1503, %r1408;
	mov.b32 	%r1504, %r1408;
	mov.b32 	%r1505, %r1408;
	mov.b32 	%r1506, %r1408;
	mov.b32 	%r1507, %r1408;
	mov.b32 	%r1508, %r1408;
	mov.b32 	%r1509, %r1408;
	mov.b32 	%r1510, %r1408;
	mov.b32 	%r1511, %r1408;
	mov.b32 	%r1512, %r1408;
	mov.b32 	%r1513, %r1408;
	mov.b32 	%r1514, %r1408;
	mov.b32 	%r1515, %r1408;
	mov.b32 	%r1516, %r1408;
	mov.b32 	%r1517, %r1408;
	mov.b32 	%r1518, %r1408;
	mov.b32 	%r1519, %r1408;
	mov.b32 	%r1520, %r1408;
	mov.b32 	%r1521, %r1408;
	mov.b32 	%r1522, %r1408;
	mov.b32 	%r1523, %r1408;
	mov.b32 	%r1524, %r1408;
	mov.b32 	%r1525, %r1408;
	mov.b32 	%r1526, %r1408;
	mov.b32 	%r1527, %r1408;
	mov.b32 	%r1528, %r1408;
	mov.b32 	%r1529, %r1408;
	mov.b32 	%r1530, %r1408;
	mov.b32 	%r1531, %r1408;
	mov.b32 	%r1532, %r1408;
	mov.b32 	%r1533, %r1408;
	mov.b32 	%r1534, %r1408;
	mov.b32 	%r1535, %r1408;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	@%p1 bra 	$L__BB0_3;
// %bb.1:                               // %.lr.ph
	.loc	1 0 22                          // sk04_fa_o_w4a8.py:0:22
	ld.param.b32 	%r35, [_sk04_fa_o_w4a8_kernel_param_13];
	ld.param.b64 	%rd18, [_sk04_fa_o_w4a8_kernel_param_5];
	.loc	1 825 27                        // sk04_fa_o_w4a8.py:825:27
	shr.u32 	%r100, %r32, 6;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	add.s32 	%r101, %r95, %r5;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r102, %r10, %r12;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r103, %r102, %r31;
	add.s32 	%r104, %r100, -2;
	mul.lo.s32 	%r105, %r14, 1056;
	xor.b32 	%r19, %r105, %r4;
	xor.b32 	%r20, %r19, 264;
	xor.b32 	%r21, %r19, 528;
	xor.b32 	%r22, %r19, 792;
	xor.b32 	%r23, %r19, 64;
	xor.b32 	%r24, %r19, 328;
	xor.b32 	%r25, %r19, 592;
	xor.b32 	%r26, %r19, 856;
	and.b32 	%r106, %r16, 320;
	shl.b32 	%r107, %r14, 3;
	or.b32 	%r108, %r106, %r107;
	or.b32 	%r27, %r108, %r17;
	xor.b32 	%r28, %r27, 32;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	mad.wide.s32 	%rd10, %r103, 4, %rd18;
	shl.b32 	%r109, %r12, 2;
	add.s32 	%r110, %r95, %r109;
	add.s32 	%r321, %r110, 32768;
	add.s32 	%r29, %r101, %r107;
	cvt.s64.s32 	%rd11, %r104;
	cvt.u64.u32 	%rd12, %r100;
	shl.b64 	%rd37, %rd9, 1;
	add.s64 	%rd38, %rd37, %rd8;
	add.s64 	%rd132, %rd14, %rd38;
	add.s64 	%rd39, %rd3, %rd2;
	add.s64 	%rd40, %rd39, %rd13;
	add.s64 	%rd131, %rd40, 128;
	add.s64 	%rd41, %rd3, %rd1;
	add.s64 	%rd42, %rd41, %rd13;
	add.s64 	%rd130, %rd42, 128;
	mov.b32 	%r1408, 0f00000000;
	mov.b32 	%r1407, 1;
	mov.b32 	%r1406, -1;
	mov.b64 	%rd133, 0;
	mov.b32 	%r111, 0;
	mov.b32 	%r1405, %r111;
	mov.b32 	%r1409, %r1408;
	mov.b32 	%r1410, %r1408;
	mov.b32 	%r1411, %r1408;
	mov.b32 	%r1412, %r1408;
	mov.b32 	%r1413, %r1408;
	mov.b32 	%r1414, %r1408;
	mov.b32 	%r1415, %r1408;
	mov.b32 	%r1416, %r1408;
	mov.b32 	%r1417, %r1408;
	mov.b32 	%r1418, %r1408;
	mov.b32 	%r1419, %r1408;
	mov.b32 	%r1420, %r1408;
	mov.b32 	%r1421, %r1408;
	mov.b32 	%r1422, %r1408;
	mov.b32 	%r1423, %r1408;
	mov.b32 	%r1424, %r1408;
	mov.b32 	%r1425, %r1408;
	mov.b32 	%r1426, %r1408;
	mov.b32 	%r1427, %r1408;
	mov.b32 	%r1428, %r1408;
	mov.b32 	%r1429, %r1408;
	mov.b32 	%r1430, %r1408;
	mov.b32 	%r1431, %r1408;
	mov.b32 	%r1432, %r1408;
	mov.b32 	%r1433, %r1408;
	mov.b32 	%r1434, %r1408;
	mov.b32 	%r1435, %r1408;
	mov.b32 	%r1436, %r1408;
	mov.b32 	%r1437, %r1408;
	mov.b32 	%r1438, %r1408;
	mov.b32 	%r1439, %r1408;
	mov.b32 	%r1440, %r1408;
	mov.b32 	%r1441, %r1408;
	mov.b32 	%r1442, %r1408;
	mov.b32 	%r1443, %r1408;
	mov.b32 	%r1444, %r1408;
	mov.b32 	%r1445, %r1408;
	mov.b32 	%r1446, %r1408;
	mov.b32 	%r1447, %r1408;
	mov.b32 	%r1448, %r1408;
	mov.b32 	%r1449, %r1408;
	mov.b32 	%r1450, %r1408;
	mov.b32 	%r1451, %r1408;
	mov.b32 	%r1452, %r1408;
	mov.b32 	%r1453, %r1408;
	mov.b32 	%r1454, %r1408;
	mov.b32 	%r1455, %r1408;
	mov.b32 	%r1456, %r1408;
	mov.b32 	%r1457, %r1408;
	mov.b32 	%r1458, %r1408;
	mov.b32 	%r1459, %r1408;
	mov.b32 	%r1460, %r1408;
	mov.b32 	%r1461, %r1408;
	mov.b32 	%r1462, %r1408;
	mov.b32 	%r1463, %r1408;
	mov.b32 	%r1464, %r1408;
	mov.b32 	%r1465, %r1408;
	mov.b32 	%r1466, %r1408;
	mov.b32 	%r1467, %r1408;
	mov.b32 	%r1468, %r1408;
	mov.b32 	%r1469, %r1408;
	mov.b32 	%r1470, %r1408;
	mov.b32 	%r1471, %r1408;
	mov.b32 	%r1472, %r1408;
	mov.b32 	%r1473, %r1408;
	mov.b32 	%r1474, %r1408;
	mov.b32 	%r1475, %r1408;
	mov.b32 	%r1476, %r1408;
	mov.b32 	%r1477, %r1408;
	mov.b32 	%r1478, %r1408;
	mov.b32 	%r1479, %r1408;
	mov.b32 	%r1480, %r1408;
	mov.b32 	%r1481, %r1408;
	mov.b32 	%r1482, %r1408;
	mov.b32 	%r1483, %r1408;
	mov.b32 	%r1484, %r1408;
	mov.b32 	%r1485, %r1408;
	mov.b32 	%r1486, %r1408;
	mov.b32 	%r1487, %r1408;
	mov.b32 	%r1488, %r1408;
	mov.b32 	%r1489, %r1408;
	mov.b32 	%r1490, %r1408;
	mov.b32 	%r1491, %r1408;
	mov.b32 	%r1492, %r1408;
	mov.b32 	%r1493, %r1408;
	mov.b32 	%r1494, %r1408;
	mov.b32 	%r1495, %r1408;
	mov.b32 	%r1496, %r1408;
	mov.b32 	%r1497, %r1408;
	mov.b32 	%r1498, %r1408;
	mov.b32 	%r1499, %r1408;
	mov.b32 	%r1500, %r1408;
	mov.b32 	%r1501, %r1408;
	mov.b32 	%r1502, %r1408;
	mov.b32 	%r1503, %r1408;
	mov.b32 	%r1504, %r1408;
	mov.b32 	%r1505, %r1408;
	mov.b32 	%r1506, %r1408;
	mov.b32 	%r1507, %r1408;
	mov.b32 	%r1508, %r1408;
	mov.b32 	%r1509, %r1408;
	mov.b32 	%r1510, %r1408;
	mov.b32 	%r1511, %r1408;
	mov.b32 	%r1512, %r1408;
	mov.b32 	%r1513, %r1408;
	mov.b32 	%r1514, %r1408;
	mov.b32 	%r1515, %r1408;
	mov.b32 	%r1516, %r1408;
	mov.b32 	%r1517, %r1408;
	mov.b32 	%r1518, %r1408;
	mov.b32 	%r1519, %r1408;
	mov.b32 	%r1520, %r1408;
	mov.b32 	%r1521, %r1408;
	mov.b32 	%r1522, %r1408;
	mov.b32 	%r1523, %r1408;
	mov.b32 	%r1524, %r1408;
	mov.b32 	%r1525, %r1408;
	mov.b32 	%r1526, %r1408;
	mov.b32 	%r1527, %r1408;
	mov.b32 	%r1528, %r1408;
	mov.b32 	%r1529, %r1408;
	mov.b32 	%r1530, %r1408;
	mov.b32 	%r1531, %r1408;
	mov.b32 	%r1532, %r1408;
	mov.b32 	%r1533, %r1408;
	mov.b32 	%r1534, %r1408;
	mov.b32 	%r1535, %r1408;
$L__BB0_2:                              // =>This Inner Loop Header: Depth=1
	setp.lt.s64 	%p4, %rd133, %rd11;
	add.s32 	%r330, %r1406, 1;
	setp.gt.s32 	%p5, %r330, 1;
	selp.b32 	%r1406, 0, %r330, %p5;
	.loc	1 827 25                        // sk04_fa_o_w4a8.py:827:25
	cp.async.wait_group 	2;
	bar.sync 	0;
	shl.b32 	%r331, %r1406, 13;
	add.s32 	%r332, %r95, %r331;
	.loc	1 828 38                        // sk04_fa_o_w4a8.py:828:38
	add.s32 	%r333, %r332, %r19;
	ld.shared.b8 	%rs1, [%r333];
	ld.shared.b8 	%rs2, [%r333+4096];
	ld.shared.b8 	%rs3, [%r333+128];
	ld.shared.b8 	%rs4, [%r333+4224];
	add.s32 	%r334, %r332, %r20;
	ld.shared.b8 	%rs5, [%r334];
	ld.shared.b8 	%rs6, [%r334+4096];
	ld.shared.b8 	%rs7, [%r334+128];
	ld.shared.b8 	%rs8, [%r334+4224];
	add.s32 	%r335, %r332, %r21;
	ld.shared.b8 	%rs9, [%r335];
	ld.shared.b8 	%rs10, [%r335+4096];
	ld.shared.b8 	%rs11, [%r335+128];
	ld.shared.b8 	%rs12, [%r335+4224];
	add.s32 	%r336, %r332, %r22;
	ld.shared.b8 	%rs13, [%r336];
	ld.shared.b8 	%rs14, [%r336+4096];
	ld.shared.b8 	%rs15, [%r336+128];
	ld.shared.b8 	%rs16, [%r336+4224];
	add.s32 	%r337, %r332, %r23;
	ld.shared.b8 	%rs17, [%r337];
	ld.shared.b8 	%rs18, [%r337+4096];
	ld.shared.b8 	%rs19, [%r337+128];
	ld.shared.b8 	%rs20, [%r337+4224];
	add.s32 	%r338, %r332, %r24;
	ld.shared.b8 	%rs21, [%r338];
	ld.shared.b8 	%rs22, [%r338+4096];
	ld.shared.b8 	%rs23, [%r338+128];
	ld.shared.b8 	%rs24, [%r338+4224];
	add.s32 	%r339, %r332, %r25;
	ld.shared.b8 	%rs25, [%r339];
	ld.shared.b8 	%rs26, [%r339+4096];
	ld.shared.b8 	%rs27, [%r339+128];
	ld.shared.b8 	%rs28, [%r339+4224];
	add.s32 	%r340, %r332, %r26;
	ld.shared.b8 	%rs29, [%r340];
	ld.shared.b8 	%rs30, [%r340+4096];
	ld.shared.b8 	%rs31, [%r340+128];
	ld.shared.b8 	%rs32, [%r340+4224];
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r341, %rs13;
	cvt.u32.u16 	%r342, %rs9;
	prmt.b32 	%r343, %r342, %r341, 0x3340U;
	cvt.u32.u16 	%r344, %rs5;
	cvt.u32.u16 	%r345, %rs1;
	prmt.b32 	%r346, %r345, %r344, 0x3340U;
	prmt.b32 	%r347, %r346, %r343, 0x5410U;
	and.b32 	%r348, %r347, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r349, %r348, 0, 0x7773U;
	cvt.u16.u32 	%rs33, %r349;
	add.s16 	%rs34, %rs33, -8;
	cvt.u32.u16 	%r350, %rs34;
	prmt.b32 	%r351, %r348, 0, 0x7772U;
	cvt.u16.u32 	%rs35, %r351;
	add.s16 	%rs36, %rs35, -8;
	cvt.u32.u16 	%r352, %rs36;
	prmt.b32 	%r353, %r352, %r350, 0x3340U;
	prmt.b32 	%r354, %r348, 0, 0x7771U;
	cvt.u16.u32 	%rs37, %r354;
	add.s16 	%rs38, %rs37, -8;
	cvt.u32.u16 	%r355, %rs38;
	prmt.b32 	%r356, %r348, 0, 0x7770U;
	cvt.u16.u32 	%rs39, %r356;
	add.s16 	%rs40, %rs39, -8;
	cvt.u32.u16 	%r357, %rs40;
	prmt.b32 	%r358, %r357, %r355, 0x3340U;
	prmt.b32 	%r116, %r358, %r353, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r359, %rs14;
	cvt.u32.u16 	%r360, %rs10;
	prmt.b32 	%r361, %r360, %r359, 0x3340U;
	cvt.u32.u16 	%r362, %rs6;
	cvt.u32.u16 	%r363, %rs2;
	prmt.b32 	%r364, %r363, %r362, 0x3340U;
	prmt.b32 	%r365, %r364, %r361, 0x5410U;
	and.b32 	%r366, %r365, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r367, %r366, 0, 0x7773U;
	cvt.u16.u32 	%rs41, %r367;
	add.s16 	%rs42, %rs41, -8;
	cvt.u32.u16 	%r368, %rs42;
	prmt.b32 	%r369, %r366, 0, 0x7772U;
	cvt.u16.u32 	%rs43, %r369;
	add.s16 	%rs44, %rs43, -8;
	cvt.u32.u16 	%r370, %rs44;
	prmt.b32 	%r371, %r370, %r368, 0x3340U;
	prmt.b32 	%r372, %r366, 0, 0x7771U;
	cvt.u16.u32 	%rs45, %r372;
	add.s16 	%rs46, %rs45, -8;
	cvt.u32.u16 	%r373, %rs46;
	prmt.b32 	%r374, %r366, 0, 0x7770U;
	cvt.u16.u32 	%rs47, %r374;
	add.s16 	%rs48, %rs47, -8;
	cvt.u32.u16 	%r375, %rs48;
	prmt.b32 	%r376, %r375, %r373, 0x3340U;
	prmt.b32 	%r117, %r376, %r371, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r377, %rs29;
	cvt.u32.u16 	%r378, %rs25;
	prmt.b32 	%r379, %r378, %r377, 0x3340U;
	cvt.u32.u16 	%r380, %rs21;
	cvt.u32.u16 	%r381, %rs17;
	prmt.b32 	%r382, %r381, %r380, 0x3340U;
	prmt.b32 	%r383, %r382, %r379, 0x5410U;
	and.b32 	%r384, %r383, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r385, %r384, 0, 0x7773U;
	cvt.u16.u32 	%rs49, %r385;
	add.s16 	%rs50, %rs49, -8;
	cvt.u32.u16 	%r386, %rs50;
	prmt.b32 	%r387, %r384, 0, 0x7772U;
	cvt.u16.u32 	%rs51, %r387;
	add.s16 	%rs52, %rs51, -8;
	cvt.u32.u16 	%r388, %rs52;
	prmt.b32 	%r389, %r388, %r386, 0x3340U;
	prmt.b32 	%r390, %r384, 0, 0x7771U;
	cvt.u16.u32 	%rs53, %r390;
	add.s16 	%rs54, %rs53, -8;
	cvt.u32.u16 	%r391, %rs54;
	prmt.b32 	%r392, %r384, 0, 0x7770U;
	cvt.u16.u32 	%rs55, %r392;
	add.s16 	%rs56, %rs55, -8;
	cvt.u32.u16 	%r393, %rs56;
	prmt.b32 	%r394, %r393, %r391, 0x3340U;
	prmt.b32 	%r122, %r394, %r389, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r395, %rs30;
	cvt.u32.u16 	%r396, %rs26;
	prmt.b32 	%r397, %r396, %r395, 0x3340U;
	cvt.u32.u16 	%r398, %rs22;
	cvt.u32.u16 	%r399, %rs18;
	prmt.b32 	%r400, %r399, %r398, 0x3340U;
	prmt.b32 	%r401, %r400, %r397, 0x5410U;
	and.b32 	%r402, %r401, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r403, %r402, 0, 0x7773U;
	cvt.u16.u32 	%rs57, %r403;
	add.s16 	%rs58, %rs57, -8;
	cvt.u32.u16 	%r404, %rs58;
	prmt.b32 	%r405, %r402, 0, 0x7772U;
	cvt.u16.u32 	%rs59, %r405;
	add.s16 	%rs60, %rs59, -8;
	cvt.u32.u16 	%r406, %rs60;
	prmt.b32 	%r407, %r406, %r404, 0x3340U;
	prmt.b32 	%r408, %r402, 0, 0x7771U;
	cvt.u16.u32 	%rs61, %r408;
	add.s16 	%rs62, %rs61, -8;
	cvt.u32.u16 	%r409, %rs62;
	prmt.b32 	%r410, %r402, 0, 0x7770U;
	cvt.u16.u32 	%rs63, %r410;
	add.s16 	%rs64, %rs63, -8;
	cvt.u32.u16 	%r411, %rs64;
	prmt.b32 	%r412, %r411, %r409, 0x3340U;
	prmt.b32 	%r123, %r412, %r407, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r413, %rs15;
	cvt.u32.u16 	%r414, %rs11;
	prmt.b32 	%r415, %r414, %r413, 0x3340U;
	cvt.u32.u16 	%r416, %rs7;
	cvt.u32.u16 	%r417, %rs3;
	prmt.b32 	%r418, %r417, %r416, 0x3340U;
	prmt.b32 	%r419, %r418, %r415, 0x5410U;
	and.b32 	%r420, %r419, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r421, %r420, 0, 0x7773U;
	cvt.u16.u32 	%rs65, %r421;
	add.s16 	%rs66, %rs65, -8;
	cvt.u32.u16 	%r422, %rs66;
	prmt.b32 	%r423, %r420, 0, 0x7772U;
	cvt.u16.u32 	%rs67, %r423;
	add.s16 	%rs68, %rs67, -8;
	cvt.u32.u16 	%r424, %rs68;
	prmt.b32 	%r425, %r424, %r422, 0x3340U;
	prmt.b32 	%r426, %r420, 0, 0x7771U;
	cvt.u16.u32 	%rs69, %r426;
	add.s16 	%rs70, %rs69, -8;
	cvt.u32.u16 	%r427, %rs70;
	prmt.b32 	%r428, %r420, 0, 0x7770U;
	cvt.u16.u32 	%rs71, %r428;
	add.s16 	%rs72, %rs71, -8;
	cvt.u32.u16 	%r429, %rs72;
	prmt.b32 	%r430, %r429, %r427, 0x3340U;
	prmt.b32 	%r124, %r430, %r425, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r431, %rs16;
	cvt.u32.u16 	%r432, %rs12;
	prmt.b32 	%r433, %r432, %r431, 0x3340U;
	cvt.u32.u16 	%r434, %rs8;
	cvt.u32.u16 	%r435, %rs4;
	prmt.b32 	%r436, %r435, %r434, 0x3340U;
	prmt.b32 	%r437, %r436, %r433, 0x5410U;
	and.b32 	%r438, %r437, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r439, %r438, 0, 0x7773U;
	cvt.u16.u32 	%rs73, %r439;
	add.s16 	%rs74, %rs73, -8;
	cvt.u32.u16 	%r440, %rs74;
	prmt.b32 	%r441, %r438, 0, 0x7772U;
	cvt.u16.u32 	%rs75, %r441;
	add.s16 	%rs76, %rs75, -8;
	cvt.u32.u16 	%r442, %rs76;
	prmt.b32 	%r443, %r442, %r440, 0x3340U;
	prmt.b32 	%r444, %r438, 0, 0x7771U;
	cvt.u16.u32 	%rs77, %r444;
	add.s16 	%rs78, %rs77, -8;
	cvt.u32.u16 	%r445, %rs78;
	prmt.b32 	%r446, %r438, 0, 0x7770U;
	cvt.u16.u32 	%rs79, %r446;
	add.s16 	%rs80, %rs79, -8;
	cvt.u32.u16 	%r447, %rs80;
	prmt.b32 	%r448, %r447, %r445, 0x3340U;
	prmt.b32 	%r125, %r448, %r443, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r449, %rs31;
	cvt.u32.u16 	%r450, %rs27;
	prmt.b32 	%r451, %r450, %r449, 0x3340U;
	cvt.u32.u16 	%r452, %rs23;
	cvt.u32.u16 	%r453, %rs19;
	prmt.b32 	%r454, %r453, %r452, 0x3340U;
	prmt.b32 	%r455, %r454, %r451, 0x5410U;
	and.b32 	%r456, %r455, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r457, %r456, 0, 0x7773U;
	cvt.u16.u32 	%rs81, %r457;
	add.s16 	%rs82, %rs81, -8;
	cvt.u32.u16 	%r458, %rs82;
	prmt.b32 	%r459, %r456, 0, 0x7772U;
	cvt.u16.u32 	%rs83, %r459;
	add.s16 	%rs84, %rs83, -8;
	cvt.u32.u16 	%r460, %rs84;
	prmt.b32 	%r461, %r460, %r458, 0x3340U;
	prmt.b32 	%r462, %r456, 0, 0x7771U;
	cvt.u16.u32 	%rs85, %r462;
	add.s16 	%rs86, %rs85, -8;
	cvt.u32.u16 	%r463, %rs86;
	prmt.b32 	%r464, %r456, 0, 0x7770U;
	cvt.u16.u32 	%rs87, %r464;
	add.s16 	%rs88, %rs87, -8;
	cvt.u32.u16 	%r465, %rs88;
	prmt.b32 	%r466, %r465, %r463, 0x3340U;
	prmt.b32 	%r126, %r466, %r461, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r467, %rs32;
	cvt.u32.u16 	%r468, %rs28;
	prmt.b32 	%r469, %r468, %r467, 0x3340U;
	cvt.u32.u16 	%r470, %rs24;
	cvt.u32.u16 	%r471, %rs20;
	prmt.b32 	%r472, %r471, %r470, 0x3340U;
	prmt.b32 	%r473, %r472, %r469, 0x5410U;
	and.b32 	%r474, %r473, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r475, %r474, 0, 0x7773U;
	cvt.u16.u32 	%rs89, %r475;
	add.s16 	%rs90, %rs89, -8;
	cvt.u32.u16 	%r476, %rs90;
	prmt.b32 	%r477, %r474, 0, 0x7772U;
	cvt.u16.u32 	%rs91, %r477;
	add.s16 	%rs92, %rs91, -8;
	cvt.u32.u16 	%r478, %rs92;
	prmt.b32 	%r479, %r478, %r476, 0x3340U;
	prmt.b32 	%r480, %r474, 0, 0x7771U;
	cvt.u16.u32 	%rs93, %r480;
	add.s16 	%rs94, %rs93, -8;
	cvt.u32.u16 	%r481, %rs94;
	prmt.b32 	%r482, %r474, 0, 0x7770U;
	cvt.u16.u32 	%rs95, %r482;
	add.s16 	%rs96, %rs95, -8;
	cvt.u32.u16 	%r483, %rs96;
	prmt.b32 	%r484, %r483, %r481, 0x3340U;
	prmt.b32 	%r127, %r484, %r479, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs97, %rs1, 4;
	shr.u16 	%rs98, %rs5, 4;
	shr.u16 	%rs99, %rs9, 4;
	shr.u16 	%rs100, %rs13, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs101, %rs100, -8;
	cvt.u32.u16 	%r485, %rs101;
	add.s16 	%rs102, %rs99, -8;
	cvt.u32.u16 	%r486, %rs102;
	prmt.b32 	%r487, %r486, %r485, 0x3340U;
	add.s16 	%rs103, %rs98, -8;
	cvt.u32.u16 	%r488, %rs103;
	add.s16 	%rs104, %rs97, -8;
	cvt.u32.u16 	%r489, %rs104;
	prmt.b32 	%r490, %r489, %r488, 0x3340U;
	prmt.b32 	%r176, %r490, %r487, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs105, %rs2, 4;
	shr.u16 	%rs106, %rs6, 4;
	shr.u16 	%rs107, %rs10, 4;
	shr.u16 	%rs108, %rs14, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs109, %rs108, -8;
	cvt.u32.u16 	%r491, %rs109;
	add.s16 	%rs110, %rs107, -8;
	cvt.u32.u16 	%r492, %rs110;
	prmt.b32 	%r493, %r492, %r491, 0x3340U;
	add.s16 	%rs111, %rs106, -8;
	cvt.u32.u16 	%r494, %rs111;
	add.s16 	%rs112, %rs105, -8;
	cvt.u32.u16 	%r495, %rs112;
	prmt.b32 	%r496, %r495, %r494, 0x3340U;
	prmt.b32 	%r177, %r496, %r493, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs113, %rs17, 4;
	shr.u16 	%rs114, %rs21, 4;
	shr.u16 	%rs115, %rs25, 4;
	shr.u16 	%rs116, %rs29, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs117, %rs116, -8;
	cvt.u32.u16 	%r497, %rs117;
	add.s16 	%rs118, %rs115, -8;
	cvt.u32.u16 	%r498, %rs118;
	prmt.b32 	%r499, %r498, %r497, 0x3340U;
	add.s16 	%rs119, %rs114, -8;
	cvt.u32.u16 	%r500, %rs119;
	add.s16 	%rs120, %rs113, -8;
	cvt.u32.u16 	%r501, %rs120;
	prmt.b32 	%r502, %r501, %r500, 0x3340U;
	prmt.b32 	%r186, %r502, %r499, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs121, %rs18, 4;
	shr.u16 	%rs122, %rs22, 4;
	shr.u16 	%rs123, %rs26, 4;
	shr.u16 	%rs124, %rs30, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs125, %rs124, -8;
	cvt.u32.u16 	%r503, %rs125;
	add.s16 	%rs126, %rs123, -8;
	cvt.u32.u16 	%r504, %rs126;
	prmt.b32 	%r505, %r504, %r503, 0x3340U;
	add.s16 	%rs127, %rs122, -8;
	cvt.u32.u16 	%r506, %rs127;
	add.s16 	%rs128, %rs121, -8;
	cvt.u32.u16 	%r507, %rs128;
	prmt.b32 	%r508, %r507, %r506, 0x3340U;
	prmt.b32 	%r187, %r508, %r505, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs129, %rs3, 4;
	shr.u16 	%rs130, %rs7, 4;
	shr.u16 	%rs131, %rs11, 4;
	shr.u16 	%rs132, %rs15, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs133, %rs132, -8;
	cvt.u32.u16 	%r509, %rs133;
	add.s16 	%rs134, %rs131, -8;
	cvt.u32.u16 	%r510, %rs134;
	prmt.b32 	%r511, %r510, %r509, 0x3340U;
	add.s16 	%rs135, %rs130, -8;
	cvt.u32.u16 	%r512, %rs135;
	add.s16 	%rs136, %rs129, -8;
	cvt.u32.u16 	%r513, %rs136;
	prmt.b32 	%r514, %r513, %r512, 0x3340U;
	prmt.b32 	%r192, %r514, %r511, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs137, %rs4, 4;
	shr.u16 	%rs138, %rs8, 4;
	shr.u16 	%rs139, %rs12, 4;
	shr.u16 	%rs140, %rs16, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs141, %rs140, -8;
	cvt.u32.u16 	%r515, %rs141;
	add.s16 	%rs142, %rs139, -8;
	cvt.u32.u16 	%r516, %rs142;
	prmt.b32 	%r517, %r516, %r515, 0x3340U;
	add.s16 	%rs143, %rs138, -8;
	cvt.u32.u16 	%r518, %rs143;
	add.s16 	%rs144, %rs137, -8;
	cvt.u32.u16 	%r519, %rs144;
	prmt.b32 	%r520, %r519, %r518, 0x3340U;
	prmt.b32 	%r193, %r520, %r517, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs145, %rs19, 4;
	shr.u16 	%rs146, %rs23, 4;
	shr.u16 	%rs147, %rs27, 4;
	shr.u16 	%rs148, %rs31, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs149, %rs148, -8;
	cvt.u32.u16 	%r521, %rs149;
	add.s16 	%rs150, %rs147, -8;
	cvt.u32.u16 	%r522, %rs150;
	prmt.b32 	%r523, %r522, %r521, 0x3340U;
	add.s16 	%rs151, %rs146, -8;
	cvt.u32.u16 	%r524, %rs151;
	add.s16 	%rs152, %rs145, -8;
	cvt.u32.u16 	%r525, %rs152;
	prmt.b32 	%r526, %r525, %r524, 0x3340U;
	prmt.b32 	%r198, %r526, %r523, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs153, %rs20, 4;
	shr.u16 	%rs154, %rs24, 4;
	shr.u16 	%rs155, %rs28, 4;
	shr.u16 	%rs156, %rs32, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs157, %rs156, -8;
	cvt.u32.u16 	%r527, %rs157;
	add.s16 	%rs158, %rs155, -8;
	cvt.u32.u16 	%r528, %rs158;
	prmt.b32 	%r529, %r528, %r527, 0x3340U;
	add.s16 	%rs159, %rs154, -8;
	cvt.u32.u16 	%r530, %rs159;
	add.s16 	%rs160, %rs153, -8;
	cvt.u32.u16 	%r531, %rs160;
	prmt.b32 	%r532, %r531, %r530, 0x3340U;
	prmt.b32 	%r199, %r532, %r529, 0x5410U;
	.loc	1 831 33                        // sk04_fa_o_w4a8.py:831:33
	add.s32 	%r533, %r332, %r27;
	ld.shared.v2.b32 	{%r534, %r535}, [%r533+16384];
	ld.shared.v2.b32 	{%r536, %r537}, [%r533+16896];
	ld.shared.v2.b32 	{%r538, %r539}, [%r533+17408];
	ld.shared.v2.b32 	{%r540, %r541}, [%r533+17920];
	ld.shared.v2.b32 	{%r542, %r543}, [%r533+18432];
	ld.shared.v2.b32 	{%r544, %r545}, [%r533+18944];
	ld.shared.v2.b32 	{%r546, %r547}, [%r533+19456];
	ld.shared.v2.b32 	{%r548, %r549}, [%r533+19968];
	ld.shared.v2.b32 	{%r550, %r551}, [%r533+20480];
	ld.shared.v2.b32 	{%r552, %r553}, [%r533+20992];
	ld.shared.v2.b32 	{%r554, %r555}, [%r533+21504];
	ld.shared.v2.b32 	{%r556, %r557}, [%r533+22016];
	ld.shared.v2.b32 	{%r558, %r559}, [%r533+22528];
	ld.shared.v2.b32 	{%r560, %r561}, [%r533+23040];
	ld.shared.v2.b32 	{%r562, %r563}, [%r533+23552];
	ld.shared.v2.b32 	{%r564, %r565}, [%r533+24064];
	add.s32 	%r566, %r332, %r28;
	ld.shared.v2.b32 	{%r567, %r568}, [%r566+16384];
	ld.shared.v2.b32 	{%r569, %r570}, [%r566+16896];
	ld.shared.v2.b32 	{%r571, %r572}, [%r566+17408];
	ld.shared.v2.b32 	{%r573, %r574}, [%r566+17920];
	ld.shared.v2.b32 	{%r575, %r576}, [%r566+18432];
	ld.shared.v2.b32 	{%r577, %r578}, [%r566+18944];
	ld.shared.v2.b32 	{%r579, %r580}, [%r566+19456];
	ld.shared.v2.b32 	{%r581, %r582}, [%r566+19968];
	ld.shared.v2.b32 	{%r583, %r584}, [%r566+20480];
	ld.shared.v2.b32 	{%r585, %r586}, [%r566+20992];
	ld.shared.v2.b32 	{%r587, %r588}, [%r566+21504];
	ld.shared.v2.b32 	{%r589, %r590}, [%r566+22016];
	ld.shared.v2.b32 	{%r591, %r592}, [%r566+22528];
	ld.shared.v2.b32 	{%r593, %r594}, [%r566+23040];
	ld.shared.v2.b32 	{%r595, %r596}, [%r566+23552];
	ld.shared.v2.b32 	{%r597, %r598}, [%r566+24064];
	.loc	1 832 33                        // sk04_fa_o_w4a8.py:832:33
	prmt.b32 	%r112, %r534, %r535, 0x6420U;
	prmt.b32 	%r113, %r536, %r537, 0x6420U;
	prmt.b32 	%r114, %r567, %r568, 0x6420U;
	prmt.b32 	%r115, %r569, %r570, 0x6420U;
	prmt.b32 	%r118, %r538, %r539, 0x6420U;
	prmt.b32 	%r119, %r540, %r541, 0x6420U;
	prmt.b32 	%r120, %r571, %r572, 0x6420U;
	prmt.b32 	%r121, %r573, %r574, 0x6420U;
	prmt.b32 	%r128, %r542, %r543, 0x6420U;
	prmt.b32 	%r129, %r544, %r545, 0x6420U;
	prmt.b32 	%r130, %r575, %r576, 0x6420U;
	prmt.b32 	%r131, %r577, %r578, 0x6420U;
	prmt.b32 	%r132, %r546, %r547, 0x6420U;
	prmt.b32 	%r133, %r548, %r549, 0x6420U;
	prmt.b32 	%r134, %r579, %r580, 0x6420U;
	prmt.b32 	%r135, %r581, %r582, 0x6420U;
	prmt.b32 	%r136, %r550, %r551, 0x6420U;
	prmt.b32 	%r137, %r552, %r553, 0x6420U;
	prmt.b32 	%r138, %r583, %r584, 0x6420U;
	prmt.b32 	%r139, %r585, %r586, 0x6420U;
	prmt.b32 	%r140, %r554, %r555, 0x6420U;
	prmt.b32 	%r141, %r556, %r557, 0x6420U;
	prmt.b32 	%r142, %r587, %r588, 0x6420U;
	prmt.b32 	%r143, %r589, %r590, 0x6420U;
	prmt.b32 	%r144, %r558, %r559, 0x6420U;
	prmt.b32 	%r145, %r560, %r561, 0x6420U;
	prmt.b32 	%r146, %r591, %r592, 0x6420U;
	prmt.b32 	%r147, %r593, %r594, 0x6420U;
	prmt.b32 	%r148, %r562, %r563, 0x6420U;
	prmt.b32 	%r149, %r564, %r565, 0x6420U;
	prmt.b32 	%r150, %r595, %r596, 0x6420U;
	prmt.b32 	%r151, %r597, %r598, 0x6420U;
	mov.b32 	%r152, %r111;
	mov.b32 	%r153, %r111;
	mov.b32 	%r154, %r111;
	mov.b32 	%r155, %r111;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r152, %r153, %r154, %r155 }, { %r112, %r113, %r114, %r115 }, { %r116, %r117 }, { %r152, %r153, %r154, %r155 };
	// end inline asm
	mov.b32 	%r156, %r111;
	mov.b32 	%r157, %r111;
	mov.b32 	%r158, %r111;
	mov.b32 	%r159, %r111;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r156, %r157, %r158, %r159 }, { %r112, %r113, %r114, %r115 }, { %r122, %r123 }, { %r156, %r157, %r158, %r159 };
	// end inline asm
	mov.b32 	%r164, %r111;
	mov.b32 	%r165, %r111;
	mov.b32 	%r166, %r111;
	mov.b32 	%r167, %r111;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r164, %r165, %r166, %r167 }, { %r112, %r113, %r114, %r115 }, { %r124, %r125 }, { %r164, %r165, %r166, %r167 };
	// end inline asm
	mov.b32 	%r168, %r111;
	mov.b32 	%r169, %r111;
	mov.b32 	%r170, %r111;
	mov.b32 	%r171, %r111;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r168, %r169, %r170, %r171 }, { %r112, %r113, %r114, %r115 }, { %r126, %r127 }, { %r168, %r169, %r170, %r171 };
	// end inline asm
	mov.b32 	%r172, %r111;
	mov.b32 	%r173, %r111;
	mov.b32 	%r174, %r111;
	mov.b32 	%r175, %r111;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r172, %r173, %r174, %r175 }, { %r118, %r119, %r120, %r121 }, { %r116, %r117 }, { %r172, %r173, %r174, %r175 };
	// end inline asm
	mov.b32 	%r178, %r111;
	mov.b32 	%r179, %r111;
	mov.b32 	%r180, %r111;
	mov.b32 	%r181, %r111;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r178, %r179, %r180, %r181 }, { %r118, %r119, %r120, %r121 }, { %r122, %r123 }, { %r178, %r179, %r180, %r181 };
	// end inline asm
	mov.b32 	%r188, %r111;
	mov.b32 	%r189, %r111;
	mov.b32 	%r190, %r111;
	mov.b32 	%r191, %r111;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r188, %r189, %r190, %r191 }, { %r118, %r119, %r120, %r121 }, { %r124, %r125 }, { %r188, %r189, %r190, %r191 };
	// end inline asm
	mov.b32 	%r194, %r111;
	mov.b32 	%r195, %r111;
	mov.b32 	%r196, %r111;
	mov.b32 	%r197, %r111;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r194, %r195, %r196, %r197 }, { %r118, %r119, %r120, %r121 }, { %r126, %r127 }, { %r194, %r195, %r196, %r197 };
	// end inline asm
	mov.b32 	%r200, %r111;
	mov.b32 	%r201, %r111;
	mov.b32 	%r202, %r111;
	mov.b32 	%r203, %r111;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r200, %r201, %r202, %r203 }, { %r128, %r129, %r130, %r131 }, { %r116, %r117 }, { %r200, %r201, %r202, %r203 };
	// end inline asm
	mov.b32 	%r204, %r111;
	mov.b32 	%r205, %r111;
	mov.b32 	%r206, %r111;
	mov.b32 	%r207, %r111;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r204, %r205, %r206, %r207 }, { %r128, %r129, %r130, %r131 }, { %r122, %r123 }, { %r204, %r205, %r206, %r207 };
	// end inline asm
	mov.b32 	%r212, %r111;
	mov.b32 	%r213, %r111;
	mov.b32 	%r214, %r111;
	mov.b32 	%r215, %r111;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r212, %r213, %r214, %r215 }, { %r128, %r129, %r130, %r131 }, { %r124, %r125 }, { %r212, %r213, %r214, %r215 };
	// end inline asm
	mov.b32 	%r216, %r111;
	mov.b32 	%r217, %r111;
	mov.b32 	%r218, %r111;
	mov.b32 	%r219, %r111;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r216, %r217, %r218, %r219 }, { %r128, %r129, %r130, %r131 }, { %r126, %r127 }, { %r216, %r217, %r218, %r219 };
	// end inline asm
	mov.b32 	%r220, %r111;
	mov.b32 	%r221, %r111;
	mov.b32 	%r222, %r111;
	mov.b32 	%r223, %r111;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r220, %r221, %r222, %r223 }, { %r132, %r133, %r134, %r135 }, { %r116, %r117 }, { %r220, %r221, %r222, %r223 };
	// end inline asm
	mov.b32 	%r224, %r111;
	mov.b32 	%r225, %r111;
	mov.b32 	%r226, %r111;
	mov.b32 	%r227, %r111;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r224, %r225, %r226, %r227 }, { %r132, %r133, %r134, %r135 }, { %r122, %r123 }, { %r224, %r225, %r226, %r227 };
	// end inline asm
	mov.b32 	%r232, %r111;
	mov.b32 	%r233, %r111;
	mov.b32 	%r234, %r111;
	mov.b32 	%r235, %r111;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r232, %r233, %r234, %r235 }, { %r132, %r133, %r134, %r135 }, { %r124, %r125 }, { %r232, %r233, %r234, %r235 };
	// end inline asm
	mov.b32 	%r236, %r111;
	mov.b32 	%r237, %r111;
	mov.b32 	%r238, %r111;
	mov.b32 	%r239, %r111;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r236, %r237, %r238, %r239 }, { %r132, %r133, %r134, %r135 }, { %r126, %r127 }, { %r236, %r237, %r238, %r239 };
	// end inline asm
	mov.b32 	%r240, %r111;
	mov.b32 	%r241, %r111;
	mov.b32 	%r242, %r111;
	mov.b32 	%r243, %r111;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r240, %r241, %r242, %r243 }, { %r136, %r137, %r138, %r139 }, { %r116, %r117 }, { %r240, %r241, %r242, %r243 };
	// end inline asm
	mov.b32 	%r244, %r111;
	mov.b32 	%r245, %r111;
	mov.b32 	%r246, %r111;
	mov.b32 	%r247, %r111;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r244, %r245, %r246, %r247 }, { %r136, %r137, %r138, %r139 }, { %r122, %r123 }, { %r244, %r245, %r246, %r247 };
	// end inline asm
	mov.b32 	%r252, %r111;
	mov.b32 	%r253, %r111;
	mov.b32 	%r254, %r111;
	mov.b32 	%r255, %r111;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r252, %r253, %r254, %r255 }, { %r136, %r137, %r138, %r139 }, { %r124, %r125 }, { %r252, %r253, %r254, %r255 };
	// end inline asm
	mov.b32 	%r256, %r111;
	mov.b32 	%r257, %r111;
	mov.b32 	%r258, %r111;
	mov.b32 	%r259, %r111;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r256, %r257, %r258, %r259 }, { %r136, %r137, %r138, %r139 }, { %r126, %r127 }, { %r256, %r257, %r258, %r259 };
	// end inline asm
	mov.b32 	%r260, %r111;
	mov.b32 	%r261, %r111;
	mov.b32 	%r262, %r111;
	mov.b32 	%r263, %r111;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r260, %r261, %r262, %r263 }, { %r140, %r141, %r142, %r143 }, { %r116, %r117 }, { %r260, %r261, %r262, %r263 };
	// end inline asm
	mov.b32 	%r264, %r111;
	mov.b32 	%r265, %r111;
	mov.b32 	%r266, %r111;
	mov.b32 	%r267, %r111;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r264, %r265, %r266, %r267 }, { %r140, %r141, %r142, %r143 }, { %r122, %r123 }, { %r264, %r265, %r266, %r267 };
	// end inline asm
	mov.b32 	%r272, %r111;
	mov.b32 	%r273, %r111;
	mov.b32 	%r274, %r111;
	mov.b32 	%r275, %r111;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r272, %r273, %r274, %r275 }, { %r140, %r141, %r142, %r143 }, { %r124, %r125 }, { %r272, %r273, %r274, %r275 };
	// end inline asm
	mov.b32 	%r276, %r111;
	mov.b32 	%r277, %r111;
	mov.b32 	%r278, %r111;
	mov.b32 	%r279, %r111;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r276, %r277, %r278, %r279 }, { %r140, %r141, %r142, %r143 }, { %r126, %r127 }, { %r276, %r277, %r278, %r279 };
	// end inline asm
	mov.b32 	%r280, %r111;
	mov.b32 	%r281, %r111;
	mov.b32 	%r282, %r111;
	mov.b32 	%r283, %r111;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r280, %r281, %r282, %r283 }, { %r144, %r145, %r146, %r147 }, { %r116, %r117 }, { %r280, %r281, %r282, %r283 };
	// end inline asm
	mov.b32 	%r284, %r111;
	mov.b32 	%r285, %r111;
	mov.b32 	%r286, %r111;
	mov.b32 	%r287, %r111;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r284, %r285, %r286, %r287 }, { %r144, %r145, %r146, %r147 }, { %r122, %r123 }, { %r284, %r285, %r286, %r287 };
	// end inline asm
	mov.b32 	%r292, %r111;
	mov.b32 	%r293, %r111;
	mov.b32 	%r294, %r111;
	mov.b32 	%r295, %r111;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r292, %r293, %r294, %r295 }, { %r144, %r145, %r146, %r147 }, { %r124, %r125 }, { %r292, %r293, %r294, %r295 };
	// end inline asm
	mov.b32 	%r296, %r111;
	mov.b32 	%r297, %r111;
	mov.b32 	%r298, %r111;
	mov.b32 	%r299, %r111;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r296, %r297, %r298, %r299 }, { %r144, %r145, %r146, %r147 }, { %r126, %r127 }, { %r296, %r297, %r298, %r299 };
	// end inline asm
	mov.b32 	%r300, %r111;
	mov.b32 	%r301, %r111;
	mov.b32 	%r302, %r111;
	mov.b32 	%r303, %r111;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r300, %r301, %r302, %r303 }, { %r148, %r149, %r150, %r151 }, { %r116, %r117 }, { %r300, %r301, %r302, %r303 };
	// end inline asm
	mov.b32 	%r304, %r111;
	mov.b32 	%r305, %r111;
	mov.b32 	%r306, %r111;
	mov.b32 	%r307, %r111;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r304, %r305, %r306, %r307 }, { %r148, %r149, %r150, %r151 }, { %r122, %r123 }, { %r304, %r305, %r306, %r307 };
	// end inline asm
	mov.b32 	%r312, %r111;
	mov.b32 	%r313, %r111;
	mov.b32 	%r314, %r111;
	mov.b32 	%r315, %r111;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r312, %r313, %r314, %r315 }, { %r148, %r149, %r150, %r151 }, { %r124, %r125 }, { %r312, %r313, %r314, %r315 };
	// end inline asm
	mov.b32 	%r316, %r111;
	mov.b32 	%r317, %r111;
	mov.b32 	%r318, %r111;
	mov.b32 	%r319, %r111;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r316, %r317, %r318, %r319 }, { %r148, %r149, %r150, %r151 }, { %r126, %r127 }, { %r316, %r317, %r318, %r319 };
	// end inline asm
	.loc	1 832 75                        // sk04_fa_o_w4a8.py:832:75
	prmt.b32 	%r160, %r534, %r535, 0x7531U;
	prmt.b32 	%r161, %r536, %r537, 0x7531U;
	prmt.b32 	%r162, %r567, %r568, 0x7531U;
	prmt.b32 	%r163, %r569, %r570, 0x7531U;
	prmt.b32 	%r182, %r538, %r539, 0x7531U;
	prmt.b32 	%r183, %r540, %r541, 0x7531U;
	prmt.b32 	%r184, %r571, %r572, 0x7531U;
	prmt.b32 	%r185, %r573, %r574, 0x7531U;
	prmt.b32 	%r208, %r542, %r543, 0x7531U;
	prmt.b32 	%r209, %r544, %r545, 0x7531U;
	prmt.b32 	%r210, %r575, %r576, 0x7531U;
	prmt.b32 	%r211, %r577, %r578, 0x7531U;
	prmt.b32 	%r228, %r546, %r547, 0x7531U;
	prmt.b32 	%r229, %r548, %r549, 0x7531U;
	prmt.b32 	%r230, %r579, %r580, 0x7531U;
	prmt.b32 	%r231, %r581, %r582, 0x7531U;
	prmt.b32 	%r248, %r550, %r551, 0x7531U;
	prmt.b32 	%r249, %r552, %r553, 0x7531U;
	prmt.b32 	%r250, %r583, %r584, 0x7531U;
	prmt.b32 	%r251, %r585, %r586, 0x7531U;
	prmt.b32 	%r268, %r554, %r555, 0x7531U;
	prmt.b32 	%r269, %r556, %r557, 0x7531U;
	prmt.b32 	%r270, %r587, %r588, 0x7531U;
	prmt.b32 	%r271, %r589, %r590, 0x7531U;
	prmt.b32 	%r288, %r558, %r559, 0x7531U;
	prmt.b32 	%r289, %r560, %r561, 0x7531U;
	prmt.b32 	%r290, %r591, %r592, 0x7531U;
	prmt.b32 	%r291, %r593, %r594, 0x7531U;
	prmt.b32 	%r308, %r562, %r563, 0x7531U;
	prmt.b32 	%r309, %r564, %r565, 0x7531U;
	prmt.b32 	%r310, %r595, %r596, 0x7531U;
	prmt.b32 	%r311, %r597, %r598, 0x7531U;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r152, %r153, %r154, %r155 }, { %r160, %r161, %r162, %r163 }, { %r176, %r177 }, { %r152, %r153, %r154, %r155 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r156, %r157, %r158, %r159 }, { %r160, %r161, %r162, %r163 }, { %r186, %r187 }, { %r156, %r157, %r158, %r159 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r164, %r165, %r166, %r167 }, { %r160, %r161, %r162, %r163 }, { %r192, %r193 }, { %r164, %r165, %r166, %r167 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r168, %r169, %r170, %r171 }, { %r160, %r161, %r162, %r163 }, { %r198, %r199 }, { %r168, %r169, %r170, %r171 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r172, %r173, %r174, %r175 }, { %r182, %r183, %r184, %r185 }, { %r176, %r177 }, { %r172, %r173, %r174, %r175 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r178, %r179, %r180, %r181 }, { %r182, %r183, %r184, %r185 }, { %r186, %r187 }, { %r178, %r179, %r180, %r181 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r188, %r189, %r190, %r191 }, { %r182, %r183, %r184, %r185 }, { %r192, %r193 }, { %r188, %r189, %r190, %r191 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r194, %r195, %r196, %r197 }, { %r182, %r183, %r184, %r185 }, { %r198, %r199 }, { %r194, %r195, %r196, %r197 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r200, %r201, %r202, %r203 }, { %r208, %r209, %r210, %r211 }, { %r176, %r177 }, { %r200, %r201, %r202, %r203 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r204, %r205, %r206, %r207 }, { %r208, %r209, %r210, %r211 }, { %r186, %r187 }, { %r204, %r205, %r206, %r207 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r212, %r213, %r214, %r215 }, { %r208, %r209, %r210, %r211 }, { %r192, %r193 }, { %r212, %r213, %r214, %r215 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r216, %r217, %r218, %r219 }, { %r208, %r209, %r210, %r211 }, { %r198, %r199 }, { %r216, %r217, %r218, %r219 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r220, %r221, %r222, %r223 }, { %r228, %r229, %r230, %r231 }, { %r176, %r177 }, { %r220, %r221, %r222, %r223 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r224, %r225, %r226, %r227 }, { %r228, %r229, %r230, %r231 }, { %r186, %r187 }, { %r224, %r225, %r226, %r227 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r232, %r233, %r234, %r235 }, { %r228, %r229, %r230, %r231 }, { %r192, %r193 }, { %r232, %r233, %r234, %r235 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r236, %r237, %r238, %r239 }, { %r228, %r229, %r230, %r231 }, { %r198, %r199 }, { %r236, %r237, %r238, %r239 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r240, %r241, %r242, %r243 }, { %r248, %r249, %r250, %r251 }, { %r176, %r177 }, { %r240, %r241, %r242, %r243 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r244, %r245, %r246, %r247 }, { %r248, %r249, %r250, %r251 }, { %r186, %r187 }, { %r244, %r245, %r246, %r247 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r252, %r253, %r254, %r255 }, { %r248, %r249, %r250, %r251 }, { %r192, %r193 }, { %r252, %r253, %r254, %r255 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r256, %r257, %r258, %r259 }, { %r248, %r249, %r250, %r251 }, { %r198, %r199 }, { %r256, %r257, %r258, %r259 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r260, %r261, %r262, %r263 }, { %r268, %r269, %r270, %r271 }, { %r176, %r177 }, { %r260, %r261, %r262, %r263 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r264, %r265, %r266, %r267 }, { %r268, %r269, %r270, %r271 }, { %r186, %r187 }, { %r264, %r265, %r266, %r267 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r272, %r273, %r274, %r275 }, { %r268, %r269, %r270, %r271 }, { %r192, %r193 }, { %r272, %r273, %r274, %r275 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r276, %r277, %r278, %r279 }, { %r268, %r269, %r270, %r271 }, { %r198, %r199 }, { %r276, %r277, %r278, %r279 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r280, %r281, %r282, %r283 }, { %r288, %r289, %r290, %r291 }, { %r176, %r177 }, { %r280, %r281, %r282, %r283 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r284, %r285, %r286, %r287 }, { %r288, %r289, %r290, %r291 }, { %r186, %r187 }, { %r284, %r285, %r286, %r287 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r292, %r293, %r294, %r295 }, { %r288, %r289, %r290, %r291 }, { %r192, %r193 }, { %r292, %r293, %r294, %r295 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r296, %r297, %r298, %r299 }, { %r288, %r289, %r290, %r291 }, { %r198, %r199 }, { %r296, %r297, %r298, %r299 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r300, %r301, %r302, %r303 }, { %r308, %r309, %r310, %r311 }, { %r176, %r177 }, { %r300, %r301, %r302, %r303 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r304, %r305, %r306, %r307 }, { %r308, %r309, %r310, %r311 }, { %r186, %r187 }, { %r304, %r305, %r306, %r307 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r312, %r313, %r314, %r315 }, { %r308, %r309, %r310, %r311 }, { %r192, %r193 }, { %r312, %r313, %r314, %r315 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r316, %r317, %r318, %r319 }, { %r308, %r309, %r310, %r311 }, { %r198, %r199 }, { %r316, %r317, %r318, %r319 };
	// end inline asm
	.loc	1 833 59                        // sk04_fa_o_w4a8.py:833:59
	mad.wide.s32 	%rd43, %r1405, 4, %rd10;
	.loc	1 833 27                        // sk04_fa_o_w4a8.py:833:27
	// begin inline asm
	mov.u32 %r320, 0x0;
	ld.global.b32 { %r320 }, [ %rd43 + 0 ];
	// end inline asm
	.loc	1 834 40                        // sk04_fa_o_w4a8.py:834:40
	// begin inline asm
	st.shared.b32 [ %r321 + 0 ], %r320;
	// end inline asm
	bar.sync 	0;
	.loc	1 834 26                        // sk04_fa_o_w4a8.py:834:26
	cvt.rn.f32.s32 	%r599, %r319;
	cvt.rn.f32.s32 	%r600, %r318;
	cvt.rn.f32.s32 	%r601, %r317;
	cvt.rn.f32.s32 	%r602, %r316;
	cvt.rn.f32.s32 	%r603, %r315;
	cvt.rn.f32.s32 	%r604, %r314;
	cvt.rn.f32.s32 	%r605, %r313;
	cvt.rn.f32.s32 	%r606, %r312;
	cvt.rn.f32.s32 	%r607, %r307;
	cvt.rn.f32.s32 	%r608, %r306;
	cvt.rn.f32.s32 	%r609, %r305;
	cvt.rn.f32.s32 	%r610, %r304;
	cvt.rn.f32.s32 	%r611, %r303;
	cvt.rn.f32.s32 	%r612, %r302;
	cvt.rn.f32.s32 	%r613, %r301;
	cvt.rn.f32.s32 	%r614, %r300;
	cvt.rn.f32.s32 	%r615, %r299;
	cvt.rn.f32.s32 	%r616, %r298;
	cvt.rn.f32.s32 	%r617, %r297;
	cvt.rn.f32.s32 	%r618, %r296;
	cvt.rn.f32.s32 	%r619, %r295;
	cvt.rn.f32.s32 	%r620, %r294;
	cvt.rn.f32.s32 	%r621, %r293;
	cvt.rn.f32.s32 	%r622, %r292;
	cvt.rn.f32.s32 	%r623, %r287;
	cvt.rn.f32.s32 	%r624, %r286;
	cvt.rn.f32.s32 	%r625, %r285;
	cvt.rn.f32.s32 	%r626, %r284;
	cvt.rn.f32.s32 	%r627, %r283;
	cvt.rn.f32.s32 	%r628, %r282;
	cvt.rn.f32.s32 	%r629, %r281;
	cvt.rn.f32.s32 	%r630, %r280;
	cvt.rn.f32.s32 	%r631, %r279;
	cvt.rn.f32.s32 	%r632, %r278;
	cvt.rn.f32.s32 	%r633, %r277;
	cvt.rn.f32.s32 	%r634, %r276;
	cvt.rn.f32.s32 	%r635, %r275;
	cvt.rn.f32.s32 	%r636, %r274;
	cvt.rn.f32.s32 	%r637, %r273;
	cvt.rn.f32.s32 	%r638, %r272;
	cvt.rn.f32.s32 	%r639, %r267;
	cvt.rn.f32.s32 	%r640, %r266;
	cvt.rn.f32.s32 	%r641, %r152;
	cvt.rn.f32.s32 	%r642, %r153;
	cvt.rn.f32.s32 	%r643, %r154;
	cvt.rn.f32.s32 	%r644, %r155;
	cvt.rn.f32.s32 	%r645, %r156;
	cvt.rn.f32.s32 	%r646, %r157;
	cvt.rn.f32.s32 	%r647, %r158;
	cvt.rn.f32.s32 	%r648, %r159;
	cvt.rn.f32.s32 	%r649, %r164;
	cvt.rn.f32.s32 	%r650, %r165;
	cvt.rn.f32.s32 	%r651, %r166;
	cvt.rn.f32.s32 	%r652, %r167;
	cvt.rn.f32.s32 	%r653, %r168;
	cvt.rn.f32.s32 	%r654, %r169;
	cvt.rn.f32.s32 	%r655, %r170;
	cvt.rn.f32.s32 	%r656, %r171;
	cvt.rn.f32.s32 	%r657, %r172;
	cvt.rn.f32.s32 	%r658, %r173;
	cvt.rn.f32.s32 	%r659, %r174;
	cvt.rn.f32.s32 	%r660, %r175;
	cvt.rn.f32.s32 	%r661, %r178;
	cvt.rn.f32.s32 	%r662, %r179;
	cvt.rn.f32.s32 	%r663, %r180;
	cvt.rn.f32.s32 	%r664, %r181;
	cvt.rn.f32.s32 	%r665, %r188;
	cvt.rn.f32.s32 	%r666, %r189;
	cvt.rn.f32.s32 	%r667, %r190;
	cvt.rn.f32.s32 	%r668, %r191;
	cvt.rn.f32.s32 	%r669, %r194;
	cvt.rn.f32.s32 	%r670, %r195;
	cvt.rn.f32.s32 	%r671, %r196;
	cvt.rn.f32.s32 	%r672, %r197;
	cvt.rn.f32.s32 	%r673, %r200;
	cvt.rn.f32.s32 	%r674, %r201;
	cvt.rn.f32.s32 	%r675, %r202;
	cvt.rn.f32.s32 	%r676, %r203;
	cvt.rn.f32.s32 	%r677, %r204;
	cvt.rn.f32.s32 	%r678, %r205;
	cvt.rn.f32.s32 	%r679, %r206;
	cvt.rn.f32.s32 	%r680, %r207;
	cvt.rn.f32.s32 	%r681, %r212;
	cvt.rn.f32.s32 	%r682, %r213;
	cvt.rn.f32.s32 	%r683, %r214;
	cvt.rn.f32.s32 	%r684, %r215;
	cvt.rn.f32.s32 	%r685, %r216;
	cvt.rn.f32.s32 	%r686, %r217;
	cvt.rn.f32.s32 	%r687, %r218;
	cvt.rn.f32.s32 	%r688, %r219;
	cvt.rn.f32.s32 	%r689, %r220;
	cvt.rn.f32.s32 	%r690, %r221;
	cvt.rn.f32.s32 	%r691, %r222;
	cvt.rn.f32.s32 	%r692, %r223;
	cvt.rn.f32.s32 	%r693, %r224;
	cvt.rn.f32.s32 	%r694, %r225;
	cvt.rn.f32.s32 	%r695, %r226;
	cvt.rn.f32.s32 	%r696, %r227;
	cvt.rn.f32.s32 	%r697, %r232;
	cvt.rn.f32.s32 	%r698, %r233;
	cvt.rn.f32.s32 	%r699, %r234;
	cvt.rn.f32.s32 	%r700, %r235;
	cvt.rn.f32.s32 	%r701, %r236;
	cvt.rn.f32.s32 	%r702, %r237;
	cvt.rn.f32.s32 	%r703, %r238;
	cvt.rn.f32.s32 	%r704, %r239;
	cvt.rn.f32.s32 	%r705, %r240;
	cvt.rn.f32.s32 	%r706, %r241;
	cvt.rn.f32.s32 	%r707, %r242;
	cvt.rn.f32.s32 	%r708, %r243;
	cvt.rn.f32.s32 	%r709, %r244;
	cvt.rn.f32.s32 	%r710, %r245;
	cvt.rn.f32.s32 	%r711, %r246;
	cvt.rn.f32.s32 	%r712, %r247;
	cvt.rn.f32.s32 	%r713, %r252;
	cvt.rn.f32.s32 	%r714, %r253;
	cvt.rn.f32.s32 	%r715, %r254;
	cvt.rn.f32.s32 	%r716, %r255;
	cvt.rn.f32.s32 	%r717, %r256;
	cvt.rn.f32.s32 	%r718, %r257;
	cvt.rn.f32.s32 	%r719, %r258;
	cvt.rn.f32.s32 	%r720, %r259;
	cvt.rn.f32.s32 	%r721, %r260;
	cvt.rn.f32.s32 	%r722, %r261;
	cvt.rn.f32.s32 	%r723, %r262;
	cvt.rn.f32.s32 	%r724, %r263;
	cvt.rn.f32.s32 	%r725, %r264;
	cvt.rn.f32.s32 	%r726, %r265;
	.loc	1 834 40                        // sk04_fa_o_w4a8.py:834:40
	ld.shared.v2.b32 	{%r727, %r728}, [%r29+32768];
	ld.shared.v2.b32 	{%r729, %r730}, [%r29+33024];
	ld.shared.v2.b32 	{%r731, %r732}, [%r29+33280];
	ld.shared.v2.b32 	{%r733, %r734}, [%r29+33536];
	.loc	1 834 15                        // sk04_fa_o_w4a8.py:834:15
	fma.rn.f32 	%r1493, %r730, %r726, %r1493;
	fma.rn.f32 	%r1492, %r729, %r725, %r1492;
	fma.rn.f32 	%r1491, %r728, %r724, %r1491;
	fma.rn.f32 	%r1490, %r727, %r723, %r1490;
	fma.rn.f32 	%r1489, %r728, %r722, %r1489;
	fma.rn.f32 	%r1488, %r727, %r721, %r1488;
	fma.rn.f32 	%r1487, %r734, %r720, %r1487;
	fma.rn.f32 	%r1486, %r733, %r719, %r1486;
	fma.rn.f32 	%r1485, %r734, %r718, %r1485;
	fma.rn.f32 	%r1484, %r733, %r717, %r1484;
	fma.rn.f32 	%r1483, %r732, %r716, %r1483;
	fma.rn.f32 	%r1482, %r731, %r715, %r1482;
	fma.rn.f32 	%r1481, %r732, %r714, %r1481;
	fma.rn.f32 	%r1480, %r731, %r713, %r1480;
	fma.rn.f32 	%r1479, %r730, %r712, %r1479;
	fma.rn.f32 	%r1478, %r729, %r711, %r1478;
	fma.rn.f32 	%r1477, %r730, %r710, %r1477;
	fma.rn.f32 	%r1476, %r729, %r709, %r1476;
	fma.rn.f32 	%r1475, %r728, %r708, %r1475;
	fma.rn.f32 	%r1474, %r727, %r707, %r1474;
	fma.rn.f32 	%r1473, %r728, %r706, %r1473;
	fma.rn.f32 	%r1472, %r727, %r705, %r1472;
	fma.rn.f32 	%r1471, %r734, %r704, %r1471;
	fma.rn.f32 	%r1470, %r733, %r703, %r1470;
	fma.rn.f32 	%r1469, %r734, %r702, %r1469;
	fma.rn.f32 	%r1468, %r733, %r701, %r1468;
	fma.rn.f32 	%r1467, %r732, %r700, %r1467;
	fma.rn.f32 	%r1466, %r731, %r699, %r1466;
	fma.rn.f32 	%r1465, %r732, %r698, %r1465;
	fma.rn.f32 	%r1464, %r731, %r697, %r1464;
	fma.rn.f32 	%r1463, %r730, %r696, %r1463;
	fma.rn.f32 	%r1462, %r729, %r695, %r1462;
	fma.rn.f32 	%r1461, %r730, %r694, %r1461;
	fma.rn.f32 	%r1460, %r729, %r693, %r1460;
	fma.rn.f32 	%r1459, %r728, %r692, %r1459;
	fma.rn.f32 	%r1458, %r727, %r691, %r1458;
	fma.rn.f32 	%r1457, %r728, %r690, %r1457;
	fma.rn.f32 	%r1456, %r727, %r689, %r1456;
	fma.rn.f32 	%r1455, %r734, %r688, %r1455;
	fma.rn.f32 	%r1454, %r733, %r687, %r1454;
	fma.rn.f32 	%r1453, %r734, %r686, %r1453;
	fma.rn.f32 	%r1452, %r733, %r685, %r1452;
	fma.rn.f32 	%r1451, %r732, %r684, %r1451;
	fma.rn.f32 	%r1450, %r731, %r683, %r1450;
	fma.rn.f32 	%r1449, %r732, %r682, %r1449;
	fma.rn.f32 	%r1448, %r731, %r681, %r1448;
	fma.rn.f32 	%r1447, %r730, %r680, %r1447;
	fma.rn.f32 	%r1446, %r729, %r679, %r1446;
	fma.rn.f32 	%r1445, %r730, %r678, %r1445;
	fma.rn.f32 	%r1444, %r729, %r677, %r1444;
	fma.rn.f32 	%r1443, %r728, %r676, %r1443;
	fma.rn.f32 	%r1442, %r727, %r675, %r1442;
	fma.rn.f32 	%r1441, %r728, %r674, %r1441;
	fma.rn.f32 	%r1440, %r727, %r673, %r1440;
	fma.rn.f32 	%r1439, %r734, %r672, %r1439;
	fma.rn.f32 	%r1438, %r733, %r671, %r1438;
	fma.rn.f32 	%r1437, %r734, %r670, %r1437;
	fma.rn.f32 	%r1436, %r733, %r669, %r1436;
	fma.rn.f32 	%r1435, %r732, %r668, %r1435;
	fma.rn.f32 	%r1434, %r731, %r667, %r1434;
	fma.rn.f32 	%r1433, %r732, %r666, %r1433;
	fma.rn.f32 	%r1432, %r731, %r665, %r1432;
	fma.rn.f32 	%r1431, %r730, %r664, %r1431;
	fma.rn.f32 	%r1430, %r729, %r663, %r1430;
	fma.rn.f32 	%r1429, %r730, %r662, %r1429;
	fma.rn.f32 	%r1428, %r729, %r661, %r1428;
	fma.rn.f32 	%r1427, %r728, %r660, %r1427;
	fma.rn.f32 	%r1426, %r727, %r659, %r1426;
	fma.rn.f32 	%r1425, %r728, %r658, %r1425;
	fma.rn.f32 	%r1424, %r727, %r657, %r1424;
	fma.rn.f32 	%r1423, %r734, %r656, %r1423;
	fma.rn.f32 	%r1422, %r733, %r655, %r1422;
	fma.rn.f32 	%r1421, %r734, %r654, %r1421;
	fma.rn.f32 	%r1420, %r733, %r653, %r1420;
	fma.rn.f32 	%r1419, %r732, %r652, %r1419;
	fma.rn.f32 	%r1418, %r731, %r651, %r1418;
	fma.rn.f32 	%r1417, %r732, %r650, %r1417;
	fma.rn.f32 	%r1416, %r731, %r649, %r1416;
	fma.rn.f32 	%r1415, %r730, %r648, %r1415;
	fma.rn.f32 	%r1414, %r729, %r647, %r1414;
	fma.rn.f32 	%r1413, %r730, %r646, %r1413;
	fma.rn.f32 	%r1412, %r729, %r645, %r1412;
	fma.rn.f32 	%r1411, %r728, %r644, %r1411;
	fma.rn.f32 	%r1410, %r727, %r643, %r1410;
	fma.rn.f32 	%r1409, %r728, %r642, %r1409;
	fma.rn.f32 	%r1408, %r727, %r641, %r1408;
	fma.rn.f32 	%r1494, %r729, %r640, %r1494;
	fma.rn.f32 	%r1495, %r730, %r639, %r1495;
	fma.rn.f32 	%r1496, %r731, %r638, %r1496;
	fma.rn.f32 	%r1497, %r732, %r637, %r1497;
	fma.rn.f32 	%r1498, %r731, %r636, %r1498;
	fma.rn.f32 	%r1499, %r732, %r635, %r1499;
	fma.rn.f32 	%r1500, %r733, %r634, %r1500;
	fma.rn.f32 	%r1501, %r734, %r633, %r1501;
	fma.rn.f32 	%r1502, %r733, %r632, %r1502;
	fma.rn.f32 	%r1503, %r734, %r631, %r1503;
	fma.rn.f32 	%r1504, %r727, %r630, %r1504;
	fma.rn.f32 	%r1505, %r728, %r629, %r1505;
	fma.rn.f32 	%r1506, %r727, %r628, %r1506;
	fma.rn.f32 	%r1507, %r728, %r627, %r1507;
	fma.rn.f32 	%r1508, %r729, %r626, %r1508;
	fma.rn.f32 	%r1509, %r730, %r625, %r1509;
	fma.rn.f32 	%r1510, %r729, %r624, %r1510;
	fma.rn.f32 	%r1511, %r730, %r623, %r1511;
	fma.rn.f32 	%r1512, %r731, %r622, %r1512;
	fma.rn.f32 	%r1513, %r732, %r621, %r1513;
	fma.rn.f32 	%r1514, %r731, %r620, %r1514;
	fma.rn.f32 	%r1515, %r732, %r619, %r1515;
	fma.rn.f32 	%r1516, %r733, %r618, %r1516;
	fma.rn.f32 	%r1517, %r734, %r617, %r1517;
	fma.rn.f32 	%r1518, %r733, %r616, %r1518;
	fma.rn.f32 	%r1519, %r734, %r615, %r1519;
	fma.rn.f32 	%r1520, %r727, %r614, %r1520;
	fma.rn.f32 	%r1521, %r728, %r613, %r1521;
	fma.rn.f32 	%r1522, %r727, %r612, %r1522;
	fma.rn.f32 	%r1523, %r728, %r611, %r1523;
	fma.rn.f32 	%r1524, %r729, %r610, %r1524;
	fma.rn.f32 	%r1525, %r730, %r609, %r1525;
	fma.rn.f32 	%r1526, %r729, %r608, %r1526;
	fma.rn.f32 	%r1527, %r730, %r607, %r1527;
	fma.rn.f32 	%r1528, %r731, %r606, %r1528;
	fma.rn.f32 	%r1529, %r732, %r605, %r1529;
	fma.rn.f32 	%r1530, %r731, %r604, %r1530;
	fma.rn.f32 	%r1531, %r732, %r603, %r1531;
	fma.rn.f32 	%r1532, %r733, %r602, %r1532;
	fma.rn.f32 	%r1533, %r734, %r601, %r1533;
	fma.rn.f32 	%r1534, %r733, %r600, %r1534;
	fma.rn.f32 	%r1535, %r734, %r599, %r1535;
	.loc	1 836 18                        // sk04_fa_o_w4a8.py:836:18
	add.s64 	%rd44, %rd132, %rd4;
	add.s64 	%rd46, %rd132, %rd5;
	add.s64 	%rd45, %rd132, %rd6;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	add.s64 	%rd47, %rd132, %rd7;
	add.s32 	%r735, %r1407, 1;
	setp.gt.s32 	%p6, %r735, 1;
	selp.b32 	%r1407, 0, %r735, %p6;
	.loc	1 827 25                        // sk04_fa_o_w4a8.py:827:25
	shl.b32 	%r736, %r1407, 13;
	add.s32 	%r737, %r95, %r736;
	add.s32 	%r322, %r36, %r736;
	selp.b32 	%r323, 8, 0, %p4;
	// begin inline asm
	cp.async.ca.shared.global [ %r322 + 0 ], [ %rd44 + 0 ], 0x8, %r323;
	// end inline asm
	add.s32 	%r324, %r322, 4096;
	// begin inline asm
	cp.async.ca.shared.global [ %r324 + 0 ], [ %rd45 + 0 ], 0x8, %r323;
	// end inline asm
	add.s32 	%r738, %r737, %r15;
	add.s32 	%r325, %r738, 2048;
	// begin inline asm
	cp.async.ca.shared.global [ %r325 + 0 ], [ %rd46 + 0 ], 0x8, %r323;
	// end inline asm
	add.s32 	%r326, %r738, 6144;
	// begin inline asm
	cp.async.ca.shared.global [ %r326 + 0 ], [ %rd47 + 0 ], 0x8, %r323;
	// end inline asm
	cp.async.commit_group;
	.loc	1 831 52                        // sk04_fa_o_w4a8.py:831:52
	add.s32 	%r739, %r18, %r736;
	add.s32 	%r327, %r739, 16384;
	selp.b32 	%r328, 16, 0, %p4;
	// begin inline asm
	cp.async.cg.shared.global [ %r327 + 0 ], [ %rd130 + 0 ], 0x10, %r328;
	// end inline asm
	add.s32 	%r329, %r739, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r329 + 0 ], [ %rd131 + 0 ], 0x10, %r328;
	// end inline asm
	cp.async.commit_group;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	add.s64 	%rd133, %rd133, 1;
	add.s32 	%r1405, %r1405, %r35;
	add.s64 	%rd132, %rd132, %rd9;
	add.s64 	%rd131, %rd131, 64;
	add.s64 	%rd130, %rd130, 64;
	setp.ne.b64 	%p7, %rd12, %rd133;
	@%p7 bra 	$L__BB0_2;
$L__BB0_3:                              // %._crit_edge
	.loc	1 816 45                        // sk04_fa_o_w4a8.py:816:45
	or.b32 	%r953, %r1, %r6;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r954, %r953, 120;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r955, %r954, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r956, %r953, 112;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r957, %r956, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r958, %r953, 104;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r959, %r958, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r960, %r953, 96;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r961, %r960, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r962, %r953, 88;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r963, %r962, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r964, %r953, 80;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r965, %r964, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r966, %r953, 72;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r967, %r966, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r968, %r953, 64;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r969, %r968, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r970, %r953, 56;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r971, %r970, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r972, %r953, 48;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r973, %r972, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r974, %r953, 40;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r975, %r974, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r976, %r953, 32;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r977, %r976, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r978, %r1, %r9;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r979, %r978, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r980, %r1, %r8;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r981, %r980, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r982, %r1, %r7;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r983, %r982, %r30;
	rem.s32 	%r984, %r953, %r30;
	.loc	1 816 45                        // sk04_fa_o_w4a8.py:816:45
	and.b32 	%r985, %r3, 7;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r986, %r985, %r1;
	or.b32 	%r987, %r986, 120;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r988, %r987, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r989, %r986, 112;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r990, %r989, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r991, %r986, 104;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r992, %r991, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r993, %r986, 96;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r994, %r993, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r995, %r986, 88;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r996, %r995, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r997, %r986, 80;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r998, %r997, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r999, %r986, 72;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1000, %r999, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1001, %r986, 64;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1002, %r1001, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1003, %r986, 56;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1004, %r1003, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1005, %r986, 48;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1006, %r1005, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1007, %r986, 40;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1008, %r1007, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1009, %r986, 32;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1010, %r1009, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1011, %r986, 24;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1012, %r1011, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1013, %r986, 16;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1014, %r1013, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1015, %r986, 8;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1016, %r1015, %r30;
	rem.s32 	%r1017, %r986, %r30;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 838 38                        // sk04_fa_o_w4a8.py:838:38
	mad.wide.s32 	%rd48, %r1017, 4, %rd17;
	mad.wide.s32 	%rd49, %r1016, 4, %rd17;
	mad.wide.s32 	%rd50, %r1014, 4, %rd17;
	mad.wide.s32 	%rd51, %r1012, 4, %rd17;
	mad.wide.s32 	%rd52, %r1010, 4, %rd17;
	mad.wide.s32 	%rd53, %r1008, 4, %rd17;
	mad.wide.s32 	%rd54, %r1006, 4, %rd17;
	mad.wide.s32 	%rd55, %r1004, 4, %rd17;
	mad.wide.s32 	%rd56, %r1002, 4, %rd17;
	mad.wide.s32 	%rd57, %r1000, 4, %rd17;
	mad.wide.s32 	%rd58, %r998, 4, %rd17;
	mad.wide.s32 	%rd59, %r996, 4, %rd17;
	mad.wide.s32 	%rd60, %r994, 4, %rd17;
	mad.wide.s32 	%rd61, %r992, 4, %rd17;
	mad.wide.s32 	%rd62, %r990, 4, %rd17;
	mad.wide.s32 	%rd63, %r988, 4, %rd17;
	.loc	1 838 24                        // sk04_fa_o_w4a8.py:838:24
	// begin inline asm
	mov.u32 %r740, 0x0;
	ld.global.b32 { %r740 }, [ %rd48 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r741, 0x0;
	ld.global.b32 { %r741 }, [ %rd49 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r742, 0x0;
	ld.global.b32 { %r742 }, [ %rd50 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r743, 0x0;
	ld.global.b32 { %r743 }, [ %rd51 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r744, 0x0;
	ld.global.b32 { %r744 }, [ %rd52 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r745, 0x0;
	ld.global.b32 { %r745 }, [ %rd53 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r746, 0x0;
	ld.global.b32 { %r746 }, [ %rd54 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r747, 0x0;
	ld.global.b32 { %r747 }, [ %rd55 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r748, 0x0;
	ld.global.b32 { %r748 }, [ %rd56 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r749, 0x0;
	ld.global.b32 { %r749 }, [ %rd57 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r750, 0x0;
	ld.global.b32 { %r750 }, [ %rd58 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r751, 0x0;
	ld.global.b32 { %r751 }, [ %rd59 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r752, 0x0;
	ld.global.b32 { %r752 }, [ %rd60 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r753, 0x0;
	ld.global.b32 { %r753 }, [ %rd61 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r754, 0x0;
	ld.global.b32 { %r754 }, [ %rd62 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r755, 0x0;
	ld.global.b32 { %r755 }, [ %rd63 + 0 ];
	// end inline asm
	.loc	1 839 49                        // sk04_fa_o_w4a8.py:839:49
	mul.lo.s32 	%r1018, %r984, %r34;
	mul.lo.s32 	%r1019, %r983, %r34;
	mul.lo.s32 	%r1020, %r981, %r34;
	mul.lo.s32 	%r1021, %r979, %r34;
	mul.lo.s32 	%r1022, %r977, %r34;
	mul.lo.s32 	%r1023, %r975, %r34;
	mul.lo.s32 	%r1024, %r973, %r34;
	mul.lo.s32 	%r1025, %r971, %r34;
	mul.lo.s32 	%r1026, %r969, %r34;
	mul.lo.s32 	%r1027, %r967, %r34;
	mul.lo.s32 	%r1028, %r965, %r34;
	mul.lo.s32 	%r1029, %r963, %r34;
	mul.lo.s32 	%r1030, %r961, %r34;
	mul.lo.s32 	%r1031, %r959, %r34;
	mul.lo.s32 	%r1032, %r957, %r34;
	mul.lo.s32 	%r1033, %r955, %r34;
	.loc	1 839 31                        // sk04_fa_o_w4a8.py:839:31
	mad.wide.s32 	%rd96, %r1018, 2, %rd16;
	mad.wide.s32 	%rd97, %r1019, 2, %rd16;
	mad.wide.s32 	%rd98, %r1020, 2, %rd16;
	mad.wide.s32 	%rd99, %r1021, 2, %rd16;
	mad.wide.s32 	%rd100, %r1022, 2, %rd16;
	mad.wide.s32 	%rd101, %r1023, 2, %rd16;
	mad.wide.s32 	%rd102, %r1024, 2, %rd16;
	mad.wide.s32 	%rd103, %r1025, 2, %rd16;
	mad.wide.s32 	%rd104, %r1026, 2, %rd16;
	mad.wide.s32 	%rd105, %r1027, 2, %rd16;
	mad.wide.s32 	%rd106, %r1028, 2, %rd16;
	mad.wide.s32 	%rd107, %r1029, 2, %rd16;
	mad.wide.s32 	%rd108, %r1030, 2, %rd16;
	mad.wide.s32 	%rd109, %r1031, 2, %rd16;
	mad.wide.s32 	%rd110, %r1032, 2, %rd16;
	mad.wide.s32 	%rd111, %r1033, 2, %rd16;
	.loc	1 839 64                        // sk04_fa_o_w4a8.py:839:64
	shl.b64 	%rd112, %rd8, 1;
	add.s64 	%rd64, %rd96, %rd112;
	add.s64 	%rd65, %rd97, %rd112;
	add.s64 	%rd66, %rd98, %rd112;
	add.s64 	%rd67, %rd99, %rd112;
	add.s64 	%rd68, %rd100, %rd112;
	add.s64 	%rd69, %rd101, %rd112;
	add.s64 	%rd70, %rd102, %rd112;
	add.s64 	%rd71, %rd103, %rd112;
	add.s64 	%rd72, %rd104, %rd112;
	add.s64 	%rd73, %rd105, %rd112;
	add.s64 	%rd74, %rd106, %rd112;
	add.s64 	%rd75, %rd107, %rd112;
	add.s64 	%rd76, %rd108, %rd112;
	add.s64 	%rd77, %rd109, %rd112;
	add.s64 	%rd78, %rd110, %rd112;
	add.s64 	%rd79, %rd111, %rd112;
	.loc	1 839 19                        // sk04_fa_o_w4a8.py:839:19
	// begin inline asm
	mov.u32 %r757, 0x0;
	mov.u32 %r758, 0x0;
	mov.u32 %r759, 0x0;
	mov.u32 %r760, 0x0;
	ld.global.v4.b32 { %r757, %r758, %r759, %r760 }, [ %rd64 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r761, 0x0;
	mov.u32 %r762, 0x0;
	mov.u32 %r763, 0x0;
	mov.u32 %r764, 0x0;
	ld.global.v4.b32 { %r761, %r762, %r763, %r764 }, [ %rd65 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r765, 0x0;
	mov.u32 %r766, 0x0;
	mov.u32 %r767, 0x0;
	mov.u32 %r768, 0x0;
	ld.global.v4.b32 { %r765, %r766, %r767, %r768 }, [ %rd66 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r769, 0x0;
	mov.u32 %r770, 0x0;
	mov.u32 %r771, 0x0;
	mov.u32 %r772, 0x0;
	ld.global.v4.b32 { %r769, %r770, %r771, %r772 }, [ %rd67 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r773, 0x0;
	mov.u32 %r774, 0x0;
	mov.u32 %r775, 0x0;
	mov.u32 %r776, 0x0;
	ld.global.v4.b32 { %r773, %r774, %r775, %r776 }, [ %rd68 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r777, 0x0;
	mov.u32 %r778, 0x0;
	mov.u32 %r779, 0x0;
	mov.u32 %r780, 0x0;
	ld.global.v4.b32 { %r777, %r778, %r779, %r780 }, [ %rd69 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r781, 0x0;
	mov.u32 %r782, 0x0;
	mov.u32 %r783, 0x0;
	mov.u32 %r784, 0x0;
	ld.global.v4.b32 { %r781, %r782, %r783, %r784 }, [ %rd70 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r785, 0x0;
	mov.u32 %r786, 0x0;
	mov.u32 %r787, 0x0;
	mov.u32 %r788, 0x0;
	ld.global.v4.b32 { %r785, %r786, %r787, %r788 }, [ %rd71 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r789, 0x0;
	mov.u32 %r790, 0x0;
	mov.u32 %r791, 0x0;
	mov.u32 %r792, 0x0;
	ld.global.v4.b32 { %r789, %r790, %r791, %r792 }, [ %rd72 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r793, 0x0;
	mov.u32 %r794, 0x0;
	mov.u32 %r795, 0x0;
	mov.u32 %r796, 0x0;
	ld.global.v4.b32 { %r793, %r794, %r795, %r796 }, [ %rd73 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r797, 0x0;
	mov.u32 %r798, 0x0;
	mov.u32 %r799, 0x0;
	mov.u32 %r800, 0x0;
	ld.global.v4.b32 { %r797, %r798, %r799, %r800 }, [ %rd74 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r801, 0x0;
	mov.u32 %r802, 0x0;
	mov.u32 %r803, 0x0;
	mov.u32 %r804, 0x0;
	ld.global.v4.b32 { %r801, %r802, %r803, %r804 }, [ %rd75 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r805, 0x0;
	mov.u32 %r806, 0x0;
	mov.u32 %r807, 0x0;
	mov.u32 %r808, 0x0;
	ld.global.v4.b32 { %r805, %r806, %r807, %r808 }, [ %rd76 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r809, 0x0;
	mov.u32 %r810, 0x0;
	mov.u32 %r811, 0x0;
	mov.u32 %r812, 0x0;
	ld.global.v4.b32 { %r809, %r810, %r811, %r812 }, [ %rd77 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r813, 0x0;
	mov.u32 %r814, 0x0;
	mov.u32 %r815, 0x0;
	mov.u32 %r816, 0x0;
	ld.global.v4.b32 { %r813, %r814, %r815, %r816 }, [ %rd78 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r817, 0x0;
	mov.u32 %r818, 0x0;
	mov.u32 %r819, 0x0;
	mov.u32 %r820, 0x0;
	ld.global.v4.b32 { %r817, %r818, %r819, %r820 }, [ %rd79 + 0 ];
	// end inline asm
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	shl.b32 	%r1034, %r12, 4;
	shr.u32 	%r1035, %r5, 1;
	xor.b32 	%r1036, %r1034, %r1035;
	add.s32 	%r756, %r95, %r1036;
	// begin inline asm
	st.shared.v4.b32 [ %r756 + 0 ], { %r757, %r758, %r759, %r760 };
	// end inline asm
	bar.sync 	0;
	and.b32 	%r1037, %r2, 7;
	shl.b32 	%r1038, %r1037, 9;
	shl.b32 	%r1039, %r11, 4;
	xor.b32 	%r1040, %r1039, %r1035;
	add.s32 	%r1041, %r95, %r1038;
	add.s32 	%r1042, %r1041, %r1040;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1043, %r1044, %r1045, %r1046}, [%r1042];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r756 + 0 ], { %r761, %r762, %r763, %r764 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1047, %r1048, %r1049, %r1050}, [%r1042];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r756 + 0 ], { %r765, %r766, %r767, %r768 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1051, %r1052, %r1053, %r1054}, [%r1042];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r756 + 0 ], { %r769, %r770, %r771, %r772 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1055, %r1056, %r1057, %r1058}, [%r1042];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r756 + 0 ], { %r773, %r774, %r775, %r776 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1059, %r1060, %r1061, %r1062}, [%r1042];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r756 + 0 ], { %r777, %r778, %r779, %r780 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1063, %r1064, %r1065, %r1066}, [%r1042];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r756 + 0 ], { %r781, %r782, %r783, %r784 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1067, %r1068, %r1069, %r1070}, [%r1042];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r756 + 0 ], { %r785, %r786, %r787, %r788 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1071, %r1072, %r1073, %r1074}, [%r1042];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r756 + 0 ], { %r789, %r790, %r791, %r792 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1075, %r1076, %r1077, %r1078}, [%r1042];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r756 + 0 ], { %r793, %r794, %r795, %r796 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1079, %r1080, %r1081, %r1082}, [%r1042];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r756 + 0 ], { %r797, %r798, %r799, %r800 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1083, %r1084, %r1085, %r1086}, [%r1042];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r756 + 0 ], { %r801, %r802, %r803, %r804 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1087, %r1088, %r1089, %r1090}, [%r1042];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r756 + 0 ], { %r805, %r806, %r807, %r808 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1091, %r1092, %r1093, %r1094}, [%r1042];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r756 + 0 ], { %r809, %r810, %r811, %r812 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1095, %r1096, %r1097, %r1098}, [%r1042];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r756 + 0 ], { %r813, %r814, %r815, %r816 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1099, %r1100, %r1101, %r1102}, [%r1042];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r756 + 0 ], { %r817, %r818, %r819, %r820 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1103, %r1104, %r1105, %r1106}, [%r1042];
	.loc	1 846 31                        // sk04_fa_o_w4a8.py:846:31
	setp.lt.s32 	%p24, %r953, %r30;
	setp.lt.s32 	%p25, %r982, %r30;
	setp.lt.s32 	%p26, %r980, %r30;
	setp.lt.s32 	%p27, %r978, %r30;
	setp.lt.s32 	%p28, %r976, %r30;
	setp.lt.s32 	%p29, %r974, %r30;
	setp.lt.s32 	%p30, %r972, %r30;
	setp.lt.s32 	%p31, %r970, %r30;
	setp.lt.s32 	%p32, %r968, %r30;
	setp.lt.s32 	%p33, %r966, %r30;
	setp.lt.s32 	%p34, %r964, %r30;
	setp.lt.s32 	%p35, %r962, %r30;
	setp.lt.s32 	%p36, %r960, %r30;
	setp.lt.s32 	%p37, %r958, %r30;
	setp.lt.s32 	%p38, %r956, %r30;
	setp.lt.s32 	%p39, %r954, %r30;
	.loc	1 846 54                        // sk04_fa_o_w4a8.py:846:54
	setp.lt.s32 	%p40, %r13, %r31;
	.loc	1 846 37                        // sk04_fa_o_w4a8.py:846:37
	and.pred 	%p8, %p24, %p40;
	and.pred 	%p9, %p25, %p40;
	and.pred 	%p10, %p26, %p40;
	and.pred 	%p11, %p27, %p40;
	and.pred 	%p12, %p28, %p40;
	and.pred 	%p13, %p29, %p40;
	and.pred 	%p14, %p30, %p40;
	and.pred 	%p15, %p31, %p40;
	and.pred 	%p16, %p32, %p40;
	and.pred 	%p17, %p33, %p40;
	and.pred 	%p18, %p34, %p40;
	and.pred 	%p19, %p35, %p40;
	and.pred 	%p20, %p36, %p40;
	and.pred 	%p21, %p37, %p40;
	and.pred 	%p22, %p38, %p40;
	and.pred 	%p23, %p39, %p40;
	.loc	1 844 35                        // sk04_fa_o_w4a8.py:844:35
	mul.lo.s32 	%r1107, %r953, %r33;
	mul.lo.s32 	%r1108, %r982, %r33;
	mul.lo.s32 	%r1109, %r980, %r33;
	mul.lo.s32 	%r1110, %r978, %r33;
	mul.lo.s32 	%r1111, %r33, %r976;
	mul.lo.s32 	%r1112, %r33, %r974;
	mul.lo.s32 	%r1113, %r33, %r972;
	mul.lo.s32 	%r1114, %r33, %r970;
	mul.lo.s32 	%r1115, %r33, %r968;
	mul.lo.s32 	%r1116, %r33, %r966;
	mul.lo.s32 	%r1117, %r33, %r964;
	mul.lo.s32 	%r1118, %r33, %r962;
	mul.lo.s32 	%r1119, %r33, %r960;
	mul.lo.s32 	%r1120, %r33, %r958;
	mul.lo.s32 	%r1121, %r33, %r956;
	mul.lo.s32 	%r1122, %r33, %r954;
	.loc	1 844 18                        // sk04_fa_o_w4a8.py:844:18
	mad.wide.s32 	%rd113, %r1107, 2, %rd15;
	mad.wide.s32 	%rd114, %r1108, 2, %rd15;
	mad.wide.s32 	%rd115, %r1109, 2, %rd15;
	mad.wide.s32 	%rd116, %r1110, 2, %rd15;
	mad.wide.s32 	%rd117, %r1111, 2, %rd15;
	mad.wide.s32 	%rd118, %r1112, 2, %rd15;
	mad.wide.s32 	%rd119, %r1113, 2, %rd15;
	mad.wide.s32 	%rd120, %r1114, 2, %rd15;
	mad.wide.s32 	%rd121, %r1115, 2, %rd15;
	mad.wide.s32 	%rd122, %r1116, 2, %rd15;
	mad.wide.s32 	%rd123, %r1117, 2, %rd15;
	mad.wide.s32 	%rd124, %r1118, 2, %rd15;
	mad.wide.s32 	%rd125, %r1119, 2, %rd15;
	mad.wide.s32 	%rd126, %r1120, 2, %rd15;
	mad.wide.s32 	%rd127, %r1121, 2, %rd15;
	mad.wide.s32 	%rd128, %r1122, 2, %rd15;
	.loc	1 844 50                        // sk04_fa_o_w4a8.py:844:50
	mul.wide.s32 	%rd129, %r13, 2;
	add.s64 	%rd80, %rd113, %rd129;
	add.s64 	%rd81, %rd114, %rd129;
	add.s64 	%rd82, %rd115, %rd129;
	add.s64 	%rd83, %rd116, %rd129;
	add.s64 	%rd84, %rd117, %rd129;
	add.s64 	%rd85, %rd118, %rd129;
	add.s64 	%rd86, %rd119, %rd129;
	add.s64 	%rd87, %rd120, %rd129;
	add.s64 	%rd88, %rd121, %rd129;
	add.s64 	%rd89, %rd122, %rd129;
	add.s64 	%rd90, %rd123, %rd129;
	add.s64 	%rd91, %rd124, %rd129;
	add.s64 	%rd92, %rd125, %rd129;
	add.s64 	%rd93, %rd126, %rd129;
	add.s64 	%rd94, %rd127, %rd129;
	add.s64 	%rd95, %rd128, %rd129;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs161, %rs162}, %r1043;
	cvt.f32.bf16 	%r1123, %rs162;
	cvt.f32.bf16 	%r1124, %rs161;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1125, %r1408, %r740, %r1124;
	fma.rn.f32 	%r1126, %r1409, %r740, %r1123;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r822, %r1126, %r1125;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs163, %rs164}, %r1047;
	cvt.f32.bf16 	%r1127, %rs164;
	cvt.f32.bf16 	%r1128, %rs163;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1129, %r1410, %r741, %r1128;
	fma.rn.f32 	%r1130, %r1411, %r741, %r1127;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r823, %r1130, %r1129;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs165, %rs166}, %r1044;
	cvt.f32.bf16 	%r1131, %rs166;
	cvt.f32.bf16 	%r1132, %rs165;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1133, %r1412, %r740, %r1132;
	fma.rn.f32 	%r1134, %r1413, %r740, %r1131;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r827, %r1134, %r1133;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs167, %rs168}, %r1048;
	cvt.f32.bf16 	%r1135, %rs168;
	cvt.f32.bf16 	%r1136, %rs167;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1137, %r1414, %r741, %r1136;
	fma.rn.f32 	%r1138, %r1415, %r741, %r1135;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r828, %r1138, %r1137;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs169, %rs170}, %r1045;
	cvt.f32.bf16 	%r1139, %rs170;
	cvt.f32.bf16 	%r1140, %rs169;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1141, %r1416, %r740, %r1140;
	fma.rn.f32 	%r1142, %r1417, %r740, %r1139;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r832, %r1142, %r1141;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs171, %rs172}, %r1049;
	cvt.f32.bf16 	%r1143, %rs172;
	cvt.f32.bf16 	%r1144, %rs171;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1145, %r1418, %r741, %r1144;
	fma.rn.f32 	%r1146, %r1419, %r741, %r1143;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r833, %r1146, %r1145;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs173, %rs174}, %r1046;
	cvt.f32.bf16 	%r1147, %rs174;
	cvt.f32.bf16 	%r1148, %rs173;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1149, %r1420, %r740, %r1148;
	fma.rn.f32 	%r1150, %r1421, %r740, %r1147;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r837, %r1150, %r1149;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs175, %rs176}, %r1050;
	cvt.f32.bf16 	%r1151, %rs176;
	cvt.f32.bf16 	%r1152, %rs175;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1153, %r1422, %r741, %r1152;
	fma.rn.f32 	%r1154, %r1423, %r741, %r1151;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r838, %r1154, %r1153;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs177, %rs178}, %r1051;
	cvt.f32.bf16 	%r1155, %rs178;
	cvt.f32.bf16 	%r1156, %rs177;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1157, %r1424, %r742, %r1156;
	fma.rn.f32 	%r1158, %r1425, %r742, %r1155;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r824, %r1158, %r1157;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs179, %rs180}, %r1055;
	cvt.f32.bf16 	%r1159, %rs180;
	cvt.f32.bf16 	%r1160, %rs179;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1161, %r1426, %r743, %r1160;
	fma.rn.f32 	%r1162, %r1427, %r743, %r1159;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r825, %r1162, %r1161;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs181, %rs182}, %r1052;
	cvt.f32.bf16 	%r1163, %rs182;
	cvt.f32.bf16 	%r1164, %rs181;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1165, %r1428, %r742, %r1164;
	fma.rn.f32 	%r1166, %r1429, %r742, %r1163;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r829, %r1166, %r1165;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs183, %rs184}, %r1056;
	cvt.f32.bf16 	%r1167, %rs184;
	cvt.f32.bf16 	%r1168, %rs183;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1169, %r1430, %r743, %r1168;
	fma.rn.f32 	%r1170, %r1431, %r743, %r1167;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r830, %r1170, %r1169;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs185, %rs186}, %r1053;
	cvt.f32.bf16 	%r1171, %rs186;
	cvt.f32.bf16 	%r1172, %rs185;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1173, %r1432, %r742, %r1172;
	fma.rn.f32 	%r1174, %r1433, %r742, %r1171;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r834, %r1174, %r1173;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs187, %rs188}, %r1057;
	cvt.f32.bf16 	%r1175, %rs188;
	cvt.f32.bf16 	%r1176, %rs187;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1177, %r1434, %r743, %r1176;
	fma.rn.f32 	%r1178, %r1435, %r743, %r1175;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r835, %r1178, %r1177;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs189, %rs190}, %r1054;
	cvt.f32.bf16 	%r1179, %rs190;
	cvt.f32.bf16 	%r1180, %rs189;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1181, %r1436, %r742, %r1180;
	fma.rn.f32 	%r1182, %r1437, %r742, %r1179;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r839, %r1182, %r1181;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs191, %rs192}, %r1058;
	cvt.f32.bf16 	%r1183, %rs192;
	cvt.f32.bf16 	%r1184, %rs191;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1185, %r1438, %r743, %r1184;
	fma.rn.f32 	%r1186, %r1439, %r743, %r1183;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r840, %r1186, %r1185;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs193, %rs194}, %r1059;
	cvt.f32.bf16 	%r1187, %rs194;
	cvt.f32.bf16 	%r1188, %rs193;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1189, %r1440, %r744, %r1188;
	fma.rn.f32 	%r1190, %r1441, %r744, %r1187;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r841, %r1190, %r1189;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs195, %rs196}, %r1063;
	cvt.f32.bf16 	%r1191, %rs196;
	cvt.f32.bf16 	%r1192, %rs195;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1193, %r1442, %r745, %r1192;
	fma.rn.f32 	%r1194, %r1443, %r745, %r1191;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r842, %r1194, %r1193;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs197, %rs198}, %r1060;
	cvt.f32.bf16 	%r1195, %rs198;
	cvt.f32.bf16 	%r1196, %rs197;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1197, %r1444, %r744, %r1196;
	fma.rn.f32 	%r1198, %r1445, %r744, %r1195;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r845, %r1198, %r1197;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs199, %rs200}, %r1064;
	cvt.f32.bf16 	%r1199, %rs200;
	cvt.f32.bf16 	%r1200, %rs199;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1201, %r1446, %r745, %r1200;
	fma.rn.f32 	%r1202, %r1447, %r745, %r1199;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r846, %r1202, %r1201;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs201, %rs202}, %r1061;
	cvt.f32.bf16 	%r1203, %rs202;
	cvt.f32.bf16 	%r1204, %rs201;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1205, %r1448, %r744, %r1204;
	fma.rn.f32 	%r1206, %r1449, %r744, %r1203;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r849, %r1206, %r1205;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs203, %rs204}, %r1065;
	cvt.f32.bf16 	%r1207, %rs204;
	cvt.f32.bf16 	%r1208, %rs203;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1209, %r1450, %r745, %r1208;
	fma.rn.f32 	%r1210, %r1451, %r745, %r1207;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r850, %r1210, %r1209;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs205, %rs206}, %r1062;
	cvt.f32.bf16 	%r1211, %rs206;
	cvt.f32.bf16 	%r1212, %rs205;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1213, %r1452, %r744, %r1212;
	fma.rn.f32 	%r1214, %r1453, %r744, %r1211;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r853, %r1214, %r1213;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs207, %rs208}, %r1066;
	cvt.f32.bf16 	%r1215, %rs208;
	cvt.f32.bf16 	%r1216, %rs207;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1217, %r1454, %r745, %r1216;
	fma.rn.f32 	%r1218, %r1455, %r745, %r1215;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r854, %r1218, %r1217;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs209, %rs210}, %r1067;
	cvt.f32.bf16 	%r1219, %rs210;
	cvt.f32.bf16 	%r1220, %rs209;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1221, %r1456, %r746, %r1220;
	fma.rn.f32 	%r1222, %r1457, %r746, %r1219;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r843, %r1222, %r1221;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs211, %rs212}, %r1071;
	cvt.f32.bf16 	%r1223, %rs212;
	cvt.f32.bf16 	%r1224, %rs211;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1225, %r1458, %r747, %r1224;
	fma.rn.f32 	%r1226, %r1459, %r747, %r1223;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r844, %r1226, %r1225;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs213, %rs214}, %r1068;
	cvt.f32.bf16 	%r1227, %rs214;
	cvt.f32.bf16 	%r1228, %rs213;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1229, %r1460, %r746, %r1228;
	fma.rn.f32 	%r1230, %r1461, %r746, %r1227;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r847, %r1230, %r1229;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs215, %rs216}, %r1072;
	cvt.f32.bf16 	%r1231, %rs216;
	cvt.f32.bf16 	%r1232, %rs215;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1233, %r1462, %r747, %r1232;
	fma.rn.f32 	%r1234, %r1463, %r747, %r1231;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r848, %r1234, %r1233;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs217, %rs218}, %r1069;
	cvt.f32.bf16 	%r1235, %rs218;
	cvt.f32.bf16 	%r1236, %rs217;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1237, %r1464, %r746, %r1236;
	fma.rn.f32 	%r1238, %r1465, %r746, %r1235;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r851, %r1238, %r1237;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs219, %rs220}, %r1073;
	cvt.f32.bf16 	%r1239, %rs220;
	cvt.f32.bf16 	%r1240, %rs219;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1241, %r1466, %r747, %r1240;
	fma.rn.f32 	%r1242, %r1467, %r747, %r1239;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r852, %r1242, %r1241;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs221, %rs222}, %r1070;
	cvt.f32.bf16 	%r1243, %rs222;
	cvt.f32.bf16 	%r1244, %rs221;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1245, %r1468, %r746, %r1244;
	fma.rn.f32 	%r1246, %r1469, %r746, %r1243;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r855, %r1246, %r1245;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs223, %rs224}, %r1074;
	cvt.f32.bf16 	%r1247, %rs224;
	cvt.f32.bf16 	%r1248, %rs223;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1249, %r1470, %r747, %r1248;
	fma.rn.f32 	%r1250, %r1471, %r747, %r1247;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r856, %r1250, %r1249;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs225, %rs226}, %r1075;
	cvt.f32.bf16 	%r1251, %rs226;
	cvt.f32.bf16 	%r1252, %rs225;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1253, %r1472, %r748, %r1252;
	fma.rn.f32 	%r1254, %r1473, %r748, %r1251;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r857, %r1254, %r1253;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs227, %rs228}, %r1079;
	cvt.f32.bf16 	%r1255, %rs228;
	cvt.f32.bf16 	%r1256, %rs227;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1257, %r1474, %r749, %r1256;
	fma.rn.f32 	%r1258, %r1475, %r749, %r1255;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r858, %r1258, %r1257;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs229, %rs230}, %r1076;
	cvt.f32.bf16 	%r1259, %rs230;
	cvt.f32.bf16 	%r1260, %rs229;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1261, %r1476, %r748, %r1260;
	fma.rn.f32 	%r1262, %r1477, %r748, %r1259;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r861, %r1262, %r1261;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs231, %rs232}, %r1080;
	cvt.f32.bf16 	%r1263, %rs232;
	cvt.f32.bf16 	%r1264, %rs231;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1265, %r1478, %r749, %r1264;
	fma.rn.f32 	%r1266, %r1479, %r749, %r1263;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r862, %r1266, %r1265;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs233, %rs234}, %r1077;
	cvt.f32.bf16 	%r1267, %rs234;
	cvt.f32.bf16 	%r1268, %rs233;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1269, %r1480, %r748, %r1268;
	fma.rn.f32 	%r1270, %r1481, %r748, %r1267;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r865, %r1270, %r1269;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs235, %rs236}, %r1081;
	cvt.f32.bf16 	%r1271, %rs236;
	cvt.f32.bf16 	%r1272, %rs235;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1273, %r1482, %r749, %r1272;
	fma.rn.f32 	%r1274, %r1483, %r749, %r1271;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r866, %r1274, %r1273;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs237, %rs238}, %r1078;
	cvt.f32.bf16 	%r1275, %rs238;
	cvt.f32.bf16 	%r1276, %rs237;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1277, %r1484, %r748, %r1276;
	fma.rn.f32 	%r1278, %r1485, %r748, %r1275;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r869, %r1278, %r1277;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs239, %rs240}, %r1082;
	cvt.f32.bf16 	%r1279, %rs240;
	cvt.f32.bf16 	%r1280, %rs239;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1281, %r1486, %r749, %r1280;
	fma.rn.f32 	%r1282, %r1487, %r749, %r1279;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r870, %r1282, %r1281;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs241, %rs242}, %r1083;
	cvt.f32.bf16 	%r1283, %rs242;
	cvt.f32.bf16 	%r1284, %rs241;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1285, %r1488, %r750, %r1284;
	fma.rn.f32 	%r1286, %r1489, %r750, %r1283;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r859, %r1286, %r1285;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs243, %rs244}, %r1087;
	cvt.f32.bf16 	%r1287, %rs244;
	cvt.f32.bf16 	%r1288, %rs243;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1289, %r1490, %r751, %r1288;
	fma.rn.f32 	%r1290, %r1491, %r751, %r1287;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r860, %r1290, %r1289;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs245, %rs246}, %r1084;
	cvt.f32.bf16 	%r1291, %rs246;
	cvt.f32.bf16 	%r1292, %rs245;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1293, %r1492, %r750, %r1292;
	fma.rn.f32 	%r1294, %r1493, %r750, %r1291;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r863, %r1294, %r1293;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs247, %rs248}, %r1088;
	cvt.f32.bf16 	%r1295, %rs248;
	cvt.f32.bf16 	%r1296, %rs247;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1297, %r1494, %r751, %r1296;
	fma.rn.f32 	%r1298, %r1495, %r751, %r1295;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r864, %r1298, %r1297;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs249, %rs250}, %r1085;
	cvt.f32.bf16 	%r1299, %rs250;
	cvt.f32.bf16 	%r1300, %rs249;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1301, %r1496, %r750, %r1300;
	fma.rn.f32 	%r1302, %r1497, %r750, %r1299;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r867, %r1302, %r1301;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs251, %rs252}, %r1089;
	cvt.f32.bf16 	%r1303, %rs252;
	cvt.f32.bf16 	%r1304, %rs251;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1305, %r1498, %r751, %r1304;
	fma.rn.f32 	%r1306, %r1499, %r751, %r1303;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r868, %r1306, %r1305;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs253, %rs254}, %r1086;
	cvt.f32.bf16 	%r1307, %rs254;
	cvt.f32.bf16 	%r1308, %rs253;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1309, %r1500, %r750, %r1308;
	fma.rn.f32 	%r1310, %r1501, %r750, %r1307;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r871, %r1310, %r1309;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs255, %rs256}, %r1090;
	cvt.f32.bf16 	%r1311, %rs256;
	cvt.f32.bf16 	%r1312, %rs255;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1313, %r1502, %r751, %r1312;
	fma.rn.f32 	%r1314, %r1503, %r751, %r1311;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r872, %r1314, %r1313;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs257, %rs258}, %r1091;
	cvt.f32.bf16 	%r1315, %rs258;
	cvt.f32.bf16 	%r1316, %rs257;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1317, %r1504, %r752, %r1316;
	fma.rn.f32 	%r1318, %r1505, %r752, %r1315;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r873, %r1318, %r1317;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs259, %rs260}, %r1095;
	cvt.f32.bf16 	%r1319, %rs260;
	cvt.f32.bf16 	%r1320, %rs259;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1321, %r1506, %r753, %r1320;
	fma.rn.f32 	%r1322, %r1507, %r753, %r1319;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r874, %r1322, %r1321;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs261, %rs262}, %r1092;
	cvt.f32.bf16 	%r1323, %rs262;
	cvt.f32.bf16 	%r1324, %rs261;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1325, %r1508, %r752, %r1324;
	fma.rn.f32 	%r1326, %r1509, %r752, %r1323;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r877, %r1326, %r1325;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs263, %rs264}, %r1096;
	cvt.f32.bf16 	%r1327, %rs264;
	cvt.f32.bf16 	%r1328, %rs263;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1329, %r1510, %r753, %r1328;
	fma.rn.f32 	%r1330, %r1511, %r753, %r1327;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r878, %r1330, %r1329;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs265, %rs266}, %r1093;
	cvt.f32.bf16 	%r1331, %rs266;
	cvt.f32.bf16 	%r1332, %rs265;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1333, %r1512, %r752, %r1332;
	fma.rn.f32 	%r1334, %r1513, %r752, %r1331;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r881, %r1334, %r1333;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs267, %rs268}, %r1097;
	cvt.f32.bf16 	%r1335, %rs268;
	cvt.f32.bf16 	%r1336, %rs267;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1337, %r1514, %r753, %r1336;
	fma.rn.f32 	%r1338, %r1515, %r753, %r1335;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r882, %r1338, %r1337;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs269, %rs270}, %r1094;
	cvt.f32.bf16 	%r1339, %rs270;
	cvt.f32.bf16 	%r1340, %rs269;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1341, %r1516, %r752, %r1340;
	fma.rn.f32 	%r1342, %r1517, %r752, %r1339;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r885, %r1342, %r1341;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs271, %rs272}, %r1098;
	cvt.f32.bf16 	%r1343, %rs272;
	cvt.f32.bf16 	%r1344, %rs271;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1345, %r1518, %r753, %r1344;
	fma.rn.f32 	%r1346, %r1519, %r753, %r1343;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r886, %r1346, %r1345;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs273, %rs274}, %r1099;
	cvt.f32.bf16 	%r1347, %rs274;
	cvt.f32.bf16 	%r1348, %rs273;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1349, %r1520, %r754, %r1348;
	fma.rn.f32 	%r1350, %r1521, %r754, %r1347;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r875, %r1350, %r1349;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs275, %rs276}, %r1103;
	cvt.f32.bf16 	%r1351, %rs276;
	cvt.f32.bf16 	%r1352, %rs275;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1353, %r1522, %r755, %r1352;
	fma.rn.f32 	%r1354, %r1523, %r755, %r1351;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r876, %r1354, %r1353;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs277, %rs278}, %r1100;
	cvt.f32.bf16 	%r1355, %rs278;
	cvt.f32.bf16 	%r1356, %rs277;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1357, %r1524, %r754, %r1356;
	fma.rn.f32 	%r1358, %r1525, %r754, %r1355;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r879, %r1358, %r1357;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs279, %rs280}, %r1104;
	cvt.f32.bf16 	%r1359, %rs280;
	cvt.f32.bf16 	%r1360, %rs279;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1361, %r1526, %r755, %r1360;
	fma.rn.f32 	%r1362, %r1527, %r755, %r1359;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r880, %r1362, %r1361;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs281, %rs282}, %r1101;
	cvt.f32.bf16 	%r1363, %rs282;
	cvt.f32.bf16 	%r1364, %rs281;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1365, %r1528, %r754, %r1364;
	fma.rn.f32 	%r1366, %r1529, %r754, %r1363;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r883, %r1366, %r1365;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs283, %rs284}, %r1105;
	cvt.f32.bf16 	%r1367, %rs284;
	cvt.f32.bf16 	%r1368, %rs283;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1369, %r1530, %r755, %r1368;
	fma.rn.f32 	%r1370, %r1531, %r755, %r1367;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r884, %r1370, %r1369;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs285, %rs286}, %r1102;
	cvt.f32.bf16 	%r1371, %rs286;
	cvt.f32.bf16 	%r1372, %rs285;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1373, %r1532, %r754, %r1372;
	fma.rn.f32 	%r1374, %r1533, %r754, %r1371;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r887, %r1374, %r1373;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs287, %rs288}, %r1106;
	cvt.f32.bf16 	%r1375, %rs288;
	cvt.f32.bf16 	%r1376, %rs287;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1377, %r1534, %r755, %r1376;
	fma.rn.f32 	%r1378, %r1535, %r755, %r1375;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r888, %r1378, %r1377;
	bar.sync 	0;
	shl.b32 	%r1379, %r14, 12;
	shl.b32 	%r1380, %r14, 5;
	and.b32 	%r1381, %r2, 24;
	shl.b32 	%r1382, %r1381, 4;
	bfe.s32 	%r1383, %r2, 2, 1;
	and.b32 	%r1384, %r1383, 2064;
	or.b32 	%r1385, %r1380, %r1382;
	or.b32 	%r1386, %r1384, %r1385;
	xor.b32 	%r1387, %r1386, %r1035;
	add.s32 	%r1388, %r95, %r1379;
	add.s32 	%r821, %r1388, %r1387;
	// begin inline asm
	st.shared.v4.b32 [ %r821 + 0 ], { %r822, %r823, %r824, %r825 };
	// end inline asm
	add.s32 	%r826, %r821, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r826 + 0 ], { %r827, %r828, %r829, %r830 };
	// end inline asm
	add.s32 	%r831, %r821, 1024;
	// begin inline asm
	st.shared.v4.b32 [ %r831 + 0 ], { %r832, %r833, %r834, %r835 };
	// end inline asm
	add.s32 	%r836, %r821, 1536;
	// begin inline asm
	st.shared.v4.b32 [ %r836 + 0 ], { %r837, %r838, %r839, %r840 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r1389, %r1381, 6;
	shl.b32 	%r1390, %r1037, 4;
	shl.b32 	%r1391, %r2, 1;
	and.b32 	%r1392, %r1391, 384;
	bfe.s32 	%r1393, %r2, 5, 1;
	and.b32 	%r1394, %r1393, 2064;
	or.b32 	%r1395, %r1389, %r1390;
	or.b32 	%r1396, %r1394, %r1392;
	xor.b32 	%r1397, %r1396, %r1395;
	add.s32 	%r1398, %r95, %r1397;
	ld.shared.v4.b32 	{%r889, %r893, %r897, %r901}, [%r1398];
	xor.b32 	%r1399, %r1397, 32;
	add.s32 	%r1400, %r95, %r1399;
	ld.shared.v4.b32 	{%r890, %r894, %r898, %r902}, [%r1400+4096];
	xor.b32 	%r1401, %r1397, 64;
	add.s32 	%r1402, %r95, %r1401;
	ld.shared.v4.b32 	{%r891, %r895, %r899, %r903}, [%r1402+8192];
	xor.b32 	%r1403, %r1397, 96;
	add.s32 	%r1404, %r95, %r1403;
	ld.shared.v4.b32 	{%r892, %r896, %r900, %r904}, [%r1404+12288];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r821 + 0 ], { %r841, %r842, %r843, %r844 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r826 + 0 ], { %r845, %r846, %r847, %r848 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r831 + 0 ], { %r849, %r850, %r851, %r852 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r836 + 0 ], { %r853, %r854, %r855, %r856 };
	// end inline asm
	bar.sync 	0;
	ld.shared.v4.b32 	{%r905, %r909, %r913, %r917}, [%r1398];
	ld.shared.v4.b32 	{%r906, %r910, %r914, %r918}, [%r1400+4096];
	ld.shared.v4.b32 	{%r907, %r911, %r915, %r919}, [%r1402+8192];
	ld.shared.v4.b32 	{%r908, %r912, %r916, %r920}, [%r1404+12288];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r821 + 0 ], { %r857, %r858, %r859, %r860 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r826 + 0 ], { %r861, %r862, %r863, %r864 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r831 + 0 ], { %r865, %r866, %r867, %r868 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r836 + 0 ], { %r869, %r870, %r871, %r872 };
	// end inline asm
	bar.sync 	0;
	ld.shared.v4.b32 	{%r921, %r925, %r929, %r933}, [%r1398];
	ld.shared.v4.b32 	{%r922, %r926, %r930, %r934}, [%r1400+4096];
	ld.shared.v4.b32 	{%r923, %r927, %r931, %r935}, [%r1402+8192];
	ld.shared.v4.b32 	{%r924, %r928, %r932, %r936}, [%r1404+12288];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r821 + 0 ], { %r873, %r874, %r875, %r876 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r826 + 0 ], { %r877, %r878, %r879, %r880 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r831 + 0 ], { %r881, %r882, %r883, %r884 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r836 + 0 ], { %r885, %r886, %r887, %r888 };
	// end inline asm
	bar.sync 	0;
	ld.shared.v4.b32 	{%r937, %r941, %r945, %r949}, [%r1398];
	ld.shared.v4.b32 	{%r938, %r942, %r946, %r950}, [%r1400+4096];
	ld.shared.v4.b32 	{%r939, %r943, %r947, %r951}, [%r1402+8192];
	ld.shared.v4.b32 	{%r940, %r944, %r948, %r952}, [%r1404+12288];
	.loc	1 845 8                         // sk04_fa_o_w4a8.py:845:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd80 + 0 ], { %r889, %r890, %r891, %r892 };
	// end inline asm
	// begin inline asm
	@%p9 st.global.v4.b32 [ %rd81 + 0 ], { %r893, %r894, %r895, %r896 };
	// end inline asm
	// begin inline asm
	@%p10 st.global.v4.b32 [ %rd82 + 0 ], { %r897, %r898, %r899, %r900 };
	// end inline asm
	// begin inline asm
	@%p11 st.global.v4.b32 [ %rd83 + 0 ], { %r901, %r902, %r903, %r904 };
	// end inline asm
	// begin inline asm
	@%p12 st.global.v4.b32 [ %rd84 + 0 ], { %r905, %r906, %r907, %r908 };
	// end inline asm
	// begin inline asm
	@%p13 st.global.v4.b32 [ %rd85 + 0 ], { %r909, %r910, %r911, %r912 };
	// end inline asm
	// begin inline asm
	@%p14 st.global.v4.b32 [ %rd86 + 0 ], { %r913, %r914, %r915, %r916 };
	// end inline asm
	// begin inline asm
	@%p15 st.global.v4.b32 [ %rd87 + 0 ], { %r917, %r918, %r919, %r920 };
	// end inline asm
	// begin inline asm
	@%p16 st.global.v4.b32 [ %rd88 + 0 ], { %r921, %r922, %r923, %r924 };
	// end inline asm
	// begin inline asm
	@%p17 st.global.v4.b32 [ %rd89 + 0 ], { %r925, %r926, %r927, %r928 };
	// end inline asm
	// begin inline asm
	@%p18 st.global.v4.b32 [ %rd90 + 0 ], { %r929, %r930, %r931, %r932 };
	// end inline asm
	// begin inline asm
	@%p19 st.global.v4.b32 [ %rd91 + 0 ], { %r933, %r934, %r935, %r936 };
	// end inline asm
	// begin inline asm
	@%p20 st.global.v4.b32 [ %rd92 + 0 ], { %r937, %r938, %r939, %r940 };
	// end inline asm
	// begin inline asm
	@%p21 st.global.v4.b32 [ %rd93 + 0 ], { %r941, %r942, %r943, %r944 };
	// end inline asm
	// begin inline asm
	@%p22 st.global.v4.b32 [ %rd94 + 0 ], { %r945, %r946, %r947, %r948 };
	// end inline asm
	// begin inline asm
	@%p23 st.global.v4.b32 [ %rd95 + 0 ], { %r949, %r950, %r951, %r952 };
	// end inline asm
	.loc	1 843 4                         // sk04_fa_o_w4a8.py:843:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk04_fa_o_w4a8.py"
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
.b8 5                                   // DW_FORM_data2
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 165                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x9e DW_TAG_compile_unit
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
.b8 52
.b8 95
.b8 102
.b8 97
.b8 95
.b8 111
.b8 95
.b8 119
.b8 52
.b8 97
.b8 56
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
.b8 2                                   // Abbrev [2] 0x47:0x19 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 52
.b8 95
.b8 102
.b8 97
.b8 95
.b8 111
.b8 95
.b8 119
.b8 52
.b8 97
.b8 56
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x60:0x48 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 71                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x75:0x19 DW_TAG_inlined_subroutine
.b32 71                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 40                                  // DW_AT_call_line
.b8 3
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x8e:0x19 DW_TAG_inlined_subroutine
.b32 71                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 41                                  // DW_AT_call_line
.b8 3
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_4 = _Nativo(
    "sk04_fa_o_w4a8/tile128x256x64_shift0_abi14",
    _PTX_4, "_sk04_fa_o_w4a8_kernel",
    warps=8, shared=33792,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 17],
    horneado={10: 1, 12: 1, 14: 1, 16: 1, 18: 1, 19: 128, 20: 256, 21: 64, 22: 8},
    div16=[7, 8, 9, 11, 13, 15, 17],
)

_PTX_5 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk04_fa_o_w4a8_kernel  // -- Begin function _sk04_fa_o_w4a8_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk04_fa_o_w4a8_kernel
.visible .entry _sk04_fa_o_w4a8_kernel(
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_5,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_6,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_7,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_8,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_9,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_10,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_11,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_12,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_13,
	.param .u32 _sk04_fa_o_w4a8_kernel_param_14,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk04_fa_o_w4a8_kernel_param_16
)
.reqntid 256
{
	.reg .pred 	%p<41>;
	.reg .b16 	%rs<417>;
	.reg .b32 	%r<1560>;
	.reg .b64 	%rd<253>;
	.loc	1 797 0                         // sk04_fa_o_w4a8.py:797:0
$L__func_begin0:
	.loc	1 797 0                         // sk04_fa_o_w4a8.py:797:0

// %bb.0:
	ld.param.b32 	%r35, [_sk04_fa_o_w4a8_kernel_param_13];
	ld.param.b32 	%r34, [_sk04_fa_o_w4a8_kernel_param_12];
	ld.param.b32 	%r33, [_sk04_fa_o_w4a8_kernel_param_11];
	ld.param.b32 	%r32, [_sk04_fa_o_w4a8_kernel_param_8];
	ld.param.b32 	%r31, [_sk04_fa_o_w4a8_kernel_param_7];
	ld.param.b32 	%r30, [_sk04_fa_o_w4a8_kernel_param_6];
	ld.param.b64 	%rd17, [_sk04_fa_o_w4a8_kernel_param_4];
	ld.param.b64 	%rd16, [_sk04_fa_o_w4a8_kernel_param_3];
	ld.param.b64 	%rd15, [_sk04_fa_o_w4a8_kernel_param_2];
	ld.param.b64 	%rd14, [_sk04_fa_o_w4a8_kernel_param_1];
	ld.param.b64 	%rd13, [_sk04_fa_o_w4a8_kernel_param_0];
$L__tmp0:
	.loc	1 807 24                        // sk04_fa_o_w4a8.py:807:24
	mov.u32 	%r53, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk04_fa_o_w4a8.py:808:27 ]
	add.s32 	%r54, %r30, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk04_fa_o_w4a8.py:808:27 ]
	shr.s32 	%r55, %r54, 31;
	shr.u32 	%r56, %r55, 25;
	add.s32 	%r57, %r54, %r56;
	shr.s32 	%r58, %r57, 7;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk04_fa_o_w4a8.py:809:27 ]
	add.s32 	%r59, %r31, 255;
	.loc	2 43 30                         // standard.py:43:30 @[ sk04_fa_o_w4a8.py:809:27 ]
	shr.s32 	%r60, %r59, 31;
	shr.u32 	%r61, %r60, 24;
	add.s32 	%r62, %r59, %r61;
	shr.s32 	%r63, %r62, 8;
$L__tmp3:
	.loc	1 810 29                        // sk04_fa_o_w4a8.py:810:29
	shl.b32 	%r64, %r63, 3;
	.loc	1 811 22                        // sk04_fa_o_w4a8.py:811:22
	div.s32 	%r65, %r53, %r64;
	.loc	1 811 38                        // sk04_fa_o_w4a8.py:811:38
	shl.b32 	%r66, %r65, 3;
	ld.param.b32 	%r67, [_sk04_fa_o_w4a8_kernel_param_9];
	.loc	1 812 30                        // sk04_fa_o_w4a8.py:812:30
	sub.s32 	%r68, %r58, %r66;
	ld.param.b32 	%r69, [_sk04_fa_o_w4a8_kernel_param_10];
	.loc	1 812 39                        // sk04_fa_o_w4a8.py:812:39
	min.s32 	%r70, %r68, 8;
	.loc	1 813 30                        // sk04_fa_o_w4a8.py:813:30
	mul.lo.s32 	%r71, %r65, %r64;
	sub.s32 	%r72, %r53, %r71;
	.loc	1 814 36                        // sk04_fa_o_w4a8.py:814:36
	div.s32 	%r73, %r72, %r70;
	.loc	1 813 46                        // sk04_fa_o_w4a8.py:813:46
	mul.lo.s32 	%r74, %r73, %r70;
	sub.s32 	%r75, %r72, %r74;
	.loc	1 813 23                        // sk04_fa_o_w4a8.py:813:23
	add.s32 	%r76, %r75, %r66;
	.loc	1 816 22                        // sk04_fa_o_w4a8.py:816:22
	shl.b32 	%r1, %r76, 7;
	.loc	1 816 45                        // sk04_fa_o_w4a8.py:816:45
	mov.u32 	%r2, %tid.x;
	shr.u32 	%r3, %r2, 2;
	bfe.u32 	%r4, %r2, 2, 6;
	and.b32 	%r5, %r2, 224;
	bfe.u32 	%r6, %r2, 5, 3;
	or.b32 	%r7, %r6, 8;
	or.b32 	%r8, %r6, 16;
	or.b32 	%r9, %r6, 24;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r77, %r1, %r4;
	or.b32 	%r78, %r77, 64;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r79, %r77, %r30;
	rem.s32 	%r80, %r78, %r30;
	.loc	1 817 22                        // sk04_fa_o_w4a8.py:817:22
	shl.b32 	%r10, %r73, 8;
	.loc	1 817 45                        // sk04_fa_o_w4a8.py:817:45
	and.b32 	%r11, %r2, 31;
	shl.b32 	%r81, %r11, 3;
	and.b32 	%r12, %r2, 255;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r13, %r10, %r81;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r82, %r13, %r31;
	.loc	1 821 39                        // sk04_fa_o_w4a8.py:821:39
	mul.lo.s32 	%r83, %r79, %r67;
	mul.lo.s32 	%r84, %r80, %r67;
	.loc	1 821 21                        // sk04_fa_o_w4a8.py:821:21
	cvt.s64.s32 	%rd1, %r83;
	add.s64 	%rd31, %rd13, %rd1;
	cvt.s64.s32 	%rd2, %r84;
	add.s64 	%rd32, %rd13, %rd2;
	.loc	1 821 58                        // sk04_fa_o_w4a8.py:821:58
	and.b32 	%r14, %r2, 3;
	shl.b32 	%r85, %r14, 4;
	.loc	1 821 51                        // sk04_fa_o_w4a8.py:821:51
	cvt.u64.u32 	%rd3, %r85;
	add.s64 	%rd23, %rd31, %rd3;
	add.s64 	%rd24, %rd32, %rd3;
	.loc	1 822 40                        // sk04_fa_o_w4a8.py:822:40
	mul.lo.s32 	%r86, %r69, %r6;
	shl.b32 	%r87, %r69, 3;
	add.s32 	%r88, %r86, %r87;
	shl.b32 	%r89, %r69, 4;
	add.s32 	%r90, %r86, %r89;
	mad.lo.s32 	%r91, %r69, 24, %r86;
	.loc	1 822 21                        // sk04_fa_o_w4a8.py:822:21
	cvt.s64.s32 	%rd4, %r86;
	add.s64 	%rd33, %rd14, %rd4;
	cvt.s64.s32 	%rd5, %r88;
	add.s64 	%rd34, %rd14, %rd5;
	cvt.s64.s32 	%rd6, %r90;
	add.s64 	%rd35, %rd14, %rd6;
	cvt.s64.s32 	%rd7, %r91;
	add.s64 	%rd36, %rd14, %rd7;
	.loc	1 822 52                        // sk04_fa_o_w4a8.py:822:52
	cvt.s64.s32 	%rd8, %r82;
	add.s64 	%rd19, %rd33, %rd8;
	add.s64 	%rd21, %rd34, %rd8;
	add.s64 	%rd20, %rd35, %rd8;
	add.s64 	%rd22, %rd36, %rd8;
	.loc	1 836 35                        // sk04_fa_o_w4a8.py:836:35
	shl.b32 	%r92, %r69, 5;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	setp.lt.s32 	%p1, %r32, 64;
	setp.gt.s32 	%p2, %r32, 63;
	.loc	1 827 25                        // sk04_fa_o_w4a8.py:827:25
	shl.b32 	%r93, %r12, 3;
	shr.u32 	%r94, %r5, 2;
	xor.b32 	%r95, %r93, %r94;
	mov.b32 	%r96, global_smem;
	add.s32 	%r37, %r96, %r95;
	selp.b32 	%r38, 8, 0, %p2;
	// begin inline asm
	cp.async.ca.shared.global [ %r37 + 0 ], [ %rd19 + 0 ], 0x8, %r38;
	// end inline asm
	add.s32 	%r39, %r37, 4096;
	// begin inline asm
	cp.async.ca.shared.global [ %r39 + 0 ], [ %rd20 + 0 ], 0x8, %r38;
	// end inline asm
	xor.b32 	%r15, %r95, 64;
	add.s32 	%r97, %r96, %r15;
	add.s32 	%r40, %r97, 2048;
	// begin inline asm
	cp.async.ca.shared.global [ %r40 + 0 ], [ %rd21 + 0 ], 0x8, %r38;
	// end inline asm
	add.s32 	%r41, %r97, 6144;
	// begin inline asm
	cp.async.ca.shared.global [ %r41 + 0 ], [ %rd22 + 0 ], 0x8, %r38;
	// end inline asm
	cp.async.commit_group;
	.loc	1 831 52                        // sk04_fa_o_w4a8.py:831:52
	shl.b32 	%r16, %r2, 4;
	and.b32 	%r98, %r16, 3952;
	bfe.s32 	%r99, %r2, 3, 1;
	and.b32 	%r17, %r99, 160;
	xor.b32 	%r100, %r17, %r98;
	add.s32 	%r18, %r96, %r100;
	add.s32 	%r42, %r18, 16384;
	selp.b32 	%r43, 16, 0, %p2;
	// begin inline asm
	cp.async.cg.shared.global [ %r42 + 0 ], [ %rd23 + 0 ], 0x10, %r43;
	// end inline asm
	add.s32 	%r44, %r18, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r44 + 0 ], [ %rd24 + 0 ], 0x10, %r43;
	// end inline asm
	cp.async.commit_group;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	setp.gt.s32 	%p3, %r32, 127;
	.loc	1 835 18                        // sk04_fa_o_w4a8.py:835:18
	add.s64 	%rd29, %rd23, 64;
	add.s64 	%rd30, %rd24, 64;
	.loc	1 836 18                        // sk04_fa_o_w4a8.py:836:18
	cvt.s64.s32 	%rd9, %r92;
	add.s64 	%rd25, %rd19, %rd9;
	add.s64 	%rd27, %rd21, %rd9;
	add.s64 	%rd26, %rd20, %rd9;
	add.s64 	%rd28, %rd22, %rd9;
	.loc	1 827 25                        // sk04_fa_o_w4a8.py:827:25
	bar.sync 	0;
	add.s32 	%r45, %r37, 8192;
	selp.b32 	%r46, 8, 0, %p3;
	// begin inline asm
	cp.async.ca.shared.global [ %r45 + 0 ], [ %rd25 + 0 ], 0x8, %r46;
	// end inline asm
	add.s32 	%r47, %r37, 12288;
	// begin inline asm
	cp.async.ca.shared.global [ %r47 + 0 ], [ %rd26 + 0 ], 0x8, %r46;
	// end inline asm
	add.s32 	%r48, %r97, 10240;
	// begin inline asm
	cp.async.ca.shared.global [ %r48 + 0 ], [ %rd27 + 0 ], 0x8, %r46;
	// end inline asm
	add.s32 	%r49, %r97, 14336;
	// begin inline asm
	cp.async.ca.shared.global [ %r49 + 0 ], [ %rd28 + 0 ], 0x8, %r46;
	// end inline asm
	cp.async.commit_group;
	.loc	1 831 52                        // sk04_fa_o_w4a8.py:831:52
	add.s32 	%r50, %r18, 24576;
	selp.b32 	%r51, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r50 + 0 ], [ %rd29 + 0 ], 0x10, %r51;
	// end inline asm
	add.s32 	%r52, %r18, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r52 + 0 ], [ %rd30 + 0 ], 0x10, %r51;
	// end inline asm
	cp.async.commit_group;
	mov.b32 	%r1432, 0f00000000;
	mov.b32 	%r1433, %r1432;
	mov.b32 	%r1434, %r1432;
	mov.b32 	%r1435, %r1432;
	mov.b32 	%r1436, %r1432;
	mov.b32 	%r1437, %r1432;
	mov.b32 	%r1438, %r1432;
	mov.b32 	%r1439, %r1432;
	mov.b32 	%r1440, %r1432;
	mov.b32 	%r1441, %r1432;
	mov.b32 	%r1442, %r1432;
	mov.b32 	%r1443, %r1432;
	mov.b32 	%r1444, %r1432;
	mov.b32 	%r1445, %r1432;
	mov.b32 	%r1446, %r1432;
	mov.b32 	%r1447, %r1432;
	mov.b32 	%r1448, %r1432;
	mov.b32 	%r1449, %r1432;
	mov.b32 	%r1450, %r1432;
	mov.b32 	%r1451, %r1432;
	mov.b32 	%r1452, %r1432;
	mov.b32 	%r1453, %r1432;
	mov.b32 	%r1454, %r1432;
	mov.b32 	%r1455, %r1432;
	mov.b32 	%r1456, %r1432;
	mov.b32 	%r1457, %r1432;
	mov.b32 	%r1458, %r1432;
	mov.b32 	%r1459, %r1432;
	mov.b32 	%r1460, %r1432;
	mov.b32 	%r1461, %r1432;
	mov.b32 	%r1462, %r1432;
	mov.b32 	%r1463, %r1432;
	mov.b32 	%r1464, %r1432;
	mov.b32 	%r1465, %r1432;
	mov.b32 	%r1466, %r1432;
	mov.b32 	%r1467, %r1432;
	mov.b32 	%r1468, %r1432;
	mov.b32 	%r1469, %r1432;
	mov.b32 	%r1470, %r1432;
	mov.b32 	%r1471, %r1432;
	mov.b32 	%r1472, %r1432;
	mov.b32 	%r1473, %r1432;
	mov.b32 	%r1474, %r1432;
	mov.b32 	%r1475, %r1432;
	mov.b32 	%r1476, %r1432;
	mov.b32 	%r1477, %r1432;
	mov.b32 	%r1478, %r1432;
	mov.b32 	%r1479, %r1432;
	mov.b32 	%r1480, %r1432;
	mov.b32 	%r1481, %r1432;
	mov.b32 	%r1482, %r1432;
	mov.b32 	%r1483, %r1432;
	mov.b32 	%r1484, %r1432;
	mov.b32 	%r1485, %r1432;
	mov.b32 	%r1486, %r1432;
	mov.b32 	%r1487, %r1432;
	mov.b32 	%r1488, %r1432;
	mov.b32 	%r1489, %r1432;
	mov.b32 	%r1490, %r1432;
	mov.b32 	%r1491, %r1432;
	mov.b32 	%r1492, %r1432;
	mov.b32 	%r1493, %r1432;
	mov.b32 	%r1494, %r1432;
	mov.b32 	%r1495, %r1432;
	mov.b32 	%r1496, %r1432;
	mov.b32 	%r1497, %r1432;
	mov.b32 	%r1498, %r1432;
	mov.b32 	%r1499, %r1432;
	mov.b32 	%r1500, %r1432;
	mov.b32 	%r1501, %r1432;
	mov.b32 	%r1502, %r1432;
	mov.b32 	%r1503, %r1432;
	mov.b32 	%r1504, %r1432;
	mov.b32 	%r1505, %r1432;
	mov.b32 	%r1506, %r1432;
	mov.b32 	%r1507, %r1432;
	mov.b32 	%r1508, %r1432;
	mov.b32 	%r1509, %r1432;
	mov.b32 	%r1510, %r1432;
	mov.b32 	%r1511, %r1432;
	mov.b32 	%r1512, %r1432;
	mov.b32 	%r1513, %r1432;
	mov.b32 	%r1514, %r1432;
	mov.b32 	%r1515, %r1432;
	mov.b32 	%r1516, %r1432;
	mov.b32 	%r1517, %r1432;
	mov.b32 	%r1518, %r1432;
	mov.b32 	%r1519, %r1432;
	mov.b32 	%r1520, %r1432;
	mov.b32 	%r1521, %r1432;
	mov.b32 	%r1522, %r1432;
	mov.b32 	%r1523, %r1432;
	mov.b32 	%r1524, %r1432;
	mov.b32 	%r1525, %r1432;
	mov.b32 	%r1526, %r1432;
	mov.b32 	%r1527, %r1432;
	mov.b32 	%r1528, %r1432;
	mov.b32 	%r1529, %r1432;
	mov.b32 	%r1530, %r1432;
	mov.b32 	%r1531, %r1432;
	mov.b32 	%r1532, %r1432;
	mov.b32 	%r1533, %r1432;
	mov.b32 	%r1534, %r1432;
	mov.b32 	%r1535, %r1432;
	mov.b32 	%r1536, %r1432;
	mov.b32 	%r1537, %r1432;
	mov.b32 	%r1538, %r1432;
	mov.b32 	%r1539, %r1432;
	mov.b32 	%r1540, %r1432;
	mov.b32 	%r1541, %r1432;
	mov.b32 	%r1542, %r1432;
	mov.b32 	%r1543, %r1432;
	mov.b32 	%r1544, %r1432;
	mov.b32 	%r1545, %r1432;
	mov.b32 	%r1546, %r1432;
	mov.b32 	%r1547, %r1432;
	mov.b32 	%r1548, %r1432;
	mov.b32 	%r1549, %r1432;
	mov.b32 	%r1550, %r1432;
	mov.b32 	%r1551, %r1432;
	mov.b32 	%r1552, %r1432;
	mov.b32 	%r1553, %r1432;
	mov.b32 	%r1554, %r1432;
	mov.b32 	%r1555, %r1432;
	mov.b32 	%r1556, %r1432;
	mov.b32 	%r1557, %r1432;
	mov.b32 	%r1558, %r1432;
	mov.b32 	%r1559, %r1432;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	@%p1 bra 	$L__BB0_3;
// %bb.1:                               // %.lr.ph
	.loc	1 0 22                          // sk04_fa_o_w4a8.py:0:22
	ld.param.b32 	%r36, [_sk04_fa_o_w4a8_kernel_param_14];
	ld.param.b64 	%rd18, [_sk04_fa_o_w4a8_kernel_param_5];
	.loc	1 825 27                        // sk04_fa_o_w4a8.py:825:27
	shr.u32 	%r101, %r32, 6;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	add.s32 	%r102, %r96, %r5;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r103, %r10, %r12;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r104, %r103, %r31;
	add.s32 	%r105, %r101, -2;
	mul.lo.s32 	%r106, %r14, 1056;
	xor.b32 	%r19, %r106, %r4;
	xor.b32 	%r20, %r19, 264;
	xor.b32 	%r21, %r19, 528;
	xor.b32 	%r22, %r19, 792;
	xor.b32 	%r23, %r19, 64;
	xor.b32 	%r24, %r19, 328;
	xor.b32 	%r25, %r19, 592;
	xor.b32 	%r26, %r19, 856;
	and.b32 	%r107, %r16, 320;
	shl.b32 	%r108, %r14, 3;
	or.b32 	%r109, %r107, %r108;
	or.b32 	%r27, %r109, %r17;
	xor.b32 	%r28, %r27, 32;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	mad.wide.s32 	%rd10, %r104, 4, %rd18;
	shl.b32 	%r110, %r12, 2;
	add.s32 	%r111, %r96, %r110;
	add.s32 	%r322, %r111, 32768;
	add.s32 	%r29, %r102, %r108;
	cvt.s64.s32 	%rd11, %r105;
	cvt.u64.u32 	%rd12, %r101;
	shl.b64 	%rd37, %rd9, 1;
	add.s64 	%rd38, %rd37, %rd8;
	add.s64 	%rd251, %rd14, %rd38;
	add.s64 	%rd39, %rd3, %rd2;
	add.s64 	%rd40, %rd39, %rd13;
	add.s64 	%rd250, %rd40, 128;
	add.s64 	%rd41, %rd3, %rd1;
	add.s64 	%rd42, %rd41, %rd13;
	add.s64 	%rd249, %rd42, 128;
	mov.b32 	%r1432, 0f00000000;
	mov.b32 	%r1431, 1;
	mov.b32 	%r1430, -1;
	mov.b64 	%rd252, 0;
	mov.b32 	%r112, 0;
	mov.b32 	%r1429, %r112;
	mov.b32 	%r1433, %r1432;
	mov.b32 	%r1434, %r1432;
	mov.b32 	%r1435, %r1432;
	mov.b32 	%r1436, %r1432;
	mov.b32 	%r1437, %r1432;
	mov.b32 	%r1438, %r1432;
	mov.b32 	%r1439, %r1432;
	mov.b32 	%r1440, %r1432;
	mov.b32 	%r1441, %r1432;
	mov.b32 	%r1442, %r1432;
	mov.b32 	%r1443, %r1432;
	mov.b32 	%r1444, %r1432;
	mov.b32 	%r1445, %r1432;
	mov.b32 	%r1446, %r1432;
	mov.b32 	%r1447, %r1432;
	mov.b32 	%r1448, %r1432;
	mov.b32 	%r1449, %r1432;
	mov.b32 	%r1450, %r1432;
	mov.b32 	%r1451, %r1432;
	mov.b32 	%r1452, %r1432;
	mov.b32 	%r1453, %r1432;
	mov.b32 	%r1454, %r1432;
	mov.b32 	%r1455, %r1432;
	mov.b32 	%r1456, %r1432;
	mov.b32 	%r1457, %r1432;
	mov.b32 	%r1458, %r1432;
	mov.b32 	%r1459, %r1432;
	mov.b32 	%r1460, %r1432;
	mov.b32 	%r1461, %r1432;
	mov.b32 	%r1462, %r1432;
	mov.b32 	%r1463, %r1432;
	mov.b32 	%r1464, %r1432;
	mov.b32 	%r1465, %r1432;
	mov.b32 	%r1466, %r1432;
	mov.b32 	%r1467, %r1432;
	mov.b32 	%r1468, %r1432;
	mov.b32 	%r1469, %r1432;
	mov.b32 	%r1470, %r1432;
	mov.b32 	%r1471, %r1432;
	mov.b32 	%r1472, %r1432;
	mov.b32 	%r1473, %r1432;
	mov.b32 	%r1474, %r1432;
	mov.b32 	%r1475, %r1432;
	mov.b32 	%r1476, %r1432;
	mov.b32 	%r1477, %r1432;
	mov.b32 	%r1478, %r1432;
	mov.b32 	%r1479, %r1432;
	mov.b32 	%r1480, %r1432;
	mov.b32 	%r1481, %r1432;
	mov.b32 	%r1482, %r1432;
	mov.b32 	%r1483, %r1432;
	mov.b32 	%r1484, %r1432;
	mov.b32 	%r1485, %r1432;
	mov.b32 	%r1486, %r1432;
	mov.b32 	%r1487, %r1432;
	mov.b32 	%r1488, %r1432;
	mov.b32 	%r1489, %r1432;
	mov.b32 	%r1490, %r1432;
	mov.b32 	%r1491, %r1432;
	mov.b32 	%r1492, %r1432;
	mov.b32 	%r1493, %r1432;
	mov.b32 	%r1494, %r1432;
	mov.b32 	%r1495, %r1432;
	mov.b32 	%r1496, %r1432;
	mov.b32 	%r1497, %r1432;
	mov.b32 	%r1498, %r1432;
	mov.b32 	%r1499, %r1432;
	mov.b32 	%r1500, %r1432;
	mov.b32 	%r1501, %r1432;
	mov.b32 	%r1502, %r1432;
	mov.b32 	%r1503, %r1432;
	mov.b32 	%r1504, %r1432;
	mov.b32 	%r1505, %r1432;
	mov.b32 	%r1506, %r1432;
	mov.b32 	%r1507, %r1432;
	mov.b32 	%r1508, %r1432;
	mov.b32 	%r1509, %r1432;
	mov.b32 	%r1510, %r1432;
	mov.b32 	%r1511, %r1432;
	mov.b32 	%r1512, %r1432;
	mov.b32 	%r1513, %r1432;
	mov.b32 	%r1514, %r1432;
	mov.b32 	%r1515, %r1432;
	mov.b32 	%r1516, %r1432;
	mov.b32 	%r1517, %r1432;
	mov.b32 	%r1518, %r1432;
	mov.b32 	%r1519, %r1432;
	mov.b32 	%r1520, %r1432;
	mov.b32 	%r1521, %r1432;
	mov.b32 	%r1522, %r1432;
	mov.b32 	%r1523, %r1432;
	mov.b32 	%r1524, %r1432;
	mov.b32 	%r1525, %r1432;
	mov.b32 	%r1526, %r1432;
	mov.b32 	%r1527, %r1432;
	mov.b32 	%r1528, %r1432;
	mov.b32 	%r1529, %r1432;
	mov.b32 	%r1530, %r1432;
	mov.b32 	%r1531, %r1432;
	mov.b32 	%r1532, %r1432;
	mov.b32 	%r1533, %r1432;
	mov.b32 	%r1534, %r1432;
	mov.b32 	%r1535, %r1432;
	mov.b32 	%r1536, %r1432;
	mov.b32 	%r1537, %r1432;
	mov.b32 	%r1538, %r1432;
	mov.b32 	%r1539, %r1432;
	mov.b32 	%r1540, %r1432;
	mov.b32 	%r1541, %r1432;
	mov.b32 	%r1542, %r1432;
	mov.b32 	%r1543, %r1432;
	mov.b32 	%r1544, %r1432;
	mov.b32 	%r1545, %r1432;
	mov.b32 	%r1546, %r1432;
	mov.b32 	%r1547, %r1432;
	mov.b32 	%r1548, %r1432;
	mov.b32 	%r1549, %r1432;
	mov.b32 	%r1550, %r1432;
	mov.b32 	%r1551, %r1432;
	mov.b32 	%r1552, %r1432;
	mov.b32 	%r1553, %r1432;
	mov.b32 	%r1554, %r1432;
	mov.b32 	%r1555, %r1432;
	mov.b32 	%r1556, %r1432;
	mov.b32 	%r1557, %r1432;
	mov.b32 	%r1558, %r1432;
	mov.b32 	%r1559, %r1432;
$L__BB0_2:                              // =>This Inner Loop Header: Depth=1
	setp.lt.s64 	%p4, %rd252, %rd11;
	add.s32 	%r331, %r1430, 1;
	setp.gt.s32 	%p5, %r331, 1;
	selp.b32 	%r1430, 0, %r331, %p5;
	.loc	1 827 25                        // sk04_fa_o_w4a8.py:827:25
	cp.async.wait_group 	2;
	bar.sync 	0;
	shl.b32 	%r332, %r1430, 13;
	add.s32 	%r333, %r96, %r332;
	.loc	1 828 38                        // sk04_fa_o_w4a8.py:828:38
	add.s32 	%r334, %r333, %r19;
	ld.shared.b8 	%rs1, [%r334];
	ld.shared.b8 	%rs2, [%r334+4096];
	ld.shared.b8 	%rs3, [%r334+128];
	ld.shared.b8 	%rs4, [%r334+4224];
	add.s32 	%r335, %r333, %r20;
	ld.shared.b8 	%rs5, [%r335];
	ld.shared.b8 	%rs6, [%r335+4096];
	ld.shared.b8 	%rs7, [%r335+128];
	ld.shared.b8 	%rs8, [%r335+4224];
	add.s32 	%r336, %r333, %r21;
	ld.shared.b8 	%rs9, [%r336];
	ld.shared.b8 	%rs10, [%r336+4096];
	ld.shared.b8 	%rs11, [%r336+128];
	ld.shared.b8 	%rs12, [%r336+4224];
	add.s32 	%r337, %r333, %r22;
	ld.shared.b8 	%rs13, [%r337];
	ld.shared.b8 	%rs14, [%r337+4096];
	ld.shared.b8 	%rs15, [%r337+128];
	ld.shared.b8 	%rs16, [%r337+4224];
	add.s32 	%r338, %r333, %r23;
	ld.shared.b8 	%rs17, [%r338];
	ld.shared.b8 	%rs18, [%r338+4096];
	ld.shared.b8 	%rs19, [%r338+128];
	ld.shared.b8 	%rs20, [%r338+4224];
	add.s32 	%r339, %r333, %r24;
	ld.shared.b8 	%rs21, [%r339];
	ld.shared.b8 	%rs22, [%r339+4096];
	ld.shared.b8 	%rs23, [%r339+128];
	ld.shared.b8 	%rs24, [%r339+4224];
	add.s32 	%r340, %r333, %r25;
	ld.shared.b8 	%rs25, [%r340];
	ld.shared.b8 	%rs26, [%r340+4096];
	ld.shared.b8 	%rs27, [%r340+128];
	ld.shared.b8 	%rs28, [%r340+4224];
	add.s32 	%r341, %r333, %r26;
	ld.shared.b8 	%rs29, [%r341];
	ld.shared.b8 	%rs30, [%r341+4096];
	ld.shared.b8 	%rs31, [%r341+128];
	ld.shared.b8 	%rs32, [%r341+4224];
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r342, %rs13;
	cvt.u32.u16 	%r343, %rs9;
	prmt.b32 	%r344, %r343, %r342, 0x3340U;
	cvt.u32.u16 	%r345, %rs5;
	cvt.u32.u16 	%r346, %rs1;
	prmt.b32 	%r347, %r346, %r345, 0x3340U;
	prmt.b32 	%r348, %r347, %r344, 0x5410U;
	and.b32 	%r349, %r348, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r350, %r349, 0, 0x7773U;
	cvt.u16.u32 	%rs33, %r350;
	add.s16 	%rs34, %rs33, -8;
	cvt.u32.u16 	%r351, %rs34;
	prmt.b32 	%r352, %r349, 0, 0x7772U;
	cvt.u16.u32 	%rs35, %r352;
	add.s16 	%rs36, %rs35, -8;
	cvt.u32.u16 	%r353, %rs36;
	prmt.b32 	%r354, %r353, %r351, 0x3340U;
	prmt.b32 	%r355, %r349, 0, 0x7771U;
	cvt.u16.u32 	%rs37, %r355;
	add.s16 	%rs38, %rs37, -8;
	cvt.u32.u16 	%r356, %rs38;
	prmt.b32 	%r357, %r349, 0, 0x7770U;
	cvt.u16.u32 	%rs39, %r357;
	add.s16 	%rs40, %rs39, -8;
	cvt.u32.u16 	%r358, %rs40;
	prmt.b32 	%r359, %r358, %r356, 0x3340U;
	prmt.b32 	%r117, %r359, %r354, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r360, %rs14;
	cvt.u32.u16 	%r361, %rs10;
	prmt.b32 	%r362, %r361, %r360, 0x3340U;
	cvt.u32.u16 	%r363, %rs6;
	cvt.u32.u16 	%r364, %rs2;
	prmt.b32 	%r365, %r364, %r363, 0x3340U;
	prmt.b32 	%r366, %r365, %r362, 0x5410U;
	and.b32 	%r367, %r366, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r368, %r367, 0, 0x7773U;
	cvt.u16.u32 	%rs41, %r368;
	add.s16 	%rs42, %rs41, -8;
	cvt.u32.u16 	%r369, %rs42;
	prmt.b32 	%r370, %r367, 0, 0x7772U;
	cvt.u16.u32 	%rs43, %r370;
	add.s16 	%rs44, %rs43, -8;
	cvt.u32.u16 	%r371, %rs44;
	prmt.b32 	%r372, %r371, %r369, 0x3340U;
	prmt.b32 	%r373, %r367, 0, 0x7771U;
	cvt.u16.u32 	%rs45, %r373;
	add.s16 	%rs46, %rs45, -8;
	cvt.u32.u16 	%r374, %rs46;
	prmt.b32 	%r375, %r367, 0, 0x7770U;
	cvt.u16.u32 	%rs47, %r375;
	add.s16 	%rs48, %rs47, -8;
	cvt.u32.u16 	%r376, %rs48;
	prmt.b32 	%r377, %r376, %r374, 0x3340U;
	prmt.b32 	%r118, %r377, %r372, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r378, %rs29;
	cvt.u32.u16 	%r379, %rs25;
	prmt.b32 	%r380, %r379, %r378, 0x3340U;
	cvt.u32.u16 	%r381, %rs21;
	cvt.u32.u16 	%r382, %rs17;
	prmt.b32 	%r383, %r382, %r381, 0x3340U;
	prmt.b32 	%r384, %r383, %r380, 0x5410U;
	and.b32 	%r385, %r384, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r386, %r385, 0, 0x7773U;
	cvt.u16.u32 	%rs49, %r386;
	add.s16 	%rs50, %rs49, -8;
	cvt.u32.u16 	%r387, %rs50;
	prmt.b32 	%r388, %r385, 0, 0x7772U;
	cvt.u16.u32 	%rs51, %r388;
	add.s16 	%rs52, %rs51, -8;
	cvt.u32.u16 	%r389, %rs52;
	prmt.b32 	%r390, %r389, %r387, 0x3340U;
	prmt.b32 	%r391, %r385, 0, 0x7771U;
	cvt.u16.u32 	%rs53, %r391;
	add.s16 	%rs54, %rs53, -8;
	cvt.u32.u16 	%r392, %rs54;
	prmt.b32 	%r393, %r385, 0, 0x7770U;
	cvt.u16.u32 	%rs55, %r393;
	add.s16 	%rs56, %rs55, -8;
	cvt.u32.u16 	%r394, %rs56;
	prmt.b32 	%r395, %r394, %r392, 0x3340U;
	prmt.b32 	%r123, %r395, %r390, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r396, %rs30;
	cvt.u32.u16 	%r397, %rs26;
	prmt.b32 	%r398, %r397, %r396, 0x3340U;
	cvt.u32.u16 	%r399, %rs22;
	cvt.u32.u16 	%r400, %rs18;
	prmt.b32 	%r401, %r400, %r399, 0x3340U;
	prmt.b32 	%r402, %r401, %r398, 0x5410U;
	and.b32 	%r403, %r402, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r404, %r403, 0, 0x7773U;
	cvt.u16.u32 	%rs57, %r404;
	add.s16 	%rs58, %rs57, -8;
	cvt.u32.u16 	%r405, %rs58;
	prmt.b32 	%r406, %r403, 0, 0x7772U;
	cvt.u16.u32 	%rs59, %r406;
	add.s16 	%rs60, %rs59, -8;
	cvt.u32.u16 	%r407, %rs60;
	prmt.b32 	%r408, %r407, %r405, 0x3340U;
	prmt.b32 	%r409, %r403, 0, 0x7771U;
	cvt.u16.u32 	%rs61, %r409;
	add.s16 	%rs62, %rs61, -8;
	cvt.u32.u16 	%r410, %rs62;
	prmt.b32 	%r411, %r403, 0, 0x7770U;
	cvt.u16.u32 	%rs63, %r411;
	add.s16 	%rs64, %rs63, -8;
	cvt.u32.u16 	%r412, %rs64;
	prmt.b32 	%r413, %r412, %r410, 0x3340U;
	prmt.b32 	%r124, %r413, %r408, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r414, %rs15;
	cvt.u32.u16 	%r415, %rs11;
	prmt.b32 	%r416, %r415, %r414, 0x3340U;
	cvt.u32.u16 	%r417, %rs7;
	cvt.u32.u16 	%r418, %rs3;
	prmt.b32 	%r419, %r418, %r417, 0x3340U;
	prmt.b32 	%r420, %r419, %r416, 0x5410U;
	and.b32 	%r421, %r420, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r422, %r421, 0, 0x7773U;
	cvt.u16.u32 	%rs65, %r422;
	add.s16 	%rs66, %rs65, -8;
	cvt.u32.u16 	%r423, %rs66;
	prmt.b32 	%r424, %r421, 0, 0x7772U;
	cvt.u16.u32 	%rs67, %r424;
	add.s16 	%rs68, %rs67, -8;
	cvt.u32.u16 	%r425, %rs68;
	prmt.b32 	%r426, %r425, %r423, 0x3340U;
	prmt.b32 	%r427, %r421, 0, 0x7771U;
	cvt.u16.u32 	%rs69, %r427;
	add.s16 	%rs70, %rs69, -8;
	cvt.u32.u16 	%r428, %rs70;
	prmt.b32 	%r429, %r421, 0, 0x7770U;
	cvt.u16.u32 	%rs71, %r429;
	add.s16 	%rs72, %rs71, -8;
	cvt.u32.u16 	%r430, %rs72;
	prmt.b32 	%r431, %r430, %r428, 0x3340U;
	prmt.b32 	%r125, %r431, %r426, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r432, %rs16;
	cvt.u32.u16 	%r433, %rs12;
	prmt.b32 	%r434, %r433, %r432, 0x3340U;
	cvt.u32.u16 	%r435, %rs8;
	cvt.u32.u16 	%r436, %rs4;
	prmt.b32 	%r437, %r436, %r435, 0x3340U;
	prmt.b32 	%r438, %r437, %r434, 0x5410U;
	and.b32 	%r439, %r438, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r440, %r439, 0, 0x7773U;
	cvt.u16.u32 	%rs73, %r440;
	add.s16 	%rs74, %rs73, -8;
	cvt.u32.u16 	%r441, %rs74;
	prmt.b32 	%r442, %r439, 0, 0x7772U;
	cvt.u16.u32 	%rs75, %r442;
	add.s16 	%rs76, %rs75, -8;
	cvt.u32.u16 	%r443, %rs76;
	prmt.b32 	%r444, %r443, %r441, 0x3340U;
	prmt.b32 	%r445, %r439, 0, 0x7771U;
	cvt.u16.u32 	%rs77, %r445;
	add.s16 	%rs78, %rs77, -8;
	cvt.u32.u16 	%r446, %rs78;
	prmt.b32 	%r447, %r439, 0, 0x7770U;
	cvt.u16.u32 	%rs79, %r447;
	add.s16 	%rs80, %rs79, -8;
	cvt.u32.u16 	%r448, %rs80;
	prmt.b32 	%r449, %r448, %r446, 0x3340U;
	prmt.b32 	%r126, %r449, %r444, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r450, %rs31;
	cvt.u32.u16 	%r451, %rs27;
	prmt.b32 	%r452, %r451, %r450, 0x3340U;
	cvt.u32.u16 	%r453, %rs23;
	cvt.u32.u16 	%r454, %rs19;
	prmt.b32 	%r455, %r454, %r453, 0x3340U;
	prmt.b32 	%r456, %r455, %r452, 0x5410U;
	and.b32 	%r457, %r456, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r458, %r457, 0, 0x7773U;
	cvt.u16.u32 	%rs81, %r458;
	add.s16 	%rs82, %rs81, -8;
	cvt.u32.u16 	%r459, %rs82;
	prmt.b32 	%r460, %r457, 0, 0x7772U;
	cvt.u16.u32 	%rs83, %r460;
	add.s16 	%rs84, %rs83, -8;
	cvt.u32.u16 	%r461, %rs84;
	prmt.b32 	%r462, %r461, %r459, 0x3340U;
	prmt.b32 	%r463, %r457, 0, 0x7771U;
	cvt.u16.u32 	%rs85, %r463;
	add.s16 	%rs86, %rs85, -8;
	cvt.u32.u16 	%r464, %rs86;
	prmt.b32 	%r465, %r457, 0, 0x7770U;
	cvt.u16.u32 	%rs87, %r465;
	add.s16 	%rs88, %rs87, -8;
	cvt.u32.u16 	%r466, %rs88;
	prmt.b32 	%r467, %r466, %r464, 0x3340U;
	prmt.b32 	%r127, %r467, %r462, 0x5410U;
	.loc	1 828 26                        // sk04_fa_o_w4a8.py:828:26
	cvt.u32.u16 	%r468, %rs32;
	cvt.u32.u16 	%r469, %rs28;
	prmt.b32 	%r470, %r469, %r468, 0x3340U;
	cvt.u32.u16 	%r471, %rs24;
	cvt.u32.u16 	%r472, %rs20;
	prmt.b32 	%r473, %r472, %r471, 0x3340U;
	prmt.b32 	%r474, %r473, %r470, 0x5410U;
	and.b32 	%r475, %r474, 252645135;
	.loc	1 828 32                        // sk04_fa_o_w4a8.py:828:32
	prmt.b32 	%r476, %r475, 0, 0x7773U;
	cvt.u16.u32 	%rs89, %r476;
	add.s16 	%rs90, %rs89, -8;
	cvt.u32.u16 	%r477, %rs90;
	prmt.b32 	%r478, %r475, 0, 0x7772U;
	cvt.u16.u32 	%rs91, %r478;
	add.s16 	%rs92, %rs91, -8;
	cvt.u32.u16 	%r479, %rs92;
	prmt.b32 	%r480, %r479, %r477, 0x3340U;
	prmt.b32 	%r481, %r475, 0, 0x7771U;
	cvt.u16.u32 	%rs93, %r481;
	add.s16 	%rs94, %rs93, -8;
	cvt.u32.u16 	%r482, %rs94;
	prmt.b32 	%r483, %r475, 0, 0x7770U;
	cvt.u16.u32 	%rs95, %r483;
	add.s16 	%rs96, %rs95, -8;
	cvt.u32.u16 	%r484, %rs96;
	prmt.b32 	%r485, %r484, %r482, 0x3340U;
	prmt.b32 	%r128, %r485, %r480, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs97, %rs1, 4;
	shr.u16 	%rs98, %rs5, 4;
	shr.u16 	%rs99, %rs9, 4;
	shr.u16 	%rs100, %rs13, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs101, %rs100, -8;
	cvt.u32.u16 	%r486, %rs101;
	add.s16 	%rs102, %rs99, -8;
	cvt.u32.u16 	%r487, %rs102;
	prmt.b32 	%r488, %r487, %r486, 0x3340U;
	add.s16 	%rs103, %rs98, -8;
	cvt.u32.u16 	%r489, %rs103;
	add.s16 	%rs104, %rs97, -8;
	cvt.u32.u16 	%r490, %rs104;
	prmt.b32 	%r491, %r490, %r489, 0x3340U;
	prmt.b32 	%r177, %r491, %r488, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs105, %rs2, 4;
	shr.u16 	%rs106, %rs6, 4;
	shr.u16 	%rs107, %rs10, 4;
	shr.u16 	%rs108, %rs14, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs109, %rs108, -8;
	cvt.u32.u16 	%r492, %rs109;
	add.s16 	%rs110, %rs107, -8;
	cvt.u32.u16 	%r493, %rs110;
	prmt.b32 	%r494, %r493, %r492, 0x3340U;
	add.s16 	%rs111, %rs106, -8;
	cvt.u32.u16 	%r495, %rs111;
	add.s16 	%rs112, %rs105, -8;
	cvt.u32.u16 	%r496, %rs112;
	prmt.b32 	%r497, %r496, %r495, 0x3340U;
	prmt.b32 	%r178, %r497, %r494, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs113, %rs17, 4;
	shr.u16 	%rs114, %rs21, 4;
	shr.u16 	%rs115, %rs25, 4;
	shr.u16 	%rs116, %rs29, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs117, %rs116, -8;
	cvt.u32.u16 	%r498, %rs117;
	add.s16 	%rs118, %rs115, -8;
	cvt.u32.u16 	%r499, %rs118;
	prmt.b32 	%r500, %r499, %r498, 0x3340U;
	add.s16 	%rs119, %rs114, -8;
	cvt.u32.u16 	%r501, %rs119;
	add.s16 	%rs120, %rs113, -8;
	cvt.u32.u16 	%r502, %rs120;
	prmt.b32 	%r503, %r502, %r501, 0x3340U;
	prmt.b32 	%r187, %r503, %r500, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs121, %rs18, 4;
	shr.u16 	%rs122, %rs22, 4;
	shr.u16 	%rs123, %rs26, 4;
	shr.u16 	%rs124, %rs30, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs125, %rs124, -8;
	cvt.u32.u16 	%r504, %rs125;
	add.s16 	%rs126, %rs123, -8;
	cvt.u32.u16 	%r505, %rs126;
	prmt.b32 	%r506, %r505, %r504, 0x3340U;
	add.s16 	%rs127, %rs122, -8;
	cvt.u32.u16 	%r507, %rs127;
	add.s16 	%rs128, %rs121, -8;
	cvt.u32.u16 	%r508, %rs128;
	prmt.b32 	%r509, %r508, %r507, 0x3340U;
	prmt.b32 	%r188, %r509, %r506, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs129, %rs3, 4;
	shr.u16 	%rs130, %rs7, 4;
	shr.u16 	%rs131, %rs11, 4;
	shr.u16 	%rs132, %rs15, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs133, %rs132, -8;
	cvt.u32.u16 	%r510, %rs133;
	add.s16 	%rs134, %rs131, -8;
	cvt.u32.u16 	%r511, %rs134;
	prmt.b32 	%r512, %r511, %r510, 0x3340U;
	add.s16 	%rs135, %rs130, -8;
	cvt.u32.u16 	%r513, %rs135;
	add.s16 	%rs136, %rs129, -8;
	cvt.u32.u16 	%r514, %rs136;
	prmt.b32 	%r515, %r514, %r513, 0x3340U;
	prmt.b32 	%r193, %r515, %r512, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs137, %rs4, 4;
	shr.u16 	%rs138, %rs8, 4;
	shr.u16 	%rs139, %rs12, 4;
	shr.u16 	%rs140, %rs16, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs141, %rs140, -8;
	cvt.u32.u16 	%r516, %rs141;
	add.s16 	%rs142, %rs139, -8;
	cvt.u32.u16 	%r517, %rs142;
	prmt.b32 	%r518, %r517, %r516, 0x3340U;
	add.s16 	%rs143, %rs138, -8;
	cvt.u32.u16 	%r519, %rs143;
	add.s16 	%rs144, %rs137, -8;
	cvt.u32.u16 	%r520, %rs144;
	prmt.b32 	%r521, %r520, %r519, 0x3340U;
	prmt.b32 	%r194, %r521, %r518, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs145, %rs19, 4;
	shr.u16 	%rs146, %rs23, 4;
	shr.u16 	%rs147, %rs27, 4;
	shr.u16 	%rs148, %rs31, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs149, %rs148, -8;
	cvt.u32.u16 	%r522, %rs149;
	add.s16 	%rs150, %rs147, -8;
	cvt.u32.u16 	%r523, %rs150;
	prmt.b32 	%r524, %r523, %r522, 0x3340U;
	add.s16 	%rs151, %rs146, -8;
	cvt.u32.u16 	%r525, %rs151;
	add.s16 	%rs152, %rs145, -8;
	cvt.u32.u16 	%r526, %rs152;
	prmt.b32 	%r527, %r526, %r525, 0x3340U;
	prmt.b32 	%r199, %r527, %r524, 0x5410U;
	.loc	1 829 28                        // sk04_fa_o_w4a8.py:829:28
	shr.u16 	%rs153, %rs20, 4;
	shr.u16 	%rs154, %rs24, 4;
	shr.u16 	%rs155, %rs28, 4;
	shr.u16 	%rs156, %rs32, 4;
	.loc	1 829 39                        // sk04_fa_o_w4a8.py:829:39
	add.s16 	%rs157, %rs156, -8;
	cvt.u32.u16 	%r528, %rs157;
	add.s16 	%rs158, %rs155, -8;
	cvt.u32.u16 	%r529, %rs158;
	prmt.b32 	%r530, %r529, %r528, 0x3340U;
	add.s16 	%rs159, %rs154, -8;
	cvt.u32.u16 	%r531, %rs159;
	add.s16 	%rs160, %rs153, -8;
	cvt.u32.u16 	%r532, %rs160;
	prmt.b32 	%r533, %r532, %r531, 0x3340U;
	prmt.b32 	%r200, %r533, %r530, 0x5410U;
	.loc	1 831 33                        // sk04_fa_o_w4a8.py:831:33
	add.s32 	%r534, %r333, %r27;
	ld.shared.v2.b32 	{%r535, %r536}, [%r534+16384];
	ld.shared.v2.b32 	{%r537, %r538}, [%r534+16896];
	ld.shared.v2.b32 	{%r539, %r540}, [%r534+17408];
	ld.shared.v2.b32 	{%r541, %r542}, [%r534+17920];
	ld.shared.v2.b32 	{%r543, %r544}, [%r534+18432];
	ld.shared.v2.b32 	{%r545, %r546}, [%r534+18944];
	ld.shared.v2.b32 	{%r547, %r548}, [%r534+19456];
	ld.shared.v2.b32 	{%r549, %r550}, [%r534+19968];
	ld.shared.v2.b32 	{%r551, %r552}, [%r534+20480];
	ld.shared.v2.b32 	{%r553, %r554}, [%r534+20992];
	ld.shared.v2.b32 	{%r555, %r556}, [%r534+21504];
	ld.shared.v2.b32 	{%r557, %r558}, [%r534+22016];
	ld.shared.v2.b32 	{%r559, %r560}, [%r534+22528];
	ld.shared.v2.b32 	{%r561, %r562}, [%r534+23040];
	ld.shared.v2.b32 	{%r563, %r564}, [%r534+23552];
	ld.shared.v2.b32 	{%r565, %r566}, [%r534+24064];
	add.s32 	%r567, %r333, %r28;
	ld.shared.v2.b32 	{%r568, %r569}, [%r567+16384];
	ld.shared.v2.b32 	{%r570, %r571}, [%r567+16896];
	ld.shared.v2.b32 	{%r572, %r573}, [%r567+17408];
	ld.shared.v2.b32 	{%r574, %r575}, [%r567+17920];
	ld.shared.v2.b32 	{%r576, %r577}, [%r567+18432];
	ld.shared.v2.b32 	{%r578, %r579}, [%r567+18944];
	ld.shared.v2.b32 	{%r580, %r581}, [%r567+19456];
	ld.shared.v2.b32 	{%r582, %r583}, [%r567+19968];
	ld.shared.v2.b32 	{%r584, %r585}, [%r567+20480];
	ld.shared.v2.b32 	{%r586, %r587}, [%r567+20992];
	ld.shared.v2.b32 	{%r588, %r589}, [%r567+21504];
	ld.shared.v2.b32 	{%r590, %r591}, [%r567+22016];
	ld.shared.v2.b32 	{%r592, %r593}, [%r567+22528];
	ld.shared.v2.b32 	{%r594, %r595}, [%r567+23040];
	ld.shared.v2.b32 	{%r596, %r597}, [%r567+23552];
	ld.shared.v2.b32 	{%r598, %r599}, [%r567+24064];
	.loc	1 832 33                        // sk04_fa_o_w4a8.py:832:33
	prmt.b32 	%r113, %r535, %r536, 0x6420U;
	prmt.b32 	%r114, %r537, %r538, 0x6420U;
	prmt.b32 	%r115, %r568, %r569, 0x6420U;
	prmt.b32 	%r116, %r570, %r571, 0x6420U;
	prmt.b32 	%r119, %r539, %r540, 0x6420U;
	prmt.b32 	%r120, %r541, %r542, 0x6420U;
	prmt.b32 	%r121, %r572, %r573, 0x6420U;
	prmt.b32 	%r122, %r574, %r575, 0x6420U;
	prmt.b32 	%r129, %r543, %r544, 0x6420U;
	prmt.b32 	%r130, %r545, %r546, 0x6420U;
	prmt.b32 	%r131, %r576, %r577, 0x6420U;
	prmt.b32 	%r132, %r578, %r579, 0x6420U;
	prmt.b32 	%r133, %r547, %r548, 0x6420U;
	prmt.b32 	%r134, %r549, %r550, 0x6420U;
	prmt.b32 	%r135, %r580, %r581, 0x6420U;
	prmt.b32 	%r136, %r582, %r583, 0x6420U;
	prmt.b32 	%r137, %r551, %r552, 0x6420U;
	prmt.b32 	%r138, %r553, %r554, 0x6420U;
	prmt.b32 	%r139, %r584, %r585, 0x6420U;
	prmt.b32 	%r140, %r586, %r587, 0x6420U;
	prmt.b32 	%r141, %r555, %r556, 0x6420U;
	prmt.b32 	%r142, %r557, %r558, 0x6420U;
	prmt.b32 	%r143, %r588, %r589, 0x6420U;
	prmt.b32 	%r144, %r590, %r591, 0x6420U;
	prmt.b32 	%r145, %r559, %r560, 0x6420U;
	prmt.b32 	%r146, %r561, %r562, 0x6420U;
	prmt.b32 	%r147, %r592, %r593, 0x6420U;
	prmt.b32 	%r148, %r594, %r595, 0x6420U;
	prmt.b32 	%r149, %r563, %r564, 0x6420U;
	prmt.b32 	%r150, %r565, %r566, 0x6420U;
	prmt.b32 	%r151, %r596, %r597, 0x6420U;
	prmt.b32 	%r152, %r598, %r599, 0x6420U;
	mov.b32 	%r153, %r112;
	mov.b32 	%r154, %r112;
	mov.b32 	%r155, %r112;
	mov.b32 	%r156, %r112;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r153, %r154, %r155, %r156 }, { %r113, %r114, %r115, %r116 }, { %r117, %r118 }, { %r153, %r154, %r155, %r156 };
	// end inline asm
	mov.b32 	%r157, %r112;
	mov.b32 	%r158, %r112;
	mov.b32 	%r159, %r112;
	mov.b32 	%r160, %r112;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r157, %r158, %r159, %r160 }, { %r113, %r114, %r115, %r116 }, { %r123, %r124 }, { %r157, %r158, %r159, %r160 };
	// end inline asm
	mov.b32 	%r165, %r112;
	mov.b32 	%r166, %r112;
	mov.b32 	%r167, %r112;
	mov.b32 	%r168, %r112;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r165, %r166, %r167, %r168 }, { %r113, %r114, %r115, %r116 }, { %r125, %r126 }, { %r165, %r166, %r167, %r168 };
	// end inline asm
	mov.b32 	%r169, %r112;
	mov.b32 	%r170, %r112;
	mov.b32 	%r171, %r112;
	mov.b32 	%r172, %r112;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r169, %r170, %r171, %r172 }, { %r113, %r114, %r115, %r116 }, { %r127, %r128 }, { %r169, %r170, %r171, %r172 };
	// end inline asm
	mov.b32 	%r173, %r112;
	mov.b32 	%r174, %r112;
	mov.b32 	%r175, %r112;
	mov.b32 	%r176, %r112;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r173, %r174, %r175, %r176 }, { %r119, %r120, %r121, %r122 }, { %r117, %r118 }, { %r173, %r174, %r175, %r176 };
	// end inline asm
	mov.b32 	%r179, %r112;
	mov.b32 	%r180, %r112;
	mov.b32 	%r181, %r112;
	mov.b32 	%r182, %r112;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r179, %r180, %r181, %r182 }, { %r119, %r120, %r121, %r122 }, { %r123, %r124 }, { %r179, %r180, %r181, %r182 };
	// end inline asm
	mov.b32 	%r189, %r112;
	mov.b32 	%r190, %r112;
	mov.b32 	%r191, %r112;
	mov.b32 	%r192, %r112;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r189, %r190, %r191, %r192 }, { %r119, %r120, %r121, %r122 }, { %r125, %r126 }, { %r189, %r190, %r191, %r192 };
	// end inline asm
	mov.b32 	%r195, %r112;
	mov.b32 	%r196, %r112;
	mov.b32 	%r197, %r112;
	mov.b32 	%r198, %r112;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r195, %r196, %r197, %r198 }, { %r119, %r120, %r121, %r122 }, { %r127, %r128 }, { %r195, %r196, %r197, %r198 };
	// end inline asm
	mov.b32 	%r201, %r112;
	mov.b32 	%r202, %r112;
	mov.b32 	%r203, %r112;
	mov.b32 	%r204, %r112;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r201, %r202, %r203, %r204 }, { %r129, %r130, %r131, %r132 }, { %r117, %r118 }, { %r201, %r202, %r203, %r204 };
	// end inline asm
	mov.b32 	%r205, %r112;
	mov.b32 	%r206, %r112;
	mov.b32 	%r207, %r112;
	mov.b32 	%r208, %r112;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r205, %r206, %r207, %r208 }, { %r129, %r130, %r131, %r132 }, { %r123, %r124 }, { %r205, %r206, %r207, %r208 };
	// end inline asm
	mov.b32 	%r213, %r112;
	mov.b32 	%r214, %r112;
	mov.b32 	%r215, %r112;
	mov.b32 	%r216, %r112;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r213, %r214, %r215, %r216 }, { %r129, %r130, %r131, %r132 }, { %r125, %r126 }, { %r213, %r214, %r215, %r216 };
	// end inline asm
	mov.b32 	%r217, %r112;
	mov.b32 	%r218, %r112;
	mov.b32 	%r219, %r112;
	mov.b32 	%r220, %r112;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r217, %r218, %r219, %r220 }, { %r129, %r130, %r131, %r132 }, { %r127, %r128 }, { %r217, %r218, %r219, %r220 };
	// end inline asm
	mov.b32 	%r221, %r112;
	mov.b32 	%r222, %r112;
	mov.b32 	%r223, %r112;
	mov.b32 	%r224, %r112;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r221, %r222, %r223, %r224 }, { %r133, %r134, %r135, %r136 }, { %r117, %r118 }, { %r221, %r222, %r223, %r224 };
	// end inline asm
	mov.b32 	%r225, %r112;
	mov.b32 	%r226, %r112;
	mov.b32 	%r227, %r112;
	mov.b32 	%r228, %r112;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r225, %r226, %r227, %r228 }, { %r133, %r134, %r135, %r136 }, { %r123, %r124 }, { %r225, %r226, %r227, %r228 };
	// end inline asm
	mov.b32 	%r233, %r112;
	mov.b32 	%r234, %r112;
	mov.b32 	%r235, %r112;
	mov.b32 	%r236, %r112;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r233, %r234, %r235, %r236 }, { %r133, %r134, %r135, %r136 }, { %r125, %r126 }, { %r233, %r234, %r235, %r236 };
	// end inline asm
	mov.b32 	%r237, %r112;
	mov.b32 	%r238, %r112;
	mov.b32 	%r239, %r112;
	mov.b32 	%r240, %r112;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r237, %r238, %r239, %r240 }, { %r133, %r134, %r135, %r136 }, { %r127, %r128 }, { %r237, %r238, %r239, %r240 };
	// end inline asm
	mov.b32 	%r241, %r112;
	mov.b32 	%r242, %r112;
	mov.b32 	%r243, %r112;
	mov.b32 	%r244, %r112;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r241, %r242, %r243, %r244 }, { %r137, %r138, %r139, %r140 }, { %r117, %r118 }, { %r241, %r242, %r243, %r244 };
	// end inline asm
	mov.b32 	%r245, %r112;
	mov.b32 	%r246, %r112;
	mov.b32 	%r247, %r112;
	mov.b32 	%r248, %r112;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r245, %r246, %r247, %r248 }, { %r137, %r138, %r139, %r140 }, { %r123, %r124 }, { %r245, %r246, %r247, %r248 };
	// end inline asm
	mov.b32 	%r253, %r112;
	mov.b32 	%r254, %r112;
	mov.b32 	%r255, %r112;
	mov.b32 	%r256, %r112;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r253, %r254, %r255, %r256 }, { %r137, %r138, %r139, %r140 }, { %r125, %r126 }, { %r253, %r254, %r255, %r256 };
	// end inline asm
	mov.b32 	%r257, %r112;
	mov.b32 	%r258, %r112;
	mov.b32 	%r259, %r112;
	mov.b32 	%r260, %r112;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r257, %r258, %r259, %r260 }, { %r137, %r138, %r139, %r140 }, { %r127, %r128 }, { %r257, %r258, %r259, %r260 };
	// end inline asm
	mov.b32 	%r261, %r112;
	mov.b32 	%r262, %r112;
	mov.b32 	%r263, %r112;
	mov.b32 	%r264, %r112;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r261, %r262, %r263, %r264 }, { %r141, %r142, %r143, %r144 }, { %r117, %r118 }, { %r261, %r262, %r263, %r264 };
	// end inline asm
	mov.b32 	%r265, %r112;
	mov.b32 	%r266, %r112;
	mov.b32 	%r267, %r112;
	mov.b32 	%r268, %r112;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r265, %r266, %r267, %r268 }, { %r141, %r142, %r143, %r144 }, { %r123, %r124 }, { %r265, %r266, %r267, %r268 };
	// end inline asm
	mov.b32 	%r273, %r112;
	mov.b32 	%r274, %r112;
	mov.b32 	%r275, %r112;
	mov.b32 	%r276, %r112;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r273, %r274, %r275, %r276 }, { %r141, %r142, %r143, %r144 }, { %r125, %r126 }, { %r273, %r274, %r275, %r276 };
	// end inline asm
	mov.b32 	%r277, %r112;
	mov.b32 	%r278, %r112;
	mov.b32 	%r279, %r112;
	mov.b32 	%r280, %r112;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r277, %r278, %r279, %r280 }, { %r141, %r142, %r143, %r144 }, { %r127, %r128 }, { %r277, %r278, %r279, %r280 };
	// end inline asm
	mov.b32 	%r281, %r112;
	mov.b32 	%r282, %r112;
	mov.b32 	%r283, %r112;
	mov.b32 	%r284, %r112;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r281, %r282, %r283, %r284 }, { %r145, %r146, %r147, %r148 }, { %r117, %r118 }, { %r281, %r282, %r283, %r284 };
	// end inline asm
	mov.b32 	%r285, %r112;
	mov.b32 	%r286, %r112;
	mov.b32 	%r287, %r112;
	mov.b32 	%r288, %r112;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r285, %r286, %r287, %r288 }, { %r145, %r146, %r147, %r148 }, { %r123, %r124 }, { %r285, %r286, %r287, %r288 };
	// end inline asm
	mov.b32 	%r293, %r112;
	mov.b32 	%r294, %r112;
	mov.b32 	%r295, %r112;
	mov.b32 	%r296, %r112;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r293, %r294, %r295, %r296 }, { %r145, %r146, %r147, %r148 }, { %r125, %r126 }, { %r293, %r294, %r295, %r296 };
	// end inline asm
	mov.b32 	%r297, %r112;
	mov.b32 	%r298, %r112;
	mov.b32 	%r299, %r112;
	mov.b32 	%r300, %r112;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r297, %r298, %r299, %r300 }, { %r145, %r146, %r147, %r148 }, { %r127, %r128 }, { %r297, %r298, %r299, %r300 };
	// end inline asm
	mov.b32 	%r301, %r112;
	mov.b32 	%r302, %r112;
	mov.b32 	%r303, %r112;
	mov.b32 	%r304, %r112;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r301, %r302, %r303, %r304 }, { %r149, %r150, %r151, %r152 }, { %r117, %r118 }, { %r301, %r302, %r303, %r304 };
	// end inline asm
	mov.b32 	%r305, %r112;
	mov.b32 	%r306, %r112;
	mov.b32 	%r307, %r112;
	mov.b32 	%r308, %r112;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r305, %r306, %r307, %r308 }, { %r149, %r150, %r151, %r152 }, { %r123, %r124 }, { %r305, %r306, %r307, %r308 };
	// end inline asm
	mov.b32 	%r313, %r112;
	mov.b32 	%r314, %r112;
	mov.b32 	%r315, %r112;
	mov.b32 	%r316, %r112;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r313, %r314, %r315, %r316 }, { %r149, %r150, %r151, %r152 }, { %r125, %r126 }, { %r313, %r314, %r315, %r316 };
	// end inline asm
	mov.b32 	%r320, %r112;
	mov.b32 	%r317, %r112;
	mov.b32 	%r318, %r112;
	mov.b32 	%r319, %r112;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r317, %r318, %r319, %r320 }, { %r149, %r150, %r151, %r152 }, { %r127, %r128 }, { %r317, %r318, %r319, %r320 };
	// end inline asm
	.loc	1 832 75                        // sk04_fa_o_w4a8.py:832:75
	prmt.b32 	%r161, %r535, %r536, 0x7531U;
	prmt.b32 	%r162, %r537, %r538, 0x7531U;
	prmt.b32 	%r163, %r568, %r569, 0x7531U;
	prmt.b32 	%r164, %r570, %r571, 0x7531U;
	prmt.b32 	%r183, %r539, %r540, 0x7531U;
	prmt.b32 	%r184, %r541, %r542, 0x7531U;
	prmt.b32 	%r185, %r572, %r573, 0x7531U;
	prmt.b32 	%r186, %r574, %r575, 0x7531U;
	prmt.b32 	%r209, %r543, %r544, 0x7531U;
	prmt.b32 	%r210, %r545, %r546, 0x7531U;
	prmt.b32 	%r211, %r576, %r577, 0x7531U;
	prmt.b32 	%r212, %r578, %r579, 0x7531U;
	prmt.b32 	%r229, %r547, %r548, 0x7531U;
	prmt.b32 	%r230, %r549, %r550, 0x7531U;
	prmt.b32 	%r231, %r580, %r581, 0x7531U;
	prmt.b32 	%r232, %r582, %r583, 0x7531U;
	prmt.b32 	%r249, %r551, %r552, 0x7531U;
	prmt.b32 	%r250, %r553, %r554, 0x7531U;
	prmt.b32 	%r251, %r584, %r585, 0x7531U;
	prmt.b32 	%r252, %r586, %r587, 0x7531U;
	prmt.b32 	%r269, %r555, %r556, 0x7531U;
	prmt.b32 	%r270, %r557, %r558, 0x7531U;
	prmt.b32 	%r271, %r588, %r589, 0x7531U;
	prmt.b32 	%r272, %r590, %r591, 0x7531U;
	prmt.b32 	%r289, %r559, %r560, 0x7531U;
	prmt.b32 	%r290, %r561, %r562, 0x7531U;
	prmt.b32 	%r291, %r592, %r593, 0x7531U;
	prmt.b32 	%r292, %r594, %r595, 0x7531U;
	prmt.b32 	%r309, %r563, %r564, 0x7531U;
	prmt.b32 	%r310, %r565, %r566, 0x7531U;
	prmt.b32 	%r311, %r596, %r597, 0x7531U;
	prmt.b32 	%r312, %r598, %r599, 0x7531U;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r153, %r154, %r155, %r156 }, { %r161, %r162, %r163, %r164 }, { %r177, %r178 }, { %r153, %r154, %r155, %r156 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r157, %r158, %r159, %r160 }, { %r161, %r162, %r163, %r164 }, { %r187, %r188 }, { %r157, %r158, %r159, %r160 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r165, %r166, %r167, %r168 }, { %r161, %r162, %r163, %r164 }, { %r193, %r194 }, { %r165, %r166, %r167, %r168 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r169, %r170, %r171, %r172 }, { %r161, %r162, %r163, %r164 }, { %r199, %r200 }, { %r169, %r170, %r171, %r172 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r173, %r174, %r175, %r176 }, { %r183, %r184, %r185, %r186 }, { %r177, %r178 }, { %r173, %r174, %r175, %r176 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r179, %r180, %r181, %r182 }, { %r183, %r184, %r185, %r186 }, { %r187, %r188 }, { %r179, %r180, %r181, %r182 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r189, %r190, %r191, %r192 }, { %r183, %r184, %r185, %r186 }, { %r193, %r194 }, { %r189, %r190, %r191, %r192 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r195, %r196, %r197, %r198 }, { %r183, %r184, %r185, %r186 }, { %r199, %r200 }, { %r195, %r196, %r197, %r198 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r201, %r202, %r203, %r204 }, { %r209, %r210, %r211, %r212 }, { %r177, %r178 }, { %r201, %r202, %r203, %r204 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r205, %r206, %r207, %r208 }, { %r209, %r210, %r211, %r212 }, { %r187, %r188 }, { %r205, %r206, %r207, %r208 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r213, %r214, %r215, %r216 }, { %r209, %r210, %r211, %r212 }, { %r193, %r194 }, { %r213, %r214, %r215, %r216 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r217, %r218, %r219, %r220 }, { %r209, %r210, %r211, %r212 }, { %r199, %r200 }, { %r217, %r218, %r219, %r220 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r221, %r222, %r223, %r224 }, { %r229, %r230, %r231, %r232 }, { %r177, %r178 }, { %r221, %r222, %r223, %r224 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r225, %r226, %r227, %r228 }, { %r229, %r230, %r231, %r232 }, { %r187, %r188 }, { %r225, %r226, %r227, %r228 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r233, %r234, %r235, %r236 }, { %r229, %r230, %r231, %r232 }, { %r193, %r194 }, { %r233, %r234, %r235, %r236 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r237, %r238, %r239, %r240 }, { %r229, %r230, %r231, %r232 }, { %r199, %r200 }, { %r237, %r238, %r239, %r240 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r241, %r242, %r243, %r244 }, { %r249, %r250, %r251, %r252 }, { %r177, %r178 }, { %r241, %r242, %r243, %r244 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r245, %r246, %r247, %r248 }, { %r249, %r250, %r251, %r252 }, { %r187, %r188 }, { %r245, %r246, %r247, %r248 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r253, %r254, %r255, %r256 }, { %r249, %r250, %r251, %r252 }, { %r193, %r194 }, { %r253, %r254, %r255, %r256 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r257, %r258, %r259, %r260 }, { %r249, %r250, %r251, %r252 }, { %r199, %r200 }, { %r257, %r258, %r259, %r260 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r261, %r262, %r263, %r264 }, { %r269, %r270, %r271, %r272 }, { %r177, %r178 }, { %r261, %r262, %r263, %r264 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r265, %r266, %r267, %r268 }, { %r269, %r270, %r271, %r272 }, { %r187, %r188 }, { %r265, %r266, %r267, %r268 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r273, %r274, %r275, %r276 }, { %r269, %r270, %r271, %r272 }, { %r193, %r194 }, { %r273, %r274, %r275, %r276 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r277, %r278, %r279, %r280 }, { %r269, %r270, %r271, %r272 }, { %r199, %r200 }, { %r277, %r278, %r279, %r280 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r281, %r282, %r283, %r284 }, { %r289, %r290, %r291, %r292 }, { %r177, %r178 }, { %r281, %r282, %r283, %r284 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r285, %r286, %r287, %r288 }, { %r289, %r290, %r291, %r292 }, { %r187, %r188 }, { %r285, %r286, %r287, %r288 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r293, %r294, %r295, %r296 }, { %r289, %r290, %r291, %r292 }, { %r193, %r194 }, { %r293, %r294, %r295, %r296 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r297, %r298, %r299, %r300 }, { %r289, %r290, %r291, %r292 }, { %r199, %r200 }, { %r297, %r298, %r299, %r300 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r301, %r302, %r303, %r304 }, { %r309, %r310, %r311, %r312 }, { %r177, %r178 }, { %r301, %r302, %r303, %r304 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r305, %r306, %r307, %r308 }, { %r309, %r310, %r311, %r312 }, { %r187, %r188 }, { %r305, %r306, %r307, %r308 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r313, %r314, %r315, %r316 }, { %r309, %r310, %r311, %r312 }, { %r193, %r194 }, { %r313, %r314, %r315, %r316 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r317, %r318, %r319, %r320 }, { %r309, %r310, %r311, %r312 }, { %r199, %r200 }, { %r317, %r318, %r319, %r320 };
	// end inline asm
	.loc	1 833 59                        // sk04_fa_o_w4a8.py:833:59
	mad.wide.s32 	%rd43, %r1429, 4, %rd10;
	.loc	1 833 27                        // sk04_fa_o_w4a8.py:833:27
	// begin inline asm
	mov.u32 %r321, 0x0;
	ld.global.b32 { %r321 }, [ %rd43 + 0 ];
	// end inline asm
	.loc	1 834 40                        // sk04_fa_o_w4a8.py:834:40
	// begin inline asm
	st.shared.b32 [ %r322 + 0 ], %r321;
	// end inline asm
	bar.sync 	0;
	.loc	1 834 26                        // sk04_fa_o_w4a8.py:834:26
	cvt.rn.f32.s32 	%r600, %r320;
	cvt.rn.f32.s32 	%r601, %r319;
	cvt.rn.f32.s32 	%r602, %r318;
	cvt.rn.f32.s32 	%r603, %r317;
	cvt.rn.f32.s32 	%r604, %r316;
	cvt.rn.f32.s32 	%r605, %r315;
	cvt.rn.f32.s32 	%r606, %r314;
	cvt.rn.f32.s32 	%r607, %r313;
	cvt.rn.f32.s32 	%r608, %r308;
	cvt.rn.f32.s32 	%r609, %r307;
	cvt.rn.f32.s32 	%r610, %r306;
	cvt.rn.f32.s32 	%r611, %r305;
	cvt.rn.f32.s32 	%r612, %r304;
	cvt.rn.f32.s32 	%r613, %r303;
	cvt.rn.f32.s32 	%r614, %r302;
	cvt.rn.f32.s32 	%r615, %r301;
	cvt.rn.f32.s32 	%r616, %r300;
	cvt.rn.f32.s32 	%r617, %r299;
	cvt.rn.f32.s32 	%r618, %r298;
	cvt.rn.f32.s32 	%r619, %r297;
	cvt.rn.f32.s32 	%r620, %r296;
	cvt.rn.f32.s32 	%r621, %r295;
	cvt.rn.f32.s32 	%r622, %r294;
	cvt.rn.f32.s32 	%r623, %r293;
	cvt.rn.f32.s32 	%r624, %r288;
	cvt.rn.f32.s32 	%r625, %r287;
	cvt.rn.f32.s32 	%r626, %r286;
	cvt.rn.f32.s32 	%r627, %r285;
	cvt.rn.f32.s32 	%r628, %r284;
	cvt.rn.f32.s32 	%r629, %r283;
	cvt.rn.f32.s32 	%r630, %r282;
	cvt.rn.f32.s32 	%r631, %r281;
	cvt.rn.f32.s32 	%r632, %r280;
	cvt.rn.f32.s32 	%r633, %r279;
	cvt.rn.f32.s32 	%r634, %r278;
	cvt.rn.f32.s32 	%r635, %r277;
	cvt.rn.f32.s32 	%r636, %r276;
	cvt.rn.f32.s32 	%r637, %r275;
	cvt.rn.f32.s32 	%r638, %r274;
	cvt.rn.f32.s32 	%r639, %r273;
	cvt.rn.f32.s32 	%r640, %r268;
	cvt.rn.f32.s32 	%r641, %r267;
	cvt.rn.f32.s32 	%r642, %r153;
	cvt.rn.f32.s32 	%r643, %r154;
	cvt.rn.f32.s32 	%r644, %r155;
	cvt.rn.f32.s32 	%r645, %r156;
	cvt.rn.f32.s32 	%r646, %r157;
	cvt.rn.f32.s32 	%r647, %r158;
	cvt.rn.f32.s32 	%r648, %r159;
	cvt.rn.f32.s32 	%r649, %r160;
	cvt.rn.f32.s32 	%r650, %r165;
	cvt.rn.f32.s32 	%r651, %r166;
	cvt.rn.f32.s32 	%r652, %r167;
	cvt.rn.f32.s32 	%r653, %r168;
	cvt.rn.f32.s32 	%r654, %r169;
	cvt.rn.f32.s32 	%r655, %r170;
	cvt.rn.f32.s32 	%r656, %r171;
	cvt.rn.f32.s32 	%r657, %r172;
	cvt.rn.f32.s32 	%r658, %r173;
	cvt.rn.f32.s32 	%r659, %r174;
	cvt.rn.f32.s32 	%r660, %r175;
	cvt.rn.f32.s32 	%r661, %r176;
	cvt.rn.f32.s32 	%r662, %r179;
	cvt.rn.f32.s32 	%r663, %r180;
	cvt.rn.f32.s32 	%r664, %r181;
	cvt.rn.f32.s32 	%r665, %r182;
	cvt.rn.f32.s32 	%r666, %r189;
	cvt.rn.f32.s32 	%r667, %r190;
	cvt.rn.f32.s32 	%r668, %r191;
	cvt.rn.f32.s32 	%r669, %r192;
	cvt.rn.f32.s32 	%r670, %r195;
	cvt.rn.f32.s32 	%r671, %r196;
	cvt.rn.f32.s32 	%r672, %r197;
	cvt.rn.f32.s32 	%r673, %r198;
	cvt.rn.f32.s32 	%r674, %r201;
	cvt.rn.f32.s32 	%r675, %r202;
	cvt.rn.f32.s32 	%r676, %r203;
	cvt.rn.f32.s32 	%r677, %r204;
	cvt.rn.f32.s32 	%r678, %r205;
	cvt.rn.f32.s32 	%r679, %r206;
	cvt.rn.f32.s32 	%r680, %r207;
	cvt.rn.f32.s32 	%r681, %r208;
	cvt.rn.f32.s32 	%r682, %r213;
	cvt.rn.f32.s32 	%r683, %r214;
	cvt.rn.f32.s32 	%r684, %r215;
	cvt.rn.f32.s32 	%r685, %r216;
	cvt.rn.f32.s32 	%r686, %r217;
	cvt.rn.f32.s32 	%r687, %r218;
	cvt.rn.f32.s32 	%r688, %r219;
	cvt.rn.f32.s32 	%r689, %r220;
	cvt.rn.f32.s32 	%r690, %r221;
	cvt.rn.f32.s32 	%r691, %r222;
	cvt.rn.f32.s32 	%r692, %r223;
	cvt.rn.f32.s32 	%r693, %r224;
	cvt.rn.f32.s32 	%r694, %r225;
	cvt.rn.f32.s32 	%r695, %r226;
	cvt.rn.f32.s32 	%r696, %r227;
	cvt.rn.f32.s32 	%r697, %r228;
	cvt.rn.f32.s32 	%r698, %r233;
	cvt.rn.f32.s32 	%r699, %r234;
	cvt.rn.f32.s32 	%r700, %r235;
	cvt.rn.f32.s32 	%r701, %r236;
	cvt.rn.f32.s32 	%r702, %r237;
	cvt.rn.f32.s32 	%r703, %r238;
	cvt.rn.f32.s32 	%r704, %r239;
	cvt.rn.f32.s32 	%r705, %r240;
	cvt.rn.f32.s32 	%r706, %r241;
	cvt.rn.f32.s32 	%r707, %r242;
	cvt.rn.f32.s32 	%r708, %r243;
	cvt.rn.f32.s32 	%r709, %r244;
	cvt.rn.f32.s32 	%r710, %r245;
	cvt.rn.f32.s32 	%r711, %r246;
	cvt.rn.f32.s32 	%r712, %r247;
	cvt.rn.f32.s32 	%r713, %r248;
	cvt.rn.f32.s32 	%r714, %r253;
	cvt.rn.f32.s32 	%r715, %r254;
	cvt.rn.f32.s32 	%r716, %r255;
	cvt.rn.f32.s32 	%r717, %r256;
	cvt.rn.f32.s32 	%r718, %r257;
	cvt.rn.f32.s32 	%r719, %r258;
	cvt.rn.f32.s32 	%r720, %r259;
	cvt.rn.f32.s32 	%r721, %r260;
	cvt.rn.f32.s32 	%r722, %r261;
	cvt.rn.f32.s32 	%r723, %r262;
	cvt.rn.f32.s32 	%r724, %r263;
	cvt.rn.f32.s32 	%r725, %r264;
	cvt.rn.f32.s32 	%r726, %r265;
	cvt.rn.f32.s32 	%r727, %r266;
	.loc	1 834 40                        // sk04_fa_o_w4a8.py:834:40
	ld.shared.v2.b32 	{%r728, %r729}, [%r29+32768];
	ld.shared.v2.b32 	{%r730, %r731}, [%r29+33024];
	ld.shared.v2.b32 	{%r732, %r733}, [%r29+33280];
	ld.shared.v2.b32 	{%r734, %r735}, [%r29+33536];
	.loc	1 834 15                        // sk04_fa_o_w4a8.py:834:15
	fma.rn.f32 	%r1517, %r731, %r727, %r1517;
	fma.rn.f32 	%r1516, %r730, %r726, %r1516;
	fma.rn.f32 	%r1515, %r729, %r725, %r1515;
	fma.rn.f32 	%r1514, %r728, %r724, %r1514;
	fma.rn.f32 	%r1513, %r729, %r723, %r1513;
	fma.rn.f32 	%r1512, %r728, %r722, %r1512;
	fma.rn.f32 	%r1511, %r735, %r721, %r1511;
	fma.rn.f32 	%r1510, %r734, %r720, %r1510;
	fma.rn.f32 	%r1509, %r735, %r719, %r1509;
	fma.rn.f32 	%r1508, %r734, %r718, %r1508;
	fma.rn.f32 	%r1507, %r733, %r717, %r1507;
	fma.rn.f32 	%r1506, %r732, %r716, %r1506;
	fma.rn.f32 	%r1505, %r733, %r715, %r1505;
	fma.rn.f32 	%r1504, %r732, %r714, %r1504;
	fma.rn.f32 	%r1503, %r731, %r713, %r1503;
	fma.rn.f32 	%r1502, %r730, %r712, %r1502;
	fma.rn.f32 	%r1501, %r731, %r711, %r1501;
	fma.rn.f32 	%r1500, %r730, %r710, %r1500;
	fma.rn.f32 	%r1499, %r729, %r709, %r1499;
	fma.rn.f32 	%r1498, %r728, %r708, %r1498;
	fma.rn.f32 	%r1497, %r729, %r707, %r1497;
	fma.rn.f32 	%r1496, %r728, %r706, %r1496;
	fma.rn.f32 	%r1495, %r735, %r705, %r1495;
	fma.rn.f32 	%r1494, %r734, %r704, %r1494;
	fma.rn.f32 	%r1493, %r735, %r703, %r1493;
	fma.rn.f32 	%r1492, %r734, %r702, %r1492;
	fma.rn.f32 	%r1491, %r733, %r701, %r1491;
	fma.rn.f32 	%r1490, %r732, %r700, %r1490;
	fma.rn.f32 	%r1489, %r733, %r699, %r1489;
	fma.rn.f32 	%r1488, %r732, %r698, %r1488;
	fma.rn.f32 	%r1487, %r731, %r697, %r1487;
	fma.rn.f32 	%r1486, %r730, %r696, %r1486;
	fma.rn.f32 	%r1485, %r731, %r695, %r1485;
	fma.rn.f32 	%r1484, %r730, %r694, %r1484;
	fma.rn.f32 	%r1483, %r729, %r693, %r1483;
	fma.rn.f32 	%r1482, %r728, %r692, %r1482;
	fma.rn.f32 	%r1481, %r729, %r691, %r1481;
	fma.rn.f32 	%r1480, %r728, %r690, %r1480;
	fma.rn.f32 	%r1479, %r735, %r689, %r1479;
	fma.rn.f32 	%r1478, %r734, %r688, %r1478;
	fma.rn.f32 	%r1477, %r735, %r687, %r1477;
	fma.rn.f32 	%r1476, %r734, %r686, %r1476;
	fma.rn.f32 	%r1475, %r733, %r685, %r1475;
	fma.rn.f32 	%r1474, %r732, %r684, %r1474;
	fma.rn.f32 	%r1473, %r733, %r683, %r1473;
	fma.rn.f32 	%r1472, %r732, %r682, %r1472;
	fma.rn.f32 	%r1471, %r731, %r681, %r1471;
	fma.rn.f32 	%r1470, %r730, %r680, %r1470;
	fma.rn.f32 	%r1469, %r731, %r679, %r1469;
	fma.rn.f32 	%r1468, %r730, %r678, %r1468;
	fma.rn.f32 	%r1467, %r729, %r677, %r1467;
	fma.rn.f32 	%r1466, %r728, %r676, %r1466;
	fma.rn.f32 	%r1465, %r729, %r675, %r1465;
	fma.rn.f32 	%r1464, %r728, %r674, %r1464;
	fma.rn.f32 	%r1463, %r735, %r673, %r1463;
	fma.rn.f32 	%r1462, %r734, %r672, %r1462;
	fma.rn.f32 	%r1461, %r735, %r671, %r1461;
	fma.rn.f32 	%r1460, %r734, %r670, %r1460;
	fma.rn.f32 	%r1459, %r733, %r669, %r1459;
	fma.rn.f32 	%r1458, %r732, %r668, %r1458;
	fma.rn.f32 	%r1457, %r733, %r667, %r1457;
	fma.rn.f32 	%r1456, %r732, %r666, %r1456;
	fma.rn.f32 	%r1455, %r731, %r665, %r1455;
	fma.rn.f32 	%r1454, %r730, %r664, %r1454;
	fma.rn.f32 	%r1453, %r731, %r663, %r1453;
	fma.rn.f32 	%r1452, %r730, %r662, %r1452;
	fma.rn.f32 	%r1451, %r729, %r661, %r1451;
	fma.rn.f32 	%r1450, %r728, %r660, %r1450;
	fma.rn.f32 	%r1449, %r729, %r659, %r1449;
	fma.rn.f32 	%r1448, %r728, %r658, %r1448;
	fma.rn.f32 	%r1447, %r735, %r657, %r1447;
	fma.rn.f32 	%r1446, %r734, %r656, %r1446;
	fma.rn.f32 	%r1445, %r735, %r655, %r1445;
	fma.rn.f32 	%r1444, %r734, %r654, %r1444;
	fma.rn.f32 	%r1443, %r733, %r653, %r1443;
	fma.rn.f32 	%r1442, %r732, %r652, %r1442;
	fma.rn.f32 	%r1441, %r733, %r651, %r1441;
	fma.rn.f32 	%r1440, %r732, %r650, %r1440;
	fma.rn.f32 	%r1439, %r731, %r649, %r1439;
	fma.rn.f32 	%r1438, %r730, %r648, %r1438;
	fma.rn.f32 	%r1437, %r731, %r647, %r1437;
	fma.rn.f32 	%r1436, %r730, %r646, %r1436;
	fma.rn.f32 	%r1435, %r729, %r645, %r1435;
	fma.rn.f32 	%r1434, %r728, %r644, %r1434;
	fma.rn.f32 	%r1433, %r729, %r643, %r1433;
	fma.rn.f32 	%r1432, %r728, %r642, %r1432;
	fma.rn.f32 	%r1518, %r730, %r641, %r1518;
	fma.rn.f32 	%r1519, %r731, %r640, %r1519;
	fma.rn.f32 	%r1520, %r732, %r639, %r1520;
	fma.rn.f32 	%r1521, %r733, %r638, %r1521;
	fma.rn.f32 	%r1522, %r732, %r637, %r1522;
	fma.rn.f32 	%r1523, %r733, %r636, %r1523;
	fma.rn.f32 	%r1524, %r734, %r635, %r1524;
	fma.rn.f32 	%r1525, %r735, %r634, %r1525;
	fma.rn.f32 	%r1526, %r734, %r633, %r1526;
	fma.rn.f32 	%r1527, %r735, %r632, %r1527;
	fma.rn.f32 	%r1528, %r728, %r631, %r1528;
	fma.rn.f32 	%r1529, %r729, %r630, %r1529;
	fma.rn.f32 	%r1530, %r728, %r629, %r1530;
	fma.rn.f32 	%r1531, %r729, %r628, %r1531;
	fma.rn.f32 	%r1532, %r730, %r627, %r1532;
	fma.rn.f32 	%r1533, %r731, %r626, %r1533;
	fma.rn.f32 	%r1534, %r730, %r625, %r1534;
	fma.rn.f32 	%r1535, %r731, %r624, %r1535;
	fma.rn.f32 	%r1536, %r732, %r623, %r1536;
	fma.rn.f32 	%r1537, %r733, %r622, %r1537;
	fma.rn.f32 	%r1538, %r732, %r621, %r1538;
	fma.rn.f32 	%r1539, %r733, %r620, %r1539;
	fma.rn.f32 	%r1540, %r734, %r619, %r1540;
	fma.rn.f32 	%r1541, %r735, %r618, %r1541;
	fma.rn.f32 	%r1542, %r734, %r617, %r1542;
	fma.rn.f32 	%r1543, %r735, %r616, %r1543;
	fma.rn.f32 	%r1544, %r728, %r615, %r1544;
	fma.rn.f32 	%r1545, %r729, %r614, %r1545;
	fma.rn.f32 	%r1546, %r728, %r613, %r1546;
	fma.rn.f32 	%r1547, %r729, %r612, %r1547;
	fma.rn.f32 	%r1548, %r730, %r611, %r1548;
	fma.rn.f32 	%r1549, %r731, %r610, %r1549;
	fma.rn.f32 	%r1550, %r730, %r609, %r1550;
	fma.rn.f32 	%r1551, %r731, %r608, %r1551;
	fma.rn.f32 	%r1552, %r732, %r607, %r1552;
	fma.rn.f32 	%r1553, %r733, %r606, %r1553;
	fma.rn.f32 	%r1554, %r732, %r605, %r1554;
	fma.rn.f32 	%r1555, %r733, %r604, %r1555;
	fma.rn.f32 	%r1556, %r734, %r603, %r1556;
	fma.rn.f32 	%r1557, %r735, %r602, %r1557;
	fma.rn.f32 	%r1558, %r734, %r601, %r1558;
	fma.rn.f32 	%r1559, %r735, %r600, %r1559;
	.loc	1 836 18                        // sk04_fa_o_w4a8.py:836:18
	add.s64 	%rd44, %rd251, %rd4;
	add.s64 	%rd46, %rd251, %rd5;
	add.s64 	%rd45, %rd251, %rd6;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	add.s64 	%rd47, %rd251, %rd7;
	add.s32 	%r736, %r1431, 1;
	setp.gt.s32 	%p6, %r736, 1;
	selp.b32 	%r1431, 0, %r736, %p6;
	.loc	1 827 25                        // sk04_fa_o_w4a8.py:827:25
	shl.b32 	%r737, %r1431, 13;
	add.s32 	%r738, %r96, %r737;
	add.s32 	%r323, %r37, %r737;
	selp.b32 	%r324, 8, 0, %p4;
	// begin inline asm
	cp.async.ca.shared.global [ %r323 + 0 ], [ %rd44 + 0 ], 0x8, %r324;
	// end inline asm
	add.s32 	%r325, %r323, 4096;
	// begin inline asm
	cp.async.ca.shared.global [ %r325 + 0 ], [ %rd45 + 0 ], 0x8, %r324;
	// end inline asm
	add.s32 	%r739, %r738, %r15;
	add.s32 	%r326, %r739, 2048;
	// begin inline asm
	cp.async.ca.shared.global [ %r326 + 0 ], [ %rd46 + 0 ], 0x8, %r324;
	// end inline asm
	add.s32 	%r327, %r739, 6144;
	// begin inline asm
	cp.async.ca.shared.global [ %r327 + 0 ], [ %rd47 + 0 ], 0x8, %r324;
	// end inline asm
	cp.async.commit_group;
	.loc	1 831 52                        // sk04_fa_o_w4a8.py:831:52
	add.s32 	%r740, %r18, %r737;
	add.s32 	%r328, %r740, 16384;
	selp.b32 	%r329, 16, 0, %p4;
	// begin inline asm
	cp.async.cg.shared.global [ %r328 + 0 ], [ %rd249 + 0 ], 0x10, %r329;
	// end inline asm
	add.s32 	%r330, %r740, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r330 + 0 ], [ %rd250 + 0 ], 0x10, %r329;
	// end inline asm
	cp.async.commit_group;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	add.s64 	%rd252, %rd252, 1;
	add.s32 	%r1429, %r1429, %r36;
	add.s64 	%rd251, %rd251, %rd9;
	add.s64 	%rd250, %rd250, 64;
	add.s64 	%rd249, %rd249, 64;
	setp.ne.b64 	%p7, %rd12, %rd252;
	@%p7 bra 	$L__BB0_2;
$L__BB0_3:                              // %._crit_edge
	.loc	1 0 22                          // sk04_fa_o_w4a8.py:0:22
	cvt.u32.u64 	%r954, %rd8;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r955, %r13, 7;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r956, %r955, %r31;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r957, %r13, 6;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r958, %r957, %r31;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r959, %r13, 5;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r960, %r959, %r31;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r961, %r13, 4;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r962, %r961, %r31;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r963, %r13, 3;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r964, %r963, %r31;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r965, %r13, 2;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r966, %r965, %r31;
	.loc	1 817 32                        // sk04_fa_o_w4a8.py:817:32
	or.b32 	%r967, %r13, 1;
	.loc	1 817 57                        // sk04_fa_o_w4a8.py:817:57
	rem.s32 	%r968, %r967, %r31;
	.loc	1 816 45                        // sk04_fa_o_w4a8.py:816:45
	or.b32 	%r969, %r1, %r6;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r970, %r969, 120;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r971, %r970, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r972, %r969, 112;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r973, %r972, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r974, %r969, 104;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r975, %r974, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r976, %r969, 96;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r977, %r976, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r978, %r969, 88;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r979, %r978, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r980, %r969, 80;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r981, %r980, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r982, %r969, 72;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r983, %r982, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r984, %r969, 64;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r985, %r984, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r986, %r969, 56;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r987, %r986, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r988, %r969, 48;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r989, %r988, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r990, %r969, 40;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r991, %r990, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r992, %r969, 32;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r993, %r992, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r994, %r1, %r9;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r995, %r994, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r996, %r1, %r8;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r997, %r996, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r998, %r1, %r7;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r999, %r998, %r30;
	rem.s32 	%r1000, %r969, %r30;
	.loc	1 816 45                        // sk04_fa_o_w4a8.py:816:45
	and.b32 	%r1001, %r3, 7;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1002, %r1001, %r1;
	or.b32 	%r1003, %r1002, 120;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1004, %r1003, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1005, %r1002, 112;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1006, %r1005, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1007, %r1002, 104;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1008, %r1007, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1009, %r1002, 96;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1010, %r1009, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1011, %r1002, 88;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1012, %r1011, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1013, %r1002, 80;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1014, %r1013, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1015, %r1002, 72;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1016, %r1015, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1017, %r1002, 64;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1018, %r1017, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1019, %r1002, 56;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1020, %r1019, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1021, %r1002, 48;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1022, %r1021, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1023, %r1002, 40;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1024, %r1023, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1025, %r1002, 32;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1026, %r1025, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1027, %r1002, 24;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1028, %r1027, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1029, %r1002, 16;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1030, %r1029, %r30;
	.loc	1 816 32                        // sk04_fa_o_w4a8.py:816:32
	or.b32 	%r1031, %r1002, 8;
	.loc	1 816 57                        // sk04_fa_o_w4a8.py:816:57
	rem.s32 	%r1032, %r1031, %r30;
	rem.s32 	%r1033, %r1002, %r30;
	.loc	1 825 22                        // sk04_fa_o_w4a8.py:825:22
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 838 38                        // sk04_fa_o_w4a8.py:838:38
	mad.wide.s32 	%rd48, %r1033, 4, %rd17;
	mad.wide.s32 	%rd49, %r1032, 4, %rd17;
	mad.wide.s32 	%rd50, %r1030, 4, %rd17;
	mad.wide.s32 	%rd51, %r1028, 4, %rd17;
	mad.wide.s32 	%rd52, %r1026, 4, %rd17;
	mad.wide.s32 	%rd53, %r1024, 4, %rd17;
	mad.wide.s32 	%rd54, %r1022, 4, %rd17;
	mad.wide.s32 	%rd55, %r1020, 4, %rd17;
	mad.wide.s32 	%rd56, %r1018, 4, %rd17;
	mad.wide.s32 	%rd57, %r1016, 4, %rd17;
	mad.wide.s32 	%rd58, %r1014, 4, %rd17;
	mad.wide.s32 	%rd59, %r1012, 4, %rd17;
	mad.wide.s32 	%rd60, %r1010, 4, %rd17;
	mad.wide.s32 	%rd61, %r1008, 4, %rd17;
	mad.wide.s32 	%rd62, %r1006, 4, %rd17;
	mad.wide.s32 	%rd63, %r1004, 4, %rd17;
	.loc	1 838 24                        // sk04_fa_o_w4a8.py:838:24
	// begin inline asm
	mov.u32 %r741, 0x0;
	ld.global.b32 { %r741 }, [ %rd48 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r742, 0x0;
	ld.global.b32 { %r742 }, [ %rd49 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r743, 0x0;
	ld.global.b32 { %r743 }, [ %rd50 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r744, 0x0;
	ld.global.b32 { %r744 }, [ %rd51 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r745, 0x0;
	ld.global.b32 { %r745 }, [ %rd52 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r746, 0x0;
	ld.global.b32 { %r746 }, [ %rd53 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r747, 0x0;
	ld.global.b32 { %r747 }, [ %rd54 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r748, 0x0;
	ld.global.b32 { %r748 }, [ %rd55 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r749, 0x0;
	ld.global.b32 { %r749 }, [ %rd56 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r750, 0x0;
	ld.global.b32 { %r750 }, [ %rd57 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r751, 0x0;
	ld.global.b32 { %r751 }, [ %rd58 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r752, 0x0;
	ld.global.b32 { %r752 }, [ %rd59 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r753, 0x0;
	ld.global.b32 { %r753 }, [ %rd60 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r754, 0x0;
	ld.global.b32 { %r754 }, [ %rd61 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r755, 0x0;
	ld.global.b32 { %r755 }, [ %rd62 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r756, 0x0;
	ld.global.b32 { %r756 }, [ %rd63 + 0 ];
	// end inline asm
	.loc	1 839 49                        // sk04_fa_o_w4a8.py:839:49
	mul.lo.s32 	%r1034, %r1000, %r34;
	mul.lo.s32 	%r1035, %r999, %r34;
	mul.lo.s32 	%r1036, %r997, %r34;
	mul.lo.s32 	%r1037, %r995, %r34;
	mul.lo.s32 	%r1038, %r993, %r34;
	mul.lo.s32 	%r1039, %r991, %r34;
	mul.lo.s32 	%r1040, %r989, %r34;
	mul.lo.s32 	%r1041, %r987, %r34;
	mul.lo.s32 	%r1042, %r985, %r34;
	mul.lo.s32 	%r1043, %r983, %r34;
	mul.lo.s32 	%r1044, %r981, %r34;
	mul.lo.s32 	%r1045, %r979, %r34;
	mul.lo.s32 	%r1046, %r977, %r34;
	mul.lo.s32 	%r1047, %r975, %r34;
	mul.lo.s32 	%r1048, %r973, %r34;
	mul.lo.s32 	%r1049, %r971, %r34;
	.loc	1 839 31                        // sk04_fa_o_w4a8.py:839:31
	mad.wide.s32 	%rd208, %r1034, 2, %rd16;
	mad.wide.s32 	%rd209, %r1035, 2, %rd16;
	mad.wide.s32 	%rd210, %r1036, 2, %rd16;
	mad.wide.s32 	%rd211, %r1037, 2, %rd16;
	mad.wide.s32 	%rd212, %r1038, 2, %rd16;
	mad.wide.s32 	%rd213, %r1039, 2, %rd16;
	mad.wide.s32 	%rd214, %r1040, 2, %rd16;
	mad.wide.s32 	%rd215, %r1041, 2, %rd16;
	mad.wide.s32 	%rd216, %r1042, 2, %rd16;
	mad.wide.s32 	%rd217, %r1043, 2, %rd16;
	mad.wide.s32 	%rd218, %r1044, 2, %rd16;
	mad.wide.s32 	%rd219, %r1045, 2, %rd16;
	mad.wide.s32 	%rd220, %r1046, 2, %rd16;
	mad.wide.s32 	%rd221, %r1047, 2, %rd16;
	mad.wide.s32 	%rd222, %r1048, 2, %rd16;
	mad.wide.s32 	%rd223, %r1049, 2, %rd16;
	.loc	1 839 82                        // sk04_fa_o_w4a8.py:839:82
	mul.lo.s32 	%r1050, %r954, %r35;
	mul.lo.s32 	%r1051, %r968, %r35;
	mul.lo.s32 	%r1052, %r966, %r35;
	mul.lo.s32 	%r1053, %r964, %r35;
	mul.lo.s32 	%r1054, %r962, %r35;
	mul.lo.s32 	%r1055, %r960, %r35;
	mul.lo.s32 	%r1056, %r958, %r35;
	mul.lo.s32 	%r1057, %r956, %r35;
	.loc	1 839 64                        // sk04_fa_o_w4a8.py:839:64
	mul.wide.s32 	%rd224, %r1050, 2;
	add.s64 	%rd64, %rd208, %rd224;
	mul.wide.s32 	%rd225, %r1051, 2;
	add.s64 	%rd65, %rd208, %rd225;
	mul.wide.s32 	%rd226, %r1052, 2;
	add.s64 	%rd66, %rd208, %rd226;
	mul.wide.s32 	%rd227, %r1053, 2;
	add.s64 	%rd67, %rd208, %rd227;
	mul.wide.s32 	%rd228, %r1054, 2;
	add.s64 	%rd68, %rd208, %rd228;
	mul.wide.s32 	%rd229, %r1055, 2;
	add.s64 	%rd69, %rd208, %rd229;
	mul.wide.s32 	%rd230, %r1056, 2;
	add.s64 	%rd70, %rd208, %rd230;
	mul.wide.s32 	%rd231, %r1057, 2;
	add.s64 	%rd71, %rd208, %rd231;
	add.s64 	%rd72, %rd209, %rd224;
	add.s64 	%rd73, %rd209, %rd225;
	add.s64 	%rd74, %rd209, %rd226;
	add.s64 	%rd75, %rd209, %rd227;
	add.s64 	%rd76, %rd209, %rd228;
	add.s64 	%rd77, %rd209, %rd229;
	add.s64 	%rd78, %rd209, %rd230;
	add.s64 	%rd79, %rd209, %rd231;
	add.s64 	%rd80, %rd210, %rd224;
	add.s64 	%rd81, %rd210, %rd225;
	add.s64 	%rd82, %rd210, %rd226;
	add.s64 	%rd83, %rd210, %rd227;
	add.s64 	%rd84, %rd210, %rd228;
	add.s64 	%rd85, %rd210, %rd229;
	add.s64 	%rd86, %rd210, %rd230;
	add.s64 	%rd87, %rd210, %rd231;
	add.s64 	%rd88, %rd211, %rd224;
	add.s64 	%rd89, %rd211, %rd225;
	add.s64 	%rd90, %rd211, %rd226;
	add.s64 	%rd91, %rd211, %rd227;
	add.s64 	%rd92, %rd211, %rd228;
	add.s64 	%rd93, %rd211, %rd229;
	add.s64 	%rd94, %rd211, %rd230;
	add.s64 	%rd95, %rd211, %rd231;
	add.s64 	%rd96, %rd212, %rd224;
	add.s64 	%rd97, %rd212, %rd225;
	add.s64 	%rd98, %rd212, %rd226;
	add.s64 	%rd99, %rd212, %rd227;
	add.s64 	%rd100, %rd212, %rd228;
	add.s64 	%rd101, %rd212, %rd229;
	add.s64 	%rd102, %rd212, %rd230;
	add.s64 	%rd103, %rd212, %rd231;
	add.s64 	%rd104, %rd213, %rd224;
	add.s64 	%rd105, %rd213, %rd225;
	add.s64 	%rd106, %rd213, %rd226;
	add.s64 	%rd107, %rd213, %rd227;
	add.s64 	%rd108, %rd213, %rd228;
	add.s64 	%rd109, %rd213, %rd229;
	add.s64 	%rd110, %rd213, %rd230;
	add.s64 	%rd111, %rd213, %rd231;
	add.s64 	%rd112, %rd214, %rd224;
	add.s64 	%rd113, %rd214, %rd225;
	add.s64 	%rd114, %rd214, %rd226;
	add.s64 	%rd115, %rd214, %rd227;
	add.s64 	%rd116, %rd214, %rd228;
	add.s64 	%rd117, %rd214, %rd229;
	add.s64 	%rd118, %rd214, %rd230;
	add.s64 	%rd119, %rd214, %rd231;
	add.s64 	%rd120, %rd215, %rd224;
	add.s64 	%rd121, %rd215, %rd225;
	add.s64 	%rd122, %rd215, %rd226;
	add.s64 	%rd123, %rd215, %rd227;
	add.s64 	%rd124, %rd215, %rd228;
	add.s64 	%rd125, %rd215, %rd229;
	add.s64 	%rd126, %rd215, %rd230;
	add.s64 	%rd127, %rd215, %rd231;
	add.s64 	%rd128, %rd216, %rd224;
	add.s64 	%rd129, %rd216, %rd225;
	add.s64 	%rd130, %rd216, %rd226;
	add.s64 	%rd131, %rd216, %rd227;
	add.s64 	%rd132, %rd216, %rd228;
	add.s64 	%rd133, %rd216, %rd229;
	add.s64 	%rd134, %rd216, %rd230;
	add.s64 	%rd135, %rd216, %rd231;
	add.s64 	%rd136, %rd217, %rd224;
	add.s64 	%rd137, %rd217, %rd225;
	add.s64 	%rd138, %rd217, %rd226;
	add.s64 	%rd139, %rd217, %rd227;
	add.s64 	%rd140, %rd217, %rd228;
	add.s64 	%rd141, %rd217, %rd229;
	add.s64 	%rd142, %rd217, %rd230;
	add.s64 	%rd143, %rd217, %rd231;
	add.s64 	%rd144, %rd218, %rd224;
	add.s64 	%rd145, %rd218, %rd225;
	add.s64 	%rd146, %rd218, %rd226;
	add.s64 	%rd147, %rd218, %rd227;
	add.s64 	%rd148, %rd218, %rd228;
	add.s64 	%rd149, %rd218, %rd229;
	add.s64 	%rd150, %rd218, %rd230;
	add.s64 	%rd151, %rd218, %rd231;
	add.s64 	%rd152, %rd219, %rd224;
	add.s64 	%rd153, %rd219, %rd225;
	add.s64 	%rd154, %rd219, %rd226;
	add.s64 	%rd155, %rd219, %rd227;
	add.s64 	%rd156, %rd219, %rd228;
	add.s64 	%rd157, %rd219, %rd229;
	add.s64 	%rd158, %rd219, %rd230;
	add.s64 	%rd159, %rd219, %rd231;
	add.s64 	%rd160, %rd220, %rd224;
	add.s64 	%rd161, %rd220, %rd225;
	add.s64 	%rd162, %rd220, %rd226;
	add.s64 	%rd163, %rd220, %rd227;
	add.s64 	%rd164, %rd220, %rd228;
	add.s64 	%rd165, %rd220, %rd229;
	add.s64 	%rd166, %rd220, %rd230;
	add.s64 	%rd167, %rd220, %rd231;
	add.s64 	%rd168, %rd221, %rd224;
	add.s64 	%rd169, %rd221, %rd225;
	add.s64 	%rd170, %rd221, %rd226;
	add.s64 	%rd171, %rd221, %rd227;
	add.s64 	%rd172, %rd221, %rd228;
	add.s64 	%rd173, %rd221, %rd229;
	add.s64 	%rd174, %rd221, %rd230;
	add.s64 	%rd175, %rd221, %rd231;
	add.s64 	%rd176, %rd222, %rd224;
	add.s64 	%rd177, %rd222, %rd225;
	add.s64 	%rd178, %rd222, %rd226;
	add.s64 	%rd179, %rd222, %rd227;
	add.s64 	%rd180, %rd222, %rd228;
	add.s64 	%rd181, %rd222, %rd229;
	add.s64 	%rd182, %rd222, %rd230;
	add.s64 	%rd183, %rd222, %rd231;
	add.s64 	%rd184, %rd223, %rd224;
	add.s64 	%rd185, %rd223, %rd225;
	add.s64 	%rd186, %rd223, %rd226;
	add.s64 	%rd187, %rd223, %rd227;
	add.s64 	%rd188, %rd223, %rd228;
	add.s64 	%rd189, %rd223, %rd229;
	add.s64 	%rd190, %rd223, %rd230;
	add.s64 	%rd191, %rd223, %rd231;
	.loc	1 839 19                        // sk04_fa_o_w4a8.py:839:19
	// begin inline asm
	mov.u16 %rs161, 0x0;
	ld.global.b16 { %rs161 }, [ %rd64 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs162, 0x0;
	ld.global.b16 { %rs162 }, [ %rd65 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs163, 0x0;
	ld.global.b16 { %rs163 }, [ %rd66 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs164, 0x0;
	ld.global.b16 { %rs164 }, [ %rd67 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs165, 0x0;
	ld.global.b16 { %rs165 }, [ %rd68 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs166, 0x0;
	ld.global.b16 { %rs166 }, [ %rd69 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs167, 0x0;
	ld.global.b16 { %rs167 }, [ %rd70 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs168, 0x0;
	ld.global.b16 { %rs168 }, [ %rd71 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs169, 0x0;
	ld.global.b16 { %rs169 }, [ %rd72 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs170, 0x0;
	ld.global.b16 { %rs170 }, [ %rd73 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs171, 0x0;
	ld.global.b16 { %rs171 }, [ %rd74 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs172, 0x0;
	ld.global.b16 { %rs172 }, [ %rd75 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs173, 0x0;
	ld.global.b16 { %rs173 }, [ %rd76 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs174, 0x0;
	ld.global.b16 { %rs174 }, [ %rd77 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs175, 0x0;
	ld.global.b16 { %rs175 }, [ %rd78 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs176, 0x0;
	ld.global.b16 { %rs176 }, [ %rd79 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs177, 0x0;
	ld.global.b16 { %rs177 }, [ %rd80 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs178, 0x0;
	ld.global.b16 { %rs178 }, [ %rd81 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs179, 0x0;
	ld.global.b16 { %rs179 }, [ %rd82 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs180, 0x0;
	ld.global.b16 { %rs180 }, [ %rd83 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs181, 0x0;
	ld.global.b16 { %rs181 }, [ %rd84 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs182, 0x0;
	ld.global.b16 { %rs182 }, [ %rd85 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs183, 0x0;
	ld.global.b16 { %rs183 }, [ %rd86 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs184, 0x0;
	ld.global.b16 { %rs184 }, [ %rd87 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs185, 0x0;
	ld.global.b16 { %rs185 }, [ %rd88 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs186, 0x0;
	ld.global.b16 { %rs186 }, [ %rd89 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs187, 0x0;
	ld.global.b16 { %rs187 }, [ %rd90 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs188, 0x0;
	ld.global.b16 { %rs188 }, [ %rd91 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs189, 0x0;
	ld.global.b16 { %rs189 }, [ %rd92 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs190, 0x0;
	ld.global.b16 { %rs190 }, [ %rd93 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs191, 0x0;
	ld.global.b16 { %rs191 }, [ %rd94 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs192, 0x0;
	ld.global.b16 { %rs192 }, [ %rd95 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs193, 0x0;
	ld.global.b16 { %rs193 }, [ %rd96 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs194, 0x0;
	ld.global.b16 { %rs194 }, [ %rd97 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs195, 0x0;
	ld.global.b16 { %rs195 }, [ %rd98 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs196, 0x0;
	ld.global.b16 { %rs196 }, [ %rd99 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs197, 0x0;
	ld.global.b16 { %rs197 }, [ %rd100 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs198, 0x0;
	ld.global.b16 { %rs198 }, [ %rd101 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs199, 0x0;
	ld.global.b16 { %rs199 }, [ %rd102 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs200, 0x0;
	ld.global.b16 { %rs200 }, [ %rd103 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs201, 0x0;
	ld.global.b16 { %rs201 }, [ %rd104 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs202, 0x0;
	ld.global.b16 { %rs202 }, [ %rd105 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs203, 0x0;
	ld.global.b16 { %rs203 }, [ %rd106 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs204, 0x0;
	ld.global.b16 { %rs204 }, [ %rd107 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs205, 0x0;
	ld.global.b16 { %rs205 }, [ %rd108 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs206, 0x0;
	ld.global.b16 { %rs206 }, [ %rd109 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs207, 0x0;
	ld.global.b16 { %rs207 }, [ %rd110 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs208, 0x0;
	ld.global.b16 { %rs208 }, [ %rd111 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs209, 0x0;
	ld.global.b16 { %rs209 }, [ %rd112 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs210, 0x0;
	ld.global.b16 { %rs210 }, [ %rd113 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs211, 0x0;
	ld.global.b16 { %rs211 }, [ %rd114 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs212, 0x0;
	ld.global.b16 { %rs212 }, [ %rd115 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs213, 0x0;
	ld.global.b16 { %rs213 }, [ %rd116 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs214, 0x0;
	ld.global.b16 { %rs214 }, [ %rd117 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs215, 0x0;
	ld.global.b16 { %rs215 }, [ %rd118 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs216, 0x0;
	ld.global.b16 { %rs216 }, [ %rd119 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs217, 0x0;
	ld.global.b16 { %rs217 }, [ %rd120 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs218, 0x0;
	ld.global.b16 { %rs218 }, [ %rd121 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs219, 0x0;
	ld.global.b16 { %rs219 }, [ %rd122 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs220, 0x0;
	ld.global.b16 { %rs220 }, [ %rd123 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs221, 0x0;
	ld.global.b16 { %rs221 }, [ %rd124 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs222, 0x0;
	ld.global.b16 { %rs222 }, [ %rd125 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs223, 0x0;
	ld.global.b16 { %rs223 }, [ %rd126 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs224, 0x0;
	ld.global.b16 { %rs224 }, [ %rd127 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs225, 0x0;
	ld.global.b16 { %rs225 }, [ %rd128 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs226, 0x0;
	ld.global.b16 { %rs226 }, [ %rd129 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs227, 0x0;
	ld.global.b16 { %rs227 }, [ %rd130 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs228, 0x0;
	ld.global.b16 { %rs228 }, [ %rd131 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs229, 0x0;
	ld.global.b16 { %rs229 }, [ %rd132 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs230, 0x0;
	ld.global.b16 { %rs230 }, [ %rd133 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs231, 0x0;
	ld.global.b16 { %rs231 }, [ %rd134 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs232, 0x0;
	ld.global.b16 { %rs232 }, [ %rd135 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs233, 0x0;
	ld.global.b16 { %rs233 }, [ %rd136 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs234, 0x0;
	ld.global.b16 { %rs234 }, [ %rd137 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs235, 0x0;
	ld.global.b16 { %rs235 }, [ %rd138 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs236, 0x0;
	ld.global.b16 { %rs236 }, [ %rd139 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs237, 0x0;
	ld.global.b16 { %rs237 }, [ %rd140 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs238, 0x0;
	ld.global.b16 { %rs238 }, [ %rd141 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs239, 0x0;
	ld.global.b16 { %rs239 }, [ %rd142 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs240, 0x0;
	ld.global.b16 { %rs240 }, [ %rd143 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs241, 0x0;
	ld.global.b16 { %rs241 }, [ %rd144 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs242, 0x0;
	ld.global.b16 { %rs242 }, [ %rd145 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs243, 0x0;
	ld.global.b16 { %rs243 }, [ %rd146 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs244, 0x0;
	ld.global.b16 { %rs244 }, [ %rd147 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs245, 0x0;
	ld.global.b16 { %rs245 }, [ %rd148 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs246, 0x0;
	ld.global.b16 { %rs246 }, [ %rd149 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs247, 0x0;
	ld.global.b16 { %rs247 }, [ %rd150 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs248, 0x0;
	ld.global.b16 { %rs248 }, [ %rd151 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs249, 0x0;
	ld.global.b16 { %rs249 }, [ %rd152 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs250, 0x0;
	ld.global.b16 { %rs250 }, [ %rd153 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs251, 0x0;
	ld.global.b16 { %rs251 }, [ %rd154 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs252, 0x0;
	ld.global.b16 { %rs252 }, [ %rd155 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs253, 0x0;
	ld.global.b16 { %rs253 }, [ %rd156 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs254, 0x0;
	ld.global.b16 { %rs254 }, [ %rd157 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs255, 0x0;
	ld.global.b16 { %rs255 }, [ %rd158 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs256, 0x0;
	ld.global.b16 { %rs256 }, [ %rd159 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs257, 0x0;
	ld.global.b16 { %rs257 }, [ %rd160 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs258, 0x0;
	ld.global.b16 { %rs258 }, [ %rd161 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs259, 0x0;
	ld.global.b16 { %rs259 }, [ %rd162 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs260, 0x0;
	ld.global.b16 { %rs260 }, [ %rd163 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs261, 0x0;
	ld.global.b16 { %rs261 }, [ %rd164 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs262, 0x0;
	ld.global.b16 { %rs262 }, [ %rd165 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs263, 0x0;
	ld.global.b16 { %rs263 }, [ %rd166 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs264, 0x0;
	ld.global.b16 { %rs264 }, [ %rd167 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs265, 0x0;
	ld.global.b16 { %rs265 }, [ %rd168 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs266, 0x0;
	ld.global.b16 { %rs266 }, [ %rd169 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs267, 0x0;
	ld.global.b16 { %rs267 }, [ %rd170 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs268, 0x0;
	ld.global.b16 { %rs268 }, [ %rd171 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs269, 0x0;
	ld.global.b16 { %rs269 }, [ %rd172 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs270, 0x0;
	ld.global.b16 { %rs270 }, [ %rd173 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs271, 0x0;
	ld.global.b16 { %rs271 }, [ %rd174 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs272, 0x0;
	ld.global.b16 { %rs272 }, [ %rd175 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs273, 0x0;
	ld.global.b16 { %rs273 }, [ %rd176 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs274, 0x0;
	ld.global.b16 { %rs274 }, [ %rd177 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs275, 0x0;
	ld.global.b16 { %rs275 }, [ %rd178 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs276, 0x0;
	ld.global.b16 { %rs276 }, [ %rd179 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs277, 0x0;
	ld.global.b16 { %rs277 }, [ %rd180 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs278, 0x0;
	ld.global.b16 { %rs278 }, [ %rd181 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs279, 0x0;
	ld.global.b16 { %rs279 }, [ %rd182 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs280, 0x0;
	ld.global.b16 { %rs280 }, [ %rd183 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs281, 0x0;
	ld.global.b16 { %rs281 }, [ %rd184 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs282, 0x0;
	ld.global.b16 { %rs282 }, [ %rd185 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs283, 0x0;
	ld.global.b16 { %rs283 }, [ %rd186 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs284, 0x0;
	ld.global.b16 { %rs284 }, [ %rd187 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs285, 0x0;
	ld.global.b16 { %rs285 }, [ %rd188 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs286, 0x0;
	ld.global.b16 { %rs286 }, [ %rd189 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs287, 0x0;
	ld.global.b16 { %rs287 }, [ %rd190 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs288, 0x0;
	ld.global.b16 { %rs288 }, [ %rd191 + 0 ];
	// end inline asm
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	shl.b32 	%r1058, %r12, 4;
	shr.u32 	%r1059, %r5, 1;
	xor.b32 	%r1060, %r1058, %r1059;
	add.s32 	%r757, %r96, %r1060;
	mov.b32 	%r758, {%rs161, %rs162};
	mov.b32 	%r759, {%rs163, %rs164};
	mov.b32 	%r760, {%rs165, %rs166};
	mov.b32 	%r761, {%rs167, %rs168};
	// begin inline asm
	st.shared.v4.b32 [ %r757 + 0 ], { %r758, %r759, %r760, %r761 };
	// end inline asm
	bar.sync 	0;
	and.b32 	%r1061, %r2, 7;
	shl.b32 	%r1062, %r1061, 9;
	shl.b32 	%r1063, %r11, 4;
	xor.b32 	%r1064, %r1063, %r1059;
	add.s32 	%r1065, %r96, %r1062;
	add.s32 	%r1066, %r1065, %r1064;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1067, %r1068, %r1069, %r1070}, [%r1066];
	bar.sync 	0;
	mov.b32 	%r762, {%rs169, %rs170};
	mov.b32 	%r763, {%rs171, %rs172};
	mov.b32 	%r764, {%rs173, %rs174};
	mov.b32 	%r765, {%rs175, %rs176};
	// begin inline asm
	st.shared.v4.b32 [ %r757 + 0 ], { %r762, %r763, %r764, %r765 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1071, %r1072, %r1073, %r1074}, [%r1066];
	bar.sync 	0;
	mov.b32 	%r766, {%rs177, %rs178};
	mov.b32 	%r767, {%rs179, %rs180};
	mov.b32 	%r768, {%rs181, %rs182};
	mov.b32 	%r769, {%rs183, %rs184};
	// begin inline asm
	st.shared.v4.b32 [ %r757 + 0 ], { %r766, %r767, %r768, %r769 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1075, %r1076, %r1077, %r1078}, [%r1066];
	bar.sync 	0;
	mov.b32 	%r770, {%rs185, %rs186};
	mov.b32 	%r771, {%rs187, %rs188};
	mov.b32 	%r772, {%rs189, %rs190};
	mov.b32 	%r773, {%rs191, %rs192};
	// begin inline asm
	st.shared.v4.b32 [ %r757 + 0 ], { %r770, %r771, %r772, %r773 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1079, %r1080, %r1081, %r1082}, [%r1066];
	bar.sync 	0;
	mov.b32 	%r774, {%rs193, %rs194};
	mov.b32 	%r775, {%rs195, %rs196};
	mov.b32 	%r776, {%rs197, %rs198};
	mov.b32 	%r777, {%rs199, %rs200};
	// begin inline asm
	st.shared.v4.b32 [ %r757 + 0 ], { %r774, %r775, %r776, %r777 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1083, %r1084, %r1085, %r1086}, [%r1066];
	bar.sync 	0;
	mov.b32 	%r778, {%rs201, %rs202};
	mov.b32 	%r779, {%rs203, %rs204};
	mov.b32 	%r780, {%rs205, %rs206};
	mov.b32 	%r781, {%rs207, %rs208};
	// begin inline asm
	st.shared.v4.b32 [ %r757 + 0 ], { %r778, %r779, %r780, %r781 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1087, %r1088, %r1089, %r1090}, [%r1066];
	bar.sync 	0;
	mov.b32 	%r782, {%rs209, %rs210};
	mov.b32 	%r783, {%rs211, %rs212};
	mov.b32 	%r784, {%rs213, %rs214};
	mov.b32 	%r785, {%rs215, %rs216};
	// begin inline asm
	st.shared.v4.b32 [ %r757 + 0 ], { %r782, %r783, %r784, %r785 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1091, %r1092, %r1093, %r1094}, [%r1066];
	bar.sync 	0;
	mov.b32 	%r786, {%rs217, %rs218};
	mov.b32 	%r787, {%rs219, %rs220};
	mov.b32 	%r788, {%rs221, %rs222};
	mov.b32 	%r789, {%rs223, %rs224};
	// begin inline asm
	st.shared.v4.b32 [ %r757 + 0 ], { %r786, %r787, %r788, %r789 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1095, %r1096, %r1097, %r1098}, [%r1066];
	bar.sync 	0;
	mov.b32 	%r790, {%rs225, %rs226};
	mov.b32 	%r791, {%rs227, %rs228};
	mov.b32 	%r792, {%rs229, %rs230};
	mov.b32 	%r793, {%rs231, %rs232};
	// begin inline asm
	st.shared.v4.b32 [ %r757 + 0 ], { %r790, %r791, %r792, %r793 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1099, %r1100, %r1101, %r1102}, [%r1066];
	bar.sync 	0;
	mov.b32 	%r794, {%rs233, %rs234};
	mov.b32 	%r795, {%rs235, %rs236};
	mov.b32 	%r796, {%rs237, %rs238};
	mov.b32 	%r797, {%rs239, %rs240};
	// begin inline asm
	st.shared.v4.b32 [ %r757 + 0 ], { %r794, %r795, %r796, %r797 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1103, %r1104, %r1105, %r1106}, [%r1066];
	bar.sync 	0;
	mov.b32 	%r798, {%rs241, %rs242};
	mov.b32 	%r799, {%rs243, %rs244};
	mov.b32 	%r800, {%rs245, %rs246};
	mov.b32 	%r801, {%rs247, %rs248};
	// begin inline asm
	st.shared.v4.b32 [ %r757 + 0 ], { %r798, %r799, %r800, %r801 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1107, %r1108, %r1109, %r1110}, [%r1066];
	bar.sync 	0;
	mov.b32 	%r802, {%rs249, %rs250};
	mov.b32 	%r803, {%rs251, %rs252};
	mov.b32 	%r804, {%rs253, %rs254};
	mov.b32 	%r805, {%rs255, %rs256};
	// begin inline asm
	st.shared.v4.b32 [ %r757 + 0 ], { %r802, %r803, %r804, %r805 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1111, %r1112, %r1113, %r1114}, [%r1066];
	bar.sync 	0;
	mov.b32 	%r806, {%rs257, %rs258};
	mov.b32 	%r807, {%rs259, %rs260};
	mov.b32 	%r808, {%rs261, %rs262};
	mov.b32 	%r809, {%rs263, %rs264};
	// begin inline asm
	st.shared.v4.b32 [ %r757 + 0 ], { %r806, %r807, %r808, %r809 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1115, %r1116, %r1117, %r1118}, [%r1066];
	bar.sync 	0;
	mov.b32 	%r810, {%rs265, %rs266};
	mov.b32 	%r811, {%rs267, %rs268};
	mov.b32 	%r812, {%rs269, %rs270};
	mov.b32 	%r813, {%rs271, %rs272};
	// begin inline asm
	st.shared.v4.b32 [ %r757 + 0 ], { %r810, %r811, %r812, %r813 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1119, %r1120, %r1121, %r1122}, [%r1066];
	bar.sync 	0;
	mov.b32 	%r814, {%rs273, %rs274};
	mov.b32 	%r815, {%rs275, %rs276};
	mov.b32 	%r816, {%rs277, %rs278};
	mov.b32 	%r817, {%rs279, %rs280};
	// begin inline asm
	st.shared.v4.b32 [ %r757 + 0 ], { %r814, %r815, %r816, %r817 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1123, %r1124, %r1125, %r1126}, [%r1066];
	bar.sync 	0;
	mov.b32 	%r818, {%rs281, %rs282};
	mov.b32 	%r819, {%rs283, %rs284};
	mov.b32 	%r820, {%rs285, %rs286};
	mov.b32 	%r821, {%rs287, %rs288};
	// begin inline asm
	st.shared.v4.b32 [ %r757 + 0 ], { %r818, %r819, %r820, %r821 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1127, %r1128, %r1129, %r1130}, [%r1066];
	.loc	1 846 31                        // sk04_fa_o_w4a8.py:846:31
	setp.lt.s32 	%p24, %r969, %r30;
	setp.lt.s32 	%p25, %r998, %r30;
	setp.lt.s32 	%p26, %r996, %r30;
	setp.lt.s32 	%p27, %r994, %r30;
	setp.lt.s32 	%p28, %r992, %r30;
	setp.lt.s32 	%p29, %r990, %r30;
	setp.lt.s32 	%p30, %r988, %r30;
	setp.lt.s32 	%p31, %r986, %r30;
	setp.lt.s32 	%p32, %r984, %r30;
	setp.lt.s32 	%p33, %r982, %r30;
	setp.lt.s32 	%p34, %r980, %r30;
	setp.lt.s32 	%p35, %r978, %r30;
	setp.lt.s32 	%p36, %r976, %r30;
	setp.lt.s32 	%p37, %r974, %r30;
	setp.lt.s32 	%p38, %r972, %r30;
	setp.lt.s32 	%p39, %r970, %r30;
	.loc	1 846 54                        // sk04_fa_o_w4a8.py:846:54
	setp.lt.s32 	%p40, %r13, %r31;
	.loc	1 846 37                        // sk04_fa_o_w4a8.py:846:37
	and.pred 	%p8, %p24, %p40;
	and.pred 	%p9, %p25, %p40;
	and.pred 	%p10, %p26, %p40;
	and.pred 	%p11, %p27, %p40;
	and.pred 	%p12, %p28, %p40;
	and.pred 	%p13, %p29, %p40;
	and.pred 	%p14, %p30, %p40;
	and.pred 	%p15, %p31, %p40;
	and.pred 	%p16, %p32, %p40;
	and.pred 	%p17, %p33, %p40;
	and.pred 	%p18, %p34, %p40;
	and.pred 	%p19, %p35, %p40;
	and.pred 	%p20, %p36, %p40;
	and.pred 	%p21, %p37, %p40;
	and.pred 	%p22, %p38, %p40;
	and.pred 	%p23, %p39, %p40;
	.loc	1 844 35                        // sk04_fa_o_w4a8.py:844:35
	mul.lo.s32 	%r1131, %r969, %r33;
	mul.lo.s32 	%r1132, %r998, %r33;
	mul.lo.s32 	%r1133, %r996, %r33;
	mul.lo.s32 	%r1134, %r994, %r33;
	mul.lo.s32 	%r1135, %r33, %r992;
	mul.lo.s32 	%r1136, %r33, %r990;
	mul.lo.s32 	%r1137, %r33, %r988;
	mul.lo.s32 	%r1138, %r33, %r986;
	mul.lo.s32 	%r1139, %r33, %r984;
	mul.lo.s32 	%r1140, %r33, %r982;
	mul.lo.s32 	%r1141, %r33, %r980;
	mul.lo.s32 	%r1142, %r33, %r978;
	mul.lo.s32 	%r1143, %r33, %r976;
	mul.lo.s32 	%r1144, %r33, %r974;
	mul.lo.s32 	%r1145, %r33, %r972;
	mul.lo.s32 	%r1146, %r33, %r970;
	.loc	1 844 18                        // sk04_fa_o_w4a8.py:844:18
	mad.wide.s32 	%rd232, %r1131, 2, %rd15;
	mad.wide.s32 	%rd233, %r1132, 2, %rd15;
	mad.wide.s32 	%rd234, %r1133, 2, %rd15;
	mad.wide.s32 	%rd235, %r1134, 2, %rd15;
	mad.wide.s32 	%rd236, %r1135, 2, %rd15;
	mad.wide.s32 	%rd237, %r1136, 2, %rd15;
	mad.wide.s32 	%rd238, %r1137, 2, %rd15;
	mad.wide.s32 	%rd239, %r1138, 2, %rd15;
	mad.wide.s32 	%rd240, %r1139, 2, %rd15;
	mad.wide.s32 	%rd241, %r1140, 2, %rd15;
	mad.wide.s32 	%rd242, %r1141, 2, %rd15;
	mad.wide.s32 	%rd243, %r1142, 2, %rd15;
	mad.wide.s32 	%rd244, %r1143, 2, %rd15;
	mad.wide.s32 	%rd245, %r1144, 2, %rd15;
	mad.wide.s32 	%rd246, %r1145, 2, %rd15;
	mad.wide.s32 	%rd247, %r1146, 2, %rd15;
	.loc	1 844 50                        // sk04_fa_o_w4a8.py:844:50
	mul.wide.s32 	%rd248, %r13, 2;
	add.s64 	%rd192, %rd232, %rd248;
	add.s64 	%rd193, %rd233, %rd248;
	add.s64 	%rd194, %rd234, %rd248;
	add.s64 	%rd195, %rd235, %rd248;
	add.s64 	%rd196, %rd236, %rd248;
	add.s64 	%rd197, %rd237, %rd248;
	add.s64 	%rd198, %rd238, %rd248;
	add.s64 	%rd199, %rd239, %rd248;
	add.s64 	%rd200, %rd240, %rd248;
	add.s64 	%rd201, %rd241, %rd248;
	add.s64 	%rd202, %rd242, %rd248;
	add.s64 	%rd203, %rd243, %rd248;
	add.s64 	%rd204, %rd244, %rd248;
	add.s64 	%rd205, %rd245, %rd248;
	add.s64 	%rd206, %rd246, %rd248;
	add.s64 	%rd207, %rd247, %rd248;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs289, %rs290}, %r1067;
	cvt.f32.bf16 	%r1147, %rs290;
	cvt.f32.bf16 	%r1148, %rs289;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1149, %r1432, %r741, %r1148;
	fma.rn.f32 	%r1150, %r1433, %r741, %r1147;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r823, %r1150, %r1149;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs291, %rs292}, %r1071;
	cvt.f32.bf16 	%r1151, %rs292;
	cvt.f32.bf16 	%r1152, %rs291;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1153, %r1434, %r742, %r1152;
	fma.rn.f32 	%r1154, %r1435, %r742, %r1151;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r824, %r1154, %r1153;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs293, %rs294}, %r1068;
	cvt.f32.bf16 	%r1155, %rs294;
	cvt.f32.bf16 	%r1156, %rs293;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1157, %r1436, %r741, %r1156;
	fma.rn.f32 	%r1158, %r1437, %r741, %r1155;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r828, %r1158, %r1157;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs295, %rs296}, %r1072;
	cvt.f32.bf16 	%r1159, %rs296;
	cvt.f32.bf16 	%r1160, %rs295;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1161, %r1438, %r742, %r1160;
	fma.rn.f32 	%r1162, %r1439, %r742, %r1159;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r829, %r1162, %r1161;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs297, %rs298}, %r1069;
	cvt.f32.bf16 	%r1163, %rs298;
	cvt.f32.bf16 	%r1164, %rs297;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1165, %r1440, %r741, %r1164;
	fma.rn.f32 	%r1166, %r1441, %r741, %r1163;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r833, %r1166, %r1165;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs299, %rs300}, %r1073;
	cvt.f32.bf16 	%r1167, %rs300;
	cvt.f32.bf16 	%r1168, %rs299;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1169, %r1442, %r742, %r1168;
	fma.rn.f32 	%r1170, %r1443, %r742, %r1167;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r834, %r1170, %r1169;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs301, %rs302}, %r1070;
	cvt.f32.bf16 	%r1171, %rs302;
	cvt.f32.bf16 	%r1172, %rs301;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1173, %r1444, %r741, %r1172;
	fma.rn.f32 	%r1174, %r1445, %r741, %r1171;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r838, %r1174, %r1173;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs303, %rs304}, %r1074;
	cvt.f32.bf16 	%r1175, %rs304;
	cvt.f32.bf16 	%r1176, %rs303;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1177, %r1446, %r742, %r1176;
	fma.rn.f32 	%r1178, %r1447, %r742, %r1175;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r839, %r1178, %r1177;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs305, %rs306}, %r1075;
	cvt.f32.bf16 	%r1179, %rs306;
	cvt.f32.bf16 	%r1180, %rs305;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1181, %r1448, %r743, %r1180;
	fma.rn.f32 	%r1182, %r1449, %r743, %r1179;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r825, %r1182, %r1181;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs307, %rs308}, %r1079;
	cvt.f32.bf16 	%r1183, %rs308;
	cvt.f32.bf16 	%r1184, %rs307;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1185, %r1450, %r744, %r1184;
	fma.rn.f32 	%r1186, %r1451, %r744, %r1183;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r826, %r1186, %r1185;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs309, %rs310}, %r1076;
	cvt.f32.bf16 	%r1187, %rs310;
	cvt.f32.bf16 	%r1188, %rs309;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1189, %r1452, %r743, %r1188;
	fma.rn.f32 	%r1190, %r1453, %r743, %r1187;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r830, %r1190, %r1189;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs311, %rs312}, %r1080;
	cvt.f32.bf16 	%r1191, %rs312;
	cvt.f32.bf16 	%r1192, %rs311;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1193, %r1454, %r744, %r1192;
	fma.rn.f32 	%r1194, %r1455, %r744, %r1191;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r831, %r1194, %r1193;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs313, %rs314}, %r1077;
	cvt.f32.bf16 	%r1195, %rs314;
	cvt.f32.bf16 	%r1196, %rs313;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1197, %r1456, %r743, %r1196;
	fma.rn.f32 	%r1198, %r1457, %r743, %r1195;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r835, %r1198, %r1197;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs315, %rs316}, %r1081;
	cvt.f32.bf16 	%r1199, %rs316;
	cvt.f32.bf16 	%r1200, %rs315;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1201, %r1458, %r744, %r1200;
	fma.rn.f32 	%r1202, %r1459, %r744, %r1199;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r836, %r1202, %r1201;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs317, %rs318}, %r1078;
	cvt.f32.bf16 	%r1203, %rs318;
	cvt.f32.bf16 	%r1204, %rs317;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1205, %r1460, %r743, %r1204;
	fma.rn.f32 	%r1206, %r1461, %r743, %r1203;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r840, %r1206, %r1205;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs319, %rs320}, %r1082;
	cvt.f32.bf16 	%r1207, %rs320;
	cvt.f32.bf16 	%r1208, %rs319;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1209, %r1462, %r744, %r1208;
	fma.rn.f32 	%r1210, %r1463, %r744, %r1207;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r841, %r1210, %r1209;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs321, %rs322}, %r1083;
	cvt.f32.bf16 	%r1211, %rs322;
	cvt.f32.bf16 	%r1212, %rs321;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1213, %r1464, %r745, %r1212;
	fma.rn.f32 	%r1214, %r1465, %r745, %r1211;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r842, %r1214, %r1213;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs323, %rs324}, %r1087;
	cvt.f32.bf16 	%r1215, %rs324;
	cvt.f32.bf16 	%r1216, %rs323;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1217, %r1466, %r746, %r1216;
	fma.rn.f32 	%r1218, %r1467, %r746, %r1215;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r843, %r1218, %r1217;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs325, %rs326}, %r1084;
	cvt.f32.bf16 	%r1219, %rs326;
	cvt.f32.bf16 	%r1220, %rs325;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1221, %r1468, %r745, %r1220;
	fma.rn.f32 	%r1222, %r1469, %r745, %r1219;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r846, %r1222, %r1221;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs327, %rs328}, %r1088;
	cvt.f32.bf16 	%r1223, %rs328;
	cvt.f32.bf16 	%r1224, %rs327;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1225, %r1470, %r746, %r1224;
	fma.rn.f32 	%r1226, %r1471, %r746, %r1223;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r847, %r1226, %r1225;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs329, %rs330}, %r1085;
	cvt.f32.bf16 	%r1227, %rs330;
	cvt.f32.bf16 	%r1228, %rs329;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1229, %r1472, %r745, %r1228;
	fma.rn.f32 	%r1230, %r1473, %r745, %r1227;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r850, %r1230, %r1229;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs331, %rs332}, %r1089;
	cvt.f32.bf16 	%r1231, %rs332;
	cvt.f32.bf16 	%r1232, %rs331;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1233, %r1474, %r746, %r1232;
	fma.rn.f32 	%r1234, %r1475, %r746, %r1231;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r851, %r1234, %r1233;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs333, %rs334}, %r1086;
	cvt.f32.bf16 	%r1235, %rs334;
	cvt.f32.bf16 	%r1236, %rs333;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1237, %r1476, %r745, %r1236;
	fma.rn.f32 	%r1238, %r1477, %r745, %r1235;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r854, %r1238, %r1237;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs335, %rs336}, %r1090;
	cvt.f32.bf16 	%r1239, %rs336;
	cvt.f32.bf16 	%r1240, %rs335;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1241, %r1478, %r746, %r1240;
	fma.rn.f32 	%r1242, %r1479, %r746, %r1239;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r855, %r1242, %r1241;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs337, %rs338}, %r1091;
	cvt.f32.bf16 	%r1243, %rs338;
	cvt.f32.bf16 	%r1244, %rs337;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1245, %r1480, %r747, %r1244;
	fma.rn.f32 	%r1246, %r1481, %r747, %r1243;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r844, %r1246, %r1245;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs339, %rs340}, %r1095;
	cvt.f32.bf16 	%r1247, %rs340;
	cvt.f32.bf16 	%r1248, %rs339;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1249, %r1482, %r748, %r1248;
	fma.rn.f32 	%r1250, %r1483, %r748, %r1247;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r845, %r1250, %r1249;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs341, %rs342}, %r1092;
	cvt.f32.bf16 	%r1251, %rs342;
	cvt.f32.bf16 	%r1252, %rs341;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1253, %r1484, %r747, %r1252;
	fma.rn.f32 	%r1254, %r1485, %r747, %r1251;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r848, %r1254, %r1253;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs343, %rs344}, %r1096;
	cvt.f32.bf16 	%r1255, %rs344;
	cvt.f32.bf16 	%r1256, %rs343;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1257, %r1486, %r748, %r1256;
	fma.rn.f32 	%r1258, %r1487, %r748, %r1255;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r849, %r1258, %r1257;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs345, %rs346}, %r1093;
	cvt.f32.bf16 	%r1259, %rs346;
	cvt.f32.bf16 	%r1260, %rs345;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1261, %r1488, %r747, %r1260;
	fma.rn.f32 	%r1262, %r1489, %r747, %r1259;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r852, %r1262, %r1261;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs347, %rs348}, %r1097;
	cvt.f32.bf16 	%r1263, %rs348;
	cvt.f32.bf16 	%r1264, %rs347;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1265, %r1490, %r748, %r1264;
	fma.rn.f32 	%r1266, %r1491, %r748, %r1263;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r853, %r1266, %r1265;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs349, %rs350}, %r1094;
	cvt.f32.bf16 	%r1267, %rs350;
	cvt.f32.bf16 	%r1268, %rs349;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1269, %r1492, %r747, %r1268;
	fma.rn.f32 	%r1270, %r1493, %r747, %r1267;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r856, %r1270, %r1269;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs351, %rs352}, %r1098;
	cvt.f32.bf16 	%r1271, %rs352;
	cvt.f32.bf16 	%r1272, %rs351;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1273, %r1494, %r748, %r1272;
	fma.rn.f32 	%r1274, %r1495, %r748, %r1271;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r857, %r1274, %r1273;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs353, %rs354}, %r1099;
	cvt.f32.bf16 	%r1275, %rs354;
	cvt.f32.bf16 	%r1276, %rs353;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1277, %r1496, %r749, %r1276;
	fma.rn.f32 	%r1278, %r1497, %r749, %r1275;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r858, %r1278, %r1277;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs355, %rs356}, %r1103;
	cvt.f32.bf16 	%r1279, %rs356;
	cvt.f32.bf16 	%r1280, %rs355;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1281, %r1498, %r750, %r1280;
	fma.rn.f32 	%r1282, %r1499, %r750, %r1279;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r859, %r1282, %r1281;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs357, %rs358}, %r1100;
	cvt.f32.bf16 	%r1283, %rs358;
	cvt.f32.bf16 	%r1284, %rs357;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1285, %r1500, %r749, %r1284;
	fma.rn.f32 	%r1286, %r1501, %r749, %r1283;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r862, %r1286, %r1285;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs359, %rs360}, %r1104;
	cvt.f32.bf16 	%r1287, %rs360;
	cvt.f32.bf16 	%r1288, %rs359;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1289, %r1502, %r750, %r1288;
	fma.rn.f32 	%r1290, %r1503, %r750, %r1287;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r863, %r1290, %r1289;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs361, %rs362}, %r1101;
	cvt.f32.bf16 	%r1291, %rs362;
	cvt.f32.bf16 	%r1292, %rs361;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1293, %r1504, %r749, %r1292;
	fma.rn.f32 	%r1294, %r1505, %r749, %r1291;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r866, %r1294, %r1293;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs363, %rs364}, %r1105;
	cvt.f32.bf16 	%r1295, %rs364;
	cvt.f32.bf16 	%r1296, %rs363;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1297, %r1506, %r750, %r1296;
	fma.rn.f32 	%r1298, %r1507, %r750, %r1295;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r867, %r1298, %r1297;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs365, %rs366}, %r1102;
	cvt.f32.bf16 	%r1299, %rs366;
	cvt.f32.bf16 	%r1300, %rs365;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1301, %r1508, %r749, %r1300;
	fma.rn.f32 	%r1302, %r1509, %r749, %r1299;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r870, %r1302, %r1301;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs367, %rs368}, %r1106;
	cvt.f32.bf16 	%r1303, %rs368;
	cvt.f32.bf16 	%r1304, %rs367;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1305, %r1510, %r750, %r1304;
	fma.rn.f32 	%r1306, %r1511, %r750, %r1303;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r871, %r1306, %r1305;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs369, %rs370}, %r1107;
	cvt.f32.bf16 	%r1307, %rs370;
	cvt.f32.bf16 	%r1308, %rs369;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1309, %r1512, %r751, %r1308;
	fma.rn.f32 	%r1310, %r1513, %r751, %r1307;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r860, %r1310, %r1309;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs371, %rs372}, %r1111;
	cvt.f32.bf16 	%r1311, %rs372;
	cvt.f32.bf16 	%r1312, %rs371;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1313, %r1514, %r752, %r1312;
	fma.rn.f32 	%r1314, %r1515, %r752, %r1311;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r861, %r1314, %r1313;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs373, %rs374}, %r1108;
	cvt.f32.bf16 	%r1315, %rs374;
	cvt.f32.bf16 	%r1316, %rs373;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1317, %r1516, %r751, %r1316;
	fma.rn.f32 	%r1318, %r1517, %r751, %r1315;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r864, %r1318, %r1317;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs375, %rs376}, %r1112;
	cvt.f32.bf16 	%r1319, %rs376;
	cvt.f32.bf16 	%r1320, %rs375;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1321, %r1518, %r752, %r1320;
	fma.rn.f32 	%r1322, %r1519, %r752, %r1319;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r865, %r1322, %r1321;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs377, %rs378}, %r1109;
	cvt.f32.bf16 	%r1323, %rs378;
	cvt.f32.bf16 	%r1324, %rs377;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1325, %r1520, %r751, %r1324;
	fma.rn.f32 	%r1326, %r1521, %r751, %r1323;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r868, %r1326, %r1325;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs379, %rs380}, %r1113;
	cvt.f32.bf16 	%r1327, %rs380;
	cvt.f32.bf16 	%r1328, %rs379;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1329, %r1522, %r752, %r1328;
	fma.rn.f32 	%r1330, %r1523, %r752, %r1327;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r869, %r1330, %r1329;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs381, %rs382}, %r1110;
	cvt.f32.bf16 	%r1331, %rs382;
	cvt.f32.bf16 	%r1332, %rs381;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1333, %r1524, %r751, %r1332;
	fma.rn.f32 	%r1334, %r1525, %r751, %r1331;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r872, %r1334, %r1333;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs383, %rs384}, %r1114;
	cvt.f32.bf16 	%r1335, %rs384;
	cvt.f32.bf16 	%r1336, %rs383;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1337, %r1526, %r752, %r1336;
	fma.rn.f32 	%r1338, %r1527, %r752, %r1335;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r873, %r1338, %r1337;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs385, %rs386}, %r1115;
	cvt.f32.bf16 	%r1339, %rs386;
	cvt.f32.bf16 	%r1340, %rs385;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1341, %r1528, %r753, %r1340;
	fma.rn.f32 	%r1342, %r1529, %r753, %r1339;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r874, %r1342, %r1341;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs387, %rs388}, %r1119;
	cvt.f32.bf16 	%r1343, %rs388;
	cvt.f32.bf16 	%r1344, %rs387;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1345, %r1530, %r754, %r1344;
	fma.rn.f32 	%r1346, %r1531, %r754, %r1343;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r875, %r1346, %r1345;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs389, %rs390}, %r1116;
	cvt.f32.bf16 	%r1347, %rs390;
	cvt.f32.bf16 	%r1348, %rs389;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1349, %r1532, %r753, %r1348;
	fma.rn.f32 	%r1350, %r1533, %r753, %r1347;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r878, %r1350, %r1349;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs391, %rs392}, %r1120;
	cvt.f32.bf16 	%r1351, %rs392;
	cvt.f32.bf16 	%r1352, %rs391;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1353, %r1534, %r754, %r1352;
	fma.rn.f32 	%r1354, %r1535, %r754, %r1351;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r879, %r1354, %r1353;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs393, %rs394}, %r1117;
	cvt.f32.bf16 	%r1355, %rs394;
	cvt.f32.bf16 	%r1356, %rs393;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1357, %r1536, %r753, %r1356;
	fma.rn.f32 	%r1358, %r1537, %r753, %r1355;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r882, %r1358, %r1357;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs395, %rs396}, %r1121;
	cvt.f32.bf16 	%r1359, %rs396;
	cvt.f32.bf16 	%r1360, %rs395;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1361, %r1538, %r754, %r1360;
	fma.rn.f32 	%r1362, %r1539, %r754, %r1359;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r883, %r1362, %r1361;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs397, %rs398}, %r1118;
	cvt.f32.bf16 	%r1363, %rs398;
	cvt.f32.bf16 	%r1364, %rs397;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1365, %r1540, %r753, %r1364;
	fma.rn.f32 	%r1366, %r1541, %r753, %r1363;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r886, %r1366, %r1365;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs399, %rs400}, %r1122;
	cvt.f32.bf16 	%r1367, %rs400;
	cvt.f32.bf16 	%r1368, %rs399;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1369, %r1542, %r754, %r1368;
	fma.rn.f32 	%r1370, %r1543, %r754, %r1367;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r887, %r1370, %r1369;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs401, %rs402}, %r1123;
	cvt.f32.bf16 	%r1371, %rs402;
	cvt.f32.bf16 	%r1372, %rs401;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1373, %r1544, %r755, %r1372;
	fma.rn.f32 	%r1374, %r1545, %r755, %r1371;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r876, %r1374, %r1373;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs403, %rs404}, %r1127;
	cvt.f32.bf16 	%r1375, %rs404;
	cvt.f32.bf16 	%r1376, %rs403;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1377, %r1546, %r756, %r1376;
	fma.rn.f32 	%r1378, %r1547, %r756, %r1375;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r877, %r1378, %r1377;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs405, %rs406}, %r1124;
	cvt.f32.bf16 	%r1379, %rs406;
	cvt.f32.bf16 	%r1380, %rs405;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1381, %r1548, %r755, %r1380;
	fma.rn.f32 	%r1382, %r1549, %r755, %r1379;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r880, %r1382, %r1381;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs407, %rs408}, %r1128;
	cvt.f32.bf16 	%r1383, %rs408;
	cvt.f32.bf16 	%r1384, %rs407;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1385, %r1550, %r756, %r1384;
	fma.rn.f32 	%r1386, %r1551, %r756, %r1383;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r881, %r1386, %r1385;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs409, %rs410}, %r1125;
	cvt.f32.bf16 	%r1387, %rs410;
	cvt.f32.bf16 	%r1388, %rs409;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1389, %r1552, %r755, %r1388;
	fma.rn.f32 	%r1390, %r1553, %r755, %r1387;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r884, %r1390, %r1389;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs411, %rs412}, %r1129;
	cvt.f32.bf16 	%r1391, %rs412;
	cvt.f32.bf16 	%r1392, %rs411;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1393, %r1554, %r756, %r1392;
	fma.rn.f32 	%r1394, %r1555, %r756, %r1391;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r885, %r1394, %r1393;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs413, %rs414}, %r1126;
	cvt.f32.bf16 	%r1395, %rs414;
	cvt.f32.bf16 	%r1396, %rs413;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1397, %r1556, %r755, %r1396;
	fma.rn.f32 	%r1398, %r1557, %r755, %r1395;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r888, %r1398, %r1397;
	.loc	1 839 99                        // sk04_fa_o_w4a8.py:839:99
	mov.b32 	{%rs415, %rs416}, %r1130;
	cvt.f32.bf16 	%r1399, %rs416;
	cvt.f32.bf16 	%r1400, %rs415;
	.loc	1 839 11                        // sk04_fa_o_w4a8.py:839:11
	fma.rn.f32 	%r1401, %r1558, %r756, %r1400;
	fma.rn.f32 	%r1402, %r1559, %r756, %r1399;
	.loc	1 845 15                        // sk04_fa_o_w4a8.py:845:15
	cvt.rn.bf16x2.f32 	%r889, %r1402, %r1401;
	bar.sync 	0;
	shl.b32 	%r1403, %r14, 12;
	shl.b32 	%r1404, %r14, 5;
	and.b32 	%r1405, %r2, 24;
	shl.b32 	%r1406, %r1405, 4;
	bfe.s32 	%r1407, %r2, 2, 1;
	and.b32 	%r1408, %r1407, 2064;
	or.b32 	%r1409, %r1404, %r1406;
	or.b32 	%r1410, %r1408, %r1409;
	xor.b32 	%r1411, %r1410, %r1059;
	add.s32 	%r1412, %r96, %r1403;
	add.s32 	%r822, %r1412, %r1411;
	// begin inline asm
	st.shared.v4.b32 [ %r822 + 0 ], { %r823, %r824, %r825, %r826 };
	// end inline asm
	add.s32 	%r827, %r822, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r827 + 0 ], { %r828, %r829, %r830, %r831 };
	// end inline asm
	add.s32 	%r832, %r822, 1024;
	// begin inline asm
	st.shared.v4.b32 [ %r832 + 0 ], { %r833, %r834, %r835, %r836 };
	// end inline asm
	add.s32 	%r837, %r822, 1536;
	// begin inline asm
	st.shared.v4.b32 [ %r837 + 0 ], { %r838, %r839, %r840, %r841 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r1413, %r1405, 6;
	shl.b32 	%r1414, %r1061, 4;
	shl.b32 	%r1415, %r2, 1;
	and.b32 	%r1416, %r1415, 384;
	bfe.s32 	%r1417, %r2, 5, 1;
	and.b32 	%r1418, %r1417, 2064;
	or.b32 	%r1419, %r1413, %r1414;
	or.b32 	%r1420, %r1418, %r1416;
	xor.b32 	%r1421, %r1420, %r1419;
	add.s32 	%r1422, %r96, %r1421;
	ld.shared.v4.b32 	{%r890, %r894, %r898, %r902}, [%r1422];
	xor.b32 	%r1423, %r1421, 32;
	add.s32 	%r1424, %r96, %r1423;
	ld.shared.v4.b32 	{%r891, %r895, %r899, %r903}, [%r1424+4096];
	xor.b32 	%r1425, %r1421, 64;
	add.s32 	%r1426, %r96, %r1425;
	ld.shared.v4.b32 	{%r892, %r896, %r900, %r904}, [%r1426+8192];
	xor.b32 	%r1427, %r1421, 96;
	add.s32 	%r1428, %r96, %r1427;
	ld.shared.v4.b32 	{%r893, %r897, %r901, %r905}, [%r1428+12288];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r822 + 0 ], { %r842, %r843, %r844, %r845 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r827 + 0 ], { %r846, %r847, %r848, %r849 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r832 + 0 ], { %r850, %r851, %r852, %r853 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r837 + 0 ], { %r854, %r855, %r856, %r857 };
	// end inline asm
	bar.sync 	0;
	ld.shared.v4.b32 	{%r906, %r910, %r914, %r918}, [%r1422];
	ld.shared.v4.b32 	{%r907, %r911, %r915, %r919}, [%r1424+4096];
	ld.shared.v4.b32 	{%r908, %r912, %r916, %r920}, [%r1426+8192];
	ld.shared.v4.b32 	{%r909, %r913, %r917, %r921}, [%r1428+12288];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r822 + 0 ], { %r858, %r859, %r860, %r861 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r827 + 0 ], { %r862, %r863, %r864, %r865 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r832 + 0 ], { %r866, %r867, %r868, %r869 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r837 + 0 ], { %r870, %r871, %r872, %r873 };
	// end inline asm
	bar.sync 	0;
	ld.shared.v4.b32 	{%r922, %r926, %r930, %r934}, [%r1422];
	ld.shared.v4.b32 	{%r923, %r927, %r931, %r935}, [%r1424+4096];
	ld.shared.v4.b32 	{%r924, %r928, %r932, %r936}, [%r1426+8192];
	ld.shared.v4.b32 	{%r925, %r929, %r933, %r937}, [%r1428+12288];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r822 + 0 ], { %r874, %r875, %r876, %r877 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r827 + 0 ], { %r878, %r879, %r880, %r881 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r832 + 0 ], { %r882, %r883, %r884, %r885 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r837 + 0 ], { %r886, %r887, %r888, %r889 };
	// end inline asm
	bar.sync 	0;
	ld.shared.v4.b32 	{%r938, %r942, %r946, %r950}, [%r1422];
	ld.shared.v4.b32 	{%r939, %r943, %r947, %r951}, [%r1424+4096];
	ld.shared.v4.b32 	{%r940, %r944, %r948, %r952}, [%r1426+8192];
	ld.shared.v4.b32 	{%r941, %r945, %r949, %r953}, [%r1428+12288];
	.loc	1 845 8                         // sk04_fa_o_w4a8.py:845:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd192 + 0 ], { %r890, %r891, %r892, %r893 };
	// end inline asm
	// begin inline asm
	@%p9 st.global.v4.b32 [ %rd193 + 0 ], { %r894, %r895, %r896, %r897 };
	// end inline asm
	// begin inline asm
	@%p10 st.global.v4.b32 [ %rd194 + 0 ], { %r898, %r899, %r900, %r901 };
	// end inline asm
	// begin inline asm
	@%p11 st.global.v4.b32 [ %rd195 + 0 ], { %r902, %r903, %r904, %r905 };
	// end inline asm
	// begin inline asm
	@%p12 st.global.v4.b32 [ %rd196 + 0 ], { %r906, %r907, %r908, %r909 };
	// end inline asm
	// begin inline asm
	@%p13 st.global.v4.b32 [ %rd197 + 0 ], { %r910, %r911, %r912, %r913 };
	// end inline asm
	// begin inline asm
	@%p14 st.global.v4.b32 [ %rd198 + 0 ], { %r914, %r915, %r916, %r917 };
	// end inline asm
	// begin inline asm
	@%p15 st.global.v4.b32 [ %rd199 + 0 ], { %r918, %r919, %r920, %r921 };
	// end inline asm
	// begin inline asm
	@%p16 st.global.v4.b32 [ %rd200 + 0 ], { %r922, %r923, %r924, %r925 };
	// end inline asm
	// begin inline asm
	@%p17 st.global.v4.b32 [ %rd201 + 0 ], { %r926, %r927, %r928, %r929 };
	// end inline asm
	// begin inline asm
	@%p18 st.global.v4.b32 [ %rd202 + 0 ], { %r930, %r931, %r932, %r933 };
	// end inline asm
	// begin inline asm
	@%p19 st.global.v4.b32 [ %rd203 + 0 ], { %r934, %r935, %r936, %r937 };
	// end inline asm
	// begin inline asm
	@%p20 st.global.v4.b32 [ %rd204 + 0 ], { %r938, %r939, %r940, %r941 };
	// end inline asm
	// begin inline asm
	@%p21 st.global.v4.b32 [ %rd205 + 0 ], { %r942, %r943, %r944, %r945 };
	// end inline asm
	// begin inline asm
	@%p22 st.global.v4.b32 [ %rd206 + 0 ], { %r946, %r947, %r948, %r949 };
	// end inline asm
	// begin inline asm
	@%p23 st.global.v4.b32 [ %rd207 + 0 ], { %r950, %r951, %r952, %r953 };
	// end inline asm
	.loc	1 843 4                         // sk04_fa_o_w4a8.py:843:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk04_fa_o_w4a8.py"
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
.b8 5                                   // DW_FORM_data2
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 165                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x9e DW_TAG_compile_unit
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
.b8 52
.b8 95
.b8 102
.b8 97
.b8 95
.b8 111
.b8 95
.b8 119
.b8 52
.b8 97
.b8 56
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
.b8 2                                   // Abbrev [2] 0x47:0x19 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 52
.b8 95
.b8 102
.b8 97
.b8 95
.b8 111
.b8 95
.b8 119
.b8 52
.b8 97
.b8 56
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x60:0x48 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 71                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x75:0x19 DW_TAG_inlined_subroutine
.b32 71                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 40                                  // DW_AT_call_line
.b8 3
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x8e:0x19 DW_TAG_inlined_subroutine
.b32 71                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 41                                  // DW_AT_call_line
.b8 3
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_5 = _Nativo(
    "sk04_fa_o_w4a8/tile128x256x64_shift0_abi15",
    _PTX_5, "_sk04_fa_o_w4a8_kernel",
    warps=8, shared=33792,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 16, 17],
    horneado={10: 1, 12: 1, 14: 1, 18: 1, 19: 128, 20: 256, 21: 64, 22: 8},
    div16=[7, 8, 9, 11, 13, 15, 16, 17],
)


# (cfg, has_shift) -> variantes candidatas. La eleccion final
# entre ellas la hace _lanzar comparando los valores horneados:
# con residual y sin residual el ABI tiene distinta cantidad de
# params, porque stride_res_n solo se especializa cuando vale 1.
_POR_CFG = {}
_POR_CFG.setdefault(((16, 128, 128, 8, 8, 3), False), []).append(_VAR_0)
_POR_CFG.setdefault(((16, 128, 128, 8, 8, 3), False), []).append(_VAR_1)
_POR_CFG.setdefault(((64, 128, 128, 8, 4, 4), False), []).append(_VAR_2)
_POR_CFG.setdefault(((64, 128, 128, 8, 4, 4), False), []).append(_VAR_3)
_POR_CFG.setdefault(((128, 256, 64, 8, 8, 3), False), []).append(_VAR_4)
_POR_CFG.setdefault(((128, 256, 64, 8, 8, 3), False), []).append(_VAR_5)


def _lanzar(grid, cfg, has_shift, *args):
    """Elige la variante cuyos valores horneados coinciden con estos args."""
    cands = _POR_CFG.get((cfg, bool(has_shift)))
    if not cands:
        raise KeyError("no hay PTX embebido para cfg=%r has_shift=%r" % (cfg, has_shift))
    for v in cands:
        if all(args[p] == val for p, val in v.horneado.items()):
            return v((grid,) if isinstance(grid, int) else grid, *args)
    raise KeyError("no hay variante para cfg=%r has_shift=%r con estos strides" % (cfg, has_shift))
