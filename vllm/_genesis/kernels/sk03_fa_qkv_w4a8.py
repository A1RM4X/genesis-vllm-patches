# SPDX-License-Identifier: Apache-2.0
"""SK-03 W4A8 — FA_QKV_W4A8 — capa completa Full-Attention QKV en INT4/INT8.

Capa completa: RMSNorm(input_layernorm) + quant per-token -> GEMM W4A8 ->
split Q/gate/K/V. Geometría per-rank TP=2: K=5120, N=7168.

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

Además el acumulador pasó de bf16 (8 bits de mantisa, 160 parciales
encadenados) a INT32 por grupo con acumulación fp32, y el epílogo de escala
salió del bucle interno: antes se aplicaba una vez por cada ``BLOCK_K=32``.

Diseño (sm_86, mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32):
  * ``BLOCK_K == GROUP_SIZE == 128`` -> una escala de grupo por iteración.
  * Tiles grandes, punteros que avanzan, grid 1-D con swizzle L2, num_stages 3-4.

Los kernels son branchless y no validan nada: se asume todo comprobado.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

try:
    from vllm._genesis.kernels.ptx_runtime import KernelNativo, habilitado
except ImportError:
    from ptx_runtime import KernelNativo, habilitado

SK_ID = "SK-03-W4A8"
SK_NAME = "FA_QKV_W4A8"
N_GLOBAL: int = 14336
N_PER_RANK: int = 7168
K_PER_RANK: int = 5120

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
def _sk03_fa_qkv_w4a8_rmsnorm_quant_kernel(
    x_ptr, w_ptr, q_ptr, s_ptr, K, stride_xm, stride_qm,
    BLOCK: tl.constexpr, EPS: tl.constexpr,
):
    """RMSNorm + quant per-token en una pasada, fila por programa."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < K
    x = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * tl.rsqrt(tl.sum(x * x, axis=0) / K + EPS) * w
    amax = tl.maximum(tl.max(tl.abs(y), axis=0), 1e-30)
    yq = y * (127.0 / amax)
    q = (yq + tl.where(yq >= 0.0, 0.5, -0.5)).to(tl.int32)
    tl.store(q_ptr + row * stride_qm + offs, tl.minimum(tl.maximum(q, -127), 127).to(tl.int8), mask=mask)
    tl.store(s_ptr + row, amax * (1.0 / 127.0))


@triton.jit
def _sk03_fa_qkv_w4a8_splitk_kernel(
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
def _sk03_fa_qkv_w4a8_kernel(
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


# PTX editado (E1: copysign(0.5) -> lop3.b32 LUT 0xF8; E2: div.full por
# potencia de 2 -> mul por reciproco), gate bit-exacto vs JIT de Triton en
# 3 semillas. Horneado: BLOCK=8192 (K=5120), EPS=1e-6.
_PTX_RMSNORM_QUANT = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk03_fa_qkv_w4a8_rmsnorm_quant_kernel // -- Begin function _sk03_fa_qkv_w4a8_rmsnorm_quant_kernel
.extern .shared .align 16 .b8 global_smem[];
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
                                        // @_sk03_fa_qkv_w4a8_rmsnorm_quant_kernel
.visible .entry _sk03_fa_qkv_w4a8_rmsnorm_quant_kernel(
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_w4a8_rmsnorm_quant_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_w4a8_rmsnorm_quant_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_w4a8_rmsnorm_quant_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_w4a8_rmsnorm_quant_kernel_param_3,
	.param .u32 _sk03_fa_qkv_w4a8_rmsnorm_quant_kernel_param_4,
	.param .u32 _sk03_fa_qkv_w4a8_rmsnorm_quant_kernel_param_5,
	.param .u32 _sk03_fa_qkv_w4a8_rmsnorm_quant_kernel_param_6,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_w4a8_rmsnorm_quant_kernel_param_7,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_w4a8_rmsnorm_quant_kernel_param_8
)
.reqntid 256
{
	.reg .pred 	%p<40>;
	.reg .b16 	%rs<65>;
	.reg .b32 	%r<538>;
	.reg .b64 	%rd<21>;
	.loc	1 106 0                         // sk03_fa_qkv_w4a8.py:106:0
$L__func_begin0:
	.loc	1 106 0                         // sk03_fa_qkv_w4a8.py:106:0

// %bb.0:                               // %__nv_rsqrtf.exit
	ld.param.b64 	%rd12, [_sk03_fa_qkv_w4a8_rmsnorm_quant_kernel_param_0];
	ld.param.b64 	%rd13, [_sk03_fa_qkv_w4a8_rmsnorm_quant_kernel_param_1];
$L__tmp0:
	.loc	1 111 24                        // sk03_fa_qkv_w4a8.py:111:24
	mov.u32 	%r51, %ctaid.x;
	ld.param.b64 	%rd14, [_sk03_fa_qkv_w4a8_rmsnorm_quant_kernel_param_2];
	.loc	1 112 24                        // sk03_fa_qkv_w4a8.py:112:24
	mov.u32 	%r52, %tid.x;
	and.b32 	%r53, %r52, 255;
	ld.param.b64 	%rd15, [_sk03_fa_qkv_w4a8_rmsnorm_quant_kernel_param_3];
	and.b32 	%r54, %r52, 31;
	ld.param.b32 	%r55, [_sk03_fa_qkv_w4a8_rmsnorm_quant_kernel_param_4];
	shr.u32 	%r56, %r52, 5;
	ld.param.b32 	%r57, [_sk03_fa_qkv_w4a8_rmsnorm_quant_kernel_param_5];
	shl.b32 	%r58, %r52, 4;
	ld.param.b32 	%r59, [_sk03_fa_qkv_w4a8_rmsnorm_quant_kernel_param_6];
	and.b32 	%r60, %r58, 4080;
	or.b32 	%r61, %r60, 4096;
	.loc	1 113 18                        // sk03_fa_qkv_w4a8.py:113:18
	setp.lt.s32 	%p1, %r60, %r55;
	setp.lt.s32 	%p2, %r61, %r55;
	.loc	1 114 30                        // sk03_fa_qkv_w4a8.py:114:30
	mul.lo.s32 	%r62, %r57, %r51;
	.loc	1 114 24                        // sk03_fa_qkv_w4a8.py:114:24
	mad.wide.s32 	%rd16, %r62, 2, %rd12;
	.loc	1 114 42                        // sk03_fa_qkv_w4a8.py:114:42
	cvt.u64.u32 	%rd17, %r60;
	mul.wide.u32 	%rd18, %r60, 2;
	add.s64 	%rd1, %rd16, %rd18;
	add.s64 	%rd2, %rd1, 16;
	add.s64 	%rd3, %rd1, 8192;
	add.s64 	%rd4, %rd1, 8208;
	mov.b32 	%r5, 0;
	.loc	1 114 16                        // sk03_fa_qkv_w4a8.py:114:16
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
	.loc	1 114 73                        // sk03_fa_qkv_w4a8.py:114:73
	mov.b32 	{%rs1, %rs2}, %r2;
	cvt.f32.bf16 	%r63, %rs2;
	cvt.f32.bf16 	%r64, %rs1;
	mov.b32 	{%rs3, %rs4}, %r1;
	cvt.f32.bf16 	%r65, %rs3;
	cvt.f32.bf16 	%r66, %rs4;
	mov.b32 	{%rs5, %rs6}, %r4;
	cvt.f32.bf16 	%r67, %rs6;
	cvt.f32.bf16 	%r68, %rs5;
	mov.b32 	{%rs7, %rs8}, %r3;
	cvt.f32.bf16 	%r69, %rs8;
	cvt.f32.bf16 	%r70, %rs7;
	mov.b32 	{%rs9, %rs10}, %r7;
	cvt.f32.bf16 	%r71, %rs10;
	cvt.f32.bf16 	%r72, %rs9;
	mov.b32 	{%rs11, %rs12}, %r6;
	cvt.f32.bf16 	%r73, %rs12;
	cvt.f32.bf16 	%r74, %rs11;
	mov.b32 	{%rs13, %rs14}, %r9;
	cvt.f32.bf16 	%r75, %rs14;
	cvt.f32.bf16 	%r76, %rs13;
	mov.b32 	{%rs15, %rs16}, %r8;
	cvt.f32.bf16 	%r77, %rs16;
	cvt.f32.bf16 	%r78, %rs15;
	mov.b32 	{%rs17, %rs18}, %r11;
	cvt.f32.bf16 	%r79, %rs18;
	cvt.f32.bf16 	%r80, %rs17;
	mov.b32 	{%rs19, %rs20}, %r10;
	cvt.f32.bf16 	%r81, %rs20;
	cvt.f32.bf16 	%r82, %rs19;
	mov.b32 	{%rs21, %rs22}, %r13;
	cvt.f32.bf16 	%r83, %rs22;
	cvt.f32.bf16 	%r84, %rs21;
	mov.b32 	{%rs23, %rs24}, %r12;
	cvt.f32.bf16 	%r85, %rs24;
	cvt.f32.bf16 	%r86, %rs23;
	mov.b32 	{%rs25, %rs26}, %r15;
	cvt.f32.bf16 	%r87, %rs26;
	cvt.f32.bf16 	%r88, %rs25;
	mov.b32 	{%rs27, %rs28}, %r14;
	cvt.f32.bf16 	%r89, %rs28;
	cvt.f32.bf16 	%r90, %rs27;
	mov.b32 	{%rs29, %rs30}, %r17;
	cvt.f32.bf16 	%r91, %rs30;
	cvt.f32.bf16 	%r92, %rs29;
	mov.b32 	{%rs31, %rs32}, %r16;
	cvt.f32.bf16 	%r93, %rs32;
	cvt.f32.bf16 	%r94, %rs31;
	.loc	1 115 24                        // sk03_fa_qkv_w4a8.py:115:24
	add.s64 	%rd5, %rd13, %rd18;
	add.s64 	%rd6, %rd5, 16;
	add.s64 	%rd7, %rd5, 8192;
	add.s64 	%rd8, %rd5, 8208;
	.loc	1 115 16                        // sk03_fa_qkv_w4a8.py:115:16
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
	.loc	1 116 32                        // sk03_fa_qkv_w4a8.py:116:32
	mul.f32 	%r95, %r66, %r66;
$L__tmp1:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk03_fa_qkv_w4a8.py:116:28 ] ]
	fma.rn.f32 	%r96, %r65, %r65, %r95;
	fma.rn.f32 	%r97, %r64, %r64, %r96;
	fma.rn.f32 	%r98, %r63, %r63, %r97;
	fma.rn.f32 	%r99, %r70, %r70, %r98;
	fma.rn.f32 	%r100, %r69, %r69, %r99;
	fma.rn.f32 	%r101, %r68, %r68, %r100;
	fma.rn.f32 	%r102, %r67, %r67, %r101;
	fma.rn.f32 	%r103, %r74, %r74, %r102;
	fma.rn.f32 	%r104, %r73, %r73, %r103;
	fma.rn.f32 	%r105, %r72, %r72, %r104;
	fma.rn.f32 	%r106, %r71, %r71, %r105;
	fma.rn.f32 	%r107, %r78, %r78, %r106;
	fma.rn.f32 	%r108, %r77, %r77, %r107;
	fma.rn.f32 	%r109, %r76, %r76, %r108;
	fma.rn.f32 	%r110, %r75, %r75, %r109;
	fma.rn.f32 	%r111, %r82, %r82, %r110;
	fma.rn.f32 	%r112, %r81, %r81, %r111;
	fma.rn.f32 	%r113, %r80, %r80, %r112;
	fma.rn.f32 	%r114, %r79, %r79, %r113;
	fma.rn.f32 	%r115, %r86, %r86, %r114;
	fma.rn.f32 	%r116, %r85, %r85, %r115;
	fma.rn.f32 	%r117, %r84, %r84, %r116;
	fma.rn.f32 	%r118, %r83, %r83, %r117;
	fma.rn.f32 	%r119, %r90, %r90, %r118;
	fma.rn.f32 	%r120, %r89, %r89, %r119;
	fma.rn.f32 	%r121, %r88, %r88, %r120;
	fma.rn.f32 	%r122, %r87, %r87, %r121;
	fma.rn.f32 	%r123, %r94, %r94, %r122;
	fma.rn.f32 	%r124, %r93, %r93, %r123;
	fma.rn.f32 	%r125, %r92, %r92, %r124;
	fma.rn.f32 	%r126, %r91, %r91, %r125;
$L__tmp2:
	.loc	2 293 36                        // standard.py:293:36 @[ sk03_fa_qkv_w4a8.py:116:28 ]
	shfl.sync.bfly.b32 	%r127, %r126, 16, 31, -1;
$L__tmp3:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk03_fa_qkv_w4a8.py:116:28 ] ]
	add.f32 	%r128, %r126, %r127;
$L__tmp4:
	.loc	2 293 36                        // standard.py:293:36 @[ sk03_fa_qkv_w4a8.py:116:28 ]
	shfl.sync.bfly.b32 	%r129, %r128, 8, 31, -1;
$L__tmp5:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk03_fa_qkv_w4a8.py:116:28 ] ]
	add.f32 	%r130, %r128, %r129;
$L__tmp6:
	.loc	2 293 36                        // standard.py:293:36 @[ sk03_fa_qkv_w4a8.py:116:28 ]
	shfl.sync.bfly.b32 	%r131, %r130, 4, 31, -1;
$L__tmp7:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk03_fa_qkv_w4a8.py:116:28 ] ]
	add.f32 	%r132, %r130, %r131;
$L__tmp8:
	.loc	2 293 36                        // standard.py:293:36 @[ sk03_fa_qkv_w4a8.py:116:28 ]
	shfl.sync.bfly.b32 	%r133, %r132, 2, 31, -1;
$L__tmp9:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk03_fa_qkv_w4a8.py:116:28 ] ]
	add.f32 	%r134, %r132, %r133;
$L__tmp10:
	.loc	2 293 36                        // standard.py:293:36 @[ sk03_fa_qkv_w4a8.py:116:28 ]
	shfl.sync.bfly.b32 	%r135, %r134, 1, 31, -1;
$L__tmp11:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk03_fa_qkv_w4a8.py:116:28 ] ]
	add.f32 	%r35, %r134, %r135;
$L__tmp12:
	.loc	2 293 36                        // standard.py:293:36 @[ sk03_fa_qkv_w4a8.py:116:28 ]
	setp.eq.b32 	%p3, %r54, 0;
	shr.u32 	%r136, %r52, 3;
	and.b32 	%r137, %r136, 28;
	mov.b32 	%r138, global_smem;
	add.s32 	%r34, %r138, %r137;
	// begin inline asm
	@%p3 st.shared.b32 [ %r34 + 0 ], %r35;
	// end inline asm
	bar.sync 	0;
	setp.lt.u32 	%p4, %r53, 8;
	shl.b32 	%r139, %r53, 2;
	add.s32 	%r37, %r138, %r139;
	// begin inline asm
	@%p4 ld.shared.b32 %r36, [ %r37 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r140, %r36, 4, 31, -1;
$L__tmp13:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk03_fa_qkv_w4a8.py:116:28 ] ]
	add.f32 	%r141, %r36, %r140;
$L__tmp14:
	.loc	2 293 36                        // standard.py:293:36 @[ sk03_fa_qkv_w4a8.py:116:28 ]
	shfl.sync.bfly.b32 	%r142, %r141, 2, 31, -1;
$L__tmp15:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk03_fa_qkv_w4a8.py:116:28 ] ]
	add.f32 	%r143, %r141, %r142;
$L__tmp16:
	.loc	2 293 36                        // standard.py:293:36 @[ sk03_fa_qkv_w4a8.py:116:28 ]
	shfl.sync.bfly.b32 	%r144, %r143, 1, 31, -1;
$L__tmp17:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk03_fa_qkv_w4a8.py:116:28 ] ]
	add.f32 	%r38, %r143, %r144;
$L__tmp18:
	.loc	2 293 36                        // standard.py:293:36 @[ sk03_fa_qkv_w4a8.py:116:28 ]
	and.b32 	%r145, %r52, 7;
	setp.eq.b32 	%p7, %r145, 0;
	and.pred 	%p5, %p4, %p7;
	// begin inline asm
	@%p5 st.shared.b32 [ %r37 + 0 ], %r38;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r146, [global_smem];
$L__tmp19:
	.loc	1 116 45                        // sk03_fa_qkv_w4a8.py:116:45
	cvt.rn.f32.s32 	%r147, %r55;
	div.full.f32 	%r148, %r146, %r147;
	.loc	1 116 49                        // sk03_fa_qkv_w4a8.py:116:49
	add.f32 	%r149, %r148, 0f358637BD;
	.loc	1 116 21                        // sk03_fa_qkv_w4a8.py:116:21
	rsqrt.approx.ftz.f32 	%r150, %r149;
	.loc	1 116 12                        // sk03_fa_qkv_w4a8.py:116:12
	mul.f32 	%r151, %r150, %r65;
	mul.f32 	%r152, %r150, %r66;
	mul.f32 	%r153, %r150, %r64;
	mul.f32 	%r154, %r150, %r63;
	mul.f32 	%r155, %r150, %r70;
	mul.f32 	%r156, %r150, %r69;
	mul.f32 	%r157, %r150, %r68;
	mul.f32 	%r158, %r150, %r67;
	mul.f32 	%r159, %r150, %r74;
	mul.f32 	%r160, %r150, %r73;
	mul.f32 	%r161, %r150, %r72;
	mul.f32 	%r162, %r150, %r71;
	mul.f32 	%r163, %r150, %r78;
	mul.f32 	%r164, %r150, %r77;
	mul.f32 	%r165, %r150, %r76;
	mul.f32 	%r166, %r150, %r75;
	mul.f32 	%r167, %r150, %r82;
	mul.f32 	%r168, %r150, %r81;
	mul.f32 	%r169, %r150, %r80;
	mul.f32 	%r170, %r150, %r79;
	mul.f32 	%r171, %r150, %r86;
	mul.f32 	%r172, %r150, %r85;
	mul.f32 	%r173, %r150, %r84;
	mul.f32 	%r174, %r150, %r83;
	mul.f32 	%r175, %r150, %r90;
	mul.f32 	%r176, %r150, %r89;
	mul.f32 	%r177, %r150, %r88;
	mul.f32 	%r178, %r150, %r87;
	mul.f32 	%r179, %r150, %r94;
	mul.f32 	%r180, %r150, %r93;
	mul.f32 	%r181, %r150, %r92;
	mul.f32 	%r182, %r150, %r91;
$L__tmp20:
	.loc	2 191 40                        // standard.py:191:40 @[ sk03_fa_qkv_w4a8.py:117:29 ]
	bar.sync 	0;
$L__tmp21:
	.loc	1 120 27                        // sk03_fa_qkv_w4a8.py:120:27
	mul.lo.s32 	%r183, %r59, %r51;
	.loc	1 120 21                        // sk03_fa_qkv_w4a8.py:120:21
	cvt.s64.s32 	%rd19, %r183;
	add.s64 	%rd20, %rd14, %rd19;
	.loc	1 120 39                        // sk03_fa_qkv_w4a8.py:120:39
	add.s64 	%rd9, %rd20, %rd17;
	add.s64 	%rd10, %rd9, 4096;
	.loc	1 115 55                        // sk03_fa_qkv_w4a8.py:115:55
	mov.b32 	{%rs33, %rs34}, %r24;
	cvt.f32.bf16 	%r184, %rs33;
	cvt.f32.bf16 	%r185, %rs34;
	mov.b32 	{%rs35, %rs36}, %r25;
	cvt.f32.bf16 	%r186, %rs35;
	cvt.f32.bf16 	%r187, %rs36;
	.loc	1 116 56                        // sk03_fa_qkv_w4a8.py:116:56
	mul.f32 	%r188, %r166, %r187;
	mul.f32 	%r189, %r165, %r186;
	mul.f32 	%r190, %r164, %r185;
	mul.f32 	%r191, %r163, %r184;
	.loc	1 117 36                        // sk03_fa_qkv_w4a8.py:117:36
	abs.f32 	%r192, %r191;
	abs.f32 	%r193, %r190;
	abs.f32 	%r194, %r189;
	abs.f32 	%r195, %r188;
	.loc	1 115 55                        // sk03_fa_qkv_w4a8.py:115:55
	mov.b32 	{%rs37, %rs38}, %r22;
	cvt.f32.bf16 	%r196, %rs37;
	cvt.f32.bf16 	%r197, %rs38;
	mov.b32 	{%rs39, %rs40}, %r23;
	cvt.f32.bf16 	%r198, %rs39;
	cvt.f32.bf16 	%r199, %rs40;
	.loc	1 116 56                        // sk03_fa_qkv_w4a8.py:116:56
	mul.f32 	%r200, %r162, %r199;
	mul.f32 	%r201, %r161, %r198;
	mul.f32 	%r202, %r160, %r197;
	mul.f32 	%r203, %r159, %r196;
	.loc	1 117 36                        // sk03_fa_qkv_w4a8.py:117:36
	abs.f32 	%r204, %r203;
	abs.f32 	%r205, %r202;
	abs.f32 	%r206, %r201;
	abs.f32 	%r207, %r200;
	.loc	1 115 55                        // sk03_fa_qkv_w4a8.py:115:55
	mov.b32 	{%rs41, %rs42}, %r20;
	cvt.f32.bf16 	%r208, %rs41;
	cvt.f32.bf16 	%r209, %rs42;
	mov.b32 	{%rs43, %rs44}, %r21;
	cvt.f32.bf16 	%r210, %rs43;
	cvt.f32.bf16 	%r211, %rs44;
	.loc	1 116 56                        // sk03_fa_qkv_w4a8.py:116:56
	mul.f32 	%r212, %r158, %r211;
	mul.f32 	%r213, %r157, %r210;
	mul.f32 	%r214, %r156, %r209;
	mul.f32 	%r215, %r155, %r208;
	.loc	1 117 36                        // sk03_fa_qkv_w4a8.py:117:36
	abs.f32 	%r216, %r215;
	abs.f32 	%r217, %r214;
	abs.f32 	%r218, %r213;
	abs.f32 	%r219, %r212;
	.loc	1 115 55                        // sk03_fa_qkv_w4a8.py:115:55
	mov.b32 	{%rs45, %rs46}, %r18;
	cvt.f32.bf16 	%r220, %rs45;
	cvt.f32.bf16 	%r221, %rs46;
	mov.b32 	{%rs47, %rs48}, %r19;
	cvt.f32.bf16 	%r222, %rs47;
	cvt.f32.bf16 	%r223, %rs48;
	.loc	1 116 56                        // sk03_fa_qkv_w4a8.py:116:56
	mul.f32 	%r224, %r154, %r223;
	mul.f32 	%r225, %r153, %r222;
	mul.f32 	%r226, %r152, %r221;
	mul.f32 	%r227, %r151, %r220;
	.loc	1 117 36                        // sk03_fa_qkv_w4a8.py:117:36
	abs.f32 	%r228, %r227;
	abs.f32 	%r229, %r226;
	abs.f32 	%r230, %r225;
	abs.f32 	%r231, %r224;
$L__tmp22:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk03_fa_qkv_w4a8.py:117:29 ] ]
	max.f32 	%r232, %r228, %r229;
	max.f32 	%r233, %r232, %r230;
	max.f32 	%r234, %r233, %r231;
	max.f32 	%r235, %r234, %r216;
	max.f32 	%r236, %r235, %r217;
	max.f32 	%r237, %r236, %r218;
	max.f32 	%r238, %r237, %r219;
	max.f32 	%r239, %r238, %r204;
	max.f32 	%r240, %r239, %r205;
	max.f32 	%r241, %r240, %r206;
	max.f32 	%r242, %r241, %r207;
	max.f32 	%r243, %r242, %r192;
	max.f32 	%r244, %r243, %r193;
	max.f32 	%r245, %r244, %r194;
	max.f32 	%r246, %r245, %r195;
$L__tmp23:
	.loc	1 115 55                        // sk03_fa_qkv_w4a8.py:115:55
	mov.b32 	{%rs49, %rs50}, %r32;
	cvt.f32.bf16 	%r247, %rs49;
	cvt.f32.bf16 	%r248, %rs50;
	mov.b32 	{%rs51, %rs52}, %r33;
	cvt.f32.bf16 	%r249, %rs51;
	cvt.f32.bf16 	%r250, %rs52;
	.loc	1 116 56                        // sk03_fa_qkv_w4a8.py:116:56
	mul.f32 	%r251, %r182, %r250;
	mul.f32 	%r252, %r181, %r249;
	mul.f32 	%r253, %r180, %r248;
	mul.f32 	%r254, %r179, %r247;
	.loc	1 117 36                        // sk03_fa_qkv_w4a8.py:117:36
	abs.f32 	%r255, %r254;
	abs.f32 	%r256, %r253;
	abs.f32 	%r257, %r252;
	abs.f32 	%r258, %r251;
	.loc	1 115 55                        // sk03_fa_qkv_w4a8.py:115:55
	mov.b32 	{%rs53, %rs54}, %r30;
	cvt.f32.bf16 	%r259, %rs53;
	cvt.f32.bf16 	%r260, %rs54;
	mov.b32 	{%rs55, %rs56}, %r31;
	cvt.f32.bf16 	%r261, %rs55;
	cvt.f32.bf16 	%r262, %rs56;
	.loc	1 116 56                        // sk03_fa_qkv_w4a8.py:116:56
	mul.f32 	%r263, %r178, %r262;
	mul.f32 	%r264, %r177, %r261;
	mul.f32 	%r265, %r176, %r260;
	mul.f32 	%r266, %r175, %r259;
	.loc	1 117 36                        // sk03_fa_qkv_w4a8.py:117:36
	abs.f32 	%r267, %r266;
	abs.f32 	%r268, %r265;
	abs.f32 	%r269, %r264;
	abs.f32 	%r270, %r263;
	.loc	1 115 55                        // sk03_fa_qkv_w4a8.py:115:55
	mov.b32 	{%rs57, %rs58}, %r28;
	cvt.f32.bf16 	%r271, %rs57;
	cvt.f32.bf16 	%r272, %rs58;
	mov.b32 	{%rs59, %rs60}, %r29;
	cvt.f32.bf16 	%r273, %rs59;
	cvt.f32.bf16 	%r274, %rs60;
	.loc	1 116 56                        // sk03_fa_qkv_w4a8.py:116:56
	mul.f32 	%r275, %r174, %r274;
	mul.f32 	%r276, %r173, %r273;
	mul.f32 	%r277, %r172, %r272;
	mul.f32 	%r278, %r171, %r271;
	.loc	1 117 36                        // sk03_fa_qkv_w4a8.py:117:36
	abs.f32 	%r279, %r278;
	abs.f32 	%r280, %r277;
	abs.f32 	%r281, %r276;
	abs.f32 	%r282, %r275;
	.loc	1 115 55                        // sk03_fa_qkv_w4a8.py:115:55
	mov.b32 	{%rs61, %rs62}, %r26;
	cvt.f32.bf16 	%r283, %rs61;
	cvt.f32.bf16 	%r284, %rs62;
	mov.b32 	{%rs63, %rs64}, %r27;
	cvt.f32.bf16 	%r285, %rs63;
	cvt.f32.bf16 	%r286, %rs64;
	.loc	1 116 56                        // sk03_fa_qkv_w4a8.py:116:56
	mul.f32 	%r287, %r170, %r286;
	mul.f32 	%r288, %r169, %r285;
	mul.f32 	%r289, %r168, %r284;
	mul.f32 	%r290, %r167, %r283;
	.loc	1 117 36                        // sk03_fa_qkv_w4a8.py:117:36
	abs.f32 	%r291, %r290;
	abs.f32 	%r292, %r289;
	abs.f32 	%r293, %r288;
	abs.f32 	%r294, %r287;
$L__tmp24:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk03_fa_qkv_w4a8.py:117:29 ] ]
	max.f32 	%r295, %r246, %r291;
	max.f32 	%r296, %r295, %r292;
	max.f32 	%r297, %r296, %r293;
	max.f32 	%r298, %r297, %r294;
	max.f32 	%r299, %r298, %r279;
	max.f32 	%r300, %r299, %r280;
	max.f32 	%r301, %r300, %r281;
	max.f32 	%r302, %r301, %r282;
	max.f32 	%r303, %r302, %r267;
	max.f32 	%r304, %r303, %r268;
	max.f32 	%r305, %r304, %r269;
	max.f32 	%r306, %r305, %r270;
	max.f32 	%r307, %r306, %r255;
	max.f32 	%r308, %r307, %r256;
	max.f32 	%r309, %r308, %r257;
	max.f32 	%r310, %r309, %r258;
$L__tmp25:
	.loc	2 191 40                        // standard.py:191:40 @[ sk03_fa_qkv_w4a8.py:117:29 ]
	shfl.sync.bfly.b32 	%r311, %r310, 16, 31, -1;
$L__tmp26:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk03_fa_qkv_w4a8.py:117:29 ] ]
	max.f32 	%r312, %r310, %r311;
$L__tmp27:
	.loc	2 191 40                        // standard.py:191:40 @[ sk03_fa_qkv_w4a8.py:117:29 ]
	shfl.sync.bfly.b32 	%r313, %r312, 8, 31, -1;
$L__tmp28:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk03_fa_qkv_w4a8.py:117:29 ] ]
	max.f32 	%r314, %r312, %r313;
$L__tmp29:
	.loc	2 191 40                        // standard.py:191:40 @[ sk03_fa_qkv_w4a8.py:117:29 ]
	shfl.sync.bfly.b32 	%r315, %r314, 4, 31, -1;
$L__tmp30:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk03_fa_qkv_w4a8.py:117:29 ] ]
	max.f32 	%r316, %r314, %r315;
$L__tmp31:
	.loc	2 191 40                        // standard.py:191:40 @[ sk03_fa_qkv_w4a8.py:117:29 ]
	shfl.sync.bfly.b32 	%r317, %r316, 2, 31, -1;
$L__tmp32:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk03_fa_qkv_w4a8.py:117:29 ] ]
	max.f32 	%r318, %r316, %r317;
$L__tmp33:
	.loc	2 191 40                        // standard.py:191:40 @[ sk03_fa_qkv_w4a8.py:117:29 ]
	shfl.sync.bfly.b32 	%r319, %r318, 1, 31, -1;
$L__tmp34:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk03_fa_qkv_w4a8.py:117:29 ] ]
	max.f32 	%r39, %r318, %r319;
$L__tmp35:
	.loc	2 191 40                        // standard.py:191:40 @[ sk03_fa_qkv_w4a8.py:117:29 ]
	// begin inline asm
	@%p3 st.shared.b32 [ %r34 + 0 ], %r39;
	// end inline asm
	bar.sync 	0;
	// begin inline asm
	@%p4 ld.shared.b32 %r40, [ %r37 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r320, %r40, 4, 31, -1;
$L__tmp36:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk03_fa_qkv_w4a8.py:117:29 ] ]
	max.f32 	%r321, %r40, %r320;
$L__tmp37:
	.loc	2 191 40                        // standard.py:191:40 @[ sk03_fa_qkv_w4a8.py:117:29 ]
	shfl.sync.bfly.b32 	%r322, %r321, 2, 31, -1;
$L__tmp38:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk03_fa_qkv_w4a8.py:117:29 ] ]
	max.f32 	%r323, %r321, %r322;
$L__tmp39:
	.loc	2 191 40                        // standard.py:191:40 @[ sk03_fa_qkv_w4a8.py:117:29 ]
	shfl.sync.bfly.b32 	%r324, %r323, 1, 31, -1;
$L__tmp40:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk03_fa_qkv_w4a8.py:117:29 ] ]
	max.f32 	%r41, %r323, %r324;
$L__tmp41:
	.loc	2 191 40                        // standard.py:191:40 @[ sk03_fa_qkv_w4a8.py:117:29 ]
	// begin inline asm
	@%p5 st.shared.b32 [ %r37 + 0 ], %r41;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r325, [global_smem];
$L__tmp42:
	.loc	1 117 49                        // sk03_fa_qkv_w4a8.py:117:49
	max.f32 	%r326, %r325, 0f0DA24260;
	mov.b32 	%r327, 0f42FE0000;
	.loc	1 118 22                        // sk03_fa_qkv_w4a8.py:118:22
	div.full.f32 	%r328, %r327, %r326;
	.loc	1 118 14                        // sk03_fa_qkv_w4a8.py:118:14
	mul.f32 	%r329, %r226, %r328;
	mul.f32 	%r330, %r227, %r328;
	mul.f32 	%r331, %r224, %r328;
	mul.f32 	%r332, %r225, %r328;
	mul.f32 	%r333, %r214, %r328;
	mul.f32 	%r334, %r215, %r328;
	mul.f32 	%r335, %r212, %r328;
	mul.f32 	%r336, %r213, %r328;
	mul.f32 	%r337, %r202, %r328;
	mul.f32 	%r338, %r203, %r328;
	mul.f32 	%r339, %r200, %r328;
	mul.f32 	%r340, %r201, %r328;
	mul.f32 	%r341, %r190, %r328;
	mul.f32 	%r342, %r191, %r328;
	mul.f32 	%r343, %r188, %r328;
	mul.f32 	%r344, %r189, %r328;
	mul.f32 	%r345, %r289, %r328;
	mul.f32 	%r346, %r290, %r328;
	mul.f32 	%r347, %r287, %r328;
	mul.f32 	%r348, %r288, %r328;
	mul.f32 	%r349, %r277, %r328;
	mul.f32 	%r350, %r278, %r328;
	mul.f32 	%r351, %r275, %r328;
	mul.f32 	%r352, %r276, %r328;
	mul.f32 	%r353, %r265, %r328;
	mul.f32 	%r354, %r266, %r328;
	mul.f32 	%r355, %r263, %r328;
	mul.f32 	%r356, %r264, %r328;
	mul.f32 	%r357, %r253, %r328;
	mul.f32 	%r358, %r254, %r328;
	mul.f32 	%r359, %r251, %r328;
	mul.f32 	%r360, %r252, %r328;
	.loc	1 119 29                        // sk03_fa_qkv_w4a8.py:119:29
	.loc	1 119 39                        // sk03_fa_qkv_w4a8.py:119:39
	lop3.b32 	%r361, 0x3f000000, %r329, 0x80000000, 0xF8;
	lop3.b32 	%r362, 0x3f000000, %r330, 0x80000000, 0xF8;
	lop3.b32 	%r363, 0x3f000000, %r331, 0x80000000, 0xF8;
	lop3.b32 	%r364, 0x3f000000, %r332, 0x80000000, 0xF8;
	lop3.b32 	%r365, 0x3f000000, %r333, 0x80000000, 0xF8;
	lop3.b32 	%r366, 0x3f000000, %r334, 0x80000000, 0xF8;
	lop3.b32 	%r367, 0x3f000000, %r335, 0x80000000, 0xF8;
	lop3.b32 	%r368, 0x3f000000, %r336, 0x80000000, 0xF8;
	lop3.b32 	%r369, 0x3f000000, %r337, 0x80000000, 0xF8;
	lop3.b32 	%r370, 0x3f000000, %r338, 0x80000000, 0xF8;
	lop3.b32 	%r371, 0x3f000000, %r339, 0x80000000, 0xF8;
	lop3.b32 	%r372, 0x3f000000, %r340, 0x80000000, 0xF8;
	lop3.b32 	%r373, 0x3f000000, %r341, 0x80000000, 0xF8;
	lop3.b32 	%r374, 0x3f000000, %r342, 0x80000000, 0xF8;
	lop3.b32 	%r375, 0x3f000000, %r343, 0x80000000, 0xF8;
	lop3.b32 	%r376, 0x3f000000, %r344, 0x80000000, 0xF8;
	lop3.b32 	%r377, 0x3f000000, %r345, 0x80000000, 0xF8;
	lop3.b32 	%r378, 0x3f000000, %r346, 0x80000000, 0xF8;
	lop3.b32 	%r379, 0x3f000000, %r347, 0x80000000, 0xF8;
	lop3.b32 	%r380, 0x3f000000, %r348, 0x80000000, 0xF8;
	lop3.b32 	%r381, 0x3f000000, %r349, 0x80000000, 0xF8;
	lop3.b32 	%r382, 0x3f000000, %r350, 0x80000000, 0xF8;
	lop3.b32 	%r383, 0x3f000000, %r351, 0x80000000, 0xF8;
	lop3.b32 	%r384, 0x3f000000, %r352, 0x80000000, 0xF8;
	lop3.b32 	%r385, 0x3f000000, %r353, 0x80000000, 0xF8;
	lop3.b32 	%r386, 0x3f000000, %r354, 0x80000000, 0xF8;
	lop3.b32 	%r387, 0x3f000000, %r355, 0x80000000, 0xF8;
	lop3.b32 	%r388, 0x3f000000, %r356, 0x80000000, 0xF8;
	lop3.b32 	%r389, 0x3f000000, %r357, 0x80000000, 0xF8;
	lop3.b32 	%r390, 0x3f000000, %r358, 0x80000000, 0xF8;
	lop3.b32 	%r391, 0x3f000000, %r359, 0x80000000, 0xF8;
	lop3.b32 	%r392, 0x3f000000, %r360, 0x80000000, 0xF8;
	.loc	1 119 14                        // sk03_fa_qkv_w4a8.py:119:14
	fma.rn.f32 	%r393, %r225, %r328, %r364;
	fma.rn.f32 	%r394, %r224, %r328, %r363;
	fma.rn.f32 	%r395, %r227, %r328, %r362;
	fma.rn.f32 	%r396, %r226, %r328, %r361;
	fma.rn.f32 	%r397, %r213, %r328, %r368;
	fma.rn.f32 	%r398, %r212, %r328, %r367;
	fma.rn.f32 	%r399, %r215, %r328, %r366;
	fma.rn.f32 	%r400, %r214, %r328, %r365;
	fma.rn.f32 	%r401, %r201, %r328, %r372;
	fma.rn.f32 	%r402, %r200, %r328, %r371;
	fma.rn.f32 	%r403, %r203, %r328, %r370;
	fma.rn.f32 	%r404, %r202, %r328, %r369;
	fma.rn.f32 	%r405, %r189, %r328, %r376;
	fma.rn.f32 	%r406, %r188, %r328, %r375;
	fma.rn.f32 	%r407, %r191, %r328, %r374;
	fma.rn.f32 	%r408, %r190, %r328, %r373;
	fma.rn.f32 	%r409, %r288, %r328, %r380;
	fma.rn.f32 	%r410, %r287, %r328, %r379;
	fma.rn.f32 	%r411, %r290, %r328, %r378;
	fma.rn.f32 	%r412, %r289, %r328, %r377;
	fma.rn.f32 	%r413, %r276, %r328, %r384;
	fma.rn.f32 	%r414, %r275, %r328, %r383;
	fma.rn.f32 	%r415, %r278, %r328, %r382;
	fma.rn.f32 	%r416, %r277, %r328, %r381;
	fma.rn.f32 	%r417, %r264, %r328, %r388;
	fma.rn.f32 	%r418, %r263, %r328, %r387;
	fma.rn.f32 	%r419, %r266, %r328, %r386;
	fma.rn.f32 	%r420, %r265, %r328, %r385;
	fma.rn.f32 	%r421, %r252, %r328, %r392;
	fma.rn.f32 	%r422, %r251, %r328, %r391;
	fma.rn.f32 	%r423, %r254, %r328, %r390;
	fma.rn.f32 	%r424, %r253, %r328, %r389;
	.loc	1 119 49                        // sk03_fa_qkv_w4a8.py:119:49
	cvt.rzi.s32.f32 	%r425, %r396;
	cvt.rzi.s32.f32 	%r426, %r395;
	cvt.rzi.s32.f32 	%r427, %r394;
	cvt.rzi.s32.f32 	%r428, %r393;
	cvt.rzi.s32.f32 	%r429, %r400;
	cvt.rzi.s32.f32 	%r430, %r399;
	cvt.rzi.s32.f32 	%r431, %r398;
	cvt.rzi.s32.f32 	%r432, %r397;
	cvt.rzi.s32.f32 	%r433, %r404;
	cvt.rzi.s32.f32 	%r434, %r403;
	cvt.rzi.s32.f32 	%r435, %r402;
	cvt.rzi.s32.f32 	%r436, %r401;
	cvt.rzi.s32.f32 	%r437, %r408;
	cvt.rzi.s32.f32 	%r438, %r407;
	cvt.rzi.s32.f32 	%r439, %r406;
	cvt.rzi.s32.f32 	%r440, %r405;
	cvt.rzi.s32.f32 	%r441, %r412;
	cvt.rzi.s32.f32 	%r442, %r411;
	cvt.rzi.s32.f32 	%r443, %r410;
	cvt.rzi.s32.f32 	%r444, %r409;
	cvt.rzi.s32.f32 	%r445, %r416;
	cvt.rzi.s32.f32 	%r446, %r415;
	cvt.rzi.s32.f32 	%r447, %r414;
	cvt.rzi.s32.f32 	%r448, %r413;
	cvt.rzi.s32.f32 	%r449, %r420;
	cvt.rzi.s32.f32 	%r450, %r419;
	cvt.rzi.s32.f32 	%r451, %r418;
	cvt.rzi.s32.f32 	%r452, %r417;
	cvt.rzi.s32.f32 	%r453, %r424;
	cvt.rzi.s32.f32 	%r454, %r423;
	cvt.rzi.s32.f32 	%r455, %r422;
	cvt.rzi.s32.f32 	%r456, %r421;
	.loc	1 120 70                        // sk03_fa_qkv_w4a8.py:120:70
	max.s32 	%r457, %r428, -127;
	max.s32 	%r458, %r427, -127;
	max.s32 	%r459, %r426, -127;
	max.s32 	%r460, %r425, -127;
	max.s32 	%r461, %r432, -127;
	max.s32 	%r462, %r431, -127;
	max.s32 	%r463, %r430, -127;
	max.s32 	%r464, %r429, -127;
	max.s32 	%r465, %r436, -127;
	max.s32 	%r466, %r435, -127;
	max.s32 	%r467, %r434, -127;
	max.s32 	%r468, %r433, -127;
	max.s32 	%r469, %r440, -127;
	max.s32 	%r470, %r439, -127;
	max.s32 	%r471, %r438, -127;
	max.s32 	%r472, %r437, -127;
	max.s32 	%r473, %r444, -127;
	max.s32 	%r474, %r443, -127;
	max.s32 	%r475, %r442, -127;
	max.s32 	%r476, %r441, -127;
	max.s32 	%r477, %r448, -127;
	max.s32 	%r478, %r447, -127;
	max.s32 	%r479, %r446, -127;
	max.s32 	%r480, %r445, -127;
	max.s32 	%r481, %r452, -127;
	max.s32 	%r482, %r451, -127;
	max.s32 	%r483, %r450, -127;
	max.s32 	%r484, %r449, -127;
	max.s32 	%r485, %r456, -127;
	max.s32 	%r486, %r455, -127;
	max.s32 	%r487, %r454, -127;
	max.s32 	%r488, %r453, -127;
	.loc	1 120 77                        // sk03_fa_qkv_w4a8.py:120:77
	min.s32 	%r489, %r460, 127;
	min.s32 	%r490, %r459, 127;
	min.s32 	%r491, %r458, 127;
	min.s32 	%r492, %r457, 127;
	min.s32 	%r493, %r464, 127;
	min.s32 	%r494, %r463, 127;
	min.s32 	%r495, %r462, 127;
	min.s32 	%r496, %r461, 127;
	min.s32 	%r497, %r468, 127;
	min.s32 	%r498, %r467, 127;
	min.s32 	%r499, %r466, 127;
	min.s32 	%r500, %r465, 127;
	min.s32 	%r501, %r472, 127;
	min.s32 	%r502, %r471, 127;
	min.s32 	%r503, %r470, 127;
	min.s32 	%r504, %r469, 127;
	min.s32 	%r505, %r476, 127;
	min.s32 	%r506, %r475, 127;
	min.s32 	%r507, %r474, 127;
	min.s32 	%r508, %r473, 127;
	min.s32 	%r509, %r480, 127;
	min.s32 	%r510, %r479, 127;
	min.s32 	%r511, %r478, 127;
	min.s32 	%r512, %r477, 127;
	min.s32 	%r513, %r484, 127;
	min.s32 	%r514, %r483, 127;
	min.s32 	%r515, %r482, 127;
	min.s32 	%r516, %r481, 127;
	min.s32 	%r517, %r488, 127;
	min.s32 	%r518, %r487, 127;
	min.s32 	%r519, %r486, 127;
	min.s32 	%r520, %r485, 127;
	.loc	1 120 85                        // sk03_fa_qkv_w4a8.py:120:85
	prmt.b32 	%r521, %r492, %r491, 0x3340U;
	prmt.b32 	%r522, %r490, %r489, 0x3340U;
	prmt.b32 	%r42, %r522, %r521, 0x5410U;
	prmt.b32 	%r523, %r496, %r495, 0x3340U;
	prmt.b32 	%r524, %r494, %r493, 0x3340U;
	prmt.b32 	%r43, %r524, %r523, 0x5410U;
	prmt.b32 	%r525, %r500, %r499, 0x3340U;
	prmt.b32 	%r526, %r498, %r497, 0x3340U;
	prmt.b32 	%r44, %r526, %r525, 0x5410U;
	prmt.b32 	%r527, %r504, %r503, 0x3340U;
	prmt.b32 	%r528, %r502, %r501, 0x3340U;
	prmt.b32 	%r45, %r528, %r527, 0x5410U;
	prmt.b32 	%r529, %r508, %r507, 0x3340U;
	prmt.b32 	%r530, %r506, %r505, 0x3340U;
	prmt.b32 	%r46, %r530, %r529, 0x5410U;
	prmt.b32 	%r531, %r512, %r511, 0x3340U;
	prmt.b32 	%r532, %r510, %r509, 0x3340U;
	prmt.b32 	%r47, %r532, %r531, 0x5410U;
	prmt.b32 	%r533, %r516, %r515, 0x3340U;
	prmt.b32 	%r534, %r514, %r513, 0x3340U;
	prmt.b32 	%r48, %r534, %r533, 0x5410U;
	prmt.b32 	%r535, %r520, %r519, 0x3340U;
	prmt.b32 	%r536, %r518, %r517, 0x3340U;
	prmt.b32 	%r49, %r536, %r535, 0x5410U;
	.loc	1 120 45                        // sk03_fa_qkv_w4a8.py:120:45
	// begin inline asm
	@%p1 st.global.v4.b32 [ %rd9 + 0 ], { %r42, %r43, %r44, %r45 };
	// end inline asm
	// begin inline asm
	@%p2 st.global.v4.b32 [ %rd10 + 0 ], { %r46, %r47, %r48, %r49 };
	// end inline asm
	.loc	1 121 21                        // sk03_fa_qkv_w4a8.py:121:21
	mad.wide.u32 	%rd11, %r51, 4, %rd15;
	.loc	1 121 34                        // sk03_fa_qkv_w4a8.py:121:34
	mul.f32 	%r50, %r326, 0f3C010204;
	.loc	1 121 26                        // sk03_fa_qkv_w4a8.py:121:26
	or.b32 	%r537, %r54, %r56;
	setp.eq.b32 	%p6, %r537, 0;
	// begin inline asm
	@%p6 st.global.b32 [ %rd11 + 0 ], { %r50 };
	// end inline asm
	.loc	1 121 4                         // sk03_fa_qkv_w4a8.py:121:4
	ret;
$L__tmp43:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/home/usuario/Proyectos/genesis-vllm-patches/vllm/_genesis/kernels/sk03_fa_qkv_w4a8.py"
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
.b32 271                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x108 DW_TAG_compile_unit
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
.b8 51
.b8 95
.b8 102
.b8 97
.b8 95
.b8 113
.b8 107
.b8 118
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
.b8 2                                   // Abbrev [2] 0x70:0x29 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 51
.b8 95
.b8 102
.b8 97
.b8 95
.b8 113
.b8 107
.b8 118
.b8 95
.b8 119
.b8 52
.b8 97
.b8 56
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
.b8 3                                   // Abbrev [3] 0x99:0x79 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 112                                // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0xae:0x32 DW_TAG_inlined_subroutine
.b32 112                                // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp19                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 116                                 // DW_AT_call_line
.b8 28                                  // DW_AT_call_column
.b8 5                                   // Abbrev [5] 0xc6:0x19 DW_TAG_inlined_subroutine
.b32 112                                // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp18                          // DW_AT_high_pc
.b8 2                                   // DW_AT_call_file
.b8 37                                  // DW_AT_call_line
.b8 1
.b8 36                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 4                                   // Abbrev [4] 0xe0:0x31 DW_TAG_inlined_subroutine
.b32 112                                // DW_AT_abstract_origin
.b64 $L__tmp20                          // DW_AT_low_pc
.b64 $L__tmp42                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 117                                 // DW_AT_call_line
.b8 29                                  // DW_AT_call_column
.b8 6                                   // Abbrev [6] 0xf8:0x18 DW_TAG_inlined_subroutine
.b32 112                                // DW_AT_abstract_origin
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

_PTX_KERNEL = KernelNativo(
    "B2/sk03_fa_qkv_w4a8.py__rmsnorm_quant",
    _PTX_RMSNORM_QUANT,
    num_warps=8, shared=32,
    idx=[0, 1, 2, 3, 4, 5, 6], n_runtime=7,
    horneado={7: 8192, 8: 1e-6},
)


def sk03_fa_qkv_w4a8_rmsnorm_quant(
    hidden: torch.Tensor, ln_weight: torch.Tensor, eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm + quant per-token INT8 -> ``(q [M,K] int8, s [M] fp32)``.

    Producción: lanza el PTX embebido por libcuda cruda (sin JIT de Triton).
    ``GENESIS_PTQ_NATIVO=0`` cae al camino Triton de referencia.
    """
    if not habilitado():
        return _sk03_fa_qkv_w4a8_rmsnorm_quant_triton(hidden, ln_weight, eps)
    M, K = hidden.shape
    q = torch.empty((M, K), dtype=torch.int8, device=hidden.device)
    s = torch.empty((M,), dtype=torch.float32, device=hidden.device)
    _PTX_KERNEL((M,), hidden, ln_weight, q, s, K,
                hidden.stride(0), q.stride(0),
                triton.next_power_of_2(K), eps)
    return q, s


def _sk03_fa_qkv_w4a8_rmsnorm_quant_triton(
    hidden: torch.Tensor, ln_weight: torch.Tensor, eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Referencia para tests (misma firma): lanza el kernel Triton por JIT."""
    M, K = hidden.shape
    q = torch.empty((M, K), dtype=torch.int8, device=hidden.device)
    s = torch.empty((M,), dtype=torch.float32, device=hidden.device)
    _sk03_fa_qkv_w4a8_rmsnorm_quant_kernel[(M,)](
        hidden, ln_weight, q, s, K, hidden.stride(0), q.stride(0),
        BLOCK=triton.next_power_of_2(K), EPS=eps, num_warps=8, num_stages=1,
    )
    return q, s


def sk03_fa_qkv_w4a8_gemm(
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
        _sk03_fa_qkv_w4a8_splitk_kernel[(triton.cdiv(M, 16), triton.cdiv(N, SPLITK_BLOCK_N), SPLIT_K)](
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
    _sk03_fa_qkv_w4a8_kernel[grid](
        a, w_packed, out, res, a_scales, w_scales,
        M, N, K,
        a.stride(0), a.stride(1), w_packed.stride(0), w_packed.stride(1),
        out.stride(0), out.stride(1), rm, rn,
        w_scales.stride(0), w_scales.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=gm,
        num_warps=warps, num_stages=stages,
    )
    return out


def sk03_fa_qkv_w4a8_forward(
    hidden: torch.Tensor,
    ln_weight: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    eps: float = 1e-6,
    out_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Capa completa FA qkv W4A8: RMSNorm+quant -> GEMM -> split Q/gate/K/V."""
    a, a_scales = sk03_fa_qkv_w4a8_rmsnorm_quant(hidden, ln_weight, eps)
    qkv = sk03_fa_qkv_w4a8_gemm(a, w_packed, a_scales, w_scales, None, out_dtype)
    n = qkv.shape[1]
    q_n = (n * 3) // 7
    kv_n = n // 14
    o = q_n
    return qkv[:, 0:q_n], qkv[:, o:o + q_n], qkv[:, o + q_n:o + q_n + kv_n], qkv[:, o + q_n + kv_n:o + q_n + 2 * kv_n], qkv


__all__ = [
    "SK_ID", "SK_NAME", "N_GLOBAL", "N_PER_RANK", "K_PER_RANK",
    "GROUP_SIZE", "BLOCK_M", "BLOCK_N", "BLOCK_K", "SPLIT_K", "SPLITK_MAX_M", "SM_COUNT",
    "sk03_fa_qkv_w4a8_rmsnorm_quant", "sk03_fa_qkv_w4a8_gemm", "sk03_fa_qkv_w4a8_forward",
]
