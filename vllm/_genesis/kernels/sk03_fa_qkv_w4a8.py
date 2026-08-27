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


def sk03_fa_qkv_w4a8_rmsnorm_quant(
    hidden: torch.Tensor, ln_weight: torch.Tensor, eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm + quant per-token INT8 -> ``(q [M,K] int8, s [M] fp32)``."""
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
