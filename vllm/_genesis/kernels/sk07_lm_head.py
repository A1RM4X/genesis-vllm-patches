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

import torch
import triton
import triton.language as tl

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


def _cfg(m: int) -> tuple[int, int, int, int, int, int]:
    return _CFG[(m > 32) + (m > 128) + (m > 1024)]


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
        p2 = tl.exp2(tl.load(sh_ptrs + kb * stride_shift_k).to(tl.float32))
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
    _sk07_quant_kernel[(M,)](
        hidden, q, s, K, hidden.stride(0), q.stride(0),
        BLOCK=triton.next_power_of_2(K), num_warps=8, num_stages=1,
    )
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
    _sk07_lm_head_kernel[grid](
        a, weight, out, sampled_ids, res, a_scales, b_scales, shifts,
        M, S, K,
        a.stride(0), a.stride(1), weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1), rm, rs,
        shifts.stride(0), shifts.stride(1),
        BLOCK_M=bm, BLOCK_S=bs_, BLOCK_K=bk, GROUP_M=gm, SHIFT_BLOCK=SHIFT_BLOCK,
        num_warps=warps, num_stages=stages,
    )
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
