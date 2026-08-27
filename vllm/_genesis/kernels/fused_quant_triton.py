# SPDX-License-Identifier: Apache-2.0
"""Fused Triton per-token INT8 quant — single-pass bf16->amax->scale->int8."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK_MAX = 8192


def is_triton_available() -> bool:
    return True


def is_available() -> bool:
    return torch.cuda.is_available()


@triton.jit
def _fused_quant_per_token_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    N,
    stride_xm,
    stride_xn,
    stride_out_m,
    stride_out_n,
    stride_scale,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + pid * stride_xm + offs * stride_xn, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    amax = tl.max(tl.abs(x_f32), axis=0)
    scale = tl.where(amax / 127.0 > 0, amax / 127.0, 1.0)
    tl.store(scale_ptr + pid * stride_scale, scale)
    q_scaled = x_f32 / scale
    bias = tl.where(q_scaled >= 0, 0.5, -0.5)
    q_int = (q_scaled + bias).to(tl.int32)
    q_int = tl.where(q_int > 127, 127, q_int)
    q_int = tl.where(q_int < -127, -127, q_int)
    tl.store(out_ptr + pid * stride_out_m + offs * stride_out_n, q_int.to(tl.int8), mask=mask)


def quant_activation_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token INT8 quant via fused Triton single-pass. [...,K] -> ([...,K] int8, [...,1] fp32)."""
    if not isinstance(x, torch.Tensor):
        raise ValueError("x debe ser torch.Tensor")
    if x.dim() < 1:
        raise ValueError(f"x.dim >=1 requerido, got {x.dim()}")
    if x.numel() == 0:
        return torch.empty_like(x, dtype=torch.int8), torch.ones(x.shape[:-1] + (1,), dtype=torch.float32, device=x.device)
    orig_shape = x.shape
    k = int(orig_shape[-1])
    m = x.numel() // k
    x_2d = x.reshape(m, k)
    n = x_2d.shape[1]
    assert n <= BLOCK_MAX, f"N={n} > BLOCK_MAX={BLOCK_MAX} — fail fast sin fallback"
    block = triton.next_power_of_2(n)
    out_2d = torch.empty((m, n), dtype=torch.int8, device=x.device)
    scale_2d = torch.empty((m,), dtype=torch.float32, device=x.device)
    _fused_quant_per_token_kernel[(m,)](
        x_2d, out_2d, scale_2d, n,
        x_2d.stride(0), x_2d.stride(1), out_2d.stride(0), out_2d.stride(1), scale_2d.stride(0),
        BLOCK=block,
    )
    return out_2d.reshape(orig_shape), scale_2d.reshape(orig_shape[:-1] + (1,)).contiguous()


fused_quant_per_token = quant_activation_per_token
fused_quant_triton = quant_activation_per_token

__all__ = ["quant_activation_per_token", "fused_quant_per_token", "fused_quant_triton", "is_available", "is_triton_available"]
