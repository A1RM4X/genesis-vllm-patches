# SPDX-License-Identifier: Apache-2.0
"""Fused quant per-token + GEMM int8 — super-kernel Triton single-launch."""
from __future__ import annotations
import torch
import triton
import triton.language as tl

BLOCK_M = 32
BLOCK_N = 64
BLOCK_K = 32

def is_available() -> bool:
    return torch.cuda.is_available()

def is_triton_available() -> bool:
    return torch.cuda.is_available()

@triton.jit
def _fused_quant_gemm_kernel(
    a_ptr, b_ptr, out_ptr, b_scale_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_out_m, stride_out_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N
    amax = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        cur_k = k + offs_k
        mask_k = cur_k < K
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + cur_k[None, :] * stride_ak
        a_tile = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        amax = tl.maximum(amax, tl.max(tl.abs(a_tile.to(tl.float32)), axis=1))
    scale = tl.where(amax / 127.0 > 0, amax / 127.0, 1.0)
    b_scale = tl.load(b_scale_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, K, BLOCK_K):
        cur_k = k + offs_k
        mask_k = cur_k < K
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + cur_k[None, :] * stride_ak
        a_tile = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        q_scaled = a_tile / scale[:, None]
        q_int = (q_scaled + tl.where(q_scaled >= 0, 0.5, -0.5)).to(tl.int32)
        q_int = tl.where(q_int > 127, 127, q_int)
        q_int = tl.where(q_int < -127, -127, q_int)
        q_i8 = q_int.to(tl.int8)
        b_ptrs = b_ptr + cur_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_tile = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0).to(tl.int8)
        acc = acc + tl.dot(q_i8, b_tile)
    acc_f32 = acc.to(tl.float32) * scale[:, None] * b_scale[None, :]
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    tl.store(out_ptrs, acc_f32, mask=mask_m[:, None] & mask_n[None, :])

def fused_quant_gemm(
    a: torch.Tensor, b: torch.Tensor, b_scale: torch.Tensor,
    out_dtype: torch.dtype | None = None,
    block_m: int = BLOCK_M, block_n: int = BLOCK_N, block_k: int = BLOCK_K,
) -> torch.Tensor:
    if out_dtype is None:
        out_dtype = torch.bfloat16 if a.dtype == torch.bfloat16 else torch.float16
    orig_shape = a.shape
    flatten = a.dim() != 2
    if flatten:
        k = int(orig_shape[-1])
        m = a.numel() // k if k else 0
        a_2d = a.reshape(m, k)
        n = int(b.shape[1])
        out_shape = tuple(orig_shape[:-1]) + (n,)
    else:
        a_2d = a
        m, k = a_2d.shape
        n = int(b.shape[1])
        out_shape = None  # type: ignore
    if k != int(b.shape[0]):
        raise ValueError(f"fused_quant_gemm: K mismatch a {k} vs b {b.shape[0]}")
    if a_2d.numel() == 0 or m == 0 or n == 0:
        empty_shape = out_shape if flatten else (m, n)
        return torch.empty(empty_shape, dtype=out_dtype, device=a.device)  # type: ignore
    b_scale_1d = b_scale.reshape(-1).to(torch.float32)[:n].contiguous()
    if b_scale_1d.numel() != n:
        b_scale_1d = b_scale_1d.reshape(-1)[:n].contiguous()
        if b_scale_1d.numel() < n:
            b_scale_1d = torch.cat([b_scale_1d, torch.ones(n - b_scale_1d.numel(), dtype=torch.float32, device=b_scale_1d.device)])
    out_fp32 = torch.empty((m, n), dtype=torch.float32, device=a_2d.device)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _fused_quant_gemm_kernel[grid](
        a_2d, b, out_fp32, b_scale_1d, m, n, k,
        a_2d.stride(0), a_2d.stride(1), b.stride(0), b.stride(1),
        out_fp32.stride(0), out_fp32.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, num_warps=4, num_stages=2,
    )
    out_2d = out_fp32.to(out_dtype) if out_dtype != torch.float32 else out_fp32
    return out_2d.reshape(out_shape) if flatten else out_2d  # type: ignore

quant_gemm = fused_quant_gemm
fused_gemm = fused_quant_gemm
__all__ = ["fused_quant_gemm", "quant_gemm", "fused_gemm", "is_available", "is_triton_available", "BLOCK_M", "BLOCK_N", "BLOCK_K"]
