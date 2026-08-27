# SPDX-License-Identifier: Apache-2.0
"""int8 hybrid GEMM Diseño C — per-block shift 128. BLOCK_M=32 BLOCK_N=64 BLOCK_K=32 SHIFT_BLOCK=128. Branchless tl.where."""
from __future__ import annotations
import torch
import triton
import triton.language as tl
BLOCK_M = 32
BLOCK_N = 64
BLOCK_K = 32
SHIFT_BLOCK = 128
def is_triton_available() -> bool:
    return True
def is_available() -> bool:
    return torch.cuda.is_available()
@triton.jit
def _int8_hybrid_gemm_kernel(a_ptr, b_ptr, out_ptr, a_scale_ptr, b_scale_ptr, shifts_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_out_m, stride_out_n, stride_scales_a, stride_scales_b, stride_shift_k, stride_shift_n, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SHIFT_BLOCK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    a_scales = tl.load(a_scale_ptr + offs_m * stride_scales_a, mask=mask_m, other=0.0).to(tl.float32)
    b_scales = tl.load(b_scale_ptr + offs_n * stride_scales_b, mask=mask_n, other=0.0).to(tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    num_kb = K // SHIFT_BLOCK
    for kb in range(num_kb):
        shift_col = (pid_n * BLOCK_N) // SHIFT_BLOCK
        shift_val = tl.load(shifts_ptr + kb * stride_shift_k + shift_col * stride_shift_n).to(tl.int32)
        int_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        k_base = kb * SHIFT_BLOCK
        for sub in range(SHIFT_BLOCK // BLOCK_K):
            k_offs = k_base + sub * BLOCK_K + tl.arange(0, BLOCK_K)
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
            mask_a = mask_m[:, None] & (k_offs[None, :] < K)
            a_tile = tl.load(a_ptrs, mask=mask_a, other=0).to(tl.int8)
            b_ptrs = b_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
            mask_b = (k_offs[:, None] < K) & mask_n[None, :]
            b_tile = tl.load(b_ptrs, mask=mask_b, other=0).to(tl.int8)
            int_acc = int_acc + tl.dot(a_tile, b_tile)
        shifted = tl.where(shift_val >= 0, int_acc << shift_val, int_acc >> (-shift_val))
        shifted_f = shifted.to(tl.float32)
        scaled = shifted_f * a_scales[:, None] * b_scales[None, :]
        acc = acc + scaled
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    mask_out = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc, mask=mask_out)
def int8_hybrid_gemm(a: torch.Tensor, b: torch.Tensor, a_scales: torch.Tensor, b_scales: torch.Tensor, shifts: torch.Tensor, out_dtype: torch.dtype | None = None, block_k: int = 128) -> torch.Tensor:
    if out_dtype is None:
        out_dtype = torch.float16
    assert a.dtype == torch.int8
    assert b.dtype == torch.int8
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    assert K % block_k == 0
    assert N % block_k == 0
    assert shifts.shape == (K // block_k, N // block_k)
    if a_scales.dim() == 2:
        a_vec = a_scales.squeeze(1).contiguous().to(torch.float32) if a_scales.shape[1] == 1 else a_scales.reshape(-1).contiguous().to(torch.float32)[:M]
    else:
        a_vec = a_scales.contiguous().to(torch.float32)
        if a_vec.numel() > M:
            a_vec = a_vec[:M]
    if b_scales.dim() == 2:
        b_vec = b_scales.squeeze(1).contiguous().to(torch.float32) if b_scales.shape[0] == N else b_scales.reshape(-1).contiguous().to(torch.float32)[:N]
    else:
        b_vec = b_scales.contiguous().to(torch.float32)
        if b_vec.numel() > N:
            b_vec = b_vec[:N]
    out_fp32 = torch.empty((M, N), dtype=torch.float32, device=a.device)
    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bn = b.stride(0), b.stride(1)
    stride_out_m, stride_out_n = out_fp32.stride(0), out_fp32.stride(1)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _int8_hybrid_gemm_kernel[grid](a, b, out_fp32, a_vec, b_vec, shifts, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_out_m, stride_out_n, 1, 1, shifts.stride(0), shifts.stride(1), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, SHIFT_BLOCK=SHIFT_BLOCK, num_warps=4, num_stages=2)
    if out_dtype != torch.float32:
        return out_fp32.to(out_dtype)
    return out_fp32
hybrid_gemm = int8_hybrid_gemm
__all__ = ["int8_hybrid_gemm", "hybrid_gemm", "is_available", "is_triton_available"]
