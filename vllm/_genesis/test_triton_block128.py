import torch
import triton
import triton.language as tl
from safetensors import safe_open
import os

@triton.jit
def _block128_gemm_kernel(
    a_ptr, b_ptr, out_ptr, a_scale_ptr, b_scales_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_out_m, stride_out_n,
    stride_sk, stride_sn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    sc_ptrs = b_scales_ptr + (offs_n // 128) * stride_sn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, K // BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (kb * BLOCK_K + offs_k[None, :] < K), other=0)
        b = tl.load(b_ptrs, mask=(kb * BLOCK_K + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0)
        d = tl.dot(a, b, out_dtype=tl.float32)
        sc = tl.load(sc_ptrs + kb * stride_sk, mask=offs_n < N, other=0.0).to(tl.float32)
        acc += d * sc[None, :]
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    a_sc = tl.load(a_scale_ptr + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)
    out = acc * a_sc[:, None]
    tl.store(out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n,
             out.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


snap_dir = "/root/.cache/huggingface/hub/models--orcarouter--Qwen3.8-27B-Uncensored-FP8/snapshots"
snap_dir = os.path.join(snap_dir, os.listdir(snap_dir)[0])
f_path = os.path.join(snap_dir, "model-00001-of-00007.safetensors")

with safe_open(f_path, framework="pt", device="cuda:0") as f:
    w_fp8 = f.get_tensor("model.language_model.layers.0.mlp.down_proj.weight")[:, :8704].contiguous()
    s_fp8 = f.get_tensor("model.language_model.layers.0.mlp.down_proj.weight_scale_inv")[:, :68].contiguous()

N, K = w_fp8.shape
bk, bn = 128, 128
n_k, n_n = K // bk, N // bn

w_ref = torch.empty((N, K), dtype=torch.float32, device="cuda:0")
for r in range(0, N, bn):
    for c in range(0, K, bk):
        w_ref[r:r+bn, c:c+bk] = w_fp8[r:r+bn, c:c+bk].to(torch.float32) * s_fp8[r//bn, c//bk].to(torch.float32)

w_i8_blocks = torch.empty((K, N), dtype=torch.int8, device="cuda:0")
block_scales = torch.empty((n_k, n_n), dtype=torch.float32, device="cuda:0")

for r_blk in range(n_n):
    for c_blk in range(n_k):
        r_start, r_end = r_blk * bn, (r_blk + 1) * bn
        c_start, c_end = c_blk * bk, (c_blk + 1) * bk
        sub_ref = w_ref[r_start:r_end, c_start:c_end]
        amax = sub_ref.abs().max().item()
        scale = max(amax / 127.0, 1e-12)
        sub_i8 = (sub_ref / scale).round().clamp(-127, 127).to(torch.int8)
        w_i8_blocks[c_start:c_end, r_start:r_end] = sub_i8.t().contiguous()
        block_scales[c_blk, r_blk] = scale

for M in [1, 4, 16, 69, 128]:
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda:0")
    s_x = (x.float().abs().amax(dim=1) / 127.0).clamp(min=1e-30)
    x_i8 = (x.float() / s_x[:, None]).round().clamp(-127, 127).to(torch.int8)
    out_triton = torch.empty((M, N), dtype=torch.bfloat16, device="cuda:0")

    grid = (triton.cdiv(M, 16), triton.cdiv(N, 128))
    _block128_gemm_kernel[grid](
        x_i8, w_i8_blocks, out_triton, s_x, block_scales,
        M, N, K,
        x_i8.stride(0), x_i8.stride(1), w_i8_blocks.stride(0), w_i8_blocks.stride(1),
        out_triton.stride(0), out_triton.stride(1),
        block_scales.stride(0), block_scales.stride(1),
        BLOCK_M=16, BLOCK_N=128, BLOCK_K=128,
        num_warps=4, num_stages=3
    )

    y_ref = (x.float() @ w_ref.t()).bfloat16()
    cos_sim = torch.nn.functional.cosine_similarity(out_triton.float().reshape(-1), y_ref.float().reshape(-1), dim=0).item()
    max_diff = (out_triton.float() - y_ref.float()).abs().max().item()
    print(f"Triton Block-128 Kernel | M={M:4d} | CosSim={cos_sim:.6f} | MaxDiff={max_diff:.4f}")

