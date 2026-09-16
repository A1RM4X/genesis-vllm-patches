import torch
import triton
import triton.language as tl

@triton.jit
def _sk06_grouped_kernel(
    a_ptr, b_ptr, out_ptr, resid_ptr,
    a_scale_ptr, b_scale_ptr, shifts_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_out_m, stride_out_n,
    stride_res_m, stride_res_n,
    stride_asc_m, stride_asc_k,
    stride_shift_k, stride_shift_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    sh_ptrs = shifts_ptr + (offs_n // BLOCK_K) * stride_shift_n
    asc_ptrs = a_scale_ptr + offs_m[:, None] * stride_asc_m

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, K // BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        d = tl.dot(a, b, out_dtype=tl.float32)
        
        # Scale for this K-block (128):
        # b_sc is [1, BLOCK_N]
        b_sc = tl.load(sh_ptrs + kb * stride_shift_k).to(tl.float32)[None, :]
        # a_sc is [BLOCK_M, 1]
        a_sc = tl.load(asc_ptrs + kb * stride_asc_k).to(tl.float32)
        
        acc += d * (b_sc * a_sc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    out = acc
    if resid_ptr is not None:
        out += tl.load(resid_ptr + offs_m[:, None] * stride_res_m + offs_n[None, :] * stride_res_n).to(tl.float32)

    out_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(
        out_ptr + out_m[:, None] * stride_out_m + out_n[None, :] * stride_out_n,
        out.to(out_ptr.dtype.element_ty),
        mask=(out_m[:, None] < M) & (out_n[None, :] < N),
    )

def quant_grouped_128(x: torch.Tensor):
    # x: [M, K]
    M, K = x.shape
    assert K % 128 == 0
    x_reshaped = x.view(M, K // 128, 128).float()
    amax = x_reshaped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5) # [M, K//128, 1]
    scales = amax / 127.0 # [M, K//128, 1]
    q = (x_reshaped / scales).round().clamp(-128, 127).to(torch.int8)
    return q.view(M, K), scales.squeeze(-1) # q: [M, K], scales: [M, K//128]

def test():
    M, K, N = 1, 8704, 5120
    device = "cuda:0"
    
    # Simulate real activation with SwiGLU outliers
    torch.manual_seed(42)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.5
    outliers = [10, 100, 500, 1000, 2500, 4000, 6000, 8000]
    x[0, outliers] = torch.tensor([80.0, -70.0, 95.0, -85.0, 60.0, -75.0, 90.0, -65.0], dtype=torch.bfloat16, device=device)
    
    # Weight: FP8 block 128x128
    w_fp32 = torch.randn(K, N, dtype=torch.float32, device=device) * 0.01
    w_reshaped = w_fp32.view(K // 128, 128, N // 128, 128).permute(0, 2, 1, 3) # [K_b, N_b, 128, 128]
    w_amax = w_reshaped.abs().amax(dim=(2, 3), keepdim=True).clamp_min(1e-5)
    b_shifts = (w_amax / 127.0).squeeze(-1).squeeze(-1) # [K_b, N_b]
    w_i8_tiles = (w_reshaped / w_amax * 127.0).round().clamp(-128, 127).to(torch.int8)
    w_i8 = w_i8_tiles.permute(0, 2, 1, 3).reshape(K, N)
    
    # Column major layout for INT8 GEMM
    b_col_cm = w_i8.t().contiguous().t()
    
    # Reference full FP32
    ref_out = (x.float() @ w_fp32).to(torch.bfloat16)
    
    # Grouped quant
    a_i8, a_scales = quant_grouped_128(x)
    
    out = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    resid = torch.zeros(M, N, dtype=torch.bfloat16, device=device)
    
    grid = (triton.cdiv(M, 16) * triton.cdiv(N, 128),)
    _sk06_grouped_kernel[grid](
        a_i8, b_col_cm, out, resid,
        a_scales, None, b_shifts,
        M, N, K,
        a_i8.stride(0), a_i8.stride(1),
        b_col_cm.stride(0), b_col_cm.stride(1),
        out.stride(0), out.stride(1),
        resid.stride(0), resid.stride(1),
        a_scales.stride(0), a_scales.stride(1),
        b_shifts.stride(0), b_shifts.stride(1),
        BLOCK_M=16, BLOCK_N=128, BLOCK_K=128,
        GROUP_M=8,
    )
    
    torch.cuda.synchronize()
    cos = torch.nn.functional.cosine_similarity(ref_out.float(), out.float(), dim=-1).item()
    diff = (ref_out.float() - out.float()).abs()
    print(f"Grouped W8A8 Cosine Similarity: {cos:.6f}")
    print(f"Max abs diff: {diff.max().item():.6f}, Mean abs diff: {diff.mean().item():.6f}")

if __name__ == "__main__":
    test()
