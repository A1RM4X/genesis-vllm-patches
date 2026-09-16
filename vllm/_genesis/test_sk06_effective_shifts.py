import torch
from vllm._genesis.kernels.sk06_mlp_down import mlp_down_gemm

def test():
    device = "cuda:0"
    torch.manual_seed(42)
    
    M, K, N = 1, 8704, 5120
    
    # 1. Realistic input with outliers
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.5
    outliers = [10, 100, 500, 1000, 2500, 4000, 6000, 8000]
    x[0, outliers] = torch.tensor([80.0, -70.0, 95.0, -85.0, 60.0, -75.0, 90.0, -65.0], dtype=torch.bfloat16, device=device)
    
    # 2. Block 128x128 FP8 weight simulation
    w_fp32 = torch.randn(K, N, dtype=torch.float32, device=device) * 0.01
    w_reshaped = w_fp32.view(K // 128, 128, N // 128, 128).permute(0, 2, 1, 3)
    w_amax = w_reshaped.abs().amax(dim=(2, 3), keepdim=True).clamp_min(1e-5)
    b_shifts = (w_amax / 127.0).squeeze(-1).squeeze(-1) # [68, 40] fp32
    w_i8_tiles = (w_reshaped / w_amax * 127.0).round().clamp(-128, 127).to(torch.int8)
    w_i8 = w_i8_tiles.permute(0, 2, 1, 3).reshape(K, N)
    b_col_cm = w_i8.t().contiguous().t()
    
    # Reference full FP32
    resid = torch.randn(M, N, dtype=torch.bfloat16, device=device)
    ref_out = (x.float() @ w_fp32).to(torch.bfloat16) + resid
    
    # --- OLD WAY (global token amax) ---
    amax_global = x.abs().amax(dim=-1).clamp_min(1e-5) # [M]
    s_global = amax_global / 127.0
    a_i8_old = (x.float() / s_global[:, None]).round().clamp(-128, 127).to(torch.int8)
    
    dummy_b_scales = torch.ones(N, dtype=torch.float32, device=device)
    out_old = mlp_down_gemm(
        a_i8_old, b_col_cm, s_global, dummy_b_scales, b_shifts,
        residual=resid, out_dtype=torch.bfloat16
    )
    cos_old = torch.nn.functional.cosine_similarity(ref_out.float(), out_old.float(), dim=-1).item()
    
    # --- NEW WAY (grouped block amax folded into shifts) ---
    # x: [M, 68, 128]
    x_b = x.view(M, K // 128, 128).float()
    amax_b = x_b.abs().amax(dim=(0, 2)).clamp_min(1e-5) # [68]
    a_sc_b = amax_b / 127.0 # [68]
    
    # Quantize activations per block of 128
    a_i8_new = (x_b / a_sc_b[None, :, None]).round().clamp(-128, 127).to(torch.int8).view(M, K)
    
    # Fold a_sc_b [68, 1] into b_shifts [68, 40]!
    shifts_effective = b_shifts * a_sc_b[:, None] # [68, 40]
    
    # a_scales passed to kernel is now simply 1.0!
    ones_a_scale = torch.ones(M, dtype=torch.float32, device=device)
    
    out_new = mlp_down_gemm(
        a_i8_new, b_col_cm, ones_a_scale, dummy_b_scales, shifts_effective,
        residual=resid, out_dtype=torch.bfloat16
    )
    cos_new = torch.nn.functional.cosine_similarity(ref_out.float(), out_new.float(), dim=-1).item()
    
    print(f"OLD (global quant) Cosine Similarity: {cos_old:.6f}")
    print(f"NEW (grouped fold)  Cosine Similarity: {cos_new:.6f}")
    print(f"OLD zeros: {(a_i8_old == 0).sum().item()} / {K}")
    print(f"NEW zeros: {(a_i8_new == 0).sum().item()} / {K}")

if __name__ == "__main__":
    test()
