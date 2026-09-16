import torch
import numpy as np
from vllm._genesis.kernels.sk_ops import SK_POR_ID, sk_gemm_op, registrar
from vllm._genesis.kernels.int8_hybrid_gemm import int8_hybrid_gemm
from vllm._genesis.kernels.sk09_norm_embed import quant_per_token

registrar()

def test_kernel(name, sk_id_str, K, N, M=4):
    print(f"=== Testing {name} ({sk_id_str}) K={K} N={N} M={M} ===")
    torch.manual_seed(42)
    device = "cuda:0"
    
    # Generate random activations and weights
    x = torch.randn((M, K), dtype=torch.bfloat16, device=device)
    
    # Quantize activations per-token
    a_i8, a_scales = quant_per_token(x) # a_i8 [M, K], a_scales [M]
    
    # Generate weights in int8 and shifts/scales
    b_col = torch.randint(-127, 127, (K, N), dtype=torch.int8, device=device)
    # Ensure column-major stride (1, K)
    b_col_cm = b_col.t().contiguous().t()
    
    w_scales = torch.rand(N, dtype=torch.float32, device=device) * 0.005 + 1e-4
    b_scales = w_scales.unsqueeze(0) # [1, N]
    
    shifts = torch.randint(-3, 1, (K // 128, N // 128), dtype=torch.int8, device=device)
    
    # 1. Reference via int8_hybrid_gemm
    out_hybrid = int8_hybrid_gemm(
        a_i8, b_col_cm, a_scales, b_scales, shifts, out_dtype=torch.bfloat16
    )
    
    # 2. Reference via sk_gemm_op
    sk_id = SK_POR_ID[sk_id_str]
    epi = torch.zeros(1, dtype=torch.bfloat16, device=device).as_strided((1, 1), (0, 0))
    out_sk = sk_gemm_op(
        a_i8, b_col_cm, a_scales.reshape(-1), w_scales.reshape(-1), shifts, epi, sk_id, torch.bfloat16
    )
    
    torch.cuda.synchronize()
    
    # Compare
    diff = (out_hybrid.float() - out_sk.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_diff = (diff / (out_hybrid.float().abs() + 1e-6)).max().item()
    
    print(f"out_hybrid shape: {out_hybrid.shape}, out_sk shape: {out_sk.shape}")
    print(f"Max abs diff: {max_diff:.6f}, Mean abs diff: {mean_diff:.6f}, Max rel diff: {rel_diff:.6f}")
    if max_diff > 0.1:
        print(f"FAIL: Large discrepancy in {name}!")
        print("out_hybrid sample:", out_hybrid[0, :8])
        print("out_sk sample:", out_sk[0, :8])
    else:
        print(f"PASS: {name} matches hybrid kernel!")
    print()

def main():
    test_kernel("SK-01 GDN QKVZ", "SK-01", 5120, 8192, M=4)
    test_kernel("SK-02 GDN OUT", "SK-02", 3072, 5120, M=4)
    test_kernel("SK-03 FA QKV", "SK-03", 5120, 7168, M=4)
    test_kernel("SK-04 FA O", "SK-04", 3072, 5120, M=4)
    test_kernel("SK-05 GATEUP", "SK-05", 5120, 17408, M=4)
    test_kernel("SK-06 MLP DOWN", "SK-06", 8704, 5120, M=4)
    test_kernel("SK-10 MTP DRAFT", "SK-10", 5120, 7168, M=4)

if __name__ == "__main__":
    main()
