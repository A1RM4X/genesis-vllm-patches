import torch
from safetensors import safe_open
import os

# Load real safetensors weights for down_proj
snap_dir = "/root/.cache/huggingface/hub/models--orcarouter--Qwen3.8-27B-Uncensored-FP8/snapshots"
snap_dir = os.path.join(snap_dir, os.listdir(snap_dir)[0])
f_path = os.path.join(snap_dir, "model-00001-of-00007.safetensors")

with safe_open(f_path, framework="pt", device="cuda:0") as f:
    w_fp8 = f.get_tensor("model.language_model.layers.0.mlp.down_proj.weight")[:, :8704].contiguous()
    s_fp8 = f.get_tensor("model.language_model.layers.0.mlp.down_proj.weight_scale_inv")[:, :68].contiguous()

N, K = w_fp8.shape  # 5120, 8704
bk, bn = 128, 128
n_k = K // bk  # 68
n_n = N // bn  # 40

# 1. Exact Reference FP32
w_ref = torch.empty((N, K), dtype=torch.float32, device="cuda:0")
for r in range(0, N, bn):
    for c in range(0, K, bk):
        w_ref[r:r+bn, c:c+bk] = w_fp8[r:r+bn, c:c+bk].to(torch.float32) * s_fp8[r//bn, c//bk].to(torch.float32)

# 2. INT8 Quantization PER 128x128 BLOCK (preserves exact block scale)
w_i8_blocks = torch.empty((K, N), dtype=torch.int8, device="cuda:0")
block_scales = torch.empty((n_k, n_n), dtype=torch.float32, device="cuda:0")

for r_blk in range(n_n):
    for c_blk in range(n_k):
        r_start, r_end = r_blk * bn, (r_blk + 1) * bn
        c_start, c_end = c_blk * bk, (c_blk + 1) * bk
        sub_ref = w_ref[r_start:r_end, c_start:c_end]  # shape (128, 128)
        amax = sub_ref.abs().max().item()
        if amax < 1e-12:
            scale = 1.0
            sub_i8 = torch.zeros((128, 128), dtype=torch.int8, device="cuda:0")
        else:
            scale = amax / 127.0
            sub_i8 = (sub_ref / scale).round().clamp(-127, 127).to(torch.int8)
        w_i8_blocks[c_start:c_end, r_start:r_end] = sub_i8.t().contiguous()
        block_scales[c_blk, r_blk] = scale

# 3. Simulate GEMM with Block-128 Scaling across M
for M in [1, 4, 16, 69, 128]:
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda:0")
    # Token-wise quant
    s_x = x.float().abs().amax(dim=1, keepdim=True) / 127.0
    x_i8 = (x.float() / s_x).round().clamp(-127, 127).to(torch.int8)

    # Simulated GEMM loop over K blocks
    acc_fp32 = torch.zeros((M, N), dtype=torch.float32, device="cuda:0")
    for c_blk in range(n_k):
        c_start, c_end = c_blk * bk, (c_blk + 1) * bk
        a_sub = x_i8[:, c_start:c_end].float()  # (M, 128)
        for r_blk in range(n_n):
            r_start, r_end = r_blk * bn, (r_blk + 1) * bn
            b_sub = w_i8_blocks[c_start:c_end, r_start:r_end].float()  # (128, 128)
            dot = a_sub @ b_sub  # (M, 128)
            scale = block_scales[c_blk, r_blk]
            acc_fp32[:, r_start:r_end] += dot * scale

    y_sk = (acc_fp32 * s_x).bfloat16()
    y_ref = (x.float() @ w_ref.t()).bfloat16()

    cos_sim = torch.nn.functional.cosine_similarity(y_sk.float().reshape(-1), y_ref.float().reshape(-1), dim=0).item()
    max_diff = (y_sk.float() - y_ref.float()).abs().max().item()
    mean_diff = (y_sk.float() - y_ref.float()).abs().mean().item()
    print(f"M={M:4d} | CosSim={cos_sim:.6f} | MaxDiff={max_diff:.4f} | MeanDiff={mean_diff:.4f}")

