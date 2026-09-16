import torch
from safetensors import safe_open
import os
from vllm._genesis.kernels.requant_inplace import requantizar_fp8_a_int8_inplace

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

# Reference FP32
w_ref = torch.empty((N, K), dtype=torch.float32, device="cuda:0")
for r in range(0, N, bn):
    for c in range(0, K, bk):
        w_ref[r:r+bn, c:c+bk] = w_fp8[r:r+bn, c:c+bk].to(torch.float32) * s_fp8[r//bn, c//bk].to(torch.float32)

# Hybrid Requantization Inplace
w_clone = w_fp8.clone()
b_col_h, w_scales_h, w_shifts_h = requantizar_fp8_a_int8_inplace(w_clone, s_fp8, (bk, bn))
# b_col_h is [K, N] column-major
# w_scales_h is [N, 1] fp32
# w_shifts_h is [K//128, N//128] int8

for M in [1, 4, 16, 69, 128]:
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda:0")
    s_x = x.float().abs().amax(dim=1, keepdim=True) / 127.0
    x_i8 = (x.float() / s_x).round().clamp(-127, 127).to(torch.int8)

    # Simulated Kernel execution with exp2(shift) in FP32
    acc_fp32 = torch.zeros((M, N), dtype=torch.float32, device="cuda:0")
    for kb in range(n_k):
        a_sub = x_i8[:, kb*bk : (kb+1)*bk].float()  # (M, 128)
        b_sub = b_col_h[kb*bk : (kb+1)*bk, :].float()  # (128, N)
        dot = a_sub @ b_sub  # (M, N)
        shift_k = w_shifts_h[kb, :].float()  # (N // 128,)
        # Expand shift to N columns
        shift_expanded = shift_k.repeat_interleave(bn)  # (N,)
        scale_shift = torch.exp2(shift_expanded)  # (N,)
        acc_fp32 += dot * scale_shift.unsqueeze(0)

    y_hybrid = (acc_fp32 * s_x * w_scales_h.t()).bfloat16()
    y_ref = (x.float() @ w_ref.t()).bfloat16()

    cos_sim = torch.nn.functional.cosine_similarity(y_hybrid.float().reshape(-1), y_ref.float().reshape(-1), dim=0).item()
    max_diff = (y_hybrid.float() - y_ref.float()).abs().max().item()
    mean_diff = (y_hybrid.float() - y_ref.float()).abs().mean().item()
    print(f"Hybrid exp2(shift) | M={M:4d} | CosSim={cos_sim:.6f} | MaxDiff={max_diff:.4f} | MeanDiff={mean_diff:.4f}")

