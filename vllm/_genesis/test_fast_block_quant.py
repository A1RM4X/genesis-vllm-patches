import torch
import time
from safetensors import safe_open
import os

snap_dir = "/root/.cache/huggingface/hub/models--orcarouter--Qwen3.8-27B-Uncensored-FP8/snapshots"
snap_dir = os.path.join(snap_dir, os.listdir(snap_dir)[0])
f_path = os.path.join(snap_dir, "model-00001-of-00007.safetensors")

with safe_open(f_path, framework="pt", device="cuda:0") as f:
    w_fp8 = f.get_tensor("model.language_model.layers.0.mlp.down_proj.weight")[:, :8704].contiguous()
    s_fp8 = f.get_tensor("model.language_model.layers.0.mlp.down_proj.weight_scale_inv")[:, :68].contiguous()

N, K = w_fp8.shape  # 5120, 8704
bk, bn = 128, 128
w_fp8_T = w_fp8.t().contiguous()
s_T = s_fp8.t().contiguous()

t0 = time.perf_counter()
# Vectorized block-128 quantization
w_4d = w_fp8_T.reshape(K // bk, bk, N // bn, bn)
s_4d = s_T.reshape(K // bk, 1, N // bn, 1)
w_fp32_4d = w_4d.to(torch.float32) * s_4d.to(torch.float32)
amax_2d = w_fp32_4d.abs().amax(dim=(1, 3))
block_scales = torch.where(amax_2d > 1e-12, amax_2d / 127.0, torch.ones_like(amax_2d))
w_i8_4d = (w_fp32_4d / block_scales.reshape(K // bk, 1, N // bn, 1)).round().clamp(-127, 127).to(torch.int8)
b_col = torch.empty_strided((K, N), (1, K), dtype=torch.int8, device="cuda:0")
b_col.copy_(w_i8_4d.reshape(K, N))
torch.cuda.synchronize()
t1 = time.perf_counter()

print(f"Quantization took {(t1-t0)*1000:.2f} ms")
print(f"b_col shape: {b_col.shape}, strides: {b_col.stride()}")
print(f"block_scales shape: {block_scales.shape}, dtype: {block_scales.dtype}")

