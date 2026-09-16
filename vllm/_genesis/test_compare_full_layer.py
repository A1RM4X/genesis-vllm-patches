import torch
import os
from vllm.config import ModelConfig, ParallelConfig, VllmConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

# Test if we can trace down_proj in isolation
from vllm._genesis.wiring.quantization.patch_PN110_int8_phase_dispatch import _build_int8_state
from vllm._genesis.kernels.sk06_mlp_down import mlp_down_gemm
from vllm._genesis.kernels.sk09_norm_embed import quant_per_token
from safetensors import safe_open

snap_dir = "/root/.cache/huggingface/hub/models--orcarouter--Qwen3.8-27B-Uncensored-FP8/snapshots"
snap_dir = os.path.join(snap_dir, os.listdir(snap_dir)[0])

# Cargar rank0 y rank1 de layer 0 down_proj
f1 = os.path.join(snap_dir, "model-00001-of-00007.safetensors")
with safe_open(f1, framework="pt", device="cuda:0") as f:
    w0 = f.get_tensor("model.language_model.layers.0.mlp.down_proj.weight")[:, :8704].contiguous()
    s0 = f.get_tensor("model.language_model.layers.0.mlp.down_proj.weight_scale_inv")[:, :68].contiguous()
    w1 = f.get_tensor("model.language_model.layers.0.mlp.down_proj.weight")[:, 8704:].contiguous()
    s1 = f.get_tensor("model.language_model.layers.0.mlp.down_proj.weight_scale_inv")[:, 68:].contiguous()

class Dummy: pass
l0, l1 = Dummy(), Dummy()
l0.input_size_per_partition, l0.output_size_per_partition = 8704, 5120
l1.input_size_per_partition, l1.output_size_per_partition = 8704, 5120

st0 = _build_int8_state(l0, w0, s0, (128, 128))
st1 = _build_int8_state(l1, w1, s1, (128, 128))

# Input completo x_full [M, 17408]
torch.manual_seed(123)
x_full = torch.randn((1, 17408), dtype=torch.bfloat16, device="cuda:0")
x0 = x_full[:, :8704].contiguous()
x1 = x_full[:, 8704:].contiguous()

# Reference exacta:
# W_ref_full: [5120, 17408]
w_full = torch.cat([w0, w1], dim=1)
s_full = torch.cat([s0, s1], dim=1)
w_ref_full = torch.empty((5120, 17408), dtype=torch.float32, device="cuda:0")
for r in range(0, 5120, 128):
    for c in range(0, 17408, 128):
        w_ref_full[r:r+128, c:c+128] = w_full[r:r+128, c:c+128].float() * s_full[r//128, c//128].float()
y_ref_full = (x_full.float() @ w_ref_full.t()).bfloat16()

# SK-06 por particion:
a0_i8, a0_sc = quant_per_token(x0)
y0_sk = mlp_down_gemm(a0_i8, st0["b_col"], a0_sc.reshape(-1), st0["b_scales"].reshape(-1), st0["w_shifts"], None, torch.bfloat16)

a1_i8, a1_sc = quant_per_token(x1)
y1_sk = mlp_down_gemm(a1_i8, st1["b_col"], a1_sc.reshape(-1), st1["b_scales"].reshape(-1), st1["w_shifts"], None, torch.bfloat16)

# All-reduce (suma de particiones)
y_sk_full = y0_sk + y1_sk

cos_sim = torch.nn.functional.cosine_similarity(y_sk_full.float().reshape(-1), y_ref_full.float().reshape(-1), dim=0).item()
diff = (y_sk_full.float() - y_ref_full.float()).abs().max().item()
print(f"FULL TP=2 down_proj | CosSim = {cos_sim:.6f} | MaxDiff = {diff:.4f}")
print("y_ref [:6]:", y_ref_full[0, :6].tolist())
print("y_sk  [:6]:", y_sk_full[0, :6].tolist())
