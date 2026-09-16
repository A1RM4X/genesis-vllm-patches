import torch
from safetensors import safe_open
import os
from vllm._genesis.wiring.quantization.patch_PN110_int8_phase_dispatch import _build_int8_state
from vllm._genesis.kernels.sk05_mlp_gateup import sk05_gateup_gemm
from vllm._genesis.kernels.sk06_mlp_down import mlp_down_gemm
from vllm._genesis.kernels.sk09_norm_embed import quant_per_token

snap_dir = "/root/.cache/huggingface/hub/models--orcarouter--Qwen3.8-27B-Uncensored-FP8/snapshots"
snap_dir = os.path.join(snap_dir, os.listdir(snap_dir)[0])
f_path = os.path.join(snap_dir, "model-00001-of-00007.safetensors")

with safe_open(f_path, framework="pt", device="cuda:0") as f:
    # Layer 0 MLP:
    wg = f.get_tensor("model.language_model.layers.0.mlp.gate_proj.weight")[:8704, :]
    wu = f.get_tensor("model.language_model.layers.0.mlp.up_proj.weight")[:8704, :]
    w_gu = torch.cat([wg, wu], dim=0).contiguous()
    sg = f.get_tensor("model.language_model.layers.0.mlp.gate_proj.weight_scale_inv")[:68, :]
    su = f.get_tensor("model.language_model.layers.0.mlp.up_proj.weight_scale_inv")[:68, :]
    s_gu = torch.cat([sg, su], dim=0).contiguous()

    wd = f.get_tensor("model.language_model.layers.0.mlp.down_proj.weight")[:, :8704].contiguous()
    sd = f.get_tensor("model.language_model.layers.0.mlp.down_proj.weight_scale_inv")[:, :68].contiguous()

class Layer:
    def __init__(self, w, s, k, n):
        self.weight = w
        self.weight_scale_inv = s
        self.input_size_per_partition = k
        self.output_size_per_partition = n
        self.bias = None

l_gu = Layer(w_gu, s_gu, 5120, 17408)
state_gu = _build_int8_state(l_gu, w_gu, s_gu, (128, 128))
bsc_gu = state_gu["b_scales"].reshape(-1)
sh_gu = torch.zeros((5120 // 128, 17408 // 128), dtype=torch.float32, device="cuda:0")

l_d = Layer(wd, sd, 8704, 5120)
state_d = _build_int8_state(l_d, wd, sd, (128, 128))
bsc_d = state_d["b_scales"].reshape(-1)
sh_d = torch.zeros((8704 // 128, 5120 // 128), dtype=torch.float32, device="cuda:0")

# Dequantized FP32 reference weights
w_gu_fp32 = torch.empty((17408, 5120), dtype=torch.float32, device="cuda:0")
for r in range(0, 17408, 128):
    for c in range(0, 5120, 128):
        w_gu_fp32[r:r+128, c:c+128] = w_gu[r:r+128, c:c+128].to(torch.float32) * s_gu[r//128, c//128].to(torch.float32)

w_d_fp32 = torch.empty((5120, 8704), dtype=torch.float32, device="cuda:0")
for r in range(0, 5120, 128):
    for c in range(0, 8704, 128):
        w_d_fp32[r:r+128, c:c+128] = wd[r:r+128, c:c+128].to(torch.float32) * sd[r//128, c//128].to(torch.float32)

torch.manual_seed(42)
# Simulate 1 token entering MLP
x = torch.randn((1, 5120), dtype=torch.bfloat16, device="cuda:0")

# 1. FP32 Reference MLP forward
gu_ref = (x.float() @ w_gu_fp32.t())
g_ref, u_ref = gu_ref.chunk(2, dim=-1)
inter_ref = torch.nn.functional.silu(g_ref) * u_ref
out_mlp_ref = (inter_ref @ w_d_fp32.t()).bfloat16()

# 2. Super Kernel MLP forward
# Step A: gate_up
a_i8, a_sc = quant_per_token(x)
gu_sk = sk05_gateup_gemm(a_i8, state_gu["b_col"], a_sc.reshape(-1), bsc_gu, sh_gu, None, torch.bfloat16)
g_sk, u_sk = gu_sk.chunk(2, dim=-1)
inter_sk = (torch.nn.functional.silu(g_sk.float()) * u_sk.float()).bfloat16()

# Step B: down_proj
inter_i8, inter_sc = quant_per_token(inter_sk)
out_mlp_sk = mlp_down_gemm(inter_i8, state_d["b_col"], inter_sc.reshape(-1), bsc_d, sh_d, None, torch.bfloat16)

cos_sim = torch.nn.functional.cosine_similarity(out_mlp_sk.float(), out_mlp_ref.float(), dim=-1).item()
max_diff = (out_mlp_sk.float() - out_mlp_ref.float()).abs().max().item()

print(f"MLP Layer 0 Output Cosine Similarity: {cos_sim:.6f}")
print(f"MLP Layer 0 Output Max Diff: {max_diff:.4f}")
print("out_mlp_sk sample :", out_mlp_sk[0, :8].tolist())
print("out_mlp_ref sample:", out_mlp_ref[0, :8].tolist())

