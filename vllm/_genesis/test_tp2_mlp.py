import torch
import torch.distributed as dist
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
    # Rank 0
    wg0 = f.get_tensor("model.language_model.layers.0.mlp.gate_proj.weight")[:8704, :]
    wu0 = f.get_tensor("model.language_model.layers.0.mlp.up_proj.weight")[:8704, :]
    w_gu0 = torch.cat([wg0, wu0], dim=0).contiguous()
    sg0 = f.get_tensor("model.language_model.layers.0.mlp.gate_proj.weight_scale_inv")[:68, :]
    su0 = f.get_tensor("model.language_model.layers.0.mlp.up_proj.weight_scale_inv")[:68, :]
    s_gu0 = torch.cat([sg0, su0], dim=0).contiguous()

    wd0 = f.get_tensor("model.language_model.layers.0.mlp.down_proj.weight")[:, :8704].contiguous()
    sd0 = f.get_tensor("model.language_model.layers.0.mlp.down_proj.weight_scale_inv")[:, :68].contiguous()

    # Rank 1
    wg1 = f.get_tensor("model.language_model.layers.0.mlp.gate_proj.weight")[8704:, :]
    wu1 = f.get_tensor("model.language_model.layers.0.mlp.up_proj.weight")[8704:, :]
    w_gu1 = torch.cat([wg1, wu1], dim=0).contiguous()
    sg1 = f.get_tensor("model.language_model.layers.0.mlp.gate_proj.weight_scale_inv")[68:, :]
    su1 = f.get_tensor("model.language_model.layers.0.mlp.up_proj.weight_scale_inv")[68:, :]
    s_gu1 = torch.cat([sg1, su1], dim=0).contiguous()

    wd1 = f.get_tensor("model.language_model.layers.0.mlp.down_proj.weight")[:, 8704:].contiguous()
    sd1 = f.get_tensor("model.language_model.layers.0.mlp.down_proj.weight_scale_inv")[:, 68:].contiguous()

class Layer:
    def __init__(self, w, s, k, n):
        self.weight = w
        self.weight_scale_inv = s
        self.input_size_per_partition = k
        self.output_size_per_partition = n
        self.bias = None

# Build states for Rank 0
state_gu0 = _build_int8_state(Layer(w_gu0, s_gu0, 5120, 17408), w_gu0, s_gu0, (128, 128))
state_d0 = _build_int8_state(Layer(wd0, sd0, 8704, 5120), wd0, sd0, (128, 128))

# Build states for Rank 1
state_gu1 = _build_int8_state(Layer(w_gu1, s_gu1, 5120, 17408), w_gu1, s_gu1, (128, 128))
state_d1 = _build_int8_state(Layer(wd1, sd1, 8704, 5120), wd1, sd1, (128, 128))

torch.manual_seed(42)
x = torch.randn((1, 5120), dtype=torch.bfloat16, device="cuda:0")

# Rank 0 forward:
a_i8_0, a_sc_0 = quant_per_token(x)
gu_0 = sk05_gateup_gemm(a_i8_0, state_gu0["b_col"], a_sc_0.reshape(-1), state_gu0["b_scales"].reshape(-1), torch.zeros((40, 136), device="cuda:0"), None, torch.bfloat16)
g0, u0 = gu_0.chunk(2, dim=-1)
inter_0 = (torch.nn.functional.silu(g0.float()) * u0.float()).bfloat16()
inter_i8_0, inter_sc_0 = quant_per_token(inter_0)
out_0 = mlp_down_gemm(inter_i8_0, state_d0["b_col"], inter_sc_0.reshape(-1), state_d0["b_scales"].reshape(-1), torch.zeros((68, 40), device="cuda:0"), None, torch.bfloat16)

# Rank 1 forward:
a_i8_1, a_sc_1 = quant_per_token(x)
gu_1 = sk05_gateup_gemm(a_i8_1, state_gu1["b_col"], a_sc_1.reshape(-1), state_gu1["b_scales"].reshape(-1), torch.zeros((40, 136), device="cuda:0"), None, torch.bfloat16)
g1, u1 = gu_1.chunk(2, dim=-1)
inter_1 = (torch.nn.functional.silu(g1.float()) * u1.float()).bfloat16()
inter_i8_1, inter_sc_1 = quant_per_token(inter_1)
out_1 = mlp_down_gemm(inter_i8_1, state_d1["b_col"], inter_sc_1.reshape(-1), state_d1["b_scales"].reshape(-1), torch.zeros((68, 40), device="cuda:0"), None, torch.bfloat16)

# All-Reduce sum:
out_tp2 = out_0 + out_1

# Reference Full MLP forward:
wg_all = torch.cat([wg0, wg1], dim=0) # 17408
wu_all = torch.cat([wu0, wu1], dim=0) # 17408
w_gu_all = torch.cat([wg_all, wu_all], dim=0) # 34816
s_gu_all = torch.cat([torch.cat([sg0, sg1], dim=0), torch.cat([su0, su1], dim=0)], dim=0)

w_gu_all_fp32 = torch.empty((34816, 5120), dtype=torch.float32, device="cuda:0")
for r in range(0, 34816, 128):
    for c in range(0, 5120, 128):
        w_gu_all_fp32[r:r+128, c:c+128] = w_gu_all[r:r+128, c:c+128].to(torch.float32) * s_gu_all[r//128, c//128].to(torch.float32)

wd_all = torch.cat([wd0, wd1], dim=1) # [5120, 17408]
sd_all = torch.cat([sd0, sd1], dim=1) # [40, 136]
wd_all_fp32 = torch.empty((5120, 17408), dtype=torch.float32, device="cuda:0")
for r in range(0, 5120, 128):
    for c in range(0, 17408, 128):
        wd_all_fp32[r:r+128, c:c+128] = wd_all[r:r+128, c:c+128].to(torch.float32) * sd_all[r//128, c//128].to(torch.float32)

gu_all_ref = (x.float() @ w_gu_all_fp32.t())
g_ref, u_ref = gu_all_ref.chunk(2, dim=-1)
inter_ref = torch.nn.functional.silu(g_ref) * u_ref
out_ref_all = (inter_ref @ wd_all_fp32.t()).bfloat16()

cos_sim = torch.nn.functional.cosine_similarity(out_tp2.float(), out_ref_all.float(), dim=-1).item()
max_diff = (out_tp2.float() - out_ref_all.float()).abs().max().item()

print(f"TP=2 Full MLP Output Cosine Similarity: {cos_sim:.6f}")
print(f"TP=2 Full MLP Output Max Diff: {max_diff:.4f}")
print("out_tp2 sample    :", out_tp2[0, :8].tolist())
print("out_ref_all sample:", out_ref_all[0, :8].tolist())

