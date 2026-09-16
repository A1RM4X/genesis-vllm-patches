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

state_gu = _build_int8_state(Layer(w_gu, s_gu, 5120, 17408), w_gu, s_gu, (128, 128))
bsc_gu = state_gu["b_scales"].reshape(-1)
sh_gu = torch.zeros((5120 // 128, 17408 // 128), dtype=torch.float32, device="cuda:0")

state_d = _build_int8_state(Layer(wd, sd, 8704, 5120), wd, sd, (128, 128))
bsc_d = state_d["b_scales"].reshape(-1)
sh_d = torch.zeros((8704 // 128, 5120 // 128), dtype=torch.float32, device="cuda:0")

w_gu_fp32 = torch.empty((17408, 5120), dtype=torch.float32, device="cuda:0")
for r in range(0, 17408, 128):
    for c in range(0, 5120, 128):
        w_gu_fp32[r:r+128, c:c+128] = w_gu[r:r+128, c:c+128].to(torch.float32) * s_gu[r//128, c//128].to(torch.float32)

for M in [1, 4, 8, 16, 32, 64, 69, 128, 512, 1024]:
    torch.manual_seed(42)
    x = torch.randn((M, 5120), dtype=torch.bfloat16, device="cuda:0")
    a_i8, a_sc = quant_per_token(x)
    gu_sk = sk05_gateup_gemm(a_i8, state_gu["b_col"], a_sc.reshape(-1), bsc_gu, sh_gu, None, torch.bfloat16)
    gu_ref = (x.float() @ w_gu_fp32.t()).bfloat16()
    cos_sim = torch.nn.functional.cosine_similarity(gu_sk.float().reshape(-1), gu_ref.float().reshape(-1), dim=0).item()
    max_diff = (gu_sk.float() - gu_ref.float()).abs().max().item()
    print(f"M={M:4d}: SK-05 cos_sim={cos_sim:.6f}, max_diff={max_diff:.4f}")

