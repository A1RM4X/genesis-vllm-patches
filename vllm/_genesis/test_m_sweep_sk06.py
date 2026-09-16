import torch
from safetensors import safe_open
import os
from vllm._genesis.wiring.quantization.patch_PN110_int8_phase_dispatch import _build_int8_state
from vllm._genesis.kernels.sk06_mlp_down import mlp_down_gemm
from vllm._genesis.kernels.sk09_norm_embed import quant_per_token

snap_dir = "/root/.cache/huggingface/hub/models--orcarouter--Qwen3.8-27B-Uncensored-FP8/snapshots"
snap_dir = os.path.join(snap_dir, os.listdir(snap_dir)[0])
f_path = os.path.join(snap_dir, "model-00001-of-00007.safetensors")

with safe_open(f_path, framework="pt", device="cuda:0") as f:
    wd = f.get_tensor("model.language_model.layers.0.mlp.down_proj.weight")[:, :8704].contiguous()
    sd = f.get_tensor("model.language_model.layers.0.mlp.down_proj.weight_scale_inv")[:, :68].contiguous()

class Layer:
    def __init__(self, w, s, k, n):
        self.weight = w
        self.weight_scale_inv = s
        self.input_size_per_partition = k
        self.output_size_per_partition = n
        self.bias = None

state_d = _build_int8_state(Layer(wd, sd, 8704, 5120), wd, sd, (128, 128))
bsc_d = state_d["b_scales"].reshape(-1)
sh_d = torch.zeros((8704 // 128, 5120 // 128), dtype=torch.float32, device="cuda:0")

w_d_fp32 = torch.empty((5120, 8704), dtype=torch.float32, device="cuda:0")
for r in range(0, 5120, 128):
    for c in range(0, 8704, 128):
        w_d_fp32[r:r+128, c:c+128] = wd[r:r+128, c:c+128].to(torch.float32) * sd[r//128, c//128].to(torch.float32)

for M in [1, 4, 8, 16, 32, 64, 69, 128, 512, 1024]:
    torch.manual_seed(42)
    x = torch.randn((M, 8704), dtype=torch.bfloat16, device="cuda:0")
    a_i8, a_sc = quant_per_token(x)
    d_sk = mlp_down_gemm(a_i8, state_d["b_col"], a_sc.reshape(-1), bsc_d, sh_d, None, torch.bfloat16)
    d_ref = (x.float() @ w_d_fp32.t()).bfloat16()
    cos_sim = torch.nn.functional.cosine_similarity(d_sk.float().reshape(-1), d_ref.float().reshape(-1), dim=0).item()
    max_diff = (d_sk.float() - d_ref.float()).abs().max().item()
    print(f"M={M:4d}: SK-06 cos_sim={cos_sim:.6f}, max_diff={max_diff:.4f}")

