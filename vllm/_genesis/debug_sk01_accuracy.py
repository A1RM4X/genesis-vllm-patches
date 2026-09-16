import torch
from safetensors import safe_open
import os
from vllm._genesis.wiring.quantization.patch_PN110_int8_phase_dispatch import _build_int8_state
from vllm._genesis.kernels.sk01_gdn_qkvz import sk01_gdn_qkvz_gemm
from vllm._genesis.kernels.sk09_norm_embed import quant_per_token

snap_dir = "/root/.cache/huggingface/hub/models--orcarouter--Qwen3.8-27B-Uncensored-FP8/snapshots"
snap_dir = os.path.join(snap_dir, os.listdir(snap_dir)[0])
file_path = os.path.join(snap_dir, "model-00001-of-00007.safetensors")

with safe_open(file_path, framework="pt", device="cuda:0") as f:
    # in_proj_qkv per rank (rank 0 has first 8192 rows)
    w0 = f.get_tensor("model.language_model.layers.0.linear_attn.in_proj_qkv.weight")[:8192, :].contiguous()
    s0 = f.get_tensor("model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale_inv")[:64, :].contiguous()

class Layer:
    def __init__(self, w, s):
        self.weight = w
        self.weight_scale_inv = s
        self.input_size_per_partition = 5120
        self.output_size_per_partition = 8192
        self.bias = None

layer = Layer(w0, s0)
state = _build_int8_state(layer, w0, s0, (128, 128))
b_scales = state["b_scales"].reshape(-1)

bk, bn = 128, 128
w0_fp32 = torch.empty((8192, 5120), dtype=torch.float32, device="cuda:0")
for r in range(0, 8192, bn):
    for c in range(0, 5120, bk):
        w0_fp32[r:r+bn, c:c+bk] = w0[r:r+bn, c:c+bk].to(torch.float32) * s0[r//bn, c//bk].to(torch.float32)

torch.manual_seed(42)
x = torch.randn((1, 5120), dtype=torch.bfloat16, device="cuda:0")
a_i8, a_scales = quant_per_token(x)
shifts = torch.zeros((5120 // 128, 8192 // 128), dtype=torch.float32, device="cuda:0")

out_sk = sk01_gdn_qkvz_gemm(a_i8, state["b_col"], a_scales.reshape(-1), b_scales, shifts, None, torch.bfloat16)
out_ref = (x.float() @ w0_fp32.t()).bfloat16()

print("out_sk sample :", out_sk[0, :8].tolist())
print("out_ref sample:", out_ref[0, :8].tolist())
print("Max diff:", (out_sk.float() - out_ref.float()).abs().max().item())

