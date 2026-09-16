import torch
import os
from safetensors import safe_open
from vllm._genesis.wiring.quantization.patch_PN110_int8_phase_dispatch import _build_int8_state
from vllm._genesis.kernels.sk06_mlp_down import mlp_down_gemm
from vllm._genesis.kernels.sk09_norm_embed import quant_per_token

snap_dir = "/home/usuario/Proyectos/models-cache/hub/models--orcarouter--Qwen3.8-27B-Uncensored-FP8/snapshots"
snap_dir = os.path.join(snap_dir, os.listdir(snap_dir)[0])
f1 = os.path.join(snap_dir, "model-00001-of-00007.safetensors")

class DummyLayer:
    pass

layer = DummyLayer()

with safe_open(f1, framework="pt", device="cuda:0") as f:
    w0 = f.get_tensor("model.language_model.layers.1.mlp.down_proj.weight")[:, :8704].contiguous()
    s0 = f.get_tensor("model.language_model.layers.1.mlp.down_proj.weight_scale_inv")[:, :68].contiguous()
    w1 = f.get_tensor("model.language_model.layers.1.mlp.down_proj.weight")[:, 8704:].contiguous()
    s1 = f.get_tensor("model.language_model.layers.1.mlp.down_proj.weight_scale_inv")[:, 68:].contiguous()

st0 = _build_int8_state(layer, w0, s0, (128, 128))
st1 = _build_int8_state(layer, w1, s1, (128, 128))

# Test with M=4
torch.manual_seed(42)
x_full = torch.randn((4, 17408), dtype=torch.bfloat16, device="cuda:0")
x0 = x_full[:, :8704]
x1 = x_full[:, 8704:]

# Ref FP32
w0_f = w0.to(torch.float32).t().contiguous()
s0_f = s0.to(torch.float32).t().contiguous()
w0_f = (w0_f.unflatten(0, (68, 128)).unflatten(2, (40, 128)).permute(0,2,1,3) * s0_f.unsqueeze(-1).unsqueeze(-1)).permute(0,2,1,3).reshape(8704, 5120)

w1_f = w1.to(torch.float32).t().contiguous()
s1_f = s1.to(torch.float32).t().contiguous()
w1_f = (w1_f.unflatten(0, (68, 128)).unflatten(2, (40, 128)).permute(0,2,1,3) * s1_f.unsqueeze(-1).unsqueeze(-1)).permute(0,2,1,3).reshape(8704, 5120)

y0_ref = torch.matmul(x0.to(torch.float32), w0_f).to(torch.bfloat16)
y1_ref = torch.matmul(x1.to(torch.float32), w1_f).to(torch.bfloat16)
y_ref = y0_ref + y1_ref

a0_i8, a0_sc = quant_per_token(x0)
a1_i8, a1_sc = quant_per_token(x1)

y0_sk = mlp_down_gemm(a0_i8, st0["b_col"], a0_sc.reshape(-1), st0["b_scales"].reshape(-1), st0["w_shifts"], None, torch.bfloat16)
y1_sk = mlp_down_gemm(a1_i8, st1["b_col"], a1_sc.reshape(-1), st1["b_scales"].reshape(-1), st1["w_shifts"], None, torch.bfloat16)
y_sk = y0_sk + y1_sk

cos = torch.nn.functional.cosine_similarity(y_ref.flatten().float(), y_sk.flatten().float(), dim=0)
diff = (y_ref.float() - y_sk.float()).abs().max()
print(f"Layer 1 | CosSim = {cos.item():.6f} | MaxDiff = {diff.item():.4f}")
