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

# Load down_proj weights for layer 0 and layer 1
with safe_open(f1, framework="pt", device="cuda:0") as f:
    w0_0 = f.get_tensor("model.language_model.layers.0.mlp.down_proj.weight")[:, :8704].contiguous()
    s0_0 = f.get_tensor("model.language_model.layers.0.mlp.down_proj.weight_scale_inv")[:, :68].contiguous()
    w0_1 = f.get_tensor("model.language_model.layers.0.mlp.down_proj.weight")[:, 8704:].contiguous()
    s0_1 = f.get_tensor("model.language_model.layers.0.mlp.down_proj.weight_scale_inv")[:, 68:].contiguous()

st0_0 = _build_int8_state(layer, w0_0, s0_0, (128, 128))
st0_1 = _build_int8_state(layer, w0_1, s0_1, (128, 128))

print("Built states OK")
