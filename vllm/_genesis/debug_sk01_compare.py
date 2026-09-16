import torch
from safetensors import safe_open
import os
from vllm._genesis.wiring.quantization.patch_PN110_int8_phase_dispatch import _build_int8_state
from vllm._genesis.kernels.sk01_gdn_qkvz import sk01_gdn_qkvz_gemm
from vllm._genesis.kernels.sk09_norm_embed import quant_per_token

snap_dir = "/root/.cache/huggingface/hub/models--orcarouter--Qwen3.8-27B-Uncensored-FP8/snapshots"
snap_dir = os.path.join(snap_dir, os.listdir(snap_dir)[0])
file_path = os.path.join(snap_dir, "model-00001-of-00007.safetensors")

# Layer 0 in_proj_qkvz (ColumnParallel: N=8192 per rank, K=5120)
# Let's inspect safetensors keys for layer 0
with safe_open(file_path, framework="pt", device="cuda:0") as f:
    keys = [k for k in f.keys() if "layers.0." in k]
    print("Layer 0 keys:", [k for k in keys if "qkv" in k or "linear_attn" in k or "self_attn" in k])
    
    # Try finding in_proj or qkv
    target_k = None
    for k in keys:
        if "in_proj" in k or "qkv" in k:
            target_k = k
            break
    if target_k:
        print("Testing tensor:", target_k)
        w0 = f.get_tensor(target_k)
        s0_key = target_k.replace(".weight", ".weight_scale_inv")
        s0 = f.get_tensor(s0_key) if s0_key in f.keys() else None
        print("w0 shape:", w0.shape, "s0 shape:", s0.shape if s0 is not None else None)

