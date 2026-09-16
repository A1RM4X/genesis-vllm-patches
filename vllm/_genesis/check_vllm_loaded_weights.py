import os
from safetensors import safe_open

snap_dir = "/root/.cache/huggingface/hub/models--orcarouter--Qwen3.8-27B-Uncensored-FP8/snapshots"
snap_dir = os.path.join(snap_dir, os.listdir(snap_dir)[0])

for f_name in sorted(os.listdir(snap_dir)):
    if not f_name.endswith(".safetensors"):
        continue
    f_path = os.path.join(snap_dir, f_name)
    with safe_open(f_path, framework="pt", device="cpu") as f:
        for k in sorted(f.keys()):
            if "layers.0.mlp.down_proj" in k or "layers.1.mlp.down_proj" in k or "layers.0.linear_attn" in k:
                t = f.get_tensor(k)
                print(f"{k}: shape={t.shape}, dtype={t.dtype}")

