from safetensors import safe_open
import os

snap_dir = "/root/.cache/huggingface/hub/models--orcarouter--Qwen3.8-27B-Uncensored-FP8/snapshots"
snap_dir = os.path.join(snap_dir, os.listdir(snap_dir)[0])
f_path = os.path.join(snap_dir, "model-00001-of-00007.safetensors")

with safe_open(f_path, framework="pt", device="cpu") as f:
    for k in sorted(f.keys()):
        if "layers.3.self_attn" in k:
            t = f.get_tensor(k)
            print(f"{k}: shape={t.shape}, dtype={t.dtype}")

