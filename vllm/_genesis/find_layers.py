from safetensors import safe_open
import os

snap_dir = "/root/.cache/huggingface/hub/models--orcarouter--Qwen3.8-27B-Uncensored-FP8/snapshots"
snap_dir = os.path.join(snap_dir, os.listdir(snap_dir)[0])

for f_name in sorted(os.listdir(snap_dir)):
    if not f_name.endswith(".safetensors"):
        continue
    f_path = os.path.join(snap_dir, f_name)
    with safe_open(f_path, framework="pt", device="cpu") as f:
        keys = f.keys()
        self_attn_keys = [k for k in keys if "self_attn" in k]
        if self_attn_keys:
            print(f"{f_name}: {self_attn_keys[:5]}")

