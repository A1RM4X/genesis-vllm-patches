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
    w = f.get_tensor("model.language_model.layers.0.mlp.down_proj.weight")[:, :8704].contiguous()
    s = f.get_tensor("model.language_model.layers.0.mlp.down_proj.weight_scale_inv")[:, :68].contiguous()

N, K = w.shape
class Dummy: pass
layer = Dummy()
layer.input_size_per_partition = K
layer.output_size_per_partition = N
state = _build_int8_state(layer, w, s, (128, 128))

# Generar input representativo
torch.manual_seed(42)
x = torch.randn((1, K), dtype=torch.bfloat16, device="cuda:0")

# Dequant referencia exacta
bn, bk = 128, 128
w_ref = torch.empty((N, K), dtype=torch.float32, device="cuda:0")
for r in range(0, N, bn):
    for c in range(0, K, bk):
        w_ref[r:r+bn, c:c+bk] = w[r:r+bn, c:c+bk].to(torch.float32) * s[r//bn, c//bk].to(torch.float32)
y_ref = (x.float() @ w_ref.t()).bfloat16()

# SK-06
a_i8, a_sc = quant_per_token(x)
y_sk = mlp_down_gemm(
    a_i8, state["b_col"], a_sc.reshape(-1), state["b_scales"].reshape(-1),
    state["w_shifts"], None, torch.bfloat16
)

cos_sim = torch.nn.functional.cosine_similarity(y_sk.float().reshape(-1), y_ref.float().reshape(-1), dim=0).item()
diff = (y_sk.float() - y_ref.float()).abs().max().item()
print(f"Layer 0 down_proj | CosSim = {cos_sim:.6f} | MaxDiff = {diff:.4f}")
print(f"y_ref [:8]: {y_ref[0, :8]}")
print(f"y_sk  [:8]: {y_sk[0, :8]}")

