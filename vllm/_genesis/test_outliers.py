"""Que le hace el quant per-token INT8 a activaciones con outliers.

El quant es amax/127 POR FILA. Si un canal de la fila es 50-100x mas grande que
el resto -- que es exactamente el perfil de la salida de SwiGLU y de los estados
ocultos de un transformer -- ese canal fija el paso de cuantizacion y todo el
resto se aplasta contra el cero. Con gaussianas puras el efecto no aparece, y
por eso los tests sinteticos dan cos=0.9997.
"""
import os
import torch
from safetensors import safe_open
from vllm._genesis.wiring.quantization.patch_PN110_int8_phase_dispatch import (
    _build_int8_state, _sk_shifts_para)
from vllm._genesis.kernels.sk06_mlp_down import mlp_down_gemm
from vllm._genesis.kernels.sk09_norm_embed import quant_per_token

snap = "/home/usuario/Proyectos/models-cache/hub/models--orcarouter--Qwen3.8-27B-Uncensored-FP8/snapshots"
snap = os.path.join(snap, os.listdir(snap)[0])


class L:
    pass


with safe_open(os.path.join(snap, "model-00001-of-00007.safetensors"),
               framework="pt", device="cuda:0") as f:
    w = f.get_tensor("model.language_model.layers.0.mlp.down_proj.weight")[:, :8704].contiguous()
    s = f.get_tensor("model.language_model.layers.0.mlp.down_proj.weight_scale_inv")[:, :68].contiguous()

K, N = w.shape[1], w.shape[0]
st = _build_int8_state(L(), w, s, (128, 128))
wt = w.to(torch.float32).t().contiguous()
sT = s.to(torch.float32).t().contiguous()
w_ref = (wt.unflatten(0, (K // 128, 128)).unflatten(2, (N // 128, 128)).permute(0, 2, 1, 3)
         * sT.unsqueeze(-1).unsqueeze(-1)).permute(0, 2, 1, 3).reshape(K, N)
sh = _sk_shifts_para("SK-06", st.get("w_shifts"), K, N, w.device)

torch.manual_seed(0)
M = 16
print(f"{'perfil de activacion':<34}{'ratio outlier':>14}{'cos':>11}{'esc':>9}")
for nombre, factor, ncan in [("gaussiana pura", 1.0, 0),
                             ("outliers 10x en 8 canales", 10.0, 8),
                             ("outliers 50x en 8 canales", 50.0, 8),
                             ("outliers 100x en 4 canales", 100.0, 4),
                             ("outliers 300x en 2 canales", 300.0, 2)]:
    x = torch.randn((M, K), dtype=torch.float32, device="cuda")
    if ncan:
        idx = torch.randperm(K, device="cuda")[:ncan]
        x[:, idx] *= factor
    xb = x.to(torch.bfloat16)
    ref = x @ w_ref
    a_i8, a_sc = quant_per_token(xb)
    got = mlp_down_gemm(a_i8, st["b_col"], a_sc.reshape(-1),
                        st["b_scales"].reshape(-1), sh, None, torch.bfloat16).float()
    cos = torch.nn.functional.cosine_similarity(ref.flatten(), got.flatten(), dim=0).item()
    esc = (got.abs().mean() / ref.abs().mean()).item()
    ratio = (x.abs().amax(1) / x.abs().median(1).values).mean().item()
    print(f"{nombre:<34}{ratio:>14.1f}{cos:>11.6f}{esc:>9.4f}")
