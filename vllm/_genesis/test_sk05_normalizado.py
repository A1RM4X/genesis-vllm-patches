"""SK-05 con pesos reales y escalas por bloque, via la normalizacion del bind.

SK-05 nunca se migro del shift diadico: su bucle hace exp2(sh). Con la
cuantizacion por bloques el tensor es el factor, no el exponente, asi que
`_sk_shifts_para` le pasa log2(factor) y el exp2 del bucle lo deshace.
"""
import os
import torch
from safetensors import safe_open
from vllm._genesis.wiring.quantization.patch_PN110_int8_phase_dispatch import (
    _build_int8_state, _sk_shifts_para)
from vllm._genesis.kernels.sk05_mlp_gateup import sk05_gateup_gemm
from vllm._genesis.kernels.sk06_mlp_down import mlp_down_gemm
from vllm._genesis.kernels.sk09_norm_embed import quant_per_token

snap = "/home/usuario/Proyectos/models-cache/hub/models--orcarouter--Qwen3.8-27B-Uncensored-FP8/snapshots"
snap = os.path.join(snap, os.listdir(snap)[0])


class L:
    pass


def deq(w, s, bk=128, bn=128):
    K, N = w.shape[1], w.shape[0]
    wt = w.to(torch.float32).t().contiguous()
    st = s.to(torch.float32).t().contiguous()
    return (wt.unflatten(0, (K // bk, bk)).unflatten(2, (N // bn, bn)).permute(0, 2, 1, 3)
            * st.unsqueeze(-1).unsqueeze(-1)).permute(0, 2, 1, 3).reshape(K, N)


with safe_open(os.path.join(snap, "model-00001-of-00007.safetensors"),
               framework="pt", device="cuda:0") as f:
    # gate_up de la capa 0, rank 0 de TP=2: mitad de gate y mitad de up.
    wg = f.get_tensor("model.language_model.layers.0.mlp.gate_proj.weight")
    sg = f.get_tensor("model.language_model.layers.0.mlp.gate_proj.weight_scale_inv")
    wu = f.get_tensor("model.language_model.layers.0.mlp.up_proj.weight")
    su = f.get_tensor("model.language_model.layers.0.mlp.up_proj.weight_scale_inv")

n2 = wg.shape[0] // 2                     # per-rank
w = torch.cat([wg[:n2], wu[:n2]], dim=0).contiguous()
s = torch.cat([sg[:n2 // 128], su[:n2 // 128]], dim=0).contiguous()
K, N = w.shape[1], w.shape[0]
print(f"gate_up per-rank: K={K} N={N}")

st = _build_int8_state(L(), w, s, (128, 128))
w_ref = deq(w, s)

torch.manual_seed(0)
for M in (1, 4, 16, 130):
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda:0")
    ref = (x.to(torch.float32) @ w_ref).to(torch.float32)
    a_i8, a_sc = quant_per_token(x)
    sh = _sk_shifts_para("SK-05", st.get("w_shifts"), K, N, w.device)
    got = sk05_gateup_gemm(a_i8, st["b_col"], a_sc.reshape(-1),
                           st["b_scales"].reshape(-1), sh, None, torch.bfloat16)
    cos = torch.nn.functional.cosine_similarity(ref.flatten(), got.float().flatten(), dim=0).item()
    esc = (got.float().abs().mean() / ref.abs().mean()).item()
    print(f"M={M:>4}  cos={cos:.6f}  escala_got/ref={esc:.4f}  "
          f"{'OK' if cos > 0.99 and 0.9 < esc < 1.1 else 'ROTO'}")
