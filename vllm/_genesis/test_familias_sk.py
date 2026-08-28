"""Las seis familias de super kernel contra pesos reales y referencia fp32.

Pasa los shifts por `_sk_shifts_para`, o sea ejercita exactamente lo que arma
el bind. Sin sharding: usa el peso completo. Si una familia falla aca, el
problema es el kernel o el estado; si pasan todas, el problema es el cableado.
"""
import os
import torch
from safetensors import safe_open
from vllm._genesis.wiring.quantization.patch_PN110_int8_phase_dispatch import (
    _build_int8_state, _sk_shifts_para)
from vllm._genesis.kernels.sk01_gdn_qkvz import sk01_gdn_qkvz_gemm
from vllm._genesis.kernels.sk02_gdn_out import sk02_gemm_int8_scaled
from vllm._genesis.kernels.sk03_fa_qkv import sk03_fa_qkv_gemm
from vllm._genesis.kernels.sk04_fa_o import fa_o_int8_scaled_gemm
from vllm._genesis.kernels.sk05_mlp_gateup import sk05_gateup_gemm
from vllm._genesis.kernels.sk06_mlp_down import mlp_down_gemm
from vllm._genesis.kernels.sk09_norm_embed import quant_per_token

snap = "/home/usuario/Proyectos/models-cache/hub/models--orcarouter--Qwen3.8-27B-Uncensored-FP8/snapshots"
snap = os.path.join(snap, os.listdir(snap)[0])
F1 = os.path.join(snap, "model-00001-of-00007.safetensors")
F2 = os.path.join(snap, "model-00002-of-00007.safetensors")


class L:
    pass


def deq(w, s):
    """peso fp8 [N,K] + scale_inv [N/128,K/128] -> fp32 [K,N] dequantizado."""
    K, N = w.shape[1], w.shape[0]
    wt = w.to(torch.float32).t().contiguous()
    st = s.to(torch.float32).t().contiguous()
    return (wt.unflatten(0, (K // 128, 128)).unflatten(2, (N // 128, 128)).permute(0, 2, 1, 3)
            * st.unsqueeze(-1).unsqueeze(-1)).permute(0, 2, 1, 3).reshape(K, N)


def cargar(fp, nombres, fila):
    """Sharding TP=2, rank 0, igual que produccion.

    ColumnParallel (`fila=False`): cada componente del merge se parte por N, o
    sea por filas del peso [N,K], y recien despues se concatenan. Partir el
    merge ya concatenado daria un orden distinto.
    RowParallel (`fila=True`): se parte por K, las columnas.
    """
    ws, ss = [], []
    with safe_open(fp, framework="pt", device="cuda:0") as f:
        for n in nombres:
            w = f.get_tensor(n + ".weight")
            sc = f.get_tensor(n + ".weight_scale_inv")
            if fila:
                w, sc = w[:, : w.shape[1] // 2], sc[:, : sc.shape[1] // 2]
            else:
                w, sc = w[: w.shape[0] // 2], sc[: sc.shape[0] // 2]
            ws.append(w)
            ss.append(sc)
    return torch.cat(ws, 0).contiguous(), torch.cat(ss, 0).contiguous()


P = "model.language_model.layers"
# (familia, gemm, shard, tensores, es_row_parallel)
CASOS = [
    ("SK-01", sk01_gdn_qkvz_gemm, F1, [f"{P}.0.linear_attn.in_proj_qkv", f"{P}.0.linear_attn.in_proj_z"], False),
    ("SK-02", sk02_gemm_int8_scaled, F1, [f"{P}.0.linear_attn.out_proj"], True),
    ("SK-03", sk03_fa_qkv_gemm, F2, [f"{P}.11.self_attn.q_proj", f"{P}.11.self_attn.k_proj", f"{P}.11.self_attn.v_proj"], False),
    ("SK-04", fa_o_int8_scaled_gemm, F2, [f"{P}.11.self_attn.o_proj"], True),
    ("SK-05", sk05_gateup_gemm, F1, [f"{P}.0.mlp.gate_proj", f"{P}.0.mlp.up_proj"], False),
    ("SK-06", mlp_down_gemm, F1, [f"{P}.0.mlp.down_proj"], True),
]

torch.manual_seed(0)
print(f"{'fam':<7}{'K':>6}{'N':>7}{'M':>5}{'cos':>11}{'escala':>9}  veredicto")
for nombre, fn, fp, tensores, fila in CASOS:
    try:
        w, s = cargar(fp, tensores, fila)
    except Exception as e:
        print(f"{nombre:<7} no se pudo cargar: {type(e).__name__}: {e}")
        continue
    K, N = w.shape[1], w.shape[0]
    st = _build_int8_state(L(), w, s, (128, 128))
    w_ref = deq(w, s)
    sh = _sk_shifts_para(nombre, st.get("w_shifts"), K, N, w.device)
    for M in (1, 16, 130, 512, 1200, 4096):
        x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda:0")
        ref = x.to(torch.float32) @ w_ref
        a_i8, a_sc = quant_per_token(x)
        got = fn(a_i8, st["b_col"], a_sc.reshape(-1), st["b_scales"].reshape(-1),
                 sh, None, torch.bfloat16).float()
        cos = torch.nn.functional.cosine_similarity(ref.flatten(), got.flatten(), dim=0).item()
        esc = (got.abs().mean() / ref.abs().mean()).item()
        ok = cos > 0.99 and 0.9 < esc < 1.1
        print(f"{nombre:<7}{K:>6}{N:>7}{M:>5}{cos:>11.6f}{esc:>9.4f}  {'OK' if ok else '*** ROTO ***'}")
    del w, s, w_ref, st
    torch.cuda.empty_cache()
