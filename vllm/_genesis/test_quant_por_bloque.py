"""Revision del quant por bloque de K para SK-06.

Tres esquemas contra referencia fp32, sobre activaciones con outliers
PERSISTENTES POR CANAL, que es la estructura real de la salida de SwiGLU:

  A  per-token            -- lo que hace quant_per_token hoy
  B  per-bloque-K global   -- el codigo nuevo: amax sobre (tokens, canales)
  C  per-token x per-bloque-K -- las dos escalas a la vez

C es posible con el kernel TAL CUAL esta: la escala por fila viaja en
`a_scales` (que el epilogo ya aplica) y la escala por bloque de K se pliega en
`shifts`, que se indexa justamente por bloque de K. No hace falta tocar el asm
ni perder la adaptacion por token.
"""
import os
import time
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
sh0 = _sk_shifts_para("SK-06", st.get("w_shifts"), K, N, w.device)
b_col, bsc = st["b_col"], st["b_scales"].reshape(-1)


def activacion(M, factor, ncan, semilla=0):
    """Outliers persistentes por CANAL: los mismos canales, en todos los tokens."""
    g = torch.Generator(device="cuda").manual_seed(semilla)
    x = torch.randn((M, K), dtype=torch.float32, device="cuda", generator=g)
    if ncan:
        idx = torch.randperm(K, device="cuda", generator=g)[:ncan]
        x[:, idx] *= factor
    return x.to(torch.bfloat16)


def esquema_A(xb):
    a_i8, a_sc = quant_per_token(xb)
    return a_i8, a_sc.reshape(-1), sh0


def esquema_B(xb):
    x_b = xb.view(-1, K // 128, 128).float()
    a_sc_b = x_b.abs().amax(dim=(0, 2)).clamp_min(1e-5) / 127.0
    a_i8 = (x_b / a_sc_b[None, :, None]).round().clamp(-128, 127).to(torch.int8).view(xb.shape[0], K)
    return a_i8, torch.ones(xb.shape[0], dtype=torch.float32, device="cuda"), sh0 * a_sc_b[:, None]


def esquema_C(xb):
    """Per-token x per-bloque-K. La parte por bloque se normaliza a media
    geometrica 1, asi que la por-fila sigue llevando la magnitud del token."""
    x_b = xb.view(-1, K // 128, 128).float()
    blk = x_b.abs().amax(dim=(0, 2)).clamp_min(1e-5)          # [K/128]
    blk = blk / blk.log().mean().exp()                         # media geometrica 1
    xn = x_b / blk[None, :, None]                              # aplanado por canal
    fila = xn.abs().amax(dim=(1, 2)).clamp_min(1e-30) / 127.0  # [M]
    a_i8 = (xn / fila[:, None, None]).round().clamp(-127, 127).to(torch.int8).view(xb.shape[0], K)
    return a_i8, fila.contiguous(), sh0 * blk[:, None]


ESQ = [("A per-token", esquema_A), ("B bloque-K global", esquema_B),
       ("C token x bloque-K", esquema_C)]

print("=== exactitud (M=256) ===")
print(f"{'activacion':<28}" + "".join(f"{n:>22}" for n, _ in ESQ))
for nombre, factor, ncan in [("gaussiana", 1.0, 0), ("outliers 20x, 16 canales", 20.0, 16),
                             ("outliers 100x, 8 canales", 100.0, 8),
                             ("outliers 500x, 4 canales", 500.0, 4)]:
    xb = activacion(256, factor, ncan)
    ref = xb.float() @ w_ref
    fila = f"{nombre:<28}"
    for _, fn in ESQ:
        a_i8, a_sc, sh = fn(xb)
        got = mlp_down_gemm(a_i8, b_col, a_sc, bsc, sh.contiguous(), None, torch.bfloat16).float()
        cos = torch.nn.functional.cosine_similarity(ref.flatten(), got.flatten(), dim=0).item()
        fila += f"{cos:>22.6f}"
    print(fila)

print("\n=== dependencia del batch: el MISMO token en dos batches distintos ===")
x1 = activacion(1, 100.0, 8, semilla=1)
otros = activacion(255, 100.0, 8, semilla=2)
for nombre, fn in ESQ:
    a1, s1, h1 = fn(x1)
    o1 = mlp_down_gemm(a1, b_col, s1, bsc, h1.contiguous(), None, torch.bfloat16).float()
    xb = torch.cat([x1, otros], 0)
    a2, s2, h2 = fn(xb)
    o2 = mlp_down_gemm(a2, b_col, s2, bsc, h2.contiguous(), None, torch.bfloat16)[:1].float()
    d = (o1 - o2).abs().max().item() / o1.abs().max().item()
    print(f"  {nombre:<22} dif relativa del token 0: {d:.3e}  {'IDEM' if d < 1e-6 else '*** CAMBIA ***'}")

print("\n=== costo del quant (us, mediana de 30) ===")
for M in (16, 512, 4096):
    xb = activacion(M, 100.0, 8)
    fila = f"  M={M:<6}"
    for nombre, fn in ESQ:
        for _ in range(5):
            fn(xb)
        torch.cuda.synchronize()
        ts = []
        for _ in range(30):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record(); fn(xb); e1.record()
            torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1) * 1000)
        fila += f"  {nombre}={sorted(ts)[15]:8.1f}"
    print(fila)
