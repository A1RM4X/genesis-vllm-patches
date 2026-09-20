#!/usr/bin/env python3
"""Error de la SALIDA de la atencion por esquema de cuantizacion de la KV.

Responde dos cosas que estaban sin medir:

  1. ¿La rotacion Hadamard mejora el int8, o no? El estudio viejo comparaba
     Hadamard contra WUSH —las dos rotadas— asi que el "sin rotar" nunca se midio.
  2. ¿Las escalas POR GRUPO a lo largo de la cabeza rescatan al int4? Todo lo
     medido antes usaba escalas por token-cabeza.

Se mide el error de la salida de la atencion contra float, no el error del tensor:
lo que importa es lo que llega a la capa siguiente. Hadamard es ortogonal, asi que
en float es exacta y toda diferencia es de cuantizacion.
"""
import sys, torch

# Volcados de PN123. Para generarlos: GENESIS_ENABLE_PN123_VOLCADO_QKV=1 con
# GENESIS_PN123_CAPAS=3,35, y mandar >=4 prefills de mas de GENESIS_PN123_M_MIN
# tokens. Usar CODIGO, no prosa: el modo de falla de int4 aparece ahi.
import os
DIR = os.environ.get("PN123_DIR", "/home/usuario/Proyectos/kv-offload/pn123_qkv")
D, HKV, HQ = 256, 2, 12
G = HQ // HKV                      # GQA: 6 cabezas de query por cabeza KV
NQ = 192                           # consultas evaluadas, las ultimas (ven casi todo el contexto)
torch.manual_seed(0)


def cargar(capa, idxs=(1, 2, 3)):
    qs, ks, vs = [], [], []
    for i in idxs:
        d = torch.load(f"{DIR}/capa{capa:02d}_{i}.pt", map_location="cpu")
        qs.append(d["q"]); ks.append(d["k"]); vs.append(d["v"])
    q = torch.cat([x.reshape(-1, HQ, D) for x in qs]).float()
    k = torch.cat([x.reshape(-1, HKV, D) for x in ks]).float()
    v = torch.cat([x.reshape(-1, HKV, D) for x in vs]).float()
    return q, k, v


def hadamard(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / (n ** 0.5)


def q_int(x, bits, grupo):
    """Cuantiza simetrico. grupo=None -> una escala por (token, cabeza).
    grupo=g -> una escala por (token, cabeza, bloque de g dims a lo largo de la cabeza)."""
    qmax = 2 ** (bits - 1) - 1
    if grupo is None:
        s = x.abs().amax(-1, keepdim=True) / qmax
    else:
        f = x.unflatten(-1, (D // grupo, grupo))
        s = f.abs().amax(-1, keepdim=True) / qmax
        s = s.expand_as(f).reshape(x.shape)
    s = s.clamp_min(1e-12)
    return torch.round(x / s).clamp(-qmax - 1, qmax) * s


def q_fp8(x):
    """e4m3 con escala por tensor, que es lo que corre hoy en el camino fp8."""
    return x.to(torch.float8_e4m3fn).float()


def atencion(q, k, v, pos):
    """Salida de la atencion para las consultas en `pos`, causal, en float32."""
    out = torch.empty(len(pos), HQ, D)
    esc = D ** -0.5
    for h in range(HQ):
        kh, vh = k[:, h // G], v[:, h // G]
        for j, p in enumerate(pos):
            s = (kh[: p + 1] @ q[p, h]) * esc
            out[j, h] = torch.softmax(s, 0) @ vh[: p + 1]
    return out


def err(a, b):
    return (100 * (a - b).norm() / b.norm()).item()


H = hadamard(D)
print(f"  {'esquema':38s} {'bits':>5s}  capa 3   capa 35")
filas = {}
for capa in (3, 35):
    q, k, v = cargar(capa)
    N = q.shape[0]
    pos = list(range(N - NQ, N))
    ref = atencion(q, k, v, pos)

    qr, kr, vr = q @ H, k @ H, v @ H          # exacto en float; el error es solo de cuantizar

    casos = [
        ("fp8_e4m3 (escala por tensor)", 8.0, lambda: (q, q_fp8(k), q_fp8(v), False)),
        ("int8 por token-cabeza", 8.1, lambda: (q, q_int(k, 8, None), q_int(v, 8, None), False)),
        ("int8 por token-cabeza + Hadamard", 8.1, lambda: (qr, q_int(kr, 8, None), q_int(vr, 8, None), True)),
        ("int4 por token-cabeza", 4.1, lambda: (q, q_int(k, 4, None), q_int(v, 4, None), False)),
        ("int4 por token-cabeza + Hadamard", 4.1, lambda: (qr, q_int(kr, 4, None), q_int(vr, 4, None), True)),
        ("int4 grupo 64", 4.25, lambda: (q, q_int(k, 4, 64), q_int(v, 4, 64), False)),
        ("int4 grupo 64 + Hadamard", 4.25, lambda: (qr, q_int(kr, 4, 64), q_int(vr, 4, 64), True)),
        ("int4 grupo 32", 4.5, lambda: (q, q_int(k, 4, 32), q_int(v, 4, 32), False)),
        ("int4 grupo 32 + Hadamard", 4.5, lambda: (qr, q_int(kr, 4, 32), q_int(vr, 4, 32), True)),
        ("int4 grupo 16 + Hadamard", 5.0, lambda: (qr, q_int(kr, 4, 16), q_int(vr, 4, 16), True)),
    ]
    for nombre, bits, f in casos:
        qq, kk, vv, rotado = f()
        o = atencion(qq, kk, vv, pos)
        if rotado:
            o = o @ H.T                        # deshacer la rotacion de V: la salida es lineal en V
        filas.setdefault((nombre, bits), {})[capa] = err(o, ref)

for (nombre, bits), d in filas.items():
    print(f"  {nombre:38s} {bits:5.2f}  {d[3]:6.2f}%  {d[35]:6.2f}%")
