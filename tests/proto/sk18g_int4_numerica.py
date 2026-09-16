"""Numerica del camino int4 de SK-18 (mma s4.s4: los DOS operandos en int4).
Q.K = sum_g sq_g * sk_g * (q4 . k4)_g con grupos de G dims (64 = un fragmento k64 del mma s4),
V int4 con escala por grupo de dims, pesos de 15 bits (en 4 nibbles con signo para el mma s4).
Softmax float para aislar la cuantizacion; error de la salida de atencion vs float."""
import math, sys, torch
import sk18_a0bis_ptx as A
dev = "cuda"; D = 256; G = 6

def cuant(x, bits, grupo):
    m = (1 << (bits - 1)) - 1
    T = x.shape[0]
    xg = x.reshape(T, -1, grupo)
    s = xg.abs().amax(-1, keepdim=True).clamp_min(1e-8) / m
    return ((xg / s).round().clamp(-m, m) * s).reshape(T, D)

def atn(qx, kx, vx, lim):
    lg = (qx @ kx.t()) / math.sqrt(D)
    lg = lg.masked_fill(torch.arange(kx.shape[0], device=dev)[None] > lim[:, None], float("-inf"))
    return lg.softmax(-1) @ vx

for capa in [int(c) for c in (sys.argv[1:] or ["3", "35"])]:
    q, k, v = A.cargar(capa); Nt = k.shape[0]
    kr = A.rotar_ptx(k)
    acum = {}
    for p0 in torch.linspace(8000, Nt - 5, 8).long().tolist():
        pos = torch.arange(p0, p0 + 4, device=dev); N = p0 + 4
        ref = A.referencia(q, k, v, pos)
        qr = A.rotar_ptx(q[pos])
        for h in range(2):
            Q = qr[:, h * G:(h + 1) * G].reshape(-1, D).double(); Kx = kr[:N, h].double(); V = v[:N, h].double()
            lim = pos.repeat_interleave(G)
            r = ref[:, h * G:(h + 1) * G].reshape(-1, D).double()
            vrot = A.rotar_ptx(v[:N, h:h + 1])[:, 0].double()
            Rm = A.R.to(dev)
            casos = {
                "Q8 K8 V8 (por token)": (cuant(Q, 8, 256), cuant(Kx, 8, 256), cuant(V, 8, 256), False),
                "Q4 K4 V4 (por token)": (cuant(Q, 4, 256), cuant(Kx, 4, 256), cuant(V, 4, 256), False),
                "Q4 K4 V4 g64": (cuant(Q, 4, 64), cuant(Kx, 4, 64), cuant(V, 4, 64), False),
                "Q4 K4 V4 g32": (cuant(Q, 4, 32), cuant(Kx, 4, 32), cuant(V, 4, 32), False),
                "Q4 K4 V4 g16": (cuant(Q, 4, 16), cuant(Kx, 4, 16), cuant(V, 4, 16), False),
                "Q8 K4 V4 g32": (cuant(Q, 8, 256), cuant(Kx, 4, 32), cuant(V, 4, 32), False),
                "Q4 K4 g32, V4 g32 rotada": (cuant(Q, 4, 32), cuant(Kx, 4, 32), cuant(vrot, 4, 32), True),
                "Q4 K4 g16, V4 g16 rotada": (cuant(Q, 4, 16), cuant(Kx, 4, 16), cuant(vrot, 4, 16), True),
            }
            for nom, (qq, kk, vv, rot) in casos.items():
                o = atn(qq, kk, vv, lim)
                if rot: o = o @ Rm      # des-rotar (V rotada con la misma H)
                acum.setdefault(nom, []).append(((o - r).norm(dim=-1) / r.norm(dim=-1)).mean().item())
    print(f"\ncapa {capa}")
    for kx, vx in acum.items():
        print(f"  {kx:28s} {100*sum(vx)/len(vx):6.3f}%", flush=True)
