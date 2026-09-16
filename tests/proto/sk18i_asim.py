"""int4 KV: cuanto baja el error si la cuantizacion es ASIMETRICA (con cero por grupo).

Mide el error relativo de la salida de atencion (softmax float, q en fp16, como el camino de
prefill de PN131) para la configuracion que ya corre (K simetrica g64, V simetrica por token)
y para variantes con cero por grupo / otras granularidades. Sirve para decidir si vale la pena
gastar los bytes de escala que quedan en el bloque (8 por token-cabeza).
"""
import math, sys, torch
import sk18_a0bis_ptx as A
dev = "cuda"; D = 256; G = 6; HKV = 2
torch.manual_seed(0)
H = A.R.to(dev).float()


def rot(x):        # A.R ya viene normalizada (H / 16, ortonormal)
    return x @ H.T


def qsim(x, bits, g):
    m = (1 << (bits - 1)) - 1
    sh = x.shape
    xg = x.reshape(*sh[:-1], -1, g)
    s = xg.abs().amax(-1, keepdim=True).clamp_min(1e-8) / m
    return (torch.round(xg / s).clamp(-m, m) * s).reshape(sh)


def qasim(x, bits, g):
    n = (1 << bits) - 1
    sh = x.shape
    xg = x.reshape(*sh[:-1], -1, g)
    lo = xg.amin(-1, keepdim=True); hi = xg.amax(-1, keepdim=True)
    s = ((hi - lo) / n).clamp_min(1e-8)
    q = torch.round((xg - lo) / s).clamp(0, n)
    return (q * s + lo).reshape(sh)


def atn(q, k, v, lim):
    lg = (q @ k.transpose(-1, -2)) / math.sqrt(D)
    mask = torch.arange(k.shape[-2], device=dev)[None] > lim[:, None]
    return lg.masked_fill(mask, float("-inf")).softmax(-1) @ v


CFGS = {
    "K sim g64 | V sim g256 (lo que corre)": (("sim", 64), ("sim", 256)),
    "K asim g64 | V sim g256":               (("asim", 64), ("sim", 256)),
    "K asim g64 | V asim g256":              (("asim", 64), ("asim", 256)),
    "K sim g32 | V sim g256":                (("sim", 32), ("sim", 256)),
    "K asim g32 | V asim g256":              (("asim", 32), ("asim", 256)),
    "K sim g64 | V sim g64":                 (("sim", 64), ("sim", 64)),
}
F = {"sim": qsim, "asim": qasim}

for capa in [int(c) for c in (sys.argv[1:] or ["3", "35"])]:
    q, k, v = A.cargar(capa); Nt = k.shape[0]
    posl = [torch.arange(p, p + 4, device=dev) for p in torch.linspace(8000, Nt - 5, 8).long().tolist()]
    res = {}
    for h in range(HKV):
        K = rot(k[:, h].float()); V = rot(v[:, h].float())
        for nombre, ((fk, gk), (fv, gv)) in CFGS.items():
            Kq = F[fk](K, 4, gk); Vq = F[fv](V, 4, gv)
            e = []
            for pos in posl:
                N = int(pos.max()) + 1
                Q = rot(q[pos][:, h * G:(h + 1) * G].reshape(-1, D).float())
                lim = pos.repeat_interleave(G)
                r = A.referencia(q, k, v, pos).float()[:, h * G:(h + 1) * G].reshape(-1, D)
                o = atn(Q, Kq[:N], Vq[:N], lim) @ H
                e.append((((o - r).norm(dim=-1)) / r.norm(dim=-1).clamp_min(1e-6)).mean().item())
            res.setdefault(nombre, []).append(sum(e) / len(e))
    print(f"\ncapa {capa}")
    for kx, vx in res.items():
        print(f"  {kx:40s} {100 * sum(vx) / len(vx):6.3f}%", flush=True)
