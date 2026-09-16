"""int4 KV: barrido del factor de RECORTE de la escala (clipping), que no cuesta ni un byte.

La escala de cada grupo/token sale hoy del maximo absoluto. Con 4 bits conviene recortar: la
escala = c * max (c < 1) satura los outliers pero baja el paso de cuantizacion del resto.
Se barre c para K (grupos de 64 dims) y para V (por token-cabeza), y se combina con la version
asimetrica (recorte sobre min y max).
"""
import math, sys, torch
import sk18_a0bis_ptx as A
dev = "cuda"; D = 256; G = 6; HKV = 2
torch.manual_seed(0)
H = A.R.to(dev).float()
rot = lambda x: x @ H.T
rnd = torch.round


def k_sim(K, c=1.0):
    Kg = K.reshape(-1, 4, 64)
    mg = Kg.abs().amax(-1, keepdim=True) * c
    mt = mg.amax(1, keepdim=True)
    r = torch.ceil(mg / mt.clamp_min(1e-9) * 255).clamp(1, 255)
    s = (mt * r / 255 / 7).clamp_min(1e-9)
    return (rnd(Kg / s).clamp(-7, 7) * s).reshape(K.shape)


def k_asim(K, c=1.0):
    Kg = K.reshape(-1, 4, 64)
    mu = Kg.mean(-1, keepdim=True)
    lo = mu + (Kg.amin(-1, keepdim=True) - mu) * c
    hi = mu + (Kg.amax(-1, keepdim=True) - mu) * c
    s = ((hi - lo) / 15).clamp_min(1e-9)
    smax = s.amax(1, keepdim=True)
    esc = 2 ** torch.ceil(torch.log2(smax))
    smax = rnd(smax / esc * 4095) / 4095 * esc
    r = torch.ceil(s / smax.clamp_min(1e-9) * 16).clamp(1, 16)
    s = smax * r / 16
    zp = rnd(-lo / s - 8).clamp(-8, 7)
    return ((rnd(Kg / s + zp).clamp(-8, 7) - zp) * s).reshape(K.shape)


def v_sim(V, c=1.0):
    s = (V.abs().amax(-1, keepdim=True) * c / 7).clamp_min(1e-9)
    return rnd(V / s).clamp(-7, 7) * s


def v_asim(V, c=1.0):
    mu = V.mean(-1, keepdim=True)
    lo = mu + (V.amin(-1, keepdim=True) - mu) * c
    hi = mu + (V.amax(-1, keepdim=True) - mu) * c
    s = ((hi - lo) / 15).clamp_min(1e-9)
    zp = rnd(-lo / s - 8).clamp(-8, 7)
    return (rnd(V / s + zp).clamp(-8, 7) - zp) * s


def atn(q, k, v, lim):
    lg = (q @ k.transpose(-1, -2)) / math.sqrt(D)
    mask = torch.arange(k.shape[-2], device=dev)[None] > lim[:, None]
    return lg.masked_fill(mask, float("-inf")).softmax(-1) @ v


CS = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
for capa in [int(c) for c in (sys.argv[1:] or ["3", "35"])]:
    q, k, v = A.cargar(capa); Nt = k.shape[0]
    posl = [torch.arange(p, p + 4, device=dev) for p in torch.linspace(8000, Nt - 5, 8).long().tolist()]
    res = {}
    for h in range(HKV):
        K = rot(k[:, h].float()); V = rot(v[:, h].float())
        pruebas = ([(f"sim  cK={c:.2f} cV=1.00", k_sim(K, c), v_sim(V, 1.0)) for c in CS]
                   + [(f"sim  cK=mejor cV={c:.2f}", k_sim(K, 0.9), v_sim(V, c)) for c in CS]
                   + [(f"asim cK={c:.2f} cV={c:.2f}", k_asim(K, c), v_asim(V, c)) for c in CS])
        for nombre, Kq, Vq in pruebas:
            e = []
            for pos in posl:
                N = int(pos.max()) + 1
                Q = rot(q[pos][:, h * G:(h + 1) * G].reshape(-1, D).float())
                lim = pos.repeat_interleave(G)
                r = A.referencia(q, k, v, pos).float()[:, h * G:(h + 1) * G].reshape(-1, D)
                o = atn(Q, Kq[:N], Vq[:N], lim) @ H
                e.append(((o - r).norm(dim=-1) / r.norm(dim=-1).clamp_min(1e-6)).mean().item())
            res.setdefault(nombre, []).append(sum(e) / len(e))
    print(f"\ncapa {capa}")
    for kx, vx in res.items():
        print(f"  {kx:28s} {100 * sum(vx) / len(vx):6.3f}%", flush=True)
