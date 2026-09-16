"""int4 KV: cuanto baja el error si una parte del contexto se guarda EXACTA (fp16).

Idea: la atencion se concentra en los sinks (primeros tokens) y en la ventana reciente. Si el
error viene de ahi, conviene un camino hibrido (ventana/sinks en int8 aparte, el resto int4).
Mide el error de salida con int4 asimetrico (el formato que corre) dejando exacto:
  - nada, los primeros S tokens, los ultimos W, o las dos cosas.
"""
import math, sys, torch
import sk18_a0bis_ptx as A
dev = "cuda"; D = 256; G = 6; HKV = 2
torch.manual_seed(0)
H = A.R.to(dev).float()
rot = lambda x: x @ H.T
rnd = torch.round
CLIP = 243 / 256


def k_asim(K):
    Kg = K.reshape(-1, 4, 64)
    mu = Kg.mean(-1, keepdim=True)
    lo = mu + (Kg.amin(-1, keepdim=True) - mu) * CLIP
    hi = mu + (Kg.amax(-1, keepdim=True) - mu) * CLIP
    s = ((hi - lo) / 15).clamp_min(1e-9)
    smax = s.amax(1, keepdim=True)
    esc = 2 ** torch.ceil(torch.log2(smax))
    smax = rnd(smax / esc * 4095) / 4095 * esc
    r = torch.ceil(s / smax.clamp_min(1e-9) * 16).clamp(1, 16)
    s = smax * r / 16
    zp = rnd(-lo / s - 8).clamp(-8, 7)
    return ((rnd(Kg / s + zp).clamp(-8, 7) - zp) * s).reshape(K.shape)


def v_asim(V):
    mu = V.mean(-1, keepdim=True)
    lo = mu + (V.amin(-1, keepdim=True) - mu) * CLIP
    hi = mu + (V.amax(-1, keepdim=True) - mu) * CLIP
    s = ((hi - lo) / 15).clamp_min(1e-9)
    zp = rnd(-lo / s - 8).clamp(-8, 7)
    return (rnd(V / s + zp).clamp(-8, 7) - zp) * s


def atn(q, k, v, lim, devolver_pesos=False):
    lg = (q @ k.transpose(-1, -2)) / math.sqrt(D)
    mask = torch.arange(k.shape[-2], device=dev)[None] > lim[:, None]
    w = lg.masked_fill(mask, float("-inf")).softmax(-1)
    return (w @ v, w) if devolver_pesos else w @ v


CASOS = [("int4 asim (lo que corre)", 0, 0), ("+ ultimos 832 exactos", 0, 832),
         ("+ ultimos 2048 exactos", 0, 2048), ("+ primeros 64 exactos", 64, 0),
         ("+ primeros 64 y ultimos 832", 64, 832), ("+ primeros 64 y ultimos 2048", 64, 2048),
         ("todo exacto (piso)", 10 ** 9, 0)]

for capa in [int(c) for c in (sys.argv[1:] or ["3", "35"])]:
    q, k, v = A.cargar(capa); Nt = k.shape[0]
    posl = [torch.arange(p, p + 4, device=dev) for p in torch.linspace(8000, Nt - 5, 8).long().tolist()]
    res = {}
    masa = []
    for h in range(HKV):
        K = rot(k[:, h].float()); V = rot(v[:, h].float())
        Kq0 = k_asim(K); Vq0 = v_asim(V)
        for nombre, S, W in CASOS:
            e = []
            for pos in posl:
                N = int(pos.max()) + 1
                Kq = Kq0[:N].clone(); Vq = Vq0[:N].clone()
                if S:
                    ns = min(S, N); Kq[:ns] = K[:ns]; Vq[:ns] = V[:ns]
                if W:
                    Kq[max(0, N - W):] = K[max(0, N - W):N]; Vq[max(0, N - W):] = V[max(0, N - W):N]
                Q = rot(q[pos][:, h * G:(h + 1) * G].reshape(-1, D).float())
                lim = pos.repeat_interleave(G)
                r = A.referencia(q, k, v, pos).float()[:, h * G:(h + 1) * G].reshape(-1, D)
                o = atn(Q, Kq, Vq, lim) @ H
                e.append(((o - r).norm(dim=-1) / r.norm(dim=-1).clamp_min(1e-6)).mean().item())
            res.setdefault(nombre, []).append(sum(e) / len(e))
        # masa de atencion en sinks y ventana (con KV exacta)
        pos = posl[-1]; N = int(pos.max()) + 1
        Q = rot(q[pos][:, h * G:(h + 1) * G].reshape(-1, D).float())
        _, w = atn(Q, K[:N], V[:N], pos.repeat_interleave(G), True)
        masa.append((w[:, :64].sum(-1).mean().item(), w[:, -832:].sum(-1).mean().item(),
                     w[:, -2048:].sum(-1).mean().item()))
    print(f"\ncapa {capa}  masa de atencion: primeros 64 = {100*sum(m[0] for m in masa)/2:.1f}%, "
          f"ultimos 832 = {100*sum(m[1] for m in masa)/2:.1f}%, ultimos 2048 = {100*sum(m[2] for m in masa)/2:.1f}%")
    for kx, vx in res.items():
        print(f"  {kx:32s} {100 * sum(vx) / len(vx):6.3f}%", flush=True)
