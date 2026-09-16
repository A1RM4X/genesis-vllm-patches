"""int4 KV con TODAS las mejoras, medido offline (capas 3 y 35, volcados PN123).

Calibracion: primer tramo del volcado (tokens < 7488, consultas en ese rango).
Prueba: posiciones de decode en los tramos 2 y 3 (no vistas).
Rotacion aprendida (tipo SpinQuant): R ortogonal por cabeza KV = exp(A - A^T) * H (inicio en
Hadamard), entrenada sobre el error de la SALIDA de atencion con cuantizacion int4 simulada
(STE). Una R para q/k (misma para los dos: q.k invariante) y otra para V (se des-rota la salida).
Mejoras combinables: grupos g64/g32, Q int8, suavizado de K por canal, escala log de 1 byte,
pesos de 12 bits.
"""
import math, sys, time, torch
import sk18_a0bis_ptx as A
dev = "cuda"; D = 256; G = 6; HKV = 2
torch.manual_seed(0)
H = A.R.to(dev).float()


def ste_round(x):
    return (x.round() - x).detach() + x


def fq(x, bits, g, log8=False):
    """cuantizacion simetrica por grupos de dims (ultima dim), diferenciable por STE."""
    m = (1 << (bits - 1)) - 1
    sh = x.shape
    xg = x.reshape(*sh[:-1], -1, g)
    s = xg.detach().abs().amax(-1, keepdim=True).clamp_min(1e-8) / m
    if log8:
        s = torch.pow(2.0, torch.ceil(torch.log2(s) * 8) / 8)
    return (ste_round(xg / s).clamp(-m, m) * s).reshape(sh)


def atn(q, k, v, lim, wbits=None):
    lg = (q @ k.transpose(-1, -2)) / math.sqrt(D)
    mask = torch.arange(k.shape[-2], device=dev)[None] > lim[:, None]
    w = lg.masked_fill(mask, float("-inf")).softmax(-1)
    if wbits:
        mx = w.detach().amax(-1, keepdim=True)
        w = ste_round(w / mx * ((1 << wbits) - 1)) / ((1 << wbits) - 1) * mx
        w = w / w.sum(-1, keepdim=True)
    return w @ v


def rot(Ap):
    return torch.linalg.matrix_exp(Ap - Ap.T) @ H


def salida(cfg, Q, K, V, lim, Rqk, Rv, kmean):
    Qr = Q @ Rqk.T; Kr = K @ Rqk.T
    if cfg.get("ksuav"):
        Kr = Kr - kmean @ Rqk.T
    Vr = V @ Rv.T
    g = cfg["g"]
    qb = 8 if cfg.get("q8") else 4
    qq = fq(Qr, qb, 256 if qb == 8 else g)
    kk = fq(Kr, 4, g, cfg.get("log8", False))
    vv = fq(Vr, 4, cfg.get("gv", g), cfg.get("log8", False))
    o = atn(qq, kk, vv, lim, cfg.get("w12") and 12)
    return o @ Rv


def datos(q, k, v, h, pos):
    N = int(pos.max()) + 1
    Q = q[pos][:, h * G:(h + 1) * G].reshape(-1, D).float()
    lim = pos.repeat_interleave(G)
    return Q, k[:N, h].float(), v[:N, h].float(), lim


def error(o, r):
    return ((o - r).norm(dim=-1) / r.norm(dim=-1).clamp_min(1e-6)).mean()


CFGS = {
    "base Hadamard Q4 K4 V4 g64": dict(g=64),
    "+ K suavizada": dict(g=64, ksuav=1),
    "+ g32": dict(g=32),
    "+ Q int8": dict(g=64, q8=1),
    "todo sin rot. aprendida (g32, Q8, Ksuav)": dict(g=32, q8=1, ksuav=1),
    "todo + log8 + w12 (formato real)": dict(g=32, q8=1, ksuav=1, log8=1, w12=1),
}

for capa in [int(c) for c in (sys.argv[1:] or ["3", "35"])]:
    q, k, v = A.cargar(capa); Nt = k.shape[0]
    ref_f = lambda pos: A.referencia(q, k, v, pos).float()
    cal_pos = [torch.arange(p, p + 4, device=dev) for p in torch.linspace(1500, 7400, 24).long().tolist()]
    test_pos = [torch.arange(p, p + 4, device=dev) for p in torch.linspace(8000, Nt - 5, 8).long().tolist()]
    res = {}
    t0 = time.time()
    for h in range(HKV):
        kmean = k[:7488, h].float().mean(0, keepdim=True)
        for nombre, cfg in CFGS.items():
            for aprender in (False, True):
                Aqk = torch.zeros(D, D, device=dev, requires_grad=aprender)
                Av = torch.zeros(D, D, device=dev, requires_grad=aprender)
                if aprender:
                    opt = torch.optim.Adam([Aqk, Av], lr=2e-3)
                    for it in range(120):
                        pos = cal_pos[it % len(cal_pos)]
                        Q, K, V, lim = datos(q, k, v, h, pos)
                        r = ref_f(pos)[:, h * G:(h + 1) * G].reshape(-1, D)
                        loss = error(salida(cfg, Q, K, V, lim, rot(Aqk), rot(Av), kmean), r)
                        opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    e = []
                    for pos in test_pos:
                        Q, K, V, lim = datos(q, k, v, h, pos)
                        r = ref_f(pos)[:, h * G:(h + 1) * G].reshape(-1, D)
                        e.append(error(salida(cfg, Q, K, V, lim, rot(Aqk), rot(Av), kmean), r).item())
                clave = nombre + (" | rot. APRENDIDA" if aprender else "")
                res.setdefault(clave, []).append(sum(e) / len(e))
    print(f"\ncapa {capa} ({time.time() - t0:.0f}s)")
    for kx, vx in res.items():
        print(f"  {kx:55s} {100 * sum(vx) / len(vx):6.3f}%", flush=True)
