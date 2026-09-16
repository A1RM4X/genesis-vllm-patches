"""SK-18e numerica: softmax ENTERO en streaming (una pasada, sin tabla), emulado en int64.
Por cabeza KV, keys en tiles de T en orden; por query:
  z' = ((q8.k8 >> 8) * skf) >> 11  (como SK-18d), escala mq de SK-18d
  t  = distancia al maximo CORRIENTE en medios-logs (Q8): t = (d * mqb + 2^15) >> 16
  w  = 2^-t aproximado: n = t >> 8, f = t & 255, g(f) en Q15 (lineal o cuadratica),
       w = redondeo((g * 32639 >> 15) >> n)  (15 bits relativo al maximo)
  o += w' * v8 con w' = (w * svf + 2^14) >> 15 ; S += w
  si el maximo sube Delta: o, S *= 2^-Delta (misma aproximacion, Q15, redondeo)
Salida o * svmax / S. Error vs atencion float (capas 3/35, decode: 4 posiciones x 6 cabezas)."""
import math, torch
import sk18_a0bis_ptx as A
dev = "cuda"; D = 256; G = 6

def g_q15(f, modo):
    if modo == "lineal":
        return 32768 - ((f * 16384 + 128) >> 8)
    if modo == "cuad":   # 3 puntos 0, .5, 1: 1 - .6716x + .1716x^2
        return 32768 - ((f * 22007 + 128) >> 8) + ((f * f * 5623 + 32768) >> 16)
    if modo == "cuad2":  # minimos cuadrados en [0,1): 1 - .68475x + .18475x^2 ajustado abajo
        return 32768 - ((f * CA + 128) >> 8) + ((f * f * CB + 32768) >> 16)
    raise ValueError

# ajuste por minimos cuadrados con extremos fijos (g(0)=1, g(1)=.5) -> un grado de libertad b
x = torch.linspace(0, 1, 1000, dtype=torch.float64)
bs = torch.linspace(0.1, 0.25, 3001, dtype=torch.float64)
err = torch.stack([((1 - (0.5 + b) * x + b * x * x) / 2 ** (-x) - 1).abs().max() for b in bs])
bopt = bs[err.argmin()].item(); CA = round((0.5 + bopt) * 32768); CB = round(bopt * 32768)
for modo, (a_, b_) in (("lineal", (.5, 0)), ("cuad", (.6716, .1716)), ("cuad2", (.5 + bopt, bopt))):
    e = ((1 - a_ * x + b_ * x * x) / 2 ** (-x) - 1).abs().max().item()
    print(f"g {modo}: error relativo maximo del peso {100*e:.3f}%")

def pesos(d, mqb, modo):
    t = (d * mqb[:, None] + 32768) >> 16
    n = (t >> 8).clamp(max=40); f = t & 255
    w0 = (g_q15(f, modo) * 32639) >> 15
    r = torch.where(n > 0, (w0 + (1 << (n - 1).clamp(min=0))) >> n, w0)
    return torch.where(n >= 16, torch.zeros_like(r), r)

def streaming(zp, mq, svf, V8, T, modo, rescale=True):
    """zp int64 [M,N] (PAD = muy negativo), mq [M]; svf [N] Q15, V8 [N,D]."""
    M, N = zp.shape
    mqb = torch.round(mq.double() * 16 * 256 / (1023 * math.log(2))).to(torch.int64)
    m = torch.full((M,), -(1 << 40), dtype=torch.int64, device=dev)
    o = torch.zeros(M, D, dtype=torch.int64, device=dev); S = torch.zeros(M, dtype=torch.int64, device=dev)
    nres = 0
    for k0 in range(0, N, T):
        z = zp[:, k0:k0 + T]
        mn = torch.maximum(m, z.amax(1))
        delta = mn - m
        sube = (delta > 0) & (m > -(1 << 39))
        if rescale and bool(sube.any()):
            nres += int(sube.sum())
            t = (torch.where(sube, delta, torch.zeros_like(delta)) * mqb + 32768) >> 16
            n = (t >> 8).clamp(max=40); f = t & 255
            g = g_q15(f, modo)                                   # Q15, 32768 = 1
            sc = torch.where(n > 0, (g + (1 << (n - 1).clamp(min=0))) >> n, g)
            sc = torch.where(n >= 31, torch.zeros_like(sc), sc)
            o = (o * sc[:, None] + 16384) >> 15
            S = (S * sc + 16384) >> 15
        m = mn
        w = pesos((m[:, None] - z).clamp(min=0), mqb, modo)
        w = torch.where(z < -(1 << 27), torch.zeros_like(w), w)
        wp = (w * svf[None, k0:k0 + T] + 16384) >> 15
        o += torch.round(wp.double() @ V8[k0:k0 + T].double()).long()
        S += w.sum(1)
    return o, S, nres / M


def referencia_fija(zp, mq, svf, V8, lim, H, S0=256, R0=256):
    M, N = zp.shape
    mqb = torch.round(mq.double() * 16 * 256 / (1023 * math.log(2))).to(torch.int64)
    idx = torch.arange(N, device=dev)[None]
    vis = (idx < S0) | ((idx > lim[:, None] - R0) & (idx <= lim[:, None]))
    m0 = torch.where(vis, zp, torch.full_like(zp, -(1 << 40))).amax(1)
    # H medios-logs en unidades de z': t = d*mqb>>16 en Q8 -> d = H*256*65536/mqb
    zref = m0 + (H * 256 * 65536) // mqb
    exced = (zp.amax(1) > zref).double().mean().item()
    d = (zref[:, None] - zp).clamp(min=0)
    w = pesos(d, mqb, "cuad2")
    w = torch.where(zp < -(1 << 27), torch.zeros_like(w), w)
    wp = (w * svf[None] + 16384) >> 15
    return torch.round(wp.double() @ V8.double()).long(), w.sum(1), exced

def denso_lut(zp, mq, svf, V8):
    zm = zp.amax(1)
    idx = (((zm[:, None] - zp) * mq[:, None] + 32768) >> 16).clamp(0, 1023)
    w = A.LUT.to(torch.int64)[idx]; w = torch.where(zp < -(1 << 27), torch.zeros_like(w), w)
    wp = (w * svf[None] + 16384) >> 15
    return torch.round(wp.double() @ V8.double()).long(), w.sum(1)

import sys
for capa in [int(c) for c in (sys.argv[1:] or ["3", "35"])]:
    q, k, v = A.cargar(capa); Nt = k.shape[0]
    kr = A.rotar_ptx(k)
    acum = {}
    for p0 in torch.linspace(8000, Nt - 5, 8).long().tolist():
        pos = torch.arange(p0, p0 + 4, device=dev); N = p0 + 4
        ref = A.referencia(q, k, v, pos)
        qr = A.rotar_ptx(q[pos])
        for h in range(2):
            Q8, sq = A.q8_ptx(qr[:, h * G:(h + 1) * G].reshape(-1, 1, D)); K8, sk = A.q8_ptx(kr[:N, h:h + 1]); V8, sv = A.q8_ptx(v[:N, h:h + 1])
            Q8, sq, K8, sk, V8, sv = Q8[:, 0].long(), sq[:, 0].double(), K8[:, 0].long(), sk[:, 0].double(), V8[:, 0].long(), sv[:, 0].double()
            z = torch.round(Q8.double() @ K8.t().double()).long()
            skf = torch.round(sk / sk.max() * 32767).long()
            zp = ((z >> 8) * skf[None]) >> 11
            lim = pos.repeat_interleave(G)
            zp = torch.where(torch.arange(N, device=dev)[None] > lim[:, None], torch.full_like(zp, -(1 << 28)), zp)
            mq = torch.round(sk.max() * sq * 1023 / 16.0 * 65536).clamp(1, (1 << 31) - 1).long()
            svf = torch.round(sv / sv.max() * 32767).long()
            r = ref[:, h * G:(h + 1) * G].reshape(-1, D).double()
            def err(o, S):
                out = o.double() * sv.max() / S.double().clamp_min(1)[:, None]
                return ((out - r).norm(dim=-1) / r.norm(dim=-1)).mean().item()
            acum.setdefault("denso tabla u24", []).append(err(*denso_lut(zp, mq, svf, V8)))
            o, S, nr = streaming(zp, mq, svf, V8, 256, "cuad2")
            acum.setdefault("streaming cuad2 T256", []).append(err(o, S))
            for H in (0, 1, 2, 3, 4):
                o, S, ex = referencia_fija(zp, mq, svf, V8, lim, H)
                acum.setdefault(f"ref fija H{H} (sink256+rec256)", []).append(err(o, S))
                acum.setdefault(f"  (queries con max > ref, H{H})", []).append(ex)
    print(f"\ncapa {capa}")
    for kx, vx in acum.items():
        val = sum(vx) / len(vx)
        print(f"  {kx:34s} {'%.3f%%' % (100*val) if 'reesc' not in kx else '%.1f%%' % (100*val)}", flush=True)
