"""Numerica int4 para SK-18i con el presupuesto de la reserva int4_per_token_head de vLLM
(264 B/token-cabeza: 128 K + 128 V + 8 B de escalas). Q, K, V en int4 (mma s4.s4), q/k con
Hadamard. Variantes de escala y de escala cuantizada a 1 byte (log 2^(e/8), redondeo hacia
arriba). Softmax float y pesos de 12/16 bits para aislar. Error de la salida vs float."""
import math, sys, torch
import sk18_a0bis_ptx as A
dev = "cuda"; D = 256; G = 6

def cuant(x, bits, grupo, log8=False):
    m = (1 << (bits - 1)) - 1
    T = x.shape[0]
    xg = x.reshape(T, -1, grupo)
    s = xg.abs().amax(-1, keepdim=True).clamp_min(1e-8) / m
    if log8:
        s = torch.pow(2.0, torch.ceil(torch.log2(s) * 8) / 8)
    return ((xg / s).round().clamp(-m, m) * s).reshape(T, D)

def atn(qx, kx, vx, lim, wbits=None):
    lg = (qx @ kx.t()) / math.sqrt(D)
    lg = lg.masked_fill(torch.arange(kx.shape[0], device=dev)[None] > lim[:, None], float("-inf"))
    w = lg.softmax(-1)
    if wbits:
        mx = w.amax(-1, keepdim=True)
        w = torch.round(w / mx * ((1 << wbits) - 1)) / ((1 << wbits) - 1) * mx
        w = w / w.sum(-1, keepdim=True)
    return w @ vx

for capa in [int(c) for c in (sys.argv[1:] or ["3", "35"])]:
    q, k, v = A.cargar(capa); Nt = k.shape[0]
    kr = A.rotar_ptx(k)
    Rm = A.R.to(dev).double()
    acum = {}
    for p0 in torch.linspace(8000, Nt - 5, 8).long().tolist():
        pos = torch.arange(p0, p0 + 4, device=dev); N = p0 + 4
        ref = A.referencia(q, k, v, pos)
        qr = A.rotar_ptx(q[pos])
        for h in range(2):
            Q = qr[:, h * G:(h + 1) * G].reshape(-1, D).double(); K = kr[:N, h].double(); V = v[:N, h].double()
            Vr = V @ Rm.t()
            lim = pos.repeat_interleave(G)
            r = ref[:, h * G:(h + 1) * G].reshape(-1, D).double()
            Km = K - K[: min(N, 7488)].mean(0, keepdim=True)          # media por canal (del primer chunk, fija)
            casos = {
                "Q8 K8 V8 tok (int8 actual)": (cuant(Q, 8, 256), cuant(K, 8, 256), cuant(V, 8, 256), False, None),
                "Q4 K4 V4 tok": (cuant(Q, 4, 256), cuant(K, 4, 256), cuant(V, 4, 256), False, None),
                "Q4g64 K4g64 V4tok": (cuant(Q, 4, 64), cuant(K, 4, 64), cuant(V, 4, 256), False, None),
                "Q4g64 K4g64 V4tok Vhad": (cuant(Q, 4, 64), cuant(K, 4, 64), cuant(Vr, 4, 256), True, None),
                "Q4g64 K4g64 V4g64 Vhad": (cuant(Q, 4, 64), cuant(K, 4, 64), cuant(Vr, 4, 64), True, None),
                "Q4g32 K4g64 V4tok Vhad": (cuant(Q, 4, 32), cuant(K, 4, 64), cuant(Vr, 4, 256), True, None),
                "Q4g64 K4g64 V4g64 Vhad log8": (cuant(Q, 4, 64), cuant(K, 4, 64, True), cuant(Vr, 4, 64, True), True, None),
                "Q4g64 K4g64 V4g64 Vhad log8 w12": (cuant(Q, 4, 64), cuant(K, 4, 64, True), cuant(Vr, 4, 64, True), True, 12),
                "Q4g64 K4g64 V4g64 Vhad log8 w8": (cuant(Q, 4, 64), cuant(K, 4, 64, True), cuant(Vr, 4, 64, True), True, 8),
                "Ksuav: Q4g64 K4g64 V4g64 Vhad": (cuant(Q, 4, 64), cuant(Km, 4, 64), cuant(Vr, 4, 64), True, None),
                "Ksuav: Q4g64 K4g64 V4g64 Vhad log8 w12": (cuant(Q, 4, 64), cuant(Km, 4, 64, True), cuant(Vr, 4, 64, True), True, 12),
                "Ksuav: Q4g32 K4g32 V4g64 Vhad": (cuant(Q, 4, 32), cuant(Km, 4, 32), cuant(Vr, 4, 64), True, None),
                "Ksuav: Q8 K4g64 V4g64 Vhad (ref)": (cuant(Q, 8, 256), cuant(Km, 4, 64), cuant(Vr, 4, 64), True, None),
                "Ksuav: Q4g64 K4g64 V8 (solo K4)": (cuant(Q, 4, 64), cuant(Km, 4, 64), cuant(V, 8, 256), False, None),
                "Ksuav: Q8 K8 V4g64 Vhad (solo V4)": (cuant(Q, 8, 256), cuant(Km, 8, 256), cuant(Vr, 4, 64), True, None),
            }
            for nom, (qq, kk, vv, rot, wb) in casos.items():
                o = atn(qq, kk, vv, lim, wb)
                if rot: o = o @ Rm
                acum.setdefault(nom, []).append(((o - r).norm(dim=-1) / r.norm(dim=-1)).mean().item())
    print(f"\ncapa {capa}")
    for kx, vx in acum.items():
        print(f"  {kx:34s} {100*sum(vx)/len(vx):6.3f}%", flush=True)
