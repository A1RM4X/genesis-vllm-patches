"""SK-18d + salto de bloques. La seleccion sale del Q.K EXACTO que SK-18d/1 ya calcula
(max por tramo, sin costo extra): el escaneo int4 cuesta lo mismo que el Q.K s8
(0,038 vs 0,036 ms @57k), asi que en esta cadena solo agregaria trabajo.
Criterio: tramo elegido si zmax - max_tramo < delta (en logits) para ALGUNA query de
la cabeza (union de las 24); siempre el primero (sink) y el ultimo.
Parte 1: error vs atencion float (capas 3/35, decode real: 4 posiciones x 6 cabezas Q).
Parte 2: tiempo por capa con keys reales repetidas a 57k/100k."""
import math, sys, time, torch
import sk18_a0bis_ptx as A
from vllm._genesis.kernels.ptx_lab import Kernel
dev = "cuda"; D = 256; G = 6; PAD = -(1 << 28)
LUT = A.LUT
QK_DEFS = []  # defaults de sk18d_qk: BM32
kqk = Kernel("sk18d_qk.cu", "sk18d_qk", warps=8); SH_Q = 2 * (32 * 128 + 64 * 256)
kwv = Kernel("sk18d_wv.cu", "sk18d_wv", warps=8); SH_W = 2 * (2 * 32 * 128 + 64 * 256)
NTH = 256

def preparar(qr, kr, v, lim, CH):
    """qr [M,256] float rotada, kr/v [N,256]; lim int32 [M]."""
    M, N = qr.shape[0], kr.shape[0]
    Q8, sq = A.q8_ptx(qr[:, None]); K8, sk = A.q8_ptx(kr[:, None]); V8, sv = A.q8_ptx(v[:, None])
    Q8, sq, K8, sk, V8, sv = Q8[:, 0].contiguous(), sq[:, 0].float(), K8[:, 0].contiguous(), sk[:, 0].float(), V8[:, 0], sv[:, 0].float()
    K = ((N + CH - 1) // CH) * CH
    skf = torch.round(sk / sk.max() * 32767).to(torch.int16).contiguous()
    alpha = 256.0 * sk.max() * sq / 16.0 / 16.0
    mq = torch.round(alpha * 1023 / 16.0 * 65536).clamp(1, (1 << 31) - 1).to(torch.int32)
    dcap = ((1023 * 65536 - 32768 + mq.to(torch.int64) - 1) // mq.to(torch.int64)).clamp(max=1 << 30).to(torch.int32)
    Vt = torch.zeros(D, K, dtype=torch.int8, device=dev); Vt[:, :N] = V8.t()
    svf = torch.zeros(K, dtype=torch.int16, device=dev); svf[:N] = torch.round(sv / sv.max() * 32767).to(torch.int16)
    NCH = K // CH; tiles = (M + 31) // 32
    return dict(M=M, N=N, K=K, CH=CH, NCH=NCH, Q8=Q8, K8=K8, skf=skf, lim=lim.to(torch.int32).contiguous(), mq=mq, dcap=dcap,
                Vt=Vt.contiguous(), svf=svf, svmax=sv.max(),
                Sz=torch.empty(M, K, dtype=torch.int32, device=dev),
                out=torch.empty(NCH, M, D, dtype=torch.int32, device=dev), lo=torch.empty(NCH, M, D, dtype=torch.int32, device=dev),
                sumw=torch.empty(NCH * tiles * NTH, dtype=torch.int64, device=dev),
                arange=torch.arange(NCH, dtype=torch.int32, device=dev), ncht=torch.full((1,), NCH, dtype=torch.int32, device=dev))

def cadena(p, delta):
    M, N, K, CH, NCH = p["M"], p["N"], p["K"], p["CH"], p["NCH"]
    Sz = p["Sz"]; Sz.fill_(PAD)
    kqk.lanzar(((N + 63) // 64, (M + 31) // 32), [p["Q8"], p["K8"], p["skf"], p["lim"], Sz, M, N, 256, K], shared=SH_Q)
    zm = Sz.amax(1)
    if delta is None:
        tabla, nsel = p["arange"], p["ncht"]
    else:
        bmax = Sz.view(M, NCH, CH).amax(2)
        dsel = (p["dcap"].to(torch.int64) * delta) >> 4
        keep = ((zm[:, None].to(torch.int64) - bmax) < dsel[:, None]).any(0)
        keep[0] = True; keep[-1] = True
        tabla = torch.argsort(keep.to(torch.int8), descending=True, stable=True).to(torch.int32)
        nsel = keep.sum().to(torch.int32).view(1)
    p["out"].zero_(); p["lo"].zero_(); p["sumw"].zero_()
    kwv.lanzar(((D + 63) // 64, ((M + 31) // 32) * NCH), [Sz, zm, p["mq"], p["dcap"], LUT, p["svf"], p["Vt"], p["out"], p["lo"],
                                                         tabla, nsel, p["sumw"], M, D, K, CH, NCH], shared=SH_W)
    oi = p["out"].sum(0, dtype=torch.int32).to(torch.int64) * 256 + p["lo"].sum(0, dtype=torch.int32)
    sw = p["sumw"].view(NCH, -1, 32, 8).sum((0, 3)).reshape(-1)[:M]
    return oi, sw, nsel

def salida(p, oi, sw):
    return oi.double() * p["svmax"].double() / sw.double()[:, None].clamp_min(1)

DELTAS = (None, 10, 7, 5)
print("== Parte 1: error (48 puntos de decode por capa x 2 cabezas KV)")
for capa in (3, 35):
    q, k, v = A.cargar(capa)
    Nt = k.shape[0]
    kr = A.rotar_ptx(k)
    for CH in (128, 512):
        acum = {d: [[], []] for d in DELTAS}
        for p0 in torch.linspace(8000, Nt - 5, 12).long().tolist():
            pos = torch.arange(p0, p0 + 4, device=dev)
            ref = A.referencia(q, k, v, pos)                     # [4, 12, 256]
            qr = A.rotar_ptx(q[pos])
            for h in range(2):
                qh = qr[:, h * G:(h + 1) * G].reshape(-1, D)
                lim = pos.repeat_interleave(G)
                p = preparar(qh, kr[:p0 + 4, h], v[:p0 + 4, h], lim, CH)
                for d in DELTAS:
                    oi, sw, nsel = cadena(p, d)
                    o = salida(p, oi, sw).float().view(4, G, D)
                    r = ref[:, h * G:(h + 1) * G]
                    acum[d][0].append(((o - r).norm(dim=-1) / r.norm(dim=-1)).mean().item())
                    acum[d][1].append(min(1.0, nsel.item() * CH / (p0 + 4)))
        for d in DELTAS:
            e, f = acum[d]
            print(f"capa {capa} CH{CH} {'denso' if d is None else f'delta {d}':9s}: error {100*sum(e)/len(e):.3f}%  keys leidas {100*sum(f)/len(f):5.1f}%", flush=True)

print("\n== Parte 2: tiempo por capa (2 cabezas KV, M=24, capa 35 repetida)")
q, k, v = A.cargar(35)
kr = A.rotar_ptx(k); Nt = k.shape[0]
qr = A.rotar_ptx(q[Nt - 4:])
def medir(f, n=20):
    for _ in range(3): f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e3
for N in (16000, 57000, 100000):
    reps = (N + Nt - 1) // Nt
    for CH in (128, 512):
        ps = []
        for h in range(2):
            lim = (N - 4 + torch.arange(4, device=dev)).repeat_interleave(G)
            ps.append(preparar(qr[:, h * G:(h + 1) * G].reshape(-1, D), kr[:, h].repeat(reps, 1)[:N], v[:, h].repeat(reps, 1)[:N], lim, CH))
        fila = []
        for d in DELTAS:
            t = medir(lambda: [cadena(p, d) for p in ps])
            fr = sum(min(1.0, cadena(p, d)[2].item() * CH / N) for p in ps) / 2
            fila.append(f"{'denso' if d is None else f'd{d}'} {t:.3f} ms ({100*fr:.0f}%)")
        print(f"N={N} CH{CH}: " + " | ".join(fila), flush=True)
