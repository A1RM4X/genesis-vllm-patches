"""SK-18d con las 2 cabezas KV en un lanzamiento por etapa y sin resets por paso.
Exactitud contra 2 lanzamientos NH=1; tiempo por capa con todo el pegamento."""
import math, time, torch
import sk18_a0bis_ptx as A
from vllm._genesis.kernels.ptx_lab import Kernel
dev = "cuda"; D = 256; G = 6; PAD = -(1 << 28); CH = 512; NTH = 256
LUT = A.LUT
kqk = Kernel("sk18d_qk.cu", "sk18d_qk", warps=8); SH_Q = 2 * (32 * 128 + 64 * 256)
kwv = Kernel("sk18d_wv.cu", "sk18d_wv", warps=8); SH_W = 2 * (2 * 32 * 128 + 64 * 256)

def preparar(qrs, krs, vs, lim):
    """listas por cabeza: qr [M,256], kr/v [N,256]; lim [M] (igual para todas)."""
    NH = len(qrs); M, N = qrs[0].shape[0], krs[0].shape[0]
    K = ((N + CH - 1) // CH) * CH; NCH = K // CH; TM = (M + 31) // 32
    Q8s, K8s, skfs, mqs, dcaps, Vts, svfs, svmax = [], [], [], [], [], [], [], []
    for qr, kr, v in zip(qrs, krs, vs):
        Q8, sq = A.q8_ptx(qr[:, None]); K8, sk = A.q8_ptx(kr[:, None]); V8, sv = A.q8_ptx(v[:, None])
        sq, sk, sv = sq[:, 0].float(), sk[:, 0].float(), sv[:, 0].float()
        Q8s.append(Q8[:, 0]); K8s.append(K8[:, 0])
        skfs.append(torch.round(sk / sk.max() * 32767).to(torch.int16))
        mq = torch.round(256.0 * sk.max() * sq / 256.0 * 1023 / 16.0 * 65536).clamp(1, (1 << 31) - 1).to(torch.int32)
        mqs.append(mq); dcaps.append(((1023 * 65536 - 32768 + mq.to(torch.int64) - 1) // mq.to(torch.int64)).clamp(max=1 << 30).to(torch.int32))
        Vt = torch.zeros(D, K, dtype=torch.int8, device=dev); Vt[:, :N] = V8[:, 0].t(); Vts.append(Vt)
        svf = torch.zeros(K, dtype=torch.int16, device=dev); svf[:N] = torch.round(sv / sv.max() * 32767).to(torch.int16); svfs.append(svf)
        svmax.append(sv.max())
    c = lambda xs: torch.cat(xs).contiguous()
    return dict(NH=NH, M=M, N=N, K=K, NCH=NCH, TM=TM, Q8=c(Q8s), K8=c(K8s), skf=c(skfs), lim=lim.repeat(NH).to(torch.int32).contiguous(),
                mq=c(mqs), dcap=c(dcaps), Vt=c(Vts), svf=c(svfs), svmax=torch.stack(svmax),
                Sz=torch.full((NH * M, K), PAD, dtype=torch.int32, device=dev),           # relleno UNA vez
                out=torch.zeros(NCH, NH * M, D, dtype=torch.int32, device=dev), lo=torch.zeros(NCH, NH * M, D, dtype=torch.int32, device=dev),
                sumw=torch.zeros(NCH * NH * TM * NTH, dtype=torch.int64, device=dev),
                tabla=torch.arange(NCH, dtype=torch.int32, device=dev), nsel=torch.full((1,), NCH, dtype=torch.int32, device=dev))

def cadena(p):
    NH, M, N, K, NCH, TM = p["NH"], p["M"], p["N"], p["K"], p["NCH"], p["TM"]
    Sz = p["Sz"]
    kqk.lanzar(((N + 63) // 64, NH * TM), [p["Q8"], p["K8"], p["skf"], p["lim"], Sz, M, N, 256, K, NH], shared=SH_Q)
    zm = Sz.amax(1)
    kwv.lanzar(((D + 63) // 64, NH * TM * NCH), [Sz, zm, p["mq"], p["dcap"], LUT, p["svf"], p["Vt"], p["out"], p["lo"],
                                                p["tabla"], p["nsel"], p["sumw"], M, D, K, CH, NCH, NH], shared=SH_W)
    oi = p["out"].sum(0, dtype=torch.int32).to(torch.int64) * 256 + p["lo"].sum(0, dtype=torch.int32)
    sw = p["sumw"].view(NCH, NH, TM, 32, 8).sum((0, 4)).reshape(NH, -1)[:, :M].reshape(-1)
    return oi, sw

def medir(f, n=30):
    for _ in range(5): f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e3

q, k, v = A.cargar(35)
kr = A.rotar_ptx(k); Nt = k.shape[0]
qr = A.rotar_ptx(q[Nt - 4:])
for N in (16000, 57000, 100000):
    reps = (N + Nt - 1) // Nt
    lim = (N - 4 + torch.arange(4, device=dev)).repeat_interleave(G)
    qh = [qr[:, h * G:(h + 1) * G].reshape(-1, D) for h in range(2)]
    kh = [kr[:, h].repeat(reps, 1)[:N] for h in range(2)]
    vh = [v[:, h].repeat(reps, 1)[:N] for h in range(2)]
    p2 = preparar(qh, kh, vh, lim)
    p1 = [preparar([qh[h]], [kh[h]], [vh[h]], lim) for h in range(2)]
    oi2, sw2 = cadena(p2)
    r = [cadena(p) for p in p1]
    exacto = bool((oi2 == torch.cat([x[0] for x in r])).all() and (sw2 == torch.cat([x[1] for x in r])).all())
    t2 = medir(lambda: cadena(p2)); t1 = medir(lambda: [cadena(p) for p in p1])
    print(f"N={N}: 2 lanzamientos por etapa {t1:.3f} ms/capa | 1 lanzamiento (NH=2) {t2:.3f} ms/capa | exacto={exacto}", flush=True)
