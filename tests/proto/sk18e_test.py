"""SK-18e: exactitud bit a bit contra un emulador entero que replica el kernel, error vs
atencion float, y tiempo por capa contra SK-18d (NH=2) y FlashInfer."""
import math, sys, time, torch
import sk18_a0bis_ptx as A
from vllm._genesis.kernels.ptx_lab import Kernel
dev = "cuda"; D = 256; G = 6; PADZ = -(1 << 28); MINIT = -(1 << 30)
x = torch.linspace(0, 1, 2000, dtype=torch.float64)
bs = torch.linspace(0.1, 0.25, 3001, dtype=torch.float64)
bopt = bs[torch.stack([((1 - (0.5 + b) * x + b * x * x) / 2 ** (-x) - 1).abs().max() for b in bs]).argmin()].item()
QA, QB = round((0.5 + bopt) * 32768), round(bopt * 32768)
SH = (8 * int(__import__("os").environ.get("SK18_NW", "4")) * (256 + 128) + 2 * 64 * 256 + 2 * 256 * 64 + 2 * 256) if __import__("os").environ.get("SK18_VAR", "e") == "f" else (16 * 256 + 2 * 64 * 256 + 2 * 256 * 64 + 16 * 128 + 64)
VAR = __import__("os").environ.get("SK18_VAR", "e")
NW = int(__import__("os").environ.get("SK18_NW", "4"))
ke = Kernel(f"sk18{VAR}_stream.cu", f"sk18{VAR}_stream", defs=[f"-DQA={QA}", f"-DQB={QB}"] + ([f"-DNWARPS={NW}"] if VAR == "f" else []), warps=NW if VAR == "f" else 4)
BQ = 8 * NW if VAR == "f" else 16

def pot2neg(t):
    n = t >> 8; f = t & 255
    g = 32768 - ((f * QA + 128) >> 8) + ((f * f * QB + 32768) >> 16)
    nn = n.clamp(max=15)
    r = (2 * g + (1 << nn)) >> (nn + 1)
    return torch.where(n >= 16, torch.zeros_like(r), r)

def resc(a, sc):
    return (a >> 15) * sc + (((a & 0x7fff) * sc) >> 15)

def preparar(qrs, krs, vs, lim, CHK):
    NH = len(qrs); M, N = qrs[0].shape[0], krs[0].shape[0]
    MB = ((M + BQ - 1) // BQ) * BQ
    K = ((N + 63) // 64) * 64
    Qs, Ks, Vts, skfs, svfs, lims, mqbs, dcaps, svmax, mqs = [], [], [], [], [], [], [], [], [], []
    for qr, kr, v in zip(qrs, krs, vs):
        Q8, sq = A.q8_ptx(qr[:, None]); K8, sk = A.q8_ptx(kr[:, None]); V8, sv = A.q8_ptx(v[:, None])
        sq, sk, sv = sq[:, 0].double(), sk[:, 0].double(), sv[:, 0].double()
        Qp = torch.zeros(MB, D, dtype=torch.int8, device=dev); Qp[:M] = Q8[:, 0]; Qs.append(Qp)
        Ks.append(K8[:, 0])
        Vt = torch.zeros(D, K, dtype=torch.int8, device=dev); Vt[:, :N] = V8[:, 0].t(); Vts.append(Vt)
        skfs.append(torch.round(sk / sk.max() * 32767).to(torch.int16)); svfs.append(torch.round(sv / sv.max() * 32767).to(torch.int16))
        mq = torch.round(sk.max() * sq * 1023 / 16.0 * 65536).clamp(1, (1 << 31) - 1)
        mqb = torch.round(mq * 16 * 256 / (1023 * math.log(2))).clamp(1, (1 << 31) - 1).to(torch.int64)
        dcap = ((16 * 256 * 65536 + mqb - 1) // mqb).clamp(max=1 << 30)
        l = torch.full((MB,), -1, dtype=torch.int32, device=dev); l[:M] = lim
        mb = torch.ones(MB, dtype=torch.int64, device=dev); mb[:M] = mqb
        dc = torch.zeros(MB, dtype=torch.int64, device=dev); dc[:M] = dcap
        lims.append(l); mqbs.append(mb.to(torch.int32)); dcaps.append(dc.to(torch.int32)); svmax.append(sv.max())
    c = lambda xs: torch.cat(xs).contiguous()
    NCH = (N + CHK - 1) // CHK
    Vpag = torch.stack(Vts).permute(0, 2, 1).reshape(NH, K // 64, 64, D).permute(0, 1, 3, 2).contiguous()
    return dict(NH=NH, M=M, MB=MB, N=N, K=K, CHK=CHK, NCH=NCH, Q=c(Qs), Kc=c(Ks), Vt=c(Vts), Vpag=Vpag, skf=c(skfs), svf=c(svfs),
                lim=c(lims), mqb=c(mqbs), dcap=c(dcaps), svmax=torch.stack(svmax),
                oh=torch.zeros(NCH, NH * MB, D, dtype=torch.int32, device=dev), ol=torch.zeros(NCH, NH * MB, D, dtype=torch.int32, device=dev),
                om=torch.zeros(NCH, NH * MB, dtype=torch.int32, device=dev), os=torch.zeros(NCH, NH * MB, dtype=torch.int32, device=dev))

def escalas(p):
    NH, N, K = p["NH"], p["N"], p["K"]
    e = torch.zeros(NH, K, 2, dtype=torch.int16, device=dev)
    e[:, :N, 0] = p["skf"].view(NH, N); e[:, :N, 1] = p["svf"].view(NH, N)
    return e.contiguous()

def lanzar(p):
    if VAR == "f":
        if p.get("_esc_de") is not p["skf"]:
            p["esc"] = escalas(p); p["_esc_de"] = p["skf"]
        ke.lanzar((p["NCH"], p["NH"] * p["MB"] // BQ), [p["Q"], p["Kc"], p["Vpag"], p["esc"], p["lim"], p["mqb"], p["dcap"],
                                                        p["oh"], p["ol"], p["om"], p["os"], p["MB"], p["N"], p["K"], p["CHK"], p["NCH"], p["NH"]], shared=SH)
        return
    ke.lanzar((p["NCH"], p["NH"] * p["MB"] // BQ), [p["Q"], p["Kc"], p["Vpag"], p["skf"], p["svf"], p["lim"], p["mqb"], p["dcap"],
                                                    p["oh"], p["ol"], p["om"], p["os"], p["MB"], p["N"], p["K"], p["CHK"], p["NCH"], p["NH"]], shared=SH)

ku = Kernel("sk18f_union.cu", "sk18f_union", defs=[f"-DQA={QA}", f"-DQB={QB}"], warps=1)
def unir_ptx(p):
    R = p["NH"] * p["MB"]; NCH = p["NCH"]
    CPG = int(__import__("os").environ.get("SK18_CPG", "8"))
    NG = (NCH + CPG - 1) // CPG
    if p.get("_ug") != (R, NG):
        p["Og"] = torch.empty(NG, R, D, dtype=torch.int64, device=dev); p["Sg"] = torch.empty(NG, R, dtype=torch.int64, device=dev)
        p["_ug"] = (R, NG)
    ku.lanzar((R, NG), [p["oh"], p["ol"], p["om"], p["os"], p["mqb"], p["dcap"], p["Og"], p["Sg"], R, NCH, CPG])
    return p["Og"].sum(0), p["Sg"].sum(0)

def unir(p, oh, ol, om, os_):
    """int64: une los tramos reescalando al maximo global."""
    NH, MB = p["NH"], p["MB"]
    O = oh.to(torch.int64) * 256 + ol.to(torch.int64)                   # [NCH, R, D]
    mg = om.amax(0)
    dm = torch.minimum(mg[None].to(torch.int64) - om.to(torch.int64), p["dcap"].to(torch.int64)[None])
    sc = pot2neg((dm * p["mqb"].to(torch.int64)[None] + 32768) >> 16)
    Ot = ((O * sc[..., None] + 16384) >> 15).sum(0)
    St = ((os_.to(torch.int64) * sc + 16384) >> 15).sum(0)
    return Ot, St

def emulador(p):
    NH, MB, N, K, CHK, NCH = p["NH"], p["MB"], p["N"], p["K"], p["CHK"], p["NCH"]
    L = lambda t: t.to(torch.int64)
    oh = torch.zeros(NCH, NH * MB, D, dtype=torch.int64, device=dev); ol = torch.zeros_like(oh)
    om = torch.full((NCH, NH * MB), MINIT, dtype=torch.int64, device=dev); os_ = torch.zeros((NCH, NH * MB), dtype=torch.int64, device=dev)
    for h in range(NH):
        Q = L(p["Q"][h * MB:(h + 1) * MB]); Kc = L(p["Kc"][h * N:(h + 1) * N]); V = L(p["Vt"][h * D:(h + 1) * D])
        skf = L(p["skf"][h * N:(h + 1) * N]); svf = L(p["svf"][h * N:(h + 1) * N])
        lim = L(p["lim"][h * MB:(h + 1) * MB]); mq = L(p["mqb"][h * MB:(h + 1) * MB]); dc = L(p["dcap"][h * MB:(h + 1) * MB])
        z = torch.round(Q.double() @ Kc.double().t()).long()
        zp = ((z >> 8) * skf[None]) >> 11
        vis = torch.arange(N, device=dev)[None] <= lim[:, None]
        zp = torch.where(vis, zp, torch.full_like(zp, PADZ))
        for c in range(NCH):
            m = torch.full((MB,), MINIT, dtype=torch.int64, device=dev); S = torch.zeros(MB, 8 if VAR == 'f' else 4, dtype=torch.int64, device=dev)
            ah = torch.zeros(MB, D, dtype=torch.int64, device=dev); al = torch.zeros_like(ah)
            for k0 in range(c * CHK, min(c * CHK + CHK, N), 64):
                k1 = min(k0 + 64, N, c * CHK + CHK)
                zq = zp[:, k0:k1]
                mn = torch.maximum(m, zq.amax(1))
                dm = torch.minimum(mn - m, dc)
                sc = pot2neg((dm * mq + 32768) >> 16)
                S = resc(S, sc[:, None]); ah = resc(ah, sc[:, None]); al = resc(al, sc[:, None])
                m = mn
                d = torch.minimum(m[:, None] - zq, dc[:, None])
                w0 = pot2neg((d * mq[:, None] + 32768) >> 16)
                pad = zq == PADZ
                w = torch.where(pad, torch.zeros_like(w0), (w0 * 32639) >> 15)
                sv = torch.where(pad, torch.zeros_like(w), svf[None, k0:k1].expand_as(w))
                wp = (w * sv + 16384) >> 15
                hi = (wp + 128) >> 8; lo = wp - (hi << 8)
                tg = ((torch.arange(k0, k1, device=dev) - k0) % 8) // (1 if VAR == 'f' else 2)
                S = S + torch.zeros_like(S).index_add_(1, tg, w)
                ah = ah + torch.round(hi.double() @ V[:, k0:k1].t().double()).long()
                al = al + torch.round(lo.double() @ V[:, k0:k1].t().double()).long()
            om[c, h * MB:(h + 1) * MB] = m; os_[c, h * MB:(h + 1) * MB] = S.sum(1)
            oh[c, h * MB:(h + 1) * MB] = ah; ol[c, h * MB:(h + 1) * MB] = al
    return oh, ol, om, os_

q, k, v = A.cargar(35); Nt = k.shape[0]
kr = A.rotar_ptx(k)
modo = sys.argv[1] if len(sys.argv) > 1 else "exacto"
if modo == "exacto":
    for N, CHK in ((700, 256), (3000, 1024)):
        pos = torch.arange(N - 4, N, device=dev); qr = A.rotar_ptx(q[pos])
        lim = pos.repeat_interleave(G).to(torch.int32)
        p = preparar([qr[:, h * G:(h + 1) * G].reshape(-1, D) for h in range(2)], [kr[:N, h] for h in range(2)], [v[:N, h] for h in range(2)], lim, CHK)
        lanzar(p); torch.cuda.synchronize()
        eh, el, em, es = emulador(p)
        ok = [bool((p["oh"].long() == eh).all()), bool((p["ol"].long() == el).all()), bool((p["om"].long() == em).all()), bool((p["os"].long() == es).all())]
        print(f"N={N} CHK={CHK}: hi/lo/m/S iguales = {ok}", flush=True)
        if not all(ok):
            dif = (p["om"].long() != em).nonzero()[:5]
            print("  m kernel", p["om"][0, :8].tolist(), "\n  m emul  ", em[0, :8].tolist())
            print("  S kernel", p["os"][0, :8].tolist(), "\n  S emul  ", es[0, :8].tolist())
            print("  hi kernel", p["oh"][0, 0, :6].tolist(), "\n  hi emul  ", eh[0, 0, :6].tolist())
        # error vs float
        ref = A.referencia(q, k, v, pos)
        Ot, St = unir(p, p["oh"], p["ol"], p["om"], p["os"])
        out = torch.stack([Ot[h * p["MB"]:h * p["MB"] + 24].double() * p["svmax"][h] / St[h * p["MB"]:h * p["MB"] + 24].double().clamp_min(1)[:, None] for h in range(2)])
        r = torch.stack([ref[:, h * G:(h + 1) * G].reshape(-1, D) for h in range(2)]).double()
        print(f"  error vs float: {100*((out - r).norm(dim=-1) / r.norm(dim=-1)).mean().item():.3f}%", flush=True)
else:
    qr = A.rotar_ptx(q[Nt - 4:])
    def medir(f, n=20):
        for _ in range(3): f()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(n): f()
        torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e3
    lim0 = torch.zeros(24, dtype=torch.int32, device=dev)
    base = preparar([qr[:, h * G:(h + 1) * G].reshape(-1, D) for h in range(2)], [kr[:, h] for h in range(2)], [v[:, h] for h in range(2)], lim0, 1024)
    Kb = base["Kc"].view(2, Nt, D); Vb = base["Vt"].view(2, D, -1)[:, :, :Nt]; skb = base["skf"].view(2, Nt); svb = base["svf"].view(2, Nt)
    for N in (16000, 57000, 100000):
        reps = (N + Nt - 1) // Nt
        K = ((N + 63) // 64) * 64
        Kc = Kb.repeat(1, reps, 1)[:, :N].reshape(-1, D).contiguous()
        Vt = torch.zeros(2, D, K, dtype=torch.int8, device=dev); Vt[:, :, :N] = Vb.repeat(1, 1, reps)[:, :, :N]
        Vpag = Vt.permute(0, 2, 1).reshape(2, K // 64, 64, D).permute(0, 1, 3, 2).contiguous()
        skf = skb.repeat(1, reps)[:, :N].reshape(-1).contiguous(); svf = svb.repeat(1, reps)[:, :N].reshape(-1).contiguous()
        lim = torch.full((2, base["MB"]), -1, dtype=torch.int32, device=dev)
        lim[:, :24] = (N - 4 + torch.arange(4, device=dev)).repeat_interleave(G)
        for CHK in [int(c) for c in __import__("os").environ.get("SK18_CHK", "512,1024,2048").split(",")]:
            NCH = (N + CHK - 1) // CHK
            p = dict(base); p.update(N=N, K=K, CHK=CHK, NCH=NCH, Kc=Kc, Vt=None, Vpag=Vpag, skf=skf, svf=svf, lim=lim.reshape(-1).contiguous(),
                     oh=torch.zeros(NCH, 2 * base["MB"], D, dtype=torch.int32, device=dev), ol=torch.zeros(NCH, 2 * base["MB"], D, dtype=torch.int32, device=dev),
                     om=torch.zeros(NCH, 2 * base["MB"], dtype=torch.int32, device=dev), os=torch.zeros(NCH, 2 * base["MB"], dtype=torch.int32, device=dev))
            tk = medir(lambda: lanzar(p))
            tu = medir(lambda: unir(p, p["oh"], p["ol"], p["om"], p["os"]))
            lanzar(p); O1, S1 = unir(p, p["oh"], p["ol"], p["om"], p["os"]); O2, S2 = unir_ptx(p)
            igual = bool((O1 == O2).all() and (S1 == S2).all())
            tp = medir(lambda: unir_ptx(p))
            tt = medir(lambda: (lanzar(p), unir_ptx(p)))
            print(f"    union PTX {tp:.3f} ms (igual a torch: {igual}); kernel + union PTX = {tt:.3f} ms/capa", flush=True)
            print(f"N={N} CHK={CHK}: kernel {tk:.3f} ms + union {tu:.3f} ms = {tk+tu:.3f} ms/capa", flush=True)
            del p
        del Kc, Vt, Vpag; torch.cuda.empty_cache()
