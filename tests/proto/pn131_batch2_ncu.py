import sys, types, time, torch
exec(open("/p/pn131_offline.py").read().split("q, k, v = A.cargar(35)")[0])
q, k, v = A.cargar(35); Nt = k.shape[0]
from vllm._genesis.kernels.ptx_lab import Kernel
qa, qb = P._coef()
k2 = Kernel("sk18h_batch2.cu", "sk18h_batch2", defs=[f"-DQA={qa}", f"-DQB={qb}"], warps=4)
SH2 = 32 * 256 + 2 * 64 * 256 + 2 * 256 * 64 + 32 * 128 + 2 * 64 * 2 * 4
k1 = Kernel("sk18h_batch.cu", "sk18h_batch", defs=[f"-DQA={qa}", f"-DQB={qb}", "-DSALTEAR=0"], warps=4)
QA_, QB_ = qa, qb
def pot2neg(t):
    n = t >> 8; f = t & 255
    g = 32768 - ((f * QA_ + 128) >> 8) + ((f * f * QB_ + 32768) >> 16)
    nn = n.clamp(max=15)
    r = (2 * g + (1 << nn)) >> (nn + 1)
    return torch.where(n >= 16, torch.zeros_like(r), r)
def medir(f, n=30):
    for _ in range(5): f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e3
for N in (57000,):
    reps = (N + Nt - 1) // Nt
    kk = k.repeat(reps, 1, 1)[:N]; vv = v.repeat(reps, 1, 1)[:N]
    nblocks = (N + BS - 1) // BS + 4
    kv = kv_nuevo(nblocks)
    perm = torch.randperm(nblocks)[: (N + BS - 1) // BS].to(torch.int32)
    slots = (perm[torch.arange(N) // BS].to(torch.int64) * BS + torch.arange(N) % BS).to(dev)
    for t0 in range(0, N, 7488):
        P.escribir(impl, layer, kk[t0:t0 + 7488].half(), vv[t0:t0 + 7488].half(), kv, slots[t0:t0 + 7488])
    bt = torch.zeros(1, 316, dtype=torch.int32); bt[0, :perm.numel()] = perm
    md = meta(torch.tensor([0, 4], dtype=torch.int32), torch.tensor([N], dtype=torch.int32), bt, 4)
    qq = q[Nt - 4:].half(); out = torch.zeros(4, 12, D, dtype=torch.float16, device=dev)
    P._decode_uniforme(impl, qq, kv, md, out, 1, 4, False)
    bf = P._bufs[0]; nb, nh, bs_, blk, raw = P._geom(kv)
    NCH = (N + BS - 1) // BS; R = 64
    args = [bf.Q[: R * D], raw, md.block_table, md.seq_lens, bf.lim[:R], bf.mqb[:R], bf.dcap[:R],
            bf.oh[: NCH * R * D], bf.ol[: NCH * R * D], bf.om[: NCH * R], bf.os[: NCH * R],
            blk, md.block_table.stride(0), BS, NCH, 2, P.ZSH, P.VSH]
    def correr(kr, sh=P.SH_H):
        for b_ in (bf.oh, bf.ol, bf.om, bf.os): b_.zero_()
        kr.lanzar((NCH, 2), args, shared=sh); torch.cuda.synchronize()
        return [bf.oh[: NCH * R * D].clone().view(NCH, R, D), bf.ol[: NCH * R * D].clone().view(NCH, R, D),
                bf.om[: NCH * R].clone().view(NCH, R), bf.os[: NCH * R].clone().view(NCH, R)]
    r0 = correr(k2, SH2); rep_ok = all(all(bool((a == b).all()) for a, b in zip(r0, correr(k2, SH2))) for _ in range(3))
    # emulador entero de la pasada doble
    L64 = lambda t: t.to(torch.int64)
    Q8 = L64(bf.Q[: R * D].view(2, 32, D)); lim = L64(bf.lim[:R].view(2, 32)); mq = L64(bf.mqb[:R].view(2, 32)); dc = L64(bf.dcap[:R].view(2, 32))
    koff = BS * 2 * D
    eh = torch.zeros(NCH, R, D, dtype=torch.int64, device=dev); el = torch.zeros_like(eh)
    em = torch.full((NCH, R), -(1 << 30), dtype=torch.int64, device=dev); es = torch.zeros((NCH, R), dtype=torch.int64, device=dev)
    for c in range(NCH):
        pg = raw[int(perm[c])]
        K8 = L64(pg[:koff].view(BS, 2, D)); V8 = L64(pg[koff:2 * koff].view(2, D, BS))
        E = pg[2 * koff:2 * koff + BS * 2 * 4].contiguous().view(torch.int16).view(BS, 2, 2).to(torch.int64)
        kfin = min(N, (c + 1) * BS) - c * BS
        for h in range(2):
            z = torch.round(Q8[h].double() @ K8[:, h].double().t()).long()        # [32, BS]
            zp = (z * E[:, h, 0][None]) >> P.ZSH
            pos = c * BS + torch.arange(BS, device=dev)
            vis = (torch.arange(BS, device=dev)[None] < kfin) & (pos[None] <= lim[h][:, None])
            zp = torch.where(vis, zp, torch.full_like(zp, -(1 << 28)))
            m = torch.where(vis, zp, torch.full_like(zp, -(1 << 30))).amax(1)
            d = torch.minimum(m[:, None] - zp, dc[h][:, None])
            w0 = pot2neg((d * mq[h][:, None] + 32768) >> 16)
            w = torch.where(vis, (w0 * 32639) >> 15, torch.zeros_like(w0))
            sv = torch.where(vis, E[:, h, 1][None].expand_as(w), torch.zeros_like(w))
            wp = (w * sv + (1 << (P.VSH - 1))) >> P.VSH
            hi = (wp + 128) >> 8; lo = wp - (hi << 8)
            eh[c, h * 32:(h + 1) * 32] = torch.round(hi.double() @ V8[h].t().double()).long()
            el[c, h * 32:(h + 1) * 32] = torch.round(lo.double() @ V8[h].t().double()).long()
            em[c, h * 32:(h + 1) * 32] = m; es[c, h * 32:(h + 1) * 32] = w.sum(1)
    exact = [bool((r0[0].long() == eh).all()), bool((r0[1].long() == el).all()), bool((r0[2].long() == em).all()), bool((r0[3].long() == es).all())]
    for _ in range(12):
        k1.lanzar((NCH, 2), args, shared=P.SH_H); k2.lanzar((NCH, 2), args, shared=SH2)
    torch.cuda.synchronize()
    print(f"N={N} listo exacto={exact} rep={rep_ok}", flush=True)
