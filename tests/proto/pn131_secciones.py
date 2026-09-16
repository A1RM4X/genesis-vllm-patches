"""Perfil por seccion del kernel principal SK-18h (int8) con ablaciones DIAG, a 57k y 100k."""
import sys, types, time, torch
exec(open("/p/pn131_offline.py").read().split("q, k, v = A.cargar(35)")[0])
q, k, v = A.cargar(35); Nt = k.shape[0]
from vllm._genesis.kernels.ptx_lab import Kernel
def medir(f, n=40):
    for _ in range(5): f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e3
qa, qb = P._coef()
NOMBRES = {0: "sincronico", 1: "sincronico + SALTEAR", 2: "sincronico (repeticion)", 3: "pipeline viejo"}
kers = {dg: Kernel("sk18h_batch.cu", "sk18h_batch", defs=[f"-DQA={qa}", f"-DQB={qb}", f"-DSALTEAR={1 if dg == 1 else 0}", f"-DREP={dg}", f"-DSINCRONO={0 if dg == 3 else 1}"], warps=4) for dg in NOMBRES}
for N in (57000, 100000):
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
    outs = {}
    for dg in NOMBRES:
        for b_ in (bf.oh, bf.ol, bf.om, bf.os): b_.zero_()
        kers[dg].lanzar((NCH, 2), args, shared=P.SH_H); torch.cuda.synchronize()
        outs[dg] = [x[: y].clone() for x, y in ((bf.oh, NCH * R * D), (bf.ol, NCH * R * D), (bf.om, NCH * R), (bf.os, NCH * R))]
    print(f"N={N}: exacto SALTEAR vs antes = {[bool((a == b).all()) for a, b in zip(outs[0], outs[1])]} | antes vs antes = {[bool((a == b).all()) for a, b in zip(outs[0], outs[2])]}", flush=True)
    for rep in range(3):
        for b_ in (bf.oh, bf.ol, bf.om, bf.os): b_.zero_()
        kers[0].lanzar((NCH, 2), args, shared=P.SH_H); torch.cuda.synchronize()
        print("  repeticion", rep, "igual:", bool((bf.oh[: NCH * R * D] == outs[0][0]).all() and (bf.ol[: NCH * R * D] == outs[0][1]).all()), flush=True)
    a0 = outs[0][0].view(NCH, R, D); a2 = outs[2][0].view(NCH, R, D)
    dif = (a0 != a2)
    filas = dif.any(-1).any(0).nonzero().flatten().tolist(); pags = dif.any(-1).any(-1).nonzero().flatten().tolist()
    print("  filas distintas", filas, "paginas", pags[:10], "n", len(pags), "max|dif|", (a0.long() - a2.long()).abs().max().item(), "lim filas", bf.lim[:R].tolist()[:32], flush=True)
    pass
    for dg in NOMBRES:
        pass
    res = {dg: [] for dg in NOMBRES}
    for ronda in range(5):
        for dg in NOMBRES:
            res[dg].append(medir(lambda: kers[dg].lanzar((NCH, 2), args, shared=P.SH_H), n=30))
    base = sorted(res[0])[2]
    for dg, nom in NOMBRES.items():
        t = sorted(res[dg])[2]
        print(f"N={N} DIAG={dg} {nom:22s} mediana {t:.3f} ms  rango {min(res[dg]):.3f}-{max(res[dg]):.3f}  ahorro {100*(base-t)/base:+.0f}%", flush=True)
