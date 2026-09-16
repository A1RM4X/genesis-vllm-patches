import sys, types, time, torch
exec(open("/p/pn131_offline.py").read().split("q, k, v = A.cargar(35)")[0])
q, k, v = A.cargar(35); Nt = k.shape[0]
from vllm._genesis.kernels.ptx_lab import Kernel
qa, qb = P._coef()
VARS = {"completo": 0, "sin mma Q.K": 1, "sin reescalado": 2, "sin mma w.v": 3, "sin carga V": 4, "sin carga K": 5,
        "sin guardar pesos": 6, "sin barrera w.v": 7, "sin calculo pesos": 8}
N = 57000
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
for nom, dg in VARS.items():
    kr = Kernel("sk18h_batch.cu", "sk18h_batch", defs=[f"-DQA={qa}", f"-DQB={qb}", f"-DDIAG={dg}", "-DSALTEAR=0", "-DSINCRONO=1"], warps=4)
    prev = None; iguales = []
    for rep in range(4):
        for b_ in (bf.oh, bf.ol, bf.om, bf.os): b_.zero_()
        kr.lanzar((NCH, 2), args, shared=P.SH_H); torch.cuda.synchronize()
        cur = torch.cat([bf.oh[: NCH * R * D], bf.ol[: NCH * R * D]]).clone()
        if prev is not None: iguales.append(bool((cur == prev).all()))
        prev = cur
    print(f"{nom:20s} repeticiones iguales: {iguales}", flush=True)
