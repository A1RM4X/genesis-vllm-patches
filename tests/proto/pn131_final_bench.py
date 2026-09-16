import sys, types, time, torch
exec(open("/p/pn131_offline.py").read().split("q, k, v = A.cargar(35)")[0])
q, k, v = A.cargar(35); Nt = k.shape[0]
def medir(f, n=40):
    for _ in range(5): f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e3
from vllm._genesis.kernels.ptx_lab import Kernel
qa, qb = P._coef(); defs = [f"-DQA={qa}", f"-DQB={qb}"]
ku = Kernel("sk18h_union.cu", "sk18h_union", defs=defs, warps=1)
ksal = Kernel("sk18h_salida.cu", "sk18h_salida", warps=1)
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
    for capt in (False, True):
        P._decode_uniforme(impl, qq, kv, md, out, 1, 4, capt)
        bf = P._bufs[0]; ks = P._kernels(); c = P._capa(impl, dev)
        NCH = bf.nchmax if capt else (N + BS - 1) // BS
        R = 2 * 32; NG = (NCH + 15) // 16
        oh = bf.oh[: NCH * R * D]; ol = bf.ol[: NCH * R * D]; om = bf.om[: NCH * R]; os_ = bf.os[: NCH * R]
        mqb = bf.mqb[:R]; dcap = bf.dcap[:R]; seq = md.seq_lens
        o16 = out.view(torch.int16)
        Og4 = torch.empty(NG * R * D, dtype=torch.int64, device=dev); Sg4 = torch.empty(NG * R, dtype=torch.int64, device=dev)
        def nuevo():
            ks["union"].lanzar((R, NG), [oh, ol, om, os_, mqb, dcap, seq, Og4, Sg4, R, NCH, 16, 64, BS])
            ks["salida"].lanzar((4, 12), [Og4, Sg4, c.refs, o16, NG, R, 4, 2, 6, 32, P.VSH])
        tf = medir(nuevo)
        mg = torch.empty(R, dtype=torch.int32, device=dev); Og = torch.empty(NG * R * D, dtype=torch.int64, device=dev); Sg = torch.empty(NG * R, dtype=torch.int64, device=dev)
        def viejo():
            torch.amax(om.view(NCH, R), 0, out=mg)
            ku.lanzar((R, NG), [oh, ol, om, os_, mqb, dcap, mg, Og, Sg, R, NCH, 16])
            ksal.lanzar((4, 12), [Og, Sg, c.refs, o16, NG, R, 4, 2, 6, 32, P.VSH])
        tv = medir(viejo)
        print(f"N={N} NCH={NCH}: union4+salida {tf:.3f} ms | amax+union+salida {tv:.3f} ms", flush=True)
