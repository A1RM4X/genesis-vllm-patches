"""SK-18d: Q.K (escala+causal en epilogo) -> amax -> softmax entero + sum(w*v) 32 bits.
Exactitud bit a bit contra la cadena SK-18a + pegamento torch + SK-18b, y tiempo por
cabeza KV con todo el pegamento incluido. Datos: q/k/v reales (capa 35) repetidas."""
import math, sys, time, torch
from vllm._genesis.kernels.ptx_lab import Kernel
dev = "cuda"; D = 256; PAD = -(1 << 28)
LUT = torch.round(32639 * torch.exp(-16.0 * torch.arange(1024).float() / 1023)).to(torch.int32).to(dev); LUT[-1] = 0
SH_A = 2 * (256 * 128 + 64 * 2 * 128)
SH_B = 2 * (2 * 64 * 128 + 64 * 2 * 128)
k18a = Kernel("sk18a_qk_i32.cu", "sk18a_qk_i32", warps=8)
k18b = Kernel("sk18b_wv.cu", "sk18b_wv", warps=4)
CFG_QK = {"BM256": (["-DBM=256", "-DWM=64", "-DWN=32", "-DGROUP_M=8"], 8, SH_A),
          "BM32": (["-DBM=32", "-DWM=32", "-DBN=64", "-DWN=8", "-DNWARPS=8", "-DGROUP_M=1"], 8, 2 * (32 * 128 + 64 * 256)),
          "BM16": (["-DBM=16", "-DWM=16", "-DBN=64", "-DWN=8", "-DNWARPS=8", "-DGROUP_M=1"], 8, 2 * (16 * 128 + 64 * 256))}
SH_D = 2 * (2 * 32 * 128 + 64 * 2 * 128)
kdw = Kernel("sk18d_wv.cu", "sk18d_wv", warps=8)

def q8(x):
    s = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
    return (x / s).round().clamp(-127, 127).to(torch.int8), s.squeeze(-1)

d = [torch.load(f"/kv/capa35_{i}.pt") for i in (1, 2, 3)]
q = torch.cat([x["q"] for x in d]).view(-1, 12, D).float().to(dev)
k = torch.cat([x["k"] for x in d]).view(-1, 2, D).float().to(dev)
v = torch.cat([x["v"] for x in d]).view(-1, 2, D).float().to(dev)

def preparar(N, M, CH=512):
    reps = (N + k.shape[0] - 1) // k.shape[0]
    kk = k[:, 0].repeat(reps, 1)[:N]; vv = v[:, 0].repeat(reps, 1)[:N]
    qq = q[-M // 6:, :6].reshape(-1, D)[:M]
    Q8, sq = q8(qq); K8, sk = q8(kk); V8, sv = q8(vv)
    K = ((N + CH - 1) // CH) * CH
    skf = torch.round(sk / sk.max() * 32767).to(torch.int16)
    lim = torch.full((M,), N - 1, dtype=torch.int32, device=dev) - torch.arange(M, device=dev, dtype=torch.int32) % 4
    alpha = 256.0 * sk.max() * sq / math.sqrt(D) / 16.0
    mq = torch.round(alpha * 1023 / 16.0 * 65536).clamp(1, (1 << 31) - 1).to(torch.int32)
    dcap = ((1023 * 65536 - 32768 + mq.to(torch.int64) - 1) // mq.to(torch.int64)).clamp(max=1 << 30).to(torch.int32)
    Vt = torch.zeros(D, K, dtype=torch.int8, device=dev); Vt[:, :N] = V8.t()
    svf = torch.zeros(K, dtype=torch.int16, device=dev); svf[:N] = torch.round(sv / sv.max() * 32767).to(torch.int16)
    return dict(N=N, M=M, K=K, CH=CH, Q8=Q8.contiguous(), K8=K8.contiguous(), skf=skf.contiguous(), lim=lim,
                mq=mq, dcap=dcap, Vt=Vt.contiguous(), svf=svf)

def cadena_vieja(p):
    N, M, K, CH = p["N"], p["M"], p["K"], p["CH"]
    z = torch.empty(M, N, dtype=torch.int32, device=dev)
    k18a.lanzar(((N + 63) // 64, (M + 255) // 256), [p["Q8"], p["K8"], z, M, N, 256], shared=SH_A)
    zp = ((z >> 8) * p["skf"][None, :].to(torch.int32)) >> 11
    zp = zp.masked_fill(torch.arange(N, device=dev)[None, :] > p["lim"][:, None], PAD)
    Szp = torch.full((M, K), PAD, dtype=torch.int32, device=dev); Szp[:, :N] = zp
    zm = zp.amax(1).to(torch.int32)
    out = torch.empty(K // CH, M, D, dtype=torch.int32, device=dev); lo = torch.empty_like(out)
    k18b.lanzar(((D + 63) // 64, ((M + 63) // 64) * (K // CH)), [Szp, zm, p["mq"], zm, LUT, p["svf"], p["Vt"], out, lo, M, D, K, CH, K // CH], shared=SH_B)
    return out.sum(0, dtype=torch.int64) * 256 + lo.sum(0, dtype=torch.int64)

def cadena_d(p, cfg, kqk, buf):
    N, M, K, CH = p["N"], p["M"], p["K"], p["CH"]
    _, _, shq = CFG_QK[cfg]; bm = int([x for x in CFG_QK[cfg][0] if "BM=" in x][0][5:]) if CFG_QK[cfg][0] else 256
    Sz, out, lo = buf
    kqk.lanzar(((N + 63) // 64, (M + bm - 1) // bm), [p["Q8"], p["K8"], p["skf"], p["lim"], Sz, M, N, 256, K], shared=shq)
    zm = Sz.amax(1)
    kdw.lanzar(((D + 63) // 64, ((M + 31) // 32) * (K // CH)), [Sz, zm, p["mq"], p["dcap"], LUT, p["svf"], p["Vt"], out, lo, M, D, K, CH, K // CH], shared=SH_D)
    return out.sum(0, dtype=torch.int32).to(torch.int64) * 256 + lo.sum(0, dtype=torch.int32)   # int32 alcanza hasta 133k keys

def medir(f, n=20):
    for _ in range(3): f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e3

for N in (16000, 57000, 100000):
    M = 24
    p = preparar(N, M)
    ref = cadena_vieja(p)
    t_v = medir(lambda: cadena_vieja(p))
    print(f"\nN={N} M={M}: vieja (18a + torch + 18b) {t_v:.3f} ms/cabeza = {2*t_v:.3f} ms/capa", flush=True)
    for cfg in ("BM32",):
        kqk = Kernel("sk18d_qk.cu", "sk18d_qk", defs=CFG_QK[cfg][0], warps=CFG_QK[cfg][1])
        Sz = torch.full((M, p["K"]), PAD, dtype=torch.int32, device=dev)
        buf = (Sz, torch.empty(p["K"] // p["CH"], M, D, dtype=torch.int32, device=dev), None)
        buf = (Sz, buf[1], torch.empty_like(buf[1]))
        got = cadena_d(p, cfg, kqk, buf)
        t = medir(lambda: cadena_d(p, cfg, kqk, buf))
        print(f"  SK-18d qk {cfg}: exacto={bool((got == ref).all())}  {t:.3f} ms/cabeza = {2*t:.3f} ms/capa", flush=True)
