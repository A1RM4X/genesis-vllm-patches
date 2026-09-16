import sys, time, torch
sys.argv=[sys.argv[0]]
exec(open("/p/sk18d_bench.py").read().split("for N in (16000")[0])
def medir(f, n=30):
    for _ in range(3): f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e3
N, M = 57000, 24
p = preparar(N, M)
K = p["K"]
kqk = Kernel("sk18d_qk.cu", "sk18d_qk", defs=CFG_QK["BM32"][0], warps=8)
Sz = torch.full((M, K), PAD, dtype=torch.int32, device=dev)
fq = lambda: kqk.lanzar(((N + 63) // 64, 1), [p["Q8"], p["K8"], p["skf"], p["lim"], Sz, M, N, 256, K], shared=CFG_QK["BM32"][2])
print(f"qk BM32 {medir(fq):.3f}")
fq(); zm = Sz.amax(1)
print(f"amax {medir(lambda: Sz.amax(1)):.3f}")
for bm, wm, warps in ((16, 16, 4), (32, 32, 4), (32, 16, 8), (16, 16, 8)):
  for CH in (512, 1024, 2048, 4096):
    Kc = ((N + CH - 1) // CH) * CH
    if Kc != K:
        Szc = torch.full((M, Kc), PAD, dtype=torch.int32, device=dev); Szc[:, :N] = Sz[:, :N]
        Vt = torch.zeros(D, Kc, dtype=torch.int8, device=dev); Vt[:, :N] = p["Vt"][:, :N]
        svf = torch.zeros(Kc, dtype=torch.int16, device=dev); svf[:N] = p["svf"][:N]
    else:
        Szc, Vt, svf = Sz, p["Vt"], p["svf"]
    wn = 64 // (warps // (bm // wm))
    try:
        kw = Kernel("sk18d_wv.cu", "sk18d_wv", defs=[f"-DBM={bm}", f"-DWM={wm}", "-DBN=64", f"-DWN={wn}", f"-DNWARPS={warps}"], warps=warps)
        sh = 2 * (2 * bm * 128 + 64 * 256)
        out = torch.empty(Kc // CH, M, D, dtype=torch.int32, device=dev); lo = torch.empty_like(out)
        fw = lambda: kw.lanzar((4, ((M + bm - 1) // bm) * (Kc // CH)), [Szc, zm, p["mq"], p["dcap"], LUT, svf, Vt, out, lo, M, D, Kc, CH, Kc // CH], shared=sh)
        print(f"wv BM{bm} WM{wm} W{warps} CH{CH}: {medir(fw):.3f}", flush=True)
    except Exception as e:
        print(f"wv BM{bm} WM{wm} W{warps}: {e}")
print(f"sumas {medir(lambda: out.sum(0, dtype=torch.int64) * 256 + lo.sum(0, dtype=torch.int64)):.3f}")
