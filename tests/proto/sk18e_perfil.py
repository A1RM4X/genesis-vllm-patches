import sys, time, torch
sys.argv = [sys.argv[0], "nada"]
src = open("/p/sk18e_test.py").read().split("q, k, v = A.cargar(35)")[0]
exec(src)
q, k, v = A.cargar(35); Nt = k.shape[0]; kr = A.rotar_ptx(k); qr = A.rotar_ptx(q[Nt - 4:])
lim = (Nt - 4 + torch.arange(4, device=dev)).repeat_interleave(G).to(torch.int32)
p = preparar([qr[:, h * G:(h + 1) * G].reshape(-1, D) for h in range(2)], [kr[:, h] for h in range(2)], [v[:, h] for h in range(2)], lim, 1024)
Vpag = p["Vt"].view(2, D, -1).permute(0, 2, 1).reshape(2, -1, 64, D).permute(0, 1, 3, 2).contiguous()   # [NH, NPAG, 256, 64]
oh0 = None
for dg, kd, vp in ((0, 1, 0), (0, 0, 0), (0, 1, 1), (0, 0, 1)):
    kk = Kernel("sk18e_stream.cu", "sk18e_stream", defs=[f"-DQA={QA}", f"-DQB={QB}", f"-DDIAG={dg}", f"-DKDUP={kd}", f"-DVPAG={vp}"], warps=4)
    VV = Vpag if vp else p["Vt"]
    f = lambda: kk.lanzar((p["NCH"], p["NH"] * p["MB"] // 16), [p["Q"], p["Kc"], VV, p["skf"], p["svf"], p["lim"], p["mqb"], p["dcap"],
                        p["oh"], p["ol"], p["om"], p["os"], p["MB"], p["N"], p["K"], p["CHK"], p["NCH"], p["NH"]], shared=SH)
    for _ in range(3): f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(20): f()
    igual = None
    if oh0 is None: oh0 = p["oh"].clone()
    else: igual = bool((p["oh"] == oh0).all())
    torch.cuda.synchronize(); print(f"KDUP={kd} VPAG={vp} igual={igual} N={Nt} DIAG={dg}: {(time.perf_counter()-t)/20*1e3:.3f} ms", flush=True)
