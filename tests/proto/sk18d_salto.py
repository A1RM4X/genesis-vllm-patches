import sys, time, torch
exec(open("/p/sk18d_etapas.py").read().split("for bm, wm, warps")[0])
ref = None
for ss in (1,):
  for CH in (128, 256, 512):
    Kc = ((N + CH - 1) // CH) * CH
    Szc = torch.full((M, Kc), PAD, dtype=torch.int32, device=dev); Szc[:, :N] = Sz[:, :N]
    Vt = torch.zeros(D, Kc, dtype=torch.int8, device=dev); Vt[:, :N] = p["Vt"][:, :N]
    svf = torch.zeros(Kc, dtype=torch.int16, device=dev); svf[:N] = p["svf"][:N]
    kw = Kernel("sk18d_wv.cu", "sk18d_wv", defs=["-DBM=32", "-DWM=16", "-DBN=64", "-DWN=16", "-DNWARPS=8", f"-DSIN_SALTO={ss}"], warps=8)
    sh = 2 * (2 * 32 * 128 + 64 * 256)
    out = torch.empty(Kc // CH, M, D, dtype=torch.int32, device=dev); lo = torch.empty_like(out)
    fw = lambda: kw.lanzar((4, ((M + 31) // 32) * (Kc // CH)), [Szc, zm, p["mq"], p["dcap"], LUT, svf, Vt, out, lo, M, D, Kc, CH, Kc // CH], shared=sh)
    fw(); r = out.sum(0, dtype=torch.int64) * 256 + lo.sum(0, dtype=torch.int64)
    if ref is None: ref = r
    print(f"SIN_SALTO={ss} CH{CH}: {medir(fw):.3f} ms exacto={bool((r == ref).all())}", flush=True)
print(f"suma int64 {medir(lambda: out.sum(0, dtype=torch.int64) * 256 + lo.sum(0, dtype=torch.int64)):.3f}  suma int32 {medir(lambda: out.sum(0, dtype=torch.int32).to(torch.int64) * 256 + lo.sum(0, dtype=torch.int32)):.3f}")
# fraccion de keys con peso no nulo
d = (zm[:, None].to(torch.int64) - Sz[:, :N].to(torch.int64))
print("keys con d<dcap:", (d < p["dcap"][:, None]).float().mean().item())
