"""Aisla por que SK-15 es lento: geometria de warps vs volcado por grupo."""
import time, torch
from vllm._genesis.kernels.ptx_lab import Kernel
from vllm._genesis.kernels import sk15_w4a4g as S
dev = "cuda"; torch.manual_seed(0)
M, K, N = 7488, 5120, 8704
Kb = K // 2
BM, BN, BK, ST = 256, 64, 128, 2
SH = ST * (BM * BK + BN * 2 * BK)
x = torch.randn(M, K, device=dev) * 3
Wg = torch.randn(N, K, device=dev) * 0.02; Wu = torch.randn(N, K, device=dev) * 0.02
an, sa = S.preparar_act(x); gn, sg = S.preparar_pesos(Wg); un, su = S.preparar_pesos(Wu)
sa1 = sa.mean(1).contiguous(); sg1 = sg.mean(1).contiguous(); su1 = su.mean(1).contiguous()
out = torch.empty(M, N, dtype=torch.float16, device=dev)
grid = ((N + BN - 1) // BN, (M + BM - 1) // BM)

def k14(w, wm, wn):
    k = Kernel("sk14_gateup_w4a4.cu", "sk14_gateup_w4a4", defs=["-DS4=1", f"-DNWARPS={w}", f"-DWM={wm}", f"-DWN={wn}"], warps=w)
    return lambda: k.lanzar(grid, [an, gn, un, sa1, sg1, su1, out, M, N, Kb], shared=SH)

def k15(w, wm, wn, extra=()):
    k = Kernel("sk15_gateup_w4a4g.cu", "sk15_gateup_w4a4g", defs=[f"-DNWARPS={w}", f"-DWM={wm}", f"-DWN={wn}", *extra], warps=w)
    return lambda: k.lanzar(grid, [an, gn, un, sa, sg, su, out, M, N, Kb], shared=SH)

casos = {
    "SK-14 s4 8 warps 64x32 (original)": k14(8, 64, 32),
    "SK-15 volcado entero shift const": k15(32, 32, 16, ["-DSK15_DIAG=3"]),
    "SK-15 volcado entero sin syncthreads": k15(32, 32, 16, ["-DSK15_DIAG=5"]),
}
for f in casos.values():
    for _ in range(2): f()
torch.cuda.synchronize()
res = {k: [] for k in casos}
for _ in range(5):
    for k, f in casos.items():
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(4): f()
        torch.cuda.synchronize(); res[k].append((time.perf_counter() - t0) / 4)
for k, v in res.items():
    print(f"  {k:40s} {sorted(v)[2]*1e3:8.2f} ms", flush=True)
