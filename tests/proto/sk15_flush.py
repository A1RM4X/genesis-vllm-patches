"""SK-15: costo del traspaso segun cada cuantos tiles se vuelca (GPU libre)."""
import time, torch
from vllm._genesis.kernels.ptx_lab import Kernel
from vllm._genesis.kernels import sk15_w4a4g as S
M, K, N = 7488, 5120, 8704; Kb = K // 2
SH = 2 * (256 * 128 + 64 * 2 * 128)
torch.manual_seed(0)
x = torch.randn(M, K, device="cuda") * 3
an, sa = S.preparar_act(x); del x
W = torch.randn(N, K, device="cuda") * 0.02
gn, sg = S.preparar_pesos(W); un, su = S.preparar_pesos(W); del W
out = torch.empty(M, N, dtype=torch.float16, device="cuda")
grid = ((N + 63) // 64, (M + 255) // 256)
casos = {}
k14 = Kernel("sk14_gateup_w4a4.cu", "sk14_gateup_w4a4", defs=["-DS4=1"], warps=8)
sa1, sg1, su1 = sa.mean(1).contiguous(), sg.mean(1).contiguous(), su.mean(1).contiguous()
casos["SK-14 (sin volcado)"] = lambda: k14.lanzar(grid, [an, gn, un, sa1, sg1, su1, out, M, N, Kb], shared=SH)
for c in (1, 2, 4, 10, 20):
    k = Kernel("sk15_gateup_w4a4g.cu", "sk15_gateup_w4a4g", defs=["-DSK15_DIAG=3", f"-DFLUSH_CADA={c}"], warps=32)
    casos[f"volcado entero cada {c}"] = (lambda k=k: k.lanzar(grid, [an, gn, un, sa, sg, su, out, M, N, Kb], shared=SH))
for f in casos.values():
    for _ in range(2): f()
torch.cuda.synchronize(); res = {k: [] for k in casos}
for _ in range(5):
    for k, f in casos.items():
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(3): f()
        torch.cuda.synchronize(); res[k].append((time.perf_counter() - t0) / 3)
for k, v in res.items():
    print(f"  {k:28s} {sorted(v)[2]*1e3:8.2f} ms", flush=True)
