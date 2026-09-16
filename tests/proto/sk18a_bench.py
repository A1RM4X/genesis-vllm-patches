"""SK-18a: GEMM Q.K int8 -> int32 en la forma de decode. Exactitud y tiempo."""
import sys, time, torch
from vllm._genesis.kernels.ptx_lab import Kernel
dev = "cuda"; torch.manual_seed(0)
BM, BN, BK, ST = 256, 64, 128, 2
SH = ST * (BM * BK + BN * 2 * BK)
k = Kernel("sk18a_qk_i32.cu", "sk18a_qk_i32", warps=8)
def qk(A, B):
    M, K = A.shape; N = B.shape[0]
    out = torch.empty(M, N, dtype=torch.int32, device=dev)
    k.lanzar(((N + BN - 1) // BN, (M + BM - 1) // BM), [A, B, out, M, N, K], shared=SH)
    return out
for (M, N) in [(24, 1000), (24, 57001), (48, 100000)]:
    A = torch.randint(-127, 128, (M, 256), dtype=torch.int8, device=dev)
    B = torch.randint(-127, 128, (N, 256), dtype=torch.int8, device=dev)
    ref = (A.double() @ B.double().t()).to(torch.int64)
    got = qk(A, B).to(torch.int64)
    for _ in range(3): qk(A, B)
    torch.cuda.synchronize(); t = time.perf_counter(); n = 20
    for _ in range(n): qk(A, B)
    torch.cuda.synchronize(); dt = (time.perf_counter() - t) / n
    print(f"M={M} N={N}: exacto={bool((got == ref).all())}  {dt*1e3:.3f} ms  (x2 cabezas KV = {2*dt*1e3:.3f} ms por capa)", flush=True)
