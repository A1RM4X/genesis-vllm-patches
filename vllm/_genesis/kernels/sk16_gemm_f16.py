# SPDX-License-Identifier: Apache-2.0
"""SK-16 — GEMM fp16 en PTX para rotar activaciones (Hadamard por bloques, densa, WUSH).

``y = x @ R^T``. Kernel en ``kernels/cuda/sk16_gemm_f16.cu`` (esqueleto SK-12,
instruccion mma m16n8k16 f16). ``--check`` valida contra torch y ``--bench``
mide contra cuBLAS (torch.matmul fp16) en las dos formas de uso.
"""
from __future__ import annotations
import math, os, re, sys, time
import torch
from vllm._genesis.kernels.ptx_lab import Kernel

_DEFS = os.environ.get("GENESIS_SK16_DEFS", "").split()
def _def(n, d):
    m = re.search(rf"-D{n}=(\d+)", " ".join(_DEFS)); return int(m.group(1)) if m else d
BM, BN, BK, STAGES, NWARPS = _def("BM", 256), _def("BN", 64), _def("BK", 128), _def("STAGES", 2), _def("NWARPS", 8)
SHARED = STAGES * (BM * BK + BN * 2 * BK)
_k = None

def kernel():
    global _k
    if _k is None:
        _k = Kernel("sk16_gemm_f16.cu", "sk16_gemm_f16", defs=_DEFS, warps=NWARPS)
    return _k

def gemm(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """x fp16 [M, K], R fp16 [N, K] -> fp16 [M, N]. K*2 multiplo de BK."""
    M, K = x.shape; N = R.shape[0]
    assert (2 * K) % BK == 0 and x.dtype == R.dtype == torch.float16
    out = torch.empty((M, N), dtype=torch.float16, device=x.device)
    grid = ((N + BN - 1) // BN, (M + BM - 1) // BM)
    kernel().lanzar(grid, [x.contiguous(), R.contiguous(), out, M, N, 2 * K], shared=SHARED)
    return out

def hadamard(n, device):
    H = torch.ones(1, 1, device=device, dtype=torch.float64)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(n)).half()

def check():
    ok = True
    for (M, K, N) in [(64, 64, 64), (300, 256, 256), (1000, 5120, 700), (7488 * 20 // 20, 256, 256)]:
        torch.manual_seed(0)
        x = torch.randn(M, K, device="cuda").half(); R = (torch.randn(N, K, device="cuda") / math.sqrt(K)).half()
        ref = x.float() @ R.float().t(); got = gemm(x, R).float()
        e = ((got - ref).norm() / ref.norm()).item(); ok &= e < 2e-3
        print(f"  M={M:6d} K={K:5d} N={N:5d} err {e:.2e} {'OK' if e < 2e-3 else 'FALLA'}", flush=True)
    return ok

def bench(rondas=7, it=8):
    torch.manual_seed(0)
    casos = {}
    xh = torch.randn(7488 * 20, 256, device="cuda").half(); H = hadamard(256, "cuda")
    casos["Hadamard bloques: SK-16"] = lambda: gemm(xh, H)
    casos["Hadamard bloques: cuBLAS"] = lambda: xh @ H.t()
    xd = torch.randn(7488, 5120, device="cuda").half(); Q = (torch.randn(5120, 5120, device="cuda") / 72).half()
    casos["densa 5120: SK-16"] = lambda: gemm(xd, Q)
    casos["densa 5120: cuBLAS"] = lambda: xd @ Q.t()
    for f in casos.values():
        for _ in range(3): f()
    torch.cuda.synchronize(); res = {k: [] for k in casos}
    for _ in range(rondas):
        for k, f in casos.items():
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(it): f()
            torch.cuda.synchronize(); res[k].append((time.perf_counter() - t0) / it)
    for k, v in res.items():
        print(f"  {k:28s} {sorted(v)[len(v)//2]*1e3:8.2f} ms", flush=True)

if __name__ == "__main__":
    if "--bench" in sys.argv: bench()
    else: raise SystemExit(0 if check() else 1)
