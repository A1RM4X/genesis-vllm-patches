"""SK-18c: Q.K en mma s4 (nibbles) -> int32. Exactitud contra torch y tiempo vs s8."""
import time, torch
from vllm._genesis.kernels.ptx_lab import Kernel
dev = "cuda"; torch.manual_seed(0)
SH = 2 * (256 * 128 + 64 * 2 * 128)
k8 = Kernel("sk18a_qk_i32.cu", "sk18a_qk_i32", defs=["-DS4=0"], warps=8)
k4 = Kernel("sk18a_qk_i32.cu", "sk18a_qk_i32", defs=["-DS4=1"], warps=8)
def nib(x):
    u = (x & 0xF).to(torch.uint8)
    return (u[:, 0::2] | (u[:, 1::2] << 4)).view(torch.int8).contiguous()
def correr(k, A, B, Kb):
    M, N = A.shape[0], B.shape[0]
    out = torch.empty(M, N, dtype=torch.int32, device=dev)
    k.lanzar(((N + 63) // 64, (M + 255) // 256), [A, B, out, M, N, Kb], shared=SH)
    return out
for (M, N) in [(24, 1000), (288, 22464), (24, 57000)]:
    A4 = torch.randint(-7, 8, (M, 256), dtype=torch.int8, device=dev); B4 = torch.randint(-7, 8, (N, 256), dtype=torch.int8, device=dev)
    ref = (A4.double() @ B4.double().t()).to(torch.int64)
    got = correr(k4, nib(A4), nib(B4), 128).to(torch.int64)
    A8 = torch.randint(-127, 128, (M, 256), dtype=torch.int8, device=dev); B8 = torch.randint(-127, 128, (N, 256), dtype=torch.int8, device=dev)
    A4n, B4n = nib(A4), nib(B4)
    t = {}
    for nombre, f in (("s8", lambda: correr(k8, A8, B8, 256)), ("s4", lambda: correr(k4, A4n, B4n, 128))):
        for _ in range(3): f()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(10): f()
        torch.cuda.synchronize(); t[nombre] = (time.perf_counter() - t0) / 10
    print(f"M={M} N={N}: s4 exacto={bool((got == ref).all())}  s8 {t['s8']*1e3:.3f} ms  s4 {t['s4']*1e3:.3f} ms  ({t['s8']/t['s4']:.2f}x)", flush=True)
