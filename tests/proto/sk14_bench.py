"""SK-14: GEMM gate_up+SiLU con carga real, s8 (layout SK-12) contra s4 (mma k64)."""
import time, torch
from vllm._genesis.kernels.ptx_lab import Kernel
dev = "cuda"; torch.manual_seed(0)
BM, BN, BK, ST, W = 256, 64, 128, 2, 8
SHARED = ST * (BM * BK + BN * 2 * BK)

def nib_bytes(x):                          # [R, Kelem] int4 -> [R, Kelem/2] int8 (nibble par = bits bajos)
    u = (x & 0xF).to(torch.uint8)
    return (u[:, 0::2] | (u[:, 1::2] << 4)).view(torch.int8).contiguous()

def correr(k, A, Wg, Wu, sa, sg, su, M, N, Kb):
    out = torch.empty((M, N), dtype=torch.float16, device=dev)
    k.lanzar(((N + BN - 1) // BN, (M + BM - 1) // BM), [A, Wg, Wu, sa, sg, su, out, M, N, Kb], shared=SHARED)
    return out

def ref(A, Wg, Wu, sa, sg, su):
    a = A.float()
    g = (a @ Wg.float().t()) * sa[:, None] * sg[None, :]
    u = (a @ Wu.float().t()) * sa[:, None] * su[None, :]
    return (torch.nn.functional.silu(g) * u)

def cron(fn, it=10):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / it

for (M, Ke, N) in [(512, 1024, 512), (2048, 5120, 8704), (8192, 5120, 8704)]:
    res = {}
    for nombre, s4 in (("s8", 0), ("s4", 1)):
        k = Kernel("sk14_gateup_w4a4.cu", "sk14_gateup_w4a4", defs=[f"-DS4={s4}"], warps=W)
        lo, hi = (-8, 8) if s4 else (-127, 128)
        A = torch.randint(lo, hi, (M, Ke), dtype=torch.int8, device=dev)
        Wg = torch.randint(lo, hi, (N, Ke), dtype=torch.int8, device=dev)
        Wu = torch.randint(lo, hi, (N, Ke), dtype=torch.int8, device=dev)
        sa = torch.full((M,), 0.01, device=dev); sg = torch.full((N,), 0.01, device=dev); su = torch.full((N,), 0.01, device=dev)
        if s4:
            Ab, Wgb, Wub, Kb = nib_bytes(A), nib_bytes(Wg), nib_bytes(Wu), Ke // 2
        else:
            Ab, Wgb, Wub, Kb = A.contiguous(), Wg.contiguous(), Wu.contiguous(), Ke
        o = correr(k, Ab, Wgb, Wub, sa, sg, su, M, N, Kb).float()
        r = ref(A, Wg, Wu, sa, sg, su)
        err = ((o - r).norm() / r.norm()).item()
        dt = cron(lambda: correr(k, Ab, Wgb, Wub, sa, sg, su, M, N, Kb))
        res[nombre] = dt
        print(f"M={M:5d} K={Ke} N={N:5d} {nombre}: err {err:.2e}  {dt*1e3:8.2f} ms  {4*M*Ke*N/dt/1e12:7.1f} TOPS", flush=True)
        del A, Wg, Wu, Ab, Wgb, Wub; torch.cuda.empty_cache()
    print(f"   s4 vs s8: {res['s8']/res['s4']:.2f}x", flush=True)
