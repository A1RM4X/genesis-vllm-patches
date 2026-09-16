import time, torch
from vllm._genesis.kernels.ptx_lab import Kernel
dev = "cuda"; M, N, K = 24, 57000, 256; SH = 2 * (32 * 128 + 64 * 256)
A_ = torch.randint(-127, 128, (M, K), dtype=torch.int8, device=dev); B_ = torch.randint(-127, 128, (N, K), dtype=torch.int8, device=dev)
skf = torch.full((N,), 32767, dtype=torch.int16, device=dev); lim = torch.full((M,), N, dtype=torch.int32, device=dev)
ref = None
for fl in (0, 1, 2):
    k = Kernel("sk18d_qk.cu", "sk18d_qk", defs=[f"-DFLUSH={fl}"], warps=8)
    out = torch.empty(M, N, dtype=torch.int32, device=dev)
    f = lambda: k.lanzar(((N + 63) // 64, 1), [A_, B_, skf, lim, out, M, N, K, N, 1], shared=SH)
    for _ in range(3): f()
    if ref is None: ref = out.clone()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(30): f()
    torch.cuda.synchronize()
    print(f"FLUSH={fl}: {(time.perf_counter()-t)/30*1e3:.3f} ms  igual={bool((out==ref).all())}", flush=True)
