"""Carga DRAM-bound sostenida para muestrear relojes y potencia reales."""
import torch, triton, time
from vllm._genesis.kernels.sk06_mlp_down import _sk06_mlp_down_kernel
K, N, M = 8704, 5120, 16
b = torch.randint(-127, 127, (K, N), dtype=torch.int8, device="cuda").t().contiguous().t()
a = torch.randint(-127, 127, (M, K), dtype=torch.int8, device="cuda")
out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
res = torch.zeros((), dtype=torch.bfloat16, device="cuda")
asc = torch.rand(M, dtype=torch.float32, device="cuda")
bsc = torch.ones(N, dtype=torch.float32, device="cuda")
sh = torch.rand((K // 128, N // 128), dtype=torch.float32, device="cuda")
grid = (triton.cdiv(M, 16) * triton.cdiv(N, 128),)
def go():
    _sk06_mlp_down_kernel[grid](a, b, out, res, asc, bsc, sh, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        out.stride(0), out.stride(1), 0, 0, sh.stride(0), sh.stride(1),
        BLOCK_M=16, BLOCK_N=128, BLOCK_K=128, GROUP_M=8,
        SHIFT_BLOCK=128, HAS_SHIFT=True, num_warps=8, num_stages=3)
for _ in range(20): go()
torch.cuda.synchronize()
t0 = time.time(); n = 0
while time.time() - t0 < 25:
    for _ in range(200): go()
    torch.cuda.synchronize(); n += 200
dt = time.time() - t0
print(f"{n} lanzamientos en {dt:.1f}s -> {dt/n*1e6:.1f} us/lanzamiento, "
      f"{K*N*n/dt/1e9:.0f} GB/s sostenidos")
