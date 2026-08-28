"""Costo de CPU por lanzamiento. Con --enforce-eager esto esta en el camino
critico de cada forward: no hay cudagraph que lo amortice."""
import time, torch, triton
import vllm._genesis.kernels.sk06_mlp_down as m

K, N, M = 8704, 5120, 16
b = torch.randint(-127, 127, (K, N), dtype=torch.int8, device="cuda").t().contiguous().t()
a = torch.randint(-127, 127, (M, K), dtype=torch.int8, device="cuda")
asc = torch.rand(M, dtype=torch.float32, device="cuda")
bsc = torch.ones(N, dtype=torch.float32, device="cuda")
sh = torch.rand((K // 128, N // 128), dtype=torch.float32, device="cuda") * 0.01

# 1) camino publico completo (lo que corre en produccion)
m.mlp_down_gemm(a, b, asc, bsc, sh, None, torch.bfloat16)
torch.cuda.synchronize()
R = 2000
t0 = time.perf_counter()
for _ in range(R):
    m.mlp_down_gemm(a, b, asc, bsc, sh, None, torch.bfloat16)
t_pub = (time.perf_counter() - t0) / R * 1e6
torch.cuda.synchronize()

# 2) solo el _Nativo.__call__ (ctypes), sin el envoltorio de Python de arriba
cfg = m._cfg(M, N)
out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
res = torch.zeros((), dtype=torch.bfloat16, device="cuda")
args = (a, b, out, res, asc, bsc, sh, M, N, K, a.stride(0), a.stride(1),
        b.stride(0), b.stride(1), out.stride(0), out.stride(1), 0, 0,
        sh.stride(0), sh.stride(1), cfg[0], cfg[1], cfg[2], cfg[3], 128, True)
grid = (triton.cdiv(M, cfg[0]) * triton.cdiv(N, cfg[1]),)
# la eleccion de variante depende de los strides del residual: misma logica
# que _lanzar, que prueba candidatos hasta que uno acepta los horneados.
nat = None
for cand in m._POR_CFG[(cfg, True)]:
    try:
        cand(grid, *args)
        nat = cand
        break
    except ValueError:
        continue
assert nat is not None, "ninguna variante acepta estos strides"
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(R):
    nat(grid, *args)
t_nat = (time.perf_counter() - t0) / R * 1e6
torch.cuda.synchronize()

# 3) referencia: launcher propio de Triton, mismo kernel JIT
m._sk06_mlp_down_kernel[grid](
    a, b, out, res, asc, bsc, sh, M, N, K, a.stride(0), a.stride(1),
    b.stride(0), b.stride(1), out.stride(0), out.stride(1), 0, 0,
    sh.stride(0), sh.stride(1), BLOCK_M=cfg[0], BLOCK_N=cfg[1], BLOCK_K=cfg[2],
    GROUP_M=cfg[3], SHIFT_BLOCK=128, HAS_SHIFT=True, num_warps=cfg[4], num_stages=cfg[5])
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(R):
    m._sk06_mlp_down_kernel[grid](
        a, b, out, res, asc, bsc, sh, M, N, K, a.stride(0), a.stride(1),
        b.stride(0), b.stride(1), out.stride(0), out.stride(1), 0, 0,
        sh.stride(0), sh.stride(1), BLOCK_M=cfg[0], BLOCK_N=cfg[1], BLOCK_K=cfg[2],
        GROUP_M=cfg[3], SHIFT_BLOCK=128, HAS_SHIFT=True, num_warps=cfg[4], num_stages=cfg[5])
t_tri = (time.perf_counter() - t0) / R * 1e6
torch.cuda.synchronize()

# Medicion limpia del lado CPU: grid=(1,) -> un solo CTA, GPU ~2 us, asi que
# lo que se mide es puro despacho de CPU y no la cola llena de la GPU.
g1 = (1,)
nat(g1, *args)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(R):
    nat(g1, *args)
torch.cuda.synchronize()
t_cpu = (time.perf_counter() - t0) / R * 1e6
print(f"despacho puro de CPU (grid=1)  : {t_cpu:6.2f} us")
print(f"camino publico (mlp_down_gemm) : {t_pub:6.2f} us de CPU por lanzamiento")
print(f"_Nativo.__call__ (ctypes)      : {t_nat:6.2f} us")
print(f"launcher de Triton (referencia): {t_tri:6.2f} us")
print(f"\nel kernel en GPU tarda ~62 us. Lanzamientos por forward en decode:")
print(f"  ~200 GEMM + 256 quant = ~456. A {t_cpu:.1f} us de CPU eso son "
      f"{456 * t_cpu / 1000:.1f} ms de CPU por forward.")
