"""Costo de la conv causal EN ARBOL contra la de upstream (48 capas por paso).

Es el otro componente que corre una vez por capa GDN, igual que el estado. Con grafos CUDA, que es
como corre en produccion (sin ellos se mide el lanzamiento desde Python).
"""
import sys, torch, time
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
import vllm._genesis.arbol_conv as ac
dev, dt = "cuda", torch.float16
DIM, W, SPEC = 5120, 4, 8
T = SPEC + 1; SL = W - 1 + SPEC
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
cs = torch.randn(N + 4, SL, DIM, device=dev, dtype=dt).transpose(-1, -2)
w = torch.randn(DIM, W, device=dev, dtype=dt) * 0.3
sidx = torch.arange(1, N + 1, device=dev, dtype=torch.int32)
cu = torch.arange(0, (N + 1) * T, T, device=dev, dtype=torch.int32)
x = torch.randn(N * T, DIM, device=dev, dtype=dt)
acc = torch.full((N,), 3, device=dev, dtype=torch.int32)
ar = torch.arange(T, device=dev, dtype=torch.int32)
anc3_cad = torch.stack([ar - 1, ar - 2, ar - 3], 1).repeat(N, 1).contiguous()
# arbol tipico: padres [-1,0,1,2,3,2,1,6,0] -> 3 ancestros mas cercanos por token
pv = [-1, 0, 1, 2, 3, 2, 1, 6, 0]
filas = []
for t in range(T):
    c, j = [], (pv[t] if t > 0 else -1)
    while len(c) < 3:
        c.append(j); j = pv[j] if j > 0 else j - 1
    filas.append(c)
anc3_arb = torch.tensor(filas, dtype=torch.int32, device=dev).repeat(N, 1).contiguous()


def cron(f):
    for _ in range(20): f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): f()
    for _ in range(10): g.replay()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(300): g.replay()
    torch.cuda.synchronize(); return (time.time() - t0) / 300 * 1e6


def upstream():
    causal_conv1d_update(x.clone(), cs, w, None, "silu", conv_state_indices=sidx,
                         num_accepted_tokens=acc, query_start_loc=cu, max_query_len=T,
                         validate_data=False)


def arbol(anc3, fusionado=False):
    def f():
        if fusionado:                      # un solo kernel: salidas por camino + estado
            ac.salidas(x, cs, w, "silu", sidx, acc, cu, anc3, escribir_estado=True)
        else:                              # como estaba: la conv corre DOS veces
            ac.salidas(x, cs, w, "silu", sidx, acc, cu, anc3)
            causal_conv1d_update(x.clone(), cs, w, None, "silu", conv_state_indices=sidx,
                                 num_accepted_tokens=acc, query_start_loc=cu, max_query_len=T,
                                 validate_data=False)
    return f


u = cron(upstream)
a_arb = cron(arbol(anc3_arb)); a_fus = cron(arbol(anc3_arb, True))
print(f"N={N}  upstream solo (lo de produccion):      {u:7.1f} us/capa  x48 = {u * 48 / 1000:.2f} ms/paso")
print(f"      arbol, dos kernels (como estaba):       {a_arb:7.1f} us/capa  x48 = {a_arb * 48 / 1000:.2f} ms/paso"
      f"  ({(a_arb - u) * 48 / 1000:+.2f} ms)")
print(f"      arbol FUSIONADO (un kernel):            {a_fus:7.1f} us/capa  x48 = {a_fus * 48 / 1000:.2f} ms/paso"
      f"  ({(a_fus - u) * 48 / 1000:+.2f} ms)")
