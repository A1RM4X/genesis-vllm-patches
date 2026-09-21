"""Costo del GDN en arbol contra la cadena, por capa y por paso (6 pedidos, K=8)."""
import os, sys, types, torch
import vllm._genesis.gdn_cinta as g
import vllm._genesis.arbol_borrador as ab
dev, dt = "cuda", torch.float16
H, HV, K, V, SPEC = 8, 24, 128, 128, 8
T = SPEC + 1
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
S = 1 + N
h = (torch.randn(S, HV * V * K, device=dev, dtype=dt) * 0.05).view(S, HV, V, K)
A_log = torch.randn(HV, device=dev) * 0.5; dt_bias = torch.randn(HV, device=dev) * 0.5
cols = torch.arange(1, S, device=dev, dtype=torch.int32).view(N, 1)
g._init_slots(8, dev)
layer = types.SimpleNamespace(tp_size=2, num_k_heads=2 * H, num_v_heads=2 * HV, head_k_dim=K,
                              head_v_dim=V, num_spec=SPEC, prefix="t")
g.enlazar(layer, dev)
slots = torch.arange(1, N + 1, device=dev, dtype=torch.int32)
cu = torch.arange(0, (N + 1) * T, T, device=dev, dtype=torch.int32)
q = torch.randn(1, N * T, H, K, device=dev, dtype=dt); k = torch.randn_like(q)
v = torch.randn(1, N * T, HV, V, device=dev, dtype=dt)
a = torch.randn(N * T, HV, device=dev, dtype=dt); b = torch.randn_like(a)
acc = torch.full((N,), 3, device=dev, dtype=torch.int32)
anc = torch.zeros(N * T, dtype=torch.int32, device=dev)
FORMAS = {   # padres de [ancla]+8 nodos, en preorden
    "cadena":            [-1, 0, 1, 2, 3, 4, 5, 6, 7],
    "tipico (3 ramas)":  [-1, 0, 1, 2, 3, 2, 1, 6, 0],
    "ancho (estrella)":  [-1, 0, 0, 0, 0, 0, 0, 0, 0],
    "peor (2 cadenas)":  [-1, 0, 1, 2, 3, 0, 5, 6, 7],
}
def medir():
    """Con GRAFO CUDA: sin el, se mide el lanzamiento desde Python (la CPU es el cuello) y un
    kernel de mas parece costar 40 us que en produccion no existen, porque el paso de decode
    corre capturado."""
    for _ in range(20):
        g.spec_update(layer, A_log, a, b, dt_bias, q, k, v, h, cu, cols, acc, slots)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        g.spec_update(layer, A_log, a, b, dt_bias, q, k, v, h, cu, cols, acc, slots)
    for _ in range(10):
        gr.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(300):
        gr.replay()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / 300
g.fijar_ancestros(None)
base = medir()
print(f"N={N}  produccion (cadena, _k_spec): {base*1000:7.1f} us/capa   x48 capas = {base*48:.2f} ms/paso")
g.fijar_ancestros(anc)
for nombre, p in FORMAS.items():
    anc.copy_(ab.bits_ancestros(torch.tensor([p]))[0].repeat(N).to(dev))
    x = medir()
    print(f"      arbol {nombre:18s}: {x*1000:7.1f} us/capa   x48 = {x*48:.2f} ms/paso  ({(x-base)*48:+.2f} ms)")
