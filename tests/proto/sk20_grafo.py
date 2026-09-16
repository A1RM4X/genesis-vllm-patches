"""SK-20 vs los 3 kernels de vLLM, medido con grafos CUDA (el lanzar de Python cuesta ~10 us y
tapa el tiempo real del kernel)."""
import sys, torch
from vllm._genesis.kernels.ptx_lab import Kernel
from vllm import _custom_ops as ops
dev = "cuda"; torch.manual_seed(0); torch.zeros(1, device=dev)
EPS = 1e-6; REP = 50

def por_grafo(f, rep=REP, n=50):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): f()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(rep): f()
    for _ in range(3): g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(n): g.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / (n * rep) * 1e3

print(f"{'M':>6} {'K':>6} {'vLLM 3 kernels':>15} {'SK20':>8} {'ganancia':>9}")
for M, K in ((5, 5120), (40, 5120), (512, 5120), (8192, 5120)):
    x = torch.randn(M, K, device=dev, dtype=torch.float16)
    res = torch.randn(M, K, device=dev, dtype=torch.float16)
    w = (1 + 0.1*torch.randn(K, device=dev, dtype=torch.float16))
    g_ = torch.tensor([0.000173], device=dev, dtype=torch.float32)
    r_out = torch.empty(M, K, dtype=torch.float16, device=dev)
    q_out = torch.empty(M, K, dtype=torch.int8, device=dev)
    esc = torch.empty(M, dtype=torch.float32, device=dev)
    h, r = x.clone(), res.clone()
    qb = torch.empty(M, K, dtype=torch.int8, device=dev)
    sb = torch.empty(M, 1, dtype=torch.float32, device=dev)
    def vllm():
        ops.fused_add_rms_norm(h, r, w, EPS)
        ops.scaled_int8_quant(h, scale=None, azp=None)
        sb.mul_(g_)
    tv = por_grafo(vllm)
    best = None
    for hilos in (128, 256, 512, 1024):
        k = Kernel("sk20_norm_quant.cu", "sk20_norm_quant", defs=[f"-DHILOS={hilos}"],
                   warps=hilos//32)
        k.cargar()
        args = [x.view(torch.int16), res.view(torch.int16), w.view(torch.int16),
                g_.view(torch.int32), r_out.view(torch.int16), q_out, esc.view(torch.int32),
                K, x.stride(0), 1]
        t = por_grafo(lambda: k.lanzar((M, 1), args, 4*K))
        best = (t, hilos) if best is None or t < best[0] else best
    print(f"{M:6d} {K:6d} {tv:14.2f}u {best[0]:7.2f}u (h={best[1]}) {tv/best[0]:8.2f}x")
