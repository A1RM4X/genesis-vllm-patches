"""SK-20: cuanto cuesta cada parte (suma del residuo, cuantizacion, hilos) y cuanto los 3 de vLLM."""
import sys, torch
from vllm._genesis.kernels.ptx_lab import Kernel
from vllm import _custom_ops as ops

dev = "cuda"
torch.manual_seed(0)
torch.zeros(1, device=dev)
EPS = 1e-6


def medir(f, n=300):
    for _ in range(30): f()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(n): f()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / n * 1e3


M, K = 5, 5120
x = torch.randn(M, K, device=dev, dtype=torch.float16)
res = torch.randn(M, K, device=dev, dtype=torch.float16)
w = (1 + 0.1 * torch.randn(K, device=dev, dtype=torch.float16))
g = torch.tensor([0.000173], device=dev, dtype=torch.float32)
r_out = torch.empty(M, K, dtype=torch.float16, device=dev)
q_out = torch.empty(M, K, dtype=torch.int8, device=dev)
esc = torch.empty(M, dtype=torch.float32, device=dev)

# los tres kernels de vLLM, uno por uno
h, r = x.clone(), res.clone()
print(f"{'vLLM fused_add_rms_norm':>28}: {medir(lambda: ops.fused_add_rms_norm(h, r, w, EPS)):6.2f} us")
print(f"{'vLLM scaled_int8_quant':>28}: {medir(lambda: ops.scaled_int8_quant(h)):6.2f} us")
s0 = torch.empty(M, dtype=torch.float32, device=dev)
print(f"{'vLLM escala * global':>28}: {medir(lambda: s0.mul(g)):6.2f} us")

for hilos in (128, 256, 512):
    for hay_res in (1, 0):
        k = Kernel("sk20_norm_quant.cu", "sk20_norm_quant", defs=[f"-DHILOS={hilos}"],
                   warps=hilos // 32)
        k.cargar()
        args = [x.view(torch.int16), res.view(torch.int16), w.view(torch.int16), g.view(torch.int32),
                r_out.view(torch.int16), q_out, esc.view(torch.int32), K, x.stride(0), hay_res]
        t = medir(lambda: k.lanzar((M, 1), args, 2 * K))
        print(f"      SK20 HILOS={hilos:3d} res={hay_res}: {t:6.2f} us")
