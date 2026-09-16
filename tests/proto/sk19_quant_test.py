"""SK-19/quant: exactitud y tiempo contra el camino de vLLM (per_token_quant_int8 + mul)."""
import time, torch
from vllm._genesis.kernels.ptx_lab import Kernel
from vllm import _custom_ops as ops

dev = "cuda"
torch.manual_seed(0)
torch.zeros(1, device=dev)          # inicializa el contexto antes de cargar el modulo PTX
import sys
HILOS = int(sys.argv[1]) if len(sys.argv) > 1 else 128
k = Kernel("sk19_quant.cu", "sk19_quant", defs=[f"-DHILOS={HILOS}"], warps=HILOS // 32)
k.cargar()


def medir(f, n=300):
    """Tiempo de GPU con eventos: el reloj de pared mide el overhead de Python, no el kernel."""
    for _ in range(20): f()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(n): f()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / n * 1e3


print(f"HILOS={HILOS}")
print(f"{'M':>6} {'K':>6} {'dif int8':>10} {'max dif':>8} {'err escala':>11} "
      f"{'vLLM us':>8} {'SK19 us':>8}")
for M, K in ((5, 5120), (5, 17408), (40, 5120), (512, 5120), (8192, 5120)):
    x = (torch.randn(M, K, device=dev, dtype=torch.float16) * 3)
    x[:, ::97] *= 12                                   # algunos outliers, como en la realidad
    g = torch.tensor([0.000173], device=dev, dtype=torch.float32)

    ref_q, ref_s = ops.scaled_int8_quant(x)[:2]
    ref_s = (ref_s.float().view(-1) * g)

    xq = torch.empty(M, K, dtype=torch.int8, device=dev)
    esc = torch.empty(M, dtype=torch.float32, device=dev)
    k.lanzar((M, 1), [x.view(torch.int16), g.view(torch.int32), xq, esc.view(torch.int32),
                      K, x.stride(0)])
    torch.cuda.synchronize()

    dif = (xq.int() - ref_q.int()).abs()
    err = ((esc - ref_s).abs() / ref_s.abs().clamp_min(1e-30)).max().item()
    t_v = medir(lambda: (lambda a: a[1].float().view(-1) * g)(ops.scaled_int8_quant(x)))
    t_s = medir(lambda: k.lanzar((M, 1), [x.view(torch.int16), g.view(torch.int32), xq,
                                          esc.view(torch.int32), K, x.stride(0)]))
    print(f"{M:6d} {K:6d} {100*(dif > 0).float().mean():9.3f}% {int(dif.max()):8d} "
          f"{100*err:10.4f}% {t_v:8.1f} {t_s:8.1f}")
