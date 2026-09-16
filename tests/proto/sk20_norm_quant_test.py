"""SK-20/norm_quant: exactitud contra el camino de vLLM (fused_add_rms_norm + quant + mul)."""
import sys, time, torch
from vllm._genesis.kernels.ptx_lab import Kernel
from vllm import _custom_ops as ops

dev = "cuda"
torch.manual_seed(0)
torch.zeros(1, device=dev)
HILOS = int(sys.argv[1]) if len(sys.argv) > 1 else 256
k = Kernel("sk20_norm_quant.cu", "sk20_norm_quant", defs=[f"-DHILOS={HILOS}"], warps=HILOS // 32)
k.cargar()
EPS = 1e-6


def referencia(x, res, w):
    """Lo que hace vLLM: residual, RMSNorm en fp32 y cuantizacion por token."""
    r = (x.float() + res.float()) if res is not None else x.float()
    var = r.pow(2).mean(-1, keepdim=True)
    y = (r * torch.rsqrt(var + EPS)) * w.float()
    esc = y.abs().amax(-1, keepdim=True) / 127.0
    q = torch.round(y / esc).clamp(-127, 127).to(torch.int8)
    return r.half(), q, esc.float()


def medir(f, n=200):
    for _ in range(20): f()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(n): f()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / n * 1e3


def camino_vllm(x, res, w, g):
    """Los tres kernels que reemplazamos."""
    h, r = x.clone(), res.clone()
    ops.fused_add_rms_norm(h, r, w, EPS)
    q, s = ops.scaled_int8_quant(h)[:2]
    return h, q, s.float().view(-1) * g


print(f"HILOS={HILOS}")
print(f"{'M':>6} {'K':>6} {'res ulp':>9} {'int8 dif':>9} {'max':>4} {'err escala':>11} "
      f"{'vLLM us':>8} {'SK20 us':>8}")
for M, K in ((5, 5120), (40, 5120), (512, 5120), (8192, 5120)):
    x = torch.randn(M, K, device=dev, dtype=torch.float16)
    res = torch.randn(M, K, device=dev, dtype=torch.float16)
    w = (1 + 0.1 * torch.randn(K, device=dev, dtype=torch.float16))
    g = torch.tensor([0.000173], device=dev, dtype=torch.float32)

    r_ref, q_ref, s_ref = referencia(x, res, w)
    s_ref = s_ref.view(-1) * g

    r_out = torch.empty(M, K, dtype=torch.float16, device=dev)
    q_out = torch.empty(M, K, dtype=torch.int8, device=dev)
    esc = torch.empty(M, dtype=torch.float32, device=dev)
    args = [x.view(torch.int16), res.view(torch.int16), w.view(torch.int16), g.view(torch.int32),
            r_out.view(torch.int16), q_out, esc.view(torch.int32), K, x.stride(0), 1]
    k.lanzar((M, 1), args, 4 * K)
    torch.cuda.synchronize()

    # diferencia en ulps de fp16 (los bits ordenados restados)
    ulp = (r_out.view(torch.int16).int() - r_ref.view(torch.int16).int()).abs()
    dq = (q_out.int() - q_ref.int()).abs()
    es = ((esc - s_ref).abs() / s_ref.abs().clamp_min(1e-30)).max().item()
    t = medir(lambda: k.lanzar((M, 1), args, 4 * K))
    tv = medir(lambda: camino_vllm(x, res, w, g))
    print(f"{M:6d} {K:6d} {100*(ulp > 0).float().mean():8.3f}% {100*(dq > 0).float().mean():8.3f}% "
          f"{int(dq.max()):4d} {100*es:10.4f}% {tv:8.1f} {t:8.1f}")
