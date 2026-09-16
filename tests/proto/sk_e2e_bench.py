"""Bench de punta a punta del gate_up+SiLU con GPU libre, forma real de prefill.

  cutlass W8A8:  scaled_int8_quant + 2x cutlass_scaled_mm + SiLU*mul   (PN118)
  W4A4 g256:     PTX Q8b + SK-17 (Hadamard INT8) + Q4  ->  SK-15
  W4A4 token:    PTX Q8b + SK-17 + Q4 (escalas promediadas por token) -> SK-14 s4
  SK-16:         Hadamard fp16 por bloques y densa 5120 contra cuBLAS
"""
import sys, time, torch
from vllm import _custom_ops as ops
from vllm._genesis.kernels import sk15_w4a4g as S15, sk17_act_int8 as S17, sk16_gemm_f16 as S16
from vllm._genesis.kernels.ptx_lab import Kernel
M = int(sys.argv[1]) if len(sys.argv) > 1 else 7488
K, N = 5120, 8704
torch.manual_seed(0)
x = (torch.randn(M, K, device="cuda") * 2).half()
Wg = torch.randn(N, K, device="cuda") * 0.02; Wu = torch.randn(N, K, device="cuda") * 0.02
gn, sg = S15.preparar_pesos(Wg); un, su = S15.preparar_pesos(Wu)
sg1, su1 = sg.mean(1).contiguous(), su.mean(1).contiguous()
def w8(W):
    s = W.abs().amax(1).clamp_min(1e-8) / 127
    return (W / s[:, None]).round().clamp(-127, 127).to(torch.int8), s
wg8, sg8 = w8(Wg); wu8, su8 = w8(Wu)
del Wg, Wu; torch.cuda.empty_cache()
k14 = Kernel("sk14_gateup_w4a4.cu", "sk14_gateup_w4a4", defs=["-DS4=1"], warps=8)
SH = 2 * (256 * 128 + 64 * 2 * 128)
out = torch.empty(M, N, dtype=torch.float16, device="cuda")
grid = ((N + 63) // 64, (M + 255) // 256)

def cutlass():
    xq, xs, _ = ops.scaled_int8_quant(x, symmetric=True)
    g = ops.cutlass_scaled_mm(xq, wg8.t(), xs, sg8.view(1, -1), torch.float16)
    u = ops.cutlass_scaled_mm(xq, wu8.t(), xs, su8.view(1, -1), torch.float16)
    return torch.nn.functional.silu(g) * u

def w4a4_g256():
    q, s = S17.act_int4(x)
    return S15.gemm(q, s, gn, sg, un, su)

def w4a4_token():
    q, s = S17.act_int4(x)
    sa1 = s.mean(1).contiguous()
    k14.lanzar(grid, [q, gn, un, sa1, sg1, su1, out, M, N, K // 2], shared=SH)
    return out

def act_ptx():
    return S17.act_int4(x)

xh = x.view(-1, 256)
H = S16.hadamard(256, "cuda")
casos = {"cutlass W8A8 total": cutlass, "W4A4 g256 (PTX act + SK-15)": w4a4_g256,
         "W4A4 token (PTX act + SK-14)": w4a4_token, "solo act PTX (Q8b+SK17+Q4)": act_ptx,
         "Hadamard bloques SK-16": lambda: S16.gemm(xh, H), "Hadamard bloques cuBLAS": lambda: xh @ H.t()}
for f in casos.values():
    for _ in range(2): f()
torch.cuda.synchronize(); res = {k: [] for k in casos}
for _ in range(7):
    for k, f in casos.items():
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(4): f()
        torch.cuda.synchronize(); res[k].append((time.perf_counter() - t0) / 4)
for k, v in res.items():
    print(f"  M={M} {k:32s} {sorted(v)[3]*1e3:8.2f} ms", flush=True)
