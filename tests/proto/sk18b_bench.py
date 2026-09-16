"""SK-18b: softmax entero + sum(w*v) en enteros. Exactitud contra torch int64 y tiempo."""
import math, sys, time, torch
from vllm._genesis.kernels.ptx_lab import Kernel
dev = "cuda"; torch.manual_seed(0)
BM, BN, BK, ST, W = 64, 64, 128, 2, 4
SH = ST * (2 * BM * BK + BN * 2 * BK)
N = 256
k18b = Kernel("sk18b_wv.cu", "sk18b_wv", warps=W)
LUT = torch.round((1 << 24) * torch.exp(-16.0 * torch.arange(1024).float() / 1023)).to(torch.int32).to(dev)
LUT[-1] = 0

def preparar(M, Kreal, CH):
    K = ((Kreal + CH - 1) // CH) * CH
    # logits realistas: la mayoria lejos del maximo, unas pocas keys dominantes.
    # Unidad: 1 logit real = 2^16 enteros; distancia recortada en c=16 -> indice 1023.
    base = (torch.randn(M, K, device=dev) * 2.5 * (1 << 16)).to(torch.int64)
    picos = torch.randint(0, Kreal, (M, 8), device=dev)
    base.scatter_(1, picos, (6 * (1 << 16)) + torch.randint(0, 1 << 16, (M, 8), device=dev))
    Sz = base.clamp(-(1 << 26), 1 << 26).to(torch.int32)
    Sz[:, Kreal:] = -(1 << 28)
    zmax = Sz[:, :Kreal].amax(1).to(torch.int32)
    mq = torch.full((M,), 1023 * (1 << 16) // (16 * (1 << 16)), dtype=torch.int32, device=dev)   # d*mq>>16 = d_real*1023/16
    idx = (((zmax[:, None].to(torch.int64) - Sz.to(torch.int64)) * mq[:, None].to(torch.int64)) >> 16).clamp(max=1023)
    lsum = LUT.to(torch.int64)[idx].sum(1).clamp_min(1)
    invS = ((32000 << 16) // lsum).clamp(max=(1 << 31) - 1).to(torch.int32)
    svf = torch.randint(16384, 32768, (K,), dtype=torch.int16, device=dev)
    Vt = torch.randint(-127, 128, (N, K), dtype=torch.int8, device=dev)
    return K, Sz, zmax, mq, invS, svf, Vt, idx

def referencia(Sz, zmax, mq, invS, svf, Vt, idx):
    wv = (LUT.to(torch.int64)[idx] * invS[:, None].to(torch.int64)) >> 16
    w = (wv * svf[None, :].to(torch.int64)) >> 15
    hi = (w + 128) >> 8; lo = w - (hi << 8)
    Vd = Vt.to(torch.float64)
    return ((hi.double() @ Vd.t()) * 256 + lo.double() @ Vd.t()).round().to(torch.int64), w

def kernel(M, K, CH, Sz, zmax, mq, invS, svf, Vt):
    NCH = K // CH
    out = torch.zeros(NCH, M, N, dtype=torch.int32, device=dev)
    grid = ((N + BN - 1) // BN, ((M + BM - 1) // BM) * NCH)
    k18b.lanzar(grid, [Sz, zmax, mq, invS, LUT, svf, Vt, out, M, N, K, CH, NCH], shared=SH)
    return out

casos = [(24, 1000, 1024)] + [(24, 57000, ch) for ch in (512, 1024, 2048, 4096)] + [(48, 100000, 1024)]
for (M, Kreal, CH) in casos:
    K, Sz, zmax, mq, invS, svf, Vt, idx = preparar(M, Kreal, CH)
    ref, w = referencia(Sz, zmax, mq, invS, svf, Vt, idx)
    got = kernel(M, K, CH, Sz, zmax, mq, invS, svf, Vt).to(torch.int64).sum(0)
    exacto = bool((got == ref).all())
    maxdif = (got - ref).abs().max().item()
    for _ in range(3): kernel(M, K, CH, Sz, zmax, mq, invS, svf, Vt)
    torch.cuda.synchronize(); t = time.perf_counter(); n = 10
    for _ in range(n): kernel(M, K, CH, Sz, zmax, mq, invS, svf, Vt)
    torch.cuda.synchronize(); dt = (time.perf_counter() - t) / n
    print(f"M={M} K={Kreal} CH={CH}: exacto={exacto} (max dif {maxdif}, sum w/fila ~{w.sum(1).float().mean().item():.0f})  "
          f"{dt*1e3:.3f} ms  (x2 cabezas KV = {2*dt*1e3:.3f} ms por capa)", flush=True)
