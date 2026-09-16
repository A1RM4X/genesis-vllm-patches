# SPDX-License-Identifier: Apache-2.0
"""SK-17 — camino de activaciones W4A4 en INT8 hasta el int4:

    x fp16 --SK-17-Q8b--> xq int8, sx                (por token, hilo por bloque)
    xq por bloques de 256 x S(+-1) --SK-17 mma s8--> y int16    (Hadamard exacta)
    y --SK-17-Q4--> nibbles int4 + escalas por grupo (sx/16 incluida)

``--check`` compara contra torch (Hadamard float + int4 g256 de SK-15) y
``--bench`` mide contra el mismo camino en torch y contra cutlass W8A8 total.
"""
from __future__ import annotations
import math, sys, time
import torch
from vllm._genesis.kernels.ptx_lab import Kernel

G = 256
BM, BN, BK, STAGES, NWARPS = 256, 64, 128, 2, 8
SHARED = STAGES * (BM * BK + BN * 2 * BK)
_had = None
_q4 = None
_S = {}


def _kh():
    global _had
    if _had is None:
        _had = Kernel("sk17_had_int8.cu", "sk17_had_int8", warps=NWARPS)
    return _had


_q8 = None


def _k8():
    global _q8
    if _q8 is None:
        _q8 = Kernel("sk17_q8.cu", "sk17_q8", defs=["-DNWARPS=8"], warps=8)
    return _q8


def _kq(w=8):
    global _q4
    if _q4 is None:
        _q4 = Kernel("sk17_q4_i16.cu", "sk17_q4_i16", defs=[f"-DNWARPS={w}"], warps=w)
    return _q4


def signos(device) -> torch.Tensor:
    key = str(device)
    if key not in _S:
        H = torch.ones(1, 1, device=device, dtype=torch.int8)
        while H.shape[0] < G:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        _S[key] = H.contiguous()
    return _S[key]


def act_int4(x: torch.Tensor):
    """x fp16 [M, K] -> (nibbles int8 [M, K/2], escalas fp32 [M, K/256])."""
    M, K = x.shape
    NG = K // G
    x = x.contiguous()
    xq, sx = q8b(x)
    y = torch.empty((M * NG, G), dtype=torch.int16, device=x.device)
    A = xq.view(M * NG, G)
    _kh().lanzar(((G + BN - 1) // BN, (M * NG + BM - 1) // BM),
                 [A, signos(x.device), y, M * NG, G, G], shared=SHARED)
    q = torch.empty((M, K // 2), dtype=torch.int8, device=x.device)
    s = torch.empty((M, NG), dtype=torch.float32, device=x.device)
    w = 8
    celdas = M * NG
    _kq(w).lanzar(((celdas + w * 32 - 1) // (w * 32), 1), [y, sx, q, s, M, K, G])
    return q, s


_q8b = None


def _k8b():
    global _q8b
    if _q8b is None:
        _q8b = (Kernel("sk17_q8b.cu", "sk17_q8_max", defs=["-DNWARPS=8"], warps=8),
                Kernel("sk17_q8b.cu", "sk17_q8_quant", defs=["-DNWARPS=8"], warps=8))
    return _q8b


def q8b(x):
    M, K = x.shape
    NG = K // G
    kmax, kq = _k8b()
    bmx = torch.empty((M * NG,), dtype=torch.float32, device=x.device)
    xq = torch.empty((M, K), dtype=torch.int8, device=x.device)
    sx = torch.empty((M,), dtype=torch.float32, device=x.device)
    grid = ((M * NG + 8 * 32 - 1) // (8 * 32), 1)
    kmax.lanzar(grid, [x, bmx, M, K])
    kq.lanzar(grid, [x, bmx, xq, sx, M, K])
    return xq, sx


def _solo_q8(x):
    M, K = x.shape
    xq = torch.empty((M, K), dtype=torch.int8, device=x.device)
    sx = torch.empty((M,), dtype=torch.float32, device=x.device)
    _k8().lanzar(((M + 8 * 32 - 1) // (8 * 32), 1), [x, xq, sx, M, K])
    return xq, sx


def referencia_entera(x: torch.Tensor):
    """La MISMA cuenta entera en torch: int8 por token, Hadamard +-1, int4 por grupo."""
    M, K = x.shape
    NG = K // G
    xf = x.float()
    mx = xf.abs().amax(1).clamp_min(1e-8)
    sx = mx / 127
    xq = torch.round(xf * (127 / mx)[:, None]).clamp(-127, 127).to(torch.int8)
    # float64 es exacto para enteros de este tamano (|y| <= 32512)
    y = (xq.view(M * NG, G).double() @ signos(x.device).double().t()).round().to(torch.int32).view(M, NG, G)
    my = y.abs().amax(-1).clamp_min(1)
    q = torch.round(y.float() * (7.0 / my.float())[..., None]).clamp(-7, 7).to(torch.int8).view(M, K)
    s = my.float() * sx[:, None] / 112.0
    u = (q & 0xF).to(torch.uint8)
    nib = (u[:, 0::2] | (u[:, 1::2] << 4)).view(torch.int8)
    return nib, s, xq, sx


def check():
    ok = True
    for (M, K) in [(1, 256), (77, 1024), (64, 5120), (256, 5120)]:
        torch.manual_seed(0)
        x = (torch.randn(M, K, device="cuda") * 2).half()
        q, s = act_int4(x)
        qr, sr, xqr, sxr = referencia_entera(x)
        xq, sx = q8b(x)
        ig8 = (xq == xqr).float().mean().item()
        ig4 = (q == qr).float().mean().item()
        es = ((s - sr).abs() / sr).max().item()
        # empates de redondeo (~1 en 50k) y el max hecho en fp16: no son errores del kernel
        bien = ig8 > 0.9999 and ig4 > 0.9999 and es < 1e-3
        ok &= bien
        print(f"  M={M:5d} K={K:5d}  int8 iguales {100*ig8:.3f}%  int4 iguales {100*ig4:.3f}%  "
              f"escala err max {es:.1e}  {'OK' if bien else 'FALLA'}", flush=True)
    return ok


def bench(M=int(sys.argv[2]) if len(sys.argv) > 2 else 7488, K=5120, rondas=7, it=8):
    from vllm import _custom_ops as ops
    from vllm._genesis.kernels import sk15_w4a4g as S15
    torch.manual_seed(0)
    x = (torch.randn(M, K, device="cuda") * 2).half()
    casos = {"PTX: Q8 + SK-17 + Q4 (todo)": lambda: act_int4(x),
             "PTX: solo Q8 (hilo por fila)": lambda: _solo_q8(x),
             "PTX: solo Q8b (hilo por bloque)": lambda: q8b(x),
             "torch: misma cuenta entera": lambda: referencia_entera(x),
             "torch: Hadamard fp16 + int4": lambda: S15.preparar_act(x),
             "cutlass: scaled_int8_quant": lambda: ops.scaled_int8_quant(x, symmetric=True)}
    for f in casos.values():
        for _ in range(3): f()
    torch.cuda.synchronize(); res = {k: [] for k in casos}
    for _ in range(rondas):
        for k, f in casos.items():
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(it): f()
            torch.cuda.synchronize(); res[k].append((time.perf_counter() - t0) / it)
    for k, v in res.items():
        print(f"  {k:32s} {sorted(v)[len(v)//2]*1e3:8.2f} ms", flush=True)


if __name__ == "__main__":
    if "--bench" in sys.argv: bench()
    else: raise SystemExit(0 if check() else 1)
