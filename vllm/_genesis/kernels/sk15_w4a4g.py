# SPDX-License-Identifier: Apache-2.0
"""SK-15 — gate_up + SiLU W4A4 con Hadamard 256 y escalas por grupo de 256.

Kernel: ``kernels/cuda/sk15_gateup_w4a4g.cu`` (SK-14 + volcado por grupo).

Preparacion (misma matematica que la simulacion validada en calidad,
``GENESIS_PN118_FAKE=w4a4 FAKE_ROT=bloque FAKE_ROT_BLOQUE=256 GRUPO_W=256
GRUPO_A=256``, KL 0,0187 contra fp16 en gate_up):

    x~ = x por bloques de 256 @ H^T        W~ = W por bloques de 256 @ H^T
    q  = round(t / s).clamp(-7, 7),  s = max|t_g| / 7  por grupo de 256
    nibbles: byte i = q[2i] (bits bajos) | q[2i+1] << 4

``python -m vllm._genesis.kernels.sk15_w4a4g --check`` valida contra torch y
``--bench`` mide contra cutlass W8A8 + SiLU (el camino de PN118).
"""

from __future__ import annotations

import math
import os
import re
import sys
import time

import torch

from vllm._genesis.kernels.ptx_lab import Kernel

G = 256          # elementos por grupo = BK*2 nibbles
_DEFS = os.environ.get("GENESIS_SK15_DEFS", "").split()


def _def(nombre: str, por_defecto: int) -> int:
    m = re.search(rf"-D{nombre}=(\d+)", " ".join(_DEFS))
    return int(m.group(1)) if m else por_defecto


BM, BN, BK = _def("BM", 256), _def("BN", 64), _def("BK", 128)
STAGES = _def("STAGES", 2)
NWARPS = _def("NWARPS", 32)
SHARED = STAGES * (BM * BK + BN * 2 * BK)
_kernel = None
_H = {}


def kernel() -> Kernel:
    global _kernel
    if _kernel is None:
        _kernel = Kernel("sk15_gateup_w4a4g.cu", "sk15_gateup_w4a4g", defs=_DEFS, warps=NWARPS)
    return _kernel


def hadamard(device, dtype=torch.float32) -> torch.Tensor:
    key = (str(device), dtype)
    if key not in _H:
        H = torch.ones(1, 1, device=device, dtype=torch.float64)
        while H.shape[0] < G:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        _H[key] = (H / math.sqrt(G)).to(dtype)
    return _H[key]


def rotar(t: torch.Tensor) -> torch.Tensor:
    """[..., K] (K multiplo de 256) -> por bloques @ H^T."""
    K = t.shape[-1]
    H = hadamard(t.device, t.dtype)
    return (t.reshape(*t.shape[:-1], K // G, G) @ H.t()).reshape(t.shape)


def cuantizar(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[R, K] float -> (q int8 en [-7,7] [R, K], s fp32 [R, K/256])."""
    R, K = t.shape
    y = t.float().reshape(R, K // G, G)
    s = y.abs().amax(-1).clamp_min(1e-8) / 7.0
    q = (y / s[..., None]).round().clamp(-7, 7).to(torch.int8).reshape(R, K)
    return q, s.contiguous()


def nibbles(q: torch.Tensor) -> torch.Tensor:
    u = (q & 0xF).to(torch.uint8)
    return (u[:, 0::2] | (u[:, 1::2] << 4)).view(torch.int8).contiguous()


def preparar_pesos(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """W float [N, K] -> (nibbles int8 [N, K/2], escalas fp32 [N, K/256])."""
    q, s = cuantizar(rotar(W.float()))
    return nibbles(q), s


def preparar_act(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q, s = cuantizar(rotar(x.float()))
    return nibbles(q), s


def gemm(a_nib, sa, wg_nib, sg, wu_nib, su) -> torch.Tensor:
    M, Kb = a_nib.shape
    N = wg_nib.shape[0]
    assert Kb % BK == 0 and sa.shape == (M, Kb // BK) and sg.shape == (N, Kb // BK)
    out = torch.empty((M, N), dtype=torch.float16, device=a_nib.device)
    grid = ((N + BN - 1) // BN, (M + BM - 1) // BM)
    kernel().lanzar(grid, [a_nib, wg_nib, wu_nib, sa, sg, su, out, M, N, Kb], shared=SHARED)
    return out


def referencia(qa, sa, qg, sg, qu, su) -> torch.Tensor:
    """Misma cuenta en torch con los int4 ya cuantizados (sin redondeo del kernel)."""
    M, K = qa.shape
    NG = K // G
    a = qa.float().reshape(M, NG, G)

    def parte(qw, sw):
        w = qw.float().reshape(qw.shape[0], NG, G)
        acc = torch.einsum("mgk,ngk->mng", a, w)          # int por grupo
        return (acc * sa[:, None, :] * sw[None, :, :]).sum(-1)

    return torch.nn.functional.silu(parte(qg, sg)) * parte(qu, su)


def check() -> bool:
    ok = True
    for (M, K, N) in [(64, 512, 64), (300, 1024, 130), (2048, 5120, 1024), (1000, 2048, 777)]:
        torch.manual_seed(0)
        x = torch.randn(M, K, device="cuda") * 3
        Wg = torch.randn(N, K, device="cuda") * 0.02
        Wu = torch.randn(N, K, device="cuda") * 0.02
        qa, sa = cuantizar(rotar(x)); qg, sg = cuantizar(rotar(Wg)); qu, su = cuantizar(rotar(Wu))
        ref = referencia(qa, sa, qg, sg, qu, su)
        got = gemm(nibbles(qa), sa, nibbles(qg), sg, nibbles(qu), su).float()
        err = ((got - ref).norm() / ref.norm()).item()
        exacto = torch.nn.functional.silu(x @ Wg.t()) * (x @ Wu.t())
        err_q = ((ref - exacto).norm() / exacto.norm()).item()
        bien = err < 5e-3
        ok &= bien
        print(f"  M={M:5d} K={K:5d} N={N:5d}  kernel vs ref {err:.2e}  ({'OK' if bien else 'FALLA'})"
              f"  | ref W4A4 vs float {100*err_q:.1f}%", flush=True)
    return ok


def bench(M=7488, K=5120, N=8704, rondas=7, it=8):
    """Mide el bloque gate_up+SiLU completo contra el camino de PN118 (cutlass W8A8).

    Nuestro: rotacion+cuantizacion de x (torch por ahora) + SK-15.
    Rival:   scaled_int8_quant(x) + 2x cutlass_scaled_mm + SiLU*mul.
    """
    from vllm import _custom_ops as ops
    torch.manual_seed(0)
    x = (torch.randn(M, K, device="cuda") * 3).half()
    Wg = torch.randn(N, K, device="cuda") * 0.02
    Wu = torch.randn(N, K, device="cuda") * 0.02
    wg_n, sg = preparar_pesos(Wg); wu_n, su = preparar_pesos(Wu)
    # rival: W8 per-canal (formato PN118)
    def w8(W):
        s = W.abs().amax(1).clamp_min(1e-8) / 127
        return (W / s[:, None]).round().clamp(-127, 127).to(torch.int8), s
    wg8, sg8 = w8(Wg); wu8, su8 = w8(Wu)
    wg8t, wu8t = wg8.t(), wu8.t()          # [N,K] contiguo -> strides (1,K) como pide cutlass

    def nuestro_gemm(an, sa):
        return gemm(an, sa, wg_n, sg, wu_n, su)

    def nuestro_total():
        an, sa = preparar_act(x)
        return nuestro_gemm(an, sa)

    def rival_gemm(xq, xs):
        g = ops.cutlass_scaled_mm(xq, wg8t, xs, sg8.view(1, -1), torch.float16)
        u = ops.cutlass_scaled_mm(xq, wu8t, xs, su8.view(1, -1), torch.float16)
        return torch.nn.functional.silu(g) * u

    def rival_total():
        xq, xs, _ = ops.scaled_int8_quant(x, symmetric=True)
        return rival_gemm(xq, xs)

    an, sa = preparar_act(x)
    xq, xs, _ = ops.scaled_int8_quant(x, symmetric=True)
    casos = {"SK-15 GEMM": lambda: nuestro_gemm(an, sa), "cutlass GEMM+SiLU": lambda: rival_gemm(xq, xs),
             "SK-15 total (act torch)": nuestro_total, "cutlass total": rival_total,
             "act rot+q4 torch": lambda: preparar_act(x)}
    for f in casos.values():
        for _ in range(3):
            f()
    torch.cuda.synchronize()
    res = {k: [] for k in casos}
    for _ in range(rondas):                  # intercalado: el cap de 220 W contamina A-luego-B
        for k, f in casos.items():
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(it):
                f()
            torch.cuda.synchronize(); res[k].append((time.perf_counter() - t0) / it)
    med = {k: sorted(v)[len(v) // 2] for k, v in res.items()}
    flops = 4 * M * K * N
    for k, v in med.items():
        print(f"  {k:26s} {v*1e3:8.2f} ms   {flops/v/1e12:7.1f} TOPS-eq", flush=True)
    print(f"  GEMM: SK-15 / cutlass = {med['cutlass GEMM+SiLU']/med['SK-15 GEMM']:.2f}x   "
          f"total: {med['cutlass total']/med['SK-15 total (act torch)']:.2f}x", flush=True)


if __name__ == "__main__":
    if "--ptx" in sys.argv:
        print(kernel().ptx(forzar=True)[:2000])
    elif "--bench" in sys.argv:
        args = [int(a) for a in sys.argv[2:]] if len(sys.argv) > 2 else []
        bench(*args)
    else:
        raise SystemExit(0 if check() else 1)
