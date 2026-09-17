#!/usr/bin/env python3
"""Valida y mide SK-22, el GEMM W4A8 propio, contra el Marlin de PN130.

Primero correctitud contra una referencia en float64 (que representa exacto todo el rango que
aparece aca), y recien despues velocidad. Un GEMM que anda rapido y da mal no sirve, y en
enteros la referencia puede ser exacta, asi que no hay excusa para comparar con tolerancias.

Uso: sk22_banco.py [--solo-correctitud]
"""
from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")

G = 128
TN = 64


def empaquetar(q: torch.Tensor) -> torch.Tensor:
    """q [K, N] en [0,15] -> [K/16, N*8] int32, 8 nibbles por entero a lo largo de K.

    Es el layout que ya usa Marlin, para poder comparar con el mismo peso.
    """
    K, N = q.shape
    out = torch.zeros((K // 16, N * 8), dtype=torch.int32, device=q.device)
    for i in range(8):
        # los 8 nibbles de un int32 son 8 filas consecutivas de K
        out[:, :] |= (q[i::8][: K // 16].to(torch.int32) << (4 * i))[:, :N].repeat(1, 8)[:, : N * 8]
    return out


def referencia(a, q, esc, factor, a_esc):
    """C = sum_k a_k * (q_k - 8) * esc, en float64 — exacto para este rango."""
    M, K = a.shape
    N = q.shape[1]
    out = torch.zeros(M, N, dtype=torch.float64, device=a.device)
    for g in range(K // G):
        aa = a[:, g * G:(g + 1) * G].double()
        qq = q[g * G:(g + 1) * G, :].double() - 8.0
        out += (aa @ qq) * esc[g].double().unsqueeze(0)
    return out * factor * a_esc.double().unsqueeze(1)


def main() -> None:
    from vllm._genesis.kernels.ptx_lab import Kernel

    dev = "cuda"
    torch.manual_seed(7)
    k22 = Kernel("sk22_gemm_w4a8.cu", "sk22_gemm_w4a8", defs=[f"-DTN={TN}"], warps=4)

    for (N, K, M) in [(256, 512, 4), (512, 1024, 8), (1024, 1024, 16)]:
        a = (torch.randn(M, K, device=dev) * 24).round().clamp(-127, 127).to(torch.int8)
        q = torch.randint(0, 16, (K, N), dtype=torch.int32, device=dev)
        esc = torch.randint(-2000, 2000, (K // G, N), dtype=torch.int16, device=dev)
        sumas = a.to(torch.int32).reshape(M, K // G, G).sum(2).t().contiguous()
        a_esc = torch.full((M,), 1 / 127, dtype=torch.float32, device=dev)
        factor = 1.0 / 4096
        b = empaquetar(q)
        c = torch.zeros(M, N, dtype=torch.float16, device=dev)

        k22.lanzar((N // TN, 1, 1),
                   [a, b, esc, sumas, a_esc, c, M, N, K, factor],
                   shared=16 * 32 + 3 * (32 // 16) * TN * 4)
        torch.cuda.synchronize()

        ref = referencia(a, q, esc, factor, a_esc)
        d = float((c.double() - ref).abs().max())
        rel = d / max(float(ref.abs().max()), 1e-9)
        print(f"  N={N:5d} K={K:5d} M={M:3d}   max|dif|={d:12.4f}   relativo={rel:.2e}"
              f"   {'OK' if rel < 1e-3 else 'MAL'}")


if __name__ == "__main__":
    main()
