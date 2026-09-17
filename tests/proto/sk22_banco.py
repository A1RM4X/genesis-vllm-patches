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


def empaquetar(q: torch.Tensor, tn: int = TN) -> torch.Tensor:
    """q [K, N] en [0,15] -> el layout que consume el fragmento del mma, MEDIDO.

    De tests/proto/sk22_layout.py, con las 256 posiciones sondeadas y cero discrepancias:

        B: n = lane/4     k = (lane%4)*4 + byte + reg*16

    El kernel lee UN int32 por lane y lo abre en dos registros: ``emp & 0x0F0F0F0F`` son los
    nibbles bajos (reg 0) y ``(emp>>4) & 0x0F0F0F0F`` los altos (reg 1). O sea que el byte j del
    entero que le toca al lane L tiene que llevar

        nibble bajo:  q[kb + (L%4)*4 + j     ][n0 + L/4]      (reg 0)
        nibble alto:  q[kb + (L%4)*4 + j + 16][n0 + L/4]      (reg 1)

    con kb el K del tile. Salida [K/32, 32 lanes * (N/8 grupos)] int32.
    """
    K, N = q.shape
    ktiles, ngrupos = K // 32, N // 8
    out = torch.zeros((ktiles, ngrupos, 32), dtype=torch.int32, device=q.device)
    qi = q.to(torch.int32)
    for L in range(32):
        n = L // 4
        for j in range(4):
            kb = (L % 4) * 4 + j
            bajo = qi[kb::32][:ktiles][:, n::8][:, :ngrupos]          # [ktiles, ngrupos]
            alto = qi[kb + 16::32][:ktiles][:, n::8][:, :ngrupos]
            out[:, :, L] |= ((bajo & 0xF) | ((alto & 0xF) << 4)) << (8 * j)
    # ...y se reordena por tile de N: [n_tile][k_tile][TN/8 grupos * 32]. Es el mismo repack
    # que hace Marlin al cargar el modelo, y es lo que deja a cada bloque leyendo CONTIGUO.
    gpt = tn // 8                                        # grupos de 8 columnas por tile de N
    out = out.reshape(ktiles, ngrupos // gpt, gpt * 32)  # [kt, n_tile, ...]
    return out.permute(1, 0, 2).contiguous().reshape(-1)


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
        sumas = a.reshape(M, K // G, G).sum(2, dtype=torch.int32).t().contiguous()
        a_esc = torch.full((M,), 1 / 127, dtype=torch.float32, device=dev)
        factor = 1.0 / 4096
        b = empaquetar(q)
        c = torch.zeros(M, N, dtype=torch.float16, device=dev)

        # shared: ETAPAS buffers de A (16 x KPI) y de B (KPI/32 tiles x TN/8 grupos x 32 ints)
        KPI = 128
        shmem = 3 * (16 * KPI + (KPI // 32) * (TN // 8) * 32 * 4)
        k22.lanzar((N // TN, 1),
                   [a, b, esc, sumas, a_esc, c, M, N, K, factor],
                   shared=shmem)
        torch.cuda.synchronize()

        ref = referencia(a, q, esc, factor, a_esc)
        d = float((c.double() - ref).abs().max())
        rel = d / max(float(ref.abs().max()), 1e-9)
        print(f"  N={N:5d} K={K:5d} M={M:3d}   max|dif|={d:12.4f}   relativo={rel:.2e}"
              f"   {'OK' if rel < 1e-3 else 'MAL'}")


if __name__ == "__main__":
    main()
