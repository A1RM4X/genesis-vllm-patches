#!/usr/bin/env python3
"""Mide SK-22 contra el Marlin de PN130 en las formas reales del modelo, para M chico.

SK-22 existe por una sola razon: en decode M vale 1..16 y Marlin, que es un kernel persistente
pensado para prefill, paga ahi un reparto de stripes que no le rinde. Se compara a igualdad de
entrada (mismos pesos, mismas escalas int16 con signo) y se reporta el cociente.
"""
from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
sys.path.insert(0, "/w/tests/proto")

G, TN = 128, 64
FORMAS = [(5120, 5120)]
MES = [1, 16]


def medir(fn, rep=50):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / rep * 1e6


def main() -> None:
    from sk22_banco import empaquetar
    from vllm._genesis.kernels.ptx_lab import Kernel
    from vllm.scalar_type import scalar_types

    torch.ops.load_library(os.environ["S16_SO"])
    ET = int(os.environ.get("ETAPAS", "3"))
    k22 = Kernel("sk22_gemm_w4a8.cu", "sk22_gemm_w4a8",
                 defs=[f"-DTN={TN}", f"-DETAPAS={ET}"], warps=4)
    dev = "cuda"
    shmem = ET * 16 * 32 + ET * (TN // 8) * 32 * 4
    vacio = torch.empty(0, dtype=torch.int32, device=dev)

    print(f"{'forma':>14}{'M':>4}{'marlin us':>11}{'sk22 us':>10}{'cociente':>10}")
    for N, K in FORMAS:
        torch.manual_seed(1234)
        q = torch.randint(0, 16, (K, N), dtype=torch.int32, device=dev)
        esc = torch.randint(-2000, 2000, (K // G, N), dtype=torch.int16, device=dev)
        b22 = empaquetar(q)
        # Marlin come su propio empaquetado; para VELOCIDAD alcanza con bits del mismo tamaño
        b_q = torch.randint(-(2**31), 2**31 - 1, (K // 16, N * 16 // 8),
                            dtype=torch.int32, device=dev)
        ws = torch.zeros(N // 64 * 16, dtype=torch.int32, device=dev)
        for M in MES:
            a = (torch.randn(M, K, device=dev) * 24).round().clamp(-127, 127).to(torch.int8)
            sumas = a.reshape(M, K // G, G).sum(2, dtype=torch.int32).t().contiguous()
            a_esc = torch.full((M,), 1 / 127, dtype=torch.float32, device=dev)
            c = torch.zeros(M, N, dtype=torch.float16, device=dev)

            def marlin():
                ws.zero_()
                torch.ops.genesis_marlin.marlin_gemm_s16(
                    a, None, b_q, None, esc.view(torch.float16), a_esc.reshape(M, 1),
                    None, None, vacio, vacio, ws,
                    scalar_types.uint4b8.id, M, N, K, True, False, True, False)

            def sk22():
                k22.lanzar((N // TN, 1), [a, b22, esc, sumas, a_esc, c, M, N, K, 1 / 4096],
                           shared=shmem)

            tm, ts = medir(marlin), medir(sk22)
            print(f"{f'{N}x{K}':>14}{M:>4}{tm:>11.1f}{ts:>10.1f}{tm / ts:>9.2f}x")
        del q, esc, b22, b_q, ws
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
