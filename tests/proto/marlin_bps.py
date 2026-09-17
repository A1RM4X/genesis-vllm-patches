#!/usr/bin/env python3
"""Mide el GEMM de Marlin con una .so dada. Una tanda por invocacion, para poder alternar
procesos entre variantes (las dos .so registran el mismo op y no conviven en un proceso).

Imprime una linea CSV por caso: forma,M,us
"""
from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")

G = 128
FORMAS = [(17408, 5120), (5120, 5120), (7168, 5120)]
MES = [1, 8, 16, 32, 64, 128, 256]


def main() -> None:
    torch.ops.load_library(os.environ["S16_SO"])
    from vllm.scalar_type import scalar_types

    dev = "cuda"
    vacio = torch.empty(0, dtype=torch.int32, device=dev)
    for N, K in FORMAS:
        torch.manual_seed(1234)
        b_q = torch.randint(-(2**31), 2**31 - 1, (K // 16, N * 16 // 8),
                            dtype=torch.int32, device=dev)
        b_s = torch.randint(-2000, 2000, (K // G, N), dtype=torch.int16,
                            device=dev).view(torch.float16)
        # el workspace se dimensiona para 2 bloques por SM: con 1 sobra, con 2 alcanza
        ws = torch.zeros(82 * 2 * 16, dtype=torch.int32, device=dev)
        for M in MES:
            a = (torch.randn(M, K, device=dev) * 24).round().clamp(-127, 127).to(torch.int8)
            a_s = torch.full((M, 1), 1 / 127, dtype=torch.float32, device=dev)

            def gemm():
                ws.zero_()
                torch.ops.genesis_marlin.marlin_gemm_s16(
                    a, None, b_q, None, b_s, a_s, None, None, None, None, None, ws,
                    scalar_types.uint4b8.id, M, N, K, True, False, True, False)

            fin = time.perf_counter() + 0.25          # calentar: el reloj arranca en 210 MHz
            while time.perf_counter() < fin:
                for _ in range(20):
                    gemm()
                torch.cuda.synchronize()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(100):
                gemm()
            torch.cuda.synchronize()
            print(f"{N}x{K},{M},{(time.perf_counter() - t0) / 100 * 1e6:.2f}", flush=True)
        del b_q, b_s, ws
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
