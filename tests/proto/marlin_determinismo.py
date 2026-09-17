#!/usr/bin/env python3
"""Llama DOS veces al mismo GEMM con los mismos datos, en el mismo proceso, y compara.

Si difiere, el kernel tiene una carrera: no depende de la version de la .so ni del banco.
WS_MULT agranda el workspace para descartar que sea eso.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
G = 128
WS_MULT = int(os.environ.get("WS_MULT", "1"))


def main() -> None:
    torch.ops.load_library(os.environ["S16_SO"])
    from vllm.scalar_type import scalar_types
    dev = "cuda"
    print(f"  workspace x{WS_MULT}")
    for N, K in [(17408, 5120), (5120, 5120)]:
        torch.manual_seed(1234)
        b_q = torch.randint(-(2**31), 2**31 - 1, (K // 16, N * 16 // 8),
                            dtype=torch.int32, device=dev)
        b_s = torch.randint(-64, 64, (K // G, N), dtype=torch.int16, device=dev).view(torch.float16)
        ws = torch.zeros(82 * 2 * 16 * WS_MULT, dtype=torch.int32, device=dev)
        for M in (64, 80, 88, 96, 104, 112, 128, 160):
            a = (torch.randn(M, K, device=dev) * 24).round().clamp(-127, 127).to(torch.int8)
            a_s = torch.full((M, 1), 1 / 127 / 4096, dtype=torch.float32, device=dev)

            def una():
                ws.zero_()
                return torch.ops.genesis_marlin.marlin_gemm_s16(
                    a, None, b_q, None, b_s, a_s, None, None, None, None, None, ws,
                    scalar_types.uint4b8.id, M, N, K, True, False, True, False).clone()

            r = [una() for _ in range(4)]
            torch.cuda.synchronize()
            peor = max(float((r[i].float() - r[0].float()).abs().max()) for i in range(1, 4))
            mag = float(r[0].float().abs().max())
            print(f"  {N}x{K} M={M:4d}  max|dif| entre 4 llamadas = {peor:11.4f}"
                  f"   (magnitud {mag:9.1f})   {'estable' if peor == 0 else 'CARRERA'}")
        del b_q, b_s, ws
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
