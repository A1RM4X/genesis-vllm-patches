#!/usr/bin/env python3
"""Ejecuta UNA invocacion del kernel que se le pida, para que ncu la perfile.

ncu instrumenta cada lanzamiento y los replays son caros, asi que este script hace el minimo:
prepara los tensores, calienta fuera del rango perfilado y lanza una sola vez bajo
cudaProfilerStart/Stop. Variables: KERNEL (marlin|sk22), N, K, M, WARPS.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
sys.path.insert(0, "/w/tests/proto")

G = 128
N = int(os.environ.get("N", "5120"))
K = int(os.environ.get("K", "5120"))
M = int(os.environ.get("M", "1"))
KERNEL = os.environ.get("KERNEL", "marlin")


def main() -> None:
    dev = "cuda"
    torch.manual_seed(1234)
    a = (torch.randn(M, K, device=dev) * 24).round().clamp(-127, 127).to(torch.int8)
    a_esc = torch.full((M,), 1 / 127, dtype=torch.float32, device=dev)
    esc = torch.randint(-2000, 2000, (K // G, N), dtype=torch.int16, device=dev)

    if KERNEL == "marlin":
        torch.ops.load_library(os.environ["S16_SO"])
        from vllm.scalar_type import scalar_types
        b_q = torch.randint(-(2**31), 2**31 - 1, (K // 16, N * 16 // 8),
                            dtype=torch.int32, device=dev)
        ws = torch.zeros(82 * 2 * 16, dtype=torch.int32, device=dev)
        b_s = esc.view(torch.float16)

        def lanzar():
            ws.zero_()
            torch.ops.genesis_marlin.marlin_gemm_s16(
                a, None, b_q, None, b_s, a_esc.reshape(M, 1), None, None, None, None, None,
                ws, scalar_types.uint4b8.id, M, N, K, True, False, True, False)
    else:
        from sk22_banco import empaquetar
        from vllm._genesis.kernels.ptx_lab import Kernel
        W = int(os.environ.get("WARPS", "2"))
        TN, KPI = W * 16, 128
        ET = int(os.environ.get("ETAPAS", "3"))
        k22 = Kernel("sk22_gemm_w4a8.cu", "sk22_gemm_w4a8",
                     defs=[f"-DTN={TN}", f"-DETAPAS={ET}", f"-DKPI={KPI}", f"-DWARPS={W}"],
                     warps=W)
        q = torch.randint(0, 16, (K, N), dtype=torch.int32, device=dev)
        b22 = empaquetar(q, TN)
        sumas = a.reshape(M, K // G, G).sum(2, dtype=torch.int32).t().contiguous()
        c = torch.zeros(M, N, dtype=torch.float16, device=dev)
        shmem = ET * (16 * KPI + (KPI // 32) * (TN // 8) * 32 * 4)

        def lanzar():
            k22.lanzar((N // TN, int(os.environ.get("SK", "1"))), [a, b22, esc, sumas, a_esc, c, M, N, K, 1 / 4096],
                       shared=shmem)

    for _ in range(30):                      # calentar fuera del rango perfilado
        lanzar()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    lanzar()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
