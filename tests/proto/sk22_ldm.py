#!/usr/bin/env python3
"""¿ldmatrix sirve para el fragmento A del mma entero? Se compara contra la lectura escalar."""
import sys
import torch
sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
from vllm._genesis.kernels.ptx_lab import Kernel

k = Kernel("sk22_sondal.cu", "sk22_sondal", warps=1)
torch.manual_seed(5)
a = torch.randint(-128, 128, (16, 32), dtype=torch.int8, device="cuda")
for swz in (0, 1):
    sal = torch.zeros(32 * 8, dtype=torch.int32, device="cuda")
    k.lanzar((1, 1), [a, sal, swz])
    torch.cuda.synchronize()
    s = sal.reshape(32, 8)
    esc, ldm = s[:, :4], s[:, 4:]
    mal = int((esc != ldm).sum())
    print(f"  swizzle={swz}:  posiciones distintas = {mal:3d} / 128   "
          f"{'IDENTICO — ldmatrix sirve' if mal == 0 else 'NO coincide'}")
    if mal:
        for L in range(4):
            print(f"    lane {L}: escalar={[hex(x & 0xffffffff) for x in esc[L].tolist()]}")
            print(f"            ldmatrix={[hex(x & 0xffffffff) for x in ldm[L].tolist()]}")
