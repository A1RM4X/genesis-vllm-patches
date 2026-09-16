# SPDX-License-Identifier: Apache-2.0
"""Layout de fragmentos de mma.sync.aligned.m16n8k{32,64}.row.col en sm_86.

Medido con sondas (tests/proto/lab_mma_sondas.py), identico para s8 m16n8k32
(per=4 elementos por registro, K=32) y s4 m16n8k64 (per=8 nibbles, K=64). Un warp
de 32 hilos multiplica A[16, K] x B[K, 8] -> C[16, 8]:

    A  hilo t, registro r (0..3), sub j:  fila 2*(t//4) + (r&1)
                                          k    (t%4)*2*per + (r>>1)*per + j
    B  hilo t, registro r (0..1), sub j:  col  t//4
                                          k    (t%4)*2*per + r*per + j
    C  hilo t, salida e (0..3):           fila 2*(t//4) + (e>>1)
                                          col  2*(t%4) + (e&1)

El sub j ocupa los bits [j*bits, (j+1)*bits) del registro (little endian).
"""

from __future__ import annotations

import functools

import torch


@functools.lru_cache(maxsize=None)
def indices(per: int, device: str = "cuda"):
    t = torch.arange(32).view(32, 1, 1)
    rA = torch.arange(4).view(1, 4, 1)
    rB = torch.arange(2).view(1, 2, 1)
    j = torch.arange(per).view(1, 1, per)
    filaA = (2 * (t // 4) + (rA & 1)).expand(32, 4, per)
    kA = ((t % 4) * 2 * per + (rA >> 1) * per + j).expand(32, 4, per)
    colB = (t // 4).expand(32, 2, per)
    kB = ((t % 4) * 2 * per + rB * per + j).expand(32, 2, per)
    e = torch.arange(4).view(1, 4)
    t2 = torch.arange(32).view(32, 1)
    filaC = (2 * (t2 // 4) + (e >> 1)).expand(32, 4)
    colC = (2 * (t2 % 4) + (e & 1)).expand(32, 4)
    d = dict(filaA=filaA, kA=kA, colB=colB, kB=kB, filaC=filaC, colC=colC)
    return {k: v.to(device) for k, v in d.items()}


def _a_registros(subs: torch.Tensor, bits: int) -> torch.Tensor:
    per = 32 // bits
    u = subs.to(torch.int64) & ((1 << bits) - 1)
    sh = torch.arange(0, 32, bits, device=subs.device, dtype=torch.int64)
    v = (u << sh).sum(-1)
    return torch.where(v >= 2**31, v - 2**32, v).to(torch.int32)


def empacar(A: torch.Tensor, B: torch.Tensor, bits: int):
    """A [G,16,K], B [G,K,8] (enteros con signo) -> registros [G,32,4], [G,32,2]."""
    per = 32 // bits
    ix = indices(per, str(A.device))
    G = A.shape[0]
    subsA = A[:, ix["filaA"], ix["kA"]]          # [G,32,4,per]
    subsB = B[:, ix["kB"], ix["colB"]]           # [G,32,2,per]
    return _a_registros(subsA, bits).contiguous(), _a_registros(subsB, bits).contiguous()


def desempacar_salida(out: torch.Tensor, per: int) -> torch.Tensor:
    """out [G,32,4] int32 -> C [G,16,8]."""
    ix = indices(per, str(out.device))
    G = out.shape[0]
    C = torch.empty(G, 16, 8, dtype=out.dtype, device=out.device)
    C[:, ix["filaC"], ix["colC"]] = out
    return C
