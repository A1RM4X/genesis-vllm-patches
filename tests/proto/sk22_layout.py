#!/usr/bin/env python3
"""Mide el layout de mma.m16n8k32.s8 en vez de deducirlo de la doc.

Modo 1 (A): se pone un unico 1 en el fragmento de B del lane/byte que se indica y se cargan en A
identificadores. La columna de C que se enciende dice que elemento de A vio ese producto.
Modo 0 (B): al reves.

De aca sale la tabla que necesita el empaquetado de SK-22.
"""
import torch
from vllm._genesis.kernels.ptx_lab import Kernel

k = Kernel("sk22_sonda.cu", "sk22_sonda", warps=1)
out = torch.zeros(128, dtype=torch.int32, device="cuda")

def corre(modo, fa=0, ka=0, lb=0, bb=0):
    out.zero_()
    k.lanzar((1, 1), [out, modo, fa, ka, lb, bb])
    torch.cuda.synchronize()
    return out.clone().cpu()

# ── layout de B: para cada (lane, byte) del fragmento, que (k, n) le toca ────────────────
print("LAYOUT DE B  (fragmento -> posicion en la matriz 32k x 8n)")
mapa = {}
for lb in range(32):
    for bb in range(8):
        c = corre(1, lb=lb, bb=bb)
        nz = (c != 0).nonzero().flatten().tolist()
        if not nz:
            continue
        # el hilo h, indice i de C  ->  fila = h//4 + (i//2)*8, col = (h%4)*2 + i%2
        h, i = nz[0] // 4, nz[0] % 4
        col = (h % 4) * 2 + (i % 2)
        val = int(c[nz[0]])          # el identificador de A que participo
        # el identificador dice lane_a y byte_a: id = 1 + lane*16 + r*4 + j
        ida = val - 1
        lane_a, resto = ida // 16, ida % 16
        r_a, j_a = resto // 4, resto % 4
        mapa[(lb, bb)] = (col, lane_a, r_a, j_a)
print(f"  {len(mapa)} entradas medidas de 256")
for key in sorted(mapa)[:8]:
    lb, bb = key
    col, la, ra, ja = mapa[key]
    print(f"    B lane={lb:2d} byte={bb}  ->  n={col}   (via A lane={la:2d} reg={ra} byte={ja})")

# ── deducir la regla cerrada y verificarla contra las 256 mediciones ─────────────────────
print("\nREGLA DEDUCIDA")
# A: lane L, reg r, byte j -> fila = L/4 + (r%2)*8 , k = (L%4)*4 + j + (r/2)*16
# B: lane L, reg r, byte j -> n    = L/4           , k = (L%4)*4 + j + r*16
def kA(L, r, j): return (L % 4) * 4 + j + (r // 2) * 16
def filaA(L, r):  return L // 4 + (r % 2) * 8
def kB(L, r, j): return (L % 4) * 4 + j + r * 16
def nB(L):        return L // 4

malos = 0
for (lb, bb), (col, la, ra, ja) in mapa.items():
    rb, jb = bb // 4, bb % 4
    # el producto que se encendio une A[fila][kA] con B[kB][n]; para ser el mismo termino,
    # kA tiene que ser igual a kB y n tiene que ser la columna medida
    if kA(la, ra, ja) != kB(lb, rb, jb) or nB(lb) != col:
        malos += 1
print(f"  A: fila = lane/4 + (reg%2)*8      k = (lane%4)*4 + byte + (reg/2)*16")
print(f"  B: n    = lane/4                  k = (lane%4)*4 + byte + reg*16")
print(f"  discrepancias contra las 256 mediciones: {malos}")
print(f"  {'REGLA CONFIRMADA' if malos == 0 else 'LA REGLA NO CIERRA'}")
