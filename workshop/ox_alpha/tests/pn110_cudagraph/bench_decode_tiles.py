#!/usr/bin/env python3
"""bench_decode_tiles.py — busca el tile optimo de DECODE para los GEMM W8A8.

Hipotesis que prueba: en decode (M pequeno) el GEMM es memory-bound leyendo
pesos, y la grilla es cdiv(M,BLOCK_M) * cdiv(N,BLOCK_N). Con BLOCK_M=16 y
BLOCK_N=128 varias capas del modelo quedan en 40 CTAs sobre 82 SMs, o sea media
GPU ociosa. Bajar BLOCK_N sube el numero de CTAs.

Metodologia (sin esto el ruido es +-25% y la busqueda devuelve incoherencias):
  * relojes fijos (el llamador hace nvidia-smi -lgc)
  * CUDA events, no wall clock
  * salida prealocada, fuera del bucle medido
  * configs intercaladas round-robin, no una tras otra
  * mediana de N repeticiones
"""
import itertools, statistics, sys
import torch, triton

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
from vllm._genesis.kernels.sk03_fa_qkv import _sk03_fa_qkv_kernel

FORMAS = [("qkv", 5120, 7168), ("o_proj", 3072, 5120), ("gate_up", 5120, 17408),
          ("down", 8704, 5120), ("gdn_qkvz", 5120, 8192), ("gdn_out", 3072, 5120)]
# Sin BLOCK_K > 128: romperia el shift diadico (ver INVARIANTE en los kernels).
CFGS = [
    (16, 128, 128, 8, 8, 3),
    (16, 128, 128, 8, 4, 4),
    (16,  64, 128, 8, 4, 4),
    (32,  64, 128, 8, 4, 4),
    (32, 128, 128, 8, 8, 3),
    (32, 128, 128, 8, 4, 4),
    (32,  64, 128, 8, 4, 3),
    (64,  64, 128, 8, 4, 4),
]
SM = 82
REPS = 40

def operandos(M, K, N, dev):
    a = torch.randint(-127, 128, (M, K), dtype=torch.int8, device=dev)
    b = torch.empty_strided((K, N), (1, K), dtype=torch.int8, device=dev)
    b.copy_(torch.randint(-127, 128, (N, K), dtype=torch.int8, device=dev).t())
    return (a, b,
            torch.rand(M, dtype=torch.float32, device=dev) * 0.01 + 1e-3,
            torch.rand(N, dtype=torch.float32, device=dev) * 0.01 + 1e-3,
            torch.zeros((K // 128, N // 128), dtype=torch.int8, device=dev),
            torch.zeros((), dtype=torch.bfloat16, device=dev).as_strided((M, N), (0, 0)),
            torch.empty((M, N), dtype=torch.bfloat16, device=dev))

def corre(cfg, ops, M, K, N):
    bm, bn, bk, gm, w, st = cfg
    a, b, a_s, b_s, sh, epi, out = ops
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    _sk03_fa_qkv_kernel[grid](
        a, b, a_s, b_s, sh, epi, out, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        sh.stride(0), sh.stride(1), epi.stride(0), epi.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=gm, SHIFT_BLOCK=128,
        HAS_SHIFT=True, num_warps=w, num_stages=st)
    return grid[0]

def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    dev = torch.device("cuda")
    BW = 936e9
    print(f"M={M}  (decode). Techo = K*N/936GB/s. CTAs = ceil(M/BM)*ceil(N/BN), SMs={SM}\n")
    for nombre, K, N in FORMAS:
        ops = operandos(M, K, N, dev)
        techo = K * N / BW * 1e6
        validas = []
        for cfg in CFGS:
            try:
                ctas = corre(cfg, ops, M, K, N); torch.cuda.synchronize(); validas.append(cfg)
            except Exception as e:
                print(f"  {cfg} -> no compila: {type(e).__name__}")
        tiempos = {c: [] for c in validas}
        ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in validas]
        for _ in range(REPS):                      # round-robin
            for i, c in enumerate(validas):
                e0, e1 = ev[i]
                e0.record(); corre(c, ops, M, K, N); e1.record()
                torch.cuda.synchronize()
                tiempos[c].append(e0.elapsed_time(e1) * 1000.0)
        print(f"{nombre:<10} K={K:<5} N={N:<6} techo={techo:6.1f}us")
        base = None
        for c in sorted(validas, key=lambda c: statistics.median(tiempos[c])):
            t = statistics.median(tiempos[c])
            if base is None: base = t
            ctas = triton.cdiv(M, c[0]) * triton.cdiv(N, c[1])
            marca = " <-- actual" if c == CFGS[0] else ""
            print(f"   BM={c[0]:<4}BN={c[1]:<4}BK={c[2]:<4}w={c[4]} st={c[5]}  {t:7.1f}us  "
                  f"{100*techo/t:5.1f}% techo  {ctas:4} CTAs ({ctas/SM:4.1f} olas){marca}")
        print()

main()
