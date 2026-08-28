"""Barrido de BLOCK_N/BLOCK_M/stages para el bucket de DECODE (M<=32).

Motivo: la tabla de tiles usa BLOCK_N=128 fijo. Con N=5120 por rank eso da
cdiv(5120,128)=40 CTAs para 82 SM de la 3090: la mitad de la GPU parada. El
GEMM de decode es DRAM-bound y el trafico de pesos es K*N sea cual sea el tile
(cada byte del peso se lee UNA vez), asi que bajar BLOCK_N no agrega trafico:
solo re-lee la activacion, que son M<=16 filas.

Metodologia obligatoria del proyecto: rafagas ALTERNADAS A,B,A,B. Un
synchronize por lanzamiento mete ~10 us fijos y comprime toda razon hacia 1.
"""
import itertools
import torch
import triton
from vllm._genesis.kernels.sk06_mlp_down import _sk06_mlp_down_kernel

SM = 82
FORMAS = [
    ("SK-01 qkvz", 5120, 8192, 48),
    ("SK-02 gdn_out", 3072, 5120, 48),
    ("SK-03 fa_qkv", 5120, 7168, 16),
    ("SK-04 fa_o", 3072, 5120, 16),
    ("SK-05 gate_up", 5120, 17408, 64),
    ("SK-06 down", 8704, 5120, 64),
]
CFGS = [
    (16, 128, 128, 8, 8, 3),   # la actual
    (16,  64, 128, 8, 4, 3),
    (16,  64, 128, 8, 4, 4),
    (16,  32, 128, 8, 4, 4),
    (16,  64, 128, 8, 8, 3),
    (32,  64, 128, 8, 4, 3),
]


def lanzar(cfg, a, b, out, res, asc, bsc, sh, M, N, K):
    bm, bn, bk, gm, w, st = cfg
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    _sk06_mlp_down_kernel[grid](
        a, b, out, res, asc, bsc, sh, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        out.stride(0), out.stride(1), 0, 0, sh.stride(0), sh.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=gm,
        SHIFT_BLOCK=128, HAS_SHIFT=True, num_warps=w, num_stages=st)


def medir(cfgs, args, M, N, K, rondas=60):
    """Rafagas alternadas: una iteracion lanza TODAS las configs, en orden."""
    tiempos = {c: 0.0 for c in cfgs}
    ev = {c: [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(rondas)] for c in cfgs}
    for c in cfgs:                      # warmup / compilacion
        for _ in range(3):
            lanzar(c, *args, M, N, K)
    torch.cuda.synchronize()
    for r in range(rondas):
        for c in cfgs:
            e0, e1 = ev[c][r]
            e0.record()
            lanzar(c, *args, M, N, K)
            e1.record()
    torch.cuda.synchronize()
    for c in cfgs:
        ts = sorted(e0.elapsed_time(e1) * 1000 for e0, e1 in ev[c])
        tiempos[c] = ts[len(ts) // 2]   # mediana, en us
    return tiempos


print(f"{'forma':<14} {'M':>3}  " + "  ".join(f"{bm}x{bn}/w{w}s{st}" for bm, bn, _, _, w, st in CFGS))
for nombre, K, N, veces in FORMAS:
    for M in (1, 4, 8, 16):
        b = torch.randint(-127, 127, (K, N), dtype=torch.int8, device="cuda").t().contiguous().t()
        a = torch.randint(-127, 127, (M, K), dtype=torch.int8, device="cuda")
        out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
        res = torch.zeros((), dtype=torch.bfloat16, device="cuda")
        asc = torch.rand(M, dtype=torch.float32, device="cuda") * 0.01
        bsc = torch.ones(N, dtype=torch.float32, device="cuda")
        sh = torch.rand((K // 128, N // 128), dtype=torch.float32, device="cuda") * 0.01
        t = medir(CFGS, (a, b, out, res, asc, bsc, sh), M, N, K)
        base = t[CFGS[0]]
        celdas = []
        for c in CFGS:
            ctas = triton.cdiv(M, c[0]) * triton.cdiv(N, c[1])
            celdas.append(f"{t[c]:7.1f}us {base/t[c]:4.2f}x c{ctas:<4}")
        print(f"{nombre:<14} {M:>3}  " + "  ".join(celdas))
        del a, b, out, sh
        torch.cuda.empty_cache()
