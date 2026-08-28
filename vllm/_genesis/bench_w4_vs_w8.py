"""Cuanto vale pasar de W8A8 a W4A8 en decode. Rafagas alternadas A,B,A,B.

El GEMM de decode es DRAM-bound sobre el peso: el techo medido de la 3090 en
esta forma de kernel es ~730 GB/s. Si W8 ya corre al 100% del techo, la unica
palanca que queda es mover la mitad de los bytes.
"""
import torch
import triton
from vllm._genesis.kernels.sk06_mlp_down import _sk06_mlp_down_kernel
from vllm._genesis.kernels.sk06_mlp_down_w4a8 import sk06_mlp_down_w4a8_gemm

FORMAS = [("SK-02/04", 3072, 5120, 64), ("SK-03", 5120, 7168, 16),
          ("SK-01", 5120, 8192, 48), ("SK-06", 8704, 5120, 64),
          ("SK-05", 5120, 17408, 64)]


def w8(a, b, out, res, asc, bsc, sh, M, N, K):
    grid = (triton.cdiv(M, 16) * triton.cdiv(N, 128),)
    _sk06_mlp_down_kernel[grid](
        a, b, out, res, asc, bsc, sh, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        out.stride(0), out.stride(1), 0, 0, sh.stride(0), sh.stride(1),
        BLOCK_M=16, BLOCK_N=128, BLOCK_K=128, GROUP_M=8,
        SHIFT_BLOCK=128, HAS_SHIFT=True, num_warps=8, num_stages=3)


print(f"{'forma':<9}{'M':>3}{'K':>6}{'N':>7}{'W8 us':>9}{'W8 GB/s':>9}{'W4 us':>9}{'W4 GB/s':>9}{'gana':>7}")
for nombre, K, N, _ in FORMAS:
    for M in (1, 8, 16):
        b8 = torch.randint(-127, 127, (K, N), dtype=torch.int8, device="cuda").t().contiguous().t()
        a = torch.randint(-127, 127, (M, K), dtype=torch.int8, device="cuda")
        out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
        res = torch.zeros((), dtype=torch.bfloat16, device="cuda")
        asc = torch.rand(M, dtype=torch.float32, device="cuda") * 0.01
        bsc = torch.ones(N, dtype=torch.float32, device="cuda")
        sh = torch.rand((K // 128, N // 128), dtype=torch.float32, device="cuda") * 0.01
        w4 = torch.randint(-127, 127, (K // 2, N), dtype=torch.int8, device="cuda")
        ws = torch.rand((K // 128, N), dtype=torch.float32, device="cuda") * 0.01

        for _ in range(3):
            w8(a, b8, out, res, asc, bsc, sh, M, N, K)
            sk06_mlp_down_w4a8_gemm(a, w4, asc, ws, None, torch.bfloat16)
        torch.cuda.synchronize()
        R = 40
        ev8 = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(R)]
        ev4 = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(R)]
        for r in range(R):
            ev8[r][0].record(); w8(a, b8, out, res, asc, bsc, sh, M, N, K); ev8[r][1].record()
            ev4[r][0].record(); sk06_mlp_down_w4a8_gemm(a, w4, asc, ws, None, torch.bfloat16); ev4[r][1].record()
        torch.cuda.synchronize()
        t8 = sorted(e0.elapsed_time(e1) * 1000 for e0, e1 in ev8)[R // 2]
        t4 = sorted(e0.elapsed_time(e1) * 1000 for e0, e1 in ev4)[R // 2]
        g8 = K * N / (t8 * 1e-6) / 1e9
        g4 = K * N / 2 / (t4 * 1e-6) / 1e9
        print(f"{nombre:<9}{M:>3}{K:>6}{N:>7}{t8:9.1f}{g8:9.0f}{t4:9.1f}{g4:9.0f}{t8/t4:7.2f}x")
        del a, b8, w4, out, sh, ws
        torch.cuda.empty_cache()
