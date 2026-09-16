"""Verifica el layout medido con datos al azar y mide TOPS de emision con el layout correcto."""
import time, torch
from vllm._genesis.kernels.ptx_lab import Kernel
from vllm._genesis.kernels.mma_layout import empacar, desempacar_salida
dev = "cuda"; torch.manual_seed(1)
for nombre, entrada, bits, K, lo, hi in (("s8", "lab_mma_s8", 8, 32, -127, 128), ("s4", "lab_mma_s4", 4, 64, -8, 8)):
    per = 32 // bits
    G = 8192
    A = torch.randint(lo, hi, (G, 16, K), dtype=torch.int8, device=dev)
    B = torch.randint(lo, hi, (G, K, 8), dtype=torch.int8, device=dev)
    ra, rb = empacar(A, B, bits)
    k = Kernel("lab_mma.cu", entrada, warps=1)
    out = torch.zeros(G, 32, 4, dtype=torch.int32, device=dev)
    k.lanzar((G, 1), [ra, rb, out, 1], sync=True)
    C = desempacar_salida(out, per)
    ref = torch.bmm(A.double(), B.double()).round().to(torch.int32)
    print(f"{nombre}: correcto={torch.equal(C, ref)} max dif={(C-ref).abs().max().item()}", flush=True)
    for reps in (256, 1024):
        k.lanzar((G, 1), [ra, rb, out, reps], sync=True)
        t0 = time.perf_counter(); it = 5
        for _ in range(it): k.lanzar((G, 1), [ra, rb, out, reps])
        torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / it
        print(f"   REPS={reps}: {G*reps*16*8*K/dt/1e12:7.1f} TOPS", flush=True)
