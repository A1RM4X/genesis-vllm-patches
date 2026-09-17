"""Compara bit a bit la salida del Marlin original contra el de la division magica.

Esencial: cambiar divisiones por multiplicacion magica toca el calculo de INDICES de tile. Si la
magia esta mal para algun divisor, el kernel lee del lugar equivocado — y da un resultado
incorrecto MAS RAPIDO, que es la peor forma de "optimizar". Se barren varias formas y varios M,
porque el camino de slices depende de k_tiles/n_tiles y del reparto entre bloques.
"""
import os, sys, torch
sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
torch.ops.load_library(os.environ["S16_SO"])
from vllm.scalar_type import scalar_types

G = 128
FORMAS = [(17408, 5120), (5120, 5120), (7168, 5120), (2560, 5120), (124160, 5120)]
MES = [1, 5, 16, 32, 64, 70, 80, 96, 112, 128, 144, 160, 256]

torch.manual_seed(0)
malos = 0
for N, H in FORMAS:
    dev = "cuda"
    b_q = torch.randint(-(2**31), 2**31 - 1, (H // 16, N * 16 // 8), dtype=torch.int32, device=dev)
    b_s = (torch.randn((H // G, N), dtype=torch.float16, device=dev).abs() * 0.01 + 0.001)
    m = b_s.abs().max()
    b_s16 = (b_s / m * 4096).round().to(torch.int16).view(b_s.dtype)
    gs = (m.float() / 4096).reshape(1)
    ws = torch.zeros(N // 64 * 16, dtype=torch.int32, device=dev)
    vacio = torch.empty(0, dtype=torch.int32, device=dev)
    for M in MES:
        x = (torch.randn(M, H, device=dev) * 20).round().clamp(-127, 127).to(torch.int8)
        a_s = torch.full((M, 1), 1 / 127, dtype=torch.float32, device=dev)
        ws.zero_()
        o = torch.ops.genesis_marlin.marlin_gemm_s16(
            x, None, b_q, None, b_s16, a_s, None, None, vacio, vacio, ws,
            scalar_types.uint4b8.id, M, N, H, True, False, True, False)
        torch.save(o.float().cpu(), f"/tmp/ref_{N}_{M}.pt" if os.environ.get("GUARDAR")
                   else "/dev/null")
        if not os.environ.get("GUARDAR"):
            ref = torch.load(f"/tmp/ref_{N}_{M}.pt")
            d = (o.float().cpu() - ref).abs().max().item()
            if d > 0:
                malos += 1
                print(f"  DISTINTO  N={N:6d} M={M:4d}  max|dif| = {d:.6f}")
if os.environ.get("GUARDAR"):
    print(f"referencias guardadas ({len(FORMAS)*len(MES)} casos)")
else:
    print(f"{'TODO IGUAL' if malos == 0 else str(malos) + ' CASOS DISTINTOS'} "
          f"({len(FORMAS)*len(MES)} casos, comparacion exacta)")
