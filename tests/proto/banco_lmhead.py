#!/usr/bin/env python3
"""Banco aislado del GEMM del lm_head en decode.

Por que existe: el lm_head es el segundo kernel mas caro del decode (3,74 ms/paso en 5
lanzamientos) y corre al 68% del techo de DRAM, mientras los lineales de al lado estan al 98%.
Medirlo adentro del servidor cuesta un reinicio de 7 minutos por variante, asi que aca se lo
aisla: misma forma, mismos dtypes, y al lado un lector puro que da el techo REAL de la placa hoy
(no el de la hoja de datos, que con las placas capadas sobreestima ~11%).

Formas: M es 1 (una pasada del borrador MTP) o 5 (el modelo principal verificando K+1 tokens).

Uso: banco_lmhead.py [--rep N]
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")

V = 248320          # vocabulario del checkpoint
H = 5120            # hidden
TP = 2
N = V // TP         # columnas por GPU
G = 128             # group_size


SOSTENIDO = int(os.environ.get("GENESIS_BANCO_SOSTENIDO", "0"))


def medir(fn, rep: int = 50, calentar: int = 10) -> float:
    """Mediana en us, con grafo CUDA para sacar el ruido de lanzamiento (a M=1 domina).

    Dos regimenes, y la diferencia entre ellos NO es un detalle:

    * rafaga (por defecto): un replay, sincronizar, repetir. La placa alcanza a recuperar boost
      entre medicion y medicion y da ~830 GB/s.
    * sostenido (``GENESIS_BANCO_SOSTENIDO=N``): N replays encadenados sin sincronizar, que es
      como corre de verdad adentro del servidor. Con las placas capadas a 229/243 W la frecuencia
      cae y el techo con ella.

    Medir en rafaga y comparar contra un numero del servidor es como se llega a conclusiones
    falsas sobre en que porcentaje del techo esta un kernel.
    """
    for _ in range(calentar):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    if SOSTENIDO:
        for _ in range(SOSTENIDO // 2):      # calentar hasta que baje la frecuencia
            g.replay()
        torch.cuda.synchronize()
        ini, fin = torch.cuda.Event(True), torch.cuda.Event(True)
        ini.record()
        for _ in range(SOSTENIDO):
            g.replay()
        fin.record()
        torch.cuda.synchronize()
        return ini.elapsed_time(fin) * 1000.0 / SOSTENIDO
    tiempos = []
    for _ in range(rep):
        ini, fin = torch.cuda.Event(True), torch.cuda.Event(True)
        ini.record()
        g.replay()
        fin.record()
        torch.cuda.synchronize()
        tiempos.append(ini.elapsed_time(fin) * 1000.0)
    tiempos.sort()
    return tiempos[len(tiempos) // 2]


def bytes_peso(n: int = N, k: int = H) -> int:
    """Lo que el kernel TIENE que leer si o si: el peso int4 y sus escalas."""
    return n * k // 2 + (k // G) * n * 2


def techo_de_trafico(nbytes: int, rep: int) -> float:
    """Techo real de la placa HOY: una copia D2D que mueve exactamente ``nbytes`` en total.

    Se usa copy_ y no una reduccion: ``sum()`` de torch hace varias pasadas y mide 171 GB/s, o sea
    mide a torch, no a la placa. La copia lee la mitad y escribe la mitad, que es el trafico mas
    parecido a lo que hace un GEMM memory-bound.
    """
    mitad = nbytes // 2
    orig = torch.empty(mitad, dtype=torch.uint8, device="cuda")
    dest = torch.empty(mitad, dtype=torch.uint8, device="cuda")
    return medir(lambda: dest.copy_(orig), rep)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", type=int, default=50)
    a = ap.parse_args()

    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"placa: {props.name}  SMs={props.multi_processor_count}  "
          f"sm_{props.major}{props.minor}")
    nb = bytes_peso()
    print(f"lm_head por GPU: {N} x {H} int4 = {nb/1e6:.1f} MB (peso + escalas)\n")

    t_lec = techo_de_trafico(nb, a.rep)
    print(f"{'referencia':<34} {'us':>9} {'GB/s':>8} {'% del techo':>12}")
    print(f"{'copia D2D de ' + f'{nb/1e6:.0f} MB':<34} {t_lec:9.1f} {nb/t_lec/1e3:8.1f} "
          f"{100.0:11.0f}%")
    techo_gbs = nb / t_lec / 1e3

    # ── el kernel de vLLM, que es el que corre hoy ──────────────────────────────────────────
    from vllm import _custom_ops as ops  # noqa: F401
    from vllm.scalar_type import scalar_types

    print()
    for M in (1, 5):
        b_q = torch.randint(-(2**31), 2**31 - 1, (H // 16, N * 16 // 8),
                            dtype=torch.int32, device=dev)
        b_s = torch.ones((H // G, N), dtype=torch.float16, device=dev) * 0.01
        ws = torch.zeros(N // 64 * 16, dtype=torch.int32, device=dev)
        x = torch.randn(M, H, dtype=torch.float16, device=dev) * 0.1
        vacio = torch.empty(0, dtype=torch.int32, device=dev)
        xq = (x * 127).round().clamp(-127, 127).to(torch.int8)
        a_s = torch.full((M, 1), 1 / 127, dtype=torch.float32, device=dev)
        gs = torch.ones(1, dtype=torch.float32, device=dev)

        # ── lo que corre HOY: peso fp8, activacion fp16 -> HMMA + dequant en fp ─────────────
        nb8 = N * H + (H // G) * N * 2
        b_q8 = torch.randint(-(2**31), 2**31 - 1, (H // 16, N * 16 // 4),
                             dtype=torch.int32, device=dev)
        ws8 = torch.zeros(N // 64 * 16, dtype=torch.int32, device=dev)

        def correr_fp8():
            return torch.ops._C.marlin_gemm(
                x, None, b_q8, None, b_s, None, None, None, vacio, vacio, ws8,
                scalar_types.float8_e4m3fn.id, M, N, H, True, False, True, False)

        try:
            t8 = medir(correr_fp8, a.rep)
            print(f"{'HOY  fp8 + act fp16 (HMMA)  M=' + str(M):<34} {t8:9.1f} "
                  f"{nb8/t8/1e3:8.1f} {100*nb8/t8/1e3/techo_gbs:11.0f}%   {nb8/1e6:.0f} MB")
        except Exception as e:
            print(f"{'fp8 M=' + str(M):<34} falla: {str(e)[:60]}")

        # ── solo el peso a int4, activacion sigue en fp16 -> HMMA igual ────────────────────
        def correr():
            return torch.ops._C.marlin_gemm(
                x, None, b_q, None, b_s, None, None, None, vacio, vacio, ws,
                scalar_types.uint4b8.id, M, N, H, True, False, True, False)

        try:
            t = medir(correr, a.rep)
            print(f"{'int4 + act fp16 (HMMA)      M=' + str(M):<34} {t:9.1f} {nb/t/1e3:8.1f} "
                  f"{100*nb/t/1e3/techo_gbs:11.0f}%   {nb/1e6:.0f} MB")
        except Exception as e:
            print(f"{'int4+fp16 M=' + str(M):<34} falla: {str(e)[:60]}")

        # ── int4 + activacion int8 + escalas int16: IMMA puro, cero fp (PN130) ─────────────
        try:
            t4 = medir(lambda: torch.ops._C.marlin_gemm(
                xq, None, b_q, None, b_s, a_s, gs, None, vacio, vacio, ws,
                scalar_types.uint4b8.id, M, N, H, True, False, True, False), a.rep)
            print(f"{'int4 + act int8 (IMMA)      M=' + str(M):<34} {t4:9.1f} {nb/t4/1e3:8.1f} "
                  f"{100*nb/t4/1e3/techo_gbs:11.0f}%   {nb/1e6:.0f} MB")
        except Exception as e:
            print(f"{'int4+int8 M=' + str(M):<34} falla: {str(e)[:70]}")
        print()


if __name__ == "__main__":
    main()
