#!/usr/bin/env python3
"""Verifica el GEMM de Marlin (PN130) contra float64, con el peso pasado por el REPACK REAL.

El intento anterior generaba `b_q` como bits al azar, y eso NO sirve: Marlin espera el peso
pre-permutado por ``gptq_marlin_repack``, y esa permutacion depende de la configuracion de
tiling, asi que cada thread_m_blocks interpreta los mismos bits como una matriz distinta. Aca se
parte de la matriz LOGICA de nibbles, se la empaqueta en formato GPTQ, se la pasa por el repack
de verdad y se compara contra la misma matriz logica en float64.

    C[m,n] = sum_g ( sum_{k in g} a[m,k] * (q[k,n] - 8) ) * s[g,n] * a_esc[m]

El nibble es uint4b8: el valor real es nibble - 8.

Uso:  marlin_correcto.py [--determinismo]
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")

G = 128
MES = [int(x) for x in os.environ.get("MES", "16,32,64,70,80,88,96,104,112,128,160").split(",")]
FORMAS = [tuple(int(x) for x in f.split("x"))
          for f in os.environ.get("FORMAS", "2048x2048,5120x5120").split(",")]


def empacar_gptq(q: torch.Tensor) -> torch.Tensor:
    """q [K, N] con nibbles en [0,15] -> qweight [K/8, N] int32, formato GPTQ."""
    K, N = q.shape
    out = torch.zeros((K // 8, N), dtype=torch.int32, device=q.device)
    for i in range(8):
        out |= (q[i::8].to(torch.int32) & 0xF) << (4 * i)
    return out


def referencia(a, q, esc, a_esc) -> torch.Tensor:
    """float64 representa exacto todo lo que aparece aca (el maximo es ~1e10, muy debajo de 2^53)."""
    M, K = a.shape
    N = q.shape[1]
    out = torch.zeros(M, N, dtype=torch.float64, device=a.device)
    for g in range(K // G):
        aa = a[:, g * G:(g + 1) * G].double()
        qq = q[g * G:(g + 1) * G, :].double() - 8.0
        out += (aa @ qq) * esc[g].double().unsqueeze(0)
    return out * a_esc.double().unsqueeze(1)


def main() -> None:
    torch.ops.load_library(os.environ["S16_SO"])
    from vllm.scalar_type import scalar_types
    determinismo = "--determinismo" in sys.argv
    dev = "cuda"
    vacio = torch.empty(0, dtype=torch.int32, device=dev)
    peor = 0.0
    malos = []

    for N, K in FORMAS:
        torch.manual_seed(7)
        q = torch.randint(0, 16, (K, N), dtype=torch.int32, device=dev)
        b_q = torch.ops._C.gptq_marlin_repack(empacar_gptq(q), vacio, K, N, 4, True)
        # escalas chicas para que el resultado quede holgado dentro del fp16 de salida
        esc = torch.randint(-64, 64, (K // G, N), dtype=torch.int16, device=dev)
        # Las escalas tambien van permutadas, igual que el peso: es lo que hace vLLM al cargar el
        # modelo. Con escalas constantes la permutacion no se nota, y por eso el caso trivial daba
        # exacto mientras el aleatorio no.
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_permute_scales)
        b_s = marlin_permute_scales(esc.view(torch.float16), K, N, G, True)
        ws = torch.zeros(82 * 2 * 16, dtype=torch.int32, device=dev)
        print(f"\n  {N}x{K}")
        for M in MES:
            a = (torch.randn(M, K, device=dev) * 24).round().clamp(-127, 127).to(torch.int8)
            ae = 1 / 127 / 4096
            a_s = torch.full((M, 1), ae, dtype=torch.float32, device=dev)

            def una():
                ws.zero_()
                return torch.ops.genesis_marlin.marlin_gemm_s16(
                    a, None, b_q, None, b_s, a_s, None, None, None, None, None, ws,
                    scalar_types.uint4b8.id, M, N, K, True, False, True, False).clone()

            out = una()
            torch.cuda.synchronize()
            ref = referencia(a, q, esc, torch.full((M,), ae, device=dev))
            d = float((out.double() - ref).abs().max())
            # Con datos de media cero hay cancelacion masiva: terminos de ~1e5 que suman ~1.
            # Dividir por el resultado final infla el error hasta el absurdo. La cota correcta es
            # la magnitud del ACUMULADOR, que es contra lo que redondea el fp16 de salida.
            cota = float(referencia(a.abs(), (q - 8).abs() + 8, esc.abs(),
                                    torch.full((M,), ae, device=dev)).abs().max())
            rel = d / max(cota, 1e-9)
            peor = max(peor, rel)
            ok = rel < 1e-3                       # el redondeo de la salida fp16 (2^-11)
            if not ok:
                malos.append((N, K, M))
            extra = ""
            if determinismo:
                rep = [una() for _ in range(3)]
                torch.cuda.synchronize()
                dd = max(float((r.float() - out.float()).abs().max()) for r in rep)
                extra = f"   repeticion: {'estable' if dd == 0 else f'CARRERA {dd:.3f}'}"
            print(f"    M={M:4d}  max|dif|={d:10.4f}  cota={cota:11.1f}  rel={rel:8.2e}  "
                  f"{'OK ' if ok else 'MAL'}{extra}")
        del q, b_q, esc, ws
        torch.cuda.empty_cache()
    print(f"\n  peor relativo: {peor:.2e}   "
          f"{'TODO BIEN' if not malos else 'MAL en ' + str(malos)}")


if __name__ == "__main__":
    main()
