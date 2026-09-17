#!/usr/bin/env python3
"""Valida el algebra de QServe end to end, antes de escribir una linea de CUDA.

La identidad que hay que sostener, por grupo de K:

    suma_k a_k * (q_k - 8) * s  ==  s * [ suma_k a_k*q_k  -  8 * suma_k a_k ]
    \\_______ lo que hace Marlin hoy ____/  \\___ mma con q crudo ___/  \\_ correccion _/

Si esto no da BIT A BIT igual en enteros, no hay nada que programar. Y tiene que dar exacto: no
es una aproximacion como LiquidQuant, es sacar factor comun.

Se valida sobre los pesos reales del checkpoint y activaciones int8, con la cadena completa que
usa PN130 — incluida la escala int16 con signo, que es donde podria aparecer una sorpresa.

Uso: qserve_algebra.py [n_tensores]
"""
from __future__ import annotations

import glob
import json
import os
import sys

import torch

G = 128
N_TENSORES = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def cargar(n: int):
    from safetensors import safe_open
    base = glob.glob("/root/.cache/huggingface/hub/models--noon-at-cgn--*/snapshots/*")[0]
    idx = json.load(open(os.path.join(base, "model.safetensors.index.json")))["weight_map"]
    qs = sorted(k for k in idx if k.endswith(".weight_packed") and ".layers." in k)
    paso = max(1, len(qs) // n)
    for nombre in qs[::paso][:n]:
        pref = nombre[: -len(".weight_packed")]
        with safe_open(os.path.join(base, idx[nombre]), framework="pt") as h:
            qw = h.get_tensor(nombre).cuda()
            sc = h.get_tensor(pref + ".weight_scale").cuda().float()
        N, K = qw.shape[0], qw.shape[1] * 8
        q = torch.zeros((N, K), dtype=torch.int32, device="cuda")
        for i in range(8):
            q[:, i::8] = (qw >> (4 * i)) & 0xF
        yield pref, q, sc
        del q, qw, sc
        torch.cuda.empty_cache()


def main() -> None:
    torch.manual_seed(0)
    print("identidad de QServe, en ENTEROS, sobre pesos reales\n")
    print(f"  {'tensor':<40}{'max|dif|':>12}{'exacto':>9}{'corr/total':>12}")
    todo_exacto = True
    for nombre, q, sc in cargar(N_TENSORES):
        N, K = q.shape
        M, ng = 16, K // G
        # float64 y no int64: matmul entero no existe en CUDA, y float64 representa enteros
        # EXACTOS hasta 2^53. Aca el maximo es ~1,4e13 (acumulador x escala x 40 grupos), muy por
        # debajo, asi que la comparacion sigue siendo exacta.
        a = (torch.randn(M, K, device="cuda") * 24).round().clamp(-127, 127).double()
        qi = q.double()
        # escala int16 con signo, como la deja PN130 (normalizada por |s|.max() a Q15/4096)
        s16 = (sc / sc.abs().max() * 4096).round().double()

        hoy = torch.zeros(M, N, dtype=torch.float64, device="cuda")
        qserve = torch.zeros(M, N, dtype=torch.float64, device="cuda")
        magnitud_corr = 0
        for g in range(ng):
            aa = a[:, g * G:(g + 1) * G]
            qq = qi[:, g * G:(g + 1) * G]
            s = s16[:, g].unsqueeze(0)                      # [1, N]
            # lo de hoy: el peso llega al mma ya con signo
            hoy += (aa @ (qq - 8).t()) * s
            # QServe: el mma ve el nibble crudo y el offset se corrige despues
            crudo = aa @ qq.t()
            corr = 8.0 * aa.sum(1, keepdim=True)            # [M, 1], NO depende de N
            qserve += (crudo - corr) * s
            magnitud_corr += float((corr * s).abs().sum())
        d = float((hoy - qserve).abs().max())
        ok = d == 0
        todo_exacto &= ok
        rel = magnitud_corr / max(float(hoy.abs().sum()), 1.0)
        corto = nombre.replace("model.language_model.layers.", "L")[:40]
        print(f"  {corto:<40}{d:>12.0f}{'SI' if ok else 'NO':>9}{rel:>11.2f}x")
        del q, sc, a, qi, hoy, qserve
        torch.cuda.empty_cache()

    print(f"\n  {'TODO EXACTO — el algebra cierra' if todo_exacto else 'HAY DIFERENCIAS'}")
    print("  corr/total: cuanto pesa la correccion contra el resultado; si fuera enorme,")
    print("  restarla podria perder digitos significativos aun siendo exacta en enteros.")


if __name__ == "__main__":
    main()
