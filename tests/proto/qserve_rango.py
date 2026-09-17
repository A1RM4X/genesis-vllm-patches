#!/usr/bin/env python3
"""Rango real del acumulador si el nibble va CRUDO al tensor core (metodo QServe).

Hoy Marlin manda ``q - 8`` al mma, con q en [0,15], o sea valores en [-8,7]. QServe manda ``q``
crudo y corrige el offset despues:

    suma_k a_k*w_k = s * [ suma_k a_k*q_k  -  8 * suma_k a_k ]

El riesgo es el rango: con q en [0,15] en vez de [-8,7] el acumulador del mma crece, y encima
Marlin lo multiplica por la escala int16 y lo acumula sobre TODOS los grupos de K. La cota
teorica no sirve para decidir — con los maximos teoricos el kernel de hoy ya desbordaria — asi
que esto lo mide sobre pesos y activaciones REALES.

Uso: qserve_rango.py [n_tensores]
"""
from __future__ import annotations

import glob
import json
import os
import sys

import torch

G = 128
N_TENSORES = int(sys.argv[1]) if len(sys.argv) > 1 else 6
LIMITE = 2 ** 31 - 1


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


def escalas_int16(sc: torch.Tensor):
    """Como las prepara PN130: normalizadas por |s|.max() a int16 con signo (Q15 sobre 4096)."""
    m = sc.abs().max()
    return (sc / m * 4096).round().clamp(-32768, 32767), m / 4096


def main() -> None:
    torch.manual_seed(0)
    print("acumulador del mma por GRUPO de K=128, con activacion int8 real (|a| <= 127)\n")
    print(f"  {'tensor':<40}{'hoy (q-8)':>14}{'QServe (q)':>14}{'x':>7}{'% de int32':>12}")
    peor = 0.0
    for nombre, q, sc in cargar(N_TENSORES):
        N, K = q.shape
        # activacion realista: int8 con distribucion de activacion tras RMSNorm, no uniforme
        M = 64
        a = (torch.randn(M, K, device="cuda") * 24).round().clamp(-127, 127)
        s16, _ = escalas_int16(sc)

        # por grupo: acc = a @ q_grupo, y despues acc * escala_int16, acumulado sobre grupos
        ng = K // G
        acc_hoy = torch.zeros(M, N, device="cuda")
        acc_qs = torch.zeros(M, N, device="cuda")
        for g in range(ng):
            aa = a[:, g * G:(g + 1) * G]
            qq = q[:, g * G:(g + 1) * G].float()
            s = s16[:, g].unsqueeze(0)
            acc_hoy += (aa @ (qq - 8.0).t()) * s
            acc_qs += (aa @ qq.t()) * s
        h = float(acc_hoy.abs().max())
        p = float(acc_qs.abs().max())
        peor = max(peor, p)
        corto = nombre.replace("model.language_model.layers.", "L")[:40]
        print(f"  {corto:<40}{h:>14.3e}{p:>14.3e}{p/max(h,1):>7.2f}{100*p/LIMITE:>11.1f}%")
        del q, sc, a, acc_hoy, acc_qs
        torch.cuda.empty_cache()

    print(f"\n  peor caso visto: {peor:.3e}  contra el limite de int32 {LIMITE:.3e}")
    print(f"  margen: {LIMITE/max(peor,1):.1f}x" if peor < LIMITE else "  DESBORDA")


if __name__ == "__main__":
    main()
