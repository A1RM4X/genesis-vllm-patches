#!/usr/bin/env python3
"""Calibra el suavizado por grupo a partir de las muestras de activacion.

Para cada lineal alimentado por una RMSNorm calcula:
  * ``s``: un factor por GRUPO de 128 canales de entrada (el maximo absoluto del grupo,
    normalizado a media 1). Se pliega exacto en las escalas de grupo del GPTQ y en el peso de la
    norma previa.
  * ``escala``: la escala estatica de activacion DESPUES de suavizar, que es lo unico que hace
    falta en ejecucion.

Medido: alfa=1,0 y el MAXIMO del grupo (no un percentil) es lo mejor; ver
tests/proto/estaticos_offline.py.

Uso: calibrar_suave.py <dir_muestras> <salida.json> [margen]
"""
import glob
import json
import os
import sys

import torch

DIR = sys.argv[1] if len(sys.argv) > 1 else "muestras"
SALIDA = sys.argv[2] if len(sys.argv) > 2 else "suave.json"
MARGEN = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
G = 128
SITIOS = ("qkv_proj", "in_proj_qkvz", "gate_up_proj")


def limpiar(x):
    """Saca relleno del prefill chunked: NaN (memoria sin inicializar) y filas en cero."""
    bien = torch.isfinite(x).all(1) & (x.abs().amax(1) > 0)
    return x[bien]


fuera = {}
saltados = []
for f in sorted(glob.glob(os.path.join(DIR, "*.pt"))):
    base = os.path.basename(f)[:-3]
    if not any(base.endswith(t) for t in SITIOS):
        continue
    x = limpiar(torch.load(f, map_location="cpu").float())
    if x.shape[0] < 32:
        saltados.append(base)
        continue
    k = x.shape[1]
    if k % G:
        saltados.append(base)
        continue
    s = x.abs().reshape(x.shape[0], -1, G).amax(dim=(0, 2)).clamp_min(1e-9)
    s = s / s.mean()                       # media 1: no mueve el rango global
    xs = x / s.repeat_interleave(G)
    escala = float(xs.abs().max()) * (1.0 + MARGEN) / 127.0
    # el nombre del archivo tiene los puntos cambiados por guiones bajos
    fuera[base] = {"s": s.tolist(), "escala": escala}

with open(SALIDA, "w") as fh:
    json.dump(fuera, fh)

print(f"{len(fuera)} capas calibradas -> {SALIDA} (margen {MARGEN:.0%})"
      + (f"; {len(saltados)} salteadas" if saltados else ""))
if fuera:
    import statistics
    disp = [max(v["s"]) / min(v["s"]) for v in fuera.values()]
    print(f"dispersion del factor dentro de una capa (max/min): "
          f"mediana {statistics.median(disp):.1f}x, max {max(disp):.1f}x")
