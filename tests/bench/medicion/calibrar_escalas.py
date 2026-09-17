#!/usr/bin/env python3
"""Convierte el volcado de act_stats en un archivo de escalas estaticas por capa.

La escala tiene que cubrir el rango de la capa sin recortar nada importante. Se usa el maximo
EXACTO observado (que act_stats guarda aparte del histograma) con un margen, dividido por 127:

    s = amax * (1 + MARGEN) / 127

El margen existe porque la calibracion ve una muestra del trafico, no todo: si despues aparece un
token un poco mas grande, se recorta en vez de desbordar. Con las distribuciones medidas (los
sitios alimentados por RMSNorm tienen p50 y p99,9 casi pegados) un margen del 25% cuesta 0,3 bits,
que es ruido al lado de los 8 que hay.

Uso: calibrar_escalas.py <act_stats.json> [salida.json] [margen]
"""
import json
import sys

ENTRADA = sys.argv[1] if len(sys.argv) > 1 else "act_stats_porcapa.json"
SALIDA = sys.argv[2] if len(sys.argv) > 2 else "escalas.json"
MARGEN = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25

d = json.load(open(ENTRADA))
escalas = {}
sin_amax = 0
for nombre, v in d.items():
    if not v:
        continue
    amax = v.get("amax")
    if amax is not None and (amax != amax or amax in (float("inf"), float("-inf"))):
        amax = None                       # NaN/inf: no sirve para calibrar
    if not amax:
        # respaldo: el borde superior del bin del maximo (el histograma va en log2, paso 0,25).
        # Es conservador por construccion, asi que sirve igual.
        if "p100" not in v:
            sin_amax += 1
            continue
        amax = 2.0 ** (v["p100"] + 0.125)
    escalas[nombre] = amax * (1.0 + MARGEN) / 127.0

with open(SALIDA, "w") as f:
    json.dump(escalas, f, indent=1)

por_tipo = {}
for n, s in escalas.items():
    por_tipo.setdefault(n.split(".")[-1], []).append(s)
print(f"{len(escalas)} capas -> {SALIDA} (margen {MARGEN:.0%})"
      + (f"  [{sin_amax} sin dato]" if sin_amax else ""))
print(f"{'tipo':>16} {'capas':>6} {'escala min':>12} {'escala max':>12}")
for t, ss in sorted(por_tipo.items()):
    print(f"{t:>16} {len(ss):6d} {min(ss):12.3e} {max(ss):12.3e}")
