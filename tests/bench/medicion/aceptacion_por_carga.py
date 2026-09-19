#!/usr/bin/env python3
"""Aceptacion del MTP DESGLOSADA POR TIPO DE CARGA, con las posiciones por separado.

Existe porque un numero solo de aceptacion no quiere decir nada: la misma config da 50% en prosa
libre y >90% copiando literales. Comparar dos corridas con cargas distintas es comparar cualquier
cosa. Y el acumulado de Prometheus mezcla TODO el trafico anterior del server, asi que siempre hay
que medir por delta.

Uso: aceptacion_por_carga.py [tokens_de_salida]
"""
import json
import re
import sys
import time
import urllib.request
import os

BASE = "http://127.0.0.1:8320"
H = {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ.get("VLLM_API_KEY", "")}
MAXTOK = int(sys.argv[1]) if len(sys.argv) > 1 else 300

CARGAS = {
    "prosa libre": [
        "Explica en detalle como funciona un recolector de basura generacional.",
        "Resumi las diferencias entre TCP y UDP y cuando conviene cada uno.",
    ],
    "codigo": [
        "Escribi una funcion en Python que resuelva la mochila con memoizacion. Solo el codigo.",
        "Escribi una clase LRUCache en Python con get y put en O(1). Solo el codigo.",
    ],
    "copia literal": [
        "Repeti exactamente este texto, sin cambiar nada:\n" + ("La rana canta al anochecer "
         "junto al arroyo de piedras frias. " * 12),
        "Copia esta lista tal cual:\n" + "\n".join(f"{i}. elemento numero {i} de la lista" for i in range(40)),
    ],
}


def foto():
    t = urllib.request.urlopen(f"{BASE}/metrics", timeout=30).read().decode()
    d = {}
    for c in ("num_draft_tokens_total", "num_accepted_tokens_total", "num_drafts_total"):
        m = re.search(rf"vllm:spec_decode_{c}\{{[^}}]*\}} ([0-9.e+]+)", t)
        d[c] = float(m.group(1)) if m else 0.0
    d["pos"] = [float(m) for m in re.findall(
        r'vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="\d+"\} ([0-9.e+]+)', t)]
    return d


print(f"{'carga':>14} {'borr':>6} {'prop':>6} {'acep':>6} {'tasa':>7}   por posicion")
for nombre, prompts in CARGAS.items():
    a = foto()
    for i, p in enumerate(prompts):
        cuerpo = json.dumps({"model": "qwen3.8",
                             "messages": [{"role": "user", "content": f"[{time.time()}-{i}] " + p}],
                             "max_tokens": MAXTOK, "temperature": 0}).encode()
        urllib.request.urlopen(urllib.request.Request(f"{BASE}/v1/chat/completions", cuerpo, H),
                               timeout=600).read()
    b = foto()
    dr = b["num_draft_tokens_total"] - a["num_draft_tokens_total"]
    ac = b["num_accepted_tokens_total"] - a["num_accepted_tokens_total"]
    nd = b["num_drafts_total"] - a["num_drafts_total"]
    if dr <= 0:
        print(f"{nombre:>14}  sin trafico de MTP")
        continue
    pos = [(y - x) / nd for x, y in zip(a["pos"], b["pos"])] if nd else []
    print(f"{nombre:>14} {int(nd):6d} {int(dr):6d} {int(ac):6d} {100*ac/dr:6.1f}%   "
          + " ".join(f"{100*p:5.1f}%" for p in pos))
