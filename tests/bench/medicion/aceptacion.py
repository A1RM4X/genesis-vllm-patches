#!/usr/bin/env python3
"""Aceptacion del MTP por DELTA de contadores, que es la unica forma honesta de medirla.

Los contadores de Prometheus son ACUMULADOS desde que arranco el server, asi que leerlos sueltos
mezcla todo el trafico anterior. Aca se toma una foto, se genera carga, se toma otra, y se
reporta la diferencia.

Uso: aceptacion.py <tag> [tokens_de_salida]
"""
import json
import re
import sys
import time
import urllib.request
import os

BASE = "http://127.0.0.1:8320"
H = {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ.get("VLLM_API_KEY", "")}
TAG = sys.argv[1] if len(sys.argv) > 1 else "acc"
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 400


def foto():
    t = urllib.request.urlopen(f"{BASE}/metrics", timeout=30).read().decode()
    d = {}
    for clave in ("num_draft_tokens_total", "num_accepted_tokens_total", "num_drafts_total"):
        m = re.search(rf"vllm:spec_decode_{clave}\{{[^}}]*\}} ([0-9.e+]+)", t)
        d[clave] = float(m.group(1)) if m else 0.0
    return d


def generar(n=3):
    temas = ["Explica en detalle como funciona un recolector de basura generacional.",
             "Escribi una funcion en Python que resuelva el problema de la mochila con memoizacion,"
             " y explica cada paso.",
             "Resumi las diferencias entre TCP y UDP y cuando conviene cada uno."]
    for i in range(n):
        cuerpo = json.dumps({"model": "qwen3.8",
                             "messages": [{"role": "user",
                                           "content": f"[{TAG}-{i}-{time.time()}] " + temas[i % len(temas)]}],
                             "max_tokens": MAXTOK, "temperature": 0}).encode()
        urllib.request.urlopen(urllib.request.Request(f"{BASE}/v1/chat/completions", cuerpo, H),
                               timeout=600).read()


a = foto()
generar()
b = foto()
draft = b["num_draft_tokens_total"] - a["num_draft_tokens_total"]
acep = b["num_accepted_tokens_total"] - a["num_accepted_tokens_total"]
drafts = b["num_drafts_total"] - a["num_drafts_total"]
if draft <= 0:
    print("sin trafico de MTP en la ventana")
else:
    print(f"{TAG}: borradores={int(drafts)} propuestos={int(draft)} aceptados={int(acep)} "
          f"-> aceptacion {100*acep/draft:.1f}% | {acep/max(1,drafts):.2f} tok extra por paso")
