#!/usr/bin/env python3
"""Aceptacion del MTP durante una carga de codigo, por delta de los contadores de Prometheus.

Los contadores son acumulados y globales (y se contaminan con trafico ajeno), asi que hay que
leerlos antes y despues y quedarse con la diferencia.
"""
from __future__ import annotations

import json
import re
import sys
import threading
import time
import urllib.request
import os

H = {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ.get("VLLM_API_KEY", "")}
BASE = "http://127.0.0.1:8320"
CONC = int(sys.argv[1]) if len(sys.argv) > 1 else 1
TOK = 600


def metricas() -> dict:
    r = urllib.request.urlopen(urllib.request.Request(BASE + "/metrics", headers=H), timeout=30)
    d = {}
    for ln in r.read().decode().splitlines():
        m = re.match(r"(vllm:[a-z_]+(?:\{[^}]*\})?)\s+([0-9.e+]+)$", ln)
        if m:
            d[m.group(1)] = float(m.group(2))
    return d


def val(d: dict, pref: str) -> float:
    return sum(v for k, v in d.items() if k.startswith(pref) and "_created" not in k)


def pedir(i, salida):
    cuerpo = json.dumps({
        "model": "qwen3.8", "max_tokens": TOK, "temperature": 0.0,
        "messages": [{"role": "user", "content":
                      f"[{i}-{time.time()}] Escribi en Python, con comentarios, una tabla hash "
                      f"con direccionamiento abierto y redimension, variante {i}."}],
        "chat_template_kwargs": {"enable_thinking": False}}).encode()
    t0 = time.perf_counter()
    r = json.load(urllib.request.urlopen(
        urllib.request.Request(BASE + "/v1/chat/completions", cuerpo, H), timeout=900))
    salida[i] = (r["usage"]["completion_tokens"], time.perf_counter() - t0)


def main() -> None:
    a = metricas()
    salida = [None] * CONC
    hilos = [threading.Thread(target=pedir, args=(i, salida)) for i in range(CONC)]
    t0 = time.perf_counter()
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()
    pared = time.perf_counter() - t0
    b = metricas()

    dr = val(b, "vllm:spec_decode_num_drafts_total") - val(a, "vllm:spec_decode_num_drafts_total")
    dt = (val(b, "vllm:spec_decode_num_draft_tokens_total")
          - val(a, "vllm:spec_decode_num_draft_tokens_total"))
    ac = (val(b, "vllm:spec_decode_num_accepted_tokens_total")
          - val(a, "vllm:spec_decode_num_accepted_tokens_total"))
    gen = val(b, "vllm:generation_tokens_total") - val(a, "vllm:generation_tokens_total")
    toks = sum(s[0] for s in salida)
    print(f"  concurrencia={CONC}  pared={pared:.1f}s  tokens={toks}  "
          f"tok/s={toks / pared:.1f}")
    print(f"  pasos (drafts)={dr:.0f}   pasos/s={dr / pared:.1f}   "
          f"tokens por paso={gen / max(dr, 1):.2f}")
    print(f"  aceptacion={ac / max(dt, 1):.1%}  ({ac:.0f} de {dt:.0f} tokens borrador)")
    for pos in range(3):
        k = f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{engine="0",model_name="qwen3.8",position="{pos}"}}'
        if k in a and k in b:
            print(f"    posicion {pos}: {(b[k] - a[k]) / max(dr, 1):.1%}")


if __name__ == "__main__":
    main()
