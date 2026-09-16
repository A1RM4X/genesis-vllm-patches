#!/usr/bin/env python3
"""Prefill puro: prompt frio largo, max_tokens=1, tok/s = prompt_tokens / TTFT.
Tag unico por corrida para no comer prefix cache."""
import json, sys, time, urllib.request
BASE = "http://127.0.0.1:8320/v1/chat/completions"
KEY = "<REDACTADO: clave rotada 2026-09-19>"
TAG = sys.argv[1]
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 950
FILLER = ("La expedicion avanzo entre grietas de hielo mientras los sensores "
          "registraban anomalias termicas en el subsuelo de la luna helada, y "
          "el comandante anoto cada lectura en su bitacora personal. ")
for i in range(3):
    body = json.dumps({
        "model": "qwen3.8",
        "messages": [{"role": "user",
                      "content": f"[{TAG}-{i}-{time.time()}] Resumi:\n\n" + FILLER * REPS}],
        "max_tokens": 1, "temperature": 0,
    }).encode()
    req = urllib.request.Request(BASE, body, {
        "Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.perf_counter()
    r = json.load(urllib.request.urlopen(req, timeout=600))
    dt = time.perf_counter() - t0
    n = r["usage"]["prompt_tokens"]
    print(f"  corrida {i}: {n} tok en {dt:.2f}s = {n/dt:.0f} tok/s de prefill")
