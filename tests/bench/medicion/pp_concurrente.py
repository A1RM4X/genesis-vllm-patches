#!/usr/bin/env python3
"""Prefill CONCURRENTE: N prompts largos y frios a la vez, max_tokens=1.

pp.py manda una peticion por vez, y asi no se ve como se comporta el motor con el patron real
(un hilo principal largo mas varios subagentes). Con varias en vuelo cambian el armado de los
lotes, el tamano de los chunks y cuanto se pisan la comunicacion y el computo.

Reporta el agregado (tokens de prompt por segundo de pared, sumando todas las peticiones) y el
TTFT de cada una.

Uso: pp_concurrente.py <tag> [n_concurrentes] [repeticiones] [palabras_filler]
"""
import json
import statistics
import sys
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8320/v1/chat/completions"
H = {"Content-Type": "application/json", "Authorization": "Bearer <REDACTADO: clave rotada 2026-09-19>"}
TAG = sys.argv[1] if len(sys.argv) > 1 else "cc"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 5
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
FILLER_N = int(sys.argv[4]) if len(sys.argv) > 4 else 650

FILLER = ("La expedicion avanzo entre grietas de hielo mientras los sensores "
          "registraban anomalias termicas en el subsuelo de la luna helada, y "
          "el comandante anoto cada lectura en su bitacora personal. ")

res = []
lock = threading.Lock()


def uno(i: int, rep: int):
    # tag unico por peticion: si se repite el prefijo, el cache de prefijos se come el prefill
    cuerpo = json.dumps({
        "model": "qwen3.8",
        "messages": [{"role": "user",
                      "content": f"[{TAG}-{i}-{rep}-{time.time()}] Resumi:\n\n" + FILLER * FILLER_N}],
        "max_tokens": 1, "temperature": 0,
    }).encode()
    t0 = time.perf_counter()
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(BASE, cuerpo, H), timeout=900))
        dt = time.perf_counter() - t0
        with lock:
            res.append((r["usage"]["prompt_tokens"], dt))
    except Exception as e:
        with lock:
            res.append((0, -1.0))
        print(f"  ERR {i}: {str(e)[:100]}", flush=True)


print(f"{N} peticiones concurrentes x {REPS} rondas")
for rep in range(REPS):
    res.clear()
    hilos = [threading.Thread(target=uno, args=(i, rep)) for i in range(N)]
    t0 = time.perf_counter()
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()
    pared = time.perf_counter() - t0
    buenos = [(t, d) for t, d in res if d > 0]
    if not buenos:
        print(f"  ronda {rep}: todas fallaron")
        continue
    toks = sum(t for t, _ in buenos)
    ttfts = sorted(d for _, d in buenos)
    print(f"  ronda {rep}: {len(buenos)}/{N} ok | {toks} tok en {pared:.2f}s = "
          f"{toks/pared:.0f} tok/s agregados | TTFT min {ttfts[0]:.1f}s "
          f"mediana {statistics.median(ttfts):.1f}s max {ttfts[-1]:.1f}s")
