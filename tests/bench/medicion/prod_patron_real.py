#!/usr/bin/env python3
"""El patron real de carga: un hilo largo (60-100k) mas varios subagentes cortos.

Es el caso que importa (ver la nota de memoria del patron opencode) y el que decide si conviene
subir --max-num-seqs: con muchos slots y contextos largos puede aparecer preempcion, que sale
mas caro que hacer cola.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request
import uuid

H = {"Content-Type": "application/json", "Authorization": "Bearer <REDACTADO: clave rotada 2026-09-19>"}
BASE = "http://127.0.0.1:8320"
SUBAGENTES = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def relleno(tokens: int, sal: str) -> str:
    base = ("El modulo de facturacion toma los remitos confirmados, agrupa por cliente y periodo, "
            "aplica la lista de precios vigente, calcula impuestos por jurisdiccion y emite el "
            "comprobante electronico, dejando la respuesta del organismo en el historial. ")
    pal = int(tokens / 1.4)
    return sal + " " + " ".join((base * (pal // len(base.split()) + 2)).split()[:pal])


def pedir(prompt, mx, salida, i):
    cuerpo = json.dumps({
        "model": "qwen3.8", "max_tokens": mx, "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": False}}).encode()
    t0 = time.perf_counter()
    r = json.load(urllib.request.urlopen(
        urllib.request.Request(BASE + "/v1/chat/completions", cuerpo, H), timeout=1800))
    salida[i] = (r["usage"]["completion_tokens"], time.perf_counter() - t0)


def contador(nombre: str) -> float:
    r = urllib.request.urlopen(urllib.request.Request(BASE + "/metrics", headers=H), timeout=30)
    for ln in r.read().decode().splitlines():
        if ln.startswith(nombre) and "_created" not in ln:
            return float(ln.split()[-1])
    return 0.0


def main() -> None:
    p0 = contador("vllm:num_preemptions_total")
    salida = [None] * (SUBAGENTES + 1)
    largo = relleno(80000, f"[hilo {uuid.uuid4().hex}] ")
    hilos = [threading.Thread(target=pedir, args=(
        largo + "\n\nResumi el circuito completo y sus riesgos.", 500, salida, 0))]
    for i in range(SUBAGENTES):
        p = relleno(6000, f"[sub {uuid.uuid4().hex}] ") + "\n\nExtrae los puntos de control."
        hilos.append(threading.Thread(target=pedir, args=(p, 300, salida, i + 1)))
    t0 = time.perf_counter()
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()
    pared = time.perf_counter() - t0
    p1 = contador("vllm:num_preemptions_total")
    toks = sum(s[0] for s in salida)
    cortos = sorted(s[1] for s in salida[1:])
    print(f"  1 hilo de 80k + {SUBAGENTES} subagentes de 6k")
    print(f"  pared={pared:.1f}s   tokens={toks}   tok/s={toks / pared:.1f}   "
          f"PREEMPCIONES={p1 - p0:.0f}")
    print(f"  hilo largo: {salida[0][0]} tokens en {salida[0][1]:.1f}s")
    print(f"  subagentes: mediana {cortos[len(cortos) // 2]:.1f}s   peor {cortos[-1]:.1f}s")


if __name__ == "__main__":
    main()
