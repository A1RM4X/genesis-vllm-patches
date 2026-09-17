#!/usr/bin/env python3
"""Escalera de concurrencia: N pedidos en paralelo, todos con el mismo trabajo.

Interesa el throughput AGREGADO y el que ve cada pedido. Con MTP K=3 cada pedido aporta 4 tokens
al lote de decode, asi que 12 pedidos son ~48 filas: justo la zona donde estaba el acantilado.
"""
from __future__ import annotations

import json
import statistics
import sys
import threading
import time
import urllib.request

import os

HOST = os.environ.get("VLLM_HOST", "http://127.0.0.1:8320")
CLAVE = os.environ.get("VLLM_API_KEY", "<REDACTADO: clave rotada 2026-09-19>")
CAB = {"Content-Type": "application/json", "Authorization": f"Bearer {CLAVE}"}
URL = HOST + "/v1/chat/completions"
CONCURRENCIAS = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1
                                  else "1,2,4,6,8,12,16".split(","))]
PROMPT = ("Explica en detalle, paso a paso y con ejemplos concretos, como funciona un sistema "
          "de control de inventario con reposicion automatica: que datos guarda, como calcula "
          "el punto de reposicion, que pasa cuando un proveedor no cumple, y como se audita.")
MAX_TOKENS = 400


def modelo() -> str:
    with urllib.request.urlopen(urllib.request.Request(HOST + "/v1/models", headers=CAB)) as r:
        return json.load(r)["data"][0]["id"]


def uno(mdl: str, salida: list, i: int) -> None:
    # SIN streaming: el cliente de Python no da abasto con el SSE y topea en ~40 tok/s, o sea
    # uno mide el cliente. El conteo real sale de `usage`.
    cuerpo = json.dumps({
        "model": mdl, "max_tokens": MAX_TOKENS, "temperature": 0.0,
        "messages": [{"role": "user", "content": f"[{i}-{time.time()}] " + PROMPT}],
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    t0 = time.perf_counter()
    r = json.load(urllib.request.urlopen(
        urllib.request.Request(URL, cuerpo, CAB), timeout=1200))
    salida[i] = (0.0, time.perf_counter() - t0, r["usage"]["completion_tokens"])


def main() -> None:
    mdl = modelo()
    print(f"  {'pedidos':>8}{'pared s':>9}{'tokens':>8}{'tok/s total':>13}"
          f"{'por pedido':>12}{'s/pedido':>10}{'escala':>9}")
    base = None
    for c in CONCURRENCIAS:
        salida = [None] * c
        hilos = [threading.Thread(target=uno, args=(mdl, salida, i)) for i in range(c)]
        t0 = time.perf_counter()
        for h in hilos:
            h.start()
        for h in hilos:
            h.join()
        pared = time.perf_counter() - t0
        toks = sum(s[2] for s in salida)
        ttfts = [s[0] for s in salida]
        agregado = toks / pared
        if base is None:
            base = agregado
        print(f"  {c:>8}{pared:>9.1f}{toks:>8}{agregado:>13.1f}{agregado / c:>12.1f}"
              f"{statistics.median([s[1] for s in salida]):>10.2f}{agregado / base:>8.2f}x")
        time.sleep(2)


if __name__ == "__main__":
    main()
