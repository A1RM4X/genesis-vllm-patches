#!/usr/bin/env python3
"""Como escala el decode con varios pedidos a la vez.

Por que importa: mirar un kernel y decir "esta al techo de DRAM, no hay nada que hacer" solo vale
con UN pedido. Los tres kernels mas caros piden 99 KB de los 100 KB de shared de un SM, asi que
ocupan los 82 SMs con 8-17% de ocupacion de warps e impiden que entre nadie mas. Si el decode no
escala con la concurrencia, ahi esta la factura — y el patron real de carga son 1 hilo largo mas
varios subagentes.

Uso: decode_concurrente.py [max_concurrentes]
"""
import json
import sys
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8320"
H = {"Content-Type": "application/json", "Authorization": "Bearer <REDACTADO: clave rotada 2026-09-19>"}
MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 6
TOK = 300


def pedir(i, salida, etiqueta):
    cuerpo = json.dumps({
        "model": "qwen3.8",
        "messages": [{"role": "user",
                      "content": f"[{etiqueta}-{i}-{time.time()}] Escribi una implementacion "
                                 f"comentada de un arbol AVL en Python, variante {i}."}],
        "max_tokens": TOK, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False}}).encode()
    t0 = time.time()
    r = json.load(urllib.request.urlopen(
        urllib.request.Request(f"{BASE}/v1/chat/completions", cuerpo, H), timeout=900))
    salida[i] = (r["usage"]["completion_tokens"], time.time() - t0)


print(f"{'concurrentes':>13}{'tok/s agregado':>16}{'tok/s por pedido':>18}{'escala':>9}")
base = None
# Hasta 6, que es el patron real (un hilo largo + subagentes). Mas arriba el dato existe y esta
# anotado —  con 12 el throughput CAE de 611 a 404 tok/s porque Marlin relee el peso con M>64 —
# pero no es lo que manda hoy: pasar 12 pedidos a la vez no es el caso de uso.
ESCALERA = [1, 2, 4, 6]
for n in ESCALERA:
    if n > MAX:
        break
    salida = {}
    hilos = [threading.Thread(target=pedir, args=(i, salida, f"cc{n}")) for i in range(n)]
    t0 = time.time()
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()
    pared = time.time() - t0
    tot = sum(v[0] for v in salida.values())
    agregado = tot / pared
    porped = sum(v[0] / v[1] for v in salida.values()) / n
    if base is None:
        base = agregado
    print(f"{n:>13}{agregado:16.1f}{porped:18.1f}{agregado/base:8.2f}x")
