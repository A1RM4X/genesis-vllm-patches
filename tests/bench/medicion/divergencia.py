#!/usr/bin/env python3
"""Divergencia token a token contra una referencia: genera greedy con prompts fijos y guarda el
texto; despues compara dos corridas (largo del prefijo comun y porcentaje de tokens iguales).

Uso:
  divergencia.py grabar <etiqueta> [n_tokens]      corre y guarda /tmp/divergencia_<etiqueta>.json
  divergencia.py comparar <ref> <otro>             compara dos etiquetas ya grabadas
"""
import json, sys, urllib.request
import os

BASE = "http://127.0.0.1:8320/v1/chat/completions"
KEY = os.environ.get("VLLM_API_KEY", "")

PROMPTS = [
    "Escribí una función en Python que reciba una lista de diccionarios con las claves 'nombre' y "
    "'edad' y devuelva los nombres ordenados por edad descendente. Solo el código.",
    "Explicá en tres oraciones qué es la cuantización de la caché KV en inferencia de LLMs.",
    "Devolvé un JSON con las claves: nombre (string), version (string), dependencias (lista de "
    "strings) para un paquete ficticio llamado pagoda. Solo el JSON.",
    "Escribí un archivo docker-compose.yml mínimo con un servicio nginx en el puerto 8080 y un "
    "volumen para /usr/share/nginx/html. Solo el YAML.",
    "Listá los primeros 15 números primos separados por comas, sin texto adicional.",
    "Escribí una consulta SQL que devuelva, por cliente, el total facturado en 2025 ordenado de "
    "mayor a menor, usando las tablas clientes(id, nombre) y facturas(id, cliente_id, fecha, total).",
]


def generar(n_tok):
    salidas = []
    for i, p in enumerate(PROMPTS):
        cuerpo = {"model": "qwen3.8", "messages": [{"role": "user", "content": p}],
                  "temperature": 0.0, "top_p": 1.0, "max_tokens": n_tok, "seed": 7,
                  "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(BASE, json.dumps(cuerpo).encode(),
                                     {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.load(r)
        salidas.append(d["choices"][0]["message"].get("content") or "")
        print(f"  prompt {i}: {len(salidas[-1])} caracteres", flush=True)
    return salidas


def comparar(a, b):
    print(f"{'prompt':>7} {'prefijo comun':>14} {'de':>7} {'% igual':>9}")
    tot_pref = tot_len = 0
    for i, (x, y) in enumerate(zip(a, b)):
        n = min(len(x), len(y))
        k = 0
        while k < n and x[k] == y[k]:
            k += 1
        iguales = sum(1 for u, v in zip(x, y) if u == v)
        print(f"{i:7d} {k:14d} {len(x):7d} {100 * iguales / max(len(x), 1):8.1f}%")
        tot_pref += k
        tot_len += len(x)
    print(f"  TOTAL: prefijo comun {tot_pref} de {tot_len} caracteres ({100*tot_pref/max(tot_len,1):.1f}%)")


if __name__ == "__main__":
    if sys.argv[1] == "grabar":
        tag = sys.argv[2]
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 400
        s = generar(n)
        json.dump(s, open(f"/tmp/divergencia_{tag}.json", "w"))
        print(f"grabado /tmp/divergencia_{tag}.json")
    else:
        a = json.load(open(f"/tmp/divergencia_{sys.argv[2]}.json"))
        b = json.load(open(f"/tmp/divergencia_{sys.argv[3]}.json"))
        comparar(a, b)
