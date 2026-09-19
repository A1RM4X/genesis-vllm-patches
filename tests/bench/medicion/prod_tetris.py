#!/usr/bin/env python3
"""Prueba de codigo de verdad: pedir un Tetris con reglas modernas y ver si el codigo CORRE.

No alcanza con mirar el texto: se extrae el bloque de python, se ejecuta en un proceso aparte
con un arnes que simula partidas, y se reporta que funciono y que no. Es la unica forma de que
"el modelo responde bien" signifique algo.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import urllib.request

import os

HOST = os.environ.get("VLLM_HOST", "http://127.0.0.1:8320")
CLAVE = os.environ.get("VLLM_API_KEY", os.environ.get("VLLM_API_KEY", ""))
CAB = {"Content-Type": "application/json", "Authorization": f"Bearer {CLAVE}"}
URL = HOST + "/v1/chat/completions"
PEDIDO = """Escribi un Tetris completo en Python, en un solo archivo, sin dependencias externas
(nada de pygame: la interfaz es texto).

Requisitos, todos obligatorios:
1. Las 7 piezas estandar (I, O, T, S, Z, J, L) con sus colores/letras.
2. Rotacion con el sistema SRS, incluyendo los wall kicks (las 5 pruebas de patada por rotacion,
   con la tabla distinta para la pieza I).
3. Cola de 7 piezas con bolsa aleatoria (7-bag), y vista de las proximas 3.
4. Hold de pieza, con la regla de que no se puede volver a holdear hasta apoyar.
5. Puntaje estandar: 100/300/500/800 por 1/2/3/4 lineas, multiplicado por el nivel; y deteccion
   de T-spin, que puntua 400/800/1200/1600.
6. Niveles: sube uno cada 10 lineas y la gravedad se acelera.
7. Soft drop y hard drop, con sus puntos (1 y 2 por celda).
8. Deteccion de fin de juego.

El tablero es de 10x20 (con 20 filas ocultas arriba para el spawn).

IMPORTANTE: exponé una clase `Tetris` con esta interfaz exacta, para poder probarla sin la
interfaz interactiva:
  - `Tetris(seed=None)` construye el juego
  - `.tablero` -> lista de 20 listas de 10 elementos (None o la letra de la pieza)
  - `.mover(dx)` mueve la pieza, `.rotar(sentido)` con sentido 1 o -1, `.bajar()` un paso,
    `.hard_drop()`, `.hold()`
  - `.puntaje`, `.lineas`, `.nivel`, `.terminado`
  - `.paso()` avanza la gravedad un tick
Escribi el codigo completo, sin omitir nada ni dejar TODOs."""

ARNES = r'''
import random, sys
sys.path.insert(0, "/tmp")
import solucion_tetris as T

fallos, ok = [], []
def chequear(nombre, fn):
    try:
        fn(); ok.append(nombre)
    except Exception as e:
        fallos.append(f"{nombre}: {type(e).__name__}: {e}")

def api():
    j = T.Tetris(seed=1)
    # 20 filas visibles o 40 con las ocultas del spawn: las dos lecturas del enunciado valen
    assert len(j.tablero) in (20, 40), f"filas={len(j.tablero)}"
    assert all(len(f) == 10 for f in j.tablero), "columnas != 10"
    for a in ("mover","rotar","bajar","hard_drop","hold","paso"):
        assert callable(getattr(j, a, None)), f"falta {a}"
    for a in ("puntaje","lineas","nivel","terminado"):
        getattr(j, a)
chequear("interfaz completa", api)

def determinismo():
    a, b = T.Tetris(seed=7), T.Tetris(seed=7)
    for _ in range(200):
        a.paso(); b.paso()
    assert a.tablero == b.tablero, "misma semilla da tableros distintos"
chequear("misma semilla, misma partida", determinismo)

def bolsa():
    j = T.Tetris(seed=3)
    vistas = []
    for _ in range(400):
        p = getattr(j, "pieza_actual", None) or getattr(j, "pieza", None)
        n = getattr(p, "tipo", None) or getattr(p, "letra", None) or str(p)[:1]
        if not vistas or vistas[-1] != n: vistas.append(n)
        j.hard_drop()
        if j.terminado: break
    for i in range(0, len(vistas) - 7, 7):
        grupo = vistas[i:i+7]
        if len(set(grupo)) != len(grupo):
            raise AssertionError(f"la bolsa repite dentro de 7: {grupo}")
chequear("bolsa de 7 sin repetidos", bolsa)

def partida_larga():
    j = T.Tetris(seed=11)
    for i in range(3000):
        r = random.Random(i)
        acc = r.choice(["m","r","b","h","d"])
        if acc == "m": j.mover(r.choice([-1,1]))
        elif acc == "r": j.rotar(r.choice([-1,1]))
        elif acc == "b": j.bajar()
        elif acc == "h": j.hold()
        else: j.hard_drop()
        j.paso()
        if j.terminado: break
        for f in j.tablero:
            assert len(f) == 10, "el tablero se deformo"
        assert len(j.tablero) in (20, 40), "cambio la cantidad de filas"
    assert j.puntaje >= 0 and j.lineas >= 0
chequear("3000 acciones al azar sin romperse", partida_larga)

def lineas_y_puntaje():
    j = T.Tetris(seed=5)
    antes = j.lineas
    for _ in range(600):
        j.hard_drop()
        if j.terminado: break
    assert j.terminado or j.lineas >= antes, "las lineas bajaron"
chequear("termina o suma lineas", lineas_y_puntaje)

def hold_una_vez():
    j = T.Tetris(seed=2)
    j.hold(); antes = j.tablero
    j.hold()   # no deberia poder holdear de nuevo sin apoyar
chequear("hold no explota al repetirse", hold_una_vez)

print("PRUEBAS_OK=" + str(len(ok)))
print("PRUEBAS_FALLO=" + str(len(fallos)))
for f in fallos: print("  FALLO " + f)
'''


def main() -> None:
    with urllib.request.urlopen(urllib.request.Request(HOST + "/v1/models", headers=CAB)) as r:
        mdl = json.load(r)["data"][0]["id"]
    pensar = "--thinking" in sys.argv
    cuerpo = json.dumps({
        "model": mdl, "stream": True, "stream_options": {"include_usage": True},
        "max_tokens": 32000, "temperature": 0.2,
        "messages": [{"role": "user", "content": PEDIDO}],
        "chat_template_kwargs": {"enable_thinking": pensar},
    }).encode()
    req = urllib.request.Request(URL, cuerpo, CAB)
    t0 = time.perf_counter()
    ttft, n, npens, real, trozos = None, 0, 0, 0, []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for linea in r:
            if not linea.startswith(b"data: "):
                continue
            d = linea[6:].strip()
            if d == b"[DONE]":
                break
            j = json.loads(d)
            if j.get("usage"):
                # CONTAR CHUNKS SUBESTIMA. Con MTP, vLLM manda los tokens aceptados de un paso
                # juntos en un mismo chunk SSE, asi que los chunks son ~2,8x menos que los
                # tokens. El numero real es el de `usage`.
                real = j["usage"]["completion_tokens"]
            if not j["choices"]:
                continue
            delta = j["choices"][0]["delta"]
            # con thinking el texto sale por reasoning_content, no por content: si solo se mira
            # content, ttft queda en None y no se cuenta el razonamiento
            razon = delta.get("reasoning_content")
            c = delta.get("content")
            if razon or c:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n += 1
                if razon:
                    npens += 1
                if c:
                    trozos.append(c)
    total = time.perf_counter() - t0
    texto = "".join(trozos)
    ttft = ttft or total
    real = real or n
    print(f"  thinking={'si' if pensar else 'no'}  TTFT={ttft:.2f}s  total={total:.1f}s  "
          f"tokens={real} (chunks SSE {n}, razonamiento {npens})  "
          f"decode={real / max(total - ttft, 1e-9):.1f} tok/s")

    bloques = re.findall(r"```(?:python)?\n(.*?)```", texto, re.S)
    if not bloques:
        print("  NO devolvio un bloque de codigo"); return
    codigo = max(bloques, key=len)
    print(f"  bloque de codigo: {len(codigo)} caracteres, {codigo.count(chr(10))} lineas")
    open("/tmp/solucion_tetris.py", "w").write(codigo)
    open("/tmp/arnes_tetris.py", "w").write(ARNES)
    r = subprocess.run([sys.executable, "/tmp/arnes_tetris.py"],
                       capture_output=True, text=True, timeout=180)
    print("  --- pruebas sobre el codigo generado ---")
    for ln in (r.stdout + r.stderr).strip().splitlines()[-25:]:
        print("   ", ln)


if __name__ == "__main__":
    main()
