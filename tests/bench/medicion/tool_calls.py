#!/usr/bin/env python3
"""Salud de los tool calls: manda N pedidos con herramientas de edicion de archivos (como los de
open-webui / opencode) y cuenta cuantos vuelven con JSON valido en los argumentos.

El sintoma que estamos cazando es un cierre de etiqueta mal emitido en el formato de Qwen
(`</>` en vez de `</parameter>`), que el parser convierte en JSON roto.

Uso: tool_calls.py [etiqueta] [N] [temperatura] [tokens_de_contexto]
"""
import json, sys, time, urllib.request, glob, random
import os

BASE = "http://127.0.0.1:8320/v1/chat/completions"
KEY = os.environ.get("VLLM_API_KEY", "")
TAG = sys.argv[1] if len(sys.argv) > 1 else "prueba"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
TEMP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
CTX = int(sys.argv[4]) if len(sys.argv) > 4 else 0        # tokens de relleno de contexto

HERRAMIENTAS = [
    {"type": "function", "function": {
        "name": "replace_file_content",
        "description": "Reemplaza un fragmento exacto de un archivo por otro.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Ruta absoluta del archivo"},
            "old_str": {"type": "string", "description": "Texto exacto a reemplazar"},
            "new_str": {"type": "string", "description": "Texto nuevo"}},
            "required": ["path", "old_str", "new_str"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Escribe un archivo completo.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
]

PEDIDOS = [
    "En /home/user/pagoda/index.html, cambiá el título de la página por 'Pagoda — inicio' y agregá "
    "un <meta name=\"viewport\"> si no está. Usá las herramientas.",
    "Creá /home/user/pagoda/estilos.css con una hoja de estilos mínima: variables de color, tipografía "
    "y una grilla de tres columnas que se apile en mobile.",
    "En /home/user/pagoda/app.js reemplazá la función saludar() por una versión que reciba un nombre "
    "y devuelva un template string.",
    "Arreglá el <footer> de /home/user/pagoda/index.html: tiene una etiqueta sin cerrar y el año está "
    "hardcodeado. Dejalo con el año dinámico.",
]

relleno = ""
if CTX:
    fs = sorted(glob.glob("/home/usuario/Proyectos/genesis-vllm-patches/vllm/_genesis/**/*.py", recursive=True))
    txt = "".join(open(f, errors="ignore").read() for f in fs)
    relleno = txt[: CTX * 4]      # ~4 caracteres por token


def pedir(i):
    msgs = []
    if relleno:
        msgs.append({"role": "user", "content": "Contexto del proyecto:\n\n" + relleno})
        msgs.append({"role": "assistant", "content": "Listo, leí el proyecto."})
    msgs.append({"role": "user", "content": PEDIDOS[i % len(PEDIDOS)]})
    cuerpo = {"model": "qwen3.8", "messages": msgs, "tools": HERRAMIENTAS, "tool_choice": "auto",
              "temperature": TEMP, "max_tokens": 1200, "seed": 1000 + i}
    req = urllib.request.Request(BASE, json.dumps(cuerpo).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


ok = mal = sin = err = 0
razones = {}
fallas = []
t0 = time.time()
for i in range(N):
    try:
        d = pedir(i)
    except Exception as e:
        err += 1; fallas.append(("http", str(e)[:120])); continue
    fr = d["choices"][0].get("finish_reason")
    razones[fr] = razones.get(fr, 0) + 1
    m = d["choices"][0]["message"]
    tc = m.get("tool_calls") or []
    if not tc:
        sin += 1
        cont = (m.get("content") or "")
        if "<tool_call" in cont or "<function=" in cont:      # el parser no lo pudo convertir
            mal += 1; sin -= 1; fallas.append(("sin_parsear", cont[-200:]))
        continue
    for c in tc:
        try:
            a = json.loads(c["function"]["arguments"])
            if not isinstance(a, dict) or not a:
                raise ValueError("vacio")
            ok += 1
        except Exception as e:
            mal += 1
            fallas.append((c["function"]["name"],
                           f"args={c['function']['arguments']!r} fin={d['choices'][0].get('finish_reason')} "
                           f"content={(m.get('content') or '')[-220:]!r}"))
print(f"{TAG}: N={N} temp={TEMP} ctx={CTX} | tool calls OK={ok} MALFORMADOS={mal} sin_tool={sin} "
      f"errores_http={err} | razones={razones} | {time.time()-t0:.0f}s")
for n, s in fallas[:6]:
    print(f"   [{n}] {s}")
