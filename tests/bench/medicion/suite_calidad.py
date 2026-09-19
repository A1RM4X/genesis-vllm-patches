#!/usr/bin/env python3
"""Tareas con UNA respuesta correcta, temperature=0. Cuenta aciertos."""
import json, sys, urllib.request
import os
BASE="http://127.0.0.1:8320/v1/chat/completions"; KEY=os.environ.get("VLLM_API_KEY", "")
TAREAS=[
 ("¿Cuánto es 847 por 23? Respondé solo el número.", "19481"),
 ("¿Cuánto es 1234 más 5678? Respondé solo el número.", "6912"),
 ("¿Cuánto es 96 dividido 8? Respondé solo el número.", "12"),
 ("¿Cuánto es 15 al cuadrado? Respondé solo el número.", "225"),
 ("¿Cuántos días tiene un año bisiesto? Respondé solo el número.", "366"),
 ("¿Cuál es la capital de Australia? Respondé solo el nombre.", "Canberra"),
 ("¿En qué año llegó el hombre a la Luna? Respondé solo el número.", "1969"),
 ("¿Cuánto es 7 factorial? Respondé solo el número.", "5040"),
 ("Contá las letras de la palabra 'murcielago'. Respondé solo el número.", "10"),
 ("¿Cuánto es 3 elevado a 5? Respondé solo el número.", "243"),
]
ok=0
for p,esp in TAREAS:
    body=json.dumps({"model":"qwen3.8","messages":[{"role":"user","content":p}],
                     "max_tokens":120,"temperature":0,
                     "chat_template_kwargs":{"enable_thinking":False}}).encode()
    req=urllib.request.Request(BASE,body,{"Content-Type":"application/json",
                                          "Authorization":"Bearer "+KEY})
    m=json.load(urllib.request.urlopen(req,timeout=300))["choices"][0]["message"]
    t=((m.get("content") or "")+" "+(m.get("reasoning_content") or "")).strip()
    bien=esp.lower() in t.lower().replace(".","").replace(",","")
    ok+=bien
    print(f"  {'OK ' if bien else 'MAL'} esperado={esp:<9} -> {t[:70].replace(chr(10),' ')}")
print(f"\n  aciertos: {ok}/{len(TAREAS)}")
