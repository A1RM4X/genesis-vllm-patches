#!/usr/bin/env python3
"""Contexto largo: (1) agujas — 12 datos repartidos en ~60k tokens de codigo real,
se piden todos al final; (2) velocidad de decode con ese contexto (streaming,
sin contar el TTFT)."""
import json, sys, time, glob, random, urllib.request
import os
BASE="http://127.0.0.1:8320/v1/chat/completions"; KEY=os.environ.get("VLLM_API_KEY", "")
random.seed(7)
fs=sorted(glob.glob("/home/usuario/Proyectos/genesis-vllm-patches/vllm/_genesis/**/*.py", recursive=True))
txt="".join(open(f,errors="ignore").read() for f in fs)[:200_000]
nombres=["ALFA","BRAVO","CHARLIE","DELTA","ECO","FOXTROT","GOLF","HOTEL","INDIA","JULIETT","KILO","LIMA"]
valores={n: str(random.randint(10000,99999)) for n in nombres}
partes=[txt[i*len(txt)//12:(i+1)*len(txt)//12] for i in range(12)]
doc="".join(f"{p}\n# CLAVE_{n} = {valores[n]}\n" for p,n in zip(partes,nombres))
pregunta="\n\nListá el valor de cada CLAVE_ que aparece en el código anterior, una por línea con formato NOMBRE=valor, sin nada más."
body=json.dumps({"model":"qwen3.8","messages":[{"role":"user","content":f"[{sys.argv[1]}-{time.time()}]\n"+doc+pregunta}],
                 "max_tokens":400,"temperature":0,"stream":True,"stream_options":{"include_usage":True},
                 "chat_template_kwargs":{"enable_thinking":False}}).encode()
req=urllib.request.Request(BASE,body,{"Content-Type":"application/json","Authorization":"Bearer "+KEY})
t0=time.perf_counter(); primero=None; texto=""; uso=None
for linea in urllib.request.urlopen(req,timeout=1800):
    l=linea.decode().strip()
    if not l.startswith("data:") or l=="data: [DONE]": continue
    d=json.loads(l[5:])
    if d.get("usage"): uso=d["usage"]
    for c in d.get("choices",[]):
        t=c.get("delta",{}).get("content")
        if t:
            if primero is None: primero=time.perf_counter()
            texto+=t
fin=time.perf_counter()
ok=sum(1 for n in nombres if f"{n}={valores[n]}" in texto.replace(" ",""))
gen=uso["completion_tokens"]
print(f"{sys.argv[1]}: prompt={uso['prompt_tokens']} agujas={ok}/12 TTFT={primero-t0:.1f}s decode={(gen-1)/(fin-primero):.1f} tok/s ({gen} tok)")
