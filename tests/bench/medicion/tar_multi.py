#!/usr/bin/env python3
"""TAR agregado sobre varios prompts (deltas de contadores) + textos para comparar."""
import json, sys, time, urllib.request
import os
BASE="http://127.0.0.1:8320"; KEY=os.environ.get("VLLM_API_KEY", "")
def met():
    r=urllib.request.Request(BASE+"/metrics",headers={"Authorization":"Bearer "+KEY}); d={}
    for l in urllib.request.urlopen(r,timeout=30).read().decode().split("\n"):
        for k in ("vllm:spec_decode_num_accepted_tokens_total","vllm:spec_decode_num_draft_tokens_total","vllm:prompt_tokens_total","vllm:generation_tokens_total"):
            if l.startswith(k) and "created" not in l: d[k]=float(l.rsplit(" ",1)[1])
    return d
P=["Escribí una clase de Python para una cola circular, con docstrings.",
   "Implementá en TypeScript un debounce genérico con tests en Jest.",
   "Escribí una función en Rust que parsee un CSV simple a un Vec de structs.",
   "Escribí un script bash que rote logs de /var/log/app manteniendo 7 días.",
   "Escribí un endpoint FastAPI con SQLAlchemy para CRUD de usuarios.",
   "Explicá cómo funciona el paginado de memoria virtual en Linux.",
   "Escribí un cuento de 300 palabras sobre un faro en la Patagonia.",
   "Resumí las diferencias entre TCP y QUIC en una tabla markdown."]
tot_a=tot_d=tot_g=tot_t=0; textos=[]
for p in P:
    b=json.dumps({"model":"qwen3.8","messages":[{"role":"user","content":p}],"max_tokens":400,"temperature":0,
                  "chat_template_kwargs":{"enable_thinking":False}}).encode()
    a=met(); t0=time.perf_counter()
    j=json.load(urllib.request.urlopen(urllib.request.Request(BASE+"/v1/chat/completions",b,{"Content-Type":"application/json","Authorization":"Bearer "+KEY}),timeout=900))
    dt=time.perf_counter()-t0; c=met()
    acc=c["vllm:spec_decode_num_accepted_tokens_total"]-a["vllm:spec_decode_num_accepted_tokens_total"]
    dr=c["vllm:spec_decode_num_draft_tokens_total"]-a["vllm:spec_decode_num_draft_tokens_total"]
    g=j["usage"]["completion_tokens"]; tot_a+=acc; tot_d+=dr; tot_g+=g; tot_t+=dt
    ajenos=(c["vllm:generation_tokens_total"]-a["vllm:generation_tokens_total"])-g
    if abs(ajenos)>0: print(f"  !! CONTAMINADO: {ajenos:.0f} tokens generados por otros requests")
    textos.append(j["choices"][0]["message"]["content"])
    print(f"  TAR={100*acc/dr:5.1f}%  {g/dt:5.0f} tok/s  {p[:45]}")
print(f"TOTAL {sys.argv[1]}: TAR={100*tot_a/tot_d:.1f}%  {tot_g/tot_t:.0f} tok/s")
json.dump(textos, open(f"/tmp/g115/textos_{sys.argv[1]}.json","w"))
