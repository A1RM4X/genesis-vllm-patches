import json,urllib.request,re,time
H={"Content-Type":"application/json","Authorization":"Bearer <REDACTADO: clave rotada 2026-09-19>"}
def chat(msgs,mt):
    b=json.dumps({"model":"qwen3.8","messages":msgs,"max_tokens":mt,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}).encode()
    t=time.time(); d=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8320/v1/chat/completions",b,H),timeout=900))
    return d["choices"][0]["message"]["content"], d["usage"], time.time()-t
def chequear(txt, desde):
    nums=[int(x) for x in re.findall(r"\d+", txt)]
    esperado=list(range(desde, desde+len(nums)))
    malos=[(i,a,b) for i,(a,b) in enumerate(zip(nums,esperado)) if a!=b]
    return len(nums), malos[:3]
contexto=("Notas del proyecto: " + "el scheduler reparte bloques de memoria entre las requests activas. "*260)
m=[{"role":"user","content":contexto+"\n\nEscribí los números del 1 al 700 separados por comas, sin nada más."}]
r,u,t=chat(m,3000); n,mal=chequear(r,1)
print(f"turno 1: prompt={u['prompt_tokens']} gen={u['completion_tokens']} ({u['completion_tokens']/t:.0f} tok/s) numeros={n} errores={mal}")
m+= [{"role":"assistant","content":r},{"role":"user","content":"Seguí desde el 701 hasta el 1000, mismo formato."}]
r2,u2,t2=chat(m,2000); n2,mal2=chequear(r2,701)
print(f"turno 2: prompt={u2['prompt_tokens']} cacheados={u2.get('prompt_tokens_details',{}).get('cached_tokens')} gen={u2['completion_tokens']} numeros={n2} errores={mal2}")
