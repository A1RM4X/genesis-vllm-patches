import json,urllib.request,sys
import os
H={"Content-Type":"application/json","Authorization":"Bearer " + os.environ.get("VLLM_API_KEY", "")}
T=[("¿Cuánto es 847 por 23? Respondé solo el número.","19481"),("¿Cuánto es 1234 más 5678? Respondé solo el número.","6912"),
   ("¿Cuánto es 15 al cuadrado? Respondé solo el número.","225"),("¿En qué año llegó el hombre a la Luna? Respondé solo el número.","1969"),
   ("¿Cuánto es 7 factorial? Respondé solo el número.","5040"),("¿Cuántos días tiene un año bisiesto? Respondé solo el número.","366")]
ok=0; mal=[]
for p,e in T:
    b=json.dumps({"model":"qwen3.8","messages":[{"role":"user","content":p}],"max_tokens":60,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}).encode()
    t=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8320/v1/chat/completions",b,H),timeout=300))["choices"][0]["message"]["content"]
    if e in t.replace(".","").replace(",",""): ok+=1
    else: mal.append(t[:25].replace("\n"," "))
print(f"{sys.argv[1] if len(sys.argv)>1 else ''}: {ok}/{len(T)} {mal}")
