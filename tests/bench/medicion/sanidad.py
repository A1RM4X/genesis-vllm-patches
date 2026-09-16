import json,urllib.request,time
H={"Content-Type":"application/json","Authorization":"Bearer <REDACTADO: clave rotada 2026-09-19>"}
def ask(txt,mt):
    b=json.dumps({"model":"qwen3.8","messages":[{"role":"user","content":txt}],"max_tokens":mt,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}).encode()
    t=time.time(); d=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8320/v1/chat/completions",b,H),timeout=600))
    return time.time()-t, d
t,d=ask(("El sistema de archivos de Linux organiza los datos en inodos y bloques. "*400)+"\nResumí en una frase.",60)
print(round(t,1),"s |",d["choices"][0]["message"]["content"][:200])
t,d=ask("Escribí una función en Python que devuelva los primeros n números primos, con docstring y un ejemplo de uso.",400)
u=d["usage"]; print(round(t,1),"s |",u["completion_tokens"],"tok |",round(u["completion_tokens"]/t,1),"tok/s")
print(d["choices"][0]["message"]["content"][:700])
