import json,urllib.request,time,threading,collections
import os
H={"Content-Type":"application/json","Authorization":"Bearer " + os.environ.get("VLLM_API_KEY", "")}
def req(n,seed,mt):
    txt=(f"[{seed}] Documento tecnico sobre planificacion de memoria en GPUs. "*n)+"\nEscribi un ensayo largo sobre el tema."
    b=json.dumps({"model":"qwen3.8","messages":[{"role":"user","content":txt}],"max_tokens":mt,"temperature":0.7,"chat_template_kwargs":{"enable_thinking":False}}).encode()
    urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8320/v1/chat/completions",b,H),timeout=900).read()
req(600,"warm",20)
ts=[threading.Thread(target=req,args=(1800,s,900)) for s in ("a","b","c")]
[t.start() for t in ts]
vistos=collections.Counter(); ej=None
while any(t.is_alive() for t in ts):
    time.sleep(1)
    try: d=json.load(open('/dev/shm/genesis_pid_status.json'))
    except Exception: continue
    bp=d.get("bloques_por_request")
    if not bp or "requests" not in bp: 
        if bp: print(bp); 
        continue
    for r in bp["requests"]:
        vistos[(r["tokens"]//832, tuple(r["bloques"]))]+=1
    ej=bp
print("grupos:",ej and ej["grupos"])
for (blk,b),c in sorted(vistos.items()):
    print(f"tokens/832={blk:>3}  bloques por grupo={list(b)}  (visto {c}x)")
