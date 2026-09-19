#!/usr/bin/env python3
"""Fidelidad de la KV por teacher forcing: prompt_logprobs sobre un texto largo real.

Uso: fidelidad.py <nombre>            -> guarda /tmp/g115/fid_<nombre>.json
     fidelidad.py --comparar ref otro -> top-1 y |dlogp| contra la referencia, por tramo
"""
import json, sys, glob, urllib.request, math
import os
BASE="http://127.0.0.1:8320"; KEY=os.environ.get("VLLM_API_KEY", "")
def texto(objetivo_chars=110_000):
    fs=sorted(glob.glob("/home/usuario/Proyectos/genesis-vllm-patches/vllm/_genesis/**/*.py", recursive=True))
    out=[]; n=0
    for f in fs:
        t=open(f,errors="ignore").read()
        out.append(f"# ==== {f.split('_genesis/')[-1]} ====\n{t}\n"); n+=len(t)
        if n>objetivo_chars: break
    return "".join(out)[:objetivo_chars]
def medir(nombre):
    b=json.dumps({"model":"qwen3.8","prompt":texto(),"max_tokens":1,"temperature":0,
                  "prompt_logprobs":1}).encode()
    r=urllib.request.Request(BASE+"/v1/completions",b,{"Content-Type":"application/json","Authorization":"Bearer "+KEY})
    d=json.load(urllib.request.urlopen(r,timeout=1800))
    pl=d["choices"][0]["prompt_logprobs"]
    reales=[]; top1=[]
    for pos in pl[1:]:
        # cada pos: {token_id: {logprob, rank, decoded_token}}
        items=list(pos.values())
        real=[x for x in items if x.get("rank") is not None]
        # el token real es el que vino en el prompt: vLLM lo incluye siempre
        lp_real=max(items, key=lambda x: x["rank"] if x["rank"] else 0)  # fallback
        tok_real=None
        for tid,x in pos.items():
            if x["rank"]!=1 or len(pos)==1: tok_real=(tid,x); break
        if tok_real is None: tok_real=next(iter(pos.items()))
        best=min(pos.items(), key=lambda kv: kv[1]["rank"])
        reales.append(tok_real[1]["logprob"]); top1.append(best[0])
    json.dump({"lp":reales,"top1":top1,"n":len(reales)}, open(f"/tmp/g115/fid_{nombre}.json","w"))
    print(f"{nombre}: {len(reales)} posiciones, logp medio {sum(reales)/len(reales):.4f}")
def comparar(a,b):
    A=json.load(open(f"/tmp/g115/fid_{a}.json")); B=json.load(open(f"/tmp/g115/fid_{b}.json"))
    n=min(A["n"],B["n"]); tramos=8; paso=n//tramos
    print(f"{b} contra {a} ({n} posiciones)")
    print(f"{'tramo (tokens)':>16} {'top1 igual':>11} {'|dlogp| medio':>14} {'dlogp medio':>12}")
    for t in range(tramos):
        lo,hi=t*paso,(t+1)*paso
        ig=sum(A["top1"][i]==B["top1"][i] for i in range(lo,hi))/(hi-lo)
        dl=[B["lp"][i]-A["lp"][i] for i in range(lo,hi)]
        print(f"{lo:>7}-{hi:<8} {100*ig:>10.2f}% {sum(map(abs,dl))/len(dl):>14.5f} {sum(dl)/len(dl):>+12.5f}")
    ig=sum(A["top1"][i]==B["top1"][i] for i in range(n))/n
    dl=[B["lp"][i]-A["lp"][i] for i in range(n)]
    print(f"{'TOTAL':>16} {100*ig:>10.2f}% {sum(map(abs,dl))/n:>14.5f} {sum(dl)/n:>+12.5f}")
if sys.argv[1]=="--comparar": comparar(sys.argv[2], sys.argv[3])
else: medir(sys.argv[1])
