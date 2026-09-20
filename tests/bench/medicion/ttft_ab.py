import json,os,sys,time,random,string,threading,urllib.request
URL="http://localhost:8360"; KEY=os.environ["VLLM_API_KEY"]; H={"Authorization":"Bearer "+KEY,"Content-Type":"application/json"}
M=json.load(urllib.request.urlopen(urllib.request.Request(URL+"/v1/models",headers=H)))["data"][0]["id"]
PAL="the quick brown fox jumps over a lazy dog while engineers measure latency on two graphics cards".split()
def texto(ntok,semilla):
    r=random.Random(semilla); n=int(ntok/1.35)
    return "id "+"".join(r.choices(string.ascii_lowercase,k=12))+". "+" ".join(r.choice(PAL) for _ in range(n))
def pedir(prompt,mt=24,think=False,prio=None):
    b={"model":M,"messages":[{"role":"user","content":prompt+"\n\nReply with one word."}],"max_tokens":mt,"temperature":0,"stream":True,
       "stream_options":{"include_usage":True},"chat_template_kwargs":{"enable_thinking":think}}
    if prio is not None: b["priority"]=prio
    t0=time.time(); ttft=None; pt=cached=0
    with urllib.request.urlopen(urllib.request.Request(URL+"/v1/chat/completions",data=json.dumps(b).encode(),headers=H),timeout=900) as r:
        for l in r:
            l=l.decode().strip()
            if not l.startswith("data: ") or l=="data: [DONE]": continue
            d=json.loads(l[6:])
            if d.get("usage"):
                pt=d["usage"]["prompt_tokens"]; cached=(d["usage"].get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            for c in d.get("choices",[]):
                dl=c.get("delta",{})
                if ttft is None and (dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning")): ttft=time.time()-t0
    return ttft,pt,cached,time.time()-t0
modo=sys.argv[1]
if modo=="barrido":
    print(f"{'tokens':>8} {'frio ms':>9} {'tok/s pp':>9} {'repetido ms':>12} {'cacheados':>10}")
    for n in (100,2000,10000,30000,60000):
        p=texto(n,n)
        a=pedir(p); b=pedir(p)
        print(f"{a[1]:8d} {a[0]*1e3:9.0f} {a[1]/a[0]:9.0f} {b[0]*1e3:12.0f} {b[2]:10d}",flush=True)
elif modo=="cola":
    # un prefill largo en curso + un pedido corto que llega 1.5 s despues
    largo=texto(int(sys.argv[2]),999+int(time.time())); res={}
    th=threading.Thread(target=lambda: res.__setitem__("L",pedir(largo))); th.start(); time.sleep(1.5)
    v=[]
    for i in range(4):
        v.append(pedir(texto(300,i+int(time.time())))[0]*1e3); time.sleep(0.3)
    th.join()
    print(f"largo: {res['L'][1]} tok, TTFT {res['L'][0]*1e3:.0f} ms | cortos (300 tok) llegando durante el prefill: "+", ".join(f"{x:.0f}" for x in v)+" ms")

elif modo=="brazo":
    s0=int(time.time())
    print("grano del prefix-cache (frio -> repetido):")
    for n in (10000,30000):
        p=texto(n,s0+n); a=pedir(p); b=pedir(p)
        print(f"  {a[1]:6d} tok: frio {a[0]:6.2f} s ({a[1]/a[0]:5.0f} tok/s) -> repetido {b[0]:5.2f} s, cacheados {b[2]} ({100*b[2]/a[1]:.0f}%)",flush=True)
    a=pedir(texto(122000,s0+7))
    print(f"prefill sin contencion: {a[1]} tok, TTFT {a[0]:.2f} s = {a[1]/a[0]:.0f} tok/s",flush=True)
    for rep in range(2):
        largo=texto(80000,s0+100+rep); res={}
        th=threading.Thread(target=lambda: res.__setitem__("L",pedir(largo))); th.start(); time.sleep(1.5)
        v=[]
        while th.is_alive() and len(v)<8:
            v.append(pedir(texto(300,s0+200+10*rep+len(v)))[0]); time.sleep(0.5)
        th.join()
        print(f"cola {rep+1}: largo {res['L'][1]} tok TTFT {res['L'][0]:.1f} s | cortos durante el prefill: "+", ".join(f"{x:.2f}" for x in v)+" s",flush=True)

elif modo=="colaprio":
    s0=int(time.time()); PR=int(sys.argv[2])
    for rep in range(2):
        largo=texto(80000,s0+100+rep); res={}
        th=threading.Thread(target=lambda: res.__setitem__("L",pedir(largo))); th.start(); time.sleep(1.5)
        v=[]
        while th.is_alive() and len(v)<8:
            v.append(pedir(texto(300,s0+200+10*rep+len(v)),prio=PR)[0]); time.sleep(0.5)
        th.join()
        print(f"cola {rep+1} (cortos con priority={PR}): largo {res['L'][1]} tok TTFT {res['L'][0]:.1f} s | cortos: "+", ".join(f"{x:.2f}" for x in v)+" s",flush=True)

elif modo=="largos":
    # 3 prefills largos a la vez + cortos SIN prioridad llegando en el medio
    s0=int(time.time()); res={}; t0=time.time()
    def L(i):
        r=pedir(texto(40000,s0+i)); res[i]=(r[0],r[1])
    ths=[threading.Thread(target=L,args=(i,)) for i in range(3)]
    for th in ths: th.start(); time.sleep(0.2)
    time.sleep(1.5); v=[]
    while any(th.is_alive() for th in ths) and len(v)<10:
        v.append(pedir(texto(300,s0+50+len(v)))[0]); time.sleep(2.0)
    for th in ths: th.join()
    print("3 largos a la vez (%d tok c/u), TTFT: "%res[0][1]+", ".join(f"{res[i][0]:.1f}" for i in range(3))+" s | todo listo a los %.1f s"%(time.time()-t0))
    print("cortos sin prioridad en el medio: "+", ".join(f"{x:.2f}" for x in v)+" s",flush=True)
