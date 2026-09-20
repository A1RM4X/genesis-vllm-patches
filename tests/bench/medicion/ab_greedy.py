import json,os,sys,time,urllib.request,re,statistics as st
URL="http://localhost:8360"; KEY=os.environ["VLLM_API_KEY"]; H={"Authorization":"Bearer "+KEY,"Content-Type":"application/json"}
PROSA="""crea una nueva, sin usar ningun codigo viejo pagoda japonesa en three.js , voxel , muy detallado, en una isla flotando en el espacio, con su propia atmosfera, y parque japones perfecto, cascada, montaña y arcoiris añadirle niebla volumétrica real (fog shader), sombras dinámicas, sonido ambiental o más estructuras (puente, torii en el estanque, más pisos)
revisa la consola del navegador al levantarlo y controla no haya errores."""
COD="Write a complete, production-quality Python implementation of a red-black tree (insert, delete, search, in-order iteration, invariant checker) followed by a thorough pytest test suite. Output only code."
def met():
    t=urllib.request.urlopen(urllib.request.Request(URL+"/metrics",headers=H)).read().decode()
    g=lambda n: sum(float(x) for x in re.findall(r"^vllm:%s(?:\{[^}]*\})? ([0-9.e+]+)$"%n,t,re.M))
    return g("spec_decode_num_drafts_total"),g("spec_decode_num_accepted_tokens_total"),g("generation_tokens_total")
M=json.load(urllib.request.urlopen(urllib.request.Request(URL+"/v1/models",headers=H)))["data"][0]["id"]
def run(p,think,mt):
    b={"model":M,"messages":[{"role":"user","content":p}],"max_tokens":mt,"temperature":0,"chat_template_kwargs":{"enable_thinking":think}}
    t0=time.time(); d=json.load(urllib.request.urlopen(urllib.request.Request(URL+"/v1/chat/completions",data=json.dumps(b).encode(),headers=H),timeout=900)); dt=time.time()-t0
    return d["usage"]["completion_tokens"],dt
N=int(sys.argv[1]) if len(sys.argv)>1 else 3; MT=1500
for nom,p,th in (("prosa",PROSA,True),("codigo",COD,False)):
    run(p,th,300)
    v=[];d0,a0,g0=met();tot=0
    for _ in range(N):
        n,dt=run(p,th,MT); v.append(n/dt); tot+=n
    d1,a1,g1=met()
    aj=(g1-g0)-tot
    print(f"{nom:7s} tok/s {st.mean(v):6.1f} ±{(st.pstdev(v)):4.1f}  acept.media {1+(a1-a0)/max(d1-d0,1):.2f}  ajeno={aj:.0f}")
