import time, urllib.request
B="http://127.0.0.1:8320/metrics"; K="<REDACTADO: clave rotada 2026-09-19>"
KS=["prefix_cache_queries_total","prefix_cache_hits_total","external_prefix_cache_queries_total","external_prefix_cache_hits_total","num_preemptions_total","prompt_tokens_total","generation_tokens_total"]
def leer():
    d={}
    for l in urllib.request.urlopen(urllib.request.Request(B,headers={"Authorization":"Bearer "+K}),timeout=20).read().decode().split("\n"):
        if l.startswith("#") or "created" in l: continue
        for k in KS+["kv_cache_usage_perc","num_requests_running","num_requests_waiting{"]:
            if l.startswith("vllm:"+k): d[k.rstrip("{")]=float(l.rsplit(" ",1)[1])
    return d
a=leer(); print("  t   L1q    L1hit   extq  exthit  pre  prompt  gen  KV%  run wait",flush=True)
for i in range(18):
    time.sleep(10); b=leer(); dd=lambda k:b.get(k,0)-a.get(k,0)
    print(f"{(i+1)*10:>4} {dd(KS[0]):>6.0f} {dd(KS[1]):>7.0f} {dd(KS[2]):>6.0f} {dd(KS[3]):>6.0f} {dd(KS[4]):>4.0f} {dd(KS[5]):>7.0f} {dd(KS[6]):>4.0f} {100*b['kv_cache_usage_perc']:>4.0f} {b['num_requests_running']:>3.0f} {b['num_requests_waiting']:>3.0f}",flush=True)
    a=b
