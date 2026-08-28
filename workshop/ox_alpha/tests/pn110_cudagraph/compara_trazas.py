#!/usr/bin/env python3
"""compara_trazas.py — compara el desglose por kernel GPU de dos trazas de vLLM.

Pensado para responder una pregunta concreta: cuando PN110 cambia el camino de
ejecucion de cada capa Linear, ¿donde se va el tiempo que no se va en el GEMM?

Normaliza por numero de pasos de decode perfilados (los dos lados no ven
necesariamente la misma cantidad de iteraciones), asi las columnas son
comparables aunque las trazas tengan distinta longitud.

Uso: compara_trazas.py <base.json[.gz]> <nueva.json[.gz]> [--top N]
"""
import gzip, json, sys, collections

def cargar(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)

def familia(n):
    l = n.lower()
    # Orden importa: lo mas especifico primero.
    if "marlin" in l or "machete" in l:                       return "GEMM Marlin"
    if any(k in l for k in ("sk01","sk02","sk03","sk04","sk05","sk06","sk07","sk10",
                            "gdn_qkvz","mlp_down","fa_o","fa_qkv","gateup","lm_head")):
        return "GEMM super kernel"
    if "cutlass" in l or "scaled_mm" in l or "s16816" in l or "gemm" in l or "cublas" in l:
        return "GEMM cutlass/cuBLAS"
    # Kernels generados por inductor: aca viven las fusiones norm_quant/act_quant.
    if l.startswith("triton_poi_fused") or l.startswith("triton_red_fused") or l.startswith("triton_per_fused"):
        return "inductor fusionado"
    if "quant" in l:                                          return "quant suelto"
    if "rmsnorm" in l or "layernorm" in l or "norm" in l:      return "norm suelta"
    if any(k in l for k in ("attn","flash","paged","bmm","chunk","delta","conv","causal","mamba","ssm")):
        return "atencion/GDN"
    if "nccl" in l or "allreduce" in l or "all_reduce" in l:   return "NCCL"
    if any(k in l for k in ("elementwise","vectorized","silu","activation","add","copy","cat","index","gather")):
        return "elementwise/copias"
    if "reduce" in l or "sort" in l or "topk" in l or "sampl" in l or "argmax" in l: return "sampling/reduc."
    return "otros"

def pasos(ev):
    """Numero de iteraciones del engine en la traza, para normalizar."""
    n = sum(1 for e in ev if e.get("cat") in ("user_annotation","cpu_op")
            and isinstance(e.get("name"), str)
            and ("Preprocess" in e["name"] or "EngineCore" in e["name"] or "execute_model" in e["name"]))
    return max(n, 1)

def resumen(path):
    ev = cargar(path).get("traceEvents", [])
    ker = [e for e in ev if e.get("cat") == "kernel" and "dur" in e]
    porn = collections.defaultdict(lambda: [0.0, 0])
    porf = collections.defaultdict(lambda: [0.0, 0])
    for e in ker:
        n, d = e["name"], e["dur"]
        porn[n][0] += d; porn[n][1] += 1
        f = familia(n); porf[f][0] += d; porf[f][1] += 1
    return porn, porf, sum(v[0] for v in porn.values()), len(ker), pasos(ev)

def main():
    a, b = sys.argv[1], sys.argv[2]
    top = int(sys.argv[sys.argv.index("--top")+1]) if "--top" in sys.argv else 20
    na, fa, ta, ka, pa = resumen(a)
    nb, fb, tb, kb, pb = resumen(b)
    print(f"BASE  {a}\n   {ta/1000:9.2f} ms GPU | {ka:6} lanzamientos | ~{pa} pasos | {ta/pa:8.1f} us/paso")
    print(f"NUEVA {b}\n   {tb/1000:9.2f} ms GPU | {kb:6} lanzamientos | ~{pb} pasos | {tb/pb:8.1f} us/paso\n")

    print(f"{'familia':<22}{'BASE us/paso':>14}{'NUEVA us/paso':>15}{'delta':>11}{'lanz.B':>9}{'lanz.N':>9}")
    print("-"*80)
    fams = sorted(set(fa) | set(fb), key=lambda f: -(fb.get(f,[0,0])[0]/pb - fa.get(f,[0,0])[0]/pa))
    for f in fams:
        da, ca = fa.get(f, [0.0, 0]); db, cb = fb.get(f, [0.0, 0])
        ua, ub = da/pa, db/pb
        print(f"{f:<22}{ua:14.1f}{ub:15.1f}{ub-ua:+11.1f}{ca/pa:9.1f}{cb/pb:9.1f}")
    print("-"*80)
    print(f"{'TOTAL':<22}{ta/pa:14.1f}{tb/pb:15.1f}{tb/pb-ta/pa:+11.1f}\n")

    print(f"Kernels que mas CRECEN de BASE a NUEVA (us/paso):")
    print(f"{'kernel':<62}{'BASE':>9}{'NUEVA':>9}{'delta':>9}")
    d = {}
    for n in set(na) | set(nb):
        d[n] = nb.get(n,[0,0])[0]/pb - na.get(n,[0,0])[0]/pa
    for n in sorted(d, key=lambda x: -d[x])[:top]:
        print(f"{n[:62]:<62}{na.get(n,[0,0])[0]/pa:9.1f}{nb.get(n,[0,0])[0]/pb:9.1f}{d[n]:+9.1f}")
    print(f"\nKernels que mas DESAPARECEN (us/paso):")
    for n in sorted(d, key=lambda x: d[x])[:top//2]:
        print(f"{n[:62]:<62}{na.get(n,[0,0])[0]/pa:9.1f}{nb.get(n,[0,0])[0]/pb:9.1f}{d[n]:+9.1f}")

main()
