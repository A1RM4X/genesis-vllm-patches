#!/usr/bin/env python3
"""analiza_traza.py — desglose por kernel GPU de una traza de vLLM.

Agrupa los eventos de categoria 'kernel' (los que corren en la GPU) por nombre,
suma su duracion y los ordena. Ademas clasifica en familias para poder comparar
dos configuraciones: cuanto se va en GEMM, cuanto en quant de activacion,
cuanto en atencion, cuanto en elementwise.

Uso: analiza_traza.py <archivo.json[.gz]> [--top N]
"""
import gzip, json, sys, collections, re

def cargar(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)

def familia(n):
    l = n.lower()
    if "quant" in l or "rmsnorm" in l: return "quant/norm"
    if any(k in l for k in ("sk01","sk02","sk03","sk04","sk05","sk06","sk10","gdn_qkvz","mlp_down","fa_o","fa_qkv","gateup")): return "super kernel GEMM"
    if "marlin" in l: return "Marlin GEMM"
    if "cutlass" in l or "scaled_mm" in l or "gemm" in l or "s16816" in l or "cutlass" in l: return "GEMM (cutlass/cuBLAS)"
    if "attn" in l or "flash" in l or "paged" in l or "chunk" in l or "delta" in l or "conv" in l: return "atencion/GDN"
    if "elementwise" in l or "vectorized" in l or "silu" in l or "act" in l or "add" in l or "copy" in l or "cat" in l: return "elementwise/copias"
    if "reduce" in l or "norm" in l: return "reducciones"
    if "nccl" in l or "allreduce" in l or "all_reduce" in l: return "NCCL"
    return "otros"

def main():
    path = sys.argv[1]
    top = int(sys.argv[sys.argv.index("--top")+1]) if "--top" in sys.argv else 18
    ev = cargar(path).get("traceEvents", [])
    ker = [e for e in ev if e.get("cat") == "kernel" and "dur" in e]
    if not ker:
        cats = collections.Counter(e.get("cat") for e in ev)
        print("sin eventos 'kernel'. categorias presentes:", dict(cats.most_common(8))); return
    porn = collections.defaultdict(lambda: [0.0, 0])
    porf = collections.defaultdict(lambda: [0.0, 0])
    for e in ker:
        n = e["name"]; d = e["dur"]
        porn[n][0] += d; porn[n][1] += 1
        f = familia(n); porf[f][0] += d; porf[f][1] += 1
    tot = sum(v[0] for v in porn.values())
    print(f"total GPU: {tot/1000:.2f} ms  |  {len(ker)} lanzamientos  |  {len(porn)} kernels distintos\n")
    print(f"{'familia':<24}{'ms':>10}{'%':>7}{'lanzam.':>10}")
    for f,(d,c) in sorted(porf.items(), key=lambda x:-x[1][0]):
        print(f"{f:<24}{d/1000:10.2f}{100*d/tot:7.1f}{c:10}")
    print(f"\n{'kernel':<66}{'ms':>9}{'%':>7}{'n':>7}{'us/l':>8}")
    for n,(d,c) in sorted(porn.items(), key=lambda x:-x[1][0])[:top]:
        print(f"{n[:66]:<66}{d/1000:9.2f}{100*d/tot:7.1f}{c:7}{d/c:8.1f}")

main()
