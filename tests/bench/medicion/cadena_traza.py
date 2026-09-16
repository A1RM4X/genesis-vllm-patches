#!/usr/bin/env python3
"""cadena_traza.py — la secuencia real de kernels de un paso, para ver que intermedio va y vuelve
por memoria global entre dos kernels vecinos.

Uso: cadena_traza.py <dir_traza> [n_pasos]
Imprime el patron que se repite por capa con el tiempo de cada eslabon.
"""
import collections, glob, gzip, json, os, sys

d = sys.argv[1]
arch = sorted(glob.glob(os.path.join(d, "*.pt.trace.json*")))
arch = [a for a in arch if "rank0" in a or len(arch) == 1] or arch
f = arch[0]
ab = gzip.open if f.endswith(".gz") else open
ev = json.load(ab(f, "rt"))["traceEvents"]
ker = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
ker.sort(key=lambda e: e["ts"])
print(f"{len(ker)} kernels en la traza")

def corto(n):
    n = n.split("(")[0]
    for p in ("void ", "at::native::", "vllm::", "cutlass::", "cute::"):
        n = n.replace(p, "")
    return n[:58]

# tiempo total y lanzamientos por kernel
tot = collections.defaultdict(float); cnt = collections.Counter()
for e in ker:
    tot[corto(e["name"])] += e["dur"]; cnt[corto(e["name"])] += 1
print(f"\n{'us tot':>9} {'n':>6} {'us c/u':>7}  kernel")
for n, t in sorted(tot.items(), key=lambda kv: -kv[1])[:28]:
    print(f"{t:9.0f} {cnt[n]:6d} {t/cnt[n]:7.2f}  {n}")

# la cadena: bigramas mas frecuentes (productor -> consumidor)
big = collections.Counter()
for a, b in zip(ker, ker[1:]):
    big[(corto(a["name"]), corto(b["name"]))] += 1
print(f"\n{'n':>6}  par consecutivo mas frecuente")
for (a, b), n in big.most_common(22):
    print(f"{n:6d}  {a}\n        -> {b}")
