"""Trabajo real de cada rank (sin contar los colectivos): ¿estan balanceados?"""
import collections, glob, gzip, json, os, sys
d = sys.argv[1]; pasos = float(sys.argv[2])
COL = ("nccl", "cross_device", "allreduce", "all_reduce", "allgather")
res = {}
for f in sorted(glob.glob(os.path.join(d, "*.pt.trace.json*"))):
    ab = gzip.open if f.endswith(".gz") else open
    ev = json.load(ab(f, "rt"))["traceEvents"]
    ker = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy")]
    r = "rank1" if "rank1" in os.path.basename(f) else "rank0"
    comp = [e for e in ker if not any(k in e["name"].lower() for k in COL)]
    g = collections.defaultdict(float)
    for e in comp:
        g[e["name"].split("(")[0][:44]] += e["dur"]
    res[r] = (sum(e["dur"] for e in comp), g, len(comp))
for r, (t, g, n) in res.items():
    print(f"{r}: computo {t/pasos/1000:8.1f} ms/paso en {n/pasos:.0f} kernels")
if len(res) == 2:
    t0, t1 = res["rank0"][0], res["rank1"][0]
    print(f"\ndesbalance de computo: rank1 - rank0 = {(t1-t0)/pasos/1000:+.1f} ms/paso "
          f"({100*(t1-t0)/t0:+.1f}%)")
    print(f"\nlos kernels donde mas difieren (ms/paso):")
    print(f"{'rank0':>9} {'rank1':>9} {'dif':>9}  kernel")
    g0, g1 = res["rank0"][1], res["rank1"][1]
    difs = [(abs(g1.get(k, 0) - g0.get(k, 0)), k) for k in set(g0) | set(g1)]
    for dv, k in sorted(difs, reverse=True)[:12]:
        print(f"{g0.get(k,0)/pasos/1000:9.1f} {g1.get(k,0)/pasos/1000:9.1f} "
              f"{(g1.get(k,0)-g0.get(k,0))/pasos/1000:+9.1f}  {k}")
