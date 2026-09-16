import collections, glob, gzip, json, os, sys
d = sys.argv[1]
f = sorted(glob.glob(os.path.join(d, "*.pt.trace.json*")))[0]
ab = gzip.open if f.endswith(".gz") else open
ev = json.load(ab(f, "rt"))["traceEvents"]
ker = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel","gpu_memcpy","gpu_memset")]
ker.sort(key=lambda e: e["ts"])
PASOS = 60
# los kernels chicos y sus argumentos (la traza trae dims si el profiler las guardo)
tot = collections.defaultdict(float); cnt = collections.Counter(); arg = {}
for e in ker:
    n = e["name"].split("(")[0][:70]
    tot[n] += e["dur"]; cnt[n] += 1
    if n not in arg and e.get("args"): arg[n] = {k: v for k, v in e["args"].items()
                                                if k in ("grid","block","bytes","stream")}
print(f"{'us/paso':>8} {'lanz/paso':>9} {'us c/u':>7}  kernel  [grid/bytes]")
for n, t in sorted(tot.items(), key=lambda kv: -kv[1]):
    if t/PASOS < 60: continue
    print(f"{t/PASOS:8.0f} {cnt[n]/PASOS:9.1f} {t/cnt[n]:7.2f}  {n}  {arg.get(n,{})}")
