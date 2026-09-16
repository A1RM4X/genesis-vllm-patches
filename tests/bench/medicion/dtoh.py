"""Quien manda datos a la RAM del host durante el prefill."""
import collections, glob, gzip, json, os, sys
d = sys.argv[1]; pasos = float(sys.argv[2])
f = sorted(glob.glob(os.path.join(d, "*.pt.trace.json*")))[0]
ab = gzip.open if f.endswith(".gz") else open
tr = json.load(ab(f, "rt"))["traceEvents"]
ev = [e for e in tr if e.get("ph") == "X"]
mc = [e for e in ev if e.get("cat") == "gpu_memcpy" and "DtoH" in e["name"]]
print(f"{len(mc)} copias a host ({len(mc)/pasos:.0f}/paso), "
      f"{sum(e['dur'] for e in mc)/pasos/1000:.2f} ms/paso, "
      f"{sum(int(e.get('args',{}).get('bytes',0) or 0) for e in mc)/pasos/1e6:.1f} MB/paso")
tam = collections.Counter(int(e.get("args", {}).get("bytes", 0) or 0) for e in mc)
print("\ntamanos:")
for b, n in tam.most_common(8):
    print(f"  {b/1e6:9.2f} MB x {n:4d}  ({n*b/pasos/1e6:7.1f} MB/paso)")
# quien las lanza: buscar el ac2g / flow o el evento de CPU que las correlaciona
ext = {e["args"]["correlation"]: e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver")
       and e.get("args", {}).get("correlation") is not None}
cpu = sorted([e for e in ev if e.get("cat") in ("cpu_op", "user_annotation")], key=lambda e: e["ts"])
print("\nquien las pide (el cpu_op que las envuelve):")
c = collections.Counter()
for e in mc:
    r = ext.get(e.get("args", {}).get("correlation"))
    if r is None:
        continue
    ts = r["ts"]
    mejor = None
    for o in cpu:
        if o["ts"] <= ts <= o["ts"] + o["dur"]:
            if mejor is None or o["dur"] < mejor["dur"]:
                mejor = o
    c[mejor["name"] if mejor else "?"] += 1
for n, k in c.most_common(10):
    print(f"  {k:4d}  {n[:80]}")
