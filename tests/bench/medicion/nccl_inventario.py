"""Inventario de TODO el trafico entre placas en un paso: quien lo pide, cuanto mueve, a que ritmo."""
import collections, glob, gzip, json, os, sys
d = sys.argv[1]; pasos = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
f = sorted(glob.glob(os.path.join(d, "*.pt.trace.json*")))[0]
ab = gzip.open if f.endswith(".gz") else open
ev = json.load(ab(f, "rt"))["traceEvents"]
ker = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy")]
red = [e for e in ker if any(k in e["name"].lower() for k in
       ("nccl", "cross_device", "allreduce", "all_reduce", "allgather", "peer"))]
tot = sum(e["dur"] for e in ker)
print(f"total GPU {tot/pasos/1000:.1f} ms/paso | red {sum(e['dur'] for e in red)/pasos/1000:.1f} ms/paso "
      f"({100*sum(e['dur'] for e in red)/tot:.1f}%)")
# agrupar por nombre y por duracion (proxy del tamano)
g = collections.defaultdict(lambda: [0.0, 0])
for e in red:
    corto = e["name"].split("(")[0][:46]
    # bucket por orden de magnitud de la duracion
    d_ = e["dur"]
    b = "<50us" if d_ < 50 else ("50-500us" if d_ < 500 else ("0,5-3ms" if d_ < 3000 else ">3ms"))
    g[(corto, b)][0] += d_
    g[(corto, b)][1] += 1
print(f"\n{'ms/paso':>9} {'lanz/paso':>10} {'us c/u':>9}  kernel / tamano")
for (n, b), (t, c) in sorted(g.items(), key=lambda kv: -kv[1][0]):
    print(f"{t/pasos/1000:9.1f} {c/pasos:10.1f} {t/c:9.1f}  {n}  [{b}]")

# memcpy entre devices (los que NO son nccl)
mc = [e for e in ker if e.get("cat") == "gpu_memcpy"]
gm = collections.defaultdict(lambda: [0.0, 0, 0])
for e in mc:
    n = e["name"]
    gm[n][0] += e["dur"]; gm[n][1] += 1
    gm[n][2] += int(e.get("args", {}).get("bytes", 0) or 0)
if gm:
    print(f"\n{'ms/paso':>9} {'lanz/paso':>10} {'MB/paso':>9}  memcpy")
    for n, (t, c, by) in sorted(gm.items(), key=lambda kv: -kv[1][0]):
        print(f"{t/pasos/1000:9.2f} {c/pasos:10.1f} {by/pasos/1e6:9.1f}  {n}")
