import collections, glob, gzip, json, os, sys
f = sorted(glob.glob(os.path.join(sys.argv[1], "*.pt.trace.json*")))[0]
ab = gzip.open if f.endswith(".gz") else open
ev = json.load(ab(f, "rt"))["traceEvents"]
ker = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel" and "nccl" in e["name"].lower()]
ker.sort(key=lambda e: e["ts"])
d = sorted(e["dur"] for e in ker)
n = len(d)
print(f"{n} lanzamientos nccl, total {sum(d)/1000:.0f} ms")
for p in (0, 10, 25, 50, 75, 90, 99, 100):
    print(f"  p{p:3d}: {d[min(n-1, p*n//100)]:9.1f} us")
# agrupar por duracion (chicos vs grandes)
chicos = [x for x in d if x < 100]
grandes = [x for x in d if x >= 100]
print(f"  < 100us: {len(chicos):5d} lanz, {sum(chicos)/1000:8.1f} ms")
print(f" >= 100us: {len(grandes):5d} lanz, {sum(grandes)/1000:8.1f} ms")
# huecos entre nccl consecutivos (latencia de sincronizacion)
print("\nprimeros 12 nccl con su duracion y el hueco al anterior:")
prev = None
for e in ker[:12]:
    hueco = "" if prev is None else f"  hueco {e['ts']-prev:8.1f} us"
    print(f"  {e['dur']:9.1f} us  {e['name'][:52]}{hueco}")
    prev = e["ts"] + e["dur"]
