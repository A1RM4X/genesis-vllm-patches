"""¿El AllGather tarda porque transfiere, o porque espera a que el otro rank llegue?

Si los dos ranks no llegan al mismo tiempo al colectivo, el que llega primero se queda adentro del
kernel esperando, y esa espera se cuenta como duracion. Comparando las dos trazas se ve.
"""
import glob, gzip, json, os, sys
d = sys.argv[1]; pasos = float(sys.argv[2])
fs = sorted(glob.glob(os.path.join(d, "*.pt.trace.json*")))
print(f"{len(fs)} trazas")
por_rank = {}
for f in fs:
    ab = gzip.open if f.endswith(".gz") else open
    ev = json.load(ab(f, "rt"))["traceEvents"]
    ker = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel"]
    ag = sorted([e for e in ker if "AllGather" in e["name"]], key=lambda e: e["ts"])
    ag = [e for e in ag if e["dur"] > 500]          # solo los grandes (los datos)
    r = "rank1" if "rank1" in os.path.basename(f) else "rank0"
    por_rank[r] = ag
    tot = sum(e["dur"] for e in ag)
    print(f"  {r}: {len(ag)} allgather grandes, {tot/pasos/1000:.1f} ms/paso, "
          f"media {tot/max(1,len(ag)):.0f} us, min {min(e['dur'] for e in ag):.0f}, "
          f"max {max(e['dur'] for e in ag):.0f}")
if len(por_rank) == 2:
    a, b = por_rank["rank0"], por_rank["rank1"]
    n = min(len(a), len(b))
    print(f"\npor colectivo (los primeros {min(n,14)}): duracion en cada rank y la diferencia")
    print(f"{'#':>3} {'rank0 us':>9} {'rank1 us':>9} {'dif':>9}  {'el que espera':>14}")
    difs = []
    for i in range(n):
        d0, d1 = a[i]["dur"], b[i]["dur"]
        difs.append(abs(d0 - d1))
        if i < 14:
            q = "rank0" if d0 > d1 else "rank1"
            print(f"{i:3d} {d0:9.0f} {d1:9.0f} {d0-d1:9.0f}  {q:>14}")
    difs.sort()
    print(f"\ndiferencia mediana {difs[len(difs)//2]:.0f} us, media {sum(difs)/len(difs):.0f} us")
    # el tiempo REAL de transferencia es el minimo de los dos (el que llego ultimo casi no espera)
    reales = [min(a[i]["dur"], b[i]["dur"]) for i in range(n)]
    print(f"transferencia real (el minimo de los dos): {sum(reales)/pasos/1000:.1f} ms/paso")
    print(f"medido en rank0:                          {sum(e['dur'] for e in a)/pasos/1000:.1f} ms/paso")
    print(f"-> ESPERA por desbalance: "
          f"{(sum(e['dur'] for e in a)-sum(reales))/pasos/1000:.1f} ms/paso")
