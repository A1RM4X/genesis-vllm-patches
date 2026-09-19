#!/usr/bin/env python3
"""N requests en paralelo contra el servidor: throughput agregado, fallas y uso de KV.

Uso: concurrencia.py <ip:puerto> <n> [tokens_de_prompt] [tokens_de_salida]

Cada request lleva un nonce distinto: con prefix-caching prendido, N requests iguales miden
el cache y no el motor. Se reporta el pico de uso de KV durante la corrida, que es el numero
que dice si la memoria alcanzo o si hubo preempcion.
"""

import json
import sys
import threading
import time
import urllib.request
import os

IP = sys.argv[1]
N = int(sys.argv[2])
NTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
SALIDA = int(sys.argv[4]) if len(sys.argv) > 4 else 200
H = {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ.get("VLLM_API_KEY", "")}

res = [None] * N
pico = {"kv": 0.0, "corriendo": 0, "esperando": 0, "preempt": 0.0}
parar = threading.Event()


def vigilar():
    while not parar.is_set():
        try:
            req = urllib.request.Request(f"http://{IP}/metrics", headers=H)
            for l in urllib.request.urlopen(req, timeout=10).read().decode().splitlines():
                if l.startswith("#"):
                    continue
                v = l.rsplit(" ", 1)
                if len(v) != 2:
                    continue
                try:
                    x = float(v[1])
                except ValueError:
                    continue
                if "kv_cache_usage_perc" in l:
                    pico["kv"] = max(pico["kv"], x)
                elif "num_requests_running" in l:
                    pico["corriendo"] = max(pico["corriendo"], int(x))
                elif "num_requests_waiting{" in l:
                    pico["esperando"] = max(pico["esperando"], int(x))
                elif l.startswith("vllm:num_preemptions_total"):
                    pico["preempt"] = max(pico["preempt"], x)
        except Exception:
            pass
        time.sleep(0.5)


def uno(i):
    base = (f"Expediente {i}-{time.time()}. Analisis tecnico del subsistema de memoria bajo "
            "carga sostenida, con foco en el trafico entre niveles de cache y el reparto de "
            "trabajo entre unidades de computo. ")
    texto = (base * (NTOK // 40 + 2))[: NTOK * 4]
    cuerpo = json.dumps({"model": "qwen3.8", "prompt": texto + "\n\nResumi lo anterior.",
                         "max_tokens": SALIDA, "temperature": 0.7}).encode()
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(
            urllib.request.Request(f"http://{IP}/v1/completions", cuerpo, H), timeout=900))
        res[i] = (d["usage"]["completion_tokens"], time.time() - t0, None)
    except Exception as e:
        res[i] = (0, time.time() - t0, f"{type(e).__name__}: {e}")


threading.Thread(target=vigilar, daemon=True).start()
t0 = time.time()
hilos = [threading.Thread(target=uno, args=(i,)) for i in range(N)]
for h in hilos:
    h.start()
for h in hilos:
    h.join()
total = time.time() - t0
parar.set()

ok = [r for r in res if r and not r[2]]
mal = [r for r in res if r and r[2]]
toks = sum(r[0] for r in ok)
print(f"  {len(ok)}/{N} OK   pared {total:.1f} s   {toks} tokens   "
      f"agregado {toks/total:.1f} tok/s")
if ok:
    lat = sorted(r[1] for r in ok)
    print(f"  latencia por request: min {lat[0]:.1f} s  mediana {lat[len(lat)//2]:.1f} s  "
          f"max {lat[-1]:.1f} s")
print(f"  pico KV {100*pico['kv']:.1f}%   corriendo {pico['corriendo']}   "
      f"en cola {pico['esperando']}   preempciones {pico['preempt']:.0f}")
for r in mal:
    print(f"  FALLA: {r[2][:150]}")
