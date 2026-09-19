"""Estres de concurrencia MTP: N requests en paralelo, mezcla de prompts largos y
generaciones largas, durante DURACION segundos. Reporta OK/ERR y tok/s."""
import json, random, sys, threading, time, urllib.request
import os
BASE = "http://127.0.0.1:8320/v1/chat/completions"
H = {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ.get("VLLM_API_KEY", "")}
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
DUR = int(sys.argv[2]) if len(sys.argv) > 2 else 300
FILL = open("/home/usuario/Proyectos/genesis-vllm-patches/docs/LOGROS.md").read()
TEMAS = ["Escribi una funcion en Python que implemente un LRU cache con tests.",
         "Explica paso a paso como funciona el scheduler de un sistema operativo.",
         "Refactoriza este codigo y explica los cambios:\n" + FILL[:6000],
         "Resumi el siguiente documento en 20 puntos:\n" + FILL[:20000]]
ok = err = toks = 0
lock = threading.Lock()
fin = time.time() + DUR
def trabajador(i):
    global ok, err, toks
    r = random.Random(i)
    while time.time() < fin:
        body = json.dumps({"model": "qwen3.8", "messages": [{"role": "user", "content": f"[{i}-{time.time()}] " + r.choice(TEMAS)}],
                           "max_tokens": r.choice([300, 800, 1500]), "temperature": 0.7}).encode()
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request(BASE, body, H), timeout=600))
            with lock:
                ok += 1; toks += d["usage"]["completion_tokens"]
        except Exception as e:
            with lock:
                err += 1
            print(f"  ERR worker {i}: {str(e)[:120]}", flush=True)
            time.sleep(5)
t0 = time.time()
hs = [threading.Thread(target=trabajador, args=(i,)) for i in range(N)]
[h.start() for h in hs]; [h.join() for h in hs]
dt = time.time() - t0
print(f"concurrencia={N} duracion={dt:.0f}s OK={ok} ERR={err} tokens={toks} ({toks/dt:.0f} tok/s agregado)")
