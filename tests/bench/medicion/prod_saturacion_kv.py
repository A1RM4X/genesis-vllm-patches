#!/usr/bin/env python3
"""Llena la KV con pedidos en paralelo y vigila si el servidor se clava.

Interesa saber tres cosas que no se ven en un banco de throughput:
  1. si con la KV por encima del 50% los pedidos siguen avanzando o se frenan,
  2. cuanta VRAM queda libre de verdad en ese momento (no la que vLLM reservo),
  3. si aparecen preempciones, que es como vLLM desaloja cuando no le entra.

Uso: prod_saturacion_kv.py [pedidos] [tokens_de_prompt] [tokens_de_salida]
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
import os

H = {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ.get("VLLM_API_KEY", "")}
BASE = "http://127.0.0.1:8320"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
PROMPT_TOK = int(sys.argv[2]) if len(sys.argv) > 2 else 32000
SALIDA_TOK = int(sys.argv[3]) if len(sys.argv) > 3 else 1500

_parar = threading.Event()
_muestras: list[dict] = []


def relleno(tokens: int, sal: str) -> str:
    base = ("El expediente registra cada actuacion con su fecha, el organismo interviniente, "
            "el funcionario responsable, la norma que se invoca y el plazo que corre a partir "
            "de la notificacion. Las vistas se conceden por cinco dias habiles y su vencimiento "
            "habilita a resolver de oficio, salvo que medie pedido de prorroga fundado. ")
    pal = int(tokens / 1.4)
    return sal + " " + " ".join((base * (pal // len(base.split()) + 2)).split()[:pal])


def metricas() -> dict:
    try:
        r = urllib.request.urlopen(urllib.request.Request(BASE + "/metrics", headers=H), timeout=10)
        d = {}
        for ln in r.read().decode().splitlines():
            if ln.startswith("#") or " " not in ln:
                continue
            k, _, v = ln.rpartition(" ")
            try:
                d[k.split("{")[0]] = float(v)
            except ValueError:
                pass
        return d
    except Exception:
        return {}


def vram() -> list[tuple[int, int, int]]:
    """(usada, libre, total) en MiB por GPU, del driver — no lo que vLLM cree."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        return [tuple(int(x) for x in ln.split(", ")) for ln in out.strip().splitlines()]
    except Exception:
        return []


def vigilar() -> None:
    while not _parar.is_set():
        m = metricas()
        _muestras.append({
            "t": time.perf_counter(),
            "kv_pct": m.get("vllm:gpu_cache_usage_perc", m.get("vllm:kv_cache_usage_perc", 0.0)) * 100,
            "corriendo": m.get("vllm:num_requests_running", 0),
            "en_cola": m.get("vllm:num_requests_waiting", 0),
            "preempt": m.get("vllm:num_preemptions_total", 0),
            "gen": m.get("vllm:generation_tokens_total", 0),
            "vram": vram(),
        })
        _parar.wait(2.0)


def pedir(i: int, salida: list) -> None:
    p = relleno(PROMPT_TOK, f"[caso {uuid.uuid4().hex}] ")
    cuerpo = json.dumps({
        "model": "qwen3.8", "max_tokens": SALIDA_TOK, "temperature": 0.7, "seed": i,
        "messages": [{"role": "user", "content": p + "\n\nAnaliza el expediente: resumi las "
                      "actuaciones, identifica los plazos que corren y que pasa si vencen. "
                      "Se exhaustivo."}],
        "chat_template_kwargs": {"enable_thinking": False}}).encode()
    t0 = time.perf_counter()
    try:
        r = json.load(urllib.request.urlopen(
            urllib.request.Request(BASE + "/v1/chat/completions", cuerpo, H), timeout=3600))
        salida[i] = (r["usage"]["completion_tokens"], r["usage"]["prompt_tokens"],
                     time.perf_counter() - t0, None)
    except Exception as e:
        salida[i] = (0, 0, time.perf_counter() - t0, f"{type(e).__name__}: {e}")


def main() -> None:
    libre0 = vram()
    print(f"  {N} pedidos x {PROMPT_TOK} tokens de prompt x {SALIDA_TOK} de salida")
    print(f"  VRAM antes de empezar: " +
          " | ".join(f"GPU{i} {u} usados / {l} libres de {t} MiB"
                     for i, (u, l, t) in enumerate(libre0)))
    hilo = threading.Thread(target=vigilar, daemon=True)
    hilo.start()
    salida = [None] * N
    hilos = [threading.Thread(target=pedir, args=(i, salida)) for i in range(N)]
    t0 = time.perf_counter()
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()
    pared = time.perf_counter() - t0
    _parar.set()
    hilo.join(timeout=5)

    pico = max(_muestras, key=lambda m: m["kv_pct"]) if _muestras else None
    altas = [m for m in _muestras if m["kv_pct"] >= 50]
    print(f"\n  pared={pared:.1f}s")
    if pico:
        print(f"  KV pico = {pico['kv_pct']:.1f}%   muestras con KV >= 50%: {len(altas)} de {len(_muestras)}")
        print(f"  preempciones = {max(m['preempt'] for m in _muestras) - min(m['preempt'] for m in _muestras):.0f}")
    if altas:
        print(f"\n  VRAM CON LA KV POR ENCIMA DEL 50%:")
        for m in altas[:: max(1, len(altas) // 6)]:
            v = " | ".join(f"GPU{i}: {l} MiB libres ({u}/{t} usados)"
                           for i, (u, l, t) in enumerate(m["vram"]))
            print(f"    KV {m['kv_pct']:5.1f}%  corriendo {m['corriendo']:.0f}  "
                  f"cola {m['en_cola']:.0f}   {v}")
        mn = min(min(l for _, l, _ in m["vram"]) for m in altas if m["vram"])
        print(f"    minimo de VRAM libre observado con KV>=50%: {mn} MiB")

    print(f"\n  avance de la generacion (para ver si se clava):")
    prev = None
    for m in _muestras[:: max(1, len(_muestras) // 8)]:
        d = "" if prev is None else f"  +{m['gen'] - prev:.0f} tok"
        print(f"    t={m['t'] - _muestras[0]['t']:6.1f}s  KV {m['kv_pct']:5.1f}%  "
              f"corriendo {m['corriendo']:.0f}  cola {m['en_cola']:.0f}{d}")
        prev = m["gen"]

    fallos = [(i, s[3]) for i, s in enumerate(salida) if s and s[3]]
    ok = [s for s in salida if s and not s[3]]
    print(f"\n  pedidos OK: {len(ok)}/{N}   tokens generados: {sum(s[0] for s in ok)}")
    if ok:
        tiempos = sorted(s[2] for s in ok)
        print(f"  por pedido: mediana {tiempos[len(tiempos) // 2]:.1f}s   peor {tiempos[-1]:.1f}s")
    for i, e in fallos:
        print(f"  FALLO pedido {i}: {e}")


if __name__ == "__main__":
    main()
