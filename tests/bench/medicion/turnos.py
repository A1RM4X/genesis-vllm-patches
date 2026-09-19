"""Conversacion de agente: prefijo largo + N turnos que agregan contenido nuevo.
Mide por turno el TTFT (streaming) y los tokens de prefix cache reutilizados."""
import glob, json, sys, time, urllib.request
import os
BASE = "http://127.0.0.1:8320/v1/chat/completions"
H = {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ.get("VLLM_API_KEY", "")}
TAG = sys.argv[1]; TURNOS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
fs = sorted(glob.glob("/home/usuario/Proyectos/genesis-vllm-patches/vllm/_genesis/wiring/**/*.py", recursive=True))
corpus = "".join(f"# ==== {f.split('wiring/')[-1]} ====\n" + open(f, errors="ignore").read() for f in fs)
base = corpus[:110_000]          # ~30k tokens
trozos = [corpus[110_000 + i * 7000: 110_000 + (i + 1) * 7000] for i in range(TURNOS)]   # ~2k tokens c/u
msgs = [{"role": "system", "content": f"[{TAG}-{time.time()}] Sos un asistente de codigo."},
        {"role": "user", "content": "Leé este código y esperá instrucciones:\n\n" + base}]
tot_ttft = 0.0
for t in range(TURNOS):
    msgs.append({"role": "user", "content": f"Turno {t}: ahora mirá este archivo y decime en una línea qué hace:\n\n" + trozos[t]})
    body = json.dumps({"model": "qwen3.8", "messages": msgs, "max_tokens": 96, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    t0 = time.perf_counter(); ttft = None; texto = ""; uso = None
    with urllib.request.urlopen(urllib.request.Request(BASE, body, H), timeout=1800) as r:
        for linea in r:
            linea = linea.decode().strip()
            if not linea.startswith("data: ") or linea == "data: [DONE]":
                continue
            d = json.loads(linea[6:])
            if d.get("usage"):
                uso = d["usage"]
            for ch in d.get("choices", []):
                c = ch.get("delta", {}).get("content")
                if c:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    texto += c
    msgs.append({"role": "assistant", "content": texto})
    cached = (uso.get("prompt_tokens_details") or {}).get("cached_tokens", 0) if uso else 0
    nuevos = uso["prompt_tokens"] - cached if uso else 0
    tot_ttft += ttft or 0
    print(f"{TAG} turno {t:2d}: prompt={uso['prompt_tokens'] if uso else '?':>6} cacheados={cached:>6} nuevos={nuevos:>5} TTFT={ttft:.2f}s", flush=True)
print(f"TOTAL {TAG}: TTFT sumado {tot_ttft:.1f}s en {TURNOS} turnos")
