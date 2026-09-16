#!/usr/bin/env python3
"""Fidelidad de la KV por teacher forcing con sondeos (no usa prompt_logprobs, que
esta roto en este servidor).

Toma un documento real largo tokenizado, y en N puntos repartidos por profundidad
pide 1 token con top-20 logprobs enviando los token ids exactos del prefijo. Guarda
por sondeo: token predicho (top-1), logprob del token real (o piso si no esta en el
top-20) y la distribucion top-20 para calcular KL aproximado.

Uso: sondeos.py <nombre> [n_sondeos]      -> /tmp/g115/sond_<nombre>.json
     sondeos.py --comparar ref otro [otro2]
"""
import glob, json, math, sys, time, urllib.request
BASE = "http://127.0.0.1:8320"; KEY = "<REDACTADO: clave rotada 2026-09-19>"
H = {"Content-Type": "application/json", "Authorization": "Bearer " + KEY}
PISO = -20.0

def post(ruta, d, timeout=900):
    return json.load(urllib.request.urlopen(urllib.request.Request(BASE + ruta, json.dumps(d).encode(), H), timeout=timeout))

def documento():
    fs = sorted(glob.glob("/home/usuario/Proyectos/genesis-vllm-patches/vllm/_genesis/**/*.py", recursive=True))
    return "".join(f"# ==== {f.split('_genesis/')[-1]} ====\n" + open(f, errors="ignore").read() for f in fs)[:420_000]

def medir(nombre, n):
    ids = post("/tokenize", {"model": "qwen3.8", "prompt": documento()})["tokens"][:100_000]
    L = len(ids)
    # profundidades log-espaciadas entre 512 y L-1, ordenadas: el prefix cache
    # hace que cada sondeo solo procese la cola desde el anterior
    cortes = sorted({int(math.exp(math.log(512) + i * (math.log(L - 2) - math.log(512)) / (n - 1))) for i in range(n)})
    res = []; t0 = time.time()
    for c in cortes:
        d = post("/v1/completions", {"model": "qwen3.8", "prompt": ids[:c], "max_tokens": 1, "temperature": 0,
                                     "logprobs": 20, "detokenize": False, "return_tokens_as_token_ids": True})
        lp = d["choices"][0]["logprobs"]
        top = lp["top_logprobs"][0]                      # {"token_id:N": logprob}
        tops = {int(k.split(":")[-1]): v for k, v in top.items()}
        real = ids[c]
        res.append({"pos": c, "real": real, "pred": max(tops, key=tops.get),
                    "lp_real": tops.get(real, PISO), "top": tops})
    json.dump(res, open(f"/tmp/g115/sond_{nombre}.json", "w"))
    acc = sum(r["pred"] == r["real"] for r in res) / len(res)
    print(f"{nombre}: {len(res)} sondeos hasta {cortes[-1]} tokens en {time.time()-t0:.0f}s | top-1 == token real {100*acc:.1f}% | "
          f"logp medio del real {sum(r['lp_real'] for r in res)/len(res):.3f}")

def comparar(ref, otros):
    A = json.load(open(f"/tmp/g115/sond_{ref}.json"))
    tramos = [(0, 4096), (4096, 16384), (16384, 50000), (50000, 10**9)]
    for o in otros:
        B = json.load(open(f"/tmp/g115/sond_{o}.json"))
        print(f"\n{o} contra {ref}")
        print(f"{'profundidad':>16} {'n':>4} {'top-1 igual':>11} {'|dlogp real|':>13} {'KL~ (top20)':>12}")
        for lo, hi in tramos + [(0, 10**9)]:
            pares = [(a, b) for a, b in zip(A, B) if lo <= a["pos"] < hi]
            if not pares: continue
            ig = sum(a["pred"] == b["pred"] for a, b in pares) / len(pares)
            dl = sum(abs(a["lp_real"] - b["lp_real"]) for a, b in pares) / len(pares)
            kl = 0.0
            for a, b in pares:
                for t, la in a["top"].items():
                    lb = b["top"].get(t, min(b["top"].values()))
                    kl += math.exp(la) * (la - lb)
            kl /= len(pares)
            et = "TOTAL" if hi == 10**9 and lo == 0 else f"{lo}-{hi if hi < 10**9 else 'fin'}"
            print(f"{et:>16} {len(pares):>4} {100*ig:>10.1f}% {dl:>13.4f} {kl:>12.4f}")

if sys.argv[1] == "--comparar":
    comparar(sys.argv[2], sys.argv[3:])
else:
    medir(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 160)
