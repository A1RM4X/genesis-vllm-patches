"""Carga tipo opencode: 1 hilo largo (~35k tokens, 3 vueltas) + 6 subagentes escalonados con
prompts de 3-4k tokens, 2 tareas cada uno, temperatura 0,6. Mide tok/s agregados, tokens por paso
y cuantas respuestas de codigo parsean. `nonce` cambia el prefijo: cada brazo paga su prefill y no
comparte prefix-cache ni offload con el otro.

Uso: VLLM_API_KEY=... carga_opencode.py <url> <nonce> [temperatura]
"""
import ast, glob, json, os, re, sys, threading, time, urllib.request
URL, NONCE = sys.argv[1], sys.argv[2]
TEMP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
H = {"Authorization": "Bearer " + os.environ["VLLM_API_KEY"], "Content-Type": "application/json"}
M = json.load(urllib.request.urlopen(urllib.request.Request(URL + "/v1/models", headers=H)))["data"][0]["id"]
RAIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "vllm", "_genesis")
fuentes = sorted(glob.glob(os.path.join(RAIZ, "*.py")), key=os.path.getsize, reverse=True)
texto = "\n\n".join(f"# ==== {os.path.basename(f)} ====\n" + open(f).read() for f in fuentes)


def met():
    t = urllib.request.urlopen(urllib.request.Request(URL + "/metrics", headers=H)).read().decode()
    g = lambda n: sum(float(x) for x in re.findall(r"^vllm:%s(?:\{[^}]*\})? ([0-9.e+]+)$" % n, t, re.M))
    return g("spec_decode_num_drafts_total"), g("spec_decode_num_accepted_tokens_total"), g("generation_tokens_total")


def chat(msgs, mt, think=False):
    b = {"model": M, "messages": msgs, "max_tokens": mt, "temperature": TEMP, "top_p": 0.95,
         "chat_template_kwargs": {"enable_thinking": think}}
    d = json.load(urllib.request.urlopen(urllib.request.Request(
        URL + "/v1/chat/completions", data=json.dumps(b).encode(), headers=H), timeout=1800))
    m = d["choices"][0]["message"]
    return (m.get("content") or ""), d["usage"]["completion_tokens"]


def parsea(txt):
    m = re.search(r"```(?:python)?\n(.*?)(?:\n```|$)", txt, re.S)
    L = (m.group(1) if m else txt).split("\n")
    for c in range(len(L), max(len(L) - 25, 0), -1):
        try:
            ast.parse("\n".join(L[:c])); return True
        except SyntaxError:
            pass
    return False


res, lock = {"tok": 0, "cod": 0, "cod_ok": 0, "malas": []}, threading.Lock()


def hilo_largo():
    msgs = [{"role": "user", "content": f"[sesion {NONCE}] Este es el codigo del proyecto:\n\n{texto[:130000]}\n\n"
             "Explica en detalle, en español, como se relacionan gdn_cinta.py y sk18_attn.py."}]
    for preg in ("Ahora describi los riesgos de concurrencia que ves en ese codigo.",
                 "Proponé tres tests nuevos para ese modulo y explicá que cubre cada uno.", None):
        txt, n = chat(msgs, 600, think=True)
        with lock: res["tok"] += n
        if preg is None: break
        msgs += [{"role": "assistant", "content": txt}, {"role": "user", "content": preg}]


TAREAS = ["a function that parses this module's environment flags into a dataclass",
          "a pytest suite with at least 8 tests for the pure-Python helpers in this file",
          "a CLI (argparse) that prints a summary table of the functions defined in this file",
          "a refactor of the largest function in this file into three smaller documented functions"]


def subagente(i):
    time.sleep(6 + 5 * i)
    for v in range(2):
        ini = 9000 * ((i * 2 + v) % 12)
        p = f"[subagente {NONCE}-{i}-{v}] File:\n\n{texto[ini: ini + 13000]}\n\nWrite {TAREAS[(i + v) % 4]}. Output only Python code."
        txt, n = chat([{"role": "user", "content": p}], 500)
        ok = parsea(txt)
        with lock:
            res["tok"] += n; res["cod"] += 1; res["cod_ok"] += ok
            if not ok: res["malas"].append(txt[-200:])


chat([{"role": "user", "content": f"[calentar {NONCE}] " + texto[:20000] + "\n\nResumi en una linea."}], 32)
d0, a0, g0 = met(); t0 = time.time()
hs = [threading.Thread(target=hilo_largo)] + [threading.Thread(target=subagente, args=(i,)) for i in range(6)]
[h.start() for h in hs]; [h.join() for h in hs]
dt = time.time() - t0; d1, a1, g1 = met()
print(f"temp {TEMP}: {res['tok']} tokens en {dt:.0f}s = {res['tok'] / dt:.1f} tok/s agregados | "
      f"{1 + (a1 - a0) / max(d1 - d0, 1):.2f} tok/paso ({int(d1 - d0)} pasos) | codigo que parsea "
      f"{res['cod_ok']}/{res['cod']} | ajeno {g1 - g0 - res['tok']:.0f}")
for m in res["malas"][:3]: print("   COLA ROTA:", repr(m))
