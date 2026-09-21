"""Recolecta pares (volcado del borrador, tokens reales) para simular DDTree / CaDDTree.

Un pedido por vez: las lineas que el volcado agrega mientras corre el pedido son suyas (las del
healthcheck se filtran despues por posicion). Greedy + return_token_ids: la respuesta final ES lo
que el target acepta. Uso: ddtree_recolectar.py <dir del volcado en el host> <salida.json>
"""
import glob, json, os, random, sys, time, urllib.request

URL = os.environ.get("URL", "http://localhost:8360")
H = {"Authorization": "Bearer " + os.environ["VLLM_API_KEY"], "Content-Type": "application/json"}
DIR, SALIDA = sys.argv[1], sys.argv[2]
M = json.load(urllib.request.urlopen(urllib.request.Request(URL + "/v1/models", headers=H)))["data"][0]["id"]

PROSA = ("crea una nueva, sin usar ningun codigo viejo pagoda japonesa en three.js , voxel , muy detallado, en una "
         "isla flotando en el espacio, con su propia atmosfera, y parque japones perfecto, cascada, montaña y arcoiris "
         "añadirle niebla volumétrica real (fog shader), sombras dinámicas, sonido ambiental o más estructuras")
CODIGO = ("Write a complete, production-quality Python implementation of {} with a thorough pytest test suite. "
          "Output only code.")
TEMAS = ["a red-black tree", "an LRU cache with TTL", "a trie with deletion and prefix search"]


def contexto_largo(semilla):
    r = random.Random(semilla)
    noms = "parse load merge split render fetch store index scan build plan route cache flush".split()
    def fn(i):
        a, b = r.choice(noms), r.choice(noms)
        return (f"def {a}_{b}_{i}(items, limit={r.randint(2, 99)}):\n    \"\"\"{a} then {b} the items.\"\"\"\n"
                f"    out = []\n    for k, item in enumerate(items):\n        if k >= limit:\n            break\n"
                f"        out.append(({r.randint(1, 9)} * k, str(item).{r.choice(['upper', 'lower', 'strip', 'title'])}()))\n"
                f"    return out\n\n")
    return (f"# corrida ddtree {semilla}\n" + "".join(fn(i) for i in range(560)) +
            "\n\nWrite a new module that reuses several of the functions above: a class `Pipeline` with methods to "
            "chain them, full type hints, and pytest tests. Output only code.")


def lineas():
    return sum(sum(1 for _ in open(f)) for f in glob.glob(os.path.join(DIR, "dump_*.jsonl")))


def pedir(nombre, prompt, mt, think):
    n0 = lineas()
    b = {"model": M, "messages": [{"role": "user", "content": prompt}], "max_tokens": mt, "temperature": 0,
         "return_token_ids": True, "chat_template_kwargs": {"enable_thinking": think}}
    t0 = time.time()
    d = json.load(urllib.request.urlopen(urllib.request.Request(
        URL + "/v1/chat/completions", data=json.dumps(b).encode(), headers=H), timeout=1800))
    dt = time.time() - t0
    time.sleep(1.5)                                       # que el volcado termine de escribir
    c = d["choices"][0]
    prompt_ids = d.get("prompt_token_ids") or c.get("prompt_token_ids")
    out_ids = c.get("token_ids")
    assert prompt_ids and out_ids, "la API no devolvio los ids de token"
    print(f"{nombre}: prompt {len(prompt_ids)} tok, salida {len(out_ids)} tok, {dt:.1f}s, lineas {n0}..{lineas()}", flush=True)
    return {"nombre": nombre, "prompt_ids": prompt_ids, "out_ids": out_ids, "linea0": n0, "linea1": lineas()}


reg = []
for i, t in enumerate(TEMAS):
    reg.append(pedir(f"codigo-{i}", CODIGO.format(t), 1500, False))
for i in range(2):
    reg.append(pedir(f"prosa-{i}", PROSA + f" (variante {i})", 1500, True))
for i in range(2):
    reg.append(pedir(f"largo-{i}", contexto_largo(int(time.time()) + i), 800, False))
json.dump(reg, open(SALIDA, "w"))
print("guardado", SALIDA)
