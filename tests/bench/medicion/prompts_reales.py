"""Banco con prompts REALES (no plantillas sinteticas): cuento infantil, Tetris en TUI, explicar un
motor, un email, un parser de JSON en C... Mezcla español/ingles, prosa/codigo, corto/largo, con y
sin thinking, que es lo que de verdad se le pide al modelo.

Cada tarea trae su propio ``mt`` (max_tokens), CALIBRADO (2026-09-21) para que la respuesta TERMINE
sola: si corta por longitud se mide un texto truncado, el chequeo de sintaxis da falso negativo y
el largo deja de ser comparable. Calibrado en dos pasadas: con los techos iniciales cortaban las
ocho, y a las tres de codigo largo no habia que subirles el techo sino ACOTAR el pedido (las tres
seguian cortando con 5000-5500). El script avisa si alguna vuelve a cortar por `length`.

Uso:  VLLM_API_KEY=... prompts_reales.py <url> <nonce> [temp] [concurrencia]
"""
import ast, json, os, re, sys, threading, time, urllib.request

URL, NONCE = sys.argv[1], sys.argv[2]
TEMP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
CONC = int(sys.argv[4]) if len(sys.argv) > 4 else 1
H = {"Authorization": "Bearer " + os.environ["VLLM_API_KEY"], "Content-Type": "application/json"}
M = json.load(urllib.request.urlopen(urllib.request.Request(URL + "/v1/models", headers=H)))["data"][0]["id"]

# (nombre, prompt, max_tokens, thinking, es_codigo)
TAREAS = [
    ("cuento", "Escribí un cuento para niños sobre un dragón que le tiene miedo a volar. "
     "Que tenga principio, nudo y desenlace, y termine bien.", 2500, False, False),
    ("tetris", "Programá en Python un Tetris jugable en la terminal con curses: las 7 piezas, "
     "rotación, líneas completas y puntaje. Código conciso, sin comentarios largos ni menús.",
     3500, False, True),
    ("motor", "Explicame en detalle cómo funciona un motor de combustión interna de cuatro "
     "tiempos, incluyendo qué pasa en cada tiempo y por qué hace falta el volante de inercia.",
     2500, False, False),
    ("backup", "Write a bash script that does incremental backups with rsync, keeps 7 daily and "
     "4 weekly snapshots with hardlinks, logs to syslog and handles errors. Complete script.",
     3000, False, False),
    ("ciudad", "Hacé un análisis equilibrado de los pros y contras de vivir en una ciudad grande "
     "frente a un pueblo chico, pensando en alguien de 30 años que trabaja de forma remota.",
     3500, True, False),
    ("parser", "Implement a JSON parser in C with no dependencies: tokenizer and recursive "
     "descent parser over a tagged union value type. Concise, no tests, no example main.",
     3500, False, True),
    ("email", "Escribí un email formal y bien argumentado para pedirle a mi jefe un aumento de "
     "sueldo, mencionando logros concretos del último año.", 1500, False, False),
    ("api", "Write a FastAPI task manager: JWT auth plus three endpoints (create, list, delete) "
     "with Pydantic schemas and an in-memory store. Concise, no database, no tests.",
     3000, False, True),
]


def pedir(t):
    nombre, p, mt, think, es_cod = t
    # El nonce NO puede ir en el prompt: cambia el texto generado y entonces cada corrida mide
    # otra cosa (medido: 14262 a 17244 tokens entre brazos, 20% de dispersion con 6 pedidos).
    # Va en `user`, que vLLM no mete en el contexto, asi que el prompt es IDENTICO en todos los
    # brazos y en goloso el texto tambien. Para que no se compartan bloques cacheados entre brazos
    # alcanza con recrear el contenedor, que es lo que hace el script del A/B.
    b = {"model": M, "messages": [{"role": "user", "content": p}], "user": NONCE,
         "max_tokens": mt, "temperature": TEMP,
         "chat_template_kwargs": {"enable_thinking": think}}
    if TEMP > 0:
        b["top_p"] = 0.95
    t0 = time.time()
    d = json.load(urllib.request.urlopen(urllib.request.Request(
        URL + "/v1/chat/completions", data=json.dumps(b).encode(), headers=H), timeout=1800))
    dt = time.time() - t0
    ch = d["choices"][0]
    txt = ch["message"].get("content") or ""
    ok = None
    if es_cod:
        m = re.search(r"```(?:\w+)?\n(.*?)(?:\n```|$)", txt, re.S)
        src = (m.group(1) if m else txt)
        if "python" in p.lower() or "FastAPI" in p:
            L = src.split("\n"); ok = False
            for c in range(len(L), max(len(L) - 20, 0), -1):
                try:
                    ast.parse("\n".join(L[:c])); ok = True; break
                except SyntaxError:
                    pass
        else:                                   # C / bash: balance de llaves como chequeo minimo
            ok = src.count("{") == src.count("}") and src.count("(") == src.count(")")
    return dict(nombre=nombre, n=d["usage"]["completion_tokens"], dt=dt, fin=ch["finish_reason"],
                ok=ok, mt=mt)


def met():
    t = urllib.request.urlopen(urllib.request.Request(URL + "/metrics", headers=H)).read().decode()
    g = lambda n: sum(float(x) for x in re.findall(r"^vllm:%s(?:\{[^}]*\})? ([0-9.e+]+)$" % n, t, re.M))
    return g("spec_decode_num_drafts_total"), g("spec_decode_num_accepted_tokens_total"), g("generation_tokens_total")


pedir(("calentar", "decime hola", 16, False, False))
d0, a0, g0 = met()
res, lock = [], threading.Lock()
t0 = time.time()
if CONC == 1:
    for t in TAREAS:
        r = pedir(t); res.append(r)
else:
    def correr(i):
        r = pedir(TAREAS[i % len(TAREAS)])
        with lock:
            res.append(r)
    hs = [threading.Thread(target=correr, args=(i,)) for i in range(CONC)]
    [h.start() for h in hs]; [h.join() for h in hs]
dt_tot = time.time() - t0
d1, a1, g1 = met()
mios = sum(r["n"] for r in res)
print(f"  temp {TEMP} conc {CONC}: {mios} tokens en {dt_tot:.1f}s = {mios / dt_tot:6.1f} tok/s agregados"
      f" | {1 + (a1 - a0) / max(d1 - d0, 1):.2f} tok/paso | ajeno {g1 - g0 - mios:.0f}")
if CONC == 1:
    for r in sorted(res, key=lambda r: r["nombre"]):
        aviso = "  <-- CORTADO por max_tokens" if r["fin"] == "length" else ""
        cal = "" if r["ok"] is None else (" codigo OK" if r["ok"] else " codigo ROTO")
        print(f"    {r['nombre']:8s} {r['n']:5d}/{r['mt']:<5d} tok {r['n'] / r['dt']:6.1f} tok/s"
              f" fin={r['fin']}{cal}{aviso}")
