"""Simula OFFLINE injertar nodos RECUPERADOS en el arbol de 8 nodos (Graft, arXiv 2605.20104), sobre
el mismo volcado real que ddtree_sim.py. El presupuesto queda fijo en 8: j nodos del arbol
best-first (los de menor probabilidad de camino) se cambian por una cadena de j tokens recuperados
que cuelga del ancla. Dos fuentes:

  bigrama   M[token] -> sucesor visto mas reciente (lo de Graft: tabla [vocab x k] en GPU), aprendida
            en linea de todo lo anterior (prompts y salidas de los pedidos previos + este hasta aca).
  sufijo    la continuacion del match mas largo (4..1 tokens) del final del contexto DENTRO del propio
            pedido (SuffixDecoding / prompt-lookup): lo que sirve cuando el modelo copia.

Aceptados del paso = max(arbol de 8-j, cadena recuperada): cota superior (si la cadena recuperada
repite un nodo del arbol no suma nada, y aca igual se cuenta). Goloso, misma salvedad de
estacionariedad que ddtree_sim.
"""
import glob, heapq, json, math, sys
DIR = sys.argv[1]
verdad = json.load(open(f"{DIR}/verdad.json"))
lineas = [json.loads(l) for f in sorted(glob.glob(f"{DIR}/dump_*.jsonl")) for l in open(f)]
por_pedido = {v["nombre"]: [] for v in verdad}
cur, ultimo = 0, -1
for ln in lineas:
    if len(ln["pos"]) != 1:
        continue
    p0 = ln["pos"][0][0]
    def en_rango(k):
        L, n = len(verdad[k]["prompt_ids"]), len(verdad[k]["out_ids"])
        return L < p0 <= L + n
    if en_rango(cur) and p0 >= ultimo:
        pass
    elif cur + 1 < len(verdad) and en_rango(cur + 1) and (p0 < ultimo or p0 > ultimo + 40):
        cur += 1
    else:
        continue
    por_pedido[verdad[cur]["nombre"]].append(ln); ultimo = p0


def lsm(v):
    m = max(v); z = m + math.log(sum(math.exp(x - m) for x in v)); return [x - z for x in v]


def conf_paso(ln):
    """Confianza del borrador en este paso: probabilidad del MEJOR camino de 4 pasos (lo que Graft
    mira en sus checkpoints). Es lo unico que decide si conviene ceder nodos a la recuperacion."""
    sc = ln["sc"][0]
    lp, j = 0.0, 0
    for t in range(min(4, len(sc))):
        ls = lsm(sc[t][j] if t else sc[0][0])
        j = max(range(len(ls)), key=lambda k: ls[k])
        lp += ls[j]
    return math.exp(lp)


def arbol(ln, real, nmax):
    cand, sc = ln["cand"][0], ln["sc"][0]; S = len(real)
    ireal = [cand[t].index(real[t]) if real[t] in cand[t] else -1 for t in range(S)]
    heap = [(-x, 0, j, ireal[0] == j) for j, x in enumerate(lsm(sc[0][0]))]; heapq.heapify(heap)
    acept, prof = [], 0
    while heap and len(acept) < nmax:
        nlp, t, j, enreal = heapq.heappop(heap)
        if enreal: prof = max(prof, t + 1)
        acept.append(prof)
        if t + 1 < S:
            for k, x in enumerate(lsm(sc[t + 1][j])):
                heapq.heappush(heap, (nlp - x, t + 1, k, enreal and ireal[t + 1] == k))
    return acept


JS = (1, 2, 3, 4)
res = {}
bigrama = {}
for v in verdad:
    toks = v["prompt_ids"] + v["out_ids"]; L = len(v["prompt_ids"])
    g = res.setdefault(v["nombre"].split("-")[0], [])
    ngr = {}                     # n-grama -> posicion siguiente a su ultima aparicion (en este pedido)
    hecho = 0                    # hasta donde se indexo toks
    for ln in por_pedido[v["nombre"]]:
        p0 = ln["pos"][0][0]; S = len(ln["pos"][0])
        if p0 >= len(toks): continue
        # indexar el contexto toks[:p0] (sin mirar el futuro)
        while hecho < p0:
            i = hecho
            if i >= 1: bigrama[toks[i - 1]] = toks[i]
            for n in (1, 2, 3, 4):
                if i - n >= 0: ngr[tuple(toks[i - n:i])] = i
            hecho += 1
        real = [toks[p] if p < len(toks) else None for p in ln["pos"][0]]
        ar = arbol(ln, real, 8)
        cf = conf_paso(ln)
        # cadena por bigrama desde el ancla
        cb, x = [], toks[p0 - 1]
        for _ in range(4):
            x = bigrama.get(x)
            if x is None: break
            cb.append(x)
        # cadena por sufijo: match mas largo del final del contexto, excluyendo el propio final
        cs = []
        for n in (4, 3, 2, 1):
            k = ngr.get(tuple(toks[p0 - n:p0])) if p0 - n >= 0 else None
            # ngr guarda la ULTIMA aparicion, que es el propio final: se busca la anterior a mano
            if k is not None:
                clave = toks[p0 - n:p0]; q = -1
                for s in range(p0 - n - 1, -1, -1):
                    if toks[s:s + n] == clave: q = s + n; break
                if q > 0: cs = toks[q:q + 4]; break
        def acc(c, j):
            a = 0
            while a < min(j, len(c)) and real[a] is not None and c[a] == real[a]: a += 1
            return a
        g.append((ar, {j: acc(cb, j) for j in JS}, {j: acc(cs, j) for j in JS}, cf))
    # los tokens que quedaron sin indexar de este pedido alimentan el bigrama de los siguientes
    for i in range(max(hecho, 1), len(toks)): bigrama[toks[i - 1]] = toks[i]

for nom, g in res.items():
    n = len(g); base = 1 + sum(x[0][7] for x in g) / n
    print(f"== {nom}: {n} pasos | arbol de 8 nodos acepta {base:.3f} por paso")
    print(f"   {'nodos recuperados j':22s} {'arbol 8-j solo':>15s} {'+ bigrama':>11s} {'+ sufijo':>10s}")
    for j in JS:
        solo = 1 + sum(x[0][7 - j] for x in g) / n
        bi = 1 + sum(max(x[0][7 - j], x[1][j]) for x in g) / n
        su = 1 + sum(max(x[0][7 - j], x[2][j]) for x in g) / n
        print(f"   j={j:<20d} {solo:15.3f} {bi:8.3f} ({bi / base - 1:+.1%}) {su:6.3f} ({su / base - 1:+.1%})")
    print(f"   la cadena recuperada sola acierta el 1er token: bigrama {sum(x[1][1] for x in g) / n:.1%}, sufijo {sum(x[2][1] for x in g) / n:.1%}")
    # Graft de verdad: NO poda siempre, poda solo cuando la confianza del borrador cae bajo tau.
    # En los pasos de confianza alta el arbol queda intacto, asi que ahi no se pierde nada.
    print(f"   poda ADAPTATIVA (solo si la confianza del paso < tau), mejor de bigrama y sufijo:")
    print(f"   {'tau':>8s} {'% pasos podados':>16s} " + " ".join(f"{'j=' + str(j):>14s}" for j in JS))
    for tau in (0.0, 0.05, 0.2, 0.5, 0.8, 0.95):
        pod = sum(1 for x in g if x[3] < tau) / n
        fila = []
        for j in JS:
            tot = sum((max(x[0][7 - j], x[1][j], x[2][j]) if x[3] < tau else x[0][7]) for x in g)
            v = 1 + tot / n
            fila.append(f"{v:7.3f} ({v / base - 1:+5.1%})")
        print(f"   {tau:8.2f} {pod:15.1%} " + " ".join(fila))
    print()
