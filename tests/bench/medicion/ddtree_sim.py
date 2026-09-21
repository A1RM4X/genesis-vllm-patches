"""Simula OFFLINE verificar un arbol en vez de una cadena con el borrador DFlash2.

Entrada: el volcado de ``vllm/_genesis/diag_ddtree_dump.py`` (por paso: 16 candidatos por posicion
y la tabla de puntajes de transicion condicionada al candidato previo) y ``verdad.json`` (los
tokens reales, greedy = lo que el target acepta). Politicas:

  cadena     lo de hoy: un camino, recorrido goloso.
  arbol N    best-first sobre la probabilidad de camino (DDTree, arXiv 2604.12989), N nodos.
  por costo  el presupuesto se elige EN CADA PASO maximizando aceptados esperados / costo
             (CaDDTree, arXiv 2606.01813), con el modelo medido en este rig:
             paso = FIJO/B + C * (tokens verificados),  FIJO ~17 ms, C ~0,65 ms.

Limite del metodo: los pasos salen de la trayectoria de la cadena; con un arbol el paso siguiente
arrancaria en otra posicion. Se asume estacionariedad: sirve para estimar el techo, no mas.
"""
import glob, heapq, json, math, sys

DIR = sys.argv[1]
FIJO, C = float(sys.argv[2]) if len(sys.argv) > 2 else 17.0, float(sys.argv[3]) if len(sys.argv) > 3 else 0.65
verdad = json.load(open(f"{DIR}/verdad.json"))
lineas = [json.loads(l) for f in sorted(glob.glob(f"{DIR}/dump_*.jsonl")) for l in open(f)]

# Asignar cada paso a su pedido. Los pedidos corrieron de a uno y en orden, asi que alcanza con
# un puntero: un paso es del pedido actual si predice una posicion de SU salida y no retrocede.
# Se descartan solos los pasos del healthcheck (posiciones chicas), los cruzados con otro pedido
# y las propuestas que el borrador hace DURANTE el prefill por trozos (posicion <= largo del prompt).
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
    # Solo se pasa al pedido siguiente si la posicion RETROCEDE o SALTA: los borradores se pasan
    # unos tokens del final de cada pedido y esas lineas caen en el rango del que sigue.
    elif cur + 1 < len(verdad) and en_rango(cur + 1) and (p0 < ultimo or p0 > ultimo + 40):
        cur += 1
    else:
        continue
    por_pedido[verdad[cur]["nombre"]].append(ln); ultimo = p0


def logsoftmax(v):
    m = max(v); z = m + math.log(sum(math.exp(x - m) for x in v)); return [x - z for x in v]


def paso(ln, toks, nmax):
    """-> (aceptados cadena, lista de aceptados del arbol para N=1..nmax, esperado acumulado por N)."""
    pos, cand, sc, borr = ln["pos"][0], ln["cand"][0], ln["sc"][0], ln["borrador"][0]
    S = len(pos)
    real = [toks[p] if p < len(toks) else None for p in pos]
    if real[0] is None:
        return None
    ch = 0
    while ch < S and real[ch] is not None and borr[ch] == real[ch]:
        ch += 1
    # indice del candidato real en cada profundidad (o -1)
    ireal = [cand[t].index(real[t]) if real[t] in cand[t] else -1 for t in range(S)]
    # best-first: nodo = (profundidad t, indice i elegido en t, logp, viene_del_camino_real)
    heap = []
    ls0 = logsoftmax(sc[0][0])
    for j in range(len(ls0)):
        heapq.heappush(heap, (-ls0[j], 0, j, ireal[0] == j))
    acept, esper, prof, e = [], [], 0, 0.0
    while heap and len(acept) < nmax:
        nlp, t, j, enreal = heapq.heappop(heap)
        e += math.exp(-nlp)
        if enreal:
            prof = max(prof, t + 1)
        acept.append(prof); esper.append(e)
        if t + 1 < S:
            ls = logsoftmax(sc[t + 1][j])
            for k in range(len(ls)):
                heapq.heappush(heap, (nlp - ls[k], t + 1, k, enreal and ireal[t + 1] == k))
    return ch, acept, esper


NMAX = 64
grupos = {}
for v in verdad:
    toks = v["prompt_ids"] + v["out_ids"]
    g = grupos.setdefault(v["nombre"].split("-")[0], [])
    for ln in por_pedido[v["nombre"]]:
        x = paso(ln, toks, NMAX)
        if x:
            g.append(x)


def rendimiento(acc_por_paso, tokens_verif_por_paso, B):
    return sum(a + 1 for a in acc_por_paso) / sum(FIJO / B + C * v for v in tokens_verif_por_paso)


print(f"modelo de costo: paso = {FIJO}/B + {C} ms por token verificado\n")
for nom, g in grupos.items():
    n = len(g); S = 8
    cad = [x[0] for x in g]
    print(f"== {nom}: {n} pasos | cadena K=8 acepta {1 + sum(cad) / n:.2f} por paso")
    print(f"   {'politica':24s} {'aceptados':>9s} {'verif.':>7s} | {'B=1':>7s} {'B=6':>7s}   (tok/s relativo a la cadena)")
    base = {B: rendimiento(cad, [S + 1] * n, B) for B in (1, 6)}
    for N in (8, 9, 12, 15, 20, 31):
        ar = [x[1][N - 1] for x in g]
        rel = {B: rendimiento(ar, [N + 1] * n, B) / base[B] for B in (1, 6)}
        nota = "  <- entra en 2 bloques de PN131" if N == 9 else ("  <- 3 bloques" if N == 15 else "")
        print(f"   arbol {N:2d} nodos{'':11s} {1 + sum(ar) / n:9.2f} {N + 1:7d} | {rel[1]:6.1%} {rel[6]:6.1%}{nota}")
    for B in (1, 6):
        ar, ver = [], []
        for ch, acept, esper in g:
            mejor = max(range(1, len(esper) + 1), key=lambda N: (1 + esper[N - 1]) / (FIJO / B + C * (N + 1)))
            ar.append(acept[mejor - 1]); ver.append(mejor + 1)
        print(f"   por costo (optimo B={B}){'':3s} {1 + sum(ar) / n:9.2f} {sum(ver) / n:7.1f} | "
              f"{'':7s} {rendimiento(ar, ver, B) / base[B]:6.1%}  (para B={B})" if B == 6 else
              f"   por costo (optimo B={B}){'':3s} {1 + sum(ar) / n:9.2f} {sum(ver) / n:7.1f} | {rendimiento(ar, ver, B) / base[B]:6.1%}")
    # validacion: el primer token del borrador contra la verdad
    print()
