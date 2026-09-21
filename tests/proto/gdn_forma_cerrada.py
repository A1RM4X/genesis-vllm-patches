"""Forma CERRADA del gated delta rule sobre un arbol, contra la recurrencia secuencial.

Hoy `gdn_cinta._k_spec_arbol` recorre los nodos y, en cada salto de rama, REHACE el estado de los
ancestros con actualizaciones de rango 1 (ver [[arbol-codigo-6-pedidos-cuesta-el-paso]]: ~6% del
paso con 6 pedidos). Bole (arXiv 2608.01651) y TreeWY (arXiv 2608.20961) muestran que no hace
falta: la salida de TODOS los nodos sale de un sistema triangular.

Derivacion (la recurrencia es la del kernel):
    S_t = S_{pa(t)} * exp(g_t) ;  d_t = (v_t - S_t^- k_t) * beta_t ;  S_t = S_t^- + d_t k_t^T
Con P_t = exp(suma de g_j sobre el camino raiz->t) y i <= t = "i es t o ancestro de t":
    S_t   = P_t S_0 + sum_{i <= t} (P_t/P_i) d_i k_i^T
    d_t   = beta_t [ v_t - P_t S_0 k_t - sum_{i < t} (P_t/P_i) (k_i·k_t) d_i ]      (i ESTRICTO)
    o_t   = P_t (S_0 q_t) + sum_{i <= t} (P_t/P_i) (k_i·q_t) d_i
La del medio es (I + diag(beta) G) D = R con G[t,i] = (P_t/P_i)(k_i·k_t) solo si i < t, que es
estrictamente triangular inferior en PREORDEN (el padre siempre antes que el hijo): se resuelve por
sustitucion hacia adelante, sin tocar S. Para T=9 la matriz es 9x9.

Costo: el secuencial toca S (V x K) cuatro veces por token (decay, S k, rango 1, S q) y otra vez
por cada ancestro rehecho; la forma cerrada NO toca S: solo dos productos S_0·k_t y S_0·q_t por
token, mas terminos T^2 que son despreciables.
"""
import torch

torch.manual_seed(0)
V, K, T = 128, 128, 9
DT = torch.float64                      # fp64: se valida la MATEMATICA, no el redondeo


def datos(padres, escala_g=0.3):
    S0 = torch.randn(V, K, dtype=DT) * 0.05
    k = torch.nn.functional.normalize(torch.randn(T, K, dtype=DT), dim=-1)   # L2, como el kernel
    q = torch.nn.functional.normalize(torch.randn(T, K, dtype=DT), dim=-1) * K ** -0.5
    v = torch.randn(T, V, dtype=DT)
    g = -torch.rand(T, dtype=DT) * escala_g                                  # g <= 0
    beta = torch.sigmoid(torch.randn(T, dtype=DT))
    return S0, k, q, v, g, beta


def secuencial(padres, S0, k, q, v, g, beta):
    """La referencia: el estado de cada nodo sale del de SU padre."""
    est, o = {}, torch.zeros(T, V, dtype=DT)
    for t in range(T):
        base = S0 if padres[t] < 0 else est[padres[t]]
        S = base * torch.exp(g[t])
        d = (v[t] - S @ k[t]) * beta[t]
        S = S + torch.outer(d, k[t])
        est[t] = S
        o[t] = S @ q[t]
    return o, est


def cerrada(padres, S0, k, q, v, g, beta):
    """Todos los nodos a la vez: un sistema triangular de T x T."""
    # P[t] y la mascara de ancestros ESTRICTOS
    P = torch.zeros(T, dtype=DT)
    anc = torch.zeros(T, T, dtype=torch.bool)
    for t in range(T):
        P[t] = (P[padres[t]] if padres[t] >= 0 else 0.0) + g[t]              # en log
        j = padres[t]
        while j >= 0:
            anc[t, j] = True
            j = padres[j]
    rel = torch.exp(P[:, None] - P[None, :])                                 # P_t / P_i
    G = torch.where(anc, rel * (k @ k.T), torch.zeros((), dtype=DT))         # G[t,i], i < t
    R = beta[:, None] * (v - torch.exp(P)[:, None] * (k @ S0.T))             # [T, V]
    A = torch.eye(T, dtype=DT) + beta[:, None] * G
    D = torch.linalg.solve_triangular(A, R, upper=False)                     # sustitucion adelante
    C = torch.where(anc | torch.eye(T, dtype=torch.bool), rel * (q @ k.T), torch.zeros((), dtype=DT))
    o = torch.exp(P)[:, None] * (q @ S0.T) + C @ D
    return o, P, D


def estado_aceptado(a, padres, S0, k, P, D):
    """El estado del nodo aceptado, para el paso siguiente (lo que hoy escribe el kernel)."""
    S = torch.exp(P[a]) * S0
    j = a
    while j >= 0:
        S = S + torch.exp(P[a] - P[j]) * torch.outer(D[j], k[j])
        j = padres[j]
    return S


ARBOLES = {
    "cadena":            [-1, 0, 1, 2, 3, 4, 5, 6, 7],
    "tipico (3 ramas)":  [-1, 0, 1, 2, 3, 2, 1, 6, 0],
    "estrella":          [-1, 0, 0, 0, 0, 0, 0, 0, 0],
    "dos cadenas":       [-1, 0, 1, 2, 3, 0, 5, 6, 7],
    "profundo y ancho":  [-1, 0, 1, 1, 3, 3, 5, 0, 7],
}
print(f"{'arbol':20s} {'err salida':>12s} {'err estado aceptado (peor nodo)':>34s}")
for nombre, pad in ARBOLES.items():
    S0, k, q, v, g, beta = datos(pad)
    o_ref, est = secuencial(pad, S0, k, q, v, g, beta)
    o_cer, P, D = cerrada(pad, S0, k, q, v, g, beta)
    e_o = ((o_cer - o_ref).norm() / o_ref.norm()).item()
    e_s = max(((estado_aceptado(a, pad, S0, k, P, D) - est[a]).norm() / est[a].norm()).item()
              for a in range(T))
    print(f"{nombre:20s} {e_o:12.2e} {e_s:34.2e}")

# g mas agresiva (decay fuerte) y beta cerca de 1: el caso numericamente dificil
print()
for esc in (1.0, 3.0, 8.0):
    pad = ARBOLES["tipico (3 ramas)"]
    S0, k, q, v, g, beta = datos(pad, escala_g=esc)
    o_ref, _ = secuencial(pad, S0, k, q, v, g, beta)
    o_cer, _, _ = cerrada(pad, S0, k, q, v, g, beta)
    print(f"decay g ~ -{esc}: err salida {((o_cer - o_ref).norm() / o_ref.norm()).item():.2e}")


# ───────────────────────── int8: donde SI se puede ─────────────────────────
# El grueso del kernel son los productos S_0·k_t y S_0·q_t: [V,K] x [K] por cada uno de los T
# tokens, o sea 2T pasadas sobre el estado. En la forma cerrada S_0 NO se toca durante el arbol
# (esa es toda la gracia), asi que se puede cuantizar UNA vez y usar 2T veces -> mma.s8, 4x el
# throughput de fp32 en sm_86. Y como el estado que se escribe al slot se reconstruye desde el
# S_0 original, el error NO se acumula entre pasos: queda acotado a este paso.
# k y q ya vienen normalizados L2 (|x| <= 1), que es el caso ideal para int8.
def q8_filas(X):
    """int8 por fila (por v): una escala cada K valores."""
    s = X.abs().amax(-1, keepdim=True).clamp(min=1e-30) / 127.0
    return torch.round(X / s).clamp(-127, 127), s


def cerrada_int8(padres, S0, k, q, v, g, beta, solo_kq=False, solo_S=False, ancla_fp=False):
    T_ = T
    P = torch.zeros(T_, dtype=DT); anc = torch.zeros(T_, T_, dtype=torch.bool)
    for t in range(T_):
        P[t] = (P[padres[t]] if padres[t] >= 0 else 0.0) + g[t]
        j = padres[t]
        while j >= 0:
            anc[t, j] = True; j = padres[j]
    rel = torch.exp(P[:, None] - P[None, :])
    k8, sk = q8_filas(k); q8, sq = q8_filas(q)
    if solo_S:      # el prep (T x T) queda en fp: es barato y es el que amplifica el error, porque
        KK = k @ k.T                                  # k_i·k_t son vectores casi ortogonales
        QK = q @ k.T
    else:
        KK = (k8 @ k8.T) * sk * sk.T
        QK = (q8 @ k8.T) * sq * sk.T
    G = torch.where(anc, rel * KK, torch.zeros((), dtype=DT))
    if solo_kq:                                        # solo KK/QK en int8, S_0 en float
        Sk = k @ S0.T; Sq = q @ S0.T
    else:
        S8, ss = q8_filas(S0)                          # UNA vez, y se usa 2T veces
        Sk = (k8.to(DT) @ S8.T.to(DT)) * sk * ss.T     # [T, V]
        Sq = (q8.to(DT) @ S8.T.to(DT)) * sq * ss.T
        if ancla_fp:   # el token 0 es el unico cuyo d_t entra en el estado que se escribe al slot
            Sk[0] = k[0] @ S0.T
            Sq[0] = q[0] @ S0.T
    R = beta[:, None] * (v - torch.exp(P)[:, None] * Sk)
    A = torch.eye(T_, dtype=DT) + beta[:, None] * G
    D = torch.linalg.solve_triangular(A, R, upper=False)
    C = torch.where(anc | torch.eye(T_, dtype=torch.bool), rel * QK, torch.zeros((), dtype=DT))
    return torch.exp(P)[:, None] * Sq + C @ D


print("\nint8 en los productos (el estado se cuantiza UNA vez por paso, no se acumula):")
print(f"  {'arbol':20s} {'fp (hoy)':>10s} {'int8 k,q':>10s} {'int8 k,q,S_0':>13s}")
for nombre, pad in ARBOLES.items():
    S0, k, q, v, g, beta = datos(pad)
    o_ref, _ = secuencial(pad, S0, k, q, v, g, beta)
    e = lambda o: ((o - o_ref).norm() / o_ref.norm()).item()
    o_f, _, _ = cerrada(pad, S0, k, q, v, g, beta)
    print(f"  {nombre:20s} {e(o_f):10.2e} {e(cerrada_int8(pad, S0, k, q, v, g, beta, True)):10.2e}"
          f" {e(cerrada_int8(pad, S0, k, q, v, g, beta)):13.2e}")

print("\n  con OUTLIERS en el estado (x30 en el 1% de las filas), que es el caso que rompe int8:")
for nombre in ("cadena", "tipico (3 ramas)"):
    pad = ARBOLES[nombre]
    S0, k, q, v, g, beta = datos(pad)
    filas = torch.randperm(V)[: max(1, V // 100)]
    S0[filas] *= 30.0
    o_ref, _ = secuencial(pad, S0, k, q, v, g, beta)
    e = lambda o: ((o - o_ref).norm() / o_ref.norm()).item()
    o_f, _, _ = cerrada(pad, S0, k, q, v, g, beta)
    print(f"  {nombre:20s} {e(o_f):10.2e} {e(cerrada_int8(pad, S0, k, q, v, g, beta, True)):10.2e}"
          f" {e(cerrada_int8(pad, S0, k, q, v, g, beta)):13.2e}")


print("\nreparto: el prep (T x T) en fp y solo los productos con S_0 en int8")
print(f"  {'arbol':20s} {'todo int8':>10s} {'solo S_0 int8':>14s} {'+ ancla en fp':>14s}")
for nombre, pad in ARBOLES.items():
    S0, k, q, v, g, beta = datos(pad)
    o_ref, est = secuencial(pad, S0, k, q, v, g, beta)
    e = lambda o: ((o - o_ref).norm() / o_ref.norm()).item()
    print(f"  {nombre:20s} {e(cerrada_int8(pad, S0, k, q, v, g, beta)):10.2e}"
          f" {e(cerrada_int8(pad, S0, k, q, v, g, beta, solo_S=True)):14.2e}"
          f" {e(cerrada_int8(pad, S0, k, q, v, g, beta, solo_S=True, ancla_fp=True)):14.2e}")

# lo que de verdad importa: el ESTADO que queda escrito, que es lo unico que se propaga al paso
# siguiente (las salidas de los nodos rechazados se tiran, y el camino aceptado se rehace desde la
# cinta en fp)
print("\n  error del ESTADO tras el ancla (lo unico que se propaga):")
for nombre in ("cadena", "tipico (3 ramas)"):
    pad = ARBOLES[nombre]
    S0, k, q, v, g, beta = datos(pad)
    _, est = secuencial(pad, S0, k, q, v, g, beta)
    for etq, kw in (("todo int8", {}), ("solo S_0", {"solo_S": True}),
                    ("solo S_0 + ancla fp", {"solo_S": True, "ancla_fp": True})):
        P = torch.zeros(T, dtype=DT)
        for t in range(T):
            P[t] = (P[pad[t]] if pad[t] >= 0 else 0.0) + g[t]
        k8, sk = q8_filas(k); S8, ss = q8_filas(S0)
        Sk0 = (k[0] @ S0.T) if kw.get("ancla_fp") else ((k8[0].to(DT) @ S8.T.to(DT)) * sk[0] * ss[:, 0])
        d0 = beta[0] * (v[0] - torch.exp(P[0]) * Sk0)
        S_est = torch.exp(P[0]) * S0 + torch.outer(d0, k[0])
        print(f"  {nombre:20s} {etq:22s} {((S_est - est[0]).norm() / est[0].norm()).item():.2e}")


# ───────────────── rotacion Hadamard antes de cuantizar ─────────────────
# Con H ortogonal y simetrica (Hadamard normalizada, H·H = I):  S·k = (S H)·(H k)
# o sea que rotar el estado por la derecha y la key por la izquierda deja el producto INTACTO, pero
# reparte los outliers entre las K componentes, que es lo que estira la escala de int8. Es la misma
# receta que PN126/PN131 usan en la atencion. Cuesta: con FWHT, V·K·log2(K) por el estado (una vez
# por paso) y K·log2(K) por cada k/q.
def hadamard(n, dtype=DT):
    H = torch.ones(1, 1, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / n ** 0.5


H = hadamard(K)


def prod_int8(S0, x, rotar):
    """S0 @ x^T con int8 por fila, con o sin rotacion. Devuelve [T, V]."""
    S, xx = (S0 @ H, x @ H) if rotar else (S0, x)
    S8, ss = q8_filas(S); x8, sx = q8_filas(xx)
    return (x8.to(DT) @ S8.T.to(DT)) * sx * ss.T


print("\\nrotacion Hadamard antes de cuantizar (el producto S_0·k, que es el caro):")
print(f"  {'caso':34s} {'int8 directo':>13s} {'int8 rotado':>12s}")
for etq, mult in (("estado normal", 1.0), ("outliers x30 en 1% de filas", 30.0),
                  ("outliers x100 en 1% de columnas", 100.0)):
    S0, k, q, v, g, beta = datos(ARBOLES["cadena"])
    if mult > 1 and "filas" in etq:
        S0[torch.randperm(V)[: max(1, V // 100)]] *= mult
    elif mult > 1:
        S0[:, torch.randperm(K)[: max(1, K // 100)]] *= mult
    ref = k @ S0.T
    e = lambda P: ((P - ref).norm() / ref.norm()).item()
    print(f"  {etq:34s} {e(prod_int8(S0, k, False)):13.2e} {e(prod_int8(S0, k, True)):12.2e}")

print("\\n  cresta (max/mediana de |S_0| por fila), que es lo que decide si rotar sirve:")
for etq, mult, eje in (("normal", 1.0, None), ("outliers en columnas", 100.0, "col")):
    S0, *_ = datos(ARBOLES["cadena"])
    if eje:
        S0[:, torch.randperm(K)[: max(1, K // 100)]] *= mult
    for nom, X in (("sin rotar", S0), ("rotado", S0 @ H)):
        c = (X.abs().amax(-1) / X.abs().median(-1).values).median().item()
        print(f"  {etq:22s} {nom:10s} cresta {c:6.1f}")
