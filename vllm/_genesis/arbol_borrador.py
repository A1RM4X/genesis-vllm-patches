# SPDX-License-Identifier: Apache-2.0
"""Arbol de borrador para DFlash2 — hito 1: el CONSTRUCTOR (todavia sin cablear a nada).

Por que un arbol, y por que de 8 nodos
--------------------------------------
DFlash2 propone, por cada una de sus 8 posiciones, 16 candidatos y una tabla de puntajes de
transicion ``scores[paso, candidato previo, candidato]``; hoy se recorre UN camino, goloso. La
aceptacion es una cadena: el primer fallo tira todo lo que sigue, asi que las ultimas posiciones
del camino casi nunca se aprovechan (en prosa aceptan < 10%).

Medido en este rig (2026-09-21): el paso de decode cuesta ~18,8 ms + 3,8 ms por pedido extra +
0,42 ms por POSICION, y con 6 pedidos esta parado justo en el codo de ~54 tokens por lote. O sea
que no se puede verificar mas tokens ni mas profundidad sin pagarlo. Lo que si sale gratis es
gastar MEJOR los mismos 8: reemplazar la cola improbable del camino por ramas cerca de la raiz.
Simulado sobre 2.080 pasos reales: aceptacion +17% en prosa, +2% en codigo.

Que hace
--------
``construir`` elige los N nodos de mayor probabilidad de camino (best-first, como DDTree,
arXiv 2604.12989, pero con puntajes condicionados al padre, que DFlash2 ya da). Salida por nodo:
token, padre (-1 = la raiz/ancla), profundidad e indice de candidato. Los nodos salen en orden
topologico (un padre siempre antes que sus hijos), que es lo que necesitan la mascara de
ancestros, el GDN y la compactacion de la KV.

Todo en la GPU y sin sincronizar: son operaciones de tensores de forma fija, capturables en el
CUDA graph del borrador.
"""

from __future__ import annotations

import torch


def construir(cand: torch.Tensor, sc: torch.Tensor, n_nodos: int):
    """cand [R, S, K] int64, sc [R, S, K, K] float (logits de transicion; la fila 0 del paso 0
    es la del ancla). Devuelve (token, padre, prof, idx), todos [R, n_nodos].

    Best-first exacto sin heap: la frontera cabe en un tensor de forma fija. Cada nodo elegido
    agrega a la frontera a sus K hijos; se hacen ``n_nodos`` rondas de argmax.
    """
    R, S, K = cand.shape
    dev = cand.device
    lp = torch.log_softmax(sc.float(), dim=-1)                       # [R, S, K(prev), K]
    F = K * (n_nodos + 1)                                            # tope de la frontera
    f_lp = torch.full((R, F), float("-inf"), device=dev)
    f_padre = torch.full((R, F), -1, dtype=torch.int64, device=dev)  # nodo padre (-1 = ancla)
    f_prof = torch.zeros((R, F), dtype=torch.int64, device=dev)      # profundidad 0-based
    f_idx = torch.zeros((R, F), dtype=torch.int64, device=dev)       # indice de candidato
    ar = torch.arange(K, device=dev)
    f_lp[:, :K] = lp[:, 0, 0, :]
    f_idx[:, :K] = ar
    token = torch.zeros((R, n_nodos), dtype=torch.int64, device=dev)
    padre = torch.full((R, n_nodos), -1, dtype=torch.int64, device=dev)
    prof = torch.zeros((R, n_nodos), dtype=torch.int64, device=dev)
    idx = torch.zeros((R, n_nodos), dtype=torch.int64, device=dev)
    filas = torch.arange(R, device=dev)
    for n in range(n_nodos):
        j = f_lp.argmax(dim=1)                                       # [R]
        p, d, i = f_lp[filas, j], f_prof[filas, j], f_idx[filas, j]
        token[:, n] = cand[filas, d, i]
        padre[:, n] = f_padre[filas, j]
        prof[:, n] = d
        idx[:, n] = i
        # sale de la frontera (scatter_ y no f_lp[filas, j] = -inf: la asignacion indexada con un
        # escalar de Python arma un tensor en CPU y lo copia, y eso no entra en un grafo CUDA)
        f_lp.scatter_(1, j[:, None], float("-inf"))
        # hijos: solo si no es el ultimo nivel
        d1 = (d + 1).clamp(max=S - 1)
        hijos = p[:, None] + lp[filas, d1, i, :]                     # [R, K]
        hijos = torch.where((d + 1 < S)[:, None], hijos, torch.full_like(hijos, float("-inf")))
        base = K * (n + 1)
        f_lp[:, base: base + K] = hijos
        f_padre[:, base: base + K] = n
        f_prof[:, base: base + K] = (d + 1)[:, None]
        f_idx[:, base: base + K] = ar
    return token, padre, prof, idx


def ancestros(padre: torch.Tensor) -> torch.Tensor:
    """Mascara de ancestros [R, N, N] bool: m[r, a, b] = b es ancestro de a, o a == b."""
    padre = padre.long()
    R, N = padre.shape
    m = torch.eye(N, dtype=torch.bool, device=padre.device)[None].repeat(R, 1, 1)
    cur = padre.clone()
    for _ in range(N):
        ok = cur >= 0
        m |= torch.nn.functional.one_hot(cur.clamp(min=0), N).bool() & ok[..., None]
        cur = torch.where(ok, padre.gather(1, cur.clamp(min=0)), cur)
    return m


def orden_dfs(token, padre, prof, idx):
    """Reordena los nodos en preorden DFS (sigue siendo topologico), todo en GPU y forma fija.

    Para que sirve: el GDN en arbol (``gdn_cinta``, ARBOL) sigue con el estado corriente
    cuando el padre de un nodo es el nodo anterior, y solo recalcula desde la raiz en los
    saltos de rama. En preorden cada rama es una corrida contigua, asi que los saltos son
    los minimos. La atencion no depende del orden (usa la mascara de ancestros).

    La clave de orden es el camino raiz->nodo escrito en base n+1 (el prefijo de un padre,
    rellenado con ceros, es menor que el de sus hijos): 17^8 entra holgado en int64.
    """
    R, n = padre.shape
    D = int(min(n, 8))
    ar = torch.arange(n, device=padre.device)[None].expand(R, n)
    # camino[d] = ancestro de profundidad d (o -1): se sube D veces desde el nodo
    # (la columna D+1 es un descarte: ahi escriben los que ya pasaron del ancla)
    cam = torch.full((R, n, D + 2), -1, dtype=torch.long, device=padre.device)
    cur = ar.clone()
    for _ in range(D + 1):
        ok = cur >= 0
        pr = prof.gather(1, cur.clamp(min=0)).long().clamp(max=D)
        pr = torch.where(ok, pr, pr.new_full((), D + 1))
        cam.scatter_(2, pr[..., None], cur[..., None])
        cur = torch.where(ok, padre.gather(1, cur.clamp(min=0)).long(), cur)
    cam = cam[..., : D + 1]
    pot = (n + 1) ** torch.arange(D, -1, -1, device=padre.device, dtype=torch.long)
    clave = ((cam + 1) * pot).sum(-1)
    perm = clave.argsort(dim=1, stable=True)                  # posicion nueva -> nodo viejo
    inv = torch.empty_like(perm).scatter_(1, perm, ar)        # nodo viejo -> posicion nueva
    p_v = padre.gather(1, perm).long()
    p_n = torch.where(p_v >= 0, inv.gather(1, p_v.clamp(min=0)), p_v).to(padre.dtype)
    return token.gather(1, perm), p_n, prof.gather(1, perm), idx.gather(1, perm), perm


def padres_verificacion(padre: torch.Tensor) -> torch.Tensor:
    """Padres en la secuencia que se VERIFICA: [ancla] + nodos. El ancla es el token 0 (padre
    -1) y cada nodo pasa a la posicion i+1, con padre ``padre+1`` (los hijos del ancla, 0)."""
    raiz = padre.new_full((padre.shape[0], 1), -1)
    return torch.cat([raiz, padre + 1], dim=1)


def bits_ancestros(padre: torch.Tensor) -> torch.Tensor:
    """``padre`` = padres de la secuencia verificada (``padres_verificacion``).
    Mascara por token en el formato de los kernels (PN131 y GDN): int32 [R, N], con el
    bit j-1 = "el nodo j es ancestro o el mismo", para j >= 1. La raiz (nodo 0) es ancestro
    de todos y no lleva bit. La cadena da (1 << t) - 1."""
    m = ancestros(padre)[:, :, 1:]
    w = (1 << torch.arange(m.shape[-1], device=padre.device, dtype=torch.int64))
    return (m.long() * w).sum(-1).to(torch.int32)


_k_arbol = None


def construir_dfs_kernel(cand, sc, out_tok, out_padre_v, out_prof_v, n_nodos: int,
                         out_bits=None) -> None:
    """``construir`` + ``orden_dfs`` + ``padres_verificacion`` en UN kernel, un programa por pedido.

    Por que: en torch son ~200 kernels chicos por paso adentro del grafo del borrador, y el arbol
    solo paga si el paso no se encarece. Mismo resultado que el camino de torch (mismo desempate:
    el primer maximo; claves de preorden unicas).

    ``cand`` [R, S, K] int64, ``sc`` [R, S, K, K] float contiguos. Escribe ``out_tok[r, :n]`` (tokens
    en preorden), ``out_padre_v[r, 1:n+1]`` (padre en la secuencia verificada; 0 = el ancla) y
    ``out_prof_v[r, 1:n+1]`` (profundidad, el ancla es 0). La columna 0 no se toca.
    ``out_bits`` [R, n+1] int32 (opcional): los bits de ancestros de ``bits_ancestros``, ya listos
    para los kernels, asi el runner no los recalcula en cada paso.
    """
    global _k_arbol
    from vllm.triton_utils import tl, triton
    if _k_arbol is None:
        @triton.jit
        def _k(cand, sc, out_tok, s_tok, out_pad, s_pad, out_prof, s_prof, out_bits, s_bits,
               BITS: tl.constexpr,
               S: tl.constexpr, K: tl.constexpr, N: tl.constexpr, BF: tl.constexpr,
               BN: tl.constexpr, BKK: tl.constexpr):
            r = tl.program_id(0)
            af = tl.arange(0, BF)
            an = tl.arange(0, BN)
            ak = tl.arange(0, BKK)
            mkk = ak < K
            NEG = float("-inf")
            # frontera: los K hijos del ancla
            fila0 = tl.load(sc + (r * S * K + 0) * K + ak, mask=mkk, other=NEG).to(tl.float32)
            mx = tl.max(fila0, 0)
            lse0 = mx + tl.log(tl.sum(tl.where(mkk, tl.exp(fila0 - mx), 0.0), 0))
            k0 = tl.where(af < K, af, 0)
            f_lp = tl.where(af < K, tl.load(sc + (r * S * K) * K + k0, mask=af < K, other=NEG).to(tl.float32) - lse0, NEG)
            f_pad = tl.full((BF,), -1, tl.int64)
            f_prof = tl.zeros((BF,), tl.int64)
            f_idx = tl.where(af < K, af, 0).to(tl.int64)
            n_tok = tl.zeros((BN,), tl.int64)
            n_pad = tl.full((BN,), -1, tl.int64)
            n_prof = tl.zeros((BN,), tl.int64)
            n_clave = tl.zeros((BN,), tl.int64)
            for n in range(0, N):
                j = tl.argmax(f_lp, 0)
                sel = af == j
                p = tl.sum(tl.where(sel, f_lp, 0.0), 0)
                d = tl.sum(tl.where(sel, f_prof, 0), 0)
                i = tl.sum(tl.where(sel, f_idx, 0), 0)
                pa = tl.sum(tl.where(sel, f_pad, 0), 0)
                tok = tl.load(cand + (r * S + d) * K + i)
                # clave de preorden: el camino en base N+1, el digito de la profundidad d
                pot = tl.full((), 1, tl.int64)
                for _ in range(0, S - 1 - d):
                    pot = pot * (N + 1)
                cpad = tl.sum(tl.where(an == pa, n_clave, 0), 0)      # pa = -1 no coincide: 0
                clave = cpad + (n + 1) * pot
                aqui = an == n
                n_tok = tl.where(aqui, tok, n_tok)
                n_pad = tl.where(aqui, pa, n_pad)
                n_prof = tl.where(aqui, d, n_prof)
                n_clave = tl.where(aqui, clave, n_clave)
                f_lp = tl.where(sel, NEG, f_lp)
                # los K hijos del elegido entran en [K*(n+1), K*(n+2))
                base = K * (n + 1)
                kid = af - base
                mh = (kid >= 0) & (kid < K)
                if d + 1 < S:
                    fb = ((r * S + d + 1) * K + i) * K
                    fila = tl.load(sc + fb + ak, mask=mkk, other=NEG).to(tl.float32)
                    mx2 = tl.max(fila, 0)
                    lse = mx2 + tl.log(tl.sum(tl.where(mkk, tl.exp(fila - mx2), 0.0), 0))
                    h = tl.load(sc + fb + tl.where(mh, kid, 0), mask=mh, other=NEG).to(tl.float32)
                    f_lp = tl.where(mh, p + h - lse, f_lp)
                f_pad = tl.where(mh, n, f_pad)
                f_prof = tl.where(mh, d + 1, f_prof)
                f_idx = tl.where(mh, kid, f_idx).to(tl.int64)
            # preorden: la posicion de un nodo es cuantos tienen clave menor
            val = an < N
            menor = (n_clave[None, :] < n_clave[:, None]) & val[None, :]
            rango = tl.sum(menor.to(tl.int64), 1)                     # [BN]: nodo -> posicion
            rpad = tl.sum(tl.where(an[None, :] == n_pad[:, None], rango[None, :] + 1, 0), 1)  # ancla -> 0
            tl.store(out_tok + r * s_tok + rango, n_tok.to(out_tok.dtype.element_ty), mask=val)
            tl.store(out_pad + r * s_pad + 1 + rango, rpad.to(out_pad.dtype.element_ty), mask=val)
            tl.store(out_prof + r * s_prof + 1 + rango, (n_prof + 1).to(out_prof.dtype.element_ty), mask=val)
            if BITS:
                # bits[nodo] = bits[padre] | (1 << posicion): el padre siempre se eligio antes
                n_bits = tl.zeros((BN,), tl.int64)
                for n in range(0, N):
                    pa = tl.sum(tl.where(an == n, n_pad, 0), 0)
                    rg = tl.sum(tl.where(an == n, rango, 0), 0)
                    bp = tl.sum(tl.where(an == pa, n_bits, 0), 0)
                    one = tl.full((), 1, tl.int64)
                    n_bits = tl.where(an == n, bp | (one << rg), n_bits)
                tl.store(out_bits + r * s_bits + 1 + rango, n_bits.to(out_bits.dtype.element_ty), mask=val)
        _k_arbol = _k
    R, S, K = cand.shape
    assert cand.is_contiguous() and sc.is_contiguous() and n_nodos <= S
    _k_arbol[(R,)](cand, sc, out_tok, out_tok.stride(0), out_padre_v, out_padre_v.stride(0),
                   out_prof_v, out_prof_v.stride(0),
                   out_bits if out_bits is not None else out_prof_v,
                   out_bits.stride(0) if out_bits is not None else 0, BITS=out_bits is not None,
                   S=S, K=K, N=n_nodos,
                   BF=triton.next_power_of_2(K * (n_nodos + 1)), BN=triton.next_power_of_2(n_nodos),
                   BKK=triton.next_power_of_2(K), num_warps=1)


def aceptar(token_v: torch.Tensor, padre_v: torch.Tensor, muestra: torch.Tensor):
    """Aceptacion por recorrido del arbol, en GPU, forma fija y sin sincronizar.

    Todo en indices de la secuencia VERIFICADA ([ancla] + nodos): ``token_v`` [R, T] (el del
    ancla no se mira), ``padre_v`` [R, T] (``padres_verificacion``), ``muestra`` [R, T] = el
    token que el target saca DESPUES de cada posicion: el argmax con decodificacion golosa, una
    muestra de su distribucion con temperatura.

    Por que es exacto tambien con temperatura: el borrador propone tokens, no distribuciones
    (q es one-hot). Muestrear y ~ p en cada nodo y bajar al hijo cuyo token coincide con y es
    el rejection sampling multi-borrador de SpecInfer en ese caso limite: se acepta algun hijo
    con probabilidad sum p(hijos), lo mismo que rechazar de a uno y renormalizar, y si no
    coincide ninguno, y YA es una muestra del residuo. Cada token emitido es una muestra
    exacta del target; el arbol solo decide cuantas se aprovechan. Las T muestras salen del
    mismo forward, en paralelo: no hay lazo sobre datos.

    Devuelve ``(camino [R, T-1], nacc [R], bono [R])``: posiciones verificadas aceptadas en
    orden (relleno -1), aceptados contando el ancla (= largo + 1, lo que vLLM llama
    ``num_accepted_tokens``) y el token que sigue al ultimo aceptado.
    """
    R, T = token_v.shape
    dev = token_v.device
    act = torch.zeros(R, dtype=torch.long, device=dev)               # posicion actual: el ancla
    vivo = torch.ones(R, dtype=torch.bool, device=dev)
    camino = torch.full((R, T - 1), -1, dtype=torch.long, device=dev)
    nacc = torch.ones(R, dtype=torch.long, device=dev)
    pv = padre_v.long()
    for d in range(T - 1):
        quiero = muestra.gather(1, act[:, None])                     # [R, 1]
        ok = (pv == act[:, None]) & (token_v == quiero)              # [R, T]
        ok[:, 0] = False
        hay = ok.any(dim=1) & vivo
        sig = ok.float().argmax(dim=1)
        camino[:, d] = torch.where(hay, sig, camino[:, d])
        act = torch.where(hay, sig, act)
        nacc += hay.long()
        vivo = hay
    return camino, nacc, muestra.gather(1, act[:, None])[:, 0]


_k_acep = None


def aceptar_kernel(tok_f, muestra_f, cu, padre_v, alcanzable, T: int, tm: int):
    """``aceptar`` + armado de la salida en UN kernel, sobre las filas PLANAS del rejection sampler
    (sin densificar). ``tok_f``/``muestra_f`` [n] (token de entrada y muestra del target por fila de
    logits), ``cu`` [R+1], ``padre_v`` [R, T] y ``alcanzable`` [R, T] bool (nodos que se pueden
    aceptar). Devuelve ``(sampled [R, T] int64, nacc [R] int32, camino [R, T-1] int64 con -1,
    filas [R, tm] int32 para la cinta)``."""
    global _k_acep
    from vllm.triton_utils import tl, triton
    if _k_acep is None:
        @triton.jit
        def _k(tok_f, mu_f, cu, pv, s_pv, alc, s_alc, sampled, nacc, camino, filas,
               T: tl.constexpr, TM: tl.constexpr, BT: tl.constexpr):
            r = tl.program_id(0)
            at = tl.arange(0, BT)
            ini = tl.load(cu + r).to(tl.int64)
            nf = tl.load(cu + r + 1).to(tl.int64) - ini
            mt = (at < nf) & (at < T)
            tk = tl.load(tok_f + ini + at, mask=mt, other=-1).to(tl.int64)
            mu = tl.load(mu_f + ini + at, mask=mt, other=-2).to(tl.int64)
            pa = tl.load(pv + r * s_pv + at, mask=at < T, other=-2).to(tl.int64)
            ok0 = tl.load(alc + r * s_alc + at, mask=at < T, other=0) != 0
            ok0 = ok0 & mt & (at > 0)
            act = tl.zeros((), tl.int64)
            vivo = tl.full((), 1, tl.int64)
            n = tl.zeros((), tl.int64)
            for d in range(0, T - 1):
                quiero = tl.sum(tl.where(at == act, mu, 0), 0)
                cand = ok0 & (pa == act) & (tk == quiero)
                hay = (tl.sum(cand.to(tl.int64), 0) > 0) & (vivo != 0)
                sig = tl.argmax(cand.to(tl.int64), 0).to(tl.int64)
                if hay:
                    tl.store(sampled + r * T + d, quiero)
                    tl.store(camino + r * (T - 1) + d, sig)
                    if d < TM:
                        tl.store(filas + r * TM + d, (sig - 1).to(tl.int32))
                    act = sig
                    n += 1
                else:
                    vivo = vivo * 0
            tl.store(sampled + r * T + n, tl.sum(tl.where(at == act, mu, 0), 0))
            tl.store(nacc + r, (n + 1).to(tl.int32))
        _k_acep = _k
    R = cu.shape[0] - 1
    dev = tok_f.device
    sampled = torch.zeros((R, T), dtype=torch.int64, device=dev)
    nacc = torch.ones(R, dtype=torch.int32, device=dev)
    camino = torch.full((R, T - 1), -1, dtype=torch.int64, device=dev)
    filas = torch.arange(tm, dtype=torch.int32, device=dev)[None].repeat(R, 1)
    _k_acep[(R,)](tok_f, muestra_f, cu, padre_v, padre_v.stride(0), alcanzable, alcanzable.stride(0),
                  sampled, nacc, camino, filas, T=T, TM=tm, BT=triton.next_power_of_2(T), num_warps=1)
    return sampled, nacc, camino, filas


def filas_de_cinta(camino: torch.Tensor, tm: int) -> torch.Tensor:
    """``camino`` de ``aceptar`` -> filas de cinta para ``gdn_cinta.camino_gpu()`` y
    ``arbol_conv.compactar`` (fila = posicion verificada - 1). El relleno queda en la
    identidad, que nadie lee: solo se usan las primeras ``nacc - 1``."""
    ident = torch.arange(tm, device=camino.device)[None].expand(camino.shape[0], tm)
    n = min(tm, camino.shape[1])
    c = camino[:, :n]
    ident = ident.clone()
    ident[:, :n] = torch.where(c > 0, c - 1, ident[:, :n])
    return ident.to(torch.int32)


def aceptar_goloso(token, padre, objetivo_por_nodo, objetivo_raiz):
    """Recorrido del arbol con decodificacion golosa. ``objetivo_por_nodo[r, n]`` es el token que
    el target elige DESPUES del nodo n; ``objetivo_raiz[r]`` el que elige despues del ancla.
    Devuelve (camino [R, N] con -1 de relleno, largo [R]). Referencia en CPU para los tests."""
    R, N = token.shape
    caminos, largos = [], []
    for r in range(R):
        cam, act, quiero = [], -1, int(objetivo_raiz[r])
        while True:
            sig = [n for n in range(N) if int(padre[r, n]) == act and int(token[r, n]) == quiero]
            if not sig:
                break
            act = sig[0]; cam.append(act); quiero = int(objetivo_por_nodo[r, act])
        caminos.append(cam + [-1] * (N - len(cam))); largos.append(len(cam))
    return torch.tensor(caminos), torch.tensor(largos)


__all__ = ["construir", "construir_dfs_kernel", "ancestros", "aceptar", "aceptar_kernel", "filas_de_cinta", "aceptar_goloso", "orden_dfs", "padres_verificacion",
           "bits_ancestros"]
