# SPDX-License-Identifier: Apache-2.0
"""Hito 1 del arbol de borrador: el constructor, contra una referencia con heap y contra los
pasos reales volcados del servidor (si estan)."""

import glob
import heapq
import json
import math
import os

import pytest
import torch

from vllm._genesis import arbol_borrador as ab


def _ref(cand, sc, n):
    """Best-first con heap, en Python puro. Devuelve [(token, padre, prof, idx)]."""
    def lsm(v):
        m = max(v); z = m + math.log(sum(math.exp(x - m) for x in v)); return [x - z for x in v]
    S = len(cand)
    heap, nodos, cnt = [], [], 0
    for j, l in enumerate(lsm(sc[0][0])):
        heapq.heappush(heap, (-l, cnt, 0, j, -1)); cnt += 1
    while heap and len(nodos) < n:
        nlp, _, d, j, p = heapq.heappop(heap)
        nodos.append((cand[d][j], p, d, j)); yo = len(nodos) - 1
        if d + 1 < S:
            for k, l in enumerate(lsm(sc[d + 1][j])):
                heapq.heappush(heap, (nlp - l, cnt, d + 1, k, yo)); cnt += 1
    return nodos


def _azar(R=3, S=8, K=16, semilla=0):
    g = torch.Generator().manual_seed(semilla)
    cand = torch.stack([torch.stack([torch.randperm(5000, generator=g)[:K] for _ in range(S)]) for _ in range(R)])
    sc = torch.randn(R, S, K, K, generator=g) * 3
    return cand, sc


@pytest.mark.parametrize("n", [1, 4, 8, 15])
def test_igual_que_el_heap(n):
    cand, sc = _azar()
    tok, pad, prof, idx = ab.construir(cand, sc, n)
    for r in range(cand.shape[0]):
        ref = _ref(cand[r].tolist(), sc[r].tolist(), n)
        got = list(zip(tok[r].tolist(), pad[r].tolist(), prof[r].tolist(), idx[r].tolist()))
        assert got == ref


def test_orden_topologico_y_profundidades():
    cand, sc = _azar(semilla=1)
    tok, pad, prof, _ = ab.construir(cand, sc, 8)
    for r in range(cand.shape[0]):
        for n in range(8):
            p = int(pad[r, n])
            assert p < n                                   # el padre siempre antes
            assert int(prof[r, n]) == (0 if p < 0 else int(prof[r, p]) + 1)


def test_con_un_candidato_dominante_es_la_cadena_golosa():
    """Si en cada paso hay un candidato con casi toda la masa, el arbol de 8 ES el camino goloso."""
    cand, sc = _azar(R=2, semilla=2)
    sc = sc * 0.01
    sc[:, :, :, 3] += 30.0
    tok, pad, prof, idx = ab.construir(cand, sc, 8)
    assert pad.tolist() == [[-1, 0, 1, 2, 3, 4, 5, 6]] * 2
    assert (idx == 3).all()


def test_mascara_de_ancestros():
    padre = torch.tensor([[-1, 0, 0, 1, -1, 4, 3, 2]])
    m = ab.ancestros(padre)[0]
    assert m[3].tolist() == [True, True, False, True, False, False, False, False]
    assert m[6].tolist() == [True, True, False, True, False, False, True, False]
    assert m[5].tolist() == [False, False, False, False, True, True, False, False]
    assert m[7].tolist() == [True, False, True, False, False, False, False, True]


def test_aceptacion_golosa_baja_por_la_rama_correcta():
    token = torch.tensor([[10, 20, 11, 21, 30, 22, 12, 40]])
    padre = torch.tensor([[-1, 0, -1, 1, 3, 2, -1, 4]])
    # el target: tras el ancla quiere 10; tras el nodo 0 quiere 20; tras el 1 quiere 21; tras el 3
    # quiere 99 (no esta) -> camino [0, 1, 3], largo 3
    obj = torch.tensor([[20, 21, 0, 99, 0, 0, 0, 0]])
    cam, largo = ab.aceptar_goloso(token, padre, obj, torch.tensor([10]))
    assert largo.tolist() == [3] and cam[0, :3].tolist() == [0, 1, 3]


DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "tests", "bench", "medicion", "trazas", "ddtree")


@pytest.mark.skipif(not glob.glob(os.path.join(DIR, "dump_*.jsonl")), reason="sin el volcado real")
def test_sobre_pasos_reales_del_servidor():
    """En los pasos reales el arbol de 8 contiene SIEMPRE al primer token del camino goloso, y el
    camino goloso entero cuando el selector esta seguro."""
    vistos = 0
    for f in glob.glob(os.path.join(DIR, "dump_*.jsonl")):
        for k, linea in enumerate(open(f)):
            if k % 40:
                continue
            ln = json.loads(linea)
            if len(ln["pos"]) != 1:
                continue
            cand, sc = torch.tensor(ln["cand"]), torch.tensor(ln["sc"])
            tok, pad, prof, _ = ab.construir(cand, sc, 8)
            ref = _ref(ln["cand"][0], ln["sc"][0], 8)
            assert list(zip(tok[0].tolist(), pad[0].tolist(), prof[0].tolist())) == [(a, b, c) for a, b, c, _ in ref]
            assert int(tok[0, 0]) == ln["borrador"][0][0]          # la raiz del arbol = 1er token goloso
            vistos += 1
    assert vistos > 20


def test_orden_dfs_es_preorden_y_conserva_el_arbol():
    import torch
    from vllm._genesis import arbol_borrador as ab
    g = torch.Generator().manual_seed(7)
    for n in (4, 8, 9, 15):
        cand = torch.randint(0, 1000, (5, 8, 16), generator=g)
        sc = torch.randn(5, 8, 16, 16, generator=g).log_softmax(-1)
        tok, pad, prof, idx = ab.construir(cand, sc, n)
        t2, p2, d2, i2, perm = ab.orden_dfs(tok, pad, prof, idx)
        for r in range(5):
            viejo = {(int(tok[r, i]), int(prof[r, i]), int(tok[r, pad[r, i]]) if pad[r, i] >= 0 else -1)
                     for i in range(n)}
            nuevo = {(int(t2[r, i]), int(d2[r, i]), int(t2[r, p2[r, i]]) if p2[r, i] >= 0 else -1)
                     for i in range(n)}
            assert viejo == nuevo
            assert int(p2[r, 0]) == -1
            for i in range(1, n):
                assert -1 <= int(p2[r, i]) < i                     # topologico (-1 = el ancla)
                assert (int(p2[r, i]) == -1) == (int(d2[r, i]) == 0)
                # preorden: el padre es el anterior, o el anterior cerro una rama mas honda
                assert int(p2[r, i]) == i - 1 or int(d2[r, i]) <= int(d2[r, i - 1])


def test_bits_ancestros_de_la_cadena():
    import torch
    from vllm._genesis import arbol_borrador as ab
    pad = ab.padres_verificacion(torch.arange(-1, 7)[None])       # 8 nodos en cadena + ancla
    assert pad[0].tolist() == list(range(-1, 8))
    assert ab.bits_ancestros(pad)[0].tolist() == [(1 << t) - 1 for t in range(9)]
    assert ab.bits_ancestros(pad.to(torch.int32))[0].tolist() == [(1 << t) - 1 for t in range(9)]


def test_aceptar_en_gpu_igual_que_la_referencia():
    import torch
    from vllm._genesis import arbol_borrador as ab
    g = torch.Generator().manual_seed(21)
    for n in (4, 8, 15):
        R = 64
        cand = torch.randint(0, 6, (R, 8, 16), generator=g)       # vocabulario chico: que acepte
        cand = torch.stack([torch.stack([torch.randperm(40, generator=g)[:16] for _ in range(8)])
                            for _ in range(R)])                  # hermanos con tokens distintos
        sc = (torch.randn(R, 8, 16, 16, generator=g) * 3).log_softmax(-1)
        tok, pad, prof, idx, _ = ab.orden_dfs(*ab.construir(cand, sc, n))
        pv = ab.padres_verificacion(pad)
        tv = torch.cat([torch.zeros(R, 1, dtype=tok.dtype), tok], 1)
        # el target "quiere", despues de cada posicion, el token de alguno de sus hijos (o nada)
        muestra = torch.randint(0, 40, (R, n + 1), generator=g)
        for r in range(R):
            for t in range(n + 1):
                hijos = [c for c in range(1, n + 1) if int(pv[r, c]) == t]
                if hijos and float(torch.rand(1, generator=g)) < 0.7:
                    muestra[r, t] = tv[r, hijos[int(torch.randint(0, len(hijos), (1,), generator=g))]]
        cam, nacc, bono = ab.aceptar(tv, pv, muestra)
        cam_ref, largo_ref = ab.aceptar_goloso(tok, pad, muestra[:, 1:], muestra[:, 0])
        assert (nacc - 1).tolist() == largo_ref.tolist()
        assert int((nacc - 1).max()) >= 3                          # el test ejercita caminos largos
        for r in range(R):
            L = int(largo_ref[r])
            assert (cam[r, :L] - 1).tolist() == cam_ref[r, :L].tolist()
            assert cam[r, L:].eq(-1).all()
            ult = int(cam[r, L - 1]) if L else 0
            assert int(bono[r]) == int(muestra[r, ult])
        filas = ab.filas_de_cinta(cam, 8)
        for r in range(R):
            L = min(int(largo_ref[r]), 8)
            assert filas[r, :L].tolist() == cam_ref[r, :L].tolist()
