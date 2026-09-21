# SPDX-License-Identifier: Apache-2.0
"""P66: con GENESIS_P66_GENERAR_TALLAS=1 las tallas salen de K, no de una lista escrita a mano."""

import logging
import textwrap
from types import SimpleNamespace

import pytest

from vllm._genesis.wiring.spec_decode import patch_66_cudagraph_size_divisibility_filter as p66


def _correr(K, lista, max_num_seqs=10, max_num_tokens=8192):
    """Ejecuta el TEXTO que P66 inyecta en config/vllm.py, con un `self` de mentira."""
    cuerpo = "def f(self, cudagraph_capture_sizes, max_num_tokens, logger):\n" \
             + textwrap.indent(textwrap.dedent(p66.P66_NEW), "    ") \
             + "\n    return cudagraph_capture_sizes\n"
    ns: dict = {}
    exec(compile(cuerpo, "p66_inyectado", "exec"), ns)
    yo = SimpleNamespace(speculative_config=SimpleNamespace(num_speculative_tokens=K),
                         scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs))
    return ns["f"](yo, list(lista), max_num_tokens, logging.getLogger("p66-test"))


LISTA_DEL_COMPOSE = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]


def test_sin_la_variable_filtra_como_siempre(monkeypatch):
    monkeypatch.delenv("GENESIS_P66_GENERAR_TALLAS", raising=False)
    assert _correr(8, LISTA_DEL_COMPOSE) == [9, 36]
    assert _correr(10, LISTA_DEL_COMPOSE) == [11]          # el caso que rompia K=10


@pytest.mark.parametrize("K", [3, 8, 9, 10, 11, 12, 15])
def test_generadas_cubren_cualquier_K_hasta_max_num_seqs(monkeypatch, K):
    monkeypatch.setenv("GENESIS_P66_GENERAR_TALLAS", "1")
    monkeypatch.delenv("GENESIS_P66_MAX_REQS", raising=False)
    assert _correr(K, LISTA_DEL_COMPOSE) == [(K + 1) * n for n in range(1, 11)]


def test_max_reqs_acota_y_max_num_tokens_tambien(monkeypatch):
    monkeypatch.setenv("GENESIS_P66_GENERAR_TALLAS", "1")
    monkeypatch.setenv("GENESIS_P66_MAX_REQS", "4")
    assert _correr(8, LISTA_DEL_COMPOSE) == [9, 18, 27, 36]
    monkeypatch.delenv("GENESIS_P66_MAX_REQS")
    assert _correr(8, LISTA_DEL_COMPOSE, max_num_tokens=30) == [9, 18, 27]


def test_sin_spec_decode_no_toca_nada(monkeypatch):
    monkeypatch.setenv("GENESIS_P66_GENERAR_TALLAS", "1")
    assert _correr(0, LISTA_DEL_COMPOSE) == LISTA_DEL_COMPOSE
