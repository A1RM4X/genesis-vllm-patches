# SPDX-License-Identifier: Apache-2.0
"""TDD para PN116 — super kernel de MLP aislado al prefill.

Lo que estos tests protegen es el UNICO motivo por el que PN116 existe
aparte de P113: que el super kernel se llame **solo** en el rango de M donde
gana. El docstring de P113 tiene la medicion: cutlass gana 1.55x a M=512 y
1.13x a M=1664, empatan a M=8000. Llamarlo por debajo del umbral es una
regresion, y es justo lo que pasaba con el prefill chunkeado a 2048.

No se ejercita el kernel (necesita GPU y el estado INT8 de PN110): se ejercita
la LOGICA DE DISPATCH, que es donde estaba el error.
"""
from __future__ import annotations

import importlib
import types

import pytest


@pytest.fixture()
def pn116(monkeypatch):
    monkeypatch.setenv("GENESIS_ENABLE_PN116_PREFILL_MLP_SK", "1")
    monkeypatch.setenv("GENESIS_PN116_M_MIN", "4096")
    monkeypatch.setenv("GENESIS_PN116_AB", "0")
    mod = importlib.import_module(
        "vllm._genesis.wiring.models.patch_PN116_prefill_mlp_superkernel")
    mod = importlib.reload(mod)
    yield mod
    importlib.reload(mod)


class FakeTensor:
    """Stand-in de un tensor: solo hace falta shape y reshape."""

    def __init__(self, m, k):
        self.shape = (m, k)

    def reshape(self, *dims):
        if dims == (-1, self.shape[-1]):
            return self
        return self


class FakeMLP:
    """MLP con el forward original instrumentado."""

    def __init__(self):
        self.expert_gate = None
        self.gate_up_proj = types.SimpleNamespace()
        self.original_calls = 0

    def original(self, x):
        self.original_calls += 1
        return "ORIGINAL"


class FakeAct:
    """Salida del camino fusionado: solo necesita reshape()."""

    def reshape(self, *dims):
        return self


def _install(mod, mlp):
    """Envuelve el forward y mockea el camino fusionado."""
    mod._fused_gate_up = lambda proj, x2d: FakeAct()
    mlp.down_proj = lambda act: ("FUSIONADO", None)
    return mod._make_forward(FakeMLP.original)


def test_decode_nunca_toca_el_super_kernel(pn116):
    """M = 4..40 es decode. Ahi el kernel pierde por goleada."""
    mlp = FakeMLP()
    fwd = _install(pn116, mlp)
    for m in (4, 8, 16, 40):
        assert fwd(mlp, FakeTensor(m, 5120)) == "ORIGINAL"
    assert pn116.STATS.dispatched == 0
    assert pn116.STATS.skipped_small_m == 4


def test_prefill_chunkeado_a_2048_no_dispatcha(pn116):
    """El caso que hunde a P113: M=2048 cae donde cutlass gana 1.13-1.55x."""
    mlp = FakeMLP()
    fwd = _install(pn116, mlp)
    assert fwd(mlp, FakeTensor(2048, 5120)) == "ORIGINAL"
    assert pn116.STATS.dispatched == 0


def test_prefill_grande_si_dispatcha(pn116):
    mlp = FakeMLP()
    fwd = _install(pn116, mlp)
    assert fwd(mlp, FakeTensor(8192, 5120)) == "FUSIONADO"
    assert pn116.STATS.dispatched == 1
    assert mlp.original_calls == 0


def test_el_umbral_es_configurable(monkeypatch):
    monkeypatch.setenv("GENESIS_ENABLE_PN116_PREFILL_MLP_SK", "1")
    monkeypatch.setenv("GENESIS_PN116_M_MIN", "1024")
    mod = importlib.reload(importlib.import_module(
        "vllm._genesis.wiring.models.patch_PN116_prefill_mlp_superkernel"))
    try:
        mlp = FakeMLP()
        fwd = _install(mod, mlp)
        assert fwd(mlp, FakeTensor(2048, 5120)) == "FUSIONADO"
    finally:
        importlib.reload(mod)


def test_sin_estado_int8_cae_al_original(pn116):
    """Sin PN110 no hay sk_perm_w: el forward original tiene que quedar intacto."""
    mlp = FakeMLP()
    pn116._fused_gate_up = lambda proj, x2d: None
    fwd = pn116._make_forward(FakeMLP.original)
    assert fwd(mlp, FakeTensor(8192, 5120)) == "ORIGINAL"
    assert pn116.STATS.skipped_no_state == 1
    assert pn116.STATS.dispatched == 0


def test_capa_moe_queda_intacta(pn116):
    """Con expert_gate el camino no aplica: es otra topologia."""
    mlp = FakeMLP()
    mlp.expert_gate = object()
    fwd = _install(pn116, mlp)
    assert fwd(mlp, FakeTensor(8192, 5120)) == "ORIGINAL"
    assert pn116.STATS.dispatched == 0


def test_aviso_cuando_el_chunk_hace_que_nunca_se_llame(pn116, monkeypatch):
    """El modo de falla silencioso de P113: kernel instalado que no corre nunca."""
    monkeypatch.setattr(
        "sys.argv",
        ["vllm", "serve", "--long-prefill-token-threshold", "2048"])
    w = pn116._chunk_config_warning()
    assert w is not None and "2048" in w and "4096" in w


def test_sin_aviso_si_el_chunk_alcanza(pn116, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["vllm", "serve", "--long-prefill-token-threshold", "8192"])
    assert pn116._chunk_config_warning() is None


def test_stats_expone_el_dispatch(pn116):
    mlp = FakeMLP()
    fwd = _install(pn116, mlp)
    fwd(mlp, FakeTensor(8192, 5120))
    fwd(mlp, FakeTensor(16, 5120))
    s = pn116.stats()
    assert s["dispatched"] == 1
    assert s["skipped_small_m"] == 1
    assert s["m_min"] == 4096


def test_esta_en_el_registry_y_es_opt_in():
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    meta = PATCH_REGISTRY["PN116"]
    assert meta["default_on"] is False
    assert meta["env_flag"] == "GENESIS_ENABLE_PN116_PREFILL_MLP_SK"
