# SPDX-License-Identifier: Apache-2.0
"""PN81 (namespace): todo lo que cambia el significado de los bytes de la KV tiene que cambiar la huella."""

import importlib
import sys

import pytest

BASE_ARGV = ["vllm", "serve", "m", "--kv-cache-dtype", "int8_per_token_head",
             "--speculative-config", '{"method":"dflash","model":"/x","num_speculative_tokens":8}']
BASE_ENV = {"GENESIS_ENABLE_PN131_SK18": "1", "GENESIS_ENABLE_PN126_ROT_QK": "1",
            "GENESIS_PN126_ROT": "hadamard", "GENESIS_ENABLE_PN122_GDN_CINTA": "1",
            "GENESIS_PN144_ESCALA_RESIDUAL": "32"}


def _huella(monkeypatch, env=None, argv=None):
    for k in list(__import__("os").environ):
        if k.startswith("GENESIS_"):
            monkeypatch.delenv(k)
    for k, v in {**BASE_ENV, **(env or {})}.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    monkeypatch.setattr(sys, "argv", argv or BASE_ARGV)
    mod = importlib.import_module("vllm._genesis.kv_formato")
    return mod.huella()


def test_es_estable(monkeypatch):
    assert _huella(monkeypatch) == _huella(monkeypatch)


@pytest.mark.parametrize("env", [
    {"GENESIS_ENABLE_PN131_SK18": "0"},            # otro kernel escribe, otras escalas
    {"GENESIS_PN131_ROT": "off"},                  # k sin rotar en las capas de PN131
    {"GENESIS_PN126_DFLASH": "0"},                 # k del borrador sin rotar
    {"GENESIS_PN126_ROT": "wush:/x"},              # otra rotacion
    {"GENESIS_ENABLE_PN122_GDN_CINTA": "0"},       # otras columnas de estado GDN
    {"GENESIS_PN144_ESCALA_RESIDUAL": "16"},       # otro residual del borrador
    {"GENESIS_PN131_VENTANA": "0"},                # cualquier PN131_* que no sea solo runtime
])
def test_cada_variable_de_formato_cambia_la_huella(monkeypatch, env):
    assert _huella(monkeypatch, env) != _huella(monkeypatch)


@pytest.mark.parametrize("argv", [
    BASE_ARGV[:4] + ["fp8"] + BASE_ARGV[5:],
    BASE_ARGV[:6] + ['{"method":"dflash","model":"/x","num_speculative_tokens":8,"kv_cache_dtype":"auto"}'],
])
def test_el_dtype_de_la_kv_cambia_la_huella(monkeypatch, argv):
    assert _huella(monkeypatch, argv=argv) != _huella(monkeypatch)


@pytest.mark.parametrize("env", [
    {"GENESIS_PN131_BMAX": "4"}, {"GENESIS_PN131_DIAG": "x"}, {"GENESIS_PN131_MAXTOK": "10"},
    {"GENESIS_ENABLE_PN139_LM_HEAD_INT4": "1"},    # el lm_head no toca la KV
])
def test_lo_que_no_cambia_el_formato_no_cambia_la_huella(monkeypatch, env):
    assert _huella(monkeypatch, env) == _huella(monkeypatch)


def test_K_no_cambia_la_huella(monkeypatch):
    """num_speculative_tokens no cambia lo que se guarda: subir K no debe tirar la cache."""
    otro = BASE_ARGV[:6] + ['{"method":"dflash","model":"/x","num_speculative_tokens":9}']
    assert _huella(monkeypatch, argv=otro) == _huella(monkeypatch)
