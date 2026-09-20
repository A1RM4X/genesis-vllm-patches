# SPDX-License-Identifier: Apache-2.0
"""Los endpoints HTTP de Genesis no pueden quedar abiertos por el prefijo de la ruta.

El middleware de vLLM solo cubre /v1/*; aca se arma una app SIN ese middleware para probar
que la proteccion es del router de Genesis y no depende de el.
"""

from types import SimpleNamespace

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

CLAVE = "clave-de-inferencia"
ADMIN = "clave-de-admin"


def _app(monkeypatch, *, api_key=CLAVE, admin=None):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    if admin is None:
        monkeypatch.delenv("GENESIS_ADMIN_API_KEY", raising=False)
    else:
        monkeypatch.setenv("GENESIS_ADMIN_API_KEY", admin)
    from vllm._genesis import kv_offload_tracker as t
    app = FastAPI()
    app.state.args = SimpleNamespace(api_key=[api_key] if api_key else None)
    app.include_router(t.router)
    return TestClient(app), t


def _h(k):
    return {"Authorization": f"Bearer {k}"}


def test_lectura_exige_la_clave(monkeypatch):
    c, _ = _app(monkeypatch)
    assert c.get("/v1/kv-offload/requests").status_code == 401
    assert c.get("/v1/kv-offload/requests", headers=_h("otra")).status_code == 401
    assert c.get("/v1/kv-offload/requests", headers={"Authorization": CLAVE}).status_code == 401
    assert c.get("/v1/kv-offload/requests", headers=_h(CLAVE)).status_code == 200
    assert c.get("/v1/genesis/pid").status_code == 401


@pytest.mark.parametrize("metodo,ruta", [
    ("get", "/kv-offload/requests"),
    ("post", "/kv-offload/reset"),
    ("post", "/reset_prefix_cache"),
])
def test_los_alias_sin_v1_ya_no_existen(monkeypatch, metodo, ruta):
    """Eran los que quedaban fuera del middleware de vLLM. Con clave y todo: 404/405."""
    c, _ = _app(monkeypatch)
    assert getattr(c, metodo)(ruta, headers=_h(CLAVE)).status_code in (404, 405)


def test_toda_ruta_del_router_esta_autenticada(monkeypatch):
    """Guardia para el futuro: una ruta nueva sin /v1 no puede nacer abierta."""
    c, t = _app(monkeypatch)
    for r in t.router.routes:
        for m in r.methods - {"HEAD", "OPTIONS"}:
            assert c.request(m, r.path).status_code == 401, (m, r.path)


def test_sin_clave_configurada_es_abierto_como_vllm(monkeypatch):
    c, _ = _app(monkeypatch, api_key=None)
    assert c.get("/v1/kv-offload/requests").status_code == 200


def test_la_clave_tambien_sale_del_entorno(monkeypatch):
    c, _ = _app(monkeypatch, api_key=None)
    monkeypatch.setenv("VLLM_API_KEY", CLAVE)
    assert c.get("/v1/kv-offload/requests").status_code == 401
    assert c.get("/v1/kv-offload/requests", headers=_h(CLAVE)).status_code == 200


def test_con_clave_admin_la_de_inferencia_no_muta(monkeypatch):
    c, _ = _app(monkeypatch, admin=ADMIN)
    # leer sigue andando con la clave normal...
    assert c.get("/v1/genesis/pid", headers=_h(CLAVE)).status_code == 200
    # ...pero mutar no.
    assert c.post("/v1/genesis/pid", headers=_h(CLAVE), json={}).status_code == 401
    assert c.post("/v1/kv-offload/reset", headers=_h(CLAVE)).status_code == 401


def test_la_clave_admin_lee_y_muta(monkeypatch, tmp_path):
    """Un cliente manda un solo Bearer: la admin tiene que pasar las dos dependencias."""
    c, _ = _app(monkeypatch, admin=ADMIN)
    from vllm._genesis import dynamic_pid_gating as g
    monkeypatch.setattr(g, "PID_CONTROL_FILE", str(tmp_path / "ctl.json"))
    assert c.get("/v1/genesis/pid", headers=_h(ADMIN)).status_code == 200
    r = c.post("/v1/genesis/pid", headers=_h(ADMIN), json={"short_prompt_tokens": 1500})
    assert r.status_code == 200, r.text


def test_sin_clave_admin_muta_la_normal(monkeypatch, tmp_path):
    c, t = _app(monkeypatch)
    from vllm._genesis import dynamic_pid_gating as g
    monkeypatch.setattr(g, "PID_CONTROL_FILE", str(tmp_path / "ctl.json"))
    r = c.post("/v1/genesis/pid", headers=_h(CLAVE), json={"short_prompt_tokens": 1500})
    assert r.status_code == 200, r.text


@pytest.mark.parametrize("body", [
    {"no_existe": 1},
    {"short_prompt_tokens": "muchos"},
    {"short_prompt_tokens": 1.5},
    {"short_prompt_tokens": -1},
    {"enabled": 1},
    {"max_concurrent_prefills": True},
    {"headroom_ratio": 5},
])
def test_el_body_de_pid_se_valida(monkeypatch, tmp_path, body):
    c, _ = _app(monkeypatch)
    from vllm._genesis import dynamic_pid_gating as g
    ctl = tmp_path / "ctl.json"
    monkeypatch.setattr(g, "PID_CONTROL_FILE", str(ctl))
    r = c.post("/v1/genesis/pid", headers=_h(CLAVE), json=body)
    assert r.status_code == 400, r.text
    assert not ctl.exists(), "una config invalida no puede llegar al archivo de control"
