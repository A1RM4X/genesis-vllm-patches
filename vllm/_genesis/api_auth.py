# SPDX-License-Identifier: Apache-2.0
"""Autenticacion de los endpoints HTTP que agrega Genesis.

Por que existe
--------------
El ``AuthenticationMiddleware`` de vLLM solo protege las rutas que empiezan con ``/v1``. Todo
lo demas (``/health``, ``/metrics``...) queda abierto a proposito. Genesis montaba alias sin
ese prefijo (``/kv-offload/reset``, ``/reset_prefix_cache``, ``/kv-offload/requests``) y
quedaban abiertos sin que nadie lo hubiera decidido: cualquiera que llegara al puerto podia
leer el historial de requests o vaciar las tres capas de cache y cortar los pedidos en curso.
Verificado el 2026-09-20 contra el servidor: ``GET /kv-offload/requests`` sin clave daba 200.

La proteccion va como dependencia del ROUTER, no ruta por ruta: una ruta nueva nace
autenticada tenga el prefijo que tenga.

Dos niveles
-----------
``require_api_key``    la misma clave que el resto de la API (``--api-key`` / ``VLLM_API_KEY``).
                       Sin clave configurada el servidor es abierto, igual que vLLM.
``require_admin_key``  para las rutas que MUTAN el servidor (reset de caches, config de
                       PN115). Con ``GENESIS_ADMIN_API_KEY`` definida exigen ESA clave, asi un
                       cliente de inferencia no puede reconfigurar ni vaciar el servidor. Sin
                       ella, valen las claves normales.
"""

from __future__ import annotations

import hashlib
import os
import secrets

from fastapi import HTTPException, Request


def _hash(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


def _api_tokens(request: Request) -> list[str]:
    """Misma fuente y misma precedencia que ``serve/middleware/register.py`` de vLLM."""
    args = getattr(request.app.state, "args", None)
    cli = getattr(args, "api_key", None)
    if isinstance(cli, str):
        cli = [cli]
    return [k for k in (cli or [os.environ.get("VLLM_API_KEY")]) if k]


def _bearer(request: Request) -> str | None:
    scheme, _, param = (request.headers.get("Authorization") or "").partition(" ")
    return param if scheme.lower() == "bearer" and param else None


def _coincide(presentado: str | None, validos: list[str]) -> bool:
    if presentado is None:
        return False
    h = _hash(presentado)
    ok = False
    for t in validos:      # sin cortocircuito: tiempo constante, como vLLM
        ok |= secrets.compare_digest(h, _hash(t))
    return ok


def _rechazar() -> HTTPException:
    return HTTPException(status_code=401, detail={"error": "Unauthorized"},
                         headers={"WWW-Authenticate": "Bearer"})


def _admin_token() -> str:
    return os.environ.get("GENESIS_ADMIN_API_KEY", "").strip()


async def require_api_key(request: Request) -> None:
    validos = _api_tokens(request)
    if not validos:
        return
    # La clave de admin tambien lee: un cliente manda UN solo Bearer, y las rutas que mutan
    # pasan por las dos dependencias (la del router y la propia).
    admin = _admin_token()
    if not _coincide(_bearer(request), validos + ([admin] if admin else [])):
        raise _rechazar()


async def require_admin_key(request: Request) -> None:
    admin = _admin_token()
    validos = [admin] if admin else _api_tokens(request)
    if validos and not _coincide(_bearer(request), validos):
        raise _rechazar()


__all__ = ["require_api_key", "require_admin_key"]
