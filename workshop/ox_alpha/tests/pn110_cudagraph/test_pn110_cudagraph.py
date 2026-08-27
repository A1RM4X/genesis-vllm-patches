#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""test_pn110_cudagraph.py — verifica PN110 + superkernels + cudagraphs en un contenedor vLLM.

Se ejecuta DESPUES de levantar el contenedor (ver run_ab.sh). Hace poll de
``/health`` hasta 200, envía una completion de prueba, y escanea los logs del
contenedor buscando el error de captura de CUDA graphs que rompe el despacho de
super kernels:

    "Assigning / modifying buffers of nn.Module during forward pass is not
     allowed when using cudagraph inside the compiler"

Ese es el síntoma que el usuario atribuye al truco del epílogo con stride 0
(``_zero(a.device)`` alojando un tensor y mutando un dict global en la primera
llamada dentro del forward compilado). El fix bajo prueba es precomputar
``state["sk_zero"]`` en carga (as_strided con strides (0,0)), de modo que el
forward caliente solo lea el tensor y no mute nada.

Criterio de éxito (exit 0):
    * ``/health`` responde 200 dentro de ``--timeout``.
    * Una completion de chat devuelve contenido (status 200 + campo "content").
    * Cero patrones CRÍTICOS en los logs del contenedor.

Criterio de fallo (exit 1):
    * ``/health`` no llega a 200 (timeout de captura de cudagraphs / crash).
    * La completion falla.
    * Aparece cualquier patrón CRÍTICO (error de captura, Traceback, OOM, etc.).

Autor: ox-alpha workshop — genesis-vllm-patches
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time

import requests

HEALTH_URL_TPL = "http://{host}:{port}/health"
CHAT_URL_TPL = "http://{host}:{port}/v1/chat/completions"

# Patrones CRÍTICOS: cualquiera presente => el test falla. Son los síntomas de
# que la captura de CUDA graphs se rompió (el bug de PN110 + superkernels) o de
# un crash de arranque que impide servir tráfico.
CRITICAL_PATTERNS = [
    "Assigning / modifying buffers of nn.Module during forward pass",
    "CUDA graph capture failed",
    "cudagraph capture",
    "Engine core initialization failed",
    "Traceback (most recent call last)",
    "RuntimeError",
    "OutOfMemoryError",
    "CUDA out of memory",
    "AssertionError",
]


def _docker_logs(container: str, tail: int = 3000) -> str:
    """Devuelve stdout+stderr del contenedor; vacío si no se puede leer."""
    try:
        proc = subprocess.run(
            ["docker", "logs", "--tail", str(tail), container],
            capture_output=True, text=True, timeout=60,
        )
        return (proc.stdout or "") + "\n" + (proc.stderr or "")
    except Exception as exc:  # noqa: BLE001 - el test no debe morir por logs
        return f"<no se pudieron leer logs: {exc}>"


def _wait_health(url: str, timeout: float) -> tuple[bool, float]:
    """Hace poll de /health hasta 200. Devuelve (ok, segundos_transcurridos)."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                return True, time.monotonic() - t0
        except Exception:  # noqa: BLE001 - reintenta hasta el timeout
            pass
        time.sleep(3)
    return False, timeout


def _send_chat(url: str, api_key: str, model: str, timeout: int) -> tuple[int | None, str]:
    """POST a /v1/chat/completions con un prompt trivial. Devuelve (status, body)."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [{"role": "user",
                      "content": "Responde con una sola palabra: ¿cuál es la "
                                 "capital de Francia?"}],
        "max_tokens": 24,
        "temperature": 0.0,
        "stream": False,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        return resp.status_code, resp.text
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--container", default="genesis-pn110-cg")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8390)
    ap.add_argument("--sk", default="?", help="valor de GENESIS_PN110_SK (solo para el reporte)")
    ap.add_argument("--model", default="qwen3.8")
    ap.add_argument("--api-key", default="<REDACTADO: clave rotada 2026-09-19>")
    ap.add_argument("--timeout", type=int, default=480, help="segundos para /health")
    ap.add_argument("--chat-timeout", type=int, default=150)
    args = ap.parse_args()

    health_url = HEALTH_URL_TPL.format(host=args.host, port=args.port)
    chat_url = CHAT_URL_TPL.format(host=args.host, port=args.port)

    print(f"[test] SK={args.sk} container={args.container} health={health_url}", flush=True)

    ok, elapsed = _wait_health(health_url, args.timeout)
    if not ok:
        print(f"[test] FAIL: /health no respondió 200 en {args.timeout}s", flush=True)
        logs = _docker_logs(args.container)
        print("=== LOG TAIL (health timeout) ===", flush=True)
        print("\n".join(logs.splitlines()[-60:]), flush=True)
        crit = [p for p in CRITICAL_PATTERNS if p.lower() in logs.lower()]
        if crit:
            print("[test] ERRORES CRÍTICOS en logs:", crit, flush=True)
        sys.exit(1)
    print(f"[test] /health OK en {elapsed:.1f}s", flush=True)

    status, body = _send_chat(chat_url, args.api_key, args.model, args.chat_timeout)
    completion_ok = bool(status == 200 and body and '"content"' in body)
    print(f"[test] chat status={status} completion_ok={completion_ok}", flush=True)
    if not completion_ok:
        print(f"[test] body: {body[:600]}", flush=True)

    logs = _docker_logs(args.container)
    low = logs.lower()
    crit_hits = [p for p in CRITICAL_PATTERNS if p.lower() in low]
    err_count = len(re.findall(r"\berror\b", low))

    print("=== SCAN DE LOGS ===", flush=True)
    print(f"[test] ocurrencias de 'error' (info): {err_count}", flush=True)
    if crit_hits:
        print("[test] ERRORES CRÍTICOS:", crit_hits, flush=True)
        for pat in crit_hits:
            m = re.search(re.escape(pat), logs, re.IGNORECASE)
            if m:
                s = max(0, m.start() - 220)
                e = min(len(logs), m.end() + 220)
                print("  --- " + logs[s:e].replace("\n", "\n      "), flush=True)
                break
    else:
        print("[test] sin errores críticos en logs", flush=True)

    passed = completion_ok and not crit_hits
    print(f"[test] VERDICT SK={args.sk}: {'PASS' if passed else 'FAIL'}", flush=True)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
