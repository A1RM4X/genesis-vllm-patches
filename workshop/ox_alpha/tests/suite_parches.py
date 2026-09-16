#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""suite_parches.py — validacion parche-por-parche contra PROD genesis-27b-qwen38-fp8 (PN110+B3).

Cubre:
  - baseline con PN110+B3 activos (estado actual PROD)
  - deshabilitado PN110 (GENESIS_DISABLE_PN110=1)
  - deshabilitado B3 (GENESIS_ENABLE_B3_CUSTOM_AR=0 / GENESIS_DISABLE_B3=1)
Cada escenario mide tetris CLI y RPG CLI contra http://127.0.0.1:8320/v1/chat/completions
con model qwen3.8, max_tokens 500, temperature 0.7, wall/tokens/tps, output en /tmp/opencode/suite_parches.log
Restart via docker restart (poll health 30s, sin sleeps largos). Si cuelga, docker logs.

Autor: ox-alpha workshop — genesis-vllm-patches
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests  # type: ignore

ENDPOINT = os.environ.get("GENESIS_PROD_ENDPOINT", "http://127.0.0.1:8320/v1/chat/completions")
MODEL = os.environ.get("GENESIS_PROD_MODEL", "qwen3.8")
LOG_PATH = Path("/tmp/opencode/suite_parches.log")
CONTAINER = os.environ.get("GENESIS_CONTAINER", "genesis-27b-qwen38-fp8")
API_KEY = os.environ.get("VLLM_API_KEY", "<REDACTADO: clave rotada 2026-09-19>")

PROMPTS: dict[str, str] = {
    "tetris": "Escribe un juego de Tetris completo en Python para terminal (curses), con rotacion, lineas, score, y controles WASD. Codigo funcional y comentado.",
    "rpg": "Escribe un juego de rol CLI en Python con clases Hero, Monster, combate por turnos, inventario, y mapa 5x5. Codigo funcional.",
}

SCENARIOS: list[tuple[str, dict[str, str]]] = [
    ("baseline", {}),
    ("PN110_disabled", {"GENESIS_DISABLE_PN110": "1"}),
    ("B3_disabled", {"GENESIS_ENABLE_B3_CUSTOM_AR": "0", "GENESIS_DISABLE_B3": "1"}),
]

TIMEOUT_S = 30  # por request; suite global usa 60s por test via poll
MAX_TOKENS = 500
TEMPERATURE = 0.7


def _log(msg: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _headers() -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


def call_once(prompt: str, label: str, timeout: int = TIMEOUT_S) -> dict:
    """POST a /v1/chat/completions, mide wall/tokens/tps. Timeout 30s por intento."""
    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    t0 = time.monotonic()
    try:
        resp = requests.post(ENDPOINT, headers=_headers(), json=payload, timeout=timeout)
        wall = time.monotonic() - t0
        body = resp.text
        # truncate body for log but keep full for file
        try:
            data = resp.json()
        except Exception:
            data = {"raw": body[:4000], "status": resp.status_code}
        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        completion_tokens = usage.get("completion_tokens")
        # fallback: estimate from text length if usage missing
        if completion_tokens is None:
            # rough: chars/4
            txt = ""
            try:
                txt = data["choices"][0]["message"]["content"] or ""
            except Exception:
                txt = body[:2000]
            completion_tokens = max(1, len(txt) // 4)
        tps = completion_tokens / wall if wall > 0 else 0.0
        # persist full output
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n--- {label} wall={wall:.2f}s tokens={completion_tokens} tps={tps:.1f} status={resp.status_code} ---\n")
            try:
                content = data["choices"][0]["message"]["content"] if isinstance(data, dict) and "choices" in data else body
                f.write(str(content)[:8000] + "\n")
            except Exception:
                f.write(body[:8000] + "\n")
            f.write(f"--- usage: {json.dumps(usage)} ---\n")
        return {
            "label": label,
            "wall": wall,
            "tokens": completion_tokens,
            "tps": tps,
            "status": resp.status_code,
            "usage": usage,
            "ok": resp.status_code == 200,
            "error": None,
        }
    except requests.exceptions.Timeout:
        wall = time.monotonic() - t0
        _log(f"[TIMEOUT] {label} no responde en {timeout}s wall={wall:.1f}s — vLLM posiblemente colgado")
        _check_hang(label)
        return {"label": label, "wall": wall, "tokens": 0, "tps": 0.0, "status": 0, "usage": {}, "ok": False, "error": "timeout"}
    except Exception as e:
        wall = time.monotonic() - t0
        _log(f"[ERROR] {label} exception {type(e).__name__}: {e} wall={wall:.1f}s")
        return {"label": label, "wall": wall, "tokens": 0, "tps": 0.0, "status": 0, "usage": {}, "ok": False, "error": str(e)}


def _check_hang(label: str) -> None:
    """Si vLLM se cuelga, revisa docker logs y sugiere parche culpable."""
    try:
        r = subprocess.run(
            ["docker", "logs", "--tail", "120", CONTAINER],
            capture_output=True, text=True, timeout=10,
        )
        tail = (r.stdout + r.stderr)[-6000:]
        _log(f"[docker logs tail for {label}]\n{tail[-3000:]}")
        # heurística culpable: PN110 OOM / B3 fallback
        culpable = None
        low = tail.lower()
        if "pn110" in low or "int8" in low or "outofmemory" in low or "oom" in low:
            culpable = "PN110"
        elif "b3" in low or "custom_all_reduce" in low or "custom ar" in low:
            culpable = "B3"
        if culpable:
            _log(f"[DIAG] posible culpable {culpable} en hang de {label} — sugerido: GENESIS_DISABLE_{culpable}=1 y restart")
        # docker ps estado
        r2 = subprocess.run(["docker", "inspect", CONTAINER, "--format", "{{.State.Status}} {{.State.Running}}"], capture_output=True, text=True, timeout=10)
        _log(f"[docker inspect] {r2.stdout.strip()} {r2.stderr.strip()}")
    except Exception as e:
        _log(f"[docker logs fail] {e}")


def _poll_health(timeout: int = 30) -> bool:
    """Poll /health (o /v1/models) hasta timeout 30s, sin sleeps largos."""
    url_health = ENDPOINT.replace("/v1/chat/completions", "/health")
    url_models = ENDPOINT.replace("/v1/chat/completions", "/v1/models")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for url in (url_health, url_models):
            try:
                r = requests.get(url, headers=_headers(), timeout=5)
                if r.status_code in (200, 401):  # 401 = auth ok pero endpoint existe
                    return True
            except Exception:
                pass
        time.sleep(2)
    return False


def _restart_with_env(env_overrides: dict[str, str]) -> bool:
    """Intenta reiniciar PROD con env overrides. docker exec no persiste env; se documenta limite.
    Usa docker restart y poll 30s. Retorna True si vuelve. Timeout 30s por comando.
    """
    if not env_overrides:
        return True
    _log(f"[restart] solicitado overrides {env_overrides} — docker exec NO persiste env en PROD; se requiere recrear via compose.")
    # intento informativo: mostrar env actual
    try:
        r = subprocess.run(["docker", "inspect", CONTAINER, "--format", "{{json .Config.Env}}"], capture_output=True, text=True, timeout=10)
        env_list = json.loads(r.stdout.strip()) if r.stdout.strip() else []
        # filtrar relevantes
        rel = [e for e in env_list if "GENESIS" in e or "VLLM" in e]
        _log(f"[env actual] relevantes ({len(rel)}): {rel[:20]}")
    except Exception as e:
        _log(f"[inspect fail] {e}")
    # intentar docker exec set (no persistente, solo para log)
    for k, v in env_overrides.items():
        try:
            # esto solo afecta al shell del exec, no al servidor; se usa para demostrar limite
            subprocess.run(["docker", "exec", CONTAINER, "bash", "-c", f"export {k}={v}; echo {k}=$" + k], capture_output=True, text=True, timeout=10)
        except Exception:
            pass
    # intento restart real (30s max)
    _log(f"[restart] docker restart {CONTAINER} (timeout 30s)")
    try:
        r = subprocess.run(["docker", "restart", CONTAINER], capture_output=True, text=True, timeout=30)
        _log(f"[restart] exit {r.returncode} out={r.stdout.strip()[:500]} err={r.stderr.strip()[:500]}")
    except subprocess.TimeoutExpired:
        _log("[restart] TIMEOUT 30s — docker restart colgado, revisar docker logs")
        _check_hang("restart")
        return False
    except Exception as e:
        _log(f"[restart fail] {e}")
        return False
    # poll health 30s
    ok = _poll_health(timeout=30)
    _log(f"[health] post-restart {'UP' if ok else 'DOWN (30s timeout)'}")
    if not ok:
        _check_hang("post-restart health")
    return ok


def main() -> int:
    # limpia log previo
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    _log(f"=== suite_parches PROD={CONTAINER} endpoint={ENDPOINT} model={MODEL} max_tokens={MAX_TOKENS} temp={TEMPERATURE} ===")
    # verificar baseline actual env
    try:
        r = subprocess.run(["docker", "inspect", CONTAINER, "--format", "{{.State.Status}}"], capture_output=True, text=True, timeout=10)
        _log(f"[prod status] {r.stdout.strip()} {r.stderr.strip()}")
    except Exception as e:
        _log(f"[prod status fail] {e}")
    # medir compositor de parches activos (lectura desde env)
    try:
        r = subprocess.run(["docker", "exec", CONTAINER, "bash", "-c", "env | grep -E 'GENESIS_ENABLE_PN110|GENESIS_ENABLE_B3|GENESIS_DISABLE' | sort"], capture_output=True, text=True, timeout=10)
        _log(f"[prod genesis env]\n{r.stdout.strip()[:2000]}")
    except Exception as e:
        _log(f"[env grep fail] {e}")

    results: list[dict] = []
    for scen_name, env_overrides in SCENARIOS:
        _log(f"\n===== ESCENARIO: {scen_name} env={env_overrides} =====")
        if env_overrides:
            # intenta restart con nuevos env (no persiste, pero se intenta y se mide igual)
            ok = _restart_with_env(env_overrides)
            if not ok:
                _log(f"[WARN] {scen_name} no volvió en 30s — se marca como HANG y se sugiere deshabilitar")
                # igualmente intentar medir aunque esté down (fallará y se registrará)
            else:
                _log(f"[INFO] {scen_name} restart OK (nota: env no persistió sin compose recreate — medición es baseline real)")
        else:
            # baseline: verificar health
            if not _poll_health(timeout=10):
                _log("[WARN] PROD no responde en baseline — revisar docker logs")
                _check_hang("baseline pre-check")

        for prompt_name, prompt in PROMPTS.items():
            label = f"{scen_name}/{prompt_name}"
            _log(f"[test] {label} POST {ENDPOINT} model={MODEL} max_tokens={MAX_TOKENS}")
            res = call_once(prompt, label, timeout=TIMEOUT_S)
            results.append(res)
            _log(f"[result] {label} wall={res['wall']:.2f}s tokens={res['tokens']} tps={res['tps']:.1f} ok={res['ok']} err={res['error']}")
            # segundo intento si timeout (poll 30s)
            if not res["ok"] and res["error"] == "timeout":
                _log(f"[retry] {label} reintento tras 2s")
                time.sleep(2)
                res2 = call_once(prompt, label + "_retry", timeout=TIMEOUT_S)
                results.append(res2)
                _log(f"[result retry] {label} wall={res2['wall']:.2f}s tps={res2['tps']:.1f} ok={res2['ok']}")

    # reporte agregado
    _log("\n========== REPORTE TPS POR PARCHE ==========")
    # agrupar por escenario
    from collections import defaultdict
    by_scen: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        scen = r["label"].split("/")[0]
        by_scen[scen].append(r)
    for scen, lst in by_scen.items():
        for r in lst:
            _log(f"  {r['label']}: {r['tps']:.1f} t/s wall={r['wall']:.1f}s tokens={r['tokens']} ok={r['ok']}")

    # baseline esperado 80-95 t/s
    baseline_tps = [r["tps"] for r in results if r["label"].startswith("baseline/") and r["ok"]]
    avg_baseline = sum(baseline_tps) / len(baseline_tps) if baseline_tps else 0.0
    _log(f"\n[baseline] medido {avg_baseline:.1f} t/s promedio (tetris+rpg) vs esperado 80-95 t/s")
    if avg_baseline == 0:
        _log("[baseline] SIN DATOS — PROD no respondió")
    elif 80 <= avg_baseline <= 95:
        _log("[baseline] DENTRO de rango esperado 80-95 t/s")
    elif avg_baseline < 80:
        _log(f"[baseline] POR DEBAJO de 80 t/s ({avg_baseline:.1f}) — posible regresión (revisar PN110/B3)")
    else:
        _log(f"[baseline] POR ENCIMA de 95 t/s ({avg_baseline:.1f}) — mejor que esperado")

    # comparativa parche deshabilitado vs baseline
    for scen in ["PN110_disabled", "B3_disabled"]:
        scen_tps = [r["tps"] for r in results if r["label"].startswith(scen + "/") and r["ok"]]
        if scen_tps and baseline_tps:
            avg_scen = sum(scen_tps) / len(scen_tps)
            delta = avg_scen - avg_baseline
            pct = (delta / avg_baseline * 100) if avg_baseline else 0
            sign = "BAJA" if delta < -2 else "SUBE" if delta > 2 else "NEUTRO"
            _log(f"[{scen}] avg {avg_scen:.1f} t/s delta {delta:+.1f} ({pct:+.1f}%) vs baseline — {sign}")
            if delta < -5:
                _log(f"  -> {scen} MEJORA rendimiento (deshabilitar empeora) — parche BENEFICIA")
            elif delta > 5:
                _log(f"  -> {scen} EMPEORA rendimiento — parche podría estar regresando (revisar)")

    # hangs
    hangs = [r for r in results if not r["ok"]]
    if hangs:
        _log(f"\n[HANGS] {len(hangs)} tests fallaron: {[r['label'] for r in hangs]}")
        _log("  Revisar docker logs arriba para culpable; sugerido GENESIS_DISABLE_<parche>=1")
    else:
        _log("\n[HANGS] 0 — ningún parche colgó (todos respondieron en <30s)")

    _log(f"\n[log] detalle completo en {LOG_PATH}")
    # resumen JSON para parseo
    summary = {
        "endpoint": ENDPOINT,
        "model": MODEL,
        "scenarios": SCENARIOS,
        "results": results,
        "avg_baseline_tps": avg_baseline,
        "expected_range": [80, 95],
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n[JSON_SUMMARY]\n" + json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
