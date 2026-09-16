#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Suite de calidad y regresion — workshop/ox_alpha parte 2.

Mismos prompts tetris y rpg que suite_parches, pero valida que el codigo
generado sea FUNCIONAL (contiene tokens esperados) sin ejecutarlo. No usa
exec/eval, solo verifica substrings.

Tokens esperados:
  tetris: import curses, class Tetris, defs de rotacion, board, wrapper
  rpg:    class Hero, class Monster, combate/attack, inventario/map

Ejecucion:
    python workshop/ox_alpha/tests/suite_calidad.py
    # guarda outputs en /tmp/opencode/suite_calidad.log
    # valida contra PROD qwen3.8 con PN110 activo

Referencias:
    workshop/ox_alpha/tests/suite_parches.py: PROMPTS (mismos prompts)
    compose/docker-compose.qwen38-27b-fp8.yml: GENESIS_ENABLE_PN110_INT8_PHASE_DISPATCH=1
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

ENDPOINT = os.environ.get("GENESIS_PROD_ENDPOINT", "http://127.0.0.1:8320/v1/chat/completions")
MODEL = os.environ.get("GENESIS_PROD_MODEL", "qwen3.8")
LOG_PATH = Path("/tmp/opencode/suite_calidad.log")
BASELINE_PATH = Path("/tmp/opencode/suite_calidad_baseline.json")
CONTAINER = os.environ.get("GENESIS_CONTAINER", "genesis-27b-qwen38-fp8")
API_KEY = os.environ.get("VLLM_API_KEY", "<REDACTADO: clave rotada 2026-09-19>")

# mismos prompts que suite_parches (tetris y rpg) workshop/ox_alpha/tests/suite_parches.py:32
PROMPTS: dict[str, str] = {
    "tetris": "Escribe un juego de Tetris completo en Python para terminal (curses), con rotacion, lineas, score, y controles WASD. Codigo funcional y comentado.",
    "rpg": "Escribe un juego de rol CLI en Python con clases Hero, Monster, combate por turnos, inventario, y mapa 5x5. Codigo funcional.",
}

# tokens esperados — no ejecutar codigo, solo validar substrings
EXPECTED: dict[str, list[str]] = {
    "tetris": [
        "import curses",
        "class Tetris",
        "rotate",           # def rotate / def _rotate / rotacion
        "board",            # tablero / board / grid
        "curses.wrapper",   # bucle principal
        "score",            # lineas/score
    ],
    "rpg": [
        "class Hero",
        "class Monster",
        "attack",           # def attack / combate
        "def ",             # al menos una funcion
        "inventory",        # inventario (ingles/español acepta ambos)
        "map",              # mapa 5x5
    ],
}

# variantes aceptadas para algunos tokens (ingles/español)
ALIASES: dict[str, list[str]] = {
    "inventory": ["inventory", "inventario"],
    "map": ["map", "mapa"],
    "attack": ["attack", "ataque", "combate", "combat", "battle"],
    "rotate": ["rotate", "rotar", "rotacion", "rotación"],
    "board": ["board", "tablero", "grid"],
    "score": ["score", "puntaje", "puntos", "line"],
}

TIMEOUT_S = 30
MAX_TOKENS = 1500  # suficiente para ver clases/funciones completas; 800 truncaba antes de class Tetris/Monster (finish=length)
TEMPERATURE = 0.7

# baseline de referencia: lo que dio suite_parches baseline (500 tokens) y
# lo que se espera para un codigo completo. Si no hay baseline previo,
# se usa este como referencia inmutable.
HARDCODED_BASELINE = {
    "tetris": {"chars": 2800, "tokens": 500, "matched": 5},
    "rpg": {"chars": 2700, "tokens": 500, "matched": 5},
}


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


def _check_token(text: str, token: str) -> bool:
    low = text.lower()
    tok_low = token.lower()
    # si es alias, probar variantes
    if tok_low in ALIASES:
        return any(v in low for v in ALIASES[tok_low])
    # para tokens con espacio (ej "import curses") buscar exacto case-insensitive
    return tok_low in low


def call_once(prompt: str, label: str, timeout: int = TIMEOUT_S) -> dict:
    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": False,
    }
    t0 = time.monotonic()
    try:
        resp = requests.post(ENDPOINT, headers=_headers(), json=payload, timeout=timeout)
        wall = time.monotonic() - t0
        try:
            data = resp.json()
        except Exception:
            data = {"raw": resp.text[:4000], "status": resp.status_code}
        if resp.status_code != 200:
            _log(f"[{label}] HTTP {resp.status_code} body={resp.text[:1000]}")
            return {"label": label, "wall": wall, "tokens": 0, "chars": 0, "text": "", "ok": False, "status": resp.status_code, "error": f"HTTP {resp.status_code}", "usage": {}}
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = msg.get("content") or ""
        # fallback: si content null y reasoning tiene codigo, usar reasoning
        if not text and msg.get("reasoning"):
            text = msg.get("reasoning") or ""
        if not text and msg.get("reasoning_content"):
            text = msg.get("reasoning_content") or ""
        usage = data.get("usage") or {}
        ct = usage.get("completion_tokens") or 0
        if ct == 0:
            ct = max(1, len(text) // 4)
        # guardar output completo en log
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n--- {label} wall={wall:.2f}s tokens={ct} chars={len(text)} status={resp.status_code} finish={choice.get('finish_reason')} ---\n")
            f.write(text[:12000] + "\n")
            f.write(f"--- usage: {json.dumps(usage)} ---\n")
        return {"label": label, "wall": wall, "tokens": ct, "chars": len(text), "text": text, "ok": True, "status": resp.status_code, "usage": usage, "finish": choice.get("finish_reason"), "error": None}
    except requests.exceptions.Timeout:
        wall = time.monotonic() - t0
        _log(f"[TIMEOUT] {label} timeout {timeout}s wall={wall:.1f}s")
        return {"label": label, "wall": wall, "tokens": 0, "chars": 0, "text": "", "ok": False, "status": 0, "error": "timeout", "usage": {}}
    except Exception as e:
        wall = time.monotonic() - t0
        _log(f"[ERROR] {label} {type(e).__name__}: {e} wall={wall:.1f}s")
        return {"label": label, "wall": wall, "tokens": 0, "chars": 0, "text": "", "ok": False, "status": 0, "error": str(e), "usage": {}}


def validate(prompt_name: str, text: str) -> dict:
    expected = EXPECTED[prompt_name]
    hits: dict[str, bool] = {}
    for tok in expected:
        hits[tok] = _check_token(text, tok)
    matched = sum(1 for v in hits.values() if v)
    total = len(expected)
    # funcional si >= 4/6 para tetris y >=4/6 para rpg (umbral 66%)
    functional = matched >= 4 and text.strip() != "" and len(text) > 500
    # detalle extra: tetris exige import curses y class Tetris ambos
    if prompt_name == "tetris":
        functional = functional and hits.get("import curses", False) and hits.get("class Tetris", False)
    if prompt_name == "rpg":
        functional = functional and hits.get("class Hero", False) and hits.get("class Monster", False)
    return {"hits": hits, "matched": matched, "total": total, "functional": functional, "completeness": matched / total if total else 0}


def _poll_health(timeout: int = 30) -> bool:
    url_health = ENDPOINT.replace("/v1/chat/completions", "/health")
    url_models = ENDPOINT.replace("/v1/chat/completions", "/v1/models")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for url in (url_health, url_models):
            try:
                r = requests.get(url, headers=_headers(), timeout=5)
                if r.status_code in (200, 401):
                    return True
            except Exception:
                pass
        time.sleep(2)
    return False


def _verify_pn110() -> dict:
    info: dict = {"active": False, "env": "", "log": ""}
    try:
        r = subprocess.run(["docker", "inspect", CONTAINER, "--format", "{{json .Config.Env}}"], capture_output=True, text=True, timeout=10)
        env_list = json.loads(r.stdout.strip()) if r.stdout.strip() else []
        env_str = "\n".join(e for e in env_list if "PN110" in e or "GENESIS" in e)
        # buscar flag
        for e in env_list:
            if e.startswith("GENESIS_ENABLE_PN110_INT8_PHASE_DISPATCH=1"):
                info["active"] = True
            if e.startswith("GENESIS_DISABLE_PN110=1"):
                info["active"] = False
        info["env"] = env_str[:3000]
    except Exception as e:
        info["env"] = f"inspect fail: {e}"
    try:
        r2 = subprocess.run(["docker", "logs", "--tail", "200", CONTAINER], capture_output=True, text=True, timeout=10)
        tail = r2.stdout + r2.stderr
        if "PN110" in tail:
            # buscar linea de aplicado
            for line in tail.splitlines():
                if "PN110" in line:
                    info["log"] += line + "\n"
            if "applied: PN110" in tail or "PN110 instalado" in tail:
                info["active"] = True
        info["log"] = info["log"][:3000]
    except Exception as e:
        info["log"] = f"logs fail: {e}"
    return info


def main() -> int:
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    _log(f"=== suite_calidad PROD={CONTAINER} endpoint={ENDPOINT} model={MODEL} max_tokens={MAX_TOKENS} timeout={TIMEOUT_S}s ===")
    # verificar estado prod
    try:
        r = subprocess.run(["docker", "inspect", CONTAINER, "--format", "{{.State.Status}} {{.State.Running}}"], capture_output=True, text=True, timeout=10)
        _log(f"[prod status] {r.stdout.strip()} {r.stderr.strip()}")
    except Exception as e:
        _log(f"[prod status fail] {e}")

    pn = _verify_pn110()
    _log(f"[PN110 activo] {pn['active']} (env flag + log)")
    _log(f"[PN110 env]\n{pn['env'][:2000]}")
    if pn["log"]:
        _log(f"[PN110 log tail]\n{pn['log'][:2000]}")

    if not _poll_health(timeout=30):
        _log("[WARN] PROD no responde en 30s — esperando 15s extra y reintentando")
        time.sleep(5)
        if not _poll_health(timeout=30):
            _log("[FATAL] PROD sigue DOWN — revisar docker logs genesis-27b-qwen38-fp8")
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write("\n[JSON_SUMMARY]\n" + json.dumps({"fatal": "PROD_DOWN", "pn110": pn}, indent=2) + "\n")
            return 2
    _log("[health] PROD UP")

    results: list[dict] = []
    for name, prompt in PROMPTS.items():
        label = name
        _log(f"[test] {label} POST {ENDPOINT} max_tokens={MAX_TOKENS}")
        res = call_once(prompt, label, timeout=TIMEOUT_S)
        _log(f"[result] {label} wall={res['wall']:.2f}s tokens={res['tokens']} chars={res['chars']} ok={res['ok']} err={res.get('error')} finish={res.get('finish')}")
        if not res["ok"] and res.get("error") == "timeout":
            _log(f"[retry] {label} reintento tras 2s")
            time.sleep(2)
            res2 = call_once(prompt, label + "_retry", timeout=TIMEOUT_S)
            results.append(res2)
            _log(f"[retry result] {label} wall={res2['wall']:.2f}s ok={res2['ok']}")
            # validar el reintento si el primero fallo
            val = validate(name, res2.get("text", ""))
            res2["validation"] = val
            results.append({**res, "validation": validate(name, res.get("text", ""))})
        else:
            val = validate(name, res.get("text", ""))
            res["validation"] = val
            results.append(res)
        _log(f"[validacion {name}] matched {val['matched']}/{val['total']} hits={val['hits']} funcional={val['functional']} completeness={val['completeness']:.0%}")

    # reporte de calidad
    _log("\n========== REPORTE CALIDAD ==========")
    all_func = True
    for r in results:
        # solo contar los primarios (sin _retry)
        if "_retry" in r["label"]:
            continue
        name = r["label"]
        v = r.get("validation", {})
        _log(f"  {name}: wall={r['wall']:.1f}s tokens={r['tokens']} chars={r['chars']} matched={v.get('matched')}/{v.get('total')} funcional={v.get('functional')} completeness={v.get('completeness',0):.0%}")
        for tok, hit in v.get("hits", {}).items():
            _log(f"    {'✅' if hit else '❌'} {tok}")
        if not v.get("functional"):
            all_func = False

    if all_func:
        _log("\n[CALIDAD] ✅ FUNCIONAL — ambos prompts contienen clases/funciones esperadas (import curses, class Tetris, rotate, class Hero/Monster, attack, etc.)")
    else:
        _log("\n[CALIDAD] ❌ NO FUNCIONAL — al menos un prompt no contiene tokens esperados (ver hits arriba)")

    # regresion vs baseline (longitud y completitud)
    _log("\n========== REGRESION VS BASELINE ==========")
    baseline = HARDCODED_BASELINE
    # si hay baseline previo en disco, usarlo
    disk_baseline = None
    if BASELINE_PATH.exists():
        try:
            disk_baseline = json.loads(BASELINE_PATH.read_text())
            _log(f"[baseline disco] cargado desde {BASELINE_PATH}: {disk_baseline}")
            baseline = disk_baseline
        except Exception as e:
            _log(f"[baseline disco fail] {e} — usando hardcoded")
    else:
        _log(f"[baseline] no hay {BASELINE_PATH} — usando HARDCODED_BASELINE {baseline} y guardando actual como nuevo baseline si funcional")

    degraded = False
    for r in results:
        if "_retry" in r["label"]:
            continue
        name = r["label"]
        v = r.get("validation", {})
        cur_chars = r["chars"]
        cur_tokens = r["tokens"]
        cur_matched = v.get("matched", 0)
        base = baseline.get(name, {})
        base_chars = base.get("chars", 0)
        base_tokens = base.get("tokens", 0)
        base_matched = base.get("matched", 0)
        # longitud: degradacion si <70% del baseline
        if base_chars and cur_chars < base_chars * 0.7:
            _log(f"  {name} DEGRADADO longitud: {cur_chars} chars vs baseline {base_chars} ({cur_chars/base_chars:.0%} <70%)")
            degraded = True
        else:
            pct = (cur_chars / base_chars * 100) if base_chars else 0
            _log(f"  {name} longitud OK: {cur_chars} vs {base_chars} ({pct:.0f}%)")
        # tokens truncados?
        if base_tokens and cur_tokens < base_tokens * 0.7:
            _log(f"  {name} DEGRADADO tokens: {cur_tokens} vs {base_tokens} ({cur_tokens/base_tokens:.0%} <70%)")
            degraded = True
        # completitud
        if base_matched and cur_matched < base_matched:
            _log(f"  {name} DEGRADADO completitud: {cur_matched}/{v.get('total')} vs baseline {base_matched} (faltan {base_matched - cur_matched})")
            degraded = True
        else:
            _log(f"  {name} completitud OK: {cur_matched}/{v.get('total')} vs baseline {base_matched}")

    if degraded:
        _log("\n[REGRESION] ⚠️ DEGRADACION detectada — longitud o completitud por debajo de baseline (ver lineas arriba)")
    else:
        _log("\n[REGRESION] ✅ SIN DEGRADACION — longitud y completitud >=70% baseline y sin perdida de tokens esperados")

    # guardar baseline si funcional y no existia
    if all_func and disk_baseline is None:
        new_base = {r["label"]: {"chars": r["chars"], "tokens": r["tokens"], "matched": r["validation"]["matched"]} for r in results if "_retry" not in r["label"]}
        try:
            BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
            BASELINE_PATH.write_text(json.dumps(new_base, indent=2))
            _log(f"[baseline] guardado nuevo baseline en {BASELINE_PATH}: {new_base}")
        except Exception as e:
            _log(f"[baseline save fail] {e}")

    # summary json
    summary = {
        "endpoint": ENDPOINT,
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "timeout_s": TIMEOUT_S,
        "pn110_active": pn["active"],
        "results": [
            {
                "label": r["label"],
                "wall": r["wall"],
                "tokens": r["tokens"],
                "chars": r["chars"],
                "ok": r["ok"],
                "finish": r.get("finish"),
                "validation": r.get("validation"),
            }
            for r in results
        ],
        "all_functional": all_func,
        "degraded": degraded,
        "baseline": baseline,
        "pn110": pn,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n[JSON_SUMMARY]\n" + json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    _log(f"\n[log] detalle completo en {LOG_PATH}")

    if not all_func:
        return 1
    if degraded:
        return 3  # funcional pero degradado
    return 0


if __name__ == "__main__":
    sys.exit(main())
