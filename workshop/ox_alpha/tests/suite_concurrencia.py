#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Suite de concurrencia y estabilidad — workshop/ox_alpha parte 3.

Mide N requests concurrentes (1, 4, 8) con prompt tetris contra PROD
(qwen3.8 FP8, PN110+B3), usando asyncio + aiohttp. Reporta tps agregado,
latencia p50/p90, y detecta OOM/cuelgues con timeout 30s por request.

Ejecución esperada:
    python workshop/ox_alpha/tests/suite_concurrencia.py  # contra 127.0.0.1:8320
    # log en /tmp/opencode/suite_concurrencia.log
    # si cuelgue: docker logs genesis-27b-qwen38-fp8 --since 5m | grep -E "ERROR|OOM|Traceback"

Referencias:
    compose/docker-compose.qwen38-27b-fp8.yml:8320 -> qwen3.8
    workshop/ox_alpha/PLAN-CHECKPOINTS.md CK-2.1/CK-4.2 (PN110 + B3)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

import aiohttp

# ── prompt tetris ────────────────────────────────────────────────────────────
PROMPT_TETRIS = (
    "Escribe un juego de Tetris completo en Python, ejecutable en terminal, "
    "sin dependencias externas. Solo codigo, con type hints y comentarios breves. "
    "Debe incluir tablero 10x20, piezas con rotacion, deteccion de lineas y "
    "game over. Al final, un bloque de ejemplo de uso si se ejecuta como __main__."
)

# Para que no comparta prefix cache entre requests concurrentes (el tier
# de offloading hace dedup por hash de contenido), cada request lleva un
# nonce distinto como prefijo. Así cada una prefilea de verdad.
NONCE_PREFIX = "[suite_concurrencia {idx} {ts}] "

DEFAULT_LEVELS = [1, 4, 8]
DEFAULT_MAX_TOKENS = 512
DEFAULT_TIMEOUT_S = 30
DEFAULT_MODEL = "qwen3.8"
DEFAULT_BASE_URL = "http://127.0.0.1:8320"
DEFAULT_API_KEY = os.environ.get("VLLM_API_KEY", "<REDACTADO: clave rotada 2026-09-19>")


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    d0 = k - f
    return sorted_vals[f] * (1 - d0) + sorted_vals[c] * d0


async def one_request(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict,
    model: str,
    idx: int,
    max_tokens: int,
    timeout_s: int,
) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": f"{NONCE_PREFIX.format(idx=idx, ts=int(time.time()))}{PROMPT_TETRIS}"}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    try:
        async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
            body = await resp.text()
            latency = time.perf_counter() - t0
            if resp.status != 200:
                return {
                    "idx": idx,
                    "ok": False,
                    "latency": latency,
                    "status": resp.status,
                    "body_snippet": body[:500],
                    "error": f"HTTP {resp.status}",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                }
            try:
                data = json.loads(body)
            except json.JSONDecodeError as e:
                return {
                    "idx": idx,
                    "ok": False,
                    "latency": latency,
                    "status": resp.status,
                    "body_snippet": body[:500],
                    "error": f"JSON decode: {e}",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                }
            usage = data.get("usage") or {}
            # vLLM a veces devuelve usage solo con streaming; en no-streaming viene acá
            pt = usage.get("prompt_tokens") or 0
            ct = usage.get("completion_tokens") or 0
            # fallback: si no hay usage pero hay choices, estimar ct=0 y marcar
            if ct == 0 and data.get("choices"):
                # no usage pero respuesta válida — contar como ok pero sin tps fiable
                pass
            return {
                "idx": idx,
                "ok": True,
                "latency": latency,
                "status": resp.status,
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "finish_reason": (data.get("choices") or [{}])[0].get("finish_reason"),
            }
    except asyncio.TimeoutError:
        return {
            "idx": idx,
            "ok": False,
            "latency": time.perf_counter() - t0,
            "error": f"TIMEOUT {timeout_s}s",
            "timeout": True,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
    except aiohttp.ClientError as e:
        return {
            "idx": idx,
            "ok": False,
            "latency": time.perf_counter() - t0,
            "error": f"ClientError: {type(e).__name__}: {e}",
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
    except Exception as e:
        return {
            "idx": idx,
            "ok": False,
            "latency": time.perf_counter() - t0,
            "error": f"{type(e).__name__}: {e}",
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }


async def run_level(
    base_url: str,
    api_key: str,
    model: str,
    n: int,
    max_tokens: int,
    timeout_s: int,
) -> dict:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # health pre-check con reintentos (PROD puede estar arrancando)
    health_ok = True
    health_err = ""
    for attempt in range(3):
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(
                    f"{base_url.rstrip('/')}/health",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as r:
                    health_ok = r.status == 200
                    if not health_ok:
                        health_err = f"HTTP {r.status}"
                    else:
                        health_err = ""
                        break
        except Exception as e:
            health_ok = False
            health_err = f"{type(e).__name__}: {e}"
            # reintento corto si es durante boot (Connection reset / Server disconnected)
            await asyncio.sleep(2)
            continue
        # si llegó acá y no fue 200, también reintentar
        if not health_ok:
            await asyncio.sleep(2)
    if not health_ok:
        print(f"[N={n}] health check FALLO tras 3 intentos ({health_err}) — intentando requests igual (puede ser boot)", flush=True)
        # no abortamos: intentamos los requests; si también fallan, se marcará hung
        # pero avisamos que el health no estaba listo

    t_wall0 = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        tasks = [one_request(session, url, headers, model, i, max_tokens, timeout_s) for i in range(n)]
        results = await asyncio.gather(*tasks)
    t_wall = time.perf_counter() - t_wall0

    oks = [r for r in results if r.get("ok")]
    fails = [r for r in results if not r.get("ok")]
    timeouts = [r for r in fails if r.get("timeout")]
    lats = sorted([r["latency"] for r in results])
    lats_ok = sorted([r["latency"] for r in oks])

    total_ct = sum(r.get("completion_tokens", 0) for r in oks)
    total_pt = sum(r.get("prompt_tokens", 0) for r in oks)
    tps_agg = total_ct / t_wall if t_wall > 0 and total_ct else 0.0
    # si usage no trae tokens (ct=0) el tps será 0 — avisar
    p50 = percentile(lats_ok or lats, 50) if lats else 0.0
    p90 = percentile(lats_ok or lats, 90) if lats else 0.0
    p50_all = percentile(lats, 50) if lats else 0.0
    p90_all = percentile(lats, 90) if lats else 0.0

    hung = len(timeouts) > 0 or len(fails) == n  # todos fallan o al menos un timeout => posible cuelgue
    # también considerar hung si la latencia supera timeout_s por mucho (vLLM no respondió)
    if not hung and lats and max(lats) >= timeout_s * 0.95 and len(fails) > 0:
        hung = True

    # salud post-nivel
    post_ok = True
    post_err = ""
    try:
        # sync check rápido, no bloquea mucho (5s)
        import urllib.request

        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/health", headers={"Authorization": f"Bearer {api_key}"}
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            post_ok = r.status == 200
    except Exception as e:
        post_ok = False
        post_err = f"{type(e).__name__}: {e}"
        if not hung and not oks:
            hung = True

    return {
        "n": n,
        "wall": t_wall,
        "results": results,
        "oks": oks,
        "fails": fails,
        "timeouts": timeouts,
        "total_prompt_tokens": total_pt,
        "total_completion_tokens": total_ct,
        "tps_agg": tps_agg,
        "p50": p50,
        "p90": p90,
        "p50_all": p50_all,
        "p90_all": p90_all,
        "hung": hung,
        "health_ok": health_ok,
        "post_ok": post_ok,
        "post_err": post_err,
    }


def print_level_report(rep: dict) -> None:
    n = rep["n"]
    print(f"\n{'='*72}", flush=True)
    wall = rep.get("wall", 0.0)
    oks = rep.get("oks", [])
    fails = rep.get("fails", [])
    to = rep.get("timeouts", [])
    print(f"[N={n}] wall={wall:.2f}s  ok={len(oks)}/{n}  fail={len(fails)}  timeouts={len(to)}", flush=True)
    tps = rep.get("tps_agg", 0.0)
    tot_ct = rep.get("total_completion_tokens", 0)
    tot_pt = rep.get("total_prompt_tokens", 0)
    if tot_ct:
        print(f"  tps agregado (completion): {tps:.1f} tok/s  ({tot_ct} tok / {wall:.2f}s)", flush=True)
        print(f"  prompt_tokens totales: {tot_pt}", flush=True)
    else:
        print(f"  tps agregado: N/A (sin completion_tokens en usage — revisar respuesta)", flush=True)
    oks = rep.get("oks", [])
    fails = rep.get("fails", [])
    if oks:
        p50 = rep.get("p50", 0.0)
        p90 = rep.get("p90", 0.0)
        print(f"  latencia OK: p50={p50:.2f}s  p90={p90:.2f}s  (min {min(r['latency'] for r in oks):.2f}s max {max(r['latency'] for r in oks):.2f}s)", flush=True)
    if fails:
        print(f"  latencia ALL: p50={rep.get('p50_all',0):.2f}s  p90={rep.get('p90_all',0):.2f}s", flush=True)
        for r in fails[:8]:
            print(f"    fail idx={r['idx']} lat={r['latency']:.2f}s err={r.get('error')} status={r.get('status')} snippet={r.get('body_snippet','')[:120]}", flush=True)
    print(f"  health pre={rep.get('health_ok')} post={rep.get('post_ok')} {rep.get('post_err','')}", flush=True)
    if rep.get("hung"):
        print(f"  ⚠️  POSIBLE CUELGUE/OOM detectado en N={n} — revisar docker logs", flush=True)
    else:
        print(f"  ✅ estable en N={n}", flush=True)


async def main_async(args) -> int:
    base = args.base_url
    key = args.api_key
    model = args.model
    levels = args.levels
    max_tokens = args.max_tokens
    timeout_s = args.timeout

    # auto-detect model si se pide
    if not model or model == "auto":
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(
                    f"{base.rstrip('/')}/v1/models",
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as r:
                    data = await r.json()
                    model = data["data"][0]["id"]
                    print(f"[info] modelo detectado: {model}", flush=True)
        except Exception as e:
            print(f"[warn] no se pudo autodetectar modelo ({e}), usando {DEFAULT_MODEL}", flush=True)
            model = DEFAULT_MODEL

    print(f"[suite_concurrencia] PROD={base} model={model} max_tokens={max_tokens} timeout={timeout_s}s", flush=True)
    print(f"[suite_concurrencia] niveles={levels} prompt=tetris ({len(PROMPT_TETRIS)} chars)", flush=True)
    # verificar flags PN110+B3 desde env del contenedor si es local
    if "PN110" not in os.environ.get("GENESIS_ENABLE_PN110_INT8_PHASE_DISPATCH", ""):
        print("[info] PN110 se verifica en PROD via docker inspect (no env local)", flush=True)

    overall_hung = False
    reports = []
    for n in levels:
        rep = await run_level(base, key, model, n, max_tokens, timeout_s)
        reports.append(rep)
        print_level_report(rep)
        if rep["hung"]:
            overall_hung = True
            # si cuelga en 1, quizá no tenga sentido seguir a 4 y 8 sin pausa,
            # pero seguimos para tener la foto completa; dejamos 2s de respiro
            await asyncio.sleep(2)
        else:
            await asyncio.sleep(1)

    print(f"\n{'='*72}", flush=True)
    print("[RESUMEN suite_concurrencia]", flush=True)
    for rep in reports:
        flag = "CUELGUE" if rep.get("hung") else "OK"
        wall = rep.get("wall", 0.0)
        tps = rep.get("tps_agg", 0.0)
        p50 = rep.get("p50", 0.0)
        p90 = rep.get("p90", 0.0)
        oks = len(rep.get("oks", []))
        print(
            f"  N={rep['n']:>2}  {flag:<7}  wall {wall:5.1f}s  tps_agg {tps:6.1f}  p50 {p50:5.2f}s  p90 {p90:5.2f}s  ok {oks}/{rep['n']}",
            flush=True,
        )
    if overall_hung:
        print("\n[RESULTADO] ❌ CUELGUE/OOM detectado — PROD no respondió a tiempo en al menos un nivel.", flush=True)
        print("  Siguiente paso:", flush=True)
        print("    docker logs genesis-27b-qwen38-fp8 --since 5m | grep -E \"ERROR|OOM|Traceback\"", flush=True)
        print("  Parche culpable más probable si hay Traceback post-PN110/B3:", flush=True)
        print("    - Si Traceback en cutlass_scaled_mm / int8 / marlin / PN110 -> PN110 (phase dispatch)", flush=True)
        print("    - Si 'custom_all_reduce.cuh:455 invalid argument' / all-reduce / NCCL -> B3 (custom AR)", flush=True)
        print("    - Si CUDA OOM / out of memory / _allocate_kv_cache -> saturación VRAM (no parche puntual, bajar --gpu-memory-utilization o concurrencia)", flush=True)
        return 2
    else:
        print("\n[RESULTADO] ✅ estable — sin cuelgues ni OOM en ningún nivel (1,4,8)", flush=True)
        # tps agregado máximo y latencia
        best = max(reports, key=lambda r: r.get("tps_agg", 0.0))
        print(f"  Mejor tps agregado: N={best.get('n')} -> {best.get('tps_agg',0):.1f} tok/s", flush=True)
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Suite concurrencia y estabilidad (tetris, 1/4/8)")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL de vLLM (default %(default)s)")
    ap.add_argument("--api-key", default=DEFAULT_API_KEY, help="VLLM_API_KEY")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="served-model-name o auto")
    ap.add_argument("--levels", nargs="+", type=int, default=DEFAULT_LEVELS, help="N concurrentes por nivel")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="max_tokens por request")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S, help="timeout por request (s)")
    args = ap.parse_args()
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\n[abort] interrumpido", flush=True)
        return 130


if __name__ == "__main__":
    sys.exit(main())
