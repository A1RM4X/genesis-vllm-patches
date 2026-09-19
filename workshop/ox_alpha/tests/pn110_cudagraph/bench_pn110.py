#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""bench_pn110.py — bench de performance + diagnóstico de cudagraphs para PN110.

Arranca tras levantar el contenedor (ver run_ab.sh). Mide la performance real
de vLLM bajo el escenario que se está debugueando (PN110 + superkernels +
cudagraphs) y detecta si la captura de CUDA graphs falló — que por sí sola
explicaría la lentitud, porque vLLM cae a modo eager (~2x más lento que con
cudagraphs capturados).

Métricas (vía streaming SSE para tiempos finos):
  * TTFT                : tiempo al primer token (proxy de prefill).
  * decode tok/s        : tokens generados / (t_último - t_primer_token).
  * prefill tok/s       : prompt_tokens / TTFT (solo en el request de prefill).

Diagnóstico de cudagraphs (grep de logs del contenedor):
  * ÉXITO : aparece "Capturing cuda graphs" / "cuda graph" capturado.
  * FALLO : aparece "Assigning / modifying buffers of nn.Module during forward
    pass" -> la captura aborta y vLLM corre eager (lento).

Uso:
  python3 bench_pn110.py --container genesis-pn110-cg --label "SK=1 full" \
      --port 8390 --prompt-decode-len 24 --gen-decode 200 \
      --prompt-prefill-len 2000 --gen-prefill 16 --requests 3

Autor: ox-alpha workshop — genesis-vllm-patches
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time

import requests
import os

HEALTH_URL_TPL = "http://{host}:{port}/health"
CHAT_URL_TPL = "http://{host}:{port}/v1/chat/completions"

# Señales de diagnóstico de cudagraphs en los logs.
CG_OK_PATTERNS = [
    "Capturing cuda graphs",
    "capturing cuda graph",
    "cuda graph capture",
]
CG_FAIL_PATTERNS = [
    "Assigning / modifying buffers of nn.Module during forward pass",
    "CUDA graph capture failed",
    "cudagraph capture",
]


def _docker_logs(container: str, tail: int = 4000) -> str:
    try:
        proc = subprocess.run(
            ["docker", "logs", "--tail", str(tail), container],
            capture_output=True, text=True, timeout=60,
        )
        return (proc.stdout or "") + "\n" + (proc.stderr or "")
    except Exception as exc:  # noqa: BLE001
        return f"<no se pudieron leer logs: {exc}>"


def _wait_health(url: str, timeout: float) -> tuple[bool, float]:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            if requests.get(url, timeout=5).status_code == 200:
                return True, time.monotonic() - t0
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)
    return False, timeout


def _stream_once(url: str, headers: dict, payload: dict, timeout: int) -> dict:
    """Envía un request en streaming y devuelve tiempos/tokens.

    Retorna dict con: ok, ttft, first_t, last_t, n_tokens, total_wall, error.
    """
    t_start = time.monotonic()
    out = {"ok": False, "ttft": None, "first_t": None, "last_t": None,
           "n_tokens": 0, "total_wall": None, "error": None}
    try:
        resp = requests.post(url, headers=headers, json=payload,
                             stream=True, timeout=timeout)
        if resp.status_code != 200:
            out["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
            return out
        n = 0
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                continue
            try:
                obj = json.loads(data)
            except Exception:  # noqa: BLE001
                continue
            # usage final (si stream_options.include_usage=true)
            if obj.get("usage"):
                n = obj["usage"].get("completion_tokens", n) or n
                out["prompt_tokens"] = obj["usage"].get("prompt_tokens", out.get("prompt_tokens"))
            for ch in obj.get("choices", []):
                delta = ch.get("delta", {}).get("content")
                if delta:
                    n += 1
                    now = time.monotonic()
                    if out["first_t"] is None:
                        out["first_t"] = now
                        out["ttft"] = now - t_start
                    out["last_t"] = now
        out["total_wall"] = time.monotonic() - t_start
        out["n_tokens"] = n
        out["ok"] = out["first_t"] is not None and n > 0
        return out
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
        out["total_wall"] = time.monotonic() - t_start
        return out


def _measure_decode(url: str, headers: dict, model: str, prompt_len: int,
                    gen_tokens: int, requests: int, warmup: int,
                    timeout: int) -> dict:
    """Mide decode tok/s con prompt corto y generación larga."""
    filler = ("Resuelve paso a paso: si tengo 3 cajas con 4 manzanas cada una "
              "y como 2, ¿cuántas quedan? Explica el razonamiento. ")
    reps = max(1, prompt_len // len(filler)) if prompt_len else 1
    prompt = filler * reps
    base = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    for _ in range(warmup):
        _stream_once(url, headers, base, timeout)
    ttfts, dtps, wtok = [], [], []
    for _ in range(requests):
        r = _stream_once(url, headers, base, timeout)
        if not r["ok"]:
            return {"ok": False, "error": r["error"], "samples": []}
        gen_t = (r["last_t"] - r["first_t"]) or 1e-6
        dtps.append((r["n_tokens"] - 1) / gen_t)
        ttfts.append(r["ttft"])
        wtok.append(r["n_tokens"])
    return {
        "ok": True,
        "decode_tok_s": sum(dtps) / len(dtps),
        "ttft_s": sum(ttfts) / len(ttfts),
        "gen_tokens": sum(wtok) / len(wtok),
        "samples": dtps,
    }


def _measure_prefill(url: str, headers: dict, model: str, prompt_len: int,
                     gen_tokens: int, timeout: int) -> dict:
    """Mide prefill tok/s con prompt largo y generación corta (prefill-dom).

    Usa un POST no-streaming (robusto): mide el wall total y lee
    ``usage.prompt_tokens``. Como la generación es corta (gen_tokens), el wall
    es una buena aproximación del tiempo de prefill, así que
    prefill_tok_s ≈ prompt_tokens / wall.
    """
    filler = ("Analiza el siguiente problema considerando todas las "
              "restricciones y casos borde posibles. ")
    reps = max(1, prompt_len * 12 // len(filler))
    prompt = filler * reps
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    t0 = time.monotonic()
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
    wall = time.monotonic() - t0
    if resp.status_code != 200:
        return {"ok": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    try:
        data = resp.json()
        ptk = data.get("usage", {}).get("prompt_tokens") or reps
        ctk = data.get("usage", {}).get("completion_tokens") or 0
    except Exception:  # noqa: BLE001
        ptk, ctk = reps, 0
    return {
        "ok": True,
        "ttft_s": round(wall, 3),
        "prefill_tok_s": round(ptk / (wall or 1e-6), 1),
        "gen_tokens": ctk,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--container", default="genesis-pn110-cg")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8390)
    ap.add_argument("--label", default="?", help="etiqueta del config (p.ej. SK=1)")
    ap.add_argument("--model", default="qwen3.8")
    ap.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""))
    ap.add_argument("--timeout", type=int, default=480, help="segundos para /health")
    ap.add_argument("--prompt-decode-len", type=int, default=24)
    ap.add_argument("--gen-decode", type=int, default=200)
    ap.add_argument("--prompt-prefill-len", type=int, default=2000)
    ap.add_argument("--gen-prefill", type=int, default=16)
    ap.add_argument("--requests", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--req-timeout", type=int, default=180)
    ap.add_argument("--json-out", default="", help="ruta para volcar el reporte JSON")
    args = ap.parse_args()

    health_url = HEALTH_URL_TPL.format(host=args.host, port=args.port)
    chat_url = CHAT_URL_TPL.format(host=args.host, port=args.port)
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    print(f"[bench] label={args.label} container={args.container}", flush=True)
    ok, elapsed = _wait_health(health_url, args.timeout)
    if not ok:
        print(f"[bench] FAIL: /health no respondió 200 en {args.timeout}s", flush=True)
        print(_docker_logs(args.container).splitlines()[-50:], flush=True)
        sys.exit(2)
    print(f"[bench] /health OK en {elapsed:.1f}s", flush=True)

    print(f"[bench] midiendo DECODE ({args.requests} req, ~{args.gen_decode} tok) ...",
          flush=True)
    dec = _measure_decode(chat_url, headers, args.model, args.prompt_decode_len,
                          args.gen_decode, args.requests, args.warmup,
                          args.req_timeout)
    print(f"[bench] midiendo PREFILL (prompt ~{args.prompt_prefill_len} tok) ...",
          flush=True)
    pre = _measure_prefill(chat_url, headers, args.model, args.prompt_prefill_len,
                           args.gen_prefill, args.req_timeout)

    logs = _docker_logs(args.container)
    low = logs.lower()
    cg_ok = [p for p in CG_OK_PATTERNS if p.lower() in low]
    cg_fail = [p for p in CG_FAIL_PATTERNS if p.lower() in low]

    report = {
        "label": args.label,
        "health_ok": True,
        "decode_tok_s": round(dec.get("decode_tok_s", 0.0), 2) if dec.get("ok") else None,
        "ttft_s": round(dec.get("ttft_s", 0.0), 3) if dec.get("ok") else None,
        "gen_tokens": round(dec.get("gen_tokens", 0.0), 1) if dec.get("ok") else None,
        "prefill_ttft_s": round(pre.get("ttft_s", 0.0), 3) if pre.get("ok") else None,
        "prefill_tok_s": round(pre.get("prefill_tok_s", 0.0), 1) if pre.get("ok") else None,
        "cudagraphs_captured": bool(cg_ok) and not cg_fail,
        "cudagraph_error": bool(cg_fail),
        "decode_error": dec.get("error"),
        "prefill_error": pre.get("error"),
    }
    print("=== REPORT ===", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh)
    if cg_fail:
        print("[bench] ⚠ CUDA GRAPHS NO CAPTURADOS -> modo eager (lento). "
              "Causa probable: mutación de buffers en forward compilado.", flush=True)
    elif cg_ok:
        print("[bench] ✓ CUDA graphs capturados OK.", flush=True)
    else:
        print("[bench] ? sin señal clara de cudagraphs en logs (revisar).", flush=True)

    # Exit 0 si health OK y al menos una métrica salió; el análisis es offline.
    sys.exit(0 if (dec.get("ok") or pre.get("ok")) else 1)


if __name__ == "__main__":
    main()
