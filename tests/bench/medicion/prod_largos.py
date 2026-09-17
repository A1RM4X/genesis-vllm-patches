#!/usr/bin/env python3
"""Un pedido a la vez, con prompts de largo creciente. Mide TTFT y velocidad de decode.

El primer pedido tiene que ser LARGO: hay un latch en FlashInfer que se queda con el
_max_total_num_rows del primero y rompe el engine si arranca corto (ver la nota de memoria).
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

import os

HOST = os.environ.get("VLLM_HOST", "http://127.0.0.1:8320")
CLAVE = os.environ.get("VLLM_API_KEY", "<REDACTADO: clave rotada 2026-09-19>")
CAB = {"Content-Type": "application/json", "Authorization": f"Bearer {CLAVE}"}
URL = HOST + "/v1/chat/completions"
MODELO = sys.argv[1] if len(sys.argv) > 1 else None


def modelo() -> str:
    with urllib.request.urlopen(urllib.request.Request(HOST + "/v1/models", headers=CAB)) as r:
        return json.load(r)["data"][0]["id"]


def relleno(tokens: int, sal: str = "") -> str:
    """Texto de ~tokens tokens. Se usa prosa variada y no una palabra repetida, porque el
    prefix-cache y el propio modelo se comportan distinto con texto degenerado."""
    base = ("El sistema de inventario registra cada movimiento de mercaderia con su fecha, "
            "el deposito de origen, el de destino, el responsable de la operacion y el motivo "
            "declarado. Los ajustes por diferencia de conteo se asientan aparte y requieren la "
            "firma de un supervisor. Cuando el stock de un articulo cae por debajo del punto de "
            "reposicion se emite una solicitud automatica al proveedor habitual, salvo que el "
            "articulo este marcado como discontinuo. ")
    # ~1,4 tokens por palabra en castellano
    palabras = int(tokens / 1.4)
    texto = (base * (palabras // len(base.split()) + 2)).split()[:palabras]
    # El prefijo unico va ADELANTE a proposito: sin el, el prompt de 10k es un prefijo exacto del
    # de 50k y el prefix-cache devuelve TTFT de 0,86 s en vez de 20,7. Medir eso seria medir la
    # cache, no el prefill.
    return sal + " " + " ".join(texto)


def pedir(prompt: str, max_tokens: int, mdl: str) -> dict:
    """SIN streaming, a proposito.

    Con `stream: True` el cliente de Python procesa el SSE token por token y NO DA ABASTO: topea
    en ~40 tok/s y uno termina midiendo el cliente en vez del servidor. Con el mismo servidor,
    streaming daba 41,5 tok/s y sin streaming 119,7. El conteo sale de `usage`, que es el real.
    """
    cuerpo = json.dumps({
        "model": mdl, "max_tokens": max_tokens, "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    t0 = time.perf_counter()
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, cuerpo, CAB), timeout=1200))
    total = time.perf_counter() - t0
    return {"total": total, "tokens": r["usage"]["completion_tokens"],
            "prompt_tokens": r["usage"]["prompt_tokens"],
            "texto": r["choices"][0]["message"]["content"]}


def main() -> None:
    mdl = MODELO or modelo()
    print(f"  modelo: {mdl}\n")
    print(f"  {'pedido':>9}{'prompt tok':>11}{'prefill s':>10}{'total s':>9}{'tokens':>8}"
          f"{'decode tok/s':>14}{'prefill tok/s':>15}")
    import uuid
    PREG = ("\n\nAnaliza el texto anterior: explica el circuito completo de reposicion, que "
            "controles tiene, donde estan sus puntos debiles y como los mejorarias. Se extenso.")
    # el primero largo, a proposito (latch de FlashInfer)
    for largo in (50000, 1000, 5000, 10000, 50000):
        # DOS prompts distintos del MISMO largo: si se reusa el mismo, el segundo pedido acierta
        # el prefix-cache del primero y la resta da basura (con 50k daba prefill 20 s y total
        # 4,9 s). Distintos y del mismo largo, el costo de prefill es el mismo y la resta vale.
        pa = relleno(largo, f"[caso {uuid.uuid4().hex}] ") + PREG
        pb = relleno(largo, f"[caso {uuid.uuid4().hex}] ") + PREG
        a = pedir(pa, 1, mdl)          # prefill solo
        b = pedir(pb, 400, mdl)        # prefill + decode
        dec = (b["tokens"] - a["tokens"]) / max(b["total"] - a["total"], 1e-9)
        print(f"  {largo:>9}{b['prompt_tokens']:>11}{a['total']:>10.2f}{b['total']:>9.2f}"
              f"{b['tokens']:>8}{dec:>14.1f}{b['prompt_tokens'] / a['total']:>15.0f}")


if __name__ == "__main__":
    main()
