# SPDX-License-Identifier: Apache-2.0
"""PN129 — buffer de trabajo del DECODE de FlashInfer sin memoria pinned.

Bug (vllm#40756, hilo del 2026-08-17; fix de club-3090): ``plan()`` de
FlashInfer reusa UN buffer host pinned por wrapper y lo copia a la GPU en
forma asincronica sin proteccion. El drafter MTP re-planifica el wrapper de
decode K-1 veces por paso: con >=4 requests MTP en vuelo, un plan viejo le da
al merge de split-KV una fila basura -> Xid 31 MMU fault VIRT_READ.

Visto en este equipo: Xid 31 VIRT_READ el 2026-09-13 23:03 (python3, GPU 0f).
Corremos max-num-seqs 10 y opencode lanza hasta 6 subagentes a la vez.

Arreglo: ``pin_memory=True,`` -> ``pin_memory=False,`` en las reservas
``_pin_memory_int_workspace_buffer`` de ``flashinfer/decode.py`` (club-3090:
necesario y suficiente; copia sincronica del plan en cada paso, velocidad
async completa, c=4 425 s sin errores). Guarda: la cantidad de ``pin_memory=
True,`` tiene que coincidir con la de reservas del buffer, si no se aborta.
"""

from __future__ import annotations

import os
import re

MARKER = "# [Genesis PN129: decode de FlashInfer sin pin (vllm#40756)]"
TOKEN = "pin_memory=True,"
ALLOC_RE = re.compile(r"_pin_memory_int_workspace_buffer[a-z_]* = torch\.empty\(")


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN129")
    log_decision("PN129", decision, reason)
    if not decision:
        return "skipped", reason
    try:
        import importlib.util
        spec = importlib.util.find_spec("flashinfer")
        if spec is None or not spec.origin:
            return "skipped", "flashinfer no instalado"
        path = os.path.join(os.path.dirname(spec.origin), "decode.py")
    except Exception as e:
        return "skipped", f"flashinfer no localizable: {e}"
    try:
        src = open(path, encoding="utf-8").read()
    except OSError as e:
        return "failed", f"no se puede leer {path}: {e}"
    if MARKER in src:
        return "applied", "ya aplicado"
    pins = src.count(TOKEN)
    allocs = len(ALLOC_RE.findall(src))
    if pins == 0:
        return "skipped", "decode.py sin buffers pinned"
    if pins != allocs:
        return "failed", f"{pins} pin_memory=True contra {allocs} reservas del buffer: deriva, no se toca"
    nuevo = src.replace(TOKEN, "pin_memory=False,")
    nuevo = MARKER + "\n" + nuevo
    try:
        compile(nuevo, path, "exec")
    except SyntaxError as e:
        return "failed", f"sintaxis tras el parche: {e}"
    with open(path, "w", encoding="utf-8") as f:
        f.write(nuevo)
    return "applied", f"decode.py: {pins} buffers de trabajo sin pin"
