# SPDX-License-Identifier: Apache-2.0
"""PN133 — las lineales del modulo MTP en W8A8 (ver ``vllm._genesis.mtp_int8``).

Engancha al final de ``load_weights`` del modelo MTP: cuando terminaron de cargarse los pesos,
cuantiza las lineales del borrador a int8 con escala por canal de salida y les cambia el metodo
por uno que hace activacion int8 por token + GEMM int8 de cutlass.
"""

from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN133: MTP en int8]"

def _anclas(texto: str):
    """El load_weights del MTP lo tocan otros parches Genesis (P112, PN348), asi que el ancla se
    arma leyendo el archivo: se busca el ultimo `return loader.load_weights(...)` del modulo MTP."""
    import re
    m = None
    for m in re.finditer(r"( +)loader = AutoWeightsLoader\(self\)\n +return (loader\.load_weights\([^\n]*\))\n", texto):
        pass
    if m is None:
        return None, None
    sangria, llamada = m.group(1), m.group(2)
    viejo = m.group(0)
    nuevo = (f"{sangria}loader = AutoWeightsLoader(self)\n"
             f"{sangria}_cargados = {llamada}  # {MARKER}\n"
             f"{sangria}from vllm._genesis import mtp_int8 as _g133  # {MARKER}\n"
             f"{sangria}_g133.cuantizar(self)  # {MARKER}\n"
             f"{sangria}return _cargados  # {MARKER}\n")
    return viejo, nuevo

ARCHIVOS = ("model_executor/models/qwen3_5_mtp.py", "model_executor/models/qwen3_next_mtp.py")


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN133")
    log_decision("PN133", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root no localizable"
    hechos = []
    for rel in ARCHIVOS:
        target = resolve_vllm_file(rel)
        if target is None:
            continue
        viejo, nuevo = _anclas(open(str(target)).read())
        if viejo is None:
            hechos.append(("skipped", f"sin ancla en {rel.split('/')[-1]}"))
            continue
        p = TextPatcher(
            patch_name=f"PN133 MTP int8 ({rel.split('/')[-1]})", target_file=str(target), marker=MARKER,
            sub_patches=[TextPatch(name="pn133_load", anchor=viejo, replacement=nuevo, required=True)],
            upstream_drift_markers=[])
        result, failure = p.apply()
        hechos.append(result_to_wiring_status(result, failure,
                                              applied_message="lineales del MTP en int8",
                                              patch_name=p.patch_name))
    if not hechos:
        return "failed", "no se encontro ningun modelo MTP para parchear"
    ok = [h for h in hechos if h[0] == "applied"]
    if not ok:
        return hechos[0]
    return "applied", f"lineales del MTP en int8 en {len(ok)} modelo(s)"
