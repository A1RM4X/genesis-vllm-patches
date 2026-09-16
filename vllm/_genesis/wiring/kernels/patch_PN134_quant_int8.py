# SPDX-License-Identifier: Apache-2.0
"""PN134 — cuantizacion de activacion int8 con el factor global fusionado.

Ver ``vllm._genesis.quant_int8``. Engancha en ``apply_gptq_marlin_linear`` de
``marlin_utils.py``, que hoy hace la cuantizacion y despues multiplica las escalas por el factor
global de la capa en un segundo kernel.
"""

from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN134: quant int8 con factor global]"

OLD = (
    "        reshaped_x, a_scales = marlin_quant_input(reshaped_x, input_dtype)\n"
    "        a_scales = a_scales * input_global_scale\n"
)
NEW = (
    "        from vllm._genesis import quant_int8 as _g134  # " + MARKER + "\n"
    "        if _g134.aplicable(reshaped_x, input_global_scale):  # " + MARKER + "\n"
    "            reshaped_x, a_scales = _g134.cuantizar(reshaped_x, input_global_scale)\n"
    "        else:  # " + MARKER + "\n"
    "            reshaped_x, a_scales = marlin_quant_input(reshaped_x, input_dtype)\n"
    "            a_scales = a_scales * input_global_scale\n"
)


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN134")
    log_decision("PN134", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root no localizable"
    target = resolve_vllm_file("model_executor/layers/quantization/utils/marlin_utils.py")
    if target is None:
        return "failed", "marlin_utils.py no encontrado"
    p = TextPatcher(
        patch_name="PN134 quant int8 con factor global", target_file=str(target), marker=MARKER,
        sub_patches=[TextPatch(name="pn134_quant", anchor=OLD, replacement=NEW, required=True)],
        upstream_drift_markers=[])
    result, failure = p.apply()
    return result_to_wiring_status(result, failure,
                                   applied_message="cuantizacion int8 fusionada con el factor global",
                                   patch_name=p.patch_name)
