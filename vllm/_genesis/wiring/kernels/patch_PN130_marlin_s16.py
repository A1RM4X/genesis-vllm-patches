# SPDX-License-Identifier: Apache-2.0
"""PN130 — Marlin W4A8 propio con escalas int16 con signo (ver ``vllm._genesis.marlin_s16``).

Compila la extension (una vez, cacheada) y engancha dos puntos de
``marlin_utils.py``: el proceso de escalas (por ``|s|.max()``) y la llamada
``ops.marlin_gemm`` de ``apply_gptq_marlin_linear`` cuando las activaciones son
int8. Con PN130 cargado, PN125 no modifica los pesos.
"""

from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN130: Marlin W4A8 con escalas int16 con signo]"

IMPORT_OLD = "from vllm.logger import init_logger\n"
IMPORT_NEW = IMPORT_OLD + "from vllm._genesis import marlin_s16 as _g130  # " + MARKER + "\n"

ESC_OLD = (
    "def marlin_act_int8_process_scales(s: torch.Tensor):\n"
    "    a_scales_scale_factor = 1 / 4096 * s.max().float()\n"
)
ESC_NEW = (
    "def marlin_act_int8_process_scales(s: torch.Tensor):\n"
    "    if _g130.ACTIVO_Y_CARGADO:  # " + MARKER + "\n"
    "        return _g130.procesar_escalas(s)\n"
    "    a_scales_scale_factor = 1 / 4096 * s.max().float()\n"
)

GEMM_OLD = (
    "    output = ops.marlin_gemm(\n"
    "        reshaped_x,\n"
)
GEMM_NEW = (
    "    # " + MARKER + "\n"
    "    _gemm = (_g130.marlin_gemm\n"
    "             if (_g130.ACTIVO_Y_CARGADO and input_dtype == torch.int8)\n"
    "             else ops.marlin_gemm)\n"
    "    output = _gemm(\n"
    "        reshaped_x,\n"
)


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN130")
    log_decision("PN130", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root no localizable"
    target = resolve_vllm_file("model_executor/layers/quantization/utils/marlin_utils.py")
    if target is None:
        return "failed", "marlin_utils.py no encontrado"
    try:
        from vllm._genesis import marlin_s16
        so = marlin_s16.construir()
    except Exception as e:
        return "failed", f"no compila la extension: {str(e)[-400:]}"
    p = TextPatcher(
        patch_name="PN130 Marlin W4A8 escalas con signo", target_file=str(target), marker=MARKER,
        sub_patches=[TextPatch(name="pn130_import", anchor=IMPORT_OLD, replacement=IMPORT_NEW, required=True),
                     TextPatch(name="pn130_escalas", anchor=ESC_OLD, replacement=ESC_NEW, required=True),
                     TextPatch(name="pn130_gemm", anchor=GEMM_OLD, replacement=GEMM_NEW, required=True)],
        upstream_drift_markers=[])
    result, failure = p.apply()
    return result_to_wiring_status(result, failure, applied_message=f"Marlin s16 enganchado ({so})",
                                   patch_name=p.patch_name)
