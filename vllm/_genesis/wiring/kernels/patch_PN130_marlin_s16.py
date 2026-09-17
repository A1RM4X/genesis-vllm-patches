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


# El workspace de Marlin tiene un slot de lock por BLOQUE y se dimensiona con
# sms * max_blocks_per_sm. Por defecto pide 1, o sea sms slots, y con eso el kernel no puede
# lanzar mas de un bloque por SM: se queda en 8 warps de 48 (17% de ocupacion) reservando toda
# la shared del SM aunque use 41 KB de 99. Pidiendo 2, el kernel puede duplicar la ocupacion — y
# si igual decide quedarse en 1, el workspace de mas son 82 enteros, nada.
WS_OLD = (
    "        self.workspace = marlin_make_workspace_new(\n"
    '            device, existing=getattr(self, "workspace", None)\n'
    "        )\n"
)
WS_NEW = (
    "        self.workspace = marlin_make_workspace_new(  # " + MARKER + "\n"
    "            device, max_blocks_per_sm=2,  # " + MARKER + "\n"
    '            existing=getattr(self, "workspace", None)\n'
    "        )\n"
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
    estado = result_to_wiring_status(result, failure,
                                     applied_message=f"Marlin s16 enganchado ({so})",
                                     patch_name=p.patch_name)

    # El workspace vive en OTRO archivo (el kernel de linear), no en marlin_utils.py, asi que va
    # en su propio patcher. Es opcional a proposito: si el anchor se mueve, PN130 sigue andando
    # con un bloque por SM en vez de quedar entero afuera.
    destino_ws = resolve_vllm_file(
        "model_executor/kernels/linear/mixed_precision/marlin.py")
    if destino_ws is not None:
        pw = TextPatcher(
            patch_name="PN130 workspace para 2 bloques/SM", target_file=str(destino_ws),
            marker=MARKER,
            sub_patches=[TextPatch(name="pn130_workspace", anchor=WS_OLD, replacement=WS_NEW,
                                   required=False)],
            upstream_drift_markers=[])
        pw.apply()

    return estado
