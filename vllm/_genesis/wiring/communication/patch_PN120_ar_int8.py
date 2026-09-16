# SPDX-License-Identifier: Apache-2.0
"""PN120 — all-reduce de TP comprimido a INT8 en el prefill.

El perfil con nsys del prompt processing dio que el all-reduce de TP se lleva el
**24,4% del prefill**: 6.538 llamadas de 6,3 ms moviendo 76,7 MB cada una. Y no
hay margen del lado del kernel — una copia P2P pura de ese tamano tarda 6,06 ms,
o sea que NCCL ya corre al 96% del limite del PCIe 4.0 x8.

Lo unico que queda es mandar menos bytes. Ver `vllm._genesis.ar_int8` para el
detalle del metodo, las alternativas descartadas y los numeros.

Es un TEXT PATCH por el mismo motivo que PN118 y PN119: el modelo corre en
procesos WORKER donde `apply_all` no llega, asi que un monkeypatch nunca se
aplica.

El camino va TRAZADO, no como custom op opaco: la fusion de inductor sobre el
(des)cuantizado es lo que lo lleva de 1,16x a 1,72x.
"""

from __future__ import annotations

import logging

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

log = logging.getLogger("genesis.wiring.pn120")

GENESIS_PN120_MARKER = "[Genesis PN120: all-reduce INT8]"

# El import va al TOPE del archivo. Adentro del forward dispararia trabajo de
# import durante el trazado de dynamo, que es como se rompio PN119 la primera vez.
IMPORT_OLD = "from vllm.logger import init_logger\n"
IMPORT_NEW = (
    "from vllm.logger import init_logger\n"
    "# " + GENESIS_PN120_MARKER + " import a nivel de modulo a proposito\n"
    "from vllm._genesis import ar_int8 as _g120\n"
)

ANCHOR_OLD = (
    "        if self.reduce_results and self.tp_size > 1:\n"
    "            output = tensor_model_parallel_all_reduce(output_parallel)\n"
)

ANCHOR_NEW = (
    "        if self.reduce_results and self.tp_size > 1:\n"
    "            # " + GENESIS_PN120_MARKER + "\n"
    "            # Con M grande manda los parciales en int8 (la mitad de bytes)\n"
    "            # y suma en el dtype original. Debajo de GENESIS_PN120_M_MIN\n"
    "            # cae al all-reduce exacto: en decode el cuello es LATENCIA,\n"
    "            # no ancho de banda, y comprimir solo agregaria kernels.\n"
    "            if _g120.activo():\n"
    "                output = _g120.all_reduce_int8(output_parallel)\n"
    "            else:\n"
    "                output = tensor_model_parallel_all_reduce(output_parallel)\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("model_executor/layers/linear.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN120 all-reduce de TP en INT8",
        target_file=str(target),
        marker=GENESIS_PN120_MARKER,
        sub_patches=[
            TextPatch(name="pn120_import", anchor=IMPORT_OLD,
                      replacement=IMPORT_NEW, required=True),
            TextPatch(name="pn120_row_parallel", anchor=ANCHOR_OLD,
                      replacement=ANCHOR_NEW, required=True),
        ],
        upstream_drift_markers=[],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN120")
    log_decision("PN120", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root no localizable"
    patcher = _make_patcher()
    if patcher is None:
        return "failed", "linear.py no encontrado"
    result, failure = patcher.apply()
    return result_to_wiring_status(
        result, failure,
        applied_message="all-reduce de TP comprimido a INT8 en prefill",
        patch_name=patcher.patch_name)


__all__ = ["apply", "GENESIS_PN120_MARKER", "ANCHOR_OLD", "ANCHOR_NEW",
           "IMPORT_OLD", "IMPORT_NEW"]
