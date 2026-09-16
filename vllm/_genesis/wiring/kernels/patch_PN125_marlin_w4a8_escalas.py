# SPDX-License-Identifier: Apache-2.0
"""PN125 — Marlin W4A8-INT8 con checkpoints de escalas negativas (AutoRound).

Con ``VLLM_MARLIN_INPUT_DTYPE=int8`` el kernel int8 de Marlin toma las escalas
por grupo como int16 sin signo relativas a ``s.max()``. AutoRound guarda el
signo en la escala (50,5% negativas en noon-at-cgn/Qwen3.8-27B-Uncensored-
W4A16-AutoRound) y el modelo sale "!!!!" sin error ni aviso: medido 3.754% de
error de salida en una capa real contra 0,03% en W4A16.

Inserta ``positivizar`` (``vllm/_genesis/marlin_w4a8_escalas.py``) justo antes
del repack en ``MarlinLinearKernel.process_weights_after_loading``: los grupos
negativos pasan a ``|s|`` con ``q -> 16 - q``. Solo con act int8 + uint4b8.
"""

from __future__ import annotations

import logging

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

log = logging.getLogger("genesis.wiring.pn125")

GENESIS_PN125_MARKER = "[Genesis PN125: escalas positivas para Marlin W4A8]"

ANCHOR_OLD = (
    "        self._transform_param(layer, self.w_q_name, transform_w_q)\n"
    "        self._transform_param(layer, self.w_s_name, transform_w_s)\n"
)

ANCHOR_NEW = (
    "        # " + GENESIS_PN125_MARKER + "\n"
    "        from vllm._genesis.marlin_w4a8_escalas import positivizar as _g125\n"
    "        _g125(layer, self.w_q_name, self.w_s_name, c)\n"
    "        self._transform_param(layer, self.w_q_name, transform_w_q)\n"
    "        self._transform_param(layer, self.w_s_name, transform_w_s)\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("model_executor/kernels/linear/mixed_precision/marlin.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN125 escalas positivas Marlin W4A8",
        target_file=str(target),
        marker=GENESIS_PN125_MARKER,
        sub_patches=[
            TextPatch(name="pn125_positivizar", anchor=ANCHOR_OLD,
                      replacement=ANCHOR_NEW, required=True),
        ],
        upstream_drift_markers=[],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN125")
    log_decision("PN125", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root no localizable"
    patcher = _make_patcher()
    if patcher is None:
        return "failed", "marlin.py no encontrado"
    result, failure = patcher.apply()
    return result_to_wiring_status(
        result, failure,
        applied_message="escalas negativas corregidas antes del repack de Marlin",
        patch_name=patcher.patch_name)


__all__ = ["apply", "GENESIS_PN125_MARKER", "ANCHOR_OLD", "ANCHOR_NEW"]
