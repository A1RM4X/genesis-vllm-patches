# SPDX-License-Identifier: Apache-2.0
"""PN123 (diagnostico) — volcado de q/k/v de la atencion de Qwen3Next.

Ver ``vllm._genesis.volcado_qkv``.
"""

from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

MARKER = "[Genesis PN123: volcado qkv]"
IMPORT_OLD = "from vllm.logger import init_logger\n"
IMPORT_NEW = IMPORT_OLD + "from vllm._genesis import volcado_qkv as _g123  # " + MARKER + "\n"
FWD_OLD = "        attn_output = self.attn(q, k, v)\n"
FWD_NEW = (
    "        if _g123.activo():  # " + MARKER + "\n"
    "            _g123.volcar(q, k, v, int(self.attn.layer_name.split('layers.')[-1].split('.')[0]))\n"
    "        attn_output = self.attn(q, k, v)\n"
)


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN123")
    log_decision("PN123", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root no localizable"
    target = resolve_vllm_file("model_executor/models/qwen3_next.py")
    if target is None:
        return "failed", "qwen3_next.py no encontrado"
    p = TextPatcher(
        patch_name="PN123 volcado qkv", target_file=str(target), marker=MARKER,
        sub_patches=[TextPatch(name="pn123_import", anchor=IMPORT_OLD, replacement=IMPORT_NEW, required=True),
                     TextPatch(name="pn123_fwd", anchor=FWD_OLD, replacement=FWD_NEW, required=True)],
        upstream_drift_markers=[])
    result, failure = p.apply()
    return result_to_wiring_status(result, failure, applied_message="volcado qkv armado",
                                   patch_name=p.patch_name)
