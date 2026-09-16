# SPDX-License-Identifier: Apache-2.0
"""PN126 — rotacion de q/k (Hadamard o WUSH) despues de RoPE en Qwen3NextAttention.

Ver ``vllm._genesis.rot_qk``. Mismo punto de enganche que PN123: justo antes de
``self.attn(q, k, v)``, cuando q y k ya tienen q_norm/k_norm y RoPE.
"""

from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

MARKER = "[Genesis PN126: rotacion qk]"
IMPORT_OLD = "from vllm.logger import init_logger\n"
IMPORT_NEW = IMPORT_OLD + "from vllm._genesis import rot_qk as _g126  # " + MARKER + "\n"
FWD_OLD = "        attn_output = self.attn(q, k, v)\n"
FWD_NEW = (
    "        if _g126.activo():  # " + MARKER + "\n"
    "            _g126.rotar(q, k, self.attn.layer_name, self.num_kv_heads, self.head_dim)\n"
    "        attn_output = self.attn(q, k, v)\n"
)


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN126")
    log_decision("PN126", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root no localizable"
    target = resolve_vllm_file("model_executor/models/qwen3_next.py")
    if target is None:
        return "failed", "qwen3_next.py no encontrado"
    p = TextPatcher(
        patch_name="PN126 rotacion qk", target_file=str(target), marker=MARKER,
        sub_patches=[TextPatch(name="pn126_import", anchor=IMPORT_OLD, replacement=IMPORT_NEW, required=True),
                     TextPatch(name="pn126_fwd", anchor=FWD_OLD, replacement=FWD_NEW, required=True)],
        upstream_drift_markers=[])
    result, failure = p.apply()
    return result_to_wiring_status(result, failure, applied_message="rotacion de q/k armada",
                                   patch_name=p.patch_name)
