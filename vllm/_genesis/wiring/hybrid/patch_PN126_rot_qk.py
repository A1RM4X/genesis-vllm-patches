# SPDX-License-Identifier: Apache-2.0
"""PN126 — rotacion Hadamard/FWHT ENTERA de q/k despues de RoPE en Qwen3NextAttention y DFlash.

Ver ``vllm._genesis.rot_qk``.
Puntos de enganche:
1) Qwen3NextAttention (qwen3_next.py): justo antes de ``self.attn(q, k, v)``.
2) DFlashQwen3Attention (qwen3_dflash.py): justo antes de ``self.attn(q, k, v)``.
3) precompute_and_store_context_kv (qwen3_dflash.py): rotacion de ``all_k_final``
   antes de llamar a ``attn.impl.do_kv_cache_update`` para que la KV int8 guarde k ya rotada.
"""

from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

MARKER = "[Genesis PN126: rotacion qk]"

# ── 1) qwen3_next.py ──
IMPORT_OLD = "from torch import nn\n"
IMPORT_NEW = IMPORT_OLD + "from vllm._genesis import rot_qk as _g126  # " + MARKER + "\n"
FWD_OLD = "        attn_output = self.attn(q, k, v)\n"
FWD_NEW = (
    "        if _g126.activo(self.head_dim):  # " + MARKER + "\n"
    "            _g126.rotar(q, k, self.attn.layer_name, self.num_kv_heads, self.head_dim)\n"
    "        attn_output = self.attn(q, k, v)\n"
)

# ── 2) qwen3_dflash.py ──
DF_IMPORT_OLD = "from collections.abc import Iterable\n"
DF_IMPORT_NEW = (
    "from collections.abc import Iterable\n"
    "from vllm._genesis import rot_qk as _g126  # " + MARKER + "\n"
)
DF_FWD_OLD = (
    "        q, k = self.rotary_emb(positions, q, k)\n"
    "\n"
    "        attn_output = self.attn(q, k, v)\n"
)
DF_FWD_NEW = (
    "        q, k = self.rotary_emb(positions, q, k)\n"
    "        if _g126.activo(self.head_dim):  # " + MARKER + "\n"
    "            _g126.rotar(q, k, self.layer_name, self.num_kv_heads, self.head_dim)\n"
    "        attn_output = self.attn(q, k, v)\n"
)
DF_PRE_OLD = "        all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)\n"
DF_PRE_NEW = (
    "        all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)\n"
    "        if _g126.activo(hd):  # " + MARKER + "\n"
    "            _g126.rotar_tensor(all_k_final, hd)\n"
)


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN126")
    log_decision("PN126", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root no localizable"

    target_next = resolve_vllm_file("model_executor/models/qwen3_next.py")
    if target_next is None:
        return "failed", "qwen3_next.py no encontrado"
    p1 = TextPatcher(
        patch_name="PN126 rotacion qk", target_file=str(target_next), marker=MARKER,
        sub_patches=[TextPatch(name="pn126_import", anchor=IMPORT_OLD, replacement=IMPORT_NEW, required=True),
                     TextPatch(name="pn126_fwd", anchor=FWD_OLD, replacement=FWD_NEW, required=True)],
        upstream_drift_markers=[])
    r1, f1 = p1.apply()

    target_df = resolve_vllm_file("model_executor/models/qwen3_dflash.py")
    # GENESIS_PN126_DFLASH=0 deja al borrador sin rotar. Se decide ACA y no en rot_qk.activo():
    # asi cambia el fuente parcheado y la cache de torch.compile no hornea la decision vieja.
    import os
    if target_df is not None and os.environ.get("GENESIS_PN126_DFLASH", "1") != "0":
        p2 = TextPatcher(
            patch_name="PN126 rotacion qk dflash", target_file=str(target_df), marker=MARKER,
            sub_patches=[
                TextPatch(name="pn126_df_import", anchor=DF_IMPORT_OLD, replacement=DF_IMPORT_NEW, required=True),
                TextPatch(name="pn126_df_fwd", anchor=DF_FWD_OLD, replacement=DF_FWD_NEW, required=True),
                TextPatch(name="pn126_df_pre", anchor=DF_PRE_OLD, replacement=DF_PRE_NEW, required=True),
            ],
            upstream_drift_markers=[])
        r2, f2 = p2.apply()
        if r2 == "failed":
            return "failed", f"dflash patch failed: {f2}"

    return result_to_wiring_status(r1, f1, applied_message="rotacion de q/k armada (target + dflash)",
                                   patch_name=p1.patch_name)
