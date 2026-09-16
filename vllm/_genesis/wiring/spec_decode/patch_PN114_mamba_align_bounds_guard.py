# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch PN114 — Mamba align copy bounds guard (vllm#35288 fix).

Fixes CUDA illegal memory access and cache corruption in MTP speculative
decoding with concurrency >= 4 on hybrid models (Qwen3.5 / Qwen3.8 / GDN).

Issue:
------
GitHub vllm-project/vllm#35288:
"MTP speculative decoding produces corrupted output at concurrency >= 4 (V1 engine)"

Root cause:
-----------
In `vllm/v1/worker/mamba_utils.py`:
During spec decode postprocessing on hybrid models (`postprocess_mamba_fused_kernel`),
when concurrency >= 4, requests that are starting, prefilled, or non-aligned can
have unallocated or negative block table indices (`src_block_idx < 0`,
`dest_block_idx < 0`, or `block_table[dst_col] == -1`).
The kernel was calling `_copy_mamba_state_block` without bounds checks, which loaded
from `block_table_base - 1` and computed `state_base_addr + (-1) * state_block_stride`.
Writing or reading to this out-of-bounds GPU address triggered CUDA illegal memory
access and corrupted the mamba recurrent state across concurrent sequences in the
batch, producing corrupted tokens (repeating token loops) or premature stops.

Fix:
----
Hay DOS clases de guarda, y no son igual de seguras:

A) Guardas ARITMETICAS (siempre activas).
   `dest_block_idx = aligned_new_computed // block_size - 1` vale -1 cuando el
   estado todavia cabe en el primer bloque, o sea cuando no hay frontera previa
   de la cual copiar. Saltear ahi es el caso "fresh" y es correcto por
   definicion: no se pierde estado porque no habia estado anterior. Cubre lo que
   describe vllm#35288.
     - `src_block_idx < 0` / `dest_block_idx < 0` en postprocess_mamba_fused_kernel
     - `src_col < 0` / `dst_col < 0` y `dest_block_id < 0` en _copy_mamba_state_block

B) Guardas sobre el VALOR DEL BLOCK TABLE (opt-in, default OFF).
   `block_table[src_col] == -1` con `src_col >= 0` NO es el caso fresh: es una
   inconsistencia, el block table tiene un hueco donde el codigo espera una
   asignacion. Saltear ahi deja el bloque de estado recurrente destino con lo
   que hubiera dejado el inquilino anterior, o sea contaminacion de estado
   entre secuencias: el MISMO sintoma que el issue (loops de tokens, cortes
   prematuros) pero silencioso, sin crash que lo delate.
   Por eso van apagadas por defecto. Se activan con
   `GENESIS_PN114_GUARD_BLOCK_TABLE=1` si efectivamente aparece el illegal
   memory access, entendiendo que cambian un fallo ruidoso por uno callado.
     - `src_block_id < 0` en las ramas DS conv y SD conv
     - `actual_src_block_id < 0` en la rama de estado temporal

Author: Genesis PN114 (vllm#35288 fix)
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)

log = logging.getLogger("genesis.wiring.pn114_mamba_align_bounds_guard")

GENESIS_PN114_MARKER = "Genesis PN114 mamba align copy bounds guard (vllm#35288)"
GENESIS_PN114_BT_MARKER = "Genesis PN114 mamba align block-table guard (vllm#35288)"


def _is_enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN114_MAMBA_ALIGN_BOUNDS_GUARD", "1"
    ).strip().lower() in ("1", "true", "yes", "on")


# ─── Anchor 1: _copy_mamba_state_block column & dest_block_id guard ─────────
ANCHOR_OLD_1 = (
    "    block_table_base = block_table_typed + bt_row_idx * block_table_stride_req\n"
    "\n"
    "    # Widen block ids to int64 before they reach `block_id * state_block_stride`\n"
    "    # below: state_block_stride can exceed 2**31 bytes for large mamba caches,\n"
    "    # and Triton would otherwise do the multiply in int32 and wrap.\n"
    "    dest_block_id = tl.load(block_table_base + dst_col).to(tl.int64)\n"
    "    dst_addr = state_base_addr + dest_block_id * state_block_stride"
)

ANCHOR_NEW_1 = (
    "    block_table_base = block_table_typed + bt_row_idx * block_table_stride_req\n"
    "\n"
    "    # [Genesis PN114 vllm#35288] Bounds guard for mamba align copy:\n"
    "    # Under MTP / spec decode with concurrency >= 4, unallocated blocks (-1) or\n"
    "    # invalid column offsets trigger CUDA illegal memory access and corrupt state.\n"
    "    if src_col < 0 or dst_col < 0:\n"
    "        return\n"
    "\n"
    "    # Widen block ids to int64 before they reach `block_id * state_block_stride`\n"
    "    # below: state_block_stride can exceed 2**31 bytes for large mamba caches,\n"
    "    # and Triton would otherwise do the multiply in int32 and wrap.\n"
    "    dest_block_id = tl.load(block_table_base + dst_col).to(tl.int64)\n"
    "    if dest_block_id < 0:\n"
    "        return\n"
    "    dst_addr = state_base_addr + dest_block_id * state_block_stride"
)

# ─── Anchor 2: _copy_mamba_state_block DS conv layout src_block_id guard ────
ANCHOR_OLD_2 = (
    "    if CONV_STATE_DIM_FIRST and is_conv_state:\n"
    "        # DS conv layout: state_len is the slide axis; copy per dim row.\n"
    "        src_block_id = tl.load(block_table_base + src_col).to(tl.int64)\n"
    "        dim_rows = tl.load(state_dim_row_count_ptr + state_idx)"
)

ANCHOR_NEW_2 = (
    "    if CONV_STATE_DIM_FIRST and is_conv_state:\n"
    "        # DS conv layout: state_len is the slide axis; copy per dim row.\n"
    "        src_block_id = tl.load(block_table_base + src_col).to(tl.int64)\n"
    "        if src_block_id < 0:\n"
    "            return\n"
    "        dim_rows = tl.load(state_dim_row_count_ptr + state_idx)"
)

# ─── Anchor 3: _copy_mamba_state_block SD conv layout src_block_id guard ────
ANCHOR_OLD_3 = (
    "    if is_conv_state:\n"
    "        # SD conv: copy\n"
    "        #   state[bt[src_col], token_bias:] ->\n"
    "        #   state[bt[dst_col], :conv_width - token_bias]\n"
    "        src_block_id = tl.load(block_table_base + src_col).to(tl.int64)\n"
    "        src_offset = token_bias.to(tl.int64) * state_inner_size * state_elem_size"
)

ANCHOR_NEW_3 = (
    "    if is_conv_state:\n"
    "        # SD conv: copy\n"
    "        #   state[bt[src_col], token_bias:] ->\n"
    "        #   state[bt[dst_col], :conv_width - token_bias]\n"
    "        src_block_id = tl.load(block_table_base + src_col).to(tl.int64)\n"
    "        if src_block_id < 0:\n"
    "            return\n"
    "        src_offset = token_bias.to(tl.int64) * state_inner_size * state_elem_size"
)

# ─── Anchor 4: _copy_mamba_state_block temporal state actual_src_block_id ───
ANCHOR_OLD_4 = (
    "    # Temporal state: copy state[bt[src_col + token_bias]] -> state[bt[dst_col]]\n"
    "    actual_src_block_id = tl.load(block_table_base + src_col + token_bias).to(tl.int64)\n"
    "    src_addr = state_base_addr + actual_src_block_id * state_block_stride"
)

ANCHOR_NEW_4 = (
    "    # Temporal state: copy state[bt[src_col + token_bias]] -> state[bt[dst_col]]\n"
    "    actual_src_block_id = tl.load(block_table_base + src_col + token_bias).to(tl.int64)\n"
    "    if actual_src_block_id < 0:\n"
    "        return\n"
    "    src_addr = state_base_addr + actual_src_block_id * state_block_stride"
)

# ─── Anchor 5: postprocess_mamba_fused_kernel src_block_idx / dest_block_idx
ANCHOR_OLD_5 = (
    "    # Skip no-op self-copy.\n"
    "    if src_block_idx == dest_block_idx and accept_token_bias == 0:\n"
    "        return\n"
    "\n"
    "    bt_row_idx = batch_idx if HAS_IDX_MAPPING else req_idx"
)

ANCHOR_NEW_5 = (
    "    # Skip no-op self-copy.\n"
    "    if src_block_idx == dest_block_idx and accept_token_bias == 0:\n"
    "        return\n"
    "\n"
    "    # [Genesis PN114 vllm#35288] Bounds guard: skip uninitialized or invalid block indices\n"
    "    if src_block_idx < 0 or dest_block_idx < 0:\n"
    "        return\n"
    "\n"
    "    bt_row_idx = batch_idx if HAS_IDX_MAPPING else req_idx"
)


def _guard_block_table_values() -> bool:
    """Guardas sobre el valor del block table (clase B). Default OFF: ver docstring."""
    return os.environ.get(
        "GENESIS_PN114_GUARD_BLOCK_TABLE", "0"
    ).strip().lower() in ("1", "true", "yes", "on")


def _make_patcher_arithmetic() -> TextPatcher | None:
    """Clase A — guardas aritmeticas. Siempre. Todas required=True."""
    target = resolve_vllm_file("v1/worker/mamba_utils.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN114 mamba align: guardas aritmeticas (vllm#35288)",
        target_file=str(target),
        marker=GENESIS_PN114_MARKER,
        sub_patches=[
            TextPatch(
                name="pn114_col_and_dst_guard",
                anchor=ANCHOR_OLD_1,
                replacement=ANCHOR_NEW_1,
                required=True,
            ),
            TextPatch(
                name="pn114_kernel_indices_guard",
                anchor=ANCHOR_OLD_5,
                replacement=ANCHOR_NEW_5,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "PN114 vllm#35288",
        ],
    )


def _make_patcher_block_table() -> TextPatcher | None:
    """Clase B — guardas sobre el VALOR del block table. Marker propio.

    Van en su propio TextPatcher para que encender
    GENESIS_PN114_GUARD_BLOCK_TABLE despues de un arranque sin ellas las
    aplique igual: con un marker compartido, el de la clase A ya estaria
    escrito y estas no se reintentarian nunca.
    """
    target = resolve_vllm_file("v1/worker/mamba_utils.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN114 mamba align: guardas de block-table (vllm#35288)",
        target_file=str(target),
        marker=GENESIS_PN114_BT_MARKER,
        sub_patches=[
            TextPatch(
                name="pn114_ds_conv_src_guard",
                anchor=ANCHOR_OLD_2,
                replacement=ANCHOR_NEW_2,
                required=True,
            ),
            TextPatch(
                name="pn114_sd_conv_src_guard",
                anchor=ANCHOR_OLD_3,
                replacement=ANCHOR_NEW_3,
                required=True,
            ),
            TextPatch(
                name="pn114_temporal_src_guard",
                anchor=ANCHOR_OLD_4,
                replacement=ANCHOR_NEW_4,
                required=True,
            ),
        ],
        upstream_drift_markers=[],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN114")
    log_decision("PN114", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher_arithmetic()
    if patcher is None:
        return "skipped", "mamba_utils.py not found"
    result, failure = patcher.apply()
    if result == TextPatchResult.SKIPPED:
        msg = failure.reason if failure else "anchor not found"
        return "skipped", f"{msg} — probablemente upstream ya lo absorbio"
    if result not in (TextPatchResult.APPLIED, TextPatchResult.IDEMPOTENT):
        return "failed", failure.reason if failure else "unknown failure"

    extra = " (guardas de block-table OFF; GENESIS_PN114_GUARD_BLOCK_TABLE=1 las activa)"
    if _guard_block_table_values():
        bt = _make_patcher_block_table()
        if bt is not None:
            bt_result, bt_failure = bt.apply()
            if bt_result in (TextPatchResult.APPLIED, TextPatchResult.IDEMPOTENT):
                extra = (
                    " + guardas de block-table ACTIVAS: saltean la copia si el "
                    "bloque origen esta sin asignar, lo que puede dejar estado "
                    "recurrente rancio en el destino"
                )
            else:
                extra = (
                    " — las guardas de block-table NO aplicaron: "
                    f"{bt_failure.reason if bt_failure else 'anchor no encontrado'}"
                )

    if result == TextPatchResult.IDEMPOTENT:
        return "applied", f"ya aplicado (idempotente){extra}"
    return "applied", f"PN114 aplicado: guardas aritmeticas de limites{extra}"
