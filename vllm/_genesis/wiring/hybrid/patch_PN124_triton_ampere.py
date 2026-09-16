# SPDX-License-Identifier: Apache-2.0
"""PN124 — TRITON_ATTN rápido en Ampere para head_dim 256.

Ver ``vllm._genesis.triton_attn_ampere``. Dos archivos:

* ``v1/attention/ops/triton_unified_attention.py``: parámetros de lanzamiento
  del camino 2D (prefill) para SM8x + head 256.
* ``v1/attention/backends/triton_attn.py``: la llamada pasa por
  ``_g124.llamar``, que aplana el decode spec uniforme al kernel 3D y baja de
  configuración si el kernel no entra en la SRAM de Ampere.
"""

from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

MARKER = "[Genesis PN124: triton ampere]"
_IMP = "from vllm._genesis import triton_attn_ampere as _g124  # " + MARKER + "\n"

UA_IMPORT_OLD = "from vllm.platforms import current_platform\n"
UA_IMPORT_NEW = UA_IMPORT_OLD + _IMP
UA_PARAMS_OLD = (
    "    if tuned_large_head:\n"
    "        BLOCK_M = 32\n"
    "        BLOCK_Q = BLOCK_M // num_queries_per_kv\n"
    "        launch_num_warps = 8\n"
    "        launch_num_stages = 2\n"
)
UA_PARAMS_NEW = UA_PARAMS_OLD + (
    "    # " + MARKER + " el afinado de upstream es solo Blackwell y no entra\n"
    "    # en la SRAM de Ampere; estos parametros si (medido 2,84x en prefill).\n"
    "    _g124_amp = (\n"
    "        max_seqlen_q > 1\n"
    "        and num_queries_per_kv <= 16\n"
    "        and _g124.ampere_head256(head_size)\n"
    "    )\n"
    "    if _g124_amp:\n"
    "        BLOCK_M, _g124_tile, launch_num_stages, launch_num_warps = _g124.params()\n"
    "        BLOCK_Q = BLOCK_M // num_queries_per_kv\n"
)
UA_TILE_OLD = (
    "    if tuned_large_head:\n"
    "        TILE_SIZE_PREFILL = 128\n"
)
UA_TILE_NEW = UA_TILE_OLD + (
    "    if _g124_amp:  # " + MARKER + "\n"
    "        TILE_SIZE_PREFILL = _g124_tile\n"
)

BK_IMPORT_OLD = "from vllm.v1.attention.ops.triton_unified_attention import unified_attention\n"
BK_IMPORT_NEW = BK_IMPORT_OLD + _IMP
BK_CALL_OLD = (
    "        unified_attention(\n"
    "            q=query[:num_actual_tokens],\n"
)
BK_CALL_NEW = (
    "        _g124.llamar(  # " + MARKER + " aplana el decode spec al kernel 3D\n"
    "            unified_attention,\n"
    "            q=query[:num_actual_tokens],\n"
)

I4_IMPORT_OLD = "from vllm.platforms import current_platform\n"
I4_IMPORT_NEW = I4_IMPORT_OLD + _IMP
I4_PARAMS_OLD = (
    "    BLOCK_Q = BLOCK_M // num_queries_per_kv\n"
    "    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs\n"
    "    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0\n"
)
I4_PARAMS_NEW = (
    "    BLOCK_Q = BLOCK_M // num_queries_per_kv\n"
    "    # " + MARKER + " mismos parametros de Ampere en el kernel INT4 empaquetado\n"
    "    _g124_amp = (\n"
    "        max_seqlen_q > 1\n"
    "        and num_queries_per_kv <= 16\n"
    "        and _g124.ampere_head256(head_size)\n"
    "    )\n"
    "    _g124_launch = {}\n"
    "    if _g124_amp:\n"
    "        BLOCK_M, _g124_tile, _g124_st, _g124_w = _g124.params_int4()\n"
    "        BLOCK_Q = BLOCK_M // num_queries_per_kv\n"
    "        _g124_launch = {'num_stages': _g124_st, 'num_warps': _g124_w}\n"
    "    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs\n"
    "    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0\n"
)
I4_TILE_OLD = (
    "    TILE_SIZE_DECODE = _get_tile_size(\n"
    "        head_size, sliding_window_val, q.element_size(), is_prefill=False\n"
    "    )\n"
    "\n"
    "    use_3d = not (\n"
)
I4_TILE_NEW = (
    "    TILE_SIZE_DECODE = _get_tile_size(\n"
    "        head_size, sliding_window_val, q.element_size(), is_prefill=False\n"
    "    )\n"
    "    if _g124_amp:  # " + MARKER + "\n"
    "        TILE_SIZE_PREFILL = _g124_tile\n"
    "    if _g124.ampere_head256(head_size):  # " + MARKER + " decode 3D: 32 > 16\n"
    "        TILE_SIZE_DECODE = _g124.tile_decode_int4()\n"
    "\n"
    "    use_3d = not (\n"
)
I4_LAUNCH_OLD = (
    "        PACKING_FACTOR=packing_factor,\n"
    "    )\n"
    "\n"
    "    if use_3d:\n"
)
I4_LAUNCH_NEW = (
    "        PACKING_FACTOR=packing_factor,\n"
    "        **_g124_launch,  # " + MARKER + "\n"
    "    )\n"
    "\n"
    "    if use_3d:\n"
)

_PATCHES = [
    ("v1/attention/ops/triton_unified_attention.py", [
        ("pn124_ua_import", UA_IMPORT_OLD, UA_IMPORT_NEW),
        ("pn124_ua_params", UA_PARAMS_OLD, UA_PARAMS_NEW),
        ("pn124_ua_tile", UA_TILE_OLD, UA_TILE_NEW),
    ]),
    ("v1/attention/ops/int4_per_token_head.py", [
        ("pn124_i4_import", I4_IMPORT_OLD, I4_IMPORT_NEW),
        ("pn124_i4_params", I4_PARAMS_OLD, I4_PARAMS_NEW),
        ("pn124_i4_tile", I4_TILE_OLD, I4_TILE_NEW),
        ("pn124_i4_launch", I4_LAUNCH_OLD, I4_LAUNCH_NEW),
    ]),
    ("v1/attention/backends/triton_attn.py", [
        ("pn124_bk_import", BK_IMPORT_OLD, BK_IMPORT_NEW),
        ("pn124_bk_call", BK_CALL_OLD, BK_CALL_NEW),
    ]),
]


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN124")
    log_decision("PN124", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root no localizable"
    for rel, subs in _PATCHES:
        target = resolve_vllm_file(rel)
        if target is None:
            return "failed", f"{rel} no encontrado"
        p = TextPatcher(
            patch_name=f"PN124 triton ampere ({rel.rsplit('/', 1)[-1]})",
            target_file=str(target), marker=MARKER,
            sub_patches=[TextPatch(name=n, anchor=o, replacement=r, required=True) for n, o, r in subs],
            upstream_drift_markers=[])
        result, failure = p.apply()
        status, msg = result_to_wiring_status(result, failure, applied_message="ok", patch_name=p.patch_name)
        if status == "failed":
            return status, f"{p.patch_name}: {msg}"
    return "applied", "TRITON_ATTN afinado para Ampere + decode spec al kernel 3D"


__all__ = ["apply", "MARKER", "_PATCHES"]
