# SPDX-License-Identifier: Apache-2.0
"""PN122 — rollback del MTP en GDN con cinta (sin bloques especulativos).

Ver ``vllm._genesis.gdn_cinta`` para el metodo y las mediciones. Toca cuatro
archivos, todos como TEXT PATCH (el modelo corre en procesos worker):

* ``mamba/abstract.py``: ``num_speculative_blocks`` -> 0 y crear la cinta de
  cada capa al enlazar el KV cache.
* ``gdn/qwen_gdn_linear_attn.py``: el camino de decode spec usa el kernel con
  cinta.
* ``attention/backends/gdn_attn.py``: la metadata lleva el slot de cinta de
  cada request spec (buffer persistente para CUDA graph).
* ``v1/worker/mamba_utils.py``: slots por request en ``preprocess_mamba``; las
  copias ``align`` con bias > 0 materializan el estado desde la cinta en vez de
  leer una columna especulativa que ya no existe.
"""

from __future__ import annotations

import logging

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

log = logging.getLogger("genesis.wiring.pn122")

MARKER = "[Genesis PN122: rollback GDN con cinta]"
_IMP = "from vllm._genesis import gdn_cinta as _g122  # " + MARKER + "\n"

# ── abstract.py ──
ABS_IMPORT_OLD = "from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec\n"
ABS_IMPORT_NEW = ABS_IMPORT_OLD + _IMP
ABS_SPEC_OLD = (
    "            num_speculative_blocks=(\n"
    "                vllm_config.speculative_config.num_speculative_tokens\n"
    "                if vllm_config.speculative_config\n"
    "                else 0\n"
    "            ),\n"
)
ABS_SPEC_NEW = (
    "            # " + MARKER + " 0 con la cinta activa: el rollback del MTP\n"
    "            # ya no necesita K copias del estado en el pool.\n"
    "            num_speculative_blocks=_g122.num_speculative_blocks(vllm_config),\n"
)

# En v0.29.0 upstream reescribio la expresion: el atajo de num_speculative_tokens subio a
# vllm_config y aparecio la rama de RecoverSSM. El reemplazo es el mismo — la cuenta de
# upstream la reproduce `gdn_cinta.num_speculative_blocks`, rama de RecoverSSM incluida.
ABS_SPEC_OLD_V029 = (
    "            num_speculative_blocks=(\n"
    "                0\n"
    "                if vllm_config.cache_config.use_kda_recoverssm\n"
    "                else vllm_config.num_speculative_tokens\n"
    "            ),\n"
)
ABS_BIND_OLD = "        self.kv_cache = tuple(states)\n"
ABS_BIND_NEW = (
    "        self.kv_cache = tuple(states)\n"
    "        if _g122.activo():  # " + MARKER + "\n"
    "            _g122.enlazar(self, pages.device)\n"
)

# ── qwen_gdn_linear_attn.py ──
GDN_IMPORT_OLD = "from vllm import envs\n"
GDN_IMPORT_NEW = GDN_IMPORT_OLD + _IMP
GDN_SPEC_OLD = (
    "        # 2.1: Process the multi-query part\n"
    "        if spec_sequence_masks is not None:\n"
    "            core_attn_out_spec, last_recurrent_state = (\n"
)
GDN_SPEC_NEW = (
    "        # 2.1: Process the multi-query part\n"
    "        if spec_sequence_masks is not None and _g122.activo() and not (\n"
    "            _g122.sync_bits() & 512\n"
    "        ):\n"
    "            # " + MARKER + "\n"
    "            core_attn_out_spec, last_recurrent_state = _g122.spec_update(\n"
    "                self, self.A_log, a, b, self.dt_bias, query_spec, key_spec,\n"
    "                value_spec, ssm_state,\n"
    "                spec_query_start_loc[: attn_metadata.num_spec_decodes + 1],\n"
    "                spec_state_indices_tensor, num_accepted_tokens,\n"
    "                attn_metadata.g122_slots,\n"
    "            )\n"
    "        elif spec_sequence_masks is not None:\n"
    "            core_attn_out_spec, last_recurrent_state = (\n"
)

# ── gdn_attn.py ──
ATT_IMPORT_OLD = "from vllm.v1.kv_cache_interface import MambaSpec\n"
ATT_IMPORT_NEW = ATT_IMPORT_OLD + _IMP
ATT_BUF_OLD = (
    "        self.num_accepted_tokens: torch.Tensor = torch.empty(\n"
    "            (self.decode_cudagraph_max_bs,),\n"
    "            dtype=torch.int32,\n"
    "            device=device,\n"
    "        )\n"
)
ATT_BUF_NEW = ATT_BUF_OLD + (
    "        # " + MARKER + " slot de cinta por request spec\n"
    "        self._g122_slots: torch.Tensor = torch.zeros(\n"
    "            (self.decode_cudagraph_max_bs,),\n"
    "            dtype=torch.int32,\n"
    "            device=device,\n"
    "        )\n"
)
ATT_CALC_OLD = (
    "        # Prepare per-request tensors for cudagraph. m.num_actual_tokens is\n"
)
ATT_CALC_NEW = (
    "        # " + MARKER + " slots en el mismo orden que spec_state_indices\n"
    "        g122_slots = None\n"
    "        # Con num_speculative_blocks=0 hay UNA columna, pero el conv spec usa\n"
    "        # spec_state_indices_tensor.size(-1) como max_query_len (= K+1).\n"
    "        if (\n"
    "            _g122.activo()\n"
    "            and spec_state_indices_tensor is not None\n"
    "            and spec_state_indices_tensor.shape[-1] != self.num_spec + 1\n"
    "        ):\n"
    "            spec_state_indices_tensor = spec_state_indices_tensor[:, :1].expand(\n"
    "                -1, self.num_spec + 1\n"
    "            ).contiguous()\n"
    "        if _g122.activo() and spec_sequence_masks_cpu is not None:\n"
    "            _sl = _g122.slots_gpu()\n"
    "            _msk = spec_sequence_masks_cpu[: _sl.shape[0]]\n"
    "            # Decode puro: las filas spec son un prefijo del batch y alcanza\n"
    "            # con un slice (sin indexacion con mascara, que cuesta ops).\n"
    "            if num_spec_decodes == _msk.shape[0] or bool(_msk[:num_spec_decodes].all()):\n"
    "                g122_slots = _sl[:num_spec_decodes]\n"
    "            else:\n"
    "                g122_slots = _sl[: _msk.shape[0]][_msk]\n"
    + ATT_CALC_OLD
)
ATT_CG_OLD = (
    "            num_accepted_tokens = self.num_accepted_tokens[:batch_size]\n"
    "            num_accepted_tokens[num_spec_decodes:].fill_(1)\n"
)
ATT_CG_NEW = ATT_CG_OLD + (
    "            if g122_slots is not None:  # " + MARKER + "\n"
    "                self._g122_slots[:num_spec_decodes].copy_(\n"
    "                    g122_slots, non_blocking=True\n"
    "                )\n"
    "                g122_slots = self._g122_slots[:batch_size]\n"
    "                g122_slots[num_spec_decodes:].fill_(0)\n"
)
ATT_RET_OLD = (
    "            token_chunk_offset_ptr=token_chunk_offset_ptr,\n"
    "        )\n"
    "        return attn_metadata\n"
)
ATT_RET_NEW = (
    "            token_chunk_offset_ptr=token_chunk_offset_ptr,\n"
    "        )\n"
    "        attn_metadata.g122_slots = g122_slots  # " + MARKER + "\n"
    "        _g122.debug_builder(num_spec_decodes, num_accepted_tokens, g122_slots,\n"
    "                            spec_state_indices_tensor, spec_query_start_loc,\n"
    "                            m.seq_lens)\n"
    "        return attn_metadata\n"
)

# ── mamba_utils.py ──
MU_IMPORT_OLD = "from vllm.v1.worker.lora_model_runner_mixin import GPUInputBatch\n"
MU_IMPORT_NEW = MU_IMPORT_OLD + _IMP + (
    "# Constexpr global: con la cinta, las copias temporales con bias > 0 las hace\n"
    "# _g122.materializar_* (la columna especulativa ya no existe).\n"
    "_G122_CINTA = tl.constexpr(_g122.activo())\n"
)
MU_SKIP_OLD = (
    "    # Temporal state: copy state[bt[src_col + token_bias]] -> state[bt[dst_col]]\n"
)
MU_SKIP_NEW = (
    "    if _G122_CINTA:  # " + MARKER + "\n"
    "        if token_bias > 0:\n"
    "            return\n"
    + MU_SKIP_OLD
)
MU_SLOTS_OLD = (
    "    cleanup_mamba_state_idx(scheduler_output, mamba_state_idx)\n"
    "\n"
    "    copy_bufs.offset = 0\n"
)
MU_SLOTS_NEW = (
    "    cleanup_mamba_state_idx(scheduler_output, mamba_state_idx)\n"
    "    if _g122.activo():  # " + MARKER + "\n"
    "        assert fused is not None, 'PN122 requiere el precopy fusionado'\n"
    "        _g122.actualizar_slots(\n"
    "            scheduler_output, list(input_batch.req_ids),\n"
    "            input_batch.max_num_reqs, input_batch.device, requests,\n"
    "        )\n"
    "\n"
    "    copy_bufs.offset = 0\n"
)
MU_PRE_OLD = (
    "        fused.ctx.run_fused_precopy(\n"
    "            num_reqs=num_reqs,\n"
)
MU_PRE_NEW = (
    "        if _g122.activo():  # " + MARKER + "\n"
    "            _g122.materializar_pre(\n"
    "                fused.ctx, kv_cache_config, forward_context, num_reqs,\n"
    "                fused.state_idx, fused.src_col, fused.token_bias,\n"
    "            )\n"
    + MU_PRE_OLD
)
MU_POST_OLD = (
    "    ctx.run_fused_postprocess(\n"
    "        num_reqs=num_reqs,\n"
    "        num_accepted_tokens_gpu=num_accepted_tokens_gpu,\n"
)
MU_POST_NEW = (
    "    if _g122.activo():  # " + MARKER + "\n"
    "        _g122.materializar_post(\n"
    "            ctx, kv_cache_config, forward_context, num_reqs,\n"
    "            num_accepted_tokens_gpu, ctx.mamba_state_idx_buf,\n"
    "            ctx.num_scheduled_tokens_buf, ctx.num_computed_tokens_buf,\n"
    "            ctx.num_draft_tokens_buf,\n"
    "        )\n"
    + MU_POST_OLD
)

_PATCHES = [
    ("model_executor/layers/mamba/abstract.py", [
        ("pn122_abs_import", ABS_IMPORT_OLD, ABS_IMPORT_NEW),
        ("pn122_abs_spec_blocks", [ABS_SPEC_OLD, ABS_SPEC_OLD_V029], ABS_SPEC_NEW),
        ("pn122_abs_bind", ABS_BIND_OLD, ABS_BIND_NEW),
    ]),
    ("model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py", [
        ("pn122_gdn_import", GDN_IMPORT_OLD, GDN_IMPORT_NEW),
        ("pn122_gdn_spec", GDN_SPEC_OLD, GDN_SPEC_NEW),
    ]),
    ("v1/attention/backends/gdn_attn.py", [
        ("pn122_att_import", ATT_IMPORT_OLD, ATT_IMPORT_NEW),
        ("pn122_att_buf", ATT_BUF_OLD, ATT_BUF_NEW),
        ("pn122_att_calc", ATT_CALC_OLD, ATT_CALC_NEW),
        ("pn122_att_cg", ATT_CG_OLD, ATT_CG_NEW),
        ("pn122_att_ret", ATT_RET_OLD, ATT_RET_NEW),
    ]),
    ("v1/worker/mamba_utils.py", [
        ("pn122_mu_import", MU_IMPORT_OLD, MU_IMPORT_NEW),
        ("pn122_mu_skip", MU_SKIP_OLD, MU_SKIP_NEW),
        ("pn122_mu_slots", MU_SLOTS_OLD, MU_SLOTS_NEW),
        ("pn122_mu_pre", MU_PRE_OLD, MU_PRE_NEW),
        ("pn122_mu_post", MU_POST_OLD, MU_POST_NEW),
    ]),
]


def _patchers() -> list[TextPatcher] | None:
    out = []
    for rel, subs in _PATCHES:
        target = resolve_vllm_file(rel)
        if target is None:
            return None
        out.append(TextPatcher(
            patch_name=f"PN122 cinta GDN ({rel.rsplit('/', 1)[-1]})",
            target_file=str(target),
            marker=MARKER,
            sub_patches=[TextPatch(name=n, anchor=o, replacement=r, required=True)
                         for n, o, r in subs],
            upstream_drift_markers=[],
        ))
    return out


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN122")
    log_decision("PN122", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root no localizable"
    patchers = _patchers()
    if patchers is None:
        return "failed", "algun archivo objetivo no se encontro"
    for p in patchers:
        result, failure = p.apply()
        status, msg = result_to_wiring_status(
            result, failure, applied_message="ok", patch_name=p.patch_name)
        if status == "failed":
            return status, f"{p.patch_name}: {msg}"
    return "applied", "rollback del MTP en GDN con cinta (4 archivos)"


__all__ = ["apply", "MARKER", "_PATCHES"]
