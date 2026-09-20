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

# ─────────────── model runner v2 (DFlash2): mamba_hybrid.py ───────────────
# El v2 no pasa por preprocess_mamba/postprocess_mamba_all: migra el estado desde
# MambaHybridModelState, llamando directo a los kernels fusionados. Sin estos ganchos el
# salteo de bias>0 (pn122_mu_skip, que SI le llega) dejaba el estado GDN viejo en cada borde
# de bloque: la salida degeneraba a los ~1000 tokens. En v0.27.1 el archivo no existe.
MH_IMPORT_OLD = "from vllm.v1.worker.mamba_utils import (\n"
MH_IMPORT_NEW = _IMP + MH_IMPORT_OLD
MH_PRE_OLD = (
    "        ctx.run_fused_precopy(\n"
    "            num_reqs,\n"
    "            self._mamba_state_idx_gpu,\n"
)
MH_PRE_NEW = (
    "        if _g122.activo():  # " + MARKER + "\n"
    "            _g122.v2_pre(\n"
    "                ctx, kv_cache_config,\n"
    "                self.vllm_config.compilation_config.static_forward_context,\n"
    "                num_reqs, input_batch.idx_mapping, self._mamba_state_idx_gpu,\n"
    "                self._mamba_src_col_gpu, self._mamba_src_off_gpu,\n"
    "            )\n"
    + MH_PRE_OLD
)
MH_POST_OLD = (
    "            self._mamba_ctx.run_fused_postprocess_align(\n"
)
MH_POST_NEW = (
    "            if _g122.activo():  # " + MARKER + "\n"
    "                _g122.v2_post(\n"
    "                    self._mamba_ctx, num_reqs, self.num_accepted_tokens_gpu,\n"
    "                    self._mamba_state_idx_gpu, num_computed_tokens, idx_mapping,\n"
    "                )\n"
    + MH_POST_OLD
)

# ─────────────── v0.29.0: el early-return que marcaba el request antes de tiempo ───────────────
# PN122 pone `num_speculative_blocks = 0`: ese es TODO su beneficio (libera K bloques de estado
# GDN por request del pool). En v0.29.0 upstream agrego una linea adentro del early-return de
# `MambaManager.allocate_new_blocks`:
#
#     if num_required_blocks <= len(req_blocks) and not has_partial_hit:
#         self._allocated_block_reqs.add(request_id)      # <-- nueva en v0.29.0
#         return []
#
# Unas lineas mas abajo, `blocks_allocated = request_id in self._allocated_block_reqs` decide
# como se fija `last_state_block_idx`, y ESE indice es el que `remove_skipped_blocks` usa para
# liberar el bloque de estado viejo y reemplazarlo por el bloque nulo.
#
# Con K > 0 el early-return casi no se toma, porque `num_required_blocks` lleva los K bloques de
# sobra sumados. Con K = 0 se toma seguido: el request queda marcado antes de tiempo,
# `last_state_block_idx` apunta al bloque equivocado y se libera un bloque que todavia tiene el
# estado GDN vivo. El paso siguiente lee estado nulo, el modelo se corrompe y emite EOS — que es
# exactamente lo que se medio (2 tokens con PN122 prendido, 40 con GENESIS_PN122_SIN_LIBERAR=1,
# que es la misma cinta y el mismo kernel pero conservando los bloques).
#
# Saltear ese `add` con la cinta activa es inocuo: la marca existe para no volver a sumar
# `num_speculative_blocks` a un request ya alocado, y con PN122 ese numero es 0.
MGR_IMPORT_OLD = "from vllm.v1.core.block_pool import BlockPool\n"
MGR_IMPORT_NEW = (
    MGR_IMPORT_OLD
    + "from vllm._genesis import gdn_cinta as _g122  # " + MARKER + "\n"
)

MGR_EARLY_OLD = (
    "            if num_required_blocks <= len(req_blocks) and not has_partial_hit:\n"
    "                self._allocated_block_reqs.add(request_id)\n"
    "                return []\n"
)
MGR_EARLY_NEW = (
    "            if num_required_blocks <= len(req_blocks) and not has_partial_hit:\n"
    "                if not _g122.activo():  # " + MARKER + "\n"
    "                    self._allocated_block_reqs.add(request_id)\n"
    "                return []\n"
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
    # Opcional a proposito: estas dos anclas solo existen en v0.29.0+. En v0.27.1 no hay nada
    # que arreglar y los sub-parches se saltan sin ruido (required=False).
    ("v1/core/single_type_kv_cache_manager.py", [
        ("pn122_mgr_import", MGR_IMPORT_OLD, MGR_IMPORT_NEW, False),
        ("pn122_mgr_early_return", MGR_EARLY_OLD, MGR_EARLY_NEW, False),
    ]),
    ("v1/worker/mamba_utils.py", [
        ("pn122_mu_import", MU_IMPORT_OLD, MU_IMPORT_NEW),
        ("pn122_mu_skip", MU_SKIP_OLD, MU_SKIP_NEW),
        ("pn122_mu_slots", MU_SLOTS_OLD, MU_SLOTS_NEW),
        ("pn122_mu_pre", MU_PRE_OLD, MU_PRE_NEW),
        ("pn122_mu_post", MU_POST_OLD, MU_POST_NEW),
    ]),
    ("v1/worker/gpu/model_states/mamba_hybrid.py", [
        ("pn122_mh_import", MH_IMPORT_OLD, MH_IMPORT_NEW),
        ("pn122_mh_pre", MH_PRE_OLD, MH_PRE_NEW),
        ("pn122_mh_post", MH_POST_OLD, MH_POST_NEW),
    ]),
]


_SOLO_V2 = {"v1/worker/gpu/model_states/mamba_hybrid.py"}


def _patchers() -> list[TextPatcher] | None:
    out = []
    for rel, subs in _PATCHES:
        target = resolve_vllm_file(rel)
        if target is None:
            if rel in _SOLO_V2:   # v0.27.1 no tiene model runner v2
                continue
            return None
        out.append(TextPatcher(
            patch_name=f"PN122 cinta GDN ({rel.rsplit('/', 1)[-1]})",
            target_file=str(target),
            marker=MARKER,
            sub_patches=[TextPatch(name=e[0], anchor=e[1], replacement=e[2],
                                   required=(e[3] if len(e) > 3 else True))
                         for e in subs],
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
