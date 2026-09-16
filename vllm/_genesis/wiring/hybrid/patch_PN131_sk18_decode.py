# SPDX-License-Identifier: Apache-2.0
"""PN131 — decode de atencion entero (SK-18h, PTX) sobre KV int8_per_token_head.

Ver ``vllm._genesis.sk18_attn``. Engancha en ``triton_attn.py``:
* ``do_kv_cache_update``: escritura con layout propio (K por token, V por dimension,
  escalas int16) dentro de la misma reserva de 520 B por token-cabeza.
* ``forward``: decode (<= 5 tokens por pedido) por SK-18h; pasos con prefill por
  decuantizacion + Triton fp16.
* builder: ``query_start_loc_cpu`` en la metadata (sin sincronizar) y soporte de
  CUDA graph UNIFORM_BATCH: el decode uniforme (1 + tokens de MTP) va en FULL graph;
  los lanzamientos PTX son cuLaunchKernel y quedan grabados.
"""

from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN131: SK-18 decode entero]"

IMPORT_OLD = "from vllm.logger import init_logger\n"
IMPORT_NEW = IMPORT_OLD + "from vllm._genesis import sk18_attn as _g131  # " + MARKER + "\n"

CG_OLD = "    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS\n"
CG_NEW = CG_OLD + (
    "\n"
    "    @classmethod\n"
    "    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):  # " + MARKER + "\n"
    "        # decode entero PTX: FULL graph solo para decode uniforme (1 + tokens MTP)\n"
    "        if _g131.dtype_activo(getattr(vllm_config.cache_config, \"cache_dtype\", \"\")):\n"
    "            return AttentionCGSupport.UNIFORM_BATCH\n"
    "        return cls._cudagraph_support\n"
)

META_OLD = "        mm_ranges = common_attn_metadata.mm_req_doc_ranges\n"
META_NEW = (
    "        attn_metadata.genesis_qsl_cpu = common_attn_metadata.query_start_loc_cpu  # " + MARKER + "\n"
    + META_OLD
)

FWD_OLD = "        assert attn_metadata.use_cascade is False\n"
FWD_NEW = FWD_OLD + (
    "        if _g131.activo(self, layer):  # " + MARKER + "\n"
    "            return _g131.forward(self, layer, query, kv_cache, attn_metadata, output)\n"
)

UPD_OLD = "        # Reshape the input keys and values and store them in the cache.\n"
UPD_NEW = (
    "        if _g131.activo(self, layer):  # " + MARKER + "\n"
    "            return _g131.escribir(self, layer, key, value, kv_cache, slot_mapping)\n"
    + UPD_OLD
)


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN131")
    log_decision("PN131", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root no localizable"
    target = resolve_vllm_file("v1/attention/backends/triton_attn.py")
    if target is None:
        return "failed", "triton_attn.py no encontrado"
    p = TextPatcher(
        patch_name="PN131 SK-18 decode entero", target_file=str(target), marker=MARKER,
        sub_patches=[
            TextPatch(name="pn131_import", anchor=IMPORT_OLD, replacement=IMPORT_NEW, required=True),
            TextPatch(name="pn131_cg", anchor=CG_OLD, replacement=CG_NEW, required=True),
            TextPatch(name="pn131_meta", anchor=META_OLD, replacement=META_NEW, required=True),
            TextPatch(name="pn131_fwd", anchor=FWD_OLD, replacement=FWD_NEW, required=True),
            TextPatch(name="pn131_upd", anchor=UPD_OLD, replacement=UPD_NEW, required=True),
        ],
        upstream_drift_markers=[])
    result, failure = p.apply()
    return result_to_wiring_status(result, failure, applied_message="SK-18 decode entero armado",
                                   patch_name=p.patch_name)
