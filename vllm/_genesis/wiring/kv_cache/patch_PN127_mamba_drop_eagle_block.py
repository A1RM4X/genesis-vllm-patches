# SPDX-License-Identifier: Apache-2.0
"""PN127 — MambaManager respeta ``drop_eagle_block`` (backport de vllm#48375).

Bug (vllm#43559): con MTP/EAGLE + prefix caching en un hibrido Qwen3-Next,
``MambaManager.find_longest_cache_hit`` recibe ``drop_eagle_block`` y lo
IGNORA. El estado recurrente guardado en la ultima pagina que matchea puede
haberse tomado sobre tokens de borrador que la verificacion rechazo despues, y
se reusa como si fuera bueno. Upstream reprodujo ~20% de caida de precision en
el 35B-A3B; club-3090 lo porta en sus composes MTP de Qwen3.8 (costo medido:
~5% menos tokens de prefix cache).

Verificado en nuestro 0.27.1 (2026-09-15): ni P83 ni PN84 lo cubren.

Arreglo: bajar una pagina el techo de busqueda (Mamba guarda solo el bloque
real mas a la derecha: un pop borraria el bloque de estado). Se aplica tambien
a la rama fina (``prefix_match_unit`` < block_size), que el PR original no toca:
ahi se baja una pagina entera (``scale_factor`` unidades finas).
"""

from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN127: MambaManager drop_eagle_block]"

GRUESA_OLD = (
    "        max_num_blocks = max_length // block_size\n"
    "        # Search from right to left and early stop when a match is found.\n"
    "        for i in range(max_num_blocks - 1, -1, -1):\n"
    "            if cached_block := block_pool.get_cached_block(\n"
    "                block_hashes[i], kv_cache_group_ids\n"
    "            ):\n"
    "                # When enable Mamba prefix caching, `block_size` will be aligned\n"
)
GRUESA_NEW = (
    "        max_num_blocks = max_length // block_size\n"
    "        # " + MARKER + " EAGLE/MTP: la ultima pagina puede tener el estado\n"
    "        # recurrente tomado sobre tokens de borrador rechazados (vllm#48375).\n"
    "        if drop_eagle_block and max_num_blocks > 0:\n"
    "            max_num_blocks -= 1\n"
    "        # Search from right to left and early stop when a match is found.\n"
    "        for i in range(max_num_blocks - 1, -1, -1):\n"
    "            if cached_block := block_pool.get_cached_block(\n"
    "                block_hashes[i], kv_cache_group_ids\n"
    "            ):\n"
    "                # When enable Mamba prefix caching, `block_size` will be aligned\n"
)
FINA_OLD = (
    "            max_num_partial_units = min(\n"
    "                max_length // hash_block_size, len(block_hashes)\n"
    "            )\n"
)
FINA_NEW = (
    "            max_num_partial_units = min(\n"
    "                max_length // hash_block_size, len(block_hashes)\n"
    "            )\n"
    "            # " + MARKER + " rama fina: bajar una pagina entera.\n"
    "            if drop_eagle_block:\n"
    "                max_num_partial_units = max(0, max_num_partial_units - scale_factor)\n"
)


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN127")
    log_decision("PN127", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root no localizable"
    target = resolve_vllm_file("v1/core/single_type_kv_cache_manager.py")
    if target is None:
        return "failed", "single_type_kv_cache_manager.py no encontrado"
    p = TextPatcher(
        patch_name="PN127 MambaManager drop_eagle_block", target_file=str(target), marker=MARKER,
        sub_patches=[TextPatch(name="pn127_gruesa", anchor=GRUESA_OLD, replacement=GRUESA_NEW, required=True),
                     TextPatch(name="pn127_fina", anchor=FINA_OLD, replacement=FINA_NEW, required=True)],
        upstream_drift_markers=["drop_eagle_block and max_num_blocks > 0"])
    result, failure = p.apply()
    return result_to_wiring_status(result, failure, applied_message="MambaManager respeta drop_eagle_block",
                                   patch_name=p.patch_name)
