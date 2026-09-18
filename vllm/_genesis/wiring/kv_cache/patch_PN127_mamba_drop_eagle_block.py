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


# ── Guarda: no cachear mas bloques que hashes hay ─────────────────────────────────────
# El bug, capturado por el volcado de diagnostico en DOS crashes con la misma firma exacta:
#
#     num_cached_blocks = 55        num_full_blocks = 56       len(block_hashes) = 55
#     num_cached_blocks = 22        num_full_blocks = 23       len(block_hashes) = 22
#
# siempre en el grupo de Mamba (block_size=816) y siempre UN bloque de mas. De ahi
# `new_block_hashes = block_hashes[num_cached_blocks:]` sale VACIA, el bucle indexa [0] y
# el EngineCore muere con IndexError.
#
# El bloque de mas viene de upstream a proposito: en `kv_cache_coordinator.cache_blocks`,
# para grupos con use_eagle (y "mtp" cuenta como eagle),
#     num_tokens_to_cache = min(num_finalized, aligned + manager.block_size)
# o sea que habilita a cachear una pagina PASADO el borde alineado. Pero la lista de hashes
# del request solo llega hasta el borde, asi que ese bloque no tiene hash.
#
# Recortar es lo correcto, no un parche defensivo cualquiera: ese bloque es exactamente el
# que el lado de BUSQUEDA ya descarta (`drop_eagle_block`, que es de lo que trata este mismo
# PN127). Cachearlo no serviria para nada aunque hubiera hash — nadie lo va a matchear. Y
# `num_cached_block` queda consistente porque se recorta ANTES de guardarlo.
#
# Cuando hay hashes de sobra el min() no hace nada, asi que es inocuo para los demas grupos.
GUARDA_OLD = (
    "        num_cached_blocks = self.num_cached_block.get(request.request_id, 0)\n"
    "        num_full_blocks = num_tokens // self.block_size\n"
    "\n"
    "        if num_cached_blocks >= num_full_blocks:\n"
    "            return\n"
)
GUARDA_NEW = (
    "        num_cached_blocks = self.num_cached_block.get(request.request_id, 0)\n"
    "        num_full_blocks = num_tokens // self.block_size\n"
    "        # " + MARKER + " no se puede cachear un bloque sin hash.\n"
    "        try:\n"
    "            _g127_hashes = resolve_block_hashes(\n"
    "                request.block_hashes, self.block_pool.hash_block_size, self.block_size\n"
    "            )\n"
    "            num_full_blocks = min(num_full_blocks, len(_g127_hashes))\n"
    "        except Exception:\n"
    "            pass\n"
    "\n"
    "        if num_cached_blocks >= num_full_blocks:\n"
    "            return\n"
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
        sub_patches=[TextPatch(name="pn127_guarda_hashes", anchor=GUARDA_OLD,
                               replacement=GUARDA_NEW, required=False),
                     TextPatch(name="pn127_gruesa", anchor=GRUESA_OLD, replacement=GRUESA_NEW, required=True),
                     TextPatch(name="pn127_fina", anchor=FINA_OLD, replacement=FINA_NEW, required=True)],
        upstream_drift_markers=["drop_eagle_block and max_num_blocks > 0"])
    result, failure = p.apply()
    return result_to_wiring_status(result, failure, applied_message="MambaManager respeta drop_eagle_block",
                                   patch_name=p.patch_name)
