# SPDX-License-Identifier: Apache-2.0
"""PN146 — el tamano de grupo de KV, elegible, porque la heuristica de upstream no lo acierta.

El problema
-----------
``kv_cache_utils.py`` arma los grupos de KV con ``group_size = min(capas por bucket)``, con un
escape si ``max < min * 1.5``. Con DFlash2 los buckets son ``[5 borrador, 16 atencion,
48 GDN]``: el minimo son las CINCO capitas del borrador, y el escape no dispara (48 > 7,5).
Resultado, medido: ``group_size = 5``, quince grupos, y las 16 capas de atencion del modelo
grande partidas en cuatro.

Por que importa, con la formula de upstream en la mano::

    num_blocks      = memoria_disponible / (group_size * page_por_capa)
    bloques_por_req = suma sobre grupos de cdiv(mem_del_grupo, page_del_grupo)
    concurrencia    = num_blocks / bloques_por_req

Los dos terminos tiran para lados distintos, y ahi esta la gracia: ``bytes_per_block`` lo fija
el grupo MAS GRANDE (es un ``max``, no una suma), asi que achicar el tamano de grupo da MAS
bloques totales, aunque haya mas grupos que pagar por request. Va al reves de la intuicion.

Prediccion con los numeros medidos en este rig (9,31 GiB, page 0,89 MB, bloque 880,
max_model_len 163840, y con PN145 ya puesto):

    group_size   grupos   bloques/req   num_blocks   concurrencia
        5 (hoy)    15         870          2151         2,47
        4          18         912          2678         2,94
        2          35        1802          5356         2,97
       16           5         239           669         2,80

Por eso este parche NO fija un valor: lo hace elegible. La aritmetica de arriba es mia, no de
una medicion, y ya me equivoque dos veces hoy razonando sobre este mismo codigo. Con la
variable puesta se mide cada valor contra "GPU KV cache size" y contra tok/s, y se elige con
datos. Sin la variable el parche es inerte y upstream decide como siempre.

El techo practico no esta en la formula sino en el scheduler: cada grupo lleva su propia
block_table por request, asi que ``group_size=1`` (69 grupos aca) puede ganar en la cuenta y
perder en el reloj. Por eso hay que mirar tok/s y no solo los tokens de KV.
"""

from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN146: tamano de grupo de KV]"

_OLD = (
    "    min_num_layers = min([len(layers) for layers in layer_buckets])\n"
    "    group_size = min_num_layers\n"
)

_NEW = (
    "    min_num_layers = min([len(layers) for layers in layer_buckets])\n"
    "    group_size = min_num_layers\n"
    "    # " + MARKER + " ver el modulo del parche para la aritmetica.\n"
    "    # bytes_per_block lo fija el grupo MAS GRANDE (un max, no una suma), asi que un\n"
    "    # group_size mas chico da mas bloques totales aunque haya mas grupos por request.\n"
    "    import os as _g146_os\n"
    "    _g146 = _g146_os.environ.get('GENESIS_PN146_GROUP_SIZE', '').strip()\n"
    "    if _g146:\n"
    "        try:\n"
    "            _g146_v = int(_g146)\n"
    "        except ValueError:\n"
    "            _g146_v = 0\n"
    "        if _g146_v > 0:\n"
    "            logger.warning('[Genesis PN146] group_size %d -> %d (buckets %s)',\n"
    "                           group_size, _g146_v,\n"
    "                           [len(x) for x in layer_buckets])\n"
    "            _g146_forzado = _g146_v\n"
    "        else:\n"
    "            _g146_forzado = None\n"
    "    else:\n"
    "        _g146_forzado = None\n"
)

_OLD2 = (
    "        group_size = max_num_layers\n"
    "    grouped_layers = []\n"
)
_NEW2 = (
    "        group_size = max_num_layers\n"
    "    if _g146_forzado:  # " + MARKER + " la ultima palabra es del operador\n"
    "        group_size = _g146_forzado\n"
    "    grouped_layers = []\n"
)


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN146")
    log_decision("PN146", decision, reason)
    if not decision:
        return "skipped", reason

    destino = resolve_vllm_file("v1/core/kv_cache_utils.py")
    if destino is None:
        return "skipped", "kv_cache_utils.py no esta en esta version de vLLM"

    p = TextPatcher(
        patch_name="PN146 tamano de grupo de KV",
        target_file=str(destino), marker=MARKER,
        sub_patches=[
            TextPatch(name="pn146_lee_env", anchor=_OLD, replacement=_NEW, required=True),
            TextPatch(name="pn146_aplica", anchor=_OLD2, replacement=_NEW2, required=True),
        ],
        upstream_drift_markers=["GENESIS_PN146_GROUP_SIZE"])
    result, failure = p.apply()
    return result_to_wiring_status(
        result, failure,
        applied_message="group_size elegible por GENESIS_PN146_GROUP_SIZE (inerte sin ella)",
        patch_name=p.patch_name)
