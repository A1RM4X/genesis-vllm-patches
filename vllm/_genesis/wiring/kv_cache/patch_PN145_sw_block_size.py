# SPDX-License-Identifier: Apache-2.0
"""PN145 — el grupo de ventana deslizante elige un bloque de 16 tokens para una pagina de 880.

El sintoma
----------
Con DFlash2, el servidor pasa de 611.402 tokens de KV (MTP) a 178.823 (-71%), y la
concurrencia a 1,09x: apenas entra UN request de largo completo. Medido con ``diag_kv``, el
reparto de los 2001 bloques que consume un request:

    10 grupos Mamba              100 bloques    5%
     4 grupos FullAttention      748 bloques   37%
     1 grupo  SlidingWindow     1153 bloques   58%   <-- el borrador, 5 capas

El grupo del borrador —cinco capitas— se lleva mas que las 64 del modelo grande juntas. Pide
1006 MB por request contra 653 MB de las 16 capas de atencion del target.

La causa, que son dos bugs encadenados
--------------------------------------
``max_concurrency = num_blocks / sum_grupos(cdiv(max_memory_usage, page_size))``, o sea que lo
que importa es cuantos BLOQUES necesita cada grupo, y eso sale de ``cdiv(tokens, block_size)``.
El grupo del borrador tiene ``block_size = 16`` mientras su pagina esta padeada a 0,87 MB —
dimensionada para 880 tokens. Paga una pagina entera cada 16 tokens: un factor de ~55.

De donde sale ese 16, en ``model_executor/layers/attention/attention.py``:

1. El call site pasa ``page_budget = cache_config.skip_page_size_padded``, que es un concepto
   de skip-quant/TurboQuant y en nuestra config vale ``None``. Sin presupuesto,
   ``_largest_kernel_block_within`` devuelve ``smallest``. El comentario de upstream dice que
   no importa porque "the block is handled by unify's integer scaling instead" — pero en un
   hibrido ``unify`` PADEA la pagina a 0,87 MB sin escalar el bloque, asi que la premisa no
   se cumple y el 16 queda congelado.

2. Y aunque hubiera presupuesto, tampoco alcanzaria: el backend declara sus tallas como
   ``MultipleOf(16)``, y la funcion toma unicamente ``s.base`` como candidato. No puede
   expresar "cualquier multiplo de 16 hasta N", asi que el maximo posible sigue siendo 16.
   Verificado a mano contra el backend real:

       _largest_kernel_block_within(B, 1024, None,   880) -> 16
       _largest_kernel_block_within(B, 1024, 912000, 880) -> 16    <-- con presupuesto, igual
       (la respuesta correcta es 880: el mayor multiplo de 16 que entra en 912000/1024 = 890)

El arreglo
----------
Se reescribe la eleccion para que, cuando el backend declara ``MultipleOf(base)``, considere
los MULTIPLOS de base y no solo base. Y cuando no hay presupuesto de pagina, en vez de caer al
mas chico se usa ``fallback`` —que es el ``block_size`` de la atencion primaria— como techo:
el bloque de la ventana queda alineado con el del resto en lugar de 55 veces mas fino.

Es conservador por construccion: nunca devuelve una talla que el backend no haya declarado, ni
una mayor que la que ya se usa para la atencion primaria, ni una cuya pagina natural se pase
del presupuesto cuando ese presupuesto existe.

Cuentas del premio, con los numeros medidos (max_model_len 163840, bloque 880):
    hoy   cdiv(18432, 16) + 1  = 1153 bloques -> total 2001 -> concurrencia 2184/2001 = 1,09
    con   cdiv(18432, 880) + 1 =   22 bloques -> total  870 -> concurrencia 2184/ 870 = 2,51
    o sea ~411.000 tokens de KV en vez de 178.823.

Esa cuenta es una PREDICCION, no una medicion: hay que verificarla contra el arranque real
mirando "GPU KV cache size" y el volcado de diag_kv, porque cambiar el block_size del grupo
tambien cambia el reparto de memoria y la prediccion puede quedar corta o larga.
"""

from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN145: bloque de la ventana deslizante]"

_OLD = (
    "    sizes = attn_backend.get_supported_kernel_block_sizes()\n"
    "    candidates = [s for s in sizes if isinstance(s, int)]\n"
    "    if not candidates:\n"
    "        candidates = [s.base for s in sizes if isinstance(s, MultipleOf)]\n"
    "    if not candidates:\n"
    "        return fallback\n"
    "    smallest = min(candidates)\n"
    "    if not page_budget or per_token_bytes <= 0:\n"
    "        return smallest\n"
    "    fitting = [b for b in candidates if b * per_token_bytes <= page_budget]\n"
    "    return max(fitting) if fitting else smallest\n"
)

_NEW = (
    "    # " + MARKER + "\n"
    "    # Upstream toma solo `s.base` de un MultipleOf, asi que no puede elegir un multiplo;\n"
    "    # y sin `page_budget` cae al mas chico, apostando a que `unify` despues escale el\n"
    "    # bloque. En un hibrido `unify` padea la PAGINA y deja el bloque como esta, y el\n"
    "    # grupo termina con bloques de 16 tokens en paginas de 880: paga una pagina entera\n"
    "    # cada 16 tokens. Con DFlash2 eso se comia el 58% de los bloques por request.\n"
    "    sizes = attn_backend.get_supported_kernel_block_sizes()\n"
    "    exactos = [s for s in sizes if isinstance(s, int)]\n"
    "    bases = [s.base for s in sizes if isinstance(s, MultipleOf)]\n"
    "    if not exactos and not bases:\n"
    "        return fallback\n"
    "\n"
    "    # Techo: la pagina si la hay, y siempre el bloque de la atencion primaria — no tiene\n"
    "    # sentido que la ventana use un bloque mas grueso que el resto del modelo.\n"
    "    techo = fallback if fallback and fallback > 0 else None\n"
    "    if page_budget and per_token_bytes > 0:\n"
    "        cabe = page_budget // per_token_bytes\n"
    "        techo = min(techo, cabe) if techo else cabe\n"
    "\n"
    "    candidatos = list(exactos)\n"
    "    for base in bases:\n"
    "        # Un MultipleOf(base) habilita TODOS los multiplos, no solo el primero.\n"
    "        candidatos.append(base * (techo // base) if techo and techo >= base else base)\n"
    "    if not candidatos:\n"
    "        return fallback\n"
    "\n"
    "    admisibles = [b for b in candidatos if techo is None or b <= techo]\n"
    "    return max(admisibles) if admisibles else min(candidatos)\n"
)


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN145")
    log_decision("PN145", decision, reason)
    if not decision:
        return "skipped", reason

    destino = resolve_vllm_file("model_executor/layers/attention/attention.py")
    if destino is None:
        return "skipped", "attention.py no esta en esta version de vLLM"

    p = TextPatcher(
        patch_name="PN145 bloque de la ventana deslizante",
        target_file=str(destino), marker=MARKER,
        sub_patches=[TextPatch(name="pn145_largest_kernel_block", anchor=_OLD,
                               replacement=_NEW, required=True)],
        # Si upstream llega a considerar los multiplos por su cuenta, este parche sobra.
        upstream_drift_markers=["base * (page_budget", "multiples of base"])
    result, failure = p.apply()
    return result_to_wiring_status(
        result, failure,
        applied_message="el grupo de ventana deslizante deja de pagar una pagina cada 16 tokens",
        patch_name=p.patch_name)
