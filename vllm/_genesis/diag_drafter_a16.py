# SPDX-License-Identifier: Apache-2.0
"""DIAGNOSTICO — el borrador con activaciones fp16 (W4A16) mientras el target sigue en W4A8.

``VLLM_MARLIN_INPUT_DTYPE=int8`` es global: tambien las lineales del borrador cuantizan sus
activaciones a int8. La aceptacion en contexto largo resulto MUY sensible a la precision del
borrador (2026-09-20: sacarle la rotacion Hadamard a su KV int8 la baja de 7,00 a 3,46 a 37k),
asi que antes de construir nada hay que medir el TECHO: cuanta aceptacion se recupera si sus
activaciones no se cuantizan. Esto no es una configuracion de produccion, es una regla de medir.

``GENESIS_DIAG_DRAFTER_A16=1`` hace que ``get_marlin_input_dtype(prefix)`` devuelva ``None``
(= activaciones fp16) para las capas cuyo prefijo es del borrador. Deja en el log cuantas capas
toco y con que prefijos: si dice 0, el experimento no midio nada.
"""

from __future__ import annotations

import logging
import os
import sys

log = logging.getLogger("genesis.diag.drafter_a16")

_tocadas: list[str] = []
_otras = 0


def _es_del_borrador(prefijo: str) -> bool:
    marcas = os.environ.get("GENESIS_DIAG_DRAFTER_PREFIJOS", "draft,mtp,dflash").split(",")
    p = prefijo.lower()
    return any(m and m in p for m in marcas)


def enganchar() -> None:
    from vllm.model_executor.layers.quantization.utils import marlin_utils as mu

    original = mu.get_marlin_input_dtype
    if getattr(original, "_genesis_drafter_a16", False):
        return

    def get_marlin_input_dtype(prefix: str | None = None):
        global _otras
        if prefix and _es_del_borrador(prefix):
            _tocadas.append(prefix)
            if len(_tocadas) in (1, 8, 64):
                log.warning("[DIAG drafter A16] %d capas del borrador en fp16 (ultima: %s)",
                            len(_tocadas), prefix)
            return None
        _otras += 1
        if _otras in (1, 64):
            log.warning("[DIAG drafter A16] capa del TARGET, sigue en int8: %s", prefix)
        return original(prefix)

    get_marlin_input_dtype._genesis_drafter_a16 = True
    mu.get_marlin_input_dtype = get_marlin_input_dtype
    # Los modulos que ya hicieron `from ... import get_marlin_input_dtype` tienen su copia.
    for m in list(sys.modules.values()):
        try:
            if getattr(m, "get_marlin_input_dtype", None) is original:
                m.get_marlin_input_dtype = get_marlin_input_dtype
        except Exception:                                    # noqa: BLE001
            pass
    log.warning("[DIAG drafter A16] enganchado: las lineales del borrador NO cuantizan activaciones")
