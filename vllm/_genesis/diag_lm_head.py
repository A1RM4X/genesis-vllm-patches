# SPDX-License-Identifier: Apache-2.0
"""Firma del lm_head y del embedding, una vez por proceso, para cazar el arranque roto.

Por que existe
--------------
En vLLM v0.29.0 hay arranques que dejan el modelo roto: el chat corta a los 2 tokens con el
contenido vacio, mientras el texto plano por ``/v1/completions`` genera bien. Lo que distingue
al prompt del chat son los tokens ESPECIALES del template (248045/248046/248068 sobre un vocab
de 248320), o sea las ULTIMAS filas del vocabulario. Con TP=2 esas filas viven enteras en el
rank 1, que se queda con las filas 124160..248319.

La hipotesis es que en los arranques malos la particion del rank 1 queda mal. Adivinarlo desde
afuera no se puede, y la velocidad no distingue nada (ya paso: 44,3 tok/s roto contra 43,4
andando). Esto lo mide adentro del worker, que es donde estan los pesos.

Que imprime
-----------
Una linea por capa interesante, la primera vez que corre su ``forward``: forma, dtype, y tres
sumas en float64 — el tensor entero, las primeras 512 filas y las ULTIMAS 512. Separar la cola
es el punto: si la firma global coincide entre un arranque bueno y uno malo pero la de la cola
no, el dano esta localizado ahi y no es ruido de cuantizacion.

Uso
---
``GENESIS_DIAG_LMHEAD=1`` y comparar la salida de un arranque sano contra uno roto.
Es solo diagnostico: sin la variable no engancha nada.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.diag")

_vistos: set[str] = set()


def activo() -> bool:
    return os.environ.get("GENESIS_DIAG_LMHEAD", "0").strip().lower() in ("1", "true", "yes", "on")


def _firma(t) -> str:
    """Tres sumas en float64: todo, la cabeza y la COLA. La cola es la que interesa."""
    import torch

    if t is None:
        return "(sin peso)"
    try:
        x = t.detach()
        if x.dtype in (torch.int8, torch.uint8, torch.int32, torch.int64):
            x = x.to(torch.float64)
        else:
            x = x.to(torch.float64)
        n = x.shape[0]
        k = min(512, n)
        return ("forma=%s dtype=%s | todo=%.6e cabeza[:%d]=%.6e COLA[-%d:]=%.6e"
                % (tuple(t.shape), t.dtype, float(x.sum()), k, float(x[:k].sum()),
                   k, float(x[-k:].sum())))
    except Exception as e:                                   # noqa: BLE001
        return f"(no se pudo firmar: {type(e).__name__}: {e})"


def _mirar(nombre: str, modulo) -> None:
    if nombre in _vistos:
        return
    _vistos.add(nombre)
    partes = []
    for attr in ("weight", "weight_scale", "qweight"):
        t = getattr(modulo, attr, None)
        if t is not None and hasattr(t, "shape"):
            partes.append(f"{attr}: {_firma(t)}")
    rank = os.environ.get("VLLM_DP_RANK", "?")
    log.warning("[DIAG %s rank_env=%s] %s", nombre, rank,
                " || ".join(partes) if partes else "(sin tensores reconocibles)")


def enganchar() -> None:
    """Envuelve el forward de ParallelLMHead y VocabParallelEmbedding con el aviso de una vez."""
    if not activo():
        return
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )

    for cls, nombre in ((ParallelLMHead, "lm_head"), (VocabParallelEmbedding, "embed")):
        if getattr(cls, "_genesis_diag", False):
            continue
        original = cls.forward

        def envuelto(self, *a, _orig=original, _n=nombre, **kw):
            _mirar(_n, self)
            return _orig(self, *a, **kw)

        cls.forward = envuelto
        cls._genesis_diag = True
    log.warning("[DIAG] firma de lm_head/embedding enganchada (GENESIS_DIAG_LMHEAD=1)")
