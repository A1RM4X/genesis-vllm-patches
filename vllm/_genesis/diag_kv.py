# SPDX-License-Identifier: Apache-2.0
"""Radiografia del KVCacheConfig: quien se lleva la memoria de KV y cuanto se desperdicia.

Por que hace falta
------------------
Con DFlash2 el servidor pasa de 611.402 tokens de KV (MTP) a 178.823. Las cuentas de
servilleta dicen:

    MTP      10,82 GiB / (619.256 / 832)  = 15,6 MB por bloque
    DFlash2   9,31 GiB / (178.823 / 880)  = 49,2 MB por bloque     -> 3,15x mas caro

El padding de capas que avisa vLLM explica ~30% de eso, no 215%. O sea que hay un agujero
sin explicar, y elegir entre los arreglos candidatos sin verlo es adivinar. En esta misma
sesion ya me paso dos veces: arme una historia convincente sobre `hidden_norm` que era falsa,
y no vi el epsilon de la RMSNorm hasta medirlo.

Lo que vuelca
-------------
Por grupo: cuantas capas, de que tipo, y el page_size_bytes de su spec — que es el numero que
manda, porque todos los grupos tienen que reportar el mismo y los chicos se padean al mayor.
Ademas el desperdicio por capa fantasma, calculado en BYTES y no en capas, que es lo que hay
que minimizar y no es lo que minimiza la heuristica de upstream.

La heuristica esta en `kv_cache_utils.py` (`group_size = min(capas por bucket)`, con un
escape si `max < min * 1.5`). Con 16 full + 48 GDN + 5 del borrador da group_size=5, y por eso
las 5 capitas del borrador terminan imponiendole el tamano de grupo a las 64 del modelo
grande: 16 padea a 20 (25%) y 48 padea a 50 (4,17%, y son capas de GDN, las mas caras).

Se prende con GENESIS_DIAG_KV=1. Corre UNA vez, en el arranque, y no toca nada.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.diag_kv")


def _mb(n) -> str:
    try:
        return "%.2f MB" % (int(n) / 1024 / 1024)
    except Exception:                                            # noqa: BLE001
        return str(n)


def _describe(spec) -> tuple[str, int]:
    """(tipo, page_size_bytes) de un spec, tolerante a que cambien los nombres."""
    tipo = type(spec).__name__
    for attr in ("page_size_padded", "page_size_bytes"):
        v = getattr(spec, attr, None)
        if v:
            return (tipo + ("*" if attr == "page_size_padded" else ""), int(v))
    return (tipo, 0)


def _volcar(configs, kv_cache_specs=None) -> None:
    for i, cfg in enumerate(configs if isinstance(configs, (list, tuple)) else [configs]):
        grupos = getattr(cfg, "kv_cache_groups", None)
        if grupos is None:
            continue
        nb = getattr(cfg, "num_blocks", 0)
        log.warning("[DIAG KV] === config %d: num_blocks=%d  layout=%s ===",
                    i, nb, getattr(cfg, "kv_cache_layout", None))

        total_bytes = 0
        filas = []
        for g, grupo in enumerate(grupos):
            capas = list(getattr(grupo, "layer_names", []) or [])
            spec = getattr(grupo, "kv_cache_spec", None)
            tipo, page = _describe(spec)
            bloque = getattr(spec, "block_size", None)
            # Las capas de padding entran como nombres inventados o como None.
            fantasma = sum(1 for c in capas if c is None or "padding" in str(c).lower())
            bytes_grupo = len(capas) * page
            total_bytes += bytes_grupo
            filas.append((g, len(capas), fantasma, tipo, bloque, page, bytes_grupo,
                          capas[0] if capas else "?"))

        log.warning("[DIAG KV] %-3s %-6s %-9s %-26s %-7s %-14s %-14s %s",
                    "grp", "capas", "fantasma", "spec", "bloque", "page/capa",
                    "bytes/bloque", "primera capa")
        desperdicio = 0
        for g, n, fant, tipo, bloque, page, bg, prim in filas:
            log.warning("[DIAG KV] %-3d %-6d %-9d %-26s %-7s %-14s %-14s %s",
                        g, n, fant, tipo, bloque, _mb(page), _mb(bg), prim)
            desperdicio += fant * page

        log.warning("[DIAG KV] TOTAL %s por bloque x %d bloques = %.2f GiB",
                    _mb(total_bytes), nb, total_bytes * nb / 1024 ** 3)
        if total_bytes:
            log.warning("[DIAG KV] desperdicio por capas fantasma: %s por bloque (%.1f%%) "
                        "= %.2f GiB", _mb(desperdicio), 100 * desperdicio / total_bytes,
                        desperdicio * nb / 1024 ** 3)
        # El page_size de cada grupo TIENE que ser igual; si no lo es, el mayor manda y los
        # otros se padean, y ahi puede estar la diferencia que no cierra.
        pages = sorted({f[5] for f in filas})
        if len(pages) > 1:
            log.warning("[DIAG KV] OJO: hay %d page_size distintos (%s). El mayor manda y los "
                        "demas se padean: ahi puede estar el agujero.",
                        len(pages), ", ".join(_mb(p) for p in pages))

    if kv_cache_specs:
        from collections import Counter

        for d in (kv_cache_specs if isinstance(kv_cache_specs, (list, tuple)) else [kv_cache_specs]):
            if not isinstance(d, dict):
                continue
            c = Counter(type(v).__name__ for v in d.values())
            log.warning("[DIAG KV] specs crudos del modelo (antes de agrupar): %s", dict(c))
            # El tamano de grupo que elige upstream sale de estos numeros.
            tam = sorted(c.values())
            if tam:
                log.warning("[DIAG KV] buckets=%s -> group_size de upstream = %d "
                            "(min, salvo que max < min*1.5)", tam, tam[0])


def enganchar() -> None:
    """Engancha por la CLASE y por la funcion interna. No puede tumbar el arranque.

    El primer intento envolvia ``kv_cache_utils.get_kv_cache_configs`` y no disparo nunca:
    ``v1/engine/core.py:51`` la importa POR NOMBRE, asi que reemplazar el atributo del modulo
    no toca la referencia que ya quedo en el otro modulo. Es el mismo error que cometi con
    ``register_backend`` en PN131.

    Los dos enganches de aca no tienen ese problema:
      * ``KVCacheConfig.__init__`` es un metodo de clase — se resuelve siempre por la clase,
        la haya importado quien la haya importado;
      * ``create_kv_cache_group_specs`` la llaman desde ADENTRO de kv_cache_utils, y esas
        llamadas si resuelven por el global del modulo.
    """
    if os.environ.get("GENESIS_DIAG_KV") != "1":
        return

    from vllm.v1 import kv_cache_interface
    from vllm.v1.core import kv_cache_utils

    # --- los buckets crudos, antes de agrupar: de ahi sale el group_size de upstream ---
    orig_grupos = kv_cache_utils.create_kv_cache_group_specs

    def create_kv_cache_group_specs(kv_cache_spec, grouped_layer_names, *a, **kw):
        try:
            from collections import Counter

            c = Counter(type(v).__name__ for v in kv_cache_spec.values())
            log.warning("[DIAG KV] specs crudos: %s", dict(c))
            tam = sorted(c.values())
            log.warning("[DIAG KV] buckets=%s -> group_size=%d (min; el escape max<min*1.5 "
                        "%s)", tam, tam[0],
                        "DISPARA" if tam and max(tam) < tam[0] * 1.5 else "no dispara")
            log.warning("[DIAG KV] grupos armados: %d, con %s capas cada uno",
                        len(grouped_layer_names), [len(g) for g in grouped_layer_names][:8])
        except Exception as e:                                   # noqa: BLE001
            log.warning("[DIAG KV] buckets: %s: %s", type(e).__name__, e)
        return orig_grupos(kv_cache_spec, grouped_layer_names, *a, **kw)

    kv_cache_utils.create_kv_cache_group_specs = create_kv_cache_group_specs

    # --- el config final ---
    cls = kv_cache_interface.KVCacheConfig
    orig_init = cls.__init__
    estado = {"n": 0}

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        if estado["n"] < 2:
            estado["n"] += 1
            try:
                _volcar([self])
            except Exception as e:                               # noqa: BLE001
                log.warning("[DIAG KV] no se pudo volcar (%s: %s)", type(e).__name__, e)

    cls.__init__ = __init__
    log.warning("[DIAG KV] enganchado sobre KVCacheConfig.__init__ y "
                "create_kv_cache_group_specs")
