# SPDX-License-Identifier: Apache-2.0
"""Diagnostico del offload de KV: por que se escribe a L2/L3 y nunca se lee.

Lo medido (2026-09-19, DFlash2 + PN122, KV 566.314 tokens):

    desalojo con 40 prompts de 20k:  GPU_to_CPU  50,3 GB escritos
    rescate del prompt original:     CPU_to_GPU  ni un byte
                                     external_prefix_cache_hits_total  cero
                                     TTFT 6,63 s = recomputo completo

O sea que el camino de escritura anda y el de lectura no engancha nunca. El lookup
(`_lookup_complete_chunks` en el scheduler del connector) recorre los grupos y tiene TRES
salidas tempranas que devuelven 0 sin decir cual fue:

  1. `max_hit_size_tokens - num_computed_tokens < tokens_per_chunk`  (antes de consultar)
  2. `num_hit_chunks == 0`                                           (el backend no tiene nada)
  3. `new_num_hit_tokens < tokens_per_chunk`                         (despues de acotar)

Y ademas, con grupos de ventana deslizante presentes —que aca los hay, por el borrador
DFlash2— el techo se recorta antes de empezar::

    max_hit_size_tokens -= 1
    if self._mamba_align_size is not None:
        max_hit_size_tokens = round_down(max_hit_size_tokens, self._mamba_align_size)

Sin saber cual de las cuatro cosas dispara, "arreglarlo" es adivinar. Esto lo dice: por cada
lookup vuelca el resultado de cada grupo (tipo de spec, ventana, chunks pedidos, chunks que
contesto el backend) y cual fue la salida.

Se prende con GENESIS_DIAG_OFFLOAD=1 y se apaga solo a los pocos lookups.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.diag_offload")

_TOPE = int(os.environ.get("GENESIS_DIAG_OFFLOAD_LOOKUPS", "12"))


def enganchar() -> None:
    """Envuelve el lookup del scheduler del connector. No puede tumbar el arranque."""
    if os.environ.get("GENESIS_DIAG_OFFLOAD") != "1":
        return

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler as Sch,
    )

    estado = {"n": 0, "grupos": None}

    # --- lo que contesta el backend, por grupo ---
    orig_pref = Sch._maximal_prefix_lookup
    orig_sw = Sch._sliding_window_lookup
    traza: list = []

    def _maximal_prefix_lookup(self, offload_keys, *a, **kw):
        r = orig_pref(self, offload_keys, *a, **kw)
        traza.append(("prefijo", len(offload_keys), r))
        return r

    def _sliding_window_lookup(self, offload_keys, required_window, *a, **kw):
        r = orig_sw(self, offload_keys, required_window, *a, **kw)
        traza.append(("ventana%d" % required_window, len(offload_keys), r))
        return r

    Sch._maximal_prefix_lookup = _maximal_prefix_lookup
    Sch._sliding_window_lookup = _sliding_window_lookup

    orig_lookup = Sch._lookup_complete_chunks

    def _lookup_complete_chunks(self, req_status):
        if estado["n"] >= _TOPE:
            return orig_lookup(self, req_status)

        if estado["grupos"] is None:
            estado["grupos"] = True
            try:
                log.warning("[DIAG offload] %d grupos | lookup_groups=%s | ventana=%s | "
                            "mamba_align=%s | partial_tail=%s",
                            len(self.config.kv_group_configs), self._lookup_groups,
                            self._sliding_window_groups, self._mamba_align_size,
                            getattr(self.config, "supports_partial_tail", "?"))
                for g in self.config.kv_group_configs:
                    log.warning("[DIAG offload]   grp %s: tok/bloque=%s tok/chunk=%s "
                                "ventana_chunks=%s eagle=%s cow=%s",
                                getattr(g, "group_idx", "?"),
                                getattr(g, "tokens_per_block", "?"),
                                getattr(g, "tokens_per_chunk", "?"),
                                getattr(g, "sliding_window_size_in_chunks", "?"),
                                getattr(g, "is_eagle_group", "?"),
                                getattr(g, "requires_cow_source", "?"))
            except Exception as e:                                   # noqa: BLE001
                log.warning("[DIAG offload] no pude volcar la config (%s)", type(e).__name__)

        traza.clear()
        r = orig_lookup(self, req_status)
        estado["n"] += 1
        try:
            n_tok = req_status.req.num_tokens
            # Las claves por grupo son lo que decide el pre-chequeo que corta ANTES de
            # consultar al backend:
            #   max_hit = min(max_hit, len(offload_keys) * tokens_per_chunk)
            #   if max_hit - num_computed < tokens_per_chunk: return 0
            # Con el orden de lookup_groups, el primer grupo cuyo len(claves) no alcanza es
            # el que mata el hit, y no deja rastro en la traza de consultas.
            claves = []
            for i, gs in enumerate(req_status.group_states):
                gc = self.config.kv_group_configs[i]
                claves.append("g%d:%d claves x%d tok%s" % (
                    i, len(getattr(gs, "offload_keys", ()) or ()),
                    getattr(gc, "tokens_per_chunk", 0),
                    "" if gc.sliding_window_size_in_chunks is None
                    else "/v%d" % gc.sliding_window_size_in_chunks))
            log.warning("[DIAG offload %d/%d] req de %d tokens, %d ya computados -> "
                        "devuelve %s\n    consultas: %s\n    claves: %s",
                        estado["n"], _TOPE, n_tok,
                        req_status.num_locally_computed_tokens, r,
                        traza if traza else "NINGUNA (corto antes de consultar)",
                        " | ".join(claves))
        except Exception as e:                                       # noqa: BLE001
            log.warning("[DIAG offload] %s: %s", type(e).__name__, e)
        return r

    Sch._lookup_complete_chunks = _lookup_complete_chunks
    log.warning("[DIAG offload] enganchado sobre el lookup del connector (%d lookups)", _TOPE)
