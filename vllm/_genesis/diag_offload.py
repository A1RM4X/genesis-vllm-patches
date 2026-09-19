# SPDX-License-Identifier: Apache-2.0
"""Diagnostico del offload de KV: por que se escribia a L2/L3 y nunca se leia.

RESUELTO EN PARTE el 2026-09-19. Esto documenta lo que la instrumentacion encontro, porque
hasta ese dia el archivo apostaba a la hipotesis equivocada (que cortaba el pre-chequeo del
lookup). No era eso: el lookup pedia bien y el backend no tenia nada, por tres causas
INDEPENDIENTES que habia que sacar en orden. Las tres eran parches nuestros.

1. PN97 — el interbloqueo
   "Una promocion L3->L2 solo usa slots REALMENTE libres." Se escribio cuando L2 tenia 222
   slots y un prompt costaba ~36: entraban seis y la regla era sana. PN145/PN146 subieron el
   bloque de GPU a 880 tokens, el slot de offload quedo en 29,8 MB y L2 en 72 slots, con ~28
   por request. Desde entonces L2 vive clavada al 100% (bytes_used == capacity_bytes) y "solo
   slots libres" significa NUNCA: `_initiate_promotion` devuelve False siempre y a L3 no se
   lo consulta. Medido: 7.347 consultas al tier primario contra 32 al de disco, 1 acierto.
   Con GENESIS_DISABLE_PN97=1: 73 consultas y 42 aciertos en L3.

2. PN91 — el presupuesto mas corto que el disco
   GENESIS_PN91_MAX_DEFER_SECONDS=0,2 con _STEPS=4. Una promocion L3->L2 tarda mas que eso.
   Al 5o paso PN91 arma el modo estricto, un bloque EN VUELO pasa a contar como MISS, se
   corta la racha de la ventana deslizante y el lookup ENTERO devuelve 0. O sea: se abortaba
   un rescate que ahorra 6,7 s por no esperarle dos decimas al disco. A 3,0 s / 60 pasos el
   rescate completo anda: TTFT 6,76 -> 0,97 s (-85,6%), 612 MiB por CPU_to_GPU y el primer
   external_prefix_cache_hits distinto de cero del proyecto.

3. PN81 — L3 mas chico que L1, y podando al reves
   La cuota eran 30 GB. La KV de GPU son 566.314 tokens y el offload escribe 62,9 KB por
   token: espejar L1 pide 35,6 GB. Un tier de cache mas chico que el tier que respalda es
   inutil por construccion. Y PN81 poda por MTIME —antiguedad de ESCRITURA, no de uso—, asi
   que la primera victima es justo el prefijo del hilo largo. A 64 GB desaparecen los
   desalojos de disco.

LO QUE QUEDA ABIERTO: el veto del grupo del borrador
   `_lookup_complete_chunks` exige que acierten TODOS los grupos. Con L2 de 2 GiB el prefijo
   de atencion acierta parcial (g6 8 de 20 chunks; con 6 GiB, 16 de 20) y el hit se trunca a
   ese limite. Pero del grupo de ventana deslizante del borrador DFlash2 upstream guarda
   SOLO la cola alcanzable —ver `is_store_reachable_swa_chunk`: `sliding_window + 1` chunks
   al final de cada segmento—, que corresponde al final REAL de la request, no al chunk 8.
   Entonces `('ventana4', N, 0)` da MISS y veta los 15 chunks que los otros ocho grupos SI
   acertaron. Solo un acierto de prefijo del 100% sobrevive, y por eso el test chico
   (L1 de 101k tokens, 8 prompts de desalojo) anda y el de produccion no.

   Las dos salidas, ninguna probada todavia:
     a) que L2 entre un par de requests enteras, para que el prefijo acierte 100%. Cuesta
        RAM de la placa madre, que es justo lo que el proyecto no quiere gastar. Medido que
        va en la direccion correcta (8/20 -> 16/20 al triplicar L2) pero no cruzo el umbral.
     b) sacarle el veto al grupo del borrador: es especulativo, un estado equivocado solo
        hace que el target rechace las propuestas —la salida sigue siendo correcta— a cambio
        de menos aceptacion. Mucho mas barato que recomputar 20k tokens. Riesgo: leer
        bloques sin inicializar, que en este proyecto ya dio NaN silenciosos antes (PN144).

Esto vuelca, por cada lookup, el resultado de cada grupo (tipo de spec, ventana, chunks
pedidos, chunks que contesto el backend) y cual fue la salida. Se prende con
GENESIS_DIAG_OFFLOAD=1 y se apaga solo a los GENESIS_DIAG_OFFLOAD_LOOKUPS lookups.
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
