# SPDX-License-Identifier: Apache-2.0
"""Diagnostico del offload de KV: por que se escribia a L2/L3 y nunca se leia.

RESUELTO el 2026-09-19. Esto documenta lo que la instrumentacion encontro, porque
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

2. PN91 — el presupuesto mas corto que el disco  (ERA LA CAUSA PRINCIPAL)
   GENESIS_PN91_MAX_DEFER_SECONDS=0,2 con _STEPS=4. Al agotarse, PN91 pasa a modo estricto,
   y en estricto un bloque EN VUELO (HIT_PENDING) cuenta como MISS: corta la racha de
   `_sliding_window_lookup` y el lookup ENTERO devuelve 0. Las promociones L3->L2 son
   asincronas y resuelven DE A POCO, una tanda por pasada del scheduler, asi que el
   presupuesto hay que darlo con mucha holgura:

       0,2 s /   4 pasos  ->  no resolvia ninguna; el offload nunca leyo un byte
       3   s /  60 pasos  ->  resolvia una parte, y variaba por corrida: el prefijo de
                              atencion acertaba 0, 8, 12 o 16 de 20 chunks
      15   s / 400 pasos  ->  aciertan los NUEVE grupos, 20/20 y 19/19, reproducible

   No cuesta latencia en el caso malo: sin nada en cache el backend devuelve MISS al toque,
   no HIT_PENDING. TTFT en frio 6,76-6,88 s, igual que antes del cambio.

3. PN81 — L3 mas chico que L1, y podando al reves
   La cuota eran 30 GB. La KV de GPU son 566.314 tokens y el offload escribe 62,9 KB por
   token: espejar L1 pide 35,6 GB. Un tier de cache mas chico que el tier que respalda es
   inutil por construccion. Y PN81 poda por MTIME —antiguedad de ESCRITURA, no de uso—, asi
   que la primera victima es justo el prefijo del hilo largo. A 64 GB desaparecen los
   desalojos de disco.

RESULTADO: rescate de un prompt de 20k desalojado de la GPU, TTFT 6,88 -> 1,21 s (-82,5%),
612 MiB por CPU_to_GPU, salida identica a la del acierto de L1. Tres corridas seguidas con
disco limpio y reinicio: 1,21 / 1,24 / 1,28 s.

DOS CALLEJONES SIN SALIDA, anotados para no repetirlos
------------------------------------------------------
* "El grupo del borrador veta el acierto de los demas, hay que sacarle el veto." FALSO.
  Con el presupuesto de PN91 en 3 s el lookup daba `('ventana4', N, 0)` y yo lo lei como
  "el borrador no tiene nada guardado, y no puede tenerlo porque sus chunks viejos tienen
  block_id 0 y `_build_store_jobs` los saltea". Llegue a escribir el parche (PN147, tres
  anclajes) y a medir un rescate exitoso con el. Pero era n=1 y NO REPRODUJO. Con el
  presupuesto en 15 s y PN147 APAGADO el mismo grupo contesta `('ventana4', 19, 19)`: los
  datos estaban siempre, el 0 era el modo estricto de PN91 contando lo que estaba en vuelo.
  PN147 se borro del arbol: su text-patch tocaba tres anclajes de upstream de forma
  incondicional a cambio de nada.
* "El anillo de staging de PN100 se lleva 32 de los 72 slots de L2, liberarlo da +44%."
  PEOR. Con GENESIS_DISABLE_PN100=1 el prefijo de atencion paso de 12/20 a 0/20: el anillo
  acota el trafico efimero y por eso PROTEGE el prefijo del hilo largo, que es literalmente
  lo que dice su docstring. Queda en 32.

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
