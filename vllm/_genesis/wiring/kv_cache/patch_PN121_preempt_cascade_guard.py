# SPDX-License-Identifier: Apache-2.0
"""PN121 — cortar la cascada de preempciones cuando los frees van diferidos.

El bug
------
Con un KV connector consumidor (el offloading de L2) y scheduling asíncrono,
upstream activa ``Scheduler.defer_block_free``: los bloques de un request
preemptado NO vuelven al pool en el acto, se encolan en ``deferred_frees`` hasta
que termina el paso en vuelo que todavía podría escribirlos.

Pero el bucle de preempción de ``schedule()`` no lo sabe::

    while True:
        new_blocks = allocate_slots(request, ...)
        if new_blocks is not None: break
        victima = max(running, key=prioridad)   # o running.pop()
        _preempt_request(victima)               # sus bloques -> DIFERIDOS
        if victima == request: break

Como los bloques de la víctima no vuelven, el ``allocate_slots`` siguiente
falla IGUAL y preempta a la próxima, y a la próxima, hasta preemptarse a sí
mismo. Una sola falta de 9 bloques vacía la KV entera.

Medido con la carga real de opencode (3 hilos de 60-130k): una ronda de 3
preempciones en el mismo paso tiró la KV de 99% a 14%, y otra ráfaga dio 59
preempciones en un minuto. Cada víctima re-prefillea desde cero (el prompt de
66.557 tokens que volvió entró con 0 hits en L1 y 0 en L2).

El arreglo
----------
Si la víctima que se acaba de preemptar dejó sus bloques diferidos, no se sigue
preemptando: el request actual no se agenda en ESTE paso y en el siguiente los
bloques ya drenaron. Cuesta un paso de latencia a un request, contra re-hacer
cientos de miles de tokens de prefill.
"""

from __future__ import annotations

import logging

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

log = logging.getLogger("genesis.wiring.pn121")

GENESIS_PN121_MARKER = "[Genesis PN121: guard de cascada de preempcion]"

ANCHOR_OLD = (
    "                    self._preempt_request(\n"
    "                        preempted_req,\n"
    "                        scheduled_timestamp,\n"
    "                        drop_stale_output=self.requires_kv_delivery,\n"
    "                    )\n"
    "                    preempted_reqs.append(preempted_req)\n"
    "                    if preempted_req == request:\n"
    "                        # No more request to preempt. Cannot schedule this request.\n"
    "                        break\n"
)

ANCHOR_NEW = (
    "                    # " + GENESIS_PN121_MARKER + "\n"
    "                    _g121_def = len(self.deferred_frees)\n"
    "                    self._preempt_request(\n"
    "                        preempted_req,\n"
    "                        scheduled_timestamp,\n"
    "                        drop_stale_output=self.requires_kv_delivery,\n"
    "                    )\n"
    "                    preempted_reqs.append(preempted_req)\n"
    "                    if preempted_req == request:\n"
    "                        # No more request to preempt. Cannot schedule this request.\n"
    "                        break\n"
    "                    # Los bloques de la victima quedaron diferidos: reintentar\n"
    "                    # allocate_slots fallaria igual y preemptaria a otra.\n"
    "                    # Se corta aca; el paso siguiente ya los tiene drenados.\n"
    "                    if len(self.deferred_frees) > _g121_def:\n"
    "                        new_blocks = None\n"
    "                        break\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/core/sched/scheduler.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN121 guard de cascada de preempcion",
        target_file=str(target),
        marker=GENESIS_PN121_MARKER,
        sub_patches=[
            TextPatch(name="pn121_preempt_loop", anchor=ANCHOR_OLD,
                      replacement=ANCHOR_NEW, required=True),
        ],
        upstream_drift_markers=[],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN121")
    log_decision("PN121", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root no localizable"
    patcher = _make_patcher()
    if patcher is None:
        return "failed", "scheduler.py no encontrado"
    result, failure = patcher.apply()
    return result_to_wiring_status(
        result, failure,
        applied_message="cascada de preempcion cortada con frees diferidos",
        patch_name=patcher.patch_name)


__all__ = ["apply", "GENESIS_PN121_MARKER", "ANCHOR_OLD", "ANCHOR_NEW"]
