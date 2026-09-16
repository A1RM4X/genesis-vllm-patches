# SPDX-License-Identifier: Apache-2.0
"""Wiring de Genesis PN115 — admisión por headroom de KV + bypass por prioridad.

Intercepta `vllm/v1/core/sched/scheduler.py` en dos puntos:

1. Hook de paso: registra la cadencia del scheduler y el estado del BlockPool
   para la telemetría y (opcionalmente) para el PID de latencia.

2. Bucle de admisión de la cola de waiting:
   - difiere un request cuando admitirlo dejaría al motor sin headroom
     físico de bloques KV (evita fallos de `allocate_slots` y tormentas de
     preemption);
   - permite que un request de prioridad alta desaloje a uno de prioridad
     menor cuando el motor está en `max_num_seqs`.

El bypass hace la preemption con el MISMO rollback de estado que el camino
de preemption de upstream (`Scheduler.schedule`) y después corta el paso:
para cuando corre el bucle de waiting, la víctima ya fue agendada en este
mismo paso, así que sacarla de `self.running` sin deshacer
`num_scheduled_tokens` / `req_to_new_blocks` / `scheduled_spec_decode_tokens`
dejaría al model runner ejecutando un request cuyos bloques KV se acaban de
liberar. Cortar el paso además respeta el invariante que el propio upstream
declara: todo el bucle de waiting vive bajo `if not preempted_reqs:`, o sea
que un paso que preempta no admite trabajo nuevo.

La lógica vive en `vllm/_genesis/dynamic_pid_gating.py`; acá sólo está el
cableado.
"""

from __future__ import annotations

import logging

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

log = logging.getLogger("genesis.wiring.pn115_dynamic_pid_admission")

GENESIS_PN115_MARKER = "[Genesis PN115: KV headroom gating & priority bypass v2]"


# ─────────── 1. Hook de paso: cadencia + estado del BlockPool ───────────

STEP_RECORD_OLD = (
    "        # For logging.\n"
    "        scheduled_timestamp = time.monotonic()\n"
)

STEP_RECORD_NEW = (
    "        # For logging.\n"
    "        scheduled_timestamp = time.monotonic()\n"
    "        # " + GENESIS_PN115_MARKER + "\n"
    "        from vllm._genesis import dynamic_pid_gating as _g115\n"
    "        _g115.update_step_with_scheduler(\n"
    "            scheduler=self,\n"
    "            step_duration_ms=(scheduled_timestamp - getattr(self, '_g115_last_step_start', scheduled_timestamp)) * 1000.0,\n"
    "        )\n"
    "        self._g115_last_step_start = scheduled_timestamp\n"
)


# ─────────── 2. Admisión: gating por headroom + bypass por prioridad ───────────
#
# El ancla arranca en `step_skipped_waiting = ...` para poder izar el import
# fuera del `while`: el bucle de waiting es camino caliente y hacer el import
# por iteración es un lookup en sys.modules por request.

ADMISSION_GATE_OLD = (
    "            step_skipped_waiting = create_request_queue(self.policy)\n"
    "\n"
    "            while (self.waiting or self.skipped_waiting) and token_budget > 0:\n"
    "                # Paused streaming sessions (WAITING_FOR_STREAMING_REQ) are not\n"
    "                # in `running` but still hold a model-runner request slot.\n"
    "                num_running = len(self.running) + self.num_waiting_for_streaming_input\n"
    "                if num_running >= self.max_num_running_reqs:\n"
    "                    break\n"
    "\n"
    "                request_queue = self._select_waiting_queue_for_scheduling()\n"
    "                assert request_queue is not None\n"
    "\n"
    "                request = request_queue.peek_request()\n"
    "                request_id = request.request_id\n"
)

ADMISSION_GATE_NEW = (
    "            step_skipped_waiting = create_request_queue(self.policy)\n"
    "            # " + GENESIS_PN115_MARKER + " — import izado: una vez por\n"
    "            # schedule(), no una vez por request en el bucle caliente.\n"
    "            from vllm._genesis import dynamic_pid_gating as _g115\n"
    "\n"
    "            while (self.waiting or self.skipped_waiting) and token_budget > 0:\n"
    "                # Paused streaming sessions (WAITING_FOR_STREAMING_REQ) are not\n"
    "                # in `running` but still hold a model-runner request slot.\n"
    "                num_running = len(self.running) + self.num_waiting_for_streaming_input\n"
    "                if num_running >= self.max_num_running_reqs:\n"
    "                    # " + GENESIS_PN115_MARKER + " Bypass de prioridad:\n"
    "                    # libera UN slot para el head de la cola si es de\n"
    "                    # prioridad alta, deshaciendo el estado que la victima\n"
    "                    # ya tenia agendado en este paso (mismo rollback que el\n"
    "                    # camino de preemption de upstream), y corta el paso.\n"
    "                    # Upstream ya define que un paso que preempta no admite\n"
    "                    # trabajo nuevo: este bloque entero vive bajo\n"
    "                    # `if not preempted_reqs:`. Sin bypass el delta es 0 y\n"
    "                    # esto es exactamente el `break` original.\n"
    "                    _g115_tok, _g115_enc, _g115_idx = _g115.try_priority_preempt(\n"
    "                        scheduler=self,\n"
    "                        timestamp=scheduled_timestamp,\n"
    "                        preempted_reqs=preempted_reqs,\n"
    "                        scheduled_running_reqs=scheduled_running_reqs,\n"
    "                        num_scheduled_tokens=num_scheduled_tokens,\n"
    "                        req_to_new_blocks=req_to_new_blocks,\n"
    "                        scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,\n"
    "                        scheduled_encoder_inputs=scheduled_encoder_inputs,\n"
    "                    )\n"
    "                    token_budget += _g115_tok\n"
    "                    encoder_compute_budget += _g115_enc\n"
    "                    req_index += _g115_idx\n"
    "                    break\n"
    "\n"
    "                request_queue = self._select_waiting_queue_for_scheduling()\n"
    "                assert request_queue is not None\n"
    "\n"
    "                request = request_queue.peek_request()\n"
    "                request_id = request.request_id\n"
    "\n"
    "                # " + GENESIS_PN115_MARKER + " Admision por headroom de KV.\n"
    "                if _g115.should_gate_waiting(request, self):\n"
    "                    request_queue.pop_request()\n"
    "                    step_skipped_waiting.prepend_request(request)\n"
    "                    continue\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/core/sched/scheduler.py")
    if target is None:
        log.warning("[PN115] v1/core/sched/scheduler.py no encontrado bajo %s", vllm_install_root())
        return None

    return TextPatcher(
        patch_name="PN115 KV headroom gating & priority bypass",
        target_file=str(target),
        sub_patches=[
            TextPatch(
                name="pn115_step_record",
                anchor=STEP_RECORD_OLD,
                replacement=STEP_RECORD_NEW,
                required=True,
            ),
            TextPatch(
                name="pn115_admission_gate",
                anchor=ADMISSION_GATE_OLD,
                replacement=ADMISSION_GATE_NEW,
                required=True,
            ),
        ],
        marker=GENESIS_PN115_MARKER,
    )


def apply() -> tuple[str, str]:
    """Aplica PN115. La decisión pasa por el dispatcher, como el resto."""
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN115")
    log_decision("PN115", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root no localizable"

    patcher = _make_patcher()
    if patcher is None:
        return "failed", "v1/core/sched/scheduler.py no encontrado"

    result, failure = patcher.apply()
    return result_to_wiring_status(
        result,
        failure,
        applied_message=(
            "PN115 aplicado: admision por headroom de bloques KV + bypass por "
            "prioridad con rollback completo del estado del paso"
        ),
        patch_name=patcher.patch_name,
    )
