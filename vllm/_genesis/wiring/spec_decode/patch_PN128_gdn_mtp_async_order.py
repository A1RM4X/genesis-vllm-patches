# SPDX-License-Identifier: Apache-2.0
"""PN128 — el input-prep async espera al postproceso del spec decode anterior.

Bug (club-3090#1052, vllm#52873 / hilo de #50021): hibrido GDN + MTP + prefix
caching + scheduling async (default de 0.27.1). ``synchronize_input_prep``
espera ``prepare_inputs_event``, pero ese evento se graba al FINAL del
input-prep, ANTES del forward y del postproceso fused-align del spec decode. El
``_update_states`` del paso siguiente muta la tabla de bloques mientras el
postproceso todavia lee esos buffers -> indice de bloque de estado viejo ->
escritura GPU ilegal en gdn_attn (Xid 31 VIRT_WRITE). club-3090: 5/5 corridas
sin el fix murieron a los 6-13k tokens generados; con el fix 3/3 llegaron a
33-36k, TPS neutro.

Arreglo (2 lineas, fix de club-3090, no es PR upstream): esperar tambien
``num_accepted_tokens_event``, que se graba DESPUES del postproceso. Solo con
spec decode (el evento es None si no) y async (sin evento el metodo no hace
nada). Un evento sin grabar cuenta como completo.
"""

from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN128: orden async del spec decode GDN/MTP]"

ANCHOR_OLD = (
    "        # Ensure prior step has finished with reused CPU tensors.\n"
    "        # This is required in the async scheduling case because\n"
    "        # the CPU->GPU transfer happens async.\n"
    "        self.prepare_inputs_event.synchronize()\n"
)
ANCHOR_NEW = ANCHOR_OLD + (
    "        # " + MARKER + " prepare_inputs_event se graba ANTES del forward y\n"
    "        # del postproceso del spec decode: esperar tambien a ese postproceso\n"
    "        # para que _update_states no mute buffers que todavia se leen.\n"
    "        if self.num_accepted_tokens_event is not None:\n"
    "            self.num_accepted_tokens_event.synchronize()\n"
)


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN128")
    log_decision("PN128", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root no localizable"
    target = resolve_vllm_file("v1/worker/gpu_model_runner.py")
    if target is None:
        return "failed", "gpu_model_runner.py no encontrado"
    p = TextPatcher(
        patch_name="PN128 orden async spec decode GDN/MTP", target_file=str(target), marker=MARKER,
        sub_patches=[TextPatch(name="pn128_sync", anchor=ANCHOR_OLD, replacement=ANCHOR_NEW, required=True)],
        # La llamada ya existe mas abajo en _prepare_inputs (modo align): no sirve de
        # marcador de "ya incluido upstream". Se detecta por el ancla exacta.
        upstream_drift_markers=[])
    result, failure = p.apply()
    return result_to_wiring_status(result, failure, applied_message="input-prep espera al postproceso del spec decode",
                                   patch_name=p.patch_name)
