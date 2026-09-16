# SPDX-License-Identifier: Apache-2.0
"""PN132 — el borrador MTP propone sobre un vocabulario recortado (FR-Spec).

Ver ``vllm._genesis.mtp_vocab``. Engancha en ``compute_logits`` del modelo MTP: en vez de
proyectar contra las 248.320 entradas del vocabulario una vez por token propuesto, proyecta
contra las mas frecuentes y devuelve el resto en -inf. La verificacion la sigue haciendo el
modelo grande con el vocabulario completo, asi que la salida no cambia de distribucion.
"""

from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN132: vocabulario recortado del borrador]"

OLD = (
    "    ) -> torch.Tensor | None:\n"
    "        return self.logits_processor(self.lm_head, hidden_states)\n"
)
NEW = (
    "    ) -> torch.Tensor | None:\n"
    "        from vllm._genesis import mtp_vocab as _g132  # " + MARKER + "\n"
    "        if _g132.activo():  # " + MARKER + "\n"
    "            _l = _g132.logits(self, hidden_states)  # " + MARKER + "\n"
    "            if _l is not None:  # " + MARKER + "\n"
    "                return _l  # " + MARKER + "\n"
    "        return self.logits_processor(self.lm_head, hidden_states)\n"
)

ARCHIVOS = ("model_executor/models/qwen3_5_mtp.py", "model_executor/models/qwen3_next_mtp.py")


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN132")
    log_decision("PN132", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root no localizable"
    hechos = []
    for rel in ARCHIVOS:
        target = resolve_vllm_file(rel)
        if target is None:
            continue
        p = TextPatcher(
            patch_name=f"PN132 vocabulario del borrador ({rel.split('/')[-1]})",
            target_file=str(target), marker=MARKER,
            sub_patches=[TextPatch(name="pn132_logits", anchor=OLD, replacement=NEW, required=True)],
            upstream_drift_markers=[])
        result, failure = p.apply()
        estado, motivo = result_to_wiring_status(
            result, failure, applied_message="vocabulario del borrador recortado",
            patch_name=p.patch_name)
        hechos.append((rel, estado, motivo))
    if not hechos:
        return "failed", "no se encontro ningun modelo MTP para parchear"
    ok = [h for h in hechos if h[1] == "applied"]
    if not ok:
        return hechos[0][1], hechos[0][2]
    return "applied", f"vocabulario del borrador recortado en {len(ok)} modelo(s) MTP"
