# SPDX-License-Identifier: Apache-2.0
"""PN136 — el all-reduce del MLP viaja mientras se calcula el resto del bloque.

Ver ``vllm._genesis.mlp_solapado``.

Engancha en ``GPUModelRunner.load_model``, JUSTO DESPUES de que el cargador devuelve el modelo.
Ese punto es el bueno por dos motivos: ya estan los pesos pero todavia no se capturaron los grafos
ni se midio cuanta memoria queda para la KV (que es cuando hay que reservar los buzones), y **no
depende del archivo del modelo**. El primer intento engancho el ultimo
``return loader.load_weights(...)`` de ``qwen3_next.py``, y resulto que este checkpoint usa
``Qwen3_5ForCausalLM`` de ``qwen3_5.py``, donde ademas ese "ultimo return" cae en otra clase del
mismo archivo: el parche se aplicaba y no corria nunca.
"""

from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN136: MLP solapado]"

DESTINO = "v1/worker/gpu_model_runner.py"

VIEJO = (
    "                self.model = model_loader.load_model(\n"
    "                    vllm_config=self.vllm_config, model_config=self.model_config\n"
    "                )\n"
)
NUEVO = (
    VIEJO
    + "                from vllm._genesis import mlp_solapado as _g136  # " + MARKER + "\n"
    + "                _g136.preparar(self.model)  # " + MARKER + "\n"
)


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN136")
    log_decision("PN136", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root no localizable"

    target = resolve_vllm_file(DESTINO)
    if target is None:
        return "failed", "gpu_model_runner.py no encontrado"
    p = TextPatcher(
        patch_name="PN136 MLP solapado", target_file=str(target), marker=MARKER,
        sub_patches=[TextPatch(name="pn136_preparar", anchor=VIEJO, replacement=NUEVO,
                               required=True)],
        upstream_drift_markers=[])
    r, f = p.apply()
    return result_to_wiring_status(r, f, applied_message="buzones P2P cableados",
                                   patch_name=p.patch_name)
