# SPDX-License-Identifier: Apache-2.0
"""PN119 — despacha el MLP de prefill por SK-12 (gate_up + SiLU fusionado).

Es un TEXT PATCH, igual que PN118 y por el mismo motivo: vLLM carga y ejecuta
el modelo en procesos WORKER donde `apply_all` no corre, asi que un monkeypatch
sobre la clase nunca llega (verificado: cero lineas de Genesis con prefijo
`Worker_TP` en el log). P113 y PN116 son monkeypatches y por eso nunca
hicieron nada.

Depende de PN118: consume los pesos INT8 que PN118 deja en `gate_up_proj`.
Sin PN118 no hay estado que despachar y el forward original queda intacto.

Solo actua con M >= GENESIS_PN119_M_MIN (default 512), o sea en prefill.
"""

from __future__ import annotations

import logging

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

log = logging.getLogger("genesis.wiring.pn119_sk12_mlp_forward")

GENESIS_PN119_MARKER = "[Genesis PN119: MLP de prefill por SK-12]"

ANCHOR_OLD = (
    "    def forward(self, x):\n"
    "        gate_up, _ = self.gate_up_proj(x)\n"
    "        out = self.act_fn(gate_up)\n"
    "        out, _ = self.down_proj(out)\n"
)

# El import va al TOPE del archivo, no adentro del forward: el forward lo traza
# dynamo, y ahi un import dispara la registracion del custom op durante el
# trazado -> "Attempted to call function marked as skipped".
IMPORT_OLD = "import torch.nn.functional as F\n"
IMPORT_NEW = (
    "import torch.nn.functional as F\n"
    "# " + GENESIS_PN119_MARKER + " import a nivel de modulo a proposito\n"
    "from vllm._genesis import sk12_mlp_dispatch as _g119\n"
)

ANCHOR_NEW = (
    "    def forward(self, x):\n"
    "        # " + GENESIS_PN119_MARKER + "\n"
    "        # SK-12 fusiona el GEMM con SiLU(gate)*up y devuelve [M, N], asi\n"
    "        # que reemplaza gate_up_proj Y act_fn de una. Devuelve None en\n"
    "        # decode (M chico) y si PN118 no convirtio la capa.\n"
    "        _act = _g119.gate_up_silu(self.gate_up_proj, x)\n"
    "        if _act is not None:\n"
    "            out, _ = self.down_proj(_act)\n"
    "            if self.expert_gate is not None:\n"
    "                out = F.sigmoid(self.expert_gate(x)[0]) * out\n"
    "            return out\n"
    "        gate_up, _ = self.gate_up_proj(x)\n"
    "        out = self.act_fn(gate_up)\n"
    "        out, _ = self.down_proj(out)\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("model_executor/models/qwen2_moe.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN119 MLP de prefill por SK-12",
        target_file=str(target),
        marker=GENESIS_PN119_MARKER,
        sub_patches=[
            TextPatch(name="pn119_import", anchor=IMPORT_OLD,
                      replacement=IMPORT_NEW, required=True),
            TextPatch(name="pn119_mlp_forward", anchor=ANCHOR_OLD,
                      replacement=ANCHOR_NEW, required=True),
        ],
        upstream_drift_markers=[],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN119")
    log_decision("PN119", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root no localizable"
    patcher = _make_patcher()
    if patcher is None:
        return "failed", "qwen2_moe.py no encontrado"
    result, failure = patcher.apply()
    return result_to_wiring_status(
        result, failure,
        applied_message="MLP de prefill despachado por SK-12 (gate_up+SiLU fusionado)",
        patch_name=patcher.patch_name)


__all__ = ["apply", "GENESIS_PN119_MARKER", "ANCHOR_OLD", "ANCHOR_NEW",
           "IMPORT_OLD", "IMPORT_NEW"]
