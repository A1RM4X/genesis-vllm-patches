# SPDX-License-Identifier: Apache-2.0
"""Sonda de estadistica de activaciones (ver ``vllm._genesis.act_stats``).

Engancha en ``apply_gptq_marlin_linear`` de ``marlin_utils.py``, que es el camino caliente de
verdad (``apply_weights`` de kernels/linear no se usa con este checkpoint). Ahi no llega el objeto
``layer``. La identidad se toma de las FORMAS (K y N), que agrupa las 64 capas del mismo tipo en
un solo cubo: eso mide la dispersion PESIMISTA, porque mezcla la variacion entre capas con la
variacion entre tokens. Si aun asi la distribucion es angosta, la escala estatica es segura.

(Se probo identificar por ``weight.data_ptr()``, que seria por capa: dynamo no lo puede trazar,
tira ``NotImplementedError: DataPtrVariable``.)

Es solo telemetria: no cambia el camino de ejecucion y esta apagada salvo ``GENESIS_ACT_STATS=1``.
"""

from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis act_stats]"

DESTINO = "model_executor/layers/quantization/utils/marlin_utils.py"

# El cuerpo de apply_gptq_marlin_linear y el de apply_awq_marlin_linear arrancan IGUAL, asi que el
# ancla arranca en la cola de la firma, que solo tiene el primero.
VIEJO = (
    "    is_k_full: bool,\n"
    "    input_global_scale: torch.Tensor | None = None,\n"
    "    bias: torch.Tensor | None = None,\n"
    "    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,\n"
    "    input_dtype: torch.dtype | None = None,\n"
    ") -> torch.Tensor:\n"
    "    reshaped_x = input.reshape(-1, input.shape[-1])\n"
    "    out_shape = input.shape[:-1] + (output_size_per_partition,)\n"
)
NUEVO = (
    VIEJO
    + "    from vllm._genesis import act_stats as _gas  # " + MARKER + "\n"
    + "    if _gas.activo():  # " + MARKER + "\n"
    + "        torch.ops.genesis.act_stats(reshaped_x, input_size_per_partition, "
      "output_size_per_partition)  # " + MARKER + "\n"
)


def apply() -> tuple[str, str]:
    if vllm_install_root() is None:
        return "skipped", "vllm install root no localizable"
    target = resolve_vllm_file(DESTINO)
    if target is None:
        return "failed", "marlin_utils.py no encontrado"
    p = TextPatcher(
        patch_name="Genesis act_stats", target_file=str(target), marker=MARKER,
        sub_patches=[TextPatch(name="actstats", anchor=VIEJO, replacement=NUEVO, required=True)],
        upstream_drift_markers=[])
    r, f = p.apply()
    return result_to_wiring_status(r, f, applied_message="sonda de activaciones",
                                   patch_name=p.patch_name)
