# SPDX-License-Identifier: Apache-2.0
"""Sonda de activaciones y prueba de la escala estatica.

Dos cosas en el mismo punto, las dos apagadas por omision:
  * ``act_stats``: histograma de amax por token, para decidir si la escala puede ser fija;
  * ``escala_estatica``: deja la activacion sobre la rejilla de la escala fija, para medir en el
    servidor que calidad se pierde ANTES de escribir ningun kernel.

Engancha en ``apply_weights`` del camino Marlin, que es donde todavia existe el objeto
``layer`` y por lo tanto se puede identificar cada lineal por su prefijo: la estadistica sale POR
CAPA, que es lo que hace falta para separar la variacion entre capas de la variacion entre tokens.

(Antes se engancho en ``marlin_utils.apply_gptq_marlin_linear``, donde no llega el ``layer``:
ahi la identidad tenia que ser la forma, que junta las 64 capas del mismo tipo. Y usar
``weight.data_ptr()`` no sirve: dynamo no lo puede trazar.)

Es solo telemetria: no cambia el camino de ejecucion y esta apagada salvo ``GENESIS_ACT_STATS=1``.
"""

from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis act_stats]"

DESTINO = "model_executor/kernels/linear/mixed_precision/marlin.py"

# Aca SI llega el objeto `layer`, asi que la estadistica sale POR CAPA en vez de por forma.
VIEJO = (
    "        c = self.config\n"
    "        w_q, w_s, w_zp, w_gidx = self._get_weight_params(layer)\n"
)
NUEVO = (
    VIEJO
    + "        _gas_nombre = getattr(layer, 'prefix', 'sin_nombre')  # " + MARKER + "\n"
    + "        from vllm._genesis import act_stats as _gas  # " + MARKER + "\n"
    + "        if _gas.activo():  # " + MARKER + "\n"
    + "            torch.ops.genesis.act_stats(x.reshape(-1, x.shape[-1]), _gas_nombre)  # "
      + MARKER + "\n"
    + "        from vllm._genesis import escala_estatica as _gee  # " + MARKER + "\n"
    + "        if _gee.activo():  # " + MARKER + "\n"
    + "            _gee_s = _gee.escala_de(_gas_nombre)  # " + MARKER + "\n"
    + "            if _gee_s > 0.0:  # " + MARKER + "\n"
    + "                x = x.clone()  # " + MARKER + "\n"
    + "                torch.ops.genesis.escala_estatica(x, _gee_s)  # " + MARKER + "\n"
)


def apply() -> tuple[str, str]:
    if vllm_install_root() is None:
        return "skipped", "vllm install root no localizable"
    target = resolve_vllm_file(DESTINO)
    if target is None:
        return "failed", "marlin.py (kernels/linear) no encontrado"
    p = TextPatcher(
        patch_name="Genesis act_stats", target_file=str(target), marker=MARKER,
        sub_patches=[TextPatch(name="actstats", anchor=VIEJO, replacement=NUEVO, required=True)],
        upstream_drift_markers=[])
    r, f = p.apply()
    return result_to_wiring_status(r, f, applied_message="sonda de activaciones",
                                   patch_name=p.patch_name)
