# SPDX-License-Identifier: Apache-2.0
"""PN137 — suavizado por grupo plegado en los pesos (ver ``vllm._genesis.suave_grupo``).

Son DOS mitades que tienen que ir juntas:

1. **Las escalas del peso**, al principio de ``process_weights_after_loading`` del camino Marlin.
   Tiene que ser ahi y no despues: Marlin PERMUTA las escalas, y una vez permutadas el indice de
   grupo ya no corresponde.
2. **El peso de la RMSNorm**, en ``GPUModelRunner.load_model``, cuando el modelo entero ya existe y
   se pueden aparear norma y lineal.

Si se aplicara una sola mitad el modelo daria basura en silencio, asi que ``suave_grupo.verificar()``
corta el arranque si las cuentas no cierran.
"""

from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN137: suavizado por grupo]"

# ── 1) las escalas del peso, antes de que Marlin las permute ────────────────────────────────
DEST_PESO = "model_executor/kernels/linear/mixed_precision/marlin.py"
PESO_VIEJO = (
    "    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:\n"
    "        device = getattr(layer, self.w_q_name).device\n"
)
PESO_NUEVO = (
    PESO_VIEJO
    + "        from vllm._genesis import suave_grupo as _g137  # " + MARKER + "\n"
    + "        if _g137.activo():  # " + MARKER + "\n"
    + "            _g137.aplicar_al_peso(layer, self.w_s_name)  # " + MARKER + "\n"
)

# ── 1b) la cuantizacion con escala fija, que es el premio: se va la reduccion por fila ───────
DEST_QUANT = "model_executor/layers/quantization/utils/marlin_utils.py"
QUANT_VIEJO = (
    "        reshaped_x, a_scales = marlin_quant_input(reshaped_x, input_dtype)\n"
    "        a_scales = a_scales * input_global_scale\n"
)
QUANT_NUEVO = (
    "        from vllm._genesis import suave_grupo as _g137q  # " + MARKER + "\n"
    "        _g137q_e = _g137q.escala_actual()  # " + MARKER + "\n"
    "        if _g137q_e > 0.0:  # " + MARKER + "\n"
    "            reshaped_x, a_scales = _g137q.cuantizar_estatico(reshaped_x, _g137q_e)  # "
    + MARKER + "\n"
    "            a_scales = a_scales * input_global_scale  # " + MARKER + "\n"
    "        else:  # " + MARKER + "\n"
    "            reshaped_x, a_scales = marlin_quant_input(reshaped_x, input_dtype)\n"
    "            a_scales = a_scales * input_global_scale\n"
)

# La escala de la capa se deja anotada justo antes de llamar al GEMM, que es donde todavia existe
# el objeto `layer`.
MARCA_VIEJO = (
    "        c = self.config\n"
    "        w_q, w_s, w_zp, w_gidx = self._get_weight_params(layer)\n"
    "\n"
    "        # `process_weights_after_loading` will ensure w_zp and w_gidx are not\n"
)
MARCA_NUEVO = (
    MARCA_VIEJO
    + "        from vllm._genesis import suave_grupo as _g137m  # " + MARKER + "\n"
    + "        _g137m.marcar(getattr(layer, 'prefix', ''))  # " + MARKER + "\n"
)

# ── 2) el peso de la norma, con el modelo ya armado ─────────────────────────────────────────
DEST_NORMA = "v1/worker/gpu_model_runner.py"
NORMA_VIEJO = (
    "                self.model = model_loader.load_model(\n"
    "                    vllm_config=self.vllm_config, model_config=self.model_config\n"
    "                )\n"
)
NORMA_NUEVO = (
    NORMA_VIEJO
    + "                from vllm._genesis import suave_grupo as _g137n  # " + MARKER + "\n"
    + "                _g137n.plegar_normas(self.model)  # " + MARKER + "\n"
)


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN137")
    log_decision("PN137", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root no localizable"

    # Agrupados POR ARCHIVO a proposito: ``TextPatcher`` ve el marcador y da el archivo por
    # parcheado, asi que dos patchers sobre el mismo archivo dejan el segundo mudo y sin efecto.
    hechos = []
    for rel, trozos in (
        (DEST_PESO, (("pn137_peso", PESO_VIEJO, PESO_NUEVO),
                     ("pn137_marca", MARCA_VIEJO, MARCA_NUEVO))),
        (DEST_QUANT, (("pn137_quant", QUANT_VIEJO, QUANT_NUEVO),)),
        (DEST_NORMA, (("pn137_norma", NORMA_VIEJO, NORMA_NUEVO),)),
    ):
        destino = resolve_vllm_file(rel)
        if destino is None:
            return "failed", f"{rel} no encontrado"
        p = TextPatcher(
            patch_name=f"PN137 suavizado ({rel.rsplit('/', 1)[-1]})", target_file=str(destino),
            marker=MARKER,
            sub_patches=[TextPatch(name=n, anchor=v, replacement=w, required=True)
                         for n, v, w in trozos],
            upstream_drift_markers=[])
        r, f = p.apply()
        hechos.append(result_to_wiring_status(r, f, applied_message="mitad enganchada",
                                              patch_name=p.patch_name))
    malos = [h for h in hechos if h[0] == "failed"]
    if malos:
        return malos[0]
    return "applied", "suavizado por grupo enganchado (peso + norma)"
