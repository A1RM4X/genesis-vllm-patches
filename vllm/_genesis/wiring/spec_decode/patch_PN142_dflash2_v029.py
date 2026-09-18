# SPDX-License-Identifier: Apache-2.0
"""PN142 — DFlash2 utilizable en vLLM v0.29.0: las dos mitades que upstream todavia no trae.

Portado de club-3090 (commits b9fa3115 y ce7816ec), que lo valido en imagen. Son DOS
arreglos independientes que hacen falta juntos; con cualquiera de los dos afuera, DFlash2 o
no arranca o "anda" sirviendo basura.

1) vllm#51581 — ABIERTO desde el 9/8/2026. ``qwen3_dflash.py`` no tiene NINGUNA conciencia
   de cuantizacion (cero referencias a qweight / weight_packed / quant_method), y arma sus
   buffers de KV fusionada con

       kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]

   Con un borrador de pesos cuantizados eso hace una de dos cosas, las dos malas:
     * revienta con "AttributeError: 'QKVParallelLinear' object has no attribute 'weight'",
       que es lo que pasa en v0.29.0 con un borrador W4A16; o
     * CORROMPE EN SILENCIO, cuando `.weight` existe pero guarda filas empaquetadas: se
       rebanan y se le pasan a F.linear, y sale basura sin un solo error.
   El arreglo inyecta ``_dense_kv_rows()``, que desempaqueta y decuantiza ANTES de rebanar.
   Corre desde load_weights, o sea antes del repack de Marlin, asi que weight_packed y
   weight_scale siguen en el layout del checkpoint. Un `.weight` denso de 2 ejes sale por el
   camino corto, asi que un borrador sin cuantizar no se entera.

   Alcance: compressed-tensors pack-quantized W4A16/W8A16 simetrico por grupo, mas el caso
   ya denso. FP8 / NVFP4 / MXFP4 / GPTQ-AWQ NO estan cubiertos aca.

2) fa5017a5 — aterrizo en main el 10/9, UN DIA despues del corte de v0.29.0, y no esta en
   ningun tag publicado. ``Scheduler._mamba_block_aligned_split()`` retrocede una pagina la
   ultima posicion cacheable cuando ``self.use_eagle`` es true, y ``use_eagle()`` devuelve
   true tambien para dflash/dspark. Resultado: el estado recurrente de mamba nunca se
   materializa en el ultimo borde y TODA busqueda de prefix-cache y de tier de offload
   converge a 0 — guarda pero no sirve un solo hit (vllm#53505).
   El arreglo agrega ``use_eagle_preserves_target_kv_cache()``, que es true solo para
   eagle/eagle3/mtp, y acota el retroceso a eso. DFlash y DSpark dibujan de su PROPIA KV y
   nunca escriben bloques del target, asi que no deben retroceder.

   Nota para nosotros: con MTP este segundo arreglo NO cambia nada — "mtp" esta en el
   conjunto que SI preserva, asi que el retroceso es correcto en nuestra config actual. Se
   verifico antes de portarlo, justamente porque el sintoma (cero hits de L2/L3) es identico
   al que ya teniamos por otra causa y era facil confundirse.

LA TRAMPA, documentada por club-3090 y verificada por ellos en imagen: el bundle dflash2
VIEJO no se debe re-cablear en v0.29.0+. Su ``_check_applied.py`` no ve el refactor de
upstream, el apply queda en conflicto y el arranque se rechaza. Este parche es la mitad que
sigue haciendo falta, con la que upstream ya absorbio (vllm#52816) sacada.

APAGADO por defecto: sin un checkpoint de borrador DFlash2 no sirve de nada. Para usarlo hace
falta uno — club-3090 menciona gratex/Qwen3.8-27B-DFlash2-W4A16-g128-sym-GPTQ — y cambiar
``--speculative-config`` al metodo dflash.
"""

from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    MultiFilePatchTransaction,
    TextPatch,
    TextPatcher,
)

MARKER = "[Genesis PN142: DFlash2 en v0.29.0]"

# ───────────────────── 1) vllm#51581: decuantizar antes de rebanar ─────────────────────

HELPER = '''def _dense_kv_rows(attn):
    """Filas [q_size:] de la proyeccion qkv como matriz densa. [Genesis PN142]

    Arregla vllm#51581 (ABIERTO): qwen3_dflash.py no sabe de cuantizacion y rebana
    `.weight` directo. Corre desde load_weights, o sea ANTES del repack de Marlin, asi que
    weight_packed y weight_scale siguen en el layout del checkpoint y se pueden decuantizar
    aca. Un `.weight` denso de 2 ejes sale por el camino corto: un borrador sin cuantizar
    toma el camino de siempre.
    """
    import torch

    qkv = attn.qkv_proj
    w = getattr(qkv, "weight", None)
    if w is not None and w.dim() == 2:
        import os as _os
        if _os.environ.get("GENESIS_PN142_DIAG") == "1" and not globals().get("_g142_av1"):
            globals()["_g142_av1"] = True
            print("[PN142 DIAG] camino DENSO: weight %s %s" % (tuple(w.shape), w.dtype), flush=True)
        return w[attn.q_size:]
    packed, scale = qkv.weight_packed, qkv.weight_scale
    # weight_shape solo guarda el ultimo shard cargado de una qkv fusionada: usar los tensores.
    out_f, in_f = int(packed.shape[0]), int(qkv.input_size)
    bits = 32 * packed.shape[1] // in_f
    from compressed_tensors.compressors.pack_quantized.base import unpack_from_int32

    q = unpack_from_int32(packed.data, bits, torch.Size([out_f, in_f]), packed_dim=1)
    group = in_f // scale.shape[1]
    dense = (q.to(torch.float32).reshape(out_f, in_f // group, group)
             * scale.to(torch.float32)[..., None]).reshape(out_f, in_f)
    fuera = scale.dtype if scale.dtype.is_floating_point else torch.bfloat16
    salida = dense.to(fuera)[attn.q_size:]
    import os as _os
    if _os.environ.get("GENESIS_PN142_DIAG") == "1" and not globals().get("_g142_av2"):
        globals()["_g142_av2"] = True
        print("[PN142 DIAG] camino CUANTIZADO: packed %s %s | scale %s %s | bits=%d grupo=%d "
              "| denso %s min=%.4g max=%.4g | salida %s"
              % (tuple(packed.shape), packed.dtype, tuple(scale.shape), scale.dtype, bits, group,
                 tuple(dense.shape), float(dense.min()), float(dense.max()), tuple(salida.shape)),
              flush=True)
    return salida


'''

DF_HELPER_OLD = "@support_torch_compile\nclass DFlashQwen3Model"
DF_HELPER_NEW = HELPER + DF_HELPER_OLD

DF_SLICE_OLD = "        kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]\n"
DF_SLICE_NEW = (
    "        # " + MARKER + " vllm#51581: decuantizar antes de rebanar.\n"
    "        kv_weights = [_dense_kv_rows(a) for a in layers_attn]\n"
)

# ──────────── 2) fa5017a5: acotar el retroceso a los borradores de la familia eagle ────────────

SPEC_OLD = '        return self.method in ("eagle", "eagle3", "mtp", "dflash", "dspark")\n'
SPEC_NEW = (
    SPEC_OLD
    + "\n"
    "    def use_eagle_preserves_target_kv_cache(self) -> bool:\n"
    "        # " + MARKER + " port de fa5017a5 (no esta en v0.29.0).\n"
    "        # Solo los borradores de la familia eagle comparten —y miran hacia adelante en—\n"
    "        # los grupos de atencion completa del target. DFlash y DSpark dibujan de su\n"
    "        # PROPIA KV y nunca escriben bloques del target (vllm#53505).\n"
    '        return self.method in ("eagle", "eagle3", "mtp")\n'
)

SCH_CAMPO_OLD = (
    "        speculative_config = vllm_config.speculative_config\n"
    "        self.use_eagle = False\n"
)
SCH_CAMPO_NEW = (
    SCH_CAMPO_OLD
    + "        # " + MARKER + " acota el retroceso de la ultima pagina de mamba.\n"
    "        self.use_eagle_preserves_target_kv_cache = False\n"
)

SCH_ASIGNA_OLD = "            self.use_eagle = speculative_config.use_eagle()\n"
SCH_ASIGNA_NEW = (
    SCH_ASIGNA_OLD
    + "            # " + MARKER + "\n"
    "            self.use_eagle_preserves_target_kv_cache = (\n"
    "                speculative_config.use_eagle_preserves_target_kv_cache()\n"
    "            )\n"
)

SCH_RETRO_OLD = (
    "        if self.use_eagle:\n"
    "            last_cache_position = max(last_cache_position - block_size, 0)\n"
)
SCH_RETRO_NEW = (
    "        # " + MARKER + " solo la familia eagle poda el ultimo bloque que matchea;\n"
    "        # DFlash/DSpark no, y retroceder les mata todo hit de prefix-cache y de offload.\n"
    "        if self.use_eagle_preserves_target_kv_cache:\n"
    "            last_cache_position = max(last_cache_position - block_size, 0)\n"
)

_ARCHIVOS = [
    ("model_executor/models/qwen3_dflash.py", [
        ("pn142_dflash_helper", DF_HELPER_OLD, DF_HELPER_NEW),
        ("pn142_dflash_slice", DF_SLICE_OLD, DF_SLICE_NEW),
    ]),
    ("config/speculative.py", [
        ("pn142_spec_metodo", SPEC_OLD, SPEC_NEW),
    ]),
    ("v1/core/sched/scheduler.py", [
        ("pn142_sched_campo", SCH_CAMPO_OLD, SCH_CAMPO_NEW),
        ("pn142_sched_asigna", SCH_ASIGNA_OLD, SCH_ASIGNA_NEW),
        ("pn142_sched_retro", SCH_RETRO_OLD, SCH_RETRO_NEW),
    ]),
]


def _helper_compila() -> str | None:
    """El codigo que se INYECTA tiene que compilar por si solo. Devuelve el error, o None.

    No es un lujo: la primera version llevaba `\"\"\" + MARKER + \"\"\"` adentro del literal,
    asi que se inyecto tal cual y vLLM murio con "NameError: name 'MARKER' is not defined" en
    los dos workers de TP — pero recien al CARGAR EL MODELO, mucho despues de que el parche
    dijera "applied". Compilarlo aca mueve ese fallo al momento de aplicar, que es donde se
    puede leer.
    """
    import ast

    try:
        ast.parse(HELPER)
    except SyntaxError as e:
        return f"{type(e).__name__}: {e}"
    return None


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN142")
    log_decision("PN142", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root no localizable"
    mal = _helper_compila()
    if mal is not None:
        return "failed", f"el helper _dense_kv_rows no compila, no se inyecta nada: {mal}"

    patchers = []
    for rel, subs in _ARCHIVOS:
        destino = resolve_vllm_file(rel)
        if destino is None:
            return "skipped", f"{rel} no esta en esta version de vLLM"
        patchers.append(TextPatcher(
            patch_name=f"PN142 DFlash2 v0.29.0 ({rel.rsplit('/', 1)[-1]})",
            target_file=str(destino), marker=MARKER,
            sub_patches=[TextPatch(name=n, anchor=o, replacement=r, required=True)
                         for n, o, r in subs],
            # Cuando un tag publicado traiga fa5017a5, la mitad 2 sobra y hay que soltarla.
            upstream_drift_markers=["use_eagle_preserves_target_kv_cache"],
        ))
    # Los tres archivos van juntos o no va ninguno: aplicar la mitad de esto deja un
    # scheduler que llama a un metodo que no existe.
    return MultiFilePatchTransaction(patchers, name="PN142 DFlash2 v0.29.0").apply_or_skip()
