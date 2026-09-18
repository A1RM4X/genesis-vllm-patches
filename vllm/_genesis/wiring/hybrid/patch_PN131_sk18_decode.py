# SPDX-License-Identifier: Apache-2.0
"""PN131 — decode de atencion entero (SK-18h, PTX) sobre KV int8_per_token_head.

Ver ``vllm._genesis.sk18_attn``. Engancha en ``triton_attn.py``:
* ``do_kv_cache_update``: escritura con layout propio (K por token, V por dimension,
  escalas int16) dentro de la misma reserva de 520 B por token-cabeza.
* ``forward``: decode (<= 5 tokens por pedido) por SK-18h; pasos con prefill por
  decuantizacion + Triton fp16.
* builder: ``query_start_loc_cpu`` en la metadata (sin sincronizar) y soporte de
  CUDA graph UNIFORM_BATCH: el decode uniforme (1 + tokens de MTP) va en FULL graph;
  los lanzamientos PTX son cuLaunchKernel y quedan grabados.
"""

from __future__ import annotations

import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN131: SK-18 decode entero]"

IMPORT_OLD = "from vllm.logger import init_logger\n"
IMPORT_NEW = IMPORT_OLD + "from vllm._genesis import sk18_attn as _g131  # " + MARKER + "\n"

CG_OLD = "    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS\n"
CG_NEW = CG_OLD + (
    "\n"
    "    @classmethod\n"
    "    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):  # " + MARKER + "\n"
    "        # decode entero PTX: FULL graph solo para decode uniforme (1 + tokens MTP)\n"
    "        if _g131.dtype_activo(getattr(vllm_config.cache_config, \"cache_dtype\", \"\")):\n"
    "            return AttentionCGSupport.UNIFORM_BATCH\n"
    "        return cls._cudagraph_support\n"
)

META_OLD = "        mm_ranges = common_attn_metadata.mm_req_doc_ranges\n"
META_NEW = (
    "        attn_metadata.genesis_qsl_cpu = common_attn_metadata.query_start_loc_cpu  # " + MARKER + "\n"
    + META_OLD
)

FWD_OLD = "        assert attn_metadata.use_cascade is False\n"
FWD_NEW = FWD_OLD + (
    "        if _g131.activo(self, layer):  # " + MARKER + "\n"
    "            return _g131.forward(self, layer, query, kv_cache, attn_metadata, output)\n"
)

UPD_OLD = "        # Reshape the input keys and values and store them in the cache.\n"
UPD_NEW = (
    "        if _g131.activo(self, layer):  # " + MARKER + "\n"
    "            return _g131.escribir(self, layer, key, value, kv_cache, slot_mapping)\n"
    + UPD_OLD
)


# El enganche del camino nativo: UNA ancla, en el cuerpo de `load_general_plugins()`.
# Ese archivo es IDENTICO en v0.27.1 y v0.29.0, la funcion corre en TODOS los procesos
# (servidor y workers) y una sola vez por proceso. Es lo que reemplaza a las cinco anclas
# fragiles de triton_attn.py: todo lo demas pasa a resolverse por herencia.
PLUG_OLD = (
    "    plugins = load_plugins_by_group(group=DEFAULT_PLUGINS_GROUP)\n"
    "    # general plugins, we only need to execute the loaded functions\n"
    "    for func in plugins.values():\n"
    "        func()\n"
)
PLUG_NEW = (
    "    # " + MARKER + " registros de Genesis que TIENEN que correr dentro de este proceso:\n"
    "    # apply_all corre aparte y hace exec, asi que lo que registre en memoria se pierde.\n"
    "    try:\n"
    "        from vllm._genesis import plugins_arranque as _genesis_plug\n"
    "        _genesis_plug.cargar()\n"
    "    except Exception:\n"
    "        pass\n"
    + PLUG_OLD
)


def _nativo() -> tuple[str, str]:
    """PN131 por el registro de backends de vLLM, en vez de por las cinco anclas de texto.

    Es el mismo PN131 — ver ``vllm._genesis.sk18_backend`` — pero enganchado por donde v0.29.0
    dice que hay que engancharse. Los dos caminos hacen lo mismo y NO pueden convivir: si se
    registra el backend por subclase y ademas se parchea el texto, el decode queda envuelto dos
    veces. Por eso son excluyentes y no dos parches distintos.

    Vale la pena porque el camino de anclas falla CALLADO: si upstream mueve una linea, el
    parche se saltea, la atencion cae al kernel generico y lo unico que se nota es que anda mas
    lento. La subclase, en cambio, revienta al importar si el padre dejo de definir lo que
    reemplaza. Y el ancla que queda es una sola, en un archivo que upstream no toco entre
    versiones.
    """
    try:
        from vllm.v1.attention.backends.registry import register_backend  # noqa: F401
    except Exception as e:                                               # noqa: BLE001
        return "skipped", (f"esta version de vLLM no tiene registro de backends "
                           f"({type(e).__name__}): usar el camino de anclas")

    destino = resolve_vllm_file("plugins/__init__.py")
    if destino is None:
        return "failed", "no encontre vllm/plugins/__init__.py para enganchar el registro"
    p = TextPatcher(
        patch_name="PN131 registro nativo (plugins/__init__.py)", target_file=str(destino),
        marker=MARKER,
        sub_patches=[TextPatch(name="pn131_plug_hook", anchor=PLUG_OLD, replacement=PLUG_NEW,
                               required=True)],
        upstream_drift_markers=[])
    result, failure = p.apply()
    estado, msg = result_to_wiring_status(
        result, failure, applied_message="registro nativo enganchado", patch_name=p.patch_name)
    if estado != "applied":
        return estado, msg
    return "applied", ("PN131 entra por register_backend desde load_general_plugins "
                       "(1 ancla en vez de 5, el resto por herencia)")


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN131")
    log_decision("PN131", decision, reason)
    if not decision:
        return "skipped", reason
    if os.environ.get("GENESIS_PN131_NATIVO", "0").strip().lower() in ("1", "true", "yes", "on"):
        return _nativo()
    if vllm_install_root() is None:
        return "skipped", "vllm install root no localizable"
    target = resolve_vllm_file("v1/attention/backends/triton_attn.py")
    if target is None:
        return "failed", "triton_attn.py no encontrado"
    p = TextPatcher(
        patch_name="PN131 SK-18 decode entero", target_file=str(target), marker=MARKER,
        sub_patches=[
            TextPatch(name="pn131_import", anchor=IMPORT_OLD, replacement=IMPORT_NEW, required=True),
            TextPatch(name="pn131_cg", anchor=CG_OLD, replacement=CG_NEW, required=True),
            TextPatch(name="pn131_meta", anchor=META_OLD, replacement=META_NEW, required=True),
            TextPatch(name="pn131_fwd", anchor=FWD_OLD, replacement=FWD_NEW, required=True),
            TextPatch(name="pn131_upd", anchor=UPD_OLD, replacement=UPD_NEW, required=True),
        ],
        upstream_drift_markers=[])
    result, failure = p.apply()
    return result_to_wiring_status(result, failure, applied_message="SK-18 decode entero armado",
                                   patch_name=p.patch_name)
