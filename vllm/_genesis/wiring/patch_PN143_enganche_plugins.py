# SPDX-License-Identifier: Apache-2.0
"""PN143 — la puerta de entrada de Genesis DENTRO de cada proceso de vLLM.

Una sola ancla, en el cuerpo de ``vllm.plugins.load_general_plugins()``, que llama a
``vllm._genesis.plugins_arranque.cargar()``. Hace falta porque ``apply_all`` corre como un
proceso aparte y despues hace ``exec vllm serve``: los parches de texto sobreviven porque
editan archivos, pero todo lo que se registre EN MEMORIA se pierde al cruzar el exec.

Por que es un parche propio y no parte de PN131
-----------------------------------------------
Nacio adentro de PN131 (que lo necesitaba para ``register_backend``) y quedo atado a
``GENESIS_PN131_NATIVO=1``. Eso lo convirtio en una trampa: apagar PN131 apagaba tambien el
volcado de diagnostico, el aviso de disco y cualquier otro enganche de proceso, sin que nada
lo dijera. Se cobro una sesion entera de diagnostico de DFlash2 — el hook de ``propose`` decia
estar instalado (el archivo montado, la variable puesta) y no imprimia una linea, porque
``cargar()`` nunca corria en ese contenedor.

La puerta es infraestructura, no una optimizacion: va siempre. Lo que entra POR ella sigue
decidiendose una por una adentro de ``plugins_arranque.cargar()``, que es idempotente y se
traga sus propios errores.

El archivo ``vllm/plugins/__init__.py`` es identico en v0.27.1 y v0.29.0, y la funcion corre
en TODOS los procesos (servidor y workers) una sola vez por proceso.
"""

from __future__ import annotations

from pathlib import Path

from vllm._genesis.guards import resolve_vllm_file
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN143: enganche de proceso]"

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

# PN131 escribia esta misma llamada con SU marcador. Si ya esta puesta por ese camino, no hay
# nada que hacer: la puerta existe y es la misma.
MARCA_VIEJA = "[Genesis PN131: SK-18 decode entero]"


def apply() -> tuple[str, str]:
    destino = resolve_vllm_file("plugins/__init__.py")
    if destino is None:
        return "failed", "no encontre vllm/plugins/__init__.py"
    texto = Path(destino).read_text(encoding="utf-8")
    if "plugins_arranque" in texto:
        marca = MARCA_VIEJA if MARCA_VIEJA in texto else MARKER
        return "applied", f"la puerta ya estaba puesta ({marca})"
    p = TextPatcher(
        patch_name="PN143 enganche de proceso (plugins/__init__.py)",
        target_file=str(destino), marker=MARKER,
        sub_patches=[TextPatch(name="pn143_plug_hook", anchor=PLUG_OLD, replacement=PLUG_NEW,
                               required=True)],
        upstream_drift_markers=[])
    result, failure = p.apply()
    return result_to_wiring_status(
        result, failure,
        applied_message="cargar() corre en todos los procesos de vLLM",
        patch_name=p.patch_name)
