# SPDX-License-Identifier: Apache-2.0
"""PN81 — huella del FORMATO de la KV, para el espacio de nombres del tier de offload.

Por que
-------
El tier L2/L3 guarda bloques de KV en ``<raiz>/<modelo>_<hash>_r<rank>/...`` y ese hash
(``v1/kv_offload/file_mapper.py``) sale del modelo, el paralelismo, ``tokens_per_hash``, el
dtype DEL MODELO y los grupos. No entra nada de lo que decide que SIGNIFICAN los bytes:

* ``--kv-cache-dtype`` (int8 por token-cabeza, fp8, fp16);
* si k se guarda rotada (PN126 / PN131) y con que rotacion;
* el kernel que escribe (PN131 guarda con exponentes de referencia propios);
* PN122, que cambia que columnas de estado GDN existen;
* el dtype de la KV del borrador y la escala de su residual (PN144).

Cambiar cualquiera de esos deja el MISMO directorio, y como ``/kv-offload`` esta en disco los
bloques sobreviven a recrear el contenedor: el arranque siguiente los restaura y los lee con el
significado nuevo. Medido el 2026-09-21: un prompt de 37k repetido entre brazos de un A/B
devolvia ``"\\n"`` + EOS (2 tokens) en 1,8 s, y uno nuevo del mismo largo, en el mismo arranque,
400 tokens correctos en 16,7 s. Es la cara de contexto largo del "emite dos tokens y para".

Que hace
--------
``huella()`` devuelve una cadena corta y estable que resume todo lo de arriba. PN81 la mete en
``FileMapper.fields`` como ``genesis_kv_format``, asi que participa del hash del directorio: otra
config, otro directorio. Sobre-incluir solo cuesta reuso de cache; sub-incluir cuesta salidas
corruptas, asi que ante la duda una variable ENTRA.

``FORMATO`` se sube a mano cuando un kernel cambia el layout de lo que escribe sin que cambie
ninguna variable (por ejemplo, un cambio en ``sk18h_escribir2.cu``).

Limite conocido: las referencias ek/ev de PN131 se fijan con el primer write de cada arranque y
no se conocen aca. Son estables mientras el primer pedido sea el mismo healthcheck.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys

FORMATO = 1

# Variables de PN131 que NO cambian lo que se escribe (tamaños de buffers, diagnostico, camino
# de prefill, que lee decuantizando). Todas las demas GENESIS_PN131_* entran.
_PN131_SOLO_RUNTIME = {
    "GENESIS_PN131_BMAX", "GENESIS_PN131_DIAG", "GENESIS_PN131_NATIVO",
    "GENESIS_PN131_PREFILL", "GENESIS_PN131_MAXTOK", "GENESIS_PN131_CPG",
}
_SUELTAS = (
    "GENESIS_ENABLE_PN131_SK18",
    "GENESIS_ENABLE_PN126_ROT_QK", "GENESIS_PN126_ROT", "GENESIS_PN126_DFLASH",
    "GENESIS_PN126_DAMP",
    "GENESIS_ENABLE_PN122_GDN_CINTA",
    "GENESIS_ENABLE_PN144_DFLASH2_ESCALA", "GENESIS_PN144_ESCALA_RESIDUAL",
)


def _de_argv() -> dict:
    linea = " ".join(sys.argv)
    out = {}
    m = re.search(r"--kv-cache-dtype[= ]+(\S+)", linea)
    if m:
        out["kv_cache_dtype"] = m.group(1)
    m = re.search(r"--speculative-config[= ]+'?(\{.*?\})'?(?:\s|$)", linea)
    if m:
        try:
            spec = json.loads(m.group(1))
            out["spec_method"] = spec.get("method")
            out["spec_kv_cache_dtype"] = spec.get("kv_cache_dtype")
        except Exception:                                    # noqa: BLE001
            out["spec_crudo"] = m.group(1)[:200]
    return out


def componentes() -> dict:
    c: dict = {"formato": FORMATO}
    try:
        from vllm.config import get_current_vllm_config

        cfg = get_current_vllm_config()
        c["kv_cache_dtype"] = str(cfg.cache_config.cache_dtype)
        spec = cfg.speculative_config
        if spec is not None:
            c["spec_method"] = str(getattr(spec, "method", None))
            c["spec_kv_cache_dtype"] = str(getattr(spec, "kv_cache_dtype", None))
    except Exception:                                        # noqa: BLE001
        pass
    for k, v in _de_argv().items():      # la linea de comandos siempre esta; completa lo que falte
        c.setdefault(k, v)
    for k in _SUELTAS:
        if k in os.environ:
            c[k] = os.environ[k]
    for k in sorted(os.environ):
        if k.startswith("GENESIS_PN131_") and k not in _PN131_SOLO_RUNTIME:
            c[k] = os.environ[k]
    return c


def huella() -> str:
    c = componentes()
    crudo = json.dumps(c, sort_keys=True, separators=(",", ":"))
    return f"v{FORMATO}-" + hashlib.sha256(crudo.encode()).hexdigest()[:16]


__all__ = ["FORMATO", "componentes", "huella"]
