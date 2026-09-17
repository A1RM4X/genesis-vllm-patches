# SPDX-License-Identifier: Apache-2.0
"""Escala de activacion ESTATICA por capa: primero la prueba de calidad, sin tocar el rendimiento.

Que se quiere
-------------
Hoy, antes de cada GEMM de Marlin, la activacion se cuantiza a int8 con una escala calculada en el
momento (``amax`` por fila). Eso obliga a una pasada de reduccion sobre la fila y a un kernel
aparte — 564 us de decode y 29 ms de prefill — porque el kernel que PRODUCE el dato no conoce el
maximo hasta terminar la fila. Con una escala fija por capa eso desaparece y la cuantizacion se
puede meter adentro del productor.

Lo que dice la medicion (``act_stats``, por capa, sobre un prefill real):

| tipo | capas | bits que pierde el token mediano |
|---|---|---|
| in_proj_qkvz (GDN) | 48 | 0,00 |
| qkv_proj (atencion) | 16 | 0,12 |
| gate_up_proj | 64 | 0,25 |
| o_proj | 16 | 1,88 |
| out_proj (GDN) | 48 | 2,00 |
| down_proj | 64 | 2,50 |

Los tres primeros reciben la salida de una RMSNorm, o sea ya normalizada: la distribucion de
maximos por token es angosta y la escala fija sale gratis. Los otros tres reciben la salida de la
atencion o del SiLU x gate, donde el rango entre tokens es enorme.

Como se prueba la calidad SIN tocar el camino rapido
----------------------------------------------------
No hace falta reescribir la cuantizacion para saber cuanto cuesta en calidad. Alcanza con pasar la
activacion por la rejilla de la escala estatica y devolverla:

    x' = clamp(round(x / s), -127, 127) * s

Despues sigue el camino de siempre, que vuelve a cuantizar x' con su escala dinamica — pero x' ya
vive en la rejilla gruesa, asi que ese segundo paso no pierde nada mas. El resultado numerico es
exactamente el de la escala estatica, y el codigo de produccion queda intacto. Es caro (una pasada
extra), pero esto es una prueba de calidad, no de velocidad.

Uso
---
1. Calibrar: correr con ``GENESIS_ACT_STATS=1`` sobre trafico representativo y convertir el JSON
   con ``tests/bench/medicion/calibrar_escalas.py``.
2. Probar: ``GENESIS_ESCALA_ESTATICA=1`` y ``GENESIS_ESCALA_ARCHIVO=/ruta/escalas.json``.
   Con ``GENESIS_ESCALA_SITIOS`` se elige donde aplicarla (por omision solo los tres sitios que
   la medicion dio como gratis).
"""

from __future__ import annotations

import json
import logging
import os

import torch

log = logging.getLogger("genesis.escala_estatica")

ACTIVO = os.environ.get("GENESIS_ESCALA_ESTATICA", "0") == "1"
ARCHIVO = os.environ.get("GENESIS_ESCALA_ARCHIVO", "/tmp/escalas.json")
# Por omision, solo donde la entrada viene de una RMSNorm (0,00-0,25 bits de perdida).
SITIOS = tuple(s for s in os.environ.get(
    "GENESIS_ESCALA_SITIOS", "qkv_proj,in_proj_qkvz,gate_up_proj").split(",") if s)

_escalas: dict[str, float] = {}
_avisado = False


def _cargar() -> None:
    """Se lee AL IMPORTAR, nunca en caliente: ``escala_de`` se llama desde adentro de la region que
    traza dynamo, y ahi un ``open()`` no se puede trazar (mata al worker al arrancar)."""
    if not ACTIVO:
        return
    try:
        with open(ARCHIVO) as f:
            crudo = json.load(f)
        for nombre, s in crudo.items():
            if any(nombre.endswith(t) for t in SITIOS):
                _escalas[nombre] = float(s)
        log.warning("escala_estatica: %d capas calibradas de %s (sitios: %s)",
                    len(_escalas), ARCHIVO, ",".join(SITIOS))
    except Exception as e:
        log.error("escala_estatica: no se pudo leer %s (%s); queda inerte", ARCHIVO, e)


_cargar()


def activo() -> bool:
    return ACTIVO and bool(_escalas)


def escala_de(nombre: str) -> float:
    """Escala de esta capa, o 0 si no corresponde tocarla. Es una busqueda pura en un diccionario
    ya cargado: se evalua al TRAZAR y queda horneada como constante."""
    return _escalas.get(nombre, 0.0)


@torch.library.custom_op("genesis::escala_estatica", mutates_args=("x",))
def aplicar(x: torch.Tensor, escala: float) -> None:
    """Deja `x` sobre la rejilla de la escala estatica, en el lugar.

    Es un custom op por lo mismo que la sonda: esto vive adentro de la region que vLLM compila con
    grafo completo. Y ``mutates_args`` no es un adorno: sin eso Inductor lo borra como codigo
    muerto.
    """
    global _avisado
    if escala <= 0.0:
        return
    x.div_(escala).round_().clamp_(-127.0, 127.0).mul_(escala)
    if not _avisado:
        _avisado = True
        log.warning("escala_estatica: aplicando la rejilla estatica (prueba de calidad)")


@aplicar.register_fake
def _aplicar_fake(x: torch.Tensor, escala: float) -> None:
    return None
