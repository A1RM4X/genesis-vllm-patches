# SPDX-License-Identifier: Apache-2.0
"""Estadistica de las activaciones que entran a cada lineal, para decidir si la escala puede ser
ESTATICA en vez de por token.

La pregunta
-----------
Hoy, antes de cada GEMM de Marlin, se cuantiza la activacion a int8 con una escala que se calcula
en el momento: ``s_token = amax(|x|) / 127`` por fila. Eso cuesta una pasada de reduccion sobre la
fila y 256 lanzamientos por paso (564 us en decode, 29 ms en prefill), y ademas obliga a que la
cuantizacion sea un kernel aparte: no se puede meter adentro del kernel que produce el dato, porque
el productor no conoce el maximo hasta terminar la fila.

Con una escala ESTATICA por capa eso desaparece: el epilogo del GEMM anterior (o el SiLU x gate)
podria escribir el int8 directamente. Pero se paga en calidad, y cuanto se paga depende de una sola
cosa: **que tan dispersa es la distribucion de los maximos por token**.

Si el maximo de un token es 10 veces mas chico que la escala estatica, ese token usa 1/10 del rango
del int8: pierde log2(10) = 3,3 bits y queda en ~4,7 bits efectivos. Si la distribucion es angosta,
no se pierde casi nada.

Que mide
--------
Para cada lineal, y sin tocar el camino de ejecucion, acumula un histograma de ``amax`` por token
(en log2, que es donde viven los bits) y la cuenta de tokens. Al terminar escribe un JSON con los
percentiles por capa.

Se prende con ``GENESIS_ACT_STATS=1`` y se vuelca con ``GENESIS_ACT_STATS_CADA`` pasos (por omision
al llegar a 2000 llamadas por sitio). Sale en ``GENESIS_ACT_STATS_OUT`` (por omision
``/tmp/act_stats_<rank>.json``).
"""

from __future__ import annotations

import json
import logging
import os

import torch

log = logging.getLogger("genesis.act_stats")

ACTIVO = os.environ.get("GENESIS_ACT_STATS", "0") == "1"
SALIDA = os.environ.get("GENESIS_ACT_STATS_OUT", "/tmp/act_stats")
# Cada llamada aporta miles de tokens al histograma, asi que con pocas por sitio ya alcanza. El
# disparo es por TOTAL de llamadas: hay 256 sitios (4 lineales x 64 capas) y pedir muchas por sitio
# no volcaba nunca.
# Con la clave por forma hay solo 4 cubos (qkv, o, gate_up, down), asi que el tope POR SITIO tiene
# que ser alto o no se junta nada: con 16 se cortaba en 64 llamadas y no volcaba nunca.
TOPE = int(os.environ.get("GENESIS_ACT_STATS_TOPE", "100000"))
TOTAL = int(os.environ.get("GENESIS_ACT_STATS_TOTAL", "2000"))   # llamadas entre volcados

# log2(amax) va de -20 a +12 en pasos de 1/4: 128 cajas alcanzan y sobran
LO, HI, PASO = -20.0, 12.0, 0.25
NCAJAS = int((HI - LO) / PASO)

_hist: dict[str, torch.Tensor] = {}
_maxabs: dict[str, torch.Tensor] = {}   # maximo exacto por sitio (EN GPU: leerlo sincroniza)
# Histograma de log2(amax / rms) por token: dice si el amax de un token lo hace UN canal que pincha
# (cresta alta -> la rotacion de Hadamard lo aplasta) o si la fila ya es plana (rotar no sirve).
_cresta: dict[str, torch.Tensor] = {}
# Muestras crudas de activacion, para poder probar metodos OFFLINE sin reiniciar el servidor.
MUESTRAS = os.environ.get("GENESIS_ACT_MUESTRAS", "")     # directorio donde dejarlas
FILAS = int(os.environ.get("GENESIS_ACT_FILAS", "512"))   # filas por muestra
_muestreado: set = set()
_llamadas: dict[str, int] = {}
_volcado = False
_tot = 0


def activo() -> bool:
    return ACTIVO


@torch.library.custom_op("genesis::act_stats", mutates_args=("x",))
def registrar(x: torch.Tensor, nombre: str) -> None:
    """Acumula la distribucion de amax por token de `x`. No modifica nada.

    Tiene que ser un CUSTOM OP, no una funcion suelta: este punto vive dentro de la region que
    vLLM compila con grafo completo, donde un corte de grafo es un ERROR, no un respaldo. Ni
    ``torch._dynamo.disable`` alcanza (se probo: el worker muere al arrancar). Como op propia,
    dynamo la trata como una caja negra y la deja pasar.

    Y tiene que declararse ``mutates_args=("x",)`` aunque NO mutemos nada: declarada como pura y
    sin salida, Inductor la borra como codigo muerto y la sonda queda muda. Decir que muta la
    entrada solo hace al compilador mas conservador, que para telemetria esta bien.

    `nombre` es el prefijo de la capa (p.ej. ``...layers.31.mlp.down_proj``), asi que la
    estadistica sale POR CAPA. Un str es un tipo valido en la firma de un custom op, y dynamo lo
    hornea como constante porque sale de un atributo del modulo.
    """
    if not ACTIVO:
        return
    # Durante la captura de grafos CUDA no se puede SINCRONIZAR con el host (revienta la captura).
    # El histograma si se puede acumular: vive en un tensor reservado durante el prefill, que no es
    # capturado, y escribirle desde adentro del grafo persiste entre reproducciones. Lo que NO se
    # guarda durante la captura es el maximo exacto, porque quedaria apuntando al pool privado de
    # la captura (la leccion de PN131).
    capturando = torch.cuda.is_current_stream_capturing()
    if capturando and nombre not in _hist:
        return                      # sin histograma previo habria que reservar: no durante captura
    try:
        n = _llamadas.get(nombre, 0)
        if n >= TOPE:
            return
        _llamadas[nombre] = n + 1
        plano = x.reshape(-1, x.shape[-1])
        # Las filas de relleno del prefill chunked traen memoria SIN INICIALIZAR, o sea NaN a
        # veces: sin limpiarlas el maximo queda envenenado (y el histograma, sesgado).
        amax = torch.nan_to_num(plano.abs().amax(dim=-1).float(), nan=0.0, posinf=0.0)
        amax = amax.clamp_min(1e-30)
        caja = ((amax.log2() - LO) / PASO).long().clamp_(0, NCAJAS - 1)
        h = _hist.get(nombre)
        if h is None:
            h = torch.zeros(NCAJAS, dtype=torch.long, device=x.device)
            _hist[nombre] = h
        h.scatter_add_(0, caja, torch.ones_like(caja))
        # cresta = amax / rms de la fila. Para una fila gaussiana de N=5120 da ~3,9 (log2 ~1,96);
        # mucho mas que eso significa que el maximo lo pone un canal aislado.
        rms = torch.nan_to_num(plano.float().pow(2).mean(-1), nan=1.0).clamp_min(1e-30).sqrt()
        cr = ((amax / rms).log2() * 8.0).long().clamp_(0, NCAJAS - 1)   # 1/8 de bit por caja
        hc = _cresta.get(nombre)
        if hc is None:
            if capturando:
                return
            hc = torch.zeros(NCAJAS, dtype=torch.long, device=x.device)
            _cresta[nombre] = hc
        hc.scatter_add_(0, cr, torch.ones_like(cr))
        if not capturando:
            m = amax.max()                  # se queda en GPU: leerlo aca sincronizaria
            prev = _maxabs.get(nombre)
            _maxabs[nombre] = m if prev is None else torch.maximum(prev, m)
        global _tot
        _tot += 1
        if capturando:
            return                  # el resto (volcado y muestreo) sincroniza
        # Ojo: la PRIMERA llamada de cada sitio es la corrida de perfilado de vLLM, con entradas
        # en CERO. Hay que esperar a que haya trafico de verdad, o la muestra sale vacia.
        if (MUESTRAS and nombre not in _muestreado and plano.shape[0] >= FILAS
                and _llamadas.get(nombre, 0) > 4 and float(plano.abs().amax()) > 0):
            _muestreado.add(nombre)
            try:
                paso = max(1, plano.shape[0] // FILAS)
                torch.save(plano[::paso][:FILAS].detach().cpu(),
                           f"{MUESTRAS}/{nombre.replace('.', '_')}.pt")
            except Exception as e:
                log.warning("act_stats: no se pudo guardar la muestra (%s)", e)
        if _tot == 1:
            log.warning("act_stats: primera llamada, forma %s", tuple(plano.shape))
        if _tot % TOTAL == 0:
            volcar()
    except Exception as e:                      # nunca romper el modelo por telemetria
        log.warning("act_stats: %s", e)


@registrar.register_fake
def _registrar_fake(x: torch.Tensor, nombre: str) -> None:
    return None


def _percentiles(h: torch.Tensor) -> dict:
    tot = int(h.sum())
    if tot == 0:
        return {}
    acum = torch.cumsum(h, 0)
    fuera = {"n": tot}
    for p in (0.1, 1, 10, 50, 90, 99, 99.9, 100):
        idx = int(torch.searchsorted(acum, max(1, int(tot * p / 100.0))))
        idx = min(idx, NCAJAS - 1)
        fuera[f"p{p}"] = round(LO + (idx + 0.5) * PASO, 3)     # en log2
    return fuera


def volcar() -> None:
    """Se puede llamar varias veces: reescribe el archivo con lo acumulado hasta el momento."""
    if not _hist:
        return
    # Con el PID, no por rank: en los workers de vLLM RANK no esta en el entorno, asi que los dos
    # escribian el MISMO archivo y se pisaban a mitad de escritura (JSON truncado).
    rank = f"{os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0'))}_{os.getpid()}"
    datos = {}
    for n, h in _hist.items():
        d = _percentiles(h.cpu())
        if d:
            hc = _cresta.get(n)
            if hc is not None:
                ac = torch.cumsum(hc.cpu(), 0)
                tt = int(ac[-1])
                if tt:
                    for pc in (50, 99):
                        idx = int(torch.searchsorted(ac, max(1, tt * pc // 100)))
                        d[f"cresta_p{pc}"] = round(min(idx, NCAJAS - 1) / 8.0, 3)
            t = _maxabs.get(n)
            d["amax"] = float(t) if t is not None else 0.0   # aca si se puede sincronizar
        datos[n] = d
    ruta = f"{SALIDA}_{rank}.json"
    with open(ruta, "w") as f:
        json.dump(datos, f, indent=1)
    log.warning("act_stats: %d sitios, %d llamadas -> %s", len(datos), _tot, ruta)
