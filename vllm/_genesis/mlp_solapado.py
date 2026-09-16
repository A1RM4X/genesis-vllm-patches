# SPDX-License-Identifier: Apache-2.0
"""PN136 — el all-reduce del MLP viaja MIENTRAS se calcula el resto del bloque.

CABLEADO, CORRIENDO Y EXACTO — PERO NO GANA NADA, asi que queda APAGADO por omision.

Medido en el servidor real (prefill de 42k tokens, 3 corridas x 3 repeticiones cada una):

    PN136 apagado      2.383 tok/s
    PN136, 4 trozos    2.390 tok/s   (+0,3%)
    PN136, 2 trozos    2.392 tok/s   (+0,4%)

O sea, ruido. El camino solapado corre de verdad (lo dice el log "MLP partido en N trozos") y sin
un solo error. Por que no gana: con PN120 los parciales ya viajan en int8 (la mitad de los bytes)
y, despues de emparejar las placas por potencia, lo que queda de intercambio es chico al lado del
computo del bloque MLP. NCCL en ese regimen ya no molesta.

Donde SI valdria: si el intercambio volviera a pesar — mas ranks, activaciones sin cuantizar,
o un enlace mas lento — porque el transporte esta medido en 13,0 GB/s con 105% de solape contra
los 11,0 GB/s y 25% de NCCL.

El problema
-----------
En prefill el all-reduce de TP es el 41,6% del tiempo de GPU y ya corre al techo del PCIe.
Comprimir mas no se puede: el int4 esta dominado (a igual trafico da 7x mas error que el int8 de
PN120). Lo unico que queda es que la comunicacion pase mientras el SM trabaja.

La forma
--------
El bloque MLP es **por token**: gate_up, SiLU x gate y down miran cada fila por separado. Asi que
se lo parte en trozos de filas y el trozo i sale a la red mientras se calcula el i+1::

    trozo 0: gate_up -> act -> down -> cuantiza -> EMPUJA al otro rank (stream lateral)
    trozo 1: gate_up -> act -> down -> cuantiza -> EMPUJA        <- esto corre mientras viaja el 0
    ...
    despues: para cada trozo, espera su bandera y suma los parciales

Con N trozos el unico intercambio que queda al descubierto es el ultimo. Partirlo es EXACTO
(verificado: diferencia 0 contra el bloque entero).

Por que no alcanzaba con NCCL
-----------------------------
NCCL mueve los datos con kernels, asi que compite por SM con el GEMM. Medido sobre estas dos 3090:
a nivel de bloque MLP el solape con NCCL es **0%** — el MLP llena los 82 SM y los bloques de NCCL
no consiguen lugar — y darle prioridad alta al stream no cambia nada. La copia P2P por DMA no usa
SM y solapa 98-101%. Ver ``vllm._genesis.p2p_buzon``.

Por que NO paga en esta maquina
-------------------------------
Medido con 2 procesos reales y sincronizacion de verdad (tests/proto/pn136_offline.py, exacto bit
a bit en todos los casos):

    M=8192, 4 trozos: serie 18.494 us -> solapado 18.301 us = 1,01x
    M=4096, 4 trozos: serie  9.440 us -> solapado 10.238 us = 0,92x
    (mlp solo 16.760 us; el intercambio cuesta 7.702 us por DMA y ~1.900 us por NCCL)

La causa es el ancho de banda del motor de copia (tests/proto/p2p_duplex.py, 40 MB):

    NCCL all_gather                      11,2 GB/s
    cudaMemcpyPeerAsync, una direccion    5,6 GB/s   <- el techo del motor DMA
    idem, los dos a la vez                5,7 GB/s

No es duplex ni falta de ``cudaDeviceEnablePeerAccess`` (probado): en placas GeForce el DMA P2P
esta capado, y NCCL llega al doble porque copia con los SM. El DMA si solapa —por eso se escribio
esto— pero 2x mas lento cancela exactamente lo que el solape gana.

Cuidado con como se mide: una version anterior del banco esperaba el evento de su PROPIA copia en
vez de la bandera del otro rank, o sea que no sincronizaba entre placas, y daba 1,27-1,34x. Era
una cota optimista, no un resultado.

Detalles que importan
---------------------
* ``down_proj.reduce_results`` se pone en False al cablear: la reduccion la hacemos nosotros.
* El gate por talla NO puede vivir en el forward trazado (dynamo hornea la rama con la forma del
  trazado, que es la trampa de PN134), asi que todo esto vive en un custom op OPACO que recibe el
  indice de capa y busca el modulo en un registro.
* Con M chico (decode) cae al camino de siempre: ahi el cuello es latencia, no ancho de banda, y
  partir solo agregaria lanzamientos.
* Durante la captura de grafos CUDA tampoco se usa: los buzones se reservan al cablear, nunca en
  caliente (reservar adentro de una captura sale del pool privado de la captura y da basura al
  reproducir — la leccion de PN131).

Se prende con ``GENESIS_ENABLE_PN136_MLP_SOLAPADO=1``.
"""

from __future__ import annotations

import logging
import math
import os

import torch

log = logging.getLogger("genesis.pn136")

_TRUTHY = ("1", "true", "yes", "on")


def _flag(n: str, d: str = "0") -> bool:
    return os.environ.get(n, d).strip().lower() in _TRUTHY


_ACTIVO = _flag("GENESIS_ENABLE_PN136_MLP_SOLAPADO")
_M_MIN = int(os.environ.get("GENESIS_PN136_M_MIN", "1024"))
_TROZOS = int(os.environ.get("GENESIS_PN136_TROZOS", "4"))
_GRUPO = int(os.environ.get("GENESIS_PN136_GRUPO", "64"))     # el mismo de PN120
_CAP = int(os.environ.get("GENESIS_PN136_CAP", "8192"))       # max-num-batched-tokens

_capas: list = []           # indice -> modulo MLP
_buzones = None
_comm = None
_avisado = False


def activo() -> bool:
    return _ACTIVO


def trozos() -> int:
    return _TROZOS


# ── numerica: la misma de PN120 (int8 con escala por grupo de 64) ────────────────────────────
def _cuantizar(x: torch.Tensor, g: int):
    m, h = x.shape
    v = x.view(m, h // g, g)
    s = v.abs().amax(dim=-1).clamp_min(1e-6) / 127.0
    q = (v / s.unsqueeze(-1)).round().clamp_(-127, 127).to(torch.int8).view(m, h)
    return q, s.to(torch.float16)


def _sumar(q_yo: torch.Tensor, s_yo: torch.Tensor, q_otros: torch.Tensor, s_otros: torch.Tensor,
           g: int, dt: torch.dtype) -> torch.Tensor:
    """Descuantiza el parcial propio y los de los pares, y los suma."""
    m, h = q_yo.shape
    ng = h // g
    out = q_yo.view(m, ng, g).to(dt) * s_yo.unsqueeze(-1).to(dt)
    for i in range(q_otros.shape[0]):
        out = out + q_otros[i].view(m, ng, g).to(dt) * s_otros[i].unsqueeze(-1).to(dt)
    return out.view(m, h)


_cuant_c = None
_sumar_c = None


def _compilados():
    """Se compilan una vez. ``dynamic=False`` a proposito, igual que en PN120: el planificador
    produce pocas tallas distintas de trozo y asi inductor fusiona de verdad."""
    global _cuant_c, _sumar_c
    if _cuant_c is None:
        _cuant_c = torch.compile(_cuantizar, dynamic=False)
        _sumar_c = torch.compile(_sumar, dynamic=False)
    return _cuant_c, _sumar_c


def _parcial(mlp, x: torch.Tensor) -> torch.Tensor:
    """El bloque MLP SIN reducir: gate_up -> SiLU x gate -> down. Es lo que sale de cada placa."""
    gate_up, _ = mlp.gate_up_proj(x)
    h = mlp.act_fn(gate_up)
    y, _ = mlp.down_proj(h)                 # con reduce_results=False devuelve el parcial
    return y


# ── el camino de siempre, para decode y para cuando algo no cierra ───────────────────────────
def _sin_solape(mlp, x: torch.Tensor) -> torch.Tensor:
    y = _parcial(mlp, x)
    from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
    try:
        from vllm._genesis import ar_int8 as _g120
        if _g120.activo():
            return _g120.all_reduce_int8(y)
    except Exception:
        pass
    return tensor_model_parallel_all_reduce(y)


def _con_solape(mlp, x: torch.Tensor) -> torch.Tensor:
    global _avisado
    m, _ = x.shape
    n = min(_TROZOS, _buzones.ranuras)
    paso = int(math.ceil(m / n))
    cuant, sumar = _compilados()
    principal = torch.cuda.current_stream()
    pend = []

    # 1) calcular cada trozo y mandarlo apenas esta listo
    for i in range(n):
        lo, hi = i * paso, min(m, (i + 1) * paso)
        if lo >= hi:
            break
        y = _parcial(mlp, x[lo:hi])
        q, s = cuant(y, _GRUPO)
        ev = torch.cuda.Event()
        ev.record(principal)
        with torch.cuda.stream(_comm):
            _comm.wait_event(ev)
            q.record_stream(_comm)
            s.record_stream(_comm)
            epoca = _buzones.empujar(q, s, i, _comm)
        pend.append((i, lo, hi, q, s, epoca))

    # 2) recien ahora esperar. El intercambio del trozo 0 viajo mientras se calculaban los demas.
    salida = torch.empty_like(x)
    for i, lo, hi, q, s, epoca in pend:
        _buzones.esperar(i, epoca, principal)
        dq, ds = _buzones.recibido(i, hi - lo)
        salida[lo:hi] = sumar(q, s, dq, ds, _GRUPO, x.dtype)

    if not _avisado:
        _avisado = True
        log.warning("PN136: MLP partido en %d trozos con intercambio P2P solapado (M>=%d)",
                    n, _M_MIN)
    return salida


@torch.library.custom_op("genesis::pn136_mlp", mutates_args=())
def _pn136_mlp(x: torch.Tensor, capa: int) -> torch.Tensor:
    """Op OPACA: el gate por talla tiene que evaluarse en cada forward, no al trazar."""
    mlp = _capas[capa]
    m = x.shape[0]
    puede = (_buzones is not None and m >= _M_MIN and x.dim() == 2
             and x.shape[1] == _buzones.ancho and m <= _buzones.cap * _buzones.ranuras
             and not torch.cuda.is_current_stream_capturing())
    if not puede:
        return _sin_solape(mlp, x)
    try:
        return _con_solape(mlp, x)
    except Exception as e:                      # nunca romper el modelo por esto
        log.error("PN136: fallo el camino solapado (%s); se sigue por el de siempre", e)
        return _sin_solape(mlp, x)


@_pn136_mlp.register_fake
def _pn136_mlp_fake(x: torch.Tensor, capa: int):
    return torch.empty_like(x)


def _capas_del_modelo(model):
    """Encuentra la lista de capas del decoder sin depender de como se llame el envoltorio.

    El modelo que entrega el cargador puede venir envuelto (torch.compile, LoRA, etc.), asi que en
    vez de asumir ``model.model.layers`` se baja por los hijos hasta encontrar un ``layers`` que
    tenga modulos con ``mlp``.
    """
    vistos = set()
    pila = [model]
    while pila:
        m = pila.pop(0)
        if id(m) in vistos:
            continue
        vistos.add(id(m))
        capas = getattr(m, "layers", None)
        if capas is not None and len(capas) and any(hasattr(c, "mlp") for c in capas):
            return list(capas)
        for _, hijo in m.named_children():
            pila.append(hijo)
    return []


def preparar(model, cap_filas: int | None = None) -> int:
    """Registra los MLP, reserva los buzones y devuelve cuantas capas quedaron cableadas.

    Se llama al terminar de cargar los pesos: los buzones tienen que existir ANTES de la captura
    de grafos y antes de que vLLM mida cuanta memoria queda para la KV.
    """
    global _buzones, _comm
    if not _ACTIVO or _buzones is not None:
        return 0
    from vllm.distributed.parallel_state import get_tp_group

    tp = get_tp_group()
    w = tp.world_size
    if w < 2:
        log.warning("PN136: TP=%d, no hay nada que solapar", w)
        return 0

    capas = _capas_del_modelo(model)
    if not capas:
        log.warning("PN136: no se encontraron capas en %s (hijos: %s)", type(model).__name__,
                    [n for n, _ in model.named_children()][:6])
    mlps = []
    for capa in capas:
        mlp = getattr(capa, "mlp", None)
        if mlp is None or not hasattr(mlp, "gate_up_proj") or not hasattr(mlp, "down_proj"):
            continue
        if getattr(mlp, "expert_gate", None) is not None:
            continue                                  # el shared_expert de MoE tiene otra salida
        dp = mlp.down_proj
        if not getattr(dp, "reduce_results", False):
            continue                                  # ya viene sin reducir: no es nuestro caso
        mlps.append(mlp)
    if not mlps:
        log.warning("PN136: no se encontro ningun MLP con down_proj que reduzca")
        return 0

    ancho = mlps[0].down_proj.output_size if hasattr(mlps[0].down_proj, "output_size") else None
    if ancho is None or ancho % _GRUPO != 0:
        log.warning("PN136: ancho %s no divisible por el grupo %d", ancho, _GRUPO)
        return 0

    cap = cap_filas or int(math.ceil(_CAP / _TROZOS))
    from vllm._genesis.p2p_buzon import Buzones

    # rank -> indice de placa: con una placa por rank es la posicion local; se confirma
    # intercambiando el indice real por el grupo de CPU.
    import torch.distributed as dist
    mio = torch.tensor([torch.cuda.current_device()], dtype=torch.int64)
    todos = [torch.zeros(1, dtype=torch.int64) for _ in range(w)]
    dist.all_gather(todos, mio, group=tp.cpu_group)
    devs = [int(t.item()) for t in todos]

    _buzones = Buzones(cap, ancho, _GRUPO, _TROZOS, tp.rank_in_group, w, tp.cpu_group, devs)
    _comm = torch.cuda.Stream(device=torch.cuda.current_device())

    for mlp in mlps:
        idx = len(_capas)
        _capas.append(mlp)
        mlp.down_proj.reduce_results = False      # la reduccion pasa a ser nuestra
        mlp.forward = (lambda _i: lambda x: torch.ops.genesis.pn136_mlp(x, _i))(idx)
    log.warning("PN136: %d MLP cableados (ancho %d, %d trozos de hasta %d filas)",
                len(mlps), ancho, _TROZOS, cap)
    return len(mlps)
