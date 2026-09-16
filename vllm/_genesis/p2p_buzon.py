# SPDX-License-Identifier: Apache-2.0
"""Intercambio entre placas por copia P2P (motor DMA), para poder SOLAPARLO con el computo.

Por que no NCCL
---------------
NCCL mueve los datos con KERNELS (protocolo LL: los SM copian a mano, que para mensajes chicos es
la menor latencia). Eso lo pone a competir por SM con el GEMM, y medido sobre estas dos 3090:

  * concurrencia pura, sin ninguna dependencia: NCCL solapa 23% con un GEMM; sube a 42-58%
    bajando ``NCCL_MAX_NCHANNELS`` a 2;
  * a nivel de bloque MLP (que llena los 82 SM) el solape se va a **0%**: los bloques de NCCL no
    consiguen lugar. Darle prioridad alta al stream tampoco cambia nada.

Una ``cudaMemcpyPeerAsync`` la ejecuta el MOTOR DE COPIA, que no usa SM ninguno, y sobre memoria
compartida como corresponde da **13,0 GB/s** (mas que NCCL) con **105% de solape**: o sea que se
esconde entera detras del computo y sale gratis. Ese es el transporte que usa este modulo.

DE DONDE SALE LA MEMORIA: ES LO QUE DECIDE TODO
-----------------------------------------------
Medido sobre 40 MB (tests/proto/p2p_origen_memoria.py), compartiendo el MISMO tensor de tres
formas distintas:

| forma de compartir                        | memcpyPeer | ¿un kernel puede leerlo? |
|-------------------------------------------|------------|--------------------------|
| ``multiprocessing.reductions.reduce_tensor`` | 4,9 GB/s | **NO, acceso ilegal**    |
| ``cudaIpcGetMemHandle`` a mano sobre el tensor | **13,0 GB/s** | si                  |
| ``cudaMalloc`` crudo + IPC a mano          | 13,0 GB/s  | si                       |

El camino de torch es 2,6x mas lento Y deja la memoria fuera del alcance de los kernels. Por eso
aca el handle de IPC se saca y se abre a mano. Un tensor normal de torch sirve perfecto: lo unico
que hay que hacer es el IPC uno mismo.

Como se calcula el desplazamiento: ``cudaIpcGetMemHandle`` da un handle de la RESERVA ENTERA que
contiene al puntero, y al abrirlo del otro lado se recibe la base de esa reserva, no el puntero.
Con ``cuMemGetAddressRange`` cada rank averigua su propia base, manda el desplazamiento junto con
el handle, y el que recibe se lo suma. Sin eso se escribe en el lugar equivocado.

Como se sincroniza sin gastar SM
--------------------------------
El que recibe tiene que saber que los datos ya llegaron. Un barrier serializa (mata el solape) y un
kernel que gira esperando gasta justo lo que queremos ahorrar. La salida es
``cuStreamWaitValue32``: el stream espera a que una posicion de memoria tome un valor y lo resuelve
el planificador de la placa, sin ejecutar nada. Medido: cuesta 2,8%.

Ojo, **no se puede esperar sobre memoria importada por IPC** (da CUDA_ERROR_INVALID_VALUE). Por eso
el modelo es EMPUJAR, igual que el all-reduce propio de vLLM: cada rank escribe sus datos en el
buzon del otro y despues su bandera, las dos cosas con ``cudaMemcpyPeerAsync`` en el mismo stream
(o sea ordenadas), y despues espera sobre su bandera LOCAL.
"""

from __future__ import annotations

import ctypes
import logging
import pickle

import torch

log = logging.getLogger("genesis.p2p")

CU_EQ, CU_GEQ = 0, 1
_rt = None
_drv = None


def _libs():
    global _rt, _drv
    if _rt is None:
        _rt = ctypes.CDLL("libcudart.so")
        _drv = ctypes.CDLL("libcuda.so")
    return _rt, _drv


def memcpy_peer(dst_ptr: int, dst_dev: int, src_ptr: int, src_dev: int, n: int, stream) -> None:
    rt, _ = _libs()
    r = rt.cudaMemcpyPeerAsync(ctypes.c_void_p(dst_ptr), ctypes.c_int(dst_dev),
                               ctypes.c_void_p(src_ptr), ctypes.c_int(src_dev),
                               ctypes.c_size_t(n), ctypes.c_void_p(stream.cuda_stream))
    if r != 0:
        raise RuntimeError(f"cudaMemcpyPeerAsync -> {r}")


def esperar_valor(ptr: int, valor: int, stream, flags: int = CU_GEQ) -> None:
    """El stream espera a que *ptr (local) llegue a `valor`. No gasta SM."""
    _, drv = _libs()
    r = drv.cuStreamWaitValue32_v2(ctypes.c_void_p(stream.cuda_stream), ctypes.c_ulonglong(ptr),
                                   ctypes.c_uint(valor), ctypes.c_uint(flags))
    if r != 0:
        raise RuntimeError(f"cuStreamWaitValue32 -> {r}")


def escribir_valor(ptr: int, valor: int, stream) -> None:
    _, drv = _libs()
    r = drv.cuStreamWriteValue32_v2(ctypes.c_void_p(stream.cuda_stream), ctypes.c_ulonglong(ptr),
                                    ctypes.c_uint(valor), ctypes.c_uint(0))
    if r != 0:
        raise RuntimeError(f"cuStreamWriteValue32 -> {r}")


class IpcHandle(ctypes.Structure):
    """``cudaIpcMemHandle_t``. TIENE que ser un Structure: ``cudaIpcOpenMemHandle`` lo recibe POR
    VALOR, y ctypes pasa los arrays por referencia (eso da cudaErrorInvalidValue)."""

    _fields_ = [("reserved", ctypes.c_char * 64)]


def _preparar_firmas():
    rt, _ = _libs()
    rt.cudaIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), IpcHandle, ctypes.c_uint]
    rt.cudaIpcGetMemHandle.argtypes = [ctypes.POINTER(IpcHandle), ctypes.c_void_p]


def habilitar_p2p(pares: list[int]) -> None:
    """``cudaDeviceEnablePeerAccess`` hacia cada par. 704 = ya estaba, no es error."""
    rt, _ = _libs()
    for d in pares:
        r = rt.cudaDeviceEnablePeerAccess(ctypes.c_int(d), ctypes.c_uint(0))
        if r not in (0, 704):
            log.warning("cudaDeviceEnablePeerAccess(%d) -> %d", d, r)


def _base_y_desplazamiento(ptr: int) -> tuple[int, int]:
    """Base de la reserva que contiene a `ptr`, y cuanto hay que correrse desde ahi."""
    _, drv = _libs()
    base = ctypes.c_ulonglong()
    tam = ctypes.c_size_t()
    r = drv.cuMemGetAddressRange_v2(ctypes.byref(base), ctypes.byref(tam),
                                    ctypes.c_ulonglong(ptr))
    if r != 0:
        raise RuntimeError(f"cuMemGetAddressRange -> {r}")
    return base.value, ptr - base.value


def _compartir(t: torch.Tensor, grupo_cpu, rank: int, w: int) -> list[int]:
    """Comparte `t` por IPC y devuelve el PUNTERO equivalente de cada rank (el propio incluido).

    Se hace a mano a proposito: ver el encabezado del modulo. Con ``reduce_tensor`` de torch esto
    anda a 4,9 GB/s y los kernels no pueden tocar la memoria.
    """
    import torch.distributed as dist

    _preparar_firmas()
    rt, _ = _libs()
    base, desp = _base_y_desplazamiento(t.data_ptr())

    h = IpcHandle()
    r = rt.cudaIpcGetMemHandle(ctypes.byref(h), ctypes.c_void_p(base))
    if r != 0:
        raise RuntimeError(f"cudaIpcGetMemHandle -> {r}")
    # el campo c_char*64 leido como atributo se TRUNCA en el primer NUL: hay que sacar los bytes
    crudo = ctypes.string_at(ctypes.byref(h), 64)

    mio = torch.zeros(72, dtype=torch.uint8)
    mio[:64] = torch.frombuffer(bytearray(crudo), dtype=torch.uint8)
    mio[64:72] = torch.frombuffer(desp.to_bytes(8, "little"), dtype=torch.uint8).clone()
    todos = [torch.zeros(72, dtype=torch.uint8) for _ in range(w)]
    dist.all_gather(todos, mio, group=grupo_cpu)

    fuera = []
    for i in range(w):
        if i == rank:
            fuera.append(t.data_ptr())
            continue
        ho = IpcHandle()
        ctypes.memmove(ctypes.byref(ho), bytes(todos[i][:64].numpy()), 64)
        d_otro = int.from_bytes(bytes(todos[i][64:72].numpy()), "little")
        po = ctypes.c_void_p()
        r = rt.cudaIpcOpenMemHandle(ctypes.byref(po), ho, ctypes.c_uint(1))
        if r != 0:
            raise RuntimeError(f"cudaIpcOpenMemHandle(rank {i}) -> {r}")
        fuera.append(po.value + d_otro)
    return fuera


class Buzones:
    """Buzones persistentes para intercambiar parciales int8 entre los ranks de TP.

    El buzon de cada rank guarda lo que le escriben LOS OTROS (el parcial propio ya lo tiene en la
    mano), asi que ocupa ``ranuras x (W-1) x cap x H`` bytes. Con W=2, M=8192 y H=5120 son 42 MB.
    """

    def __init__(self, cap_filas: int, ancho: int, grupo_escala: int, ranuras: int,
                 rank: int, w: int, grupo_cpu, dispositivos: list[int]):
        self.rank, self.w, self.ranuras = rank, w, ranuras
        self.ancho, self.grupo = ancho, grupo_escala
        self.pares = [r for r in range(w) if r != rank]
        self.npares = len(self.pares)
        self.dev_de_rank = dispositivos          # rank -> indice de placa
        self.dev = dispositivos[rank]
        self.cap = cap_filas
        d = f"cuda:{torch.cuda.current_device()}"
        ng = ancho // grupo_escala

        # lo que me escriben los demas
        self.datos = torch.zeros((ranuras, self.npares, cap_filas, ancho), dtype=torch.int8, device=d)
        self.escalas = torch.zeros((ranuras, self.npares, cap_filas, ng), dtype=torch.float16, device=d)
        self.banderas = torch.zeros((ranuras, self.npares), dtype=torch.int32, device=d)
        # un escalar por ranura donde se arma el valor de la bandera antes de empujarlo
        self.marca = torch.zeros((ranuras,), dtype=torch.int32, device=d)

        habilitar_p2p([dispositivos[r] for r in self.pares])
        # punteros equivalentes en cada rank (no tensores: solo hacen falta para memcpyPeer)
        self.datos_r = _compartir(self.datos, grupo_cpu, rank, w)
        self.escalas_r = _compartir(self.escalas, grupo_cpu, rank, w)
        self.banderas_r = _compartir(self.banderas, grupo_cpu, rank, w)
        self.epoca = 0
        mb = (self.datos.numel() + self.escalas.numel() * 2) / 1e6
        log.warning("PN136: buzones P2P listos (%d ranuras x %d filas x %d, %.0f MB por placa)",
                    ranuras, cap_filas, ancho, mb)

    def _indice(self, destino: int, origen: int) -> int:
        """En que ranura del buzon de `destino` escribe `origen`."""
        return origen if origen < destino else origen - 1

    def empujar(self, q: torch.Tensor, s: torch.Tensor, ranura: int, stream) -> int:
        """Manda (q, s) a todos los pares y devuelve la epoca con la que hay que esperar.

        El orden importa: primero los datos, despues la bandera, en el MISMO stream. Asi el que
        recibe no ve la bandera hasta que los datos ya llegaron.
        """
        self.epoca += 1
        e = self.epoca
        m = q.shape[0]
        escribir_valor(self.marca.data_ptr() + 4 * ranura, e, stream)
        for par in self.pares:
            i = self._indice(par, self.rank)
            dev_par = self.dev_de_rank[par]
            dq, ds, db = self.datos_r[par], self.escalas_r[par], self.banderas_r[par]
            off_q = ((ranura * self.npares + i) * self.cap) * self.ancho
            off_s = ((ranura * self.npares + i) * self.cap) * (self.ancho // self.grupo)
            memcpy_peer(dq + off_q, dev_par, q.data_ptr(), self.dev, m * self.ancho, stream)
            memcpy_peer(ds + off_s * 2, dev_par, s.data_ptr(), self.dev,
                        m * (self.ancho // self.grupo) * 2, stream)
            memcpy_peer(db + 4 * (ranura * self.npares + i), dev_par,
                        self.marca.data_ptr() + 4 * ranura, self.dev, 4, stream)
        return e

    def esperar(self, ranura: int, epoca: int, stream) -> None:
        """Bloquea el stream hasta que llegaron los parciales de todos los pares."""
        for i in range(self.npares):
            esperar_valor(self.banderas.data_ptr() + 4 * (ranura * self.npares + i), epoca, stream)

    def recibido(self, ranura: int, m: int):
        """Vistas de lo que mandaron los pares: (datos [P, m, H], escalas [P, m, H/G])."""
        return (self.datos[ranura, :, :m], self.escalas[ranura, :, :m])
