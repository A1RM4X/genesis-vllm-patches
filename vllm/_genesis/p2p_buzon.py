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

Una ``cudaMemcpyPeerAsync`` la ejecuta el MOTOR DE COPIA, que no usa SM ninguno, asi que solapa por
construccion. El problema es el ancho de banda (tests/proto/p2p_duplex.py, 40 MB):

    NCCL all_gather                      11,2 GB/s
    cudaMemcpyPeerAsync, una direccion    5,6 GB/s   <- el techo del motor DMA
    idem, los dos a la vez                5,7 GB/s

**En estas 3090 el motor de copia va a la mitad que NCCL**, y no es por duplex ni por falta de
``cudaDeviceEnablePeerAccess`` (probado, no cambia nada): en placas GeForce el DMA P2P esta capado.
NCCL llega al doble porque copia con los SM. Resultado: el solape devuelve exactamente lo que
pierde el transporte, y PN136 queda en 1,00x. Ver ``vllm._genesis.mlp_solapado``.

Estas primitivas quedan igual porque son correctas y reutilizables, y porque en placas con P2P sin
capar (Tesla/Quadro, o NVLink) la cuenta da distinto.

TRAMPA aparte, que costo media tarde: el ``copy_`` de torch toma un camino LENTO cuando el DESTINO
es un tensor importado por IPC — 5,9 GB/s; tirando (destino local) da 12,9, pero eso usa SM. Por eso
aca se llama ``cudaMemcpyPeerAsync`` directo. Medido en tests/proto/p2p_ancho.py.

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


def _compartir(t: torch.Tensor, grupo_cpu, rank: int, w: int):
    """Publica un tensor por IPC y devuelve la vista del tensor de cada rank."""
    import torch.distributed as dist
    import torch.multiprocessing.reductions as mpr

    datos = pickle.dumps(mpr.reduce_tensor(t))
    buf = torch.frombuffer(bytearray(datos), dtype=torch.uint8)
    n = torch.tensor([buf.numel()], dtype=torch.int64)
    ns = [torch.zeros(1, dtype=torch.int64) for _ in range(w)]
    dist.all_gather(ns, n, group=grupo_cpu)
    mx = int(max(int(x.item()) for x in ns))
    pad = torch.zeros(mx, dtype=torch.uint8)
    pad[: buf.numel()] = buf
    rec = [torch.zeros(mx, dtype=torch.uint8) for _ in range(w)]
    dist.all_gather(rec, pad, group=grupo_cpu)
    fuera = []
    for i in range(w):
        if i == rank:
            fuera.append(t)
        else:
            f, a = pickle.loads(bytes(rec[i][: int(ns[i].item())].numpy()))
            fuera.append(f(*a))
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
            dq = self.datos_r[par]
            ds = self.escalas_r[par]
            db = self.banderas_r[par]
            off_q = ((ranura * self.npares + i) * self.cap) * self.ancho
            off_s = ((ranura * self.npares + i) * self.cap) * (self.ancho // self.grupo)
            memcpy_peer(dq.data_ptr() + off_q, dev_par, q.data_ptr(), self.dev,
                        m * self.ancho, stream)
            memcpy_peer(ds.data_ptr() + off_s * 2, dev_par, s.data_ptr(), self.dev,
                        m * (self.ancho // self.grupo) * 2, stream)
            memcpy_peer(db.data_ptr() + 4 * (ranura * self.npares + i), dev_par,
                        self.marca.data_ptr() + 4 * ranura, self.dev, 4, stream)
        return e

    def esperar(self, ranura: int, epoca: int, stream) -> None:
        """Bloquea el stream hasta que llegaron los parciales de todos los pares."""
        for i in range(self.npares):
            esperar_valor(self.banderas.data_ptr() + 4 * (ranura * self.npares + i), epoca, stream)

    def recibido(self, ranura: int, m: int):
        """Vistas de lo que mandaron los pares: (datos [P, m, H], escalas [P, m, H/G])."""
        return (self.datos[ranura, :, :m], self.escalas[ranura, :, :m])
