# SPDX-License-Identifier: Apache-2.0
"""Cargar y lanzar un kernel desde su PTX/cubin, sin pasar por el JIT de Triton.

Para que sirve
--------------
Permite el ciclo: escribir el kernel en Triton -> compilarlo -> sacar el PTX ->
**editarlo a mano** -> cargar el PTX editado y lanzarlo. Es la unica forma de
meter optimizaciones que ni Triton ni ``tl.inline_asm_elementwise`` expresan,
porque este ultimo es **elementwise**: no alcanza los ``mma.sync``, los
``ldmatrix``, el ``cp.async`` ni la estructura del bucle.

Toda la cadena la expone el propio Triton, no hace falta nada externo::

    h = kernel[grid](...)          # compila
    h.asm.keys()                   # source, ttir, ttgir, llir, ptx, cubin
    h.asm["ptx"]                   # el asm, ~1150 lineas para un GEMM
    h.asm["cubin"]                 # el binario, ~46 KB

Ojo con el alcance
------------------
Triton compila una variante distinta por cada combinacion de ``tl.constexpr``.
Con la tabla actual son **7 buckets de M x 8 kernels x 2 de HAS_SHIFT = 112
variantes** de 900-3700 lineas cada una. Editar todas a mano no es viable; esto
esta pensado para uno o dos kernels calientes.

Y el tile se elige en RUNTIME segun M (es lo que dio 1.5x al retunear), asi que
si se reemplaza un kernel por PTX hay que cargar una variante por bucket y
despachar a mano.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import torch
import triton
from triton.runtime import driver


def _ruta_ptxas() -> str:
    """El ``ptxas`` que trae Triton, no el del sistema.

    Importa cual: Triton emite PTX con una version de ISA concreta y el ptxas
    del CUDA instalado puede ser mas viejo o mas nuevo. El que viene con Triton
    es por definicion el que acepta lo que Triton genera.
    """
    base = os.path.dirname(triton.__file__)
    cand = os.path.join(base, "backends", "nvidia", "bin", "ptxas")
    return cand if os.path.exists(cand) else "ptxas"


def compilar_ptx(ptx: str, arch: int | None = None, opt: int = 3) -> bytes:
    """Ensambla PTX (posiblemente editado a mano) y devuelve el cubin.

    Este es el eslabon que faltaba para cerrar el ciclo: ``guardar_ptx`` saca el
    asm, se lo edita, y esto lo vuelve a binario para que ``desde_ptx`` lo
    cargue. Sin esto ``KernelPTX`` solo podia relanzar el cubin que ya habia
    producido el JIT, o sea que no servia para editar nada.

    :param arch: capability sin punto (86 = sm_86 / RTX 3090). Si es ``None`` se
        toma de la GPU activa.
    :param opt: nivel ``-O`` de ptxas. Con ``0`` el asm sale tal cual se
        escribio, util para ver si una edicion a mano sobrevive; con ``3``
        ptxas reordena y puede deshacerla.
    """
    if arch is None:
        may, men = torch.cuda.get_device_capability()
        arch = may * 10 + men
    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, "k.ptx")
        fc = os.path.join(d, "k.cubin")
        with open(fp, "w") as f:
            f.write(ptx)
        r = subprocess.run(
            [_ruta_ptxas(), f"-arch=sm_{arch}", f"-O{opt}", fp, "-o", fc],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"ptxas fallo:\n{r.stderr}")
        with open(fc, "rb") as f:
            return f.read()


class KernelPTX:
    """Un kernel cargado desde cubin, lanzable a mano.

    :param nombre: nombre de la funcion dentro del modulo (``h.metadata.name``).
    :param cubin: binario compilado (``h.asm["cubin"]``).
    :param shared: bytes de shared memory que pide (``h.metadata.shared``).
    :param handle: el ``CompiledKernel`` original, del que se toman el launcher
        y la metadata empaquetada.
    """

    def __init__(self, nombre: str, cubin: bytes, shared: int, handle, device: int = 0):
        r = driver.active.utils.load_binary(nombre, cubin, shared, device)
        # (module, function, n_regs, n_spills, n_max_threads) segun version
        self.module, self.function = r[0], r[1]
        self.n_regs, self.n_spills = r[2], r[3]
        self.shared = shared
        self._run = handle.run
        # La metadata que espera el launcher viene empaquetada; el nombre del
        # atributo cambio entre versiones de Triton.
        self._meta = getattr(handle, "packed_metadata", None) or handle.metadata

    def __call__(self, grid, *args, stream=None):
        """Lanza el kernel. ``grid`` puede ser int o tupla de hasta 3."""
        if isinstance(grid, int):
            gx, gy, gz = grid, 1, 1
        else:
            gx, gy, gz = (list(grid) + [1, 1])[:3]
        if stream is None:
            stream = torch.cuda.current_stream().cuda_stream
        self._run(gx, gy, gz, stream, self.function, self._meta,
                  None, None, None, *args)


def desde_handle(handle, device: int = 0) -> KernelPTX:
    """Construye un ``KernelPTX`` a partir de un kernel ya compilado por Triton."""
    return KernelPTX(handle.metadata.name, handle.asm["cubin"],
                     handle.metadata.shared, handle, device)


def desde_ptx(ptx: str, handle, device: int = 0, arch: int | None = None,
              opt: int = 3) -> KernelPTX:
    """Ensambla un PTX editado y lo devuelve lanzable.

    ``handle`` es el kernel original compilado por Triton: de ahi salen el
    nombre de la funcion, los bytes de shared que pide y la plomeria del
    launcher (``run`` y la metadata empaquetada). O sea que la firma y el grid
    tienen que seguir siendo los mismos; esto reemplaza el CUERPO, no el
    contrato.

    :param ptx: el texto del PTX. Si es una ruta a un archivo existente, se lee.
    """
    if "\n" not in ptx and os.path.exists(ptx):
        with open(ptx) as f:
            ptx = f.read()
    cubin = compilar_ptx(ptx, arch=arch, opt=opt)
    return KernelPTX(handle.metadata.name, cubin, handle.metadata.shared,
                     handle, device)


def guardar_ptx(handle, ruta: str) -> int:
    """Vuelca el PTX a un archivo para editarlo. Devuelve la cantidad de lineas."""
    ptx = handle.asm["ptx"]
    with open(ruta, "w") as f:
        f.write(ptx)
    return len(ptx.splitlines())


__all__ = ["KernelPTX", "compilar_ptx", "desde_handle", "desde_ptx",
           "guardar_ptx"]
