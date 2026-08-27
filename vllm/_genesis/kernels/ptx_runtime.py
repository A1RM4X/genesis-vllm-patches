# SPDX-License-Identifier: Apache-2.0
"""Runtime de lanzamiento PTX nativo — libcuda cruda, sin Triton.

Que es
------
El unico pieza de plomeria compartida por los kernels cuyo cuerpo es PTX
embebido (los ``skXX.py`` tienen el asm inline donde antes estaba el kernel
Triton). El camino de ejecucion es::

    PTX (string en el .py del kernel) --ptxas--> cubin
      --cuModuleLoadData--> cuModuleGetFunction --> cuLaunchKernel

sin JIT de Triton, sin launcher de Triton, sin CompiledKernel. El cubin se
ensambla una vez por proceso (lazy, thread-safe) y se cachea.

ABI del cubin
-------------
Los params se parsean del propio ``.entry`` del asm. El ABI de Triton 3.7
es ``[params runtime en orden de declaracion] + [global_scratch ptr] +
[profile_scratch ptr]``: los ``tl.constexpr`` y los ints auto-especializados
(p.ej. strides == 1) estan horneados en el cubin y NO son params. Los
punteros de scratch se pasan en NULL (los kernels cableados tienen scratch
size 0 y nunca los desreferencian).

Seguridad de especializacion
----------------------------
``KernelNativo`` recibe ``horneado``: los valores de los params NO
consumidos por el ABI (constexpr declarados + auto-especializados) tal como
estaban cuando se genero el asm. En cada lanzamiento se verifican contra los
args recibidos y **se niega a lanzar** si alguno difiere: un desajuste daria
resultados incorrectos en silencio (un stride horneado como 1 con un valor
real distinto), asi que el fallo es explicito, nunca silencioso.

Garantia de los PTX cableados
-----------------------------
Cada PTX embebido en los kernels paso el gate **bit-exacto**
(``torch.equal`` sobre los bits de todas las salidas) contra el lanzamiento
JIT del kernel Triton original (que queda en el archivo solo como referencia
para los tests), en 3 semillas con entradas perturbadas. Suite:
``/tmp/opencode/tests_ptx/``.

Ediciones bit-exactas que contienen los asm
-------------------------------------------
* **E1**: ``setp.ge.f32`` + ``selp.f32`` (copysign de 0.5) -> un
  ``lop3.b32`` LUT ``0xF8`` (``a | (b & c)``). 2 instrucciones -> 1 por
  elemento, y libera los registros predicado. La convencion de bits del
  inmediato es indice ``(A<<2)|(B<<1)|C`` sobre los operandos en orden
  (el ``0xEA`` del ejemplo del manual usa otro orden: verificado
  empiricamente en sm_86).
* **E2**: ``div.full.f32`` por constante potencia-de-2 -> ``mul.f32`` por
  el reciproco exacto (``x/2^k == x*2^-k`` en IEEE, incluidos subnormales).
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import tempfile
import threading

import torch

_ARCH = 86
_OPT = 3
_lock = threading.Lock()
_lib = None


def habilitado() -> bool:
    """Kill-switch: ``GENESIS_PTQ_NATIVO=0`` desactiva el camino PTX.

    Los launchers de los kernels consultan esto antes de lanzar nativo;
    deshabilitado, caen al kernel Triton (que queda en el archivo como
    referencia de tests). Default: habilitado.
    """
    return os.environ.get("GENESIS_PTQ_NATIVO", "1").strip().lower() \
        not in ("0", "false", "no", "off")


def _libcuda():
    """libcuda cruda. La inicializacion del contexto es responsabilidad del
    proceso (torch la hace al primer op en GPU)."""
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL("libcuda.so.1")
    return _lib


def _compilar_ptx(ptx: str, arch: int = _ARCH, opt: int = _OPT) -> bytes:
    """Ensambla el PTX con el ptxas que trae Triton y devuelve el cubin.

    :param ptx: el texto del asm (posiblemente editado a mano).
    :param arch: capability sin punto (86 = sm_86 / RTX 3090).
    :param opt: nivel ``-O`` de ptxas.
    :raises RuntimeError: si ``ptxas`` falla (incluye stderr completo).
    """
    base = os.path.dirname(__import__("triton").__file__)
    ptxas = os.path.join(base, "backends", "nvidia", "bin", "ptxas")
    if not os.path.exists(ptxas):
        ptxas = "ptxas"
    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, "k.ptx")
        fc = os.path.join(d, "k.cubin")
        with open(fp, "w") as f:
            f.write(ptx)
        r = subprocess.run([ptxas, f"-arch=sm_{arch}", f"-O{opt}", fp,
                            "-o", fc], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ptxas fallo:\n{r.stderr}")
        with open(fc, "rb") as f:
            return f.read()


_MAPA_CTYPE = {"u64": ctypes.c_uint64, "i64": ctypes.c_int64,
               "u32": ctypes.c_uint32, "i32": ctypes.c_int32,
               "f32": ctypes.c_float, "f64": ctypes.c_double,
               "b64": ctypes.c_uint64, "b32": ctypes.c_uint32}


def _params_de_entry(ptx: str) -> tuple[str, list[tuple[str, str]]]:
    """Parsea el ``.entry`` del asm.

    :return: ``(nombre_entry, [(tipo, nombre_param), ...])``. Los params
        vienen con decoraciones (``.param .u64 .ptr .global .align 1 x``);
        el tipo es el primer token (``u64``) y el nombre el ultimo.
    """
    ini = ptx.index(".visible .entry")
    par = ptx.index("(", ini)
    fin = ptx.index(")", par)
    entry = ptx[ini:par].split()[-1]
    params = []
    for linea in ptx[par + 1:fin].splitlines():
        toks = linea.replace(",", " ").split()
        if not toks or toks[0] != ".param":
            continue
        params.append((toks[1].lstrip("."), toks[-1]))
    return entry, params


class KernelNativo:
    """Un kernel PTX embebido: ensambla bajo demanda y lanza por libcuda.

    :param caso: id de diagnostico (``archivo/kernel``).
    :param ptx: el asm completo, embebido en el .py del kernel.
    :param num_warps: warps por bloque (blockDim.x = num_warps * 32).
    :param shared: bytes de shared memory estatica que pide el cubin.
    :param idx: posicion de cada param runtime en la lista completa de args
        en orden de declaracion (los constexpr incluidos; el kernel solo
        consume lo que su ABI declara).
    :param n_runtime: cantidad de params runtime (los restantes del ABI son
        los 2 punteros de scratch de Triton, que van en NULL).
    :param horneado: mapa ``posicion -> valor`` de los params horneados en el
        cubin (constexpr + auto-especializados); se verifican en cada
        lanzamiento y cualquier desajuste levanta ``ValueError``.
    """

    def __init__(self, caso: str, ptx: str, num_warps: int, shared: int,
                 idx: list[int], n_runtime: int,
                 horneado: dict | None = None):
        self.caso = caso
        self.ptx = ptx
        self.num_warps = num_warps
        self.shared = shared
        self.idx = idx
        self.n_runtime = n_runtime
        self.horneado = horneado or {}
        self._mod = None
        self._fn = None
        self.entry, self._params = _params_de_entry(ptx)

    def _cargar(self):
        """Ensambla y carga el modulo CUDA una sola vez (thread-safe)."""
        if self._fn is not None:
            return
        with _lock:
            if self._fn is not None:
                return
            cubin = _compilar_ptx(self.ptx)
            img = (ctypes.c_char * len(cubin)).from_buffer_copy(cubin)
            mod = ctypes.c_void_p()
            res = _libcuda().cuModuleLoadData(ctypes.byref(mod), img)
            if res != 0:
                raise RuntimeError(f"cuModuleLoadData({self.caso}): {res}")
            fn = ctypes.c_void_p()
            res = _libcuda().cuModuleGetFunction(ctypes.byref(fn), mod,
                                                 self.entry.encode())
            if res != 0:
                raise RuntimeError(
                    f"cuModuleGetFunction({self.caso}, {self.entry}): {res}")
            self._mod, self._fn = mod, fn

    def __call__(self, grid, *args):
        """Lanza el kernel.

        :param grid: tupla de hasta 3 dimensiones.
        :param args: lista completa de valores en orden de declaracion del
            kernel original (tensores, escalares y constexpr inline). Se
            consumen las posiciones de ``idx``; ``horneado`` se verifica y
            los scratch van en NULL.
        :raises ValueError: si faltan args o si un valor horneado no
            coincide (el cubin seria incorrecto para esos inputs).
        """
        self._cargar()
        if len(args) < max(self.idx, default=-1) + 1:
            raise ValueError(f"{self.caso}: esperaba >= "
                             f"{max(self.idx) + 1} args, llegaron {len(args)}")
        for pos, val in self.horneado.items():
            if args[pos] != val:
                raise ValueError(
                    f"{self.caso}: el cubin esta horneado con "
                    f"param[{pos}]={val!r} y llego {args[pos]!r}; "
                    "el asm embebido no aplica a estos inputs")
        vals = []
        for k, (tipo, _nombre) in enumerate(self._params):
            ct = _MAPA_CTYPE[tipo]
            if k < self.n_runtime:
                v = args[self.idx[k]]
                if isinstance(v, torch.Tensor):
                    if ct is not ctypes.c_uint64:
                        raise TypeError(f"{self.caso}: tensor en param "
                                        f"no-puntero ({_nombre})")
                    vals.append(ct(v.data_ptr()))
                elif isinstance(v, float):
                    vals.append(ct(v))
                else:
                    vals.append(ct(int(v)))
            else:
                vals.append(ct(0))  # scratch: NULL
        arr = (ctypes.c_void_p * len(vals))(
            *[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in vals])
        gx, gy, gz = (list(grid) + [1, 1])[:3]
        stream = torch.cuda.current_stream().cuda_stream
        res = _libcuda().cuLaunchKernel(
            self._fn, gx, gy, gz, self.num_warps * 32, 1, 1,
            ctypes.c_uint(self.shared), ctypes.c_void_p(stream),
            arr, None)
        if res != 0:
            raise RuntimeError(f"cuLaunchKernel({self.caso}): {res}")
