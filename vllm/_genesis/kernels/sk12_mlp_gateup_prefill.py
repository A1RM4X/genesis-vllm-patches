# SPDX-License-Identifier: Apache-2.0
"""SK-12 — MLP gate_up + SiLU para PREFILL. Cadena sin Triton, de punta a punta.

El kernel vive en ``kernels/cuda/sk12_mlp_gateup_prefill.cu``. Este modulo lo
compila con ``nvcc -ptx`` (cacheado en disco), lo ensambla con ``ptxas`` y lo
lanza por ``libcuda``. En ningun punto interviene Triton: ni para compilar, ni
para elegir el ptxas, ni en runtime.

Diferencia con SK-01..SK-11
---------------------------
Esos se compilaron desde ``@triton.jit`` y sus PTX congelados
(``assets/ptx_sk/*.ptx``) son todos de M16/M20/M32/M40, o sea las tallas de
cudagraph de DECODE. Para prefill (M = 2048..8192) no habia ninguno. Un GEMM
tuneado a M=16 y uno a M=8192 no comparten ni el tile ni el reparto de warps.

El ciclo de mejora
------------------
1. Editar el ``.cu``.
2. ``python -m vllm._genesis.kernels.sk12_mlp_gateup_prefill --check`` valida
   contra la referencia de torch.
3. ``--bench`` mide contra ``cutlass_scaled_mm`` + SiLU por separado, que es el
   camino que hay que batir.
4. ``--ptx`` escupe el PTX generado para editarlo a mano si hace falta bajar
   mas (es el mismo flujo que documenta ``ptx_launcher.py``, pero arrancando de
   CUDA C en vez de Triton).

El PTX compilado se cachea en ``~/.cache/genesis/sk12/`` con el hash del fuente
y el arch, asi que recompilar sale solo cuando el ``.cu`` cambia.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import pathlib
import subprocess
import tempfile
import threading

import torch

_ARCH = 86
_OPT = 3
_ENTRY = "sk12_mlp_gateup_prefill"
import re as _re
# Tiene que seguir a GENESIS_SK12_DEFS: el .cu y el lanzamiento comparten NWARPS.
_DEFS = os.environ.get("GENESIS_SK12_DEFS", "")


def _def(nombre: str, por_defecto: int) -> int:
    m = _re.search(rf"-D{nombre}=(\d+)", _DEFS)
    return int(m.group(1)) if m else por_defecto


_WARPS = _def("NWARPS", 8)
_BM = _def("BM", 256)
_ATRIB_MAX_SHARED = 8   # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
_BN = _def("BN", 64)
_BK = _def("BK", 128)
# 2 etapas de A mas 2 etapas de B (gate y up intercalados). Supera los 48 KB
# estaticos, asi que va como shared dinamica con opt-in a 100 KB via
# cuFuncSetAttribute. sm_86 admite hasta 99 KB de dinamica por bloque.
_STAGES = _def("STAGES", 2)
_SHARED = _STAGES * (_BM * _BK + _BN * 2 * _BK)

_cerrojo = threading.RLock()
_libcuda_cache = None
_nativo = None


def _libcuda():
    """libcuda cruda. El contexto lo inicializa torch en su primer op de GPU."""
    global _libcuda_cache
    if _libcuda_cache is None:
        _libcuda_cache = ctypes.CDLL("libcuda.so.1")
    return _libcuda_cache


def fuente_cu() -> pathlib.Path:
    """Ruta del .cu.

    Vive al lado del modulo (``kernels/cuda/``) a proposito: asi viaja con el
    bind mount de ``_genesis`` y el contenedor lo encuentra sin montar nada mas.
    """
    cand = pathlib.Path(__file__).resolve().parent / "cuda" / "sk12_mlp_gateup_prefill.cu"
    if cand.is_file():
        return cand
    raise FileNotFoundError(f"no encuentro {cand}")


def _dir_cache() -> pathlib.Path:
    d = pathlib.Path(os.environ.get(
        "GENESIS_SK12_CACHE",
        os.path.expanduser("~/.cache/genesis/sk12")))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _nvcc() -> str:
    for c in ("/usr/local/cuda/bin/nvcc", "nvcc"):
        if c == "nvcc" or os.path.exists(c):
            return c
    return "nvcc"


def _ptxas() -> str:
    """El ptxas del CUDA del sistema.

    A diferencia de los SK viejos, aca NO se usa el ptxas que trae Triton: el
    PTX lo genera nvcc, asi que el ensamblador que corresponde es el del mismo
    toolkit.
    """
    for c in ("/usr/local/cuda/bin/ptxas", "ptxas"):
        if c == "ptxas" or os.path.exists(c):
            return c
    return "ptxas"


def compilar_ptx(forzar: bool = False) -> str:
    """``.cu`` -> PTX, cacheado por hash del fuente + arch."""
    src = fuente_cu()
    texto = src.read_text()
    # Barrido de configuracion sin editar el fuente:
    #   GENESIS_SK12_DEFS="-DNWARPS=8 -DWM=32"
    defs = os.environ.get("GENESIS_SK12_DEFS", "").split()
    clave = hashlib.sha256(
        (texto + f"|sm_{_ARCH}|O{_OPT}|{' '.join(defs)}").encode()).hexdigest()[:16]
    destino = _dir_cache() / f"sk12_{clave}_sm{_ARCH}.ptx"
    if destino.is_file() and not forzar:
        return destino.read_text()

    with tempfile.TemporaryDirectory() as d:
        salida = os.path.join(d, "k.ptx")
        r = subprocess.run(
            [_nvcc(), "-ptx", f"-arch=sm_{_ARCH}", f"-O{_OPT}",
             # Mata las divisiones IEEE y las conversiones exactas que
             # queden sueltas. Lo critico ya esta explicito con __fdividef
             # y __expf; esto barre el resto.
             "--use_fast_math", *defs,
             str(src), "-o", salida],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("nvcc fallo:\n" + r.stderr)
        ptx = pathlib.Path(salida).read_text()
    destino.write_text(ptx)
    return ptx


def _ensamblar(ptx: str) -> bytes:
    """PTX -> cubin. Levanta RuntimeError con el stderr completo si falla."""
    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, "k.ptx")
        fc = os.path.join(d, "k.cubin")
        pathlib.Path(fp).write_text(ptx)
        r = subprocess.run(
            [_ptxas(), f"-arch=sm_{_ARCH}", f"-O{_OPT}", fp, "-o", fc],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("ptxas fallo:\n" + r.stderr)
        return pathlib.Path(fc).read_bytes()


class _Nativo:
    """El cubin cargado y listo para lanzar. ABI de CUDA C, sin scratch."""

    def __init__(self) -> None:
        self._fn = None
        self._mod = None

    def _cargar(self):
        if self._fn is None:
            self._cargar_lento()

    @torch.compiler.disable
    def _cargar_lento(self):
        """Compila, ensambla y carga. Una sola vez, thread-safe.

        Va marcado con ``torch.compiler.disable`` por el mismo motivo que en los
        SK viejos: usa un lock y dynamo no sabe entrar a ese context manager.
        """
        with _cerrojo:
            if self._fn is not None:
                return
            cubin = _ensamblar(compilar_ptx())
            img = (ctypes.c_char * len(cubin)).from_buffer_copy(cubin)
            mod = ctypes.c_void_p()
            res = _libcuda().cuModuleLoadData(ctypes.byref(mod), img)
            if res != 0:
                raise RuntimeError(f"cuModuleLoadData: {res}")
            fn = ctypes.c_void_p()
            res = _libcuda().cuModuleGetFunction(
                ctypes.byref(fn), mod, _ENTRY.encode())
            if res != 0:
                raise RuntimeError(f"cuModuleGetFunction({_ENTRY}): {res}")
            # Sin este opt-in el lanzamiento falla con CUDA_ERROR_INVALID_VALUE:
            # el default por bloque son 48 KB y pedimos 64.
            if _SHARED > 48 * 1024:
                res = _libcuda().cuFuncSetAttribute(
                    fn, _ATRIB_MAX_SHARED, ctypes.c_int(_SHARED))
                if res != 0:
                    raise RuntimeError(
                        f"cuFuncSetAttribute(max_shared={_SHARED}): {res}")
            self._mod, self._fn = mod, fn

    def __call__(self, grid, args):
        self._cargar()
        vals = []
        for v in args:
            if isinstance(v, torch.Tensor):
                vals.append(ctypes.c_uint64(v.data_ptr()))
            else:
                vals.append(ctypes.c_int32(int(v)))
        arr = (ctypes.c_void_p * len(vals))(
            *[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in vals])
        gx, gy = grid
        res = _libcuda().cuLaunchKernel(
            self._fn, gx, gy, 1, _WARPS * 32, 1, 1, ctypes.c_uint(_SHARED),
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
            arr, None)
        if res != 0:
            raise RuntimeError(f"cuLaunchKernel: {res}")


def _obtener() -> _Nativo:
    global _nativo
    if _nativo is None:
        _nativo = _Nativo()
    return _nativo


def sk12_gateup_silu_gemm(a_i8, wg_i8, wu_i8, sa, sg, su):
    """``SiLU(A@Wg^T * escalas) * (A@Wu^T * escalas)`` en una pasada.

    :param a_i8:  int8 [M, K] contiguo, activaciones cuantizadas per-token.
    :param wg_i8: int8 [N, K] contiguo, pesos de gate.
    :param wu_i8: int8 [N, K] contiguo, pesos de up.
    :param sa:    fp32 [M] escala per-token.
    :param sg:    fp32 [N] escala per-canal de gate.
    :param su:    fp32 [N] escala per-canal de up.
    :returns:     fp16 [M, N].
    """
    M, K = a_i8.shape
    N = wg_i8.shape[0]
    if wg_i8.shape[1] != K or wu_i8.shape != wg_i8.shape:
        raise ValueError(
            f"formas incompatibles: A{tuple(a_i8.shape)} "
            f"Wg{tuple(wg_i8.shape)} Wu{tuple(wu_i8.shape)}")
    if K % _BK:
        raise ValueError(f"K={K} tiene que ser multiplo de {_BK} (BK del kernel)")

    out = torch.empty((M, N), dtype=torch.float16, device=a_i8.device)
    grid = ((N + _BN - 1) // _BN, (M + _BM - 1) // _BM)
    _obtener()(grid, [a_i8, wg_i8, wu_i8, sa, sg, su, out, M, N, K])
    return out


def referencia(a_i8, wg_i8, wu_i8, sa, sg, su):
    """La misma cuenta en torch, en fp32. Es la verdad contra la que se valida."""
    a = a_i8.float()
    g = (a @ wg_i8.float().t()) * sa[:, None] * sg[None, :]
    u = (a @ wu_i8.float().t()) * sa[:, None] * su[None, :]
    return (torch.nn.functional.silu(g) * u).half()


def _datos(M, K, N, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(0)
    a = torch.randint(-127, 128, (M, K), dtype=torch.int8, device=dev, generator=g)
    wg = torch.randint(-127, 128, (N, K), dtype=torch.int8, device=dev, generator=g)
    wu = torch.randint(-127, 128, (N, K), dtype=torch.int8, device=dev, generator=g)
    sa = torch.full((M,), 0.01, dtype=torch.float32, device=dev)
    sg = torch.full((N,), 0.01, dtype=torch.float32, device=dev)
    su = torch.full((N,), 0.01, dtype=torch.float32, device=dev)
    return a, wg, wu, sa, sg, su


def check(shapes=None) -> bool:
    """Valida contra la referencia. Devuelve True si todas pasan."""
    # K multiplo de 128 (BK). M y N a proposito NO alineados en dos casos,
    # para ejercitar las guardas de borde.
    shapes = shapes or [(64, 128, 64), (128, 256, 128), (256, 512, 256),
                        (2048, 5120, 512), (100, 256, 70)]
    ok = True
    for (M, K, N) in shapes:
        a, wg, wu, sa, sg, su = _datos(M, K, N)
        got = sk12_gateup_silu_gemm(a, wg, wu, sa, sg, su)
        exp = referencia(a, wg, wu, sa, sg, su)
        g, e = got.float(), exp.float()
        # Error relativo en norma L2: es la metrica correcta para un kernel
        # numerico. La anterior dividia elemento a elemento con clamp a 1e-2, y
        # eso castiga sin sentido los elementos donde la referencia es ~0 (que
        # con SiLU son muchos: para g negativo el resultado tiende a cero).
        l2 = ((g - e).norm() / e.norm().clamp_min(1e-12)).item()
        # ulp de fp16 en el rango del resultado, como referencia de que es
        # "tan bueno como fp16 puede ser"
        esc = e.abs().max().item()
        ulp = 2.0 ** (max(-24, int(__import__("math").floor(
            __import__("math").log2(max(esc, 1e-6)))) - 10))
        d = (g - e).abs()
        bien = l2 < 5e-3
        ok &= bien
        print(f"  M={M:<6} K={K:<6} N={N:<6} err_L2={l2:.6f} "
              f"max_abs={d.max():.4f} (ulp fp16 ~{ulp:.4f})  "
              f"{'OK' if bien else 'FALLA'}")
    return ok


def bench(M=2048, K=5120, N=8704, it=20):
    """Mide contra cutlass_scaled_mm + SiLU por separado, que es el rival."""
    a, wg, wu, sa, sg, su = _datos(M, K, N)
    flops = 2 * 2 * M * K * N  # dos GEMM

    def cron(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        import time
        t0 = time.perf_counter()
        for _ in range(it):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / it

    t_sk = cron(lambda: sk12_gateup_silu_gemm(a, wg, wu, sa, sg, su))
    print(f"  SK-12 fusionado        {t_sk*1e3:8.3f} ms   {flops/t_sk/1e12:7.1f} TOPS")

    try:
        from vllm import _custom_ops as ops
        wgt = wg.t().contiguous().t()
        wut = wu.t().contiguous().t()
        sa2 = sa.view(-1, 1)

        def rival():
            g = ops.cutlass_scaled_mm(a, wgt, sa2, sg.view(1, -1), torch.float16)
            u = ops.cutlass_scaled_mm(a, wut, sa2, su.view(1, -1), torch.float16)
            return torch.nn.functional.silu(g) * u

        rival()
        t_cl = cron(rival)
        print(f"  cutlass x2 + SiLU      {t_cl*1e3:8.3f} ms   "
              f"{flops/t_cl/1e12:7.1f} TOPS")
        print(f"  -> SK-12 {'GANA' if t_sk < t_cl else 'PIERDE'} "
              f"{max(t_cl, t_sk)/min(t_cl, t_sk):.2f}x")
    except Exception as e:
        print(f"  cutlass no medible: {type(e).__name__}: {e}")


if __name__ == "__main__":
    import sys
    if "--ptx" in sys.argv:
        print(compilar_ptx(forzar=True))
    elif "--bench" in sys.argv:
        bench()
    else:
        print("validacion contra referencia:")
        raise SystemExit(0 if check() else 1)
