# SPDX-License-Identifier: Apache-2.0
"""SK-13 — MLP gate_up + SiLU para PREFILL. Cadena sin Triton, de punta a punta.

El kernel vive en ``kernels/cuda/sk13_mlp_gateup_w4a8.cu``. Este modulo lo
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
2. ``python -m vllm._genesis.kernels.sk13_mlp_gateup_w4a8 --check`` valida
   contra la referencia de torch.
3. ``--bench`` mide contra ``cutlass_scaled_mm`` + SiLU por separado, que es el
   camino que hay que batir.
4. ``--ptx`` escupe el PTX generado para editarlo a mano si hace falta bajar
   mas (es el mismo flujo que documenta ``ptx_launcher.py``, pero arrancando de
   CUDA C en vez de Triton).

El PTX compilado se cachea en ``~/.cache/genesis/sk13/`` con el hash del fuente
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
_ENTRY = "sk13_mlp_gateup_w4a8"
_WARPS = 8
_BM = 128
_BN = 64

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
    cand = pathlib.Path(__file__).resolve().parent / "cuda" / "sk13_mlp_gateup_w4a8.cu"
    if cand.is_file():
        return cand
    raise FileNotFoundError(f"no encuentro {cand}")


def _dir_cache() -> pathlib.Path:
    d = pathlib.Path(os.environ.get(
        "GENESIS_SK12_CACHE",
        os.path.expanduser("~/.cache/genesis/sk13")))
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
    clave = hashlib.sha256(
        (texto + f"|sm_{_ARCH}|O{_OPT}").encode()).hexdigest()[:16]
    destino = _dir_cache() / f"sk13_{clave}_sm{_ARCH}.ptx"
    if destino.is_file() and not forzar:
        return destino.read_text()

    with tempfile.TemporaryDirectory() as d:
        salida = os.path.join(d, "k.ptx")
        r = subprocess.run(
            [_nvcc(), "-ptx", f"-arch=sm_{_ARCH}", f"-O{_OPT}",
             # Mata las divisiones IEEE y las conversiones exactas que
             # queden sueltas. Lo critico ya esta explicito con __fdividef
             # y __expf; esto barre el resto.
             "--use_fast_math",
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
            self._fn, gx, gy, 1, _WARPS * 32, 1, 1, ctypes.c_uint(0),
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
            arr, None)
        if res != 0:
            raise RuntimeError(f"cuLaunchKernel: {res}")


def _obtener() -> _Nativo:
    global _nativo
    if _nativo is None:
        _nativo = _Nativo()
    return _nativo




GRP = 128


def empaquetar(w4):
    """[N, K] uint8 con valores 0..15  ->  [N, K/2] empaquetado.

    byte i del bloque de 32 k = w4[k=i] | (w4[k=i+16] << 4). Ese orden es el
    que hace que, tras ldmatrix, un solo AND y un shift dejen los fragmentos
    b0 y b1 exactamente donde mma.m16n8k32 los espera.
    """
    N, K = w4.shape
    assert K % 32 == 0
    v = w4.reshape(N, K // 32, 32)
    lo, hi = v[..., :16], v[..., 16:]
    return (lo | (hi << 4)).reshape(N, K // 2).contiguous()


def sk13_gateup_silu_w4a8(a_i8, wp, sa, sgrp, zp, N):
    """SiLU(gate) * up con pesos int4 desempaquetados en el kernel.

    :param a_i8: int8 [M, K]
    :param wp:   uint8 [N, 2, K/2] empaquetado por `empaquetar`
    :param sa:   fp32 [M]
    :param sgrp: fp32 [K/128, 2, N] escala por grupo
    :param zp:   int8 [K/128, 2, N] zero point por grupo
    """
    M, K = a_i8.shape
    if K % GRP:
        raise ValueError(f"K={K} tiene que ser multiplo de {GRP} (grupo AWQ)")
    out = torch.empty((M, N), dtype=torch.float16, device=a_i8.device)
    grid = ((N + _BN - 1) // _BN, (M + _BM - 1) // _BM)
    _obtener()(grid, [a_i8, wp, sa, sgrp, zp, out, M, N, K])
    return out


def referencia(a_i8, w4g, w4u, sa, sgrp, zp):
    """La misma cuenta en torch fp32. w4g/w4u son [N, K] uint8 sin empaquetar."""
    N, K = w4g.shape
    G = K // GRP
    a = a_i8.float()
    accg = torch.zeros(a.shape[0], N, device=a.device)
    accu = torch.zeros_like(accg)
    for g in range(G):
        sl = slice(g * GRP, (g + 1) * GRP)
        wg = w4g[:, sl].float() - zp[g, 0].float()[:, None]
        wu = w4u[:, sl].float() - zp[g, 1].float()[:, None]
        accg += (a[:, sl] @ wg.t()) * sgrp[g, 0][None, :]
        accu += (a[:, sl] @ wu.t()) * sgrp[g, 1][None, :]
    g = accg * sa[:, None]
    u = accu * sa[:, None]
    return (torch.nn.functional.silu(g) * u).half()


def _datos(M, K, N, dev="cuda"):
    gen = torch.Generator(device=dev).manual_seed(0)
    a = torch.randint(-127, 128, (M, K), dtype=torch.int8, device=dev, generator=gen)
    w4g = torch.randint(0, 16, (N, K), dtype=torch.uint8, device=dev, generator=gen)
    w4u = torch.randint(0, 16, (N, K), dtype=torch.uint8, device=dev, generator=gen)
    G = K // GRP
    zp = torch.full((G, 2, N), 8, dtype=torch.int8, device=dev)
    sgrp = torch.full((G, 2, N), 0.01, dtype=torch.float32, device=dev)
    sa = torch.full((M,), 0.01, dtype=torch.float32, device=dev)
    wp = torch.stack([empaquetar(w4g), empaquetar(w4u)], dim=1).contiguous()
    return a, wp, w4g, w4u, sa, sgrp, zp


def check(shapes=None) -> bool:
    shapes = shapes or [(128, 128, 64), (256, 256, 128), (2048, 5120, 512),
                        (100, 256, 70)]
    ok = True
    for (M, K, N) in shapes:
        a, wp, w4g, w4u, sa, sgrp, zp = _datos(M, K, N)
        got = sk13_gateup_silu_w4a8(a, wp, sa, sgrp, zp, N)
        exp = referencia(a, w4g, w4u, sa, sgrp, zp)
        d = (got.float() - exp.float()).abs()
        rel = (d / exp.float().abs().clamp_min(1e-2)).max().item()
        bien = rel < 3e-2
        ok &= bien
        print(f"  M={M:<6} K={K:<6} N={N:<6} max_abs={d.max():.5f} "
              f"max_rel={rel:.5f}  {'OK' if bien else 'FALLA'}")
    return ok


def bench(M=2048, K=5120, N=8704, it=20):
    a, wp, w4g, w4u, sa, sgrp, zp = _datos(M, K, N)
    flops = 2 * 2 * M * K * N

    def cron(fn):
        for _ in range(5): fn()
        torch.cuda.synchronize()
        import time
        t0 = time.perf_counter()
        for _ in range(it): fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / it

    t = cron(lambda: sk13_gateup_silu_w4a8(a, wp, sa, sgrp, zp, N))
    print(f"  SK-13 W4A8            {t*1e3:8.3f} ms   {flops/t/1e12:7.1f} TOPS")
    try:
        from vllm._genesis.kernels import sk12_mlp_gateup_prefill as k12
        a2, wg2, wu2, sa2, sg2, su2 = k12._datos(M, K, N)
        t12 = cron(lambda: k12.sk12_gateup_silu_gemm(a2, wg2, wu2, sa2, sg2, su2))
        print(f"  SK-12 W8A8 (misma FLOP) {t12*1e3:6.3f} ms   {flops/t12/1e12:7.1f} TOPS")
        print(f"  -> W4A8 {'GANA' if t < t12 else 'PIERDE'} {max(t,t12)/min(t,t12):.2f}x")
    except Exception as e:
        print(f"  SK-12 no comparable: {type(e).__name__}: {e}")


if __name__ == "__main__":
    import sys
    if "--ptx" in sys.argv:
        print(compilar_ptx(forzar=True))
    elif "--bench" in sys.argv:
        bench()
    else:
        print("validacion contra referencia:")
        raise SystemExit(0 if check() else 1)
