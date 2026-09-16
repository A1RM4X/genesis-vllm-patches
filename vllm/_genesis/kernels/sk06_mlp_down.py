# SPDX-License-Identifier: Apache-2.0
"""SK-06 — MLP_DOWN_INT8_SCALED_RESIDUAL — capa completa down_proj (RowParallel) en GPU.

Capa completa (Qwen3.5-27B, 64 capas + 1 MTP, TP=2):
    act bf16 [M, 8704]               (salida de SiLU(gate)*up, sin norm delante)
      -> quant per-token amax/127                              (kernel 1)
      -> GEMM INT8 diádico + residual fusionado en el epílogo  (kernel 2)
      -> all_reduce NCCL                                       (colectivo GPU)

Geometría: global [5120, 17408] = [40*128, 136*128]; per-rank TP=2 K=8704, N=5120.

Diseño (sm_86, mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32):
  * ``BLOCK_K == SHIFT_BLOCK == 128`` -> un ``tl.dot`` por bloque diádico; con
    K=8704 son 68 bloques, antes eran 68*4 = 272 vaciados del acumulador.
  * Shift por columna como ``acc += int_acc.to(f32) * exp2(shift)``.
  * Escalas fuera del bucle k, punteros que avanzan, grid 1-D con swizzle L2.
  * Residual fusionado en el epílogo (única fusión real que ya tenía SK-06;
    aquí se conserva y se le quita la pasada fp32 intermedia).

RowParallel: con TP>1 el residual se suma después del all_reduce; con TP=1 va
fusionado en el epílogo.

Los kernels son branchless y no validan nada: se asume todo comprobado.
"""

from __future__ import annotations
import ctypes
import os
import subprocess
import tempfile
import threading

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Plomeria PTX. DUPLICADA A PROPOSITO en cada archivo de kernel: este modulo no
# importa plomeria compartida de ningun lado. Si hay que arreglar lo mismo en
# 18 archivos, se arregla en 18 archivos; es la decision explicita del
# proyecto -- codigo repetido antes que codigo compartido.
# ---------------------------------------------------------------------------

_ARCH = 86
_OPT = 3
# 48 KB es el maximo de shared DINAMICA por defecto en sm_86. Los tiles de
# prefill piden 73728 B (72 KB), asi que hay que pedirlo explicito con
# cuFuncSetAttribute o el launch falla con CUDA_ERROR_INVALID_VALUE (1).
_TECHO_SHARED = 49152
_ATRIB_MAX_SHARED = 8   # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES

_cerrojo = threading.Lock()
_libcuda_cache = None


def _libcuda():
    """libcuda cruda. El contexto lo inicializa torch en su primer op de GPU."""
    global _libcuda_cache
    if _libcuda_cache is None:
        _libcuda_cache = ctypes.CDLL("libcuda.so.1")
    return _libcuda_cache


def _ruta_ptxas() -> str:
    """El ptxas que trae Triton, no el del sistema.

    Triton emite PTX con una version de ISA concreta; el ptxas del CUDA
    instalado puede ser mas viejo o mas nuevo y rechazarlo.
    """
    base = os.path.dirname(__import__("triton").__file__)
    cand = os.path.join(base, "backends", "nvidia", "bin", "ptxas")
    return cand if os.path.exists(cand) else "ptxas"


def _ensamblar(ptx: str) -> bytes:
    """PTX -> cubin. Levanta RuntimeError con el stderr completo si falla."""
    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, "k.ptx")
        fc = os.path.join(d, "k.cubin")
        with open(fp, "w") as f:
            f.write(ptx)
        r = subprocess.run([_ruta_ptxas(), "-arch=sm_%d" % _ARCH, "-O%d" % _OPT,
                            fp, "-o", fc], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("ptxas fallo:\n" + r.stderr)
        with open(fc, "rb") as f:
            return f.read()


class _Nativo:
    """Una variante de PTX embebida: ensambla bajo demanda y lanza por libcuda.

    :param caso: id de diagnostico.
    :param ptx: el asm completo.
    :param entry: nombre de la funcion dentro del modulo.
    :param warps: warps por bloque (blockDim.x = warps * 32).
    :param shared: bytes de shared dinamica que pide el cubin.
    :param abi: indices, sobre la lista completa de args del kernel original,
        de los params que SI viajan en el ABI, en orden. Salen de
        ``handle.src.constants``: son los que NO quedaron especializados.
    :param horneado: ``posicion -> valor`` de los params horneados en el cubin.
        Se verifican en cada lanzamiento: si llega otro valor el resultado
        seria incorrecto EN SILENCIO, asi que se levanta ValueError.
    :param div16: posiciones cuyo valor tiene que seguir siendo multiplo de 16.
        Triton lo asume al compilar (``tt.divisibility``) y lo usa para
        vectorizar los accesos. Con un valor que no cumple sale mal en
        silencio: el JIT recompilaria, el PTX embebido no puede.
    """

    def __init__(self, caso, ptx, entry, warps, shared, abi, horneado, div16):
        self.caso = caso
        self.ptx = ptx
        self.entry = entry
        self.warps = warps
        self.shared = shared
        self.abi = abi
        self.horneado = horneado
        self.div16 = div16
        self._fn = None
        self._mod = None

    def _cargar(self):
        """Camino rapido: si ya esta cargado no hace nada.

        El ensamblado vive aparte y marcado con ``torch.compiler.disable``
        porque usa un lock, y dynamo no sabe entrar a un context manager de
        `lock`: falla con "Unsupported context manager". El lanzamiento cae
        DENTRO de la region que vLLM compila (el forward del modelo), asi que
        con el `with` en el camino trazado el arranque se muere en
        ``profile_run`` -> ``_dummy_run``. Medido: ensamblar y cargar cuesta
        ~25 ms por variante y pasa una sola vez, asi que el corte de grafo del
        camino lento no se paga en regimen.
        """
        if self._fn is None:
            self._ensamblar_y_cargar()

    @torch.compiler.disable
    def _ensamblar_y_cargar(self):
        """Ensambla con ptxas y carga el modulo. Una sola vez, thread-safe."""
        with _cerrojo:
            if self._fn is not None:
                return
            cubin = _ensamblar(self.ptx)
            img = (ctypes.c_char * len(cubin)).from_buffer_copy(cubin)
            mod = ctypes.c_void_p()
            res = _libcuda().cuModuleLoadData(ctypes.byref(mod), img)
            if res != 0:
                raise RuntimeError("cuModuleLoadData(%s): %d" % (self.caso, res))
            fn = ctypes.c_void_p()
            res = _libcuda().cuModuleGetFunction(ctypes.byref(fn), mod,
                                                 self.entry.encode())
            if res != 0:
                raise RuntimeError("cuModuleGetFunction(%s): %d" % (self.caso, res))
            if self.shared > _TECHO_SHARED:
                res = _libcuda().cuFuncSetAttribute(
                    fn, _ATRIB_MAX_SHARED, ctypes.c_int(self.shared))
                if res != 0:
                    raise RuntimeError("cuFuncSetAttribute(%s, %dB): %d"
                                       % (self.caso, self.shared, res))
            self._mod, self._fn = mod, fn

    def __call__(self, grid, *args):
        """Lanza. ``args`` es la lista COMPLETA en orden de declaracion del
        kernel original; se consumen las posiciones de ``abi``."""
        self._cargar()
        for pos, val in self.horneado.items():
            if args[pos] != val:
                raise ValueError(
                    "%s: el cubin esta horneado con param[%d]=%r y llego %r; "
                    "este asm no aplica a estos inputs"
                    % (self.caso, pos, val, args[pos]))
        for pos in self.div16:
            if args[pos] % 16:
                raise ValueError(
                    "%s: param[%d]=%r tiene que ser multiplo de 16 (Triton lo "
                    "asumio al compilar para vectorizar)"
                    % (self.caso, pos, args[pos]))
        vals = []
        for i in self.abi:
            v = args[i]
            if isinstance(v, torch.Tensor):
                vals.append(ctypes.c_uint64(v.data_ptr()))
            elif isinstance(v, float):
                vals.append(ctypes.c_float(v))
            else:
                vals.append(ctypes.c_int32(int(v)))
        # ABI de Triton: params del kernel + global_scratch + profile_scratch.
        # global_scratch_size es 0 en estos kernels, asi que NULL es seguro.
        vals.append(ctypes.c_uint64(0))
        vals.append(ctypes.c_uint64(0))
        arr = (ctypes.c_void_p * len(vals))(
            *[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in vals])
        gx, gy, gz = (list(grid) + [1, 1])[:3]
        res = _libcuda().cuLaunchKernel(
            self._fn, gx, gy, gz, self.warps * 32, 1, 1,
            ctypes.c_uint(self.shared),
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
            arr, None)
        if res != 0:
            raise RuntimeError("cuLaunchKernel(%s): %d" % (self.caso, res))





# ---------------------------------------------------------------------------
# Plomeria PTX. DUPLICADA A PROPOSITO en cada archivo de kernel: este modulo no
# importa plomeria compartida de ningun lado. Si hay que arreglar lo mismo en
# 18 archivos, se arregla en 18 archivos; es la decision explicita del
# proyecto -- codigo repetido antes que codigo compartido.
# ---------------------------------------------------------------------------

_ARCH = 86
_OPT = 3
# 48 KB es el maximo de shared DINAMICA por defecto en sm_86. Los tiles de
# prefill piden 73728 B (72 KB), asi que hay que pedirlo explicito con
# cuFuncSetAttribute o el launch falla con CUDA_ERROR_INVALID_VALUE (1).
_TECHO_SHARED = 49152
_ATRIB_MAX_SHARED = 8   # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES

_cerrojo = threading.Lock()
_libcuda_cache = None


def _libcuda():
    """libcuda cruda. El contexto lo inicializa torch en su primer op de GPU."""
    global _libcuda_cache
    if _libcuda_cache is None:
        _libcuda_cache = ctypes.CDLL("libcuda.so.1")
    return _libcuda_cache


def _ruta_ptxas() -> str:
    """El ptxas que trae Triton, no el del sistema.

    Triton emite PTX con una version de ISA concreta; el ptxas del CUDA
    instalado puede ser mas viejo o mas nuevo y rechazarlo.
    """
    base = os.path.dirname(__import__("triton").__file__)
    cand = os.path.join(base, "backends", "nvidia", "bin", "ptxas")
    return cand if os.path.exists(cand) else "ptxas"


def _ensamblar(ptx: str) -> bytes:
    """PTX -> cubin. Levanta RuntimeError con el stderr completo si falla."""
    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, "k.ptx")
        fc = os.path.join(d, "k.cubin")
        with open(fp, "w") as f:
            f.write(ptx)
        r = subprocess.run([_ruta_ptxas(), "-arch=sm_%d" % _ARCH, "-O%d" % _OPT,
                            fp, "-o", fc], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("ptxas fallo:\n" + r.stderr)
        with open(fc, "rb") as f:
            return f.read()


class _Nativo:
    """Una variante de PTX embebida: ensambla bajo demanda y lanza por libcuda.

    :param caso: id de diagnostico.
    :param ptx: el asm completo.
    :param entry: nombre de la funcion dentro del modulo.
    :param warps: warps por bloque (blockDim.x = warps * 32).
    :param shared: bytes de shared dinamica que pide el cubin.
    :param abi: indices, sobre la lista completa de args del kernel original,
        de los params que SI viajan en el ABI, en orden. Salen de
        ``handle.src.constants``: son los que NO quedaron especializados.
    :param horneado: ``posicion -> valor`` de los params horneados en el cubin.
        Se verifican en cada lanzamiento: si llega otro valor el resultado
        seria incorrecto EN SILENCIO, asi que se levanta ValueError.
    :param div16: posiciones cuyo valor tiene que seguir siendo multiplo de 16.
        Triton lo asume al compilar (``tt.divisibility``) y lo usa para
        vectorizar los accesos. Con un valor que no cumple sale mal en
        silencio: el JIT recompilaria, el PTX embebido no puede.
    """

    def __init__(self, caso, ptx, entry, warps, shared, abi, horneado, div16):
        self.caso = caso
        self.ptx = ptx
        self.entry = entry
        self.warps = warps
        self.shared = shared
        self.abi = abi
        self.horneado = horneado
        self.div16 = div16
        self._fn = None
        self._mod = None

    def _cargar(self):
        """Camino rapido: si ya esta cargado no hace nada.

        El ensamblado vive aparte y marcado con ``torch.compiler.disable``
        porque usa un lock, y dynamo no sabe entrar a un context manager de
        `lock`: falla con "Unsupported context manager". El lanzamiento cae
        DENTRO de la region que vLLM compila (el forward del modelo), asi que
        con el `with` en el camino trazado el arranque se muere en
        ``profile_run`` -> ``_dummy_run``. Medido: ensamblar y cargar cuesta
        ~25 ms por variante y pasa una sola vez, asi que el corte de grafo del
        camino lento no se paga en regimen.
        """
        if self._fn is None:
            self._ensamblar_y_cargar()

    @torch.compiler.disable
    def _ensamblar_y_cargar(self):
        """Ensambla con ptxas y carga el modulo. Una sola vez, thread-safe."""
        with _cerrojo:
            if self._fn is not None:
                return
            cubin = _ensamblar(self.ptx)
            img = (ctypes.c_char * len(cubin)).from_buffer_copy(cubin)
            mod = ctypes.c_void_p()
            res = _libcuda().cuModuleLoadData(ctypes.byref(mod), img)
            if res != 0:
                raise RuntimeError("cuModuleLoadData(%s): %d" % (self.caso, res))
            fn = ctypes.c_void_p()
            res = _libcuda().cuModuleGetFunction(ctypes.byref(fn), mod,
                                                 self.entry.encode())
            if res != 0:
                raise RuntimeError("cuModuleGetFunction(%s): %d" % (self.caso, res))
            if self.shared > _TECHO_SHARED:
                res = _libcuda().cuFuncSetAttribute(
                    fn, _ATRIB_MAX_SHARED, ctypes.c_int(self.shared))
                if res != 0:
                    raise RuntimeError("cuFuncSetAttribute(%s, %dB): %d"
                                       % (self.caso, self.shared, res))
            self._mod, self._fn = mod, fn

    def __call__(self, grid, *args):
        """Lanza. ``args`` es la lista COMPLETA en orden de declaracion del
        kernel original; se consumen las posiciones de ``abi``."""
        self._cargar()
        for pos, val in self.horneado.items():
            if args[pos] != val:
                raise ValueError(
                    "%s: el cubin esta horneado con param[%d]=%r y llego %r; "
                    "este asm no aplica a estos inputs"
                    % (self.caso, pos, val, args[pos]))
        for pos in self.div16:
            if args[pos] % 16:
                raise ValueError(
                    "%s: param[%d]=%r tiene que ser multiplo de 16 (Triton lo "
                    "asumio al compilar para vectorizar)"
                    % (self.caso, pos, args[pos]))
        vals = []
        for i in self.abi:
            v = args[i]
            if isinstance(v, torch.Tensor):
                vals.append(ctypes.c_uint64(v.data_ptr()))
            elif isinstance(v, float):
                vals.append(ctypes.c_float(v))
            else:
                vals.append(ctypes.c_int32(int(v)))
        # ABI de Triton: params del kernel + global_scratch + profile_scratch.
        # global_scratch_size es 0 en estos kernels, asi que NULL es seguro.
        vals.append(ctypes.c_uint64(0))
        vals.append(ctypes.c_uint64(0))
        arr = (ctypes.c_void_p * len(vals))(
            *[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in vals])
        gx, gy, gz = (list(grid) + [1, 1])[:3]
        res = _libcuda().cuLaunchKernel(
            self._fn, gx, gy, gz, self.warps * 32, 1, 1,
            ctypes.c_uint(self.shared),
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
            arr, None)
        if res != 0:
            raise RuntimeError("cuLaunchKernel(%s): %d" % (self.caso, res))




SK_ID = "SK-06"
SK_NAME = "MLP_DOWN_INT8_SCALED_RESIDUAL"
GLOBAL_N = 5120
GLOBAL_K = 17408
PER_RANK_N = 5120
PER_RANK_K = 8704
GLOBAL_SHAPE = (GLOBAL_N, GLOBAL_K)
RANK_SHAPE = (PER_RANK_N, PER_RANK_K)
NUM_LAYERS = 65
ROW_PARALLEL = True

BLOCK: int = 128
SHIFT_BLOCK: int = 128


_CFG: tuple[tuple[int, int, int, int, int, int], ...] = (
    # (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages) por bucket de M.
    # Medido en RTX 3090 con relojes fijos a 1500 MHz, CUDA events, salida
    # prealocada y configs intercaladas round-robin (sin eso el ruido es +-25%
    # y la busqueda devuelve resultados incoherentes).
    (16, 128, 128, 8, 8, 3),    # M <=   32   decode
    (128, 128, 128, 8, 8, 3),   # M <=  128
    (256, 128, 64, 8, 8, 4),    # M <= 1024
    (256, 128, 64, 8, 8, 4),    # M  > 1024   prefill
)
# Fraccion de ola por debajo de la cual conviene bajar BLOCK_M aunque se pierda
# intensidad aritmetica.
SM_COUNT: int = 82          # GA102 (RTX 3090)
_CTA_MIN: int = 49          # ~0.6 olas
_CFG_POCOS_CTA: tuple[int, int, int, int, int, int] = (64, 128, 128, 8, 4, 4)

BLOCK_M: int = 128
BLOCK_N: int = 128
BLOCK_K: int = 128

_ZERO: dict[torch.device, torch.Tensor] = {}


def _zero(device: torch.device) -> torch.Tensor:
    """Escalar cero por device, usado como residual neutro con strides 0."""
    z = _ZERO.get(device)
    if z is None:
        z = torch.zeros((), dtype=torch.bfloat16, device=device)
        _ZERO[device] = z
    return z


def _has_shift(shifts: torch.Tensor) -> bool:
    """True si el tensor de shifts tiene algun valor distinto de cero.

    Cacheado directamente como atributo del objeto Tensor (shifts._has_shift):
    - Cero sincronizaciones GPU->CPU en llamadas subsiguientes.
    - Ciclo de vida atado al tensor: imposible colision por reutilizacion de memoria.
    """
    v = getattr(shifts, "_has_shift", None)
    if v is None:
        if torch.cuda.is_current_stream_capturing():
            return True
        v = bool(shifts.any().item())
        try:
            shifts._has_shift = v
        except Exception:
            pass
    return v


def _cfg(m: int, n: int) -> tuple[int, int, int, int, int, int]:
    """Tile por bucket de M, con correccion cuando faltan CTAs.

    Se probo elegir el tile puramente por conteo de CTAs y salio PEOR que la
    tabla (mediana 0.93x, y hasta 0.35x a M=128): con BLOCK_M=16 el mma
    m16n8k32 queda al minimo y B se re-lee 8 veces. El conteo de CTAs solo
    manda cuando el tile de la tabla no llega ni a media ola, que es lo que
    pasa con N=5120 a M=128 (40 CTAs para 82 SM): ahi bajar a BLOCK_M=64 los
    duplica y da 1.41-1.45x.
    """
    c = _CFG[(m > 32) + (m > 128) + (m > 1024)]
    if m > 32 and -(-m // c[0]) * -(-n // c[1]) < _CTA_MIN:
        return _CFG_POCOS_CTA
    return c





@triton.jit
def _sk06_mlp_down_kernel(
    a_ptr, b_ptr, out_ptr, resid_ptr, a_scale_ptr, b_scale_ptr, shifts_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_out_m, stride_out_n, stride_res_m, stride_res_n,
    stride_shift_k, stride_shift_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, SHIFT_BLOCK: tl.constexpr, HAS_SHIFT: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_in_group = GROUP_M * num_pid_n
    first_m = (pid // pid_in_group) * GROUP_M
    group_m = min(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + ((pid % pid_in_group) % group_m)
    pid_n = (pid % pid_in_group) // group_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    sh_ptrs = shifts_ptr + (offs_n // SHIFT_BLOCK) * stride_shift_n

    # HAS_SHIFT es constexpr: el Diseno B (shifts todos cero, el default de
    # produccion) compila un bucle SIN nada de shift, con un unico acumulador
    # int32 vivo. Medido: 158 registros/hilo contra 234 de la version con
    # shift, y de ahi salen 1.20-1.27x en M>=512. La variante con
    # tl.where(sh>=0, d<<sh, d>>-sh) materializa DOS temporales int32 [BM,BN]
    # mas el select, o sea la misma presion de registros que tenia el fp32.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, K // BLOCK_K):
        d = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), out_dtype=tl.float32)
        if HAS_SHIFT:
            b_sc = tl.load(sh_ptrs + (kb * BLOCK_K // SHIFT_BLOCK) * stride_shift_k).to(tl.float32)[None, :]
            acc += d * b_sc
        else:
            acc += d
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    out = acc * tl.load(a_scale_ptr + offs_m).to(tl.float32)[:, None]
    if not HAS_SHIFT:
        out = out * tl.load(b_scale_ptr + offs_n).to(tl.float32)[None, :]
    out += tl.load(resid_ptr + offs_m[:, None] * stride_res_m + offs_n[None, :] * stride_res_n).to(tl.float32)

    out_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(
        out_ptr + out_m[:, None] * stride_out_m + out_n[None, :] * stride_out_n,
        out.to(out_ptr.dtype.element_ty),
        mask=(out_m[:, None] < M) & (out_n[None, :] < N),
    )


def sk06_quant(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quant per-token INT8 -> ``(q [M,K] int8, s [M] fp32)``."""
    from vllm._genesis.kernels.sk09_norm_embed import quant_per_token
    q, s = quant_per_token(hidden)
    return q, s.reshape(-1)


def mlp_down_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    shifts: torch.Tensor,
    residual: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """GEMM INT8 diádico + residual fusionado. ``a`` int8 [M,K], ``b`` int8 [K,N]."""
    if b.stride(0) != 1:
        b = b.t().contiguous().t()
    M, K = a.shape
    N = b.shape[1]
    res = _zero(a.device) if residual is None else residual
    rm, rn = (0, 0) if residual is None else (res.stride(0), res.stride(1))
    has_shift = _has_shift(shifts)
    bm, bn, bk, gm, warps, stages = _cfg(M, N)
    out = torch.empty((M, N), dtype=out_dtype, device=a.device)
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    _lanzar(grid, (bm, bn, bk, gm, warps, stages), has_shift,
            a,
            b,
            out,
            res,
            a_scales,
            b_scales,
            shifts,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
            rm,
            rn,
            shifts.stride(0),
            shifts.stride(1),
            bm,
            bn,
            bk,
            gm,
            SHIFT_BLOCK,
            has_shift)
    return out


def mlp_down_int8_scaled_residual(
    act: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    residual: torch.Tensor | None = None,
    shifts: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    do_allreduce: bool = True,
) -> torch.Tensor:
    """Capa completa down_proj: quant -> GEMM diádico -> all_reduce -> residual."""
    if shifts is None:
        shifts = torch.zeros((weight.shape[0] // SHIFT_BLOCK, weight.shape[1] // SHIFT_BLOCK),
                             dtype=torch.float32, device=weight.device)
    a, a_scales = sk06_quant(act)
    if not do_allreduce or not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return mlp_down_gemm(a, weight, a_scales, weight_scale, shifts, residual, out_dtype)
    out = mlp_down_gemm(a, weight, a_scales, weight_scale, shifts, None, out_dtype)
    torch.distributed.all_reduce(out)
    if residual is not None:
        out.add_(residual)
    return out


mlp_down = mlp_down_int8_scaled_residual
sk06_mlp_down = mlp_down_int8_scaled_residual
sk06_mlp_down_gemm = mlp_down_gemm

__all__ = [
    "SK_ID", "SK_NAME", "GLOBAL_N", "GLOBAL_K", "PER_RANK_N", "PER_RANK_K",
    "GLOBAL_SHAPE", "RANK_SHAPE", "NUM_LAYERS", "ROW_PARALLEL",
    "BLOCK", "SHIFT_BLOCK", "BLOCK_M", "BLOCK_N", "BLOCK_K",
    "sk06_quant", "mlp_down_gemm", "mlp_down_int8_scaled_residual", "mlp_down",
    "sk06_mlp_down", "sk06_mlp_down_gemm",
]



# --- variantes PTX embebidas ---

_PTX_0 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk06_mlp_down_kernel   // -- Begin function _sk06_mlp_down_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk06_mlp_down_kernel
.visible .entry _sk06_mlp_down_kernel(
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_6,
	.param .u32 _sk06_mlp_down_kernel_param_7,
	.param .u32 _sk06_mlp_down_kernel_param_8,
	.param .u32 _sk06_mlp_down_kernel_param_9,
	.param .u32 _sk06_mlp_down_kernel_param_10,
	.param .u32 _sk06_mlp_down_kernel_param_11,
	.param .u32 _sk06_mlp_down_kernel_param_12,
	.param .u32 _sk06_mlp_down_kernel_param_13,
	.param .u32 _sk06_mlp_down_kernel_param_14,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_16
)
.reqntid 256
{
	.reg .pred 	%p<11>;
	.reg .b16 	%rs<17>;
	.reg .b32 	%r<287>;
	.reg .b64 	%rd<55>;
	.loc	1 304 0                         // sk06_mlp_down.py:304:0
$L__func_begin0:
	.loc	1 304 0                         // sk06_mlp_down.py:304:0

// %bb.0:
	ld.param.b32 	%r23, [_sk06_mlp_down_kernel_param_13];
	ld.param.b32 	%r22, [_sk06_mlp_down_kernel_param_12];
	ld.param.b32 	%r21, [_sk06_mlp_down_kernel_param_8];
	ld.param.b32 	%r20, [_sk06_mlp_down_kernel_param_7];
	ld.param.b64 	%rd18, [_sk06_mlp_down_kernel_param_5];
	ld.param.b64 	%rd17, [_sk06_mlp_down_kernel_param_4];
	ld.param.b64 	%rd16, [_sk06_mlp_down_kernel_param_3];
	ld.param.b64 	%rd15, [_sk06_mlp_down_kernel_param_2];
	ld.param.b64 	%rd14, [_sk06_mlp_down_kernel_param_1];
	ld.param.b64 	%rd13, [_sk06_mlp_down_kernel_param_0];
$L__tmp0:
	.loc	1 313 24                        // sk06_mlp_down.py:313:24
	mov.u32 	%r38, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk06_mlp_down.py:314:27 ]
	add.s32 	%r39, %r20, 15;
	.loc	2 43 30                         // standard.py:43:30 @[ sk06_mlp_down.py:314:27 ]
	shr.s32 	%r40, %r39, 31;
	shr.u32 	%r41, %r40, 28;
	add.s32 	%r42, %r39, %r41;
	shr.s32 	%r43, %r42, 4;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk06_mlp_down.py:315:27 ]
	add.s32 	%r44, %r21, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk06_mlp_down.py:315:27 ]
	shr.s32 	%r45, %r44, 31;
	shr.u32 	%r46, %r45, 25;
	add.s32 	%r47, %r44, %r46;
	shr.s32 	%r48, %r47, 7;
$L__tmp3:
	.loc	1 316 29                        // sk06_mlp_down.py:316:29
	shl.b32 	%r49, %r48, 3;
	.loc	1 317 22                        // sk06_mlp_down.py:317:22
	div.s32 	%r50, %r38, %r49;
	.loc	1 317 38                        // sk06_mlp_down.py:317:38
	shl.b32 	%r51, %r50, 3;
	ld.param.b32 	%r52, [_sk06_mlp_down_kernel_param_9];
	.loc	1 318 30                        // sk06_mlp_down.py:318:30
	sub.s32 	%r53, %r43, %r51;
	ld.param.b32 	%r54, [_sk06_mlp_down_kernel_param_10];
	.loc	1 318 39                        // sk06_mlp_down.py:318:39
	min.s32 	%r55, %r53, 8;
	ld.param.b32 	%r56, [_sk06_mlp_down_kernel_param_11];
	.loc	1 319 30                        // sk06_mlp_down.py:319:30
	mul.lo.s32 	%r57, %r50, %r49;
	sub.s32 	%r58, %r38, %r57;
	.loc	1 320 36                        // sk06_mlp_down.py:320:36
	div.s32 	%r59, %r58, %r55;
	.loc	1 319 46                        // sk06_mlp_down.py:319:46
	mul.lo.s32 	%r60, %r59, %r55;
	sub.s32 	%r61, %r58, %r60;
	.loc	1 319 23                        // sk06_mlp_down.py:319:23
	add.s32 	%r62, %r61, %r51;
	.loc	1 322 22                        // sk06_mlp_down.py:322:22
	shl.b32 	%r1, %r62, 4;
	.loc	1 322 45                        // sk06_mlp_down.py:322:45
	mov.u32 	%r2, %tid.x;
	and.b32 	%r3, %r2, 240;
	bfe.u32 	%r63, %r2, 4, 4;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r4, %r1, %r63;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r5, %r4, %r20;
	.loc	1 323 22                        // sk06_mlp_down.py:323:22
	shl.b32 	%r6, %r59, 7;
	.loc	1 323 45                        // sk06_mlp_down.py:323:45
	shr.u32 	%r64, %r2, 3;
	bfe.u32 	%r65, %r2, 3, 5;
	and.b32 	%r7, %r2, 224;
	and.b32 	%r8, %r2, 15;
	shl.b32 	%r66, %r8, 3;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r67, %r6, %r65;
	or.b32 	%r68, %r67, 32;
	or.b32 	%r69, %r67, 64;
	or.b32 	%r70, %r64, %r6;
	or.b32 	%r71, %r70, 96;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r72, %r67, %r21;
	rem.s32 	%r73, %r68, %r21;
	rem.s32 	%r74, %r69, %r21;
	rem.s32 	%r75, %r71, %r21;
	.loc	1 326 39                        // sk06_mlp_down.py:326:39
	mul.lo.s32 	%r76, %r5, %r54;
	.loc	1 326 21                        // sk06_mlp_down.py:326:21
	cvt.s64.s32 	%rd1, %r76;
	add.s64 	%rd29, %rd13, %rd1;
	.loc	1 326 51                        // sk06_mlp_down.py:326:51
	cvt.u64.u32 	%rd2, %r66;
	add.s64 	%rd19, %rd29, %rd2;
	.loc	1 327 28                        // sk06_mlp_down.py:327:28
	and.b32 	%r9, %r2, 7;
	shl.b32 	%r77, %r9, 4;
	.loc	1 327 21                        // sk06_mlp_down.py:327:21
	cvt.u64.u32 	%rd3, %r77;
	add.s64 	%rd30, %rd14, %rd3;
	.loc	1 327 69                        // sk06_mlp_down.py:327:69
	mul.lo.s32 	%r78, %r72, %r56;
	mul.lo.s32 	%r79, %r73, %r56;
	mul.lo.s32 	%r80, %r74, %r56;
	mul.lo.s32 	%r81, %r75, %r56;
	.loc	1 327 51                        // sk06_mlp_down.py:327:51
	cvt.s64.s32 	%rd4, %r78;
	add.s64 	%rd20, %rd30, %rd4;
	cvt.s64.s32 	%rd5, %r79;
	add.s64 	%rd21, %rd30, %rd5;
	cvt.s64.s32 	%rd6, %r80;
	add.s64 	%rd22, %rd30, %rd6;
	cvt.s64.s32 	%rd7, %r81;
	add.s64 	%rd23, %rd30, %rd7;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	setp.lt.s32 	%p1, %r52, 128;
	setp.gt.s32 	%p2, %r52, 127;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	and.b32 	%r85, %r2, 255;
	shl.b32 	%r86, %r85, 3;
	and.b32 	%r87, %r2, 112;
	xor.b32 	%r88, %r86, %r87;
	mov.b32 	%r89, global_smem;
	add.s32 	%r11, %r89, %r88;
	add.s32 	%r24, %r11, 32768;
	selp.b32 	%r25, 8, 0, %p2;
	// begin inline asm
	cp.async.ca.shared.global [ %r24 + 0 ], [ %rd19 + 0 ], 0x8, %r25;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	shl.b32 	%r90, %r85, 4;
	shl.b32 	%r12, %r2, 1;
	and.b32 	%r91, %r12, 112;
	xor.b32 	%r92, %r90, %r91;
	add.s32 	%r26, %r89, %r92;
	selp.b32 	%r27, 16, 0, %p2;
	// begin inline asm
	cp.async.cg.shared.global [ %r26 + 0 ], [ %rd20 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r28, %r26, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r28 + 0 ], [ %rd21 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r29, %r26, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r29 + 0 ], [ %rd22 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r30, %r26, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r30 + 0 ], [ %rd23 + 0 ], 0x10, %r27;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	setp.gt.s32 	%p3, %r52, 255;
	.loc	1 344 18                        // sk06_mlp_down.py:344:18
	add.s64 	%rd24, %rd19, 128;
	.loc	1 345 18                        // sk06_mlp_down.py:345:18
	add.s64 	%rd25, %rd20, 128;
	add.s64 	%rd26, %rd21, 128;
	add.s64 	%rd27, %rd22, 128;
	add.s64 	%rd28, %rd23, 128;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	bar.sync 	0;
	add.s32 	%r31, %r11, 34816;
	selp.b32 	%r32, 8, 0, %p3;
	// begin inline asm
	cp.async.ca.shared.global [ %r31 + 0 ], [ %rd24 + 0 ], 0x8, %r32;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	add.s32 	%r33, %r26, 16384;
	selp.b32 	%r34, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r33 + 0 ], [ %rd25 + 0 ], 0x10, %r34;
	// end inline asm
	add.s32 	%r35, %r26, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r35 + 0 ], [ %rd26 + 0 ], 0x10, %r34;
	// end inline asm
	add.s32 	%r36, %r26, 24576;
	// begin inline asm
	cp.async.cg.shared.global [ %r36 + 0 ], [ %rd27 + 0 ], 0x10, %r34;
	// end inline asm
	add.s32 	%r37, %r26, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r37 + 0 ], [ %rd28 + 0 ], 0x10, %r34;
	// end inline asm
	cp.async.commit_group;
	mov.b32 	%r279, 0f00000000;
	cvt.u32.u64 	%r275, %rd3;
	mov.b32 	%r280, %r279;
	mov.b32 	%r281, %r279;
	mov.b32 	%r282, %r279;
	mov.b32 	%r283, %r279;
	mov.b32 	%r284, %r279;
	mov.b32 	%r285, %r279;
	mov.b32 	%r286, %r279;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	@%p1 bra 	$L__BB0_3;
// %bb.1:                               // %.lr.ph
	.loc	1 0 23                          // sk06_mlp_down.py:0:23
	shr.s32 	%r82, %r52, 31;
	shr.u32 	%r83, %r82, 25;
	add.s32 	%r84, %r52, %r83;
	shr.s32 	%r10, %r84, 7;
	add.s32 	%r13, %r10, -2;
	shl.b32 	%r93, %r8, 7;
	and.b32 	%r94, %r2, 16;
	xor.b32 	%r95, %r275, %r94;
	or.b32 	%r14, %r95, %r93;
	xor.b32 	%r15, %r14, 32;
	xor.b32 	%r16, %r14, 64;
	xor.b32 	%r17, %r14, 96;
	shl.b32 	%r96, %r9, 7;
	shl.b32 	%r97, %r7, 5;
	and.b32 	%r98, %r12, 48;
	or.b32 	%r99, %r96, %r97;
	xor.b32 	%r100, %r275, %r98;
	or.b32 	%r18, %r99, %r100;
	xor.b32 	%r19, %r18, 64;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	add.s64 	%rd31, %rd3, %rd7;
	add.s64 	%rd32, %rd31, %rd14;
	add.s64 	%rd8, %rd32, 256;
	add.s64 	%rd33, %rd3, %rd6;
	add.s64 	%rd34, %rd33, %rd14;
	add.s64 	%rd9, %rd34, 256;
	add.s64 	%rd35, %rd3, %rd5;
	add.s64 	%rd36, %rd35, %rd14;
	add.s64 	%rd10, %rd36, 256;
	add.s64 	%rd37, %rd3, %rd4;
	add.s64 	%rd38, %rd37, %rd14;
	add.s64 	%rd11, %rd38, 256;
	add.s64 	%rd39, %rd2, %rd1;
	add.s64 	%rd40, %rd39, %rd13;
	add.s64 	%rd12, %rd40, 256;
	mov.b32 	%r279, 0f00000000;
	mov.b32 	%r101, 0;
	mov.b32 	%r277, 1;
	mov.b32 	%r276, -1;
	mov.b64 	%rd54, 0;
	mov.b32 	%r278, %r101;
	mov.b32 	%r280, %r279;
	mov.b32 	%r281, %r279;
	mov.b32 	%r282, %r279;
	mov.b32 	%r283, %r279;
	mov.b32 	%r284, %r279;
	mov.b32 	%r285, %r279;
	mov.b32 	%r286, %r279;
$L__BB0_2:                              // =>This Inner Loop Header: Depth=1
	setp.lt.s32 	%p4, %r278, %r13;
	add.s32 	%r149, %r276, 1;
	setp.gt.s32 	%p5, %r149, 1;
	selp.b32 	%r276, 0, %r149, %p5;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	cp.async.wait_group 	2;
	bar.sync 	0;
	shl.b32 	%r150, %r276, 11;
	add.s32 	%r151, %r89, %r150;
	add.s32 	%r152, %r151, %r14;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r102, %r103, %r104, %r105}, [%r152+32768];
	add.s32 	%r153, %r151, %r15;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r114, %r115, %r116, %r117}, [%r153+32768];
	add.s32 	%r154, %r151, %r16;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r126, %r127, %r128, %r129}, [%r154+32768];
	add.s32 	%r155, %r151, %r17;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r134, %r135, %r136, %r137}, [%r155+32768];
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	shl.b32 	%r156, %r276, 14;
	add.s32 	%r157, %r89, %r156;
	add.s32 	%r158, %r157, %r18;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r106, %r107, %r118, %r119}, [%r158];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r108, %r109, %r124, %r125}, [%r158+8192];
	add.s32 	%r159, %r157, %r19;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r130, %r131, %r138, %r139}, [%r159];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r132, %r133, %r140, %r141}, [%r159+8192];
	.loc	1 338 36                        // sk06_mlp_down.py:338:36
	mov.b32 	%r110, %r101;
	mov.b32 	%r111, %r101;
	mov.b32 	%r112, %r101;
	mov.b32 	%r113, %r101;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r110, %r111, %r112, %r113 }, { %r102, %r103, %r104, %r105 }, { %r106, %r107 }, { %r110, %r111, %r112, %r113 };
	// end inline asm
	mov.b32 	%r120, %r101;
	mov.b32 	%r121, %r101;
	mov.b32 	%r122, %r101;
	mov.b32 	%r123, %r101;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r120, %r121, %r122, %r123 }, { %r102, %r103, %r104, %r105 }, { %r108, %r109 }, { %r120, %r121, %r122, %r123 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r110, %r111, %r112, %r113 }, { %r114, %r115, %r116, %r117 }, { %r118, %r119 }, { %r110, %r111, %r112, %r113 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r120, %r121, %r122, %r123 }, { %r114, %r115, %r116, %r117 }, { %r124, %r125 }, { %r120, %r121, %r122, %r123 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r110, %r111, %r112, %r113 }, { %r126, %r127, %r128, %r129 }, { %r130, %r131 }, { %r110, %r111, %r112, %r113 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r120, %r121, %r122, %r123 }, { %r126, %r127, %r128, %r129 }, { %r132, %r133 }, { %r120, %r121, %r122, %r123 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r110, %r111, %r112, %r113 }, { %r134, %r135, %r136, %r137 }, { %r138, %r139 }, { %r110, %r111, %r112, %r113 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r120, %r121, %r122, %r123 }, { %r134, %r135, %r136, %r137 }, { %r140, %r141 }, { %r120, %r121, %r122, %r123 };
	// end inline asm
	.loc	1 343 19                        // sk06_mlp_down.py:343:19
	cvt.rn.f32.s32 	%r160, %r120;
	cvt.rn.f32.s32 	%r161, %r121;
	cvt.rn.f32.s32 	%r162, %r122;
	cvt.rn.f32.s32 	%r163, %r123;
	cvt.rn.f32.s32 	%r164, %r110;
	cvt.rn.f32.s32 	%r165, %r111;
	cvt.rn.f32.s32 	%r166, %r112;
	cvt.rn.f32.s32 	%r167, %r113;
	add.f32 	%r282, %r282, %r167;
	add.f32 	%r281, %r281, %r166;
	add.f32 	%r280, %r280, %r165;
	add.f32 	%r279, %r279, %r164;
	add.f32 	%r286, %r286, %r163;
	add.f32 	%r285, %r285, %r162;
	add.f32 	%r284, %r284, %r161;
	add.f32 	%r283, %r283, %r160;
	.loc	1 345 18                        // sk06_mlp_down.py:345:18
	add.s64 	%rd41, %rd12, %rd54;
	add.s64 	%rd42, %rd11, %rd54;
	add.s64 	%rd43, %rd10, %rd54;
	add.s64 	%rd44, %rd9, %rd54;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	add.s64 	%rd45, %rd8, %rd54;
	add.s32 	%r168, %r277, 1;
	setp.gt.s32 	%p6, %r168, 1;
	selp.b32 	%r277, 0, %r168, %p6;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	shl.b32 	%r169, %r277, 11;
	bar.sync 	0;
	add.s32 	%r170, %r11, %r169;
	add.s32 	%r142, %r170, 32768;
	selp.b32 	%r143, 8, 0, %p4;
	// begin inline asm
	cp.async.ca.shared.global [ %r142 + 0 ], [ %rd41 + 0 ], 0x8, %r143;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	shl.b32 	%r171, %r277, 14;
	add.s32 	%r144, %r26, %r171;
	selp.b32 	%r145, 16, 0, %p4;
	// begin inline asm
	cp.async.cg.shared.global [ %r144 + 0 ], [ %rd42 + 0 ], 0x10, %r145;
	// end inline asm
	add.s32 	%r146, %r144, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r146 + 0 ], [ %rd43 + 0 ], 0x10, %r145;
	// end inline asm
	add.s32 	%r147, %r144, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r147 + 0 ], [ %rd44 + 0 ], 0x10, %r145;
	// end inline asm
	add.s32 	%r148, %r144, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r148 + 0 ], [ %rd45 + 0 ], 0x10, %r145;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	add.s32 	%r278, %r278, 1;
	add.s64 	%rd54, %rd54, 128;
	setp.ne.b32 	%p7, %r10, %r278;
	@%p7 bra 	$L__BB0_2;
$L__BB0_3:                              // %._crit_edge
	.loc	1 0 23                          // sk06_mlp_down.py:0:23
	cvt.u32.u64 	%r191, %rd2;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r192, %r6, %r191;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r193, %r192, %r21;
	.loc	1 323 45                        // sk06_mlp_down.py:323:45
	and.b32 	%r194, %r12, 6;
	shr.u32 	%r195, %r7, 2;
	or.b32 	%r196, %r194, %r195;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r197, %r196, %r6;
	or.b32 	%r198, %r197, 64;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r199, %r198, %r21;
	rem.s32 	%r200, %r197, %r21;
	.loc	1 322 45                        // sk06_mlp_down.py:322:45
	and.b32 	%r201, %r2, 28;
	bfe.u32 	%r202, %r2, 2, 3;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r203, %r202, %r1;
	or.b32 	%r204, %r203, 8;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r205, %r204, %r20;
	rem.s32 	%r206, %r203, %r20;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 347 38                        // sk06_mlp_down.py:347:38
	mad.wide.s32 	%rd46, %r206, 4, %rd17;
	mad.wide.s32 	%rd47, %r205, 4, %rd17;
	.loc	1 347 24                        // sk06_mlp_down.py:347:24
	// begin inline asm
	mov.u32 %r172, 0x0;
	ld.global.b32 { %r172 }, [ %rd46 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r173, 0x0;
	ld.global.b32 { %r173 }, [ %rd47 + 0 ];
	// end inline asm
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r207, %r279, %r172;
	mul.f32 	%r208, %r280, %r172;
	mul.f32 	%r209, %r281, %r173;
	mul.f32 	%r210, %r282, %r173;
	mul.f32 	%r211, %r283, %r172;
	mul.f32 	%r212, %r284, %r172;
	mul.f32 	%r213, %r285, %r173;
	mul.f32 	%r214, %r286, %r173;
	.loc	1 349 42                        // sk06_mlp_down.py:349:42
	mad.wide.s32 	%rd48, %r200, 4, %rd18;
	mad.wide.s32 	%rd49, %r199, 4, %rd18;
	.loc	1 349 28                        // sk06_mlp_down.py:349:28
	// begin inline asm
	mov.u32 %r174, 0x0;
	mov.u32 %r175, 0x0;
	ld.global.v2.b32 { %r174, %r175 }, [ %rd48 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r176, 0x0;
	mov.u32 %r177, 0x0;
	ld.global.v2.b32 { %r176, %r177 }, [ %rd49 + 0 ];
	// end inline asm
	.loc	1 350 49                        // sk06_mlp_down.py:350:49
	mul.lo.s32 	%r215, %r5, %r23;
	.loc	1 350 31                        // sk06_mlp_down.py:350:31
	mad.wide.s32 	%rd52, %r215, 2, %rd16;
	.loc	1 350 64                        // sk06_mlp_down.py:350:64
	mad.wide.s32 	%rd50, %r193, 2, %rd52;
	.loc	1 350 19                        // sk06_mlp_down.py:350:19
	// begin inline asm
	mov.u32 %r179, 0x0;
	mov.u32 %r180, 0x0;
	mov.u32 %r181, 0x0;
	mov.u32 %r182, 0x0;
	ld.global.v4.b32 { %r179, %r180, %r181, %r182 }, [ %rd50 + 0 ];
	// end inline asm
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	and.b32 	%r216, %r2, 120;
	shl.b32 	%r217, %r216, 5;
	or.b32 	%r218, %r217, %r275;
	xor.b32 	%r219, %r218, %r3;
	add.s32 	%r178, %r89, %r219;
	// begin inline asm
	st.shared.v4.b32 [ %r178 + 0 ], { %r179, %r180, %r181, %r182 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r220, %r9, 9;
	shl.b32 	%r221, %r2, 4;
	and.b32 	%r222, %r221, 496;
	shr.u32 	%r223, %r7, 1;
	xor.b32 	%r224, %r222, %r223;
	add.s32 	%r225, %r89, %r220;
	add.s32 	%r226, %r225, %r224;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r227, %r228, %r229, %r230}, [%r226];
	mov.b32 	{%rs9, %rs10}, %r227;
	mov.b32 	{%rs11, %rs12}, %r228;
	mov.b32 	{%rs13, %rs14}, %r229;
	mov.b32 	{%rs15, %rs16}, %r230;
	cvt.f32.bf16 	%r231, %rs9;
	cvt.f32.bf16 	%r232, %rs10;
	cvt.f32.bf16 	%r233, %rs11;
	cvt.f32.bf16 	%r234, %rs12;
	cvt.f32.bf16 	%r235, %rs13;
	cvt.f32.bf16 	%r236, %rs14;
	cvt.f32.bf16 	%r237, %rs15;
	cvt.f32.bf16 	%r238, %rs16;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r239, %r207, %r174, %r231;
	fma.rn.f32 	%r240, %r208, %r175, %r232;
	fma.rn.f32 	%r241, %r209, %r174, %r233;
	fma.rn.f32 	%r242, %r210, %r175, %r234;
	fma.rn.f32 	%r243, %r211, %r176, %r235;
	fma.rn.f32 	%r244, %r212, %r177, %r236;
	fma.rn.f32 	%r245, %r213, %r176, %r237;
	fma.rn.f32 	%r246, %r214, %r177, %r238;
	.loc	1 357 31                        // sk06_mlp_down.py:357:31
	setp.lt.s32 	%p9, %r4, %r20;
	.loc	1 357 54                        // sk06_mlp_down.py:357:54
	setp.lt.s32 	%p10, %r192, %r21;
	.loc	1 357 37                        // sk06_mlp_down.py:357:37
	and.pred 	%p8, %p9, %p10;
	.loc	1 355 35                        // sk06_mlp_down.py:355:35
	mul.lo.s32 	%r247, %r4, %r22;
	.loc	1 355 18                        // sk06_mlp_down.py:355:18
	mad.wide.s32 	%rd53, %r247, 2, %rd15;
	.loc	1 355 50                        // sk06_mlp_down.py:355:50
	mad.wide.s32 	%rd51, %r192, 2, %rd53;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16.f32 	%rs1, %r239;
	cvt.rn.bf16.f32 	%rs2, %r240;
	cvt.rn.bf16.f32 	%rs3, %r241;
	cvt.rn.bf16.f32 	%rs4, %r242;
	cvt.rn.bf16.f32 	%rs5, %r243;
	cvt.rn.bf16.f32 	%rs6, %r244;
	cvt.rn.bf16.f32 	%rs7, %r245;
	cvt.rn.bf16.f32 	%rs8, %r246;
	bar.sync 	0;
	shl.b32 	%r248, %r2, 5;
	and.b32 	%r249, %r248, 768;
	shl.b32 	%r250, %r201, 1;
	and.b32 	%r251, %r2, 1;
	neg.s32 	%r252, %r251;
	and.b32 	%r253, %r252, 1088;
	bfe.s32 	%r254, %r2, 1, 1;
	and.b32 	%r255, %r254, 2052;
	or.b32 	%r256, %r249, %r250;
	or.b32 	%r257, %r253, %r256;
	xor.b32 	%r258, %r257, %r223;
	or.b32 	%r259, %r258, %r255;
	add.s32 	%r183, %r89, %r259;
	// begin inline asm
	st.shared.v2.b16 [ %r183 + 0 ], { %rs1, %rs2 };
	// end inline asm
	add.s32 	%r184, %r183, 128;
	// begin inline asm
	st.shared.v2.b16 [ %r184 + 0 ], { %rs3, %rs4 };
	// end inline asm
	xor.b32 	%r260, %r259, 4;
	add.s32 	%r185, %r89, %r260;
	// begin inline asm
	st.shared.v2.b16 [ %r185 + 0 ], { %rs5, %rs6 };
	// end inline asm
	add.s32 	%r186, %r185, 128;
	// begin inline asm
	st.shared.v2.b16 [ %r186 + 0 ], { %rs7, %rs8 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r261, %r2, 3;
	and.b32 	%r262, %r261, 768;
	shr.u32 	%r263, %r216, 1;
	and.b32 	%r264, %r2, 128;
	or.b32 	%r265, %r275, %r262;
	xor.b32 	%r266, %r265, %r263;
	or.b32 	%r267, %r266, %r264;
	add.s32 	%r268, %r89, %r267;
	ld.shared.b32 	%r187, [%r268];
	xor.b32 	%r269, %r267, 64;
	add.s32 	%r270, %r89, %r269;
	ld.shared.b32 	%r188, [%r270+1024];
	xor.b32 	%r271, %r267, 4;
	add.s32 	%r272, %r89, %r271;
	ld.shared.b32 	%r189, [%r272+2048];
	xor.b32 	%r273, %r267, 68;
	add.s32 	%r274, %r89, %r273;
	ld.shared.b32 	%r190, [%r274+3072];
	.loc	1 356 8                         // sk06_mlp_down.py:356:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd51 + 0 ], { %r187, %r188, %r189, %r190 };
	// end inline asm
	.loc	1 354 4                         // sk06_mlp_down.py:354:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/workspace/vllm/_genesis/kernels/sk06_mlp_down.py"
	.file	2 "/usr/local/lib/python3.12/dist-packages/triton/language/standard.py"
	.section	.debug_abbrev
	{
.b8 1                                   // Abbreviation Code
.b8 17                                  // DW_TAG_compile_unit
.b8 1                                   // DW_CHILDREN_yes
.b8 37                                  // DW_AT_producer
.b8 8                                   // DW_FORM_string
.b8 19                                  // DW_AT_language
.b8 5                                   // DW_FORM_data2
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 16                                  // DW_AT_stmt_list
.b8 6                                   // DW_FORM_data4
.b8 27                                  // DW_AT_comp_dir
.b8 8                                   // DW_FORM_string
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 2                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 0                                   // DW_CHILDREN_no
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 32                                  // DW_AT_inline
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 3                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 1                                   // DW_CHILDREN_yes
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 4                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 0                                   // DW_CHILDREN_no
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 5                                   // DW_FORM_data2
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 168                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0xa1 DW_TAG_compile_unit
.b8 116                                 // DW_AT_producer
.b8 114
.b8 105
.b8 116
.b8 111
.b8 110
.b8 0
.b8 2                                   // DW_AT_language
.b8 0
.b8 115                                 // DW_AT_name
.b8 107
.b8 48
.b8 54
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 100
.b8 111
.b8 119
.b8 110
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 119
.b8 111
.b8 114
.b8 107
.b8 115
.b8 112
.b8 97
.b8 99
.b8 101
.b8 47
.b8 118
.b8 108
.b8 108
.b8 109
.b8 47
.b8 95
.b8 103
.b8 101
.b8 110
.b8 101
.b8 115
.b8 105
.b8 115
.b8 47
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 115
.b8 0
.b8 2                                   // Abbrev [2] 0x4b:0x18 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 54
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 100
.b8 111
.b8 119
.b8 110
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x63:0x48 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 75                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x78:0x19 DW_TAG_inlined_subroutine
.b32 75                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 58                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x91:0x19 DW_TAG_inlined_subroutine
.b32 75                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 59                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_0 = _Nativo(
    "sk06_mlp_down/tile16x128x128_shift0_abi15",
    _PTX_0, "_sk06_mlp_down_kernel",
    warps=8, shared=36864,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 18],
    horneado={11: 1, 12: 1, 15: 1, 17: 1, 19: 1, 20: 16, 21: 128, 22: 128, 23: 8, 24: 128, 25: False},
    div16=[8, 9, 10, 13, 14, 16],
)

_PTX_1 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk06_mlp_down_kernel   // -- Begin function _sk06_mlp_down_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk06_mlp_down_kernel
.visible .entry _sk06_mlp_down_kernel(
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_6,
	.param .u32 _sk06_mlp_down_kernel_param_7,
	.param .u32 _sk06_mlp_down_kernel_param_8,
	.param .u32 _sk06_mlp_down_kernel_param_9,
	.param .u32 _sk06_mlp_down_kernel_param_10,
	.param .u32 _sk06_mlp_down_kernel_param_11,
	.param .u32 _sk06_mlp_down_kernel_param_12,
	.param .u32 _sk06_mlp_down_kernel_param_13,
	.param .u32 _sk06_mlp_down_kernel_param_14,
	.param .u32 _sk06_mlp_down_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_16,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_17
)
.reqntid 256
{
	.reg .pred 	%p<11>;
	.reg .b16 	%rs<25>;
	.reg .b32 	%r<310>;
	.reg .b64 	%rd<62>;
	.loc	1 304 0                         // sk06_mlp_down.py:304:0
$L__func_begin0:
	.loc	1 304 0                         // sk06_mlp_down.py:304:0

// %bb.0:
	ld.param.b32 	%r24, [_sk06_mlp_down_kernel_param_14];
	ld.param.b32 	%r23, [_sk06_mlp_down_kernel_param_13];
	ld.param.b32 	%r22, [_sk06_mlp_down_kernel_param_12];
	ld.param.b32 	%r21, [_sk06_mlp_down_kernel_param_8];
	ld.param.b32 	%r20, [_sk06_mlp_down_kernel_param_7];
	ld.param.b64 	%rd18, [_sk06_mlp_down_kernel_param_5];
	ld.param.b64 	%rd17, [_sk06_mlp_down_kernel_param_4];
	ld.param.b64 	%rd16, [_sk06_mlp_down_kernel_param_3];
	ld.param.b64 	%rd15, [_sk06_mlp_down_kernel_param_2];
	ld.param.b64 	%rd14, [_sk06_mlp_down_kernel_param_1];
	ld.param.b64 	%rd13, [_sk06_mlp_down_kernel_param_0];
$L__tmp0:
	.loc	1 313 24                        // sk06_mlp_down.py:313:24
	mov.u32 	%r39, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk06_mlp_down.py:314:27 ]
	add.s32 	%r40, %r20, 15;
	.loc	2 43 30                         // standard.py:43:30 @[ sk06_mlp_down.py:314:27 ]
	shr.s32 	%r41, %r40, 31;
	shr.u32 	%r42, %r41, 28;
	add.s32 	%r43, %r40, %r42;
	shr.s32 	%r44, %r43, 4;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk06_mlp_down.py:315:27 ]
	add.s32 	%r45, %r21, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk06_mlp_down.py:315:27 ]
	shr.s32 	%r46, %r45, 31;
	shr.u32 	%r47, %r46, 25;
	add.s32 	%r48, %r45, %r47;
	shr.s32 	%r49, %r48, 7;
$L__tmp3:
	.loc	1 316 29                        // sk06_mlp_down.py:316:29
	shl.b32 	%r50, %r49, 3;
	.loc	1 317 22                        // sk06_mlp_down.py:317:22
	div.s32 	%r51, %r39, %r50;
	.loc	1 317 38                        // sk06_mlp_down.py:317:38
	shl.b32 	%r52, %r51, 3;
	ld.param.b32 	%r53, [_sk06_mlp_down_kernel_param_9];
	.loc	1 318 30                        // sk06_mlp_down.py:318:30
	sub.s32 	%r54, %r44, %r52;
	ld.param.b32 	%r55, [_sk06_mlp_down_kernel_param_10];
	.loc	1 318 39                        // sk06_mlp_down.py:318:39
	min.s32 	%r56, %r54, 8;
	ld.param.b32 	%r57, [_sk06_mlp_down_kernel_param_11];
	.loc	1 319 30                        // sk06_mlp_down.py:319:30
	mul.lo.s32 	%r58, %r51, %r50;
	sub.s32 	%r59, %r39, %r58;
	.loc	1 320 36                        // sk06_mlp_down.py:320:36
	div.s32 	%r60, %r59, %r56;
	.loc	1 319 46                        // sk06_mlp_down.py:319:46
	mul.lo.s32 	%r61, %r60, %r56;
	sub.s32 	%r62, %r59, %r61;
	.loc	1 319 23                        // sk06_mlp_down.py:319:23
	add.s32 	%r63, %r62, %r52;
	.loc	1 322 22                        // sk06_mlp_down.py:322:22
	shl.b32 	%r1, %r63, 4;
	.loc	1 322 45                        // sk06_mlp_down.py:322:45
	mov.u32 	%r2, %tid.x;
	and.b32 	%r3, %r2, 240;
	bfe.u32 	%r64, %r2, 4, 4;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r4, %r1, %r64;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r5, %r4, %r20;
	.loc	1 323 22                        // sk06_mlp_down.py:323:22
	shl.b32 	%r6, %r60, 7;
	.loc	1 323 45                        // sk06_mlp_down.py:323:45
	shr.u32 	%r65, %r2, 3;
	bfe.u32 	%r66, %r2, 3, 5;
	and.b32 	%r7, %r2, 224;
	and.b32 	%r8, %r2, 15;
	shl.b32 	%r67, %r8, 3;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r68, %r6, %r66;
	or.b32 	%r69, %r68, 32;
	or.b32 	%r70, %r68, 64;
	or.b32 	%r71, %r65, %r6;
	or.b32 	%r72, %r71, 96;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r73, %r68, %r21;
	rem.s32 	%r74, %r69, %r21;
	rem.s32 	%r75, %r70, %r21;
	rem.s32 	%r76, %r72, %r21;
	.loc	1 326 39                        // sk06_mlp_down.py:326:39
	mul.lo.s32 	%r77, %r5, %r55;
	.loc	1 326 21                        // sk06_mlp_down.py:326:21
	cvt.s64.s32 	%rd1, %r77;
	add.s64 	%rd29, %rd13, %rd1;
	.loc	1 326 51                        // sk06_mlp_down.py:326:51
	cvt.u64.u32 	%rd2, %r67;
	add.s64 	%rd19, %rd29, %rd2;
	.loc	1 327 28                        // sk06_mlp_down.py:327:28
	and.b32 	%r9, %r2, 7;
	shl.b32 	%r78, %r9, 4;
	.loc	1 327 21                        // sk06_mlp_down.py:327:21
	cvt.u64.u32 	%rd3, %r78;
	add.s64 	%rd30, %rd14, %rd3;
	.loc	1 327 69                        // sk06_mlp_down.py:327:69
	mul.lo.s32 	%r79, %r73, %r57;
	mul.lo.s32 	%r80, %r74, %r57;
	mul.lo.s32 	%r81, %r75, %r57;
	mul.lo.s32 	%r82, %r76, %r57;
	.loc	1 327 51                        // sk06_mlp_down.py:327:51
	cvt.s64.s32 	%rd4, %r79;
	add.s64 	%rd20, %rd30, %rd4;
	cvt.s64.s32 	%rd5, %r80;
	add.s64 	%rd21, %rd30, %rd5;
	cvt.s64.s32 	%rd6, %r81;
	add.s64 	%rd22, %rd30, %rd6;
	cvt.s64.s32 	%rd7, %r82;
	add.s64 	%rd23, %rd30, %rd7;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	setp.lt.s32 	%p1, %r53, 128;
	setp.gt.s32 	%p2, %r53, 127;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	and.b32 	%r86, %r2, 255;
	shl.b32 	%r87, %r86, 3;
	and.b32 	%r88, %r2, 112;
	xor.b32 	%r89, %r87, %r88;
	mov.b32 	%r90, global_smem;
	add.s32 	%r11, %r90, %r89;
	add.s32 	%r25, %r11, 32768;
	selp.b32 	%r26, 8, 0, %p2;
	// begin inline asm
	cp.async.ca.shared.global [ %r25 + 0 ], [ %rd19 + 0 ], 0x8, %r26;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	shl.b32 	%r91, %r86, 4;
	shl.b32 	%r12, %r2, 1;
	and.b32 	%r92, %r12, 112;
	xor.b32 	%r93, %r91, %r92;
	add.s32 	%r27, %r90, %r93;
	selp.b32 	%r28, 16, 0, %p2;
	// begin inline asm
	cp.async.cg.shared.global [ %r27 + 0 ], [ %rd20 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r29, %r27, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r29 + 0 ], [ %rd21 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r30, %r27, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r30 + 0 ], [ %rd22 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r31, %r27, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r31 + 0 ], [ %rd23 + 0 ], 0x10, %r28;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	setp.gt.s32 	%p3, %r53, 255;
	.loc	1 344 18                        // sk06_mlp_down.py:344:18
	add.s64 	%rd24, %rd19, 128;
	.loc	1 345 18                        // sk06_mlp_down.py:345:18
	add.s64 	%rd25, %rd20, 128;
	add.s64 	%rd26, %rd21, 128;
	add.s64 	%rd27, %rd22, 128;
	add.s64 	%rd28, %rd23, 128;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	bar.sync 	0;
	add.s32 	%r32, %r11, 34816;
	selp.b32 	%r33, 8, 0, %p3;
	// begin inline asm
	cp.async.ca.shared.global [ %r32 + 0 ], [ %rd24 + 0 ], 0x8, %r33;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	add.s32 	%r34, %r27, 16384;
	selp.b32 	%r35, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r34 + 0 ], [ %rd25 + 0 ], 0x10, %r35;
	// end inline asm
	add.s32 	%r36, %r27, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r36 + 0 ], [ %rd26 + 0 ], 0x10, %r35;
	// end inline asm
	add.s32 	%r37, %r27, 24576;
	// begin inline asm
	cp.async.cg.shared.global [ %r37 + 0 ], [ %rd27 + 0 ], 0x10, %r35;
	// end inline asm
	add.s32 	%r38, %r27, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r38 + 0 ], [ %rd28 + 0 ], 0x10, %r35;
	// end inline asm
	cp.async.commit_group;
	mov.b32 	%r302, 0f00000000;
	cvt.u32.u64 	%r298, %rd3;
	mov.b32 	%r303, %r302;
	mov.b32 	%r304, %r302;
	mov.b32 	%r305, %r302;
	mov.b32 	%r306, %r302;
	mov.b32 	%r307, %r302;
	mov.b32 	%r308, %r302;
	mov.b32 	%r309, %r302;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	@%p1 bra 	$L__BB0_3;
// %bb.1:                               // %.lr.ph
	.loc	1 0 23                          // sk06_mlp_down.py:0:23
	shr.s32 	%r83, %r53, 31;
	shr.u32 	%r84, %r83, 25;
	add.s32 	%r85, %r53, %r84;
	shr.s32 	%r10, %r85, 7;
	add.s32 	%r13, %r10, -2;
	shl.b32 	%r94, %r8, 7;
	and.b32 	%r95, %r2, 16;
	xor.b32 	%r96, %r298, %r95;
	or.b32 	%r14, %r96, %r94;
	xor.b32 	%r15, %r14, 32;
	xor.b32 	%r16, %r14, 64;
	xor.b32 	%r17, %r14, 96;
	shl.b32 	%r97, %r9, 7;
	shl.b32 	%r98, %r7, 5;
	and.b32 	%r99, %r12, 48;
	or.b32 	%r100, %r97, %r98;
	xor.b32 	%r101, %r298, %r99;
	or.b32 	%r18, %r100, %r101;
	xor.b32 	%r19, %r18, 64;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	add.s64 	%rd31, %rd3, %rd7;
	add.s64 	%rd32, %rd31, %rd14;
	add.s64 	%rd8, %rd32, 256;
	add.s64 	%rd33, %rd3, %rd6;
	add.s64 	%rd34, %rd33, %rd14;
	add.s64 	%rd9, %rd34, 256;
	add.s64 	%rd35, %rd3, %rd5;
	add.s64 	%rd36, %rd35, %rd14;
	add.s64 	%rd10, %rd36, 256;
	add.s64 	%rd37, %rd3, %rd4;
	add.s64 	%rd38, %rd37, %rd14;
	add.s64 	%rd11, %rd38, 256;
	add.s64 	%rd39, %rd2, %rd1;
	add.s64 	%rd40, %rd39, %rd13;
	add.s64 	%rd12, %rd40, 256;
	mov.b32 	%r302, 0f00000000;
	mov.b32 	%r102, 0;
	mov.b32 	%r300, 1;
	mov.b32 	%r299, -1;
	mov.b64 	%rd61, 0;
	mov.b32 	%r301, %r102;
	mov.b32 	%r303, %r302;
	mov.b32 	%r304, %r302;
	mov.b32 	%r305, %r302;
	mov.b32 	%r306, %r302;
	mov.b32 	%r307, %r302;
	mov.b32 	%r308, %r302;
	mov.b32 	%r309, %r302;
$L__BB0_2:                              // =>This Inner Loop Header: Depth=1
	setp.lt.s32 	%p4, %r301, %r13;
	add.s32 	%r150, %r299, 1;
	setp.gt.s32 	%p5, %r150, 1;
	selp.b32 	%r299, 0, %r150, %p5;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	cp.async.wait_group 	2;
	bar.sync 	0;
	shl.b32 	%r151, %r299, 11;
	add.s32 	%r152, %r90, %r151;
	add.s32 	%r153, %r152, %r14;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r103, %r104, %r105, %r106}, [%r153+32768];
	add.s32 	%r154, %r152, %r15;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r115, %r116, %r117, %r118}, [%r154+32768];
	add.s32 	%r155, %r152, %r16;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r127, %r128, %r129, %r130}, [%r155+32768];
	add.s32 	%r156, %r152, %r17;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r135, %r136, %r137, %r138}, [%r156+32768];
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	shl.b32 	%r157, %r299, 14;
	add.s32 	%r158, %r90, %r157;
	add.s32 	%r159, %r158, %r18;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r107, %r108, %r119, %r120}, [%r159];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r109, %r110, %r125, %r126}, [%r159+8192];
	add.s32 	%r160, %r158, %r19;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r131, %r132, %r139, %r140}, [%r160];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r133, %r134, %r141, %r142}, [%r160+8192];
	.loc	1 338 36                        // sk06_mlp_down.py:338:36
	mov.b32 	%r111, %r102;
	mov.b32 	%r112, %r102;
	mov.b32 	%r113, %r102;
	mov.b32 	%r114, %r102;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r111, %r112, %r113, %r114 }, { %r103, %r104, %r105, %r106 }, { %r107, %r108 }, { %r111, %r112, %r113, %r114 };
	// end inline asm
	mov.b32 	%r121, %r102;
	mov.b32 	%r122, %r102;
	mov.b32 	%r123, %r102;
	mov.b32 	%r124, %r102;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r121, %r122, %r123, %r124 }, { %r103, %r104, %r105, %r106 }, { %r109, %r110 }, { %r121, %r122, %r123, %r124 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r111, %r112, %r113, %r114 }, { %r115, %r116, %r117, %r118 }, { %r119, %r120 }, { %r111, %r112, %r113, %r114 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r121, %r122, %r123, %r124 }, { %r115, %r116, %r117, %r118 }, { %r125, %r126 }, { %r121, %r122, %r123, %r124 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r111, %r112, %r113, %r114 }, { %r127, %r128, %r129, %r130 }, { %r131, %r132 }, { %r111, %r112, %r113, %r114 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r121, %r122, %r123, %r124 }, { %r127, %r128, %r129, %r130 }, { %r133, %r134 }, { %r121, %r122, %r123, %r124 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r111, %r112, %r113, %r114 }, { %r135, %r136, %r137, %r138 }, { %r139, %r140 }, { %r111, %r112, %r113, %r114 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r121, %r122, %r123, %r124 }, { %r135, %r136, %r137, %r138 }, { %r141, %r142 }, { %r121, %r122, %r123, %r124 };
	// end inline asm
	.loc	1 343 19                        // sk06_mlp_down.py:343:19
	cvt.rn.f32.s32 	%r161, %r121;
	cvt.rn.f32.s32 	%r162, %r122;
	cvt.rn.f32.s32 	%r163, %r123;
	cvt.rn.f32.s32 	%r164, %r124;
	cvt.rn.f32.s32 	%r165, %r111;
	cvt.rn.f32.s32 	%r166, %r112;
	cvt.rn.f32.s32 	%r167, %r113;
	cvt.rn.f32.s32 	%r168, %r114;
	add.f32 	%r305, %r305, %r168;
	add.f32 	%r304, %r304, %r167;
	add.f32 	%r303, %r303, %r166;
	add.f32 	%r302, %r302, %r165;
	add.f32 	%r309, %r309, %r164;
	add.f32 	%r308, %r308, %r163;
	add.f32 	%r307, %r307, %r162;
	add.f32 	%r306, %r306, %r161;
	.loc	1 345 18                        // sk06_mlp_down.py:345:18
	add.s64 	%rd41, %rd12, %rd61;
	add.s64 	%rd42, %rd11, %rd61;
	add.s64 	%rd43, %rd10, %rd61;
	add.s64 	%rd44, %rd9, %rd61;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	add.s64 	%rd45, %rd8, %rd61;
	add.s32 	%r169, %r300, 1;
	setp.gt.s32 	%p6, %r169, 1;
	selp.b32 	%r300, 0, %r169, %p6;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	shl.b32 	%r170, %r300, 11;
	bar.sync 	0;
	add.s32 	%r171, %r11, %r170;
	add.s32 	%r143, %r171, 32768;
	selp.b32 	%r144, 8, 0, %p4;
	// begin inline asm
	cp.async.ca.shared.global [ %r143 + 0 ], [ %rd41 + 0 ], 0x8, %r144;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	shl.b32 	%r172, %r300, 14;
	add.s32 	%r145, %r27, %r172;
	selp.b32 	%r146, 16, 0, %p4;
	// begin inline asm
	cp.async.cg.shared.global [ %r145 + 0 ], [ %rd42 + 0 ], 0x10, %r146;
	// end inline asm
	add.s32 	%r147, %r145, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r147 + 0 ], [ %rd43 + 0 ], 0x10, %r146;
	// end inline asm
	add.s32 	%r148, %r145, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r148 + 0 ], [ %rd44 + 0 ], 0x10, %r146;
	// end inline asm
	add.s32 	%r149, %r145, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r149 + 0 ], [ %rd45 + 0 ], 0x10, %r146;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	add.s32 	%r301, %r301, 1;
	add.s64 	%rd61, %rd61, 128;
	setp.ne.b32 	%p7, %r10, %r301;
	@%p7 bra 	$L__BB0_2;
$L__BB0_3:                              // %._crit_edge
	.loc	1 0 23                          // sk06_mlp_down.py:0:23
	cvt.u32.u64 	%r192, %rd2;
	.loc	1 323 45                        // sk06_mlp_down.py:323:45
	or.b32 	%r193, %r6, %r192;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r194, %r193, 7;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r195, %r194, %r21;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r196, %r193, 6;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r197, %r196, %r21;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r198, %r193, 5;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r199, %r198, %r21;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r200, %r193, 4;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r201, %r200, %r21;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r202, %r193, 3;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r203, %r202, %r21;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r204, %r193, 2;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r205, %r204, %r21;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r206, %r193, 1;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r207, %r206, %r21;
	rem.s32 	%r208, %r193, %r21;
	.loc	1 323 45                        // sk06_mlp_down.py:323:45
	and.b32 	%r209, %r12, 6;
	shr.u32 	%r210, %r7, 2;
	or.b32 	%r211, %r209, %r210;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r212, %r211, %r6;
	or.b32 	%r213, %r212, 64;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r214, %r213, %r21;
	rem.s32 	%r215, %r212, %r21;
	.loc	1 322 45                        // sk06_mlp_down.py:322:45
	and.b32 	%r216, %r2, 28;
	bfe.u32 	%r217, %r2, 2, 3;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r218, %r217, %r1;
	or.b32 	%r219, %r218, 8;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r220, %r219, %r20;
	rem.s32 	%r221, %r218, %r20;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 347 38                        // sk06_mlp_down.py:347:38
	mad.wide.s32 	%rd46, %r221, 4, %rd17;
	mad.wide.s32 	%rd47, %r220, 4, %rd17;
	.loc	1 347 24                        // sk06_mlp_down.py:347:24
	// begin inline asm
	mov.u32 %r173, 0x0;
	ld.global.b32 { %r173 }, [ %rd46 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r174, 0x0;
	ld.global.b32 { %r174 }, [ %rd47 + 0 ];
	// end inline asm
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r222, %r302, %r173;
	mul.f32 	%r223, %r303, %r173;
	mul.f32 	%r224, %r304, %r174;
	mul.f32 	%r225, %r305, %r174;
	mul.f32 	%r226, %r306, %r173;
	mul.f32 	%r227, %r307, %r173;
	mul.f32 	%r228, %r308, %r174;
	mul.f32 	%r229, %r309, %r174;
	.loc	1 349 42                        // sk06_mlp_down.py:349:42
	mad.wide.s32 	%rd48, %r215, 4, %rd18;
	mad.wide.s32 	%rd49, %r214, 4, %rd18;
	.loc	1 349 28                        // sk06_mlp_down.py:349:28
	// begin inline asm
	mov.u32 %r175, 0x0;
	mov.u32 %r176, 0x0;
	ld.global.v2.b32 { %r175, %r176 }, [ %rd48 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r177, 0x0;
	mov.u32 %r178, 0x0;
	ld.global.v2.b32 { %r177, %r178 }, [ %rd49 + 0 ];
	// end inline asm
	.loc	1 350 49                        // sk06_mlp_down.py:350:49
	mul.lo.s32 	%r230, %r5, %r23;
	.loc	1 350 31                        // sk06_mlp_down.py:350:31
	mad.wide.s32 	%rd59, %r230, 2, %rd16;
	.loc	1 350 82                        // sk06_mlp_down.py:350:82
	mul.lo.s32 	%r231, %r208, %r24;
	mul.lo.s32 	%r232, %r207, %r24;
	mul.lo.s32 	%r233, %r205, %r24;
	mul.lo.s32 	%r234, %r203, %r24;
	mul.lo.s32 	%r235, %r201, %r24;
	mul.lo.s32 	%r236, %r199, %r24;
	mul.lo.s32 	%r237, %r197, %r24;
	mul.lo.s32 	%r238, %r195, %r24;
	.loc	1 350 64                        // sk06_mlp_down.py:350:64
	mad.wide.s32 	%rd50, %r231, 2, %rd59;
	mad.wide.s32 	%rd51, %r232, 2, %rd59;
	mad.wide.s32 	%rd52, %r233, 2, %rd59;
	mad.wide.s32 	%rd53, %r234, 2, %rd59;
	mad.wide.s32 	%rd54, %r235, 2, %rd59;
	mad.wide.s32 	%rd55, %r236, 2, %rd59;
	mad.wide.s32 	%rd56, %r237, 2, %rd59;
	mad.wide.s32 	%rd57, %r238, 2, %rd59;
	.loc	1 350 19                        // sk06_mlp_down.py:350:19
	// begin inline asm
	mov.u16 %rs1, 0x0;
	ld.global.b16 { %rs1 }, [ %rd50 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs2, 0x0;
	ld.global.b16 { %rs2 }, [ %rd51 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs3, 0x0;
	ld.global.b16 { %rs3 }, [ %rd52 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs4, 0x0;
	ld.global.b16 { %rs4 }, [ %rd53 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs5, 0x0;
	ld.global.b16 { %rs5 }, [ %rd54 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs6, 0x0;
	ld.global.b16 { %rs6 }, [ %rd55 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs7, 0x0;
	ld.global.b16 { %rs7 }, [ %rd56 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs8, 0x0;
	ld.global.b16 { %rs8 }, [ %rd57 + 0 ];
	// end inline asm
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	and.b32 	%r239, %r2, 120;
	shl.b32 	%r240, %r239, 5;
	or.b32 	%r241, %r240, %r298;
	xor.b32 	%r242, %r241, %r3;
	add.s32 	%r179, %r90, %r242;
	mov.b32 	%r180, {%rs1, %rs2};
	mov.b32 	%r181, {%rs3, %rs4};
	mov.b32 	%r182, {%rs5, %rs6};
	mov.b32 	%r183, {%rs7, %rs8};
	// begin inline asm
	st.shared.v4.b32 [ %r179 + 0 ], { %r180, %r181, %r182, %r183 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r243, %r9, 9;
	shl.b32 	%r244, %r2, 4;
	and.b32 	%r245, %r244, 496;
	shr.u32 	%r246, %r7, 1;
	xor.b32 	%r247, %r245, %r246;
	add.s32 	%r248, %r90, %r243;
	add.s32 	%r249, %r248, %r247;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r250, %r251, %r252, %r253}, [%r249];
	mov.b32 	{%rs17, %rs18}, %r250;
	mov.b32 	{%rs19, %rs20}, %r251;
	mov.b32 	{%rs21, %rs22}, %r252;
	mov.b32 	{%rs23, %rs24}, %r253;
	cvt.f32.bf16 	%r254, %rs17;
	cvt.f32.bf16 	%r255, %rs18;
	cvt.f32.bf16 	%r256, %rs19;
	cvt.f32.bf16 	%r257, %rs20;
	cvt.f32.bf16 	%r258, %rs21;
	cvt.f32.bf16 	%r259, %rs22;
	cvt.f32.bf16 	%r260, %rs23;
	cvt.f32.bf16 	%r261, %rs24;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r262, %r222, %r175, %r254;
	fma.rn.f32 	%r263, %r223, %r176, %r255;
	fma.rn.f32 	%r264, %r224, %r175, %r256;
	fma.rn.f32 	%r265, %r225, %r176, %r257;
	fma.rn.f32 	%r266, %r226, %r177, %r258;
	fma.rn.f32 	%r267, %r227, %r178, %r259;
	fma.rn.f32 	%r268, %r228, %r177, %r260;
	fma.rn.f32 	%r269, %r229, %r178, %r261;
	.loc	1 357 31                        // sk06_mlp_down.py:357:31
	setp.lt.s32 	%p9, %r4, %r20;
	.loc	1 357 54                        // sk06_mlp_down.py:357:54
	setp.lt.s32 	%p10, %r193, %r21;
	.loc	1 357 37                        // sk06_mlp_down.py:357:37
	and.pred 	%p8, %p9, %p10;
	.loc	1 355 35                        // sk06_mlp_down.py:355:35
	mul.lo.s32 	%r270, %r4, %r22;
	.loc	1 355 18                        // sk06_mlp_down.py:355:18
	mad.wide.s32 	%rd60, %r270, 2, %rd15;
	.loc	1 355 50                        // sk06_mlp_down.py:355:50
	mad.wide.s32 	%rd58, %r193, 2, %rd60;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16.f32 	%rs9, %r262;
	cvt.rn.bf16.f32 	%rs10, %r263;
	cvt.rn.bf16.f32 	%rs11, %r264;
	cvt.rn.bf16.f32 	%rs12, %r265;
	cvt.rn.bf16.f32 	%rs13, %r266;
	cvt.rn.bf16.f32 	%rs14, %r267;
	cvt.rn.bf16.f32 	%rs15, %r268;
	cvt.rn.bf16.f32 	%rs16, %r269;
	bar.sync 	0;
	shl.b32 	%r271, %r2, 5;
	and.b32 	%r272, %r271, 768;
	shl.b32 	%r273, %r216, 1;
	and.b32 	%r274, %r2, 1;
	neg.s32 	%r275, %r274;
	and.b32 	%r276, %r275, 1088;
	bfe.s32 	%r277, %r2, 1, 1;
	and.b32 	%r278, %r277, 2052;
	or.b32 	%r279, %r272, %r273;
	or.b32 	%r280, %r276, %r279;
	xor.b32 	%r281, %r280, %r246;
	or.b32 	%r282, %r281, %r278;
	add.s32 	%r184, %r90, %r282;
	// begin inline asm
	st.shared.v2.b16 [ %r184 + 0 ], { %rs9, %rs10 };
	// end inline asm
	add.s32 	%r185, %r184, 128;
	// begin inline asm
	st.shared.v2.b16 [ %r185 + 0 ], { %rs11, %rs12 };
	// end inline asm
	xor.b32 	%r283, %r282, 4;
	add.s32 	%r186, %r90, %r283;
	// begin inline asm
	st.shared.v2.b16 [ %r186 + 0 ], { %rs13, %rs14 };
	// end inline asm
	add.s32 	%r187, %r186, 128;
	// begin inline asm
	st.shared.v2.b16 [ %r187 + 0 ], { %rs15, %rs16 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r284, %r2, 3;
	and.b32 	%r285, %r284, 768;
	shr.u32 	%r286, %r239, 1;
	and.b32 	%r287, %r2, 128;
	or.b32 	%r288, %r298, %r285;
	xor.b32 	%r289, %r288, %r286;
	or.b32 	%r290, %r289, %r287;
	add.s32 	%r291, %r90, %r290;
	ld.shared.b32 	%r188, [%r291];
	xor.b32 	%r292, %r290, 64;
	add.s32 	%r293, %r90, %r292;
	ld.shared.b32 	%r189, [%r293+1024];
	xor.b32 	%r294, %r290, 4;
	add.s32 	%r295, %r90, %r294;
	ld.shared.b32 	%r190, [%r295+2048];
	xor.b32 	%r296, %r290, 68;
	add.s32 	%r297, %r90, %r296;
	ld.shared.b32 	%r191, [%r297+3072];
	.loc	1 356 8                         // sk06_mlp_down.py:356:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd58 + 0 ], { %r188, %r189, %r190, %r191 };
	// end inline asm
	.loc	1 354 4                         // sk06_mlp_down.py:354:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/workspace/vllm/_genesis/kernels/sk06_mlp_down.py"
	.file	2 "/usr/local/lib/python3.12/dist-packages/triton/language/standard.py"
	.section	.debug_abbrev
	{
.b8 1                                   // Abbreviation Code
.b8 17                                  // DW_TAG_compile_unit
.b8 1                                   // DW_CHILDREN_yes
.b8 37                                  // DW_AT_producer
.b8 8                                   // DW_FORM_string
.b8 19                                  // DW_AT_language
.b8 5                                   // DW_FORM_data2
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 16                                  // DW_AT_stmt_list
.b8 6                                   // DW_FORM_data4
.b8 27                                  // DW_AT_comp_dir
.b8 8                                   // DW_FORM_string
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 2                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 0                                   // DW_CHILDREN_no
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 32                                  // DW_AT_inline
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 3                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 1                                   // DW_CHILDREN_yes
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 4                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 0                                   // DW_CHILDREN_no
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 5                                   // DW_FORM_data2
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 168                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0xa1 DW_TAG_compile_unit
.b8 116                                 // DW_AT_producer
.b8 114
.b8 105
.b8 116
.b8 111
.b8 110
.b8 0
.b8 2                                   // DW_AT_language
.b8 0
.b8 115                                 // DW_AT_name
.b8 107
.b8 48
.b8 54
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 100
.b8 111
.b8 119
.b8 110
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 119
.b8 111
.b8 114
.b8 107
.b8 115
.b8 112
.b8 97
.b8 99
.b8 101
.b8 47
.b8 118
.b8 108
.b8 108
.b8 109
.b8 47
.b8 95
.b8 103
.b8 101
.b8 110
.b8 101
.b8 115
.b8 105
.b8 115
.b8 47
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 115
.b8 0
.b8 2                                   // Abbrev [2] 0x4b:0x18 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 54
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 100
.b8 111
.b8 119
.b8 110
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x63:0x48 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 75                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x78:0x19 DW_TAG_inlined_subroutine
.b32 75                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 58                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x91:0x19 DW_TAG_inlined_subroutine
.b32 75                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 59                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_1 = _Nativo(
    "sk06_mlp_down/tile16x128x128_shift0_abi16",
    _PTX_1, "_sk06_mlp_down_kernel",
    warps=8, shared=36864,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 17, 18],
    horneado={11: 1, 12: 1, 15: 1, 19: 1, 20: 16, 21: 128, 22: 128, 23: 8, 24: 128, 25: False},
    div16=[8, 9, 10, 13, 14, 16, 17],
)

_PTX_2 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk06_mlp_down_kernel   // -- Begin function _sk06_mlp_down_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk06_mlp_down_kernel
.visible .entry _sk06_mlp_down_kernel(
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_6,
	.param .u32 _sk06_mlp_down_kernel_param_7,
	.param .u32 _sk06_mlp_down_kernel_param_8,
	.param .u32 _sk06_mlp_down_kernel_param_9,
	.param .u32 _sk06_mlp_down_kernel_param_10,
	.param .u32 _sk06_mlp_down_kernel_param_11,
	.param .u32 _sk06_mlp_down_kernel_param_12,
	.param .u32 _sk06_mlp_down_kernel_param_13,
	.param .u32 _sk06_mlp_down_kernel_param_14,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_16
)
.reqntid 256
{
	.reg .pred 	%p<11>;
	.reg .b16 	%rs<17>;
	.reg .b32 	%r<298>;
	.reg .b64 	%rd<65>;
	.loc	1 304 0                         // sk06_mlp_down.py:304:0
$L__func_begin0:
	.loc	1 304 0                         // sk06_mlp_down.py:304:0

// %bb.0:
	ld.param.b32 	%r22, [_sk06_mlp_down_kernel_param_13];
	ld.param.b32 	%r21, [_sk06_mlp_down_kernel_param_12];
	ld.param.b32 	%r20, [_sk06_mlp_down_kernel_param_9];
	ld.param.b32 	%r19, [_sk06_mlp_down_kernel_param_8];
	ld.param.b32 	%r18, [_sk06_mlp_down_kernel_param_7];
	ld.param.b64 	%rd23, [_sk06_mlp_down_kernel_param_4];
	ld.param.b64 	%rd22, [_sk06_mlp_down_kernel_param_3];
	ld.param.b64 	%rd21, [_sk06_mlp_down_kernel_param_2];
	ld.param.b64 	%rd20, [_sk06_mlp_down_kernel_param_1];
	ld.param.b64 	%rd19, [_sk06_mlp_down_kernel_param_0];
$L__tmp0:
	.loc	1 313 24                        // sk06_mlp_down.py:313:24
	mov.u32 	%r38, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk06_mlp_down.py:314:27 ]
	add.s32 	%r39, %r18, 15;
	.loc	2 43 30                         // standard.py:43:30 @[ sk06_mlp_down.py:314:27 ]
	shr.s32 	%r40, %r39, 31;
	shr.u32 	%r41, %r40, 28;
	add.s32 	%r42, %r39, %r41;
	shr.s32 	%r43, %r42, 4;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk06_mlp_down.py:315:27 ]
	add.s32 	%r44, %r19, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk06_mlp_down.py:315:27 ]
	shr.s32 	%r45, %r44, 31;
	shr.u32 	%r46, %r45, 25;
	add.s32 	%r47, %r44, %r46;
	shr.s32 	%r48, %r47, 7;
$L__tmp3:
	.loc	1 316 29                        // sk06_mlp_down.py:316:29
	shl.b32 	%r49, %r48, 3;
	.loc	1 317 22                        // sk06_mlp_down.py:317:22
	div.s32 	%r50, %r38, %r49;
	.loc	1 317 38                        // sk06_mlp_down.py:317:38
	shl.b32 	%r51, %r50, 3;
	.loc	1 318 30                        // sk06_mlp_down.py:318:30
	sub.s32 	%r52, %r43, %r51;
	ld.param.b32 	%r53, [_sk06_mlp_down_kernel_param_10];
	.loc	1 318 39                        // sk06_mlp_down.py:318:39
	min.s32 	%r54, %r52, 8;
	ld.param.b32 	%r55, [_sk06_mlp_down_kernel_param_11];
	.loc	1 319 30                        // sk06_mlp_down.py:319:30
	mul.lo.s32 	%r56, %r50, %r49;
	sub.s32 	%r57, %r38, %r56;
	.loc	1 320 36                        // sk06_mlp_down.py:320:36
	div.s32 	%r58, %r57, %r54;
	.loc	1 319 46                        // sk06_mlp_down.py:319:46
	mul.lo.s32 	%r59, %r58, %r54;
	sub.s32 	%r60, %r57, %r59;
	.loc	1 319 23                        // sk06_mlp_down.py:319:23
	add.s32 	%r61, %r60, %r51;
	.loc	1 322 22                        // sk06_mlp_down.py:322:22
	shl.b32 	%r1, %r61, 4;
	.loc	1 322 45                        // sk06_mlp_down.py:322:45
	mov.u32 	%r2, %tid.x;
	and.b32 	%r3, %r2, 240;
	bfe.u32 	%r62, %r2, 4, 4;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r4, %r1, %r62;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r5, %r4, %r18;
	.loc	1 323 22                        // sk06_mlp_down.py:323:22
	shl.b32 	%r6, %r58, 7;
	.loc	1 323 45                        // sk06_mlp_down.py:323:45
	shr.u32 	%r63, %r2, 3;
	bfe.u32 	%r64, %r2, 3, 5;
	shl.b32 	%r7, %r2, 1;
	and.b32 	%r8, %r2, 224;
	and.b32 	%r9, %r2, 15;
	shl.b32 	%r68, %r9, 3;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r69, %r6, %r64;
	or.b32 	%r70, %r69, 32;
	or.b32 	%r71, %r69, 64;
	or.b32 	%r72, %r63, %r6;
	or.b32 	%r73, %r72, 96;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r78, %r69, %r19;
	rem.s32 	%r79, %r70, %r19;
	rem.s32 	%r80, %r71, %r19;
	rem.s32 	%r81, %r73, %r19;
	.loc	1 326 39                        // sk06_mlp_down.py:326:39
	mul.lo.s32 	%r86, %r5, %r53;
	.loc	1 326 21                        // sk06_mlp_down.py:326:21
	cvt.s64.s32 	%rd1, %r86;
	add.s64 	%rd35, %rd19, %rd1;
	.loc	1 326 51                        // sk06_mlp_down.py:326:51
	cvt.u64.u32 	%rd2, %r68;
	add.s64 	%rd24, %rd35, %rd2;
	.loc	1 327 28                        // sk06_mlp_down.py:327:28
	and.b32 	%r10, %r2, 7;
	shl.b32 	%r87, %r10, 4;
	.loc	1 327 21                        // sk06_mlp_down.py:327:21
	cvt.u64.u32 	%rd3, %r87;
	add.s64 	%rd36, %rd20, %rd3;
	.loc	1 327 69                        // sk06_mlp_down.py:327:69
	mul.lo.s32 	%r88, %r78, %r55;
	mul.lo.s32 	%r89, %r79, %r55;
	mul.lo.s32 	%r90, %r80, %r55;
	mul.lo.s32 	%r91, %r81, %r55;
	.loc	1 327 51                        // sk06_mlp_down.py:327:51
	cvt.s64.s32 	%rd4, %r88;
	add.s64 	%rd25, %rd36, %rd4;
	cvt.s64.s32 	%rd5, %r89;
	add.s64 	%rd26, %rd36, %rd5;
	cvt.s64.s32 	%rd6, %r90;
	add.s64 	%rd27, %rd36, %rd6;
	cvt.s64.s32 	%rd7, %r91;
	add.s64 	%rd28, %rd36, %rd7;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	setp.lt.s32 	%p1, %r20, 128;
	setp.gt.s32 	%p2, %r20, 127;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	and.b32 	%r108, %r2, 255;
	shl.b32 	%r109, %r108, 3;
	and.b32 	%r110, %r2, 112;
	xor.b32 	%r111, %r109, %r110;
	mov.b32 	%r112, global_smem;
	add.s32 	%r11, %r112, %r111;
	add.s32 	%r24, %r11, 32768;
	selp.b32 	%r25, 8, 0, %p2;
	// begin inline asm
	cp.async.ca.shared.global [ %r24 + 0 ], [ %rd24 + 0 ], 0x8, %r25;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	shl.b32 	%r113, %r108, 4;
	and.b32 	%r114, %r7, 112;
	xor.b32 	%r115, %r113, %r114;
	add.s32 	%r26, %r112, %r115;
	selp.b32 	%r27, 16, 0, %p2;
	// begin inline asm
	cp.async.cg.shared.global [ %r26 + 0 ], [ %rd25 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r28, %r26, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r28 + 0 ], [ %rd26 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r29, %r26, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r29 + 0 ], [ %rd27 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r30, %r26, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r30 + 0 ], [ %rd28 + 0 ], 0x10, %r27;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	setp.gt.s32 	%p3, %r20, 255;
	.loc	1 344 18                        // sk06_mlp_down.py:344:18
	add.s64 	%rd29, %rd24, 128;
	.loc	1 345 18                        // sk06_mlp_down.py:345:18
	add.s64 	%rd30, %rd25, 128;
	add.s64 	%rd31, %rd26, 128;
	add.s64 	%rd32, %rd27, 128;
	add.s64 	%rd33, %rd28, 128;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	bar.sync 	0;
	add.s32 	%r31, %r11, 34816;
	selp.b32 	%r32, 8, 0, %p3;
	// begin inline asm
	cp.async.ca.shared.global [ %r31 + 0 ], [ %rd29 + 0 ], 0x8, %r32;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	add.s32 	%r33, %r26, 16384;
	selp.b32 	%r34, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r33 + 0 ], [ %rd30 + 0 ], 0x10, %r34;
	// end inline asm
	add.s32 	%r35, %r26, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r35 + 0 ], [ %rd31 + 0 ], 0x10, %r34;
	// end inline asm
	add.s32 	%r36, %r26, 24576;
	// begin inline asm
	cp.async.cg.shared.global [ %r36 + 0 ], [ %rd32 + 0 ], 0x10, %r34;
	// end inline asm
	add.s32 	%r37, %r26, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r37 + 0 ], [ %rd33 + 0 ], 0x10, %r34;
	// end inline asm
	cp.async.commit_group;
	mov.b32 	%r290, 0f00000000;
	cvt.u32.u64 	%r286, %rd3;
	mov.b32 	%r291, %r290;
	mov.b32 	%r292, %r290;
	mov.b32 	%r293, %r290;
	mov.b32 	%r294, %r290;
	mov.b32 	%r295, %r290;
	mov.b32 	%r296, %r290;
	mov.b32 	%r297, %r290;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	@%p1 bra 	$L__BB0_3;
// %bb.1:                               // %.lr.ph
	.loc	1 0 23                          // sk06_mlp_down.py:0:23
	ld.param.b32 	%r23, [_sk06_mlp_down_kernel_param_14];
	ld.param.b64 	%rd34, [_sk06_mlp_down_kernel_param_6];
	and.b32 	%r65, %r7, 6;
	shr.u32 	%r66, %r8, 2;
	or.b32 	%r67, %r65, %r66;
	or.b32 	%r74, %r6, %r67;
	or.b32 	%r75, %r74, 1;
	or.b32 	%r76, %r74, 64;
	or.b32 	%r77, %r74, 65;
	rem.s32 	%r82, %r74, %r19;
	rem.s32 	%r83, %r75, %r19;
	rem.s32 	%r84, %r76, %r19;
	rem.s32 	%r85, %r77, %r19;
	shr.s32 	%r92, %r82, 31;
	shr.u32 	%r93, %r92, 25;
	add.s32 	%r94, %r82, %r93;
	shr.s32 	%r95, %r94, 7;
	shr.s32 	%r96, %r83, 31;
	shr.u32 	%r97, %r96, 25;
	add.s32 	%r98, %r83, %r97;
	shr.s32 	%r99, %r98, 7;
	shr.s32 	%r100, %r84, 31;
	shr.u32 	%r101, %r100, 25;
	add.s32 	%r102, %r84, %r101;
	shr.s32 	%r103, %r102, 7;
	shr.s32 	%r104, %r85, 31;
	shr.u32 	%r105, %r104, 25;
	add.s32 	%r106, %r85, %r105;
	shr.s32 	%r107, %r106, 7;
	mad.wide.s32 	%rd8, %r95, 4, %rd34;
	mad.wide.s32 	%rd9, %r99, 4, %rd34;
	mad.wide.s32 	%rd10, %r103, 4, %rd34;
	mad.wide.s32 	%rd11, %r107, 4, %rd34;
	.loc	1 337 28                        // sk06_mlp_down.py:337:28
	shr.u32 	%r116, %r20, 7;
	add.s32 	%r117, %r116, -2;
	shl.b32 	%r118, %r9, 7;
	and.b32 	%r119, %r2, 16;
	xor.b32 	%r120, %r286, %r119;
	or.b32 	%r12, %r120, %r118;
	xor.b32 	%r13, %r12, 32;
	xor.b32 	%r14, %r12, 64;
	xor.b32 	%r15, %r12, 96;
	shl.b32 	%r121, %r10, 7;
	shl.b32 	%r122, %r8, 5;
	and.b32 	%r123, %r7, 48;
	or.b32 	%r124, %r121, %r122;
	xor.b32 	%r125, %r286, %r123;
	or.b32 	%r16, %r124, %r125;
	xor.b32 	%r17, %r16, 64;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	cvt.s64.s32 	%rd12, %r117;
	and.b32 	%r126, %r20, -128;
	cvt.u64.u32 	%rd13, %r126;
	add.s64 	%rd37, %rd3, %rd7;
	add.s64 	%rd38, %rd37, %rd20;
	add.s64 	%rd14, %rd38, 256;
	add.s64 	%rd39, %rd3, %rd6;
	add.s64 	%rd40, %rd39, %rd20;
	add.s64 	%rd15, %rd40, 256;
	add.s64 	%rd41, %rd3, %rd5;
	add.s64 	%rd42, %rd41, %rd20;
	add.s64 	%rd16, %rd42, 256;
	add.s64 	%rd43, %rd3, %rd4;
	add.s64 	%rd44, %rd43, %rd20;
	add.s64 	%rd17, %rd44, 256;
	add.s64 	%rd45, %rd2, %rd1;
	add.s64 	%rd46, %rd45, %rd19;
	add.s64 	%rd18, %rd46, 256;
	mov.b32 	%r290, 0f00000000;
	mov.b32 	%r289, 1;
	mov.b32 	%r288, -1;
	mov.b64 	%rd63, 0;
	mov.b32 	%r127, 0;
	mov.b32 	%r287, %r127;
	mov.b64 	%rd64, %rd63;
	mov.b32 	%r291, %r290;
	mov.b32 	%r292, %r290;
	mov.b32 	%r293, %r290;
	mov.b32 	%r294, %r290;
	mov.b32 	%r295, %r290;
	mov.b32 	%r296, %r290;
	mov.b32 	%r297, %r290;
$L__BB0_2:                              // =>This Inner Loop Header: Depth=1
	setp.lt.s64 	%p4, %rd64, %rd12;
	add.s32 	%r179, %r288, 1;
	setp.gt.s32 	%p5, %r179, 1;
	selp.b32 	%r288, 0, %r179, %p5;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	cp.async.wait_group 	2;
	bar.sync 	0;
	shl.b32 	%r180, %r288, 11;
	add.s32 	%r181, %r112, %r180;
	add.s32 	%r182, %r181, %r12;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r128, %r129, %r130, %r131}, [%r182+32768];
	add.s32 	%r183, %r181, %r13;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r140, %r141, %r142, %r143}, [%r183+32768];
	add.s32 	%r184, %r181, %r14;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r152, %r153, %r154, %r155}, [%r184+32768];
	add.s32 	%r185, %r181, %r15;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r160, %r161, %r162, %r163}, [%r185+32768];
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	shl.b32 	%r186, %r288, 14;
	add.s32 	%r187, %r112, %r186;
	add.s32 	%r188, %r187, %r16;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r132, %r133, %r144, %r145}, [%r188];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r134, %r135, %r150, %r151}, [%r188+8192];
	add.s32 	%r189, %r187, %r17;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r156, %r157, %r164, %r165}, [%r189];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r158, %r159, %r166, %r167}, [%r189+8192];
	.loc	1 338 36                        // sk06_mlp_down.py:338:36
	mov.b32 	%r136, %r127;
	mov.b32 	%r137, %r127;
	mov.b32 	%r138, %r127;
	mov.b32 	%r139, %r127;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r136, %r137, %r138, %r139 }, { %r128, %r129, %r130, %r131 }, { %r132, %r133 }, { %r136, %r137, %r138, %r139 };
	// end inline asm
	mov.b32 	%r149, %r127;
	mov.b32 	%r146, %r127;
	mov.b32 	%r147, %r127;
	mov.b32 	%r148, %r127;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r146, %r147, %r148, %r149 }, { %r128, %r129, %r130, %r131 }, { %r134, %r135 }, { %r146, %r147, %r148, %r149 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r136, %r137, %r138, %r139 }, { %r140, %r141, %r142, %r143 }, { %r144, %r145 }, { %r136, %r137, %r138, %r139 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r146, %r147, %r148, %r149 }, { %r140, %r141, %r142, %r143 }, { %r150, %r151 }, { %r146, %r147, %r148, %r149 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r136, %r137, %r138, %r139 }, { %r152, %r153, %r154, %r155 }, { %r156, %r157 }, { %r136, %r137, %r138, %r139 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r146, %r147, %r148, %r149 }, { %r152, %r153, %r154, %r155 }, { %r158, %r159 }, { %r146, %r147, %r148, %r149 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r136, %r137, %r138, %r139 }, { %r160, %r161, %r162, %r163 }, { %r164, %r165 }, { %r136, %r137, %r138, %r139 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r146, %r147, %r148, %r149 }, { %r160, %r161, %r162, %r163 }, { %r166, %r167 }, { %r146, %r147, %r148, %r149 };
	// end inline asm
	.loc	1 340 37                        // sk06_mlp_down.py:340:37
	mul.wide.s32 	%rd56, %r287, 4;
	add.s64 	%rd47, %rd8, %rd56;
	add.s64 	%rd48, %rd9, %rd56;
	add.s64 	%rd49, %rd10, %rd56;
	add.s64 	%rd50, %rd11, %rd56;
	.loc	1 340 27                        // sk06_mlp_down.py:340:27
	// begin inline asm
	mov.u32 %r168, 0x0;
	ld.global.b32 { %r168 }, [ %rd47 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r169, 0x0;
	ld.global.b32 { %r169 }, [ %rd48 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r170, 0x0;
	ld.global.b32 { %r170 }, [ %rd49 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r171, 0x0;
	ld.global.b32 { %r171 }, [ %rd50 + 0 ];
	// end inline asm
	.loc	1 341 23                        // sk06_mlp_down.py:341:23
	cvt.rn.f32.s32 	%r190, %r147;
	cvt.rn.f32.s32 	%r191, %r149;
	cvt.rn.f32.s32 	%r192, %r146;
	cvt.rn.f32.s32 	%r193, %r148;
	cvt.rn.f32.s32 	%r194, %r137;
	cvt.rn.f32.s32 	%r195, %r139;
	cvt.rn.f32.s32 	%r196, %r136;
	cvt.rn.f32.s32 	%r197, %r138;
	.loc	1 341 19                        // sk06_mlp_down.py:341:19
	fma.rn.f32 	%r292, %r168, %r197, %r292;
	fma.rn.f32 	%r290, %r168, %r196, %r290;
	fma.rn.f32 	%r293, %r169, %r195, %r293;
	fma.rn.f32 	%r291, %r169, %r194, %r291;
	fma.rn.f32 	%r296, %r170, %r193, %r296;
	fma.rn.f32 	%r294, %r170, %r192, %r294;
	fma.rn.f32 	%r297, %r171, %r191, %r297;
	fma.rn.f32 	%r295, %r171, %r190, %r295;
	.loc	1 345 18                        // sk06_mlp_down.py:345:18
	add.s64 	%rd51, %rd18, %rd63;
	add.s64 	%rd52, %rd17, %rd63;
	add.s64 	%rd53, %rd16, %rd63;
	add.s64 	%rd54, %rd15, %rd63;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	add.s64 	%rd55, %rd14, %rd63;
	add.s32 	%r198, %r289, 1;
	setp.gt.s32 	%p6, %r198, 1;
	selp.b32 	%r289, 0, %r198, %p6;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	shl.b32 	%r199, %r289, 11;
	bar.sync 	0;
	add.s32 	%r200, %r11, %r199;
	add.s32 	%r172, %r200, 32768;
	selp.b32 	%r173, 8, 0, %p4;
	// begin inline asm
	cp.async.ca.shared.global [ %r172 + 0 ], [ %rd51 + 0 ], 0x8, %r173;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	shl.b32 	%r201, %r289, 14;
	add.s32 	%r174, %r26, %r201;
	selp.b32 	%r175, 16, 0, %p4;
	// begin inline asm
	cp.async.cg.shared.global [ %r174 + 0 ], [ %rd52 + 0 ], 0x10, %r175;
	// end inline asm
	add.s32 	%r176, %r174, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r176 + 0 ], [ %rd53 + 0 ], 0x10, %r175;
	// end inline asm
	add.s32 	%r177, %r174, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r177 + 0 ], [ %rd54 + 0 ], 0x10, %r175;
	// end inline asm
	add.s32 	%r178, %r174, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r178 + 0 ], [ %rd55 + 0 ], 0x10, %r175;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	add.s64 	%rd64, %rd64, 1;
	add.s64 	%rd63, %rd63, 128;
	add.s32 	%r287, %r287, %r23;
	setp.ne.b64 	%p7, %rd13, %rd63;
	@%p7 bra 	$L__BB0_2;
$L__BB0_3:                              // %._crit_edge
	.loc	1 0 23                          // sk06_mlp_down.py:0:23
	cvt.u32.u64 	%r217, %rd2;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r218, %r6, %r217;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r219, %r218, %r19;
	.loc	1 322 45                        // sk06_mlp_down.py:322:45
	and.b32 	%r220, %r2, 28;
	bfe.u32 	%r221, %r2, 2, 3;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r222, %r221, %r1;
	or.b32 	%r223, %r222, 8;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r224, %r223, %r18;
	rem.s32 	%r225, %r222, %r18;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 347 38                        // sk06_mlp_down.py:347:38
	mad.wide.s32 	%rd57, %r225, 4, %rd23;
	mad.wide.s32 	%rd58, %r224, 4, %rd23;
	.loc	1 347 24                        // sk06_mlp_down.py:347:24
	// begin inline asm
	mov.u32 %r202, 0x0;
	ld.global.b32 { %r202 }, [ %rd57 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r203, 0x0;
	ld.global.b32 { %r203 }, [ %rd58 + 0 ];
	// end inline asm
	.loc	1 350 49                        // sk06_mlp_down.py:350:49
	mul.lo.s32 	%r226, %r5, %r22;
	.loc	1 350 31                        // sk06_mlp_down.py:350:31
	mad.wide.s32 	%rd61, %r226, 2, %rd22;
	.loc	1 350 64                        // sk06_mlp_down.py:350:64
	mad.wide.s32 	%rd59, %r219, 2, %rd61;
	.loc	1 350 19                        // sk06_mlp_down.py:350:19
	// begin inline asm
	mov.u32 %r205, 0x0;
	mov.u32 %r206, 0x0;
	mov.u32 %r207, 0x0;
	mov.u32 %r208, 0x0;
	ld.global.v4.b32 { %r205, %r206, %r207, %r208 }, [ %rd59 + 0 ];
	// end inline asm
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	and.b32 	%r227, %r2, 120;
	shl.b32 	%r228, %r227, 5;
	or.b32 	%r229, %r228, %r286;
	xor.b32 	%r230, %r229, %r3;
	add.s32 	%r204, %r112, %r230;
	// begin inline asm
	st.shared.v4.b32 [ %r204 + 0 ], { %r205, %r206, %r207, %r208 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r231, %r10, 9;
	shl.b32 	%r232, %r2, 4;
	and.b32 	%r233, %r232, 496;
	shr.u32 	%r234, %r8, 1;
	xor.b32 	%r235, %r233, %r234;
	add.s32 	%r236, %r112, %r231;
	add.s32 	%r237, %r236, %r235;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r238, %r239, %r240, %r241}, [%r237];
	mov.b32 	{%rs9, %rs10}, %r238;
	mov.b32 	{%rs11, %rs12}, %r239;
	mov.b32 	{%rs13, %rs14}, %r240;
	mov.b32 	{%rs15, %rs16}, %r241;
	cvt.f32.bf16 	%r242, %rs9;
	cvt.f32.bf16 	%r243, %rs10;
	cvt.f32.bf16 	%r244, %rs11;
	cvt.f32.bf16 	%r245, %rs12;
	cvt.f32.bf16 	%r246, %rs13;
	cvt.f32.bf16 	%r247, %rs14;
	cvt.f32.bf16 	%r248, %rs15;
	cvt.f32.bf16 	%r249, %rs16;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r250, %r290, %r202, %r242;
	fma.rn.f32 	%r251, %r291, %r202, %r243;
	fma.rn.f32 	%r252, %r292, %r203, %r244;
	fma.rn.f32 	%r253, %r293, %r203, %r245;
	fma.rn.f32 	%r254, %r294, %r202, %r246;
	fma.rn.f32 	%r255, %r295, %r202, %r247;
	fma.rn.f32 	%r256, %r296, %r203, %r248;
	fma.rn.f32 	%r257, %r297, %r203, %r249;
	.loc	1 357 31                        // sk06_mlp_down.py:357:31
	setp.lt.s32 	%p9, %r4, %r18;
	.loc	1 357 54                        // sk06_mlp_down.py:357:54
	setp.lt.s32 	%p10, %r218, %r19;
	.loc	1 357 37                        // sk06_mlp_down.py:357:37
	and.pred 	%p8, %p9, %p10;
	.loc	1 355 35                        // sk06_mlp_down.py:355:35
	mul.lo.s32 	%r258, %r4, %r21;
	.loc	1 355 18                        // sk06_mlp_down.py:355:18
	mad.wide.s32 	%rd62, %r258, 2, %rd21;
	.loc	1 355 50                        // sk06_mlp_down.py:355:50
	mad.wide.s32 	%rd60, %r218, 2, %rd62;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16.f32 	%rs1, %r250;
	cvt.rn.bf16.f32 	%rs2, %r251;
	cvt.rn.bf16.f32 	%rs3, %r252;
	cvt.rn.bf16.f32 	%rs4, %r253;
	cvt.rn.bf16.f32 	%rs5, %r254;
	cvt.rn.bf16.f32 	%rs6, %r255;
	cvt.rn.bf16.f32 	%rs7, %r256;
	cvt.rn.bf16.f32 	%rs8, %r257;
	bar.sync 	0;
	shl.b32 	%r259, %r2, 5;
	and.b32 	%r260, %r259, 768;
	shl.b32 	%r261, %r220, 1;
	and.b32 	%r262, %r2, 1;
	neg.s32 	%r263, %r262;
	and.b32 	%r264, %r263, 1088;
	bfe.s32 	%r265, %r2, 1, 1;
	and.b32 	%r266, %r265, 2052;
	or.b32 	%r267, %r260, %r261;
	or.b32 	%r268, %r264, %r267;
	xor.b32 	%r269, %r268, %r234;
	or.b32 	%r270, %r269, %r266;
	add.s32 	%r209, %r112, %r270;
	// begin inline asm
	st.shared.v2.b16 [ %r209 + 0 ], { %rs1, %rs2 };
	// end inline asm
	add.s32 	%r210, %r209, 128;
	// begin inline asm
	st.shared.v2.b16 [ %r210 + 0 ], { %rs3, %rs4 };
	// end inline asm
	xor.b32 	%r271, %r270, 4;
	add.s32 	%r211, %r112, %r271;
	// begin inline asm
	st.shared.v2.b16 [ %r211 + 0 ], { %rs5, %rs6 };
	// end inline asm
	add.s32 	%r212, %r211, 128;
	// begin inline asm
	st.shared.v2.b16 [ %r212 + 0 ], { %rs7, %rs8 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r272, %r2, 3;
	and.b32 	%r273, %r272, 768;
	shr.u32 	%r274, %r227, 1;
	and.b32 	%r275, %r2, 128;
	or.b32 	%r276, %r286, %r273;
	xor.b32 	%r277, %r276, %r274;
	or.b32 	%r278, %r277, %r275;
	add.s32 	%r279, %r112, %r278;
	ld.shared.b32 	%r213, [%r279];
	xor.b32 	%r280, %r278, 64;
	add.s32 	%r281, %r112, %r280;
	ld.shared.b32 	%r214, [%r281+1024];
	xor.b32 	%r282, %r278, 4;
	add.s32 	%r283, %r112, %r282;
	ld.shared.b32 	%r215, [%r283+2048];
	xor.b32 	%r284, %r278, 68;
	add.s32 	%r285, %r112, %r284;
	ld.shared.b32 	%r216, [%r285+3072];
	.loc	1 356 8                         // sk06_mlp_down.py:356:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd60 + 0 ], { %r213, %r214, %r215, %r216 };
	// end inline asm
	.loc	1 354 4                         // sk06_mlp_down.py:354:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/workspace/vllm/_genesis/kernels/sk06_mlp_down.py"
	.file	2 "/usr/local/lib/python3.12/dist-packages/triton/language/standard.py"
	.section	.debug_abbrev
	{
.b8 1                                   // Abbreviation Code
.b8 17                                  // DW_TAG_compile_unit
.b8 1                                   // DW_CHILDREN_yes
.b8 37                                  // DW_AT_producer
.b8 8                                   // DW_FORM_string
.b8 19                                  // DW_AT_language
.b8 5                                   // DW_FORM_data2
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 16                                  // DW_AT_stmt_list
.b8 6                                   // DW_FORM_data4
.b8 27                                  // DW_AT_comp_dir
.b8 8                                   // DW_FORM_string
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 2                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 0                                   // DW_CHILDREN_no
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 32                                  // DW_AT_inline
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 3                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 1                                   // DW_CHILDREN_yes
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 4                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 0                                   // DW_CHILDREN_no
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 5                                   // DW_FORM_data2
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 168                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0xa1 DW_TAG_compile_unit
.b8 116                                 // DW_AT_producer
.b8 114
.b8 105
.b8 116
.b8 111
.b8 110
.b8 0
.b8 2                                   // DW_AT_language
.b8 0
.b8 115                                 // DW_AT_name
.b8 107
.b8 48
.b8 54
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 100
.b8 111
.b8 119
.b8 110
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 119
.b8 111
.b8 114
.b8 107
.b8 115
.b8 112
.b8 97
.b8 99
.b8 101
.b8 47
.b8 118
.b8 108
.b8 108
.b8 109
.b8 47
.b8 95
.b8 103
.b8 101
.b8 110
.b8 101
.b8 115
.b8 105
.b8 115
.b8 47
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 115
.b8 0
.b8 2                                   // Abbrev [2] 0x4b:0x18 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 54
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 100
.b8 111
.b8 119
.b8 110
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x63:0x48 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 75                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x78:0x19 DW_TAG_inlined_subroutine
.b32 75                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 58                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x91:0x19 DW_TAG_inlined_subroutine
.b32 75                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 59                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_2 = _Nativo(
    "sk06_mlp_down/tile16x128x128_shift1_abi15",
    _PTX_2, "_sk06_mlp_down_kernel",
    warps=8, shared=36864,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 18],
    horneado={11: 1, 12: 1, 15: 1, 17: 1, 19: 1, 20: 16, 21: 128, 22: 128, 23: 8, 24: 128, 25: True},
    div16=[8, 9, 10, 13, 14, 16],
)

_PTX_3 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk06_mlp_down_kernel   // -- Begin function _sk06_mlp_down_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk06_mlp_down_kernel
.visible .entry _sk06_mlp_down_kernel(
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_6,
	.param .u32 _sk06_mlp_down_kernel_param_7,
	.param .u32 _sk06_mlp_down_kernel_param_8,
	.param .u32 _sk06_mlp_down_kernel_param_9,
	.param .u32 _sk06_mlp_down_kernel_param_10,
	.param .u32 _sk06_mlp_down_kernel_param_11,
	.param .u32 _sk06_mlp_down_kernel_param_12,
	.param .u32 _sk06_mlp_down_kernel_param_13,
	.param .u32 _sk06_mlp_down_kernel_param_14,
	.param .u32 _sk06_mlp_down_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_16,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_17
)
.reqntid 256
{
	.reg .pred 	%p<11>;
	.reg .b16 	%rs<25>;
	.reg .b32 	%r<321>;
	.reg .b64 	%rd<72>;
	.loc	1 304 0                         // sk06_mlp_down.py:304:0
$L__func_begin0:
	.loc	1 304 0                         // sk06_mlp_down.py:304:0

// %bb.0:
	ld.param.b32 	%r23, [_sk06_mlp_down_kernel_param_14];
	ld.param.b32 	%r22, [_sk06_mlp_down_kernel_param_13];
	ld.param.b32 	%r21, [_sk06_mlp_down_kernel_param_12];
	ld.param.b32 	%r20, [_sk06_mlp_down_kernel_param_9];
	ld.param.b32 	%r19, [_sk06_mlp_down_kernel_param_8];
	ld.param.b32 	%r18, [_sk06_mlp_down_kernel_param_7];
	ld.param.b64 	%rd23, [_sk06_mlp_down_kernel_param_4];
	ld.param.b64 	%rd22, [_sk06_mlp_down_kernel_param_3];
	ld.param.b64 	%rd21, [_sk06_mlp_down_kernel_param_2];
	ld.param.b64 	%rd20, [_sk06_mlp_down_kernel_param_1];
	ld.param.b64 	%rd19, [_sk06_mlp_down_kernel_param_0];
$L__tmp0:
	.loc	1 313 24                        // sk06_mlp_down.py:313:24
	mov.u32 	%r39, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk06_mlp_down.py:314:27 ]
	add.s32 	%r40, %r18, 15;
	.loc	2 43 30                         // standard.py:43:30 @[ sk06_mlp_down.py:314:27 ]
	shr.s32 	%r41, %r40, 31;
	shr.u32 	%r42, %r41, 28;
	add.s32 	%r43, %r40, %r42;
	shr.s32 	%r44, %r43, 4;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk06_mlp_down.py:315:27 ]
	add.s32 	%r45, %r19, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk06_mlp_down.py:315:27 ]
	shr.s32 	%r46, %r45, 31;
	shr.u32 	%r47, %r46, 25;
	add.s32 	%r48, %r45, %r47;
	shr.s32 	%r49, %r48, 7;
$L__tmp3:
	.loc	1 316 29                        // sk06_mlp_down.py:316:29
	shl.b32 	%r50, %r49, 3;
	.loc	1 317 22                        // sk06_mlp_down.py:317:22
	div.s32 	%r51, %r39, %r50;
	.loc	1 317 38                        // sk06_mlp_down.py:317:38
	shl.b32 	%r52, %r51, 3;
	.loc	1 318 30                        // sk06_mlp_down.py:318:30
	sub.s32 	%r53, %r44, %r52;
	ld.param.b32 	%r54, [_sk06_mlp_down_kernel_param_10];
	.loc	1 318 39                        // sk06_mlp_down.py:318:39
	min.s32 	%r55, %r53, 8;
	ld.param.b32 	%r56, [_sk06_mlp_down_kernel_param_11];
	.loc	1 319 30                        // sk06_mlp_down.py:319:30
	mul.lo.s32 	%r57, %r51, %r50;
	sub.s32 	%r58, %r39, %r57;
	.loc	1 320 36                        // sk06_mlp_down.py:320:36
	div.s32 	%r59, %r58, %r55;
	.loc	1 319 46                        // sk06_mlp_down.py:319:46
	mul.lo.s32 	%r60, %r59, %r55;
	sub.s32 	%r61, %r58, %r60;
	.loc	1 319 23                        // sk06_mlp_down.py:319:23
	add.s32 	%r62, %r61, %r52;
	.loc	1 322 22                        // sk06_mlp_down.py:322:22
	shl.b32 	%r1, %r62, 4;
	.loc	1 322 45                        // sk06_mlp_down.py:322:45
	mov.u32 	%r2, %tid.x;
	and.b32 	%r3, %r2, 240;
	bfe.u32 	%r63, %r2, 4, 4;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r4, %r1, %r63;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r5, %r4, %r18;
	.loc	1 323 22                        // sk06_mlp_down.py:323:22
	shl.b32 	%r6, %r59, 7;
	.loc	1 323 45                        // sk06_mlp_down.py:323:45
	shr.u32 	%r64, %r2, 3;
	bfe.u32 	%r65, %r2, 3, 5;
	shl.b32 	%r7, %r2, 1;
	and.b32 	%r8, %r2, 224;
	and.b32 	%r9, %r2, 15;
	shl.b32 	%r69, %r9, 3;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r70, %r6, %r65;
	or.b32 	%r71, %r70, 32;
	or.b32 	%r72, %r70, 64;
	or.b32 	%r73, %r64, %r6;
	or.b32 	%r74, %r73, 96;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r79, %r70, %r19;
	rem.s32 	%r80, %r71, %r19;
	rem.s32 	%r81, %r72, %r19;
	rem.s32 	%r82, %r74, %r19;
	.loc	1 326 39                        // sk06_mlp_down.py:326:39
	mul.lo.s32 	%r87, %r5, %r54;
	.loc	1 326 21                        // sk06_mlp_down.py:326:21
	cvt.s64.s32 	%rd1, %r87;
	add.s64 	%rd35, %rd19, %rd1;
	.loc	1 326 51                        // sk06_mlp_down.py:326:51
	cvt.u64.u32 	%rd2, %r69;
	add.s64 	%rd24, %rd35, %rd2;
	.loc	1 327 28                        // sk06_mlp_down.py:327:28
	and.b32 	%r10, %r2, 7;
	shl.b32 	%r88, %r10, 4;
	.loc	1 327 21                        // sk06_mlp_down.py:327:21
	cvt.u64.u32 	%rd3, %r88;
	add.s64 	%rd36, %rd20, %rd3;
	.loc	1 327 69                        // sk06_mlp_down.py:327:69
	mul.lo.s32 	%r89, %r79, %r56;
	mul.lo.s32 	%r90, %r80, %r56;
	mul.lo.s32 	%r91, %r81, %r56;
	mul.lo.s32 	%r92, %r82, %r56;
	.loc	1 327 51                        // sk06_mlp_down.py:327:51
	cvt.s64.s32 	%rd4, %r89;
	add.s64 	%rd25, %rd36, %rd4;
	cvt.s64.s32 	%rd5, %r90;
	add.s64 	%rd26, %rd36, %rd5;
	cvt.s64.s32 	%rd6, %r91;
	add.s64 	%rd27, %rd36, %rd6;
	cvt.s64.s32 	%rd7, %r92;
	add.s64 	%rd28, %rd36, %rd7;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	setp.lt.s32 	%p1, %r20, 128;
	setp.gt.s32 	%p2, %r20, 127;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	and.b32 	%r109, %r2, 255;
	shl.b32 	%r110, %r109, 3;
	and.b32 	%r111, %r2, 112;
	xor.b32 	%r112, %r110, %r111;
	mov.b32 	%r113, global_smem;
	add.s32 	%r11, %r113, %r112;
	add.s32 	%r25, %r11, 32768;
	selp.b32 	%r26, 8, 0, %p2;
	// begin inline asm
	cp.async.ca.shared.global [ %r25 + 0 ], [ %rd24 + 0 ], 0x8, %r26;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	shl.b32 	%r114, %r109, 4;
	and.b32 	%r115, %r7, 112;
	xor.b32 	%r116, %r114, %r115;
	add.s32 	%r27, %r113, %r116;
	selp.b32 	%r28, 16, 0, %p2;
	// begin inline asm
	cp.async.cg.shared.global [ %r27 + 0 ], [ %rd25 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r29, %r27, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r29 + 0 ], [ %rd26 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r30, %r27, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r30 + 0 ], [ %rd27 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r31, %r27, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r31 + 0 ], [ %rd28 + 0 ], 0x10, %r28;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	setp.gt.s32 	%p3, %r20, 255;
	.loc	1 344 18                        // sk06_mlp_down.py:344:18
	add.s64 	%rd29, %rd24, 128;
	.loc	1 345 18                        // sk06_mlp_down.py:345:18
	add.s64 	%rd30, %rd25, 128;
	add.s64 	%rd31, %rd26, 128;
	add.s64 	%rd32, %rd27, 128;
	add.s64 	%rd33, %rd28, 128;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	bar.sync 	0;
	add.s32 	%r32, %r11, 34816;
	selp.b32 	%r33, 8, 0, %p3;
	// begin inline asm
	cp.async.ca.shared.global [ %r32 + 0 ], [ %rd29 + 0 ], 0x8, %r33;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	add.s32 	%r34, %r27, 16384;
	selp.b32 	%r35, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r34 + 0 ], [ %rd30 + 0 ], 0x10, %r35;
	// end inline asm
	add.s32 	%r36, %r27, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r36 + 0 ], [ %rd31 + 0 ], 0x10, %r35;
	// end inline asm
	add.s32 	%r37, %r27, 24576;
	// begin inline asm
	cp.async.cg.shared.global [ %r37 + 0 ], [ %rd32 + 0 ], 0x10, %r35;
	// end inline asm
	add.s32 	%r38, %r27, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r38 + 0 ], [ %rd33 + 0 ], 0x10, %r35;
	// end inline asm
	cp.async.commit_group;
	mov.b32 	%r313, 0f00000000;
	cvt.u32.u64 	%r309, %rd3;
	mov.b32 	%r314, %r313;
	mov.b32 	%r315, %r313;
	mov.b32 	%r316, %r313;
	mov.b32 	%r317, %r313;
	mov.b32 	%r318, %r313;
	mov.b32 	%r319, %r313;
	mov.b32 	%r320, %r313;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	@%p1 bra 	$L__BB0_3;
// %bb.1:                               // %.lr.ph
	.loc	1 0 23                          // sk06_mlp_down.py:0:23
	ld.param.b32 	%r24, [_sk06_mlp_down_kernel_param_15];
	ld.param.b64 	%rd34, [_sk06_mlp_down_kernel_param_6];
	and.b32 	%r66, %r7, 6;
	shr.u32 	%r67, %r8, 2;
	or.b32 	%r68, %r66, %r67;
	or.b32 	%r75, %r6, %r68;
	or.b32 	%r76, %r75, 1;
	or.b32 	%r77, %r75, 64;
	or.b32 	%r78, %r75, 65;
	rem.s32 	%r83, %r75, %r19;
	rem.s32 	%r84, %r76, %r19;
	rem.s32 	%r85, %r77, %r19;
	rem.s32 	%r86, %r78, %r19;
	shr.s32 	%r93, %r83, 31;
	shr.u32 	%r94, %r93, 25;
	add.s32 	%r95, %r83, %r94;
	shr.s32 	%r96, %r95, 7;
	shr.s32 	%r97, %r84, 31;
	shr.u32 	%r98, %r97, 25;
	add.s32 	%r99, %r84, %r98;
	shr.s32 	%r100, %r99, 7;
	shr.s32 	%r101, %r85, 31;
	shr.u32 	%r102, %r101, 25;
	add.s32 	%r103, %r85, %r102;
	shr.s32 	%r104, %r103, 7;
	shr.s32 	%r105, %r86, 31;
	shr.u32 	%r106, %r105, 25;
	add.s32 	%r107, %r86, %r106;
	shr.s32 	%r108, %r107, 7;
	mad.wide.s32 	%rd8, %r96, 4, %rd34;
	mad.wide.s32 	%rd9, %r100, 4, %rd34;
	mad.wide.s32 	%rd10, %r104, 4, %rd34;
	mad.wide.s32 	%rd11, %r108, 4, %rd34;
	.loc	1 337 28                        // sk06_mlp_down.py:337:28
	shr.u32 	%r117, %r20, 7;
	add.s32 	%r118, %r117, -2;
	shl.b32 	%r119, %r9, 7;
	and.b32 	%r120, %r2, 16;
	xor.b32 	%r121, %r309, %r120;
	or.b32 	%r12, %r121, %r119;
	xor.b32 	%r13, %r12, 32;
	xor.b32 	%r14, %r12, 64;
	xor.b32 	%r15, %r12, 96;
	shl.b32 	%r122, %r10, 7;
	shl.b32 	%r123, %r8, 5;
	and.b32 	%r124, %r7, 48;
	or.b32 	%r125, %r122, %r123;
	xor.b32 	%r126, %r309, %r124;
	or.b32 	%r16, %r125, %r126;
	xor.b32 	%r17, %r16, 64;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	cvt.s64.s32 	%rd12, %r118;
	and.b32 	%r127, %r20, -128;
	cvt.u64.u32 	%rd13, %r127;
	add.s64 	%rd37, %rd3, %rd7;
	add.s64 	%rd38, %rd37, %rd20;
	add.s64 	%rd14, %rd38, 256;
	add.s64 	%rd39, %rd3, %rd6;
	add.s64 	%rd40, %rd39, %rd20;
	add.s64 	%rd15, %rd40, 256;
	add.s64 	%rd41, %rd3, %rd5;
	add.s64 	%rd42, %rd41, %rd20;
	add.s64 	%rd16, %rd42, 256;
	add.s64 	%rd43, %rd3, %rd4;
	add.s64 	%rd44, %rd43, %rd20;
	add.s64 	%rd17, %rd44, 256;
	add.s64 	%rd45, %rd2, %rd1;
	add.s64 	%rd46, %rd45, %rd19;
	add.s64 	%rd18, %rd46, 256;
	mov.b32 	%r313, 0f00000000;
	mov.b32 	%r312, 1;
	mov.b32 	%r311, -1;
	mov.b64 	%rd70, 0;
	mov.b32 	%r128, 0;
	mov.b32 	%r310, %r128;
	mov.b64 	%rd71, %rd70;
	mov.b32 	%r314, %r313;
	mov.b32 	%r315, %r313;
	mov.b32 	%r316, %r313;
	mov.b32 	%r317, %r313;
	mov.b32 	%r318, %r313;
	mov.b32 	%r319, %r313;
	mov.b32 	%r320, %r313;
$L__BB0_2:                              // =>This Inner Loop Header: Depth=1
	setp.lt.s64 	%p4, %rd71, %rd12;
	add.s32 	%r180, %r311, 1;
	setp.gt.s32 	%p5, %r180, 1;
	selp.b32 	%r311, 0, %r180, %p5;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	cp.async.wait_group 	2;
	bar.sync 	0;
	shl.b32 	%r181, %r311, 11;
	add.s32 	%r182, %r113, %r181;
	add.s32 	%r183, %r182, %r12;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r129, %r130, %r131, %r132}, [%r183+32768];
	add.s32 	%r184, %r182, %r13;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r141, %r142, %r143, %r144}, [%r184+32768];
	add.s32 	%r185, %r182, %r14;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r153, %r154, %r155, %r156}, [%r185+32768];
	add.s32 	%r186, %r182, %r15;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r161, %r162, %r163, %r164}, [%r186+32768];
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	shl.b32 	%r187, %r311, 14;
	add.s32 	%r188, %r113, %r187;
	add.s32 	%r189, %r188, %r16;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r133, %r134, %r145, %r146}, [%r189];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r135, %r136, %r151, %r152}, [%r189+8192];
	add.s32 	%r190, %r188, %r17;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r157, %r158, %r165, %r166}, [%r190];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r159, %r160, %r167, %r168}, [%r190+8192];
	.loc	1 338 36                        // sk06_mlp_down.py:338:36
	mov.b32 	%r137, %r128;
	mov.b32 	%r138, %r128;
	mov.b32 	%r139, %r128;
	mov.b32 	%r140, %r128;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r137, %r138, %r139, %r140 }, { %r129, %r130, %r131, %r132 }, { %r133, %r134 }, { %r137, %r138, %r139, %r140 };
	// end inline asm
	mov.b32 	%r147, %r128;
	mov.b32 	%r148, %r128;
	mov.b32 	%r149, %r128;
	mov.b32 	%r150, %r128;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r147, %r148, %r149, %r150 }, { %r129, %r130, %r131, %r132 }, { %r135, %r136 }, { %r147, %r148, %r149, %r150 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r137, %r138, %r139, %r140 }, { %r141, %r142, %r143, %r144 }, { %r145, %r146 }, { %r137, %r138, %r139, %r140 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r147, %r148, %r149, %r150 }, { %r141, %r142, %r143, %r144 }, { %r151, %r152 }, { %r147, %r148, %r149, %r150 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r137, %r138, %r139, %r140 }, { %r153, %r154, %r155, %r156 }, { %r157, %r158 }, { %r137, %r138, %r139, %r140 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r147, %r148, %r149, %r150 }, { %r153, %r154, %r155, %r156 }, { %r159, %r160 }, { %r147, %r148, %r149, %r150 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r137, %r138, %r139, %r140 }, { %r161, %r162, %r163, %r164 }, { %r165, %r166 }, { %r137, %r138, %r139, %r140 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r147, %r148, %r149, %r150 }, { %r161, %r162, %r163, %r164 }, { %r167, %r168 }, { %r147, %r148, %r149, %r150 };
	// end inline asm
	.loc	1 340 37                        // sk06_mlp_down.py:340:37
	mul.wide.s32 	%rd56, %r310, 4;
	add.s64 	%rd47, %rd8, %rd56;
	add.s64 	%rd48, %rd9, %rd56;
	add.s64 	%rd49, %rd10, %rd56;
	add.s64 	%rd50, %rd11, %rd56;
	.loc	1 340 27                        // sk06_mlp_down.py:340:27
	// begin inline asm
	mov.u32 %r169, 0x0;
	ld.global.b32 { %r169 }, [ %rd47 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r170, 0x0;
	ld.global.b32 { %r170 }, [ %rd48 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r171, 0x0;
	ld.global.b32 { %r171 }, [ %rd49 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r172, 0x0;
	ld.global.b32 { %r172 }, [ %rd50 + 0 ];
	// end inline asm
	.loc	1 341 23                        // sk06_mlp_down.py:341:23
	cvt.rn.f32.s32 	%r191, %r148;
	cvt.rn.f32.s32 	%r192, %r150;
	cvt.rn.f32.s32 	%r193, %r147;
	cvt.rn.f32.s32 	%r194, %r149;
	cvt.rn.f32.s32 	%r195, %r138;
	cvt.rn.f32.s32 	%r196, %r140;
	cvt.rn.f32.s32 	%r197, %r137;
	cvt.rn.f32.s32 	%r198, %r139;
	.loc	1 341 19                        // sk06_mlp_down.py:341:19
	fma.rn.f32 	%r315, %r169, %r198, %r315;
	fma.rn.f32 	%r313, %r169, %r197, %r313;
	fma.rn.f32 	%r316, %r170, %r196, %r316;
	fma.rn.f32 	%r314, %r170, %r195, %r314;
	fma.rn.f32 	%r319, %r171, %r194, %r319;
	fma.rn.f32 	%r317, %r171, %r193, %r317;
	fma.rn.f32 	%r320, %r172, %r192, %r320;
	fma.rn.f32 	%r318, %r172, %r191, %r318;
	.loc	1 345 18                        // sk06_mlp_down.py:345:18
	add.s64 	%rd51, %rd18, %rd70;
	add.s64 	%rd52, %rd17, %rd70;
	add.s64 	%rd53, %rd16, %rd70;
	add.s64 	%rd54, %rd15, %rd70;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	add.s64 	%rd55, %rd14, %rd70;
	add.s32 	%r199, %r312, 1;
	setp.gt.s32 	%p6, %r199, 1;
	selp.b32 	%r312, 0, %r199, %p6;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	shl.b32 	%r200, %r312, 11;
	bar.sync 	0;
	add.s32 	%r201, %r11, %r200;
	add.s32 	%r173, %r201, 32768;
	selp.b32 	%r174, 8, 0, %p4;
	// begin inline asm
	cp.async.ca.shared.global [ %r173 + 0 ], [ %rd51 + 0 ], 0x8, %r174;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	shl.b32 	%r202, %r312, 14;
	add.s32 	%r175, %r27, %r202;
	selp.b32 	%r176, 16, 0, %p4;
	// begin inline asm
	cp.async.cg.shared.global [ %r175 + 0 ], [ %rd52 + 0 ], 0x10, %r176;
	// end inline asm
	add.s32 	%r177, %r175, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r177 + 0 ], [ %rd53 + 0 ], 0x10, %r176;
	// end inline asm
	add.s32 	%r178, %r175, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r178 + 0 ], [ %rd54 + 0 ], 0x10, %r176;
	// end inline asm
	add.s32 	%r179, %r175, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r179 + 0 ], [ %rd55 + 0 ], 0x10, %r176;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	add.s64 	%rd71, %rd71, 1;
	add.s64 	%rd70, %rd70, 128;
	add.s32 	%r310, %r310, %r24;
	setp.ne.b64 	%p7, %rd13, %rd70;
	@%p7 bra 	$L__BB0_2;
$L__BB0_3:                              // %._crit_edge
	.loc	1 0 23                          // sk06_mlp_down.py:0:23
	cvt.u32.u64 	%r218, %rd2;
	.loc	1 323 45                        // sk06_mlp_down.py:323:45
	or.b32 	%r219, %r6, %r218;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r220, %r219, 7;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r221, %r220, %r19;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r222, %r219, 6;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r223, %r222, %r19;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r224, %r219, 5;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r225, %r224, %r19;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r226, %r219, 4;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r227, %r226, %r19;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r228, %r219, 3;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r229, %r228, %r19;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r230, %r219, 2;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r231, %r230, %r19;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r232, %r219, 1;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r233, %r232, %r19;
	rem.s32 	%r234, %r219, %r19;
	.loc	1 322 45                        // sk06_mlp_down.py:322:45
	and.b32 	%r235, %r2, 28;
	bfe.u32 	%r236, %r2, 2, 3;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r237, %r236, %r1;
	or.b32 	%r238, %r237, 8;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r239, %r238, %r18;
	rem.s32 	%r240, %r237, %r18;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 347 38                        // sk06_mlp_down.py:347:38
	mad.wide.s32 	%rd57, %r240, 4, %rd23;
	mad.wide.s32 	%rd58, %r239, 4, %rd23;
	.loc	1 347 24                        // sk06_mlp_down.py:347:24
	// begin inline asm
	mov.u32 %r203, 0x0;
	ld.global.b32 { %r203 }, [ %rd57 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r204, 0x0;
	ld.global.b32 { %r204 }, [ %rd58 + 0 ];
	// end inline asm
	.loc	1 350 49                        // sk06_mlp_down.py:350:49
	mul.lo.s32 	%r241, %r5, %r22;
	.loc	1 350 31                        // sk06_mlp_down.py:350:31
	mad.wide.s32 	%rd68, %r241, 2, %rd22;
	.loc	1 350 82                        // sk06_mlp_down.py:350:82
	mul.lo.s32 	%r242, %r234, %r23;
	mul.lo.s32 	%r243, %r233, %r23;
	mul.lo.s32 	%r244, %r231, %r23;
	mul.lo.s32 	%r245, %r229, %r23;
	mul.lo.s32 	%r246, %r227, %r23;
	mul.lo.s32 	%r247, %r225, %r23;
	mul.lo.s32 	%r248, %r223, %r23;
	mul.lo.s32 	%r249, %r221, %r23;
	.loc	1 350 64                        // sk06_mlp_down.py:350:64
	mad.wide.s32 	%rd59, %r242, 2, %rd68;
	mad.wide.s32 	%rd60, %r243, 2, %rd68;
	mad.wide.s32 	%rd61, %r244, 2, %rd68;
	mad.wide.s32 	%rd62, %r245, 2, %rd68;
	mad.wide.s32 	%rd63, %r246, 2, %rd68;
	mad.wide.s32 	%rd64, %r247, 2, %rd68;
	mad.wide.s32 	%rd65, %r248, 2, %rd68;
	mad.wide.s32 	%rd66, %r249, 2, %rd68;
	.loc	1 350 19                        // sk06_mlp_down.py:350:19
	// begin inline asm
	mov.u16 %rs1, 0x0;
	ld.global.b16 { %rs1 }, [ %rd59 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs2, 0x0;
	ld.global.b16 { %rs2 }, [ %rd60 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs3, 0x0;
	ld.global.b16 { %rs3 }, [ %rd61 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs4, 0x0;
	ld.global.b16 { %rs4 }, [ %rd62 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs5, 0x0;
	ld.global.b16 { %rs5 }, [ %rd63 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs6, 0x0;
	ld.global.b16 { %rs6 }, [ %rd64 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs7, 0x0;
	ld.global.b16 { %rs7 }, [ %rd65 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs8, 0x0;
	ld.global.b16 { %rs8 }, [ %rd66 + 0 ];
	// end inline asm
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	and.b32 	%r250, %r2, 120;
	shl.b32 	%r251, %r250, 5;
	or.b32 	%r252, %r251, %r309;
	xor.b32 	%r253, %r252, %r3;
	add.s32 	%r205, %r113, %r253;
	mov.b32 	%r206, {%rs1, %rs2};
	mov.b32 	%r207, {%rs3, %rs4};
	mov.b32 	%r208, {%rs5, %rs6};
	mov.b32 	%r209, {%rs7, %rs8};
	// begin inline asm
	st.shared.v4.b32 [ %r205 + 0 ], { %r206, %r207, %r208, %r209 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r254, %r10, 9;
	shl.b32 	%r255, %r2, 4;
	and.b32 	%r256, %r255, 496;
	shr.u32 	%r257, %r8, 1;
	xor.b32 	%r258, %r256, %r257;
	add.s32 	%r259, %r113, %r254;
	add.s32 	%r260, %r259, %r258;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r261, %r262, %r263, %r264}, [%r260];
	mov.b32 	{%rs17, %rs18}, %r261;
	mov.b32 	{%rs19, %rs20}, %r262;
	mov.b32 	{%rs21, %rs22}, %r263;
	mov.b32 	{%rs23, %rs24}, %r264;
	cvt.f32.bf16 	%r265, %rs17;
	cvt.f32.bf16 	%r266, %rs18;
	cvt.f32.bf16 	%r267, %rs19;
	cvt.f32.bf16 	%r268, %rs20;
	cvt.f32.bf16 	%r269, %rs21;
	cvt.f32.bf16 	%r270, %rs22;
	cvt.f32.bf16 	%r271, %rs23;
	cvt.f32.bf16 	%r272, %rs24;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r273, %r313, %r203, %r265;
	fma.rn.f32 	%r274, %r314, %r203, %r266;
	fma.rn.f32 	%r275, %r315, %r204, %r267;
	fma.rn.f32 	%r276, %r316, %r204, %r268;
	fma.rn.f32 	%r277, %r317, %r203, %r269;
	fma.rn.f32 	%r278, %r318, %r203, %r270;
	fma.rn.f32 	%r279, %r319, %r204, %r271;
	fma.rn.f32 	%r280, %r320, %r204, %r272;
	.loc	1 357 31                        // sk06_mlp_down.py:357:31
	setp.lt.s32 	%p9, %r4, %r18;
	.loc	1 357 54                        // sk06_mlp_down.py:357:54
	setp.lt.s32 	%p10, %r219, %r19;
	.loc	1 357 37                        // sk06_mlp_down.py:357:37
	and.pred 	%p8, %p9, %p10;
	.loc	1 355 35                        // sk06_mlp_down.py:355:35
	mul.lo.s32 	%r281, %r4, %r21;
	.loc	1 355 18                        // sk06_mlp_down.py:355:18
	mad.wide.s32 	%rd69, %r281, 2, %rd21;
	.loc	1 355 50                        // sk06_mlp_down.py:355:50
	mad.wide.s32 	%rd67, %r219, 2, %rd69;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16.f32 	%rs9, %r273;
	cvt.rn.bf16.f32 	%rs10, %r274;
	cvt.rn.bf16.f32 	%rs11, %r275;
	cvt.rn.bf16.f32 	%rs12, %r276;
	cvt.rn.bf16.f32 	%rs13, %r277;
	cvt.rn.bf16.f32 	%rs14, %r278;
	cvt.rn.bf16.f32 	%rs15, %r279;
	cvt.rn.bf16.f32 	%rs16, %r280;
	bar.sync 	0;
	shl.b32 	%r282, %r2, 5;
	and.b32 	%r283, %r282, 768;
	shl.b32 	%r284, %r235, 1;
	and.b32 	%r285, %r2, 1;
	neg.s32 	%r286, %r285;
	and.b32 	%r287, %r286, 1088;
	bfe.s32 	%r288, %r2, 1, 1;
	and.b32 	%r289, %r288, 2052;
	or.b32 	%r290, %r283, %r284;
	or.b32 	%r291, %r287, %r290;
	xor.b32 	%r292, %r291, %r257;
	or.b32 	%r293, %r292, %r289;
	add.s32 	%r210, %r113, %r293;
	// begin inline asm
	st.shared.v2.b16 [ %r210 + 0 ], { %rs9, %rs10 };
	// end inline asm
	add.s32 	%r211, %r210, 128;
	// begin inline asm
	st.shared.v2.b16 [ %r211 + 0 ], { %rs11, %rs12 };
	// end inline asm
	xor.b32 	%r294, %r293, 4;
	add.s32 	%r212, %r113, %r294;
	// begin inline asm
	st.shared.v2.b16 [ %r212 + 0 ], { %rs13, %rs14 };
	// end inline asm
	add.s32 	%r213, %r212, 128;
	// begin inline asm
	st.shared.v2.b16 [ %r213 + 0 ], { %rs15, %rs16 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r295, %r2, 3;
	and.b32 	%r296, %r295, 768;
	shr.u32 	%r297, %r250, 1;
	and.b32 	%r298, %r2, 128;
	or.b32 	%r299, %r309, %r296;
	xor.b32 	%r300, %r299, %r297;
	or.b32 	%r301, %r300, %r298;
	add.s32 	%r302, %r113, %r301;
	ld.shared.b32 	%r214, [%r302];
	xor.b32 	%r303, %r301, 64;
	add.s32 	%r304, %r113, %r303;
	ld.shared.b32 	%r215, [%r304+1024];
	xor.b32 	%r305, %r301, 4;
	add.s32 	%r306, %r113, %r305;
	ld.shared.b32 	%r216, [%r306+2048];
	xor.b32 	%r307, %r301, 68;
	add.s32 	%r308, %r113, %r307;
	ld.shared.b32 	%r217, [%r308+3072];
	.loc	1 356 8                         // sk06_mlp_down.py:356:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd67 + 0 ], { %r214, %r215, %r216, %r217 };
	// end inline asm
	.loc	1 354 4                         // sk06_mlp_down.py:354:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/workspace/vllm/_genesis/kernels/sk06_mlp_down.py"
	.file	2 "/usr/local/lib/python3.12/dist-packages/triton/language/standard.py"
	.section	.debug_abbrev
	{
.b8 1                                   // Abbreviation Code
.b8 17                                  // DW_TAG_compile_unit
.b8 1                                   // DW_CHILDREN_yes
.b8 37                                  // DW_AT_producer
.b8 8                                   // DW_FORM_string
.b8 19                                  // DW_AT_language
.b8 5                                   // DW_FORM_data2
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 16                                  // DW_AT_stmt_list
.b8 6                                   // DW_FORM_data4
.b8 27                                  // DW_AT_comp_dir
.b8 8                                   // DW_FORM_string
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 2                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 0                                   // DW_CHILDREN_no
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 32                                  // DW_AT_inline
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 3                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 1                                   // DW_CHILDREN_yes
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 4                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 0                                   // DW_CHILDREN_no
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 5                                   // DW_FORM_data2
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 168                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0xa1 DW_TAG_compile_unit
.b8 116                                 // DW_AT_producer
.b8 114
.b8 105
.b8 116
.b8 111
.b8 110
.b8 0
.b8 2                                   // DW_AT_language
.b8 0
.b8 115                                 // DW_AT_name
.b8 107
.b8 48
.b8 54
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 100
.b8 111
.b8 119
.b8 110
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 119
.b8 111
.b8 114
.b8 107
.b8 115
.b8 112
.b8 97
.b8 99
.b8 101
.b8 47
.b8 118
.b8 108
.b8 108
.b8 109
.b8 47
.b8 95
.b8 103
.b8 101
.b8 110
.b8 101
.b8 115
.b8 105
.b8 115
.b8 47
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 115
.b8 0
.b8 2                                   // Abbrev [2] 0x4b:0x18 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 54
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 100
.b8 111
.b8 119
.b8 110
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x63:0x48 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 75                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x78:0x19 DW_TAG_inlined_subroutine
.b32 75                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 58                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x91:0x19 DW_TAG_inlined_subroutine
.b32 75                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 59                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_3 = _Nativo(
    "sk06_mlp_down/tile16x128x128_shift1_abi16",
    _PTX_3, "_sk06_mlp_down_kernel",
    warps=8, shared=36864,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 17, 18],
    horneado={11: 1, 12: 1, 15: 1, 19: 1, 20: 16, 21: 128, 22: 128, 23: 8, 24: 128, 25: True},
    div16=[8, 9, 10, 13, 14, 16, 17],
)

_PTX_4 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk06_mlp_down_kernel   // -- Begin function _sk06_mlp_down_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk06_mlp_down_kernel
.visible .entry _sk06_mlp_down_kernel(
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_6,
	.param .u32 _sk06_mlp_down_kernel_param_7,
	.param .u32 _sk06_mlp_down_kernel_param_8,
	.param .u32 _sk06_mlp_down_kernel_param_9,
	.param .u32 _sk06_mlp_down_kernel_param_10,
	.param .u32 _sk06_mlp_down_kernel_param_11,
	.param .u32 _sk06_mlp_down_kernel_param_12,
	.param .u32 _sk06_mlp_down_kernel_param_13,
	.param .u32 _sk06_mlp_down_kernel_param_14,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_16
)
.reqntid 128
{
	.reg .pred 	%p<26>;
	.reg .b16 	%rs<65>;
	.reg .b32 	%r<904>;
	.reg .b64 	%rd<137>;
	.loc	1 304 0                         // sk06_mlp_down.py:304:0
$L__func_begin0:
	.loc	1 304 0                         // sk06_mlp_down.py:304:0

// %bb.0:
	ld.param.b32 	%r24, [_sk06_mlp_down_kernel_param_13];
	ld.param.b32 	%r23, [_sk06_mlp_down_kernel_param_12];
	ld.param.b32 	%r22, [_sk06_mlp_down_kernel_param_8];
	ld.param.b32 	%r21, [_sk06_mlp_down_kernel_param_7];
	ld.param.b64 	%rd19, [_sk06_mlp_down_kernel_param_5];
	ld.param.b64 	%rd18, [_sk06_mlp_down_kernel_param_4];
	ld.param.b64 	%rd17, [_sk06_mlp_down_kernel_param_3];
	ld.param.b64 	%rd16, [_sk06_mlp_down_kernel_param_2];
	ld.param.b64 	%rd15, [_sk06_mlp_down_kernel_param_1];
	ld.param.b64 	%rd14, [_sk06_mlp_down_kernel_param_0];
$L__tmp0:
	.loc	1 313 24                        // sk06_mlp_down.py:313:24
	mov.u32 	%r64, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk06_mlp_down.py:314:27 ]
	add.s32 	%r65, %r21, 63;
	.loc	2 43 30                         // standard.py:43:30 @[ sk06_mlp_down.py:314:27 ]
	shr.s32 	%r66, %r65, 31;
	shr.u32 	%r67, %r66, 26;
	add.s32 	%r68, %r65, %r67;
	shr.s32 	%r69, %r68, 6;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk06_mlp_down.py:315:27 ]
	add.s32 	%r70, %r22, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk06_mlp_down.py:315:27 ]
	shr.s32 	%r71, %r70, 31;
	shr.u32 	%r72, %r71, 25;
	add.s32 	%r73, %r70, %r72;
	shr.s32 	%r74, %r73, 7;
$L__tmp3:
	.loc	1 316 29                        // sk06_mlp_down.py:316:29
	shl.b32 	%r75, %r74, 3;
	.loc	1 317 22                        // sk06_mlp_down.py:317:22
	div.s32 	%r76, %r64, %r75;
	.loc	1 317 38                        // sk06_mlp_down.py:317:38
	shl.b32 	%r77, %r76, 3;
	ld.param.b32 	%r78, [_sk06_mlp_down_kernel_param_9];
	.loc	1 318 30                        // sk06_mlp_down.py:318:30
	sub.s32 	%r79, %r69, %r77;
	ld.param.b32 	%r80, [_sk06_mlp_down_kernel_param_10];
	.loc	1 318 39                        // sk06_mlp_down.py:318:39
	min.s32 	%r81, %r79, 8;
	ld.param.b32 	%r82, [_sk06_mlp_down_kernel_param_11];
	.loc	1 319 30                        // sk06_mlp_down.py:319:30
	mul.lo.s32 	%r83, %r76, %r75;
	sub.s32 	%r84, %r64, %r83;
	.loc	1 320 36                        // sk06_mlp_down.py:320:36
	div.s32 	%r85, %r84, %r81;
	.loc	1 319 46                        // sk06_mlp_down.py:319:46
	mul.lo.s32 	%r86, %r85, %r81;
	sub.s32 	%r87, %r84, %r86;
	.loc	1 319 23                        // sk06_mlp_down.py:319:23
	add.s32 	%r88, %r87, %r77;
	.loc	1 322 22                        // sk06_mlp_down.py:322:22
	shl.b32 	%r1, %r88, 6;
	.loc	1 322 45                        // sk06_mlp_down.py:322:45
	mov.u32 	%r2, %tid.x;
	and.b32 	%r3, %r2, 120;
	bfe.u32 	%r89, %r2, 3, 4;
	or.b32 	%r90, %r89, 16;
	or.b32 	%r91, %r89, 32;
	or.b32 	%r92, %r89, 48;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r93, %r1, %r89;
	or.b32 	%r94, %r1, %r90;
	or.b32 	%r95, %r1, %r91;
	or.b32 	%r96, %r1, %r92;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r4, %r93, %r21;
	rem.s32 	%r5, %r94, %r21;
	rem.s32 	%r6, %r95, %r21;
	rem.s32 	%r7, %r96, %r21;
	.loc	1 323 22                        // sk06_mlp_down.py:323:22
	shl.b32 	%r8, %r85, 7;
	.loc	1 323 45                        // sk06_mlp_down.py:323:45
	and.b32 	%r9, %r2, 7;
	shl.b32 	%r97, %r9, 4;
	and.b32 	%r10, %r2, 15;
	and.b32 	%r11, %r2, 127;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r98, %r8, %r89;
	or.b32 	%r99, %r8, %r90;
	or.b32 	%r100, %r8, %r91;
	or.b32 	%r101, %r8, %r92;
	or.b32 	%r102, %r98, 64;
	or.b32 	%r103, %r98, 80;
	or.b32 	%r104, %r98, 96;
	or.b32 	%r105, %r98, 112;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r106, %r98, %r22;
	rem.s32 	%r107, %r99, %r22;
	rem.s32 	%r108, %r100, %r22;
	rem.s32 	%r109, %r101, %r22;
	rem.s32 	%r110, %r102, %r22;
	rem.s32 	%r111, %r103, %r22;
	rem.s32 	%r112, %r104, %r22;
	rem.s32 	%r113, %r105, %r22;
	.loc	1 326 39                        // sk06_mlp_down.py:326:39
	mul.lo.s32 	%r114, %r4, %r80;
	mul.lo.s32 	%r115, %r5, %r80;
	mul.lo.s32 	%r116, %r6, %r80;
	mul.lo.s32 	%r117, %r7, %r80;
	.loc	1 326 21                        // sk06_mlp_down.py:326:21
	cvt.s64.s32 	%rd1, %r114;
	add.s64 	%rd56, %rd14, %rd1;
	cvt.s64.s32 	%rd2, %r115;
	add.s64 	%rd57, %rd14, %rd2;
	cvt.s64.s32 	%rd3, %r116;
	add.s64 	%rd58, %rd14, %rd3;
	cvt.s64.s32 	%rd4, %r117;
	add.s64 	%rd59, %rd14, %rd4;
	.loc	1 326 51                        // sk06_mlp_down.py:326:51
	cvt.u64.u32 	%rd5, %r97;
	add.s64 	%rd20, %rd56, %rd5;
	add.s64 	%rd21, %rd57, %rd5;
	add.s64 	%rd22, %rd58, %rd5;
	add.s64 	%rd23, %rd59, %rd5;
	.loc	1 327 21                        // sk06_mlp_down.py:327:21
	add.s64 	%rd60, %rd15, %rd5;
	.loc	1 327 69                        // sk06_mlp_down.py:327:69
	mul.lo.s32 	%r118, %r106, %r82;
	mul.lo.s32 	%r119, %r107, %r82;
	mul.lo.s32 	%r120, %r108, %r82;
	mul.lo.s32 	%r121, %r109, %r82;
	mul.lo.s32 	%r122, %r110, %r82;
	mul.lo.s32 	%r123, %r111, %r82;
	mul.lo.s32 	%r124, %r112, %r82;
	mul.lo.s32 	%r125, %r113, %r82;
	.loc	1 327 51                        // sk06_mlp_down.py:327:51
	cvt.s64.s32 	%rd6, %r118;
	add.s64 	%rd24, %rd60, %rd6;
	cvt.s64.s32 	%rd7, %r119;
	add.s64 	%rd25, %rd60, %rd7;
	cvt.s64.s32 	%rd8, %r120;
	add.s64 	%rd26, %rd60, %rd8;
	cvt.s64.s32 	%rd9, %r121;
	add.s64 	%rd27, %rd60, %rd9;
	cvt.s64.s32 	%rd10, %r122;
	add.s64 	%rd28, %rd60, %rd10;
	cvt.s64.s32 	%rd11, %r123;
	add.s64 	%rd29, %rd60, %rd11;
	cvt.s64.s32 	%rd12, %r124;
	add.s64 	%rd30, %rd60, %rd12;
	cvt.s64.s32 	%rd13, %r125;
	add.s64 	%rd31, %rd60, %rd13;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	setp.gt.s32 	%p1, %r78, 127;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	shl.b32 	%r129, %r11, 4;
	and.b32 	%r13, %r2, 56;
	shl.b32 	%r130, %r13, 1;
	xor.b32 	%r131, %r129, %r130;
	mov.b32 	%r132, global_smem;
	add.s32 	%r30, %r132, %r131;
	add.s32 	%r25, %r30, 49152;
	selp.b32 	%r26, 16, 0, %p1;
	// begin inline asm
	cp.async.cg.shared.global [ %r25 + 0 ], [ %rd20 + 0 ], 0x10, %r26;
	// end inline asm
	add.s32 	%r27, %r30, 51200;
	// begin inline asm
	cp.async.cg.shared.global [ %r27 + 0 ], [ %rd21 + 0 ], 0x10, %r26;
	// end inline asm
	add.s32 	%r28, %r30, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r28 + 0 ], [ %rd22 + 0 ], 0x10, %r26;
	// end inline asm
	add.s32 	%r29, %r30, 55296;
	// begin inline asm
	cp.async.cg.shared.global [ %r29 + 0 ], [ %rd23 + 0 ], 0x10, %r26;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	// begin inline asm
	cp.async.cg.shared.global [ %r30 + 0 ], [ %rd24 + 0 ], 0x10, %r26;
	// end inline asm
	add.s32 	%r31, %r30, 2048;
	// begin inline asm
	cp.async.cg.shared.global [ %r31 + 0 ], [ %rd25 + 0 ], 0x10, %r26;
	// end inline asm
	add.s32 	%r32, %r30, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r32 + 0 ], [ %rd26 + 0 ], 0x10, %r26;
	// end inline asm
	add.s32 	%r33, %r30, 6144;
	// begin inline asm
	cp.async.cg.shared.global [ %r33 + 0 ], [ %rd27 + 0 ], 0x10, %r26;
	// end inline asm
	add.s32 	%r34, %r30, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r34 + 0 ], [ %rd28 + 0 ], 0x10, %r26;
	// end inline asm
	add.s32 	%r35, %r30, 10240;
	// begin inline asm
	cp.async.cg.shared.global [ %r35 + 0 ], [ %rd29 + 0 ], 0x10, %r26;
	// end inline asm
	add.s32 	%r36, %r30, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r36 + 0 ], [ %rd30 + 0 ], 0x10, %r26;
	// end inline asm
	add.s32 	%r37, %r30, 14336;
	// begin inline asm
	cp.async.cg.shared.global [ %r37 + 0 ], [ %rd31 + 0 ], 0x10, %r26;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	setp.gt.s32 	%p2, %r78, 255;
	.loc	1 344 18                        // sk06_mlp_down.py:344:18
	add.s64 	%rd32, %rd20, 128;
	add.s64 	%rd33, %rd21, 128;
	add.s64 	%rd34, %rd22, 128;
	add.s64 	%rd35, %rd23, 128;
	.loc	1 345 18                        // sk06_mlp_down.py:345:18
	add.s64 	%rd36, %rd24, 128;
	add.s64 	%rd37, %rd25, 128;
	add.s64 	%rd38, %rd26, 128;
	add.s64 	%rd39, %rd27, 128;
	add.s64 	%rd40, %rd28, 128;
	add.s64 	%rd41, %rd29, 128;
	add.s64 	%rd42, %rd30, 128;
	add.s64 	%rd43, %rd31, 128;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	bar.sync 	0;
	add.s32 	%r38, %r30, 57344;
	selp.b32 	%r39, 16, 0, %p2;
	// begin inline asm
	cp.async.cg.shared.global [ %r38 + 0 ], [ %rd32 + 0 ], 0x10, %r39;
	// end inline asm
	add.s32 	%r40, %r30, 59392;
	// begin inline asm
	cp.async.cg.shared.global [ %r40 + 0 ], [ %rd33 + 0 ], 0x10, %r39;
	// end inline asm
	add.s32 	%r41, %r30, 61440;
	// begin inline asm
	cp.async.cg.shared.global [ %r41 + 0 ], [ %rd34 + 0 ], 0x10, %r39;
	// end inline asm
	add.s32 	%r42, %r30, 63488;
	// begin inline asm
	cp.async.cg.shared.global [ %r42 + 0 ], [ %rd35 + 0 ], 0x10, %r39;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	add.s32 	%r43, %r30, 16384;
	// begin inline asm
	cp.async.cg.shared.global [ %r43 + 0 ], [ %rd36 + 0 ], 0x10, %r39;
	// end inline asm
	add.s32 	%r44, %r30, 18432;
	// begin inline asm
	cp.async.cg.shared.global [ %r44 + 0 ], [ %rd37 + 0 ], 0x10, %r39;
	// end inline asm
	add.s32 	%r45, %r30, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r45 + 0 ], [ %rd38 + 0 ], 0x10, %r39;
	// end inline asm
	add.s32 	%r46, %r30, 22528;
	// begin inline asm
	cp.async.cg.shared.global [ %r46 + 0 ], [ %rd39 + 0 ], 0x10, %r39;
	// end inline asm
	add.s32 	%r47, %r30, 24576;
	// begin inline asm
	cp.async.cg.shared.global [ %r47 + 0 ], [ %rd40 + 0 ], 0x10, %r39;
	// end inline asm
	add.s32 	%r48, %r30, 26624;
	// begin inline asm
	cp.async.cg.shared.global [ %r48 + 0 ], [ %rd41 + 0 ], 0x10, %r39;
	// end inline asm
	add.s32 	%r49, %r30, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r49 + 0 ], [ %rd42 + 0 ], 0x10, %r39;
	// end inline asm
	add.s32 	%r50, %r30, 30720;
	// begin inline asm
	cp.async.cg.shared.global [ %r50 + 0 ], [ %rd43 + 0 ], 0x10, %r39;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	setp.gt.s32 	%p3, %r78, 383;
	.loc	1 344 18                        // sk06_mlp_down.py:344:18
	add.s64 	%rd44, %rd20, 256;
	add.s64 	%rd45, %rd21, 256;
	add.s64 	%rd46, %rd22, 256;
	add.s64 	%rd47, %rd23, 256;
	.loc	1 345 18                        // sk06_mlp_down.py:345:18
	add.s64 	%rd48, %rd24, 256;
	add.s64 	%rd49, %rd25, 256;
	add.s64 	%rd50, %rd26, 256;
	add.s64 	%rd51, %rd27, 256;
	add.s64 	%rd52, %rd28, 256;
	add.s64 	%rd53, %rd29, 256;
	add.s64 	%rd54, %rd30, 256;
	add.s64 	%rd55, %rd31, 256;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	bar.sync 	0;
	add.s32 	%r51, %r30, 65536;
	selp.b32 	%r52, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r51 + 0 ], [ %rd44 + 0 ], 0x10, %r52;
	// end inline asm
	add.s32 	%r53, %r30, 67584;
	// begin inline asm
	cp.async.cg.shared.global [ %r53 + 0 ], [ %rd45 + 0 ], 0x10, %r52;
	// end inline asm
	add.s32 	%r54, %r30, 69632;
	// begin inline asm
	cp.async.cg.shared.global [ %r54 + 0 ], [ %rd46 + 0 ], 0x10, %r52;
	// end inline asm
	add.s32 	%r55, %r30, 71680;
	// begin inline asm
	cp.async.cg.shared.global [ %r55 + 0 ], [ %rd47 + 0 ], 0x10, %r52;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	add.s32 	%r56, %r30, 32768;
	// begin inline asm
	cp.async.cg.shared.global [ %r56 + 0 ], [ %rd48 + 0 ], 0x10, %r52;
	// end inline asm
	add.s32 	%r57, %r30, 34816;
	// begin inline asm
	cp.async.cg.shared.global [ %r57 + 0 ], [ %rd49 + 0 ], 0x10, %r52;
	// end inline asm
	add.s32 	%r58, %r30, 36864;
	// begin inline asm
	cp.async.cg.shared.global [ %r58 + 0 ], [ %rd50 + 0 ], 0x10, %r52;
	// end inline asm
	add.s32 	%r59, %r30, 38912;
	// begin inline asm
	cp.async.cg.shared.global [ %r59 + 0 ], [ %rd51 + 0 ], 0x10, %r52;
	// end inline asm
	add.s32 	%r60, %r30, 40960;
	// begin inline asm
	cp.async.cg.shared.global [ %r60 + 0 ], [ %rd52 + 0 ], 0x10, %r52;
	// end inline asm
	add.s32 	%r61, %r30, 43008;
	// begin inline asm
	cp.async.cg.shared.global [ %r61 + 0 ], [ %rd53 + 0 ], 0x10, %r52;
	// end inline asm
	add.s32 	%r62, %r30, 45056;
	// begin inline asm
	cp.async.cg.shared.global [ %r62 + 0 ], [ %rd54 + 0 ], 0x10, %r52;
	// end inline asm
	add.s32 	%r63, %r30, 47104;
	// begin inline asm
	cp.async.cg.shared.global [ %r63 + 0 ], [ %rd55 + 0 ], 0x10, %r52;
	// end inline asm
	cp.async.commit_group;
	cvt.u32.u64 	%r835, %rd5;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	@%p1 bra 	$L__BB0_2;
	bra.uni 	$L__BB0_1;
$L__BB0_2:                              // %.lr.ph
	.loc	1 0 23                          // sk06_mlp_down.py:0:23
	shr.s32 	%r126, %r78, 31;
	shr.u32 	%r127, %r126, 25;
	add.s32 	%r128, %r78, %r127;
	shr.s32 	%r12, %r128, 7;
	add.s32 	%r14, %r12, -3;
	shl.b32 	%r133, %r10, 7;
	and.b32 	%r839, %r2, 16;
	or.b32 	%r134, %r133, %r835;
	xor.b32 	%r15, %r134, %r839;
	xor.b32 	%r16, %r15, 32;
	xor.b32 	%r17, %r15, 64;
	xor.b32 	%r18, %r15, 96;
	shl.b32 	%r135, %r9, 7;
	shl.b32 	%r136, %r2, 5;
	and.b32 	%r137, %r136, 3072;
	shl.b32 	%r138, %r2, 1;
	and.b32 	%r139, %r138, 48;
	or.b32 	%r140, %r135, %r137;
	xor.b32 	%r141, %r835, %r139;
	or.b32 	%r19, %r140, %r141;
	xor.b32 	%r20, %r19, 64;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	add.s64 	%rd61, %rd13, %rd15;
	add.s64 	%rd136, %rd61, 384;
	add.s64 	%rd62, %rd12, %rd15;
	add.s64 	%rd135, %rd62, 384;
	add.s64 	%rd63, %rd11, %rd15;
	add.s64 	%rd134, %rd63, 384;
	add.s64 	%rd64, %rd10, %rd15;
	add.s64 	%rd133, %rd64, 384;
	add.s64 	%rd65, %rd9, %rd15;
	add.s64 	%rd132, %rd65, 384;
	add.s64 	%rd66, %rd8, %rd15;
	add.s64 	%rd131, %rd66, 384;
	add.s64 	%rd67, %rd7, %rd15;
	add.s64 	%rd130, %rd67, 384;
	add.s64 	%rd68, %rd6, %rd15;
	add.s64 	%rd129, %rd68, 384;
	add.s64 	%rd69, %rd4, %rd14;
	add.s64 	%rd128, %rd69, 384;
	add.s64 	%rd70, %rd3, %rd14;
	add.s64 	%rd127, %rd70, 384;
	add.s64 	%rd71, %rd2, %rd14;
	add.s64 	%rd126, %rd71, 384;
	add.s64 	%rd72, %rd1, %rd14;
	add.s64 	%rd125, %rd72, 384;
	mov.b32 	%r840, 0f00000000;
	mov.b32 	%r142, 0;
	mov.b32 	%r837, 2;
	mov.b32 	%r836, -1;
	mov.b32 	%r838, %r142;
	mov.b32 	%r841, %r840;
	mov.b32 	%r842, %r840;
	mov.b32 	%r843, %r840;
	mov.b32 	%r844, %r840;
	mov.b32 	%r845, %r840;
	mov.b32 	%r846, %r840;
	mov.b32 	%r847, %r840;
	mov.b32 	%r848, %r840;
	mov.b32 	%r849, %r840;
	mov.b32 	%r850, %r840;
	mov.b32 	%r851, %r840;
	mov.b32 	%r852, %r840;
	mov.b32 	%r853, %r840;
	mov.b32 	%r854, %r840;
	mov.b32 	%r855, %r840;
	mov.b32 	%r856, %r840;
	mov.b32 	%r857, %r840;
	mov.b32 	%r858, %r840;
	mov.b32 	%r859, %r840;
	mov.b32 	%r860, %r840;
	mov.b32 	%r861, %r840;
	mov.b32 	%r862, %r840;
	mov.b32 	%r863, %r840;
	mov.b32 	%r864, %r840;
	mov.b32 	%r865, %r840;
	mov.b32 	%r866, %r840;
	mov.b32 	%r867, %r840;
	mov.b32 	%r868, %r840;
	mov.b32 	%r869, %r840;
	mov.b32 	%r870, %r840;
	mov.b32 	%r871, %r840;
	mov.b32 	%r872, %r840;
	mov.b32 	%r873, %r840;
	mov.b32 	%r874, %r840;
	mov.b32 	%r875, %r840;
	mov.b32 	%r876, %r840;
	mov.b32 	%r877, %r840;
	mov.b32 	%r878, %r840;
	mov.b32 	%r879, %r840;
	mov.b32 	%r880, %r840;
	mov.b32 	%r881, %r840;
	mov.b32 	%r882, %r840;
	mov.b32 	%r883, %r840;
	mov.b32 	%r884, %r840;
	mov.b32 	%r885, %r840;
	mov.b32 	%r886, %r840;
	mov.b32 	%r887, %r840;
	mov.b32 	%r888, %r840;
	mov.b32 	%r889, %r840;
	mov.b32 	%r890, %r840;
	mov.b32 	%r891, %r840;
	mov.b32 	%r892, %r840;
	mov.b32 	%r893, %r840;
	mov.b32 	%r894, %r840;
	mov.b32 	%r895, %r840;
	mov.b32 	%r896, %r840;
	mov.b32 	%r897, %r840;
	mov.b32 	%r898, %r840;
	mov.b32 	%r899, %r840;
	mov.b32 	%r900, %r840;
	mov.b32 	%r901, %r840;
	mov.b32 	%r902, %r840;
	mov.b32 	%r903, %r840;
$L__BB0_3:                              // =>This Inner Loop Header: Depth=1
	setp.lt.s32 	%p4, %r838, %r14;
	add.s32 	%r316, %r836, 1;
	setp.gt.s32 	%p5, %r316, 2;
	selp.b32 	%r836, 0, %r316, %p5;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	cp.async.wait_group 	4;
	bar.sync 	0;
	shl.b32 	%r317, %r836, 13;
	add.s32 	%r318, %r132, %r317;
	add.s32 	%r319, %r318, %r15;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r143, %r144, %r145, %r146}, [%r319+49152];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r155, %r156, %r157, %r158}, [%r319+51200];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r159, %r160, %r161, %r162}, [%r319+53248];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r163, %r164, %r165, %r166}, [%r319+55296];
	add.s32 	%r320, %r318, %r16;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r171, %r172, %r173, %r174}, [%r320+49152];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r199, %r200, %r201, %r202}, [%r320+51200];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r219, %r220, %r221, %r222}, [%r320+53248];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r239, %r240, %r241, %r242}, [%r320+55296];
	add.s32 	%r321, %r318, %r17;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r255, %r256, %r257, %r258}, [%r321+49152];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r267, %r268, %r269, %r270}, [%r321+51200];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r271, %r272, %r273, %r274}, [%r321+53248];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r275, %r276, %r277, %r278}, [%r321+55296];
	add.s32 	%r322, %r318, %r18;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r279, %r280, %r281, %r282}, [%r322+49152];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r291, %r292, %r293, %r294}, [%r322+51200];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r295, %r296, %r297, %r298}, [%r322+53248];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r299, %r300, %r301, %r302}, [%r322+55296];
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	add.s32 	%r323, %r318, %r317;
	add.s32 	%r324, %r323, %r19;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r147, %r148, %r175, %r176}, [%r324];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r149, %r150, %r181, %r182}, [%r324+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r151, %r152, %r187, %r188}, [%r324+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r153, %r154, %r193, %r194}, [%r324+12288];
	add.s32 	%r325, %r323, %r20;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r259, %r260, %r283, %r284}, [%r325];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r261, %r262, %r285, %r286}, [%r325+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r263, %r264, %r287, %r288}, [%r325+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r265, %r266, %r289, %r290}, [%r325+12288];
	.loc	1 338 36                        // sk06_mlp_down.py:338:36
	mov.b32 	%r167, %r142;
	mov.b32 	%r168, %r142;
	mov.b32 	%r169, %r142;
	mov.b32 	%r170, %r142;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r167, %r168, %r169, %r170 }, { %r143, %r144, %r145, %r146 }, { %r147, %r148 }, { %r167, %r168, %r169, %r170 };
	// end inline asm
	mov.b32 	%r177, %r142;
	mov.b32 	%r178, %r142;
	mov.b32 	%r179, %r142;
	mov.b32 	%r180, %r142;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r177, %r178, %r179, %r180 }, { %r143, %r144, %r145, %r146 }, { %r149, %r150 }, { %r177, %r178, %r179, %r180 };
	// end inline asm
	mov.b32 	%r183, %r142;
	mov.b32 	%r184, %r142;
	mov.b32 	%r185, %r142;
	mov.b32 	%r186, %r142;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r183, %r184, %r185, %r186 }, { %r143, %r144, %r145, %r146 }, { %r151, %r152 }, { %r183, %r184, %r185, %r186 };
	// end inline asm
	mov.b32 	%r189, %r142;
	mov.b32 	%r190, %r142;
	mov.b32 	%r191, %r142;
	mov.b32 	%r192, %r142;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r189, %r190, %r191, %r192 }, { %r143, %r144, %r145, %r146 }, { %r153, %r154 }, { %r189, %r190, %r191, %r192 };
	// end inline asm
	mov.b32 	%r195, %r142;
	mov.b32 	%r196, %r142;
	mov.b32 	%r197, %r142;
	mov.b32 	%r198, %r142;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r195, %r196, %r197, %r198 }, { %r155, %r156, %r157, %r158 }, { %r147, %r148 }, { %r195, %r196, %r197, %r198 };
	// end inline asm
	mov.b32 	%r203, %r142;
	mov.b32 	%r204, %r142;
	mov.b32 	%r205, %r142;
	mov.b32 	%r206, %r142;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r203, %r204, %r205, %r206 }, { %r155, %r156, %r157, %r158 }, { %r149, %r150 }, { %r203, %r204, %r205, %r206 };
	// end inline asm
	mov.b32 	%r207, %r142;
	mov.b32 	%r208, %r142;
	mov.b32 	%r209, %r142;
	mov.b32 	%r210, %r142;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r207, %r208, %r209, %r210 }, { %r155, %r156, %r157, %r158 }, { %r151, %r152 }, { %r207, %r208, %r209, %r210 };
	// end inline asm
	mov.b32 	%r211, %r142;
	mov.b32 	%r212, %r142;
	mov.b32 	%r213, %r142;
	mov.b32 	%r214, %r142;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r211, %r212, %r213, %r214 }, { %r155, %r156, %r157, %r158 }, { %r153, %r154 }, { %r211, %r212, %r213, %r214 };
	// end inline asm
	mov.b32 	%r215, %r142;
	mov.b32 	%r216, %r142;
	mov.b32 	%r217, %r142;
	mov.b32 	%r218, %r142;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r215, %r216, %r217, %r218 }, { %r159, %r160, %r161, %r162 }, { %r147, %r148 }, { %r215, %r216, %r217, %r218 };
	// end inline asm
	mov.b32 	%r223, %r142;
	mov.b32 	%r224, %r142;
	mov.b32 	%r225, %r142;
	mov.b32 	%r226, %r142;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r223, %r224, %r225, %r226 }, { %r159, %r160, %r161, %r162 }, { %r149, %r150 }, { %r223, %r224, %r225, %r226 };
	// end inline asm
	mov.b32 	%r227, %r142;
	mov.b32 	%r228, %r142;
	mov.b32 	%r229, %r142;
	mov.b32 	%r230, %r142;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r227, %r228, %r229, %r230 }, { %r159, %r160, %r161, %r162 }, { %r151, %r152 }, { %r227, %r228, %r229, %r230 };
	// end inline asm
	mov.b32 	%r231, %r142;
	mov.b32 	%r232, %r142;
	mov.b32 	%r233, %r142;
	mov.b32 	%r234, %r142;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r231, %r232, %r233, %r234 }, { %r159, %r160, %r161, %r162 }, { %r153, %r154 }, { %r231, %r232, %r233, %r234 };
	// end inline asm
	mov.b32 	%r235, %r142;
	mov.b32 	%r236, %r142;
	mov.b32 	%r237, %r142;
	mov.b32 	%r238, %r142;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r235, %r236, %r237, %r238 }, { %r163, %r164, %r165, %r166 }, { %r147, %r148 }, { %r235, %r236, %r237, %r238 };
	// end inline asm
	mov.b32 	%r243, %r142;
	mov.b32 	%r244, %r142;
	mov.b32 	%r245, %r142;
	mov.b32 	%r246, %r142;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r243, %r244, %r245, %r246 }, { %r163, %r164, %r165, %r166 }, { %r149, %r150 }, { %r243, %r244, %r245, %r246 };
	// end inline asm
	mov.b32 	%r247, %r142;
	mov.b32 	%r248, %r142;
	mov.b32 	%r249, %r142;
	mov.b32 	%r250, %r142;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r247, %r248, %r249, %r250 }, { %r163, %r164, %r165, %r166 }, { %r151, %r152 }, { %r247, %r248, %r249, %r250 };
	// end inline asm
	mov.b32 	%r251, %r142;
	mov.b32 	%r252, %r142;
	mov.b32 	%r253, %r142;
	mov.b32 	%r254, %r142;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r251, %r252, %r253, %r254 }, { %r163, %r164, %r165, %r166 }, { %r153, %r154 }, { %r251, %r252, %r253, %r254 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r167, %r168, %r169, %r170 }, { %r171, %r172, %r173, %r174 }, { %r175, %r176 }, { %r167, %r168, %r169, %r170 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r177, %r178, %r179, %r180 }, { %r171, %r172, %r173, %r174 }, { %r181, %r182 }, { %r177, %r178, %r179, %r180 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r183, %r184, %r185, %r186 }, { %r171, %r172, %r173, %r174 }, { %r187, %r188 }, { %r183, %r184, %r185, %r186 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r189, %r190, %r191, %r192 }, { %r171, %r172, %r173, %r174 }, { %r193, %r194 }, { %r189, %r190, %r191, %r192 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r195, %r196, %r197, %r198 }, { %r199, %r200, %r201, %r202 }, { %r175, %r176 }, { %r195, %r196, %r197, %r198 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r203, %r204, %r205, %r206 }, { %r199, %r200, %r201, %r202 }, { %r181, %r182 }, { %r203, %r204, %r205, %r206 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r207, %r208, %r209, %r210 }, { %r199, %r200, %r201, %r202 }, { %r187, %r188 }, { %r207, %r208, %r209, %r210 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r211, %r212, %r213, %r214 }, { %r199, %r200, %r201, %r202 }, { %r193, %r194 }, { %r211, %r212, %r213, %r214 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r215, %r216, %r217, %r218 }, { %r219, %r220, %r221, %r222 }, { %r175, %r176 }, { %r215, %r216, %r217, %r218 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r223, %r224, %r225, %r226 }, { %r219, %r220, %r221, %r222 }, { %r181, %r182 }, { %r223, %r224, %r225, %r226 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r227, %r228, %r229, %r230 }, { %r219, %r220, %r221, %r222 }, { %r187, %r188 }, { %r227, %r228, %r229, %r230 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r231, %r232, %r233, %r234 }, { %r219, %r220, %r221, %r222 }, { %r193, %r194 }, { %r231, %r232, %r233, %r234 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r235, %r236, %r237, %r238 }, { %r239, %r240, %r241, %r242 }, { %r175, %r176 }, { %r235, %r236, %r237, %r238 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r243, %r244, %r245, %r246 }, { %r239, %r240, %r241, %r242 }, { %r181, %r182 }, { %r243, %r244, %r245, %r246 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r247, %r248, %r249, %r250 }, { %r239, %r240, %r241, %r242 }, { %r187, %r188 }, { %r247, %r248, %r249, %r250 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r251, %r252, %r253, %r254 }, { %r239, %r240, %r241, %r242 }, { %r193, %r194 }, { %r251, %r252, %r253, %r254 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r167, %r168, %r169, %r170 }, { %r255, %r256, %r257, %r258 }, { %r259, %r260 }, { %r167, %r168, %r169, %r170 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r177, %r178, %r179, %r180 }, { %r255, %r256, %r257, %r258 }, { %r261, %r262 }, { %r177, %r178, %r179, %r180 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r183, %r184, %r185, %r186 }, { %r255, %r256, %r257, %r258 }, { %r263, %r264 }, { %r183, %r184, %r185, %r186 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r189, %r190, %r191, %r192 }, { %r255, %r256, %r257, %r258 }, { %r265, %r266 }, { %r189, %r190, %r191, %r192 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r195, %r196, %r197, %r198 }, { %r267, %r268, %r269, %r270 }, { %r259, %r260 }, { %r195, %r196, %r197, %r198 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r203, %r204, %r205, %r206 }, { %r267, %r268, %r269, %r270 }, { %r261, %r262 }, { %r203, %r204, %r205, %r206 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r207, %r208, %r209, %r210 }, { %r267, %r268, %r269, %r270 }, { %r263, %r264 }, { %r207, %r208, %r209, %r210 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r211, %r212, %r213, %r214 }, { %r267, %r268, %r269, %r270 }, { %r265, %r266 }, { %r211, %r212, %r213, %r214 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r215, %r216, %r217, %r218 }, { %r271, %r272, %r273, %r274 }, { %r259, %r260 }, { %r215, %r216, %r217, %r218 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r223, %r224, %r225, %r226 }, { %r271, %r272, %r273, %r274 }, { %r261, %r262 }, { %r223, %r224, %r225, %r226 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r227, %r228, %r229, %r230 }, { %r271, %r272, %r273, %r274 }, { %r263, %r264 }, { %r227, %r228, %r229, %r230 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r231, %r232, %r233, %r234 }, { %r271, %r272, %r273, %r274 }, { %r265, %r266 }, { %r231, %r232, %r233, %r234 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r235, %r236, %r237, %r238 }, { %r275, %r276, %r277, %r278 }, { %r259, %r260 }, { %r235, %r236, %r237, %r238 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r243, %r244, %r245, %r246 }, { %r275, %r276, %r277, %r278 }, { %r261, %r262 }, { %r243, %r244, %r245, %r246 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r247, %r248, %r249, %r250 }, { %r275, %r276, %r277, %r278 }, { %r263, %r264 }, { %r247, %r248, %r249, %r250 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r251, %r252, %r253, %r254 }, { %r275, %r276, %r277, %r278 }, { %r265, %r266 }, { %r251, %r252, %r253, %r254 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r167, %r168, %r169, %r170 }, { %r279, %r280, %r281, %r282 }, { %r283, %r284 }, { %r167, %r168, %r169, %r170 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r177, %r178, %r179, %r180 }, { %r279, %r280, %r281, %r282 }, { %r285, %r286 }, { %r177, %r178, %r179, %r180 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r183, %r184, %r185, %r186 }, { %r279, %r280, %r281, %r282 }, { %r287, %r288 }, { %r183, %r184, %r185, %r186 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r189, %r190, %r191, %r192 }, { %r279, %r280, %r281, %r282 }, { %r289, %r290 }, { %r189, %r190, %r191, %r192 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r195, %r196, %r197, %r198 }, { %r291, %r292, %r293, %r294 }, { %r283, %r284 }, { %r195, %r196, %r197, %r198 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r203, %r204, %r205, %r206 }, { %r291, %r292, %r293, %r294 }, { %r285, %r286 }, { %r203, %r204, %r205, %r206 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r207, %r208, %r209, %r210 }, { %r291, %r292, %r293, %r294 }, { %r287, %r288 }, { %r207, %r208, %r209, %r210 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r211, %r212, %r213, %r214 }, { %r291, %r292, %r293, %r294 }, { %r289, %r290 }, { %r211, %r212, %r213, %r214 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r215, %r216, %r217, %r218 }, { %r295, %r296, %r297, %r298 }, { %r283, %r284 }, { %r215, %r216, %r217, %r218 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r223, %r224, %r225, %r226 }, { %r295, %r296, %r297, %r298 }, { %r285, %r286 }, { %r223, %r224, %r225, %r226 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r227, %r228, %r229, %r230 }, { %r295, %r296, %r297, %r298 }, { %r287, %r288 }, { %r227, %r228, %r229, %r230 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r231, %r232, %r233, %r234 }, { %r295, %r296, %r297, %r298 }, { %r289, %r290 }, { %r231, %r232, %r233, %r234 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r235, %r236, %r237, %r238 }, { %r299, %r300, %r301, %r302 }, { %r283, %r284 }, { %r235, %r236, %r237, %r238 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r243, %r244, %r245, %r246 }, { %r299, %r300, %r301, %r302 }, { %r285, %r286 }, { %r243, %r244, %r245, %r246 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r247, %r248, %r249, %r250 }, { %r299, %r300, %r301, %r302 }, { %r287, %r288 }, { %r247, %r248, %r249, %r250 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r251, %r252, %r253, %r254 }, { %r299, %r300, %r301, %r302 }, { %r289, %r290 }, { %r251, %r252, %r253, %r254 };
	// end inline asm
	.loc	1 343 19                        // sk06_mlp_down.py:343:19
	cvt.rn.f32.s32 	%r326, %r251;
	cvt.rn.f32.s32 	%r327, %r252;
	cvt.rn.f32.s32 	%r328, %r253;
	cvt.rn.f32.s32 	%r329, %r254;
	cvt.rn.f32.s32 	%r330, %r247;
	cvt.rn.f32.s32 	%r331, %r248;
	cvt.rn.f32.s32 	%r332, %r249;
	cvt.rn.f32.s32 	%r333, %r250;
	cvt.rn.f32.s32 	%r334, %r243;
	cvt.rn.f32.s32 	%r335, %r244;
	cvt.rn.f32.s32 	%r336, %r245;
	cvt.rn.f32.s32 	%r337, %r246;
	cvt.rn.f32.s32 	%r338, %r235;
	cvt.rn.f32.s32 	%r339, %r236;
	cvt.rn.f32.s32 	%r340, %r237;
	cvt.rn.f32.s32 	%r341, %r238;
	cvt.rn.f32.s32 	%r342, %r231;
	cvt.rn.f32.s32 	%r343, %r232;
	cvt.rn.f32.s32 	%r344, %r233;
	cvt.rn.f32.s32 	%r345, %r234;
	cvt.rn.f32.s32 	%r346, %r227;
	cvt.rn.f32.s32 	%r347, %r228;
	cvt.rn.f32.s32 	%r348, %r229;
	cvt.rn.f32.s32 	%r349, %r230;
	cvt.rn.f32.s32 	%r350, %r223;
	cvt.rn.f32.s32 	%r351, %r224;
	cvt.rn.f32.s32 	%r352, %r225;
	cvt.rn.f32.s32 	%r353, %r226;
	cvt.rn.f32.s32 	%r354, %r215;
	cvt.rn.f32.s32 	%r355, %r216;
	cvt.rn.f32.s32 	%r356, %r217;
	cvt.rn.f32.s32 	%r357, %r218;
	cvt.rn.f32.s32 	%r358, %r211;
	cvt.rn.f32.s32 	%r359, %r212;
	cvt.rn.f32.s32 	%r360, %r213;
	cvt.rn.f32.s32 	%r361, %r214;
	cvt.rn.f32.s32 	%r362, %r207;
	cvt.rn.f32.s32 	%r363, %r208;
	cvt.rn.f32.s32 	%r364, %r209;
	cvt.rn.f32.s32 	%r365, %r210;
	cvt.rn.f32.s32 	%r366, %r203;
	cvt.rn.f32.s32 	%r367, %r204;
	cvt.rn.f32.s32 	%r368, %r205;
	cvt.rn.f32.s32 	%r369, %r206;
	cvt.rn.f32.s32 	%r370, %r195;
	cvt.rn.f32.s32 	%r371, %r196;
	cvt.rn.f32.s32 	%r372, %r197;
	cvt.rn.f32.s32 	%r373, %r198;
	cvt.rn.f32.s32 	%r374, %r189;
	cvt.rn.f32.s32 	%r375, %r190;
	cvt.rn.f32.s32 	%r376, %r191;
	cvt.rn.f32.s32 	%r377, %r192;
	cvt.rn.f32.s32 	%r378, %r183;
	cvt.rn.f32.s32 	%r379, %r184;
	cvt.rn.f32.s32 	%r380, %r185;
	cvt.rn.f32.s32 	%r381, %r186;
	cvt.rn.f32.s32 	%r382, %r177;
	cvt.rn.f32.s32 	%r383, %r178;
	cvt.rn.f32.s32 	%r384, %r179;
	cvt.rn.f32.s32 	%r385, %r180;
	cvt.rn.f32.s32 	%r386, %r167;
	cvt.rn.f32.s32 	%r387, %r168;
	cvt.rn.f32.s32 	%r388, %r169;
	cvt.rn.f32.s32 	%r389, %r170;
	add.f32 	%r843, %r843, %r389;
	add.f32 	%r842, %r842, %r388;
	add.f32 	%r841, %r841, %r387;
	add.f32 	%r840, %r840, %r386;
	add.f32 	%r847, %r847, %r385;
	add.f32 	%r846, %r846, %r384;
	add.f32 	%r845, %r845, %r383;
	add.f32 	%r844, %r844, %r382;
	add.f32 	%r851, %r851, %r381;
	add.f32 	%r850, %r850, %r380;
	add.f32 	%r849, %r849, %r379;
	add.f32 	%r848, %r848, %r378;
	add.f32 	%r855, %r855, %r377;
	add.f32 	%r854, %r854, %r376;
	add.f32 	%r853, %r853, %r375;
	add.f32 	%r852, %r852, %r374;
	add.f32 	%r859, %r859, %r373;
	add.f32 	%r858, %r858, %r372;
	add.f32 	%r857, %r857, %r371;
	add.f32 	%r856, %r856, %r370;
	add.f32 	%r863, %r863, %r369;
	add.f32 	%r862, %r862, %r368;
	add.f32 	%r861, %r861, %r367;
	add.f32 	%r860, %r860, %r366;
	add.f32 	%r867, %r867, %r365;
	add.f32 	%r866, %r866, %r364;
	add.f32 	%r865, %r865, %r363;
	add.f32 	%r864, %r864, %r362;
	add.f32 	%r871, %r871, %r361;
	add.f32 	%r870, %r870, %r360;
	add.f32 	%r869, %r869, %r359;
	add.f32 	%r868, %r868, %r358;
	add.f32 	%r875, %r875, %r357;
	add.f32 	%r874, %r874, %r356;
	add.f32 	%r873, %r873, %r355;
	add.f32 	%r872, %r872, %r354;
	add.f32 	%r879, %r879, %r353;
	add.f32 	%r878, %r878, %r352;
	add.f32 	%r877, %r877, %r351;
	add.f32 	%r876, %r876, %r350;
	add.f32 	%r883, %r883, %r349;
	add.f32 	%r882, %r882, %r348;
	add.f32 	%r881, %r881, %r347;
	add.f32 	%r880, %r880, %r346;
	add.f32 	%r887, %r887, %r345;
	add.f32 	%r886, %r886, %r344;
	add.f32 	%r885, %r885, %r343;
	add.f32 	%r884, %r884, %r342;
	add.f32 	%r891, %r891, %r341;
	add.f32 	%r890, %r890, %r340;
	add.f32 	%r889, %r889, %r339;
	add.f32 	%r888, %r888, %r338;
	add.f32 	%r895, %r895, %r337;
	add.f32 	%r894, %r894, %r336;
	add.f32 	%r893, %r893, %r335;
	add.f32 	%r892, %r892, %r334;
	add.f32 	%r899, %r899, %r333;
	add.f32 	%r898, %r898, %r332;
	add.f32 	%r897, %r897, %r331;
	add.f32 	%r896, %r896, %r330;
	add.f32 	%r903, %r903, %r329;
	add.f32 	%r902, %r902, %r328;
	add.f32 	%r901, %r901, %r327;
	add.f32 	%r900, %r900, %r326;
	.loc	1 344 18                        // sk06_mlp_down.py:344:18
	add.s64 	%rd73, %rd125, %rd5;
	add.s64 	%rd74, %rd126, %rd5;
	add.s64 	%rd75, %rd127, %rd5;
	.loc	1 345 18                        // sk06_mlp_down.py:345:18
	add.s64 	%rd76, %rd128, %rd5;
	add.s64 	%rd77, %rd129, %rd5;
	add.s64 	%rd78, %rd130, %rd5;
	add.s64 	%rd79, %rd131, %rd5;
	add.s64 	%rd80, %rd132, %rd5;
	add.s64 	%rd81, %rd133, %rd5;
	add.s64 	%rd82, %rd134, %rd5;
	add.s64 	%rd83, %rd135, %rd5;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	add.s64 	%rd84, %rd136, %rd5;
	add.s32 	%r390, %r837, 1;
	setp.gt.s32 	%p6, %r390, 2;
	selp.b32 	%r837, 0, %r390, %p6;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	shl.b32 	%r391, %r837, 13;
	bar.sync 	0;
	add.s32 	%r392, %r30, %r391;
	add.s32 	%r303, %r392, 49152;
	selp.b32 	%r304, 16, 0, %p4;
	// begin inline asm
	cp.async.cg.shared.global [ %r303 + 0 ], [ %rd73 + 0 ], 0x10, %r304;
	// end inline asm
	add.s32 	%r305, %r392, 51200;
	// begin inline asm
	cp.async.cg.shared.global [ %r305 + 0 ], [ %rd74 + 0 ], 0x10, %r304;
	// end inline asm
	add.s32 	%r306, %r392, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r306 + 0 ], [ %rd75 + 0 ], 0x10, %r304;
	// end inline asm
	add.s32 	%r307, %r392, 55296;
	// begin inline asm
	cp.async.cg.shared.global [ %r307 + 0 ], [ %rd76 + 0 ], 0x10, %r304;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	add.s32 	%r308, %r392, %r391;
	// begin inline asm
	cp.async.cg.shared.global [ %r308 + 0 ], [ %rd77 + 0 ], 0x10, %r304;
	// end inline asm
	add.s32 	%r309, %r308, 2048;
	// begin inline asm
	cp.async.cg.shared.global [ %r309 + 0 ], [ %rd78 + 0 ], 0x10, %r304;
	// end inline asm
	add.s32 	%r310, %r308, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r310 + 0 ], [ %rd79 + 0 ], 0x10, %r304;
	// end inline asm
	add.s32 	%r311, %r308, 6144;
	// begin inline asm
	cp.async.cg.shared.global [ %r311 + 0 ], [ %rd80 + 0 ], 0x10, %r304;
	// end inline asm
	add.s32 	%r312, %r308, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r312 + 0 ], [ %rd81 + 0 ], 0x10, %r304;
	// end inline asm
	add.s32 	%r313, %r308, 10240;
	// begin inline asm
	cp.async.cg.shared.global [ %r313 + 0 ], [ %rd82 + 0 ], 0x10, %r304;
	// end inline asm
	add.s32 	%r314, %r308, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r314 + 0 ], [ %rd83 + 0 ], 0x10, %r304;
	// end inline asm
	add.s32 	%r315, %r308, 14336;
	// begin inline asm
	cp.async.cg.shared.global [ %r315 + 0 ], [ %rd84 + 0 ], 0x10, %r304;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	add.s32 	%r838, %r838, 1;
	add.s64 	%rd136, %rd136, 128;
	add.s64 	%rd135, %rd135, 128;
	add.s64 	%rd134, %rd134, 128;
	add.s64 	%rd133, %rd133, 128;
	add.s64 	%rd132, %rd132, 128;
	add.s64 	%rd131, %rd131, 128;
	add.s64 	%rd130, %rd130, 128;
	add.s64 	%rd129, %rd129, 128;
	add.s64 	%rd128, %rd128, 128;
	add.s64 	%rd127, %rd127, 128;
	add.s64 	%rd126, %rd126, 128;
	add.s64 	%rd125, %rd125, 128;
	setp.ne.b32 	%p7, %r12, %r838;
	@%p7 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_4;
$L__BB0_1:                              // %.._crit_edge_crit_edge
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	and.b32 	%r839, %r2, 16;
	mov.b32 	%r840, 0f00000000;
	mov.b32 	%r841, %r840;
	mov.b32 	%r842, %r840;
	mov.b32 	%r843, %r840;
	mov.b32 	%r844, %r840;
	mov.b32 	%r845, %r840;
	mov.b32 	%r846, %r840;
	mov.b32 	%r847, %r840;
	mov.b32 	%r848, %r840;
	mov.b32 	%r849, %r840;
	mov.b32 	%r850, %r840;
	mov.b32 	%r851, %r840;
	mov.b32 	%r852, %r840;
	mov.b32 	%r853, %r840;
	mov.b32 	%r854, %r840;
	mov.b32 	%r855, %r840;
	mov.b32 	%r856, %r840;
	mov.b32 	%r857, %r840;
	mov.b32 	%r858, %r840;
	mov.b32 	%r859, %r840;
	mov.b32 	%r860, %r840;
	mov.b32 	%r861, %r840;
	mov.b32 	%r862, %r840;
	mov.b32 	%r863, %r840;
	mov.b32 	%r864, %r840;
	mov.b32 	%r865, %r840;
	mov.b32 	%r866, %r840;
	mov.b32 	%r867, %r840;
	mov.b32 	%r868, %r840;
	mov.b32 	%r869, %r840;
	mov.b32 	%r870, %r840;
	mov.b32 	%r871, %r840;
	mov.b32 	%r872, %r840;
	mov.b32 	%r873, %r840;
	mov.b32 	%r874, %r840;
	mov.b32 	%r875, %r840;
	mov.b32 	%r876, %r840;
	mov.b32 	%r877, %r840;
	mov.b32 	%r878, %r840;
	mov.b32 	%r879, %r840;
	mov.b32 	%r880, %r840;
	mov.b32 	%r881, %r840;
	mov.b32 	%r882, %r840;
	mov.b32 	%r883, %r840;
	mov.b32 	%r884, %r840;
	mov.b32 	%r885, %r840;
	mov.b32 	%r886, %r840;
	mov.b32 	%r887, %r840;
	mov.b32 	%r888, %r840;
	mov.b32 	%r889, %r840;
	mov.b32 	%r890, %r840;
	mov.b32 	%r891, %r840;
	mov.b32 	%r892, %r840;
	mov.b32 	%r893, %r840;
	mov.b32 	%r894, %r840;
	mov.b32 	%r895, %r840;
	mov.b32 	%r896, %r840;
	mov.b32 	%r897, %r840;
	mov.b32 	%r898, %r840;
	mov.b32 	%r899, %r840;
	mov.b32 	%r900, %r840;
	mov.b32 	%r901, %r840;
	mov.b32 	%r902, %r840;
	mov.b32 	%r903, %r840;
$L__BB0_4:                              // %._crit_edge
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r505, %r8, %r11;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r506, %r505, %r22;
	.loc	1 323 45                        // sk06_mlp_down.py:323:45
	or.b32 	%r507, %r8, %r835;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r508, %r507, 8;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r509, %r508, %r22;
	rem.s32 	%r510, %r507, %r22;
	.loc	1 323 45                        // sk06_mlp_down.py:323:45
	shl.b32 	%r511, %r10, 3;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r512, %r8, %r511;
	.loc	1 322 45                        // sk06_mlp_down.py:322:45
	shr.u32 	%r513, %r2, 2;
	bfe.u32 	%r514, %r2, 2, 3;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r515, %r514, %r1;
	or.b32 	%r516, %r515, 56;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r517, %r516, %r21;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r518, %r515, 48;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r519, %r518, %r21;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r520, %r515, 40;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r521, %r520, %r21;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r522, %r515, 32;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r523, %r522, %r21;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r524, %r515, 24;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r525, %r524, %r21;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r526, %r515, 16;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r527, %r526, %r21;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r528, %r515, 8;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r529, %r528, %r21;
	rem.s32 	%r530, %r515, %r21;
	.loc	1 322 45                        // sk06_mlp_down.py:322:45
	shr.u32 	%r531, %r2, 4;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r532, %r531, %r1;
	or.b32 	%r533, %r532, 56;
	.loc	1 322 45                        // sk06_mlp_down.py:322:45
	bfe.u32 	%r534, %r2, 4, 3;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r535, %r534, %r1;
	or.b32 	%r536, %r535, 48;
	or.b32 	%r537, %r535, 40;
	or.b32 	%r538, %r535, 32;
	or.b32 	%r539, %r535, 24;
	or.b32 	%r540, %r535, 16;
	or.b32 	%r541, %r535, 8;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 347 38                        // sk06_mlp_down.py:347:38
	mad.wide.s32 	%rd85, %r530, 4, %rd18;
	mad.wide.s32 	%rd86, %r529, 4, %rd18;
	mad.wide.s32 	%rd87, %r527, 4, %rd18;
	mad.wide.s32 	%rd88, %r525, 4, %rd18;
	mad.wide.s32 	%rd89, %r523, 4, %rd18;
	mad.wide.s32 	%rd90, %r521, 4, %rd18;
	mad.wide.s32 	%rd91, %r519, 4, %rd18;
	mad.wide.s32 	%rd92, %r517, 4, %rd18;
	.loc	1 347 24                        // sk06_mlp_down.py:347:24
	// begin inline asm
	mov.u32 %r393, 0x0;
	ld.global.b32 { %r393 }, [ %rd85 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r394, 0x0;
	ld.global.b32 { %r394 }, [ %rd86 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r395, 0x0;
	ld.global.b32 { %r395 }, [ %rd87 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r396, 0x0;
	ld.global.b32 { %r396 }, [ %rd88 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r397, 0x0;
	ld.global.b32 { %r397 }, [ %rd89 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r398, 0x0;
	ld.global.b32 { %r398 }, [ %rd90 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r399, 0x0;
	ld.global.b32 { %r399 }, [ %rd91 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r400, 0x0;
	ld.global.b32 { %r400 }, [ %rd92 + 0 ];
	// end inline asm
	.loc	1 349 42                        // sk06_mlp_down.py:349:42
	mad.wide.s32 	%rd93, %r506, 4, %rd19;
	.loc	1 349 28                        // sk06_mlp_down.py:349:28
	// begin inline asm
	mov.u32 %r402, 0x0;
	ld.global.b32 { %r402 }, [ %rd93 + 0 ];
	// end inline asm
	.loc	1 349 20                        // sk06_mlp_down.py:349:20
	shl.b32 	%r542, %r11, 2;
	add.s32 	%r401, %r132, %r542;
	// begin inline asm
	st.shared.b32 [ %r401 + 0 ], %r402;
	// end inline asm
	bar.sync 	0;
	and.b32 	%r543, %r2, 3;
	shl.b32 	%r544, %r543, 3;
	and.b32 	%r545, %r2, 96;
	add.s32 	%r546, %r132, %r544;
	add.s32 	%r547, %r546, %r545;
	.loc	1 350 49                        // sk06_mlp_down.py:350:49
	mul.lo.s32 	%r548, %r4, %r24;
	mul.lo.s32 	%r549, %r5, %r24;
	mul.lo.s32 	%r550, %r6, %r24;
	mul.lo.s32 	%r551, %r7, %r24;
	.loc	1 350 31                        // sk06_mlp_down.py:350:31
	mad.wide.s32 	%rd110, %r548, 2, %rd17;
	mad.wide.s32 	%rd111, %r549, 2, %rd17;
	mad.wide.s32 	%rd112, %r550, 2, %rd17;
	mad.wide.s32 	%rd113, %r551, 2, %rd17;
	.loc	1 350 64                        // sk06_mlp_down.py:350:64
	mul.wide.s32 	%rd114, %r510, 2;
	add.s64 	%rd94, %rd110, %rd114;
	mul.wide.s32 	%rd115, %r509, 2;
	add.s64 	%rd95, %rd110, %rd115;
	add.s64 	%rd96, %rd111, %rd114;
	add.s64 	%rd97, %rd111, %rd115;
	add.s64 	%rd98, %rd112, %rd114;
	add.s64 	%rd99, %rd112, %rd115;
	add.s64 	%rd100, %rd113, %rd114;
	add.s64 	%rd101, %rd113, %rd115;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	shl.b32 	%r552, %r13, 6;
	shl.b32 	%r553, %r3, 1;
	or.b32 	%r554, %r552, %r835;
	xor.b32 	%r555, %r554, %r553;
	add.s32 	%r403, %r132, %r555;
	add.s32 	%r408, %r403, 256;
	shl.b32 	%r556, %r9, 9;
	shl.b32 	%r557, %r10, 4;
	setp.eq.b32 	%p16, %r839, 0;
	shl.b32 	%r558, %r839, 1;
	shl.b32 	%r559, %r2, 3;
	and.b32 	%r560, %r559, 256;
	and.b32 	%r561, %r513, 16;
	or.b32 	%r562, %r556, %r560;
	xor.b32 	%r563, %r557, %r558;
	xor.b32 	%r564, %r563, %r561;
	or.b32 	%r565, %r564, %r562;
	add.s32 	%r566, %r132, %r565;
	xor.b32 	%r567, %r565, 64;
	add.s32 	%r568, %r132, %r567;
	.loc	1 357 31                        // sk06_mlp_down.py:357:31
	setp.lt.s32 	%p17, %r535, %r21;
	setp.lt.s32 	%p18, %r541, %r21;
	setp.lt.s32 	%p19, %r540, %r21;
	setp.lt.s32 	%p20, %r539, %r21;
	setp.lt.s32 	%p21, %r538, %r21;
	setp.lt.s32 	%p22, %r537, %r21;
	setp.lt.s32 	%p23, %r536, %r21;
	setp.lt.s32 	%p24, %r533, %r21;
	.loc	1 357 54                        // sk06_mlp_down.py:357:54
	setp.lt.s32 	%p25, %r512, %r22;
	.loc	1 357 37                        // sk06_mlp_down.py:357:37
	and.pred 	%p8, %p17, %p25;
	and.pred 	%p9, %p18, %p25;
	and.pred 	%p10, %p19, %p25;
	and.pred 	%p11, %p20, %p25;
	and.pred 	%p12, %p21, %p25;
	and.pred 	%p13, %p22, %p25;
	and.pred 	%p14, %p23, %p25;
	and.pred 	%p15, %p24, %p25;
	.loc	1 355 35                        // sk06_mlp_down.py:355:35
	mul.lo.s32 	%r569, %r535, %r23;
	mul.lo.s32 	%r570, %r541, %r23;
	mul.lo.s32 	%r571, %r540, %r23;
	mul.lo.s32 	%r572, %r539, %r23;
	mul.lo.s32 	%r573, %r538, %r23;
	mul.lo.s32 	%r574, %r537, %r23;
	mul.lo.s32 	%r575, %r536, %r23;
	mul.lo.s32 	%r576, %r533, %r23;
	.loc	1 355 18                        // sk06_mlp_down.py:355:18
	mad.wide.s32 	%rd116, %r569, 2, %rd16;
	mad.wide.s32 	%rd117, %r570, 2, %rd16;
	mad.wide.s32 	%rd118, %r571, 2, %rd16;
	mad.wide.s32 	%rd119, %r572, 2, %rd16;
	mad.wide.s32 	%rd120, %r573, 2, %rd16;
	mad.wide.s32 	%rd121, %r574, 2, %rd16;
	mad.wide.s32 	%rd122, %r575, 2, %rd16;
	mad.wide.s32 	%rd123, %r576, 2, %rd16;
	.loc	1 355 50                        // sk06_mlp_down.py:355:50
	mul.wide.s32 	%rd124, %r512, 2;
	add.s64 	%rd102, %rd116, %rd124;
	add.s64 	%rd103, %rd117, %rd124;
	add.s64 	%rd104, %rd118, %rd124;
	add.s64 	%rd105, %rd119, %rd124;
	add.s64 	%rd106, %rd120, %rd124;
	add.s64 	%rd107, %rd121, %rd124;
	add.s64 	%rd108, %rd122, %rd124;
	add.s64 	%rd109, %rd123, %rd124;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r577, %r859, %r396;
	mul.f32 	%r578, %r858, %r396;
	.loc	1 349 20                        // sk06_mlp_down.py:349:20
	ld.shared.v2.b32 	{%r579, %r580}, [%r547];
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r581, %r841, %r393;
	mul.f32 	%r582, %r840, %r393;
	mul.f32 	%r583, %r843, %r394;
	mul.f32 	%r584, %r842, %r394;
	mul.f32 	%r585, %r857, %r395;
	mul.f32 	%r586, %r856, %r395;
	mul.f32 	%r587, %r867, %r396;
	mul.f32 	%r588, %r866, %r396;
	.loc	1 349 20                        // sk06_mlp_down.py:349:20
	ld.shared.v2.b32 	{%r589, %r590}, [%r547+256];
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r591, %r849, %r393;
	mul.f32 	%r592, %r848, %r393;
	mul.f32 	%r593, %r851, %r394;
	mul.f32 	%r594, %r850, %r394;
	mul.f32 	%r595, %r863, %r396;
	mul.f32 	%r596, %r862, %r396;
	.loc	1 349 20                        // sk06_mlp_down.py:349:20
	ld.shared.v2.b32 	{%r597, %r598}, [%r547+128];
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r599, %r845, %r393;
	mul.f32 	%r600, %r844, %r393;
	mul.f32 	%r601, %r847, %r394;
	mul.f32 	%r602, %r846, %r394;
	mul.f32 	%r603, %r861, %r395;
	mul.f32 	%r604, %r860, %r395;
	mul.f32 	%r605, %r865, %r395;
	mul.f32 	%r606, %r864, %r395;
	mul.f32 	%r607, %r871, %r396;
	mul.f32 	%r608, %r870, %r396;
	.loc	1 349 20                        // sk06_mlp_down.py:349:20
	ld.shared.v2.b32 	{%r609, %r610}, [%r547+384];
	.loc	1 350 19                        // sk06_mlp_down.py:350:19
	// begin inline asm
	mov.u32 %r404, 0x0;
	mov.u32 %r405, 0x0;
	mov.u32 %r406, 0x0;
	mov.u32 %r407, 0x0;
	ld.global.v4.b32 { %r404, %r405, %r406, %r407 }, [ %rd94 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r409, 0x0;
	mov.u32 %r410, 0x0;
	mov.u32 %r411, 0x0;
	mov.u32 %r412, 0x0;
	ld.global.v4.b32 { %r409, %r410, %r411, %r412 }, [ %rd95 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r413, 0x0;
	mov.u32 %r414, 0x0;
	mov.u32 %r415, 0x0;
	mov.u32 %r416, 0x0;
	ld.global.v4.b32 { %r413, %r414, %r415, %r416 }, [ %rd96 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r417, 0x0;
	mov.u32 %r418, 0x0;
	mov.u32 %r419, 0x0;
	mov.u32 %r420, 0x0;
	ld.global.v4.b32 { %r417, %r418, %r419, %r420 }, [ %rd97 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r421, 0x0;
	mov.u32 %r422, 0x0;
	mov.u32 %r423, 0x0;
	mov.u32 %r424, 0x0;
	ld.global.v4.b32 { %r421, %r422, %r423, %r424 }, [ %rd98 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r425, 0x0;
	mov.u32 %r426, 0x0;
	mov.u32 %r427, 0x0;
	mov.u32 %r428, 0x0;
	ld.global.v4.b32 { %r425, %r426, %r427, %r428 }, [ %rd99 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r429, 0x0;
	mov.u32 %r430, 0x0;
	mov.u32 %r431, 0x0;
	mov.u32 %r432, 0x0;
	ld.global.v4.b32 { %r429, %r430, %r431, %r432 }, [ %rd100 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r433, 0x0;
	mov.u32 %r434, 0x0;
	mov.u32 %r435, 0x0;
	mov.u32 %r436, 0x0;
	ld.global.v4.b32 { %r433, %r434, %r435, %r436 }, [ %rd101 + 0 ];
	// end inline asm
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r403 + 0 ], { %r404, %r405, %r406, %r407 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r408 + 0 ], { %r409, %r410, %r411, %r412 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r611, %r612, %r613, %r614}, [%r566];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r615, %r616, %r617, %r618}, [%r568];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r403 + 0 ], { %r413, %r414, %r415, %r416 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r408 + 0 ], { %r417, %r418, %r419, %r420 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r619, %r620, %r621, %r622}, [%r566];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r623, %r624, %r625, %r626}, [%r568];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r403 + 0 ], { %r421, %r422, %r423, %r424 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r408 + 0 ], { %r425, %r426, %r427, %r428 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r627, %r628, %r629, %r630}, [%r566];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r631, %r632, %r633, %r634}, [%r568];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r403 + 0 ], { %r429, %r430, %r431, %r432 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r408 + 0 ], { %r433, %r434, %r435, %r436 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r635, %r636, %r637, %r638}, [%r566];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r639, %r640, %r641, %r642}, [%r568];
	mov.b32 	{%rs1, %rs2}, %r620;
	cvt.f32.bf16 	%r643, %rs2;
	cvt.f32.bf16 	%r644, %rs1;
	mov.b32 	{%rs3, %rs4}, %r622;
	cvt.f32.bf16 	%r645, %rs4;
	cvt.f32.bf16 	%r646, %rs3;
	mov.b32 	{%rs5, %rs6}, %r624;
	cvt.f32.bf16 	%r647, %rs6;
	cvt.f32.bf16 	%r648, %rs5;
	mov.b32 	{%rs7, %rs8}, %r626;
	cvt.f32.bf16 	%r649, %rs8;
	cvt.f32.bf16 	%r650, %rs7;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r651, %r578, %r579, %r644;
	fma.rn.f32 	%r652, %r577, %r580, %r643;
	fma.rn.f32 	%r653, %r596, %r597, %r646;
	fma.rn.f32 	%r654, %r595, %r598, %r645;
	fma.rn.f32 	%r655, %r588, %r589, %r648;
	fma.rn.f32 	%r656, %r587, %r590, %r647;
	fma.rn.f32 	%r657, %r608, %r609, %r650;
	fma.rn.f32 	%r658, %r607, %r610, %r649;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs9, %rs10}, %r611;
	cvt.f32.bf16 	%r659, %rs10;
	cvt.f32.bf16 	%r660, %rs9;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r661, %r582, %r579, %r660;
	fma.rn.f32 	%r662, %r581, %r580, %r659;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r438, %r662, %r661;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs11, %rs12}, %r612;
	cvt.f32.bf16 	%r663, %rs12;
	cvt.f32.bf16 	%r664, %rs11;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r665, %r584, %r579, %r664;
	fma.rn.f32 	%r666, %r583, %r580, %r663;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r439, %r666, %r665;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs13, %rs14}, %r613;
	cvt.f32.bf16 	%r667, %rs14;
	cvt.f32.bf16 	%r668, %rs13;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r669, %r600, %r597, %r668;
	fma.rn.f32 	%r670, %r599, %r598, %r667;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r448, %r670, %r669;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs15, %rs16}, %r614;
	cvt.f32.bf16 	%r671, %rs16;
	cvt.f32.bf16 	%r672, %rs15;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r673, %r602, %r597, %r672;
	fma.rn.f32 	%r674, %r601, %r598, %r671;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r449, %r674, %r673;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs17, %rs18}, %r615;
	cvt.f32.bf16 	%r675, %rs18;
	cvt.f32.bf16 	%r676, %rs17;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r677, %r592, %r589, %r676;
	fma.rn.f32 	%r678, %r591, %r590, %r675;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r443, %r678, %r677;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs19, %rs20}, %r616;
	cvt.f32.bf16 	%r679, %rs20;
	cvt.f32.bf16 	%r680, %rs19;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r681, %r594, %r589, %r680;
	fma.rn.f32 	%r682, %r593, %r590, %r679;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r444, %r682, %r681;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r683, %r853, %r393;
	mul.f32 	%r684, %r852, %r393;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs21, %rs22}, %r617;
	cvt.f32.bf16 	%r685, %rs22;
	cvt.f32.bf16 	%r686, %rs21;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r687, %r684, %r609, %r686;
	fma.rn.f32 	%r688, %r683, %r610, %r685;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r453, %r688, %r687;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r689, %r855, %r394;
	mul.f32 	%r690, %r854, %r394;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs23, %rs24}, %r618;
	cvt.f32.bf16 	%r691, %rs24;
	cvt.f32.bf16 	%r692, %rs23;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r693, %r690, %r609, %r692;
	fma.rn.f32 	%r694, %r689, %r610, %r691;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r454, %r694, %r693;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs25, %rs26}, %r619;
	cvt.f32.bf16 	%r695, %rs26;
	cvt.f32.bf16 	%r696, %rs25;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r697, %r586, %r579, %r696;
	fma.rn.f32 	%r698, %r585, %r580, %r695;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r440, %r698, %r697;
	cvt.rn.bf16x2.f32 	%r441, %r652, %r651;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs27, %rs28}, %r621;
	cvt.f32.bf16 	%r699, %rs28;
	cvt.f32.bf16 	%r700, %rs27;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r701, %r604, %r597, %r700;
	fma.rn.f32 	%r702, %r603, %r598, %r699;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r450, %r702, %r701;
	cvt.rn.bf16x2.f32 	%r451, %r654, %r653;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs29, %rs30}, %r623;
	cvt.f32.bf16 	%r703, %rs30;
	cvt.f32.bf16 	%r704, %rs29;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r705, %r606, %r589, %r704;
	fma.rn.f32 	%r706, %r605, %r590, %r703;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r445, %r706, %r705;
	cvt.rn.bf16x2.f32 	%r446, %r656, %r655;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r707, %r869, %r395;
	mul.f32 	%r708, %r868, %r395;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs31, %rs32}, %r625;
	cvt.f32.bf16 	%r709, %rs32;
	cvt.f32.bf16 	%r710, %rs31;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r711, %r708, %r609, %r710;
	fma.rn.f32 	%r712, %r707, %r610, %r709;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r455, %r712, %r711;
	cvt.rn.bf16x2.f32 	%r456, %r658, %r657;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r713, %r873, %r397;
	mul.f32 	%r714, %r872, %r397;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs33, %rs34}, %r627;
	cvt.f32.bf16 	%r715, %rs34;
	cvt.f32.bf16 	%r716, %rs33;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r717, %r714, %r579, %r716;
	fma.rn.f32 	%r718, %r713, %r580, %r715;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r457, %r718, %r717;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r719, %r875, %r398;
	mul.f32 	%r720, %r874, %r398;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs35, %rs36}, %r628;
	cvt.f32.bf16 	%r721, %rs36;
	cvt.f32.bf16 	%r722, %rs35;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r723, %r720, %r579, %r722;
	fma.rn.f32 	%r724, %r719, %r580, %r721;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r458, %r724, %r723;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r725, %r877, %r397;
	mul.f32 	%r726, %r876, %r397;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs37, %rs38}, %r629;
	cvt.f32.bf16 	%r727, %rs38;
	cvt.f32.bf16 	%r728, %rs37;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r729, %r726, %r597, %r728;
	fma.rn.f32 	%r730, %r725, %r598, %r727;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r465, %r730, %r729;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r731, %r879, %r398;
	mul.f32 	%r732, %r878, %r398;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs39, %rs40}, %r630;
	cvt.f32.bf16 	%r733, %rs40;
	cvt.f32.bf16 	%r734, %rs39;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r735, %r732, %r597, %r734;
	fma.rn.f32 	%r736, %r731, %r598, %r733;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r466, %r736, %r735;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r737, %r881, %r397;
	mul.f32 	%r738, %r880, %r397;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs41, %rs42}, %r631;
	cvt.f32.bf16 	%r739, %rs42;
	cvt.f32.bf16 	%r740, %rs41;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r741, %r738, %r589, %r740;
	fma.rn.f32 	%r742, %r737, %r590, %r739;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r461, %r742, %r741;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r743, %r883, %r398;
	mul.f32 	%r744, %r882, %r398;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs43, %rs44}, %r632;
	cvt.f32.bf16 	%r745, %rs44;
	cvt.f32.bf16 	%r746, %rs43;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r747, %r744, %r589, %r746;
	fma.rn.f32 	%r748, %r743, %r590, %r745;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r462, %r748, %r747;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r749, %r885, %r397;
	mul.f32 	%r750, %r884, %r397;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs45, %rs46}, %r633;
	cvt.f32.bf16 	%r751, %rs46;
	cvt.f32.bf16 	%r752, %rs45;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r753, %r750, %r609, %r752;
	fma.rn.f32 	%r754, %r749, %r610, %r751;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r469, %r754, %r753;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r755, %r887, %r398;
	mul.f32 	%r756, %r886, %r398;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs47, %rs48}, %r634;
	cvt.f32.bf16 	%r757, %rs48;
	cvt.f32.bf16 	%r758, %rs47;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r759, %r756, %r609, %r758;
	fma.rn.f32 	%r760, %r755, %r610, %r757;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r470, %r760, %r759;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r761, %r889, %r399;
	mul.f32 	%r762, %r888, %r399;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs49, %rs50}, %r635;
	cvt.f32.bf16 	%r763, %rs50;
	cvt.f32.bf16 	%r764, %rs49;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r765, %r762, %r579, %r764;
	fma.rn.f32 	%r766, %r761, %r580, %r763;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r459, %r766, %r765;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r767, %r891, %r400;
	mul.f32 	%r768, %r890, %r400;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs51, %rs52}, %r636;
	cvt.f32.bf16 	%r769, %rs52;
	cvt.f32.bf16 	%r770, %rs51;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r771, %r768, %r579, %r770;
	fma.rn.f32 	%r772, %r767, %r580, %r769;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r460, %r772, %r771;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r773, %r893, %r399;
	mul.f32 	%r774, %r892, %r399;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs53, %rs54}, %r637;
	cvt.f32.bf16 	%r775, %rs54;
	cvt.f32.bf16 	%r776, %rs53;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r777, %r774, %r597, %r776;
	fma.rn.f32 	%r778, %r773, %r598, %r775;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r467, %r778, %r777;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r779, %r895, %r400;
	mul.f32 	%r780, %r894, %r400;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs55, %rs56}, %r638;
	cvt.f32.bf16 	%r781, %rs56;
	cvt.f32.bf16 	%r782, %rs55;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r783, %r780, %r597, %r782;
	fma.rn.f32 	%r784, %r779, %r598, %r781;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r468, %r784, %r783;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r785, %r897, %r399;
	mul.f32 	%r786, %r896, %r399;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs57, %rs58}, %r639;
	cvt.f32.bf16 	%r787, %rs58;
	cvt.f32.bf16 	%r788, %rs57;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r789, %r786, %r589, %r788;
	fma.rn.f32 	%r790, %r785, %r590, %r787;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r463, %r790, %r789;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r791, %r899, %r400;
	mul.f32 	%r792, %r898, %r400;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs59, %rs60}, %r640;
	cvt.f32.bf16 	%r793, %rs60;
	cvt.f32.bf16 	%r794, %rs59;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r795, %r792, %r589, %r794;
	fma.rn.f32 	%r796, %r791, %r590, %r793;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r464, %r796, %r795;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r797, %r901, %r399;
	mul.f32 	%r798, %r900, %r399;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs61, %rs62}, %r641;
	cvt.f32.bf16 	%r799, %rs62;
	cvt.f32.bf16 	%r800, %rs61;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r801, %r798, %r609, %r800;
	fma.rn.f32 	%r802, %r797, %r610, %r799;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r471, %r802, %r801;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r803, %r903, %r400;
	mul.f32 	%r804, %r902, %r400;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs63, %rs64}, %r642;
	cvt.f32.bf16 	%r805, %rs64;
	cvt.f32.bf16 	%r806, %rs63;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r807, %r804, %r609, %r806;
	fma.rn.f32 	%r808, %r803, %r610, %r805;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r472, %r808, %r807;
	bar.sync 	0;
	shl.b32 	%r809, %r543, 11;
	shl.b32 	%r810, %r543, 5;
	shl.b32 	%r811, %r2, 4;
	and.b32 	%r812, %r811, 384;
	shr.u32 	%r813, %r545, 1;
	bfe.s32 	%r814, %r2, 2, 1;
	and.b32 	%r815, %r814, 1040;
	or.b32 	%r816, %r810, %r812;
	xor.b32 	%r817, %r815, %r813;
	xor.b32 	%r818, %r817, %r816;
	or.b32 	%r819, %r818, %r809;
	add.s32 	%r437, %r132, %r819;
	// begin inline asm
	st.shared.v4.b32 [ %r437 + 0 ], { %r438, %r439, %r440, %r441 };
	// end inline asm
	add.s32 	%r442, %r437, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r442 + 0 ], { %r443, %r444, %r445, %r446 };
	// end inline asm
	xor.b32 	%r820, %r819, 64;
	add.s32 	%r447, %r132, %r820;
	// begin inline asm
	st.shared.v4.b32 [ %r447 + 0 ], { %r448, %r449, %r450, %r451 };
	// end inline asm
	add.s32 	%r452, %r447, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r452 + 0 ], { %r453, %r454, %r455, %r456 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r821, %r545, 2;
	shl.b32 	%r822, %r2, 6;
	and.b32 	%r823, %r822, 512;
	selp.b32 	%r824, 0, 1040, %p16;
	or.b32 	%r825, %r835, %r821;
	xor.b32 	%r826, %r825, %r824;
	or.b32 	%r827, %r826, %r823;
	add.s32 	%r828, %r132, %r827;
	ld.shared.v4.b32 	{%r473, %r477, %r481, %r485}, [%r828];
	xor.b32 	%r829, %r827, 32;
	add.s32 	%r830, %r132, %r829;
	ld.shared.v4.b32 	{%r474, %r478, %r482, %r486}, [%r830+2048];
	xor.b32 	%r831, %r827, 64;
	add.s32 	%r832, %r132, %r831;
	ld.shared.v4.b32 	{%r475, %r479, %r483, %r487}, [%r832+4096];
	xor.b32 	%r833, %r827, 96;
	add.s32 	%r834, %r132, %r833;
	ld.shared.v4.b32 	{%r476, %r480, %r484, %r488}, [%r834+6144];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r437 + 0 ], { %r457, %r458, %r459, %r460 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r442 + 0 ], { %r461, %r462, %r463, %r464 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r447 + 0 ], { %r465, %r466, %r467, %r468 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r452 + 0 ], { %r469, %r470, %r471, %r472 };
	// end inline asm
	bar.sync 	0;
	ld.shared.v4.b32 	{%r489, %r493, %r497, %r501}, [%r828];
	ld.shared.v4.b32 	{%r490, %r494, %r498, %r502}, [%r830+2048];
	ld.shared.v4.b32 	{%r491, %r495, %r499, %r503}, [%r832+4096];
	ld.shared.v4.b32 	{%r492, %r496, %r500, %r504}, [%r834+6144];
	.loc	1 356 8                         // sk06_mlp_down.py:356:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd102 + 0 ], { %r473, %r474, %r475, %r476 };
	// end inline asm
	// begin inline asm
	@%p9 st.global.v4.b32 [ %rd103 + 0 ], { %r477, %r478, %r479, %r480 };
	// end inline asm
	// begin inline asm
	@%p10 st.global.v4.b32 [ %rd104 + 0 ], { %r481, %r482, %r483, %r484 };
	// end inline asm
	// begin inline asm
	@%p11 st.global.v4.b32 [ %rd105 + 0 ], { %r485, %r486, %r487, %r488 };
	// end inline asm
	// begin inline asm
	@%p12 st.global.v4.b32 [ %rd106 + 0 ], { %r489, %r490, %r491, %r492 };
	// end inline asm
	// begin inline asm
	@%p13 st.global.v4.b32 [ %rd107 + 0 ], { %r493, %r494, %r495, %r496 };
	// end inline asm
	// begin inline asm
	@%p14 st.global.v4.b32 [ %rd108 + 0 ], { %r497, %r498, %r499, %r500 };
	// end inline asm
	// begin inline asm
	@%p15 st.global.v4.b32 [ %rd109 + 0 ], { %r501, %r502, %r503, %r504 };
	// end inline asm
	.loc	1 354 4                         // sk06_mlp_down.py:354:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/workspace/vllm/_genesis/kernels/sk06_mlp_down.py"
	.file	2 "/usr/local/lib/python3.12/dist-packages/triton/language/standard.py"
	.section	.debug_abbrev
	{
.b8 1                                   // Abbreviation Code
.b8 17                                  // DW_TAG_compile_unit
.b8 1                                   // DW_CHILDREN_yes
.b8 37                                  // DW_AT_producer
.b8 8                                   // DW_FORM_string
.b8 19                                  // DW_AT_language
.b8 5                                   // DW_FORM_data2
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 16                                  // DW_AT_stmt_list
.b8 6                                   // DW_FORM_data4
.b8 27                                  // DW_AT_comp_dir
.b8 8                                   // DW_FORM_string
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 2                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 0                                   // DW_CHILDREN_no
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 32                                  // DW_AT_inline
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 3                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 1                                   // DW_CHILDREN_yes
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 4                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 0                                   // DW_CHILDREN_no
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 5                                   // DW_FORM_data2
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 168                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0xa1 DW_TAG_compile_unit
.b8 116                                 // DW_AT_producer
.b8 114
.b8 105
.b8 116
.b8 111
.b8 110
.b8 0
.b8 2                                   // DW_AT_language
.b8 0
.b8 115                                 // DW_AT_name
.b8 107
.b8 48
.b8 54
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 100
.b8 111
.b8 119
.b8 110
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 119
.b8 111
.b8 114
.b8 107
.b8 115
.b8 112
.b8 97
.b8 99
.b8 101
.b8 47
.b8 118
.b8 108
.b8 108
.b8 109
.b8 47
.b8 95
.b8 103
.b8 101
.b8 110
.b8 101
.b8 115
.b8 105
.b8 115
.b8 47
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 115
.b8 0
.b8 2                                   // Abbrev [2] 0x4b:0x18 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 54
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 100
.b8 111
.b8 119
.b8 110
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x63:0x48 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 75                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x78:0x19 DW_TAG_inlined_subroutine
.b32 75                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 58                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x91:0x19 DW_TAG_inlined_subroutine
.b32 75                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 59                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_4 = _Nativo(
    "sk06_mlp_down/tile64x128x128_shift0_abi15",
    _PTX_4, "_sk06_mlp_down_kernel",
    warps=4, shared=73728,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 18],
    horneado={11: 1, 12: 1, 15: 1, 17: 1, 19: 1, 20: 64, 21: 128, 22: 128, 23: 8, 24: 128, 25: False},
    div16=[8, 9, 10, 13, 14, 16],
)

_PTX_5 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk06_mlp_down_kernel   // -- Begin function _sk06_mlp_down_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk06_mlp_down_kernel
.visible .entry _sk06_mlp_down_kernel(
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_6,
	.param .u32 _sk06_mlp_down_kernel_param_7,
	.param .u32 _sk06_mlp_down_kernel_param_8,
	.param .u32 _sk06_mlp_down_kernel_param_9,
	.param .u32 _sk06_mlp_down_kernel_param_10,
	.param .u32 _sk06_mlp_down_kernel_param_11,
	.param .u32 _sk06_mlp_down_kernel_param_12,
	.param .u32 _sk06_mlp_down_kernel_param_13,
	.param .u32 _sk06_mlp_down_kernel_param_14,
	.param .u32 _sk06_mlp_down_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_16,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_17
)
.reqntid 128
{
	.reg .pred 	%p<26>;
	.reg .b16 	%rs<129>;
	.reg .b32 	%r<949>;
	.reg .b64 	%rd<207>;
	.loc	1 304 0                         // sk06_mlp_down.py:304:0
$L__func_begin0:
	.loc	1 304 0                         // sk06_mlp_down.py:304:0

// %bb.0:
	ld.param.b32 	%r25, [_sk06_mlp_down_kernel_param_14];
	ld.param.b32 	%r24, [_sk06_mlp_down_kernel_param_13];
	ld.param.b32 	%r23, [_sk06_mlp_down_kernel_param_12];
	ld.param.b32 	%r22, [_sk06_mlp_down_kernel_param_8];
	ld.param.b32 	%r21, [_sk06_mlp_down_kernel_param_7];
	ld.param.b64 	%rd19, [_sk06_mlp_down_kernel_param_5];
	ld.param.b64 	%rd18, [_sk06_mlp_down_kernel_param_4];
	ld.param.b64 	%rd17, [_sk06_mlp_down_kernel_param_3];
	ld.param.b64 	%rd16, [_sk06_mlp_down_kernel_param_2];
	ld.param.b64 	%rd15, [_sk06_mlp_down_kernel_param_1];
	ld.param.b64 	%rd14, [_sk06_mlp_down_kernel_param_0];
$L__tmp0:
	.loc	1 313 24                        // sk06_mlp_down.py:313:24
	mov.u32 	%r65, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk06_mlp_down.py:314:27 ]
	add.s32 	%r66, %r21, 63;
	.loc	2 43 30                         // standard.py:43:30 @[ sk06_mlp_down.py:314:27 ]
	shr.s32 	%r67, %r66, 31;
	shr.u32 	%r68, %r67, 26;
	add.s32 	%r69, %r66, %r68;
	shr.s32 	%r70, %r69, 6;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk06_mlp_down.py:315:27 ]
	add.s32 	%r71, %r22, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk06_mlp_down.py:315:27 ]
	shr.s32 	%r72, %r71, 31;
	shr.u32 	%r73, %r72, 25;
	add.s32 	%r74, %r71, %r73;
	shr.s32 	%r75, %r74, 7;
$L__tmp3:
	.loc	1 316 29                        // sk06_mlp_down.py:316:29
	shl.b32 	%r76, %r75, 3;
	.loc	1 317 22                        // sk06_mlp_down.py:317:22
	div.s32 	%r77, %r65, %r76;
	.loc	1 317 38                        // sk06_mlp_down.py:317:38
	shl.b32 	%r78, %r77, 3;
	ld.param.b32 	%r79, [_sk06_mlp_down_kernel_param_9];
	.loc	1 318 30                        // sk06_mlp_down.py:318:30
	sub.s32 	%r80, %r70, %r78;
	ld.param.b32 	%r81, [_sk06_mlp_down_kernel_param_10];
	.loc	1 318 39                        // sk06_mlp_down.py:318:39
	min.s32 	%r82, %r80, 8;
	ld.param.b32 	%r83, [_sk06_mlp_down_kernel_param_11];
	.loc	1 319 30                        // sk06_mlp_down.py:319:30
	mul.lo.s32 	%r84, %r77, %r76;
	sub.s32 	%r85, %r65, %r84;
	.loc	1 320 36                        // sk06_mlp_down.py:320:36
	div.s32 	%r86, %r85, %r82;
	.loc	1 319 46                        // sk06_mlp_down.py:319:46
	mul.lo.s32 	%r87, %r86, %r82;
	sub.s32 	%r88, %r85, %r87;
	.loc	1 319 23                        // sk06_mlp_down.py:319:23
	add.s32 	%r89, %r88, %r78;
	.loc	1 322 22                        // sk06_mlp_down.py:322:22
	shl.b32 	%r1, %r89, 6;
	.loc	1 322 45                        // sk06_mlp_down.py:322:45
	mov.u32 	%r2, %tid.x;
	and.b32 	%r3, %r2, 120;
	bfe.u32 	%r90, %r2, 3, 4;
	or.b32 	%r91, %r90, 16;
	or.b32 	%r92, %r90, 32;
	or.b32 	%r93, %r90, 48;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r94, %r1, %r90;
	or.b32 	%r95, %r1, %r91;
	or.b32 	%r96, %r1, %r92;
	or.b32 	%r97, %r1, %r93;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r4, %r94, %r21;
	rem.s32 	%r5, %r95, %r21;
	rem.s32 	%r6, %r96, %r21;
	rem.s32 	%r7, %r97, %r21;
	.loc	1 323 22                        // sk06_mlp_down.py:323:22
	shl.b32 	%r8, %r86, 7;
	.loc	1 323 45                        // sk06_mlp_down.py:323:45
	and.b32 	%r9, %r2, 7;
	shl.b32 	%r98, %r9, 4;
	and.b32 	%r10, %r2, 15;
	and.b32 	%r11, %r2, 127;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r99, %r8, %r90;
	or.b32 	%r100, %r8, %r91;
	or.b32 	%r101, %r8, %r92;
	or.b32 	%r102, %r8, %r93;
	or.b32 	%r103, %r99, 64;
	or.b32 	%r104, %r99, 80;
	or.b32 	%r105, %r99, 96;
	or.b32 	%r106, %r99, 112;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r107, %r99, %r22;
	rem.s32 	%r108, %r100, %r22;
	rem.s32 	%r109, %r101, %r22;
	rem.s32 	%r110, %r102, %r22;
	rem.s32 	%r111, %r103, %r22;
	rem.s32 	%r112, %r104, %r22;
	rem.s32 	%r113, %r105, %r22;
	rem.s32 	%r114, %r106, %r22;
	.loc	1 326 39                        // sk06_mlp_down.py:326:39
	mul.lo.s32 	%r115, %r4, %r81;
	mul.lo.s32 	%r116, %r5, %r81;
	mul.lo.s32 	%r117, %r6, %r81;
	mul.lo.s32 	%r118, %r7, %r81;
	.loc	1 326 21                        // sk06_mlp_down.py:326:21
	cvt.s64.s32 	%rd1, %r115;
	add.s64 	%rd56, %rd14, %rd1;
	cvt.s64.s32 	%rd2, %r116;
	add.s64 	%rd57, %rd14, %rd2;
	cvt.s64.s32 	%rd3, %r117;
	add.s64 	%rd58, %rd14, %rd3;
	cvt.s64.s32 	%rd4, %r118;
	add.s64 	%rd59, %rd14, %rd4;
	.loc	1 326 51                        // sk06_mlp_down.py:326:51
	cvt.u64.u32 	%rd5, %r98;
	add.s64 	%rd20, %rd56, %rd5;
	add.s64 	%rd21, %rd57, %rd5;
	add.s64 	%rd22, %rd58, %rd5;
	add.s64 	%rd23, %rd59, %rd5;
	.loc	1 327 21                        // sk06_mlp_down.py:327:21
	add.s64 	%rd60, %rd15, %rd5;
	.loc	1 327 69                        // sk06_mlp_down.py:327:69
	mul.lo.s32 	%r119, %r107, %r83;
	mul.lo.s32 	%r120, %r108, %r83;
	mul.lo.s32 	%r121, %r109, %r83;
	mul.lo.s32 	%r122, %r110, %r83;
	mul.lo.s32 	%r123, %r111, %r83;
	mul.lo.s32 	%r124, %r112, %r83;
	mul.lo.s32 	%r125, %r113, %r83;
	mul.lo.s32 	%r126, %r114, %r83;
	.loc	1 327 51                        // sk06_mlp_down.py:327:51
	cvt.s64.s32 	%rd6, %r119;
	add.s64 	%rd24, %rd60, %rd6;
	cvt.s64.s32 	%rd7, %r120;
	add.s64 	%rd25, %rd60, %rd7;
	cvt.s64.s32 	%rd8, %r121;
	add.s64 	%rd26, %rd60, %rd8;
	cvt.s64.s32 	%rd9, %r122;
	add.s64 	%rd27, %rd60, %rd9;
	cvt.s64.s32 	%rd10, %r123;
	add.s64 	%rd28, %rd60, %rd10;
	cvt.s64.s32 	%rd11, %r124;
	add.s64 	%rd29, %rd60, %rd11;
	cvt.s64.s32 	%rd12, %r125;
	add.s64 	%rd30, %rd60, %rd12;
	cvt.s64.s32 	%rd13, %r126;
	add.s64 	%rd31, %rd60, %rd13;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	setp.gt.s32 	%p1, %r79, 127;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	shl.b32 	%r130, %r11, 4;
	and.b32 	%r13, %r2, 56;
	shl.b32 	%r131, %r13, 1;
	xor.b32 	%r132, %r130, %r131;
	mov.b32 	%r133, global_smem;
	add.s32 	%r31, %r133, %r132;
	add.s32 	%r26, %r31, 49152;
	selp.b32 	%r27, 16, 0, %p1;
	// begin inline asm
	cp.async.cg.shared.global [ %r26 + 0 ], [ %rd20 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r28, %r31, 51200;
	// begin inline asm
	cp.async.cg.shared.global [ %r28 + 0 ], [ %rd21 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r29, %r31, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r29 + 0 ], [ %rd22 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r30, %r31, 55296;
	// begin inline asm
	cp.async.cg.shared.global [ %r30 + 0 ], [ %rd23 + 0 ], 0x10, %r27;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	// begin inline asm
	cp.async.cg.shared.global [ %r31 + 0 ], [ %rd24 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r32, %r31, 2048;
	// begin inline asm
	cp.async.cg.shared.global [ %r32 + 0 ], [ %rd25 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r33, %r31, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r33 + 0 ], [ %rd26 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r34, %r31, 6144;
	// begin inline asm
	cp.async.cg.shared.global [ %r34 + 0 ], [ %rd27 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r35, %r31, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r35 + 0 ], [ %rd28 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r36, %r31, 10240;
	// begin inline asm
	cp.async.cg.shared.global [ %r36 + 0 ], [ %rd29 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r37, %r31, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r37 + 0 ], [ %rd30 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r38, %r31, 14336;
	// begin inline asm
	cp.async.cg.shared.global [ %r38 + 0 ], [ %rd31 + 0 ], 0x10, %r27;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	setp.gt.s32 	%p2, %r79, 255;
	.loc	1 344 18                        // sk06_mlp_down.py:344:18
	add.s64 	%rd32, %rd20, 128;
	add.s64 	%rd33, %rd21, 128;
	add.s64 	%rd34, %rd22, 128;
	add.s64 	%rd35, %rd23, 128;
	.loc	1 345 18                        // sk06_mlp_down.py:345:18
	add.s64 	%rd36, %rd24, 128;
	add.s64 	%rd37, %rd25, 128;
	add.s64 	%rd38, %rd26, 128;
	add.s64 	%rd39, %rd27, 128;
	add.s64 	%rd40, %rd28, 128;
	add.s64 	%rd41, %rd29, 128;
	add.s64 	%rd42, %rd30, 128;
	add.s64 	%rd43, %rd31, 128;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	bar.sync 	0;
	add.s32 	%r39, %r31, 57344;
	selp.b32 	%r40, 16, 0, %p2;
	// begin inline asm
	cp.async.cg.shared.global [ %r39 + 0 ], [ %rd32 + 0 ], 0x10, %r40;
	// end inline asm
	add.s32 	%r41, %r31, 59392;
	// begin inline asm
	cp.async.cg.shared.global [ %r41 + 0 ], [ %rd33 + 0 ], 0x10, %r40;
	// end inline asm
	add.s32 	%r42, %r31, 61440;
	// begin inline asm
	cp.async.cg.shared.global [ %r42 + 0 ], [ %rd34 + 0 ], 0x10, %r40;
	// end inline asm
	add.s32 	%r43, %r31, 63488;
	// begin inline asm
	cp.async.cg.shared.global [ %r43 + 0 ], [ %rd35 + 0 ], 0x10, %r40;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	add.s32 	%r44, %r31, 16384;
	// begin inline asm
	cp.async.cg.shared.global [ %r44 + 0 ], [ %rd36 + 0 ], 0x10, %r40;
	// end inline asm
	add.s32 	%r45, %r31, 18432;
	// begin inline asm
	cp.async.cg.shared.global [ %r45 + 0 ], [ %rd37 + 0 ], 0x10, %r40;
	// end inline asm
	add.s32 	%r46, %r31, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r46 + 0 ], [ %rd38 + 0 ], 0x10, %r40;
	// end inline asm
	add.s32 	%r47, %r31, 22528;
	// begin inline asm
	cp.async.cg.shared.global [ %r47 + 0 ], [ %rd39 + 0 ], 0x10, %r40;
	// end inline asm
	add.s32 	%r48, %r31, 24576;
	// begin inline asm
	cp.async.cg.shared.global [ %r48 + 0 ], [ %rd40 + 0 ], 0x10, %r40;
	// end inline asm
	add.s32 	%r49, %r31, 26624;
	// begin inline asm
	cp.async.cg.shared.global [ %r49 + 0 ], [ %rd41 + 0 ], 0x10, %r40;
	// end inline asm
	add.s32 	%r50, %r31, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r50 + 0 ], [ %rd42 + 0 ], 0x10, %r40;
	// end inline asm
	add.s32 	%r51, %r31, 30720;
	// begin inline asm
	cp.async.cg.shared.global [ %r51 + 0 ], [ %rd43 + 0 ], 0x10, %r40;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	setp.gt.s32 	%p3, %r79, 383;
	.loc	1 344 18                        // sk06_mlp_down.py:344:18
	add.s64 	%rd44, %rd20, 256;
	add.s64 	%rd45, %rd21, 256;
	add.s64 	%rd46, %rd22, 256;
	add.s64 	%rd47, %rd23, 256;
	.loc	1 345 18                        // sk06_mlp_down.py:345:18
	add.s64 	%rd48, %rd24, 256;
	add.s64 	%rd49, %rd25, 256;
	add.s64 	%rd50, %rd26, 256;
	add.s64 	%rd51, %rd27, 256;
	add.s64 	%rd52, %rd28, 256;
	add.s64 	%rd53, %rd29, 256;
	add.s64 	%rd54, %rd30, 256;
	add.s64 	%rd55, %rd31, 256;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	bar.sync 	0;
	add.s32 	%r52, %r31, 65536;
	selp.b32 	%r53, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r52 + 0 ], [ %rd44 + 0 ], 0x10, %r53;
	// end inline asm
	add.s32 	%r54, %r31, 67584;
	// begin inline asm
	cp.async.cg.shared.global [ %r54 + 0 ], [ %rd45 + 0 ], 0x10, %r53;
	// end inline asm
	add.s32 	%r55, %r31, 69632;
	// begin inline asm
	cp.async.cg.shared.global [ %r55 + 0 ], [ %rd46 + 0 ], 0x10, %r53;
	// end inline asm
	add.s32 	%r56, %r31, 71680;
	// begin inline asm
	cp.async.cg.shared.global [ %r56 + 0 ], [ %rd47 + 0 ], 0x10, %r53;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	add.s32 	%r57, %r31, 32768;
	// begin inline asm
	cp.async.cg.shared.global [ %r57 + 0 ], [ %rd48 + 0 ], 0x10, %r53;
	// end inline asm
	add.s32 	%r58, %r31, 34816;
	// begin inline asm
	cp.async.cg.shared.global [ %r58 + 0 ], [ %rd49 + 0 ], 0x10, %r53;
	// end inline asm
	add.s32 	%r59, %r31, 36864;
	// begin inline asm
	cp.async.cg.shared.global [ %r59 + 0 ], [ %rd50 + 0 ], 0x10, %r53;
	// end inline asm
	add.s32 	%r60, %r31, 38912;
	// begin inline asm
	cp.async.cg.shared.global [ %r60 + 0 ], [ %rd51 + 0 ], 0x10, %r53;
	// end inline asm
	add.s32 	%r61, %r31, 40960;
	// begin inline asm
	cp.async.cg.shared.global [ %r61 + 0 ], [ %rd52 + 0 ], 0x10, %r53;
	// end inline asm
	add.s32 	%r62, %r31, 43008;
	// begin inline asm
	cp.async.cg.shared.global [ %r62 + 0 ], [ %rd53 + 0 ], 0x10, %r53;
	// end inline asm
	add.s32 	%r63, %r31, 45056;
	// begin inline asm
	cp.async.cg.shared.global [ %r63 + 0 ], [ %rd54 + 0 ], 0x10, %r53;
	// end inline asm
	add.s32 	%r64, %r31, 47104;
	// begin inline asm
	cp.async.cg.shared.global [ %r64 + 0 ], [ %rd55 + 0 ], 0x10, %r53;
	// end inline asm
	cp.async.commit_group;
	cvt.u32.u64 	%r880, %rd5;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	@%p1 bra 	$L__BB0_2;
	bra.uni 	$L__BB0_1;
$L__BB0_2:                              // %.lr.ph
	.loc	1 0 23                          // sk06_mlp_down.py:0:23
	shr.s32 	%r127, %r79, 31;
	shr.u32 	%r128, %r127, 25;
	add.s32 	%r129, %r79, %r128;
	shr.s32 	%r12, %r129, 7;
	add.s32 	%r14, %r12, -3;
	shl.b32 	%r134, %r10, 7;
	and.b32 	%r884, %r2, 16;
	or.b32 	%r135, %r134, %r880;
	xor.b32 	%r15, %r135, %r884;
	xor.b32 	%r16, %r15, 32;
	xor.b32 	%r17, %r15, 64;
	xor.b32 	%r18, %r15, 96;
	shl.b32 	%r136, %r9, 7;
	shl.b32 	%r137, %r2, 5;
	and.b32 	%r138, %r137, 3072;
	shl.b32 	%r139, %r2, 1;
	and.b32 	%r140, %r139, 48;
	or.b32 	%r141, %r136, %r138;
	xor.b32 	%r142, %r880, %r140;
	or.b32 	%r19, %r141, %r142;
	xor.b32 	%r20, %r19, 64;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	add.s64 	%rd61, %rd13, %rd15;
	add.s64 	%rd206, %rd61, 384;
	add.s64 	%rd62, %rd12, %rd15;
	add.s64 	%rd205, %rd62, 384;
	add.s64 	%rd63, %rd11, %rd15;
	add.s64 	%rd204, %rd63, 384;
	add.s64 	%rd64, %rd10, %rd15;
	add.s64 	%rd203, %rd64, 384;
	add.s64 	%rd65, %rd9, %rd15;
	add.s64 	%rd202, %rd65, 384;
	add.s64 	%rd66, %rd8, %rd15;
	add.s64 	%rd201, %rd66, 384;
	add.s64 	%rd67, %rd7, %rd15;
	add.s64 	%rd200, %rd67, 384;
	add.s64 	%rd68, %rd6, %rd15;
	add.s64 	%rd199, %rd68, 384;
	add.s64 	%rd69, %rd4, %rd14;
	add.s64 	%rd198, %rd69, 384;
	add.s64 	%rd70, %rd3, %rd14;
	add.s64 	%rd197, %rd70, 384;
	add.s64 	%rd71, %rd2, %rd14;
	add.s64 	%rd196, %rd71, 384;
	add.s64 	%rd72, %rd1, %rd14;
	add.s64 	%rd195, %rd72, 384;
	mov.b32 	%r885, 0f00000000;
	mov.b32 	%r143, 0;
	mov.b32 	%r882, 2;
	mov.b32 	%r881, -1;
	mov.b32 	%r883, %r143;
	mov.b32 	%r886, %r885;
	mov.b32 	%r887, %r885;
	mov.b32 	%r888, %r885;
	mov.b32 	%r889, %r885;
	mov.b32 	%r890, %r885;
	mov.b32 	%r891, %r885;
	mov.b32 	%r892, %r885;
	mov.b32 	%r893, %r885;
	mov.b32 	%r894, %r885;
	mov.b32 	%r895, %r885;
	mov.b32 	%r896, %r885;
	mov.b32 	%r897, %r885;
	mov.b32 	%r898, %r885;
	mov.b32 	%r899, %r885;
	mov.b32 	%r900, %r885;
	mov.b32 	%r901, %r885;
	mov.b32 	%r902, %r885;
	mov.b32 	%r903, %r885;
	mov.b32 	%r904, %r885;
	mov.b32 	%r905, %r885;
	mov.b32 	%r906, %r885;
	mov.b32 	%r907, %r885;
	mov.b32 	%r908, %r885;
	mov.b32 	%r909, %r885;
	mov.b32 	%r910, %r885;
	mov.b32 	%r911, %r885;
	mov.b32 	%r912, %r885;
	mov.b32 	%r913, %r885;
	mov.b32 	%r914, %r885;
	mov.b32 	%r915, %r885;
	mov.b32 	%r916, %r885;
	mov.b32 	%r917, %r885;
	mov.b32 	%r918, %r885;
	mov.b32 	%r919, %r885;
	mov.b32 	%r920, %r885;
	mov.b32 	%r921, %r885;
	mov.b32 	%r922, %r885;
	mov.b32 	%r923, %r885;
	mov.b32 	%r924, %r885;
	mov.b32 	%r925, %r885;
	mov.b32 	%r926, %r885;
	mov.b32 	%r927, %r885;
	mov.b32 	%r928, %r885;
	mov.b32 	%r929, %r885;
	mov.b32 	%r930, %r885;
	mov.b32 	%r931, %r885;
	mov.b32 	%r932, %r885;
	mov.b32 	%r933, %r885;
	mov.b32 	%r934, %r885;
	mov.b32 	%r935, %r885;
	mov.b32 	%r936, %r885;
	mov.b32 	%r937, %r885;
	mov.b32 	%r938, %r885;
	mov.b32 	%r939, %r885;
	mov.b32 	%r940, %r885;
	mov.b32 	%r941, %r885;
	mov.b32 	%r942, %r885;
	mov.b32 	%r943, %r885;
	mov.b32 	%r944, %r885;
	mov.b32 	%r945, %r885;
	mov.b32 	%r946, %r885;
	mov.b32 	%r947, %r885;
	mov.b32 	%r948, %r885;
$L__BB0_3:                              // =>This Inner Loop Header: Depth=1
	setp.lt.s32 	%p4, %r883, %r14;
	add.s32 	%r317, %r881, 1;
	setp.gt.s32 	%p5, %r317, 2;
	selp.b32 	%r881, 0, %r317, %p5;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	cp.async.wait_group 	4;
	bar.sync 	0;
	shl.b32 	%r318, %r881, 13;
	add.s32 	%r319, %r133, %r318;
	add.s32 	%r320, %r319, %r15;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r144, %r145, %r146, %r147}, [%r320+49152];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r156, %r157, %r158, %r159}, [%r320+51200];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r160, %r161, %r162, %r163}, [%r320+53248];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r164, %r165, %r166, %r167}, [%r320+55296];
	add.s32 	%r321, %r319, %r16;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r172, %r173, %r174, %r175}, [%r321+49152];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r200, %r201, %r202, %r203}, [%r321+51200];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r220, %r221, %r222, %r223}, [%r321+53248];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r240, %r241, %r242, %r243}, [%r321+55296];
	add.s32 	%r322, %r319, %r17;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r256, %r257, %r258, %r259}, [%r322+49152];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r268, %r269, %r270, %r271}, [%r322+51200];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r272, %r273, %r274, %r275}, [%r322+53248];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r276, %r277, %r278, %r279}, [%r322+55296];
	add.s32 	%r323, %r319, %r18;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r280, %r281, %r282, %r283}, [%r323+49152];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r292, %r293, %r294, %r295}, [%r323+51200];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r296, %r297, %r298, %r299}, [%r323+53248];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r300, %r301, %r302, %r303}, [%r323+55296];
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	add.s32 	%r324, %r319, %r318;
	add.s32 	%r325, %r324, %r19;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r148, %r149, %r176, %r177}, [%r325];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r150, %r151, %r182, %r183}, [%r325+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r152, %r153, %r188, %r189}, [%r325+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r154, %r155, %r194, %r195}, [%r325+12288];
	add.s32 	%r326, %r324, %r20;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r260, %r261, %r284, %r285}, [%r326];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r262, %r263, %r286, %r287}, [%r326+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r264, %r265, %r288, %r289}, [%r326+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r266, %r267, %r290, %r291}, [%r326+12288];
	.loc	1 338 36                        // sk06_mlp_down.py:338:36
	mov.b32 	%r168, %r143;
	mov.b32 	%r169, %r143;
	mov.b32 	%r170, %r143;
	mov.b32 	%r171, %r143;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r168, %r169, %r170, %r171 }, { %r144, %r145, %r146, %r147 }, { %r148, %r149 }, { %r168, %r169, %r170, %r171 };
	// end inline asm
	mov.b32 	%r178, %r143;
	mov.b32 	%r179, %r143;
	mov.b32 	%r180, %r143;
	mov.b32 	%r181, %r143;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r178, %r179, %r180, %r181 }, { %r144, %r145, %r146, %r147 }, { %r150, %r151 }, { %r178, %r179, %r180, %r181 };
	// end inline asm
	mov.b32 	%r184, %r143;
	mov.b32 	%r185, %r143;
	mov.b32 	%r186, %r143;
	mov.b32 	%r187, %r143;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r184, %r185, %r186, %r187 }, { %r144, %r145, %r146, %r147 }, { %r152, %r153 }, { %r184, %r185, %r186, %r187 };
	// end inline asm
	mov.b32 	%r190, %r143;
	mov.b32 	%r191, %r143;
	mov.b32 	%r192, %r143;
	mov.b32 	%r193, %r143;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r190, %r191, %r192, %r193 }, { %r144, %r145, %r146, %r147 }, { %r154, %r155 }, { %r190, %r191, %r192, %r193 };
	// end inline asm
	mov.b32 	%r196, %r143;
	mov.b32 	%r197, %r143;
	mov.b32 	%r198, %r143;
	mov.b32 	%r199, %r143;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r196, %r197, %r198, %r199 }, { %r156, %r157, %r158, %r159 }, { %r148, %r149 }, { %r196, %r197, %r198, %r199 };
	// end inline asm
	mov.b32 	%r204, %r143;
	mov.b32 	%r205, %r143;
	mov.b32 	%r206, %r143;
	mov.b32 	%r207, %r143;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r204, %r205, %r206, %r207 }, { %r156, %r157, %r158, %r159 }, { %r150, %r151 }, { %r204, %r205, %r206, %r207 };
	// end inline asm
	mov.b32 	%r208, %r143;
	mov.b32 	%r209, %r143;
	mov.b32 	%r210, %r143;
	mov.b32 	%r211, %r143;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r208, %r209, %r210, %r211 }, { %r156, %r157, %r158, %r159 }, { %r152, %r153 }, { %r208, %r209, %r210, %r211 };
	// end inline asm
	mov.b32 	%r212, %r143;
	mov.b32 	%r213, %r143;
	mov.b32 	%r214, %r143;
	mov.b32 	%r215, %r143;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r212, %r213, %r214, %r215 }, { %r156, %r157, %r158, %r159 }, { %r154, %r155 }, { %r212, %r213, %r214, %r215 };
	// end inline asm
	mov.b32 	%r216, %r143;
	mov.b32 	%r217, %r143;
	mov.b32 	%r218, %r143;
	mov.b32 	%r219, %r143;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r216, %r217, %r218, %r219 }, { %r160, %r161, %r162, %r163 }, { %r148, %r149 }, { %r216, %r217, %r218, %r219 };
	// end inline asm
	mov.b32 	%r224, %r143;
	mov.b32 	%r225, %r143;
	mov.b32 	%r226, %r143;
	mov.b32 	%r227, %r143;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r224, %r225, %r226, %r227 }, { %r160, %r161, %r162, %r163 }, { %r150, %r151 }, { %r224, %r225, %r226, %r227 };
	// end inline asm
	mov.b32 	%r228, %r143;
	mov.b32 	%r229, %r143;
	mov.b32 	%r230, %r143;
	mov.b32 	%r231, %r143;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r228, %r229, %r230, %r231 }, { %r160, %r161, %r162, %r163 }, { %r152, %r153 }, { %r228, %r229, %r230, %r231 };
	// end inline asm
	mov.b32 	%r232, %r143;
	mov.b32 	%r233, %r143;
	mov.b32 	%r234, %r143;
	mov.b32 	%r235, %r143;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r232, %r233, %r234, %r235 }, { %r160, %r161, %r162, %r163 }, { %r154, %r155 }, { %r232, %r233, %r234, %r235 };
	// end inline asm
	mov.b32 	%r236, %r143;
	mov.b32 	%r237, %r143;
	mov.b32 	%r238, %r143;
	mov.b32 	%r239, %r143;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r236, %r237, %r238, %r239 }, { %r164, %r165, %r166, %r167 }, { %r148, %r149 }, { %r236, %r237, %r238, %r239 };
	// end inline asm
	mov.b32 	%r244, %r143;
	mov.b32 	%r245, %r143;
	mov.b32 	%r246, %r143;
	mov.b32 	%r247, %r143;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r244, %r245, %r246, %r247 }, { %r164, %r165, %r166, %r167 }, { %r150, %r151 }, { %r244, %r245, %r246, %r247 };
	// end inline asm
	mov.b32 	%r248, %r143;
	mov.b32 	%r249, %r143;
	mov.b32 	%r250, %r143;
	mov.b32 	%r251, %r143;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r248, %r249, %r250, %r251 }, { %r164, %r165, %r166, %r167 }, { %r152, %r153 }, { %r248, %r249, %r250, %r251 };
	// end inline asm
	mov.b32 	%r252, %r143;
	mov.b32 	%r253, %r143;
	mov.b32 	%r254, %r143;
	mov.b32 	%r255, %r143;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r252, %r253, %r254, %r255 }, { %r164, %r165, %r166, %r167 }, { %r154, %r155 }, { %r252, %r253, %r254, %r255 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r168, %r169, %r170, %r171 }, { %r172, %r173, %r174, %r175 }, { %r176, %r177 }, { %r168, %r169, %r170, %r171 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r178, %r179, %r180, %r181 }, { %r172, %r173, %r174, %r175 }, { %r182, %r183 }, { %r178, %r179, %r180, %r181 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r184, %r185, %r186, %r187 }, { %r172, %r173, %r174, %r175 }, { %r188, %r189 }, { %r184, %r185, %r186, %r187 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r190, %r191, %r192, %r193 }, { %r172, %r173, %r174, %r175 }, { %r194, %r195 }, { %r190, %r191, %r192, %r193 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r196, %r197, %r198, %r199 }, { %r200, %r201, %r202, %r203 }, { %r176, %r177 }, { %r196, %r197, %r198, %r199 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r204, %r205, %r206, %r207 }, { %r200, %r201, %r202, %r203 }, { %r182, %r183 }, { %r204, %r205, %r206, %r207 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r208, %r209, %r210, %r211 }, { %r200, %r201, %r202, %r203 }, { %r188, %r189 }, { %r208, %r209, %r210, %r211 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r212, %r213, %r214, %r215 }, { %r200, %r201, %r202, %r203 }, { %r194, %r195 }, { %r212, %r213, %r214, %r215 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r216, %r217, %r218, %r219 }, { %r220, %r221, %r222, %r223 }, { %r176, %r177 }, { %r216, %r217, %r218, %r219 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r224, %r225, %r226, %r227 }, { %r220, %r221, %r222, %r223 }, { %r182, %r183 }, { %r224, %r225, %r226, %r227 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r228, %r229, %r230, %r231 }, { %r220, %r221, %r222, %r223 }, { %r188, %r189 }, { %r228, %r229, %r230, %r231 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r232, %r233, %r234, %r235 }, { %r220, %r221, %r222, %r223 }, { %r194, %r195 }, { %r232, %r233, %r234, %r235 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r236, %r237, %r238, %r239 }, { %r240, %r241, %r242, %r243 }, { %r176, %r177 }, { %r236, %r237, %r238, %r239 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r244, %r245, %r246, %r247 }, { %r240, %r241, %r242, %r243 }, { %r182, %r183 }, { %r244, %r245, %r246, %r247 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r248, %r249, %r250, %r251 }, { %r240, %r241, %r242, %r243 }, { %r188, %r189 }, { %r248, %r249, %r250, %r251 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r252, %r253, %r254, %r255 }, { %r240, %r241, %r242, %r243 }, { %r194, %r195 }, { %r252, %r253, %r254, %r255 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r168, %r169, %r170, %r171 }, { %r256, %r257, %r258, %r259 }, { %r260, %r261 }, { %r168, %r169, %r170, %r171 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r178, %r179, %r180, %r181 }, { %r256, %r257, %r258, %r259 }, { %r262, %r263 }, { %r178, %r179, %r180, %r181 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r184, %r185, %r186, %r187 }, { %r256, %r257, %r258, %r259 }, { %r264, %r265 }, { %r184, %r185, %r186, %r187 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r190, %r191, %r192, %r193 }, { %r256, %r257, %r258, %r259 }, { %r266, %r267 }, { %r190, %r191, %r192, %r193 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r196, %r197, %r198, %r199 }, { %r268, %r269, %r270, %r271 }, { %r260, %r261 }, { %r196, %r197, %r198, %r199 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r204, %r205, %r206, %r207 }, { %r268, %r269, %r270, %r271 }, { %r262, %r263 }, { %r204, %r205, %r206, %r207 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r208, %r209, %r210, %r211 }, { %r268, %r269, %r270, %r271 }, { %r264, %r265 }, { %r208, %r209, %r210, %r211 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r212, %r213, %r214, %r215 }, { %r268, %r269, %r270, %r271 }, { %r266, %r267 }, { %r212, %r213, %r214, %r215 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r216, %r217, %r218, %r219 }, { %r272, %r273, %r274, %r275 }, { %r260, %r261 }, { %r216, %r217, %r218, %r219 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r224, %r225, %r226, %r227 }, { %r272, %r273, %r274, %r275 }, { %r262, %r263 }, { %r224, %r225, %r226, %r227 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r228, %r229, %r230, %r231 }, { %r272, %r273, %r274, %r275 }, { %r264, %r265 }, { %r228, %r229, %r230, %r231 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r232, %r233, %r234, %r235 }, { %r272, %r273, %r274, %r275 }, { %r266, %r267 }, { %r232, %r233, %r234, %r235 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r236, %r237, %r238, %r239 }, { %r276, %r277, %r278, %r279 }, { %r260, %r261 }, { %r236, %r237, %r238, %r239 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r244, %r245, %r246, %r247 }, { %r276, %r277, %r278, %r279 }, { %r262, %r263 }, { %r244, %r245, %r246, %r247 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r248, %r249, %r250, %r251 }, { %r276, %r277, %r278, %r279 }, { %r264, %r265 }, { %r248, %r249, %r250, %r251 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r252, %r253, %r254, %r255 }, { %r276, %r277, %r278, %r279 }, { %r266, %r267 }, { %r252, %r253, %r254, %r255 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r168, %r169, %r170, %r171 }, { %r280, %r281, %r282, %r283 }, { %r284, %r285 }, { %r168, %r169, %r170, %r171 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r178, %r179, %r180, %r181 }, { %r280, %r281, %r282, %r283 }, { %r286, %r287 }, { %r178, %r179, %r180, %r181 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r184, %r185, %r186, %r187 }, { %r280, %r281, %r282, %r283 }, { %r288, %r289 }, { %r184, %r185, %r186, %r187 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r190, %r191, %r192, %r193 }, { %r280, %r281, %r282, %r283 }, { %r290, %r291 }, { %r190, %r191, %r192, %r193 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r196, %r197, %r198, %r199 }, { %r292, %r293, %r294, %r295 }, { %r284, %r285 }, { %r196, %r197, %r198, %r199 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r204, %r205, %r206, %r207 }, { %r292, %r293, %r294, %r295 }, { %r286, %r287 }, { %r204, %r205, %r206, %r207 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r208, %r209, %r210, %r211 }, { %r292, %r293, %r294, %r295 }, { %r288, %r289 }, { %r208, %r209, %r210, %r211 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r212, %r213, %r214, %r215 }, { %r292, %r293, %r294, %r295 }, { %r290, %r291 }, { %r212, %r213, %r214, %r215 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r216, %r217, %r218, %r219 }, { %r296, %r297, %r298, %r299 }, { %r284, %r285 }, { %r216, %r217, %r218, %r219 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r224, %r225, %r226, %r227 }, { %r296, %r297, %r298, %r299 }, { %r286, %r287 }, { %r224, %r225, %r226, %r227 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r228, %r229, %r230, %r231 }, { %r296, %r297, %r298, %r299 }, { %r288, %r289 }, { %r228, %r229, %r230, %r231 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r232, %r233, %r234, %r235 }, { %r296, %r297, %r298, %r299 }, { %r290, %r291 }, { %r232, %r233, %r234, %r235 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r236, %r237, %r238, %r239 }, { %r300, %r301, %r302, %r303 }, { %r284, %r285 }, { %r236, %r237, %r238, %r239 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r244, %r245, %r246, %r247 }, { %r300, %r301, %r302, %r303 }, { %r286, %r287 }, { %r244, %r245, %r246, %r247 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r248, %r249, %r250, %r251 }, { %r300, %r301, %r302, %r303 }, { %r288, %r289 }, { %r248, %r249, %r250, %r251 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r252, %r253, %r254, %r255 }, { %r300, %r301, %r302, %r303 }, { %r290, %r291 }, { %r252, %r253, %r254, %r255 };
	// end inline asm
	.loc	1 343 19                        // sk06_mlp_down.py:343:19
	cvt.rn.f32.s32 	%r327, %r252;
	cvt.rn.f32.s32 	%r328, %r253;
	cvt.rn.f32.s32 	%r329, %r254;
	cvt.rn.f32.s32 	%r330, %r255;
	cvt.rn.f32.s32 	%r331, %r248;
	cvt.rn.f32.s32 	%r332, %r249;
	cvt.rn.f32.s32 	%r333, %r250;
	cvt.rn.f32.s32 	%r334, %r251;
	cvt.rn.f32.s32 	%r335, %r244;
	cvt.rn.f32.s32 	%r336, %r245;
	cvt.rn.f32.s32 	%r337, %r246;
	cvt.rn.f32.s32 	%r338, %r247;
	cvt.rn.f32.s32 	%r339, %r236;
	cvt.rn.f32.s32 	%r340, %r237;
	cvt.rn.f32.s32 	%r341, %r238;
	cvt.rn.f32.s32 	%r342, %r239;
	cvt.rn.f32.s32 	%r343, %r232;
	cvt.rn.f32.s32 	%r344, %r233;
	cvt.rn.f32.s32 	%r345, %r234;
	cvt.rn.f32.s32 	%r346, %r235;
	cvt.rn.f32.s32 	%r347, %r228;
	cvt.rn.f32.s32 	%r348, %r229;
	cvt.rn.f32.s32 	%r349, %r230;
	cvt.rn.f32.s32 	%r350, %r231;
	cvt.rn.f32.s32 	%r351, %r224;
	cvt.rn.f32.s32 	%r352, %r225;
	cvt.rn.f32.s32 	%r353, %r226;
	cvt.rn.f32.s32 	%r354, %r227;
	cvt.rn.f32.s32 	%r355, %r216;
	cvt.rn.f32.s32 	%r356, %r217;
	cvt.rn.f32.s32 	%r357, %r218;
	cvt.rn.f32.s32 	%r358, %r219;
	cvt.rn.f32.s32 	%r359, %r212;
	cvt.rn.f32.s32 	%r360, %r213;
	cvt.rn.f32.s32 	%r361, %r214;
	cvt.rn.f32.s32 	%r362, %r215;
	cvt.rn.f32.s32 	%r363, %r208;
	cvt.rn.f32.s32 	%r364, %r209;
	cvt.rn.f32.s32 	%r365, %r210;
	cvt.rn.f32.s32 	%r366, %r211;
	cvt.rn.f32.s32 	%r367, %r204;
	cvt.rn.f32.s32 	%r368, %r205;
	cvt.rn.f32.s32 	%r369, %r206;
	cvt.rn.f32.s32 	%r370, %r207;
	cvt.rn.f32.s32 	%r371, %r196;
	cvt.rn.f32.s32 	%r372, %r197;
	cvt.rn.f32.s32 	%r373, %r198;
	cvt.rn.f32.s32 	%r374, %r199;
	cvt.rn.f32.s32 	%r375, %r190;
	cvt.rn.f32.s32 	%r376, %r191;
	cvt.rn.f32.s32 	%r377, %r192;
	cvt.rn.f32.s32 	%r378, %r193;
	cvt.rn.f32.s32 	%r379, %r184;
	cvt.rn.f32.s32 	%r380, %r185;
	cvt.rn.f32.s32 	%r381, %r186;
	cvt.rn.f32.s32 	%r382, %r187;
	cvt.rn.f32.s32 	%r383, %r178;
	cvt.rn.f32.s32 	%r384, %r179;
	cvt.rn.f32.s32 	%r385, %r180;
	cvt.rn.f32.s32 	%r386, %r181;
	cvt.rn.f32.s32 	%r387, %r168;
	cvt.rn.f32.s32 	%r388, %r169;
	cvt.rn.f32.s32 	%r389, %r170;
	cvt.rn.f32.s32 	%r390, %r171;
	add.f32 	%r888, %r888, %r390;
	add.f32 	%r887, %r887, %r389;
	add.f32 	%r886, %r886, %r388;
	add.f32 	%r885, %r885, %r387;
	add.f32 	%r892, %r892, %r386;
	add.f32 	%r891, %r891, %r385;
	add.f32 	%r890, %r890, %r384;
	add.f32 	%r889, %r889, %r383;
	add.f32 	%r896, %r896, %r382;
	add.f32 	%r895, %r895, %r381;
	add.f32 	%r894, %r894, %r380;
	add.f32 	%r893, %r893, %r379;
	add.f32 	%r900, %r900, %r378;
	add.f32 	%r899, %r899, %r377;
	add.f32 	%r898, %r898, %r376;
	add.f32 	%r897, %r897, %r375;
	add.f32 	%r904, %r904, %r374;
	add.f32 	%r903, %r903, %r373;
	add.f32 	%r902, %r902, %r372;
	add.f32 	%r901, %r901, %r371;
	add.f32 	%r908, %r908, %r370;
	add.f32 	%r907, %r907, %r369;
	add.f32 	%r906, %r906, %r368;
	add.f32 	%r905, %r905, %r367;
	add.f32 	%r912, %r912, %r366;
	add.f32 	%r911, %r911, %r365;
	add.f32 	%r910, %r910, %r364;
	add.f32 	%r909, %r909, %r363;
	add.f32 	%r916, %r916, %r362;
	add.f32 	%r915, %r915, %r361;
	add.f32 	%r914, %r914, %r360;
	add.f32 	%r913, %r913, %r359;
	add.f32 	%r920, %r920, %r358;
	add.f32 	%r919, %r919, %r357;
	add.f32 	%r918, %r918, %r356;
	add.f32 	%r917, %r917, %r355;
	add.f32 	%r924, %r924, %r354;
	add.f32 	%r923, %r923, %r353;
	add.f32 	%r922, %r922, %r352;
	add.f32 	%r921, %r921, %r351;
	add.f32 	%r928, %r928, %r350;
	add.f32 	%r927, %r927, %r349;
	add.f32 	%r926, %r926, %r348;
	add.f32 	%r925, %r925, %r347;
	add.f32 	%r932, %r932, %r346;
	add.f32 	%r931, %r931, %r345;
	add.f32 	%r930, %r930, %r344;
	add.f32 	%r929, %r929, %r343;
	add.f32 	%r936, %r936, %r342;
	add.f32 	%r935, %r935, %r341;
	add.f32 	%r934, %r934, %r340;
	add.f32 	%r933, %r933, %r339;
	add.f32 	%r940, %r940, %r338;
	add.f32 	%r939, %r939, %r337;
	add.f32 	%r938, %r938, %r336;
	add.f32 	%r937, %r937, %r335;
	add.f32 	%r944, %r944, %r334;
	add.f32 	%r943, %r943, %r333;
	add.f32 	%r942, %r942, %r332;
	add.f32 	%r941, %r941, %r331;
	add.f32 	%r948, %r948, %r330;
	add.f32 	%r947, %r947, %r329;
	add.f32 	%r946, %r946, %r328;
	add.f32 	%r945, %r945, %r327;
	.loc	1 344 18                        // sk06_mlp_down.py:344:18
	add.s64 	%rd73, %rd195, %rd5;
	add.s64 	%rd74, %rd196, %rd5;
	add.s64 	%rd75, %rd197, %rd5;
	.loc	1 345 18                        // sk06_mlp_down.py:345:18
	add.s64 	%rd76, %rd198, %rd5;
	add.s64 	%rd77, %rd199, %rd5;
	add.s64 	%rd78, %rd200, %rd5;
	add.s64 	%rd79, %rd201, %rd5;
	add.s64 	%rd80, %rd202, %rd5;
	add.s64 	%rd81, %rd203, %rd5;
	add.s64 	%rd82, %rd204, %rd5;
	add.s64 	%rd83, %rd205, %rd5;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	add.s64 	%rd84, %rd206, %rd5;
	add.s32 	%r391, %r882, 1;
	setp.gt.s32 	%p6, %r391, 2;
	selp.b32 	%r882, 0, %r391, %p6;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	shl.b32 	%r392, %r882, 13;
	bar.sync 	0;
	add.s32 	%r393, %r31, %r392;
	add.s32 	%r304, %r393, 49152;
	selp.b32 	%r305, 16, 0, %p4;
	// begin inline asm
	cp.async.cg.shared.global [ %r304 + 0 ], [ %rd73 + 0 ], 0x10, %r305;
	// end inline asm
	add.s32 	%r306, %r393, 51200;
	// begin inline asm
	cp.async.cg.shared.global [ %r306 + 0 ], [ %rd74 + 0 ], 0x10, %r305;
	// end inline asm
	add.s32 	%r307, %r393, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r307 + 0 ], [ %rd75 + 0 ], 0x10, %r305;
	// end inline asm
	add.s32 	%r308, %r393, 55296;
	// begin inline asm
	cp.async.cg.shared.global [ %r308 + 0 ], [ %rd76 + 0 ], 0x10, %r305;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	add.s32 	%r309, %r393, %r392;
	// begin inline asm
	cp.async.cg.shared.global [ %r309 + 0 ], [ %rd77 + 0 ], 0x10, %r305;
	// end inline asm
	add.s32 	%r310, %r309, 2048;
	// begin inline asm
	cp.async.cg.shared.global [ %r310 + 0 ], [ %rd78 + 0 ], 0x10, %r305;
	// end inline asm
	add.s32 	%r311, %r309, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r311 + 0 ], [ %rd79 + 0 ], 0x10, %r305;
	// end inline asm
	add.s32 	%r312, %r309, 6144;
	// begin inline asm
	cp.async.cg.shared.global [ %r312 + 0 ], [ %rd80 + 0 ], 0x10, %r305;
	// end inline asm
	add.s32 	%r313, %r309, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r313 + 0 ], [ %rd81 + 0 ], 0x10, %r305;
	// end inline asm
	add.s32 	%r314, %r309, 10240;
	// begin inline asm
	cp.async.cg.shared.global [ %r314 + 0 ], [ %rd82 + 0 ], 0x10, %r305;
	// end inline asm
	add.s32 	%r315, %r309, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r315 + 0 ], [ %rd83 + 0 ], 0x10, %r305;
	// end inline asm
	add.s32 	%r316, %r309, 14336;
	// begin inline asm
	cp.async.cg.shared.global [ %r316 + 0 ], [ %rd84 + 0 ], 0x10, %r305;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	add.s32 	%r883, %r883, 1;
	add.s64 	%rd206, %rd206, 128;
	add.s64 	%rd205, %rd205, 128;
	add.s64 	%rd204, %rd204, 128;
	add.s64 	%rd203, %rd203, 128;
	add.s64 	%rd202, %rd202, 128;
	add.s64 	%rd201, %rd201, 128;
	add.s64 	%rd200, %rd200, 128;
	add.s64 	%rd199, %rd199, 128;
	add.s64 	%rd198, %rd198, 128;
	add.s64 	%rd197, %rd197, 128;
	add.s64 	%rd196, %rd196, 128;
	add.s64 	%rd195, %rd195, 128;
	setp.ne.b32 	%p7, %r12, %r883;
	@%p7 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_4;
$L__BB0_1:                              // %.._crit_edge_crit_edge
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	and.b32 	%r884, %r2, 16;
	mov.b32 	%r885, 0f00000000;
	mov.b32 	%r886, %r885;
	mov.b32 	%r887, %r885;
	mov.b32 	%r888, %r885;
	mov.b32 	%r889, %r885;
	mov.b32 	%r890, %r885;
	mov.b32 	%r891, %r885;
	mov.b32 	%r892, %r885;
	mov.b32 	%r893, %r885;
	mov.b32 	%r894, %r885;
	mov.b32 	%r895, %r885;
	mov.b32 	%r896, %r885;
	mov.b32 	%r897, %r885;
	mov.b32 	%r898, %r885;
	mov.b32 	%r899, %r885;
	mov.b32 	%r900, %r885;
	mov.b32 	%r901, %r885;
	mov.b32 	%r902, %r885;
	mov.b32 	%r903, %r885;
	mov.b32 	%r904, %r885;
	mov.b32 	%r905, %r885;
	mov.b32 	%r906, %r885;
	mov.b32 	%r907, %r885;
	mov.b32 	%r908, %r885;
	mov.b32 	%r909, %r885;
	mov.b32 	%r910, %r885;
	mov.b32 	%r911, %r885;
	mov.b32 	%r912, %r885;
	mov.b32 	%r913, %r885;
	mov.b32 	%r914, %r885;
	mov.b32 	%r915, %r885;
	mov.b32 	%r916, %r885;
	mov.b32 	%r917, %r885;
	mov.b32 	%r918, %r885;
	mov.b32 	%r919, %r885;
	mov.b32 	%r920, %r885;
	mov.b32 	%r921, %r885;
	mov.b32 	%r922, %r885;
	mov.b32 	%r923, %r885;
	mov.b32 	%r924, %r885;
	mov.b32 	%r925, %r885;
	mov.b32 	%r926, %r885;
	mov.b32 	%r927, %r885;
	mov.b32 	%r928, %r885;
	mov.b32 	%r929, %r885;
	mov.b32 	%r930, %r885;
	mov.b32 	%r931, %r885;
	mov.b32 	%r932, %r885;
	mov.b32 	%r933, %r885;
	mov.b32 	%r934, %r885;
	mov.b32 	%r935, %r885;
	mov.b32 	%r936, %r885;
	mov.b32 	%r937, %r885;
	mov.b32 	%r938, %r885;
	mov.b32 	%r939, %r885;
	mov.b32 	%r940, %r885;
	mov.b32 	%r941, %r885;
	mov.b32 	%r942, %r885;
	mov.b32 	%r943, %r885;
	mov.b32 	%r944, %r885;
	mov.b32 	%r945, %r885;
	mov.b32 	%r946, %r885;
	mov.b32 	%r947, %r885;
	mov.b32 	%r948, %r885;
$L__BB0_4:                              // %._crit_edge
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r506, %r8, %r11;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r507, %r506, %r22;
	.loc	1 323 45                        // sk06_mlp_down.py:323:45
	or.b32 	%r508, %r8, %r880;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r509, %r508, 15;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r510, %r509, %r22;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r511, %r508, 14;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r512, %r511, %r22;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r513, %r508, 13;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r514, %r513, %r22;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r515, %r508, 12;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r516, %r515, %r22;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r517, %r508, 11;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r518, %r517, %r22;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r519, %r508, 10;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r520, %r519, %r22;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r521, %r508, 9;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r522, %r521, %r22;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r523, %r508, 8;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r524, %r523, %r22;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r525, %r508, 7;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r526, %r525, %r22;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r527, %r508, 6;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r528, %r527, %r22;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r529, %r508, 5;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r530, %r529, %r22;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r531, %r508, 4;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r532, %r531, %r22;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r533, %r508, 3;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r534, %r533, %r22;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r535, %r508, 2;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r536, %r535, %r22;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r537, %r508, 1;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r538, %r537, %r22;
	rem.s32 	%r539, %r508, %r22;
	.loc	1 323 45                        // sk06_mlp_down.py:323:45
	shl.b32 	%r540, %r10, 3;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r541, %r8, %r540;
	.loc	1 322 45                        // sk06_mlp_down.py:322:45
	shr.u32 	%r542, %r2, 2;
	bfe.u32 	%r543, %r2, 2, 3;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r544, %r543, %r1;
	or.b32 	%r545, %r544, 56;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r546, %r545, %r21;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r547, %r544, 48;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r548, %r547, %r21;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r549, %r544, 40;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r550, %r549, %r21;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r551, %r544, 32;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r552, %r551, %r21;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r553, %r544, 24;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r554, %r553, %r21;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r555, %r544, 16;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r556, %r555, %r21;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r557, %r544, 8;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r558, %r557, %r21;
	rem.s32 	%r559, %r544, %r21;
	.loc	1 322 45                        // sk06_mlp_down.py:322:45
	shr.u32 	%r560, %r2, 4;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r561, %r560, %r1;
	or.b32 	%r562, %r561, 56;
	.loc	1 322 45                        // sk06_mlp_down.py:322:45
	bfe.u32 	%r563, %r2, 4, 3;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r564, %r563, %r1;
	or.b32 	%r565, %r564, 48;
	or.b32 	%r566, %r564, 40;
	or.b32 	%r567, %r564, 32;
	or.b32 	%r568, %r564, 24;
	or.b32 	%r569, %r564, 16;
	or.b32 	%r570, %r564, 8;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 347 38                        // sk06_mlp_down.py:347:38
	mad.wide.s32 	%rd85, %r559, 4, %rd18;
	mad.wide.s32 	%rd86, %r558, 4, %rd18;
	mad.wide.s32 	%rd87, %r556, 4, %rd18;
	mad.wide.s32 	%rd88, %r554, 4, %rd18;
	mad.wide.s32 	%rd89, %r552, 4, %rd18;
	mad.wide.s32 	%rd90, %r550, 4, %rd18;
	mad.wide.s32 	%rd91, %r548, 4, %rd18;
	mad.wide.s32 	%rd92, %r546, 4, %rd18;
	.loc	1 347 24                        // sk06_mlp_down.py:347:24
	// begin inline asm
	mov.u32 %r394, 0x0;
	ld.global.b32 { %r394 }, [ %rd85 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r395, 0x0;
	ld.global.b32 { %r395 }, [ %rd86 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r396, 0x0;
	ld.global.b32 { %r396 }, [ %rd87 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r397, 0x0;
	ld.global.b32 { %r397 }, [ %rd88 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r398, 0x0;
	ld.global.b32 { %r398 }, [ %rd89 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r399, 0x0;
	ld.global.b32 { %r399 }, [ %rd90 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r400, 0x0;
	ld.global.b32 { %r400 }, [ %rd91 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r401, 0x0;
	ld.global.b32 { %r401 }, [ %rd92 + 0 ];
	// end inline asm
	.loc	1 349 42                        // sk06_mlp_down.py:349:42
	mad.wide.s32 	%rd93, %r507, 4, %rd19;
	.loc	1 349 28                        // sk06_mlp_down.py:349:28
	// begin inline asm
	mov.u32 %r403, 0x0;
	ld.global.b32 { %r403 }, [ %rd93 + 0 ];
	// end inline asm
	.loc	1 349 20                        // sk06_mlp_down.py:349:20
	shl.b32 	%r571, %r11, 2;
	add.s32 	%r402, %r133, %r571;
	// begin inline asm
	st.shared.b32 [ %r402 + 0 ], %r403;
	// end inline asm
	bar.sync 	0;
	and.b32 	%r572, %r2, 3;
	shl.b32 	%r573, %r572, 3;
	and.b32 	%r574, %r2, 96;
	add.s32 	%r575, %r133, %r573;
	add.s32 	%r576, %r575, %r574;
	.loc	1 350 49                        // sk06_mlp_down.py:350:49
	mul.lo.s32 	%r577, %r4, %r24;
	mul.lo.s32 	%r578, %r5, %r24;
	mul.lo.s32 	%r579, %r6, %r24;
	mul.lo.s32 	%r580, %r7, %r24;
	.loc	1 350 31                        // sk06_mlp_down.py:350:31
	mad.wide.s32 	%rd166, %r577, 2, %rd17;
	mad.wide.s32 	%rd167, %r578, 2, %rd17;
	mad.wide.s32 	%rd168, %r579, 2, %rd17;
	mad.wide.s32 	%rd169, %r580, 2, %rd17;
	.loc	1 350 82                        // sk06_mlp_down.py:350:82
	mul.lo.s32 	%r581, %r539, %r25;
	mul.lo.s32 	%r582, %r538, %r25;
	mul.lo.s32 	%r583, %r536, %r25;
	mul.lo.s32 	%r584, %r534, %r25;
	mul.lo.s32 	%r585, %r532, %r25;
	mul.lo.s32 	%r586, %r530, %r25;
	mul.lo.s32 	%r587, %r528, %r25;
	mul.lo.s32 	%r588, %r526, %r25;
	mul.lo.s32 	%r589, %r524, %r25;
	mul.lo.s32 	%r590, %r522, %r25;
	mul.lo.s32 	%r591, %r520, %r25;
	mul.lo.s32 	%r592, %r518, %r25;
	mul.lo.s32 	%r593, %r516, %r25;
	mul.lo.s32 	%r594, %r514, %r25;
	mul.lo.s32 	%r595, %r512, %r25;
	mul.lo.s32 	%r596, %r510, %r25;
	.loc	1 350 64                        // sk06_mlp_down.py:350:64
	mul.wide.s32 	%rd170, %r581, 2;
	add.s64 	%rd94, %rd166, %rd170;
	mul.wide.s32 	%rd171, %r582, 2;
	add.s64 	%rd95, %rd166, %rd171;
	mul.wide.s32 	%rd172, %r583, 2;
	add.s64 	%rd96, %rd166, %rd172;
	mul.wide.s32 	%rd173, %r584, 2;
	add.s64 	%rd97, %rd166, %rd173;
	mul.wide.s32 	%rd174, %r585, 2;
	add.s64 	%rd98, %rd166, %rd174;
	mul.wide.s32 	%rd175, %r586, 2;
	add.s64 	%rd99, %rd166, %rd175;
	mul.wide.s32 	%rd176, %r587, 2;
	add.s64 	%rd100, %rd166, %rd176;
	mul.wide.s32 	%rd177, %r588, 2;
	add.s64 	%rd101, %rd166, %rd177;
	mul.wide.s32 	%rd178, %r589, 2;
	add.s64 	%rd102, %rd166, %rd178;
	mul.wide.s32 	%rd179, %r590, 2;
	add.s64 	%rd103, %rd166, %rd179;
	mul.wide.s32 	%rd180, %r591, 2;
	add.s64 	%rd104, %rd166, %rd180;
	mul.wide.s32 	%rd181, %r592, 2;
	add.s64 	%rd105, %rd166, %rd181;
	mul.wide.s32 	%rd182, %r593, 2;
	add.s64 	%rd106, %rd166, %rd182;
	mul.wide.s32 	%rd183, %r594, 2;
	add.s64 	%rd107, %rd166, %rd183;
	mul.wide.s32 	%rd184, %r595, 2;
	add.s64 	%rd108, %rd166, %rd184;
	mul.wide.s32 	%rd185, %r596, 2;
	add.s64 	%rd109, %rd166, %rd185;
	add.s64 	%rd110, %rd167, %rd170;
	add.s64 	%rd111, %rd167, %rd171;
	add.s64 	%rd112, %rd167, %rd172;
	add.s64 	%rd113, %rd167, %rd173;
	add.s64 	%rd114, %rd167, %rd174;
	add.s64 	%rd115, %rd167, %rd175;
	add.s64 	%rd116, %rd167, %rd176;
	add.s64 	%rd117, %rd167, %rd177;
	add.s64 	%rd118, %rd167, %rd178;
	add.s64 	%rd119, %rd167, %rd179;
	add.s64 	%rd120, %rd167, %rd180;
	add.s64 	%rd121, %rd167, %rd181;
	add.s64 	%rd122, %rd167, %rd182;
	add.s64 	%rd123, %rd167, %rd183;
	add.s64 	%rd124, %rd167, %rd184;
	add.s64 	%rd125, %rd167, %rd185;
	add.s64 	%rd126, %rd168, %rd170;
	add.s64 	%rd127, %rd168, %rd171;
	add.s64 	%rd128, %rd168, %rd172;
	add.s64 	%rd129, %rd168, %rd173;
	add.s64 	%rd130, %rd168, %rd174;
	add.s64 	%rd131, %rd168, %rd175;
	add.s64 	%rd132, %rd168, %rd176;
	add.s64 	%rd133, %rd168, %rd177;
	add.s64 	%rd134, %rd168, %rd178;
	add.s64 	%rd135, %rd168, %rd179;
	add.s64 	%rd136, %rd168, %rd180;
	add.s64 	%rd137, %rd168, %rd181;
	add.s64 	%rd138, %rd168, %rd182;
	add.s64 	%rd139, %rd168, %rd183;
	add.s64 	%rd140, %rd168, %rd184;
	add.s64 	%rd141, %rd168, %rd185;
	add.s64 	%rd142, %rd169, %rd170;
	add.s64 	%rd143, %rd169, %rd171;
	add.s64 	%rd144, %rd169, %rd172;
	add.s64 	%rd145, %rd169, %rd173;
	add.s64 	%rd146, %rd169, %rd174;
	add.s64 	%rd147, %rd169, %rd175;
	add.s64 	%rd148, %rd169, %rd176;
	add.s64 	%rd149, %rd169, %rd177;
	add.s64 	%rd150, %rd169, %rd178;
	add.s64 	%rd151, %rd169, %rd179;
	add.s64 	%rd152, %rd169, %rd180;
	add.s64 	%rd153, %rd169, %rd181;
	add.s64 	%rd154, %rd169, %rd182;
	add.s64 	%rd155, %rd169, %rd183;
	add.s64 	%rd156, %rd169, %rd184;
	add.s64 	%rd157, %rd169, %rd185;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	shl.b32 	%r597, %r13, 6;
	shl.b32 	%r598, %r3, 1;
	or.b32 	%r599, %r597, %r880;
	xor.b32 	%r600, %r599, %r598;
	add.s32 	%r404, %r133, %r600;
	add.s32 	%r409, %r404, 256;
	shl.b32 	%r601, %r9, 9;
	shl.b32 	%r602, %r10, 4;
	setp.eq.b32 	%p16, %r884, 0;
	shl.b32 	%r603, %r884, 1;
	shl.b32 	%r604, %r2, 3;
	and.b32 	%r605, %r604, 256;
	and.b32 	%r606, %r542, 16;
	or.b32 	%r607, %r601, %r605;
	xor.b32 	%r608, %r602, %r603;
	xor.b32 	%r609, %r608, %r606;
	or.b32 	%r610, %r609, %r607;
	add.s32 	%r611, %r133, %r610;
	xor.b32 	%r612, %r610, 64;
	add.s32 	%r613, %r133, %r612;
	.loc	1 357 31                        // sk06_mlp_down.py:357:31
	setp.lt.s32 	%p17, %r564, %r21;
	setp.lt.s32 	%p18, %r570, %r21;
	setp.lt.s32 	%p19, %r569, %r21;
	setp.lt.s32 	%p20, %r568, %r21;
	setp.lt.s32 	%p21, %r567, %r21;
	setp.lt.s32 	%p22, %r566, %r21;
	setp.lt.s32 	%p23, %r565, %r21;
	setp.lt.s32 	%p24, %r562, %r21;
	.loc	1 357 54                        // sk06_mlp_down.py:357:54
	setp.lt.s32 	%p25, %r541, %r22;
	.loc	1 357 37                        // sk06_mlp_down.py:357:37
	and.pred 	%p8, %p17, %p25;
	and.pred 	%p9, %p18, %p25;
	and.pred 	%p10, %p19, %p25;
	and.pred 	%p11, %p20, %p25;
	and.pred 	%p12, %p21, %p25;
	and.pred 	%p13, %p22, %p25;
	and.pred 	%p14, %p23, %p25;
	and.pred 	%p15, %p24, %p25;
	.loc	1 355 35                        // sk06_mlp_down.py:355:35
	mul.lo.s32 	%r614, %r564, %r23;
	mul.lo.s32 	%r615, %r570, %r23;
	mul.lo.s32 	%r616, %r569, %r23;
	mul.lo.s32 	%r617, %r568, %r23;
	mul.lo.s32 	%r618, %r567, %r23;
	mul.lo.s32 	%r619, %r566, %r23;
	mul.lo.s32 	%r620, %r565, %r23;
	mul.lo.s32 	%r621, %r562, %r23;
	.loc	1 355 18                        // sk06_mlp_down.py:355:18
	mad.wide.s32 	%rd186, %r614, 2, %rd16;
	mad.wide.s32 	%rd187, %r615, 2, %rd16;
	mad.wide.s32 	%rd188, %r616, 2, %rd16;
	mad.wide.s32 	%rd189, %r617, 2, %rd16;
	mad.wide.s32 	%rd190, %r618, 2, %rd16;
	mad.wide.s32 	%rd191, %r619, 2, %rd16;
	mad.wide.s32 	%rd192, %r620, 2, %rd16;
	mad.wide.s32 	%rd193, %r621, 2, %rd16;
	.loc	1 355 50                        // sk06_mlp_down.py:355:50
	mul.wide.s32 	%rd194, %r541, 2;
	add.s64 	%rd158, %rd186, %rd194;
	add.s64 	%rd159, %rd187, %rd194;
	add.s64 	%rd160, %rd188, %rd194;
	add.s64 	%rd161, %rd189, %rd194;
	add.s64 	%rd162, %rd190, %rd194;
	add.s64 	%rd163, %rd191, %rd194;
	add.s64 	%rd164, %rd192, %rd194;
	add.s64 	%rd165, %rd193, %rd194;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r622, %r904, %r397;
	mul.f32 	%r623, %r903, %r397;
	.loc	1 349 20                        // sk06_mlp_down.py:349:20
	ld.shared.v2.b32 	{%r624, %r625}, [%r576];
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r626, %r886, %r394;
	mul.f32 	%r627, %r885, %r394;
	mul.f32 	%r628, %r888, %r395;
	mul.f32 	%r629, %r887, %r395;
	mul.f32 	%r630, %r902, %r396;
	mul.f32 	%r631, %r901, %r396;
	mul.f32 	%r632, %r912, %r397;
	mul.f32 	%r633, %r911, %r397;
	.loc	1 349 20                        // sk06_mlp_down.py:349:20
	ld.shared.v2.b32 	{%r634, %r635}, [%r576+256];
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r636, %r894, %r394;
	mul.f32 	%r637, %r893, %r394;
	mul.f32 	%r638, %r896, %r395;
	mul.f32 	%r639, %r895, %r395;
	mul.f32 	%r640, %r908, %r397;
	mul.f32 	%r641, %r907, %r397;
	.loc	1 349 20                        // sk06_mlp_down.py:349:20
	ld.shared.v2.b32 	{%r642, %r643}, [%r576+128];
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r644, %r890, %r394;
	mul.f32 	%r645, %r889, %r394;
	mul.f32 	%r646, %r892, %r395;
	mul.f32 	%r647, %r891, %r395;
	mul.f32 	%r648, %r906, %r396;
	mul.f32 	%r649, %r905, %r396;
	mul.f32 	%r650, %r910, %r396;
	mul.f32 	%r651, %r909, %r396;
	mul.f32 	%r652, %r916, %r397;
	mul.f32 	%r653, %r915, %r397;
	.loc	1 349 20                        // sk06_mlp_down.py:349:20
	ld.shared.v2.b32 	{%r654, %r655}, [%r576+384];
	.loc	1 350 19                        // sk06_mlp_down.py:350:19
	// begin inline asm
	mov.u16 %rs1, 0x0;
	ld.global.b16 { %rs1 }, [ %rd94 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs2, 0x0;
	ld.global.b16 { %rs2 }, [ %rd95 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs3, 0x0;
	ld.global.b16 { %rs3 }, [ %rd96 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs4, 0x0;
	ld.global.b16 { %rs4 }, [ %rd97 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs5, 0x0;
	ld.global.b16 { %rs5 }, [ %rd98 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs6, 0x0;
	ld.global.b16 { %rs6 }, [ %rd99 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs7, 0x0;
	ld.global.b16 { %rs7 }, [ %rd100 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs8, 0x0;
	ld.global.b16 { %rs8 }, [ %rd101 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs9, 0x0;
	ld.global.b16 { %rs9 }, [ %rd102 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs10, 0x0;
	ld.global.b16 { %rs10 }, [ %rd103 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs11, 0x0;
	ld.global.b16 { %rs11 }, [ %rd104 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs12, 0x0;
	ld.global.b16 { %rs12 }, [ %rd105 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs13, 0x0;
	ld.global.b16 { %rs13 }, [ %rd106 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs14, 0x0;
	ld.global.b16 { %rs14 }, [ %rd107 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs15, 0x0;
	ld.global.b16 { %rs15 }, [ %rd108 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs16, 0x0;
	ld.global.b16 { %rs16 }, [ %rd109 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs17, 0x0;
	ld.global.b16 { %rs17 }, [ %rd110 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs18, 0x0;
	ld.global.b16 { %rs18 }, [ %rd111 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs19, 0x0;
	ld.global.b16 { %rs19 }, [ %rd112 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs20, 0x0;
	ld.global.b16 { %rs20 }, [ %rd113 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs21, 0x0;
	ld.global.b16 { %rs21 }, [ %rd114 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs22, 0x0;
	ld.global.b16 { %rs22 }, [ %rd115 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs23, 0x0;
	ld.global.b16 { %rs23 }, [ %rd116 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs24, 0x0;
	ld.global.b16 { %rs24 }, [ %rd117 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs25, 0x0;
	ld.global.b16 { %rs25 }, [ %rd118 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs26, 0x0;
	ld.global.b16 { %rs26 }, [ %rd119 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs27, 0x0;
	ld.global.b16 { %rs27 }, [ %rd120 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs28, 0x0;
	ld.global.b16 { %rs28 }, [ %rd121 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs29, 0x0;
	ld.global.b16 { %rs29 }, [ %rd122 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs30, 0x0;
	ld.global.b16 { %rs30 }, [ %rd123 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs31, 0x0;
	ld.global.b16 { %rs31 }, [ %rd124 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs32, 0x0;
	ld.global.b16 { %rs32 }, [ %rd125 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs33, 0x0;
	ld.global.b16 { %rs33 }, [ %rd126 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs34, 0x0;
	ld.global.b16 { %rs34 }, [ %rd127 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs35, 0x0;
	ld.global.b16 { %rs35 }, [ %rd128 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs36, 0x0;
	ld.global.b16 { %rs36 }, [ %rd129 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs37, 0x0;
	ld.global.b16 { %rs37 }, [ %rd130 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs38, 0x0;
	ld.global.b16 { %rs38 }, [ %rd131 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs39, 0x0;
	ld.global.b16 { %rs39 }, [ %rd132 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs40, 0x0;
	ld.global.b16 { %rs40 }, [ %rd133 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs41, 0x0;
	ld.global.b16 { %rs41 }, [ %rd134 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs42, 0x0;
	ld.global.b16 { %rs42 }, [ %rd135 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs43, 0x0;
	ld.global.b16 { %rs43 }, [ %rd136 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs44, 0x0;
	ld.global.b16 { %rs44 }, [ %rd137 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs45, 0x0;
	ld.global.b16 { %rs45 }, [ %rd138 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs46, 0x0;
	ld.global.b16 { %rs46 }, [ %rd139 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs47, 0x0;
	ld.global.b16 { %rs47 }, [ %rd140 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs48, 0x0;
	ld.global.b16 { %rs48 }, [ %rd141 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs49, 0x0;
	ld.global.b16 { %rs49 }, [ %rd142 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs50, 0x0;
	ld.global.b16 { %rs50 }, [ %rd143 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs51, 0x0;
	ld.global.b16 { %rs51 }, [ %rd144 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs52, 0x0;
	ld.global.b16 { %rs52 }, [ %rd145 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs53, 0x0;
	ld.global.b16 { %rs53 }, [ %rd146 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs54, 0x0;
	ld.global.b16 { %rs54 }, [ %rd147 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs55, 0x0;
	ld.global.b16 { %rs55 }, [ %rd148 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs56, 0x0;
	ld.global.b16 { %rs56 }, [ %rd149 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs57, 0x0;
	ld.global.b16 { %rs57 }, [ %rd150 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs58, 0x0;
	ld.global.b16 { %rs58 }, [ %rd151 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs59, 0x0;
	ld.global.b16 { %rs59 }, [ %rd152 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs60, 0x0;
	ld.global.b16 { %rs60 }, [ %rd153 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs61, 0x0;
	ld.global.b16 { %rs61 }, [ %rd154 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs62, 0x0;
	ld.global.b16 { %rs62 }, [ %rd155 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs63, 0x0;
	ld.global.b16 { %rs63 }, [ %rd156 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs64, 0x0;
	ld.global.b16 { %rs64 }, [ %rd157 + 0 ];
	// end inline asm
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	bar.sync 	0;
	mov.b32 	%r405, {%rs1, %rs2};
	mov.b32 	%r406, {%rs3, %rs4};
	mov.b32 	%r407, {%rs5, %rs6};
	mov.b32 	%r408, {%rs7, %rs8};
	// begin inline asm
	st.shared.v4.b32 [ %r404 + 0 ], { %r405, %r406, %r407, %r408 };
	// end inline asm
	mov.b32 	%r410, {%rs9, %rs10};
	mov.b32 	%r411, {%rs11, %rs12};
	mov.b32 	%r412, {%rs13, %rs14};
	mov.b32 	%r413, {%rs15, %rs16};
	// begin inline asm
	st.shared.v4.b32 [ %r409 + 0 ], { %r410, %r411, %r412, %r413 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r656, %r657, %r658, %r659}, [%r611];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r660, %r661, %r662, %r663}, [%r613];
	bar.sync 	0;
	mov.b32 	%r414, {%rs17, %rs18};
	mov.b32 	%r415, {%rs19, %rs20};
	mov.b32 	%r416, {%rs21, %rs22};
	mov.b32 	%r417, {%rs23, %rs24};
	// begin inline asm
	st.shared.v4.b32 [ %r404 + 0 ], { %r414, %r415, %r416, %r417 };
	// end inline asm
	mov.b32 	%r418, {%rs25, %rs26};
	mov.b32 	%r419, {%rs27, %rs28};
	mov.b32 	%r420, {%rs29, %rs30};
	mov.b32 	%r421, {%rs31, %rs32};
	// begin inline asm
	st.shared.v4.b32 [ %r409 + 0 ], { %r418, %r419, %r420, %r421 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r664, %r665, %r666, %r667}, [%r611];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r668, %r669, %r670, %r671}, [%r613];
	bar.sync 	0;
	mov.b32 	%r422, {%rs33, %rs34};
	mov.b32 	%r423, {%rs35, %rs36};
	mov.b32 	%r424, {%rs37, %rs38};
	mov.b32 	%r425, {%rs39, %rs40};
	// begin inline asm
	st.shared.v4.b32 [ %r404 + 0 ], { %r422, %r423, %r424, %r425 };
	// end inline asm
	mov.b32 	%r426, {%rs41, %rs42};
	mov.b32 	%r427, {%rs43, %rs44};
	mov.b32 	%r428, {%rs45, %rs46};
	mov.b32 	%r429, {%rs47, %rs48};
	// begin inline asm
	st.shared.v4.b32 [ %r409 + 0 ], { %r426, %r427, %r428, %r429 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r672, %r673, %r674, %r675}, [%r611];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r676, %r677, %r678, %r679}, [%r613];
	bar.sync 	0;
	mov.b32 	%r430, {%rs49, %rs50};
	mov.b32 	%r431, {%rs51, %rs52};
	mov.b32 	%r432, {%rs53, %rs54};
	mov.b32 	%r433, {%rs55, %rs56};
	// begin inline asm
	st.shared.v4.b32 [ %r404 + 0 ], { %r430, %r431, %r432, %r433 };
	// end inline asm
	mov.b32 	%r434, {%rs57, %rs58};
	mov.b32 	%r435, {%rs59, %rs60};
	mov.b32 	%r436, {%rs61, %rs62};
	mov.b32 	%r437, {%rs63, %rs64};
	// begin inline asm
	st.shared.v4.b32 [ %r409 + 0 ], { %r434, %r435, %r436, %r437 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r680, %r681, %r682, %r683}, [%r611];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r684, %r685, %r686, %r687}, [%r613];
	mov.b32 	{%rs65, %rs66}, %r665;
	cvt.f32.bf16 	%r688, %rs66;
	cvt.f32.bf16 	%r689, %rs65;
	mov.b32 	{%rs67, %rs68}, %r667;
	cvt.f32.bf16 	%r690, %rs68;
	cvt.f32.bf16 	%r691, %rs67;
	mov.b32 	{%rs69, %rs70}, %r669;
	cvt.f32.bf16 	%r692, %rs70;
	cvt.f32.bf16 	%r693, %rs69;
	mov.b32 	{%rs71, %rs72}, %r671;
	cvt.f32.bf16 	%r694, %rs72;
	cvt.f32.bf16 	%r695, %rs71;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r696, %r623, %r624, %r689;
	fma.rn.f32 	%r697, %r622, %r625, %r688;
	fma.rn.f32 	%r698, %r641, %r642, %r691;
	fma.rn.f32 	%r699, %r640, %r643, %r690;
	fma.rn.f32 	%r700, %r633, %r634, %r693;
	fma.rn.f32 	%r701, %r632, %r635, %r692;
	fma.rn.f32 	%r702, %r653, %r654, %r695;
	fma.rn.f32 	%r703, %r652, %r655, %r694;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs73, %rs74}, %r656;
	cvt.f32.bf16 	%r704, %rs74;
	cvt.f32.bf16 	%r705, %rs73;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r706, %r627, %r624, %r705;
	fma.rn.f32 	%r707, %r626, %r625, %r704;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r439, %r707, %r706;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs75, %rs76}, %r657;
	cvt.f32.bf16 	%r708, %rs76;
	cvt.f32.bf16 	%r709, %rs75;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r710, %r629, %r624, %r709;
	fma.rn.f32 	%r711, %r628, %r625, %r708;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r440, %r711, %r710;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs77, %rs78}, %r658;
	cvt.f32.bf16 	%r712, %rs78;
	cvt.f32.bf16 	%r713, %rs77;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r714, %r645, %r642, %r713;
	fma.rn.f32 	%r715, %r644, %r643, %r712;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r449, %r715, %r714;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs79, %rs80}, %r659;
	cvt.f32.bf16 	%r716, %rs80;
	cvt.f32.bf16 	%r717, %rs79;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r718, %r647, %r642, %r717;
	fma.rn.f32 	%r719, %r646, %r643, %r716;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r450, %r719, %r718;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs81, %rs82}, %r660;
	cvt.f32.bf16 	%r720, %rs82;
	cvt.f32.bf16 	%r721, %rs81;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r722, %r637, %r634, %r721;
	fma.rn.f32 	%r723, %r636, %r635, %r720;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r444, %r723, %r722;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs83, %rs84}, %r661;
	cvt.f32.bf16 	%r724, %rs84;
	cvt.f32.bf16 	%r725, %rs83;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r726, %r639, %r634, %r725;
	fma.rn.f32 	%r727, %r638, %r635, %r724;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r445, %r727, %r726;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r728, %r898, %r394;
	mul.f32 	%r729, %r897, %r394;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs85, %rs86}, %r662;
	cvt.f32.bf16 	%r730, %rs86;
	cvt.f32.bf16 	%r731, %rs85;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r732, %r729, %r654, %r731;
	fma.rn.f32 	%r733, %r728, %r655, %r730;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r454, %r733, %r732;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r734, %r900, %r395;
	mul.f32 	%r735, %r899, %r395;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs87, %rs88}, %r663;
	cvt.f32.bf16 	%r736, %rs88;
	cvt.f32.bf16 	%r737, %rs87;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r738, %r735, %r654, %r737;
	fma.rn.f32 	%r739, %r734, %r655, %r736;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r455, %r739, %r738;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs89, %rs90}, %r664;
	cvt.f32.bf16 	%r740, %rs90;
	cvt.f32.bf16 	%r741, %rs89;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r742, %r631, %r624, %r741;
	fma.rn.f32 	%r743, %r630, %r625, %r740;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r441, %r743, %r742;
	cvt.rn.bf16x2.f32 	%r442, %r697, %r696;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs91, %rs92}, %r666;
	cvt.f32.bf16 	%r744, %rs92;
	cvt.f32.bf16 	%r745, %rs91;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r746, %r649, %r642, %r745;
	fma.rn.f32 	%r747, %r648, %r643, %r744;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r451, %r747, %r746;
	cvt.rn.bf16x2.f32 	%r452, %r699, %r698;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs93, %rs94}, %r668;
	cvt.f32.bf16 	%r748, %rs94;
	cvt.f32.bf16 	%r749, %rs93;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r750, %r651, %r634, %r749;
	fma.rn.f32 	%r751, %r650, %r635, %r748;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r446, %r751, %r750;
	cvt.rn.bf16x2.f32 	%r447, %r701, %r700;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r752, %r914, %r396;
	mul.f32 	%r753, %r913, %r396;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs95, %rs96}, %r670;
	cvt.f32.bf16 	%r754, %rs96;
	cvt.f32.bf16 	%r755, %rs95;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r756, %r753, %r654, %r755;
	fma.rn.f32 	%r757, %r752, %r655, %r754;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r456, %r757, %r756;
	cvt.rn.bf16x2.f32 	%r457, %r703, %r702;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r758, %r918, %r398;
	mul.f32 	%r759, %r917, %r398;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs97, %rs98}, %r672;
	cvt.f32.bf16 	%r760, %rs98;
	cvt.f32.bf16 	%r761, %rs97;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r762, %r759, %r624, %r761;
	fma.rn.f32 	%r763, %r758, %r625, %r760;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r458, %r763, %r762;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r764, %r920, %r399;
	mul.f32 	%r765, %r919, %r399;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs99, %rs100}, %r673;
	cvt.f32.bf16 	%r766, %rs100;
	cvt.f32.bf16 	%r767, %rs99;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r768, %r765, %r624, %r767;
	fma.rn.f32 	%r769, %r764, %r625, %r766;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r459, %r769, %r768;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r770, %r922, %r398;
	mul.f32 	%r771, %r921, %r398;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs101, %rs102}, %r674;
	cvt.f32.bf16 	%r772, %rs102;
	cvt.f32.bf16 	%r773, %rs101;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r774, %r771, %r642, %r773;
	fma.rn.f32 	%r775, %r770, %r643, %r772;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r466, %r775, %r774;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r776, %r924, %r399;
	mul.f32 	%r777, %r923, %r399;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs103, %rs104}, %r675;
	cvt.f32.bf16 	%r778, %rs104;
	cvt.f32.bf16 	%r779, %rs103;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r780, %r777, %r642, %r779;
	fma.rn.f32 	%r781, %r776, %r643, %r778;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r467, %r781, %r780;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r782, %r926, %r398;
	mul.f32 	%r783, %r925, %r398;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs105, %rs106}, %r676;
	cvt.f32.bf16 	%r784, %rs106;
	cvt.f32.bf16 	%r785, %rs105;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r786, %r783, %r634, %r785;
	fma.rn.f32 	%r787, %r782, %r635, %r784;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r462, %r787, %r786;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r788, %r928, %r399;
	mul.f32 	%r789, %r927, %r399;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs107, %rs108}, %r677;
	cvt.f32.bf16 	%r790, %rs108;
	cvt.f32.bf16 	%r791, %rs107;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r792, %r789, %r634, %r791;
	fma.rn.f32 	%r793, %r788, %r635, %r790;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r463, %r793, %r792;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r794, %r930, %r398;
	mul.f32 	%r795, %r929, %r398;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs109, %rs110}, %r678;
	cvt.f32.bf16 	%r796, %rs110;
	cvt.f32.bf16 	%r797, %rs109;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r798, %r795, %r654, %r797;
	fma.rn.f32 	%r799, %r794, %r655, %r796;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r470, %r799, %r798;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r800, %r932, %r399;
	mul.f32 	%r801, %r931, %r399;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs111, %rs112}, %r679;
	cvt.f32.bf16 	%r802, %rs112;
	cvt.f32.bf16 	%r803, %rs111;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r804, %r801, %r654, %r803;
	fma.rn.f32 	%r805, %r800, %r655, %r802;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r471, %r805, %r804;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r806, %r934, %r400;
	mul.f32 	%r807, %r933, %r400;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs113, %rs114}, %r680;
	cvt.f32.bf16 	%r808, %rs114;
	cvt.f32.bf16 	%r809, %rs113;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r810, %r807, %r624, %r809;
	fma.rn.f32 	%r811, %r806, %r625, %r808;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r460, %r811, %r810;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r812, %r936, %r401;
	mul.f32 	%r813, %r935, %r401;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs115, %rs116}, %r681;
	cvt.f32.bf16 	%r814, %rs116;
	cvt.f32.bf16 	%r815, %rs115;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r816, %r813, %r624, %r815;
	fma.rn.f32 	%r817, %r812, %r625, %r814;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r461, %r817, %r816;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r818, %r938, %r400;
	mul.f32 	%r819, %r937, %r400;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs117, %rs118}, %r682;
	cvt.f32.bf16 	%r820, %rs118;
	cvt.f32.bf16 	%r821, %rs117;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r822, %r819, %r642, %r821;
	fma.rn.f32 	%r823, %r818, %r643, %r820;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r468, %r823, %r822;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r824, %r940, %r401;
	mul.f32 	%r825, %r939, %r401;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs119, %rs120}, %r683;
	cvt.f32.bf16 	%r826, %rs120;
	cvt.f32.bf16 	%r827, %rs119;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r828, %r825, %r642, %r827;
	fma.rn.f32 	%r829, %r824, %r643, %r826;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r469, %r829, %r828;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r830, %r942, %r400;
	mul.f32 	%r831, %r941, %r400;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs121, %rs122}, %r684;
	cvt.f32.bf16 	%r832, %rs122;
	cvt.f32.bf16 	%r833, %rs121;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r834, %r831, %r634, %r833;
	fma.rn.f32 	%r835, %r830, %r635, %r832;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r464, %r835, %r834;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r836, %r944, %r401;
	mul.f32 	%r837, %r943, %r401;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs123, %rs124}, %r685;
	cvt.f32.bf16 	%r838, %rs124;
	cvt.f32.bf16 	%r839, %rs123;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r840, %r837, %r634, %r839;
	fma.rn.f32 	%r841, %r836, %r635, %r838;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r465, %r841, %r840;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r842, %r946, %r400;
	mul.f32 	%r843, %r945, %r400;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs125, %rs126}, %r686;
	cvt.f32.bf16 	%r844, %rs126;
	cvt.f32.bf16 	%r845, %rs125;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r846, %r843, %r654, %r845;
	fma.rn.f32 	%r847, %r842, %r655, %r844;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r472, %r847, %r846;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r848, %r948, %r401;
	mul.f32 	%r849, %r947, %r401;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs127, %rs128}, %r687;
	cvt.f32.bf16 	%r850, %rs128;
	cvt.f32.bf16 	%r851, %rs127;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r852, %r849, %r654, %r851;
	fma.rn.f32 	%r853, %r848, %r655, %r850;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r473, %r853, %r852;
	bar.sync 	0;
	shl.b32 	%r854, %r572, 11;
	shl.b32 	%r855, %r572, 5;
	shl.b32 	%r856, %r2, 4;
	and.b32 	%r857, %r856, 384;
	shr.u32 	%r858, %r574, 1;
	bfe.s32 	%r859, %r2, 2, 1;
	and.b32 	%r860, %r859, 1040;
	or.b32 	%r861, %r855, %r857;
	xor.b32 	%r862, %r860, %r858;
	xor.b32 	%r863, %r862, %r861;
	or.b32 	%r864, %r863, %r854;
	add.s32 	%r438, %r133, %r864;
	// begin inline asm
	st.shared.v4.b32 [ %r438 + 0 ], { %r439, %r440, %r441, %r442 };
	// end inline asm
	add.s32 	%r443, %r438, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r443 + 0 ], { %r444, %r445, %r446, %r447 };
	// end inline asm
	xor.b32 	%r865, %r864, 64;
	add.s32 	%r448, %r133, %r865;
	// begin inline asm
	st.shared.v4.b32 [ %r448 + 0 ], { %r449, %r450, %r451, %r452 };
	// end inline asm
	add.s32 	%r453, %r448, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r453 + 0 ], { %r454, %r455, %r456, %r457 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r866, %r574, 2;
	shl.b32 	%r867, %r2, 6;
	and.b32 	%r868, %r867, 512;
	selp.b32 	%r869, 0, 1040, %p16;
	or.b32 	%r870, %r880, %r866;
	xor.b32 	%r871, %r870, %r869;
	or.b32 	%r872, %r871, %r868;
	add.s32 	%r873, %r133, %r872;
	ld.shared.v4.b32 	{%r474, %r478, %r482, %r486}, [%r873];
	xor.b32 	%r874, %r872, 32;
	add.s32 	%r875, %r133, %r874;
	ld.shared.v4.b32 	{%r475, %r479, %r483, %r487}, [%r875+2048];
	xor.b32 	%r876, %r872, 64;
	add.s32 	%r877, %r133, %r876;
	ld.shared.v4.b32 	{%r476, %r480, %r484, %r488}, [%r877+4096];
	xor.b32 	%r878, %r872, 96;
	add.s32 	%r879, %r133, %r878;
	ld.shared.v4.b32 	{%r477, %r481, %r485, %r489}, [%r879+6144];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r438 + 0 ], { %r458, %r459, %r460, %r461 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r443 + 0 ], { %r462, %r463, %r464, %r465 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r448 + 0 ], { %r466, %r467, %r468, %r469 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r453 + 0 ], { %r470, %r471, %r472, %r473 };
	// end inline asm
	bar.sync 	0;
	ld.shared.v4.b32 	{%r490, %r494, %r498, %r502}, [%r873];
	ld.shared.v4.b32 	{%r491, %r495, %r499, %r503}, [%r875+2048];
	ld.shared.v4.b32 	{%r492, %r496, %r500, %r504}, [%r877+4096];
	ld.shared.v4.b32 	{%r493, %r497, %r501, %r505}, [%r879+6144];
	.loc	1 356 8                         // sk06_mlp_down.py:356:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd158 + 0 ], { %r474, %r475, %r476, %r477 };
	// end inline asm
	// begin inline asm
	@%p9 st.global.v4.b32 [ %rd159 + 0 ], { %r478, %r479, %r480, %r481 };
	// end inline asm
	// begin inline asm
	@%p10 st.global.v4.b32 [ %rd160 + 0 ], { %r482, %r483, %r484, %r485 };
	// end inline asm
	// begin inline asm
	@%p11 st.global.v4.b32 [ %rd161 + 0 ], { %r486, %r487, %r488, %r489 };
	// end inline asm
	// begin inline asm
	@%p12 st.global.v4.b32 [ %rd162 + 0 ], { %r490, %r491, %r492, %r493 };
	// end inline asm
	// begin inline asm
	@%p13 st.global.v4.b32 [ %rd163 + 0 ], { %r494, %r495, %r496, %r497 };
	// end inline asm
	// begin inline asm
	@%p14 st.global.v4.b32 [ %rd164 + 0 ], { %r498, %r499, %r500, %r501 };
	// end inline asm
	// begin inline asm
	@%p15 st.global.v4.b32 [ %rd165 + 0 ], { %r502, %r503, %r504, %r505 };
	// end inline asm
	.loc	1 354 4                         // sk06_mlp_down.py:354:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/workspace/vllm/_genesis/kernels/sk06_mlp_down.py"
	.file	2 "/usr/local/lib/python3.12/dist-packages/triton/language/standard.py"
	.section	.debug_abbrev
	{
.b8 1                                   // Abbreviation Code
.b8 17                                  // DW_TAG_compile_unit
.b8 1                                   // DW_CHILDREN_yes
.b8 37                                  // DW_AT_producer
.b8 8                                   // DW_FORM_string
.b8 19                                  // DW_AT_language
.b8 5                                   // DW_FORM_data2
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 16                                  // DW_AT_stmt_list
.b8 6                                   // DW_FORM_data4
.b8 27                                  // DW_AT_comp_dir
.b8 8                                   // DW_FORM_string
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 2                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 0                                   // DW_CHILDREN_no
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 32                                  // DW_AT_inline
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 3                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 1                                   // DW_CHILDREN_yes
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 4                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 0                                   // DW_CHILDREN_no
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 5                                   // DW_FORM_data2
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 168                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0xa1 DW_TAG_compile_unit
.b8 116                                 // DW_AT_producer
.b8 114
.b8 105
.b8 116
.b8 111
.b8 110
.b8 0
.b8 2                                   // DW_AT_language
.b8 0
.b8 115                                 // DW_AT_name
.b8 107
.b8 48
.b8 54
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 100
.b8 111
.b8 119
.b8 110
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 119
.b8 111
.b8 114
.b8 107
.b8 115
.b8 112
.b8 97
.b8 99
.b8 101
.b8 47
.b8 118
.b8 108
.b8 108
.b8 109
.b8 47
.b8 95
.b8 103
.b8 101
.b8 110
.b8 101
.b8 115
.b8 105
.b8 115
.b8 47
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 115
.b8 0
.b8 2                                   // Abbrev [2] 0x4b:0x18 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 54
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 100
.b8 111
.b8 119
.b8 110
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x63:0x48 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 75                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x78:0x19 DW_TAG_inlined_subroutine
.b32 75                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 58                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x91:0x19 DW_TAG_inlined_subroutine
.b32 75                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 59                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_5 = _Nativo(
    "sk06_mlp_down/tile64x128x128_shift0_abi16",
    _PTX_5, "_sk06_mlp_down_kernel",
    warps=4, shared=73728,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 17, 18],
    horneado={11: 1, 12: 1, 15: 1, 19: 1, 20: 64, 21: 128, 22: 128, 23: 8, 24: 128, 25: False},
    div16=[8, 9, 10, 13, 14, 16, 17],
)

_PTX_6 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk06_mlp_down_kernel   // -- Begin function _sk06_mlp_down_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk06_mlp_down_kernel
.visible .entry _sk06_mlp_down_kernel(
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_6,
	.param .u32 _sk06_mlp_down_kernel_param_7,
	.param .u32 _sk06_mlp_down_kernel_param_8,
	.param .u32 _sk06_mlp_down_kernel_param_9,
	.param .u32 _sk06_mlp_down_kernel_param_10,
	.param .u32 _sk06_mlp_down_kernel_param_11,
	.param .u32 _sk06_mlp_down_kernel_param_12,
	.param .u32 _sk06_mlp_down_kernel_param_13,
	.param .u32 _sk06_mlp_down_kernel_param_14,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_16
)
.reqntid 128
{
	.reg .pred 	%p<26>;
	.reg .b16 	%rs<65>;
	.reg .b32 	%r<843>;
	.reg .b64 	%rd<154>;
	.loc	1 304 0                         // sk06_mlp_down.py:304:0
$L__func_begin0:
	.loc	1 304 0                         // sk06_mlp_down.py:304:0

// %bb.0:
	ld.param.b32 	%r24, [_sk06_mlp_down_kernel_param_13];
	ld.param.b32 	%r23, [_sk06_mlp_down_kernel_param_12];
	ld.param.b32 	%r22, [_sk06_mlp_down_kernel_param_9];
	ld.param.b32 	%r21, [_sk06_mlp_down_kernel_param_8];
	ld.param.b32 	%r20, [_sk06_mlp_down_kernel_param_7];
	ld.param.b64 	%rd33, [_sk06_mlp_down_kernel_param_4];
	ld.param.b64 	%rd32, [_sk06_mlp_down_kernel_param_3];
	ld.param.b64 	%rd31, [_sk06_mlp_down_kernel_param_2];
	ld.param.b64 	%rd30, [_sk06_mlp_down_kernel_param_1];
	ld.param.b64 	%rd29, [_sk06_mlp_down_kernel_param_0];
$L__tmp0:
	.loc	1 313 24                        // sk06_mlp_down.py:313:24
	mov.u32 	%r65, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk06_mlp_down.py:314:27 ]
	add.s32 	%r66, %r20, 63;
	.loc	2 43 30                         // standard.py:43:30 @[ sk06_mlp_down.py:314:27 ]
	shr.s32 	%r67, %r66, 31;
	shr.u32 	%r68, %r67, 26;
	add.s32 	%r69, %r66, %r68;
	shr.s32 	%r70, %r69, 6;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk06_mlp_down.py:315:27 ]
	add.s32 	%r71, %r21, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk06_mlp_down.py:315:27 ]
	shr.s32 	%r72, %r71, 31;
	shr.u32 	%r73, %r72, 25;
	add.s32 	%r74, %r71, %r73;
	shr.s32 	%r75, %r74, 7;
$L__tmp3:
	.loc	1 316 29                        // sk06_mlp_down.py:316:29
	shl.b32 	%r76, %r75, 3;
	.loc	1 317 22                        // sk06_mlp_down.py:317:22
	div.s32 	%r77, %r65, %r76;
	.loc	1 317 38                        // sk06_mlp_down.py:317:38
	shl.b32 	%r78, %r77, 3;
	.loc	1 318 30                        // sk06_mlp_down.py:318:30
	sub.s32 	%r79, %r70, %r78;
	ld.param.b32 	%r80, [_sk06_mlp_down_kernel_param_10];
	.loc	1 318 39                        // sk06_mlp_down.py:318:39
	min.s32 	%r81, %r79, 8;
	ld.param.b32 	%r82, [_sk06_mlp_down_kernel_param_11];
	.loc	1 319 30                        // sk06_mlp_down.py:319:30
	mul.lo.s32 	%r83, %r77, %r76;
	sub.s32 	%r84, %r65, %r83;
	.loc	1 320 36                        // sk06_mlp_down.py:320:36
	div.s32 	%r85, %r84, %r81;
	.loc	1 319 46                        // sk06_mlp_down.py:319:46
	mul.lo.s32 	%r86, %r85, %r81;
	sub.s32 	%r87, %r84, %r86;
	.loc	1 319 23                        // sk06_mlp_down.py:319:23
	add.s32 	%r88, %r87, %r78;
	.loc	1 322 22                        // sk06_mlp_down.py:322:22
	shl.b32 	%r1, %r88, 6;
	.loc	1 322 45                        // sk06_mlp_down.py:322:45
	mov.u32 	%r2, %tid.x;
	and.b32 	%r3, %r2, 120;
	bfe.u32 	%r89, %r2, 3, 4;
	or.b32 	%r90, %r89, 16;
	or.b32 	%r91, %r89, 32;
	or.b32 	%r92, %r89, 48;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r93, %r1, %r89;
	or.b32 	%r94, %r1, %r90;
	or.b32 	%r95, %r1, %r91;
	or.b32 	%r96, %r1, %r92;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r4, %r93, %r20;
	rem.s32 	%r5, %r94, %r20;
	rem.s32 	%r6, %r95, %r20;
	rem.s32 	%r7, %r96, %r20;
	.loc	1 323 22                        // sk06_mlp_down.py:323:22
	shl.b32 	%r8, %r85, 7;
	.loc	1 323 45                        // sk06_mlp_down.py:323:45
	and.b32 	%r9, %r2, 7;
	shl.b32 	%r97, %r9, 4;
	and.b32 	%r10, %r2, 15;
	and.b32 	%r11, %r2, 127;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r98, %r8, %r89;
	or.b32 	%r99, %r8, %r90;
	or.b32 	%r100, %r8, %r91;
	or.b32 	%r101, %r8, %r92;
	or.b32 	%r102, %r98, 64;
	or.b32 	%r103, %r98, 80;
	or.b32 	%r104, %r98, 96;
	or.b32 	%r105, %r98, 112;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r107, %r98, %r21;
	rem.s32 	%r108, %r99, %r21;
	rem.s32 	%r109, %r100, %r21;
	rem.s32 	%r110, %r101, %r21;
	rem.s32 	%r111, %r102, %r21;
	rem.s32 	%r112, %r103, %r21;
	rem.s32 	%r113, %r104, %r21;
	rem.s32 	%r114, %r105, %r21;
	.loc	1 326 39                        // sk06_mlp_down.py:326:39
	mul.lo.s32 	%r116, %r4, %r80;
	mul.lo.s32 	%r117, %r5, %r80;
	mul.lo.s32 	%r118, %r6, %r80;
	mul.lo.s32 	%r119, %r7, %r80;
	.loc	1 326 21                        // sk06_mlp_down.py:326:21
	cvt.s64.s32 	%rd1, %r116;
	add.s64 	%rd71, %rd29, %rd1;
	cvt.s64.s32 	%rd2, %r117;
	add.s64 	%rd72, %rd29, %rd2;
	cvt.s64.s32 	%rd3, %r118;
	add.s64 	%rd73, %rd29, %rd3;
	cvt.s64.s32 	%rd4, %r119;
	add.s64 	%rd74, %rd29, %rd4;
	.loc	1 326 51                        // sk06_mlp_down.py:326:51
	cvt.u64.u32 	%rd5, %r97;
	add.s64 	%rd34, %rd71, %rd5;
	add.s64 	%rd35, %rd72, %rd5;
	add.s64 	%rd36, %rd73, %rd5;
	add.s64 	%rd37, %rd74, %rd5;
	.loc	1 327 21                        // sk06_mlp_down.py:327:21
	add.s64 	%rd75, %rd30, %rd5;
	.loc	1 327 69                        // sk06_mlp_down.py:327:69
	mul.lo.s32 	%r120, %r107, %r82;
	mul.lo.s32 	%r121, %r108, %r82;
	mul.lo.s32 	%r122, %r109, %r82;
	mul.lo.s32 	%r123, %r110, %r82;
	mul.lo.s32 	%r124, %r111, %r82;
	mul.lo.s32 	%r125, %r112, %r82;
	mul.lo.s32 	%r126, %r113, %r82;
	mul.lo.s32 	%r127, %r114, %r82;
	.loc	1 327 51                        // sk06_mlp_down.py:327:51
	cvt.s64.s32 	%rd6, %r120;
	add.s64 	%rd38, %rd75, %rd6;
	cvt.s64.s32 	%rd7, %r121;
	add.s64 	%rd39, %rd75, %rd7;
	cvt.s64.s32 	%rd8, %r122;
	add.s64 	%rd40, %rd75, %rd8;
	cvt.s64.s32 	%rd9, %r123;
	add.s64 	%rd41, %rd75, %rd9;
	cvt.s64.s32 	%rd10, %r124;
	add.s64 	%rd42, %rd75, %rd10;
	cvt.s64.s32 	%rd11, %r125;
	add.s64 	%rd43, %rd75, %rd11;
	cvt.s64.s32 	%rd12, %r126;
	add.s64 	%rd44, %rd75, %rd12;
	cvt.s64.s32 	%rd13, %r127;
	add.s64 	%rd45, %rd75, %rd13;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	setp.gt.s32 	%p1, %r22, 127;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	shl.b32 	%r132, %r11, 4;
	and.b32 	%r12, %r2, 56;
	shl.b32 	%r133, %r12, 1;
	xor.b32 	%r134, %r132, %r133;
	mov.b32 	%r135, global_smem;
	add.s32 	%r31, %r135, %r134;
	add.s32 	%r26, %r31, 49152;
	selp.b32 	%r27, 16, 0, %p1;
	// begin inline asm
	cp.async.cg.shared.global [ %r26 + 0 ], [ %rd34 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r28, %r31, 51200;
	// begin inline asm
	cp.async.cg.shared.global [ %r28 + 0 ], [ %rd35 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r29, %r31, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r29 + 0 ], [ %rd36 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r30, %r31, 55296;
	// begin inline asm
	cp.async.cg.shared.global [ %r30 + 0 ], [ %rd37 + 0 ], 0x10, %r27;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	// begin inline asm
	cp.async.cg.shared.global [ %r31 + 0 ], [ %rd38 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r32, %r31, 2048;
	// begin inline asm
	cp.async.cg.shared.global [ %r32 + 0 ], [ %rd39 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r33, %r31, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r33 + 0 ], [ %rd40 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r34, %r31, 6144;
	// begin inline asm
	cp.async.cg.shared.global [ %r34 + 0 ], [ %rd41 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r35, %r31, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r35 + 0 ], [ %rd42 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r36, %r31, 10240;
	// begin inline asm
	cp.async.cg.shared.global [ %r36 + 0 ], [ %rd43 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r37, %r31, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r37 + 0 ], [ %rd44 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r38, %r31, 14336;
	// begin inline asm
	cp.async.cg.shared.global [ %r38 + 0 ], [ %rd45 + 0 ], 0x10, %r27;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	setp.gt.s32 	%p2, %r22, 255;
	.loc	1 344 18                        // sk06_mlp_down.py:344:18
	add.s64 	%rd46, %rd34, 128;
	add.s64 	%rd47, %rd35, 128;
	add.s64 	%rd48, %rd36, 128;
	add.s64 	%rd49, %rd37, 128;
	.loc	1 345 18                        // sk06_mlp_down.py:345:18
	add.s64 	%rd50, %rd38, 128;
	add.s64 	%rd51, %rd39, 128;
	add.s64 	%rd52, %rd40, 128;
	add.s64 	%rd53, %rd41, 128;
	add.s64 	%rd54, %rd42, 128;
	add.s64 	%rd55, %rd43, 128;
	add.s64 	%rd56, %rd44, 128;
	add.s64 	%rd57, %rd45, 128;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	bar.sync 	0;
	add.s32 	%r39, %r31, 57344;
	selp.b32 	%r40, 16, 0, %p2;
	// begin inline asm
	cp.async.cg.shared.global [ %r39 + 0 ], [ %rd46 + 0 ], 0x10, %r40;
	// end inline asm
	add.s32 	%r41, %r31, 59392;
	// begin inline asm
	cp.async.cg.shared.global [ %r41 + 0 ], [ %rd47 + 0 ], 0x10, %r40;
	// end inline asm
	add.s32 	%r42, %r31, 61440;
	// begin inline asm
	cp.async.cg.shared.global [ %r42 + 0 ], [ %rd48 + 0 ], 0x10, %r40;
	// end inline asm
	add.s32 	%r43, %r31, 63488;
	// begin inline asm
	cp.async.cg.shared.global [ %r43 + 0 ], [ %rd49 + 0 ], 0x10, %r40;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	add.s32 	%r44, %r31, 16384;
	// begin inline asm
	cp.async.cg.shared.global [ %r44 + 0 ], [ %rd50 + 0 ], 0x10, %r40;
	// end inline asm
	add.s32 	%r45, %r31, 18432;
	// begin inline asm
	cp.async.cg.shared.global [ %r45 + 0 ], [ %rd51 + 0 ], 0x10, %r40;
	// end inline asm
	add.s32 	%r46, %r31, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r46 + 0 ], [ %rd52 + 0 ], 0x10, %r40;
	// end inline asm
	add.s32 	%r47, %r31, 22528;
	// begin inline asm
	cp.async.cg.shared.global [ %r47 + 0 ], [ %rd53 + 0 ], 0x10, %r40;
	// end inline asm
	add.s32 	%r48, %r31, 24576;
	// begin inline asm
	cp.async.cg.shared.global [ %r48 + 0 ], [ %rd54 + 0 ], 0x10, %r40;
	// end inline asm
	add.s32 	%r49, %r31, 26624;
	// begin inline asm
	cp.async.cg.shared.global [ %r49 + 0 ], [ %rd55 + 0 ], 0x10, %r40;
	// end inline asm
	add.s32 	%r50, %r31, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r50 + 0 ], [ %rd56 + 0 ], 0x10, %r40;
	// end inline asm
	add.s32 	%r51, %r31, 30720;
	// begin inline asm
	cp.async.cg.shared.global [ %r51 + 0 ], [ %rd57 + 0 ], 0x10, %r40;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	setp.gt.s32 	%p3, %r22, 383;
	.loc	1 344 18                        // sk06_mlp_down.py:344:18
	add.s64 	%rd58, %rd34, 256;
	add.s64 	%rd59, %rd35, 256;
	add.s64 	%rd60, %rd36, 256;
	add.s64 	%rd61, %rd37, 256;
	.loc	1 345 18                        // sk06_mlp_down.py:345:18
	add.s64 	%rd62, %rd38, 256;
	add.s64 	%rd63, %rd39, 256;
	add.s64 	%rd64, %rd40, 256;
	add.s64 	%rd65, %rd41, 256;
	add.s64 	%rd66, %rd42, 256;
	add.s64 	%rd67, %rd43, 256;
	add.s64 	%rd68, %rd44, 256;
	add.s64 	%rd69, %rd45, 256;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	bar.sync 	0;
	add.s32 	%r52, %r31, 65536;
	selp.b32 	%r53, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r52 + 0 ], [ %rd58 + 0 ], 0x10, %r53;
	// end inline asm
	add.s32 	%r54, %r31, 67584;
	// begin inline asm
	cp.async.cg.shared.global [ %r54 + 0 ], [ %rd59 + 0 ], 0x10, %r53;
	// end inline asm
	add.s32 	%r55, %r31, 69632;
	// begin inline asm
	cp.async.cg.shared.global [ %r55 + 0 ], [ %rd60 + 0 ], 0x10, %r53;
	// end inline asm
	add.s32 	%r56, %r31, 71680;
	// begin inline asm
	cp.async.cg.shared.global [ %r56 + 0 ], [ %rd61 + 0 ], 0x10, %r53;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	add.s32 	%r57, %r31, 32768;
	// begin inline asm
	cp.async.cg.shared.global [ %r57 + 0 ], [ %rd62 + 0 ], 0x10, %r53;
	// end inline asm
	add.s32 	%r58, %r31, 34816;
	// begin inline asm
	cp.async.cg.shared.global [ %r58 + 0 ], [ %rd63 + 0 ], 0x10, %r53;
	// end inline asm
	add.s32 	%r59, %r31, 36864;
	// begin inline asm
	cp.async.cg.shared.global [ %r59 + 0 ], [ %rd64 + 0 ], 0x10, %r53;
	// end inline asm
	add.s32 	%r60, %r31, 38912;
	// begin inline asm
	cp.async.cg.shared.global [ %r60 + 0 ], [ %rd65 + 0 ], 0x10, %r53;
	// end inline asm
	add.s32 	%r61, %r31, 40960;
	// begin inline asm
	cp.async.cg.shared.global [ %r61 + 0 ], [ %rd66 + 0 ], 0x10, %r53;
	// end inline asm
	add.s32 	%r62, %r31, 43008;
	// begin inline asm
	cp.async.cg.shared.global [ %r62 + 0 ], [ %rd67 + 0 ], 0x10, %r53;
	// end inline asm
	add.s32 	%r63, %r31, 45056;
	// begin inline asm
	cp.async.cg.shared.global [ %r63 + 0 ], [ %rd68 + 0 ], 0x10, %r53;
	// end inline asm
	add.s32 	%r64, %r31, 47104;
	// begin inline asm
	cp.async.cg.shared.global [ %r64 + 0 ], [ %rd69 + 0 ], 0x10, %r53;
	// end inline asm
	cp.async.commit_group;
	cvt.u32.u64 	%r772, %rd5;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	@%p1 bra 	$L__BB0_2;
	bra.uni 	$L__BB0_1;
$L__BB0_2:                              // %.lr.ph
	.loc	1 0 23                          // sk06_mlp_down.py:0:23
	ld.param.b32 	%r25, [_sk06_mlp_down_kernel_param_14];
	ld.param.b64 	%rd70, [_sk06_mlp_down_kernel_param_6];
	or.b32 	%r106, %r8, %r11;
	rem.s32 	%r115, %r106, %r21;
	shr.s32 	%r128, %r115, 31;
	shr.u32 	%r129, %r128, 25;
	add.s32 	%r130, %r115, %r129;
	shr.s32 	%r131, %r130, 7;
	mad.wide.s32 	%rd14, %r131, 4, %rd70;
	.loc	1 337 28                        // sk06_mlp_down.py:337:28
	shr.u32 	%r136, %r22, 7;
	add.s32 	%r137, %r136, -3;
	shl.b32 	%r138, %r10, 7;
	and.b32 	%r778, %r2, 16;
	or.b32 	%r139, %r138, %r772;
	xor.b32 	%r13, %r139, %r778;
	xor.b32 	%r14, %r13, 32;
	xor.b32 	%r15, %r13, 64;
	xor.b32 	%r16, %r13, 96;
	shl.b32 	%r140, %r9, 7;
	and.b32 	%r776, %r2, 96;
	shl.b32 	%r141, %r776, 5;
	shl.b32 	%r142, %r2, 1;
	and.b32 	%r143, %r142, 48;
	or.b32 	%r144, %r140, %r141;
	xor.b32 	%r145, %r772, %r143;
	or.b32 	%r17, %r144, %r145;
	xor.b32 	%r18, %r17, 64;
	shl.b32 	%r146, %r11, 2;
	add.s32 	%r147, %r135, %r146;
	add.s32 	%r313, %r147, 73728;
	shl.b32 	%r777, %r2, 3;
	and.b32 	%r148, %r777, 24;
	add.s32 	%r149, %r135, %r148;
	add.s32 	%r19, %r149, %r776;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	cvt.s64.s32 	%rd15, %r137;
	and.b32 	%r150, %r22, -128;
	cvt.u64.u32 	%rd16, %r150;
	add.s64 	%rd76, %rd5, %rd13;
	add.s64 	%rd77, %rd76, %rd30;
	add.s64 	%rd17, %rd77, 384;
	add.s64 	%rd78, %rd5, %rd12;
	add.s64 	%rd79, %rd78, %rd30;
	add.s64 	%rd18, %rd79, 384;
	add.s64 	%rd80, %rd5, %rd11;
	add.s64 	%rd81, %rd80, %rd30;
	add.s64 	%rd19, %rd81, 384;
	add.s64 	%rd82, %rd5, %rd10;
	add.s64 	%rd83, %rd82, %rd30;
	add.s64 	%rd20, %rd83, 384;
	add.s64 	%rd84, %rd5, %rd9;
	add.s64 	%rd85, %rd84, %rd30;
	add.s64 	%rd21, %rd85, 384;
	add.s64 	%rd86, %rd5, %rd8;
	add.s64 	%rd87, %rd86, %rd30;
	add.s64 	%rd22, %rd87, 384;
	add.s64 	%rd88, %rd5, %rd7;
	add.s64 	%rd89, %rd88, %rd30;
	add.s64 	%rd23, %rd89, 384;
	add.s64 	%rd90, %rd5, %rd6;
	add.s64 	%rd91, %rd90, %rd30;
	add.s64 	%rd24, %rd91, 384;
	add.s64 	%rd92, %rd5, %rd4;
	add.s64 	%rd93, %rd92, %rd29;
	add.s64 	%rd25, %rd93, 384;
	add.s64 	%rd94, %rd5, %rd3;
	add.s64 	%rd95, %rd94, %rd29;
	add.s64 	%rd26, %rd95, 384;
	add.s64 	%rd96, %rd5, %rd2;
	add.s64 	%rd97, %rd96, %rd29;
	add.s64 	%rd27, %rd97, 384;
	add.s64 	%rd98, %rd5, %rd1;
	add.s64 	%rd99, %rd98, %rd29;
	add.s64 	%rd28, %rd99, 384;
	mov.b32 	%r779, 0f00000000;
	mov.b32 	%r775, 2;
	mov.b32 	%r774, -1;
	mov.b64 	%rd152, 0;
	mov.b32 	%r151, 0;
	mov.b32 	%r773, %r151;
	mov.b64 	%rd153, %rd152;
	mov.b32 	%r780, %r779;
	mov.b32 	%r781, %r779;
	mov.b32 	%r782, %r779;
	mov.b32 	%r783, %r779;
	mov.b32 	%r784, %r779;
	mov.b32 	%r785, %r779;
	mov.b32 	%r786, %r779;
	mov.b32 	%r787, %r779;
	mov.b32 	%r788, %r779;
	mov.b32 	%r789, %r779;
	mov.b32 	%r790, %r779;
	mov.b32 	%r791, %r779;
	mov.b32 	%r792, %r779;
	mov.b32 	%r793, %r779;
	mov.b32 	%r794, %r779;
	mov.b32 	%r795, %r779;
	mov.b32 	%r796, %r779;
	mov.b32 	%r797, %r779;
	mov.b32 	%r798, %r779;
	mov.b32 	%r799, %r779;
	mov.b32 	%r800, %r779;
	mov.b32 	%r801, %r779;
	mov.b32 	%r802, %r779;
	mov.b32 	%r803, %r779;
	mov.b32 	%r804, %r779;
	mov.b32 	%r805, %r779;
	mov.b32 	%r806, %r779;
	mov.b32 	%r807, %r779;
	mov.b32 	%r808, %r779;
	mov.b32 	%r809, %r779;
	mov.b32 	%r810, %r779;
	mov.b32 	%r811, %r779;
	mov.b32 	%r812, %r779;
	mov.b32 	%r813, %r779;
	mov.b32 	%r814, %r779;
	mov.b32 	%r815, %r779;
	mov.b32 	%r816, %r779;
	mov.b32 	%r817, %r779;
	mov.b32 	%r818, %r779;
	mov.b32 	%r819, %r779;
	mov.b32 	%r820, %r779;
	mov.b32 	%r821, %r779;
	mov.b32 	%r822, %r779;
	mov.b32 	%r823, %r779;
	mov.b32 	%r824, %r779;
	mov.b32 	%r825, %r779;
	mov.b32 	%r826, %r779;
	mov.b32 	%r827, %r779;
	mov.b32 	%r828, %r779;
	mov.b32 	%r829, %r779;
	mov.b32 	%r830, %r779;
	mov.b32 	%r831, %r779;
	mov.b32 	%r832, %r779;
	mov.b32 	%r833, %r779;
	mov.b32 	%r834, %r779;
	mov.b32 	%r835, %r779;
	mov.b32 	%r836, %r779;
	mov.b32 	%r837, %r779;
	mov.b32 	%r838, %r779;
	mov.b32 	%r839, %r779;
	mov.b32 	%r840, %r779;
	mov.b32 	%r841, %r779;
	mov.b32 	%r842, %r779;
$L__BB0_3:                              // =>This Inner Loop Header: Depth=1
	setp.lt.s64 	%p4, %rd153, %rd15;
	add.s32 	%r327, %r774, 1;
	setp.gt.s32 	%p5, %r327, 2;
	selp.b32 	%r774, 0, %r327, %p5;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	cp.async.wait_group 	4;
	bar.sync 	0;
	shl.b32 	%r328, %r774, 13;
	add.s32 	%r329, %r135, %r328;
	add.s32 	%r330, %r329, %r13;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r152, %r153, %r154, %r155}, [%r330+49152];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r164, %r165, %r166, %r167}, [%r330+51200];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r168, %r169, %r170, %r171}, [%r330+53248];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r172, %r173, %r174, %r175}, [%r330+55296];
	add.s32 	%r331, %r329, %r14;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r180, %r181, %r182, %r183}, [%r331+49152];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r208, %r209, %r210, %r211}, [%r331+51200];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r228, %r229, %r230, %r231}, [%r331+53248];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r248, %r249, %r250, %r251}, [%r331+55296];
	add.s32 	%r332, %r329, %r15;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r264, %r265, %r266, %r267}, [%r332+49152];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r276, %r277, %r278, %r279}, [%r332+51200];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r280, %r281, %r282, %r283}, [%r332+53248];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r284, %r285, %r286, %r287}, [%r332+55296];
	add.s32 	%r333, %r329, %r16;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r288, %r289, %r290, %r291}, [%r333+49152];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r300, %r301, %r302, %r303}, [%r333+51200];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r304, %r305, %r306, %r307}, [%r333+53248];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r308, %r309, %r310, %r311}, [%r333+55296];
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	add.s32 	%r334, %r329, %r328;
	add.s32 	%r335, %r334, %r17;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r156, %r157, %r184, %r185}, [%r335];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r158, %r159, %r190, %r191}, [%r335+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r160, %r161, %r196, %r197}, [%r335+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r162, %r163, %r202, %r203}, [%r335+12288];
	add.s32 	%r336, %r334, %r18;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r268, %r269, %r292, %r293}, [%r336];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r270, %r271, %r294, %r295}, [%r336+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r272, %r273, %r296, %r297}, [%r336+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r274, %r275, %r298, %r299}, [%r336+12288];
	.loc	1 338 36                        // sk06_mlp_down.py:338:36
	mov.b32 	%r176, %r151;
	mov.b32 	%r177, %r151;
	mov.b32 	%r178, %r151;
	mov.b32 	%r179, %r151;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r176, %r177, %r178, %r179 }, { %r152, %r153, %r154, %r155 }, { %r156, %r157 }, { %r176, %r177, %r178, %r179 };
	// end inline asm
	mov.b32 	%r186, %r151;
	mov.b32 	%r187, %r151;
	mov.b32 	%r188, %r151;
	mov.b32 	%r189, %r151;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r186, %r187, %r188, %r189 }, { %r152, %r153, %r154, %r155 }, { %r158, %r159 }, { %r186, %r187, %r188, %r189 };
	// end inline asm
	mov.b32 	%r192, %r151;
	mov.b32 	%r193, %r151;
	mov.b32 	%r194, %r151;
	mov.b32 	%r195, %r151;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r192, %r193, %r194, %r195 }, { %r152, %r153, %r154, %r155 }, { %r160, %r161 }, { %r192, %r193, %r194, %r195 };
	// end inline asm
	mov.b32 	%r198, %r151;
	mov.b32 	%r199, %r151;
	mov.b32 	%r200, %r151;
	mov.b32 	%r201, %r151;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r198, %r199, %r200, %r201 }, { %r152, %r153, %r154, %r155 }, { %r162, %r163 }, { %r198, %r199, %r200, %r201 };
	// end inline asm
	mov.b32 	%r204, %r151;
	mov.b32 	%r205, %r151;
	mov.b32 	%r206, %r151;
	mov.b32 	%r207, %r151;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r204, %r205, %r206, %r207 }, { %r164, %r165, %r166, %r167 }, { %r156, %r157 }, { %r204, %r205, %r206, %r207 };
	// end inline asm
	mov.b32 	%r212, %r151;
	mov.b32 	%r213, %r151;
	mov.b32 	%r214, %r151;
	mov.b32 	%r215, %r151;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r212, %r213, %r214, %r215 }, { %r164, %r165, %r166, %r167 }, { %r158, %r159 }, { %r212, %r213, %r214, %r215 };
	// end inline asm
	mov.b32 	%r216, %r151;
	mov.b32 	%r217, %r151;
	mov.b32 	%r218, %r151;
	mov.b32 	%r219, %r151;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r216, %r217, %r218, %r219 }, { %r164, %r165, %r166, %r167 }, { %r160, %r161 }, { %r216, %r217, %r218, %r219 };
	// end inline asm
	mov.b32 	%r220, %r151;
	mov.b32 	%r221, %r151;
	mov.b32 	%r222, %r151;
	mov.b32 	%r223, %r151;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r220, %r221, %r222, %r223 }, { %r164, %r165, %r166, %r167 }, { %r162, %r163 }, { %r220, %r221, %r222, %r223 };
	// end inline asm
	mov.b32 	%r224, %r151;
	mov.b32 	%r225, %r151;
	mov.b32 	%r226, %r151;
	mov.b32 	%r227, %r151;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r224, %r225, %r226, %r227 }, { %r168, %r169, %r170, %r171 }, { %r156, %r157 }, { %r224, %r225, %r226, %r227 };
	// end inline asm
	mov.b32 	%r232, %r151;
	mov.b32 	%r233, %r151;
	mov.b32 	%r234, %r151;
	mov.b32 	%r235, %r151;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r232, %r233, %r234, %r235 }, { %r168, %r169, %r170, %r171 }, { %r158, %r159 }, { %r232, %r233, %r234, %r235 };
	// end inline asm
	mov.b32 	%r236, %r151;
	mov.b32 	%r237, %r151;
	mov.b32 	%r238, %r151;
	mov.b32 	%r239, %r151;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r236, %r237, %r238, %r239 }, { %r168, %r169, %r170, %r171 }, { %r160, %r161 }, { %r236, %r237, %r238, %r239 };
	// end inline asm
	mov.b32 	%r240, %r151;
	mov.b32 	%r241, %r151;
	mov.b32 	%r242, %r151;
	mov.b32 	%r243, %r151;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r240, %r241, %r242, %r243 }, { %r168, %r169, %r170, %r171 }, { %r162, %r163 }, { %r240, %r241, %r242, %r243 };
	// end inline asm
	mov.b32 	%r244, %r151;
	mov.b32 	%r245, %r151;
	mov.b32 	%r246, %r151;
	mov.b32 	%r247, %r151;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r244, %r245, %r246, %r247 }, { %r172, %r173, %r174, %r175 }, { %r156, %r157 }, { %r244, %r245, %r246, %r247 };
	// end inline asm
	mov.b32 	%r252, %r151;
	mov.b32 	%r253, %r151;
	mov.b32 	%r254, %r151;
	mov.b32 	%r255, %r151;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r252, %r253, %r254, %r255 }, { %r172, %r173, %r174, %r175 }, { %r158, %r159 }, { %r252, %r253, %r254, %r255 };
	// end inline asm
	mov.b32 	%r256, %r151;
	mov.b32 	%r257, %r151;
	mov.b32 	%r258, %r151;
	mov.b32 	%r259, %r151;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r256, %r257, %r258, %r259 }, { %r172, %r173, %r174, %r175 }, { %r160, %r161 }, { %r256, %r257, %r258, %r259 };
	// end inline asm
	mov.b32 	%r263, %r151;
	mov.b32 	%r260, %r151;
	mov.b32 	%r261, %r151;
	mov.b32 	%r262, %r151;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r260, %r261, %r262, %r263 }, { %r172, %r173, %r174, %r175 }, { %r162, %r163 }, { %r260, %r261, %r262, %r263 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r176, %r177, %r178, %r179 }, { %r180, %r181, %r182, %r183 }, { %r184, %r185 }, { %r176, %r177, %r178, %r179 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r186, %r187, %r188, %r189 }, { %r180, %r181, %r182, %r183 }, { %r190, %r191 }, { %r186, %r187, %r188, %r189 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r192, %r193, %r194, %r195 }, { %r180, %r181, %r182, %r183 }, { %r196, %r197 }, { %r192, %r193, %r194, %r195 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r198, %r199, %r200, %r201 }, { %r180, %r181, %r182, %r183 }, { %r202, %r203 }, { %r198, %r199, %r200, %r201 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r204, %r205, %r206, %r207 }, { %r208, %r209, %r210, %r211 }, { %r184, %r185 }, { %r204, %r205, %r206, %r207 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r212, %r213, %r214, %r215 }, { %r208, %r209, %r210, %r211 }, { %r190, %r191 }, { %r212, %r213, %r214, %r215 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r216, %r217, %r218, %r219 }, { %r208, %r209, %r210, %r211 }, { %r196, %r197 }, { %r216, %r217, %r218, %r219 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r220, %r221, %r222, %r223 }, { %r208, %r209, %r210, %r211 }, { %r202, %r203 }, { %r220, %r221, %r222, %r223 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r224, %r225, %r226, %r227 }, { %r228, %r229, %r230, %r231 }, { %r184, %r185 }, { %r224, %r225, %r226, %r227 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r232, %r233, %r234, %r235 }, { %r228, %r229, %r230, %r231 }, { %r190, %r191 }, { %r232, %r233, %r234, %r235 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r236, %r237, %r238, %r239 }, { %r228, %r229, %r230, %r231 }, { %r196, %r197 }, { %r236, %r237, %r238, %r239 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r240, %r241, %r242, %r243 }, { %r228, %r229, %r230, %r231 }, { %r202, %r203 }, { %r240, %r241, %r242, %r243 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r244, %r245, %r246, %r247 }, { %r248, %r249, %r250, %r251 }, { %r184, %r185 }, { %r244, %r245, %r246, %r247 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r252, %r253, %r254, %r255 }, { %r248, %r249, %r250, %r251 }, { %r190, %r191 }, { %r252, %r253, %r254, %r255 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r256, %r257, %r258, %r259 }, { %r248, %r249, %r250, %r251 }, { %r196, %r197 }, { %r256, %r257, %r258, %r259 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r260, %r261, %r262, %r263 }, { %r248, %r249, %r250, %r251 }, { %r202, %r203 }, { %r260, %r261, %r262, %r263 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r176, %r177, %r178, %r179 }, { %r264, %r265, %r266, %r267 }, { %r268, %r269 }, { %r176, %r177, %r178, %r179 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r186, %r187, %r188, %r189 }, { %r264, %r265, %r266, %r267 }, { %r270, %r271 }, { %r186, %r187, %r188, %r189 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r192, %r193, %r194, %r195 }, { %r264, %r265, %r266, %r267 }, { %r272, %r273 }, { %r192, %r193, %r194, %r195 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r198, %r199, %r200, %r201 }, { %r264, %r265, %r266, %r267 }, { %r274, %r275 }, { %r198, %r199, %r200, %r201 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r204, %r205, %r206, %r207 }, { %r276, %r277, %r278, %r279 }, { %r268, %r269 }, { %r204, %r205, %r206, %r207 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r212, %r213, %r214, %r215 }, { %r276, %r277, %r278, %r279 }, { %r270, %r271 }, { %r212, %r213, %r214, %r215 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r216, %r217, %r218, %r219 }, { %r276, %r277, %r278, %r279 }, { %r272, %r273 }, { %r216, %r217, %r218, %r219 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r220, %r221, %r222, %r223 }, { %r276, %r277, %r278, %r279 }, { %r274, %r275 }, { %r220, %r221, %r222, %r223 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r224, %r225, %r226, %r227 }, { %r280, %r281, %r282, %r283 }, { %r268, %r269 }, { %r224, %r225, %r226, %r227 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r232, %r233, %r234, %r235 }, { %r280, %r281, %r282, %r283 }, { %r270, %r271 }, { %r232, %r233, %r234, %r235 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r236, %r237, %r238, %r239 }, { %r280, %r281, %r282, %r283 }, { %r272, %r273 }, { %r236, %r237, %r238, %r239 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r240, %r241, %r242, %r243 }, { %r280, %r281, %r282, %r283 }, { %r274, %r275 }, { %r240, %r241, %r242, %r243 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r244, %r245, %r246, %r247 }, { %r284, %r285, %r286, %r287 }, { %r268, %r269 }, { %r244, %r245, %r246, %r247 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r252, %r253, %r254, %r255 }, { %r284, %r285, %r286, %r287 }, { %r270, %r271 }, { %r252, %r253, %r254, %r255 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r256, %r257, %r258, %r259 }, { %r284, %r285, %r286, %r287 }, { %r272, %r273 }, { %r256, %r257, %r258, %r259 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r260, %r261, %r262, %r263 }, { %r284, %r285, %r286, %r287 }, { %r274, %r275 }, { %r260, %r261, %r262, %r263 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r176, %r177, %r178, %r179 }, { %r288, %r289, %r290, %r291 }, { %r292, %r293 }, { %r176, %r177, %r178, %r179 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r186, %r187, %r188, %r189 }, { %r288, %r289, %r290, %r291 }, { %r294, %r295 }, { %r186, %r187, %r188, %r189 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r192, %r193, %r194, %r195 }, { %r288, %r289, %r290, %r291 }, { %r296, %r297 }, { %r192, %r193, %r194, %r195 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r198, %r199, %r200, %r201 }, { %r288, %r289, %r290, %r291 }, { %r298, %r299 }, { %r198, %r199, %r200, %r201 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r204, %r205, %r206, %r207 }, { %r300, %r301, %r302, %r303 }, { %r292, %r293 }, { %r204, %r205, %r206, %r207 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r212, %r213, %r214, %r215 }, { %r300, %r301, %r302, %r303 }, { %r294, %r295 }, { %r212, %r213, %r214, %r215 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r216, %r217, %r218, %r219 }, { %r300, %r301, %r302, %r303 }, { %r296, %r297 }, { %r216, %r217, %r218, %r219 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r220, %r221, %r222, %r223 }, { %r300, %r301, %r302, %r303 }, { %r298, %r299 }, { %r220, %r221, %r222, %r223 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r224, %r225, %r226, %r227 }, { %r304, %r305, %r306, %r307 }, { %r292, %r293 }, { %r224, %r225, %r226, %r227 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r232, %r233, %r234, %r235 }, { %r304, %r305, %r306, %r307 }, { %r294, %r295 }, { %r232, %r233, %r234, %r235 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r236, %r237, %r238, %r239 }, { %r304, %r305, %r306, %r307 }, { %r296, %r297 }, { %r236, %r237, %r238, %r239 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r240, %r241, %r242, %r243 }, { %r304, %r305, %r306, %r307 }, { %r298, %r299 }, { %r240, %r241, %r242, %r243 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r244, %r245, %r246, %r247 }, { %r308, %r309, %r310, %r311 }, { %r292, %r293 }, { %r244, %r245, %r246, %r247 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r252, %r253, %r254, %r255 }, { %r308, %r309, %r310, %r311 }, { %r294, %r295 }, { %r252, %r253, %r254, %r255 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r256, %r257, %r258, %r259 }, { %r308, %r309, %r310, %r311 }, { %r296, %r297 }, { %r256, %r257, %r258, %r259 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r260, %r261, %r262, %r263 }, { %r308, %r309, %r310, %r311 }, { %r298, %r299 }, { %r260, %r261, %r262, %r263 };
	// end inline asm
	.loc	1 340 37                        // sk06_mlp_down.py:340:37
	mad.wide.s32 	%rd100, %r773, 4, %rd14;
	.loc	1 340 27                        // sk06_mlp_down.py:340:27
	// begin inline asm
	mov.u32 %r312, 0x0;
	ld.global.b32 { %r312 }, [ %rd100 + 0 ];
	// end inline asm
	.loc	1 341 23                        // sk06_mlp_down.py:341:23
	// begin inline asm
	st.shared.b32 [ %r313 + 0 ], %r312;
	// end inline asm
	bar.sync 	0;
	ld.shared.v2.b32 	{%r337, %r338}, [%r19+73728];
	ld.shared.v2.b32 	{%r339, %r340}, [%r19+73856];
	ld.shared.v2.b32 	{%r341, %r342}, [%r19+73984];
	ld.shared.v2.b32 	{%r343, %r344}, [%r19+74112];
	cvt.rn.f32.s32 	%r345, %r176;
	cvt.rn.f32.s32 	%r346, %r177;
	cvt.rn.f32.s32 	%r347, %r178;
	cvt.rn.f32.s32 	%r348, %r179;
	cvt.rn.f32.s32 	%r349, %r186;
	cvt.rn.f32.s32 	%r350, %r187;
	cvt.rn.f32.s32 	%r351, %r188;
	cvt.rn.f32.s32 	%r352, %r189;
	cvt.rn.f32.s32 	%r353, %r192;
	cvt.rn.f32.s32 	%r354, %r193;
	cvt.rn.f32.s32 	%r355, %r194;
	cvt.rn.f32.s32 	%r356, %r195;
	cvt.rn.f32.s32 	%r357, %r198;
	cvt.rn.f32.s32 	%r358, %r199;
	cvt.rn.f32.s32 	%r359, %r200;
	cvt.rn.f32.s32 	%r360, %r201;
	cvt.rn.f32.s32 	%r361, %r204;
	cvt.rn.f32.s32 	%r362, %r205;
	cvt.rn.f32.s32 	%r363, %r206;
	cvt.rn.f32.s32 	%r364, %r207;
	cvt.rn.f32.s32 	%r365, %r212;
	cvt.rn.f32.s32 	%r366, %r213;
	cvt.rn.f32.s32 	%r367, %r214;
	cvt.rn.f32.s32 	%r368, %r215;
	cvt.rn.f32.s32 	%r369, %r216;
	cvt.rn.f32.s32 	%r370, %r217;
	cvt.rn.f32.s32 	%r371, %r218;
	cvt.rn.f32.s32 	%r372, %r219;
	cvt.rn.f32.s32 	%r373, %r220;
	cvt.rn.f32.s32 	%r374, %r221;
	cvt.rn.f32.s32 	%r375, %r222;
	cvt.rn.f32.s32 	%r376, %r223;
	cvt.rn.f32.s32 	%r377, %r224;
	cvt.rn.f32.s32 	%r378, %r225;
	cvt.rn.f32.s32 	%r379, %r226;
	cvt.rn.f32.s32 	%r380, %r227;
	cvt.rn.f32.s32 	%r381, %r232;
	cvt.rn.f32.s32 	%r382, %r233;
	cvt.rn.f32.s32 	%r383, %r234;
	cvt.rn.f32.s32 	%r384, %r235;
	cvt.rn.f32.s32 	%r385, %r236;
	cvt.rn.f32.s32 	%r386, %r237;
	cvt.rn.f32.s32 	%r387, %r238;
	cvt.rn.f32.s32 	%r388, %r239;
	cvt.rn.f32.s32 	%r389, %r240;
	cvt.rn.f32.s32 	%r390, %r241;
	cvt.rn.f32.s32 	%r391, %r242;
	cvt.rn.f32.s32 	%r392, %r243;
	cvt.rn.f32.s32 	%r393, %r244;
	cvt.rn.f32.s32 	%r394, %r245;
	cvt.rn.f32.s32 	%r395, %r246;
	cvt.rn.f32.s32 	%r396, %r247;
	cvt.rn.f32.s32 	%r397, %r252;
	cvt.rn.f32.s32 	%r398, %r253;
	cvt.rn.f32.s32 	%r399, %r254;
	cvt.rn.f32.s32 	%r400, %r255;
	cvt.rn.f32.s32 	%r401, %r256;
	cvt.rn.f32.s32 	%r402, %r257;
	cvt.rn.f32.s32 	%r403, %r258;
	cvt.rn.f32.s32 	%r404, %r259;
	cvt.rn.f32.s32 	%r405, %r260;
	cvt.rn.f32.s32 	%r406, %r261;
	cvt.rn.f32.s32 	%r407, %r262;
	cvt.rn.f32.s32 	%r408, %r263;
	.loc	1 341 19                        // sk06_mlp_down.py:341:19
	fma.rn.f32 	%r842, %r344, %r408, %r842;
	fma.rn.f32 	%r841, %r343, %r407, %r841;
	fma.rn.f32 	%r840, %r344, %r406, %r840;
	fma.rn.f32 	%r839, %r343, %r405, %r839;
	fma.rn.f32 	%r838, %r342, %r404, %r838;
	fma.rn.f32 	%r837, %r341, %r403, %r837;
	fma.rn.f32 	%r836, %r342, %r402, %r836;
	fma.rn.f32 	%r835, %r341, %r401, %r835;
	fma.rn.f32 	%r834, %r340, %r400, %r834;
	fma.rn.f32 	%r833, %r339, %r399, %r833;
	fma.rn.f32 	%r832, %r340, %r398, %r832;
	fma.rn.f32 	%r831, %r339, %r397, %r831;
	fma.rn.f32 	%r830, %r338, %r396, %r830;
	fma.rn.f32 	%r829, %r337, %r395, %r829;
	fma.rn.f32 	%r828, %r338, %r394, %r828;
	fma.rn.f32 	%r827, %r337, %r393, %r827;
	fma.rn.f32 	%r826, %r344, %r392, %r826;
	fma.rn.f32 	%r825, %r343, %r391, %r825;
	fma.rn.f32 	%r824, %r344, %r390, %r824;
	fma.rn.f32 	%r823, %r343, %r389, %r823;
	fma.rn.f32 	%r822, %r342, %r388, %r822;
	fma.rn.f32 	%r821, %r341, %r387, %r821;
	fma.rn.f32 	%r820, %r342, %r386, %r820;
	fma.rn.f32 	%r819, %r341, %r385, %r819;
	fma.rn.f32 	%r818, %r340, %r384, %r818;
	fma.rn.f32 	%r817, %r339, %r383, %r817;
	fma.rn.f32 	%r816, %r340, %r382, %r816;
	fma.rn.f32 	%r815, %r339, %r381, %r815;
	fma.rn.f32 	%r814, %r338, %r380, %r814;
	fma.rn.f32 	%r813, %r337, %r379, %r813;
	fma.rn.f32 	%r812, %r338, %r378, %r812;
	fma.rn.f32 	%r811, %r337, %r377, %r811;
	fma.rn.f32 	%r810, %r344, %r376, %r810;
	fma.rn.f32 	%r809, %r343, %r375, %r809;
	fma.rn.f32 	%r808, %r344, %r374, %r808;
	fma.rn.f32 	%r807, %r343, %r373, %r807;
	fma.rn.f32 	%r806, %r342, %r372, %r806;
	fma.rn.f32 	%r805, %r341, %r371, %r805;
	fma.rn.f32 	%r804, %r342, %r370, %r804;
	fma.rn.f32 	%r803, %r341, %r369, %r803;
	fma.rn.f32 	%r802, %r340, %r368, %r802;
	fma.rn.f32 	%r801, %r339, %r367, %r801;
	fma.rn.f32 	%r800, %r340, %r366, %r800;
	fma.rn.f32 	%r799, %r339, %r365, %r799;
	fma.rn.f32 	%r798, %r338, %r364, %r798;
	fma.rn.f32 	%r797, %r337, %r363, %r797;
	fma.rn.f32 	%r796, %r338, %r362, %r796;
	fma.rn.f32 	%r795, %r337, %r361, %r795;
	fma.rn.f32 	%r794, %r344, %r360, %r794;
	fma.rn.f32 	%r793, %r343, %r359, %r793;
	fma.rn.f32 	%r792, %r344, %r358, %r792;
	fma.rn.f32 	%r791, %r343, %r357, %r791;
	fma.rn.f32 	%r790, %r342, %r356, %r790;
	fma.rn.f32 	%r789, %r341, %r355, %r789;
	fma.rn.f32 	%r788, %r342, %r354, %r788;
	fma.rn.f32 	%r787, %r341, %r353, %r787;
	fma.rn.f32 	%r786, %r340, %r352, %r786;
	fma.rn.f32 	%r785, %r339, %r351, %r785;
	fma.rn.f32 	%r784, %r340, %r350, %r784;
	fma.rn.f32 	%r783, %r339, %r349, %r783;
	fma.rn.f32 	%r782, %r338, %r348, %r782;
	fma.rn.f32 	%r781, %r337, %r347, %r781;
	fma.rn.f32 	%r780, %r338, %r346, %r780;
	fma.rn.f32 	%r779, %r337, %r345, %r779;
	.loc	1 344 18                        // sk06_mlp_down.py:344:18
	add.s64 	%rd101, %rd28, %rd152;
	add.s64 	%rd102, %rd27, %rd152;
	add.s64 	%rd103, %rd26, %rd152;
	.loc	1 345 18                        // sk06_mlp_down.py:345:18
	add.s64 	%rd104, %rd25, %rd152;
	add.s64 	%rd105, %rd24, %rd152;
	add.s64 	%rd106, %rd23, %rd152;
	add.s64 	%rd107, %rd22, %rd152;
	add.s64 	%rd108, %rd21, %rd152;
	add.s64 	%rd109, %rd20, %rd152;
	add.s64 	%rd110, %rd19, %rd152;
	add.s64 	%rd111, %rd18, %rd152;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	add.s64 	%rd112, %rd17, %rd152;
	add.s32 	%r409, %r775, 1;
	setp.gt.s32 	%p6, %r409, 2;
	selp.b32 	%r775, 0, %r409, %p6;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	shl.b32 	%r410, %r775, 13;
	add.s32 	%r411, %r31, %r410;
	add.s32 	%r314, %r411, 49152;
	selp.b32 	%r315, 16, 0, %p4;
	// begin inline asm
	cp.async.cg.shared.global [ %r314 + 0 ], [ %rd101 + 0 ], 0x10, %r315;
	// end inline asm
	add.s32 	%r316, %r411, 51200;
	// begin inline asm
	cp.async.cg.shared.global [ %r316 + 0 ], [ %rd102 + 0 ], 0x10, %r315;
	// end inline asm
	add.s32 	%r317, %r411, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r317 + 0 ], [ %rd103 + 0 ], 0x10, %r315;
	// end inline asm
	add.s32 	%r318, %r411, 55296;
	// begin inline asm
	cp.async.cg.shared.global [ %r318 + 0 ], [ %rd104 + 0 ], 0x10, %r315;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	add.s32 	%r319, %r411, %r410;
	// begin inline asm
	cp.async.cg.shared.global [ %r319 + 0 ], [ %rd105 + 0 ], 0x10, %r315;
	// end inline asm
	add.s32 	%r320, %r319, 2048;
	// begin inline asm
	cp.async.cg.shared.global [ %r320 + 0 ], [ %rd106 + 0 ], 0x10, %r315;
	// end inline asm
	add.s32 	%r321, %r319, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r321 + 0 ], [ %rd107 + 0 ], 0x10, %r315;
	// end inline asm
	add.s32 	%r322, %r319, 6144;
	// begin inline asm
	cp.async.cg.shared.global [ %r322 + 0 ], [ %rd108 + 0 ], 0x10, %r315;
	// end inline asm
	add.s32 	%r323, %r319, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r323 + 0 ], [ %rd109 + 0 ], 0x10, %r315;
	// end inline asm
	add.s32 	%r324, %r319, 10240;
	// begin inline asm
	cp.async.cg.shared.global [ %r324 + 0 ], [ %rd110 + 0 ], 0x10, %r315;
	// end inline asm
	add.s32 	%r325, %r319, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r325 + 0 ], [ %rd111 + 0 ], 0x10, %r315;
	// end inline asm
	add.s32 	%r326, %r319, 14336;
	// begin inline asm
	cp.async.cg.shared.global [ %r326 + 0 ], [ %rd112 + 0 ], 0x10, %r315;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	add.s64 	%rd153, %rd153, 1;
	add.s64 	%rd152, %rd152, 128;
	add.s32 	%r773, %r773, %r25;
	setp.ne.b64 	%p7, %rd16, %rd152;
	@%p7 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_4;
$L__BB0_1:                              // %.._crit_edge_crit_edge
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	and.b32 	%r778, %r2, 16;
	shl.b32 	%r777, %r2, 3;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	and.b32 	%r776, %r2, 96;
	mov.b32 	%r779, 0f00000000;
	mov.b32 	%r780, %r779;
	mov.b32 	%r781, %r779;
	mov.b32 	%r782, %r779;
	mov.b32 	%r783, %r779;
	mov.b32 	%r784, %r779;
	mov.b32 	%r785, %r779;
	mov.b32 	%r786, %r779;
	mov.b32 	%r787, %r779;
	mov.b32 	%r788, %r779;
	mov.b32 	%r789, %r779;
	mov.b32 	%r790, %r779;
	mov.b32 	%r791, %r779;
	mov.b32 	%r792, %r779;
	mov.b32 	%r793, %r779;
	mov.b32 	%r794, %r779;
	mov.b32 	%r795, %r779;
	mov.b32 	%r796, %r779;
	mov.b32 	%r797, %r779;
	mov.b32 	%r798, %r779;
	mov.b32 	%r799, %r779;
	mov.b32 	%r800, %r779;
	mov.b32 	%r801, %r779;
	mov.b32 	%r802, %r779;
	mov.b32 	%r803, %r779;
	mov.b32 	%r804, %r779;
	mov.b32 	%r805, %r779;
	mov.b32 	%r806, %r779;
	mov.b32 	%r807, %r779;
	mov.b32 	%r808, %r779;
	mov.b32 	%r809, %r779;
	mov.b32 	%r810, %r779;
	mov.b32 	%r811, %r779;
	mov.b32 	%r812, %r779;
	mov.b32 	%r813, %r779;
	mov.b32 	%r814, %r779;
	mov.b32 	%r815, %r779;
	mov.b32 	%r816, %r779;
	mov.b32 	%r817, %r779;
	mov.b32 	%r818, %r779;
	mov.b32 	%r819, %r779;
	mov.b32 	%r820, %r779;
	mov.b32 	%r821, %r779;
	mov.b32 	%r822, %r779;
	mov.b32 	%r823, %r779;
	mov.b32 	%r824, %r779;
	mov.b32 	%r825, %r779;
	mov.b32 	%r826, %r779;
	mov.b32 	%r827, %r779;
	mov.b32 	%r828, %r779;
	mov.b32 	%r829, %r779;
	mov.b32 	%r830, %r779;
	mov.b32 	%r831, %r779;
	mov.b32 	%r832, %r779;
	mov.b32 	%r833, %r779;
	mov.b32 	%r834, %r779;
	mov.b32 	%r835, %r779;
	mov.b32 	%r836, %r779;
	mov.b32 	%r837, %r779;
	mov.b32 	%r838, %r779;
	mov.b32 	%r839, %r779;
	mov.b32 	%r840, %r779;
	mov.b32 	%r841, %r779;
	mov.b32 	%r842, %r779;
$L__BB0_4:                              // %._crit_edge
	.loc	1 323 45                        // sk06_mlp_down.py:323:45
	or.b32 	%r522, %r8, %r772;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r523, %r522, 8;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r524, %r523, %r21;
	rem.s32 	%r525, %r522, %r21;
	.loc	1 323 45                        // sk06_mlp_down.py:323:45
	shl.b32 	%r526, %r10, 3;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r527, %r8, %r526;
	.loc	1 322 45                        // sk06_mlp_down.py:322:45
	shr.u32 	%r528, %r2, 2;
	bfe.u32 	%r529, %r2, 2, 3;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r530, %r529, %r1;
	or.b32 	%r531, %r530, 56;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r532, %r531, %r20;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r533, %r530, 48;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r534, %r533, %r20;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r535, %r530, 40;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r536, %r535, %r20;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r537, %r530, 32;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r538, %r537, %r20;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r539, %r530, 24;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r540, %r539, %r20;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r541, %r530, 16;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r542, %r541, %r20;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r543, %r530, 8;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r544, %r543, %r20;
	rem.s32 	%r545, %r530, %r20;
	.loc	1 322 45                        // sk06_mlp_down.py:322:45
	shr.u32 	%r546, %r2, 4;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r547, %r546, %r1;
	or.b32 	%r548, %r547, 56;
	.loc	1 322 45                        // sk06_mlp_down.py:322:45
	bfe.u32 	%r549, %r2, 4, 3;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r550, %r549, %r1;
	or.b32 	%r551, %r550, 48;
	or.b32 	%r552, %r550, 40;
	or.b32 	%r553, %r550, 32;
	or.b32 	%r554, %r550, 24;
	or.b32 	%r555, %r550, 16;
	or.b32 	%r556, %r550, 8;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 347 38                        // sk06_mlp_down.py:347:38
	mad.wide.s32 	%rd113, %r545, 4, %rd33;
	mad.wide.s32 	%rd114, %r544, 4, %rd33;
	mad.wide.s32 	%rd115, %r542, 4, %rd33;
	mad.wide.s32 	%rd116, %r540, 4, %rd33;
	mad.wide.s32 	%rd117, %r538, 4, %rd33;
	mad.wide.s32 	%rd118, %r536, 4, %rd33;
	mad.wide.s32 	%rd119, %r534, 4, %rd33;
	mad.wide.s32 	%rd120, %r532, 4, %rd33;
	.loc	1 347 24                        // sk06_mlp_down.py:347:24
	// begin inline asm
	mov.u32 %r412, 0x0;
	ld.global.b32 { %r412 }, [ %rd113 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r413, 0x0;
	ld.global.b32 { %r413 }, [ %rd114 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r414, 0x0;
	ld.global.b32 { %r414 }, [ %rd115 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r415, 0x0;
	ld.global.b32 { %r415 }, [ %rd116 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r416, 0x0;
	ld.global.b32 { %r416 }, [ %rd117 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r417, 0x0;
	ld.global.b32 { %r417 }, [ %rd118 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r418, 0x0;
	ld.global.b32 { %r418 }, [ %rd119 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r419, 0x0;
	ld.global.b32 { %r419 }, [ %rd120 + 0 ];
	// end inline asm
	.loc	1 350 49                        // sk06_mlp_down.py:350:49
	mul.lo.s32 	%r557, %r4, %r24;
	mul.lo.s32 	%r558, %r5, %r24;
	mul.lo.s32 	%r559, %r6, %r24;
	mul.lo.s32 	%r560, %r7, %r24;
	.loc	1 350 31                        // sk06_mlp_down.py:350:31
	mad.wide.s32 	%rd137, %r557, 2, %rd32;
	mad.wide.s32 	%rd138, %r558, 2, %rd32;
	mad.wide.s32 	%rd139, %r559, 2, %rd32;
	mad.wide.s32 	%rd140, %r560, 2, %rd32;
	.loc	1 350 64                        // sk06_mlp_down.py:350:64
	mul.wide.s32 	%rd141, %r525, 2;
	add.s64 	%rd121, %rd137, %rd141;
	mul.wide.s32 	%rd142, %r524, 2;
	add.s64 	%rd122, %rd137, %rd142;
	add.s64 	%rd123, %rd138, %rd141;
	add.s64 	%rd124, %rd138, %rd142;
	add.s64 	%rd125, %rd139, %rd141;
	add.s64 	%rd126, %rd139, %rd142;
	add.s64 	%rd127, %rd140, %rd141;
	add.s64 	%rd128, %rd140, %rd142;
	.loc	1 350 19                        // sk06_mlp_down.py:350:19
	// begin inline asm
	mov.u32 %r421, 0x0;
	mov.u32 %r422, 0x0;
	mov.u32 %r423, 0x0;
	mov.u32 %r424, 0x0;
	ld.global.v4.b32 { %r421, %r422, %r423, %r424 }, [ %rd121 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r426, 0x0;
	mov.u32 %r427, 0x0;
	mov.u32 %r428, 0x0;
	mov.u32 %r429, 0x0;
	ld.global.v4.b32 { %r426, %r427, %r428, %r429 }, [ %rd122 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r430, 0x0;
	mov.u32 %r431, 0x0;
	mov.u32 %r432, 0x0;
	mov.u32 %r433, 0x0;
	ld.global.v4.b32 { %r430, %r431, %r432, %r433 }, [ %rd123 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r434, 0x0;
	mov.u32 %r435, 0x0;
	mov.u32 %r436, 0x0;
	mov.u32 %r437, 0x0;
	ld.global.v4.b32 { %r434, %r435, %r436, %r437 }, [ %rd124 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r438, 0x0;
	mov.u32 %r439, 0x0;
	mov.u32 %r440, 0x0;
	mov.u32 %r441, 0x0;
	ld.global.v4.b32 { %r438, %r439, %r440, %r441 }, [ %rd125 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r442, 0x0;
	mov.u32 %r443, 0x0;
	mov.u32 %r444, 0x0;
	mov.u32 %r445, 0x0;
	ld.global.v4.b32 { %r442, %r443, %r444, %r445 }, [ %rd126 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r446, 0x0;
	mov.u32 %r447, 0x0;
	mov.u32 %r448, 0x0;
	mov.u32 %r449, 0x0;
	ld.global.v4.b32 { %r446, %r447, %r448, %r449 }, [ %rd127 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r450, 0x0;
	mov.u32 %r451, 0x0;
	mov.u32 %r452, 0x0;
	mov.u32 %r453, 0x0;
	ld.global.v4.b32 { %r450, %r451, %r452, %r453 }, [ %rd128 + 0 ];
	// end inline asm
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	shl.b32 	%r561, %r12, 6;
	shl.b32 	%r562, %r3, 1;
	or.b32 	%r563, %r561, %r772;
	xor.b32 	%r564, %r563, %r562;
	add.s32 	%r420, %r135, %r564;
	// begin inline asm
	st.shared.v4.b32 [ %r420 + 0 ], { %r421, %r422, %r423, %r424 };
	// end inline asm
	add.s32 	%r425, %r420, 256;
	// begin inline asm
	st.shared.v4.b32 [ %r425 + 0 ], { %r426, %r427, %r428, %r429 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r565, %r9, 9;
	shl.b32 	%r566, %r10, 4;
	setp.eq.b32 	%p16, %r778, 0;
	shl.b32 	%r567, %r778, 1;
	and.b32 	%r568, %r777, 256;
	and.b32 	%r569, %r528, 16;
	or.b32 	%r570, %r565, %r568;
	xor.b32 	%r571, %r566, %r567;
	xor.b32 	%r572, %r571, %r569;
	or.b32 	%r573, %r572, %r570;
	add.s32 	%r574, %r135, %r573;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r575, %r576, %r577, %r578}, [%r574];
	xor.b32 	%r579, %r573, 64;
	add.s32 	%r580, %r135, %r579;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r581, %r582, %r583, %r584}, [%r580];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r420 + 0 ], { %r430, %r431, %r432, %r433 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r425 + 0 ], { %r434, %r435, %r436, %r437 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r585, %r586, %r587, %r588}, [%r574];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r589, %r590, %r591, %r592}, [%r580];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r420 + 0 ], { %r438, %r439, %r440, %r441 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r425 + 0 ], { %r442, %r443, %r444, %r445 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r593, %r594, %r595, %r596}, [%r574];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r597, %r598, %r599, %r600}, [%r580];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r420 + 0 ], { %r446, %r447, %r448, %r449 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r425 + 0 ], { %r450, %r451, %r452, %r453 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r601, %r602, %r603, %r604}, [%r574];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r605, %r606, %r607, %r608}, [%r580];
	.loc	1 357 31                        // sk06_mlp_down.py:357:31
	setp.lt.s32 	%p17, %r550, %r20;
	setp.lt.s32 	%p18, %r556, %r20;
	setp.lt.s32 	%p19, %r555, %r20;
	setp.lt.s32 	%p20, %r554, %r20;
	setp.lt.s32 	%p21, %r553, %r20;
	setp.lt.s32 	%p22, %r552, %r20;
	setp.lt.s32 	%p23, %r551, %r20;
	setp.lt.s32 	%p24, %r548, %r20;
	.loc	1 357 54                        // sk06_mlp_down.py:357:54
	setp.lt.s32 	%p25, %r527, %r21;
	.loc	1 357 37                        // sk06_mlp_down.py:357:37
	and.pred 	%p8, %p17, %p25;
	and.pred 	%p9, %p18, %p25;
	and.pred 	%p10, %p19, %p25;
	and.pred 	%p11, %p20, %p25;
	and.pred 	%p12, %p21, %p25;
	and.pred 	%p13, %p22, %p25;
	and.pred 	%p14, %p23, %p25;
	and.pred 	%p15, %p24, %p25;
	.loc	1 355 35                        // sk06_mlp_down.py:355:35
	mul.lo.s32 	%r609, %r550, %r23;
	mul.lo.s32 	%r610, %r556, %r23;
	mul.lo.s32 	%r611, %r555, %r23;
	mul.lo.s32 	%r612, %r554, %r23;
	mul.lo.s32 	%r613, %r553, %r23;
	mul.lo.s32 	%r614, %r552, %r23;
	mul.lo.s32 	%r615, %r551, %r23;
	mul.lo.s32 	%r616, %r548, %r23;
	.loc	1 355 18                        // sk06_mlp_down.py:355:18
	mad.wide.s32 	%rd143, %r609, 2, %rd31;
	mad.wide.s32 	%rd144, %r610, 2, %rd31;
	mad.wide.s32 	%rd145, %r611, 2, %rd31;
	mad.wide.s32 	%rd146, %r612, 2, %rd31;
	mad.wide.s32 	%rd147, %r613, 2, %rd31;
	mad.wide.s32 	%rd148, %r614, 2, %rd31;
	mad.wide.s32 	%rd149, %r615, 2, %rd31;
	mad.wide.s32 	%rd150, %r616, 2, %rd31;
	.loc	1 355 50                        // sk06_mlp_down.py:355:50
	mul.wide.s32 	%rd151, %r527, 2;
	add.s64 	%rd129, %rd143, %rd151;
	add.s64 	%rd130, %rd144, %rd151;
	add.s64 	%rd131, %rd145, %rd151;
	add.s64 	%rd132, %rd146, %rd151;
	add.s64 	%rd133, %rd147, %rd151;
	add.s64 	%rd134, %rd148, %rd151;
	add.s64 	%rd135, %rd149, %rd151;
	add.s64 	%rd136, %rd150, %rd151;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs1, %rs2}, %r575;
	cvt.f32.bf16 	%r617, %rs2;
	cvt.f32.bf16 	%r618, %rs1;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r619, %r779, %r412, %r618;
	fma.rn.f32 	%r620, %r780, %r412, %r617;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r455, %r620, %r619;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs3, %rs4}, %r576;
	cvt.f32.bf16 	%r621, %rs4;
	cvt.f32.bf16 	%r622, %rs3;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r623, %r781, %r413, %r622;
	fma.rn.f32 	%r624, %r782, %r413, %r621;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r456, %r624, %r623;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs5, %rs6}, %r577;
	cvt.f32.bf16 	%r625, %rs6;
	cvt.f32.bf16 	%r626, %rs5;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r627, %r783, %r412, %r626;
	fma.rn.f32 	%r628, %r784, %r412, %r625;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r465, %r628, %r627;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs7, %rs8}, %r578;
	cvt.f32.bf16 	%r629, %rs8;
	cvt.f32.bf16 	%r630, %rs7;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r631, %r785, %r413, %r630;
	fma.rn.f32 	%r632, %r786, %r413, %r629;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r466, %r632, %r631;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs9, %rs10}, %r581;
	cvt.f32.bf16 	%r633, %rs10;
	cvt.f32.bf16 	%r634, %rs9;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r635, %r787, %r412, %r634;
	fma.rn.f32 	%r636, %r788, %r412, %r633;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r460, %r636, %r635;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs11, %rs12}, %r582;
	cvt.f32.bf16 	%r637, %rs12;
	cvt.f32.bf16 	%r638, %rs11;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r639, %r789, %r413, %r638;
	fma.rn.f32 	%r640, %r790, %r413, %r637;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r461, %r640, %r639;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs13, %rs14}, %r583;
	cvt.f32.bf16 	%r641, %rs14;
	cvt.f32.bf16 	%r642, %rs13;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r643, %r791, %r412, %r642;
	fma.rn.f32 	%r644, %r792, %r412, %r641;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r470, %r644, %r643;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs15, %rs16}, %r584;
	cvt.f32.bf16 	%r645, %rs16;
	cvt.f32.bf16 	%r646, %rs15;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r647, %r793, %r413, %r646;
	fma.rn.f32 	%r648, %r794, %r413, %r645;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r471, %r648, %r647;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs17, %rs18}, %r585;
	cvt.f32.bf16 	%r649, %rs18;
	cvt.f32.bf16 	%r650, %rs17;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r651, %r795, %r414, %r650;
	fma.rn.f32 	%r652, %r796, %r414, %r649;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r457, %r652, %r651;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs19, %rs20}, %r586;
	cvt.f32.bf16 	%r653, %rs20;
	cvt.f32.bf16 	%r654, %rs19;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r655, %r797, %r415, %r654;
	fma.rn.f32 	%r656, %r798, %r415, %r653;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r458, %r656, %r655;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs21, %rs22}, %r587;
	cvt.f32.bf16 	%r657, %rs22;
	cvt.f32.bf16 	%r658, %rs21;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r659, %r799, %r414, %r658;
	fma.rn.f32 	%r660, %r800, %r414, %r657;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r467, %r660, %r659;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs23, %rs24}, %r588;
	cvt.f32.bf16 	%r661, %rs24;
	cvt.f32.bf16 	%r662, %rs23;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r663, %r801, %r415, %r662;
	fma.rn.f32 	%r664, %r802, %r415, %r661;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r468, %r664, %r663;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs25, %rs26}, %r589;
	cvt.f32.bf16 	%r665, %rs26;
	cvt.f32.bf16 	%r666, %rs25;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r667, %r803, %r414, %r666;
	fma.rn.f32 	%r668, %r804, %r414, %r665;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r462, %r668, %r667;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs27, %rs28}, %r590;
	cvt.f32.bf16 	%r669, %rs28;
	cvt.f32.bf16 	%r670, %rs27;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r671, %r805, %r415, %r670;
	fma.rn.f32 	%r672, %r806, %r415, %r669;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r463, %r672, %r671;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs29, %rs30}, %r591;
	cvt.f32.bf16 	%r673, %rs30;
	cvt.f32.bf16 	%r674, %rs29;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r675, %r807, %r414, %r674;
	fma.rn.f32 	%r676, %r808, %r414, %r673;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r472, %r676, %r675;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs31, %rs32}, %r592;
	cvt.f32.bf16 	%r677, %rs32;
	cvt.f32.bf16 	%r678, %rs31;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r679, %r809, %r415, %r678;
	fma.rn.f32 	%r680, %r810, %r415, %r677;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r473, %r680, %r679;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs33, %rs34}, %r593;
	cvt.f32.bf16 	%r681, %rs34;
	cvt.f32.bf16 	%r682, %rs33;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r683, %r811, %r416, %r682;
	fma.rn.f32 	%r684, %r812, %r416, %r681;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r474, %r684, %r683;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs35, %rs36}, %r594;
	cvt.f32.bf16 	%r685, %rs36;
	cvt.f32.bf16 	%r686, %rs35;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r687, %r813, %r417, %r686;
	fma.rn.f32 	%r688, %r814, %r417, %r685;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r475, %r688, %r687;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs37, %rs38}, %r595;
	cvt.f32.bf16 	%r689, %rs38;
	cvt.f32.bf16 	%r690, %rs37;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r691, %r815, %r416, %r690;
	fma.rn.f32 	%r692, %r816, %r416, %r689;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r482, %r692, %r691;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs39, %rs40}, %r596;
	cvt.f32.bf16 	%r693, %rs40;
	cvt.f32.bf16 	%r694, %rs39;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r695, %r817, %r417, %r694;
	fma.rn.f32 	%r696, %r818, %r417, %r693;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r483, %r696, %r695;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs41, %rs42}, %r597;
	cvt.f32.bf16 	%r697, %rs42;
	cvt.f32.bf16 	%r698, %rs41;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r699, %r819, %r416, %r698;
	fma.rn.f32 	%r700, %r820, %r416, %r697;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r478, %r700, %r699;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs43, %rs44}, %r598;
	cvt.f32.bf16 	%r701, %rs44;
	cvt.f32.bf16 	%r702, %rs43;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r703, %r821, %r417, %r702;
	fma.rn.f32 	%r704, %r822, %r417, %r701;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r479, %r704, %r703;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs45, %rs46}, %r599;
	cvt.f32.bf16 	%r705, %rs46;
	cvt.f32.bf16 	%r706, %rs45;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r707, %r823, %r416, %r706;
	fma.rn.f32 	%r708, %r824, %r416, %r705;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r486, %r708, %r707;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs47, %rs48}, %r600;
	cvt.f32.bf16 	%r709, %rs48;
	cvt.f32.bf16 	%r710, %rs47;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r711, %r825, %r417, %r710;
	fma.rn.f32 	%r712, %r826, %r417, %r709;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r487, %r712, %r711;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs49, %rs50}, %r601;
	cvt.f32.bf16 	%r713, %rs50;
	cvt.f32.bf16 	%r714, %rs49;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r715, %r827, %r418, %r714;
	fma.rn.f32 	%r716, %r828, %r418, %r713;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r476, %r716, %r715;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs51, %rs52}, %r602;
	cvt.f32.bf16 	%r717, %rs52;
	cvt.f32.bf16 	%r718, %rs51;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r719, %r829, %r419, %r718;
	fma.rn.f32 	%r720, %r830, %r419, %r717;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r477, %r720, %r719;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs53, %rs54}, %r603;
	cvt.f32.bf16 	%r721, %rs54;
	cvt.f32.bf16 	%r722, %rs53;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r723, %r831, %r418, %r722;
	fma.rn.f32 	%r724, %r832, %r418, %r721;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r484, %r724, %r723;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs55, %rs56}, %r604;
	cvt.f32.bf16 	%r725, %rs56;
	cvt.f32.bf16 	%r726, %rs55;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r727, %r833, %r419, %r726;
	fma.rn.f32 	%r728, %r834, %r419, %r725;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r485, %r728, %r727;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs57, %rs58}, %r605;
	cvt.f32.bf16 	%r729, %rs58;
	cvt.f32.bf16 	%r730, %rs57;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r731, %r835, %r418, %r730;
	fma.rn.f32 	%r732, %r836, %r418, %r729;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r480, %r732, %r731;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs59, %rs60}, %r606;
	cvt.f32.bf16 	%r733, %rs60;
	cvt.f32.bf16 	%r734, %rs59;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r735, %r837, %r419, %r734;
	fma.rn.f32 	%r736, %r838, %r419, %r733;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r481, %r736, %r735;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs61, %rs62}, %r607;
	cvt.f32.bf16 	%r737, %rs62;
	cvt.f32.bf16 	%r738, %rs61;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r739, %r839, %r418, %r738;
	fma.rn.f32 	%r740, %r840, %r418, %r737;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r488, %r740, %r739;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs63, %rs64}, %r608;
	cvt.f32.bf16 	%r741, %rs64;
	cvt.f32.bf16 	%r742, %rs63;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r743, %r841, %r419, %r742;
	fma.rn.f32 	%r744, %r842, %r419, %r741;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r489, %r744, %r743;
	bar.sync 	0;
	and.b32 	%r745, %r2, 3;
	shl.b32 	%r746, %r745, 11;
	shl.b32 	%r747, %r745, 5;
	shl.b32 	%r748, %r2, 4;
	and.b32 	%r749, %r748, 384;
	shr.u32 	%r750, %r776, 1;
	bfe.s32 	%r751, %r2, 2, 1;
	and.b32 	%r752, %r751, 1040;
	or.b32 	%r753, %r747, %r749;
	xor.b32 	%r754, %r752, %r750;
	xor.b32 	%r755, %r754, %r753;
	or.b32 	%r756, %r755, %r746;
	add.s32 	%r454, %r135, %r756;
	// begin inline asm
	st.shared.v4.b32 [ %r454 + 0 ], { %r455, %r456, %r457, %r458 };
	// end inline asm
	add.s32 	%r459, %r454, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r459 + 0 ], { %r460, %r461, %r462, %r463 };
	// end inline asm
	xor.b32 	%r757, %r756, 64;
	add.s32 	%r464, %r135, %r757;
	// begin inline asm
	st.shared.v4.b32 [ %r464 + 0 ], { %r465, %r466, %r467, %r468 };
	// end inline asm
	add.s32 	%r469, %r464, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r469 + 0 ], { %r470, %r471, %r472, %r473 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r758, %r776, 2;
	shl.b32 	%r759, %r2, 6;
	and.b32 	%r760, %r759, 512;
	selp.b32 	%r761, 0, 1040, %p16;
	or.b32 	%r762, %r772, %r758;
	xor.b32 	%r763, %r762, %r761;
	or.b32 	%r764, %r763, %r760;
	add.s32 	%r765, %r135, %r764;
	ld.shared.v4.b32 	{%r490, %r494, %r498, %r502}, [%r765];
	xor.b32 	%r766, %r764, 32;
	add.s32 	%r767, %r135, %r766;
	ld.shared.v4.b32 	{%r491, %r495, %r499, %r503}, [%r767+2048];
	xor.b32 	%r768, %r764, 64;
	add.s32 	%r769, %r135, %r768;
	ld.shared.v4.b32 	{%r492, %r496, %r500, %r504}, [%r769+4096];
	xor.b32 	%r770, %r764, 96;
	add.s32 	%r771, %r135, %r770;
	ld.shared.v4.b32 	{%r493, %r497, %r501, %r505}, [%r771+6144];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r454 + 0 ], { %r474, %r475, %r476, %r477 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r459 + 0 ], { %r478, %r479, %r480, %r481 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r464 + 0 ], { %r482, %r483, %r484, %r485 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r469 + 0 ], { %r486, %r487, %r488, %r489 };
	// end inline asm
	bar.sync 	0;
	ld.shared.v4.b32 	{%r506, %r510, %r514, %r518}, [%r765];
	ld.shared.v4.b32 	{%r507, %r511, %r515, %r519}, [%r767+2048];
	ld.shared.v4.b32 	{%r508, %r512, %r516, %r520}, [%r769+4096];
	ld.shared.v4.b32 	{%r509, %r513, %r517, %r521}, [%r771+6144];
	.loc	1 356 8                         // sk06_mlp_down.py:356:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd129 + 0 ], { %r490, %r491, %r492, %r493 };
	// end inline asm
	// begin inline asm
	@%p9 st.global.v4.b32 [ %rd130 + 0 ], { %r494, %r495, %r496, %r497 };
	// end inline asm
	// begin inline asm
	@%p10 st.global.v4.b32 [ %rd131 + 0 ], { %r498, %r499, %r500, %r501 };
	// end inline asm
	// begin inline asm
	@%p11 st.global.v4.b32 [ %rd132 + 0 ], { %r502, %r503, %r504, %r505 };
	// end inline asm
	// begin inline asm
	@%p12 st.global.v4.b32 [ %rd133 + 0 ], { %r506, %r507, %r508, %r509 };
	// end inline asm
	// begin inline asm
	@%p13 st.global.v4.b32 [ %rd134 + 0 ], { %r510, %r511, %r512, %r513 };
	// end inline asm
	// begin inline asm
	@%p14 st.global.v4.b32 [ %rd135 + 0 ], { %r514, %r515, %r516, %r517 };
	// end inline asm
	// begin inline asm
	@%p15 st.global.v4.b32 [ %rd136 + 0 ], { %r518, %r519, %r520, %r521 };
	// end inline asm
	.loc	1 354 4                         // sk06_mlp_down.py:354:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/workspace/vllm/_genesis/kernels/sk06_mlp_down.py"
	.file	2 "/usr/local/lib/python3.12/dist-packages/triton/language/standard.py"
	.section	.debug_abbrev
	{
.b8 1                                   // Abbreviation Code
.b8 17                                  // DW_TAG_compile_unit
.b8 1                                   // DW_CHILDREN_yes
.b8 37                                  // DW_AT_producer
.b8 8                                   // DW_FORM_string
.b8 19                                  // DW_AT_language
.b8 5                                   // DW_FORM_data2
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 16                                  // DW_AT_stmt_list
.b8 6                                   // DW_FORM_data4
.b8 27                                  // DW_AT_comp_dir
.b8 8                                   // DW_FORM_string
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 2                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 0                                   // DW_CHILDREN_no
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 32                                  // DW_AT_inline
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 3                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 1                                   // DW_CHILDREN_yes
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 4                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 0                                   // DW_CHILDREN_no
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 5                                   // DW_FORM_data2
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 168                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0xa1 DW_TAG_compile_unit
.b8 116                                 // DW_AT_producer
.b8 114
.b8 105
.b8 116
.b8 111
.b8 110
.b8 0
.b8 2                                   // DW_AT_language
.b8 0
.b8 115                                 // DW_AT_name
.b8 107
.b8 48
.b8 54
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 100
.b8 111
.b8 119
.b8 110
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 119
.b8 111
.b8 114
.b8 107
.b8 115
.b8 112
.b8 97
.b8 99
.b8 101
.b8 47
.b8 118
.b8 108
.b8 108
.b8 109
.b8 47
.b8 95
.b8 103
.b8 101
.b8 110
.b8 101
.b8 115
.b8 105
.b8 115
.b8 47
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 115
.b8 0
.b8 2                                   // Abbrev [2] 0x4b:0x18 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 54
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 100
.b8 111
.b8 119
.b8 110
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x63:0x48 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 75                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x78:0x19 DW_TAG_inlined_subroutine
.b32 75                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 58                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x91:0x19 DW_TAG_inlined_subroutine
.b32 75                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 59                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_6 = _Nativo(
    "sk06_mlp_down/tile64x128x128_shift1_abi15",
    _PTX_6, "_sk06_mlp_down_kernel",
    warps=4, shared=74240,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 18],
    horneado={11: 1, 12: 1, 15: 1, 17: 1, 19: 1, 20: 64, 21: 128, 22: 128, 23: 8, 24: 128, 25: True},
    div16=[8, 9, 10, 13, 14, 16],
)

_PTX_7 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk06_mlp_down_kernel   // -- Begin function _sk06_mlp_down_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk06_mlp_down_kernel
.visible .entry _sk06_mlp_down_kernel(
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_6,
	.param .u32 _sk06_mlp_down_kernel_param_7,
	.param .u32 _sk06_mlp_down_kernel_param_8,
	.param .u32 _sk06_mlp_down_kernel_param_9,
	.param .u32 _sk06_mlp_down_kernel_param_10,
	.param .u32 _sk06_mlp_down_kernel_param_11,
	.param .u32 _sk06_mlp_down_kernel_param_12,
	.param .u32 _sk06_mlp_down_kernel_param_13,
	.param .u32 _sk06_mlp_down_kernel_param_14,
	.param .u32 _sk06_mlp_down_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_16,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_17
)
.reqntid 128
{
	.reg .pred 	%p<26>;
	.reg .b16 	%rs<129>;
	.reg .b32 	%r<888>;
	.reg .b64 	%rd<224>;
	.loc	1 304 0                         // sk06_mlp_down.py:304:0
$L__func_begin0:
	.loc	1 304 0                         // sk06_mlp_down.py:304:0

// %bb.0:
	ld.param.b32 	%r25, [_sk06_mlp_down_kernel_param_14];
	ld.param.b32 	%r24, [_sk06_mlp_down_kernel_param_13];
	ld.param.b32 	%r23, [_sk06_mlp_down_kernel_param_12];
	ld.param.b32 	%r22, [_sk06_mlp_down_kernel_param_9];
	ld.param.b32 	%r21, [_sk06_mlp_down_kernel_param_8];
	ld.param.b32 	%r20, [_sk06_mlp_down_kernel_param_7];
	ld.param.b64 	%rd33, [_sk06_mlp_down_kernel_param_4];
	ld.param.b64 	%rd32, [_sk06_mlp_down_kernel_param_3];
	ld.param.b64 	%rd31, [_sk06_mlp_down_kernel_param_2];
	ld.param.b64 	%rd30, [_sk06_mlp_down_kernel_param_1];
	ld.param.b64 	%rd29, [_sk06_mlp_down_kernel_param_0];
$L__tmp0:
	.loc	1 313 24                        // sk06_mlp_down.py:313:24
	mov.u32 	%r66, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk06_mlp_down.py:314:27 ]
	add.s32 	%r67, %r20, 63;
	.loc	2 43 30                         // standard.py:43:30 @[ sk06_mlp_down.py:314:27 ]
	shr.s32 	%r68, %r67, 31;
	shr.u32 	%r69, %r68, 26;
	add.s32 	%r70, %r67, %r69;
	shr.s32 	%r71, %r70, 6;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk06_mlp_down.py:315:27 ]
	add.s32 	%r72, %r21, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk06_mlp_down.py:315:27 ]
	shr.s32 	%r73, %r72, 31;
	shr.u32 	%r74, %r73, 25;
	add.s32 	%r75, %r72, %r74;
	shr.s32 	%r76, %r75, 7;
$L__tmp3:
	.loc	1 316 29                        // sk06_mlp_down.py:316:29
	shl.b32 	%r77, %r76, 3;
	.loc	1 317 22                        // sk06_mlp_down.py:317:22
	div.s32 	%r78, %r66, %r77;
	.loc	1 317 38                        // sk06_mlp_down.py:317:38
	shl.b32 	%r79, %r78, 3;
	.loc	1 318 30                        // sk06_mlp_down.py:318:30
	sub.s32 	%r80, %r71, %r79;
	ld.param.b32 	%r81, [_sk06_mlp_down_kernel_param_10];
	.loc	1 318 39                        // sk06_mlp_down.py:318:39
	min.s32 	%r82, %r80, 8;
	ld.param.b32 	%r83, [_sk06_mlp_down_kernel_param_11];
	.loc	1 319 30                        // sk06_mlp_down.py:319:30
	mul.lo.s32 	%r84, %r78, %r77;
	sub.s32 	%r85, %r66, %r84;
	.loc	1 320 36                        // sk06_mlp_down.py:320:36
	div.s32 	%r86, %r85, %r82;
	.loc	1 319 46                        // sk06_mlp_down.py:319:46
	mul.lo.s32 	%r87, %r86, %r82;
	sub.s32 	%r88, %r85, %r87;
	.loc	1 319 23                        // sk06_mlp_down.py:319:23
	add.s32 	%r89, %r88, %r79;
	.loc	1 322 22                        // sk06_mlp_down.py:322:22
	shl.b32 	%r1, %r89, 6;
	.loc	1 322 45                        // sk06_mlp_down.py:322:45
	mov.u32 	%r2, %tid.x;
	and.b32 	%r3, %r2, 120;
	bfe.u32 	%r90, %r2, 3, 4;
	or.b32 	%r91, %r90, 16;
	or.b32 	%r92, %r90, 32;
	or.b32 	%r93, %r90, 48;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r94, %r1, %r90;
	or.b32 	%r95, %r1, %r91;
	or.b32 	%r96, %r1, %r92;
	or.b32 	%r97, %r1, %r93;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r4, %r94, %r20;
	rem.s32 	%r5, %r95, %r20;
	rem.s32 	%r6, %r96, %r20;
	rem.s32 	%r7, %r97, %r20;
	.loc	1 323 22                        // sk06_mlp_down.py:323:22
	shl.b32 	%r8, %r86, 7;
	.loc	1 323 45                        // sk06_mlp_down.py:323:45
	and.b32 	%r9, %r2, 7;
	shl.b32 	%r98, %r9, 4;
	and.b32 	%r10, %r2, 15;
	and.b32 	%r11, %r2, 127;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r99, %r8, %r90;
	or.b32 	%r100, %r8, %r91;
	or.b32 	%r101, %r8, %r92;
	or.b32 	%r102, %r8, %r93;
	or.b32 	%r103, %r99, 64;
	or.b32 	%r104, %r99, 80;
	or.b32 	%r105, %r99, 96;
	or.b32 	%r106, %r99, 112;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r108, %r99, %r21;
	rem.s32 	%r109, %r100, %r21;
	rem.s32 	%r110, %r101, %r21;
	rem.s32 	%r111, %r102, %r21;
	rem.s32 	%r112, %r103, %r21;
	rem.s32 	%r113, %r104, %r21;
	rem.s32 	%r114, %r105, %r21;
	rem.s32 	%r115, %r106, %r21;
	.loc	1 326 39                        // sk06_mlp_down.py:326:39
	mul.lo.s32 	%r117, %r4, %r81;
	mul.lo.s32 	%r118, %r5, %r81;
	mul.lo.s32 	%r119, %r6, %r81;
	mul.lo.s32 	%r120, %r7, %r81;
	.loc	1 326 21                        // sk06_mlp_down.py:326:21
	cvt.s64.s32 	%rd1, %r117;
	add.s64 	%rd71, %rd29, %rd1;
	cvt.s64.s32 	%rd2, %r118;
	add.s64 	%rd72, %rd29, %rd2;
	cvt.s64.s32 	%rd3, %r119;
	add.s64 	%rd73, %rd29, %rd3;
	cvt.s64.s32 	%rd4, %r120;
	add.s64 	%rd74, %rd29, %rd4;
	.loc	1 326 51                        // sk06_mlp_down.py:326:51
	cvt.u64.u32 	%rd5, %r98;
	add.s64 	%rd34, %rd71, %rd5;
	add.s64 	%rd35, %rd72, %rd5;
	add.s64 	%rd36, %rd73, %rd5;
	add.s64 	%rd37, %rd74, %rd5;
	.loc	1 327 21                        // sk06_mlp_down.py:327:21
	add.s64 	%rd75, %rd30, %rd5;
	.loc	1 327 69                        // sk06_mlp_down.py:327:69
	mul.lo.s32 	%r121, %r108, %r83;
	mul.lo.s32 	%r122, %r109, %r83;
	mul.lo.s32 	%r123, %r110, %r83;
	mul.lo.s32 	%r124, %r111, %r83;
	mul.lo.s32 	%r125, %r112, %r83;
	mul.lo.s32 	%r126, %r113, %r83;
	mul.lo.s32 	%r127, %r114, %r83;
	mul.lo.s32 	%r128, %r115, %r83;
	.loc	1 327 51                        // sk06_mlp_down.py:327:51
	cvt.s64.s32 	%rd6, %r121;
	add.s64 	%rd38, %rd75, %rd6;
	cvt.s64.s32 	%rd7, %r122;
	add.s64 	%rd39, %rd75, %rd7;
	cvt.s64.s32 	%rd8, %r123;
	add.s64 	%rd40, %rd75, %rd8;
	cvt.s64.s32 	%rd9, %r124;
	add.s64 	%rd41, %rd75, %rd9;
	cvt.s64.s32 	%rd10, %r125;
	add.s64 	%rd42, %rd75, %rd10;
	cvt.s64.s32 	%rd11, %r126;
	add.s64 	%rd43, %rd75, %rd11;
	cvt.s64.s32 	%rd12, %r127;
	add.s64 	%rd44, %rd75, %rd12;
	cvt.s64.s32 	%rd13, %r128;
	add.s64 	%rd45, %rd75, %rd13;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	setp.gt.s32 	%p1, %r22, 127;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	shl.b32 	%r133, %r11, 4;
	and.b32 	%r12, %r2, 56;
	shl.b32 	%r134, %r12, 1;
	xor.b32 	%r135, %r133, %r134;
	mov.b32 	%r136, global_smem;
	add.s32 	%r32, %r136, %r135;
	add.s32 	%r27, %r32, 49152;
	selp.b32 	%r28, 16, 0, %p1;
	// begin inline asm
	cp.async.cg.shared.global [ %r27 + 0 ], [ %rd34 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r29, %r32, 51200;
	// begin inline asm
	cp.async.cg.shared.global [ %r29 + 0 ], [ %rd35 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r30, %r32, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r30 + 0 ], [ %rd36 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r31, %r32, 55296;
	// begin inline asm
	cp.async.cg.shared.global [ %r31 + 0 ], [ %rd37 + 0 ], 0x10, %r28;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	// begin inline asm
	cp.async.cg.shared.global [ %r32 + 0 ], [ %rd38 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r33, %r32, 2048;
	// begin inline asm
	cp.async.cg.shared.global [ %r33 + 0 ], [ %rd39 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r34, %r32, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r34 + 0 ], [ %rd40 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r35, %r32, 6144;
	// begin inline asm
	cp.async.cg.shared.global [ %r35 + 0 ], [ %rd41 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r36, %r32, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r36 + 0 ], [ %rd42 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r37, %r32, 10240;
	// begin inline asm
	cp.async.cg.shared.global [ %r37 + 0 ], [ %rd43 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r38, %r32, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r38 + 0 ], [ %rd44 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r39, %r32, 14336;
	// begin inline asm
	cp.async.cg.shared.global [ %r39 + 0 ], [ %rd45 + 0 ], 0x10, %r28;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	setp.gt.s32 	%p2, %r22, 255;
	.loc	1 344 18                        // sk06_mlp_down.py:344:18
	add.s64 	%rd46, %rd34, 128;
	add.s64 	%rd47, %rd35, 128;
	add.s64 	%rd48, %rd36, 128;
	add.s64 	%rd49, %rd37, 128;
	.loc	1 345 18                        // sk06_mlp_down.py:345:18
	add.s64 	%rd50, %rd38, 128;
	add.s64 	%rd51, %rd39, 128;
	add.s64 	%rd52, %rd40, 128;
	add.s64 	%rd53, %rd41, 128;
	add.s64 	%rd54, %rd42, 128;
	add.s64 	%rd55, %rd43, 128;
	add.s64 	%rd56, %rd44, 128;
	add.s64 	%rd57, %rd45, 128;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	bar.sync 	0;
	add.s32 	%r40, %r32, 57344;
	selp.b32 	%r41, 16, 0, %p2;
	// begin inline asm
	cp.async.cg.shared.global [ %r40 + 0 ], [ %rd46 + 0 ], 0x10, %r41;
	// end inline asm
	add.s32 	%r42, %r32, 59392;
	// begin inline asm
	cp.async.cg.shared.global [ %r42 + 0 ], [ %rd47 + 0 ], 0x10, %r41;
	// end inline asm
	add.s32 	%r43, %r32, 61440;
	// begin inline asm
	cp.async.cg.shared.global [ %r43 + 0 ], [ %rd48 + 0 ], 0x10, %r41;
	// end inline asm
	add.s32 	%r44, %r32, 63488;
	// begin inline asm
	cp.async.cg.shared.global [ %r44 + 0 ], [ %rd49 + 0 ], 0x10, %r41;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	add.s32 	%r45, %r32, 16384;
	// begin inline asm
	cp.async.cg.shared.global [ %r45 + 0 ], [ %rd50 + 0 ], 0x10, %r41;
	// end inline asm
	add.s32 	%r46, %r32, 18432;
	// begin inline asm
	cp.async.cg.shared.global [ %r46 + 0 ], [ %rd51 + 0 ], 0x10, %r41;
	// end inline asm
	add.s32 	%r47, %r32, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r47 + 0 ], [ %rd52 + 0 ], 0x10, %r41;
	// end inline asm
	add.s32 	%r48, %r32, 22528;
	// begin inline asm
	cp.async.cg.shared.global [ %r48 + 0 ], [ %rd53 + 0 ], 0x10, %r41;
	// end inline asm
	add.s32 	%r49, %r32, 24576;
	// begin inline asm
	cp.async.cg.shared.global [ %r49 + 0 ], [ %rd54 + 0 ], 0x10, %r41;
	// end inline asm
	add.s32 	%r50, %r32, 26624;
	// begin inline asm
	cp.async.cg.shared.global [ %r50 + 0 ], [ %rd55 + 0 ], 0x10, %r41;
	// end inline asm
	add.s32 	%r51, %r32, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r51 + 0 ], [ %rd56 + 0 ], 0x10, %r41;
	// end inline asm
	add.s32 	%r52, %r32, 30720;
	// begin inline asm
	cp.async.cg.shared.global [ %r52 + 0 ], [ %rd57 + 0 ], 0x10, %r41;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	setp.gt.s32 	%p3, %r22, 383;
	.loc	1 344 18                        // sk06_mlp_down.py:344:18
	add.s64 	%rd58, %rd34, 256;
	add.s64 	%rd59, %rd35, 256;
	add.s64 	%rd60, %rd36, 256;
	add.s64 	%rd61, %rd37, 256;
	.loc	1 345 18                        // sk06_mlp_down.py:345:18
	add.s64 	%rd62, %rd38, 256;
	add.s64 	%rd63, %rd39, 256;
	add.s64 	%rd64, %rd40, 256;
	add.s64 	%rd65, %rd41, 256;
	add.s64 	%rd66, %rd42, 256;
	add.s64 	%rd67, %rd43, 256;
	add.s64 	%rd68, %rd44, 256;
	add.s64 	%rd69, %rd45, 256;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	bar.sync 	0;
	add.s32 	%r53, %r32, 65536;
	selp.b32 	%r54, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r53 + 0 ], [ %rd58 + 0 ], 0x10, %r54;
	// end inline asm
	add.s32 	%r55, %r32, 67584;
	// begin inline asm
	cp.async.cg.shared.global [ %r55 + 0 ], [ %rd59 + 0 ], 0x10, %r54;
	// end inline asm
	add.s32 	%r56, %r32, 69632;
	// begin inline asm
	cp.async.cg.shared.global [ %r56 + 0 ], [ %rd60 + 0 ], 0x10, %r54;
	// end inline asm
	add.s32 	%r57, %r32, 71680;
	// begin inline asm
	cp.async.cg.shared.global [ %r57 + 0 ], [ %rd61 + 0 ], 0x10, %r54;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	add.s32 	%r58, %r32, 32768;
	// begin inline asm
	cp.async.cg.shared.global [ %r58 + 0 ], [ %rd62 + 0 ], 0x10, %r54;
	// end inline asm
	add.s32 	%r59, %r32, 34816;
	// begin inline asm
	cp.async.cg.shared.global [ %r59 + 0 ], [ %rd63 + 0 ], 0x10, %r54;
	// end inline asm
	add.s32 	%r60, %r32, 36864;
	// begin inline asm
	cp.async.cg.shared.global [ %r60 + 0 ], [ %rd64 + 0 ], 0x10, %r54;
	// end inline asm
	add.s32 	%r61, %r32, 38912;
	// begin inline asm
	cp.async.cg.shared.global [ %r61 + 0 ], [ %rd65 + 0 ], 0x10, %r54;
	// end inline asm
	add.s32 	%r62, %r32, 40960;
	// begin inline asm
	cp.async.cg.shared.global [ %r62 + 0 ], [ %rd66 + 0 ], 0x10, %r54;
	// end inline asm
	add.s32 	%r63, %r32, 43008;
	// begin inline asm
	cp.async.cg.shared.global [ %r63 + 0 ], [ %rd67 + 0 ], 0x10, %r54;
	// end inline asm
	add.s32 	%r64, %r32, 45056;
	// begin inline asm
	cp.async.cg.shared.global [ %r64 + 0 ], [ %rd68 + 0 ], 0x10, %r54;
	// end inline asm
	add.s32 	%r65, %r32, 47104;
	// begin inline asm
	cp.async.cg.shared.global [ %r65 + 0 ], [ %rd69 + 0 ], 0x10, %r54;
	// end inline asm
	cp.async.commit_group;
	cvt.u32.u64 	%r817, %rd5;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	@%p1 bra 	$L__BB0_2;
	bra.uni 	$L__BB0_1;
$L__BB0_2:                              // %.lr.ph
	.loc	1 0 23                          // sk06_mlp_down.py:0:23
	ld.param.b32 	%r26, [_sk06_mlp_down_kernel_param_15];
	ld.param.b64 	%rd70, [_sk06_mlp_down_kernel_param_6];
	or.b32 	%r107, %r8, %r11;
	rem.s32 	%r116, %r107, %r21;
	shr.s32 	%r129, %r116, 31;
	shr.u32 	%r130, %r129, 25;
	add.s32 	%r131, %r116, %r130;
	shr.s32 	%r132, %r131, 7;
	mad.wide.s32 	%rd14, %r132, 4, %rd70;
	.loc	1 337 28                        // sk06_mlp_down.py:337:28
	shr.u32 	%r137, %r22, 7;
	add.s32 	%r138, %r137, -3;
	shl.b32 	%r139, %r10, 7;
	and.b32 	%r823, %r2, 16;
	or.b32 	%r140, %r139, %r817;
	xor.b32 	%r13, %r140, %r823;
	xor.b32 	%r14, %r13, 32;
	xor.b32 	%r15, %r13, 64;
	xor.b32 	%r16, %r13, 96;
	shl.b32 	%r141, %r9, 7;
	and.b32 	%r821, %r2, 96;
	shl.b32 	%r142, %r821, 5;
	shl.b32 	%r143, %r2, 1;
	and.b32 	%r144, %r143, 48;
	or.b32 	%r145, %r141, %r142;
	xor.b32 	%r146, %r817, %r144;
	or.b32 	%r17, %r145, %r146;
	xor.b32 	%r18, %r17, 64;
	shl.b32 	%r147, %r11, 2;
	add.s32 	%r148, %r136, %r147;
	add.s32 	%r314, %r148, 73728;
	shl.b32 	%r822, %r2, 3;
	and.b32 	%r149, %r822, 24;
	add.s32 	%r150, %r136, %r149;
	add.s32 	%r19, %r150, %r821;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	cvt.s64.s32 	%rd15, %r138;
	and.b32 	%r151, %r22, -128;
	cvt.u64.u32 	%rd16, %r151;
	add.s64 	%rd76, %rd5, %rd13;
	add.s64 	%rd77, %rd76, %rd30;
	add.s64 	%rd17, %rd77, 384;
	add.s64 	%rd78, %rd5, %rd12;
	add.s64 	%rd79, %rd78, %rd30;
	add.s64 	%rd18, %rd79, 384;
	add.s64 	%rd80, %rd5, %rd11;
	add.s64 	%rd81, %rd80, %rd30;
	add.s64 	%rd19, %rd81, 384;
	add.s64 	%rd82, %rd5, %rd10;
	add.s64 	%rd83, %rd82, %rd30;
	add.s64 	%rd20, %rd83, 384;
	add.s64 	%rd84, %rd5, %rd9;
	add.s64 	%rd85, %rd84, %rd30;
	add.s64 	%rd21, %rd85, 384;
	add.s64 	%rd86, %rd5, %rd8;
	add.s64 	%rd87, %rd86, %rd30;
	add.s64 	%rd22, %rd87, 384;
	add.s64 	%rd88, %rd5, %rd7;
	add.s64 	%rd89, %rd88, %rd30;
	add.s64 	%rd23, %rd89, 384;
	add.s64 	%rd90, %rd5, %rd6;
	add.s64 	%rd91, %rd90, %rd30;
	add.s64 	%rd24, %rd91, 384;
	add.s64 	%rd92, %rd5, %rd4;
	add.s64 	%rd93, %rd92, %rd29;
	add.s64 	%rd25, %rd93, 384;
	add.s64 	%rd94, %rd5, %rd3;
	add.s64 	%rd95, %rd94, %rd29;
	add.s64 	%rd26, %rd95, 384;
	add.s64 	%rd96, %rd5, %rd2;
	add.s64 	%rd97, %rd96, %rd29;
	add.s64 	%rd27, %rd97, 384;
	add.s64 	%rd98, %rd5, %rd1;
	add.s64 	%rd99, %rd98, %rd29;
	add.s64 	%rd28, %rd99, 384;
	mov.b32 	%r824, 0f00000000;
	mov.b32 	%r820, 2;
	mov.b32 	%r819, -1;
	mov.b64 	%rd222, 0;
	mov.b32 	%r152, 0;
	mov.b32 	%r818, %r152;
	mov.b64 	%rd223, %rd222;
	mov.b32 	%r825, %r824;
	mov.b32 	%r826, %r824;
	mov.b32 	%r827, %r824;
	mov.b32 	%r828, %r824;
	mov.b32 	%r829, %r824;
	mov.b32 	%r830, %r824;
	mov.b32 	%r831, %r824;
	mov.b32 	%r832, %r824;
	mov.b32 	%r833, %r824;
	mov.b32 	%r834, %r824;
	mov.b32 	%r835, %r824;
	mov.b32 	%r836, %r824;
	mov.b32 	%r837, %r824;
	mov.b32 	%r838, %r824;
	mov.b32 	%r839, %r824;
	mov.b32 	%r840, %r824;
	mov.b32 	%r841, %r824;
	mov.b32 	%r842, %r824;
	mov.b32 	%r843, %r824;
	mov.b32 	%r844, %r824;
	mov.b32 	%r845, %r824;
	mov.b32 	%r846, %r824;
	mov.b32 	%r847, %r824;
	mov.b32 	%r848, %r824;
	mov.b32 	%r849, %r824;
	mov.b32 	%r850, %r824;
	mov.b32 	%r851, %r824;
	mov.b32 	%r852, %r824;
	mov.b32 	%r853, %r824;
	mov.b32 	%r854, %r824;
	mov.b32 	%r855, %r824;
	mov.b32 	%r856, %r824;
	mov.b32 	%r857, %r824;
	mov.b32 	%r858, %r824;
	mov.b32 	%r859, %r824;
	mov.b32 	%r860, %r824;
	mov.b32 	%r861, %r824;
	mov.b32 	%r862, %r824;
	mov.b32 	%r863, %r824;
	mov.b32 	%r864, %r824;
	mov.b32 	%r865, %r824;
	mov.b32 	%r866, %r824;
	mov.b32 	%r867, %r824;
	mov.b32 	%r868, %r824;
	mov.b32 	%r869, %r824;
	mov.b32 	%r870, %r824;
	mov.b32 	%r871, %r824;
	mov.b32 	%r872, %r824;
	mov.b32 	%r873, %r824;
	mov.b32 	%r874, %r824;
	mov.b32 	%r875, %r824;
	mov.b32 	%r876, %r824;
	mov.b32 	%r877, %r824;
	mov.b32 	%r878, %r824;
	mov.b32 	%r879, %r824;
	mov.b32 	%r880, %r824;
	mov.b32 	%r881, %r824;
	mov.b32 	%r882, %r824;
	mov.b32 	%r883, %r824;
	mov.b32 	%r884, %r824;
	mov.b32 	%r885, %r824;
	mov.b32 	%r886, %r824;
	mov.b32 	%r887, %r824;
$L__BB0_3:                              // =>This Inner Loop Header: Depth=1
	setp.lt.s64 	%p4, %rd223, %rd15;
	add.s32 	%r328, %r819, 1;
	setp.gt.s32 	%p5, %r328, 2;
	selp.b32 	%r819, 0, %r328, %p5;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	cp.async.wait_group 	4;
	bar.sync 	0;
	shl.b32 	%r329, %r819, 13;
	add.s32 	%r330, %r136, %r329;
	add.s32 	%r331, %r330, %r13;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r153, %r154, %r155, %r156}, [%r331+49152];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r165, %r166, %r167, %r168}, [%r331+51200];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r169, %r170, %r171, %r172}, [%r331+53248];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r173, %r174, %r175, %r176}, [%r331+55296];
	add.s32 	%r332, %r330, %r14;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r181, %r182, %r183, %r184}, [%r332+49152];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r209, %r210, %r211, %r212}, [%r332+51200];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r229, %r230, %r231, %r232}, [%r332+53248];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r249, %r250, %r251, %r252}, [%r332+55296];
	add.s32 	%r333, %r330, %r15;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r265, %r266, %r267, %r268}, [%r333+49152];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r277, %r278, %r279, %r280}, [%r333+51200];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r281, %r282, %r283, %r284}, [%r333+53248];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r285, %r286, %r287, %r288}, [%r333+55296];
	add.s32 	%r334, %r330, %r16;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r289, %r290, %r291, %r292}, [%r334+49152];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r301, %r302, %r303, %r304}, [%r334+51200];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r305, %r306, %r307, %r308}, [%r334+53248];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r309, %r310, %r311, %r312}, [%r334+55296];
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	add.s32 	%r335, %r330, %r329;
	add.s32 	%r336, %r335, %r17;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r157, %r158, %r185, %r186}, [%r336];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r159, %r160, %r191, %r192}, [%r336+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r161, %r162, %r197, %r198}, [%r336+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r163, %r164, %r203, %r204}, [%r336+12288];
	add.s32 	%r337, %r335, %r18;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r269, %r270, %r293, %r294}, [%r337];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r271, %r272, %r295, %r296}, [%r337+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r273, %r274, %r297, %r298}, [%r337+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r275, %r276, %r299, %r300}, [%r337+12288];
	.loc	1 338 36                        // sk06_mlp_down.py:338:36
	mov.b32 	%r177, %r152;
	mov.b32 	%r178, %r152;
	mov.b32 	%r179, %r152;
	mov.b32 	%r180, %r152;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r177, %r178, %r179, %r180 }, { %r153, %r154, %r155, %r156 }, { %r157, %r158 }, { %r177, %r178, %r179, %r180 };
	// end inline asm
	mov.b32 	%r187, %r152;
	mov.b32 	%r188, %r152;
	mov.b32 	%r189, %r152;
	mov.b32 	%r190, %r152;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r187, %r188, %r189, %r190 }, { %r153, %r154, %r155, %r156 }, { %r159, %r160 }, { %r187, %r188, %r189, %r190 };
	// end inline asm
	mov.b32 	%r193, %r152;
	mov.b32 	%r194, %r152;
	mov.b32 	%r195, %r152;
	mov.b32 	%r196, %r152;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r193, %r194, %r195, %r196 }, { %r153, %r154, %r155, %r156 }, { %r161, %r162 }, { %r193, %r194, %r195, %r196 };
	// end inline asm
	mov.b32 	%r199, %r152;
	mov.b32 	%r200, %r152;
	mov.b32 	%r201, %r152;
	mov.b32 	%r202, %r152;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r199, %r200, %r201, %r202 }, { %r153, %r154, %r155, %r156 }, { %r163, %r164 }, { %r199, %r200, %r201, %r202 };
	// end inline asm
	mov.b32 	%r205, %r152;
	mov.b32 	%r206, %r152;
	mov.b32 	%r207, %r152;
	mov.b32 	%r208, %r152;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r205, %r206, %r207, %r208 }, { %r165, %r166, %r167, %r168 }, { %r157, %r158 }, { %r205, %r206, %r207, %r208 };
	// end inline asm
	mov.b32 	%r213, %r152;
	mov.b32 	%r214, %r152;
	mov.b32 	%r215, %r152;
	mov.b32 	%r216, %r152;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r213, %r214, %r215, %r216 }, { %r165, %r166, %r167, %r168 }, { %r159, %r160 }, { %r213, %r214, %r215, %r216 };
	// end inline asm
	mov.b32 	%r217, %r152;
	mov.b32 	%r218, %r152;
	mov.b32 	%r219, %r152;
	mov.b32 	%r220, %r152;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r217, %r218, %r219, %r220 }, { %r165, %r166, %r167, %r168 }, { %r161, %r162 }, { %r217, %r218, %r219, %r220 };
	// end inline asm
	mov.b32 	%r221, %r152;
	mov.b32 	%r222, %r152;
	mov.b32 	%r223, %r152;
	mov.b32 	%r224, %r152;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r221, %r222, %r223, %r224 }, { %r165, %r166, %r167, %r168 }, { %r163, %r164 }, { %r221, %r222, %r223, %r224 };
	// end inline asm
	mov.b32 	%r225, %r152;
	mov.b32 	%r226, %r152;
	mov.b32 	%r227, %r152;
	mov.b32 	%r228, %r152;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r225, %r226, %r227, %r228 }, { %r169, %r170, %r171, %r172 }, { %r157, %r158 }, { %r225, %r226, %r227, %r228 };
	// end inline asm
	mov.b32 	%r233, %r152;
	mov.b32 	%r234, %r152;
	mov.b32 	%r235, %r152;
	mov.b32 	%r236, %r152;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r233, %r234, %r235, %r236 }, { %r169, %r170, %r171, %r172 }, { %r159, %r160 }, { %r233, %r234, %r235, %r236 };
	// end inline asm
	mov.b32 	%r237, %r152;
	mov.b32 	%r238, %r152;
	mov.b32 	%r239, %r152;
	mov.b32 	%r240, %r152;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r237, %r238, %r239, %r240 }, { %r169, %r170, %r171, %r172 }, { %r161, %r162 }, { %r237, %r238, %r239, %r240 };
	// end inline asm
	mov.b32 	%r241, %r152;
	mov.b32 	%r242, %r152;
	mov.b32 	%r243, %r152;
	mov.b32 	%r244, %r152;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r241, %r242, %r243, %r244 }, { %r169, %r170, %r171, %r172 }, { %r163, %r164 }, { %r241, %r242, %r243, %r244 };
	// end inline asm
	mov.b32 	%r245, %r152;
	mov.b32 	%r246, %r152;
	mov.b32 	%r247, %r152;
	mov.b32 	%r248, %r152;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r245, %r246, %r247, %r248 }, { %r173, %r174, %r175, %r176 }, { %r157, %r158 }, { %r245, %r246, %r247, %r248 };
	// end inline asm
	mov.b32 	%r253, %r152;
	mov.b32 	%r254, %r152;
	mov.b32 	%r255, %r152;
	mov.b32 	%r256, %r152;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r253, %r254, %r255, %r256 }, { %r173, %r174, %r175, %r176 }, { %r159, %r160 }, { %r253, %r254, %r255, %r256 };
	// end inline asm
	mov.b32 	%r257, %r152;
	mov.b32 	%r258, %r152;
	mov.b32 	%r259, %r152;
	mov.b32 	%r260, %r152;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r257, %r258, %r259, %r260 }, { %r173, %r174, %r175, %r176 }, { %r161, %r162 }, { %r257, %r258, %r259, %r260 };
	// end inline asm
	mov.b32 	%r264, %r152;
	mov.b32 	%r261, %r152;
	mov.b32 	%r262, %r152;
	mov.b32 	%r263, %r152;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r261, %r262, %r263, %r264 }, { %r173, %r174, %r175, %r176 }, { %r163, %r164 }, { %r261, %r262, %r263, %r264 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r177, %r178, %r179, %r180 }, { %r181, %r182, %r183, %r184 }, { %r185, %r186 }, { %r177, %r178, %r179, %r180 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r187, %r188, %r189, %r190 }, { %r181, %r182, %r183, %r184 }, { %r191, %r192 }, { %r187, %r188, %r189, %r190 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r193, %r194, %r195, %r196 }, { %r181, %r182, %r183, %r184 }, { %r197, %r198 }, { %r193, %r194, %r195, %r196 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r199, %r200, %r201, %r202 }, { %r181, %r182, %r183, %r184 }, { %r203, %r204 }, { %r199, %r200, %r201, %r202 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r205, %r206, %r207, %r208 }, { %r209, %r210, %r211, %r212 }, { %r185, %r186 }, { %r205, %r206, %r207, %r208 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r213, %r214, %r215, %r216 }, { %r209, %r210, %r211, %r212 }, { %r191, %r192 }, { %r213, %r214, %r215, %r216 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r217, %r218, %r219, %r220 }, { %r209, %r210, %r211, %r212 }, { %r197, %r198 }, { %r217, %r218, %r219, %r220 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r221, %r222, %r223, %r224 }, { %r209, %r210, %r211, %r212 }, { %r203, %r204 }, { %r221, %r222, %r223, %r224 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r225, %r226, %r227, %r228 }, { %r229, %r230, %r231, %r232 }, { %r185, %r186 }, { %r225, %r226, %r227, %r228 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r233, %r234, %r235, %r236 }, { %r229, %r230, %r231, %r232 }, { %r191, %r192 }, { %r233, %r234, %r235, %r236 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r237, %r238, %r239, %r240 }, { %r229, %r230, %r231, %r232 }, { %r197, %r198 }, { %r237, %r238, %r239, %r240 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r241, %r242, %r243, %r244 }, { %r229, %r230, %r231, %r232 }, { %r203, %r204 }, { %r241, %r242, %r243, %r244 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r245, %r246, %r247, %r248 }, { %r249, %r250, %r251, %r252 }, { %r185, %r186 }, { %r245, %r246, %r247, %r248 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r253, %r254, %r255, %r256 }, { %r249, %r250, %r251, %r252 }, { %r191, %r192 }, { %r253, %r254, %r255, %r256 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r257, %r258, %r259, %r260 }, { %r249, %r250, %r251, %r252 }, { %r197, %r198 }, { %r257, %r258, %r259, %r260 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r261, %r262, %r263, %r264 }, { %r249, %r250, %r251, %r252 }, { %r203, %r204 }, { %r261, %r262, %r263, %r264 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r177, %r178, %r179, %r180 }, { %r265, %r266, %r267, %r268 }, { %r269, %r270 }, { %r177, %r178, %r179, %r180 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r187, %r188, %r189, %r190 }, { %r265, %r266, %r267, %r268 }, { %r271, %r272 }, { %r187, %r188, %r189, %r190 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r193, %r194, %r195, %r196 }, { %r265, %r266, %r267, %r268 }, { %r273, %r274 }, { %r193, %r194, %r195, %r196 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r199, %r200, %r201, %r202 }, { %r265, %r266, %r267, %r268 }, { %r275, %r276 }, { %r199, %r200, %r201, %r202 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r205, %r206, %r207, %r208 }, { %r277, %r278, %r279, %r280 }, { %r269, %r270 }, { %r205, %r206, %r207, %r208 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r213, %r214, %r215, %r216 }, { %r277, %r278, %r279, %r280 }, { %r271, %r272 }, { %r213, %r214, %r215, %r216 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r217, %r218, %r219, %r220 }, { %r277, %r278, %r279, %r280 }, { %r273, %r274 }, { %r217, %r218, %r219, %r220 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r221, %r222, %r223, %r224 }, { %r277, %r278, %r279, %r280 }, { %r275, %r276 }, { %r221, %r222, %r223, %r224 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r225, %r226, %r227, %r228 }, { %r281, %r282, %r283, %r284 }, { %r269, %r270 }, { %r225, %r226, %r227, %r228 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r233, %r234, %r235, %r236 }, { %r281, %r282, %r283, %r284 }, { %r271, %r272 }, { %r233, %r234, %r235, %r236 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r237, %r238, %r239, %r240 }, { %r281, %r282, %r283, %r284 }, { %r273, %r274 }, { %r237, %r238, %r239, %r240 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r241, %r242, %r243, %r244 }, { %r281, %r282, %r283, %r284 }, { %r275, %r276 }, { %r241, %r242, %r243, %r244 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r245, %r246, %r247, %r248 }, { %r285, %r286, %r287, %r288 }, { %r269, %r270 }, { %r245, %r246, %r247, %r248 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r253, %r254, %r255, %r256 }, { %r285, %r286, %r287, %r288 }, { %r271, %r272 }, { %r253, %r254, %r255, %r256 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r257, %r258, %r259, %r260 }, { %r285, %r286, %r287, %r288 }, { %r273, %r274 }, { %r257, %r258, %r259, %r260 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r261, %r262, %r263, %r264 }, { %r285, %r286, %r287, %r288 }, { %r275, %r276 }, { %r261, %r262, %r263, %r264 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r177, %r178, %r179, %r180 }, { %r289, %r290, %r291, %r292 }, { %r293, %r294 }, { %r177, %r178, %r179, %r180 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r187, %r188, %r189, %r190 }, { %r289, %r290, %r291, %r292 }, { %r295, %r296 }, { %r187, %r188, %r189, %r190 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r193, %r194, %r195, %r196 }, { %r289, %r290, %r291, %r292 }, { %r297, %r298 }, { %r193, %r194, %r195, %r196 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r199, %r200, %r201, %r202 }, { %r289, %r290, %r291, %r292 }, { %r299, %r300 }, { %r199, %r200, %r201, %r202 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r205, %r206, %r207, %r208 }, { %r301, %r302, %r303, %r304 }, { %r293, %r294 }, { %r205, %r206, %r207, %r208 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r213, %r214, %r215, %r216 }, { %r301, %r302, %r303, %r304 }, { %r295, %r296 }, { %r213, %r214, %r215, %r216 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r217, %r218, %r219, %r220 }, { %r301, %r302, %r303, %r304 }, { %r297, %r298 }, { %r217, %r218, %r219, %r220 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r221, %r222, %r223, %r224 }, { %r301, %r302, %r303, %r304 }, { %r299, %r300 }, { %r221, %r222, %r223, %r224 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r225, %r226, %r227, %r228 }, { %r305, %r306, %r307, %r308 }, { %r293, %r294 }, { %r225, %r226, %r227, %r228 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r233, %r234, %r235, %r236 }, { %r305, %r306, %r307, %r308 }, { %r295, %r296 }, { %r233, %r234, %r235, %r236 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r237, %r238, %r239, %r240 }, { %r305, %r306, %r307, %r308 }, { %r297, %r298 }, { %r237, %r238, %r239, %r240 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r241, %r242, %r243, %r244 }, { %r305, %r306, %r307, %r308 }, { %r299, %r300 }, { %r241, %r242, %r243, %r244 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r245, %r246, %r247, %r248 }, { %r309, %r310, %r311, %r312 }, { %r293, %r294 }, { %r245, %r246, %r247, %r248 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r253, %r254, %r255, %r256 }, { %r309, %r310, %r311, %r312 }, { %r295, %r296 }, { %r253, %r254, %r255, %r256 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r257, %r258, %r259, %r260 }, { %r309, %r310, %r311, %r312 }, { %r297, %r298 }, { %r257, %r258, %r259, %r260 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r261, %r262, %r263, %r264 }, { %r309, %r310, %r311, %r312 }, { %r299, %r300 }, { %r261, %r262, %r263, %r264 };
	// end inline asm
	.loc	1 340 37                        // sk06_mlp_down.py:340:37
	mad.wide.s32 	%rd100, %r818, 4, %rd14;
	.loc	1 340 27                        // sk06_mlp_down.py:340:27
	// begin inline asm
	mov.u32 %r313, 0x0;
	ld.global.b32 { %r313 }, [ %rd100 + 0 ];
	// end inline asm
	.loc	1 341 23                        // sk06_mlp_down.py:341:23
	// begin inline asm
	st.shared.b32 [ %r314 + 0 ], %r313;
	// end inline asm
	bar.sync 	0;
	ld.shared.v2.b32 	{%r338, %r339}, [%r19+73728];
	ld.shared.v2.b32 	{%r340, %r341}, [%r19+73856];
	ld.shared.v2.b32 	{%r342, %r343}, [%r19+73984];
	ld.shared.v2.b32 	{%r344, %r345}, [%r19+74112];
	cvt.rn.f32.s32 	%r346, %r177;
	cvt.rn.f32.s32 	%r347, %r178;
	cvt.rn.f32.s32 	%r348, %r179;
	cvt.rn.f32.s32 	%r349, %r180;
	cvt.rn.f32.s32 	%r350, %r187;
	cvt.rn.f32.s32 	%r351, %r188;
	cvt.rn.f32.s32 	%r352, %r189;
	cvt.rn.f32.s32 	%r353, %r190;
	cvt.rn.f32.s32 	%r354, %r193;
	cvt.rn.f32.s32 	%r355, %r194;
	cvt.rn.f32.s32 	%r356, %r195;
	cvt.rn.f32.s32 	%r357, %r196;
	cvt.rn.f32.s32 	%r358, %r199;
	cvt.rn.f32.s32 	%r359, %r200;
	cvt.rn.f32.s32 	%r360, %r201;
	cvt.rn.f32.s32 	%r361, %r202;
	cvt.rn.f32.s32 	%r362, %r205;
	cvt.rn.f32.s32 	%r363, %r206;
	cvt.rn.f32.s32 	%r364, %r207;
	cvt.rn.f32.s32 	%r365, %r208;
	cvt.rn.f32.s32 	%r366, %r213;
	cvt.rn.f32.s32 	%r367, %r214;
	cvt.rn.f32.s32 	%r368, %r215;
	cvt.rn.f32.s32 	%r369, %r216;
	cvt.rn.f32.s32 	%r370, %r217;
	cvt.rn.f32.s32 	%r371, %r218;
	cvt.rn.f32.s32 	%r372, %r219;
	cvt.rn.f32.s32 	%r373, %r220;
	cvt.rn.f32.s32 	%r374, %r221;
	cvt.rn.f32.s32 	%r375, %r222;
	cvt.rn.f32.s32 	%r376, %r223;
	cvt.rn.f32.s32 	%r377, %r224;
	cvt.rn.f32.s32 	%r378, %r225;
	cvt.rn.f32.s32 	%r379, %r226;
	cvt.rn.f32.s32 	%r380, %r227;
	cvt.rn.f32.s32 	%r381, %r228;
	cvt.rn.f32.s32 	%r382, %r233;
	cvt.rn.f32.s32 	%r383, %r234;
	cvt.rn.f32.s32 	%r384, %r235;
	cvt.rn.f32.s32 	%r385, %r236;
	cvt.rn.f32.s32 	%r386, %r237;
	cvt.rn.f32.s32 	%r387, %r238;
	cvt.rn.f32.s32 	%r388, %r239;
	cvt.rn.f32.s32 	%r389, %r240;
	cvt.rn.f32.s32 	%r390, %r241;
	cvt.rn.f32.s32 	%r391, %r242;
	cvt.rn.f32.s32 	%r392, %r243;
	cvt.rn.f32.s32 	%r393, %r244;
	cvt.rn.f32.s32 	%r394, %r245;
	cvt.rn.f32.s32 	%r395, %r246;
	cvt.rn.f32.s32 	%r396, %r247;
	cvt.rn.f32.s32 	%r397, %r248;
	cvt.rn.f32.s32 	%r398, %r253;
	cvt.rn.f32.s32 	%r399, %r254;
	cvt.rn.f32.s32 	%r400, %r255;
	cvt.rn.f32.s32 	%r401, %r256;
	cvt.rn.f32.s32 	%r402, %r257;
	cvt.rn.f32.s32 	%r403, %r258;
	cvt.rn.f32.s32 	%r404, %r259;
	cvt.rn.f32.s32 	%r405, %r260;
	cvt.rn.f32.s32 	%r406, %r261;
	cvt.rn.f32.s32 	%r407, %r262;
	cvt.rn.f32.s32 	%r408, %r263;
	cvt.rn.f32.s32 	%r409, %r264;
	.loc	1 341 19                        // sk06_mlp_down.py:341:19
	fma.rn.f32 	%r887, %r345, %r409, %r887;
	fma.rn.f32 	%r886, %r344, %r408, %r886;
	fma.rn.f32 	%r885, %r345, %r407, %r885;
	fma.rn.f32 	%r884, %r344, %r406, %r884;
	fma.rn.f32 	%r883, %r343, %r405, %r883;
	fma.rn.f32 	%r882, %r342, %r404, %r882;
	fma.rn.f32 	%r881, %r343, %r403, %r881;
	fma.rn.f32 	%r880, %r342, %r402, %r880;
	fma.rn.f32 	%r879, %r341, %r401, %r879;
	fma.rn.f32 	%r878, %r340, %r400, %r878;
	fma.rn.f32 	%r877, %r341, %r399, %r877;
	fma.rn.f32 	%r876, %r340, %r398, %r876;
	fma.rn.f32 	%r875, %r339, %r397, %r875;
	fma.rn.f32 	%r874, %r338, %r396, %r874;
	fma.rn.f32 	%r873, %r339, %r395, %r873;
	fma.rn.f32 	%r872, %r338, %r394, %r872;
	fma.rn.f32 	%r871, %r345, %r393, %r871;
	fma.rn.f32 	%r870, %r344, %r392, %r870;
	fma.rn.f32 	%r869, %r345, %r391, %r869;
	fma.rn.f32 	%r868, %r344, %r390, %r868;
	fma.rn.f32 	%r867, %r343, %r389, %r867;
	fma.rn.f32 	%r866, %r342, %r388, %r866;
	fma.rn.f32 	%r865, %r343, %r387, %r865;
	fma.rn.f32 	%r864, %r342, %r386, %r864;
	fma.rn.f32 	%r863, %r341, %r385, %r863;
	fma.rn.f32 	%r862, %r340, %r384, %r862;
	fma.rn.f32 	%r861, %r341, %r383, %r861;
	fma.rn.f32 	%r860, %r340, %r382, %r860;
	fma.rn.f32 	%r859, %r339, %r381, %r859;
	fma.rn.f32 	%r858, %r338, %r380, %r858;
	fma.rn.f32 	%r857, %r339, %r379, %r857;
	fma.rn.f32 	%r856, %r338, %r378, %r856;
	fma.rn.f32 	%r855, %r345, %r377, %r855;
	fma.rn.f32 	%r854, %r344, %r376, %r854;
	fma.rn.f32 	%r853, %r345, %r375, %r853;
	fma.rn.f32 	%r852, %r344, %r374, %r852;
	fma.rn.f32 	%r851, %r343, %r373, %r851;
	fma.rn.f32 	%r850, %r342, %r372, %r850;
	fma.rn.f32 	%r849, %r343, %r371, %r849;
	fma.rn.f32 	%r848, %r342, %r370, %r848;
	fma.rn.f32 	%r847, %r341, %r369, %r847;
	fma.rn.f32 	%r846, %r340, %r368, %r846;
	fma.rn.f32 	%r845, %r341, %r367, %r845;
	fma.rn.f32 	%r844, %r340, %r366, %r844;
	fma.rn.f32 	%r843, %r339, %r365, %r843;
	fma.rn.f32 	%r842, %r338, %r364, %r842;
	fma.rn.f32 	%r841, %r339, %r363, %r841;
	fma.rn.f32 	%r840, %r338, %r362, %r840;
	fma.rn.f32 	%r839, %r345, %r361, %r839;
	fma.rn.f32 	%r838, %r344, %r360, %r838;
	fma.rn.f32 	%r837, %r345, %r359, %r837;
	fma.rn.f32 	%r836, %r344, %r358, %r836;
	fma.rn.f32 	%r835, %r343, %r357, %r835;
	fma.rn.f32 	%r834, %r342, %r356, %r834;
	fma.rn.f32 	%r833, %r343, %r355, %r833;
	fma.rn.f32 	%r832, %r342, %r354, %r832;
	fma.rn.f32 	%r831, %r341, %r353, %r831;
	fma.rn.f32 	%r830, %r340, %r352, %r830;
	fma.rn.f32 	%r829, %r341, %r351, %r829;
	fma.rn.f32 	%r828, %r340, %r350, %r828;
	fma.rn.f32 	%r827, %r339, %r349, %r827;
	fma.rn.f32 	%r826, %r338, %r348, %r826;
	fma.rn.f32 	%r825, %r339, %r347, %r825;
	fma.rn.f32 	%r824, %r338, %r346, %r824;
	.loc	1 344 18                        // sk06_mlp_down.py:344:18
	add.s64 	%rd101, %rd28, %rd222;
	add.s64 	%rd102, %rd27, %rd222;
	add.s64 	%rd103, %rd26, %rd222;
	.loc	1 345 18                        // sk06_mlp_down.py:345:18
	add.s64 	%rd104, %rd25, %rd222;
	add.s64 	%rd105, %rd24, %rd222;
	add.s64 	%rd106, %rd23, %rd222;
	add.s64 	%rd107, %rd22, %rd222;
	add.s64 	%rd108, %rd21, %rd222;
	add.s64 	%rd109, %rd20, %rd222;
	add.s64 	%rd110, %rd19, %rd222;
	add.s64 	%rd111, %rd18, %rd222;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	add.s64 	%rd112, %rd17, %rd222;
	add.s32 	%r410, %r820, 1;
	setp.gt.s32 	%p6, %r410, 2;
	selp.b32 	%r820, 0, %r410, %p6;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	shl.b32 	%r411, %r820, 13;
	add.s32 	%r412, %r32, %r411;
	add.s32 	%r315, %r412, 49152;
	selp.b32 	%r316, 16, 0, %p4;
	// begin inline asm
	cp.async.cg.shared.global [ %r315 + 0 ], [ %rd101 + 0 ], 0x10, %r316;
	// end inline asm
	add.s32 	%r317, %r412, 51200;
	// begin inline asm
	cp.async.cg.shared.global [ %r317 + 0 ], [ %rd102 + 0 ], 0x10, %r316;
	// end inline asm
	add.s32 	%r318, %r412, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r318 + 0 ], [ %rd103 + 0 ], 0x10, %r316;
	// end inline asm
	add.s32 	%r319, %r412, 55296;
	// begin inline asm
	cp.async.cg.shared.global [ %r319 + 0 ], [ %rd104 + 0 ], 0x10, %r316;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	add.s32 	%r320, %r412, %r411;
	// begin inline asm
	cp.async.cg.shared.global [ %r320 + 0 ], [ %rd105 + 0 ], 0x10, %r316;
	// end inline asm
	add.s32 	%r321, %r320, 2048;
	// begin inline asm
	cp.async.cg.shared.global [ %r321 + 0 ], [ %rd106 + 0 ], 0x10, %r316;
	// end inline asm
	add.s32 	%r322, %r320, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r322 + 0 ], [ %rd107 + 0 ], 0x10, %r316;
	// end inline asm
	add.s32 	%r323, %r320, 6144;
	// begin inline asm
	cp.async.cg.shared.global [ %r323 + 0 ], [ %rd108 + 0 ], 0x10, %r316;
	// end inline asm
	add.s32 	%r324, %r320, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r324 + 0 ], [ %rd109 + 0 ], 0x10, %r316;
	// end inline asm
	add.s32 	%r325, %r320, 10240;
	// begin inline asm
	cp.async.cg.shared.global [ %r325 + 0 ], [ %rd110 + 0 ], 0x10, %r316;
	// end inline asm
	add.s32 	%r326, %r320, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r326 + 0 ], [ %rd111 + 0 ], 0x10, %r316;
	// end inline asm
	add.s32 	%r327, %r320, 14336;
	// begin inline asm
	cp.async.cg.shared.global [ %r327 + 0 ], [ %rd112 + 0 ], 0x10, %r316;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	add.s64 	%rd223, %rd223, 1;
	add.s64 	%rd222, %rd222, 128;
	add.s32 	%r818, %r818, %r26;
	setp.ne.b64 	%p7, %rd16, %rd222;
	@%p7 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_4;
$L__BB0_1:                              // %.._crit_edge_crit_edge
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	and.b32 	%r823, %r2, 16;
	shl.b32 	%r822, %r2, 3;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	and.b32 	%r821, %r2, 96;
	mov.b32 	%r824, 0f00000000;
	mov.b32 	%r825, %r824;
	mov.b32 	%r826, %r824;
	mov.b32 	%r827, %r824;
	mov.b32 	%r828, %r824;
	mov.b32 	%r829, %r824;
	mov.b32 	%r830, %r824;
	mov.b32 	%r831, %r824;
	mov.b32 	%r832, %r824;
	mov.b32 	%r833, %r824;
	mov.b32 	%r834, %r824;
	mov.b32 	%r835, %r824;
	mov.b32 	%r836, %r824;
	mov.b32 	%r837, %r824;
	mov.b32 	%r838, %r824;
	mov.b32 	%r839, %r824;
	mov.b32 	%r840, %r824;
	mov.b32 	%r841, %r824;
	mov.b32 	%r842, %r824;
	mov.b32 	%r843, %r824;
	mov.b32 	%r844, %r824;
	mov.b32 	%r845, %r824;
	mov.b32 	%r846, %r824;
	mov.b32 	%r847, %r824;
	mov.b32 	%r848, %r824;
	mov.b32 	%r849, %r824;
	mov.b32 	%r850, %r824;
	mov.b32 	%r851, %r824;
	mov.b32 	%r852, %r824;
	mov.b32 	%r853, %r824;
	mov.b32 	%r854, %r824;
	mov.b32 	%r855, %r824;
	mov.b32 	%r856, %r824;
	mov.b32 	%r857, %r824;
	mov.b32 	%r858, %r824;
	mov.b32 	%r859, %r824;
	mov.b32 	%r860, %r824;
	mov.b32 	%r861, %r824;
	mov.b32 	%r862, %r824;
	mov.b32 	%r863, %r824;
	mov.b32 	%r864, %r824;
	mov.b32 	%r865, %r824;
	mov.b32 	%r866, %r824;
	mov.b32 	%r867, %r824;
	mov.b32 	%r868, %r824;
	mov.b32 	%r869, %r824;
	mov.b32 	%r870, %r824;
	mov.b32 	%r871, %r824;
	mov.b32 	%r872, %r824;
	mov.b32 	%r873, %r824;
	mov.b32 	%r874, %r824;
	mov.b32 	%r875, %r824;
	mov.b32 	%r876, %r824;
	mov.b32 	%r877, %r824;
	mov.b32 	%r878, %r824;
	mov.b32 	%r879, %r824;
	mov.b32 	%r880, %r824;
	mov.b32 	%r881, %r824;
	mov.b32 	%r882, %r824;
	mov.b32 	%r883, %r824;
	mov.b32 	%r884, %r824;
	mov.b32 	%r885, %r824;
	mov.b32 	%r886, %r824;
	mov.b32 	%r887, %r824;
$L__BB0_4:                              // %._crit_edge
	.loc	1 323 45                        // sk06_mlp_down.py:323:45
	or.b32 	%r523, %r8, %r817;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r524, %r523, 15;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r525, %r524, %r21;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r526, %r523, 14;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r527, %r526, %r21;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r528, %r523, 13;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r529, %r528, %r21;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r530, %r523, 12;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r531, %r530, %r21;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r532, %r523, 11;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r533, %r532, %r21;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r534, %r523, 10;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r535, %r534, %r21;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r536, %r523, 9;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r537, %r536, %r21;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r538, %r523, 8;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r539, %r538, %r21;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r540, %r523, 7;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r541, %r540, %r21;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r542, %r523, 6;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r543, %r542, %r21;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r544, %r523, 5;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r545, %r544, %r21;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r546, %r523, 4;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r547, %r546, %r21;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r548, %r523, 3;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r549, %r548, %r21;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r550, %r523, 2;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r551, %r550, %r21;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r552, %r523, 1;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r553, %r552, %r21;
	rem.s32 	%r554, %r523, %r21;
	.loc	1 323 45                        // sk06_mlp_down.py:323:45
	shl.b32 	%r555, %r10, 3;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r556, %r8, %r555;
	.loc	1 322 45                        // sk06_mlp_down.py:322:45
	shr.u32 	%r557, %r2, 2;
	bfe.u32 	%r558, %r2, 2, 3;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r559, %r558, %r1;
	or.b32 	%r560, %r559, 56;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r561, %r560, %r20;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r562, %r559, 48;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r563, %r562, %r20;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r564, %r559, 40;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r565, %r564, %r20;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r566, %r559, 32;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r567, %r566, %r20;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r568, %r559, 24;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r569, %r568, %r20;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r570, %r559, 16;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r571, %r570, %r20;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r572, %r559, 8;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r573, %r572, %r20;
	rem.s32 	%r574, %r559, %r20;
	.loc	1 322 45                        // sk06_mlp_down.py:322:45
	shr.u32 	%r575, %r2, 4;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r576, %r575, %r1;
	or.b32 	%r577, %r576, 56;
	.loc	1 322 45                        // sk06_mlp_down.py:322:45
	bfe.u32 	%r578, %r2, 4, 3;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r579, %r578, %r1;
	or.b32 	%r580, %r579, 48;
	or.b32 	%r581, %r579, 40;
	or.b32 	%r582, %r579, 32;
	or.b32 	%r583, %r579, 24;
	or.b32 	%r584, %r579, 16;
	or.b32 	%r585, %r579, 8;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 347 38                        // sk06_mlp_down.py:347:38
	mad.wide.s32 	%rd113, %r574, 4, %rd33;
	mad.wide.s32 	%rd114, %r573, 4, %rd33;
	mad.wide.s32 	%rd115, %r571, 4, %rd33;
	mad.wide.s32 	%rd116, %r569, 4, %rd33;
	mad.wide.s32 	%rd117, %r567, 4, %rd33;
	mad.wide.s32 	%rd118, %r565, 4, %rd33;
	mad.wide.s32 	%rd119, %r563, 4, %rd33;
	mad.wide.s32 	%rd120, %r561, 4, %rd33;
	.loc	1 347 24                        // sk06_mlp_down.py:347:24
	// begin inline asm
	mov.u32 %r413, 0x0;
	ld.global.b32 { %r413 }, [ %rd113 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r414, 0x0;
	ld.global.b32 { %r414 }, [ %rd114 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r415, 0x0;
	ld.global.b32 { %r415 }, [ %rd115 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r416, 0x0;
	ld.global.b32 { %r416 }, [ %rd116 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r417, 0x0;
	ld.global.b32 { %r417 }, [ %rd117 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r418, 0x0;
	ld.global.b32 { %r418 }, [ %rd118 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r419, 0x0;
	ld.global.b32 { %r419 }, [ %rd119 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r420, 0x0;
	ld.global.b32 { %r420 }, [ %rd120 + 0 ];
	// end inline asm
	.loc	1 350 49                        // sk06_mlp_down.py:350:49
	mul.lo.s32 	%r586, %r4, %r24;
	mul.lo.s32 	%r587, %r5, %r24;
	mul.lo.s32 	%r588, %r6, %r24;
	mul.lo.s32 	%r589, %r7, %r24;
	.loc	1 350 31                        // sk06_mlp_down.py:350:31
	mad.wide.s32 	%rd193, %r586, 2, %rd32;
	mad.wide.s32 	%rd194, %r587, 2, %rd32;
	mad.wide.s32 	%rd195, %r588, 2, %rd32;
	mad.wide.s32 	%rd196, %r589, 2, %rd32;
	.loc	1 350 82                        // sk06_mlp_down.py:350:82
	mul.lo.s32 	%r590, %r554, %r25;
	mul.lo.s32 	%r591, %r553, %r25;
	mul.lo.s32 	%r592, %r551, %r25;
	mul.lo.s32 	%r593, %r549, %r25;
	mul.lo.s32 	%r594, %r547, %r25;
	mul.lo.s32 	%r595, %r545, %r25;
	mul.lo.s32 	%r596, %r543, %r25;
	mul.lo.s32 	%r597, %r541, %r25;
	mul.lo.s32 	%r598, %r539, %r25;
	mul.lo.s32 	%r599, %r537, %r25;
	mul.lo.s32 	%r600, %r535, %r25;
	mul.lo.s32 	%r601, %r533, %r25;
	mul.lo.s32 	%r602, %r531, %r25;
	mul.lo.s32 	%r603, %r529, %r25;
	mul.lo.s32 	%r604, %r527, %r25;
	mul.lo.s32 	%r605, %r525, %r25;
	.loc	1 350 64                        // sk06_mlp_down.py:350:64
	mul.wide.s32 	%rd197, %r590, 2;
	add.s64 	%rd121, %rd193, %rd197;
	mul.wide.s32 	%rd198, %r591, 2;
	add.s64 	%rd122, %rd193, %rd198;
	mul.wide.s32 	%rd199, %r592, 2;
	add.s64 	%rd123, %rd193, %rd199;
	mul.wide.s32 	%rd200, %r593, 2;
	add.s64 	%rd124, %rd193, %rd200;
	mul.wide.s32 	%rd201, %r594, 2;
	add.s64 	%rd125, %rd193, %rd201;
	mul.wide.s32 	%rd202, %r595, 2;
	add.s64 	%rd126, %rd193, %rd202;
	mul.wide.s32 	%rd203, %r596, 2;
	add.s64 	%rd127, %rd193, %rd203;
	mul.wide.s32 	%rd204, %r597, 2;
	add.s64 	%rd128, %rd193, %rd204;
	mul.wide.s32 	%rd205, %r598, 2;
	add.s64 	%rd129, %rd193, %rd205;
	mul.wide.s32 	%rd206, %r599, 2;
	add.s64 	%rd130, %rd193, %rd206;
	mul.wide.s32 	%rd207, %r600, 2;
	add.s64 	%rd131, %rd193, %rd207;
	mul.wide.s32 	%rd208, %r601, 2;
	add.s64 	%rd132, %rd193, %rd208;
	mul.wide.s32 	%rd209, %r602, 2;
	add.s64 	%rd133, %rd193, %rd209;
	mul.wide.s32 	%rd210, %r603, 2;
	add.s64 	%rd134, %rd193, %rd210;
	mul.wide.s32 	%rd211, %r604, 2;
	add.s64 	%rd135, %rd193, %rd211;
	mul.wide.s32 	%rd212, %r605, 2;
	add.s64 	%rd136, %rd193, %rd212;
	add.s64 	%rd137, %rd194, %rd197;
	add.s64 	%rd138, %rd194, %rd198;
	add.s64 	%rd139, %rd194, %rd199;
	add.s64 	%rd140, %rd194, %rd200;
	add.s64 	%rd141, %rd194, %rd201;
	add.s64 	%rd142, %rd194, %rd202;
	add.s64 	%rd143, %rd194, %rd203;
	add.s64 	%rd144, %rd194, %rd204;
	add.s64 	%rd145, %rd194, %rd205;
	add.s64 	%rd146, %rd194, %rd206;
	add.s64 	%rd147, %rd194, %rd207;
	add.s64 	%rd148, %rd194, %rd208;
	add.s64 	%rd149, %rd194, %rd209;
	add.s64 	%rd150, %rd194, %rd210;
	add.s64 	%rd151, %rd194, %rd211;
	add.s64 	%rd152, %rd194, %rd212;
	add.s64 	%rd153, %rd195, %rd197;
	add.s64 	%rd154, %rd195, %rd198;
	add.s64 	%rd155, %rd195, %rd199;
	add.s64 	%rd156, %rd195, %rd200;
	add.s64 	%rd157, %rd195, %rd201;
	add.s64 	%rd158, %rd195, %rd202;
	add.s64 	%rd159, %rd195, %rd203;
	add.s64 	%rd160, %rd195, %rd204;
	add.s64 	%rd161, %rd195, %rd205;
	add.s64 	%rd162, %rd195, %rd206;
	add.s64 	%rd163, %rd195, %rd207;
	add.s64 	%rd164, %rd195, %rd208;
	add.s64 	%rd165, %rd195, %rd209;
	add.s64 	%rd166, %rd195, %rd210;
	add.s64 	%rd167, %rd195, %rd211;
	add.s64 	%rd168, %rd195, %rd212;
	add.s64 	%rd169, %rd196, %rd197;
	add.s64 	%rd170, %rd196, %rd198;
	add.s64 	%rd171, %rd196, %rd199;
	add.s64 	%rd172, %rd196, %rd200;
	add.s64 	%rd173, %rd196, %rd201;
	add.s64 	%rd174, %rd196, %rd202;
	add.s64 	%rd175, %rd196, %rd203;
	add.s64 	%rd176, %rd196, %rd204;
	add.s64 	%rd177, %rd196, %rd205;
	add.s64 	%rd178, %rd196, %rd206;
	add.s64 	%rd179, %rd196, %rd207;
	add.s64 	%rd180, %rd196, %rd208;
	add.s64 	%rd181, %rd196, %rd209;
	add.s64 	%rd182, %rd196, %rd210;
	add.s64 	%rd183, %rd196, %rd211;
	add.s64 	%rd184, %rd196, %rd212;
	.loc	1 350 19                        // sk06_mlp_down.py:350:19
	// begin inline asm
	mov.u16 %rs1, 0x0;
	ld.global.b16 { %rs1 }, [ %rd121 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs2, 0x0;
	ld.global.b16 { %rs2 }, [ %rd122 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs3, 0x0;
	ld.global.b16 { %rs3 }, [ %rd123 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs4, 0x0;
	ld.global.b16 { %rs4 }, [ %rd124 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs5, 0x0;
	ld.global.b16 { %rs5 }, [ %rd125 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs6, 0x0;
	ld.global.b16 { %rs6 }, [ %rd126 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs7, 0x0;
	ld.global.b16 { %rs7 }, [ %rd127 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs8, 0x0;
	ld.global.b16 { %rs8 }, [ %rd128 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs9, 0x0;
	ld.global.b16 { %rs9 }, [ %rd129 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs10, 0x0;
	ld.global.b16 { %rs10 }, [ %rd130 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs11, 0x0;
	ld.global.b16 { %rs11 }, [ %rd131 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs12, 0x0;
	ld.global.b16 { %rs12 }, [ %rd132 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs13, 0x0;
	ld.global.b16 { %rs13 }, [ %rd133 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs14, 0x0;
	ld.global.b16 { %rs14 }, [ %rd134 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs15, 0x0;
	ld.global.b16 { %rs15 }, [ %rd135 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs16, 0x0;
	ld.global.b16 { %rs16 }, [ %rd136 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs17, 0x0;
	ld.global.b16 { %rs17 }, [ %rd137 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs18, 0x0;
	ld.global.b16 { %rs18 }, [ %rd138 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs19, 0x0;
	ld.global.b16 { %rs19 }, [ %rd139 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs20, 0x0;
	ld.global.b16 { %rs20 }, [ %rd140 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs21, 0x0;
	ld.global.b16 { %rs21 }, [ %rd141 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs22, 0x0;
	ld.global.b16 { %rs22 }, [ %rd142 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs23, 0x0;
	ld.global.b16 { %rs23 }, [ %rd143 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs24, 0x0;
	ld.global.b16 { %rs24 }, [ %rd144 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs25, 0x0;
	ld.global.b16 { %rs25 }, [ %rd145 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs26, 0x0;
	ld.global.b16 { %rs26 }, [ %rd146 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs27, 0x0;
	ld.global.b16 { %rs27 }, [ %rd147 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs28, 0x0;
	ld.global.b16 { %rs28 }, [ %rd148 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs29, 0x0;
	ld.global.b16 { %rs29 }, [ %rd149 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs30, 0x0;
	ld.global.b16 { %rs30 }, [ %rd150 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs31, 0x0;
	ld.global.b16 { %rs31 }, [ %rd151 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs32, 0x0;
	ld.global.b16 { %rs32 }, [ %rd152 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs33, 0x0;
	ld.global.b16 { %rs33 }, [ %rd153 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs34, 0x0;
	ld.global.b16 { %rs34 }, [ %rd154 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs35, 0x0;
	ld.global.b16 { %rs35 }, [ %rd155 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs36, 0x0;
	ld.global.b16 { %rs36 }, [ %rd156 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs37, 0x0;
	ld.global.b16 { %rs37 }, [ %rd157 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs38, 0x0;
	ld.global.b16 { %rs38 }, [ %rd158 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs39, 0x0;
	ld.global.b16 { %rs39 }, [ %rd159 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs40, 0x0;
	ld.global.b16 { %rs40 }, [ %rd160 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs41, 0x0;
	ld.global.b16 { %rs41 }, [ %rd161 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs42, 0x0;
	ld.global.b16 { %rs42 }, [ %rd162 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs43, 0x0;
	ld.global.b16 { %rs43 }, [ %rd163 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs44, 0x0;
	ld.global.b16 { %rs44 }, [ %rd164 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs45, 0x0;
	ld.global.b16 { %rs45 }, [ %rd165 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs46, 0x0;
	ld.global.b16 { %rs46 }, [ %rd166 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs47, 0x0;
	ld.global.b16 { %rs47 }, [ %rd167 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs48, 0x0;
	ld.global.b16 { %rs48 }, [ %rd168 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs49, 0x0;
	ld.global.b16 { %rs49 }, [ %rd169 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs50, 0x0;
	ld.global.b16 { %rs50 }, [ %rd170 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs51, 0x0;
	ld.global.b16 { %rs51 }, [ %rd171 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs52, 0x0;
	ld.global.b16 { %rs52 }, [ %rd172 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs53, 0x0;
	ld.global.b16 { %rs53 }, [ %rd173 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs54, 0x0;
	ld.global.b16 { %rs54 }, [ %rd174 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs55, 0x0;
	ld.global.b16 { %rs55 }, [ %rd175 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs56, 0x0;
	ld.global.b16 { %rs56 }, [ %rd176 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs57, 0x0;
	ld.global.b16 { %rs57 }, [ %rd177 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs58, 0x0;
	ld.global.b16 { %rs58 }, [ %rd178 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs59, 0x0;
	ld.global.b16 { %rs59 }, [ %rd179 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs60, 0x0;
	ld.global.b16 { %rs60 }, [ %rd180 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs61, 0x0;
	ld.global.b16 { %rs61 }, [ %rd181 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs62, 0x0;
	ld.global.b16 { %rs62 }, [ %rd182 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs63, 0x0;
	ld.global.b16 { %rs63 }, [ %rd183 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs64, 0x0;
	ld.global.b16 { %rs64 }, [ %rd184 + 0 ];
	// end inline asm
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	shl.b32 	%r606, %r12, 6;
	shl.b32 	%r607, %r3, 1;
	or.b32 	%r608, %r606, %r817;
	xor.b32 	%r609, %r608, %r607;
	add.s32 	%r421, %r136, %r609;
	mov.b32 	%r422, {%rs1, %rs2};
	mov.b32 	%r423, {%rs3, %rs4};
	mov.b32 	%r424, {%rs5, %rs6};
	mov.b32 	%r425, {%rs7, %rs8};
	// begin inline asm
	st.shared.v4.b32 [ %r421 + 0 ], { %r422, %r423, %r424, %r425 };
	// end inline asm
	add.s32 	%r426, %r421, 256;
	mov.b32 	%r427, {%rs9, %rs10};
	mov.b32 	%r428, {%rs11, %rs12};
	mov.b32 	%r429, {%rs13, %rs14};
	mov.b32 	%r430, {%rs15, %rs16};
	// begin inline asm
	st.shared.v4.b32 [ %r426 + 0 ], { %r427, %r428, %r429, %r430 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r610, %r9, 9;
	shl.b32 	%r611, %r10, 4;
	setp.eq.b32 	%p16, %r823, 0;
	shl.b32 	%r612, %r823, 1;
	and.b32 	%r613, %r822, 256;
	and.b32 	%r614, %r557, 16;
	or.b32 	%r615, %r610, %r613;
	xor.b32 	%r616, %r611, %r612;
	xor.b32 	%r617, %r616, %r614;
	or.b32 	%r618, %r617, %r615;
	add.s32 	%r619, %r136, %r618;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r620, %r621, %r622, %r623}, [%r619];
	xor.b32 	%r624, %r618, 64;
	add.s32 	%r625, %r136, %r624;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r626, %r627, %r628, %r629}, [%r625];
	bar.sync 	0;
	mov.b32 	%r431, {%rs17, %rs18};
	mov.b32 	%r432, {%rs19, %rs20};
	mov.b32 	%r433, {%rs21, %rs22};
	mov.b32 	%r434, {%rs23, %rs24};
	// begin inline asm
	st.shared.v4.b32 [ %r421 + 0 ], { %r431, %r432, %r433, %r434 };
	// end inline asm
	mov.b32 	%r435, {%rs25, %rs26};
	mov.b32 	%r436, {%rs27, %rs28};
	mov.b32 	%r437, {%rs29, %rs30};
	mov.b32 	%r438, {%rs31, %rs32};
	// begin inline asm
	st.shared.v4.b32 [ %r426 + 0 ], { %r435, %r436, %r437, %r438 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r630, %r631, %r632, %r633}, [%r619];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r634, %r635, %r636, %r637}, [%r625];
	bar.sync 	0;
	mov.b32 	%r439, {%rs33, %rs34};
	mov.b32 	%r440, {%rs35, %rs36};
	mov.b32 	%r441, {%rs37, %rs38};
	mov.b32 	%r442, {%rs39, %rs40};
	// begin inline asm
	st.shared.v4.b32 [ %r421 + 0 ], { %r439, %r440, %r441, %r442 };
	// end inline asm
	mov.b32 	%r443, {%rs41, %rs42};
	mov.b32 	%r444, {%rs43, %rs44};
	mov.b32 	%r445, {%rs45, %rs46};
	mov.b32 	%r446, {%rs47, %rs48};
	// begin inline asm
	st.shared.v4.b32 [ %r426 + 0 ], { %r443, %r444, %r445, %r446 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r638, %r639, %r640, %r641}, [%r619];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r642, %r643, %r644, %r645}, [%r625];
	bar.sync 	0;
	mov.b32 	%r447, {%rs49, %rs50};
	mov.b32 	%r448, {%rs51, %rs52};
	mov.b32 	%r449, {%rs53, %rs54};
	mov.b32 	%r450, {%rs55, %rs56};
	// begin inline asm
	st.shared.v4.b32 [ %r421 + 0 ], { %r447, %r448, %r449, %r450 };
	// end inline asm
	mov.b32 	%r451, {%rs57, %rs58};
	mov.b32 	%r452, {%rs59, %rs60};
	mov.b32 	%r453, {%rs61, %rs62};
	mov.b32 	%r454, {%rs63, %rs64};
	// begin inline asm
	st.shared.v4.b32 [ %r426 + 0 ], { %r451, %r452, %r453, %r454 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r646, %r647, %r648, %r649}, [%r619];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r650, %r651, %r652, %r653}, [%r625];
	.loc	1 357 31                        // sk06_mlp_down.py:357:31
	setp.lt.s32 	%p17, %r579, %r20;
	setp.lt.s32 	%p18, %r585, %r20;
	setp.lt.s32 	%p19, %r584, %r20;
	setp.lt.s32 	%p20, %r583, %r20;
	setp.lt.s32 	%p21, %r582, %r20;
	setp.lt.s32 	%p22, %r581, %r20;
	setp.lt.s32 	%p23, %r580, %r20;
	setp.lt.s32 	%p24, %r577, %r20;
	.loc	1 357 54                        // sk06_mlp_down.py:357:54
	setp.lt.s32 	%p25, %r556, %r21;
	.loc	1 357 37                        // sk06_mlp_down.py:357:37
	and.pred 	%p8, %p17, %p25;
	and.pred 	%p9, %p18, %p25;
	and.pred 	%p10, %p19, %p25;
	and.pred 	%p11, %p20, %p25;
	and.pred 	%p12, %p21, %p25;
	and.pred 	%p13, %p22, %p25;
	and.pred 	%p14, %p23, %p25;
	and.pred 	%p15, %p24, %p25;
	.loc	1 355 35                        // sk06_mlp_down.py:355:35
	mul.lo.s32 	%r654, %r579, %r23;
	mul.lo.s32 	%r655, %r585, %r23;
	mul.lo.s32 	%r656, %r584, %r23;
	mul.lo.s32 	%r657, %r583, %r23;
	mul.lo.s32 	%r658, %r582, %r23;
	mul.lo.s32 	%r659, %r581, %r23;
	mul.lo.s32 	%r660, %r580, %r23;
	mul.lo.s32 	%r661, %r577, %r23;
	.loc	1 355 18                        // sk06_mlp_down.py:355:18
	mad.wide.s32 	%rd213, %r654, 2, %rd31;
	mad.wide.s32 	%rd214, %r655, 2, %rd31;
	mad.wide.s32 	%rd215, %r656, 2, %rd31;
	mad.wide.s32 	%rd216, %r657, 2, %rd31;
	mad.wide.s32 	%rd217, %r658, 2, %rd31;
	mad.wide.s32 	%rd218, %r659, 2, %rd31;
	mad.wide.s32 	%rd219, %r660, 2, %rd31;
	mad.wide.s32 	%rd220, %r661, 2, %rd31;
	.loc	1 355 50                        // sk06_mlp_down.py:355:50
	mul.wide.s32 	%rd221, %r556, 2;
	add.s64 	%rd185, %rd213, %rd221;
	add.s64 	%rd186, %rd214, %rd221;
	add.s64 	%rd187, %rd215, %rd221;
	add.s64 	%rd188, %rd216, %rd221;
	add.s64 	%rd189, %rd217, %rd221;
	add.s64 	%rd190, %rd218, %rd221;
	add.s64 	%rd191, %rd219, %rd221;
	add.s64 	%rd192, %rd220, %rd221;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs65, %rs66}, %r620;
	cvt.f32.bf16 	%r662, %rs66;
	cvt.f32.bf16 	%r663, %rs65;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r664, %r824, %r413, %r663;
	fma.rn.f32 	%r665, %r825, %r413, %r662;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r456, %r665, %r664;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs67, %rs68}, %r621;
	cvt.f32.bf16 	%r666, %rs68;
	cvt.f32.bf16 	%r667, %rs67;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r668, %r826, %r414, %r667;
	fma.rn.f32 	%r669, %r827, %r414, %r666;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r457, %r669, %r668;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs69, %rs70}, %r622;
	cvt.f32.bf16 	%r670, %rs70;
	cvt.f32.bf16 	%r671, %rs69;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r672, %r828, %r413, %r671;
	fma.rn.f32 	%r673, %r829, %r413, %r670;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r466, %r673, %r672;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs71, %rs72}, %r623;
	cvt.f32.bf16 	%r674, %rs72;
	cvt.f32.bf16 	%r675, %rs71;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r676, %r830, %r414, %r675;
	fma.rn.f32 	%r677, %r831, %r414, %r674;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r467, %r677, %r676;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs73, %rs74}, %r626;
	cvt.f32.bf16 	%r678, %rs74;
	cvt.f32.bf16 	%r679, %rs73;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r680, %r832, %r413, %r679;
	fma.rn.f32 	%r681, %r833, %r413, %r678;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r461, %r681, %r680;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs75, %rs76}, %r627;
	cvt.f32.bf16 	%r682, %rs76;
	cvt.f32.bf16 	%r683, %rs75;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r684, %r834, %r414, %r683;
	fma.rn.f32 	%r685, %r835, %r414, %r682;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r462, %r685, %r684;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs77, %rs78}, %r628;
	cvt.f32.bf16 	%r686, %rs78;
	cvt.f32.bf16 	%r687, %rs77;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r688, %r836, %r413, %r687;
	fma.rn.f32 	%r689, %r837, %r413, %r686;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r471, %r689, %r688;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs79, %rs80}, %r629;
	cvt.f32.bf16 	%r690, %rs80;
	cvt.f32.bf16 	%r691, %rs79;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r692, %r838, %r414, %r691;
	fma.rn.f32 	%r693, %r839, %r414, %r690;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r472, %r693, %r692;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs81, %rs82}, %r630;
	cvt.f32.bf16 	%r694, %rs82;
	cvt.f32.bf16 	%r695, %rs81;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r696, %r840, %r415, %r695;
	fma.rn.f32 	%r697, %r841, %r415, %r694;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r458, %r697, %r696;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs83, %rs84}, %r631;
	cvt.f32.bf16 	%r698, %rs84;
	cvt.f32.bf16 	%r699, %rs83;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r700, %r842, %r416, %r699;
	fma.rn.f32 	%r701, %r843, %r416, %r698;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r459, %r701, %r700;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs85, %rs86}, %r632;
	cvt.f32.bf16 	%r702, %rs86;
	cvt.f32.bf16 	%r703, %rs85;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r704, %r844, %r415, %r703;
	fma.rn.f32 	%r705, %r845, %r415, %r702;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r468, %r705, %r704;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs87, %rs88}, %r633;
	cvt.f32.bf16 	%r706, %rs88;
	cvt.f32.bf16 	%r707, %rs87;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r708, %r846, %r416, %r707;
	fma.rn.f32 	%r709, %r847, %r416, %r706;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r469, %r709, %r708;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs89, %rs90}, %r634;
	cvt.f32.bf16 	%r710, %rs90;
	cvt.f32.bf16 	%r711, %rs89;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r712, %r848, %r415, %r711;
	fma.rn.f32 	%r713, %r849, %r415, %r710;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r463, %r713, %r712;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs91, %rs92}, %r635;
	cvt.f32.bf16 	%r714, %rs92;
	cvt.f32.bf16 	%r715, %rs91;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r716, %r850, %r416, %r715;
	fma.rn.f32 	%r717, %r851, %r416, %r714;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r464, %r717, %r716;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs93, %rs94}, %r636;
	cvt.f32.bf16 	%r718, %rs94;
	cvt.f32.bf16 	%r719, %rs93;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r720, %r852, %r415, %r719;
	fma.rn.f32 	%r721, %r853, %r415, %r718;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r473, %r721, %r720;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs95, %rs96}, %r637;
	cvt.f32.bf16 	%r722, %rs96;
	cvt.f32.bf16 	%r723, %rs95;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r724, %r854, %r416, %r723;
	fma.rn.f32 	%r725, %r855, %r416, %r722;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r474, %r725, %r724;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs97, %rs98}, %r638;
	cvt.f32.bf16 	%r726, %rs98;
	cvt.f32.bf16 	%r727, %rs97;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r728, %r856, %r417, %r727;
	fma.rn.f32 	%r729, %r857, %r417, %r726;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r475, %r729, %r728;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs99, %rs100}, %r639;
	cvt.f32.bf16 	%r730, %rs100;
	cvt.f32.bf16 	%r731, %rs99;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r732, %r858, %r418, %r731;
	fma.rn.f32 	%r733, %r859, %r418, %r730;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r476, %r733, %r732;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs101, %rs102}, %r640;
	cvt.f32.bf16 	%r734, %rs102;
	cvt.f32.bf16 	%r735, %rs101;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r736, %r860, %r417, %r735;
	fma.rn.f32 	%r737, %r861, %r417, %r734;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r483, %r737, %r736;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs103, %rs104}, %r641;
	cvt.f32.bf16 	%r738, %rs104;
	cvt.f32.bf16 	%r739, %rs103;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r740, %r862, %r418, %r739;
	fma.rn.f32 	%r741, %r863, %r418, %r738;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r484, %r741, %r740;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs105, %rs106}, %r642;
	cvt.f32.bf16 	%r742, %rs106;
	cvt.f32.bf16 	%r743, %rs105;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r744, %r864, %r417, %r743;
	fma.rn.f32 	%r745, %r865, %r417, %r742;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r479, %r745, %r744;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs107, %rs108}, %r643;
	cvt.f32.bf16 	%r746, %rs108;
	cvt.f32.bf16 	%r747, %rs107;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r748, %r866, %r418, %r747;
	fma.rn.f32 	%r749, %r867, %r418, %r746;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r480, %r749, %r748;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs109, %rs110}, %r644;
	cvt.f32.bf16 	%r750, %rs110;
	cvt.f32.bf16 	%r751, %rs109;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r752, %r868, %r417, %r751;
	fma.rn.f32 	%r753, %r869, %r417, %r750;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r487, %r753, %r752;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs111, %rs112}, %r645;
	cvt.f32.bf16 	%r754, %rs112;
	cvt.f32.bf16 	%r755, %rs111;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r756, %r870, %r418, %r755;
	fma.rn.f32 	%r757, %r871, %r418, %r754;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r488, %r757, %r756;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs113, %rs114}, %r646;
	cvt.f32.bf16 	%r758, %rs114;
	cvt.f32.bf16 	%r759, %rs113;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r760, %r872, %r419, %r759;
	fma.rn.f32 	%r761, %r873, %r419, %r758;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r477, %r761, %r760;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs115, %rs116}, %r647;
	cvt.f32.bf16 	%r762, %rs116;
	cvt.f32.bf16 	%r763, %rs115;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r764, %r874, %r420, %r763;
	fma.rn.f32 	%r765, %r875, %r420, %r762;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r478, %r765, %r764;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs117, %rs118}, %r648;
	cvt.f32.bf16 	%r766, %rs118;
	cvt.f32.bf16 	%r767, %rs117;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r768, %r876, %r419, %r767;
	fma.rn.f32 	%r769, %r877, %r419, %r766;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r485, %r769, %r768;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs119, %rs120}, %r649;
	cvt.f32.bf16 	%r770, %rs120;
	cvt.f32.bf16 	%r771, %rs119;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r772, %r878, %r420, %r771;
	fma.rn.f32 	%r773, %r879, %r420, %r770;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r486, %r773, %r772;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs121, %rs122}, %r650;
	cvt.f32.bf16 	%r774, %rs122;
	cvt.f32.bf16 	%r775, %rs121;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r776, %r880, %r419, %r775;
	fma.rn.f32 	%r777, %r881, %r419, %r774;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r481, %r777, %r776;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs123, %rs124}, %r651;
	cvt.f32.bf16 	%r778, %rs124;
	cvt.f32.bf16 	%r779, %rs123;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r780, %r882, %r420, %r779;
	fma.rn.f32 	%r781, %r883, %r420, %r778;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r482, %r781, %r780;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs125, %rs126}, %r652;
	cvt.f32.bf16 	%r782, %rs126;
	cvt.f32.bf16 	%r783, %rs125;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r784, %r884, %r419, %r783;
	fma.rn.f32 	%r785, %r885, %r419, %r782;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r489, %r785, %r784;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs127, %rs128}, %r653;
	cvt.f32.bf16 	%r786, %rs128;
	cvt.f32.bf16 	%r787, %rs127;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r788, %r886, %r420, %r787;
	fma.rn.f32 	%r789, %r887, %r420, %r786;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r490, %r789, %r788;
	bar.sync 	0;
	and.b32 	%r790, %r2, 3;
	shl.b32 	%r791, %r790, 11;
	shl.b32 	%r792, %r790, 5;
	shl.b32 	%r793, %r2, 4;
	and.b32 	%r794, %r793, 384;
	shr.u32 	%r795, %r821, 1;
	bfe.s32 	%r796, %r2, 2, 1;
	and.b32 	%r797, %r796, 1040;
	or.b32 	%r798, %r792, %r794;
	xor.b32 	%r799, %r797, %r795;
	xor.b32 	%r800, %r799, %r798;
	or.b32 	%r801, %r800, %r791;
	add.s32 	%r455, %r136, %r801;
	// begin inline asm
	st.shared.v4.b32 [ %r455 + 0 ], { %r456, %r457, %r458, %r459 };
	// end inline asm
	add.s32 	%r460, %r455, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r460 + 0 ], { %r461, %r462, %r463, %r464 };
	// end inline asm
	xor.b32 	%r802, %r801, 64;
	add.s32 	%r465, %r136, %r802;
	// begin inline asm
	st.shared.v4.b32 [ %r465 + 0 ], { %r466, %r467, %r468, %r469 };
	// end inline asm
	add.s32 	%r470, %r465, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r470 + 0 ], { %r471, %r472, %r473, %r474 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r803, %r821, 2;
	shl.b32 	%r804, %r2, 6;
	and.b32 	%r805, %r804, 512;
	selp.b32 	%r806, 0, 1040, %p16;
	or.b32 	%r807, %r817, %r803;
	xor.b32 	%r808, %r807, %r806;
	or.b32 	%r809, %r808, %r805;
	add.s32 	%r810, %r136, %r809;
	ld.shared.v4.b32 	{%r491, %r495, %r499, %r503}, [%r810];
	xor.b32 	%r811, %r809, 32;
	add.s32 	%r812, %r136, %r811;
	ld.shared.v4.b32 	{%r492, %r496, %r500, %r504}, [%r812+2048];
	xor.b32 	%r813, %r809, 64;
	add.s32 	%r814, %r136, %r813;
	ld.shared.v4.b32 	{%r493, %r497, %r501, %r505}, [%r814+4096];
	xor.b32 	%r815, %r809, 96;
	add.s32 	%r816, %r136, %r815;
	ld.shared.v4.b32 	{%r494, %r498, %r502, %r506}, [%r816+6144];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r455 + 0 ], { %r475, %r476, %r477, %r478 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r460 + 0 ], { %r479, %r480, %r481, %r482 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r465 + 0 ], { %r483, %r484, %r485, %r486 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r470 + 0 ], { %r487, %r488, %r489, %r490 };
	// end inline asm
	bar.sync 	0;
	ld.shared.v4.b32 	{%r507, %r511, %r515, %r519}, [%r810];
	ld.shared.v4.b32 	{%r508, %r512, %r516, %r520}, [%r812+2048];
	ld.shared.v4.b32 	{%r509, %r513, %r517, %r521}, [%r814+4096];
	ld.shared.v4.b32 	{%r510, %r514, %r518, %r522}, [%r816+6144];
	.loc	1 356 8                         // sk06_mlp_down.py:356:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd185 + 0 ], { %r491, %r492, %r493, %r494 };
	// end inline asm
	// begin inline asm
	@%p9 st.global.v4.b32 [ %rd186 + 0 ], { %r495, %r496, %r497, %r498 };
	// end inline asm
	// begin inline asm
	@%p10 st.global.v4.b32 [ %rd187 + 0 ], { %r499, %r500, %r501, %r502 };
	// end inline asm
	// begin inline asm
	@%p11 st.global.v4.b32 [ %rd188 + 0 ], { %r503, %r504, %r505, %r506 };
	// end inline asm
	// begin inline asm
	@%p12 st.global.v4.b32 [ %rd189 + 0 ], { %r507, %r508, %r509, %r510 };
	// end inline asm
	// begin inline asm
	@%p13 st.global.v4.b32 [ %rd190 + 0 ], { %r511, %r512, %r513, %r514 };
	// end inline asm
	// begin inline asm
	@%p14 st.global.v4.b32 [ %rd191 + 0 ], { %r515, %r516, %r517, %r518 };
	// end inline asm
	// begin inline asm
	@%p15 st.global.v4.b32 [ %rd192 + 0 ], { %r519, %r520, %r521, %r522 };
	// end inline asm
	.loc	1 354 4                         // sk06_mlp_down.py:354:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/workspace/vllm/_genesis/kernels/sk06_mlp_down.py"
	.file	2 "/usr/local/lib/python3.12/dist-packages/triton/language/standard.py"
	.section	.debug_abbrev
	{
.b8 1                                   // Abbreviation Code
.b8 17                                  // DW_TAG_compile_unit
.b8 1                                   // DW_CHILDREN_yes
.b8 37                                  // DW_AT_producer
.b8 8                                   // DW_FORM_string
.b8 19                                  // DW_AT_language
.b8 5                                   // DW_FORM_data2
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 16                                  // DW_AT_stmt_list
.b8 6                                   // DW_FORM_data4
.b8 27                                  // DW_AT_comp_dir
.b8 8                                   // DW_FORM_string
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 2                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 0                                   // DW_CHILDREN_no
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 32                                  // DW_AT_inline
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 3                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 1                                   // DW_CHILDREN_yes
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 4                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 0                                   // DW_CHILDREN_no
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 5                                   // DW_FORM_data2
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 168                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0xa1 DW_TAG_compile_unit
.b8 116                                 // DW_AT_producer
.b8 114
.b8 105
.b8 116
.b8 111
.b8 110
.b8 0
.b8 2                                   // DW_AT_language
.b8 0
.b8 115                                 // DW_AT_name
.b8 107
.b8 48
.b8 54
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 100
.b8 111
.b8 119
.b8 110
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 119
.b8 111
.b8 114
.b8 107
.b8 115
.b8 112
.b8 97
.b8 99
.b8 101
.b8 47
.b8 118
.b8 108
.b8 108
.b8 109
.b8 47
.b8 95
.b8 103
.b8 101
.b8 110
.b8 101
.b8 115
.b8 105
.b8 115
.b8 47
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 115
.b8 0
.b8 2                                   // Abbrev [2] 0x4b:0x18 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 54
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 100
.b8 111
.b8 119
.b8 110
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x63:0x48 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 75                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x78:0x19 DW_TAG_inlined_subroutine
.b32 75                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 58                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x91:0x19 DW_TAG_inlined_subroutine
.b32 75                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 59                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_7 = _Nativo(
    "sk06_mlp_down/tile64x128x128_shift1_abi16",
    _PTX_7, "_sk06_mlp_down_kernel",
    warps=4, shared=74240,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 17, 18],
    horneado={11: 1, 12: 1, 15: 1, 19: 1, 20: 64, 21: 128, 22: 128, 23: 8, 24: 128, 25: True},
    div16=[8, 9, 10, 13, 14, 16, 17],
)

_PTX_8 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk06_mlp_down_kernel   // -- Begin function _sk06_mlp_down_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk06_mlp_down_kernel
.visible .entry _sk06_mlp_down_kernel(
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_6,
	.param .u32 _sk06_mlp_down_kernel_param_7,
	.param .u32 _sk06_mlp_down_kernel_param_8,
	.param .u32 _sk06_mlp_down_kernel_param_9,
	.param .u32 _sk06_mlp_down_kernel_param_10,
	.param .u32 _sk06_mlp_down_kernel_param_11,
	.param .u32 _sk06_mlp_down_kernel_param_12,
	.param .u32 _sk06_mlp_down_kernel_param_13,
	.param .u32 _sk06_mlp_down_kernel_param_14,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_16
)
.reqntid 256
{
	.reg .pred 	%p<42>;
	.reg .b16 	%rs<129>;
	.reg .b32 	%r<1408>;
	.reg .b64 	%rd<130>;
	.loc	1 304 0                         // sk06_mlp_down.py:304:0
$L__func_begin0:
	.loc	1 304 0                         // sk06_mlp_down.py:304:0

// %bb.0:
	ld.param.b32 	%r18, [_sk06_mlp_down_kernel_param_13];
	ld.param.b32 	%r17, [_sk06_mlp_down_kernel_param_12];
	ld.param.b32 	%r16, [_sk06_mlp_down_kernel_param_8];
	ld.param.b32 	%r15, [_sk06_mlp_down_kernel_param_7];
	ld.param.b64 	%rd13, [_sk06_mlp_down_kernel_param_5];
	ld.param.b64 	%rd12, [_sk06_mlp_down_kernel_param_4];
	ld.param.b64 	%rd11, [_sk06_mlp_down_kernel_param_3];
	ld.param.b64 	%rd10, [_sk06_mlp_down_kernel_param_2];
	ld.param.b64 	%rd9, [_sk06_mlp_down_kernel_param_1];
	ld.param.b64 	%rd8, [_sk06_mlp_down_kernel_param_0];
$L__tmp0:
	.loc	1 313 24                        // sk06_mlp_down.py:313:24
	mov.u32 	%r40, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk06_mlp_down.py:314:27 ]
	add.s32 	%r41, %r15, 255;
	.loc	2 43 30                         // standard.py:43:30 @[ sk06_mlp_down.py:314:27 ]
	shr.s32 	%r42, %r41, 31;
	shr.u32 	%r43, %r42, 24;
	add.s32 	%r44, %r41, %r43;
	shr.s32 	%r45, %r44, 8;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk06_mlp_down.py:315:27 ]
	add.s32 	%r46, %r16, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk06_mlp_down.py:315:27 ]
	shr.s32 	%r47, %r46, 31;
	shr.u32 	%r48, %r47, 25;
	add.s32 	%r49, %r46, %r48;
	shr.s32 	%r50, %r49, 7;
$L__tmp3:
	.loc	1 316 29                        // sk06_mlp_down.py:316:29
	shl.b32 	%r51, %r50, 3;
	.loc	1 317 22                        // sk06_mlp_down.py:317:22
	div.s32 	%r52, %r40, %r51;
	.loc	1 317 38                        // sk06_mlp_down.py:317:38
	shl.b32 	%r53, %r52, 3;
	ld.param.b32 	%r54, [_sk06_mlp_down_kernel_param_9];
	.loc	1 318 30                        // sk06_mlp_down.py:318:30
	sub.s32 	%r55, %r45, %r53;
	ld.param.b32 	%r56, [_sk06_mlp_down_kernel_param_10];
	.loc	1 318 39                        // sk06_mlp_down.py:318:39
	min.s32 	%r57, %r55, 8;
	ld.param.b32 	%r58, [_sk06_mlp_down_kernel_param_11];
	.loc	1 319 30                        // sk06_mlp_down.py:319:30
	mul.lo.s32 	%r59, %r52, %r51;
	sub.s32 	%r60, %r40, %r59;
	.loc	1 320 36                        // sk06_mlp_down.py:320:36
	div.s32 	%r61, %r60, %r57;
	.loc	1 319 46                        // sk06_mlp_down.py:319:46
	mul.lo.s32 	%r62, %r61, %r57;
	sub.s32 	%r63, %r60, %r62;
	.loc	1 319 23                        // sk06_mlp_down.py:319:23
	add.s32 	%r64, %r63, %r53;
	.loc	1 322 22                        // sk06_mlp_down.py:322:22
	shl.b32 	%r1, %r64, 8;
	.loc	1 322 45                        // sk06_mlp_down.py:322:45
	mov.u32 	%r2, %tid.x;
	shr.u32 	%r65, %r2, 2;
	bfe.u32 	%r66, %r2, 2, 6;
	or.b32 	%r67, %r66, 64;
	and.b32 	%r3, %r2, 255;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r68, %r1, %r66;
	or.b32 	%r69, %r1, %r67;
	or.b32 	%r70, %r68, 128;
	or.b32 	%r71, %r1, %r65;
	or.b32 	%r72, %r71, 192;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r73, %r68, %r15;
	rem.s32 	%r74, %r69, %r15;
	rem.s32 	%r75, %r70, %r15;
	rem.s32 	%r76, %r72, %r15;
	.loc	1 323 22                        // sk06_mlp_down.py:323:22
	shl.b32 	%r4, %r61, 7;
	.loc	1 323 45                        // sk06_mlp_down.py:323:45
	and.b32 	%r5, %r2, 3;
	and.b32 	%r6, %r2, 32;
	and.b32 	%r7, %r2, 15;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r77, %r4, %r66;
	or.b32 	%r78, %r4, %r67;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r79, %r77, %r16;
	rem.s32 	%r80, %r78, %r16;
	.loc	1 326 39                        // sk06_mlp_down.py:326:39
	mul.lo.s32 	%r81, %r73, %r56;
	mul.lo.s32 	%r82, %r74, %r56;
	mul.lo.s32 	%r83, %r75, %r56;
	mul.lo.s32 	%r84, %r76, %r56;
	.loc	1 326 21                        // sk06_mlp_down.py:326:21
	cvt.s64.s32 	%rd1, %r81;
	add.s64 	%rd32, %rd8, %rd1;
	cvt.s64.s32 	%rd2, %r82;
	add.s64 	%rd33, %rd8, %rd2;
	cvt.s64.s32 	%rd3, %r83;
	add.s64 	%rd34, %rd8, %rd3;
	cvt.s64.s32 	%rd4, %r84;
	add.s64 	%rd35, %rd8, %rd4;
	.loc	1 326 58                        // sk06_mlp_down.py:326:58
	shl.b32 	%r85, %r5, 4;
	.loc	1 326 51                        // sk06_mlp_down.py:326:51
	cvt.u64.u32 	%rd5, %r85;
	add.s64 	%rd14, %rd32, %rd5;
	add.s64 	%rd15, %rd33, %rd5;
	add.s64 	%rd16, %rd34, %rd5;
	add.s64 	%rd17, %rd35, %rd5;
	.loc	1 327 21                        // sk06_mlp_down.py:327:21
	add.s64 	%rd36, %rd9, %rd5;
	.loc	1 327 69                        // sk06_mlp_down.py:327:69
	mul.lo.s32 	%r86, %r79, %r58;
	mul.lo.s32 	%r87, %r80, %r58;
	.loc	1 327 51                        // sk06_mlp_down.py:327:51
	cvt.s64.s32 	%rd6, %r86;
	add.s64 	%rd18, %rd36, %rd6;
	cvt.s64.s32 	%rd7, %r87;
	add.s64 	%rd19, %rd36, %rd7;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	setp.gt.s32 	%p1, %r54, 63;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	shl.b32 	%r91, %r3, 4;
	shl.b32 	%r9, %r2, 1;
	and.b32 	%r10, %r9, 48;
	xor.b32 	%r92, %r91, %r10;
	mov.b32 	%r93, global_smem;
	add.s32 	%r19, %r93, %r92;
	selp.b32 	%r20, 16, 0, %p1;
	// begin inline asm
	cp.async.cg.shared.global [ %r19 + 0 ], [ %rd14 + 0 ], 0x10, %r20;
	// end inline asm
	add.s32 	%r21, %r19, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r21 + 0 ], [ %rd15 + 0 ], 0x10, %r20;
	// end inline asm
	add.s32 	%r22, %r19, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r22 + 0 ], [ %rd16 + 0 ], 0x10, %r20;
	// end inline asm
	add.s32 	%r23, %r19, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r23 + 0 ], [ %rd17 + 0 ], 0x10, %r20;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	add.s32 	%r24, %r19, 49152;
	// begin inline asm
	cp.async.cg.shared.global [ %r24 + 0 ], [ %rd18 + 0 ], 0x10, %r20;
	// end inline asm
	add.s32 	%r25, %r19, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r25 + 0 ], [ %rd19 + 0 ], 0x10, %r20;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	setp.gt.s32 	%p2, %r54, 127;
	.loc	1 344 18                        // sk06_mlp_down.py:344:18
	add.s64 	%rd20, %rd14, 64;
	add.s64 	%rd21, %rd15, 64;
	add.s64 	%rd22, %rd16, 64;
	add.s64 	%rd23, %rd17, 64;
	.loc	1 345 18                        // sk06_mlp_down.py:345:18
	add.s64 	%rd24, %rd18, 64;
	add.s64 	%rd25, %rd19, 64;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	bar.sync 	0;
	add.s32 	%r26, %r19, 16384;
	selp.b32 	%r27, 16, 0, %p2;
	// begin inline asm
	cp.async.cg.shared.global [ %r26 + 0 ], [ %rd20 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r28, %r19, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r28 + 0 ], [ %rd21 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r29, %r19, 24576;
	// begin inline asm
	cp.async.cg.shared.global [ %r29 + 0 ], [ %rd22 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r30, %r19, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r30 + 0 ], [ %rd23 + 0 ], 0x10, %r27;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	add.s32 	%r31, %r19, 57344;
	// begin inline asm
	cp.async.cg.shared.global [ %r31 + 0 ], [ %rd24 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r32, %r19, 61440;
	// begin inline asm
	cp.async.cg.shared.global [ %r32 + 0 ], [ %rd25 + 0 ], 0x10, %r27;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	setp.gt.s32 	%p3, %r54, 191;
	.loc	1 344 18                        // sk06_mlp_down.py:344:18
	add.s64 	%rd26, %rd14, 128;
	add.s64 	%rd27, %rd15, 128;
	add.s64 	%rd28, %rd16, 128;
	add.s64 	%rd29, %rd17, 128;
	.loc	1 345 18                        // sk06_mlp_down.py:345:18
	add.s64 	%rd30, %rd18, 128;
	add.s64 	%rd31, %rd19, 128;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	bar.sync 	0;
	add.s32 	%r33, %r19, 32768;
	selp.b32 	%r34, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r33 + 0 ], [ %rd26 + 0 ], 0x10, %r34;
	// end inline asm
	add.s32 	%r35, %r19, 36864;
	// begin inline asm
	cp.async.cg.shared.global [ %r35 + 0 ], [ %rd27 + 0 ], 0x10, %r34;
	// end inline asm
	add.s32 	%r36, %r19, 40960;
	// begin inline asm
	cp.async.cg.shared.global [ %r36 + 0 ], [ %rd28 + 0 ], 0x10, %r34;
	// end inline asm
	add.s32 	%r37, %r19, 45056;
	// begin inline asm
	cp.async.cg.shared.global [ %r37 + 0 ], [ %rd29 + 0 ], 0x10, %r34;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	add.s32 	%r38, %r19, 65536;
	// begin inline asm
	cp.async.cg.shared.global [ %r38 + 0 ], [ %rd30 + 0 ], 0x10, %r34;
	// end inline asm
	add.s32 	%r39, %r19, 69632;
	// begin inline asm
	cp.async.cg.shared.global [ %r39 + 0 ], [ %rd31 + 0 ], 0x10, %r34;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	@%p1 bra 	$L__BB0_2;
	bra.uni 	$L__BB0_1;
$L__BB0_2:                              // %.lr.ph
	.loc	1 0 23                          // sk06_mlp_down.py:0:23
	shr.s32 	%r88, %r54, 31;
	shr.u32 	%r89, %r88, 26;
	add.s32 	%r90, %r54, %r89;
	shr.s32 	%r8, %r90, 6;
	add.s32 	%r11, %r8, -3;
	shl.b32 	%r94, %r7, 6;
	shl.b32 	%r1278, %r2, 4;
	and.b32 	%r95, %r1278, 3072;
	shl.b32 	%r96, %r2, 3;
	and.b32 	%r97, %r96, 48;
	and.b32 	%r1279, %r2, 16;
	or.b32 	%r98, %r94, %r95;
	xor.b32 	%r99, %r97, %r1279;
	or.b32 	%r12, %r98, %r99;
	xor.b32 	%r13, %r12, 32;
	shl.b32 	%r100, %r2, 6;
	and.b32 	%r101, %r100, 448;
	shl.b32 	%r102, %r6, 4;
	or.b32 	%r103, %r101, %r97;
	xor.b32 	%r104, %r103, %r10;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	add.s32 	%r105, %r93, %r102;
	add.s32 	%r14, %r105, %r104;
	add.s64 	%rd37, %rd7, %rd9;
	add.s64 	%rd129, %rd37, 192;
	add.s64 	%rd38, %rd6, %rd9;
	add.s64 	%rd128, %rd38, 192;
	add.s64 	%rd39, %rd4, %rd8;
	add.s64 	%rd127, %rd39, 192;
	add.s64 	%rd40, %rd3, %rd8;
	add.s64 	%rd126, %rd40, 192;
	add.s64 	%rd41, %rd2, %rd8;
	add.s64 	%rd125, %rd41, 192;
	add.s64 	%rd42, %rd1, %rd8;
	add.s64 	%rd124, %rd42, 192;
	mov.b32 	%r1280, 0f00000000;
	mov.b32 	%r106, 0;
	mov.b32 	%r1276, 2;
	mov.b32 	%r1275, -1;
	mov.b32 	%r1277, %r106;
	mov.b32 	%r1281, %r1280;
	mov.b32 	%r1282, %r1280;
	mov.b32 	%r1283, %r1280;
	mov.b32 	%r1284, %r1280;
	mov.b32 	%r1285, %r1280;
	mov.b32 	%r1286, %r1280;
	mov.b32 	%r1287, %r1280;
	mov.b32 	%r1288, %r1280;
	mov.b32 	%r1289, %r1280;
	mov.b32 	%r1290, %r1280;
	mov.b32 	%r1291, %r1280;
	mov.b32 	%r1292, %r1280;
	mov.b32 	%r1293, %r1280;
	mov.b32 	%r1294, %r1280;
	mov.b32 	%r1295, %r1280;
	mov.b32 	%r1296, %r1280;
	mov.b32 	%r1297, %r1280;
	mov.b32 	%r1298, %r1280;
	mov.b32 	%r1299, %r1280;
	mov.b32 	%r1300, %r1280;
	mov.b32 	%r1301, %r1280;
	mov.b32 	%r1302, %r1280;
	mov.b32 	%r1303, %r1280;
	mov.b32 	%r1304, %r1280;
	mov.b32 	%r1305, %r1280;
	mov.b32 	%r1306, %r1280;
	mov.b32 	%r1307, %r1280;
	mov.b32 	%r1308, %r1280;
	mov.b32 	%r1309, %r1280;
	mov.b32 	%r1310, %r1280;
	mov.b32 	%r1311, %r1280;
	mov.b32 	%r1312, %r1280;
	mov.b32 	%r1313, %r1280;
	mov.b32 	%r1314, %r1280;
	mov.b32 	%r1315, %r1280;
	mov.b32 	%r1316, %r1280;
	mov.b32 	%r1317, %r1280;
	mov.b32 	%r1318, %r1280;
	mov.b32 	%r1319, %r1280;
	mov.b32 	%r1320, %r1280;
	mov.b32 	%r1321, %r1280;
	mov.b32 	%r1322, %r1280;
	mov.b32 	%r1323, %r1280;
	mov.b32 	%r1324, %r1280;
	mov.b32 	%r1325, %r1280;
	mov.b32 	%r1326, %r1280;
	mov.b32 	%r1327, %r1280;
	mov.b32 	%r1328, %r1280;
	mov.b32 	%r1329, %r1280;
	mov.b32 	%r1330, %r1280;
	mov.b32 	%r1331, %r1280;
	mov.b32 	%r1332, %r1280;
	mov.b32 	%r1333, %r1280;
	mov.b32 	%r1334, %r1280;
	mov.b32 	%r1335, %r1280;
	mov.b32 	%r1336, %r1280;
	mov.b32 	%r1337, %r1280;
	mov.b32 	%r1338, %r1280;
	mov.b32 	%r1339, %r1280;
	mov.b32 	%r1340, %r1280;
	mov.b32 	%r1341, %r1280;
	mov.b32 	%r1342, %r1280;
	mov.b32 	%r1343, %r1280;
	mov.b32 	%r1344, %r1280;
	mov.b32 	%r1345, %r1280;
	mov.b32 	%r1346, %r1280;
	mov.b32 	%r1347, %r1280;
	mov.b32 	%r1348, %r1280;
	mov.b32 	%r1349, %r1280;
	mov.b32 	%r1350, %r1280;
	mov.b32 	%r1351, %r1280;
	mov.b32 	%r1352, %r1280;
	mov.b32 	%r1353, %r1280;
	mov.b32 	%r1354, %r1280;
	mov.b32 	%r1355, %r1280;
	mov.b32 	%r1356, %r1280;
	mov.b32 	%r1357, %r1280;
	mov.b32 	%r1358, %r1280;
	mov.b32 	%r1359, %r1280;
	mov.b32 	%r1360, %r1280;
	mov.b32 	%r1361, %r1280;
	mov.b32 	%r1362, %r1280;
	mov.b32 	%r1363, %r1280;
	mov.b32 	%r1364, %r1280;
	mov.b32 	%r1365, %r1280;
	mov.b32 	%r1366, %r1280;
	mov.b32 	%r1367, %r1280;
	mov.b32 	%r1368, %r1280;
	mov.b32 	%r1369, %r1280;
	mov.b32 	%r1370, %r1280;
	mov.b32 	%r1371, %r1280;
	mov.b32 	%r1372, %r1280;
	mov.b32 	%r1373, %r1280;
	mov.b32 	%r1374, %r1280;
	mov.b32 	%r1375, %r1280;
	mov.b32 	%r1376, %r1280;
	mov.b32 	%r1377, %r1280;
	mov.b32 	%r1378, %r1280;
	mov.b32 	%r1379, %r1280;
	mov.b32 	%r1380, %r1280;
	mov.b32 	%r1381, %r1280;
	mov.b32 	%r1382, %r1280;
	mov.b32 	%r1383, %r1280;
	mov.b32 	%r1384, %r1280;
	mov.b32 	%r1385, %r1280;
	mov.b32 	%r1386, %r1280;
	mov.b32 	%r1387, %r1280;
	mov.b32 	%r1388, %r1280;
	mov.b32 	%r1389, %r1280;
	mov.b32 	%r1390, %r1280;
	mov.b32 	%r1391, %r1280;
	mov.b32 	%r1392, %r1280;
	mov.b32 	%r1393, %r1280;
	mov.b32 	%r1394, %r1280;
	mov.b32 	%r1395, %r1280;
	mov.b32 	%r1396, %r1280;
	mov.b32 	%r1397, %r1280;
	mov.b32 	%r1398, %r1280;
	mov.b32 	%r1399, %r1280;
	mov.b32 	%r1400, %r1280;
	mov.b32 	%r1401, %r1280;
	mov.b32 	%r1402, %r1280;
	mov.b32 	%r1403, %r1280;
	mov.b32 	%r1404, %r1280;
	mov.b32 	%r1405, %r1280;
	mov.b32 	%r1406, %r1280;
	mov.b32 	%r1407, %r1280;
$L__BB0_3:                              // =>This Inner Loop Header: Depth=1
	setp.lt.s32 	%p4, %r1277, %r11;
	add.s32 	%r306, %r1275, 1;
	setp.gt.s32 	%p5, %r306, 2;
	selp.b32 	%r1275, 0, %r306, %p5;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	cp.async.wait_group 	4;
	bar.sync 	0;
	shl.b32 	%r307, %r1275, 14;
	add.s32 	%r308, %r93, %r307;
	add.s32 	%r309, %r308, %r12;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r107, %r108, %r109, %r110}, [%r309];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r127, %r128, %r129, %r130}, [%r309+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r131, %r132, %r133, %r134}, [%r309+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r135, %r136, %r137, %r138}, [%r309+12288];
	add.s32 	%r310, %r308, %r13;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r143, %r144, %r145, %r146}, [%r310];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r195, %r196, %r197, %r198}, [%r310+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r231, %r232, %r233, %r234}, [%r310+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r267, %r268, %r269, %r270}, [%r310+12288];
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	shl.b32 	%r311, %r1275, 13;
	add.s32 	%r312, %r14, %r311;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r111, %r112, %r147, %r148}, [%r312+49152];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r113, %r114, %r153, %r154}, [%r312+50176];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r115, %r116, %r159, %r160}, [%r312+51200];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r117, %r118, %r165, %r166}, [%r312+52224];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r119, %r120, %r171, %r172}, [%r312+53248];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r121, %r122, %r177, %r178}, [%r312+54272];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r123, %r124, %r183, %r184}, [%r312+55296];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r125, %r126, %r189, %r190}, [%r312+56320];
	.loc	1 338 36                        // sk06_mlp_down.py:338:36
	mov.b32 	%r139, %r106;
	mov.b32 	%r140, %r106;
	mov.b32 	%r141, %r106;
	mov.b32 	%r142, %r106;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r139, %r140, %r141, %r142 }, { %r107, %r108, %r109, %r110 }, { %r111, %r112 }, { %r139, %r140, %r141, %r142 };
	// end inline asm
	mov.b32 	%r149, %r106;
	mov.b32 	%r150, %r106;
	mov.b32 	%r151, %r106;
	mov.b32 	%r152, %r106;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r149, %r150, %r151, %r152 }, { %r107, %r108, %r109, %r110 }, { %r113, %r114 }, { %r149, %r150, %r151, %r152 };
	// end inline asm
	mov.b32 	%r155, %r106;
	mov.b32 	%r156, %r106;
	mov.b32 	%r157, %r106;
	mov.b32 	%r158, %r106;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r155, %r156, %r157, %r158 }, { %r107, %r108, %r109, %r110 }, { %r115, %r116 }, { %r155, %r156, %r157, %r158 };
	// end inline asm
	mov.b32 	%r161, %r106;
	mov.b32 	%r162, %r106;
	mov.b32 	%r163, %r106;
	mov.b32 	%r164, %r106;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r161, %r162, %r163, %r164 }, { %r107, %r108, %r109, %r110 }, { %r117, %r118 }, { %r161, %r162, %r163, %r164 };
	// end inline asm
	mov.b32 	%r167, %r106;
	mov.b32 	%r168, %r106;
	mov.b32 	%r169, %r106;
	mov.b32 	%r170, %r106;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r167, %r168, %r169, %r170 }, { %r107, %r108, %r109, %r110 }, { %r119, %r120 }, { %r167, %r168, %r169, %r170 };
	// end inline asm
	mov.b32 	%r173, %r106;
	mov.b32 	%r174, %r106;
	mov.b32 	%r175, %r106;
	mov.b32 	%r176, %r106;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r173, %r174, %r175, %r176 }, { %r107, %r108, %r109, %r110 }, { %r121, %r122 }, { %r173, %r174, %r175, %r176 };
	// end inline asm
	mov.b32 	%r179, %r106;
	mov.b32 	%r180, %r106;
	mov.b32 	%r181, %r106;
	mov.b32 	%r182, %r106;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r179, %r180, %r181, %r182 }, { %r107, %r108, %r109, %r110 }, { %r123, %r124 }, { %r179, %r180, %r181, %r182 };
	// end inline asm
	mov.b32 	%r185, %r106;
	mov.b32 	%r186, %r106;
	mov.b32 	%r187, %r106;
	mov.b32 	%r188, %r106;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r185, %r186, %r187, %r188 }, { %r107, %r108, %r109, %r110 }, { %r125, %r126 }, { %r185, %r186, %r187, %r188 };
	// end inline asm
	mov.b32 	%r191, %r106;
	mov.b32 	%r192, %r106;
	mov.b32 	%r193, %r106;
	mov.b32 	%r194, %r106;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r191, %r192, %r193, %r194 }, { %r127, %r128, %r129, %r130 }, { %r111, %r112 }, { %r191, %r192, %r193, %r194 };
	// end inline asm
	mov.b32 	%r199, %r106;
	mov.b32 	%r200, %r106;
	mov.b32 	%r201, %r106;
	mov.b32 	%r202, %r106;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r199, %r200, %r201, %r202 }, { %r127, %r128, %r129, %r130 }, { %r113, %r114 }, { %r199, %r200, %r201, %r202 };
	// end inline asm
	mov.b32 	%r203, %r106;
	mov.b32 	%r204, %r106;
	mov.b32 	%r205, %r106;
	mov.b32 	%r206, %r106;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r203, %r204, %r205, %r206 }, { %r127, %r128, %r129, %r130 }, { %r115, %r116 }, { %r203, %r204, %r205, %r206 };
	// end inline asm
	mov.b32 	%r207, %r106;
	mov.b32 	%r208, %r106;
	mov.b32 	%r209, %r106;
	mov.b32 	%r210, %r106;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r207, %r208, %r209, %r210 }, { %r127, %r128, %r129, %r130 }, { %r117, %r118 }, { %r207, %r208, %r209, %r210 };
	// end inline asm
	mov.b32 	%r211, %r106;
	mov.b32 	%r212, %r106;
	mov.b32 	%r213, %r106;
	mov.b32 	%r214, %r106;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r211, %r212, %r213, %r214 }, { %r127, %r128, %r129, %r130 }, { %r119, %r120 }, { %r211, %r212, %r213, %r214 };
	// end inline asm
	mov.b32 	%r215, %r106;
	mov.b32 	%r216, %r106;
	mov.b32 	%r217, %r106;
	mov.b32 	%r218, %r106;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r215, %r216, %r217, %r218 }, { %r127, %r128, %r129, %r130 }, { %r121, %r122 }, { %r215, %r216, %r217, %r218 };
	// end inline asm
	mov.b32 	%r219, %r106;
	mov.b32 	%r220, %r106;
	mov.b32 	%r221, %r106;
	mov.b32 	%r222, %r106;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r219, %r220, %r221, %r222 }, { %r127, %r128, %r129, %r130 }, { %r123, %r124 }, { %r219, %r220, %r221, %r222 };
	// end inline asm
	mov.b32 	%r223, %r106;
	mov.b32 	%r224, %r106;
	mov.b32 	%r225, %r106;
	mov.b32 	%r226, %r106;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r223, %r224, %r225, %r226 }, { %r127, %r128, %r129, %r130 }, { %r125, %r126 }, { %r223, %r224, %r225, %r226 };
	// end inline asm
	mov.b32 	%r227, %r106;
	mov.b32 	%r228, %r106;
	mov.b32 	%r229, %r106;
	mov.b32 	%r230, %r106;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r227, %r228, %r229, %r230 }, { %r131, %r132, %r133, %r134 }, { %r111, %r112 }, { %r227, %r228, %r229, %r230 };
	// end inline asm
	mov.b32 	%r235, %r106;
	mov.b32 	%r236, %r106;
	mov.b32 	%r237, %r106;
	mov.b32 	%r238, %r106;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r235, %r236, %r237, %r238 }, { %r131, %r132, %r133, %r134 }, { %r113, %r114 }, { %r235, %r236, %r237, %r238 };
	// end inline asm
	mov.b32 	%r239, %r106;
	mov.b32 	%r240, %r106;
	mov.b32 	%r241, %r106;
	mov.b32 	%r242, %r106;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r239, %r240, %r241, %r242 }, { %r131, %r132, %r133, %r134 }, { %r115, %r116 }, { %r239, %r240, %r241, %r242 };
	// end inline asm
	mov.b32 	%r243, %r106;
	mov.b32 	%r244, %r106;
	mov.b32 	%r245, %r106;
	mov.b32 	%r246, %r106;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r243, %r244, %r245, %r246 }, { %r131, %r132, %r133, %r134 }, { %r117, %r118 }, { %r243, %r244, %r245, %r246 };
	// end inline asm
	mov.b32 	%r247, %r106;
	mov.b32 	%r248, %r106;
	mov.b32 	%r249, %r106;
	mov.b32 	%r250, %r106;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r247, %r248, %r249, %r250 }, { %r131, %r132, %r133, %r134 }, { %r119, %r120 }, { %r247, %r248, %r249, %r250 };
	// end inline asm
	mov.b32 	%r251, %r106;
	mov.b32 	%r252, %r106;
	mov.b32 	%r253, %r106;
	mov.b32 	%r254, %r106;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r251, %r252, %r253, %r254 }, { %r131, %r132, %r133, %r134 }, { %r121, %r122 }, { %r251, %r252, %r253, %r254 };
	// end inline asm
	mov.b32 	%r255, %r106;
	mov.b32 	%r256, %r106;
	mov.b32 	%r257, %r106;
	mov.b32 	%r258, %r106;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r255, %r256, %r257, %r258 }, { %r131, %r132, %r133, %r134 }, { %r123, %r124 }, { %r255, %r256, %r257, %r258 };
	// end inline asm
	mov.b32 	%r259, %r106;
	mov.b32 	%r260, %r106;
	mov.b32 	%r261, %r106;
	mov.b32 	%r262, %r106;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r259, %r260, %r261, %r262 }, { %r131, %r132, %r133, %r134 }, { %r125, %r126 }, { %r259, %r260, %r261, %r262 };
	// end inline asm
	mov.b32 	%r263, %r106;
	mov.b32 	%r264, %r106;
	mov.b32 	%r265, %r106;
	mov.b32 	%r266, %r106;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r263, %r264, %r265, %r266 }, { %r135, %r136, %r137, %r138 }, { %r111, %r112 }, { %r263, %r264, %r265, %r266 };
	// end inline asm
	mov.b32 	%r271, %r106;
	mov.b32 	%r272, %r106;
	mov.b32 	%r273, %r106;
	mov.b32 	%r274, %r106;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r271, %r272, %r273, %r274 }, { %r135, %r136, %r137, %r138 }, { %r113, %r114 }, { %r271, %r272, %r273, %r274 };
	// end inline asm
	mov.b32 	%r275, %r106;
	mov.b32 	%r276, %r106;
	mov.b32 	%r277, %r106;
	mov.b32 	%r278, %r106;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r275, %r276, %r277, %r278 }, { %r135, %r136, %r137, %r138 }, { %r115, %r116 }, { %r275, %r276, %r277, %r278 };
	// end inline asm
	mov.b32 	%r279, %r106;
	mov.b32 	%r280, %r106;
	mov.b32 	%r281, %r106;
	mov.b32 	%r282, %r106;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r279, %r280, %r281, %r282 }, { %r135, %r136, %r137, %r138 }, { %r117, %r118 }, { %r279, %r280, %r281, %r282 };
	// end inline asm
	mov.b32 	%r283, %r106;
	mov.b32 	%r284, %r106;
	mov.b32 	%r285, %r106;
	mov.b32 	%r286, %r106;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r283, %r284, %r285, %r286 }, { %r135, %r136, %r137, %r138 }, { %r119, %r120 }, { %r283, %r284, %r285, %r286 };
	// end inline asm
	mov.b32 	%r287, %r106;
	mov.b32 	%r288, %r106;
	mov.b32 	%r289, %r106;
	mov.b32 	%r290, %r106;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r287, %r288, %r289, %r290 }, { %r135, %r136, %r137, %r138 }, { %r121, %r122 }, { %r287, %r288, %r289, %r290 };
	// end inline asm
	mov.b32 	%r291, %r106;
	mov.b32 	%r292, %r106;
	mov.b32 	%r293, %r106;
	mov.b32 	%r294, %r106;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r291, %r292, %r293, %r294 }, { %r135, %r136, %r137, %r138 }, { %r123, %r124 }, { %r291, %r292, %r293, %r294 };
	// end inline asm
	mov.b32 	%r295, %r106;
	mov.b32 	%r296, %r106;
	mov.b32 	%r297, %r106;
	mov.b32 	%r298, %r106;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r295, %r296, %r297, %r298 }, { %r135, %r136, %r137, %r138 }, { %r125, %r126 }, { %r295, %r296, %r297, %r298 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r139, %r140, %r141, %r142 }, { %r143, %r144, %r145, %r146 }, { %r147, %r148 }, { %r139, %r140, %r141, %r142 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r149, %r150, %r151, %r152 }, { %r143, %r144, %r145, %r146 }, { %r153, %r154 }, { %r149, %r150, %r151, %r152 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r155, %r156, %r157, %r158 }, { %r143, %r144, %r145, %r146 }, { %r159, %r160 }, { %r155, %r156, %r157, %r158 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r161, %r162, %r163, %r164 }, { %r143, %r144, %r145, %r146 }, { %r165, %r166 }, { %r161, %r162, %r163, %r164 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r167, %r168, %r169, %r170 }, { %r143, %r144, %r145, %r146 }, { %r171, %r172 }, { %r167, %r168, %r169, %r170 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r173, %r174, %r175, %r176 }, { %r143, %r144, %r145, %r146 }, { %r177, %r178 }, { %r173, %r174, %r175, %r176 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r179, %r180, %r181, %r182 }, { %r143, %r144, %r145, %r146 }, { %r183, %r184 }, { %r179, %r180, %r181, %r182 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r185, %r186, %r187, %r188 }, { %r143, %r144, %r145, %r146 }, { %r189, %r190 }, { %r185, %r186, %r187, %r188 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r191, %r192, %r193, %r194 }, { %r195, %r196, %r197, %r198 }, { %r147, %r148 }, { %r191, %r192, %r193, %r194 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r199, %r200, %r201, %r202 }, { %r195, %r196, %r197, %r198 }, { %r153, %r154 }, { %r199, %r200, %r201, %r202 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r203, %r204, %r205, %r206 }, { %r195, %r196, %r197, %r198 }, { %r159, %r160 }, { %r203, %r204, %r205, %r206 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r207, %r208, %r209, %r210 }, { %r195, %r196, %r197, %r198 }, { %r165, %r166 }, { %r207, %r208, %r209, %r210 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r211, %r212, %r213, %r214 }, { %r195, %r196, %r197, %r198 }, { %r171, %r172 }, { %r211, %r212, %r213, %r214 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r215, %r216, %r217, %r218 }, { %r195, %r196, %r197, %r198 }, { %r177, %r178 }, { %r215, %r216, %r217, %r218 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r219, %r220, %r221, %r222 }, { %r195, %r196, %r197, %r198 }, { %r183, %r184 }, { %r219, %r220, %r221, %r222 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r223, %r224, %r225, %r226 }, { %r195, %r196, %r197, %r198 }, { %r189, %r190 }, { %r223, %r224, %r225, %r226 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r227, %r228, %r229, %r230 }, { %r231, %r232, %r233, %r234 }, { %r147, %r148 }, { %r227, %r228, %r229, %r230 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r235, %r236, %r237, %r238 }, { %r231, %r232, %r233, %r234 }, { %r153, %r154 }, { %r235, %r236, %r237, %r238 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r239, %r240, %r241, %r242 }, { %r231, %r232, %r233, %r234 }, { %r159, %r160 }, { %r239, %r240, %r241, %r242 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r243, %r244, %r245, %r246 }, { %r231, %r232, %r233, %r234 }, { %r165, %r166 }, { %r243, %r244, %r245, %r246 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r247, %r248, %r249, %r250 }, { %r231, %r232, %r233, %r234 }, { %r171, %r172 }, { %r247, %r248, %r249, %r250 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r251, %r252, %r253, %r254 }, { %r231, %r232, %r233, %r234 }, { %r177, %r178 }, { %r251, %r252, %r253, %r254 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r255, %r256, %r257, %r258 }, { %r231, %r232, %r233, %r234 }, { %r183, %r184 }, { %r255, %r256, %r257, %r258 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r259, %r260, %r261, %r262 }, { %r231, %r232, %r233, %r234 }, { %r189, %r190 }, { %r259, %r260, %r261, %r262 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r263, %r264, %r265, %r266 }, { %r267, %r268, %r269, %r270 }, { %r147, %r148 }, { %r263, %r264, %r265, %r266 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r271, %r272, %r273, %r274 }, { %r267, %r268, %r269, %r270 }, { %r153, %r154 }, { %r271, %r272, %r273, %r274 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r275, %r276, %r277, %r278 }, { %r267, %r268, %r269, %r270 }, { %r159, %r160 }, { %r275, %r276, %r277, %r278 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r279, %r280, %r281, %r282 }, { %r267, %r268, %r269, %r270 }, { %r165, %r166 }, { %r279, %r280, %r281, %r282 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r283, %r284, %r285, %r286 }, { %r267, %r268, %r269, %r270 }, { %r171, %r172 }, { %r283, %r284, %r285, %r286 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r287, %r288, %r289, %r290 }, { %r267, %r268, %r269, %r270 }, { %r177, %r178 }, { %r287, %r288, %r289, %r290 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r291, %r292, %r293, %r294 }, { %r267, %r268, %r269, %r270 }, { %r183, %r184 }, { %r291, %r292, %r293, %r294 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r295, %r296, %r297, %r298 }, { %r267, %r268, %r269, %r270 }, { %r189, %r190 }, { %r295, %r296, %r297, %r298 };
	// end inline asm
	.loc	1 343 19                        // sk06_mlp_down.py:343:19
	cvt.rn.f32.s32 	%r313, %r298;
	cvt.rn.f32.s32 	%r314, %r297;
	cvt.rn.f32.s32 	%r315, %r296;
	cvt.rn.f32.s32 	%r316, %r295;
	cvt.rn.f32.s32 	%r317, %r294;
	cvt.rn.f32.s32 	%r318, %r293;
	cvt.rn.f32.s32 	%r319, %r292;
	cvt.rn.f32.s32 	%r320, %r291;
	cvt.rn.f32.s32 	%r321, %r290;
	cvt.rn.f32.s32 	%r322, %r289;
	cvt.rn.f32.s32 	%r323, %r288;
	cvt.rn.f32.s32 	%r324, %r287;
	cvt.rn.f32.s32 	%r325, %r286;
	cvt.rn.f32.s32 	%r326, %r285;
	cvt.rn.f32.s32 	%r327, %r284;
	cvt.rn.f32.s32 	%r328, %r283;
	cvt.rn.f32.s32 	%r329, %r282;
	cvt.rn.f32.s32 	%r330, %r281;
	cvt.rn.f32.s32 	%r331, %r280;
	cvt.rn.f32.s32 	%r332, %r279;
	cvt.rn.f32.s32 	%r333, %r278;
	cvt.rn.f32.s32 	%r334, %r277;
	cvt.rn.f32.s32 	%r335, %r276;
	cvt.rn.f32.s32 	%r336, %r275;
	cvt.rn.f32.s32 	%r337, %r274;
	cvt.rn.f32.s32 	%r338, %r273;
	cvt.rn.f32.s32 	%r339, %r272;
	cvt.rn.f32.s32 	%r340, %r271;
	cvt.rn.f32.s32 	%r341, %r266;
	cvt.rn.f32.s32 	%r342, %r265;
	cvt.rn.f32.s32 	%r343, %r264;
	cvt.rn.f32.s32 	%r344, %r263;
	cvt.rn.f32.s32 	%r345, %r262;
	cvt.rn.f32.s32 	%r346, %r261;
	cvt.rn.f32.s32 	%r347, %r260;
	cvt.rn.f32.s32 	%r348, %r259;
	cvt.rn.f32.s32 	%r349, %r258;
	cvt.rn.f32.s32 	%r350, %r257;
	cvt.rn.f32.s32 	%r351, %r256;
	cvt.rn.f32.s32 	%r352, %r255;
	cvt.rn.f32.s32 	%r353, %r254;
	cvt.rn.f32.s32 	%r354, %r253;
	cvt.rn.f32.s32 	%r355, %r252;
	cvt.rn.f32.s32 	%r356, %r251;
	cvt.rn.f32.s32 	%r357, %r250;
	cvt.rn.f32.s32 	%r358, %r249;
	cvt.rn.f32.s32 	%r359, %r248;
	cvt.rn.f32.s32 	%r360, %r247;
	cvt.rn.f32.s32 	%r361, %r246;
	cvt.rn.f32.s32 	%r362, %r245;
	cvt.rn.f32.s32 	%r363, %r244;
	cvt.rn.f32.s32 	%r364, %r243;
	cvt.rn.f32.s32 	%r365, %r242;
	cvt.rn.f32.s32 	%r366, %r239;
	cvt.rn.f32.s32 	%r367, %r240;
	cvt.rn.f32.s32 	%r368, %r241;
	cvt.rn.f32.s32 	%r369, %r235;
	cvt.rn.f32.s32 	%r370, %r236;
	cvt.rn.f32.s32 	%r371, %r237;
	cvt.rn.f32.s32 	%r372, %r238;
	cvt.rn.f32.s32 	%r373, %r227;
	cvt.rn.f32.s32 	%r374, %r228;
	cvt.rn.f32.s32 	%r375, %r229;
	cvt.rn.f32.s32 	%r376, %r230;
	cvt.rn.f32.s32 	%r377, %r223;
	cvt.rn.f32.s32 	%r378, %r224;
	cvt.rn.f32.s32 	%r379, %r225;
	cvt.rn.f32.s32 	%r380, %r226;
	cvt.rn.f32.s32 	%r381, %r219;
	cvt.rn.f32.s32 	%r382, %r220;
	cvt.rn.f32.s32 	%r383, %r221;
	cvt.rn.f32.s32 	%r384, %r222;
	cvt.rn.f32.s32 	%r385, %r215;
	cvt.rn.f32.s32 	%r386, %r216;
	cvt.rn.f32.s32 	%r387, %r217;
	cvt.rn.f32.s32 	%r388, %r218;
	cvt.rn.f32.s32 	%r389, %r211;
	cvt.rn.f32.s32 	%r390, %r212;
	cvt.rn.f32.s32 	%r391, %r213;
	cvt.rn.f32.s32 	%r392, %r214;
	cvt.rn.f32.s32 	%r393, %r207;
	cvt.rn.f32.s32 	%r394, %r208;
	cvt.rn.f32.s32 	%r395, %r209;
	cvt.rn.f32.s32 	%r396, %r210;
	cvt.rn.f32.s32 	%r397, %r203;
	cvt.rn.f32.s32 	%r398, %r204;
	cvt.rn.f32.s32 	%r399, %r205;
	cvt.rn.f32.s32 	%r400, %r206;
	cvt.rn.f32.s32 	%r401, %r199;
	cvt.rn.f32.s32 	%r402, %r200;
	cvt.rn.f32.s32 	%r403, %r201;
	cvt.rn.f32.s32 	%r404, %r202;
	cvt.rn.f32.s32 	%r405, %r191;
	cvt.rn.f32.s32 	%r406, %r192;
	cvt.rn.f32.s32 	%r407, %r193;
	cvt.rn.f32.s32 	%r408, %r194;
	cvt.rn.f32.s32 	%r409, %r185;
	cvt.rn.f32.s32 	%r410, %r186;
	cvt.rn.f32.s32 	%r411, %r187;
	cvt.rn.f32.s32 	%r412, %r188;
	cvt.rn.f32.s32 	%r413, %r179;
	cvt.rn.f32.s32 	%r414, %r180;
	cvt.rn.f32.s32 	%r415, %r181;
	cvt.rn.f32.s32 	%r416, %r182;
	cvt.rn.f32.s32 	%r417, %r173;
	cvt.rn.f32.s32 	%r418, %r174;
	cvt.rn.f32.s32 	%r419, %r175;
	cvt.rn.f32.s32 	%r420, %r176;
	cvt.rn.f32.s32 	%r421, %r167;
	cvt.rn.f32.s32 	%r422, %r168;
	cvt.rn.f32.s32 	%r423, %r169;
	cvt.rn.f32.s32 	%r424, %r170;
	cvt.rn.f32.s32 	%r425, %r161;
	cvt.rn.f32.s32 	%r426, %r162;
	cvt.rn.f32.s32 	%r427, %r163;
	cvt.rn.f32.s32 	%r428, %r164;
	cvt.rn.f32.s32 	%r429, %r155;
	cvt.rn.f32.s32 	%r430, %r156;
	cvt.rn.f32.s32 	%r431, %r157;
	cvt.rn.f32.s32 	%r432, %r158;
	cvt.rn.f32.s32 	%r433, %r149;
	cvt.rn.f32.s32 	%r434, %r150;
	cvt.rn.f32.s32 	%r435, %r151;
	cvt.rn.f32.s32 	%r436, %r152;
	cvt.rn.f32.s32 	%r437, %r139;
	cvt.rn.f32.s32 	%r438, %r140;
	cvt.rn.f32.s32 	%r439, %r141;
	cvt.rn.f32.s32 	%r440, %r142;
	add.f32 	%r1283, %r1283, %r440;
	add.f32 	%r1282, %r1282, %r439;
	add.f32 	%r1281, %r1281, %r438;
	add.f32 	%r1280, %r1280, %r437;
	add.f32 	%r1287, %r1287, %r436;
	add.f32 	%r1286, %r1286, %r435;
	add.f32 	%r1285, %r1285, %r434;
	add.f32 	%r1284, %r1284, %r433;
	add.f32 	%r1291, %r1291, %r432;
	add.f32 	%r1290, %r1290, %r431;
	add.f32 	%r1289, %r1289, %r430;
	add.f32 	%r1288, %r1288, %r429;
	add.f32 	%r1295, %r1295, %r428;
	add.f32 	%r1294, %r1294, %r427;
	add.f32 	%r1293, %r1293, %r426;
	add.f32 	%r1292, %r1292, %r425;
	add.f32 	%r1299, %r1299, %r424;
	add.f32 	%r1298, %r1298, %r423;
	add.f32 	%r1297, %r1297, %r422;
	add.f32 	%r1296, %r1296, %r421;
	add.f32 	%r1303, %r1303, %r420;
	add.f32 	%r1302, %r1302, %r419;
	add.f32 	%r1301, %r1301, %r418;
	add.f32 	%r1300, %r1300, %r417;
	add.f32 	%r1307, %r1307, %r416;
	add.f32 	%r1306, %r1306, %r415;
	add.f32 	%r1305, %r1305, %r414;
	add.f32 	%r1304, %r1304, %r413;
	add.f32 	%r1311, %r1311, %r412;
	add.f32 	%r1310, %r1310, %r411;
	add.f32 	%r1309, %r1309, %r410;
	add.f32 	%r1308, %r1308, %r409;
	add.f32 	%r1315, %r1315, %r408;
	add.f32 	%r1314, %r1314, %r407;
	add.f32 	%r1313, %r1313, %r406;
	add.f32 	%r1312, %r1312, %r405;
	add.f32 	%r1319, %r1319, %r404;
	add.f32 	%r1318, %r1318, %r403;
	add.f32 	%r1317, %r1317, %r402;
	add.f32 	%r1316, %r1316, %r401;
	add.f32 	%r1323, %r1323, %r400;
	add.f32 	%r1322, %r1322, %r399;
	add.f32 	%r1321, %r1321, %r398;
	add.f32 	%r1320, %r1320, %r397;
	add.f32 	%r1327, %r1327, %r396;
	add.f32 	%r1326, %r1326, %r395;
	add.f32 	%r1325, %r1325, %r394;
	add.f32 	%r1324, %r1324, %r393;
	add.f32 	%r1331, %r1331, %r392;
	add.f32 	%r1330, %r1330, %r391;
	add.f32 	%r1329, %r1329, %r390;
	add.f32 	%r1328, %r1328, %r389;
	add.f32 	%r1335, %r1335, %r388;
	add.f32 	%r1334, %r1334, %r387;
	add.f32 	%r1333, %r1333, %r386;
	add.f32 	%r1332, %r1332, %r385;
	add.f32 	%r1339, %r1339, %r384;
	add.f32 	%r1338, %r1338, %r383;
	add.f32 	%r1337, %r1337, %r382;
	add.f32 	%r1336, %r1336, %r381;
	add.f32 	%r1343, %r1343, %r380;
	add.f32 	%r1342, %r1342, %r379;
	add.f32 	%r1341, %r1341, %r378;
	add.f32 	%r1340, %r1340, %r377;
	add.f32 	%r1347, %r1347, %r376;
	add.f32 	%r1346, %r1346, %r375;
	add.f32 	%r1345, %r1345, %r374;
	add.f32 	%r1344, %r1344, %r373;
	add.f32 	%r1351, %r1351, %r372;
	add.f32 	%r1350, %r1350, %r371;
	add.f32 	%r1349, %r1349, %r370;
	add.f32 	%r1348, %r1348, %r369;
	add.f32 	%r1354, %r1354, %r368;
	add.f32 	%r1353, %r1353, %r367;
	add.f32 	%r1352, %r1352, %r366;
	add.f32 	%r1355, %r1355, %r365;
	add.f32 	%r1356, %r1356, %r364;
	add.f32 	%r1357, %r1357, %r363;
	add.f32 	%r1358, %r1358, %r362;
	add.f32 	%r1359, %r1359, %r361;
	add.f32 	%r1360, %r1360, %r360;
	add.f32 	%r1361, %r1361, %r359;
	add.f32 	%r1362, %r1362, %r358;
	add.f32 	%r1363, %r1363, %r357;
	add.f32 	%r1364, %r1364, %r356;
	add.f32 	%r1365, %r1365, %r355;
	add.f32 	%r1366, %r1366, %r354;
	add.f32 	%r1367, %r1367, %r353;
	add.f32 	%r1368, %r1368, %r352;
	add.f32 	%r1369, %r1369, %r351;
	add.f32 	%r1370, %r1370, %r350;
	add.f32 	%r1371, %r1371, %r349;
	add.f32 	%r1372, %r1372, %r348;
	add.f32 	%r1373, %r1373, %r347;
	add.f32 	%r1374, %r1374, %r346;
	add.f32 	%r1375, %r1375, %r345;
	add.f32 	%r1376, %r1376, %r344;
	add.f32 	%r1377, %r1377, %r343;
	add.f32 	%r1378, %r1378, %r342;
	add.f32 	%r1379, %r1379, %r341;
	add.f32 	%r1380, %r1380, %r340;
	add.f32 	%r1381, %r1381, %r339;
	add.f32 	%r1382, %r1382, %r338;
	add.f32 	%r1383, %r1383, %r337;
	add.f32 	%r1384, %r1384, %r336;
	add.f32 	%r1385, %r1385, %r335;
	add.f32 	%r1386, %r1386, %r334;
	add.f32 	%r1387, %r1387, %r333;
	add.f32 	%r1388, %r1388, %r332;
	add.f32 	%r1389, %r1389, %r331;
	add.f32 	%r1390, %r1390, %r330;
	add.f32 	%r1391, %r1391, %r329;
	add.f32 	%r1392, %r1392, %r328;
	add.f32 	%r1393, %r1393, %r327;
	add.f32 	%r1394, %r1394, %r326;
	add.f32 	%r1395, %r1395, %r325;
	add.f32 	%r1396, %r1396, %r324;
	add.f32 	%r1397, %r1397, %r323;
	add.f32 	%r1398, %r1398, %r322;
	add.f32 	%r1399, %r1399, %r321;
	add.f32 	%r1400, %r1400, %r320;
	add.f32 	%r1401, %r1401, %r319;
	add.f32 	%r1402, %r1402, %r318;
	add.f32 	%r1403, %r1403, %r317;
	add.f32 	%r1404, %r1404, %r316;
	add.f32 	%r1405, %r1405, %r315;
	add.f32 	%r1406, %r1406, %r314;
	add.f32 	%r1407, %r1407, %r313;
	.loc	1 344 18                        // sk06_mlp_down.py:344:18
	add.s64 	%rd43, %rd124, %rd5;
	add.s64 	%rd44, %rd125, %rd5;
	add.s64 	%rd45, %rd126, %rd5;
	.loc	1 345 18                        // sk06_mlp_down.py:345:18
	add.s64 	%rd46, %rd127, %rd5;
	add.s64 	%rd47, %rd128, %rd5;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	add.s64 	%rd48, %rd129, %rd5;
	add.s32 	%r441, %r1276, 1;
	setp.gt.s32 	%p6, %r441, 2;
	selp.b32 	%r1276, 0, %r441, %p6;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	shl.b32 	%r442, %r1276, 14;
	bar.sync 	0;
	add.s32 	%r299, %r19, %r442;
	selp.b32 	%r300, 16, 0, %p4;
	// begin inline asm
	cp.async.cg.shared.global [ %r299 + 0 ], [ %rd43 + 0 ], 0x10, %r300;
	// end inline asm
	add.s32 	%r301, %r299, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r301 + 0 ], [ %rd44 + 0 ], 0x10, %r300;
	// end inline asm
	add.s32 	%r302, %r299, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r302 + 0 ], [ %rd45 + 0 ], 0x10, %r300;
	// end inline asm
	add.s32 	%r303, %r299, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r303 + 0 ], [ %rd46 + 0 ], 0x10, %r300;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	shl.b32 	%r443, %r1276, 13;
	add.s32 	%r444, %r19, %r443;
	add.s32 	%r304, %r444, 49152;
	// begin inline asm
	cp.async.cg.shared.global [ %r304 + 0 ], [ %rd47 + 0 ], 0x10, %r300;
	// end inline asm
	add.s32 	%r305, %r444, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r305 + 0 ], [ %rd48 + 0 ], 0x10, %r300;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	add.s32 	%r1277, %r1277, 1;
	add.s64 	%rd129, %rd129, 64;
	add.s64 	%rd128, %rd128, 64;
	add.s64 	%rd127, %rd127, 64;
	add.s64 	%rd126, %rd126, 64;
	add.s64 	%rd125, %rd125, 64;
	add.s64 	%rd124, %rd124, 64;
	setp.ne.b32 	%p7, %r8, %r1277;
	@%p7 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_4;
$L__BB0_1:                              // %.._crit_edge_crit_edge
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	and.b32 	%r1279, %r2, 16;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	shl.b32 	%r1278, %r2, 4;
	mov.b32 	%r1280, 0f00000000;
	mov.b32 	%r1281, %r1280;
	mov.b32 	%r1282, %r1280;
	mov.b32 	%r1283, %r1280;
	mov.b32 	%r1284, %r1280;
	mov.b32 	%r1285, %r1280;
	mov.b32 	%r1286, %r1280;
	mov.b32 	%r1287, %r1280;
	mov.b32 	%r1288, %r1280;
	mov.b32 	%r1289, %r1280;
	mov.b32 	%r1290, %r1280;
	mov.b32 	%r1291, %r1280;
	mov.b32 	%r1292, %r1280;
	mov.b32 	%r1293, %r1280;
	mov.b32 	%r1294, %r1280;
	mov.b32 	%r1295, %r1280;
	mov.b32 	%r1296, %r1280;
	mov.b32 	%r1297, %r1280;
	mov.b32 	%r1298, %r1280;
	mov.b32 	%r1299, %r1280;
	mov.b32 	%r1300, %r1280;
	mov.b32 	%r1301, %r1280;
	mov.b32 	%r1302, %r1280;
	mov.b32 	%r1303, %r1280;
	mov.b32 	%r1304, %r1280;
	mov.b32 	%r1305, %r1280;
	mov.b32 	%r1306, %r1280;
	mov.b32 	%r1307, %r1280;
	mov.b32 	%r1308, %r1280;
	mov.b32 	%r1309, %r1280;
	mov.b32 	%r1310, %r1280;
	mov.b32 	%r1311, %r1280;
	mov.b32 	%r1312, %r1280;
	mov.b32 	%r1313, %r1280;
	mov.b32 	%r1314, %r1280;
	mov.b32 	%r1315, %r1280;
	mov.b32 	%r1316, %r1280;
	mov.b32 	%r1317, %r1280;
	mov.b32 	%r1318, %r1280;
	mov.b32 	%r1319, %r1280;
	mov.b32 	%r1320, %r1280;
	mov.b32 	%r1321, %r1280;
	mov.b32 	%r1322, %r1280;
	mov.b32 	%r1323, %r1280;
	mov.b32 	%r1324, %r1280;
	mov.b32 	%r1325, %r1280;
	mov.b32 	%r1326, %r1280;
	mov.b32 	%r1327, %r1280;
	mov.b32 	%r1328, %r1280;
	mov.b32 	%r1329, %r1280;
	mov.b32 	%r1330, %r1280;
	mov.b32 	%r1331, %r1280;
	mov.b32 	%r1332, %r1280;
	mov.b32 	%r1333, %r1280;
	mov.b32 	%r1334, %r1280;
	mov.b32 	%r1335, %r1280;
	mov.b32 	%r1336, %r1280;
	mov.b32 	%r1337, %r1280;
	mov.b32 	%r1338, %r1280;
	mov.b32 	%r1339, %r1280;
	mov.b32 	%r1340, %r1280;
	mov.b32 	%r1341, %r1280;
	mov.b32 	%r1342, %r1280;
	mov.b32 	%r1343, %r1280;
	mov.b32 	%r1344, %r1280;
	mov.b32 	%r1345, %r1280;
	mov.b32 	%r1346, %r1280;
	mov.b32 	%r1347, %r1280;
	mov.b32 	%r1348, %r1280;
	mov.b32 	%r1349, %r1280;
	mov.b32 	%r1350, %r1280;
	mov.b32 	%r1351, %r1280;
	mov.b32 	%r1352, %r1280;
	mov.b32 	%r1353, %r1280;
	mov.b32 	%r1354, %r1280;
	mov.b32 	%r1355, %r1280;
	mov.b32 	%r1356, %r1280;
	mov.b32 	%r1357, %r1280;
	mov.b32 	%r1358, %r1280;
	mov.b32 	%r1359, %r1280;
	mov.b32 	%r1360, %r1280;
	mov.b32 	%r1361, %r1280;
	mov.b32 	%r1362, %r1280;
	mov.b32 	%r1363, %r1280;
	mov.b32 	%r1364, %r1280;
	mov.b32 	%r1365, %r1280;
	mov.b32 	%r1366, %r1280;
	mov.b32 	%r1367, %r1280;
	mov.b32 	%r1368, %r1280;
	mov.b32 	%r1369, %r1280;
	mov.b32 	%r1370, %r1280;
	mov.b32 	%r1371, %r1280;
	mov.b32 	%r1372, %r1280;
	mov.b32 	%r1373, %r1280;
	mov.b32 	%r1374, %r1280;
	mov.b32 	%r1375, %r1280;
	mov.b32 	%r1376, %r1280;
	mov.b32 	%r1377, %r1280;
	mov.b32 	%r1378, %r1280;
	mov.b32 	%r1379, %r1280;
	mov.b32 	%r1380, %r1280;
	mov.b32 	%r1381, %r1280;
	mov.b32 	%r1382, %r1280;
	mov.b32 	%r1383, %r1280;
	mov.b32 	%r1384, %r1280;
	mov.b32 	%r1385, %r1280;
	mov.b32 	%r1386, %r1280;
	mov.b32 	%r1387, %r1280;
	mov.b32 	%r1388, %r1280;
	mov.b32 	%r1389, %r1280;
	mov.b32 	%r1390, %r1280;
	mov.b32 	%r1391, %r1280;
	mov.b32 	%r1392, %r1280;
	mov.b32 	%r1393, %r1280;
	mov.b32 	%r1394, %r1280;
	mov.b32 	%r1395, %r1280;
	mov.b32 	%r1396, %r1280;
	mov.b32 	%r1397, %r1280;
	mov.b32 	%r1398, %r1280;
	mov.b32 	%r1399, %r1280;
	mov.b32 	%r1400, %r1280;
	mov.b32 	%r1401, %r1280;
	mov.b32 	%r1402, %r1280;
	mov.b32 	%r1403, %r1280;
	mov.b32 	%r1404, %r1280;
	mov.b32 	%r1405, %r1280;
	mov.b32 	%r1406, %r1280;
	mov.b32 	%r1407, %r1280;
$L__BB0_4:                              // %._crit_edge
	.loc	1 323 45                        // sk06_mlp_down.py:323:45
	shl.b32 	%r675, %r7, 3;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r676, %r4, %r675;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r677, %r676, %r16;
	.loc	1 323 45                        // sk06_mlp_down.py:323:45
	shr.u32 	%r678, %r6, 2;
	shl.b32 	%r679, %r5, 1;
	or.b32 	%r680, %r678, %r679;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r681, %r680, %r4;
	or.b32 	%r682, %r681, 112;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r683, %r682, %r16;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r684, %r681, 96;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r685, %r684, %r16;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r686, %r681, 80;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r687, %r686, %r16;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r688, %r681, 64;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r689, %r688, %r16;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r690, %r681, 48;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r691, %r690, %r16;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r692, %r681, 32;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r693, %r692, %r16;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r694, %r681, 16;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r695, %r694, %r16;
	rem.s32 	%r696, %r681, %r16;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r697, %r1, %r3;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r698, %r697, %r15;
	.loc	1 322 45                        // sk06_mlp_down.py:322:45
	and.b32 	%r699, %r2, 240;
	bfe.u32 	%r700, %r2, 4, 4;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r701, %r700, %r1;
	or.b32 	%r702, %r701, 240;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r703, %r702, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r704, %r701, 224;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r705, %r704, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r706, %r701, 208;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r707, %r706, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r708, %r701, 192;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r709, %r708, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r710, %r701, 176;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r711, %r710, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r712, %r701, 160;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r713, %r712, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r714, %r701, 144;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r715, %r714, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r716, %r701, 128;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r717, %r716, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r718, %r701, 112;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r719, %r718, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r720, %r701, 96;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r721, %r720, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r722, %r701, 80;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r723, %r722, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r724, %r701, 64;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r725, %r724, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r726, %r701, 48;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r727, %r726, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r728, %r701, 32;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r729, %r728, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r730, %r701, 16;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r731, %r730, %r15;
	rem.s32 	%r732, %r701, %r15;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 347 38                        // sk06_mlp_down.py:347:38
	mad.wide.s32 	%rd49, %r698, 4, %rd12;
	.loc	1 347 24                        // sk06_mlp_down.py:347:24
	// begin inline asm
	mov.u32 %r446, 0x0;
	ld.global.b32 { %r446 }, [ %rd49 + 0 ];
	// end inline asm
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	and.b32 	%r733, %r2, 7;
	shl.b32 	%r734, %r733, 3;
	shl.b32 	%r735, %r699, 2;
	and.b32 	%r736, %r2, 8;
	shr.u32 	%r737, %r736, 1;
	add.s32 	%r738, %r93, %r734;
	add.s32 	%r739, %r738, %r735;
	add.s32 	%r445, %r739, %r737;
	// begin inline asm
	st.shared.b32 [ %r445 + 0 ], %r446;
	// end inline asm
	bar.sync 	0;
	and.b32 	%r740, %r9, 56;
	and.b32 	%r741, %r2, 192;
	add.s32 	%r742, %r93, %r740;
	add.s32 	%r743, %r742, %r741;
	ld.shared.v2.b32 	{%r744, %r745}, [%r743];
	ld.shared.v2.b32 	{%r746, %r747}, [%r743+256];
	ld.shared.v2.b32 	{%r748, %r749}, [%r743+512];
	ld.shared.v2.b32 	{%r750, %r751}, [%r743+768];
	.loc	1 349 42                        // sk06_mlp_down.py:349:42
	mad.wide.s32 	%rd50, %r696, 4, %rd13;
	mad.wide.s32 	%rd51, %r695, 4, %rd13;
	mad.wide.s32 	%rd52, %r693, 4, %rd13;
	mad.wide.s32 	%rd53, %r691, 4, %rd13;
	mad.wide.s32 	%rd54, %r689, 4, %rd13;
	mad.wide.s32 	%rd55, %r687, 4, %rd13;
	mad.wide.s32 	%rd56, %r685, 4, %rd13;
	mad.wide.s32 	%rd57, %r683, 4, %rd13;
	.loc	1 349 28                        // sk06_mlp_down.py:349:28
	// begin inline asm
	mov.u32 %r447, 0x0;
	mov.u32 %r448, 0x0;
	ld.global.v2.b32 { %r447, %r448 }, [ %rd50 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r449, 0x0;
	mov.u32 %r450, 0x0;
	ld.global.v2.b32 { %r449, %r450 }, [ %rd51 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r451, 0x0;
	mov.u32 %r452, 0x0;
	ld.global.v2.b32 { %r451, %r452 }, [ %rd52 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r453, 0x0;
	mov.u32 %r454, 0x0;
	ld.global.v2.b32 { %r453, %r454 }, [ %rd53 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r455, 0x0;
	mov.u32 %r456, 0x0;
	ld.global.v2.b32 { %r455, %r456 }, [ %rd54 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r457, 0x0;
	mov.u32 %r458, 0x0;
	ld.global.v2.b32 { %r457, %r458 }, [ %rd55 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r459, 0x0;
	mov.u32 %r460, 0x0;
	ld.global.v2.b32 { %r459, %r460 }, [ %rd56 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r461, 0x0;
	mov.u32 %r462, 0x0;
	ld.global.v2.b32 { %r461, %r462 }, [ %rd57 + 0 ];
	// end inline asm
	.loc	1 350 49                        // sk06_mlp_down.py:350:49
	mul.lo.s32 	%r752, %r732, %r18;
	mul.lo.s32 	%r753, %r731, %r18;
	mul.lo.s32 	%r754, %r729, %r18;
	mul.lo.s32 	%r755, %r727, %r18;
	mul.lo.s32 	%r756, %r725, %r18;
	mul.lo.s32 	%r757, %r723, %r18;
	mul.lo.s32 	%r758, %r721, %r18;
	mul.lo.s32 	%r759, %r719, %r18;
	mul.lo.s32 	%r760, %r717, %r18;
	mul.lo.s32 	%r761, %r715, %r18;
	mul.lo.s32 	%r762, %r713, %r18;
	mul.lo.s32 	%r763, %r711, %r18;
	mul.lo.s32 	%r764, %r709, %r18;
	mul.lo.s32 	%r765, %r707, %r18;
	mul.lo.s32 	%r766, %r705, %r18;
	mul.lo.s32 	%r767, %r703, %r18;
	.loc	1 350 31                        // sk06_mlp_down.py:350:31
	mad.wide.s32 	%rd90, %r752, 2, %rd11;
	mad.wide.s32 	%rd91, %r753, 2, %rd11;
	mad.wide.s32 	%rd92, %r754, 2, %rd11;
	mad.wide.s32 	%rd93, %r755, 2, %rd11;
	mad.wide.s32 	%rd94, %r756, 2, %rd11;
	mad.wide.s32 	%rd95, %r757, 2, %rd11;
	mad.wide.s32 	%rd96, %r758, 2, %rd11;
	mad.wide.s32 	%rd97, %r759, 2, %rd11;
	mad.wide.s32 	%rd98, %r760, 2, %rd11;
	mad.wide.s32 	%rd99, %r761, 2, %rd11;
	mad.wide.s32 	%rd100, %r762, 2, %rd11;
	mad.wide.s32 	%rd101, %r763, 2, %rd11;
	mad.wide.s32 	%rd102, %r764, 2, %rd11;
	mad.wide.s32 	%rd103, %r765, 2, %rd11;
	mad.wide.s32 	%rd104, %r766, 2, %rd11;
	mad.wide.s32 	%rd105, %r767, 2, %rd11;
	.loc	1 350 64                        // sk06_mlp_down.py:350:64
	mul.wide.s32 	%rd106, %r677, 2;
	add.s64 	%rd58, %rd90, %rd106;
	add.s64 	%rd59, %rd91, %rd106;
	add.s64 	%rd60, %rd92, %rd106;
	add.s64 	%rd61, %rd93, %rd106;
	add.s64 	%rd62, %rd94, %rd106;
	add.s64 	%rd63, %rd95, %rd106;
	add.s64 	%rd64, %rd96, %rd106;
	add.s64 	%rd65, %rd97, %rd106;
	add.s64 	%rd66, %rd98, %rd106;
	add.s64 	%rd67, %rd99, %rd106;
	add.s64 	%rd68, %rd100, %rd106;
	add.s64 	%rd69, %rd101, %rd106;
	add.s64 	%rd70, %rd102, %rd106;
	add.s64 	%rd71, %rd103, %rd106;
	add.s64 	%rd72, %rd104, %rd106;
	add.s64 	%rd73, %rd105, %rd106;
	.loc	1 350 19                        // sk06_mlp_down.py:350:19
	// begin inline asm
	mov.u32 %r464, 0x0;
	mov.u32 %r465, 0x0;
	mov.u32 %r466, 0x0;
	mov.u32 %r467, 0x0;
	ld.global.v4.b32 { %r464, %r465, %r466, %r467 }, [ %rd58 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r469, 0x0;
	mov.u32 %r470, 0x0;
	mov.u32 %r471, 0x0;
	mov.u32 %r472, 0x0;
	ld.global.v4.b32 { %r469, %r470, %r471, %r472 }, [ %rd59 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r474, 0x0;
	mov.u32 %r475, 0x0;
	mov.u32 %r476, 0x0;
	mov.u32 %r477, 0x0;
	ld.global.v4.b32 { %r474, %r475, %r476, %r477 }, [ %rd60 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r479, 0x0;
	mov.u32 %r480, 0x0;
	mov.u32 %r481, 0x0;
	mov.u32 %r482, 0x0;
	ld.global.v4.b32 { %r479, %r480, %r481, %r482 }, [ %rd61 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r483, 0x0;
	mov.u32 %r484, 0x0;
	mov.u32 %r485, 0x0;
	mov.u32 %r486, 0x0;
	ld.global.v4.b32 { %r483, %r484, %r485, %r486 }, [ %rd62 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r487, 0x0;
	mov.u32 %r488, 0x0;
	mov.u32 %r489, 0x0;
	mov.u32 %r490, 0x0;
	ld.global.v4.b32 { %r487, %r488, %r489, %r490 }, [ %rd63 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r491, 0x0;
	mov.u32 %r492, 0x0;
	mov.u32 %r493, 0x0;
	mov.u32 %r494, 0x0;
	ld.global.v4.b32 { %r491, %r492, %r493, %r494 }, [ %rd64 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r495, 0x0;
	mov.u32 %r496, 0x0;
	mov.u32 %r497, 0x0;
	mov.u32 %r498, 0x0;
	ld.global.v4.b32 { %r495, %r496, %r497, %r498 }, [ %rd65 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r499, 0x0;
	mov.u32 %r500, 0x0;
	mov.u32 %r501, 0x0;
	mov.u32 %r502, 0x0;
	ld.global.v4.b32 { %r499, %r500, %r501, %r502 }, [ %rd66 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r503, 0x0;
	mov.u32 %r504, 0x0;
	mov.u32 %r505, 0x0;
	mov.u32 %r506, 0x0;
	ld.global.v4.b32 { %r503, %r504, %r505, %r506 }, [ %rd67 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r507, 0x0;
	mov.u32 %r508, 0x0;
	mov.u32 %r509, 0x0;
	mov.u32 %r510, 0x0;
	ld.global.v4.b32 { %r507, %r508, %r509, %r510 }, [ %rd68 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r511, 0x0;
	mov.u32 %r512, 0x0;
	mov.u32 %r513, 0x0;
	mov.u32 %r514, 0x0;
	ld.global.v4.b32 { %r511, %r512, %r513, %r514 }, [ %rd69 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r515, 0x0;
	mov.u32 %r516, 0x0;
	mov.u32 %r517, 0x0;
	mov.u32 %r518, 0x0;
	ld.global.v4.b32 { %r515, %r516, %r517, %r518 }, [ %rd70 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r519, 0x0;
	mov.u32 %r520, 0x0;
	mov.u32 %r521, 0x0;
	mov.u32 %r522, 0x0;
	ld.global.v4.b32 { %r519, %r520, %r521, %r522 }, [ %rd71 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r523, 0x0;
	mov.u32 %r524, 0x0;
	mov.u32 %r525, 0x0;
	mov.u32 %r526, 0x0;
	ld.global.v4.b32 { %r523, %r524, %r525, %r526 }, [ %rd72 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r527, 0x0;
	mov.u32 %r528, 0x0;
	mov.u32 %r529, 0x0;
	mov.u32 %r530, 0x0;
	ld.global.v4.b32 { %r527, %r528, %r529, %r530 }, [ %rd73 + 0 ];
	// end inline asm
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	bar.sync 	0;
	shl.b32 	%r768, %r2, 7;
	and.b32 	%r769, %r768, 15360;
	shl.b32 	%r770, %r733, 4;
	or.b32 	%r771, %r769, %r770;
	xor.b32 	%r772, %r771, %r699;
	add.s32 	%r463, %r93, %r772;
	// begin inline asm
	st.shared.v4.b32 [ %r463 + 0 ], { %r464, %r465, %r466, %r467 };
	// end inline asm
	add.s32 	%r468, %r463, 256;
	// begin inline asm
	st.shared.v4.b32 [ %r468 + 0 ], { %r469, %r470, %r471, %r472 };
	// end inline asm
	add.s32 	%r473, %r463, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r473 + 0 ], { %r474, %r475, %r476, %r477 };
	// end inline asm
	add.s32 	%r478, %r463, 768;
	// begin inline asm
	st.shared.v4.b32 [ %r478 + 0 ], { %r479, %r480, %r481, %r482 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r773, %r733, 11;
	shl.b32 	%r774, %r7, 4;
	shl.b32 	%r775, %r741, 2;
	setp.eq.b32 	%p24, %r1279, 0;
	shl.b32 	%r776, %r1279, 1;
	shr.u32 	%r777, %r6, 1;
	or.b32 	%r778, %r774, %r775;
	or.b32 	%r779, %r776, %r777;
	xor.b32 	%r780, %r778, %r779;
	or.b32 	%r781, %r780, %r773;
	add.s32 	%r782, %r93, %r781;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r783, %r784, %r785, %r786}, [%r782];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r787, %r788, %r789, %r790}, [%r782+1024];
	xor.b32 	%r791, %r781, 64;
	add.s32 	%r792, %r93, %r791;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r793, %r794, %r795, %r796}, [%r792];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r797, %r798, %r799, %r800}, [%r792+1024];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r463 + 0 ], { %r483, %r484, %r485, %r486 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r468 + 0 ], { %r487, %r488, %r489, %r490 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r473 + 0 ], { %r491, %r492, %r493, %r494 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r478 + 0 ], { %r495, %r496, %r497, %r498 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r801, %r802, %r803, %r804}, [%r782];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r805, %r806, %r807, %r808}, [%r782+1024];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r809, %r810, %r811, %r812}, [%r792];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r813, %r814, %r815, %r816}, [%r792+1024];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r463 + 0 ], { %r499, %r500, %r501, %r502 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r468 + 0 ], { %r503, %r504, %r505, %r506 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r473 + 0 ], { %r507, %r508, %r509, %r510 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r478 + 0 ], { %r511, %r512, %r513, %r514 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r817, %r818, %r819, %r820}, [%r782];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r821, %r822, %r823, %r824}, [%r782+1024];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r825, %r826, %r827, %r828}, [%r792];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r829, %r830, %r831, %r832}, [%r792+1024];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r463 + 0 ], { %r515, %r516, %r517, %r518 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r468 + 0 ], { %r519, %r520, %r521, %r522 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r473 + 0 ], { %r523, %r524, %r525, %r526 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r478 + 0 ], { %r527, %r528, %r529, %r530 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r833, %r834, %r835, %r836}, [%r782];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r837, %r838, %r839, %r840}, [%r782+1024];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r841, %r842, %r843, %r844}, [%r792];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r845, %r846, %r847, %r848}, [%r792+1024];
	.loc	1 357 31                        // sk06_mlp_down.py:357:31
	setp.lt.s32 	%p25, %r701, %r15;
	setp.lt.s32 	%p26, %r730, %r15;
	setp.lt.s32 	%p27, %r728, %r15;
	setp.lt.s32 	%p28, %r726, %r15;
	setp.lt.s32 	%p29, %r724, %r15;
	setp.lt.s32 	%p30, %r722, %r15;
	setp.lt.s32 	%p31, %r720, %r15;
	setp.lt.s32 	%p32, %r718, %r15;
	setp.lt.s32 	%p33, %r716, %r15;
	setp.lt.s32 	%p34, %r714, %r15;
	setp.lt.s32 	%p35, %r712, %r15;
	setp.lt.s32 	%p36, %r710, %r15;
	setp.lt.s32 	%p37, %r708, %r15;
	setp.lt.s32 	%p38, %r706, %r15;
	setp.lt.s32 	%p39, %r704, %r15;
	setp.lt.s32 	%p40, %r702, %r15;
	.loc	1 357 54                        // sk06_mlp_down.py:357:54
	setp.lt.s32 	%p41, %r676, %r16;
	.loc	1 357 37                        // sk06_mlp_down.py:357:37
	and.pred 	%p8, %p25, %p41;
	and.pred 	%p9, %p26, %p41;
	and.pred 	%p10, %p27, %p41;
	and.pred 	%p11, %p28, %p41;
	and.pred 	%p12, %p29, %p41;
	and.pred 	%p13, %p30, %p41;
	and.pred 	%p14, %p31, %p41;
	and.pred 	%p15, %p32, %p41;
	and.pred 	%p16, %p33, %p41;
	and.pred 	%p17, %p34, %p41;
	and.pred 	%p18, %p35, %p41;
	and.pred 	%p19, %p36, %p41;
	and.pred 	%p20, %p37, %p41;
	and.pred 	%p21, %p38, %p41;
	and.pred 	%p22, %p39, %p41;
	and.pred 	%p23, %p40, %p41;
	.loc	1 355 35                        // sk06_mlp_down.py:355:35
	mul.lo.s32 	%r849, %r701, %r17;
	mul.lo.s32 	%r850, %r730, %r17;
	mul.lo.s32 	%r851, %r728, %r17;
	mul.lo.s32 	%r852, %r726, %r17;
	mul.lo.s32 	%r853, %r724, %r17;
	mul.lo.s32 	%r854, %r722, %r17;
	mul.lo.s32 	%r855, %r720, %r17;
	mul.lo.s32 	%r856, %r718, %r17;
	mul.lo.s32 	%r857, %r716, %r17;
	mul.lo.s32 	%r858, %r714, %r17;
	mul.lo.s32 	%r859, %r712, %r17;
	mul.lo.s32 	%r860, %r710, %r17;
	mul.lo.s32 	%r861, %r708, %r17;
	mul.lo.s32 	%r862, %r706, %r17;
	mul.lo.s32 	%r863, %r704, %r17;
	mul.lo.s32 	%r864, %r702, %r17;
	.loc	1 355 18                        // sk06_mlp_down.py:355:18
	mad.wide.s32 	%rd107, %r849, 2, %rd10;
	mad.wide.s32 	%rd108, %r850, 2, %rd10;
	mad.wide.s32 	%rd109, %r851, 2, %rd10;
	mad.wide.s32 	%rd110, %r852, 2, %rd10;
	mad.wide.s32 	%rd111, %r853, 2, %rd10;
	mad.wide.s32 	%rd112, %r854, 2, %rd10;
	mad.wide.s32 	%rd113, %r855, 2, %rd10;
	mad.wide.s32 	%rd114, %r856, 2, %rd10;
	mad.wide.s32 	%rd115, %r857, 2, %rd10;
	mad.wide.s32 	%rd116, %r858, 2, %rd10;
	mad.wide.s32 	%rd117, %r859, 2, %rd10;
	mad.wide.s32 	%rd118, %r860, 2, %rd10;
	mad.wide.s32 	%rd119, %r861, 2, %rd10;
	mad.wide.s32 	%rd120, %r862, 2, %rd10;
	mad.wide.s32 	%rd121, %r863, 2, %rd10;
	mad.wide.s32 	%rd122, %r864, 2, %rd10;
	.loc	1 355 50                        // sk06_mlp_down.py:355:50
	mul.wide.s32 	%rd123, %r676, 2;
	add.s64 	%rd74, %rd107, %rd123;
	add.s64 	%rd75, %rd108, %rd123;
	add.s64 	%rd76, %rd109, %rd123;
	add.s64 	%rd77, %rd110, %rd123;
	add.s64 	%rd78, %rd111, %rd123;
	add.s64 	%rd79, %rd112, %rd123;
	add.s64 	%rd80, %rd113, %rd123;
	add.s64 	%rd81, %rd114, %rd123;
	add.s64 	%rd82, %rd115, %rd123;
	add.s64 	%rd83, %rd116, %rd123;
	add.s64 	%rd84, %rd117, %rd123;
	add.s64 	%rd85, %rd118, %rd123;
	add.s64 	%rd86, %rd119, %rd123;
	add.s64 	%rd87, %rd120, %rd123;
	add.s64 	%rd88, %rd121, %rd123;
	add.s64 	%rd89, %rd122, %rd123;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r865, %r1377, %r750;
	mul.f32 	%r866, %r1376, %r750;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs1, %rs2}, %r833;
	cvt.f32.bf16 	%r867, %rs2;
	cvt.f32.bf16 	%r868, %rs1;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r869, %r866, %r447, %r868;
	fma.rn.f32 	%r870, %r865, %r448, %r867;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r871, %r1281, %r744;
	mul.f32 	%r872, %r1280, %r744;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs3, %rs4}, %r783;
	cvt.f32.bf16 	%r873, %rs4;
	cvt.f32.bf16 	%r874, %rs3;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r875, %r872, %r447, %r874;
	fma.rn.f32 	%r876, %r871, %r448, %r873;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r532, %r876, %r875;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r877, %r1283, %r745;
	mul.f32 	%r878, %r1282, %r745;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs5, %rs6}, %r784;
	cvt.f32.bf16 	%r879, %rs6;
	cvt.f32.bf16 	%r880, %rs5;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r881, %r878, %r447, %r880;
	fma.rn.f32 	%r882, %r877, %r448, %r879;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r537, %r882, %r881;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r883, %r1313, %r746;
	mul.f32 	%r884, %r1312, %r746;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs7, %rs8}, %r801;
	cvt.f32.bf16 	%r885, %rs8;
	cvt.f32.bf16 	%r886, %rs7;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r887, %r884, %r447, %r886;
	fma.rn.f32 	%r888, %r883, %r448, %r885;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r533, %r888, %r887;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r889, %r1315, %r747;
	mul.f32 	%r890, %r1314, %r747;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs9, %rs10}, %r802;
	cvt.f32.bf16 	%r891, %rs10;
	cvt.f32.bf16 	%r892, %rs9;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r893, %r890, %r447, %r892;
	fma.rn.f32 	%r894, %r889, %r448, %r891;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r538, %r894, %r893;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r895, %r1345, %r748;
	mul.f32 	%r896, %r1344, %r748;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs11, %rs12}, %r817;
	cvt.f32.bf16 	%r897, %rs12;
	cvt.f32.bf16 	%r898, %rs11;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r899, %r896, %r447, %r898;
	fma.rn.f32 	%r900, %r895, %r448, %r897;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r534, %r900, %r899;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r901, %r1347, %r749;
	mul.f32 	%r902, %r1346, %r749;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs13, %rs14}, %r818;
	cvt.f32.bf16 	%r903, %rs14;
	cvt.f32.bf16 	%r904, %rs13;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r905, %r902, %r447, %r904;
	fma.rn.f32 	%r906, %r901, %r448, %r903;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r539, %r906, %r905;
	cvt.rn.bf16x2.f32 	%r535, %r870, %r869;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r907, %r1379, %r751;
	mul.f32 	%r908, %r1378, %r751;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs15, %rs16}, %r834;
	cvt.f32.bf16 	%r909, %rs16;
	cvt.f32.bf16 	%r910, %rs15;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r911, %r908, %r447, %r910;
	fma.rn.f32 	%r912, %r907, %r448, %r909;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r540, %r912, %r911;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r913, %r1381, %r750;
	mul.f32 	%r914, %r1380, %r750;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs17, %rs18}, %r835;
	cvt.f32.bf16 	%r915, %rs18;
	cvt.f32.bf16 	%r916, %rs17;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r917, %r914, %r449, %r916;
	fma.rn.f32 	%r918, %r913, %r450, %r915;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r919, %r1285, %r744;
	mul.f32 	%r920, %r1284, %r744;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs19, %rs20}, %r785;
	cvt.f32.bf16 	%r921, %rs20;
	cvt.f32.bf16 	%r922, %rs19;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r923, %r920, %r449, %r922;
	fma.rn.f32 	%r924, %r919, %r450, %r921;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r552, %r924, %r923;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r925, %r1287, %r745;
	mul.f32 	%r926, %r1286, %r745;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs21, %rs22}, %r786;
	cvt.f32.bf16 	%r927, %rs22;
	cvt.f32.bf16 	%r928, %rs21;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r929, %r926, %r449, %r928;
	fma.rn.f32 	%r930, %r925, %r450, %r927;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r557, %r930, %r929;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r931, %r1317, %r746;
	mul.f32 	%r932, %r1316, %r746;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs23, %rs24}, %r803;
	cvt.f32.bf16 	%r933, %rs24;
	cvt.f32.bf16 	%r934, %rs23;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r935, %r932, %r449, %r934;
	fma.rn.f32 	%r936, %r931, %r450, %r933;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r553, %r936, %r935;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r937, %r1319, %r747;
	mul.f32 	%r938, %r1318, %r747;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs25, %rs26}, %r804;
	cvt.f32.bf16 	%r939, %rs26;
	cvt.f32.bf16 	%r940, %rs25;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r941, %r938, %r449, %r940;
	fma.rn.f32 	%r942, %r937, %r450, %r939;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r558, %r942, %r941;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r943, %r1349, %r748;
	mul.f32 	%r944, %r1348, %r748;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs27, %rs28}, %r819;
	cvt.f32.bf16 	%r945, %rs28;
	cvt.f32.bf16 	%r946, %rs27;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r947, %r944, %r449, %r946;
	fma.rn.f32 	%r948, %r943, %r450, %r945;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r554, %r948, %r947;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r949, %r1351, %r749;
	mul.f32 	%r950, %r1350, %r749;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs29, %rs30}, %r820;
	cvt.f32.bf16 	%r951, %rs30;
	cvt.f32.bf16 	%r952, %rs29;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r953, %r950, %r449, %r952;
	fma.rn.f32 	%r954, %r949, %r450, %r951;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r559, %r954, %r953;
	cvt.rn.bf16x2.f32 	%r555, %r918, %r917;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r955, %r1383, %r751;
	mul.f32 	%r956, %r1382, %r751;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs31, %rs32}, %r836;
	cvt.f32.bf16 	%r957, %rs32;
	cvt.f32.bf16 	%r958, %rs31;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r959, %r956, %r449, %r958;
	fma.rn.f32 	%r960, %r955, %r450, %r957;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r560, %r960, %r959;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r961, %r1385, %r750;
	mul.f32 	%r962, %r1384, %r750;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs33, %rs34}, %r841;
	cvt.f32.bf16 	%r963, %rs34;
	cvt.f32.bf16 	%r964, %rs33;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r965, %r962, %r451, %r964;
	fma.rn.f32 	%r966, %r961, %r452, %r963;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r967, %r1289, %r744;
	mul.f32 	%r968, %r1288, %r744;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs35, %rs36}, %r793;
	cvt.f32.bf16 	%r969, %rs36;
	cvt.f32.bf16 	%r970, %rs35;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r971, %r968, %r451, %r970;
	fma.rn.f32 	%r972, %r967, %r452, %r969;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r572, %r972, %r971;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r973, %r1291, %r745;
	mul.f32 	%r974, %r1290, %r745;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs37, %rs38}, %r794;
	cvt.f32.bf16 	%r975, %rs38;
	cvt.f32.bf16 	%r976, %rs37;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r977, %r974, %r451, %r976;
	fma.rn.f32 	%r978, %r973, %r452, %r975;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r577, %r978, %r977;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r979, %r1321, %r746;
	mul.f32 	%r980, %r1320, %r746;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs39, %rs40}, %r809;
	cvt.f32.bf16 	%r981, %rs40;
	cvt.f32.bf16 	%r982, %rs39;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r983, %r980, %r451, %r982;
	fma.rn.f32 	%r984, %r979, %r452, %r981;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r573, %r984, %r983;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r985, %r1323, %r747;
	mul.f32 	%r986, %r1322, %r747;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs41, %rs42}, %r810;
	cvt.f32.bf16 	%r987, %rs42;
	cvt.f32.bf16 	%r988, %rs41;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r989, %r986, %r451, %r988;
	fma.rn.f32 	%r990, %r985, %r452, %r987;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r578, %r990, %r989;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r991, %r1353, %r748;
	mul.f32 	%r992, %r1352, %r748;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs43, %rs44}, %r825;
	cvt.f32.bf16 	%r993, %rs44;
	cvt.f32.bf16 	%r994, %rs43;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r995, %r992, %r451, %r994;
	fma.rn.f32 	%r996, %r991, %r452, %r993;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r574, %r996, %r995;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r997, %r1355, %r749;
	mul.f32 	%r998, %r1354, %r749;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs45, %rs46}, %r826;
	cvt.f32.bf16 	%r999, %rs46;
	cvt.f32.bf16 	%r1000, %rs45;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1001, %r998, %r451, %r1000;
	fma.rn.f32 	%r1002, %r997, %r452, %r999;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r579, %r1002, %r1001;
	cvt.rn.bf16x2.f32 	%r575, %r966, %r965;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1003, %r1387, %r751;
	mul.f32 	%r1004, %r1386, %r751;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs47, %rs48}, %r842;
	cvt.f32.bf16 	%r1005, %rs48;
	cvt.f32.bf16 	%r1006, %rs47;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1007, %r1004, %r451, %r1006;
	fma.rn.f32 	%r1008, %r1003, %r452, %r1005;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r580, %r1008, %r1007;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1009, %r1389, %r750;
	mul.f32 	%r1010, %r1388, %r750;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs49, %rs50}, %r843;
	cvt.f32.bf16 	%r1011, %rs50;
	cvt.f32.bf16 	%r1012, %rs49;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1013, %r1010, %r453, %r1012;
	fma.rn.f32 	%r1014, %r1009, %r454, %r1011;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1015, %r1293, %r744;
	mul.f32 	%r1016, %r1292, %r744;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs51, %rs52}, %r795;
	cvt.f32.bf16 	%r1017, %rs52;
	cvt.f32.bf16 	%r1018, %rs51;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1019, %r1016, %r453, %r1018;
	fma.rn.f32 	%r1020, %r1015, %r454, %r1017;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r592, %r1020, %r1019;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1021, %r1295, %r745;
	mul.f32 	%r1022, %r1294, %r745;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs53, %rs54}, %r796;
	cvt.f32.bf16 	%r1023, %rs54;
	cvt.f32.bf16 	%r1024, %rs53;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1025, %r1022, %r453, %r1024;
	fma.rn.f32 	%r1026, %r1021, %r454, %r1023;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r597, %r1026, %r1025;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1027, %r1325, %r746;
	mul.f32 	%r1028, %r1324, %r746;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs55, %rs56}, %r811;
	cvt.f32.bf16 	%r1029, %rs56;
	cvt.f32.bf16 	%r1030, %rs55;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1031, %r1028, %r453, %r1030;
	fma.rn.f32 	%r1032, %r1027, %r454, %r1029;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r593, %r1032, %r1031;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1033, %r1327, %r747;
	mul.f32 	%r1034, %r1326, %r747;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs57, %rs58}, %r812;
	cvt.f32.bf16 	%r1035, %rs58;
	cvt.f32.bf16 	%r1036, %rs57;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1037, %r1034, %r453, %r1036;
	fma.rn.f32 	%r1038, %r1033, %r454, %r1035;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r598, %r1038, %r1037;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1039, %r1357, %r748;
	mul.f32 	%r1040, %r1356, %r748;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs59, %rs60}, %r827;
	cvt.f32.bf16 	%r1041, %rs60;
	cvt.f32.bf16 	%r1042, %rs59;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1043, %r1040, %r453, %r1042;
	fma.rn.f32 	%r1044, %r1039, %r454, %r1041;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r594, %r1044, %r1043;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1045, %r1359, %r749;
	mul.f32 	%r1046, %r1358, %r749;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs61, %rs62}, %r828;
	cvt.f32.bf16 	%r1047, %rs62;
	cvt.f32.bf16 	%r1048, %rs61;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1049, %r1046, %r453, %r1048;
	fma.rn.f32 	%r1050, %r1045, %r454, %r1047;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r599, %r1050, %r1049;
	cvt.rn.bf16x2.f32 	%r595, %r1014, %r1013;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1051, %r1391, %r751;
	mul.f32 	%r1052, %r1390, %r751;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs63, %rs64}, %r844;
	cvt.f32.bf16 	%r1053, %rs64;
	cvt.f32.bf16 	%r1054, %rs63;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1055, %r1052, %r453, %r1054;
	fma.rn.f32 	%r1056, %r1051, %r454, %r1053;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r600, %r1056, %r1055;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1057, %r1393, %r750;
	mul.f32 	%r1058, %r1392, %r750;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs65, %rs66}, %r837;
	cvt.f32.bf16 	%r1059, %rs66;
	cvt.f32.bf16 	%r1060, %rs65;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1061, %r1058, %r455, %r1060;
	fma.rn.f32 	%r1062, %r1057, %r456, %r1059;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1063, %r1297, %r744;
	mul.f32 	%r1064, %r1296, %r744;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs67, %rs68}, %r787;
	cvt.f32.bf16 	%r1065, %rs68;
	cvt.f32.bf16 	%r1066, %rs67;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1067, %r1064, %r455, %r1066;
	fma.rn.f32 	%r1068, %r1063, %r456, %r1065;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r542, %r1068, %r1067;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1069, %r1299, %r745;
	mul.f32 	%r1070, %r1298, %r745;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs69, %rs70}, %r788;
	cvt.f32.bf16 	%r1071, %rs70;
	cvt.f32.bf16 	%r1072, %rs69;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1073, %r1070, %r455, %r1072;
	fma.rn.f32 	%r1074, %r1069, %r456, %r1071;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r547, %r1074, %r1073;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1075, %r1329, %r746;
	mul.f32 	%r1076, %r1328, %r746;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs71, %rs72}, %r805;
	cvt.f32.bf16 	%r1077, %rs72;
	cvt.f32.bf16 	%r1078, %rs71;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1079, %r1076, %r455, %r1078;
	fma.rn.f32 	%r1080, %r1075, %r456, %r1077;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r543, %r1080, %r1079;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1081, %r1331, %r747;
	mul.f32 	%r1082, %r1330, %r747;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs73, %rs74}, %r806;
	cvt.f32.bf16 	%r1083, %rs74;
	cvt.f32.bf16 	%r1084, %rs73;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1085, %r1082, %r455, %r1084;
	fma.rn.f32 	%r1086, %r1081, %r456, %r1083;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r548, %r1086, %r1085;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1087, %r1361, %r748;
	mul.f32 	%r1088, %r1360, %r748;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs75, %rs76}, %r821;
	cvt.f32.bf16 	%r1089, %rs76;
	cvt.f32.bf16 	%r1090, %rs75;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1091, %r1088, %r455, %r1090;
	fma.rn.f32 	%r1092, %r1087, %r456, %r1089;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r544, %r1092, %r1091;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1093, %r1363, %r749;
	mul.f32 	%r1094, %r1362, %r749;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs77, %rs78}, %r822;
	cvt.f32.bf16 	%r1095, %rs78;
	cvt.f32.bf16 	%r1096, %rs77;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1097, %r1094, %r455, %r1096;
	fma.rn.f32 	%r1098, %r1093, %r456, %r1095;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r549, %r1098, %r1097;
	cvt.rn.bf16x2.f32 	%r545, %r1062, %r1061;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1099, %r1395, %r751;
	mul.f32 	%r1100, %r1394, %r751;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs79, %rs80}, %r838;
	cvt.f32.bf16 	%r1101, %rs80;
	cvt.f32.bf16 	%r1102, %rs79;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1103, %r1100, %r455, %r1102;
	fma.rn.f32 	%r1104, %r1099, %r456, %r1101;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r550, %r1104, %r1103;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1105, %r1397, %r750;
	mul.f32 	%r1106, %r1396, %r750;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs81, %rs82}, %r839;
	cvt.f32.bf16 	%r1107, %rs82;
	cvt.f32.bf16 	%r1108, %rs81;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1109, %r1106, %r457, %r1108;
	fma.rn.f32 	%r1110, %r1105, %r458, %r1107;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1111, %r1301, %r744;
	mul.f32 	%r1112, %r1300, %r744;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs83, %rs84}, %r789;
	cvt.f32.bf16 	%r1113, %rs84;
	cvt.f32.bf16 	%r1114, %rs83;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1115, %r1112, %r457, %r1114;
	fma.rn.f32 	%r1116, %r1111, %r458, %r1113;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r562, %r1116, %r1115;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1117, %r1303, %r745;
	mul.f32 	%r1118, %r1302, %r745;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs85, %rs86}, %r790;
	cvt.f32.bf16 	%r1119, %rs86;
	cvt.f32.bf16 	%r1120, %rs85;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1121, %r1118, %r457, %r1120;
	fma.rn.f32 	%r1122, %r1117, %r458, %r1119;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r567, %r1122, %r1121;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1123, %r1333, %r746;
	mul.f32 	%r1124, %r1332, %r746;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs87, %rs88}, %r807;
	cvt.f32.bf16 	%r1125, %rs88;
	cvt.f32.bf16 	%r1126, %rs87;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1127, %r1124, %r457, %r1126;
	fma.rn.f32 	%r1128, %r1123, %r458, %r1125;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r563, %r1128, %r1127;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1129, %r1335, %r747;
	mul.f32 	%r1130, %r1334, %r747;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs89, %rs90}, %r808;
	cvt.f32.bf16 	%r1131, %rs90;
	cvt.f32.bf16 	%r1132, %rs89;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1133, %r1130, %r457, %r1132;
	fma.rn.f32 	%r1134, %r1129, %r458, %r1131;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r568, %r1134, %r1133;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1135, %r1365, %r748;
	mul.f32 	%r1136, %r1364, %r748;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs91, %rs92}, %r823;
	cvt.f32.bf16 	%r1137, %rs92;
	cvt.f32.bf16 	%r1138, %rs91;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1139, %r1136, %r457, %r1138;
	fma.rn.f32 	%r1140, %r1135, %r458, %r1137;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r564, %r1140, %r1139;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1141, %r1367, %r749;
	mul.f32 	%r1142, %r1366, %r749;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs93, %rs94}, %r824;
	cvt.f32.bf16 	%r1143, %rs94;
	cvt.f32.bf16 	%r1144, %rs93;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1145, %r1142, %r457, %r1144;
	fma.rn.f32 	%r1146, %r1141, %r458, %r1143;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r569, %r1146, %r1145;
	cvt.rn.bf16x2.f32 	%r565, %r1110, %r1109;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1147, %r1399, %r751;
	mul.f32 	%r1148, %r1398, %r751;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs95, %rs96}, %r840;
	cvt.f32.bf16 	%r1149, %rs96;
	cvt.f32.bf16 	%r1150, %rs95;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1151, %r1148, %r457, %r1150;
	fma.rn.f32 	%r1152, %r1147, %r458, %r1149;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r570, %r1152, %r1151;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1153, %r1401, %r750;
	mul.f32 	%r1154, %r1400, %r750;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs97, %rs98}, %r845;
	cvt.f32.bf16 	%r1155, %rs98;
	cvt.f32.bf16 	%r1156, %rs97;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1157, %r1154, %r459, %r1156;
	fma.rn.f32 	%r1158, %r1153, %r460, %r1155;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1159, %r1305, %r744;
	mul.f32 	%r1160, %r1304, %r744;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs99, %rs100}, %r797;
	cvt.f32.bf16 	%r1161, %rs100;
	cvt.f32.bf16 	%r1162, %rs99;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1163, %r1160, %r459, %r1162;
	fma.rn.f32 	%r1164, %r1159, %r460, %r1161;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r582, %r1164, %r1163;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1165, %r1307, %r745;
	mul.f32 	%r1166, %r1306, %r745;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs101, %rs102}, %r798;
	cvt.f32.bf16 	%r1167, %rs102;
	cvt.f32.bf16 	%r1168, %rs101;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1169, %r1166, %r459, %r1168;
	fma.rn.f32 	%r1170, %r1165, %r460, %r1167;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r587, %r1170, %r1169;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1171, %r1337, %r746;
	mul.f32 	%r1172, %r1336, %r746;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs103, %rs104}, %r813;
	cvt.f32.bf16 	%r1173, %rs104;
	cvt.f32.bf16 	%r1174, %rs103;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1175, %r1172, %r459, %r1174;
	fma.rn.f32 	%r1176, %r1171, %r460, %r1173;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r583, %r1176, %r1175;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1177, %r1339, %r747;
	mul.f32 	%r1178, %r1338, %r747;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs105, %rs106}, %r814;
	cvt.f32.bf16 	%r1179, %rs106;
	cvt.f32.bf16 	%r1180, %rs105;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1181, %r1178, %r459, %r1180;
	fma.rn.f32 	%r1182, %r1177, %r460, %r1179;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r588, %r1182, %r1181;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1183, %r1369, %r748;
	mul.f32 	%r1184, %r1368, %r748;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs107, %rs108}, %r829;
	cvt.f32.bf16 	%r1185, %rs108;
	cvt.f32.bf16 	%r1186, %rs107;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1187, %r1184, %r459, %r1186;
	fma.rn.f32 	%r1188, %r1183, %r460, %r1185;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r584, %r1188, %r1187;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1189, %r1371, %r749;
	mul.f32 	%r1190, %r1370, %r749;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs109, %rs110}, %r830;
	cvt.f32.bf16 	%r1191, %rs110;
	cvt.f32.bf16 	%r1192, %rs109;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1193, %r1190, %r459, %r1192;
	fma.rn.f32 	%r1194, %r1189, %r460, %r1191;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r589, %r1194, %r1193;
	cvt.rn.bf16x2.f32 	%r585, %r1158, %r1157;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1195, %r1403, %r751;
	mul.f32 	%r1196, %r1402, %r751;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs111, %rs112}, %r846;
	cvt.f32.bf16 	%r1197, %rs112;
	cvt.f32.bf16 	%r1198, %rs111;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1199, %r1196, %r459, %r1198;
	fma.rn.f32 	%r1200, %r1195, %r460, %r1197;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r590, %r1200, %r1199;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1201, %r1405, %r750;
	mul.f32 	%r1202, %r1404, %r750;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs113, %rs114}, %r847;
	cvt.f32.bf16 	%r1203, %rs114;
	cvt.f32.bf16 	%r1204, %rs113;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1205, %r1202, %r461, %r1204;
	fma.rn.f32 	%r1206, %r1201, %r462, %r1203;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1207, %r1309, %r744;
	mul.f32 	%r1208, %r1308, %r744;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs115, %rs116}, %r799;
	cvt.f32.bf16 	%r1209, %rs116;
	cvt.f32.bf16 	%r1210, %rs115;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1211, %r1208, %r461, %r1210;
	fma.rn.f32 	%r1212, %r1207, %r462, %r1209;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r602, %r1212, %r1211;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1213, %r1311, %r745;
	mul.f32 	%r1214, %r1310, %r745;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs117, %rs118}, %r800;
	cvt.f32.bf16 	%r1215, %rs118;
	cvt.f32.bf16 	%r1216, %rs117;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1217, %r1214, %r461, %r1216;
	fma.rn.f32 	%r1218, %r1213, %r462, %r1215;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r607, %r1218, %r1217;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1219, %r1341, %r746;
	mul.f32 	%r1220, %r1340, %r746;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs119, %rs120}, %r815;
	cvt.f32.bf16 	%r1221, %rs120;
	cvt.f32.bf16 	%r1222, %rs119;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1223, %r1220, %r461, %r1222;
	fma.rn.f32 	%r1224, %r1219, %r462, %r1221;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r603, %r1224, %r1223;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1225, %r1343, %r747;
	mul.f32 	%r1226, %r1342, %r747;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs121, %rs122}, %r816;
	cvt.f32.bf16 	%r1227, %rs122;
	cvt.f32.bf16 	%r1228, %rs121;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1229, %r1226, %r461, %r1228;
	fma.rn.f32 	%r1230, %r1225, %r462, %r1227;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r608, %r1230, %r1229;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1231, %r1373, %r748;
	mul.f32 	%r1232, %r1372, %r748;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs123, %rs124}, %r831;
	cvt.f32.bf16 	%r1233, %rs124;
	cvt.f32.bf16 	%r1234, %rs123;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1235, %r1232, %r461, %r1234;
	fma.rn.f32 	%r1236, %r1231, %r462, %r1233;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r604, %r1236, %r1235;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1237, %r1375, %r749;
	mul.f32 	%r1238, %r1374, %r749;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs125, %rs126}, %r832;
	cvt.f32.bf16 	%r1239, %rs126;
	cvt.f32.bf16 	%r1240, %rs125;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1241, %r1238, %r461, %r1240;
	fma.rn.f32 	%r1242, %r1237, %r462, %r1239;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r609, %r1242, %r1241;
	cvt.rn.bf16x2.f32 	%r605, %r1206, %r1205;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1243, %r1407, %r751;
	mul.f32 	%r1244, %r1406, %r751;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs127, %rs128}, %r848;
	cvt.f32.bf16 	%r1245, %rs128;
	cvt.f32.bf16 	%r1246, %rs127;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1247, %r1244, %r461, %r1246;
	fma.rn.f32 	%r1248, %r1243, %r462, %r1245;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r610, %r1248, %r1247;
	bar.sync 	0;
	shl.b32 	%r1249, %r5, 14;
	shl.b32 	%r1250, %r5, 5;
	and.b32 	%r1251, %r1278, 3456;
	bfe.s32 	%r1252, %r2, 2, 1;
	and.b32 	%r1253, %r1252, 8208;
	or.b32 	%r1254, %r1250, %r1251;
	xor.b32 	%r1255, %r1253, %r777;
	or.b32 	%r1256, %r1255, %r1254;
	or.b32 	%r1257, %r1256, %r1249;
	add.s32 	%r531, %r93, %r1257;
	// begin inline asm
	st.shared.v4.b32 [ %r531 + 0 ], { %r532, %r533, %r534, %r535 };
	// end inline asm
	add.s32 	%r536, %r531, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r536 + 0 ], { %r537, %r538, %r539, %r540 };
	// end inline asm
	add.s32 	%r541, %r531, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r541 + 0 ], { %r542, %r543, %r544, %r545 };
	// end inline asm
	add.s32 	%r546, %r531, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r546 + 0 ], { %r547, %r548, %r549, %r550 };
	// end inline asm
	xor.b32 	%r1258, %r1257, 32;
	add.s32 	%r551, %r93, %r1258;
	// begin inline asm
	st.shared.v4.b32 [ %r551 + 0 ], { %r552, %r553, %r554, %r555 };
	// end inline asm
	add.s32 	%r556, %r551, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r556 + 0 ], { %r557, %r558, %r559, %r560 };
	// end inline asm
	add.s32 	%r561, %r551, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r561 + 0 ], { %r562, %r563, %r564, %r565 };
	// end inline asm
	add.s32 	%r566, %r551, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r566 + 0 ], { %r567, %r568, %r569, %r570 };
	// end inline asm
	xor.b32 	%r1259, %r1257, 64;
	add.s32 	%r571, %r93, %r1259;
	// begin inline asm
	st.shared.v4.b32 [ %r571 + 0 ], { %r572, %r573, %r574, %r575 };
	// end inline asm
	add.s32 	%r576, %r571, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r576 + 0 ], { %r577, %r578, %r579, %r580 };
	// end inline asm
	add.s32 	%r581, %r571, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r581 + 0 ], { %r582, %r583, %r584, %r585 };
	// end inline asm
	add.s32 	%r586, %r571, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r586 + 0 ], { %r587, %r588, %r589, %r590 };
	// end inline asm
	xor.b32 	%r1260, %r1257, 96;
	add.s32 	%r591, %r93, %r1260;
	// begin inline asm
	st.shared.v4.b32 [ %r591 + 0 ], { %r592, %r593, %r594, %r595 };
	// end inline asm
	add.s32 	%r596, %r591, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r596 + 0 ], { %r597, %r598, %r599, %r600 };
	// end inline asm
	add.s32 	%r601, %r591, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r601 + 0 ], { %r602, %r603, %r604, %r605 };
	// end inline asm
	add.s32 	%r606, %r591, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r606 + 0 ], { %r607, %r608, %r609, %r610 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r1261, %r2, 2;
	and.b32 	%r1262, %r1261, 896;
	shl.b32 	%r1263, %r736, 9;
	selp.b32 	%r1264, 0, 8208, %p24;
	or.b32 	%r1265, %r770, %r1262;
	xor.b32 	%r1266, %r1265, %r1264;
	or.b32 	%r1267, %r1266, %r1263;
	add.s32 	%r1268, %r93, %r1267;
	ld.shared.v4.b32 	{%r611, %r627, %r643, %r659}, [%r1268];
	ld.shared.v4.b32 	{%r615, %r631, %r647, %r663}, [%r1268+1024];
	ld.shared.v4.b32 	{%r619, %r635, %r651, %r667}, [%r1268+2048];
	ld.shared.v4.b32 	{%r623, %r639, %r655, %r671}, [%r1268+3072];
	xor.b32 	%r1269, %r1267, 32;
	add.s32 	%r1270, %r93, %r1269;
	ld.shared.v4.b32 	{%r612, %r628, %r644, %r660}, [%r1270+16384];
	ld.shared.v4.b32 	{%r616, %r632, %r648, %r664}, [%r1270+17408];
	ld.shared.v4.b32 	{%r620, %r636, %r652, %r668}, [%r1270+18432];
	ld.shared.v4.b32 	{%r624, %r640, %r656, %r672}, [%r1270+19456];
	xor.b32 	%r1271, %r1267, 64;
	add.s32 	%r1272, %r93, %r1271;
	ld.shared.v4.b32 	{%r613, %r629, %r645, %r661}, [%r1272+32768];
	ld.shared.v4.b32 	{%r617, %r633, %r649, %r665}, [%r1272+33792];
	ld.shared.v4.b32 	{%r621, %r637, %r653, %r669}, [%r1272+34816];
	ld.shared.v4.b32 	{%r625, %r641, %r657, %r673}, [%r1272+35840];
	xor.b32 	%r1273, %r1267, 96;
	add.s32 	%r1274, %r93, %r1273;
	ld.shared.v4.b32 	{%r614, %r630, %r646, %r662}, [%r1274+49152];
	ld.shared.v4.b32 	{%r618, %r634, %r650, %r666}, [%r1274+50176];
	ld.shared.v4.b32 	{%r622, %r638, %r654, %r670}, [%r1274+51200];
	ld.shared.v4.b32 	{%r626, %r642, %r658, %r674}, [%r1274+52224];
	.loc	1 356 8                         // sk06_mlp_down.py:356:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd74 + 0 ], { %r611, %r612, %r613, %r614 };
	// end inline asm
	// begin inline asm
	@%p9 st.global.v4.b32 [ %rd75 + 0 ], { %r615, %r616, %r617, %r618 };
	// end inline asm
	// begin inline asm
	@%p10 st.global.v4.b32 [ %rd76 + 0 ], { %r619, %r620, %r621, %r622 };
	// end inline asm
	// begin inline asm
	@%p11 st.global.v4.b32 [ %rd77 + 0 ], { %r623, %r624, %r625, %r626 };
	// end inline asm
	// begin inline asm
	@%p12 st.global.v4.b32 [ %rd78 + 0 ], { %r627, %r628, %r629, %r630 };
	// end inline asm
	// begin inline asm
	@%p13 st.global.v4.b32 [ %rd79 + 0 ], { %r631, %r632, %r633, %r634 };
	// end inline asm
	// begin inline asm
	@%p14 st.global.v4.b32 [ %rd80 + 0 ], { %r635, %r636, %r637, %r638 };
	// end inline asm
	// begin inline asm
	@%p15 st.global.v4.b32 [ %rd81 + 0 ], { %r639, %r640, %r641, %r642 };
	// end inline asm
	// begin inline asm
	@%p16 st.global.v4.b32 [ %rd82 + 0 ], { %r643, %r644, %r645, %r646 };
	// end inline asm
	// begin inline asm
	@%p17 st.global.v4.b32 [ %rd83 + 0 ], { %r647, %r648, %r649, %r650 };
	// end inline asm
	// begin inline asm
	@%p18 st.global.v4.b32 [ %rd84 + 0 ], { %r651, %r652, %r653, %r654 };
	// end inline asm
	// begin inline asm
	@%p19 st.global.v4.b32 [ %rd85 + 0 ], { %r655, %r656, %r657, %r658 };
	// end inline asm
	// begin inline asm
	@%p20 st.global.v4.b32 [ %rd86 + 0 ], { %r659, %r660, %r661, %r662 };
	// end inline asm
	// begin inline asm
	@%p21 st.global.v4.b32 [ %rd87 + 0 ], { %r663, %r664, %r665, %r666 };
	// end inline asm
	// begin inline asm
	@%p22 st.global.v4.b32 [ %rd88 + 0 ], { %r667, %r668, %r669, %r670 };
	// end inline asm
	// begin inline asm
	@%p23 st.global.v4.b32 [ %rd89 + 0 ], { %r671, %r672, %r673, %r674 };
	// end inline asm
	.loc	1 354 4                         // sk06_mlp_down.py:354:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/workspace/vllm/_genesis/kernels/sk06_mlp_down.py"
	.file	2 "/usr/local/lib/python3.12/dist-packages/triton/language/standard.py"
	.section	.debug_abbrev
	{
.b8 1                                   // Abbreviation Code
.b8 17                                  // DW_TAG_compile_unit
.b8 1                                   // DW_CHILDREN_yes
.b8 37                                  // DW_AT_producer
.b8 8                                   // DW_FORM_string
.b8 19                                  // DW_AT_language
.b8 5                                   // DW_FORM_data2
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 16                                  // DW_AT_stmt_list
.b8 6                                   // DW_FORM_data4
.b8 27                                  // DW_AT_comp_dir
.b8 8                                   // DW_FORM_string
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 2                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 0                                   // DW_CHILDREN_no
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 32                                  // DW_AT_inline
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 3                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 1                                   // DW_CHILDREN_yes
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 4                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 0                                   // DW_CHILDREN_no
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 5                                   // DW_FORM_data2
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 168                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0xa1 DW_TAG_compile_unit
.b8 116                                 // DW_AT_producer
.b8 114
.b8 105
.b8 116
.b8 111
.b8 110
.b8 0
.b8 2                                   // DW_AT_language
.b8 0
.b8 115                                 // DW_AT_name
.b8 107
.b8 48
.b8 54
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 100
.b8 111
.b8 119
.b8 110
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 119
.b8 111
.b8 114
.b8 107
.b8 115
.b8 112
.b8 97
.b8 99
.b8 101
.b8 47
.b8 118
.b8 108
.b8 108
.b8 109
.b8 47
.b8 95
.b8 103
.b8 101
.b8 110
.b8 101
.b8 115
.b8 105
.b8 115
.b8 47
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 115
.b8 0
.b8 2                                   // Abbrev [2] 0x4b:0x18 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 54
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 100
.b8 111
.b8 119
.b8 110
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x63:0x48 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 75                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x78:0x19 DW_TAG_inlined_subroutine
.b32 75                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 58                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x91:0x19 DW_TAG_inlined_subroutine
.b32 75                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 59                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_8 = _Nativo(
    "sk06_mlp_down/tile256x128x64_shift0_abi15",
    _PTX_8, "_sk06_mlp_down_kernel",
    warps=8, shared=73728,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 18],
    horneado={11: 1, 12: 1, 15: 1, 17: 1, 19: 1, 20: 256, 21: 128, 22: 64, 23: 8, 24: 128, 25: False},
    div16=[8, 9, 10, 13, 14, 16],
)

_PTX_9 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk06_mlp_down_kernel   // -- Begin function _sk06_mlp_down_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk06_mlp_down_kernel
.visible .entry _sk06_mlp_down_kernel(
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_6,
	.param .u32 _sk06_mlp_down_kernel_param_7,
	.param .u32 _sk06_mlp_down_kernel_param_8,
	.param .u32 _sk06_mlp_down_kernel_param_9,
	.param .u32 _sk06_mlp_down_kernel_param_10,
	.param .u32 _sk06_mlp_down_kernel_param_11,
	.param .u32 _sk06_mlp_down_kernel_param_12,
	.param .u32 _sk06_mlp_down_kernel_param_13,
	.param .u32 _sk06_mlp_down_kernel_param_14,
	.param .u32 _sk06_mlp_down_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_16,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_17
)
.reqntid 256
{
	.reg .pred 	%p<42>;
	.reg .b16 	%rs<257>;
	.reg .b32 	%r<1431>;
	.reg .b64 	%rd<249>;
	.loc	1 304 0                         // sk06_mlp_down.py:304:0
$L__func_begin0:
	.loc	1 304 0                         // sk06_mlp_down.py:304:0

// %bb.0:
	ld.param.b32 	%r19, [_sk06_mlp_down_kernel_param_14];
	ld.param.b32 	%r18, [_sk06_mlp_down_kernel_param_13];
	ld.param.b32 	%r17, [_sk06_mlp_down_kernel_param_12];
	ld.param.b32 	%r16, [_sk06_mlp_down_kernel_param_8];
	ld.param.b32 	%r15, [_sk06_mlp_down_kernel_param_7];
	ld.param.b64 	%rd13, [_sk06_mlp_down_kernel_param_5];
	ld.param.b64 	%rd12, [_sk06_mlp_down_kernel_param_4];
	ld.param.b64 	%rd11, [_sk06_mlp_down_kernel_param_3];
	ld.param.b64 	%rd10, [_sk06_mlp_down_kernel_param_2];
	ld.param.b64 	%rd9, [_sk06_mlp_down_kernel_param_1];
	ld.param.b64 	%rd8, [_sk06_mlp_down_kernel_param_0];
$L__tmp0:
	.loc	1 313 24                        // sk06_mlp_down.py:313:24
	mov.u32 	%r41, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk06_mlp_down.py:314:27 ]
	add.s32 	%r42, %r15, 255;
	.loc	2 43 30                         // standard.py:43:30 @[ sk06_mlp_down.py:314:27 ]
	shr.s32 	%r43, %r42, 31;
	shr.u32 	%r44, %r43, 24;
	add.s32 	%r45, %r42, %r44;
	shr.s32 	%r46, %r45, 8;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk06_mlp_down.py:315:27 ]
	add.s32 	%r47, %r16, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk06_mlp_down.py:315:27 ]
	shr.s32 	%r48, %r47, 31;
	shr.u32 	%r49, %r48, 25;
	add.s32 	%r50, %r47, %r49;
	shr.s32 	%r51, %r50, 7;
$L__tmp3:
	.loc	1 316 29                        // sk06_mlp_down.py:316:29
	shl.b32 	%r52, %r51, 3;
	.loc	1 317 22                        // sk06_mlp_down.py:317:22
	div.s32 	%r53, %r41, %r52;
	.loc	1 317 38                        // sk06_mlp_down.py:317:38
	shl.b32 	%r54, %r53, 3;
	ld.param.b32 	%r55, [_sk06_mlp_down_kernel_param_9];
	.loc	1 318 30                        // sk06_mlp_down.py:318:30
	sub.s32 	%r56, %r46, %r54;
	ld.param.b32 	%r57, [_sk06_mlp_down_kernel_param_10];
	.loc	1 318 39                        // sk06_mlp_down.py:318:39
	min.s32 	%r58, %r56, 8;
	ld.param.b32 	%r59, [_sk06_mlp_down_kernel_param_11];
	.loc	1 319 30                        // sk06_mlp_down.py:319:30
	mul.lo.s32 	%r60, %r53, %r52;
	sub.s32 	%r61, %r41, %r60;
	.loc	1 320 36                        // sk06_mlp_down.py:320:36
	div.s32 	%r62, %r61, %r58;
	.loc	1 319 46                        // sk06_mlp_down.py:319:46
	mul.lo.s32 	%r63, %r62, %r58;
	sub.s32 	%r64, %r61, %r63;
	.loc	1 319 23                        // sk06_mlp_down.py:319:23
	add.s32 	%r65, %r64, %r54;
	.loc	1 322 22                        // sk06_mlp_down.py:322:22
	shl.b32 	%r1, %r65, 8;
	.loc	1 322 45                        // sk06_mlp_down.py:322:45
	mov.u32 	%r2, %tid.x;
	shr.u32 	%r66, %r2, 2;
	bfe.u32 	%r67, %r2, 2, 6;
	or.b32 	%r68, %r67, 64;
	and.b32 	%r3, %r2, 255;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r69, %r1, %r67;
	or.b32 	%r70, %r1, %r68;
	or.b32 	%r71, %r69, 128;
	or.b32 	%r72, %r1, %r66;
	or.b32 	%r73, %r72, 192;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r74, %r69, %r15;
	rem.s32 	%r75, %r70, %r15;
	rem.s32 	%r76, %r71, %r15;
	rem.s32 	%r77, %r73, %r15;
	.loc	1 323 22                        // sk06_mlp_down.py:323:22
	shl.b32 	%r4, %r62, 7;
	.loc	1 323 45                        // sk06_mlp_down.py:323:45
	and.b32 	%r5, %r2, 3;
	and.b32 	%r6, %r2, 32;
	and.b32 	%r7, %r2, 15;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r78, %r4, %r67;
	or.b32 	%r79, %r4, %r68;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r80, %r78, %r16;
	rem.s32 	%r81, %r79, %r16;
	.loc	1 326 39                        // sk06_mlp_down.py:326:39
	mul.lo.s32 	%r82, %r74, %r57;
	mul.lo.s32 	%r83, %r75, %r57;
	mul.lo.s32 	%r84, %r76, %r57;
	mul.lo.s32 	%r85, %r77, %r57;
	.loc	1 326 21                        // sk06_mlp_down.py:326:21
	cvt.s64.s32 	%rd1, %r82;
	add.s64 	%rd32, %rd8, %rd1;
	cvt.s64.s32 	%rd2, %r83;
	add.s64 	%rd33, %rd8, %rd2;
	cvt.s64.s32 	%rd3, %r84;
	add.s64 	%rd34, %rd8, %rd3;
	cvt.s64.s32 	%rd4, %r85;
	add.s64 	%rd35, %rd8, %rd4;
	.loc	1 326 58                        // sk06_mlp_down.py:326:58
	shl.b32 	%r86, %r5, 4;
	.loc	1 326 51                        // sk06_mlp_down.py:326:51
	cvt.u64.u32 	%rd5, %r86;
	add.s64 	%rd14, %rd32, %rd5;
	add.s64 	%rd15, %rd33, %rd5;
	add.s64 	%rd16, %rd34, %rd5;
	add.s64 	%rd17, %rd35, %rd5;
	.loc	1 327 21                        // sk06_mlp_down.py:327:21
	add.s64 	%rd36, %rd9, %rd5;
	.loc	1 327 69                        // sk06_mlp_down.py:327:69
	mul.lo.s32 	%r87, %r80, %r59;
	mul.lo.s32 	%r88, %r81, %r59;
	.loc	1 327 51                        // sk06_mlp_down.py:327:51
	cvt.s64.s32 	%rd6, %r87;
	add.s64 	%rd18, %rd36, %rd6;
	cvt.s64.s32 	%rd7, %r88;
	add.s64 	%rd19, %rd36, %rd7;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	setp.gt.s32 	%p1, %r55, 63;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	shl.b32 	%r92, %r3, 4;
	shl.b32 	%r9, %r2, 1;
	and.b32 	%r10, %r9, 48;
	xor.b32 	%r93, %r92, %r10;
	mov.b32 	%r94, global_smem;
	add.s32 	%r20, %r94, %r93;
	selp.b32 	%r21, 16, 0, %p1;
	// begin inline asm
	cp.async.cg.shared.global [ %r20 + 0 ], [ %rd14 + 0 ], 0x10, %r21;
	// end inline asm
	add.s32 	%r22, %r20, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r22 + 0 ], [ %rd15 + 0 ], 0x10, %r21;
	// end inline asm
	add.s32 	%r23, %r20, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r23 + 0 ], [ %rd16 + 0 ], 0x10, %r21;
	// end inline asm
	add.s32 	%r24, %r20, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r24 + 0 ], [ %rd17 + 0 ], 0x10, %r21;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	add.s32 	%r25, %r20, 49152;
	// begin inline asm
	cp.async.cg.shared.global [ %r25 + 0 ], [ %rd18 + 0 ], 0x10, %r21;
	// end inline asm
	add.s32 	%r26, %r20, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r26 + 0 ], [ %rd19 + 0 ], 0x10, %r21;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	setp.gt.s32 	%p2, %r55, 127;
	.loc	1 344 18                        // sk06_mlp_down.py:344:18
	add.s64 	%rd20, %rd14, 64;
	add.s64 	%rd21, %rd15, 64;
	add.s64 	%rd22, %rd16, 64;
	add.s64 	%rd23, %rd17, 64;
	.loc	1 345 18                        // sk06_mlp_down.py:345:18
	add.s64 	%rd24, %rd18, 64;
	add.s64 	%rd25, %rd19, 64;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	bar.sync 	0;
	add.s32 	%r27, %r20, 16384;
	selp.b32 	%r28, 16, 0, %p2;
	// begin inline asm
	cp.async.cg.shared.global [ %r27 + 0 ], [ %rd20 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r29, %r20, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r29 + 0 ], [ %rd21 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r30, %r20, 24576;
	// begin inline asm
	cp.async.cg.shared.global [ %r30 + 0 ], [ %rd22 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r31, %r20, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r31 + 0 ], [ %rd23 + 0 ], 0x10, %r28;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	add.s32 	%r32, %r20, 57344;
	// begin inline asm
	cp.async.cg.shared.global [ %r32 + 0 ], [ %rd24 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r33, %r20, 61440;
	// begin inline asm
	cp.async.cg.shared.global [ %r33 + 0 ], [ %rd25 + 0 ], 0x10, %r28;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	setp.gt.s32 	%p3, %r55, 191;
	.loc	1 344 18                        // sk06_mlp_down.py:344:18
	add.s64 	%rd26, %rd14, 128;
	add.s64 	%rd27, %rd15, 128;
	add.s64 	%rd28, %rd16, 128;
	add.s64 	%rd29, %rd17, 128;
	.loc	1 345 18                        // sk06_mlp_down.py:345:18
	add.s64 	%rd30, %rd18, 128;
	add.s64 	%rd31, %rd19, 128;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	bar.sync 	0;
	add.s32 	%r34, %r20, 32768;
	selp.b32 	%r35, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r34 + 0 ], [ %rd26 + 0 ], 0x10, %r35;
	// end inline asm
	add.s32 	%r36, %r20, 36864;
	// begin inline asm
	cp.async.cg.shared.global [ %r36 + 0 ], [ %rd27 + 0 ], 0x10, %r35;
	// end inline asm
	add.s32 	%r37, %r20, 40960;
	// begin inline asm
	cp.async.cg.shared.global [ %r37 + 0 ], [ %rd28 + 0 ], 0x10, %r35;
	// end inline asm
	add.s32 	%r38, %r20, 45056;
	// begin inline asm
	cp.async.cg.shared.global [ %r38 + 0 ], [ %rd29 + 0 ], 0x10, %r35;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	add.s32 	%r39, %r20, 65536;
	// begin inline asm
	cp.async.cg.shared.global [ %r39 + 0 ], [ %rd30 + 0 ], 0x10, %r35;
	// end inline asm
	add.s32 	%r40, %r20, 69632;
	// begin inline asm
	cp.async.cg.shared.global [ %r40 + 0 ], [ %rd31 + 0 ], 0x10, %r35;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	@%p1 bra 	$L__BB0_2;
	bra.uni 	$L__BB0_1;
$L__BB0_2:                              // %.lr.ph
	.loc	1 0 23                          // sk06_mlp_down.py:0:23
	shr.s32 	%r89, %r55, 31;
	shr.u32 	%r90, %r89, 26;
	add.s32 	%r91, %r55, %r90;
	shr.s32 	%r8, %r91, 6;
	add.s32 	%r11, %r8, -3;
	shl.b32 	%r95, %r7, 6;
	shl.b32 	%r1301, %r2, 4;
	and.b32 	%r96, %r1301, 3072;
	shl.b32 	%r97, %r2, 3;
	and.b32 	%r98, %r97, 48;
	and.b32 	%r1302, %r2, 16;
	or.b32 	%r99, %r95, %r96;
	xor.b32 	%r100, %r98, %r1302;
	or.b32 	%r12, %r99, %r100;
	xor.b32 	%r13, %r12, 32;
	shl.b32 	%r101, %r2, 6;
	and.b32 	%r102, %r101, 448;
	shl.b32 	%r103, %r6, 4;
	or.b32 	%r104, %r102, %r98;
	xor.b32 	%r105, %r104, %r10;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	add.s32 	%r106, %r94, %r103;
	add.s32 	%r14, %r106, %r105;
	add.s64 	%rd37, %rd7, %rd9;
	add.s64 	%rd248, %rd37, 192;
	add.s64 	%rd38, %rd6, %rd9;
	add.s64 	%rd247, %rd38, 192;
	add.s64 	%rd39, %rd4, %rd8;
	add.s64 	%rd246, %rd39, 192;
	add.s64 	%rd40, %rd3, %rd8;
	add.s64 	%rd245, %rd40, 192;
	add.s64 	%rd41, %rd2, %rd8;
	add.s64 	%rd244, %rd41, 192;
	add.s64 	%rd42, %rd1, %rd8;
	add.s64 	%rd243, %rd42, 192;
	mov.b32 	%r1303, 0f00000000;
	mov.b32 	%r107, 0;
	mov.b32 	%r1299, 2;
	mov.b32 	%r1298, -1;
	mov.b32 	%r1300, %r107;
	mov.b32 	%r1304, %r1303;
	mov.b32 	%r1305, %r1303;
	mov.b32 	%r1306, %r1303;
	mov.b32 	%r1307, %r1303;
	mov.b32 	%r1308, %r1303;
	mov.b32 	%r1309, %r1303;
	mov.b32 	%r1310, %r1303;
	mov.b32 	%r1311, %r1303;
	mov.b32 	%r1312, %r1303;
	mov.b32 	%r1313, %r1303;
	mov.b32 	%r1314, %r1303;
	mov.b32 	%r1315, %r1303;
	mov.b32 	%r1316, %r1303;
	mov.b32 	%r1317, %r1303;
	mov.b32 	%r1318, %r1303;
	mov.b32 	%r1319, %r1303;
	mov.b32 	%r1320, %r1303;
	mov.b32 	%r1321, %r1303;
	mov.b32 	%r1322, %r1303;
	mov.b32 	%r1323, %r1303;
	mov.b32 	%r1324, %r1303;
	mov.b32 	%r1325, %r1303;
	mov.b32 	%r1326, %r1303;
	mov.b32 	%r1327, %r1303;
	mov.b32 	%r1328, %r1303;
	mov.b32 	%r1329, %r1303;
	mov.b32 	%r1330, %r1303;
	mov.b32 	%r1331, %r1303;
	mov.b32 	%r1332, %r1303;
	mov.b32 	%r1333, %r1303;
	mov.b32 	%r1334, %r1303;
	mov.b32 	%r1335, %r1303;
	mov.b32 	%r1336, %r1303;
	mov.b32 	%r1337, %r1303;
	mov.b32 	%r1338, %r1303;
	mov.b32 	%r1339, %r1303;
	mov.b32 	%r1340, %r1303;
	mov.b32 	%r1341, %r1303;
	mov.b32 	%r1342, %r1303;
	mov.b32 	%r1343, %r1303;
	mov.b32 	%r1344, %r1303;
	mov.b32 	%r1345, %r1303;
	mov.b32 	%r1346, %r1303;
	mov.b32 	%r1347, %r1303;
	mov.b32 	%r1348, %r1303;
	mov.b32 	%r1349, %r1303;
	mov.b32 	%r1350, %r1303;
	mov.b32 	%r1351, %r1303;
	mov.b32 	%r1352, %r1303;
	mov.b32 	%r1353, %r1303;
	mov.b32 	%r1354, %r1303;
	mov.b32 	%r1355, %r1303;
	mov.b32 	%r1356, %r1303;
	mov.b32 	%r1357, %r1303;
	mov.b32 	%r1358, %r1303;
	mov.b32 	%r1359, %r1303;
	mov.b32 	%r1360, %r1303;
	mov.b32 	%r1361, %r1303;
	mov.b32 	%r1362, %r1303;
	mov.b32 	%r1363, %r1303;
	mov.b32 	%r1364, %r1303;
	mov.b32 	%r1365, %r1303;
	mov.b32 	%r1366, %r1303;
	mov.b32 	%r1367, %r1303;
	mov.b32 	%r1368, %r1303;
	mov.b32 	%r1369, %r1303;
	mov.b32 	%r1370, %r1303;
	mov.b32 	%r1371, %r1303;
	mov.b32 	%r1372, %r1303;
	mov.b32 	%r1373, %r1303;
	mov.b32 	%r1374, %r1303;
	mov.b32 	%r1375, %r1303;
	mov.b32 	%r1376, %r1303;
	mov.b32 	%r1377, %r1303;
	mov.b32 	%r1378, %r1303;
	mov.b32 	%r1379, %r1303;
	mov.b32 	%r1380, %r1303;
	mov.b32 	%r1381, %r1303;
	mov.b32 	%r1382, %r1303;
	mov.b32 	%r1383, %r1303;
	mov.b32 	%r1384, %r1303;
	mov.b32 	%r1385, %r1303;
	mov.b32 	%r1386, %r1303;
	mov.b32 	%r1387, %r1303;
	mov.b32 	%r1388, %r1303;
	mov.b32 	%r1389, %r1303;
	mov.b32 	%r1390, %r1303;
	mov.b32 	%r1391, %r1303;
	mov.b32 	%r1392, %r1303;
	mov.b32 	%r1393, %r1303;
	mov.b32 	%r1394, %r1303;
	mov.b32 	%r1395, %r1303;
	mov.b32 	%r1396, %r1303;
	mov.b32 	%r1397, %r1303;
	mov.b32 	%r1398, %r1303;
	mov.b32 	%r1399, %r1303;
	mov.b32 	%r1400, %r1303;
	mov.b32 	%r1401, %r1303;
	mov.b32 	%r1402, %r1303;
	mov.b32 	%r1403, %r1303;
	mov.b32 	%r1404, %r1303;
	mov.b32 	%r1405, %r1303;
	mov.b32 	%r1406, %r1303;
	mov.b32 	%r1407, %r1303;
	mov.b32 	%r1408, %r1303;
	mov.b32 	%r1409, %r1303;
	mov.b32 	%r1410, %r1303;
	mov.b32 	%r1411, %r1303;
	mov.b32 	%r1412, %r1303;
	mov.b32 	%r1413, %r1303;
	mov.b32 	%r1414, %r1303;
	mov.b32 	%r1415, %r1303;
	mov.b32 	%r1416, %r1303;
	mov.b32 	%r1417, %r1303;
	mov.b32 	%r1418, %r1303;
	mov.b32 	%r1419, %r1303;
	mov.b32 	%r1420, %r1303;
	mov.b32 	%r1421, %r1303;
	mov.b32 	%r1422, %r1303;
	mov.b32 	%r1423, %r1303;
	mov.b32 	%r1424, %r1303;
	mov.b32 	%r1425, %r1303;
	mov.b32 	%r1426, %r1303;
	mov.b32 	%r1427, %r1303;
	mov.b32 	%r1428, %r1303;
	mov.b32 	%r1429, %r1303;
	mov.b32 	%r1430, %r1303;
$L__BB0_3:                              // =>This Inner Loop Header: Depth=1
	setp.lt.s32 	%p4, %r1300, %r11;
	add.s32 	%r307, %r1298, 1;
	setp.gt.s32 	%p5, %r307, 2;
	selp.b32 	%r1298, 0, %r307, %p5;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	cp.async.wait_group 	4;
	bar.sync 	0;
	shl.b32 	%r308, %r1298, 14;
	add.s32 	%r309, %r94, %r308;
	add.s32 	%r310, %r309, %r12;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r108, %r109, %r110, %r111}, [%r310];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r128, %r129, %r130, %r131}, [%r310+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r132, %r133, %r134, %r135}, [%r310+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r136, %r137, %r138, %r139}, [%r310+12288];
	add.s32 	%r311, %r309, %r13;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r144, %r145, %r146, %r147}, [%r311];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r196, %r197, %r198, %r199}, [%r311+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r232, %r233, %r234, %r235}, [%r311+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r268, %r269, %r270, %r271}, [%r311+12288];
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	shl.b32 	%r312, %r1298, 13;
	add.s32 	%r313, %r14, %r312;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r112, %r113, %r148, %r149}, [%r313+49152];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r114, %r115, %r154, %r155}, [%r313+50176];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r116, %r117, %r160, %r161}, [%r313+51200];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r118, %r119, %r166, %r167}, [%r313+52224];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r120, %r121, %r172, %r173}, [%r313+53248];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r122, %r123, %r178, %r179}, [%r313+54272];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r124, %r125, %r184, %r185}, [%r313+55296];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r126, %r127, %r190, %r191}, [%r313+56320];
	.loc	1 338 36                        // sk06_mlp_down.py:338:36
	mov.b32 	%r140, %r107;
	mov.b32 	%r141, %r107;
	mov.b32 	%r142, %r107;
	mov.b32 	%r143, %r107;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r140, %r141, %r142, %r143 }, { %r108, %r109, %r110, %r111 }, { %r112, %r113 }, { %r140, %r141, %r142, %r143 };
	// end inline asm
	mov.b32 	%r150, %r107;
	mov.b32 	%r151, %r107;
	mov.b32 	%r152, %r107;
	mov.b32 	%r153, %r107;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r150, %r151, %r152, %r153 }, { %r108, %r109, %r110, %r111 }, { %r114, %r115 }, { %r150, %r151, %r152, %r153 };
	// end inline asm
	mov.b32 	%r156, %r107;
	mov.b32 	%r157, %r107;
	mov.b32 	%r158, %r107;
	mov.b32 	%r159, %r107;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r156, %r157, %r158, %r159 }, { %r108, %r109, %r110, %r111 }, { %r116, %r117 }, { %r156, %r157, %r158, %r159 };
	// end inline asm
	mov.b32 	%r162, %r107;
	mov.b32 	%r163, %r107;
	mov.b32 	%r164, %r107;
	mov.b32 	%r165, %r107;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r162, %r163, %r164, %r165 }, { %r108, %r109, %r110, %r111 }, { %r118, %r119 }, { %r162, %r163, %r164, %r165 };
	// end inline asm
	mov.b32 	%r168, %r107;
	mov.b32 	%r169, %r107;
	mov.b32 	%r170, %r107;
	mov.b32 	%r171, %r107;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r168, %r169, %r170, %r171 }, { %r108, %r109, %r110, %r111 }, { %r120, %r121 }, { %r168, %r169, %r170, %r171 };
	// end inline asm
	mov.b32 	%r174, %r107;
	mov.b32 	%r175, %r107;
	mov.b32 	%r176, %r107;
	mov.b32 	%r177, %r107;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r174, %r175, %r176, %r177 }, { %r108, %r109, %r110, %r111 }, { %r122, %r123 }, { %r174, %r175, %r176, %r177 };
	// end inline asm
	mov.b32 	%r180, %r107;
	mov.b32 	%r181, %r107;
	mov.b32 	%r182, %r107;
	mov.b32 	%r183, %r107;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r180, %r181, %r182, %r183 }, { %r108, %r109, %r110, %r111 }, { %r124, %r125 }, { %r180, %r181, %r182, %r183 };
	// end inline asm
	mov.b32 	%r186, %r107;
	mov.b32 	%r187, %r107;
	mov.b32 	%r188, %r107;
	mov.b32 	%r189, %r107;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r186, %r187, %r188, %r189 }, { %r108, %r109, %r110, %r111 }, { %r126, %r127 }, { %r186, %r187, %r188, %r189 };
	// end inline asm
	mov.b32 	%r192, %r107;
	mov.b32 	%r193, %r107;
	mov.b32 	%r194, %r107;
	mov.b32 	%r195, %r107;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r192, %r193, %r194, %r195 }, { %r128, %r129, %r130, %r131 }, { %r112, %r113 }, { %r192, %r193, %r194, %r195 };
	// end inline asm
	mov.b32 	%r200, %r107;
	mov.b32 	%r201, %r107;
	mov.b32 	%r202, %r107;
	mov.b32 	%r203, %r107;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r200, %r201, %r202, %r203 }, { %r128, %r129, %r130, %r131 }, { %r114, %r115 }, { %r200, %r201, %r202, %r203 };
	// end inline asm
	mov.b32 	%r204, %r107;
	mov.b32 	%r205, %r107;
	mov.b32 	%r206, %r107;
	mov.b32 	%r207, %r107;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r204, %r205, %r206, %r207 }, { %r128, %r129, %r130, %r131 }, { %r116, %r117 }, { %r204, %r205, %r206, %r207 };
	// end inline asm
	mov.b32 	%r208, %r107;
	mov.b32 	%r209, %r107;
	mov.b32 	%r210, %r107;
	mov.b32 	%r211, %r107;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r208, %r209, %r210, %r211 }, { %r128, %r129, %r130, %r131 }, { %r118, %r119 }, { %r208, %r209, %r210, %r211 };
	// end inline asm
	mov.b32 	%r212, %r107;
	mov.b32 	%r213, %r107;
	mov.b32 	%r214, %r107;
	mov.b32 	%r215, %r107;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r212, %r213, %r214, %r215 }, { %r128, %r129, %r130, %r131 }, { %r120, %r121 }, { %r212, %r213, %r214, %r215 };
	// end inline asm
	mov.b32 	%r216, %r107;
	mov.b32 	%r217, %r107;
	mov.b32 	%r218, %r107;
	mov.b32 	%r219, %r107;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r216, %r217, %r218, %r219 }, { %r128, %r129, %r130, %r131 }, { %r122, %r123 }, { %r216, %r217, %r218, %r219 };
	// end inline asm
	mov.b32 	%r220, %r107;
	mov.b32 	%r221, %r107;
	mov.b32 	%r222, %r107;
	mov.b32 	%r223, %r107;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r220, %r221, %r222, %r223 }, { %r128, %r129, %r130, %r131 }, { %r124, %r125 }, { %r220, %r221, %r222, %r223 };
	// end inline asm
	mov.b32 	%r224, %r107;
	mov.b32 	%r225, %r107;
	mov.b32 	%r226, %r107;
	mov.b32 	%r227, %r107;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r224, %r225, %r226, %r227 }, { %r128, %r129, %r130, %r131 }, { %r126, %r127 }, { %r224, %r225, %r226, %r227 };
	// end inline asm
	mov.b32 	%r228, %r107;
	mov.b32 	%r229, %r107;
	mov.b32 	%r230, %r107;
	mov.b32 	%r231, %r107;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r228, %r229, %r230, %r231 }, { %r132, %r133, %r134, %r135 }, { %r112, %r113 }, { %r228, %r229, %r230, %r231 };
	// end inline asm
	mov.b32 	%r236, %r107;
	mov.b32 	%r237, %r107;
	mov.b32 	%r238, %r107;
	mov.b32 	%r239, %r107;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r236, %r237, %r238, %r239 }, { %r132, %r133, %r134, %r135 }, { %r114, %r115 }, { %r236, %r237, %r238, %r239 };
	// end inline asm
	mov.b32 	%r240, %r107;
	mov.b32 	%r241, %r107;
	mov.b32 	%r242, %r107;
	mov.b32 	%r243, %r107;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r240, %r241, %r242, %r243 }, { %r132, %r133, %r134, %r135 }, { %r116, %r117 }, { %r240, %r241, %r242, %r243 };
	// end inline asm
	mov.b32 	%r244, %r107;
	mov.b32 	%r245, %r107;
	mov.b32 	%r246, %r107;
	mov.b32 	%r247, %r107;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r244, %r245, %r246, %r247 }, { %r132, %r133, %r134, %r135 }, { %r118, %r119 }, { %r244, %r245, %r246, %r247 };
	// end inline asm
	mov.b32 	%r248, %r107;
	mov.b32 	%r249, %r107;
	mov.b32 	%r250, %r107;
	mov.b32 	%r251, %r107;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r248, %r249, %r250, %r251 }, { %r132, %r133, %r134, %r135 }, { %r120, %r121 }, { %r248, %r249, %r250, %r251 };
	// end inline asm
	mov.b32 	%r252, %r107;
	mov.b32 	%r253, %r107;
	mov.b32 	%r254, %r107;
	mov.b32 	%r255, %r107;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r252, %r253, %r254, %r255 }, { %r132, %r133, %r134, %r135 }, { %r122, %r123 }, { %r252, %r253, %r254, %r255 };
	// end inline asm
	mov.b32 	%r256, %r107;
	mov.b32 	%r257, %r107;
	mov.b32 	%r258, %r107;
	mov.b32 	%r259, %r107;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r256, %r257, %r258, %r259 }, { %r132, %r133, %r134, %r135 }, { %r124, %r125 }, { %r256, %r257, %r258, %r259 };
	// end inline asm
	mov.b32 	%r260, %r107;
	mov.b32 	%r261, %r107;
	mov.b32 	%r262, %r107;
	mov.b32 	%r263, %r107;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r260, %r261, %r262, %r263 }, { %r132, %r133, %r134, %r135 }, { %r126, %r127 }, { %r260, %r261, %r262, %r263 };
	// end inline asm
	mov.b32 	%r264, %r107;
	mov.b32 	%r265, %r107;
	mov.b32 	%r266, %r107;
	mov.b32 	%r267, %r107;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r264, %r265, %r266, %r267 }, { %r136, %r137, %r138, %r139 }, { %r112, %r113 }, { %r264, %r265, %r266, %r267 };
	// end inline asm
	mov.b32 	%r272, %r107;
	mov.b32 	%r273, %r107;
	mov.b32 	%r274, %r107;
	mov.b32 	%r275, %r107;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r272, %r273, %r274, %r275 }, { %r136, %r137, %r138, %r139 }, { %r114, %r115 }, { %r272, %r273, %r274, %r275 };
	// end inline asm
	mov.b32 	%r276, %r107;
	mov.b32 	%r277, %r107;
	mov.b32 	%r278, %r107;
	mov.b32 	%r279, %r107;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r276, %r277, %r278, %r279 }, { %r136, %r137, %r138, %r139 }, { %r116, %r117 }, { %r276, %r277, %r278, %r279 };
	// end inline asm
	mov.b32 	%r280, %r107;
	mov.b32 	%r281, %r107;
	mov.b32 	%r282, %r107;
	mov.b32 	%r283, %r107;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r280, %r281, %r282, %r283 }, { %r136, %r137, %r138, %r139 }, { %r118, %r119 }, { %r280, %r281, %r282, %r283 };
	// end inline asm
	mov.b32 	%r284, %r107;
	mov.b32 	%r285, %r107;
	mov.b32 	%r286, %r107;
	mov.b32 	%r287, %r107;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r284, %r285, %r286, %r287 }, { %r136, %r137, %r138, %r139 }, { %r120, %r121 }, { %r284, %r285, %r286, %r287 };
	// end inline asm
	mov.b32 	%r288, %r107;
	mov.b32 	%r289, %r107;
	mov.b32 	%r290, %r107;
	mov.b32 	%r291, %r107;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r288, %r289, %r290, %r291 }, { %r136, %r137, %r138, %r139 }, { %r122, %r123 }, { %r288, %r289, %r290, %r291 };
	// end inline asm
	mov.b32 	%r292, %r107;
	mov.b32 	%r293, %r107;
	mov.b32 	%r294, %r107;
	mov.b32 	%r295, %r107;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r292, %r293, %r294, %r295 }, { %r136, %r137, %r138, %r139 }, { %r124, %r125 }, { %r292, %r293, %r294, %r295 };
	// end inline asm
	mov.b32 	%r299, %r107;
	mov.b32 	%r296, %r107;
	mov.b32 	%r297, %r107;
	mov.b32 	%r298, %r107;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r296, %r297, %r298, %r299 }, { %r136, %r137, %r138, %r139 }, { %r126, %r127 }, { %r296, %r297, %r298, %r299 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r140, %r141, %r142, %r143 }, { %r144, %r145, %r146, %r147 }, { %r148, %r149 }, { %r140, %r141, %r142, %r143 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r150, %r151, %r152, %r153 }, { %r144, %r145, %r146, %r147 }, { %r154, %r155 }, { %r150, %r151, %r152, %r153 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r156, %r157, %r158, %r159 }, { %r144, %r145, %r146, %r147 }, { %r160, %r161 }, { %r156, %r157, %r158, %r159 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r162, %r163, %r164, %r165 }, { %r144, %r145, %r146, %r147 }, { %r166, %r167 }, { %r162, %r163, %r164, %r165 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r168, %r169, %r170, %r171 }, { %r144, %r145, %r146, %r147 }, { %r172, %r173 }, { %r168, %r169, %r170, %r171 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r174, %r175, %r176, %r177 }, { %r144, %r145, %r146, %r147 }, { %r178, %r179 }, { %r174, %r175, %r176, %r177 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r180, %r181, %r182, %r183 }, { %r144, %r145, %r146, %r147 }, { %r184, %r185 }, { %r180, %r181, %r182, %r183 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r186, %r187, %r188, %r189 }, { %r144, %r145, %r146, %r147 }, { %r190, %r191 }, { %r186, %r187, %r188, %r189 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r192, %r193, %r194, %r195 }, { %r196, %r197, %r198, %r199 }, { %r148, %r149 }, { %r192, %r193, %r194, %r195 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r200, %r201, %r202, %r203 }, { %r196, %r197, %r198, %r199 }, { %r154, %r155 }, { %r200, %r201, %r202, %r203 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r204, %r205, %r206, %r207 }, { %r196, %r197, %r198, %r199 }, { %r160, %r161 }, { %r204, %r205, %r206, %r207 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r208, %r209, %r210, %r211 }, { %r196, %r197, %r198, %r199 }, { %r166, %r167 }, { %r208, %r209, %r210, %r211 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r212, %r213, %r214, %r215 }, { %r196, %r197, %r198, %r199 }, { %r172, %r173 }, { %r212, %r213, %r214, %r215 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r216, %r217, %r218, %r219 }, { %r196, %r197, %r198, %r199 }, { %r178, %r179 }, { %r216, %r217, %r218, %r219 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r220, %r221, %r222, %r223 }, { %r196, %r197, %r198, %r199 }, { %r184, %r185 }, { %r220, %r221, %r222, %r223 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r224, %r225, %r226, %r227 }, { %r196, %r197, %r198, %r199 }, { %r190, %r191 }, { %r224, %r225, %r226, %r227 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r228, %r229, %r230, %r231 }, { %r232, %r233, %r234, %r235 }, { %r148, %r149 }, { %r228, %r229, %r230, %r231 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r236, %r237, %r238, %r239 }, { %r232, %r233, %r234, %r235 }, { %r154, %r155 }, { %r236, %r237, %r238, %r239 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r240, %r241, %r242, %r243 }, { %r232, %r233, %r234, %r235 }, { %r160, %r161 }, { %r240, %r241, %r242, %r243 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r244, %r245, %r246, %r247 }, { %r232, %r233, %r234, %r235 }, { %r166, %r167 }, { %r244, %r245, %r246, %r247 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r248, %r249, %r250, %r251 }, { %r232, %r233, %r234, %r235 }, { %r172, %r173 }, { %r248, %r249, %r250, %r251 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r252, %r253, %r254, %r255 }, { %r232, %r233, %r234, %r235 }, { %r178, %r179 }, { %r252, %r253, %r254, %r255 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r256, %r257, %r258, %r259 }, { %r232, %r233, %r234, %r235 }, { %r184, %r185 }, { %r256, %r257, %r258, %r259 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r260, %r261, %r262, %r263 }, { %r232, %r233, %r234, %r235 }, { %r190, %r191 }, { %r260, %r261, %r262, %r263 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r264, %r265, %r266, %r267 }, { %r268, %r269, %r270, %r271 }, { %r148, %r149 }, { %r264, %r265, %r266, %r267 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r272, %r273, %r274, %r275 }, { %r268, %r269, %r270, %r271 }, { %r154, %r155 }, { %r272, %r273, %r274, %r275 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r276, %r277, %r278, %r279 }, { %r268, %r269, %r270, %r271 }, { %r160, %r161 }, { %r276, %r277, %r278, %r279 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r280, %r281, %r282, %r283 }, { %r268, %r269, %r270, %r271 }, { %r166, %r167 }, { %r280, %r281, %r282, %r283 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r284, %r285, %r286, %r287 }, { %r268, %r269, %r270, %r271 }, { %r172, %r173 }, { %r284, %r285, %r286, %r287 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r288, %r289, %r290, %r291 }, { %r268, %r269, %r270, %r271 }, { %r178, %r179 }, { %r288, %r289, %r290, %r291 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r292, %r293, %r294, %r295 }, { %r268, %r269, %r270, %r271 }, { %r184, %r185 }, { %r292, %r293, %r294, %r295 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r296, %r297, %r298, %r299 }, { %r268, %r269, %r270, %r271 }, { %r190, %r191 }, { %r296, %r297, %r298, %r299 };
	// end inline asm
	.loc	1 343 19                        // sk06_mlp_down.py:343:19
	cvt.rn.f32.s32 	%r314, %r299;
	cvt.rn.f32.s32 	%r315, %r298;
	cvt.rn.f32.s32 	%r316, %r297;
	cvt.rn.f32.s32 	%r317, %r296;
	cvt.rn.f32.s32 	%r318, %r295;
	cvt.rn.f32.s32 	%r319, %r294;
	cvt.rn.f32.s32 	%r320, %r293;
	cvt.rn.f32.s32 	%r321, %r292;
	cvt.rn.f32.s32 	%r322, %r291;
	cvt.rn.f32.s32 	%r323, %r290;
	cvt.rn.f32.s32 	%r324, %r289;
	cvt.rn.f32.s32 	%r325, %r288;
	cvt.rn.f32.s32 	%r326, %r287;
	cvt.rn.f32.s32 	%r327, %r286;
	cvt.rn.f32.s32 	%r328, %r285;
	cvt.rn.f32.s32 	%r329, %r284;
	cvt.rn.f32.s32 	%r330, %r283;
	cvt.rn.f32.s32 	%r331, %r282;
	cvt.rn.f32.s32 	%r332, %r281;
	cvt.rn.f32.s32 	%r333, %r280;
	cvt.rn.f32.s32 	%r334, %r279;
	cvt.rn.f32.s32 	%r335, %r278;
	cvt.rn.f32.s32 	%r336, %r277;
	cvt.rn.f32.s32 	%r337, %r276;
	cvt.rn.f32.s32 	%r338, %r275;
	cvt.rn.f32.s32 	%r339, %r274;
	cvt.rn.f32.s32 	%r340, %r273;
	cvt.rn.f32.s32 	%r341, %r272;
	cvt.rn.f32.s32 	%r342, %r267;
	cvt.rn.f32.s32 	%r343, %r266;
	cvt.rn.f32.s32 	%r344, %r265;
	cvt.rn.f32.s32 	%r345, %r264;
	cvt.rn.f32.s32 	%r346, %r263;
	cvt.rn.f32.s32 	%r347, %r262;
	cvt.rn.f32.s32 	%r348, %r261;
	cvt.rn.f32.s32 	%r349, %r260;
	cvt.rn.f32.s32 	%r350, %r259;
	cvt.rn.f32.s32 	%r351, %r258;
	cvt.rn.f32.s32 	%r352, %r257;
	cvt.rn.f32.s32 	%r353, %r256;
	cvt.rn.f32.s32 	%r354, %r255;
	cvt.rn.f32.s32 	%r355, %r254;
	cvt.rn.f32.s32 	%r356, %r253;
	cvt.rn.f32.s32 	%r357, %r252;
	cvt.rn.f32.s32 	%r358, %r251;
	cvt.rn.f32.s32 	%r359, %r250;
	cvt.rn.f32.s32 	%r360, %r249;
	cvt.rn.f32.s32 	%r361, %r248;
	cvt.rn.f32.s32 	%r362, %r247;
	cvt.rn.f32.s32 	%r363, %r246;
	cvt.rn.f32.s32 	%r364, %r245;
	cvt.rn.f32.s32 	%r365, %r244;
	cvt.rn.f32.s32 	%r366, %r243;
	cvt.rn.f32.s32 	%r367, %r240;
	cvt.rn.f32.s32 	%r368, %r241;
	cvt.rn.f32.s32 	%r369, %r242;
	cvt.rn.f32.s32 	%r370, %r236;
	cvt.rn.f32.s32 	%r371, %r237;
	cvt.rn.f32.s32 	%r372, %r238;
	cvt.rn.f32.s32 	%r373, %r239;
	cvt.rn.f32.s32 	%r374, %r228;
	cvt.rn.f32.s32 	%r375, %r229;
	cvt.rn.f32.s32 	%r376, %r230;
	cvt.rn.f32.s32 	%r377, %r231;
	cvt.rn.f32.s32 	%r378, %r224;
	cvt.rn.f32.s32 	%r379, %r225;
	cvt.rn.f32.s32 	%r380, %r226;
	cvt.rn.f32.s32 	%r381, %r227;
	cvt.rn.f32.s32 	%r382, %r220;
	cvt.rn.f32.s32 	%r383, %r221;
	cvt.rn.f32.s32 	%r384, %r222;
	cvt.rn.f32.s32 	%r385, %r223;
	cvt.rn.f32.s32 	%r386, %r216;
	cvt.rn.f32.s32 	%r387, %r217;
	cvt.rn.f32.s32 	%r388, %r218;
	cvt.rn.f32.s32 	%r389, %r219;
	cvt.rn.f32.s32 	%r390, %r212;
	cvt.rn.f32.s32 	%r391, %r213;
	cvt.rn.f32.s32 	%r392, %r214;
	cvt.rn.f32.s32 	%r393, %r215;
	cvt.rn.f32.s32 	%r394, %r208;
	cvt.rn.f32.s32 	%r395, %r209;
	cvt.rn.f32.s32 	%r396, %r210;
	cvt.rn.f32.s32 	%r397, %r211;
	cvt.rn.f32.s32 	%r398, %r204;
	cvt.rn.f32.s32 	%r399, %r205;
	cvt.rn.f32.s32 	%r400, %r206;
	cvt.rn.f32.s32 	%r401, %r207;
	cvt.rn.f32.s32 	%r402, %r200;
	cvt.rn.f32.s32 	%r403, %r201;
	cvt.rn.f32.s32 	%r404, %r202;
	cvt.rn.f32.s32 	%r405, %r203;
	cvt.rn.f32.s32 	%r406, %r192;
	cvt.rn.f32.s32 	%r407, %r193;
	cvt.rn.f32.s32 	%r408, %r194;
	cvt.rn.f32.s32 	%r409, %r195;
	cvt.rn.f32.s32 	%r410, %r186;
	cvt.rn.f32.s32 	%r411, %r187;
	cvt.rn.f32.s32 	%r412, %r188;
	cvt.rn.f32.s32 	%r413, %r189;
	cvt.rn.f32.s32 	%r414, %r180;
	cvt.rn.f32.s32 	%r415, %r181;
	cvt.rn.f32.s32 	%r416, %r182;
	cvt.rn.f32.s32 	%r417, %r183;
	cvt.rn.f32.s32 	%r418, %r174;
	cvt.rn.f32.s32 	%r419, %r175;
	cvt.rn.f32.s32 	%r420, %r176;
	cvt.rn.f32.s32 	%r421, %r177;
	cvt.rn.f32.s32 	%r422, %r168;
	cvt.rn.f32.s32 	%r423, %r169;
	cvt.rn.f32.s32 	%r424, %r170;
	cvt.rn.f32.s32 	%r425, %r171;
	cvt.rn.f32.s32 	%r426, %r162;
	cvt.rn.f32.s32 	%r427, %r163;
	cvt.rn.f32.s32 	%r428, %r164;
	cvt.rn.f32.s32 	%r429, %r165;
	cvt.rn.f32.s32 	%r430, %r156;
	cvt.rn.f32.s32 	%r431, %r157;
	cvt.rn.f32.s32 	%r432, %r158;
	cvt.rn.f32.s32 	%r433, %r159;
	cvt.rn.f32.s32 	%r434, %r150;
	cvt.rn.f32.s32 	%r435, %r151;
	cvt.rn.f32.s32 	%r436, %r152;
	cvt.rn.f32.s32 	%r437, %r153;
	cvt.rn.f32.s32 	%r438, %r140;
	cvt.rn.f32.s32 	%r439, %r141;
	cvt.rn.f32.s32 	%r440, %r142;
	cvt.rn.f32.s32 	%r441, %r143;
	add.f32 	%r1306, %r1306, %r441;
	add.f32 	%r1305, %r1305, %r440;
	add.f32 	%r1304, %r1304, %r439;
	add.f32 	%r1303, %r1303, %r438;
	add.f32 	%r1310, %r1310, %r437;
	add.f32 	%r1309, %r1309, %r436;
	add.f32 	%r1308, %r1308, %r435;
	add.f32 	%r1307, %r1307, %r434;
	add.f32 	%r1314, %r1314, %r433;
	add.f32 	%r1313, %r1313, %r432;
	add.f32 	%r1312, %r1312, %r431;
	add.f32 	%r1311, %r1311, %r430;
	add.f32 	%r1318, %r1318, %r429;
	add.f32 	%r1317, %r1317, %r428;
	add.f32 	%r1316, %r1316, %r427;
	add.f32 	%r1315, %r1315, %r426;
	add.f32 	%r1322, %r1322, %r425;
	add.f32 	%r1321, %r1321, %r424;
	add.f32 	%r1320, %r1320, %r423;
	add.f32 	%r1319, %r1319, %r422;
	add.f32 	%r1326, %r1326, %r421;
	add.f32 	%r1325, %r1325, %r420;
	add.f32 	%r1324, %r1324, %r419;
	add.f32 	%r1323, %r1323, %r418;
	add.f32 	%r1330, %r1330, %r417;
	add.f32 	%r1329, %r1329, %r416;
	add.f32 	%r1328, %r1328, %r415;
	add.f32 	%r1327, %r1327, %r414;
	add.f32 	%r1334, %r1334, %r413;
	add.f32 	%r1333, %r1333, %r412;
	add.f32 	%r1332, %r1332, %r411;
	add.f32 	%r1331, %r1331, %r410;
	add.f32 	%r1338, %r1338, %r409;
	add.f32 	%r1337, %r1337, %r408;
	add.f32 	%r1336, %r1336, %r407;
	add.f32 	%r1335, %r1335, %r406;
	add.f32 	%r1342, %r1342, %r405;
	add.f32 	%r1341, %r1341, %r404;
	add.f32 	%r1340, %r1340, %r403;
	add.f32 	%r1339, %r1339, %r402;
	add.f32 	%r1346, %r1346, %r401;
	add.f32 	%r1345, %r1345, %r400;
	add.f32 	%r1344, %r1344, %r399;
	add.f32 	%r1343, %r1343, %r398;
	add.f32 	%r1350, %r1350, %r397;
	add.f32 	%r1349, %r1349, %r396;
	add.f32 	%r1348, %r1348, %r395;
	add.f32 	%r1347, %r1347, %r394;
	add.f32 	%r1354, %r1354, %r393;
	add.f32 	%r1353, %r1353, %r392;
	add.f32 	%r1352, %r1352, %r391;
	add.f32 	%r1351, %r1351, %r390;
	add.f32 	%r1358, %r1358, %r389;
	add.f32 	%r1357, %r1357, %r388;
	add.f32 	%r1356, %r1356, %r387;
	add.f32 	%r1355, %r1355, %r386;
	add.f32 	%r1362, %r1362, %r385;
	add.f32 	%r1361, %r1361, %r384;
	add.f32 	%r1360, %r1360, %r383;
	add.f32 	%r1359, %r1359, %r382;
	add.f32 	%r1366, %r1366, %r381;
	add.f32 	%r1365, %r1365, %r380;
	add.f32 	%r1364, %r1364, %r379;
	add.f32 	%r1363, %r1363, %r378;
	add.f32 	%r1370, %r1370, %r377;
	add.f32 	%r1369, %r1369, %r376;
	add.f32 	%r1368, %r1368, %r375;
	add.f32 	%r1367, %r1367, %r374;
	add.f32 	%r1374, %r1374, %r373;
	add.f32 	%r1373, %r1373, %r372;
	add.f32 	%r1372, %r1372, %r371;
	add.f32 	%r1371, %r1371, %r370;
	add.f32 	%r1377, %r1377, %r369;
	add.f32 	%r1376, %r1376, %r368;
	add.f32 	%r1375, %r1375, %r367;
	add.f32 	%r1378, %r1378, %r366;
	add.f32 	%r1379, %r1379, %r365;
	add.f32 	%r1380, %r1380, %r364;
	add.f32 	%r1381, %r1381, %r363;
	add.f32 	%r1382, %r1382, %r362;
	add.f32 	%r1383, %r1383, %r361;
	add.f32 	%r1384, %r1384, %r360;
	add.f32 	%r1385, %r1385, %r359;
	add.f32 	%r1386, %r1386, %r358;
	add.f32 	%r1387, %r1387, %r357;
	add.f32 	%r1388, %r1388, %r356;
	add.f32 	%r1389, %r1389, %r355;
	add.f32 	%r1390, %r1390, %r354;
	add.f32 	%r1391, %r1391, %r353;
	add.f32 	%r1392, %r1392, %r352;
	add.f32 	%r1393, %r1393, %r351;
	add.f32 	%r1394, %r1394, %r350;
	add.f32 	%r1395, %r1395, %r349;
	add.f32 	%r1396, %r1396, %r348;
	add.f32 	%r1397, %r1397, %r347;
	add.f32 	%r1398, %r1398, %r346;
	add.f32 	%r1399, %r1399, %r345;
	add.f32 	%r1400, %r1400, %r344;
	add.f32 	%r1401, %r1401, %r343;
	add.f32 	%r1402, %r1402, %r342;
	add.f32 	%r1403, %r1403, %r341;
	add.f32 	%r1404, %r1404, %r340;
	add.f32 	%r1405, %r1405, %r339;
	add.f32 	%r1406, %r1406, %r338;
	add.f32 	%r1407, %r1407, %r337;
	add.f32 	%r1408, %r1408, %r336;
	add.f32 	%r1409, %r1409, %r335;
	add.f32 	%r1410, %r1410, %r334;
	add.f32 	%r1411, %r1411, %r333;
	add.f32 	%r1412, %r1412, %r332;
	add.f32 	%r1413, %r1413, %r331;
	add.f32 	%r1414, %r1414, %r330;
	add.f32 	%r1415, %r1415, %r329;
	add.f32 	%r1416, %r1416, %r328;
	add.f32 	%r1417, %r1417, %r327;
	add.f32 	%r1418, %r1418, %r326;
	add.f32 	%r1419, %r1419, %r325;
	add.f32 	%r1420, %r1420, %r324;
	add.f32 	%r1421, %r1421, %r323;
	add.f32 	%r1422, %r1422, %r322;
	add.f32 	%r1423, %r1423, %r321;
	add.f32 	%r1424, %r1424, %r320;
	add.f32 	%r1425, %r1425, %r319;
	add.f32 	%r1426, %r1426, %r318;
	add.f32 	%r1427, %r1427, %r317;
	add.f32 	%r1428, %r1428, %r316;
	add.f32 	%r1429, %r1429, %r315;
	add.f32 	%r1430, %r1430, %r314;
	.loc	1 344 18                        // sk06_mlp_down.py:344:18
	add.s64 	%rd43, %rd243, %rd5;
	add.s64 	%rd44, %rd244, %rd5;
	add.s64 	%rd45, %rd245, %rd5;
	.loc	1 345 18                        // sk06_mlp_down.py:345:18
	add.s64 	%rd46, %rd246, %rd5;
	add.s64 	%rd47, %rd247, %rd5;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	add.s64 	%rd48, %rd248, %rd5;
	add.s32 	%r442, %r1299, 1;
	setp.gt.s32 	%p6, %r442, 2;
	selp.b32 	%r1299, 0, %r442, %p6;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	shl.b32 	%r443, %r1299, 14;
	bar.sync 	0;
	add.s32 	%r300, %r20, %r443;
	selp.b32 	%r301, 16, 0, %p4;
	// begin inline asm
	cp.async.cg.shared.global [ %r300 + 0 ], [ %rd43 + 0 ], 0x10, %r301;
	// end inline asm
	add.s32 	%r302, %r300, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r302 + 0 ], [ %rd44 + 0 ], 0x10, %r301;
	// end inline asm
	add.s32 	%r303, %r300, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r303 + 0 ], [ %rd45 + 0 ], 0x10, %r301;
	// end inline asm
	add.s32 	%r304, %r300, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r304 + 0 ], [ %rd46 + 0 ], 0x10, %r301;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	shl.b32 	%r444, %r1299, 13;
	add.s32 	%r445, %r20, %r444;
	add.s32 	%r305, %r445, 49152;
	// begin inline asm
	cp.async.cg.shared.global [ %r305 + 0 ], [ %rd47 + 0 ], 0x10, %r301;
	// end inline asm
	add.s32 	%r306, %r445, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r306 + 0 ], [ %rd48 + 0 ], 0x10, %r301;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	add.s32 	%r1300, %r1300, 1;
	add.s64 	%rd248, %rd248, 64;
	add.s64 	%rd247, %rd247, 64;
	add.s64 	%rd246, %rd246, 64;
	add.s64 	%rd245, %rd245, 64;
	add.s64 	%rd244, %rd244, 64;
	add.s64 	%rd243, %rd243, 64;
	setp.ne.b32 	%p7, %r8, %r1300;
	@%p7 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_4;
$L__BB0_1:                              // %.._crit_edge_crit_edge
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	and.b32 	%r1302, %r2, 16;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	shl.b32 	%r1301, %r2, 4;
	mov.b32 	%r1303, 0f00000000;
	mov.b32 	%r1304, %r1303;
	mov.b32 	%r1305, %r1303;
	mov.b32 	%r1306, %r1303;
	mov.b32 	%r1307, %r1303;
	mov.b32 	%r1308, %r1303;
	mov.b32 	%r1309, %r1303;
	mov.b32 	%r1310, %r1303;
	mov.b32 	%r1311, %r1303;
	mov.b32 	%r1312, %r1303;
	mov.b32 	%r1313, %r1303;
	mov.b32 	%r1314, %r1303;
	mov.b32 	%r1315, %r1303;
	mov.b32 	%r1316, %r1303;
	mov.b32 	%r1317, %r1303;
	mov.b32 	%r1318, %r1303;
	mov.b32 	%r1319, %r1303;
	mov.b32 	%r1320, %r1303;
	mov.b32 	%r1321, %r1303;
	mov.b32 	%r1322, %r1303;
	mov.b32 	%r1323, %r1303;
	mov.b32 	%r1324, %r1303;
	mov.b32 	%r1325, %r1303;
	mov.b32 	%r1326, %r1303;
	mov.b32 	%r1327, %r1303;
	mov.b32 	%r1328, %r1303;
	mov.b32 	%r1329, %r1303;
	mov.b32 	%r1330, %r1303;
	mov.b32 	%r1331, %r1303;
	mov.b32 	%r1332, %r1303;
	mov.b32 	%r1333, %r1303;
	mov.b32 	%r1334, %r1303;
	mov.b32 	%r1335, %r1303;
	mov.b32 	%r1336, %r1303;
	mov.b32 	%r1337, %r1303;
	mov.b32 	%r1338, %r1303;
	mov.b32 	%r1339, %r1303;
	mov.b32 	%r1340, %r1303;
	mov.b32 	%r1341, %r1303;
	mov.b32 	%r1342, %r1303;
	mov.b32 	%r1343, %r1303;
	mov.b32 	%r1344, %r1303;
	mov.b32 	%r1345, %r1303;
	mov.b32 	%r1346, %r1303;
	mov.b32 	%r1347, %r1303;
	mov.b32 	%r1348, %r1303;
	mov.b32 	%r1349, %r1303;
	mov.b32 	%r1350, %r1303;
	mov.b32 	%r1351, %r1303;
	mov.b32 	%r1352, %r1303;
	mov.b32 	%r1353, %r1303;
	mov.b32 	%r1354, %r1303;
	mov.b32 	%r1355, %r1303;
	mov.b32 	%r1356, %r1303;
	mov.b32 	%r1357, %r1303;
	mov.b32 	%r1358, %r1303;
	mov.b32 	%r1359, %r1303;
	mov.b32 	%r1360, %r1303;
	mov.b32 	%r1361, %r1303;
	mov.b32 	%r1362, %r1303;
	mov.b32 	%r1363, %r1303;
	mov.b32 	%r1364, %r1303;
	mov.b32 	%r1365, %r1303;
	mov.b32 	%r1366, %r1303;
	mov.b32 	%r1367, %r1303;
	mov.b32 	%r1368, %r1303;
	mov.b32 	%r1369, %r1303;
	mov.b32 	%r1370, %r1303;
	mov.b32 	%r1371, %r1303;
	mov.b32 	%r1372, %r1303;
	mov.b32 	%r1373, %r1303;
	mov.b32 	%r1374, %r1303;
	mov.b32 	%r1375, %r1303;
	mov.b32 	%r1376, %r1303;
	mov.b32 	%r1377, %r1303;
	mov.b32 	%r1378, %r1303;
	mov.b32 	%r1379, %r1303;
	mov.b32 	%r1380, %r1303;
	mov.b32 	%r1381, %r1303;
	mov.b32 	%r1382, %r1303;
	mov.b32 	%r1383, %r1303;
	mov.b32 	%r1384, %r1303;
	mov.b32 	%r1385, %r1303;
	mov.b32 	%r1386, %r1303;
	mov.b32 	%r1387, %r1303;
	mov.b32 	%r1388, %r1303;
	mov.b32 	%r1389, %r1303;
	mov.b32 	%r1390, %r1303;
	mov.b32 	%r1391, %r1303;
	mov.b32 	%r1392, %r1303;
	mov.b32 	%r1393, %r1303;
	mov.b32 	%r1394, %r1303;
	mov.b32 	%r1395, %r1303;
	mov.b32 	%r1396, %r1303;
	mov.b32 	%r1397, %r1303;
	mov.b32 	%r1398, %r1303;
	mov.b32 	%r1399, %r1303;
	mov.b32 	%r1400, %r1303;
	mov.b32 	%r1401, %r1303;
	mov.b32 	%r1402, %r1303;
	mov.b32 	%r1403, %r1303;
	mov.b32 	%r1404, %r1303;
	mov.b32 	%r1405, %r1303;
	mov.b32 	%r1406, %r1303;
	mov.b32 	%r1407, %r1303;
	mov.b32 	%r1408, %r1303;
	mov.b32 	%r1409, %r1303;
	mov.b32 	%r1410, %r1303;
	mov.b32 	%r1411, %r1303;
	mov.b32 	%r1412, %r1303;
	mov.b32 	%r1413, %r1303;
	mov.b32 	%r1414, %r1303;
	mov.b32 	%r1415, %r1303;
	mov.b32 	%r1416, %r1303;
	mov.b32 	%r1417, %r1303;
	mov.b32 	%r1418, %r1303;
	mov.b32 	%r1419, %r1303;
	mov.b32 	%r1420, %r1303;
	mov.b32 	%r1421, %r1303;
	mov.b32 	%r1422, %r1303;
	mov.b32 	%r1423, %r1303;
	mov.b32 	%r1424, %r1303;
	mov.b32 	%r1425, %r1303;
	mov.b32 	%r1426, %r1303;
	mov.b32 	%r1427, %r1303;
	mov.b32 	%r1428, %r1303;
	mov.b32 	%r1429, %r1303;
	mov.b32 	%r1430, %r1303;
$L__BB0_4:                              // %._crit_edge
	.loc	1 323 45                        // sk06_mlp_down.py:323:45
	shl.b32 	%r676, %r7, 3;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r677, %r676, %r4;
	or.b32 	%r678, %r677, 7;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r679, %r678, %r16;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r680, %r677, 6;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r681, %r680, %r16;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r682, %r677, 5;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r683, %r682, %r16;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r684, %r677, 4;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r685, %r684, %r16;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r686, %r677, 3;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r687, %r686, %r16;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r688, %r677, 2;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r689, %r688, %r16;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r690, %r677, 1;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r691, %r690, %r16;
	rem.s32 	%r692, %r677, %r16;
	.loc	1 323 45                        // sk06_mlp_down.py:323:45
	shr.u32 	%r693, %r6, 2;
	shl.b32 	%r694, %r5, 1;
	or.b32 	%r695, %r693, %r694;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r696, %r695, %r4;
	or.b32 	%r697, %r696, 112;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r698, %r697, %r16;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r699, %r696, 96;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r700, %r699, %r16;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r701, %r696, 80;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r702, %r701, %r16;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r703, %r696, 64;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r704, %r703, %r16;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r705, %r696, 48;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r706, %r705, %r16;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r707, %r696, 32;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r708, %r707, %r16;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r709, %r696, 16;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r710, %r709, %r16;
	rem.s32 	%r711, %r696, %r16;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r712, %r1, %r3;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r713, %r712, %r15;
	.loc	1 322 45                        // sk06_mlp_down.py:322:45
	and.b32 	%r714, %r2, 240;
	bfe.u32 	%r715, %r2, 4, 4;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r716, %r715, %r1;
	or.b32 	%r717, %r716, 240;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r718, %r717, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r719, %r716, 224;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r720, %r719, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r721, %r716, 208;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r722, %r721, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r723, %r716, 192;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r724, %r723, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r725, %r716, 176;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r726, %r725, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r727, %r716, 160;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r728, %r727, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r729, %r716, 144;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r730, %r729, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r731, %r716, 128;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r732, %r731, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r733, %r716, 112;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r734, %r733, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r735, %r716, 96;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r736, %r735, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r737, %r716, 80;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r738, %r737, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r739, %r716, 64;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r740, %r739, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r741, %r716, 48;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r742, %r741, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r743, %r716, 32;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r744, %r743, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r745, %r716, 16;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r746, %r745, %r15;
	rem.s32 	%r747, %r716, %r15;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 347 38                        // sk06_mlp_down.py:347:38
	mad.wide.s32 	%rd49, %r713, 4, %rd12;
	.loc	1 347 24                        // sk06_mlp_down.py:347:24
	// begin inline asm
	mov.u32 %r447, 0x0;
	ld.global.b32 { %r447 }, [ %rd49 + 0 ];
	// end inline asm
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	and.b32 	%r748, %r2, 7;
	shl.b32 	%r749, %r748, 3;
	shl.b32 	%r750, %r714, 2;
	and.b32 	%r751, %r2, 8;
	shr.u32 	%r752, %r751, 1;
	add.s32 	%r753, %r94, %r749;
	add.s32 	%r754, %r753, %r750;
	add.s32 	%r446, %r754, %r752;
	// begin inline asm
	st.shared.b32 [ %r446 + 0 ], %r447;
	// end inline asm
	bar.sync 	0;
	and.b32 	%r755, %r9, 56;
	and.b32 	%r756, %r2, 192;
	add.s32 	%r757, %r94, %r755;
	add.s32 	%r758, %r757, %r756;
	ld.shared.v2.b32 	{%r759, %r760}, [%r758];
	ld.shared.v2.b32 	{%r761, %r762}, [%r758+256];
	ld.shared.v2.b32 	{%r763, %r764}, [%r758+512];
	ld.shared.v2.b32 	{%r765, %r766}, [%r758+768];
	.loc	1 349 42                        // sk06_mlp_down.py:349:42
	mad.wide.s32 	%rd50, %r711, 4, %rd13;
	mad.wide.s32 	%rd51, %r710, 4, %rd13;
	mad.wide.s32 	%rd52, %r708, 4, %rd13;
	mad.wide.s32 	%rd53, %r706, 4, %rd13;
	mad.wide.s32 	%rd54, %r704, 4, %rd13;
	mad.wide.s32 	%rd55, %r702, 4, %rd13;
	mad.wide.s32 	%rd56, %r700, 4, %rd13;
	mad.wide.s32 	%rd57, %r698, 4, %rd13;
	.loc	1 349 28                        // sk06_mlp_down.py:349:28
	// begin inline asm
	mov.u32 %r448, 0x0;
	mov.u32 %r449, 0x0;
	ld.global.v2.b32 { %r448, %r449 }, [ %rd50 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r450, 0x0;
	mov.u32 %r451, 0x0;
	ld.global.v2.b32 { %r450, %r451 }, [ %rd51 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r452, 0x0;
	mov.u32 %r453, 0x0;
	ld.global.v2.b32 { %r452, %r453 }, [ %rd52 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r454, 0x0;
	mov.u32 %r455, 0x0;
	ld.global.v2.b32 { %r454, %r455 }, [ %rd53 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r456, 0x0;
	mov.u32 %r457, 0x0;
	ld.global.v2.b32 { %r456, %r457 }, [ %rd54 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r458, 0x0;
	mov.u32 %r459, 0x0;
	ld.global.v2.b32 { %r458, %r459 }, [ %rd55 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r460, 0x0;
	mov.u32 %r461, 0x0;
	ld.global.v2.b32 { %r460, %r461 }, [ %rd56 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r462, 0x0;
	mov.u32 %r463, 0x0;
	ld.global.v2.b32 { %r462, %r463 }, [ %rd57 + 0 ];
	// end inline asm
	.loc	1 350 49                        // sk06_mlp_down.py:350:49
	mul.lo.s32 	%r767, %r747, %r18;
	mul.lo.s32 	%r768, %r746, %r18;
	mul.lo.s32 	%r769, %r744, %r18;
	mul.lo.s32 	%r770, %r742, %r18;
	mul.lo.s32 	%r771, %r740, %r18;
	mul.lo.s32 	%r772, %r738, %r18;
	mul.lo.s32 	%r773, %r736, %r18;
	mul.lo.s32 	%r774, %r734, %r18;
	mul.lo.s32 	%r775, %r732, %r18;
	mul.lo.s32 	%r776, %r730, %r18;
	mul.lo.s32 	%r777, %r728, %r18;
	mul.lo.s32 	%r778, %r726, %r18;
	mul.lo.s32 	%r779, %r724, %r18;
	mul.lo.s32 	%r780, %r722, %r18;
	mul.lo.s32 	%r781, %r720, %r18;
	mul.lo.s32 	%r782, %r718, %r18;
	.loc	1 350 31                        // sk06_mlp_down.py:350:31
	mad.wide.s32 	%rd202, %r767, 2, %rd11;
	mad.wide.s32 	%rd203, %r768, 2, %rd11;
	mad.wide.s32 	%rd204, %r769, 2, %rd11;
	mad.wide.s32 	%rd205, %r770, 2, %rd11;
	mad.wide.s32 	%rd206, %r771, 2, %rd11;
	mad.wide.s32 	%rd207, %r772, 2, %rd11;
	mad.wide.s32 	%rd208, %r773, 2, %rd11;
	mad.wide.s32 	%rd209, %r774, 2, %rd11;
	mad.wide.s32 	%rd210, %r775, 2, %rd11;
	mad.wide.s32 	%rd211, %r776, 2, %rd11;
	mad.wide.s32 	%rd212, %r777, 2, %rd11;
	mad.wide.s32 	%rd213, %r778, 2, %rd11;
	mad.wide.s32 	%rd214, %r779, 2, %rd11;
	mad.wide.s32 	%rd215, %r780, 2, %rd11;
	mad.wide.s32 	%rd216, %r781, 2, %rd11;
	mad.wide.s32 	%rd217, %r782, 2, %rd11;
	.loc	1 350 82                        // sk06_mlp_down.py:350:82
	mul.lo.s32 	%r783, %r692, %r19;
	mul.lo.s32 	%r784, %r691, %r19;
	mul.lo.s32 	%r785, %r689, %r19;
	mul.lo.s32 	%r786, %r687, %r19;
	mul.lo.s32 	%r787, %r685, %r19;
	mul.lo.s32 	%r788, %r683, %r19;
	mul.lo.s32 	%r789, %r681, %r19;
	mul.lo.s32 	%r790, %r679, %r19;
	.loc	1 350 64                        // sk06_mlp_down.py:350:64
	mul.wide.s32 	%rd218, %r783, 2;
	add.s64 	%rd58, %rd202, %rd218;
	mul.wide.s32 	%rd219, %r784, 2;
	add.s64 	%rd59, %rd202, %rd219;
	mul.wide.s32 	%rd220, %r785, 2;
	add.s64 	%rd60, %rd202, %rd220;
	mul.wide.s32 	%rd221, %r786, 2;
	add.s64 	%rd61, %rd202, %rd221;
	mul.wide.s32 	%rd222, %r787, 2;
	add.s64 	%rd62, %rd202, %rd222;
	mul.wide.s32 	%rd223, %r788, 2;
	add.s64 	%rd63, %rd202, %rd223;
	mul.wide.s32 	%rd224, %r789, 2;
	add.s64 	%rd64, %rd202, %rd224;
	mul.wide.s32 	%rd225, %r790, 2;
	add.s64 	%rd65, %rd202, %rd225;
	add.s64 	%rd66, %rd203, %rd218;
	add.s64 	%rd67, %rd203, %rd219;
	add.s64 	%rd68, %rd203, %rd220;
	add.s64 	%rd69, %rd203, %rd221;
	add.s64 	%rd70, %rd203, %rd222;
	add.s64 	%rd71, %rd203, %rd223;
	add.s64 	%rd72, %rd203, %rd224;
	add.s64 	%rd73, %rd203, %rd225;
	add.s64 	%rd74, %rd204, %rd218;
	add.s64 	%rd75, %rd204, %rd219;
	add.s64 	%rd76, %rd204, %rd220;
	add.s64 	%rd77, %rd204, %rd221;
	add.s64 	%rd78, %rd204, %rd222;
	add.s64 	%rd79, %rd204, %rd223;
	add.s64 	%rd80, %rd204, %rd224;
	add.s64 	%rd81, %rd204, %rd225;
	add.s64 	%rd82, %rd205, %rd218;
	add.s64 	%rd83, %rd205, %rd219;
	add.s64 	%rd84, %rd205, %rd220;
	add.s64 	%rd85, %rd205, %rd221;
	add.s64 	%rd86, %rd205, %rd222;
	add.s64 	%rd87, %rd205, %rd223;
	add.s64 	%rd88, %rd205, %rd224;
	add.s64 	%rd89, %rd205, %rd225;
	add.s64 	%rd90, %rd206, %rd218;
	add.s64 	%rd91, %rd206, %rd219;
	add.s64 	%rd92, %rd206, %rd220;
	add.s64 	%rd93, %rd206, %rd221;
	add.s64 	%rd94, %rd206, %rd222;
	add.s64 	%rd95, %rd206, %rd223;
	add.s64 	%rd96, %rd206, %rd224;
	add.s64 	%rd97, %rd206, %rd225;
	add.s64 	%rd98, %rd207, %rd218;
	add.s64 	%rd99, %rd207, %rd219;
	add.s64 	%rd100, %rd207, %rd220;
	add.s64 	%rd101, %rd207, %rd221;
	add.s64 	%rd102, %rd207, %rd222;
	add.s64 	%rd103, %rd207, %rd223;
	add.s64 	%rd104, %rd207, %rd224;
	add.s64 	%rd105, %rd207, %rd225;
	add.s64 	%rd106, %rd208, %rd218;
	add.s64 	%rd107, %rd208, %rd219;
	add.s64 	%rd108, %rd208, %rd220;
	add.s64 	%rd109, %rd208, %rd221;
	add.s64 	%rd110, %rd208, %rd222;
	add.s64 	%rd111, %rd208, %rd223;
	add.s64 	%rd112, %rd208, %rd224;
	add.s64 	%rd113, %rd208, %rd225;
	add.s64 	%rd114, %rd209, %rd218;
	add.s64 	%rd115, %rd209, %rd219;
	add.s64 	%rd116, %rd209, %rd220;
	add.s64 	%rd117, %rd209, %rd221;
	add.s64 	%rd118, %rd209, %rd222;
	add.s64 	%rd119, %rd209, %rd223;
	add.s64 	%rd120, %rd209, %rd224;
	add.s64 	%rd121, %rd209, %rd225;
	add.s64 	%rd122, %rd210, %rd218;
	add.s64 	%rd123, %rd210, %rd219;
	add.s64 	%rd124, %rd210, %rd220;
	add.s64 	%rd125, %rd210, %rd221;
	add.s64 	%rd126, %rd210, %rd222;
	add.s64 	%rd127, %rd210, %rd223;
	add.s64 	%rd128, %rd210, %rd224;
	add.s64 	%rd129, %rd210, %rd225;
	add.s64 	%rd130, %rd211, %rd218;
	add.s64 	%rd131, %rd211, %rd219;
	add.s64 	%rd132, %rd211, %rd220;
	add.s64 	%rd133, %rd211, %rd221;
	add.s64 	%rd134, %rd211, %rd222;
	add.s64 	%rd135, %rd211, %rd223;
	add.s64 	%rd136, %rd211, %rd224;
	add.s64 	%rd137, %rd211, %rd225;
	add.s64 	%rd138, %rd212, %rd218;
	add.s64 	%rd139, %rd212, %rd219;
	add.s64 	%rd140, %rd212, %rd220;
	add.s64 	%rd141, %rd212, %rd221;
	add.s64 	%rd142, %rd212, %rd222;
	add.s64 	%rd143, %rd212, %rd223;
	add.s64 	%rd144, %rd212, %rd224;
	add.s64 	%rd145, %rd212, %rd225;
	add.s64 	%rd146, %rd213, %rd218;
	add.s64 	%rd147, %rd213, %rd219;
	add.s64 	%rd148, %rd213, %rd220;
	add.s64 	%rd149, %rd213, %rd221;
	add.s64 	%rd150, %rd213, %rd222;
	add.s64 	%rd151, %rd213, %rd223;
	add.s64 	%rd152, %rd213, %rd224;
	add.s64 	%rd153, %rd213, %rd225;
	add.s64 	%rd154, %rd214, %rd218;
	add.s64 	%rd155, %rd214, %rd219;
	add.s64 	%rd156, %rd214, %rd220;
	add.s64 	%rd157, %rd214, %rd221;
	add.s64 	%rd158, %rd214, %rd222;
	add.s64 	%rd159, %rd214, %rd223;
	add.s64 	%rd160, %rd214, %rd224;
	add.s64 	%rd161, %rd214, %rd225;
	add.s64 	%rd162, %rd215, %rd218;
	add.s64 	%rd163, %rd215, %rd219;
	add.s64 	%rd164, %rd215, %rd220;
	add.s64 	%rd165, %rd215, %rd221;
	add.s64 	%rd166, %rd215, %rd222;
	add.s64 	%rd167, %rd215, %rd223;
	add.s64 	%rd168, %rd215, %rd224;
	add.s64 	%rd169, %rd215, %rd225;
	add.s64 	%rd170, %rd216, %rd218;
	add.s64 	%rd171, %rd216, %rd219;
	add.s64 	%rd172, %rd216, %rd220;
	add.s64 	%rd173, %rd216, %rd221;
	add.s64 	%rd174, %rd216, %rd222;
	add.s64 	%rd175, %rd216, %rd223;
	add.s64 	%rd176, %rd216, %rd224;
	add.s64 	%rd177, %rd216, %rd225;
	add.s64 	%rd178, %rd217, %rd218;
	add.s64 	%rd179, %rd217, %rd219;
	add.s64 	%rd180, %rd217, %rd220;
	add.s64 	%rd181, %rd217, %rd221;
	add.s64 	%rd182, %rd217, %rd222;
	add.s64 	%rd183, %rd217, %rd223;
	add.s64 	%rd184, %rd217, %rd224;
	add.s64 	%rd185, %rd217, %rd225;
	.loc	1 350 19                        // sk06_mlp_down.py:350:19
	// begin inline asm
	mov.u16 %rs1, 0x0;
	ld.global.b16 { %rs1 }, [ %rd58 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs2, 0x0;
	ld.global.b16 { %rs2 }, [ %rd59 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs3, 0x0;
	ld.global.b16 { %rs3 }, [ %rd60 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs4, 0x0;
	ld.global.b16 { %rs4 }, [ %rd61 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs5, 0x0;
	ld.global.b16 { %rs5 }, [ %rd62 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs6, 0x0;
	ld.global.b16 { %rs6 }, [ %rd63 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs7, 0x0;
	ld.global.b16 { %rs7 }, [ %rd64 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs8, 0x0;
	ld.global.b16 { %rs8 }, [ %rd65 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs9, 0x0;
	ld.global.b16 { %rs9 }, [ %rd66 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs10, 0x0;
	ld.global.b16 { %rs10 }, [ %rd67 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs11, 0x0;
	ld.global.b16 { %rs11 }, [ %rd68 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs12, 0x0;
	ld.global.b16 { %rs12 }, [ %rd69 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs13, 0x0;
	ld.global.b16 { %rs13 }, [ %rd70 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs14, 0x0;
	ld.global.b16 { %rs14 }, [ %rd71 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs15, 0x0;
	ld.global.b16 { %rs15 }, [ %rd72 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs16, 0x0;
	ld.global.b16 { %rs16 }, [ %rd73 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs17, 0x0;
	ld.global.b16 { %rs17 }, [ %rd74 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs18, 0x0;
	ld.global.b16 { %rs18 }, [ %rd75 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs19, 0x0;
	ld.global.b16 { %rs19 }, [ %rd76 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs20, 0x0;
	ld.global.b16 { %rs20 }, [ %rd77 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs21, 0x0;
	ld.global.b16 { %rs21 }, [ %rd78 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs22, 0x0;
	ld.global.b16 { %rs22 }, [ %rd79 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs23, 0x0;
	ld.global.b16 { %rs23 }, [ %rd80 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs24, 0x0;
	ld.global.b16 { %rs24 }, [ %rd81 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs25, 0x0;
	ld.global.b16 { %rs25 }, [ %rd82 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs26, 0x0;
	ld.global.b16 { %rs26 }, [ %rd83 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs27, 0x0;
	ld.global.b16 { %rs27 }, [ %rd84 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs28, 0x0;
	ld.global.b16 { %rs28 }, [ %rd85 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs29, 0x0;
	ld.global.b16 { %rs29 }, [ %rd86 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs30, 0x0;
	ld.global.b16 { %rs30 }, [ %rd87 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs31, 0x0;
	ld.global.b16 { %rs31 }, [ %rd88 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs32, 0x0;
	ld.global.b16 { %rs32 }, [ %rd89 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs33, 0x0;
	ld.global.b16 { %rs33 }, [ %rd90 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs34, 0x0;
	ld.global.b16 { %rs34 }, [ %rd91 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs35, 0x0;
	ld.global.b16 { %rs35 }, [ %rd92 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs36, 0x0;
	ld.global.b16 { %rs36 }, [ %rd93 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs37, 0x0;
	ld.global.b16 { %rs37 }, [ %rd94 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs38, 0x0;
	ld.global.b16 { %rs38 }, [ %rd95 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs39, 0x0;
	ld.global.b16 { %rs39 }, [ %rd96 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs40, 0x0;
	ld.global.b16 { %rs40 }, [ %rd97 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs41, 0x0;
	ld.global.b16 { %rs41 }, [ %rd98 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs42, 0x0;
	ld.global.b16 { %rs42 }, [ %rd99 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs43, 0x0;
	ld.global.b16 { %rs43 }, [ %rd100 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs44, 0x0;
	ld.global.b16 { %rs44 }, [ %rd101 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs45, 0x0;
	ld.global.b16 { %rs45 }, [ %rd102 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs46, 0x0;
	ld.global.b16 { %rs46 }, [ %rd103 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs47, 0x0;
	ld.global.b16 { %rs47 }, [ %rd104 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs48, 0x0;
	ld.global.b16 { %rs48 }, [ %rd105 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs49, 0x0;
	ld.global.b16 { %rs49 }, [ %rd106 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs50, 0x0;
	ld.global.b16 { %rs50 }, [ %rd107 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs51, 0x0;
	ld.global.b16 { %rs51 }, [ %rd108 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs52, 0x0;
	ld.global.b16 { %rs52 }, [ %rd109 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs53, 0x0;
	ld.global.b16 { %rs53 }, [ %rd110 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs54, 0x0;
	ld.global.b16 { %rs54 }, [ %rd111 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs55, 0x0;
	ld.global.b16 { %rs55 }, [ %rd112 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs56, 0x0;
	ld.global.b16 { %rs56 }, [ %rd113 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs57, 0x0;
	ld.global.b16 { %rs57 }, [ %rd114 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs58, 0x0;
	ld.global.b16 { %rs58 }, [ %rd115 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs59, 0x0;
	ld.global.b16 { %rs59 }, [ %rd116 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs60, 0x0;
	ld.global.b16 { %rs60 }, [ %rd117 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs61, 0x0;
	ld.global.b16 { %rs61 }, [ %rd118 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs62, 0x0;
	ld.global.b16 { %rs62 }, [ %rd119 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs63, 0x0;
	ld.global.b16 { %rs63 }, [ %rd120 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs64, 0x0;
	ld.global.b16 { %rs64 }, [ %rd121 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs65, 0x0;
	ld.global.b16 { %rs65 }, [ %rd122 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs66, 0x0;
	ld.global.b16 { %rs66 }, [ %rd123 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs67, 0x0;
	ld.global.b16 { %rs67 }, [ %rd124 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs68, 0x0;
	ld.global.b16 { %rs68 }, [ %rd125 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs69, 0x0;
	ld.global.b16 { %rs69 }, [ %rd126 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs70, 0x0;
	ld.global.b16 { %rs70 }, [ %rd127 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs71, 0x0;
	ld.global.b16 { %rs71 }, [ %rd128 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs72, 0x0;
	ld.global.b16 { %rs72 }, [ %rd129 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs73, 0x0;
	ld.global.b16 { %rs73 }, [ %rd130 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs74, 0x0;
	ld.global.b16 { %rs74 }, [ %rd131 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs75, 0x0;
	ld.global.b16 { %rs75 }, [ %rd132 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs76, 0x0;
	ld.global.b16 { %rs76 }, [ %rd133 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs77, 0x0;
	ld.global.b16 { %rs77 }, [ %rd134 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs78, 0x0;
	ld.global.b16 { %rs78 }, [ %rd135 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs79, 0x0;
	ld.global.b16 { %rs79 }, [ %rd136 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs80, 0x0;
	ld.global.b16 { %rs80 }, [ %rd137 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs81, 0x0;
	ld.global.b16 { %rs81 }, [ %rd138 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs82, 0x0;
	ld.global.b16 { %rs82 }, [ %rd139 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs83, 0x0;
	ld.global.b16 { %rs83 }, [ %rd140 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs84, 0x0;
	ld.global.b16 { %rs84 }, [ %rd141 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs85, 0x0;
	ld.global.b16 { %rs85 }, [ %rd142 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs86, 0x0;
	ld.global.b16 { %rs86 }, [ %rd143 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs87, 0x0;
	ld.global.b16 { %rs87 }, [ %rd144 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs88, 0x0;
	ld.global.b16 { %rs88 }, [ %rd145 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs89, 0x0;
	ld.global.b16 { %rs89 }, [ %rd146 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs90, 0x0;
	ld.global.b16 { %rs90 }, [ %rd147 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs91, 0x0;
	ld.global.b16 { %rs91 }, [ %rd148 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs92, 0x0;
	ld.global.b16 { %rs92 }, [ %rd149 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs93, 0x0;
	ld.global.b16 { %rs93 }, [ %rd150 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs94, 0x0;
	ld.global.b16 { %rs94 }, [ %rd151 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs95, 0x0;
	ld.global.b16 { %rs95 }, [ %rd152 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs96, 0x0;
	ld.global.b16 { %rs96 }, [ %rd153 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs97, 0x0;
	ld.global.b16 { %rs97 }, [ %rd154 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs98, 0x0;
	ld.global.b16 { %rs98 }, [ %rd155 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs99, 0x0;
	ld.global.b16 { %rs99 }, [ %rd156 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs100, 0x0;
	ld.global.b16 { %rs100 }, [ %rd157 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs101, 0x0;
	ld.global.b16 { %rs101 }, [ %rd158 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs102, 0x0;
	ld.global.b16 { %rs102 }, [ %rd159 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs103, 0x0;
	ld.global.b16 { %rs103 }, [ %rd160 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs104, 0x0;
	ld.global.b16 { %rs104 }, [ %rd161 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs105, 0x0;
	ld.global.b16 { %rs105 }, [ %rd162 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs106, 0x0;
	ld.global.b16 { %rs106 }, [ %rd163 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs107, 0x0;
	ld.global.b16 { %rs107 }, [ %rd164 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs108, 0x0;
	ld.global.b16 { %rs108 }, [ %rd165 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs109, 0x0;
	ld.global.b16 { %rs109 }, [ %rd166 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs110, 0x0;
	ld.global.b16 { %rs110 }, [ %rd167 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs111, 0x0;
	ld.global.b16 { %rs111 }, [ %rd168 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs112, 0x0;
	ld.global.b16 { %rs112 }, [ %rd169 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs113, 0x0;
	ld.global.b16 { %rs113 }, [ %rd170 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs114, 0x0;
	ld.global.b16 { %rs114 }, [ %rd171 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs115, 0x0;
	ld.global.b16 { %rs115 }, [ %rd172 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs116, 0x0;
	ld.global.b16 { %rs116 }, [ %rd173 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs117, 0x0;
	ld.global.b16 { %rs117 }, [ %rd174 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs118, 0x0;
	ld.global.b16 { %rs118 }, [ %rd175 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs119, 0x0;
	ld.global.b16 { %rs119 }, [ %rd176 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs120, 0x0;
	ld.global.b16 { %rs120 }, [ %rd177 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs121, 0x0;
	ld.global.b16 { %rs121 }, [ %rd178 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs122, 0x0;
	ld.global.b16 { %rs122 }, [ %rd179 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs123, 0x0;
	ld.global.b16 { %rs123 }, [ %rd180 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs124, 0x0;
	ld.global.b16 { %rs124 }, [ %rd181 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs125, 0x0;
	ld.global.b16 { %rs125 }, [ %rd182 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs126, 0x0;
	ld.global.b16 { %rs126 }, [ %rd183 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs127, 0x0;
	ld.global.b16 { %rs127 }, [ %rd184 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs128, 0x0;
	ld.global.b16 { %rs128 }, [ %rd185 + 0 ];
	// end inline asm
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	bar.sync 	0;
	shl.b32 	%r791, %r2, 7;
	and.b32 	%r792, %r791, 15360;
	shl.b32 	%r793, %r748, 4;
	or.b32 	%r794, %r792, %r793;
	xor.b32 	%r795, %r794, %r714;
	add.s32 	%r464, %r94, %r795;
	mov.b32 	%r465, {%rs1, %rs2};
	mov.b32 	%r466, {%rs3, %rs4};
	mov.b32 	%r467, {%rs5, %rs6};
	mov.b32 	%r468, {%rs7, %rs8};
	// begin inline asm
	st.shared.v4.b32 [ %r464 + 0 ], { %r465, %r466, %r467, %r468 };
	// end inline asm
	add.s32 	%r469, %r464, 256;
	mov.b32 	%r470, {%rs9, %rs10};
	mov.b32 	%r471, {%rs11, %rs12};
	mov.b32 	%r472, {%rs13, %rs14};
	mov.b32 	%r473, {%rs15, %rs16};
	// begin inline asm
	st.shared.v4.b32 [ %r469 + 0 ], { %r470, %r471, %r472, %r473 };
	// end inline asm
	add.s32 	%r474, %r464, 512;
	mov.b32 	%r475, {%rs17, %rs18};
	mov.b32 	%r476, {%rs19, %rs20};
	mov.b32 	%r477, {%rs21, %rs22};
	mov.b32 	%r478, {%rs23, %rs24};
	// begin inline asm
	st.shared.v4.b32 [ %r474 + 0 ], { %r475, %r476, %r477, %r478 };
	// end inline asm
	add.s32 	%r479, %r464, 768;
	mov.b32 	%r480, {%rs25, %rs26};
	mov.b32 	%r481, {%rs27, %rs28};
	mov.b32 	%r482, {%rs29, %rs30};
	mov.b32 	%r483, {%rs31, %rs32};
	// begin inline asm
	st.shared.v4.b32 [ %r479 + 0 ], { %r480, %r481, %r482, %r483 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r796, %r748, 11;
	shl.b32 	%r797, %r7, 4;
	shl.b32 	%r798, %r756, 2;
	setp.eq.b32 	%p24, %r1302, 0;
	shl.b32 	%r799, %r1302, 1;
	shr.u32 	%r800, %r6, 1;
	or.b32 	%r801, %r797, %r798;
	or.b32 	%r802, %r799, %r800;
	xor.b32 	%r803, %r801, %r802;
	or.b32 	%r804, %r803, %r796;
	add.s32 	%r805, %r94, %r804;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r806, %r807, %r808, %r809}, [%r805];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r810, %r811, %r812, %r813}, [%r805+1024];
	xor.b32 	%r814, %r804, 64;
	add.s32 	%r815, %r94, %r814;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r816, %r817, %r818, %r819}, [%r815];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r820, %r821, %r822, %r823}, [%r815+1024];
	bar.sync 	0;
	mov.b32 	%r484, {%rs33, %rs34};
	mov.b32 	%r485, {%rs35, %rs36};
	mov.b32 	%r486, {%rs37, %rs38};
	mov.b32 	%r487, {%rs39, %rs40};
	// begin inline asm
	st.shared.v4.b32 [ %r464 + 0 ], { %r484, %r485, %r486, %r487 };
	// end inline asm
	mov.b32 	%r488, {%rs41, %rs42};
	mov.b32 	%r489, {%rs43, %rs44};
	mov.b32 	%r490, {%rs45, %rs46};
	mov.b32 	%r491, {%rs47, %rs48};
	// begin inline asm
	st.shared.v4.b32 [ %r469 + 0 ], { %r488, %r489, %r490, %r491 };
	// end inline asm
	mov.b32 	%r492, {%rs49, %rs50};
	mov.b32 	%r493, {%rs51, %rs52};
	mov.b32 	%r494, {%rs53, %rs54};
	mov.b32 	%r495, {%rs55, %rs56};
	// begin inline asm
	st.shared.v4.b32 [ %r474 + 0 ], { %r492, %r493, %r494, %r495 };
	// end inline asm
	mov.b32 	%r496, {%rs57, %rs58};
	mov.b32 	%r497, {%rs59, %rs60};
	mov.b32 	%r498, {%rs61, %rs62};
	mov.b32 	%r499, {%rs63, %rs64};
	// begin inline asm
	st.shared.v4.b32 [ %r479 + 0 ], { %r496, %r497, %r498, %r499 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r824, %r825, %r826, %r827}, [%r805];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r828, %r829, %r830, %r831}, [%r805+1024];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r832, %r833, %r834, %r835}, [%r815];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r836, %r837, %r838, %r839}, [%r815+1024];
	bar.sync 	0;
	mov.b32 	%r500, {%rs65, %rs66};
	mov.b32 	%r501, {%rs67, %rs68};
	mov.b32 	%r502, {%rs69, %rs70};
	mov.b32 	%r503, {%rs71, %rs72};
	// begin inline asm
	st.shared.v4.b32 [ %r464 + 0 ], { %r500, %r501, %r502, %r503 };
	// end inline asm
	mov.b32 	%r504, {%rs73, %rs74};
	mov.b32 	%r505, {%rs75, %rs76};
	mov.b32 	%r506, {%rs77, %rs78};
	mov.b32 	%r507, {%rs79, %rs80};
	// begin inline asm
	st.shared.v4.b32 [ %r469 + 0 ], { %r504, %r505, %r506, %r507 };
	// end inline asm
	mov.b32 	%r508, {%rs81, %rs82};
	mov.b32 	%r509, {%rs83, %rs84};
	mov.b32 	%r510, {%rs85, %rs86};
	mov.b32 	%r511, {%rs87, %rs88};
	// begin inline asm
	st.shared.v4.b32 [ %r474 + 0 ], { %r508, %r509, %r510, %r511 };
	// end inline asm
	mov.b32 	%r512, {%rs89, %rs90};
	mov.b32 	%r513, {%rs91, %rs92};
	mov.b32 	%r514, {%rs93, %rs94};
	mov.b32 	%r515, {%rs95, %rs96};
	// begin inline asm
	st.shared.v4.b32 [ %r479 + 0 ], { %r512, %r513, %r514, %r515 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r840, %r841, %r842, %r843}, [%r805];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r844, %r845, %r846, %r847}, [%r805+1024];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r848, %r849, %r850, %r851}, [%r815];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r852, %r853, %r854, %r855}, [%r815+1024];
	bar.sync 	0;
	mov.b32 	%r516, {%rs97, %rs98};
	mov.b32 	%r517, {%rs99, %rs100};
	mov.b32 	%r518, {%rs101, %rs102};
	mov.b32 	%r519, {%rs103, %rs104};
	// begin inline asm
	st.shared.v4.b32 [ %r464 + 0 ], { %r516, %r517, %r518, %r519 };
	// end inline asm
	mov.b32 	%r520, {%rs105, %rs106};
	mov.b32 	%r521, {%rs107, %rs108};
	mov.b32 	%r522, {%rs109, %rs110};
	mov.b32 	%r523, {%rs111, %rs112};
	// begin inline asm
	st.shared.v4.b32 [ %r469 + 0 ], { %r520, %r521, %r522, %r523 };
	// end inline asm
	mov.b32 	%r524, {%rs113, %rs114};
	mov.b32 	%r525, {%rs115, %rs116};
	mov.b32 	%r526, {%rs117, %rs118};
	mov.b32 	%r527, {%rs119, %rs120};
	// begin inline asm
	st.shared.v4.b32 [ %r474 + 0 ], { %r524, %r525, %r526, %r527 };
	// end inline asm
	mov.b32 	%r528, {%rs121, %rs122};
	mov.b32 	%r529, {%rs123, %rs124};
	mov.b32 	%r530, {%rs125, %rs126};
	mov.b32 	%r531, {%rs127, %rs128};
	// begin inline asm
	st.shared.v4.b32 [ %r479 + 0 ], { %r528, %r529, %r530, %r531 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r856, %r857, %r858, %r859}, [%r805];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r860, %r861, %r862, %r863}, [%r805+1024];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r864, %r865, %r866, %r867}, [%r815];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r868, %r869, %r870, %r871}, [%r815+1024];
	.loc	1 357 31                        // sk06_mlp_down.py:357:31
	setp.lt.s32 	%p25, %r716, %r15;
	setp.lt.s32 	%p26, %r745, %r15;
	setp.lt.s32 	%p27, %r743, %r15;
	setp.lt.s32 	%p28, %r741, %r15;
	setp.lt.s32 	%p29, %r739, %r15;
	setp.lt.s32 	%p30, %r737, %r15;
	setp.lt.s32 	%p31, %r735, %r15;
	setp.lt.s32 	%p32, %r733, %r15;
	setp.lt.s32 	%p33, %r731, %r15;
	setp.lt.s32 	%p34, %r729, %r15;
	setp.lt.s32 	%p35, %r727, %r15;
	setp.lt.s32 	%p36, %r725, %r15;
	setp.lt.s32 	%p37, %r723, %r15;
	setp.lt.s32 	%p38, %r721, %r15;
	setp.lt.s32 	%p39, %r719, %r15;
	setp.lt.s32 	%p40, %r717, %r15;
	.loc	1 357 54                        // sk06_mlp_down.py:357:54
	setp.lt.s32 	%p41, %r677, %r16;
	.loc	1 357 37                        // sk06_mlp_down.py:357:37
	and.pred 	%p8, %p25, %p41;
	and.pred 	%p9, %p26, %p41;
	and.pred 	%p10, %p27, %p41;
	and.pred 	%p11, %p28, %p41;
	and.pred 	%p12, %p29, %p41;
	and.pred 	%p13, %p30, %p41;
	and.pred 	%p14, %p31, %p41;
	and.pred 	%p15, %p32, %p41;
	and.pred 	%p16, %p33, %p41;
	and.pred 	%p17, %p34, %p41;
	and.pred 	%p18, %p35, %p41;
	and.pred 	%p19, %p36, %p41;
	and.pred 	%p20, %p37, %p41;
	and.pred 	%p21, %p38, %p41;
	and.pred 	%p22, %p39, %p41;
	and.pred 	%p23, %p40, %p41;
	.loc	1 355 35                        // sk06_mlp_down.py:355:35
	mul.lo.s32 	%r872, %r716, %r17;
	mul.lo.s32 	%r873, %r745, %r17;
	mul.lo.s32 	%r874, %r743, %r17;
	mul.lo.s32 	%r875, %r741, %r17;
	mul.lo.s32 	%r876, %r739, %r17;
	mul.lo.s32 	%r877, %r737, %r17;
	mul.lo.s32 	%r878, %r735, %r17;
	mul.lo.s32 	%r879, %r733, %r17;
	mul.lo.s32 	%r880, %r731, %r17;
	mul.lo.s32 	%r881, %r729, %r17;
	mul.lo.s32 	%r882, %r727, %r17;
	mul.lo.s32 	%r883, %r725, %r17;
	mul.lo.s32 	%r884, %r723, %r17;
	mul.lo.s32 	%r885, %r721, %r17;
	mul.lo.s32 	%r886, %r719, %r17;
	mul.lo.s32 	%r887, %r717, %r17;
	.loc	1 355 18                        // sk06_mlp_down.py:355:18
	mad.wide.s32 	%rd226, %r872, 2, %rd10;
	mad.wide.s32 	%rd227, %r873, 2, %rd10;
	mad.wide.s32 	%rd228, %r874, 2, %rd10;
	mad.wide.s32 	%rd229, %r875, 2, %rd10;
	mad.wide.s32 	%rd230, %r876, 2, %rd10;
	mad.wide.s32 	%rd231, %r877, 2, %rd10;
	mad.wide.s32 	%rd232, %r878, 2, %rd10;
	mad.wide.s32 	%rd233, %r879, 2, %rd10;
	mad.wide.s32 	%rd234, %r880, 2, %rd10;
	mad.wide.s32 	%rd235, %r881, 2, %rd10;
	mad.wide.s32 	%rd236, %r882, 2, %rd10;
	mad.wide.s32 	%rd237, %r883, 2, %rd10;
	mad.wide.s32 	%rd238, %r884, 2, %rd10;
	mad.wide.s32 	%rd239, %r885, 2, %rd10;
	mad.wide.s32 	%rd240, %r886, 2, %rd10;
	mad.wide.s32 	%rd241, %r887, 2, %rd10;
	.loc	1 355 50                        // sk06_mlp_down.py:355:50
	mul.wide.s32 	%rd242, %r677, 2;
	add.s64 	%rd186, %rd226, %rd242;
	add.s64 	%rd187, %rd227, %rd242;
	add.s64 	%rd188, %rd228, %rd242;
	add.s64 	%rd189, %rd229, %rd242;
	add.s64 	%rd190, %rd230, %rd242;
	add.s64 	%rd191, %rd231, %rd242;
	add.s64 	%rd192, %rd232, %rd242;
	add.s64 	%rd193, %rd233, %rd242;
	add.s64 	%rd194, %rd234, %rd242;
	add.s64 	%rd195, %rd235, %rd242;
	add.s64 	%rd196, %rd236, %rd242;
	add.s64 	%rd197, %rd237, %rd242;
	add.s64 	%rd198, %rd238, %rd242;
	add.s64 	%rd199, %rd239, %rd242;
	add.s64 	%rd200, %rd240, %rd242;
	add.s64 	%rd201, %rd241, %rd242;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r888, %r1400, %r765;
	mul.f32 	%r889, %r1399, %r765;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs129, %rs130}, %r856;
	cvt.f32.bf16 	%r890, %rs130;
	cvt.f32.bf16 	%r891, %rs129;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r892, %r889, %r448, %r891;
	fma.rn.f32 	%r893, %r888, %r449, %r890;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r894, %r1304, %r759;
	mul.f32 	%r895, %r1303, %r759;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs131, %rs132}, %r806;
	cvt.f32.bf16 	%r896, %rs132;
	cvt.f32.bf16 	%r897, %rs131;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r898, %r895, %r448, %r897;
	fma.rn.f32 	%r899, %r894, %r449, %r896;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r533, %r899, %r898;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r900, %r1306, %r760;
	mul.f32 	%r901, %r1305, %r760;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs133, %rs134}, %r807;
	cvt.f32.bf16 	%r902, %rs134;
	cvt.f32.bf16 	%r903, %rs133;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r904, %r901, %r448, %r903;
	fma.rn.f32 	%r905, %r900, %r449, %r902;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r538, %r905, %r904;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r906, %r1336, %r761;
	mul.f32 	%r907, %r1335, %r761;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs135, %rs136}, %r824;
	cvt.f32.bf16 	%r908, %rs136;
	cvt.f32.bf16 	%r909, %rs135;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r910, %r907, %r448, %r909;
	fma.rn.f32 	%r911, %r906, %r449, %r908;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r534, %r911, %r910;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r912, %r1338, %r762;
	mul.f32 	%r913, %r1337, %r762;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs137, %rs138}, %r825;
	cvt.f32.bf16 	%r914, %rs138;
	cvt.f32.bf16 	%r915, %rs137;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r916, %r913, %r448, %r915;
	fma.rn.f32 	%r917, %r912, %r449, %r914;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r539, %r917, %r916;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r918, %r1368, %r763;
	mul.f32 	%r919, %r1367, %r763;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs139, %rs140}, %r840;
	cvt.f32.bf16 	%r920, %rs140;
	cvt.f32.bf16 	%r921, %rs139;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r922, %r919, %r448, %r921;
	fma.rn.f32 	%r923, %r918, %r449, %r920;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r535, %r923, %r922;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r924, %r1370, %r764;
	mul.f32 	%r925, %r1369, %r764;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs141, %rs142}, %r841;
	cvt.f32.bf16 	%r926, %rs142;
	cvt.f32.bf16 	%r927, %rs141;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r928, %r925, %r448, %r927;
	fma.rn.f32 	%r929, %r924, %r449, %r926;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r540, %r929, %r928;
	cvt.rn.bf16x2.f32 	%r536, %r893, %r892;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r930, %r1402, %r766;
	mul.f32 	%r931, %r1401, %r766;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs143, %rs144}, %r857;
	cvt.f32.bf16 	%r932, %rs144;
	cvt.f32.bf16 	%r933, %rs143;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r934, %r931, %r448, %r933;
	fma.rn.f32 	%r935, %r930, %r449, %r932;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r541, %r935, %r934;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r936, %r1404, %r765;
	mul.f32 	%r937, %r1403, %r765;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs145, %rs146}, %r858;
	cvt.f32.bf16 	%r938, %rs146;
	cvt.f32.bf16 	%r939, %rs145;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r940, %r937, %r450, %r939;
	fma.rn.f32 	%r941, %r936, %r451, %r938;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r942, %r1308, %r759;
	mul.f32 	%r943, %r1307, %r759;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs147, %rs148}, %r808;
	cvt.f32.bf16 	%r944, %rs148;
	cvt.f32.bf16 	%r945, %rs147;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r946, %r943, %r450, %r945;
	fma.rn.f32 	%r947, %r942, %r451, %r944;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r553, %r947, %r946;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r948, %r1310, %r760;
	mul.f32 	%r949, %r1309, %r760;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs149, %rs150}, %r809;
	cvt.f32.bf16 	%r950, %rs150;
	cvt.f32.bf16 	%r951, %rs149;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r952, %r949, %r450, %r951;
	fma.rn.f32 	%r953, %r948, %r451, %r950;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r558, %r953, %r952;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r954, %r1340, %r761;
	mul.f32 	%r955, %r1339, %r761;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs151, %rs152}, %r826;
	cvt.f32.bf16 	%r956, %rs152;
	cvt.f32.bf16 	%r957, %rs151;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r958, %r955, %r450, %r957;
	fma.rn.f32 	%r959, %r954, %r451, %r956;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r554, %r959, %r958;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r960, %r1342, %r762;
	mul.f32 	%r961, %r1341, %r762;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs153, %rs154}, %r827;
	cvt.f32.bf16 	%r962, %rs154;
	cvt.f32.bf16 	%r963, %rs153;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r964, %r961, %r450, %r963;
	fma.rn.f32 	%r965, %r960, %r451, %r962;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r559, %r965, %r964;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r966, %r1372, %r763;
	mul.f32 	%r967, %r1371, %r763;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs155, %rs156}, %r842;
	cvt.f32.bf16 	%r968, %rs156;
	cvt.f32.bf16 	%r969, %rs155;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r970, %r967, %r450, %r969;
	fma.rn.f32 	%r971, %r966, %r451, %r968;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r555, %r971, %r970;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r972, %r1374, %r764;
	mul.f32 	%r973, %r1373, %r764;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs157, %rs158}, %r843;
	cvt.f32.bf16 	%r974, %rs158;
	cvt.f32.bf16 	%r975, %rs157;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r976, %r973, %r450, %r975;
	fma.rn.f32 	%r977, %r972, %r451, %r974;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r560, %r977, %r976;
	cvt.rn.bf16x2.f32 	%r556, %r941, %r940;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r978, %r1406, %r766;
	mul.f32 	%r979, %r1405, %r766;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs159, %rs160}, %r859;
	cvt.f32.bf16 	%r980, %rs160;
	cvt.f32.bf16 	%r981, %rs159;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r982, %r979, %r450, %r981;
	fma.rn.f32 	%r983, %r978, %r451, %r980;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r561, %r983, %r982;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r984, %r1408, %r765;
	mul.f32 	%r985, %r1407, %r765;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs161, %rs162}, %r864;
	cvt.f32.bf16 	%r986, %rs162;
	cvt.f32.bf16 	%r987, %rs161;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r988, %r985, %r452, %r987;
	fma.rn.f32 	%r989, %r984, %r453, %r986;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r990, %r1312, %r759;
	mul.f32 	%r991, %r1311, %r759;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs163, %rs164}, %r816;
	cvt.f32.bf16 	%r992, %rs164;
	cvt.f32.bf16 	%r993, %rs163;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r994, %r991, %r452, %r993;
	fma.rn.f32 	%r995, %r990, %r453, %r992;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r573, %r995, %r994;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r996, %r1314, %r760;
	mul.f32 	%r997, %r1313, %r760;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs165, %rs166}, %r817;
	cvt.f32.bf16 	%r998, %rs166;
	cvt.f32.bf16 	%r999, %rs165;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1000, %r997, %r452, %r999;
	fma.rn.f32 	%r1001, %r996, %r453, %r998;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r578, %r1001, %r1000;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1002, %r1344, %r761;
	mul.f32 	%r1003, %r1343, %r761;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs167, %rs168}, %r832;
	cvt.f32.bf16 	%r1004, %rs168;
	cvt.f32.bf16 	%r1005, %rs167;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1006, %r1003, %r452, %r1005;
	fma.rn.f32 	%r1007, %r1002, %r453, %r1004;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r574, %r1007, %r1006;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1008, %r1346, %r762;
	mul.f32 	%r1009, %r1345, %r762;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs169, %rs170}, %r833;
	cvt.f32.bf16 	%r1010, %rs170;
	cvt.f32.bf16 	%r1011, %rs169;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1012, %r1009, %r452, %r1011;
	fma.rn.f32 	%r1013, %r1008, %r453, %r1010;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r579, %r1013, %r1012;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1014, %r1376, %r763;
	mul.f32 	%r1015, %r1375, %r763;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs171, %rs172}, %r848;
	cvt.f32.bf16 	%r1016, %rs172;
	cvt.f32.bf16 	%r1017, %rs171;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1018, %r1015, %r452, %r1017;
	fma.rn.f32 	%r1019, %r1014, %r453, %r1016;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r575, %r1019, %r1018;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1020, %r1378, %r764;
	mul.f32 	%r1021, %r1377, %r764;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs173, %rs174}, %r849;
	cvt.f32.bf16 	%r1022, %rs174;
	cvt.f32.bf16 	%r1023, %rs173;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1024, %r1021, %r452, %r1023;
	fma.rn.f32 	%r1025, %r1020, %r453, %r1022;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r580, %r1025, %r1024;
	cvt.rn.bf16x2.f32 	%r576, %r989, %r988;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1026, %r1410, %r766;
	mul.f32 	%r1027, %r1409, %r766;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs175, %rs176}, %r865;
	cvt.f32.bf16 	%r1028, %rs176;
	cvt.f32.bf16 	%r1029, %rs175;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1030, %r1027, %r452, %r1029;
	fma.rn.f32 	%r1031, %r1026, %r453, %r1028;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r581, %r1031, %r1030;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1032, %r1412, %r765;
	mul.f32 	%r1033, %r1411, %r765;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs177, %rs178}, %r866;
	cvt.f32.bf16 	%r1034, %rs178;
	cvt.f32.bf16 	%r1035, %rs177;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1036, %r1033, %r454, %r1035;
	fma.rn.f32 	%r1037, %r1032, %r455, %r1034;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1038, %r1316, %r759;
	mul.f32 	%r1039, %r1315, %r759;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs179, %rs180}, %r818;
	cvt.f32.bf16 	%r1040, %rs180;
	cvt.f32.bf16 	%r1041, %rs179;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1042, %r1039, %r454, %r1041;
	fma.rn.f32 	%r1043, %r1038, %r455, %r1040;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r593, %r1043, %r1042;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1044, %r1318, %r760;
	mul.f32 	%r1045, %r1317, %r760;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs181, %rs182}, %r819;
	cvt.f32.bf16 	%r1046, %rs182;
	cvt.f32.bf16 	%r1047, %rs181;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1048, %r1045, %r454, %r1047;
	fma.rn.f32 	%r1049, %r1044, %r455, %r1046;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r598, %r1049, %r1048;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1050, %r1348, %r761;
	mul.f32 	%r1051, %r1347, %r761;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs183, %rs184}, %r834;
	cvt.f32.bf16 	%r1052, %rs184;
	cvt.f32.bf16 	%r1053, %rs183;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1054, %r1051, %r454, %r1053;
	fma.rn.f32 	%r1055, %r1050, %r455, %r1052;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r594, %r1055, %r1054;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1056, %r1350, %r762;
	mul.f32 	%r1057, %r1349, %r762;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs185, %rs186}, %r835;
	cvt.f32.bf16 	%r1058, %rs186;
	cvt.f32.bf16 	%r1059, %rs185;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1060, %r1057, %r454, %r1059;
	fma.rn.f32 	%r1061, %r1056, %r455, %r1058;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r599, %r1061, %r1060;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1062, %r1380, %r763;
	mul.f32 	%r1063, %r1379, %r763;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs187, %rs188}, %r850;
	cvt.f32.bf16 	%r1064, %rs188;
	cvt.f32.bf16 	%r1065, %rs187;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1066, %r1063, %r454, %r1065;
	fma.rn.f32 	%r1067, %r1062, %r455, %r1064;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r595, %r1067, %r1066;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1068, %r1382, %r764;
	mul.f32 	%r1069, %r1381, %r764;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs189, %rs190}, %r851;
	cvt.f32.bf16 	%r1070, %rs190;
	cvt.f32.bf16 	%r1071, %rs189;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1072, %r1069, %r454, %r1071;
	fma.rn.f32 	%r1073, %r1068, %r455, %r1070;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r600, %r1073, %r1072;
	cvt.rn.bf16x2.f32 	%r596, %r1037, %r1036;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1074, %r1414, %r766;
	mul.f32 	%r1075, %r1413, %r766;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs191, %rs192}, %r867;
	cvt.f32.bf16 	%r1076, %rs192;
	cvt.f32.bf16 	%r1077, %rs191;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1078, %r1075, %r454, %r1077;
	fma.rn.f32 	%r1079, %r1074, %r455, %r1076;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r601, %r1079, %r1078;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1080, %r1416, %r765;
	mul.f32 	%r1081, %r1415, %r765;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs193, %rs194}, %r860;
	cvt.f32.bf16 	%r1082, %rs194;
	cvt.f32.bf16 	%r1083, %rs193;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1084, %r1081, %r456, %r1083;
	fma.rn.f32 	%r1085, %r1080, %r457, %r1082;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1086, %r1320, %r759;
	mul.f32 	%r1087, %r1319, %r759;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs195, %rs196}, %r810;
	cvt.f32.bf16 	%r1088, %rs196;
	cvt.f32.bf16 	%r1089, %rs195;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1090, %r1087, %r456, %r1089;
	fma.rn.f32 	%r1091, %r1086, %r457, %r1088;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r543, %r1091, %r1090;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1092, %r1322, %r760;
	mul.f32 	%r1093, %r1321, %r760;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs197, %rs198}, %r811;
	cvt.f32.bf16 	%r1094, %rs198;
	cvt.f32.bf16 	%r1095, %rs197;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1096, %r1093, %r456, %r1095;
	fma.rn.f32 	%r1097, %r1092, %r457, %r1094;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r548, %r1097, %r1096;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1098, %r1352, %r761;
	mul.f32 	%r1099, %r1351, %r761;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs199, %rs200}, %r828;
	cvt.f32.bf16 	%r1100, %rs200;
	cvt.f32.bf16 	%r1101, %rs199;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1102, %r1099, %r456, %r1101;
	fma.rn.f32 	%r1103, %r1098, %r457, %r1100;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r544, %r1103, %r1102;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1104, %r1354, %r762;
	mul.f32 	%r1105, %r1353, %r762;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs201, %rs202}, %r829;
	cvt.f32.bf16 	%r1106, %rs202;
	cvt.f32.bf16 	%r1107, %rs201;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1108, %r1105, %r456, %r1107;
	fma.rn.f32 	%r1109, %r1104, %r457, %r1106;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r549, %r1109, %r1108;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1110, %r1384, %r763;
	mul.f32 	%r1111, %r1383, %r763;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs203, %rs204}, %r844;
	cvt.f32.bf16 	%r1112, %rs204;
	cvt.f32.bf16 	%r1113, %rs203;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1114, %r1111, %r456, %r1113;
	fma.rn.f32 	%r1115, %r1110, %r457, %r1112;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r545, %r1115, %r1114;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1116, %r1386, %r764;
	mul.f32 	%r1117, %r1385, %r764;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs205, %rs206}, %r845;
	cvt.f32.bf16 	%r1118, %rs206;
	cvt.f32.bf16 	%r1119, %rs205;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1120, %r1117, %r456, %r1119;
	fma.rn.f32 	%r1121, %r1116, %r457, %r1118;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r550, %r1121, %r1120;
	cvt.rn.bf16x2.f32 	%r546, %r1085, %r1084;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1122, %r1418, %r766;
	mul.f32 	%r1123, %r1417, %r766;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs207, %rs208}, %r861;
	cvt.f32.bf16 	%r1124, %rs208;
	cvt.f32.bf16 	%r1125, %rs207;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1126, %r1123, %r456, %r1125;
	fma.rn.f32 	%r1127, %r1122, %r457, %r1124;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r551, %r1127, %r1126;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1128, %r1420, %r765;
	mul.f32 	%r1129, %r1419, %r765;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs209, %rs210}, %r862;
	cvt.f32.bf16 	%r1130, %rs210;
	cvt.f32.bf16 	%r1131, %rs209;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1132, %r1129, %r458, %r1131;
	fma.rn.f32 	%r1133, %r1128, %r459, %r1130;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1134, %r1324, %r759;
	mul.f32 	%r1135, %r1323, %r759;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs211, %rs212}, %r812;
	cvt.f32.bf16 	%r1136, %rs212;
	cvt.f32.bf16 	%r1137, %rs211;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1138, %r1135, %r458, %r1137;
	fma.rn.f32 	%r1139, %r1134, %r459, %r1136;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r563, %r1139, %r1138;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1140, %r1326, %r760;
	mul.f32 	%r1141, %r1325, %r760;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs213, %rs214}, %r813;
	cvt.f32.bf16 	%r1142, %rs214;
	cvt.f32.bf16 	%r1143, %rs213;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1144, %r1141, %r458, %r1143;
	fma.rn.f32 	%r1145, %r1140, %r459, %r1142;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r568, %r1145, %r1144;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1146, %r1356, %r761;
	mul.f32 	%r1147, %r1355, %r761;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs215, %rs216}, %r830;
	cvt.f32.bf16 	%r1148, %rs216;
	cvt.f32.bf16 	%r1149, %rs215;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1150, %r1147, %r458, %r1149;
	fma.rn.f32 	%r1151, %r1146, %r459, %r1148;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r564, %r1151, %r1150;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1152, %r1358, %r762;
	mul.f32 	%r1153, %r1357, %r762;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs217, %rs218}, %r831;
	cvt.f32.bf16 	%r1154, %rs218;
	cvt.f32.bf16 	%r1155, %rs217;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1156, %r1153, %r458, %r1155;
	fma.rn.f32 	%r1157, %r1152, %r459, %r1154;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r569, %r1157, %r1156;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1158, %r1388, %r763;
	mul.f32 	%r1159, %r1387, %r763;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs219, %rs220}, %r846;
	cvt.f32.bf16 	%r1160, %rs220;
	cvt.f32.bf16 	%r1161, %rs219;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1162, %r1159, %r458, %r1161;
	fma.rn.f32 	%r1163, %r1158, %r459, %r1160;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r565, %r1163, %r1162;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1164, %r1390, %r764;
	mul.f32 	%r1165, %r1389, %r764;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs221, %rs222}, %r847;
	cvt.f32.bf16 	%r1166, %rs222;
	cvt.f32.bf16 	%r1167, %rs221;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1168, %r1165, %r458, %r1167;
	fma.rn.f32 	%r1169, %r1164, %r459, %r1166;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r570, %r1169, %r1168;
	cvt.rn.bf16x2.f32 	%r566, %r1133, %r1132;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1170, %r1422, %r766;
	mul.f32 	%r1171, %r1421, %r766;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs223, %rs224}, %r863;
	cvt.f32.bf16 	%r1172, %rs224;
	cvt.f32.bf16 	%r1173, %rs223;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1174, %r1171, %r458, %r1173;
	fma.rn.f32 	%r1175, %r1170, %r459, %r1172;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r571, %r1175, %r1174;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1176, %r1424, %r765;
	mul.f32 	%r1177, %r1423, %r765;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs225, %rs226}, %r868;
	cvt.f32.bf16 	%r1178, %rs226;
	cvt.f32.bf16 	%r1179, %rs225;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1180, %r1177, %r460, %r1179;
	fma.rn.f32 	%r1181, %r1176, %r461, %r1178;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1182, %r1328, %r759;
	mul.f32 	%r1183, %r1327, %r759;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs227, %rs228}, %r820;
	cvt.f32.bf16 	%r1184, %rs228;
	cvt.f32.bf16 	%r1185, %rs227;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1186, %r1183, %r460, %r1185;
	fma.rn.f32 	%r1187, %r1182, %r461, %r1184;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r583, %r1187, %r1186;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1188, %r1330, %r760;
	mul.f32 	%r1189, %r1329, %r760;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs229, %rs230}, %r821;
	cvt.f32.bf16 	%r1190, %rs230;
	cvt.f32.bf16 	%r1191, %rs229;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1192, %r1189, %r460, %r1191;
	fma.rn.f32 	%r1193, %r1188, %r461, %r1190;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r588, %r1193, %r1192;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1194, %r1360, %r761;
	mul.f32 	%r1195, %r1359, %r761;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs231, %rs232}, %r836;
	cvt.f32.bf16 	%r1196, %rs232;
	cvt.f32.bf16 	%r1197, %rs231;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1198, %r1195, %r460, %r1197;
	fma.rn.f32 	%r1199, %r1194, %r461, %r1196;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r584, %r1199, %r1198;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1200, %r1362, %r762;
	mul.f32 	%r1201, %r1361, %r762;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs233, %rs234}, %r837;
	cvt.f32.bf16 	%r1202, %rs234;
	cvt.f32.bf16 	%r1203, %rs233;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1204, %r1201, %r460, %r1203;
	fma.rn.f32 	%r1205, %r1200, %r461, %r1202;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r589, %r1205, %r1204;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1206, %r1392, %r763;
	mul.f32 	%r1207, %r1391, %r763;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs235, %rs236}, %r852;
	cvt.f32.bf16 	%r1208, %rs236;
	cvt.f32.bf16 	%r1209, %rs235;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1210, %r1207, %r460, %r1209;
	fma.rn.f32 	%r1211, %r1206, %r461, %r1208;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r585, %r1211, %r1210;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1212, %r1394, %r764;
	mul.f32 	%r1213, %r1393, %r764;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs237, %rs238}, %r853;
	cvt.f32.bf16 	%r1214, %rs238;
	cvt.f32.bf16 	%r1215, %rs237;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1216, %r1213, %r460, %r1215;
	fma.rn.f32 	%r1217, %r1212, %r461, %r1214;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r590, %r1217, %r1216;
	cvt.rn.bf16x2.f32 	%r586, %r1181, %r1180;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1218, %r1426, %r766;
	mul.f32 	%r1219, %r1425, %r766;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs239, %rs240}, %r869;
	cvt.f32.bf16 	%r1220, %rs240;
	cvt.f32.bf16 	%r1221, %rs239;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1222, %r1219, %r460, %r1221;
	fma.rn.f32 	%r1223, %r1218, %r461, %r1220;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r591, %r1223, %r1222;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1224, %r1428, %r765;
	mul.f32 	%r1225, %r1427, %r765;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs241, %rs242}, %r870;
	cvt.f32.bf16 	%r1226, %rs242;
	cvt.f32.bf16 	%r1227, %rs241;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1228, %r1225, %r462, %r1227;
	fma.rn.f32 	%r1229, %r1224, %r463, %r1226;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1230, %r1332, %r759;
	mul.f32 	%r1231, %r1331, %r759;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs243, %rs244}, %r822;
	cvt.f32.bf16 	%r1232, %rs244;
	cvt.f32.bf16 	%r1233, %rs243;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1234, %r1231, %r462, %r1233;
	fma.rn.f32 	%r1235, %r1230, %r463, %r1232;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r603, %r1235, %r1234;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1236, %r1334, %r760;
	mul.f32 	%r1237, %r1333, %r760;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs245, %rs246}, %r823;
	cvt.f32.bf16 	%r1238, %rs246;
	cvt.f32.bf16 	%r1239, %rs245;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1240, %r1237, %r462, %r1239;
	fma.rn.f32 	%r1241, %r1236, %r463, %r1238;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r608, %r1241, %r1240;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1242, %r1364, %r761;
	mul.f32 	%r1243, %r1363, %r761;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs247, %rs248}, %r838;
	cvt.f32.bf16 	%r1244, %rs248;
	cvt.f32.bf16 	%r1245, %rs247;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1246, %r1243, %r462, %r1245;
	fma.rn.f32 	%r1247, %r1242, %r463, %r1244;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r604, %r1247, %r1246;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1248, %r1366, %r762;
	mul.f32 	%r1249, %r1365, %r762;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs249, %rs250}, %r839;
	cvt.f32.bf16 	%r1250, %rs250;
	cvt.f32.bf16 	%r1251, %rs249;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1252, %r1249, %r462, %r1251;
	fma.rn.f32 	%r1253, %r1248, %r463, %r1250;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r609, %r1253, %r1252;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1254, %r1396, %r763;
	mul.f32 	%r1255, %r1395, %r763;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs251, %rs252}, %r854;
	cvt.f32.bf16 	%r1256, %rs252;
	cvt.f32.bf16 	%r1257, %rs251;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1258, %r1255, %r462, %r1257;
	fma.rn.f32 	%r1259, %r1254, %r463, %r1256;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r605, %r1259, %r1258;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1260, %r1398, %r764;
	mul.f32 	%r1261, %r1397, %r764;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs253, %rs254}, %r855;
	cvt.f32.bf16 	%r1262, %rs254;
	cvt.f32.bf16 	%r1263, %rs253;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1264, %r1261, %r462, %r1263;
	fma.rn.f32 	%r1265, %r1260, %r463, %r1262;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r610, %r1265, %r1264;
	cvt.rn.bf16x2.f32 	%r606, %r1229, %r1228;
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	mul.f32 	%r1266, %r1430, %r766;
	mul.f32 	%r1267, %r1429, %r766;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs255, %rs256}, %r871;
	cvt.f32.bf16 	%r1268, %rs256;
	cvt.f32.bf16 	%r1269, %rs255;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1270, %r1267, %r462, %r1269;
	fma.rn.f32 	%r1271, %r1266, %r463, %r1268;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r611, %r1271, %r1270;
	bar.sync 	0;
	shl.b32 	%r1272, %r5, 14;
	shl.b32 	%r1273, %r5, 5;
	and.b32 	%r1274, %r1301, 3456;
	bfe.s32 	%r1275, %r2, 2, 1;
	and.b32 	%r1276, %r1275, 8208;
	or.b32 	%r1277, %r1273, %r1274;
	xor.b32 	%r1278, %r1276, %r800;
	or.b32 	%r1279, %r1278, %r1277;
	or.b32 	%r1280, %r1279, %r1272;
	add.s32 	%r532, %r94, %r1280;
	// begin inline asm
	st.shared.v4.b32 [ %r532 + 0 ], { %r533, %r534, %r535, %r536 };
	// end inline asm
	add.s32 	%r537, %r532, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r537 + 0 ], { %r538, %r539, %r540, %r541 };
	// end inline asm
	add.s32 	%r542, %r532, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r542 + 0 ], { %r543, %r544, %r545, %r546 };
	// end inline asm
	add.s32 	%r547, %r532, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r547 + 0 ], { %r548, %r549, %r550, %r551 };
	// end inline asm
	xor.b32 	%r1281, %r1280, 32;
	add.s32 	%r552, %r94, %r1281;
	// begin inline asm
	st.shared.v4.b32 [ %r552 + 0 ], { %r553, %r554, %r555, %r556 };
	// end inline asm
	add.s32 	%r557, %r552, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r557 + 0 ], { %r558, %r559, %r560, %r561 };
	// end inline asm
	add.s32 	%r562, %r552, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r562 + 0 ], { %r563, %r564, %r565, %r566 };
	// end inline asm
	add.s32 	%r567, %r552, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r567 + 0 ], { %r568, %r569, %r570, %r571 };
	// end inline asm
	xor.b32 	%r1282, %r1280, 64;
	add.s32 	%r572, %r94, %r1282;
	// begin inline asm
	st.shared.v4.b32 [ %r572 + 0 ], { %r573, %r574, %r575, %r576 };
	// end inline asm
	add.s32 	%r577, %r572, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r577 + 0 ], { %r578, %r579, %r580, %r581 };
	// end inline asm
	add.s32 	%r582, %r572, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r582 + 0 ], { %r583, %r584, %r585, %r586 };
	// end inline asm
	add.s32 	%r587, %r572, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r587 + 0 ], { %r588, %r589, %r590, %r591 };
	// end inline asm
	xor.b32 	%r1283, %r1280, 96;
	add.s32 	%r592, %r94, %r1283;
	// begin inline asm
	st.shared.v4.b32 [ %r592 + 0 ], { %r593, %r594, %r595, %r596 };
	// end inline asm
	add.s32 	%r597, %r592, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r597 + 0 ], { %r598, %r599, %r600, %r601 };
	// end inline asm
	add.s32 	%r602, %r592, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r602 + 0 ], { %r603, %r604, %r605, %r606 };
	// end inline asm
	add.s32 	%r607, %r592, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r607 + 0 ], { %r608, %r609, %r610, %r611 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r1284, %r2, 2;
	and.b32 	%r1285, %r1284, 896;
	shl.b32 	%r1286, %r751, 9;
	selp.b32 	%r1287, 0, 8208, %p24;
	or.b32 	%r1288, %r793, %r1285;
	xor.b32 	%r1289, %r1288, %r1287;
	or.b32 	%r1290, %r1289, %r1286;
	add.s32 	%r1291, %r94, %r1290;
	ld.shared.v4.b32 	{%r612, %r628, %r644, %r660}, [%r1291];
	ld.shared.v4.b32 	{%r616, %r632, %r648, %r664}, [%r1291+1024];
	ld.shared.v4.b32 	{%r620, %r636, %r652, %r668}, [%r1291+2048];
	ld.shared.v4.b32 	{%r624, %r640, %r656, %r672}, [%r1291+3072];
	xor.b32 	%r1292, %r1290, 32;
	add.s32 	%r1293, %r94, %r1292;
	ld.shared.v4.b32 	{%r613, %r629, %r645, %r661}, [%r1293+16384];
	ld.shared.v4.b32 	{%r617, %r633, %r649, %r665}, [%r1293+17408];
	ld.shared.v4.b32 	{%r621, %r637, %r653, %r669}, [%r1293+18432];
	ld.shared.v4.b32 	{%r625, %r641, %r657, %r673}, [%r1293+19456];
	xor.b32 	%r1294, %r1290, 64;
	add.s32 	%r1295, %r94, %r1294;
	ld.shared.v4.b32 	{%r614, %r630, %r646, %r662}, [%r1295+32768];
	ld.shared.v4.b32 	{%r618, %r634, %r650, %r666}, [%r1295+33792];
	ld.shared.v4.b32 	{%r622, %r638, %r654, %r670}, [%r1295+34816];
	ld.shared.v4.b32 	{%r626, %r642, %r658, %r674}, [%r1295+35840];
	xor.b32 	%r1296, %r1290, 96;
	add.s32 	%r1297, %r94, %r1296;
	ld.shared.v4.b32 	{%r615, %r631, %r647, %r663}, [%r1297+49152];
	ld.shared.v4.b32 	{%r619, %r635, %r651, %r667}, [%r1297+50176];
	ld.shared.v4.b32 	{%r623, %r639, %r655, %r671}, [%r1297+51200];
	ld.shared.v4.b32 	{%r627, %r643, %r659, %r675}, [%r1297+52224];
	.loc	1 356 8                         // sk06_mlp_down.py:356:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd186 + 0 ], { %r612, %r613, %r614, %r615 };
	// end inline asm
	// begin inline asm
	@%p9 st.global.v4.b32 [ %rd187 + 0 ], { %r616, %r617, %r618, %r619 };
	// end inline asm
	// begin inline asm
	@%p10 st.global.v4.b32 [ %rd188 + 0 ], { %r620, %r621, %r622, %r623 };
	// end inline asm
	// begin inline asm
	@%p11 st.global.v4.b32 [ %rd189 + 0 ], { %r624, %r625, %r626, %r627 };
	// end inline asm
	// begin inline asm
	@%p12 st.global.v4.b32 [ %rd190 + 0 ], { %r628, %r629, %r630, %r631 };
	// end inline asm
	// begin inline asm
	@%p13 st.global.v4.b32 [ %rd191 + 0 ], { %r632, %r633, %r634, %r635 };
	// end inline asm
	// begin inline asm
	@%p14 st.global.v4.b32 [ %rd192 + 0 ], { %r636, %r637, %r638, %r639 };
	// end inline asm
	// begin inline asm
	@%p15 st.global.v4.b32 [ %rd193 + 0 ], { %r640, %r641, %r642, %r643 };
	// end inline asm
	// begin inline asm
	@%p16 st.global.v4.b32 [ %rd194 + 0 ], { %r644, %r645, %r646, %r647 };
	// end inline asm
	// begin inline asm
	@%p17 st.global.v4.b32 [ %rd195 + 0 ], { %r648, %r649, %r650, %r651 };
	// end inline asm
	// begin inline asm
	@%p18 st.global.v4.b32 [ %rd196 + 0 ], { %r652, %r653, %r654, %r655 };
	// end inline asm
	// begin inline asm
	@%p19 st.global.v4.b32 [ %rd197 + 0 ], { %r656, %r657, %r658, %r659 };
	// end inline asm
	// begin inline asm
	@%p20 st.global.v4.b32 [ %rd198 + 0 ], { %r660, %r661, %r662, %r663 };
	// end inline asm
	// begin inline asm
	@%p21 st.global.v4.b32 [ %rd199 + 0 ], { %r664, %r665, %r666, %r667 };
	// end inline asm
	// begin inline asm
	@%p22 st.global.v4.b32 [ %rd200 + 0 ], { %r668, %r669, %r670, %r671 };
	// end inline asm
	// begin inline asm
	@%p23 st.global.v4.b32 [ %rd201 + 0 ], { %r672, %r673, %r674, %r675 };
	// end inline asm
	.loc	1 354 4                         // sk06_mlp_down.py:354:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/workspace/vllm/_genesis/kernels/sk06_mlp_down.py"
	.file	2 "/usr/local/lib/python3.12/dist-packages/triton/language/standard.py"
	.section	.debug_abbrev
	{
.b8 1                                   // Abbreviation Code
.b8 17                                  // DW_TAG_compile_unit
.b8 1                                   // DW_CHILDREN_yes
.b8 37                                  // DW_AT_producer
.b8 8                                   // DW_FORM_string
.b8 19                                  // DW_AT_language
.b8 5                                   // DW_FORM_data2
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 16                                  // DW_AT_stmt_list
.b8 6                                   // DW_FORM_data4
.b8 27                                  // DW_AT_comp_dir
.b8 8                                   // DW_FORM_string
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 2                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 0                                   // DW_CHILDREN_no
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 32                                  // DW_AT_inline
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 3                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 1                                   // DW_CHILDREN_yes
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 4                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 0                                   // DW_CHILDREN_no
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 5                                   // DW_FORM_data2
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 168                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0xa1 DW_TAG_compile_unit
.b8 116                                 // DW_AT_producer
.b8 114
.b8 105
.b8 116
.b8 111
.b8 110
.b8 0
.b8 2                                   // DW_AT_language
.b8 0
.b8 115                                 // DW_AT_name
.b8 107
.b8 48
.b8 54
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 100
.b8 111
.b8 119
.b8 110
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 119
.b8 111
.b8 114
.b8 107
.b8 115
.b8 112
.b8 97
.b8 99
.b8 101
.b8 47
.b8 118
.b8 108
.b8 108
.b8 109
.b8 47
.b8 95
.b8 103
.b8 101
.b8 110
.b8 101
.b8 115
.b8 105
.b8 115
.b8 47
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 115
.b8 0
.b8 2                                   // Abbrev [2] 0x4b:0x18 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 54
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 100
.b8 111
.b8 119
.b8 110
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x63:0x48 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 75                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x78:0x19 DW_TAG_inlined_subroutine
.b32 75                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 58                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x91:0x19 DW_TAG_inlined_subroutine
.b32 75                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 59                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_9 = _Nativo(
    "sk06_mlp_down/tile256x128x64_shift0_abi16",
    _PTX_9, "_sk06_mlp_down_kernel",
    warps=8, shared=73728,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 17, 18],
    horneado={11: 1, 12: 1, 15: 1, 19: 1, 20: 256, 21: 128, 22: 64, 23: 8, 24: 128, 25: False},
    div16=[8, 9, 10, 13, 14, 16, 17],
)

_PTX_10 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk06_mlp_down_kernel   // -- Begin function _sk06_mlp_down_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk06_mlp_down_kernel
.visible .entry _sk06_mlp_down_kernel(
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_6,
	.param .u32 _sk06_mlp_down_kernel_param_7,
	.param .u32 _sk06_mlp_down_kernel_param_8,
	.param .u32 _sk06_mlp_down_kernel_param_9,
	.param .u32 _sk06_mlp_down_kernel_param_10,
	.param .u32 _sk06_mlp_down_kernel_param_11,
	.param .u32 _sk06_mlp_down_kernel_param_12,
	.param .u32 _sk06_mlp_down_kernel_param_13,
	.param .u32 _sk06_mlp_down_kernel_param_14,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_16
)
.reqntid 256
{
	.reg .pred 	%p<42>;
	.reg .b16 	%rs<129>;
	.reg .b32 	%r<1363>;
	.reg .b64 	%rd<155>;
	.loc	1 304 0                         // sk06_mlp_down.py:304:0
$L__func_begin0:
	.loc	1 304 0                         // sk06_mlp_down.py:304:0

// %bb.0:
	ld.param.b32 	%r18, [_sk06_mlp_down_kernel_param_13];
	ld.param.b32 	%r17, [_sk06_mlp_down_kernel_param_12];
	ld.param.b32 	%r16, [_sk06_mlp_down_kernel_param_8];
	ld.param.b32 	%r15, [_sk06_mlp_down_kernel_param_7];
	ld.param.b64 	%rd28, [_sk06_mlp_down_kernel_param_4];
	ld.param.b64 	%rd27, [_sk06_mlp_down_kernel_param_3];
	ld.param.b64 	%rd26, [_sk06_mlp_down_kernel_param_2];
	ld.param.b64 	%rd25, [_sk06_mlp_down_kernel_param_1];
	ld.param.b64 	%rd24, [_sk06_mlp_down_kernel_param_0];
$L__tmp0:
	.loc	1 313 24                        // sk06_mlp_down.py:313:24
	mov.u32 	%r41, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk06_mlp_down.py:314:27 ]
	add.s32 	%r42, %r15, 255;
	.loc	2 43 30                         // standard.py:43:30 @[ sk06_mlp_down.py:314:27 ]
	shr.s32 	%r43, %r42, 31;
	shr.u32 	%r44, %r43, 24;
	add.s32 	%r45, %r42, %r44;
	shr.s32 	%r46, %r45, 8;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk06_mlp_down.py:315:27 ]
	add.s32 	%r47, %r16, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk06_mlp_down.py:315:27 ]
	shr.s32 	%r48, %r47, 31;
	shr.u32 	%r49, %r48, 25;
	add.s32 	%r50, %r47, %r49;
	shr.s32 	%r51, %r50, 7;
$L__tmp3:
	.loc	1 316 29                        // sk06_mlp_down.py:316:29
	shl.b32 	%r52, %r51, 3;
	.loc	1 317 22                        // sk06_mlp_down.py:317:22
	div.s32 	%r53, %r41, %r52;
	.loc	1 317 38                        // sk06_mlp_down.py:317:38
	shl.b32 	%r54, %r53, 3;
	ld.param.b32 	%r55, [_sk06_mlp_down_kernel_param_9];
	.loc	1 318 30                        // sk06_mlp_down.py:318:30
	sub.s32 	%r56, %r46, %r54;
	ld.param.b32 	%r57, [_sk06_mlp_down_kernel_param_10];
	.loc	1 318 39                        // sk06_mlp_down.py:318:39
	min.s32 	%r58, %r56, 8;
	ld.param.b32 	%r59, [_sk06_mlp_down_kernel_param_11];
	.loc	1 319 30                        // sk06_mlp_down.py:319:30
	mul.lo.s32 	%r60, %r53, %r52;
	sub.s32 	%r61, %r41, %r60;
	.loc	1 320 36                        // sk06_mlp_down.py:320:36
	div.s32 	%r62, %r61, %r58;
	.loc	1 319 46                        // sk06_mlp_down.py:319:46
	mul.lo.s32 	%r63, %r62, %r58;
	sub.s32 	%r64, %r61, %r63;
	.loc	1 319 23                        // sk06_mlp_down.py:319:23
	add.s32 	%r65, %r64, %r54;
	.loc	1 322 22                        // sk06_mlp_down.py:322:22
	shl.b32 	%r1, %r65, 8;
	.loc	1 322 45                        // sk06_mlp_down.py:322:45
	mov.u32 	%r2, %tid.x;
	shr.u32 	%r66, %r2, 2;
	bfe.u32 	%r67, %r2, 2, 6;
	or.b32 	%r68, %r67, 64;
	and.b32 	%r3, %r2, 255;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r69, %r1, %r67;
	or.b32 	%r70, %r1, %r68;
	or.b32 	%r71, %r69, 128;
	or.b32 	%r72, %r1, %r66;
	or.b32 	%r73, %r72, 192;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r74, %r69, %r15;
	rem.s32 	%r75, %r70, %r15;
	rem.s32 	%r76, %r71, %r15;
	rem.s32 	%r77, %r73, %r15;
	.loc	1 323 22                        // sk06_mlp_down.py:323:22
	shl.b32 	%r4, %r62, 7;
	.loc	1 323 45                        // sk06_mlp_down.py:323:45
	and.b32 	%r5, %r2, 3;
	and.b32 	%r6, %r2, 32;
	and.b32 	%r7, %r2, 15;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r81, %r4, %r67;
	or.b32 	%r82, %r4, %r68;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r99, %r81, %r16;
	rem.s32 	%r100, %r82, %r16;
	.loc	1 326 39                        // sk06_mlp_down.py:326:39
	mul.lo.s32 	%r117, %r74, %r57;
	mul.lo.s32 	%r118, %r75, %r57;
	mul.lo.s32 	%r119, %r76, %r57;
	mul.lo.s32 	%r120, %r77, %r57;
	.loc	1 326 21                        // sk06_mlp_down.py:326:21
	cvt.s64.s32 	%rd1, %r117;
	add.s64 	%rd48, %rd24, %rd1;
	cvt.s64.s32 	%rd2, %r118;
	add.s64 	%rd49, %rd24, %rd2;
	cvt.s64.s32 	%rd3, %r119;
	add.s64 	%rd50, %rd24, %rd3;
	cvt.s64.s32 	%rd4, %r120;
	add.s64 	%rd51, %rd24, %rd4;
	.loc	1 326 58                        // sk06_mlp_down.py:326:58
	shl.b32 	%r121, %r5, 4;
	.loc	1 326 51                        // sk06_mlp_down.py:326:51
	cvt.u64.u32 	%rd5, %r121;
	add.s64 	%rd29, %rd48, %rd5;
	add.s64 	%rd30, %rd49, %rd5;
	add.s64 	%rd31, %rd50, %rd5;
	add.s64 	%rd32, %rd51, %rd5;
	.loc	1 327 21                        // sk06_mlp_down.py:327:21
	add.s64 	%rd52, %rd25, %rd5;
	.loc	1 327 69                        // sk06_mlp_down.py:327:69
	mul.lo.s32 	%r122, %r99, %r59;
	mul.lo.s32 	%r123, %r100, %r59;
	.loc	1 327 51                        // sk06_mlp_down.py:327:51
	cvt.s64.s32 	%rd6, %r122;
	add.s64 	%rd33, %rd52, %rd6;
	cvt.s64.s32 	%rd7, %r123;
	add.s64 	%rd34, %rd52, %rd7;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	setp.gt.s32 	%p1, %r55, 63;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	shl.b32 	%r191, %r3, 4;
	shl.b32 	%r9, %r2, 1;
	and.b32 	%r10, %r9, 48;
	xor.b32 	%r192, %r191, %r10;
	mov.b32 	%r193, global_smem;
	add.s32 	%r20, %r193, %r192;
	selp.b32 	%r21, 16, 0, %p1;
	// begin inline asm
	cp.async.cg.shared.global [ %r20 + 0 ], [ %rd29 + 0 ], 0x10, %r21;
	// end inline asm
	add.s32 	%r22, %r20, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r22 + 0 ], [ %rd30 + 0 ], 0x10, %r21;
	// end inline asm
	add.s32 	%r23, %r20, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r23 + 0 ], [ %rd31 + 0 ], 0x10, %r21;
	// end inline asm
	add.s32 	%r24, %r20, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r24 + 0 ], [ %rd32 + 0 ], 0x10, %r21;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	add.s32 	%r25, %r20, 49152;
	// begin inline asm
	cp.async.cg.shared.global [ %r25 + 0 ], [ %rd33 + 0 ], 0x10, %r21;
	// end inline asm
	add.s32 	%r26, %r20, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r26 + 0 ], [ %rd34 + 0 ], 0x10, %r21;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	setp.gt.s32 	%p2, %r55, 127;
	.loc	1 344 18                        // sk06_mlp_down.py:344:18
	add.s64 	%rd35, %rd29, 64;
	add.s64 	%rd36, %rd30, 64;
	add.s64 	%rd37, %rd31, 64;
	add.s64 	%rd38, %rd32, 64;
	.loc	1 345 18                        // sk06_mlp_down.py:345:18
	add.s64 	%rd39, %rd33, 64;
	add.s64 	%rd40, %rd34, 64;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	bar.sync 	0;
	add.s32 	%r27, %r20, 16384;
	selp.b32 	%r28, 16, 0, %p2;
	// begin inline asm
	cp.async.cg.shared.global [ %r27 + 0 ], [ %rd35 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r29, %r20, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r29 + 0 ], [ %rd36 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r30, %r20, 24576;
	// begin inline asm
	cp.async.cg.shared.global [ %r30 + 0 ], [ %rd37 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r31, %r20, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r31 + 0 ], [ %rd38 + 0 ], 0x10, %r28;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	add.s32 	%r32, %r20, 57344;
	// begin inline asm
	cp.async.cg.shared.global [ %r32 + 0 ], [ %rd39 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r33, %r20, 61440;
	// begin inline asm
	cp.async.cg.shared.global [ %r33 + 0 ], [ %rd40 + 0 ], 0x10, %r28;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	setp.gt.s32 	%p3, %r55, 191;
	.loc	1 344 18                        // sk06_mlp_down.py:344:18
	add.s64 	%rd41, %rd29, 128;
	add.s64 	%rd42, %rd30, 128;
	add.s64 	%rd43, %rd31, 128;
	add.s64 	%rd44, %rd32, 128;
	.loc	1 345 18                        // sk06_mlp_down.py:345:18
	add.s64 	%rd45, %rd33, 128;
	add.s64 	%rd46, %rd34, 128;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	bar.sync 	0;
	add.s32 	%r34, %r20, 32768;
	selp.b32 	%r35, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r34 + 0 ], [ %rd41 + 0 ], 0x10, %r35;
	// end inline asm
	add.s32 	%r36, %r20, 36864;
	// begin inline asm
	cp.async.cg.shared.global [ %r36 + 0 ], [ %rd42 + 0 ], 0x10, %r35;
	// end inline asm
	add.s32 	%r37, %r20, 40960;
	// begin inline asm
	cp.async.cg.shared.global [ %r37 + 0 ], [ %rd43 + 0 ], 0x10, %r35;
	// end inline asm
	add.s32 	%r38, %r20, 45056;
	// begin inline asm
	cp.async.cg.shared.global [ %r38 + 0 ], [ %rd44 + 0 ], 0x10, %r35;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	add.s32 	%r39, %r20, 65536;
	// begin inline asm
	cp.async.cg.shared.global [ %r39 + 0 ], [ %rd45 + 0 ], 0x10, %r35;
	// end inline asm
	add.s32 	%r40, %r20, 69632;
	// begin inline asm
	cp.async.cg.shared.global [ %r40 + 0 ], [ %rd46 + 0 ], 0x10, %r35;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	@%p1 bra 	$L__BB0_2;
	bra.uni 	$L__BB0_1;
$L__BB0_2:                              // %.lr.ph
	.loc	1 0 23                          // sk06_mlp_down.py:0:23
	ld.param.b32 	%r19, [_sk06_mlp_down_kernel_param_14];
	ld.param.b64 	%rd47, [_sk06_mlp_down_kernel_param_6];
	shl.b32 	%r78, %r5, 1;
	shr.u32 	%r79, %r6, 2;
	or.b32 	%r80, %r79, %r78;
	or.b32 	%r83, %r4, %r80;
	or.b32 	%r84, %r83, 1;
	or.b32 	%r85, %r83, 16;
	or.b32 	%r86, %r83, 17;
	or.b32 	%r87, %r83, 32;
	or.b32 	%r88, %r83, 33;
	or.b32 	%r89, %r83, 48;
	or.b32 	%r90, %r83, 49;
	or.b32 	%r91, %r83, 64;
	or.b32 	%r92, %r83, 65;
	or.b32 	%r93, %r83, 80;
	or.b32 	%r94, %r83, 81;
	or.b32 	%r95, %r83, 96;
	or.b32 	%r96, %r83, 97;
	or.b32 	%r97, %r83, 112;
	or.b32 	%r98, %r83, 113;
	rem.s32 	%r101, %r83, %r16;
	rem.s32 	%r102, %r84, %r16;
	rem.s32 	%r103, %r85, %r16;
	rem.s32 	%r104, %r86, %r16;
	rem.s32 	%r105, %r87, %r16;
	rem.s32 	%r106, %r88, %r16;
	rem.s32 	%r107, %r89, %r16;
	rem.s32 	%r108, %r90, %r16;
	rem.s32 	%r109, %r91, %r16;
	rem.s32 	%r110, %r92, %r16;
	rem.s32 	%r111, %r93, %r16;
	rem.s32 	%r112, %r94, %r16;
	rem.s32 	%r113, %r95, %r16;
	rem.s32 	%r114, %r96, %r16;
	rem.s32 	%r115, %r97, %r16;
	rem.s32 	%r116, %r98, %r16;
	shr.s32 	%r124, %r101, 31;
	shr.u32 	%r125, %r124, 25;
	add.s32 	%r126, %r101, %r125;
	shr.s32 	%r127, %r126, 7;
	shr.s32 	%r128, %r102, 31;
	shr.u32 	%r129, %r128, 25;
	add.s32 	%r130, %r102, %r129;
	shr.s32 	%r131, %r130, 7;
	shr.s32 	%r132, %r103, 31;
	shr.u32 	%r133, %r132, 25;
	add.s32 	%r134, %r103, %r133;
	shr.s32 	%r135, %r134, 7;
	shr.s32 	%r136, %r104, 31;
	shr.u32 	%r137, %r136, 25;
	add.s32 	%r138, %r104, %r137;
	shr.s32 	%r139, %r138, 7;
	shr.s32 	%r140, %r105, 31;
	shr.u32 	%r141, %r140, 25;
	add.s32 	%r142, %r105, %r141;
	shr.s32 	%r143, %r142, 7;
	shr.s32 	%r144, %r106, 31;
	shr.u32 	%r145, %r144, 25;
	add.s32 	%r146, %r106, %r145;
	shr.s32 	%r147, %r146, 7;
	shr.s32 	%r148, %r107, 31;
	shr.u32 	%r149, %r148, 25;
	add.s32 	%r150, %r107, %r149;
	shr.s32 	%r151, %r150, 7;
	shr.s32 	%r152, %r108, 31;
	shr.u32 	%r153, %r152, 25;
	add.s32 	%r154, %r108, %r153;
	shr.s32 	%r155, %r154, 7;
	shr.s32 	%r156, %r109, 31;
	shr.u32 	%r157, %r156, 25;
	add.s32 	%r158, %r109, %r157;
	shr.s32 	%r159, %r158, 7;
	shr.s32 	%r160, %r110, 31;
	shr.u32 	%r161, %r160, 25;
	add.s32 	%r162, %r110, %r161;
	shr.s32 	%r163, %r162, 7;
	shr.s32 	%r164, %r111, 31;
	shr.u32 	%r165, %r164, 25;
	add.s32 	%r166, %r111, %r165;
	shr.s32 	%r167, %r166, 7;
	shr.s32 	%r168, %r112, 31;
	shr.u32 	%r169, %r168, 25;
	add.s32 	%r170, %r112, %r169;
	shr.s32 	%r171, %r170, 7;
	shr.s32 	%r172, %r113, 31;
	shr.u32 	%r173, %r172, 25;
	add.s32 	%r174, %r113, %r173;
	shr.s32 	%r175, %r174, 7;
	shr.s32 	%r176, %r114, 31;
	shr.u32 	%r177, %r176, 25;
	add.s32 	%r178, %r114, %r177;
	shr.s32 	%r179, %r178, 7;
	shr.s32 	%r180, %r115, 31;
	shr.u32 	%r181, %r180, 25;
	add.s32 	%r182, %r115, %r181;
	shr.s32 	%r183, %r182, 7;
	shr.s32 	%r184, %r116, 31;
	shr.u32 	%r185, %r184, 25;
	add.s32 	%r186, %r116, %r185;
	shr.s32 	%r187, %r186, 7;
	mad.wide.s32 	%rd8, %r127, 4, %rd47;
	mad.wide.s32 	%rd9, %r131, 4, %rd47;
	mad.wide.s32 	%rd10, %r135, 4, %rd47;
	mad.wide.s32 	%rd11, %r139, 4, %rd47;
	mad.wide.s32 	%rd12, %r143, 4, %rd47;
	mad.wide.s32 	%rd13, %r147, 4, %rd47;
	mad.wide.s32 	%rd14, %r151, 4, %rd47;
	mad.wide.s32 	%rd15, %r155, 4, %rd47;
	mad.wide.s32 	%rd16, %r159, 4, %rd47;
	mad.wide.s32 	%rd17, %r163, 4, %rd47;
	mad.wide.s32 	%rd18, %r167, 4, %rd47;
	mad.wide.s32 	%rd19, %r171, 4, %rd47;
	mad.wide.s32 	%rd20, %r175, 4, %rd47;
	mad.wide.s32 	%rd21, %r179, 4, %rd47;
	mad.wide.s32 	%rd22, %r183, 4, %rd47;
	mad.wide.s32 	%rd23, %r187, 4, %rd47;
	shr.s32 	%r188, %r55, 31;
	shr.u32 	%r189, %r188, 26;
	add.s32 	%r190, %r55, %r189;
	shr.s32 	%r8, %r190, 6;
	add.s32 	%r11, %r8, -3;
	shl.b32 	%r194, %r7, 6;
	shl.b32 	%r1233, %r2, 4;
	and.b32 	%r195, %r1233, 3072;
	shl.b32 	%r196, %r2, 3;
	and.b32 	%r197, %r196, 48;
	and.b32 	%r1234, %r2, 16;
	or.b32 	%r198, %r194, %r195;
	xor.b32 	%r199, %r197, %r1234;
	or.b32 	%r12, %r198, %r199;
	xor.b32 	%r13, %r12, 32;
	shl.b32 	%r200, %r2, 6;
	and.b32 	%r201, %r200, 448;
	shl.b32 	%r202, %r6, 4;
	or.b32 	%r203, %r201, %r197;
	xor.b32 	%r204, %r203, %r10;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	add.s32 	%r205, %r193, %r202;
	add.s32 	%r14, %r205, %r204;
	add.s64 	%rd53, %rd7, %rd25;
	add.s64 	%rd154, %rd53, 192;
	add.s64 	%rd54, %rd6, %rd25;
	add.s64 	%rd153, %rd54, 192;
	add.s64 	%rd55, %rd4, %rd24;
	add.s64 	%rd152, %rd55, 192;
	add.s64 	%rd56, %rd3, %rd24;
	add.s64 	%rd151, %rd56, 192;
	add.s64 	%rd57, %rd2, %rd24;
	add.s64 	%rd150, %rd57, 192;
	add.s64 	%rd58, %rd1, %rd24;
	add.s64 	%rd149, %rd58, 192;
	mov.b32 	%r1235, 0f00000000;
	mov.b32 	%r206, 0;
	mov.b32 	%r1231, 2;
	mov.b32 	%r1230, -1;
	mov.b32 	%r1232, %r206;
	mov.b32 	%r1236, %r1235;
	mov.b32 	%r1237, %r1235;
	mov.b32 	%r1238, %r1235;
	mov.b32 	%r1239, %r1235;
	mov.b32 	%r1240, %r1235;
	mov.b32 	%r1241, %r1235;
	mov.b32 	%r1242, %r1235;
	mov.b32 	%r1243, %r1235;
	mov.b32 	%r1244, %r1235;
	mov.b32 	%r1245, %r1235;
	mov.b32 	%r1246, %r1235;
	mov.b32 	%r1247, %r1235;
	mov.b32 	%r1248, %r1235;
	mov.b32 	%r1249, %r1235;
	mov.b32 	%r1250, %r1235;
	mov.b32 	%r1251, %r1235;
	mov.b32 	%r1252, %r1235;
	mov.b32 	%r1253, %r1235;
	mov.b32 	%r1254, %r1235;
	mov.b32 	%r1255, %r1235;
	mov.b32 	%r1256, %r1235;
	mov.b32 	%r1257, %r1235;
	mov.b32 	%r1258, %r1235;
	mov.b32 	%r1259, %r1235;
	mov.b32 	%r1260, %r1235;
	mov.b32 	%r1261, %r1235;
	mov.b32 	%r1262, %r1235;
	mov.b32 	%r1263, %r1235;
	mov.b32 	%r1264, %r1235;
	mov.b32 	%r1265, %r1235;
	mov.b32 	%r1266, %r1235;
	mov.b32 	%r1267, %r1235;
	mov.b32 	%r1268, %r1235;
	mov.b32 	%r1269, %r1235;
	mov.b32 	%r1270, %r1235;
	mov.b32 	%r1271, %r1235;
	mov.b32 	%r1272, %r1235;
	mov.b32 	%r1273, %r1235;
	mov.b32 	%r1274, %r1235;
	mov.b32 	%r1275, %r1235;
	mov.b32 	%r1276, %r1235;
	mov.b32 	%r1277, %r1235;
	mov.b32 	%r1278, %r1235;
	mov.b32 	%r1279, %r1235;
	mov.b32 	%r1280, %r1235;
	mov.b32 	%r1281, %r1235;
	mov.b32 	%r1282, %r1235;
	mov.b32 	%r1283, %r1235;
	mov.b32 	%r1284, %r1235;
	mov.b32 	%r1285, %r1235;
	mov.b32 	%r1286, %r1235;
	mov.b32 	%r1287, %r1235;
	mov.b32 	%r1288, %r1235;
	mov.b32 	%r1289, %r1235;
	mov.b32 	%r1290, %r1235;
	mov.b32 	%r1291, %r1235;
	mov.b32 	%r1292, %r1235;
	mov.b32 	%r1293, %r1235;
	mov.b32 	%r1294, %r1235;
	mov.b32 	%r1295, %r1235;
	mov.b32 	%r1296, %r1235;
	mov.b32 	%r1297, %r1235;
	mov.b32 	%r1298, %r1235;
	mov.b32 	%r1299, %r1235;
	mov.b32 	%r1300, %r1235;
	mov.b32 	%r1301, %r1235;
	mov.b32 	%r1302, %r1235;
	mov.b32 	%r1303, %r1235;
	mov.b32 	%r1304, %r1235;
	mov.b32 	%r1305, %r1235;
	mov.b32 	%r1306, %r1235;
	mov.b32 	%r1307, %r1235;
	mov.b32 	%r1308, %r1235;
	mov.b32 	%r1309, %r1235;
	mov.b32 	%r1310, %r1235;
	mov.b32 	%r1311, %r1235;
	mov.b32 	%r1312, %r1235;
	mov.b32 	%r1313, %r1235;
	mov.b32 	%r1314, %r1235;
	mov.b32 	%r1315, %r1235;
	mov.b32 	%r1316, %r1235;
	mov.b32 	%r1317, %r1235;
	mov.b32 	%r1318, %r1235;
	mov.b32 	%r1319, %r1235;
	mov.b32 	%r1320, %r1235;
	mov.b32 	%r1321, %r1235;
	mov.b32 	%r1322, %r1235;
	mov.b32 	%r1323, %r1235;
	mov.b32 	%r1324, %r1235;
	mov.b32 	%r1325, %r1235;
	mov.b32 	%r1326, %r1235;
	mov.b32 	%r1327, %r1235;
	mov.b32 	%r1328, %r1235;
	mov.b32 	%r1329, %r1235;
	mov.b32 	%r1330, %r1235;
	mov.b32 	%r1331, %r1235;
	mov.b32 	%r1332, %r1235;
	mov.b32 	%r1333, %r1235;
	mov.b32 	%r1334, %r1235;
	mov.b32 	%r1335, %r1235;
	mov.b32 	%r1336, %r1235;
	mov.b32 	%r1337, %r1235;
	mov.b32 	%r1338, %r1235;
	mov.b32 	%r1339, %r1235;
	mov.b32 	%r1340, %r1235;
	mov.b32 	%r1341, %r1235;
	mov.b32 	%r1342, %r1235;
	mov.b32 	%r1343, %r1235;
	mov.b32 	%r1344, %r1235;
	mov.b32 	%r1345, %r1235;
	mov.b32 	%r1346, %r1235;
	mov.b32 	%r1347, %r1235;
	mov.b32 	%r1348, %r1235;
	mov.b32 	%r1349, %r1235;
	mov.b32 	%r1350, %r1235;
	mov.b32 	%r1351, %r1235;
	mov.b32 	%r1352, %r1235;
	mov.b32 	%r1353, %r1235;
	mov.b32 	%r1354, %r1235;
	mov.b32 	%r1355, %r1235;
	mov.b32 	%r1356, %r1235;
	mov.b32 	%r1357, %r1235;
	mov.b32 	%r1358, %r1235;
	mov.b32 	%r1359, %r1235;
	mov.b32 	%r1360, %r1235;
	mov.b32 	%r1361, %r1235;
	mov.b32 	%r1362, %r1235;
$L__BB0_3:                              // =>This Inner Loop Header: Depth=1
	setp.lt.s32 	%p4, %r1232, %r11;
	add.s32 	%r422, %r1230, 1;
	setp.gt.s32 	%p5, %r422, 2;
	selp.b32 	%r1230, 0, %r422, %p5;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	cp.async.wait_group 	4;
	bar.sync 	0;
	shl.b32 	%r423, %r1230, 14;
	add.s32 	%r424, %r193, %r423;
	add.s32 	%r425, %r424, %r12;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r207, %r208, %r209, %r210}, [%r425];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r227, %r228, %r229, %r230}, [%r425+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r231, %r232, %r233, %r234}, [%r425+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r235, %r236, %r237, %r238}, [%r425+12288];
	add.s32 	%r426, %r424, %r13;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r243, %r244, %r245, %r246}, [%r426];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r295, %r296, %r297, %r298}, [%r426+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r331, %r332, %r333, %r334}, [%r426+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r367, %r368, %r369, %r370}, [%r426+12288];
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	shl.b32 	%r427, %r1230, 13;
	add.s32 	%r428, %r14, %r427;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r211, %r212, %r247, %r248}, [%r428+49152];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r213, %r214, %r253, %r254}, [%r428+50176];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r215, %r216, %r259, %r260}, [%r428+51200];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r217, %r218, %r265, %r266}, [%r428+52224];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r219, %r220, %r271, %r272}, [%r428+53248];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r221, %r222, %r277, %r278}, [%r428+54272];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r223, %r224, %r283, %r284}, [%r428+55296];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r225, %r226, %r289, %r290}, [%r428+56320];
	.loc	1 338 36                        // sk06_mlp_down.py:338:36
	mov.b32 	%r239, %r206;
	mov.b32 	%r240, %r206;
	mov.b32 	%r241, %r206;
	mov.b32 	%r242, %r206;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r239, %r240, %r241, %r242 }, { %r207, %r208, %r209, %r210 }, { %r211, %r212 }, { %r239, %r240, %r241, %r242 };
	// end inline asm
	mov.b32 	%r249, %r206;
	mov.b32 	%r250, %r206;
	mov.b32 	%r251, %r206;
	mov.b32 	%r252, %r206;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r249, %r250, %r251, %r252 }, { %r207, %r208, %r209, %r210 }, { %r213, %r214 }, { %r249, %r250, %r251, %r252 };
	// end inline asm
	mov.b32 	%r255, %r206;
	mov.b32 	%r256, %r206;
	mov.b32 	%r257, %r206;
	mov.b32 	%r258, %r206;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r255, %r256, %r257, %r258 }, { %r207, %r208, %r209, %r210 }, { %r215, %r216 }, { %r255, %r256, %r257, %r258 };
	// end inline asm
	mov.b32 	%r261, %r206;
	mov.b32 	%r262, %r206;
	mov.b32 	%r263, %r206;
	mov.b32 	%r264, %r206;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r261, %r262, %r263, %r264 }, { %r207, %r208, %r209, %r210 }, { %r217, %r218 }, { %r261, %r262, %r263, %r264 };
	// end inline asm
	mov.b32 	%r267, %r206;
	mov.b32 	%r268, %r206;
	mov.b32 	%r269, %r206;
	mov.b32 	%r270, %r206;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r267, %r268, %r269, %r270 }, { %r207, %r208, %r209, %r210 }, { %r219, %r220 }, { %r267, %r268, %r269, %r270 };
	// end inline asm
	mov.b32 	%r273, %r206;
	mov.b32 	%r274, %r206;
	mov.b32 	%r275, %r206;
	mov.b32 	%r276, %r206;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r273, %r274, %r275, %r276 }, { %r207, %r208, %r209, %r210 }, { %r221, %r222 }, { %r273, %r274, %r275, %r276 };
	// end inline asm
	mov.b32 	%r279, %r206;
	mov.b32 	%r280, %r206;
	mov.b32 	%r281, %r206;
	mov.b32 	%r282, %r206;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r279, %r280, %r281, %r282 }, { %r207, %r208, %r209, %r210 }, { %r223, %r224 }, { %r279, %r280, %r281, %r282 };
	// end inline asm
	mov.b32 	%r285, %r206;
	mov.b32 	%r286, %r206;
	mov.b32 	%r287, %r206;
	mov.b32 	%r288, %r206;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r285, %r286, %r287, %r288 }, { %r207, %r208, %r209, %r210 }, { %r225, %r226 }, { %r285, %r286, %r287, %r288 };
	// end inline asm
	mov.b32 	%r291, %r206;
	mov.b32 	%r292, %r206;
	mov.b32 	%r293, %r206;
	mov.b32 	%r294, %r206;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r291, %r292, %r293, %r294 }, { %r227, %r228, %r229, %r230 }, { %r211, %r212 }, { %r291, %r292, %r293, %r294 };
	// end inline asm
	mov.b32 	%r299, %r206;
	mov.b32 	%r300, %r206;
	mov.b32 	%r301, %r206;
	mov.b32 	%r302, %r206;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r299, %r300, %r301, %r302 }, { %r227, %r228, %r229, %r230 }, { %r213, %r214 }, { %r299, %r300, %r301, %r302 };
	// end inline asm
	mov.b32 	%r303, %r206;
	mov.b32 	%r304, %r206;
	mov.b32 	%r305, %r206;
	mov.b32 	%r306, %r206;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r303, %r304, %r305, %r306 }, { %r227, %r228, %r229, %r230 }, { %r215, %r216 }, { %r303, %r304, %r305, %r306 };
	// end inline asm
	mov.b32 	%r307, %r206;
	mov.b32 	%r308, %r206;
	mov.b32 	%r309, %r206;
	mov.b32 	%r310, %r206;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r307, %r308, %r309, %r310 }, { %r227, %r228, %r229, %r230 }, { %r217, %r218 }, { %r307, %r308, %r309, %r310 };
	// end inline asm
	mov.b32 	%r311, %r206;
	mov.b32 	%r312, %r206;
	mov.b32 	%r313, %r206;
	mov.b32 	%r314, %r206;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r311, %r312, %r313, %r314 }, { %r227, %r228, %r229, %r230 }, { %r219, %r220 }, { %r311, %r312, %r313, %r314 };
	// end inline asm
	mov.b32 	%r315, %r206;
	mov.b32 	%r316, %r206;
	mov.b32 	%r317, %r206;
	mov.b32 	%r318, %r206;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r315, %r316, %r317, %r318 }, { %r227, %r228, %r229, %r230 }, { %r221, %r222 }, { %r315, %r316, %r317, %r318 };
	// end inline asm
	mov.b32 	%r319, %r206;
	mov.b32 	%r320, %r206;
	mov.b32 	%r321, %r206;
	mov.b32 	%r322, %r206;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r319, %r320, %r321, %r322 }, { %r227, %r228, %r229, %r230 }, { %r223, %r224 }, { %r319, %r320, %r321, %r322 };
	// end inline asm
	mov.b32 	%r323, %r206;
	mov.b32 	%r324, %r206;
	mov.b32 	%r325, %r206;
	mov.b32 	%r326, %r206;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r323, %r324, %r325, %r326 }, { %r227, %r228, %r229, %r230 }, { %r225, %r226 }, { %r323, %r324, %r325, %r326 };
	// end inline asm
	mov.b32 	%r327, %r206;
	mov.b32 	%r328, %r206;
	mov.b32 	%r329, %r206;
	mov.b32 	%r330, %r206;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r327, %r328, %r329, %r330 }, { %r231, %r232, %r233, %r234 }, { %r211, %r212 }, { %r327, %r328, %r329, %r330 };
	// end inline asm
	mov.b32 	%r335, %r206;
	mov.b32 	%r336, %r206;
	mov.b32 	%r337, %r206;
	mov.b32 	%r338, %r206;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r335, %r336, %r337, %r338 }, { %r231, %r232, %r233, %r234 }, { %r213, %r214 }, { %r335, %r336, %r337, %r338 };
	// end inline asm
	mov.b32 	%r339, %r206;
	mov.b32 	%r340, %r206;
	mov.b32 	%r341, %r206;
	mov.b32 	%r342, %r206;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r339, %r340, %r341, %r342 }, { %r231, %r232, %r233, %r234 }, { %r215, %r216 }, { %r339, %r340, %r341, %r342 };
	// end inline asm
	mov.b32 	%r343, %r206;
	mov.b32 	%r344, %r206;
	mov.b32 	%r345, %r206;
	mov.b32 	%r346, %r206;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r343, %r344, %r345, %r346 }, { %r231, %r232, %r233, %r234 }, { %r217, %r218 }, { %r343, %r344, %r345, %r346 };
	// end inline asm
	mov.b32 	%r347, %r206;
	mov.b32 	%r348, %r206;
	mov.b32 	%r349, %r206;
	mov.b32 	%r350, %r206;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r347, %r348, %r349, %r350 }, { %r231, %r232, %r233, %r234 }, { %r219, %r220 }, { %r347, %r348, %r349, %r350 };
	// end inline asm
	mov.b32 	%r351, %r206;
	mov.b32 	%r352, %r206;
	mov.b32 	%r353, %r206;
	mov.b32 	%r354, %r206;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r351, %r352, %r353, %r354 }, { %r231, %r232, %r233, %r234 }, { %r221, %r222 }, { %r351, %r352, %r353, %r354 };
	// end inline asm
	mov.b32 	%r355, %r206;
	mov.b32 	%r356, %r206;
	mov.b32 	%r357, %r206;
	mov.b32 	%r358, %r206;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r355, %r356, %r357, %r358 }, { %r231, %r232, %r233, %r234 }, { %r223, %r224 }, { %r355, %r356, %r357, %r358 };
	// end inline asm
	mov.b32 	%r359, %r206;
	mov.b32 	%r360, %r206;
	mov.b32 	%r361, %r206;
	mov.b32 	%r362, %r206;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r359, %r360, %r361, %r362 }, { %r231, %r232, %r233, %r234 }, { %r225, %r226 }, { %r359, %r360, %r361, %r362 };
	// end inline asm
	mov.b32 	%r363, %r206;
	mov.b32 	%r364, %r206;
	mov.b32 	%r365, %r206;
	mov.b32 	%r366, %r206;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r363, %r364, %r365, %r366 }, { %r235, %r236, %r237, %r238 }, { %r211, %r212 }, { %r363, %r364, %r365, %r366 };
	// end inline asm
	mov.b32 	%r371, %r206;
	mov.b32 	%r372, %r206;
	mov.b32 	%r373, %r206;
	mov.b32 	%r374, %r206;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r371, %r372, %r373, %r374 }, { %r235, %r236, %r237, %r238 }, { %r213, %r214 }, { %r371, %r372, %r373, %r374 };
	// end inline asm
	mov.b32 	%r375, %r206;
	mov.b32 	%r376, %r206;
	mov.b32 	%r377, %r206;
	mov.b32 	%r378, %r206;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r375, %r376, %r377, %r378 }, { %r235, %r236, %r237, %r238 }, { %r215, %r216 }, { %r375, %r376, %r377, %r378 };
	// end inline asm
	mov.b32 	%r379, %r206;
	mov.b32 	%r380, %r206;
	mov.b32 	%r381, %r206;
	mov.b32 	%r382, %r206;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r379, %r380, %r381, %r382 }, { %r235, %r236, %r237, %r238 }, { %r217, %r218 }, { %r379, %r380, %r381, %r382 };
	// end inline asm
	mov.b32 	%r383, %r206;
	mov.b32 	%r384, %r206;
	mov.b32 	%r385, %r206;
	mov.b32 	%r386, %r206;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r383, %r384, %r385, %r386 }, { %r235, %r236, %r237, %r238 }, { %r219, %r220 }, { %r383, %r384, %r385, %r386 };
	// end inline asm
	mov.b32 	%r387, %r206;
	mov.b32 	%r388, %r206;
	mov.b32 	%r389, %r206;
	mov.b32 	%r390, %r206;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r387, %r388, %r389, %r390 }, { %r235, %r236, %r237, %r238 }, { %r221, %r222 }, { %r387, %r388, %r389, %r390 };
	// end inline asm
	mov.b32 	%r391, %r206;
	mov.b32 	%r392, %r206;
	mov.b32 	%r393, %r206;
	mov.b32 	%r394, %r206;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r391, %r392, %r393, %r394 }, { %r235, %r236, %r237, %r238 }, { %r223, %r224 }, { %r391, %r392, %r393, %r394 };
	// end inline asm
	mov.b32 	%r395, %r206;
	mov.b32 	%r396, %r206;
	mov.b32 	%r397, %r206;
	mov.b32 	%r398, %r206;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r395, %r396, %r397, %r398 }, { %r235, %r236, %r237, %r238 }, { %r225, %r226 }, { %r395, %r396, %r397, %r398 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r239, %r240, %r241, %r242 }, { %r243, %r244, %r245, %r246 }, { %r247, %r248 }, { %r239, %r240, %r241, %r242 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r249, %r250, %r251, %r252 }, { %r243, %r244, %r245, %r246 }, { %r253, %r254 }, { %r249, %r250, %r251, %r252 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r255, %r256, %r257, %r258 }, { %r243, %r244, %r245, %r246 }, { %r259, %r260 }, { %r255, %r256, %r257, %r258 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r261, %r262, %r263, %r264 }, { %r243, %r244, %r245, %r246 }, { %r265, %r266 }, { %r261, %r262, %r263, %r264 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r267, %r268, %r269, %r270 }, { %r243, %r244, %r245, %r246 }, { %r271, %r272 }, { %r267, %r268, %r269, %r270 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r273, %r274, %r275, %r276 }, { %r243, %r244, %r245, %r246 }, { %r277, %r278 }, { %r273, %r274, %r275, %r276 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r279, %r280, %r281, %r282 }, { %r243, %r244, %r245, %r246 }, { %r283, %r284 }, { %r279, %r280, %r281, %r282 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r285, %r286, %r287, %r288 }, { %r243, %r244, %r245, %r246 }, { %r289, %r290 }, { %r285, %r286, %r287, %r288 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r291, %r292, %r293, %r294 }, { %r295, %r296, %r297, %r298 }, { %r247, %r248 }, { %r291, %r292, %r293, %r294 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r299, %r300, %r301, %r302 }, { %r295, %r296, %r297, %r298 }, { %r253, %r254 }, { %r299, %r300, %r301, %r302 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r303, %r304, %r305, %r306 }, { %r295, %r296, %r297, %r298 }, { %r259, %r260 }, { %r303, %r304, %r305, %r306 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r307, %r308, %r309, %r310 }, { %r295, %r296, %r297, %r298 }, { %r265, %r266 }, { %r307, %r308, %r309, %r310 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r311, %r312, %r313, %r314 }, { %r295, %r296, %r297, %r298 }, { %r271, %r272 }, { %r311, %r312, %r313, %r314 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r315, %r316, %r317, %r318 }, { %r295, %r296, %r297, %r298 }, { %r277, %r278 }, { %r315, %r316, %r317, %r318 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r319, %r320, %r321, %r322 }, { %r295, %r296, %r297, %r298 }, { %r283, %r284 }, { %r319, %r320, %r321, %r322 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r323, %r324, %r325, %r326 }, { %r295, %r296, %r297, %r298 }, { %r289, %r290 }, { %r323, %r324, %r325, %r326 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r327, %r328, %r329, %r330 }, { %r331, %r332, %r333, %r334 }, { %r247, %r248 }, { %r327, %r328, %r329, %r330 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r335, %r336, %r337, %r338 }, { %r331, %r332, %r333, %r334 }, { %r253, %r254 }, { %r335, %r336, %r337, %r338 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r339, %r340, %r341, %r342 }, { %r331, %r332, %r333, %r334 }, { %r259, %r260 }, { %r339, %r340, %r341, %r342 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r343, %r344, %r345, %r346 }, { %r331, %r332, %r333, %r334 }, { %r265, %r266 }, { %r343, %r344, %r345, %r346 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r347, %r348, %r349, %r350 }, { %r331, %r332, %r333, %r334 }, { %r271, %r272 }, { %r347, %r348, %r349, %r350 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r351, %r352, %r353, %r354 }, { %r331, %r332, %r333, %r334 }, { %r277, %r278 }, { %r351, %r352, %r353, %r354 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r355, %r356, %r357, %r358 }, { %r331, %r332, %r333, %r334 }, { %r283, %r284 }, { %r355, %r356, %r357, %r358 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r359, %r360, %r361, %r362 }, { %r331, %r332, %r333, %r334 }, { %r289, %r290 }, { %r359, %r360, %r361, %r362 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r363, %r364, %r365, %r366 }, { %r367, %r368, %r369, %r370 }, { %r247, %r248 }, { %r363, %r364, %r365, %r366 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r371, %r372, %r373, %r374 }, { %r367, %r368, %r369, %r370 }, { %r253, %r254 }, { %r371, %r372, %r373, %r374 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r375, %r376, %r377, %r378 }, { %r367, %r368, %r369, %r370 }, { %r259, %r260 }, { %r375, %r376, %r377, %r378 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r379, %r380, %r381, %r382 }, { %r367, %r368, %r369, %r370 }, { %r265, %r266 }, { %r379, %r380, %r381, %r382 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r383, %r384, %r385, %r386 }, { %r367, %r368, %r369, %r370 }, { %r271, %r272 }, { %r383, %r384, %r385, %r386 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r387, %r388, %r389, %r390 }, { %r367, %r368, %r369, %r370 }, { %r277, %r278 }, { %r387, %r388, %r389, %r390 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r391, %r392, %r393, %r394 }, { %r367, %r368, %r369, %r370 }, { %r283, %r284 }, { %r391, %r392, %r393, %r394 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r395, %r396, %r397, %r398 }, { %r367, %r368, %r369, %r370 }, { %r289, %r290 }, { %r395, %r396, %r397, %r398 };
	// end inline asm
	.loc	1 340 54                        // sk06_mlp_down.py:340:54
	bfe.u32 	%r429, %r1232, 1, 25;
	.loc	1 340 69                        // sk06_mlp_down.py:340:69
	mul.lo.s32 	%r430, %r429, %r19;
	.loc	1 340 37                        // sk06_mlp_down.py:340:37
	mul.wide.s32 	%rd81, %r430, 4;
	add.s64 	%rd59, %rd8, %rd81;
	add.s64 	%rd60, %rd9, %rd81;
	add.s64 	%rd61, %rd10, %rd81;
	add.s64 	%rd62, %rd11, %rd81;
	add.s64 	%rd63, %rd12, %rd81;
	add.s64 	%rd64, %rd13, %rd81;
	add.s64 	%rd65, %rd14, %rd81;
	add.s64 	%rd66, %rd15, %rd81;
	add.s64 	%rd67, %rd16, %rd81;
	add.s64 	%rd68, %rd17, %rd81;
	add.s64 	%rd69, %rd18, %rd81;
	add.s64 	%rd70, %rd19, %rd81;
	add.s64 	%rd71, %rd20, %rd81;
	add.s64 	%rd72, %rd21, %rd81;
	add.s64 	%rd73, %rd22, %rd81;
	add.s64 	%rd74, %rd23, %rd81;
	.loc	1 340 27                        // sk06_mlp_down.py:340:27
	// begin inline asm
	mov.u32 %r399, 0x0;
	ld.global.b32 { %r399 }, [ %rd59 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r400, 0x0;
	ld.global.b32 { %r400 }, [ %rd60 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r401, 0x0;
	ld.global.b32 { %r401 }, [ %rd61 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r402, 0x0;
	ld.global.b32 { %r402 }, [ %rd62 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r403, 0x0;
	ld.global.b32 { %r403 }, [ %rd63 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r404, 0x0;
	ld.global.b32 { %r404 }, [ %rd64 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r405, 0x0;
	ld.global.b32 { %r405 }, [ %rd65 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r406, 0x0;
	ld.global.b32 { %r406 }, [ %rd66 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r407, 0x0;
	ld.global.b32 { %r407 }, [ %rd67 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r408, 0x0;
	ld.global.b32 { %r408 }, [ %rd68 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r409, 0x0;
	ld.global.b32 { %r409 }, [ %rd69 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r410, 0x0;
	ld.global.b32 { %r410 }, [ %rd70 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r411, 0x0;
	ld.global.b32 { %r411 }, [ %rd71 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r412, 0x0;
	ld.global.b32 { %r412 }, [ %rd72 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r413, 0x0;
	ld.global.b32 { %r413 }, [ %rd73 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r414, 0x0;
	ld.global.b32 { %r414 }, [ %rd74 + 0 ];
	// end inline asm
	.loc	1 341 23                        // sk06_mlp_down.py:341:23
	cvt.rn.f32.s32 	%r431, %r398;
	cvt.rn.f32.s32 	%r432, %r397;
	cvt.rn.f32.s32 	%r433, %r396;
	cvt.rn.f32.s32 	%r434, %r395;
	cvt.rn.f32.s32 	%r435, %r394;
	cvt.rn.f32.s32 	%r436, %r393;
	cvt.rn.f32.s32 	%r437, %r392;
	cvt.rn.f32.s32 	%r438, %r391;
	cvt.rn.f32.s32 	%r439, %r390;
	cvt.rn.f32.s32 	%r440, %r389;
	cvt.rn.f32.s32 	%r441, %r388;
	cvt.rn.f32.s32 	%r442, %r387;
	cvt.rn.f32.s32 	%r443, %r386;
	cvt.rn.f32.s32 	%r444, %r385;
	cvt.rn.f32.s32 	%r445, %r384;
	cvt.rn.f32.s32 	%r446, %r383;
	cvt.rn.f32.s32 	%r447, %r382;
	cvt.rn.f32.s32 	%r448, %r381;
	cvt.rn.f32.s32 	%r449, %r380;
	cvt.rn.f32.s32 	%r450, %r379;
	cvt.rn.f32.s32 	%r451, %r378;
	cvt.rn.f32.s32 	%r452, %r377;
	cvt.rn.f32.s32 	%r453, %r376;
	cvt.rn.f32.s32 	%r454, %r375;
	cvt.rn.f32.s32 	%r455, %r374;
	cvt.rn.f32.s32 	%r456, %r286;
	cvt.rn.f32.s32 	%r457, %r288;
	cvt.rn.f32.s32 	%r458, %r324;
	cvt.rn.f32.s32 	%r459, %r326;
	cvt.rn.f32.s32 	%r460, %r360;
	cvt.rn.f32.s32 	%r461, %r362;
	cvt.rn.f32.s32 	%r462, %r373;
	cvt.rn.f32.s32 	%r463, %r285;
	cvt.rn.f32.s32 	%r464, %r287;
	cvt.rn.f32.s32 	%r465, %r323;
	cvt.rn.f32.s32 	%r466, %r325;
	cvt.rn.f32.s32 	%r467, %r359;
	cvt.rn.f32.s32 	%r468, %r361;
	cvt.rn.f32.s32 	%r469, %r372;
	cvt.rn.f32.s32 	%r470, %r280;
	cvt.rn.f32.s32 	%r471, %r282;
	cvt.rn.f32.s32 	%r472, %r320;
	cvt.rn.f32.s32 	%r473, %r322;
	cvt.rn.f32.s32 	%r474, %r356;
	cvt.rn.f32.s32 	%r475, %r358;
	cvt.rn.f32.s32 	%r476, %r371;
	cvt.rn.f32.s32 	%r477, %r279;
	cvt.rn.f32.s32 	%r478, %r281;
	cvt.rn.f32.s32 	%r479, %r319;
	cvt.rn.f32.s32 	%r480, %r321;
	cvt.rn.f32.s32 	%r481, %r355;
	cvt.rn.f32.s32 	%r482, %r357;
	cvt.rn.f32.s32 	%r483, %r366;
	cvt.rn.f32.s32 	%r484, %r274;
	cvt.rn.f32.s32 	%r485, %r276;
	cvt.rn.f32.s32 	%r486, %r316;
	cvt.rn.f32.s32 	%r487, %r318;
	cvt.rn.f32.s32 	%r488, %r352;
	cvt.rn.f32.s32 	%r489, %r354;
	cvt.rn.f32.s32 	%r490, %r365;
	cvt.rn.f32.s32 	%r491, %r273;
	cvt.rn.f32.s32 	%r492, %r275;
	cvt.rn.f32.s32 	%r493, %r315;
	cvt.rn.f32.s32 	%r494, %r317;
	cvt.rn.f32.s32 	%r495, %r351;
	cvt.rn.f32.s32 	%r496, %r353;
	cvt.rn.f32.s32 	%r497, %r364;
	cvt.rn.f32.s32 	%r498, %r268;
	cvt.rn.f32.s32 	%r499, %r270;
	cvt.rn.f32.s32 	%r500, %r312;
	cvt.rn.f32.s32 	%r501, %r314;
	cvt.rn.f32.s32 	%r502, %r348;
	cvt.rn.f32.s32 	%r503, %r350;
	cvt.rn.f32.s32 	%r504, %r363;
	cvt.rn.f32.s32 	%r505, %r267;
	cvt.rn.f32.s32 	%r506, %r269;
	cvt.rn.f32.s32 	%r507, %r311;
	cvt.rn.f32.s32 	%r508, %r313;
	cvt.rn.f32.s32 	%r509, %r347;
	cvt.rn.f32.s32 	%r510, %r349;
	cvt.rn.f32.s32 	%r511, %r262;
	cvt.rn.f32.s32 	%r512, %r264;
	cvt.rn.f32.s32 	%r513, %r308;
	cvt.rn.f32.s32 	%r514, %r310;
	cvt.rn.f32.s32 	%r515, %r344;
	cvt.rn.f32.s32 	%r516, %r346;
	cvt.rn.f32.s32 	%r517, %r261;
	cvt.rn.f32.s32 	%r518, %r263;
	cvt.rn.f32.s32 	%r519, %r307;
	cvt.rn.f32.s32 	%r520, %r309;
	cvt.rn.f32.s32 	%r521, %r343;
	cvt.rn.f32.s32 	%r522, %r345;
	cvt.rn.f32.s32 	%r523, %r256;
	cvt.rn.f32.s32 	%r524, %r258;
	cvt.rn.f32.s32 	%r525, %r304;
	cvt.rn.f32.s32 	%r526, %r306;
	cvt.rn.f32.s32 	%r527, %r340;
	cvt.rn.f32.s32 	%r528, %r342;
	cvt.rn.f32.s32 	%r529, %r255;
	cvt.rn.f32.s32 	%r530, %r257;
	cvt.rn.f32.s32 	%r531, %r303;
	cvt.rn.f32.s32 	%r532, %r305;
	cvt.rn.f32.s32 	%r533, %r339;
	cvt.rn.f32.s32 	%r534, %r341;
	cvt.rn.f32.s32 	%r535, %r250;
	cvt.rn.f32.s32 	%r536, %r252;
	cvt.rn.f32.s32 	%r537, %r300;
	cvt.rn.f32.s32 	%r538, %r302;
	cvt.rn.f32.s32 	%r539, %r336;
	cvt.rn.f32.s32 	%r540, %r338;
	cvt.rn.f32.s32 	%r541, %r249;
	cvt.rn.f32.s32 	%r542, %r251;
	cvt.rn.f32.s32 	%r543, %r299;
	cvt.rn.f32.s32 	%r544, %r301;
	cvt.rn.f32.s32 	%r545, %r335;
	cvt.rn.f32.s32 	%r546, %r337;
	cvt.rn.f32.s32 	%r547, %r240;
	cvt.rn.f32.s32 	%r548, %r242;
	cvt.rn.f32.s32 	%r549, %r292;
	cvt.rn.f32.s32 	%r550, %r294;
	cvt.rn.f32.s32 	%r551, %r328;
	cvt.rn.f32.s32 	%r552, %r330;
	cvt.rn.f32.s32 	%r553, %r239;
	cvt.rn.f32.s32 	%r554, %r241;
	cvt.rn.f32.s32 	%r555, %r291;
	cvt.rn.f32.s32 	%r556, %r293;
	cvt.rn.f32.s32 	%r557, %r327;
	cvt.rn.f32.s32 	%r558, %r329;
	.loc	1 341 19                        // sk06_mlp_down.py:341:19
	fma.rn.f32 	%r1301, %r399, %r558, %r1301;
	fma.rn.f32 	%r1299, %r399, %r557, %r1299;
	fma.rn.f32 	%r1269, %r399, %r556, %r1269;
	fma.rn.f32 	%r1267, %r399, %r555, %r1267;
	fma.rn.f32 	%r1237, %r399, %r554, %r1237;
	fma.rn.f32 	%r1235, %r399, %r553, %r1235;
	fma.rn.f32 	%r1302, %r400, %r552, %r1302;
	fma.rn.f32 	%r1300, %r400, %r551, %r1300;
	fma.rn.f32 	%r1270, %r400, %r550, %r1270;
	fma.rn.f32 	%r1268, %r400, %r549, %r1268;
	fma.rn.f32 	%r1238, %r400, %r548, %r1238;
	fma.rn.f32 	%r1236, %r400, %r547, %r1236;
	fma.rn.f32 	%r1305, %r401, %r546, %r1305;
	fma.rn.f32 	%r1303, %r401, %r545, %r1303;
	fma.rn.f32 	%r1273, %r401, %r544, %r1273;
	fma.rn.f32 	%r1271, %r401, %r543, %r1271;
	fma.rn.f32 	%r1241, %r401, %r542, %r1241;
	fma.rn.f32 	%r1239, %r401, %r541, %r1239;
	fma.rn.f32 	%r1306, %r402, %r540, %r1306;
	fma.rn.f32 	%r1304, %r402, %r539, %r1304;
	fma.rn.f32 	%r1274, %r402, %r538, %r1274;
	fma.rn.f32 	%r1272, %r402, %r537, %r1272;
	fma.rn.f32 	%r1242, %r402, %r536, %r1242;
	fma.rn.f32 	%r1240, %r402, %r535, %r1240;
	fma.rn.f32 	%r1309, %r403, %r534, %r1309;
	fma.rn.f32 	%r1307, %r403, %r533, %r1307;
	fma.rn.f32 	%r1277, %r403, %r532, %r1277;
	fma.rn.f32 	%r1275, %r403, %r531, %r1275;
	fma.rn.f32 	%r1245, %r403, %r530, %r1245;
	fma.rn.f32 	%r1243, %r403, %r529, %r1243;
	fma.rn.f32 	%r1310, %r404, %r528, %r1310;
	fma.rn.f32 	%r1308, %r404, %r527, %r1308;
	fma.rn.f32 	%r1278, %r404, %r526, %r1278;
	fma.rn.f32 	%r1276, %r404, %r525, %r1276;
	fma.rn.f32 	%r1246, %r404, %r524, %r1246;
	fma.rn.f32 	%r1244, %r404, %r523, %r1244;
	fma.rn.f32 	%r1313, %r405, %r522, %r1313;
	fma.rn.f32 	%r1311, %r405, %r521, %r1311;
	fma.rn.f32 	%r1281, %r405, %r520, %r1281;
	fma.rn.f32 	%r1279, %r405, %r519, %r1279;
	fma.rn.f32 	%r1249, %r405, %r518, %r1249;
	fma.rn.f32 	%r1247, %r405, %r517, %r1247;
	fma.rn.f32 	%r1314, %r406, %r516, %r1314;
	fma.rn.f32 	%r1312, %r406, %r515, %r1312;
	fma.rn.f32 	%r1282, %r406, %r514, %r1282;
	fma.rn.f32 	%r1280, %r406, %r513, %r1280;
	fma.rn.f32 	%r1250, %r406, %r512, %r1250;
	fma.rn.f32 	%r1248, %r406, %r511, %r1248;
	fma.rn.f32 	%r1317, %r407, %r510, %r1317;
	fma.rn.f32 	%r1315, %r407, %r509, %r1315;
	fma.rn.f32 	%r1285, %r407, %r508, %r1285;
	fma.rn.f32 	%r1283, %r407, %r507, %r1283;
	fma.rn.f32 	%r1253, %r407, %r506, %r1253;
	fma.rn.f32 	%r1251, %r407, %r505, %r1251;
	fma.rn.f32 	%r1331, %r399, %r504, %r1331;
	fma.rn.f32 	%r1318, %r408, %r503, %r1318;
	fma.rn.f32 	%r1316, %r408, %r502, %r1316;
	fma.rn.f32 	%r1286, %r408, %r501, %r1286;
	fma.rn.f32 	%r1284, %r408, %r500, %r1284;
	fma.rn.f32 	%r1254, %r408, %r499, %r1254;
	fma.rn.f32 	%r1252, %r408, %r498, %r1252;
	fma.rn.f32 	%r1332, %r400, %r497, %r1332;
	fma.rn.f32 	%r1321, %r409, %r496, %r1321;
	fma.rn.f32 	%r1319, %r409, %r495, %r1319;
	fma.rn.f32 	%r1289, %r409, %r494, %r1289;
	fma.rn.f32 	%r1287, %r409, %r493, %r1287;
	fma.rn.f32 	%r1257, %r409, %r492, %r1257;
	fma.rn.f32 	%r1255, %r409, %r491, %r1255;
	fma.rn.f32 	%r1333, %r399, %r490, %r1333;
	fma.rn.f32 	%r1322, %r410, %r489, %r1322;
	fma.rn.f32 	%r1320, %r410, %r488, %r1320;
	fma.rn.f32 	%r1290, %r410, %r487, %r1290;
	fma.rn.f32 	%r1288, %r410, %r486, %r1288;
	fma.rn.f32 	%r1258, %r410, %r485, %r1258;
	fma.rn.f32 	%r1256, %r410, %r484, %r1256;
	fma.rn.f32 	%r1334, %r400, %r483, %r1334;
	fma.rn.f32 	%r1325, %r411, %r482, %r1325;
	fma.rn.f32 	%r1323, %r411, %r481, %r1323;
	fma.rn.f32 	%r1293, %r411, %r480, %r1293;
	fma.rn.f32 	%r1291, %r411, %r479, %r1291;
	fma.rn.f32 	%r1261, %r411, %r478, %r1261;
	fma.rn.f32 	%r1259, %r411, %r477, %r1259;
	fma.rn.f32 	%r1335, %r401, %r476, %r1335;
	fma.rn.f32 	%r1326, %r412, %r475, %r1326;
	fma.rn.f32 	%r1324, %r412, %r474, %r1324;
	fma.rn.f32 	%r1294, %r412, %r473, %r1294;
	fma.rn.f32 	%r1292, %r412, %r472, %r1292;
	fma.rn.f32 	%r1262, %r412, %r471, %r1262;
	fma.rn.f32 	%r1260, %r412, %r470, %r1260;
	fma.rn.f32 	%r1336, %r402, %r469, %r1336;
	fma.rn.f32 	%r1329, %r413, %r468, %r1329;
	fma.rn.f32 	%r1327, %r413, %r467, %r1327;
	fma.rn.f32 	%r1297, %r413, %r466, %r1297;
	fma.rn.f32 	%r1295, %r413, %r465, %r1295;
	fma.rn.f32 	%r1265, %r413, %r464, %r1265;
	fma.rn.f32 	%r1263, %r413, %r463, %r1263;
	fma.rn.f32 	%r1337, %r401, %r462, %r1337;
	fma.rn.f32 	%r1330, %r414, %r461, %r1330;
	fma.rn.f32 	%r1328, %r414, %r460, %r1328;
	fma.rn.f32 	%r1298, %r414, %r459, %r1298;
	fma.rn.f32 	%r1296, %r414, %r458, %r1296;
	fma.rn.f32 	%r1266, %r414, %r457, %r1266;
	fma.rn.f32 	%r1264, %r414, %r456, %r1264;
	fma.rn.f32 	%r1338, %r402, %r455, %r1338;
	fma.rn.f32 	%r1339, %r403, %r454, %r1339;
	fma.rn.f32 	%r1340, %r404, %r453, %r1340;
	fma.rn.f32 	%r1341, %r403, %r452, %r1341;
	fma.rn.f32 	%r1342, %r404, %r451, %r1342;
	fma.rn.f32 	%r1343, %r405, %r450, %r1343;
	fma.rn.f32 	%r1344, %r406, %r449, %r1344;
	fma.rn.f32 	%r1345, %r405, %r448, %r1345;
	fma.rn.f32 	%r1346, %r406, %r447, %r1346;
	fma.rn.f32 	%r1347, %r407, %r446, %r1347;
	fma.rn.f32 	%r1348, %r408, %r445, %r1348;
	fma.rn.f32 	%r1349, %r407, %r444, %r1349;
	fma.rn.f32 	%r1350, %r408, %r443, %r1350;
	fma.rn.f32 	%r1351, %r409, %r442, %r1351;
	fma.rn.f32 	%r1352, %r410, %r441, %r1352;
	fma.rn.f32 	%r1353, %r409, %r440, %r1353;
	fma.rn.f32 	%r1354, %r410, %r439, %r1354;
	fma.rn.f32 	%r1355, %r411, %r438, %r1355;
	fma.rn.f32 	%r1356, %r412, %r437, %r1356;
	fma.rn.f32 	%r1357, %r411, %r436, %r1357;
	fma.rn.f32 	%r1358, %r412, %r435, %r1358;
	fma.rn.f32 	%r1359, %r413, %r434, %r1359;
	fma.rn.f32 	%r1360, %r414, %r433, %r1360;
	fma.rn.f32 	%r1361, %r413, %r432, %r1361;
	fma.rn.f32 	%r1362, %r414, %r431, %r1362;
	.loc	1 344 18                        // sk06_mlp_down.py:344:18
	add.s64 	%rd75, %rd149, %rd5;
	add.s64 	%rd76, %rd150, %rd5;
	add.s64 	%rd77, %rd151, %rd5;
	.loc	1 345 18                        // sk06_mlp_down.py:345:18
	add.s64 	%rd78, %rd152, %rd5;
	add.s64 	%rd79, %rd153, %rd5;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	add.s64 	%rd80, %rd154, %rd5;
	add.s32 	%r559, %r1231, 1;
	setp.gt.s32 	%p6, %r559, 2;
	selp.b32 	%r1231, 0, %r559, %p6;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	shl.b32 	%r560, %r1231, 14;
	bar.sync 	0;
	add.s32 	%r415, %r20, %r560;
	selp.b32 	%r416, 16, 0, %p4;
	// begin inline asm
	cp.async.cg.shared.global [ %r415 + 0 ], [ %rd75 + 0 ], 0x10, %r416;
	// end inline asm
	add.s32 	%r417, %r415, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r417 + 0 ], [ %rd76 + 0 ], 0x10, %r416;
	// end inline asm
	add.s32 	%r418, %r415, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r418 + 0 ], [ %rd77 + 0 ], 0x10, %r416;
	// end inline asm
	add.s32 	%r419, %r415, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r419 + 0 ], [ %rd78 + 0 ], 0x10, %r416;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	shl.b32 	%r561, %r1231, 13;
	add.s32 	%r562, %r20, %r561;
	add.s32 	%r420, %r562, 49152;
	// begin inline asm
	cp.async.cg.shared.global [ %r420 + 0 ], [ %rd79 + 0 ], 0x10, %r416;
	// end inline asm
	add.s32 	%r421, %r562, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r421 + 0 ], [ %rd80 + 0 ], 0x10, %r416;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	add.s32 	%r1232, %r1232, 1;
	add.s64 	%rd154, %rd154, 64;
	add.s64 	%rd153, %rd153, 64;
	add.s64 	%rd152, %rd152, 64;
	add.s64 	%rd151, %rd151, 64;
	add.s64 	%rd150, %rd150, 64;
	add.s64 	%rd149, %rd149, 64;
	setp.ne.b32 	%p7, %r8, %r1232;
	@%p7 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_4;
$L__BB0_1:                              // %.._crit_edge_crit_edge
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	and.b32 	%r1234, %r2, 16;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	shl.b32 	%r1233, %r2, 4;
	mov.b32 	%r1235, 0f00000000;
	mov.b32 	%r1236, %r1235;
	mov.b32 	%r1237, %r1235;
	mov.b32 	%r1238, %r1235;
	mov.b32 	%r1239, %r1235;
	mov.b32 	%r1240, %r1235;
	mov.b32 	%r1241, %r1235;
	mov.b32 	%r1242, %r1235;
	mov.b32 	%r1243, %r1235;
	mov.b32 	%r1244, %r1235;
	mov.b32 	%r1245, %r1235;
	mov.b32 	%r1246, %r1235;
	mov.b32 	%r1247, %r1235;
	mov.b32 	%r1248, %r1235;
	mov.b32 	%r1249, %r1235;
	mov.b32 	%r1250, %r1235;
	mov.b32 	%r1251, %r1235;
	mov.b32 	%r1252, %r1235;
	mov.b32 	%r1253, %r1235;
	mov.b32 	%r1254, %r1235;
	mov.b32 	%r1255, %r1235;
	mov.b32 	%r1256, %r1235;
	mov.b32 	%r1257, %r1235;
	mov.b32 	%r1258, %r1235;
	mov.b32 	%r1259, %r1235;
	mov.b32 	%r1260, %r1235;
	mov.b32 	%r1261, %r1235;
	mov.b32 	%r1262, %r1235;
	mov.b32 	%r1263, %r1235;
	mov.b32 	%r1264, %r1235;
	mov.b32 	%r1265, %r1235;
	mov.b32 	%r1266, %r1235;
	mov.b32 	%r1267, %r1235;
	mov.b32 	%r1268, %r1235;
	mov.b32 	%r1269, %r1235;
	mov.b32 	%r1270, %r1235;
	mov.b32 	%r1271, %r1235;
	mov.b32 	%r1272, %r1235;
	mov.b32 	%r1273, %r1235;
	mov.b32 	%r1274, %r1235;
	mov.b32 	%r1275, %r1235;
	mov.b32 	%r1276, %r1235;
	mov.b32 	%r1277, %r1235;
	mov.b32 	%r1278, %r1235;
	mov.b32 	%r1279, %r1235;
	mov.b32 	%r1280, %r1235;
	mov.b32 	%r1281, %r1235;
	mov.b32 	%r1282, %r1235;
	mov.b32 	%r1283, %r1235;
	mov.b32 	%r1284, %r1235;
	mov.b32 	%r1285, %r1235;
	mov.b32 	%r1286, %r1235;
	mov.b32 	%r1287, %r1235;
	mov.b32 	%r1288, %r1235;
	mov.b32 	%r1289, %r1235;
	mov.b32 	%r1290, %r1235;
	mov.b32 	%r1291, %r1235;
	mov.b32 	%r1292, %r1235;
	mov.b32 	%r1293, %r1235;
	mov.b32 	%r1294, %r1235;
	mov.b32 	%r1295, %r1235;
	mov.b32 	%r1296, %r1235;
	mov.b32 	%r1297, %r1235;
	mov.b32 	%r1298, %r1235;
	mov.b32 	%r1299, %r1235;
	mov.b32 	%r1300, %r1235;
	mov.b32 	%r1301, %r1235;
	mov.b32 	%r1302, %r1235;
	mov.b32 	%r1303, %r1235;
	mov.b32 	%r1304, %r1235;
	mov.b32 	%r1305, %r1235;
	mov.b32 	%r1306, %r1235;
	mov.b32 	%r1307, %r1235;
	mov.b32 	%r1308, %r1235;
	mov.b32 	%r1309, %r1235;
	mov.b32 	%r1310, %r1235;
	mov.b32 	%r1311, %r1235;
	mov.b32 	%r1312, %r1235;
	mov.b32 	%r1313, %r1235;
	mov.b32 	%r1314, %r1235;
	mov.b32 	%r1315, %r1235;
	mov.b32 	%r1316, %r1235;
	mov.b32 	%r1317, %r1235;
	mov.b32 	%r1318, %r1235;
	mov.b32 	%r1319, %r1235;
	mov.b32 	%r1320, %r1235;
	mov.b32 	%r1321, %r1235;
	mov.b32 	%r1322, %r1235;
	mov.b32 	%r1323, %r1235;
	mov.b32 	%r1324, %r1235;
	mov.b32 	%r1325, %r1235;
	mov.b32 	%r1326, %r1235;
	mov.b32 	%r1327, %r1235;
	mov.b32 	%r1328, %r1235;
	mov.b32 	%r1329, %r1235;
	mov.b32 	%r1330, %r1235;
	mov.b32 	%r1331, %r1235;
	mov.b32 	%r1332, %r1235;
	mov.b32 	%r1333, %r1235;
	mov.b32 	%r1334, %r1235;
	mov.b32 	%r1335, %r1235;
	mov.b32 	%r1336, %r1235;
	mov.b32 	%r1337, %r1235;
	mov.b32 	%r1338, %r1235;
	mov.b32 	%r1339, %r1235;
	mov.b32 	%r1340, %r1235;
	mov.b32 	%r1341, %r1235;
	mov.b32 	%r1342, %r1235;
	mov.b32 	%r1343, %r1235;
	mov.b32 	%r1344, %r1235;
	mov.b32 	%r1345, %r1235;
	mov.b32 	%r1346, %r1235;
	mov.b32 	%r1347, %r1235;
	mov.b32 	%r1348, %r1235;
	mov.b32 	%r1349, %r1235;
	mov.b32 	%r1350, %r1235;
	mov.b32 	%r1351, %r1235;
	mov.b32 	%r1352, %r1235;
	mov.b32 	%r1353, %r1235;
	mov.b32 	%r1354, %r1235;
	mov.b32 	%r1355, %r1235;
	mov.b32 	%r1356, %r1235;
	mov.b32 	%r1357, %r1235;
	mov.b32 	%r1358, %r1235;
	mov.b32 	%r1359, %r1235;
	mov.b32 	%r1360, %r1235;
	mov.b32 	%r1361, %r1235;
	mov.b32 	%r1362, %r1235;
$L__BB0_4:                              // %._crit_edge
	.loc	1 323 45                        // sk06_mlp_down.py:323:45
	shl.b32 	%r777, %r7, 3;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r778, %r4, %r777;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r779, %r778, %r16;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r780, %r1, %r3;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r781, %r780, %r15;
	.loc	1 322 45                        // sk06_mlp_down.py:322:45
	and.b32 	%r782, %r2, 240;
	bfe.u32 	%r783, %r2, 4, 4;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r784, %r783, %r1;
	or.b32 	%r785, %r784, 240;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r786, %r785, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r787, %r784, 224;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r788, %r787, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r789, %r784, 208;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r790, %r789, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r791, %r784, 192;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r792, %r791, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r793, %r784, 176;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r794, %r793, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r795, %r784, 160;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r796, %r795, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r797, %r784, 144;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r798, %r797, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r799, %r784, 128;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r800, %r799, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r801, %r784, 112;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r802, %r801, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r803, %r784, 96;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r804, %r803, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r805, %r784, 80;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r806, %r805, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r807, %r784, 64;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r808, %r807, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r809, %r784, 48;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r810, %r809, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r811, %r784, 32;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r812, %r811, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r813, %r784, 16;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r814, %r813, %r15;
	rem.s32 	%r815, %r784, %r15;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 347 38                        // sk06_mlp_down.py:347:38
	mad.wide.s32 	%rd82, %r781, 4, %rd28;
	.loc	1 347 24                        // sk06_mlp_down.py:347:24
	// begin inline asm
	mov.u32 %r564, 0x0;
	ld.global.b32 { %r564 }, [ %rd82 + 0 ];
	// end inline asm
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	and.b32 	%r816, %r2, 7;
	shl.b32 	%r817, %r816, 3;
	shl.b32 	%r818, %r782, 2;
	and.b32 	%r819, %r2, 8;
	shr.u32 	%r820, %r819, 1;
	add.s32 	%r821, %r193, %r817;
	add.s32 	%r822, %r821, %r818;
	add.s32 	%r563, %r822, %r820;
	// begin inline asm
	st.shared.b32 [ %r563 + 0 ], %r564;
	// end inline asm
	bar.sync 	0;
	and.b32 	%r823, %r9, 56;
	and.b32 	%r824, %r2, 192;
	add.s32 	%r825, %r193, %r823;
	add.s32 	%r826, %r825, %r824;
	ld.shared.v2.b32 	{%r827, %r828}, [%r826];
	ld.shared.v2.b32 	{%r829, %r830}, [%r826+256];
	ld.shared.v2.b32 	{%r831, %r832}, [%r826+512];
	ld.shared.v2.b32 	{%r833, %r834}, [%r826+768];
	.loc	1 350 49                        // sk06_mlp_down.py:350:49
	mul.lo.s32 	%r835, %r815, %r18;
	mul.lo.s32 	%r836, %r814, %r18;
	mul.lo.s32 	%r837, %r812, %r18;
	mul.lo.s32 	%r838, %r810, %r18;
	mul.lo.s32 	%r839, %r808, %r18;
	mul.lo.s32 	%r840, %r806, %r18;
	mul.lo.s32 	%r841, %r804, %r18;
	mul.lo.s32 	%r842, %r802, %r18;
	mul.lo.s32 	%r843, %r800, %r18;
	mul.lo.s32 	%r844, %r798, %r18;
	mul.lo.s32 	%r845, %r796, %r18;
	mul.lo.s32 	%r846, %r794, %r18;
	mul.lo.s32 	%r847, %r792, %r18;
	mul.lo.s32 	%r848, %r790, %r18;
	mul.lo.s32 	%r849, %r788, %r18;
	mul.lo.s32 	%r850, %r786, %r18;
	.loc	1 350 31                        // sk06_mlp_down.py:350:31
	mad.wide.s32 	%rd115, %r835, 2, %rd27;
	mad.wide.s32 	%rd116, %r836, 2, %rd27;
	mad.wide.s32 	%rd117, %r837, 2, %rd27;
	mad.wide.s32 	%rd118, %r838, 2, %rd27;
	mad.wide.s32 	%rd119, %r839, 2, %rd27;
	mad.wide.s32 	%rd120, %r840, 2, %rd27;
	mad.wide.s32 	%rd121, %r841, 2, %rd27;
	mad.wide.s32 	%rd122, %r842, 2, %rd27;
	mad.wide.s32 	%rd123, %r843, 2, %rd27;
	mad.wide.s32 	%rd124, %r844, 2, %rd27;
	mad.wide.s32 	%rd125, %r845, 2, %rd27;
	mad.wide.s32 	%rd126, %r846, 2, %rd27;
	mad.wide.s32 	%rd127, %r847, 2, %rd27;
	mad.wide.s32 	%rd128, %r848, 2, %rd27;
	mad.wide.s32 	%rd129, %r849, 2, %rd27;
	mad.wide.s32 	%rd130, %r850, 2, %rd27;
	.loc	1 350 64                        // sk06_mlp_down.py:350:64
	mul.wide.s32 	%rd131, %r779, 2;
	add.s64 	%rd83, %rd115, %rd131;
	add.s64 	%rd84, %rd116, %rd131;
	add.s64 	%rd85, %rd117, %rd131;
	add.s64 	%rd86, %rd118, %rd131;
	add.s64 	%rd87, %rd119, %rd131;
	add.s64 	%rd88, %rd120, %rd131;
	add.s64 	%rd89, %rd121, %rd131;
	add.s64 	%rd90, %rd122, %rd131;
	add.s64 	%rd91, %rd123, %rd131;
	add.s64 	%rd92, %rd124, %rd131;
	add.s64 	%rd93, %rd125, %rd131;
	add.s64 	%rd94, %rd126, %rd131;
	add.s64 	%rd95, %rd127, %rd131;
	add.s64 	%rd96, %rd128, %rd131;
	add.s64 	%rd97, %rd129, %rd131;
	add.s64 	%rd98, %rd130, %rd131;
	.loc	1 350 19                        // sk06_mlp_down.py:350:19
	// begin inline asm
	mov.u32 %r566, 0x0;
	mov.u32 %r567, 0x0;
	mov.u32 %r568, 0x0;
	mov.u32 %r569, 0x0;
	ld.global.v4.b32 { %r566, %r567, %r568, %r569 }, [ %rd83 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r571, 0x0;
	mov.u32 %r572, 0x0;
	mov.u32 %r573, 0x0;
	mov.u32 %r574, 0x0;
	ld.global.v4.b32 { %r571, %r572, %r573, %r574 }, [ %rd84 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r576, 0x0;
	mov.u32 %r577, 0x0;
	mov.u32 %r578, 0x0;
	mov.u32 %r579, 0x0;
	ld.global.v4.b32 { %r576, %r577, %r578, %r579 }, [ %rd85 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r581, 0x0;
	mov.u32 %r582, 0x0;
	mov.u32 %r583, 0x0;
	mov.u32 %r584, 0x0;
	ld.global.v4.b32 { %r581, %r582, %r583, %r584 }, [ %rd86 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r585, 0x0;
	mov.u32 %r586, 0x0;
	mov.u32 %r587, 0x0;
	mov.u32 %r588, 0x0;
	ld.global.v4.b32 { %r585, %r586, %r587, %r588 }, [ %rd87 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r589, 0x0;
	mov.u32 %r590, 0x0;
	mov.u32 %r591, 0x0;
	mov.u32 %r592, 0x0;
	ld.global.v4.b32 { %r589, %r590, %r591, %r592 }, [ %rd88 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r593, 0x0;
	mov.u32 %r594, 0x0;
	mov.u32 %r595, 0x0;
	mov.u32 %r596, 0x0;
	ld.global.v4.b32 { %r593, %r594, %r595, %r596 }, [ %rd89 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r597, 0x0;
	mov.u32 %r598, 0x0;
	mov.u32 %r599, 0x0;
	mov.u32 %r600, 0x0;
	ld.global.v4.b32 { %r597, %r598, %r599, %r600 }, [ %rd90 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r601, 0x0;
	mov.u32 %r602, 0x0;
	mov.u32 %r603, 0x0;
	mov.u32 %r604, 0x0;
	ld.global.v4.b32 { %r601, %r602, %r603, %r604 }, [ %rd91 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r605, 0x0;
	mov.u32 %r606, 0x0;
	mov.u32 %r607, 0x0;
	mov.u32 %r608, 0x0;
	ld.global.v4.b32 { %r605, %r606, %r607, %r608 }, [ %rd92 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r609, 0x0;
	mov.u32 %r610, 0x0;
	mov.u32 %r611, 0x0;
	mov.u32 %r612, 0x0;
	ld.global.v4.b32 { %r609, %r610, %r611, %r612 }, [ %rd93 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r613, 0x0;
	mov.u32 %r614, 0x0;
	mov.u32 %r615, 0x0;
	mov.u32 %r616, 0x0;
	ld.global.v4.b32 { %r613, %r614, %r615, %r616 }, [ %rd94 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r617, 0x0;
	mov.u32 %r618, 0x0;
	mov.u32 %r619, 0x0;
	mov.u32 %r620, 0x0;
	ld.global.v4.b32 { %r617, %r618, %r619, %r620 }, [ %rd95 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r621, 0x0;
	mov.u32 %r622, 0x0;
	mov.u32 %r623, 0x0;
	mov.u32 %r624, 0x0;
	ld.global.v4.b32 { %r621, %r622, %r623, %r624 }, [ %rd96 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r625, 0x0;
	mov.u32 %r626, 0x0;
	mov.u32 %r627, 0x0;
	mov.u32 %r628, 0x0;
	ld.global.v4.b32 { %r625, %r626, %r627, %r628 }, [ %rd97 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r629, 0x0;
	mov.u32 %r630, 0x0;
	mov.u32 %r631, 0x0;
	mov.u32 %r632, 0x0;
	ld.global.v4.b32 { %r629, %r630, %r631, %r632 }, [ %rd98 + 0 ];
	// end inline asm
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	bar.sync 	0;
	shl.b32 	%r851, %r2, 7;
	and.b32 	%r852, %r851, 15360;
	shl.b32 	%r853, %r816, 4;
	or.b32 	%r854, %r852, %r853;
	xor.b32 	%r855, %r854, %r782;
	add.s32 	%r565, %r193, %r855;
	// begin inline asm
	st.shared.v4.b32 [ %r565 + 0 ], { %r566, %r567, %r568, %r569 };
	// end inline asm
	add.s32 	%r570, %r565, 256;
	// begin inline asm
	st.shared.v4.b32 [ %r570 + 0 ], { %r571, %r572, %r573, %r574 };
	// end inline asm
	add.s32 	%r575, %r565, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r575 + 0 ], { %r576, %r577, %r578, %r579 };
	// end inline asm
	add.s32 	%r580, %r565, 768;
	// begin inline asm
	st.shared.v4.b32 [ %r580 + 0 ], { %r581, %r582, %r583, %r584 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r856, %r816, 11;
	shl.b32 	%r857, %r7, 4;
	shl.b32 	%r858, %r824, 2;
	setp.eq.b32 	%p24, %r1234, 0;
	shl.b32 	%r859, %r1234, 1;
	shr.u32 	%r860, %r6, 1;
	or.b32 	%r861, %r857, %r858;
	or.b32 	%r862, %r859, %r860;
	xor.b32 	%r863, %r861, %r862;
	or.b32 	%r864, %r863, %r856;
	add.s32 	%r865, %r193, %r864;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r866, %r867, %r868, %r869}, [%r865];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r870, %r871, %r872, %r873}, [%r865+1024];
	xor.b32 	%r874, %r864, 64;
	add.s32 	%r875, %r193, %r874;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r876, %r877, %r878, %r879}, [%r875];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r880, %r881, %r882, %r883}, [%r875+1024];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r565 + 0 ], { %r585, %r586, %r587, %r588 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r570 + 0 ], { %r589, %r590, %r591, %r592 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r575 + 0 ], { %r593, %r594, %r595, %r596 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r580 + 0 ], { %r597, %r598, %r599, %r600 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r884, %r885, %r886, %r887}, [%r865];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r888, %r889, %r890, %r891}, [%r865+1024];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r892, %r893, %r894, %r895}, [%r875];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r896, %r897, %r898, %r899}, [%r875+1024];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r565 + 0 ], { %r601, %r602, %r603, %r604 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r570 + 0 ], { %r605, %r606, %r607, %r608 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r575 + 0 ], { %r609, %r610, %r611, %r612 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r580 + 0 ], { %r613, %r614, %r615, %r616 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r900, %r901, %r902, %r903}, [%r865];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r904, %r905, %r906, %r907}, [%r865+1024];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r908, %r909, %r910, %r911}, [%r875];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r912, %r913, %r914, %r915}, [%r875+1024];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r565 + 0 ], { %r617, %r618, %r619, %r620 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r570 + 0 ], { %r621, %r622, %r623, %r624 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r575 + 0 ], { %r625, %r626, %r627, %r628 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r580 + 0 ], { %r629, %r630, %r631, %r632 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r916, %r917, %r918, %r919}, [%r865];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r920, %r921, %r922, %r923}, [%r865+1024];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r924, %r925, %r926, %r927}, [%r875];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r928, %r929, %r930, %r931}, [%r875+1024];
	.loc	1 357 31                        // sk06_mlp_down.py:357:31
	setp.lt.s32 	%p25, %r784, %r15;
	setp.lt.s32 	%p26, %r813, %r15;
	setp.lt.s32 	%p27, %r811, %r15;
	setp.lt.s32 	%p28, %r809, %r15;
	setp.lt.s32 	%p29, %r807, %r15;
	setp.lt.s32 	%p30, %r805, %r15;
	setp.lt.s32 	%p31, %r803, %r15;
	setp.lt.s32 	%p32, %r801, %r15;
	setp.lt.s32 	%p33, %r799, %r15;
	setp.lt.s32 	%p34, %r797, %r15;
	setp.lt.s32 	%p35, %r795, %r15;
	setp.lt.s32 	%p36, %r793, %r15;
	setp.lt.s32 	%p37, %r791, %r15;
	setp.lt.s32 	%p38, %r789, %r15;
	setp.lt.s32 	%p39, %r787, %r15;
	setp.lt.s32 	%p40, %r785, %r15;
	.loc	1 357 54                        // sk06_mlp_down.py:357:54
	setp.lt.s32 	%p41, %r778, %r16;
	.loc	1 357 37                        // sk06_mlp_down.py:357:37
	and.pred 	%p8, %p25, %p41;
	and.pred 	%p9, %p26, %p41;
	and.pred 	%p10, %p27, %p41;
	and.pred 	%p11, %p28, %p41;
	and.pred 	%p12, %p29, %p41;
	and.pred 	%p13, %p30, %p41;
	and.pred 	%p14, %p31, %p41;
	and.pred 	%p15, %p32, %p41;
	and.pred 	%p16, %p33, %p41;
	and.pred 	%p17, %p34, %p41;
	and.pred 	%p18, %p35, %p41;
	and.pred 	%p19, %p36, %p41;
	and.pred 	%p20, %p37, %p41;
	and.pred 	%p21, %p38, %p41;
	and.pred 	%p22, %p39, %p41;
	and.pred 	%p23, %p40, %p41;
	.loc	1 355 35                        // sk06_mlp_down.py:355:35
	mul.lo.s32 	%r932, %r784, %r17;
	mul.lo.s32 	%r933, %r813, %r17;
	mul.lo.s32 	%r934, %r811, %r17;
	mul.lo.s32 	%r935, %r809, %r17;
	mul.lo.s32 	%r936, %r807, %r17;
	mul.lo.s32 	%r937, %r805, %r17;
	mul.lo.s32 	%r938, %r803, %r17;
	mul.lo.s32 	%r939, %r801, %r17;
	mul.lo.s32 	%r940, %r799, %r17;
	mul.lo.s32 	%r941, %r797, %r17;
	mul.lo.s32 	%r942, %r795, %r17;
	mul.lo.s32 	%r943, %r793, %r17;
	mul.lo.s32 	%r944, %r791, %r17;
	mul.lo.s32 	%r945, %r789, %r17;
	mul.lo.s32 	%r946, %r787, %r17;
	mul.lo.s32 	%r947, %r785, %r17;
	.loc	1 355 18                        // sk06_mlp_down.py:355:18
	mad.wide.s32 	%rd132, %r932, 2, %rd26;
	mad.wide.s32 	%rd133, %r933, 2, %rd26;
	mad.wide.s32 	%rd134, %r934, 2, %rd26;
	mad.wide.s32 	%rd135, %r935, 2, %rd26;
	mad.wide.s32 	%rd136, %r936, 2, %rd26;
	mad.wide.s32 	%rd137, %r937, 2, %rd26;
	mad.wide.s32 	%rd138, %r938, 2, %rd26;
	mad.wide.s32 	%rd139, %r939, 2, %rd26;
	mad.wide.s32 	%rd140, %r940, 2, %rd26;
	mad.wide.s32 	%rd141, %r941, 2, %rd26;
	mad.wide.s32 	%rd142, %r942, 2, %rd26;
	mad.wide.s32 	%rd143, %r943, 2, %rd26;
	mad.wide.s32 	%rd144, %r944, 2, %rd26;
	mad.wide.s32 	%rd145, %r945, 2, %rd26;
	mad.wide.s32 	%rd146, %r946, 2, %rd26;
	mad.wide.s32 	%rd147, %r947, 2, %rd26;
	.loc	1 355 50                        // sk06_mlp_down.py:355:50
	mul.wide.s32 	%rd148, %r778, 2;
	add.s64 	%rd99, %rd132, %rd148;
	add.s64 	%rd100, %rd133, %rd148;
	add.s64 	%rd101, %rd134, %rd148;
	add.s64 	%rd102, %rd135, %rd148;
	add.s64 	%rd103, %rd136, %rd148;
	add.s64 	%rd104, %rd137, %rd148;
	add.s64 	%rd105, %rd138, %rd148;
	add.s64 	%rd106, %rd139, %rd148;
	add.s64 	%rd107, %rd140, %rd148;
	add.s64 	%rd108, %rd141, %rd148;
	add.s64 	%rd109, %rd142, %rd148;
	add.s64 	%rd110, %rd143, %rd148;
	add.s64 	%rd111, %rd144, %rd148;
	add.s64 	%rd112, %rd145, %rd148;
	add.s64 	%rd113, %rd146, %rd148;
	add.s64 	%rd114, %rd147, %rd148;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs1, %rs2}, %r866;
	cvt.f32.bf16 	%r948, %rs2;
	cvt.f32.bf16 	%r949, %rs1;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r950, %r1235, %r827, %r949;
	fma.rn.f32 	%r951, %r1236, %r827, %r948;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r634, %r951, %r950;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs3, %rs4}, %r867;
	cvt.f32.bf16 	%r952, %rs4;
	cvt.f32.bf16 	%r953, %rs3;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r954, %r1237, %r828, %r953;
	fma.rn.f32 	%r955, %r1238, %r828, %r952;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r639, %r955, %r954;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs5, %rs6}, %r868;
	cvt.f32.bf16 	%r956, %rs6;
	cvt.f32.bf16 	%r957, %rs5;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r958, %r1239, %r827, %r957;
	fma.rn.f32 	%r959, %r1240, %r827, %r956;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r654, %r959, %r958;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs7, %rs8}, %r869;
	cvt.f32.bf16 	%r960, %rs8;
	cvt.f32.bf16 	%r961, %rs7;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r962, %r1241, %r828, %r961;
	fma.rn.f32 	%r963, %r1242, %r828, %r960;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r659, %r963, %r962;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs9, %rs10}, %r876;
	cvt.f32.bf16 	%r964, %rs10;
	cvt.f32.bf16 	%r965, %rs9;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r966, %r1243, %r827, %r965;
	fma.rn.f32 	%r967, %r1244, %r827, %r964;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r674, %r967, %r966;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs11, %rs12}, %r877;
	cvt.f32.bf16 	%r968, %rs12;
	cvt.f32.bf16 	%r969, %rs11;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r970, %r1245, %r828, %r969;
	fma.rn.f32 	%r971, %r1246, %r828, %r968;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r679, %r971, %r970;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs13, %rs14}, %r878;
	cvt.f32.bf16 	%r972, %rs14;
	cvt.f32.bf16 	%r973, %rs13;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r974, %r1247, %r827, %r973;
	fma.rn.f32 	%r975, %r1248, %r827, %r972;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r694, %r975, %r974;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs15, %rs16}, %r879;
	cvt.f32.bf16 	%r976, %rs16;
	cvt.f32.bf16 	%r977, %rs15;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r978, %r1249, %r828, %r977;
	fma.rn.f32 	%r979, %r1250, %r828, %r976;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r699, %r979, %r978;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs17, %rs18}, %r870;
	cvt.f32.bf16 	%r980, %rs18;
	cvt.f32.bf16 	%r981, %rs17;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r982, %r1251, %r827, %r981;
	fma.rn.f32 	%r983, %r1252, %r827, %r980;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r644, %r983, %r982;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs19, %rs20}, %r871;
	cvt.f32.bf16 	%r984, %rs20;
	cvt.f32.bf16 	%r985, %rs19;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r986, %r1253, %r828, %r985;
	fma.rn.f32 	%r987, %r1254, %r828, %r984;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r649, %r987, %r986;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs21, %rs22}, %r872;
	cvt.f32.bf16 	%r988, %rs22;
	cvt.f32.bf16 	%r989, %rs21;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r990, %r1255, %r827, %r989;
	fma.rn.f32 	%r991, %r1256, %r827, %r988;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r664, %r991, %r990;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs23, %rs24}, %r873;
	cvt.f32.bf16 	%r992, %rs24;
	cvt.f32.bf16 	%r993, %rs23;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r994, %r1257, %r828, %r993;
	fma.rn.f32 	%r995, %r1258, %r828, %r992;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r669, %r995, %r994;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs25, %rs26}, %r880;
	cvt.f32.bf16 	%r996, %rs26;
	cvt.f32.bf16 	%r997, %rs25;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r998, %r1259, %r827, %r997;
	fma.rn.f32 	%r999, %r1260, %r827, %r996;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r684, %r999, %r998;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs27, %rs28}, %r881;
	cvt.f32.bf16 	%r1000, %rs28;
	cvt.f32.bf16 	%r1001, %rs27;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1002, %r1261, %r828, %r1001;
	fma.rn.f32 	%r1003, %r1262, %r828, %r1000;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r689, %r1003, %r1002;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs29, %rs30}, %r882;
	cvt.f32.bf16 	%r1004, %rs30;
	cvt.f32.bf16 	%r1005, %rs29;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1006, %r1263, %r827, %r1005;
	fma.rn.f32 	%r1007, %r1264, %r827, %r1004;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r704, %r1007, %r1006;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs31, %rs32}, %r883;
	cvt.f32.bf16 	%r1008, %rs32;
	cvt.f32.bf16 	%r1009, %rs31;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1010, %r1265, %r828, %r1009;
	fma.rn.f32 	%r1011, %r1266, %r828, %r1008;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r709, %r1011, %r1010;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs33, %rs34}, %r884;
	cvt.f32.bf16 	%r1012, %rs34;
	cvt.f32.bf16 	%r1013, %rs33;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1014, %r1267, %r829, %r1013;
	fma.rn.f32 	%r1015, %r1268, %r829, %r1012;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r635, %r1015, %r1014;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs35, %rs36}, %r885;
	cvt.f32.bf16 	%r1016, %rs36;
	cvt.f32.bf16 	%r1017, %rs35;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1018, %r1269, %r830, %r1017;
	fma.rn.f32 	%r1019, %r1270, %r830, %r1016;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r640, %r1019, %r1018;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs37, %rs38}, %r886;
	cvt.f32.bf16 	%r1020, %rs38;
	cvt.f32.bf16 	%r1021, %rs37;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1022, %r1271, %r829, %r1021;
	fma.rn.f32 	%r1023, %r1272, %r829, %r1020;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r655, %r1023, %r1022;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs39, %rs40}, %r887;
	cvt.f32.bf16 	%r1024, %rs40;
	cvt.f32.bf16 	%r1025, %rs39;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1026, %r1273, %r830, %r1025;
	fma.rn.f32 	%r1027, %r1274, %r830, %r1024;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r660, %r1027, %r1026;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs41, %rs42}, %r892;
	cvt.f32.bf16 	%r1028, %rs42;
	cvt.f32.bf16 	%r1029, %rs41;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1030, %r1275, %r829, %r1029;
	fma.rn.f32 	%r1031, %r1276, %r829, %r1028;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r675, %r1031, %r1030;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs43, %rs44}, %r893;
	cvt.f32.bf16 	%r1032, %rs44;
	cvt.f32.bf16 	%r1033, %rs43;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1034, %r1277, %r830, %r1033;
	fma.rn.f32 	%r1035, %r1278, %r830, %r1032;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r680, %r1035, %r1034;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs45, %rs46}, %r894;
	cvt.f32.bf16 	%r1036, %rs46;
	cvt.f32.bf16 	%r1037, %rs45;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1038, %r1279, %r829, %r1037;
	fma.rn.f32 	%r1039, %r1280, %r829, %r1036;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r695, %r1039, %r1038;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs47, %rs48}, %r895;
	cvt.f32.bf16 	%r1040, %rs48;
	cvt.f32.bf16 	%r1041, %rs47;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1042, %r1281, %r830, %r1041;
	fma.rn.f32 	%r1043, %r1282, %r830, %r1040;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r700, %r1043, %r1042;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs49, %rs50}, %r888;
	cvt.f32.bf16 	%r1044, %rs50;
	cvt.f32.bf16 	%r1045, %rs49;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1046, %r1283, %r829, %r1045;
	fma.rn.f32 	%r1047, %r1284, %r829, %r1044;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r645, %r1047, %r1046;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs51, %rs52}, %r889;
	cvt.f32.bf16 	%r1048, %rs52;
	cvt.f32.bf16 	%r1049, %rs51;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1050, %r1285, %r830, %r1049;
	fma.rn.f32 	%r1051, %r1286, %r830, %r1048;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r650, %r1051, %r1050;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs53, %rs54}, %r890;
	cvt.f32.bf16 	%r1052, %rs54;
	cvt.f32.bf16 	%r1053, %rs53;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1054, %r1287, %r829, %r1053;
	fma.rn.f32 	%r1055, %r1288, %r829, %r1052;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r665, %r1055, %r1054;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs55, %rs56}, %r891;
	cvt.f32.bf16 	%r1056, %rs56;
	cvt.f32.bf16 	%r1057, %rs55;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1058, %r1289, %r830, %r1057;
	fma.rn.f32 	%r1059, %r1290, %r830, %r1056;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r670, %r1059, %r1058;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs57, %rs58}, %r896;
	cvt.f32.bf16 	%r1060, %rs58;
	cvt.f32.bf16 	%r1061, %rs57;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1062, %r1291, %r829, %r1061;
	fma.rn.f32 	%r1063, %r1292, %r829, %r1060;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r685, %r1063, %r1062;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs59, %rs60}, %r897;
	cvt.f32.bf16 	%r1064, %rs60;
	cvt.f32.bf16 	%r1065, %rs59;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1066, %r1293, %r830, %r1065;
	fma.rn.f32 	%r1067, %r1294, %r830, %r1064;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r690, %r1067, %r1066;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs61, %rs62}, %r898;
	cvt.f32.bf16 	%r1068, %rs62;
	cvt.f32.bf16 	%r1069, %rs61;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1070, %r1295, %r829, %r1069;
	fma.rn.f32 	%r1071, %r1296, %r829, %r1068;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r705, %r1071, %r1070;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs63, %rs64}, %r899;
	cvt.f32.bf16 	%r1072, %rs64;
	cvt.f32.bf16 	%r1073, %rs63;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1074, %r1297, %r830, %r1073;
	fma.rn.f32 	%r1075, %r1298, %r830, %r1072;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r710, %r1075, %r1074;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs65, %rs66}, %r900;
	cvt.f32.bf16 	%r1076, %rs66;
	cvt.f32.bf16 	%r1077, %rs65;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1078, %r1299, %r831, %r1077;
	fma.rn.f32 	%r1079, %r1300, %r831, %r1076;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r636, %r1079, %r1078;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs67, %rs68}, %r901;
	cvt.f32.bf16 	%r1080, %rs68;
	cvt.f32.bf16 	%r1081, %rs67;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1082, %r1301, %r832, %r1081;
	fma.rn.f32 	%r1083, %r1302, %r832, %r1080;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r641, %r1083, %r1082;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs69, %rs70}, %r902;
	cvt.f32.bf16 	%r1084, %rs70;
	cvt.f32.bf16 	%r1085, %rs69;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1086, %r1303, %r831, %r1085;
	fma.rn.f32 	%r1087, %r1304, %r831, %r1084;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r656, %r1087, %r1086;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs71, %rs72}, %r903;
	cvt.f32.bf16 	%r1088, %rs72;
	cvt.f32.bf16 	%r1089, %rs71;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1090, %r1305, %r832, %r1089;
	fma.rn.f32 	%r1091, %r1306, %r832, %r1088;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r661, %r1091, %r1090;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs73, %rs74}, %r908;
	cvt.f32.bf16 	%r1092, %rs74;
	cvt.f32.bf16 	%r1093, %rs73;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1094, %r1307, %r831, %r1093;
	fma.rn.f32 	%r1095, %r1308, %r831, %r1092;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r676, %r1095, %r1094;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs75, %rs76}, %r909;
	cvt.f32.bf16 	%r1096, %rs76;
	cvt.f32.bf16 	%r1097, %rs75;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1098, %r1309, %r832, %r1097;
	fma.rn.f32 	%r1099, %r1310, %r832, %r1096;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r681, %r1099, %r1098;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs77, %rs78}, %r910;
	cvt.f32.bf16 	%r1100, %rs78;
	cvt.f32.bf16 	%r1101, %rs77;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1102, %r1311, %r831, %r1101;
	fma.rn.f32 	%r1103, %r1312, %r831, %r1100;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r696, %r1103, %r1102;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs79, %rs80}, %r911;
	cvt.f32.bf16 	%r1104, %rs80;
	cvt.f32.bf16 	%r1105, %rs79;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1106, %r1313, %r832, %r1105;
	fma.rn.f32 	%r1107, %r1314, %r832, %r1104;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r701, %r1107, %r1106;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs81, %rs82}, %r904;
	cvt.f32.bf16 	%r1108, %rs82;
	cvt.f32.bf16 	%r1109, %rs81;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1110, %r1315, %r831, %r1109;
	fma.rn.f32 	%r1111, %r1316, %r831, %r1108;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r646, %r1111, %r1110;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs83, %rs84}, %r905;
	cvt.f32.bf16 	%r1112, %rs84;
	cvt.f32.bf16 	%r1113, %rs83;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1114, %r1317, %r832, %r1113;
	fma.rn.f32 	%r1115, %r1318, %r832, %r1112;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r651, %r1115, %r1114;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs85, %rs86}, %r906;
	cvt.f32.bf16 	%r1116, %rs86;
	cvt.f32.bf16 	%r1117, %rs85;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1118, %r1319, %r831, %r1117;
	fma.rn.f32 	%r1119, %r1320, %r831, %r1116;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r666, %r1119, %r1118;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs87, %rs88}, %r907;
	cvt.f32.bf16 	%r1120, %rs88;
	cvt.f32.bf16 	%r1121, %rs87;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1122, %r1321, %r832, %r1121;
	fma.rn.f32 	%r1123, %r1322, %r832, %r1120;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r671, %r1123, %r1122;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs89, %rs90}, %r912;
	cvt.f32.bf16 	%r1124, %rs90;
	cvt.f32.bf16 	%r1125, %rs89;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1126, %r1323, %r831, %r1125;
	fma.rn.f32 	%r1127, %r1324, %r831, %r1124;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r686, %r1127, %r1126;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs91, %rs92}, %r913;
	cvt.f32.bf16 	%r1128, %rs92;
	cvt.f32.bf16 	%r1129, %rs91;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1130, %r1325, %r832, %r1129;
	fma.rn.f32 	%r1131, %r1326, %r832, %r1128;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r691, %r1131, %r1130;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs93, %rs94}, %r914;
	cvt.f32.bf16 	%r1132, %rs94;
	cvt.f32.bf16 	%r1133, %rs93;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1134, %r1327, %r831, %r1133;
	fma.rn.f32 	%r1135, %r1328, %r831, %r1132;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r706, %r1135, %r1134;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs95, %rs96}, %r915;
	cvt.f32.bf16 	%r1136, %rs96;
	cvt.f32.bf16 	%r1137, %rs95;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1138, %r1329, %r832, %r1137;
	fma.rn.f32 	%r1139, %r1330, %r832, %r1136;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r711, %r1139, %r1138;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs97, %rs98}, %r916;
	cvt.f32.bf16 	%r1140, %rs98;
	cvt.f32.bf16 	%r1141, %rs97;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1142, %r1331, %r833, %r1141;
	fma.rn.f32 	%r1143, %r1332, %r833, %r1140;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r637, %r1143, %r1142;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs99, %rs100}, %r917;
	cvt.f32.bf16 	%r1144, %rs100;
	cvt.f32.bf16 	%r1145, %rs99;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1146, %r1333, %r834, %r1145;
	fma.rn.f32 	%r1147, %r1334, %r834, %r1144;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r642, %r1147, %r1146;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs101, %rs102}, %r918;
	cvt.f32.bf16 	%r1148, %rs102;
	cvt.f32.bf16 	%r1149, %rs101;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1150, %r1335, %r833, %r1149;
	fma.rn.f32 	%r1151, %r1336, %r833, %r1148;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r657, %r1151, %r1150;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs103, %rs104}, %r919;
	cvt.f32.bf16 	%r1152, %rs104;
	cvt.f32.bf16 	%r1153, %rs103;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1154, %r1337, %r834, %r1153;
	fma.rn.f32 	%r1155, %r1338, %r834, %r1152;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r662, %r1155, %r1154;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs105, %rs106}, %r924;
	cvt.f32.bf16 	%r1156, %rs106;
	cvt.f32.bf16 	%r1157, %rs105;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1158, %r1339, %r833, %r1157;
	fma.rn.f32 	%r1159, %r1340, %r833, %r1156;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r677, %r1159, %r1158;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs107, %rs108}, %r925;
	cvt.f32.bf16 	%r1160, %rs108;
	cvt.f32.bf16 	%r1161, %rs107;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1162, %r1341, %r834, %r1161;
	fma.rn.f32 	%r1163, %r1342, %r834, %r1160;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r682, %r1163, %r1162;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs109, %rs110}, %r926;
	cvt.f32.bf16 	%r1164, %rs110;
	cvt.f32.bf16 	%r1165, %rs109;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1166, %r1343, %r833, %r1165;
	fma.rn.f32 	%r1167, %r1344, %r833, %r1164;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r697, %r1167, %r1166;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs111, %rs112}, %r927;
	cvt.f32.bf16 	%r1168, %rs112;
	cvt.f32.bf16 	%r1169, %rs111;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1170, %r1345, %r834, %r1169;
	fma.rn.f32 	%r1171, %r1346, %r834, %r1168;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r702, %r1171, %r1170;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs113, %rs114}, %r920;
	cvt.f32.bf16 	%r1172, %rs114;
	cvt.f32.bf16 	%r1173, %rs113;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1174, %r1347, %r833, %r1173;
	fma.rn.f32 	%r1175, %r1348, %r833, %r1172;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r647, %r1175, %r1174;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs115, %rs116}, %r921;
	cvt.f32.bf16 	%r1176, %rs116;
	cvt.f32.bf16 	%r1177, %rs115;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1178, %r1349, %r834, %r1177;
	fma.rn.f32 	%r1179, %r1350, %r834, %r1176;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r652, %r1179, %r1178;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs117, %rs118}, %r922;
	cvt.f32.bf16 	%r1180, %rs118;
	cvt.f32.bf16 	%r1181, %rs117;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1182, %r1351, %r833, %r1181;
	fma.rn.f32 	%r1183, %r1352, %r833, %r1180;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r667, %r1183, %r1182;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs119, %rs120}, %r923;
	cvt.f32.bf16 	%r1184, %rs120;
	cvt.f32.bf16 	%r1185, %rs119;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1186, %r1353, %r834, %r1185;
	fma.rn.f32 	%r1187, %r1354, %r834, %r1184;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r672, %r1187, %r1186;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs121, %rs122}, %r928;
	cvt.f32.bf16 	%r1188, %rs122;
	cvt.f32.bf16 	%r1189, %rs121;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1190, %r1355, %r833, %r1189;
	fma.rn.f32 	%r1191, %r1356, %r833, %r1188;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r687, %r1191, %r1190;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs123, %rs124}, %r929;
	cvt.f32.bf16 	%r1192, %rs124;
	cvt.f32.bf16 	%r1193, %rs123;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1194, %r1357, %r834, %r1193;
	fma.rn.f32 	%r1195, %r1358, %r834, %r1192;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r692, %r1195, %r1194;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs125, %rs126}, %r930;
	cvt.f32.bf16 	%r1196, %rs126;
	cvt.f32.bf16 	%r1197, %rs125;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1198, %r1359, %r833, %r1197;
	fma.rn.f32 	%r1199, %r1360, %r833, %r1196;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r707, %r1199, %r1198;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs127, %rs128}, %r931;
	cvt.f32.bf16 	%r1200, %rs128;
	cvt.f32.bf16 	%r1201, %rs127;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1202, %r1361, %r834, %r1201;
	fma.rn.f32 	%r1203, %r1362, %r834, %r1200;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r712, %r1203, %r1202;
	bar.sync 	0;
	shl.b32 	%r1204, %r5, 14;
	shl.b32 	%r1205, %r5, 5;
	and.b32 	%r1206, %r1233, 3456;
	bfe.s32 	%r1207, %r2, 2, 1;
	and.b32 	%r1208, %r1207, 8208;
	or.b32 	%r1209, %r1205, %r1206;
	xor.b32 	%r1210, %r1208, %r860;
	or.b32 	%r1211, %r1210, %r1209;
	or.b32 	%r1212, %r1211, %r1204;
	add.s32 	%r633, %r193, %r1212;
	// begin inline asm
	st.shared.v4.b32 [ %r633 + 0 ], { %r634, %r635, %r636, %r637 };
	// end inline asm
	add.s32 	%r638, %r633, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r638 + 0 ], { %r639, %r640, %r641, %r642 };
	// end inline asm
	add.s32 	%r643, %r633, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r643 + 0 ], { %r644, %r645, %r646, %r647 };
	// end inline asm
	add.s32 	%r648, %r633, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r648 + 0 ], { %r649, %r650, %r651, %r652 };
	// end inline asm
	xor.b32 	%r1213, %r1212, 32;
	add.s32 	%r653, %r193, %r1213;
	// begin inline asm
	st.shared.v4.b32 [ %r653 + 0 ], { %r654, %r655, %r656, %r657 };
	// end inline asm
	add.s32 	%r658, %r653, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r658 + 0 ], { %r659, %r660, %r661, %r662 };
	// end inline asm
	add.s32 	%r663, %r653, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r663 + 0 ], { %r664, %r665, %r666, %r667 };
	// end inline asm
	add.s32 	%r668, %r653, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r668 + 0 ], { %r669, %r670, %r671, %r672 };
	// end inline asm
	xor.b32 	%r1214, %r1212, 64;
	add.s32 	%r673, %r193, %r1214;
	// begin inline asm
	st.shared.v4.b32 [ %r673 + 0 ], { %r674, %r675, %r676, %r677 };
	// end inline asm
	add.s32 	%r678, %r673, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r678 + 0 ], { %r679, %r680, %r681, %r682 };
	// end inline asm
	add.s32 	%r683, %r673, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r683 + 0 ], { %r684, %r685, %r686, %r687 };
	// end inline asm
	add.s32 	%r688, %r673, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r688 + 0 ], { %r689, %r690, %r691, %r692 };
	// end inline asm
	xor.b32 	%r1215, %r1212, 96;
	add.s32 	%r693, %r193, %r1215;
	// begin inline asm
	st.shared.v4.b32 [ %r693 + 0 ], { %r694, %r695, %r696, %r697 };
	// end inline asm
	add.s32 	%r698, %r693, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r698 + 0 ], { %r699, %r700, %r701, %r702 };
	// end inline asm
	add.s32 	%r703, %r693, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r703 + 0 ], { %r704, %r705, %r706, %r707 };
	// end inline asm
	add.s32 	%r708, %r693, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r708 + 0 ], { %r709, %r710, %r711, %r712 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r1216, %r2, 2;
	and.b32 	%r1217, %r1216, 896;
	shl.b32 	%r1218, %r819, 9;
	selp.b32 	%r1219, 0, 8208, %p24;
	or.b32 	%r1220, %r853, %r1217;
	xor.b32 	%r1221, %r1220, %r1219;
	or.b32 	%r1222, %r1221, %r1218;
	add.s32 	%r1223, %r193, %r1222;
	ld.shared.v4.b32 	{%r713, %r729, %r745, %r761}, [%r1223];
	ld.shared.v4.b32 	{%r717, %r733, %r749, %r765}, [%r1223+1024];
	ld.shared.v4.b32 	{%r721, %r737, %r753, %r769}, [%r1223+2048];
	ld.shared.v4.b32 	{%r725, %r741, %r757, %r773}, [%r1223+3072];
	xor.b32 	%r1224, %r1222, 32;
	add.s32 	%r1225, %r193, %r1224;
	ld.shared.v4.b32 	{%r714, %r730, %r746, %r762}, [%r1225+16384];
	ld.shared.v4.b32 	{%r718, %r734, %r750, %r766}, [%r1225+17408];
	ld.shared.v4.b32 	{%r722, %r738, %r754, %r770}, [%r1225+18432];
	ld.shared.v4.b32 	{%r726, %r742, %r758, %r774}, [%r1225+19456];
	xor.b32 	%r1226, %r1222, 64;
	add.s32 	%r1227, %r193, %r1226;
	ld.shared.v4.b32 	{%r715, %r731, %r747, %r763}, [%r1227+32768];
	ld.shared.v4.b32 	{%r719, %r735, %r751, %r767}, [%r1227+33792];
	ld.shared.v4.b32 	{%r723, %r739, %r755, %r771}, [%r1227+34816];
	ld.shared.v4.b32 	{%r727, %r743, %r759, %r775}, [%r1227+35840];
	xor.b32 	%r1228, %r1222, 96;
	add.s32 	%r1229, %r193, %r1228;
	ld.shared.v4.b32 	{%r716, %r732, %r748, %r764}, [%r1229+49152];
	ld.shared.v4.b32 	{%r720, %r736, %r752, %r768}, [%r1229+50176];
	ld.shared.v4.b32 	{%r724, %r740, %r756, %r772}, [%r1229+51200];
	ld.shared.v4.b32 	{%r728, %r744, %r760, %r776}, [%r1229+52224];
	.loc	1 356 8                         // sk06_mlp_down.py:356:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd99 + 0 ], { %r713, %r714, %r715, %r716 };
	// end inline asm
	// begin inline asm
	@%p9 st.global.v4.b32 [ %rd100 + 0 ], { %r717, %r718, %r719, %r720 };
	// end inline asm
	// begin inline asm
	@%p10 st.global.v4.b32 [ %rd101 + 0 ], { %r721, %r722, %r723, %r724 };
	// end inline asm
	// begin inline asm
	@%p11 st.global.v4.b32 [ %rd102 + 0 ], { %r725, %r726, %r727, %r728 };
	// end inline asm
	// begin inline asm
	@%p12 st.global.v4.b32 [ %rd103 + 0 ], { %r729, %r730, %r731, %r732 };
	// end inline asm
	// begin inline asm
	@%p13 st.global.v4.b32 [ %rd104 + 0 ], { %r733, %r734, %r735, %r736 };
	// end inline asm
	// begin inline asm
	@%p14 st.global.v4.b32 [ %rd105 + 0 ], { %r737, %r738, %r739, %r740 };
	// end inline asm
	// begin inline asm
	@%p15 st.global.v4.b32 [ %rd106 + 0 ], { %r741, %r742, %r743, %r744 };
	// end inline asm
	// begin inline asm
	@%p16 st.global.v4.b32 [ %rd107 + 0 ], { %r745, %r746, %r747, %r748 };
	// end inline asm
	// begin inline asm
	@%p17 st.global.v4.b32 [ %rd108 + 0 ], { %r749, %r750, %r751, %r752 };
	// end inline asm
	// begin inline asm
	@%p18 st.global.v4.b32 [ %rd109 + 0 ], { %r753, %r754, %r755, %r756 };
	// end inline asm
	// begin inline asm
	@%p19 st.global.v4.b32 [ %rd110 + 0 ], { %r757, %r758, %r759, %r760 };
	// end inline asm
	// begin inline asm
	@%p20 st.global.v4.b32 [ %rd111 + 0 ], { %r761, %r762, %r763, %r764 };
	// end inline asm
	// begin inline asm
	@%p21 st.global.v4.b32 [ %rd112 + 0 ], { %r765, %r766, %r767, %r768 };
	// end inline asm
	// begin inline asm
	@%p22 st.global.v4.b32 [ %rd113 + 0 ], { %r769, %r770, %r771, %r772 };
	// end inline asm
	// begin inline asm
	@%p23 st.global.v4.b32 [ %rd114 + 0 ], { %r773, %r774, %r775, %r776 };
	// end inline asm
	.loc	1 354 4                         // sk06_mlp_down.py:354:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/workspace/vllm/_genesis/kernels/sk06_mlp_down.py"
	.file	2 "/usr/local/lib/python3.12/dist-packages/triton/language/standard.py"
	.section	.debug_abbrev
	{
.b8 1                                   // Abbreviation Code
.b8 17                                  // DW_TAG_compile_unit
.b8 1                                   // DW_CHILDREN_yes
.b8 37                                  // DW_AT_producer
.b8 8                                   // DW_FORM_string
.b8 19                                  // DW_AT_language
.b8 5                                   // DW_FORM_data2
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 16                                  // DW_AT_stmt_list
.b8 6                                   // DW_FORM_data4
.b8 27                                  // DW_AT_comp_dir
.b8 8                                   // DW_FORM_string
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 2                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 0                                   // DW_CHILDREN_no
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 32                                  // DW_AT_inline
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 3                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 1                                   // DW_CHILDREN_yes
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 4                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 0                                   // DW_CHILDREN_no
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 5                                   // DW_FORM_data2
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 168                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0xa1 DW_TAG_compile_unit
.b8 116                                 // DW_AT_producer
.b8 114
.b8 105
.b8 116
.b8 111
.b8 110
.b8 0
.b8 2                                   // DW_AT_language
.b8 0
.b8 115                                 // DW_AT_name
.b8 107
.b8 48
.b8 54
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 100
.b8 111
.b8 119
.b8 110
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 119
.b8 111
.b8 114
.b8 107
.b8 115
.b8 112
.b8 97
.b8 99
.b8 101
.b8 47
.b8 118
.b8 108
.b8 108
.b8 109
.b8 47
.b8 95
.b8 103
.b8 101
.b8 110
.b8 101
.b8 115
.b8 105
.b8 115
.b8 47
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 115
.b8 0
.b8 2                                   // Abbrev [2] 0x4b:0x18 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 54
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 100
.b8 111
.b8 119
.b8 110
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x63:0x48 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 75                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x78:0x19 DW_TAG_inlined_subroutine
.b32 75                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 58                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x91:0x19 DW_TAG_inlined_subroutine
.b32 75                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 59                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_10 = _Nativo(
    "sk06_mlp_down/tile256x128x64_shift1_abi15",
    _PTX_10, "_sk06_mlp_down_kernel",
    warps=8, shared=73728,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 18],
    horneado={11: 1, 12: 1, 15: 1, 17: 1, 19: 1, 20: 256, 21: 128, 22: 64, 23: 8, 24: 128, 25: True},
    div16=[8, 9, 10, 13, 14, 16],
)

_PTX_11 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk06_mlp_down_kernel   // -- Begin function _sk06_mlp_down_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk06_mlp_down_kernel
.visible .entry _sk06_mlp_down_kernel(
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_6,
	.param .u32 _sk06_mlp_down_kernel_param_7,
	.param .u32 _sk06_mlp_down_kernel_param_8,
	.param .u32 _sk06_mlp_down_kernel_param_9,
	.param .u32 _sk06_mlp_down_kernel_param_10,
	.param .u32 _sk06_mlp_down_kernel_param_11,
	.param .u32 _sk06_mlp_down_kernel_param_12,
	.param .u32 _sk06_mlp_down_kernel_param_13,
	.param .u32 _sk06_mlp_down_kernel_param_14,
	.param .u32 _sk06_mlp_down_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_16,
	.param .u64 .ptr .global .align 1 _sk06_mlp_down_kernel_param_17
)
.reqntid 256
{
	.reg .pred 	%p<42>;
	.reg .b16 	%rs<257>;
	.reg .b32 	%r<1386>;
	.reg .b64 	%rd<274>;
	.loc	1 304 0                         // sk06_mlp_down.py:304:0
$L__func_begin0:
	.loc	1 304 0                         // sk06_mlp_down.py:304:0

// %bb.0:
	ld.param.b32 	%r19, [_sk06_mlp_down_kernel_param_14];
	ld.param.b32 	%r18, [_sk06_mlp_down_kernel_param_13];
	ld.param.b32 	%r17, [_sk06_mlp_down_kernel_param_12];
	ld.param.b32 	%r16, [_sk06_mlp_down_kernel_param_8];
	ld.param.b32 	%r15, [_sk06_mlp_down_kernel_param_7];
	ld.param.b64 	%rd28, [_sk06_mlp_down_kernel_param_4];
	ld.param.b64 	%rd27, [_sk06_mlp_down_kernel_param_3];
	ld.param.b64 	%rd26, [_sk06_mlp_down_kernel_param_2];
	ld.param.b64 	%rd25, [_sk06_mlp_down_kernel_param_1];
	ld.param.b64 	%rd24, [_sk06_mlp_down_kernel_param_0];
$L__tmp0:
	.loc	1 313 24                        // sk06_mlp_down.py:313:24
	mov.u32 	%r42, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk06_mlp_down.py:314:27 ]
	add.s32 	%r43, %r15, 255;
	.loc	2 43 30                         // standard.py:43:30 @[ sk06_mlp_down.py:314:27 ]
	shr.s32 	%r44, %r43, 31;
	shr.u32 	%r45, %r44, 24;
	add.s32 	%r46, %r43, %r45;
	shr.s32 	%r47, %r46, 8;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk06_mlp_down.py:315:27 ]
	add.s32 	%r48, %r16, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk06_mlp_down.py:315:27 ]
	shr.s32 	%r49, %r48, 31;
	shr.u32 	%r50, %r49, 25;
	add.s32 	%r51, %r48, %r50;
	shr.s32 	%r52, %r51, 7;
$L__tmp3:
	.loc	1 316 29                        // sk06_mlp_down.py:316:29
	shl.b32 	%r53, %r52, 3;
	.loc	1 317 22                        // sk06_mlp_down.py:317:22
	div.s32 	%r54, %r42, %r53;
	.loc	1 317 38                        // sk06_mlp_down.py:317:38
	shl.b32 	%r55, %r54, 3;
	ld.param.b32 	%r56, [_sk06_mlp_down_kernel_param_9];
	.loc	1 318 30                        // sk06_mlp_down.py:318:30
	sub.s32 	%r57, %r47, %r55;
	ld.param.b32 	%r58, [_sk06_mlp_down_kernel_param_10];
	.loc	1 318 39                        // sk06_mlp_down.py:318:39
	min.s32 	%r59, %r57, 8;
	ld.param.b32 	%r60, [_sk06_mlp_down_kernel_param_11];
	.loc	1 319 30                        // sk06_mlp_down.py:319:30
	mul.lo.s32 	%r61, %r54, %r53;
	sub.s32 	%r62, %r42, %r61;
	.loc	1 320 36                        // sk06_mlp_down.py:320:36
	div.s32 	%r63, %r62, %r59;
	.loc	1 319 46                        // sk06_mlp_down.py:319:46
	mul.lo.s32 	%r64, %r63, %r59;
	sub.s32 	%r65, %r62, %r64;
	.loc	1 319 23                        // sk06_mlp_down.py:319:23
	add.s32 	%r66, %r65, %r55;
	.loc	1 322 22                        // sk06_mlp_down.py:322:22
	shl.b32 	%r1, %r66, 8;
	.loc	1 322 45                        // sk06_mlp_down.py:322:45
	mov.u32 	%r2, %tid.x;
	shr.u32 	%r67, %r2, 2;
	bfe.u32 	%r68, %r2, 2, 6;
	or.b32 	%r69, %r68, 64;
	and.b32 	%r3, %r2, 255;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r70, %r1, %r68;
	or.b32 	%r71, %r1, %r69;
	or.b32 	%r72, %r70, 128;
	or.b32 	%r73, %r1, %r67;
	or.b32 	%r74, %r73, 192;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r75, %r70, %r15;
	rem.s32 	%r76, %r71, %r15;
	rem.s32 	%r77, %r72, %r15;
	rem.s32 	%r78, %r74, %r15;
	.loc	1 323 22                        // sk06_mlp_down.py:323:22
	shl.b32 	%r4, %r63, 7;
	.loc	1 323 45                        // sk06_mlp_down.py:323:45
	and.b32 	%r5, %r2, 3;
	and.b32 	%r6, %r2, 32;
	and.b32 	%r7, %r2, 15;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r82, %r4, %r68;
	or.b32 	%r83, %r4, %r69;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r100, %r82, %r16;
	rem.s32 	%r101, %r83, %r16;
	.loc	1 326 39                        // sk06_mlp_down.py:326:39
	mul.lo.s32 	%r118, %r75, %r58;
	mul.lo.s32 	%r119, %r76, %r58;
	mul.lo.s32 	%r120, %r77, %r58;
	mul.lo.s32 	%r121, %r78, %r58;
	.loc	1 326 21                        // sk06_mlp_down.py:326:21
	cvt.s64.s32 	%rd1, %r118;
	add.s64 	%rd48, %rd24, %rd1;
	cvt.s64.s32 	%rd2, %r119;
	add.s64 	%rd49, %rd24, %rd2;
	cvt.s64.s32 	%rd3, %r120;
	add.s64 	%rd50, %rd24, %rd3;
	cvt.s64.s32 	%rd4, %r121;
	add.s64 	%rd51, %rd24, %rd4;
	.loc	1 326 58                        // sk06_mlp_down.py:326:58
	shl.b32 	%r122, %r5, 4;
	.loc	1 326 51                        // sk06_mlp_down.py:326:51
	cvt.u64.u32 	%rd5, %r122;
	add.s64 	%rd29, %rd48, %rd5;
	add.s64 	%rd30, %rd49, %rd5;
	add.s64 	%rd31, %rd50, %rd5;
	add.s64 	%rd32, %rd51, %rd5;
	.loc	1 327 21                        // sk06_mlp_down.py:327:21
	add.s64 	%rd52, %rd25, %rd5;
	.loc	1 327 69                        // sk06_mlp_down.py:327:69
	mul.lo.s32 	%r123, %r100, %r60;
	mul.lo.s32 	%r124, %r101, %r60;
	.loc	1 327 51                        // sk06_mlp_down.py:327:51
	cvt.s64.s32 	%rd6, %r123;
	add.s64 	%rd33, %rd52, %rd6;
	cvt.s64.s32 	%rd7, %r124;
	add.s64 	%rd34, %rd52, %rd7;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	setp.gt.s32 	%p1, %r56, 63;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	shl.b32 	%r192, %r3, 4;
	shl.b32 	%r9, %r2, 1;
	and.b32 	%r10, %r9, 48;
	xor.b32 	%r193, %r192, %r10;
	mov.b32 	%r194, global_smem;
	add.s32 	%r21, %r194, %r193;
	selp.b32 	%r22, 16, 0, %p1;
	// begin inline asm
	cp.async.cg.shared.global [ %r21 + 0 ], [ %rd29 + 0 ], 0x10, %r22;
	// end inline asm
	add.s32 	%r23, %r21, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r23 + 0 ], [ %rd30 + 0 ], 0x10, %r22;
	// end inline asm
	add.s32 	%r24, %r21, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r24 + 0 ], [ %rd31 + 0 ], 0x10, %r22;
	// end inline asm
	add.s32 	%r25, %r21, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r25 + 0 ], [ %rd32 + 0 ], 0x10, %r22;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	add.s32 	%r26, %r21, 49152;
	// begin inline asm
	cp.async.cg.shared.global [ %r26 + 0 ], [ %rd33 + 0 ], 0x10, %r22;
	// end inline asm
	add.s32 	%r27, %r21, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r27 + 0 ], [ %rd34 + 0 ], 0x10, %r22;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	setp.gt.s32 	%p2, %r56, 127;
	.loc	1 344 18                        // sk06_mlp_down.py:344:18
	add.s64 	%rd35, %rd29, 64;
	add.s64 	%rd36, %rd30, 64;
	add.s64 	%rd37, %rd31, 64;
	add.s64 	%rd38, %rd32, 64;
	.loc	1 345 18                        // sk06_mlp_down.py:345:18
	add.s64 	%rd39, %rd33, 64;
	add.s64 	%rd40, %rd34, 64;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	bar.sync 	0;
	add.s32 	%r28, %r21, 16384;
	selp.b32 	%r29, 16, 0, %p2;
	// begin inline asm
	cp.async.cg.shared.global [ %r28 + 0 ], [ %rd35 + 0 ], 0x10, %r29;
	// end inline asm
	add.s32 	%r30, %r21, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r30 + 0 ], [ %rd36 + 0 ], 0x10, %r29;
	// end inline asm
	add.s32 	%r31, %r21, 24576;
	// begin inline asm
	cp.async.cg.shared.global [ %r31 + 0 ], [ %rd37 + 0 ], 0x10, %r29;
	// end inline asm
	add.s32 	%r32, %r21, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r32 + 0 ], [ %rd38 + 0 ], 0x10, %r29;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	add.s32 	%r33, %r21, 57344;
	// begin inline asm
	cp.async.cg.shared.global [ %r33 + 0 ], [ %rd39 + 0 ], 0x10, %r29;
	// end inline asm
	add.s32 	%r34, %r21, 61440;
	// begin inline asm
	cp.async.cg.shared.global [ %r34 + 0 ], [ %rd40 + 0 ], 0x10, %r29;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	setp.gt.s32 	%p3, %r56, 191;
	.loc	1 344 18                        // sk06_mlp_down.py:344:18
	add.s64 	%rd41, %rd29, 128;
	add.s64 	%rd42, %rd30, 128;
	add.s64 	%rd43, %rd31, 128;
	add.s64 	%rd44, %rd32, 128;
	.loc	1 345 18                        // sk06_mlp_down.py:345:18
	add.s64 	%rd45, %rd33, 128;
	add.s64 	%rd46, %rd34, 128;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	bar.sync 	0;
	add.s32 	%r35, %r21, 32768;
	selp.b32 	%r36, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r35 + 0 ], [ %rd41 + 0 ], 0x10, %r36;
	// end inline asm
	add.s32 	%r37, %r21, 36864;
	// begin inline asm
	cp.async.cg.shared.global [ %r37 + 0 ], [ %rd42 + 0 ], 0x10, %r36;
	// end inline asm
	add.s32 	%r38, %r21, 40960;
	// begin inline asm
	cp.async.cg.shared.global [ %r38 + 0 ], [ %rd43 + 0 ], 0x10, %r36;
	// end inline asm
	add.s32 	%r39, %r21, 45056;
	// begin inline asm
	cp.async.cg.shared.global [ %r39 + 0 ], [ %rd44 + 0 ], 0x10, %r36;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	add.s32 	%r40, %r21, 65536;
	// begin inline asm
	cp.async.cg.shared.global [ %r40 + 0 ], [ %rd45 + 0 ], 0x10, %r36;
	// end inline asm
	add.s32 	%r41, %r21, 69632;
	// begin inline asm
	cp.async.cg.shared.global [ %r41 + 0 ], [ %rd46 + 0 ], 0x10, %r36;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	@%p1 bra 	$L__BB0_2;
	bra.uni 	$L__BB0_1;
$L__BB0_2:                              // %.lr.ph
	.loc	1 0 23                          // sk06_mlp_down.py:0:23
	ld.param.b32 	%r20, [_sk06_mlp_down_kernel_param_15];
	ld.param.b64 	%rd47, [_sk06_mlp_down_kernel_param_6];
	shl.b32 	%r79, %r5, 1;
	shr.u32 	%r80, %r6, 2;
	or.b32 	%r81, %r80, %r79;
	or.b32 	%r84, %r4, %r81;
	or.b32 	%r85, %r84, 1;
	or.b32 	%r86, %r84, 16;
	or.b32 	%r87, %r84, 17;
	or.b32 	%r88, %r84, 32;
	or.b32 	%r89, %r84, 33;
	or.b32 	%r90, %r84, 48;
	or.b32 	%r91, %r84, 49;
	or.b32 	%r92, %r84, 64;
	or.b32 	%r93, %r84, 65;
	or.b32 	%r94, %r84, 80;
	or.b32 	%r95, %r84, 81;
	or.b32 	%r96, %r84, 96;
	or.b32 	%r97, %r84, 97;
	or.b32 	%r98, %r84, 112;
	or.b32 	%r99, %r84, 113;
	rem.s32 	%r102, %r84, %r16;
	rem.s32 	%r103, %r85, %r16;
	rem.s32 	%r104, %r86, %r16;
	rem.s32 	%r105, %r87, %r16;
	rem.s32 	%r106, %r88, %r16;
	rem.s32 	%r107, %r89, %r16;
	rem.s32 	%r108, %r90, %r16;
	rem.s32 	%r109, %r91, %r16;
	rem.s32 	%r110, %r92, %r16;
	rem.s32 	%r111, %r93, %r16;
	rem.s32 	%r112, %r94, %r16;
	rem.s32 	%r113, %r95, %r16;
	rem.s32 	%r114, %r96, %r16;
	rem.s32 	%r115, %r97, %r16;
	rem.s32 	%r116, %r98, %r16;
	rem.s32 	%r117, %r99, %r16;
	shr.s32 	%r125, %r102, 31;
	shr.u32 	%r126, %r125, 25;
	add.s32 	%r127, %r102, %r126;
	shr.s32 	%r128, %r127, 7;
	shr.s32 	%r129, %r103, 31;
	shr.u32 	%r130, %r129, 25;
	add.s32 	%r131, %r103, %r130;
	shr.s32 	%r132, %r131, 7;
	shr.s32 	%r133, %r104, 31;
	shr.u32 	%r134, %r133, 25;
	add.s32 	%r135, %r104, %r134;
	shr.s32 	%r136, %r135, 7;
	shr.s32 	%r137, %r105, 31;
	shr.u32 	%r138, %r137, 25;
	add.s32 	%r139, %r105, %r138;
	shr.s32 	%r140, %r139, 7;
	shr.s32 	%r141, %r106, 31;
	shr.u32 	%r142, %r141, 25;
	add.s32 	%r143, %r106, %r142;
	shr.s32 	%r144, %r143, 7;
	shr.s32 	%r145, %r107, 31;
	shr.u32 	%r146, %r145, 25;
	add.s32 	%r147, %r107, %r146;
	shr.s32 	%r148, %r147, 7;
	shr.s32 	%r149, %r108, 31;
	shr.u32 	%r150, %r149, 25;
	add.s32 	%r151, %r108, %r150;
	shr.s32 	%r152, %r151, 7;
	shr.s32 	%r153, %r109, 31;
	shr.u32 	%r154, %r153, 25;
	add.s32 	%r155, %r109, %r154;
	shr.s32 	%r156, %r155, 7;
	shr.s32 	%r157, %r110, 31;
	shr.u32 	%r158, %r157, 25;
	add.s32 	%r159, %r110, %r158;
	shr.s32 	%r160, %r159, 7;
	shr.s32 	%r161, %r111, 31;
	shr.u32 	%r162, %r161, 25;
	add.s32 	%r163, %r111, %r162;
	shr.s32 	%r164, %r163, 7;
	shr.s32 	%r165, %r112, 31;
	shr.u32 	%r166, %r165, 25;
	add.s32 	%r167, %r112, %r166;
	shr.s32 	%r168, %r167, 7;
	shr.s32 	%r169, %r113, 31;
	shr.u32 	%r170, %r169, 25;
	add.s32 	%r171, %r113, %r170;
	shr.s32 	%r172, %r171, 7;
	shr.s32 	%r173, %r114, 31;
	shr.u32 	%r174, %r173, 25;
	add.s32 	%r175, %r114, %r174;
	shr.s32 	%r176, %r175, 7;
	shr.s32 	%r177, %r115, 31;
	shr.u32 	%r178, %r177, 25;
	add.s32 	%r179, %r115, %r178;
	shr.s32 	%r180, %r179, 7;
	shr.s32 	%r181, %r116, 31;
	shr.u32 	%r182, %r181, 25;
	add.s32 	%r183, %r116, %r182;
	shr.s32 	%r184, %r183, 7;
	shr.s32 	%r185, %r117, 31;
	shr.u32 	%r186, %r185, 25;
	add.s32 	%r187, %r117, %r186;
	shr.s32 	%r188, %r187, 7;
	mad.wide.s32 	%rd8, %r128, 4, %rd47;
	mad.wide.s32 	%rd9, %r132, 4, %rd47;
	mad.wide.s32 	%rd10, %r136, 4, %rd47;
	mad.wide.s32 	%rd11, %r140, 4, %rd47;
	mad.wide.s32 	%rd12, %r144, 4, %rd47;
	mad.wide.s32 	%rd13, %r148, 4, %rd47;
	mad.wide.s32 	%rd14, %r152, 4, %rd47;
	mad.wide.s32 	%rd15, %r156, 4, %rd47;
	mad.wide.s32 	%rd16, %r160, 4, %rd47;
	mad.wide.s32 	%rd17, %r164, 4, %rd47;
	mad.wide.s32 	%rd18, %r168, 4, %rd47;
	mad.wide.s32 	%rd19, %r172, 4, %rd47;
	mad.wide.s32 	%rd20, %r176, 4, %rd47;
	mad.wide.s32 	%rd21, %r180, 4, %rd47;
	mad.wide.s32 	%rd22, %r184, 4, %rd47;
	mad.wide.s32 	%rd23, %r188, 4, %rd47;
	shr.s32 	%r189, %r56, 31;
	shr.u32 	%r190, %r189, 26;
	add.s32 	%r191, %r56, %r190;
	shr.s32 	%r8, %r191, 6;
	add.s32 	%r11, %r8, -3;
	shl.b32 	%r195, %r7, 6;
	shl.b32 	%r1256, %r2, 4;
	and.b32 	%r196, %r1256, 3072;
	shl.b32 	%r197, %r2, 3;
	and.b32 	%r198, %r197, 48;
	and.b32 	%r1257, %r2, 16;
	or.b32 	%r199, %r195, %r196;
	xor.b32 	%r200, %r198, %r1257;
	or.b32 	%r12, %r199, %r200;
	xor.b32 	%r13, %r12, 32;
	shl.b32 	%r201, %r2, 6;
	and.b32 	%r202, %r201, 448;
	shl.b32 	%r203, %r6, 4;
	or.b32 	%r204, %r202, %r198;
	xor.b32 	%r205, %r204, %r10;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	add.s32 	%r206, %r194, %r203;
	add.s32 	%r14, %r206, %r205;
	add.s64 	%rd53, %rd7, %rd25;
	add.s64 	%rd273, %rd53, 192;
	add.s64 	%rd54, %rd6, %rd25;
	add.s64 	%rd272, %rd54, 192;
	add.s64 	%rd55, %rd4, %rd24;
	add.s64 	%rd271, %rd55, 192;
	add.s64 	%rd56, %rd3, %rd24;
	add.s64 	%rd270, %rd56, 192;
	add.s64 	%rd57, %rd2, %rd24;
	add.s64 	%rd269, %rd57, 192;
	add.s64 	%rd58, %rd1, %rd24;
	add.s64 	%rd268, %rd58, 192;
	mov.b32 	%r1258, 0f00000000;
	mov.b32 	%r207, 0;
	mov.b32 	%r1254, 2;
	mov.b32 	%r1253, -1;
	mov.b32 	%r1255, %r207;
	mov.b32 	%r1259, %r1258;
	mov.b32 	%r1260, %r1258;
	mov.b32 	%r1261, %r1258;
	mov.b32 	%r1262, %r1258;
	mov.b32 	%r1263, %r1258;
	mov.b32 	%r1264, %r1258;
	mov.b32 	%r1265, %r1258;
	mov.b32 	%r1266, %r1258;
	mov.b32 	%r1267, %r1258;
	mov.b32 	%r1268, %r1258;
	mov.b32 	%r1269, %r1258;
	mov.b32 	%r1270, %r1258;
	mov.b32 	%r1271, %r1258;
	mov.b32 	%r1272, %r1258;
	mov.b32 	%r1273, %r1258;
	mov.b32 	%r1274, %r1258;
	mov.b32 	%r1275, %r1258;
	mov.b32 	%r1276, %r1258;
	mov.b32 	%r1277, %r1258;
	mov.b32 	%r1278, %r1258;
	mov.b32 	%r1279, %r1258;
	mov.b32 	%r1280, %r1258;
	mov.b32 	%r1281, %r1258;
	mov.b32 	%r1282, %r1258;
	mov.b32 	%r1283, %r1258;
	mov.b32 	%r1284, %r1258;
	mov.b32 	%r1285, %r1258;
	mov.b32 	%r1286, %r1258;
	mov.b32 	%r1287, %r1258;
	mov.b32 	%r1288, %r1258;
	mov.b32 	%r1289, %r1258;
	mov.b32 	%r1290, %r1258;
	mov.b32 	%r1291, %r1258;
	mov.b32 	%r1292, %r1258;
	mov.b32 	%r1293, %r1258;
	mov.b32 	%r1294, %r1258;
	mov.b32 	%r1295, %r1258;
	mov.b32 	%r1296, %r1258;
	mov.b32 	%r1297, %r1258;
	mov.b32 	%r1298, %r1258;
	mov.b32 	%r1299, %r1258;
	mov.b32 	%r1300, %r1258;
	mov.b32 	%r1301, %r1258;
	mov.b32 	%r1302, %r1258;
	mov.b32 	%r1303, %r1258;
	mov.b32 	%r1304, %r1258;
	mov.b32 	%r1305, %r1258;
	mov.b32 	%r1306, %r1258;
	mov.b32 	%r1307, %r1258;
	mov.b32 	%r1308, %r1258;
	mov.b32 	%r1309, %r1258;
	mov.b32 	%r1310, %r1258;
	mov.b32 	%r1311, %r1258;
	mov.b32 	%r1312, %r1258;
	mov.b32 	%r1313, %r1258;
	mov.b32 	%r1314, %r1258;
	mov.b32 	%r1315, %r1258;
	mov.b32 	%r1316, %r1258;
	mov.b32 	%r1317, %r1258;
	mov.b32 	%r1318, %r1258;
	mov.b32 	%r1319, %r1258;
	mov.b32 	%r1320, %r1258;
	mov.b32 	%r1321, %r1258;
	mov.b32 	%r1322, %r1258;
	mov.b32 	%r1323, %r1258;
	mov.b32 	%r1324, %r1258;
	mov.b32 	%r1325, %r1258;
	mov.b32 	%r1326, %r1258;
	mov.b32 	%r1327, %r1258;
	mov.b32 	%r1328, %r1258;
	mov.b32 	%r1329, %r1258;
	mov.b32 	%r1330, %r1258;
	mov.b32 	%r1331, %r1258;
	mov.b32 	%r1332, %r1258;
	mov.b32 	%r1333, %r1258;
	mov.b32 	%r1334, %r1258;
	mov.b32 	%r1335, %r1258;
	mov.b32 	%r1336, %r1258;
	mov.b32 	%r1337, %r1258;
	mov.b32 	%r1338, %r1258;
	mov.b32 	%r1339, %r1258;
	mov.b32 	%r1340, %r1258;
	mov.b32 	%r1341, %r1258;
	mov.b32 	%r1342, %r1258;
	mov.b32 	%r1343, %r1258;
	mov.b32 	%r1344, %r1258;
	mov.b32 	%r1345, %r1258;
	mov.b32 	%r1346, %r1258;
	mov.b32 	%r1347, %r1258;
	mov.b32 	%r1348, %r1258;
	mov.b32 	%r1349, %r1258;
	mov.b32 	%r1350, %r1258;
	mov.b32 	%r1351, %r1258;
	mov.b32 	%r1352, %r1258;
	mov.b32 	%r1353, %r1258;
	mov.b32 	%r1354, %r1258;
	mov.b32 	%r1355, %r1258;
	mov.b32 	%r1356, %r1258;
	mov.b32 	%r1357, %r1258;
	mov.b32 	%r1358, %r1258;
	mov.b32 	%r1359, %r1258;
	mov.b32 	%r1360, %r1258;
	mov.b32 	%r1361, %r1258;
	mov.b32 	%r1362, %r1258;
	mov.b32 	%r1363, %r1258;
	mov.b32 	%r1364, %r1258;
	mov.b32 	%r1365, %r1258;
	mov.b32 	%r1366, %r1258;
	mov.b32 	%r1367, %r1258;
	mov.b32 	%r1368, %r1258;
	mov.b32 	%r1369, %r1258;
	mov.b32 	%r1370, %r1258;
	mov.b32 	%r1371, %r1258;
	mov.b32 	%r1372, %r1258;
	mov.b32 	%r1373, %r1258;
	mov.b32 	%r1374, %r1258;
	mov.b32 	%r1375, %r1258;
	mov.b32 	%r1376, %r1258;
	mov.b32 	%r1377, %r1258;
	mov.b32 	%r1378, %r1258;
	mov.b32 	%r1379, %r1258;
	mov.b32 	%r1380, %r1258;
	mov.b32 	%r1381, %r1258;
	mov.b32 	%r1382, %r1258;
	mov.b32 	%r1383, %r1258;
	mov.b32 	%r1384, %r1258;
	mov.b32 	%r1385, %r1258;
$L__BB0_3:                              // =>This Inner Loop Header: Depth=1
	setp.lt.s32 	%p4, %r1255, %r11;
	add.s32 	%r423, %r1253, 1;
	setp.gt.s32 	%p5, %r423, 2;
	selp.b32 	%r1253, 0, %r423, %p5;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	cp.async.wait_group 	4;
	bar.sync 	0;
	shl.b32 	%r424, %r1253, 14;
	add.s32 	%r425, %r194, %r424;
	add.s32 	%r426, %r425, %r12;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r208, %r209, %r210, %r211}, [%r426];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r228, %r229, %r230, %r231}, [%r426+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r232, %r233, %r234, %r235}, [%r426+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r236, %r237, %r238, %r239}, [%r426+12288];
	add.s32 	%r427, %r425, %r13;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r244, %r245, %r246, %r247}, [%r427];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r296, %r297, %r298, %r299}, [%r427+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r332, %r333, %r334, %r335}, [%r427+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r368, %r369, %r370, %r371}, [%r427+12288];
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	shl.b32 	%r428, %r1253, 13;
	add.s32 	%r429, %r14, %r428;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r212, %r213, %r248, %r249}, [%r429+49152];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r214, %r215, %r254, %r255}, [%r429+50176];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r216, %r217, %r260, %r261}, [%r429+51200];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r218, %r219, %r266, %r267}, [%r429+52224];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r220, %r221, %r272, %r273}, [%r429+53248];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r222, %r223, %r278, %r279}, [%r429+54272];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r224, %r225, %r284, %r285}, [%r429+55296];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r226, %r227, %r290, %r291}, [%r429+56320];
	.loc	1 338 36                        // sk06_mlp_down.py:338:36
	mov.b32 	%r240, %r207;
	mov.b32 	%r241, %r207;
	mov.b32 	%r242, %r207;
	mov.b32 	%r243, %r207;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r240, %r241, %r242, %r243 }, { %r208, %r209, %r210, %r211 }, { %r212, %r213 }, { %r240, %r241, %r242, %r243 };
	// end inline asm
	mov.b32 	%r250, %r207;
	mov.b32 	%r251, %r207;
	mov.b32 	%r252, %r207;
	mov.b32 	%r253, %r207;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r250, %r251, %r252, %r253 }, { %r208, %r209, %r210, %r211 }, { %r214, %r215 }, { %r250, %r251, %r252, %r253 };
	// end inline asm
	mov.b32 	%r256, %r207;
	mov.b32 	%r257, %r207;
	mov.b32 	%r258, %r207;
	mov.b32 	%r259, %r207;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r256, %r257, %r258, %r259 }, { %r208, %r209, %r210, %r211 }, { %r216, %r217 }, { %r256, %r257, %r258, %r259 };
	// end inline asm
	mov.b32 	%r262, %r207;
	mov.b32 	%r263, %r207;
	mov.b32 	%r264, %r207;
	mov.b32 	%r265, %r207;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r262, %r263, %r264, %r265 }, { %r208, %r209, %r210, %r211 }, { %r218, %r219 }, { %r262, %r263, %r264, %r265 };
	// end inline asm
	mov.b32 	%r268, %r207;
	mov.b32 	%r269, %r207;
	mov.b32 	%r270, %r207;
	mov.b32 	%r271, %r207;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r268, %r269, %r270, %r271 }, { %r208, %r209, %r210, %r211 }, { %r220, %r221 }, { %r268, %r269, %r270, %r271 };
	// end inline asm
	mov.b32 	%r274, %r207;
	mov.b32 	%r275, %r207;
	mov.b32 	%r276, %r207;
	mov.b32 	%r277, %r207;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r274, %r275, %r276, %r277 }, { %r208, %r209, %r210, %r211 }, { %r222, %r223 }, { %r274, %r275, %r276, %r277 };
	// end inline asm
	mov.b32 	%r280, %r207;
	mov.b32 	%r281, %r207;
	mov.b32 	%r282, %r207;
	mov.b32 	%r283, %r207;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r280, %r281, %r282, %r283 }, { %r208, %r209, %r210, %r211 }, { %r224, %r225 }, { %r280, %r281, %r282, %r283 };
	// end inline asm
	mov.b32 	%r286, %r207;
	mov.b32 	%r287, %r207;
	mov.b32 	%r288, %r207;
	mov.b32 	%r289, %r207;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r286, %r287, %r288, %r289 }, { %r208, %r209, %r210, %r211 }, { %r226, %r227 }, { %r286, %r287, %r288, %r289 };
	// end inline asm
	mov.b32 	%r292, %r207;
	mov.b32 	%r293, %r207;
	mov.b32 	%r294, %r207;
	mov.b32 	%r295, %r207;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r292, %r293, %r294, %r295 }, { %r228, %r229, %r230, %r231 }, { %r212, %r213 }, { %r292, %r293, %r294, %r295 };
	// end inline asm
	mov.b32 	%r300, %r207;
	mov.b32 	%r301, %r207;
	mov.b32 	%r302, %r207;
	mov.b32 	%r303, %r207;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r300, %r301, %r302, %r303 }, { %r228, %r229, %r230, %r231 }, { %r214, %r215 }, { %r300, %r301, %r302, %r303 };
	// end inline asm
	mov.b32 	%r304, %r207;
	mov.b32 	%r305, %r207;
	mov.b32 	%r306, %r207;
	mov.b32 	%r307, %r207;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r304, %r305, %r306, %r307 }, { %r228, %r229, %r230, %r231 }, { %r216, %r217 }, { %r304, %r305, %r306, %r307 };
	// end inline asm
	mov.b32 	%r308, %r207;
	mov.b32 	%r309, %r207;
	mov.b32 	%r310, %r207;
	mov.b32 	%r311, %r207;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r308, %r309, %r310, %r311 }, { %r228, %r229, %r230, %r231 }, { %r218, %r219 }, { %r308, %r309, %r310, %r311 };
	// end inline asm
	mov.b32 	%r312, %r207;
	mov.b32 	%r313, %r207;
	mov.b32 	%r314, %r207;
	mov.b32 	%r315, %r207;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r312, %r313, %r314, %r315 }, { %r228, %r229, %r230, %r231 }, { %r220, %r221 }, { %r312, %r313, %r314, %r315 };
	// end inline asm
	mov.b32 	%r316, %r207;
	mov.b32 	%r317, %r207;
	mov.b32 	%r318, %r207;
	mov.b32 	%r319, %r207;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r316, %r317, %r318, %r319 }, { %r228, %r229, %r230, %r231 }, { %r222, %r223 }, { %r316, %r317, %r318, %r319 };
	// end inline asm
	mov.b32 	%r320, %r207;
	mov.b32 	%r321, %r207;
	mov.b32 	%r322, %r207;
	mov.b32 	%r323, %r207;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r320, %r321, %r322, %r323 }, { %r228, %r229, %r230, %r231 }, { %r224, %r225 }, { %r320, %r321, %r322, %r323 };
	// end inline asm
	mov.b32 	%r324, %r207;
	mov.b32 	%r325, %r207;
	mov.b32 	%r326, %r207;
	mov.b32 	%r327, %r207;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r324, %r325, %r326, %r327 }, { %r228, %r229, %r230, %r231 }, { %r226, %r227 }, { %r324, %r325, %r326, %r327 };
	// end inline asm
	mov.b32 	%r328, %r207;
	mov.b32 	%r329, %r207;
	mov.b32 	%r330, %r207;
	mov.b32 	%r331, %r207;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r328, %r329, %r330, %r331 }, { %r232, %r233, %r234, %r235 }, { %r212, %r213 }, { %r328, %r329, %r330, %r331 };
	// end inline asm
	mov.b32 	%r336, %r207;
	mov.b32 	%r337, %r207;
	mov.b32 	%r338, %r207;
	mov.b32 	%r339, %r207;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r336, %r337, %r338, %r339 }, { %r232, %r233, %r234, %r235 }, { %r214, %r215 }, { %r336, %r337, %r338, %r339 };
	// end inline asm
	mov.b32 	%r340, %r207;
	mov.b32 	%r341, %r207;
	mov.b32 	%r342, %r207;
	mov.b32 	%r343, %r207;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r340, %r341, %r342, %r343 }, { %r232, %r233, %r234, %r235 }, { %r216, %r217 }, { %r340, %r341, %r342, %r343 };
	// end inline asm
	mov.b32 	%r344, %r207;
	mov.b32 	%r345, %r207;
	mov.b32 	%r346, %r207;
	mov.b32 	%r347, %r207;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r344, %r345, %r346, %r347 }, { %r232, %r233, %r234, %r235 }, { %r218, %r219 }, { %r344, %r345, %r346, %r347 };
	// end inline asm
	mov.b32 	%r348, %r207;
	mov.b32 	%r349, %r207;
	mov.b32 	%r350, %r207;
	mov.b32 	%r351, %r207;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r348, %r349, %r350, %r351 }, { %r232, %r233, %r234, %r235 }, { %r220, %r221 }, { %r348, %r349, %r350, %r351 };
	// end inline asm
	mov.b32 	%r352, %r207;
	mov.b32 	%r353, %r207;
	mov.b32 	%r354, %r207;
	mov.b32 	%r355, %r207;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r352, %r353, %r354, %r355 }, { %r232, %r233, %r234, %r235 }, { %r222, %r223 }, { %r352, %r353, %r354, %r355 };
	// end inline asm
	mov.b32 	%r356, %r207;
	mov.b32 	%r357, %r207;
	mov.b32 	%r358, %r207;
	mov.b32 	%r359, %r207;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r356, %r357, %r358, %r359 }, { %r232, %r233, %r234, %r235 }, { %r224, %r225 }, { %r356, %r357, %r358, %r359 };
	// end inline asm
	mov.b32 	%r360, %r207;
	mov.b32 	%r361, %r207;
	mov.b32 	%r362, %r207;
	mov.b32 	%r363, %r207;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r360, %r361, %r362, %r363 }, { %r232, %r233, %r234, %r235 }, { %r226, %r227 }, { %r360, %r361, %r362, %r363 };
	// end inline asm
	mov.b32 	%r364, %r207;
	mov.b32 	%r365, %r207;
	mov.b32 	%r366, %r207;
	mov.b32 	%r367, %r207;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r364, %r365, %r366, %r367 }, { %r236, %r237, %r238, %r239 }, { %r212, %r213 }, { %r364, %r365, %r366, %r367 };
	// end inline asm
	mov.b32 	%r372, %r207;
	mov.b32 	%r373, %r207;
	mov.b32 	%r374, %r207;
	mov.b32 	%r375, %r207;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r372, %r373, %r374, %r375 }, { %r236, %r237, %r238, %r239 }, { %r214, %r215 }, { %r372, %r373, %r374, %r375 };
	// end inline asm
	mov.b32 	%r376, %r207;
	mov.b32 	%r377, %r207;
	mov.b32 	%r378, %r207;
	mov.b32 	%r379, %r207;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r376, %r377, %r378, %r379 }, { %r236, %r237, %r238, %r239 }, { %r216, %r217 }, { %r376, %r377, %r378, %r379 };
	// end inline asm
	mov.b32 	%r380, %r207;
	mov.b32 	%r381, %r207;
	mov.b32 	%r382, %r207;
	mov.b32 	%r383, %r207;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r380, %r381, %r382, %r383 }, { %r236, %r237, %r238, %r239 }, { %r218, %r219 }, { %r380, %r381, %r382, %r383 };
	// end inline asm
	mov.b32 	%r384, %r207;
	mov.b32 	%r385, %r207;
	mov.b32 	%r386, %r207;
	mov.b32 	%r387, %r207;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r384, %r385, %r386, %r387 }, { %r236, %r237, %r238, %r239 }, { %r220, %r221 }, { %r384, %r385, %r386, %r387 };
	// end inline asm
	mov.b32 	%r388, %r207;
	mov.b32 	%r389, %r207;
	mov.b32 	%r390, %r207;
	mov.b32 	%r391, %r207;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r388, %r389, %r390, %r391 }, { %r236, %r237, %r238, %r239 }, { %r222, %r223 }, { %r388, %r389, %r390, %r391 };
	// end inline asm
	mov.b32 	%r392, %r207;
	mov.b32 	%r393, %r207;
	mov.b32 	%r394, %r207;
	mov.b32 	%r395, %r207;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r392, %r393, %r394, %r395 }, { %r236, %r237, %r238, %r239 }, { %r224, %r225 }, { %r392, %r393, %r394, %r395 };
	// end inline asm
	mov.b32 	%r399, %r207;
	mov.b32 	%r396, %r207;
	mov.b32 	%r397, %r207;
	mov.b32 	%r398, %r207;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r396, %r397, %r398, %r399 }, { %r236, %r237, %r238, %r239 }, { %r226, %r227 }, { %r396, %r397, %r398, %r399 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r240, %r241, %r242, %r243 }, { %r244, %r245, %r246, %r247 }, { %r248, %r249 }, { %r240, %r241, %r242, %r243 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r250, %r251, %r252, %r253 }, { %r244, %r245, %r246, %r247 }, { %r254, %r255 }, { %r250, %r251, %r252, %r253 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r256, %r257, %r258, %r259 }, { %r244, %r245, %r246, %r247 }, { %r260, %r261 }, { %r256, %r257, %r258, %r259 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r262, %r263, %r264, %r265 }, { %r244, %r245, %r246, %r247 }, { %r266, %r267 }, { %r262, %r263, %r264, %r265 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r268, %r269, %r270, %r271 }, { %r244, %r245, %r246, %r247 }, { %r272, %r273 }, { %r268, %r269, %r270, %r271 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r274, %r275, %r276, %r277 }, { %r244, %r245, %r246, %r247 }, { %r278, %r279 }, { %r274, %r275, %r276, %r277 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r280, %r281, %r282, %r283 }, { %r244, %r245, %r246, %r247 }, { %r284, %r285 }, { %r280, %r281, %r282, %r283 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r286, %r287, %r288, %r289 }, { %r244, %r245, %r246, %r247 }, { %r290, %r291 }, { %r286, %r287, %r288, %r289 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r292, %r293, %r294, %r295 }, { %r296, %r297, %r298, %r299 }, { %r248, %r249 }, { %r292, %r293, %r294, %r295 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r300, %r301, %r302, %r303 }, { %r296, %r297, %r298, %r299 }, { %r254, %r255 }, { %r300, %r301, %r302, %r303 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r304, %r305, %r306, %r307 }, { %r296, %r297, %r298, %r299 }, { %r260, %r261 }, { %r304, %r305, %r306, %r307 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r308, %r309, %r310, %r311 }, { %r296, %r297, %r298, %r299 }, { %r266, %r267 }, { %r308, %r309, %r310, %r311 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r312, %r313, %r314, %r315 }, { %r296, %r297, %r298, %r299 }, { %r272, %r273 }, { %r312, %r313, %r314, %r315 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r316, %r317, %r318, %r319 }, { %r296, %r297, %r298, %r299 }, { %r278, %r279 }, { %r316, %r317, %r318, %r319 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r320, %r321, %r322, %r323 }, { %r296, %r297, %r298, %r299 }, { %r284, %r285 }, { %r320, %r321, %r322, %r323 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r324, %r325, %r326, %r327 }, { %r296, %r297, %r298, %r299 }, { %r290, %r291 }, { %r324, %r325, %r326, %r327 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r328, %r329, %r330, %r331 }, { %r332, %r333, %r334, %r335 }, { %r248, %r249 }, { %r328, %r329, %r330, %r331 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r336, %r337, %r338, %r339 }, { %r332, %r333, %r334, %r335 }, { %r254, %r255 }, { %r336, %r337, %r338, %r339 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r340, %r341, %r342, %r343 }, { %r332, %r333, %r334, %r335 }, { %r260, %r261 }, { %r340, %r341, %r342, %r343 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r344, %r345, %r346, %r347 }, { %r332, %r333, %r334, %r335 }, { %r266, %r267 }, { %r344, %r345, %r346, %r347 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r348, %r349, %r350, %r351 }, { %r332, %r333, %r334, %r335 }, { %r272, %r273 }, { %r348, %r349, %r350, %r351 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r352, %r353, %r354, %r355 }, { %r332, %r333, %r334, %r335 }, { %r278, %r279 }, { %r352, %r353, %r354, %r355 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r356, %r357, %r358, %r359 }, { %r332, %r333, %r334, %r335 }, { %r284, %r285 }, { %r356, %r357, %r358, %r359 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r360, %r361, %r362, %r363 }, { %r332, %r333, %r334, %r335 }, { %r290, %r291 }, { %r360, %r361, %r362, %r363 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r364, %r365, %r366, %r367 }, { %r368, %r369, %r370, %r371 }, { %r248, %r249 }, { %r364, %r365, %r366, %r367 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r372, %r373, %r374, %r375 }, { %r368, %r369, %r370, %r371 }, { %r254, %r255 }, { %r372, %r373, %r374, %r375 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r376, %r377, %r378, %r379 }, { %r368, %r369, %r370, %r371 }, { %r260, %r261 }, { %r376, %r377, %r378, %r379 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r380, %r381, %r382, %r383 }, { %r368, %r369, %r370, %r371 }, { %r266, %r267 }, { %r380, %r381, %r382, %r383 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r384, %r385, %r386, %r387 }, { %r368, %r369, %r370, %r371 }, { %r272, %r273 }, { %r384, %r385, %r386, %r387 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r388, %r389, %r390, %r391 }, { %r368, %r369, %r370, %r371 }, { %r278, %r279 }, { %r388, %r389, %r390, %r391 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r392, %r393, %r394, %r395 }, { %r368, %r369, %r370, %r371 }, { %r284, %r285 }, { %r392, %r393, %r394, %r395 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r396, %r397, %r398, %r399 }, { %r368, %r369, %r370, %r371 }, { %r290, %r291 }, { %r396, %r397, %r398, %r399 };
	// end inline asm
	.loc	1 340 54                        // sk06_mlp_down.py:340:54
	bfe.u32 	%r430, %r1255, 1, 25;
	.loc	1 340 69                        // sk06_mlp_down.py:340:69
	mul.lo.s32 	%r431, %r430, %r20;
	.loc	1 340 37                        // sk06_mlp_down.py:340:37
	mul.wide.s32 	%rd81, %r431, 4;
	add.s64 	%rd59, %rd8, %rd81;
	add.s64 	%rd60, %rd9, %rd81;
	add.s64 	%rd61, %rd10, %rd81;
	add.s64 	%rd62, %rd11, %rd81;
	add.s64 	%rd63, %rd12, %rd81;
	add.s64 	%rd64, %rd13, %rd81;
	add.s64 	%rd65, %rd14, %rd81;
	add.s64 	%rd66, %rd15, %rd81;
	add.s64 	%rd67, %rd16, %rd81;
	add.s64 	%rd68, %rd17, %rd81;
	add.s64 	%rd69, %rd18, %rd81;
	add.s64 	%rd70, %rd19, %rd81;
	add.s64 	%rd71, %rd20, %rd81;
	add.s64 	%rd72, %rd21, %rd81;
	add.s64 	%rd73, %rd22, %rd81;
	add.s64 	%rd74, %rd23, %rd81;
	.loc	1 340 27                        // sk06_mlp_down.py:340:27
	// begin inline asm
	mov.u32 %r400, 0x0;
	ld.global.b32 { %r400 }, [ %rd59 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r401, 0x0;
	ld.global.b32 { %r401 }, [ %rd60 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r402, 0x0;
	ld.global.b32 { %r402 }, [ %rd61 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r403, 0x0;
	ld.global.b32 { %r403 }, [ %rd62 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r404, 0x0;
	ld.global.b32 { %r404 }, [ %rd63 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r405, 0x0;
	ld.global.b32 { %r405 }, [ %rd64 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r406, 0x0;
	ld.global.b32 { %r406 }, [ %rd65 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r407, 0x0;
	ld.global.b32 { %r407 }, [ %rd66 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r408, 0x0;
	ld.global.b32 { %r408 }, [ %rd67 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r409, 0x0;
	ld.global.b32 { %r409 }, [ %rd68 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r410, 0x0;
	ld.global.b32 { %r410 }, [ %rd69 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r411, 0x0;
	ld.global.b32 { %r411 }, [ %rd70 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r412, 0x0;
	ld.global.b32 { %r412 }, [ %rd71 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r413, 0x0;
	ld.global.b32 { %r413 }, [ %rd72 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r414, 0x0;
	ld.global.b32 { %r414 }, [ %rd73 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r415, 0x0;
	ld.global.b32 { %r415 }, [ %rd74 + 0 ];
	// end inline asm
	.loc	1 341 23                        // sk06_mlp_down.py:341:23
	cvt.rn.f32.s32 	%r432, %r399;
	cvt.rn.f32.s32 	%r433, %r398;
	cvt.rn.f32.s32 	%r434, %r397;
	cvt.rn.f32.s32 	%r435, %r396;
	cvt.rn.f32.s32 	%r436, %r395;
	cvt.rn.f32.s32 	%r437, %r394;
	cvt.rn.f32.s32 	%r438, %r393;
	cvt.rn.f32.s32 	%r439, %r392;
	cvt.rn.f32.s32 	%r440, %r391;
	cvt.rn.f32.s32 	%r441, %r390;
	cvt.rn.f32.s32 	%r442, %r389;
	cvt.rn.f32.s32 	%r443, %r388;
	cvt.rn.f32.s32 	%r444, %r387;
	cvt.rn.f32.s32 	%r445, %r386;
	cvt.rn.f32.s32 	%r446, %r385;
	cvt.rn.f32.s32 	%r447, %r384;
	cvt.rn.f32.s32 	%r448, %r383;
	cvt.rn.f32.s32 	%r449, %r382;
	cvt.rn.f32.s32 	%r450, %r381;
	cvt.rn.f32.s32 	%r451, %r380;
	cvt.rn.f32.s32 	%r452, %r379;
	cvt.rn.f32.s32 	%r453, %r378;
	cvt.rn.f32.s32 	%r454, %r377;
	cvt.rn.f32.s32 	%r455, %r376;
	cvt.rn.f32.s32 	%r456, %r375;
	cvt.rn.f32.s32 	%r457, %r287;
	cvt.rn.f32.s32 	%r458, %r289;
	cvt.rn.f32.s32 	%r459, %r325;
	cvt.rn.f32.s32 	%r460, %r327;
	cvt.rn.f32.s32 	%r461, %r361;
	cvt.rn.f32.s32 	%r462, %r363;
	cvt.rn.f32.s32 	%r463, %r374;
	cvt.rn.f32.s32 	%r464, %r286;
	cvt.rn.f32.s32 	%r465, %r288;
	cvt.rn.f32.s32 	%r466, %r324;
	cvt.rn.f32.s32 	%r467, %r326;
	cvt.rn.f32.s32 	%r468, %r360;
	cvt.rn.f32.s32 	%r469, %r362;
	cvt.rn.f32.s32 	%r470, %r373;
	cvt.rn.f32.s32 	%r471, %r281;
	cvt.rn.f32.s32 	%r472, %r283;
	cvt.rn.f32.s32 	%r473, %r321;
	cvt.rn.f32.s32 	%r474, %r323;
	cvt.rn.f32.s32 	%r475, %r357;
	cvt.rn.f32.s32 	%r476, %r359;
	cvt.rn.f32.s32 	%r477, %r372;
	cvt.rn.f32.s32 	%r478, %r280;
	cvt.rn.f32.s32 	%r479, %r282;
	cvt.rn.f32.s32 	%r480, %r320;
	cvt.rn.f32.s32 	%r481, %r322;
	cvt.rn.f32.s32 	%r482, %r356;
	cvt.rn.f32.s32 	%r483, %r358;
	cvt.rn.f32.s32 	%r484, %r367;
	cvt.rn.f32.s32 	%r485, %r275;
	cvt.rn.f32.s32 	%r486, %r277;
	cvt.rn.f32.s32 	%r487, %r317;
	cvt.rn.f32.s32 	%r488, %r319;
	cvt.rn.f32.s32 	%r489, %r353;
	cvt.rn.f32.s32 	%r490, %r355;
	cvt.rn.f32.s32 	%r491, %r366;
	cvt.rn.f32.s32 	%r492, %r274;
	cvt.rn.f32.s32 	%r493, %r276;
	cvt.rn.f32.s32 	%r494, %r316;
	cvt.rn.f32.s32 	%r495, %r318;
	cvt.rn.f32.s32 	%r496, %r352;
	cvt.rn.f32.s32 	%r497, %r354;
	cvt.rn.f32.s32 	%r498, %r365;
	cvt.rn.f32.s32 	%r499, %r269;
	cvt.rn.f32.s32 	%r500, %r271;
	cvt.rn.f32.s32 	%r501, %r313;
	cvt.rn.f32.s32 	%r502, %r315;
	cvt.rn.f32.s32 	%r503, %r349;
	cvt.rn.f32.s32 	%r504, %r351;
	cvt.rn.f32.s32 	%r505, %r364;
	cvt.rn.f32.s32 	%r506, %r268;
	cvt.rn.f32.s32 	%r507, %r270;
	cvt.rn.f32.s32 	%r508, %r312;
	cvt.rn.f32.s32 	%r509, %r314;
	cvt.rn.f32.s32 	%r510, %r348;
	cvt.rn.f32.s32 	%r511, %r350;
	cvt.rn.f32.s32 	%r512, %r263;
	cvt.rn.f32.s32 	%r513, %r265;
	cvt.rn.f32.s32 	%r514, %r309;
	cvt.rn.f32.s32 	%r515, %r311;
	cvt.rn.f32.s32 	%r516, %r345;
	cvt.rn.f32.s32 	%r517, %r347;
	cvt.rn.f32.s32 	%r518, %r262;
	cvt.rn.f32.s32 	%r519, %r264;
	cvt.rn.f32.s32 	%r520, %r308;
	cvt.rn.f32.s32 	%r521, %r310;
	cvt.rn.f32.s32 	%r522, %r344;
	cvt.rn.f32.s32 	%r523, %r346;
	cvt.rn.f32.s32 	%r524, %r257;
	cvt.rn.f32.s32 	%r525, %r259;
	cvt.rn.f32.s32 	%r526, %r305;
	cvt.rn.f32.s32 	%r527, %r307;
	cvt.rn.f32.s32 	%r528, %r341;
	cvt.rn.f32.s32 	%r529, %r343;
	cvt.rn.f32.s32 	%r530, %r256;
	cvt.rn.f32.s32 	%r531, %r258;
	cvt.rn.f32.s32 	%r532, %r304;
	cvt.rn.f32.s32 	%r533, %r306;
	cvt.rn.f32.s32 	%r534, %r340;
	cvt.rn.f32.s32 	%r535, %r342;
	cvt.rn.f32.s32 	%r536, %r251;
	cvt.rn.f32.s32 	%r537, %r253;
	cvt.rn.f32.s32 	%r538, %r301;
	cvt.rn.f32.s32 	%r539, %r303;
	cvt.rn.f32.s32 	%r540, %r337;
	cvt.rn.f32.s32 	%r541, %r339;
	cvt.rn.f32.s32 	%r542, %r250;
	cvt.rn.f32.s32 	%r543, %r252;
	cvt.rn.f32.s32 	%r544, %r300;
	cvt.rn.f32.s32 	%r545, %r302;
	cvt.rn.f32.s32 	%r546, %r336;
	cvt.rn.f32.s32 	%r547, %r338;
	cvt.rn.f32.s32 	%r548, %r241;
	cvt.rn.f32.s32 	%r549, %r243;
	cvt.rn.f32.s32 	%r550, %r293;
	cvt.rn.f32.s32 	%r551, %r295;
	cvt.rn.f32.s32 	%r552, %r329;
	cvt.rn.f32.s32 	%r553, %r331;
	cvt.rn.f32.s32 	%r554, %r240;
	cvt.rn.f32.s32 	%r555, %r242;
	cvt.rn.f32.s32 	%r556, %r292;
	cvt.rn.f32.s32 	%r557, %r294;
	cvt.rn.f32.s32 	%r558, %r328;
	cvt.rn.f32.s32 	%r559, %r330;
	.loc	1 341 19                        // sk06_mlp_down.py:341:19
	fma.rn.f32 	%r1324, %r400, %r559, %r1324;
	fma.rn.f32 	%r1322, %r400, %r558, %r1322;
	fma.rn.f32 	%r1292, %r400, %r557, %r1292;
	fma.rn.f32 	%r1290, %r400, %r556, %r1290;
	fma.rn.f32 	%r1260, %r400, %r555, %r1260;
	fma.rn.f32 	%r1258, %r400, %r554, %r1258;
	fma.rn.f32 	%r1325, %r401, %r553, %r1325;
	fma.rn.f32 	%r1323, %r401, %r552, %r1323;
	fma.rn.f32 	%r1293, %r401, %r551, %r1293;
	fma.rn.f32 	%r1291, %r401, %r550, %r1291;
	fma.rn.f32 	%r1261, %r401, %r549, %r1261;
	fma.rn.f32 	%r1259, %r401, %r548, %r1259;
	fma.rn.f32 	%r1328, %r402, %r547, %r1328;
	fma.rn.f32 	%r1326, %r402, %r546, %r1326;
	fma.rn.f32 	%r1296, %r402, %r545, %r1296;
	fma.rn.f32 	%r1294, %r402, %r544, %r1294;
	fma.rn.f32 	%r1264, %r402, %r543, %r1264;
	fma.rn.f32 	%r1262, %r402, %r542, %r1262;
	fma.rn.f32 	%r1329, %r403, %r541, %r1329;
	fma.rn.f32 	%r1327, %r403, %r540, %r1327;
	fma.rn.f32 	%r1297, %r403, %r539, %r1297;
	fma.rn.f32 	%r1295, %r403, %r538, %r1295;
	fma.rn.f32 	%r1265, %r403, %r537, %r1265;
	fma.rn.f32 	%r1263, %r403, %r536, %r1263;
	fma.rn.f32 	%r1332, %r404, %r535, %r1332;
	fma.rn.f32 	%r1330, %r404, %r534, %r1330;
	fma.rn.f32 	%r1300, %r404, %r533, %r1300;
	fma.rn.f32 	%r1298, %r404, %r532, %r1298;
	fma.rn.f32 	%r1268, %r404, %r531, %r1268;
	fma.rn.f32 	%r1266, %r404, %r530, %r1266;
	fma.rn.f32 	%r1333, %r405, %r529, %r1333;
	fma.rn.f32 	%r1331, %r405, %r528, %r1331;
	fma.rn.f32 	%r1301, %r405, %r527, %r1301;
	fma.rn.f32 	%r1299, %r405, %r526, %r1299;
	fma.rn.f32 	%r1269, %r405, %r525, %r1269;
	fma.rn.f32 	%r1267, %r405, %r524, %r1267;
	fma.rn.f32 	%r1336, %r406, %r523, %r1336;
	fma.rn.f32 	%r1334, %r406, %r522, %r1334;
	fma.rn.f32 	%r1304, %r406, %r521, %r1304;
	fma.rn.f32 	%r1302, %r406, %r520, %r1302;
	fma.rn.f32 	%r1272, %r406, %r519, %r1272;
	fma.rn.f32 	%r1270, %r406, %r518, %r1270;
	fma.rn.f32 	%r1337, %r407, %r517, %r1337;
	fma.rn.f32 	%r1335, %r407, %r516, %r1335;
	fma.rn.f32 	%r1305, %r407, %r515, %r1305;
	fma.rn.f32 	%r1303, %r407, %r514, %r1303;
	fma.rn.f32 	%r1273, %r407, %r513, %r1273;
	fma.rn.f32 	%r1271, %r407, %r512, %r1271;
	fma.rn.f32 	%r1340, %r408, %r511, %r1340;
	fma.rn.f32 	%r1338, %r408, %r510, %r1338;
	fma.rn.f32 	%r1308, %r408, %r509, %r1308;
	fma.rn.f32 	%r1306, %r408, %r508, %r1306;
	fma.rn.f32 	%r1276, %r408, %r507, %r1276;
	fma.rn.f32 	%r1274, %r408, %r506, %r1274;
	fma.rn.f32 	%r1354, %r400, %r505, %r1354;
	fma.rn.f32 	%r1341, %r409, %r504, %r1341;
	fma.rn.f32 	%r1339, %r409, %r503, %r1339;
	fma.rn.f32 	%r1309, %r409, %r502, %r1309;
	fma.rn.f32 	%r1307, %r409, %r501, %r1307;
	fma.rn.f32 	%r1277, %r409, %r500, %r1277;
	fma.rn.f32 	%r1275, %r409, %r499, %r1275;
	fma.rn.f32 	%r1355, %r401, %r498, %r1355;
	fma.rn.f32 	%r1344, %r410, %r497, %r1344;
	fma.rn.f32 	%r1342, %r410, %r496, %r1342;
	fma.rn.f32 	%r1312, %r410, %r495, %r1312;
	fma.rn.f32 	%r1310, %r410, %r494, %r1310;
	fma.rn.f32 	%r1280, %r410, %r493, %r1280;
	fma.rn.f32 	%r1278, %r410, %r492, %r1278;
	fma.rn.f32 	%r1356, %r400, %r491, %r1356;
	fma.rn.f32 	%r1345, %r411, %r490, %r1345;
	fma.rn.f32 	%r1343, %r411, %r489, %r1343;
	fma.rn.f32 	%r1313, %r411, %r488, %r1313;
	fma.rn.f32 	%r1311, %r411, %r487, %r1311;
	fma.rn.f32 	%r1281, %r411, %r486, %r1281;
	fma.rn.f32 	%r1279, %r411, %r485, %r1279;
	fma.rn.f32 	%r1357, %r401, %r484, %r1357;
	fma.rn.f32 	%r1348, %r412, %r483, %r1348;
	fma.rn.f32 	%r1346, %r412, %r482, %r1346;
	fma.rn.f32 	%r1316, %r412, %r481, %r1316;
	fma.rn.f32 	%r1314, %r412, %r480, %r1314;
	fma.rn.f32 	%r1284, %r412, %r479, %r1284;
	fma.rn.f32 	%r1282, %r412, %r478, %r1282;
	fma.rn.f32 	%r1358, %r402, %r477, %r1358;
	fma.rn.f32 	%r1349, %r413, %r476, %r1349;
	fma.rn.f32 	%r1347, %r413, %r475, %r1347;
	fma.rn.f32 	%r1317, %r413, %r474, %r1317;
	fma.rn.f32 	%r1315, %r413, %r473, %r1315;
	fma.rn.f32 	%r1285, %r413, %r472, %r1285;
	fma.rn.f32 	%r1283, %r413, %r471, %r1283;
	fma.rn.f32 	%r1359, %r403, %r470, %r1359;
	fma.rn.f32 	%r1352, %r414, %r469, %r1352;
	fma.rn.f32 	%r1350, %r414, %r468, %r1350;
	fma.rn.f32 	%r1320, %r414, %r467, %r1320;
	fma.rn.f32 	%r1318, %r414, %r466, %r1318;
	fma.rn.f32 	%r1288, %r414, %r465, %r1288;
	fma.rn.f32 	%r1286, %r414, %r464, %r1286;
	fma.rn.f32 	%r1360, %r402, %r463, %r1360;
	fma.rn.f32 	%r1353, %r415, %r462, %r1353;
	fma.rn.f32 	%r1351, %r415, %r461, %r1351;
	fma.rn.f32 	%r1321, %r415, %r460, %r1321;
	fma.rn.f32 	%r1319, %r415, %r459, %r1319;
	fma.rn.f32 	%r1289, %r415, %r458, %r1289;
	fma.rn.f32 	%r1287, %r415, %r457, %r1287;
	fma.rn.f32 	%r1361, %r403, %r456, %r1361;
	fma.rn.f32 	%r1362, %r404, %r455, %r1362;
	fma.rn.f32 	%r1363, %r405, %r454, %r1363;
	fma.rn.f32 	%r1364, %r404, %r453, %r1364;
	fma.rn.f32 	%r1365, %r405, %r452, %r1365;
	fma.rn.f32 	%r1366, %r406, %r451, %r1366;
	fma.rn.f32 	%r1367, %r407, %r450, %r1367;
	fma.rn.f32 	%r1368, %r406, %r449, %r1368;
	fma.rn.f32 	%r1369, %r407, %r448, %r1369;
	fma.rn.f32 	%r1370, %r408, %r447, %r1370;
	fma.rn.f32 	%r1371, %r409, %r446, %r1371;
	fma.rn.f32 	%r1372, %r408, %r445, %r1372;
	fma.rn.f32 	%r1373, %r409, %r444, %r1373;
	fma.rn.f32 	%r1374, %r410, %r443, %r1374;
	fma.rn.f32 	%r1375, %r411, %r442, %r1375;
	fma.rn.f32 	%r1376, %r410, %r441, %r1376;
	fma.rn.f32 	%r1377, %r411, %r440, %r1377;
	fma.rn.f32 	%r1378, %r412, %r439, %r1378;
	fma.rn.f32 	%r1379, %r413, %r438, %r1379;
	fma.rn.f32 	%r1380, %r412, %r437, %r1380;
	fma.rn.f32 	%r1381, %r413, %r436, %r1381;
	fma.rn.f32 	%r1382, %r414, %r435, %r1382;
	fma.rn.f32 	%r1383, %r415, %r434, %r1383;
	fma.rn.f32 	%r1384, %r414, %r433, %r1384;
	fma.rn.f32 	%r1385, %r415, %r432, %r1385;
	.loc	1 344 18                        // sk06_mlp_down.py:344:18
	add.s64 	%rd75, %rd268, %rd5;
	add.s64 	%rd76, %rd269, %rd5;
	add.s64 	%rd77, %rd270, %rd5;
	.loc	1 345 18                        // sk06_mlp_down.py:345:18
	add.s64 	%rd78, %rd271, %rd5;
	add.s64 	%rd79, %rd272, %rd5;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	add.s64 	%rd80, %rd273, %rd5;
	add.s32 	%r560, %r1254, 1;
	setp.gt.s32 	%p6, %r560, 2;
	selp.b32 	%r1254, 0, %r560, %p6;
	.loc	1 338 27                        // sk06_mlp_down.py:338:27
	shl.b32 	%r561, %r1254, 14;
	bar.sync 	0;
	add.s32 	%r416, %r21, %r561;
	selp.b32 	%r417, 16, 0, %p4;
	// begin inline asm
	cp.async.cg.shared.global [ %r416 + 0 ], [ %rd75 + 0 ], 0x10, %r417;
	// end inline asm
	add.s32 	%r418, %r416, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r418 + 0 ], [ %rd76 + 0 ], 0x10, %r417;
	// end inline asm
	add.s32 	%r419, %r416, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r419 + 0 ], [ %rd77 + 0 ], 0x10, %r417;
	// end inline asm
	add.s32 	%r420, %r416, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r420 + 0 ], [ %rd78 + 0 ], 0x10, %r417;
	// end inline asm
	cp.async.commit_group;
	.loc	1 338 44                        // sk06_mlp_down.py:338:44
	shl.b32 	%r562, %r1254, 13;
	add.s32 	%r563, %r21, %r562;
	add.s32 	%r421, %r563, 49152;
	// begin inline asm
	cp.async.cg.shared.global [ %r421 + 0 ], [ %rd79 + 0 ], 0x10, %r417;
	// end inline asm
	add.s32 	%r422, %r563, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r422 + 0 ], [ %rd80 + 0 ], 0x10, %r417;
	// end inline asm
	cp.async.commit_group;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	add.s32 	%r1255, %r1255, 1;
	add.s64 	%rd273, %rd273, 64;
	add.s64 	%rd272, %rd272, 64;
	add.s64 	%rd271, %rd271, 64;
	add.s64 	%rd270, %rd270, 64;
	add.s64 	%rd269, %rd269, 64;
	add.s64 	%rd268, %rd268, 64;
	setp.ne.b32 	%p7, %r8, %r1255;
	@%p7 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_4;
$L__BB0_1:                              // %.._crit_edge_crit_edge
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	and.b32 	%r1257, %r2, 16;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	shl.b32 	%r1256, %r2, 4;
	mov.b32 	%r1258, 0f00000000;
	mov.b32 	%r1259, %r1258;
	mov.b32 	%r1260, %r1258;
	mov.b32 	%r1261, %r1258;
	mov.b32 	%r1262, %r1258;
	mov.b32 	%r1263, %r1258;
	mov.b32 	%r1264, %r1258;
	mov.b32 	%r1265, %r1258;
	mov.b32 	%r1266, %r1258;
	mov.b32 	%r1267, %r1258;
	mov.b32 	%r1268, %r1258;
	mov.b32 	%r1269, %r1258;
	mov.b32 	%r1270, %r1258;
	mov.b32 	%r1271, %r1258;
	mov.b32 	%r1272, %r1258;
	mov.b32 	%r1273, %r1258;
	mov.b32 	%r1274, %r1258;
	mov.b32 	%r1275, %r1258;
	mov.b32 	%r1276, %r1258;
	mov.b32 	%r1277, %r1258;
	mov.b32 	%r1278, %r1258;
	mov.b32 	%r1279, %r1258;
	mov.b32 	%r1280, %r1258;
	mov.b32 	%r1281, %r1258;
	mov.b32 	%r1282, %r1258;
	mov.b32 	%r1283, %r1258;
	mov.b32 	%r1284, %r1258;
	mov.b32 	%r1285, %r1258;
	mov.b32 	%r1286, %r1258;
	mov.b32 	%r1287, %r1258;
	mov.b32 	%r1288, %r1258;
	mov.b32 	%r1289, %r1258;
	mov.b32 	%r1290, %r1258;
	mov.b32 	%r1291, %r1258;
	mov.b32 	%r1292, %r1258;
	mov.b32 	%r1293, %r1258;
	mov.b32 	%r1294, %r1258;
	mov.b32 	%r1295, %r1258;
	mov.b32 	%r1296, %r1258;
	mov.b32 	%r1297, %r1258;
	mov.b32 	%r1298, %r1258;
	mov.b32 	%r1299, %r1258;
	mov.b32 	%r1300, %r1258;
	mov.b32 	%r1301, %r1258;
	mov.b32 	%r1302, %r1258;
	mov.b32 	%r1303, %r1258;
	mov.b32 	%r1304, %r1258;
	mov.b32 	%r1305, %r1258;
	mov.b32 	%r1306, %r1258;
	mov.b32 	%r1307, %r1258;
	mov.b32 	%r1308, %r1258;
	mov.b32 	%r1309, %r1258;
	mov.b32 	%r1310, %r1258;
	mov.b32 	%r1311, %r1258;
	mov.b32 	%r1312, %r1258;
	mov.b32 	%r1313, %r1258;
	mov.b32 	%r1314, %r1258;
	mov.b32 	%r1315, %r1258;
	mov.b32 	%r1316, %r1258;
	mov.b32 	%r1317, %r1258;
	mov.b32 	%r1318, %r1258;
	mov.b32 	%r1319, %r1258;
	mov.b32 	%r1320, %r1258;
	mov.b32 	%r1321, %r1258;
	mov.b32 	%r1322, %r1258;
	mov.b32 	%r1323, %r1258;
	mov.b32 	%r1324, %r1258;
	mov.b32 	%r1325, %r1258;
	mov.b32 	%r1326, %r1258;
	mov.b32 	%r1327, %r1258;
	mov.b32 	%r1328, %r1258;
	mov.b32 	%r1329, %r1258;
	mov.b32 	%r1330, %r1258;
	mov.b32 	%r1331, %r1258;
	mov.b32 	%r1332, %r1258;
	mov.b32 	%r1333, %r1258;
	mov.b32 	%r1334, %r1258;
	mov.b32 	%r1335, %r1258;
	mov.b32 	%r1336, %r1258;
	mov.b32 	%r1337, %r1258;
	mov.b32 	%r1338, %r1258;
	mov.b32 	%r1339, %r1258;
	mov.b32 	%r1340, %r1258;
	mov.b32 	%r1341, %r1258;
	mov.b32 	%r1342, %r1258;
	mov.b32 	%r1343, %r1258;
	mov.b32 	%r1344, %r1258;
	mov.b32 	%r1345, %r1258;
	mov.b32 	%r1346, %r1258;
	mov.b32 	%r1347, %r1258;
	mov.b32 	%r1348, %r1258;
	mov.b32 	%r1349, %r1258;
	mov.b32 	%r1350, %r1258;
	mov.b32 	%r1351, %r1258;
	mov.b32 	%r1352, %r1258;
	mov.b32 	%r1353, %r1258;
	mov.b32 	%r1354, %r1258;
	mov.b32 	%r1355, %r1258;
	mov.b32 	%r1356, %r1258;
	mov.b32 	%r1357, %r1258;
	mov.b32 	%r1358, %r1258;
	mov.b32 	%r1359, %r1258;
	mov.b32 	%r1360, %r1258;
	mov.b32 	%r1361, %r1258;
	mov.b32 	%r1362, %r1258;
	mov.b32 	%r1363, %r1258;
	mov.b32 	%r1364, %r1258;
	mov.b32 	%r1365, %r1258;
	mov.b32 	%r1366, %r1258;
	mov.b32 	%r1367, %r1258;
	mov.b32 	%r1368, %r1258;
	mov.b32 	%r1369, %r1258;
	mov.b32 	%r1370, %r1258;
	mov.b32 	%r1371, %r1258;
	mov.b32 	%r1372, %r1258;
	mov.b32 	%r1373, %r1258;
	mov.b32 	%r1374, %r1258;
	mov.b32 	%r1375, %r1258;
	mov.b32 	%r1376, %r1258;
	mov.b32 	%r1377, %r1258;
	mov.b32 	%r1378, %r1258;
	mov.b32 	%r1379, %r1258;
	mov.b32 	%r1380, %r1258;
	mov.b32 	%r1381, %r1258;
	mov.b32 	%r1382, %r1258;
	mov.b32 	%r1383, %r1258;
	mov.b32 	%r1384, %r1258;
	mov.b32 	%r1385, %r1258;
$L__BB0_4:                              // %._crit_edge
	.loc	1 323 45                        // sk06_mlp_down.py:323:45
	shl.b32 	%r778, %r7, 3;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r779, %r778, %r4;
	or.b32 	%r780, %r779, 7;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r781, %r780, %r16;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r782, %r779, 6;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r783, %r782, %r16;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r784, %r779, 5;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r785, %r784, %r16;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r786, %r779, 4;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r787, %r786, %r16;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r788, %r779, 3;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r789, %r788, %r16;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r790, %r779, 2;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r791, %r790, %r16;
	.loc	1 323 32                        // sk06_mlp_down.py:323:32
	or.b32 	%r792, %r779, 1;
	.loc	1 323 57                        // sk06_mlp_down.py:323:57
	rem.s32 	%r793, %r792, %r16;
	rem.s32 	%r794, %r779, %r16;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r795, %r1, %r3;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r796, %r795, %r15;
	.loc	1 322 45                        // sk06_mlp_down.py:322:45
	and.b32 	%r797, %r2, 240;
	bfe.u32 	%r798, %r2, 4, 4;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r799, %r798, %r1;
	or.b32 	%r800, %r799, 240;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r801, %r800, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r802, %r799, 224;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r803, %r802, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r804, %r799, 208;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r805, %r804, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r806, %r799, 192;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r807, %r806, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r808, %r799, 176;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r809, %r808, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r810, %r799, 160;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r811, %r810, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r812, %r799, 144;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r813, %r812, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r814, %r799, 128;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r815, %r814, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r816, %r799, 112;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r817, %r816, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r818, %r799, 96;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r819, %r818, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r820, %r799, 80;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r821, %r820, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r822, %r799, 64;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r823, %r822, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r824, %r799, 48;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r825, %r824, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r826, %r799, 32;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r827, %r826, %r15;
	.loc	1 322 32                        // sk06_mlp_down.py:322:32
	or.b32 	%r828, %r799, 16;
	.loc	1 322 57                        // sk06_mlp_down.py:322:57
	rem.s32 	%r829, %r828, %r15;
	rem.s32 	%r830, %r799, %r15;
	.loc	1 337 23                        // sk06_mlp_down.py:337:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 347 38                        // sk06_mlp_down.py:347:38
	mad.wide.s32 	%rd82, %r796, 4, %rd28;
	.loc	1 347 24                        // sk06_mlp_down.py:347:24
	// begin inline asm
	mov.u32 %r565, 0x0;
	ld.global.b32 { %r565 }, [ %rd82 + 0 ];
	// end inline asm
	.loc	1 347 16                        // sk06_mlp_down.py:347:16
	and.b32 	%r831, %r2, 7;
	shl.b32 	%r832, %r831, 3;
	shl.b32 	%r833, %r797, 2;
	and.b32 	%r834, %r2, 8;
	shr.u32 	%r835, %r834, 1;
	add.s32 	%r836, %r194, %r832;
	add.s32 	%r837, %r836, %r833;
	add.s32 	%r564, %r837, %r835;
	// begin inline asm
	st.shared.b32 [ %r564 + 0 ], %r565;
	// end inline asm
	bar.sync 	0;
	and.b32 	%r838, %r9, 56;
	and.b32 	%r839, %r2, 192;
	add.s32 	%r840, %r194, %r838;
	add.s32 	%r841, %r840, %r839;
	ld.shared.v2.b32 	{%r842, %r843}, [%r841];
	ld.shared.v2.b32 	{%r844, %r845}, [%r841+256];
	ld.shared.v2.b32 	{%r846, %r847}, [%r841+512];
	ld.shared.v2.b32 	{%r848, %r849}, [%r841+768];
	.loc	1 350 49                        // sk06_mlp_down.py:350:49
	mul.lo.s32 	%r850, %r830, %r18;
	mul.lo.s32 	%r851, %r829, %r18;
	mul.lo.s32 	%r852, %r827, %r18;
	mul.lo.s32 	%r853, %r825, %r18;
	mul.lo.s32 	%r854, %r823, %r18;
	mul.lo.s32 	%r855, %r821, %r18;
	mul.lo.s32 	%r856, %r819, %r18;
	mul.lo.s32 	%r857, %r817, %r18;
	mul.lo.s32 	%r858, %r815, %r18;
	mul.lo.s32 	%r859, %r813, %r18;
	mul.lo.s32 	%r860, %r811, %r18;
	mul.lo.s32 	%r861, %r809, %r18;
	mul.lo.s32 	%r862, %r807, %r18;
	mul.lo.s32 	%r863, %r805, %r18;
	mul.lo.s32 	%r864, %r803, %r18;
	mul.lo.s32 	%r865, %r801, %r18;
	.loc	1 350 31                        // sk06_mlp_down.py:350:31
	mad.wide.s32 	%rd227, %r850, 2, %rd27;
	mad.wide.s32 	%rd228, %r851, 2, %rd27;
	mad.wide.s32 	%rd229, %r852, 2, %rd27;
	mad.wide.s32 	%rd230, %r853, 2, %rd27;
	mad.wide.s32 	%rd231, %r854, 2, %rd27;
	mad.wide.s32 	%rd232, %r855, 2, %rd27;
	mad.wide.s32 	%rd233, %r856, 2, %rd27;
	mad.wide.s32 	%rd234, %r857, 2, %rd27;
	mad.wide.s32 	%rd235, %r858, 2, %rd27;
	mad.wide.s32 	%rd236, %r859, 2, %rd27;
	mad.wide.s32 	%rd237, %r860, 2, %rd27;
	mad.wide.s32 	%rd238, %r861, 2, %rd27;
	mad.wide.s32 	%rd239, %r862, 2, %rd27;
	mad.wide.s32 	%rd240, %r863, 2, %rd27;
	mad.wide.s32 	%rd241, %r864, 2, %rd27;
	mad.wide.s32 	%rd242, %r865, 2, %rd27;
	.loc	1 350 82                        // sk06_mlp_down.py:350:82
	mul.lo.s32 	%r866, %r794, %r19;
	mul.lo.s32 	%r867, %r793, %r19;
	mul.lo.s32 	%r868, %r791, %r19;
	mul.lo.s32 	%r869, %r789, %r19;
	mul.lo.s32 	%r870, %r787, %r19;
	mul.lo.s32 	%r871, %r785, %r19;
	mul.lo.s32 	%r872, %r783, %r19;
	mul.lo.s32 	%r873, %r781, %r19;
	.loc	1 350 64                        // sk06_mlp_down.py:350:64
	mul.wide.s32 	%rd243, %r866, 2;
	add.s64 	%rd83, %rd227, %rd243;
	mul.wide.s32 	%rd244, %r867, 2;
	add.s64 	%rd84, %rd227, %rd244;
	mul.wide.s32 	%rd245, %r868, 2;
	add.s64 	%rd85, %rd227, %rd245;
	mul.wide.s32 	%rd246, %r869, 2;
	add.s64 	%rd86, %rd227, %rd246;
	mul.wide.s32 	%rd247, %r870, 2;
	add.s64 	%rd87, %rd227, %rd247;
	mul.wide.s32 	%rd248, %r871, 2;
	add.s64 	%rd88, %rd227, %rd248;
	mul.wide.s32 	%rd249, %r872, 2;
	add.s64 	%rd89, %rd227, %rd249;
	mul.wide.s32 	%rd250, %r873, 2;
	add.s64 	%rd90, %rd227, %rd250;
	add.s64 	%rd91, %rd228, %rd243;
	add.s64 	%rd92, %rd228, %rd244;
	add.s64 	%rd93, %rd228, %rd245;
	add.s64 	%rd94, %rd228, %rd246;
	add.s64 	%rd95, %rd228, %rd247;
	add.s64 	%rd96, %rd228, %rd248;
	add.s64 	%rd97, %rd228, %rd249;
	add.s64 	%rd98, %rd228, %rd250;
	add.s64 	%rd99, %rd229, %rd243;
	add.s64 	%rd100, %rd229, %rd244;
	add.s64 	%rd101, %rd229, %rd245;
	add.s64 	%rd102, %rd229, %rd246;
	add.s64 	%rd103, %rd229, %rd247;
	add.s64 	%rd104, %rd229, %rd248;
	add.s64 	%rd105, %rd229, %rd249;
	add.s64 	%rd106, %rd229, %rd250;
	add.s64 	%rd107, %rd230, %rd243;
	add.s64 	%rd108, %rd230, %rd244;
	add.s64 	%rd109, %rd230, %rd245;
	add.s64 	%rd110, %rd230, %rd246;
	add.s64 	%rd111, %rd230, %rd247;
	add.s64 	%rd112, %rd230, %rd248;
	add.s64 	%rd113, %rd230, %rd249;
	add.s64 	%rd114, %rd230, %rd250;
	add.s64 	%rd115, %rd231, %rd243;
	add.s64 	%rd116, %rd231, %rd244;
	add.s64 	%rd117, %rd231, %rd245;
	add.s64 	%rd118, %rd231, %rd246;
	add.s64 	%rd119, %rd231, %rd247;
	add.s64 	%rd120, %rd231, %rd248;
	add.s64 	%rd121, %rd231, %rd249;
	add.s64 	%rd122, %rd231, %rd250;
	add.s64 	%rd123, %rd232, %rd243;
	add.s64 	%rd124, %rd232, %rd244;
	add.s64 	%rd125, %rd232, %rd245;
	add.s64 	%rd126, %rd232, %rd246;
	add.s64 	%rd127, %rd232, %rd247;
	add.s64 	%rd128, %rd232, %rd248;
	add.s64 	%rd129, %rd232, %rd249;
	add.s64 	%rd130, %rd232, %rd250;
	add.s64 	%rd131, %rd233, %rd243;
	add.s64 	%rd132, %rd233, %rd244;
	add.s64 	%rd133, %rd233, %rd245;
	add.s64 	%rd134, %rd233, %rd246;
	add.s64 	%rd135, %rd233, %rd247;
	add.s64 	%rd136, %rd233, %rd248;
	add.s64 	%rd137, %rd233, %rd249;
	add.s64 	%rd138, %rd233, %rd250;
	add.s64 	%rd139, %rd234, %rd243;
	add.s64 	%rd140, %rd234, %rd244;
	add.s64 	%rd141, %rd234, %rd245;
	add.s64 	%rd142, %rd234, %rd246;
	add.s64 	%rd143, %rd234, %rd247;
	add.s64 	%rd144, %rd234, %rd248;
	add.s64 	%rd145, %rd234, %rd249;
	add.s64 	%rd146, %rd234, %rd250;
	add.s64 	%rd147, %rd235, %rd243;
	add.s64 	%rd148, %rd235, %rd244;
	add.s64 	%rd149, %rd235, %rd245;
	add.s64 	%rd150, %rd235, %rd246;
	add.s64 	%rd151, %rd235, %rd247;
	add.s64 	%rd152, %rd235, %rd248;
	add.s64 	%rd153, %rd235, %rd249;
	add.s64 	%rd154, %rd235, %rd250;
	add.s64 	%rd155, %rd236, %rd243;
	add.s64 	%rd156, %rd236, %rd244;
	add.s64 	%rd157, %rd236, %rd245;
	add.s64 	%rd158, %rd236, %rd246;
	add.s64 	%rd159, %rd236, %rd247;
	add.s64 	%rd160, %rd236, %rd248;
	add.s64 	%rd161, %rd236, %rd249;
	add.s64 	%rd162, %rd236, %rd250;
	add.s64 	%rd163, %rd237, %rd243;
	add.s64 	%rd164, %rd237, %rd244;
	add.s64 	%rd165, %rd237, %rd245;
	add.s64 	%rd166, %rd237, %rd246;
	add.s64 	%rd167, %rd237, %rd247;
	add.s64 	%rd168, %rd237, %rd248;
	add.s64 	%rd169, %rd237, %rd249;
	add.s64 	%rd170, %rd237, %rd250;
	add.s64 	%rd171, %rd238, %rd243;
	add.s64 	%rd172, %rd238, %rd244;
	add.s64 	%rd173, %rd238, %rd245;
	add.s64 	%rd174, %rd238, %rd246;
	add.s64 	%rd175, %rd238, %rd247;
	add.s64 	%rd176, %rd238, %rd248;
	add.s64 	%rd177, %rd238, %rd249;
	add.s64 	%rd178, %rd238, %rd250;
	add.s64 	%rd179, %rd239, %rd243;
	add.s64 	%rd180, %rd239, %rd244;
	add.s64 	%rd181, %rd239, %rd245;
	add.s64 	%rd182, %rd239, %rd246;
	add.s64 	%rd183, %rd239, %rd247;
	add.s64 	%rd184, %rd239, %rd248;
	add.s64 	%rd185, %rd239, %rd249;
	add.s64 	%rd186, %rd239, %rd250;
	add.s64 	%rd187, %rd240, %rd243;
	add.s64 	%rd188, %rd240, %rd244;
	add.s64 	%rd189, %rd240, %rd245;
	add.s64 	%rd190, %rd240, %rd246;
	add.s64 	%rd191, %rd240, %rd247;
	add.s64 	%rd192, %rd240, %rd248;
	add.s64 	%rd193, %rd240, %rd249;
	add.s64 	%rd194, %rd240, %rd250;
	add.s64 	%rd195, %rd241, %rd243;
	add.s64 	%rd196, %rd241, %rd244;
	add.s64 	%rd197, %rd241, %rd245;
	add.s64 	%rd198, %rd241, %rd246;
	add.s64 	%rd199, %rd241, %rd247;
	add.s64 	%rd200, %rd241, %rd248;
	add.s64 	%rd201, %rd241, %rd249;
	add.s64 	%rd202, %rd241, %rd250;
	add.s64 	%rd203, %rd242, %rd243;
	add.s64 	%rd204, %rd242, %rd244;
	add.s64 	%rd205, %rd242, %rd245;
	add.s64 	%rd206, %rd242, %rd246;
	add.s64 	%rd207, %rd242, %rd247;
	add.s64 	%rd208, %rd242, %rd248;
	add.s64 	%rd209, %rd242, %rd249;
	add.s64 	%rd210, %rd242, %rd250;
	.loc	1 350 19                        // sk06_mlp_down.py:350:19
	// begin inline asm
	mov.u16 %rs1, 0x0;
	ld.global.b16 { %rs1 }, [ %rd83 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs2, 0x0;
	ld.global.b16 { %rs2 }, [ %rd84 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs3, 0x0;
	ld.global.b16 { %rs3 }, [ %rd85 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs4, 0x0;
	ld.global.b16 { %rs4 }, [ %rd86 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs5, 0x0;
	ld.global.b16 { %rs5 }, [ %rd87 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs6, 0x0;
	ld.global.b16 { %rs6 }, [ %rd88 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs7, 0x0;
	ld.global.b16 { %rs7 }, [ %rd89 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs8, 0x0;
	ld.global.b16 { %rs8 }, [ %rd90 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs9, 0x0;
	ld.global.b16 { %rs9 }, [ %rd91 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs10, 0x0;
	ld.global.b16 { %rs10 }, [ %rd92 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs11, 0x0;
	ld.global.b16 { %rs11 }, [ %rd93 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs12, 0x0;
	ld.global.b16 { %rs12 }, [ %rd94 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs13, 0x0;
	ld.global.b16 { %rs13 }, [ %rd95 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs14, 0x0;
	ld.global.b16 { %rs14 }, [ %rd96 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs15, 0x0;
	ld.global.b16 { %rs15 }, [ %rd97 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs16, 0x0;
	ld.global.b16 { %rs16 }, [ %rd98 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs17, 0x0;
	ld.global.b16 { %rs17 }, [ %rd99 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs18, 0x0;
	ld.global.b16 { %rs18 }, [ %rd100 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs19, 0x0;
	ld.global.b16 { %rs19 }, [ %rd101 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs20, 0x0;
	ld.global.b16 { %rs20 }, [ %rd102 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs21, 0x0;
	ld.global.b16 { %rs21 }, [ %rd103 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs22, 0x0;
	ld.global.b16 { %rs22 }, [ %rd104 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs23, 0x0;
	ld.global.b16 { %rs23 }, [ %rd105 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs24, 0x0;
	ld.global.b16 { %rs24 }, [ %rd106 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs25, 0x0;
	ld.global.b16 { %rs25 }, [ %rd107 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs26, 0x0;
	ld.global.b16 { %rs26 }, [ %rd108 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs27, 0x0;
	ld.global.b16 { %rs27 }, [ %rd109 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs28, 0x0;
	ld.global.b16 { %rs28 }, [ %rd110 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs29, 0x0;
	ld.global.b16 { %rs29 }, [ %rd111 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs30, 0x0;
	ld.global.b16 { %rs30 }, [ %rd112 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs31, 0x0;
	ld.global.b16 { %rs31 }, [ %rd113 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs32, 0x0;
	ld.global.b16 { %rs32 }, [ %rd114 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs33, 0x0;
	ld.global.b16 { %rs33 }, [ %rd115 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs34, 0x0;
	ld.global.b16 { %rs34 }, [ %rd116 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs35, 0x0;
	ld.global.b16 { %rs35 }, [ %rd117 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs36, 0x0;
	ld.global.b16 { %rs36 }, [ %rd118 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs37, 0x0;
	ld.global.b16 { %rs37 }, [ %rd119 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs38, 0x0;
	ld.global.b16 { %rs38 }, [ %rd120 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs39, 0x0;
	ld.global.b16 { %rs39 }, [ %rd121 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs40, 0x0;
	ld.global.b16 { %rs40 }, [ %rd122 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs41, 0x0;
	ld.global.b16 { %rs41 }, [ %rd123 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs42, 0x0;
	ld.global.b16 { %rs42 }, [ %rd124 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs43, 0x0;
	ld.global.b16 { %rs43 }, [ %rd125 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs44, 0x0;
	ld.global.b16 { %rs44 }, [ %rd126 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs45, 0x0;
	ld.global.b16 { %rs45 }, [ %rd127 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs46, 0x0;
	ld.global.b16 { %rs46 }, [ %rd128 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs47, 0x0;
	ld.global.b16 { %rs47 }, [ %rd129 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs48, 0x0;
	ld.global.b16 { %rs48 }, [ %rd130 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs49, 0x0;
	ld.global.b16 { %rs49 }, [ %rd131 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs50, 0x0;
	ld.global.b16 { %rs50 }, [ %rd132 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs51, 0x0;
	ld.global.b16 { %rs51 }, [ %rd133 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs52, 0x0;
	ld.global.b16 { %rs52 }, [ %rd134 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs53, 0x0;
	ld.global.b16 { %rs53 }, [ %rd135 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs54, 0x0;
	ld.global.b16 { %rs54 }, [ %rd136 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs55, 0x0;
	ld.global.b16 { %rs55 }, [ %rd137 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs56, 0x0;
	ld.global.b16 { %rs56 }, [ %rd138 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs57, 0x0;
	ld.global.b16 { %rs57 }, [ %rd139 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs58, 0x0;
	ld.global.b16 { %rs58 }, [ %rd140 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs59, 0x0;
	ld.global.b16 { %rs59 }, [ %rd141 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs60, 0x0;
	ld.global.b16 { %rs60 }, [ %rd142 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs61, 0x0;
	ld.global.b16 { %rs61 }, [ %rd143 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs62, 0x0;
	ld.global.b16 { %rs62 }, [ %rd144 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs63, 0x0;
	ld.global.b16 { %rs63 }, [ %rd145 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs64, 0x0;
	ld.global.b16 { %rs64 }, [ %rd146 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs65, 0x0;
	ld.global.b16 { %rs65 }, [ %rd147 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs66, 0x0;
	ld.global.b16 { %rs66 }, [ %rd148 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs67, 0x0;
	ld.global.b16 { %rs67 }, [ %rd149 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs68, 0x0;
	ld.global.b16 { %rs68 }, [ %rd150 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs69, 0x0;
	ld.global.b16 { %rs69 }, [ %rd151 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs70, 0x0;
	ld.global.b16 { %rs70 }, [ %rd152 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs71, 0x0;
	ld.global.b16 { %rs71 }, [ %rd153 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs72, 0x0;
	ld.global.b16 { %rs72 }, [ %rd154 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs73, 0x0;
	ld.global.b16 { %rs73 }, [ %rd155 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs74, 0x0;
	ld.global.b16 { %rs74 }, [ %rd156 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs75, 0x0;
	ld.global.b16 { %rs75 }, [ %rd157 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs76, 0x0;
	ld.global.b16 { %rs76 }, [ %rd158 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs77, 0x0;
	ld.global.b16 { %rs77 }, [ %rd159 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs78, 0x0;
	ld.global.b16 { %rs78 }, [ %rd160 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs79, 0x0;
	ld.global.b16 { %rs79 }, [ %rd161 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs80, 0x0;
	ld.global.b16 { %rs80 }, [ %rd162 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs81, 0x0;
	ld.global.b16 { %rs81 }, [ %rd163 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs82, 0x0;
	ld.global.b16 { %rs82 }, [ %rd164 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs83, 0x0;
	ld.global.b16 { %rs83 }, [ %rd165 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs84, 0x0;
	ld.global.b16 { %rs84 }, [ %rd166 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs85, 0x0;
	ld.global.b16 { %rs85 }, [ %rd167 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs86, 0x0;
	ld.global.b16 { %rs86 }, [ %rd168 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs87, 0x0;
	ld.global.b16 { %rs87 }, [ %rd169 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs88, 0x0;
	ld.global.b16 { %rs88 }, [ %rd170 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs89, 0x0;
	ld.global.b16 { %rs89 }, [ %rd171 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs90, 0x0;
	ld.global.b16 { %rs90 }, [ %rd172 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs91, 0x0;
	ld.global.b16 { %rs91 }, [ %rd173 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs92, 0x0;
	ld.global.b16 { %rs92 }, [ %rd174 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs93, 0x0;
	ld.global.b16 { %rs93 }, [ %rd175 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs94, 0x0;
	ld.global.b16 { %rs94 }, [ %rd176 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs95, 0x0;
	ld.global.b16 { %rs95 }, [ %rd177 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs96, 0x0;
	ld.global.b16 { %rs96 }, [ %rd178 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs97, 0x0;
	ld.global.b16 { %rs97 }, [ %rd179 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs98, 0x0;
	ld.global.b16 { %rs98 }, [ %rd180 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs99, 0x0;
	ld.global.b16 { %rs99 }, [ %rd181 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs100, 0x0;
	ld.global.b16 { %rs100 }, [ %rd182 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs101, 0x0;
	ld.global.b16 { %rs101 }, [ %rd183 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs102, 0x0;
	ld.global.b16 { %rs102 }, [ %rd184 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs103, 0x0;
	ld.global.b16 { %rs103 }, [ %rd185 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs104, 0x0;
	ld.global.b16 { %rs104 }, [ %rd186 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs105, 0x0;
	ld.global.b16 { %rs105 }, [ %rd187 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs106, 0x0;
	ld.global.b16 { %rs106 }, [ %rd188 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs107, 0x0;
	ld.global.b16 { %rs107 }, [ %rd189 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs108, 0x0;
	ld.global.b16 { %rs108 }, [ %rd190 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs109, 0x0;
	ld.global.b16 { %rs109 }, [ %rd191 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs110, 0x0;
	ld.global.b16 { %rs110 }, [ %rd192 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs111, 0x0;
	ld.global.b16 { %rs111 }, [ %rd193 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs112, 0x0;
	ld.global.b16 { %rs112 }, [ %rd194 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs113, 0x0;
	ld.global.b16 { %rs113 }, [ %rd195 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs114, 0x0;
	ld.global.b16 { %rs114 }, [ %rd196 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs115, 0x0;
	ld.global.b16 { %rs115 }, [ %rd197 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs116, 0x0;
	ld.global.b16 { %rs116 }, [ %rd198 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs117, 0x0;
	ld.global.b16 { %rs117 }, [ %rd199 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs118, 0x0;
	ld.global.b16 { %rs118 }, [ %rd200 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs119, 0x0;
	ld.global.b16 { %rs119 }, [ %rd201 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs120, 0x0;
	ld.global.b16 { %rs120 }, [ %rd202 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs121, 0x0;
	ld.global.b16 { %rs121 }, [ %rd203 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs122, 0x0;
	ld.global.b16 { %rs122 }, [ %rd204 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs123, 0x0;
	ld.global.b16 { %rs123 }, [ %rd205 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs124, 0x0;
	ld.global.b16 { %rs124 }, [ %rd206 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs125, 0x0;
	ld.global.b16 { %rs125 }, [ %rd207 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs126, 0x0;
	ld.global.b16 { %rs126 }, [ %rd208 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs127, 0x0;
	ld.global.b16 { %rs127 }, [ %rd209 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs128, 0x0;
	ld.global.b16 { %rs128 }, [ %rd210 + 0 ];
	// end inline asm
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	bar.sync 	0;
	shl.b32 	%r874, %r2, 7;
	and.b32 	%r875, %r874, 15360;
	shl.b32 	%r876, %r831, 4;
	or.b32 	%r877, %r875, %r876;
	xor.b32 	%r878, %r877, %r797;
	add.s32 	%r566, %r194, %r878;
	mov.b32 	%r567, {%rs1, %rs2};
	mov.b32 	%r568, {%rs3, %rs4};
	mov.b32 	%r569, {%rs5, %rs6};
	mov.b32 	%r570, {%rs7, %rs8};
	// begin inline asm
	st.shared.v4.b32 [ %r566 + 0 ], { %r567, %r568, %r569, %r570 };
	// end inline asm
	add.s32 	%r571, %r566, 256;
	mov.b32 	%r572, {%rs9, %rs10};
	mov.b32 	%r573, {%rs11, %rs12};
	mov.b32 	%r574, {%rs13, %rs14};
	mov.b32 	%r575, {%rs15, %rs16};
	// begin inline asm
	st.shared.v4.b32 [ %r571 + 0 ], { %r572, %r573, %r574, %r575 };
	// end inline asm
	add.s32 	%r576, %r566, 512;
	mov.b32 	%r577, {%rs17, %rs18};
	mov.b32 	%r578, {%rs19, %rs20};
	mov.b32 	%r579, {%rs21, %rs22};
	mov.b32 	%r580, {%rs23, %rs24};
	// begin inline asm
	st.shared.v4.b32 [ %r576 + 0 ], { %r577, %r578, %r579, %r580 };
	// end inline asm
	add.s32 	%r581, %r566, 768;
	mov.b32 	%r582, {%rs25, %rs26};
	mov.b32 	%r583, {%rs27, %rs28};
	mov.b32 	%r584, {%rs29, %rs30};
	mov.b32 	%r585, {%rs31, %rs32};
	// begin inline asm
	st.shared.v4.b32 [ %r581 + 0 ], { %r582, %r583, %r584, %r585 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r879, %r831, 11;
	shl.b32 	%r880, %r7, 4;
	shl.b32 	%r881, %r839, 2;
	setp.eq.b32 	%p24, %r1257, 0;
	shl.b32 	%r882, %r1257, 1;
	shr.u32 	%r883, %r6, 1;
	or.b32 	%r884, %r880, %r881;
	or.b32 	%r885, %r882, %r883;
	xor.b32 	%r886, %r884, %r885;
	or.b32 	%r887, %r886, %r879;
	add.s32 	%r888, %r194, %r887;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r889, %r890, %r891, %r892}, [%r888];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r893, %r894, %r895, %r896}, [%r888+1024];
	xor.b32 	%r897, %r887, 64;
	add.s32 	%r898, %r194, %r897;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r899, %r900, %r901, %r902}, [%r898];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r903, %r904, %r905, %r906}, [%r898+1024];
	bar.sync 	0;
	mov.b32 	%r586, {%rs33, %rs34};
	mov.b32 	%r587, {%rs35, %rs36};
	mov.b32 	%r588, {%rs37, %rs38};
	mov.b32 	%r589, {%rs39, %rs40};
	// begin inline asm
	st.shared.v4.b32 [ %r566 + 0 ], { %r586, %r587, %r588, %r589 };
	// end inline asm
	mov.b32 	%r590, {%rs41, %rs42};
	mov.b32 	%r591, {%rs43, %rs44};
	mov.b32 	%r592, {%rs45, %rs46};
	mov.b32 	%r593, {%rs47, %rs48};
	// begin inline asm
	st.shared.v4.b32 [ %r571 + 0 ], { %r590, %r591, %r592, %r593 };
	// end inline asm
	mov.b32 	%r594, {%rs49, %rs50};
	mov.b32 	%r595, {%rs51, %rs52};
	mov.b32 	%r596, {%rs53, %rs54};
	mov.b32 	%r597, {%rs55, %rs56};
	// begin inline asm
	st.shared.v4.b32 [ %r576 + 0 ], { %r594, %r595, %r596, %r597 };
	// end inline asm
	mov.b32 	%r598, {%rs57, %rs58};
	mov.b32 	%r599, {%rs59, %rs60};
	mov.b32 	%r600, {%rs61, %rs62};
	mov.b32 	%r601, {%rs63, %rs64};
	// begin inline asm
	st.shared.v4.b32 [ %r581 + 0 ], { %r598, %r599, %r600, %r601 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r907, %r908, %r909, %r910}, [%r888];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r911, %r912, %r913, %r914}, [%r888+1024];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r915, %r916, %r917, %r918}, [%r898];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r919, %r920, %r921, %r922}, [%r898+1024];
	bar.sync 	0;
	mov.b32 	%r602, {%rs65, %rs66};
	mov.b32 	%r603, {%rs67, %rs68};
	mov.b32 	%r604, {%rs69, %rs70};
	mov.b32 	%r605, {%rs71, %rs72};
	// begin inline asm
	st.shared.v4.b32 [ %r566 + 0 ], { %r602, %r603, %r604, %r605 };
	// end inline asm
	mov.b32 	%r606, {%rs73, %rs74};
	mov.b32 	%r607, {%rs75, %rs76};
	mov.b32 	%r608, {%rs77, %rs78};
	mov.b32 	%r609, {%rs79, %rs80};
	// begin inline asm
	st.shared.v4.b32 [ %r571 + 0 ], { %r606, %r607, %r608, %r609 };
	// end inline asm
	mov.b32 	%r610, {%rs81, %rs82};
	mov.b32 	%r611, {%rs83, %rs84};
	mov.b32 	%r612, {%rs85, %rs86};
	mov.b32 	%r613, {%rs87, %rs88};
	// begin inline asm
	st.shared.v4.b32 [ %r576 + 0 ], { %r610, %r611, %r612, %r613 };
	// end inline asm
	mov.b32 	%r614, {%rs89, %rs90};
	mov.b32 	%r615, {%rs91, %rs92};
	mov.b32 	%r616, {%rs93, %rs94};
	mov.b32 	%r617, {%rs95, %rs96};
	// begin inline asm
	st.shared.v4.b32 [ %r581 + 0 ], { %r614, %r615, %r616, %r617 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r923, %r924, %r925, %r926}, [%r888];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r927, %r928, %r929, %r930}, [%r888+1024];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r931, %r932, %r933, %r934}, [%r898];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r935, %r936, %r937, %r938}, [%r898+1024];
	bar.sync 	0;
	mov.b32 	%r618, {%rs97, %rs98};
	mov.b32 	%r619, {%rs99, %rs100};
	mov.b32 	%r620, {%rs101, %rs102};
	mov.b32 	%r621, {%rs103, %rs104};
	// begin inline asm
	st.shared.v4.b32 [ %r566 + 0 ], { %r618, %r619, %r620, %r621 };
	// end inline asm
	mov.b32 	%r622, {%rs105, %rs106};
	mov.b32 	%r623, {%rs107, %rs108};
	mov.b32 	%r624, {%rs109, %rs110};
	mov.b32 	%r625, {%rs111, %rs112};
	// begin inline asm
	st.shared.v4.b32 [ %r571 + 0 ], { %r622, %r623, %r624, %r625 };
	// end inline asm
	mov.b32 	%r626, {%rs113, %rs114};
	mov.b32 	%r627, {%rs115, %rs116};
	mov.b32 	%r628, {%rs117, %rs118};
	mov.b32 	%r629, {%rs119, %rs120};
	// begin inline asm
	st.shared.v4.b32 [ %r576 + 0 ], { %r626, %r627, %r628, %r629 };
	// end inline asm
	mov.b32 	%r630, {%rs121, %rs122};
	mov.b32 	%r631, {%rs123, %rs124};
	mov.b32 	%r632, {%rs125, %rs126};
	mov.b32 	%r633, {%rs127, %rs128};
	// begin inline asm
	st.shared.v4.b32 [ %r581 + 0 ], { %r630, %r631, %r632, %r633 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r939, %r940, %r941, %r942}, [%r888];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r943, %r944, %r945, %r946}, [%r888+1024];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r947, %r948, %r949, %r950}, [%r898];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r951, %r952, %r953, %r954}, [%r898+1024];
	.loc	1 357 31                        // sk06_mlp_down.py:357:31
	setp.lt.s32 	%p25, %r799, %r15;
	setp.lt.s32 	%p26, %r828, %r15;
	setp.lt.s32 	%p27, %r826, %r15;
	setp.lt.s32 	%p28, %r824, %r15;
	setp.lt.s32 	%p29, %r822, %r15;
	setp.lt.s32 	%p30, %r820, %r15;
	setp.lt.s32 	%p31, %r818, %r15;
	setp.lt.s32 	%p32, %r816, %r15;
	setp.lt.s32 	%p33, %r814, %r15;
	setp.lt.s32 	%p34, %r812, %r15;
	setp.lt.s32 	%p35, %r810, %r15;
	setp.lt.s32 	%p36, %r808, %r15;
	setp.lt.s32 	%p37, %r806, %r15;
	setp.lt.s32 	%p38, %r804, %r15;
	setp.lt.s32 	%p39, %r802, %r15;
	setp.lt.s32 	%p40, %r800, %r15;
	.loc	1 357 54                        // sk06_mlp_down.py:357:54
	setp.lt.s32 	%p41, %r779, %r16;
	.loc	1 357 37                        // sk06_mlp_down.py:357:37
	and.pred 	%p8, %p25, %p41;
	and.pred 	%p9, %p26, %p41;
	and.pred 	%p10, %p27, %p41;
	and.pred 	%p11, %p28, %p41;
	and.pred 	%p12, %p29, %p41;
	and.pred 	%p13, %p30, %p41;
	and.pred 	%p14, %p31, %p41;
	and.pred 	%p15, %p32, %p41;
	and.pred 	%p16, %p33, %p41;
	and.pred 	%p17, %p34, %p41;
	and.pred 	%p18, %p35, %p41;
	and.pred 	%p19, %p36, %p41;
	and.pred 	%p20, %p37, %p41;
	and.pred 	%p21, %p38, %p41;
	and.pred 	%p22, %p39, %p41;
	and.pred 	%p23, %p40, %p41;
	.loc	1 355 35                        // sk06_mlp_down.py:355:35
	mul.lo.s32 	%r955, %r799, %r17;
	mul.lo.s32 	%r956, %r828, %r17;
	mul.lo.s32 	%r957, %r826, %r17;
	mul.lo.s32 	%r958, %r824, %r17;
	mul.lo.s32 	%r959, %r822, %r17;
	mul.lo.s32 	%r960, %r820, %r17;
	mul.lo.s32 	%r961, %r818, %r17;
	mul.lo.s32 	%r962, %r816, %r17;
	mul.lo.s32 	%r963, %r814, %r17;
	mul.lo.s32 	%r964, %r812, %r17;
	mul.lo.s32 	%r965, %r810, %r17;
	mul.lo.s32 	%r966, %r808, %r17;
	mul.lo.s32 	%r967, %r806, %r17;
	mul.lo.s32 	%r968, %r804, %r17;
	mul.lo.s32 	%r969, %r802, %r17;
	mul.lo.s32 	%r970, %r800, %r17;
	.loc	1 355 18                        // sk06_mlp_down.py:355:18
	mad.wide.s32 	%rd251, %r955, 2, %rd26;
	mad.wide.s32 	%rd252, %r956, 2, %rd26;
	mad.wide.s32 	%rd253, %r957, 2, %rd26;
	mad.wide.s32 	%rd254, %r958, 2, %rd26;
	mad.wide.s32 	%rd255, %r959, 2, %rd26;
	mad.wide.s32 	%rd256, %r960, 2, %rd26;
	mad.wide.s32 	%rd257, %r961, 2, %rd26;
	mad.wide.s32 	%rd258, %r962, 2, %rd26;
	mad.wide.s32 	%rd259, %r963, 2, %rd26;
	mad.wide.s32 	%rd260, %r964, 2, %rd26;
	mad.wide.s32 	%rd261, %r965, 2, %rd26;
	mad.wide.s32 	%rd262, %r966, 2, %rd26;
	mad.wide.s32 	%rd263, %r967, 2, %rd26;
	mad.wide.s32 	%rd264, %r968, 2, %rd26;
	mad.wide.s32 	%rd265, %r969, 2, %rd26;
	mad.wide.s32 	%rd266, %r970, 2, %rd26;
	.loc	1 355 50                        // sk06_mlp_down.py:355:50
	mul.wide.s32 	%rd267, %r779, 2;
	add.s64 	%rd211, %rd251, %rd267;
	add.s64 	%rd212, %rd252, %rd267;
	add.s64 	%rd213, %rd253, %rd267;
	add.s64 	%rd214, %rd254, %rd267;
	add.s64 	%rd215, %rd255, %rd267;
	add.s64 	%rd216, %rd256, %rd267;
	add.s64 	%rd217, %rd257, %rd267;
	add.s64 	%rd218, %rd258, %rd267;
	add.s64 	%rd219, %rd259, %rd267;
	add.s64 	%rd220, %rd260, %rd267;
	add.s64 	%rd221, %rd261, %rd267;
	add.s64 	%rd222, %rd262, %rd267;
	add.s64 	%rd223, %rd263, %rd267;
	add.s64 	%rd224, %rd264, %rd267;
	add.s64 	%rd225, %rd265, %rd267;
	add.s64 	%rd226, %rd266, %rd267;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs129, %rs130}, %r889;
	cvt.f32.bf16 	%r971, %rs130;
	cvt.f32.bf16 	%r972, %rs129;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r973, %r1258, %r842, %r972;
	fma.rn.f32 	%r974, %r1259, %r842, %r971;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r635, %r974, %r973;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs131, %rs132}, %r890;
	cvt.f32.bf16 	%r975, %rs132;
	cvt.f32.bf16 	%r976, %rs131;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r977, %r1260, %r843, %r976;
	fma.rn.f32 	%r978, %r1261, %r843, %r975;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r640, %r978, %r977;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs133, %rs134}, %r891;
	cvt.f32.bf16 	%r979, %rs134;
	cvt.f32.bf16 	%r980, %rs133;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r981, %r1262, %r842, %r980;
	fma.rn.f32 	%r982, %r1263, %r842, %r979;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r655, %r982, %r981;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs135, %rs136}, %r892;
	cvt.f32.bf16 	%r983, %rs136;
	cvt.f32.bf16 	%r984, %rs135;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r985, %r1264, %r843, %r984;
	fma.rn.f32 	%r986, %r1265, %r843, %r983;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r660, %r986, %r985;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs137, %rs138}, %r899;
	cvt.f32.bf16 	%r987, %rs138;
	cvt.f32.bf16 	%r988, %rs137;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r989, %r1266, %r842, %r988;
	fma.rn.f32 	%r990, %r1267, %r842, %r987;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r675, %r990, %r989;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs139, %rs140}, %r900;
	cvt.f32.bf16 	%r991, %rs140;
	cvt.f32.bf16 	%r992, %rs139;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r993, %r1268, %r843, %r992;
	fma.rn.f32 	%r994, %r1269, %r843, %r991;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r680, %r994, %r993;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs141, %rs142}, %r901;
	cvt.f32.bf16 	%r995, %rs142;
	cvt.f32.bf16 	%r996, %rs141;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r997, %r1270, %r842, %r996;
	fma.rn.f32 	%r998, %r1271, %r842, %r995;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r695, %r998, %r997;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs143, %rs144}, %r902;
	cvt.f32.bf16 	%r999, %rs144;
	cvt.f32.bf16 	%r1000, %rs143;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1001, %r1272, %r843, %r1000;
	fma.rn.f32 	%r1002, %r1273, %r843, %r999;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r700, %r1002, %r1001;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs145, %rs146}, %r893;
	cvt.f32.bf16 	%r1003, %rs146;
	cvt.f32.bf16 	%r1004, %rs145;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1005, %r1274, %r842, %r1004;
	fma.rn.f32 	%r1006, %r1275, %r842, %r1003;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r645, %r1006, %r1005;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs147, %rs148}, %r894;
	cvt.f32.bf16 	%r1007, %rs148;
	cvt.f32.bf16 	%r1008, %rs147;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1009, %r1276, %r843, %r1008;
	fma.rn.f32 	%r1010, %r1277, %r843, %r1007;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r650, %r1010, %r1009;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs149, %rs150}, %r895;
	cvt.f32.bf16 	%r1011, %rs150;
	cvt.f32.bf16 	%r1012, %rs149;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1013, %r1278, %r842, %r1012;
	fma.rn.f32 	%r1014, %r1279, %r842, %r1011;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r665, %r1014, %r1013;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs151, %rs152}, %r896;
	cvt.f32.bf16 	%r1015, %rs152;
	cvt.f32.bf16 	%r1016, %rs151;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1017, %r1280, %r843, %r1016;
	fma.rn.f32 	%r1018, %r1281, %r843, %r1015;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r670, %r1018, %r1017;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs153, %rs154}, %r903;
	cvt.f32.bf16 	%r1019, %rs154;
	cvt.f32.bf16 	%r1020, %rs153;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1021, %r1282, %r842, %r1020;
	fma.rn.f32 	%r1022, %r1283, %r842, %r1019;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r685, %r1022, %r1021;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs155, %rs156}, %r904;
	cvt.f32.bf16 	%r1023, %rs156;
	cvt.f32.bf16 	%r1024, %rs155;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1025, %r1284, %r843, %r1024;
	fma.rn.f32 	%r1026, %r1285, %r843, %r1023;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r690, %r1026, %r1025;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs157, %rs158}, %r905;
	cvt.f32.bf16 	%r1027, %rs158;
	cvt.f32.bf16 	%r1028, %rs157;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1029, %r1286, %r842, %r1028;
	fma.rn.f32 	%r1030, %r1287, %r842, %r1027;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r705, %r1030, %r1029;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs159, %rs160}, %r906;
	cvt.f32.bf16 	%r1031, %rs160;
	cvt.f32.bf16 	%r1032, %rs159;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1033, %r1288, %r843, %r1032;
	fma.rn.f32 	%r1034, %r1289, %r843, %r1031;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r710, %r1034, %r1033;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs161, %rs162}, %r907;
	cvt.f32.bf16 	%r1035, %rs162;
	cvt.f32.bf16 	%r1036, %rs161;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1037, %r1290, %r844, %r1036;
	fma.rn.f32 	%r1038, %r1291, %r844, %r1035;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r636, %r1038, %r1037;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs163, %rs164}, %r908;
	cvt.f32.bf16 	%r1039, %rs164;
	cvt.f32.bf16 	%r1040, %rs163;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1041, %r1292, %r845, %r1040;
	fma.rn.f32 	%r1042, %r1293, %r845, %r1039;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r641, %r1042, %r1041;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs165, %rs166}, %r909;
	cvt.f32.bf16 	%r1043, %rs166;
	cvt.f32.bf16 	%r1044, %rs165;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1045, %r1294, %r844, %r1044;
	fma.rn.f32 	%r1046, %r1295, %r844, %r1043;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r656, %r1046, %r1045;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs167, %rs168}, %r910;
	cvt.f32.bf16 	%r1047, %rs168;
	cvt.f32.bf16 	%r1048, %rs167;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1049, %r1296, %r845, %r1048;
	fma.rn.f32 	%r1050, %r1297, %r845, %r1047;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r661, %r1050, %r1049;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs169, %rs170}, %r915;
	cvt.f32.bf16 	%r1051, %rs170;
	cvt.f32.bf16 	%r1052, %rs169;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1053, %r1298, %r844, %r1052;
	fma.rn.f32 	%r1054, %r1299, %r844, %r1051;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r676, %r1054, %r1053;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs171, %rs172}, %r916;
	cvt.f32.bf16 	%r1055, %rs172;
	cvt.f32.bf16 	%r1056, %rs171;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1057, %r1300, %r845, %r1056;
	fma.rn.f32 	%r1058, %r1301, %r845, %r1055;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r681, %r1058, %r1057;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs173, %rs174}, %r917;
	cvt.f32.bf16 	%r1059, %rs174;
	cvt.f32.bf16 	%r1060, %rs173;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1061, %r1302, %r844, %r1060;
	fma.rn.f32 	%r1062, %r1303, %r844, %r1059;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r696, %r1062, %r1061;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs175, %rs176}, %r918;
	cvt.f32.bf16 	%r1063, %rs176;
	cvt.f32.bf16 	%r1064, %rs175;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1065, %r1304, %r845, %r1064;
	fma.rn.f32 	%r1066, %r1305, %r845, %r1063;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r701, %r1066, %r1065;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs177, %rs178}, %r911;
	cvt.f32.bf16 	%r1067, %rs178;
	cvt.f32.bf16 	%r1068, %rs177;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1069, %r1306, %r844, %r1068;
	fma.rn.f32 	%r1070, %r1307, %r844, %r1067;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r646, %r1070, %r1069;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs179, %rs180}, %r912;
	cvt.f32.bf16 	%r1071, %rs180;
	cvt.f32.bf16 	%r1072, %rs179;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1073, %r1308, %r845, %r1072;
	fma.rn.f32 	%r1074, %r1309, %r845, %r1071;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r651, %r1074, %r1073;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs181, %rs182}, %r913;
	cvt.f32.bf16 	%r1075, %rs182;
	cvt.f32.bf16 	%r1076, %rs181;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1077, %r1310, %r844, %r1076;
	fma.rn.f32 	%r1078, %r1311, %r844, %r1075;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r666, %r1078, %r1077;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs183, %rs184}, %r914;
	cvt.f32.bf16 	%r1079, %rs184;
	cvt.f32.bf16 	%r1080, %rs183;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1081, %r1312, %r845, %r1080;
	fma.rn.f32 	%r1082, %r1313, %r845, %r1079;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r671, %r1082, %r1081;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs185, %rs186}, %r919;
	cvt.f32.bf16 	%r1083, %rs186;
	cvt.f32.bf16 	%r1084, %rs185;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1085, %r1314, %r844, %r1084;
	fma.rn.f32 	%r1086, %r1315, %r844, %r1083;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r686, %r1086, %r1085;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs187, %rs188}, %r920;
	cvt.f32.bf16 	%r1087, %rs188;
	cvt.f32.bf16 	%r1088, %rs187;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1089, %r1316, %r845, %r1088;
	fma.rn.f32 	%r1090, %r1317, %r845, %r1087;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r691, %r1090, %r1089;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs189, %rs190}, %r921;
	cvt.f32.bf16 	%r1091, %rs190;
	cvt.f32.bf16 	%r1092, %rs189;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1093, %r1318, %r844, %r1092;
	fma.rn.f32 	%r1094, %r1319, %r844, %r1091;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r706, %r1094, %r1093;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs191, %rs192}, %r922;
	cvt.f32.bf16 	%r1095, %rs192;
	cvt.f32.bf16 	%r1096, %rs191;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1097, %r1320, %r845, %r1096;
	fma.rn.f32 	%r1098, %r1321, %r845, %r1095;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r711, %r1098, %r1097;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs193, %rs194}, %r923;
	cvt.f32.bf16 	%r1099, %rs194;
	cvt.f32.bf16 	%r1100, %rs193;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1101, %r1322, %r846, %r1100;
	fma.rn.f32 	%r1102, %r1323, %r846, %r1099;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r637, %r1102, %r1101;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs195, %rs196}, %r924;
	cvt.f32.bf16 	%r1103, %rs196;
	cvt.f32.bf16 	%r1104, %rs195;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1105, %r1324, %r847, %r1104;
	fma.rn.f32 	%r1106, %r1325, %r847, %r1103;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r642, %r1106, %r1105;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs197, %rs198}, %r925;
	cvt.f32.bf16 	%r1107, %rs198;
	cvt.f32.bf16 	%r1108, %rs197;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1109, %r1326, %r846, %r1108;
	fma.rn.f32 	%r1110, %r1327, %r846, %r1107;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r657, %r1110, %r1109;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs199, %rs200}, %r926;
	cvt.f32.bf16 	%r1111, %rs200;
	cvt.f32.bf16 	%r1112, %rs199;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1113, %r1328, %r847, %r1112;
	fma.rn.f32 	%r1114, %r1329, %r847, %r1111;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r662, %r1114, %r1113;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs201, %rs202}, %r931;
	cvt.f32.bf16 	%r1115, %rs202;
	cvt.f32.bf16 	%r1116, %rs201;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1117, %r1330, %r846, %r1116;
	fma.rn.f32 	%r1118, %r1331, %r846, %r1115;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r677, %r1118, %r1117;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs203, %rs204}, %r932;
	cvt.f32.bf16 	%r1119, %rs204;
	cvt.f32.bf16 	%r1120, %rs203;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1121, %r1332, %r847, %r1120;
	fma.rn.f32 	%r1122, %r1333, %r847, %r1119;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r682, %r1122, %r1121;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs205, %rs206}, %r933;
	cvt.f32.bf16 	%r1123, %rs206;
	cvt.f32.bf16 	%r1124, %rs205;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1125, %r1334, %r846, %r1124;
	fma.rn.f32 	%r1126, %r1335, %r846, %r1123;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r697, %r1126, %r1125;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs207, %rs208}, %r934;
	cvt.f32.bf16 	%r1127, %rs208;
	cvt.f32.bf16 	%r1128, %rs207;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1129, %r1336, %r847, %r1128;
	fma.rn.f32 	%r1130, %r1337, %r847, %r1127;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r702, %r1130, %r1129;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs209, %rs210}, %r927;
	cvt.f32.bf16 	%r1131, %rs210;
	cvt.f32.bf16 	%r1132, %rs209;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1133, %r1338, %r846, %r1132;
	fma.rn.f32 	%r1134, %r1339, %r846, %r1131;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r647, %r1134, %r1133;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs211, %rs212}, %r928;
	cvt.f32.bf16 	%r1135, %rs212;
	cvt.f32.bf16 	%r1136, %rs211;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1137, %r1340, %r847, %r1136;
	fma.rn.f32 	%r1138, %r1341, %r847, %r1135;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r652, %r1138, %r1137;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs213, %rs214}, %r929;
	cvt.f32.bf16 	%r1139, %rs214;
	cvt.f32.bf16 	%r1140, %rs213;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1141, %r1342, %r846, %r1140;
	fma.rn.f32 	%r1142, %r1343, %r846, %r1139;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r667, %r1142, %r1141;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs215, %rs216}, %r930;
	cvt.f32.bf16 	%r1143, %rs216;
	cvt.f32.bf16 	%r1144, %rs215;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1145, %r1344, %r847, %r1144;
	fma.rn.f32 	%r1146, %r1345, %r847, %r1143;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r672, %r1146, %r1145;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs217, %rs218}, %r935;
	cvt.f32.bf16 	%r1147, %rs218;
	cvt.f32.bf16 	%r1148, %rs217;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1149, %r1346, %r846, %r1148;
	fma.rn.f32 	%r1150, %r1347, %r846, %r1147;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r687, %r1150, %r1149;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs219, %rs220}, %r936;
	cvt.f32.bf16 	%r1151, %rs220;
	cvt.f32.bf16 	%r1152, %rs219;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1153, %r1348, %r847, %r1152;
	fma.rn.f32 	%r1154, %r1349, %r847, %r1151;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r692, %r1154, %r1153;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs221, %rs222}, %r937;
	cvt.f32.bf16 	%r1155, %rs222;
	cvt.f32.bf16 	%r1156, %rs221;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1157, %r1350, %r846, %r1156;
	fma.rn.f32 	%r1158, %r1351, %r846, %r1155;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r707, %r1158, %r1157;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs223, %rs224}, %r938;
	cvt.f32.bf16 	%r1159, %rs224;
	cvt.f32.bf16 	%r1160, %rs223;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1161, %r1352, %r847, %r1160;
	fma.rn.f32 	%r1162, %r1353, %r847, %r1159;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r712, %r1162, %r1161;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs225, %rs226}, %r939;
	cvt.f32.bf16 	%r1163, %rs226;
	cvt.f32.bf16 	%r1164, %rs225;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1165, %r1354, %r848, %r1164;
	fma.rn.f32 	%r1166, %r1355, %r848, %r1163;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r638, %r1166, %r1165;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs227, %rs228}, %r940;
	cvt.f32.bf16 	%r1167, %rs228;
	cvt.f32.bf16 	%r1168, %rs227;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1169, %r1356, %r849, %r1168;
	fma.rn.f32 	%r1170, %r1357, %r849, %r1167;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r643, %r1170, %r1169;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs229, %rs230}, %r941;
	cvt.f32.bf16 	%r1171, %rs230;
	cvt.f32.bf16 	%r1172, %rs229;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1173, %r1358, %r848, %r1172;
	fma.rn.f32 	%r1174, %r1359, %r848, %r1171;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r658, %r1174, %r1173;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs231, %rs232}, %r942;
	cvt.f32.bf16 	%r1175, %rs232;
	cvt.f32.bf16 	%r1176, %rs231;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1177, %r1360, %r849, %r1176;
	fma.rn.f32 	%r1178, %r1361, %r849, %r1175;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r663, %r1178, %r1177;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs233, %rs234}, %r947;
	cvt.f32.bf16 	%r1179, %rs234;
	cvt.f32.bf16 	%r1180, %rs233;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1181, %r1362, %r848, %r1180;
	fma.rn.f32 	%r1182, %r1363, %r848, %r1179;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r678, %r1182, %r1181;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs235, %rs236}, %r948;
	cvt.f32.bf16 	%r1183, %rs236;
	cvt.f32.bf16 	%r1184, %rs235;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1185, %r1364, %r849, %r1184;
	fma.rn.f32 	%r1186, %r1365, %r849, %r1183;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r683, %r1186, %r1185;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs237, %rs238}, %r949;
	cvt.f32.bf16 	%r1187, %rs238;
	cvt.f32.bf16 	%r1188, %rs237;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1189, %r1366, %r848, %r1188;
	fma.rn.f32 	%r1190, %r1367, %r848, %r1187;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r698, %r1190, %r1189;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs239, %rs240}, %r950;
	cvt.f32.bf16 	%r1191, %rs240;
	cvt.f32.bf16 	%r1192, %rs239;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1193, %r1368, %r849, %r1192;
	fma.rn.f32 	%r1194, %r1369, %r849, %r1191;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r703, %r1194, %r1193;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs241, %rs242}, %r943;
	cvt.f32.bf16 	%r1195, %rs242;
	cvt.f32.bf16 	%r1196, %rs241;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1197, %r1370, %r848, %r1196;
	fma.rn.f32 	%r1198, %r1371, %r848, %r1195;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r648, %r1198, %r1197;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs243, %rs244}, %r944;
	cvt.f32.bf16 	%r1199, %rs244;
	cvt.f32.bf16 	%r1200, %rs243;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1201, %r1372, %r849, %r1200;
	fma.rn.f32 	%r1202, %r1373, %r849, %r1199;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r653, %r1202, %r1201;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs245, %rs246}, %r945;
	cvt.f32.bf16 	%r1203, %rs246;
	cvt.f32.bf16 	%r1204, %rs245;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1205, %r1374, %r848, %r1204;
	fma.rn.f32 	%r1206, %r1375, %r848, %r1203;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r668, %r1206, %r1205;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs247, %rs248}, %r946;
	cvt.f32.bf16 	%r1207, %rs248;
	cvt.f32.bf16 	%r1208, %rs247;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1209, %r1376, %r849, %r1208;
	fma.rn.f32 	%r1210, %r1377, %r849, %r1207;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r673, %r1210, %r1209;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs249, %rs250}, %r951;
	cvt.f32.bf16 	%r1211, %rs250;
	cvt.f32.bf16 	%r1212, %rs249;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1213, %r1378, %r848, %r1212;
	fma.rn.f32 	%r1214, %r1379, %r848, %r1211;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r688, %r1214, %r1213;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs251, %rs252}, %r952;
	cvt.f32.bf16 	%r1215, %rs252;
	cvt.f32.bf16 	%r1216, %rs251;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1217, %r1380, %r849, %r1216;
	fma.rn.f32 	%r1218, %r1381, %r849, %r1215;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r693, %r1218, %r1217;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs253, %rs254}, %r953;
	cvt.f32.bf16 	%r1219, %rs254;
	cvt.f32.bf16 	%r1220, %rs253;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1221, %r1382, %r848, %r1220;
	fma.rn.f32 	%r1222, %r1383, %r848, %r1219;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r708, %r1222, %r1221;
	.loc	1 350 99                        // sk06_mlp_down.py:350:99
	mov.b32 	{%rs255, %rs256}, %r954;
	cvt.f32.bf16 	%r1223, %rs256;
	cvt.f32.bf16 	%r1224, %rs255;
	.loc	1 350 11                        // sk06_mlp_down.py:350:11
	fma.rn.f32 	%r1225, %r1384, %r849, %r1224;
	fma.rn.f32 	%r1226, %r1385, %r849, %r1223;
	.loc	1 356 15                        // sk06_mlp_down.py:356:15
	cvt.rn.bf16x2.f32 	%r713, %r1226, %r1225;
	bar.sync 	0;
	shl.b32 	%r1227, %r5, 14;
	shl.b32 	%r1228, %r5, 5;
	and.b32 	%r1229, %r1256, 3456;
	bfe.s32 	%r1230, %r2, 2, 1;
	and.b32 	%r1231, %r1230, 8208;
	or.b32 	%r1232, %r1228, %r1229;
	xor.b32 	%r1233, %r1231, %r883;
	or.b32 	%r1234, %r1233, %r1232;
	or.b32 	%r1235, %r1234, %r1227;
	add.s32 	%r634, %r194, %r1235;
	// begin inline asm
	st.shared.v4.b32 [ %r634 + 0 ], { %r635, %r636, %r637, %r638 };
	// end inline asm
	add.s32 	%r639, %r634, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r639 + 0 ], { %r640, %r641, %r642, %r643 };
	// end inline asm
	add.s32 	%r644, %r634, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r644 + 0 ], { %r645, %r646, %r647, %r648 };
	// end inline asm
	add.s32 	%r649, %r634, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r649 + 0 ], { %r650, %r651, %r652, %r653 };
	// end inline asm
	xor.b32 	%r1236, %r1235, 32;
	add.s32 	%r654, %r194, %r1236;
	// begin inline asm
	st.shared.v4.b32 [ %r654 + 0 ], { %r655, %r656, %r657, %r658 };
	// end inline asm
	add.s32 	%r659, %r654, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r659 + 0 ], { %r660, %r661, %r662, %r663 };
	// end inline asm
	add.s32 	%r664, %r654, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r664 + 0 ], { %r665, %r666, %r667, %r668 };
	// end inline asm
	add.s32 	%r669, %r654, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r669 + 0 ], { %r670, %r671, %r672, %r673 };
	// end inline asm
	xor.b32 	%r1237, %r1235, 64;
	add.s32 	%r674, %r194, %r1237;
	// begin inline asm
	st.shared.v4.b32 [ %r674 + 0 ], { %r675, %r676, %r677, %r678 };
	// end inline asm
	add.s32 	%r679, %r674, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r679 + 0 ], { %r680, %r681, %r682, %r683 };
	// end inline asm
	add.s32 	%r684, %r674, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r684 + 0 ], { %r685, %r686, %r687, %r688 };
	// end inline asm
	add.s32 	%r689, %r674, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r689 + 0 ], { %r690, %r691, %r692, %r693 };
	// end inline asm
	xor.b32 	%r1238, %r1235, 96;
	add.s32 	%r694, %r194, %r1238;
	// begin inline asm
	st.shared.v4.b32 [ %r694 + 0 ], { %r695, %r696, %r697, %r698 };
	// end inline asm
	add.s32 	%r699, %r694, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r699 + 0 ], { %r700, %r701, %r702, %r703 };
	// end inline asm
	add.s32 	%r704, %r694, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r704 + 0 ], { %r705, %r706, %r707, %r708 };
	// end inline asm
	add.s32 	%r709, %r694, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r709 + 0 ], { %r710, %r711, %r712, %r713 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r1239, %r2, 2;
	and.b32 	%r1240, %r1239, 896;
	shl.b32 	%r1241, %r834, 9;
	selp.b32 	%r1242, 0, 8208, %p24;
	or.b32 	%r1243, %r876, %r1240;
	xor.b32 	%r1244, %r1243, %r1242;
	or.b32 	%r1245, %r1244, %r1241;
	add.s32 	%r1246, %r194, %r1245;
	ld.shared.v4.b32 	{%r714, %r730, %r746, %r762}, [%r1246];
	ld.shared.v4.b32 	{%r718, %r734, %r750, %r766}, [%r1246+1024];
	ld.shared.v4.b32 	{%r722, %r738, %r754, %r770}, [%r1246+2048];
	ld.shared.v4.b32 	{%r726, %r742, %r758, %r774}, [%r1246+3072];
	xor.b32 	%r1247, %r1245, 32;
	add.s32 	%r1248, %r194, %r1247;
	ld.shared.v4.b32 	{%r715, %r731, %r747, %r763}, [%r1248+16384];
	ld.shared.v4.b32 	{%r719, %r735, %r751, %r767}, [%r1248+17408];
	ld.shared.v4.b32 	{%r723, %r739, %r755, %r771}, [%r1248+18432];
	ld.shared.v4.b32 	{%r727, %r743, %r759, %r775}, [%r1248+19456];
	xor.b32 	%r1249, %r1245, 64;
	add.s32 	%r1250, %r194, %r1249;
	ld.shared.v4.b32 	{%r716, %r732, %r748, %r764}, [%r1250+32768];
	ld.shared.v4.b32 	{%r720, %r736, %r752, %r768}, [%r1250+33792];
	ld.shared.v4.b32 	{%r724, %r740, %r756, %r772}, [%r1250+34816];
	ld.shared.v4.b32 	{%r728, %r744, %r760, %r776}, [%r1250+35840];
	xor.b32 	%r1251, %r1245, 96;
	add.s32 	%r1252, %r194, %r1251;
	ld.shared.v4.b32 	{%r717, %r733, %r749, %r765}, [%r1252+49152];
	ld.shared.v4.b32 	{%r721, %r737, %r753, %r769}, [%r1252+50176];
	ld.shared.v4.b32 	{%r725, %r741, %r757, %r773}, [%r1252+51200];
	ld.shared.v4.b32 	{%r729, %r745, %r761, %r777}, [%r1252+52224];
	.loc	1 356 8                         // sk06_mlp_down.py:356:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd211 + 0 ], { %r714, %r715, %r716, %r717 };
	// end inline asm
	// begin inline asm
	@%p9 st.global.v4.b32 [ %rd212 + 0 ], { %r718, %r719, %r720, %r721 };
	// end inline asm
	// begin inline asm
	@%p10 st.global.v4.b32 [ %rd213 + 0 ], { %r722, %r723, %r724, %r725 };
	// end inline asm
	// begin inline asm
	@%p11 st.global.v4.b32 [ %rd214 + 0 ], { %r726, %r727, %r728, %r729 };
	// end inline asm
	// begin inline asm
	@%p12 st.global.v4.b32 [ %rd215 + 0 ], { %r730, %r731, %r732, %r733 };
	// end inline asm
	// begin inline asm
	@%p13 st.global.v4.b32 [ %rd216 + 0 ], { %r734, %r735, %r736, %r737 };
	// end inline asm
	// begin inline asm
	@%p14 st.global.v4.b32 [ %rd217 + 0 ], { %r738, %r739, %r740, %r741 };
	// end inline asm
	// begin inline asm
	@%p15 st.global.v4.b32 [ %rd218 + 0 ], { %r742, %r743, %r744, %r745 };
	// end inline asm
	// begin inline asm
	@%p16 st.global.v4.b32 [ %rd219 + 0 ], { %r746, %r747, %r748, %r749 };
	// end inline asm
	// begin inline asm
	@%p17 st.global.v4.b32 [ %rd220 + 0 ], { %r750, %r751, %r752, %r753 };
	// end inline asm
	// begin inline asm
	@%p18 st.global.v4.b32 [ %rd221 + 0 ], { %r754, %r755, %r756, %r757 };
	// end inline asm
	// begin inline asm
	@%p19 st.global.v4.b32 [ %rd222 + 0 ], { %r758, %r759, %r760, %r761 };
	// end inline asm
	// begin inline asm
	@%p20 st.global.v4.b32 [ %rd223 + 0 ], { %r762, %r763, %r764, %r765 };
	// end inline asm
	// begin inline asm
	@%p21 st.global.v4.b32 [ %rd224 + 0 ], { %r766, %r767, %r768, %r769 };
	// end inline asm
	// begin inline asm
	@%p22 st.global.v4.b32 [ %rd225 + 0 ], { %r770, %r771, %r772, %r773 };
	// end inline asm
	// begin inline asm
	@%p23 st.global.v4.b32 [ %rd226 + 0 ], { %r774, %r775, %r776, %r777 };
	// end inline asm
	.loc	1 354 4                         // sk06_mlp_down.py:354:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/workspace/vllm/_genesis/kernels/sk06_mlp_down.py"
	.file	2 "/usr/local/lib/python3.12/dist-packages/triton/language/standard.py"
	.section	.debug_abbrev
	{
.b8 1                                   // Abbreviation Code
.b8 17                                  // DW_TAG_compile_unit
.b8 1                                   // DW_CHILDREN_yes
.b8 37                                  // DW_AT_producer
.b8 8                                   // DW_FORM_string
.b8 19                                  // DW_AT_language
.b8 5                                   // DW_FORM_data2
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 16                                  // DW_AT_stmt_list
.b8 6                                   // DW_FORM_data4
.b8 27                                  // DW_AT_comp_dir
.b8 8                                   // DW_FORM_string
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 2                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 0                                   // DW_CHILDREN_no
.b8 3                                   // DW_AT_name
.b8 8                                   // DW_FORM_string
.b8 32                                  // DW_AT_inline
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 3                                   // Abbreviation Code
.b8 46                                  // DW_TAG_subprogram
.b8 1                                   // DW_CHILDREN_yes
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 4                                   // Abbreviation Code
.b8 29                                  // DW_TAG_inlined_subroutine
.b8 0                                   // DW_CHILDREN_no
.b8 49                                  // DW_AT_abstract_origin
.b8 19                                  // DW_FORM_ref4
.b8 17                                  // DW_AT_low_pc
.b8 1                                   // DW_FORM_addr
.b8 18                                  // DW_AT_high_pc
.b8 1                                   // DW_FORM_addr
.b8 88                                  // DW_AT_call_file
.b8 11                                  // DW_FORM_data1
.b8 89                                  // DW_AT_call_line
.b8 5                                   // DW_FORM_data2
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 168                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0xa1 DW_TAG_compile_unit
.b8 116                                 // DW_AT_producer
.b8 114
.b8 105
.b8 116
.b8 111
.b8 110
.b8 0
.b8 2                                   // DW_AT_language
.b8 0
.b8 115                                 // DW_AT_name
.b8 107
.b8 48
.b8 54
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 100
.b8 111
.b8 119
.b8 110
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 119
.b8 111
.b8 114
.b8 107
.b8 115
.b8 112
.b8 97
.b8 99
.b8 101
.b8 47
.b8 118
.b8 108
.b8 108
.b8 109
.b8 47
.b8 95
.b8 103
.b8 101
.b8 110
.b8 101
.b8 115
.b8 105
.b8 115
.b8 47
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 115
.b8 0
.b8 2                                   // Abbrev [2] 0x4b:0x18 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 54
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 100
.b8 111
.b8 119
.b8 110
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x63:0x48 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 75                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x78:0x19 DW_TAG_inlined_subroutine
.b32 75                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 58                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x91:0x19 DW_TAG_inlined_subroutine
.b32 75                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 59                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_11 = _Nativo(
    "sk06_mlp_down/tile256x128x64_shift1_abi16",
    _PTX_11, "_sk06_mlp_down_kernel",
    warps=8, shared=73728,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 17, 18],
    horneado={11: 1, 12: 1, 15: 1, 19: 1, 20: 256, 21: 128, 22: 64, 23: 8, 24: 128, 25: True},
    div16=[8, 9, 10, 13, 14, 16, 17],
)


# (cfg, has_shift) -> variantes candidatas. La eleccion final
# entre ellas la hace _lanzar comparando los valores horneados:
# con residual y sin residual el ABI tiene distinta cantidad de
# params, porque stride_res_n solo se especializa cuando vale 1.
_POR_CFG = {}
_POR_CFG.setdefault(((16, 128, 128, 8, 8, 3), False), []).append(_VAR_0)
_POR_CFG.setdefault(((16, 128, 128, 8, 8, 3), False), []).append(_VAR_1)
_POR_CFG.setdefault(((16, 128, 128, 8, 8, 3), True), []).append(_VAR_2)
_POR_CFG.setdefault(((16, 128, 128, 8, 8, 3), True), []).append(_VAR_3)
_POR_CFG.setdefault(((64, 128, 128, 8, 4, 4), False), []).append(_VAR_4)
_POR_CFG.setdefault(((64, 128, 128, 8, 4, 4), False), []).append(_VAR_5)
_POR_CFG.setdefault(((64, 128, 128, 8, 4, 4), True), []).append(_VAR_6)
_POR_CFG.setdefault(((64, 128, 128, 8, 4, 4), True), []).append(_VAR_7)
_POR_CFG.setdefault(((256, 128, 64, 8, 8, 4), False), []).append(_VAR_8)
_POR_CFG.setdefault(((256, 128, 64, 8, 8, 4), False), []).append(_VAR_9)
_POR_CFG.setdefault(((256, 128, 64, 8, 8, 4), True), []).append(_VAR_10)
_POR_CFG.setdefault(((256, 128, 64, 8, 8, 4), True), []).append(_VAR_11)


def _lanzar(grid, cfg, has_shift, *args):
    """Elige la variante cuyos valores horneados coinciden con estos args."""
    cands = _POR_CFG.get((cfg, bool(has_shift)))
    if not cands:
        raise KeyError("no hay PTX embebido para cfg=%r has_shift=%r" % (cfg, has_shift))
    for v in cands:
        if all(args[p] == val for p, val in v.horneado.items()):
            return v((grid,) if isinstance(grid, int) else grid, *args)
    raise KeyError("no hay variante para cfg=%r has_shift=%r con estos strides" % (cfg, has_shift))
