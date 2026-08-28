# SPDX-License-Identifier: Apache-2.0
"""SK-05 — MLP_GATEUP_FUSED_INT8_DIADIC — capa completa gate_up + SiLU en GPU.

Capa completa (Qwen3.5-27B, 64 capas + 1 MTP, TP=2):
    hidden bf16 [M, 5120]
      -> RMSNorm(post_attention_layernorm) + quant per-token  (kernel 1)
      -> GEMM INT8 diádico gate|up                            (kernel 2)
      -> SiLU(gate)*up  (+ quant per-token si el consumidor es INT8) (kernel 3)

Geometría: N global 34816, per-rank TP=2 N=17408 = gate 8704 | up 8704, K=5120.

Qué cambió respecto de la versión anterior
------------------------------------------
La anterior era un 3-stage con la mitad del trabajo en torch: RMSNorm en host
(``pow``/``mean``/``rsqrt``/``mul``/``to``/``contiguous`` -> ~7 lanzamientos más
una copia fp32 de M*5120), GEMM, y ``F.silu(gate)*up`` otra vez en torch. Eso
son **más** lanzamientos que el camino sin fusionar. Además recalculaba el
``amax`` per-token dentro del GEMM, dos pasadas completas sobre la activación
por cada uno de los 272 bloques de N.

Ahora son dos kernels y nada de torch:

  * **Kernel 1** — RMSNorm + quant per-token en una pasada: la fila entra una
    vez a registros y se reusa para varianza, amax y cuantización.
  * **Kernel 2** — GEMM INT8 diádico con tile 128x128x128.
  * **Kernel 3** — ``SiLU(gate)*up`` + quant per-token en una sola pasada.

Por qué SiLU no va dentro del GEMM: ``gate`` y ``up`` están separados por N/2
columnas, así que un tile tendría que cargar dos bloques de B a la vez. En
sm_86 eso obliga a ``BLOCK_N=64`` para no pasarse de los 99 KB de shared, y el
tile angosto cuesta 10-17% — más de lo que ahorra la fusión. Medido. En cambio
fusionar SiLU con el **quant de la capa siguiente** sí gana: ``down_proj``
consume INT8, así que ``mlp_gateup_int8_quantized`` entrega ``(q, s)`` y el
intermedio bf16 [M, 8704] nunca se materializa.

Diseño (sm_86, mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32):
  * ``BLOCK_K == SHIFT_BLOCK == 128`` -> un ``tl.dot`` por bloque diádico y por
    mitad, sin sub-bucle.
  * Shift por columna como ``acc += int_acc.to(f32) * exp2(shift)``: exacto y
    de una sola operación fp32.
  * Acumuladores fp32, escalas fuera del bucle k, punteros que avanzan,
    grid 1-D con swizzle L2.

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




HIDDEN_SIZE: int = 5120
INTERMEDIATE_SIZE: int = 17408
GATEUP_N_GLOBAL: int = 34816
GATEUP_N_PER_RANK: int = 17408
GATEUP_K: int = 5120

BLOCK: int = 128
SHIFT_BLOCK: int = 128

# (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages) por bucket de M.
# shared = (BLOCK_M + BLOCK_N) * BLOCK_K * num_stages <= 99 KB en sm_86.
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
    """Escalar cero por device, usado como epílogo neutro con strides 0."""
    z = _ZERO.get(device)
    if z is None:
        z = torch.zeros((), dtype=torch.bfloat16, device=device)
        _ZERO[device] = z
    return z


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
def _sk05_rmsnorm_quant_kernel(
    x_ptr, w_ptr, q_ptr, s_ptr,
    K, stride_xm, stride_qm,
    BLOCK: tl.constexpr, EPS: tl.constexpr,
):
    """RMSNorm + quant per-token en una pasada, fila por programa."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < K
    x = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * tl.rsqrt(tl.sum(x * x, axis=0) / K + EPS) * w
    amax = tl.maximum(tl.max(tl.abs(y), axis=0), 1e-30)
    yq = y * (127.0 / amax)
    q = (yq + tl.where(yq >= 0.0, 0.5, -0.5)).to(tl.int32)
    tl.store(q_ptr + row * stride_qm + offs, tl.minimum(tl.maximum(q, -127), 127).to(tl.int8), mask=mask)
    tl.store(s_ptr + row, amax * (1.0 / 127.0))


@triton.jit
def _sk05_mlp_gateup_kernel(
    a_ptr, b_ptr, out_ptr, resid_ptr, a_scale_ptr, b_scale_ptr, shifts_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_out_m, stride_out_n, stride_res_m, stride_res_n,
    stride_shift_k, stride_shift_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, SHIFT_BLOCK: tl.constexpr,
):
    """GEMM INT8 diádico gate|up -> ``[M, N]`` bf16."""
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, K // BLOCK_K):
        # `kb * stride_shift_k` indexaba FUERA DE RANGO: el shift es por
        # bloque de SHIFT_BLOCK=128 filas de K, no por iteracion, y con
        # BLOCK_K=64 hay dos iteraciones por bloque diadico. Con K=3072
        # son 48 iteraciones sobre una tabla de 24 filas. Coincidian solo
        # con BLOCK_K=128, asi que el fallo aparecia recien en el tile de
        # prefill: CUDA_ERROR_ILLEGAL_ADDRESS al lanzar.
        p2 = tl.exp2(tl.load(sh_ptrs + (kb * BLOCK_K // SHIFT_BLOCK)
                             * stride_shift_k).to(tl.float32))
        acc += tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), out_dtype=tl.float32).to(tl.float32) * p2[None, :]
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    acc = acc * tl.load(a_scale_ptr + offs_m).to(tl.float32)[:, None]
    acc = acc * tl.load(b_scale_ptr + offs_n).to(tl.float32)[None, :]
    acc += tl.load(resid_ptr + offs_m[:, None] * stride_res_m + offs_n[None, :] * stride_res_n).to(tl.float32)

    out_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(
        out_ptr + out_m[:, None] * stride_out_m + out_n[None, :] * stride_out_n,
        acc.to(out_ptr.dtype.element_ty),
        mask=(out_m[:, None] < M) & (out_n[None, :] < N),
    )


@triton.jit
def _sk05_gateup_silu_kernel(
    a_ptr, b_ptr, out_ptr, a_scale_ptr, b_scale_ptr, shifts_ptr,
    M, N2, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_out_m, stride_out_n,
    stride_shift_k, stride_shift_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, SHIFT_BLOCK: tl.constexpr,
):
    """GEMM INT8 diádico con **SiLU fusionado**, sobre el peso permutado.

    El peso llega con las columnas intercaladas ``gate0,up0,gate1,up1,...``
    (ver :func:`sk05_permute_gateup`), así que un tile de ``BLOCK_N=128``
    columnas contiene 64 pares gate/up completos. Eso permite resolver
    ``SiLU(gate)*up`` en el epílogo **sin cargar un solo byte extra de B**: el
    tile de B es tan ancho como el de un GEMM normal, y sólo la salida es la
    mitad. La versión con gate y up separados por N/2 obligaba a cargar dos
    tiles de B por paso y a bajar a ``BLOCK_N=64`` para no pasarse de los 99 KB
    de shared en sm_86, lo que costaba más de lo que ahorraba.

    El shift diádico sigue siendo uniforme dentro de cada mitad del tile: las
    64 columnas ``gate`` del tile caen todas en el mismo bloque de 128 del
    peso original, y las 64 ``up`` en otro. Dos escalares por iteración de k.
    """
    pid = tl.program_id(0)
    N = 2 * N2
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

    # Columna j del tile: par -> gate_j, impar -> up_j (peso original).
    j0 = pid_n * (BLOCK_N // 2)
    g_col = j0 // SHIFT_BLOCK
    u_col = (N2 + j0) // SHIFT_BLOCK
    is_up = (tl.arange(0, BLOCK_N) % 2) == 1

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, K // BLOCK_K):
        # `kb * stride_shift_k` indexaba FUERA DE RANGO: el shift es por
        # bloque de SHIFT_BLOCK=128 filas de K, no por iteracion, y con
        # BLOCK_K=64 hay dos iteraciones por bloque diadico. Con K=3072
        # son 48 iteraciones sobre una tabla de 24 filas. Coincidian solo
        # con BLOCK_K=128, asi que el fallo aparecia recien en el tile de
        # prefill: CUDA_ERROR_ILLEGAL_ADDRESS al lanzar.
        _kb = (kb * BLOCK_K // SHIFT_BLOCK) * stride_shift_k
        sh_g = tl.load(shifts_ptr + _kb + g_col * stride_shift_n).to(tl.float32)
        sh_u = tl.load(shifts_ptr + _kb + u_col * stride_shift_n).to(tl.float32)
        p2 = tl.exp2(tl.where(is_up, sh_u, sh_g))
        acc += tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), out_dtype=tl.float32).to(tl.float32) * p2[None, :]
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    acc = acc * tl.load(a_scale_ptr + offs_m).to(tl.float32)[:, None]
    acc = acc * tl.load(b_scale_ptr + offs_n).to(tl.float32)[None, :]
    gate, up = tl.split(tl.reshape(acc, (BLOCK_M, BLOCK_N // 2, 2)))
    out = gate * tl.sigmoid(gate) * up

    out_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_n = j0 + tl.arange(0, BLOCK_N // 2)
    tl.store(
        out_ptr + out_m[:, None] * stride_out_m + out_n[None, :] * stride_out_n,
        out.to(out_ptr.dtype.element_ty),
        mask=(out_m[:, None] < M) & (out_n[None, :] < N2),
    )


@triton.jit
def _sk05_silu_mul_quant_kernel(
    gu_ptr, q_ptr, s_ptr, N2, stride_gu_m, stride_q_m, BLOCK: tl.constexpr,
):
    """``SiLU(gate) * up`` + quant per-token INT8, una fila por programa.

    Lee ``gate_up`` [M, 2*N2] una sola vez y escribe directamente el INT8 que
    consume ``down_proj``: el tensor intermedio bf16 [M, N2] nunca existe.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N2
    base = gu_ptr + row * stride_gu_m
    g = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(base + N2 + offs, mask=mask, other=0.0).to(tl.float32)
    y = g * tl.sigmoid(g) * u
    amax = tl.maximum(tl.max(tl.abs(y), axis=0), 1e-30)
    yq = y * (127.0 / amax)
    q = (yq + tl.where(yq >= 0.0, 0.5, -0.5)).to(tl.int32)
    tl.store(q_ptr + row * stride_q_m + offs, tl.minimum(tl.maximum(q, -127), 127).to(tl.int8), mask=mask)
    tl.store(s_ptr + row, amax * (1.0 / 127.0))


@triton.jit
def _sk05_silu_mul_kernel(gu_ptr, out_ptr, N2, stride_gu_m, stride_o_m, BLOCK: tl.constexpr):
    """``SiLU(gate) * up`` -> bf16 [M, N2], para llamadores que quieren la activación."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N2
    base = gu_ptr + row * stride_gu_m
    g = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(base + N2 + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * stride_o_m + offs, (g * tl.sigmoid(g) * u).to(tl.bfloat16), mask=mask)


def sk05_rmsnorm_quant(
    hidden: torch.Tensor,
    ln_weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm + quant per-token INT8 -> ``(q [M,K] int8, s [M] fp32)``."""
    M, K = hidden.shape
    q = torch.empty((M, K), dtype=torch.int8, device=hidden.device)
    s = torch.empty((M,), dtype=torch.float32, device=hidden.device)
    _lanzar_quant0((M,),
            hidden, ln_weight, q, s, K, hidden.stride(0), q.stride(0), triton.next_power_of_2(K), eps)
    return q, s


def sk05_gateup_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    shifts: torch.Tensor,
    residual: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """GEMM INT8 diádico gate|up -> ``[M, N]`` bf16."""
    if b.stride(0) != 1:
        b = b.t().contiguous().t()
    M, K = a.shape
    N = b.shape[1]
    res = _zero(a.device) if residual is None else residual
    rm, rn = (0, 0) if residual is None else (res.stride(0), res.stride(1))
    bm, bn, bk, gm, warps, stages = _cfg(M, N)
    out = torch.empty((M, N), dtype=out_dtype, device=a.device)
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    if shifts is None or shifts.ndim == 0 or shifts.numel() == 0:
        shifts = torch.zeros((K // SHIFT_BLOCK, N // SHIFT_BLOCK), dtype=torch.float32, device=a.device)
    elif shifts.ndim == 1:
        if shifts.shape[0] == N // SHIFT_BLOCK:
            shifts = shifts.unsqueeze(0).expand(K // SHIFT_BLOCK, N // SHIFT_BLOCK).contiguous()
        else:
            shifts = torch.zeros((K // SHIFT_BLOCK, N // SHIFT_BLOCK), dtype=torch.float32, device=a.device)
    stride_shift_0 = shifts.stride(0)
    stride_shift_1 = shifts.stride(1)
    _lanzar(grid, (bm, bn, bk, gm, warps, stages), False,
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
            stride_shift_0,
            stride_shift_1,
            bm,
            bn,
            bk,
            gm,
            SHIFT_BLOCK)
    return out


def sk05_permute_gateup(
    b_col: torch.Tensor, b_scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reordena las columnas de ``gate_up`` a pares intercalados.

    De ``[gate_0..gate_{N2-1} | up_0..up_{N2-1}]`` a
    ``[gate_0, up_0, gate_1, up_1, ...]``. Se hace una sola vez, al cargar.
    Devuelve ``(w_perm [K,N] int8 column-major, b_scales_perm [N] fp32)``.

    Con este layout, ``_sk05_gateup_silu_kernel`` resuelve SiLU en el epílogo
    sin ampliar el tile de B ni tocar el shift diádico.
    """
    K, N = b_col.shape
    n2 = N // 2
    perm = torch.empty(N, dtype=torch.long, device=b_col.device)
    idx = torch.arange(n2, dtype=torch.long, device=b_col.device)
    perm[0::2] = idx
    perm[1::2] = idx + n2
    # [K,N] col-major -> [N,K] contiguo -> permutar filas -> volver a col-major
    w_perm = b_col.t()[perm].contiguous().t()
    return w_perm, b_scales.reshape(-1)[perm].contiguous()


def sk05_gateup_silu_gemm(
    a: torch.Tensor,
    b_perm: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales_perm: torch.Tensor,
    shifts: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """GEMM INT8 diádico + SiLU fusionado -> ``[M, N//2]``.

    ``b_perm``/``b_scales_perm`` vienen de :func:`sk05_permute_gateup`;
    ``shifts`` es el original ``[K/128, N/128]``, sin permutar.
    """
    M, K = a.shape
    N = b_perm.shape[1]
    n2 = N // 2
    bm, bn, bk, gm, warps, stages = _cfg(M, N)
    out = torch.empty((M, n2), dtype=out_dtype, device=a.device)
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    _sk05_gateup_silu_kernel[grid](
        a, b_perm, out, a_scales, b_scales_perm, shifts,
        M, n2, K,
        a.stride(0), a.stride(1), b_perm.stride(0), b_perm.stride(1),
        out.stride(0), out.stride(1),
        shifts.stride(0), shifts.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=gm, SHIFT_BLOCK=SHIFT_BLOCK,
        num_warps=warps, num_stages=stages,
    )
    return out


def sk05_silu_mul_quant(gate_up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``SiLU(gate)*up`` + quant per-token -> ``(q [M,N/2] int8, s [M] fp32)``."""
    M, N = gate_up.shape
    n2 = N // 2
    q = torch.empty((M, n2), dtype=torch.int8, device=gate_up.device)
    s = torch.empty((M,), dtype=torch.float32, device=gate_up.device)
    _sk05_silu_mul_quant_kernel[(M,)](
        gate_up, q, s, n2, gate_up.stride(0), q.stride(0),
        BLOCK=triton.next_power_of_2(n2), num_warps=8, num_stages=1,
    )
    return q, s


def sk05_silu_mul(gate_up: torch.Tensor, out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """``SiLU(gate)*up`` -> ``[M, N/2]`` bf16."""
    M, N = gate_up.shape
    n2 = N // 2
    out = torch.empty((M, n2), dtype=out_dtype, device=gate_up.device)
    _sk05_silu_mul_kernel[(M,)](
        gate_up, out, n2, gate_up.stride(0), out.stride(0),
        BLOCK=triton.next_power_of_2(n2), num_warps=8, num_stages=1,
    )
    return out


def mlp_gateup_int8_quantized(
    hidden: torch.Tensor,
    gateup_weight: torch.Tensor,
    gateup_scale: torch.Tensor,
    gateup_shifts: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Capa completa gate_up entregando INT8 listo para ``down_proj``.

    RMSNorm+quant -> GEMM diádico -> SiLU*mul+quant. Devuelve
    ``(q [M, N/2] int8, s [M] fp32)``. El intermedio bf16 [M, N/2] nunca se
    materializa: ``down_proj`` (SK-06) consume ``q``/``s`` directamente.
    """
    a, a_scales = sk05_rmsnorm_quant(hidden, post_attention_layernorm_weight, eps)
    gate_up = sk05_gateup_gemm(a, gateup_weight, a_scales, gateup_scale, gateup_shifts, None)
    return sk05_silu_mul_quant(gate_up)


def mlp_gateup_fused_int8_diadic(
    hidden: torch.Tensor,
    gateup_weight: torch.Tensor,
    gateup_scale: torch.Tensor,
    gateup_shifts: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor,
    eps: float = 1e-6,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Capa completa gate_up devolviendo ``SiLU(gate)*up`` ``[M, N//2]`` bf16.

    RMSNorm+quant -> GEMM diádico -> SiLU*mul. Cero operaciones torch.
    """
    a, a_scales = sk05_rmsnorm_quant(hidden, post_attention_layernorm_weight, eps)
    gate_up = sk05_gateup_gemm(a, gateup_weight, a_scales, gateup_scale, gateup_shifts, None, out_dtype)
    return sk05_silu_mul(gate_up, out_dtype)


gateup_fused = mlp_gateup_fused_int8_diadic
mlp_gateup_fused = mlp_gateup_fused_int8_diadic
sk05_mlp_gateup = mlp_gateup_fused_int8_diadic

__all__ = [
    "HIDDEN_SIZE", "INTERMEDIATE_SIZE", "GATEUP_N_GLOBAL", "GATEUP_N_PER_RANK",
    "GATEUP_K", "BLOCK", "SHIFT_BLOCK", "BLOCK_M", "BLOCK_N", "BLOCK_K",
    "sk05_rmsnorm_quant", "sk05_gateup_gemm", "sk05_silu_mul", "sk05_silu_mul_quant",
    "sk05_permute_gateup", "sk05_gateup_silu_gemm",
    "mlp_gateup_int8_quantized", "mlp_gateup_fused_int8_diadic",
    "gateup_fused", "mlp_gateup_fused", "sk05_mlp_gateup",
]


# --- variantes PTX embebidas ---

_PTX_0 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk05_mlp_gateup_kernel // -- Begin function _sk05_mlp_gateup_kernel
.extern .shared .align 16 .b8 global_smem[];
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
                                        // @_sk05_mlp_gateup_kernel
.visible .entry _sk05_mlp_gateup_kernel(
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_6,
	.param .u32 _sk05_mlp_gateup_kernel_param_7,
	.param .u32 _sk05_mlp_gateup_kernel_param_8,
	.param .u32 _sk05_mlp_gateup_kernel_param_9,
	.param .u32 _sk05_mlp_gateup_kernel_param_10,
	.param .u32 _sk05_mlp_gateup_kernel_param_11,
	.param .u32 _sk05_mlp_gateup_kernel_param_12,
	.param .u32 _sk05_mlp_gateup_kernel_param_13,
	.param .u32 _sk05_mlp_gateup_kernel_param_14,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_16
)
.reqntid 256
{
	.reg .pred 	%p<11>;
	.reg .b16 	%rs<17>;
	.reg .b32 	%r<314>;
	.reg .b64 	%rd<68>;
	.loc	1 307 0                         // sk05_mlp_gateup.py:307:0
$L__func_begin0:
	.loc	1 307 0                         // sk05_mlp_gateup.py:307:0

// %bb.0:
	ld.param.b32 	%r24, [_sk05_mlp_gateup_kernel_param_13];
	ld.param.b32 	%r23, [_sk05_mlp_gateup_kernel_param_12];
	ld.param.b32 	%r22, [_sk05_mlp_gateup_kernel_param_9];
	ld.param.b32 	%r21, [_sk05_mlp_gateup_kernel_param_8];
	ld.param.b32 	%r20, [_sk05_mlp_gateup_kernel_param_7];
	ld.param.b64 	%rd24, [_sk05_mlp_gateup_kernel_param_5];
	ld.param.b64 	%rd23, [_sk05_mlp_gateup_kernel_param_4];
	ld.param.b64 	%rd22, [_sk05_mlp_gateup_kernel_param_3];
	ld.param.b64 	%rd21, [_sk05_mlp_gateup_kernel_param_2];
	ld.param.b64 	%rd20, [_sk05_mlp_gateup_kernel_param_1];
	ld.param.b64 	%rd19, [_sk05_mlp_gateup_kernel_param_0];
$L__tmp0:
	.loc	1 317 24                        // sk05_mlp_gateup.py:317:24
	mov.u32 	%r40, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk05_mlp_gateup.py:318:27 ]
	add.s32 	%r41, %r20, 15;
	.loc	2 43 30                         // standard.py:43:30 @[ sk05_mlp_gateup.py:318:27 ]
	shr.s32 	%r42, %r41, 31;
	shr.u32 	%r43, %r42, 28;
	add.s32 	%r44, %r41, %r43;
	shr.s32 	%r45, %r44, 4;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk05_mlp_gateup.py:319:27 ]
	add.s32 	%r46, %r21, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk05_mlp_gateup.py:319:27 ]
	shr.s32 	%r47, %r46, 31;
	shr.u32 	%r48, %r47, 25;
	add.s32 	%r49, %r46, %r48;
	shr.s32 	%r50, %r49, 7;
$L__tmp3:
	.loc	1 320 29                        // sk05_mlp_gateup.py:320:29
	shl.b32 	%r51, %r50, 3;
	.loc	1 321 22                        // sk05_mlp_gateup.py:321:22
	div.s32 	%r52, %r40, %r51;
	.loc	1 321 38                        // sk05_mlp_gateup.py:321:38
	shl.b32 	%r53, %r52, 3;
	.loc	1 322 30                        // sk05_mlp_gateup.py:322:30
	sub.s32 	%r54, %r45, %r53;
	ld.param.b32 	%r55, [_sk05_mlp_gateup_kernel_param_10];
	.loc	1 322 39                        // sk05_mlp_gateup.py:322:39
	min.s32 	%r56, %r54, 8;
	ld.param.b32 	%r57, [_sk05_mlp_gateup_kernel_param_11];
	.loc	1 323 30                        // sk05_mlp_gateup.py:323:30
	mul.lo.s32 	%r58, %r52, %r51;
	sub.s32 	%r59, %r40, %r58;
	.loc	1 324 36                        // sk05_mlp_gateup.py:324:36
	div.s32 	%r60, %r59, %r56;
	.loc	1 323 46                        // sk05_mlp_gateup.py:323:46
	mul.lo.s32 	%r61, %r60, %r56;
	sub.s32 	%r62, %r59, %r61;
	.loc	1 323 23                        // sk05_mlp_gateup.py:323:23
	add.s32 	%r63, %r62, %r53;
	.loc	1 326 22                        // sk05_mlp_gateup.py:326:22
	shl.b32 	%r1, %r63, 4;
	.loc	1 326 45                        // sk05_mlp_gateup.py:326:45
	mov.u32 	%r2, %tid.x;
	and.b32 	%r3, %r2, 240;
	bfe.u32 	%r64, %r2, 4, 4;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r4, %r1, %r64;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r5, %r4, %r20;
	.loc	1 327 22                        // sk05_mlp_gateup.py:327:22
	shl.b32 	%r6, %r60, 7;
	.loc	1 327 45                        // sk05_mlp_gateup.py:327:45
	shr.u32 	%r65, %r2, 3;
	bfe.u32 	%r66, %r2, 3, 5;
	shl.b32 	%r7, %r2, 1;
	and.b32 	%r67, %r7, 6;
	and.b32 	%r8, %r2, 224;
	shr.u32 	%r68, %r8, 2;
	or.b32 	%r69, %r67, %r68;
	and.b32 	%r9, %r2, 15;
	shl.b32 	%r70, %r9, 3;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r71, %r6, %r66;
	or.b32 	%r72, %r71, 32;
	or.b32 	%r73, %r71, 64;
	or.b32 	%r74, %r65, %r6;
	or.b32 	%r75, %r74, 96;
	or.b32 	%r76, %r6, %r69;
	or.b32 	%r78, %r76, 64;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r80, %r71, %r21;
	rem.s32 	%r81, %r72, %r21;
	rem.s32 	%r82, %r73, %r21;
	rem.s32 	%r83, %r75, %r21;
	rem.s32 	%r10, %r76, %r21;
	rem.s32 	%r11, %r78, %r21;
	.loc	1 330 39                        // sk05_mlp_gateup.py:330:39
	mul.lo.s32 	%r86, %r5, %r55;
	.loc	1 330 21                        // sk05_mlp_gateup.py:330:21
	cvt.s64.s32 	%rd1, %r86;
	add.s64 	%rd36, %rd19, %rd1;
	.loc	1 330 51                        // sk05_mlp_gateup.py:330:51
	cvt.u64.u32 	%rd2, %r70;
	add.s64 	%rd25, %rd36, %rd2;
	.loc	1 331 28                        // sk05_mlp_gateup.py:331:28
	and.b32 	%r12, %r2, 7;
	shl.b32 	%r87, %r12, 4;
	.loc	1 331 21                        // sk05_mlp_gateup.py:331:21
	cvt.u64.u32 	%rd3, %r87;
	add.s64 	%rd37, %rd20, %rd3;
	.loc	1 331 69                        // sk05_mlp_gateup.py:331:69
	mul.lo.s32 	%r88, %r80, %r57;
	mul.lo.s32 	%r89, %r81, %r57;
	mul.lo.s32 	%r90, %r82, %r57;
	mul.lo.s32 	%r91, %r83, %r57;
	.loc	1 331 51                        // sk05_mlp_gateup.py:331:51
	cvt.s64.s32 	%rd4, %r88;
	add.s64 	%rd26, %rd37, %rd4;
	cvt.s64.s32 	%rd5, %r89;
	add.s64 	%rd27, %rd37, %rd5;
	cvt.s64.s32 	%rd6, %r90;
	add.s64 	%rd28, %rd37, %rd6;
	cvt.s64.s32 	%rd7, %r91;
	add.s64 	%rd29, %rd37, %rd7;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	setp.lt.s32 	%p1, %r22, 128;
	setp.gt.s32 	%p2, %r22, 127;
	.loc	1 344 30                        // sk05_mlp_gateup.py:344:30
	and.b32 	%r108, %r2, 255;
	shl.b32 	%r109, %r108, 3;
	and.b32 	%r110, %r2, 112;
	xor.b32 	%r111, %r109, %r110;
	mov.b32 	%r112, global_smem;
	add.s32 	%r13, %r112, %r111;
	add.s32 	%r26, %r13, 32768;
	selp.b32 	%r27, 8, 0, %p2;
	// begin inline asm
	cp.async.ca.shared.global [ %r26 + 0 ], [ %rd25 + 0 ], 0x8, %r27;
	// end inline asm
	cp.async.commit_group;
	.loc	1 344 47                        // sk05_mlp_gateup.py:344:47
	shl.b32 	%r113, %r108, 4;
	and.b32 	%r114, %r7, 112;
	xor.b32 	%r115, %r113, %r114;
	add.s32 	%r28, %r112, %r115;
	selp.b32 	%r29, 16, 0, %p2;
	// begin inline asm
	cp.async.cg.shared.global [ %r28 + 0 ], [ %rd26 + 0 ], 0x10, %r29;
	// end inline asm
	add.s32 	%r30, %r28, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r30 + 0 ], [ %rd27 + 0 ], 0x10, %r29;
	// end inline asm
	add.s32 	%r31, %r28, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r31 + 0 ], [ %rd28 + 0 ], 0x10, %r29;
	// end inline asm
	add.s32 	%r32, %r28, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r32 + 0 ], [ %rd29 + 0 ], 0x10, %r29;
	// end inline asm
	cp.async.commit_group;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	setp.gt.s32 	%p3, %r22, 255;
	.loc	1 345 18                        // sk05_mlp_gateup.py:345:18
	add.s64 	%rd30, %rd25, 128;
	.loc	1 346 18                        // sk05_mlp_gateup.py:346:18
	add.s64 	%rd31, %rd26, 128;
	add.s64 	%rd32, %rd27, 128;
	add.s64 	%rd33, %rd28, 128;
	add.s64 	%rd34, %rd29, 128;
	.loc	1 344 30                        // sk05_mlp_gateup.py:344:30
	bar.sync 	0;
	add.s32 	%r33, %r13, 34816;
	selp.b32 	%r34, 8, 0, %p3;
	// begin inline asm
	cp.async.ca.shared.global [ %r33 + 0 ], [ %rd30 + 0 ], 0x8, %r34;
	// end inline asm
	cp.async.commit_group;
	.loc	1 344 47                        // sk05_mlp_gateup.py:344:47
	add.s32 	%r35, %r28, 16384;
	selp.b32 	%r36, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r35 + 0 ], [ %rd31 + 0 ], 0x10, %r36;
	// end inline asm
	add.s32 	%r37, %r28, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r37 + 0 ], [ %rd32 + 0 ], 0x10, %r36;
	// end inline asm
	add.s32 	%r38, %r28, 24576;
	// begin inline asm
	cp.async.cg.shared.global [ %r38 + 0 ], [ %rd33 + 0 ], 0x10, %r36;
	// end inline asm
	add.s32 	%r39, %r28, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r39 + 0 ], [ %rd34 + 0 ], 0x10, %r36;
	// end inline asm
	cp.async.commit_group;
	mov.b32 	%r306, 0f00000000;
	cvt.u32.u64 	%r302, %rd3;
	mov.b32 	%r307, %r306;
	mov.b32 	%r308, %r306;
	mov.b32 	%r309, %r306;
	mov.b32 	%r310, %r306;
	mov.b32 	%r311, %r306;
	mov.b32 	%r312, %r306;
	mov.b32 	%r313, %r306;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	@%p1 bra 	$L__BB0_3;
// %bb.1:                               // %.lr.ph
	.loc	1 0 23                          // sk05_mlp_gateup.py:0:23
	ld.param.b32 	%r25, [_sk05_mlp_gateup_kernel_param_14];
	ld.param.b64 	%rd35, [_sk05_mlp_gateup_kernel_param_6];
	or.b32 	%r77, %r76, 1;
	or.b32 	%r79, %r76, 65;
	rem.s32 	%r84, %r77, %r21;
	rem.s32 	%r85, %r79, %r21;
	shr.s32 	%r92, %r10, 31;
	shr.u32 	%r93, %r92, 25;
	add.s32 	%r94, %r10, %r93;
	shr.s32 	%r95, %r94, 7;
	shr.s32 	%r96, %r84, 31;
	shr.u32 	%r97, %r96, 25;
	add.s32 	%r98, %r84, %r97;
	shr.s32 	%r99, %r98, 7;
	shr.s32 	%r100, %r11, 31;
	shr.u32 	%r101, %r100, 25;
	add.s32 	%r102, %r11, %r101;
	shr.s32 	%r103, %r102, 7;
	shr.s32 	%r104, %r85, 31;
	shr.u32 	%r105, %r104, 25;
	add.s32 	%r106, %r85, %r105;
	shr.s32 	%r107, %r106, 7;
	mad.wide.s32 	%rd8, %r95, 4, %rd35;
	mad.wide.s32 	%rd9, %r99, 4, %rd35;
	mad.wide.s32 	%rd10, %r103, 4, %rd35;
	mad.wide.s32 	%rd11, %r107, 4, %rd35;
	.loc	1 335 28                        // sk05_mlp_gateup.py:335:28
	shr.u32 	%r116, %r22, 7;
	add.s32 	%r117, %r116, -2;
	shl.b32 	%r118, %r9, 7;
	and.b32 	%r119, %r2, 16;
	xor.b32 	%r120, %r302, %r119;
	or.b32 	%r14, %r120, %r118;
	xor.b32 	%r15, %r14, 32;
	xor.b32 	%r16, %r14, 64;
	xor.b32 	%r17, %r14, 96;
	shl.b32 	%r121, %r12, 7;
	shl.b32 	%r122, %r8, 5;
	and.b32 	%r123, %r7, 48;
	or.b32 	%r124, %r121, %r122;
	xor.b32 	%r125, %r302, %r123;
	or.b32 	%r18, %r124, %r125;
	xor.b32 	%r19, %r18, 64;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	cvt.s64.s32 	%rd12, %r117;
	and.b32 	%r126, %r22, -128;
	cvt.u64.u32 	%rd13, %r126;
	add.s64 	%rd38, %rd3, %rd7;
	add.s64 	%rd39, %rd38, %rd20;
	add.s64 	%rd14, %rd39, 256;
	add.s64 	%rd40, %rd3, %rd6;
	add.s64 	%rd41, %rd40, %rd20;
	add.s64 	%rd15, %rd41, 256;
	add.s64 	%rd42, %rd3, %rd5;
	add.s64 	%rd43, %rd42, %rd20;
	add.s64 	%rd16, %rd43, 256;
	add.s64 	%rd44, %rd3, %rd4;
	add.s64 	%rd45, %rd44, %rd20;
	add.s64 	%rd17, %rd45, 256;
	add.s64 	%rd46, %rd2, %rd1;
	add.s64 	%rd47, %rd46, %rd19;
	add.s64 	%rd18, %rd47, 256;
	mov.b32 	%r306, 0f00000000;
	mov.b32 	%r305, 1;
	mov.b32 	%r304, -1;
	mov.b64 	%rd66, 0;
	mov.b32 	%r131, 0;
	mov.b32 	%r303, %r131;
	mov.b64 	%rd67, %rd66;
	mov.b32 	%r307, %r306;
	mov.b32 	%r308, %r306;
	mov.b32 	%r309, %r306;
	mov.b32 	%r310, %r306;
	mov.b32 	%r311, %r306;
	mov.b32 	%r312, %r306;
	mov.b32 	%r313, %r306;
$L__BB0_2:                              // %__nv_exp2f.exit
                                        // =>This Inner Loop Header: Depth=1
	setp.lt.s64 	%p4, %rd67, %rd12;
	add.s32 	%r179, %r304, 1;
	setp.gt.s32 	%p5, %r179, 1;
	selp.b32 	%r304, 0, %r179, %p5;
	.loc	1 342 39                        // sk05_mlp_gateup.py:342:39
	mul.wide.s32 	%rd57, %r303, 4;
	add.s64 	%rd48, %rd8, %rd57;
	add.s64 	%rd49, %rd9, %rd57;
	add.s64 	%rd50, %rd10, %rd57;
	add.s64 	%rd51, %rd11, %rd57;
	.loc	1 342 29                        // sk05_mlp_gateup.py:342:29
	// begin inline asm
	mov.u32 %r127, 0x0;
	ld.global.b32 { %r127 }, [ %rd48 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r128, 0x0;
	ld.global.b32 { %r128 }, [ %rd49 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r129, 0x0;
	ld.global.b32 { %r129 }, [ %rd50 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r130, 0x0;
	ld.global.b32 { %r130 }, [ %rd51 + 0 ];
	// end inline asm
	.loc	1 342 21                        // sk05_mlp_gateup.py:342:21
	ex2.approx.ftz.f32 	%r180, %r127;
	ex2.approx.ftz.f32 	%r181, %r128;
	ex2.approx.ftz.f32 	%r182, %r129;
	ex2.approx.ftz.f32 	%r183, %r130;
	.loc	1 344 30                        // sk05_mlp_gateup.py:344:30
	cp.async.wait_group 	2;
	bar.sync 	0;
	shl.b32 	%r184, %r304, 11;
	add.s32 	%r185, %r112, %r184;
	add.s32 	%r186, %r185, %r14;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r132, %r133, %r134, %r135}, [%r186+32768];
	add.s32 	%r187, %r185, %r15;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r144, %r145, %r146, %r147}, [%r187+32768];
	add.s32 	%r188, %r185, %r16;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r156, %r157, %r158, %r159}, [%r188+32768];
	add.s32 	%r189, %r185, %r17;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r164, %r165, %r166, %r167}, [%r189+32768];
	.loc	1 344 47                        // sk05_mlp_gateup.py:344:47
	shl.b32 	%r190, %r304, 14;
	add.s32 	%r191, %r112, %r190;
	add.s32 	%r192, %r191, %r18;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r136, %r137, %r148, %r149}, [%r192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r138, %r139, %r154, %r155}, [%r192+8192];
	add.s32 	%r193, %r191, %r19;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r160, %r161, %r168, %r169}, [%r193];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r162, %r163, %r170, %r171}, [%r193+8192];
	.loc	1 344 39                        // sk05_mlp_gateup.py:344:39
	mov.b32 	%r140, %r131;
	mov.b32 	%r141, %r131;
	mov.b32 	%r142, %r131;
	mov.b32 	%r143, %r131;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r140, %r141, %r142, %r143 }, { %r132, %r133, %r134, %r135 }, { %r136, %r137 }, { %r140, %r141, %r142, %r143 };
	// end inline asm
	mov.b32 	%r150, %r131;
	mov.b32 	%r151, %r131;
	mov.b32 	%r152, %r131;
	mov.b32 	%r153, %r131;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r150, %r151, %r152, %r153 }, { %r132, %r133, %r134, %r135 }, { %r138, %r139 }, { %r150, %r151, %r152, %r153 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r140, %r141, %r142, %r143 }, { %r144, %r145, %r146, %r147 }, { %r148, %r149 }, { %r140, %r141, %r142, %r143 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r150, %r151, %r152, %r153 }, { %r144, %r145, %r146, %r147 }, { %r154, %r155 }, { %r150, %r151, %r152, %r153 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r140, %r141, %r142, %r143 }, { %r156, %r157, %r158, %r159 }, { %r160, %r161 }, { %r140, %r141, %r142, %r143 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r150, %r151, %r152, %r153 }, { %r156, %r157, %r158, %r159 }, { %r162, %r163 }, { %r150, %r151, %r152, %r153 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r140, %r141, %r142, %r143 }, { %r164, %r165, %r166, %r167 }, { %r168, %r169 }, { %r140, %r141, %r142, %r143 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r150, %r151, %r152, %r153 }, { %r164, %r165, %r166, %r167 }, { %r170, %r171 }, { %r150, %r151, %r152, %r153 };
	// end inline asm
	.loc	1 344 81                        // sk05_mlp_gateup.py:344:81
	cvt.rn.f32.s32 	%r194, %r150;
	cvt.rn.f32.s32 	%r195, %r151;
	cvt.rn.f32.s32 	%r196, %r152;
	cvt.rn.f32.s32 	%r197, %r153;
	cvt.rn.f32.s32 	%r198, %r140;
	cvt.rn.f32.s32 	%r199, %r141;
	cvt.rn.f32.s32 	%r200, %r142;
	cvt.rn.f32.s32 	%r201, %r143;
	.loc	1 344 15                        // sk05_mlp_gateup.py:344:15
	fma.rn.f32 	%r309, %r181, %r201, %r309;
	fma.rn.f32 	%r308, %r180, %r200, %r308;
	fma.rn.f32 	%r307, %r181, %r199, %r307;
	fma.rn.f32 	%r306, %r180, %r198, %r306;
	fma.rn.f32 	%r313, %r183, %r197, %r313;
	fma.rn.f32 	%r312, %r182, %r196, %r312;
	fma.rn.f32 	%r311, %r183, %r195, %r311;
	fma.rn.f32 	%r310, %r182, %r194, %r310;
	.loc	1 346 18                        // sk05_mlp_gateup.py:346:18
	add.s64 	%rd52, %rd18, %rd66;
	add.s64 	%rd53, %rd17, %rd66;
	add.s64 	%rd54, %rd16, %rd66;
	add.s64 	%rd55, %rd15, %rd66;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	add.s64 	%rd56, %rd14, %rd66;
	add.s32 	%r202, %r305, 1;
	setp.gt.s32 	%p6, %r202, 1;
	selp.b32 	%r305, 0, %r202, %p6;
	.loc	1 344 30                        // sk05_mlp_gateup.py:344:30
	shl.b32 	%r203, %r305, 11;
	bar.sync 	0;
	add.s32 	%r204, %r13, %r203;
	add.s32 	%r172, %r204, 32768;
	selp.b32 	%r173, 8, 0, %p4;
	// begin inline asm
	cp.async.ca.shared.global [ %r172 + 0 ], [ %rd52 + 0 ], 0x8, %r173;
	// end inline asm
	cp.async.commit_group;
	.loc	1 344 47                        // sk05_mlp_gateup.py:344:47
	shl.b32 	%r205, %r305, 14;
	add.s32 	%r174, %r28, %r205;
	selp.b32 	%r175, 16, 0, %p4;
	// begin inline asm
	cp.async.cg.shared.global [ %r174 + 0 ], [ %rd53 + 0 ], 0x10, %r175;
	// end inline asm
	add.s32 	%r176, %r174, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r176 + 0 ], [ %rd54 + 0 ], 0x10, %r175;
	// end inline asm
	add.s32 	%r177, %r174, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r177 + 0 ], [ %rd55 + 0 ], 0x10, %r175;
	// end inline asm
	add.s32 	%r178, %r174, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r178 + 0 ], [ %rd56 + 0 ], 0x10, %r175;
	// end inline asm
	cp.async.commit_group;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	add.s64 	%rd67, %rd67, 1;
	add.s64 	%rd66, %rd66, 128;
	add.s32 	%r303, %r303, %r25;
	setp.ne.b64 	%p7, %rd13, %rd66;
	@%p7 bra 	$L__BB0_2;
$L__BB0_3:                              // %._crit_edge
	.loc	1 0 23                          // sk05_mlp_gateup.py:0:23
	cvt.u32.u64 	%r225, %rd2;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r226, %r6, %r225;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r227, %r226, %r21;
	.loc	1 326 45                        // sk05_mlp_gateup.py:326:45
	and.b32 	%r228, %r2, 28;
	bfe.u32 	%r229, %r2, 2, 3;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r230, %r229, %r1;
	or.b32 	%r231, %r230, 8;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r232, %r231, %r20;
	rem.s32 	%r233, %r230, %r20;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 348 38                        // sk05_mlp_gateup.py:348:38
	mad.wide.s32 	%rd58, %r233, 4, %rd23;
	mad.wide.s32 	%rd59, %r232, 4, %rd23;
	.loc	1 348 24                        // sk05_mlp_gateup.py:348:24
	// begin inline asm
	mov.u32 %r206, 0x0;
	ld.global.b32 { %r206 }, [ %rd58 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r207, 0x0;
	ld.global.b32 { %r207 }, [ %rd59 + 0 ];
	// end inline asm
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r234, %r306, %r206;
	mul.f32 	%r235, %r307, %r206;
	mul.f32 	%r236, %r308, %r207;
	mul.f32 	%r237, %r309, %r207;
	mul.f32 	%r238, %r310, %r206;
	mul.f32 	%r239, %r311, %r206;
	mul.f32 	%r240, %r312, %r207;
	mul.f32 	%r241, %r313, %r207;
	.loc	1 349 38                        // sk05_mlp_gateup.py:349:38
	mad.wide.s32 	%rd60, %r10, 4, %rd24;
	mad.wide.s32 	%rd61, %r11, 4, %rd24;
	.loc	1 349 24                        // sk05_mlp_gateup.py:349:24
	// begin inline asm
	mov.u32 %r208, 0x0;
	mov.u32 %r209, 0x0;
	ld.global.v2.b32 { %r208, %r209 }, [ %rd60 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r210, 0x0;
	mov.u32 %r211, 0x0;
	ld.global.v2.b32 { %r210, %r211 }, [ %rd61 + 0 ];
	// end inline asm
	.loc	1 350 49                        // sk05_mlp_gateup.py:350:49
	mul.lo.s32 	%r242, %r5, %r24;
	.loc	1 350 31                        // sk05_mlp_gateup.py:350:31
	mad.wide.s32 	%rd64, %r242, 2, %rd22;
	.loc	1 350 64                        // sk05_mlp_gateup.py:350:64
	mad.wide.s32 	%rd62, %r227, 2, %rd64;
	.loc	1 350 19                        // sk05_mlp_gateup.py:350:19
	// begin inline asm
	mov.u32 %r213, 0x0;
	mov.u32 %r214, 0x0;
	mov.u32 %r215, 0x0;
	mov.u32 %r216, 0x0;
	ld.global.v4.b32 { %r213, %r214, %r215, %r216 }, [ %rd62 + 0 ];
	// end inline asm
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	and.b32 	%r243, %r2, 120;
	shl.b32 	%r244, %r243, 5;
	or.b32 	%r245, %r244, %r302;
	xor.b32 	%r246, %r245, %r3;
	add.s32 	%r212, %r112, %r246;
	// begin inline asm
	st.shared.v4.b32 [ %r212 + 0 ], { %r213, %r214, %r215, %r216 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r247, %r12, 9;
	shl.b32 	%r248, %r2, 4;
	and.b32 	%r249, %r248, 496;
	shr.u32 	%r250, %r8, 1;
	xor.b32 	%r251, %r249, %r250;
	add.s32 	%r252, %r112, %r247;
	add.s32 	%r253, %r252, %r251;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r254, %r255, %r256, %r257}, [%r253];
	mov.b32 	{%rs9, %rs10}, %r254;
	mov.b32 	{%rs11, %rs12}, %r255;
	mov.b32 	{%rs13, %rs14}, %r256;
	mov.b32 	{%rs15, %rs16}, %r257;
	cvt.f32.bf16 	%r258, %rs9;
	cvt.f32.bf16 	%r259, %rs10;
	cvt.f32.bf16 	%r260, %rs11;
	cvt.f32.bf16 	%r261, %rs12;
	cvt.f32.bf16 	%r262, %rs13;
	cvt.f32.bf16 	%r263, %rs14;
	cvt.f32.bf16 	%r264, %rs15;
	cvt.f32.bf16 	%r265, %rs16;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r266, %r234, %r208, %r258;
	fma.rn.f32 	%r267, %r235, %r209, %r259;
	fma.rn.f32 	%r268, %r236, %r208, %r260;
	fma.rn.f32 	%r269, %r237, %r209, %r261;
	fma.rn.f32 	%r270, %r238, %r210, %r262;
	fma.rn.f32 	%r271, %r239, %r211, %r263;
	fma.rn.f32 	%r272, %r240, %r210, %r264;
	fma.rn.f32 	%r273, %r241, %r211, %r265;
	.loc	1 357 31                        // sk05_mlp_gateup.py:357:31
	setp.lt.s32 	%p9, %r4, %r20;
	.loc	1 357 54                        // sk05_mlp_gateup.py:357:54
	setp.lt.s32 	%p10, %r226, %r21;
	.loc	1 357 37                        // sk05_mlp_gateup.py:357:37
	and.pred 	%p8, %p9, %p10;
	.loc	1 355 35                        // sk05_mlp_gateup.py:355:35
	mul.lo.s32 	%r274, %r4, %r23;
	.loc	1 355 18                        // sk05_mlp_gateup.py:355:18
	mad.wide.s32 	%rd65, %r274, 2, %rd21;
	.loc	1 355 50                        // sk05_mlp_gateup.py:355:50
	mad.wide.s32 	%rd63, %r226, 2, %rd65;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16.f32 	%rs1, %r266;
	cvt.rn.bf16.f32 	%rs2, %r267;
	cvt.rn.bf16.f32 	%rs3, %r268;
	cvt.rn.bf16.f32 	%rs4, %r269;
	cvt.rn.bf16.f32 	%rs5, %r270;
	cvt.rn.bf16.f32 	%rs6, %r271;
	cvt.rn.bf16.f32 	%rs7, %r272;
	cvt.rn.bf16.f32 	%rs8, %r273;
	bar.sync 	0;
	shl.b32 	%r275, %r2, 5;
	and.b32 	%r276, %r275, 768;
	shl.b32 	%r277, %r228, 1;
	and.b32 	%r278, %r2, 1;
	neg.s32 	%r279, %r278;
	and.b32 	%r280, %r279, 1088;
	bfe.s32 	%r281, %r2, 1, 1;
	and.b32 	%r282, %r281, 2052;
	or.b32 	%r283, %r276, %r277;
	or.b32 	%r284, %r280, %r283;
	xor.b32 	%r285, %r284, %r250;
	or.b32 	%r286, %r285, %r282;
	add.s32 	%r217, %r112, %r286;
	// begin inline asm
	st.shared.v2.b16 [ %r217 + 0 ], { %rs1, %rs2 };
	// end inline asm
	add.s32 	%r218, %r217, 128;
	// begin inline asm
	st.shared.v2.b16 [ %r218 + 0 ], { %rs3, %rs4 };
	// end inline asm
	xor.b32 	%r287, %r286, 4;
	add.s32 	%r219, %r112, %r287;
	// begin inline asm
	st.shared.v2.b16 [ %r219 + 0 ], { %rs5, %rs6 };
	// end inline asm
	add.s32 	%r220, %r219, 128;
	// begin inline asm
	st.shared.v2.b16 [ %r220 + 0 ], { %rs7, %rs8 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r288, %r2, 3;
	and.b32 	%r289, %r288, 768;
	shr.u32 	%r290, %r243, 1;
	and.b32 	%r291, %r2, 128;
	or.b32 	%r292, %r302, %r289;
	xor.b32 	%r293, %r292, %r290;
	or.b32 	%r294, %r293, %r291;
	add.s32 	%r295, %r112, %r294;
	ld.shared.b32 	%r221, [%r295];
	xor.b32 	%r296, %r294, 64;
	add.s32 	%r297, %r112, %r296;
	ld.shared.b32 	%r222, [%r297+1024];
	xor.b32 	%r298, %r294, 4;
	add.s32 	%r299, %r112, %r298;
	ld.shared.b32 	%r223, [%r299+2048];
	xor.b32 	%r300, %r294, 68;
	add.s32 	%r301, %r112, %r300;
	ld.shared.b32 	%r224, [%r301+3072];
	.loc	1 356 8                         // sk05_mlp_gateup.py:356:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd63 + 0 ], { %r221, %r222, %r223, %r224 };
	// end inline asm
	.loc	1 354 4                         // sk05_mlp_gateup.py:354:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/usr/local/lib/python3.12/dist-packages/vllm/_genesis/kernels/sk05_mlp_gateup.py"
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
.b32 201                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0xc2 DW_TAG_compile_unit
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
.b8 53
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 103
.b8 97
.b8 116
.b8 101
.b8 117
.b8 112
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 117
.b8 115
.b8 114
.b8 47
.b8 108
.b8 111
.b8 99
.b8 97
.b8 108
.b8 47
.b8 108
.b8 105
.b8 98
.b8 47
.b8 112
.b8 121
.b8 116
.b8 104
.b8 111
.b8 110
.b8 51
.b8 46
.b8 49
.b8 50
.b8 47
.b8 100
.b8 105
.b8 115
.b8 116
.b8 45
.b8 112
.b8 97
.b8 99
.b8 107
.b8 97
.b8 103
.b8 101
.b8 115
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
.b8 2                                   // Abbrev [2] 0x6a:0x1a DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 53
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 103
.b8 97
.b8 116
.b8 101
.b8 117
.b8 112
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x84:0x48 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 106                                // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x99:0x19 DW_TAG_inlined_subroutine
.b32 106                                // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 62                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0xb2:0x19 DW_TAG_inlined_subroutine
.b32 106                                // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 63                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_0 = _Nativo(
    "sk05_mlp_gateup/tile16x128x128_shift0_abi15",
    _PTX_0, "_sk05_mlp_gateup_kernel",
    warps=8, shared=36864,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 18],
    horneado={11: 1, 12: 1, 15: 1, 17: 1, 19: 1, 20: 16, 21: 128, 22: 128, 23: 8, 24: 128},
    div16=[8, 9, 10, 13, 14, 16],
)

_PTX_1 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk05_mlp_gateup_kernel // -- Begin function _sk05_mlp_gateup_kernel
.extern .shared .align 16 .b8 global_smem[];
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
                                        // @_sk05_mlp_gateup_kernel
.visible .entry _sk05_mlp_gateup_kernel(
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_6,
	.param .u32 _sk05_mlp_gateup_kernel_param_7,
	.param .u32 _sk05_mlp_gateup_kernel_param_8,
	.param .u32 _sk05_mlp_gateup_kernel_param_9,
	.param .u32 _sk05_mlp_gateup_kernel_param_10,
	.param .u32 _sk05_mlp_gateup_kernel_param_11,
	.param .u32 _sk05_mlp_gateup_kernel_param_12,
	.param .u32 _sk05_mlp_gateup_kernel_param_13,
	.param .u32 _sk05_mlp_gateup_kernel_param_14,
	.param .u32 _sk05_mlp_gateup_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_16,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_17
)
.reqntid 256
{
	.reg .pred 	%p<11>;
	.reg .b16 	%rs<25>;
	.reg .b32 	%r<337>;
	.reg .b64 	%rd<75>;
	.loc	1 307 0                         // sk05_mlp_gateup.py:307:0
$L__func_begin0:
	.loc	1 307 0                         // sk05_mlp_gateup.py:307:0

// %bb.0:
	ld.param.b32 	%r25, [_sk05_mlp_gateup_kernel_param_14];
	ld.param.b32 	%r24, [_sk05_mlp_gateup_kernel_param_13];
	ld.param.b32 	%r23, [_sk05_mlp_gateup_kernel_param_12];
	ld.param.b32 	%r22, [_sk05_mlp_gateup_kernel_param_9];
	ld.param.b32 	%r21, [_sk05_mlp_gateup_kernel_param_8];
	ld.param.b32 	%r20, [_sk05_mlp_gateup_kernel_param_7];
	ld.param.b64 	%rd24, [_sk05_mlp_gateup_kernel_param_5];
	ld.param.b64 	%rd23, [_sk05_mlp_gateup_kernel_param_4];
	ld.param.b64 	%rd22, [_sk05_mlp_gateup_kernel_param_3];
	ld.param.b64 	%rd21, [_sk05_mlp_gateup_kernel_param_2];
	ld.param.b64 	%rd20, [_sk05_mlp_gateup_kernel_param_1];
	ld.param.b64 	%rd19, [_sk05_mlp_gateup_kernel_param_0];
$L__tmp0:
	.loc	1 317 24                        // sk05_mlp_gateup.py:317:24
	mov.u32 	%r41, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk05_mlp_gateup.py:318:27 ]
	add.s32 	%r42, %r20, 15;
	.loc	2 43 30                         // standard.py:43:30 @[ sk05_mlp_gateup.py:318:27 ]
	shr.s32 	%r43, %r42, 31;
	shr.u32 	%r44, %r43, 28;
	add.s32 	%r45, %r42, %r44;
	shr.s32 	%r46, %r45, 4;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk05_mlp_gateup.py:319:27 ]
	add.s32 	%r47, %r21, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk05_mlp_gateup.py:319:27 ]
	shr.s32 	%r48, %r47, 31;
	shr.u32 	%r49, %r48, 25;
	add.s32 	%r50, %r47, %r49;
	shr.s32 	%r51, %r50, 7;
$L__tmp3:
	.loc	1 320 29                        // sk05_mlp_gateup.py:320:29
	shl.b32 	%r52, %r51, 3;
	.loc	1 321 22                        // sk05_mlp_gateup.py:321:22
	div.s32 	%r53, %r41, %r52;
	.loc	1 321 38                        // sk05_mlp_gateup.py:321:38
	shl.b32 	%r54, %r53, 3;
	.loc	1 322 30                        // sk05_mlp_gateup.py:322:30
	sub.s32 	%r55, %r46, %r54;
	ld.param.b32 	%r56, [_sk05_mlp_gateup_kernel_param_10];
	.loc	1 322 39                        // sk05_mlp_gateup.py:322:39
	min.s32 	%r57, %r55, 8;
	ld.param.b32 	%r58, [_sk05_mlp_gateup_kernel_param_11];
	.loc	1 323 30                        // sk05_mlp_gateup.py:323:30
	mul.lo.s32 	%r59, %r53, %r52;
	sub.s32 	%r60, %r41, %r59;
	.loc	1 324 36                        // sk05_mlp_gateup.py:324:36
	div.s32 	%r61, %r60, %r57;
	.loc	1 323 46                        // sk05_mlp_gateup.py:323:46
	mul.lo.s32 	%r62, %r61, %r57;
	sub.s32 	%r63, %r60, %r62;
	.loc	1 323 23                        // sk05_mlp_gateup.py:323:23
	add.s32 	%r64, %r63, %r54;
	.loc	1 326 22                        // sk05_mlp_gateup.py:326:22
	shl.b32 	%r1, %r64, 4;
	.loc	1 326 45                        // sk05_mlp_gateup.py:326:45
	mov.u32 	%r2, %tid.x;
	and.b32 	%r3, %r2, 240;
	bfe.u32 	%r65, %r2, 4, 4;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r4, %r1, %r65;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r5, %r4, %r20;
	.loc	1 327 22                        // sk05_mlp_gateup.py:327:22
	shl.b32 	%r6, %r61, 7;
	.loc	1 327 45                        // sk05_mlp_gateup.py:327:45
	shr.u32 	%r66, %r2, 3;
	bfe.u32 	%r67, %r2, 3, 5;
	shl.b32 	%r7, %r2, 1;
	and.b32 	%r68, %r7, 6;
	and.b32 	%r8, %r2, 224;
	shr.u32 	%r69, %r8, 2;
	or.b32 	%r70, %r68, %r69;
	and.b32 	%r9, %r2, 15;
	shl.b32 	%r71, %r9, 3;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r72, %r6, %r67;
	or.b32 	%r73, %r72, 32;
	or.b32 	%r74, %r72, 64;
	or.b32 	%r75, %r66, %r6;
	or.b32 	%r76, %r75, 96;
	or.b32 	%r77, %r6, %r70;
	or.b32 	%r79, %r77, 64;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r81, %r72, %r21;
	rem.s32 	%r82, %r73, %r21;
	rem.s32 	%r83, %r74, %r21;
	rem.s32 	%r84, %r76, %r21;
	rem.s32 	%r10, %r77, %r21;
	rem.s32 	%r11, %r79, %r21;
	.loc	1 330 39                        // sk05_mlp_gateup.py:330:39
	mul.lo.s32 	%r87, %r5, %r56;
	.loc	1 330 21                        // sk05_mlp_gateup.py:330:21
	cvt.s64.s32 	%rd1, %r87;
	add.s64 	%rd36, %rd19, %rd1;
	.loc	1 330 51                        // sk05_mlp_gateup.py:330:51
	cvt.u64.u32 	%rd2, %r71;
	add.s64 	%rd25, %rd36, %rd2;
	.loc	1 331 28                        // sk05_mlp_gateup.py:331:28
	and.b32 	%r12, %r2, 7;
	shl.b32 	%r88, %r12, 4;
	.loc	1 331 21                        // sk05_mlp_gateup.py:331:21
	cvt.u64.u32 	%rd3, %r88;
	add.s64 	%rd37, %rd20, %rd3;
	.loc	1 331 69                        // sk05_mlp_gateup.py:331:69
	mul.lo.s32 	%r89, %r81, %r58;
	mul.lo.s32 	%r90, %r82, %r58;
	mul.lo.s32 	%r91, %r83, %r58;
	mul.lo.s32 	%r92, %r84, %r58;
	.loc	1 331 51                        // sk05_mlp_gateup.py:331:51
	cvt.s64.s32 	%rd4, %r89;
	add.s64 	%rd26, %rd37, %rd4;
	cvt.s64.s32 	%rd5, %r90;
	add.s64 	%rd27, %rd37, %rd5;
	cvt.s64.s32 	%rd6, %r91;
	add.s64 	%rd28, %rd37, %rd6;
	cvt.s64.s32 	%rd7, %r92;
	add.s64 	%rd29, %rd37, %rd7;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	setp.lt.s32 	%p1, %r22, 128;
	setp.gt.s32 	%p2, %r22, 127;
	.loc	1 344 30                        // sk05_mlp_gateup.py:344:30
	and.b32 	%r109, %r2, 255;
	shl.b32 	%r110, %r109, 3;
	and.b32 	%r111, %r2, 112;
	xor.b32 	%r112, %r110, %r111;
	mov.b32 	%r113, global_smem;
	add.s32 	%r13, %r113, %r112;
	add.s32 	%r27, %r13, 32768;
	selp.b32 	%r28, 8, 0, %p2;
	// begin inline asm
	cp.async.ca.shared.global [ %r27 + 0 ], [ %rd25 + 0 ], 0x8, %r28;
	// end inline asm
	cp.async.commit_group;
	.loc	1 344 47                        // sk05_mlp_gateup.py:344:47
	shl.b32 	%r114, %r109, 4;
	and.b32 	%r115, %r7, 112;
	xor.b32 	%r116, %r114, %r115;
	add.s32 	%r29, %r113, %r116;
	selp.b32 	%r30, 16, 0, %p2;
	// begin inline asm
	cp.async.cg.shared.global [ %r29 + 0 ], [ %rd26 + 0 ], 0x10, %r30;
	// end inline asm
	add.s32 	%r31, %r29, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r31 + 0 ], [ %rd27 + 0 ], 0x10, %r30;
	// end inline asm
	add.s32 	%r32, %r29, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r32 + 0 ], [ %rd28 + 0 ], 0x10, %r30;
	// end inline asm
	add.s32 	%r33, %r29, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r33 + 0 ], [ %rd29 + 0 ], 0x10, %r30;
	// end inline asm
	cp.async.commit_group;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	setp.gt.s32 	%p3, %r22, 255;
	.loc	1 345 18                        // sk05_mlp_gateup.py:345:18
	add.s64 	%rd30, %rd25, 128;
	.loc	1 346 18                        // sk05_mlp_gateup.py:346:18
	add.s64 	%rd31, %rd26, 128;
	add.s64 	%rd32, %rd27, 128;
	add.s64 	%rd33, %rd28, 128;
	add.s64 	%rd34, %rd29, 128;
	.loc	1 344 30                        // sk05_mlp_gateup.py:344:30
	bar.sync 	0;
	add.s32 	%r34, %r13, 34816;
	selp.b32 	%r35, 8, 0, %p3;
	// begin inline asm
	cp.async.ca.shared.global [ %r34 + 0 ], [ %rd30 + 0 ], 0x8, %r35;
	// end inline asm
	cp.async.commit_group;
	.loc	1 344 47                        // sk05_mlp_gateup.py:344:47
	add.s32 	%r36, %r29, 16384;
	selp.b32 	%r37, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r36 + 0 ], [ %rd31 + 0 ], 0x10, %r37;
	// end inline asm
	add.s32 	%r38, %r29, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r38 + 0 ], [ %rd32 + 0 ], 0x10, %r37;
	// end inline asm
	add.s32 	%r39, %r29, 24576;
	// begin inline asm
	cp.async.cg.shared.global [ %r39 + 0 ], [ %rd33 + 0 ], 0x10, %r37;
	// end inline asm
	add.s32 	%r40, %r29, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r40 + 0 ], [ %rd34 + 0 ], 0x10, %r37;
	// end inline asm
	cp.async.commit_group;
	mov.b32 	%r329, 0f00000000;
	cvt.u32.u64 	%r325, %rd3;
	mov.b32 	%r330, %r329;
	mov.b32 	%r331, %r329;
	mov.b32 	%r332, %r329;
	mov.b32 	%r333, %r329;
	mov.b32 	%r334, %r329;
	mov.b32 	%r335, %r329;
	mov.b32 	%r336, %r329;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	@%p1 bra 	$L__BB0_3;
// %bb.1:                               // %.lr.ph
	.loc	1 0 23                          // sk05_mlp_gateup.py:0:23
	ld.param.b32 	%r26, [_sk05_mlp_gateup_kernel_param_15];
	ld.param.b64 	%rd35, [_sk05_mlp_gateup_kernel_param_6];
	or.b32 	%r78, %r77, 1;
	or.b32 	%r80, %r77, 65;
	rem.s32 	%r85, %r78, %r21;
	rem.s32 	%r86, %r80, %r21;
	shr.s32 	%r93, %r10, 31;
	shr.u32 	%r94, %r93, 25;
	add.s32 	%r95, %r10, %r94;
	shr.s32 	%r96, %r95, 7;
	shr.s32 	%r97, %r85, 31;
	shr.u32 	%r98, %r97, 25;
	add.s32 	%r99, %r85, %r98;
	shr.s32 	%r100, %r99, 7;
	shr.s32 	%r101, %r11, 31;
	shr.u32 	%r102, %r101, 25;
	add.s32 	%r103, %r11, %r102;
	shr.s32 	%r104, %r103, 7;
	shr.s32 	%r105, %r86, 31;
	shr.u32 	%r106, %r105, 25;
	add.s32 	%r107, %r86, %r106;
	shr.s32 	%r108, %r107, 7;
	mad.wide.s32 	%rd8, %r96, 4, %rd35;
	mad.wide.s32 	%rd9, %r100, 4, %rd35;
	mad.wide.s32 	%rd10, %r104, 4, %rd35;
	mad.wide.s32 	%rd11, %r108, 4, %rd35;
	.loc	1 335 28                        // sk05_mlp_gateup.py:335:28
	shr.u32 	%r117, %r22, 7;
	add.s32 	%r118, %r117, -2;
	shl.b32 	%r119, %r9, 7;
	and.b32 	%r120, %r2, 16;
	xor.b32 	%r121, %r325, %r120;
	or.b32 	%r14, %r121, %r119;
	xor.b32 	%r15, %r14, 32;
	xor.b32 	%r16, %r14, 64;
	xor.b32 	%r17, %r14, 96;
	shl.b32 	%r122, %r12, 7;
	shl.b32 	%r123, %r8, 5;
	and.b32 	%r124, %r7, 48;
	or.b32 	%r125, %r122, %r123;
	xor.b32 	%r126, %r325, %r124;
	or.b32 	%r18, %r125, %r126;
	xor.b32 	%r19, %r18, 64;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	cvt.s64.s32 	%rd12, %r118;
	and.b32 	%r127, %r22, -128;
	cvt.u64.u32 	%rd13, %r127;
	add.s64 	%rd38, %rd3, %rd7;
	add.s64 	%rd39, %rd38, %rd20;
	add.s64 	%rd14, %rd39, 256;
	add.s64 	%rd40, %rd3, %rd6;
	add.s64 	%rd41, %rd40, %rd20;
	add.s64 	%rd15, %rd41, 256;
	add.s64 	%rd42, %rd3, %rd5;
	add.s64 	%rd43, %rd42, %rd20;
	add.s64 	%rd16, %rd43, 256;
	add.s64 	%rd44, %rd3, %rd4;
	add.s64 	%rd45, %rd44, %rd20;
	add.s64 	%rd17, %rd45, 256;
	add.s64 	%rd46, %rd2, %rd1;
	add.s64 	%rd47, %rd46, %rd19;
	add.s64 	%rd18, %rd47, 256;
	mov.b32 	%r329, 0f00000000;
	mov.b32 	%r328, 1;
	mov.b32 	%r327, -1;
	mov.b64 	%rd73, 0;
	mov.b32 	%r132, 0;
	mov.b32 	%r326, %r132;
	mov.b64 	%rd74, %rd73;
	mov.b32 	%r330, %r329;
	mov.b32 	%r331, %r329;
	mov.b32 	%r332, %r329;
	mov.b32 	%r333, %r329;
	mov.b32 	%r334, %r329;
	mov.b32 	%r335, %r329;
	mov.b32 	%r336, %r329;
$L__BB0_2:                              // %__nv_exp2f.exit
                                        // =>This Inner Loop Header: Depth=1
	setp.lt.s64 	%p4, %rd74, %rd12;
	add.s32 	%r180, %r327, 1;
	setp.gt.s32 	%p5, %r180, 1;
	selp.b32 	%r327, 0, %r180, %p5;
	.loc	1 342 39                        // sk05_mlp_gateup.py:342:39
	mul.wide.s32 	%rd57, %r326, 4;
	add.s64 	%rd48, %rd8, %rd57;
	add.s64 	%rd49, %rd9, %rd57;
	add.s64 	%rd50, %rd10, %rd57;
	add.s64 	%rd51, %rd11, %rd57;
	.loc	1 342 29                        // sk05_mlp_gateup.py:342:29
	// begin inline asm
	mov.u32 %r128, 0x0;
	ld.global.b32 { %r128 }, [ %rd48 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r129, 0x0;
	ld.global.b32 { %r129 }, [ %rd49 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r130, 0x0;
	ld.global.b32 { %r130 }, [ %rd50 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r131, 0x0;
	ld.global.b32 { %r131 }, [ %rd51 + 0 ];
	// end inline asm
	.loc	1 342 21                        // sk05_mlp_gateup.py:342:21
	ex2.approx.ftz.f32 	%r181, %r128;
	ex2.approx.ftz.f32 	%r182, %r129;
	ex2.approx.ftz.f32 	%r183, %r130;
	ex2.approx.ftz.f32 	%r184, %r131;
	.loc	1 344 30                        // sk05_mlp_gateup.py:344:30
	cp.async.wait_group 	2;
	bar.sync 	0;
	shl.b32 	%r185, %r327, 11;
	add.s32 	%r186, %r113, %r185;
	add.s32 	%r187, %r186, %r14;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r133, %r134, %r135, %r136}, [%r187+32768];
	add.s32 	%r188, %r186, %r15;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r145, %r146, %r147, %r148}, [%r188+32768];
	add.s32 	%r189, %r186, %r16;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r157, %r158, %r159, %r160}, [%r189+32768];
	add.s32 	%r190, %r186, %r17;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r165, %r166, %r167, %r168}, [%r190+32768];
	.loc	1 344 47                        // sk05_mlp_gateup.py:344:47
	shl.b32 	%r191, %r327, 14;
	add.s32 	%r192, %r113, %r191;
	add.s32 	%r193, %r192, %r18;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r137, %r138, %r149, %r150}, [%r193];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r139, %r140, %r155, %r156}, [%r193+8192];
	add.s32 	%r194, %r192, %r19;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r161, %r162, %r169, %r170}, [%r194];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r163, %r164, %r171, %r172}, [%r194+8192];
	.loc	1 344 39                        // sk05_mlp_gateup.py:344:39
	mov.b32 	%r141, %r132;
	mov.b32 	%r142, %r132;
	mov.b32 	%r143, %r132;
	mov.b32 	%r144, %r132;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r141, %r142, %r143, %r144 }, { %r133, %r134, %r135, %r136 }, { %r137, %r138 }, { %r141, %r142, %r143, %r144 };
	// end inline asm
	mov.b32 	%r151, %r132;
	mov.b32 	%r152, %r132;
	mov.b32 	%r153, %r132;
	mov.b32 	%r154, %r132;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r151, %r152, %r153, %r154 }, { %r133, %r134, %r135, %r136 }, { %r139, %r140 }, { %r151, %r152, %r153, %r154 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r141, %r142, %r143, %r144 }, { %r145, %r146, %r147, %r148 }, { %r149, %r150 }, { %r141, %r142, %r143, %r144 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r151, %r152, %r153, %r154 }, { %r145, %r146, %r147, %r148 }, { %r155, %r156 }, { %r151, %r152, %r153, %r154 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r141, %r142, %r143, %r144 }, { %r157, %r158, %r159, %r160 }, { %r161, %r162 }, { %r141, %r142, %r143, %r144 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r151, %r152, %r153, %r154 }, { %r157, %r158, %r159, %r160 }, { %r163, %r164 }, { %r151, %r152, %r153, %r154 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r141, %r142, %r143, %r144 }, { %r165, %r166, %r167, %r168 }, { %r169, %r170 }, { %r141, %r142, %r143, %r144 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r151, %r152, %r153, %r154 }, { %r165, %r166, %r167, %r168 }, { %r171, %r172 }, { %r151, %r152, %r153, %r154 };
	// end inline asm
	.loc	1 344 81                        // sk05_mlp_gateup.py:344:81
	cvt.rn.f32.s32 	%r195, %r151;
	cvt.rn.f32.s32 	%r196, %r152;
	cvt.rn.f32.s32 	%r197, %r153;
	cvt.rn.f32.s32 	%r198, %r154;
	cvt.rn.f32.s32 	%r199, %r141;
	cvt.rn.f32.s32 	%r200, %r142;
	cvt.rn.f32.s32 	%r201, %r143;
	cvt.rn.f32.s32 	%r202, %r144;
	.loc	1 344 15                        // sk05_mlp_gateup.py:344:15
	fma.rn.f32 	%r332, %r182, %r202, %r332;
	fma.rn.f32 	%r331, %r181, %r201, %r331;
	fma.rn.f32 	%r330, %r182, %r200, %r330;
	fma.rn.f32 	%r329, %r181, %r199, %r329;
	fma.rn.f32 	%r336, %r184, %r198, %r336;
	fma.rn.f32 	%r335, %r183, %r197, %r335;
	fma.rn.f32 	%r334, %r184, %r196, %r334;
	fma.rn.f32 	%r333, %r183, %r195, %r333;
	.loc	1 346 18                        // sk05_mlp_gateup.py:346:18
	add.s64 	%rd52, %rd18, %rd73;
	add.s64 	%rd53, %rd17, %rd73;
	add.s64 	%rd54, %rd16, %rd73;
	add.s64 	%rd55, %rd15, %rd73;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	add.s64 	%rd56, %rd14, %rd73;
	add.s32 	%r203, %r328, 1;
	setp.gt.s32 	%p6, %r203, 1;
	selp.b32 	%r328, 0, %r203, %p6;
	.loc	1 344 30                        // sk05_mlp_gateup.py:344:30
	shl.b32 	%r204, %r328, 11;
	bar.sync 	0;
	add.s32 	%r205, %r13, %r204;
	add.s32 	%r173, %r205, 32768;
	selp.b32 	%r174, 8, 0, %p4;
	// begin inline asm
	cp.async.ca.shared.global [ %r173 + 0 ], [ %rd52 + 0 ], 0x8, %r174;
	// end inline asm
	cp.async.commit_group;
	.loc	1 344 47                        // sk05_mlp_gateup.py:344:47
	shl.b32 	%r206, %r328, 14;
	add.s32 	%r175, %r29, %r206;
	selp.b32 	%r176, 16, 0, %p4;
	// begin inline asm
	cp.async.cg.shared.global [ %r175 + 0 ], [ %rd53 + 0 ], 0x10, %r176;
	// end inline asm
	add.s32 	%r177, %r175, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r177 + 0 ], [ %rd54 + 0 ], 0x10, %r176;
	// end inline asm
	add.s32 	%r178, %r175, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r178 + 0 ], [ %rd55 + 0 ], 0x10, %r176;
	// end inline asm
	add.s32 	%r179, %r175, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r179 + 0 ], [ %rd56 + 0 ], 0x10, %r176;
	// end inline asm
	cp.async.commit_group;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	add.s64 	%rd74, %rd74, 1;
	add.s64 	%rd73, %rd73, 128;
	add.s32 	%r326, %r326, %r26;
	setp.ne.b64 	%p7, %rd13, %rd73;
	@%p7 bra 	$L__BB0_2;
$L__BB0_3:                              // %._crit_edge
	.loc	1 0 23                          // sk05_mlp_gateup.py:0:23
	cvt.u32.u64 	%r226, %rd2;
	.loc	1 327 45                        // sk05_mlp_gateup.py:327:45
	or.b32 	%r227, %r6, %r226;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r228, %r227, 7;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r229, %r228, %r21;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r230, %r227, 6;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r231, %r230, %r21;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r232, %r227, 5;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r233, %r232, %r21;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r234, %r227, 4;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r235, %r234, %r21;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r236, %r227, 3;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r237, %r236, %r21;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r238, %r227, 2;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r239, %r238, %r21;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r240, %r227, 1;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r241, %r240, %r21;
	rem.s32 	%r242, %r227, %r21;
	.loc	1 326 45                        // sk05_mlp_gateup.py:326:45
	and.b32 	%r243, %r2, 28;
	bfe.u32 	%r244, %r2, 2, 3;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r245, %r244, %r1;
	or.b32 	%r246, %r245, 8;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r247, %r246, %r20;
	rem.s32 	%r248, %r245, %r20;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 348 38                        // sk05_mlp_gateup.py:348:38
	mad.wide.s32 	%rd58, %r248, 4, %rd23;
	mad.wide.s32 	%rd59, %r247, 4, %rd23;
	.loc	1 348 24                        // sk05_mlp_gateup.py:348:24
	// begin inline asm
	mov.u32 %r207, 0x0;
	ld.global.b32 { %r207 }, [ %rd58 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r208, 0x0;
	ld.global.b32 { %r208 }, [ %rd59 + 0 ];
	// end inline asm
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r249, %r329, %r207;
	mul.f32 	%r250, %r330, %r207;
	mul.f32 	%r251, %r331, %r208;
	mul.f32 	%r252, %r332, %r208;
	mul.f32 	%r253, %r333, %r207;
	mul.f32 	%r254, %r334, %r207;
	mul.f32 	%r255, %r335, %r208;
	mul.f32 	%r256, %r336, %r208;
	.loc	1 349 38                        // sk05_mlp_gateup.py:349:38
	mad.wide.s32 	%rd60, %r10, 4, %rd24;
	mad.wide.s32 	%rd61, %r11, 4, %rd24;
	.loc	1 349 24                        // sk05_mlp_gateup.py:349:24
	// begin inline asm
	mov.u32 %r209, 0x0;
	mov.u32 %r210, 0x0;
	ld.global.v2.b32 { %r209, %r210 }, [ %rd60 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r211, 0x0;
	mov.u32 %r212, 0x0;
	ld.global.v2.b32 { %r211, %r212 }, [ %rd61 + 0 ];
	// end inline asm
	.loc	1 350 49                        // sk05_mlp_gateup.py:350:49
	mul.lo.s32 	%r257, %r5, %r24;
	.loc	1 350 31                        // sk05_mlp_gateup.py:350:31
	mad.wide.s32 	%rd71, %r257, 2, %rd22;
	.loc	1 350 82                        // sk05_mlp_gateup.py:350:82
	mul.lo.s32 	%r258, %r242, %r25;
	mul.lo.s32 	%r259, %r241, %r25;
	mul.lo.s32 	%r260, %r239, %r25;
	mul.lo.s32 	%r261, %r237, %r25;
	mul.lo.s32 	%r262, %r235, %r25;
	mul.lo.s32 	%r263, %r233, %r25;
	mul.lo.s32 	%r264, %r231, %r25;
	mul.lo.s32 	%r265, %r229, %r25;
	.loc	1 350 64                        // sk05_mlp_gateup.py:350:64
	mad.wide.s32 	%rd62, %r258, 2, %rd71;
	mad.wide.s32 	%rd63, %r259, 2, %rd71;
	mad.wide.s32 	%rd64, %r260, 2, %rd71;
	mad.wide.s32 	%rd65, %r261, 2, %rd71;
	mad.wide.s32 	%rd66, %r262, 2, %rd71;
	mad.wide.s32 	%rd67, %r263, 2, %rd71;
	mad.wide.s32 	%rd68, %r264, 2, %rd71;
	mad.wide.s32 	%rd69, %r265, 2, %rd71;
	.loc	1 350 19                        // sk05_mlp_gateup.py:350:19
	// begin inline asm
	mov.u16 %rs1, 0x0;
	ld.global.b16 { %rs1 }, [ %rd62 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs2, 0x0;
	ld.global.b16 { %rs2 }, [ %rd63 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs3, 0x0;
	ld.global.b16 { %rs3 }, [ %rd64 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs4, 0x0;
	ld.global.b16 { %rs4 }, [ %rd65 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs5, 0x0;
	ld.global.b16 { %rs5 }, [ %rd66 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs6, 0x0;
	ld.global.b16 { %rs6 }, [ %rd67 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs7, 0x0;
	ld.global.b16 { %rs7 }, [ %rd68 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs8, 0x0;
	ld.global.b16 { %rs8 }, [ %rd69 + 0 ];
	// end inline asm
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	and.b32 	%r266, %r2, 120;
	shl.b32 	%r267, %r266, 5;
	or.b32 	%r268, %r267, %r325;
	xor.b32 	%r269, %r268, %r3;
	add.s32 	%r213, %r113, %r269;
	mov.b32 	%r214, {%rs1, %rs2};
	mov.b32 	%r215, {%rs3, %rs4};
	mov.b32 	%r216, {%rs5, %rs6};
	mov.b32 	%r217, {%rs7, %rs8};
	// begin inline asm
	st.shared.v4.b32 [ %r213 + 0 ], { %r214, %r215, %r216, %r217 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r270, %r12, 9;
	shl.b32 	%r271, %r2, 4;
	and.b32 	%r272, %r271, 496;
	shr.u32 	%r273, %r8, 1;
	xor.b32 	%r274, %r272, %r273;
	add.s32 	%r275, %r113, %r270;
	add.s32 	%r276, %r275, %r274;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r277, %r278, %r279, %r280}, [%r276];
	mov.b32 	{%rs17, %rs18}, %r277;
	mov.b32 	{%rs19, %rs20}, %r278;
	mov.b32 	{%rs21, %rs22}, %r279;
	mov.b32 	{%rs23, %rs24}, %r280;
	cvt.f32.bf16 	%r281, %rs17;
	cvt.f32.bf16 	%r282, %rs18;
	cvt.f32.bf16 	%r283, %rs19;
	cvt.f32.bf16 	%r284, %rs20;
	cvt.f32.bf16 	%r285, %rs21;
	cvt.f32.bf16 	%r286, %rs22;
	cvt.f32.bf16 	%r287, %rs23;
	cvt.f32.bf16 	%r288, %rs24;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r289, %r249, %r209, %r281;
	fma.rn.f32 	%r290, %r250, %r210, %r282;
	fma.rn.f32 	%r291, %r251, %r209, %r283;
	fma.rn.f32 	%r292, %r252, %r210, %r284;
	fma.rn.f32 	%r293, %r253, %r211, %r285;
	fma.rn.f32 	%r294, %r254, %r212, %r286;
	fma.rn.f32 	%r295, %r255, %r211, %r287;
	fma.rn.f32 	%r296, %r256, %r212, %r288;
	.loc	1 357 31                        // sk05_mlp_gateup.py:357:31
	setp.lt.s32 	%p9, %r4, %r20;
	.loc	1 357 54                        // sk05_mlp_gateup.py:357:54
	setp.lt.s32 	%p10, %r227, %r21;
	.loc	1 357 37                        // sk05_mlp_gateup.py:357:37
	and.pred 	%p8, %p9, %p10;
	.loc	1 355 35                        // sk05_mlp_gateup.py:355:35
	mul.lo.s32 	%r297, %r4, %r23;
	.loc	1 355 18                        // sk05_mlp_gateup.py:355:18
	mad.wide.s32 	%rd72, %r297, 2, %rd21;
	.loc	1 355 50                        // sk05_mlp_gateup.py:355:50
	mad.wide.s32 	%rd70, %r227, 2, %rd72;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16.f32 	%rs9, %r289;
	cvt.rn.bf16.f32 	%rs10, %r290;
	cvt.rn.bf16.f32 	%rs11, %r291;
	cvt.rn.bf16.f32 	%rs12, %r292;
	cvt.rn.bf16.f32 	%rs13, %r293;
	cvt.rn.bf16.f32 	%rs14, %r294;
	cvt.rn.bf16.f32 	%rs15, %r295;
	cvt.rn.bf16.f32 	%rs16, %r296;
	bar.sync 	0;
	shl.b32 	%r298, %r2, 5;
	and.b32 	%r299, %r298, 768;
	shl.b32 	%r300, %r243, 1;
	and.b32 	%r301, %r2, 1;
	neg.s32 	%r302, %r301;
	and.b32 	%r303, %r302, 1088;
	bfe.s32 	%r304, %r2, 1, 1;
	and.b32 	%r305, %r304, 2052;
	or.b32 	%r306, %r299, %r300;
	or.b32 	%r307, %r303, %r306;
	xor.b32 	%r308, %r307, %r273;
	or.b32 	%r309, %r308, %r305;
	add.s32 	%r218, %r113, %r309;
	// begin inline asm
	st.shared.v2.b16 [ %r218 + 0 ], { %rs9, %rs10 };
	// end inline asm
	add.s32 	%r219, %r218, 128;
	// begin inline asm
	st.shared.v2.b16 [ %r219 + 0 ], { %rs11, %rs12 };
	// end inline asm
	xor.b32 	%r310, %r309, 4;
	add.s32 	%r220, %r113, %r310;
	// begin inline asm
	st.shared.v2.b16 [ %r220 + 0 ], { %rs13, %rs14 };
	// end inline asm
	add.s32 	%r221, %r220, 128;
	// begin inline asm
	st.shared.v2.b16 [ %r221 + 0 ], { %rs15, %rs16 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r311, %r2, 3;
	and.b32 	%r312, %r311, 768;
	shr.u32 	%r313, %r266, 1;
	and.b32 	%r314, %r2, 128;
	or.b32 	%r315, %r325, %r312;
	xor.b32 	%r316, %r315, %r313;
	or.b32 	%r317, %r316, %r314;
	add.s32 	%r318, %r113, %r317;
	ld.shared.b32 	%r222, [%r318];
	xor.b32 	%r319, %r317, 64;
	add.s32 	%r320, %r113, %r319;
	ld.shared.b32 	%r223, [%r320+1024];
	xor.b32 	%r321, %r317, 4;
	add.s32 	%r322, %r113, %r321;
	ld.shared.b32 	%r224, [%r322+2048];
	xor.b32 	%r323, %r317, 68;
	add.s32 	%r324, %r113, %r323;
	ld.shared.b32 	%r225, [%r324+3072];
	.loc	1 356 8                         // sk05_mlp_gateup.py:356:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd70 + 0 ], { %r222, %r223, %r224, %r225 };
	// end inline asm
	.loc	1 354 4                         // sk05_mlp_gateup.py:354:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/usr/local/lib/python3.12/dist-packages/vllm/_genesis/kernels/sk05_mlp_gateup.py"
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
.b32 201                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0xc2 DW_TAG_compile_unit
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
.b8 53
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 103
.b8 97
.b8 116
.b8 101
.b8 117
.b8 112
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 117
.b8 115
.b8 114
.b8 47
.b8 108
.b8 111
.b8 99
.b8 97
.b8 108
.b8 47
.b8 108
.b8 105
.b8 98
.b8 47
.b8 112
.b8 121
.b8 116
.b8 104
.b8 111
.b8 110
.b8 51
.b8 46
.b8 49
.b8 50
.b8 47
.b8 100
.b8 105
.b8 115
.b8 116
.b8 45
.b8 112
.b8 97
.b8 99
.b8 107
.b8 97
.b8 103
.b8 101
.b8 115
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
.b8 2                                   // Abbrev [2] 0x6a:0x1a DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 53
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 103
.b8 97
.b8 116
.b8 101
.b8 117
.b8 112
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x84:0x48 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 106                                // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x99:0x19 DW_TAG_inlined_subroutine
.b32 106                                // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 62                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0xb2:0x19 DW_TAG_inlined_subroutine
.b32 106                                // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 63                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_1 = _Nativo(
    "sk05_mlp_gateup/tile16x128x128_shift0_abi16",
    _PTX_1, "_sk05_mlp_gateup_kernel",
    warps=8, shared=36864,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 17, 18],
    horneado={11: 1, 12: 1, 15: 1, 19: 1, 20: 16, 21: 128, 22: 128, 23: 8, 24: 128},
    div16=[8, 9, 10, 13, 14, 16, 17],
)

_PTX_2 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk05_mlp_gateup_kernel // -- Begin function _sk05_mlp_gateup_kernel
.extern .shared .align 16 .b8 global_smem[];
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
                                        // @_sk05_mlp_gateup_kernel
.visible .entry _sk05_mlp_gateup_kernel(
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_6,
	.param .u32 _sk05_mlp_gateup_kernel_param_7,
	.param .u32 _sk05_mlp_gateup_kernel_param_8,
	.param .u32 _sk05_mlp_gateup_kernel_param_9,
	.param .u32 _sk05_mlp_gateup_kernel_param_10,
	.param .u32 _sk05_mlp_gateup_kernel_param_11,
	.param .u32 _sk05_mlp_gateup_kernel_param_12,
	.param .u32 _sk05_mlp_gateup_kernel_param_13,
	.param .u32 _sk05_mlp_gateup_kernel_param_14,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_16
)
.reqntid 256
{
	.reg .pred 	%p<25>;
	.reg .b16 	%rs<65>;
	.reg .b32 	%r<929>;
	.reg .b64 	%rd<134>;
	.loc	1 307 0                         // sk05_mlp_gateup.py:307:0
$L__func_begin0:
	.loc	1 307 0                         // sk05_mlp_gateup.py:307:0

// %bb.0:
	ld.param.b32 	%r29, [_sk05_mlp_gateup_kernel_param_13];
	ld.param.b32 	%r28, [_sk05_mlp_gateup_kernel_param_12];
	ld.param.b32 	%r27, [_sk05_mlp_gateup_kernel_param_9];
	ld.param.b32 	%r26, [_sk05_mlp_gateup_kernel_param_8];
	ld.param.b32 	%r25, [_sk05_mlp_gateup_kernel_param_7];
	ld.param.b64 	%rd33, [_sk05_mlp_gateup_kernel_param_5];
	ld.param.b64 	%rd32, [_sk05_mlp_gateup_kernel_param_4];
	ld.param.b64 	%rd31, [_sk05_mlp_gateup_kernel_param_3];
	ld.param.b64 	%rd30, [_sk05_mlp_gateup_kernel_param_2];
	ld.param.b64 	%rd29, [_sk05_mlp_gateup_kernel_param_1];
	ld.param.b64 	%rd28, [_sk05_mlp_gateup_kernel_param_0];
$L__tmp0:
	.loc	1 317 24                        // sk05_mlp_gateup.py:317:24
	mov.u32 	%r49, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk05_mlp_gateup.py:318:27 ]
	add.s32 	%r50, %r25, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk05_mlp_gateup.py:318:27 ]
	shr.s32 	%r51, %r50, 31;
	shr.u32 	%r52, %r51, 25;
	add.s32 	%r53, %r50, %r52;
	shr.s32 	%r54, %r53, 7;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk05_mlp_gateup.py:319:27 ]
	add.s32 	%r55, %r26, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk05_mlp_gateup.py:319:27 ]
	shr.s32 	%r56, %r55, 31;
	shr.u32 	%r57, %r56, 25;
	add.s32 	%r58, %r55, %r57;
	shr.s32 	%r59, %r58, 7;
$L__tmp3:
	.loc	1 320 29                        // sk05_mlp_gateup.py:320:29
	shl.b32 	%r60, %r59, 3;
	.loc	1 321 22                        // sk05_mlp_gateup.py:321:22
	div.s32 	%r61, %r49, %r60;
	.loc	1 321 38                        // sk05_mlp_gateup.py:321:38
	shl.b32 	%r62, %r61, 3;
	.loc	1 322 30                        // sk05_mlp_gateup.py:322:30
	sub.s32 	%r63, %r54, %r62;
	ld.param.b32 	%r64, [_sk05_mlp_gateup_kernel_param_10];
	.loc	1 322 39                        // sk05_mlp_gateup.py:322:39
	min.s32 	%r65, %r63, 8;
	ld.param.b32 	%r66, [_sk05_mlp_gateup_kernel_param_11];
	.loc	1 323 30                        // sk05_mlp_gateup.py:323:30
	mul.lo.s32 	%r67, %r61, %r60;
	sub.s32 	%r68, %r49, %r67;
	.loc	1 324 36                        // sk05_mlp_gateup.py:324:36
	div.s32 	%r69, %r68, %r65;
	.loc	1 323 46                        // sk05_mlp_gateup.py:323:46
	mul.lo.s32 	%r70, %r69, %r65;
	sub.s32 	%r71, %r68, %r70;
	.loc	1 323 23                        // sk05_mlp_gateup.py:323:23
	add.s32 	%r72, %r71, %r62;
	.loc	1 326 22                        // sk05_mlp_gateup.py:326:22
	shl.b32 	%r1, %r72, 7;
	.loc	1 326 45                        // sk05_mlp_gateup.py:326:45
	mov.u32 	%r2, %tid.x;
	and.b32 	%r3, %r2, 248;
	bfe.u32 	%r73, %r2, 3, 5;
	or.b32 	%r74, %r73, 32;
	or.b32 	%r75, %r73, 64;
	or.b32 	%r76, %r73, 96;
	and.b32 	%r4, %r2, 3;
	shl.b32 	%r77, %r4, 1;
	and.b32 	%r5, %r2, 96;
	shr.u32 	%r78, %r5, 2;
	or.b32 	%r79, %r77, %r78;
	and.b32 	%r6, %r2, 7;
	shl.b32 	%r80, %r6, 4;
	and.b32 	%r7, %r2, 15;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r81, %r1, %r73;
	or.b32 	%r82, %r1, %r74;
	or.b32 	%r83, %r1, %r75;
	or.b32 	%r84, %r1, %r76;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r8, %r81, %r25;
	rem.s32 	%r9, %r82, %r25;
	rem.s32 	%r10, %r83, %r25;
	rem.s32 	%r11, %r84, %r25;
	.loc	1 327 22                        // sk05_mlp_gateup.py:327:22
	shl.b32 	%r12, %r69, 7;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r85, %r12, %r73;
	or.b32 	%r86, %r12, %r74;
	or.b32 	%r87, %r12, %r75;
	or.b32 	%r88, %r12, %r76;
	or.b32 	%r89, %r12, %r79;
	or.b32 	%r91, %r89, 32;
	or.b32 	%r93, %r89, 64;
	or.b32 	%r95, %r89, 96;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r97, %r85, %r26;
	rem.s32 	%r98, %r86, %r26;
	rem.s32 	%r99, %r87, %r26;
	rem.s32 	%r100, %r88, %r26;
	rem.s32 	%r13, %r89, %r26;
	rem.s32 	%r14, %r91, %r26;
	rem.s32 	%r15, %r93, %r26;
	rem.s32 	%r16, %r95, %r26;
	.loc	1 330 39                        // sk05_mlp_gateup.py:330:39
	mul.lo.s32 	%r105, %r8, %r64;
	mul.lo.s32 	%r106, %r9, %r64;
	mul.lo.s32 	%r107, %r10, %r64;
	mul.lo.s32 	%r108, %r11, %r64;
	.loc	1 330 21                        // sk05_mlp_gateup.py:330:21
	cvt.s64.s32 	%rd1, %r105;
	add.s64 	%rd51, %rd28, %rd1;
	cvt.s64.s32 	%rd2, %r106;
	add.s64 	%rd52, %rd28, %rd2;
	cvt.s64.s32 	%rd3, %r107;
	add.s64 	%rd53, %rd28, %rd3;
	cvt.s64.s32 	%rd4, %r108;
	add.s64 	%rd54, %rd28, %rd4;
	.loc	1 330 51                        // sk05_mlp_gateup.py:330:51
	cvt.u64.u32 	%rd5, %r80;
	add.s64 	%rd34, %rd51, %rd5;
	add.s64 	%rd35, %rd52, %rd5;
	add.s64 	%rd36, %rd53, %rd5;
	add.s64 	%rd37, %rd54, %rd5;
	.loc	1 331 21                        // sk05_mlp_gateup.py:331:21
	add.s64 	%rd55, %rd29, %rd5;
	.loc	1 331 69                        // sk05_mlp_gateup.py:331:69
	mul.lo.s32 	%r109, %r97, %r66;
	mul.lo.s32 	%r110, %r98, %r66;
	mul.lo.s32 	%r111, %r99, %r66;
	mul.lo.s32 	%r112, %r100, %r66;
	.loc	1 331 51                        // sk05_mlp_gateup.py:331:51
	cvt.s64.s32 	%rd6, %r109;
	add.s64 	%rd38, %rd55, %rd6;
	cvt.s64.s32 	%rd7, %r110;
	add.s64 	%rd39, %rd55, %rd7;
	cvt.s64.s32 	%rd8, %r111;
	add.s64 	%rd40, %rd55, %rd8;
	cvt.s64.s32 	%rd9, %r112;
	add.s64 	%rd41, %rd55, %rd9;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	setp.gt.s32 	%p1, %r27, 127;
	.loc	1 344 30                        // sk05_mlp_gateup.py:344:30
	shl.b32 	%r17, %r2, 4;
	and.b32 	%r145, %r17, 4080;
	and.b32 	%r18, %r2, 56;
	shl.b32 	%r146, %r18, 1;
	xor.b32 	%r147, %r145, %r146;
	mov.b32 	%r148, global_smem;
	add.s32 	%r31, %r148, %r147;
	selp.b32 	%r32, 16, 0, %p1;
	// begin inline asm
	cp.async.cg.shared.global [ %r31 + 0 ], [ %rd34 + 0 ], 0x10, %r32;
	// end inline asm
	add.s32 	%r33, %r31, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r33 + 0 ], [ %rd35 + 0 ], 0x10, %r32;
	// end inline asm
	add.s32 	%r34, %r31, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r34 + 0 ], [ %rd36 + 0 ], 0x10, %r32;
	// end inline asm
	add.s32 	%r35, %r31, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r35 + 0 ], [ %rd37 + 0 ], 0x10, %r32;
	// end inline asm
	cp.async.commit_group;
	.loc	1 344 47                        // sk05_mlp_gateup.py:344:47
	add.s32 	%r36, %r31, 32768;
	// begin inline asm
	cp.async.cg.shared.global [ %r36 + 0 ], [ %rd38 + 0 ], 0x10, %r32;
	// end inline asm
	add.s32 	%r37, %r31, 36864;
	// begin inline asm
	cp.async.cg.shared.global [ %r37 + 0 ], [ %rd39 + 0 ], 0x10, %r32;
	// end inline asm
	add.s32 	%r38, %r31, 40960;
	// begin inline asm
	cp.async.cg.shared.global [ %r38 + 0 ], [ %rd40 + 0 ], 0x10, %r32;
	// end inline asm
	add.s32 	%r39, %r31, 45056;
	// begin inline asm
	cp.async.cg.shared.global [ %r39 + 0 ], [ %rd41 + 0 ], 0x10, %r32;
	// end inline asm
	cp.async.commit_group;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	setp.gt.s32 	%p2, %r27, 255;
	.loc	1 345 18                        // sk05_mlp_gateup.py:345:18
	add.s64 	%rd42, %rd34, 128;
	add.s64 	%rd43, %rd35, 128;
	add.s64 	%rd44, %rd36, 128;
	add.s64 	%rd45, %rd37, 128;
	.loc	1 346 18                        // sk05_mlp_gateup.py:346:18
	add.s64 	%rd46, %rd38, 128;
	add.s64 	%rd47, %rd39, 128;
	add.s64 	%rd48, %rd40, 128;
	add.s64 	%rd49, %rd41, 128;
	.loc	1 344 30                        // sk05_mlp_gateup.py:344:30
	bar.sync 	0;
	add.s32 	%r40, %r31, 16384;
	selp.b32 	%r41, 16, 0, %p2;
	// begin inline asm
	cp.async.cg.shared.global [ %r40 + 0 ], [ %rd42 + 0 ], 0x10, %r41;
	// end inline asm
	add.s32 	%r42, %r31, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r42 + 0 ], [ %rd43 + 0 ], 0x10, %r41;
	// end inline asm
	add.s32 	%r43, %r31, 24576;
	// begin inline asm
	cp.async.cg.shared.global [ %r43 + 0 ], [ %rd44 + 0 ], 0x10, %r41;
	// end inline asm
	add.s32 	%r44, %r31, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r44 + 0 ], [ %rd45 + 0 ], 0x10, %r41;
	// end inline asm
	cp.async.commit_group;
	.loc	1 344 47                        // sk05_mlp_gateup.py:344:47
	add.s32 	%r45, %r31, 49152;
	// begin inline asm
	cp.async.cg.shared.global [ %r45 + 0 ], [ %rd46 + 0 ], 0x10, %r41;
	// end inline asm
	add.s32 	%r46, %r31, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r46 + 0 ], [ %rd47 + 0 ], 0x10, %r41;
	// end inline asm
	add.s32 	%r47, %r31, 57344;
	// begin inline asm
	cp.async.cg.shared.global [ %r47 + 0 ], [ %rd48 + 0 ], 0x10, %r41;
	// end inline asm
	add.s32 	%r48, %r31, 61440;
	// begin inline asm
	cp.async.cg.shared.global [ %r48 + 0 ], [ %rd49 + 0 ], 0x10, %r41;
	// end inline asm
	cp.async.commit_group;
	cvt.u32.u64 	%r859, %rd5;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	@%p1 bra 	$L__BB0_2;
	bra.uni 	$L__BB0_1;
$L__BB0_2:                              // %.lr.ph
	.loc	1 0 23                          // sk05_mlp_gateup.py:0:23
	ld.param.b32 	%r30, [_sk05_mlp_gateup_kernel_param_14];
	ld.param.b64 	%rd50, [_sk05_mlp_gateup_kernel_param_6];
	or.b32 	%r90, %r89, 1;
	or.b32 	%r92, %r89, 33;
	or.b32 	%r94, %r89, 65;
	or.b32 	%r96, %r89, 97;
	rem.s32 	%r101, %r90, %r26;
	rem.s32 	%r102, %r92, %r26;
	rem.s32 	%r103, %r94, %r26;
	rem.s32 	%r104, %r96, %r26;
	shr.s32 	%r113, %r13, 31;
	shr.u32 	%r114, %r113, 25;
	add.s32 	%r115, %r13, %r114;
	shr.s32 	%r116, %r115, 7;
	shr.s32 	%r117, %r101, 31;
	shr.u32 	%r118, %r117, 25;
	add.s32 	%r119, %r101, %r118;
	shr.s32 	%r120, %r119, 7;
	shr.s32 	%r121, %r14, 31;
	shr.u32 	%r122, %r121, 25;
	add.s32 	%r123, %r14, %r122;
	shr.s32 	%r124, %r123, 7;
	shr.s32 	%r125, %r102, 31;
	shr.u32 	%r126, %r125, 25;
	add.s32 	%r127, %r102, %r126;
	shr.s32 	%r128, %r127, 7;
	shr.s32 	%r129, %r15, 31;
	shr.u32 	%r130, %r129, 25;
	add.s32 	%r131, %r15, %r130;
	shr.s32 	%r132, %r131, 7;
	shr.s32 	%r133, %r103, 31;
	shr.u32 	%r134, %r133, 25;
	add.s32 	%r135, %r103, %r134;
	shr.s32 	%r136, %r135, 7;
	shr.s32 	%r137, %r16, 31;
	shr.u32 	%r138, %r137, 25;
	add.s32 	%r139, %r16, %r138;
	shr.s32 	%r140, %r139, 7;
	shr.s32 	%r141, %r104, 31;
	shr.u32 	%r142, %r141, 25;
	add.s32 	%r143, %r104, %r142;
	shr.s32 	%r144, %r143, 7;
	mad.wide.s32 	%rd10, %r116, 4, %rd50;
	mad.wide.s32 	%rd11, %r120, 4, %rd50;
	mad.wide.s32 	%rd12, %r124, 4, %rd50;
	mad.wide.s32 	%rd13, %r128, 4, %rd50;
	mad.wide.s32 	%rd14, %r132, 4, %rd50;
	mad.wide.s32 	%rd15, %r136, 4, %rd50;
	mad.wide.s32 	%rd16, %r140, 4, %rd50;
	mad.wide.s32 	%rd17, %r144, 4, %rd50;
	.loc	1 335 28                        // sk05_mlp_gateup.py:335:28
	shr.u32 	%r149, %r27, 7;
	add.s32 	%r150, %r149, -2;
	shl.b32 	%r151, %r7, 7;
	and.b32 	%r152, %r17, 2160;
	and.b32 	%r863, %r2, 16;
	or.b32 	%r153, %r151, %r152;
	xor.b32 	%r19, %r153, %r863;
	xor.b32 	%r20, %r19, 32;
	xor.b32 	%r21, %r19, 64;
	xor.b32 	%r22, %r19, 96;
	shl.b32 	%r154, %r6, 7;
	shl.b32 	%r155, %r5, 5;
	shl.b32 	%r864, %r2, 1;
	and.b32 	%r156, %r864, 48;
	or.b32 	%r157, %r154, %r155;
	xor.b32 	%r158, %r859, %r156;
	or.b32 	%r23, %r157, %r158;
	xor.b32 	%r24, %r23, 64;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	cvt.s64.s32 	%rd18, %r150;
	and.b32 	%r159, %r27, -128;
	cvt.u64.u32 	%rd19, %r159;
	add.s64 	%rd56, %rd5, %rd9;
	add.s64 	%rd57, %rd56, %rd29;
	add.s64 	%rd20, %rd57, 256;
	add.s64 	%rd58, %rd5, %rd8;
	add.s64 	%rd59, %rd58, %rd29;
	add.s64 	%rd21, %rd59, 256;
	add.s64 	%rd60, %rd5, %rd7;
	add.s64 	%rd61, %rd60, %rd29;
	add.s64 	%rd22, %rd61, 256;
	add.s64 	%rd62, %rd5, %rd6;
	add.s64 	%rd63, %rd62, %rd29;
	add.s64 	%rd23, %rd63, 256;
	add.s64 	%rd64, %rd5, %rd4;
	add.s64 	%rd65, %rd64, %rd28;
	add.s64 	%rd24, %rd65, 256;
	add.s64 	%rd66, %rd5, %rd3;
	add.s64 	%rd67, %rd66, %rd28;
	add.s64 	%rd25, %rd67, 256;
	add.s64 	%rd68, %rd5, %rd2;
	add.s64 	%rd69, %rd68, %rd28;
	add.s64 	%rd26, %rd69, 256;
	add.s64 	%rd70, %rd5, %rd1;
	add.s64 	%rd71, %rd70, %rd28;
	add.s64 	%rd27, %rd71, 256;
	mov.b32 	%r865, 0f00000000;
	mov.b32 	%r862, 1;
	mov.b32 	%r861, -1;
	mov.b64 	%rd132, 0;
	mov.b32 	%r168, 0;
	mov.b32 	%r860, %r168;
	mov.b64 	%rd133, %rd132;
	mov.b32 	%r866, %r865;
	mov.b32 	%r867, %r865;
	mov.b32 	%r868, %r865;
	mov.b32 	%r869, %r865;
	mov.b32 	%r870, %r865;
	mov.b32 	%r871, %r865;
	mov.b32 	%r872, %r865;
	mov.b32 	%r873, %r865;
	mov.b32 	%r874, %r865;
	mov.b32 	%r875, %r865;
	mov.b32 	%r876, %r865;
	mov.b32 	%r877, %r865;
	mov.b32 	%r878, %r865;
	mov.b32 	%r879, %r865;
	mov.b32 	%r880, %r865;
	mov.b32 	%r881, %r865;
	mov.b32 	%r882, %r865;
	mov.b32 	%r883, %r865;
	mov.b32 	%r884, %r865;
	mov.b32 	%r885, %r865;
	mov.b32 	%r886, %r865;
	mov.b32 	%r887, %r865;
	mov.b32 	%r888, %r865;
	mov.b32 	%r889, %r865;
	mov.b32 	%r890, %r865;
	mov.b32 	%r891, %r865;
	mov.b32 	%r892, %r865;
	mov.b32 	%r893, %r865;
	mov.b32 	%r894, %r865;
	mov.b32 	%r895, %r865;
	mov.b32 	%r896, %r865;
	mov.b32 	%r897, %r865;
	mov.b32 	%r898, %r865;
	mov.b32 	%r899, %r865;
	mov.b32 	%r900, %r865;
	mov.b32 	%r901, %r865;
	mov.b32 	%r902, %r865;
	mov.b32 	%r903, %r865;
	mov.b32 	%r904, %r865;
	mov.b32 	%r905, %r865;
	mov.b32 	%r906, %r865;
	mov.b32 	%r907, %r865;
	mov.b32 	%r908, %r865;
	mov.b32 	%r909, %r865;
	mov.b32 	%r910, %r865;
	mov.b32 	%r911, %r865;
	mov.b32 	%r912, %r865;
	mov.b32 	%r913, %r865;
	mov.b32 	%r914, %r865;
	mov.b32 	%r915, %r865;
	mov.b32 	%r916, %r865;
	mov.b32 	%r917, %r865;
	mov.b32 	%r918, %r865;
	mov.b32 	%r919, %r865;
	mov.b32 	%r920, %r865;
	mov.b32 	%r921, %r865;
	mov.b32 	%r922, %r865;
	mov.b32 	%r923, %r865;
	mov.b32 	%r924, %r865;
	mov.b32 	%r925, %r865;
	mov.b32 	%r926, %r865;
	mov.b32 	%r927, %r865;
	mov.b32 	%r928, %r865;
$L__BB0_3:                              // %__nv_exp2f.exit
                                        // =>This Inner Loop Header: Depth=1
	setp.lt.s64 	%p3, %rd133, %rd18;
	add.s32 	%r338, %r861, 1;
	setp.gt.s32 	%p4, %r338, 1;
	selp.b32 	%r861, 0, %r338, %p4;
	.loc	1 342 39                        // sk05_mlp_gateup.py:342:39
	mul.wide.s32 	%rd88, %r860, 4;
	add.s64 	%rd72, %rd10, %rd88;
	add.s64 	%rd73, %rd11, %rd88;
	add.s64 	%rd74, %rd12, %rd88;
	add.s64 	%rd75, %rd13, %rd88;
	add.s64 	%rd76, %rd14, %rd88;
	add.s64 	%rd77, %rd15, %rd88;
	add.s64 	%rd78, %rd16, %rd88;
	add.s64 	%rd79, %rd17, %rd88;
	.loc	1 342 29                        // sk05_mlp_gateup.py:342:29
	// begin inline asm
	mov.u32 %r160, 0x0;
	ld.global.b32 { %r160 }, [ %rd72 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r161, 0x0;
	ld.global.b32 { %r161 }, [ %rd73 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r162, 0x0;
	ld.global.b32 { %r162 }, [ %rd74 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r163, 0x0;
	ld.global.b32 { %r163 }, [ %rd75 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r164, 0x0;
	ld.global.b32 { %r164 }, [ %rd76 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r165, 0x0;
	ld.global.b32 { %r165 }, [ %rd77 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r166, 0x0;
	ld.global.b32 { %r166 }, [ %rd78 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r167, 0x0;
	ld.global.b32 { %r167 }, [ %rd79 + 0 ];
	// end inline asm
	.loc	1 342 21                        // sk05_mlp_gateup.py:342:21
	ex2.approx.ftz.f32 	%r339, %r160;
	ex2.approx.ftz.f32 	%r340, %r161;
	ex2.approx.ftz.f32 	%r341, %r162;
	ex2.approx.ftz.f32 	%r342, %r163;
	ex2.approx.ftz.f32 	%r343, %r164;
	ex2.approx.ftz.f32 	%r344, %r165;
	ex2.approx.ftz.f32 	%r345, %r166;
	ex2.approx.ftz.f32 	%r346, %r167;
	.loc	1 344 30                        // sk05_mlp_gateup.py:344:30
	cp.async.wait_group 	2;
	bar.sync 	0;
	shl.b32 	%r347, %r861, 14;
	add.s32 	%r348, %r148, %r347;
	add.s32 	%r349, %r348, %r19;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r169, %r170, %r171, %r172}, [%r349];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r181, %r182, %r183, %r184}, [%r349+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r185, %r186, %r187, %r188}, [%r349+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r189, %r190, %r191, %r192}, [%r349+12288];
	add.s32 	%r350, %r348, %r20;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r197, %r198, %r199, %r200}, [%r350];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r225, %r226, %r227, %r228}, [%r350+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r245, %r246, %r247, %r248}, [%r350+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r265, %r266, %r267, %r268}, [%r350+12288];
	add.s32 	%r351, %r348, %r21;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r281, %r282, %r283, %r284}, [%r351];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r293, %r294, %r295, %r296}, [%r351+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r297, %r298, %r299, %r300}, [%r351+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r301, %r302, %r303, %r304}, [%r351+12288];
	add.s32 	%r352, %r348, %r22;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r305, %r306, %r307, %r308}, [%r352];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r317, %r318, %r319, %r320}, [%r352+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r321, %r322, %r323, %r324}, [%r352+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r325, %r326, %r327, %r328}, [%r352+12288];
	.loc	1 344 47                        // sk05_mlp_gateup.py:344:47
	add.s32 	%r353, %r348, %r23;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r173, %r174, %r201, %r202}, [%r353+32768];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r175, %r176, %r207, %r208}, [%r353+36864];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r177, %r178, %r213, %r214}, [%r353+40960];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r179, %r180, %r219, %r220}, [%r353+45056];
	add.s32 	%r354, %r348, %r24;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r285, %r286, %r309, %r310}, [%r354+32768];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r287, %r288, %r311, %r312}, [%r354+36864];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r289, %r290, %r313, %r314}, [%r354+40960];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r291, %r292, %r315, %r316}, [%r354+45056];
	.loc	1 344 39                        // sk05_mlp_gateup.py:344:39
	mov.b32 	%r193, %r168;
	mov.b32 	%r194, %r168;
	mov.b32 	%r195, %r168;
	mov.b32 	%r196, %r168;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r193, %r194, %r195, %r196 }, { %r169, %r170, %r171, %r172 }, { %r173, %r174 }, { %r193, %r194, %r195, %r196 };
	// end inline asm
	mov.b32 	%r203, %r168;
	mov.b32 	%r204, %r168;
	mov.b32 	%r205, %r168;
	mov.b32 	%r206, %r168;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r203, %r204, %r205, %r206 }, { %r169, %r170, %r171, %r172 }, { %r175, %r176 }, { %r203, %r204, %r205, %r206 };
	// end inline asm
	mov.b32 	%r209, %r168;
	mov.b32 	%r210, %r168;
	mov.b32 	%r211, %r168;
	mov.b32 	%r212, %r168;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r209, %r210, %r211, %r212 }, { %r169, %r170, %r171, %r172 }, { %r177, %r178 }, { %r209, %r210, %r211, %r212 };
	// end inline asm
	mov.b32 	%r215, %r168;
	mov.b32 	%r216, %r168;
	mov.b32 	%r217, %r168;
	mov.b32 	%r218, %r168;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r215, %r216, %r217, %r218 }, { %r169, %r170, %r171, %r172 }, { %r179, %r180 }, { %r215, %r216, %r217, %r218 };
	// end inline asm
	mov.b32 	%r221, %r168;
	mov.b32 	%r222, %r168;
	mov.b32 	%r223, %r168;
	mov.b32 	%r224, %r168;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r221, %r222, %r223, %r224 }, { %r181, %r182, %r183, %r184 }, { %r173, %r174 }, { %r221, %r222, %r223, %r224 };
	// end inline asm
	mov.b32 	%r229, %r168;
	mov.b32 	%r230, %r168;
	mov.b32 	%r231, %r168;
	mov.b32 	%r232, %r168;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r229, %r230, %r231, %r232 }, { %r181, %r182, %r183, %r184 }, { %r175, %r176 }, { %r229, %r230, %r231, %r232 };
	// end inline asm
	mov.b32 	%r233, %r168;
	mov.b32 	%r234, %r168;
	mov.b32 	%r235, %r168;
	mov.b32 	%r236, %r168;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r233, %r234, %r235, %r236 }, { %r181, %r182, %r183, %r184 }, { %r177, %r178 }, { %r233, %r234, %r235, %r236 };
	// end inline asm
	mov.b32 	%r237, %r168;
	mov.b32 	%r238, %r168;
	mov.b32 	%r239, %r168;
	mov.b32 	%r240, %r168;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r237, %r238, %r239, %r240 }, { %r181, %r182, %r183, %r184 }, { %r179, %r180 }, { %r237, %r238, %r239, %r240 };
	// end inline asm
	mov.b32 	%r241, %r168;
	mov.b32 	%r242, %r168;
	mov.b32 	%r243, %r168;
	mov.b32 	%r244, %r168;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r241, %r242, %r243, %r244 }, { %r185, %r186, %r187, %r188 }, { %r173, %r174 }, { %r241, %r242, %r243, %r244 };
	// end inline asm
	mov.b32 	%r249, %r168;
	mov.b32 	%r250, %r168;
	mov.b32 	%r251, %r168;
	mov.b32 	%r252, %r168;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r249, %r250, %r251, %r252 }, { %r185, %r186, %r187, %r188 }, { %r175, %r176 }, { %r249, %r250, %r251, %r252 };
	// end inline asm
	mov.b32 	%r253, %r168;
	mov.b32 	%r254, %r168;
	mov.b32 	%r255, %r168;
	mov.b32 	%r256, %r168;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r253, %r254, %r255, %r256 }, { %r185, %r186, %r187, %r188 }, { %r177, %r178 }, { %r253, %r254, %r255, %r256 };
	// end inline asm
	mov.b32 	%r257, %r168;
	mov.b32 	%r258, %r168;
	mov.b32 	%r259, %r168;
	mov.b32 	%r260, %r168;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r257, %r258, %r259, %r260 }, { %r185, %r186, %r187, %r188 }, { %r179, %r180 }, { %r257, %r258, %r259, %r260 };
	// end inline asm
	mov.b32 	%r261, %r168;
	mov.b32 	%r262, %r168;
	mov.b32 	%r263, %r168;
	mov.b32 	%r264, %r168;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r261, %r262, %r263, %r264 }, { %r189, %r190, %r191, %r192 }, { %r173, %r174 }, { %r261, %r262, %r263, %r264 };
	// end inline asm
	mov.b32 	%r269, %r168;
	mov.b32 	%r270, %r168;
	mov.b32 	%r271, %r168;
	mov.b32 	%r272, %r168;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r269, %r270, %r271, %r272 }, { %r189, %r190, %r191, %r192 }, { %r175, %r176 }, { %r269, %r270, %r271, %r272 };
	// end inline asm
	mov.b32 	%r273, %r168;
	mov.b32 	%r274, %r168;
	mov.b32 	%r275, %r168;
	mov.b32 	%r276, %r168;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r273, %r274, %r275, %r276 }, { %r189, %r190, %r191, %r192 }, { %r177, %r178 }, { %r273, %r274, %r275, %r276 };
	// end inline asm
	mov.b32 	%r277, %r168;
	mov.b32 	%r278, %r168;
	mov.b32 	%r279, %r168;
	mov.b32 	%r280, %r168;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r277, %r278, %r279, %r280 }, { %r189, %r190, %r191, %r192 }, { %r179, %r180 }, { %r277, %r278, %r279, %r280 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r193, %r194, %r195, %r196 }, { %r197, %r198, %r199, %r200 }, { %r201, %r202 }, { %r193, %r194, %r195, %r196 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r203, %r204, %r205, %r206 }, { %r197, %r198, %r199, %r200 }, { %r207, %r208 }, { %r203, %r204, %r205, %r206 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r209, %r210, %r211, %r212 }, { %r197, %r198, %r199, %r200 }, { %r213, %r214 }, { %r209, %r210, %r211, %r212 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r215, %r216, %r217, %r218 }, { %r197, %r198, %r199, %r200 }, { %r219, %r220 }, { %r215, %r216, %r217, %r218 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r221, %r222, %r223, %r224 }, { %r225, %r226, %r227, %r228 }, { %r201, %r202 }, { %r221, %r222, %r223, %r224 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r229, %r230, %r231, %r232 }, { %r225, %r226, %r227, %r228 }, { %r207, %r208 }, { %r229, %r230, %r231, %r232 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r233, %r234, %r235, %r236 }, { %r225, %r226, %r227, %r228 }, { %r213, %r214 }, { %r233, %r234, %r235, %r236 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r237, %r238, %r239, %r240 }, { %r225, %r226, %r227, %r228 }, { %r219, %r220 }, { %r237, %r238, %r239, %r240 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r241, %r242, %r243, %r244 }, { %r245, %r246, %r247, %r248 }, { %r201, %r202 }, { %r241, %r242, %r243, %r244 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r249, %r250, %r251, %r252 }, { %r245, %r246, %r247, %r248 }, { %r207, %r208 }, { %r249, %r250, %r251, %r252 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r253, %r254, %r255, %r256 }, { %r245, %r246, %r247, %r248 }, { %r213, %r214 }, { %r253, %r254, %r255, %r256 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r257, %r258, %r259, %r260 }, { %r245, %r246, %r247, %r248 }, { %r219, %r220 }, { %r257, %r258, %r259, %r260 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r261, %r262, %r263, %r264 }, { %r265, %r266, %r267, %r268 }, { %r201, %r202 }, { %r261, %r262, %r263, %r264 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r269, %r270, %r271, %r272 }, { %r265, %r266, %r267, %r268 }, { %r207, %r208 }, { %r269, %r270, %r271, %r272 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r273, %r274, %r275, %r276 }, { %r265, %r266, %r267, %r268 }, { %r213, %r214 }, { %r273, %r274, %r275, %r276 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r277, %r278, %r279, %r280 }, { %r265, %r266, %r267, %r268 }, { %r219, %r220 }, { %r277, %r278, %r279, %r280 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r193, %r194, %r195, %r196 }, { %r281, %r282, %r283, %r284 }, { %r285, %r286 }, { %r193, %r194, %r195, %r196 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r203, %r204, %r205, %r206 }, { %r281, %r282, %r283, %r284 }, { %r287, %r288 }, { %r203, %r204, %r205, %r206 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r209, %r210, %r211, %r212 }, { %r281, %r282, %r283, %r284 }, { %r289, %r290 }, { %r209, %r210, %r211, %r212 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r215, %r216, %r217, %r218 }, { %r281, %r282, %r283, %r284 }, { %r291, %r292 }, { %r215, %r216, %r217, %r218 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r221, %r222, %r223, %r224 }, { %r293, %r294, %r295, %r296 }, { %r285, %r286 }, { %r221, %r222, %r223, %r224 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r229, %r230, %r231, %r232 }, { %r293, %r294, %r295, %r296 }, { %r287, %r288 }, { %r229, %r230, %r231, %r232 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r233, %r234, %r235, %r236 }, { %r293, %r294, %r295, %r296 }, { %r289, %r290 }, { %r233, %r234, %r235, %r236 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r237, %r238, %r239, %r240 }, { %r293, %r294, %r295, %r296 }, { %r291, %r292 }, { %r237, %r238, %r239, %r240 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r241, %r242, %r243, %r244 }, { %r297, %r298, %r299, %r300 }, { %r285, %r286 }, { %r241, %r242, %r243, %r244 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r249, %r250, %r251, %r252 }, { %r297, %r298, %r299, %r300 }, { %r287, %r288 }, { %r249, %r250, %r251, %r252 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r253, %r254, %r255, %r256 }, { %r297, %r298, %r299, %r300 }, { %r289, %r290 }, { %r253, %r254, %r255, %r256 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r257, %r258, %r259, %r260 }, { %r297, %r298, %r299, %r300 }, { %r291, %r292 }, { %r257, %r258, %r259, %r260 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r261, %r262, %r263, %r264 }, { %r301, %r302, %r303, %r304 }, { %r285, %r286 }, { %r261, %r262, %r263, %r264 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r269, %r270, %r271, %r272 }, { %r301, %r302, %r303, %r304 }, { %r287, %r288 }, { %r269, %r270, %r271, %r272 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r273, %r274, %r275, %r276 }, { %r301, %r302, %r303, %r304 }, { %r289, %r290 }, { %r273, %r274, %r275, %r276 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r277, %r278, %r279, %r280 }, { %r301, %r302, %r303, %r304 }, { %r291, %r292 }, { %r277, %r278, %r279, %r280 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r193, %r194, %r195, %r196 }, { %r305, %r306, %r307, %r308 }, { %r309, %r310 }, { %r193, %r194, %r195, %r196 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r203, %r204, %r205, %r206 }, { %r305, %r306, %r307, %r308 }, { %r311, %r312 }, { %r203, %r204, %r205, %r206 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r209, %r210, %r211, %r212 }, { %r305, %r306, %r307, %r308 }, { %r313, %r314 }, { %r209, %r210, %r211, %r212 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r215, %r216, %r217, %r218 }, { %r305, %r306, %r307, %r308 }, { %r315, %r316 }, { %r215, %r216, %r217, %r218 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r221, %r222, %r223, %r224 }, { %r317, %r318, %r319, %r320 }, { %r309, %r310 }, { %r221, %r222, %r223, %r224 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r229, %r230, %r231, %r232 }, { %r317, %r318, %r319, %r320 }, { %r311, %r312 }, { %r229, %r230, %r231, %r232 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r233, %r234, %r235, %r236 }, { %r317, %r318, %r319, %r320 }, { %r313, %r314 }, { %r233, %r234, %r235, %r236 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r237, %r238, %r239, %r240 }, { %r317, %r318, %r319, %r320 }, { %r315, %r316 }, { %r237, %r238, %r239, %r240 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r241, %r242, %r243, %r244 }, { %r321, %r322, %r323, %r324 }, { %r309, %r310 }, { %r241, %r242, %r243, %r244 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r249, %r250, %r251, %r252 }, { %r321, %r322, %r323, %r324 }, { %r311, %r312 }, { %r249, %r250, %r251, %r252 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r253, %r254, %r255, %r256 }, { %r321, %r322, %r323, %r324 }, { %r313, %r314 }, { %r253, %r254, %r255, %r256 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r257, %r258, %r259, %r260 }, { %r321, %r322, %r323, %r324 }, { %r315, %r316 }, { %r257, %r258, %r259, %r260 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r261, %r262, %r263, %r264 }, { %r325, %r326, %r327, %r328 }, { %r309, %r310 }, { %r261, %r262, %r263, %r264 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r269, %r270, %r271, %r272 }, { %r325, %r326, %r327, %r328 }, { %r311, %r312 }, { %r269, %r270, %r271, %r272 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r273, %r274, %r275, %r276 }, { %r325, %r326, %r327, %r328 }, { %r313, %r314 }, { %r273, %r274, %r275, %r276 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r277, %r278, %r279, %r280 }, { %r325, %r326, %r327, %r328 }, { %r315, %r316 }, { %r277, %r278, %r279, %r280 };
	// end inline asm
	.loc	1 344 81                        // sk05_mlp_gateup.py:344:81
	cvt.rn.f32.s32 	%r355, %r277;
	cvt.rn.f32.s32 	%r356, %r278;
	cvt.rn.f32.s32 	%r357, %r279;
	cvt.rn.f32.s32 	%r358, %r280;
	cvt.rn.f32.s32 	%r359, %r273;
	cvt.rn.f32.s32 	%r360, %r274;
	cvt.rn.f32.s32 	%r361, %r275;
	cvt.rn.f32.s32 	%r362, %r276;
	cvt.rn.f32.s32 	%r363, %r269;
	cvt.rn.f32.s32 	%r364, %r270;
	cvt.rn.f32.s32 	%r365, %r271;
	cvt.rn.f32.s32 	%r366, %r272;
	cvt.rn.f32.s32 	%r367, %r261;
	cvt.rn.f32.s32 	%r368, %r262;
	cvt.rn.f32.s32 	%r369, %r263;
	cvt.rn.f32.s32 	%r370, %r264;
	cvt.rn.f32.s32 	%r371, %r257;
	cvt.rn.f32.s32 	%r372, %r258;
	cvt.rn.f32.s32 	%r373, %r259;
	cvt.rn.f32.s32 	%r374, %r260;
	cvt.rn.f32.s32 	%r375, %r253;
	cvt.rn.f32.s32 	%r376, %r254;
	cvt.rn.f32.s32 	%r377, %r255;
	cvt.rn.f32.s32 	%r378, %r256;
	cvt.rn.f32.s32 	%r379, %r249;
	cvt.rn.f32.s32 	%r380, %r250;
	cvt.rn.f32.s32 	%r381, %r251;
	cvt.rn.f32.s32 	%r382, %r252;
	cvt.rn.f32.s32 	%r383, %r241;
	cvt.rn.f32.s32 	%r384, %r242;
	cvt.rn.f32.s32 	%r385, %r243;
	cvt.rn.f32.s32 	%r386, %r244;
	cvt.rn.f32.s32 	%r387, %r237;
	cvt.rn.f32.s32 	%r388, %r238;
	cvt.rn.f32.s32 	%r389, %r239;
	cvt.rn.f32.s32 	%r390, %r240;
	cvt.rn.f32.s32 	%r391, %r233;
	cvt.rn.f32.s32 	%r392, %r234;
	cvt.rn.f32.s32 	%r393, %r235;
	cvt.rn.f32.s32 	%r394, %r236;
	cvt.rn.f32.s32 	%r395, %r229;
	cvt.rn.f32.s32 	%r396, %r230;
	cvt.rn.f32.s32 	%r397, %r231;
	cvt.rn.f32.s32 	%r398, %r232;
	cvt.rn.f32.s32 	%r399, %r221;
	cvt.rn.f32.s32 	%r400, %r222;
	cvt.rn.f32.s32 	%r401, %r223;
	cvt.rn.f32.s32 	%r402, %r224;
	cvt.rn.f32.s32 	%r403, %r215;
	cvt.rn.f32.s32 	%r404, %r216;
	cvt.rn.f32.s32 	%r405, %r217;
	cvt.rn.f32.s32 	%r406, %r218;
	cvt.rn.f32.s32 	%r407, %r209;
	cvt.rn.f32.s32 	%r408, %r210;
	cvt.rn.f32.s32 	%r409, %r211;
	cvt.rn.f32.s32 	%r410, %r212;
	cvt.rn.f32.s32 	%r411, %r203;
	cvt.rn.f32.s32 	%r412, %r204;
	cvt.rn.f32.s32 	%r413, %r205;
	cvt.rn.f32.s32 	%r414, %r206;
	cvt.rn.f32.s32 	%r415, %r193;
	cvt.rn.f32.s32 	%r416, %r194;
	cvt.rn.f32.s32 	%r417, %r195;
	cvt.rn.f32.s32 	%r418, %r196;
	.loc	1 344 15                        // sk05_mlp_gateup.py:344:15
	fma.rn.f32 	%r868, %r340, %r418, %r868;
	fma.rn.f32 	%r867, %r339, %r417, %r867;
	fma.rn.f32 	%r866, %r340, %r416, %r866;
	fma.rn.f32 	%r865, %r339, %r415, %r865;
	fma.rn.f32 	%r872, %r342, %r414, %r872;
	fma.rn.f32 	%r871, %r341, %r413, %r871;
	fma.rn.f32 	%r870, %r342, %r412, %r870;
	fma.rn.f32 	%r869, %r341, %r411, %r869;
	fma.rn.f32 	%r876, %r344, %r410, %r876;
	fma.rn.f32 	%r875, %r343, %r409, %r875;
	fma.rn.f32 	%r874, %r344, %r408, %r874;
	fma.rn.f32 	%r873, %r343, %r407, %r873;
	fma.rn.f32 	%r880, %r346, %r406, %r880;
	fma.rn.f32 	%r879, %r345, %r405, %r879;
	fma.rn.f32 	%r878, %r346, %r404, %r878;
	fma.rn.f32 	%r877, %r345, %r403, %r877;
	fma.rn.f32 	%r884, %r340, %r402, %r884;
	fma.rn.f32 	%r883, %r339, %r401, %r883;
	fma.rn.f32 	%r882, %r340, %r400, %r882;
	fma.rn.f32 	%r881, %r339, %r399, %r881;
	fma.rn.f32 	%r888, %r342, %r398, %r888;
	fma.rn.f32 	%r887, %r341, %r397, %r887;
	fma.rn.f32 	%r886, %r342, %r396, %r886;
	fma.rn.f32 	%r885, %r341, %r395, %r885;
	fma.rn.f32 	%r892, %r344, %r394, %r892;
	fma.rn.f32 	%r891, %r343, %r393, %r891;
	fma.rn.f32 	%r890, %r344, %r392, %r890;
	fma.rn.f32 	%r889, %r343, %r391, %r889;
	fma.rn.f32 	%r896, %r346, %r390, %r896;
	fma.rn.f32 	%r895, %r345, %r389, %r895;
	fma.rn.f32 	%r894, %r346, %r388, %r894;
	fma.rn.f32 	%r893, %r345, %r387, %r893;
	fma.rn.f32 	%r900, %r340, %r386, %r900;
	fma.rn.f32 	%r899, %r339, %r385, %r899;
	fma.rn.f32 	%r898, %r340, %r384, %r898;
	fma.rn.f32 	%r897, %r339, %r383, %r897;
	fma.rn.f32 	%r904, %r342, %r382, %r904;
	fma.rn.f32 	%r903, %r341, %r381, %r903;
	fma.rn.f32 	%r902, %r342, %r380, %r902;
	fma.rn.f32 	%r901, %r341, %r379, %r901;
	fma.rn.f32 	%r908, %r344, %r378, %r908;
	fma.rn.f32 	%r907, %r343, %r377, %r907;
	fma.rn.f32 	%r906, %r344, %r376, %r906;
	fma.rn.f32 	%r905, %r343, %r375, %r905;
	fma.rn.f32 	%r912, %r346, %r374, %r912;
	fma.rn.f32 	%r911, %r345, %r373, %r911;
	fma.rn.f32 	%r910, %r346, %r372, %r910;
	fma.rn.f32 	%r909, %r345, %r371, %r909;
	fma.rn.f32 	%r916, %r340, %r370, %r916;
	fma.rn.f32 	%r915, %r339, %r369, %r915;
	fma.rn.f32 	%r914, %r340, %r368, %r914;
	fma.rn.f32 	%r913, %r339, %r367, %r913;
	fma.rn.f32 	%r920, %r342, %r366, %r920;
	fma.rn.f32 	%r919, %r341, %r365, %r919;
	fma.rn.f32 	%r918, %r342, %r364, %r918;
	fma.rn.f32 	%r917, %r341, %r363, %r917;
	fma.rn.f32 	%r924, %r344, %r362, %r924;
	fma.rn.f32 	%r923, %r343, %r361, %r923;
	fma.rn.f32 	%r922, %r344, %r360, %r922;
	fma.rn.f32 	%r921, %r343, %r359, %r921;
	fma.rn.f32 	%r928, %r346, %r358, %r928;
	fma.rn.f32 	%r927, %r345, %r357, %r927;
	fma.rn.f32 	%r926, %r346, %r356, %r926;
	fma.rn.f32 	%r925, %r345, %r355, %r925;
	.loc	1 345 18                        // sk05_mlp_gateup.py:345:18
	add.s64 	%rd80, %rd27, %rd132;
	add.s64 	%rd81, %rd26, %rd132;
	add.s64 	%rd82, %rd25, %rd132;
	.loc	1 346 18                        // sk05_mlp_gateup.py:346:18
	add.s64 	%rd83, %rd24, %rd132;
	add.s64 	%rd84, %rd23, %rd132;
	add.s64 	%rd85, %rd22, %rd132;
	add.s64 	%rd86, %rd21, %rd132;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	add.s64 	%rd87, %rd20, %rd132;
	add.s32 	%r419, %r862, 1;
	setp.gt.s32 	%p5, %r419, 1;
	selp.b32 	%r862, 0, %r419, %p5;
	.loc	1 344 30                        // sk05_mlp_gateup.py:344:30
	shl.b32 	%r420, %r862, 14;
	bar.sync 	0;
	add.s32 	%r329, %r31, %r420;
	selp.b32 	%r330, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r329 + 0 ], [ %rd80 + 0 ], 0x10, %r330;
	// end inline asm
	add.s32 	%r331, %r329, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r331 + 0 ], [ %rd81 + 0 ], 0x10, %r330;
	// end inline asm
	add.s32 	%r332, %r329, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r332 + 0 ], [ %rd82 + 0 ], 0x10, %r330;
	// end inline asm
	add.s32 	%r333, %r329, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r333 + 0 ], [ %rd83 + 0 ], 0x10, %r330;
	// end inline asm
	cp.async.commit_group;
	.loc	1 344 47                        // sk05_mlp_gateup.py:344:47
	add.s32 	%r334, %r329, 32768;
	// begin inline asm
	cp.async.cg.shared.global [ %r334 + 0 ], [ %rd84 + 0 ], 0x10, %r330;
	// end inline asm
	add.s32 	%r335, %r329, 36864;
	// begin inline asm
	cp.async.cg.shared.global [ %r335 + 0 ], [ %rd85 + 0 ], 0x10, %r330;
	// end inline asm
	add.s32 	%r336, %r329, 40960;
	// begin inline asm
	cp.async.cg.shared.global [ %r336 + 0 ], [ %rd86 + 0 ], 0x10, %r330;
	// end inline asm
	add.s32 	%r337, %r329, 45056;
	// begin inline asm
	cp.async.cg.shared.global [ %r337 + 0 ], [ %rd87 + 0 ], 0x10, %r330;
	// end inline asm
	cp.async.commit_group;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	add.s64 	%rd133, %rd133, 1;
	add.s64 	%rd132, %rd132, 128;
	add.s32 	%r860, %r860, %r30;
	setp.ne.b64 	%p6, %rd19, %rd132;
	@%p6 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_4;
$L__BB0_1:                              // %.._crit_edge_crit_edge
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	shl.b32 	%r864, %r2, 1;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	and.b32 	%r863, %r2, 16;
	mov.b32 	%r865, 0f00000000;
	mov.b32 	%r866, %r865;
	mov.b32 	%r867, %r865;
	mov.b32 	%r868, %r865;
	mov.b32 	%r869, %r865;
	mov.b32 	%r870, %r865;
	mov.b32 	%r871, %r865;
	mov.b32 	%r872, %r865;
	mov.b32 	%r873, %r865;
	mov.b32 	%r874, %r865;
	mov.b32 	%r875, %r865;
	mov.b32 	%r876, %r865;
	mov.b32 	%r877, %r865;
	mov.b32 	%r878, %r865;
	mov.b32 	%r879, %r865;
	mov.b32 	%r880, %r865;
	mov.b32 	%r881, %r865;
	mov.b32 	%r882, %r865;
	mov.b32 	%r883, %r865;
	mov.b32 	%r884, %r865;
	mov.b32 	%r885, %r865;
	mov.b32 	%r886, %r865;
	mov.b32 	%r887, %r865;
	mov.b32 	%r888, %r865;
	mov.b32 	%r889, %r865;
	mov.b32 	%r890, %r865;
	mov.b32 	%r891, %r865;
	mov.b32 	%r892, %r865;
	mov.b32 	%r893, %r865;
	mov.b32 	%r894, %r865;
	mov.b32 	%r895, %r865;
	mov.b32 	%r896, %r865;
	mov.b32 	%r897, %r865;
	mov.b32 	%r898, %r865;
	mov.b32 	%r899, %r865;
	mov.b32 	%r900, %r865;
	mov.b32 	%r901, %r865;
	mov.b32 	%r902, %r865;
	mov.b32 	%r903, %r865;
	mov.b32 	%r904, %r865;
	mov.b32 	%r905, %r865;
	mov.b32 	%r906, %r865;
	mov.b32 	%r907, %r865;
	mov.b32 	%r908, %r865;
	mov.b32 	%r909, %r865;
	mov.b32 	%r910, %r865;
	mov.b32 	%r911, %r865;
	mov.b32 	%r912, %r865;
	mov.b32 	%r913, %r865;
	mov.b32 	%r914, %r865;
	mov.b32 	%r915, %r865;
	mov.b32 	%r916, %r865;
	mov.b32 	%r917, %r865;
	mov.b32 	%r918, %r865;
	mov.b32 	%r919, %r865;
	mov.b32 	%r920, %r865;
	mov.b32 	%r921, %r865;
	mov.b32 	%r922, %r865;
	mov.b32 	%r923, %r865;
	mov.b32 	%r924, %r865;
	mov.b32 	%r925, %r865;
	mov.b32 	%r926, %r865;
	mov.b32 	%r927, %r865;
	mov.b32 	%r928, %r865;
$L__BB0_4:                              // %._crit_edge
	.loc	1 326 45                        // sk05_mlp_gateup.py:326:45
	or.b32 	%r543, %r12, %r859;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r544, %r543, 8;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r545, %r544, %r26;
	rem.s32 	%r546, %r543, %r26;
	.loc	1 326 45                        // sk05_mlp_gateup.py:326:45
	shl.b32 	%r547, %r7, 3;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r548, %r12, %r547;
	.loc	1 326 45                        // sk05_mlp_gateup.py:326:45
	and.b32 	%r549, %r2, 128;
	shr.u32 	%r550, %r549, 3;
	shr.u32 	%r551, %r2, 2;
	bfe.u32 	%r552, %r2, 2, 3;
	or.b32 	%r553, %r550, %r552;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r554, %r553, %r1;
	or.b32 	%r555, %r554, 104;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r556, %r555, %r25;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r557, %r554, 96;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r558, %r557, %r25;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r559, %r554, 72;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r560, %r559, %r25;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r561, %r554, 64;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r562, %r561, %r25;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r563, %r554, 40;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r564, %r563, %r25;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r565, %r554, 32;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r566, %r565, %r25;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r567, %r554, 8;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r568, %r567, %r25;
	rem.s32 	%r569, %r554, %r25;
	.loc	1 326 45                        // sk05_mlp_gateup.py:326:45
	shr.u32 	%r570, %r2, 4;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r571, %r570, %r1;
	or.b32 	%r572, %r571, 112;
	.loc	1 326 45                        // sk05_mlp_gateup.py:326:45
	bfe.u32 	%r573, %r2, 4, 4;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r574, %r573, %r1;
	or.b32 	%r575, %r574, 96;
	or.b32 	%r576, %r574, 80;
	or.b32 	%r577, %r574, 64;
	or.b32 	%r578, %r571, 48;
	or.b32 	%r579, %r574, 32;
	or.b32 	%r580, %r574, 16;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 348 38                        // sk05_mlp_gateup.py:348:38
	mad.wide.s32 	%rd89, %r569, 4, %rd32;
	mad.wide.s32 	%rd90, %r568, 4, %rd32;
	mad.wide.s32 	%rd91, %r566, 4, %rd32;
	mad.wide.s32 	%rd92, %r564, 4, %rd32;
	mad.wide.s32 	%rd93, %r562, 4, %rd32;
	mad.wide.s32 	%rd94, %r560, 4, %rd32;
	mad.wide.s32 	%rd95, %r558, 4, %rd32;
	mad.wide.s32 	%rd96, %r556, 4, %rd32;
	.loc	1 348 24                        // sk05_mlp_gateup.py:348:24
	// begin inline asm
	mov.u32 %r421, 0x0;
	ld.global.b32 { %r421 }, [ %rd89 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r422, 0x0;
	ld.global.b32 { %r422 }, [ %rd90 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r423, 0x0;
	ld.global.b32 { %r423 }, [ %rd91 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r424, 0x0;
	ld.global.b32 { %r424 }, [ %rd92 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r425, 0x0;
	ld.global.b32 { %r425 }, [ %rd93 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r426, 0x0;
	ld.global.b32 { %r426 }, [ %rd94 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r427, 0x0;
	ld.global.b32 { %r427 }, [ %rd95 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r428, 0x0;
	ld.global.b32 { %r428 }, [ %rd96 + 0 ];
	// end inline asm
	.loc	1 349 38                        // sk05_mlp_gateup.py:349:38
	mad.wide.s32 	%rd97, %r13, 4, %rd33;
	mad.wide.s32 	%rd98, %r14, 4, %rd33;
	mad.wide.s32 	%rd99, %r15, 4, %rd33;
	mad.wide.s32 	%rd100, %r16, 4, %rd33;
	.loc	1 349 24                        // sk05_mlp_gateup.py:349:24
	// begin inline asm
	mov.u32 %r429, 0x0;
	mov.u32 %r430, 0x0;
	ld.global.v2.b32 { %r429, %r430 }, [ %rd97 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r431, 0x0;
	mov.u32 %r432, 0x0;
	ld.global.v2.b32 { %r431, %r432 }, [ %rd98 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r433, 0x0;
	mov.u32 %r434, 0x0;
	ld.global.v2.b32 { %r433, %r434 }, [ %rd99 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r435, 0x0;
	mov.u32 %r436, 0x0;
	ld.global.v2.b32 { %r435, %r436 }, [ %rd100 + 0 ];
	// end inline asm
	.loc	1 350 49                        // sk05_mlp_gateup.py:350:49
	mul.lo.s32 	%r581, %r8, %r29;
	mul.lo.s32 	%r582, %r9, %r29;
	mul.lo.s32 	%r583, %r10, %r29;
	mul.lo.s32 	%r584, %r11, %r29;
	.loc	1 350 31                        // sk05_mlp_gateup.py:350:31
	mad.wide.s32 	%rd117, %r581, 2, %rd31;
	mad.wide.s32 	%rd118, %r582, 2, %rd31;
	mad.wide.s32 	%rd119, %r583, 2, %rd31;
	mad.wide.s32 	%rd120, %r584, 2, %rd31;
	.loc	1 350 64                        // sk05_mlp_gateup.py:350:64
	mul.wide.s32 	%rd121, %r546, 2;
	add.s64 	%rd101, %rd117, %rd121;
	mul.wide.s32 	%rd122, %r545, 2;
	add.s64 	%rd102, %rd117, %rd122;
	add.s64 	%rd103, %rd118, %rd121;
	add.s64 	%rd104, %rd118, %rd122;
	add.s64 	%rd105, %rd119, %rd121;
	add.s64 	%rd106, %rd119, %rd122;
	add.s64 	%rd107, %rd120, %rd121;
	add.s64 	%rd108, %rd120, %rd122;
	.loc	1 350 19                        // sk05_mlp_gateup.py:350:19
	// begin inline asm
	mov.u32 %r438, 0x0;
	mov.u32 %r439, 0x0;
	mov.u32 %r440, 0x0;
	mov.u32 %r441, 0x0;
	ld.global.v4.b32 { %r438, %r439, %r440, %r441 }, [ %rd101 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r443, 0x0;
	mov.u32 %r444, 0x0;
	mov.u32 %r445, 0x0;
	mov.u32 %r446, 0x0;
	ld.global.v4.b32 { %r443, %r444, %r445, %r446 }, [ %rd102 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r447, 0x0;
	mov.u32 %r448, 0x0;
	mov.u32 %r449, 0x0;
	mov.u32 %r450, 0x0;
	ld.global.v4.b32 { %r447, %r448, %r449, %r450 }, [ %rd103 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r451, 0x0;
	mov.u32 %r452, 0x0;
	mov.u32 %r453, 0x0;
	mov.u32 %r454, 0x0;
	ld.global.v4.b32 { %r451, %r452, %r453, %r454 }, [ %rd104 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r455, 0x0;
	mov.u32 %r456, 0x0;
	mov.u32 %r457, 0x0;
	mov.u32 %r458, 0x0;
	ld.global.v4.b32 { %r455, %r456, %r457, %r458 }, [ %rd105 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r459, 0x0;
	mov.u32 %r460, 0x0;
	mov.u32 %r461, 0x0;
	mov.u32 %r462, 0x0;
	ld.global.v4.b32 { %r459, %r460, %r461, %r462 }, [ %rd106 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r463, 0x0;
	mov.u32 %r464, 0x0;
	mov.u32 %r465, 0x0;
	mov.u32 %r466, 0x0;
	ld.global.v4.b32 { %r463, %r464, %r465, %r466 }, [ %rd107 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r467, 0x0;
	mov.u32 %r468, 0x0;
	mov.u32 %r469, 0x0;
	mov.u32 %r470, 0x0;
	ld.global.v4.b32 { %r467, %r468, %r469, %r470 }, [ %rd108 + 0 ];
	// end inline asm
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	shl.b32 	%r585, %r18, 7;
	shl.b32 	%r586, %r3, 1;
	or.b32 	%r587, %r585, %r859;
	xor.b32 	%r588, %r587, %r586;
	add.s32 	%r437, %r148, %r588;
	// begin inline asm
	st.shared.v4.b32 [ %r437 + 0 ], { %r438, %r439, %r440, %r441 };
	// end inline asm
	add.s32 	%r442, %r437, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r442 + 0 ], { %r443, %r444, %r445, %r446 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r589, %r6, 10;
	and.b32 	%r590, %r17, 752;
	and.b32 	%r591, %r864, 288;
	and.b32 	%r592, %r551, 16;
	xor.b32 	%r593, %r590, %r591;
	xor.b32 	%r594, %r593, %r592;
	or.b32 	%r595, %r594, %r589;
	add.s32 	%r596, %r148, %r595;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r597, %r598, %r599, %r600}, [%r596];
	xor.b32 	%r601, %r595, 64;
	add.s32 	%r602, %r148, %r601;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r603, %r604, %r605, %r606}, [%r602];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r437 + 0 ], { %r447, %r448, %r449, %r450 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r442 + 0 ], { %r451, %r452, %r453, %r454 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r607, %r608, %r609, %r610}, [%r596];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r611, %r612, %r613, %r614}, [%r602];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r437 + 0 ], { %r455, %r456, %r457, %r458 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r442 + 0 ], { %r459, %r460, %r461, %r462 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r615, %r616, %r617, %r618}, [%r596];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r619, %r620, %r621, %r622}, [%r602];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r437 + 0 ], { %r463, %r464, %r465, %r466 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r442 + 0 ], { %r467, %r468, %r469, %r470 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r623, %r624, %r625, %r626}, [%r596];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r627, %r628, %r629, %r630}, [%r602];
	.loc	1 357 31                        // sk05_mlp_gateup.py:357:31
	setp.lt.s32 	%p15, %r574, %r25;
	setp.lt.s32 	%p16, %r580, %r25;
	setp.lt.s32 	%p17, %r579, %r25;
	setp.lt.s32 	%p18, %r578, %r25;
	setp.lt.s32 	%p19, %r577, %r25;
	setp.lt.s32 	%p20, %r576, %r25;
	setp.lt.s32 	%p21, %r575, %r25;
	setp.lt.s32 	%p22, %r572, %r25;
	.loc	1 357 54                        // sk05_mlp_gateup.py:357:54
	setp.lt.s32 	%p23, %r548, %r26;
	.loc	1 357 37                        // sk05_mlp_gateup.py:357:37
	and.pred 	%p7, %p15, %p23;
	and.pred 	%p8, %p16, %p23;
	and.pred 	%p9, %p17, %p23;
	and.pred 	%p10, %p18, %p23;
	and.pred 	%p11, %p19, %p23;
	and.pred 	%p12, %p20, %p23;
	and.pred 	%p13, %p21, %p23;
	and.pred 	%p14, %p22, %p23;
	.loc	1 355 35                        // sk05_mlp_gateup.py:355:35
	mul.lo.s32 	%r631, %r574, %r28;
	mul.lo.s32 	%r632, %r580, %r28;
	mul.lo.s32 	%r633, %r579, %r28;
	mul.lo.s32 	%r634, %r578, %r28;
	mul.lo.s32 	%r635, %r577, %r28;
	mul.lo.s32 	%r636, %r576, %r28;
	mul.lo.s32 	%r637, %r575, %r28;
	mul.lo.s32 	%r638, %r572, %r28;
	.loc	1 355 18                        // sk05_mlp_gateup.py:355:18
	mad.wide.s32 	%rd123, %r631, 2, %rd30;
	mad.wide.s32 	%rd124, %r632, 2, %rd30;
	mad.wide.s32 	%rd125, %r633, 2, %rd30;
	mad.wide.s32 	%rd126, %r634, 2, %rd30;
	mad.wide.s32 	%rd127, %r635, 2, %rd30;
	mad.wide.s32 	%rd128, %r636, 2, %rd30;
	mad.wide.s32 	%rd129, %r637, 2, %rd30;
	mad.wide.s32 	%rd130, %r638, 2, %rd30;
	.loc	1 355 50                        // sk05_mlp_gateup.py:355:50
	mul.wide.s32 	%rd131, %r548, 2;
	add.s64 	%rd109, %rd123, %rd131;
	add.s64 	%rd110, %rd124, %rd131;
	add.s64 	%rd111, %rd125, %rd131;
	add.s64 	%rd112, %rd126, %rd131;
	add.s64 	%rd113, %rd127, %rd131;
	add.s64 	%rd114, %rd128, %rd131;
	add.s64 	%rd115, %rd129, %rd131;
	add.s64 	%rd116, %rd130, %rd131;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r639, %r914, %r427;
	mul.f32 	%r640, %r913, %r427;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs1, %rs2}, %r623;
	cvt.f32.bf16 	%r641, %rs2;
	cvt.f32.bf16 	%r642, %rs1;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r643, %r640, %r429, %r642;
	fma.rn.f32 	%r644, %r639, %r430, %r641;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r645, %r866, %r421;
	mul.f32 	%r646, %r865, %r421;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs3, %rs4}, %r597;
	cvt.f32.bf16 	%r647, %rs4;
	cvt.f32.bf16 	%r648, %rs3;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r649, %r646, %r429, %r648;
	fma.rn.f32 	%r650, %r645, %r430, %r647;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r472, %r650, %r649;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r651, %r868, %r422;
	mul.f32 	%r652, %r867, %r422;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs5, %rs6}, %r598;
	cvt.f32.bf16 	%r653, %rs6;
	cvt.f32.bf16 	%r654, %rs5;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r655, %r652, %r429, %r654;
	fma.rn.f32 	%r656, %r651, %r430, %r653;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r477, %r656, %r655;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r657, %r882, %r423;
	mul.f32 	%r658, %r881, %r423;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs7, %rs8}, %r607;
	cvt.f32.bf16 	%r659, %rs8;
	cvt.f32.bf16 	%r660, %rs7;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r661, %r658, %r429, %r660;
	fma.rn.f32 	%r662, %r657, %r430, %r659;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r473, %r662, %r661;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r663, %r884, %r424;
	mul.f32 	%r664, %r883, %r424;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs9, %rs10}, %r608;
	cvt.f32.bf16 	%r665, %rs10;
	cvt.f32.bf16 	%r666, %rs9;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r667, %r664, %r429, %r666;
	fma.rn.f32 	%r668, %r663, %r430, %r665;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r478, %r668, %r667;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r669, %r898, %r425;
	mul.f32 	%r670, %r897, %r425;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs11, %rs12}, %r615;
	cvt.f32.bf16 	%r671, %rs12;
	cvt.f32.bf16 	%r672, %rs11;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r673, %r670, %r429, %r672;
	fma.rn.f32 	%r674, %r669, %r430, %r671;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r474, %r674, %r673;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r675, %r900, %r426;
	mul.f32 	%r676, %r899, %r426;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs13, %rs14}, %r616;
	cvt.f32.bf16 	%r677, %rs14;
	cvt.f32.bf16 	%r678, %rs13;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r679, %r676, %r429, %r678;
	fma.rn.f32 	%r680, %r675, %r430, %r677;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r479, %r680, %r679;
	cvt.rn.bf16x2.f32 	%r475, %r644, %r643;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r681, %r916, %r428;
	mul.f32 	%r682, %r915, %r428;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs15, %rs16}, %r624;
	cvt.f32.bf16 	%r683, %rs16;
	cvt.f32.bf16 	%r684, %rs15;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r685, %r682, %r429, %r684;
	fma.rn.f32 	%r686, %r681, %r430, %r683;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r480, %r686, %r685;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r687, %r918, %r427;
	mul.f32 	%r688, %r917, %r427;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs17, %rs18}, %r625;
	cvt.f32.bf16 	%r689, %rs18;
	cvt.f32.bf16 	%r690, %rs17;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r691, %r688, %r431, %r690;
	fma.rn.f32 	%r692, %r687, %r432, %r689;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r693, %r870, %r421;
	mul.f32 	%r694, %r869, %r421;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs19, %rs20}, %r599;
	cvt.f32.bf16 	%r695, %rs20;
	cvt.f32.bf16 	%r696, %rs19;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r697, %r694, %r431, %r696;
	fma.rn.f32 	%r698, %r693, %r432, %r695;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r492, %r698, %r697;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r699, %r872, %r422;
	mul.f32 	%r700, %r871, %r422;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs21, %rs22}, %r600;
	cvt.f32.bf16 	%r701, %rs22;
	cvt.f32.bf16 	%r702, %rs21;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r703, %r700, %r431, %r702;
	fma.rn.f32 	%r704, %r699, %r432, %r701;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r497, %r704, %r703;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r705, %r886, %r423;
	mul.f32 	%r706, %r885, %r423;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs23, %rs24}, %r609;
	cvt.f32.bf16 	%r707, %rs24;
	cvt.f32.bf16 	%r708, %rs23;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r709, %r706, %r431, %r708;
	fma.rn.f32 	%r710, %r705, %r432, %r707;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r493, %r710, %r709;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r711, %r888, %r424;
	mul.f32 	%r712, %r887, %r424;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs25, %rs26}, %r610;
	cvt.f32.bf16 	%r713, %rs26;
	cvt.f32.bf16 	%r714, %rs25;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r715, %r712, %r431, %r714;
	fma.rn.f32 	%r716, %r711, %r432, %r713;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r498, %r716, %r715;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r717, %r902, %r425;
	mul.f32 	%r718, %r901, %r425;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs27, %rs28}, %r617;
	cvt.f32.bf16 	%r719, %rs28;
	cvt.f32.bf16 	%r720, %rs27;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r721, %r718, %r431, %r720;
	fma.rn.f32 	%r722, %r717, %r432, %r719;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r494, %r722, %r721;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r723, %r904, %r426;
	mul.f32 	%r724, %r903, %r426;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs29, %rs30}, %r618;
	cvt.f32.bf16 	%r725, %rs30;
	cvt.f32.bf16 	%r726, %rs29;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r727, %r724, %r431, %r726;
	fma.rn.f32 	%r728, %r723, %r432, %r725;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r499, %r728, %r727;
	cvt.rn.bf16x2.f32 	%r495, %r692, %r691;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r729, %r920, %r428;
	mul.f32 	%r730, %r919, %r428;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs31, %rs32}, %r626;
	cvt.f32.bf16 	%r731, %rs32;
	cvt.f32.bf16 	%r732, %rs31;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r733, %r730, %r431, %r732;
	fma.rn.f32 	%r734, %r729, %r432, %r731;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r500, %r734, %r733;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r735, %r922, %r427;
	mul.f32 	%r736, %r921, %r427;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs33, %rs34}, %r627;
	cvt.f32.bf16 	%r737, %rs34;
	cvt.f32.bf16 	%r738, %rs33;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r739, %r736, %r433, %r738;
	fma.rn.f32 	%r740, %r735, %r434, %r737;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r741, %r874, %r421;
	mul.f32 	%r742, %r873, %r421;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs35, %rs36}, %r603;
	cvt.f32.bf16 	%r743, %rs36;
	cvt.f32.bf16 	%r744, %rs35;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r745, %r742, %r433, %r744;
	fma.rn.f32 	%r746, %r741, %r434, %r743;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r482, %r746, %r745;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r747, %r876, %r422;
	mul.f32 	%r748, %r875, %r422;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs37, %rs38}, %r604;
	cvt.f32.bf16 	%r749, %rs38;
	cvt.f32.bf16 	%r750, %rs37;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r751, %r748, %r433, %r750;
	fma.rn.f32 	%r752, %r747, %r434, %r749;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r487, %r752, %r751;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r753, %r890, %r423;
	mul.f32 	%r754, %r889, %r423;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs39, %rs40}, %r611;
	cvt.f32.bf16 	%r755, %rs40;
	cvt.f32.bf16 	%r756, %rs39;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r757, %r754, %r433, %r756;
	fma.rn.f32 	%r758, %r753, %r434, %r755;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r483, %r758, %r757;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r759, %r892, %r424;
	mul.f32 	%r760, %r891, %r424;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs41, %rs42}, %r612;
	cvt.f32.bf16 	%r761, %rs42;
	cvt.f32.bf16 	%r762, %rs41;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r763, %r760, %r433, %r762;
	fma.rn.f32 	%r764, %r759, %r434, %r761;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r488, %r764, %r763;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r765, %r906, %r425;
	mul.f32 	%r766, %r905, %r425;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs43, %rs44}, %r619;
	cvt.f32.bf16 	%r767, %rs44;
	cvt.f32.bf16 	%r768, %rs43;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r769, %r766, %r433, %r768;
	fma.rn.f32 	%r770, %r765, %r434, %r767;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r484, %r770, %r769;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r771, %r908, %r426;
	mul.f32 	%r772, %r907, %r426;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs45, %rs46}, %r620;
	cvt.f32.bf16 	%r773, %rs46;
	cvt.f32.bf16 	%r774, %rs45;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r775, %r772, %r433, %r774;
	fma.rn.f32 	%r776, %r771, %r434, %r773;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r489, %r776, %r775;
	cvt.rn.bf16x2.f32 	%r485, %r740, %r739;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r777, %r924, %r428;
	mul.f32 	%r778, %r923, %r428;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs47, %rs48}, %r628;
	cvt.f32.bf16 	%r779, %rs48;
	cvt.f32.bf16 	%r780, %rs47;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r781, %r778, %r433, %r780;
	fma.rn.f32 	%r782, %r777, %r434, %r779;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r490, %r782, %r781;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r783, %r926, %r427;
	mul.f32 	%r784, %r925, %r427;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs49, %rs50}, %r629;
	cvt.f32.bf16 	%r785, %rs50;
	cvt.f32.bf16 	%r786, %rs49;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r787, %r784, %r435, %r786;
	fma.rn.f32 	%r788, %r783, %r436, %r785;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r789, %r878, %r421;
	mul.f32 	%r790, %r877, %r421;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs51, %rs52}, %r605;
	cvt.f32.bf16 	%r791, %rs52;
	cvt.f32.bf16 	%r792, %rs51;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r793, %r790, %r435, %r792;
	fma.rn.f32 	%r794, %r789, %r436, %r791;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r502, %r794, %r793;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r795, %r880, %r422;
	mul.f32 	%r796, %r879, %r422;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs53, %rs54}, %r606;
	cvt.f32.bf16 	%r797, %rs54;
	cvt.f32.bf16 	%r798, %rs53;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r799, %r796, %r435, %r798;
	fma.rn.f32 	%r800, %r795, %r436, %r797;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r507, %r800, %r799;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r801, %r894, %r423;
	mul.f32 	%r802, %r893, %r423;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs55, %rs56}, %r613;
	cvt.f32.bf16 	%r803, %rs56;
	cvt.f32.bf16 	%r804, %rs55;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r805, %r802, %r435, %r804;
	fma.rn.f32 	%r806, %r801, %r436, %r803;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r503, %r806, %r805;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r807, %r896, %r424;
	mul.f32 	%r808, %r895, %r424;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs57, %rs58}, %r614;
	cvt.f32.bf16 	%r809, %rs58;
	cvt.f32.bf16 	%r810, %rs57;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r811, %r808, %r435, %r810;
	fma.rn.f32 	%r812, %r807, %r436, %r809;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r508, %r812, %r811;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r813, %r910, %r425;
	mul.f32 	%r814, %r909, %r425;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs59, %rs60}, %r621;
	cvt.f32.bf16 	%r815, %rs60;
	cvt.f32.bf16 	%r816, %rs59;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r817, %r814, %r435, %r816;
	fma.rn.f32 	%r818, %r813, %r436, %r815;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r504, %r818, %r817;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r819, %r912, %r426;
	mul.f32 	%r820, %r911, %r426;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs61, %rs62}, %r622;
	cvt.f32.bf16 	%r821, %rs62;
	cvt.f32.bf16 	%r822, %rs61;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r823, %r820, %r435, %r822;
	fma.rn.f32 	%r824, %r819, %r436, %r821;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r509, %r824, %r823;
	cvt.rn.bf16x2.f32 	%r505, %r788, %r787;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r825, %r928, %r428;
	mul.f32 	%r826, %r927, %r428;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs63, %rs64}, %r630;
	cvt.f32.bf16 	%r827, %rs64;
	cvt.f32.bf16 	%r828, %rs63;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r829, %r826, %r435, %r828;
	fma.rn.f32 	%r830, %r825, %r436, %r827;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r510, %r830, %r829;
	bar.sync 	0;
	shl.b32 	%r831, %r4, 13;
	shl.b32 	%r832, %r4, 5;
	and.b32 	%r833, %r17, 384;
	shr.u32 	%r834, %r5, 1;
	bfe.s32 	%r835, %r2, 2, 1;
	and.b32 	%r836, %r835, 4112;
	shl.b32 	%r837, %r549, 3;
	or.b32 	%r838, %r831, %r837;
	or.b32 	%r839, %r832, %r833;
	xor.b32 	%r840, %r836, %r834;
	xor.b32 	%r841, %r840, %r839;
	or.b32 	%r842, %r841, %r838;
	add.s32 	%r471, %r148, %r842;
	// begin inline asm
	st.shared.v4.b32 [ %r471 + 0 ], { %r472, %r473, %r474, %r475 };
	// end inline asm
	add.s32 	%r476, %r471, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r476 + 0 ], { %r477, %r478, %r479, %r480 };
	// end inline asm
	add.s32 	%r481, %r471, 2048;
	// begin inline asm
	st.shared.v4.b32 [ %r481 + 0 ], { %r482, %r483, %r484, %r485 };
	// end inline asm
	add.s32 	%r486, %r471, 2560;
	// begin inline asm
	st.shared.v4.b32 [ %r486 + 0 ], { %r487, %r488, %r489, %r490 };
	// end inline asm
	xor.b32 	%r843, %r842, 64;
	add.s32 	%r491, %r148, %r843;
	// begin inline asm
	st.shared.v4.b32 [ %r491 + 0 ], { %r492, %r493, %r494, %r495 };
	// end inline asm
	add.s32 	%r496, %r491, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r496 + 0 ], { %r497, %r498, %r499, %r500 };
	// end inline asm
	add.s32 	%r501, %r491, 2048;
	// begin inline asm
	st.shared.v4.b32 [ %r501 + 0 ], { %r502, %r503, %r504, %r505 };
	// end inline asm
	add.s32 	%r506, %r491, 2560;
	// begin inline asm
	st.shared.v4.b32 [ %r506 + 0 ], { %r507, %r508, %r509, %r510 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r844, %r2, 2;
	and.b32 	%r845, %r844, 896;
	shl.b32 	%r846, %r2, 8;
	and.b32 	%r847, %r846, 2048;
	setp.eq.b32 	%p24, %r863, 0;
	selp.b32 	%r848, 0, 4112, %p24;
	or.b32 	%r849, %r859, %r845;
	xor.b32 	%r850, %r849, %r848;
	or.b32 	%r851, %r850, %r847;
	add.s32 	%r852, %r148, %r851;
	ld.shared.v4.b32 	{%r511, %r519, %r527, %r535}, [%r852];
	ld.shared.v4.b32 	{%r515, %r523, %r531, %r539}, [%r852+1024];
	xor.b32 	%r853, %r851, 32;
	add.s32 	%r854, %r148, %r853;
	ld.shared.v4.b32 	{%r512, %r520, %r528, %r536}, [%r854+8192];
	ld.shared.v4.b32 	{%r516, %r524, %r532, %r540}, [%r854+9216];
	xor.b32 	%r855, %r851, 64;
	add.s32 	%r856, %r148, %r855;
	ld.shared.v4.b32 	{%r513, %r521, %r529, %r537}, [%r856+16384];
	ld.shared.v4.b32 	{%r517, %r525, %r533, %r541}, [%r856+17408];
	xor.b32 	%r857, %r851, 96;
	add.s32 	%r858, %r148, %r857;
	ld.shared.v4.b32 	{%r514, %r522, %r530, %r538}, [%r858+24576];
	ld.shared.v4.b32 	{%r518, %r526, %r534, %r542}, [%r858+25600];
	.loc	1 356 8                         // sk05_mlp_gateup.py:356:8
	// begin inline asm
	@%p7 st.global.v4.b32 [ %rd109 + 0 ], { %r511, %r512, %r513, %r514 };
	// end inline asm
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd110 + 0 ], { %r515, %r516, %r517, %r518 };
	// end inline asm
	// begin inline asm
	@%p9 st.global.v4.b32 [ %rd111 + 0 ], { %r519, %r520, %r521, %r522 };
	// end inline asm
	// begin inline asm
	@%p10 st.global.v4.b32 [ %rd112 + 0 ], { %r523, %r524, %r525, %r526 };
	// end inline asm
	// begin inline asm
	@%p11 st.global.v4.b32 [ %rd113 + 0 ], { %r527, %r528, %r529, %r530 };
	// end inline asm
	// begin inline asm
	@%p12 st.global.v4.b32 [ %rd114 + 0 ], { %r531, %r532, %r533, %r534 };
	// end inline asm
	// begin inline asm
	@%p13 st.global.v4.b32 [ %rd115 + 0 ], { %r535, %r536, %r537, %r538 };
	// end inline asm
	// begin inline asm
	@%p14 st.global.v4.b32 [ %rd116 + 0 ], { %r539, %r540, %r541, %r542 };
	// end inline asm
	.loc	1 354 4                         // sk05_mlp_gateup.py:354:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/usr/local/lib/python3.12/dist-packages/vllm/_genesis/kernels/sk05_mlp_gateup.py"
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
.b32 201                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0xc2 DW_TAG_compile_unit
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
.b8 53
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 103
.b8 97
.b8 116
.b8 101
.b8 117
.b8 112
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 117
.b8 115
.b8 114
.b8 47
.b8 108
.b8 111
.b8 99
.b8 97
.b8 108
.b8 47
.b8 108
.b8 105
.b8 98
.b8 47
.b8 112
.b8 121
.b8 116
.b8 104
.b8 111
.b8 110
.b8 51
.b8 46
.b8 49
.b8 50
.b8 47
.b8 100
.b8 105
.b8 115
.b8 116
.b8 45
.b8 112
.b8 97
.b8 99
.b8 107
.b8 97
.b8 103
.b8 101
.b8 115
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
.b8 2                                   // Abbrev [2] 0x6a:0x1a DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 53
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 103
.b8 97
.b8 116
.b8 101
.b8 117
.b8 112
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x84:0x48 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 106                                // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x99:0x19 DW_TAG_inlined_subroutine
.b32 106                                // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 62                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0xb2:0x19 DW_TAG_inlined_subroutine
.b32 106                                // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 63                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_2 = _Nativo(
    "sk05_mlp_gateup/tile128x128x128_shift0_abi15",
    _PTX_2, "_sk05_mlp_gateup_kernel",
    warps=8, shared=65536,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 18],
    horneado={11: 1, 12: 1, 15: 1, 17: 1, 19: 1, 20: 128, 21: 128, 22: 128, 23: 8, 24: 128},
    div16=[8, 9, 10, 13, 14, 16],
)

_PTX_3 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk05_mlp_gateup_kernel // -- Begin function _sk05_mlp_gateup_kernel
.extern .shared .align 16 .b8 global_smem[];
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
                                        // @_sk05_mlp_gateup_kernel
.visible .entry _sk05_mlp_gateup_kernel(
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_6,
	.param .u32 _sk05_mlp_gateup_kernel_param_7,
	.param .u32 _sk05_mlp_gateup_kernel_param_8,
	.param .u32 _sk05_mlp_gateup_kernel_param_9,
	.param .u32 _sk05_mlp_gateup_kernel_param_10,
	.param .u32 _sk05_mlp_gateup_kernel_param_11,
	.param .u32 _sk05_mlp_gateup_kernel_param_12,
	.param .u32 _sk05_mlp_gateup_kernel_param_13,
	.param .u32 _sk05_mlp_gateup_kernel_param_14,
	.param .u32 _sk05_mlp_gateup_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_16,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_17
)
.reqntid 256
{
	.reg .pred 	%p<25>;
	.reg .b16 	%rs<129>;
	.reg .b32 	%r<974>;
	.reg .b64 	%rd<204>;
	.loc	1 307 0                         // sk05_mlp_gateup.py:307:0
$L__func_begin0:
	.loc	1 307 0                         // sk05_mlp_gateup.py:307:0

// %bb.0:
	ld.param.b32 	%r30, [_sk05_mlp_gateup_kernel_param_14];
	ld.param.b32 	%r29, [_sk05_mlp_gateup_kernel_param_13];
	ld.param.b32 	%r28, [_sk05_mlp_gateup_kernel_param_12];
	ld.param.b32 	%r27, [_sk05_mlp_gateup_kernel_param_9];
	ld.param.b32 	%r26, [_sk05_mlp_gateup_kernel_param_8];
	ld.param.b32 	%r25, [_sk05_mlp_gateup_kernel_param_7];
	ld.param.b64 	%rd33, [_sk05_mlp_gateup_kernel_param_5];
	ld.param.b64 	%rd32, [_sk05_mlp_gateup_kernel_param_4];
	ld.param.b64 	%rd31, [_sk05_mlp_gateup_kernel_param_3];
	ld.param.b64 	%rd30, [_sk05_mlp_gateup_kernel_param_2];
	ld.param.b64 	%rd29, [_sk05_mlp_gateup_kernel_param_1];
	ld.param.b64 	%rd28, [_sk05_mlp_gateup_kernel_param_0];
$L__tmp0:
	.loc	1 317 24                        // sk05_mlp_gateup.py:317:24
	mov.u32 	%r50, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk05_mlp_gateup.py:318:27 ]
	add.s32 	%r51, %r25, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk05_mlp_gateup.py:318:27 ]
	shr.s32 	%r52, %r51, 31;
	shr.u32 	%r53, %r52, 25;
	add.s32 	%r54, %r51, %r53;
	shr.s32 	%r55, %r54, 7;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk05_mlp_gateup.py:319:27 ]
	add.s32 	%r56, %r26, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk05_mlp_gateup.py:319:27 ]
	shr.s32 	%r57, %r56, 31;
	shr.u32 	%r58, %r57, 25;
	add.s32 	%r59, %r56, %r58;
	shr.s32 	%r60, %r59, 7;
$L__tmp3:
	.loc	1 320 29                        // sk05_mlp_gateup.py:320:29
	shl.b32 	%r61, %r60, 3;
	.loc	1 321 22                        // sk05_mlp_gateup.py:321:22
	div.s32 	%r62, %r50, %r61;
	.loc	1 321 38                        // sk05_mlp_gateup.py:321:38
	shl.b32 	%r63, %r62, 3;
	.loc	1 322 30                        // sk05_mlp_gateup.py:322:30
	sub.s32 	%r64, %r55, %r63;
	ld.param.b32 	%r65, [_sk05_mlp_gateup_kernel_param_10];
	.loc	1 322 39                        // sk05_mlp_gateup.py:322:39
	min.s32 	%r66, %r64, 8;
	ld.param.b32 	%r67, [_sk05_mlp_gateup_kernel_param_11];
	.loc	1 323 30                        // sk05_mlp_gateup.py:323:30
	mul.lo.s32 	%r68, %r62, %r61;
	sub.s32 	%r69, %r50, %r68;
	.loc	1 324 36                        // sk05_mlp_gateup.py:324:36
	div.s32 	%r70, %r69, %r66;
	.loc	1 323 46                        // sk05_mlp_gateup.py:323:46
	mul.lo.s32 	%r71, %r70, %r66;
	sub.s32 	%r72, %r69, %r71;
	.loc	1 323 23                        // sk05_mlp_gateup.py:323:23
	add.s32 	%r73, %r72, %r63;
	.loc	1 326 22                        // sk05_mlp_gateup.py:326:22
	shl.b32 	%r1, %r73, 7;
	.loc	1 326 45                        // sk05_mlp_gateup.py:326:45
	mov.u32 	%r2, %tid.x;
	and.b32 	%r3, %r2, 248;
	bfe.u32 	%r74, %r2, 3, 5;
	or.b32 	%r75, %r74, 32;
	or.b32 	%r76, %r74, 64;
	or.b32 	%r77, %r74, 96;
	and.b32 	%r4, %r2, 3;
	shl.b32 	%r78, %r4, 1;
	and.b32 	%r5, %r2, 96;
	shr.u32 	%r79, %r5, 2;
	or.b32 	%r80, %r78, %r79;
	and.b32 	%r6, %r2, 7;
	shl.b32 	%r81, %r6, 4;
	and.b32 	%r7, %r2, 15;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r82, %r1, %r74;
	or.b32 	%r83, %r1, %r75;
	or.b32 	%r84, %r1, %r76;
	or.b32 	%r85, %r1, %r77;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r8, %r82, %r25;
	rem.s32 	%r9, %r83, %r25;
	rem.s32 	%r10, %r84, %r25;
	rem.s32 	%r11, %r85, %r25;
	.loc	1 327 22                        // sk05_mlp_gateup.py:327:22
	shl.b32 	%r12, %r70, 7;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r86, %r12, %r74;
	or.b32 	%r87, %r12, %r75;
	or.b32 	%r88, %r12, %r76;
	or.b32 	%r89, %r12, %r77;
	or.b32 	%r90, %r12, %r80;
	or.b32 	%r92, %r90, 32;
	or.b32 	%r94, %r90, 64;
	or.b32 	%r96, %r90, 96;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r98, %r86, %r26;
	rem.s32 	%r99, %r87, %r26;
	rem.s32 	%r100, %r88, %r26;
	rem.s32 	%r101, %r89, %r26;
	rem.s32 	%r13, %r90, %r26;
	rem.s32 	%r14, %r92, %r26;
	rem.s32 	%r15, %r94, %r26;
	rem.s32 	%r16, %r96, %r26;
	.loc	1 330 39                        // sk05_mlp_gateup.py:330:39
	mul.lo.s32 	%r106, %r8, %r65;
	mul.lo.s32 	%r107, %r9, %r65;
	mul.lo.s32 	%r108, %r10, %r65;
	mul.lo.s32 	%r109, %r11, %r65;
	.loc	1 330 21                        // sk05_mlp_gateup.py:330:21
	cvt.s64.s32 	%rd1, %r106;
	add.s64 	%rd51, %rd28, %rd1;
	cvt.s64.s32 	%rd2, %r107;
	add.s64 	%rd52, %rd28, %rd2;
	cvt.s64.s32 	%rd3, %r108;
	add.s64 	%rd53, %rd28, %rd3;
	cvt.s64.s32 	%rd4, %r109;
	add.s64 	%rd54, %rd28, %rd4;
	.loc	1 330 51                        // sk05_mlp_gateup.py:330:51
	cvt.u64.u32 	%rd5, %r81;
	add.s64 	%rd34, %rd51, %rd5;
	add.s64 	%rd35, %rd52, %rd5;
	add.s64 	%rd36, %rd53, %rd5;
	add.s64 	%rd37, %rd54, %rd5;
	.loc	1 331 21                        // sk05_mlp_gateup.py:331:21
	add.s64 	%rd55, %rd29, %rd5;
	.loc	1 331 69                        // sk05_mlp_gateup.py:331:69
	mul.lo.s32 	%r110, %r98, %r67;
	mul.lo.s32 	%r111, %r99, %r67;
	mul.lo.s32 	%r112, %r100, %r67;
	mul.lo.s32 	%r113, %r101, %r67;
	.loc	1 331 51                        // sk05_mlp_gateup.py:331:51
	cvt.s64.s32 	%rd6, %r110;
	add.s64 	%rd38, %rd55, %rd6;
	cvt.s64.s32 	%rd7, %r111;
	add.s64 	%rd39, %rd55, %rd7;
	cvt.s64.s32 	%rd8, %r112;
	add.s64 	%rd40, %rd55, %rd8;
	cvt.s64.s32 	%rd9, %r113;
	add.s64 	%rd41, %rd55, %rd9;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	setp.gt.s32 	%p1, %r27, 127;
	.loc	1 344 30                        // sk05_mlp_gateup.py:344:30
	shl.b32 	%r17, %r2, 4;
	and.b32 	%r146, %r17, 4080;
	and.b32 	%r18, %r2, 56;
	shl.b32 	%r147, %r18, 1;
	xor.b32 	%r148, %r146, %r147;
	mov.b32 	%r149, global_smem;
	add.s32 	%r32, %r149, %r148;
	selp.b32 	%r33, 16, 0, %p1;
	// begin inline asm
	cp.async.cg.shared.global [ %r32 + 0 ], [ %rd34 + 0 ], 0x10, %r33;
	// end inline asm
	add.s32 	%r34, %r32, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r34 + 0 ], [ %rd35 + 0 ], 0x10, %r33;
	// end inline asm
	add.s32 	%r35, %r32, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r35 + 0 ], [ %rd36 + 0 ], 0x10, %r33;
	// end inline asm
	add.s32 	%r36, %r32, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r36 + 0 ], [ %rd37 + 0 ], 0x10, %r33;
	// end inline asm
	cp.async.commit_group;
	.loc	1 344 47                        // sk05_mlp_gateup.py:344:47
	add.s32 	%r37, %r32, 32768;
	// begin inline asm
	cp.async.cg.shared.global [ %r37 + 0 ], [ %rd38 + 0 ], 0x10, %r33;
	// end inline asm
	add.s32 	%r38, %r32, 36864;
	// begin inline asm
	cp.async.cg.shared.global [ %r38 + 0 ], [ %rd39 + 0 ], 0x10, %r33;
	// end inline asm
	add.s32 	%r39, %r32, 40960;
	// begin inline asm
	cp.async.cg.shared.global [ %r39 + 0 ], [ %rd40 + 0 ], 0x10, %r33;
	// end inline asm
	add.s32 	%r40, %r32, 45056;
	// begin inline asm
	cp.async.cg.shared.global [ %r40 + 0 ], [ %rd41 + 0 ], 0x10, %r33;
	// end inline asm
	cp.async.commit_group;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	setp.gt.s32 	%p2, %r27, 255;
	.loc	1 345 18                        // sk05_mlp_gateup.py:345:18
	add.s64 	%rd42, %rd34, 128;
	add.s64 	%rd43, %rd35, 128;
	add.s64 	%rd44, %rd36, 128;
	add.s64 	%rd45, %rd37, 128;
	.loc	1 346 18                        // sk05_mlp_gateup.py:346:18
	add.s64 	%rd46, %rd38, 128;
	add.s64 	%rd47, %rd39, 128;
	add.s64 	%rd48, %rd40, 128;
	add.s64 	%rd49, %rd41, 128;
	.loc	1 344 30                        // sk05_mlp_gateup.py:344:30
	bar.sync 	0;
	add.s32 	%r41, %r32, 16384;
	selp.b32 	%r42, 16, 0, %p2;
	// begin inline asm
	cp.async.cg.shared.global [ %r41 + 0 ], [ %rd42 + 0 ], 0x10, %r42;
	// end inline asm
	add.s32 	%r43, %r32, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r43 + 0 ], [ %rd43 + 0 ], 0x10, %r42;
	// end inline asm
	add.s32 	%r44, %r32, 24576;
	// begin inline asm
	cp.async.cg.shared.global [ %r44 + 0 ], [ %rd44 + 0 ], 0x10, %r42;
	// end inline asm
	add.s32 	%r45, %r32, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r45 + 0 ], [ %rd45 + 0 ], 0x10, %r42;
	// end inline asm
	cp.async.commit_group;
	.loc	1 344 47                        // sk05_mlp_gateup.py:344:47
	add.s32 	%r46, %r32, 49152;
	// begin inline asm
	cp.async.cg.shared.global [ %r46 + 0 ], [ %rd46 + 0 ], 0x10, %r42;
	// end inline asm
	add.s32 	%r47, %r32, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r47 + 0 ], [ %rd47 + 0 ], 0x10, %r42;
	// end inline asm
	add.s32 	%r48, %r32, 57344;
	// begin inline asm
	cp.async.cg.shared.global [ %r48 + 0 ], [ %rd48 + 0 ], 0x10, %r42;
	// end inline asm
	add.s32 	%r49, %r32, 61440;
	// begin inline asm
	cp.async.cg.shared.global [ %r49 + 0 ], [ %rd49 + 0 ], 0x10, %r42;
	// end inline asm
	cp.async.commit_group;
	cvt.u32.u64 	%r904, %rd5;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	@%p1 bra 	$L__BB0_2;
	bra.uni 	$L__BB0_1;
$L__BB0_2:                              // %.lr.ph
	.loc	1 0 23                          // sk05_mlp_gateup.py:0:23
	ld.param.b32 	%r31, [_sk05_mlp_gateup_kernel_param_15];
	ld.param.b64 	%rd50, [_sk05_mlp_gateup_kernel_param_6];
	or.b32 	%r91, %r90, 1;
	or.b32 	%r93, %r90, 33;
	or.b32 	%r95, %r90, 65;
	or.b32 	%r97, %r90, 97;
	rem.s32 	%r102, %r91, %r26;
	rem.s32 	%r103, %r93, %r26;
	rem.s32 	%r104, %r95, %r26;
	rem.s32 	%r105, %r97, %r26;
	shr.s32 	%r114, %r13, 31;
	shr.u32 	%r115, %r114, 25;
	add.s32 	%r116, %r13, %r115;
	shr.s32 	%r117, %r116, 7;
	shr.s32 	%r118, %r102, 31;
	shr.u32 	%r119, %r118, 25;
	add.s32 	%r120, %r102, %r119;
	shr.s32 	%r121, %r120, 7;
	shr.s32 	%r122, %r14, 31;
	shr.u32 	%r123, %r122, 25;
	add.s32 	%r124, %r14, %r123;
	shr.s32 	%r125, %r124, 7;
	shr.s32 	%r126, %r103, 31;
	shr.u32 	%r127, %r126, 25;
	add.s32 	%r128, %r103, %r127;
	shr.s32 	%r129, %r128, 7;
	shr.s32 	%r130, %r15, 31;
	shr.u32 	%r131, %r130, 25;
	add.s32 	%r132, %r15, %r131;
	shr.s32 	%r133, %r132, 7;
	shr.s32 	%r134, %r104, 31;
	shr.u32 	%r135, %r134, 25;
	add.s32 	%r136, %r104, %r135;
	shr.s32 	%r137, %r136, 7;
	shr.s32 	%r138, %r16, 31;
	shr.u32 	%r139, %r138, 25;
	add.s32 	%r140, %r16, %r139;
	shr.s32 	%r141, %r140, 7;
	shr.s32 	%r142, %r105, 31;
	shr.u32 	%r143, %r142, 25;
	add.s32 	%r144, %r105, %r143;
	shr.s32 	%r145, %r144, 7;
	mad.wide.s32 	%rd10, %r117, 4, %rd50;
	mad.wide.s32 	%rd11, %r121, 4, %rd50;
	mad.wide.s32 	%rd12, %r125, 4, %rd50;
	mad.wide.s32 	%rd13, %r129, 4, %rd50;
	mad.wide.s32 	%rd14, %r133, 4, %rd50;
	mad.wide.s32 	%rd15, %r137, 4, %rd50;
	mad.wide.s32 	%rd16, %r141, 4, %rd50;
	mad.wide.s32 	%rd17, %r145, 4, %rd50;
	.loc	1 335 28                        // sk05_mlp_gateup.py:335:28
	shr.u32 	%r150, %r27, 7;
	add.s32 	%r151, %r150, -2;
	shl.b32 	%r152, %r7, 7;
	and.b32 	%r153, %r17, 2160;
	and.b32 	%r908, %r2, 16;
	or.b32 	%r154, %r152, %r153;
	xor.b32 	%r19, %r154, %r908;
	xor.b32 	%r20, %r19, 32;
	xor.b32 	%r21, %r19, 64;
	xor.b32 	%r22, %r19, 96;
	shl.b32 	%r155, %r6, 7;
	shl.b32 	%r156, %r5, 5;
	shl.b32 	%r909, %r2, 1;
	and.b32 	%r157, %r909, 48;
	or.b32 	%r158, %r155, %r156;
	xor.b32 	%r159, %r904, %r157;
	or.b32 	%r23, %r158, %r159;
	xor.b32 	%r24, %r23, 64;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	cvt.s64.s32 	%rd18, %r151;
	and.b32 	%r160, %r27, -128;
	cvt.u64.u32 	%rd19, %r160;
	add.s64 	%rd56, %rd5, %rd9;
	add.s64 	%rd57, %rd56, %rd29;
	add.s64 	%rd20, %rd57, 256;
	add.s64 	%rd58, %rd5, %rd8;
	add.s64 	%rd59, %rd58, %rd29;
	add.s64 	%rd21, %rd59, 256;
	add.s64 	%rd60, %rd5, %rd7;
	add.s64 	%rd61, %rd60, %rd29;
	add.s64 	%rd22, %rd61, 256;
	add.s64 	%rd62, %rd5, %rd6;
	add.s64 	%rd63, %rd62, %rd29;
	add.s64 	%rd23, %rd63, 256;
	add.s64 	%rd64, %rd5, %rd4;
	add.s64 	%rd65, %rd64, %rd28;
	add.s64 	%rd24, %rd65, 256;
	add.s64 	%rd66, %rd5, %rd3;
	add.s64 	%rd67, %rd66, %rd28;
	add.s64 	%rd25, %rd67, 256;
	add.s64 	%rd68, %rd5, %rd2;
	add.s64 	%rd69, %rd68, %rd28;
	add.s64 	%rd26, %rd69, 256;
	add.s64 	%rd70, %rd5, %rd1;
	add.s64 	%rd71, %rd70, %rd28;
	add.s64 	%rd27, %rd71, 256;
	mov.b32 	%r910, 0f00000000;
	mov.b32 	%r907, 1;
	mov.b32 	%r906, -1;
	mov.b64 	%rd202, 0;
	mov.b32 	%r169, 0;
	mov.b32 	%r905, %r169;
	mov.b64 	%rd203, %rd202;
	mov.b32 	%r911, %r910;
	mov.b32 	%r912, %r910;
	mov.b32 	%r913, %r910;
	mov.b32 	%r914, %r910;
	mov.b32 	%r915, %r910;
	mov.b32 	%r916, %r910;
	mov.b32 	%r917, %r910;
	mov.b32 	%r918, %r910;
	mov.b32 	%r919, %r910;
	mov.b32 	%r920, %r910;
	mov.b32 	%r921, %r910;
	mov.b32 	%r922, %r910;
	mov.b32 	%r923, %r910;
	mov.b32 	%r924, %r910;
	mov.b32 	%r925, %r910;
	mov.b32 	%r926, %r910;
	mov.b32 	%r927, %r910;
	mov.b32 	%r928, %r910;
	mov.b32 	%r929, %r910;
	mov.b32 	%r930, %r910;
	mov.b32 	%r931, %r910;
	mov.b32 	%r932, %r910;
	mov.b32 	%r933, %r910;
	mov.b32 	%r934, %r910;
	mov.b32 	%r935, %r910;
	mov.b32 	%r936, %r910;
	mov.b32 	%r937, %r910;
	mov.b32 	%r938, %r910;
	mov.b32 	%r939, %r910;
	mov.b32 	%r940, %r910;
	mov.b32 	%r941, %r910;
	mov.b32 	%r942, %r910;
	mov.b32 	%r943, %r910;
	mov.b32 	%r944, %r910;
	mov.b32 	%r945, %r910;
	mov.b32 	%r946, %r910;
	mov.b32 	%r947, %r910;
	mov.b32 	%r948, %r910;
	mov.b32 	%r949, %r910;
	mov.b32 	%r950, %r910;
	mov.b32 	%r951, %r910;
	mov.b32 	%r952, %r910;
	mov.b32 	%r953, %r910;
	mov.b32 	%r954, %r910;
	mov.b32 	%r955, %r910;
	mov.b32 	%r956, %r910;
	mov.b32 	%r957, %r910;
	mov.b32 	%r958, %r910;
	mov.b32 	%r959, %r910;
	mov.b32 	%r960, %r910;
	mov.b32 	%r961, %r910;
	mov.b32 	%r962, %r910;
	mov.b32 	%r963, %r910;
	mov.b32 	%r964, %r910;
	mov.b32 	%r965, %r910;
	mov.b32 	%r966, %r910;
	mov.b32 	%r967, %r910;
	mov.b32 	%r968, %r910;
	mov.b32 	%r969, %r910;
	mov.b32 	%r970, %r910;
	mov.b32 	%r971, %r910;
	mov.b32 	%r972, %r910;
	mov.b32 	%r973, %r910;
$L__BB0_3:                              // %__nv_exp2f.exit
                                        // =>This Inner Loop Header: Depth=1
	setp.lt.s64 	%p3, %rd203, %rd18;
	add.s32 	%r339, %r906, 1;
	setp.gt.s32 	%p4, %r339, 1;
	selp.b32 	%r906, 0, %r339, %p4;
	.loc	1 342 39                        // sk05_mlp_gateup.py:342:39
	mul.wide.s32 	%rd88, %r905, 4;
	add.s64 	%rd72, %rd10, %rd88;
	add.s64 	%rd73, %rd11, %rd88;
	add.s64 	%rd74, %rd12, %rd88;
	add.s64 	%rd75, %rd13, %rd88;
	add.s64 	%rd76, %rd14, %rd88;
	add.s64 	%rd77, %rd15, %rd88;
	add.s64 	%rd78, %rd16, %rd88;
	add.s64 	%rd79, %rd17, %rd88;
	.loc	1 342 29                        // sk05_mlp_gateup.py:342:29
	// begin inline asm
	mov.u32 %r161, 0x0;
	ld.global.b32 { %r161 }, [ %rd72 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r162, 0x0;
	ld.global.b32 { %r162 }, [ %rd73 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r163, 0x0;
	ld.global.b32 { %r163 }, [ %rd74 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r164, 0x0;
	ld.global.b32 { %r164 }, [ %rd75 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r165, 0x0;
	ld.global.b32 { %r165 }, [ %rd76 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r166, 0x0;
	ld.global.b32 { %r166 }, [ %rd77 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r167, 0x0;
	ld.global.b32 { %r167 }, [ %rd78 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r168, 0x0;
	ld.global.b32 { %r168 }, [ %rd79 + 0 ];
	// end inline asm
	.loc	1 342 21                        // sk05_mlp_gateup.py:342:21
	ex2.approx.ftz.f32 	%r340, %r161;
	ex2.approx.ftz.f32 	%r341, %r162;
	ex2.approx.ftz.f32 	%r342, %r163;
	ex2.approx.ftz.f32 	%r343, %r164;
	ex2.approx.ftz.f32 	%r344, %r165;
	ex2.approx.ftz.f32 	%r345, %r166;
	ex2.approx.ftz.f32 	%r346, %r167;
	ex2.approx.ftz.f32 	%r347, %r168;
	.loc	1 344 30                        // sk05_mlp_gateup.py:344:30
	cp.async.wait_group 	2;
	bar.sync 	0;
	shl.b32 	%r348, %r906, 14;
	add.s32 	%r349, %r149, %r348;
	add.s32 	%r350, %r349, %r19;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r170, %r171, %r172, %r173}, [%r350];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r182, %r183, %r184, %r185}, [%r350+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r186, %r187, %r188, %r189}, [%r350+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r190, %r191, %r192, %r193}, [%r350+12288];
	add.s32 	%r351, %r349, %r20;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r198, %r199, %r200, %r201}, [%r351];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r226, %r227, %r228, %r229}, [%r351+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r246, %r247, %r248, %r249}, [%r351+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r266, %r267, %r268, %r269}, [%r351+12288];
	add.s32 	%r352, %r349, %r21;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r282, %r283, %r284, %r285}, [%r352];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r294, %r295, %r296, %r297}, [%r352+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r298, %r299, %r300, %r301}, [%r352+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r302, %r303, %r304, %r305}, [%r352+12288];
	add.s32 	%r353, %r349, %r22;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r306, %r307, %r308, %r309}, [%r353];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r318, %r319, %r320, %r321}, [%r353+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r322, %r323, %r324, %r325}, [%r353+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r326, %r327, %r328, %r329}, [%r353+12288];
	.loc	1 344 47                        // sk05_mlp_gateup.py:344:47
	add.s32 	%r354, %r349, %r23;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r174, %r175, %r202, %r203}, [%r354+32768];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r176, %r177, %r208, %r209}, [%r354+36864];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r178, %r179, %r214, %r215}, [%r354+40960];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r180, %r181, %r220, %r221}, [%r354+45056];
	add.s32 	%r355, %r349, %r24;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r286, %r287, %r310, %r311}, [%r355+32768];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r288, %r289, %r312, %r313}, [%r355+36864];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r290, %r291, %r314, %r315}, [%r355+40960];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r292, %r293, %r316, %r317}, [%r355+45056];
	.loc	1 344 39                        // sk05_mlp_gateup.py:344:39
	mov.b32 	%r194, %r169;
	mov.b32 	%r195, %r169;
	mov.b32 	%r196, %r169;
	mov.b32 	%r197, %r169;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r194, %r195, %r196, %r197 }, { %r170, %r171, %r172, %r173 }, { %r174, %r175 }, { %r194, %r195, %r196, %r197 };
	// end inline asm
	mov.b32 	%r204, %r169;
	mov.b32 	%r205, %r169;
	mov.b32 	%r206, %r169;
	mov.b32 	%r207, %r169;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r204, %r205, %r206, %r207 }, { %r170, %r171, %r172, %r173 }, { %r176, %r177 }, { %r204, %r205, %r206, %r207 };
	// end inline asm
	mov.b32 	%r210, %r169;
	mov.b32 	%r211, %r169;
	mov.b32 	%r212, %r169;
	mov.b32 	%r213, %r169;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r210, %r211, %r212, %r213 }, { %r170, %r171, %r172, %r173 }, { %r178, %r179 }, { %r210, %r211, %r212, %r213 };
	// end inline asm
	mov.b32 	%r216, %r169;
	mov.b32 	%r217, %r169;
	mov.b32 	%r218, %r169;
	mov.b32 	%r219, %r169;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r216, %r217, %r218, %r219 }, { %r170, %r171, %r172, %r173 }, { %r180, %r181 }, { %r216, %r217, %r218, %r219 };
	// end inline asm
	mov.b32 	%r222, %r169;
	mov.b32 	%r223, %r169;
	mov.b32 	%r224, %r169;
	mov.b32 	%r225, %r169;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r222, %r223, %r224, %r225 }, { %r182, %r183, %r184, %r185 }, { %r174, %r175 }, { %r222, %r223, %r224, %r225 };
	// end inline asm
	mov.b32 	%r230, %r169;
	mov.b32 	%r231, %r169;
	mov.b32 	%r232, %r169;
	mov.b32 	%r233, %r169;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r230, %r231, %r232, %r233 }, { %r182, %r183, %r184, %r185 }, { %r176, %r177 }, { %r230, %r231, %r232, %r233 };
	// end inline asm
	mov.b32 	%r234, %r169;
	mov.b32 	%r235, %r169;
	mov.b32 	%r236, %r169;
	mov.b32 	%r237, %r169;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r234, %r235, %r236, %r237 }, { %r182, %r183, %r184, %r185 }, { %r178, %r179 }, { %r234, %r235, %r236, %r237 };
	// end inline asm
	mov.b32 	%r238, %r169;
	mov.b32 	%r239, %r169;
	mov.b32 	%r240, %r169;
	mov.b32 	%r241, %r169;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r238, %r239, %r240, %r241 }, { %r182, %r183, %r184, %r185 }, { %r180, %r181 }, { %r238, %r239, %r240, %r241 };
	// end inline asm
	mov.b32 	%r242, %r169;
	mov.b32 	%r243, %r169;
	mov.b32 	%r244, %r169;
	mov.b32 	%r245, %r169;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r242, %r243, %r244, %r245 }, { %r186, %r187, %r188, %r189 }, { %r174, %r175 }, { %r242, %r243, %r244, %r245 };
	// end inline asm
	mov.b32 	%r250, %r169;
	mov.b32 	%r251, %r169;
	mov.b32 	%r252, %r169;
	mov.b32 	%r253, %r169;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r250, %r251, %r252, %r253 }, { %r186, %r187, %r188, %r189 }, { %r176, %r177 }, { %r250, %r251, %r252, %r253 };
	// end inline asm
	mov.b32 	%r254, %r169;
	mov.b32 	%r255, %r169;
	mov.b32 	%r256, %r169;
	mov.b32 	%r257, %r169;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r254, %r255, %r256, %r257 }, { %r186, %r187, %r188, %r189 }, { %r178, %r179 }, { %r254, %r255, %r256, %r257 };
	// end inline asm
	mov.b32 	%r258, %r169;
	mov.b32 	%r259, %r169;
	mov.b32 	%r260, %r169;
	mov.b32 	%r261, %r169;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r258, %r259, %r260, %r261 }, { %r186, %r187, %r188, %r189 }, { %r180, %r181 }, { %r258, %r259, %r260, %r261 };
	// end inline asm
	mov.b32 	%r262, %r169;
	mov.b32 	%r263, %r169;
	mov.b32 	%r264, %r169;
	mov.b32 	%r265, %r169;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r262, %r263, %r264, %r265 }, { %r190, %r191, %r192, %r193 }, { %r174, %r175 }, { %r262, %r263, %r264, %r265 };
	// end inline asm
	mov.b32 	%r270, %r169;
	mov.b32 	%r271, %r169;
	mov.b32 	%r272, %r169;
	mov.b32 	%r273, %r169;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r270, %r271, %r272, %r273 }, { %r190, %r191, %r192, %r193 }, { %r176, %r177 }, { %r270, %r271, %r272, %r273 };
	// end inline asm
	mov.b32 	%r274, %r169;
	mov.b32 	%r275, %r169;
	mov.b32 	%r276, %r169;
	mov.b32 	%r277, %r169;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r274, %r275, %r276, %r277 }, { %r190, %r191, %r192, %r193 }, { %r178, %r179 }, { %r274, %r275, %r276, %r277 };
	// end inline asm
	mov.b32 	%r281, %r169;
	mov.b32 	%r278, %r169;
	mov.b32 	%r279, %r169;
	mov.b32 	%r280, %r169;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r278, %r279, %r280, %r281 }, { %r190, %r191, %r192, %r193 }, { %r180, %r181 }, { %r278, %r279, %r280, %r281 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r194, %r195, %r196, %r197 }, { %r198, %r199, %r200, %r201 }, { %r202, %r203 }, { %r194, %r195, %r196, %r197 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r204, %r205, %r206, %r207 }, { %r198, %r199, %r200, %r201 }, { %r208, %r209 }, { %r204, %r205, %r206, %r207 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r210, %r211, %r212, %r213 }, { %r198, %r199, %r200, %r201 }, { %r214, %r215 }, { %r210, %r211, %r212, %r213 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r216, %r217, %r218, %r219 }, { %r198, %r199, %r200, %r201 }, { %r220, %r221 }, { %r216, %r217, %r218, %r219 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r222, %r223, %r224, %r225 }, { %r226, %r227, %r228, %r229 }, { %r202, %r203 }, { %r222, %r223, %r224, %r225 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r230, %r231, %r232, %r233 }, { %r226, %r227, %r228, %r229 }, { %r208, %r209 }, { %r230, %r231, %r232, %r233 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r234, %r235, %r236, %r237 }, { %r226, %r227, %r228, %r229 }, { %r214, %r215 }, { %r234, %r235, %r236, %r237 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r238, %r239, %r240, %r241 }, { %r226, %r227, %r228, %r229 }, { %r220, %r221 }, { %r238, %r239, %r240, %r241 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r242, %r243, %r244, %r245 }, { %r246, %r247, %r248, %r249 }, { %r202, %r203 }, { %r242, %r243, %r244, %r245 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r250, %r251, %r252, %r253 }, { %r246, %r247, %r248, %r249 }, { %r208, %r209 }, { %r250, %r251, %r252, %r253 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r254, %r255, %r256, %r257 }, { %r246, %r247, %r248, %r249 }, { %r214, %r215 }, { %r254, %r255, %r256, %r257 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r258, %r259, %r260, %r261 }, { %r246, %r247, %r248, %r249 }, { %r220, %r221 }, { %r258, %r259, %r260, %r261 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r262, %r263, %r264, %r265 }, { %r266, %r267, %r268, %r269 }, { %r202, %r203 }, { %r262, %r263, %r264, %r265 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r270, %r271, %r272, %r273 }, { %r266, %r267, %r268, %r269 }, { %r208, %r209 }, { %r270, %r271, %r272, %r273 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r274, %r275, %r276, %r277 }, { %r266, %r267, %r268, %r269 }, { %r214, %r215 }, { %r274, %r275, %r276, %r277 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r278, %r279, %r280, %r281 }, { %r266, %r267, %r268, %r269 }, { %r220, %r221 }, { %r278, %r279, %r280, %r281 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r194, %r195, %r196, %r197 }, { %r282, %r283, %r284, %r285 }, { %r286, %r287 }, { %r194, %r195, %r196, %r197 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r204, %r205, %r206, %r207 }, { %r282, %r283, %r284, %r285 }, { %r288, %r289 }, { %r204, %r205, %r206, %r207 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r210, %r211, %r212, %r213 }, { %r282, %r283, %r284, %r285 }, { %r290, %r291 }, { %r210, %r211, %r212, %r213 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r216, %r217, %r218, %r219 }, { %r282, %r283, %r284, %r285 }, { %r292, %r293 }, { %r216, %r217, %r218, %r219 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r222, %r223, %r224, %r225 }, { %r294, %r295, %r296, %r297 }, { %r286, %r287 }, { %r222, %r223, %r224, %r225 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r230, %r231, %r232, %r233 }, { %r294, %r295, %r296, %r297 }, { %r288, %r289 }, { %r230, %r231, %r232, %r233 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r234, %r235, %r236, %r237 }, { %r294, %r295, %r296, %r297 }, { %r290, %r291 }, { %r234, %r235, %r236, %r237 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r238, %r239, %r240, %r241 }, { %r294, %r295, %r296, %r297 }, { %r292, %r293 }, { %r238, %r239, %r240, %r241 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r242, %r243, %r244, %r245 }, { %r298, %r299, %r300, %r301 }, { %r286, %r287 }, { %r242, %r243, %r244, %r245 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r250, %r251, %r252, %r253 }, { %r298, %r299, %r300, %r301 }, { %r288, %r289 }, { %r250, %r251, %r252, %r253 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r254, %r255, %r256, %r257 }, { %r298, %r299, %r300, %r301 }, { %r290, %r291 }, { %r254, %r255, %r256, %r257 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r258, %r259, %r260, %r261 }, { %r298, %r299, %r300, %r301 }, { %r292, %r293 }, { %r258, %r259, %r260, %r261 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r262, %r263, %r264, %r265 }, { %r302, %r303, %r304, %r305 }, { %r286, %r287 }, { %r262, %r263, %r264, %r265 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r270, %r271, %r272, %r273 }, { %r302, %r303, %r304, %r305 }, { %r288, %r289 }, { %r270, %r271, %r272, %r273 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r274, %r275, %r276, %r277 }, { %r302, %r303, %r304, %r305 }, { %r290, %r291 }, { %r274, %r275, %r276, %r277 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r278, %r279, %r280, %r281 }, { %r302, %r303, %r304, %r305 }, { %r292, %r293 }, { %r278, %r279, %r280, %r281 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r194, %r195, %r196, %r197 }, { %r306, %r307, %r308, %r309 }, { %r310, %r311 }, { %r194, %r195, %r196, %r197 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r204, %r205, %r206, %r207 }, { %r306, %r307, %r308, %r309 }, { %r312, %r313 }, { %r204, %r205, %r206, %r207 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r210, %r211, %r212, %r213 }, { %r306, %r307, %r308, %r309 }, { %r314, %r315 }, { %r210, %r211, %r212, %r213 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r216, %r217, %r218, %r219 }, { %r306, %r307, %r308, %r309 }, { %r316, %r317 }, { %r216, %r217, %r218, %r219 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r222, %r223, %r224, %r225 }, { %r318, %r319, %r320, %r321 }, { %r310, %r311 }, { %r222, %r223, %r224, %r225 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r230, %r231, %r232, %r233 }, { %r318, %r319, %r320, %r321 }, { %r312, %r313 }, { %r230, %r231, %r232, %r233 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r234, %r235, %r236, %r237 }, { %r318, %r319, %r320, %r321 }, { %r314, %r315 }, { %r234, %r235, %r236, %r237 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r238, %r239, %r240, %r241 }, { %r318, %r319, %r320, %r321 }, { %r316, %r317 }, { %r238, %r239, %r240, %r241 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r242, %r243, %r244, %r245 }, { %r322, %r323, %r324, %r325 }, { %r310, %r311 }, { %r242, %r243, %r244, %r245 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r250, %r251, %r252, %r253 }, { %r322, %r323, %r324, %r325 }, { %r312, %r313 }, { %r250, %r251, %r252, %r253 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r254, %r255, %r256, %r257 }, { %r322, %r323, %r324, %r325 }, { %r314, %r315 }, { %r254, %r255, %r256, %r257 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r258, %r259, %r260, %r261 }, { %r322, %r323, %r324, %r325 }, { %r316, %r317 }, { %r258, %r259, %r260, %r261 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r262, %r263, %r264, %r265 }, { %r326, %r327, %r328, %r329 }, { %r310, %r311 }, { %r262, %r263, %r264, %r265 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r270, %r271, %r272, %r273 }, { %r326, %r327, %r328, %r329 }, { %r312, %r313 }, { %r270, %r271, %r272, %r273 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r274, %r275, %r276, %r277 }, { %r326, %r327, %r328, %r329 }, { %r314, %r315 }, { %r274, %r275, %r276, %r277 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r278, %r279, %r280, %r281 }, { %r326, %r327, %r328, %r329 }, { %r316, %r317 }, { %r278, %r279, %r280, %r281 };
	// end inline asm
	.loc	1 344 81                        // sk05_mlp_gateup.py:344:81
	cvt.rn.f32.s32 	%r356, %r278;
	cvt.rn.f32.s32 	%r357, %r279;
	cvt.rn.f32.s32 	%r358, %r280;
	cvt.rn.f32.s32 	%r359, %r281;
	cvt.rn.f32.s32 	%r360, %r274;
	cvt.rn.f32.s32 	%r361, %r275;
	cvt.rn.f32.s32 	%r362, %r276;
	cvt.rn.f32.s32 	%r363, %r277;
	cvt.rn.f32.s32 	%r364, %r270;
	cvt.rn.f32.s32 	%r365, %r271;
	cvt.rn.f32.s32 	%r366, %r272;
	cvt.rn.f32.s32 	%r367, %r273;
	cvt.rn.f32.s32 	%r368, %r262;
	cvt.rn.f32.s32 	%r369, %r263;
	cvt.rn.f32.s32 	%r370, %r264;
	cvt.rn.f32.s32 	%r371, %r265;
	cvt.rn.f32.s32 	%r372, %r258;
	cvt.rn.f32.s32 	%r373, %r259;
	cvt.rn.f32.s32 	%r374, %r260;
	cvt.rn.f32.s32 	%r375, %r261;
	cvt.rn.f32.s32 	%r376, %r254;
	cvt.rn.f32.s32 	%r377, %r255;
	cvt.rn.f32.s32 	%r378, %r256;
	cvt.rn.f32.s32 	%r379, %r257;
	cvt.rn.f32.s32 	%r380, %r250;
	cvt.rn.f32.s32 	%r381, %r251;
	cvt.rn.f32.s32 	%r382, %r252;
	cvt.rn.f32.s32 	%r383, %r253;
	cvt.rn.f32.s32 	%r384, %r242;
	cvt.rn.f32.s32 	%r385, %r243;
	cvt.rn.f32.s32 	%r386, %r244;
	cvt.rn.f32.s32 	%r387, %r245;
	cvt.rn.f32.s32 	%r388, %r238;
	cvt.rn.f32.s32 	%r389, %r239;
	cvt.rn.f32.s32 	%r390, %r240;
	cvt.rn.f32.s32 	%r391, %r241;
	cvt.rn.f32.s32 	%r392, %r234;
	cvt.rn.f32.s32 	%r393, %r235;
	cvt.rn.f32.s32 	%r394, %r236;
	cvt.rn.f32.s32 	%r395, %r237;
	cvt.rn.f32.s32 	%r396, %r230;
	cvt.rn.f32.s32 	%r397, %r231;
	cvt.rn.f32.s32 	%r398, %r232;
	cvt.rn.f32.s32 	%r399, %r233;
	cvt.rn.f32.s32 	%r400, %r222;
	cvt.rn.f32.s32 	%r401, %r223;
	cvt.rn.f32.s32 	%r402, %r224;
	cvt.rn.f32.s32 	%r403, %r225;
	cvt.rn.f32.s32 	%r404, %r216;
	cvt.rn.f32.s32 	%r405, %r217;
	cvt.rn.f32.s32 	%r406, %r218;
	cvt.rn.f32.s32 	%r407, %r219;
	cvt.rn.f32.s32 	%r408, %r210;
	cvt.rn.f32.s32 	%r409, %r211;
	cvt.rn.f32.s32 	%r410, %r212;
	cvt.rn.f32.s32 	%r411, %r213;
	cvt.rn.f32.s32 	%r412, %r204;
	cvt.rn.f32.s32 	%r413, %r205;
	cvt.rn.f32.s32 	%r414, %r206;
	cvt.rn.f32.s32 	%r415, %r207;
	cvt.rn.f32.s32 	%r416, %r194;
	cvt.rn.f32.s32 	%r417, %r195;
	cvt.rn.f32.s32 	%r418, %r196;
	cvt.rn.f32.s32 	%r419, %r197;
	.loc	1 344 15                        // sk05_mlp_gateup.py:344:15
	fma.rn.f32 	%r913, %r341, %r419, %r913;
	fma.rn.f32 	%r912, %r340, %r418, %r912;
	fma.rn.f32 	%r911, %r341, %r417, %r911;
	fma.rn.f32 	%r910, %r340, %r416, %r910;
	fma.rn.f32 	%r917, %r343, %r415, %r917;
	fma.rn.f32 	%r916, %r342, %r414, %r916;
	fma.rn.f32 	%r915, %r343, %r413, %r915;
	fma.rn.f32 	%r914, %r342, %r412, %r914;
	fma.rn.f32 	%r921, %r345, %r411, %r921;
	fma.rn.f32 	%r920, %r344, %r410, %r920;
	fma.rn.f32 	%r919, %r345, %r409, %r919;
	fma.rn.f32 	%r918, %r344, %r408, %r918;
	fma.rn.f32 	%r925, %r347, %r407, %r925;
	fma.rn.f32 	%r924, %r346, %r406, %r924;
	fma.rn.f32 	%r923, %r347, %r405, %r923;
	fma.rn.f32 	%r922, %r346, %r404, %r922;
	fma.rn.f32 	%r929, %r341, %r403, %r929;
	fma.rn.f32 	%r928, %r340, %r402, %r928;
	fma.rn.f32 	%r927, %r341, %r401, %r927;
	fma.rn.f32 	%r926, %r340, %r400, %r926;
	fma.rn.f32 	%r933, %r343, %r399, %r933;
	fma.rn.f32 	%r932, %r342, %r398, %r932;
	fma.rn.f32 	%r931, %r343, %r397, %r931;
	fma.rn.f32 	%r930, %r342, %r396, %r930;
	fma.rn.f32 	%r937, %r345, %r395, %r937;
	fma.rn.f32 	%r936, %r344, %r394, %r936;
	fma.rn.f32 	%r935, %r345, %r393, %r935;
	fma.rn.f32 	%r934, %r344, %r392, %r934;
	fma.rn.f32 	%r941, %r347, %r391, %r941;
	fma.rn.f32 	%r940, %r346, %r390, %r940;
	fma.rn.f32 	%r939, %r347, %r389, %r939;
	fma.rn.f32 	%r938, %r346, %r388, %r938;
	fma.rn.f32 	%r945, %r341, %r387, %r945;
	fma.rn.f32 	%r944, %r340, %r386, %r944;
	fma.rn.f32 	%r943, %r341, %r385, %r943;
	fma.rn.f32 	%r942, %r340, %r384, %r942;
	fma.rn.f32 	%r949, %r343, %r383, %r949;
	fma.rn.f32 	%r948, %r342, %r382, %r948;
	fma.rn.f32 	%r947, %r343, %r381, %r947;
	fma.rn.f32 	%r946, %r342, %r380, %r946;
	fma.rn.f32 	%r953, %r345, %r379, %r953;
	fma.rn.f32 	%r952, %r344, %r378, %r952;
	fma.rn.f32 	%r951, %r345, %r377, %r951;
	fma.rn.f32 	%r950, %r344, %r376, %r950;
	fma.rn.f32 	%r957, %r347, %r375, %r957;
	fma.rn.f32 	%r956, %r346, %r374, %r956;
	fma.rn.f32 	%r955, %r347, %r373, %r955;
	fma.rn.f32 	%r954, %r346, %r372, %r954;
	fma.rn.f32 	%r961, %r341, %r371, %r961;
	fma.rn.f32 	%r960, %r340, %r370, %r960;
	fma.rn.f32 	%r959, %r341, %r369, %r959;
	fma.rn.f32 	%r958, %r340, %r368, %r958;
	fma.rn.f32 	%r965, %r343, %r367, %r965;
	fma.rn.f32 	%r964, %r342, %r366, %r964;
	fma.rn.f32 	%r963, %r343, %r365, %r963;
	fma.rn.f32 	%r962, %r342, %r364, %r962;
	fma.rn.f32 	%r969, %r345, %r363, %r969;
	fma.rn.f32 	%r968, %r344, %r362, %r968;
	fma.rn.f32 	%r967, %r345, %r361, %r967;
	fma.rn.f32 	%r966, %r344, %r360, %r966;
	fma.rn.f32 	%r973, %r347, %r359, %r973;
	fma.rn.f32 	%r972, %r346, %r358, %r972;
	fma.rn.f32 	%r971, %r347, %r357, %r971;
	fma.rn.f32 	%r970, %r346, %r356, %r970;
	.loc	1 345 18                        // sk05_mlp_gateup.py:345:18
	add.s64 	%rd80, %rd27, %rd202;
	add.s64 	%rd81, %rd26, %rd202;
	add.s64 	%rd82, %rd25, %rd202;
	.loc	1 346 18                        // sk05_mlp_gateup.py:346:18
	add.s64 	%rd83, %rd24, %rd202;
	add.s64 	%rd84, %rd23, %rd202;
	add.s64 	%rd85, %rd22, %rd202;
	add.s64 	%rd86, %rd21, %rd202;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	add.s64 	%rd87, %rd20, %rd202;
	add.s32 	%r420, %r907, 1;
	setp.gt.s32 	%p5, %r420, 1;
	selp.b32 	%r907, 0, %r420, %p5;
	.loc	1 344 30                        // sk05_mlp_gateup.py:344:30
	shl.b32 	%r421, %r907, 14;
	bar.sync 	0;
	add.s32 	%r330, %r32, %r421;
	selp.b32 	%r331, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r330 + 0 ], [ %rd80 + 0 ], 0x10, %r331;
	// end inline asm
	add.s32 	%r332, %r330, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r332 + 0 ], [ %rd81 + 0 ], 0x10, %r331;
	// end inline asm
	add.s32 	%r333, %r330, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r333 + 0 ], [ %rd82 + 0 ], 0x10, %r331;
	// end inline asm
	add.s32 	%r334, %r330, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r334 + 0 ], [ %rd83 + 0 ], 0x10, %r331;
	// end inline asm
	cp.async.commit_group;
	.loc	1 344 47                        // sk05_mlp_gateup.py:344:47
	add.s32 	%r335, %r330, 32768;
	// begin inline asm
	cp.async.cg.shared.global [ %r335 + 0 ], [ %rd84 + 0 ], 0x10, %r331;
	// end inline asm
	add.s32 	%r336, %r330, 36864;
	// begin inline asm
	cp.async.cg.shared.global [ %r336 + 0 ], [ %rd85 + 0 ], 0x10, %r331;
	// end inline asm
	add.s32 	%r337, %r330, 40960;
	// begin inline asm
	cp.async.cg.shared.global [ %r337 + 0 ], [ %rd86 + 0 ], 0x10, %r331;
	// end inline asm
	add.s32 	%r338, %r330, 45056;
	// begin inline asm
	cp.async.cg.shared.global [ %r338 + 0 ], [ %rd87 + 0 ], 0x10, %r331;
	// end inline asm
	cp.async.commit_group;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	add.s64 	%rd203, %rd203, 1;
	add.s64 	%rd202, %rd202, 128;
	add.s32 	%r905, %r905, %r31;
	setp.ne.b64 	%p6, %rd19, %rd202;
	@%p6 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_4;
$L__BB0_1:                              // %.._crit_edge_crit_edge
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	shl.b32 	%r909, %r2, 1;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	and.b32 	%r908, %r2, 16;
	mov.b32 	%r910, 0f00000000;
	mov.b32 	%r911, %r910;
	mov.b32 	%r912, %r910;
	mov.b32 	%r913, %r910;
	mov.b32 	%r914, %r910;
	mov.b32 	%r915, %r910;
	mov.b32 	%r916, %r910;
	mov.b32 	%r917, %r910;
	mov.b32 	%r918, %r910;
	mov.b32 	%r919, %r910;
	mov.b32 	%r920, %r910;
	mov.b32 	%r921, %r910;
	mov.b32 	%r922, %r910;
	mov.b32 	%r923, %r910;
	mov.b32 	%r924, %r910;
	mov.b32 	%r925, %r910;
	mov.b32 	%r926, %r910;
	mov.b32 	%r927, %r910;
	mov.b32 	%r928, %r910;
	mov.b32 	%r929, %r910;
	mov.b32 	%r930, %r910;
	mov.b32 	%r931, %r910;
	mov.b32 	%r932, %r910;
	mov.b32 	%r933, %r910;
	mov.b32 	%r934, %r910;
	mov.b32 	%r935, %r910;
	mov.b32 	%r936, %r910;
	mov.b32 	%r937, %r910;
	mov.b32 	%r938, %r910;
	mov.b32 	%r939, %r910;
	mov.b32 	%r940, %r910;
	mov.b32 	%r941, %r910;
	mov.b32 	%r942, %r910;
	mov.b32 	%r943, %r910;
	mov.b32 	%r944, %r910;
	mov.b32 	%r945, %r910;
	mov.b32 	%r946, %r910;
	mov.b32 	%r947, %r910;
	mov.b32 	%r948, %r910;
	mov.b32 	%r949, %r910;
	mov.b32 	%r950, %r910;
	mov.b32 	%r951, %r910;
	mov.b32 	%r952, %r910;
	mov.b32 	%r953, %r910;
	mov.b32 	%r954, %r910;
	mov.b32 	%r955, %r910;
	mov.b32 	%r956, %r910;
	mov.b32 	%r957, %r910;
	mov.b32 	%r958, %r910;
	mov.b32 	%r959, %r910;
	mov.b32 	%r960, %r910;
	mov.b32 	%r961, %r910;
	mov.b32 	%r962, %r910;
	mov.b32 	%r963, %r910;
	mov.b32 	%r964, %r910;
	mov.b32 	%r965, %r910;
	mov.b32 	%r966, %r910;
	mov.b32 	%r967, %r910;
	mov.b32 	%r968, %r910;
	mov.b32 	%r969, %r910;
	mov.b32 	%r970, %r910;
	mov.b32 	%r971, %r910;
	mov.b32 	%r972, %r910;
	mov.b32 	%r973, %r910;
$L__BB0_4:                              // %._crit_edge
	.loc	1 326 45                        // sk05_mlp_gateup.py:326:45
	or.b32 	%r544, %r12, %r904;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r545, %r544, 15;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r546, %r545, %r26;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r547, %r544, 14;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r548, %r547, %r26;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r549, %r544, 13;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r550, %r549, %r26;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r551, %r544, 12;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r552, %r551, %r26;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r553, %r544, 11;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r554, %r553, %r26;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r555, %r544, 10;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r556, %r555, %r26;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r557, %r544, 9;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r558, %r557, %r26;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r559, %r544, 8;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r560, %r559, %r26;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r561, %r544, 7;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r562, %r561, %r26;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r563, %r544, 6;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r564, %r563, %r26;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r565, %r544, 5;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r566, %r565, %r26;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r567, %r544, 4;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r568, %r567, %r26;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r569, %r544, 3;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r570, %r569, %r26;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r571, %r544, 2;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r572, %r571, %r26;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r573, %r544, 1;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r574, %r573, %r26;
	rem.s32 	%r575, %r544, %r26;
	.loc	1 326 45                        // sk05_mlp_gateup.py:326:45
	shl.b32 	%r576, %r7, 3;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r577, %r12, %r576;
	.loc	1 326 45                        // sk05_mlp_gateup.py:326:45
	and.b32 	%r578, %r2, 128;
	shr.u32 	%r579, %r578, 3;
	shr.u32 	%r580, %r2, 2;
	bfe.u32 	%r581, %r2, 2, 3;
	or.b32 	%r582, %r579, %r581;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r583, %r582, %r1;
	or.b32 	%r584, %r583, 104;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r585, %r584, %r25;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r586, %r583, 96;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r587, %r586, %r25;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r588, %r583, 72;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r589, %r588, %r25;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r590, %r583, 64;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r591, %r590, %r25;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r592, %r583, 40;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r593, %r592, %r25;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r594, %r583, 32;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r595, %r594, %r25;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r596, %r583, 8;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r597, %r596, %r25;
	rem.s32 	%r598, %r583, %r25;
	.loc	1 326 45                        // sk05_mlp_gateup.py:326:45
	shr.u32 	%r599, %r2, 4;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r600, %r599, %r1;
	or.b32 	%r601, %r600, 112;
	.loc	1 326 45                        // sk05_mlp_gateup.py:326:45
	bfe.u32 	%r602, %r2, 4, 4;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r603, %r602, %r1;
	or.b32 	%r604, %r603, 96;
	or.b32 	%r605, %r603, 80;
	or.b32 	%r606, %r603, 64;
	or.b32 	%r607, %r600, 48;
	or.b32 	%r608, %r603, 32;
	or.b32 	%r609, %r603, 16;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 348 38                        // sk05_mlp_gateup.py:348:38
	mad.wide.s32 	%rd89, %r598, 4, %rd32;
	mad.wide.s32 	%rd90, %r597, 4, %rd32;
	mad.wide.s32 	%rd91, %r595, 4, %rd32;
	mad.wide.s32 	%rd92, %r593, 4, %rd32;
	mad.wide.s32 	%rd93, %r591, 4, %rd32;
	mad.wide.s32 	%rd94, %r589, 4, %rd32;
	mad.wide.s32 	%rd95, %r587, 4, %rd32;
	mad.wide.s32 	%rd96, %r585, 4, %rd32;
	.loc	1 348 24                        // sk05_mlp_gateup.py:348:24
	// begin inline asm
	mov.u32 %r422, 0x0;
	ld.global.b32 { %r422 }, [ %rd89 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r423, 0x0;
	ld.global.b32 { %r423 }, [ %rd90 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r424, 0x0;
	ld.global.b32 { %r424 }, [ %rd91 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r425, 0x0;
	ld.global.b32 { %r425 }, [ %rd92 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r426, 0x0;
	ld.global.b32 { %r426 }, [ %rd93 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r427, 0x0;
	ld.global.b32 { %r427 }, [ %rd94 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r428, 0x0;
	ld.global.b32 { %r428 }, [ %rd95 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r429, 0x0;
	ld.global.b32 { %r429 }, [ %rd96 + 0 ];
	// end inline asm
	.loc	1 349 38                        // sk05_mlp_gateup.py:349:38
	mad.wide.s32 	%rd97, %r13, 4, %rd33;
	mad.wide.s32 	%rd98, %r14, 4, %rd33;
	mad.wide.s32 	%rd99, %r15, 4, %rd33;
	mad.wide.s32 	%rd100, %r16, 4, %rd33;
	.loc	1 349 24                        // sk05_mlp_gateup.py:349:24
	// begin inline asm
	mov.u32 %r430, 0x0;
	mov.u32 %r431, 0x0;
	ld.global.v2.b32 { %r430, %r431 }, [ %rd97 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r432, 0x0;
	mov.u32 %r433, 0x0;
	ld.global.v2.b32 { %r432, %r433 }, [ %rd98 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r434, 0x0;
	mov.u32 %r435, 0x0;
	ld.global.v2.b32 { %r434, %r435 }, [ %rd99 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r436, 0x0;
	mov.u32 %r437, 0x0;
	ld.global.v2.b32 { %r436, %r437 }, [ %rd100 + 0 ];
	// end inline asm
	.loc	1 350 49                        // sk05_mlp_gateup.py:350:49
	mul.lo.s32 	%r610, %r8, %r29;
	mul.lo.s32 	%r611, %r9, %r29;
	mul.lo.s32 	%r612, %r10, %r29;
	mul.lo.s32 	%r613, %r11, %r29;
	.loc	1 350 31                        // sk05_mlp_gateup.py:350:31
	mad.wide.s32 	%rd173, %r610, 2, %rd31;
	mad.wide.s32 	%rd174, %r611, 2, %rd31;
	mad.wide.s32 	%rd175, %r612, 2, %rd31;
	mad.wide.s32 	%rd176, %r613, 2, %rd31;
	.loc	1 350 82                        // sk05_mlp_gateup.py:350:82
	mul.lo.s32 	%r614, %r575, %r30;
	mul.lo.s32 	%r615, %r574, %r30;
	mul.lo.s32 	%r616, %r572, %r30;
	mul.lo.s32 	%r617, %r570, %r30;
	mul.lo.s32 	%r618, %r568, %r30;
	mul.lo.s32 	%r619, %r566, %r30;
	mul.lo.s32 	%r620, %r564, %r30;
	mul.lo.s32 	%r621, %r562, %r30;
	mul.lo.s32 	%r622, %r560, %r30;
	mul.lo.s32 	%r623, %r558, %r30;
	mul.lo.s32 	%r624, %r556, %r30;
	mul.lo.s32 	%r625, %r554, %r30;
	mul.lo.s32 	%r626, %r552, %r30;
	mul.lo.s32 	%r627, %r550, %r30;
	mul.lo.s32 	%r628, %r548, %r30;
	mul.lo.s32 	%r629, %r546, %r30;
	.loc	1 350 64                        // sk05_mlp_gateup.py:350:64
	mul.wide.s32 	%rd177, %r614, 2;
	add.s64 	%rd101, %rd173, %rd177;
	mul.wide.s32 	%rd178, %r615, 2;
	add.s64 	%rd102, %rd173, %rd178;
	mul.wide.s32 	%rd179, %r616, 2;
	add.s64 	%rd103, %rd173, %rd179;
	mul.wide.s32 	%rd180, %r617, 2;
	add.s64 	%rd104, %rd173, %rd180;
	mul.wide.s32 	%rd181, %r618, 2;
	add.s64 	%rd105, %rd173, %rd181;
	mul.wide.s32 	%rd182, %r619, 2;
	add.s64 	%rd106, %rd173, %rd182;
	mul.wide.s32 	%rd183, %r620, 2;
	add.s64 	%rd107, %rd173, %rd183;
	mul.wide.s32 	%rd184, %r621, 2;
	add.s64 	%rd108, %rd173, %rd184;
	mul.wide.s32 	%rd185, %r622, 2;
	add.s64 	%rd109, %rd173, %rd185;
	mul.wide.s32 	%rd186, %r623, 2;
	add.s64 	%rd110, %rd173, %rd186;
	mul.wide.s32 	%rd187, %r624, 2;
	add.s64 	%rd111, %rd173, %rd187;
	mul.wide.s32 	%rd188, %r625, 2;
	add.s64 	%rd112, %rd173, %rd188;
	mul.wide.s32 	%rd189, %r626, 2;
	add.s64 	%rd113, %rd173, %rd189;
	mul.wide.s32 	%rd190, %r627, 2;
	add.s64 	%rd114, %rd173, %rd190;
	mul.wide.s32 	%rd191, %r628, 2;
	add.s64 	%rd115, %rd173, %rd191;
	mul.wide.s32 	%rd192, %r629, 2;
	add.s64 	%rd116, %rd173, %rd192;
	add.s64 	%rd117, %rd174, %rd177;
	add.s64 	%rd118, %rd174, %rd178;
	add.s64 	%rd119, %rd174, %rd179;
	add.s64 	%rd120, %rd174, %rd180;
	add.s64 	%rd121, %rd174, %rd181;
	add.s64 	%rd122, %rd174, %rd182;
	add.s64 	%rd123, %rd174, %rd183;
	add.s64 	%rd124, %rd174, %rd184;
	add.s64 	%rd125, %rd174, %rd185;
	add.s64 	%rd126, %rd174, %rd186;
	add.s64 	%rd127, %rd174, %rd187;
	add.s64 	%rd128, %rd174, %rd188;
	add.s64 	%rd129, %rd174, %rd189;
	add.s64 	%rd130, %rd174, %rd190;
	add.s64 	%rd131, %rd174, %rd191;
	add.s64 	%rd132, %rd174, %rd192;
	add.s64 	%rd133, %rd175, %rd177;
	add.s64 	%rd134, %rd175, %rd178;
	add.s64 	%rd135, %rd175, %rd179;
	add.s64 	%rd136, %rd175, %rd180;
	add.s64 	%rd137, %rd175, %rd181;
	add.s64 	%rd138, %rd175, %rd182;
	add.s64 	%rd139, %rd175, %rd183;
	add.s64 	%rd140, %rd175, %rd184;
	add.s64 	%rd141, %rd175, %rd185;
	add.s64 	%rd142, %rd175, %rd186;
	add.s64 	%rd143, %rd175, %rd187;
	add.s64 	%rd144, %rd175, %rd188;
	add.s64 	%rd145, %rd175, %rd189;
	add.s64 	%rd146, %rd175, %rd190;
	add.s64 	%rd147, %rd175, %rd191;
	add.s64 	%rd148, %rd175, %rd192;
	add.s64 	%rd149, %rd176, %rd177;
	add.s64 	%rd150, %rd176, %rd178;
	add.s64 	%rd151, %rd176, %rd179;
	add.s64 	%rd152, %rd176, %rd180;
	add.s64 	%rd153, %rd176, %rd181;
	add.s64 	%rd154, %rd176, %rd182;
	add.s64 	%rd155, %rd176, %rd183;
	add.s64 	%rd156, %rd176, %rd184;
	add.s64 	%rd157, %rd176, %rd185;
	add.s64 	%rd158, %rd176, %rd186;
	add.s64 	%rd159, %rd176, %rd187;
	add.s64 	%rd160, %rd176, %rd188;
	add.s64 	%rd161, %rd176, %rd189;
	add.s64 	%rd162, %rd176, %rd190;
	add.s64 	%rd163, %rd176, %rd191;
	add.s64 	%rd164, %rd176, %rd192;
	.loc	1 350 19                        // sk05_mlp_gateup.py:350:19
	// begin inline asm
	mov.u16 %rs1, 0x0;
	ld.global.b16 { %rs1 }, [ %rd101 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs2, 0x0;
	ld.global.b16 { %rs2 }, [ %rd102 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs3, 0x0;
	ld.global.b16 { %rs3 }, [ %rd103 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs4, 0x0;
	ld.global.b16 { %rs4 }, [ %rd104 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs5, 0x0;
	ld.global.b16 { %rs5 }, [ %rd105 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs6, 0x0;
	ld.global.b16 { %rs6 }, [ %rd106 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs7, 0x0;
	ld.global.b16 { %rs7 }, [ %rd107 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs8, 0x0;
	ld.global.b16 { %rs8 }, [ %rd108 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs9, 0x0;
	ld.global.b16 { %rs9 }, [ %rd109 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs10, 0x0;
	ld.global.b16 { %rs10 }, [ %rd110 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs11, 0x0;
	ld.global.b16 { %rs11 }, [ %rd111 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs12, 0x0;
	ld.global.b16 { %rs12 }, [ %rd112 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs13, 0x0;
	ld.global.b16 { %rs13 }, [ %rd113 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs14, 0x0;
	ld.global.b16 { %rs14 }, [ %rd114 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs15, 0x0;
	ld.global.b16 { %rs15 }, [ %rd115 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs16, 0x0;
	ld.global.b16 { %rs16 }, [ %rd116 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs17, 0x0;
	ld.global.b16 { %rs17 }, [ %rd117 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs18, 0x0;
	ld.global.b16 { %rs18 }, [ %rd118 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs19, 0x0;
	ld.global.b16 { %rs19 }, [ %rd119 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs20, 0x0;
	ld.global.b16 { %rs20 }, [ %rd120 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs21, 0x0;
	ld.global.b16 { %rs21 }, [ %rd121 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs22, 0x0;
	ld.global.b16 { %rs22 }, [ %rd122 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs23, 0x0;
	ld.global.b16 { %rs23 }, [ %rd123 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs24, 0x0;
	ld.global.b16 { %rs24 }, [ %rd124 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs25, 0x0;
	ld.global.b16 { %rs25 }, [ %rd125 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs26, 0x0;
	ld.global.b16 { %rs26 }, [ %rd126 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs27, 0x0;
	ld.global.b16 { %rs27 }, [ %rd127 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs28, 0x0;
	ld.global.b16 { %rs28 }, [ %rd128 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs29, 0x0;
	ld.global.b16 { %rs29 }, [ %rd129 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs30, 0x0;
	ld.global.b16 { %rs30 }, [ %rd130 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs31, 0x0;
	ld.global.b16 { %rs31 }, [ %rd131 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs32, 0x0;
	ld.global.b16 { %rs32 }, [ %rd132 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs33, 0x0;
	ld.global.b16 { %rs33 }, [ %rd133 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs34, 0x0;
	ld.global.b16 { %rs34 }, [ %rd134 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs35, 0x0;
	ld.global.b16 { %rs35 }, [ %rd135 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs36, 0x0;
	ld.global.b16 { %rs36 }, [ %rd136 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs37, 0x0;
	ld.global.b16 { %rs37 }, [ %rd137 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs38, 0x0;
	ld.global.b16 { %rs38 }, [ %rd138 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs39, 0x0;
	ld.global.b16 { %rs39 }, [ %rd139 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs40, 0x0;
	ld.global.b16 { %rs40 }, [ %rd140 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs41, 0x0;
	ld.global.b16 { %rs41 }, [ %rd141 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs42, 0x0;
	ld.global.b16 { %rs42 }, [ %rd142 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs43, 0x0;
	ld.global.b16 { %rs43 }, [ %rd143 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs44, 0x0;
	ld.global.b16 { %rs44 }, [ %rd144 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs45, 0x0;
	ld.global.b16 { %rs45 }, [ %rd145 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs46, 0x0;
	ld.global.b16 { %rs46 }, [ %rd146 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs47, 0x0;
	ld.global.b16 { %rs47 }, [ %rd147 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs48, 0x0;
	ld.global.b16 { %rs48 }, [ %rd148 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs49, 0x0;
	ld.global.b16 { %rs49 }, [ %rd149 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs50, 0x0;
	ld.global.b16 { %rs50 }, [ %rd150 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs51, 0x0;
	ld.global.b16 { %rs51 }, [ %rd151 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs52, 0x0;
	ld.global.b16 { %rs52 }, [ %rd152 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs53, 0x0;
	ld.global.b16 { %rs53 }, [ %rd153 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs54, 0x0;
	ld.global.b16 { %rs54 }, [ %rd154 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs55, 0x0;
	ld.global.b16 { %rs55 }, [ %rd155 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs56, 0x0;
	ld.global.b16 { %rs56 }, [ %rd156 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs57, 0x0;
	ld.global.b16 { %rs57 }, [ %rd157 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs58, 0x0;
	ld.global.b16 { %rs58 }, [ %rd158 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs59, 0x0;
	ld.global.b16 { %rs59 }, [ %rd159 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs60, 0x0;
	ld.global.b16 { %rs60 }, [ %rd160 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs61, 0x0;
	ld.global.b16 { %rs61 }, [ %rd161 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs62, 0x0;
	ld.global.b16 { %rs62 }, [ %rd162 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs63, 0x0;
	ld.global.b16 { %rs63 }, [ %rd163 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs64, 0x0;
	ld.global.b16 { %rs64 }, [ %rd164 + 0 ];
	// end inline asm
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	shl.b32 	%r630, %r18, 7;
	shl.b32 	%r631, %r3, 1;
	or.b32 	%r632, %r630, %r904;
	xor.b32 	%r633, %r632, %r631;
	add.s32 	%r438, %r149, %r633;
	mov.b32 	%r439, {%rs1, %rs2};
	mov.b32 	%r440, {%rs3, %rs4};
	mov.b32 	%r441, {%rs5, %rs6};
	mov.b32 	%r442, {%rs7, %rs8};
	// begin inline asm
	st.shared.v4.b32 [ %r438 + 0 ], { %r439, %r440, %r441, %r442 };
	// end inline asm
	add.s32 	%r443, %r438, 512;
	mov.b32 	%r444, {%rs9, %rs10};
	mov.b32 	%r445, {%rs11, %rs12};
	mov.b32 	%r446, {%rs13, %rs14};
	mov.b32 	%r447, {%rs15, %rs16};
	// begin inline asm
	st.shared.v4.b32 [ %r443 + 0 ], { %r444, %r445, %r446, %r447 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r634, %r6, 10;
	and.b32 	%r635, %r17, 752;
	and.b32 	%r636, %r909, 288;
	and.b32 	%r637, %r580, 16;
	xor.b32 	%r638, %r635, %r636;
	xor.b32 	%r639, %r638, %r637;
	or.b32 	%r640, %r639, %r634;
	add.s32 	%r641, %r149, %r640;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r642, %r643, %r644, %r645}, [%r641];
	xor.b32 	%r646, %r640, 64;
	add.s32 	%r647, %r149, %r646;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r648, %r649, %r650, %r651}, [%r647];
	bar.sync 	0;
	mov.b32 	%r448, {%rs17, %rs18};
	mov.b32 	%r449, {%rs19, %rs20};
	mov.b32 	%r450, {%rs21, %rs22};
	mov.b32 	%r451, {%rs23, %rs24};
	// begin inline asm
	st.shared.v4.b32 [ %r438 + 0 ], { %r448, %r449, %r450, %r451 };
	// end inline asm
	mov.b32 	%r452, {%rs25, %rs26};
	mov.b32 	%r453, {%rs27, %rs28};
	mov.b32 	%r454, {%rs29, %rs30};
	mov.b32 	%r455, {%rs31, %rs32};
	// begin inline asm
	st.shared.v4.b32 [ %r443 + 0 ], { %r452, %r453, %r454, %r455 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r652, %r653, %r654, %r655}, [%r641];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r656, %r657, %r658, %r659}, [%r647];
	bar.sync 	0;
	mov.b32 	%r456, {%rs33, %rs34};
	mov.b32 	%r457, {%rs35, %rs36};
	mov.b32 	%r458, {%rs37, %rs38};
	mov.b32 	%r459, {%rs39, %rs40};
	// begin inline asm
	st.shared.v4.b32 [ %r438 + 0 ], { %r456, %r457, %r458, %r459 };
	// end inline asm
	mov.b32 	%r460, {%rs41, %rs42};
	mov.b32 	%r461, {%rs43, %rs44};
	mov.b32 	%r462, {%rs45, %rs46};
	mov.b32 	%r463, {%rs47, %rs48};
	// begin inline asm
	st.shared.v4.b32 [ %r443 + 0 ], { %r460, %r461, %r462, %r463 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r660, %r661, %r662, %r663}, [%r641];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r664, %r665, %r666, %r667}, [%r647];
	bar.sync 	0;
	mov.b32 	%r464, {%rs49, %rs50};
	mov.b32 	%r465, {%rs51, %rs52};
	mov.b32 	%r466, {%rs53, %rs54};
	mov.b32 	%r467, {%rs55, %rs56};
	// begin inline asm
	st.shared.v4.b32 [ %r438 + 0 ], { %r464, %r465, %r466, %r467 };
	// end inline asm
	mov.b32 	%r468, {%rs57, %rs58};
	mov.b32 	%r469, {%rs59, %rs60};
	mov.b32 	%r470, {%rs61, %rs62};
	mov.b32 	%r471, {%rs63, %rs64};
	// begin inline asm
	st.shared.v4.b32 [ %r443 + 0 ], { %r468, %r469, %r470, %r471 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r668, %r669, %r670, %r671}, [%r641];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r672, %r673, %r674, %r675}, [%r647];
	.loc	1 357 31                        // sk05_mlp_gateup.py:357:31
	setp.lt.s32 	%p15, %r603, %r25;
	setp.lt.s32 	%p16, %r609, %r25;
	setp.lt.s32 	%p17, %r608, %r25;
	setp.lt.s32 	%p18, %r607, %r25;
	setp.lt.s32 	%p19, %r606, %r25;
	setp.lt.s32 	%p20, %r605, %r25;
	setp.lt.s32 	%p21, %r604, %r25;
	setp.lt.s32 	%p22, %r601, %r25;
	.loc	1 357 54                        // sk05_mlp_gateup.py:357:54
	setp.lt.s32 	%p23, %r577, %r26;
	.loc	1 357 37                        // sk05_mlp_gateup.py:357:37
	and.pred 	%p7, %p15, %p23;
	and.pred 	%p8, %p16, %p23;
	and.pred 	%p9, %p17, %p23;
	and.pred 	%p10, %p18, %p23;
	and.pred 	%p11, %p19, %p23;
	and.pred 	%p12, %p20, %p23;
	and.pred 	%p13, %p21, %p23;
	and.pred 	%p14, %p22, %p23;
	.loc	1 355 35                        // sk05_mlp_gateup.py:355:35
	mul.lo.s32 	%r676, %r603, %r28;
	mul.lo.s32 	%r677, %r609, %r28;
	mul.lo.s32 	%r678, %r608, %r28;
	mul.lo.s32 	%r679, %r607, %r28;
	mul.lo.s32 	%r680, %r606, %r28;
	mul.lo.s32 	%r681, %r605, %r28;
	mul.lo.s32 	%r682, %r604, %r28;
	mul.lo.s32 	%r683, %r601, %r28;
	.loc	1 355 18                        // sk05_mlp_gateup.py:355:18
	mad.wide.s32 	%rd193, %r676, 2, %rd30;
	mad.wide.s32 	%rd194, %r677, 2, %rd30;
	mad.wide.s32 	%rd195, %r678, 2, %rd30;
	mad.wide.s32 	%rd196, %r679, 2, %rd30;
	mad.wide.s32 	%rd197, %r680, 2, %rd30;
	mad.wide.s32 	%rd198, %r681, 2, %rd30;
	mad.wide.s32 	%rd199, %r682, 2, %rd30;
	mad.wide.s32 	%rd200, %r683, 2, %rd30;
	.loc	1 355 50                        // sk05_mlp_gateup.py:355:50
	mul.wide.s32 	%rd201, %r577, 2;
	add.s64 	%rd165, %rd193, %rd201;
	add.s64 	%rd166, %rd194, %rd201;
	add.s64 	%rd167, %rd195, %rd201;
	add.s64 	%rd168, %rd196, %rd201;
	add.s64 	%rd169, %rd197, %rd201;
	add.s64 	%rd170, %rd198, %rd201;
	add.s64 	%rd171, %rd199, %rd201;
	add.s64 	%rd172, %rd200, %rd201;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r684, %r959, %r428;
	mul.f32 	%r685, %r958, %r428;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs65, %rs66}, %r668;
	cvt.f32.bf16 	%r686, %rs66;
	cvt.f32.bf16 	%r687, %rs65;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r688, %r685, %r430, %r687;
	fma.rn.f32 	%r689, %r684, %r431, %r686;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r690, %r911, %r422;
	mul.f32 	%r691, %r910, %r422;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs67, %rs68}, %r642;
	cvt.f32.bf16 	%r692, %rs68;
	cvt.f32.bf16 	%r693, %rs67;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r694, %r691, %r430, %r693;
	fma.rn.f32 	%r695, %r690, %r431, %r692;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r473, %r695, %r694;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r696, %r913, %r423;
	mul.f32 	%r697, %r912, %r423;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs69, %rs70}, %r643;
	cvt.f32.bf16 	%r698, %rs70;
	cvt.f32.bf16 	%r699, %rs69;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r700, %r697, %r430, %r699;
	fma.rn.f32 	%r701, %r696, %r431, %r698;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r478, %r701, %r700;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r702, %r927, %r424;
	mul.f32 	%r703, %r926, %r424;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs71, %rs72}, %r652;
	cvt.f32.bf16 	%r704, %rs72;
	cvt.f32.bf16 	%r705, %rs71;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r706, %r703, %r430, %r705;
	fma.rn.f32 	%r707, %r702, %r431, %r704;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r474, %r707, %r706;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r708, %r929, %r425;
	mul.f32 	%r709, %r928, %r425;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs73, %rs74}, %r653;
	cvt.f32.bf16 	%r710, %rs74;
	cvt.f32.bf16 	%r711, %rs73;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r712, %r709, %r430, %r711;
	fma.rn.f32 	%r713, %r708, %r431, %r710;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r479, %r713, %r712;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r714, %r943, %r426;
	mul.f32 	%r715, %r942, %r426;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs75, %rs76}, %r660;
	cvt.f32.bf16 	%r716, %rs76;
	cvt.f32.bf16 	%r717, %rs75;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r718, %r715, %r430, %r717;
	fma.rn.f32 	%r719, %r714, %r431, %r716;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r475, %r719, %r718;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r720, %r945, %r427;
	mul.f32 	%r721, %r944, %r427;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs77, %rs78}, %r661;
	cvt.f32.bf16 	%r722, %rs78;
	cvt.f32.bf16 	%r723, %rs77;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r724, %r721, %r430, %r723;
	fma.rn.f32 	%r725, %r720, %r431, %r722;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r480, %r725, %r724;
	cvt.rn.bf16x2.f32 	%r476, %r689, %r688;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r726, %r961, %r429;
	mul.f32 	%r727, %r960, %r429;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs79, %rs80}, %r669;
	cvt.f32.bf16 	%r728, %rs80;
	cvt.f32.bf16 	%r729, %rs79;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r730, %r727, %r430, %r729;
	fma.rn.f32 	%r731, %r726, %r431, %r728;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r481, %r731, %r730;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r732, %r963, %r428;
	mul.f32 	%r733, %r962, %r428;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs81, %rs82}, %r670;
	cvt.f32.bf16 	%r734, %rs82;
	cvt.f32.bf16 	%r735, %rs81;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r736, %r733, %r432, %r735;
	fma.rn.f32 	%r737, %r732, %r433, %r734;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r738, %r915, %r422;
	mul.f32 	%r739, %r914, %r422;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs83, %rs84}, %r644;
	cvt.f32.bf16 	%r740, %rs84;
	cvt.f32.bf16 	%r741, %rs83;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r742, %r739, %r432, %r741;
	fma.rn.f32 	%r743, %r738, %r433, %r740;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r493, %r743, %r742;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r744, %r917, %r423;
	mul.f32 	%r745, %r916, %r423;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs85, %rs86}, %r645;
	cvt.f32.bf16 	%r746, %rs86;
	cvt.f32.bf16 	%r747, %rs85;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r748, %r745, %r432, %r747;
	fma.rn.f32 	%r749, %r744, %r433, %r746;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r498, %r749, %r748;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r750, %r931, %r424;
	mul.f32 	%r751, %r930, %r424;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs87, %rs88}, %r654;
	cvt.f32.bf16 	%r752, %rs88;
	cvt.f32.bf16 	%r753, %rs87;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r754, %r751, %r432, %r753;
	fma.rn.f32 	%r755, %r750, %r433, %r752;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r494, %r755, %r754;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r756, %r933, %r425;
	mul.f32 	%r757, %r932, %r425;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs89, %rs90}, %r655;
	cvt.f32.bf16 	%r758, %rs90;
	cvt.f32.bf16 	%r759, %rs89;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r760, %r757, %r432, %r759;
	fma.rn.f32 	%r761, %r756, %r433, %r758;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r499, %r761, %r760;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r762, %r947, %r426;
	mul.f32 	%r763, %r946, %r426;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs91, %rs92}, %r662;
	cvt.f32.bf16 	%r764, %rs92;
	cvt.f32.bf16 	%r765, %rs91;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r766, %r763, %r432, %r765;
	fma.rn.f32 	%r767, %r762, %r433, %r764;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r495, %r767, %r766;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r768, %r949, %r427;
	mul.f32 	%r769, %r948, %r427;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs93, %rs94}, %r663;
	cvt.f32.bf16 	%r770, %rs94;
	cvt.f32.bf16 	%r771, %rs93;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r772, %r769, %r432, %r771;
	fma.rn.f32 	%r773, %r768, %r433, %r770;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r500, %r773, %r772;
	cvt.rn.bf16x2.f32 	%r496, %r737, %r736;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r774, %r965, %r429;
	mul.f32 	%r775, %r964, %r429;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs95, %rs96}, %r671;
	cvt.f32.bf16 	%r776, %rs96;
	cvt.f32.bf16 	%r777, %rs95;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r778, %r775, %r432, %r777;
	fma.rn.f32 	%r779, %r774, %r433, %r776;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r501, %r779, %r778;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r780, %r967, %r428;
	mul.f32 	%r781, %r966, %r428;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs97, %rs98}, %r672;
	cvt.f32.bf16 	%r782, %rs98;
	cvt.f32.bf16 	%r783, %rs97;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r784, %r781, %r434, %r783;
	fma.rn.f32 	%r785, %r780, %r435, %r782;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r786, %r919, %r422;
	mul.f32 	%r787, %r918, %r422;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs99, %rs100}, %r648;
	cvt.f32.bf16 	%r788, %rs100;
	cvt.f32.bf16 	%r789, %rs99;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r790, %r787, %r434, %r789;
	fma.rn.f32 	%r791, %r786, %r435, %r788;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r483, %r791, %r790;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r792, %r921, %r423;
	mul.f32 	%r793, %r920, %r423;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs101, %rs102}, %r649;
	cvt.f32.bf16 	%r794, %rs102;
	cvt.f32.bf16 	%r795, %rs101;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r796, %r793, %r434, %r795;
	fma.rn.f32 	%r797, %r792, %r435, %r794;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r488, %r797, %r796;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r798, %r935, %r424;
	mul.f32 	%r799, %r934, %r424;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs103, %rs104}, %r656;
	cvt.f32.bf16 	%r800, %rs104;
	cvt.f32.bf16 	%r801, %rs103;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r802, %r799, %r434, %r801;
	fma.rn.f32 	%r803, %r798, %r435, %r800;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r484, %r803, %r802;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r804, %r937, %r425;
	mul.f32 	%r805, %r936, %r425;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs105, %rs106}, %r657;
	cvt.f32.bf16 	%r806, %rs106;
	cvt.f32.bf16 	%r807, %rs105;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r808, %r805, %r434, %r807;
	fma.rn.f32 	%r809, %r804, %r435, %r806;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r489, %r809, %r808;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r810, %r951, %r426;
	mul.f32 	%r811, %r950, %r426;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs107, %rs108}, %r664;
	cvt.f32.bf16 	%r812, %rs108;
	cvt.f32.bf16 	%r813, %rs107;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r814, %r811, %r434, %r813;
	fma.rn.f32 	%r815, %r810, %r435, %r812;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r485, %r815, %r814;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r816, %r953, %r427;
	mul.f32 	%r817, %r952, %r427;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs109, %rs110}, %r665;
	cvt.f32.bf16 	%r818, %rs110;
	cvt.f32.bf16 	%r819, %rs109;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r820, %r817, %r434, %r819;
	fma.rn.f32 	%r821, %r816, %r435, %r818;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r490, %r821, %r820;
	cvt.rn.bf16x2.f32 	%r486, %r785, %r784;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r822, %r969, %r429;
	mul.f32 	%r823, %r968, %r429;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs111, %rs112}, %r673;
	cvt.f32.bf16 	%r824, %rs112;
	cvt.f32.bf16 	%r825, %rs111;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r826, %r823, %r434, %r825;
	fma.rn.f32 	%r827, %r822, %r435, %r824;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r491, %r827, %r826;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r828, %r971, %r428;
	mul.f32 	%r829, %r970, %r428;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs113, %rs114}, %r674;
	cvt.f32.bf16 	%r830, %rs114;
	cvt.f32.bf16 	%r831, %rs113;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r832, %r829, %r436, %r831;
	fma.rn.f32 	%r833, %r828, %r437, %r830;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r834, %r923, %r422;
	mul.f32 	%r835, %r922, %r422;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs115, %rs116}, %r650;
	cvt.f32.bf16 	%r836, %rs116;
	cvt.f32.bf16 	%r837, %rs115;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r838, %r835, %r436, %r837;
	fma.rn.f32 	%r839, %r834, %r437, %r836;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r503, %r839, %r838;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r840, %r925, %r423;
	mul.f32 	%r841, %r924, %r423;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs117, %rs118}, %r651;
	cvt.f32.bf16 	%r842, %rs118;
	cvt.f32.bf16 	%r843, %rs117;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r844, %r841, %r436, %r843;
	fma.rn.f32 	%r845, %r840, %r437, %r842;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r508, %r845, %r844;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r846, %r939, %r424;
	mul.f32 	%r847, %r938, %r424;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs119, %rs120}, %r658;
	cvt.f32.bf16 	%r848, %rs120;
	cvt.f32.bf16 	%r849, %rs119;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r850, %r847, %r436, %r849;
	fma.rn.f32 	%r851, %r846, %r437, %r848;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r504, %r851, %r850;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r852, %r941, %r425;
	mul.f32 	%r853, %r940, %r425;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs121, %rs122}, %r659;
	cvt.f32.bf16 	%r854, %rs122;
	cvt.f32.bf16 	%r855, %rs121;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r856, %r853, %r436, %r855;
	fma.rn.f32 	%r857, %r852, %r437, %r854;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r509, %r857, %r856;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r858, %r955, %r426;
	mul.f32 	%r859, %r954, %r426;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs123, %rs124}, %r666;
	cvt.f32.bf16 	%r860, %rs124;
	cvt.f32.bf16 	%r861, %rs123;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r862, %r859, %r436, %r861;
	fma.rn.f32 	%r863, %r858, %r437, %r860;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r505, %r863, %r862;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r864, %r957, %r427;
	mul.f32 	%r865, %r956, %r427;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs125, %rs126}, %r667;
	cvt.f32.bf16 	%r866, %rs126;
	cvt.f32.bf16 	%r867, %rs125;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r868, %r865, %r436, %r867;
	fma.rn.f32 	%r869, %r864, %r437, %r866;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r510, %r869, %r868;
	cvt.rn.bf16x2.f32 	%r506, %r833, %r832;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r870, %r973, %r429;
	mul.f32 	%r871, %r972, %r429;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs127, %rs128}, %r675;
	cvt.f32.bf16 	%r872, %rs128;
	cvt.f32.bf16 	%r873, %rs127;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r874, %r871, %r436, %r873;
	fma.rn.f32 	%r875, %r870, %r437, %r872;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r511, %r875, %r874;
	bar.sync 	0;
	shl.b32 	%r876, %r4, 13;
	shl.b32 	%r877, %r4, 5;
	and.b32 	%r878, %r17, 384;
	shr.u32 	%r879, %r5, 1;
	bfe.s32 	%r880, %r2, 2, 1;
	and.b32 	%r881, %r880, 4112;
	shl.b32 	%r882, %r578, 3;
	or.b32 	%r883, %r876, %r882;
	or.b32 	%r884, %r877, %r878;
	xor.b32 	%r885, %r881, %r879;
	xor.b32 	%r886, %r885, %r884;
	or.b32 	%r887, %r886, %r883;
	add.s32 	%r472, %r149, %r887;
	// begin inline asm
	st.shared.v4.b32 [ %r472 + 0 ], { %r473, %r474, %r475, %r476 };
	// end inline asm
	add.s32 	%r477, %r472, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r477 + 0 ], { %r478, %r479, %r480, %r481 };
	// end inline asm
	add.s32 	%r482, %r472, 2048;
	// begin inline asm
	st.shared.v4.b32 [ %r482 + 0 ], { %r483, %r484, %r485, %r486 };
	// end inline asm
	add.s32 	%r487, %r472, 2560;
	// begin inline asm
	st.shared.v4.b32 [ %r487 + 0 ], { %r488, %r489, %r490, %r491 };
	// end inline asm
	xor.b32 	%r888, %r887, 64;
	add.s32 	%r492, %r149, %r888;
	// begin inline asm
	st.shared.v4.b32 [ %r492 + 0 ], { %r493, %r494, %r495, %r496 };
	// end inline asm
	add.s32 	%r497, %r492, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r497 + 0 ], { %r498, %r499, %r500, %r501 };
	// end inline asm
	add.s32 	%r502, %r492, 2048;
	// begin inline asm
	st.shared.v4.b32 [ %r502 + 0 ], { %r503, %r504, %r505, %r506 };
	// end inline asm
	add.s32 	%r507, %r492, 2560;
	// begin inline asm
	st.shared.v4.b32 [ %r507 + 0 ], { %r508, %r509, %r510, %r511 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r889, %r2, 2;
	and.b32 	%r890, %r889, 896;
	shl.b32 	%r891, %r2, 8;
	and.b32 	%r892, %r891, 2048;
	setp.eq.b32 	%p24, %r908, 0;
	selp.b32 	%r893, 0, 4112, %p24;
	or.b32 	%r894, %r904, %r890;
	xor.b32 	%r895, %r894, %r893;
	or.b32 	%r896, %r895, %r892;
	add.s32 	%r897, %r149, %r896;
	ld.shared.v4.b32 	{%r512, %r520, %r528, %r536}, [%r897];
	ld.shared.v4.b32 	{%r516, %r524, %r532, %r540}, [%r897+1024];
	xor.b32 	%r898, %r896, 32;
	add.s32 	%r899, %r149, %r898;
	ld.shared.v4.b32 	{%r513, %r521, %r529, %r537}, [%r899+8192];
	ld.shared.v4.b32 	{%r517, %r525, %r533, %r541}, [%r899+9216];
	xor.b32 	%r900, %r896, 64;
	add.s32 	%r901, %r149, %r900;
	ld.shared.v4.b32 	{%r514, %r522, %r530, %r538}, [%r901+16384];
	ld.shared.v4.b32 	{%r518, %r526, %r534, %r542}, [%r901+17408];
	xor.b32 	%r902, %r896, 96;
	add.s32 	%r903, %r149, %r902;
	ld.shared.v4.b32 	{%r515, %r523, %r531, %r539}, [%r903+24576];
	ld.shared.v4.b32 	{%r519, %r527, %r535, %r543}, [%r903+25600];
	.loc	1 356 8                         // sk05_mlp_gateup.py:356:8
	// begin inline asm
	@%p7 st.global.v4.b32 [ %rd165 + 0 ], { %r512, %r513, %r514, %r515 };
	// end inline asm
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd166 + 0 ], { %r516, %r517, %r518, %r519 };
	// end inline asm
	// begin inline asm
	@%p9 st.global.v4.b32 [ %rd167 + 0 ], { %r520, %r521, %r522, %r523 };
	// end inline asm
	// begin inline asm
	@%p10 st.global.v4.b32 [ %rd168 + 0 ], { %r524, %r525, %r526, %r527 };
	// end inline asm
	// begin inline asm
	@%p11 st.global.v4.b32 [ %rd169 + 0 ], { %r528, %r529, %r530, %r531 };
	// end inline asm
	// begin inline asm
	@%p12 st.global.v4.b32 [ %rd170 + 0 ], { %r532, %r533, %r534, %r535 };
	// end inline asm
	// begin inline asm
	@%p13 st.global.v4.b32 [ %rd171 + 0 ], { %r536, %r537, %r538, %r539 };
	// end inline asm
	// begin inline asm
	@%p14 st.global.v4.b32 [ %rd172 + 0 ], { %r540, %r541, %r542, %r543 };
	// end inline asm
	.loc	1 354 4                         // sk05_mlp_gateup.py:354:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/usr/local/lib/python3.12/dist-packages/vllm/_genesis/kernels/sk05_mlp_gateup.py"
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
.b32 201                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0xc2 DW_TAG_compile_unit
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
.b8 53
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 103
.b8 97
.b8 116
.b8 101
.b8 117
.b8 112
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 117
.b8 115
.b8 114
.b8 47
.b8 108
.b8 111
.b8 99
.b8 97
.b8 108
.b8 47
.b8 108
.b8 105
.b8 98
.b8 47
.b8 112
.b8 121
.b8 116
.b8 104
.b8 111
.b8 110
.b8 51
.b8 46
.b8 49
.b8 50
.b8 47
.b8 100
.b8 105
.b8 115
.b8 116
.b8 45
.b8 112
.b8 97
.b8 99
.b8 107
.b8 97
.b8 103
.b8 101
.b8 115
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
.b8 2                                   // Abbrev [2] 0x6a:0x1a DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 53
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 103
.b8 97
.b8 116
.b8 101
.b8 117
.b8 112
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x84:0x48 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 106                                // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x99:0x19 DW_TAG_inlined_subroutine
.b32 106                                // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 62                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0xb2:0x19 DW_TAG_inlined_subroutine
.b32 106                                // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 63                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_3 = _Nativo(
    "sk05_mlp_gateup/tile128x128x128_shift0_abi16",
    _PTX_3, "_sk05_mlp_gateup_kernel",
    warps=8, shared=65536,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 17, 18],
    horneado={11: 1, 12: 1, 15: 1, 19: 1, 20: 128, 21: 128, 22: 128, 23: 8, 24: 128},
    div16=[8, 9, 10, 13, 14, 16, 17],
)

_PTX_4 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk05_mlp_gateup_kernel // -- Begin function _sk05_mlp_gateup_kernel
.extern .shared .align 16 .b8 global_smem[];
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
                                        // @_sk05_mlp_gateup_kernel
.visible .entry _sk05_mlp_gateup_kernel(
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_6,
	.param .u32 _sk05_mlp_gateup_kernel_param_7,
	.param .u32 _sk05_mlp_gateup_kernel_param_8,
	.param .u32 _sk05_mlp_gateup_kernel_param_9,
	.param .u32 _sk05_mlp_gateup_kernel_param_10,
	.param .u32 _sk05_mlp_gateup_kernel_param_11,
	.param .u32 _sk05_mlp_gateup_kernel_param_12,
	.param .u32 _sk05_mlp_gateup_kernel_param_13,
	.param .u32 _sk05_mlp_gateup_kernel_param_14,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_16
)
.reqntid 256
{
	.reg .pred 	%p<42>;
	.reg .b16 	%rs<129>;
	.reg .b32 	%r<1523>;
	.reg .b64 	%rd<164>;
	.loc	1 307 0                         // sk05_mlp_gateup.py:307:0
$L__func_begin0:
	.loc	1 307 0                         // sk05_mlp_gateup.py:307:0

// %bb.0:
	ld.param.b32 	%r26, [_sk05_mlp_gateup_kernel_param_13];
	ld.param.b32 	%r25, [_sk05_mlp_gateup_kernel_param_12];
	ld.param.b32 	%r24, [_sk05_mlp_gateup_kernel_param_8];
	ld.param.b32 	%r23, [_sk05_mlp_gateup_kernel_param_7];
	ld.param.b64 	%rd29, [_sk05_mlp_gateup_kernel_param_5];
	ld.param.b64 	%rd28, [_sk05_mlp_gateup_kernel_param_4];
	ld.param.b64 	%rd27, [_sk05_mlp_gateup_kernel_param_3];
	ld.param.b64 	%rd26, [_sk05_mlp_gateup_kernel_param_2];
	ld.param.b64 	%rd25, [_sk05_mlp_gateup_kernel_param_1];
	ld.param.b64 	%rd24, [_sk05_mlp_gateup_kernel_param_0];
$L__tmp0:
	.loc	1 317 24                        // sk05_mlp_gateup.py:317:24
	mov.u32 	%r49, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk05_mlp_gateup.py:318:27 ]
	add.s32 	%r50, %r23, 255;
	.loc	2 43 30                         // standard.py:43:30 @[ sk05_mlp_gateup.py:318:27 ]
	shr.s32 	%r51, %r50, 31;
	shr.u32 	%r52, %r51, 24;
	add.s32 	%r53, %r50, %r52;
	shr.s32 	%r54, %r53, 8;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk05_mlp_gateup.py:319:27 ]
	add.s32 	%r55, %r24, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk05_mlp_gateup.py:319:27 ]
	shr.s32 	%r56, %r55, 31;
	shr.u32 	%r57, %r56, 25;
	add.s32 	%r58, %r55, %r57;
	shr.s32 	%r59, %r58, 7;
$L__tmp3:
	.loc	1 320 29                        // sk05_mlp_gateup.py:320:29
	shl.b32 	%r60, %r59, 3;
	.loc	1 321 22                        // sk05_mlp_gateup.py:321:22
	div.s32 	%r61, %r49, %r60;
	.loc	1 321 38                        // sk05_mlp_gateup.py:321:38
	shl.b32 	%r62, %r61, 3;
	ld.param.b32 	%r63, [_sk05_mlp_gateup_kernel_param_9];
	.loc	1 322 30                        // sk05_mlp_gateup.py:322:30
	sub.s32 	%r64, %r54, %r62;
	ld.param.b32 	%r65, [_sk05_mlp_gateup_kernel_param_10];
	.loc	1 322 39                        // sk05_mlp_gateup.py:322:39
	min.s32 	%r66, %r64, 8;
	ld.param.b32 	%r67, [_sk05_mlp_gateup_kernel_param_11];
	.loc	1 323 30                        // sk05_mlp_gateup.py:323:30
	mul.lo.s32 	%r68, %r61, %r60;
	sub.s32 	%r69, %r49, %r68;
	.loc	1 324 36                        // sk05_mlp_gateup.py:324:36
	div.s32 	%r70, %r69, %r66;
	.loc	1 323 46                        // sk05_mlp_gateup.py:323:46
	mul.lo.s32 	%r71, %r70, %r66;
	sub.s32 	%r72, %r69, %r71;
	.loc	1 323 23                        // sk05_mlp_gateup.py:323:23
	add.s32 	%r73, %r72, %r62;
	.loc	1 326 22                        // sk05_mlp_gateup.py:326:22
	shl.b32 	%r1, %r73, 8;
	.loc	1 326 45                        // sk05_mlp_gateup.py:326:45
	mov.u32 	%r2, %tid.x;
	shr.u32 	%r74, %r2, 2;
	bfe.u32 	%r75, %r2, 2, 6;
	or.b32 	%r76, %r75, 64;
	and.b32 	%r3, %r2, 255;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r77, %r1, %r75;
	or.b32 	%r78, %r1, %r76;
	or.b32 	%r79, %r77, 128;
	or.b32 	%r80, %r1, %r74;
	or.b32 	%r81, %r80, 192;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r82, %r77, %r23;
	rem.s32 	%r83, %r78, %r23;
	rem.s32 	%r84, %r79, %r23;
	rem.s32 	%r85, %r81, %r23;
	.loc	1 327 22                        // sk05_mlp_gateup.py:327:22
	shl.b32 	%r4, %r70, 7;
	.loc	1 327 45                        // sk05_mlp_gateup.py:327:45
	and.b32 	%r5, %r2, 3;
	shl.b32 	%r86, %r5, 1;
	and.b32 	%r6, %r2, 32;
	shr.u32 	%r87, %r6, 2;
	or.b32 	%r88, %r87, %r86;
	and.b32 	%r7, %r2, 15;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r89, %r4, %r75;
	or.b32 	%r90, %r4, %r76;
	or.b32 	%r91, %r4, %r88;
	or.b32 	%r93, %r91, 16;
	or.b32 	%r95, %r91, 32;
	or.b32 	%r97, %r91, 48;
	or.b32 	%r99, %r91, 64;
	or.b32 	%r101, %r91, 80;
	or.b32 	%r103, %r91, 96;
	or.b32 	%r105, %r91, 112;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r107, %r89, %r24;
	rem.s32 	%r108, %r90, %r24;
	rem.s32 	%r8, %r91, %r24;
	rem.s32 	%r9, %r93, %r24;
	rem.s32 	%r10, %r95, %r24;
	rem.s32 	%r11, %r97, %r24;
	rem.s32 	%r12, %r99, %r24;
	rem.s32 	%r13, %r101, %r24;
	rem.s32 	%r14, %r103, %r24;
	rem.s32 	%r15, %r105, %r24;
	.loc	1 330 39                        // sk05_mlp_gateup.py:330:39
	mul.lo.s32 	%r117, %r82, %r65;
	mul.lo.s32 	%r118, %r83, %r65;
	mul.lo.s32 	%r119, %r84, %r65;
	mul.lo.s32 	%r120, %r85, %r65;
	.loc	1 330 21                        // sk05_mlp_gateup.py:330:21
	cvt.s64.s32 	%rd1, %r117;
	add.s64 	%rd49, %rd24, %rd1;
	cvt.s64.s32 	%rd2, %r118;
	add.s64 	%rd50, %rd24, %rd2;
	cvt.s64.s32 	%rd3, %r119;
	add.s64 	%rd51, %rd24, %rd3;
	cvt.s64.s32 	%rd4, %r120;
	add.s64 	%rd52, %rd24, %rd4;
	.loc	1 330 58                        // sk05_mlp_gateup.py:330:58
	shl.b32 	%r121, %r5, 4;
	.loc	1 330 51                        // sk05_mlp_gateup.py:330:51
	cvt.u64.u32 	%rd5, %r121;
	add.s64 	%rd30, %rd49, %rd5;
	add.s64 	%rd31, %rd50, %rd5;
	add.s64 	%rd32, %rd51, %rd5;
	add.s64 	%rd33, %rd52, %rd5;
	.loc	1 331 21                        // sk05_mlp_gateup.py:331:21
	add.s64 	%rd53, %rd25, %rd5;
	.loc	1 331 69                        // sk05_mlp_gateup.py:331:69
	mul.lo.s32 	%r122, %r107, %r67;
	mul.lo.s32 	%r123, %r108, %r67;
	.loc	1 331 51                        // sk05_mlp_gateup.py:331:51
	cvt.s64.s32 	%rd6, %r122;
	add.s64 	%rd34, %rd53, %rd6;
	cvt.s64.s32 	%rd7, %r123;
	add.s64 	%rd35, %rd53, %rd7;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	setp.gt.s32 	%p1, %r63, 63;
	.loc	1 344 30                        // sk05_mlp_gateup.py:344:30
	shl.b32 	%r191, %r3, 4;
	shl.b32 	%r17, %r2, 1;
	and.b32 	%r18, %r17, 48;
	xor.b32 	%r192, %r191, %r18;
	mov.b32 	%r193, global_smem;
	add.s32 	%r28, %r193, %r192;
	selp.b32 	%r29, 16, 0, %p1;
	// begin inline asm
	cp.async.cg.shared.global [ %r28 + 0 ], [ %rd30 + 0 ], 0x10, %r29;
	// end inline asm
	add.s32 	%r30, %r28, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r30 + 0 ], [ %rd31 + 0 ], 0x10, %r29;
	// end inline asm
	add.s32 	%r31, %r28, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r31 + 0 ], [ %rd32 + 0 ], 0x10, %r29;
	// end inline asm
	add.s32 	%r32, %r28, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r32 + 0 ], [ %rd33 + 0 ], 0x10, %r29;
	// end inline asm
	cp.async.commit_group;
	.loc	1 344 47                        // sk05_mlp_gateup.py:344:47
	add.s32 	%r33, %r28, 49152;
	// begin inline asm
	cp.async.cg.shared.global [ %r33 + 0 ], [ %rd34 + 0 ], 0x10, %r29;
	// end inline asm
	add.s32 	%r34, %r28, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r34 + 0 ], [ %rd35 + 0 ], 0x10, %r29;
	// end inline asm
	cp.async.commit_group;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	setp.gt.s32 	%p2, %r63, 127;
	.loc	1 345 18                        // sk05_mlp_gateup.py:345:18
	add.s64 	%rd36, %rd30, 64;
	add.s64 	%rd37, %rd31, 64;
	add.s64 	%rd38, %rd32, 64;
	add.s64 	%rd39, %rd33, 64;
	.loc	1 346 18                        // sk05_mlp_gateup.py:346:18
	add.s64 	%rd40, %rd34, 64;
	add.s64 	%rd41, %rd35, 64;
	.loc	1 344 30                        // sk05_mlp_gateup.py:344:30
	bar.sync 	0;
	add.s32 	%r35, %r28, 16384;
	selp.b32 	%r36, 16, 0, %p2;
	// begin inline asm
	cp.async.cg.shared.global [ %r35 + 0 ], [ %rd36 + 0 ], 0x10, %r36;
	// end inline asm
	add.s32 	%r37, %r28, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r37 + 0 ], [ %rd37 + 0 ], 0x10, %r36;
	// end inline asm
	add.s32 	%r38, %r28, 24576;
	// begin inline asm
	cp.async.cg.shared.global [ %r38 + 0 ], [ %rd38 + 0 ], 0x10, %r36;
	// end inline asm
	add.s32 	%r39, %r28, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r39 + 0 ], [ %rd39 + 0 ], 0x10, %r36;
	// end inline asm
	cp.async.commit_group;
	.loc	1 344 47                        // sk05_mlp_gateup.py:344:47
	add.s32 	%r40, %r28, 57344;
	// begin inline asm
	cp.async.cg.shared.global [ %r40 + 0 ], [ %rd40 + 0 ], 0x10, %r36;
	// end inline asm
	add.s32 	%r41, %r28, 61440;
	// begin inline asm
	cp.async.cg.shared.global [ %r41 + 0 ], [ %rd41 + 0 ], 0x10, %r36;
	// end inline asm
	cp.async.commit_group;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	setp.gt.s32 	%p3, %r63, 191;
	.loc	1 345 18                        // sk05_mlp_gateup.py:345:18
	add.s64 	%rd42, %rd30, 128;
	add.s64 	%rd43, %rd31, 128;
	add.s64 	%rd44, %rd32, 128;
	add.s64 	%rd45, %rd33, 128;
	.loc	1 346 18                        // sk05_mlp_gateup.py:346:18
	add.s64 	%rd46, %rd34, 128;
	add.s64 	%rd47, %rd35, 128;
	.loc	1 344 30                        // sk05_mlp_gateup.py:344:30
	bar.sync 	0;
	add.s32 	%r42, %r28, 32768;
	selp.b32 	%r43, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r42 + 0 ], [ %rd42 + 0 ], 0x10, %r43;
	// end inline asm
	add.s32 	%r44, %r28, 36864;
	// begin inline asm
	cp.async.cg.shared.global [ %r44 + 0 ], [ %rd43 + 0 ], 0x10, %r43;
	// end inline asm
	add.s32 	%r45, %r28, 40960;
	// begin inline asm
	cp.async.cg.shared.global [ %r45 + 0 ], [ %rd44 + 0 ], 0x10, %r43;
	// end inline asm
	add.s32 	%r46, %r28, 45056;
	// begin inline asm
	cp.async.cg.shared.global [ %r46 + 0 ], [ %rd45 + 0 ], 0x10, %r43;
	// end inline asm
	cp.async.commit_group;
	.loc	1 344 47                        // sk05_mlp_gateup.py:344:47
	add.s32 	%r47, %r28, 65536;
	// begin inline asm
	cp.async.cg.shared.global [ %r47 + 0 ], [ %rd46 + 0 ], 0x10, %r43;
	// end inline asm
	add.s32 	%r48, %r28, 69632;
	// begin inline asm
	cp.async.cg.shared.global [ %r48 + 0 ], [ %rd47 + 0 ], 0x10, %r43;
	// end inline asm
	cp.async.commit_group;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	@%p1 bra 	$L__BB0_2;
	bra.uni 	$L__BB0_1;
$L__BB0_2:                              // %.lr.ph
	.loc	1 0 23                          // sk05_mlp_gateup.py:0:23
	ld.param.b32 	%r27, [_sk05_mlp_gateup_kernel_param_14];
	ld.param.b64 	%rd48, [_sk05_mlp_gateup_kernel_param_6];
	or.b32 	%r92, %r91, 1;
	or.b32 	%r94, %r91, 17;
	or.b32 	%r96, %r91, 33;
	or.b32 	%r98, %r91, 49;
	or.b32 	%r100, %r91, 65;
	or.b32 	%r102, %r91, 81;
	or.b32 	%r104, %r91, 97;
	or.b32 	%r106, %r91, 113;
	rem.s32 	%r109, %r92, %r24;
	rem.s32 	%r110, %r94, %r24;
	rem.s32 	%r111, %r96, %r24;
	rem.s32 	%r112, %r98, %r24;
	rem.s32 	%r113, %r100, %r24;
	rem.s32 	%r114, %r102, %r24;
	rem.s32 	%r115, %r104, %r24;
	rem.s32 	%r116, %r106, %r24;
	shr.s32 	%r124, %r8, 31;
	shr.u32 	%r125, %r124, 25;
	add.s32 	%r126, %r8, %r125;
	shr.s32 	%r127, %r126, 7;
	shr.s32 	%r128, %r109, 31;
	shr.u32 	%r129, %r128, 25;
	add.s32 	%r130, %r109, %r129;
	shr.s32 	%r131, %r130, 7;
	shr.s32 	%r132, %r9, 31;
	shr.u32 	%r133, %r132, 25;
	add.s32 	%r134, %r9, %r133;
	shr.s32 	%r135, %r134, 7;
	shr.s32 	%r136, %r110, 31;
	shr.u32 	%r137, %r136, 25;
	add.s32 	%r138, %r110, %r137;
	shr.s32 	%r139, %r138, 7;
	shr.s32 	%r140, %r10, 31;
	shr.u32 	%r141, %r140, 25;
	add.s32 	%r142, %r10, %r141;
	shr.s32 	%r143, %r142, 7;
	shr.s32 	%r144, %r111, 31;
	shr.u32 	%r145, %r144, 25;
	add.s32 	%r146, %r111, %r145;
	shr.s32 	%r147, %r146, 7;
	shr.s32 	%r148, %r11, 31;
	shr.u32 	%r149, %r148, 25;
	add.s32 	%r150, %r11, %r149;
	shr.s32 	%r151, %r150, 7;
	shr.s32 	%r152, %r112, 31;
	shr.u32 	%r153, %r152, 25;
	add.s32 	%r154, %r112, %r153;
	shr.s32 	%r155, %r154, 7;
	shr.s32 	%r156, %r12, 31;
	shr.u32 	%r157, %r156, 25;
	add.s32 	%r158, %r12, %r157;
	shr.s32 	%r159, %r158, 7;
	shr.s32 	%r160, %r113, 31;
	shr.u32 	%r161, %r160, 25;
	add.s32 	%r162, %r113, %r161;
	shr.s32 	%r163, %r162, 7;
	shr.s32 	%r164, %r13, 31;
	shr.u32 	%r165, %r164, 25;
	add.s32 	%r166, %r13, %r165;
	shr.s32 	%r167, %r166, 7;
	shr.s32 	%r168, %r114, 31;
	shr.u32 	%r169, %r168, 25;
	add.s32 	%r170, %r114, %r169;
	shr.s32 	%r171, %r170, 7;
	shr.s32 	%r172, %r14, 31;
	shr.u32 	%r173, %r172, 25;
	add.s32 	%r174, %r14, %r173;
	shr.s32 	%r175, %r174, 7;
	shr.s32 	%r176, %r115, 31;
	shr.u32 	%r177, %r176, 25;
	add.s32 	%r178, %r115, %r177;
	shr.s32 	%r179, %r178, 7;
	shr.s32 	%r180, %r15, 31;
	shr.u32 	%r181, %r180, 25;
	add.s32 	%r182, %r15, %r181;
	shr.s32 	%r183, %r182, 7;
	shr.s32 	%r184, %r116, 31;
	shr.u32 	%r185, %r184, 25;
	add.s32 	%r186, %r116, %r185;
	shr.s32 	%r187, %r186, 7;
	mad.wide.s32 	%rd8, %r127, 4, %rd48;
	mad.wide.s32 	%rd9, %r131, 4, %rd48;
	mad.wide.s32 	%rd10, %r135, 4, %rd48;
	mad.wide.s32 	%rd11, %r139, 4, %rd48;
	mad.wide.s32 	%rd12, %r143, 4, %rd48;
	mad.wide.s32 	%rd13, %r147, 4, %rd48;
	mad.wide.s32 	%rd14, %r151, 4, %rd48;
	mad.wide.s32 	%rd15, %r155, 4, %rd48;
	mad.wide.s32 	%rd16, %r159, 4, %rd48;
	mad.wide.s32 	%rd17, %r163, 4, %rd48;
	mad.wide.s32 	%rd18, %r167, 4, %rd48;
	mad.wide.s32 	%rd19, %r171, 4, %rd48;
	mad.wide.s32 	%rd20, %r175, 4, %rd48;
	mad.wide.s32 	%rd21, %r179, 4, %rd48;
	mad.wide.s32 	%rd22, %r183, 4, %rd48;
	mad.wide.s32 	%rd23, %r187, 4, %rd48;
	shr.s32 	%r188, %r63, 31;
	shr.u32 	%r189, %r188, 26;
	add.s32 	%r190, %r63, %r189;
	shr.s32 	%r16, %r190, 6;
	add.s32 	%r19, %r16, -3;
	shl.b32 	%r194, %r7, 6;
	shl.b32 	%r1393, %r2, 4;
	and.b32 	%r195, %r1393, 3072;
	shl.b32 	%r196, %r2, 3;
	and.b32 	%r197, %r196, 48;
	and.b32 	%r1394, %r2, 16;
	or.b32 	%r198, %r194, %r195;
	xor.b32 	%r199, %r197, %r1394;
	or.b32 	%r20, %r198, %r199;
	xor.b32 	%r21, %r20, 32;
	shl.b32 	%r200, %r2, 6;
	and.b32 	%r201, %r200, 448;
	shl.b32 	%r202, %r6, 4;
	or.b32 	%r203, %r201, %r197;
	xor.b32 	%r204, %r203, %r18;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	add.s32 	%r205, %r193, %r202;
	add.s32 	%r22, %r205, %r204;
	add.s64 	%rd54, %rd7, %rd25;
	add.s64 	%rd163, %rd54, 192;
	add.s64 	%rd55, %rd6, %rd25;
	add.s64 	%rd162, %rd55, 192;
	add.s64 	%rd56, %rd4, %rd24;
	add.s64 	%rd161, %rd56, 192;
	add.s64 	%rd57, %rd3, %rd24;
	add.s64 	%rd160, %rd57, 192;
	add.s64 	%rd58, %rd2, %rd24;
	add.s64 	%rd159, %rd58, 192;
	add.s64 	%rd59, %rd1, %rd24;
	add.s64 	%rd158, %rd59, 192;
	mov.b32 	%r1395, 0f00000000;
	mov.b32 	%r222, 0;
	mov.b32 	%r1391, 2;
	mov.b32 	%r1390, -1;
	mov.b32 	%r1392, %r222;
	mov.b32 	%r1396, %r1395;
	mov.b32 	%r1397, %r1395;
	mov.b32 	%r1398, %r1395;
	mov.b32 	%r1399, %r1395;
	mov.b32 	%r1400, %r1395;
	mov.b32 	%r1401, %r1395;
	mov.b32 	%r1402, %r1395;
	mov.b32 	%r1403, %r1395;
	mov.b32 	%r1404, %r1395;
	mov.b32 	%r1405, %r1395;
	mov.b32 	%r1406, %r1395;
	mov.b32 	%r1407, %r1395;
	mov.b32 	%r1408, %r1395;
	mov.b32 	%r1409, %r1395;
	mov.b32 	%r1410, %r1395;
	mov.b32 	%r1411, %r1395;
	mov.b32 	%r1412, %r1395;
	mov.b32 	%r1413, %r1395;
	mov.b32 	%r1414, %r1395;
	mov.b32 	%r1415, %r1395;
	mov.b32 	%r1416, %r1395;
	mov.b32 	%r1417, %r1395;
	mov.b32 	%r1418, %r1395;
	mov.b32 	%r1419, %r1395;
	mov.b32 	%r1420, %r1395;
	mov.b32 	%r1421, %r1395;
	mov.b32 	%r1422, %r1395;
	mov.b32 	%r1423, %r1395;
	mov.b32 	%r1424, %r1395;
	mov.b32 	%r1425, %r1395;
	mov.b32 	%r1426, %r1395;
	mov.b32 	%r1427, %r1395;
	mov.b32 	%r1428, %r1395;
	mov.b32 	%r1429, %r1395;
	mov.b32 	%r1430, %r1395;
	mov.b32 	%r1431, %r1395;
	mov.b32 	%r1432, %r1395;
	mov.b32 	%r1433, %r1395;
	mov.b32 	%r1434, %r1395;
	mov.b32 	%r1435, %r1395;
	mov.b32 	%r1436, %r1395;
	mov.b32 	%r1437, %r1395;
	mov.b32 	%r1438, %r1395;
	mov.b32 	%r1439, %r1395;
	mov.b32 	%r1440, %r1395;
	mov.b32 	%r1441, %r1395;
	mov.b32 	%r1442, %r1395;
	mov.b32 	%r1443, %r1395;
	mov.b32 	%r1444, %r1395;
	mov.b32 	%r1445, %r1395;
	mov.b32 	%r1446, %r1395;
	mov.b32 	%r1447, %r1395;
	mov.b32 	%r1448, %r1395;
	mov.b32 	%r1449, %r1395;
	mov.b32 	%r1450, %r1395;
	mov.b32 	%r1451, %r1395;
	mov.b32 	%r1452, %r1395;
	mov.b32 	%r1453, %r1395;
	mov.b32 	%r1454, %r1395;
	mov.b32 	%r1455, %r1395;
	mov.b32 	%r1456, %r1395;
	mov.b32 	%r1457, %r1395;
	mov.b32 	%r1458, %r1395;
	mov.b32 	%r1459, %r1395;
	mov.b32 	%r1460, %r1395;
	mov.b32 	%r1461, %r1395;
	mov.b32 	%r1462, %r1395;
	mov.b32 	%r1463, %r1395;
	mov.b32 	%r1464, %r1395;
	mov.b32 	%r1465, %r1395;
	mov.b32 	%r1466, %r1395;
	mov.b32 	%r1467, %r1395;
	mov.b32 	%r1468, %r1395;
	mov.b32 	%r1469, %r1395;
	mov.b32 	%r1470, %r1395;
	mov.b32 	%r1471, %r1395;
	mov.b32 	%r1472, %r1395;
	mov.b32 	%r1473, %r1395;
	mov.b32 	%r1474, %r1395;
	mov.b32 	%r1475, %r1395;
	mov.b32 	%r1476, %r1395;
	mov.b32 	%r1477, %r1395;
	mov.b32 	%r1478, %r1395;
	mov.b32 	%r1479, %r1395;
	mov.b32 	%r1480, %r1395;
	mov.b32 	%r1481, %r1395;
	mov.b32 	%r1482, %r1395;
	mov.b32 	%r1483, %r1395;
	mov.b32 	%r1484, %r1395;
	mov.b32 	%r1485, %r1395;
	mov.b32 	%r1486, %r1395;
	mov.b32 	%r1487, %r1395;
	mov.b32 	%r1488, %r1395;
	mov.b32 	%r1489, %r1395;
	mov.b32 	%r1490, %r1395;
	mov.b32 	%r1491, %r1395;
	mov.b32 	%r1492, %r1395;
	mov.b32 	%r1493, %r1395;
	mov.b32 	%r1494, %r1395;
	mov.b32 	%r1495, %r1395;
	mov.b32 	%r1496, %r1395;
	mov.b32 	%r1497, %r1395;
	mov.b32 	%r1498, %r1395;
	mov.b32 	%r1499, %r1395;
	mov.b32 	%r1500, %r1395;
	mov.b32 	%r1501, %r1395;
	mov.b32 	%r1502, %r1395;
	mov.b32 	%r1503, %r1395;
	mov.b32 	%r1504, %r1395;
	mov.b32 	%r1505, %r1395;
	mov.b32 	%r1506, %r1395;
	mov.b32 	%r1507, %r1395;
	mov.b32 	%r1508, %r1395;
	mov.b32 	%r1509, %r1395;
	mov.b32 	%r1510, %r1395;
	mov.b32 	%r1511, %r1395;
	mov.b32 	%r1512, %r1395;
	mov.b32 	%r1513, %r1395;
	mov.b32 	%r1514, %r1395;
	mov.b32 	%r1515, %r1395;
	mov.b32 	%r1516, %r1395;
	mov.b32 	%r1517, %r1395;
	mov.b32 	%r1518, %r1395;
	mov.b32 	%r1519, %r1395;
	mov.b32 	%r1520, %r1395;
	mov.b32 	%r1521, %r1395;
	mov.b32 	%r1522, %r1395;
$L__BB0_3:                              // %__nv_exp2f.exit
                                        // =>This Inner Loop Header: Depth=1
	setp.lt.s32 	%p4, %r1392, %r19;
	add.s32 	%r422, %r1390, 1;
	setp.gt.s32 	%p5, %r422, 2;
	selp.b32 	%r1390, 0, %r422, %p5;
	.loc	1 342 56                        // sk05_mlp_gateup.py:342:56
	bfe.u32 	%r423, %r1392, 1, 25;
	.loc	1 343 31                        // sk05_mlp_gateup.py:343:31
	mul.lo.s32 	%r424, %r423, %r27;
	.loc	1 342 39                        // sk05_mlp_gateup.py:342:39
	mul.wide.s32 	%rd82, %r424, 4;
	add.s64 	%rd60, %rd8, %rd82;
	add.s64 	%rd61, %rd9, %rd82;
	add.s64 	%rd62, %rd10, %rd82;
	add.s64 	%rd63, %rd11, %rd82;
	add.s64 	%rd64, %rd12, %rd82;
	add.s64 	%rd65, %rd13, %rd82;
	add.s64 	%rd66, %rd14, %rd82;
	add.s64 	%rd67, %rd15, %rd82;
	add.s64 	%rd68, %rd16, %rd82;
	add.s64 	%rd69, %rd17, %rd82;
	add.s64 	%rd70, %rd18, %rd82;
	add.s64 	%rd71, %rd19, %rd82;
	add.s64 	%rd72, %rd20, %rd82;
	add.s64 	%rd73, %rd21, %rd82;
	add.s64 	%rd74, %rd22, %rd82;
	add.s64 	%rd75, %rd23, %rd82;
	.loc	1 342 29                        // sk05_mlp_gateup.py:342:29
	// begin inline asm
	mov.u32 %r206, 0x0;
	ld.global.b32 { %r206 }, [ %rd60 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r207, 0x0;
	ld.global.b32 { %r207 }, [ %rd61 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r208, 0x0;
	ld.global.b32 { %r208 }, [ %rd62 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r209, 0x0;
	ld.global.b32 { %r209 }, [ %rd63 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r210, 0x0;
	ld.global.b32 { %r210 }, [ %rd64 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r211, 0x0;
	ld.global.b32 { %r211 }, [ %rd65 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r212, 0x0;
	ld.global.b32 { %r212 }, [ %rd66 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r213, 0x0;
	ld.global.b32 { %r213 }, [ %rd67 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r214, 0x0;
	ld.global.b32 { %r214 }, [ %rd68 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r215, 0x0;
	ld.global.b32 { %r215 }, [ %rd69 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r216, 0x0;
	ld.global.b32 { %r216 }, [ %rd70 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r217, 0x0;
	ld.global.b32 { %r217 }, [ %rd71 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r218, 0x0;
	ld.global.b32 { %r218 }, [ %rd72 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r219, 0x0;
	ld.global.b32 { %r219 }, [ %rd73 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r220, 0x0;
	ld.global.b32 { %r220 }, [ %rd74 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r221, 0x0;
	ld.global.b32 { %r221 }, [ %rd75 + 0 ];
	// end inline asm
	.loc	1 342 21                        // sk05_mlp_gateup.py:342:21
	ex2.approx.ftz.f32 	%r425, %r206;
	ex2.approx.ftz.f32 	%r426, %r207;
	ex2.approx.ftz.f32 	%r427, %r208;
	ex2.approx.ftz.f32 	%r428, %r209;
	ex2.approx.ftz.f32 	%r429, %r210;
	ex2.approx.ftz.f32 	%r430, %r211;
	ex2.approx.ftz.f32 	%r431, %r212;
	ex2.approx.ftz.f32 	%r432, %r213;
	ex2.approx.ftz.f32 	%r433, %r214;
	ex2.approx.ftz.f32 	%r434, %r215;
	ex2.approx.ftz.f32 	%r435, %r216;
	ex2.approx.ftz.f32 	%r436, %r217;
	ex2.approx.ftz.f32 	%r437, %r218;
	ex2.approx.ftz.f32 	%r438, %r219;
	ex2.approx.ftz.f32 	%r439, %r220;
	ex2.approx.ftz.f32 	%r440, %r221;
	.loc	1 344 30                        // sk05_mlp_gateup.py:344:30
	cp.async.wait_group 	4;
	bar.sync 	0;
	shl.b32 	%r441, %r1390, 14;
	add.s32 	%r442, %r193, %r441;
	add.s32 	%r443, %r442, %r20;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r223, %r224, %r225, %r226}, [%r443];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r243, %r244, %r245, %r246}, [%r443+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r247, %r248, %r249, %r250}, [%r443+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r251, %r252, %r253, %r254}, [%r443+12288];
	add.s32 	%r444, %r442, %r21;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r259, %r260, %r261, %r262}, [%r444];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r311, %r312, %r313, %r314}, [%r444+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r347, %r348, %r349, %r350}, [%r444+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r383, %r384, %r385, %r386}, [%r444+12288];
	.loc	1 344 47                        // sk05_mlp_gateup.py:344:47
	shl.b32 	%r445, %r1390, 13;
	add.s32 	%r446, %r22, %r445;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r227, %r228, %r263, %r264}, [%r446+49152];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r229, %r230, %r269, %r270}, [%r446+50176];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r231, %r232, %r275, %r276}, [%r446+51200];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r233, %r234, %r281, %r282}, [%r446+52224];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r235, %r236, %r287, %r288}, [%r446+53248];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r237, %r238, %r293, %r294}, [%r446+54272];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r239, %r240, %r299, %r300}, [%r446+55296];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r241, %r242, %r305, %r306}, [%r446+56320];
	.loc	1 344 39                        // sk05_mlp_gateup.py:344:39
	mov.b32 	%r255, %r222;
	mov.b32 	%r256, %r222;
	mov.b32 	%r257, %r222;
	mov.b32 	%r258, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r255, %r256, %r257, %r258 }, { %r223, %r224, %r225, %r226 }, { %r227, %r228 }, { %r255, %r256, %r257, %r258 };
	// end inline asm
	mov.b32 	%r265, %r222;
	mov.b32 	%r266, %r222;
	mov.b32 	%r267, %r222;
	mov.b32 	%r268, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r265, %r266, %r267, %r268 }, { %r223, %r224, %r225, %r226 }, { %r229, %r230 }, { %r265, %r266, %r267, %r268 };
	// end inline asm
	mov.b32 	%r271, %r222;
	mov.b32 	%r272, %r222;
	mov.b32 	%r273, %r222;
	mov.b32 	%r274, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r271, %r272, %r273, %r274 }, { %r223, %r224, %r225, %r226 }, { %r231, %r232 }, { %r271, %r272, %r273, %r274 };
	// end inline asm
	mov.b32 	%r277, %r222;
	mov.b32 	%r278, %r222;
	mov.b32 	%r279, %r222;
	mov.b32 	%r280, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r277, %r278, %r279, %r280 }, { %r223, %r224, %r225, %r226 }, { %r233, %r234 }, { %r277, %r278, %r279, %r280 };
	// end inline asm
	mov.b32 	%r283, %r222;
	mov.b32 	%r284, %r222;
	mov.b32 	%r285, %r222;
	mov.b32 	%r286, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r283, %r284, %r285, %r286 }, { %r223, %r224, %r225, %r226 }, { %r235, %r236 }, { %r283, %r284, %r285, %r286 };
	// end inline asm
	mov.b32 	%r289, %r222;
	mov.b32 	%r290, %r222;
	mov.b32 	%r291, %r222;
	mov.b32 	%r292, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r289, %r290, %r291, %r292 }, { %r223, %r224, %r225, %r226 }, { %r237, %r238 }, { %r289, %r290, %r291, %r292 };
	// end inline asm
	mov.b32 	%r295, %r222;
	mov.b32 	%r296, %r222;
	mov.b32 	%r297, %r222;
	mov.b32 	%r298, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r295, %r296, %r297, %r298 }, { %r223, %r224, %r225, %r226 }, { %r239, %r240 }, { %r295, %r296, %r297, %r298 };
	// end inline asm
	mov.b32 	%r301, %r222;
	mov.b32 	%r302, %r222;
	mov.b32 	%r303, %r222;
	mov.b32 	%r304, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r301, %r302, %r303, %r304 }, { %r223, %r224, %r225, %r226 }, { %r241, %r242 }, { %r301, %r302, %r303, %r304 };
	// end inline asm
	mov.b32 	%r307, %r222;
	mov.b32 	%r308, %r222;
	mov.b32 	%r309, %r222;
	mov.b32 	%r310, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r307, %r308, %r309, %r310 }, { %r243, %r244, %r245, %r246 }, { %r227, %r228 }, { %r307, %r308, %r309, %r310 };
	// end inline asm
	mov.b32 	%r315, %r222;
	mov.b32 	%r316, %r222;
	mov.b32 	%r317, %r222;
	mov.b32 	%r318, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r315, %r316, %r317, %r318 }, { %r243, %r244, %r245, %r246 }, { %r229, %r230 }, { %r315, %r316, %r317, %r318 };
	// end inline asm
	mov.b32 	%r319, %r222;
	mov.b32 	%r320, %r222;
	mov.b32 	%r321, %r222;
	mov.b32 	%r322, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r319, %r320, %r321, %r322 }, { %r243, %r244, %r245, %r246 }, { %r231, %r232 }, { %r319, %r320, %r321, %r322 };
	// end inline asm
	mov.b32 	%r323, %r222;
	mov.b32 	%r324, %r222;
	mov.b32 	%r325, %r222;
	mov.b32 	%r326, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r323, %r324, %r325, %r326 }, { %r243, %r244, %r245, %r246 }, { %r233, %r234 }, { %r323, %r324, %r325, %r326 };
	// end inline asm
	mov.b32 	%r327, %r222;
	mov.b32 	%r328, %r222;
	mov.b32 	%r329, %r222;
	mov.b32 	%r330, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r327, %r328, %r329, %r330 }, { %r243, %r244, %r245, %r246 }, { %r235, %r236 }, { %r327, %r328, %r329, %r330 };
	// end inline asm
	mov.b32 	%r331, %r222;
	mov.b32 	%r332, %r222;
	mov.b32 	%r333, %r222;
	mov.b32 	%r334, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r331, %r332, %r333, %r334 }, { %r243, %r244, %r245, %r246 }, { %r237, %r238 }, { %r331, %r332, %r333, %r334 };
	// end inline asm
	mov.b32 	%r335, %r222;
	mov.b32 	%r336, %r222;
	mov.b32 	%r337, %r222;
	mov.b32 	%r338, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r335, %r336, %r337, %r338 }, { %r243, %r244, %r245, %r246 }, { %r239, %r240 }, { %r335, %r336, %r337, %r338 };
	// end inline asm
	mov.b32 	%r339, %r222;
	mov.b32 	%r340, %r222;
	mov.b32 	%r341, %r222;
	mov.b32 	%r342, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r339, %r340, %r341, %r342 }, { %r243, %r244, %r245, %r246 }, { %r241, %r242 }, { %r339, %r340, %r341, %r342 };
	// end inline asm
	mov.b32 	%r343, %r222;
	mov.b32 	%r344, %r222;
	mov.b32 	%r345, %r222;
	mov.b32 	%r346, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r343, %r344, %r345, %r346 }, { %r247, %r248, %r249, %r250 }, { %r227, %r228 }, { %r343, %r344, %r345, %r346 };
	// end inline asm
	mov.b32 	%r351, %r222;
	mov.b32 	%r352, %r222;
	mov.b32 	%r353, %r222;
	mov.b32 	%r354, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r351, %r352, %r353, %r354 }, { %r247, %r248, %r249, %r250 }, { %r229, %r230 }, { %r351, %r352, %r353, %r354 };
	// end inline asm
	mov.b32 	%r355, %r222;
	mov.b32 	%r356, %r222;
	mov.b32 	%r357, %r222;
	mov.b32 	%r358, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r355, %r356, %r357, %r358 }, { %r247, %r248, %r249, %r250 }, { %r231, %r232 }, { %r355, %r356, %r357, %r358 };
	// end inline asm
	mov.b32 	%r359, %r222;
	mov.b32 	%r360, %r222;
	mov.b32 	%r361, %r222;
	mov.b32 	%r362, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r359, %r360, %r361, %r362 }, { %r247, %r248, %r249, %r250 }, { %r233, %r234 }, { %r359, %r360, %r361, %r362 };
	// end inline asm
	mov.b32 	%r363, %r222;
	mov.b32 	%r364, %r222;
	mov.b32 	%r365, %r222;
	mov.b32 	%r366, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r363, %r364, %r365, %r366 }, { %r247, %r248, %r249, %r250 }, { %r235, %r236 }, { %r363, %r364, %r365, %r366 };
	// end inline asm
	mov.b32 	%r367, %r222;
	mov.b32 	%r368, %r222;
	mov.b32 	%r369, %r222;
	mov.b32 	%r370, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r367, %r368, %r369, %r370 }, { %r247, %r248, %r249, %r250 }, { %r237, %r238 }, { %r367, %r368, %r369, %r370 };
	// end inline asm
	mov.b32 	%r371, %r222;
	mov.b32 	%r372, %r222;
	mov.b32 	%r373, %r222;
	mov.b32 	%r374, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r371, %r372, %r373, %r374 }, { %r247, %r248, %r249, %r250 }, { %r239, %r240 }, { %r371, %r372, %r373, %r374 };
	// end inline asm
	mov.b32 	%r375, %r222;
	mov.b32 	%r376, %r222;
	mov.b32 	%r377, %r222;
	mov.b32 	%r378, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r375, %r376, %r377, %r378 }, { %r247, %r248, %r249, %r250 }, { %r241, %r242 }, { %r375, %r376, %r377, %r378 };
	// end inline asm
	mov.b32 	%r379, %r222;
	mov.b32 	%r380, %r222;
	mov.b32 	%r381, %r222;
	mov.b32 	%r382, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r379, %r380, %r381, %r382 }, { %r251, %r252, %r253, %r254 }, { %r227, %r228 }, { %r379, %r380, %r381, %r382 };
	// end inline asm
	mov.b32 	%r387, %r222;
	mov.b32 	%r388, %r222;
	mov.b32 	%r389, %r222;
	mov.b32 	%r390, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r387, %r388, %r389, %r390 }, { %r251, %r252, %r253, %r254 }, { %r229, %r230 }, { %r387, %r388, %r389, %r390 };
	// end inline asm
	mov.b32 	%r391, %r222;
	mov.b32 	%r392, %r222;
	mov.b32 	%r393, %r222;
	mov.b32 	%r394, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r391, %r392, %r393, %r394 }, { %r251, %r252, %r253, %r254 }, { %r231, %r232 }, { %r391, %r392, %r393, %r394 };
	// end inline asm
	mov.b32 	%r395, %r222;
	mov.b32 	%r396, %r222;
	mov.b32 	%r397, %r222;
	mov.b32 	%r398, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r395, %r396, %r397, %r398 }, { %r251, %r252, %r253, %r254 }, { %r233, %r234 }, { %r395, %r396, %r397, %r398 };
	// end inline asm
	mov.b32 	%r399, %r222;
	mov.b32 	%r400, %r222;
	mov.b32 	%r401, %r222;
	mov.b32 	%r402, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r399, %r400, %r401, %r402 }, { %r251, %r252, %r253, %r254 }, { %r235, %r236 }, { %r399, %r400, %r401, %r402 };
	// end inline asm
	mov.b32 	%r403, %r222;
	mov.b32 	%r404, %r222;
	mov.b32 	%r405, %r222;
	mov.b32 	%r406, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r403, %r404, %r405, %r406 }, { %r251, %r252, %r253, %r254 }, { %r237, %r238 }, { %r403, %r404, %r405, %r406 };
	// end inline asm
	mov.b32 	%r407, %r222;
	mov.b32 	%r408, %r222;
	mov.b32 	%r409, %r222;
	mov.b32 	%r410, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r407, %r408, %r409, %r410 }, { %r251, %r252, %r253, %r254 }, { %r239, %r240 }, { %r407, %r408, %r409, %r410 };
	// end inline asm
	mov.b32 	%r414, %r222;
	mov.b32 	%r411, %r222;
	mov.b32 	%r412, %r222;
	mov.b32 	%r413, %r222;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r411, %r412, %r413, %r414 }, { %r251, %r252, %r253, %r254 }, { %r241, %r242 }, { %r411, %r412, %r413, %r414 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r255, %r256, %r257, %r258 }, { %r259, %r260, %r261, %r262 }, { %r263, %r264 }, { %r255, %r256, %r257, %r258 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r265, %r266, %r267, %r268 }, { %r259, %r260, %r261, %r262 }, { %r269, %r270 }, { %r265, %r266, %r267, %r268 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r271, %r272, %r273, %r274 }, { %r259, %r260, %r261, %r262 }, { %r275, %r276 }, { %r271, %r272, %r273, %r274 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r277, %r278, %r279, %r280 }, { %r259, %r260, %r261, %r262 }, { %r281, %r282 }, { %r277, %r278, %r279, %r280 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r283, %r284, %r285, %r286 }, { %r259, %r260, %r261, %r262 }, { %r287, %r288 }, { %r283, %r284, %r285, %r286 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r289, %r290, %r291, %r292 }, { %r259, %r260, %r261, %r262 }, { %r293, %r294 }, { %r289, %r290, %r291, %r292 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r295, %r296, %r297, %r298 }, { %r259, %r260, %r261, %r262 }, { %r299, %r300 }, { %r295, %r296, %r297, %r298 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r301, %r302, %r303, %r304 }, { %r259, %r260, %r261, %r262 }, { %r305, %r306 }, { %r301, %r302, %r303, %r304 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r307, %r308, %r309, %r310 }, { %r311, %r312, %r313, %r314 }, { %r263, %r264 }, { %r307, %r308, %r309, %r310 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r315, %r316, %r317, %r318 }, { %r311, %r312, %r313, %r314 }, { %r269, %r270 }, { %r315, %r316, %r317, %r318 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r319, %r320, %r321, %r322 }, { %r311, %r312, %r313, %r314 }, { %r275, %r276 }, { %r319, %r320, %r321, %r322 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r323, %r324, %r325, %r326 }, { %r311, %r312, %r313, %r314 }, { %r281, %r282 }, { %r323, %r324, %r325, %r326 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r327, %r328, %r329, %r330 }, { %r311, %r312, %r313, %r314 }, { %r287, %r288 }, { %r327, %r328, %r329, %r330 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r331, %r332, %r333, %r334 }, { %r311, %r312, %r313, %r314 }, { %r293, %r294 }, { %r331, %r332, %r333, %r334 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r335, %r336, %r337, %r338 }, { %r311, %r312, %r313, %r314 }, { %r299, %r300 }, { %r335, %r336, %r337, %r338 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r339, %r340, %r341, %r342 }, { %r311, %r312, %r313, %r314 }, { %r305, %r306 }, { %r339, %r340, %r341, %r342 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r343, %r344, %r345, %r346 }, { %r347, %r348, %r349, %r350 }, { %r263, %r264 }, { %r343, %r344, %r345, %r346 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r351, %r352, %r353, %r354 }, { %r347, %r348, %r349, %r350 }, { %r269, %r270 }, { %r351, %r352, %r353, %r354 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r355, %r356, %r357, %r358 }, { %r347, %r348, %r349, %r350 }, { %r275, %r276 }, { %r355, %r356, %r357, %r358 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r359, %r360, %r361, %r362 }, { %r347, %r348, %r349, %r350 }, { %r281, %r282 }, { %r359, %r360, %r361, %r362 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r363, %r364, %r365, %r366 }, { %r347, %r348, %r349, %r350 }, { %r287, %r288 }, { %r363, %r364, %r365, %r366 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r367, %r368, %r369, %r370 }, { %r347, %r348, %r349, %r350 }, { %r293, %r294 }, { %r367, %r368, %r369, %r370 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r371, %r372, %r373, %r374 }, { %r347, %r348, %r349, %r350 }, { %r299, %r300 }, { %r371, %r372, %r373, %r374 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r375, %r376, %r377, %r378 }, { %r347, %r348, %r349, %r350 }, { %r305, %r306 }, { %r375, %r376, %r377, %r378 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r379, %r380, %r381, %r382 }, { %r383, %r384, %r385, %r386 }, { %r263, %r264 }, { %r379, %r380, %r381, %r382 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r387, %r388, %r389, %r390 }, { %r383, %r384, %r385, %r386 }, { %r269, %r270 }, { %r387, %r388, %r389, %r390 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r391, %r392, %r393, %r394 }, { %r383, %r384, %r385, %r386 }, { %r275, %r276 }, { %r391, %r392, %r393, %r394 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r395, %r396, %r397, %r398 }, { %r383, %r384, %r385, %r386 }, { %r281, %r282 }, { %r395, %r396, %r397, %r398 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r399, %r400, %r401, %r402 }, { %r383, %r384, %r385, %r386 }, { %r287, %r288 }, { %r399, %r400, %r401, %r402 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r403, %r404, %r405, %r406 }, { %r383, %r384, %r385, %r386 }, { %r293, %r294 }, { %r403, %r404, %r405, %r406 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r407, %r408, %r409, %r410 }, { %r383, %r384, %r385, %r386 }, { %r299, %r300 }, { %r407, %r408, %r409, %r410 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r411, %r412, %r413, %r414 }, { %r383, %r384, %r385, %r386 }, { %r305, %r306 }, { %r411, %r412, %r413, %r414 };
	// end inline asm
	.loc	1 344 81                        // sk05_mlp_gateup.py:344:81
	cvt.rn.f32.s32 	%r447, %r414;
	cvt.rn.f32.s32 	%r448, %r413;
	cvt.rn.f32.s32 	%r449, %r412;
	cvt.rn.f32.s32 	%r450, %r411;
	cvt.rn.f32.s32 	%r451, %r410;
	cvt.rn.f32.s32 	%r452, %r409;
	cvt.rn.f32.s32 	%r453, %r408;
	cvt.rn.f32.s32 	%r454, %r407;
	cvt.rn.f32.s32 	%r455, %r406;
	cvt.rn.f32.s32 	%r456, %r405;
	cvt.rn.f32.s32 	%r457, %r404;
	cvt.rn.f32.s32 	%r458, %r403;
	cvt.rn.f32.s32 	%r459, %r402;
	cvt.rn.f32.s32 	%r460, %r401;
	cvt.rn.f32.s32 	%r461, %r400;
	cvt.rn.f32.s32 	%r462, %r399;
	cvt.rn.f32.s32 	%r463, %r398;
	cvt.rn.f32.s32 	%r464, %r397;
	cvt.rn.f32.s32 	%r465, %r396;
	cvt.rn.f32.s32 	%r466, %r395;
	cvt.rn.f32.s32 	%r467, %r394;
	cvt.rn.f32.s32 	%r468, %r393;
	cvt.rn.f32.s32 	%r469, %r392;
	cvt.rn.f32.s32 	%r470, %r391;
	cvt.rn.f32.s32 	%r471, %r390;
	cvt.rn.f32.s32 	%r472, %r389;
	cvt.rn.f32.s32 	%r473, %r388;
	cvt.rn.f32.s32 	%r474, %r387;
	cvt.rn.f32.s32 	%r475, %r382;
	cvt.rn.f32.s32 	%r476, %r381;
	cvt.rn.f32.s32 	%r477, %r380;
	cvt.rn.f32.s32 	%r478, %r379;
	cvt.rn.f32.s32 	%r479, %r375;
	cvt.rn.f32.s32 	%r480, %r376;
	cvt.rn.f32.s32 	%r481, %r377;
	cvt.rn.f32.s32 	%r482, %r378;
	cvt.rn.f32.s32 	%r483, %r371;
	cvt.rn.f32.s32 	%r484, %r372;
	cvt.rn.f32.s32 	%r485, %r373;
	cvt.rn.f32.s32 	%r486, %r374;
	cvt.rn.f32.s32 	%r487, %r367;
	cvt.rn.f32.s32 	%r488, %r368;
	cvt.rn.f32.s32 	%r489, %r369;
	cvt.rn.f32.s32 	%r490, %r370;
	cvt.rn.f32.s32 	%r491, %r363;
	cvt.rn.f32.s32 	%r492, %r364;
	cvt.rn.f32.s32 	%r493, %r365;
	cvt.rn.f32.s32 	%r494, %r366;
	cvt.rn.f32.s32 	%r495, %r359;
	cvt.rn.f32.s32 	%r496, %r360;
	cvt.rn.f32.s32 	%r497, %r361;
	cvt.rn.f32.s32 	%r498, %r362;
	cvt.rn.f32.s32 	%r499, %r355;
	cvt.rn.f32.s32 	%r500, %r356;
	cvt.rn.f32.s32 	%r501, %r357;
	cvt.rn.f32.s32 	%r502, %r358;
	cvt.rn.f32.s32 	%r503, %r351;
	cvt.rn.f32.s32 	%r504, %r352;
	cvt.rn.f32.s32 	%r505, %r353;
	cvt.rn.f32.s32 	%r506, %r354;
	cvt.rn.f32.s32 	%r507, %r343;
	cvt.rn.f32.s32 	%r508, %r344;
	cvt.rn.f32.s32 	%r509, %r345;
	cvt.rn.f32.s32 	%r510, %r346;
	cvt.rn.f32.s32 	%r511, %r339;
	cvt.rn.f32.s32 	%r512, %r340;
	cvt.rn.f32.s32 	%r513, %r341;
	cvt.rn.f32.s32 	%r514, %r342;
	cvt.rn.f32.s32 	%r515, %r335;
	cvt.rn.f32.s32 	%r516, %r336;
	cvt.rn.f32.s32 	%r517, %r337;
	cvt.rn.f32.s32 	%r518, %r338;
	cvt.rn.f32.s32 	%r519, %r331;
	cvt.rn.f32.s32 	%r520, %r332;
	cvt.rn.f32.s32 	%r521, %r333;
	cvt.rn.f32.s32 	%r522, %r334;
	cvt.rn.f32.s32 	%r523, %r327;
	cvt.rn.f32.s32 	%r524, %r328;
	cvt.rn.f32.s32 	%r525, %r329;
	cvt.rn.f32.s32 	%r526, %r330;
	cvt.rn.f32.s32 	%r527, %r323;
	cvt.rn.f32.s32 	%r528, %r324;
	cvt.rn.f32.s32 	%r529, %r325;
	cvt.rn.f32.s32 	%r530, %r326;
	cvt.rn.f32.s32 	%r531, %r319;
	cvt.rn.f32.s32 	%r532, %r320;
	cvt.rn.f32.s32 	%r533, %r321;
	cvt.rn.f32.s32 	%r534, %r322;
	cvt.rn.f32.s32 	%r535, %r315;
	cvt.rn.f32.s32 	%r536, %r316;
	cvt.rn.f32.s32 	%r537, %r317;
	cvt.rn.f32.s32 	%r538, %r318;
	cvt.rn.f32.s32 	%r539, %r307;
	cvt.rn.f32.s32 	%r540, %r308;
	cvt.rn.f32.s32 	%r541, %r309;
	cvt.rn.f32.s32 	%r542, %r310;
	cvt.rn.f32.s32 	%r543, %r301;
	cvt.rn.f32.s32 	%r544, %r302;
	cvt.rn.f32.s32 	%r545, %r303;
	cvt.rn.f32.s32 	%r546, %r304;
	cvt.rn.f32.s32 	%r547, %r295;
	cvt.rn.f32.s32 	%r548, %r296;
	cvt.rn.f32.s32 	%r549, %r297;
	cvt.rn.f32.s32 	%r550, %r298;
	cvt.rn.f32.s32 	%r551, %r289;
	cvt.rn.f32.s32 	%r552, %r290;
	cvt.rn.f32.s32 	%r553, %r291;
	cvt.rn.f32.s32 	%r554, %r292;
	cvt.rn.f32.s32 	%r555, %r283;
	cvt.rn.f32.s32 	%r556, %r284;
	cvt.rn.f32.s32 	%r557, %r285;
	cvt.rn.f32.s32 	%r558, %r286;
	cvt.rn.f32.s32 	%r559, %r277;
	cvt.rn.f32.s32 	%r560, %r278;
	cvt.rn.f32.s32 	%r561, %r279;
	cvt.rn.f32.s32 	%r562, %r280;
	cvt.rn.f32.s32 	%r563, %r271;
	cvt.rn.f32.s32 	%r564, %r272;
	cvt.rn.f32.s32 	%r565, %r273;
	cvt.rn.f32.s32 	%r566, %r274;
	cvt.rn.f32.s32 	%r567, %r265;
	cvt.rn.f32.s32 	%r568, %r266;
	cvt.rn.f32.s32 	%r569, %r267;
	cvt.rn.f32.s32 	%r570, %r268;
	cvt.rn.f32.s32 	%r571, %r255;
	cvt.rn.f32.s32 	%r572, %r256;
	cvt.rn.f32.s32 	%r573, %r257;
	cvt.rn.f32.s32 	%r574, %r258;
	.loc	1 344 15                        // sk05_mlp_gateup.py:344:15
	fma.rn.f32 	%r1398, %r426, %r574, %r1398;
	fma.rn.f32 	%r1397, %r425, %r573, %r1397;
	fma.rn.f32 	%r1396, %r426, %r572, %r1396;
	fma.rn.f32 	%r1395, %r425, %r571, %r1395;
	fma.rn.f32 	%r1402, %r428, %r570, %r1402;
	fma.rn.f32 	%r1401, %r427, %r569, %r1401;
	fma.rn.f32 	%r1400, %r428, %r568, %r1400;
	fma.rn.f32 	%r1399, %r427, %r567, %r1399;
	fma.rn.f32 	%r1406, %r430, %r566, %r1406;
	fma.rn.f32 	%r1405, %r429, %r565, %r1405;
	fma.rn.f32 	%r1404, %r430, %r564, %r1404;
	fma.rn.f32 	%r1403, %r429, %r563, %r1403;
	fma.rn.f32 	%r1410, %r432, %r562, %r1410;
	fma.rn.f32 	%r1409, %r431, %r561, %r1409;
	fma.rn.f32 	%r1408, %r432, %r560, %r1408;
	fma.rn.f32 	%r1407, %r431, %r559, %r1407;
	fma.rn.f32 	%r1414, %r434, %r558, %r1414;
	fma.rn.f32 	%r1413, %r433, %r557, %r1413;
	fma.rn.f32 	%r1412, %r434, %r556, %r1412;
	fma.rn.f32 	%r1411, %r433, %r555, %r1411;
	fma.rn.f32 	%r1418, %r436, %r554, %r1418;
	fma.rn.f32 	%r1417, %r435, %r553, %r1417;
	fma.rn.f32 	%r1416, %r436, %r552, %r1416;
	fma.rn.f32 	%r1415, %r435, %r551, %r1415;
	fma.rn.f32 	%r1422, %r438, %r550, %r1422;
	fma.rn.f32 	%r1421, %r437, %r549, %r1421;
	fma.rn.f32 	%r1420, %r438, %r548, %r1420;
	fma.rn.f32 	%r1419, %r437, %r547, %r1419;
	fma.rn.f32 	%r1426, %r440, %r546, %r1426;
	fma.rn.f32 	%r1425, %r439, %r545, %r1425;
	fma.rn.f32 	%r1424, %r440, %r544, %r1424;
	fma.rn.f32 	%r1423, %r439, %r543, %r1423;
	fma.rn.f32 	%r1430, %r426, %r542, %r1430;
	fma.rn.f32 	%r1429, %r425, %r541, %r1429;
	fma.rn.f32 	%r1428, %r426, %r540, %r1428;
	fma.rn.f32 	%r1427, %r425, %r539, %r1427;
	fma.rn.f32 	%r1434, %r428, %r538, %r1434;
	fma.rn.f32 	%r1433, %r427, %r537, %r1433;
	fma.rn.f32 	%r1432, %r428, %r536, %r1432;
	fma.rn.f32 	%r1431, %r427, %r535, %r1431;
	fma.rn.f32 	%r1438, %r430, %r534, %r1438;
	fma.rn.f32 	%r1437, %r429, %r533, %r1437;
	fma.rn.f32 	%r1436, %r430, %r532, %r1436;
	fma.rn.f32 	%r1435, %r429, %r531, %r1435;
	fma.rn.f32 	%r1442, %r432, %r530, %r1442;
	fma.rn.f32 	%r1441, %r431, %r529, %r1441;
	fma.rn.f32 	%r1440, %r432, %r528, %r1440;
	fma.rn.f32 	%r1439, %r431, %r527, %r1439;
	fma.rn.f32 	%r1446, %r434, %r526, %r1446;
	fma.rn.f32 	%r1445, %r433, %r525, %r1445;
	fma.rn.f32 	%r1444, %r434, %r524, %r1444;
	fma.rn.f32 	%r1443, %r433, %r523, %r1443;
	fma.rn.f32 	%r1450, %r436, %r522, %r1450;
	fma.rn.f32 	%r1449, %r435, %r521, %r1449;
	fma.rn.f32 	%r1448, %r436, %r520, %r1448;
	fma.rn.f32 	%r1447, %r435, %r519, %r1447;
	fma.rn.f32 	%r1454, %r438, %r518, %r1454;
	fma.rn.f32 	%r1453, %r437, %r517, %r1453;
	fma.rn.f32 	%r1452, %r438, %r516, %r1452;
	fma.rn.f32 	%r1451, %r437, %r515, %r1451;
	fma.rn.f32 	%r1458, %r440, %r514, %r1458;
	fma.rn.f32 	%r1457, %r439, %r513, %r1457;
	fma.rn.f32 	%r1456, %r440, %r512, %r1456;
	fma.rn.f32 	%r1455, %r439, %r511, %r1455;
	fma.rn.f32 	%r1462, %r426, %r510, %r1462;
	fma.rn.f32 	%r1461, %r425, %r509, %r1461;
	fma.rn.f32 	%r1460, %r426, %r508, %r1460;
	fma.rn.f32 	%r1459, %r425, %r507, %r1459;
	fma.rn.f32 	%r1466, %r428, %r506, %r1466;
	fma.rn.f32 	%r1465, %r427, %r505, %r1465;
	fma.rn.f32 	%r1464, %r428, %r504, %r1464;
	fma.rn.f32 	%r1463, %r427, %r503, %r1463;
	fma.rn.f32 	%r1470, %r430, %r502, %r1470;
	fma.rn.f32 	%r1469, %r429, %r501, %r1469;
	fma.rn.f32 	%r1468, %r430, %r500, %r1468;
	fma.rn.f32 	%r1467, %r429, %r499, %r1467;
	fma.rn.f32 	%r1474, %r432, %r498, %r1474;
	fma.rn.f32 	%r1473, %r431, %r497, %r1473;
	fma.rn.f32 	%r1472, %r432, %r496, %r1472;
	fma.rn.f32 	%r1471, %r431, %r495, %r1471;
	fma.rn.f32 	%r1478, %r434, %r494, %r1478;
	fma.rn.f32 	%r1477, %r433, %r493, %r1477;
	fma.rn.f32 	%r1476, %r434, %r492, %r1476;
	fma.rn.f32 	%r1475, %r433, %r491, %r1475;
	fma.rn.f32 	%r1482, %r436, %r490, %r1482;
	fma.rn.f32 	%r1481, %r435, %r489, %r1481;
	fma.rn.f32 	%r1480, %r436, %r488, %r1480;
	fma.rn.f32 	%r1479, %r435, %r487, %r1479;
	fma.rn.f32 	%r1486, %r438, %r486, %r1486;
	fma.rn.f32 	%r1485, %r437, %r485, %r1485;
	fma.rn.f32 	%r1484, %r438, %r484, %r1484;
	fma.rn.f32 	%r1483, %r437, %r483, %r1483;
	fma.rn.f32 	%r1490, %r440, %r482, %r1490;
	fma.rn.f32 	%r1489, %r439, %r481, %r1489;
	fma.rn.f32 	%r1488, %r440, %r480, %r1488;
	fma.rn.f32 	%r1487, %r439, %r479, %r1487;
	fma.rn.f32 	%r1491, %r425, %r478, %r1491;
	fma.rn.f32 	%r1492, %r426, %r477, %r1492;
	fma.rn.f32 	%r1493, %r425, %r476, %r1493;
	fma.rn.f32 	%r1494, %r426, %r475, %r1494;
	fma.rn.f32 	%r1495, %r427, %r474, %r1495;
	fma.rn.f32 	%r1496, %r428, %r473, %r1496;
	fma.rn.f32 	%r1497, %r427, %r472, %r1497;
	fma.rn.f32 	%r1498, %r428, %r471, %r1498;
	fma.rn.f32 	%r1499, %r429, %r470, %r1499;
	fma.rn.f32 	%r1500, %r430, %r469, %r1500;
	fma.rn.f32 	%r1501, %r429, %r468, %r1501;
	fma.rn.f32 	%r1502, %r430, %r467, %r1502;
	fma.rn.f32 	%r1503, %r431, %r466, %r1503;
	fma.rn.f32 	%r1504, %r432, %r465, %r1504;
	fma.rn.f32 	%r1505, %r431, %r464, %r1505;
	fma.rn.f32 	%r1506, %r432, %r463, %r1506;
	fma.rn.f32 	%r1507, %r433, %r462, %r1507;
	fma.rn.f32 	%r1508, %r434, %r461, %r1508;
	fma.rn.f32 	%r1509, %r433, %r460, %r1509;
	fma.rn.f32 	%r1510, %r434, %r459, %r1510;
	fma.rn.f32 	%r1511, %r435, %r458, %r1511;
	fma.rn.f32 	%r1512, %r436, %r457, %r1512;
	fma.rn.f32 	%r1513, %r435, %r456, %r1513;
	fma.rn.f32 	%r1514, %r436, %r455, %r1514;
	fma.rn.f32 	%r1515, %r437, %r454, %r1515;
	fma.rn.f32 	%r1516, %r438, %r453, %r1516;
	fma.rn.f32 	%r1517, %r437, %r452, %r1517;
	fma.rn.f32 	%r1518, %r438, %r451, %r1518;
	fma.rn.f32 	%r1519, %r439, %r450, %r1519;
	fma.rn.f32 	%r1520, %r440, %r449, %r1520;
	fma.rn.f32 	%r1521, %r439, %r448, %r1521;
	fma.rn.f32 	%r1522, %r440, %r447, %r1522;
	.loc	1 345 18                        // sk05_mlp_gateup.py:345:18
	add.s64 	%rd76, %rd158, %rd5;
	add.s64 	%rd77, %rd159, %rd5;
	add.s64 	%rd78, %rd160, %rd5;
	.loc	1 346 18                        // sk05_mlp_gateup.py:346:18
	add.s64 	%rd79, %rd161, %rd5;
	add.s64 	%rd80, %rd162, %rd5;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	add.s64 	%rd81, %rd163, %rd5;
	add.s32 	%r575, %r1391, 1;
	setp.gt.s32 	%p6, %r575, 2;
	selp.b32 	%r1391, 0, %r575, %p6;
	.loc	1 344 30                        // sk05_mlp_gateup.py:344:30
	shl.b32 	%r576, %r1391, 14;
	bar.sync 	0;
	add.s32 	%r415, %r28, %r576;
	selp.b32 	%r416, 16, 0, %p4;
	// begin inline asm
	cp.async.cg.shared.global [ %r415 + 0 ], [ %rd76 + 0 ], 0x10, %r416;
	// end inline asm
	add.s32 	%r417, %r415, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r417 + 0 ], [ %rd77 + 0 ], 0x10, %r416;
	// end inline asm
	add.s32 	%r418, %r415, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r418 + 0 ], [ %rd78 + 0 ], 0x10, %r416;
	// end inline asm
	add.s32 	%r419, %r415, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r419 + 0 ], [ %rd79 + 0 ], 0x10, %r416;
	// end inline asm
	cp.async.commit_group;
	.loc	1 344 47                        // sk05_mlp_gateup.py:344:47
	shl.b32 	%r577, %r1391, 13;
	add.s32 	%r578, %r28, %r577;
	add.s32 	%r420, %r578, 49152;
	// begin inline asm
	cp.async.cg.shared.global [ %r420 + 0 ], [ %rd80 + 0 ], 0x10, %r416;
	// end inline asm
	add.s32 	%r421, %r578, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r421 + 0 ], [ %rd81 + 0 ], 0x10, %r416;
	// end inline asm
	cp.async.commit_group;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	add.s32 	%r1392, %r1392, 1;
	add.s64 	%rd163, %rd163, 64;
	add.s64 	%rd162, %rd162, 64;
	add.s64 	%rd161, %rd161, 64;
	add.s64 	%rd160, %rd160, 64;
	add.s64 	%rd159, %rd159, 64;
	add.s64 	%rd158, %rd158, 64;
	setp.ne.b32 	%p7, %r16, %r1392;
	@%p7 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_4;
$L__BB0_1:                              // %.._crit_edge_crit_edge
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	and.b32 	%r1394, %r2, 16;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	shl.b32 	%r1393, %r2, 4;
	mov.b32 	%r1395, 0f00000000;
	mov.b32 	%r1396, %r1395;
	mov.b32 	%r1397, %r1395;
	mov.b32 	%r1398, %r1395;
	mov.b32 	%r1399, %r1395;
	mov.b32 	%r1400, %r1395;
	mov.b32 	%r1401, %r1395;
	mov.b32 	%r1402, %r1395;
	mov.b32 	%r1403, %r1395;
	mov.b32 	%r1404, %r1395;
	mov.b32 	%r1405, %r1395;
	mov.b32 	%r1406, %r1395;
	mov.b32 	%r1407, %r1395;
	mov.b32 	%r1408, %r1395;
	mov.b32 	%r1409, %r1395;
	mov.b32 	%r1410, %r1395;
	mov.b32 	%r1411, %r1395;
	mov.b32 	%r1412, %r1395;
	mov.b32 	%r1413, %r1395;
	mov.b32 	%r1414, %r1395;
	mov.b32 	%r1415, %r1395;
	mov.b32 	%r1416, %r1395;
	mov.b32 	%r1417, %r1395;
	mov.b32 	%r1418, %r1395;
	mov.b32 	%r1419, %r1395;
	mov.b32 	%r1420, %r1395;
	mov.b32 	%r1421, %r1395;
	mov.b32 	%r1422, %r1395;
	mov.b32 	%r1423, %r1395;
	mov.b32 	%r1424, %r1395;
	mov.b32 	%r1425, %r1395;
	mov.b32 	%r1426, %r1395;
	mov.b32 	%r1427, %r1395;
	mov.b32 	%r1428, %r1395;
	mov.b32 	%r1429, %r1395;
	mov.b32 	%r1430, %r1395;
	mov.b32 	%r1431, %r1395;
	mov.b32 	%r1432, %r1395;
	mov.b32 	%r1433, %r1395;
	mov.b32 	%r1434, %r1395;
	mov.b32 	%r1435, %r1395;
	mov.b32 	%r1436, %r1395;
	mov.b32 	%r1437, %r1395;
	mov.b32 	%r1438, %r1395;
	mov.b32 	%r1439, %r1395;
	mov.b32 	%r1440, %r1395;
	mov.b32 	%r1441, %r1395;
	mov.b32 	%r1442, %r1395;
	mov.b32 	%r1443, %r1395;
	mov.b32 	%r1444, %r1395;
	mov.b32 	%r1445, %r1395;
	mov.b32 	%r1446, %r1395;
	mov.b32 	%r1447, %r1395;
	mov.b32 	%r1448, %r1395;
	mov.b32 	%r1449, %r1395;
	mov.b32 	%r1450, %r1395;
	mov.b32 	%r1451, %r1395;
	mov.b32 	%r1452, %r1395;
	mov.b32 	%r1453, %r1395;
	mov.b32 	%r1454, %r1395;
	mov.b32 	%r1455, %r1395;
	mov.b32 	%r1456, %r1395;
	mov.b32 	%r1457, %r1395;
	mov.b32 	%r1458, %r1395;
	mov.b32 	%r1459, %r1395;
	mov.b32 	%r1460, %r1395;
	mov.b32 	%r1461, %r1395;
	mov.b32 	%r1462, %r1395;
	mov.b32 	%r1463, %r1395;
	mov.b32 	%r1464, %r1395;
	mov.b32 	%r1465, %r1395;
	mov.b32 	%r1466, %r1395;
	mov.b32 	%r1467, %r1395;
	mov.b32 	%r1468, %r1395;
	mov.b32 	%r1469, %r1395;
	mov.b32 	%r1470, %r1395;
	mov.b32 	%r1471, %r1395;
	mov.b32 	%r1472, %r1395;
	mov.b32 	%r1473, %r1395;
	mov.b32 	%r1474, %r1395;
	mov.b32 	%r1475, %r1395;
	mov.b32 	%r1476, %r1395;
	mov.b32 	%r1477, %r1395;
	mov.b32 	%r1478, %r1395;
	mov.b32 	%r1479, %r1395;
	mov.b32 	%r1480, %r1395;
	mov.b32 	%r1481, %r1395;
	mov.b32 	%r1482, %r1395;
	mov.b32 	%r1483, %r1395;
	mov.b32 	%r1484, %r1395;
	mov.b32 	%r1485, %r1395;
	mov.b32 	%r1486, %r1395;
	mov.b32 	%r1487, %r1395;
	mov.b32 	%r1488, %r1395;
	mov.b32 	%r1489, %r1395;
	mov.b32 	%r1490, %r1395;
	mov.b32 	%r1491, %r1395;
	mov.b32 	%r1492, %r1395;
	mov.b32 	%r1493, %r1395;
	mov.b32 	%r1494, %r1395;
	mov.b32 	%r1495, %r1395;
	mov.b32 	%r1496, %r1395;
	mov.b32 	%r1497, %r1395;
	mov.b32 	%r1498, %r1395;
	mov.b32 	%r1499, %r1395;
	mov.b32 	%r1500, %r1395;
	mov.b32 	%r1501, %r1395;
	mov.b32 	%r1502, %r1395;
	mov.b32 	%r1503, %r1395;
	mov.b32 	%r1504, %r1395;
	mov.b32 	%r1505, %r1395;
	mov.b32 	%r1506, %r1395;
	mov.b32 	%r1507, %r1395;
	mov.b32 	%r1508, %r1395;
	mov.b32 	%r1509, %r1395;
	mov.b32 	%r1510, %r1395;
	mov.b32 	%r1511, %r1395;
	mov.b32 	%r1512, %r1395;
	mov.b32 	%r1513, %r1395;
	mov.b32 	%r1514, %r1395;
	mov.b32 	%r1515, %r1395;
	mov.b32 	%r1516, %r1395;
	mov.b32 	%r1517, %r1395;
	mov.b32 	%r1518, %r1395;
	mov.b32 	%r1519, %r1395;
	mov.b32 	%r1520, %r1395;
	mov.b32 	%r1521, %r1395;
	mov.b32 	%r1522, %r1395;
$L__BB0_4:                              // %._crit_edge
	.loc	1 327 45                        // sk05_mlp_gateup.py:327:45
	shl.b32 	%r809, %r7, 3;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r810, %r4, %r809;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r811, %r810, %r24;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r812, %r1, %r3;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r813, %r812, %r23;
	.loc	1 326 45                        // sk05_mlp_gateup.py:326:45
	and.b32 	%r814, %r2, 240;
	bfe.u32 	%r815, %r2, 4, 4;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r816, %r815, %r1;
	or.b32 	%r817, %r816, 240;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r818, %r817, %r23;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r819, %r816, 224;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r820, %r819, %r23;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r821, %r816, 208;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r822, %r821, %r23;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r823, %r816, 192;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r824, %r823, %r23;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r825, %r816, 176;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r826, %r825, %r23;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r827, %r816, 160;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r828, %r827, %r23;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r829, %r816, 144;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r830, %r829, %r23;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r831, %r816, 128;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r832, %r831, %r23;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r833, %r816, 112;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r834, %r833, %r23;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r835, %r816, 96;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r836, %r835, %r23;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r837, %r816, 80;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r838, %r837, %r23;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r839, %r816, 64;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r840, %r839, %r23;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r841, %r816, 48;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r842, %r841, %r23;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r843, %r816, 32;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r844, %r843, %r23;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r845, %r816, 16;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r846, %r845, %r23;
	rem.s32 	%r847, %r816, %r23;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 348 38                        // sk05_mlp_gateup.py:348:38
	mad.wide.s32 	%rd83, %r813, 4, %rd28;
	.loc	1 348 24                        // sk05_mlp_gateup.py:348:24
	// begin inline asm
	mov.u32 %r580, 0x0;
	ld.global.b32 { %r580 }, [ %rd83 + 0 ];
	// end inline asm
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	and.b32 	%r848, %r2, 7;
	shl.b32 	%r849, %r848, 3;
	shl.b32 	%r850, %r814, 2;
	and.b32 	%r851, %r2, 8;
	shr.u32 	%r852, %r851, 1;
	add.s32 	%r853, %r193, %r849;
	add.s32 	%r854, %r853, %r850;
	add.s32 	%r579, %r854, %r852;
	// begin inline asm
	st.shared.b32 [ %r579 + 0 ], %r580;
	// end inline asm
	bar.sync 	0;
	and.b32 	%r855, %r17, 56;
	and.b32 	%r856, %r2, 192;
	add.s32 	%r857, %r193, %r855;
	add.s32 	%r858, %r857, %r856;
	ld.shared.v2.b32 	{%r859, %r860}, [%r858];
	ld.shared.v2.b32 	{%r861, %r862}, [%r858+256];
	ld.shared.v2.b32 	{%r863, %r864}, [%r858+512];
	ld.shared.v2.b32 	{%r865, %r866}, [%r858+768];
	.loc	1 349 38                        // sk05_mlp_gateup.py:349:38
	mad.wide.s32 	%rd84, %r8, 4, %rd29;
	mad.wide.s32 	%rd85, %r9, 4, %rd29;
	mad.wide.s32 	%rd86, %r10, 4, %rd29;
	mad.wide.s32 	%rd87, %r11, 4, %rd29;
	mad.wide.s32 	%rd88, %r12, 4, %rd29;
	mad.wide.s32 	%rd89, %r13, 4, %rd29;
	mad.wide.s32 	%rd90, %r14, 4, %rd29;
	mad.wide.s32 	%rd91, %r15, 4, %rd29;
	.loc	1 349 24                        // sk05_mlp_gateup.py:349:24
	// begin inline asm
	mov.u32 %r581, 0x0;
	mov.u32 %r582, 0x0;
	ld.global.v2.b32 { %r581, %r582 }, [ %rd84 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r583, 0x0;
	mov.u32 %r584, 0x0;
	ld.global.v2.b32 { %r583, %r584 }, [ %rd85 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r585, 0x0;
	mov.u32 %r586, 0x0;
	ld.global.v2.b32 { %r585, %r586 }, [ %rd86 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r587, 0x0;
	mov.u32 %r588, 0x0;
	ld.global.v2.b32 { %r587, %r588 }, [ %rd87 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r589, 0x0;
	mov.u32 %r590, 0x0;
	ld.global.v2.b32 { %r589, %r590 }, [ %rd88 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r591, 0x0;
	mov.u32 %r592, 0x0;
	ld.global.v2.b32 { %r591, %r592 }, [ %rd89 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r593, 0x0;
	mov.u32 %r594, 0x0;
	ld.global.v2.b32 { %r593, %r594 }, [ %rd90 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r595, 0x0;
	mov.u32 %r596, 0x0;
	ld.global.v2.b32 { %r595, %r596 }, [ %rd91 + 0 ];
	// end inline asm
	.loc	1 350 49                        // sk05_mlp_gateup.py:350:49
	mul.lo.s32 	%r867, %r847, %r26;
	mul.lo.s32 	%r868, %r846, %r26;
	mul.lo.s32 	%r869, %r844, %r26;
	mul.lo.s32 	%r870, %r842, %r26;
	mul.lo.s32 	%r871, %r840, %r26;
	mul.lo.s32 	%r872, %r838, %r26;
	mul.lo.s32 	%r873, %r836, %r26;
	mul.lo.s32 	%r874, %r834, %r26;
	mul.lo.s32 	%r875, %r832, %r26;
	mul.lo.s32 	%r876, %r830, %r26;
	mul.lo.s32 	%r877, %r828, %r26;
	mul.lo.s32 	%r878, %r826, %r26;
	mul.lo.s32 	%r879, %r824, %r26;
	mul.lo.s32 	%r880, %r822, %r26;
	mul.lo.s32 	%r881, %r820, %r26;
	mul.lo.s32 	%r882, %r818, %r26;
	.loc	1 350 31                        // sk05_mlp_gateup.py:350:31
	mad.wide.s32 	%rd124, %r867, 2, %rd27;
	mad.wide.s32 	%rd125, %r868, 2, %rd27;
	mad.wide.s32 	%rd126, %r869, 2, %rd27;
	mad.wide.s32 	%rd127, %r870, 2, %rd27;
	mad.wide.s32 	%rd128, %r871, 2, %rd27;
	mad.wide.s32 	%rd129, %r872, 2, %rd27;
	mad.wide.s32 	%rd130, %r873, 2, %rd27;
	mad.wide.s32 	%rd131, %r874, 2, %rd27;
	mad.wide.s32 	%rd132, %r875, 2, %rd27;
	mad.wide.s32 	%rd133, %r876, 2, %rd27;
	mad.wide.s32 	%rd134, %r877, 2, %rd27;
	mad.wide.s32 	%rd135, %r878, 2, %rd27;
	mad.wide.s32 	%rd136, %r879, 2, %rd27;
	mad.wide.s32 	%rd137, %r880, 2, %rd27;
	mad.wide.s32 	%rd138, %r881, 2, %rd27;
	mad.wide.s32 	%rd139, %r882, 2, %rd27;
	.loc	1 350 64                        // sk05_mlp_gateup.py:350:64
	mul.wide.s32 	%rd140, %r811, 2;
	add.s64 	%rd92, %rd124, %rd140;
	add.s64 	%rd93, %rd125, %rd140;
	add.s64 	%rd94, %rd126, %rd140;
	add.s64 	%rd95, %rd127, %rd140;
	add.s64 	%rd96, %rd128, %rd140;
	add.s64 	%rd97, %rd129, %rd140;
	add.s64 	%rd98, %rd130, %rd140;
	add.s64 	%rd99, %rd131, %rd140;
	add.s64 	%rd100, %rd132, %rd140;
	add.s64 	%rd101, %rd133, %rd140;
	add.s64 	%rd102, %rd134, %rd140;
	add.s64 	%rd103, %rd135, %rd140;
	add.s64 	%rd104, %rd136, %rd140;
	add.s64 	%rd105, %rd137, %rd140;
	add.s64 	%rd106, %rd138, %rd140;
	add.s64 	%rd107, %rd139, %rd140;
	.loc	1 350 19                        // sk05_mlp_gateup.py:350:19
	// begin inline asm
	mov.u32 %r598, 0x0;
	mov.u32 %r599, 0x0;
	mov.u32 %r600, 0x0;
	mov.u32 %r601, 0x0;
	ld.global.v4.b32 { %r598, %r599, %r600, %r601 }, [ %rd92 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r603, 0x0;
	mov.u32 %r604, 0x0;
	mov.u32 %r605, 0x0;
	mov.u32 %r606, 0x0;
	ld.global.v4.b32 { %r603, %r604, %r605, %r606 }, [ %rd93 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r608, 0x0;
	mov.u32 %r609, 0x0;
	mov.u32 %r610, 0x0;
	mov.u32 %r611, 0x0;
	ld.global.v4.b32 { %r608, %r609, %r610, %r611 }, [ %rd94 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r613, 0x0;
	mov.u32 %r614, 0x0;
	mov.u32 %r615, 0x0;
	mov.u32 %r616, 0x0;
	ld.global.v4.b32 { %r613, %r614, %r615, %r616 }, [ %rd95 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r617, 0x0;
	mov.u32 %r618, 0x0;
	mov.u32 %r619, 0x0;
	mov.u32 %r620, 0x0;
	ld.global.v4.b32 { %r617, %r618, %r619, %r620 }, [ %rd96 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r621, 0x0;
	mov.u32 %r622, 0x0;
	mov.u32 %r623, 0x0;
	mov.u32 %r624, 0x0;
	ld.global.v4.b32 { %r621, %r622, %r623, %r624 }, [ %rd97 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r625, 0x0;
	mov.u32 %r626, 0x0;
	mov.u32 %r627, 0x0;
	mov.u32 %r628, 0x0;
	ld.global.v4.b32 { %r625, %r626, %r627, %r628 }, [ %rd98 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r629, 0x0;
	mov.u32 %r630, 0x0;
	mov.u32 %r631, 0x0;
	mov.u32 %r632, 0x0;
	ld.global.v4.b32 { %r629, %r630, %r631, %r632 }, [ %rd99 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r633, 0x0;
	mov.u32 %r634, 0x0;
	mov.u32 %r635, 0x0;
	mov.u32 %r636, 0x0;
	ld.global.v4.b32 { %r633, %r634, %r635, %r636 }, [ %rd100 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r637, 0x0;
	mov.u32 %r638, 0x0;
	mov.u32 %r639, 0x0;
	mov.u32 %r640, 0x0;
	ld.global.v4.b32 { %r637, %r638, %r639, %r640 }, [ %rd101 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r641, 0x0;
	mov.u32 %r642, 0x0;
	mov.u32 %r643, 0x0;
	mov.u32 %r644, 0x0;
	ld.global.v4.b32 { %r641, %r642, %r643, %r644 }, [ %rd102 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r645, 0x0;
	mov.u32 %r646, 0x0;
	mov.u32 %r647, 0x0;
	mov.u32 %r648, 0x0;
	ld.global.v4.b32 { %r645, %r646, %r647, %r648 }, [ %rd103 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r649, 0x0;
	mov.u32 %r650, 0x0;
	mov.u32 %r651, 0x0;
	mov.u32 %r652, 0x0;
	ld.global.v4.b32 { %r649, %r650, %r651, %r652 }, [ %rd104 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r653, 0x0;
	mov.u32 %r654, 0x0;
	mov.u32 %r655, 0x0;
	mov.u32 %r656, 0x0;
	ld.global.v4.b32 { %r653, %r654, %r655, %r656 }, [ %rd105 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r657, 0x0;
	mov.u32 %r658, 0x0;
	mov.u32 %r659, 0x0;
	mov.u32 %r660, 0x0;
	ld.global.v4.b32 { %r657, %r658, %r659, %r660 }, [ %rd106 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r661, 0x0;
	mov.u32 %r662, 0x0;
	mov.u32 %r663, 0x0;
	mov.u32 %r664, 0x0;
	ld.global.v4.b32 { %r661, %r662, %r663, %r664 }, [ %rd107 + 0 ];
	// end inline asm
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	bar.sync 	0;
	shl.b32 	%r883, %r2, 7;
	and.b32 	%r884, %r883, 15360;
	shl.b32 	%r885, %r848, 4;
	or.b32 	%r886, %r884, %r885;
	xor.b32 	%r887, %r886, %r814;
	add.s32 	%r597, %r193, %r887;
	// begin inline asm
	st.shared.v4.b32 [ %r597 + 0 ], { %r598, %r599, %r600, %r601 };
	// end inline asm
	add.s32 	%r602, %r597, 256;
	// begin inline asm
	st.shared.v4.b32 [ %r602 + 0 ], { %r603, %r604, %r605, %r606 };
	// end inline asm
	add.s32 	%r607, %r597, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r607 + 0 ], { %r608, %r609, %r610, %r611 };
	// end inline asm
	add.s32 	%r612, %r597, 768;
	// begin inline asm
	st.shared.v4.b32 [ %r612 + 0 ], { %r613, %r614, %r615, %r616 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r888, %r848, 11;
	shl.b32 	%r889, %r7, 4;
	shl.b32 	%r890, %r856, 2;
	setp.eq.b32 	%p24, %r1394, 0;
	shl.b32 	%r891, %r1394, 1;
	shr.u32 	%r892, %r6, 1;
	or.b32 	%r893, %r889, %r890;
	or.b32 	%r894, %r891, %r892;
	xor.b32 	%r895, %r893, %r894;
	or.b32 	%r896, %r895, %r888;
	add.s32 	%r897, %r193, %r896;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r898, %r899, %r900, %r901}, [%r897];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r902, %r903, %r904, %r905}, [%r897+1024];
	xor.b32 	%r906, %r896, 64;
	add.s32 	%r907, %r193, %r906;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r908, %r909, %r910, %r911}, [%r907];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r912, %r913, %r914, %r915}, [%r907+1024];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r597 + 0 ], { %r617, %r618, %r619, %r620 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r602 + 0 ], { %r621, %r622, %r623, %r624 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r607 + 0 ], { %r625, %r626, %r627, %r628 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r612 + 0 ], { %r629, %r630, %r631, %r632 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r916, %r917, %r918, %r919}, [%r897];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r920, %r921, %r922, %r923}, [%r897+1024];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r924, %r925, %r926, %r927}, [%r907];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r928, %r929, %r930, %r931}, [%r907+1024];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r597 + 0 ], { %r633, %r634, %r635, %r636 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r602 + 0 ], { %r637, %r638, %r639, %r640 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r607 + 0 ], { %r641, %r642, %r643, %r644 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r612 + 0 ], { %r645, %r646, %r647, %r648 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r932, %r933, %r934, %r935}, [%r897];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r936, %r937, %r938, %r939}, [%r897+1024];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r940, %r941, %r942, %r943}, [%r907];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r944, %r945, %r946, %r947}, [%r907+1024];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r597 + 0 ], { %r649, %r650, %r651, %r652 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r602 + 0 ], { %r653, %r654, %r655, %r656 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r607 + 0 ], { %r657, %r658, %r659, %r660 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r612 + 0 ], { %r661, %r662, %r663, %r664 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r948, %r949, %r950, %r951}, [%r897];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r952, %r953, %r954, %r955}, [%r897+1024];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r956, %r957, %r958, %r959}, [%r907];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r960, %r961, %r962, %r963}, [%r907+1024];
	.loc	1 357 31                        // sk05_mlp_gateup.py:357:31
	setp.lt.s32 	%p25, %r816, %r23;
	setp.lt.s32 	%p26, %r845, %r23;
	setp.lt.s32 	%p27, %r843, %r23;
	setp.lt.s32 	%p28, %r841, %r23;
	setp.lt.s32 	%p29, %r839, %r23;
	setp.lt.s32 	%p30, %r837, %r23;
	setp.lt.s32 	%p31, %r835, %r23;
	setp.lt.s32 	%p32, %r833, %r23;
	setp.lt.s32 	%p33, %r831, %r23;
	setp.lt.s32 	%p34, %r829, %r23;
	setp.lt.s32 	%p35, %r827, %r23;
	setp.lt.s32 	%p36, %r825, %r23;
	setp.lt.s32 	%p37, %r823, %r23;
	setp.lt.s32 	%p38, %r821, %r23;
	setp.lt.s32 	%p39, %r819, %r23;
	setp.lt.s32 	%p40, %r817, %r23;
	.loc	1 357 54                        // sk05_mlp_gateup.py:357:54
	setp.lt.s32 	%p41, %r810, %r24;
	.loc	1 357 37                        // sk05_mlp_gateup.py:357:37
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
	.loc	1 355 35                        // sk05_mlp_gateup.py:355:35
	mul.lo.s32 	%r964, %r816, %r25;
	mul.lo.s32 	%r965, %r845, %r25;
	mul.lo.s32 	%r966, %r843, %r25;
	mul.lo.s32 	%r967, %r841, %r25;
	mul.lo.s32 	%r968, %r839, %r25;
	mul.lo.s32 	%r969, %r837, %r25;
	mul.lo.s32 	%r970, %r835, %r25;
	mul.lo.s32 	%r971, %r833, %r25;
	mul.lo.s32 	%r972, %r831, %r25;
	mul.lo.s32 	%r973, %r829, %r25;
	mul.lo.s32 	%r974, %r827, %r25;
	mul.lo.s32 	%r975, %r825, %r25;
	mul.lo.s32 	%r976, %r823, %r25;
	mul.lo.s32 	%r977, %r821, %r25;
	mul.lo.s32 	%r978, %r819, %r25;
	mul.lo.s32 	%r979, %r817, %r25;
	.loc	1 355 18                        // sk05_mlp_gateup.py:355:18
	mad.wide.s32 	%rd141, %r964, 2, %rd26;
	mad.wide.s32 	%rd142, %r965, 2, %rd26;
	mad.wide.s32 	%rd143, %r966, 2, %rd26;
	mad.wide.s32 	%rd144, %r967, 2, %rd26;
	mad.wide.s32 	%rd145, %r968, 2, %rd26;
	mad.wide.s32 	%rd146, %r969, 2, %rd26;
	mad.wide.s32 	%rd147, %r970, 2, %rd26;
	mad.wide.s32 	%rd148, %r971, 2, %rd26;
	mad.wide.s32 	%rd149, %r972, 2, %rd26;
	mad.wide.s32 	%rd150, %r973, 2, %rd26;
	mad.wide.s32 	%rd151, %r974, 2, %rd26;
	mad.wide.s32 	%rd152, %r975, 2, %rd26;
	mad.wide.s32 	%rd153, %r976, 2, %rd26;
	mad.wide.s32 	%rd154, %r977, 2, %rd26;
	mad.wide.s32 	%rd155, %r978, 2, %rd26;
	mad.wide.s32 	%rd156, %r979, 2, %rd26;
	.loc	1 355 50                        // sk05_mlp_gateup.py:355:50
	mul.wide.s32 	%rd157, %r810, 2;
	add.s64 	%rd108, %rd141, %rd157;
	add.s64 	%rd109, %rd142, %rd157;
	add.s64 	%rd110, %rd143, %rd157;
	add.s64 	%rd111, %rd144, %rd157;
	add.s64 	%rd112, %rd145, %rd157;
	add.s64 	%rd113, %rd146, %rd157;
	add.s64 	%rd114, %rd147, %rd157;
	add.s64 	%rd115, %rd148, %rd157;
	add.s64 	%rd116, %rd149, %rd157;
	add.s64 	%rd117, %rd150, %rd157;
	add.s64 	%rd118, %rd151, %rd157;
	add.s64 	%rd119, %rd152, %rd157;
	add.s64 	%rd120, %rd153, %rd157;
	add.s64 	%rd121, %rd154, %rd157;
	add.s64 	%rd122, %rd155, %rd157;
	add.s64 	%rd123, %rd156, %rd157;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r980, %r1492, %r865;
	mul.f32 	%r981, %r1491, %r865;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs1, %rs2}, %r948;
	cvt.f32.bf16 	%r982, %rs2;
	cvt.f32.bf16 	%r983, %rs1;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r984, %r981, %r581, %r983;
	fma.rn.f32 	%r985, %r980, %r582, %r982;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r986, %r1396, %r859;
	mul.f32 	%r987, %r1395, %r859;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs3, %rs4}, %r898;
	cvt.f32.bf16 	%r988, %rs4;
	cvt.f32.bf16 	%r989, %rs3;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r990, %r987, %r581, %r989;
	fma.rn.f32 	%r991, %r986, %r582, %r988;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r666, %r991, %r990;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r992, %r1398, %r860;
	mul.f32 	%r993, %r1397, %r860;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs5, %rs6}, %r899;
	cvt.f32.bf16 	%r994, %rs6;
	cvt.f32.bf16 	%r995, %rs5;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r996, %r993, %r581, %r995;
	fma.rn.f32 	%r997, %r992, %r582, %r994;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r671, %r997, %r996;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r998, %r1428, %r861;
	mul.f32 	%r999, %r1427, %r861;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs7, %rs8}, %r916;
	cvt.f32.bf16 	%r1000, %rs8;
	cvt.f32.bf16 	%r1001, %rs7;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1002, %r999, %r581, %r1001;
	fma.rn.f32 	%r1003, %r998, %r582, %r1000;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r667, %r1003, %r1002;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1004, %r1430, %r862;
	mul.f32 	%r1005, %r1429, %r862;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs9, %rs10}, %r917;
	cvt.f32.bf16 	%r1006, %rs10;
	cvt.f32.bf16 	%r1007, %rs9;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1008, %r1005, %r581, %r1007;
	fma.rn.f32 	%r1009, %r1004, %r582, %r1006;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r672, %r1009, %r1008;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1010, %r1460, %r863;
	mul.f32 	%r1011, %r1459, %r863;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs11, %rs12}, %r932;
	cvt.f32.bf16 	%r1012, %rs12;
	cvt.f32.bf16 	%r1013, %rs11;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1014, %r1011, %r581, %r1013;
	fma.rn.f32 	%r1015, %r1010, %r582, %r1012;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r668, %r1015, %r1014;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1016, %r1462, %r864;
	mul.f32 	%r1017, %r1461, %r864;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs13, %rs14}, %r933;
	cvt.f32.bf16 	%r1018, %rs14;
	cvt.f32.bf16 	%r1019, %rs13;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1020, %r1017, %r581, %r1019;
	fma.rn.f32 	%r1021, %r1016, %r582, %r1018;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r673, %r1021, %r1020;
	cvt.rn.bf16x2.f32 	%r669, %r985, %r984;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1022, %r1494, %r866;
	mul.f32 	%r1023, %r1493, %r866;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs15, %rs16}, %r949;
	cvt.f32.bf16 	%r1024, %rs16;
	cvt.f32.bf16 	%r1025, %rs15;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1026, %r1023, %r581, %r1025;
	fma.rn.f32 	%r1027, %r1022, %r582, %r1024;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r674, %r1027, %r1026;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1028, %r1496, %r865;
	mul.f32 	%r1029, %r1495, %r865;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs17, %rs18}, %r950;
	cvt.f32.bf16 	%r1030, %rs18;
	cvt.f32.bf16 	%r1031, %rs17;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1032, %r1029, %r583, %r1031;
	fma.rn.f32 	%r1033, %r1028, %r584, %r1030;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1034, %r1400, %r859;
	mul.f32 	%r1035, %r1399, %r859;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs19, %rs20}, %r900;
	cvt.f32.bf16 	%r1036, %rs20;
	cvt.f32.bf16 	%r1037, %rs19;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1038, %r1035, %r583, %r1037;
	fma.rn.f32 	%r1039, %r1034, %r584, %r1036;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r686, %r1039, %r1038;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1040, %r1402, %r860;
	mul.f32 	%r1041, %r1401, %r860;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs21, %rs22}, %r901;
	cvt.f32.bf16 	%r1042, %rs22;
	cvt.f32.bf16 	%r1043, %rs21;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1044, %r1041, %r583, %r1043;
	fma.rn.f32 	%r1045, %r1040, %r584, %r1042;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r691, %r1045, %r1044;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1046, %r1432, %r861;
	mul.f32 	%r1047, %r1431, %r861;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs23, %rs24}, %r918;
	cvt.f32.bf16 	%r1048, %rs24;
	cvt.f32.bf16 	%r1049, %rs23;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1050, %r1047, %r583, %r1049;
	fma.rn.f32 	%r1051, %r1046, %r584, %r1048;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r687, %r1051, %r1050;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1052, %r1434, %r862;
	mul.f32 	%r1053, %r1433, %r862;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs25, %rs26}, %r919;
	cvt.f32.bf16 	%r1054, %rs26;
	cvt.f32.bf16 	%r1055, %rs25;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1056, %r1053, %r583, %r1055;
	fma.rn.f32 	%r1057, %r1052, %r584, %r1054;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r692, %r1057, %r1056;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1058, %r1464, %r863;
	mul.f32 	%r1059, %r1463, %r863;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs27, %rs28}, %r934;
	cvt.f32.bf16 	%r1060, %rs28;
	cvt.f32.bf16 	%r1061, %rs27;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1062, %r1059, %r583, %r1061;
	fma.rn.f32 	%r1063, %r1058, %r584, %r1060;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r688, %r1063, %r1062;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1064, %r1466, %r864;
	mul.f32 	%r1065, %r1465, %r864;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs29, %rs30}, %r935;
	cvt.f32.bf16 	%r1066, %rs30;
	cvt.f32.bf16 	%r1067, %rs29;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1068, %r1065, %r583, %r1067;
	fma.rn.f32 	%r1069, %r1064, %r584, %r1066;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r693, %r1069, %r1068;
	cvt.rn.bf16x2.f32 	%r689, %r1033, %r1032;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1070, %r1498, %r866;
	mul.f32 	%r1071, %r1497, %r866;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs31, %rs32}, %r951;
	cvt.f32.bf16 	%r1072, %rs32;
	cvt.f32.bf16 	%r1073, %rs31;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1074, %r1071, %r583, %r1073;
	fma.rn.f32 	%r1075, %r1070, %r584, %r1072;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r694, %r1075, %r1074;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1076, %r1500, %r865;
	mul.f32 	%r1077, %r1499, %r865;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs33, %rs34}, %r956;
	cvt.f32.bf16 	%r1078, %rs34;
	cvt.f32.bf16 	%r1079, %rs33;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1080, %r1077, %r585, %r1079;
	fma.rn.f32 	%r1081, %r1076, %r586, %r1078;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1082, %r1404, %r859;
	mul.f32 	%r1083, %r1403, %r859;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs35, %rs36}, %r908;
	cvt.f32.bf16 	%r1084, %rs36;
	cvt.f32.bf16 	%r1085, %rs35;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1086, %r1083, %r585, %r1085;
	fma.rn.f32 	%r1087, %r1082, %r586, %r1084;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r706, %r1087, %r1086;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1088, %r1406, %r860;
	mul.f32 	%r1089, %r1405, %r860;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs37, %rs38}, %r909;
	cvt.f32.bf16 	%r1090, %rs38;
	cvt.f32.bf16 	%r1091, %rs37;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1092, %r1089, %r585, %r1091;
	fma.rn.f32 	%r1093, %r1088, %r586, %r1090;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r711, %r1093, %r1092;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1094, %r1436, %r861;
	mul.f32 	%r1095, %r1435, %r861;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs39, %rs40}, %r924;
	cvt.f32.bf16 	%r1096, %rs40;
	cvt.f32.bf16 	%r1097, %rs39;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1098, %r1095, %r585, %r1097;
	fma.rn.f32 	%r1099, %r1094, %r586, %r1096;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r707, %r1099, %r1098;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1100, %r1438, %r862;
	mul.f32 	%r1101, %r1437, %r862;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs41, %rs42}, %r925;
	cvt.f32.bf16 	%r1102, %rs42;
	cvt.f32.bf16 	%r1103, %rs41;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1104, %r1101, %r585, %r1103;
	fma.rn.f32 	%r1105, %r1100, %r586, %r1102;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r712, %r1105, %r1104;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1106, %r1468, %r863;
	mul.f32 	%r1107, %r1467, %r863;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs43, %rs44}, %r940;
	cvt.f32.bf16 	%r1108, %rs44;
	cvt.f32.bf16 	%r1109, %rs43;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1110, %r1107, %r585, %r1109;
	fma.rn.f32 	%r1111, %r1106, %r586, %r1108;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r708, %r1111, %r1110;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1112, %r1470, %r864;
	mul.f32 	%r1113, %r1469, %r864;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs45, %rs46}, %r941;
	cvt.f32.bf16 	%r1114, %rs46;
	cvt.f32.bf16 	%r1115, %rs45;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1116, %r1113, %r585, %r1115;
	fma.rn.f32 	%r1117, %r1112, %r586, %r1114;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r713, %r1117, %r1116;
	cvt.rn.bf16x2.f32 	%r709, %r1081, %r1080;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1118, %r1502, %r866;
	mul.f32 	%r1119, %r1501, %r866;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs47, %rs48}, %r957;
	cvt.f32.bf16 	%r1120, %rs48;
	cvt.f32.bf16 	%r1121, %rs47;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1122, %r1119, %r585, %r1121;
	fma.rn.f32 	%r1123, %r1118, %r586, %r1120;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r714, %r1123, %r1122;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1124, %r1504, %r865;
	mul.f32 	%r1125, %r1503, %r865;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs49, %rs50}, %r958;
	cvt.f32.bf16 	%r1126, %rs50;
	cvt.f32.bf16 	%r1127, %rs49;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1128, %r1125, %r587, %r1127;
	fma.rn.f32 	%r1129, %r1124, %r588, %r1126;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1130, %r1408, %r859;
	mul.f32 	%r1131, %r1407, %r859;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs51, %rs52}, %r910;
	cvt.f32.bf16 	%r1132, %rs52;
	cvt.f32.bf16 	%r1133, %rs51;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1134, %r1131, %r587, %r1133;
	fma.rn.f32 	%r1135, %r1130, %r588, %r1132;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r726, %r1135, %r1134;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1136, %r1410, %r860;
	mul.f32 	%r1137, %r1409, %r860;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs53, %rs54}, %r911;
	cvt.f32.bf16 	%r1138, %rs54;
	cvt.f32.bf16 	%r1139, %rs53;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1140, %r1137, %r587, %r1139;
	fma.rn.f32 	%r1141, %r1136, %r588, %r1138;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r731, %r1141, %r1140;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1142, %r1440, %r861;
	mul.f32 	%r1143, %r1439, %r861;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs55, %rs56}, %r926;
	cvt.f32.bf16 	%r1144, %rs56;
	cvt.f32.bf16 	%r1145, %rs55;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1146, %r1143, %r587, %r1145;
	fma.rn.f32 	%r1147, %r1142, %r588, %r1144;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r727, %r1147, %r1146;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1148, %r1442, %r862;
	mul.f32 	%r1149, %r1441, %r862;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs57, %rs58}, %r927;
	cvt.f32.bf16 	%r1150, %rs58;
	cvt.f32.bf16 	%r1151, %rs57;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1152, %r1149, %r587, %r1151;
	fma.rn.f32 	%r1153, %r1148, %r588, %r1150;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r732, %r1153, %r1152;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1154, %r1472, %r863;
	mul.f32 	%r1155, %r1471, %r863;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs59, %rs60}, %r942;
	cvt.f32.bf16 	%r1156, %rs60;
	cvt.f32.bf16 	%r1157, %rs59;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1158, %r1155, %r587, %r1157;
	fma.rn.f32 	%r1159, %r1154, %r588, %r1156;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r728, %r1159, %r1158;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1160, %r1474, %r864;
	mul.f32 	%r1161, %r1473, %r864;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs61, %rs62}, %r943;
	cvt.f32.bf16 	%r1162, %rs62;
	cvt.f32.bf16 	%r1163, %rs61;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1164, %r1161, %r587, %r1163;
	fma.rn.f32 	%r1165, %r1160, %r588, %r1162;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r733, %r1165, %r1164;
	cvt.rn.bf16x2.f32 	%r729, %r1129, %r1128;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1166, %r1506, %r866;
	mul.f32 	%r1167, %r1505, %r866;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs63, %rs64}, %r959;
	cvt.f32.bf16 	%r1168, %rs64;
	cvt.f32.bf16 	%r1169, %rs63;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1170, %r1167, %r587, %r1169;
	fma.rn.f32 	%r1171, %r1166, %r588, %r1168;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r734, %r1171, %r1170;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1172, %r1508, %r865;
	mul.f32 	%r1173, %r1507, %r865;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs65, %rs66}, %r952;
	cvt.f32.bf16 	%r1174, %rs66;
	cvt.f32.bf16 	%r1175, %rs65;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1176, %r1173, %r589, %r1175;
	fma.rn.f32 	%r1177, %r1172, %r590, %r1174;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1178, %r1412, %r859;
	mul.f32 	%r1179, %r1411, %r859;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs67, %rs68}, %r902;
	cvt.f32.bf16 	%r1180, %rs68;
	cvt.f32.bf16 	%r1181, %rs67;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1182, %r1179, %r589, %r1181;
	fma.rn.f32 	%r1183, %r1178, %r590, %r1180;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r676, %r1183, %r1182;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1184, %r1414, %r860;
	mul.f32 	%r1185, %r1413, %r860;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs69, %rs70}, %r903;
	cvt.f32.bf16 	%r1186, %rs70;
	cvt.f32.bf16 	%r1187, %rs69;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1188, %r1185, %r589, %r1187;
	fma.rn.f32 	%r1189, %r1184, %r590, %r1186;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r681, %r1189, %r1188;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1190, %r1444, %r861;
	mul.f32 	%r1191, %r1443, %r861;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs71, %rs72}, %r920;
	cvt.f32.bf16 	%r1192, %rs72;
	cvt.f32.bf16 	%r1193, %rs71;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1194, %r1191, %r589, %r1193;
	fma.rn.f32 	%r1195, %r1190, %r590, %r1192;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r677, %r1195, %r1194;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1196, %r1446, %r862;
	mul.f32 	%r1197, %r1445, %r862;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs73, %rs74}, %r921;
	cvt.f32.bf16 	%r1198, %rs74;
	cvt.f32.bf16 	%r1199, %rs73;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1200, %r1197, %r589, %r1199;
	fma.rn.f32 	%r1201, %r1196, %r590, %r1198;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r682, %r1201, %r1200;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1202, %r1476, %r863;
	mul.f32 	%r1203, %r1475, %r863;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs75, %rs76}, %r936;
	cvt.f32.bf16 	%r1204, %rs76;
	cvt.f32.bf16 	%r1205, %rs75;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1206, %r1203, %r589, %r1205;
	fma.rn.f32 	%r1207, %r1202, %r590, %r1204;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r678, %r1207, %r1206;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1208, %r1478, %r864;
	mul.f32 	%r1209, %r1477, %r864;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs77, %rs78}, %r937;
	cvt.f32.bf16 	%r1210, %rs78;
	cvt.f32.bf16 	%r1211, %rs77;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1212, %r1209, %r589, %r1211;
	fma.rn.f32 	%r1213, %r1208, %r590, %r1210;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r683, %r1213, %r1212;
	cvt.rn.bf16x2.f32 	%r679, %r1177, %r1176;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1214, %r1510, %r866;
	mul.f32 	%r1215, %r1509, %r866;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs79, %rs80}, %r953;
	cvt.f32.bf16 	%r1216, %rs80;
	cvt.f32.bf16 	%r1217, %rs79;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1218, %r1215, %r589, %r1217;
	fma.rn.f32 	%r1219, %r1214, %r590, %r1216;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r684, %r1219, %r1218;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1220, %r1512, %r865;
	mul.f32 	%r1221, %r1511, %r865;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs81, %rs82}, %r954;
	cvt.f32.bf16 	%r1222, %rs82;
	cvt.f32.bf16 	%r1223, %rs81;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1224, %r1221, %r591, %r1223;
	fma.rn.f32 	%r1225, %r1220, %r592, %r1222;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1226, %r1416, %r859;
	mul.f32 	%r1227, %r1415, %r859;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs83, %rs84}, %r904;
	cvt.f32.bf16 	%r1228, %rs84;
	cvt.f32.bf16 	%r1229, %rs83;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1230, %r1227, %r591, %r1229;
	fma.rn.f32 	%r1231, %r1226, %r592, %r1228;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r696, %r1231, %r1230;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1232, %r1418, %r860;
	mul.f32 	%r1233, %r1417, %r860;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs85, %rs86}, %r905;
	cvt.f32.bf16 	%r1234, %rs86;
	cvt.f32.bf16 	%r1235, %rs85;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1236, %r1233, %r591, %r1235;
	fma.rn.f32 	%r1237, %r1232, %r592, %r1234;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r701, %r1237, %r1236;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1238, %r1448, %r861;
	mul.f32 	%r1239, %r1447, %r861;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs87, %rs88}, %r922;
	cvt.f32.bf16 	%r1240, %rs88;
	cvt.f32.bf16 	%r1241, %rs87;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1242, %r1239, %r591, %r1241;
	fma.rn.f32 	%r1243, %r1238, %r592, %r1240;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r697, %r1243, %r1242;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1244, %r1450, %r862;
	mul.f32 	%r1245, %r1449, %r862;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs89, %rs90}, %r923;
	cvt.f32.bf16 	%r1246, %rs90;
	cvt.f32.bf16 	%r1247, %rs89;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1248, %r1245, %r591, %r1247;
	fma.rn.f32 	%r1249, %r1244, %r592, %r1246;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r702, %r1249, %r1248;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1250, %r1480, %r863;
	mul.f32 	%r1251, %r1479, %r863;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs91, %rs92}, %r938;
	cvt.f32.bf16 	%r1252, %rs92;
	cvt.f32.bf16 	%r1253, %rs91;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1254, %r1251, %r591, %r1253;
	fma.rn.f32 	%r1255, %r1250, %r592, %r1252;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r698, %r1255, %r1254;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1256, %r1482, %r864;
	mul.f32 	%r1257, %r1481, %r864;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs93, %rs94}, %r939;
	cvt.f32.bf16 	%r1258, %rs94;
	cvt.f32.bf16 	%r1259, %rs93;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1260, %r1257, %r591, %r1259;
	fma.rn.f32 	%r1261, %r1256, %r592, %r1258;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r703, %r1261, %r1260;
	cvt.rn.bf16x2.f32 	%r699, %r1225, %r1224;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1262, %r1514, %r866;
	mul.f32 	%r1263, %r1513, %r866;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs95, %rs96}, %r955;
	cvt.f32.bf16 	%r1264, %rs96;
	cvt.f32.bf16 	%r1265, %rs95;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1266, %r1263, %r591, %r1265;
	fma.rn.f32 	%r1267, %r1262, %r592, %r1264;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r704, %r1267, %r1266;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1268, %r1516, %r865;
	mul.f32 	%r1269, %r1515, %r865;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs97, %rs98}, %r960;
	cvt.f32.bf16 	%r1270, %rs98;
	cvt.f32.bf16 	%r1271, %rs97;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1272, %r1269, %r593, %r1271;
	fma.rn.f32 	%r1273, %r1268, %r594, %r1270;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1274, %r1420, %r859;
	mul.f32 	%r1275, %r1419, %r859;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs99, %rs100}, %r912;
	cvt.f32.bf16 	%r1276, %rs100;
	cvt.f32.bf16 	%r1277, %rs99;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1278, %r1275, %r593, %r1277;
	fma.rn.f32 	%r1279, %r1274, %r594, %r1276;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r716, %r1279, %r1278;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1280, %r1422, %r860;
	mul.f32 	%r1281, %r1421, %r860;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs101, %rs102}, %r913;
	cvt.f32.bf16 	%r1282, %rs102;
	cvt.f32.bf16 	%r1283, %rs101;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1284, %r1281, %r593, %r1283;
	fma.rn.f32 	%r1285, %r1280, %r594, %r1282;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r721, %r1285, %r1284;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1286, %r1452, %r861;
	mul.f32 	%r1287, %r1451, %r861;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs103, %rs104}, %r928;
	cvt.f32.bf16 	%r1288, %rs104;
	cvt.f32.bf16 	%r1289, %rs103;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1290, %r1287, %r593, %r1289;
	fma.rn.f32 	%r1291, %r1286, %r594, %r1288;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r717, %r1291, %r1290;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1292, %r1454, %r862;
	mul.f32 	%r1293, %r1453, %r862;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs105, %rs106}, %r929;
	cvt.f32.bf16 	%r1294, %rs106;
	cvt.f32.bf16 	%r1295, %rs105;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1296, %r1293, %r593, %r1295;
	fma.rn.f32 	%r1297, %r1292, %r594, %r1294;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r722, %r1297, %r1296;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1298, %r1484, %r863;
	mul.f32 	%r1299, %r1483, %r863;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs107, %rs108}, %r944;
	cvt.f32.bf16 	%r1300, %rs108;
	cvt.f32.bf16 	%r1301, %rs107;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1302, %r1299, %r593, %r1301;
	fma.rn.f32 	%r1303, %r1298, %r594, %r1300;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r718, %r1303, %r1302;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1304, %r1486, %r864;
	mul.f32 	%r1305, %r1485, %r864;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs109, %rs110}, %r945;
	cvt.f32.bf16 	%r1306, %rs110;
	cvt.f32.bf16 	%r1307, %rs109;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1308, %r1305, %r593, %r1307;
	fma.rn.f32 	%r1309, %r1304, %r594, %r1306;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r723, %r1309, %r1308;
	cvt.rn.bf16x2.f32 	%r719, %r1273, %r1272;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1310, %r1518, %r866;
	mul.f32 	%r1311, %r1517, %r866;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs111, %rs112}, %r961;
	cvt.f32.bf16 	%r1312, %rs112;
	cvt.f32.bf16 	%r1313, %rs111;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1314, %r1311, %r593, %r1313;
	fma.rn.f32 	%r1315, %r1310, %r594, %r1312;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r724, %r1315, %r1314;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1316, %r1520, %r865;
	mul.f32 	%r1317, %r1519, %r865;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs113, %rs114}, %r962;
	cvt.f32.bf16 	%r1318, %rs114;
	cvt.f32.bf16 	%r1319, %rs113;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1320, %r1317, %r595, %r1319;
	fma.rn.f32 	%r1321, %r1316, %r596, %r1318;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1322, %r1424, %r859;
	mul.f32 	%r1323, %r1423, %r859;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs115, %rs116}, %r914;
	cvt.f32.bf16 	%r1324, %rs116;
	cvt.f32.bf16 	%r1325, %rs115;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1326, %r1323, %r595, %r1325;
	fma.rn.f32 	%r1327, %r1322, %r596, %r1324;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r736, %r1327, %r1326;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1328, %r1426, %r860;
	mul.f32 	%r1329, %r1425, %r860;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs117, %rs118}, %r915;
	cvt.f32.bf16 	%r1330, %rs118;
	cvt.f32.bf16 	%r1331, %rs117;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1332, %r1329, %r595, %r1331;
	fma.rn.f32 	%r1333, %r1328, %r596, %r1330;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r741, %r1333, %r1332;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1334, %r1456, %r861;
	mul.f32 	%r1335, %r1455, %r861;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs119, %rs120}, %r930;
	cvt.f32.bf16 	%r1336, %rs120;
	cvt.f32.bf16 	%r1337, %rs119;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1338, %r1335, %r595, %r1337;
	fma.rn.f32 	%r1339, %r1334, %r596, %r1336;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r737, %r1339, %r1338;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1340, %r1458, %r862;
	mul.f32 	%r1341, %r1457, %r862;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs121, %rs122}, %r931;
	cvt.f32.bf16 	%r1342, %rs122;
	cvt.f32.bf16 	%r1343, %rs121;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1344, %r1341, %r595, %r1343;
	fma.rn.f32 	%r1345, %r1340, %r596, %r1342;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r742, %r1345, %r1344;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1346, %r1488, %r863;
	mul.f32 	%r1347, %r1487, %r863;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs123, %rs124}, %r946;
	cvt.f32.bf16 	%r1348, %rs124;
	cvt.f32.bf16 	%r1349, %rs123;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1350, %r1347, %r595, %r1349;
	fma.rn.f32 	%r1351, %r1346, %r596, %r1348;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r738, %r1351, %r1350;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1352, %r1490, %r864;
	mul.f32 	%r1353, %r1489, %r864;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs125, %rs126}, %r947;
	cvt.f32.bf16 	%r1354, %rs126;
	cvt.f32.bf16 	%r1355, %rs125;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1356, %r1353, %r595, %r1355;
	fma.rn.f32 	%r1357, %r1352, %r596, %r1354;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r743, %r1357, %r1356;
	cvt.rn.bf16x2.f32 	%r739, %r1321, %r1320;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1358, %r1522, %r866;
	mul.f32 	%r1359, %r1521, %r866;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs127, %rs128}, %r963;
	cvt.f32.bf16 	%r1360, %rs128;
	cvt.f32.bf16 	%r1361, %rs127;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1362, %r1359, %r595, %r1361;
	fma.rn.f32 	%r1363, %r1358, %r596, %r1360;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r744, %r1363, %r1362;
	bar.sync 	0;
	shl.b32 	%r1364, %r5, 14;
	shl.b32 	%r1365, %r5, 5;
	and.b32 	%r1366, %r1393, 3456;
	bfe.s32 	%r1367, %r2, 2, 1;
	and.b32 	%r1368, %r1367, 8208;
	or.b32 	%r1369, %r1365, %r1366;
	xor.b32 	%r1370, %r1368, %r892;
	or.b32 	%r1371, %r1370, %r1369;
	or.b32 	%r1372, %r1371, %r1364;
	add.s32 	%r665, %r193, %r1372;
	// begin inline asm
	st.shared.v4.b32 [ %r665 + 0 ], { %r666, %r667, %r668, %r669 };
	// end inline asm
	add.s32 	%r670, %r665, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r670 + 0 ], { %r671, %r672, %r673, %r674 };
	// end inline asm
	add.s32 	%r675, %r665, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r675 + 0 ], { %r676, %r677, %r678, %r679 };
	// end inline asm
	add.s32 	%r680, %r665, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r680 + 0 ], { %r681, %r682, %r683, %r684 };
	// end inline asm
	xor.b32 	%r1373, %r1372, 32;
	add.s32 	%r685, %r193, %r1373;
	// begin inline asm
	st.shared.v4.b32 [ %r685 + 0 ], { %r686, %r687, %r688, %r689 };
	// end inline asm
	add.s32 	%r690, %r685, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r690 + 0 ], { %r691, %r692, %r693, %r694 };
	// end inline asm
	add.s32 	%r695, %r685, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r695 + 0 ], { %r696, %r697, %r698, %r699 };
	// end inline asm
	add.s32 	%r700, %r685, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r700 + 0 ], { %r701, %r702, %r703, %r704 };
	// end inline asm
	xor.b32 	%r1374, %r1372, 64;
	add.s32 	%r705, %r193, %r1374;
	// begin inline asm
	st.shared.v4.b32 [ %r705 + 0 ], { %r706, %r707, %r708, %r709 };
	// end inline asm
	add.s32 	%r710, %r705, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r710 + 0 ], { %r711, %r712, %r713, %r714 };
	// end inline asm
	add.s32 	%r715, %r705, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r715 + 0 ], { %r716, %r717, %r718, %r719 };
	// end inline asm
	add.s32 	%r720, %r705, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r720 + 0 ], { %r721, %r722, %r723, %r724 };
	// end inline asm
	xor.b32 	%r1375, %r1372, 96;
	add.s32 	%r725, %r193, %r1375;
	// begin inline asm
	st.shared.v4.b32 [ %r725 + 0 ], { %r726, %r727, %r728, %r729 };
	// end inline asm
	add.s32 	%r730, %r725, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r730 + 0 ], { %r731, %r732, %r733, %r734 };
	// end inline asm
	add.s32 	%r735, %r725, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r735 + 0 ], { %r736, %r737, %r738, %r739 };
	// end inline asm
	add.s32 	%r740, %r725, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r740 + 0 ], { %r741, %r742, %r743, %r744 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r1376, %r2, 2;
	and.b32 	%r1377, %r1376, 896;
	shl.b32 	%r1378, %r851, 9;
	selp.b32 	%r1379, 0, 8208, %p24;
	or.b32 	%r1380, %r885, %r1377;
	xor.b32 	%r1381, %r1380, %r1379;
	or.b32 	%r1382, %r1381, %r1378;
	add.s32 	%r1383, %r193, %r1382;
	ld.shared.v4.b32 	{%r745, %r761, %r777, %r793}, [%r1383];
	ld.shared.v4.b32 	{%r749, %r765, %r781, %r797}, [%r1383+1024];
	ld.shared.v4.b32 	{%r753, %r769, %r785, %r801}, [%r1383+2048];
	ld.shared.v4.b32 	{%r757, %r773, %r789, %r805}, [%r1383+3072];
	xor.b32 	%r1384, %r1382, 32;
	add.s32 	%r1385, %r193, %r1384;
	ld.shared.v4.b32 	{%r746, %r762, %r778, %r794}, [%r1385+16384];
	ld.shared.v4.b32 	{%r750, %r766, %r782, %r798}, [%r1385+17408];
	ld.shared.v4.b32 	{%r754, %r770, %r786, %r802}, [%r1385+18432];
	ld.shared.v4.b32 	{%r758, %r774, %r790, %r806}, [%r1385+19456];
	xor.b32 	%r1386, %r1382, 64;
	add.s32 	%r1387, %r193, %r1386;
	ld.shared.v4.b32 	{%r747, %r763, %r779, %r795}, [%r1387+32768];
	ld.shared.v4.b32 	{%r751, %r767, %r783, %r799}, [%r1387+33792];
	ld.shared.v4.b32 	{%r755, %r771, %r787, %r803}, [%r1387+34816];
	ld.shared.v4.b32 	{%r759, %r775, %r791, %r807}, [%r1387+35840];
	xor.b32 	%r1388, %r1382, 96;
	add.s32 	%r1389, %r193, %r1388;
	ld.shared.v4.b32 	{%r748, %r764, %r780, %r796}, [%r1389+49152];
	ld.shared.v4.b32 	{%r752, %r768, %r784, %r800}, [%r1389+50176];
	ld.shared.v4.b32 	{%r756, %r772, %r788, %r804}, [%r1389+51200];
	ld.shared.v4.b32 	{%r760, %r776, %r792, %r808}, [%r1389+52224];
	.loc	1 356 8                         // sk05_mlp_gateup.py:356:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd108 + 0 ], { %r745, %r746, %r747, %r748 };
	// end inline asm
	// begin inline asm
	@%p9 st.global.v4.b32 [ %rd109 + 0 ], { %r749, %r750, %r751, %r752 };
	// end inline asm
	// begin inline asm
	@%p10 st.global.v4.b32 [ %rd110 + 0 ], { %r753, %r754, %r755, %r756 };
	// end inline asm
	// begin inline asm
	@%p11 st.global.v4.b32 [ %rd111 + 0 ], { %r757, %r758, %r759, %r760 };
	// end inline asm
	// begin inline asm
	@%p12 st.global.v4.b32 [ %rd112 + 0 ], { %r761, %r762, %r763, %r764 };
	// end inline asm
	// begin inline asm
	@%p13 st.global.v4.b32 [ %rd113 + 0 ], { %r765, %r766, %r767, %r768 };
	// end inline asm
	// begin inline asm
	@%p14 st.global.v4.b32 [ %rd114 + 0 ], { %r769, %r770, %r771, %r772 };
	// end inline asm
	// begin inline asm
	@%p15 st.global.v4.b32 [ %rd115 + 0 ], { %r773, %r774, %r775, %r776 };
	// end inline asm
	// begin inline asm
	@%p16 st.global.v4.b32 [ %rd116 + 0 ], { %r777, %r778, %r779, %r780 };
	// end inline asm
	// begin inline asm
	@%p17 st.global.v4.b32 [ %rd117 + 0 ], { %r781, %r782, %r783, %r784 };
	// end inline asm
	// begin inline asm
	@%p18 st.global.v4.b32 [ %rd118 + 0 ], { %r785, %r786, %r787, %r788 };
	// end inline asm
	// begin inline asm
	@%p19 st.global.v4.b32 [ %rd119 + 0 ], { %r789, %r790, %r791, %r792 };
	// end inline asm
	// begin inline asm
	@%p20 st.global.v4.b32 [ %rd120 + 0 ], { %r793, %r794, %r795, %r796 };
	// end inline asm
	// begin inline asm
	@%p21 st.global.v4.b32 [ %rd121 + 0 ], { %r797, %r798, %r799, %r800 };
	// end inline asm
	// begin inline asm
	@%p22 st.global.v4.b32 [ %rd122 + 0 ], { %r801, %r802, %r803, %r804 };
	// end inline asm
	// begin inline asm
	@%p23 st.global.v4.b32 [ %rd123 + 0 ], { %r805, %r806, %r807, %r808 };
	// end inline asm
	.loc	1 354 4                         // sk05_mlp_gateup.py:354:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/usr/local/lib/python3.12/dist-packages/vllm/_genesis/kernels/sk05_mlp_gateup.py"
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
.b32 201                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0xc2 DW_TAG_compile_unit
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
.b8 53
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 103
.b8 97
.b8 116
.b8 101
.b8 117
.b8 112
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 117
.b8 115
.b8 114
.b8 47
.b8 108
.b8 111
.b8 99
.b8 97
.b8 108
.b8 47
.b8 108
.b8 105
.b8 98
.b8 47
.b8 112
.b8 121
.b8 116
.b8 104
.b8 111
.b8 110
.b8 51
.b8 46
.b8 49
.b8 50
.b8 47
.b8 100
.b8 105
.b8 115
.b8 116
.b8 45
.b8 112
.b8 97
.b8 99
.b8 107
.b8 97
.b8 103
.b8 101
.b8 115
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
.b8 2                                   // Abbrev [2] 0x6a:0x1a DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 53
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 103
.b8 97
.b8 116
.b8 101
.b8 117
.b8 112
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x84:0x48 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 106                                // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x99:0x19 DW_TAG_inlined_subroutine
.b32 106                                // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 62                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0xb2:0x19 DW_TAG_inlined_subroutine
.b32 106                                // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 63                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_4 = _Nativo(
    "sk05_mlp_gateup/tile256x128x64_shift0_abi15",
    _PTX_4, "_sk05_mlp_gateup_kernel",
    warps=8, shared=73728,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 18],
    horneado={11: 1, 12: 1, 15: 1, 17: 1, 19: 1, 20: 256, 21: 128, 22: 64, 23: 8, 24: 128},
    div16=[8, 9, 10, 13, 14, 16],
)

_PTX_5 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk05_mlp_gateup_kernel // -- Begin function _sk05_mlp_gateup_kernel
.extern .shared .align 16 .b8 global_smem[];
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
                                        // @_sk05_mlp_gateup_kernel
.visible .entry _sk05_mlp_gateup_kernel(
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_6,
	.param .u32 _sk05_mlp_gateup_kernel_param_7,
	.param .u32 _sk05_mlp_gateup_kernel_param_8,
	.param .u32 _sk05_mlp_gateup_kernel_param_9,
	.param .u32 _sk05_mlp_gateup_kernel_param_10,
	.param .u32 _sk05_mlp_gateup_kernel_param_11,
	.param .u32 _sk05_mlp_gateup_kernel_param_12,
	.param .u32 _sk05_mlp_gateup_kernel_param_13,
	.param .u32 _sk05_mlp_gateup_kernel_param_14,
	.param .u32 _sk05_mlp_gateup_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_16,
	.param .u64 .ptr .global .align 1 _sk05_mlp_gateup_kernel_param_17
)
.reqntid 256
{
	.reg .pred 	%p<42>;
	.reg .b16 	%rs<257>;
	.reg .b32 	%r<1546>;
	.reg .b64 	%rd<283>;
	.loc	1 307 0                         // sk05_mlp_gateup.py:307:0
$L__func_begin0:
	.loc	1 307 0                         // sk05_mlp_gateup.py:307:0

// %bb.0:
	ld.param.b32 	%r27, [_sk05_mlp_gateup_kernel_param_14];
	ld.param.b32 	%r26, [_sk05_mlp_gateup_kernel_param_13];
	ld.param.b32 	%r25, [_sk05_mlp_gateup_kernel_param_12];
	ld.param.b32 	%r24, [_sk05_mlp_gateup_kernel_param_8];
	ld.param.b32 	%r23, [_sk05_mlp_gateup_kernel_param_7];
	ld.param.b64 	%rd29, [_sk05_mlp_gateup_kernel_param_5];
	ld.param.b64 	%rd28, [_sk05_mlp_gateup_kernel_param_4];
	ld.param.b64 	%rd27, [_sk05_mlp_gateup_kernel_param_3];
	ld.param.b64 	%rd26, [_sk05_mlp_gateup_kernel_param_2];
	ld.param.b64 	%rd25, [_sk05_mlp_gateup_kernel_param_1];
	ld.param.b64 	%rd24, [_sk05_mlp_gateup_kernel_param_0];
$L__tmp0:
	.loc	1 317 24                        // sk05_mlp_gateup.py:317:24
	mov.u32 	%r50, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk05_mlp_gateup.py:318:27 ]
	add.s32 	%r51, %r23, 255;
	.loc	2 43 30                         // standard.py:43:30 @[ sk05_mlp_gateup.py:318:27 ]
	shr.s32 	%r52, %r51, 31;
	shr.u32 	%r53, %r52, 24;
	add.s32 	%r54, %r51, %r53;
	shr.s32 	%r55, %r54, 8;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk05_mlp_gateup.py:319:27 ]
	add.s32 	%r56, %r24, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk05_mlp_gateup.py:319:27 ]
	shr.s32 	%r57, %r56, 31;
	shr.u32 	%r58, %r57, 25;
	add.s32 	%r59, %r56, %r58;
	shr.s32 	%r60, %r59, 7;
$L__tmp3:
	.loc	1 320 29                        // sk05_mlp_gateup.py:320:29
	shl.b32 	%r61, %r60, 3;
	.loc	1 321 22                        // sk05_mlp_gateup.py:321:22
	div.s32 	%r62, %r50, %r61;
	.loc	1 321 38                        // sk05_mlp_gateup.py:321:38
	shl.b32 	%r63, %r62, 3;
	ld.param.b32 	%r64, [_sk05_mlp_gateup_kernel_param_9];
	.loc	1 322 30                        // sk05_mlp_gateup.py:322:30
	sub.s32 	%r65, %r55, %r63;
	ld.param.b32 	%r66, [_sk05_mlp_gateup_kernel_param_10];
	.loc	1 322 39                        // sk05_mlp_gateup.py:322:39
	min.s32 	%r67, %r65, 8;
	ld.param.b32 	%r68, [_sk05_mlp_gateup_kernel_param_11];
	.loc	1 323 30                        // sk05_mlp_gateup.py:323:30
	mul.lo.s32 	%r69, %r62, %r61;
	sub.s32 	%r70, %r50, %r69;
	.loc	1 324 36                        // sk05_mlp_gateup.py:324:36
	div.s32 	%r71, %r70, %r67;
	.loc	1 323 46                        // sk05_mlp_gateup.py:323:46
	mul.lo.s32 	%r72, %r71, %r67;
	sub.s32 	%r73, %r70, %r72;
	.loc	1 323 23                        // sk05_mlp_gateup.py:323:23
	add.s32 	%r74, %r73, %r63;
	.loc	1 326 22                        // sk05_mlp_gateup.py:326:22
	shl.b32 	%r1, %r74, 8;
	.loc	1 326 45                        // sk05_mlp_gateup.py:326:45
	mov.u32 	%r2, %tid.x;
	shr.u32 	%r75, %r2, 2;
	bfe.u32 	%r76, %r2, 2, 6;
	or.b32 	%r77, %r76, 64;
	and.b32 	%r3, %r2, 255;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r78, %r1, %r76;
	or.b32 	%r79, %r1, %r77;
	or.b32 	%r80, %r78, 128;
	or.b32 	%r81, %r1, %r75;
	or.b32 	%r82, %r81, 192;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r83, %r78, %r23;
	rem.s32 	%r84, %r79, %r23;
	rem.s32 	%r85, %r80, %r23;
	rem.s32 	%r86, %r82, %r23;
	.loc	1 327 22                        // sk05_mlp_gateup.py:327:22
	shl.b32 	%r4, %r71, 7;
	.loc	1 327 45                        // sk05_mlp_gateup.py:327:45
	and.b32 	%r5, %r2, 3;
	shl.b32 	%r87, %r5, 1;
	and.b32 	%r6, %r2, 32;
	shr.u32 	%r88, %r6, 2;
	or.b32 	%r89, %r88, %r87;
	and.b32 	%r7, %r2, 15;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r90, %r4, %r76;
	or.b32 	%r91, %r4, %r77;
	or.b32 	%r92, %r4, %r89;
	or.b32 	%r94, %r92, 16;
	or.b32 	%r96, %r92, 32;
	or.b32 	%r98, %r92, 48;
	or.b32 	%r100, %r92, 64;
	or.b32 	%r102, %r92, 80;
	or.b32 	%r104, %r92, 96;
	or.b32 	%r106, %r92, 112;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r108, %r90, %r24;
	rem.s32 	%r109, %r91, %r24;
	rem.s32 	%r8, %r92, %r24;
	rem.s32 	%r9, %r94, %r24;
	rem.s32 	%r10, %r96, %r24;
	rem.s32 	%r11, %r98, %r24;
	rem.s32 	%r12, %r100, %r24;
	rem.s32 	%r13, %r102, %r24;
	rem.s32 	%r14, %r104, %r24;
	rem.s32 	%r15, %r106, %r24;
	.loc	1 330 39                        // sk05_mlp_gateup.py:330:39
	mul.lo.s32 	%r118, %r83, %r66;
	mul.lo.s32 	%r119, %r84, %r66;
	mul.lo.s32 	%r120, %r85, %r66;
	mul.lo.s32 	%r121, %r86, %r66;
	.loc	1 330 21                        // sk05_mlp_gateup.py:330:21
	cvt.s64.s32 	%rd1, %r118;
	add.s64 	%rd49, %rd24, %rd1;
	cvt.s64.s32 	%rd2, %r119;
	add.s64 	%rd50, %rd24, %rd2;
	cvt.s64.s32 	%rd3, %r120;
	add.s64 	%rd51, %rd24, %rd3;
	cvt.s64.s32 	%rd4, %r121;
	add.s64 	%rd52, %rd24, %rd4;
	.loc	1 330 58                        // sk05_mlp_gateup.py:330:58
	shl.b32 	%r122, %r5, 4;
	.loc	1 330 51                        // sk05_mlp_gateup.py:330:51
	cvt.u64.u32 	%rd5, %r122;
	add.s64 	%rd30, %rd49, %rd5;
	add.s64 	%rd31, %rd50, %rd5;
	add.s64 	%rd32, %rd51, %rd5;
	add.s64 	%rd33, %rd52, %rd5;
	.loc	1 331 21                        // sk05_mlp_gateup.py:331:21
	add.s64 	%rd53, %rd25, %rd5;
	.loc	1 331 69                        // sk05_mlp_gateup.py:331:69
	mul.lo.s32 	%r123, %r108, %r68;
	mul.lo.s32 	%r124, %r109, %r68;
	.loc	1 331 51                        // sk05_mlp_gateup.py:331:51
	cvt.s64.s32 	%rd6, %r123;
	add.s64 	%rd34, %rd53, %rd6;
	cvt.s64.s32 	%rd7, %r124;
	add.s64 	%rd35, %rd53, %rd7;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	setp.gt.s32 	%p1, %r64, 63;
	.loc	1 344 30                        // sk05_mlp_gateup.py:344:30
	shl.b32 	%r192, %r3, 4;
	shl.b32 	%r17, %r2, 1;
	and.b32 	%r18, %r17, 48;
	xor.b32 	%r193, %r192, %r18;
	mov.b32 	%r194, global_smem;
	add.s32 	%r29, %r194, %r193;
	selp.b32 	%r30, 16, 0, %p1;
	// begin inline asm
	cp.async.cg.shared.global [ %r29 + 0 ], [ %rd30 + 0 ], 0x10, %r30;
	// end inline asm
	add.s32 	%r31, %r29, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r31 + 0 ], [ %rd31 + 0 ], 0x10, %r30;
	// end inline asm
	add.s32 	%r32, %r29, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r32 + 0 ], [ %rd32 + 0 ], 0x10, %r30;
	// end inline asm
	add.s32 	%r33, %r29, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r33 + 0 ], [ %rd33 + 0 ], 0x10, %r30;
	// end inline asm
	cp.async.commit_group;
	.loc	1 344 47                        // sk05_mlp_gateup.py:344:47
	add.s32 	%r34, %r29, 49152;
	// begin inline asm
	cp.async.cg.shared.global [ %r34 + 0 ], [ %rd34 + 0 ], 0x10, %r30;
	// end inline asm
	add.s32 	%r35, %r29, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r35 + 0 ], [ %rd35 + 0 ], 0x10, %r30;
	// end inline asm
	cp.async.commit_group;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	setp.gt.s32 	%p2, %r64, 127;
	.loc	1 345 18                        // sk05_mlp_gateup.py:345:18
	add.s64 	%rd36, %rd30, 64;
	add.s64 	%rd37, %rd31, 64;
	add.s64 	%rd38, %rd32, 64;
	add.s64 	%rd39, %rd33, 64;
	.loc	1 346 18                        // sk05_mlp_gateup.py:346:18
	add.s64 	%rd40, %rd34, 64;
	add.s64 	%rd41, %rd35, 64;
	.loc	1 344 30                        // sk05_mlp_gateup.py:344:30
	bar.sync 	0;
	add.s32 	%r36, %r29, 16384;
	selp.b32 	%r37, 16, 0, %p2;
	// begin inline asm
	cp.async.cg.shared.global [ %r36 + 0 ], [ %rd36 + 0 ], 0x10, %r37;
	// end inline asm
	add.s32 	%r38, %r29, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r38 + 0 ], [ %rd37 + 0 ], 0x10, %r37;
	// end inline asm
	add.s32 	%r39, %r29, 24576;
	// begin inline asm
	cp.async.cg.shared.global [ %r39 + 0 ], [ %rd38 + 0 ], 0x10, %r37;
	// end inline asm
	add.s32 	%r40, %r29, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r40 + 0 ], [ %rd39 + 0 ], 0x10, %r37;
	// end inline asm
	cp.async.commit_group;
	.loc	1 344 47                        // sk05_mlp_gateup.py:344:47
	add.s32 	%r41, %r29, 57344;
	// begin inline asm
	cp.async.cg.shared.global [ %r41 + 0 ], [ %rd40 + 0 ], 0x10, %r37;
	// end inline asm
	add.s32 	%r42, %r29, 61440;
	// begin inline asm
	cp.async.cg.shared.global [ %r42 + 0 ], [ %rd41 + 0 ], 0x10, %r37;
	// end inline asm
	cp.async.commit_group;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	setp.gt.s32 	%p3, %r64, 191;
	.loc	1 345 18                        // sk05_mlp_gateup.py:345:18
	add.s64 	%rd42, %rd30, 128;
	add.s64 	%rd43, %rd31, 128;
	add.s64 	%rd44, %rd32, 128;
	add.s64 	%rd45, %rd33, 128;
	.loc	1 346 18                        // sk05_mlp_gateup.py:346:18
	add.s64 	%rd46, %rd34, 128;
	add.s64 	%rd47, %rd35, 128;
	.loc	1 344 30                        // sk05_mlp_gateup.py:344:30
	bar.sync 	0;
	add.s32 	%r43, %r29, 32768;
	selp.b32 	%r44, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r43 + 0 ], [ %rd42 + 0 ], 0x10, %r44;
	// end inline asm
	add.s32 	%r45, %r29, 36864;
	// begin inline asm
	cp.async.cg.shared.global [ %r45 + 0 ], [ %rd43 + 0 ], 0x10, %r44;
	// end inline asm
	add.s32 	%r46, %r29, 40960;
	// begin inline asm
	cp.async.cg.shared.global [ %r46 + 0 ], [ %rd44 + 0 ], 0x10, %r44;
	// end inline asm
	add.s32 	%r47, %r29, 45056;
	// begin inline asm
	cp.async.cg.shared.global [ %r47 + 0 ], [ %rd45 + 0 ], 0x10, %r44;
	// end inline asm
	cp.async.commit_group;
	.loc	1 344 47                        // sk05_mlp_gateup.py:344:47
	add.s32 	%r48, %r29, 65536;
	// begin inline asm
	cp.async.cg.shared.global [ %r48 + 0 ], [ %rd46 + 0 ], 0x10, %r44;
	// end inline asm
	add.s32 	%r49, %r29, 69632;
	// begin inline asm
	cp.async.cg.shared.global [ %r49 + 0 ], [ %rd47 + 0 ], 0x10, %r44;
	// end inline asm
	cp.async.commit_group;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	@%p1 bra 	$L__BB0_2;
	bra.uni 	$L__BB0_1;
$L__BB0_2:                              // %.lr.ph
	.loc	1 0 23                          // sk05_mlp_gateup.py:0:23
	ld.param.b32 	%r28, [_sk05_mlp_gateup_kernel_param_15];
	ld.param.b64 	%rd48, [_sk05_mlp_gateup_kernel_param_6];
	or.b32 	%r93, %r92, 1;
	or.b32 	%r95, %r92, 17;
	or.b32 	%r97, %r92, 33;
	or.b32 	%r99, %r92, 49;
	or.b32 	%r101, %r92, 65;
	or.b32 	%r103, %r92, 81;
	or.b32 	%r105, %r92, 97;
	or.b32 	%r107, %r92, 113;
	rem.s32 	%r110, %r93, %r24;
	rem.s32 	%r111, %r95, %r24;
	rem.s32 	%r112, %r97, %r24;
	rem.s32 	%r113, %r99, %r24;
	rem.s32 	%r114, %r101, %r24;
	rem.s32 	%r115, %r103, %r24;
	rem.s32 	%r116, %r105, %r24;
	rem.s32 	%r117, %r107, %r24;
	shr.s32 	%r125, %r8, 31;
	shr.u32 	%r126, %r125, 25;
	add.s32 	%r127, %r8, %r126;
	shr.s32 	%r128, %r127, 7;
	shr.s32 	%r129, %r110, 31;
	shr.u32 	%r130, %r129, 25;
	add.s32 	%r131, %r110, %r130;
	shr.s32 	%r132, %r131, 7;
	shr.s32 	%r133, %r9, 31;
	shr.u32 	%r134, %r133, 25;
	add.s32 	%r135, %r9, %r134;
	shr.s32 	%r136, %r135, 7;
	shr.s32 	%r137, %r111, 31;
	shr.u32 	%r138, %r137, 25;
	add.s32 	%r139, %r111, %r138;
	shr.s32 	%r140, %r139, 7;
	shr.s32 	%r141, %r10, 31;
	shr.u32 	%r142, %r141, 25;
	add.s32 	%r143, %r10, %r142;
	shr.s32 	%r144, %r143, 7;
	shr.s32 	%r145, %r112, 31;
	shr.u32 	%r146, %r145, 25;
	add.s32 	%r147, %r112, %r146;
	shr.s32 	%r148, %r147, 7;
	shr.s32 	%r149, %r11, 31;
	shr.u32 	%r150, %r149, 25;
	add.s32 	%r151, %r11, %r150;
	shr.s32 	%r152, %r151, 7;
	shr.s32 	%r153, %r113, 31;
	shr.u32 	%r154, %r153, 25;
	add.s32 	%r155, %r113, %r154;
	shr.s32 	%r156, %r155, 7;
	shr.s32 	%r157, %r12, 31;
	shr.u32 	%r158, %r157, 25;
	add.s32 	%r159, %r12, %r158;
	shr.s32 	%r160, %r159, 7;
	shr.s32 	%r161, %r114, 31;
	shr.u32 	%r162, %r161, 25;
	add.s32 	%r163, %r114, %r162;
	shr.s32 	%r164, %r163, 7;
	shr.s32 	%r165, %r13, 31;
	shr.u32 	%r166, %r165, 25;
	add.s32 	%r167, %r13, %r166;
	shr.s32 	%r168, %r167, 7;
	shr.s32 	%r169, %r115, 31;
	shr.u32 	%r170, %r169, 25;
	add.s32 	%r171, %r115, %r170;
	shr.s32 	%r172, %r171, 7;
	shr.s32 	%r173, %r14, 31;
	shr.u32 	%r174, %r173, 25;
	add.s32 	%r175, %r14, %r174;
	shr.s32 	%r176, %r175, 7;
	shr.s32 	%r177, %r116, 31;
	shr.u32 	%r178, %r177, 25;
	add.s32 	%r179, %r116, %r178;
	shr.s32 	%r180, %r179, 7;
	shr.s32 	%r181, %r15, 31;
	shr.u32 	%r182, %r181, 25;
	add.s32 	%r183, %r15, %r182;
	shr.s32 	%r184, %r183, 7;
	shr.s32 	%r185, %r117, 31;
	shr.u32 	%r186, %r185, 25;
	add.s32 	%r187, %r117, %r186;
	shr.s32 	%r188, %r187, 7;
	mad.wide.s32 	%rd8, %r128, 4, %rd48;
	mad.wide.s32 	%rd9, %r132, 4, %rd48;
	mad.wide.s32 	%rd10, %r136, 4, %rd48;
	mad.wide.s32 	%rd11, %r140, 4, %rd48;
	mad.wide.s32 	%rd12, %r144, 4, %rd48;
	mad.wide.s32 	%rd13, %r148, 4, %rd48;
	mad.wide.s32 	%rd14, %r152, 4, %rd48;
	mad.wide.s32 	%rd15, %r156, 4, %rd48;
	mad.wide.s32 	%rd16, %r160, 4, %rd48;
	mad.wide.s32 	%rd17, %r164, 4, %rd48;
	mad.wide.s32 	%rd18, %r168, 4, %rd48;
	mad.wide.s32 	%rd19, %r172, 4, %rd48;
	mad.wide.s32 	%rd20, %r176, 4, %rd48;
	mad.wide.s32 	%rd21, %r180, 4, %rd48;
	mad.wide.s32 	%rd22, %r184, 4, %rd48;
	mad.wide.s32 	%rd23, %r188, 4, %rd48;
	shr.s32 	%r189, %r64, 31;
	shr.u32 	%r190, %r189, 26;
	add.s32 	%r191, %r64, %r190;
	shr.s32 	%r16, %r191, 6;
	add.s32 	%r19, %r16, -3;
	shl.b32 	%r195, %r7, 6;
	shl.b32 	%r1416, %r2, 4;
	and.b32 	%r196, %r1416, 3072;
	shl.b32 	%r197, %r2, 3;
	and.b32 	%r198, %r197, 48;
	and.b32 	%r1417, %r2, 16;
	or.b32 	%r199, %r195, %r196;
	xor.b32 	%r200, %r198, %r1417;
	or.b32 	%r20, %r199, %r200;
	xor.b32 	%r21, %r20, 32;
	shl.b32 	%r201, %r2, 6;
	and.b32 	%r202, %r201, 448;
	shl.b32 	%r203, %r6, 4;
	or.b32 	%r204, %r202, %r198;
	xor.b32 	%r205, %r204, %r18;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	add.s32 	%r206, %r194, %r203;
	add.s32 	%r22, %r206, %r205;
	add.s64 	%rd54, %rd7, %rd25;
	add.s64 	%rd282, %rd54, 192;
	add.s64 	%rd55, %rd6, %rd25;
	add.s64 	%rd281, %rd55, 192;
	add.s64 	%rd56, %rd4, %rd24;
	add.s64 	%rd280, %rd56, 192;
	add.s64 	%rd57, %rd3, %rd24;
	add.s64 	%rd279, %rd57, 192;
	add.s64 	%rd58, %rd2, %rd24;
	add.s64 	%rd278, %rd58, 192;
	add.s64 	%rd59, %rd1, %rd24;
	add.s64 	%rd277, %rd59, 192;
	mov.b32 	%r1418, 0f00000000;
	mov.b32 	%r223, 0;
	mov.b32 	%r1414, 2;
	mov.b32 	%r1413, -1;
	mov.b32 	%r1415, %r223;
	mov.b32 	%r1419, %r1418;
	mov.b32 	%r1420, %r1418;
	mov.b32 	%r1421, %r1418;
	mov.b32 	%r1422, %r1418;
	mov.b32 	%r1423, %r1418;
	mov.b32 	%r1424, %r1418;
	mov.b32 	%r1425, %r1418;
	mov.b32 	%r1426, %r1418;
	mov.b32 	%r1427, %r1418;
	mov.b32 	%r1428, %r1418;
	mov.b32 	%r1429, %r1418;
	mov.b32 	%r1430, %r1418;
	mov.b32 	%r1431, %r1418;
	mov.b32 	%r1432, %r1418;
	mov.b32 	%r1433, %r1418;
	mov.b32 	%r1434, %r1418;
	mov.b32 	%r1435, %r1418;
	mov.b32 	%r1436, %r1418;
	mov.b32 	%r1437, %r1418;
	mov.b32 	%r1438, %r1418;
	mov.b32 	%r1439, %r1418;
	mov.b32 	%r1440, %r1418;
	mov.b32 	%r1441, %r1418;
	mov.b32 	%r1442, %r1418;
	mov.b32 	%r1443, %r1418;
	mov.b32 	%r1444, %r1418;
	mov.b32 	%r1445, %r1418;
	mov.b32 	%r1446, %r1418;
	mov.b32 	%r1447, %r1418;
	mov.b32 	%r1448, %r1418;
	mov.b32 	%r1449, %r1418;
	mov.b32 	%r1450, %r1418;
	mov.b32 	%r1451, %r1418;
	mov.b32 	%r1452, %r1418;
	mov.b32 	%r1453, %r1418;
	mov.b32 	%r1454, %r1418;
	mov.b32 	%r1455, %r1418;
	mov.b32 	%r1456, %r1418;
	mov.b32 	%r1457, %r1418;
	mov.b32 	%r1458, %r1418;
	mov.b32 	%r1459, %r1418;
	mov.b32 	%r1460, %r1418;
	mov.b32 	%r1461, %r1418;
	mov.b32 	%r1462, %r1418;
	mov.b32 	%r1463, %r1418;
	mov.b32 	%r1464, %r1418;
	mov.b32 	%r1465, %r1418;
	mov.b32 	%r1466, %r1418;
	mov.b32 	%r1467, %r1418;
	mov.b32 	%r1468, %r1418;
	mov.b32 	%r1469, %r1418;
	mov.b32 	%r1470, %r1418;
	mov.b32 	%r1471, %r1418;
	mov.b32 	%r1472, %r1418;
	mov.b32 	%r1473, %r1418;
	mov.b32 	%r1474, %r1418;
	mov.b32 	%r1475, %r1418;
	mov.b32 	%r1476, %r1418;
	mov.b32 	%r1477, %r1418;
	mov.b32 	%r1478, %r1418;
	mov.b32 	%r1479, %r1418;
	mov.b32 	%r1480, %r1418;
	mov.b32 	%r1481, %r1418;
	mov.b32 	%r1482, %r1418;
	mov.b32 	%r1483, %r1418;
	mov.b32 	%r1484, %r1418;
	mov.b32 	%r1485, %r1418;
	mov.b32 	%r1486, %r1418;
	mov.b32 	%r1487, %r1418;
	mov.b32 	%r1488, %r1418;
	mov.b32 	%r1489, %r1418;
	mov.b32 	%r1490, %r1418;
	mov.b32 	%r1491, %r1418;
	mov.b32 	%r1492, %r1418;
	mov.b32 	%r1493, %r1418;
	mov.b32 	%r1494, %r1418;
	mov.b32 	%r1495, %r1418;
	mov.b32 	%r1496, %r1418;
	mov.b32 	%r1497, %r1418;
	mov.b32 	%r1498, %r1418;
	mov.b32 	%r1499, %r1418;
	mov.b32 	%r1500, %r1418;
	mov.b32 	%r1501, %r1418;
	mov.b32 	%r1502, %r1418;
	mov.b32 	%r1503, %r1418;
	mov.b32 	%r1504, %r1418;
	mov.b32 	%r1505, %r1418;
	mov.b32 	%r1506, %r1418;
	mov.b32 	%r1507, %r1418;
	mov.b32 	%r1508, %r1418;
	mov.b32 	%r1509, %r1418;
	mov.b32 	%r1510, %r1418;
	mov.b32 	%r1511, %r1418;
	mov.b32 	%r1512, %r1418;
	mov.b32 	%r1513, %r1418;
	mov.b32 	%r1514, %r1418;
	mov.b32 	%r1515, %r1418;
	mov.b32 	%r1516, %r1418;
	mov.b32 	%r1517, %r1418;
	mov.b32 	%r1518, %r1418;
	mov.b32 	%r1519, %r1418;
	mov.b32 	%r1520, %r1418;
	mov.b32 	%r1521, %r1418;
	mov.b32 	%r1522, %r1418;
	mov.b32 	%r1523, %r1418;
	mov.b32 	%r1524, %r1418;
	mov.b32 	%r1525, %r1418;
	mov.b32 	%r1526, %r1418;
	mov.b32 	%r1527, %r1418;
	mov.b32 	%r1528, %r1418;
	mov.b32 	%r1529, %r1418;
	mov.b32 	%r1530, %r1418;
	mov.b32 	%r1531, %r1418;
	mov.b32 	%r1532, %r1418;
	mov.b32 	%r1533, %r1418;
	mov.b32 	%r1534, %r1418;
	mov.b32 	%r1535, %r1418;
	mov.b32 	%r1536, %r1418;
	mov.b32 	%r1537, %r1418;
	mov.b32 	%r1538, %r1418;
	mov.b32 	%r1539, %r1418;
	mov.b32 	%r1540, %r1418;
	mov.b32 	%r1541, %r1418;
	mov.b32 	%r1542, %r1418;
	mov.b32 	%r1543, %r1418;
	mov.b32 	%r1544, %r1418;
	mov.b32 	%r1545, %r1418;
$L__BB0_3:                              // %__nv_exp2f.exit
                                        // =>This Inner Loop Header: Depth=1
	setp.lt.s32 	%p4, %r1415, %r19;
	add.s32 	%r423, %r1413, 1;
	setp.gt.s32 	%p5, %r423, 2;
	selp.b32 	%r1413, 0, %r423, %p5;
	.loc	1 342 56                        // sk05_mlp_gateup.py:342:56
	bfe.u32 	%r424, %r1415, 1, 25;
	.loc	1 343 31                        // sk05_mlp_gateup.py:343:31
	mul.lo.s32 	%r425, %r424, %r28;
	.loc	1 342 39                        // sk05_mlp_gateup.py:342:39
	mul.wide.s32 	%rd82, %r425, 4;
	add.s64 	%rd60, %rd8, %rd82;
	add.s64 	%rd61, %rd9, %rd82;
	add.s64 	%rd62, %rd10, %rd82;
	add.s64 	%rd63, %rd11, %rd82;
	add.s64 	%rd64, %rd12, %rd82;
	add.s64 	%rd65, %rd13, %rd82;
	add.s64 	%rd66, %rd14, %rd82;
	add.s64 	%rd67, %rd15, %rd82;
	add.s64 	%rd68, %rd16, %rd82;
	add.s64 	%rd69, %rd17, %rd82;
	add.s64 	%rd70, %rd18, %rd82;
	add.s64 	%rd71, %rd19, %rd82;
	add.s64 	%rd72, %rd20, %rd82;
	add.s64 	%rd73, %rd21, %rd82;
	add.s64 	%rd74, %rd22, %rd82;
	add.s64 	%rd75, %rd23, %rd82;
	.loc	1 342 29                        // sk05_mlp_gateup.py:342:29
	// begin inline asm
	mov.u32 %r207, 0x0;
	ld.global.b32 { %r207 }, [ %rd60 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r208, 0x0;
	ld.global.b32 { %r208 }, [ %rd61 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r209, 0x0;
	ld.global.b32 { %r209 }, [ %rd62 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r210, 0x0;
	ld.global.b32 { %r210 }, [ %rd63 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r211, 0x0;
	ld.global.b32 { %r211 }, [ %rd64 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r212, 0x0;
	ld.global.b32 { %r212 }, [ %rd65 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r213, 0x0;
	ld.global.b32 { %r213 }, [ %rd66 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r214, 0x0;
	ld.global.b32 { %r214 }, [ %rd67 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r215, 0x0;
	ld.global.b32 { %r215 }, [ %rd68 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r216, 0x0;
	ld.global.b32 { %r216 }, [ %rd69 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r217, 0x0;
	ld.global.b32 { %r217 }, [ %rd70 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r218, 0x0;
	ld.global.b32 { %r218 }, [ %rd71 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r219, 0x0;
	ld.global.b32 { %r219 }, [ %rd72 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r220, 0x0;
	ld.global.b32 { %r220 }, [ %rd73 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r221, 0x0;
	ld.global.b32 { %r221 }, [ %rd74 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r222, 0x0;
	ld.global.b32 { %r222 }, [ %rd75 + 0 ];
	// end inline asm
	.loc	1 342 21                        // sk05_mlp_gateup.py:342:21
	ex2.approx.ftz.f32 	%r426, %r207;
	ex2.approx.ftz.f32 	%r427, %r208;
	ex2.approx.ftz.f32 	%r428, %r209;
	ex2.approx.ftz.f32 	%r429, %r210;
	ex2.approx.ftz.f32 	%r430, %r211;
	ex2.approx.ftz.f32 	%r431, %r212;
	ex2.approx.ftz.f32 	%r432, %r213;
	ex2.approx.ftz.f32 	%r433, %r214;
	ex2.approx.ftz.f32 	%r434, %r215;
	ex2.approx.ftz.f32 	%r435, %r216;
	ex2.approx.ftz.f32 	%r436, %r217;
	ex2.approx.ftz.f32 	%r437, %r218;
	ex2.approx.ftz.f32 	%r438, %r219;
	ex2.approx.ftz.f32 	%r439, %r220;
	ex2.approx.ftz.f32 	%r440, %r221;
	ex2.approx.ftz.f32 	%r441, %r222;
	.loc	1 344 30                        // sk05_mlp_gateup.py:344:30
	cp.async.wait_group 	4;
	bar.sync 	0;
	shl.b32 	%r442, %r1413, 14;
	add.s32 	%r443, %r194, %r442;
	add.s32 	%r444, %r443, %r20;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r224, %r225, %r226, %r227}, [%r444];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r244, %r245, %r246, %r247}, [%r444+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r248, %r249, %r250, %r251}, [%r444+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r252, %r253, %r254, %r255}, [%r444+12288];
	add.s32 	%r445, %r443, %r21;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r260, %r261, %r262, %r263}, [%r445];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r312, %r313, %r314, %r315}, [%r445+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r348, %r349, %r350, %r351}, [%r445+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r384, %r385, %r386, %r387}, [%r445+12288];
	.loc	1 344 47                        // sk05_mlp_gateup.py:344:47
	shl.b32 	%r446, %r1413, 13;
	add.s32 	%r447, %r22, %r446;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r228, %r229, %r264, %r265}, [%r447+49152];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r230, %r231, %r270, %r271}, [%r447+50176];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r232, %r233, %r276, %r277}, [%r447+51200];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r234, %r235, %r282, %r283}, [%r447+52224];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r236, %r237, %r288, %r289}, [%r447+53248];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r238, %r239, %r294, %r295}, [%r447+54272];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r240, %r241, %r300, %r301}, [%r447+55296];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r242, %r243, %r306, %r307}, [%r447+56320];
	.loc	1 344 39                        // sk05_mlp_gateup.py:344:39
	mov.b32 	%r256, %r223;
	mov.b32 	%r257, %r223;
	mov.b32 	%r258, %r223;
	mov.b32 	%r259, %r223;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r256, %r257, %r258, %r259 }, { %r224, %r225, %r226, %r227 }, { %r228, %r229 }, { %r256, %r257, %r258, %r259 };
	// end inline asm
	mov.b32 	%r266, %r223;
	mov.b32 	%r267, %r223;
	mov.b32 	%r268, %r223;
	mov.b32 	%r269, %r223;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r266, %r267, %r268, %r269 }, { %r224, %r225, %r226, %r227 }, { %r230, %r231 }, { %r266, %r267, %r268, %r269 };
	// end inline asm
	mov.b32 	%r272, %r223;
	mov.b32 	%r273, %r223;
	mov.b32 	%r274, %r223;
	mov.b32 	%r275, %r223;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r272, %r273, %r274, %r275 }, { %r224, %r225, %r226, %r227 }, { %r232, %r233 }, { %r272, %r273, %r274, %r275 };
	// end inline asm
	mov.b32 	%r278, %r223;
	mov.b32 	%r279, %r223;
	mov.b32 	%r280, %r223;
	mov.b32 	%r281, %r223;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r278, %r279, %r280, %r281 }, { %r224, %r225, %r226, %r227 }, { %r234, %r235 }, { %r278, %r279, %r280, %r281 };
	// end inline asm
	mov.b32 	%r284, %r223;
	mov.b32 	%r285, %r223;
	mov.b32 	%r286, %r223;
	mov.b32 	%r287, %r223;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r284, %r285, %r286, %r287 }, { %r224, %r225, %r226, %r227 }, { %r236, %r237 }, { %r284, %r285, %r286, %r287 };
	// end inline asm
	mov.b32 	%r290, %r223;
	mov.b32 	%r291, %r223;
	mov.b32 	%r292, %r223;
	mov.b32 	%r293, %r223;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r290, %r291, %r292, %r293 }, { %r224, %r225, %r226, %r227 }, { %r238, %r239 }, { %r290, %r291, %r292, %r293 };
	// end inline asm
	mov.b32 	%r296, %r223;
	mov.b32 	%r297, %r223;
	mov.b32 	%r298, %r223;
	mov.b32 	%r299, %r223;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r296, %r297, %r298, %r299 }, { %r224, %r225, %r226, %r227 }, { %r240, %r241 }, { %r296, %r297, %r298, %r299 };
	// end inline asm
	mov.b32 	%r302, %r223;
	mov.b32 	%r303, %r223;
	mov.b32 	%r304, %r223;
	mov.b32 	%r305, %r223;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r302, %r303, %r304, %r305 }, { %r224, %r225, %r226, %r227 }, { %r242, %r243 }, { %r302, %r303, %r304, %r305 };
	// end inline asm
	mov.b32 	%r308, %r223;
	mov.b32 	%r309, %r223;
	mov.b32 	%r310, %r223;
	mov.b32 	%r311, %r223;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r308, %r309, %r310, %r311 }, { %r244, %r245, %r246, %r247 }, { %r228, %r229 }, { %r308, %r309, %r310, %r311 };
	// end inline asm
	mov.b32 	%r316, %r223;
	mov.b32 	%r317, %r223;
	mov.b32 	%r318, %r223;
	mov.b32 	%r319, %r223;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r316, %r317, %r318, %r319 }, { %r244, %r245, %r246, %r247 }, { %r230, %r231 }, { %r316, %r317, %r318, %r319 };
	// end inline asm
	mov.b32 	%r320, %r223;
	mov.b32 	%r321, %r223;
	mov.b32 	%r322, %r223;
	mov.b32 	%r323, %r223;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r320, %r321, %r322, %r323 }, { %r244, %r245, %r246, %r247 }, { %r232, %r233 }, { %r320, %r321, %r322, %r323 };
	// end inline asm
	mov.b32 	%r324, %r223;
	mov.b32 	%r325, %r223;
	mov.b32 	%r326, %r223;
	mov.b32 	%r327, %r223;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r324, %r325, %r326, %r327 }, { %r244, %r245, %r246, %r247 }, { %r234, %r235 }, { %r324, %r325, %r326, %r327 };
	// end inline asm
	mov.b32 	%r328, %r223;
	mov.b32 	%r329, %r223;
	mov.b32 	%r330, %r223;
	mov.b32 	%r331, %r223;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r328, %r329, %r330, %r331 }, { %r244, %r245, %r246, %r247 }, { %r236, %r237 }, { %r328, %r329, %r330, %r331 };
	// end inline asm
	mov.b32 	%r332, %r223;
	mov.b32 	%r333, %r223;
	mov.b32 	%r334, %r223;
	mov.b32 	%r335, %r223;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r332, %r333, %r334, %r335 }, { %r244, %r245, %r246, %r247 }, { %r238, %r239 }, { %r332, %r333, %r334, %r335 };
	// end inline asm
	mov.b32 	%r336, %r223;
	mov.b32 	%r337, %r223;
	mov.b32 	%r338, %r223;
	mov.b32 	%r339, %r223;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r336, %r337, %r338, %r339 }, { %r244, %r245, %r246, %r247 }, { %r240, %r241 }, { %r336, %r337, %r338, %r339 };
	// end inline asm
	mov.b32 	%r340, %r223;
	mov.b32 	%r341, %r223;
	mov.b32 	%r342, %r223;
	mov.b32 	%r343, %r223;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r340, %r341, %r342, %r343 }, { %r244, %r245, %r246, %r247 }, { %r242, %r243 }, { %r340, %r341, %r342, %r343 };
	// end inline asm
	mov.b32 	%r344, %r223;
	mov.b32 	%r345, %r223;
	mov.b32 	%r346, %r223;
	mov.b32 	%r347, %r223;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r344, %r345, %r346, %r347 }, { %r248, %r249, %r250, %r251 }, { %r228, %r229 }, { %r344, %r345, %r346, %r347 };
	// end inline asm
	mov.b32 	%r352, %r223;
	mov.b32 	%r353, %r223;
	mov.b32 	%r354, %r223;
	mov.b32 	%r355, %r223;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r352, %r353, %r354, %r355 }, { %r248, %r249, %r250, %r251 }, { %r230, %r231 }, { %r352, %r353, %r354, %r355 };
	// end inline asm
	mov.b32 	%r356, %r223;
	mov.b32 	%r357, %r223;
	mov.b32 	%r358, %r223;
	mov.b32 	%r359, %r223;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r356, %r357, %r358, %r359 }, { %r248, %r249, %r250, %r251 }, { %r232, %r233 }, { %r356, %r357, %r358, %r359 };
	// end inline asm
	mov.b32 	%r360, %r223;
	mov.b32 	%r361, %r223;
	mov.b32 	%r362, %r223;
	mov.b32 	%r363, %r223;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r360, %r361, %r362, %r363 }, { %r248, %r249, %r250, %r251 }, { %r234, %r235 }, { %r360, %r361, %r362, %r363 };
	// end inline asm
	mov.b32 	%r364, %r223;
	mov.b32 	%r365, %r223;
	mov.b32 	%r366, %r223;
	mov.b32 	%r367, %r223;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r364, %r365, %r366, %r367 }, { %r248, %r249, %r250, %r251 }, { %r236, %r237 }, { %r364, %r365, %r366, %r367 };
	// end inline asm
	mov.b32 	%r368, %r223;
	mov.b32 	%r369, %r223;
	mov.b32 	%r370, %r223;
	mov.b32 	%r371, %r223;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r368, %r369, %r370, %r371 }, { %r248, %r249, %r250, %r251 }, { %r238, %r239 }, { %r368, %r369, %r370, %r371 };
	// end inline asm
	mov.b32 	%r372, %r223;
	mov.b32 	%r373, %r223;
	mov.b32 	%r374, %r223;
	mov.b32 	%r375, %r223;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r372, %r373, %r374, %r375 }, { %r248, %r249, %r250, %r251 }, { %r240, %r241 }, { %r372, %r373, %r374, %r375 };
	// end inline asm
	mov.b32 	%r376, %r223;
	mov.b32 	%r377, %r223;
	mov.b32 	%r378, %r223;
	mov.b32 	%r379, %r223;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r376, %r377, %r378, %r379 }, { %r248, %r249, %r250, %r251 }, { %r242, %r243 }, { %r376, %r377, %r378, %r379 };
	// end inline asm
	mov.b32 	%r380, %r223;
	mov.b32 	%r381, %r223;
	mov.b32 	%r382, %r223;
	mov.b32 	%r383, %r223;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r380, %r381, %r382, %r383 }, { %r252, %r253, %r254, %r255 }, { %r228, %r229 }, { %r380, %r381, %r382, %r383 };
	// end inline asm
	mov.b32 	%r388, %r223;
	mov.b32 	%r389, %r223;
	mov.b32 	%r390, %r223;
	mov.b32 	%r391, %r223;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r388, %r389, %r390, %r391 }, { %r252, %r253, %r254, %r255 }, { %r230, %r231 }, { %r388, %r389, %r390, %r391 };
	// end inline asm
	mov.b32 	%r392, %r223;
	mov.b32 	%r393, %r223;
	mov.b32 	%r394, %r223;
	mov.b32 	%r395, %r223;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r392, %r393, %r394, %r395 }, { %r252, %r253, %r254, %r255 }, { %r232, %r233 }, { %r392, %r393, %r394, %r395 };
	// end inline asm
	mov.b32 	%r396, %r223;
	mov.b32 	%r397, %r223;
	mov.b32 	%r398, %r223;
	mov.b32 	%r399, %r223;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r396, %r397, %r398, %r399 }, { %r252, %r253, %r254, %r255 }, { %r234, %r235 }, { %r396, %r397, %r398, %r399 };
	// end inline asm
	mov.b32 	%r400, %r223;
	mov.b32 	%r401, %r223;
	mov.b32 	%r402, %r223;
	mov.b32 	%r403, %r223;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r400, %r401, %r402, %r403 }, { %r252, %r253, %r254, %r255 }, { %r236, %r237 }, { %r400, %r401, %r402, %r403 };
	// end inline asm
	mov.b32 	%r404, %r223;
	mov.b32 	%r405, %r223;
	mov.b32 	%r406, %r223;
	mov.b32 	%r407, %r223;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r404, %r405, %r406, %r407 }, { %r252, %r253, %r254, %r255 }, { %r238, %r239 }, { %r404, %r405, %r406, %r407 };
	// end inline asm
	mov.b32 	%r408, %r223;
	mov.b32 	%r409, %r223;
	mov.b32 	%r410, %r223;
	mov.b32 	%r411, %r223;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r408, %r409, %r410, %r411 }, { %r252, %r253, %r254, %r255 }, { %r240, %r241 }, { %r408, %r409, %r410, %r411 };
	// end inline asm
	mov.b32 	%r412, %r223;
	mov.b32 	%r413, %r223;
	mov.b32 	%r414, %r223;
	mov.b32 	%r415, %r223;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r412, %r413, %r414, %r415 }, { %r252, %r253, %r254, %r255 }, { %r242, %r243 }, { %r412, %r413, %r414, %r415 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r256, %r257, %r258, %r259 }, { %r260, %r261, %r262, %r263 }, { %r264, %r265 }, { %r256, %r257, %r258, %r259 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r266, %r267, %r268, %r269 }, { %r260, %r261, %r262, %r263 }, { %r270, %r271 }, { %r266, %r267, %r268, %r269 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r272, %r273, %r274, %r275 }, { %r260, %r261, %r262, %r263 }, { %r276, %r277 }, { %r272, %r273, %r274, %r275 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r278, %r279, %r280, %r281 }, { %r260, %r261, %r262, %r263 }, { %r282, %r283 }, { %r278, %r279, %r280, %r281 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r284, %r285, %r286, %r287 }, { %r260, %r261, %r262, %r263 }, { %r288, %r289 }, { %r284, %r285, %r286, %r287 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r290, %r291, %r292, %r293 }, { %r260, %r261, %r262, %r263 }, { %r294, %r295 }, { %r290, %r291, %r292, %r293 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r296, %r297, %r298, %r299 }, { %r260, %r261, %r262, %r263 }, { %r300, %r301 }, { %r296, %r297, %r298, %r299 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r302, %r303, %r304, %r305 }, { %r260, %r261, %r262, %r263 }, { %r306, %r307 }, { %r302, %r303, %r304, %r305 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r308, %r309, %r310, %r311 }, { %r312, %r313, %r314, %r315 }, { %r264, %r265 }, { %r308, %r309, %r310, %r311 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r316, %r317, %r318, %r319 }, { %r312, %r313, %r314, %r315 }, { %r270, %r271 }, { %r316, %r317, %r318, %r319 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r320, %r321, %r322, %r323 }, { %r312, %r313, %r314, %r315 }, { %r276, %r277 }, { %r320, %r321, %r322, %r323 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r324, %r325, %r326, %r327 }, { %r312, %r313, %r314, %r315 }, { %r282, %r283 }, { %r324, %r325, %r326, %r327 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r328, %r329, %r330, %r331 }, { %r312, %r313, %r314, %r315 }, { %r288, %r289 }, { %r328, %r329, %r330, %r331 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r332, %r333, %r334, %r335 }, { %r312, %r313, %r314, %r315 }, { %r294, %r295 }, { %r332, %r333, %r334, %r335 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r336, %r337, %r338, %r339 }, { %r312, %r313, %r314, %r315 }, { %r300, %r301 }, { %r336, %r337, %r338, %r339 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r340, %r341, %r342, %r343 }, { %r312, %r313, %r314, %r315 }, { %r306, %r307 }, { %r340, %r341, %r342, %r343 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r344, %r345, %r346, %r347 }, { %r348, %r349, %r350, %r351 }, { %r264, %r265 }, { %r344, %r345, %r346, %r347 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r352, %r353, %r354, %r355 }, { %r348, %r349, %r350, %r351 }, { %r270, %r271 }, { %r352, %r353, %r354, %r355 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r356, %r357, %r358, %r359 }, { %r348, %r349, %r350, %r351 }, { %r276, %r277 }, { %r356, %r357, %r358, %r359 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r360, %r361, %r362, %r363 }, { %r348, %r349, %r350, %r351 }, { %r282, %r283 }, { %r360, %r361, %r362, %r363 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r364, %r365, %r366, %r367 }, { %r348, %r349, %r350, %r351 }, { %r288, %r289 }, { %r364, %r365, %r366, %r367 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r368, %r369, %r370, %r371 }, { %r348, %r349, %r350, %r351 }, { %r294, %r295 }, { %r368, %r369, %r370, %r371 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r372, %r373, %r374, %r375 }, { %r348, %r349, %r350, %r351 }, { %r300, %r301 }, { %r372, %r373, %r374, %r375 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r376, %r377, %r378, %r379 }, { %r348, %r349, %r350, %r351 }, { %r306, %r307 }, { %r376, %r377, %r378, %r379 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r380, %r381, %r382, %r383 }, { %r384, %r385, %r386, %r387 }, { %r264, %r265 }, { %r380, %r381, %r382, %r383 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r388, %r389, %r390, %r391 }, { %r384, %r385, %r386, %r387 }, { %r270, %r271 }, { %r388, %r389, %r390, %r391 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r392, %r393, %r394, %r395 }, { %r384, %r385, %r386, %r387 }, { %r276, %r277 }, { %r392, %r393, %r394, %r395 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r396, %r397, %r398, %r399 }, { %r384, %r385, %r386, %r387 }, { %r282, %r283 }, { %r396, %r397, %r398, %r399 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r400, %r401, %r402, %r403 }, { %r384, %r385, %r386, %r387 }, { %r288, %r289 }, { %r400, %r401, %r402, %r403 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r404, %r405, %r406, %r407 }, { %r384, %r385, %r386, %r387 }, { %r294, %r295 }, { %r404, %r405, %r406, %r407 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r408, %r409, %r410, %r411 }, { %r384, %r385, %r386, %r387 }, { %r300, %r301 }, { %r408, %r409, %r410, %r411 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r412, %r413, %r414, %r415 }, { %r384, %r385, %r386, %r387 }, { %r306, %r307 }, { %r412, %r413, %r414, %r415 };
	// end inline asm
	.loc	1 344 81                        // sk05_mlp_gateup.py:344:81
	cvt.rn.f32.s32 	%r448, %r415;
	cvt.rn.f32.s32 	%r449, %r414;
	cvt.rn.f32.s32 	%r450, %r413;
	cvt.rn.f32.s32 	%r451, %r412;
	cvt.rn.f32.s32 	%r452, %r411;
	cvt.rn.f32.s32 	%r453, %r410;
	cvt.rn.f32.s32 	%r454, %r409;
	cvt.rn.f32.s32 	%r455, %r408;
	cvt.rn.f32.s32 	%r456, %r407;
	cvt.rn.f32.s32 	%r457, %r406;
	cvt.rn.f32.s32 	%r458, %r405;
	cvt.rn.f32.s32 	%r459, %r404;
	cvt.rn.f32.s32 	%r460, %r403;
	cvt.rn.f32.s32 	%r461, %r402;
	cvt.rn.f32.s32 	%r462, %r401;
	cvt.rn.f32.s32 	%r463, %r400;
	cvt.rn.f32.s32 	%r464, %r399;
	cvt.rn.f32.s32 	%r465, %r398;
	cvt.rn.f32.s32 	%r466, %r397;
	cvt.rn.f32.s32 	%r467, %r396;
	cvt.rn.f32.s32 	%r468, %r395;
	cvt.rn.f32.s32 	%r469, %r394;
	cvt.rn.f32.s32 	%r470, %r393;
	cvt.rn.f32.s32 	%r471, %r392;
	cvt.rn.f32.s32 	%r472, %r391;
	cvt.rn.f32.s32 	%r473, %r390;
	cvt.rn.f32.s32 	%r474, %r389;
	cvt.rn.f32.s32 	%r475, %r388;
	cvt.rn.f32.s32 	%r476, %r383;
	cvt.rn.f32.s32 	%r477, %r382;
	cvt.rn.f32.s32 	%r478, %r381;
	cvt.rn.f32.s32 	%r479, %r380;
	cvt.rn.f32.s32 	%r480, %r376;
	cvt.rn.f32.s32 	%r481, %r377;
	cvt.rn.f32.s32 	%r482, %r378;
	cvt.rn.f32.s32 	%r483, %r379;
	cvt.rn.f32.s32 	%r484, %r372;
	cvt.rn.f32.s32 	%r485, %r373;
	cvt.rn.f32.s32 	%r486, %r374;
	cvt.rn.f32.s32 	%r487, %r375;
	cvt.rn.f32.s32 	%r488, %r368;
	cvt.rn.f32.s32 	%r489, %r369;
	cvt.rn.f32.s32 	%r490, %r370;
	cvt.rn.f32.s32 	%r491, %r371;
	cvt.rn.f32.s32 	%r492, %r364;
	cvt.rn.f32.s32 	%r493, %r365;
	cvt.rn.f32.s32 	%r494, %r366;
	cvt.rn.f32.s32 	%r495, %r367;
	cvt.rn.f32.s32 	%r496, %r360;
	cvt.rn.f32.s32 	%r497, %r361;
	cvt.rn.f32.s32 	%r498, %r362;
	cvt.rn.f32.s32 	%r499, %r363;
	cvt.rn.f32.s32 	%r500, %r356;
	cvt.rn.f32.s32 	%r501, %r357;
	cvt.rn.f32.s32 	%r502, %r358;
	cvt.rn.f32.s32 	%r503, %r359;
	cvt.rn.f32.s32 	%r504, %r352;
	cvt.rn.f32.s32 	%r505, %r353;
	cvt.rn.f32.s32 	%r506, %r354;
	cvt.rn.f32.s32 	%r507, %r355;
	cvt.rn.f32.s32 	%r508, %r344;
	cvt.rn.f32.s32 	%r509, %r345;
	cvt.rn.f32.s32 	%r510, %r346;
	cvt.rn.f32.s32 	%r511, %r347;
	cvt.rn.f32.s32 	%r512, %r340;
	cvt.rn.f32.s32 	%r513, %r341;
	cvt.rn.f32.s32 	%r514, %r342;
	cvt.rn.f32.s32 	%r515, %r343;
	cvt.rn.f32.s32 	%r516, %r336;
	cvt.rn.f32.s32 	%r517, %r337;
	cvt.rn.f32.s32 	%r518, %r338;
	cvt.rn.f32.s32 	%r519, %r339;
	cvt.rn.f32.s32 	%r520, %r332;
	cvt.rn.f32.s32 	%r521, %r333;
	cvt.rn.f32.s32 	%r522, %r334;
	cvt.rn.f32.s32 	%r523, %r335;
	cvt.rn.f32.s32 	%r524, %r328;
	cvt.rn.f32.s32 	%r525, %r329;
	cvt.rn.f32.s32 	%r526, %r330;
	cvt.rn.f32.s32 	%r527, %r331;
	cvt.rn.f32.s32 	%r528, %r324;
	cvt.rn.f32.s32 	%r529, %r325;
	cvt.rn.f32.s32 	%r530, %r326;
	cvt.rn.f32.s32 	%r531, %r327;
	cvt.rn.f32.s32 	%r532, %r320;
	cvt.rn.f32.s32 	%r533, %r321;
	cvt.rn.f32.s32 	%r534, %r322;
	cvt.rn.f32.s32 	%r535, %r323;
	cvt.rn.f32.s32 	%r536, %r316;
	cvt.rn.f32.s32 	%r537, %r317;
	cvt.rn.f32.s32 	%r538, %r318;
	cvt.rn.f32.s32 	%r539, %r319;
	cvt.rn.f32.s32 	%r540, %r308;
	cvt.rn.f32.s32 	%r541, %r309;
	cvt.rn.f32.s32 	%r542, %r310;
	cvt.rn.f32.s32 	%r543, %r311;
	cvt.rn.f32.s32 	%r544, %r302;
	cvt.rn.f32.s32 	%r545, %r303;
	cvt.rn.f32.s32 	%r546, %r304;
	cvt.rn.f32.s32 	%r547, %r305;
	cvt.rn.f32.s32 	%r548, %r296;
	cvt.rn.f32.s32 	%r549, %r297;
	cvt.rn.f32.s32 	%r550, %r298;
	cvt.rn.f32.s32 	%r551, %r299;
	cvt.rn.f32.s32 	%r552, %r290;
	cvt.rn.f32.s32 	%r553, %r291;
	cvt.rn.f32.s32 	%r554, %r292;
	cvt.rn.f32.s32 	%r555, %r293;
	cvt.rn.f32.s32 	%r556, %r284;
	cvt.rn.f32.s32 	%r557, %r285;
	cvt.rn.f32.s32 	%r558, %r286;
	cvt.rn.f32.s32 	%r559, %r287;
	cvt.rn.f32.s32 	%r560, %r278;
	cvt.rn.f32.s32 	%r561, %r279;
	cvt.rn.f32.s32 	%r562, %r280;
	cvt.rn.f32.s32 	%r563, %r281;
	cvt.rn.f32.s32 	%r564, %r272;
	cvt.rn.f32.s32 	%r565, %r273;
	cvt.rn.f32.s32 	%r566, %r274;
	cvt.rn.f32.s32 	%r567, %r275;
	cvt.rn.f32.s32 	%r568, %r266;
	cvt.rn.f32.s32 	%r569, %r267;
	cvt.rn.f32.s32 	%r570, %r268;
	cvt.rn.f32.s32 	%r571, %r269;
	cvt.rn.f32.s32 	%r572, %r256;
	cvt.rn.f32.s32 	%r573, %r257;
	cvt.rn.f32.s32 	%r574, %r258;
	cvt.rn.f32.s32 	%r575, %r259;
	.loc	1 344 15                        // sk05_mlp_gateup.py:344:15
	fma.rn.f32 	%r1421, %r427, %r575, %r1421;
	fma.rn.f32 	%r1420, %r426, %r574, %r1420;
	fma.rn.f32 	%r1419, %r427, %r573, %r1419;
	fma.rn.f32 	%r1418, %r426, %r572, %r1418;
	fma.rn.f32 	%r1425, %r429, %r571, %r1425;
	fma.rn.f32 	%r1424, %r428, %r570, %r1424;
	fma.rn.f32 	%r1423, %r429, %r569, %r1423;
	fma.rn.f32 	%r1422, %r428, %r568, %r1422;
	fma.rn.f32 	%r1429, %r431, %r567, %r1429;
	fma.rn.f32 	%r1428, %r430, %r566, %r1428;
	fma.rn.f32 	%r1427, %r431, %r565, %r1427;
	fma.rn.f32 	%r1426, %r430, %r564, %r1426;
	fma.rn.f32 	%r1433, %r433, %r563, %r1433;
	fma.rn.f32 	%r1432, %r432, %r562, %r1432;
	fma.rn.f32 	%r1431, %r433, %r561, %r1431;
	fma.rn.f32 	%r1430, %r432, %r560, %r1430;
	fma.rn.f32 	%r1437, %r435, %r559, %r1437;
	fma.rn.f32 	%r1436, %r434, %r558, %r1436;
	fma.rn.f32 	%r1435, %r435, %r557, %r1435;
	fma.rn.f32 	%r1434, %r434, %r556, %r1434;
	fma.rn.f32 	%r1441, %r437, %r555, %r1441;
	fma.rn.f32 	%r1440, %r436, %r554, %r1440;
	fma.rn.f32 	%r1439, %r437, %r553, %r1439;
	fma.rn.f32 	%r1438, %r436, %r552, %r1438;
	fma.rn.f32 	%r1445, %r439, %r551, %r1445;
	fma.rn.f32 	%r1444, %r438, %r550, %r1444;
	fma.rn.f32 	%r1443, %r439, %r549, %r1443;
	fma.rn.f32 	%r1442, %r438, %r548, %r1442;
	fma.rn.f32 	%r1449, %r441, %r547, %r1449;
	fma.rn.f32 	%r1448, %r440, %r546, %r1448;
	fma.rn.f32 	%r1447, %r441, %r545, %r1447;
	fma.rn.f32 	%r1446, %r440, %r544, %r1446;
	fma.rn.f32 	%r1453, %r427, %r543, %r1453;
	fma.rn.f32 	%r1452, %r426, %r542, %r1452;
	fma.rn.f32 	%r1451, %r427, %r541, %r1451;
	fma.rn.f32 	%r1450, %r426, %r540, %r1450;
	fma.rn.f32 	%r1457, %r429, %r539, %r1457;
	fma.rn.f32 	%r1456, %r428, %r538, %r1456;
	fma.rn.f32 	%r1455, %r429, %r537, %r1455;
	fma.rn.f32 	%r1454, %r428, %r536, %r1454;
	fma.rn.f32 	%r1461, %r431, %r535, %r1461;
	fma.rn.f32 	%r1460, %r430, %r534, %r1460;
	fma.rn.f32 	%r1459, %r431, %r533, %r1459;
	fma.rn.f32 	%r1458, %r430, %r532, %r1458;
	fma.rn.f32 	%r1465, %r433, %r531, %r1465;
	fma.rn.f32 	%r1464, %r432, %r530, %r1464;
	fma.rn.f32 	%r1463, %r433, %r529, %r1463;
	fma.rn.f32 	%r1462, %r432, %r528, %r1462;
	fma.rn.f32 	%r1469, %r435, %r527, %r1469;
	fma.rn.f32 	%r1468, %r434, %r526, %r1468;
	fma.rn.f32 	%r1467, %r435, %r525, %r1467;
	fma.rn.f32 	%r1466, %r434, %r524, %r1466;
	fma.rn.f32 	%r1473, %r437, %r523, %r1473;
	fma.rn.f32 	%r1472, %r436, %r522, %r1472;
	fma.rn.f32 	%r1471, %r437, %r521, %r1471;
	fma.rn.f32 	%r1470, %r436, %r520, %r1470;
	fma.rn.f32 	%r1477, %r439, %r519, %r1477;
	fma.rn.f32 	%r1476, %r438, %r518, %r1476;
	fma.rn.f32 	%r1475, %r439, %r517, %r1475;
	fma.rn.f32 	%r1474, %r438, %r516, %r1474;
	fma.rn.f32 	%r1481, %r441, %r515, %r1481;
	fma.rn.f32 	%r1480, %r440, %r514, %r1480;
	fma.rn.f32 	%r1479, %r441, %r513, %r1479;
	fma.rn.f32 	%r1478, %r440, %r512, %r1478;
	fma.rn.f32 	%r1485, %r427, %r511, %r1485;
	fma.rn.f32 	%r1484, %r426, %r510, %r1484;
	fma.rn.f32 	%r1483, %r427, %r509, %r1483;
	fma.rn.f32 	%r1482, %r426, %r508, %r1482;
	fma.rn.f32 	%r1489, %r429, %r507, %r1489;
	fma.rn.f32 	%r1488, %r428, %r506, %r1488;
	fma.rn.f32 	%r1487, %r429, %r505, %r1487;
	fma.rn.f32 	%r1486, %r428, %r504, %r1486;
	fma.rn.f32 	%r1493, %r431, %r503, %r1493;
	fma.rn.f32 	%r1492, %r430, %r502, %r1492;
	fma.rn.f32 	%r1491, %r431, %r501, %r1491;
	fma.rn.f32 	%r1490, %r430, %r500, %r1490;
	fma.rn.f32 	%r1497, %r433, %r499, %r1497;
	fma.rn.f32 	%r1496, %r432, %r498, %r1496;
	fma.rn.f32 	%r1495, %r433, %r497, %r1495;
	fma.rn.f32 	%r1494, %r432, %r496, %r1494;
	fma.rn.f32 	%r1501, %r435, %r495, %r1501;
	fma.rn.f32 	%r1500, %r434, %r494, %r1500;
	fma.rn.f32 	%r1499, %r435, %r493, %r1499;
	fma.rn.f32 	%r1498, %r434, %r492, %r1498;
	fma.rn.f32 	%r1505, %r437, %r491, %r1505;
	fma.rn.f32 	%r1504, %r436, %r490, %r1504;
	fma.rn.f32 	%r1503, %r437, %r489, %r1503;
	fma.rn.f32 	%r1502, %r436, %r488, %r1502;
	fma.rn.f32 	%r1509, %r439, %r487, %r1509;
	fma.rn.f32 	%r1508, %r438, %r486, %r1508;
	fma.rn.f32 	%r1507, %r439, %r485, %r1507;
	fma.rn.f32 	%r1506, %r438, %r484, %r1506;
	fma.rn.f32 	%r1513, %r441, %r483, %r1513;
	fma.rn.f32 	%r1512, %r440, %r482, %r1512;
	fma.rn.f32 	%r1511, %r441, %r481, %r1511;
	fma.rn.f32 	%r1510, %r440, %r480, %r1510;
	fma.rn.f32 	%r1514, %r426, %r479, %r1514;
	fma.rn.f32 	%r1515, %r427, %r478, %r1515;
	fma.rn.f32 	%r1516, %r426, %r477, %r1516;
	fma.rn.f32 	%r1517, %r427, %r476, %r1517;
	fma.rn.f32 	%r1518, %r428, %r475, %r1518;
	fma.rn.f32 	%r1519, %r429, %r474, %r1519;
	fma.rn.f32 	%r1520, %r428, %r473, %r1520;
	fma.rn.f32 	%r1521, %r429, %r472, %r1521;
	fma.rn.f32 	%r1522, %r430, %r471, %r1522;
	fma.rn.f32 	%r1523, %r431, %r470, %r1523;
	fma.rn.f32 	%r1524, %r430, %r469, %r1524;
	fma.rn.f32 	%r1525, %r431, %r468, %r1525;
	fma.rn.f32 	%r1526, %r432, %r467, %r1526;
	fma.rn.f32 	%r1527, %r433, %r466, %r1527;
	fma.rn.f32 	%r1528, %r432, %r465, %r1528;
	fma.rn.f32 	%r1529, %r433, %r464, %r1529;
	fma.rn.f32 	%r1530, %r434, %r463, %r1530;
	fma.rn.f32 	%r1531, %r435, %r462, %r1531;
	fma.rn.f32 	%r1532, %r434, %r461, %r1532;
	fma.rn.f32 	%r1533, %r435, %r460, %r1533;
	fma.rn.f32 	%r1534, %r436, %r459, %r1534;
	fma.rn.f32 	%r1535, %r437, %r458, %r1535;
	fma.rn.f32 	%r1536, %r436, %r457, %r1536;
	fma.rn.f32 	%r1537, %r437, %r456, %r1537;
	fma.rn.f32 	%r1538, %r438, %r455, %r1538;
	fma.rn.f32 	%r1539, %r439, %r454, %r1539;
	fma.rn.f32 	%r1540, %r438, %r453, %r1540;
	fma.rn.f32 	%r1541, %r439, %r452, %r1541;
	fma.rn.f32 	%r1542, %r440, %r451, %r1542;
	fma.rn.f32 	%r1543, %r441, %r450, %r1543;
	fma.rn.f32 	%r1544, %r440, %r449, %r1544;
	fma.rn.f32 	%r1545, %r441, %r448, %r1545;
	.loc	1 345 18                        // sk05_mlp_gateup.py:345:18
	add.s64 	%rd76, %rd277, %rd5;
	add.s64 	%rd77, %rd278, %rd5;
	add.s64 	%rd78, %rd279, %rd5;
	.loc	1 346 18                        // sk05_mlp_gateup.py:346:18
	add.s64 	%rd79, %rd280, %rd5;
	add.s64 	%rd80, %rd281, %rd5;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	add.s64 	%rd81, %rd282, %rd5;
	add.s32 	%r576, %r1414, 1;
	setp.gt.s32 	%p6, %r576, 2;
	selp.b32 	%r1414, 0, %r576, %p6;
	.loc	1 344 30                        // sk05_mlp_gateup.py:344:30
	shl.b32 	%r577, %r1414, 14;
	bar.sync 	0;
	add.s32 	%r416, %r29, %r577;
	selp.b32 	%r417, 16, 0, %p4;
	// begin inline asm
	cp.async.cg.shared.global [ %r416 + 0 ], [ %rd76 + 0 ], 0x10, %r417;
	// end inline asm
	add.s32 	%r418, %r416, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r418 + 0 ], [ %rd77 + 0 ], 0x10, %r417;
	// end inline asm
	add.s32 	%r419, %r416, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r419 + 0 ], [ %rd78 + 0 ], 0x10, %r417;
	// end inline asm
	add.s32 	%r420, %r416, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r420 + 0 ], [ %rd79 + 0 ], 0x10, %r417;
	// end inline asm
	cp.async.commit_group;
	.loc	1 344 47                        // sk05_mlp_gateup.py:344:47
	shl.b32 	%r578, %r1414, 13;
	add.s32 	%r579, %r29, %r578;
	add.s32 	%r421, %r579, 49152;
	// begin inline asm
	cp.async.cg.shared.global [ %r421 + 0 ], [ %rd80 + 0 ], 0x10, %r417;
	// end inline asm
	add.s32 	%r422, %r579, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r422 + 0 ], [ %rd81 + 0 ], 0x10, %r417;
	// end inline asm
	cp.async.commit_group;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	add.s32 	%r1415, %r1415, 1;
	add.s64 	%rd282, %rd282, 64;
	add.s64 	%rd281, %rd281, 64;
	add.s64 	%rd280, %rd280, 64;
	add.s64 	%rd279, %rd279, 64;
	add.s64 	%rd278, %rd278, 64;
	add.s64 	%rd277, %rd277, 64;
	setp.ne.b32 	%p7, %r16, %r1415;
	@%p7 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_4;
$L__BB0_1:                              // %.._crit_edge_crit_edge
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	and.b32 	%r1417, %r2, 16;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	shl.b32 	%r1416, %r2, 4;
	mov.b32 	%r1418, 0f00000000;
	mov.b32 	%r1419, %r1418;
	mov.b32 	%r1420, %r1418;
	mov.b32 	%r1421, %r1418;
	mov.b32 	%r1422, %r1418;
	mov.b32 	%r1423, %r1418;
	mov.b32 	%r1424, %r1418;
	mov.b32 	%r1425, %r1418;
	mov.b32 	%r1426, %r1418;
	mov.b32 	%r1427, %r1418;
	mov.b32 	%r1428, %r1418;
	mov.b32 	%r1429, %r1418;
	mov.b32 	%r1430, %r1418;
	mov.b32 	%r1431, %r1418;
	mov.b32 	%r1432, %r1418;
	mov.b32 	%r1433, %r1418;
	mov.b32 	%r1434, %r1418;
	mov.b32 	%r1435, %r1418;
	mov.b32 	%r1436, %r1418;
	mov.b32 	%r1437, %r1418;
	mov.b32 	%r1438, %r1418;
	mov.b32 	%r1439, %r1418;
	mov.b32 	%r1440, %r1418;
	mov.b32 	%r1441, %r1418;
	mov.b32 	%r1442, %r1418;
	mov.b32 	%r1443, %r1418;
	mov.b32 	%r1444, %r1418;
	mov.b32 	%r1445, %r1418;
	mov.b32 	%r1446, %r1418;
	mov.b32 	%r1447, %r1418;
	mov.b32 	%r1448, %r1418;
	mov.b32 	%r1449, %r1418;
	mov.b32 	%r1450, %r1418;
	mov.b32 	%r1451, %r1418;
	mov.b32 	%r1452, %r1418;
	mov.b32 	%r1453, %r1418;
	mov.b32 	%r1454, %r1418;
	mov.b32 	%r1455, %r1418;
	mov.b32 	%r1456, %r1418;
	mov.b32 	%r1457, %r1418;
	mov.b32 	%r1458, %r1418;
	mov.b32 	%r1459, %r1418;
	mov.b32 	%r1460, %r1418;
	mov.b32 	%r1461, %r1418;
	mov.b32 	%r1462, %r1418;
	mov.b32 	%r1463, %r1418;
	mov.b32 	%r1464, %r1418;
	mov.b32 	%r1465, %r1418;
	mov.b32 	%r1466, %r1418;
	mov.b32 	%r1467, %r1418;
	mov.b32 	%r1468, %r1418;
	mov.b32 	%r1469, %r1418;
	mov.b32 	%r1470, %r1418;
	mov.b32 	%r1471, %r1418;
	mov.b32 	%r1472, %r1418;
	mov.b32 	%r1473, %r1418;
	mov.b32 	%r1474, %r1418;
	mov.b32 	%r1475, %r1418;
	mov.b32 	%r1476, %r1418;
	mov.b32 	%r1477, %r1418;
	mov.b32 	%r1478, %r1418;
	mov.b32 	%r1479, %r1418;
	mov.b32 	%r1480, %r1418;
	mov.b32 	%r1481, %r1418;
	mov.b32 	%r1482, %r1418;
	mov.b32 	%r1483, %r1418;
	mov.b32 	%r1484, %r1418;
	mov.b32 	%r1485, %r1418;
	mov.b32 	%r1486, %r1418;
	mov.b32 	%r1487, %r1418;
	mov.b32 	%r1488, %r1418;
	mov.b32 	%r1489, %r1418;
	mov.b32 	%r1490, %r1418;
	mov.b32 	%r1491, %r1418;
	mov.b32 	%r1492, %r1418;
	mov.b32 	%r1493, %r1418;
	mov.b32 	%r1494, %r1418;
	mov.b32 	%r1495, %r1418;
	mov.b32 	%r1496, %r1418;
	mov.b32 	%r1497, %r1418;
	mov.b32 	%r1498, %r1418;
	mov.b32 	%r1499, %r1418;
	mov.b32 	%r1500, %r1418;
	mov.b32 	%r1501, %r1418;
	mov.b32 	%r1502, %r1418;
	mov.b32 	%r1503, %r1418;
	mov.b32 	%r1504, %r1418;
	mov.b32 	%r1505, %r1418;
	mov.b32 	%r1506, %r1418;
	mov.b32 	%r1507, %r1418;
	mov.b32 	%r1508, %r1418;
	mov.b32 	%r1509, %r1418;
	mov.b32 	%r1510, %r1418;
	mov.b32 	%r1511, %r1418;
	mov.b32 	%r1512, %r1418;
	mov.b32 	%r1513, %r1418;
	mov.b32 	%r1514, %r1418;
	mov.b32 	%r1515, %r1418;
	mov.b32 	%r1516, %r1418;
	mov.b32 	%r1517, %r1418;
	mov.b32 	%r1518, %r1418;
	mov.b32 	%r1519, %r1418;
	mov.b32 	%r1520, %r1418;
	mov.b32 	%r1521, %r1418;
	mov.b32 	%r1522, %r1418;
	mov.b32 	%r1523, %r1418;
	mov.b32 	%r1524, %r1418;
	mov.b32 	%r1525, %r1418;
	mov.b32 	%r1526, %r1418;
	mov.b32 	%r1527, %r1418;
	mov.b32 	%r1528, %r1418;
	mov.b32 	%r1529, %r1418;
	mov.b32 	%r1530, %r1418;
	mov.b32 	%r1531, %r1418;
	mov.b32 	%r1532, %r1418;
	mov.b32 	%r1533, %r1418;
	mov.b32 	%r1534, %r1418;
	mov.b32 	%r1535, %r1418;
	mov.b32 	%r1536, %r1418;
	mov.b32 	%r1537, %r1418;
	mov.b32 	%r1538, %r1418;
	mov.b32 	%r1539, %r1418;
	mov.b32 	%r1540, %r1418;
	mov.b32 	%r1541, %r1418;
	mov.b32 	%r1542, %r1418;
	mov.b32 	%r1543, %r1418;
	mov.b32 	%r1544, %r1418;
	mov.b32 	%r1545, %r1418;
$L__BB0_4:                              // %._crit_edge
	.loc	1 327 45                        // sk05_mlp_gateup.py:327:45
	shl.b32 	%r810, %r7, 3;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r811, %r810, %r4;
	or.b32 	%r812, %r811, 7;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r813, %r812, %r24;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r814, %r811, 6;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r815, %r814, %r24;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r816, %r811, 5;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r817, %r816, %r24;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r818, %r811, 4;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r819, %r818, %r24;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r820, %r811, 3;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r821, %r820, %r24;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r822, %r811, 2;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r823, %r822, %r24;
	.loc	1 327 32                        // sk05_mlp_gateup.py:327:32
	or.b32 	%r824, %r811, 1;
	.loc	1 327 57                        // sk05_mlp_gateup.py:327:57
	rem.s32 	%r825, %r824, %r24;
	rem.s32 	%r826, %r811, %r24;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r827, %r1, %r3;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r828, %r827, %r23;
	.loc	1 326 45                        // sk05_mlp_gateup.py:326:45
	and.b32 	%r829, %r2, 240;
	bfe.u32 	%r830, %r2, 4, 4;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r831, %r830, %r1;
	or.b32 	%r832, %r831, 240;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r833, %r832, %r23;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r834, %r831, 224;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r835, %r834, %r23;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r836, %r831, 208;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r837, %r836, %r23;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r838, %r831, 192;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r839, %r838, %r23;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r840, %r831, 176;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r841, %r840, %r23;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r842, %r831, 160;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r843, %r842, %r23;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r844, %r831, 144;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r845, %r844, %r23;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r846, %r831, 128;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r847, %r846, %r23;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r848, %r831, 112;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r849, %r848, %r23;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r850, %r831, 96;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r851, %r850, %r23;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r852, %r831, 80;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r853, %r852, %r23;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r854, %r831, 64;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r855, %r854, %r23;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r856, %r831, 48;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r857, %r856, %r23;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r858, %r831, 32;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r859, %r858, %r23;
	.loc	1 326 32                        // sk05_mlp_gateup.py:326:32
	or.b32 	%r860, %r831, 16;
	.loc	1 326 57                        // sk05_mlp_gateup.py:326:57
	rem.s32 	%r861, %r860, %r23;
	rem.s32 	%r862, %r831, %r23;
	.loc	1 335 23                        // sk05_mlp_gateup.py:335:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 348 38                        // sk05_mlp_gateup.py:348:38
	mad.wide.s32 	%rd83, %r828, 4, %rd28;
	.loc	1 348 24                        // sk05_mlp_gateup.py:348:24
	// begin inline asm
	mov.u32 %r581, 0x0;
	ld.global.b32 { %r581 }, [ %rd83 + 0 ];
	// end inline asm
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	and.b32 	%r863, %r2, 7;
	shl.b32 	%r864, %r863, 3;
	shl.b32 	%r865, %r829, 2;
	and.b32 	%r866, %r2, 8;
	shr.u32 	%r867, %r866, 1;
	add.s32 	%r868, %r194, %r864;
	add.s32 	%r869, %r868, %r865;
	add.s32 	%r580, %r869, %r867;
	// begin inline asm
	st.shared.b32 [ %r580 + 0 ], %r581;
	// end inline asm
	bar.sync 	0;
	and.b32 	%r870, %r17, 56;
	and.b32 	%r871, %r2, 192;
	add.s32 	%r872, %r194, %r870;
	add.s32 	%r873, %r872, %r871;
	ld.shared.v2.b32 	{%r874, %r875}, [%r873];
	ld.shared.v2.b32 	{%r876, %r877}, [%r873+256];
	ld.shared.v2.b32 	{%r878, %r879}, [%r873+512];
	ld.shared.v2.b32 	{%r880, %r881}, [%r873+768];
	.loc	1 349 38                        // sk05_mlp_gateup.py:349:38
	mad.wide.s32 	%rd84, %r8, 4, %rd29;
	mad.wide.s32 	%rd85, %r9, 4, %rd29;
	mad.wide.s32 	%rd86, %r10, 4, %rd29;
	mad.wide.s32 	%rd87, %r11, 4, %rd29;
	mad.wide.s32 	%rd88, %r12, 4, %rd29;
	mad.wide.s32 	%rd89, %r13, 4, %rd29;
	mad.wide.s32 	%rd90, %r14, 4, %rd29;
	mad.wide.s32 	%rd91, %r15, 4, %rd29;
	.loc	1 349 24                        // sk05_mlp_gateup.py:349:24
	// begin inline asm
	mov.u32 %r582, 0x0;
	mov.u32 %r583, 0x0;
	ld.global.v2.b32 { %r582, %r583 }, [ %rd84 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r584, 0x0;
	mov.u32 %r585, 0x0;
	ld.global.v2.b32 { %r584, %r585 }, [ %rd85 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r586, 0x0;
	mov.u32 %r587, 0x0;
	ld.global.v2.b32 { %r586, %r587 }, [ %rd86 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r588, 0x0;
	mov.u32 %r589, 0x0;
	ld.global.v2.b32 { %r588, %r589 }, [ %rd87 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r590, 0x0;
	mov.u32 %r591, 0x0;
	ld.global.v2.b32 { %r590, %r591 }, [ %rd88 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r592, 0x0;
	mov.u32 %r593, 0x0;
	ld.global.v2.b32 { %r592, %r593 }, [ %rd89 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r594, 0x0;
	mov.u32 %r595, 0x0;
	ld.global.v2.b32 { %r594, %r595 }, [ %rd90 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r596, 0x0;
	mov.u32 %r597, 0x0;
	ld.global.v2.b32 { %r596, %r597 }, [ %rd91 + 0 ];
	// end inline asm
	.loc	1 350 49                        // sk05_mlp_gateup.py:350:49
	mul.lo.s32 	%r882, %r862, %r26;
	mul.lo.s32 	%r883, %r861, %r26;
	mul.lo.s32 	%r884, %r859, %r26;
	mul.lo.s32 	%r885, %r857, %r26;
	mul.lo.s32 	%r886, %r855, %r26;
	mul.lo.s32 	%r887, %r853, %r26;
	mul.lo.s32 	%r888, %r851, %r26;
	mul.lo.s32 	%r889, %r849, %r26;
	mul.lo.s32 	%r890, %r847, %r26;
	mul.lo.s32 	%r891, %r845, %r26;
	mul.lo.s32 	%r892, %r843, %r26;
	mul.lo.s32 	%r893, %r841, %r26;
	mul.lo.s32 	%r894, %r839, %r26;
	mul.lo.s32 	%r895, %r837, %r26;
	mul.lo.s32 	%r896, %r835, %r26;
	mul.lo.s32 	%r897, %r833, %r26;
	.loc	1 350 31                        // sk05_mlp_gateup.py:350:31
	mad.wide.s32 	%rd236, %r882, 2, %rd27;
	mad.wide.s32 	%rd237, %r883, 2, %rd27;
	mad.wide.s32 	%rd238, %r884, 2, %rd27;
	mad.wide.s32 	%rd239, %r885, 2, %rd27;
	mad.wide.s32 	%rd240, %r886, 2, %rd27;
	mad.wide.s32 	%rd241, %r887, 2, %rd27;
	mad.wide.s32 	%rd242, %r888, 2, %rd27;
	mad.wide.s32 	%rd243, %r889, 2, %rd27;
	mad.wide.s32 	%rd244, %r890, 2, %rd27;
	mad.wide.s32 	%rd245, %r891, 2, %rd27;
	mad.wide.s32 	%rd246, %r892, 2, %rd27;
	mad.wide.s32 	%rd247, %r893, 2, %rd27;
	mad.wide.s32 	%rd248, %r894, 2, %rd27;
	mad.wide.s32 	%rd249, %r895, 2, %rd27;
	mad.wide.s32 	%rd250, %r896, 2, %rd27;
	mad.wide.s32 	%rd251, %r897, 2, %rd27;
	.loc	1 350 82                        // sk05_mlp_gateup.py:350:82
	mul.lo.s32 	%r898, %r826, %r27;
	mul.lo.s32 	%r899, %r825, %r27;
	mul.lo.s32 	%r900, %r823, %r27;
	mul.lo.s32 	%r901, %r821, %r27;
	mul.lo.s32 	%r902, %r819, %r27;
	mul.lo.s32 	%r903, %r817, %r27;
	mul.lo.s32 	%r904, %r815, %r27;
	mul.lo.s32 	%r905, %r813, %r27;
	.loc	1 350 64                        // sk05_mlp_gateup.py:350:64
	mul.wide.s32 	%rd252, %r898, 2;
	add.s64 	%rd92, %rd236, %rd252;
	mul.wide.s32 	%rd253, %r899, 2;
	add.s64 	%rd93, %rd236, %rd253;
	mul.wide.s32 	%rd254, %r900, 2;
	add.s64 	%rd94, %rd236, %rd254;
	mul.wide.s32 	%rd255, %r901, 2;
	add.s64 	%rd95, %rd236, %rd255;
	mul.wide.s32 	%rd256, %r902, 2;
	add.s64 	%rd96, %rd236, %rd256;
	mul.wide.s32 	%rd257, %r903, 2;
	add.s64 	%rd97, %rd236, %rd257;
	mul.wide.s32 	%rd258, %r904, 2;
	add.s64 	%rd98, %rd236, %rd258;
	mul.wide.s32 	%rd259, %r905, 2;
	add.s64 	%rd99, %rd236, %rd259;
	add.s64 	%rd100, %rd237, %rd252;
	add.s64 	%rd101, %rd237, %rd253;
	add.s64 	%rd102, %rd237, %rd254;
	add.s64 	%rd103, %rd237, %rd255;
	add.s64 	%rd104, %rd237, %rd256;
	add.s64 	%rd105, %rd237, %rd257;
	add.s64 	%rd106, %rd237, %rd258;
	add.s64 	%rd107, %rd237, %rd259;
	add.s64 	%rd108, %rd238, %rd252;
	add.s64 	%rd109, %rd238, %rd253;
	add.s64 	%rd110, %rd238, %rd254;
	add.s64 	%rd111, %rd238, %rd255;
	add.s64 	%rd112, %rd238, %rd256;
	add.s64 	%rd113, %rd238, %rd257;
	add.s64 	%rd114, %rd238, %rd258;
	add.s64 	%rd115, %rd238, %rd259;
	add.s64 	%rd116, %rd239, %rd252;
	add.s64 	%rd117, %rd239, %rd253;
	add.s64 	%rd118, %rd239, %rd254;
	add.s64 	%rd119, %rd239, %rd255;
	add.s64 	%rd120, %rd239, %rd256;
	add.s64 	%rd121, %rd239, %rd257;
	add.s64 	%rd122, %rd239, %rd258;
	add.s64 	%rd123, %rd239, %rd259;
	add.s64 	%rd124, %rd240, %rd252;
	add.s64 	%rd125, %rd240, %rd253;
	add.s64 	%rd126, %rd240, %rd254;
	add.s64 	%rd127, %rd240, %rd255;
	add.s64 	%rd128, %rd240, %rd256;
	add.s64 	%rd129, %rd240, %rd257;
	add.s64 	%rd130, %rd240, %rd258;
	add.s64 	%rd131, %rd240, %rd259;
	add.s64 	%rd132, %rd241, %rd252;
	add.s64 	%rd133, %rd241, %rd253;
	add.s64 	%rd134, %rd241, %rd254;
	add.s64 	%rd135, %rd241, %rd255;
	add.s64 	%rd136, %rd241, %rd256;
	add.s64 	%rd137, %rd241, %rd257;
	add.s64 	%rd138, %rd241, %rd258;
	add.s64 	%rd139, %rd241, %rd259;
	add.s64 	%rd140, %rd242, %rd252;
	add.s64 	%rd141, %rd242, %rd253;
	add.s64 	%rd142, %rd242, %rd254;
	add.s64 	%rd143, %rd242, %rd255;
	add.s64 	%rd144, %rd242, %rd256;
	add.s64 	%rd145, %rd242, %rd257;
	add.s64 	%rd146, %rd242, %rd258;
	add.s64 	%rd147, %rd242, %rd259;
	add.s64 	%rd148, %rd243, %rd252;
	add.s64 	%rd149, %rd243, %rd253;
	add.s64 	%rd150, %rd243, %rd254;
	add.s64 	%rd151, %rd243, %rd255;
	add.s64 	%rd152, %rd243, %rd256;
	add.s64 	%rd153, %rd243, %rd257;
	add.s64 	%rd154, %rd243, %rd258;
	add.s64 	%rd155, %rd243, %rd259;
	add.s64 	%rd156, %rd244, %rd252;
	add.s64 	%rd157, %rd244, %rd253;
	add.s64 	%rd158, %rd244, %rd254;
	add.s64 	%rd159, %rd244, %rd255;
	add.s64 	%rd160, %rd244, %rd256;
	add.s64 	%rd161, %rd244, %rd257;
	add.s64 	%rd162, %rd244, %rd258;
	add.s64 	%rd163, %rd244, %rd259;
	add.s64 	%rd164, %rd245, %rd252;
	add.s64 	%rd165, %rd245, %rd253;
	add.s64 	%rd166, %rd245, %rd254;
	add.s64 	%rd167, %rd245, %rd255;
	add.s64 	%rd168, %rd245, %rd256;
	add.s64 	%rd169, %rd245, %rd257;
	add.s64 	%rd170, %rd245, %rd258;
	add.s64 	%rd171, %rd245, %rd259;
	add.s64 	%rd172, %rd246, %rd252;
	add.s64 	%rd173, %rd246, %rd253;
	add.s64 	%rd174, %rd246, %rd254;
	add.s64 	%rd175, %rd246, %rd255;
	add.s64 	%rd176, %rd246, %rd256;
	add.s64 	%rd177, %rd246, %rd257;
	add.s64 	%rd178, %rd246, %rd258;
	add.s64 	%rd179, %rd246, %rd259;
	add.s64 	%rd180, %rd247, %rd252;
	add.s64 	%rd181, %rd247, %rd253;
	add.s64 	%rd182, %rd247, %rd254;
	add.s64 	%rd183, %rd247, %rd255;
	add.s64 	%rd184, %rd247, %rd256;
	add.s64 	%rd185, %rd247, %rd257;
	add.s64 	%rd186, %rd247, %rd258;
	add.s64 	%rd187, %rd247, %rd259;
	add.s64 	%rd188, %rd248, %rd252;
	add.s64 	%rd189, %rd248, %rd253;
	add.s64 	%rd190, %rd248, %rd254;
	add.s64 	%rd191, %rd248, %rd255;
	add.s64 	%rd192, %rd248, %rd256;
	add.s64 	%rd193, %rd248, %rd257;
	add.s64 	%rd194, %rd248, %rd258;
	add.s64 	%rd195, %rd248, %rd259;
	add.s64 	%rd196, %rd249, %rd252;
	add.s64 	%rd197, %rd249, %rd253;
	add.s64 	%rd198, %rd249, %rd254;
	add.s64 	%rd199, %rd249, %rd255;
	add.s64 	%rd200, %rd249, %rd256;
	add.s64 	%rd201, %rd249, %rd257;
	add.s64 	%rd202, %rd249, %rd258;
	add.s64 	%rd203, %rd249, %rd259;
	add.s64 	%rd204, %rd250, %rd252;
	add.s64 	%rd205, %rd250, %rd253;
	add.s64 	%rd206, %rd250, %rd254;
	add.s64 	%rd207, %rd250, %rd255;
	add.s64 	%rd208, %rd250, %rd256;
	add.s64 	%rd209, %rd250, %rd257;
	add.s64 	%rd210, %rd250, %rd258;
	add.s64 	%rd211, %rd250, %rd259;
	add.s64 	%rd212, %rd251, %rd252;
	add.s64 	%rd213, %rd251, %rd253;
	add.s64 	%rd214, %rd251, %rd254;
	add.s64 	%rd215, %rd251, %rd255;
	add.s64 	%rd216, %rd251, %rd256;
	add.s64 	%rd217, %rd251, %rd257;
	add.s64 	%rd218, %rd251, %rd258;
	add.s64 	%rd219, %rd251, %rd259;
	.loc	1 350 19                        // sk05_mlp_gateup.py:350:19
	// begin inline asm
	mov.u16 %rs1, 0x0;
	ld.global.b16 { %rs1 }, [ %rd92 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs2, 0x0;
	ld.global.b16 { %rs2 }, [ %rd93 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs3, 0x0;
	ld.global.b16 { %rs3 }, [ %rd94 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs4, 0x0;
	ld.global.b16 { %rs4 }, [ %rd95 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs5, 0x0;
	ld.global.b16 { %rs5 }, [ %rd96 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs6, 0x0;
	ld.global.b16 { %rs6 }, [ %rd97 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs7, 0x0;
	ld.global.b16 { %rs7 }, [ %rd98 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs8, 0x0;
	ld.global.b16 { %rs8 }, [ %rd99 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs9, 0x0;
	ld.global.b16 { %rs9 }, [ %rd100 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs10, 0x0;
	ld.global.b16 { %rs10 }, [ %rd101 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs11, 0x0;
	ld.global.b16 { %rs11 }, [ %rd102 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs12, 0x0;
	ld.global.b16 { %rs12 }, [ %rd103 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs13, 0x0;
	ld.global.b16 { %rs13 }, [ %rd104 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs14, 0x0;
	ld.global.b16 { %rs14 }, [ %rd105 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs15, 0x0;
	ld.global.b16 { %rs15 }, [ %rd106 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs16, 0x0;
	ld.global.b16 { %rs16 }, [ %rd107 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs17, 0x0;
	ld.global.b16 { %rs17 }, [ %rd108 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs18, 0x0;
	ld.global.b16 { %rs18 }, [ %rd109 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs19, 0x0;
	ld.global.b16 { %rs19 }, [ %rd110 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs20, 0x0;
	ld.global.b16 { %rs20 }, [ %rd111 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs21, 0x0;
	ld.global.b16 { %rs21 }, [ %rd112 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs22, 0x0;
	ld.global.b16 { %rs22 }, [ %rd113 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs23, 0x0;
	ld.global.b16 { %rs23 }, [ %rd114 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs24, 0x0;
	ld.global.b16 { %rs24 }, [ %rd115 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs25, 0x0;
	ld.global.b16 { %rs25 }, [ %rd116 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs26, 0x0;
	ld.global.b16 { %rs26 }, [ %rd117 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs27, 0x0;
	ld.global.b16 { %rs27 }, [ %rd118 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs28, 0x0;
	ld.global.b16 { %rs28 }, [ %rd119 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs29, 0x0;
	ld.global.b16 { %rs29 }, [ %rd120 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs30, 0x0;
	ld.global.b16 { %rs30 }, [ %rd121 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs31, 0x0;
	ld.global.b16 { %rs31 }, [ %rd122 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs32, 0x0;
	ld.global.b16 { %rs32 }, [ %rd123 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs33, 0x0;
	ld.global.b16 { %rs33 }, [ %rd124 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs34, 0x0;
	ld.global.b16 { %rs34 }, [ %rd125 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs35, 0x0;
	ld.global.b16 { %rs35 }, [ %rd126 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs36, 0x0;
	ld.global.b16 { %rs36 }, [ %rd127 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs37, 0x0;
	ld.global.b16 { %rs37 }, [ %rd128 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs38, 0x0;
	ld.global.b16 { %rs38 }, [ %rd129 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs39, 0x0;
	ld.global.b16 { %rs39 }, [ %rd130 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs40, 0x0;
	ld.global.b16 { %rs40 }, [ %rd131 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs41, 0x0;
	ld.global.b16 { %rs41 }, [ %rd132 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs42, 0x0;
	ld.global.b16 { %rs42 }, [ %rd133 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs43, 0x0;
	ld.global.b16 { %rs43 }, [ %rd134 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs44, 0x0;
	ld.global.b16 { %rs44 }, [ %rd135 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs45, 0x0;
	ld.global.b16 { %rs45 }, [ %rd136 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs46, 0x0;
	ld.global.b16 { %rs46 }, [ %rd137 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs47, 0x0;
	ld.global.b16 { %rs47 }, [ %rd138 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs48, 0x0;
	ld.global.b16 { %rs48 }, [ %rd139 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs49, 0x0;
	ld.global.b16 { %rs49 }, [ %rd140 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs50, 0x0;
	ld.global.b16 { %rs50 }, [ %rd141 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs51, 0x0;
	ld.global.b16 { %rs51 }, [ %rd142 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs52, 0x0;
	ld.global.b16 { %rs52 }, [ %rd143 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs53, 0x0;
	ld.global.b16 { %rs53 }, [ %rd144 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs54, 0x0;
	ld.global.b16 { %rs54 }, [ %rd145 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs55, 0x0;
	ld.global.b16 { %rs55 }, [ %rd146 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs56, 0x0;
	ld.global.b16 { %rs56 }, [ %rd147 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs57, 0x0;
	ld.global.b16 { %rs57 }, [ %rd148 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs58, 0x0;
	ld.global.b16 { %rs58 }, [ %rd149 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs59, 0x0;
	ld.global.b16 { %rs59 }, [ %rd150 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs60, 0x0;
	ld.global.b16 { %rs60 }, [ %rd151 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs61, 0x0;
	ld.global.b16 { %rs61 }, [ %rd152 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs62, 0x0;
	ld.global.b16 { %rs62 }, [ %rd153 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs63, 0x0;
	ld.global.b16 { %rs63 }, [ %rd154 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs64, 0x0;
	ld.global.b16 { %rs64 }, [ %rd155 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs65, 0x0;
	ld.global.b16 { %rs65 }, [ %rd156 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs66, 0x0;
	ld.global.b16 { %rs66 }, [ %rd157 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs67, 0x0;
	ld.global.b16 { %rs67 }, [ %rd158 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs68, 0x0;
	ld.global.b16 { %rs68 }, [ %rd159 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs69, 0x0;
	ld.global.b16 { %rs69 }, [ %rd160 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs70, 0x0;
	ld.global.b16 { %rs70 }, [ %rd161 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs71, 0x0;
	ld.global.b16 { %rs71 }, [ %rd162 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs72, 0x0;
	ld.global.b16 { %rs72 }, [ %rd163 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs73, 0x0;
	ld.global.b16 { %rs73 }, [ %rd164 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs74, 0x0;
	ld.global.b16 { %rs74 }, [ %rd165 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs75, 0x0;
	ld.global.b16 { %rs75 }, [ %rd166 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs76, 0x0;
	ld.global.b16 { %rs76 }, [ %rd167 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs77, 0x0;
	ld.global.b16 { %rs77 }, [ %rd168 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs78, 0x0;
	ld.global.b16 { %rs78 }, [ %rd169 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs79, 0x0;
	ld.global.b16 { %rs79 }, [ %rd170 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs80, 0x0;
	ld.global.b16 { %rs80 }, [ %rd171 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs81, 0x0;
	ld.global.b16 { %rs81 }, [ %rd172 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs82, 0x0;
	ld.global.b16 { %rs82 }, [ %rd173 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs83, 0x0;
	ld.global.b16 { %rs83 }, [ %rd174 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs84, 0x0;
	ld.global.b16 { %rs84 }, [ %rd175 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs85, 0x0;
	ld.global.b16 { %rs85 }, [ %rd176 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs86, 0x0;
	ld.global.b16 { %rs86 }, [ %rd177 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs87, 0x0;
	ld.global.b16 { %rs87 }, [ %rd178 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs88, 0x0;
	ld.global.b16 { %rs88 }, [ %rd179 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs89, 0x0;
	ld.global.b16 { %rs89 }, [ %rd180 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs90, 0x0;
	ld.global.b16 { %rs90 }, [ %rd181 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs91, 0x0;
	ld.global.b16 { %rs91 }, [ %rd182 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs92, 0x0;
	ld.global.b16 { %rs92 }, [ %rd183 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs93, 0x0;
	ld.global.b16 { %rs93 }, [ %rd184 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs94, 0x0;
	ld.global.b16 { %rs94 }, [ %rd185 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs95, 0x0;
	ld.global.b16 { %rs95 }, [ %rd186 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs96, 0x0;
	ld.global.b16 { %rs96 }, [ %rd187 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs97, 0x0;
	ld.global.b16 { %rs97 }, [ %rd188 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs98, 0x0;
	ld.global.b16 { %rs98 }, [ %rd189 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs99, 0x0;
	ld.global.b16 { %rs99 }, [ %rd190 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs100, 0x0;
	ld.global.b16 { %rs100 }, [ %rd191 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs101, 0x0;
	ld.global.b16 { %rs101 }, [ %rd192 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs102, 0x0;
	ld.global.b16 { %rs102 }, [ %rd193 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs103, 0x0;
	ld.global.b16 { %rs103 }, [ %rd194 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs104, 0x0;
	ld.global.b16 { %rs104 }, [ %rd195 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs105, 0x0;
	ld.global.b16 { %rs105 }, [ %rd196 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs106, 0x0;
	ld.global.b16 { %rs106 }, [ %rd197 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs107, 0x0;
	ld.global.b16 { %rs107 }, [ %rd198 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs108, 0x0;
	ld.global.b16 { %rs108 }, [ %rd199 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs109, 0x0;
	ld.global.b16 { %rs109 }, [ %rd200 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs110, 0x0;
	ld.global.b16 { %rs110 }, [ %rd201 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs111, 0x0;
	ld.global.b16 { %rs111 }, [ %rd202 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs112, 0x0;
	ld.global.b16 { %rs112 }, [ %rd203 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs113, 0x0;
	ld.global.b16 { %rs113 }, [ %rd204 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs114, 0x0;
	ld.global.b16 { %rs114 }, [ %rd205 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs115, 0x0;
	ld.global.b16 { %rs115 }, [ %rd206 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs116, 0x0;
	ld.global.b16 { %rs116 }, [ %rd207 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs117, 0x0;
	ld.global.b16 { %rs117 }, [ %rd208 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs118, 0x0;
	ld.global.b16 { %rs118 }, [ %rd209 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs119, 0x0;
	ld.global.b16 { %rs119 }, [ %rd210 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs120, 0x0;
	ld.global.b16 { %rs120 }, [ %rd211 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs121, 0x0;
	ld.global.b16 { %rs121 }, [ %rd212 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs122, 0x0;
	ld.global.b16 { %rs122 }, [ %rd213 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs123, 0x0;
	ld.global.b16 { %rs123 }, [ %rd214 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs124, 0x0;
	ld.global.b16 { %rs124 }, [ %rd215 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs125, 0x0;
	ld.global.b16 { %rs125 }, [ %rd216 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs126, 0x0;
	ld.global.b16 { %rs126 }, [ %rd217 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs127, 0x0;
	ld.global.b16 { %rs127 }, [ %rd218 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs128, 0x0;
	ld.global.b16 { %rs128 }, [ %rd219 + 0 ];
	// end inline asm
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	bar.sync 	0;
	shl.b32 	%r906, %r2, 7;
	and.b32 	%r907, %r906, 15360;
	shl.b32 	%r908, %r863, 4;
	or.b32 	%r909, %r907, %r908;
	xor.b32 	%r910, %r909, %r829;
	add.s32 	%r598, %r194, %r910;
	mov.b32 	%r599, {%rs1, %rs2};
	mov.b32 	%r600, {%rs3, %rs4};
	mov.b32 	%r601, {%rs5, %rs6};
	mov.b32 	%r602, {%rs7, %rs8};
	// begin inline asm
	st.shared.v4.b32 [ %r598 + 0 ], { %r599, %r600, %r601, %r602 };
	// end inline asm
	add.s32 	%r603, %r598, 256;
	mov.b32 	%r604, {%rs9, %rs10};
	mov.b32 	%r605, {%rs11, %rs12};
	mov.b32 	%r606, {%rs13, %rs14};
	mov.b32 	%r607, {%rs15, %rs16};
	// begin inline asm
	st.shared.v4.b32 [ %r603 + 0 ], { %r604, %r605, %r606, %r607 };
	// end inline asm
	add.s32 	%r608, %r598, 512;
	mov.b32 	%r609, {%rs17, %rs18};
	mov.b32 	%r610, {%rs19, %rs20};
	mov.b32 	%r611, {%rs21, %rs22};
	mov.b32 	%r612, {%rs23, %rs24};
	// begin inline asm
	st.shared.v4.b32 [ %r608 + 0 ], { %r609, %r610, %r611, %r612 };
	// end inline asm
	add.s32 	%r613, %r598, 768;
	mov.b32 	%r614, {%rs25, %rs26};
	mov.b32 	%r615, {%rs27, %rs28};
	mov.b32 	%r616, {%rs29, %rs30};
	mov.b32 	%r617, {%rs31, %rs32};
	// begin inline asm
	st.shared.v4.b32 [ %r613 + 0 ], { %r614, %r615, %r616, %r617 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r911, %r863, 11;
	shl.b32 	%r912, %r7, 4;
	shl.b32 	%r913, %r871, 2;
	setp.eq.b32 	%p24, %r1417, 0;
	shl.b32 	%r914, %r1417, 1;
	shr.u32 	%r915, %r6, 1;
	or.b32 	%r916, %r912, %r913;
	or.b32 	%r917, %r914, %r915;
	xor.b32 	%r918, %r916, %r917;
	or.b32 	%r919, %r918, %r911;
	add.s32 	%r920, %r194, %r919;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r921, %r922, %r923, %r924}, [%r920];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r925, %r926, %r927, %r928}, [%r920+1024];
	xor.b32 	%r929, %r919, 64;
	add.s32 	%r930, %r194, %r929;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r931, %r932, %r933, %r934}, [%r930];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r935, %r936, %r937, %r938}, [%r930+1024];
	bar.sync 	0;
	mov.b32 	%r618, {%rs33, %rs34};
	mov.b32 	%r619, {%rs35, %rs36};
	mov.b32 	%r620, {%rs37, %rs38};
	mov.b32 	%r621, {%rs39, %rs40};
	// begin inline asm
	st.shared.v4.b32 [ %r598 + 0 ], { %r618, %r619, %r620, %r621 };
	// end inline asm
	mov.b32 	%r622, {%rs41, %rs42};
	mov.b32 	%r623, {%rs43, %rs44};
	mov.b32 	%r624, {%rs45, %rs46};
	mov.b32 	%r625, {%rs47, %rs48};
	// begin inline asm
	st.shared.v4.b32 [ %r603 + 0 ], { %r622, %r623, %r624, %r625 };
	// end inline asm
	mov.b32 	%r626, {%rs49, %rs50};
	mov.b32 	%r627, {%rs51, %rs52};
	mov.b32 	%r628, {%rs53, %rs54};
	mov.b32 	%r629, {%rs55, %rs56};
	// begin inline asm
	st.shared.v4.b32 [ %r608 + 0 ], { %r626, %r627, %r628, %r629 };
	// end inline asm
	mov.b32 	%r630, {%rs57, %rs58};
	mov.b32 	%r631, {%rs59, %rs60};
	mov.b32 	%r632, {%rs61, %rs62};
	mov.b32 	%r633, {%rs63, %rs64};
	// begin inline asm
	st.shared.v4.b32 [ %r613 + 0 ], { %r630, %r631, %r632, %r633 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r939, %r940, %r941, %r942}, [%r920];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r943, %r944, %r945, %r946}, [%r920+1024];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r947, %r948, %r949, %r950}, [%r930];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r951, %r952, %r953, %r954}, [%r930+1024];
	bar.sync 	0;
	mov.b32 	%r634, {%rs65, %rs66};
	mov.b32 	%r635, {%rs67, %rs68};
	mov.b32 	%r636, {%rs69, %rs70};
	mov.b32 	%r637, {%rs71, %rs72};
	// begin inline asm
	st.shared.v4.b32 [ %r598 + 0 ], { %r634, %r635, %r636, %r637 };
	// end inline asm
	mov.b32 	%r638, {%rs73, %rs74};
	mov.b32 	%r639, {%rs75, %rs76};
	mov.b32 	%r640, {%rs77, %rs78};
	mov.b32 	%r641, {%rs79, %rs80};
	// begin inline asm
	st.shared.v4.b32 [ %r603 + 0 ], { %r638, %r639, %r640, %r641 };
	// end inline asm
	mov.b32 	%r642, {%rs81, %rs82};
	mov.b32 	%r643, {%rs83, %rs84};
	mov.b32 	%r644, {%rs85, %rs86};
	mov.b32 	%r645, {%rs87, %rs88};
	// begin inline asm
	st.shared.v4.b32 [ %r608 + 0 ], { %r642, %r643, %r644, %r645 };
	// end inline asm
	mov.b32 	%r646, {%rs89, %rs90};
	mov.b32 	%r647, {%rs91, %rs92};
	mov.b32 	%r648, {%rs93, %rs94};
	mov.b32 	%r649, {%rs95, %rs96};
	// begin inline asm
	st.shared.v4.b32 [ %r613 + 0 ], { %r646, %r647, %r648, %r649 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r955, %r956, %r957, %r958}, [%r920];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r959, %r960, %r961, %r962}, [%r920+1024];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r963, %r964, %r965, %r966}, [%r930];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r967, %r968, %r969, %r970}, [%r930+1024];
	bar.sync 	0;
	mov.b32 	%r650, {%rs97, %rs98};
	mov.b32 	%r651, {%rs99, %rs100};
	mov.b32 	%r652, {%rs101, %rs102};
	mov.b32 	%r653, {%rs103, %rs104};
	// begin inline asm
	st.shared.v4.b32 [ %r598 + 0 ], { %r650, %r651, %r652, %r653 };
	// end inline asm
	mov.b32 	%r654, {%rs105, %rs106};
	mov.b32 	%r655, {%rs107, %rs108};
	mov.b32 	%r656, {%rs109, %rs110};
	mov.b32 	%r657, {%rs111, %rs112};
	// begin inline asm
	st.shared.v4.b32 [ %r603 + 0 ], { %r654, %r655, %r656, %r657 };
	// end inline asm
	mov.b32 	%r658, {%rs113, %rs114};
	mov.b32 	%r659, {%rs115, %rs116};
	mov.b32 	%r660, {%rs117, %rs118};
	mov.b32 	%r661, {%rs119, %rs120};
	// begin inline asm
	st.shared.v4.b32 [ %r608 + 0 ], { %r658, %r659, %r660, %r661 };
	// end inline asm
	mov.b32 	%r662, {%rs121, %rs122};
	mov.b32 	%r663, {%rs123, %rs124};
	mov.b32 	%r664, {%rs125, %rs126};
	mov.b32 	%r665, {%rs127, %rs128};
	// begin inline asm
	st.shared.v4.b32 [ %r613 + 0 ], { %r662, %r663, %r664, %r665 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r971, %r972, %r973, %r974}, [%r920];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r975, %r976, %r977, %r978}, [%r920+1024];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r979, %r980, %r981, %r982}, [%r930];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r983, %r984, %r985, %r986}, [%r930+1024];
	.loc	1 357 31                        // sk05_mlp_gateup.py:357:31
	setp.lt.s32 	%p25, %r831, %r23;
	setp.lt.s32 	%p26, %r860, %r23;
	setp.lt.s32 	%p27, %r858, %r23;
	setp.lt.s32 	%p28, %r856, %r23;
	setp.lt.s32 	%p29, %r854, %r23;
	setp.lt.s32 	%p30, %r852, %r23;
	setp.lt.s32 	%p31, %r850, %r23;
	setp.lt.s32 	%p32, %r848, %r23;
	setp.lt.s32 	%p33, %r846, %r23;
	setp.lt.s32 	%p34, %r844, %r23;
	setp.lt.s32 	%p35, %r842, %r23;
	setp.lt.s32 	%p36, %r840, %r23;
	setp.lt.s32 	%p37, %r838, %r23;
	setp.lt.s32 	%p38, %r836, %r23;
	setp.lt.s32 	%p39, %r834, %r23;
	setp.lt.s32 	%p40, %r832, %r23;
	.loc	1 357 54                        // sk05_mlp_gateup.py:357:54
	setp.lt.s32 	%p41, %r811, %r24;
	.loc	1 357 37                        // sk05_mlp_gateup.py:357:37
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
	.loc	1 355 35                        // sk05_mlp_gateup.py:355:35
	mul.lo.s32 	%r987, %r831, %r25;
	mul.lo.s32 	%r988, %r860, %r25;
	mul.lo.s32 	%r989, %r858, %r25;
	mul.lo.s32 	%r990, %r856, %r25;
	mul.lo.s32 	%r991, %r854, %r25;
	mul.lo.s32 	%r992, %r852, %r25;
	mul.lo.s32 	%r993, %r850, %r25;
	mul.lo.s32 	%r994, %r848, %r25;
	mul.lo.s32 	%r995, %r846, %r25;
	mul.lo.s32 	%r996, %r844, %r25;
	mul.lo.s32 	%r997, %r842, %r25;
	mul.lo.s32 	%r998, %r840, %r25;
	mul.lo.s32 	%r999, %r838, %r25;
	mul.lo.s32 	%r1000, %r836, %r25;
	mul.lo.s32 	%r1001, %r834, %r25;
	mul.lo.s32 	%r1002, %r832, %r25;
	.loc	1 355 18                        // sk05_mlp_gateup.py:355:18
	mad.wide.s32 	%rd260, %r987, 2, %rd26;
	mad.wide.s32 	%rd261, %r988, 2, %rd26;
	mad.wide.s32 	%rd262, %r989, 2, %rd26;
	mad.wide.s32 	%rd263, %r990, 2, %rd26;
	mad.wide.s32 	%rd264, %r991, 2, %rd26;
	mad.wide.s32 	%rd265, %r992, 2, %rd26;
	mad.wide.s32 	%rd266, %r993, 2, %rd26;
	mad.wide.s32 	%rd267, %r994, 2, %rd26;
	mad.wide.s32 	%rd268, %r995, 2, %rd26;
	mad.wide.s32 	%rd269, %r996, 2, %rd26;
	mad.wide.s32 	%rd270, %r997, 2, %rd26;
	mad.wide.s32 	%rd271, %r998, 2, %rd26;
	mad.wide.s32 	%rd272, %r999, 2, %rd26;
	mad.wide.s32 	%rd273, %r1000, 2, %rd26;
	mad.wide.s32 	%rd274, %r1001, 2, %rd26;
	mad.wide.s32 	%rd275, %r1002, 2, %rd26;
	.loc	1 355 50                        // sk05_mlp_gateup.py:355:50
	mul.wide.s32 	%rd276, %r811, 2;
	add.s64 	%rd220, %rd260, %rd276;
	add.s64 	%rd221, %rd261, %rd276;
	add.s64 	%rd222, %rd262, %rd276;
	add.s64 	%rd223, %rd263, %rd276;
	add.s64 	%rd224, %rd264, %rd276;
	add.s64 	%rd225, %rd265, %rd276;
	add.s64 	%rd226, %rd266, %rd276;
	add.s64 	%rd227, %rd267, %rd276;
	add.s64 	%rd228, %rd268, %rd276;
	add.s64 	%rd229, %rd269, %rd276;
	add.s64 	%rd230, %rd270, %rd276;
	add.s64 	%rd231, %rd271, %rd276;
	add.s64 	%rd232, %rd272, %rd276;
	add.s64 	%rd233, %rd273, %rd276;
	add.s64 	%rd234, %rd274, %rd276;
	add.s64 	%rd235, %rd275, %rd276;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1003, %r1515, %r880;
	mul.f32 	%r1004, %r1514, %r880;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs129, %rs130}, %r971;
	cvt.f32.bf16 	%r1005, %rs130;
	cvt.f32.bf16 	%r1006, %rs129;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1007, %r1004, %r582, %r1006;
	fma.rn.f32 	%r1008, %r1003, %r583, %r1005;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1009, %r1419, %r874;
	mul.f32 	%r1010, %r1418, %r874;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs131, %rs132}, %r921;
	cvt.f32.bf16 	%r1011, %rs132;
	cvt.f32.bf16 	%r1012, %rs131;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1013, %r1010, %r582, %r1012;
	fma.rn.f32 	%r1014, %r1009, %r583, %r1011;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r667, %r1014, %r1013;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1015, %r1421, %r875;
	mul.f32 	%r1016, %r1420, %r875;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs133, %rs134}, %r922;
	cvt.f32.bf16 	%r1017, %rs134;
	cvt.f32.bf16 	%r1018, %rs133;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1019, %r1016, %r582, %r1018;
	fma.rn.f32 	%r1020, %r1015, %r583, %r1017;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r672, %r1020, %r1019;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1021, %r1451, %r876;
	mul.f32 	%r1022, %r1450, %r876;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs135, %rs136}, %r939;
	cvt.f32.bf16 	%r1023, %rs136;
	cvt.f32.bf16 	%r1024, %rs135;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1025, %r1022, %r582, %r1024;
	fma.rn.f32 	%r1026, %r1021, %r583, %r1023;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r668, %r1026, %r1025;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1027, %r1453, %r877;
	mul.f32 	%r1028, %r1452, %r877;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs137, %rs138}, %r940;
	cvt.f32.bf16 	%r1029, %rs138;
	cvt.f32.bf16 	%r1030, %rs137;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1031, %r1028, %r582, %r1030;
	fma.rn.f32 	%r1032, %r1027, %r583, %r1029;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r673, %r1032, %r1031;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1033, %r1483, %r878;
	mul.f32 	%r1034, %r1482, %r878;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs139, %rs140}, %r955;
	cvt.f32.bf16 	%r1035, %rs140;
	cvt.f32.bf16 	%r1036, %rs139;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1037, %r1034, %r582, %r1036;
	fma.rn.f32 	%r1038, %r1033, %r583, %r1035;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r669, %r1038, %r1037;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1039, %r1485, %r879;
	mul.f32 	%r1040, %r1484, %r879;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs141, %rs142}, %r956;
	cvt.f32.bf16 	%r1041, %rs142;
	cvt.f32.bf16 	%r1042, %rs141;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1043, %r1040, %r582, %r1042;
	fma.rn.f32 	%r1044, %r1039, %r583, %r1041;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r674, %r1044, %r1043;
	cvt.rn.bf16x2.f32 	%r670, %r1008, %r1007;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1045, %r1517, %r881;
	mul.f32 	%r1046, %r1516, %r881;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs143, %rs144}, %r972;
	cvt.f32.bf16 	%r1047, %rs144;
	cvt.f32.bf16 	%r1048, %rs143;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1049, %r1046, %r582, %r1048;
	fma.rn.f32 	%r1050, %r1045, %r583, %r1047;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r675, %r1050, %r1049;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1051, %r1519, %r880;
	mul.f32 	%r1052, %r1518, %r880;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs145, %rs146}, %r973;
	cvt.f32.bf16 	%r1053, %rs146;
	cvt.f32.bf16 	%r1054, %rs145;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1055, %r1052, %r584, %r1054;
	fma.rn.f32 	%r1056, %r1051, %r585, %r1053;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1057, %r1423, %r874;
	mul.f32 	%r1058, %r1422, %r874;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs147, %rs148}, %r923;
	cvt.f32.bf16 	%r1059, %rs148;
	cvt.f32.bf16 	%r1060, %rs147;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1061, %r1058, %r584, %r1060;
	fma.rn.f32 	%r1062, %r1057, %r585, %r1059;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r687, %r1062, %r1061;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1063, %r1425, %r875;
	mul.f32 	%r1064, %r1424, %r875;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs149, %rs150}, %r924;
	cvt.f32.bf16 	%r1065, %rs150;
	cvt.f32.bf16 	%r1066, %rs149;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1067, %r1064, %r584, %r1066;
	fma.rn.f32 	%r1068, %r1063, %r585, %r1065;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r692, %r1068, %r1067;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1069, %r1455, %r876;
	mul.f32 	%r1070, %r1454, %r876;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs151, %rs152}, %r941;
	cvt.f32.bf16 	%r1071, %rs152;
	cvt.f32.bf16 	%r1072, %rs151;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1073, %r1070, %r584, %r1072;
	fma.rn.f32 	%r1074, %r1069, %r585, %r1071;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r688, %r1074, %r1073;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1075, %r1457, %r877;
	mul.f32 	%r1076, %r1456, %r877;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs153, %rs154}, %r942;
	cvt.f32.bf16 	%r1077, %rs154;
	cvt.f32.bf16 	%r1078, %rs153;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1079, %r1076, %r584, %r1078;
	fma.rn.f32 	%r1080, %r1075, %r585, %r1077;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r693, %r1080, %r1079;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1081, %r1487, %r878;
	mul.f32 	%r1082, %r1486, %r878;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs155, %rs156}, %r957;
	cvt.f32.bf16 	%r1083, %rs156;
	cvt.f32.bf16 	%r1084, %rs155;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1085, %r1082, %r584, %r1084;
	fma.rn.f32 	%r1086, %r1081, %r585, %r1083;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r689, %r1086, %r1085;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1087, %r1489, %r879;
	mul.f32 	%r1088, %r1488, %r879;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs157, %rs158}, %r958;
	cvt.f32.bf16 	%r1089, %rs158;
	cvt.f32.bf16 	%r1090, %rs157;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1091, %r1088, %r584, %r1090;
	fma.rn.f32 	%r1092, %r1087, %r585, %r1089;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r694, %r1092, %r1091;
	cvt.rn.bf16x2.f32 	%r690, %r1056, %r1055;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1093, %r1521, %r881;
	mul.f32 	%r1094, %r1520, %r881;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs159, %rs160}, %r974;
	cvt.f32.bf16 	%r1095, %rs160;
	cvt.f32.bf16 	%r1096, %rs159;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1097, %r1094, %r584, %r1096;
	fma.rn.f32 	%r1098, %r1093, %r585, %r1095;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r695, %r1098, %r1097;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1099, %r1523, %r880;
	mul.f32 	%r1100, %r1522, %r880;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs161, %rs162}, %r979;
	cvt.f32.bf16 	%r1101, %rs162;
	cvt.f32.bf16 	%r1102, %rs161;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1103, %r1100, %r586, %r1102;
	fma.rn.f32 	%r1104, %r1099, %r587, %r1101;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1105, %r1427, %r874;
	mul.f32 	%r1106, %r1426, %r874;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs163, %rs164}, %r931;
	cvt.f32.bf16 	%r1107, %rs164;
	cvt.f32.bf16 	%r1108, %rs163;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1109, %r1106, %r586, %r1108;
	fma.rn.f32 	%r1110, %r1105, %r587, %r1107;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r707, %r1110, %r1109;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1111, %r1429, %r875;
	mul.f32 	%r1112, %r1428, %r875;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs165, %rs166}, %r932;
	cvt.f32.bf16 	%r1113, %rs166;
	cvt.f32.bf16 	%r1114, %rs165;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1115, %r1112, %r586, %r1114;
	fma.rn.f32 	%r1116, %r1111, %r587, %r1113;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r712, %r1116, %r1115;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1117, %r1459, %r876;
	mul.f32 	%r1118, %r1458, %r876;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs167, %rs168}, %r947;
	cvt.f32.bf16 	%r1119, %rs168;
	cvt.f32.bf16 	%r1120, %rs167;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1121, %r1118, %r586, %r1120;
	fma.rn.f32 	%r1122, %r1117, %r587, %r1119;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r708, %r1122, %r1121;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1123, %r1461, %r877;
	mul.f32 	%r1124, %r1460, %r877;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs169, %rs170}, %r948;
	cvt.f32.bf16 	%r1125, %rs170;
	cvt.f32.bf16 	%r1126, %rs169;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1127, %r1124, %r586, %r1126;
	fma.rn.f32 	%r1128, %r1123, %r587, %r1125;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r713, %r1128, %r1127;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1129, %r1491, %r878;
	mul.f32 	%r1130, %r1490, %r878;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs171, %rs172}, %r963;
	cvt.f32.bf16 	%r1131, %rs172;
	cvt.f32.bf16 	%r1132, %rs171;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1133, %r1130, %r586, %r1132;
	fma.rn.f32 	%r1134, %r1129, %r587, %r1131;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r709, %r1134, %r1133;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1135, %r1493, %r879;
	mul.f32 	%r1136, %r1492, %r879;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs173, %rs174}, %r964;
	cvt.f32.bf16 	%r1137, %rs174;
	cvt.f32.bf16 	%r1138, %rs173;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1139, %r1136, %r586, %r1138;
	fma.rn.f32 	%r1140, %r1135, %r587, %r1137;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r714, %r1140, %r1139;
	cvt.rn.bf16x2.f32 	%r710, %r1104, %r1103;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1141, %r1525, %r881;
	mul.f32 	%r1142, %r1524, %r881;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs175, %rs176}, %r980;
	cvt.f32.bf16 	%r1143, %rs176;
	cvt.f32.bf16 	%r1144, %rs175;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1145, %r1142, %r586, %r1144;
	fma.rn.f32 	%r1146, %r1141, %r587, %r1143;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r715, %r1146, %r1145;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1147, %r1527, %r880;
	mul.f32 	%r1148, %r1526, %r880;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs177, %rs178}, %r981;
	cvt.f32.bf16 	%r1149, %rs178;
	cvt.f32.bf16 	%r1150, %rs177;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1151, %r1148, %r588, %r1150;
	fma.rn.f32 	%r1152, %r1147, %r589, %r1149;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1153, %r1431, %r874;
	mul.f32 	%r1154, %r1430, %r874;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs179, %rs180}, %r933;
	cvt.f32.bf16 	%r1155, %rs180;
	cvt.f32.bf16 	%r1156, %rs179;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1157, %r1154, %r588, %r1156;
	fma.rn.f32 	%r1158, %r1153, %r589, %r1155;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r727, %r1158, %r1157;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1159, %r1433, %r875;
	mul.f32 	%r1160, %r1432, %r875;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs181, %rs182}, %r934;
	cvt.f32.bf16 	%r1161, %rs182;
	cvt.f32.bf16 	%r1162, %rs181;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1163, %r1160, %r588, %r1162;
	fma.rn.f32 	%r1164, %r1159, %r589, %r1161;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r732, %r1164, %r1163;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1165, %r1463, %r876;
	mul.f32 	%r1166, %r1462, %r876;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs183, %rs184}, %r949;
	cvt.f32.bf16 	%r1167, %rs184;
	cvt.f32.bf16 	%r1168, %rs183;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1169, %r1166, %r588, %r1168;
	fma.rn.f32 	%r1170, %r1165, %r589, %r1167;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r728, %r1170, %r1169;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1171, %r1465, %r877;
	mul.f32 	%r1172, %r1464, %r877;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs185, %rs186}, %r950;
	cvt.f32.bf16 	%r1173, %rs186;
	cvt.f32.bf16 	%r1174, %rs185;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1175, %r1172, %r588, %r1174;
	fma.rn.f32 	%r1176, %r1171, %r589, %r1173;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r733, %r1176, %r1175;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1177, %r1495, %r878;
	mul.f32 	%r1178, %r1494, %r878;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs187, %rs188}, %r965;
	cvt.f32.bf16 	%r1179, %rs188;
	cvt.f32.bf16 	%r1180, %rs187;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1181, %r1178, %r588, %r1180;
	fma.rn.f32 	%r1182, %r1177, %r589, %r1179;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r729, %r1182, %r1181;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1183, %r1497, %r879;
	mul.f32 	%r1184, %r1496, %r879;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs189, %rs190}, %r966;
	cvt.f32.bf16 	%r1185, %rs190;
	cvt.f32.bf16 	%r1186, %rs189;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1187, %r1184, %r588, %r1186;
	fma.rn.f32 	%r1188, %r1183, %r589, %r1185;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r734, %r1188, %r1187;
	cvt.rn.bf16x2.f32 	%r730, %r1152, %r1151;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1189, %r1529, %r881;
	mul.f32 	%r1190, %r1528, %r881;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs191, %rs192}, %r982;
	cvt.f32.bf16 	%r1191, %rs192;
	cvt.f32.bf16 	%r1192, %rs191;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1193, %r1190, %r588, %r1192;
	fma.rn.f32 	%r1194, %r1189, %r589, %r1191;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r735, %r1194, %r1193;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1195, %r1531, %r880;
	mul.f32 	%r1196, %r1530, %r880;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs193, %rs194}, %r975;
	cvt.f32.bf16 	%r1197, %rs194;
	cvt.f32.bf16 	%r1198, %rs193;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1199, %r1196, %r590, %r1198;
	fma.rn.f32 	%r1200, %r1195, %r591, %r1197;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1201, %r1435, %r874;
	mul.f32 	%r1202, %r1434, %r874;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs195, %rs196}, %r925;
	cvt.f32.bf16 	%r1203, %rs196;
	cvt.f32.bf16 	%r1204, %rs195;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1205, %r1202, %r590, %r1204;
	fma.rn.f32 	%r1206, %r1201, %r591, %r1203;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r677, %r1206, %r1205;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1207, %r1437, %r875;
	mul.f32 	%r1208, %r1436, %r875;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs197, %rs198}, %r926;
	cvt.f32.bf16 	%r1209, %rs198;
	cvt.f32.bf16 	%r1210, %rs197;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1211, %r1208, %r590, %r1210;
	fma.rn.f32 	%r1212, %r1207, %r591, %r1209;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r682, %r1212, %r1211;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1213, %r1467, %r876;
	mul.f32 	%r1214, %r1466, %r876;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs199, %rs200}, %r943;
	cvt.f32.bf16 	%r1215, %rs200;
	cvt.f32.bf16 	%r1216, %rs199;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1217, %r1214, %r590, %r1216;
	fma.rn.f32 	%r1218, %r1213, %r591, %r1215;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r678, %r1218, %r1217;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1219, %r1469, %r877;
	mul.f32 	%r1220, %r1468, %r877;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs201, %rs202}, %r944;
	cvt.f32.bf16 	%r1221, %rs202;
	cvt.f32.bf16 	%r1222, %rs201;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1223, %r1220, %r590, %r1222;
	fma.rn.f32 	%r1224, %r1219, %r591, %r1221;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r683, %r1224, %r1223;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1225, %r1499, %r878;
	mul.f32 	%r1226, %r1498, %r878;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs203, %rs204}, %r959;
	cvt.f32.bf16 	%r1227, %rs204;
	cvt.f32.bf16 	%r1228, %rs203;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1229, %r1226, %r590, %r1228;
	fma.rn.f32 	%r1230, %r1225, %r591, %r1227;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r679, %r1230, %r1229;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1231, %r1501, %r879;
	mul.f32 	%r1232, %r1500, %r879;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs205, %rs206}, %r960;
	cvt.f32.bf16 	%r1233, %rs206;
	cvt.f32.bf16 	%r1234, %rs205;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1235, %r1232, %r590, %r1234;
	fma.rn.f32 	%r1236, %r1231, %r591, %r1233;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r684, %r1236, %r1235;
	cvt.rn.bf16x2.f32 	%r680, %r1200, %r1199;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1237, %r1533, %r881;
	mul.f32 	%r1238, %r1532, %r881;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs207, %rs208}, %r976;
	cvt.f32.bf16 	%r1239, %rs208;
	cvt.f32.bf16 	%r1240, %rs207;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1241, %r1238, %r590, %r1240;
	fma.rn.f32 	%r1242, %r1237, %r591, %r1239;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r685, %r1242, %r1241;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1243, %r1535, %r880;
	mul.f32 	%r1244, %r1534, %r880;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs209, %rs210}, %r977;
	cvt.f32.bf16 	%r1245, %rs210;
	cvt.f32.bf16 	%r1246, %rs209;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1247, %r1244, %r592, %r1246;
	fma.rn.f32 	%r1248, %r1243, %r593, %r1245;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1249, %r1439, %r874;
	mul.f32 	%r1250, %r1438, %r874;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs211, %rs212}, %r927;
	cvt.f32.bf16 	%r1251, %rs212;
	cvt.f32.bf16 	%r1252, %rs211;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1253, %r1250, %r592, %r1252;
	fma.rn.f32 	%r1254, %r1249, %r593, %r1251;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r697, %r1254, %r1253;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1255, %r1441, %r875;
	mul.f32 	%r1256, %r1440, %r875;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs213, %rs214}, %r928;
	cvt.f32.bf16 	%r1257, %rs214;
	cvt.f32.bf16 	%r1258, %rs213;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1259, %r1256, %r592, %r1258;
	fma.rn.f32 	%r1260, %r1255, %r593, %r1257;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r702, %r1260, %r1259;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1261, %r1471, %r876;
	mul.f32 	%r1262, %r1470, %r876;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs215, %rs216}, %r945;
	cvt.f32.bf16 	%r1263, %rs216;
	cvt.f32.bf16 	%r1264, %rs215;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1265, %r1262, %r592, %r1264;
	fma.rn.f32 	%r1266, %r1261, %r593, %r1263;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r698, %r1266, %r1265;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1267, %r1473, %r877;
	mul.f32 	%r1268, %r1472, %r877;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs217, %rs218}, %r946;
	cvt.f32.bf16 	%r1269, %rs218;
	cvt.f32.bf16 	%r1270, %rs217;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1271, %r1268, %r592, %r1270;
	fma.rn.f32 	%r1272, %r1267, %r593, %r1269;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r703, %r1272, %r1271;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1273, %r1503, %r878;
	mul.f32 	%r1274, %r1502, %r878;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs219, %rs220}, %r961;
	cvt.f32.bf16 	%r1275, %rs220;
	cvt.f32.bf16 	%r1276, %rs219;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1277, %r1274, %r592, %r1276;
	fma.rn.f32 	%r1278, %r1273, %r593, %r1275;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r699, %r1278, %r1277;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1279, %r1505, %r879;
	mul.f32 	%r1280, %r1504, %r879;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs221, %rs222}, %r962;
	cvt.f32.bf16 	%r1281, %rs222;
	cvt.f32.bf16 	%r1282, %rs221;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1283, %r1280, %r592, %r1282;
	fma.rn.f32 	%r1284, %r1279, %r593, %r1281;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r704, %r1284, %r1283;
	cvt.rn.bf16x2.f32 	%r700, %r1248, %r1247;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1285, %r1537, %r881;
	mul.f32 	%r1286, %r1536, %r881;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs223, %rs224}, %r978;
	cvt.f32.bf16 	%r1287, %rs224;
	cvt.f32.bf16 	%r1288, %rs223;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1289, %r1286, %r592, %r1288;
	fma.rn.f32 	%r1290, %r1285, %r593, %r1287;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r705, %r1290, %r1289;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1291, %r1539, %r880;
	mul.f32 	%r1292, %r1538, %r880;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs225, %rs226}, %r983;
	cvt.f32.bf16 	%r1293, %rs226;
	cvt.f32.bf16 	%r1294, %rs225;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1295, %r1292, %r594, %r1294;
	fma.rn.f32 	%r1296, %r1291, %r595, %r1293;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1297, %r1443, %r874;
	mul.f32 	%r1298, %r1442, %r874;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs227, %rs228}, %r935;
	cvt.f32.bf16 	%r1299, %rs228;
	cvt.f32.bf16 	%r1300, %rs227;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1301, %r1298, %r594, %r1300;
	fma.rn.f32 	%r1302, %r1297, %r595, %r1299;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r717, %r1302, %r1301;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1303, %r1445, %r875;
	mul.f32 	%r1304, %r1444, %r875;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs229, %rs230}, %r936;
	cvt.f32.bf16 	%r1305, %rs230;
	cvt.f32.bf16 	%r1306, %rs229;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1307, %r1304, %r594, %r1306;
	fma.rn.f32 	%r1308, %r1303, %r595, %r1305;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r722, %r1308, %r1307;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1309, %r1475, %r876;
	mul.f32 	%r1310, %r1474, %r876;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs231, %rs232}, %r951;
	cvt.f32.bf16 	%r1311, %rs232;
	cvt.f32.bf16 	%r1312, %rs231;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1313, %r1310, %r594, %r1312;
	fma.rn.f32 	%r1314, %r1309, %r595, %r1311;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r718, %r1314, %r1313;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1315, %r1477, %r877;
	mul.f32 	%r1316, %r1476, %r877;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs233, %rs234}, %r952;
	cvt.f32.bf16 	%r1317, %rs234;
	cvt.f32.bf16 	%r1318, %rs233;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1319, %r1316, %r594, %r1318;
	fma.rn.f32 	%r1320, %r1315, %r595, %r1317;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r723, %r1320, %r1319;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1321, %r1507, %r878;
	mul.f32 	%r1322, %r1506, %r878;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs235, %rs236}, %r967;
	cvt.f32.bf16 	%r1323, %rs236;
	cvt.f32.bf16 	%r1324, %rs235;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1325, %r1322, %r594, %r1324;
	fma.rn.f32 	%r1326, %r1321, %r595, %r1323;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r719, %r1326, %r1325;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1327, %r1509, %r879;
	mul.f32 	%r1328, %r1508, %r879;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs237, %rs238}, %r968;
	cvt.f32.bf16 	%r1329, %rs238;
	cvt.f32.bf16 	%r1330, %rs237;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1331, %r1328, %r594, %r1330;
	fma.rn.f32 	%r1332, %r1327, %r595, %r1329;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r724, %r1332, %r1331;
	cvt.rn.bf16x2.f32 	%r720, %r1296, %r1295;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1333, %r1541, %r881;
	mul.f32 	%r1334, %r1540, %r881;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs239, %rs240}, %r984;
	cvt.f32.bf16 	%r1335, %rs240;
	cvt.f32.bf16 	%r1336, %rs239;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1337, %r1334, %r594, %r1336;
	fma.rn.f32 	%r1338, %r1333, %r595, %r1335;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r725, %r1338, %r1337;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1339, %r1543, %r880;
	mul.f32 	%r1340, %r1542, %r880;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs241, %rs242}, %r985;
	cvt.f32.bf16 	%r1341, %rs242;
	cvt.f32.bf16 	%r1342, %rs241;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1343, %r1340, %r596, %r1342;
	fma.rn.f32 	%r1344, %r1339, %r597, %r1341;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1345, %r1447, %r874;
	mul.f32 	%r1346, %r1446, %r874;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs243, %rs244}, %r937;
	cvt.f32.bf16 	%r1347, %rs244;
	cvt.f32.bf16 	%r1348, %rs243;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1349, %r1346, %r596, %r1348;
	fma.rn.f32 	%r1350, %r1345, %r597, %r1347;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r737, %r1350, %r1349;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1351, %r1449, %r875;
	mul.f32 	%r1352, %r1448, %r875;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs245, %rs246}, %r938;
	cvt.f32.bf16 	%r1353, %rs246;
	cvt.f32.bf16 	%r1354, %rs245;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1355, %r1352, %r596, %r1354;
	fma.rn.f32 	%r1356, %r1351, %r597, %r1353;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r742, %r1356, %r1355;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1357, %r1479, %r876;
	mul.f32 	%r1358, %r1478, %r876;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs247, %rs248}, %r953;
	cvt.f32.bf16 	%r1359, %rs248;
	cvt.f32.bf16 	%r1360, %rs247;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1361, %r1358, %r596, %r1360;
	fma.rn.f32 	%r1362, %r1357, %r597, %r1359;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r738, %r1362, %r1361;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1363, %r1481, %r877;
	mul.f32 	%r1364, %r1480, %r877;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs249, %rs250}, %r954;
	cvt.f32.bf16 	%r1365, %rs250;
	cvt.f32.bf16 	%r1366, %rs249;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1367, %r1364, %r596, %r1366;
	fma.rn.f32 	%r1368, %r1363, %r597, %r1365;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r743, %r1368, %r1367;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1369, %r1511, %r878;
	mul.f32 	%r1370, %r1510, %r878;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs251, %rs252}, %r969;
	cvt.f32.bf16 	%r1371, %rs252;
	cvt.f32.bf16 	%r1372, %rs251;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1373, %r1370, %r596, %r1372;
	fma.rn.f32 	%r1374, %r1369, %r597, %r1371;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r739, %r1374, %r1373;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1375, %r1513, %r879;
	mul.f32 	%r1376, %r1512, %r879;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs253, %rs254}, %r970;
	cvt.f32.bf16 	%r1377, %rs254;
	cvt.f32.bf16 	%r1378, %rs253;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1379, %r1376, %r596, %r1378;
	fma.rn.f32 	%r1380, %r1375, %r597, %r1377;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r744, %r1380, %r1379;
	cvt.rn.bf16x2.f32 	%r740, %r1344, %r1343;
	.loc	1 348 16                        // sk05_mlp_gateup.py:348:16
	mul.f32 	%r1381, %r1545, %r881;
	mul.f32 	%r1382, %r1544, %r881;
	.loc	1 350 99                        // sk05_mlp_gateup.py:350:99
	mov.b32 	{%rs255, %rs256}, %r986;
	cvt.f32.bf16 	%r1383, %rs256;
	cvt.f32.bf16 	%r1384, %rs255;
	.loc	1 350 11                        // sk05_mlp_gateup.py:350:11
	fma.rn.f32 	%r1385, %r1382, %r596, %r1384;
	fma.rn.f32 	%r1386, %r1381, %r597, %r1383;
	.loc	1 356 15                        // sk05_mlp_gateup.py:356:15
	cvt.rn.bf16x2.f32 	%r745, %r1386, %r1385;
	bar.sync 	0;
	shl.b32 	%r1387, %r5, 14;
	shl.b32 	%r1388, %r5, 5;
	and.b32 	%r1389, %r1416, 3456;
	bfe.s32 	%r1390, %r2, 2, 1;
	and.b32 	%r1391, %r1390, 8208;
	or.b32 	%r1392, %r1388, %r1389;
	xor.b32 	%r1393, %r1391, %r915;
	or.b32 	%r1394, %r1393, %r1392;
	or.b32 	%r1395, %r1394, %r1387;
	add.s32 	%r666, %r194, %r1395;
	// begin inline asm
	st.shared.v4.b32 [ %r666 + 0 ], { %r667, %r668, %r669, %r670 };
	// end inline asm
	add.s32 	%r671, %r666, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r671 + 0 ], { %r672, %r673, %r674, %r675 };
	// end inline asm
	add.s32 	%r676, %r666, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r676 + 0 ], { %r677, %r678, %r679, %r680 };
	// end inline asm
	add.s32 	%r681, %r666, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r681 + 0 ], { %r682, %r683, %r684, %r685 };
	// end inline asm
	xor.b32 	%r1396, %r1395, 32;
	add.s32 	%r686, %r194, %r1396;
	// begin inline asm
	st.shared.v4.b32 [ %r686 + 0 ], { %r687, %r688, %r689, %r690 };
	// end inline asm
	add.s32 	%r691, %r686, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r691 + 0 ], { %r692, %r693, %r694, %r695 };
	// end inline asm
	add.s32 	%r696, %r686, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r696 + 0 ], { %r697, %r698, %r699, %r700 };
	// end inline asm
	add.s32 	%r701, %r686, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r701 + 0 ], { %r702, %r703, %r704, %r705 };
	// end inline asm
	xor.b32 	%r1397, %r1395, 64;
	add.s32 	%r706, %r194, %r1397;
	// begin inline asm
	st.shared.v4.b32 [ %r706 + 0 ], { %r707, %r708, %r709, %r710 };
	// end inline asm
	add.s32 	%r711, %r706, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r711 + 0 ], { %r712, %r713, %r714, %r715 };
	// end inline asm
	add.s32 	%r716, %r706, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r716 + 0 ], { %r717, %r718, %r719, %r720 };
	// end inline asm
	add.s32 	%r721, %r706, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r721 + 0 ], { %r722, %r723, %r724, %r725 };
	// end inline asm
	xor.b32 	%r1398, %r1395, 96;
	add.s32 	%r726, %r194, %r1398;
	// begin inline asm
	st.shared.v4.b32 [ %r726 + 0 ], { %r727, %r728, %r729, %r730 };
	// end inline asm
	add.s32 	%r731, %r726, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r731 + 0 ], { %r732, %r733, %r734, %r735 };
	// end inline asm
	add.s32 	%r736, %r726, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r736 + 0 ], { %r737, %r738, %r739, %r740 };
	// end inline asm
	add.s32 	%r741, %r726, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r741 + 0 ], { %r742, %r743, %r744, %r745 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r1399, %r2, 2;
	and.b32 	%r1400, %r1399, 896;
	shl.b32 	%r1401, %r866, 9;
	selp.b32 	%r1402, 0, 8208, %p24;
	or.b32 	%r1403, %r908, %r1400;
	xor.b32 	%r1404, %r1403, %r1402;
	or.b32 	%r1405, %r1404, %r1401;
	add.s32 	%r1406, %r194, %r1405;
	ld.shared.v4.b32 	{%r746, %r762, %r778, %r794}, [%r1406];
	ld.shared.v4.b32 	{%r750, %r766, %r782, %r798}, [%r1406+1024];
	ld.shared.v4.b32 	{%r754, %r770, %r786, %r802}, [%r1406+2048];
	ld.shared.v4.b32 	{%r758, %r774, %r790, %r806}, [%r1406+3072];
	xor.b32 	%r1407, %r1405, 32;
	add.s32 	%r1408, %r194, %r1407;
	ld.shared.v4.b32 	{%r747, %r763, %r779, %r795}, [%r1408+16384];
	ld.shared.v4.b32 	{%r751, %r767, %r783, %r799}, [%r1408+17408];
	ld.shared.v4.b32 	{%r755, %r771, %r787, %r803}, [%r1408+18432];
	ld.shared.v4.b32 	{%r759, %r775, %r791, %r807}, [%r1408+19456];
	xor.b32 	%r1409, %r1405, 64;
	add.s32 	%r1410, %r194, %r1409;
	ld.shared.v4.b32 	{%r748, %r764, %r780, %r796}, [%r1410+32768];
	ld.shared.v4.b32 	{%r752, %r768, %r784, %r800}, [%r1410+33792];
	ld.shared.v4.b32 	{%r756, %r772, %r788, %r804}, [%r1410+34816];
	ld.shared.v4.b32 	{%r760, %r776, %r792, %r808}, [%r1410+35840];
	xor.b32 	%r1411, %r1405, 96;
	add.s32 	%r1412, %r194, %r1411;
	ld.shared.v4.b32 	{%r749, %r765, %r781, %r797}, [%r1412+49152];
	ld.shared.v4.b32 	{%r753, %r769, %r785, %r801}, [%r1412+50176];
	ld.shared.v4.b32 	{%r757, %r773, %r789, %r805}, [%r1412+51200];
	ld.shared.v4.b32 	{%r761, %r777, %r793, %r809}, [%r1412+52224];
	.loc	1 356 8                         // sk05_mlp_gateup.py:356:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd220 + 0 ], { %r746, %r747, %r748, %r749 };
	// end inline asm
	// begin inline asm
	@%p9 st.global.v4.b32 [ %rd221 + 0 ], { %r750, %r751, %r752, %r753 };
	// end inline asm
	// begin inline asm
	@%p10 st.global.v4.b32 [ %rd222 + 0 ], { %r754, %r755, %r756, %r757 };
	// end inline asm
	// begin inline asm
	@%p11 st.global.v4.b32 [ %rd223 + 0 ], { %r758, %r759, %r760, %r761 };
	// end inline asm
	// begin inline asm
	@%p12 st.global.v4.b32 [ %rd224 + 0 ], { %r762, %r763, %r764, %r765 };
	// end inline asm
	// begin inline asm
	@%p13 st.global.v4.b32 [ %rd225 + 0 ], { %r766, %r767, %r768, %r769 };
	// end inline asm
	// begin inline asm
	@%p14 st.global.v4.b32 [ %rd226 + 0 ], { %r770, %r771, %r772, %r773 };
	// end inline asm
	// begin inline asm
	@%p15 st.global.v4.b32 [ %rd227 + 0 ], { %r774, %r775, %r776, %r777 };
	// end inline asm
	// begin inline asm
	@%p16 st.global.v4.b32 [ %rd228 + 0 ], { %r778, %r779, %r780, %r781 };
	// end inline asm
	// begin inline asm
	@%p17 st.global.v4.b32 [ %rd229 + 0 ], { %r782, %r783, %r784, %r785 };
	// end inline asm
	// begin inline asm
	@%p18 st.global.v4.b32 [ %rd230 + 0 ], { %r786, %r787, %r788, %r789 };
	// end inline asm
	// begin inline asm
	@%p19 st.global.v4.b32 [ %rd231 + 0 ], { %r790, %r791, %r792, %r793 };
	// end inline asm
	// begin inline asm
	@%p20 st.global.v4.b32 [ %rd232 + 0 ], { %r794, %r795, %r796, %r797 };
	// end inline asm
	// begin inline asm
	@%p21 st.global.v4.b32 [ %rd233 + 0 ], { %r798, %r799, %r800, %r801 };
	// end inline asm
	// begin inline asm
	@%p22 st.global.v4.b32 [ %rd234 + 0 ], { %r802, %r803, %r804, %r805 };
	// end inline asm
	// begin inline asm
	@%p23 st.global.v4.b32 [ %rd235 + 0 ], { %r806, %r807, %r808, %r809 };
	// end inline asm
	.loc	1 354 4                         // sk05_mlp_gateup.py:354:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/usr/local/lib/python3.12/dist-packages/vllm/_genesis/kernels/sk05_mlp_gateup.py"
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
.b32 201                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0xc2 DW_TAG_compile_unit
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
.b8 53
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 103
.b8 97
.b8 116
.b8 101
.b8 117
.b8 112
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 117
.b8 115
.b8 114
.b8 47
.b8 108
.b8 111
.b8 99
.b8 97
.b8 108
.b8 47
.b8 108
.b8 105
.b8 98
.b8 47
.b8 112
.b8 121
.b8 116
.b8 104
.b8 111
.b8 110
.b8 51
.b8 46
.b8 49
.b8 50
.b8 47
.b8 100
.b8 105
.b8 115
.b8 116
.b8 45
.b8 112
.b8 97
.b8 99
.b8 107
.b8 97
.b8 103
.b8 101
.b8 115
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
.b8 2                                   // Abbrev [2] 0x6a:0x1a DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 53
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 103
.b8 97
.b8 116
.b8 101
.b8 117
.b8 112
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x84:0x48 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 106                                // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x99:0x19 DW_TAG_inlined_subroutine
.b32 106                                // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 62                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0xb2:0x19 DW_TAG_inlined_subroutine
.b32 106                                // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 63                                  // DW_AT_call_line
.b8 1
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_5 = _Nativo(
    "sk05_mlp_gateup/tile256x128x64_shift0_abi16",
    _PTX_5, "_sk05_mlp_gateup_kernel",
    warps=8, shared=73728,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 17, 18],
    horneado={11: 1, 12: 1, 15: 1, 19: 1, 20: 256, 21: 128, 22: 64, 23: 8, 24: 128},
    div16=[8, 9, 10, 13, 14, 16, 17],
)


# (cfg, has_shift) -> variantes candidatas. La eleccion final
# entre ellas la hace _lanzar comparando los valores horneados:
# con residual y sin residual el ABI tiene distinta cantidad de
# params, porque stride_res_n solo se especializa cuando vale 1.
_POR_CFG = {}
_POR_CFG.setdefault(((16, 128, 128, 8, 8, 3), False), []).append(_VAR_0)
_POR_CFG.setdefault(((16, 128, 128, 8, 8, 3), False), []).append(_VAR_1)
_POR_CFG.setdefault(((128, 128, 128, 8, 8, 3), False), []).append(_VAR_2)
_POR_CFG.setdefault(((128, 128, 128, 8, 8, 3), False), []).append(_VAR_3)
_POR_CFG.setdefault(((256, 128, 64, 8, 8, 4), False), []).append(_VAR_4)
_POR_CFG.setdefault(((256, 128, 64, 8, 8, 4), False), []).append(_VAR_5)


def _lanzar(grid, cfg, has_shift, *args):
    """Elige la variante cuyos valores horneados coinciden con estos args."""
    cands = _POR_CFG.get((cfg, bool(has_shift)))
    if not cands:
        raise KeyError("no hay PTX embebido para cfg=%r has_shift=%r" % (cfg, has_shift))
    for v in cands:
        if all(args[p] == val for p, val in v.horneado.items()):
            return v((grid,) if isinstance(grid, int) else grid, *args)
    raise KeyError("no hay variante para cfg=%r has_shift=%r con estos strides" % (cfg, has_shift))


# --- quant _sk05_rmsnorm_quant_kernel: 4 variante(s) de PTX embebido ---

_QPTX0_0 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk05_rmsnorm_quant_kernel // -- Begin function _sk05_rmsnorm_quant_kernel
.extern .shared .align 16 .b8 global_smem[];
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
                                        // @_sk05_rmsnorm_quant_kernel
.visible .entry _sk05_rmsnorm_quant_kernel(
	.param .u64 .ptr .global .align 1 _sk05_rmsnorm_quant_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk05_rmsnorm_quant_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk05_rmsnorm_quant_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk05_rmsnorm_quant_kernel_param_3,
	.param .u32 _sk05_rmsnorm_quant_kernel_param_4,
	.param .u32 _sk05_rmsnorm_quant_kernel_param_5,
	.param .u32 _sk05_rmsnorm_quant_kernel_param_6,
	.param .u64 .ptr .global .align 1 _sk05_rmsnorm_quant_kernel_param_7,
	.param .u64 .ptr .global .align 1 _sk05_rmsnorm_quant_kernel_param_8
)
.reqntid 256
{
	.reg .pred 	%p<15>;
	.reg .b16 	%rs<17>;
	.reg .b32 	%r<183>;
	.reg .b64 	%rd<14>;
	.loc	1 287 0                         // sk05_mlp_gateup.py:287:0
$L__func_begin0:
	.loc	1 287 0                         // sk05_mlp_gateup.py:287:0

// %bb.0:                               // %__nv_rsqrtf.exit
	ld.param.b64 	%rd5, [_sk05_rmsnorm_quant_kernel_param_0];
	ld.param.b64 	%rd6, [_sk05_rmsnorm_quant_kernel_param_1];
$L__tmp0:
	.loc	1 293 24                        // sk05_mlp_gateup.py:293:24
	mov.u32 	%r21, %ctaid.x;
	ld.param.b64 	%rd7, [_sk05_rmsnorm_quant_kernel_param_2];
	.loc	1 294 24                        // sk05_mlp_gateup.py:294:24
	mov.u32 	%r22, %tid.x;
	and.b32 	%r23, %r22, 255;
	ld.param.b64 	%rd8, [_sk05_rmsnorm_quant_kernel_param_3];
	and.b32 	%r24, %r22, 31;
	ld.param.b32 	%r25, [_sk05_rmsnorm_quant_kernel_param_4];
	shr.u32 	%r26, %r22, 5;
	ld.param.b32 	%r27, [_sk05_rmsnorm_quant_kernel_param_5];
	shl.b32 	%r28, %r22, 3;
	ld.param.b32 	%r29, [_sk05_rmsnorm_quant_kernel_param_6];
	and.b32 	%r30, %r28, 2040;
	.loc	1 295 18                        // sk05_mlp_gateup.py:295:18
	setp.lt.s32 	%p1, %r30, %r25;
	.loc	1 296 30                        // sk05_mlp_gateup.py:296:30
	mul.lo.s32 	%r31, %r27, %r21;
	.loc	1 296 24                        // sk05_mlp_gateup.py:296:24
	mad.wide.s32 	%rd9, %r31, 2, %rd5;
	.loc	1 296 42                        // sk05_mlp_gateup.py:296:42
	cvt.u64.u32 	%rd10, %r30;
	mul.wide.u32 	%rd11, %r30, 2;
	add.s64 	%rd1, %rd9, %rd11;
	mov.b32 	%r5, 0;
	.loc	1 296 16                        // sk05_mlp_gateup.py:296:16
	// begin inline asm
	mov.u32 %r1, %r5;
	mov.u32 %r2, %r5;
	mov.u32 %r3, %r5;
	mov.u32 %r4, %r5;
	@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
	// end inline asm
	.loc	1 296 73                        // sk05_mlp_gateup.py:296:73
	mov.b32 	{%rs1, %rs2}, %r2;
	cvt.f32.bf16 	%r32, %rs2;
	cvt.f32.bf16 	%r33, %rs1;
	mov.b32 	{%rs3, %rs4}, %r1;
	cvt.f32.bf16 	%r34, %rs3;
	cvt.f32.bf16 	%r35, %rs4;
	mov.b32 	{%rs5, %rs6}, %r4;
	cvt.f32.bf16 	%r36, %rs6;
	cvt.f32.bf16 	%r37, %rs5;
	mov.b32 	{%rs7, %rs8}, %r3;
	cvt.f32.bf16 	%r38, %rs8;
	cvt.f32.bf16 	%r39, %rs7;
	.loc	1 297 24                        // sk05_mlp_gateup.py:297:24
	add.s64 	%rd2, %rd6, %rd11;
	.loc	1 297 16                        // sk05_mlp_gateup.py:297:16
	// begin inline asm
	mov.u32 %r6, %r5;
	mov.u32 %r7, %r5;
	mov.u32 %r8, %r5;
	mov.u32 %r9, %r5;
	@%p1 ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd2 + 0 ];
	// end inline asm
	.loc	1 298 32                        // sk05_mlp_gateup.py:298:32
	mul.f32 	%r40, %r35, %r35;
$L__tmp1:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	fma.rn.f32 	%r41, %r34, %r34, %r40;
	fma.rn.f32 	%r42, %r33, %r33, %r41;
	fma.rn.f32 	%r43, %r32, %r32, %r42;
	fma.rn.f32 	%r44, %r39, %r39, %r43;
	fma.rn.f32 	%r45, %r38, %r38, %r44;
	fma.rn.f32 	%r46, %r37, %r37, %r45;
	fma.rn.f32 	%r47, %r36, %r36, %r46;
$L__tmp2:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	shfl.sync.bfly.b32 	%r48, %r47, 16, 31, -1;
$L__tmp3:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	add.f32 	%r49, %r47, %r48;
$L__tmp4:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	shfl.sync.bfly.b32 	%r50, %r49, 8, 31, -1;
$L__tmp5:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	add.f32 	%r51, %r49, %r50;
$L__tmp6:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	shfl.sync.bfly.b32 	%r52, %r51, 4, 31, -1;
$L__tmp7:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	add.f32 	%r53, %r51, %r52;
$L__tmp8:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	shfl.sync.bfly.b32 	%r54, %r53, 2, 31, -1;
$L__tmp9:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	add.f32 	%r55, %r53, %r54;
$L__tmp10:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	shfl.sync.bfly.b32 	%r56, %r55, 1, 31, -1;
$L__tmp11:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	add.f32 	%r11, %r55, %r56;
$L__tmp12:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	setp.eq.b32 	%p2, %r24, 0;
	shr.u32 	%r57, %r22, 3;
	and.b32 	%r58, %r57, 28;
	mov.b32 	%r59, global_smem;
	add.s32 	%r10, %r59, %r58;
	// begin inline asm
	@%p2 st.shared.b32 [ %r10 + 0 ], %r11;
	// end inline asm
	bar.sync 	0;
	setp.lt.u32 	%p3, %r23, 8;
	shl.b32 	%r60, %r23, 2;
	add.s32 	%r13, %r59, %r60;
	// begin inline asm
	@%p3 ld.shared.b32 %r12, [ %r13 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r61, %r12, 4, 31, -1;
$L__tmp13:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	add.f32 	%r62, %r12, %r61;
$L__tmp14:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	shfl.sync.bfly.b32 	%r63, %r62, 2, 31, -1;
$L__tmp15:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	add.f32 	%r64, %r62, %r63;
$L__tmp16:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	shfl.sync.bfly.b32 	%r65, %r64, 1, 31, -1;
$L__tmp17:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	add.f32 	%r14, %r64, %r65;
$L__tmp18:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	and.b32 	%r66, %r22, 7;
	setp.eq.b32 	%p6, %r66, 0;
	and.pred 	%p4, %p3, %p6;
	// begin inline asm
	@%p4 st.shared.b32 [ %r13 + 0 ], %r14;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r67, [global_smem];
$L__tmp19:
	.loc	1 298 45                        // sk05_mlp_gateup.py:298:45
	cvt.rn.f32.s32 	%r68, %r25;
	div.full.f32 	%r69, %r67, %r68;
	.loc	1 298 49                        // sk05_mlp_gateup.py:298:49
	add.f32 	%r70, %r69, 0f358637BD;
	.loc	1 298 21                        // sk05_mlp_gateup.py:298:21
	rsqrt.approx.ftz.f32 	%r71, %r70;
	.loc	1 298 12                        // sk05_mlp_gateup.py:298:12
	mul.f32 	%r72, %r71, %r34;
	mul.f32 	%r73, %r71, %r35;
	mul.f32 	%r74, %r71, %r33;
	mul.f32 	%r75, %r71, %r32;
	mul.f32 	%r76, %r71, %r39;
	mul.f32 	%r77, %r71, %r38;
	mul.f32 	%r78, %r71, %r37;
	mul.f32 	%r79, %r71, %r36;
$L__tmp20:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	bar.sync 	0;
$L__tmp21:
	.loc	1 302 27                        // sk05_mlp_gateup.py:302:27
	mul.lo.s32 	%r80, %r29, %r21;
	.loc	1 302 21                        // sk05_mlp_gateup.py:302:21
	cvt.s64.s32 	%rd12, %r80;
	add.s64 	%rd13, %rd7, %rd12;
	.loc	1 302 39                        // sk05_mlp_gateup.py:302:39
	add.s64 	%rd3, %rd13, %rd10;
	.loc	1 297 55                        // sk05_mlp_gateup.py:297:55
	mov.b32 	{%rs9, %rs10}, %r8;
	cvt.f32.bf16 	%r81, %rs9;
	cvt.f32.bf16 	%r82, %rs10;
	mov.b32 	{%rs11, %rs12}, %r9;
	cvt.f32.bf16 	%r83, %rs11;
	cvt.f32.bf16 	%r84, %rs12;
	.loc	1 298 56                        // sk05_mlp_gateup.py:298:56
	mul.f32 	%r85, %r79, %r84;
	mul.f32 	%r86, %r78, %r83;
	mul.f32 	%r87, %r77, %r82;
	mul.f32 	%r88, %r76, %r81;
	.loc	1 299 36                        // sk05_mlp_gateup.py:299:36
	abs.f32 	%r89, %r88;
	abs.f32 	%r90, %r87;
	abs.f32 	%r91, %r86;
	abs.f32 	%r92, %r85;
	.loc	1 297 55                        // sk05_mlp_gateup.py:297:55
	mov.b32 	{%rs13, %rs14}, %r6;
	cvt.f32.bf16 	%r93, %rs13;
	cvt.f32.bf16 	%r94, %rs14;
	mov.b32 	{%rs15, %rs16}, %r7;
	cvt.f32.bf16 	%r95, %rs15;
	cvt.f32.bf16 	%r96, %rs16;
	.loc	1 298 56                        // sk05_mlp_gateup.py:298:56
	mul.f32 	%r97, %r75, %r96;
	mul.f32 	%r98, %r74, %r95;
	mul.f32 	%r99, %r73, %r94;
	mul.f32 	%r100, %r72, %r93;
	.loc	1 299 36                        // sk05_mlp_gateup.py:299:36
	abs.f32 	%r101, %r100;
	abs.f32 	%r102, %r99;
	abs.f32 	%r103, %r98;
	abs.f32 	%r104, %r97;
$L__tmp22:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r105, %r101, %r102;
	max.f32 	%r106, %r105, %r103;
	max.f32 	%r107, %r106, %r104;
	max.f32 	%r108, %r107, %r89;
	max.f32 	%r109, %r108, %r90;
	max.f32 	%r110, %r109, %r91;
	max.f32 	%r111, %r110, %r92;
$L__tmp23:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	shfl.sync.bfly.b32 	%r112, %r111, 16, 31, -1;
$L__tmp24:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r113, %r111, %r112;
$L__tmp25:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	shfl.sync.bfly.b32 	%r114, %r113, 8, 31, -1;
$L__tmp26:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r115, %r113, %r114;
$L__tmp27:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	shfl.sync.bfly.b32 	%r116, %r115, 4, 31, -1;
$L__tmp28:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r117, %r115, %r116;
$L__tmp29:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	shfl.sync.bfly.b32 	%r118, %r117, 2, 31, -1;
$L__tmp30:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r119, %r117, %r118;
$L__tmp31:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	shfl.sync.bfly.b32 	%r120, %r119, 1, 31, -1;
$L__tmp32:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r15, %r119, %r120;
$L__tmp33:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	// begin inline asm
	@%p2 st.shared.b32 [ %r10 + 0 ], %r15;
	// end inline asm
	bar.sync 	0;
	// begin inline asm
	@%p3 ld.shared.b32 %r16, [ %r13 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r121, %r16, 4, 31, -1;
$L__tmp34:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r122, %r16, %r121;
$L__tmp35:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	shfl.sync.bfly.b32 	%r123, %r122, 2, 31, -1;
$L__tmp36:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r124, %r122, %r123;
$L__tmp37:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	shfl.sync.bfly.b32 	%r125, %r124, 1, 31, -1;
$L__tmp38:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r17, %r124, %r125;
$L__tmp39:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	// begin inline asm
	@%p4 st.shared.b32 [ %r13 + 0 ], %r17;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r126, [global_smem];
$L__tmp40:
	.loc	1 299 49                        // sk05_mlp_gateup.py:299:49
	max.f32 	%r127, %r126, 0f0DA24260;
	mov.b32 	%r128, 0f42FE0000;
	.loc	1 300 22                        // sk05_mlp_gateup.py:300:22
	div.full.f32 	%r129, %r128, %r127;
	.loc	1 300 14                        // sk05_mlp_gateup.py:300:14
	mul.f32 	%r130, %r99, %r129;
	mul.f32 	%r131, %r100, %r129;
	mul.f32 	%r132, %r97, %r129;
	mul.f32 	%r133, %r98, %r129;
	mul.f32 	%r134, %r87, %r129;
	mul.f32 	%r135, %r88, %r129;
	mul.f32 	%r136, %r85, %r129;
	mul.f32 	%r137, %r86, %r129;
	.loc	1 301 29                        // sk05_mlp_gateup.py:301:29
	.loc	1 301 39                        // sk05_mlp_gateup.py:301:39
	lop3.b32 	%r138, 0x3f000000, %r130, 0x80000000, 0xF8;
	lop3.b32 	%r139, 0x3f000000, %r131, 0x80000000, 0xF8;
	lop3.b32 	%r140, 0x3f000000, %r132, 0x80000000, 0xF8;
	lop3.b32 	%r141, 0x3f000000, %r133, 0x80000000, 0xF8;
	lop3.b32 	%r142, 0x3f000000, %r134, 0x80000000, 0xF8;
	lop3.b32 	%r143, 0x3f000000, %r135, 0x80000000, 0xF8;
	lop3.b32 	%r144, 0x3f000000, %r136, 0x80000000, 0xF8;
	lop3.b32 	%r145, 0x3f000000, %r137, 0x80000000, 0xF8;
	.loc	1 301 14                        // sk05_mlp_gateup.py:301:14
	fma.rn.f32 	%r146, %r98, %r129, %r141;
	fma.rn.f32 	%r147, %r97, %r129, %r140;
	fma.rn.f32 	%r148, %r100, %r129, %r139;
	fma.rn.f32 	%r149, %r99, %r129, %r138;
	fma.rn.f32 	%r150, %r86, %r129, %r145;
	fma.rn.f32 	%r151, %r85, %r129, %r144;
	fma.rn.f32 	%r152, %r88, %r129, %r143;
	fma.rn.f32 	%r153, %r87, %r129, %r142;
	.loc	1 301 49                        // sk05_mlp_gateup.py:301:49
	cvt.rzi.s32.f32 	%r154, %r149;
	cvt.rzi.s32.f32 	%r155, %r148;
	cvt.rzi.s32.f32 	%r156, %r147;
	cvt.rzi.s32.f32 	%r157, %r146;
	cvt.rzi.s32.f32 	%r158, %r153;
	cvt.rzi.s32.f32 	%r159, %r152;
	cvt.rzi.s32.f32 	%r160, %r151;
	cvt.rzi.s32.f32 	%r161, %r150;
	.loc	1 302 70                        // sk05_mlp_gateup.py:302:70
	max.s32 	%r162, %r157, -127;
	max.s32 	%r163, %r156, -127;
	max.s32 	%r164, %r155, -127;
	max.s32 	%r165, %r154, -127;
	max.s32 	%r166, %r161, -127;
	max.s32 	%r167, %r160, -127;
	max.s32 	%r168, %r159, -127;
	max.s32 	%r169, %r158, -127;
	.loc	1 302 77                        // sk05_mlp_gateup.py:302:77
	min.s32 	%r170, %r165, 127;
	min.s32 	%r171, %r164, 127;
	min.s32 	%r172, %r163, 127;
	min.s32 	%r173, %r162, 127;
	min.s32 	%r174, %r169, 127;
	min.s32 	%r175, %r168, 127;
	min.s32 	%r176, %r167, 127;
	min.s32 	%r177, %r166, 127;
	.loc	1 302 85                        // sk05_mlp_gateup.py:302:85
	prmt.b32 	%r178, %r173, %r172, 0x3340U;
	prmt.b32 	%r179, %r171, %r170, 0x3340U;
	prmt.b32 	%r18, %r179, %r178, 0x5410U;
	prmt.b32 	%r180, %r177, %r176, 0x3340U;
	prmt.b32 	%r181, %r175, %r174, 0x3340U;
	prmt.b32 	%r19, %r181, %r180, 0x5410U;
	.loc	1 302 45                        // sk05_mlp_gateup.py:302:45
	// begin inline asm
	@%p1 st.global.v2.b32 [ %rd3 + 0 ], { %r18, %r19 };
	// end inline asm
	.loc	1 303 21                        // sk05_mlp_gateup.py:303:21
	mad.wide.u32 	%rd4, %r21, 4, %rd8;
	.loc	1 303 34                        // sk05_mlp_gateup.py:303:34
	mul.f32 	%r20, %r127, 0f3C010204;
	.loc	1 303 26                        // sk05_mlp_gateup.py:303:26
	or.b32 	%r182, %r24, %r26;
	setp.eq.b32 	%p5, %r182, 0;
	// begin inline asm
	@%p5 st.global.b32 [ %rd4 + 0 ], { %r20 };
	// end inline asm
	.loc	1 303 4                         // sk05_mlp_gateup.py:303:4
	ret;
$L__tmp41:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/usr/local/lib/python3.12/dist-packages/vllm/_genesis/kernels/sk05_mlp_gateup.py"
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
.b8 1                                   // DW_CHILDREN_yes
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
.b8 5                                   // Abbreviation Code
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
.b8 6                                   // Abbreviation Code
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
.b8 11                                  // DW_FORM_data1
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 255                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0xf8 DW_TAG_compile_unit
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
.b8 53
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 103
.b8 97
.b8 116
.b8 101
.b8 117
.b8 112
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 117
.b8 115
.b8 114
.b8 47
.b8 108
.b8 111
.b8 99
.b8 97
.b8 108
.b8 47
.b8 108
.b8 105
.b8 98
.b8 47
.b8 112
.b8 121
.b8 116
.b8 104
.b8 111
.b8 110
.b8 51
.b8 46
.b8 49
.b8 50
.b8 47
.b8 100
.b8 105
.b8 115
.b8 116
.b8 45
.b8 112
.b8 97
.b8 99
.b8 107
.b8 97
.b8 103
.b8 101
.b8 115
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
.b8 2                                   // Abbrev [2] 0x6a:0x1d DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 53
.b8 95
.b8 114
.b8 109
.b8 115
.b8 110
.b8 111
.b8 114
.b8 109
.b8 95
.b8 113
.b8 117
.b8 97
.b8 110
.b8 116
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x87:0x7b DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 106                                // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x9c:0x33 DW_TAG_inlined_subroutine
.b32 106                                // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp19                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 42                                  // DW_AT_call_line
.b8 1
.b8 28                                  // DW_AT_call_column
.b8 5                                   // Abbrev [5] 0xb5:0x19 DW_TAG_inlined_subroutine
.b32 106                                // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp18                          // DW_AT_high_pc
.b8 2                                   // DW_AT_call_file
.b8 37                                  // DW_AT_call_line
.b8 1
.b8 36                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 4                                   // Abbrev [4] 0xcf:0x32 DW_TAG_inlined_subroutine
.b32 106                                // DW_AT_abstract_origin
.b64 $L__tmp20                          // DW_AT_low_pc
.b64 $L__tmp40                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 43                                  // DW_AT_call_line
.b8 1
.b8 29                                  // DW_AT_call_column
.b8 6                                   // Abbrev [6] 0xe8:0x18 DW_TAG_inlined_subroutine
.b32 106                                // DW_AT_abstract_origin
.b64 $L__tmp22                          // DW_AT_low_pc
.b64 $L__tmp39                          // DW_AT_high_pc
.b8 2                                   // DW_AT_call_file
.b8 191                                 // DW_AT_call_line
.b8 40                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_QVAR0_0 = _Nativo(
    "sk05_mlp_gateup/_sk05_rmsnorm_quant_kernel/blk2048",
    _QPTX0_0, "_sk05_rmsnorm_quant_kernel",
    warps=8, shared=32,
    abi=[0, 1, 2, 3, 4, 5, 6],
    horneado={7: 2048, 8: 1e-06},
    div16=[4, 5, 6],
)
# E1=8 lop3, E2=0 mul

_QPTX0_1 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk05_rmsnorm_quant_kernel // -- Begin function _sk05_rmsnorm_quant_kernel
.extern .shared .align 16 .b8 global_smem[];
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
                                        // @_sk05_rmsnorm_quant_kernel
.visible .entry _sk05_rmsnorm_quant_kernel(
	.param .u64 .ptr .global .align 1 _sk05_rmsnorm_quant_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk05_rmsnorm_quant_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk05_rmsnorm_quant_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk05_rmsnorm_quant_kernel_param_3,
	.param .u32 _sk05_rmsnorm_quant_kernel_param_4,
	.param .u32 _sk05_rmsnorm_quant_kernel_param_5,
	.param .u32 _sk05_rmsnorm_quant_kernel_param_6,
	.param .u64 .ptr .global .align 1 _sk05_rmsnorm_quant_kernel_param_7,
	.param .u64 .ptr .global .align 1 _sk05_rmsnorm_quant_kernel_param_8
)
.reqntid 256
{
	.reg .pred 	%p<23>;
	.reg .b16 	%rs<33>;
	.reg .b32 	%r<301>;
	.reg .b64 	%rd<16>;
	.loc	1 287 0                         // sk05_mlp_gateup.py:287:0
$L__func_begin0:
	.loc	1 287 0                         // sk05_mlp_gateup.py:287:0

// %bb.0:                               // %__nv_rsqrtf.exit
	ld.param.b64 	%rd7, [_sk05_rmsnorm_quant_kernel_param_0];
	ld.param.b64 	%rd8, [_sk05_rmsnorm_quant_kernel_param_1];
$L__tmp0:
	.loc	1 293 24                        // sk05_mlp_gateup.py:293:24
	mov.u32 	%r31, %ctaid.x;
	ld.param.b64 	%rd9, [_sk05_rmsnorm_quant_kernel_param_2];
	.loc	1 294 24                        // sk05_mlp_gateup.py:294:24
	mov.u32 	%r32, %tid.x;
	and.b32 	%r33, %r32, 255;
	ld.param.b64 	%rd10, [_sk05_rmsnorm_quant_kernel_param_3];
	and.b32 	%r34, %r32, 31;
	ld.param.b32 	%r35, [_sk05_rmsnorm_quant_kernel_param_4];
	shr.u32 	%r36, %r32, 5;
	ld.param.b32 	%r37, [_sk05_rmsnorm_quant_kernel_param_5];
	shl.b32 	%r38, %r32, 4;
	ld.param.b32 	%r39, [_sk05_rmsnorm_quant_kernel_param_6];
	and.b32 	%r40, %r38, 4080;
	.loc	1 295 18                        // sk05_mlp_gateup.py:295:18
	setp.lt.s32 	%p1, %r40, %r35;
	.loc	1 296 30                        // sk05_mlp_gateup.py:296:30
	mul.lo.s32 	%r41, %r37, %r31;
	.loc	1 296 24                        // sk05_mlp_gateup.py:296:24
	mad.wide.s32 	%rd11, %r41, 2, %rd7;
	.loc	1 296 42                        // sk05_mlp_gateup.py:296:42
	cvt.u64.u32 	%rd12, %r40;
	mul.wide.u32 	%rd13, %r40, 2;
	add.s64 	%rd1, %rd11, %rd13;
	add.s64 	%rd2, %rd1, 16;
	mov.b32 	%r5, 0;
	.loc	1 296 16                        // sk05_mlp_gateup.py:296:16
	// begin inline asm
	mov.u32 %r1, %r5;
	mov.u32 %r2, %r5;
	mov.u32 %r3, %r5;
	mov.u32 %r4, %r5;
	@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r6, %r5;
	mov.u32 %r7, %r5;
	mov.u32 %r8, %r5;
	mov.u32 %r9, %r5;
	@%p1 ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd2 + 0 ];
	// end inline asm
	.loc	1 296 73                        // sk05_mlp_gateup.py:296:73
	mov.b32 	{%rs1, %rs2}, %r2;
	cvt.f32.bf16 	%r42, %rs2;
	cvt.f32.bf16 	%r43, %rs1;
	mov.b32 	{%rs3, %rs4}, %r1;
	cvt.f32.bf16 	%r44, %rs3;
	cvt.f32.bf16 	%r45, %rs4;
	mov.b32 	{%rs5, %rs6}, %r4;
	cvt.f32.bf16 	%r46, %rs6;
	cvt.f32.bf16 	%r47, %rs5;
	mov.b32 	{%rs7, %rs8}, %r3;
	cvt.f32.bf16 	%r48, %rs8;
	cvt.f32.bf16 	%r49, %rs7;
	mov.b32 	{%rs9, %rs10}, %r7;
	cvt.f32.bf16 	%r50, %rs10;
	cvt.f32.bf16 	%r51, %rs9;
	mov.b32 	{%rs11, %rs12}, %r6;
	cvt.f32.bf16 	%r52, %rs12;
	cvt.f32.bf16 	%r53, %rs11;
	mov.b32 	{%rs13, %rs14}, %r9;
	cvt.f32.bf16 	%r54, %rs14;
	cvt.f32.bf16 	%r55, %rs13;
	mov.b32 	{%rs15, %rs16}, %r8;
	cvt.f32.bf16 	%r56, %rs16;
	cvt.f32.bf16 	%r57, %rs15;
	.loc	1 297 24                        // sk05_mlp_gateup.py:297:24
	add.s64 	%rd3, %rd8, %rd13;
	add.s64 	%rd4, %rd3, 16;
	.loc	1 297 16                        // sk05_mlp_gateup.py:297:16
	// begin inline asm
	mov.u32 %r10, %r5;
	mov.u32 %r11, %r5;
	mov.u32 %r12, %r5;
	mov.u32 %r13, %r5;
	@%p1 ld.global.v4.b32 { %r10, %r11, %r12, %r13 }, [ %rd3 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r14, %r5;
	mov.u32 %r15, %r5;
	mov.u32 %r16, %r5;
	mov.u32 %r17, %r5;
	@%p1 ld.global.v4.b32 { %r14, %r15, %r16, %r17 }, [ %rd4 + 0 ];
	// end inline asm
	.loc	1 298 32                        // sk05_mlp_gateup.py:298:32
	mul.f32 	%r58, %r45, %r45;
$L__tmp1:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	fma.rn.f32 	%r59, %r44, %r44, %r58;
	fma.rn.f32 	%r60, %r43, %r43, %r59;
	fma.rn.f32 	%r61, %r42, %r42, %r60;
	fma.rn.f32 	%r62, %r49, %r49, %r61;
	fma.rn.f32 	%r63, %r48, %r48, %r62;
	fma.rn.f32 	%r64, %r47, %r47, %r63;
	fma.rn.f32 	%r65, %r46, %r46, %r64;
	fma.rn.f32 	%r66, %r53, %r53, %r65;
	fma.rn.f32 	%r67, %r52, %r52, %r66;
	fma.rn.f32 	%r68, %r51, %r51, %r67;
	fma.rn.f32 	%r69, %r50, %r50, %r68;
	fma.rn.f32 	%r70, %r57, %r57, %r69;
	fma.rn.f32 	%r71, %r56, %r56, %r70;
	fma.rn.f32 	%r72, %r55, %r55, %r71;
	fma.rn.f32 	%r73, %r54, %r54, %r72;
$L__tmp2:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	shfl.sync.bfly.b32 	%r74, %r73, 16, 31, -1;
$L__tmp3:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	add.f32 	%r75, %r73, %r74;
$L__tmp4:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	shfl.sync.bfly.b32 	%r76, %r75, 8, 31, -1;
$L__tmp5:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	add.f32 	%r77, %r75, %r76;
$L__tmp6:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	shfl.sync.bfly.b32 	%r78, %r77, 4, 31, -1;
$L__tmp7:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	add.f32 	%r79, %r77, %r78;
$L__tmp8:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	shfl.sync.bfly.b32 	%r80, %r79, 2, 31, -1;
$L__tmp9:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	add.f32 	%r81, %r79, %r80;
$L__tmp10:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	shfl.sync.bfly.b32 	%r82, %r81, 1, 31, -1;
$L__tmp11:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	add.f32 	%r19, %r81, %r82;
$L__tmp12:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	setp.eq.b32 	%p2, %r34, 0;
	shr.u32 	%r83, %r32, 3;
	and.b32 	%r84, %r83, 28;
	mov.b32 	%r85, global_smem;
	add.s32 	%r18, %r85, %r84;
	// begin inline asm
	@%p2 st.shared.b32 [ %r18 + 0 ], %r19;
	// end inline asm
	bar.sync 	0;
	setp.lt.u32 	%p3, %r33, 8;
	shl.b32 	%r86, %r33, 2;
	add.s32 	%r21, %r85, %r86;
	// begin inline asm
	@%p3 ld.shared.b32 %r20, [ %r21 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r87, %r20, 4, 31, -1;
$L__tmp13:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	add.f32 	%r88, %r20, %r87;
$L__tmp14:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	shfl.sync.bfly.b32 	%r89, %r88, 2, 31, -1;
$L__tmp15:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	add.f32 	%r90, %r88, %r89;
$L__tmp16:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	shfl.sync.bfly.b32 	%r91, %r90, 1, 31, -1;
$L__tmp17:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	add.f32 	%r22, %r90, %r91;
$L__tmp18:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	and.b32 	%r92, %r32, 7;
	setp.eq.b32 	%p6, %r92, 0;
	and.pred 	%p4, %p3, %p6;
	// begin inline asm
	@%p4 st.shared.b32 [ %r21 + 0 ], %r22;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r93, [global_smem];
$L__tmp19:
	.loc	1 298 45                        // sk05_mlp_gateup.py:298:45
	cvt.rn.f32.s32 	%r94, %r35;
	div.full.f32 	%r95, %r93, %r94;
	.loc	1 298 49                        // sk05_mlp_gateup.py:298:49
	add.f32 	%r96, %r95, 0f358637BD;
	.loc	1 298 21                        // sk05_mlp_gateup.py:298:21
	rsqrt.approx.ftz.f32 	%r97, %r96;
	.loc	1 298 12                        // sk05_mlp_gateup.py:298:12
	mul.f32 	%r98, %r97, %r44;
	mul.f32 	%r99, %r97, %r45;
	mul.f32 	%r100, %r97, %r43;
	mul.f32 	%r101, %r97, %r42;
	mul.f32 	%r102, %r97, %r49;
	mul.f32 	%r103, %r97, %r48;
	mul.f32 	%r104, %r97, %r47;
	mul.f32 	%r105, %r97, %r46;
	mul.f32 	%r106, %r97, %r53;
	mul.f32 	%r107, %r97, %r52;
	mul.f32 	%r108, %r97, %r51;
	mul.f32 	%r109, %r97, %r50;
	mul.f32 	%r110, %r97, %r57;
	mul.f32 	%r111, %r97, %r56;
	mul.f32 	%r112, %r97, %r55;
	mul.f32 	%r113, %r97, %r54;
$L__tmp20:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	bar.sync 	0;
$L__tmp21:
	.loc	1 302 27                        // sk05_mlp_gateup.py:302:27
	mul.lo.s32 	%r114, %r39, %r31;
	.loc	1 302 21                        // sk05_mlp_gateup.py:302:21
	cvt.s64.s32 	%rd14, %r114;
	add.s64 	%rd15, %rd9, %rd14;
	.loc	1 302 39                        // sk05_mlp_gateup.py:302:39
	add.s64 	%rd5, %rd15, %rd12;
	.loc	1 297 55                        // sk05_mlp_gateup.py:297:55
	mov.b32 	{%rs17, %rs18}, %r16;
	cvt.f32.bf16 	%r115, %rs17;
	cvt.f32.bf16 	%r116, %rs18;
	mov.b32 	{%rs19, %rs20}, %r17;
	cvt.f32.bf16 	%r117, %rs19;
	cvt.f32.bf16 	%r118, %rs20;
	.loc	1 298 56                        // sk05_mlp_gateup.py:298:56
	mul.f32 	%r119, %r113, %r118;
	mul.f32 	%r120, %r112, %r117;
	mul.f32 	%r121, %r111, %r116;
	mul.f32 	%r122, %r110, %r115;
	.loc	1 299 36                        // sk05_mlp_gateup.py:299:36
	abs.f32 	%r123, %r122;
	abs.f32 	%r124, %r121;
	abs.f32 	%r125, %r120;
	abs.f32 	%r126, %r119;
	.loc	1 297 55                        // sk05_mlp_gateup.py:297:55
	mov.b32 	{%rs21, %rs22}, %r14;
	cvt.f32.bf16 	%r127, %rs21;
	cvt.f32.bf16 	%r128, %rs22;
	mov.b32 	{%rs23, %rs24}, %r15;
	cvt.f32.bf16 	%r129, %rs23;
	cvt.f32.bf16 	%r130, %rs24;
	.loc	1 298 56                        // sk05_mlp_gateup.py:298:56
	mul.f32 	%r131, %r109, %r130;
	mul.f32 	%r132, %r108, %r129;
	mul.f32 	%r133, %r107, %r128;
	mul.f32 	%r134, %r106, %r127;
	.loc	1 299 36                        // sk05_mlp_gateup.py:299:36
	abs.f32 	%r135, %r134;
	abs.f32 	%r136, %r133;
	abs.f32 	%r137, %r132;
	abs.f32 	%r138, %r131;
	.loc	1 297 55                        // sk05_mlp_gateup.py:297:55
	mov.b32 	{%rs25, %rs26}, %r12;
	cvt.f32.bf16 	%r139, %rs25;
	cvt.f32.bf16 	%r140, %rs26;
	mov.b32 	{%rs27, %rs28}, %r13;
	cvt.f32.bf16 	%r141, %rs27;
	cvt.f32.bf16 	%r142, %rs28;
	.loc	1 298 56                        // sk05_mlp_gateup.py:298:56
	mul.f32 	%r143, %r105, %r142;
	mul.f32 	%r144, %r104, %r141;
	mul.f32 	%r145, %r103, %r140;
	mul.f32 	%r146, %r102, %r139;
	.loc	1 299 36                        // sk05_mlp_gateup.py:299:36
	abs.f32 	%r147, %r146;
	abs.f32 	%r148, %r145;
	abs.f32 	%r149, %r144;
	abs.f32 	%r150, %r143;
	.loc	1 297 55                        // sk05_mlp_gateup.py:297:55
	mov.b32 	{%rs29, %rs30}, %r10;
	cvt.f32.bf16 	%r151, %rs29;
	cvt.f32.bf16 	%r152, %rs30;
	mov.b32 	{%rs31, %rs32}, %r11;
	cvt.f32.bf16 	%r153, %rs31;
	cvt.f32.bf16 	%r154, %rs32;
	.loc	1 298 56                        // sk05_mlp_gateup.py:298:56
	mul.f32 	%r155, %r101, %r154;
	mul.f32 	%r156, %r100, %r153;
	mul.f32 	%r157, %r99, %r152;
	mul.f32 	%r158, %r98, %r151;
	.loc	1 299 36                        // sk05_mlp_gateup.py:299:36
	abs.f32 	%r159, %r158;
	abs.f32 	%r160, %r157;
	abs.f32 	%r161, %r156;
	abs.f32 	%r162, %r155;
$L__tmp22:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r163, %r159, %r160;
	max.f32 	%r164, %r163, %r161;
	max.f32 	%r165, %r164, %r162;
	max.f32 	%r166, %r165, %r147;
	max.f32 	%r167, %r166, %r148;
	max.f32 	%r168, %r167, %r149;
	max.f32 	%r169, %r168, %r150;
	max.f32 	%r170, %r169, %r135;
	max.f32 	%r171, %r170, %r136;
	max.f32 	%r172, %r171, %r137;
	max.f32 	%r173, %r172, %r138;
	max.f32 	%r174, %r173, %r123;
	max.f32 	%r175, %r174, %r124;
	max.f32 	%r176, %r175, %r125;
	max.f32 	%r177, %r176, %r126;
$L__tmp23:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	shfl.sync.bfly.b32 	%r178, %r177, 16, 31, -1;
$L__tmp24:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r179, %r177, %r178;
$L__tmp25:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	shfl.sync.bfly.b32 	%r180, %r179, 8, 31, -1;
$L__tmp26:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r181, %r179, %r180;
$L__tmp27:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	shfl.sync.bfly.b32 	%r182, %r181, 4, 31, -1;
$L__tmp28:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r183, %r181, %r182;
$L__tmp29:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	shfl.sync.bfly.b32 	%r184, %r183, 2, 31, -1;
$L__tmp30:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r185, %r183, %r184;
$L__tmp31:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	shfl.sync.bfly.b32 	%r186, %r185, 1, 31, -1;
$L__tmp32:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r23, %r185, %r186;
$L__tmp33:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	// begin inline asm
	@%p2 st.shared.b32 [ %r18 + 0 ], %r23;
	// end inline asm
	bar.sync 	0;
	// begin inline asm
	@%p3 ld.shared.b32 %r24, [ %r21 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r187, %r24, 4, 31, -1;
$L__tmp34:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r188, %r24, %r187;
$L__tmp35:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	shfl.sync.bfly.b32 	%r189, %r188, 2, 31, -1;
$L__tmp36:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r190, %r188, %r189;
$L__tmp37:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	shfl.sync.bfly.b32 	%r191, %r190, 1, 31, -1;
$L__tmp38:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r25, %r190, %r191;
$L__tmp39:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	// begin inline asm
	@%p4 st.shared.b32 [ %r21 + 0 ], %r25;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r192, [global_smem];
$L__tmp40:
	.loc	1 299 49                        // sk05_mlp_gateup.py:299:49
	max.f32 	%r193, %r192, 0f0DA24260;
	mov.b32 	%r194, 0f42FE0000;
	.loc	1 300 22                        // sk05_mlp_gateup.py:300:22
	div.full.f32 	%r195, %r194, %r193;
	.loc	1 300 14                        // sk05_mlp_gateup.py:300:14
	mul.f32 	%r196, %r157, %r195;
	mul.f32 	%r197, %r158, %r195;
	mul.f32 	%r198, %r155, %r195;
	mul.f32 	%r199, %r156, %r195;
	mul.f32 	%r200, %r145, %r195;
	mul.f32 	%r201, %r146, %r195;
	mul.f32 	%r202, %r143, %r195;
	mul.f32 	%r203, %r144, %r195;
	mul.f32 	%r204, %r133, %r195;
	mul.f32 	%r205, %r134, %r195;
	mul.f32 	%r206, %r131, %r195;
	mul.f32 	%r207, %r132, %r195;
	mul.f32 	%r208, %r121, %r195;
	mul.f32 	%r209, %r122, %r195;
	mul.f32 	%r210, %r119, %r195;
	mul.f32 	%r211, %r120, %r195;
	.loc	1 301 29                        // sk05_mlp_gateup.py:301:29
	.loc	1 301 39                        // sk05_mlp_gateup.py:301:39
	lop3.b32 	%r212, 0x3f000000, %r196, 0x80000000, 0xF8;
	lop3.b32 	%r213, 0x3f000000, %r197, 0x80000000, 0xF8;
	lop3.b32 	%r214, 0x3f000000, %r198, 0x80000000, 0xF8;
	lop3.b32 	%r215, 0x3f000000, %r199, 0x80000000, 0xF8;
	lop3.b32 	%r216, 0x3f000000, %r200, 0x80000000, 0xF8;
	lop3.b32 	%r217, 0x3f000000, %r201, 0x80000000, 0xF8;
	lop3.b32 	%r218, 0x3f000000, %r202, 0x80000000, 0xF8;
	lop3.b32 	%r219, 0x3f000000, %r203, 0x80000000, 0xF8;
	lop3.b32 	%r220, 0x3f000000, %r204, 0x80000000, 0xF8;
	lop3.b32 	%r221, 0x3f000000, %r205, 0x80000000, 0xF8;
	lop3.b32 	%r222, 0x3f000000, %r206, 0x80000000, 0xF8;
	lop3.b32 	%r223, 0x3f000000, %r207, 0x80000000, 0xF8;
	lop3.b32 	%r224, 0x3f000000, %r208, 0x80000000, 0xF8;
	lop3.b32 	%r225, 0x3f000000, %r209, 0x80000000, 0xF8;
	lop3.b32 	%r226, 0x3f000000, %r210, 0x80000000, 0xF8;
	lop3.b32 	%r227, 0x3f000000, %r211, 0x80000000, 0xF8;
	.loc	1 301 14                        // sk05_mlp_gateup.py:301:14
	fma.rn.f32 	%r228, %r156, %r195, %r215;
	fma.rn.f32 	%r229, %r155, %r195, %r214;
	fma.rn.f32 	%r230, %r158, %r195, %r213;
	fma.rn.f32 	%r231, %r157, %r195, %r212;
	fma.rn.f32 	%r232, %r144, %r195, %r219;
	fma.rn.f32 	%r233, %r143, %r195, %r218;
	fma.rn.f32 	%r234, %r146, %r195, %r217;
	fma.rn.f32 	%r235, %r145, %r195, %r216;
	fma.rn.f32 	%r236, %r132, %r195, %r223;
	fma.rn.f32 	%r237, %r131, %r195, %r222;
	fma.rn.f32 	%r238, %r134, %r195, %r221;
	fma.rn.f32 	%r239, %r133, %r195, %r220;
	fma.rn.f32 	%r240, %r120, %r195, %r227;
	fma.rn.f32 	%r241, %r119, %r195, %r226;
	fma.rn.f32 	%r242, %r122, %r195, %r225;
	fma.rn.f32 	%r243, %r121, %r195, %r224;
	.loc	1 301 49                        // sk05_mlp_gateup.py:301:49
	cvt.rzi.s32.f32 	%r244, %r231;
	cvt.rzi.s32.f32 	%r245, %r230;
	cvt.rzi.s32.f32 	%r246, %r229;
	cvt.rzi.s32.f32 	%r247, %r228;
	cvt.rzi.s32.f32 	%r248, %r235;
	cvt.rzi.s32.f32 	%r249, %r234;
	cvt.rzi.s32.f32 	%r250, %r233;
	cvt.rzi.s32.f32 	%r251, %r232;
	cvt.rzi.s32.f32 	%r252, %r239;
	cvt.rzi.s32.f32 	%r253, %r238;
	cvt.rzi.s32.f32 	%r254, %r237;
	cvt.rzi.s32.f32 	%r255, %r236;
	cvt.rzi.s32.f32 	%r256, %r243;
	cvt.rzi.s32.f32 	%r257, %r242;
	cvt.rzi.s32.f32 	%r258, %r241;
	cvt.rzi.s32.f32 	%r259, %r240;
	.loc	1 302 70                        // sk05_mlp_gateup.py:302:70
	max.s32 	%r260, %r247, -127;
	max.s32 	%r261, %r246, -127;
	max.s32 	%r262, %r245, -127;
	max.s32 	%r263, %r244, -127;
	max.s32 	%r264, %r251, -127;
	max.s32 	%r265, %r250, -127;
	max.s32 	%r266, %r249, -127;
	max.s32 	%r267, %r248, -127;
	max.s32 	%r268, %r255, -127;
	max.s32 	%r269, %r254, -127;
	max.s32 	%r270, %r253, -127;
	max.s32 	%r271, %r252, -127;
	max.s32 	%r272, %r259, -127;
	max.s32 	%r273, %r258, -127;
	max.s32 	%r274, %r257, -127;
	max.s32 	%r275, %r256, -127;
	.loc	1 302 77                        // sk05_mlp_gateup.py:302:77
	min.s32 	%r276, %r263, 127;
	min.s32 	%r277, %r262, 127;
	min.s32 	%r278, %r261, 127;
	min.s32 	%r279, %r260, 127;
	min.s32 	%r280, %r267, 127;
	min.s32 	%r281, %r266, 127;
	min.s32 	%r282, %r265, 127;
	min.s32 	%r283, %r264, 127;
	min.s32 	%r284, %r271, 127;
	min.s32 	%r285, %r270, 127;
	min.s32 	%r286, %r269, 127;
	min.s32 	%r287, %r268, 127;
	min.s32 	%r288, %r275, 127;
	min.s32 	%r289, %r274, 127;
	min.s32 	%r290, %r273, 127;
	min.s32 	%r291, %r272, 127;
	.loc	1 302 85                        // sk05_mlp_gateup.py:302:85
	prmt.b32 	%r292, %r279, %r278, 0x3340U;
	prmt.b32 	%r293, %r277, %r276, 0x3340U;
	prmt.b32 	%r26, %r293, %r292, 0x5410U;
	prmt.b32 	%r294, %r283, %r282, 0x3340U;
	prmt.b32 	%r295, %r281, %r280, 0x3340U;
	prmt.b32 	%r27, %r295, %r294, 0x5410U;
	prmt.b32 	%r296, %r287, %r286, 0x3340U;
	prmt.b32 	%r297, %r285, %r284, 0x3340U;
	prmt.b32 	%r28, %r297, %r296, 0x5410U;
	prmt.b32 	%r298, %r291, %r290, 0x3340U;
	prmt.b32 	%r299, %r289, %r288, 0x3340U;
	prmt.b32 	%r29, %r299, %r298, 0x5410U;
	.loc	1 302 45                        // sk05_mlp_gateup.py:302:45
	// begin inline asm
	@%p1 st.global.v4.b32 [ %rd5 + 0 ], { %r26, %r27, %r28, %r29 };
	// end inline asm
	.loc	1 303 21                        // sk05_mlp_gateup.py:303:21
	mad.wide.u32 	%rd6, %r31, 4, %rd10;
	.loc	1 303 34                        // sk05_mlp_gateup.py:303:34
	mul.f32 	%r30, %r193, 0f3C010204;
	.loc	1 303 26                        // sk05_mlp_gateup.py:303:26
	or.b32 	%r300, %r34, %r36;
	setp.eq.b32 	%p5, %r300, 0;
	// begin inline asm
	@%p5 st.global.b32 [ %rd6 + 0 ], { %r30 };
	// end inline asm
	.loc	1 303 4                         // sk05_mlp_gateup.py:303:4
	ret;
$L__tmp41:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/usr/local/lib/python3.12/dist-packages/vllm/_genesis/kernels/sk05_mlp_gateup.py"
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
.b8 1                                   // DW_CHILDREN_yes
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
.b8 5                                   // Abbreviation Code
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
.b8 6                                   // Abbreviation Code
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
.b8 11                                  // DW_FORM_data1
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 255                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0xf8 DW_TAG_compile_unit
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
.b8 53
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 103
.b8 97
.b8 116
.b8 101
.b8 117
.b8 112
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 117
.b8 115
.b8 114
.b8 47
.b8 108
.b8 111
.b8 99
.b8 97
.b8 108
.b8 47
.b8 108
.b8 105
.b8 98
.b8 47
.b8 112
.b8 121
.b8 116
.b8 104
.b8 111
.b8 110
.b8 51
.b8 46
.b8 49
.b8 50
.b8 47
.b8 100
.b8 105
.b8 115
.b8 116
.b8 45
.b8 112
.b8 97
.b8 99
.b8 107
.b8 97
.b8 103
.b8 101
.b8 115
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
.b8 2                                   // Abbrev [2] 0x6a:0x1d DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 53
.b8 95
.b8 114
.b8 109
.b8 115
.b8 110
.b8 111
.b8 114
.b8 109
.b8 95
.b8 113
.b8 117
.b8 97
.b8 110
.b8 116
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x87:0x7b DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 106                                // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x9c:0x33 DW_TAG_inlined_subroutine
.b32 106                                // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp19                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 42                                  // DW_AT_call_line
.b8 1
.b8 28                                  // DW_AT_call_column
.b8 5                                   // Abbrev [5] 0xb5:0x19 DW_TAG_inlined_subroutine
.b32 106                                // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp18                          // DW_AT_high_pc
.b8 2                                   // DW_AT_call_file
.b8 37                                  // DW_AT_call_line
.b8 1
.b8 36                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 4                                   // Abbrev [4] 0xcf:0x32 DW_TAG_inlined_subroutine
.b32 106                                // DW_AT_abstract_origin
.b64 $L__tmp20                          // DW_AT_low_pc
.b64 $L__tmp40                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 43                                  // DW_AT_call_line
.b8 1
.b8 29                                  // DW_AT_call_column
.b8 6                                   // Abbrev [6] 0xe8:0x18 DW_TAG_inlined_subroutine
.b32 106                                // DW_AT_abstract_origin
.b64 $L__tmp22                          // DW_AT_low_pc
.b64 $L__tmp39                          // DW_AT_high_pc
.b8 2                                   // DW_AT_call_file
.b8 191                                 // DW_AT_call_line
.b8 40                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_QVAR0_1 = _Nativo(
    "sk05_mlp_gateup/_sk05_rmsnorm_quant_kernel/blk4096",
    _QPTX0_1, "_sk05_rmsnorm_quant_kernel",
    warps=8, shared=32,
    abi=[0, 1, 2, 3, 4, 5, 6],
    horneado={7: 4096, 8: 1e-06},
    div16=[4, 5, 6],
)
# E1=16 lop3, E2=0 mul

_QPTX0_2 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk05_rmsnorm_quant_kernel // -- Begin function _sk05_rmsnorm_quant_kernel
.extern .shared .align 16 .b8 global_smem[];
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
                                        // @_sk05_rmsnorm_quant_kernel
.visible .entry _sk05_rmsnorm_quant_kernel(
	.param .u64 .ptr .global .align 1 _sk05_rmsnorm_quant_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk05_rmsnorm_quant_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk05_rmsnorm_quant_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk05_rmsnorm_quant_kernel_param_3,
	.param .u32 _sk05_rmsnorm_quant_kernel_param_4,
	.param .u32 _sk05_rmsnorm_quant_kernel_param_5,
	.param .u32 _sk05_rmsnorm_quant_kernel_param_6,
	.param .u64 .ptr .global .align 1 _sk05_rmsnorm_quant_kernel_param_7,
	.param .u64 .ptr .global .align 1 _sk05_rmsnorm_quant_kernel_param_8
)
.reqntid 256
{
	.reg .pred 	%p<40>;
	.reg .b16 	%rs<65>;
	.reg .b32 	%r<538>;
	.reg .b64 	%rd<21>;
	.loc	1 287 0                         // sk05_mlp_gateup.py:287:0
$L__func_begin0:
	.loc	1 287 0                         // sk05_mlp_gateup.py:287:0

// %bb.0:                               // %__nv_rsqrtf.exit
	ld.param.b64 	%rd12, [_sk05_rmsnorm_quant_kernel_param_0];
	ld.param.b64 	%rd13, [_sk05_rmsnorm_quant_kernel_param_1];
$L__tmp0:
	.loc	1 293 24                        // sk05_mlp_gateup.py:293:24
	mov.u32 	%r51, %ctaid.x;
	ld.param.b64 	%rd14, [_sk05_rmsnorm_quant_kernel_param_2];
	.loc	1 294 24                        // sk05_mlp_gateup.py:294:24
	mov.u32 	%r52, %tid.x;
	and.b32 	%r53, %r52, 255;
	ld.param.b64 	%rd15, [_sk05_rmsnorm_quant_kernel_param_3];
	and.b32 	%r54, %r52, 31;
	ld.param.b32 	%r55, [_sk05_rmsnorm_quant_kernel_param_4];
	shr.u32 	%r56, %r52, 5;
	ld.param.b32 	%r57, [_sk05_rmsnorm_quant_kernel_param_5];
	shl.b32 	%r58, %r52, 4;
	ld.param.b32 	%r59, [_sk05_rmsnorm_quant_kernel_param_6];
	and.b32 	%r60, %r58, 4080;
	or.b32 	%r61, %r60, 4096;
	.loc	1 295 18                        // sk05_mlp_gateup.py:295:18
	setp.lt.s32 	%p1, %r60, %r55;
	setp.lt.s32 	%p2, %r61, %r55;
	.loc	1 296 30                        // sk05_mlp_gateup.py:296:30
	mul.lo.s32 	%r62, %r57, %r51;
	.loc	1 296 24                        // sk05_mlp_gateup.py:296:24
	mad.wide.s32 	%rd16, %r62, 2, %rd12;
	.loc	1 296 42                        // sk05_mlp_gateup.py:296:42
	cvt.u64.u32 	%rd17, %r60;
	mul.wide.u32 	%rd18, %r60, 2;
	add.s64 	%rd1, %rd16, %rd18;
	add.s64 	%rd2, %rd1, 16;
	add.s64 	%rd3, %rd1, 8192;
	add.s64 	%rd4, %rd1, 8208;
	mov.b32 	%r5, 0;
	.loc	1 296 16                        // sk05_mlp_gateup.py:296:16
	// begin inline asm
	mov.u32 %r1, %r5;
	mov.u32 %r2, %r5;
	mov.u32 %r3, %r5;
	mov.u32 %r4, %r5;
	@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r6, %r5;
	mov.u32 %r7, %r5;
	mov.u32 %r8, %r5;
	mov.u32 %r9, %r5;
	@%p1 ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd2 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r10, %r5;
	mov.u32 %r11, %r5;
	mov.u32 %r12, %r5;
	mov.u32 %r13, %r5;
	@%p2 ld.global.v4.b32 { %r10, %r11, %r12, %r13 }, [ %rd3 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r14, %r5;
	mov.u32 %r15, %r5;
	mov.u32 %r16, %r5;
	mov.u32 %r17, %r5;
	@%p2 ld.global.v4.b32 { %r14, %r15, %r16, %r17 }, [ %rd4 + 0 ];
	// end inline asm
	.loc	1 296 73                        // sk05_mlp_gateup.py:296:73
	mov.b32 	{%rs1, %rs2}, %r2;
	cvt.f32.bf16 	%r63, %rs2;
	cvt.f32.bf16 	%r64, %rs1;
	mov.b32 	{%rs3, %rs4}, %r1;
	cvt.f32.bf16 	%r65, %rs3;
	cvt.f32.bf16 	%r66, %rs4;
	mov.b32 	{%rs5, %rs6}, %r4;
	cvt.f32.bf16 	%r67, %rs6;
	cvt.f32.bf16 	%r68, %rs5;
	mov.b32 	{%rs7, %rs8}, %r3;
	cvt.f32.bf16 	%r69, %rs8;
	cvt.f32.bf16 	%r70, %rs7;
	mov.b32 	{%rs9, %rs10}, %r7;
	cvt.f32.bf16 	%r71, %rs10;
	cvt.f32.bf16 	%r72, %rs9;
	mov.b32 	{%rs11, %rs12}, %r6;
	cvt.f32.bf16 	%r73, %rs12;
	cvt.f32.bf16 	%r74, %rs11;
	mov.b32 	{%rs13, %rs14}, %r9;
	cvt.f32.bf16 	%r75, %rs14;
	cvt.f32.bf16 	%r76, %rs13;
	mov.b32 	{%rs15, %rs16}, %r8;
	cvt.f32.bf16 	%r77, %rs16;
	cvt.f32.bf16 	%r78, %rs15;
	mov.b32 	{%rs17, %rs18}, %r11;
	cvt.f32.bf16 	%r79, %rs18;
	cvt.f32.bf16 	%r80, %rs17;
	mov.b32 	{%rs19, %rs20}, %r10;
	cvt.f32.bf16 	%r81, %rs20;
	cvt.f32.bf16 	%r82, %rs19;
	mov.b32 	{%rs21, %rs22}, %r13;
	cvt.f32.bf16 	%r83, %rs22;
	cvt.f32.bf16 	%r84, %rs21;
	mov.b32 	{%rs23, %rs24}, %r12;
	cvt.f32.bf16 	%r85, %rs24;
	cvt.f32.bf16 	%r86, %rs23;
	mov.b32 	{%rs25, %rs26}, %r15;
	cvt.f32.bf16 	%r87, %rs26;
	cvt.f32.bf16 	%r88, %rs25;
	mov.b32 	{%rs27, %rs28}, %r14;
	cvt.f32.bf16 	%r89, %rs28;
	cvt.f32.bf16 	%r90, %rs27;
	mov.b32 	{%rs29, %rs30}, %r17;
	cvt.f32.bf16 	%r91, %rs30;
	cvt.f32.bf16 	%r92, %rs29;
	mov.b32 	{%rs31, %rs32}, %r16;
	cvt.f32.bf16 	%r93, %rs32;
	cvt.f32.bf16 	%r94, %rs31;
	.loc	1 297 24                        // sk05_mlp_gateup.py:297:24
	add.s64 	%rd5, %rd13, %rd18;
	add.s64 	%rd6, %rd5, 16;
	add.s64 	%rd7, %rd5, 8192;
	add.s64 	%rd8, %rd5, 8208;
	.loc	1 297 16                        // sk05_mlp_gateup.py:297:16
	// begin inline asm
	mov.u32 %r18, %r5;
	mov.u32 %r19, %r5;
	mov.u32 %r20, %r5;
	mov.u32 %r21, %r5;
	@%p1 ld.global.v4.b32 { %r18, %r19, %r20, %r21 }, [ %rd5 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r22, %r5;
	mov.u32 %r23, %r5;
	mov.u32 %r24, %r5;
	mov.u32 %r25, %r5;
	@%p1 ld.global.v4.b32 { %r22, %r23, %r24, %r25 }, [ %rd6 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r26, %r5;
	mov.u32 %r27, %r5;
	mov.u32 %r28, %r5;
	mov.u32 %r29, %r5;
	@%p2 ld.global.v4.b32 { %r26, %r27, %r28, %r29 }, [ %rd7 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r30, %r5;
	mov.u32 %r31, %r5;
	mov.u32 %r32, %r5;
	mov.u32 %r33, %r5;
	@%p2 ld.global.v4.b32 { %r30, %r31, %r32, %r33 }, [ %rd8 + 0 ];
	// end inline asm
	.loc	1 298 32                        // sk05_mlp_gateup.py:298:32
	mul.f32 	%r95, %r66, %r66;
$L__tmp1:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	fma.rn.f32 	%r96, %r65, %r65, %r95;
	fma.rn.f32 	%r97, %r64, %r64, %r96;
	fma.rn.f32 	%r98, %r63, %r63, %r97;
	fma.rn.f32 	%r99, %r70, %r70, %r98;
	fma.rn.f32 	%r100, %r69, %r69, %r99;
	fma.rn.f32 	%r101, %r68, %r68, %r100;
	fma.rn.f32 	%r102, %r67, %r67, %r101;
	fma.rn.f32 	%r103, %r74, %r74, %r102;
	fma.rn.f32 	%r104, %r73, %r73, %r103;
	fma.rn.f32 	%r105, %r72, %r72, %r104;
	fma.rn.f32 	%r106, %r71, %r71, %r105;
	fma.rn.f32 	%r107, %r78, %r78, %r106;
	fma.rn.f32 	%r108, %r77, %r77, %r107;
	fma.rn.f32 	%r109, %r76, %r76, %r108;
	fma.rn.f32 	%r110, %r75, %r75, %r109;
	fma.rn.f32 	%r111, %r82, %r82, %r110;
	fma.rn.f32 	%r112, %r81, %r81, %r111;
	fma.rn.f32 	%r113, %r80, %r80, %r112;
	fma.rn.f32 	%r114, %r79, %r79, %r113;
	fma.rn.f32 	%r115, %r86, %r86, %r114;
	fma.rn.f32 	%r116, %r85, %r85, %r115;
	fma.rn.f32 	%r117, %r84, %r84, %r116;
	fma.rn.f32 	%r118, %r83, %r83, %r117;
	fma.rn.f32 	%r119, %r90, %r90, %r118;
	fma.rn.f32 	%r120, %r89, %r89, %r119;
	fma.rn.f32 	%r121, %r88, %r88, %r120;
	fma.rn.f32 	%r122, %r87, %r87, %r121;
	fma.rn.f32 	%r123, %r94, %r94, %r122;
	fma.rn.f32 	%r124, %r93, %r93, %r123;
	fma.rn.f32 	%r125, %r92, %r92, %r124;
	fma.rn.f32 	%r126, %r91, %r91, %r125;
$L__tmp2:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	shfl.sync.bfly.b32 	%r127, %r126, 16, 31, -1;
$L__tmp3:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	add.f32 	%r128, %r126, %r127;
$L__tmp4:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	shfl.sync.bfly.b32 	%r129, %r128, 8, 31, -1;
$L__tmp5:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	add.f32 	%r130, %r128, %r129;
$L__tmp6:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	shfl.sync.bfly.b32 	%r131, %r130, 4, 31, -1;
$L__tmp7:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	add.f32 	%r132, %r130, %r131;
$L__tmp8:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	shfl.sync.bfly.b32 	%r133, %r132, 2, 31, -1;
$L__tmp9:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	add.f32 	%r134, %r132, %r133;
$L__tmp10:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	shfl.sync.bfly.b32 	%r135, %r134, 1, 31, -1;
$L__tmp11:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	add.f32 	%r35, %r134, %r135;
$L__tmp12:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	setp.eq.b32 	%p3, %r54, 0;
	shr.u32 	%r136, %r52, 3;
	and.b32 	%r137, %r136, 28;
	mov.b32 	%r138, global_smem;
	add.s32 	%r34, %r138, %r137;
	// begin inline asm
	@%p3 st.shared.b32 [ %r34 + 0 ], %r35;
	// end inline asm
	bar.sync 	0;
	setp.lt.u32 	%p4, %r53, 8;
	shl.b32 	%r139, %r53, 2;
	add.s32 	%r37, %r138, %r139;
	// begin inline asm
	@%p4 ld.shared.b32 %r36, [ %r37 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r140, %r36, 4, 31, -1;
$L__tmp13:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	add.f32 	%r141, %r36, %r140;
$L__tmp14:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	shfl.sync.bfly.b32 	%r142, %r141, 2, 31, -1;
$L__tmp15:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	add.f32 	%r143, %r141, %r142;
$L__tmp16:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	shfl.sync.bfly.b32 	%r144, %r143, 1, 31, -1;
$L__tmp17:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	add.f32 	%r38, %r143, %r144;
$L__tmp18:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	and.b32 	%r145, %r52, 7;
	setp.eq.b32 	%p7, %r145, 0;
	and.pred 	%p5, %p4, %p7;
	// begin inline asm
	@%p5 st.shared.b32 [ %r37 + 0 ], %r38;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r146, [global_smem];
$L__tmp19:
	.loc	1 298 45                        // sk05_mlp_gateup.py:298:45
	cvt.rn.f32.s32 	%r147, %r55;
	div.full.f32 	%r148, %r146, %r147;
	.loc	1 298 49                        // sk05_mlp_gateup.py:298:49
	add.f32 	%r149, %r148, 0f358637BD;
	.loc	1 298 21                        // sk05_mlp_gateup.py:298:21
	rsqrt.approx.ftz.f32 	%r150, %r149;
	.loc	1 298 12                        // sk05_mlp_gateup.py:298:12
	mul.f32 	%r151, %r150, %r65;
	mul.f32 	%r152, %r150, %r66;
	mul.f32 	%r153, %r150, %r64;
	mul.f32 	%r154, %r150, %r63;
	mul.f32 	%r155, %r150, %r70;
	mul.f32 	%r156, %r150, %r69;
	mul.f32 	%r157, %r150, %r68;
	mul.f32 	%r158, %r150, %r67;
	mul.f32 	%r159, %r150, %r74;
	mul.f32 	%r160, %r150, %r73;
	mul.f32 	%r161, %r150, %r72;
	mul.f32 	%r162, %r150, %r71;
	mul.f32 	%r163, %r150, %r78;
	mul.f32 	%r164, %r150, %r77;
	mul.f32 	%r165, %r150, %r76;
	mul.f32 	%r166, %r150, %r75;
	mul.f32 	%r167, %r150, %r82;
	mul.f32 	%r168, %r150, %r81;
	mul.f32 	%r169, %r150, %r80;
	mul.f32 	%r170, %r150, %r79;
	mul.f32 	%r171, %r150, %r86;
	mul.f32 	%r172, %r150, %r85;
	mul.f32 	%r173, %r150, %r84;
	mul.f32 	%r174, %r150, %r83;
	mul.f32 	%r175, %r150, %r90;
	mul.f32 	%r176, %r150, %r89;
	mul.f32 	%r177, %r150, %r88;
	mul.f32 	%r178, %r150, %r87;
	mul.f32 	%r179, %r150, %r94;
	mul.f32 	%r180, %r150, %r93;
	mul.f32 	%r181, %r150, %r92;
	mul.f32 	%r182, %r150, %r91;
$L__tmp20:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	bar.sync 	0;
$L__tmp21:
	.loc	1 302 27                        // sk05_mlp_gateup.py:302:27
	mul.lo.s32 	%r183, %r59, %r51;
	.loc	1 302 21                        // sk05_mlp_gateup.py:302:21
	cvt.s64.s32 	%rd19, %r183;
	add.s64 	%rd20, %rd14, %rd19;
	.loc	1 302 39                        // sk05_mlp_gateup.py:302:39
	add.s64 	%rd9, %rd20, %rd17;
	add.s64 	%rd10, %rd9, 4096;
	.loc	1 297 55                        // sk05_mlp_gateup.py:297:55
	mov.b32 	{%rs33, %rs34}, %r24;
	cvt.f32.bf16 	%r184, %rs33;
	cvt.f32.bf16 	%r185, %rs34;
	mov.b32 	{%rs35, %rs36}, %r25;
	cvt.f32.bf16 	%r186, %rs35;
	cvt.f32.bf16 	%r187, %rs36;
	.loc	1 298 56                        // sk05_mlp_gateup.py:298:56
	mul.f32 	%r188, %r166, %r187;
	mul.f32 	%r189, %r165, %r186;
	mul.f32 	%r190, %r164, %r185;
	mul.f32 	%r191, %r163, %r184;
	.loc	1 299 36                        // sk05_mlp_gateup.py:299:36
	abs.f32 	%r192, %r191;
	abs.f32 	%r193, %r190;
	abs.f32 	%r194, %r189;
	abs.f32 	%r195, %r188;
	.loc	1 297 55                        // sk05_mlp_gateup.py:297:55
	mov.b32 	{%rs37, %rs38}, %r22;
	cvt.f32.bf16 	%r196, %rs37;
	cvt.f32.bf16 	%r197, %rs38;
	mov.b32 	{%rs39, %rs40}, %r23;
	cvt.f32.bf16 	%r198, %rs39;
	cvt.f32.bf16 	%r199, %rs40;
	.loc	1 298 56                        // sk05_mlp_gateup.py:298:56
	mul.f32 	%r200, %r162, %r199;
	mul.f32 	%r201, %r161, %r198;
	mul.f32 	%r202, %r160, %r197;
	mul.f32 	%r203, %r159, %r196;
	.loc	1 299 36                        // sk05_mlp_gateup.py:299:36
	abs.f32 	%r204, %r203;
	abs.f32 	%r205, %r202;
	abs.f32 	%r206, %r201;
	abs.f32 	%r207, %r200;
	.loc	1 297 55                        // sk05_mlp_gateup.py:297:55
	mov.b32 	{%rs41, %rs42}, %r20;
	cvt.f32.bf16 	%r208, %rs41;
	cvt.f32.bf16 	%r209, %rs42;
	mov.b32 	{%rs43, %rs44}, %r21;
	cvt.f32.bf16 	%r210, %rs43;
	cvt.f32.bf16 	%r211, %rs44;
	.loc	1 298 56                        // sk05_mlp_gateup.py:298:56
	mul.f32 	%r212, %r158, %r211;
	mul.f32 	%r213, %r157, %r210;
	mul.f32 	%r214, %r156, %r209;
	mul.f32 	%r215, %r155, %r208;
	.loc	1 299 36                        // sk05_mlp_gateup.py:299:36
	abs.f32 	%r216, %r215;
	abs.f32 	%r217, %r214;
	abs.f32 	%r218, %r213;
	abs.f32 	%r219, %r212;
	.loc	1 297 55                        // sk05_mlp_gateup.py:297:55
	mov.b32 	{%rs45, %rs46}, %r18;
	cvt.f32.bf16 	%r220, %rs45;
	cvt.f32.bf16 	%r221, %rs46;
	mov.b32 	{%rs47, %rs48}, %r19;
	cvt.f32.bf16 	%r222, %rs47;
	cvt.f32.bf16 	%r223, %rs48;
	.loc	1 298 56                        // sk05_mlp_gateup.py:298:56
	mul.f32 	%r224, %r154, %r223;
	mul.f32 	%r225, %r153, %r222;
	mul.f32 	%r226, %r152, %r221;
	mul.f32 	%r227, %r151, %r220;
	.loc	1 299 36                        // sk05_mlp_gateup.py:299:36
	abs.f32 	%r228, %r227;
	abs.f32 	%r229, %r226;
	abs.f32 	%r230, %r225;
	abs.f32 	%r231, %r224;
$L__tmp22:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r232, %r228, %r229;
	max.f32 	%r233, %r232, %r230;
	max.f32 	%r234, %r233, %r231;
	max.f32 	%r235, %r234, %r216;
	max.f32 	%r236, %r235, %r217;
	max.f32 	%r237, %r236, %r218;
	max.f32 	%r238, %r237, %r219;
	max.f32 	%r239, %r238, %r204;
	max.f32 	%r240, %r239, %r205;
	max.f32 	%r241, %r240, %r206;
	max.f32 	%r242, %r241, %r207;
	max.f32 	%r243, %r242, %r192;
	max.f32 	%r244, %r243, %r193;
	max.f32 	%r245, %r244, %r194;
	max.f32 	%r246, %r245, %r195;
$L__tmp23:
	.loc	1 297 55                        // sk05_mlp_gateup.py:297:55
	mov.b32 	{%rs49, %rs50}, %r32;
	cvt.f32.bf16 	%r247, %rs49;
	cvt.f32.bf16 	%r248, %rs50;
	mov.b32 	{%rs51, %rs52}, %r33;
	cvt.f32.bf16 	%r249, %rs51;
	cvt.f32.bf16 	%r250, %rs52;
	.loc	1 298 56                        // sk05_mlp_gateup.py:298:56
	mul.f32 	%r251, %r182, %r250;
	mul.f32 	%r252, %r181, %r249;
	mul.f32 	%r253, %r180, %r248;
	mul.f32 	%r254, %r179, %r247;
	.loc	1 299 36                        // sk05_mlp_gateup.py:299:36
	abs.f32 	%r255, %r254;
	abs.f32 	%r256, %r253;
	abs.f32 	%r257, %r252;
	abs.f32 	%r258, %r251;
	.loc	1 297 55                        // sk05_mlp_gateup.py:297:55
	mov.b32 	{%rs53, %rs54}, %r30;
	cvt.f32.bf16 	%r259, %rs53;
	cvt.f32.bf16 	%r260, %rs54;
	mov.b32 	{%rs55, %rs56}, %r31;
	cvt.f32.bf16 	%r261, %rs55;
	cvt.f32.bf16 	%r262, %rs56;
	.loc	1 298 56                        // sk05_mlp_gateup.py:298:56
	mul.f32 	%r263, %r178, %r262;
	mul.f32 	%r264, %r177, %r261;
	mul.f32 	%r265, %r176, %r260;
	mul.f32 	%r266, %r175, %r259;
	.loc	1 299 36                        // sk05_mlp_gateup.py:299:36
	abs.f32 	%r267, %r266;
	abs.f32 	%r268, %r265;
	abs.f32 	%r269, %r264;
	abs.f32 	%r270, %r263;
	.loc	1 297 55                        // sk05_mlp_gateup.py:297:55
	mov.b32 	{%rs57, %rs58}, %r28;
	cvt.f32.bf16 	%r271, %rs57;
	cvt.f32.bf16 	%r272, %rs58;
	mov.b32 	{%rs59, %rs60}, %r29;
	cvt.f32.bf16 	%r273, %rs59;
	cvt.f32.bf16 	%r274, %rs60;
	.loc	1 298 56                        // sk05_mlp_gateup.py:298:56
	mul.f32 	%r275, %r174, %r274;
	mul.f32 	%r276, %r173, %r273;
	mul.f32 	%r277, %r172, %r272;
	mul.f32 	%r278, %r171, %r271;
	.loc	1 299 36                        // sk05_mlp_gateup.py:299:36
	abs.f32 	%r279, %r278;
	abs.f32 	%r280, %r277;
	abs.f32 	%r281, %r276;
	abs.f32 	%r282, %r275;
	.loc	1 297 55                        // sk05_mlp_gateup.py:297:55
	mov.b32 	{%rs61, %rs62}, %r26;
	cvt.f32.bf16 	%r283, %rs61;
	cvt.f32.bf16 	%r284, %rs62;
	mov.b32 	{%rs63, %rs64}, %r27;
	cvt.f32.bf16 	%r285, %rs63;
	cvt.f32.bf16 	%r286, %rs64;
	.loc	1 298 56                        // sk05_mlp_gateup.py:298:56
	mul.f32 	%r287, %r170, %r286;
	mul.f32 	%r288, %r169, %r285;
	mul.f32 	%r289, %r168, %r284;
	mul.f32 	%r290, %r167, %r283;
	.loc	1 299 36                        // sk05_mlp_gateup.py:299:36
	abs.f32 	%r291, %r290;
	abs.f32 	%r292, %r289;
	abs.f32 	%r293, %r288;
	abs.f32 	%r294, %r287;
$L__tmp24:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r295, %r246, %r291;
	max.f32 	%r296, %r295, %r292;
	max.f32 	%r297, %r296, %r293;
	max.f32 	%r298, %r297, %r294;
	max.f32 	%r299, %r298, %r279;
	max.f32 	%r300, %r299, %r280;
	max.f32 	%r301, %r300, %r281;
	max.f32 	%r302, %r301, %r282;
	max.f32 	%r303, %r302, %r267;
	max.f32 	%r304, %r303, %r268;
	max.f32 	%r305, %r304, %r269;
	max.f32 	%r306, %r305, %r270;
	max.f32 	%r307, %r306, %r255;
	max.f32 	%r308, %r307, %r256;
	max.f32 	%r309, %r308, %r257;
	max.f32 	%r310, %r309, %r258;
$L__tmp25:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	shfl.sync.bfly.b32 	%r311, %r310, 16, 31, -1;
$L__tmp26:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r312, %r310, %r311;
$L__tmp27:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	shfl.sync.bfly.b32 	%r313, %r312, 8, 31, -1;
$L__tmp28:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r314, %r312, %r313;
$L__tmp29:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	shfl.sync.bfly.b32 	%r315, %r314, 4, 31, -1;
$L__tmp30:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r316, %r314, %r315;
$L__tmp31:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	shfl.sync.bfly.b32 	%r317, %r316, 2, 31, -1;
$L__tmp32:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r318, %r316, %r317;
$L__tmp33:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	shfl.sync.bfly.b32 	%r319, %r318, 1, 31, -1;
$L__tmp34:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r39, %r318, %r319;
$L__tmp35:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	// begin inline asm
	@%p3 st.shared.b32 [ %r34 + 0 ], %r39;
	// end inline asm
	bar.sync 	0;
	// begin inline asm
	@%p4 ld.shared.b32 %r40, [ %r37 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r320, %r40, 4, 31, -1;
$L__tmp36:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r321, %r40, %r320;
$L__tmp37:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	shfl.sync.bfly.b32 	%r322, %r321, 2, 31, -1;
$L__tmp38:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r323, %r321, %r322;
$L__tmp39:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	shfl.sync.bfly.b32 	%r324, %r323, 1, 31, -1;
$L__tmp40:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r41, %r323, %r324;
$L__tmp41:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	// begin inline asm
	@%p5 st.shared.b32 [ %r37 + 0 ], %r41;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r325, [global_smem];
$L__tmp42:
	.loc	1 299 49                        // sk05_mlp_gateup.py:299:49
	max.f32 	%r326, %r325, 0f0DA24260;
	mov.b32 	%r327, 0f42FE0000;
	.loc	1 300 22                        // sk05_mlp_gateup.py:300:22
	div.full.f32 	%r328, %r327, %r326;
	.loc	1 300 14                        // sk05_mlp_gateup.py:300:14
	mul.f32 	%r329, %r226, %r328;
	mul.f32 	%r330, %r227, %r328;
	mul.f32 	%r331, %r224, %r328;
	mul.f32 	%r332, %r225, %r328;
	mul.f32 	%r333, %r214, %r328;
	mul.f32 	%r334, %r215, %r328;
	mul.f32 	%r335, %r212, %r328;
	mul.f32 	%r336, %r213, %r328;
	mul.f32 	%r337, %r202, %r328;
	mul.f32 	%r338, %r203, %r328;
	mul.f32 	%r339, %r200, %r328;
	mul.f32 	%r340, %r201, %r328;
	mul.f32 	%r341, %r190, %r328;
	mul.f32 	%r342, %r191, %r328;
	mul.f32 	%r343, %r188, %r328;
	mul.f32 	%r344, %r189, %r328;
	mul.f32 	%r345, %r289, %r328;
	mul.f32 	%r346, %r290, %r328;
	mul.f32 	%r347, %r287, %r328;
	mul.f32 	%r348, %r288, %r328;
	mul.f32 	%r349, %r277, %r328;
	mul.f32 	%r350, %r278, %r328;
	mul.f32 	%r351, %r275, %r328;
	mul.f32 	%r352, %r276, %r328;
	mul.f32 	%r353, %r265, %r328;
	mul.f32 	%r354, %r266, %r328;
	mul.f32 	%r355, %r263, %r328;
	mul.f32 	%r356, %r264, %r328;
	mul.f32 	%r357, %r253, %r328;
	mul.f32 	%r358, %r254, %r328;
	mul.f32 	%r359, %r251, %r328;
	mul.f32 	%r360, %r252, %r328;
	.loc	1 301 29                        // sk05_mlp_gateup.py:301:29
	.loc	1 301 39                        // sk05_mlp_gateup.py:301:39
	lop3.b32 	%r361, 0x3f000000, %r329, 0x80000000, 0xF8;
	lop3.b32 	%r362, 0x3f000000, %r330, 0x80000000, 0xF8;
	lop3.b32 	%r363, 0x3f000000, %r331, 0x80000000, 0xF8;
	lop3.b32 	%r364, 0x3f000000, %r332, 0x80000000, 0xF8;
	lop3.b32 	%r365, 0x3f000000, %r333, 0x80000000, 0xF8;
	lop3.b32 	%r366, 0x3f000000, %r334, 0x80000000, 0xF8;
	lop3.b32 	%r367, 0x3f000000, %r335, 0x80000000, 0xF8;
	lop3.b32 	%r368, 0x3f000000, %r336, 0x80000000, 0xF8;
	lop3.b32 	%r369, 0x3f000000, %r337, 0x80000000, 0xF8;
	lop3.b32 	%r370, 0x3f000000, %r338, 0x80000000, 0xF8;
	lop3.b32 	%r371, 0x3f000000, %r339, 0x80000000, 0xF8;
	lop3.b32 	%r372, 0x3f000000, %r340, 0x80000000, 0xF8;
	lop3.b32 	%r373, 0x3f000000, %r341, 0x80000000, 0xF8;
	lop3.b32 	%r374, 0x3f000000, %r342, 0x80000000, 0xF8;
	lop3.b32 	%r375, 0x3f000000, %r343, 0x80000000, 0xF8;
	lop3.b32 	%r376, 0x3f000000, %r344, 0x80000000, 0xF8;
	lop3.b32 	%r377, 0x3f000000, %r345, 0x80000000, 0xF8;
	lop3.b32 	%r378, 0x3f000000, %r346, 0x80000000, 0xF8;
	lop3.b32 	%r379, 0x3f000000, %r347, 0x80000000, 0xF8;
	lop3.b32 	%r380, 0x3f000000, %r348, 0x80000000, 0xF8;
	lop3.b32 	%r381, 0x3f000000, %r349, 0x80000000, 0xF8;
	lop3.b32 	%r382, 0x3f000000, %r350, 0x80000000, 0xF8;
	lop3.b32 	%r383, 0x3f000000, %r351, 0x80000000, 0xF8;
	lop3.b32 	%r384, 0x3f000000, %r352, 0x80000000, 0xF8;
	lop3.b32 	%r385, 0x3f000000, %r353, 0x80000000, 0xF8;
	lop3.b32 	%r386, 0x3f000000, %r354, 0x80000000, 0xF8;
	lop3.b32 	%r387, 0x3f000000, %r355, 0x80000000, 0xF8;
	lop3.b32 	%r388, 0x3f000000, %r356, 0x80000000, 0xF8;
	lop3.b32 	%r389, 0x3f000000, %r357, 0x80000000, 0xF8;
	lop3.b32 	%r390, 0x3f000000, %r358, 0x80000000, 0xF8;
	lop3.b32 	%r391, 0x3f000000, %r359, 0x80000000, 0xF8;
	lop3.b32 	%r392, 0x3f000000, %r360, 0x80000000, 0xF8;
	.loc	1 301 14                        // sk05_mlp_gateup.py:301:14
	fma.rn.f32 	%r393, %r225, %r328, %r364;
	fma.rn.f32 	%r394, %r224, %r328, %r363;
	fma.rn.f32 	%r395, %r227, %r328, %r362;
	fma.rn.f32 	%r396, %r226, %r328, %r361;
	fma.rn.f32 	%r397, %r213, %r328, %r368;
	fma.rn.f32 	%r398, %r212, %r328, %r367;
	fma.rn.f32 	%r399, %r215, %r328, %r366;
	fma.rn.f32 	%r400, %r214, %r328, %r365;
	fma.rn.f32 	%r401, %r201, %r328, %r372;
	fma.rn.f32 	%r402, %r200, %r328, %r371;
	fma.rn.f32 	%r403, %r203, %r328, %r370;
	fma.rn.f32 	%r404, %r202, %r328, %r369;
	fma.rn.f32 	%r405, %r189, %r328, %r376;
	fma.rn.f32 	%r406, %r188, %r328, %r375;
	fma.rn.f32 	%r407, %r191, %r328, %r374;
	fma.rn.f32 	%r408, %r190, %r328, %r373;
	fma.rn.f32 	%r409, %r288, %r328, %r380;
	fma.rn.f32 	%r410, %r287, %r328, %r379;
	fma.rn.f32 	%r411, %r290, %r328, %r378;
	fma.rn.f32 	%r412, %r289, %r328, %r377;
	fma.rn.f32 	%r413, %r276, %r328, %r384;
	fma.rn.f32 	%r414, %r275, %r328, %r383;
	fma.rn.f32 	%r415, %r278, %r328, %r382;
	fma.rn.f32 	%r416, %r277, %r328, %r381;
	fma.rn.f32 	%r417, %r264, %r328, %r388;
	fma.rn.f32 	%r418, %r263, %r328, %r387;
	fma.rn.f32 	%r419, %r266, %r328, %r386;
	fma.rn.f32 	%r420, %r265, %r328, %r385;
	fma.rn.f32 	%r421, %r252, %r328, %r392;
	fma.rn.f32 	%r422, %r251, %r328, %r391;
	fma.rn.f32 	%r423, %r254, %r328, %r390;
	fma.rn.f32 	%r424, %r253, %r328, %r389;
	.loc	1 301 49                        // sk05_mlp_gateup.py:301:49
	cvt.rzi.s32.f32 	%r425, %r396;
	cvt.rzi.s32.f32 	%r426, %r395;
	cvt.rzi.s32.f32 	%r427, %r394;
	cvt.rzi.s32.f32 	%r428, %r393;
	cvt.rzi.s32.f32 	%r429, %r400;
	cvt.rzi.s32.f32 	%r430, %r399;
	cvt.rzi.s32.f32 	%r431, %r398;
	cvt.rzi.s32.f32 	%r432, %r397;
	cvt.rzi.s32.f32 	%r433, %r404;
	cvt.rzi.s32.f32 	%r434, %r403;
	cvt.rzi.s32.f32 	%r435, %r402;
	cvt.rzi.s32.f32 	%r436, %r401;
	cvt.rzi.s32.f32 	%r437, %r408;
	cvt.rzi.s32.f32 	%r438, %r407;
	cvt.rzi.s32.f32 	%r439, %r406;
	cvt.rzi.s32.f32 	%r440, %r405;
	cvt.rzi.s32.f32 	%r441, %r412;
	cvt.rzi.s32.f32 	%r442, %r411;
	cvt.rzi.s32.f32 	%r443, %r410;
	cvt.rzi.s32.f32 	%r444, %r409;
	cvt.rzi.s32.f32 	%r445, %r416;
	cvt.rzi.s32.f32 	%r446, %r415;
	cvt.rzi.s32.f32 	%r447, %r414;
	cvt.rzi.s32.f32 	%r448, %r413;
	cvt.rzi.s32.f32 	%r449, %r420;
	cvt.rzi.s32.f32 	%r450, %r419;
	cvt.rzi.s32.f32 	%r451, %r418;
	cvt.rzi.s32.f32 	%r452, %r417;
	cvt.rzi.s32.f32 	%r453, %r424;
	cvt.rzi.s32.f32 	%r454, %r423;
	cvt.rzi.s32.f32 	%r455, %r422;
	cvt.rzi.s32.f32 	%r456, %r421;
	.loc	1 302 70                        // sk05_mlp_gateup.py:302:70
	max.s32 	%r457, %r428, -127;
	max.s32 	%r458, %r427, -127;
	max.s32 	%r459, %r426, -127;
	max.s32 	%r460, %r425, -127;
	max.s32 	%r461, %r432, -127;
	max.s32 	%r462, %r431, -127;
	max.s32 	%r463, %r430, -127;
	max.s32 	%r464, %r429, -127;
	max.s32 	%r465, %r436, -127;
	max.s32 	%r466, %r435, -127;
	max.s32 	%r467, %r434, -127;
	max.s32 	%r468, %r433, -127;
	max.s32 	%r469, %r440, -127;
	max.s32 	%r470, %r439, -127;
	max.s32 	%r471, %r438, -127;
	max.s32 	%r472, %r437, -127;
	max.s32 	%r473, %r444, -127;
	max.s32 	%r474, %r443, -127;
	max.s32 	%r475, %r442, -127;
	max.s32 	%r476, %r441, -127;
	max.s32 	%r477, %r448, -127;
	max.s32 	%r478, %r447, -127;
	max.s32 	%r479, %r446, -127;
	max.s32 	%r480, %r445, -127;
	max.s32 	%r481, %r452, -127;
	max.s32 	%r482, %r451, -127;
	max.s32 	%r483, %r450, -127;
	max.s32 	%r484, %r449, -127;
	max.s32 	%r485, %r456, -127;
	max.s32 	%r486, %r455, -127;
	max.s32 	%r487, %r454, -127;
	max.s32 	%r488, %r453, -127;
	.loc	1 302 77                        // sk05_mlp_gateup.py:302:77
	min.s32 	%r489, %r460, 127;
	min.s32 	%r490, %r459, 127;
	min.s32 	%r491, %r458, 127;
	min.s32 	%r492, %r457, 127;
	min.s32 	%r493, %r464, 127;
	min.s32 	%r494, %r463, 127;
	min.s32 	%r495, %r462, 127;
	min.s32 	%r496, %r461, 127;
	min.s32 	%r497, %r468, 127;
	min.s32 	%r498, %r467, 127;
	min.s32 	%r499, %r466, 127;
	min.s32 	%r500, %r465, 127;
	min.s32 	%r501, %r472, 127;
	min.s32 	%r502, %r471, 127;
	min.s32 	%r503, %r470, 127;
	min.s32 	%r504, %r469, 127;
	min.s32 	%r505, %r476, 127;
	min.s32 	%r506, %r475, 127;
	min.s32 	%r507, %r474, 127;
	min.s32 	%r508, %r473, 127;
	min.s32 	%r509, %r480, 127;
	min.s32 	%r510, %r479, 127;
	min.s32 	%r511, %r478, 127;
	min.s32 	%r512, %r477, 127;
	min.s32 	%r513, %r484, 127;
	min.s32 	%r514, %r483, 127;
	min.s32 	%r515, %r482, 127;
	min.s32 	%r516, %r481, 127;
	min.s32 	%r517, %r488, 127;
	min.s32 	%r518, %r487, 127;
	min.s32 	%r519, %r486, 127;
	min.s32 	%r520, %r485, 127;
	.loc	1 302 85                        // sk05_mlp_gateup.py:302:85
	prmt.b32 	%r521, %r492, %r491, 0x3340U;
	prmt.b32 	%r522, %r490, %r489, 0x3340U;
	prmt.b32 	%r42, %r522, %r521, 0x5410U;
	prmt.b32 	%r523, %r496, %r495, 0x3340U;
	prmt.b32 	%r524, %r494, %r493, 0x3340U;
	prmt.b32 	%r43, %r524, %r523, 0x5410U;
	prmt.b32 	%r525, %r500, %r499, 0x3340U;
	prmt.b32 	%r526, %r498, %r497, 0x3340U;
	prmt.b32 	%r44, %r526, %r525, 0x5410U;
	prmt.b32 	%r527, %r504, %r503, 0x3340U;
	prmt.b32 	%r528, %r502, %r501, 0x3340U;
	prmt.b32 	%r45, %r528, %r527, 0x5410U;
	prmt.b32 	%r529, %r508, %r507, 0x3340U;
	prmt.b32 	%r530, %r506, %r505, 0x3340U;
	prmt.b32 	%r46, %r530, %r529, 0x5410U;
	prmt.b32 	%r531, %r512, %r511, 0x3340U;
	prmt.b32 	%r532, %r510, %r509, 0x3340U;
	prmt.b32 	%r47, %r532, %r531, 0x5410U;
	prmt.b32 	%r533, %r516, %r515, 0x3340U;
	prmt.b32 	%r534, %r514, %r513, 0x3340U;
	prmt.b32 	%r48, %r534, %r533, 0x5410U;
	prmt.b32 	%r535, %r520, %r519, 0x3340U;
	prmt.b32 	%r536, %r518, %r517, 0x3340U;
	prmt.b32 	%r49, %r536, %r535, 0x5410U;
	.loc	1 302 45                        // sk05_mlp_gateup.py:302:45
	// begin inline asm
	@%p1 st.global.v4.b32 [ %rd9 + 0 ], { %r42, %r43, %r44, %r45 };
	// end inline asm
	// begin inline asm
	@%p2 st.global.v4.b32 [ %rd10 + 0 ], { %r46, %r47, %r48, %r49 };
	// end inline asm
	.loc	1 303 21                        // sk05_mlp_gateup.py:303:21
	mad.wide.u32 	%rd11, %r51, 4, %rd15;
	.loc	1 303 34                        // sk05_mlp_gateup.py:303:34
	mul.f32 	%r50, %r326, 0f3C010204;
	.loc	1 303 26                        // sk05_mlp_gateup.py:303:26
	or.b32 	%r537, %r54, %r56;
	setp.eq.b32 	%p6, %r537, 0;
	// begin inline asm
	@%p6 st.global.b32 [ %rd11 + 0 ], { %r50 };
	// end inline asm
	.loc	1 303 4                         // sk05_mlp_gateup.py:303:4
	ret;
$L__tmp43:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/usr/local/lib/python3.12/dist-packages/vllm/_genesis/kernels/sk05_mlp_gateup.py"
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
.b8 1                                   // DW_CHILDREN_yes
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
.b8 5                                   // Abbreviation Code
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
.b8 6                                   // Abbreviation Code
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
.b8 11                                  // DW_FORM_data1
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 255                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0xf8 DW_TAG_compile_unit
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
.b8 53
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 103
.b8 97
.b8 116
.b8 101
.b8 117
.b8 112
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 117
.b8 115
.b8 114
.b8 47
.b8 108
.b8 111
.b8 99
.b8 97
.b8 108
.b8 47
.b8 108
.b8 105
.b8 98
.b8 47
.b8 112
.b8 121
.b8 116
.b8 104
.b8 111
.b8 110
.b8 51
.b8 46
.b8 49
.b8 50
.b8 47
.b8 100
.b8 105
.b8 115
.b8 116
.b8 45
.b8 112
.b8 97
.b8 99
.b8 107
.b8 97
.b8 103
.b8 101
.b8 115
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
.b8 2                                   // Abbrev [2] 0x6a:0x1d DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 53
.b8 95
.b8 114
.b8 109
.b8 115
.b8 110
.b8 111
.b8 114
.b8 109
.b8 95
.b8 113
.b8 117
.b8 97
.b8 110
.b8 116
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x87:0x7b DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 106                                // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x9c:0x33 DW_TAG_inlined_subroutine
.b32 106                                // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp19                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 42                                  // DW_AT_call_line
.b8 1
.b8 28                                  // DW_AT_call_column
.b8 5                                   // Abbrev [5] 0xb5:0x19 DW_TAG_inlined_subroutine
.b32 106                                // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp18                          // DW_AT_high_pc
.b8 2                                   // DW_AT_call_file
.b8 37                                  // DW_AT_call_line
.b8 1
.b8 36                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 4                                   // Abbrev [4] 0xcf:0x32 DW_TAG_inlined_subroutine
.b32 106                                // DW_AT_abstract_origin
.b64 $L__tmp20                          // DW_AT_low_pc
.b64 $L__tmp42                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 43                                  // DW_AT_call_line
.b8 1
.b8 29                                  // DW_AT_call_column
.b8 6                                   // Abbrev [6] 0xe8:0x18 DW_TAG_inlined_subroutine
.b32 106                                // DW_AT_abstract_origin
.b64 $L__tmp22                          // DW_AT_low_pc
.b64 $L__tmp41                          // DW_AT_high_pc
.b8 2                                   // DW_AT_call_file
.b8 191                                 // DW_AT_call_line
.b8 40                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_QVAR0_2 = _Nativo(
    "sk05_mlp_gateup/_sk05_rmsnorm_quant_kernel/blk8192",
    _QPTX0_2, "_sk05_rmsnorm_quant_kernel",
    warps=8, shared=32,
    abi=[0, 1, 2, 3, 4, 5, 6],
    horneado={7: 8192, 8: 1e-06},
    div16=[4, 5, 6],
)
# E1=32 lop3, E2=0 mul

_QPTX0_3 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk05_rmsnorm_quant_kernel // -- Begin function _sk05_rmsnorm_quant_kernel
.extern .shared .align 16 .b8 global_smem[];
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
                                        // @_sk05_rmsnorm_quant_kernel
.visible .entry _sk05_rmsnorm_quant_kernel(
	.param .u64 .ptr .global .align 1 _sk05_rmsnorm_quant_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk05_rmsnorm_quant_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk05_rmsnorm_quant_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk05_rmsnorm_quant_kernel_param_3,
	.param .u32 _sk05_rmsnorm_quant_kernel_param_4,
	.param .u32 _sk05_rmsnorm_quant_kernel_param_5,
	.param .u32 _sk05_rmsnorm_quant_kernel_param_6,
	.param .u64 .ptr .global .align 1 _sk05_rmsnorm_quant_kernel_param_7,
	.param .u64 .ptr .global .align 1 _sk05_rmsnorm_quant_kernel_param_8
)
.reqntid 256
{
	.reg .pred 	%p<74>;
	.reg .b16 	%rs<129>;
	.reg .b32 	%r<1013>;
	.reg .b64 	%rd<34>;
	.loc	1 287 0                         // sk05_mlp_gateup.py:287:0
$L__func_begin0:
	.loc	1 287 0                         // sk05_mlp_gateup.py:287:0

// %bb.0:                               // %__nv_rsqrtf.exit
	ld.param.b64 	%rd22, [_sk05_rmsnorm_quant_kernel_param_0];
	ld.param.b64 	%rd23, [_sk05_rmsnorm_quant_kernel_param_1];
$L__tmp0:
	.loc	1 293 24                        // sk05_mlp_gateup.py:293:24
	mov.u32 	%r91, %ctaid.x;
	ld.param.b64 	%rd24, [_sk05_rmsnorm_quant_kernel_param_2];
	.loc	1 294 24                        // sk05_mlp_gateup.py:294:24
	mov.u32 	%r92, %tid.x;
	and.b32 	%r93, %r92, 255;
	ld.param.b64 	%rd25, [_sk05_rmsnorm_quant_kernel_param_3];
	and.b32 	%r94, %r92, 31;
	ld.param.b32 	%r95, [_sk05_rmsnorm_quant_kernel_param_4];
	shr.u32 	%r96, %r92, 5;
	ld.param.b32 	%r97, [_sk05_rmsnorm_quant_kernel_param_5];
	shl.b32 	%r98, %r92, 4;
	ld.param.b32 	%r99, [_sk05_rmsnorm_quant_kernel_param_6];
	and.b32 	%r100, %r98, 4080;
	or.b32 	%r101, %r100, 4096;
	or.b32 	%r102, %r100, 8192;
	or.b32 	%r103, %r98, 12288;
	or.b32 	%r104, %r98, 12296;
	.loc	1 295 18                        // sk05_mlp_gateup.py:295:18
	setp.lt.s32 	%p1, %r100, %r95;
	setp.lt.s32 	%p2, %r101, %r95;
	setp.lt.s32 	%p3, %r102, %r95;
	setp.lt.s32 	%p4, %r103, %r95;
	.loc	1 296 30                        // sk05_mlp_gateup.py:296:30
	mul.lo.s32 	%r105, %r97, %r91;
	.loc	1 296 24                        // sk05_mlp_gateup.py:296:24
	mad.wide.s32 	%rd26, %r105, 2, %rd22;
	.loc	1 296 42                        // sk05_mlp_gateup.py:296:42
	cvt.u64.u32 	%rd27, %r100;
	mul.wide.u32 	%rd28, %r100, 2;
	add.s64 	%rd1, %rd26, %rd28;
	add.s64 	%rd2, %rd1, 16;
	add.s64 	%rd3, %rd1, 8192;
	add.s64 	%rd4, %rd1, 8208;
	add.s64 	%rd5, %rd1, 16384;
	add.s64 	%rd6, %rd1, 16400;
	cvt.u64.u32 	%rd29, %r103;
	mul.wide.u32 	%rd30, %r103, 2;
	add.s64 	%rd7, %rd26, %rd30;
	mul.wide.u32 	%rd31, %r104, 2;
	add.s64 	%rd8, %rd26, %rd31;
	mov.b32 	%r5, 0;
	.loc	1 296 16                        // sk05_mlp_gateup.py:296:16
	// begin inline asm
	mov.u32 %r1, %r5;
	mov.u32 %r2, %r5;
	mov.u32 %r3, %r5;
	mov.u32 %r4, %r5;
	@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r6, %r5;
	mov.u32 %r7, %r5;
	mov.u32 %r8, %r5;
	mov.u32 %r9, %r5;
	@%p1 ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd2 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r10, %r5;
	mov.u32 %r11, %r5;
	mov.u32 %r12, %r5;
	mov.u32 %r13, %r5;
	@%p2 ld.global.v4.b32 { %r10, %r11, %r12, %r13 }, [ %rd3 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r14, %r5;
	mov.u32 %r15, %r5;
	mov.u32 %r16, %r5;
	mov.u32 %r17, %r5;
	@%p2 ld.global.v4.b32 { %r14, %r15, %r16, %r17 }, [ %rd4 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r18, %r5;
	mov.u32 %r19, %r5;
	mov.u32 %r20, %r5;
	mov.u32 %r21, %r5;
	@%p3 ld.global.v4.b32 { %r18, %r19, %r20, %r21 }, [ %rd5 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r22, %r5;
	mov.u32 %r23, %r5;
	mov.u32 %r24, %r5;
	mov.u32 %r25, %r5;
	@%p3 ld.global.v4.b32 { %r22, %r23, %r24, %r25 }, [ %rd6 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r26, %r5;
	mov.u32 %r27, %r5;
	mov.u32 %r28, %r5;
	mov.u32 %r29, %r5;
	@%p4 ld.global.v4.b32 { %r26, %r27, %r28, %r29 }, [ %rd7 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r30, %r5;
	mov.u32 %r31, %r5;
	mov.u32 %r32, %r5;
	mov.u32 %r33, %r5;
	@%p4 ld.global.v4.b32 { %r30, %r31, %r32, %r33 }, [ %rd8 + 0 ];
	// end inline asm
	.loc	1 296 73                        // sk05_mlp_gateup.py:296:73
	mov.b32 	{%rs1, %rs2}, %r2;
	cvt.f32.bf16 	%r106, %rs2;
	cvt.f32.bf16 	%r107, %rs1;
	mov.b32 	{%rs3, %rs4}, %r1;
	cvt.f32.bf16 	%r108, %rs3;
	cvt.f32.bf16 	%r109, %rs4;
	mov.b32 	{%rs5, %rs6}, %r4;
	cvt.f32.bf16 	%r110, %rs6;
	cvt.f32.bf16 	%r111, %rs5;
	mov.b32 	{%rs7, %rs8}, %r3;
	cvt.f32.bf16 	%r112, %rs8;
	cvt.f32.bf16 	%r113, %rs7;
	mov.b32 	{%rs9, %rs10}, %r7;
	cvt.f32.bf16 	%r114, %rs10;
	cvt.f32.bf16 	%r115, %rs9;
	mov.b32 	{%rs11, %rs12}, %r6;
	cvt.f32.bf16 	%r116, %rs12;
	cvt.f32.bf16 	%r117, %rs11;
	mov.b32 	{%rs13, %rs14}, %r9;
	cvt.f32.bf16 	%r118, %rs14;
	cvt.f32.bf16 	%r119, %rs13;
	mov.b32 	{%rs15, %rs16}, %r8;
	cvt.f32.bf16 	%r120, %rs16;
	cvt.f32.bf16 	%r121, %rs15;
	mov.b32 	{%rs17, %rs18}, %r11;
	cvt.f32.bf16 	%r122, %rs18;
	cvt.f32.bf16 	%r123, %rs17;
	mov.b32 	{%rs19, %rs20}, %r10;
	cvt.f32.bf16 	%r124, %rs20;
	cvt.f32.bf16 	%r125, %rs19;
	mov.b32 	{%rs21, %rs22}, %r13;
	cvt.f32.bf16 	%r126, %rs22;
	cvt.f32.bf16 	%r127, %rs21;
	mov.b32 	{%rs23, %rs24}, %r12;
	cvt.f32.bf16 	%r128, %rs24;
	cvt.f32.bf16 	%r129, %rs23;
	mov.b32 	{%rs25, %rs26}, %r15;
	cvt.f32.bf16 	%r130, %rs26;
	cvt.f32.bf16 	%r131, %rs25;
	mov.b32 	{%rs27, %rs28}, %r14;
	cvt.f32.bf16 	%r132, %rs28;
	cvt.f32.bf16 	%r133, %rs27;
	mov.b32 	{%rs29, %rs30}, %r17;
	cvt.f32.bf16 	%r134, %rs30;
	cvt.f32.bf16 	%r135, %rs29;
	mov.b32 	{%rs31, %rs32}, %r16;
	cvt.f32.bf16 	%r136, %rs32;
	cvt.f32.bf16 	%r137, %rs31;
	mov.b32 	{%rs33, %rs34}, %r19;
	cvt.f32.bf16 	%r138, %rs34;
	cvt.f32.bf16 	%r139, %rs33;
	mov.b32 	{%rs35, %rs36}, %r18;
	cvt.f32.bf16 	%r140, %rs36;
	cvt.f32.bf16 	%r141, %rs35;
	mov.b32 	{%rs37, %rs38}, %r21;
	cvt.f32.bf16 	%r142, %rs38;
	cvt.f32.bf16 	%r143, %rs37;
	mov.b32 	{%rs39, %rs40}, %r20;
	cvt.f32.bf16 	%r144, %rs40;
	cvt.f32.bf16 	%r145, %rs39;
	mov.b32 	{%rs41, %rs42}, %r23;
	cvt.f32.bf16 	%r146, %rs42;
	cvt.f32.bf16 	%r147, %rs41;
	mov.b32 	{%rs43, %rs44}, %r22;
	cvt.f32.bf16 	%r148, %rs44;
	cvt.f32.bf16 	%r149, %rs43;
	mov.b32 	{%rs45, %rs46}, %r25;
	cvt.f32.bf16 	%r150, %rs46;
	cvt.f32.bf16 	%r151, %rs45;
	mov.b32 	{%rs47, %rs48}, %r24;
	cvt.f32.bf16 	%r152, %rs48;
	cvt.f32.bf16 	%r153, %rs47;
	mov.b32 	{%rs49, %rs50}, %r27;
	cvt.f32.bf16 	%r154, %rs50;
	cvt.f32.bf16 	%r155, %rs49;
	mov.b32 	{%rs51, %rs52}, %r26;
	cvt.f32.bf16 	%r156, %rs52;
	cvt.f32.bf16 	%r157, %rs51;
	mov.b32 	{%rs53, %rs54}, %r29;
	cvt.f32.bf16 	%r158, %rs54;
	cvt.f32.bf16 	%r159, %rs53;
	mov.b32 	{%rs55, %rs56}, %r28;
	cvt.f32.bf16 	%r160, %rs56;
	cvt.f32.bf16 	%r161, %rs55;
	mov.b32 	{%rs57, %rs58}, %r31;
	cvt.f32.bf16 	%r162, %rs58;
	cvt.f32.bf16 	%r163, %rs57;
	mov.b32 	{%rs59, %rs60}, %r30;
	cvt.f32.bf16 	%r164, %rs60;
	cvt.f32.bf16 	%r165, %rs59;
	mov.b32 	{%rs61, %rs62}, %r33;
	cvt.f32.bf16 	%r166, %rs62;
	cvt.f32.bf16 	%r167, %rs61;
	mov.b32 	{%rs63, %rs64}, %r32;
	cvt.f32.bf16 	%r168, %rs64;
	cvt.f32.bf16 	%r169, %rs63;
	.loc	1 297 24                        // sk05_mlp_gateup.py:297:24
	add.s64 	%rd9, %rd23, %rd28;
	add.s64 	%rd10, %rd9, 16;
	add.s64 	%rd11, %rd9, 8192;
	add.s64 	%rd12, %rd9, 8208;
	add.s64 	%rd13, %rd9, 16384;
	add.s64 	%rd14, %rd9, 16400;
	add.s64 	%rd15, %rd23, %rd30;
	add.s64 	%rd16, %rd23, %rd31;
	.loc	1 297 16                        // sk05_mlp_gateup.py:297:16
	// begin inline asm
	mov.u32 %r34, %r5;
	mov.u32 %r35, %r5;
	mov.u32 %r36, %r5;
	mov.u32 %r37, %r5;
	@%p1 ld.global.v4.b32 { %r34, %r35, %r36, %r37 }, [ %rd9 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r38, %r5;
	mov.u32 %r39, %r5;
	mov.u32 %r40, %r5;
	mov.u32 %r41, %r5;
	@%p1 ld.global.v4.b32 { %r38, %r39, %r40, %r41 }, [ %rd10 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r42, %r5;
	mov.u32 %r43, %r5;
	mov.u32 %r44, %r5;
	mov.u32 %r45, %r5;
	@%p2 ld.global.v4.b32 { %r42, %r43, %r44, %r45 }, [ %rd11 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r46, %r5;
	mov.u32 %r47, %r5;
	mov.u32 %r48, %r5;
	mov.u32 %r49, %r5;
	@%p2 ld.global.v4.b32 { %r46, %r47, %r48, %r49 }, [ %rd12 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r50, %r5;
	mov.u32 %r51, %r5;
	mov.u32 %r52, %r5;
	mov.u32 %r53, %r5;
	@%p3 ld.global.v4.b32 { %r50, %r51, %r52, %r53 }, [ %rd13 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r54, %r5;
	mov.u32 %r55, %r5;
	mov.u32 %r56, %r5;
	mov.u32 %r57, %r5;
	@%p3 ld.global.v4.b32 { %r54, %r55, %r56, %r57 }, [ %rd14 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r58, %r5;
	mov.u32 %r59, %r5;
	mov.u32 %r60, %r5;
	mov.u32 %r61, %r5;
	@%p4 ld.global.v4.b32 { %r58, %r59, %r60, %r61 }, [ %rd15 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r62, %r5;
	mov.u32 %r63, %r5;
	mov.u32 %r64, %r5;
	mov.u32 %r65, %r5;
	@%p4 ld.global.v4.b32 { %r62, %r63, %r64, %r65 }, [ %rd16 + 0 ];
	// end inline asm
	.loc	1 298 32                        // sk05_mlp_gateup.py:298:32
	mul.f32 	%r170, %r109, %r109;
$L__tmp1:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	fma.rn.f32 	%r171, %r108, %r108, %r170;
	fma.rn.f32 	%r172, %r107, %r107, %r171;
	fma.rn.f32 	%r173, %r106, %r106, %r172;
	fma.rn.f32 	%r174, %r113, %r113, %r173;
	fma.rn.f32 	%r175, %r112, %r112, %r174;
	fma.rn.f32 	%r176, %r111, %r111, %r175;
	fma.rn.f32 	%r177, %r110, %r110, %r176;
	fma.rn.f32 	%r178, %r117, %r117, %r177;
	fma.rn.f32 	%r179, %r116, %r116, %r178;
	fma.rn.f32 	%r180, %r115, %r115, %r179;
	fma.rn.f32 	%r181, %r114, %r114, %r180;
	fma.rn.f32 	%r182, %r121, %r121, %r181;
	fma.rn.f32 	%r183, %r120, %r120, %r182;
	fma.rn.f32 	%r184, %r119, %r119, %r183;
	fma.rn.f32 	%r185, %r118, %r118, %r184;
	fma.rn.f32 	%r186, %r125, %r125, %r185;
	fma.rn.f32 	%r187, %r124, %r124, %r186;
	fma.rn.f32 	%r188, %r123, %r123, %r187;
	fma.rn.f32 	%r189, %r122, %r122, %r188;
	fma.rn.f32 	%r190, %r129, %r129, %r189;
	fma.rn.f32 	%r191, %r128, %r128, %r190;
	fma.rn.f32 	%r192, %r127, %r127, %r191;
	fma.rn.f32 	%r193, %r126, %r126, %r192;
	fma.rn.f32 	%r194, %r133, %r133, %r193;
	fma.rn.f32 	%r195, %r132, %r132, %r194;
	fma.rn.f32 	%r196, %r131, %r131, %r195;
	fma.rn.f32 	%r197, %r130, %r130, %r196;
	fma.rn.f32 	%r198, %r137, %r137, %r197;
	fma.rn.f32 	%r199, %r136, %r136, %r198;
	fma.rn.f32 	%r200, %r135, %r135, %r199;
	fma.rn.f32 	%r201, %r134, %r134, %r200;
	fma.rn.f32 	%r202, %r141, %r141, %r201;
	fma.rn.f32 	%r203, %r140, %r140, %r202;
	fma.rn.f32 	%r204, %r139, %r139, %r203;
	fma.rn.f32 	%r205, %r138, %r138, %r204;
	fma.rn.f32 	%r206, %r145, %r145, %r205;
	fma.rn.f32 	%r207, %r144, %r144, %r206;
	fma.rn.f32 	%r208, %r143, %r143, %r207;
	fma.rn.f32 	%r209, %r142, %r142, %r208;
	fma.rn.f32 	%r210, %r149, %r149, %r209;
	fma.rn.f32 	%r211, %r148, %r148, %r210;
	fma.rn.f32 	%r212, %r147, %r147, %r211;
	fma.rn.f32 	%r213, %r146, %r146, %r212;
	fma.rn.f32 	%r214, %r153, %r153, %r213;
	fma.rn.f32 	%r215, %r152, %r152, %r214;
	fma.rn.f32 	%r216, %r151, %r151, %r215;
	fma.rn.f32 	%r217, %r150, %r150, %r216;
	fma.rn.f32 	%r218, %r157, %r157, %r217;
	fma.rn.f32 	%r219, %r156, %r156, %r218;
	fma.rn.f32 	%r220, %r155, %r155, %r219;
	fma.rn.f32 	%r221, %r154, %r154, %r220;
	fma.rn.f32 	%r222, %r161, %r161, %r221;
	fma.rn.f32 	%r223, %r160, %r160, %r222;
	fma.rn.f32 	%r224, %r159, %r159, %r223;
	fma.rn.f32 	%r225, %r158, %r158, %r224;
	fma.rn.f32 	%r226, %r165, %r165, %r225;
	fma.rn.f32 	%r227, %r164, %r164, %r226;
	fma.rn.f32 	%r228, %r163, %r163, %r227;
	fma.rn.f32 	%r229, %r162, %r162, %r228;
	fma.rn.f32 	%r230, %r169, %r169, %r229;
	fma.rn.f32 	%r231, %r168, %r168, %r230;
	fma.rn.f32 	%r232, %r167, %r167, %r231;
	fma.rn.f32 	%r233, %r166, %r166, %r232;
$L__tmp2:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	shfl.sync.bfly.b32 	%r234, %r233, 16, 31, -1;
$L__tmp3:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	add.f32 	%r235, %r233, %r234;
$L__tmp4:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	shfl.sync.bfly.b32 	%r236, %r235, 8, 31, -1;
$L__tmp5:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	add.f32 	%r237, %r235, %r236;
$L__tmp6:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	shfl.sync.bfly.b32 	%r238, %r237, 4, 31, -1;
$L__tmp7:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	add.f32 	%r239, %r237, %r238;
$L__tmp8:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	shfl.sync.bfly.b32 	%r240, %r239, 2, 31, -1;
$L__tmp9:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	add.f32 	%r241, %r239, %r240;
$L__tmp10:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	shfl.sync.bfly.b32 	%r242, %r241, 1, 31, -1;
$L__tmp11:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	add.f32 	%r67, %r241, %r242;
$L__tmp12:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	setp.eq.b32 	%p5, %r94, 0;
	shr.u32 	%r243, %r92, 3;
	and.b32 	%r244, %r243, 28;
	mov.b32 	%r245, global_smem;
	add.s32 	%r66, %r245, %r244;
	// begin inline asm
	@%p5 st.shared.b32 [ %r66 + 0 ], %r67;
	// end inline asm
	bar.sync 	0;
	setp.lt.u32 	%p6, %r93, 8;
	shl.b32 	%r246, %r93, 2;
	add.s32 	%r69, %r245, %r246;
	// begin inline asm
	@%p6 ld.shared.b32 %r68, [ %r69 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r247, %r68, 4, 31, -1;
$L__tmp13:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	add.f32 	%r248, %r68, %r247;
$L__tmp14:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	shfl.sync.bfly.b32 	%r249, %r248, 2, 31, -1;
$L__tmp15:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	add.f32 	%r250, %r248, %r249;
$L__tmp16:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	shfl.sync.bfly.b32 	%r251, %r250, 1, 31, -1;
$L__tmp17:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ] ]
	add.f32 	%r70, %r250, %r251;
$L__tmp18:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:298:28 ]
	and.b32 	%r252, %r92, 7;
	setp.eq.b32 	%p9, %r252, 0;
	and.pred 	%p7, %p6, %p9;
	// begin inline asm
	@%p7 st.shared.b32 [ %r69 + 0 ], %r70;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r253, [global_smem];
$L__tmp19:
	.loc	1 298 45                        // sk05_mlp_gateup.py:298:45
	cvt.rn.f32.s32 	%r254, %r95;
	div.full.f32 	%r255, %r253, %r254;
	.loc	1 298 49                        // sk05_mlp_gateup.py:298:49
	add.f32 	%r256, %r255, 0f358637BD;
	.loc	1 298 21                        // sk05_mlp_gateup.py:298:21
	rsqrt.approx.ftz.f32 	%r257, %r256;
	.loc	1 298 12                        // sk05_mlp_gateup.py:298:12
	mul.f32 	%r258, %r257, %r108;
	mul.f32 	%r259, %r257, %r109;
	mul.f32 	%r260, %r257, %r107;
	mul.f32 	%r261, %r257, %r106;
	mul.f32 	%r262, %r257, %r113;
	mul.f32 	%r263, %r257, %r112;
	mul.f32 	%r264, %r257, %r111;
	mul.f32 	%r265, %r257, %r110;
	mul.f32 	%r266, %r257, %r117;
	mul.f32 	%r267, %r257, %r116;
	mul.f32 	%r268, %r257, %r115;
	mul.f32 	%r269, %r257, %r114;
	mul.f32 	%r270, %r257, %r121;
	mul.f32 	%r271, %r257, %r120;
	mul.f32 	%r272, %r257, %r119;
	mul.f32 	%r273, %r257, %r118;
	mul.f32 	%r274, %r257, %r125;
	mul.f32 	%r275, %r257, %r124;
	mul.f32 	%r276, %r257, %r123;
	mul.f32 	%r277, %r257, %r122;
	mul.f32 	%r278, %r257, %r129;
	mul.f32 	%r279, %r257, %r128;
	mul.f32 	%r280, %r257, %r127;
	mul.f32 	%r281, %r257, %r126;
	mul.f32 	%r282, %r257, %r133;
	mul.f32 	%r283, %r257, %r132;
	mul.f32 	%r284, %r257, %r131;
	mul.f32 	%r285, %r257, %r130;
	mul.f32 	%r286, %r257, %r137;
	mul.f32 	%r287, %r257, %r136;
	mul.f32 	%r288, %r257, %r135;
	mul.f32 	%r289, %r257, %r134;
	mul.f32 	%r290, %r257, %r141;
	mul.f32 	%r291, %r257, %r140;
	mul.f32 	%r292, %r257, %r139;
	mul.f32 	%r293, %r257, %r138;
	mul.f32 	%r294, %r257, %r145;
	mul.f32 	%r295, %r257, %r144;
	mul.f32 	%r296, %r257, %r143;
	mul.f32 	%r297, %r257, %r142;
	mul.f32 	%r298, %r257, %r149;
	mul.f32 	%r299, %r257, %r148;
	mul.f32 	%r300, %r257, %r147;
	mul.f32 	%r301, %r257, %r146;
	mul.f32 	%r302, %r257, %r153;
	mul.f32 	%r303, %r257, %r152;
	mul.f32 	%r304, %r257, %r151;
	mul.f32 	%r305, %r257, %r150;
	mul.f32 	%r306, %r257, %r157;
	mul.f32 	%r307, %r257, %r156;
	mul.f32 	%r308, %r257, %r155;
	mul.f32 	%r309, %r257, %r154;
	mul.f32 	%r310, %r257, %r161;
	mul.f32 	%r311, %r257, %r160;
	mul.f32 	%r312, %r257, %r159;
	mul.f32 	%r313, %r257, %r158;
	mul.f32 	%r314, %r257, %r165;
	mul.f32 	%r315, %r257, %r164;
	mul.f32 	%r316, %r257, %r163;
	mul.f32 	%r317, %r257, %r162;
	mul.f32 	%r318, %r257, %r169;
	mul.f32 	%r319, %r257, %r168;
	mul.f32 	%r320, %r257, %r167;
	mul.f32 	%r321, %r257, %r166;
$L__tmp20:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	bar.sync 	0;
$L__tmp21:
	.loc	1 302 27                        // sk05_mlp_gateup.py:302:27
	mul.lo.s32 	%r322, %r99, %r91;
	.loc	1 302 21                        // sk05_mlp_gateup.py:302:21
	cvt.s64.s32 	%rd32, %r322;
	add.s64 	%rd33, %rd24, %rd32;
	.loc	1 302 39                        // sk05_mlp_gateup.py:302:39
	add.s64 	%rd17, %rd33, %rd27;
	add.s64 	%rd18, %rd17, 4096;
	add.s64 	%rd19, %rd17, 8192;
	add.s64 	%rd20, %rd33, %rd29;
	.loc	1 297 55                        // sk05_mlp_gateup.py:297:55
	mov.b32 	{%rs65, %rs66}, %r40;
	cvt.f32.bf16 	%r323, %rs65;
	cvt.f32.bf16 	%r324, %rs66;
	mov.b32 	{%rs67, %rs68}, %r41;
	cvt.f32.bf16 	%r325, %rs67;
	cvt.f32.bf16 	%r326, %rs68;
	.loc	1 298 56                        // sk05_mlp_gateup.py:298:56
	mul.f32 	%r327, %r273, %r326;
	mul.f32 	%r328, %r272, %r325;
	mul.f32 	%r329, %r271, %r324;
	mul.f32 	%r330, %r270, %r323;
	.loc	1 299 36                        // sk05_mlp_gateup.py:299:36
	abs.f32 	%r331, %r330;
	abs.f32 	%r332, %r329;
	abs.f32 	%r333, %r328;
	abs.f32 	%r334, %r327;
	.loc	1 297 55                        // sk05_mlp_gateup.py:297:55
	mov.b32 	{%rs69, %rs70}, %r38;
	cvt.f32.bf16 	%r335, %rs69;
	cvt.f32.bf16 	%r336, %rs70;
	mov.b32 	{%rs71, %rs72}, %r39;
	cvt.f32.bf16 	%r337, %rs71;
	cvt.f32.bf16 	%r338, %rs72;
	.loc	1 298 56                        // sk05_mlp_gateup.py:298:56
	mul.f32 	%r339, %r269, %r338;
	mul.f32 	%r340, %r268, %r337;
	mul.f32 	%r341, %r267, %r336;
	mul.f32 	%r342, %r266, %r335;
	.loc	1 299 36                        // sk05_mlp_gateup.py:299:36
	abs.f32 	%r343, %r342;
	abs.f32 	%r344, %r341;
	abs.f32 	%r345, %r340;
	abs.f32 	%r346, %r339;
	.loc	1 297 55                        // sk05_mlp_gateup.py:297:55
	mov.b32 	{%rs73, %rs74}, %r36;
	cvt.f32.bf16 	%r347, %rs73;
	cvt.f32.bf16 	%r348, %rs74;
	mov.b32 	{%rs75, %rs76}, %r37;
	cvt.f32.bf16 	%r349, %rs75;
	cvt.f32.bf16 	%r350, %rs76;
	.loc	1 298 56                        // sk05_mlp_gateup.py:298:56
	mul.f32 	%r351, %r265, %r350;
	mul.f32 	%r352, %r264, %r349;
	mul.f32 	%r353, %r263, %r348;
	mul.f32 	%r354, %r262, %r347;
	.loc	1 299 36                        // sk05_mlp_gateup.py:299:36
	abs.f32 	%r355, %r354;
	abs.f32 	%r356, %r353;
	abs.f32 	%r357, %r352;
	abs.f32 	%r358, %r351;
	.loc	1 297 55                        // sk05_mlp_gateup.py:297:55
	mov.b32 	{%rs77, %rs78}, %r34;
	cvt.f32.bf16 	%r359, %rs77;
	cvt.f32.bf16 	%r360, %rs78;
	mov.b32 	{%rs79, %rs80}, %r35;
	cvt.f32.bf16 	%r361, %rs79;
	cvt.f32.bf16 	%r362, %rs80;
	.loc	1 298 56                        // sk05_mlp_gateup.py:298:56
	mul.f32 	%r363, %r261, %r362;
	mul.f32 	%r364, %r260, %r361;
	mul.f32 	%r365, %r259, %r360;
	mul.f32 	%r366, %r258, %r359;
	.loc	1 299 36                        // sk05_mlp_gateup.py:299:36
	abs.f32 	%r367, %r366;
	abs.f32 	%r368, %r365;
	abs.f32 	%r369, %r364;
	abs.f32 	%r370, %r363;
$L__tmp22:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r371, %r367, %r368;
	max.f32 	%r372, %r371, %r369;
	max.f32 	%r373, %r372, %r370;
	max.f32 	%r374, %r373, %r355;
	max.f32 	%r375, %r374, %r356;
	max.f32 	%r376, %r375, %r357;
	max.f32 	%r377, %r376, %r358;
	max.f32 	%r378, %r377, %r343;
	max.f32 	%r379, %r378, %r344;
	max.f32 	%r380, %r379, %r345;
	max.f32 	%r381, %r380, %r346;
	max.f32 	%r382, %r381, %r331;
	max.f32 	%r383, %r382, %r332;
	max.f32 	%r384, %r383, %r333;
	max.f32 	%r385, %r384, %r334;
$L__tmp23:
	.loc	1 297 55                        // sk05_mlp_gateup.py:297:55
	mov.b32 	{%rs81, %rs82}, %r48;
	cvt.f32.bf16 	%r386, %rs81;
	cvt.f32.bf16 	%r387, %rs82;
	mov.b32 	{%rs83, %rs84}, %r49;
	cvt.f32.bf16 	%r388, %rs83;
	cvt.f32.bf16 	%r389, %rs84;
	.loc	1 298 56                        // sk05_mlp_gateup.py:298:56
	mul.f32 	%r390, %r289, %r389;
	mul.f32 	%r391, %r288, %r388;
	mul.f32 	%r392, %r287, %r387;
	mul.f32 	%r393, %r286, %r386;
	.loc	1 299 36                        // sk05_mlp_gateup.py:299:36
	abs.f32 	%r394, %r393;
	abs.f32 	%r395, %r392;
	abs.f32 	%r396, %r391;
	abs.f32 	%r397, %r390;
	.loc	1 297 55                        // sk05_mlp_gateup.py:297:55
	mov.b32 	{%rs85, %rs86}, %r46;
	cvt.f32.bf16 	%r398, %rs85;
	cvt.f32.bf16 	%r399, %rs86;
	mov.b32 	{%rs87, %rs88}, %r47;
	cvt.f32.bf16 	%r400, %rs87;
	cvt.f32.bf16 	%r401, %rs88;
	.loc	1 298 56                        // sk05_mlp_gateup.py:298:56
	mul.f32 	%r402, %r285, %r401;
	mul.f32 	%r403, %r284, %r400;
	mul.f32 	%r404, %r283, %r399;
	mul.f32 	%r405, %r282, %r398;
	.loc	1 299 36                        // sk05_mlp_gateup.py:299:36
	abs.f32 	%r406, %r405;
	abs.f32 	%r407, %r404;
	abs.f32 	%r408, %r403;
	abs.f32 	%r409, %r402;
	.loc	1 297 55                        // sk05_mlp_gateup.py:297:55
	mov.b32 	{%rs89, %rs90}, %r44;
	cvt.f32.bf16 	%r410, %rs89;
	cvt.f32.bf16 	%r411, %rs90;
	mov.b32 	{%rs91, %rs92}, %r45;
	cvt.f32.bf16 	%r412, %rs91;
	cvt.f32.bf16 	%r413, %rs92;
	.loc	1 298 56                        // sk05_mlp_gateup.py:298:56
	mul.f32 	%r414, %r281, %r413;
	mul.f32 	%r415, %r280, %r412;
	mul.f32 	%r416, %r279, %r411;
	mul.f32 	%r417, %r278, %r410;
	.loc	1 299 36                        // sk05_mlp_gateup.py:299:36
	abs.f32 	%r418, %r417;
	abs.f32 	%r419, %r416;
	abs.f32 	%r420, %r415;
	abs.f32 	%r421, %r414;
	.loc	1 297 55                        // sk05_mlp_gateup.py:297:55
	mov.b32 	{%rs93, %rs94}, %r42;
	cvt.f32.bf16 	%r422, %rs93;
	cvt.f32.bf16 	%r423, %rs94;
	mov.b32 	{%rs95, %rs96}, %r43;
	cvt.f32.bf16 	%r424, %rs95;
	cvt.f32.bf16 	%r425, %rs96;
	.loc	1 298 56                        // sk05_mlp_gateup.py:298:56
	mul.f32 	%r426, %r277, %r425;
	mul.f32 	%r427, %r276, %r424;
	mul.f32 	%r428, %r275, %r423;
	mul.f32 	%r429, %r274, %r422;
	.loc	1 299 36                        // sk05_mlp_gateup.py:299:36
	abs.f32 	%r430, %r429;
	abs.f32 	%r431, %r428;
	abs.f32 	%r432, %r427;
	abs.f32 	%r433, %r426;
$L__tmp24:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r434, %r385, %r430;
	max.f32 	%r435, %r434, %r431;
	max.f32 	%r436, %r435, %r432;
	max.f32 	%r437, %r436, %r433;
	max.f32 	%r438, %r437, %r418;
	max.f32 	%r439, %r438, %r419;
	max.f32 	%r440, %r439, %r420;
	max.f32 	%r441, %r440, %r421;
	max.f32 	%r442, %r441, %r406;
	max.f32 	%r443, %r442, %r407;
	max.f32 	%r444, %r443, %r408;
	max.f32 	%r445, %r444, %r409;
	max.f32 	%r446, %r445, %r394;
	max.f32 	%r447, %r446, %r395;
	max.f32 	%r448, %r447, %r396;
	max.f32 	%r449, %r448, %r397;
$L__tmp25:
	.loc	1 297 55                        // sk05_mlp_gateup.py:297:55
	mov.b32 	{%rs97, %rs98}, %r56;
	cvt.f32.bf16 	%r450, %rs97;
	cvt.f32.bf16 	%r451, %rs98;
	mov.b32 	{%rs99, %rs100}, %r57;
	cvt.f32.bf16 	%r452, %rs99;
	cvt.f32.bf16 	%r453, %rs100;
	.loc	1 298 56                        // sk05_mlp_gateup.py:298:56
	mul.f32 	%r454, %r305, %r453;
	mul.f32 	%r455, %r304, %r452;
	mul.f32 	%r456, %r303, %r451;
	mul.f32 	%r457, %r302, %r450;
	.loc	1 299 36                        // sk05_mlp_gateup.py:299:36
	abs.f32 	%r458, %r457;
	abs.f32 	%r459, %r456;
	abs.f32 	%r460, %r455;
	abs.f32 	%r461, %r454;
	.loc	1 297 55                        // sk05_mlp_gateup.py:297:55
	mov.b32 	{%rs101, %rs102}, %r54;
	cvt.f32.bf16 	%r462, %rs101;
	cvt.f32.bf16 	%r463, %rs102;
	mov.b32 	{%rs103, %rs104}, %r55;
	cvt.f32.bf16 	%r464, %rs103;
	cvt.f32.bf16 	%r465, %rs104;
	.loc	1 298 56                        // sk05_mlp_gateup.py:298:56
	mul.f32 	%r466, %r301, %r465;
	mul.f32 	%r467, %r300, %r464;
	mul.f32 	%r468, %r299, %r463;
	mul.f32 	%r469, %r298, %r462;
	.loc	1 299 36                        // sk05_mlp_gateup.py:299:36
	abs.f32 	%r470, %r469;
	abs.f32 	%r471, %r468;
	abs.f32 	%r472, %r467;
	abs.f32 	%r473, %r466;
	.loc	1 297 55                        // sk05_mlp_gateup.py:297:55
	mov.b32 	{%rs105, %rs106}, %r52;
	cvt.f32.bf16 	%r474, %rs105;
	cvt.f32.bf16 	%r475, %rs106;
	mov.b32 	{%rs107, %rs108}, %r53;
	cvt.f32.bf16 	%r476, %rs107;
	cvt.f32.bf16 	%r477, %rs108;
	.loc	1 298 56                        // sk05_mlp_gateup.py:298:56
	mul.f32 	%r478, %r297, %r477;
	mul.f32 	%r479, %r296, %r476;
	mul.f32 	%r480, %r295, %r475;
	mul.f32 	%r481, %r294, %r474;
	.loc	1 299 36                        // sk05_mlp_gateup.py:299:36
	abs.f32 	%r482, %r481;
	abs.f32 	%r483, %r480;
	abs.f32 	%r484, %r479;
	abs.f32 	%r485, %r478;
	.loc	1 297 55                        // sk05_mlp_gateup.py:297:55
	mov.b32 	{%rs109, %rs110}, %r50;
	cvt.f32.bf16 	%r486, %rs109;
	cvt.f32.bf16 	%r487, %rs110;
	mov.b32 	{%rs111, %rs112}, %r51;
	cvt.f32.bf16 	%r488, %rs111;
	cvt.f32.bf16 	%r489, %rs112;
	.loc	1 298 56                        // sk05_mlp_gateup.py:298:56
	mul.f32 	%r490, %r293, %r489;
	mul.f32 	%r491, %r292, %r488;
	mul.f32 	%r492, %r291, %r487;
	mul.f32 	%r493, %r290, %r486;
	.loc	1 299 36                        // sk05_mlp_gateup.py:299:36
	abs.f32 	%r494, %r493;
	abs.f32 	%r495, %r492;
	abs.f32 	%r496, %r491;
	abs.f32 	%r497, %r490;
$L__tmp26:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r498, %r449, %r494;
	max.f32 	%r499, %r498, %r495;
	max.f32 	%r500, %r499, %r496;
	max.f32 	%r501, %r500, %r497;
	max.f32 	%r502, %r501, %r482;
	max.f32 	%r503, %r502, %r483;
	max.f32 	%r504, %r503, %r484;
	max.f32 	%r505, %r504, %r485;
	max.f32 	%r506, %r505, %r470;
	max.f32 	%r507, %r506, %r471;
	max.f32 	%r508, %r507, %r472;
	max.f32 	%r509, %r508, %r473;
	max.f32 	%r510, %r509, %r458;
	max.f32 	%r511, %r510, %r459;
	max.f32 	%r512, %r511, %r460;
	max.f32 	%r513, %r512, %r461;
$L__tmp27:
	.loc	1 297 55                        // sk05_mlp_gateup.py:297:55
	mov.b32 	{%rs113, %rs114}, %r64;
	cvt.f32.bf16 	%r514, %rs113;
	cvt.f32.bf16 	%r515, %rs114;
	mov.b32 	{%rs115, %rs116}, %r65;
	cvt.f32.bf16 	%r516, %rs115;
	cvt.f32.bf16 	%r517, %rs116;
	.loc	1 298 56                        // sk05_mlp_gateup.py:298:56
	mul.f32 	%r518, %r321, %r517;
	mul.f32 	%r519, %r320, %r516;
	mul.f32 	%r520, %r319, %r515;
	mul.f32 	%r521, %r318, %r514;
	.loc	1 299 36                        // sk05_mlp_gateup.py:299:36
	abs.f32 	%r522, %r521;
	abs.f32 	%r523, %r520;
	abs.f32 	%r524, %r519;
	abs.f32 	%r525, %r518;
	.loc	1 297 55                        // sk05_mlp_gateup.py:297:55
	mov.b32 	{%rs117, %rs118}, %r62;
	cvt.f32.bf16 	%r526, %rs117;
	cvt.f32.bf16 	%r527, %rs118;
	mov.b32 	{%rs119, %rs120}, %r63;
	cvt.f32.bf16 	%r528, %rs119;
	cvt.f32.bf16 	%r529, %rs120;
	.loc	1 298 56                        // sk05_mlp_gateup.py:298:56
	mul.f32 	%r530, %r317, %r529;
	mul.f32 	%r531, %r316, %r528;
	mul.f32 	%r532, %r315, %r527;
	mul.f32 	%r533, %r314, %r526;
	.loc	1 299 36                        // sk05_mlp_gateup.py:299:36
	abs.f32 	%r534, %r533;
	abs.f32 	%r535, %r532;
	abs.f32 	%r536, %r531;
	abs.f32 	%r537, %r530;
	.loc	1 297 55                        // sk05_mlp_gateup.py:297:55
	mov.b32 	{%rs121, %rs122}, %r60;
	cvt.f32.bf16 	%r538, %rs121;
	cvt.f32.bf16 	%r539, %rs122;
	mov.b32 	{%rs123, %rs124}, %r61;
	cvt.f32.bf16 	%r540, %rs123;
	cvt.f32.bf16 	%r541, %rs124;
	.loc	1 298 56                        // sk05_mlp_gateup.py:298:56
	mul.f32 	%r542, %r313, %r541;
	mul.f32 	%r543, %r312, %r540;
	mul.f32 	%r544, %r311, %r539;
	mul.f32 	%r545, %r310, %r538;
	.loc	1 299 36                        // sk05_mlp_gateup.py:299:36
	abs.f32 	%r546, %r545;
	abs.f32 	%r547, %r544;
	abs.f32 	%r548, %r543;
	abs.f32 	%r549, %r542;
	.loc	1 297 55                        // sk05_mlp_gateup.py:297:55
	mov.b32 	{%rs125, %rs126}, %r58;
	cvt.f32.bf16 	%r550, %rs125;
	cvt.f32.bf16 	%r551, %rs126;
	mov.b32 	{%rs127, %rs128}, %r59;
	cvt.f32.bf16 	%r552, %rs127;
	cvt.f32.bf16 	%r553, %rs128;
	.loc	1 298 56                        // sk05_mlp_gateup.py:298:56
	mul.f32 	%r554, %r309, %r553;
	mul.f32 	%r555, %r308, %r552;
	mul.f32 	%r556, %r307, %r551;
	mul.f32 	%r557, %r306, %r550;
	.loc	1 299 36                        // sk05_mlp_gateup.py:299:36
	abs.f32 	%r558, %r557;
	abs.f32 	%r559, %r556;
	abs.f32 	%r560, %r555;
	abs.f32 	%r561, %r554;
$L__tmp28:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r562, %r513, %r558;
	max.f32 	%r563, %r562, %r559;
	max.f32 	%r564, %r563, %r560;
	max.f32 	%r565, %r564, %r561;
	max.f32 	%r566, %r565, %r546;
	max.f32 	%r567, %r566, %r547;
	max.f32 	%r568, %r567, %r548;
	max.f32 	%r569, %r568, %r549;
	max.f32 	%r570, %r569, %r534;
	max.f32 	%r571, %r570, %r535;
	max.f32 	%r572, %r571, %r536;
	max.f32 	%r573, %r572, %r537;
	max.f32 	%r574, %r573, %r522;
	max.f32 	%r575, %r574, %r523;
	max.f32 	%r576, %r575, %r524;
	max.f32 	%r577, %r576, %r525;
$L__tmp29:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	shfl.sync.bfly.b32 	%r578, %r577, 16, 31, -1;
$L__tmp30:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r579, %r577, %r578;
$L__tmp31:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	shfl.sync.bfly.b32 	%r580, %r579, 8, 31, -1;
$L__tmp32:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r581, %r579, %r580;
$L__tmp33:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	shfl.sync.bfly.b32 	%r582, %r581, 4, 31, -1;
$L__tmp34:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r583, %r581, %r582;
$L__tmp35:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	shfl.sync.bfly.b32 	%r584, %r583, 2, 31, -1;
$L__tmp36:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r585, %r583, %r584;
$L__tmp37:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	shfl.sync.bfly.b32 	%r586, %r585, 1, 31, -1;
$L__tmp38:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r71, %r585, %r586;
$L__tmp39:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	// begin inline asm
	@%p5 st.shared.b32 [ %r66 + 0 ], %r71;
	// end inline asm
	bar.sync 	0;
	// begin inline asm
	@%p6 ld.shared.b32 %r72, [ %r69 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r587, %r72, 4, 31, -1;
$L__tmp40:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r588, %r72, %r587;
$L__tmp41:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	shfl.sync.bfly.b32 	%r589, %r588, 2, 31, -1;
$L__tmp42:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r590, %r588, %r589;
$L__tmp43:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	shfl.sync.bfly.b32 	%r591, %r590, 1, 31, -1;
$L__tmp44:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ] ]
	max.f32 	%r73, %r590, %r591;
$L__tmp45:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:299:29 ]
	// begin inline asm
	@%p7 st.shared.b32 [ %r69 + 0 ], %r73;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r592, [global_smem];
$L__tmp46:
	.loc	1 299 49                        // sk05_mlp_gateup.py:299:49
	max.f32 	%r593, %r592, 0f0DA24260;
	mov.b32 	%r594, 0f42FE0000;
	.loc	1 300 22                        // sk05_mlp_gateup.py:300:22
	div.full.f32 	%r595, %r594, %r593;
	.loc	1 300 14                        // sk05_mlp_gateup.py:300:14
	mul.f32 	%r596, %r365, %r595;
	mul.f32 	%r597, %r366, %r595;
	mul.f32 	%r598, %r363, %r595;
	mul.f32 	%r599, %r364, %r595;
	mul.f32 	%r600, %r353, %r595;
	mul.f32 	%r601, %r354, %r595;
	mul.f32 	%r602, %r351, %r595;
	mul.f32 	%r603, %r352, %r595;
	mul.f32 	%r604, %r341, %r595;
	mul.f32 	%r605, %r342, %r595;
	mul.f32 	%r606, %r339, %r595;
	mul.f32 	%r607, %r340, %r595;
	mul.f32 	%r608, %r329, %r595;
	mul.f32 	%r609, %r330, %r595;
	mul.f32 	%r610, %r327, %r595;
	mul.f32 	%r611, %r328, %r595;
	mul.f32 	%r612, %r428, %r595;
	mul.f32 	%r613, %r429, %r595;
	mul.f32 	%r614, %r426, %r595;
	mul.f32 	%r615, %r427, %r595;
	mul.f32 	%r616, %r416, %r595;
	mul.f32 	%r617, %r417, %r595;
	mul.f32 	%r618, %r414, %r595;
	mul.f32 	%r619, %r415, %r595;
	mul.f32 	%r620, %r404, %r595;
	mul.f32 	%r621, %r405, %r595;
	mul.f32 	%r622, %r402, %r595;
	mul.f32 	%r623, %r403, %r595;
	mul.f32 	%r624, %r392, %r595;
	mul.f32 	%r625, %r393, %r595;
	mul.f32 	%r626, %r390, %r595;
	mul.f32 	%r627, %r391, %r595;
	mul.f32 	%r628, %r492, %r595;
	mul.f32 	%r629, %r493, %r595;
	mul.f32 	%r630, %r490, %r595;
	mul.f32 	%r631, %r491, %r595;
	mul.f32 	%r632, %r480, %r595;
	mul.f32 	%r633, %r481, %r595;
	mul.f32 	%r634, %r478, %r595;
	mul.f32 	%r635, %r479, %r595;
	mul.f32 	%r636, %r468, %r595;
	mul.f32 	%r637, %r469, %r595;
	mul.f32 	%r638, %r466, %r595;
	mul.f32 	%r639, %r467, %r595;
	mul.f32 	%r640, %r456, %r595;
	mul.f32 	%r641, %r457, %r595;
	mul.f32 	%r642, %r454, %r595;
	mul.f32 	%r643, %r455, %r595;
	mul.f32 	%r644, %r556, %r595;
	mul.f32 	%r645, %r557, %r595;
	mul.f32 	%r646, %r554, %r595;
	mul.f32 	%r647, %r555, %r595;
	mul.f32 	%r648, %r544, %r595;
	mul.f32 	%r649, %r545, %r595;
	mul.f32 	%r650, %r542, %r595;
	mul.f32 	%r651, %r543, %r595;
	mul.f32 	%r652, %r532, %r595;
	mul.f32 	%r653, %r533, %r595;
	mul.f32 	%r654, %r530, %r595;
	mul.f32 	%r655, %r531, %r595;
	mul.f32 	%r656, %r520, %r595;
	mul.f32 	%r657, %r521, %r595;
	mul.f32 	%r658, %r518, %r595;
	mul.f32 	%r659, %r519, %r595;
	.loc	1 301 29                        // sk05_mlp_gateup.py:301:29
	.loc	1 301 39                        // sk05_mlp_gateup.py:301:39
	lop3.b32 	%r660, 0x3f000000, %r596, 0x80000000, 0xF8;
	lop3.b32 	%r661, 0x3f000000, %r597, 0x80000000, 0xF8;
	lop3.b32 	%r662, 0x3f000000, %r598, 0x80000000, 0xF8;
	lop3.b32 	%r663, 0x3f000000, %r599, 0x80000000, 0xF8;
	lop3.b32 	%r664, 0x3f000000, %r600, 0x80000000, 0xF8;
	lop3.b32 	%r665, 0x3f000000, %r601, 0x80000000, 0xF8;
	lop3.b32 	%r666, 0x3f000000, %r602, 0x80000000, 0xF8;
	lop3.b32 	%r667, 0x3f000000, %r603, 0x80000000, 0xF8;
	lop3.b32 	%r668, 0x3f000000, %r604, 0x80000000, 0xF8;
	lop3.b32 	%r669, 0x3f000000, %r605, 0x80000000, 0xF8;
	lop3.b32 	%r670, 0x3f000000, %r606, 0x80000000, 0xF8;
	lop3.b32 	%r671, 0x3f000000, %r607, 0x80000000, 0xF8;
	lop3.b32 	%r672, 0x3f000000, %r608, 0x80000000, 0xF8;
	lop3.b32 	%r673, 0x3f000000, %r609, 0x80000000, 0xF8;
	lop3.b32 	%r674, 0x3f000000, %r610, 0x80000000, 0xF8;
	lop3.b32 	%r675, 0x3f000000, %r611, 0x80000000, 0xF8;
	lop3.b32 	%r676, 0x3f000000, %r612, 0x80000000, 0xF8;
	lop3.b32 	%r677, 0x3f000000, %r613, 0x80000000, 0xF8;
	lop3.b32 	%r678, 0x3f000000, %r614, 0x80000000, 0xF8;
	lop3.b32 	%r679, 0x3f000000, %r615, 0x80000000, 0xF8;
	lop3.b32 	%r680, 0x3f000000, %r616, 0x80000000, 0xF8;
	lop3.b32 	%r681, 0x3f000000, %r617, 0x80000000, 0xF8;
	lop3.b32 	%r682, 0x3f000000, %r618, 0x80000000, 0xF8;
	lop3.b32 	%r683, 0x3f000000, %r619, 0x80000000, 0xF8;
	lop3.b32 	%r684, 0x3f000000, %r620, 0x80000000, 0xF8;
	lop3.b32 	%r685, 0x3f000000, %r621, 0x80000000, 0xF8;
	lop3.b32 	%r686, 0x3f000000, %r622, 0x80000000, 0xF8;
	lop3.b32 	%r687, 0x3f000000, %r623, 0x80000000, 0xF8;
	lop3.b32 	%r688, 0x3f000000, %r624, 0x80000000, 0xF8;
	lop3.b32 	%r689, 0x3f000000, %r625, 0x80000000, 0xF8;
	lop3.b32 	%r690, 0x3f000000, %r626, 0x80000000, 0xF8;
	lop3.b32 	%r691, 0x3f000000, %r627, 0x80000000, 0xF8;
	lop3.b32 	%r692, 0x3f000000, %r628, 0x80000000, 0xF8;
	lop3.b32 	%r693, 0x3f000000, %r629, 0x80000000, 0xF8;
	lop3.b32 	%r694, 0x3f000000, %r630, 0x80000000, 0xF8;
	lop3.b32 	%r695, 0x3f000000, %r631, 0x80000000, 0xF8;
	lop3.b32 	%r696, 0x3f000000, %r632, 0x80000000, 0xF8;
	lop3.b32 	%r697, 0x3f000000, %r633, 0x80000000, 0xF8;
	lop3.b32 	%r698, 0x3f000000, %r634, 0x80000000, 0xF8;
	lop3.b32 	%r699, 0x3f000000, %r635, 0x80000000, 0xF8;
	lop3.b32 	%r700, 0x3f000000, %r636, 0x80000000, 0xF8;
	lop3.b32 	%r701, 0x3f000000, %r637, 0x80000000, 0xF8;
	lop3.b32 	%r702, 0x3f000000, %r638, 0x80000000, 0xF8;
	lop3.b32 	%r703, 0x3f000000, %r639, 0x80000000, 0xF8;
	lop3.b32 	%r704, 0x3f000000, %r640, 0x80000000, 0xF8;
	lop3.b32 	%r705, 0x3f000000, %r641, 0x80000000, 0xF8;
	lop3.b32 	%r706, 0x3f000000, %r642, 0x80000000, 0xF8;
	lop3.b32 	%r707, 0x3f000000, %r643, 0x80000000, 0xF8;
	lop3.b32 	%r708, 0x3f000000, %r644, 0x80000000, 0xF8;
	lop3.b32 	%r709, 0x3f000000, %r645, 0x80000000, 0xF8;
	lop3.b32 	%r710, 0x3f000000, %r646, 0x80000000, 0xF8;
	lop3.b32 	%r711, 0x3f000000, %r647, 0x80000000, 0xF8;
	lop3.b32 	%r712, 0x3f000000, %r648, 0x80000000, 0xF8;
	lop3.b32 	%r713, 0x3f000000, %r649, 0x80000000, 0xF8;
	lop3.b32 	%r714, 0x3f000000, %r650, 0x80000000, 0xF8;
	lop3.b32 	%r715, 0x3f000000, %r651, 0x80000000, 0xF8;
	lop3.b32 	%r716, 0x3f000000, %r652, 0x80000000, 0xF8;
	lop3.b32 	%r717, 0x3f000000, %r653, 0x80000000, 0xF8;
	lop3.b32 	%r718, 0x3f000000, %r654, 0x80000000, 0xF8;
	lop3.b32 	%r719, 0x3f000000, %r655, 0x80000000, 0xF8;
	lop3.b32 	%r720, 0x3f000000, %r656, 0x80000000, 0xF8;
	lop3.b32 	%r721, 0x3f000000, %r657, 0x80000000, 0xF8;
	lop3.b32 	%r722, 0x3f000000, %r658, 0x80000000, 0xF8;
	lop3.b32 	%r723, 0x3f000000, %r659, 0x80000000, 0xF8;
	.loc	1 301 14                        // sk05_mlp_gateup.py:301:14
	fma.rn.f32 	%r724, %r364, %r595, %r663;
	fma.rn.f32 	%r725, %r363, %r595, %r662;
	fma.rn.f32 	%r726, %r366, %r595, %r661;
	fma.rn.f32 	%r727, %r365, %r595, %r660;
	fma.rn.f32 	%r728, %r352, %r595, %r667;
	fma.rn.f32 	%r729, %r351, %r595, %r666;
	fma.rn.f32 	%r730, %r354, %r595, %r665;
	fma.rn.f32 	%r731, %r353, %r595, %r664;
	fma.rn.f32 	%r732, %r340, %r595, %r671;
	fma.rn.f32 	%r733, %r339, %r595, %r670;
	fma.rn.f32 	%r734, %r342, %r595, %r669;
	fma.rn.f32 	%r735, %r341, %r595, %r668;
	fma.rn.f32 	%r736, %r328, %r595, %r675;
	fma.rn.f32 	%r737, %r327, %r595, %r674;
	fma.rn.f32 	%r738, %r330, %r595, %r673;
	fma.rn.f32 	%r739, %r329, %r595, %r672;
	fma.rn.f32 	%r740, %r427, %r595, %r679;
	fma.rn.f32 	%r741, %r426, %r595, %r678;
	fma.rn.f32 	%r742, %r429, %r595, %r677;
	fma.rn.f32 	%r743, %r428, %r595, %r676;
	fma.rn.f32 	%r744, %r415, %r595, %r683;
	fma.rn.f32 	%r745, %r414, %r595, %r682;
	fma.rn.f32 	%r746, %r417, %r595, %r681;
	fma.rn.f32 	%r747, %r416, %r595, %r680;
	fma.rn.f32 	%r748, %r403, %r595, %r687;
	fma.rn.f32 	%r749, %r402, %r595, %r686;
	fma.rn.f32 	%r750, %r405, %r595, %r685;
	fma.rn.f32 	%r751, %r404, %r595, %r684;
	fma.rn.f32 	%r752, %r391, %r595, %r691;
	fma.rn.f32 	%r753, %r390, %r595, %r690;
	fma.rn.f32 	%r754, %r393, %r595, %r689;
	fma.rn.f32 	%r755, %r392, %r595, %r688;
	fma.rn.f32 	%r756, %r491, %r595, %r695;
	fma.rn.f32 	%r757, %r490, %r595, %r694;
	fma.rn.f32 	%r758, %r493, %r595, %r693;
	fma.rn.f32 	%r759, %r492, %r595, %r692;
	fma.rn.f32 	%r760, %r479, %r595, %r699;
	fma.rn.f32 	%r761, %r478, %r595, %r698;
	fma.rn.f32 	%r762, %r481, %r595, %r697;
	fma.rn.f32 	%r763, %r480, %r595, %r696;
	fma.rn.f32 	%r764, %r467, %r595, %r703;
	fma.rn.f32 	%r765, %r466, %r595, %r702;
	fma.rn.f32 	%r766, %r469, %r595, %r701;
	fma.rn.f32 	%r767, %r468, %r595, %r700;
	fma.rn.f32 	%r768, %r455, %r595, %r707;
	fma.rn.f32 	%r769, %r454, %r595, %r706;
	fma.rn.f32 	%r770, %r457, %r595, %r705;
	fma.rn.f32 	%r771, %r456, %r595, %r704;
	fma.rn.f32 	%r772, %r555, %r595, %r711;
	fma.rn.f32 	%r773, %r554, %r595, %r710;
	fma.rn.f32 	%r774, %r557, %r595, %r709;
	fma.rn.f32 	%r775, %r556, %r595, %r708;
	fma.rn.f32 	%r776, %r543, %r595, %r715;
	fma.rn.f32 	%r777, %r542, %r595, %r714;
	fma.rn.f32 	%r778, %r545, %r595, %r713;
	fma.rn.f32 	%r779, %r544, %r595, %r712;
	fma.rn.f32 	%r780, %r531, %r595, %r719;
	fma.rn.f32 	%r781, %r530, %r595, %r718;
	fma.rn.f32 	%r782, %r533, %r595, %r717;
	fma.rn.f32 	%r783, %r532, %r595, %r716;
	fma.rn.f32 	%r784, %r519, %r595, %r723;
	fma.rn.f32 	%r785, %r518, %r595, %r722;
	fma.rn.f32 	%r786, %r521, %r595, %r721;
	fma.rn.f32 	%r787, %r520, %r595, %r720;
	.loc	1 301 49                        // sk05_mlp_gateup.py:301:49
	cvt.rzi.s32.f32 	%r788, %r727;
	cvt.rzi.s32.f32 	%r789, %r726;
	cvt.rzi.s32.f32 	%r790, %r725;
	cvt.rzi.s32.f32 	%r791, %r724;
	cvt.rzi.s32.f32 	%r792, %r731;
	cvt.rzi.s32.f32 	%r793, %r730;
	cvt.rzi.s32.f32 	%r794, %r729;
	cvt.rzi.s32.f32 	%r795, %r728;
	cvt.rzi.s32.f32 	%r796, %r735;
	cvt.rzi.s32.f32 	%r797, %r734;
	cvt.rzi.s32.f32 	%r798, %r733;
	cvt.rzi.s32.f32 	%r799, %r732;
	cvt.rzi.s32.f32 	%r800, %r739;
	cvt.rzi.s32.f32 	%r801, %r738;
	cvt.rzi.s32.f32 	%r802, %r737;
	cvt.rzi.s32.f32 	%r803, %r736;
	cvt.rzi.s32.f32 	%r804, %r743;
	cvt.rzi.s32.f32 	%r805, %r742;
	cvt.rzi.s32.f32 	%r806, %r741;
	cvt.rzi.s32.f32 	%r807, %r740;
	cvt.rzi.s32.f32 	%r808, %r747;
	cvt.rzi.s32.f32 	%r809, %r746;
	cvt.rzi.s32.f32 	%r810, %r745;
	cvt.rzi.s32.f32 	%r811, %r744;
	cvt.rzi.s32.f32 	%r812, %r751;
	cvt.rzi.s32.f32 	%r813, %r750;
	cvt.rzi.s32.f32 	%r814, %r749;
	cvt.rzi.s32.f32 	%r815, %r748;
	cvt.rzi.s32.f32 	%r816, %r755;
	cvt.rzi.s32.f32 	%r817, %r754;
	cvt.rzi.s32.f32 	%r818, %r753;
	cvt.rzi.s32.f32 	%r819, %r752;
	cvt.rzi.s32.f32 	%r820, %r759;
	cvt.rzi.s32.f32 	%r821, %r758;
	cvt.rzi.s32.f32 	%r822, %r757;
	cvt.rzi.s32.f32 	%r823, %r756;
	cvt.rzi.s32.f32 	%r824, %r763;
	cvt.rzi.s32.f32 	%r825, %r762;
	cvt.rzi.s32.f32 	%r826, %r761;
	cvt.rzi.s32.f32 	%r827, %r760;
	cvt.rzi.s32.f32 	%r828, %r767;
	cvt.rzi.s32.f32 	%r829, %r766;
	cvt.rzi.s32.f32 	%r830, %r765;
	cvt.rzi.s32.f32 	%r831, %r764;
	cvt.rzi.s32.f32 	%r832, %r771;
	cvt.rzi.s32.f32 	%r833, %r770;
	cvt.rzi.s32.f32 	%r834, %r769;
	cvt.rzi.s32.f32 	%r835, %r768;
	cvt.rzi.s32.f32 	%r836, %r775;
	cvt.rzi.s32.f32 	%r837, %r774;
	cvt.rzi.s32.f32 	%r838, %r773;
	cvt.rzi.s32.f32 	%r839, %r772;
	cvt.rzi.s32.f32 	%r840, %r779;
	cvt.rzi.s32.f32 	%r841, %r778;
	cvt.rzi.s32.f32 	%r842, %r777;
	cvt.rzi.s32.f32 	%r843, %r776;
	cvt.rzi.s32.f32 	%r844, %r783;
	cvt.rzi.s32.f32 	%r845, %r782;
	cvt.rzi.s32.f32 	%r846, %r781;
	cvt.rzi.s32.f32 	%r847, %r780;
	cvt.rzi.s32.f32 	%r848, %r787;
	cvt.rzi.s32.f32 	%r849, %r786;
	cvt.rzi.s32.f32 	%r850, %r785;
	cvt.rzi.s32.f32 	%r851, %r784;
	.loc	1 302 70                        // sk05_mlp_gateup.py:302:70
	max.s32 	%r852, %r791, -127;
	max.s32 	%r853, %r790, -127;
	max.s32 	%r854, %r789, -127;
	max.s32 	%r855, %r788, -127;
	max.s32 	%r856, %r795, -127;
	max.s32 	%r857, %r794, -127;
	max.s32 	%r858, %r793, -127;
	max.s32 	%r859, %r792, -127;
	max.s32 	%r860, %r799, -127;
	max.s32 	%r861, %r798, -127;
	max.s32 	%r862, %r797, -127;
	max.s32 	%r863, %r796, -127;
	max.s32 	%r864, %r803, -127;
	max.s32 	%r865, %r802, -127;
	max.s32 	%r866, %r801, -127;
	max.s32 	%r867, %r800, -127;
	max.s32 	%r868, %r807, -127;
	max.s32 	%r869, %r806, -127;
	max.s32 	%r870, %r805, -127;
	max.s32 	%r871, %r804, -127;
	max.s32 	%r872, %r811, -127;
	max.s32 	%r873, %r810, -127;
	max.s32 	%r874, %r809, -127;
	max.s32 	%r875, %r808, -127;
	max.s32 	%r876, %r815, -127;
	max.s32 	%r877, %r814, -127;
	max.s32 	%r878, %r813, -127;
	max.s32 	%r879, %r812, -127;
	max.s32 	%r880, %r819, -127;
	max.s32 	%r881, %r818, -127;
	max.s32 	%r882, %r817, -127;
	max.s32 	%r883, %r816, -127;
	max.s32 	%r884, %r823, -127;
	max.s32 	%r885, %r822, -127;
	max.s32 	%r886, %r821, -127;
	max.s32 	%r887, %r820, -127;
	max.s32 	%r888, %r827, -127;
	max.s32 	%r889, %r826, -127;
	max.s32 	%r890, %r825, -127;
	max.s32 	%r891, %r824, -127;
	max.s32 	%r892, %r831, -127;
	max.s32 	%r893, %r830, -127;
	max.s32 	%r894, %r829, -127;
	max.s32 	%r895, %r828, -127;
	max.s32 	%r896, %r835, -127;
	max.s32 	%r897, %r834, -127;
	max.s32 	%r898, %r833, -127;
	max.s32 	%r899, %r832, -127;
	max.s32 	%r900, %r839, -127;
	max.s32 	%r901, %r838, -127;
	max.s32 	%r902, %r837, -127;
	max.s32 	%r903, %r836, -127;
	max.s32 	%r904, %r843, -127;
	max.s32 	%r905, %r842, -127;
	max.s32 	%r906, %r841, -127;
	max.s32 	%r907, %r840, -127;
	max.s32 	%r908, %r847, -127;
	max.s32 	%r909, %r846, -127;
	max.s32 	%r910, %r845, -127;
	max.s32 	%r911, %r844, -127;
	max.s32 	%r912, %r851, -127;
	max.s32 	%r913, %r850, -127;
	max.s32 	%r914, %r849, -127;
	max.s32 	%r915, %r848, -127;
	.loc	1 302 77                        // sk05_mlp_gateup.py:302:77
	min.s32 	%r916, %r855, 127;
	min.s32 	%r917, %r854, 127;
	min.s32 	%r918, %r853, 127;
	min.s32 	%r919, %r852, 127;
	min.s32 	%r920, %r859, 127;
	min.s32 	%r921, %r858, 127;
	min.s32 	%r922, %r857, 127;
	min.s32 	%r923, %r856, 127;
	min.s32 	%r924, %r863, 127;
	min.s32 	%r925, %r862, 127;
	min.s32 	%r926, %r861, 127;
	min.s32 	%r927, %r860, 127;
	min.s32 	%r928, %r867, 127;
	min.s32 	%r929, %r866, 127;
	min.s32 	%r930, %r865, 127;
	min.s32 	%r931, %r864, 127;
	min.s32 	%r932, %r871, 127;
	min.s32 	%r933, %r870, 127;
	min.s32 	%r934, %r869, 127;
	min.s32 	%r935, %r868, 127;
	min.s32 	%r936, %r875, 127;
	min.s32 	%r937, %r874, 127;
	min.s32 	%r938, %r873, 127;
	min.s32 	%r939, %r872, 127;
	min.s32 	%r940, %r879, 127;
	min.s32 	%r941, %r878, 127;
	min.s32 	%r942, %r877, 127;
	min.s32 	%r943, %r876, 127;
	min.s32 	%r944, %r883, 127;
	min.s32 	%r945, %r882, 127;
	min.s32 	%r946, %r881, 127;
	min.s32 	%r947, %r880, 127;
	min.s32 	%r948, %r887, 127;
	min.s32 	%r949, %r886, 127;
	min.s32 	%r950, %r885, 127;
	min.s32 	%r951, %r884, 127;
	min.s32 	%r952, %r891, 127;
	min.s32 	%r953, %r890, 127;
	min.s32 	%r954, %r889, 127;
	min.s32 	%r955, %r888, 127;
	min.s32 	%r956, %r895, 127;
	min.s32 	%r957, %r894, 127;
	min.s32 	%r958, %r893, 127;
	min.s32 	%r959, %r892, 127;
	min.s32 	%r960, %r899, 127;
	min.s32 	%r961, %r898, 127;
	min.s32 	%r962, %r897, 127;
	min.s32 	%r963, %r896, 127;
	min.s32 	%r964, %r903, 127;
	min.s32 	%r965, %r902, 127;
	min.s32 	%r966, %r901, 127;
	min.s32 	%r967, %r900, 127;
	min.s32 	%r968, %r907, 127;
	min.s32 	%r969, %r906, 127;
	min.s32 	%r970, %r905, 127;
	min.s32 	%r971, %r904, 127;
	min.s32 	%r972, %r911, 127;
	min.s32 	%r973, %r910, 127;
	min.s32 	%r974, %r909, 127;
	min.s32 	%r975, %r908, 127;
	min.s32 	%r976, %r915, 127;
	min.s32 	%r977, %r914, 127;
	min.s32 	%r978, %r913, 127;
	min.s32 	%r979, %r912, 127;
	.loc	1 302 85                        // sk05_mlp_gateup.py:302:85
	prmt.b32 	%r980, %r919, %r918, 0x3340U;
	prmt.b32 	%r981, %r917, %r916, 0x3340U;
	prmt.b32 	%r74, %r981, %r980, 0x5410U;
	prmt.b32 	%r982, %r923, %r922, 0x3340U;
	prmt.b32 	%r983, %r921, %r920, 0x3340U;
	prmt.b32 	%r75, %r983, %r982, 0x5410U;
	prmt.b32 	%r984, %r927, %r926, 0x3340U;
	prmt.b32 	%r985, %r925, %r924, 0x3340U;
	prmt.b32 	%r76, %r985, %r984, 0x5410U;
	prmt.b32 	%r986, %r931, %r930, 0x3340U;
	prmt.b32 	%r987, %r929, %r928, 0x3340U;
	prmt.b32 	%r77, %r987, %r986, 0x5410U;
	prmt.b32 	%r988, %r935, %r934, 0x3340U;
	prmt.b32 	%r989, %r933, %r932, 0x3340U;
	prmt.b32 	%r78, %r989, %r988, 0x5410U;
	prmt.b32 	%r990, %r939, %r938, 0x3340U;
	prmt.b32 	%r991, %r937, %r936, 0x3340U;
	prmt.b32 	%r79, %r991, %r990, 0x5410U;
	prmt.b32 	%r992, %r943, %r942, 0x3340U;
	prmt.b32 	%r993, %r941, %r940, 0x3340U;
	prmt.b32 	%r80, %r993, %r992, 0x5410U;
	prmt.b32 	%r994, %r947, %r946, 0x3340U;
	prmt.b32 	%r995, %r945, %r944, 0x3340U;
	prmt.b32 	%r81, %r995, %r994, 0x5410U;
	prmt.b32 	%r996, %r951, %r950, 0x3340U;
	prmt.b32 	%r997, %r949, %r948, 0x3340U;
	prmt.b32 	%r82, %r997, %r996, 0x5410U;
	prmt.b32 	%r998, %r955, %r954, 0x3340U;
	prmt.b32 	%r999, %r953, %r952, 0x3340U;
	prmt.b32 	%r83, %r999, %r998, 0x5410U;
	prmt.b32 	%r1000, %r959, %r958, 0x3340U;
	prmt.b32 	%r1001, %r957, %r956, 0x3340U;
	prmt.b32 	%r84, %r1001, %r1000, 0x5410U;
	prmt.b32 	%r1002, %r963, %r962, 0x3340U;
	prmt.b32 	%r1003, %r961, %r960, 0x3340U;
	prmt.b32 	%r85, %r1003, %r1002, 0x5410U;
	prmt.b32 	%r1004, %r967, %r966, 0x3340U;
	prmt.b32 	%r1005, %r965, %r964, 0x3340U;
	prmt.b32 	%r86, %r1005, %r1004, 0x5410U;
	prmt.b32 	%r1006, %r971, %r970, 0x3340U;
	prmt.b32 	%r1007, %r969, %r968, 0x3340U;
	prmt.b32 	%r87, %r1007, %r1006, 0x5410U;
	prmt.b32 	%r1008, %r975, %r974, 0x3340U;
	prmt.b32 	%r1009, %r973, %r972, 0x3340U;
	prmt.b32 	%r88, %r1009, %r1008, 0x5410U;
	prmt.b32 	%r1010, %r979, %r978, 0x3340U;
	prmt.b32 	%r1011, %r977, %r976, 0x3340U;
	prmt.b32 	%r89, %r1011, %r1010, 0x5410U;
	.loc	1 302 45                        // sk05_mlp_gateup.py:302:45
	// begin inline asm
	@%p1 st.global.v4.b32 [ %rd17 + 0 ], { %r74, %r75, %r76, %r77 };
	// end inline asm
	// begin inline asm
	@%p2 st.global.v4.b32 [ %rd18 + 0 ], { %r78, %r79, %r80, %r81 };
	// end inline asm
	// begin inline asm
	@%p3 st.global.v4.b32 [ %rd19 + 0 ], { %r82, %r83, %r84, %r85 };
	// end inline asm
	// begin inline asm
	@%p4 st.global.v4.b32 [ %rd20 + 0 ], { %r86, %r87, %r88, %r89 };
	// end inline asm
	.loc	1 303 21                        // sk05_mlp_gateup.py:303:21
	mad.wide.u32 	%rd21, %r91, 4, %rd25;
	.loc	1 303 34                        // sk05_mlp_gateup.py:303:34
	mul.f32 	%r90, %r593, 0f3C010204;
	.loc	1 303 26                        // sk05_mlp_gateup.py:303:26
	or.b32 	%r1012, %r94, %r96;
	setp.eq.b32 	%p8, %r1012, 0;
	// begin inline asm
	@%p8 st.global.b32 [ %rd21 + 0 ], { %r90 };
	// end inline asm
	.loc	1 303 4                         // sk05_mlp_gateup.py:303:4
	ret;
$L__tmp47:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/usr/local/lib/python3.12/dist-packages/vllm/_genesis/kernels/sk05_mlp_gateup.py"
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
.b8 1                                   // DW_CHILDREN_yes
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
.b8 5                                   // Abbreviation Code
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
.b8 6                                   // Abbreviation Code
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
.b8 11                                  // DW_FORM_data1
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 255                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0xf8 DW_TAG_compile_unit
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
.b8 53
.b8 95
.b8 109
.b8 108
.b8 112
.b8 95
.b8 103
.b8 97
.b8 116
.b8 101
.b8 117
.b8 112
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line                        // DW_AT_stmt_list
.b8 47                                  // DW_AT_comp_dir
.b8 117
.b8 115
.b8 114
.b8 47
.b8 108
.b8 111
.b8 99
.b8 97
.b8 108
.b8 47
.b8 108
.b8 105
.b8 98
.b8 47
.b8 112
.b8 121
.b8 116
.b8 104
.b8 111
.b8 110
.b8 51
.b8 46
.b8 49
.b8 50
.b8 47
.b8 100
.b8 105
.b8 115
.b8 116
.b8 45
.b8 112
.b8 97
.b8 99
.b8 107
.b8 97
.b8 103
.b8 101
.b8 115
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
.b8 2                                   // Abbrev [2] 0x6a:0x1d DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 53
.b8 95
.b8 114
.b8 109
.b8 115
.b8 110
.b8 111
.b8 114
.b8 109
.b8 95
.b8 113
.b8 117
.b8 97
.b8 110
.b8 116
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x87:0x7b DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 106                                // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x9c:0x33 DW_TAG_inlined_subroutine
.b32 106                                // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp19                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 42                                  // DW_AT_call_line
.b8 1
.b8 28                                  // DW_AT_call_column
.b8 5                                   // Abbrev [5] 0xb5:0x19 DW_TAG_inlined_subroutine
.b32 106                                // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp18                          // DW_AT_high_pc
.b8 2                                   // DW_AT_call_file
.b8 37                                  // DW_AT_call_line
.b8 1
.b8 36                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 4                                   // Abbrev [4] 0xcf:0x32 DW_TAG_inlined_subroutine
.b32 106                                // DW_AT_abstract_origin
.b64 $L__tmp20                          // DW_AT_low_pc
.b64 $L__tmp46                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 43                                  // DW_AT_call_line
.b8 1
.b8 29                                  // DW_AT_call_column
.b8 6                                   // Abbrev [6] 0xe8:0x18 DW_TAG_inlined_subroutine
.b32 106                                // DW_AT_abstract_origin
.b64 $L__tmp22                          // DW_AT_low_pc
.b64 $L__tmp45                          // DW_AT_high_pc
.b8 2                                   // DW_AT_call_file
.b8 191                                 // DW_AT_call_line
.b8 40                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_QVAR0_3 = _Nativo(
    "sk05_mlp_gateup/_sk05_rmsnorm_quant_kernel/blk16384",
    _QPTX0_3, "_sk05_rmsnorm_quant_kernel",
    warps=8, shared=32,
    abi=[0, 1, 2, 3, 4, 5, 6],
    horneado={7: 16384, 8: 1e-06},
    div16=[4, 5, 6],
)
# E1=64 lop3, E2=0 mul

_POR_Q0 = [_QVAR0_0, _QVAR0_1, _QVAR0_2, _QVAR0_3]


def _q0_impl(grid: list[int], x_ptr: torch.Tensor, w_ptr: torch.Tensor, q_ptr: torch.Tensor, s_ptr: torch.Tensor, K: int, stride_xm: int, stride_qm: int, BLOCK: int, EPS: float) -> None:
    """Cuerpo del custom op: elige la variante y lanza el PTX."""
    args = (x_ptr, w_ptr, q_ptr, s_ptr, K, stride_xm, stride_qm, BLOCK, EPS)
    for _v in _POR_Q0:
        if all(args[_p] == _x for _p, _x in _v.horneado.items()):
            return _v(tuple(grid), *args)
    raise ValueError(
        "genesis_sk05_mlp_gateup_q0: no hay PTX embebido para estos constexpr; "
        "horneados disponibles: %r" % [_v.horneado for _v in _POR_Q0])


def _q0_fake(grid: list[int], x_ptr: torch.Tensor, w_ptr: torch.Tensor, q_ptr: torch.Tensor, s_ptr: torch.Tensor, K: int, stride_xm: int, stride_qm: int, BLOCK: int, EPS: float) -> None:
    """Meta impl: no toca la GPU; las salidas se mutan in-place."""
    return None


# Registro AL IMPORTAR, no en el primer uso: direct_register_custom_op
# llama a torch._library.infer_schema, que dynamo se niega a trazar
# ("Attempted to call function marked as skipped"). El hasattr evita el
# choque cuando el modulo se importa dos veces con nombres distintos,
# como hace el gate de tools/monolitizar.py.
if not hasattr(torch.ops.vllm, "genesis_sk05_mlp_gateup_q0"):
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op(
        op_name="genesis_sk05_mlp_gateup_q0",
        op_func=_q0_impl,
        mutates_args=['q_ptr', 's_ptr'],
        fake_impl=_q0_fake,
    )


def _lanzar_quant0(grid, *args):
    """Lanza el PTX embebido. Es el UNICO camino ejecutable.

    No hay fallback a Triton ni kill-switch: el kernel Triton de este
    archivo es privado y solo lo llaman los tests.
    """
    g = [grid] if isinstance(grid, int) else list(grid)
    return torch.ops.vllm.genesis_sk05_mlp_gateup_q0(g, *args)
