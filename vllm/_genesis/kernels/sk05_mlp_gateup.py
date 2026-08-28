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


def habilitado() -> bool:
    """Kill-switch: ``GENESIS_PTQ_NATIVO=0`` desactiva el camino PTX.

    Definido ACA, no importado: este modulo no comparte plomeria con nadie.
    """
    return os.environ.get("GENESIS_PTQ_NATIVO", "1").strip().lower() \
        not in ("0", "false", "no", "off")



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
        p2 = tl.exp2(tl.load(sh_ptrs + kb * stride_shift_k).to(tl.float32))
        acc += tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), out_dtype=tl.int32).to(tl.float32) * p2[None, :]
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
        sh_g = tl.load(shifts_ptr + kb * stride_shift_k + g_col * stride_shift_n).to(tl.float32)
        sh_u = tl.load(shifts_ptr + kb * stride_shift_k + u_col * stride_shift_n).to(tl.float32)
        p2 = tl.exp2(tl.where(is_up, sh_u, sh_g))
        acc += tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), out_dtype=tl.int32).to(tl.float32) * p2[None, :]
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
    M, K = a.shape
    N = b.shape[1]
    res = _zero(a.device) if residual is None else residual
    rm, rn = (0, 0) if residual is None else (res.stride(0), res.stride(1))
    bm, bn, bk, gm, warps, stages = _cfg(M, N)
    out = torch.empty((M, N), dtype=out_dtype, device=a.device)
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
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
            shifts.stride(0),
            shifts.stride(1),
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
	.reg .b16 	%rs<25>;
	.reg .b32 	%r<314>;
	.reg .b64 	%rd<72>;
	.loc	1 133 0                         // sk05_mlp_gateup.py:133:0
$L__func_begin0:
	.loc	1 133 0                         // sk05_mlp_gateup.py:133:0

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
	.loc	1 143 24                        // sk05_mlp_gateup.py:143:24
	mov.u32 	%r40, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk05_mlp_gateup.py:144:27 ]
	add.s32 	%r41, %r20, 15;
	.loc	2 43 30                         // standard.py:43:30 @[ sk05_mlp_gateup.py:144:27 ]
	shr.s32 	%r42, %r41, 31;
	shr.u32 	%r43, %r42, 28;
	add.s32 	%r44, %r41, %r43;
	shr.s32 	%r45, %r44, 4;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk05_mlp_gateup.py:145:27 ]
	add.s32 	%r46, %r21, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk05_mlp_gateup.py:145:27 ]
	shr.s32 	%r47, %r46, 31;
	shr.u32 	%r48, %r47, 25;
	add.s32 	%r49, %r46, %r48;
	shr.s32 	%r50, %r49, 7;
$L__tmp3:
	.loc	1 146 29                        // sk05_mlp_gateup.py:146:29
	shl.b32 	%r51, %r50, 3;
	.loc	1 147 22                        // sk05_mlp_gateup.py:147:22
	div.s32 	%r52, %r40, %r51;
	.loc	1 147 38                        // sk05_mlp_gateup.py:147:38
	shl.b32 	%r53, %r52, 3;
	.loc	1 148 30                        // sk05_mlp_gateup.py:148:30
	sub.s32 	%r54, %r45, %r53;
	ld.param.b32 	%r55, [_sk05_mlp_gateup_kernel_param_10];
	.loc	1 148 39                        // sk05_mlp_gateup.py:148:39
	min.s32 	%r56, %r54, 8;
	ld.param.b32 	%r57, [_sk05_mlp_gateup_kernel_param_11];
	.loc	1 149 30                        // sk05_mlp_gateup.py:149:30
	mul.lo.s32 	%r58, %r52, %r51;
	sub.s32 	%r59, %r40, %r58;
	.loc	1 150 36                        // sk05_mlp_gateup.py:150:36
	div.s32 	%r60, %r59, %r56;
	.loc	1 149 46                        // sk05_mlp_gateup.py:149:46
	mul.lo.s32 	%r61, %r60, %r56;
	sub.s32 	%r62, %r59, %r61;
	.loc	1 149 23                        // sk05_mlp_gateup.py:149:23
	add.s32 	%r63, %r62, %r53;
	.loc	1 152 22                        // sk05_mlp_gateup.py:152:22
	shl.b32 	%r1, %r63, 4;
	.loc	1 152 45                        // sk05_mlp_gateup.py:152:45
	mov.u32 	%r2, %tid.x;
	and.b32 	%r3, %r2, 240;
	bfe.u32 	%r64, %r2, 4, 4;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r4, %r1, %r64;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r5, %r4, %r20;
	.loc	1 153 22                        // sk05_mlp_gateup.py:153:22
	shl.b32 	%r6, %r60, 7;
	.loc	1 153 45                        // sk05_mlp_gateup.py:153:45
	shr.u32 	%r65, %r2, 3;
	bfe.u32 	%r66, %r2, 3, 5;
	shl.b32 	%r7, %r2, 1;
	and.b32 	%r67, %r7, 6;
	and.b32 	%r8, %r2, 224;
	shr.u32 	%r68, %r8, 2;
	or.b32 	%r69, %r67, %r68;
	and.b32 	%r9, %r2, 15;
	shl.b32 	%r70, %r9, 3;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r71, %r6, %r66;
	or.b32 	%r72, %r71, 32;
	or.b32 	%r73, %r71, 64;
	or.b32 	%r74, %r65, %r6;
	or.b32 	%r75, %r74, 96;
	or.b32 	%r76, %r6, %r69;
	or.b32 	%r78, %r76, 64;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r80, %r71, %r21;
	rem.s32 	%r81, %r72, %r21;
	rem.s32 	%r82, %r73, %r21;
	rem.s32 	%r83, %r75, %r21;
	rem.s32 	%r10, %r76, %r21;
	rem.s32 	%r11, %r78, %r21;
	.loc	1 156 39                        // sk05_mlp_gateup.py:156:39
	mul.lo.s32 	%r86, %r5, %r55;
	.loc	1 156 21                        // sk05_mlp_gateup.py:156:21
	cvt.s64.s32 	%rd1, %r86;
	add.s64 	%rd36, %rd19, %rd1;
	.loc	1 156 51                        // sk05_mlp_gateup.py:156:51
	cvt.u64.u32 	%rd2, %r70;
	add.s64 	%rd25, %rd36, %rd2;
	.loc	1 157 28                        // sk05_mlp_gateup.py:157:28
	and.b32 	%r12, %r2, 7;
	shl.b32 	%r87, %r12, 4;
	.loc	1 157 21                        // sk05_mlp_gateup.py:157:21
	cvt.u64.u32 	%rd3, %r87;
	add.s64 	%rd37, %rd20, %rd3;
	.loc	1 157 69                        // sk05_mlp_gateup.py:157:69
	mul.lo.s32 	%r88, %r80, %r57;
	mul.lo.s32 	%r89, %r81, %r57;
	mul.lo.s32 	%r90, %r82, %r57;
	mul.lo.s32 	%r91, %r83, %r57;
	.loc	1 157 51                        // sk05_mlp_gateup.py:157:51
	cvt.s64.s32 	%rd4, %r88;
	add.s64 	%rd26, %rd37, %rd4;
	cvt.s64.s32 	%rd5, %r89;
	add.s64 	%rd27, %rd37, %rd5;
	cvt.s64.s32 	%rd6, %r90;
	add.s64 	%rd28, %rd37, %rd6;
	cvt.s64.s32 	%rd7, %r91;
	add.s64 	%rd29, %rd37, %rd7;
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	setp.lt.s32 	%p1, %r22, 128;
	setp.gt.s32 	%p2, %r22, 127;
	.loc	1 163 30                        // sk05_mlp_gateup.py:163:30
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
	.loc	1 163 47                        // sk05_mlp_gateup.py:163:47
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
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	setp.gt.s32 	%p3, %r22, 255;
	.loc	1 164 18                        // sk05_mlp_gateup.py:164:18
	add.s64 	%rd30, %rd25, 128;
	.loc	1 165 18                        // sk05_mlp_gateup.py:165:18
	add.s64 	%rd31, %rd26, 128;
	add.s64 	%rd32, %rd27, 128;
	add.s64 	%rd33, %rd28, 128;
	add.s64 	%rd34, %rd29, 128;
	.loc	1 163 30                        // sk05_mlp_gateup.py:163:30
	bar.sync 	0;
	add.s32 	%r33, %r13, 34816;
	selp.b32 	%r34, 8, 0, %p3;
	// begin inline asm
	cp.async.ca.shared.global [ %r33 + 0 ], [ %rd30 + 0 ], 0x8, %r34;
	// end inline asm
	cp.async.commit_group;
	.loc	1 163 47                        // sk05_mlp_gateup.py:163:47
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
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
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
	cvt.s64.s32 	%rd38, %r95;
	add.s64 	%rd8, %rd35, %rd38;
	cvt.s64.s32 	%rd39, %r99;
	add.s64 	%rd9, %rd35, %rd39;
	cvt.s64.s32 	%rd40, %r103;
	add.s64 	%rd10, %rd35, %rd40;
	cvt.s64.s32 	%rd41, %r107;
	add.s64 	%rd11, %rd35, %rd41;
	.loc	1 161 28                        // sk05_mlp_gateup.py:161:28
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
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	cvt.s64.s32 	%rd12, %r117;
	and.b32 	%r126, %r22, -128;
	cvt.u64.u32 	%rd13, %r126;
	add.s64 	%rd42, %rd3, %rd7;
	add.s64 	%rd43, %rd42, %rd20;
	add.s64 	%rd14, %rd43, 256;
	add.s64 	%rd44, %rd3, %rd6;
	add.s64 	%rd45, %rd44, %rd20;
	add.s64 	%rd15, %rd45, 256;
	add.s64 	%rd46, %rd3, %rd5;
	add.s64 	%rd47, %rd46, %rd20;
	add.s64 	%rd16, %rd47, 256;
	add.s64 	%rd48, %rd3, %rd4;
	add.s64 	%rd49, %rd48, %rd20;
	add.s64 	%rd17, %rd49, 256;
	add.s64 	%rd50, %rd2, %rd1;
	add.s64 	%rd51, %rd50, %rd19;
	add.s64 	%rd18, %rd51, 256;
	mov.b32 	%r306, 0f00000000;
	mov.b32 	%r305, 1;
	mov.b32 	%r304, -1;
	mov.b64 	%rd70, 0;
	mov.b32 	%r127, 0;
	mov.b32 	%r303, %r127;
	mov.b64 	%rd71, %rd70;
	mov.b32 	%r307, %r306;
	mov.b32 	%r308, %r306;
	mov.b32 	%r309, %r306;
	mov.b32 	%r310, %r306;
	mov.b32 	%r311, %r306;
	mov.b32 	%r312, %r306;
	mov.b32 	%r313, %r306;
$L__BB0_2:                              // %__nv_exp2f.exit
                                        // =>This Inner Loop Header: Depth=1
	setp.lt.s64 	%p4, %rd71, %rd12;
	add.s32 	%r175, %r304, 1;
	setp.gt.s32 	%p5, %r175, 1;
	selp.b32 	%r304, 0, %r175, %p5;
	.loc	1 162 39                        // sk05_mlp_gateup.py:162:39
	cvt.s64.s32 	%rd61, %r303;
	add.s64 	%rd52, %rd8, %rd61;
	add.s64 	%rd53, %rd9, %rd61;
	add.s64 	%rd54, %rd10, %rd61;
	add.s64 	%rd55, %rd11, %rd61;
	.loc	1 162 29                        // sk05_mlp_gateup.py:162:29
	// begin inline asm
	mov.u16 %rs1, 0x0;
	ld.global.b8 { %rs1 }, [ %rd52 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs5, %rs1;
	// begin inline asm
	mov.u16 %rs2, 0x0;
	ld.global.b8 { %rs2 }, [ %rd53 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs6, %rs2;
	// begin inline asm
	mov.u16 %rs3, 0x0;
	ld.global.b8 { %rs3 }, [ %rd54 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs7, %rs3;
	// begin inline asm
	mov.u16 %rs4, 0x0;
	ld.global.b8 { %rs4 }, [ %rd55 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs8, %rs4;
	.loc	1 162 63                        // sk05_mlp_gateup.py:162:63
	cvt.rn.f32.s16 	%r176, %rs5;
	cvt.rn.f32.s16 	%r177, %rs6;
	cvt.rn.f32.s16 	%r178, %rs7;
	cvt.rn.f32.s16 	%r179, %rs8;
	.loc	1 162 21                        // sk05_mlp_gateup.py:162:21
	ex2.approx.ftz.f32 	%r180, %r176;
	ex2.approx.ftz.f32 	%r181, %r177;
	ex2.approx.ftz.f32 	%r182, %r178;
	ex2.approx.ftz.f32 	%r183, %r179;
	.loc	1 163 30                        // sk05_mlp_gateup.py:163:30
	cp.async.wait_group 	2;
	bar.sync 	0;
	shl.b32 	%r184, %r304, 11;
	add.s32 	%r185, %r112, %r184;
	add.s32 	%r186, %r185, %r14;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r128, %r129, %r130, %r131}, [%r186+32768];
	add.s32 	%r187, %r185, %r15;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r140, %r141, %r142, %r143}, [%r187+32768];
	add.s32 	%r188, %r185, %r16;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r152, %r153, %r154, %r155}, [%r188+32768];
	add.s32 	%r189, %r185, %r17;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r160, %r161, %r162, %r163}, [%r189+32768];
	.loc	1 163 47                        // sk05_mlp_gateup.py:163:47
	shl.b32 	%r190, %r304, 14;
	add.s32 	%r191, %r112, %r190;
	add.s32 	%r192, %r191, %r18;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r132, %r133, %r144, %r145}, [%r192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r134, %r135, %r150, %r151}, [%r192+8192];
	add.s32 	%r193, %r191, %r19;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r156, %r157, %r164, %r165}, [%r193];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r158, %r159, %r166, %r167}, [%r193+8192];
	.loc	1 163 39                        // sk05_mlp_gateup.py:163:39
	mov.b32 	%r136, %r127;
	mov.b32 	%r137, %r127;
	mov.b32 	%r138, %r127;
	mov.b32 	%r139, %r127;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r136, %r137, %r138, %r139 }, { %r128, %r129, %r130, %r131 }, { %r132, %r133 }, { %r136, %r137, %r138, %r139 };
	// end inline asm
	mov.b32 	%r146, %r127;
	mov.b32 	%r147, %r127;
	mov.b32 	%r148, %r127;
	mov.b32 	%r149, %r127;
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
	.loc	1 163 79                        // sk05_mlp_gateup.py:163:79
	cvt.rn.f32.s32 	%r194, %r146;
	cvt.rn.f32.s32 	%r195, %r147;
	cvt.rn.f32.s32 	%r196, %r148;
	cvt.rn.f32.s32 	%r197, %r149;
	cvt.rn.f32.s32 	%r198, %r136;
	cvt.rn.f32.s32 	%r199, %r137;
	cvt.rn.f32.s32 	%r200, %r138;
	cvt.rn.f32.s32 	%r201, %r139;
	.loc	1 163 15                        // sk05_mlp_gateup.py:163:15
	fma.rn.f32 	%r309, %r181, %r201, %r309;
	fma.rn.f32 	%r308, %r180, %r200, %r308;
	fma.rn.f32 	%r307, %r181, %r199, %r307;
	fma.rn.f32 	%r306, %r180, %r198, %r306;
	fma.rn.f32 	%r313, %r183, %r197, %r313;
	fma.rn.f32 	%r312, %r182, %r196, %r312;
	fma.rn.f32 	%r311, %r183, %r195, %r311;
	fma.rn.f32 	%r310, %r182, %r194, %r310;
	.loc	1 165 18                        // sk05_mlp_gateup.py:165:18
	add.s64 	%rd56, %rd18, %rd70;
	add.s64 	%rd57, %rd17, %rd70;
	add.s64 	%rd58, %rd16, %rd70;
	add.s64 	%rd59, %rd15, %rd70;
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	add.s64 	%rd60, %rd14, %rd70;
	add.s32 	%r202, %r305, 1;
	setp.gt.s32 	%p6, %r202, 1;
	selp.b32 	%r305, 0, %r202, %p6;
	.loc	1 163 30                        // sk05_mlp_gateup.py:163:30
	shl.b32 	%r203, %r305, 11;
	bar.sync 	0;
	add.s32 	%r204, %r13, %r203;
	add.s32 	%r168, %r204, 32768;
	selp.b32 	%r169, 8, 0, %p4;
	// begin inline asm
	cp.async.ca.shared.global [ %r168 + 0 ], [ %rd56 + 0 ], 0x8, %r169;
	// end inline asm
	cp.async.commit_group;
	.loc	1 163 47                        // sk05_mlp_gateup.py:163:47
	shl.b32 	%r205, %r305, 14;
	add.s32 	%r170, %r28, %r205;
	selp.b32 	%r171, 16, 0, %p4;
	// begin inline asm
	cp.async.cg.shared.global [ %r170 + 0 ], [ %rd57 + 0 ], 0x10, %r171;
	// end inline asm
	add.s32 	%r172, %r170, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r172 + 0 ], [ %rd58 + 0 ], 0x10, %r171;
	// end inline asm
	add.s32 	%r173, %r170, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r173 + 0 ], [ %rd59 + 0 ], 0x10, %r171;
	// end inline asm
	add.s32 	%r174, %r170, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r174 + 0 ], [ %rd60 + 0 ], 0x10, %r171;
	// end inline asm
	cp.async.commit_group;
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	add.s64 	%rd71, %rd71, 1;
	add.s64 	%rd70, %rd70, 128;
	add.s32 	%r303, %r303, %r25;
	setp.ne.b64 	%p7, %rd13, %rd70;
	@%p7 bra 	$L__BB0_2;
$L__BB0_3:                              // %._crit_edge
	.loc	1 0 23                          // sk05_mlp_gateup.py:0:23
	cvt.u32.u64 	%r225, %rd2;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r226, %r6, %r225;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r227, %r226, %r21;
	.loc	1 152 45                        // sk05_mlp_gateup.py:152:45
	and.b32 	%r228, %r2, 28;
	bfe.u32 	%r229, %r2, 2, 3;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r230, %r229, %r1;
	or.b32 	%r231, %r230, 8;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r232, %r231, %r20;
	rem.s32 	%r233, %r230, %r20;
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 167 38                        // sk05_mlp_gateup.py:167:38
	mad.wide.s32 	%rd62, %r233, 4, %rd23;
	mad.wide.s32 	%rd63, %r232, 4, %rd23;
	.loc	1 167 24                        // sk05_mlp_gateup.py:167:24
	// begin inline asm
	mov.u32 %r206, 0x0;
	ld.global.b32 { %r206 }, [ %rd62 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r207, 0x0;
	ld.global.b32 { %r207 }, [ %rd63 + 0 ];
	// end inline asm
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r234, %r306, %r206;
	mul.f32 	%r235, %r307, %r206;
	mul.f32 	%r236, %r308, %r207;
	mul.f32 	%r237, %r309, %r207;
	mul.f32 	%r238, %r310, %r206;
	mul.f32 	%r239, %r311, %r206;
	mul.f32 	%r240, %r312, %r207;
	mul.f32 	%r241, %r313, %r207;
	.loc	1 168 38                        // sk05_mlp_gateup.py:168:38
	mad.wide.s32 	%rd64, %r10, 4, %rd24;
	mad.wide.s32 	%rd65, %r11, 4, %rd24;
	.loc	1 168 24                        // sk05_mlp_gateup.py:168:24
	// begin inline asm
	mov.u32 %r208, 0x0;
	mov.u32 %r209, 0x0;
	ld.global.v2.b32 { %r208, %r209 }, [ %rd64 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r210, 0x0;
	mov.u32 %r211, 0x0;
	ld.global.v2.b32 { %r210, %r211 }, [ %rd65 + 0 ];
	// end inline asm
	.loc	1 169 49                        // sk05_mlp_gateup.py:169:49
	mul.lo.s32 	%r242, %r5, %r24;
	.loc	1 169 31                        // sk05_mlp_gateup.py:169:31
	mad.wide.s32 	%rd68, %r242, 2, %rd22;
	.loc	1 169 64                        // sk05_mlp_gateup.py:169:64
	mad.wide.s32 	%rd66, %r227, 2, %rd68;
	.loc	1 169 19                        // sk05_mlp_gateup.py:169:19
	// begin inline asm
	mov.u32 %r213, 0x0;
	mov.u32 %r214, 0x0;
	mov.u32 %r215, 0x0;
	mov.u32 %r216, 0x0;
	ld.global.v4.b32 { %r213, %r214, %r215, %r216 }, [ %rd66 + 0 ];
	// end inline asm
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
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
	mov.b32 	{%rs17, %rs18}, %r254;
	mov.b32 	{%rs19, %rs20}, %r255;
	mov.b32 	{%rs21, %rs22}, %r256;
	mov.b32 	{%rs23, %rs24}, %r257;
	cvt.f32.bf16 	%r258, %rs17;
	cvt.f32.bf16 	%r259, %rs18;
	cvt.f32.bf16 	%r260, %rs19;
	cvt.f32.bf16 	%r261, %rs20;
	cvt.f32.bf16 	%r262, %rs21;
	cvt.f32.bf16 	%r263, %rs22;
	cvt.f32.bf16 	%r264, %rs23;
	cvt.f32.bf16 	%r265, %rs24;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r266, %r234, %r208, %r258;
	fma.rn.f32 	%r267, %r235, %r209, %r259;
	fma.rn.f32 	%r268, %r236, %r208, %r260;
	fma.rn.f32 	%r269, %r237, %r209, %r261;
	fma.rn.f32 	%r270, %r238, %r210, %r262;
	fma.rn.f32 	%r271, %r239, %r211, %r263;
	fma.rn.f32 	%r272, %r240, %r210, %r264;
	fma.rn.f32 	%r273, %r241, %r211, %r265;
	.loc	1 176 31                        // sk05_mlp_gateup.py:176:31
	setp.lt.s32 	%p9, %r4, %r20;
	.loc	1 176 54                        // sk05_mlp_gateup.py:176:54
	setp.lt.s32 	%p10, %r226, %r21;
	.loc	1 176 37                        // sk05_mlp_gateup.py:176:37
	and.pred 	%p8, %p9, %p10;
	.loc	1 174 35                        // sk05_mlp_gateup.py:174:35
	mul.lo.s32 	%r274, %r4, %r23;
	.loc	1 174 18                        // sk05_mlp_gateup.py:174:18
	mad.wide.s32 	%rd69, %r274, 2, %rd21;
	.loc	1 174 50                        // sk05_mlp_gateup.py:174:50
	mad.wide.s32 	%rd67, %r226, 2, %rd69;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16.f32 	%rs9, %r266;
	cvt.rn.bf16.f32 	%rs10, %r267;
	cvt.rn.bf16.f32 	%rs11, %r268;
	cvt.rn.bf16.f32 	%rs12, %r269;
	cvt.rn.bf16.f32 	%rs13, %r270;
	cvt.rn.bf16.f32 	%rs14, %r271;
	cvt.rn.bf16.f32 	%rs15, %r272;
	cvt.rn.bf16.f32 	%rs16, %r273;
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
	st.shared.v2.b16 [ %r217 + 0 ], { %rs9, %rs10 };
	// end inline asm
	add.s32 	%r218, %r217, 128;
	// begin inline asm
	st.shared.v2.b16 [ %r218 + 0 ], { %rs11, %rs12 };
	// end inline asm
	xor.b32 	%r287, %r286, 4;
	add.s32 	%r219, %r112, %r287;
	// begin inline asm
	st.shared.v2.b16 [ %r219 + 0 ], { %rs13, %rs14 };
	// end inline asm
	add.s32 	%r220, %r219, 128;
	// begin inline asm
	st.shared.v2.b16 [ %r220 + 0 ], { %rs15, %rs16 };
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
	.loc	1 175 8                         // sk05_mlp_gateup.py:175:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd67 + 0 ], { %r221, %r222, %r223, %r224 };
	// end inline asm
	.loc	1 173 4                         // sk05_mlp_gateup.py:173:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk05_mlp_gateup.py"
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
.b8 11                                  // DW_FORM_data1
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 165                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x9e DW_TAG_compile_unit
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
.b8 114
.b8 101
.b8 112
.b8 111
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
.b8 2                                   // Abbrev [2] 0x48:0x1a DW_TAG_subprogram
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
.b8 3                                   // Abbrev [3] 0x62:0x46 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 72                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x77:0x18 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 144                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x8f:0x18 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 145                                 // DW_AT_call_line
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
	.reg .b16 	%rs<33>;
	.reg .b32 	%r<337>;
	.reg .b64 	%rd<79>;
	.loc	1 133 0                         // sk05_mlp_gateup.py:133:0
$L__func_begin0:
	.loc	1 133 0                         // sk05_mlp_gateup.py:133:0

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
	.loc	1 143 24                        // sk05_mlp_gateup.py:143:24
	mov.u32 	%r41, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk05_mlp_gateup.py:144:27 ]
	add.s32 	%r42, %r20, 15;
	.loc	2 43 30                         // standard.py:43:30 @[ sk05_mlp_gateup.py:144:27 ]
	shr.s32 	%r43, %r42, 31;
	shr.u32 	%r44, %r43, 28;
	add.s32 	%r45, %r42, %r44;
	shr.s32 	%r46, %r45, 4;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk05_mlp_gateup.py:145:27 ]
	add.s32 	%r47, %r21, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk05_mlp_gateup.py:145:27 ]
	shr.s32 	%r48, %r47, 31;
	shr.u32 	%r49, %r48, 25;
	add.s32 	%r50, %r47, %r49;
	shr.s32 	%r51, %r50, 7;
$L__tmp3:
	.loc	1 146 29                        // sk05_mlp_gateup.py:146:29
	shl.b32 	%r52, %r51, 3;
	.loc	1 147 22                        // sk05_mlp_gateup.py:147:22
	div.s32 	%r53, %r41, %r52;
	.loc	1 147 38                        // sk05_mlp_gateup.py:147:38
	shl.b32 	%r54, %r53, 3;
	.loc	1 148 30                        // sk05_mlp_gateup.py:148:30
	sub.s32 	%r55, %r46, %r54;
	ld.param.b32 	%r56, [_sk05_mlp_gateup_kernel_param_10];
	.loc	1 148 39                        // sk05_mlp_gateup.py:148:39
	min.s32 	%r57, %r55, 8;
	ld.param.b32 	%r58, [_sk05_mlp_gateup_kernel_param_11];
	.loc	1 149 30                        // sk05_mlp_gateup.py:149:30
	mul.lo.s32 	%r59, %r53, %r52;
	sub.s32 	%r60, %r41, %r59;
	.loc	1 150 36                        // sk05_mlp_gateup.py:150:36
	div.s32 	%r61, %r60, %r57;
	.loc	1 149 46                        // sk05_mlp_gateup.py:149:46
	mul.lo.s32 	%r62, %r61, %r57;
	sub.s32 	%r63, %r60, %r62;
	.loc	1 149 23                        // sk05_mlp_gateup.py:149:23
	add.s32 	%r64, %r63, %r54;
	.loc	1 152 22                        // sk05_mlp_gateup.py:152:22
	shl.b32 	%r1, %r64, 4;
	.loc	1 152 45                        // sk05_mlp_gateup.py:152:45
	mov.u32 	%r2, %tid.x;
	and.b32 	%r3, %r2, 240;
	bfe.u32 	%r65, %r2, 4, 4;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r4, %r1, %r65;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r5, %r4, %r20;
	.loc	1 153 22                        // sk05_mlp_gateup.py:153:22
	shl.b32 	%r6, %r61, 7;
	.loc	1 153 45                        // sk05_mlp_gateup.py:153:45
	shr.u32 	%r66, %r2, 3;
	bfe.u32 	%r67, %r2, 3, 5;
	shl.b32 	%r7, %r2, 1;
	and.b32 	%r68, %r7, 6;
	and.b32 	%r8, %r2, 224;
	shr.u32 	%r69, %r8, 2;
	or.b32 	%r70, %r68, %r69;
	and.b32 	%r9, %r2, 15;
	shl.b32 	%r71, %r9, 3;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r72, %r6, %r67;
	or.b32 	%r73, %r72, 32;
	or.b32 	%r74, %r72, 64;
	or.b32 	%r75, %r66, %r6;
	or.b32 	%r76, %r75, 96;
	or.b32 	%r77, %r6, %r70;
	or.b32 	%r79, %r77, 64;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r81, %r72, %r21;
	rem.s32 	%r82, %r73, %r21;
	rem.s32 	%r83, %r74, %r21;
	rem.s32 	%r84, %r76, %r21;
	rem.s32 	%r10, %r77, %r21;
	rem.s32 	%r11, %r79, %r21;
	.loc	1 156 39                        // sk05_mlp_gateup.py:156:39
	mul.lo.s32 	%r87, %r5, %r56;
	.loc	1 156 21                        // sk05_mlp_gateup.py:156:21
	cvt.s64.s32 	%rd1, %r87;
	add.s64 	%rd36, %rd19, %rd1;
	.loc	1 156 51                        // sk05_mlp_gateup.py:156:51
	cvt.u64.u32 	%rd2, %r71;
	add.s64 	%rd25, %rd36, %rd2;
	.loc	1 157 28                        // sk05_mlp_gateup.py:157:28
	and.b32 	%r12, %r2, 7;
	shl.b32 	%r88, %r12, 4;
	.loc	1 157 21                        // sk05_mlp_gateup.py:157:21
	cvt.u64.u32 	%rd3, %r88;
	add.s64 	%rd37, %rd20, %rd3;
	.loc	1 157 69                        // sk05_mlp_gateup.py:157:69
	mul.lo.s32 	%r89, %r81, %r58;
	mul.lo.s32 	%r90, %r82, %r58;
	mul.lo.s32 	%r91, %r83, %r58;
	mul.lo.s32 	%r92, %r84, %r58;
	.loc	1 157 51                        // sk05_mlp_gateup.py:157:51
	cvt.s64.s32 	%rd4, %r89;
	add.s64 	%rd26, %rd37, %rd4;
	cvt.s64.s32 	%rd5, %r90;
	add.s64 	%rd27, %rd37, %rd5;
	cvt.s64.s32 	%rd6, %r91;
	add.s64 	%rd28, %rd37, %rd6;
	cvt.s64.s32 	%rd7, %r92;
	add.s64 	%rd29, %rd37, %rd7;
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	setp.lt.s32 	%p1, %r22, 128;
	setp.gt.s32 	%p2, %r22, 127;
	.loc	1 163 30                        // sk05_mlp_gateup.py:163:30
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
	.loc	1 163 47                        // sk05_mlp_gateup.py:163:47
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
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	setp.gt.s32 	%p3, %r22, 255;
	.loc	1 164 18                        // sk05_mlp_gateup.py:164:18
	add.s64 	%rd30, %rd25, 128;
	.loc	1 165 18                        // sk05_mlp_gateup.py:165:18
	add.s64 	%rd31, %rd26, 128;
	add.s64 	%rd32, %rd27, 128;
	add.s64 	%rd33, %rd28, 128;
	add.s64 	%rd34, %rd29, 128;
	.loc	1 163 30                        // sk05_mlp_gateup.py:163:30
	bar.sync 	0;
	add.s32 	%r34, %r13, 34816;
	selp.b32 	%r35, 8, 0, %p3;
	// begin inline asm
	cp.async.ca.shared.global [ %r34 + 0 ], [ %rd30 + 0 ], 0x8, %r35;
	// end inline asm
	cp.async.commit_group;
	.loc	1 163 47                        // sk05_mlp_gateup.py:163:47
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
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
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
	cvt.s64.s32 	%rd38, %r96;
	add.s64 	%rd8, %rd35, %rd38;
	cvt.s64.s32 	%rd39, %r100;
	add.s64 	%rd9, %rd35, %rd39;
	cvt.s64.s32 	%rd40, %r104;
	add.s64 	%rd10, %rd35, %rd40;
	cvt.s64.s32 	%rd41, %r108;
	add.s64 	%rd11, %rd35, %rd41;
	.loc	1 161 28                        // sk05_mlp_gateup.py:161:28
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
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	cvt.s64.s32 	%rd12, %r118;
	and.b32 	%r127, %r22, -128;
	cvt.u64.u32 	%rd13, %r127;
	add.s64 	%rd42, %rd3, %rd7;
	add.s64 	%rd43, %rd42, %rd20;
	add.s64 	%rd14, %rd43, 256;
	add.s64 	%rd44, %rd3, %rd6;
	add.s64 	%rd45, %rd44, %rd20;
	add.s64 	%rd15, %rd45, 256;
	add.s64 	%rd46, %rd3, %rd5;
	add.s64 	%rd47, %rd46, %rd20;
	add.s64 	%rd16, %rd47, 256;
	add.s64 	%rd48, %rd3, %rd4;
	add.s64 	%rd49, %rd48, %rd20;
	add.s64 	%rd17, %rd49, 256;
	add.s64 	%rd50, %rd2, %rd1;
	add.s64 	%rd51, %rd50, %rd19;
	add.s64 	%rd18, %rd51, 256;
	mov.b32 	%r329, 0f00000000;
	mov.b32 	%r328, 1;
	mov.b32 	%r327, -1;
	mov.b64 	%rd77, 0;
	mov.b32 	%r128, 0;
	mov.b32 	%r326, %r128;
	mov.b64 	%rd78, %rd77;
	mov.b32 	%r330, %r329;
	mov.b32 	%r331, %r329;
	mov.b32 	%r332, %r329;
	mov.b32 	%r333, %r329;
	mov.b32 	%r334, %r329;
	mov.b32 	%r335, %r329;
	mov.b32 	%r336, %r329;
$L__BB0_2:                              // %__nv_exp2f.exit
                                        // =>This Inner Loop Header: Depth=1
	setp.lt.s64 	%p4, %rd78, %rd12;
	add.s32 	%r176, %r327, 1;
	setp.gt.s32 	%p5, %r176, 1;
	selp.b32 	%r327, 0, %r176, %p5;
	.loc	1 162 39                        // sk05_mlp_gateup.py:162:39
	cvt.s64.s32 	%rd61, %r326;
	add.s64 	%rd52, %rd8, %rd61;
	add.s64 	%rd53, %rd9, %rd61;
	add.s64 	%rd54, %rd10, %rd61;
	add.s64 	%rd55, %rd11, %rd61;
	.loc	1 162 29                        // sk05_mlp_gateup.py:162:29
	// begin inline asm
	mov.u16 %rs1, 0x0;
	ld.global.b8 { %rs1 }, [ %rd52 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs5, %rs1;
	// begin inline asm
	mov.u16 %rs2, 0x0;
	ld.global.b8 { %rs2 }, [ %rd53 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs6, %rs2;
	// begin inline asm
	mov.u16 %rs3, 0x0;
	ld.global.b8 { %rs3 }, [ %rd54 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs7, %rs3;
	// begin inline asm
	mov.u16 %rs4, 0x0;
	ld.global.b8 { %rs4 }, [ %rd55 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs8, %rs4;
	.loc	1 162 63                        // sk05_mlp_gateup.py:162:63
	cvt.rn.f32.s16 	%r177, %rs5;
	cvt.rn.f32.s16 	%r178, %rs6;
	cvt.rn.f32.s16 	%r179, %rs7;
	cvt.rn.f32.s16 	%r180, %rs8;
	.loc	1 162 21                        // sk05_mlp_gateup.py:162:21
	ex2.approx.ftz.f32 	%r181, %r177;
	ex2.approx.ftz.f32 	%r182, %r178;
	ex2.approx.ftz.f32 	%r183, %r179;
	ex2.approx.ftz.f32 	%r184, %r180;
	.loc	1 163 30                        // sk05_mlp_gateup.py:163:30
	cp.async.wait_group 	2;
	bar.sync 	0;
	shl.b32 	%r185, %r327, 11;
	add.s32 	%r186, %r113, %r185;
	add.s32 	%r187, %r186, %r14;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r129, %r130, %r131, %r132}, [%r187+32768];
	add.s32 	%r188, %r186, %r15;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r141, %r142, %r143, %r144}, [%r188+32768];
	add.s32 	%r189, %r186, %r16;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r153, %r154, %r155, %r156}, [%r189+32768];
	add.s32 	%r190, %r186, %r17;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r161, %r162, %r163, %r164}, [%r190+32768];
	.loc	1 163 47                        // sk05_mlp_gateup.py:163:47
	shl.b32 	%r191, %r327, 14;
	add.s32 	%r192, %r113, %r191;
	add.s32 	%r193, %r192, %r18;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r133, %r134, %r145, %r146}, [%r193];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r135, %r136, %r151, %r152}, [%r193+8192];
	add.s32 	%r194, %r192, %r19;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r157, %r158, %r165, %r166}, [%r194];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r159, %r160, %r167, %r168}, [%r194+8192];
	.loc	1 163 39                        // sk05_mlp_gateup.py:163:39
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
	.loc	1 163 79                        // sk05_mlp_gateup.py:163:79
	cvt.rn.f32.s32 	%r195, %r147;
	cvt.rn.f32.s32 	%r196, %r148;
	cvt.rn.f32.s32 	%r197, %r149;
	cvt.rn.f32.s32 	%r198, %r150;
	cvt.rn.f32.s32 	%r199, %r137;
	cvt.rn.f32.s32 	%r200, %r138;
	cvt.rn.f32.s32 	%r201, %r139;
	cvt.rn.f32.s32 	%r202, %r140;
	.loc	1 163 15                        // sk05_mlp_gateup.py:163:15
	fma.rn.f32 	%r332, %r182, %r202, %r332;
	fma.rn.f32 	%r331, %r181, %r201, %r331;
	fma.rn.f32 	%r330, %r182, %r200, %r330;
	fma.rn.f32 	%r329, %r181, %r199, %r329;
	fma.rn.f32 	%r336, %r184, %r198, %r336;
	fma.rn.f32 	%r335, %r183, %r197, %r335;
	fma.rn.f32 	%r334, %r184, %r196, %r334;
	fma.rn.f32 	%r333, %r183, %r195, %r333;
	.loc	1 165 18                        // sk05_mlp_gateup.py:165:18
	add.s64 	%rd56, %rd18, %rd77;
	add.s64 	%rd57, %rd17, %rd77;
	add.s64 	%rd58, %rd16, %rd77;
	add.s64 	%rd59, %rd15, %rd77;
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	add.s64 	%rd60, %rd14, %rd77;
	add.s32 	%r203, %r328, 1;
	setp.gt.s32 	%p6, %r203, 1;
	selp.b32 	%r328, 0, %r203, %p6;
	.loc	1 163 30                        // sk05_mlp_gateup.py:163:30
	shl.b32 	%r204, %r328, 11;
	bar.sync 	0;
	add.s32 	%r205, %r13, %r204;
	add.s32 	%r169, %r205, 32768;
	selp.b32 	%r170, 8, 0, %p4;
	// begin inline asm
	cp.async.ca.shared.global [ %r169 + 0 ], [ %rd56 + 0 ], 0x8, %r170;
	// end inline asm
	cp.async.commit_group;
	.loc	1 163 47                        // sk05_mlp_gateup.py:163:47
	shl.b32 	%r206, %r328, 14;
	add.s32 	%r171, %r29, %r206;
	selp.b32 	%r172, 16, 0, %p4;
	// begin inline asm
	cp.async.cg.shared.global [ %r171 + 0 ], [ %rd57 + 0 ], 0x10, %r172;
	// end inline asm
	add.s32 	%r173, %r171, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r173 + 0 ], [ %rd58 + 0 ], 0x10, %r172;
	// end inline asm
	add.s32 	%r174, %r171, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r174 + 0 ], [ %rd59 + 0 ], 0x10, %r172;
	// end inline asm
	add.s32 	%r175, %r171, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r175 + 0 ], [ %rd60 + 0 ], 0x10, %r172;
	// end inline asm
	cp.async.commit_group;
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	add.s64 	%rd78, %rd78, 1;
	add.s64 	%rd77, %rd77, 128;
	add.s32 	%r326, %r326, %r26;
	setp.ne.b64 	%p7, %rd13, %rd77;
	@%p7 bra 	$L__BB0_2;
$L__BB0_3:                              // %._crit_edge
	.loc	1 0 23                          // sk05_mlp_gateup.py:0:23
	cvt.u32.u64 	%r226, %rd2;
	.loc	1 153 45                        // sk05_mlp_gateup.py:153:45
	or.b32 	%r227, %r6, %r226;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r228, %r227, 7;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r229, %r228, %r21;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r230, %r227, 6;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r231, %r230, %r21;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r232, %r227, 5;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r233, %r232, %r21;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r234, %r227, 4;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r235, %r234, %r21;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r236, %r227, 3;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r237, %r236, %r21;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r238, %r227, 2;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r239, %r238, %r21;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r240, %r227, 1;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r241, %r240, %r21;
	rem.s32 	%r242, %r227, %r21;
	.loc	1 152 45                        // sk05_mlp_gateup.py:152:45
	and.b32 	%r243, %r2, 28;
	bfe.u32 	%r244, %r2, 2, 3;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r245, %r244, %r1;
	or.b32 	%r246, %r245, 8;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r247, %r246, %r20;
	rem.s32 	%r248, %r245, %r20;
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 167 38                        // sk05_mlp_gateup.py:167:38
	mad.wide.s32 	%rd62, %r248, 4, %rd23;
	mad.wide.s32 	%rd63, %r247, 4, %rd23;
	.loc	1 167 24                        // sk05_mlp_gateup.py:167:24
	// begin inline asm
	mov.u32 %r207, 0x0;
	ld.global.b32 { %r207 }, [ %rd62 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r208, 0x0;
	ld.global.b32 { %r208 }, [ %rd63 + 0 ];
	// end inline asm
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r249, %r329, %r207;
	mul.f32 	%r250, %r330, %r207;
	mul.f32 	%r251, %r331, %r208;
	mul.f32 	%r252, %r332, %r208;
	mul.f32 	%r253, %r333, %r207;
	mul.f32 	%r254, %r334, %r207;
	mul.f32 	%r255, %r335, %r208;
	mul.f32 	%r256, %r336, %r208;
	.loc	1 168 38                        // sk05_mlp_gateup.py:168:38
	mad.wide.s32 	%rd64, %r10, 4, %rd24;
	mad.wide.s32 	%rd65, %r11, 4, %rd24;
	.loc	1 168 24                        // sk05_mlp_gateup.py:168:24
	// begin inline asm
	mov.u32 %r209, 0x0;
	mov.u32 %r210, 0x0;
	ld.global.v2.b32 { %r209, %r210 }, [ %rd64 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r211, 0x0;
	mov.u32 %r212, 0x0;
	ld.global.v2.b32 { %r211, %r212 }, [ %rd65 + 0 ];
	// end inline asm
	.loc	1 169 49                        // sk05_mlp_gateup.py:169:49
	mul.lo.s32 	%r257, %r5, %r24;
	.loc	1 169 31                        // sk05_mlp_gateup.py:169:31
	mad.wide.s32 	%rd75, %r257, 2, %rd22;
	.loc	1 169 82                        // sk05_mlp_gateup.py:169:82
	mul.lo.s32 	%r258, %r242, %r25;
	mul.lo.s32 	%r259, %r241, %r25;
	mul.lo.s32 	%r260, %r239, %r25;
	mul.lo.s32 	%r261, %r237, %r25;
	mul.lo.s32 	%r262, %r235, %r25;
	mul.lo.s32 	%r263, %r233, %r25;
	mul.lo.s32 	%r264, %r231, %r25;
	mul.lo.s32 	%r265, %r229, %r25;
	.loc	1 169 64                        // sk05_mlp_gateup.py:169:64
	mad.wide.s32 	%rd66, %r258, 2, %rd75;
	mad.wide.s32 	%rd67, %r259, 2, %rd75;
	mad.wide.s32 	%rd68, %r260, 2, %rd75;
	mad.wide.s32 	%rd69, %r261, 2, %rd75;
	mad.wide.s32 	%rd70, %r262, 2, %rd75;
	mad.wide.s32 	%rd71, %r263, 2, %rd75;
	mad.wide.s32 	%rd72, %r264, 2, %rd75;
	mad.wide.s32 	%rd73, %r265, 2, %rd75;
	.loc	1 169 19                        // sk05_mlp_gateup.py:169:19
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
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	and.b32 	%r266, %r2, 120;
	shl.b32 	%r267, %r266, 5;
	or.b32 	%r268, %r267, %r325;
	xor.b32 	%r269, %r268, %r3;
	add.s32 	%r213, %r113, %r269;
	mov.b32 	%r214, {%rs9, %rs10};
	mov.b32 	%r215, {%rs11, %rs12};
	mov.b32 	%r216, {%rs13, %rs14};
	mov.b32 	%r217, {%rs15, %rs16};
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
	mov.b32 	{%rs25, %rs26}, %r277;
	mov.b32 	{%rs27, %rs28}, %r278;
	mov.b32 	{%rs29, %rs30}, %r279;
	mov.b32 	{%rs31, %rs32}, %r280;
	cvt.f32.bf16 	%r281, %rs25;
	cvt.f32.bf16 	%r282, %rs26;
	cvt.f32.bf16 	%r283, %rs27;
	cvt.f32.bf16 	%r284, %rs28;
	cvt.f32.bf16 	%r285, %rs29;
	cvt.f32.bf16 	%r286, %rs30;
	cvt.f32.bf16 	%r287, %rs31;
	cvt.f32.bf16 	%r288, %rs32;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r289, %r249, %r209, %r281;
	fma.rn.f32 	%r290, %r250, %r210, %r282;
	fma.rn.f32 	%r291, %r251, %r209, %r283;
	fma.rn.f32 	%r292, %r252, %r210, %r284;
	fma.rn.f32 	%r293, %r253, %r211, %r285;
	fma.rn.f32 	%r294, %r254, %r212, %r286;
	fma.rn.f32 	%r295, %r255, %r211, %r287;
	fma.rn.f32 	%r296, %r256, %r212, %r288;
	.loc	1 176 31                        // sk05_mlp_gateup.py:176:31
	setp.lt.s32 	%p9, %r4, %r20;
	.loc	1 176 54                        // sk05_mlp_gateup.py:176:54
	setp.lt.s32 	%p10, %r227, %r21;
	.loc	1 176 37                        // sk05_mlp_gateup.py:176:37
	and.pred 	%p8, %p9, %p10;
	.loc	1 174 35                        // sk05_mlp_gateup.py:174:35
	mul.lo.s32 	%r297, %r4, %r23;
	.loc	1 174 18                        // sk05_mlp_gateup.py:174:18
	mad.wide.s32 	%rd76, %r297, 2, %rd21;
	.loc	1 174 50                        // sk05_mlp_gateup.py:174:50
	mad.wide.s32 	%rd74, %r227, 2, %rd76;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16.f32 	%rs17, %r289;
	cvt.rn.bf16.f32 	%rs18, %r290;
	cvt.rn.bf16.f32 	%rs19, %r291;
	cvt.rn.bf16.f32 	%rs20, %r292;
	cvt.rn.bf16.f32 	%rs21, %r293;
	cvt.rn.bf16.f32 	%rs22, %r294;
	cvt.rn.bf16.f32 	%rs23, %r295;
	cvt.rn.bf16.f32 	%rs24, %r296;
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
	st.shared.v2.b16 [ %r218 + 0 ], { %rs17, %rs18 };
	// end inline asm
	add.s32 	%r219, %r218, 128;
	// begin inline asm
	st.shared.v2.b16 [ %r219 + 0 ], { %rs19, %rs20 };
	// end inline asm
	xor.b32 	%r310, %r309, 4;
	add.s32 	%r220, %r113, %r310;
	// begin inline asm
	st.shared.v2.b16 [ %r220 + 0 ], { %rs21, %rs22 };
	// end inline asm
	add.s32 	%r221, %r220, 128;
	// begin inline asm
	st.shared.v2.b16 [ %r221 + 0 ], { %rs23, %rs24 };
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
	.loc	1 175 8                         // sk05_mlp_gateup.py:175:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd74 + 0 ], { %r222, %r223, %r224, %r225 };
	// end inline asm
	.loc	1 173 4                         // sk05_mlp_gateup.py:173:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk05_mlp_gateup.py"
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
.b8 11                                  // DW_FORM_data1
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 165                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x9e DW_TAG_compile_unit
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
.b8 114
.b8 101
.b8 112
.b8 111
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
.b8 2                                   // Abbrev [2] 0x48:0x1a DW_TAG_subprogram
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
.b8 3                                   // Abbrev [3] 0x62:0x46 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 72                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x77:0x18 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 144                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x8f:0x18 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 145                                 // DW_AT_call_line
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
	.reg .b16 	%rs<81>;
	.reg .b32 	%r<929>;
	.reg .b64 	%rd<142>;
	.loc	1 133 0                         // sk05_mlp_gateup.py:133:0
$L__func_begin0:
	.loc	1 133 0                         // sk05_mlp_gateup.py:133:0

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
	.loc	1 143 24                        // sk05_mlp_gateup.py:143:24
	mov.u32 	%r49, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk05_mlp_gateup.py:144:27 ]
	add.s32 	%r50, %r25, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk05_mlp_gateup.py:144:27 ]
	shr.s32 	%r51, %r50, 31;
	shr.u32 	%r52, %r51, 25;
	add.s32 	%r53, %r50, %r52;
	shr.s32 	%r54, %r53, 7;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk05_mlp_gateup.py:145:27 ]
	add.s32 	%r55, %r26, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk05_mlp_gateup.py:145:27 ]
	shr.s32 	%r56, %r55, 31;
	shr.u32 	%r57, %r56, 25;
	add.s32 	%r58, %r55, %r57;
	shr.s32 	%r59, %r58, 7;
$L__tmp3:
	.loc	1 146 29                        // sk05_mlp_gateup.py:146:29
	shl.b32 	%r60, %r59, 3;
	.loc	1 147 22                        // sk05_mlp_gateup.py:147:22
	div.s32 	%r61, %r49, %r60;
	.loc	1 147 38                        // sk05_mlp_gateup.py:147:38
	shl.b32 	%r62, %r61, 3;
	.loc	1 148 30                        // sk05_mlp_gateup.py:148:30
	sub.s32 	%r63, %r54, %r62;
	ld.param.b32 	%r64, [_sk05_mlp_gateup_kernel_param_10];
	.loc	1 148 39                        // sk05_mlp_gateup.py:148:39
	min.s32 	%r65, %r63, 8;
	ld.param.b32 	%r66, [_sk05_mlp_gateup_kernel_param_11];
	.loc	1 149 30                        // sk05_mlp_gateup.py:149:30
	mul.lo.s32 	%r67, %r61, %r60;
	sub.s32 	%r68, %r49, %r67;
	.loc	1 150 36                        // sk05_mlp_gateup.py:150:36
	div.s32 	%r69, %r68, %r65;
	.loc	1 149 46                        // sk05_mlp_gateup.py:149:46
	mul.lo.s32 	%r70, %r69, %r65;
	sub.s32 	%r71, %r68, %r70;
	.loc	1 149 23                        // sk05_mlp_gateup.py:149:23
	add.s32 	%r72, %r71, %r62;
	.loc	1 152 22                        // sk05_mlp_gateup.py:152:22
	shl.b32 	%r1, %r72, 7;
	.loc	1 152 45                        // sk05_mlp_gateup.py:152:45
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
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r81, %r1, %r73;
	or.b32 	%r82, %r1, %r74;
	or.b32 	%r83, %r1, %r75;
	or.b32 	%r84, %r1, %r76;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r8, %r81, %r25;
	rem.s32 	%r9, %r82, %r25;
	rem.s32 	%r10, %r83, %r25;
	rem.s32 	%r11, %r84, %r25;
	.loc	1 153 22                        // sk05_mlp_gateup.py:153:22
	shl.b32 	%r12, %r69, 7;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r85, %r12, %r73;
	or.b32 	%r86, %r12, %r74;
	or.b32 	%r87, %r12, %r75;
	or.b32 	%r88, %r12, %r76;
	or.b32 	%r89, %r12, %r79;
	or.b32 	%r91, %r89, 32;
	or.b32 	%r93, %r89, 64;
	or.b32 	%r95, %r89, 96;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r97, %r85, %r26;
	rem.s32 	%r98, %r86, %r26;
	rem.s32 	%r99, %r87, %r26;
	rem.s32 	%r100, %r88, %r26;
	rem.s32 	%r13, %r89, %r26;
	rem.s32 	%r14, %r91, %r26;
	rem.s32 	%r15, %r93, %r26;
	rem.s32 	%r16, %r95, %r26;
	.loc	1 156 39                        // sk05_mlp_gateup.py:156:39
	mul.lo.s32 	%r105, %r8, %r64;
	mul.lo.s32 	%r106, %r9, %r64;
	mul.lo.s32 	%r107, %r10, %r64;
	mul.lo.s32 	%r108, %r11, %r64;
	.loc	1 156 21                        // sk05_mlp_gateup.py:156:21
	cvt.s64.s32 	%rd1, %r105;
	add.s64 	%rd51, %rd28, %rd1;
	cvt.s64.s32 	%rd2, %r106;
	add.s64 	%rd52, %rd28, %rd2;
	cvt.s64.s32 	%rd3, %r107;
	add.s64 	%rd53, %rd28, %rd3;
	cvt.s64.s32 	%rd4, %r108;
	add.s64 	%rd54, %rd28, %rd4;
	.loc	1 156 51                        // sk05_mlp_gateup.py:156:51
	cvt.u64.u32 	%rd5, %r80;
	add.s64 	%rd34, %rd51, %rd5;
	add.s64 	%rd35, %rd52, %rd5;
	add.s64 	%rd36, %rd53, %rd5;
	add.s64 	%rd37, %rd54, %rd5;
	.loc	1 157 21                        // sk05_mlp_gateup.py:157:21
	add.s64 	%rd55, %rd29, %rd5;
	.loc	1 157 69                        // sk05_mlp_gateup.py:157:69
	mul.lo.s32 	%r109, %r97, %r66;
	mul.lo.s32 	%r110, %r98, %r66;
	mul.lo.s32 	%r111, %r99, %r66;
	mul.lo.s32 	%r112, %r100, %r66;
	.loc	1 157 51                        // sk05_mlp_gateup.py:157:51
	cvt.s64.s32 	%rd6, %r109;
	add.s64 	%rd38, %rd55, %rd6;
	cvt.s64.s32 	%rd7, %r110;
	add.s64 	%rd39, %rd55, %rd7;
	cvt.s64.s32 	%rd8, %r111;
	add.s64 	%rd40, %rd55, %rd8;
	cvt.s64.s32 	%rd9, %r112;
	add.s64 	%rd41, %rd55, %rd9;
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	setp.gt.s32 	%p1, %r27, 127;
	.loc	1 163 30                        // sk05_mlp_gateup.py:163:30
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
	.loc	1 163 47                        // sk05_mlp_gateup.py:163:47
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
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	setp.gt.s32 	%p2, %r27, 255;
	.loc	1 164 18                        // sk05_mlp_gateup.py:164:18
	add.s64 	%rd42, %rd34, 128;
	add.s64 	%rd43, %rd35, 128;
	add.s64 	%rd44, %rd36, 128;
	add.s64 	%rd45, %rd37, 128;
	.loc	1 165 18                        // sk05_mlp_gateup.py:165:18
	add.s64 	%rd46, %rd38, 128;
	add.s64 	%rd47, %rd39, 128;
	add.s64 	%rd48, %rd40, 128;
	add.s64 	%rd49, %rd41, 128;
	.loc	1 163 30                        // sk05_mlp_gateup.py:163:30
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
	.loc	1 163 47                        // sk05_mlp_gateup.py:163:47
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
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
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
	cvt.s64.s32 	%rd56, %r116;
	add.s64 	%rd10, %rd50, %rd56;
	cvt.s64.s32 	%rd57, %r120;
	add.s64 	%rd11, %rd50, %rd57;
	cvt.s64.s32 	%rd58, %r124;
	add.s64 	%rd12, %rd50, %rd58;
	cvt.s64.s32 	%rd59, %r128;
	add.s64 	%rd13, %rd50, %rd59;
	cvt.s64.s32 	%rd60, %r132;
	add.s64 	%rd14, %rd50, %rd60;
	cvt.s64.s32 	%rd61, %r136;
	add.s64 	%rd15, %rd50, %rd61;
	cvt.s64.s32 	%rd62, %r140;
	add.s64 	%rd16, %rd50, %rd62;
	cvt.s64.s32 	%rd63, %r144;
	add.s64 	%rd17, %rd50, %rd63;
	.loc	1 161 28                        // sk05_mlp_gateup.py:161:28
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
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	cvt.s64.s32 	%rd18, %r150;
	and.b32 	%r159, %r27, -128;
	cvt.u64.u32 	%rd19, %r159;
	add.s64 	%rd64, %rd5, %rd9;
	add.s64 	%rd65, %rd64, %rd29;
	add.s64 	%rd20, %rd65, 256;
	add.s64 	%rd66, %rd5, %rd8;
	add.s64 	%rd67, %rd66, %rd29;
	add.s64 	%rd21, %rd67, 256;
	add.s64 	%rd68, %rd5, %rd7;
	add.s64 	%rd69, %rd68, %rd29;
	add.s64 	%rd22, %rd69, 256;
	add.s64 	%rd70, %rd5, %rd6;
	add.s64 	%rd71, %rd70, %rd29;
	add.s64 	%rd23, %rd71, 256;
	add.s64 	%rd72, %rd5, %rd4;
	add.s64 	%rd73, %rd72, %rd28;
	add.s64 	%rd24, %rd73, 256;
	add.s64 	%rd74, %rd5, %rd3;
	add.s64 	%rd75, %rd74, %rd28;
	add.s64 	%rd25, %rd75, 256;
	add.s64 	%rd76, %rd5, %rd2;
	add.s64 	%rd77, %rd76, %rd28;
	add.s64 	%rd26, %rd77, 256;
	add.s64 	%rd78, %rd5, %rd1;
	add.s64 	%rd79, %rd78, %rd28;
	add.s64 	%rd27, %rd79, 256;
	mov.b32 	%r865, 0f00000000;
	mov.b32 	%r862, 1;
	mov.b32 	%r861, -1;
	mov.b64 	%rd140, 0;
	mov.b32 	%r160, 0;
	mov.b32 	%r860, %r160;
	mov.b64 	%rd141, %rd140;
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
	setp.lt.s64 	%p3, %rd141, %rd18;
	add.s32 	%r330, %r861, 1;
	setp.gt.s32 	%p4, %r330, 1;
	selp.b32 	%r861, 0, %r330, %p4;
	.loc	1 162 39                        // sk05_mlp_gateup.py:162:39
	cvt.s64.s32 	%rd96, %r860;
	add.s64 	%rd80, %rd10, %rd96;
	add.s64 	%rd81, %rd11, %rd96;
	add.s64 	%rd82, %rd12, %rd96;
	add.s64 	%rd83, %rd13, %rd96;
	add.s64 	%rd84, %rd14, %rd96;
	add.s64 	%rd85, %rd15, %rd96;
	add.s64 	%rd86, %rd16, %rd96;
	add.s64 	%rd87, %rd17, %rd96;
	.loc	1 162 29                        // sk05_mlp_gateup.py:162:29
	// begin inline asm
	mov.u16 %rs1, 0x0;
	ld.global.b8 { %rs1 }, [ %rd80 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs9, %rs1;
	// begin inline asm
	mov.u16 %rs2, 0x0;
	ld.global.b8 { %rs2 }, [ %rd81 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs10, %rs2;
	// begin inline asm
	mov.u16 %rs3, 0x0;
	ld.global.b8 { %rs3 }, [ %rd82 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs11, %rs3;
	// begin inline asm
	mov.u16 %rs4, 0x0;
	ld.global.b8 { %rs4 }, [ %rd83 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs12, %rs4;
	// begin inline asm
	mov.u16 %rs5, 0x0;
	ld.global.b8 { %rs5 }, [ %rd84 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs13, %rs5;
	// begin inline asm
	mov.u16 %rs6, 0x0;
	ld.global.b8 { %rs6 }, [ %rd85 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs14, %rs6;
	// begin inline asm
	mov.u16 %rs7, 0x0;
	ld.global.b8 { %rs7 }, [ %rd86 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs15, %rs7;
	// begin inline asm
	mov.u16 %rs8, 0x0;
	ld.global.b8 { %rs8 }, [ %rd87 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs16, %rs8;
	.loc	1 162 63                        // sk05_mlp_gateup.py:162:63
	cvt.rn.f32.s16 	%r331, %rs9;
	cvt.rn.f32.s16 	%r332, %rs10;
	cvt.rn.f32.s16 	%r333, %rs11;
	cvt.rn.f32.s16 	%r334, %rs12;
	cvt.rn.f32.s16 	%r335, %rs13;
	cvt.rn.f32.s16 	%r336, %rs14;
	cvt.rn.f32.s16 	%r337, %rs15;
	cvt.rn.f32.s16 	%r338, %rs16;
	.loc	1 162 21                        // sk05_mlp_gateup.py:162:21
	ex2.approx.ftz.f32 	%r339, %r331;
	ex2.approx.ftz.f32 	%r340, %r332;
	ex2.approx.ftz.f32 	%r341, %r333;
	ex2.approx.ftz.f32 	%r342, %r334;
	ex2.approx.ftz.f32 	%r343, %r335;
	ex2.approx.ftz.f32 	%r344, %r336;
	ex2.approx.ftz.f32 	%r345, %r337;
	ex2.approx.ftz.f32 	%r346, %r338;
	.loc	1 163 30                        // sk05_mlp_gateup.py:163:30
	cp.async.wait_group 	2;
	bar.sync 	0;
	shl.b32 	%r347, %r861, 14;
	add.s32 	%r348, %r148, %r347;
	add.s32 	%r349, %r348, %r19;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r161, %r162, %r163, %r164}, [%r349];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r173, %r174, %r175, %r176}, [%r349+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r177, %r178, %r179, %r180}, [%r349+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r181, %r182, %r183, %r184}, [%r349+12288];
	add.s32 	%r350, %r348, %r20;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r189, %r190, %r191, %r192}, [%r350];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r217, %r218, %r219, %r220}, [%r350+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r237, %r238, %r239, %r240}, [%r350+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r257, %r258, %r259, %r260}, [%r350+12288];
	add.s32 	%r351, %r348, %r21;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r273, %r274, %r275, %r276}, [%r351];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r285, %r286, %r287, %r288}, [%r351+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r289, %r290, %r291, %r292}, [%r351+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r293, %r294, %r295, %r296}, [%r351+12288];
	add.s32 	%r352, %r348, %r22;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r297, %r298, %r299, %r300}, [%r352];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r309, %r310, %r311, %r312}, [%r352+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r313, %r314, %r315, %r316}, [%r352+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r317, %r318, %r319, %r320}, [%r352+12288];
	.loc	1 163 47                        // sk05_mlp_gateup.py:163:47
	add.s32 	%r353, %r348, %r23;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r165, %r166, %r193, %r194}, [%r353+32768];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r167, %r168, %r199, %r200}, [%r353+36864];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r169, %r170, %r205, %r206}, [%r353+40960];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r171, %r172, %r211, %r212}, [%r353+45056];
	add.s32 	%r354, %r348, %r24;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r277, %r278, %r301, %r302}, [%r354+32768];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r279, %r280, %r303, %r304}, [%r354+36864];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r281, %r282, %r305, %r306}, [%r354+40960];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r283, %r284, %r307, %r308}, [%r354+45056];
	.loc	1 163 39                        // sk05_mlp_gateup.py:163:39
	mov.b32 	%r185, %r160;
	mov.b32 	%r186, %r160;
	mov.b32 	%r187, %r160;
	mov.b32 	%r188, %r160;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r185, %r186, %r187, %r188 }, { %r161, %r162, %r163, %r164 }, { %r165, %r166 }, { %r185, %r186, %r187, %r188 };
	// end inline asm
	mov.b32 	%r195, %r160;
	mov.b32 	%r196, %r160;
	mov.b32 	%r197, %r160;
	mov.b32 	%r198, %r160;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r195, %r196, %r197, %r198 }, { %r161, %r162, %r163, %r164 }, { %r167, %r168 }, { %r195, %r196, %r197, %r198 };
	// end inline asm
	mov.b32 	%r201, %r160;
	mov.b32 	%r202, %r160;
	mov.b32 	%r203, %r160;
	mov.b32 	%r204, %r160;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r201, %r202, %r203, %r204 }, { %r161, %r162, %r163, %r164 }, { %r169, %r170 }, { %r201, %r202, %r203, %r204 };
	// end inline asm
	mov.b32 	%r207, %r160;
	mov.b32 	%r208, %r160;
	mov.b32 	%r209, %r160;
	mov.b32 	%r210, %r160;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r207, %r208, %r209, %r210 }, { %r161, %r162, %r163, %r164 }, { %r171, %r172 }, { %r207, %r208, %r209, %r210 };
	// end inline asm
	mov.b32 	%r213, %r160;
	mov.b32 	%r214, %r160;
	mov.b32 	%r215, %r160;
	mov.b32 	%r216, %r160;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r213, %r214, %r215, %r216 }, { %r173, %r174, %r175, %r176 }, { %r165, %r166 }, { %r213, %r214, %r215, %r216 };
	// end inline asm
	mov.b32 	%r221, %r160;
	mov.b32 	%r222, %r160;
	mov.b32 	%r223, %r160;
	mov.b32 	%r224, %r160;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r221, %r222, %r223, %r224 }, { %r173, %r174, %r175, %r176 }, { %r167, %r168 }, { %r221, %r222, %r223, %r224 };
	// end inline asm
	mov.b32 	%r225, %r160;
	mov.b32 	%r226, %r160;
	mov.b32 	%r227, %r160;
	mov.b32 	%r228, %r160;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r225, %r226, %r227, %r228 }, { %r173, %r174, %r175, %r176 }, { %r169, %r170 }, { %r225, %r226, %r227, %r228 };
	// end inline asm
	mov.b32 	%r229, %r160;
	mov.b32 	%r230, %r160;
	mov.b32 	%r231, %r160;
	mov.b32 	%r232, %r160;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r229, %r230, %r231, %r232 }, { %r173, %r174, %r175, %r176 }, { %r171, %r172 }, { %r229, %r230, %r231, %r232 };
	// end inline asm
	mov.b32 	%r233, %r160;
	mov.b32 	%r234, %r160;
	mov.b32 	%r235, %r160;
	mov.b32 	%r236, %r160;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r233, %r234, %r235, %r236 }, { %r177, %r178, %r179, %r180 }, { %r165, %r166 }, { %r233, %r234, %r235, %r236 };
	// end inline asm
	mov.b32 	%r241, %r160;
	mov.b32 	%r242, %r160;
	mov.b32 	%r243, %r160;
	mov.b32 	%r244, %r160;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r241, %r242, %r243, %r244 }, { %r177, %r178, %r179, %r180 }, { %r167, %r168 }, { %r241, %r242, %r243, %r244 };
	// end inline asm
	mov.b32 	%r245, %r160;
	mov.b32 	%r246, %r160;
	mov.b32 	%r247, %r160;
	mov.b32 	%r248, %r160;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r245, %r246, %r247, %r248 }, { %r177, %r178, %r179, %r180 }, { %r169, %r170 }, { %r245, %r246, %r247, %r248 };
	// end inline asm
	mov.b32 	%r249, %r160;
	mov.b32 	%r250, %r160;
	mov.b32 	%r251, %r160;
	mov.b32 	%r252, %r160;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r249, %r250, %r251, %r252 }, { %r177, %r178, %r179, %r180 }, { %r171, %r172 }, { %r249, %r250, %r251, %r252 };
	// end inline asm
	mov.b32 	%r253, %r160;
	mov.b32 	%r254, %r160;
	mov.b32 	%r255, %r160;
	mov.b32 	%r256, %r160;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r253, %r254, %r255, %r256 }, { %r181, %r182, %r183, %r184 }, { %r165, %r166 }, { %r253, %r254, %r255, %r256 };
	// end inline asm
	mov.b32 	%r261, %r160;
	mov.b32 	%r262, %r160;
	mov.b32 	%r263, %r160;
	mov.b32 	%r264, %r160;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r261, %r262, %r263, %r264 }, { %r181, %r182, %r183, %r184 }, { %r167, %r168 }, { %r261, %r262, %r263, %r264 };
	// end inline asm
	mov.b32 	%r265, %r160;
	mov.b32 	%r266, %r160;
	mov.b32 	%r267, %r160;
	mov.b32 	%r268, %r160;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r265, %r266, %r267, %r268 }, { %r181, %r182, %r183, %r184 }, { %r169, %r170 }, { %r265, %r266, %r267, %r268 };
	// end inline asm
	mov.b32 	%r269, %r160;
	mov.b32 	%r270, %r160;
	mov.b32 	%r271, %r160;
	mov.b32 	%r272, %r160;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r269, %r270, %r271, %r272 }, { %r181, %r182, %r183, %r184 }, { %r171, %r172 }, { %r269, %r270, %r271, %r272 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r185, %r186, %r187, %r188 }, { %r189, %r190, %r191, %r192 }, { %r193, %r194 }, { %r185, %r186, %r187, %r188 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r195, %r196, %r197, %r198 }, { %r189, %r190, %r191, %r192 }, { %r199, %r200 }, { %r195, %r196, %r197, %r198 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r201, %r202, %r203, %r204 }, { %r189, %r190, %r191, %r192 }, { %r205, %r206 }, { %r201, %r202, %r203, %r204 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r207, %r208, %r209, %r210 }, { %r189, %r190, %r191, %r192 }, { %r211, %r212 }, { %r207, %r208, %r209, %r210 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r213, %r214, %r215, %r216 }, { %r217, %r218, %r219, %r220 }, { %r193, %r194 }, { %r213, %r214, %r215, %r216 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r221, %r222, %r223, %r224 }, { %r217, %r218, %r219, %r220 }, { %r199, %r200 }, { %r221, %r222, %r223, %r224 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r225, %r226, %r227, %r228 }, { %r217, %r218, %r219, %r220 }, { %r205, %r206 }, { %r225, %r226, %r227, %r228 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r229, %r230, %r231, %r232 }, { %r217, %r218, %r219, %r220 }, { %r211, %r212 }, { %r229, %r230, %r231, %r232 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r233, %r234, %r235, %r236 }, { %r237, %r238, %r239, %r240 }, { %r193, %r194 }, { %r233, %r234, %r235, %r236 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r241, %r242, %r243, %r244 }, { %r237, %r238, %r239, %r240 }, { %r199, %r200 }, { %r241, %r242, %r243, %r244 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r245, %r246, %r247, %r248 }, { %r237, %r238, %r239, %r240 }, { %r205, %r206 }, { %r245, %r246, %r247, %r248 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r249, %r250, %r251, %r252 }, { %r237, %r238, %r239, %r240 }, { %r211, %r212 }, { %r249, %r250, %r251, %r252 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r253, %r254, %r255, %r256 }, { %r257, %r258, %r259, %r260 }, { %r193, %r194 }, { %r253, %r254, %r255, %r256 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r261, %r262, %r263, %r264 }, { %r257, %r258, %r259, %r260 }, { %r199, %r200 }, { %r261, %r262, %r263, %r264 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r265, %r266, %r267, %r268 }, { %r257, %r258, %r259, %r260 }, { %r205, %r206 }, { %r265, %r266, %r267, %r268 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r269, %r270, %r271, %r272 }, { %r257, %r258, %r259, %r260 }, { %r211, %r212 }, { %r269, %r270, %r271, %r272 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r185, %r186, %r187, %r188 }, { %r273, %r274, %r275, %r276 }, { %r277, %r278 }, { %r185, %r186, %r187, %r188 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r195, %r196, %r197, %r198 }, { %r273, %r274, %r275, %r276 }, { %r279, %r280 }, { %r195, %r196, %r197, %r198 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r201, %r202, %r203, %r204 }, { %r273, %r274, %r275, %r276 }, { %r281, %r282 }, { %r201, %r202, %r203, %r204 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r207, %r208, %r209, %r210 }, { %r273, %r274, %r275, %r276 }, { %r283, %r284 }, { %r207, %r208, %r209, %r210 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r213, %r214, %r215, %r216 }, { %r285, %r286, %r287, %r288 }, { %r277, %r278 }, { %r213, %r214, %r215, %r216 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r221, %r222, %r223, %r224 }, { %r285, %r286, %r287, %r288 }, { %r279, %r280 }, { %r221, %r222, %r223, %r224 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r225, %r226, %r227, %r228 }, { %r285, %r286, %r287, %r288 }, { %r281, %r282 }, { %r225, %r226, %r227, %r228 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r229, %r230, %r231, %r232 }, { %r285, %r286, %r287, %r288 }, { %r283, %r284 }, { %r229, %r230, %r231, %r232 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r233, %r234, %r235, %r236 }, { %r289, %r290, %r291, %r292 }, { %r277, %r278 }, { %r233, %r234, %r235, %r236 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r241, %r242, %r243, %r244 }, { %r289, %r290, %r291, %r292 }, { %r279, %r280 }, { %r241, %r242, %r243, %r244 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r245, %r246, %r247, %r248 }, { %r289, %r290, %r291, %r292 }, { %r281, %r282 }, { %r245, %r246, %r247, %r248 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r249, %r250, %r251, %r252 }, { %r289, %r290, %r291, %r292 }, { %r283, %r284 }, { %r249, %r250, %r251, %r252 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r253, %r254, %r255, %r256 }, { %r293, %r294, %r295, %r296 }, { %r277, %r278 }, { %r253, %r254, %r255, %r256 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r261, %r262, %r263, %r264 }, { %r293, %r294, %r295, %r296 }, { %r279, %r280 }, { %r261, %r262, %r263, %r264 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r265, %r266, %r267, %r268 }, { %r293, %r294, %r295, %r296 }, { %r281, %r282 }, { %r265, %r266, %r267, %r268 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r269, %r270, %r271, %r272 }, { %r293, %r294, %r295, %r296 }, { %r283, %r284 }, { %r269, %r270, %r271, %r272 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r185, %r186, %r187, %r188 }, { %r297, %r298, %r299, %r300 }, { %r301, %r302 }, { %r185, %r186, %r187, %r188 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r195, %r196, %r197, %r198 }, { %r297, %r298, %r299, %r300 }, { %r303, %r304 }, { %r195, %r196, %r197, %r198 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r201, %r202, %r203, %r204 }, { %r297, %r298, %r299, %r300 }, { %r305, %r306 }, { %r201, %r202, %r203, %r204 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r207, %r208, %r209, %r210 }, { %r297, %r298, %r299, %r300 }, { %r307, %r308 }, { %r207, %r208, %r209, %r210 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r213, %r214, %r215, %r216 }, { %r309, %r310, %r311, %r312 }, { %r301, %r302 }, { %r213, %r214, %r215, %r216 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r221, %r222, %r223, %r224 }, { %r309, %r310, %r311, %r312 }, { %r303, %r304 }, { %r221, %r222, %r223, %r224 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r225, %r226, %r227, %r228 }, { %r309, %r310, %r311, %r312 }, { %r305, %r306 }, { %r225, %r226, %r227, %r228 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r229, %r230, %r231, %r232 }, { %r309, %r310, %r311, %r312 }, { %r307, %r308 }, { %r229, %r230, %r231, %r232 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r233, %r234, %r235, %r236 }, { %r313, %r314, %r315, %r316 }, { %r301, %r302 }, { %r233, %r234, %r235, %r236 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r241, %r242, %r243, %r244 }, { %r313, %r314, %r315, %r316 }, { %r303, %r304 }, { %r241, %r242, %r243, %r244 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r245, %r246, %r247, %r248 }, { %r313, %r314, %r315, %r316 }, { %r305, %r306 }, { %r245, %r246, %r247, %r248 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r249, %r250, %r251, %r252 }, { %r313, %r314, %r315, %r316 }, { %r307, %r308 }, { %r249, %r250, %r251, %r252 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r253, %r254, %r255, %r256 }, { %r317, %r318, %r319, %r320 }, { %r301, %r302 }, { %r253, %r254, %r255, %r256 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r261, %r262, %r263, %r264 }, { %r317, %r318, %r319, %r320 }, { %r303, %r304 }, { %r261, %r262, %r263, %r264 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r265, %r266, %r267, %r268 }, { %r317, %r318, %r319, %r320 }, { %r305, %r306 }, { %r265, %r266, %r267, %r268 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r269, %r270, %r271, %r272 }, { %r317, %r318, %r319, %r320 }, { %r307, %r308 }, { %r269, %r270, %r271, %r272 };
	// end inline asm
	.loc	1 163 79                        // sk05_mlp_gateup.py:163:79
	cvt.rn.f32.s32 	%r355, %r269;
	cvt.rn.f32.s32 	%r356, %r270;
	cvt.rn.f32.s32 	%r357, %r271;
	cvt.rn.f32.s32 	%r358, %r272;
	cvt.rn.f32.s32 	%r359, %r265;
	cvt.rn.f32.s32 	%r360, %r266;
	cvt.rn.f32.s32 	%r361, %r267;
	cvt.rn.f32.s32 	%r362, %r268;
	cvt.rn.f32.s32 	%r363, %r261;
	cvt.rn.f32.s32 	%r364, %r262;
	cvt.rn.f32.s32 	%r365, %r263;
	cvt.rn.f32.s32 	%r366, %r264;
	cvt.rn.f32.s32 	%r367, %r253;
	cvt.rn.f32.s32 	%r368, %r254;
	cvt.rn.f32.s32 	%r369, %r255;
	cvt.rn.f32.s32 	%r370, %r256;
	cvt.rn.f32.s32 	%r371, %r249;
	cvt.rn.f32.s32 	%r372, %r250;
	cvt.rn.f32.s32 	%r373, %r251;
	cvt.rn.f32.s32 	%r374, %r252;
	cvt.rn.f32.s32 	%r375, %r245;
	cvt.rn.f32.s32 	%r376, %r246;
	cvt.rn.f32.s32 	%r377, %r247;
	cvt.rn.f32.s32 	%r378, %r248;
	cvt.rn.f32.s32 	%r379, %r241;
	cvt.rn.f32.s32 	%r380, %r242;
	cvt.rn.f32.s32 	%r381, %r243;
	cvt.rn.f32.s32 	%r382, %r244;
	cvt.rn.f32.s32 	%r383, %r233;
	cvt.rn.f32.s32 	%r384, %r234;
	cvt.rn.f32.s32 	%r385, %r235;
	cvt.rn.f32.s32 	%r386, %r236;
	cvt.rn.f32.s32 	%r387, %r229;
	cvt.rn.f32.s32 	%r388, %r230;
	cvt.rn.f32.s32 	%r389, %r231;
	cvt.rn.f32.s32 	%r390, %r232;
	cvt.rn.f32.s32 	%r391, %r225;
	cvt.rn.f32.s32 	%r392, %r226;
	cvt.rn.f32.s32 	%r393, %r227;
	cvt.rn.f32.s32 	%r394, %r228;
	cvt.rn.f32.s32 	%r395, %r221;
	cvt.rn.f32.s32 	%r396, %r222;
	cvt.rn.f32.s32 	%r397, %r223;
	cvt.rn.f32.s32 	%r398, %r224;
	cvt.rn.f32.s32 	%r399, %r213;
	cvt.rn.f32.s32 	%r400, %r214;
	cvt.rn.f32.s32 	%r401, %r215;
	cvt.rn.f32.s32 	%r402, %r216;
	cvt.rn.f32.s32 	%r403, %r207;
	cvt.rn.f32.s32 	%r404, %r208;
	cvt.rn.f32.s32 	%r405, %r209;
	cvt.rn.f32.s32 	%r406, %r210;
	cvt.rn.f32.s32 	%r407, %r201;
	cvt.rn.f32.s32 	%r408, %r202;
	cvt.rn.f32.s32 	%r409, %r203;
	cvt.rn.f32.s32 	%r410, %r204;
	cvt.rn.f32.s32 	%r411, %r195;
	cvt.rn.f32.s32 	%r412, %r196;
	cvt.rn.f32.s32 	%r413, %r197;
	cvt.rn.f32.s32 	%r414, %r198;
	cvt.rn.f32.s32 	%r415, %r185;
	cvt.rn.f32.s32 	%r416, %r186;
	cvt.rn.f32.s32 	%r417, %r187;
	cvt.rn.f32.s32 	%r418, %r188;
	.loc	1 163 15                        // sk05_mlp_gateup.py:163:15
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
	.loc	1 164 18                        // sk05_mlp_gateup.py:164:18
	add.s64 	%rd88, %rd27, %rd140;
	add.s64 	%rd89, %rd26, %rd140;
	add.s64 	%rd90, %rd25, %rd140;
	.loc	1 165 18                        // sk05_mlp_gateup.py:165:18
	add.s64 	%rd91, %rd24, %rd140;
	add.s64 	%rd92, %rd23, %rd140;
	add.s64 	%rd93, %rd22, %rd140;
	add.s64 	%rd94, %rd21, %rd140;
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	add.s64 	%rd95, %rd20, %rd140;
	add.s32 	%r419, %r862, 1;
	setp.gt.s32 	%p5, %r419, 1;
	selp.b32 	%r862, 0, %r419, %p5;
	.loc	1 163 30                        // sk05_mlp_gateup.py:163:30
	shl.b32 	%r420, %r862, 14;
	bar.sync 	0;
	add.s32 	%r321, %r31, %r420;
	selp.b32 	%r322, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r321 + 0 ], [ %rd88 + 0 ], 0x10, %r322;
	// end inline asm
	add.s32 	%r323, %r321, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r323 + 0 ], [ %rd89 + 0 ], 0x10, %r322;
	// end inline asm
	add.s32 	%r324, %r321, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r324 + 0 ], [ %rd90 + 0 ], 0x10, %r322;
	// end inline asm
	add.s32 	%r325, %r321, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r325 + 0 ], [ %rd91 + 0 ], 0x10, %r322;
	// end inline asm
	cp.async.commit_group;
	.loc	1 163 47                        // sk05_mlp_gateup.py:163:47
	add.s32 	%r326, %r321, 32768;
	// begin inline asm
	cp.async.cg.shared.global [ %r326 + 0 ], [ %rd92 + 0 ], 0x10, %r322;
	// end inline asm
	add.s32 	%r327, %r321, 36864;
	// begin inline asm
	cp.async.cg.shared.global [ %r327 + 0 ], [ %rd93 + 0 ], 0x10, %r322;
	// end inline asm
	add.s32 	%r328, %r321, 40960;
	// begin inline asm
	cp.async.cg.shared.global [ %r328 + 0 ], [ %rd94 + 0 ], 0x10, %r322;
	// end inline asm
	add.s32 	%r329, %r321, 45056;
	// begin inline asm
	cp.async.cg.shared.global [ %r329 + 0 ], [ %rd95 + 0 ], 0x10, %r322;
	// end inline asm
	cp.async.commit_group;
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	add.s64 	%rd141, %rd141, 1;
	add.s64 	%rd140, %rd140, 128;
	add.s32 	%r860, %r860, %r30;
	setp.ne.b64 	%p6, %rd19, %rd140;
	@%p6 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_4;
$L__BB0_1:                              // %.._crit_edge_crit_edge
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	shl.b32 	%r864, %r2, 1;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
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
	.loc	1 152 45                        // sk05_mlp_gateup.py:152:45
	or.b32 	%r543, %r12, %r859;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r544, %r543, 8;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r545, %r544, %r26;
	rem.s32 	%r546, %r543, %r26;
	.loc	1 152 45                        // sk05_mlp_gateup.py:152:45
	shl.b32 	%r547, %r7, 3;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r548, %r12, %r547;
	.loc	1 152 45                        // sk05_mlp_gateup.py:152:45
	and.b32 	%r549, %r2, 128;
	shr.u32 	%r550, %r549, 3;
	shr.u32 	%r551, %r2, 2;
	bfe.u32 	%r552, %r2, 2, 3;
	or.b32 	%r553, %r550, %r552;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r554, %r553, %r1;
	or.b32 	%r555, %r554, 104;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r556, %r555, %r25;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r557, %r554, 96;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r558, %r557, %r25;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r559, %r554, 72;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r560, %r559, %r25;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r561, %r554, 64;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r562, %r561, %r25;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r563, %r554, 40;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r564, %r563, %r25;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r565, %r554, 32;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r566, %r565, %r25;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r567, %r554, 8;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r568, %r567, %r25;
	rem.s32 	%r569, %r554, %r25;
	.loc	1 152 45                        // sk05_mlp_gateup.py:152:45
	shr.u32 	%r570, %r2, 4;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r571, %r570, %r1;
	or.b32 	%r572, %r571, 112;
	.loc	1 152 45                        // sk05_mlp_gateup.py:152:45
	bfe.u32 	%r573, %r2, 4, 4;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r574, %r573, %r1;
	or.b32 	%r575, %r574, 96;
	or.b32 	%r576, %r574, 80;
	or.b32 	%r577, %r574, 64;
	or.b32 	%r578, %r571, 48;
	or.b32 	%r579, %r574, 32;
	or.b32 	%r580, %r574, 16;
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 167 38                        // sk05_mlp_gateup.py:167:38
	mad.wide.s32 	%rd97, %r569, 4, %rd32;
	mad.wide.s32 	%rd98, %r568, 4, %rd32;
	mad.wide.s32 	%rd99, %r566, 4, %rd32;
	mad.wide.s32 	%rd100, %r564, 4, %rd32;
	mad.wide.s32 	%rd101, %r562, 4, %rd32;
	mad.wide.s32 	%rd102, %r560, 4, %rd32;
	mad.wide.s32 	%rd103, %r558, 4, %rd32;
	mad.wide.s32 	%rd104, %r556, 4, %rd32;
	.loc	1 167 24                        // sk05_mlp_gateup.py:167:24
	// begin inline asm
	mov.u32 %r421, 0x0;
	ld.global.b32 { %r421 }, [ %rd97 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r422, 0x0;
	ld.global.b32 { %r422 }, [ %rd98 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r423, 0x0;
	ld.global.b32 { %r423 }, [ %rd99 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r424, 0x0;
	ld.global.b32 { %r424 }, [ %rd100 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r425, 0x0;
	ld.global.b32 { %r425 }, [ %rd101 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r426, 0x0;
	ld.global.b32 { %r426 }, [ %rd102 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r427, 0x0;
	ld.global.b32 { %r427 }, [ %rd103 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r428, 0x0;
	ld.global.b32 { %r428 }, [ %rd104 + 0 ];
	// end inline asm
	.loc	1 168 38                        // sk05_mlp_gateup.py:168:38
	mad.wide.s32 	%rd105, %r13, 4, %rd33;
	mad.wide.s32 	%rd106, %r14, 4, %rd33;
	mad.wide.s32 	%rd107, %r15, 4, %rd33;
	mad.wide.s32 	%rd108, %r16, 4, %rd33;
	.loc	1 168 24                        // sk05_mlp_gateup.py:168:24
	// begin inline asm
	mov.u32 %r429, 0x0;
	mov.u32 %r430, 0x0;
	ld.global.v2.b32 { %r429, %r430 }, [ %rd105 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r431, 0x0;
	mov.u32 %r432, 0x0;
	ld.global.v2.b32 { %r431, %r432 }, [ %rd106 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r433, 0x0;
	mov.u32 %r434, 0x0;
	ld.global.v2.b32 { %r433, %r434 }, [ %rd107 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r435, 0x0;
	mov.u32 %r436, 0x0;
	ld.global.v2.b32 { %r435, %r436 }, [ %rd108 + 0 ];
	// end inline asm
	.loc	1 169 49                        // sk05_mlp_gateup.py:169:49
	mul.lo.s32 	%r581, %r8, %r29;
	mul.lo.s32 	%r582, %r9, %r29;
	mul.lo.s32 	%r583, %r10, %r29;
	mul.lo.s32 	%r584, %r11, %r29;
	.loc	1 169 31                        // sk05_mlp_gateup.py:169:31
	mad.wide.s32 	%rd125, %r581, 2, %rd31;
	mad.wide.s32 	%rd126, %r582, 2, %rd31;
	mad.wide.s32 	%rd127, %r583, 2, %rd31;
	mad.wide.s32 	%rd128, %r584, 2, %rd31;
	.loc	1 169 64                        // sk05_mlp_gateup.py:169:64
	mul.wide.s32 	%rd129, %r546, 2;
	add.s64 	%rd109, %rd125, %rd129;
	mul.wide.s32 	%rd130, %r545, 2;
	add.s64 	%rd110, %rd125, %rd130;
	add.s64 	%rd111, %rd126, %rd129;
	add.s64 	%rd112, %rd126, %rd130;
	add.s64 	%rd113, %rd127, %rd129;
	add.s64 	%rd114, %rd127, %rd130;
	add.s64 	%rd115, %rd128, %rd129;
	add.s64 	%rd116, %rd128, %rd130;
	.loc	1 169 19                        // sk05_mlp_gateup.py:169:19
	// begin inline asm
	mov.u32 %r438, 0x0;
	mov.u32 %r439, 0x0;
	mov.u32 %r440, 0x0;
	mov.u32 %r441, 0x0;
	ld.global.v4.b32 { %r438, %r439, %r440, %r441 }, [ %rd109 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r443, 0x0;
	mov.u32 %r444, 0x0;
	mov.u32 %r445, 0x0;
	mov.u32 %r446, 0x0;
	ld.global.v4.b32 { %r443, %r444, %r445, %r446 }, [ %rd110 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r447, 0x0;
	mov.u32 %r448, 0x0;
	mov.u32 %r449, 0x0;
	mov.u32 %r450, 0x0;
	ld.global.v4.b32 { %r447, %r448, %r449, %r450 }, [ %rd111 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r451, 0x0;
	mov.u32 %r452, 0x0;
	mov.u32 %r453, 0x0;
	mov.u32 %r454, 0x0;
	ld.global.v4.b32 { %r451, %r452, %r453, %r454 }, [ %rd112 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r455, 0x0;
	mov.u32 %r456, 0x0;
	mov.u32 %r457, 0x0;
	mov.u32 %r458, 0x0;
	ld.global.v4.b32 { %r455, %r456, %r457, %r458 }, [ %rd113 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r459, 0x0;
	mov.u32 %r460, 0x0;
	mov.u32 %r461, 0x0;
	mov.u32 %r462, 0x0;
	ld.global.v4.b32 { %r459, %r460, %r461, %r462 }, [ %rd114 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r463, 0x0;
	mov.u32 %r464, 0x0;
	mov.u32 %r465, 0x0;
	mov.u32 %r466, 0x0;
	ld.global.v4.b32 { %r463, %r464, %r465, %r466 }, [ %rd115 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r467, 0x0;
	mov.u32 %r468, 0x0;
	mov.u32 %r469, 0x0;
	mov.u32 %r470, 0x0;
	ld.global.v4.b32 { %r467, %r468, %r469, %r470 }, [ %rd116 + 0 ];
	// end inline asm
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
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
	.loc	1 176 31                        // sk05_mlp_gateup.py:176:31
	setp.lt.s32 	%p15, %r574, %r25;
	setp.lt.s32 	%p16, %r580, %r25;
	setp.lt.s32 	%p17, %r579, %r25;
	setp.lt.s32 	%p18, %r578, %r25;
	setp.lt.s32 	%p19, %r577, %r25;
	setp.lt.s32 	%p20, %r576, %r25;
	setp.lt.s32 	%p21, %r575, %r25;
	setp.lt.s32 	%p22, %r572, %r25;
	.loc	1 176 54                        // sk05_mlp_gateup.py:176:54
	setp.lt.s32 	%p23, %r548, %r26;
	.loc	1 176 37                        // sk05_mlp_gateup.py:176:37
	and.pred 	%p7, %p15, %p23;
	and.pred 	%p8, %p16, %p23;
	and.pred 	%p9, %p17, %p23;
	and.pred 	%p10, %p18, %p23;
	and.pred 	%p11, %p19, %p23;
	and.pred 	%p12, %p20, %p23;
	and.pred 	%p13, %p21, %p23;
	and.pred 	%p14, %p22, %p23;
	.loc	1 174 35                        // sk05_mlp_gateup.py:174:35
	mul.lo.s32 	%r631, %r574, %r28;
	mul.lo.s32 	%r632, %r580, %r28;
	mul.lo.s32 	%r633, %r579, %r28;
	mul.lo.s32 	%r634, %r578, %r28;
	mul.lo.s32 	%r635, %r577, %r28;
	mul.lo.s32 	%r636, %r576, %r28;
	mul.lo.s32 	%r637, %r575, %r28;
	mul.lo.s32 	%r638, %r572, %r28;
	.loc	1 174 18                        // sk05_mlp_gateup.py:174:18
	mad.wide.s32 	%rd131, %r631, 2, %rd30;
	mad.wide.s32 	%rd132, %r632, 2, %rd30;
	mad.wide.s32 	%rd133, %r633, 2, %rd30;
	mad.wide.s32 	%rd134, %r634, 2, %rd30;
	mad.wide.s32 	%rd135, %r635, 2, %rd30;
	mad.wide.s32 	%rd136, %r636, 2, %rd30;
	mad.wide.s32 	%rd137, %r637, 2, %rd30;
	mad.wide.s32 	%rd138, %r638, 2, %rd30;
	.loc	1 174 50                        // sk05_mlp_gateup.py:174:50
	mul.wide.s32 	%rd139, %r548, 2;
	add.s64 	%rd117, %rd131, %rd139;
	add.s64 	%rd118, %rd132, %rd139;
	add.s64 	%rd119, %rd133, %rd139;
	add.s64 	%rd120, %rd134, %rd139;
	add.s64 	%rd121, %rd135, %rd139;
	add.s64 	%rd122, %rd136, %rd139;
	add.s64 	%rd123, %rd137, %rd139;
	add.s64 	%rd124, %rd138, %rd139;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r639, %r914, %r427;
	mul.f32 	%r640, %r913, %r427;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs17, %rs18}, %r623;
	cvt.f32.bf16 	%r641, %rs18;
	cvt.f32.bf16 	%r642, %rs17;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r643, %r640, %r429, %r642;
	fma.rn.f32 	%r644, %r639, %r430, %r641;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r645, %r866, %r421;
	mul.f32 	%r646, %r865, %r421;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs19, %rs20}, %r597;
	cvt.f32.bf16 	%r647, %rs20;
	cvt.f32.bf16 	%r648, %rs19;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r649, %r646, %r429, %r648;
	fma.rn.f32 	%r650, %r645, %r430, %r647;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r472, %r650, %r649;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r651, %r868, %r422;
	mul.f32 	%r652, %r867, %r422;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs21, %rs22}, %r598;
	cvt.f32.bf16 	%r653, %rs22;
	cvt.f32.bf16 	%r654, %rs21;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r655, %r652, %r429, %r654;
	fma.rn.f32 	%r656, %r651, %r430, %r653;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r477, %r656, %r655;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r657, %r882, %r423;
	mul.f32 	%r658, %r881, %r423;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs23, %rs24}, %r607;
	cvt.f32.bf16 	%r659, %rs24;
	cvt.f32.bf16 	%r660, %rs23;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r661, %r658, %r429, %r660;
	fma.rn.f32 	%r662, %r657, %r430, %r659;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r473, %r662, %r661;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r663, %r884, %r424;
	mul.f32 	%r664, %r883, %r424;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs25, %rs26}, %r608;
	cvt.f32.bf16 	%r665, %rs26;
	cvt.f32.bf16 	%r666, %rs25;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r667, %r664, %r429, %r666;
	fma.rn.f32 	%r668, %r663, %r430, %r665;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r478, %r668, %r667;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r669, %r898, %r425;
	mul.f32 	%r670, %r897, %r425;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs27, %rs28}, %r615;
	cvt.f32.bf16 	%r671, %rs28;
	cvt.f32.bf16 	%r672, %rs27;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r673, %r670, %r429, %r672;
	fma.rn.f32 	%r674, %r669, %r430, %r671;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r474, %r674, %r673;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r675, %r900, %r426;
	mul.f32 	%r676, %r899, %r426;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs29, %rs30}, %r616;
	cvt.f32.bf16 	%r677, %rs30;
	cvt.f32.bf16 	%r678, %rs29;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r679, %r676, %r429, %r678;
	fma.rn.f32 	%r680, %r675, %r430, %r677;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r479, %r680, %r679;
	cvt.rn.bf16x2.f32 	%r475, %r644, %r643;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r681, %r916, %r428;
	mul.f32 	%r682, %r915, %r428;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs31, %rs32}, %r624;
	cvt.f32.bf16 	%r683, %rs32;
	cvt.f32.bf16 	%r684, %rs31;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r685, %r682, %r429, %r684;
	fma.rn.f32 	%r686, %r681, %r430, %r683;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r480, %r686, %r685;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r687, %r918, %r427;
	mul.f32 	%r688, %r917, %r427;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs33, %rs34}, %r625;
	cvt.f32.bf16 	%r689, %rs34;
	cvt.f32.bf16 	%r690, %rs33;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r691, %r688, %r431, %r690;
	fma.rn.f32 	%r692, %r687, %r432, %r689;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r693, %r870, %r421;
	mul.f32 	%r694, %r869, %r421;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs35, %rs36}, %r599;
	cvt.f32.bf16 	%r695, %rs36;
	cvt.f32.bf16 	%r696, %rs35;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r697, %r694, %r431, %r696;
	fma.rn.f32 	%r698, %r693, %r432, %r695;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r492, %r698, %r697;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r699, %r872, %r422;
	mul.f32 	%r700, %r871, %r422;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs37, %rs38}, %r600;
	cvt.f32.bf16 	%r701, %rs38;
	cvt.f32.bf16 	%r702, %rs37;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r703, %r700, %r431, %r702;
	fma.rn.f32 	%r704, %r699, %r432, %r701;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r497, %r704, %r703;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r705, %r886, %r423;
	mul.f32 	%r706, %r885, %r423;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs39, %rs40}, %r609;
	cvt.f32.bf16 	%r707, %rs40;
	cvt.f32.bf16 	%r708, %rs39;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r709, %r706, %r431, %r708;
	fma.rn.f32 	%r710, %r705, %r432, %r707;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r493, %r710, %r709;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r711, %r888, %r424;
	mul.f32 	%r712, %r887, %r424;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs41, %rs42}, %r610;
	cvt.f32.bf16 	%r713, %rs42;
	cvt.f32.bf16 	%r714, %rs41;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r715, %r712, %r431, %r714;
	fma.rn.f32 	%r716, %r711, %r432, %r713;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r498, %r716, %r715;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r717, %r902, %r425;
	mul.f32 	%r718, %r901, %r425;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs43, %rs44}, %r617;
	cvt.f32.bf16 	%r719, %rs44;
	cvt.f32.bf16 	%r720, %rs43;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r721, %r718, %r431, %r720;
	fma.rn.f32 	%r722, %r717, %r432, %r719;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r494, %r722, %r721;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r723, %r904, %r426;
	mul.f32 	%r724, %r903, %r426;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs45, %rs46}, %r618;
	cvt.f32.bf16 	%r725, %rs46;
	cvt.f32.bf16 	%r726, %rs45;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r727, %r724, %r431, %r726;
	fma.rn.f32 	%r728, %r723, %r432, %r725;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r499, %r728, %r727;
	cvt.rn.bf16x2.f32 	%r495, %r692, %r691;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r729, %r920, %r428;
	mul.f32 	%r730, %r919, %r428;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs47, %rs48}, %r626;
	cvt.f32.bf16 	%r731, %rs48;
	cvt.f32.bf16 	%r732, %rs47;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r733, %r730, %r431, %r732;
	fma.rn.f32 	%r734, %r729, %r432, %r731;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r500, %r734, %r733;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r735, %r922, %r427;
	mul.f32 	%r736, %r921, %r427;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs49, %rs50}, %r627;
	cvt.f32.bf16 	%r737, %rs50;
	cvt.f32.bf16 	%r738, %rs49;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r739, %r736, %r433, %r738;
	fma.rn.f32 	%r740, %r735, %r434, %r737;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r741, %r874, %r421;
	mul.f32 	%r742, %r873, %r421;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs51, %rs52}, %r603;
	cvt.f32.bf16 	%r743, %rs52;
	cvt.f32.bf16 	%r744, %rs51;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r745, %r742, %r433, %r744;
	fma.rn.f32 	%r746, %r741, %r434, %r743;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r482, %r746, %r745;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r747, %r876, %r422;
	mul.f32 	%r748, %r875, %r422;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs53, %rs54}, %r604;
	cvt.f32.bf16 	%r749, %rs54;
	cvt.f32.bf16 	%r750, %rs53;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r751, %r748, %r433, %r750;
	fma.rn.f32 	%r752, %r747, %r434, %r749;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r487, %r752, %r751;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r753, %r890, %r423;
	mul.f32 	%r754, %r889, %r423;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs55, %rs56}, %r611;
	cvt.f32.bf16 	%r755, %rs56;
	cvt.f32.bf16 	%r756, %rs55;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r757, %r754, %r433, %r756;
	fma.rn.f32 	%r758, %r753, %r434, %r755;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r483, %r758, %r757;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r759, %r892, %r424;
	mul.f32 	%r760, %r891, %r424;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs57, %rs58}, %r612;
	cvt.f32.bf16 	%r761, %rs58;
	cvt.f32.bf16 	%r762, %rs57;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r763, %r760, %r433, %r762;
	fma.rn.f32 	%r764, %r759, %r434, %r761;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r488, %r764, %r763;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r765, %r906, %r425;
	mul.f32 	%r766, %r905, %r425;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs59, %rs60}, %r619;
	cvt.f32.bf16 	%r767, %rs60;
	cvt.f32.bf16 	%r768, %rs59;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r769, %r766, %r433, %r768;
	fma.rn.f32 	%r770, %r765, %r434, %r767;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r484, %r770, %r769;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r771, %r908, %r426;
	mul.f32 	%r772, %r907, %r426;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs61, %rs62}, %r620;
	cvt.f32.bf16 	%r773, %rs62;
	cvt.f32.bf16 	%r774, %rs61;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r775, %r772, %r433, %r774;
	fma.rn.f32 	%r776, %r771, %r434, %r773;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r489, %r776, %r775;
	cvt.rn.bf16x2.f32 	%r485, %r740, %r739;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r777, %r924, %r428;
	mul.f32 	%r778, %r923, %r428;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs63, %rs64}, %r628;
	cvt.f32.bf16 	%r779, %rs64;
	cvt.f32.bf16 	%r780, %rs63;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r781, %r778, %r433, %r780;
	fma.rn.f32 	%r782, %r777, %r434, %r779;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r490, %r782, %r781;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r783, %r926, %r427;
	mul.f32 	%r784, %r925, %r427;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs65, %rs66}, %r629;
	cvt.f32.bf16 	%r785, %rs66;
	cvt.f32.bf16 	%r786, %rs65;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r787, %r784, %r435, %r786;
	fma.rn.f32 	%r788, %r783, %r436, %r785;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r789, %r878, %r421;
	mul.f32 	%r790, %r877, %r421;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs67, %rs68}, %r605;
	cvt.f32.bf16 	%r791, %rs68;
	cvt.f32.bf16 	%r792, %rs67;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r793, %r790, %r435, %r792;
	fma.rn.f32 	%r794, %r789, %r436, %r791;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r502, %r794, %r793;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r795, %r880, %r422;
	mul.f32 	%r796, %r879, %r422;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs69, %rs70}, %r606;
	cvt.f32.bf16 	%r797, %rs70;
	cvt.f32.bf16 	%r798, %rs69;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r799, %r796, %r435, %r798;
	fma.rn.f32 	%r800, %r795, %r436, %r797;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r507, %r800, %r799;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r801, %r894, %r423;
	mul.f32 	%r802, %r893, %r423;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs71, %rs72}, %r613;
	cvt.f32.bf16 	%r803, %rs72;
	cvt.f32.bf16 	%r804, %rs71;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r805, %r802, %r435, %r804;
	fma.rn.f32 	%r806, %r801, %r436, %r803;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r503, %r806, %r805;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r807, %r896, %r424;
	mul.f32 	%r808, %r895, %r424;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs73, %rs74}, %r614;
	cvt.f32.bf16 	%r809, %rs74;
	cvt.f32.bf16 	%r810, %rs73;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r811, %r808, %r435, %r810;
	fma.rn.f32 	%r812, %r807, %r436, %r809;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r508, %r812, %r811;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r813, %r910, %r425;
	mul.f32 	%r814, %r909, %r425;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs75, %rs76}, %r621;
	cvt.f32.bf16 	%r815, %rs76;
	cvt.f32.bf16 	%r816, %rs75;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r817, %r814, %r435, %r816;
	fma.rn.f32 	%r818, %r813, %r436, %r815;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r504, %r818, %r817;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r819, %r912, %r426;
	mul.f32 	%r820, %r911, %r426;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs77, %rs78}, %r622;
	cvt.f32.bf16 	%r821, %rs78;
	cvt.f32.bf16 	%r822, %rs77;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r823, %r820, %r435, %r822;
	fma.rn.f32 	%r824, %r819, %r436, %r821;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r509, %r824, %r823;
	cvt.rn.bf16x2.f32 	%r505, %r788, %r787;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r825, %r928, %r428;
	mul.f32 	%r826, %r927, %r428;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs79, %rs80}, %r630;
	cvt.f32.bf16 	%r827, %rs80;
	cvt.f32.bf16 	%r828, %rs79;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r829, %r826, %r435, %r828;
	fma.rn.f32 	%r830, %r825, %r436, %r827;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
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
	.loc	1 175 8                         // sk05_mlp_gateup.py:175:8
	// begin inline asm
	@%p7 st.global.v4.b32 [ %rd117 + 0 ], { %r511, %r512, %r513, %r514 };
	// end inline asm
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd118 + 0 ], { %r515, %r516, %r517, %r518 };
	// end inline asm
	// begin inline asm
	@%p9 st.global.v4.b32 [ %rd119 + 0 ], { %r519, %r520, %r521, %r522 };
	// end inline asm
	// begin inline asm
	@%p10 st.global.v4.b32 [ %rd120 + 0 ], { %r523, %r524, %r525, %r526 };
	// end inline asm
	// begin inline asm
	@%p11 st.global.v4.b32 [ %rd121 + 0 ], { %r527, %r528, %r529, %r530 };
	// end inline asm
	// begin inline asm
	@%p12 st.global.v4.b32 [ %rd122 + 0 ], { %r531, %r532, %r533, %r534 };
	// end inline asm
	// begin inline asm
	@%p13 st.global.v4.b32 [ %rd123 + 0 ], { %r535, %r536, %r537, %r538 };
	// end inline asm
	// begin inline asm
	@%p14 st.global.v4.b32 [ %rd124 + 0 ], { %r539, %r540, %r541, %r542 };
	// end inline asm
	.loc	1 173 4                         // sk05_mlp_gateup.py:173:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk05_mlp_gateup.py"
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
.b8 11                                  // DW_FORM_data1
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 165                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x9e DW_TAG_compile_unit
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
.b8 114
.b8 101
.b8 112
.b8 111
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
.b8 2                                   // Abbrev [2] 0x48:0x1a DW_TAG_subprogram
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
.b8 3                                   // Abbrev [3] 0x62:0x46 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 72                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x77:0x18 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 144                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x8f:0x18 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 145                                 // DW_AT_call_line
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
	.reg .b16 	%rs<145>;
	.reg .b32 	%r<974>;
	.reg .b64 	%rd<212>;
	.loc	1 133 0                         // sk05_mlp_gateup.py:133:0
$L__func_begin0:
	.loc	1 133 0                         // sk05_mlp_gateup.py:133:0

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
	.loc	1 143 24                        // sk05_mlp_gateup.py:143:24
	mov.u32 	%r50, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk05_mlp_gateup.py:144:27 ]
	add.s32 	%r51, %r25, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk05_mlp_gateup.py:144:27 ]
	shr.s32 	%r52, %r51, 31;
	shr.u32 	%r53, %r52, 25;
	add.s32 	%r54, %r51, %r53;
	shr.s32 	%r55, %r54, 7;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk05_mlp_gateup.py:145:27 ]
	add.s32 	%r56, %r26, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk05_mlp_gateup.py:145:27 ]
	shr.s32 	%r57, %r56, 31;
	shr.u32 	%r58, %r57, 25;
	add.s32 	%r59, %r56, %r58;
	shr.s32 	%r60, %r59, 7;
$L__tmp3:
	.loc	1 146 29                        // sk05_mlp_gateup.py:146:29
	shl.b32 	%r61, %r60, 3;
	.loc	1 147 22                        // sk05_mlp_gateup.py:147:22
	div.s32 	%r62, %r50, %r61;
	.loc	1 147 38                        // sk05_mlp_gateup.py:147:38
	shl.b32 	%r63, %r62, 3;
	.loc	1 148 30                        // sk05_mlp_gateup.py:148:30
	sub.s32 	%r64, %r55, %r63;
	ld.param.b32 	%r65, [_sk05_mlp_gateup_kernel_param_10];
	.loc	1 148 39                        // sk05_mlp_gateup.py:148:39
	min.s32 	%r66, %r64, 8;
	ld.param.b32 	%r67, [_sk05_mlp_gateup_kernel_param_11];
	.loc	1 149 30                        // sk05_mlp_gateup.py:149:30
	mul.lo.s32 	%r68, %r62, %r61;
	sub.s32 	%r69, %r50, %r68;
	.loc	1 150 36                        // sk05_mlp_gateup.py:150:36
	div.s32 	%r70, %r69, %r66;
	.loc	1 149 46                        // sk05_mlp_gateup.py:149:46
	mul.lo.s32 	%r71, %r70, %r66;
	sub.s32 	%r72, %r69, %r71;
	.loc	1 149 23                        // sk05_mlp_gateup.py:149:23
	add.s32 	%r73, %r72, %r63;
	.loc	1 152 22                        // sk05_mlp_gateup.py:152:22
	shl.b32 	%r1, %r73, 7;
	.loc	1 152 45                        // sk05_mlp_gateup.py:152:45
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
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r82, %r1, %r74;
	or.b32 	%r83, %r1, %r75;
	or.b32 	%r84, %r1, %r76;
	or.b32 	%r85, %r1, %r77;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r8, %r82, %r25;
	rem.s32 	%r9, %r83, %r25;
	rem.s32 	%r10, %r84, %r25;
	rem.s32 	%r11, %r85, %r25;
	.loc	1 153 22                        // sk05_mlp_gateup.py:153:22
	shl.b32 	%r12, %r70, 7;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r86, %r12, %r74;
	or.b32 	%r87, %r12, %r75;
	or.b32 	%r88, %r12, %r76;
	or.b32 	%r89, %r12, %r77;
	or.b32 	%r90, %r12, %r80;
	or.b32 	%r92, %r90, 32;
	or.b32 	%r94, %r90, 64;
	or.b32 	%r96, %r90, 96;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r98, %r86, %r26;
	rem.s32 	%r99, %r87, %r26;
	rem.s32 	%r100, %r88, %r26;
	rem.s32 	%r101, %r89, %r26;
	rem.s32 	%r13, %r90, %r26;
	rem.s32 	%r14, %r92, %r26;
	rem.s32 	%r15, %r94, %r26;
	rem.s32 	%r16, %r96, %r26;
	.loc	1 156 39                        // sk05_mlp_gateup.py:156:39
	mul.lo.s32 	%r106, %r8, %r65;
	mul.lo.s32 	%r107, %r9, %r65;
	mul.lo.s32 	%r108, %r10, %r65;
	mul.lo.s32 	%r109, %r11, %r65;
	.loc	1 156 21                        // sk05_mlp_gateup.py:156:21
	cvt.s64.s32 	%rd1, %r106;
	add.s64 	%rd51, %rd28, %rd1;
	cvt.s64.s32 	%rd2, %r107;
	add.s64 	%rd52, %rd28, %rd2;
	cvt.s64.s32 	%rd3, %r108;
	add.s64 	%rd53, %rd28, %rd3;
	cvt.s64.s32 	%rd4, %r109;
	add.s64 	%rd54, %rd28, %rd4;
	.loc	1 156 51                        // sk05_mlp_gateup.py:156:51
	cvt.u64.u32 	%rd5, %r81;
	add.s64 	%rd34, %rd51, %rd5;
	add.s64 	%rd35, %rd52, %rd5;
	add.s64 	%rd36, %rd53, %rd5;
	add.s64 	%rd37, %rd54, %rd5;
	.loc	1 157 21                        // sk05_mlp_gateup.py:157:21
	add.s64 	%rd55, %rd29, %rd5;
	.loc	1 157 69                        // sk05_mlp_gateup.py:157:69
	mul.lo.s32 	%r110, %r98, %r67;
	mul.lo.s32 	%r111, %r99, %r67;
	mul.lo.s32 	%r112, %r100, %r67;
	mul.lo.s32 	%r113, %r101, %r67;
	.loc	1 157 51                        // sk05_mlp_gateup.py:157:51
	cvt.s64.s32 	%rd6, %r110;
	add.s64 	%rd38, %rd55, %rd6;
	cvt.s64.s32 	%rd7, %r111;
	add.s64 	%rd39, %rd55, %rd7;
	cvt.s64.s32 	%rd8, %r112;
	add.s64 	%rd40, %rd55, %rd8;
	cvt.s64.s32 	%rd9, %r113;
	add.s64 	%rd41, %rd55, %rd9;
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	setp.gt.s32 	%p1, %r27, 127;
	.loc	1 163 30                        // sk05_mlp_gateup.py:163:30
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
	.loc	1 163 47                        // sk05_mlp_gateup.py:163:47
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
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	setp.gt.s32 	%p2, %r27, 255;
	.loc	1 164 18                        // sk05_mlp_gateup.py:164:18
	add.s64 	%rd42, %rd34, 128;
	add.s64 	%rd43, %rd35, 128;
	add.s64 	%rd44, %rd36, 128;
	add.s64 	%rd45, %rd37, 128;
	.loc	1 165 18                        // sk05_mlp_gateup.py:165:18
	add.s64 	%rd46, %rd38, 128;
	add.s64 	%rd47, %rd39, 128;
	add.s64 	%rd48, %rd40, 128;
	add.s64 	%rd49, %rd41, 128;
	.loc	1 163 30                        // sk05_mlp_gateup.py:163:30
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
	.loc	1 163 47                        // sk05_mlp_gateup.py:163:47
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
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
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
	cvt.s64.s32 	%rd56, %r117;
	add.s64 	%rd10, %rd50, %rd56;
	cvt.s64.s32 	%rd57, %r121;
	add.s64 	%rd11, %rd50, %rd57;
	cvt.s64.s32 	%rd58, %r125;
	add.s64 	%rd12, %rd50, %rd58;
	cvt.s64.s32 	%rd59, %r129;
	add.s64 	%rd13, %rd50, %rd59;
	cvt.s64.s32 	%rd60, %r133;
	add.s64 	%rd14, %rd50, %rd60;
	cvt.s64.s32 	%rd61, %r137;
	add.s64 	%rd15, %rd50, %rd61;
	cvt.s64.s32 	%rd62, %r141;
	add.s64 	%rd16, %rd50, %rd62;
	cvt.s64.s32 	%rd63, %r145;
	add.s64 	%rd17, %rd50, %rd63;
	.loc	1 161 28                        // sk05_mlp_gateup.py:161:28
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
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	cvt.s64.s32 	%rd18, %r151;
	and.b32 	%r160, %r27, -128;
	cvt.u64.u32 	%rd19, %r160;
	add.s64 	%rd64, %rd5, %rd9;
	add.s64 	%rd65, %rd64, %rd29;
	add.s64 	%rd20, %rd65, 256;
	add.s64 	%rd66, %rd5, %rd8;
	add.s64 	%rd67, %rd66, %rd29;
	add.s64 	%rd21, %rd67, 256;
	add.s64 	%rd68, %rd5, %rd7;
	add.s64 	%rd69, %rd68, %rd29;
	add.s64 	%rd22, %rd69, 256;
	add.s64 	%rd70, %rd5, %rd6;
	add.s64 	%rd71, %rd70, %rd29;
	add.s64 	%rd23, %rd71, 256;
	add.s64 	%rd72, %rd5, %rd4;
	add.s64 	%rd73, %rd72, %rd28;
	add.s64 	%rd24, %rd73, 256;
	add.s64 	%rd74, %rd5, %rd3;
	add.s64 	%rd75, %rd74, %rd28;
	add.s64 	%rd25, %rd75, 256;
	add.s64 	%rd76, %rd5, %rd2;
	add.s64 	%rd77, %rd76, %rd28;
	add.s64 	%rd26, %rd77, 256;
	add.s64 	%rd78, %rd5, %rd1;
	add.s64 	%rd79, %rd78, %rd28;
	add.s64 	%rd27, %rd79, 256;
	mov.b32 	%r910, 0f00000000;
	mov.b32 	%r907, 1;
	mov.b32 	%r906, -1;
	mov.b64 	%rd210, 0;
	mov.b32 	%r161, 0;
	mov.b32 	%r905, %r161;
	mov.b64 	%rd211, %rd210;
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
	setp.lt.s64 	%p3, %rd211, %rd18;
	add.s32 	%r331, %r906, 1;
	setp.gt.s32 	%p4, %r331, 1;
	selp.b32 	%r906, 0, %r331, %p4;
	.loc	1 162 39                        // sk05_mlp_gateup.py:162:39
	cvt.s64.s32 	%rd96, %r905;
	add.s64 	%rd80, %rd10, %rd96;
	add.s64 	%rd81, %rd11, %rd96;
	add.s64 	%rd82, %rd12, %rd96;
	add.s64 	%rd83, %rd13, %rd96;
	add.s64 	%rd84, %rd14, %rd96;
	add.s64 	%rd85, %rd15, %rd96;
	add.s64 	%rd86, %rd16, %rd96;
	add.s64 	%rd87, %rd17, %rd96;
	.loc	1 162 29                        // sk05_mlp_gateup.py:162:29
	// begin inline asm
	mov.u16 %rs1, 0x0;
	ld.global.b8 { %rs1 }, [ %rd80 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs9, %rs1;
	// begin inline asm
	mov.u16 %rs2, 0x0;
	ld.global.b8 { %rs2 }, [ %rd81 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs10, %rs2;
	// begin inline asm
	mov.u16 %rs3, 0x0;
	ld.global.b8 { %rs3 }, [ %rd82 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs11, %rs3;
	// begin inline asm
	mov.u16 %rs4, 0x0;
	ld.global.b8 { %rs4 }, [ %rd83 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs12, %rs4;
	// begin inline asm
	mov.u16 %rs5, 0x0;
	ld.global.b8 { %rs5 }, [ %rd84 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs13, %rs5;
	// begin inline asm
	mov.u16 %rs6, 0x0;
	ld.global.b8 { %rs6 }, [ %rd85 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs14, %rs6;
	// begin inline asm
	mov.u16 %rs7, 0x0;
	ld.global.b8 { %rs7 }, [ %rd86 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs15, %rs7;
	// begin inline asm
	mov.u16 %rs8, 0x0;
	ld.global.b8 { %rs8 }, [ %rd87 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs16, %rs8;
	.loc	1 162 63                        // sk05_mlp_gateup.py:162:63
	cvt.rn.f32.s16 	%r332, %rs9;
	cvt.rn.f32.s16 	%r333, %rs10;
	cvt.rn.f32.s16 	%r334, %rs11;
	cvt.rn.f32.s16 	%r335, %rs12;
	cvt.rn.f32.s16 	%r336, %rs13;
	cvt.rn.f32.s16 	%r337, %rs14;
	cvt.rn.f32.s16 	%r338, %rs15;
	cvt.rn.f32.s16 	%r339, %rs16;
	.loc	1 162 21                        // sk05_mlp_gateup.py:162:21
	ex2.approx.ftz.f32 	%r340, %r332;
	ex2.approx.ftz.f32 	%r341, %r333;
	ex2.approx.ftz.f32 	%r342, %r334;
	ex2.approx.ftz.f32 	%r343, %r335;
	ex2.approx.ftz.f32 	%r344, %r336;
	ex2.approx.ftz.f32 	%r345, %r337;
	ex2.approx.ftz.f32 	%r346, %r338;
	ex2.approx.ftz.f32 	%r347, %r339;
	.loc	1 163 30                        // sk05_mlp_gateup.py:163:30
	cp.async.wait_group 	2;
	bar.sync 	0;
	shl.b32 	%r348, %r906, 14;
	add.s32 	%r349, %r149, %r348;
	add.s32 	%r350, %r349, %r19;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r162, %r163, %r164, %r165}, [%r350];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r174, %r175, %r176, %r177}, [%r350+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r178, %r179, %r180, %r181}, [%r350+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r182, %r183, %r184, %r185}, [%r350+12288];
	add.s32 	%r351, %r349, %r20;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r190, %r191, %r192, %r193}, [%r351];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r218, %r219, %r220, %r221}, [%r351+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r238, %r239, %r240, %r241}, [%r351+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r258, %r259, %r260, %r261}, [%r351+12288];
	add.s32 	%r352, %r349, %r21;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r274, %r275, %r276, %r277}, [%r352];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r286, %r287, %r288, %r289}, [%r352+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r290, %r291, %r292, %r293}, [%r352+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r294, %r295, %r296, %r297}, [%r352+12288];
	add.s32 	%r353, %r349, %r22;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r298, %r299, %r300, %r301}, [%r353];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r310, %r311, %r312, %r313}, [%r353+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r314, %r315, %r316, %r317}, [%r353+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r318, %r319, %r320, %r321}, [%r353+12288];
	.loc	1 163 47                        // sk05_mlp_gateup.py:163:47
	add.s32 	%r354, %r349, %r23;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r166, %r167, %r194, %r195}, [%r354+32768];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r168, %r169, %r200, %r201}, [%r354+36864];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r170, %r171, %r206, %r207}, [%r354+40960];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r172, %r173, %r212, %r213}, [%r354+45056];
	add.s32 	%r355, %r349, %r24;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r278, %r279, %r302, %r303}, [%r355+32768];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r280, %r281, %r304, %r305}, [%r355+36864];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r282, %r283, %r306, %r307}, [%r355+40960];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r284, %r285, %r308, %r309}, [%r355+45056];
	.loc	1 163 39                        // sk05_mlp_gateup.py:163:39
	mov.b32 	%r186, %r161;
	mov.b32 	%r187, %r161;
	mov.b32 	%r188, %r161;
	mov.b32 	%r189, %r161;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r186, %r187, %r188, %r189 }, { %r162, %r163, %r164, %r165 }, { %r166, %r167 }, { %r186, %r187, %r188, %r189 };
	// end inline asm
	mov.b32 	%r196, %r161;
	mov.b32 	%r197, %r161;
	mov.b32 	%r198, %r161;
	mov.b32 	%r199, %r161;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r196, %r197, %r198, %r199 }, { %r162, %r163, %r164, %r165 }, { %r168, %r169 }, { %r196, %r197, %r198, %r199 };
	// end inline asm
	mov.b32 	%r202, %r161;
	mov.b32 	%r203, %r161;
	mov.b32 	%r204, %r161;
	mov.b32 	%r205, %r161;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r202, %r203, %r204, %r205 }, { %r162, %r163, %r164, %r165 }, { %r170, %r171 }, { %r202, %r203, %r204, %r205 };
	// end inline asm
	mov.b32 	%r208, %r161;
	mov.b32 	%r209, %r161;
	mov.b32 	%r210, %r161;
	mov.b32 	%r211, %r161;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r208, %r209, %r210, %r211 }, { %r162, %r163, %r164, %r165 }, { %r172, %r173 }, { %r208, %r209, %r210, %r211 };
	// end inline asm
	mov.b32 	%r214, %r161;
	mov.b32 	%r215, %r161;
	mov.b32 	%r216, %r161;
	mov.b32 	%r217, %r161;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r214, %r215, %r216, %r217 }, { %r174, %r175, %r176, %r177 }, { %r166, %r167 }, { %r214, %r215, %r216, %r217 };
	// end inline asm
	mov.b32 	%r222, %r161;
	mov.b32 	%r223, %r161;
	mov.b32 	%r224, %r161;
	mov.b32 	%r225, %r161;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r222, %r223, %r224, %r225 }, { %r174, %r175, %r176, %r177 }, { %r168, %r169 }, { %r222, %r223, %r224, %r225 };
	// end inline asm
	mov.b32 	%r226, %r161;
	mov.b32 	%r227, %r161;
	mov.b32 	%r228, %r161;
	mov.b32 	%r229, %r161;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r226, %r227, %r228, %r229 }, { %r174, %r175, %r176, %r177 }, { %r170, %r171 }, { %r226, %r227, %r228, %r229 };
	// end inline asm
	mov.b32 	%r230, %r161;
	mov.b32 	%r231, %r161;
	mov.b32 	%r232, %r161;
	mov.b32 	%r233, %r161;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r230, %r231, %r232, %r233 }, { %r174, %r175, %r176, %r177 }, { %r172, %r173 }, { %r230, %r231, %r232, %r233 };
	// end inline asm
	mov.b32 	%r234, %r161;
	mov.b32 	%r235, %r161;
	mov.b32 	%r236, %r161;
	mov.b32 	%r237, %r161;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r234, %r235, %r236, %r237 }, { %r178, %r179, %r180, %r181 }, { %r166, %r167 }, { %r234, %r235, %r236, %r237 };
	// end inline asm
	mov.b32 	%r242, %r161;
	mov.b32 	%r243, %r161;
	mov.b32 	%r244, %r161;
	mov.b32 	%r245, %r161;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r242, %r243, %r244, %r245 }, { %r178, %r179, %r180, %r181 }, { %r168, %r169 }, { %r242, %r243, %r244, %r245 };
	// end inline asm
	mov.b32 	%r246, %r161;
	mov.b32 	%r247, %r161;
	mov.b32 	%r248, %r161;
	mov.b32 	%r249, %r161;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r246, %r247, %r248, %r249 }, { %r178, %r179, %r180, %r181 }, { %r170, %r171 }, { %r246, %r247, %r248, %r249 };
	// end inline asm
	mov.b32 	%r250, %r161;
	mov.b32 	%r251, %r161;
	mov.b32 	%r252, %r161;
	mov.b32 	%r253, %r161;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r250, %r251, %r252, %r253 }, { %r178, %r179, %r180, %r181 }, { %r172, %r173 }, { %r250, %r251, %r252, %r253 };
	// end inline asm
	mov.b32 	%r254, %r161;
	mov.b32 	%r255, %r161;
	mov.b32 	%r256, %r161;
	mov.b32 	%r257, %r161;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r254, %r255, %r256, %r257 }, { %r182, %r183, %r184, %r185 }, { %r166, %r167 }, { %r254, %r255, %r256, %r257 };
	// end inline asm
	mov.b32 	%r262, %r161;
	mov.b32 	%r263, %r161;
	mov.b32 	%r264, %r161;
	mov.b32 	%r265, %r161;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r262, %r263, %r264, %r265 }, { %r182, %r183, %r184, %r185 }, { %r168, %r169 }, { %r262, %r263, %r264, %r265 };
	// end inline asm
	mov.b32 	%r266, %r161;
	mov.b32 	%r267, %r161;
	mov.b32 	%r268, %r161;
	mov.b32 	%r269, %r161;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r266, %r267, %r268, %r269 }, { %r182, %r183, %r184, %r185 }, { %r170, %r171 }, { %r266, %r267, %r268, %r269 };
	// end inline asm
	mov.b32 	%r273, %r161;
	mov.b32 	%r270, %r161;
	mov.b32 	%r271, %r161;
	mov.b32 	%r272, %r161;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r270, %r271, %r272, %r273 }, { %r182, %r183, %r184, %r185 }, { %r172, %r173 }, { %r270, %r271, %r272, %r273 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r186, %r187, %r188, %r189 }, { %r190, %r191, %r192, %r193 }, { %r194, %r195 }, { %r186, %r187, %r188, %r189 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r196, %r197, %r198, %r199 }, { %r190, %r191, %r192, %r193 }, { %r200, %r201 }, { %r196, %r197, %r198, %r199 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r202, %r203, %r204, %r205 }, { %r190, %r191, %r192, %r193 }, { %r206, %r207 }, { %r202, %r203, %r204, %r205 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r208, %r209, %r210, %r211 }, { %r190, %r191, %r192, %r193 }, { %r212, %r213 }, { %r208, %r209, %r210, %r211 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r214, %r215, %r216, %r217 }, { %r218, %r219, %r220, %r221 }, { %r194, %r195 }, { %r214, %r215, %r216, %r217 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r222, %r223, %r224, %r225 }, { %r218, %r219, %r220, %r221 }, { %r200, %r201 }, { %r222, %r223, %r224, %r225 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r226, %r227, %r228, %r229 }, { %r218, %r219, %r220, %r221 }, { %r206, %r207 }, { %r226, %r227, %r228, %r229 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r230, %r231, %r232, %r233 }, { %r218, %r219, %r220, %r221 }, { %r212, %r213 }, { %r230, %r231, %r232, %r233 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r234, %r235, %r236, %r237 }, { %r238, %r239, %r240, %r241 }, { %r194, %r195 }, { %r234, %r235, %r236, %r237 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r242, %r243, %r244, %r245 }, { %r238, %r239, %r240, %r241 }, { %r200, %r201 }, { %r242, %r243, %r244, %r245 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r246, %r247, %r248, %r249 }, { %r238, %r239, %r240, %r241 }, { %r206, %r207 }, { %r246, %r247, %r248, %r249 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r250, %r251, %r252, %r253 }, { %r238, %r239, %r240, %r241 }, { %r212, %r213 }, { %r250, %r251, %r252, %r253 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r254, %r255, %r256, %r257 }, { %r258, %r259, %r260, %r261 }, { %r194, %r195 }, { %r254, %r255, %r256, %r257 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r262, %r263, %r264, %r265 }, { %r258, %r259, %r260, %r261 }, { %r200, %r201 }, { %r262, %r263, %r264, %r265 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r266, %r267, %r268, %r269 }, { %r258, %r259, %r260, %r261 }, { %r206, %r207 }, { %r266, %r267, %r268, %r269 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r270, %r271, %r272, %r273 }, { %r258, %r259, %r260, %r261 }, { %r212, %r213 }, { %r270, %r271, %r272, %r273 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r186, %r187, %r188, %r189 }, { %r274, %r275, %r276, %r277 }, { %r278, %r279 }, { %r186, %r187, %r188, %r189 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r196, %r197, %r198, %r199 }, { %r274, %r275, %r276, %r277 }, { %r280, %r281 }, { %r196, %r197, %r198, %r199 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r202, %r203, %r204, %r205 }, { %r274, %r275, %r276, %r277 }, { %r282, %r283 }, { %r202, %r203, %r204, %r205 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r208, %r209, %r210, %r211 }, { %r274, %r275, %r276, %r277 }, { %r284, %r285 }, { %r208, %r209, %r210, %r211 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r214, %r215, %r216, %r217 }, { %r286, %r287, %r288, %r289 }, { %r278, %r279 }, { %r214, %r215, %r216, %r217 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r222, %r223, %r224, %r225 }, { %r286, %r287, %r288, %r289 }, { %r280, %r281 }, { %r222, %r223, %r224, %r225 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r226, %r227, %r228, %r229 }, { %r286, %r287, %r288, %r289 }, { %r282, %r283 }, { %r226, %r227, %r228, %r229 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r230, %r231, %r232, %r233 }, { %r286, %r287, %r288, %r289 }, { %r284, %r285 }, { %r230, %r231, %r232, %r233 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r234, %r235, %r236, %r237 }, { %r290, %r291, %r292, %r293 }, { %r278, %r279 }, { %r234, %r235, %r236, %r237 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r242, %r243, %r244, %r245 }, { %r290, %r291, %r292, %r293 }, { %r280, %r281 }, { %r242, %r243, %r244, %r245 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r246, %r247, %r248, %r249 }, { %r290, %r291, %r292, %r293 }, { %r282, %r283 }, { %r246, %r247, %r248, %r249 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r250, %r251, %r252, %r253 }, { %r290, %r291, %r292, %r293 }, { %r284, %r285 }, { %r250, %r251, %r252, %r253 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r254, %r255, %r256, %r257 }, { %r294, %r295, %r296, %r297 }, { %r278, %r279 }, { %r254, %r255, %r256, %r257 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r262, %r263, %r264, %r265 }, { %r294, %r295, %r296, %r297 }, { %r280, %r281 }, { %r262, %r263, %r264, %r265 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r266, %r267, %r268, %r269 }, { %r294, %r295, %r296, %r297 }, { %r282, %r283 }, { %r266, %r267, %r268, %r269 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r270, %r271, %r272, %r273 }, { %r294, %r295, %r296, %r297 }, { %r284, %r285 }, { %r270, %r271, %r272, %r273 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r186, %r187, %r188, %r189 }, { %r298, %r299, %r300, %r301 }, { %r302, %r303 }, { %r186, %r187, %r188, %r189 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r196, %r197, %r198, %r199 }, { %r298, %r299, %r300, %r301 }, { %r304, %r305 }, { %r196, %r197, %r198, %r199 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r202, %r203, %r204, %r205 }, { %r298, %r299, %r300, %r301 }, { %r306, %r307 }, { %r202, %r203, %r204, %r205 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r208, %r209, %r210, %r211 }, { %r298, %r299, %r300, %r301 }, { %r308, %r309 }, { %r208, %r209, %r210, %r211 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r214, %r215, %r216, %r217 }, { %r310, %r311, %r312, %r313 }, { %r302, %r303 }, { %r214, %r215, %r216, %r217 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r222, %r223, %r224, %r225 }, { %r310, %r311, %r312, %r313 }, { %r304, %r305 }, { %r222, %r223, %r224, %r225 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r226, %r227, %r228, %r229 }, { %r310, %r311, %r312, %r313 }, { %r306, %r307 }, { %r226, %r227, %r228, %r229 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r230, %r231, %r232, %r233 }, { %r310, %r311, %r312, %r313 }, { %r308, %r309 }, { %r230, %r231, %r232, %r233 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r234, %r235, %r236, %r237 }, { %r314, %r315, %r316, %r317 }, { %r302, %r303 }, { %r234, %r235, %r236, %r237 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r242, %r243, %r244, %r245 }, { %r314, %r315, %r316, %r317 }, { %r304, %r305 }, { %r242, %r243, %r244, %r245 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r246, %r247, %r248, %r249 }, { %r314, %r315, %r316, %r317 }, { %r306, %r307 }, { %r246, %r247, %r248, %r249 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r250, %r251, %r252, %r253 }, { %r314, %r315, %r316, %r317 }, { %r308, %r309 }, { %r250, %r251, %r252, %r253 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r254, %r255, %r256, %r257 }, { %r318, %r319, %r320, %r321 }, { %r302, %r303 }, { %r254, %r255, %r256, %r257 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r262, %r263, %r264, %r265 }, { %r318, %r319, %r320, %r321 }, { %r304, %r305 }, { %r262, %r263, %r264, %r265 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r266, %r267, %r268, %r269 }, { %r318, %r319, %r320, %r321 }, { %r306, %r307 }, { %r266, %r267, %r268, %r269 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r270, %r271, %r272, %r273 }, { %r318, %r319, %r320, %r321 }, { %r308, %r309 }, { %r270, %r271, %r272, %r273 };
	// end inline asm
	.loc	1 163 79                        // sk05_mlp_gateup.py:163:79
	cvt.rn.f32.s32 	%r356, %r270;
	cvt.rn.f32.s32 	%r357, %r271;
	cvt.rn.f32.s32 	%r358, %r272;
	cvt.rn.f32.s32 	%r359, %r273;
	cvt.rn.f32.s32 	%r360, %r266;
	cvt.rn.f32.s32 	%r361, %r267;
	cvt.rn.f32.s32 	%r362, %r268;
	cvt.rn.f32.s32 	%r363, %r269;
	cvt.rn.f32.s32 	%r364, %r262;
	cvt.rn.f32.s32 	%r365, %r263;
	cvt.rn.f32.s32 	%r366, %r264;
	cvt.rn.f32.s32 	%r367, %r265;
	cvt.rn.f32.s32 	%r368, %r254;
	cvt.rn.f32.s32 	%r369, %r255;
	cvt.rn.f32.s32 	%r370, %r256;
	cvt.rn.f32.s32 	%r371, %r257;
	cvt.rn.f32.s32 	%r372, %r250;
	cvt.rn.f32.s32 	%r373, %r251;
	cvt.rn.f32.s32 	%r374, %r252;
	cvt.rn.f32.s32 	%r375, %r253;
	cvt.rn.f32.s32 	%r376, %r246;
	cvt.rn.f32.s32 	%r377, %r247;
	cvt.rn.f32.s32 	%r378, %r248;
	cvt.rn.f32.s32 	%r379, %r249;
	cvt.rn.f32.s32 	%r380, %r242;
	cvt.rn.f32.s32 	%r381, %r243;
	cvt.rn.f32.s32 	%r382, %r244;
	cvt.rn.f32.s32 	%r383, %r245;
	cvt.rn.f32.s32 	%r384, %r234;
	cvt.rn.f32.s32 	%r385, %r235;
	cvt.rn.f32.s32 	%r386, %r236;
	cvt.rn.f32.s32 	%r387, %r237;
	cvt.rn.f32.s32 	%r388, %r230;
	cvt.rn.f32.s32 	%r389, %r231;
	cvt.rn.f32.s32 	%r390, %r232;
	cvt.rn.f32.s32 	%r391, %r233;
	cvt.rn.f32.s32 	%r392, %r226;
	cvt.rn.f32.s32 	%r393, %r227;
	cvt.rn.f32.s32 	%r394, %r228;
	cvt.rn.f32.s32 	%r395, %r229;
	cvt.rn.f32.s32 	%r396, %r222;
	cvt.rn.f32.s32 	%r397, %r223;
	cvt.rn.f32.s32 	%r398, %r224;
	cvt.rn.f32.s32 	%r399, %r225;
	cvt.rn.f32.s32 	%r400, %r214;
	cvt.rn.f32.s32 	%r401, %r215;
	cvt.rn.f32.s32 	%r402, %r216;
	cvt.rn.f32.s32 	%r403, %r217;
	cvt.rn.f32.s32 	%r404, %r208;
	cvt.rn.f32.s32 	%r405, %r209;
	cvt.rn.f32.s32 	%r406, %r210;
	cvt.rn.f32.s32 	%r407, %r211;
	cvt.rn.f32.s32 	%r408, %r202;
	cvt.rn.f32.s32 	%r409, %r203;
	cvt.rn.f32.s32 	%r410, %r204;
	cvt.rn.f32.s32 	%r411, %r205;
	cvt.rn.f32.s32 	%r412, %r196;
	cvt.rn.f32.s32 	%r413, %r197;
	cvt.rn.f32.s32 	%r414, %r198;
	cvt.rn.f32.s32 	%r415, %r199;
	cvt.rn.f32.s32 	%r416, %r186;
	cvt.rn.f32.s32 	%r417, %r187;
	cvt.rn.f32.s32 	%r418, %r188;
	cvt.rn.f32.s32 	%r419, %r189;
	.loc	1 163 15                        // sk05_mlp_gateup.py:163:15
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
	.loc	1 164 18                        // sk05_mlp_gateup.py:164:18
	add.s64 	%rd88, %rd27, %rd210;
	add.s64 	%rd89, %rd26, %rd210;
	add.s64 	%rd90, %rd25, %rd210;
	.loc	1 165 18                        // sk05_mlp_gateup.py:165:18
	add.s64 	%rd91, %rd24, %rd210;
	add.s64 	%rd92, %rd23, %rd210;
	add.s64 	%rd93, %rd22, %rd210;
	add.s64 	%rd94, %rd21, %rd210;
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	add.s64 	%rd95, %rd20, %rd210;
	add.s32 	%r420, %r907, 1;
	setp.gt.s32 	%p5, %r420, 1;
	selp.b32 	%r907, 0, %r420, %p5;
	.loc	1 163 30                        // sk05_mlp_gateup.py:163:30
	shl.b32 	%r421, %r907, 14;
	bar.sync 	0;
	add.s32 	%r322, %r32, %r421;
	selp.b32 	%r323, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r322 + 0 ], [ %rd88 + 0 ], 0x10, %r323;
	// end inline asm
	add.s32 	%r324, %r322, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r324 + 0 ], [ %rd89 + 0 ], 0x10, %r323;
	// end inline asm
	add.s32 	%r325, %r322, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r325 + 0 ], [ %rd90 + 0 ], 0x10, %r323;
	// end inline asm
	add.s32 	%r326, %r322, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r326 + 0 ], [ %rd91 + 0 ], 0x10, %r323;
	// end inline asm
	cp.async.commit_group;
	.loc	1 163 47                        // sk05_mlp_gateup.py:163:47
	add.s32 	%r327, %r322, 32768;
	// begin inline asm
	cp.async.cg.shared.global [ %r327 + 0 ], [ %rd92 + 0 ], 0x10, %r323;
	// end inline asm
	add.s32 	%r328, %r322, 36864;
	// begin inline asm
	cp.async.cg.shared.global [ %r328 + 0 ], [ %rd93 + 0 ], 0x10, %r323;
	// end inline asm
	add.s32 	%r329, %r322, 40960;
	// begin inline asm
	cp.async.cg.shared.global [ %r329 + 0 ], [ %rd94 + 0 ], 0x10, %r323;
	// end inline asm
	add.s32 	%r330, %r322, 45056;
	// begin inline asm
	cp.async.cg.shared.global [ %r330 + 0 ], [ %rd95 + 0 ], 0x10, %r323;
	// end inline asm
	cp.async.commit_group;
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	add.s64 	%rd211, %rd211, 1;
	add.s64 	%rd210, %rd210, 128;
	add.s32 	%r905, %r905, %r31;
	setp.ne.b64 	%p6, %rd19, %rd210;
	@%p6 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_4;
$L__BB0_1:                              // %.._crit_edge_crit_edge
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	shl.b32 	%r909, %r2, 1;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
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
	.loc	1 152 45                        // sk05_mlp_gateup.py:152:45
	or.b32 	%r544, %r12, %r904;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r545, %r544, 15;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r546, %r545, %r26;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r547, %r544, 14;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r548, %r547, %r26;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r549, %r544, 13;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r550, %r549, %r26;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r551, %r544, 12;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r552, %r551, %r26;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r553, %r544, 11;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r554, %r553, %r26;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r555, %r544, 10;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r556, %r555, %r26;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r557, %r544, 9;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r558, %r557, %r26;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r559, %r544, 8;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r560, %r559, %r26;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r561, %r544, 7;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r562, %r561, %r26;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r563, %r544, 6;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r564, %r563, %r26;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r565, %r544, 5;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r566, %r565, %r26;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r567, %r544, 4;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r568, %r567, %r26;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r569, %r544, 3;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r570, %r569, %r26;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r571, %r544, 2;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r572, %r571, %r26;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r573, %r544, 1;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r574, %r573, %r26;
	rem.s32 	%r575, %r544, %r26;
	.loc	1 152 45                        // sk05_mlp_gateup.py:152:45
	shl.b32 	%r576, %r7, 3;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r577, %r12, %r576;
	.loc	1 152 45                        // sk05_mlp_gateup.py:152:45
	and.b32 	%r578, %r2, 128;
	shr.u32 	%r579, %r578, 3;
	shr.u32 	%r580, %r2, 2;
	bfe.u32 	%r581, %r2, 2, 3;
	or.b32 	%r582, %r579, %r581;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r583, %r582, %r1;
	or.b32 	%r584, %r583, 104;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r585, %r584, %r25;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r586, %r583, 96;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r587, %r586, %r25;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r588, %r583, 72;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r589, %r588, %r25;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r590, %r583, 64;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r591, %r590, %r25;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r592, %r583, 40;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r593, %r592, %r25;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r594, %r583, 32;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r595, %r594, %r25;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r596, %r583, 8;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r597, %r596, %r25;
	rem.s32 	%r598, %r583, %r25;
	.loc	1 152 45                        // sk05_mlp_gateup.py:152:45
	shr.u32 	%r599, %r2, 4;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r600, %r599, %r1;
	or.b32 	%r601, %r600, 112;
	.loc	1 152 45                        // sk05_mlp_gateup.py:152:45
	bfe.u32 	%r602, %r2, 4, 4;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r603, %r602, %r1;
	or.b32 	%r604, %r603, 96;
	or.b32 	%r605, %r603, 80;
	or.b32 	%r606, %r603, 64;
	or.b32 	%r607, %r600, 48;
	or.b32 	%r608, %r603, 32;
	or.b32 	%r609, %r603, 16;
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 167 38                        // sk05_mlp_gateup.py:167:38
	mad.wide.s32 	%rd97, %r598, 4, %rd32;
	mad.wide.s32 	%rd98, %r597, 4, %rd32;
	mad.wide.s32 	%rd99, %r595, 4, %rd32;
	mad.wide.s32 	%rd100, %r593, 4, %rd32;
	mad.wide.s32 	%rd101, %r591, 4, %rd32;
	mad.wide.s32 	%rd102, %r589, 4, %rd32;
	mad.wide.s32 	%rd103, %r587, 4, %rd32;
	mad.wide.s32 	%rd104, %r585, 4, %rd32;
	.loc	1 167 24                        // sk05_mlp_gateup.py:167:24
	// begin inline asm
	mov.u32 %r422, 0x0;
	ld.global.b32 { %r422 }, [ %rd97 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r423, 0x0;
	ld.global.b32 { %r423 }, [ %rd98 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r424, 0x0;
	ld.global.b32 { %r424 }, [ %rd99 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r425, 0x0;
	ld.global.b32 { %r425 }, [ %rd100 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r426, 0x0;
	ld.global.b32 { %r426 }, [ %rd101 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r427, 0x0;
	ld.global.b32 { %r427 }, [ %rd102 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r428, 0x0;
	ld.global.b32 { %r428 }, [ %rd103 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r429, 0x0;
	ld.global.b32 { %r429 }, [ %rd104 + 0 ];
	// end inline asm
	.loc	1 168 38                        // sk05_mlp_gateup.py:168:38
	mad.wide.s32 	%rd105, %r13, 4, %rd33;
	mad.wide.s32 	%rd106, %r14, 4, %rd33;
	mad.wide.s32 	%rd107, %r15, 4, %rd33;
	mad.wide.s32 	%rd108, %r16, 4, %rd33;
	.loc	1 168 24                        // sk05_mlp_gateup.py:168:24
	// begin inline asm
	mov.u32 %r430, 0x0;
	mov.u32 %r431, 0x0;
	ld.global.v2.b32 { %r430, %r431 }, [ %rd105 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r432, 0x0;
	mov.u32 %r433, 0x0;
	ld.global.v2.b32 { %r432, %r433 }, [ %rd106 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r434, 0x0;
	mov.u32 %r435, 0x0;
	ld.global.v2.b32 { %r434, %r435 }, [ %rd107 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r436, 0x0;
	mov.u32 %r437, 0x0;
	ld.global.v2.b32 { %r436, %r437 }, [ %rd108 + 0 ];
	// end inline asm
	.loc	1 169 49                        // sk05_mlp_gateup.py:169:49
	mul.lo.s32 	%r610, %r8, %r29;
	mul.lo.s32 	%r611, %r9, %r29;
	mul.lo.s32 	%r612, %r10, %r29;
	mul.lo.s32 	%r613, %r11, %r29;
	.loc	1 169 31                        // sk05_mlp_gateup.py:169:31
	mad.wide.s32 	%rd181, %r610, 2, %rd31;
	mad.wide.s32 	%rd182, %r611, 2, %rd31;
	mad.wide.s32 	%rd183, %r612, 2, %rd31;
	mad.wide.s32 	%rd184, %r613, 2, %rd31;
	.loc	1 169 82                        // sk05_mlp_gateup.py:169:82
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
	.loc	1 169 64                        // sk05_mlp_gateup.py:169:64
	mul.wide.s32 	%rd185, %r614, 2;
	add.s64 	%rd109, %rd181, %rd185;
	mul.wide.s32 	%rd186, %r615, 2;
	add.s64 	%rd110, %rd181, %rd186;
	mul.wide.s32 	%rd187, %r616, 2;
	add.s64 	%rd111, %rd181, %rd187;
	mul.wide.s32 	%rd188, %r617, 2;
	add.s64 	%rd112, %rd181, %rd188;
	mul.wide.s32 	%rd189, %r618, 2;
	add.s64 	%rd113, %rd181, %rd189;
	mul.wide.s32 	%rd190, %r619, 2;
	add.s64 	%rd114, %rd181, %rd190;
	mul.wide.s32 	%rd191, %r620, 2;
	add.s64 	%rd115, %rd181, %rd191;
	mul.wide.s32 	%rd192, %r621, 2;
	add.s64 	%rd116, %rd181, %rd192;
	mul.wide.s32 	%rd193, %r622, 2;
	add.s64 	%rd117, %rd181, %rd193;
	mul.wide.s32 	%rd194, %r623, 2;
	add.s64 	%rd118, %rd181, %rd194;
	mul.wide.s32 	%rd195, %r624, 2;
	add.s64 	%rd119, %rd181, %rd195;
	mul.wide.s32 	%rd196, %r625, 2;
	add.s64 	%rd120, %rd181, %rd196;
	mul.wide.s32 	%rd197, %r626, 2;
	add.s64 	%rd121, %rd181, %rd197;
	mul.wide.s32 	%rd198, %r627, 2;
	add.s64 	%rd122, %rd181, %rd198;
	mul.wide.s32 	%rd199, %r628, 2;
	add.s64 	%rd123, %rd181, %rd199;
	mul.wide.s32 	%rd200, %r629, 2;
	add.s64 	%rd124, %rd181, %rd200;
	add.s64 	%rd125, %rd182, %rd185;
	add.s64 	%rd126, %rd182, %rd186;
	add.s64 	%rd127, %rd182, %rd187;
	add.s64 	%rd128, %rd182, %rd188;
	add.s64 	%rd129, %rd182, %rd189;
	add.s64 	%rd130, %rd182, %rd190;
	add.s64 	%rd131, %rd182, %rd191;
	add.s64 	%rd132, %rd182, %rd192;
	add.s64 	%rd133, %rd182, %rd193;
	add.s64 	%rd134, %rd182, %rd194;
	add.s64 	%rd135, %rd182, %rd195;
	add.s64 	%rd136, %rd182, %rd196;
	add.s64 	%rd137, %rd182, %rd197;
	add.s64 	%rd138, %rd182, %rd198;
	add.s64 	%rd139, %rd182, %rd199;
	add.s64 	%rd140, %rd182, %rd200;
	add.s64 	%rd141, %rd183, %rd185;
	add.s64 	%rd142, %rd183, %rd186;
	add.s64 	%rd143, %rd183, %rd187;
	add.s64 	%rd144, %rd183, %rd188;
	add.s64 	%rd145, %rd183, %rd189;
	add.s64 	%rd146, %rd183, %rd190;
	add.s64 	%rd147, %rd183, %rd191;
	add.s64 	%rd148, %rd183, %rd192;
	add.s64 	%rd149, %rd183, %rd193;
	add.s64 	%rd150, %rd183, %rd194;
	add.s64 	%rd151, %rd183, %rd195;
	add.s64 	%rd152, %rd183, %rd196;
	add.s64 	%rd153, %rd183, %rd197;
	add.s64 	%rd154, %rd183, %rd198;
	add.s64 	%rd155, %rd183, %rd199;
	add.s64 	%rd156, %rd183, %rd200;
	add.s64 	%rd157, %rd184, %rd185;
	add.s64 	%rd158, %rd184, %rd186;
	add.s64 	%rd159, %rd184, %rd187;
	add.s64 	%rd160, %rd184, %rd188;
	add.s64 	%rd161, %rd184, %rd189;
	add.s64 	%rd162, %rd184, %rd190;
	add.s64 	%rd163, %rd184, %rd191;
	add.s64 	%rd164, %rd184, %rd192;
	add.s64 	%rd165, %rd184, %rd193;
	add.s64 	%rd166, %rd184, %rd194;
	add.s64 	%rd167, %rd184, %rd195;
	add.s64 	%rd168, %rd184, %rd196;
	add.s64 	%rd169, %rd184, %rd197;
	add.s64 	%rd170, %rd184, %rd198;
	add.s64 	%rd171, %rd184, %rd199;
	add.s64 	%rd172, %rd184, %rd200;
	.loc	1 169 19                        // sk05_mlp_gateup.py:169:19
	// begin inline asm
	mov.u16 %rs17, 0x0;
	ld.global.b16 { %rs17 }, [ %rd109 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs18, 0x0;
	ld.global.b16 { %rs18 }, [ %rd110 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs19, 0x0;
	ld.global.b16 { %rs19 }, [ %rd111 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs20, 0x0;
	ld.global.b16 { %rs20 }, [ %rd112 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs21, 0x0;
	ld.global.b16 { %rs21 }, [ %rd113 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs22, 0x0;
	ld.global.b16 { %rs22 }, [ %rd114 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs23, 0x0;
	ld.global.b16 { %rs23 }, [ %rd115 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs24, 0x0;
	ld.global.b16 { %rs24 }, [ %rd116 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs25, 0x0;
	ld.global.b16 { %rs25 }, [ %rd117 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs26, 0x0;
	ld.global.b16 { %rs26 }, [ %rd118 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs27, 0x0;
	ld.global.b16 { %rs27 }, [ %rd119 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs28, 0x0;
	ld.global.b16 { %rs28 }, [ %rd120 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs29, 0x0;
	ld.global.b16 { %rs29 }, [ %rd121 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs30, 0x0;
	ld.global.b16 { %rs30 }, [ %rd122 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs31, 0x0;
	ld.global.b16 { %rs31 }, [ %rd123 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs32, 0x0;
	ld.global.b16 { %rs32 }, [ %rd124 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs33, 0x0;
	ld.global.b16 { %rs33 }, [ %rd125 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs34, 0x0;
	ld.global.b16 { %rs34 }, [ %rd126 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs35, 0x0;
	ld.global.b16 { %rs35 }, [ %rd127 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs36, 0x0;
	ld.global.b16 { %rs36 }, [ %rd128 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs37, 0x0;
	ld.global.b16 { %rs37 }, [ %rd129 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs38, 0x0;
	ld.global.b16 { %rs38 }, [ %rd130 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs39, 0x0;
	ld.global.b16 { %rs39 }, [ %rd131 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs40, 0x0;
	ld.global.b16 { %rs40 }, [ %rd132 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs41, 0x0;
	ld.global.b16 { %rs41 }, [ %rd133 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs42, 0x0;
	ld.global.b16 { %rs42 }, [ %rd134 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs43, 0x0;
	ld.global.b16 { %rs43 }, [ %rd135 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs44, 0x0;
	ld.global.b16 { %rs44 }, [ %rd136 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs45, 0x0;
	ld.global.b16 { %rs45 }, [ %rd137 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs46, 0x0;
	ld.global.b16 { %rs46 }, [ %rd138 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs47, 0x0;
	ld.global.b16 { %rs47 }, [ %rd139 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs48, 0x0;
	ld.global.b16 { %rs48 }, [ %rd140 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs49, 0x0;
	ld.global.b16 { %rs49 }, [ %rd141 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs50, 0x0;
	ld.global.b16 { %rs50 }, [ %rd142 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs51, 0x0;
	ld.global.b16 { %rs51 }, [ %rd143 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs52, 0x0;
	ld.global.b16 { %rs52 }, [ %rd144 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs53, 0x0;
	ld.global.b16 { %rs53 }, [ %rd145 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs54, 0x0;
	ld.global.b16 { %rs54 }, [ %rd146 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs55, 0x0;
	ld.global.b16 { %rs55 }, [ %rd147 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs56, 0x0;
	ld.global.b16 { %rs56 }, [ %rd148 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs57, 0x0;
	ld.global.b16 { %rs57 }, [ %rd149 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs58, 0x0;
	ld.global.b16 { %rs58 }, [ %rd150 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs59, 0x0;
	ld.global.b16 { %rs59 }, [ %rd151 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs60, 0x0;
	ld.global.b16 { %rs60 }, [ %rd152 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs61, 0x0;
	ld.global.b16 { %rs61 }, [ %rd153 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs62, 0x0;
	ld.global.b16 { %rs62 }, [ %rd154 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs63, 0x0;
	ld.global.b16 { %rs63 }, [ %rd155 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs64, 0x0;
	ld.global.b16 { %rs64 }, [ %rd156 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs65, 0x0;
	ld.global.b16 { %rs65 }, [ %rd157 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs66, 0x0;
	ld.global.b16 { %rs66 }, [ %rd158 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs67, 0x0;
	ld.global.b16 { %rs67 }, [ %rd159 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs68, 0x0;
	ld.global.b16 { %rs68 }, [ %rd160 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs69, 0x0;
	ld.global.b16 { %rs69 }, [ %rd161 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs70, 0x0;
	ld.global.b16 { %rs70 }, [ %rd162 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs71, 0x0;
	ld.global.b16 { %rs71 }, [ %rd163 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs72, 0x0;
	ld.global.b16 { %rs72 }, [ %rd164 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs73, 0x0;
	ld.global.b16 { %rs73 }, [ %rd165 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs74, 0x0;
	ld.global.b16 { %rs74 }, [ %rd166 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs75, 0x0;
	ld.global.b16 { %rs75 }, [ %rd167 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs76, 0x0;
	ld.global.b16 { %rs76 }, [ %rd168 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs77, 0x0;
	ld.global.b16 { %rs77 }, [ %rd169 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs78, 0x0;
	ld.global.b16 { %rs78 }, [ %rd170 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs79, 0x0;
	ld.global.b16 { %rs79 }, [ %rd171 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs80, 0x0;
	ld.global.b16 { %rs80 }, [ %rd172 + 0 ];
	// end inline asm
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	shl.b32 	%r630, %r18, 7;
	shl.b32 	%r631, %r3, 1;
	or.b32 	%r632, %r630, %r904;
	xor.b32 	%r633, %r632, %r631;
	add.s32 	%r438, %r149, %r633;
	mov.b32 	%r439, {%rs17, %rs18};
	mov.b32 	%r440, {%rs19, %rs20};
	mov.b32 	%r441, {%rs21, %rs22};
	mov.b32 	%r442, {%rs23, %rs24};
	// begin inline asm
	st.shared.v4.b32 [ %r438 + 0 ], { %r439, %r440, %r441, %r442 };
	// end inline asm
	add.s32 	%r443, %r438, 512;
	mov.b32 	%r444, {%rs25, %rs26};
	mov.b32 	%r445, {%rs27, %rs28};
	mov.b32 	%r446, {%rs29, %rs30};
	mov.b32 	%r447, {%rs31, %rs32};
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
	mov.b32 	%r448, {%rs33, %rs34};
	mov.b32 	%r449, {%rs35, %rs36};
	mov.b32 	%r450, {%rs37, %rs38};
	mov.b32 	%r451, {%rs39, %rs40};
	// begin inline asm
	st.shared.v4.b32 [ %r438 + 0 ], { %r448, %r449, %r450, %r451 };
	// end inline asm
	mov.b32 	%r452, {%rs41, %rs42};
	mov.b32 	%r453, {%rs43, %rs44};
	mov.b32 	%r454, {%rs45, %rs46};
	mov.b32 	%r455, {%rs47, %rs48};
	// begin inline asm
	st.shared.v4.b32 [ %r443 + 0 ], { %r452, %r453, %r454, %r455 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r652, %r653, %r654, %r655}, [%r641];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r656, %r657, %r658, %r659}, [%r647];
	bar.sync 	0;
	mov.b32 	%r456, {%rs49, %rs50};
	mov.b32 	%r457, {%rs51, %rs52};
	mov.b32 	%r458, {%rs53, %rs54};
	mov.b32 	%r459, {%rs55, %rs56};
	// begin inline asm
	st.shared.v4.b32 [ %r438 + 0 ], { %r456, %r457, %r458, %r459 };
	// end inline asm
	mov.b32 	%r460, {%rs57, %rs58};
	mov.b32 	%r461, {%rs59, %rs60};
	mov.b32 	%r462, {%rs61, %rs62};
	mov.b32 	%r463, {%rs63, %rs64};
	// begin inline asm
	st.shared.v4.b32 [ %r443 + 0 ], { %r460, %r461, %r462, %r463 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r660, %r661, %r662, %r663}, [%r641];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r664, %r665, %r666, %r667}, [%r647];
	bar.sync 	0;
	mov.b32 	%r464, {%rs65, %rs66};
	mov.b32 	%r465, {%rs67, %rs68};
	mov.b32 	%r466, {%rs69, %rs70};
	mov.b32 	%r467, {%rs71, %rs72};
	// begin inline asm
	st.shared.v4.b32 [ %r438 + 0 ], { %r464, %r465, %r466, %r467 };
	// end inline asm
	mov.b32 	%r468, {%rs73, %rs74};
	mov.b32 	%r469, {%rs75, %rs76};
	mov.b32 	%r470, {%rs77, %rs78};
	mov.b32 	%r471, {%rs79, %rs80};
	// begin inline asm
	st.shared.v4.b32 [ %r443 + 0 ], { %r468, %r469, %r470, %r471 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r668, %r669, %r670, %r671}, [%r641];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r672, %r673, %r674, %r675}, [%r647];
	.loc	1 176 31                        // sk05_mlp_gateup.py:176:31
	setp.lt.s32 	%p15, %r603, %r25;
	setp.lt.s32 	%p16, %r609, %r25;
	setp.lt.s32 	%p17, %r608, %r25;
	setp.lt.s32 	%p18, %r607, %r25;
	setp.lt.s32 	%p19, %r606, %r25;
	setp.lt.s32 	%p20, %r605, %r25;
	setp.lt.s32 	%p21, %r604, %r25;
	setp.lt.s32 	%p22, %r601, %r25;
	.loc	1 176 54                        // sk05_mlp_gateup.py:176:54
	setp.lt.s32 	%p23, %r577, %r26;
	.loc	1 176 37                        // sk05_mlp_gateup.py:176:37
	and.pred 	%p7, %p15, %p23;
	and.pred 	%p8, %p16, %p23;
	and.pred 	%p9, %p17, %p23;
	and.pred 	%p10, %p18, %p23;
	and.pred 	%p11, %p19, %p23;
	and.pred 	%p12, %p20, %p23;
	and.pred 	%p13, %p21, %p23;
	and.pred 	%p14, %p22, %p23;
	.loc	1 174 35                        // sk05_mlp_gateup.py:174:35
	mul.lo.s32 	%r676, %r603, %r28;
	mul.lo.s32 	%r677, %r609, %r28;
	mul.lo.s32 	%r678, %r608, %r28;
	mul.lo.s32 	%r679, %r607, %r28;
	mul.lo.s32 	%r680, %r606, %r28;
	mul.lo.s32 	%r681, %r605, %r28;
	mul.lo.s32 	%r682, %r604, %r28;
	mul.lo.s32 	%r683, %r601, %r28;
	.loc	1 174 18                        // sk05_mlp_gateup.py:174:18
	mad.wide.s32 	%rd201, %r676, 2, %rd30;
	mad.wide.s32 	%rd202, %r677, 2, %rd30;
	mad.wide.s32 	%rd203, %r678, 2, %rd30;
	mad.wide.s32 	%rd204, %r679, 2, %rd30;
	mad.wide.s32 	%rd205, %r680, 2, %rd30;
	mad.wide.s32 	%rd206, %r681, 2, %rd30;
	mad.wide.s32 	%rd207, %r682, 2, %rd30;
	mad.wide.s32 	%rd208, %r683, 2, %rd30;
	.loc	1 174 50                        // sk05_mlp_gateup.py:174:50
	mul.wide.s32 	%rd209, %r577, 2;
	add.s64 	%rd173, %rd201, %rd209;
	add.s64 	%rd174, %rd202, %rd209;
	add.s64 	%rd175, %rd203, %rd209;
	add.s64 	%rd176, %rd204, %rd209;
	add.s64 	%rd177, %rd205, %rd209;
	add.s64 	%rd178, %rd206, %rd209;
	add.s64 	%rd179, %rd207, %rd209;
	add.s64 	%rd180, %rd208, %rd209;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r684, %r959, %r428;
	mul.f32 	%r685, %r958, %r428;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs81, %rs82}, %r668;
	cvt.f32.bf16 	%r686, %rs82;
	cvt.f32.bf16 	%r687, %rs81;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r688, %r685, %r430, %r687;
	fma.rn.f32 	%r689, %r684, %r431, %r686;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r690, %r911, %r422;
	mul.f32 	%r691, %r910, %r422;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs83, %rs84}, %r642;
	cvt.f32.bf16 	%r692, %rs84;
	cvt.f32.bf16 	%r693, %rs83;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r694, %r691, %r430, %r693;
	fma.rn.f32 	%r695, %r690, %r431, %r692;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r473, %r695, %r694;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r696, %r913, %r423;
	mul.f32 	%r697, %r912, %r423;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs85, %rs86}, %r643;
	cvt.f32.bf16 	%r698, %rs86;
	cvt.f32.bf16 	%r699, %rs85;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r700, %r697, %r430, %r699;
	fma.rn.f32 	%r701, %r696, %r431, %r698;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r478, %r701, %r700;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r702, %r927, %r424;
	mul.f32 	%r703, %r926, %r424;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs87, %rs88}, %r652;
	cvt.f32.bf16 	%r704, %rs88;
	cvt.f32.bf16 	%r705, %rs87;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r706, %r703, %r430, %r705;
	fma.rn.f32 	%r707, %r702, %r431, %r704;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r474, %r707, %r706;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r708, %r929, %r425;
	mul.f32 	%r709, %r928, %r425;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs89, %rs90}, %r653;
	cvt.f32.bf16 	%r710, %rs90;
	cvt.f32.bf16 	%r711, %rs89;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r712, %r709, %r430, %r711;
	fma.rn.f32 	%r713, %r708, %r431, %r710;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r479, %r713, %r712;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r714, %r943, %r426;
	mul.f32 	%r715, %r942, %r426;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs91, %rs92}, %r660;
	cvt.f32.bf16 	%r716, %rs92;
	cvt.f32.bf16 	%r717, %rs91;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r718, %r715, %r430, %r717;
	fma.rn.f32 	%r719, %r714, %r431, %r716;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r475, %r719, %r718;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r720, %r945, %r427;
	mul.f32 	%r721, %r944, %r427;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs93, %rs94}, %r661;
	cvt.f32.bf16 	%r722, %rs94;
	cvt.f32.bf16 	%r723, %rs93;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r724, %r721, %r430, %r723;
	fma.rn.f32 	%r725, %r720, %r431, %r722;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r480, %r725, %r724;
	cvt.rn.bf16x2.f32 	%r476, %r689, %r688;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r726, %r961, %r429;
	mul.f32 	%r727, %r960, %r429;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs95, %rs96}, %r669;
	cvt.f32.bf16 	%r728, %rs96;
	cvt.f32.bf16 	%r729, %rs95;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r730, %r727, %r430, %r729;
	fma.rn.f32 	%r731, %r726, %r431, %r728;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r481, %r731, %r730;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r732, %r963, %r428;
	mul.f32 	%r733, %r962, %r428;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs97, %rs98}, %r670;
	cvt.f32.bf16 	%r734, %rs98;
	cvt.f32.bf16 	%r735, %rs97;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r736, %r733, %r432, %r735;
	fma.rn.f32 	%r737, %r732, %r433, %r734;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r738, %r915, %r422;
	mul.f32 	%r739, %r914, %r422;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs99, %rs100}, %r644;
	cvt.f32.bf16 	%r740, %rs100;
	cvt.f32.bf16 	%r741, %rs99;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r742, %r739, %r432, %r741;
	fma.rn.f32 	%r743, %r738, %r433, %r740;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r493, %r743, %r742;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r744, %r917, %r423;
	mul.f32 	%r745, %r916, %r423;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs101, %rs102}, %r645;
	cvt.f32.bf16 	%r746, %rs102;
	cvt.f32.bf16 	%r747, %rs101;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r748, %r745, %r432, %r747;
	fma.rn.f32 	%r749, %r744, %r433, %r746;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r498, %r749, %r748;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r750, %r931, %r424;
	mul.f32 	%r751, %r930, %r424;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs103, %rs104}, %r654;
	cvt.f32.bf16 	%r752, %rs104;
	cvt.f32.bf16 	%r753, %rs103;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r754, %r751, %r432, %r753;
	fma.rn.f32 	%r755, %r750, %r433, %r752;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r494, %r755, %r754;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r756, %r933, %r425;
	mul.f32 	%r757, %r932, %r425;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs105, %rs106}, %r655;
	cvt.f32.bf16 	%r758, %rs106;
	cvt.f32.bf16 	%r759, %rs105;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r760, %r757, %r432, %r759;
	fma.rn.f32 	%r761, %r756, %r433, %r758;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r499, %r761, %r760;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r762, %r947, %r426;
	mul.f32 	%r763, %r946, %r426;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs107, %rs108}, %r662;
	cvt.f32.bf16 	%r764, %rs108;
	cvt.f32.bf16 	%r765, %rs107;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r766, %r763, %r432, %r765;
	fma.rn.f32 	%r767, %r762, %r433, %r764;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r495, %r767, %r766;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r768, %r949, %r427;
	mul.f32 	%r769, %r948, %r427;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs109, %rs110}, %r663;
	cvt.f32.bf16 	%r770, %rs110;
	cvt.f32.bf16 	%r771, %rs109;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r772, %r769, %r432, %r771;
	fma.rn.f32 	%r773, %r768, %r433, %r770;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r500, %r773, %r772;
	cvt.rn.bf16x2.f32 	%r496, %r737, %r736;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r774, %r965, %r429;
	mul.f32 	%r775, %r964, %r429;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs111, %rs112}, %r671;
	cvt.f32.bf16 	%r776, %rs112;
	cvt.f32.bf16 	%r777, %rs111;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r778, %r775, %r432, %r777;
	fma.rn.f32 	%r779, %r774, %r433, %r776;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r501, %r779, %r778;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r780, %r967, %r428;
	mul.f32 	%r781, %r966, %r428;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs113, %rs114}, %r672;
	cvt.f32.bf16 	%r782, %rs114;
	cvt.f32.bf16 	%r783, %rs113;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r784, %r781, %r434, %r783;
	fma.rn.f32 	%r785, %r780, %r435, %r782;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r786, %r919, %r422;
	mul.f32 	%r787, %r918, %r422;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs115, %rs116}, %r648;
	cvt.f32.bf16 	%r788, %rs116;
	cvt.f32.bf16 	%r789, %rs115;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r790, %r787, %r434, %r789;
	fma.rn.f32 	%r791, %r786, %r435, %r788;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r483, %r791, %r790;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r792, %r921, %r423;
	mul.f32 	%r793, %r920, %r423;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs117, %rs118}, %r649;
	cvt.f32.bf16 	%r794, %rs118;
	cvt.f32.bf16 	%r795, %rs117;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r796, %r793, %r434, %r795;
	fma.rn.f32 	%r797, %r792, %r435, %r794;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r488, %r797, %r796;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r798, %r935, %r424;
	mul.f32 	%r799, %r934, %r424;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs119, %rs120}, %r656;
	cvt.f32.bf16 	%r800, %rs120;
	cvt.f32.bf16 	%r801, %rs119;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r802, %r799, %r434, %r801;
	fma.rn.f32 	%r803, %r798, %r435, %r800;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r484, %r803, %r802;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r804, %r937, %r425;
	mul.f32 	%r805, %r936, %r425;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs121, %rs122}, %r657;
	cvt.f32.bf16 	%r806, %rs122;
	cvt.f32.bf16 	%r807, %rs121;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r808, %r805, %r434, %r807;
	fma.rn.f32 	%r809, %r804, %r435, %r806;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r489, %r809, %r808;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r810, %r951, %r426;
	mul.f32 	%r811, %r950, %r426;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs123, %rs124}, %r664;
	cvt.f32.bf16 	%r812, %rs124;
	cvt.f32.bf16 	%r813, %rs123;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r814, %r811, %r434, %r813;
	fma.rn.f32 	%r815, %r810, %r435, %r812;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r485, %r815, %r814;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r816, %r953, %r427;
	mul.f32 	%r817, %r952, %r427;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs125, %rs126}, %r665;
	cvt.f32.bf16 	%r818, %rs126;
	cvt.f32.bf16 	%r819, %rs125;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r820, %r817, %r434, %r819;
	fma.rn.f32 	%r821, %r816, %r435, %r818;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r490, %r821, %r820;
	cvt.rn.bf16x2.f32 	%r486, %r785, %r784;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r822, %r969, %r429;
	mul.f32 	%r823, %r968, %r429;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs127, %rs128}, %r673;
	cvt.f32.bf16 	%r824, %rs128;
	cvt.f32.bf16 	%r825, %rs127;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r826, %r823, %r434, %r825;
	fma.rn.f32 	%r827, %r822, %r435, %r824;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r491, %r827, %r826;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r828, %r971, %r428;
	mul.f32 	%r829, %r970, %r428;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs129, %rs130}, %r674;
	cvt.f32.bf16 	%r830, %rs130;
	cvt.f32.bf16 	%r831, %rs129;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r832, %r829, %r436, %r831;
	fma.rn.f32 	%r833, %r828, %r437, %r830;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r834, %r923, %r422;
	mul.f32 	%r835, %r922, %r422;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs131, %rs132}, %r650;
	cvt.f32.bf16 	%r836, %rs132;
	cvt.f32.bf16 	%r837, %rs131;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r838, %r835, %r436, %r837;
	fma.rn.f32 	%r839, %r834, %r437, %r836;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r503, %r839, %r838;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r840, %r925, %r423;
	mul.f32 	%r841, %r924, %r423;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs133, %rs134}, %r651;
	cvt.f32.bf16 	%r842, %rs134;
	cvt.f32.bf16 	%r843, %rs133;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r844, %r841, %r436, %r843;
	fma.rn.f32 	%r845, %r840, %r437, %r842;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r508, %r845, %r844;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r846, %r939, %r424;
	mul.f32 	%r847, %r938, %r424;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs135, %rs136}, %r658;
	cvt.f32.bf16 	%r848, %rs136;
	cvt.f32.bf16 	%r849, %rs135;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r850, %r847, %r436, %r849;
	fma.rn.f32 	%r851, %r846, %r437, %r848;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r504, %r851, %r850;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r852, %r941, %r425;
	mul.f32 	%r853, %r940, %r425;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs137, %rs138}, %r659;
	cvt.f32.bf16 	%r854, %rs138;
	cvt.f32.bf16 	%r855, %rs137;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r856, %r853, %r436, %r855;
	fma.rn.f32 	%r857, %r852, %r437, %r854;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r509, %r857, %r856;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r858, %r955, %r426;
	mul.f32 	%r859, %r954, %r426;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs139, %rs140}, %r666;
	cvt.f32.bf16 	%r860, %rs140;
	cvt.f32.bf16 	%r861, %rs139;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r862, %r859, %r436, %r861;
	fma.rn.f32 	%r863, %r858, %r437, %r860;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r505, %r863, %r862;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r864, %r957, %r427;
	mul.f32 	%r865, %r956, %r427;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs141, %rs142}, %r667;
	cvt.f32.bf16 	%r866, %rs142;
	cvt.f32.bf16 	%r867, %rs141;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r868, %r865, %r436, %r867;
	fma.rn.f32 	%r869, %r864, %r437, %r866;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r510, %r869, %r868;
	cvt.rn.bf16x2.f32 	%r506, %r833, %r832;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r870, %r973, %r429;
	mul.f32 	%r871, %r972, %r429;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs143, %rs144}, %r675;
	cvt.f32.bf16 	%r872, %rs144;
	cvt.f32.bf16 	%r873, %rs143;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r874, %r871, %r436, %r873;
	fma.rn.f32 	%r875, %r870, %r437, %r872;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
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
	.loc	1 175 8                         // sk05_mlp_gateup.py:175:8
	// begin inline asm
	@%p7 st.global.v4.b32 [ %rd173 + 0 ], { %r512, %r513, %r514, %r515 };
	// end inline asm
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd174 + 0 ], { %r516, %r517, %r518, %r519 };
	// end inline asm
	// begin inline asm
	@%p9 st.global.v4.b32 [ %rd175 + 0 ], { %r520, %r521, %r522, %r523 };
	// end inline asm
	// begin inline asm
	@%p10 st.global.v4.b32 [ %rd176 + 0 ], { %r524, %r525, %r526, %r527 };
	// end inline asm
	// begin inline asm
	@%p11 st.global.v4.b32 [ %rd177 + 0 ], { %r528, %r529, %r530, %r531 };
	// end inline asm
	// begin inline asm
	@%p12 st.global.v4.b32 [ %rd178 + 0 ], { %r532, %r533, %r534, %r535 };
	// end inline asm
	// begin inline asm
	@%p13 st.global.v4.b32 [ %rd179 + 0 ], { %r536, %r537, %r538, %r539 };
	// end inline asm
	// begin inline asm
	@%p14 st.global.v4.b32 [ %rd180 + 0 ], { %r540, %r541, %r542, %r543 };
	// end inline asm
	.loc	1 173 4                         // sk05_mlp_gateup.py:173:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk05_mlp_gateup.py"
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
.b8 11                                  // DW_FORM_data1
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 165                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x9e DW_TAG_compile_unit
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
.b8 114
.b8 101
.b8 112
.b8 111
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
.b8 2                                   // Abbrev [2] 0x48:0x1a DW_TAG_subprogram
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
.b8 3                                   // Abbrev [3] 0x62:0x46 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 72                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x77:0x18 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 144                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x8f:0x18 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 145                                 // DW_AT_call_line
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
	.reg .b16 	%rs<161>;
	.reg .b32 	%r<1519>;
	.reg .b64 	%rd<190>;
	.loc	1 133 0                         // sk05_mlp_gateup.py:133:0
$L__func_begin0:
	.loc	1 133 0                         // sk05_mlp_gateup.py:133:0

// %bb.0:
	ld.param.b32 	%r25, [_sk05_mlp_gateup_kernel_param_13];
	ld.param.b32 	%r24, [_sk05_mlp_gateup_kernel_param_12];
	ld.param.b32 	%r23, [_sk05_mlp_gateup_kernel_param_9];
	ld.param.b32 	%r22, [_sk05_mlp_gateup_kernel_param_8];
	ld.param.b32 	%r21, [_sk05_mlp_gateup_kernel_param_7];
	ld.param.b64 	%rd37, [_sk05_mlp_gateup_kernel_param_5];
	ld.param.b64 	%rd36, [_sk05_mlp_gateup_kernel_param_4];
	ld.param.b64 	%rd35, [_sk05_mlp_gateup_kernel_param_3];
	ld.param.b64 	%rd34, [_sk05_mlp_gateup_kernel_param_2];
	ld.param.b64 	%rd33, [_sk05_mlp_gateup_kernel_param_1];
	ld.param.b64 	%rd32, [_sk05_mlp_gateup_kernel_param_0];
$L__tmp0:
	.loc	1 143 24                        // sk05_mlp_gateup.py:143:24
	mov.u32 	%r48, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk05_mlp_gateup.py:144:27 ]
	add.s32 	%r49, %r21, 255;
	.loc	2 43 30                         // standard.py:43:30 @[ sk05_mlp_gateup.py:144:27 ]
	shr.s32 	%r50, %r49, 31;
	shr.u32 	%r51, %r50, 24;
	add.s32 	%r52, %r49, %r51;
	shr.s32 	%r53, %r52, 8;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk05_mlp_gateup.py:145:27 ]
	add.s32 	%r54, %r22, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk05_mlp_gateup.py:145:27 ]
	shr.s32 	%r55, %r54, 31;
	shr.u32 	%r56, %r55, 25;
	add.s32 	%r57, %r54, %r56;
	shr.s32 	%r58, %r57, 7;
$L__tmp3:
	.loc	1 146 29                        // sk05_mlp_gateup.py:146:29
	shl.b32 	%r59, %r58, 3;
	.loc	1 147 22                        // sk05_mlp_gateup.py:147:22
	div.s32 	%r60, %r48, %r59;
	.loc	1 147 38                        // sk05_mlp_gateup.py:147:38
	shl.b32 	%r61, %r60, 3;
	.loc	1 148 30                        // sk05_mlp_gateup.py:148:30
	sub.s32 	%r62, %r53, %r61;
	ld.param.b32 	%r63, [_sk05_mlp_gateup_kernel_param_10];
	.loc	1 148 39                        // sk05_mlp_gateup.py:148:39
	min.s32 	%r64, %r62, 8;
	ld.param.b32 	%r65, [_sk05_mlp_gateup_kernel_param_11];
	.loc	1 149 30                        // sk05_mlp_gateup.py:149:30
	mul.lo.s32 	%r66, %r60, %r59;
	sub.s32 	%r67, %r48, %r66;
	.loc	1 150 36                        // sk05_mlp_gateup.py:150:36
	div.s32 	%r68, %r67, %r64;
	.loc	1 149 46                        // sk05_mlp_gateup.py:149:46
	mul.lo.s32 	%r69, %r68, %r64;
	sub.s32 	%r70, %r67, %r69;
	.loc	1 149 23                        // sk05_mlp_gateup.py:149:23
	add.s32 	%r71, %r70, %r61;
	.loc	1 152 22                        // sk05_mlp_gateup.py:152:22
	shl.b32 	%r1, %r71, 8;
	.loc	1 152 45                        // sk05_mlp_gateup.py:152:45
	mov.u32 	%r2, %tid.x;
	shr.u32 	%r72, %r2, 2;
	bfe.u32 	%r73, %r2, 2, 6;
	or.b32 	%r74, %r73, 64;
	and.b32 	%r3, %r2, 255;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r75, %r1, %r73;
	or.b32 	%r76, %r1, %r74;
	or.b32 	%r77, %r75, 128;
	or.b32 	%r78, %r1, %r72;
	or.b32 	%r79, %r78, 192;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r80, %r75, %r21;
	rem.s32 	%r81, %r76, %r21;
	rem.s32 	%r82, %r77, %r21;
	rem.s32 	%r83, %r79, %r21;
	.loc	1 153 22                        // sk05_mlp_gateup.py:153:22
	shl.b32 	%r4, %r68, 7;
	.loc	1 153 45                        // sk05_mlp_gateup.py:153:45
	and.b32 	%r5, %r2, 3;
	shl.b32 	%r84, %r5, 1;
	and.b32 	%r6, %r2, 32;
	shr.u32 	%r85, %r6, 2;
	or.b32 	%r86, %r85, %r84;
	and.b32 	%r7, %r2, 15;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r87, %r4, %r73;
	or.b32 	%r88, %r4, %r74;
	or.b32 	%r89, %r4, %r86;
	or.b32 	%r91, %r89, 16;
	or.b32 	%r93, %r89, 32;
	or.b32 	%r95, %r89, 48;
	or.b32 	%r97, %r89, 64;
	or.b32 	%r99, %r89, 80;
	or.b32 	%r101, %r89, 96;
	or.b32 	%r103, %r89, 112;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r105, %r87, %r22;
	rem.s32 	%r106, %r88, %r22;
	rem.s32 	%r8, %r89, %r22;
	rem.s32 	%r9, %r91, %r22;
	rem.s32 	%r10, %r93, %r22;
	rem.s32 	%r11, %r95, %r22;
	rem.s32 	%r12, %r97, %r22;
	rem.s32 	%r13, %r99, %r22;
	rem.s32 	%r14, %r101, %r22;
	rem.s32 	%r15, %r103, %r22;
	.loc	1 156 39                        // sk05_mlp_gateup.py:156:39
	mul.lo.s32 	%r115, %r80, %r63;
	mul.lo.s32 	%r116, %r81, %r63;
	mul.lo.s32 	%r117, %r82, %r63;
	mul.lo.s32 	%r118, %r83, %r63;
	.loc	1 156 21                        // sk05_mlp_gateup.py:156:21
	cvt.s64.s32 	%rd1, %r115;
	add.s64 	%rd57, %rd32, %rd1;
	cvt.s64.s32 	%rd2, %r116;
	add.s64 	%rd58, %rd32, %rd2;
	cvt.s64.s32 	%rd3, %r117;
	add.s64 	%rd59, %rd32, %rd3;
	cvt.s64.s32 	%rd4, %r118;
	add.s64 	%rd60, %rd32, %rd4;
	.loc	1 156 58                        // sk05_mlp_gateup.py:156:58
	shl.b32 	%r119, %r5, 4;
	.loc	1 156 51                        // sk05_mlp_gateup.py:156:51
	cvt.u64.u32 	%rd5, %r119;
	add.s64 	%rd38, %rd57, %rd5;
	add.s64 	%rd39, %rd58, %rd5;
	add.s64 	%rd40, %rd59, %rd5;
	add.s64 	%rd41, %rd60, %rd5;
	.loc	1 157 21                        // sk05_mlp_gateup.py:157:21
	add.s64 	%rd61, %rd33, %rd5;
	.loc	1 157 69                        // sk05_mlp_gateup.py:157:69
	mul.lo.s32 	%r120, %r105, %r65;
	mul.lo.s32 	%r121, %r106, %r65;
	.loc	1 157 51                        // sk05_mlp_gateup.py:157:51
	cvt.s64.s32 	%rd6, %r120;
	add.s64 	%rd42, %rd61, %rd6;
	cvt.s64.s32 	%rd7, %r121;
	add.s64 	%rd43, %rd61, %rd7;
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	setp.gt.s32 	%p1, %r23, 63;
	.loc	1 163 30                        // sk05_mlp_gateup.py:163:30
	shl.b32 	%r186, %r3, 4;
	shl.b32 	%r16, %r2, 1;
	and.b32 	%r17, %r16, 48;
	xor.b32 	%r187, %r186, %r17;
	mov.b32 	%r188, global_smem;
	add.s32 	%r27, %r188, %r187;
	selp.b32 	%r28, 16, 0, %p1;
	// begin inline asm
	cp.async.cg.shared.global [ %r27 + 0 ], [ %rd38 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r29, %r27, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r29 + 0 ], [ %rd39 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r30, %r27, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r30 + 0 ], [ %rd40 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r31, %r27, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r31 + 0 ], [ %rd41 + 0 ], 0x10, %r28;
	// end inline asm
	cp.async.commit_group;
	.loc	1 163 47                        // sk05_mlp_gateup.py:163:47
	add.s32 	%r32, %r27, 49152;
	// begin inline asm
	cp.async.cg.shared.global [ %r32 + 0 ], [ %rd42 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r33, %r27, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r33 + 0 ], [ %rd43 + 0 ], 0x10, %r28;
	// end inline asm
	cp.async.commit_group;
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	setp.gt.s32 	%p2, %r23, 127;
	.loc	1 164 18                        // sk05_mlp_gateup.py:164:18
	add.s64 	%rd44, %rd38, 64;
	add.s64 	%rd45, %rd39, 64;
	add.s64 	%rd46, %rd40, 64;
	add.s64 	%rd47, %rd41, 64;
	.loc	1 165 18                        // sk05_mlp_gateup.py:165:18
	add.s64 	%rd48, %rd42, 64;
	add.s64 	%rd49, %rd43, 64;
	.loc	1 163 30                        // sk05_mlp_gateup.py:163:30
	bar.sync 	0;
	add.s32 	%r34, %r27, 16384;
	selp.b32 	%r35, 16, 0, %p2;
	// begin inline asm
	cp.async.cg.shared.global [ %r34 + 0 ], [ %rd44 + 0 ], 0x10, %r35;
	// end inline asm
	add.s32 	%r36, %r27, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r36 + 0 ], [ %rd45 + 0 ], 0x10, %r35;
	// end inline asm
	add.s32 	%r37, %r27, 24576;
	// begin inline asm
	cp.async.cg.shared.global [ %r37 + 0 ], [ %rd46 + 0 ], 0x10, %r35;
	// end inline asm
	add.s32 	%r38, %r27, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r38 + 0 ], [ %rd47 + 0 ], 0x10, %r35;
	// end inline asm
	cp.async.commit_group;
	.loc	1 163 47                        // sk05_mlp_gateup.py:163:47
	add.s32 	%r39, %r27, 57344;
	// begin inline asm
	cp.async.cg.shared.global [ %r39 + 0 ], [ %rd48 + 0 ], 0x10, %r35;
	// end inline asm
	add.s32 	%r40, %r27, 61440;
	// begin inline asm
	cp.async.cg.shared.global [ %r40 + 0 ], [ %rd49 + 0 ], 0x10, %r35;
	// end inline asm
	cp.async.commit_group;
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	setp.gt.s32 	%p3, %r23, 191;
	.loc	1 164 18                        // sk05_mlp_gateup.py:164:18
	add.s64 	%rd50, %rd38, 128;
	add.s64 	%rd51, %rd39, 128;
	add.s64 	%rd52, %rd40, 128;
	add.s64 	%rd53, %rd41, 128;
	.loc	1 165 18                        // sk05_mlp_gateup.py:165:18
	add.s64 	%rd54, %rd42, 128;
	add.s64 	%rd55, %rd43, 128;
	.loc	1 163 30                        // sk05_mlp_gateup.py:163:30
	bar.sync 	0;
	add.s32 	%r41, %r27, 32768;
	selp.b32 	%r42, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r41 + 0 ], [ %rd50 + 0 ], 0x10, %r42;
	// end inline asm
	add.s32 	%r43, %r27, 36864;
	// begin inline asm
	cp.async.cg.shared.global [ %r43 + 0 ], [ %rd51 + 0 ], 0x10, %r42;
	// end inline asm
	add.s32 	%r44, %r27, 40960;
	// begin inline asm
	cp.async.cg.shared.global [ %r44 + 0 ], [ %rd52 + 0 ], 0x10, %r42;
	// end inline asm
	add.s32 	%r45, %r27, 45056;
	// begin inline asm
	cp.async.cg.shared.global [ %r45 + 0 ], [ %rd53 + 0 ], 0x10, %r42;
	// end inline asm
	cp.async.commit_group;
	.loc	1 163 47                        // sk05_mlp_gateup.py:163:47
	add.s32 	%r46, %r27, 65536;
	// begin inline asm
	cp.async.cg.shared.global [ %r46 + 0 ], [ %rd54 + 0 ], 0x10, %r42;
	// end inline asm
	add.s32 	%r47, %r27, 69632;
	// begin inline asm
	cp.async.cg.shared.global [ %r47 + 0 ], [ %rd55 + 0 ], 0x10, %r42;
	// end inline asm
	cp.async.commit_group;
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	@%p1 bra 	$L__BB0_2;
	bra.uni 	$L__BB0_1;
$L__BB0_2:                              // %.lr.ph
	.loc	1 0 23                          // sk05_mlp_gateup.py:0:23
	ld.param.b32 	%r26, [_sk05_mlp_gateup_kernel_param_14];
	ld.param.b64 	%rd56, [_sk05_mlp_gateup_kernel_param_6];
	or.b32 	%r90, %r89, 1;
	or.b32 	%r92, %r89, 17;
	or.b32 	%r94, %r89, 33;
	or.b32 	%r96, %r89, 49;
	or.b32 	%r98, %r89, 65;
	or.b32 	%r100, %r89, 81;
	or.b32 	%r102, %r89, 97;
	or.b32 	%r104, %r89, 113;
	rem.s32 	%r107, %r90, %r22;
	rem.s32 	%r108, %r92, %r22;
	rem.s32 	%r109, %r94, %r22;
	rem.s32 	%r110, %r96, %r22;
	rem.s32 	%r111, %r98, %r22;
	rem.s32 	%r112, %r100, %r22;
	rem.s32 	%r113, %r102, %r22;
	rem.s32 	%r114, %r104, %r22;
	shr.s32 	%r122, %r8, 31;
	shr.u32 	%r123, %r122, 25;
	add.s32 	%r124, %r8, %r123;
	shr.s32 	%r125, %r124, 7;
	shr.s32 	%r126, %r107, 31;
	shr.u32 	%r127, %r126, 25;
	add.s32 	%r128, %r107, %r127;
	shr.s32 	%r129, %r128, 7;
	shr.s32 	%r130, %r9, 31;
	shr.u32 	%r131, %r130, 25;
	add.s32 	%r132, %r9, %r131;
	shr.s32 	%r133, %r132, 7;
	shr.s32 	%r134, %r108, 31;
	shr.u32 	%r135, %r134, 25;
	add.s32 	%r136, %r108, %r135;
	shr.s32 	%r137, %r136, 7;
	shr.s32 	%r138, %r10, 31;
	shr.u32 	%r139, %r138, 25;
	add.s32 	%r140, %r10, %r139;
	shr.s32 	%r141, %r140, 7;
	shr.s32 	%r142, %r109, 31;
	shr.u32 	%r143, %r142, 25;
	add.s32 	%r144, %r109, %r143;
	shr.s32 	%r145, %r144, 7;
	shr.s32 	%r146, %r11, 31;
	shr.u32 	%r147, %r146, 25;
	add.s32 	%r148, %r11, %r147;
	shr.s32 	%r149, %r148, 7;
	shr.s32 	%r150, %r110, 31;
	shr.u32 	%r151, %r150, 25;
	add.s32 	%r152, %r110, %r151;
	shr.s32 	%r153, %r152, 7;
	shr.s32 	%r154, %r12, 31;
	shr.u32 	%r155, %r154, 25;
	add.s32 	%r156, %r12, %r155;
	shr.s32 	%r157, %r156, 7;
	shr.s32 	%r158, %r111, 31;
	shr.u32 	%r159, %r158, 25;
	add.s32 	%r160, %r111, %r159;
	shr.s32 	%r161, %r160, 7;
	shr.s32 	%r162, %r13, 31;
	shr.u32 	%r163, %r162, 25;
	add.s32 	%r164, %r13, %r163;
	shr.s32 	%r165, %r164, 7;
	shr.s32 	%r166, %r112, 31;
	shr.u32 	%r167, %r166, 25;
	add.s32 	%r168, %r112, %r167;
	shr.s32 	%r169, %r168, 7;
	shr.s32 	%r170, %r14, 31;
	shr.u32 	%r171, %r170, 25;
	add.s32 	%r172, %r14, %r171;
	shr.s32 	%r173, %r172, 7;
	shr.s32 	%r174, %r113, 31;
	shr.u32 	%r175, %r174, 25;
	add.s32 	%r176, %r113, %r175;
	shr.s32 	%r177, %r176, 7;
	shr.s32 	%r178, %r15, 31;
	shr.u32 	%r179, %r178, 25;
	add.s32 	%r180, %r15, %r179;
	shr.s32 	%r181, %r180, 7;
	shr.s32 	%r182, %r114, 31;
	shr.u32 	%r183, %r182, 25;
	add.s32 	%r184, %r114, %r183;
	shr.s32 	%r185, %r184, 7;
	cvt.s64.s32 	%rd62, %r125;
	add.s64 	%rd8, %rd56, %rd62;
	cvt.s64.s32 	%rd63, %r129;
	add.s64 	%rd9, %rd56, %rd63;
	cvt.s64.s32 	%rd64, %r133;
	add.s64 	%rd10, %rd56, %rd64;
	cvt.s64.s32 	%rd65, %r137;
	add.s64 	%rd11, %rd56, %rd65;
	cvt.s64.s32 	%rd66, %r141;
	add.s64 	%rd12, %rd56, %rd66;
	cvt.s64.s32 	%rd67, %r145;
	add.s64 	%rd13, %rd56, %rd67;
	cvt.s64.s32 	%rd68, %r149;
	add.s64 	%rd14, %rd56, %rd68;
	cvt.s64.s32 	%rd69, %r153;
	add.s64 	%rd15, %rd56, %rd69;
	cvt.s64.s32 	%rd70, %r157;
	add.s64 	%rd16, %rd56, %rd70;
	cvt.s64.s32 	%rd71, %r161;
	add.s64 	%rd17, %rd56, %rd71;
	cvt.s64.s32 	%rd72, %r165;
	add.s64 	%rd18, %rd56, %rd72;
	cvt.s64.s32 	%rd73, %r169;
	add.s64 	%rd19, %rd56, %rd73;
	cvt.s64.s32 	%rd74, %r173;
	add.s64 	%rd20, %rd56, %rd74;
	cvt.s64.s32 	%rd75, %r177;
	add.s64 	%rd21, %rd56, %rd75;
	cvt.s64.s32 	%rd76, %r181;
	add.s64 	%rd22, %rd56, %rd76;
	cvt.s64.s32 	%rd77, %r185;
	add.s64 	%rd23, %rd56, %rd77;
	.loc	1 161 28                        // sk05_mlp_gateup.py:161:28
	shr.u32 	%r189, %r23, 6;
	add.s32 	%r190, %r189, -3;
	shl.b32 	%r191, %r7, 6;
	shl.b32 	%r1389, %r2, 4;
	and.b32 	%r192, %r1389, 3072;
	shl.b32 	%r193, %r2, 3;
	and.b32 	%r194, %r193, 48;
	and.b32 	%r1390, %r2, 16;
	or.b32 	%r195, %r191, %r192;
	xor.b32 	%r196, %r194, %r1390;
	or.b32 	%r18, %r195, %r196;
	xor.b32 	%r19, %r18, 32;
	shl.b32 	%r197, %r2, 6;
	and.b32 	%r198, %r197, 448;
	shl.b32 	%r199, %r6, 4;
	or.b32 	%r200, %r198, %r194;
	xor.b32 	%r201, %r200, %r17;
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	add.s32 	%r202, %r188, %r199;
	add.s32 	%r20, %r202, %r201;
	cvt.s64.s32 	%rd24, %r190;
	and.b32 	%r203, %r23, -64;
	cvt.u64.u32 	%rd25, %r203;
	add.s64 	%rd78, %rd5, %rd7;
	add.s64 	%rd79, %rd78, %rd33;
	add.s64 	%rd26, %rd79, 192;
	add.s64 	%rd80, %rd5, %rd6;
	add.s64 	%rd81, %rd80, %rd33;
	add.s64 	%rd27, %rd81, 192;
	add.s64 	%rd82, %rd5, %rd4;
	add.s64 	%rd83, %rd82, %rd32;
	add.s64 	%rd28, %rd83, 192;
	add.s64 	%rd84, %rd5, %rd3;
	add.s64 	%rd85, %rd84, %rd32;
	add.s64 	%rd29, %rd85, 192;
	add.s64 	%rd86, %rd5, %rd2;
	add.s64 	%rd87, %rd86, %rd32;
	add.s64 	%rd30, %rd87, 192;
	add.s64 	%rd88, %rd5, %rd1;
	add.s64 	%rd89, %rd88, %rd32;
	add.s64 	%rd31, %rd89, 192;
	mov.b32 	%r1391, 0f00000000;
	mov.b32 	%r1388, 2;
	mov.b32 	%r1387, -1;
	mov.b64 	%rd188, 0;
	mov.b32 	%r204, 0;
	mov.b32 	%r1386, %r204;
	mov.b64 	%rd189, %rd188;
	mov.b32 	%r1392, %r1391;
	mov.b32 	%r1393, %r1391;
	mov.b32 	%r1394, %r1391;
	mov.b32 	%r1395, %r1391;
	mov.b32 	%r1396, %r1391;
	mov.b32 	%r1397, %r1391;
	mov.b32 	%r1398, %r1391;
	mov.b32 	%r1399, %r1391;
	mov.b32 	%r1400, %r1391;
	mov.b32 	%r1401, %r1391;
	mov.b32 	%r1402, %r1391;
	mov.b32 	%r1403, %r1391;
	mov.b32 	%r1404, %r1391;
	mov.b32 	%r1405, %r1391;
	mov.b32 	%r1406, %r1391;
	mov.b32 	%r1407, %r1391;
	mov.b32 	%r1408, %r1391;
	mov.b32 	%r1409, %r1391;
	mov.b32 	%r1410, %r1391;
	mov.b32 	%r1411, %r1391;
	mov.b32 	%r1412, %r1391;
	mov.b32 	%r1413, %r1391;
	mov.b32 	%r1414, %r1391;
	mov.b32 	%r1415, %r1391;
	mov.b32 	%r1416, %r1391;
	mov.b32 	%r1417, %r1391;
	mov.b32 	%r1418, %r1391;
	mov.b32 	%r1419, %r1391;
	mov.b32 	%r1420, %r1391;
	mov.b32 	%r1421, %r1391;
	mov.b32 	%r1422, %r1391;
	mov.b32 	%r1423, %r1391;
	mov.b32 	%r1424, %r1391;
	mov.b32 	%r1425, %r1391;
	mov.b32 	%r1426, %r1391;
	mov.b32 	%r1427, %r1391;
	mov.b32 	%r1428, %r1391;
	mov.b32 	%r1429, %r1391;
	mov.b32 	%r1430, %r1391;
	mov.b32 	%r1431, %r1391;
	mov.b32 	%r1432, %r1391;
	mov.b32 	%r1433, %r1391;
	mov.b32 	%r1434, %r1391;
	mov.b32 	%r1435, %r1391;
	mov.b32 	%r1436, %r1391;
	mov.b32 	%r1437, %r1391;
	mov.b32 	%r1438, %r1391;
	mov.b32 	%r1439, %r1391;
	mov.b32 	%r1440, %r1391;
	mov.b32 	%r1441, %r1391;
	mov.b32 	%r1442, %r1391;
	mov.b32 	%r1443, %r1391;
	mov.b32 	%r1444, %r1391;
	mov.b32 	%r1445, %r1391;
	mov.b32 	%r1446, %r1391;
	mov.b32 	%r1447, %r1391;
	mov.b32 	%r1448, %r1391;
	mov.b32 	%r1449, %r1391;
	mov.b32 	%r1450, %r1391;
	mov.b32 	%r1451, %r1391;
	mov.b32 	%r1452, %r1391;
	mov.b32 	%r1453, %r1391;
	mov.b32 	%r1454, %r1391;
	mov.b32 	%r1455, %r1391;
	mov.b32 	%r1456, %r1391;
	mov.b32 	%r1457, %r1391;
	mov.b32 	%r1458, %r1391;
	mov.b32 	%r1459, %r1391;
	mov.b32 	%r1460, %r1391;
	mov.b32 	%r1461, %r1391;
	mov.b32 	%r1462, %r1391;
	mov.b32 	%r1463, %r1391;
	mov.b32 	%r1464, %r1391;
	mov.b32 	%r1465, %r1391;
	mov.b32 	%r1466, %r1391;
	mov.b32 	%r1467, %r1391;
	mov.b32 	%r1468, %r1391;
	mov.b32 	%r1469, %r1391;
	mov.b32 	%r1470, %r1391;
	mov.b32 	%r1471, %r1391;
	mov.b32 	%r1472, %r1391;
	mov.b32 	%r1473, %r1391;
	mov.b32 	%r1474, %r1391;
	mov.b32 	%r1475, %r1391;
	mov.b32 	%r1476, %r1391;
	mov.b32 	%r1477, %r1391;
	mov.b32 	%r1478, %r1391;
	mov.b32 	%r1479, %r1391;
	mov.b32 	%r1480, %r1391;
	mov.b32 	%r1481, %r1391;
	mov.b32 	%r1482, %r1391;
	mov.b32 	%r1483, %r1391;
	mov.b32 	%r1484, %r1391;
	mov.b32 	%r1485, %r1391;
	mov.b32 	%r1486, %r1391;
	mov.b32 	%r1487, %r1391;
	mov.b32 	%r1488, %r1391;
	mov.b32 	%r1489, %r1391;
	mov.b32 	%r1490, %r1391;
	mov.b32 	%r1491, %r1391;
	mov.b32 	%r1492, %r1391;
	mov.b32 	%r1493, %r1391;
	mov.b32 	%r1494, %r1391;
	mov.b32 	%r1495, %r1391;
	mov.b32 	%r1496, %r1391;
	mov.b32 	%r1497, %r1391;
	mov.b32 	%r1498, %r1391;
	mov.b32 	%r1499, %r1391;
	mov.b32 	%r1500, %r1391;
	mov.b32 	%r1501, %r1391;
	mov.b32 	%r1502, %r1391;
	mov.b32 	%r1503, %r1391;
	mov.b32 	%r1504, %r1391;
	mov.b32 	%r1505, %r1391;
	mov.b32 	%r1506, %r1391;
	mov.b32 	%r1507, %r1391;
	mov.b32 	%r1508, %r1391;
	mov.b32 	%r1509, %r1391;
	mov.b32 	%r1510, %r1391;
	mov.b32 	%r1511, %r1391;
	mov.b32 	%r1512, %r1391;
	mov.b32 	%r1513, %r1391;
	mov.b32 	%r1514, %r1391;
	mov.b32 	%r1515, %r1391;
	mov.b32 	%r1516, %r1391;
	mov.b32 	%r1517, %r1391;
	mov.b32 	%r1518, %r1391;
$L__BB0_3:                              // %__nv_exp2f.exit
                                        // =>This Inner Loop Header: Depth=1
	setp.lt.s64 	%p4, %rd189, %rd24;
	add.s32 	%r404, %r1387, 1;
	setp.gt.s32 	%p5, %r404, 2;
	selp.b32 	%r1387, 0, %r404, %p5;
	.loc	1 162 39                        // sk05_mlp_gateup.py:162:39
	cvt.s64.s32 	%rd112, %r1386;
	add.s64 	%rd90, %rd8, %rd112;
	add.s64 	%rd91, %rd9, %rd112;
	add.s64 	%rd92, %rd10, %rd112;
	add.s64 	%rd93, %rd11, %rd112;
	add.s64 	%rd94, %rd12, %rd112;
	add.s64 	%rd95, %rd13, %rd112;
	add.s64 	%rd96, %rd14, %rd112;
	add.s64 	%rd97, %rd15, %rd112;
	add.s64 	%rd98, %rd16, %rd112;
	add.s64 	%rd99, %rd17, %rd112;
	add.s64 	%rd100, %rd18, %rd112;
	add.s64 	%rd101, %rd19, %rd112;
	add.s64 	%rd102, %rd20, %rd112;
	add.s64 	%rd103, %rd21, %rd112;
	add.s64 	%rd104, %rd22, %rd112;
	add.s64 	%rd105, %rd23, %rd112;
	.loc	1 162 29                        // sk05_mlp_gateup.py:162:29
	// begin inline asm
	mov.u16 %rs1, 0x0;
	ld.global.b8 { %rs1 }, [ %rd90 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs17, %rs1;
	// begin inline asm
	mov.u16 %rs2, 0x0;
	ld.global.b8 { %rs2 }, [ %rd91 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs18, %rs2;
	// begin inline asm
	mov.u16 %rs3, 0x0;
	ld.global.b8 { %rs3 }, [ %rd92 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs19, %rs3;
	// begin inline asm
	mov.u16 %rs4, 0x0;
	ld.global.b8 { %rs4 }, [ %rd93 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs20, %rs4;
	// begin inline asm
	mov.u16 %rs5, 0x0;
	ld.global.b8 { %rs5 }, [ %rd94 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs21, %rs5;
	// begin inline asm
	mov.u16 %rs6, 0x0;
	ld.global.b8 { %rs6 }, [ %rd95 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs22, %rs6;
	// begin inline asm
	mov.u16 %rs7, 0x0;
	ld.global.b8 { %rs7 }, [ %rd96 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs23, %rs7;
	// begin inline asm
	mov.u16 %rs8, 0x0;
	ld.global.b8 { %rs8 }, [ %rd97 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs24, %rs8;
	// begin inline asm
	mov.u16 %rs9, 0x0;
	ld.global.b8 { %rs9 }, [ %rd98 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs25, %rs9;
	// begin inline asm
	mov.u16 %rs10, 0x0;
	ld.global.b8 { %rs10 }, [ %rd99 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs26, %rs10;
	// begin inline asm
	mov.u16 %rs11, 0x0;
	ld.global.b8 { %rs11 }, [ %rd100 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs27, %rs11;
	// begin inline asm
	mov.u16 %rs12, 0x0;
	ld.global.b8 { %rs12 }, [ %rd101 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs28, %rs12;
	// begin inline asm
	mov.u16 %rs13, 0x0;
	ld.global.b8 { %rs13 }, [ %rd102 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs29, %rs13;
	// begin inline asm
	mov.u16 %rs14, 0x0;
	ld.global.b8 { %rs14 }, [ %rd103 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs30, %rs14;
	// begin inline asm
	mov.u16 %rs15, 0x0;
	ld.global.b8 { %rs15 }, [ %rd104 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs31, %rs15;
	// begin inline asm
	mov.u16 %rs16, 0x0;
	ld.global.b8 { %rs16 }, [ %rd105 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs32, %rs16;
	.loc	1 162 63                        // sk05_mlp_gateup.py:162:63
	cvt.rn.f32.s16 	%r405, %rs17;
	cvt.rn.f32.s16 	%r406, %rs18;
	cvt.rn.f32.s16 	%r407, %rs19;
	cvt.rn.f32.s16 	%r408, %rs20;
	cvt.rn.f32.s16 	%r409, %rs21;
	cvt.rn.f32.s16 	%r410, %rs22;
	cvt.rn.f32.s16 	%r411, %rs23;
	cvt.rn.f32.s16 	%r412, %rs24;
	cvt.rn.f32.s16 	%r413, %rs25;
	cvt.rn.f32.s16 	%r414, %rs26;
	cvt.rn.f32.s16 	%r415, %rs27;
	cvt.rn.f32.s16 	%r416, %rs28;
	cvt.rn.f32.s16 	%r417, %rs29;
	cvt.rn.f32.s16 	%r418, %rs30;
	cvt.rn.f32.s16 	%r419, %rs31;
	cvt.rn.f32.s16 	%r420, %rs32;
	.loc	1 162 21                        // sk05_mlp_gateup.py:162:21
	ex2.approx.ftz.f32 	%r421, %r405;
	ex2.approx.ftz.f32 	%r422, %r406;
	ex2.approx.ftz.f32 	%r423, %r407;
	ex2.approx.ftz.f32 	%r424, %r408;
	ex2.approx.ftz.f32 	%r425, %r409;
	ex2.approx.ftz.f32 	%r426, %r410;
	ex2.approx.ftz.f32 	%r427, %r411;
	ex2.approx.ftz.f32 	%r428, %r412;
	ex2.approx.ftz.f32 	%r429, %r413;
	ex2.approx.ftz.f32 	%r430, %r414;
	ex2.approx.ftz.f32 	%r431, %r415;
	ex2.approx.ftz.f32 	%r432, %r416;
	ex2.approx.ftz.f32 	%r433, %r417;
	ex2.approx.ftz.f32 	%r434, %r418;
	ex2.approx.ftz.f32 	%r435, %r419;
	ex2.approx.ftz.f32 	%r436, %r420;
	.loc	1 163 30                        // sk05_mlp_gateup.py:163:30
	cp.async.wait_group 	4;
	bar.sync 	0;
	shl.b32 	%r437, %r1387, 14;
	add.s32 	%r438, %r188, %r437;
	add.s32 	%r439, %r438, %r18;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r205, %r206, %r207, %r208}, [%r439];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r225, %r226, %r227, %r228}, [%r439+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r229, %r230, %r231, %r232}, [%r439+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r233, %r234, %r235, %r236}, [%r439+12288];
	add.s32 	%r440, %r438, %r19;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r241, %r242, %r243, %r244}, [%r440];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r293, %r294, %r295, %r296}, [%r440+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r329, %r330, %r331, %r332}, [%r440+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r365, %r366, %r367, %r368}, [%r440+12288];
	.loc	1 163 47                        // sk05_mlp_gateup.py:163:47
	shl.b32 	%r441, %r1387, 13;
	add.s32 	%r442, %r20, %r441;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r209, %r210, %r245, %r246}, [%r442+49152];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r211, %r212, %r251, %r252}, [%r442+50176];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r213, %r214, %r257, %r258}, [%r442+51200];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r215, %r216, %r263, %r264}, [%r442+52224];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r217, %r218, %r269, %r270}, [%r442+53248];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r219, %r220, %r275, %r276}, [%r442+54272];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r221, %r222, %r281, %r282}, [%r442+55296];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r223, %r224, %r287, %r288}, [%r442+56320];
	.loc	1 163 39                        // sk05_mlp_gateup.py:163:39
	mov.b32 	%r237, %r204;
	mov.b32 	%r238, %r204;
	mov.b32 	%r239, %r204;
	mov.b32 	%r240, %r204;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r237, %r238, %r239, %r240 }, { %r205, %r206, %r207, %r208 }, { %r209, %r210 }, { %r237, %r238, %r239, %r240 };
	// end inline asm
	mov.b32 	%r247, %r204;
	mov.b32 	%r248, %r204;
	mov.b32 	%r249, %r204;
	mov.b32 	%r250, %r204;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r247, %r248, %r249, %r250 }, { %r205, %r206, %r207, %r208 }, { %r211, %r212 }, { %r247, %r248, %r249, %r250 };
	// end inline asm
	mov.b32 	%r253, %r204;
	mov.b32 	%r254, %r204;
	mov.b32 	%r255, %r204;
	mov.b32 	%r256, %r204;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r253, %r254, %r255, %r256 }, { %r205, %r206, %r207, %r208 }, { %r213, %r214 }, { %r253, %r254, %r255, %r256 };
	// end inline asm
	mov.b32 	%r259, %r204;
	mov.b32 	%r260, %r204;
	mov.b32 	%r261, %r204;
	mov.b32 	%r262, %r204;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r259, %r260, %r261, %r262 }, { %r205, %r206, %r207, %r208 }, { %r215, %r216 }, { %r259, %r260, %r261, %r262 };
	// end inline asm
	mov.b32 	%r265, %r204;
	mov.b32 	%r266, %r204;
	mov.b32 	%r267, %r204;
	mov.b32 	%r268, %r204;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r265, %r266, %r267, %r268 }, { %r205, %r206, %r207, %r208 }, { %r217, %r218 }, { %r265, %r266, %r267, %r268 };
	// end inline asm
	mov.b32 	%r271, %r204;
	mov.b32 	%r272, %r204;
	mov.b32 	%r273, %r204;
	mov.b32 	%r274, %r204;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r271, %r272, %r273, %r274 }, { %r205, %r206, %r207, %r208 }, { %r219, %r220 }, { %r271, %r272, %r273, %r274 };
	// end inline asm
	mov.b32 	%r277, %r204;
	mov.b32 	%r278, %r204;
	mov.b32 	%r279, %r204;
	mov.b32 	%r280, %r204;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r277, %r278, %r279, %r280 }, { %r205, %r206, %r207, %r208 }, { %r221, %r222 }, { %r277, %r278, %r279, %r280 };
	// end inline asm
	mov.b32 	%r283, %r204;
	mov.b32 	%r284, %r204;
	mov.b32 	%r285, %r204;
	mov.b32 	%r286, %r204;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r283, %r284, %r285, %r286 }, { %r205, %r206, %r207, %r208 }, { %r223, %r224 }, { %r283, %r284, %r285, %r286 };
	// end inline asm
	mov.b32 	%r289, %r204;
	mov.b32 	%r290, %r204;
	mov.b32 	%r291, %r204;
	mov.b32 	%r292, %r204;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r289, %r290, %r291, %r292 }, { %r225, %r226, %r227, %r228 }, { %r209, %r210 }, { %r289, %r290, %r291, %r292 };
	// end inline asm
	mov.b32 	%r297, %r204;
	mov.b32 	%r298, %r204;
	mov.b32 	%r299, %r204;
	mov.b32 	%r300, %r204;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r297, %r298, %r299, %r300 }, { %r225, %r226, %r227, %r228 }, { %r211, %r212 }, { %r297, %r298, %r299, %r300 };
	// end inline asm
	mov.b32 	%r301, %r204;
	mov.b32 	%r302, %r204;
	mov.b32 	%r303, %r204;
	mov.b32 	%r304, %r204;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r301, %r302, %r303, %r304 }, { %r225, %r226, %r227, %r228 }, { %r213, %r214 }, { %r301, %r302, %r303, %r304 };
	// end inline asm
	mov.b32 	%r305, %r204;
	mov.b32 	%r306, %r204;
	mov.b32 	%r307, %r204;
	mov.b32 	%r308, %r204;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r305, %r306, %r307, %r308 }, { %r225, %r226, %r227, %r228 }, { %r215, %r216 }, { %r305, %r306, %r307, %r308 };
	// end inline asm
	mov.b32 	%r309, %r204;
	mov.b32 	%r310, %r204;
	mov.b32 	%r311, %r204;
	mov.b32 	%r312, %r204;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r309, %r310, %r311, %r312 }, { %r225, %r226, %r227, %r228 }, { %r217, %r218 }, { %r309, %r310, %r311, %r312 };
	// end inline asm
	mov.b32 	%r313, %r204;
	mov.b32 	%r314, %r204;
	mov.b32 	%r315, %r204;
	mov.b32 	%r316, %r204;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r313, %r314, %r315, %r316 }, { %r225, %r226, %r227, %r228 }, { %r219, %r220 }, { %r313, %r314, %r315, %r316 };
	// end inline asm
	mov.b32 	%r317, %r204;
	mov.b32 	%r318, %r204;
	mov.b32 	%r319, %r204;
	mov.b32 	%r320, %r204;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r317, %r318, %r319, %r320 }, { %r225, %r226, %r227, %r228 }, { %r221, %r222 }, { %r317, %r318, %r319, %r320 };
	// end inline asm
	mov.b32 	%r321, %r204;
	mov.b32 	%r322, %r204;
	mov.b32 	%r323, %r204;
	mov.b32 	%r324, %r204;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r321, %r322, %r323, %r324 }, { %r225, %r226, %r227, %r228 }, { %r223, %r224 }, { %r321, %r322, %r323, %r324 };
	// end inline asm
	mov.b32 	%r325, %r204;
	mov.b32 	%r326, %r204;
	mov.b32 	%r327, %r204;
	mov.b32 	%r328, %r204;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r325, %r326, %r327, %r328 }, { %r229, %r230, %r231, %r232 }, { %r209, %r210 }, { %r325, %r326, %r327, %r328 };
	// end inline asm
	mov.b32 	%r333, %r204;
	mov.b32 	%r334, %r204;
	mov.b32 	%r335, %r204;
	mov.b32 	%r336, %r204;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r333, %r334, %r335, %r336 }, { %r229, %r230, %r231, %r232 }, { %r211, %r212 }, { %r333, %r334, %r335, %r336 };
	// end inline asm
	mov.b32 	%r337, %r204;
	mov.b32 	%r338, %r204;
	mov.b32 	%r339, %r204;
	mov.b32 	%r340, %r204;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r337, %r338, %r339, %r340 }, { %r229, %r230, %r231, %r232 }, { %r213, %r214 }, { %r337, %r338, %r339, %r340 };
	// end inline asm
	mov.b32 	%r341, %r204;
	mov.b32 	%r342, %r204;
	mov.b32 	%r343, %r204;
	mov.b32 	%r344, %r204;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r341, %r342, %r343, %r344 }, { %r229, %r230, %r231, %r232 }, { %r215, %r216 }, { %r341, %r342, %r343, %r344 };
	// end inline asm
	mov.b32 	%r345, %r204;
	mov.b32 	%r346, %r204;
	mov.b32 	%r347, %r204;
	mov.b32 	%r348, %r204;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r345, %r346, %r347, %r348 }, { %r229, %r230, %r231, %r232 }, { %r217, %r218 }, { %r345, %r346, %r347, %r348 };
	// end inline asm
	mov.b32 	%r349, %r204;
	mov.b32 	%r350, %r204;
	mov.b32 	%r351, %r204;
	mov.b32 	%r352, %r204;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r349, %r350, %r351, %r352 }, { %r229, %r230, %r231, %r232 }, { %r219, %r220 }, { %r349, %r350, %r351, %r352 };
	// end inline asm
	mov.b32 	%r353, %r204;
	mov.b32 	%r354, %r204;
	mov.b32 	%r355, %r204;
	mov.b32 	%r356, %r204;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r353, %r354, %r355, %r356 }, { %r229, %r230, %r231, %r232 }, { %r221, %r222 }, { %r353, %r354, %r355, %r356 };
	// end inline asm
	mov.b32 	%r357, %r204;
	mov.b32 	%r358, %r204;
	mov.b32 	%r359, %r204;
	mov.b32 	%r360, %r204;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r357, %r358, %r359, %r360 }, { %r229, %r230, %r231, %r232 }, { %r223, %r224 }, { %r357, %r358, %r359, %r360 };
	// end inline asm
	mov.b32 	%r361, %r204;
	mov.b32 	%r362, %r204;
	mov.b32 	%r363, %r204;
	mov.b32 	%r364, %r204;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r361, %r362, %r363, %r364 }, { %r233, %r234, %r235, %r236 }, { %r209, %r210 }, { %r361, %r362, %r363, %r364 };
	// end inline asm
	mov.b32 	%r369, %r204;
	mov.b32 	%r370, %r204;
	mov.b32 	%r371, %r204;
	mov.b32 	%r372, %r204;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r369, %r370, %r371, %r372 }, { %r233, %r234, %r235, %r236 }, { %r211, %r212 }, { %r369, %r370, %r371, %r372 };
	// end inline asm
	mov.b32 	%r373, %r204;
	mov.b32 	%r374, %r204;
	mov.b32 	%r375, %r204;
	mov.b32 	%r376, %r204;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r373, %r374, %r375, %r376 }, { %r233, %r234, %r235, %r236 }, { %r213, %r214 }, { %r373, %r374, %r375, %r376 };
	// end inline asm
	mov.b32 	%r377, %r204;
	mov.b32 	%r378, %r204;
	mov.b32 	%r379, %r204;
	mov.b32 	%r380, %r204;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r377, %r378, %r379, %r380 }, { %r233, %r234, %r235, %r236 }, { %r215, %r216 }, { %r377, %r378, %r379, %r380 };
	// end inline asm
	mov.b32 	%r381, %r204;
	mov.b32 	%r382, %r204;
	mov.b32 	%r383, %r204;
	mov.b32 	%r384, %r204;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r381, %r382, %r383, %r384 }, { %r233, %r234, %r235, %r236 }, { %r217, %r218 }, { %r381, %r382, %r383, %r384 };
	// end inline asm
	mov.b32 	%r385, %r204;
	mov.b32 	%r386, %r204;
	mov.b32 	%r387, %r204;
	mov.b32 	%r388, %r204;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r385, %r386, %r387, %r388 }, { %r233, %r234, %r235, %r236 }, { %r219, %r220 }, { %r385, %r386, %r387, %r388 };
	// end inline asm
	mov.b32 	%r389, %r204;
	mov.b32 	%r390, %r204;
	mov.b32 	%r391, %r204;
	mov.b32 	%r392, %r204;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r389, %r390, %r391, %r392 }, { %r233, %r234, %r235, %r236 }, { %r221, %r222 }, { %r389, %r390, %r391, %r392 };
	// end inline asm
	mov.b32 	%r393, %r204;
	mov.b32 	%r394, %r204;
	mov.b32 	%r395, %r204;
	mov.b32 	%r396, %r204;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r393, %r394, %r395, %r396 }, { %r233, %r234, %r235, %r236 }, { %r223, %r224 }, { %r393, %r394, %r395, %r396 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r237, %r238, %r239, %r240 }, { %r241, %r242, %r243, %r244 }, { %r245, %r246 }, { %r237, %r238, %r239, %r240 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r247, %r248, %r249, %r250 }, { %r241, %r242, %r243, %r244 }, { %r251, %r252 }, { %r247, %r248, %r249, %r250 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r253, %r254, %r255, %r256 }, { %r241, %r242, %r243, %r244 }, { %r257, %r258 }, { %r253, %r254, %r255, %r256 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r259, %r260, %r261, %r262 }, { %r241, %r242, %r243, %r244 }, { %r263, %r264 }, { %r259, %r260, %r261, %r262 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r265, %r266, %r267, %r268 }, { %r241, %r242, %r243, %r244 }, { %r269, %r270 }, { %r265, %r266, %r267, %r268 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r271, %r272, %r273, %r274 }, { %r241, %r242, %r243, %r244 }, { %r275, %r276 }, { %r271, %r272, %r273, %r274 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r277, %r278, %r279, %r280 }, { %r241, %r242, %r243, %r244 }, { %r281, %r282 }, { %r277, %r278, %r279, %r280 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r283, %r284, %r285, %r286 }, { %r241, %r242, %r243, %r244 }, { %r287, %r288 }, { %r283, %r284, %r285, %r286 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r289, %r290, %r291, %r292 }, { %r293, %r294, %r295, %r296 }, { %r245, %r246 }, { %r289, %r290, %r291, %r292 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r297, %r298, %r299, %r300 }, { %r293, %r294, %r295, %r296 }, { %r251, %r252 }, { %r297, %r298, %r299, %r300 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r301, %r302, %r303, %r304 }, { %r293, %r294, %r295, %r296 }, { %r257, %r258 }, { %r301, %r302, %r303, %r304 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r305, %r306, %r307, %r308 }, { %r293, %r294, %r295, %r296 }, { %r263, %r264 }, { %r305, %r306, %r307, %r308 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r309, %r310, %r311, %r312 }, { %r293, %r294, %r295, %r296 }, { %r269, %r270 }, { %r309, %r310, %r311, %r312 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r313, %r314, %r315, %r316 }, { %r293, %r294, %r295, %r296 }, { %r275, %r276 }, { %r313, %r314, %r315, %r316 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r317, %r318, %r319, %r320 }, { %r293, %r294, %r295, %r296 }, { %r281, %r282 }, { %r317, %r318, %r319, %r320 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r321, %r322, %r323, %r324 }, { %r293, %r294, %r295, %r296 }, { %r287, %r288 }, { %r321, %r322, %r323, %r324 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r325, %r326, %r327, %r328 }, { %r329, %r330, %r331, %r332 }, { %r245, %r246 }, { %r325, %r326, %r327, %r328 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r333, %r334, %r335, %r336 }, { %r329, %r330, %r331, %r332 }, { %r251, %r252 }, { %r333, %r334, %r335, %r336 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r337, %r338, %r339, %r340 }, { %r329, %r330, %r331, %r332 }, { %r257, %r258 }, { %r337, %r338, %r339, %r340 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r341, %r342, %r343, %r344 }, { %r329, %r330, %r331, %r332 }, { %r263, %r264 }, { %r341, %r342, %r343, %r344 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r345, %r346, %r347, %r348 }, { %r329, %r330, %r331, %r332 }, { %r269, %r270 }, { %r345, %r346, %r347, %r348 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r349, %r350, %r351, %r352 }, { %r329, %r330, %r331, %r332 }, { %r275, %r276 }, { %r349, %r350, %r351, %r352 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r353, %r354, %r355, %r356 }, { %r329, %r330, %r331, %r332 }, { %r281, %r282 }, { %r353, %r354, %r355, %r356 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r357, %r358, %r359, %r360 }, { %r329, %r330, %r331, %r332 }, { %r287, %r288 }, { %r357, %r358, %r359, %r360 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r361, %r362, %r363, %r364 }, { %r365, %r366, %r367, %r368 }, { %r245, %r246 }, { %r361, %r362, %r363, %r364 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r369, %r370, %r371, %r372 }, { %r365, %r366, %r367, %r368 }, { %r251, %r252 }, { %r369, %r370, %r371, %r372 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r373, %r374, %r375, %r376 }, { %r365, %r366, %r367, %r368 }, { %r257, %r258 }, { %r373, %r374, %r375, %r376 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r377, %r378, %r379, %r380 }, { %r365, %r366, %r367, %r368 }, { %r263, %r264 }, { %r377, %r378, %r379, %r380 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r381, %r382, %r383, %r384 }, { %r365, %r366, %r367, %r368 }, { %r269, %r270 }, { %r381, %r382, %r383, %r384 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r385, %r386, %r387, %r388 }, { %r365, %r366, %r367, %r368 }, { %r275, %r276 }, { %r385, %r386, %r387, %r388 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r389, %r390, %r391, %r392 }, { %r365, %r366, %r367, %r368 }, { %r281, %r282 }, { %r389, %r390, %r391, %r392 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r393, %r394, %r395, %r396 }, { %r365, %r366, %r367, %r368 }, { %r287, %r288 }, { %r393, %r394, %r395, %r396 };
	// end inline asm
	.loc	1 163 79                        // sk05_mlp_gateup.py:163:79
	cvt.rn.f32.s32 	%r443, %r396;
	cvt.rn.f32.s32 	%r444, %r395;
	cvt.rn.f32.s32 	%r445, %r394;
	cvt.rn.f32.s32 	%r446, %r393;
	cvt.rn.f32.s32 	%r447, %r392;
	cvt.rn.f32.s32 	%r448, %r391;
	cvt.rn.f32.s32 	%r449, %r390;
	cvt.rn.f32.s32 	%r450, %r389;
	cvt.rn.f32.s32 	%r451, %r388;
	cvt.rn.f32.s32 	%r452, %r387;
	cvt.rn.f32.s32 	%r453, %r386;
	cvt.rn.f32.s32 	%r454, %r385;
	cvt.rn.f32.s32 	%r455, %r384;
	cvt.rn.f32.s32 	%r456, %r383;
	cvt.rn.f32.s32 	%r457, %r382;
	cvt.rn.f32.s32 	%r458, %r381;
	cvt.rn.f32.s32 	%r459, %r380;
	cvt.rn.f32.s32 	%r460, %r379;
	cvt.rn.f32.s32 	%r461, %r378;
	cvt.rn.f32.s32 	%r462, %r377;
	cvt.rn.f32.s32 	%r463, %r376;
	cvt.rn.f32.s32 	%r464, %r375;
	cvt.rn.f32.s32 	%r465, %r374;
	cvt.rn.f32.s32 	%r466, %r373;
	cvt.rn.f32.s32 	%r467, %r372;
	cvt.rn.f32.s32 	%r468, %r371;
	cvt.rn.f32.s32 	%r469, %r370;
	cvt.rn.f32.s32 	%r470, %r369;
	cvt.rn.f32.s32 	%r471, %r364;
	cvt.rn.f32.s32 	%r472, %r363;
	cvt.rn.f32.s32 	%r473, %r362;
	cvt.rn.f32.s32 	%r474, %r361;
	cvt.rn.f32.s32 	%r475, %r360;
	cvt.rn.f32.s32 	%r476, %r359;
	cvt.rn.f32.s32 	%r477, %r357;
	cvt.rn.f32.s32 	%r478, %r358;
	cvt.rn.f32.s32 	%r479, %r353;
	cvt.rn.f32.s32 	%r480, %r354;
	cvt.rn.f32.s32 	%r481, %r355;
	cvt.rn.f32.s32 	%r482, %r356;
	cvt.rn.f32.s32 	%r483, %r349;
	cvt.rn.f32.s32 	%r484, %r350;
	cvt.rn.f32.s32 	%r485, %r351;
	cvt.rn.f32.s32 	%r486, %r352;
	cvt.rn.f32.s32 	%r487, %r345;
	cvt.rn.f32.s32 	%r488, %r346;
	cvt.rn.f32.s32 	%r489, %r347;
	cvt.rn.f32.s32 	%r490, %r348;
	cvt.rn.f32.s32 	%r491, %r341;
	cvt.rn.f32.s32 	%r492, %r342;
	cvt.rn.f32.s32 	%r493, %r343;
	cvt.rn.f32.s32 	%r494, %r344;
	cvt.rn.f32.s32 	%r495, %r337;
	cvt.rn.f32.s32 	%r496, %r338;
	cvt.rn.f32.s32 	%r497, %r339;
	cvt.rn.f32.s32 	%r498, %r340;
	cvt.rn.f32.s32 	%r499, %r333;
	cvt.rn.f32.s32 	%r500, %r334;
	cvt.rn.f32.s32 	%r501, %r335;
	cvt.rn.f32.s32 	%r502, %r336;
	cvt.rn.f32.s32 	%r503, %r325;
	cvt.rn.f32.s32 	%r504, %r326;
	cvt.rn.f32.s32 	%r505, %r327;
	cvt.rn.f32.s32 	%r506, %r328;
	cvt.rn.f32.s32 	%r507, %r321;
	cvt.rn.f32.s32 	%r508, %r322;
	cvt.rn.f32.s32 	%r509, %r323;
	cvt.rn.f32.s32 	%r510, %r324;
	cvt.rn.f32.s32 	%r511, %r317;
	cvt.rn.f32.s32 	%r512, %r318;
	cvt.rn.f32.s32 	%r513, %r319;
	cvt.rn.f32.s32 	%r514, %r320;
	cvt.rn.f32.s32 	%r515, %r313;
	cvt.rn.f32.s32 	%r516, %r314;
	cvt.rn.f32.s32 	%r517, %r315;
	cvt.rn.f32.s32 	%r518, %r316;
	cvt.rn.f32.s32 	%r519, %r309;
	cvt.rn.f32.s32 	%r520, %r310;
	cvt.rn.f32.s32 	%r521, %r311;
	cvt.rn.f32.s32 	%r522, %r312;
	cvt.rn.f32.s32 	%r523, %r305;
	cvt.rn.f32.s32 	%r524, %r306;
	cvt.rn.f32.s32 	%r525, %r307;
	cvt.rn.f32.s32 	%r526, %r308;
	cvt.rn.f32.s32 	%r527, %r301;
	cvt.rn.f32.s32 	%r528, %r302;
	cvt.rn.f32.s32 	%r529, %r303;
	cvt.rn.f32.s32 	%r530, %r304;
	cvt.rn.f32.s32 	%r531, %r297;
	cvt.rn.f32.s32 	%r532, %r298;
	cvt.rn.f32.s32 	%r533, %r299;
	cvt.rn.f32.s32 	%r534, %r300;
	cvt.rn.f32.s32 	%r535, %r289;
	cvt.rn.f32.s32 	%r536, %r290;
	cvt.rn.f32.s32 	%r537, %r291;
	cvt.rn.f32.s32 	%r538, %r292;
	cvt.rn.f32.s32 	%r539, %r283;
	cvt.rn.f32.s32 	%r540, %r284;
	cvt.rn.f32.s32 	%r541, %r285;
	cvt.rn.f32.s32 	%r542, %r286;
	cvt.rn.f32.s32 	%r543, %r277;
	cvt.rn.f32.s32 	%r544, %r278;
	cvt.rn.f32.s32 	%r545, %r279;
	cvt.rn.f32.s32 	%r546, %r280;
	cvt.rn.f32.s32 	%r547, %r271;
	cvt.rn.f32.s32 	%r548, %r272;
	cvt.rn.f32.s32 	%r549, %r273;
	cvt.rn.f32.s32 	%r550, %r274;
	cvt.rn.f32.s32 	%r551, %r265;
	cvt.rn.f32.s32 	%r552, %r266;
	cvt.rn.f32.s32 	%r553, %r267;
	cvt.rn.f32.s32 	%r554, %r268;
	cvt.rn.f32.s32 	%r555, %r259;
	cvt.rn.f32.s32 	%r556, %r260;
	cvt.rn.f32.s32 	%r557, %r261;
	cvt.rn.f32.s32 	%r558, %r262;
	cvt.rn.f32.s32 	%r559, %r253;
	cvt.rn.f32.s32 	%r560, %r254;
	cvt.rn.f32.s32 	%r561, %r255;
	cvt.rn.f32.s32 	%r562, %r256;
	cvt.rn.f32.s32 	%r563, %r247;
	cvt.rn.f32.s32 	%r564, %r248;
	cvt.rn.f32.s32 	%r565, %r249;
	cvt.rn.f32.s32 	%r566, %r250;
	cvt.rn.f32.s32 	%r567, %r237;
	cvt.rn.f32.s32 	%r568, %r238;
	cvt.rn.f32.s32 	%r569, %r239;
	cvt.rn.f32.s32 	%r570, %r240;
	.loc	1 163 15                        // sk05_mlp_gateup.py:163:15
	fma.rn.f32 	%r1394, %r422, %r570, %r1394;
	fma.rn.f32 	%r1393, %r421, %r569, %r1393;
	fma.rn.f32 	%r1392, %r422, %r568, %r1392;
	fma.rn.f32 	%r1391, %r421, %r567, %r1391;
	fma.rn.f32 	%r1398, %r424, %r566, %r1398;
	fma.rn.f32 	%r1397, %r423, %r565, %r1397;
	fma.rn.f32 	%r1396, %r424, %r564, %r1396;
	fma.rn.f32 	%r1395, %r423, %r563, %r1395;
	fma.rn.f32 	%r1402, %r426, %r562, %r1402;
	fma.rn.f32 	%r1401, %r425, %r561, %r1401;
	fma.rn.f32 	%r1400, %r426, %r560, %r1400;
	fma.rn.f32 	%r1399, %r425, %r559, %r1399;
	fma.rn.f32 	%r1406, %r428, %r558, %r1406;
	fma.rn.f32 	%r1405, %r427, %r557, %r1405;
	fma.rn.f32 	%r1404, %r428, %r556, %r1404;
	fma.rn.f32 	%r1403, %r427, %r555, %r1403;
	fma.rn.f32 	%r1410, %r430, %r554, %r1410;
	fma.rn.f32 	%r1409, %r429, %r553, %r1409;
	fma.rn.f32 	%r1408, %r430, %r552, %r1408;
	fma.rn.f32 	%r1407, %r429, %r551, %r1407;
	fma.rn.f32 	%r1414, %r432, %r550, %r1414;
	fma.rn.f32 	%r1413, %r431, %r549, %r1413;
	fma.rn.f32 	%r1412, %r432, %r548, %r1412;
	fma.rn.f32 	%r1411, %r431, %r547, %r1411;
	fma.rn.f32 	%r1418, %r434, %r546, %r1418;
	fma.rn.f32 	%r1417, %r433, %r545, %r1417;
	fma.rn.f32 	%r1416, %r434, %r544, %r1416;
	fma.rn.f32 	%r1415, %r433, %r543, %r1415;
	fma.rn.f32 	%r1422, %r436, %r542, %r1422;
	fma.rn.f32 	%r1421, %r435, %r541, %r1421;
	fma.rn.f32 	%r1420, %r436, %r540, %r1420;
	fma.rn.f32 	%r1419, %r435, %r539, %r1419;
	fma.rn.f32 	%r1426, %r422, %r538, %r1426;
	fma.rn.f32 	%r1425, %r421, %r537, %r1425;
	fma.rn.f32 	%r1424, %r422, %r536, %r1424;
	fma.rn.f32 	%r1423, %r421, %r535, %r1423;
	fma.rn.f32 	%r1430, %r424, %r534, %r1430;
	fma.rn.f32 	%r1429, %r423, %r533, %r1429;
	fma.rn.f32 	%r1428, %r424, %r532, %r1428;
	fma.rn.f32 	%r1427, %r423, %r531, %r1427;
	fma.rn.f32 	%r1434, %r426, %r530, %r1434;
	fma.rn.f32 	%r1433, %r425, %r529, %r1433;
	fma.rn.f32 	%r1432, %r426, %r528, %r1432;
	fma.rn.f32 	%r1431, %r425, %r527, %r1431;
	fma.rn.f32 	%r1438, %r428, %r526, %r1438;
	fma.rn.f32 	%r1437, %r427, %r525, %r1437;
	fma.rn.f32 	%r1436, %r428, %r524, %r1436;
	fma.rn.f32 	%r1435, %r427, %r523, %r1435;
	fma.rn.f32 	%r1442, %r430, %r522, %r1442;
	fma.rn.f32 	%r1441, %r429, %r521, %r1441;
	fma.rn.f32 	%r1440, %r430, %r520, %r1440;
	fma.rn.f32 	%r1439, %r429, %r519, %r1439;
	fma.rn.f32 	%r1446, %r432, %r518, %r1446;
	fma.rn.f32 	%r1445, %r431, %r517, %r1445;
	fma.rn.f32 	%r1444, %r432, %r516, %r1444;
	fma.rn.f32 	%r1443, %r431, %r515, %r1443;
	fma.rn.f32 	%r1450, %r434, %r514, %r1450;
	fma.rn.f32 	%r1449, %r433, %r513, %r1449;
	fma.rn.f32 	%r1448, %r434, %r512, %r1448;
	fma.rn.f32 	%r1447, %r433, %r511, %r1447;
	fma.rn.f32 	%r1454, %r436, %r510, %r1454;
	fma.rn.f32 	%r1453, %r435, %r509, %r1453;
	fma.rn.f32 	%r1452, %r436, %r508, %r1452;
	fma.rn.f32 	%r1451, %r435, %r507, %r1451;
	fma.rn.f32 	%r1458, %r422, %r506, %r1458;
	fma.rn.f32 	%r1457, %r421, %r505, %r1457;
	fma.rn.f32 	%r1456, %r422, %r504, %r1456;
	fma.rn.f32 	%r1455, %r421, %r503, %r1455;
	fma.rn.f32 	%r1462, %r424, %r502, %r1462;
	fma.rn.f32 	%r1461, %r423, %r501, %r1461;
	fma.rn.f32 	%r1460, %r424, %r500, %r1460;
	fma.rn.f32 	%r1459, %r423, %r499, %r1459;
	fma.rn.f32 	%r1466, %r426, %r498, %r1466;
	fma.rn.f32 	%r1465, %r425, %r497, %r1465;
	fma.rn.f32 	%r1464, %r426, %r496, %r1464;
	fma.rn.f32 	%r1463, %r425, %r495, %r1463;
	fma.rn.f32 	%r1470, %r428, %r494, %r1470;
	fma.rn.f32 	%r1469, %r427, %r493, %r1469;
	fma.rn.f32 	%r1468, %r428, %r492, %r1468;
	fma.rn.f32 	%r1467, %r427, %r491, %r1467;
	fma.rn.f32 	%r1474, %r430, %r490, %r1474;
	fma.rn.f32 	%r1473, %r429, %r489, %r1473;
	fma.rn.f32 	%r1472, %r430, %r488, %r1472;
	fma.rn.f32 	%r1471, %r429, %r487, %r1471;
	fma.rn.f32 	%r1478, %r432, %r486, %r1478;
	fma.rn.f32 	%r1477, %r431, %r485, %r1477;
	fma.rn.f32 	%r1476, %r432, %r484, %r1476;
	fma.rn.f32 	%r1475, %r431, %r483, %r1475;
	fma.rn.f32 	%r1482, %r434, %r482, %r1482;
	fma.rn.f32 	%r1481, %r433, %r481, %r1481;
	fma.rn.f32 	%r1480, %r434, %r480, %r1480;
	fma.rn.f32 	%r1479, %r433, %r479, %r1479;
	fma.rn.f32 	%r1484, %r436, %r478, %r1484;
	fma.rn.f32 	%r1483, %r435, %r477, %r1483;
	fma.rn.f32 	%r1485, %r435, %r476, %r1485;
	fma.rn.f32 	%r1486, %r436, %r475, %r1486;
	fma.rn.f32 	%r1487, %r421, %r474, %r1487;
	fma.rn.f32 	%r1488, %r422, %r473, %r1488;
	fma.rn.f32 	%r1489, %r421, %r472, %r1489;
	fma.rn.f32 	%r1490, %r422, %r471, %r1490;
	fma.rn.f32 	%r1491, %r423, %r470, %r1491;
	fma.rn.f32 	%r1492, %r424, %r469, %r1492;
	fma.rn.f32 	%r1493, %r423, %r468, %r1493;
	fma.rn.f32 	%r1494, %r424, %r467, %r1494;
	fma.rn.f32 	%r1495, %r425, %r466, %r1495;
	fma.rn.f32 	%r1496, %r426, %r465, %r1496;
	fma.rn.f32 	%r1497, %r425, %r464, %r1497;
	fma.rn.f32 	%r1498, %r426, %r463, %r1498;
	fma.rn.f32 	%r1499, %r427, %r462, %r1499;
	fma.rn.f32 	%r1500, %r428, %r461, %r1500;
	fma.rn.f32 	%r1501, %r427, %r460, %r1501;
	fma.rn.f32 	%r1502, %r428, %r459, %r1502;
	fma.rn.f32 	%r1503, %r429, %r458, %r1503;
	fma.rn.f32 	%r1504, %r430, %r457, %r1504;
	fma.rn.f32 	%r1505, %r429, %r456, %r1505;
	fma.rn.f32 	%r1506, %r430, %r455, %r1506;
	fma.rn.f32 	%r1507, %r431, %r454, %r1507;
	fma.rn.f32 	%r1508, %r432, %r453, %r1508;
	fma.rn.f32 	%r1509, %r431, %r452, %r1509;
	fma.rn.f32 	%r1510, %r432, %r451, %r1510;
	fma.rn.f32 	%r1511, %r433, %r450, %r1511;
	fma.rn.f32 	%r1512, %r434, %r449, %r1512;
	fma.rn.f32 	%r1513, %r433, %r448, %r1513;
	fma.rn.f32 	%r1514, %r434, %r447, %r1514;
	fma.rn.f32 	%r1515, %r435, %r446, %r1515;
	fma.rn.f32 	%r1516, %r436, %r445, %r1516;
	fma.rn.f32 	%r1517, %r435, %r444, %r1517;
	fma.rn.f32 	%r1518, %r436, %r443, %r1518;
	.loc	1 164 18                        // sk05_mlp_gateup.py:164:18
	add.s64 	%rd106, %rd31, %rd188;
	add.s64 	%rd107, %rd30, %rd188;
	add.s64 	%rd108, %rd29, %rd188;
	.loc	1 165 18                        // sk05_mlp_gateup.py:165:18
	add.s64 	%rd109, %rd28, %rd188;
	add.s64 	%rd110, %rd27, %rd188;
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	add.s64 	%rd111, %rd26, %rd188;
	add.s32 	%r571, %r1388, 1;
	setp.gt.s32 	%p6, %r571, 2;
	selp.b32 	%r1388, 0, %r571, %p6;
	.loc	1 163 30                        // sk05_mlp_gateup.py:163:30
	shl.b32 	%r572, %r1388, 14;
	bar.sync 	0;
	add.s32 	%r397, %r27, %r572;
	selp.b32 	%r398, 16, 0, %p4;
	// begin inline asm
	cp.async.cg.shared.global [ %r397 + 0 ], [ %rd106 + 0 ], 0x10, %r398;
	// end inline asm
	add.s32 	%r399, %r397, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r399 + 0 ], [ %rd107 + 0 ], 0x10, %r398;
	// end inline asm
	add.s32 	%r400, %r397, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r400 + 0 ], [ %rd108 + 0 ], 0x10, %r398;
	// end inline asm
	add.s32 	%r401, %r397, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r401 + 0 ], [ %rd109 + 0 ], 0x10, %r398;
	// end inline asm
	cp.async.commit_group;
	.loc	1 163 47                        // sk05_mlp_gateup.py:163:47
	shl.b32 	%r573, %r1388, 13;
	add.s32 	%r574, %r27, %r573;
	add.s32 	%r402, %r574, 49152;
	// begin inline asm
	cp.async.cg.shared.global [ %r402 + 0 ], [ %rd110 + 0 ], 0x10, %r398;
	// end inline asm
	add.s32 	%r403, %r574, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r403 + 0 ], [ %rd111 + 0 ], 0x10, %r398;
	// end inline asm
	cp.async.commit_group;
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	add.s64 	%rd189, %rd189, 1;
	add.s64 	%rd188, %rd188, 64;
	add.s32 	%r1386, %r1386, %r26;
	setp.ne.b64 	%p7, %rd25, %rd188;
	@%p7 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_4;
$L__BB0_1:                              // %.._crit_edge_crit_edge
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	and.b32 	%r1390, %r2, 16;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	shl.b32 	%r1389, %r2, 4;
	mov.b32 	%r1391, 0f00000000;
	mov.b32 	%r1392, %r1391;
	mov.b32 	%r1393, %r1391;
	mov.b32 	%r1394, %r1391;
	mov.b32 	%r1395, %r1391;
	mov.b32 	%r1396, %r1391;
	mov.b32 	%r1397, %r1391;
	mov.b32 	%r1398, %r1391;
	mov.b32 	%r1399, %r1391;
	mov.b32 	%r1400, %r1391;
	mov.b32 	%r1401, %r1391;
	mov.b32 	%r1402, %r1391;
	mov.b32 	%r1403, %r1391;
	mov.b32 	%r1404, %r1391;
	mov.b32 	%r1405, %r1391;
	mov.b32 	%r1406, %r1391;
	mov.b32 	%r1407, %r1391;
	mov.b32 	%r1408, %r1391;
	mov.b32 	%r1409, %r1391;
	mov.b32 	%r1410, %r1391;
	mov.b32 	%r1411, %r1391;
	mov.b32 	%r1412, %r1391;
	mov.b32 	%r1413, %r1391;
	mov.b32 	%r1414, %r1391;
	mov.b32 	%r1415, %r1391;
	mov.b32 	%r1416, %r1391;
	mov.b32 	%r1417, %r1391;
	mov.b32 	%r1418, %r1391;
	mov.b32 	%r1419, %r1391;
	mov.b32 	%r1420, %r1391;
	mov.b32 	%r1421, %r1391;
	mov.b32 	%r1422, %r1391;
	mov.b32 	%r1423, %r1391;
	mov.b32 	%r1424, %r1391;
	mov.b32 	%r1425, %r1391;
	mov.b32 	%r1426, %r1391;
	mov.b32 	%r1427, %r1391;
	mov.b32 	%r1428, %r1391;
	mov.b32 	%r1429, %r1391;
	mov.b32 	%r1430, %r1391;
	mov.b32 	%r1431, %r1391;
	mov.b32 	%r1432, %r1391;
	mov.b32 	%r1433, %r1391;
	mov.b32 	%r1434, %r1391;
	mov.b32 	%r1435, %r1391;
	mov.b32 	%r1436, %r1391;
	mov.b32 	%r1437, %r1391;
	mov.b32 	%r1438, %r1391;
	mov.b32 	%r1439, %r1391;
	mov.b32 	%r1440, %r1391;
	mov.b32 	%r1441, %r1391;
	mov.b32 	%r1442, %r1391;
	mov.b32 	%r1443, %r1391;
	mov.b32 	%r1444, %r1391;
	mov.b32 	%r1445, %r1391;
	mov.b32 	%r1446, %r1391;
	mov.b32 	%r1447, %r1391;
	mov.b32 	%r1448, %r1391;
	mov.b32 	%r1449, %r1391;
	mov.b32 	%r1450, %r1391;
	mov.b32 	%r1451, %r1391;
	mov.b32 	%r1452, %r1391;
	mov.b32 	%r1453, %r1391;
	mov.b32 	%r1454, %r1391;
	mov.b32 	%r1455, %r1391;
	mov.b32 	%r1456, %r1391;
	mov.b32 	%r1457, %r1391;
	mov.b32 	%r1458, %r1391;
	mov.b32 	%r1459, %r1391;
	mov.b32 	%r1460, %r1391;
	mov.b32 	%r1461, %r1391;
	mov.b32 	%r1462, %r1391;
	mov.b32 	%r1463, %r1391;
	mov.b32 	%r1464, %r1391;
	mov.b32 	%r1465, %r1391;
	mov.b32 	%r1466, %r1391;
	mov.b32 	%r1467, %r1391;
	mov.b32 	%r1468, %r1391;
	mov.b32 	%r1469, %r1391;
	mov.b32 	%r1470, %r1391;
	mov.b32 	%r1471, %r1391;
	mov.b32 	%r1472, %r1391;
	mov.b32 	%r1473, %r1391;
	mov.b32 	%r1474, %r1391;
	mov.b32 	%r1475, %r1391;
	mov.b32 	%r1476, %r1391;
	mov.b32 	%r1477, %r1391;
	mov.b32 	%r1478, %r1391;
	mov.b32 	%r1479, %r1391;
	mov.b32 	%r1480, %r1391;
	mov.b32 	%r1481, %r1391;
	mov.b32 	%r1482, %r1391;
	mov.b32 	%r1483, %r1391;
	mov.b32 	%r1484, %r1391;
	mov.b32 	%r1485, %r1391;
	mov.b32 	%r1486, %r1391;
	mov.b32 	%r1487, %r1391;
	mov.b32 	%r1488, %r1391;
	mov.b32 	%r1489, %r1391;
	mov.b32 	%r1490, %r1391;
	mov.b32 	%r1491, %r1391;
	mov.b32 	%r1492, %r1391;
	mov.b32 	%r1493, %r1391;
	mov.b32 	%r1494, %r1391;
	mov.b32 	%r1495, %r1391;
	mov.b32 	%r1496, %r1391;
	mov.b32 	%r1497, %r1391;
	mov.b32 	%r1498, %r1391;
	mov.b32 	%r1499, %r1391;
	mov.b32 	%r1500, %r1391;
	mov.b32 	%r1501, %r1391;
	mov.b32 	%r1502, %r1391;
	mov.b32 	%r1503, %r1391;
	mov.b32 	%r1504, %r1391;
	mov.b32 	%r1505, %r1391;
	mov.b32 	%r1506, %r1391;
	mov.b32 	%r1507, %r1391;
	mov.b32 	%r1508, %r1391;
	mov.b32 	%r1509, %r1391;
	mov.b32 	%r1510, %r1391;
	mov.b32 	%r1511, %r1391;
	mov.b32 	%r1512, %r1391;
	mov.b32 	%r1513, %r1391;
	mov.b32 	%r1514, %r1391;
	mov.b32 	%r1515, %r1391;
	mov.b32 	%r1516, %r1391;
	mov.b32 	%r1517, %r1391;
	mov.b32 	%r1518, %r1391;
$L__BB0_4:                              // %._crit_edge
	.loc	1 153 45                        // sk05_mlp_gateup.py:153:45
	shl.b32 	%r805, %r7, 3;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r806, %r4, %r805;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r807, %r806, %r22;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r808, %r1, %r3;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r809, %r808, %r21;
	.loc	1 152 45                        // sk05_mlp_gateup.py:152:45
	and.b32 	%r810, %r2, 240;
	bfe.u32 	%r811, %r2, 4, 4;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r812, %r811, %r1;
	or.b32 	%r813, %r812, 240;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r814, %r813, %r21;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r815, %r812, 224;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r816, %r815, %r21;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r817, %r812, 208;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r818, %r817, %r21;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r819, %r812, 192;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r820, %r819, %r21;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r821, %r812, 176;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r822, %r821, %r21;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r823, %r812, 160;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r824, %r823, %r21;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r825, %r812, 144;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r826, %r825, %r21;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r827, %r812, 128;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r828, %r827, %r21;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r829, %r812, 112;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r830, %r829, %r21;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r831, %r812, 96;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r832, %r831, %r21;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r833, %r812, 80;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r834, %r833, %r21;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r835, %r812, 64;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r836, %r835, %r21;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r837, %r812, 48;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r838, %r837, %r21;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r839, %r812, 32;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r840, %r839, %r21;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r841, %r812, 16;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r842, %r841, %r21;
	rem.s32 	%r843, %r812, %r21;
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 167 38                        // sk05_mlp_gateup.py:167:38
	mad.wide.s32 	%rd113, %r809, 4, %rd36;
	.loc	1 167 24                        // sk05_mlp_gateup.py:167:24
	// begin inline asm
	mov.u32 %r576, 0x0;
	ld.global.b32 { %r576 }, [ %rd113 + 0 ];
	// end inline asm
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	and.b32 	%r844, %r2, 7;
	shl.b32 	%r845, %r844, 3;
	shl.b32 	%r846, %r810, 2;
	and.b32 	%r847, %r2, 8;
	shr.u32 	%r848, %r847, 1;
	add.s32 	%r849, %r188, %r845;
	add.s32 	%r850, %r849, %r846;
	add.s32 	%r575, %r850, %r848;
	// begin inline asm
	st.shared.b32 [ %r575 + 0 ], %r576;
	// end inline asm
	bar.sync 	0;
	and.b32 	%r851, %r16, 56;
	and.b32 	%r852, %r2, 192;
	add.s32 	%r853, %r188, %r851;
	add.s32 	%r854, %r853, %r852;
	ld.shared.v2.b32 	{%r855, %r856}, [%r854];
	ld.shared.v2.b32 	{%r857, %r858}, [%r854+256];
	ld.shared.v2.b32 	{%r859, %r860}, [%r854+512];
	ld.shared.v2.b32 	{%r861, %r862}, [%r854+768];
	.loc	1 168 38                        // sk05_mlp_gateup.py:168:38
	mad.wide.s32 	%rd114, %r8, 4, %rd37;
	mad.wide.s32 	%rd115, %r9, 4, %rd37;
	mad.wide.s32 	%rd116, %r10, 4, %rd37;
	mad.wide.s32 	%rd117, %r11, 4, %rd37;
	mad.wide.s32 	%rd118, %r12, 4, %rd37;
	mad.wide.s32 	%rd119, %r13, 4, %rd37;
	mad.wide.s32 	%rd120, %r14, 4, %rd37;
	mad.wide.s32 	%rd121, %r15, 4, %rd37;
	.loc	1 168 24                        // sk05_mlp_gateup.py:168:24
	// begin inline asm
	mov.u32 %r577, 0x0;
	mov.u32 %r578, 0x0;
	ld.global.v2.b32 { %r577, %r578 }, [ %rd114 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r579, 0x0;
	mov.u32 %r580, 0x0;
	ld.global.v2.b32 { %r579, %r580 }, [ %rd115 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r581, 0x0;
	mov.u32 %r582, 0x0;
	ld.global.v2.b32 { %r581, %r582 }, [ %rd116 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r583, 0x0;
	mov.u32 %r584, 0x0;
	ld.global.v2.b32 { %r583, %r584 }, [ %rd117 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r585, 0x0;
	mov.u32 %r586, 0x0;
	ld.global.v2.b32 { %r585, %r586 }, [ %rd118 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r587, 0x0;
	mov.u32 %r588, 0x0;
	ld.global.v2.b32 { %r587, %r588 }, [ %rd119 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r589, 0x0;
	mov.u32 %r590, 0x0;
	ld.global.v2.b32 { %r589, %r590 }, [ %rd120 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r591, 0x0;
	mov.u32 %r592, 0x0;
	ld.global.v2.b32 { %r591, %r592 }, [ %rd121 + 0 ];
	// end inline asm
	.loc	1 169 49                        // sk05_mlp_gateup.py:169:49
	mul.lo.s32 	%r863, %r843, %r25;
	mul.lo.s32 	%r864, %r842, %r25;
	mul.lo.s32 	%r865, %r840, %r25;
	mul.lo.s32 	%r866, %r838, %r25;
	mul.lo.s32 	%r867, %r836, %r25;
	mul.lo.s32 	%r868, %r834, %r25;
	mul.lo.s32 	%r869, %r832, %r25;
	mul.lo.s32 	%r870, %r830, %r25;
	mul.lo.s32 	%r871, %r828, %r25;
	mul.lo.s32 	%r872, %r826, %r25;
	mul.lo.s32 	%r873, %r824, %r25;
	mul.lo.s32 	%r874, %r822, %r25;
	mul.lo.s32 	%r875, %r820, %r25;
	mul.lo.s32 	%r876, %r818, %r25;
	mul.lo.s32 	%r877, %r816, %r25;
	mul.lo.s32 	%r878, %r814, %r25;
	.loc	1 169 31                        // sk05_mlp_gateup.py:169:31
	mad.wide.s32 	%rd154, %r863, 2, %rd35;
	mad.wide.s32 	%rd155, %r864, 2, %rd35;
	mad.wide.s32 	%rd156, %r865, 2, %rd35;
	mad.wide.s32 	%rd157, %r866, 2, %rd35;
	mad.wide.s32 	%rd158, %r867, 2, %rd35;
	mad.wide.s32 	%rd159, %r868, 2, %rd35;
	mad.wide.s32 	%rd160, %r869, 2, %rd35;
	mad.wide.s32 	%rd161, %r870, 2, %rd35;
	mad.wide.s32 	%rd162, %r871, 2, %rd35;
	mad.wide.s32 	%rd163, %r872, 2, %rd35;
	mad.wide.s32 	%rd164, %r873, 2, %rd35;
	mad.wide.s32 	%rd165, %r874, 2, %rd35;
	mad.wide.s32 	%rd166, %r875, 2, %rd35;
	mad.wide.s32 	%rd167, %r876, 2, %rd35;
	mad.wide.s32 	%rd168, %r877, 2, %rd35;
	mad.wide.s32 	%rd169, %r878, 2, %rd35;
	.loc	1 169 64                        // sk05_mlp_gateup.py:169:64
	mul.wide.s32 	%rd170, %r807, 2;
	add.s64 	%rd122, %rd154, %rd170;
	add.s64 	%rd123, %rd155, %rd170;
	add.s64 	%rd124, %rd156, %rd170;
	add.s64 	%rd125, %rd157, %rd170;
	add.s64 	%rd126, %rd158, %rd170;
	add.s64 	%rd127, %rd159, %rd170;
	add.s64 	%rd128, %rd160, %rd170;
	add.s64 	%rd129, %rd161, %rd170;
	add.s64 	%rd130, %rd162, %rd170;
	add.s64 	%rd131, %rd163, %rd170;
	add.s64 	%rd132, %rd164, %rd170;
	add.s64 	%rd133, %rd165, %rd170;
	add.s64 	%rd134, %rd166, %rd170;
	add.s64 	%rd135, %rd167, %rd170;
	add.s64 	%rd136, %rd168, %rd170;
	add.s64 	%rd137, %rd169, %rd170;
	.loc	1 169 19                        // sk05_mlp_gateup.py:169:19
	// begin inline asm
	mov.u32 %r594, 0x0;
	mov.u32 %r595, 0x0;
	mov.u32 %r596, 0x0;
	mov.u32 %r597, 0x0;
	ld.global.v4.b32 { %r594, %r595, %r596, %r597 }, [ %rd122 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r599, 0x0;
	mov.u32 %r600, 0x0;
	mov.u32 %r601, 0x0;
	mov.u32 %r602, 0x0;
	ld.global.v4.b32 { %r599, %r600, %r601, %r602 }, [ %rd123 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r604, 0x0;
	mov.u32 %r605, 0x0;
	mov.u32 %r606, 0x0;
	mov.u32 %r607, 0x0;
	ld.global.v4.b32 { %r604, %r605, %r606, %r607 }, [ %rd124 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r609, 0x0;
	mov.u32 %r610, 0x0;
	mov.u32 %r611, 0x0;
	mov.u32 %r612, 0x0;
	ld.global.v4.b32 { %r609, %r610, %r611, %r612 }, [ %rd125 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r613, 0x0;
	mov.u32 %r614, 0x0;
	mov.u32 %r615, 0x0;
	mov.u32 %r616, 0x0;
	ld.global.v4.b32 { %r613, %r614, %r615, %r616 }, [ %rd126 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r617, 0x0;
	mov.u32 %r618, 0x0;
	mov.u32 %r619, 0x0;
	mov.u32 %r620, 0x0;
	ld.global.v4.b32 { %r617, %r618, %r619, %r620 }, [ %rd127 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r621, 0x0;
	mov.u32 %r622, 0x0;
	mov.u32 %r623, 0x0;
	mov.u32 %r624, 0x0;
	ld.global.v4.b32 { %r621, %r622, %r623, %r624 }, [ %rd128 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r625, 0x0;
	mov.u32 %r626, 0x0;
	mov.u32 %r627, 0x0;
	mov.u32 %r628, 0x0;
	ld.global.v4.b32 { %r625, %r626, %r627, %r628 }, [ %rd129 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r629, 0x0;
	mov.u32 %r630, 0x0;
	mov.u32 %r631, 0x0;
	mov.u32 %r632, 0x0;
	ld.global.v4.b32 { %r629, %r630, %r631, %r632 }, [ %rd130 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r633, 0x0;
	mov.u32 %r634, 0x0;
	mov.u32 %r635, 0x0;
	mov.u32 %r636, 0x0;
	ld.global.v4.b32 { %r633, %r634, %r635, %r636 }, [ %rd131 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r637, 0x0;
	mov.u32 %r638, 0x0;
	mov.u32 %r639, 0x0;
	mov.u32 %r640, 0x0;
	ld.global.v4.b32 { %r637, %r638, %r639, %r640 }, [ %rd132 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r641, 0x0;
	mov.u32 %r642, 0x0;
	mov.u32 %r643, 0x0;
	mov.u32 %r644, 0x0;
	ld.global.v4.b32 { %r641, %r642, %r643, %r644 }, [ %rd133 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r645, 0x0;
	mov.u32 %r646, 0x0;
	mov.u32 %r647, 0x0;
	mov.u32 %r648, 0x0;
	ld.global.v4.b32 { %r645, %r646, %r647, %r648 }, [ %rd134 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r649, 0x0;
	mov.u32 %r650, 0x0;
	mov.u32 %r651, 0x0;
	mov.u32 %r652, 0x0;
	ld.global.v4.b32 { %r649, %r650, %r651, %r652 }, [ %rd135 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r653, 0x0;
	mov.u32 %r654, 0x0;
	mov.u32 %r655, 0x0;
	mov.u32 %r656, 0x0;
	ld.global.v4.b32 { %r653, %r654, %r655, %r656 }, [ %rd136 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r657, 0x0;
	mov.u32 %r658, 0x0;
	mov.u32 %r659, 0x0;
	mov.u32 %r660, 0x0;
	ld.global.v4.b32 { %r657, %r658, %r659, %r660 }, [ %rd137 + 0 ];
	// end inline asm
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	bar.sync 	0;
	shl.b32 	%r879, %r2, 7;
	and.b32 	%r880, %r879, 15360;
	shl.b32 	%r881, %r844, 4;
	or.b32 	%r882, %r880, %r881;
	xor.b32 	%r883, %r882, %r810;
	add.s32 	%r593, %r188, %r883;
	// begin inline asm
	st.shared.v4.b32 [ %r593 + 0 ], { %r594, %r595, %r596, %r597 };
	// end inline asm
	add.s32 	%r598, %r593, 256;
	// begin inline asm
	st.shared.v4.b32 [ %r598 + 0 ], { %r599, %r600, %r601, %r602 };
	// end inline asm
	add.s32 	%r603, %r593, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r603 + 0 ], { %r604, %r605, %r606, %r607 };
	// end inline asm
	add.s32 	%r608, %r593, 768;
	// begin inline asm
	st.shared.v4.b32 [ %r608 + 0 ], { %r609, %r610, %r611, %r612 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r884, %r844, 11;
	shl.b32 	%r885, %r7, 4;
	shl.b32 	%r886, %r852, 2;
	setp.eq.b32 	%p24, %r1390, 0;
	shl.b32 	%r887, %r1390, 1;
	shr.u32 	%r888, %r6, 1;
	or.b32 	%r889, %r885, %r886;
	or.b32 	%r890, %r887, %r888;
	xor.b32 	%r891, %r889, %r890;
	or.b32 	%r892, %r891, %r884;
	add.s32 	%r893, %r188, %r892;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r894, %r895, %r896, %r897}, [%r893];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r898, %r899, %r900, %r901}, [%r893+1024];
	xor.b32 	%r902, %r892, 64;
	add.s32 	%r903, %r188, %r902;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r904, %r905, %r906, %r907}, [%r903];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r908, %r909, %r910, %r911}, [%r903+1024];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r593 + 0 ], { %r613, %r614, %r615, %r616 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r598 + 0 ], { %r617, %r618, %r619, %r620 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r603 + 0 ], { %r621, %r622, %r623, %r624 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r608 + 0 ], { %r625, %r626, %r627, %r628 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r912, %r913, %r914, %r915}, [%r893];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r916, %r917, %r918, %r919}, [%r893+1024];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r920, %r921, %r922, %r923}, [%r903];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r924, %r925, %r926, %r927}, [%r903+1024];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r593 + 0 ], { %r629, %r630, %r631, %r632 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r598 + 0 ], { %r633, %r634, %r635, %r636 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r603 + 0 ], { %r637, %r638, %r639, %r640 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r608 + 0 ], { %r641, %r642, %r643, %r644 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r928, %r929, %r930, %r931}, [%r893];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r932, %r933, %r934, %r935}, [%r893+1024];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r936, %r937, %r938, %r939}, [%r903];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r940, %r941, %r942, %r943}, [%r903+1024];
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r593 + 0 ], { %r645, %r646, %r647, %r648 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r598 + 0 ], { %r649, %r650, %r651, %r652 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r603 + 0 ], { %r653, %r654, %r655, %r656 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r608 + 0 ], { %r657, %r658, %r659, %r660 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r944, %r945, %r946, %r947}, [%r893];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r948, %r949, %r950, %r951}, [%r893+1024];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r952, %r953, %r954, %r955}, [%r903];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r956, %r957, %r958, %r959}, [%r903+1024];
	.loc	1 176 31                        // sk05_mlp_gateup.py:176:31
	setp.lt.s32 	%p25, %r812, %r21;
	setp.lt.s32 	%p26, %r841, %r21;
	setp.lt.s32 	%p27, %r839, %r21;
	setp.lt.s32 	%p28, %r837, %r21;
	setp.lt.s32 	%p29, %r835, %r21;
	setp.lt.s32 	%p30, %r833, %r21;
	setp.lt.s32 	%p31, %r831, %r21;
	setp.lt.s32 	%p32, %r829, %r21;
	setp.lt.s32 	%p33, %r827, %r21;
	setp.lt.s32 	%p34, %r825, %r21;
	setp.lt.s32 	%p35, %r823, %r21;
	setp.lt.s32 	%p36, %r821, %r21;
	setp.lt.s32 	%p37, %r819, %r21;
	setp.lt.s32 	%p38, %r817, %r21;
	setp.lt.s32 	%p39, %r815, %r21;
	setp.lt.s32 	%p40, %r813, %r21;
	.loc	1 176 54                        // sk05_mlp_gateup.py:176:54
	setp.lt.s32 	%p41, %r806, %r22;
	.loc	1 176 37                        // sk05_mlp_gateup.py:176:37
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
	.loc	1 174 35                        // sk05_mlp_gateup.py:174:35
	mul.lo.s32 	%r960, %r812, %r24;
	mul.lo.s32 	%r961, %r841, %r24;
	mul.lo.s32 	%r962, %r839, %r24;
	mul.lo.s32 	%r963, %r837, %r24;
	mul.lo.s32 	%r964, %r835, %r24;
	mul.lo.s32 	%r965, %r833, %r24;
	mul.lo.s32 	%r966, %r831, %r24;
	mul.lo.s32 	%r967, %r829, %r24;
	mul.lo.s32 	%r968, %r827, %r24;
	mul.lo.s32 	%r969, %r825, %r24;
	mul.lo.s32 	%r970, %r823, %r24;
	mul.lo.s32 	%r971, %r821, %r24;
	mul.lo.s32 	%r972, %r819, %r24;
	mul.lo.s32 	%r973, %r817, %r24;
	mul.lo.s32 	%r974, %r815, %r24;
	mul.lo.s32 	%r975, %r813, %r24;
	.loc	1 174 18                        // sk05_mlp_gateup.py:174:18
	mad.wide.s32 	%rd171, %r960, 2, %rd34;
	mad.wide.s32 	%rd172, %r961, 2, %rd34;
	mad.wide.s32 	%rd173, %r962, 2, %rd34;
	mad.wide.s32 	%rd174, %r963, 2, %rd34;
	mad.wide.s32 	%rd175, %r964, 2, %rd34;
	mad.wide.s32 	%rd176, %r965, 2, %rd34;
	mad.wide.s32 	%rd177, %r966, 2, %rd34;
	mad.wide.s32 	%rd178, %r967, 2, %rd34;
	mad.wide.s32 	%rd179, %r968, 2, %rd34;
	mad.wide.s32 	%rd180, %r969, 2, %rd34;
	mad.wide.s32 	%rd181, %r970, 2, %rd34;
	mad.wide.s32 	%rd182, %r971, 2, %rd34;
	mad.wide.s32 	%rd183, %r972, 2, %rd34;
	mad.wide.s32 	%rd184, %r973, 2, %rd34;
	mad.wide.s32 	%rd185, %r974, 2, %rd34;
	mad.wide.s32 	%rd186, %r975, 2, %rd34;
	.loc	1 174 50                        // sk05_mlp_gateup.py:174:50
	mul.wide.s32 	%rd187, %r806, 2;
	add.s64 	%rd138, %rd171, %rd187;
	add.s64 	%rd139, %rd172, %rd187;
	add.s64 	%rd140, %rd173, %rd187;
	add.s64 	%rd141, %rd174, %rd187;
	add.s64 	%rd142, %rd175, %rd187;
	add.s64 	%rd143, %rd176, %rd187;
	add.s64 	%rd144, %rd177, %rd187;
	add.s64 	%rd145, %rd178, %rd187;
	add.s64 	%rd146, %rd179, %rd187;
	add.s64 	%rd147, %rd180, %rd187;
	add.s64 	%rd148, %rd181, %rd187;
	add.s64 	%rd149, %rd182, %rd187;
	add.s64 	%rd150, %rd183, %rd187;
	add.s64 	%rd151, %rd184, %rd187;
	add.s64 	%rd152, %rd185, %rd187;
	add.s64 	%rd153, %rd186, %rd187;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r976, %r1488, %r861;
	mul.f32 	%r977, %r1487, %r861;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs33, %rs34}, %r944;
	cvt.f32.bf16 	%r978, %rs34;
	cvt.f32.bf16 	%r979, %rs33;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r980, %r977, %r577, %r979;
	fma.rn.f32 	%r981, %r976, %r578, %r978;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r982, %r1392, %r855;
	mul.f32 	%r983, %r1391, %r855;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs35, %rs36}, %r894;
	cvt.f32.bf16 	%r984, %rs36;
	cvt.f32.bf16 	%r985, %rs35;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r986, %r983, %r577, %r985;
	fma.rn.f32 	%r987, %r982, %r578, %r984;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r662, %r987, %r986;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r988, %r1394, %r856;
	mul.f32 	%r989, %r1393, %r856;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs37, %rs38}, %r895;
	cvt.f32.bf16 	%r990, %rs38;
	cvt.f32.bf16 	%r991, %rs37;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r992, %r989, %r577, %r991;
	fma.rn.f32 	%r993, %r988, %r578, %r990;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r667, %r993, %r992;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r994, %r1424, %r857;
	mul.f32 	%r995, %r1423, %r857;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs39, %rs40}, %r912;
	cvt.f32.bf16 	%r996, %rs40;
	cvt.f32.bf16 	%r997, %rs39;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r998, %r995, %r577, %r997;
	fma.rn.f32 	%r999, %r994, %r578, %r996;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r663, %r999, %r998;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1000, %r1426, %r858;
	mul.f32 	%r1001, %r1425, %r858;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs41, %rs42}, %r913;
	cvt.f32.bf16 	%r1002, %rs42;
	cvt.f32.bf16 	%r1003, %rs41;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1004, %r1001, %r577, %r1003;
	fma.rn.f32 	%r1005, %r1000, %r578, %r1002;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r668, %r1005, %r1004;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1006, %r1456, %r859;
	mul.f32 	%r1007, %r1455, %r859;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs43, %rs44}, %r928;
	cvt.f32.bf16 	%r1008, %rs44;
	cvt.f32.bf16 	%r1009, %rs43;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1010, %r1007, %r577, %r1009;
	fma.rn.f32 	%r1011, %r1006, %r578, %r1008;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r664, %r1011, %r1010;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1012, %r1458, %r860;
	mul.f32 	%r1013, %r1457, %r860;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs45, %rs46}, %r929;
	cvt.f32.bf16 	%r1014, %rs46;
	cvt.f32.bf16 	%r1015, %rs45;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1016, %r1013, %r577, %r1015;
	fma.rn.f32 	%r1017, %r1012, %r578, %r1014;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r669, %r1017, %r1016;
	cvt.rn.bf16x2.f32 	%r665, %r981, %r980;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1018, %r1490, %r862;
	mul.f32 	%r1019, %r1489, %r862;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs47, %rs48}, %r945;
	cvt.f32.bf16 	%r1020, %rs48;
	cvt.f32.bf16 	%r1021, %rs47;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1022, %r1019, %r577, %r1021;
	fma.rn.f32 	%r1023, %r1018, %r578, %r1020;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r670, %r1023, %r1022;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1024, %r1492, %r861;
	mul.f32 	%r1025, %r1491, %r861;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs49, %rs50}, %r946;
	cvt.f32.bf16 	%r1026, %rs50;
	cvt.f32.bf16 	%r1027, %rs49;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1028, %r1025, %r579, %r1027;
	fma.rn.f32 	%r1029, %r1024, %r580, %r1026;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1030, %r1396, %r855;
	mul.f32 	%r1031, %r1395, %r855;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs51, %rs52}, %r896;
	cvt.f32.bf16 	%r1032, %rs52;
	cvt.f32.bf16 	%r1033, %rs51;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1034, %r1031, %r579, %r1033;
	fma.rn.f32 	%r1035, %r1030, %r580, %r1032;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r682, %r1035, %r1034;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1036, %r1398, %r856;
	mul.f32 	%r1037, %r1397, %r856;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs53, %rs54}, %r897;
	cvt.f32.bf16 	%r1038, %rs54;
	cvt.f32.bf16 	%r1039, %rs53;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1040, %r1037, %r579, %r1039;
	fma.rn.f32 	%r1041, %r1036, %r580, %r1038;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r687, %r1041, %r1040;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1042, %r1428, %r857;
	mul.f32 	%r1043, %r1427, %r857;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs55, %rs56}, %r914;
	cvt.f32.bf16 	%r1044, %rs56;
	cvt.f32.bf16 	%r1045, %rs55;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1046, %r1043, %r579, %r1045;
	fma.rn.f32 	%r1047, %r1042, %r580, %r1044;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r683, %r1047, %r1046;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1048, %r1430, %r858;
	mul.f32 	%r1049, %r1429, %r858;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs57, %rs58}, %r915;
	cvt.f32.bf16 	%r1050, %rs58;
	cvt.f32.bf16 	%r1051, %rs57;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1052, %r1049, %r579, %r1051;
	fma.rn.f32 	%r1053, %r1048, %r580, %r1050;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r688, %r1053, %r1052;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1054, %r1460, %r859;
	mul.f32 	%r1055, %r1459, %r859;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs59, %rs60}, %r930;
	cvt.f32.bf16 	%r1056, %rs60;
	cvt.f32.bf16 	%r1057, %rs59;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1058, %r1055, %r579, %r1057;
	fma.rn.f32 	%r1059, %r1054, %r580, %r1056;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r684, %r1059, %r1058;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1060, %r1462, %r860;
	mul.f32 	%r1061, %r1461, %r860;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs61, %rs62}, %r931;
	cvt.f32.bf16 	%r1062, %rs62;
	cvt.f32.bf16 	%r1063, %rs61;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1064, %r1061, %r579, %r1063;
	fma.rn.f32 	%r1065, %r1060, %r580, %r1062;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r689, %r1065, %r1064;
	cvt.rn.bf16x2.f32 	%r685, %r1029, %r1028;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1066, %r1494, %r862;
	mul.f32 	%r1067, %r1493, %r862;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs63, %rs64}, %r947;
	cvt.f32.bf16 	%r1068, %rs64;
	cvt.f32.bf16 	%r1069, %rs63;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1070, %r1067, %r579, %r1069;
	fma.rn.f32 	%r1071, %r1066, %r580, %r1068;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r690, %r1071, %r1070;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1072, %r1496, %r861;
	mul.f32 	%r1073, %r1495, %r861;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs65, %rs66}, %r952;
	cvt.f32.bf16 	%r1074, %rs66;
	cvt.f32.bf16 	%r1075, %rs65;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1076, %r1073, %r581, %r1075;
	fma.rn.f32 	%r1077, %r1072, %r582, %r1074;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1078, %r1400, %r855;
	mul.f32 	%r1079, %r1399, %r855;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs67, %rs68}, %r904;
	cvt.f32.bf16 	%r1080, %rs68;
	cvt.f32.bf16 	%r1081, %rs67;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1082, %r1079, %r581, %r1081;
	fma.rn.f32 	%r1083, %r1078, %r582, %r1080;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r702, %r1083, %r1082;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1084, %r1402, %r856;
	mul.f32 	%r1085, %r1401, %r856;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs69, %rs70}, %r905;
	cvt.f32.bf16 	%r1086, %rs70;
	cvt.f32.bf16 	%r1087, %rs69;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1088, %r1085, %r581, %r1087;
	fma.rn.f32 	%r1089, %r1084, %r582, %r1086;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r707, %r1089, %r1088;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1090, %r1432, %r857;
	mul.f32 	%r1091, %r1431, %r857;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs71, %rs72}, %r920;
	cvt.f32.bf16 	%r1092, %rs72;
	cvt.f32.bf16 	%r1093, %rs71;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1094, %r1091, %r581, %r1093;
	fma.rn.f32 	%r1095, %r1090, %r582, %r1092;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r703, %r1095, %r1094;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1096, %r1434, %r858;
	mul.f32 	%r1097, %r1433, %r858;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs73, %rs74}, %r921;
	cvt.f32.bf16 	%r1098, %rs74;
	cvt.f32.bf16 	%r1099, %rs73;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1100, %r1097, %r581, %r1099;
	fma.rn.f32 	%r1101, %r1096, %r582, %r1098;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r708, %r1101, %r1100;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1102, %r1464, %r859;
	mul.f32 	%r1103, %r1463, %r859;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs75, %rs76}, %r936;
	cvt.f32.bf16 	%r1104, %rs76;
	cvt.f32.bf16 	%r1105, %rs75;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1106, %r1103, %r581, %r1105;
	fma.rn.f32 	%r1107, %r1102, %r582, %r1104;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r704, %r1107, %r1106;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1108, %r1466, %r860;
	mul.f32 	%r1109, %r1465, %r860;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs77, %rs78}, %r937;
	cvt.f32.bf16 	%r1110, %rs78;
	cvt.f32.bf16 	%r1111, %rs77;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1112, %r1109, %r581, %r1111;
	fma.rn.f32 	%r1113, %r1108, %r582, %r1110;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r709, %r1113, %r1112;
	cvt.rn.bf16x2.f32 	%r705, %r1077, %r1076;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1114, %r1498, %r862;
	mul.f32 	%r1115, %r1497, %r862;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs79, %rs80}, %r953;
	cvt.f32.bf16 	%r1116, %rs80;
	cvt.f32.bf16 	%r1117, %rs79;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1118, %r1115, %r581, %r1117;
	fma.rn.f32 	%r1119, %r1114, %r582, %r1116;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r710, %r1119, %r1118;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1120, %r1500, %r861;
	mul.f32 	%r1121, %r1499, %r861;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs81, %rs82}, %r954;
	cvt.f32.bf16 	%r1122, %rs82;
	cvt.f32.bf16 	%r1123, %rs81;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1124, %r1121, %r583, %r1123;
	fma.rn.f32 	%r1125, %r1120, %r584, %r1122;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1126, %r1404, %r855;
	mul.f32 	%r1127, %r1403, %r855;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs83, %rs84}, %r906;
	cvt.f32.bf16 	%r1128, %rs84;
	cvt.f32.bf16 	%r1129, %rs83;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1130, %r1127, %r583, %r1129;
	fma.rn.f32 	%r1131, %r1126, %r584, %r1128;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r722, %r1131, %r1130;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1132, %r1406, %r856;
	mul.f32 	%r1133, %r1405, %r856;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs85, %rs86}, %r907;
	cvt.f32.bf16 	%r1134, %rs86;
	cvt.f32.bf16 	%r1135, %rs85;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1136, %r1133, %r583, %r1135;
	fma.rn.f32 	%r1137, %r1132, %r584, %r1134;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r727, %r1137, %r1136;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1138, %r1436, %r857;
	mul.f32 	%r1139, %r1435, %r857;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs87, %rs88}, %r922;
	cvt.f32.bf16 	%r1140, %rs88;
	cvt.f32.bf16 	%r1141, %rs87;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1142, %r1139, %r583, %r1141;
	fma.rn.f32 	%r1143, %r1138, %r584, %r1140;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r723, %r1143, %r1142;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1144, %r1438, %r858;
	mul.f32 	%r1145, %r1437, %r858;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs89, %rs90}, %r923;
	cvt.f32.bf16 	%r1146, %rs90;
	cvt.f32.bf16 	%r1147, %rs89;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1148, %r1145, %r583, %r1147;
	fma.rn.f32 	%r1149, %r1144, %r584, %r1146;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r728, %r1149, %r1148;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1150, %r1468, %r859;
	mul.f32 	%r1151, %r1467, %r859;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs91, %rs92}, %r938;
	cvt.f32.bf16 	%r1152, %rs92;
	cvt.f32.bf16 	%r1153, %rs91;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1154, %r1151, %r583, %r1153;
	fma.rn.f32 	%r1155, %r1150, %r584, %r1152;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r724, %r1155, %r1154;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1156, %r1470, %r860;
	mul.f32 	%r1157, %r1469, %r860;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs93, %rs94}, %r939;
	cvt.f32.bf16 	%r1158, %rs94;
	cvt.f32.bf16 	%r1159, %rs93;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1160, %r1157, %r583, %r1159;
	fma.rn.f32 	%r1161, %r1156, %r584, %r1158;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r729, %r1161, %r1160;
	cvt.rn.bf16x2.f32 	%r725, %r1125, %r1124;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1162, %r1502, %r862;
	mul.f32 	%r1163, %r1501, %r862;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs95, %rs96}, %r955;
	cvt.f32.bf16 	%r1164, %rs96;
	cvt.f32.bf16 	%r1165, %rs95;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1166, %r1163, %r583, %r1165;
	fma.rn.f32 	%r1167, %r1162, %r584, %r1164;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r730, %r1167, %r1166;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1168, %r1504, %r861;
	mul.f32 	%r1169, %r1503, %r861;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs97, %rs98}, %r948;
	cvt.f32.bf16 	%r1170, %rs98;
	cvt.f32.bf16 	%r1171, %rs97;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1172, %r1169, %r585, %r1171;
	fma.rn.f32 	%r1173, %r1168, %r586, %r1170;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1174, %r1408, %r855;
	mul.f32 	%r1175, %r1407, %r855;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs99, %rs100}, %r898;
	cvt.f32.bf16 	%r1176, %rs100;
	cvt.f32.bf16 	%r1177, %rs99;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1178, %r1175, %r585, %r1177;
	fma.rn.f32 	%r1179, %r1174, %r586, %r1176;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r672, %r1179, %r1178;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1180, %r1410, %r856;
	mul.f32 	%r1181, %r1409, %r856;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs101, %rs102}, %r899;
	cvt.f32.bf16 	%r1182, %rs102;
	cvt.f32.bf16 	%r1183, %rs101;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1184, %r1181, %r585, %r1183;
	fma.rn.f32 	%r1185, %r1180, %r586, %r1182;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r677, %r1185, %r1184;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1186, %r1440, %r857;
	mul.f32 	%r1187, %r1439, %r857;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs103, %rs104}, %r916;
	cvt.f32.bf16 	%r1188, %rs104;
	cvt.f32.bf16 	%r1189, %rs103;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1190, %r1187, %r585, %r1189;
	fma.rn.f32 	%r1191, %r1186, %r586, %r1188;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r673, %r1191, %r1190;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1192, %r1442, %r858;
	mul.f32 	%r1193, %r1441, %r858;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs105, %rs106}, %r917;
	cvt.f32.bf16 	%r1194, %rs106;
	cvt.f32.bf16 	%r1195, %rs105;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1196, %r1193, %r585, %r1195;
	fma.rn.f32 	%r1197, %r1192, %r586, %r1194;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r678, %r1197, %r1196;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1198, %r1472, %r859;
	mul.f32 	%r1199, %r1471, %r859;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs107, %rs108}, %r932;
	cvt.f32.bf16 	%r1200, %rs108;
	cvt.f32.bf16 	%r1201, %rs107;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1202, %r1199, %r585, %r1201;
	fma.rn.f32 	%r1203, %r1198, %r586, %r1200;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r674, %r1203, %r1202;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1204, %r1474, %r860;
	mul.f32 	%r1205, %r1473, %r860;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs109, %rs110}, %r933;
	cvt.f32.bf16 	%r1206, %rs110;
	cvt.f32.bf16 	%r1207, %rs109;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1208, %r1205, %r585, %r1207;
	fma.rn.f32 	%r1209, %r1204, %r586, %r1206;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r679, %r1209, %r1208;
	cvt.rn.bf16x2.f32 	%r675, %r1173, %r1172;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1210, %r1506, %r862;
	mul.f32 	%r1211, %r1505, %r862;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs111, %rs112}, %r949;
	cvt.f32.bf16 	%r1212, %rs112;
	cvt.f32.bf16 	%r1213, %rs111;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1214, %r1211, %r585, %r1213;
	fma.rn.f32 	%r1215, %r1210, %r586, %r1212;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r680, %r1215, %r1214;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1216, %r1508, %r861;
	mul.f32 	%r1217, %r1507, %r861;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs113, %rs114}, %r950;
	cvt.f32.bf16 	%r1218, %rs114;
	cvt.f32.bf16 	%r1219, %rs113;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1220, %r1217, %r587, %r1219;
	fma.rn.f32 	%r1221, %r1216, %r588, %r1218;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1222, %r1412, %r855;
	mul.f32 	%r1223, %r1411, %r855;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs115, %rs116}, %r900;
	cvt.f32.bf16 	%r1224, %rs116;
	cvt.f32.bf16 	%r1225, %rs115;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1226, %r1223, %r587, %r1225;
	fma.rn.f32 	%r1227, %r1222, %r588, %r1224;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r692, %r1227, %r1226;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1228, %r1414, %r856;
	mul.f32 	%r1229, %r1413, %r856;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs117, %rs118}, %r901;
	cvt.f32.bf16 	%r1230, %rs118;
	cvt.f32.bf16 	%r1231, %rs117;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1232, %r1229, %r587, %r1231;
	fma.rn.f32 	%r1233, %r1228, %r588, %r1230;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r697, %r1233, %r1232;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1234, %r1444, %r857;
	mul.f32 	%r1235, %r1443, %r857;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs119, %rs120}, %r918;
	cvt.f32.bf16 	%r1236, %rs120;
	cvt.f32.bf16 	%r1237, %rs119;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1238, %r1235, %r587, %r1237;
	fma.rn.f32 	%r1239, %r1234, %r588, %r1236;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r693, %r1239, %r1238;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1240, %r1446, %r858;
	mul.f32 	%r1241, %r1445, %r858;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs121, %rs122}, %r919;
	cvt.f32.bf16 	%r1242, %rs122;
	cvt.f32.bf16 	%r1243, %rs121;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1244, %r1241, %r587, %r1243;
	fma.rn.f32 	%r1245, %r1240, %r588, %r1242;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r698, %r1245, %r1244;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1246, %r1476, %r859;
	mul.f32 	%r1247, %r1475, %r859;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs123, %rs124}, %r934;
	cvt.f32.bf16 	%r1248, %rs124;
	cvt.f32.bf16 	%r1249, %rs123;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1250, %r1247, %r587, %r1249;
	fma.rn.f32 	%r1251, %r1246, %r588, %r1248;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r694, %r1251, %r1250;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1252, %r1478, %r860;
	mul.f32 	%r1253, %r1477, %r860;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs125, %rs126}, %r935;
	cvt.f32.bf16 	%r1254, %rs126;
	cvt.f32.bf16 	%r1255, %rs125;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1256, %r1253, %r587, %r1255;
	fma.rn.f32 	%r1257, %r1252, %r588, %r1254;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r699, %r1257, %r1256;
	cvt.rn.bf16x2.f32 	%r695, %r1221, %r1220;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1258, %r1510, %r862;
	mul.f32 	%r1259, %r1509, %r862;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs127, %rs128}, %r951;
	cvt.f32.bf16 	%r1260, %rs128;
	cvt.f32.bf16 	%r1261, %rs127;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1262, %r1259, %r587, %r1261;
	fma.rn.f32 	%r1263, %r1258, %r588, %r1260;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r700, %r1263, %r1262;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1264, %r1512, %r861;
	mul.f32 	%r1265, %r1511, %r861;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs129, %rs130}, %r956;
	cvt.f32.bf16 	%r1266, %rs130;
	cvt.f32.bf16 	%r1267, %rs129;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1268, %r1265, %r589, %r1267;
	fma.rn.f32 	%r1269, %r1264, %r590, %r1266;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1270, %r1416, %r855;
	mul.f32 	%r1271, %r1415, %r855;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs131, %rs132}, %r908;
	cvt.f32.bf16 	%r1272, %rs132;
	cvt.f32.bf16 	%r1273, %rs131;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1274, %r1271, %r589, %r1273;
	fma.rn.f32 	%r1275, %r1270, %r590, %r1272;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r712, %r1275, %r1274;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1276, %r1418, %r856;
	mul.f32 	%r1277, %r1417, %r856;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs133, %rs134}, %r909;
	cvt.f32.bf16 	%r1278, %rs134;
	cvt.f32.bf16 	%r1279, %rs133;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1280, %r1277, %r589, %r1279;
	fma.rn.f32 	%r1281, %r1276, %r590, %r1278;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r717, %r1281, %r1280;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1282, %r1448, %r857;
	mul.f32 	%r1283, %r1447, %r857;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs135, %rs136}, %r924;
	cvt.f32.bf16 	%r1284, %rs136;
	cvt.f32.bf16 	%r1285, %rs135;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1286, %r1283, %r589, %r1285;
	fma.rn.f32 	%r1287, %r1282, %r590, %r1284;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r713, %r1287, %r1286;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1288, %r1450, %r858;
	mul.f32 	%r1289, %r1449, %r858;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs137, %rs138}, %r925;
	cvt.f32.bf16 	%r1290, %rs138;
	cvt.f32.bf16 	%r1291, %rs137;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1292, %r1289, %r589, %r1291;
	fma.rn.f32 	%r1293, %r1288, %r590, %r1290;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r718, %r1293, %r1292;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1294, %r1480, %r859;
	mul.f32 	%r1295, %r1479, %r859;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs139, %rs140}, %r940;
	cvt.f32.bf16 	%r1296, %rs140;
	cvt.f32.bf16 	%r1297, %rs139;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1298, %r1295, %r589, %r1297;
	fma.rn.f32 	%r1299, %r1294, %r590, %r1296;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r714, %r1299, %r1298;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1300, %r1482, %r860;
	mul.f32 	%r1301, %r1481, %r860;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs141, %rs142}, %r941;
	cvt.f32.bf16 	%r1302, %rs142;
	cvt.f32.bf16 	%r1303, %rs141;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1304, %r1301, %r589, %r1303;
	fma.rn.f32 	%r1305, %r1300, %r590, %r1302;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r719, %r1305, %r1304;
	cvt.rn.bf16x2.f32 	%r715, %r1269, %r1268;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1306, %r1514, %r862;
	mul.f32 	%r1307, %r1513, %r862;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs143, %rs144}, %r957;
	cvt.f32.bf16 	%r1308, %rs144;
	cvt.f32.bf16 	%r1309, %rs143;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1310, %r1307, %r589, %r1309;
	fma.rn.f32 	%r1311, %r1306, %r590, %r1308;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r720, %r1311, %r1310;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1312, %r1516, %r861;
	mul.f32 	%r1313, %r1515, %r861;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs145, %rs146}, %r958;
	cvt.f32.bf16 	%r1314, %rs146;
	cvt.f32.bf16 	%r1315, %rs145;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1316, %r1313, %r591, %r1315;
	fma.rn.f32 	%r1317, %r1312, %r592, %r1314;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1318, %r1420, %r855;
	mul.f32 	%r1319, %r1419, %r855;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs147, %rs148}, %r910;
	cvt.f32.bf16 	%r1320, %rs148;
	cvt.f32.bf16 	%r1321, %rs147;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1322, %r1319, %r591, %r1321;
	fma.rn.f32 	%r1323, %r1318, %r592, %r1320;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r732, %r1323, %r1322;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1324, %r1422, %r856;
	mul.f32 	%r1325, %r1421, %r856;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs149, %rs150}, %r911;
	cvt.f32.bf16 	%r1326, %rs150;
	cvt.f32.bf16 	%r1327, %rs149;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1328, %r1325, %r591, %r1327;
	fma.rn.f32 	%r1329, %r1324, %r592, %r1326;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r737, %r1329, %r1328;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1330, %r1452, %r857;
	mul.f32 	%r1331, %r1451, %r857;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs151, %rs152}, %r926;
	cvt.f32.bf16 	%r1332, %rs152;
	cvt.f32.bf16 	%r1333, %rs151;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1334, %r1331, %r591, %r1333;
	fma.rn.f32 	%r1335, %r1330, %r592, %r1332;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r733, %r1335, %r1334;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1336, %r1454, %r858;
	mul.f32 	%r1337, %r1453, %r858;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs153, %rs154}, %r927;
	cvt.f32.bf16 	%r1338, %rs154;
	cvt.f32.bf16 	%r1339, %rs153;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1340, %r1337, %r591, %r1339;
	fma.rn.f32 	%r1341, %r1336, %r592, %r1338;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r738, %r1341, %r1340;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1342, %r1484, %r859;
	mul.f32 	%r1343, %r1483, %r859;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs155, %rs156}, %r942;
	cvt.f32.bf16 	%r1344, %rs156;
	cvt.f32.bf16 	%r1345, %rs155;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1346, %r1343, %r591, %r1345;
	fma.rn.f32 	%r1347, %r1342, %r592, %r1344;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r734, %r1347, %r1346;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1348, %r1486, %r860;
	mul.f32 	%r1349, %r1485, %r860;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs157, %rs158}, %r943;
	cvt.f32.bf16 	%r1350, %rs158;
	cvt.f32.bf16 	%r1351, %rs157;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1352, %r1349, %r591, %r1351;
	fma.rn.f32 	%r1353, %r1348, %r592, %r1350;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r739, %r1353, %r1352;
	cvt.rn.bf16x2.f32 	%r735, %r1317, %r1316;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1354, %r1518, %r862;
	mul.f32 	%r1355, %r1517, %r862;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs159, %rs160}, %r959;
	cvt.f32.bf16 	%r1356, %rs160;
	cvt.f32.bf16 	%r1357, %rs159;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1358, %r1355, %r591, %r1357;
	fma.rn.f32 	%r1359, %r1354, %r592, %r1356;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r740, %r1359, %r1358;
	bar.sync 	0;
	shl.b32 	%r1360, %r5, 14;
	shl.b32 	%r1361, %r5, 5;
	and.b32 	%r1362, %r1389, 3456;
	bfe.s32 	%r1363, %r2, 2, 1;
	and.b32 	%r1364, %r1363, 8208;
	or.b32 	%r1365, %r1361, %r1362;
	xor.b32 	%r1366, %r1364, %r888;
	or.b32 	%r1367, %r1366, %r1365;
	or.b32 	%r1368, %r1367, %r1360;
	add.s32 	%r661, %r188, %r1368;
	// begin inline asm
	st.shared.v4.b32 [ %r661 + 0 ], { %r662, %r663, %r664, %r665 };
	// end inline asm
	add.s32 	%r666, %r661, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r666 + 0 ], { %r667, %r668, %r669, %r670 };
	// end inline asm
	add.s32 	%r671, %r661, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r671 + 0 ], { %r672, %r673, %r674, %r675 };
	// end inline asm
	add.s32 	%r676, %r661, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r676 + 0 ], { %r677, %r678, %r679, %r680 };
	// end inline asm
	xor.b32 	%r1369, %r1368, 32;
	add.s32 	%r681, %r188, %r1369;
	// begin inline asm
	st.shared.v4.b32 [ %r681 + 0 ], { %r682, %r683, %r684, %r685 };
	// end inline asm
	add.s32 	%r686, %r681, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r686 + 0 ], { %r687, %r688, %r689, %r690 };
	// end inline asm
	add.s32 	%r691, %r681, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r691 + 0 ], { %r692, %r693, %r694, %r695 };
	// end inline asm
	add.s32 	%r696, %r681, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r696 + 0 ], { %r697, %r698, %r699, %r700 };
	// end inline asm
	xor.b32 	%r1370, %r1368, 64;
	add.s32 	%r701, %r188, %r1370;
	// begin inline asm
	st.shared.v4.b32 [ %r701 + 0 ], { %r702, %r703, %r704, %r705 };
	// end inline asm
	add.s32 	%r706, %r701, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r706 + 0 ], { %r707, %r708, %r709, %r710 };
	// end inline asm
	add.s32 	%r711, %r701, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r711 + 0 ], { %r712, %r713, %r714, %r715 };
	// end inline asm
	add.s32 	%r716, %r701, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r716 + 0 ], { %r717, %r718, %r719, %r720 };
	// end inline asm
	xor.b32 	%r1371, %r1368, 96;
	add.s32 	%r721, %r188, %r1371;
	// begin inline asm
	st.shared.v4.b32 [ %r721 + 0 ], { %r722, %r723, %r724, %r725 };
	// end inline asm
	add.s32 	%r726, %r721, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r726 + 0 ], { %r727, %r728, %r729, %r730 };
	// end inline asm
	add.s32 	%r731, %r721, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r731 + 0 ], { %r732, %r733, %r734, %r735 };
	// end inline asm
	add.s32 	%r736, %r721, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r736 + 0 ], { %r737, %r738, %r739, %r740 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r1372, %r2, 2;
	and.b32 	%r1373, %r1372, 896;
	shl.b32 	%r1374, %r847, 9;
	selp.b32 	%r1375, 0, 8208, %p24;
	or.b32 	%r1376, %r881, %r1373;
	xor.b32 	%r1377, %r1376, %r1375;
	or.b32 	%r1378, %r1377, %r1374;
	add.s32 	%r1379, %r188, %r1378;
	ld.shared.v4.b32 	{%r741, %r757, %r773, %r789}, [%r1379];
	ld.shared.v4.b32 	{%r745, %r761, %r777, %r793}, [%r1379+1024];
	ld.shared.v4.b32 	{%r749, %r765, %r781, %r797}, [%r1379+2048];
	ld.shared.v4.b32 	{%r753, %r769, %r785, %r801}, [%r1379+3072];
	xor.b32 	%r1380, %r1378, 32;
	add.s32 	%r1381, %r188, %r1380;
	ld.shared.v4.b32 	{%r742, %r758, %r774, %r790}, [%r1381+16384];
	ld.shared.v4.b32 	{%r746, %r762, %r778, %r794}, [%r1381+17408];
	ld.shared.v4.b32 	{%r750, %r766, %r782, %r798}, [%r1381+18432];
	ld.shared.v4.b32 	{%r754, %r770, %r786, %r802}, [%r1381+19456];
	xor.b32 	%r1382, %r1378, 64;
	add.s32 	%r1383, %r188, %r1382;
	ld.shared.v4.b32 	{%r743, %r759, %r775, %r791}, [%r1383+32768];
	ld.shared.v4.b32 	{%r747, %r763, %r779, %r795}, [%r1383+33792];
	ld.shared.v4.b32 	{%r751, %r767, %r783, %r799}, [%r1383+34816];
	ld.shared.v4.b32 	{%r755, %r771, %r787, %r803}, [%r1383+35840];
	xor.b32 	%r1384, %r1378, 96;
	add.s32 	%r1385, %r188, %r1384;
	ld.shared.v4.b32 	{%r744, %r760, %r776, %r792}, [%r1385+49152];
	ld.shared.v4.b32 	{%r748, %r764, %r780, %r796}, [%r1385+50176];
	ld.shared.v4.b32 	{%r752, %r768, %r784, %r800}, [%r1385+51200];
	ld.shared.v4.b32 	{%r756, %r772, %r788, %r804}, [%r1385+52224];
	.loc	1 175 8                         // sk05_mlp_gateup.py:175:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd138 + 0 ], { %r741, %r742, %r743, %r744 };
	// end inline asm
	// begin inline asm
	@%p9 st.global.v4.b32 [ %rd139 + 0 ], { %r745, %r746, %r747, %r748 };
	// end inline asm
	// begin inline asm
	@%p10 st.global.v4.b32 [ %rd140 + 0 ], { %r749, %r750, %r751, %r752 };
	// end inline asm
	// begin inline asm
	@%p11 st.global.v4.b32 [ %rd141 + 0 ], { %r753, %r754, %r755, %r756 };
	// end inline asm
	// begin inline asm
	@%p12 st.global.v4.b32 [ %rd142 + 0 ], { %r757, %r758, %r759, %r760 };
	// end inline asm
	// begin inline asm
	@%p13 st.global.v4.b32 [ %rd143 + 0 ], { %r761, %r762, %r763, %r764 };
	// end inline asm
	// begin inline asm
	@%p14 st.global.v4.b32 [ %rd144 + 0 ], { %r765, %r766, %r767, %r768 };
	// end inline asm
	// begin inline asm
	@%p15 st.global.v4.b32 [ %rd145 + 0 ], { %r769, %r770, %r771, %r772 };
	// end inline asm
	// begin inline asm
	@%p16 st.global.v4.b32 [ %rd146 + 0 ], { %r773, %r774, %r775, %r776 };
	// end inline asm
	// begin inline asm
	@%p17 st.global.v4.b32 [ %rd147 + 0 ], { %r777, %r778, %r779, %r780 };
	// end inline asm
	// begin inline asm
	@%p18 st.global.v4.b32 [ %rd148 + 0 ], { %r781, %r782, %r783, %r784 };
	// end inline asm
	// begin inline asm
	@%p19 st.global.v4.b32 [ %rd149 + 0 ], { %r785, %r786, %r787, %r788 };
	// end inline asm
	// begin inline asm
	@%p20 st.global.v4.b32 [ %rd150 + 0 ], { %r789, %r790, %r791, %r792 };
	// end inline asm
	// begin inline asm
	@%p21 st.global.v4.b32 [ %rd151 + 0 ], { %r793, %r794, %r795, %r796 };
	// end inline asm
	// begin inline asm
	@%p22 st.global.v4.b32 [ %rd152 + 0 ], { %r797, %r798, %r799, %r800 };
	// end inline asm
	// begin inline asm
	@%p23 st.global.v4.b32 [ %rd153 + 0 ], { %r801, %r802, %r803, %r804 };
	// end inline asm
	.loc	1 173 4                         // sk05_mlp_gateup.py:173:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk05_mlp_gateup.py"
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
.b8 11                                  // DW_FORM_data1
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 165                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x9e DW_TAG_compile_unit
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
.b8 114
.b8 101
.b8 112
.b8 111
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
.b8 2                                   // Abbrev [2] 0x48:0x1a DW_TAG_subprogram
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
.b8 3                                   // Abbrev [3] 0x62:0x46 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 72                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x77:0x18 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 144                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x8f:0x18 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 145                                 // DW_AT_call_line
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
	.reg .b16 	%rs<289>;
	.reg .b32 	%r<1542>;
	.reg .b64 	%rd<309>;
	.loc	1 133 0                         // sk05_mlp_gateup.py:133:0
$L__func_begin0:
	.loc	1 133 0                         // sk05_mlp_gateup.py:133:0

// %bb.0:
	ld.param.b32 	%r26, [_sk05_mlp_gateup_kernel_param_14];
	ld.param.b32 	%r25, [_sk05_mlp_gateup_kernel_param_13];
	ld.param.b32 	%r24, [_sk05_mlp_gateup_kernel_param_12];
	ld.param.b32 	%r23, [_sk05_mlp_gateup_kernel_param_9];
	ld.param.b32 	%r22, [_sk05_mlp_gateup_kernel_param_8];
	ld.param.b32 	%r21, [_sk05_mlp_gateup_kernel_param_7];
	ld.param.b64 	%rd37, [_sk05_mlp_gateup_kernel_param_5];
	ld.param.b64 	%rd36, [_sk05_mlp_gateup_kernel_param_4];
	ld.param.b64 	%rd35, [_sk05_mlp_gateup_kernel_param_3];
	ld.param.b64 	%rd34, [_sk05_mlp_gateup_kernel_param_2];
	ld.param.b64 	%rd33, [_sk05_mlp_gateup_kernel_param_1];
	ld.param.b64 	%rd32, [_sk05_mlp_gateup_kernel_param_0];
$L__tmp0:
	.loc	1 143 24                        // sk05_mlp_gateup.py:143:24
	mov.u32 	%r49, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk05_mlp_gateup.py:144:27 ]
	add.s32 	%r50, %r21, 255;
	.loc	2 43 30                         // standard.py:43:30 @[ sk05_mlp_gateup.py:144:27 ]
	shr.s32 	%r51, %r50, 31;
	shr.u32 	%r52, %r51, 24;
	add.s32 	%r53, %r50, %r52;
	shr.s32 	%r54, %r53, 8;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk05_mlp_gateup.py:145:27 ]
	add.s32 	%r55, %r22, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk05_mlp_gateup.py:145:27 ]
	shr.s32 	%r56, %r55, 31;
	shr.u32 	%r57, %r56, 25;
	add.s32 	%r58, %r55, %r57;
	shr.s32 	%r59, %r58, 7;
$L__tmp3:
	.loc	1 146 29                        // sk05_mlp_gateup.py:146:29
	shl.b32 	%r60, %r59, 3;
	.loc	1 147 22                        // sk05_mlp_gateup.py:147:22
	div.s32 	%r61, %r49, %r60;
	.loc	1 147 38                        // sk05_mlp_gateup.py:147:38
	shl.b32 	%r62, %r61, 3;
	.loc	1 148 30                        // sk05_mlp_gateup.py:148:30
	sub.s32 	%r63, %r54, %r62;
	ld.param.b32 	%r64, [_sk05_mlp_gateup_kernel_param_10];
	.loc	1 148 39                        // sk05_mlp_gateup.py:148:39
	min.s32 	%r65, %r63, 8;
	ld.param.b32 	%r66, [_sk05_mlp_gateup_kernel_param_11];
	.loc	1 149 30                        // sk05_mlp_gateup.py:149:30
	mul.lo.s32 	%r67, %r61, %r60;
	sub.s32 	%r68, %r49, %r67;
	.loc	1 150 36                        // sk05_mlp_gateup.py:150:36
	div.s32 	%r69, %r68, %r65;
	.loc	1 149 46                        // sk05_mlp_gateup.py:149:46
	mul.lo.s32 	%r70, %r69, %r65;
	sub.s32 	%r71, %r68, %r70;
	.loc	1 149 23                        // sk05_mlp_gateup.py:149:23
	add.s32 	%r72, %r71, %r62;
	.loc	1 152 22                        // sk05_mlp_gateup.py:152:22
	shl.b32 	%r1, %r72, 8;
	.loc	1 152 45                        // sk05_mlp_gateup.py:152:45
	mov.u32 	%r2, %tid.x;
	shr.u32 	%r73, %r2, 2;
	bfe.u32 	%r74, %r2, 2, 6;
	or.b32 	%r75, %r74, 64;
	and.b32 	%r3, %r2, 255;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r76, %r1, %r74;
	or.b32 	%r77, %r1, %r75;
	or.b32 	%r78, %r76, 128;
	or.b32 	%r79, %r1, %r73;
	or.b32 	%r80, %r79, 192;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r81, %r76, %r21;
	rem.s32 	%r82, %r77, %r21;
	rem.s32 	%r83, %r78, %r21;
	rem.s32 	%r84, %r80, %r21;
	.loc	1 153 22                        // sk05_mlp_gateup.py:153:22
	shl.b32 	%r4, %r69, 7;
	.loc	1 153 45                        // sk05_mlp_gateup.py:153:45
	and.b32 	%r5, %r2, 3;
	shl.b32 	%r85, %r5, 1;
	and.b32 	%r6, %r2, 32;
	shr.u32 	%r86, %r6, 2;
	or.b32 	%r87, %r86, %r85;
	and.b32 	%r7, %r2, 15;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r88, %r4, %r74;
	or.b32 	%r89, %r4, %r75;
	or.b32 	%r90, %r4, %r87;
	or.b32 	%r92, %r90, 16;
	or.b32 	%r94, %r90, 32;
	or.b32 	%r96, %r90, 48;
	or.b32 	%r98, %r90, 64;
	or.b32 	%r100, %r90, 80;
	or.b32 	%r102, %r90, 96;
	or.b32 	%r104, %r90, 112;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r106, %r88, %r22;
	rem.s32 	%r107, %r89, %r22;
	rem.s32 	%r8, %r90, %r22;
	rem.s32 	%r9, %r92, %r22;
	rem.s32 	%r10, %r94, %r22;
	rem.s32 	%r11, %r96, %r22;
	rem.s32 	%r12, %r98, %r22;
	rem.s32 	%r13, %r100, %r22;
	rem.s32 	%r14, %r102, %r22;
	rem.s32 	%r15, %r104, %r22;
	.loc	1 156 39                        // sk05_mlp_gateup.py:156:39
	mul.lo.s32 	%r116, %r81, %r64;
	mul.lo.s32 	%r117, %r82, %r64;
	mul.lo.s32 	%r118, %r83, %r64;
	mul.lo.s32 	%r119, %r84, %r64;
	.loc	1 156 21                        // sk05_mlp_gateup.py:156:21
	cvt.s64.s32 	%rd1, %r116;
	add.s64 	%rd57, %rd32, %rd1;
	cvt.s64.s32 	%rd2, %r117;
	add.s64 	%rd58, %rd32, %rd2;
	cvt.s64.s32 	%rd3, %r118;
	add.s64 	%rd59, %rd32, %rd3;
	cvt.s64.s32 	%rd4, %r119;
	add.s64 	%rd60, %rd32, %rd4;
	.loc	1 156 58                        // sk05_mlp_gateup.py:156:58
	shl.b32 	%r120, %r5, 4;
	.loc	1 156 51                        // sk05_mlp_gateup.py:156:51
	cvt.u64.u32 	%rd5, %r120;
	add.s64 	%rd38, %rd57, %rd5;
	add.s64 	%rd39, %rd58, %rd5;
	add.s64 	%rd40, %rd59, %rd5;
	add.s64 	%rd41, %rd60, %rd5;
	.loc	1 157 21                        // sk05_mlp_gateup.py:157:21
	add.s64 	%rd61, %rd33, %rd5;
	.loc	1 157 69                        // sk05_mlp_gateup.py:157:69
	mul.lo.s32 	%r121, %r106, %r66;
	mul.lo.s32 	%r122, %r107, %r66;
	.loc	1 157 51                        // sk05_mlp_gateup.py:157:51
	cvt.s64.s32 	%rd6, %r121;
	add.s64 	%rd42, %rd61, %rd6;
	cvt.s64.s32 	%rd7, %r122;
	add.s64 	%rd43, %rd61, %rd7;
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	setp.gt.s32 	%p1, %r23, 63;
	.loc	1 163 30                        // sk05_mlp_gateup.py:163:30
	shl.b32 	%r187, %r3, 4;
	shl.b32 	%r16, %r2, 1;
	and.b32 	%r17, %r16, 48;
	xor.b32 	%r188, %r187, %r17;
	mov.b32 	%r189, global_smem;
	add.s32 	%r28, %r189, %r188;
	selp.b32 	%r29, 16, 0, %p1;
	// begin inline asm
	cp.async.cg.shared.global [ %r28 + 0 ], [ %rd38 + 0 ], 0x10, %r29;
	// end inline asm
	add.s32 	%r30, %r28, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r30 + 0 ], [ %rd39 + 0 ], 0x10, %r29;
	// end inline asm
	add.s32 	%r31, %r28, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r31 + 0 ], [ %rd40 + 0 ], 0x10, %r29;
	// end inline asm
	add.s32 	%r32, %r28, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r32 + 0 ], [ %rd41 + 0 ], 0x10, %r29;
	// end inline asm
	cp.async.commit_group;
	.loc	1 163 47                        // sk05_mlp_gateup.py:163:47
	add.s32 	%r33, %r28, 49152;
	// begin inline asm
	cp.async.cg.shared.global [ %r33 + 0 ], [ %rd42 + 0 ], 0x10, %r29;
	// end inline asm
	add.s32 	%r34, %r28, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r34 + 0 ], [ %rd43 + 0 ], 0x10, %r29;
	// end inline asm
	cp.async.commit_group;
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	setp.gt.s32 	%p2, %r23, 127;
	.loc	1 164 18                        // sk05_mlp_gateup.py:164:18
	add.s64 	%rd44, %rd38, 64;
	add.s64 	%rd45, %rd39, 64;
	add.s64 	%rd46, %rd40, 64;
	add.s64 	%rd47, %rd41, 64;
	.loc	1 165 18                        // sk05_mlp_gateup.py:165:18
	add.s64 	%rd48, %rd42, 64;
	add.s64 	%rd49, %rd43, 64;
	.loc	1 163 30                        // sk05_mlp_gateup.py:163:30
	bar.sync 	0;
	add.s32 	%r35, %r28, 16384;
	selp.b32 	%r36, 16, 0, %p2;
	// begin inline asm
	cp.async.cg.shared.global [ %r35 + 0 ], [ %rd44 + 0 ], 0x10, %r36;
	// end inline asm
	add.s32 	%r37, %r28, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r37 + 0 ], [ %rd45 + 0 ], 0x10, %r36;
	// end inline asm
	add.s32 	%r38, %r28, 24576;
	// begin inline asm
	cp.async.cg.shared.global [ %r38 + 0 ], [ %rd46 + 0 ], 0x10, %r36;
	// end inline asm
	add.s32 	%r39, %r28, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r39 + 0 ], [ %rd47 + 0 ], 0x10, %r36;
	// end inline asm
	cp.async.commit_group;
	.loc	1 163 47                        // sk05_mlp_gateup.py:163:47
	add.s32 	%r40, %r28, 57344;
	// begin inline asm
	cp.async.cg.shared.global [ %r40 + 0 ], [ %rd48 + 0 ], 0x10, %r36;
	// end inline asm
	add.s32 	%r41, %r28, 61440;
	// begin inline asm
	cp.async.cg.shared.global [ %r41 + 0 ], [ %rd49 + 0 ], 0x10, %r36;
	// end inline asm
	cp.async.commit_group;
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	setp.gt.s32 	%p3, %r23, 191;
	.loc	1 164 18                        // sk05_mlp_gateup.py:164:18
	add.s64 	%rd50, %rd38, 128;
	add.s64 	%rd51, %rd39, 128;
	add.s64 	%rd52, %rd40, 128;
	add.s64 	%rd53, %rd41, 128;
	.loc	1 165 18                        // sk05_mlp_gateup.py:165:18
	add.s64 	%rd54, %rd42, 128;
	add.s64 	%rd55, %rd43, 128;
	.loc	1 163 30                        // sk05_mlp_gateup.py:163:30
	bar.sync 	0;
	add.s32 	%r42, %r28, 32768;
	selp.b32 	%r43, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r42 + 0 ], [ %rd50 + 0 ], 0x10, %r43;
	// end inline asm
	add.s32 	%r44, %r28, 36864;
	// begin inline asm
	cp.async.cg.shared.global [ %r44 + 0 ], [ %rd51 + 0 ], 0x10, %r43;
	// end inline asm
	add.s32 	%r45, %r28, 40960;
	// begin inline asm
	cp.async.cg.shared.global [ %r45 + 0 ], [ %rd52 + 0 ], 0x10, %r43;
	// end inline asm
	add.s32 	%r46, %r28, 45056;
	// begin inline asm
	cp.async.cg.shared.global [ %r46 + 0 ], [ %rd53 + 0 ], 0x10, %r43;
	// end inline asm
	cp.async.commit_group;
	.loc	1 163 47                        // sk05_mlp_gateup.py:163:47
	add.s32 	%r47, %r28, 65536;
	// begin inline asm
	cp.async.cg.shared.global [ %r47 + 0 ], [ %rd54 + 0 ], 0x10, %r43;
	// end inline asm
	add.s32 	%r48, %r28, 69632;
	// begin inline asm
	cp.async.cg.shared.global [ %r48 + 0 ], [ %rd55 + 0 ], 0x10, %r43;
	// end inline asm
	cp.async.commit_group;
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	@%p1 bra 	$L__BB0_2;
	bra.uni 	$L__BB0_1;
$L__BB0_2:                              // %.lr.ph
	.loc	1 0 23                          // sk05_mlp_gateup.py:0:23
	ld.param.b32 	%r27, [_sk05_mlp_gateup_kernel_param_15];
	ld.param.b64 	%rd56, [_sk05_mlp_gateup_kernel_param_6];
	or.b32 	%r91, %r90, 1;
	or.b32 	%r93, %r90, 17;
	or.b32 	%r95, %r90, 33;
	or.b32 	%r97, %r90, 49;
	or.b32 	%r99, %r90, 65;
	or.b32 	%r101, %r90, 81;
	or.b32 	%r103, %r90, 97;
	or.b32 	%r105, %r90, 113;
	rem.s32 	%r108, %r91, %r22;
	rem.s32 	%r109, %r93, %r22;
	rem.s32 	%r110, %r95, %r22;
	rem.s32 	%r111, %r97, %r22;
	rem.s32 	%r112, %r99, %r22;
	rem.s32 	%r113, %r101, %r22;
	rem.s32 	%r114, %r103, %r22;
	rem.s32 	%r115, %r105, %r22;
	shr.s32 	%r123, %r8, 31;
	shr.u32 	%r124, %r123, 25;
	add.s32 	%r125, %r8, %r124;
	shr.s32 	%r126, %r125, 7;
	shr.s32 	%r127, %r108, 31;
	shr.u32 	%r128, %r127, 25;
	add.s32 	%r129, %r108, %r128;
	shr.s32 	%r130, %r129, 7;
	shr.s32 	%r131, %r9, 31;
	shr.u32 	%r132, %r131, 25;
	add.s32 	%r133, %r9, %r132;
	shr.s32 	%r134, %r133, 7;
	shr.s32 	%r135, %r109, 31;
	shr.u32 	%r136, %r135, 25;
	add.s32 	%r137, %r109, %r136;
	shr.s32 	%r138, %r137, 7;
	shr.s32 	%r139, %r10, 31;
	shr.u32 	%r140, %r139, 25;
	add.s32 	%r141, %r10, %r140;
	shr.s32 	%r142, %r141, 7;
	shr.s32 	%r143, %r110, 31;
	shr.u32 	%r144, %r143, 25;
	add.s32 	%r145, %r110, %r144;
	shr.s32 	%r146, %r145, 7;
	shr.s32 	%r147, %r11, 31;
	shr.u32 	%r148, %r147, 25;
	add.s32 	%r149, %r11, %r148;
	shr.s32 	%r150, %r149, 7;
	shr.s32 	%r151, %r111, 31;
	shr.u32 	%r152, %r151, 25;
	add.s32 	%r153, %r111, %r152;
	shr.s32 	%r154, %r153, 7;
	shr.s32 	%r155, %r12, 31;
	shr.u32 	%r156, %r155, 25;
	add.s32 	%r157, %r12, %r156;
	shr.s32 	%r158, %r157, 7;
	shr.s32 	%r159, %r112, 31;
	shr.u32 	%r160, %r159, 25;
	add.s32 	%r161, %r112, %r160;
	shr.s32 	%r162, %r161, 7;
	shr.s32 	%r163, %r13, 31;
	shr.u32 	%r164, %r163, 25;
	add.s32 	%r165, %r13, %r164;
	shr.s32 	%r166, %r165, 7;
	shr.s32 	%r167, %r113, 31;
	shr.u32 	%r168, %r167, 25;
	add.s32 	%r169, %r113, %r168;
	shr.s32 	%r170, %r169, 7;
	shr.s32 	%r171, %r14, 31;
	shr.u32 	%r172, %r171, 25;
	add.s32 	%r173, %r14, %r172;
	shr.s32 	%r174, %r173, 7;
	shr.s32 	%r175, %r114, 31;
	shr.u32 	%r176, %r175, 25;
	add.s32 	%r177, %r114, %r176;
	shr.s32 	%r178, %r177, 7;
	shr.s32 	%r179, %r15, 31;
	shr.u32 	%r180, %r179, 25;
	add.s32 	%r181, %r15, %r180;
	shr.s32 	%r182, %r181, 7;
	shr.s32 	%r183, %r115, 31;
	shr.u32 	%r184, %r183, 25;
	add.s32 	%r185, %r115, %r184;
	shr.s32 	%r186, %r185, 7;
	cvt.s64.s32 	%rd62, %r126;
	add.s64 	%rd8, %rd56, %rd62;
	cvt.s64.s32 	%rd63, %r130;
	add.s64 	%rd9, %rd56, %rd63;
	cvt.s64.s32 	%rd64, %r134;
	add.s64 	%rd10, %rd56, %rd64;
	cvt.s64.s32 	%rd65, %r138;
	add.s64 	%rd11, %rd56, %rd65;
	cvt.s64.s32 	%rd66, %r142;
	add.s64 	%rd12, %rd56, %rd66;
	cvt.s64.s32 	%rd67, %r146;
	add.s64 	%rd13, %rd56, %rd67;
	cvt.s64.s32 	%rd68, %r150;
	add.s64 	%rd14, %rd56, %rd68;
	cvt.s64.s32 	%rd69, %r154;
	add.s64 	%rd15, %rd56, %rd69;
	cvt.s64.s32 	%rd70, %r158;
	add.s64 	%rd16, %rd56, %rd70;
	cvt.s64.s32 	%rd71, %r162;
	add.s64 	%rd17, %rd56, %rd71;
	cvt.s64.s32 	%rd72, %r166;
	add.s64 	%rd18, %rd56, %rd72;
	cvt.s64.s32 	%rd73, %r170;
	add.s64 	%rd19, %rd56, %rd73;
	cvt.s64.s32 	%rd74, %r174;
	add.s64 	%rd20, %rd56, %rd74;
	cvt.s64.s32 	%rd75, %r178;
	add.s64 	%rd21, %rd56, %rd75;
	cvt.s64.s32 	%rd76, %r182;
	add.s64 	%rd22, %rd56, %rd76;
	cvt.s64.s32 	%rd77, %r186;
	add.s64 	%rd23, %rd56, %rd77;
	.loc	1 161 28                        // sk05_mlp_gateup.py:161:28
	shr.u32 	%r190, %r23, 6;
	add.s32 	%r191, %r190, -3;
	shl.b32 	%r192, %r7, 6;
	shl.b32 	%r1412, %r2, 4;
	and.b32 	%r193, %r1412, 3072;
	shl.b32 	%r194, %r2, 3;
	and.b32 	%r195, %r194, 48;
	and.b32 	%r1413, %r2, 16;
	or.b32 	%r196, %r192, %r193;
	xor.b32 	%r197, %r195, %r1413;
	or.b32 	%r18, %r196, %r197;
	xor.b32 	%r19, %r18, 32;
	shl.b32 	%r198, %r2, 6;
	and.b32 	%r199, %r198, 448;
	shl.b32 	%r200, %r6, 4;
	or.b32 	%r201, %r199, %r195;
	xor.b32 	%r202, %r201, %r17;
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	add.s32 	%r203, %r189, %r200;
	add.s32 	%r20, %r203, %r202;
	cvt.s64.s32 	%rd24, %r191;
	and.b32 	%r204, %r23, -64;
	cvt.u64.u32 	%rd25, %r204;
	add.s64 	%rd78, %rd5, %rd7;
	add.s64 	%rd79, %rd78, %rd33;
	add.s64 	%rd26, %rd79, 192;
	add.s64 	%rd80, %rd5, %rd6;
	add.s64 	%rd81, %rd80, %rd33;
	add.s64 	%rd27, %rd81, 192;
	add.s64 	%rd82, %rd5, %rd4;
	add.s64 	%rd83, %rd82, %rd32;
	add.s64 	%rd28, %rd83, 192;
	add.s64 	%rd84, %rd5, %rd3;
	add.s64 	%rd85, %rd84, %rd32;
	add.s64 	%rd29, %rd85, 192;
	add.s64 	%rd86, %rd5, %rd2;
	add.s64 	%rd87, %rd86, %rd32;
	add.s64 	%rd30, %rd87, 192;
	add.s64 	%rd88, %rd5, %rd1;
	add.s64 	%rd89, %rd88, %rd32;
	add.s64 	%rd31, %rd89, 192;
	mov.b32 	%r1414, 0f00000000;
	mov.b32 	%r1411, 2;
	mov.b32 	%r1410, -1;
	mov.b64 	%rd307, 0;
	mov.b32 	%r205, 0;
	mov.b32 	%r1409, %r205;
	mov.b64 	%rd308, %rd307;
	mov.b32 	%r1415, %r1414;
	mov.b32 	%r1416, %r1414;
	mov.b32 	%r1417, %r1414;
	mov.b32 	%r1418, %r1414;
	mov.b32 	%r1419, %r1414;
	mov.b32 	%r1420, %r1414;
	mov.b32 	%r1421, %r1414;
	mov.b32 	%r1422, %r1414;
	mov.b32 	%r1423, %r1414;
	mov.b32 	%r1424, %r1414;
	mov.b32 	%r1425, %r1414;
	mov.b32 	%r1426, %r1414;
	mov.b32 	%r1427, %r1414;
	mov.b32 	%r1428, %r1414;
	mov.b32 	%r1429, %r1414;
	mov.b32 	%r1430, %r1414;
	mov.b32 	%r1431, %r1414;
	mov.b32 	%r1432, %r1414;
	mov.b32 	%r1433, %r1414;
	mov.b32 	%r1434, %r1414;
	mov.b32 	%r1435, %r1414;
	mov.b32 	%r1436, %r1414;
	mov.b32 	%r1437, %r1414;
	mov.b32 	%r1438, %r1414;
	mov.b32 	%r1439, %r1414;
	mov.b32 	%r1440, %r1414;
	mov.b32 	%r1441, %r1414;
	mov.b32 	%r1442, %r1414;
	mov.b32 	%r1443, %r1414;
	mov.b32 	%r1444, %r1414;
	mov.b32 	%r1445, %r1414;
	mov.b32 	%r1446, %r1414;
	mov.b32 	%r1447, %r1414;
	mov.b32 	%r1448, %r1414;
	mov.b32 	%r1449, %r1414;
	mov.b32 	%r1450, %r1414;
	mov.b32 	%r1451, %r1414;
	mov.b32 	%r1452, %r1414;
	mov.b32 	%r1453, %r1414;
	mov.b32 	%r1454, %r1414;
	mov.b32 	%r1455, %r1414;
	mov.b32 	%r1456, %r1414;
	mov.b32 	%r1457, %r1414;
	mov.b32 	%r1458, %r1414;
	mov.b32 	%r1459, %r1414;
	mov.b32 	%r1460, %r1414;
	mov.b32 	%r1461, %r1414;
	mov.b32 	%r1462, %r1414;
	mov.b32 	%r1463, %r1414;
	mov.b32 	%r1464, %r1414;
	mov.b32 	%r1465, %r1414;
	mov.b32 	%r1466, %r1414;
	mov.b32 	%r1467, %r1414;
	mov.b32 	%r1468, %r1414;
	mov.b32 	%r1469, %r1414;
	mov.b32 	%r1470, %r1414;
	mov.b32 	%r1471, %r1414;
	mov.b32 	%r1472, %r1414;
	mov.b32 	%r1473, %r1414;
	mov.b32 	%r1474, %r1414;
	mov.b32 	%r1475, %r1414;
	mov.b32 	%r1476, %r1414;
	mov.b32 	%r1477, %r1414;
	mov.b32 	%r1478, %r1414;
	mov.b32 	%r1479, %r1414;
	mov.b32 	%r1480, %r1414;
	mov.b32 	%r1481, %r1414;
	mov.b32 	%r1482, %r1414;
	mov.b32 	%r1483, %r1414;
	mov.b32 	%r1484, %r1414;
	mov.b32 	%r1485, %r1414;
	mov.b32 	%r1486, %r1414;
	mov.b32 	%r1487, %r1414;
	mov.b32 	%r1488, %r1414;
	mov.b32 	%r1489, %r1414;
	mov.b32 	%r1490, %r1414;
	mov.b32 	%r1491, %r1414;
	mov.b32 	%r1492, %r1414;
	mov.b32 	%r1493, %r1414;
	mov.b32 	%r1494, %r1414;
	mov.b32 	%r1495, %r1414;
	mov.b32 	%r1496, %r1414;
	mov.b32 	%r1497, %r1414;
	mov.b32 	%r1498, %r1414;
	mov.b32 	%r1499, %r1414;
	mov.b32 	%r1500, %r1414;
	mov.b32 	%r1501, %r1414;
	mov.b32 	%r1502, %r1414;
	mov.b32 	%r1503, %r1414;
	mov.b32 	%r1504, %r1414;
	mov.b32 	%r1505, %r1414;
	mov.b32 	%r1506, %r1414;
	mov.b32 	%r1507, %r1414;
	mov.b32 	%r1508, %r1414;
	mov.b32 	%r1509, %r1414;
	mov.b32 	%r1510, %r1414;
	mov.b32 	%r1511, %r1414;
	mov.b32 	%r1512, %r1414;
	mov.b32 	%r1513, %r1414;
	mov.b32 	%r1514, %r1414;
	mov.b32 	%r1515, %r1414;
	mov.b32 	%r1516, %r1414;
	mov.b32 	%r1517, %r1414;
	mov.b32 	%r1518, %r1414;
	mov.b32 	%r1519, %r1414;
	mov.b32 	%r1520, %r1414;
	mov.b32 	%r1521, %r1414;
	mov.b32 	%r1522, %r1414;
	mov.b32 	%r1523, %r1414;
	mov.b32 	%r1524, %r1414;
	mov.b32 	%r1525, %r1414;
	mov.b32 	%r1526, %r1414;
	mov.b32 	%r1527, %r1414;
	mov.b32 	%r1528, %r1414;
	mov.b32 	%r1529, %r1414;
	mov.b32 	%r1530, %r1414;
	mov.b32 	%r1531, %r1414;
	mov.b32 	%r1532, %r1414;
	mov.b32 	%r1533, %r1414;
	mov.b32 	%r1534, %r1414;
	mov.b32 	%r1535, %r1414;
	mov.b32 	%r1536, %r1414;
	mov.b32 	%r1537, %r1414;
	mov.b32 	%r1538, %r1414;
	mov.b32 	%r1539, %r1414;
	mov.b32 	%r1540, %r1414;
	mov.b32 	%r1541, %r1414;
$L__BB0_3:                              // %__nv_exp2f.exit
                                        // =>This Inner Loop Header: Depth=1
	setp.lt.s64 	%p4, %rd308, %rd24;
	add.s32 	%r405, %r1410, 1;
	setp.gt.s32 	%p5, %r405, 2;
	selp.b32 	%r1410, 0, %r405, %p5;
	.loc	1 162 39                        // sk05_mlp_gateup.py:162:39
	cvt.s64.s32 	%rd112, %r1409;
	add.s64 	%rd90, %rd8, %rd112;
	add.s64 	%rd91, %rd9, %rd112;
	add.s64 	%rd92, %rd10, %rd112;
	add.s64 	%rd93, %rd11, %rd112;
	add.s64 	%rd94, %rd12, %rd112;
	add.s64 	%rd95, %rd13, %rd112;
	add.s64 	%rd96, %rd14, %rd112;
	add.s64 	%rd97, %rd15, %rd112;
	add.s64 	%rd98, %rd16, %rd112;
	add.s64 	%rd99, %rd17, %rd112;
	add.s64 	%rd100, %rd18, %rd112;
	add.s64 	%rd101, %rd19, %rd112;
	add.s64 	%rd102, %rd20, %rd112;
	add.s64 	%rd103, %rd21, %rd112;
	add.s64 	%rd104, %rd22, %rd112;
	add.s64 	%rd105, %rd23, %rd112;
	.loc	1 162 29                        // sk05_mlp_gateup.py:162:29
	// begin inline asm
	mov.u16 %rs1, 0x0;
	ld.global.b8 { %rs1 }, [ %rd90 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs17, %rs1;
	// begin inline asm
	mov.u16 %rs2, 0x0;
	ld.global.b8 { %rs2 }, [ %rd91 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs18, %rs2;
	// begin inline asm
	mov.u16 %rs3, 0x0;
	ld.global.b8 { %rs3 }, [ %rd92 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs19, %rs3;
	// begin inline asm
	mov.u16 %rs4, 0x0;
	ld.global.b8 { %rs4 }, [ %rd93 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs20, %rs4;
	// begin inline asm
	mov.u16 %rs5, 0x0;
	ld.global.b8 { %rs5 }, [ %rd94 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs21, %rs5;
	// begin inline asm
	mov.u16 %rs6, 0x0;
	ld.global.b8 { %rs6 }, [ %rd95 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs22, %rs6;
	// begin inline asm
	mov.u16 %rs7, 0x0;
	ld.global.b8 { %rs7 }, [ %rd96 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs23, %rs7;
	// begin inline asm
	mov.u16 %rs8, 0x0;
	ld.global.b8 { %rs8 }, [ %rd97 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs24, %rs8;
	// begin inline asm
	mov.u16 %rs9, 0x0;
	ld.global.b8 { %rs9 }, [ %rd98 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs25, %rs9;
	// begin inline asm
	mov.u16 %rs10, 0x0;
	ld.global.b8 { %rs10 }, [ %rd99 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs26, %rs10;
	// begin inline asm
	mov.u16 %rs11, 0x0;
	ld.global.b8 { %rs11 }, [ %rd100 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs27, %rs11;
	// begin inline asm
	mov.u16 %rs12, 0x0;
	ld.global.b8 { %rs12 }, [ %rd101 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs28, %rs12;
	// begin inline asm
	mov.u16 %rs13, 0x0;
	ld.global.b8 { %rs13 }, [ %rd102 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs29, %rs13;
	// begin inline asm
	mov.u16 %rs14, 0x0;
	ld.global.b8 { %rs14 }, [ %rd103 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs30, %rs14;
	// begin inline asm
	mov.u16 %rs15, 0x0;
	ld.global.b8 { %rs15 }, [ %rd104 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs31, %rs15;
	// begin inline asm
	mov.u16 %rs16, 0x0;
	ld.global.b8 { %rs16 }, [ %rd105 + 0 ];
	// end inline asm
	cvt.s16.s8 	%rs32, %rs16;
	.loc	1 162 63                        // sk05_mlp_gateup.py:162:63
	cvt.rn.f32.s16 	%r406, %rs17;
	cvt.rn.f32.s16 	%r407, %rs18;
	cvt.rn.f32.s16 	%r408, %rs19;
	cvt.rn.f32.s16 	%r409, %rs20;
	cvt.rn.f32.s16 	%r410, %rs21;
	cvt.rn.f32.s16 	%r411, %rs22;
	cvt.rn.f32.s16 	%r412, %rs23;
	cvt.rn.f32.s16 	%r413, %rs24;
	cvt.rn.f32.s16 	%r414, %rs25;
	cvt.rn.f32.s16 	%r415, %rs26;
	cvt.rn.f32.s16 	%r416, %rs27;
	cvt.rn.f32.s16 	%r417, %rs28;
	cvt.rn.f32.s16 	%r418, %rs29;
	cvt.rn.f32.s16 	%r419, %rs30;
	cvt.rn.f32.s16 	%r420, %rs31;
	cvt.rn.f32.s16 	%r421, %rs32;
	.loc	1 162 21                        // sk05_mlp_gateup.py:162:21
	ex2.approx.ftz.f32 	%r422, %r406;
	ex2.approx.ftz.f32 	%r423, %r407;
	ex2.approx.ftz.f32 	%r424, %r408;
	ex2.approx.ftz.f32 	%r425, %r409;
	ex2.approx.ftz.f32 	%r426, %r410;
	ex2.approx.ftz.f32 	%r427, %r411;
	ex2.approx.ftz.f32 	%r428, %r412;
	ex2.approx.ftz.f32 	%r429, %r413;
	ex2.approx.ftz.f32 	%r430, %r414;
	ex2.approx.ftz.f32 	%r431, %r415;
	ex2.approx.ftz.f32 	%r432, %r416;
	ex2.approx.ftz.f32 	%r433, %r417;
	ex2.approx.ftz.f32 	%r434, %r418;
	ex2.approx.ftz.f32 	%r435, %r419;
	ex2.approx.ftz.f32 	%r436, %r420;
	ex2.approx.ftz.f32 	%r437, %r421;
	.loc	1 163 30                        // sk05_mlp_gateup.py:163:30
	cp.async.wait_group 	4;
	bar.sync 	0;
	shl.b32 	%r438, %r1410, 14;
	add.s32 	%r439, %r189, %r438;
	add.s32 	%r440, %r439, %r18;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r206, %r207, %r208, %r209}, [%r440];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r226, %r227, %r228, %r229}, [%r440+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r230, %r231, %r232, %r233}, [%r440+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r234, %r235, %r236, %r237}, [%r440+12288];
	add.s32 	%r441, %r439, %r19;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r242, %r243, %r244, %r245}, [%r441];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r294, %r295, %r296, %r297}, [%r441+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r330, %r331, %r332, %r333}, [%r441+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r366, %r367, %r368, %r369}, [%r441+12288];
	.loc	1 163 47                        // sk05_mlp_gateup.py:163:47
	shl.b32 	%r442, %r1410, 13;
	add.s32 	%r443, %r20, %r442;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r210, %r211, %r246, %r247}, [%r443+49152];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r212, %r213, %r252, %r253}, [%r443+50176];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r214, %r215, %r258, %r259}, [%r443+51200];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r216, %r217, %r264, %r265}, [%r443+52224];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r218, %r219, %r270, %r271}, [%r443+53248];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r220, %r221, %r276, %r277}, [%r443+54272];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r222, %r223, %r282, %r283}, [%r443+55296];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r224, %r225, %r288, %r289}, [%r443+56320];
	.loc	1 163 39                        // sk05_mlp_gateup.py:163:39
	mov.b32 	%r238, %r205;
	mov.b32 	%r239, %r205;
	mov.b32 	%r240, %r205;
	mov.b32 	%r241, %r205;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r238, %r239, %r240, %r241 }, { %r206, %r207, %r208, %r209 }, { %r210, %r211 }, { %r238, %r239, %r240, %r241 };
	// end inline asm
	mov.b32 	%r248, %r205;
	mov.b32 	%r249, %r205;
	mov.b32 	%r250, %r205;
	mov.b32 	%r251, %r205;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r248, %r249, %r250, %r251 }, { %r206, %r207, %r208, %r209 }, { %r212, %r213 }, { %r248, %r249, %r250, %r251 };
	// end inline asm
	mov.b32 	%r254, %r205;
	mov.b32 	%r255, %r205;
	mov.b32 	%r256, %r205;
	mov.b32 	%r257, %r205;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r254, %r255, %r256, %r257 }, { %r206, %r207, %r208, %r209 }, { %r214, %r215 }, { %r254, %r255, %r256, %r257 };
	// end inline asm
	mov.b32 	%r260, %r205;
	mov.b32 	%r261, %r205;
	mov.b32 	%r262, %r205;
	mov.b32 	%r263, %r205;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r260, %r261, %r262, %r263 }, { %r206, %r207, %r208, %r209 }, { %r216, %r217 }, { %r260, %r261, %r262, %r263 };
	// end inline asm
	mov.b32 	%r266, %r205;
	mov.b32 	%r267, %r205;
	mov.b32 	%r268, %r205;
	mov.b32 	%r269, %r205;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r266, %r267, %r268, %r269 }, { %r206, %r207, %r208, %r209 }, { %r218, %r219 }, { %r266, %r267, %r268, %r269 };
	// end inline asm
	mov.b32 	%r272, %r205;
	mov.b32 	%r273, %r205;
	mov.b32 	%r274, %r205;
	mov.b32 	%r275, %r205;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r272, %r273, %r274, %r275 }, { %r206, %r207, %r208, %r209 }, { %r220, %r221 }, { %r272, %r273, %r274, %r275 };
	// end inline asm
	mov.b32 	%r278, %r205;
	mov.b32 	%r279, %r205;
	mov.b32 	%r280, %r205;
	mov.b32 	%r281, %r205;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r278, %r279, %r280, %r281 }, { %r206, %r207, %r208, %r209 }, { %r222, %r223 }, { %r278, %r279, %r280, %r281 };
	// end inline asm
	mov.b32 	%r284, %r205;
	mov.b32 	%r285, %r205;
	mov.b32 	%r286, %r205;
	mov.b32 	%r287, %r205;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r284, %r285, %r286, %r287 }, { %r206, %r207, %r208, %r209 }, { %r224, %r225 }, { %r284, %r285, %r286, %r287 };
	// end inline asm
	mov.b32 	%r290, %r205;
	mov.b32 	%r291, %r205;
	mov.b32 	%r292, %r205;
	mov.b32 	%r293, %r205;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r290, %r291, %r292, %r293 }, { %r226, %r227, %r228, %r229 }, { %r210, %r211 }, { %r290, %r291, %r292, %r293 };
	// end inline asm
	mov.b32 	%r298, %r205;
	mov.b32 	%r299, %r205;
	mov.b32 	%r300, %r205;
	mov.b32 	%r301, %r205;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r298, %r299, %r300, %r301 }, { %r226, %r227, %r228, %r229 }, { %r212, %r213 }, { %r298, %r299, %r300, %r301 };
	// end inline asm
	mov.b32 	%r302, %r205;
	mov.b32 	%r303, %r205;
	mov.b32 	%r304, %r205;
	mov.b32 	%r305, %r205;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r302, %r303, %r304, %r305 }, { %r226, %r227, %r228, %r229 }, { %r214, %r215 }, { %r302, %r303, %r304, %r305 };
	// end inline asm
	mov.b32 	%r306, %r205;
	mov.b32 	%r307, %r205;
	mov.b32 	%r308, %r205;
	mov.b32 	%r309, %r205;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r306, %r307, %r308, %r309 }, { %r226, %r227, %r228, %r229 }, { %r216, %r217 }, { %r306, %r307, %r308, %r309 };
	// end inline asm
	mov.b32 	%r310, %r205;
	mov.b32 	%r311, %r205;
	mov.b32 	%r312, %r205;
	mov.b32 	%r313, %r205;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r310, %r311, %r312, %r313 }, { %r226, %r227, %r228, %r229 }, { %r218, %r219 }, { %r310, %r311, %r312, %r313 };
	// end inline asm
	mov.b32 	%r314, %r205;
	mov.b32 	%r315, %r205;
	mov.b32 	%r316, %r205;
	mov.b32 	%r317, %r205;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r314, %r315, %r316, %r317 }, { %r226, %r227, %r228, %r229 }, { %r220, %r221 }, { %r314, %r315, %r316, %r317 };
	// end inline asm
	mov.b32 	%r318, %r205;
	mov.b32 	%r319, %r205;
	mov.b32 	%r320, %r205;
	mov.b32 	%r321, %r205;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r318, %r319, %r320, %r321 }, { %r226, %r227, %r228, %r229 }, { %r222, %r223 }, { %r318, %r319, %r320, %r321 };
	// end inline asm
	mov.b32 	%r322, %r205;
	mov.b32 	%r323, %r205;
	mov.b32 	%r324, %r205;
	mov.b32 	%r325, %r205;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r322, %r323, %r324, %r325 }, { %r226, %r227, %r228, %r229 }, { %r224, %r225 }, { %r322, %r323, %r324, %r325 };
	// end inline asm
	mov.b32 	%r326, %r205;
	mov.b32 	%r327, %r205;
	mov.b32 	%r328, %r205;
	mov.b32 	%r329, %r205;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r326, %r327, %r328, %r329 }, { %r230, %r231, %r232, %r233 }, { %r210, %r211 }, { %r326, %r327, %r328, %r329 };
	// end inline asm
	mov.b32 	%r334, %r205;
	mov.b32 	%r335, %r205;
	mov.b32 	%r336, %r205;
	mov.b32 	%r337, %r205;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r334, %r335, %r336, %r337 }, { %r230, %r231, %r232, %r233 }, { %r212, %r213 }, { %r334, %r335, %r336, %r337 };
	// end inline asm
	mov.b32 	%r338, %r205;
	mov.b32 	%r339, %r205;
	mov.b32 	%r340, %r205;
	mov.b32 	%r341, %r205;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r338, %r339, %r340, %r341 }, { %r230, %r231, %r232, %r233 }, { %r214, %r215 }, { %r338, %r339, %r340, %r341 };
	// end inline asm
	mov.b32 	%r342, %r205;
	mov.b32 	%r343, %r205;
	mov.b32 	%r344, %r205;
	mov.b32 	%r345, %r205;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r342, %r343, %r344, %r345 }, { %r230, %r231, %r232, %r233 }, { %r216, %r217 }, { %r342, %r343, %r344, %r345 };
	// end inline asm
	mov.b32 	%r346, %r205;
	mov.b32 	%r347, %r205;
	mov.b32 	%r348, %r205;
	mov.b32 	%r349, %r205;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r346, %r347, %r348, %r349 }, { %r230, %r231, %r232, %r233 }, { %r218, %r219 }, { %r346, %r347, %r348, %r349 };
	// end inline asm
	mov.b32 	%r350, %r205;
	mov.b32 	%r351, %r205;
	mov.b32 	%r352, %r205;
	mov.b32 	%r353, %r205;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r350, %r351, %r352, %r353 }, { %r230, %r231, %r232, %r233 }, { %r220, %r221 }, { %r350, %r351, %r352, %r353 };
	// end inline asm
	mov.b32 	%r354, %r205;
	mov.b32 	%r355, %r205;
	mov.b32 	%r356, %r205;
	mov.b32 	%r357, %r205;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r354, %r355, %r356, %r357 }, { %r230, %r231, %r232, %r233 }, { %r222, %r223 }, { %r354, %r355, %r356, %r357 };
	// end inline asm
	mov.b32 	%r358, %r205;
	mov.b32 	%r359, %r205;
	mov.b32 	%r360, %r205;
	mov.b32 	%r361, %r205;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r358, %r359, %r360, %r361 }, { %r230, %r231, %r232, %r233 }, { %r224, %r225 }, { %r358, %r359, %r360, %r361 };
	// end inline asm
	mov.b32 	%r362, %r205;
	mov.b32 	%r363, %r205;
	mov.b32 	%r364, %r205;
	mov.b32 	%r365, %r205;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r362, %r363, %r364, %r365 }, { %r234, %r235, %r236, %r237 }, { %r210, %r211 }, { %r362, %r363, %r364, %r365 };
	// end inline asm
	mov.b32 	%r370, %r205;
	mov.b32 	%r371, %r205;
	mov.b32 	%r372, %r205;
	mov.b32 	%r373, %r205;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r370, %r371, %r372, %r373 }, { %r234, %r235, %r236, %r237 }, { %r212, %r213 }, { %r370, %r371, %r372, %r373 };
	// end inline asm
	mov.b32 	%r374, %r205;
	mov.b32 	%r375, %r205;
	mov.b32 	%r376, %r205;
	mov.b32 	%r377, %r205;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r374, %r375, %r376, %r377 }, { %r234, %r235, %r236, %r237 }, { %r214, %r215 }, { %r374, %r375, %r376, %r377 };
	// end inline asm
	mov.b32 	%r378, %r205;
	mov.b32 	%r379, %r205;
	mov.b32 	%r380, %r205;
	mov.b32 	%r381, %r205;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r378, %r379, %r380, %r381 }, { %r234, %r235, %r236, %r237 }, { %r216, %r217 }, { %r378, %r379, %r380, %r381 };
	// end inline asm
	mov.b32 	%r382, %r205;
	mov.b32 	%r383, %r205;
	mov.b32 	%r384, %r205;
	mov.b32 	%r385, %r205;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r382, %r383, %r384, %r385 }, { %r234, %r235, %r236, %r237 }, { %r218, %r219 }, { %r382, %r383, %r384, %r385 };
	// end inline asm
	mov.b32 	%r386, %r205;
	mov.b32 	%r387, %r205;
	mov.b32 	%r388, %r205;
	mov.b32 	%r389, %r205;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r386, %r387, %r388, %r389 }, { %r234, %r235, %r236, %r237 }, { %r220, %r221 }, { %r386, %r387, %r388, %r389 };
	// end inline asm
	mov.b32 	%r390, %r205;
	mov.b32 	%r391, %r205;
	mov.b32 	%r392, %r205;
	mov.b32 	%r393, %r205;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r390, %r391, %r392, %r393 }, { %r234, %r235, %r236, %r237 }, { %r222, %r223 }, { %r390, %r391, %r392, %r393 };
	// end inline asm
	mov.b32 	%r397, %r205;
	mov.b32 	%r394, %r205;
	mov.b32 	%r395, %r205;
	mov.b32 	%r396, %r205;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r394, %r395, %r396, %r397 }, { %r234, %r235, %r236, %r237 }, { %r224, %r225 }, { %r394, %r395, %r396, %r397 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r238, %r239, %r240, %r241 }, { %r242, %r243, %r244, %r245 }, { %r246, %r247 }, { %r238, %r239, %r240, %r241 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r248, %r249, %r250, %r251 }, { %r242, %r243, %r244, %r245 }, { %r252, %r253 }, { %r248, %r249, %r250, %r251 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r254, %r255, %r256, %r257 }, { %r242, %r243, %r244, %r245 }, { %r258, %r259 }, { %r254, %r255, %r256, %r257 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r260, %r261, %r262, %r263 }, { %r242, %r243, %r244, %r245 }, { %r264, %r265 }, { %r260, %r261, %r262, %r263 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r266, %r267, %r268, %r269 }, { %r242, %r243, %r244, %r245 }, { %r270, %r271 }, { %r266, %r267, %r268, %r269 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r272, %r273, %r274, %r275 }, { %r242, %r243, %r244, %r245 }, { %r276, %r277 }, { %r272, %r273, %r274, %r275 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r278, %r279, %r280, %r281 }, { %r242, %r243, %r244, %r245 }, { %r282, %r283 }, { %r278, %r279, %r280, %r281 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r284, %r285, %r286, %r287 }, { %r242, %r243, %r244, %r245 }, { %r288, %r289 }, { %r284, %r285, %r286, %r287 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r290, %r291, %r292, %r293 }, { %r294, %r295, %r296, %r297 }, { %r246, %r247 }, { %r290, %r291, %r292, %r293 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r298, %r299, %r300, %r301 }, { %r294, %r295, %r296, %r297 }, { %r252, %r253 }, { %r298, %r299, %r300, %r301 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r302, %r303, %r304, %r305 }, { %r294, %r295, %r296, %r297 }, { %r258, %r259 }, { %r302, %r303, %r304, %r305 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r306, %r307, %r308, %r309 }, { %r294, %r295, %r296, %r297 }, { %r264, %r265 }, { %r306, %r307, %r308, %r309 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r310, %r311, %r312, %r313 }, { %r294, %r295, %r296, %r297 }, { %r270, %r271 }, { %r310, %r311, %r312, %r313 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r314, %r315, %r316, %r317 }, { %r294, %r295, %r296, %r297 }, { %r276, %r277 }, { %r314, %r315, %r316, %r317 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r318, %r319, %r320, %r321 }, { %r294, %r295, %r296, %r297 }, { %r282, %r283 }, { %r318, %r319, %r320, %r321 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r322, %r323, %r324, %r325 }, { %r294, %r295, %r296, %r297 }, { %r288, %r289 }, { %r322, %r323, %r324, %r325 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r326, %r327, %r328, %r329 }, { %r330, %r331, %r332, %r333 }, { %r246, %r247 }, { %r326, %r327, %r328, %r329 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r334, %r335, %r336, %r337 }, { %r330, %r331, %r332, %r333 }, { %r252, %r253 }, { %r334, %r335, %r336, %r337 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r338, %r339, %r340, %r341 }, { %r330, %r331, %r332, %r333 }, { %r258, %r259 }, { %r338, %r339, %r340, %r341 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r342, %r343, %r344, %r345 }, { %r330, %r331, %r332, %r333 }, { %r264, %r265 }, { %r342, %r343, %r344, %r345 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r346, %r347, %r348, %r349 }, { %r330, %r331, %r332, %r333 }, { %r270, %r271 }, { %r346, %r347, %r348, %r349 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r350, %r351, %r352, %r353 }, { %r330, %r331, %r332, %r333 }, { %r276, %r277 }, { %r350, %r351, %r352, %r353 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r354, %r355, %r356, %r357 }, { %r330, %r331, %r332, %r333 }, { %r282, %r283 }, { %r354, %r355, %r356, %r357 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r358, %r359, %r360, %r361 }, { %r330, %r331, %r332, %r333 }, { %r288, %r289 }, { %r358, %r359, %r360, %r361 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r362, %r363, %r364, %r365 }, { %r366, %r367, %r368, %r369 }, { %r246, %r247 }, { %r362, %r363, %r364, %r365 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r370, %r371, %r372, %r373 }, { %r366, %r367, %r368, %r369 }, { %r252, %r253 }, { %r370, %r371, %r372, %r373 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r374, %r375, %r376, %r377 }, { %r366, %r367, %r368, %r369 }, { %r258, %r259 }, { %r374, %r375, %r376, %r377 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r378, %r379, %r380, %r381 }, { %r366, %r367, %r368, %r369 }, { %r264, %r265 }, { %r378, %r379, %r380, %r381 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r382, %r383, %r384, %r385 }, { %r366, %r367, %r368, %r369 }, { %r270, %r271 }, { %r382, %r383, %r384, %r385 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r386, %r387, %r388, %r389 }, { %r366, %r367, %r368, %r369 }, { %r276, %r277 }, { %r386, %r387, %r388, %r389 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r390, %r391, %r392, %r393 }, { %r366, %r367, %r368, %r369 }, { %r282, %r283 }, { %r390, %r391, %r392, %r393 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r394, %r395, %r396, %r397 }, { %r366, %r367, %r368, %r369 }, { %r288, %r289 }, { %r394, %r395, %r396, %r397 };
	// end inline asm
	.loc	1 163 79                        // sk05_mlp_gateup.py:163:79
	cvt.rn.f32.s32 	%r444, %r397;
	cvt.rn.f32.s32 	%r445, %r396;
	cvt.rn.f32.s32 	%r446, %r395;
	cvt.rn.f32.s32 	%r447, %r394;
	cvt.rn.f32.s32 	%r448, %r393;
	cvt.rn.f32.s32 	%r449, %r392;
	cvt.rn.f32.s32 	%r450, %r391;
	cvt.rn.f32.s32 	%r451, %r390;
	cvt.rn.f32.s32 	%r452, %r389;
	cvt.rn.f32.s32 	%r453, %r388;
	cvt.rn.f32.s32 	%r454, %r387;
	cvt.rn.f32.s32 	%r455, %r386;
	cvt.rn.f32.s32 	%r456, %r385;
	cvt.rn.f32.s32 	%r457, %r384;
	cvt.rn.f32.s32 	%r458, %r383;
	cvt.rn.f32.s32 	%r459, %r382;
	cvt.rn.f32.s32 	%r460, %r381;
	cvt.rn.f32.s32 	%r461, %r380;
	cvt.rn.f32.s32 	%r462, %r379;
	cvt.rn.f32.s32 	%r463, %r378;
	cvt.rn.f32.s32 	%r464, %r377;
	cvt.rn.f32.s32 	%r465, %r376;
	cvt.rn.f32.s32 	%r466, %r375;
	cvt.rn.f32.s32 	%r467, %r374;
	cvt.rn.f32.s32 	%r468, %r373;
	cvt.rn.f32.s32 	%r469, %r372;
	cvt.rn.f32.s32 	%r470, %r371;
	cvt.rn.f32.s32 	%r471, %r370;
	cvt.rn.f32.s32 	%r472, %r365;
	cvt.rn.f32.s32 	%r473, %r364;
	cvt.rn.f32.s32 	%r474, %r363;
	cvt.rn.f32.s32 	%r475, %r362;
	cvt.rn.f32.s32 	%r476, %r361;
	cvt.rn.f32.s32 	%r477, %r360;
	cvt.rn.f32.s32 	%r478, %r358;
	cvt.rn.f32.s32 	%r479, %r359;
	cvt.rn.f32.s32 	%r480, %r354;
	cvt.rn.f32.s32 	%r481, %r355;
	cvt.rn.f32.s32 	%r482, %r356;
	cvt.rn.f32.s32 	%r483, %r357;
	cvt.rn.f32.s32 	%r484, %r350;
	cvt.rn.f32.s32 	%r485, %r351;
	cvt.rn.f32.s32 	%r486, %r352;
	cvt.rn.f32.s32 	%r487, %r353;
	cvt.rn.f32.s32 	%r488, %r346;
	cvt.rn.f32.s32 	%r489, %r347;
	cvt.rn.f32.s32 	%r490, %r348;
	cvt.rn.f32.s32 	%r491, %r349;
	cvt.rn.f32.s32 	%r492, %r342;
	cvt.rn.f32.s32 	%r493, %r343;
	cvt.rn.f32.s32 	%r494, %r344;
	cvt.rn.f32.s32 	%r495, %r345;
	cvt.rn.f32.s32 	%r496, %r338;
	cvt.rn.f32.s32 	%r497, %r339;
	cvt.rn.f32.s32 	%r498, %r340;
	cvt.rn.f32.s32 	%r499, %r341;
	cvt.rn.f32.s32 	%r500, %r334;
	cvt.rn.f32.s32 	%r501, %r335;
	cvt.rn.f32.s32 	%r502, %r336;
	cvt.rn.f32.s32 	%r503, %r337;
	cvt.rn.f32.s32 	%r504, %r326;
	cvt.rn.f32.s32 	%r505, %r327;
	cvt.rn.f32.s32 	%r506, %r328;
	cvt.rn.f32.s32 	%r507, %r329;
	cvt.rn.f32.s32 	%r508, %r322;
	cvt.rn.f32.s32 	%r509, %r323;
	cvt.rn.f32.s32 	%r510, %r324;
	cvt.rn.f32.s32 	%r511, %r325;
	cvt.rn.f32.s32 	%r512, %r318;
	cvt.rn.f32.s32 	%r513, %r319;
	cvt.rn.f32.s32 	%r514, %r320;
	cvt.rn.f32.s32 	%r515, %r321;
	cvt.rn.f32.s32 	%r516, %r314;
	cvt.rn.f32.s32 	%r517, %r315;
	cvt.rn.f32.s32 	%r518, %r316;
	cvt.rn.f32.s32 	%r519, %r317;
	cvt.rn.f32.s32 	%r520, %r310;
	cvt.rn.f32.s32 	%r521, %r311;
	cvt.rn.f32.s32 	%r522, %r312;
	cvt.rn.f32.s32 	%r523, %r313;
	cvt.rn.f32.s32 	%r524, %r306;
	cvt.rn.f32.s32 	%r525, %r307;
	cvt.rn.f32.s32 	%r526, %r308;
	cvt.rn.f32.s32 	%r527, %r309;
	cvt.rn.f32.s32 	%r528, %r302;
	cvt.rn.f32.s32 	%r529, %r303;
	cvt.rn.f32.s32 	%r530, %r304;
	cvt.rn.f32.s32 	%r531, %r305;
	cvt.rn.f32.s32 	%r532, %r298;
	cvt.rn.f32.s32 	%r533, %r299;
	cvt.rn.f32.s32 	%r534, %r300;
	cvt.rn.f32.s32 	%r535, %r301;
	cvt.rn.f32.s32 	%r536, %r290;
	cvt.rn.f32.s32 	%r537, %r291;
	cvt.rn.f32.s32 	%r538, %r292;
	cvt.rn.f32.s32 	%r539, %r293;
	cvt.rn.f32.s32 	%r540, %r284;
	cvt.rn.f32.s32 	%r541, %r285;
	cvt.rn.f32.s32 	%r542, %r286;
	cvt.rn.f32.s32 	%r543, %r287;
	cvt.rn.f32.s32 	%r544, %r278;
	cvt.rn.f32.s32 	%r545, %r279;
	cvt.rn.f32.s32 	%r546, %r280;
	cvt.rn.f32.s32 	%r547, %r281;
	cvt.rn.f32.s32 	%r548, %r272;
	cvt.rn.f32.s32 	%r549, %r273;
	cvt.rn.f32.s32 	%r550, %r274;
	cvt.rn.f32.s32 	%r551, %r275;
	cvt.rn.f32.s32 	%r552, %r266;
	cvt.rn.f32.s32 	%r553, %r267;
	cvt.rn.f32.s32 	%r554, %r268;
	cvt.rn.f32.s32 	%r555, %r269;
	cvt.rn.f32.s32 	%r556, %r260;
	cvt.rn.f32.s32 	%r557, %r261;
	cvt.rn.f32.s32 	%r558, %r262;
	cvt.rn.f32.s32 	%r559, %r263;
	cvt.rn.f32.s32 	%r560, %r254;
	cvt.rn.f32.s32 	%r561, %r255;
	cvt.rn.f32.s32 	%r562, %r256;
	cvt.rn.f32.s32 	%r563, %r257;
	cvt.rn.f32.s32 	%r564, %r248;
	cvt.rn.f32.s32 	%r565, %r249;
	cvt.rn.f32.s32 	%r566, %r250;
	cvt.rn.f32.s32 	%r567, %r251;
	cvt.rn.f32.s32 	%r568, %r238;
	cvt.rn.f32.s32 	%r569, %r239;
	cvt.rn.f32.s32 	%r570, %r240;
	cvt.rn.f32.s32 	%r571, %r241;
	.loc	1 163 15                        // sk05_mlp_gateup.py:163:15
	fma.rn.f32 	%r1417, %r423, %r571, %r1417;
	fma.rn.f32 	%r1416, %r422, %r570, %r1416;
	fma.rn.f32 	%r1415, %r423, %r569, %r1415;
	fma.rn.f32 	%r1414, %r422, %r568, %r1414;
	fma.rn.f32 	%r1421, %r425, %r567, %r1421;
	fma.rn.f32 	%r1420, %r424, %r566, %r1420;
	fma.rn.f32 	%r1419, %r425, %r565, %r1419;
	fma.rn.f32 	%r1418, %r424, %r564, %r1418;
	fma.rn.f32 	%r1425, %r427, %r563, %r1425;
	fma.rn.f32 	%r1424, %r426, %r562, %r1424;
	fma.rn.f32 	%r1423, %r427, %r561, %r1423;
	fma.rn.f32 	%r1422, %r426, %r560, %r1422;
	fma.rn.f32 	%r1429, %r429, %r559, %r1429;
	fma.rn.f32 	%r1428, %r428, %r558, %r1428;
	fma.rn.f32 	%r1427, %r429, %r557, %r1427;
	fma.rn.f32 	%r1426, %r428, %r556, %r1426;
	fma.rn.f32 	%r1433, %r431, %r555, %r1433;
	fma.rn.f32 	%r1432, %r430, %r554, %r1432;
	fma.rn.f32 	%r1431, %r431, %r553, %r1431;
	fma.rn.f32 	%r1430, %r430, %r552, %r1430;
	fma.rn.f32 	%r1437, %r433, %r551, %r1437;
	fma.rn.f32 	%r1436, %r432, %r550, %r1436;
	fma.rn.f32 	%r1435, %r433, %r549, %r1435;
	fma.rn.f32 	%r1434, %r432, %r548, %r1434;
	fma.rn.f32 	%r1441, %r435, %r547, %r1441;
	fma.rn.f32 	%r1440, %r434, %r546, %r1440;
	fma.rn.f32 	%r1439, %r435, %r545, %r1439;
	fma.rn.f32 	%r1438, %r434, %r544, %r1438;
	fma.rn.f32 	%r1445, %r437, %r543, %r1445;
	fma.rn.f32 	%r1444, %r436, %r542, %r1444;
	fma.rn.f32 	%r1443, %r437, %r541, %r1443;
	fma.rn.f32 	%r1442, %r436, %r540, %r1442;
	fma.rn.f32 	%r1449, %r423, %r539, %r1449;
	fma.rn.f32 	%r1448, %r422, %r538, %r1448;
	fma.rn.f32 	%r1447, %r423, %r537, %r1447;
	fma.rn.f32 	%r1446, %r422, %r536, %r1446;
	fma.rn.f32 	%r1453, %r425, %r535, %r1453;
	fma.rn.f32 	%r1452, %r424, %r534, %r1452;
	fma.rn.f32 	%r1451, %r425, %r533, %r1451;
	fma.rn.f32 	%r1450, %r424, %r532, %r1450;
	fma.rn.f32 	%r1457, %r427, %r531, %r1457;
	fma.rn.f32 	%r1456, %r426, %r530, %r1456;
	fma.rn.f32 	%r1455, %r427, %r529, %r1455;
	fma.rn.f32 	%r1454, %r426, %r528, %r1454;
	fma.rn.f32 	%r1461, %r429, %r527, %r1461;
	fma.rn.f32 	%r1460, %r428, %r526, %r1460;
	fma.rn.f32 	%r1459, %r429, %r525, %r1459;
	fma.rn.f32 	%r1458, %r428, %r524, %r1458;
	fma.rn.f32 	%r1465, %r431, %r523, %r1465;
	fma.rn.f32 	%r1464, %r430, %r522, %r1464;
	fma.rn.f32 	%r1463, %r431, %r521, %r1463;
	fma.rn.f32 	%r1462, %r430, %r520, %r1462;
	fma.rn.f32 	%r1469, %r433, %r519, %r1469;
	fma.rn.f32 	%r1468, %r432, %r518, %r1468;
	fma.rn.f32 	%r1467, %r433, %r517, %r1467;
	fma.rn.f32 	%r1466, %r432, %r516, %r1466;
	fma.rn.f32 	%r1473, %r435, %r515, %r1473;
	fma.rn.f32 	%r1472, %r434, %r514, %r1472;
	fma.rn.f32 	%r1471, %r435, %r513, %r1471;
	fma.rn.f32 	%r1470, %r434, %r512, %r1470;
	fma.rn.f32 	%r1477, %r437, %r511, %r1477;
	fma.rn.f32 	%r1476, %r436, %r510, %r1476;
	fma.rn.f32 	%r1475, %r437, %r509, %r1475;
	fma.rn.f32 	%r1474, %r436, %r508, %r1474;
	fma.rn.f32 	%r1481, %r423, %r507, %r1481;
	fma.rn.f32 	%r1480, %r422, %r506, %r1480;
	fma.rn.f32 	%r1479, %r423, %r505, %r1479;
	fma.rn.f32 	%r1478, %r422, %r504, %r1478;
	fma.rn.f32 	%r1485, %r425, %r503, %r1485;
	fma.rn.f32 	%r1484, %r424, %r502, %r1484;
	fma.rn.f32 	%r1483, %r425, %r501, %r1483;
	fma.rn.f32 	%r1482, %r424, %r500, %r1482;
	fma.rn.f32 	%r1489, %r427, %r499, %r1489;
	fma.rn.f32 	%r1488, %r426, %r498, %r1488;
	fma.rn.f32 	%r1487, %r427, %r497, %r1487;
	fma.rn.f32 	%r1486, %r426, %r496, %r1486;
	fma.rn.f32 	%r1493, %r429, %r495, %r1493;
	fma.rn.f32 	%r1492, %r428, %r494, %r1492;
	fma.rn.f32 	%r1491, %r429, %r493, %r1491;
	fma.rn.f32 	%r1490, %r428, %r492, %r1490;
	fma.rn.f32 	%r1497, %r431, %r491, %r1497;
	fma.rn.f32 	%r1496, %r430, %r490, %r1496;
	fma.rn.f32 	%r1495, %r431, %r489, %r1495;
	fma.rn.f32 	%r1494, %r430, %r488, %r1494;
	fma.rn.f32 	%r1501, %r433, %r487, %r1501;
	fma.rn.f32 	%r1500, %r432, %r486, %r1500;
	fma.rn.f32 	%r1499, %r433, %r485, %r1499;
	fma.rn.f32 	%r1498, %r432, %r484, %r1498;
	fma.rn.f32 	%r1505, %r435, %r483, %r1505;
	fma.rn.f32 	%r1504, %r434, %r482, %r1504;
	fma.rn.f32 	%r1503, %r435, %r481, %r1503;
	fma.rn.f32 	%r1502, %r434, %r480, %r1502;
	fma.rn.f32 	%r1507, %r437, %r479, %r1507;
	fma.rn.f32 	%r1506, %r436, %r478, %r1506;
	fma.rn.f32 	%r1508, %r436, %r477, %r1508;
	fma.rn.f32 	%r1509, %r437, %r476, %r1509;
	fma.rn.f32 	%r1510, %r422, %r475, %r1510;
	fma.rn.f32 	%r1511, %r423, %r474, %r1511;
	fma.rn.f32 	%r1512, %r422, %r473, %r1512;
	fma.rn.f32 	%r1513, %r423, %r472, %r1513;
	fma.rn.f32 	%r1514, %r424, %r471, %r1514;
	fma.rn.f32 	%r1515, %r425, %r470, %r1515;
	fma.rn.f32 	%r1516, %r424, %r469, %r1516;
	fma.rn.f32 	%r1517, %r425, %r468, %r1517;
	fma.rn.f32 	%r1518, %r426, %r467, %r1518;
	fma.rn.f32 	%r1519, %r427, %r466, %r1519;
	fma.rn.f32 	%r1520, %r426, %r465, %r1520;
	fma.rn.f32 	%r1521, %r427, %r464, %r1521;
	fma.rn.f32 	%r1522, %r428, %r463, %r1522;
	fma.rn.f32 	%r1523, %r429, %r462, %r1523;
	fma.rn.f32 	%r1524, %r428, %r461, %r1524;
	fma.rn.f32 	%r1525, %r429, %r460, %r1525;
	fma.rn.f32 	%r1526, %r430, %r459, %r1526;
	fma.rn.f32 	%r1527, %r431, %r458, %r1527;
	fma.rn.f32 	%r1528, %r430, %r457, %r1528;
	fma.rn.f32 	%r1529, %r431, %r456, %r1529;
	fma.rn.f32 	%r1530, %r432, %r455, %r1530;
	fma.rn.f32 	%r1531, %r433, %r454, %r1531;
	fma.rn.f32 	%r1532, %r432, %r453, %r1532;
	fma.rn.f32 	%r1533, %r433, %r452, %r1533;
	fma.rn.f32 	%r1534, %r434, %r451, %r1534;
	fma.rn.f32 	%r1535, %r435, %r450, %r1535;
	fma.rn.f32 	%r1536, %r434, %r449, %r1536;
	fma.rn.f32 	%r1537, %r435, %r448, %r1537;
	fma.rn.f32 	%r1538, %r436, %r447, %r1538;
	fma.rn.f32 	%r1539, %r437, %r446, %r1539;
	fma.rn.f32 	%r1540, %r436, %r445, %r1540;
	fma.rn.f32 	%r1541, %r437, %r444, %r1541;
	.loc	1 164 18                        // sk05_mlp_gateup.py:164:18
	add.s64 	%rd106, %rd31, %rd307;
	add.s64 	%rd107, %rd30, %rd307;
	add.s64 	%rd108, %rd29, %rd307;
	.loc	1 165 18                        // sk05_mlp_gateup.py:165:18
	add.s64 	%rd109, %rd28, %rd307;
	add.s64 	%rd110, %rd27, %rd307;
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	add.s64 	%rd111, %rd26, %rd307;
	add.s32 	%r572, %r1411, 1;
	setp.gt.s32 	%p6, %r572, 2;
	selp.b32 	%r1411, 0, %r572, %p6;
	.loc	1 163 30                        // sk05_mlp_gateup.py:163:30
	shl.b32 	%r573, %r1411, 14;
	bar.sync 	0;
	add.s32 	%r398, %r28, %r573;
	selp.b32 	%r399, 16, 0, %p4;
	// begin inline asm
	cp.async.cg.shared.global [ %r398 + 0 ], [ %rd106 + 0 ], 0x10, %r399;
	// end inline asm
	add.s32 	%r400, %r398, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r400 + 0 ], [ %rd107 + 0 ], 0x10, %r399;
	// end inline asm
	add.s32 	%r401, %r398, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r401 + 0 ], [ %rd108 + 0 ], 0x10, %r399;
	// end inline asm
	add.s32 	%r402, %r398, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r402 + 0 ], [ %rd109 + 0 ], 0x10, %r399;
	// end inline asm
	cp.async.commit_group;
	.loc	1 163 47                        // sk05_mlp_gateup.py:163:47
	shl.b32 	%r574, %r1411, 13;
	add.s32 	%r575, %r28, %r574;
	add.s32 	%r403, %r575, 49152;
	// begin inline asm
	cp.async.cg.shared.global [ %r403 + 0 ], [ %rd110 + 0 ], 0x10, %r399;
	// end inline asm
	add.s32 	%r404, %r575, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r404 + 0 ], [ %rd111 + 0 ], 0x10, %r399;
	// end inline asm
	cp.async.commit_group;
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	add.s64 	%rd308, %rd308, 1;
	add.s64 	%rd307, %rd307, 64;
	add.s32 	%r1409, %r1409, %r27;
	setp.ne.b64 	%p7, %rd25, %rd307;
	@%p7 bra 	$L__BB0_3;
	bra.uni 	$L__BB0_4;
$L__BB0_1:                              // %.._crit_edge_crit_edge
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	and.b32 	%r1413, %r2, 16;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	shl.b32 	%r1412, %r2, 4;
	mov.b32 	%r1414, 0f00000000;
	mov.b32 	%r1415, %r1414;
	mov.b32 	%r1416, %r1414;
	mov.b32 	%r1417, %r1414;
	mov.b32 	%r1418, %r1414;
	mov.b32 	%r1419, %r1414;
	mov.b32 	%r1420, %r1414;
	mov.b32 	%r1421, %r1414;
	mov.b32 	%r1422, %r1414;
	mov.b32 	%r1423, %r1414;
	mov.b32 	%r1424, %r1414;
	mov.b32 	%r1425, %r1414;
	mov.b32 	%r1426, %r1414;
	mov.b32 	%r1427, %r1414;
	mov.b32 	%r1428, %r1414;
	mov.b32 	%r1429, %r1414;
	mov.b32 	%r1430, %r1414;
	mov.b32 	%r1431, %r1414;
	mov.b32 	%r1432, %r1414;
	mov.b32 	%r1433, %r1414;
	mov.b32 	%r1434, %r1414;
	mov.b32 	%r1435, %r1414;
	mov.b32 	%r1436, %r1414;
	mov.b32 	%r1437, %r1414;
	mov.b32 	%r1438, %r1414;
	mov.b32 	%r1439, %r1414;
	mov.b32 	%r1440, %r1414;
	mov.b32 	%r1441, %r1414;
	mov.b32 	%r1442, %r1414;
	mov.b32 	%r1443, %r1414;
	mov.b32 	%r1444, %r1414;
	mov.b32 	%r1445, %r1414;
	mov.b32 	%r1446, %r1414;
	mov.b32 	%r1447, %r1414;
	mov.b32 	%r1448, %r1414;
	mov.b32 	%r1449, %r1414;
	mov.b32 	%r1450, %r1414;
	mov.b32 	%r1451, %r1414;
	mov.b32 	%r1452, %r1414;
	mov.b32 	%r1453, %r1414;
	mov.b32 	%r1454, %r1414;
	mov.b32 	%r1455, %r1414;
	mov.b32 	%r1456, %r1414;
	mov.b32 	%r1457, %r1414;
	mov.b32 	%r1458, %r1414;
	mov.b32 	%r1459, %r1414;
	mov.b32 	%r1460, %r1414;
	mov.b32 	%r1461, %r1414;
	mov.b32 	%r1462, %r1414;
	mov.b32 	%r1463, %r1414;
	mov.b32 	%r1464, %r1414;
	mov.b32 	%r1465, %r1414;
	mov.b32 	%r1466, %r1414;
	mov.b32 	%r1467, %r1414;
	mov.b32 	%r1468, %r1414;
	mov.b32 	%r1469, %r1414;
	mov.b32 	%r1470, %r1414;
	mov.b32 	%r1471, %r1414;
	mov.b32 	%r1472, %r1414;
	mov.b32 	%r1473, %r1414;
	mov.b32 	%r1474, %r1414;
	mov.b32 	%r1475, %r1414;
	mov.b32 	%r1476, %r1414;
	mov.b32 	%r1477, %r1414;
	mov.b32 	%r1478, %r1414;
	mov.b32 	%r1479, %r1414;
	mov.b32 	%r1480, %r1414;
	mov.b32 	%r1481, %r1414;
	mov.b32 	%r1482, %r1414;
	mov.b32 	%r1483, %r1414;
	mov.b32 	%r1484, %r1414;
	mov.b32 	%r1485, %r1414;
	mov.b32 	%r1486, %r1414;
	mov.b32 	%r1487, %r1414;
	mov.b32 	%r1488, %r1414;
	mov.b32 	%r1489, %r1414;
	mov.b32 	%r1490, %r1414;
	mov.b32 	%r1491, %r1414;
	mov.b32 	%r1492, %r1414;
	mov.b32 	%r1493, %r1414;
	mov.b32 	%r1494, %r1414;
	mov.b32 	%r1495, %r1414;
	mov.b32 	%r1496, %r1414;
	mov.b32 	%r1497, %r1414;
	mov.b32 	%r1498, %r1414;
	mov.b32 	%r1499, %r1414;
	mov.b32 	%r1500, %r1414;
	mov.b32 	%r1501, %r1414;
	mov.b32 	%r1502, %r1414;
	mov.b32 	%r1503, %r1414;
	mov.b32 	%r1504, %r1414;
	mov.b32 	%r1505, %r1414;
	mov.b32 	%r1506, %r1414;
	mov.b32 	%r1507, %r1414;
	mov.b32 	%r1508, %r1414;
	mov.b32 	%r1509, %r1414;
	mov.b32 	%r1510, %r1414;
	mov.b32 	%r1511, %r1414;
	mov.b32 	%r1512, %r1414;
	mov.b32 	%r1513, %r1414;
	mov.b32 	%r1514, %r1414;
	mov.b32 	%r1515, %r1414;
	mov.b32 	%r1516, %r1414;
	mov.b32 	%r1517, %r1414;
	mov.b32 	%r1518, %r1414;
	mov.b32 	%r1519, %r1414;
	mov.b32 	%r1520, %r1414;
	mov.b32 	%r1521, %r1414;
	mov.b32 	%r1522, %r1414;
	mov.b32 	%r1523, %r1414;
	mov.b32 	%r1524, %r1414;
	mov.b32 	%r1525, %r1414;
	mov.b32 	%r1526, %r1414;
	mov.b32 	%r1527, %r1414;
	mov.b32 	%r1528, %r1414;
	mov.b32 	%r1529, %r1414;
	mov.b32 	%r1530, %r1414;
	mov.b32 	%r1531, %r1414;
	mov.b32 	%r1532, %r1414;
	mov.b32 	%r1533, %r1414;
	mov.b32 	%r1534, %r1414;
	mov.b32 	%r1535, %r1414;
	mov.b32 	%r1536, %r1414;
	mov.b32 	%r1537, %r1414;
	mov.b32 	%r1538, %r1414;
	mov.b32 	%r1539, %r1414;
	mov.b32 	%r1540, %r1414;
	mov.b32 	%r1541, %r1414;
$L__BB0_4:                              // %._crit_edge
	.loc	1 153 45                        // sk05_mlp_gateup.py:153:45
	shl.b32 	%r806, %r7, 3;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r807, %r806, %r4;
	or.b32 	%r808, %r807, 7;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r809, %r808, %r22;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r810, %r807, 6;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r811, %r810, %r22;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r812, %r807, 5;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r813, %r812, %r22;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r814, %r807, 4;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r815, %r814, %r22;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r816, %r807, 3;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r817, %r816, %r22;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r818, %r807, 2;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r819, %r818, %r22;
	.loc	1 153 32                        // sk05_mlp_gateup.py:153:32
	or.b32 	%r820, %r807, 1;
	.loc	1 153 57                        // sk05_mlp_gateup.py:153:57
	rem.s32 	%r821, %r820, %r22;
	rem.s32 	%r822, %r807, %r22;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r823, %r1, %r3;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r824, %r823, %r21;
	.loc	1 152 45                        // sk05_mlp_gateup.py:152:45
	and.b32 	%r825, %r2, 240;
	bfe.u32 	%r826, %r2, 4, 4;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r827, %r826, %r1;
	or.b32 	%r828, %r827, 240;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r829, %r828, %r21;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r830, %r827, 224;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r831, %r830, %r21;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r832, %r827, 208;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r833, %r832, %r21;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r834, %r827, 192;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r835, %r834, %r21;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r836, %r827, 176;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r837, %r836, %r21;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r838, %r827, 160;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r839, %r838, %r21;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r840, %r827, 144;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r841, %r840, %r21;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r842, %r827, 128;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r843, %r842, %r21;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r844, %r827, 112;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r845, %r844, %r21;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r846, %r827, 96;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r847, %r846, %r21;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r848, %r827, 80;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r849, %r848, %r21;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r850, %r827, 64;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r851, %r850, %r21;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r852, %r827, 48;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r853, %r852, %r21;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r854, %r827, 32;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r855, %r854, %r21;
	.loc	1 152 32                        // sk05_mlp_gateup.py:152:32
	or.b32 	%r856, %r827, 16;
	.loc	1 152 57                        // sk05_mlp_gateup.py:152:57
	rem.s32 	%r857, %r856, %r21;
	rem.s32 	%r858, %r827, %r21;
	.loc	1 161 23                        // sk05_mlp_gateup.py:161:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 167 38                        // sk05_mlp_gateup.py:167:38
	mad.wide.s32 	%rd113, %r824, 4, %rd36;
	.loc	1 167 24                        // sk05_mlp_gateup.py:167:24
	// begin inline asm
	mov.u32 %r577, 0x0;
	ld.global.b32 { %r577 }, [ %rd113 + 0 ];
	// end inline asm
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	and.b32 	%r859, %r2, 7;
	shl.b32 	%r860, %r859, 3;
	shl.b32 	%r861, %r825, 2;
	and.b32 	%r862, %r2, 8;
	shr.u32 	%r863, %r862, 1;
	add.s32 	%r864, %r189, %r860;
	add.s32 	%r865, %r864, %r861;
	add.s32 	%r576, %r865, %r863;
	// begin inline asm
	st.shared.b32 [ %r576 + 0 ], %r577;
	// end inline asm
	bar.sync 	0;
	and.b32 	%r866, %r16, 56;
	and.b32 	%r867, %r2, 192;
	add.s32 	%r868, %r189, %r866;
	add.s32 	%r869, %r868, %r867;
	ld.shared.v2.b32 	{%r870, %r871}, [%r869];
	ld.shared.v2.b32 	{%r872, %r873}, [%r869+256];
	ld.shared.v2.b32 	{%r874, %r875}, [%r869+512];
	ld.shared.v2.b32 	{%r876, %r877}, [%r869+768];
	.loc	1 168 38                        // sk05_mlp_gateup.py:168:38
	mad.wide.s32 	%rd114, %r8, 4, %rd37;
	mad.wide.s32 	%rd115, %r9, 4, %rd37;
	mad.wide.s32 	%rd116, %r10, 4, %rd37;
	mad.wide.s32 	%rd117, %r11, 4, %rd37;
	mad.wide.s32 	%rd118, %r12, 4, %rd37;
	mad.wide.s32 	%rd119, %r13, 4, %rd37;
	mad.wide.s32 	%rd120, %r14, 4, %rd37;
	mad.wide.s32 	%rd121, %r15, 4, %rd37;
	.loc	1 168 24                        // sk05_mlp_gateup.py:168:24
	// begin inline asm
	mov.u32 %r578, 0x0;
	mov.u32 %r579, 0x0;
	ld.global.v2.b32 { %r578, %r579 }, [ %rd114 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r580, 0x0;
	mov.u32 %r581, 0x0;
	ld.global.v2.b32 { %r580, %r581 }, [ %rd115 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r582, 0x0;
	mov.u32 %r583, 0x0;
	ld.global.v2.b32 { %r582, %r583 }, [ %rd116 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r584, 0x0;
	mov.u32 %r585, 0x0;
	ld.global.v2.b32 { %r584, %r585 }, [ %rd117 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r586, 0x0;
	mov.u32 %r587, 0x0;
	ld.global.v2.b32 { %r586, %r587 }, [ %rd118 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r588, 0x0;
	mov.u32 %r589, 0x0;
	ld.global.v2.b32 { %r588, %r589 }, [ %rd119 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r590, 0x0;
	mov.u32 %r591, 0x0;
	ld.global.v2.b32 { %r590, %r591 }, [ %rd120 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r592, 0x0;
	mov.u32 %r593, 0x0;
	ld.global.v2.b32 { %r592, %r593 }, [ %rd121 + 0 ];
	// end inline asm
	.loc	1 169 49                        // sk05_mlp_gateup.py:169:49
	mul.lo.s32 	%r878, %r858, %r25;
	mul.lo.s32 	%r879, %r857, %r25;
	mul.lo.s32 	%r880, %r855, %r25;
	mul.lo.s32 	%r881, %r853, %r25;
	mul.lo.s32 	%r882, %r851, %r25;
	mul.lo.s32 	%r883, %r849, %r25;
	mul.lo.s32 	%r884, %r847, %r25;
	mul.lo.s32 	%r885, %r845, %r25;
	mul.lo.s32 	%r886, %r843, %r25;
	mul.lo.s32 	%r887, %r841, %r25;
	mul.lo.s32 	%r888, %r839, %r25;
	mul.lo.s32 	%r889, %r837, %r25;
	mul.lo.s32 	%r890, %r835, %r25;
	mul.lo.s32 	%r891, %r833, %r25;
	mul.lo.s32 	%r892, %r831, %r25;
	mul.lo.s32 	%r893, %r829, %r25;
	.loc	1 169 31                        // sk05_mlp_gateup.py:169:31
	mad.wide.s32 	%rd266, %r878, 2, %rd35;
	mad.wide.s32 	%rd267, %r879, 2, %rd35;
	mad.wide.s32 	%rd268, %r880, 2, %rd35;
	mad.wide.s32 	%rd269, %r881, 2, %rd35;
	mad.wide.s32 	%rd270, %r882, 2, %rd35;
	mad.wide.s32 	%rd271, %r883, 2, %rd35;
	mad.wide.s32 	%rd272, %r884, 2, %rd35;
	mad.wide.s32 	%rd273, %r885, 2, %rd35;
	mad.wide.s32 	%rd274, %r886, 2, %rd35;
	mad.wide.s32 	%rd275, %r887, 2, %rd35;
	mad.wide.s32 	%rd276, %r888, 2, %rd35;
	mad.wide.s32 	%rd277, %r889, 2, %rd35;
	mad.wide.s32 	%rd278, %r890, 2, %rd35;
	mad.wide.s32 	%rd279, %r891, 2, %rd35;
	mad.wide.s32 	%rd280, %r892, 2, %rd35;
	mad.wide.s32 	%rd281, %r893, 2, %rd35;
	.loc	1 169 82                        // sk05_mlp_gateup.py:169:82
	mul.lo.s32 	%r894, %r822, %r26;
	mul.lo.s32 	%r895, %r821, %r26;
	mul.lo.s32 	%r896, %r819, %r26;
	mul.lo.s32 	%r897, %r817, %r26;
	mul.lo.s32 	%r898, %r815, %r26;
	mul.lo.s32 	%r899, %r813, %r26;
	mul.lo.s32 	%r900, %r811, %r26;
	mul.lo.s32 	%r901, %r809, %r26;
	.loc	1 169 64                        // sk05_mlp_gateup.py:169:64
	mul.wide.s32 	%rd282, %r894, 2;
	add.s64 	%rd122, %rd266, %rd282;
	mul.wide.s32 	%rd283, %r895, 2;
	add.s64 	%rd123, %rd266, %rd283;
	mul.wide.s32 	%rd284, %r896, 2;
	add.s64 	%rd124, %rd266, %rd284;
	mul.wide.s32 	%rd285, %r897, 2;
	add.s64 	%rd125, %rd266, %rd285;
	mul.wide.s32 	%rd286, %r898, 2;
	add.s64 	%rd126, %rd266, %rd286;
	mul.wide.s32 	%rd287, %r899, 2;
	add.s64 	%rd127, %rd266, %rd287;
	mul.wide.s32 	%rd288, %r900, 2;
	add.s64 	%rd128, %rd266, %rd288;
	mul.wide.s32 	%rd289, %r901, 2;
	add.s64 	%rd129, %rd266, %rd289;
	add.s64 	%rd130, %rd267, %rd282;
	add.s64 	%rd131, %rd267, %rd283;
	add.s64 	%rd132, %rd267, %rd284;
	add.s64 	%rd133, %rd267, %rd285;
	add.s64 	%rd134, %rd267, %rd286;
	add.s64 	%rd135, %rd267, %rd287;
	add.s64 	%rd136, %rd267, %rd288;
	add.s64 	%rd137, %rd267, %rd289;
	add.s64 	%rd138, %rd268, %rd282;
	add.s64 	%rd139, %rd268, %rd283;
	add.s64 	%rd140, %rd268, %rd284;
	add.s64 	%rd141, %rd268, %rd285;
	add.s64 	%rd142, %rd268, %rd286;
	add.s64 	%rd143, %rd268, %rd287;
	add.s64 	%rd144, %rd268, %rd288;
	add.s64 	%rd145, %rd268, %rd289;
	add.s64 	%rd146, %rd269, %rd282;
	add.s64 	%rd147, %rd269, %rd283;
	add.s64 	%rd148, %rd269, %rd284;
	add.s64 	%rd149, %rd269, %rd285;
	add.s64 	%rd150, %rd269, %rd286;
	add.s64 	%rd151, %rd269, %rd287;
	add.s64 	%rd152, %rd269, %rd288;
	add.s64 	%rd153, %rd269, %rd289;
	add.s64 	%rd154, %rd270, %rd282;
	add.s64 	%rd155, %rd270, %rd283;
	add.s64 	%rd156, %rd270, %rd284;
	add.s64 	%rd157, %rd270, %rd285;
	add.s64 	%rd158, %rd270, %rd286;
	add.s64 	%rd159, %rd270, %rd287;
	add.s64 	%rd160, %rd270, %rd288;
	add.s64 	%rd161, %rd270, %rd289;
	add.s64 	%rd162, %rd271, %rd282;
	add.s64 	%rd163, %rd271, %rd283;
	add.s64 	%rd164, %rd271, %rd284;
	add.s64 	%rd165, %rd271, %rd285;
	add.s64 	%rd166, %rd271, %rd286;
	add.s64 	%rd167, %rd271, %rd287;
	add.s64 	%rd168, %rd271, %rd288;
	add.s64 	%rd169, %rd271, %rd289;
	add.s64 	%rd170, %rd272, %rd282;
	add.s64 	%rd171, %rd272, %rd283;
	add.s64 	%rd172, %rd272, %rd284;
	add.s64 	%rd173, %rd272, %rd285;
	add.s64 	%rd174, %rd272, %rd286;
	add.s64 	%rd175, %rd272, %rd287;
	add.s64 	%rd176, %rd272, %rd288;
	add.s64 	%rd177, %rd272, %rd289;
	add.s64 	%rd178, %rd273, %rd282;
	add.s64 	%rd179, %rd273, %rd283;
	add.s64 	%rd180, %rd273, %rd284;
	add.s64 	%rd181, %rd273, %rd285;
	add.s64 	%rd182, %rd273, %rd286;
	add.s64 	%rd183, %rd273, %rd287;
	add.s64 	%rd184, %rd273, %rd288;
	add.s64 	%rd185, %rd273, %rd289;
	add.s64 	%rd186, %rd274, %rd282;
	add.s64 	%rd187, %rd274, %rd283;
	add.s64 	%rd188, %rd274, %rd284;
	add.s64 	%rd189, %rd274, %rd285;
	add.s64 	%rd190, %rd274, %rd286;
	add.s64 	%rd191, %rd274, %rd287;
	add.s64 	%rd192, %rd274, %rd288;
	add.s64 	%rd193, %rd274, %rd289;
	add.s64 	%rd194, %rd275, %rd282;
	add.s64 	%rd195, %rd275, %rd283;
	add.s64 	%rd196, %rd275, %rd284;
	add.s64 	%rd197, %rd275, %rd285;
	add.s64 	%rd198, %rd275, %rd286;
	add.s64 	%rd199, %rd275, %rd287;
	add.s64 	%rd200, %rd275, %rd288;
	add.s64 	%rd201, %rd275, %rd289;
	add.s64 	%rd202, %rd276, %rd282;
	add.s64 	%rd203, %rd276, %rd283;
	add.s64 	%rd204, %rd276, %rd284;
	add.s64 	%rd205, %rd276, %rd285;
	add.s64 	%rd206, %rd276, %rd286;
	add.s64 	%rd207, %rd276, %rd287;
	add.s64 	%rd208, %rd276, %rd288;
	add.s64 	%rd209, %rd276, %rd289;
	add.s64 	%rd210, %rd277, %rd282;
	add.s64 	%rd211, %rd277, %rd283;
	add.s64 	%rd212, %rd277, %rd284;
	add.s64 	%rd213, %rd277, %rd285;
	add.s64 	%rd214, %rd277, %rd286;
	add.s64 	%rd215, %rd277, %rd287;
	add.s64 	%rd216, %rd277, %rd288;
	add.s64 	%rd217, %rd277, %rd289;
	add.s64 	%rd218, %rd278, %rd282;
	add.s64 	%rd219, %rd278, %rd283;
	add.s64 	%rd220, %rd278, %rd284;
	add.s64 	%rd221, %rd278, %rd285;
	add.s64 	%rd222, %rd278, %rd286;
	add.s64 	%rd223, %rd278, %rd287;
	add.s64 	%rd224, %rd278, %rd288;
	add.s64 	%rd225, %rd278, %rd289;
	add.s64 	%rd226, %rd279, %rd282;
	add.s64 	%rd227, %rd279, %rd283;
	add.s64 	%rd228, %rd279, %rd284;
	add.s64 	%rd229, %rd279, %rd285;
	add.s64 	%rd230, %rd279, %rd286;
	add.s64 	%rd231, %rd279, %rd287;
	add.s64 	%rd232, %rd279, %rd288;
	add.s64 	%rd233, %rd279, %rd289;
	add.s64 	%rd234, %rd280, %rd282;
	add.s64 	%rd235, %rd280, %rd283;
	add.s64 	%rd236, %rd280, %rd284;
	add.s64 	%rd237, %rd280, %rd285;
	add.s64 	%rd238, %rd280, %rd286;
	add.s64 	%rd239, %rd280, %rd287;
	add.s64 	%rd240, %rd280, %rd288;
	add.s64 	%rd241, %rd280, %rd289;
	add.s64 	%rd242, %rd281, %rd282;
	add.s64 	%rd243, %rd281, %rd283;
	add.s64 	%rd244, %rd281, %rd284;
	add.s64 	%rd245, %rd281, %rd285;
	add.s64 	%rd246, %rd281, %rd286;
	add.s64 	%rd247, %rd281, %rd287;
	add.s64 	%rd248, %rd281, %rd288;
	add.s64 	%rd249, %rd281, %rd289;
	.loc	1 169 19                        // sk05_mlp_gateup.py:169:19
	// begin inline asm
	mov.u16 %rs33, 0x0;
	ld.global.b16 { %rs33 }, [ %rd122 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs34, 0x0;
	ld.global.b16 { %rs34 }, [ %rd123 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs35, 0x0;
	ld.global.b16 { %rs35 }, [ %rd124 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs36, 0x0;
	ld.global.b16 { %rs36 }, [ %rd125 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs37, 0x0;
	ld.global.b16 { %rs37 }, [ %rd126 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs38, 0x0;
	ld.global.b16 { %rs38 }, [ %rd127 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs39, 0x0;
	ld.global.b16 { %rs39 }, [ %rd128 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs40, 0x0;
	ld.global.b16 { %rs40 }, [ %rd129 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs41, 0x0;
	ld.global.b16 { %rs41 }, [ %rd130 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs42, 0x0;
	ld.global.b16 { %rs42 }, [ %rd131 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs43, 0x0;
	ld.global.b16 { %rs43 }, [ %rd132 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs44, 0x0;
	ld.global.b16 { %rs44 }, [ %rd133 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs45, 0x0;
	ld.global.b16 { %rs45 }, [ %rd134 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs46, 0x0;
	ld.global.b16 { %rs46 }, [ %rd135 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs47, 0x0;
	ld.global.b16 { %rs47 }, [ %rd136 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs48, 0x0;
	ld.global.b16 { %rs48 }, [ %rd137 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs49, 0x0;
	ld.global.b16 { %rs49 }, [ %rd138 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs50, 0x0;
	ld.global.b16 { %rs50 }, [ %rd139 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs51, 0x0;
	ld.global.b16 { %rs51 }, [ %rd140 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs52, 0x0;
	ld.global.b16 { %rs52 }, [ %rd141 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs53, 0x0;
	ld.global.b16 { %rs53 }, [ %rd142 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs54, 0x0;
	ld.global.b16 { %rs54 }, [ %rd143 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs55, 0x0;
	ld.global.b16 { %rs55 }, [ %rd144 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs56, 0x0;
	ld.global.b16 { %rs56 }, [ %rd145 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs57, 0x0;
	ld.global.b16 { %rs57 }, [ %rd146 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs58, 0x0;
	ld.global.b16 { %rs58 }, [ %rd147 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs59, 0x0;
	ld.global.b16 { %rs59 }, [ %rd148 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs60, 0x0;
	ld.global.b16 { %rs60 }, [ %rd149 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs61, 0x0;
	ld.global.b16 { %rs61 }, [ %rd150 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs62, 0x0;
	ld.global.b16 { %rs62 }, [ %rd151 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs63, 0x0;
	ld.global.b16 { %rs63 }, [ %rd152 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs64, 0x0;
	ld.global.b16 { %rs64 }, [ %rd153 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs65, 0x0;
	ld.global.b16 { %rs65 }, [ %rd154 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs66, 0x0;
	ld.global.b16 { %rs66 }, [ %rd155 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs67, 0x0;
	ld.global.b16 { %rs67 }, [ %rd156 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs68, 0x0;
	ld.global.b16 { %rs68 }, [ %rd157 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs69, 0x0;
	ld.global.b16 { %rs69 }, [ %rd158 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs70, 0x0;
	ld.global.b16 { %rs70 }, [ %rd159 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs71, 0x0;
	ld.global.b16 { %rs71 }, [ %rd160 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs72, 0x0;
	ld.global.b16 { %rs72 }, [ %rd161 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs73, 0x0;
	ld.global.b16 { %rs73 }, [ %rd162 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs74, 0x0;
	ld.global.b16 { %rs74 }, [ %rd163 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs75, 0x0;
	ld.global.b16 { %rs75 }, [ %rd164 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs76, 0x0;
	ld.global.b16 { %rs76 }, [ %rd165 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs77, 0x0;
	ld.global.b16 { %rs77 }, [ %rd166 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs78, 0x0;
	ld.global.b16 { %rs78 }, [ %rd167 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs79, 0x0;
	ld.global.b16 { %rs79 }, [ %rd168 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs80, 0x0;
	ld.global.b16 { %rs80 }, [ %rd169 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs81, 0x0;
	ld.global.b16 { %rs81 }, [ %rd170 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs82, 0x0;
	ld.global.b16 { %rs82 }, [ %rd171 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs83, 0x0;
	ld.global.b16 { %rs83 }, [ %rd172 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs84, 0x0;
	ld.global.b16 { %rs84 }, [ %rd173 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs85, 0x0;
	ld.global.b16 { %rs85 }, [ %rd174 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs86, 0x0;
	ld.global.b16 { %rs86 }, [ %rd175 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs87, 0x0;
	ld.global.b16 { %rs87 }, [ %rd176 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs88, 0x0;
	ld.global.b16 { %rs88 }, [ %rd177 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs89, 0x0;
	ld.global.b16 { %rs89 }, [ %rd178 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs90, 0x0;
	ld.global.b16 { %rs90 }, [ %rd179 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs91, 0x0;
	ld.global.b16 { %rs91 }, [ %rd180 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs92, 0x0;
	ld.global.b16 { %rs92 }, [ %rd181 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs93, 0x0;
	ld.global.b16 { %rs93 }, [ %rd182 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs94, 0x0;
	ld.global.b16 { %rs94 }, [ %rd183 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs95, 0x0;
	ld.global.b16 { %rs95 }, [ %rd184 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs96, 0x0;
	ld.global.b16 { %rs96 }, [ %rd185 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs97, 0x0;
	ld.global.b16 { %rs97 }, [ %rd186 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs98, 0x0;
	ld.global.b16 { %rs98 }, [ %rd187 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs99, 0x0;
	ld.global.b16 { %rs99 }, [ %rd188 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs100, 0x0;
	ld.global.b16 { %rs100 }, [ %rd189 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs101, 0x0;
	ld.global.b16 { %rs101 }, [ %rd190 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs102, 0x0;
	ld.global.b16 { %rs102 }, [ %rd191 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs103, 0x0;
	ld.global.b16 { %rs103 }, [ %rd192 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs104, 0x0;
	ld.global.b16 { %rs104 }, [ %rd193 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs105, 0x0;
	ld.global.b16 { %rs105 }, [ %rd194 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs106, 0x0;
	ld.global.b16 { %rs106 }, [ %rd195 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs107, 0x0;
	ld.global.b16 { %rs107 }, [ %rd196 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs108, 0x0;
	ld.global.b16 { %rs108 }, [ %rd197 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs109, 0x0;
	ld.global.b16 { %rs109 }, [ %rd198 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs110, 0x0;
	ld.global.b16 { %rs110 }, [ %rd199 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs111, 0x0;
	ld.global.b16 { %rs111 }, [ %rd200 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs112, 0x0;
	ld.global.b16 { %rs112 }, [ %rd201 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs113, 0x0;
	ld.global.b16 { %rs113 }, [ %rd202 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs114, 0x0;
	ld.global.b16 { %rs114 }, [ %rd203 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs115, 0x0;
	ld.global.b16 { %rs115 }, [ %rd204 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs116, 0x0;
	ld.global.b16 { %rs116 }, [ %rd205 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs117, 0x0;
	ld.global.b16 { %rs117 }, [ %rd206 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs118, 0x0;
	ld.global.b16 { %rs118 }, [ %rd207 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs119, 0x0;
	ld.global.b16 { %rs119 }, [ %rd208 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs120, 0x0;
	ld.global.b16 { %rs120 }, [ %rd209 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs121, 0x0;
	ld.global.b16 { %rs121 }, [ %rd210 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs122, 0x0;
	ld.global.b16 { %rs122 }, [ %rd211 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs123, 0x0;
	ld.global.b16 { %rs123 }, [ %rd212 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs124, 0x0;
	ld.global.b16 { %rs124 }, [ %rd213 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs125, 0x0;
	ld.global.b16 { %rs125 }, [ %rd214 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs126, 0x0;
	ld.global.b16 { %rs126 }, [ %rd215 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs127, 0x0;
	ld.global.b16 { %rs127 }, [ %rd216 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs128, 0x0;
	ld.global.b16 { %rs128 }, [ %rd217 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs129, 0x0;
	ld.global.b16 { %rs129 }, [ %rd218 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs130, 0x0;
	ld.global.b16 { %rs130 }, [ %rd219 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs131, 0x0;
	ld.global.b16 { %rs131 }, [ %rd220 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs132, 0x0;
	ld.global.b16 { %rs132 }, [ %rd221 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs133, 0x0;
	ld.global.b16 { %rs133 }, [ %rd222 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs134, 0x0;
	ld.global.b16 { %rs134 }, [ %rd223 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs135, 0x0;
	ld.global.b16 { %rs135 }, [ %rd224 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs136, 0x0;
	ld.global.b16 { %rs136 }, [ %rd225 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs137, 0x0;
	ld.global.b16 { %rs137 }, [ %rd226 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs138, 0x0;
	ld.global.b16 { %rs138 }, [ %rd227 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs139, 0x0;
	ld.global.b16 { %rs139 }, [ %rd228 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs140, 0x0;
	ld.global.b16 { %rs140 }, [ %rd229 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs141, 0x0;
	ld.global.b16 { %rs141 }, [ %rd230 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs142, 0x0;
	ld.global.b16 { %rs142 }, [ %rd231 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs143, 0x0;
	ld.global.b16 { %rs143 }, [ %rd232 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs144, 0x0;
	ld.global.b16 { %rs144 }, [ %rd233 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs145, 0x0;
	ld.global.b16 { %rs145 }, [ %rd234 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs146, 0x0;
	ld.global.b16 { %rs146 }, [ %rd235 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs147, 0x0;
	ld.global.b16 { %rs147 }, [ %rd236 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs148, 0x0;
	ld.global.b16 { %rs148 }, [ %rd237 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs149, 0x0;
	ld.global.b16 { %rs149 }, [ %rd238 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs150, 0x0;
	ld.global.b16 { %rs150 }, [ %rd239 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs151, 0x0;
	ld.global.b16 { %rs151 }, [ %rd240 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs152, 0x0;
	ld.global.b16 { %rs152 }, [ %rd241 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs153, 0x0;
	ld.global.b16 { %rs153 }, [ %rd242 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs154, 0x0;
	ld.global.b16 { %rs154 }, [ %rd243 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs155, 0x0;
	ld.global.b16 { %rs155 }, [ %rd244 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs156, 0x0;
	ld.global.b16 { %rs156 }, [ %rd245 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs157, 0x0;
	ld.global.b16 { %rs157 }, [ %rd246 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs158, 0x0;
	ld.global.b16 { %rs158 }, [ %rd247 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs159, 0x0;
	ld.global.b16 { %rs159 }, [ %rd248 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs160, 0x0;
	ld.global.b16 { %rs160 }, [ %rd249 + 0 ];
	// end inline asm
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	bar.sync 	0;
	shl.b32 	%r902, %r2, 7;
	and.b32 	%r903, %r902, 15360;
	shl.b32 	%r904, %r859, 4;
	or.b32 	%r905, %r903, %r904;
	xor.b32 	%r906, %r905, %r825;
	add.s32 	%r594, %r189, %r906;
	mov.b32 	%r595, {%rs33, %rs34};
	mov.b32 	%r596, {%rs35, %rs36};
	mov.b32 	%r597, {%rs37, %rs38};
	mov.b32 	%r598, {%rs39, %rs40};
	// begin inline asm
	st.shared.v4.b32 [ %r594 + 0 ], { %r595, %r596, %r597, %r598 };
	// end inline asm
	add.s32 	%r599, %r594, 256;
	mov.b32 	%r600, {%rs41, %rs42};
	mov.b32 	%r601, {%rs43, %rs44};
	mov.b32 	%r602, {%rs45, %rs46};
	mov.b32 	%r603, {%rs47, %rs48};
	// begin inline asm
	st.shared.v4.b32 [ %r599 + 0 ], { %r600, %r601, %r602, %r603 };
	// end inline asm
	add.s32 	%r604, %r594, 512;
	mov.b32 	%r605, {%rs49, %rs50};
	mov.b32 	%r606, {%rs51, %rs52};
	mov.b32 	%r607, {%rs53, %rs54};
	mov.b32 	%r608, {%rs55, %rs56};
	// begin inline asm
	st.shared.v4.b32 [ %r604 + 0 ], { %r605, %r606, %r607, %r608 };
	// end inline asm
	add.s32 	%r609, %r594, 768;
	mov.b32 	%r610, {%rs57, %rs58};
	mov.b32 	%r611, {%rs59, %rs60};
	mov.b32 	%r612, {%rs61, %rs62};
	mov.b32 	%r613, {%rs63, %rs64};
	// begin inline asm
	st.shared.v4.b32 [ %r609 + 0 ], { %r610, %r611, %r612, %r613 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r907, %r859, 11;
	shl.b32 	%r908, %r7, 4;
	shl.b32 	%r909, %r867, 2;
	setp.eq.b32 	%p24, %r1413, 0;
	shl.b32 	%r910, %r1413, 1;
	shr.u32 	%r911, %r6, 1;
	or.b32 	%r912, %r908, %r909;
	or.b32 	%r913, %r910, %r911;
	xor.b32 	%r914, %r912, %r913;
	or.b32 	%r915, %r914, %r907;
	add.s32 	%r916, %r189, %r915;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r917, %r918, %r919, %r920}, [%r916];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r921, %r922, %r923, %r924}, [%r916+1024];
	xor.b32 	%r925, %r915, 64;
	add.s32 	%r926, %r189, %r925;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r927, %r928, %r929, %r930}, [%r926];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r931, %r932, %r933, %r934}, [%r926+1024];
	bar.sync 	0;
	mov.b32 	%r614, {%rs65, %rs66};
	mov.b32 	%r615, {%rs67, %rs68};
	mov.b32 	%r616, {%rs69, %rs70};
	mov.b32 	%r617, {%rs71, %rs72};
	// begin inline asm
	st.shared.v4.b32 [ %r594 + 0 ], { %r614, %r615, %r616, %r617 };
	// end inline asm
	mov.b32 	%r618, {%rs73, %rs74};
	mov.b32 	%r619, {%rs75, %rs76};
	mov.b32 	%r620, {%rs77, %rs78};
	mov.b32 	%r621, {%rs79, %rs80};
	// begin inline asm
	st.shared.v4.b32 [ %r599 + 0 ], { %r618, %r619, %r620, %r621 };
	// end inline asm
	mov.b32 	%r622, {%rs81, %rs82};
	mov.b32 	%r623, {%rs83, %rs84};
	mov.b32 	%r624, {%rs85, %rs86};
	mov.b32 	%r625, {%rs87, %rs88};
	// begin inline asm
	st.shared.v4.b32 [ %r604 + 0 ], { %r622, %r623, %r624, %r625 };
	// end inline asm
	mov.b32 	%r626, {%rs89, %rs90};
	mov.b32 	%r627, {%rs91, %rs92};
	mov.b32 	%r628, {%rs93, %rs94};
	mov.b32 	%r629, {%rs95, %rs96};
	// begin inline asm
	st.shared.v4.b32 [ %r609 + 0 ], { %r626, %r627, %r628, %r629 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r935, %r936, %r937, %r938}, [%r916];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r939, %r940, %r941, %r942}, [%r916+1024];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r943, %r944, %r945, %r946}, [%r926];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r947, %r948, %r949, %r950}, [%r926+1024];
	bar.sync 	0;
	mov.b32 	%r630, {%rs97, %rs98};
	mov.b32 	%r631, {%rs99, %rs100};
	mov.b32 	%r632, {%rs101, %rs102};
	mov.b32 	%r633, {%rs103, %rs104};
	// begin inline asm
	st.shared.v4.b32 [ %r594 + 0 ], { %r630, %r631, %r632, %r633 };
	// end inline asm
	mov.b32 	%r634, {%rs105, %rs106};
	mov.b32 	%r635, {%rs107, %rs108};
	mov.b32 	%r636, {%rs109, %rs110};
	mov.b32 	%r637, {%rs111, %rs112};
	// begin inline asm
	st.shared.v4.b32 [ %r599 + 0 ], { %r634, %r635, %r636, %r637 };
	// end inline asm
	mov.b32 	%r638, {%rs113, %rs114};
	mov.b32 	%r639, {%rs115, %rs116};
	mov.b32 	%r640, {%rs117, %rs118};
	mov.b32 	%r641, {%rs119, %rs120};
	// begin inline asm
	st.shared.v4.b32 [ %r604 + 0 ], { %r638, %r639, %r640, %r641 };
	// end inline asm
	mov.b32 	%r642, {%rs121, %rs122};
	mov.b32 	%r643, {%rs123, %rs124};
	mov.b32 	%r644, {%rs125, %rs126};
	mov.b32 	%r645, {%rs127, %rs128};
	// begin inline asm
	st.shared.v4.b32 [ %r609 + 0 ], { %r642, %r643, %r644, %r645 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r951, %r952, %r953, %r954}, [%r916];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r955, %r956, %r957, %r958}, [%r916+1024];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r959, %r960, %r961, %r962}, [%r926];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r963, %r964, %r965, %r966}, [%r926+1024];
	bar.sync 	0;
	mov.b32 	%r646, {%rs129, %rs130};
	mov.b32 	%r647, {%rs131, %rs132};
	mov.b32 	%r648, {%rs133, %rs134};
	mov.b32 	%r649, {%rs135, %rs136};
	// begin inline asm
	st.shared.v4.b32 [ %r594 + 0 ], { %r646, %r647, %r648, %r649 };
	// end inline asm
	mov.b32 	%r650, {%rs137, %rs138};
	mov.b32 	%r651, {%rs139, %rs140};
	mov.b32 	%r652, {%rs141, %rs142};
	mov.b32 	%r653, {%rs143, %rs144};
	// begin inline asm
	st.shared.v4.b32 [ %r599 + 0 ], { %r650, %r651, %r652, %r653 };
	// end inline asm
	mov.b32 	%r654, {%rs145, %rs146};
	mov.b32 	%r655, {%rs147, %rs148};
	mov.b32 	%r656, {%rs149, %rs150};
	mov.b32 	%r657, {%rs151, %rs152};
	// begin inline asm
	st.shared.v4.b32 [ %r604 + 0 ], { %r654, %r655, %r656, %r657 };
	// end inline asm
	mov.b32 	%r658, {%rs153, %rs154};
	mov.b32 	%r659, {%rs155, %rs156};
	mov.b32 	%r660, {%rs157, %rs158};
	mov.b32 	%r661, {%rs159, %rs160};
	// begin inline asm
	st.shared.v4.b32 [ %r609 + 0 ], { %r658, %r659, %r660, %r661 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r967, %r968, %r969, %r970}, [%r916];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r971, %r972, %r973, %r974}, [%r916+1024];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r975, %r976, %r977, %r978}, [%r926];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r979, %r980, %r981, %r982}, [%r926+1024];
	.loc	1 176 31                        // sk05_mlp_gateup.py:176:31
	setp.lt.s32 	%p25, %r827, %r21;
	setp.lt.s32 	%p26, %r856, %r21;
	setp.lt.s32 	%p27, %r854, %r21;
	setp.lt.s32 	%p28, %r852, %r21;
	setp.lt.s32 	%p29, %r850, %r21;
	setp.lt.s32 	%p30, %r848, %r21;
	setp.lt.s32 	%p31, %r846, %r21;
	setp.lt.s32 	%p32, %r844, %r21;
	setp.lt.s32 	%p33, %r842, %r21;
	setp.lt.s32 	%p34, %r840, %r21;
	setp.lt.s32 	%p35, %r838, %r21;
	setp.lt.s32 	%p36, %r836, %r21;
	setp.lt.s32 	%p37, %r834, %r21;
	setp.lt.s32 	%p38, %r832, %r21;
	setp.lt.s32 	%p39, %r830, %r21;
	setp.lt.s32 	%p40, %r828, %r21;
	.loc	1 176 54                        // sk05_mlp_gateup.py:176:54
	setp.lt.s32 	%p41, %r807, %r22;
	.loc	1 176 37                        // sk05_mlp_gateup.py:176:37
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
	.loc	1 174 35                        // sk05_mlp_gateup.py:174:35
	mul.lo.s32 	%r983, %r827, %r24;
	mul.lo.s32 	%r984, %r856, %r24;
	mul.lo.s32 	%r985, %r854, %r24;
	mul.lo.s32 	%r986, %r852, %r24;
	mul.lo.s32 	%r987, %r850, %r24;
	mul.lo.s32 	%r988, %r848, %r24;
	mul.lo.s32 	%r989, %r846, %r24;
	mul.lo.s32 	%r990, %r844, %r24;
	mul.lo.s32 	%r991, %r842, %r24;
	mul.lo.s32 	%r992, %r840, %r24;
	mul.lo.s32 	%r993, %r838, %r24;
	mul.lo.s32 	%r994, %r836, %r24;
	mul.lo.s32 	%r995, %r834, %r24;
	mul.lo.s32 	%r996, %r832, %r24;
	mul.lo.s32 	%r997, %r830, %r24;
	mul.lo.s32 	%r998, %r828, %r24;
	.loc	1 174 18                        // sk05_mlp_gateup.py:174:18
	mad.wide.s32 	%rd290, %r983, 2, %rd34;
	mad.wide.s32 	%rd291, %r984, 2, %rd34;
	mad.wide.s32 	%rd292, %r985, 2, %rd34;
	mad.wide.s32 	%rd293, %r986, 2, %rd34;
	mad.wide.s32 	%rd294, %r987, 2, %rd34;
	mad.wide.s32 	%rd295, %r988, 2, %rd34;
	mad.wide.s32 	%rd296, %r989, 2, %rd34;
	mad.wide.s32 	%rd297, %r990, 2, %rd34;
	mad.wide.s32 	%rd298, %r991, 2, %rd34;
	mad.wide.s32 	%rd299, %r992, 2, %rd34;
	mad.wide.s32 	%rd300, %r993, 2, %rd34;
	mad.wide.s32 	%rd301, %r994, 2, %rd34;
	mad.wide.s32 	%rd302, %r995, 2, %rd34;
	mad.wide.s32 	%rd303, %r996, 2, %rd34;
	mad.wide.s32 	%rd304, %r997, 2, %rd34;
	mad.wide.s32 	%rd305, %r998, 2, %rd34;
	.loc	1 174 50                        // sk05_mlp_gateup.py:174:50
	mul.wide.s32 	%rd306, %r807, 2;
	add.s64 	%rd250, %rd290, %rd306;
	add.s64 	%rd251, %rd291, %rd306;
	add.s64 	%rd252, %rd292, %rd306;
	add.s64 	%rd253, %rd293, %rd306;
	add.s64 	%rd254, %rd294, %rd306;
	add.s64 	%rd255, %rd295, %rd306;
	add.s64 	%rd256, %rd296, %rd306;
	add.s64 	%rd257, %rd297, %rd306;
	add.s64 	%rd258, %rd298, %rd306;
	add.s64 	%rd259, %rd299, %rd306;
	add.s64 	%rd260, %rd300, %rd306;
	add.s64 	%rd261, %rd301, %rd306;
	add.s64 	%rd262, %rd302, %rd306;
	add.s64 	%rd263, %rd303, %rd306;
	add.s64 	%rd264, %rd304, %rd306;
	add.s64 	%rd265, %rd305, %rd306;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r999, %r1511, %r876;
	mul.f32 	%r1000, %r1510, %r876;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs161, %rs162}, %r967;
	cvt.f32.bf16 	%r1001, %rs162;
	cvt.f32.bf16 	%r1002, %rs161;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1003, %r1000, %r578, %r1002;
	fma.rn.f32 	%r1004, %r999, %r579, %r1001;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1005, %r1415, %r870;
	mul.f32 	%r1006, %r1414, %r870;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs163, %rs164}, %r917;
	cvt.f32.bf16 	%r1007, %rs164;
	cvt.f32.bf16 	%r1008, %rs163;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1009, %r1006, %r578, %r1008;
	fma.rn.f32 	%r1010, %r1005, %r579, %r1007;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r663, %r1010, %r1009;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1011, %r1417, %r871;
	mul.f32 	%r1012, %r1416, %r871;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs165, %rs166}, %r918;
	cvt.f32.bf16 	%r1013, %rs166;
	cvt.f32.bf16 	%r1014, %rs165;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1015, %r1012, %r578, %r1014;
	fma.rn.f32 	%r1016, %r1011, %r579, %r1013;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r668, %r1016, %r1015;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1017, %r1447, %r872;
	mul.f32 	%r1018, %r1446, %r872;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs167, %rs168}, %r935;
	cvt.f32.bf16 	%r1019, %rs168;
	cvt.f32.bf16 	%r1020, %rs167;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1021, %r1018, %r578, %r1020;
	fma.rn.f32 	%r1022, %r1017, %r579, %r1019;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r664, %r1022, %r1021;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1023, %r1449, %r873;
	mul.f32 	%r1024, %r1448, %r873;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs169, %rs170}, %r936;
	cvt.f32.bf16 	%r1025, %rs170;
	cvt.f32.bf16 	%r1026, %rs169;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1027, %r1024, %r578, %r1026;
	fma.rn.f32 	%r1028, %r1023, %r579, %r1025;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r669, %r1028, %r1027;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1029, %r1479, %r874;
	mul.f32 	%r1030, %r1478, %r874;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs171, %rs172}, %r951;
	cvt.f32.bf16 	%r1031, %rs172;
	cvt.f32.bf16 	%r1032, %rs171;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1033, %r1030, %r578, %r1032;
	fma.rn.f32 	%r1034, %r1029, %r579, %r1031;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r665, %r1034, %r1033;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1035, %r1481, %r875;
	mul.f32 	%r1036, %r1480, %r875;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs173, %rs174}, %r952;
	cvt.f32.bf16 	%r1037, %rs174;
	cvt.f32.bf16 	%r1038, %rs173;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1039, %r1036, %r578, %r1038;
	fma.rn.f32 	%r1040, %r1035, %r579, %r1037;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r670, %r1040, %r1039;
	cvt.rn.bf16x2.f32 	%r666, %r1004, %r1003;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1041, %r1513, %r877;
	mul.f32 	%r1042, %r1512, %r877;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs175, %rs176}, %r968;
	cvt.f32.bf16 	%r1043, %rs176;
	cvt.f32.bf16 	%r1044, %rs175;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1045, %r1042, %r578, %r1044;
	fma.rn.f32 	%r1046, %r1041, %r579, %r1043;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r671, %r1046, %r1045;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1047, %r1515, %r876;
	mul.f32 	%r1048, %r1514, %r876;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs177, %rs178}, %r969;
	cvt.f32.bf16 	%r1049, %rs178;
	cvt.f32.bf16 	%r1050, %rs177;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1051, %r1048, %r580, %r1050;
	fma.rn.f32 	%r1052, %r1047, %r581, %r1049;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1053, %r1419, %r870;
	mul.f32 	%r1054, %r1418, %r870;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs179, %rs180}, %r919;
	cvt.f32.bf16 	%r1055, %rs180;
	cvt.f32.bf16 	%r1056, %rs179;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1057, %r1054, %r580, %r1056;
	fma.rn.f32 	%r1058, %r1053, %r581, %r1055;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r683, %r1058, %r1057;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1059, %r1421, %r871;
	mul.f32 	%r1060, %r1420, %r871;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs181, %rs182}, %r920;
	cvt.f32.bf16 	%r1061, %rs182;
	cvt.f32.bf16 	%r1062, %rs181;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1063, %r1060, %r580, %r1062;
	fma.rn.f32 	%r1064, %r1059, %r581, %r1061;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r688, %r1064, %r1063;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1065, %r1451, %r872;
	mul.f32 	%r1066, %r1450, %r872;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs183, %rs184}, %r937;
	cvt.f32.bf16 	%r1067, %rs184;
	cvt.f32.bf16 	%r1068, %rs183;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1069, %r1066, %r580, %r1068;
	fma.rn.f32 	%r1070, %r1065, %r581, %r1067;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r684, %r1070, %r1069;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1071, %r1453, %r873;
	mul.f32 	%r1072, %r1452, %r873;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs185, %rs186}, %r938;
	cvt.f32.bf16 	%r1073, %rs186;
	cvt.f32.bf16 	%r1074, %rs185;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1075, %r1072, %r580, %r1074;
	fma.rn.f32 	%r1076, %r1071, %r581, %r1073;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r689, %r1076, %r1075;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1077, %r1483, %r874;
	mul.f32 	%r1078, %r1482, %r874;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs187, %rs188}, %r953;
	cvt.f32.bf16 	%r1079, %rs188;
	cvt.f32.bf16 	%r1080, %rs187;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1081, %r1078, %r580, %r1080;
	fma.rn.f32 	%r1082, %r1077, %r581, %r1079;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r685, %r1082, %r1081;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1083, %r1485, %r875;
	mul.f32 	%r1084, %r1484, %r875;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs189, %rs190}, %r954;
	cvt.f32.bf16 	%r1085, %rs190;
	cvt.f32.bf16 	%r1086, %rs189;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1087, %r1084, %r580, %r1086;
	fma.rn.f32 	%r1088, %r1083, %r581, %r1085;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r690, %r1088, %r1087;
	cvt.rn.bf16x2.f32 	%r686, %r1052, %r1051;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1089, %r1517, %r877;
	mul.f32 	%r1090, %r1516, %r877;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs191, %rs192}, %r970;
	cvt.f32.bf16 	%r1091, %rs192;
	cvt.f32.bf16 	%r1092, %rs191;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1093, %r1090, %r580, %r1092;
	fma.rn.f32 	%r1094, %r1089, %r581, %r1091;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r691, %r1094, %r1093;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1095, %r1519, %r876;
	mul.f32 	%r1096, %r1518, %r876;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs193, %rs194}, %r975;
	cvt.f32.bf16 	%r1097, %rs194;
	cvt.f32.bf16 	%r1098, %rs193;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1099, %r1096, %r582, %r1098;
	fma.rn.f32 	%r1100, %r1095, %r583, %r1097;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1101, %r1423, %r870;
	mul.f32 	%r1102, %r1422, %r870;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs195, %rs196}, %r927;
	cvt.f32.bf16 	%r1103, %rs196;
	cvt.f32.bf16 	%r1104, %rs195;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1105, %r1102, %r582, %r1104;
	fma.rn.f32 	%r1106, %r1101, %r583, %r1103;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r703, %r1106, %r1105;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1107, %r1425, %r871;
	mul.f32 	%r1108, %r1424, %r871;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs197, %rs198}, %r928;
	cvt.f32.bf16 	%r1109, %rs198;
	cvt.f32.bf16 	%r1110, %rs197;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1111, %r1108, %r582, %r1110;
	fma.rn.f32 	%r1112, %r1107, %r583, %r1109;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r708, %r1112, %r1111;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1113, %r1455, %r872;
	mul.f32 	%r1114, %r1454, %r872;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs199, %rs200}, %r943;
	cvt.f32.bf16 	%r1115, %rs200;
	cvt.f32.bf16 	%r1116, %rs199;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1117, %r1114, %r582, %r1116;
	fma.rn.f32 	%r1118, %r1113, %r583, %r1115;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r704, %r1118, %r1117;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1119, %r1457, %r873;
	mul.f32 	%r1120, %r1456, %r873;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs201, %rs202}, %r944;
	cvt.f32.bf16 	%r1121, %rs202;
	cvt.f32.bf16 	%r1122, %rs201;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1123, %r1120, %r582, %r1122;
	fma.rn.f32 	%r1124, %r1119, %r583, %r1121;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r709, %r1124, %r1123;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1125, %r1487, %r874;
	mul.f32 	%r1126, %r1486, %r874;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs203, %rs204}, %r959;
	cvt.f32.bf16 	%r1127, %rs204;
	cvt.f32.bf16 	%r1128, %rs203;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1129, %r1126, %r582, %r1128;
	fma.rn.f32 	%r1130, %r1125, %r583, %r1127;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r705, %r1130, %r1129;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1131, %r1489, %r875;
	mul.f32 	%r1132, %r1488, %r875;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs205, %rs206}, %r960;
	cvt.f32.bf16 	%r1133, %rs206;
	cvt.f32.bf16 	%r1134, %rs205;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1135, %r1132, %r582, %r1134;
	fma.rn.f32 	%r1136, %r1131, %r583, %r1133;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r710, %r1136, %r1135;
	cvt.rn.bf16x2.f32 	%r706, %r1100, %r1099;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1137, %r1521, %r877;
	mul.f32 	%r1138, %r1520, %r877;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs207, %rs208}, %r976;
	cvt.f32.bf16 	%r1139, %rs208;
	cvt.f32.bf16 	%r1140, %rs207;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1141, %r1138, %r582, %r1140;
	fma.rn.f32 	%r1142, %r1137, %r583, %r1139;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r711, %r1142, %r1141;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1143, %r1523, %r876;
	mul.f32 	%r1144, %r1522, %r876;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs209, %rs210}, %r977;
	cvt.f32.bf16 	%r1145, %rs210;
	cvt.f32.bf16 	%r1146, %rs209;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1147, %r1144, %r584, %r1146;
	fma.rn.f32 	%r1148, %r1143, %r585, %r1145;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1149, %r1427, %r870;
	mul.f32 	%r1150, %r1426, %r870;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs211, %rs212}, %r929;
	cvt.f32.bf16 	%r1151, %rs212;
	cvt.f32.bf16 	%r1152, %rs211;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1153, %r1150, %r584, %r1152;
	fma.rn.f32 	%r1154, %r1149, %r585, %r1151;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r723, %r1154, %r1153;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1155, %r1429, %r871;
	mul.f32 	%r1156, %r1428, %r871;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs213, %rs214}, %r930;
	cvt.f32.bf16 	%r1157, %rs214;
	cvt.f32.bf16 	%r1158, %rs213;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1159, %r1156, %r584, %r1158;
	fma.rn.f32 	%r1160, %r1155, %r585, %r1157;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r728, %r1160, %r1159;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1161, %r1459, %r872;
	mul.f32 	%r1162, %r1458, %r872;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs215, %rs216}, %r945;
	cvt.f32.bf16 	%r1163, %rs216;
	cvt.f32.bf16 	%r1164, %rs215;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1165, %r1162, %r584, %r1164;
	fma.rn.f32 	%r1166, %r1161, %r585, %r1163;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r724, %r1166, %r1165;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1167, %r1461, %r873;
	mul.f32 	%r1168, %r1460, %r873;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs217, %rs218}, %r946;
	cvt.f32.bf16 	%r1169, %rs218;
	cvt.f32.bf16 	%r1170, %rs217;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1171, %r1168, %r584, %r1170;
	fma.rn.f32 	%r1172, %r1167, %r585, %r1169;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r729, %r1172, %r1171;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1173, %r1491, %r874;
	mul.f32 	%r1174, %r1490, %r874;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs219, %rs220}, %r961;
	cvt.f32.bf16 	%r1175, %rs220;
	cvt.f32.bf16 	%r1176, %rs219;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1177, %r1174, %r584, %r1176;
	fma.rn.f32 	%r1178, %r1173, %r585, %r1175;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r725, %r1178, %r1177;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1179, %r1493, %r875;
	mul.f32 	%r1180, %r1492, %r875;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs221, %rs222}, %r962;
	cvt.f32.bf16 	%r1181, %rs222;
	cvt.f32.bf16 	%r1182, %rs221;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1183, %r1180, %r584, %r1182;
	fma.rn.f32 	%r1184, %r1179, %r585, %r1181;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r730, %r1184, %r1183;
	cvt.rn.bf16x2.f32 	%r726, %r1148, %r1147;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1185, %r1525, %r877;
	mul.f32 	%r1186, %r1524, %r877;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs223, %rs224}, %r978;
	cvt.f32.bf16 	%r1187, %rs224;
	cvt.f32.bf16 	%r1188, %rs223;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1189, %r1186, %r584, %r1188;
	fma.rn.f32 	%r1190, %r1185, %r585, %r1187;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r731, %r1190, %r1189;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1191, %r1527, %r876;
	mul.f32 	%r1192, %r1526, %r876;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs225, %rs226}, %r971;
	cvt.f32.bf16 	%r1193, %rs226;
	cvt.f32.bf16 	%r1194, %rs225;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1195, %r1192, %r586, %r1194;
	fma.rn.f32 	%r1196, %r1191, %r587, %r1193;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1197, %r1431, %r870;
	mul.f32 	%r1198, %r1430, %r870;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs227, %rs228}, %r921;
	cvt.f32.bf16 	%r1199, %rs228;
	cvt.f32.bf16 	%r1200, %rs227;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1201, %r1198, %r586, %r1200;
	fma.rn.f32 	%r1202, %r1197, %r587, %r1199;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r673, %r1202, %r1201;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1203, %r1433, %r871;
	mul.f32 	%r1204, %r1432, %r871;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs229, %rs230}, %r922;
	cvt.f32.bf16 	%r1205, %rs230;
	cvt.f32.bf16 	%r1206, %rs229;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1207, %r1204, %r586, %r1206;
	fma.rn.f32 	%r1208, %r1203, %r587, %r1205;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r678, %r1208, %r1207;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1209, %r1463, %r872;
	mul.f32 	%r1210, %r1462, %r872;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs231, %rs232}, %r939;
	cvt.f32.bf16 	%r1211, %rs232;
	cvt.f32.bf16 	%r1212, %rs231;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1213, %r1210, %r586, %r1212;
	fma.rn.f32 	%r1214, %r1209, %r587, %r1211;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r674, %r1214, %r1213;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1215, %r1465, %r873;
	mul.f32 	%r1216, %r1464, %r873;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs233, %rs234}, %r940;
	cvt.f32.bf16 	%r1217, %rs234;
	cvt.f32.bf16 	%r1218, %rs233;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1219, %r1216, %r586, %r1218;
	fma.rn.f32 	%r1220, %r1215, %r587, %r1217;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r679, %r1220, %r1219;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1221, %r1495, %r874;
	mul.f32 	%r1222, %r1494, %r874;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs235, %rs236}, %r955;
	cvt.f32.bf16 	%r1223, %rs236;
	cvt.f32.bf16 	%r1224, %rs235;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1225, %r1222, %r586, %r1224;
	fma.rn.f32 	%r1226, %r1221, %r587, %r1223;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r675, %r1226, %r1225;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1227, %r1497, %r875;
	mul.f32 	%r1228, %r1496, %r875;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs237, %rs238}, %r956;
	cvt.f32.bf16 	%r1229, %rs238;
	cvt.f32.bf16 	%r1230, %rs237;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1231, %r1228, %r586, %r1230;
	fma.rn.f32 	%r1232, %r1227, %r587, %r1229;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r680, %r1232, %r1231;
	cvt.rn.bf16x2.f32 	%r676, %r1196, %r1195;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1233, %r1529, %r877;
	mul.f32 	%r1234, %r1528, %r877;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs239, %rs240}, %r972;
	cvt.f32.bf16 	%r1235, %rs240;
	cvt.f32.bf16 	%r1236, %rs239;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1237, %r1234, %r586, %r1236;
	fma.rn.f32 	%r1238, %r1233, %r587, %r1235;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r681, %r1238, %r1237;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1239, %r1531, %r876;
	mul.f32 	%r1240, %r1530, %r876;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs241, %rs242}, %r973;
	cvt.f32.bf16 	%r1241, %rs242;
	cvt.f32.bf16 	%r1242, %rs241;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1243, %r1240, %r588, %r1242;
	fma.rn.f32 	%r1244, %r1239, %r589, %r1241;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1245, %r1435, %r870;
	mul.f32 	%r1246, %r1434, %r870;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs243, %rs244}, %r923;
	cvt.f32.bf16 	%r1247, %rs244;
	cvt.f32.bf16 	%r1248, %rs243;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1249, %r1246, %r588, %r1248;
	fma.rn.f32 	%r1250, %r1245, %r589, %r1247;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r693, %r1250, %r1249;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1251, %r1437, %r871;
	mul.f32 	%r1252, %r1436, %r871;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs245, %rs246}, %r924;
	cvt.f32.bf16 	%r1253, %rs246;
	cvt.f32.bf16 	%r1254, %rs245;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1255, %r1252, %r588, %r1254;
	fma.rn.f32 	%r1256, %r1251, %r589, %r1253;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r698, %r1256, %r1255;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1257, %r1467, %r872;
	mul.f32 	%r1258, %r1466, %r872;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs247, %rs248}, %r941;
	cvt.f32.bf16 	%r1259, %rs248;
	cvt.f32.bf16 	%r1260, %rs247;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1261, %r1258, %r588, %r1260;
	fma.rn.f32 	%r1262, %r1257, %r589, %r1259;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r694, %r1262, %r1261;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1263, %r1469, %r873;
	mul.f32 	%r1264, %r1468, %r873;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs249, %rs250}, %r942;
	cvt.f32.bf16 	%r1265, %rs250;
	cvt.f32.bf16 	%r1266, %rs249;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1267, %r1264, %r588, %r1266;
	fma.rn.f32 	%r1268, %r1263, %r589, %r1265;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r699, %r1268, %r1267;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1269, %r1499, %r874;
	mul.f32 	%r1270, %r1498, %r874;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs251, %rs252}, %r957;
	cvt.f32.bf16 	%r1271, %rs252;
	cvt.f32.bf16 	%r1272, %rs251;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1273, %r1270, %r588, %r1272;
	fma.rn.f32 	%r1274, %r1269, %r589, %r1271;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r695, %r1274, %r1273;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1275, %r1501, %r875;
	mul.f32 	%r1276, %r1500, %r875;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs253, %rs254}, %r958;
	cvt.f32.bf16 	%r1277, %rs254;
	cvt.f32.bf16 	%r1278, %rs253;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1279, %r1276, %r588, %r1278;
	fma.rn.f32 	%r1280, %r1275, %r589, %r1277;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r700, %r1280, %r1279;
	cvt.rn.bf16x2.f32 	%r696, %r1244, %r1243;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1281, %r1533, %r877;
	mul.f32 	%r1282, %r1532, %r877;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs255, %rs256}, %r974;
	cvt.f32.bf16 	%r1283, %rs256;
	cvt.f32.bf16 	%r1284, %rs255;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1285, %r1282, %r588, %r1284;
	fma.rn.f32 	%r1286, %r1281, %r589, %r1283;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r701, %r1286, %r1285;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1287, %r1535, %r876;
	mul.f32 	%r1288, %r1534, %r876;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs257, %rs258}, %r979;
	cvt.f32.bf16 	%r1289, %rs258;
	cvt.f32.bf16 	%r1290, %rs257;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1291, %r1288, %r590, %r1290;
	fma.rn.f32 	%r1292, %r1287, %r591, %r1289;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1293, %r1439, %r870;
	mul.f32 	%r1294, %r1438, %r870;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs259, %rs260}, %r931;
	cvt.f32.bf16 	%r1295, %rs260;
	cvt.f32.bf16 	%r1296, %rs259;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1297, %r1294, %r590, %r1296;
	fma.rn.f32 	%r1298, %r1293, %r591, %r1295;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r713, %r1298, %r1297;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1299, %r1441, %r871;
	mul.f32 	%r1300, %r1440, %r871;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs261, %rs262}, %r932;
	cvt.f32.bf16 	%r1301, %rs262;
	cvt.f32.bf16 	%r1302, %rs261;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1303, %r1300, %r590, %r1302;
	fma.rn.f32 	%r1304, %r1299, %r591, %r1301;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r718, %r1304, %r1303;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1305, %r1471, %r872;
	mul.f32 	%r1306, %r1470, %r872;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs263, %rs264}, %r947;
	cvt.f32.bf16 	%r1307, %rs264;
	cvt.f32.bf16 	%r1308, %rs263;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1309, %r1306, %r590, %r1308;
	fma.rn.f32 	%r1310, %r1305, %r591, %r1307;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r714, %r1310, %r1309;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1311, %r1473, %r873;
	mul.f32 	%r1312, %r1472, %r873;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs265, %rs266}, %r948;
	cvt.f32.bf16 	%r1313, %rs266;
	cvt.f32.bf16 	%r1314, %rs265;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1315, %r1312, %r590, %r1314;
	fma.rn.f32 	%r1316, %r1311, %r591, %r1313;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r719, %r1316, %r1315;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1317, %r1503, %r874;
	mul.f32 	%r1318, %r1502, %r874;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs267, %rs268}, %r963;
	cvt.f32.bf16 	%r1319, %rs268;
	cvt.f32.bf16 	%r1320, %rs267;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1321, %r1318, %r590, %r1320;
	fma.rn.f32 	%r1322, %r1317, %r591, %r1319;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r715, %r1322, %r1321;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1323, %r1505, %r875;
	mul.f32 	%r1324, %r1504, %r875;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs269, %rs270}, %r964;
	cvt.f32.bf16 	%r1325, %rs270;
	cvt.f32.bf16 	%r1326, %rs269;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1327, %r1324, %r590, %r1326;
	fma.rn.f32 	%r1328, %r1323, %r591, %r1325;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r720, %r1328, %r1327;
	cvt.rn.bf16x2.f32 	%r716, %r1292, %r1291;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1329, %r1537, %r877;
	mul.f32 	%r1330, %r1536, %r877;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs271, %rs272}, %r980;
	cvt.f32.bf16 	%r1331, %rs272;
	cvt.f32.bf16 	%r1332, %rs271;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1333, %r1330, %r590, %r1332;
	fma.rn.f32 	%r1334, %r1329, %r591, %r1331;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r721, %r1334, %r1333;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1335, %r1539, %r876;
	mul.f32 	%r1336, %r1538, %r876;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs273, %rs274}, %r981;
	cvt.f32.bf16 	%r1337, %rs274;
	cvt.f32.bf16 	%r1338, %rs273;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1339, %r1336, %r592, %r1338;
	fma.rn.f32 	%r1340, %r1335, %r593, %r1337;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1341, %r1443, %r870;
	mul.f32 	%r1342, %r1442, %r870;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs275, %rs276}, %r933;
	cvt.f32.bf16 	%r1343, %rs276;
	cvt.f32.bf16 	%r1344, %rs275;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1345, %r1342, %r592, %r1344;
	fma.rn.f32 	%r1346, %r1341, %r593, %r1343;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r733, %r1346, %r1345;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1347, %r1445, %r871;
	mul.f32 	%r1348, %r1444, %r871;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs277, %rs278}, %r934;
	cvt.f32.bf16 	%r1349, %rs278;
	cvt.f32.bf16 	%r1350, %rs277;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1351, %r1348, %r592, %r1350;
	fma.rn.f32 	%r1352, %r1347, %r593, %r1349;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r738, %r1352, %r1351;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1353, %r1475, %r872;
	mul.f32 	%r1354, %r1474, %r872;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs279, %rs280}, %r949;
	cvt.f32.bf16 	%r1355, %rs280;
	cvt.f32.bf16 	%r1356, %rs279;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1357, %r1354, %r592, %r1356;
	fma.rn.f32 	%r1358, %r1353, %r593, %r1355;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r734, %r1358, %r1357;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1359, %r1477, %r873;
	mul.f32 	%r1360, %r1476, %r873;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs281, %rs282}, %r950;
	cvt.f32.bf16 	%r1361, %rs282;
	cvt.f32.bf16 	%r1362, %rs281;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1363, %r1360, %r592, %r1362;
	fma.rn.f32 	%r1364, %r1359, %r593, %r1361;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r739, %r1364, %r1363;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1365, %r1507, %r874;
	mul.f32 	%r1366, %r1506, %r874;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs283, %rs284}, %r965;
	cvt.f32.bf16 	%r1367, %rs284;
	cvt.f32.bf16 	%r1368, %rs283;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1369, %r1366, %r592, %r1368;
	fma.rn.f32 	%r1370, %r1365, %r593, %r1367;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r735, %r1370, %r1369;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1371, %r1509, %r875;
	mul.f32 	%r1372, %r1508, %r875;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs285, %rs286}, %r966;
	cvt.f32.bf16 	%r1373, %rs286;
	cvt.f32.bf16 	%r1374, %rs285;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1375, %r1372, %r592, %r1374;
	fma.rn.f32 	%r1376, %r1371, %r593, %r1373;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r740, %r1376, %r1375;
	cvt.rn.bf16x2.f32 	%r736, %r1340, %r1339;
	.loc	1 167 16                        // sk05_mlp_gateup.py:167:16
	mul.f32 	%r1377, %r1541, %r877;
	mul.f32 	%r1378, %r1540, %r877;
	.loc	1 169 99                        // sk05_mlp_gateup.py:169:99
	mov.b32 	{%rs287, %rs288}, %r982;
	cvt.f32.bf16 	%r1379, %rs288;
	cvt.f32.bf16 	%r1380, %rs287;
	.loc	1 169 11                        // sk05_mlp_gateup.py:169:11
	fma.rn.f32 	%r1381, %r1378, %r592, %r1380;
	fma.rn.f32 	%r1382, %r1377, %r593, %r1379;
	.loc	1 175 15                        // sk05_mlp_gateup.py:175:15
	cvt.rn.bf16x2.f32 	%r741, %r1382, %r1381;
	bar.sync 	0;
	shl.b32 	%r1383, %r5, 14;
	shl.b32 	%r1384, %r5, 5;
	and.b32 	%r1385, %r1412, 3456;
	bfe.s32 	%r1386, %r2, 2, 1;
	and.b32 	%r1387, %r1386, 8208;
	or.b32 	%r1388, %r1384, %r1385;
	xor.b32 	%r1389, %r1387, %r911;
	or.b32 	%r1390, %r1389, %r1388;
	or.b32 	%r1391, %r1390, %r1383;
	add.s32 	%r662, %r189, %r1391;
	// begin inline asm
	st.shared.v4.b32 [ %r662 + 0 ], { %r663, %r664, %r665, %r666 };
	// end inline asm
	add.s32 	%r667, %r662, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r667 + 0 ], { %r668, %r669, %r670, %r671 };
	// end inline asm
	add.s32 	%r672, %r662, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r672 + 0 ], { %r673, %r674, %r675, %r676 };
	// end inline asm
	add.s32 	%r677, %r662, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r677 + 0 ], { %r678, %r679, %r680, %r681 };
	// end inline asm
	xor.b32 	%r1392, %r1391, 32;
	add.s32 	%r682, %r189, %r1392;
	// begin inline asm
	st.shared.v4.b32 [ %r682 + 0 ], { %r683, %r684, %r685, %r686 };
	// end inline asm
	add.s32 	%r687, %r682, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r687 + 0 ], { %r688, %r689, %r690, %r691 };
	// end inline asm
	add.s32 	%r692, %r682, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r692 + 0 ], { %r693, %r694, %r695, %r696 };
	// end inline asm
	add.s32 	%r697, %r682, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r697 + 0 ], { %r698, %r699, %r700, %r701 };
	// end inline asm
	xor.b32 	%r1393, %r1391, 64;
	add.s32 	%r702, %r189, %r1393;
	// begin inline asm
	st.shared.v4.b32 [ %r702 + 0 ], { %r703, %r704, %r705, %r706 };
	// end inline asm
	add.s32 	%r707, %r702, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r707 + 0 ], { %r708, %r709, %r710, %r711 };
	// end inline asm
	add.s32 	%r712, %r702, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r712 + 0 ], { %r713, %r714, %r715, %r716 };
	// end inline asm
	add.s32 	%r717, %r702, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r717 + 0 ], { %r718, %r719, %r720, %r721 };
	// end inline asm
	xor.b32 	%r1394, %r1391, 96;
	add.s32 	%r722, %r189, %r1394;
	// begin inline asm
	st.shared.v4.b32 [ %r722 + 0 ], { %r723, %r724, %r725, %r726 };
	// end inline asm
	add.s32 	%r727, %r722, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r727 + 0 ], { %r728, %r729, %r730, %r731 };
	// end inline asm
	add.s32 	%r732, %r722, 4096;
	// begin inline asm
	st.shared.v4.b32 [ %r732 + 0 ], { %r733, %r734, %r735, %r736 };
	// end inline asm
	add.s32 	%r737, %r722, 4608;
	// begin inline asm
	st.shared.v4.b32 [ %r737 + 0 ], { %r738, %r739, %r740, %r741 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r1395, %r2, 2;
	and.b32 	%r1396, %r1395, 896;
	shl.b32 	%r1397, %r862, 9;
	selp.b32 	%r1398, 0, 8208, %p24;
	or.b32 	%r1399, %r904, %r1396;
	xor.b32 	%r1400, %r1399, %r1398;
	or.b32 	%r1401, %r1400, %r1397;
	add.s32 	%r1402, %r189, %r1401;
	ld.shared.v4.b32 	{%r742, %r758, %r774, %r790}, [%r1402];
	ld.shared.v4.b32 	{%r746, %r762, %r778, %r794}, [%r1402+1024];
	ld.shared.v4.b32 	{%r750, %r766, %r782, %r798}, [%r1402+2048];
	ld.shared.v4.b32 	{%r754, %r770, %r786, %r802}, [%r1402+3072];
	xor.b32 	%r1403, %r1401, 32;
	add.s32 	%r1404, %r189, %r1403;
	ld.shared.v4.b32 	{%r743, %r759, %r775, %r791}, [%r1404+16384];
	ld.shared.v4.b32 	{%r747, %r763, %r779, %r795}, [%r1404+17408];
	ld.shared.v4.b32 	{%r751, %r767, %r783, %r799}, [%r1404+18432];
	ld.shared.v4.b32 	{%r755, %r771, %r787, %r803}, [%r1404+19456];
	xor.b32 	%r1405, %r1401, 64;
	add.s32 	%r1406, %r189, %r1405;
	ld.shared.v4.b32 	{%r744, %r760, %r776, %r792}, [%r1406+32768];
	ld.shared.v4.b32 	{%r748, %r764, %r780, %r796}, [%r1406+33792];
	ld.shared.v4.b32 	{%r752, %r768, %r784, %r800}, [%r1406+34816];
	ld.shared.v4.b32 	{%r756, %r772, %r788, %r804}, [%r1406+35840];
	xor.b32 	%r1407, %r1401, 96;
	add.s32 	%r1408, %r189, %r1407;
	ld.shared.v4.b32 	{%r745, %r761, %r777, %r793}, [%r1408+49152];
	ld.shared.v4.b32 	{%r749, %r765, %r781, %r797}, [%r1408+50176];
	ld.shared.v4.b32 	{%r753, %r769, %r785, %r801}, [%r1408+51200];
	ld.shared.v4.b32 	{%r757, %r773, %r789, %r805}, [%r1408+52224];
	.loc	1 175 8                         // sk05_mlp_gateup.py:175:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd250 + 0 ], { %r742, %r743, %r744, %r745 };
	// end inline asm
	// begin inline asm
	@%p9 st.global.v4.b32 [ %rd251 + 0 ], { %r746, %r747, %r748, %r749 };
	// end inline asm
	// begin inline asm
	@%p10 st.global.v4.b32 [ %rd252 + 0 ], { %r750, %r751, %r752, %r753 };
	// end inline asm
	// begin inline asm
	@%p11 st.global.v4.b32 [ %rd253 + 0 ], { %r754, %r755, %r756, %r757 };
	// end inline asm
	// begin inline asm
	@%p12 st.global.v4.b32 [ %rd254 + 0 ], { %r758, %r759, %r760, %r761 };
	// end inline asm
	// begin inline asm
	@%p13 st.global.v4.b32 [ %rd255 + 0 ], { %r762, %r763, %r764, %r765 };
	// end inline asm
	// begin inline asm
	@%p14 st.global.v4.b32 [ %rd256 + 0 ], { %r766, %r767, %r768, %r769 };
	// end inline asm
	// begin inline asm
	@%p15 st.global.v4.b32 [ %rd257 + 0 ], { %r770, %r771, %r772, %r773 };
	// end inline asm
	// begin inline asm
	@%p16 st.global.v4.b32 [ %rd258 + 0 ], { %r774, %r775, %r776, %r777 };
	// end inline asm
	// begin inline asm
	@%p17 st.global.v4.b32 [ %rd259 + 0 ], { %r778, %r779, %r780, %r781 };
	// end inline asm
	// begin inline asm
	@%p18 st.global.v4.b32 [ %rd260 + 0 ], { %r782, %r783, %r784, %r785 };
	// end inline asm
	// begin inline asm
	@%p19 st.global.v4.b32 [ %rd261 + 0 ], { %r786, %r787, %r788, %r789 };
	// end inline asm
	// begin inline asm
	@%p20 st.global.v4.b32 [ %rd262 + 0 ], { %r790, %r791, %r792, %r793 };
	// end inline asm
	// begin inline asm
	@%p21 st.global.v4.b32 [ %rd263 + 0 ], { %r794, %r795, %r796, %r797 };
	// end inline asm
	// begin inline asm
	@%p22 st.global.v4.b32 [ %rd264 + 0 ], { %r798, %r799, %r800, %r801 };
	// end inline asm
	// begin inline asm
	@%p23 st.global.v4.b32 [ %rd265 + 0 ], { %r802, %r803, %r804, %r805 };
	// end inline asm
	.loc	1 173 4                         // sk05_mlp_gateup.py:173:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk05_mlp_gateup.py"
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
.b8 11                                  // DW_FORM_data1
.b8 87                                  // DW_AT_call_column
.b8 11                                  // DW_FORM_data1
.b8 0                                   // EOM(1)
.b8 0                                   // EOM(2)
.b8 0                                   // EOM(3)
	}
	.section	.debug_info
	{
.b32 165                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x9e DW_TAG_compile_unit
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
.b8 114
.b8 101
.b8 112
.b8 111
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
.b8 2                                   // Abbrev [2] 0x48:0x1a DW_TAG_subprogram
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
.b8 3                                   // Abbrev [3] 0x62:0x46 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 72                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x77:0x18 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 144                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x8f:0x18 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 145                                 // DW_AT_call_line
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


# --- quant: PTX embebido (E1=32 lop3, E2=0 mul) ---

_QPTX0 = r"""//
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
	.loc	1 113 0                         // sk05_mlp_gateup.py:113:0
$L__func_begin0:
	.loc	1 113 0                         // sk05_mlp_gateup.py:113:0

// %bb.0:                               // %__nv_rsqrtf.exit
	ld.param.b64 	%rd12, [_sk05_rmsnorm_quant_kernel_param_0];
	ld.param.b64 	%rd13, [_sk05_rmsnorm_quant_kernel_param_1];
$L__tmp0:
	.loc	1 119 24                        // sk05_mlp_gateup.py:119:24
	mov.u32 	%r51, %ctaid.x;
	ld.param.b64 	%rd14, [_sk05_rmsnorm_quant_kernel_param_2];
	.loc	1 120 24                        // sk05_mlp_gateup.py:120:24
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
	.loc	1 121 18                        // sk05_mlp_gateup.py:121:18
	setp.lt.s32 	%p1, %r60, %r55;
	setp.lt.s32 	%p2, %r61, %r55;
	.loc	1 122 30                        // sk05_mlp_gateup.py:122:30
	mul.lo.s32 	%r62, %r57, %r51;
	.loc	1 122 24                        // sk05_mlp_gateup.py:122:24
	mad.wide.s32 	%rd16, %r62, 2, %rd12;
	.loc	1 122 42                        // sk05_mlp_gateup.py:122:42
	cvt.u64.u32 	%rd17, %r60;
	mul.wide.u32 	%rd18, %r60, 2;
	add.s64 	%rd1, %rd16, %rd18;
	add.s64 	%rd2, %rd1, 16;
	add.s64 	%rd3, %rd1, 8192;
	add.s64 	%rd4, %rd1, 8208;
	mov.b32 	%r5, 0;
	.loc	1 122 16                        // sk05_mlp_gateup.py:122:16
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
	.loc	1 122 73                        // sk05_mlp_gateup.py:122:73
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
	.loc	1 123 24                        // sk05_mlp_gateup.py:123:24
	add.s64 	%rd5, %rd13, %rd18;
	add.s64 	%rd6, %rd5, 16;
	add.s64 	%rd7, %rd5, 8192;
	add.s64 	%rd8, %rd5, 8208;
	.loc	1 123 16                        // sk05_mlp_gateup.py:123:16
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
	.loc	1 124 32                        // sk05_mlp_gateup.py:124:32
	mul.f32 	%r95, %r66, %r66;
$L__tmp1:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:124:28 ] ]
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
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:124:28 ]
	shfl.sync.bfly.b32 	%r127, %r126, 16, 31, -1;
$L__tmp3:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:124:28 ] ]
	add.f32 	%r128, %r126, %r127;
$L__tmp4:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:124:28 ]
	shfl.sync.bfly.b32 	%r129, %r128, 8, 31, -1;
$L__tmp5:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:124:28 ] ]
	add.f32 	%r130, %r128, %r129;
$L__tmp6:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:124:28 ]
	shfl.sync.bfly.b32 	%r131, %r130, 4, 31, -1;
$L__tmp7:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:124:28 ] ]
	add.f32 	%r132, %r130, %r131;
$L__tmp8:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:124:28 ]
	shfl.sync.bfly.b32 	%r133, %r132, 2, 31, -1;
$L__tmp9:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:124:28 ] ]
	add.f32 	%r134, %r132, %r133;
$L__tmp10:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:124:28 ]
	shfl.sync.bfly.b32 	%r135, %r134, 1, 31, -1;
$L__tmp11:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:124:28 ] ]
	add.f32 	%r35, %r134, %r135;
$L__tmp12:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:124:28 ]
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
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:124:28 ] ]
	add.f32 	%r141, %r36, %r140;
$L__tmp14:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:124:28 ]
	shfl.sync.bfly.b32 	%r142, %r141, 2, 31, -1;
$L__tmp15:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:124:28 ] ]
	add.f32 	%r143, %r141, %r142;
$L__tmp16:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:124:28 ]
	shfl.sync.bfly.b32 	%r144, %r143, 1, 31, -1;
$L__tmp17:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk05_mlp_gateup.py:124:28 ] ]
	add.f32 	%r38, %r143, %r144;
$L__tmp18:
	.loc	2 293 36                        // standard.py:293:36 @[ sk05_mlp_gateup.py:124:28 ]
	and.b32 	%r145, %r52, 7;
	setp.eq.b32 	%p7, %r145, 0;
	and.pred 	%p5, %p4, %p7;
	// begin inline asm
	@%p5 st.shared.b32 [ %r37 + 0 ], %r38;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r146, [global_smem];
$L__tmp19:
	.loc	1 124 45                        // sk05_mlp_gateup.py:124:45
	cvt.rn.f32.s32 	%r147, %r55;
	div.full.f32 	%r148, %r146, %r147;
	.loc	1 124 49                        // sk05_mlp_gateup.py:124:49
	add.f32 	%r149, %r148, 0f358637BD;
	.loc	1 124 21                        // sk05_mlp_gateup.py:124:21
	rsqrt.approx.ftz.f32 	%r150, %r149;
	.loc	1 124 12                        // sk05_mlp_gateup.py:124:12
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
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:125:29 ]
	bar.sync 	0;
$L__tmp21:
	.loc	1 128 27                        // sk05_mlp_gateup.py:128:27
	mul.lo.s32 	%r183, %r59, %r51;
	.loc	1 128 21                        // sk05_mlp_gateup.py:128:21
	cvt.s64.s32 	%rd19, %r183;
	add.s64 	%rd20, %rd14, %rd19;
	.loc	1 128 39                        // sk05_mlp_gateup.py:128:39
	add.s64 	%rd9, %rd20, %rd17;
	add.s64 	%rd10, %rd9, 4096;
	.loc	1 123 55                        // sk05_mlp_gateup.py:123:55
	mov.b32 	{%rs33, %rs34}, %r24;
	cvt.f32.bf16 	%r184, %rs33;
	cvt.f32.bf16 	%r185, %rs34;
	mov.b32 	{%rs35, %rs36}, %r25;
	cvt.f32.bf16 	%r186, %rs35;
	cvt.f32.bf16 	%r187, %rs36;
	.loc	1 124 56                        // sk05_mlp_gateup.py:124:56
	mul.f32 	%r188, %r166, %r187;
	mul.f32 	%r189, %r165, %r186;
	mul.f32 	%r190, %r164, %r185;
	mul.f32 	%r191, %r163, %r184;
	.loc	1 125 36                        // sk05_mlp_gateup.py:125:36
	abs.f32 	%r192, %r191;
	abs.f32 	%r193, %r190;
	abs.f32 	%r194, %r189;
	abs.f32 	%r195, %r188;
	.loc	1 123 55                        // sk05_mlp_gateup.py:123:55
	mov.b32 	{%rs37, %rs38}, %r22;
	cvt.f32.bf16 	%r196, %rs37;
	cvt.f32.bf16 	%r197, %rs38;
	mov.b32 	{%rs39, %rs40}, %r23;
	cvt.f32.bf16 	%r198, %rs39;
	cvt.f32.bf16 	%r199, %rs40;
	.loc	1 124 56                        // sk05_mlp_gateup.py:124:56
	mul.f32 	%r200, %r162, %r199;
	mul.f32 	%r201, %r161, %r198;
	mul.f32 	%r202, %r160, %r197;
	mul.f32 	%r203, %r159, %r196;
	.loc	1 125 36                        // sk05_mlp_gateup.py:125:36
	abs.f32 	%r204, %r203;
	abs.f32 	%r205, %r202;
	abs.f32 	%r206, %r201;
	abs.f32 	%r207, %r200;
	.loc	1 123 55                        // sk05_mlp_gateup.py:123:55
	mov.b32 	{%rs41, %rs42}, %r20;
	cvt.f32.bf16 	%r208, %rs41;
	cvt.f32.bf16 	%r209, %rs42;
	mov.b32 	{%rs43, %rs44}, %r21;
	cvt.f32.bf16 	%r210, %rs43;
	cvt.f32.bf16 	%r211, %rs44;
	.loc	1 124 56                        // sk05_mlp_gateup.py:124:56
	mul.f32 	%r212, %r158, %r211;
	mul.f32 	%r213, %r157, %r210;
	mul.f32 	%r214, %r156, %r209;
	mul.f32 	%r215, %r155, %r208;
	.loc	1 125 36                        // sk05_mlp_gateup.py:125:36
	abs.f32 	%r216, %r215;
	abs.f32 	%r217, %r214;
	abs.f32 	%r218, %r213;
	abs.f32 	%r219, %r212;
	.loc	1 123 55                        // sk05_mlp_gateup.py:123:55
	mov.b32 	{%rs45, %rs46}, %r18;
	cvt.f32.bf16 	%r220, %rs45;
	cvt.f32.bf16 	%r221, %rs46;
	mov.b32 	{%rs47, %rs48}, %r19;
	cvt.f32.bf16 	%r222, %rs47;
	cvt.f32.bf16 	%r223, %rs48;
	.loc	1 124 56                        // sk05_mlp_gateup.py:124:56
	mul.f32 	%r224, %r154, %r223;
	mul.f32 	%r225, %r153, %r222;
	mul.f32 	%r226, %r152, %r221;
	mul.f32 	%r227, %r151, %r220;
	.loc	1 125 36                        // sk05_mlp_gateup.py:125:36
	abs.f32 	%r228, %r227;
	abs.f32 	%r229, %r226;
	abs.f32 	%r230, %r225;
	abs.f32 	%r231, %r224;
$L__tmp22:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:125:29 ] ]
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
	.loc	1 123 55                        // sk05_mlp_gateup.py:123:55
	mov.b32 	{%rs49, %rs50}, %r32;
	cvt.f32.bf16 	%r247, %rs49;
	cvt.f32.bf16 	%r248, %rs50;
	mov.b32 	{%rs51, %rs52}, %r33;
	cvt.f32.bf16 	%r249, %rs51;
	cvt.f32.bf16 	%r250, %rs52;
	.loc	1 124 56                        // sk05_mlp_gateup.py:124:56
	mul.f32 	%r251, %r182, %r250;
	mul.f32 	%r252, %r181, %r249;
	mul.f32 	%r253, %r180, %r248;
	mul.f32 	%r254, %r179, %r247;
	.loc	1 125 36                        // sk05_mlp_gateup.py:125:36
	abs.f32 	%r255, %r254;
	abs.f32 	%r256, %r253;
	abs.f32 	%r257, %r252;
	abs.f32 	%r258, %r251;
	.loc	1 123 55                        // sk05_mlp_gateup.py:123:55
	mov.b32 	{%rs53, %rs54}, %r30;
	cvt.f32.bf16 	%r259, %rs53;
	cvt.f32.bf16 	%r260, %rs54;
	mov.b32 	{%rs55, %rs56}, %r31;
	cvt.f32.bf16 	%r261, %rs55;
	cvt.f32.bf16 	%r262, %rs56;
	.loc	1 124 56                        // sk05_mlp_gateup.py:124:56
	mul.f32 	%r263, %r178, %r262;
	mul.f32 	%r264, %r177, %r261;
	mul.f32 	%r265, %r176, %r260;
	mul.f32 	%r266, %r175, %r259;
	.loc	1 125 36                        // sk05_mlp_gateup.py:125:36
	abs.f32 	%r267, %r266;
	abs.f32 	%r268, %r265;
	abs.f32 	%r269, %r264;
	abs.f32 	%r270, %r263;
	.loc	1 123 55                        // sk05_mlp_gateup.py:123:55
	mov.b32 	{%rs57, %rs58}, %r28;
	cvt.f32.bf16 	%r271, %rs57;
	cvt.f32.bf16 	%r272, %rs58;
	mov.b32 	{%rs59, %rs60}, %r29;
	cvt.f32.bf16 	%r273, %rs59;
	cvt.f32.bf16 	%r274, %rs60;
	.loc	1 124 56                        // sk05_mlp_gateup.py:124:56
	mul.f32 	%r275, %r174, %r274;
	mul.f32 	%r276, %r173, %r273;
	mul.f32 	%r277, %r172, %r272;
	mul.f32 	%r278, %r171, %r271;
	.loc	1 125 36                        // sk05_mlp_gateup.py:125:36
	abs.f32 	%r279, %r278;
	abs.f32 	%r280, %r277;
	abs.f32 	%r281, %r276;
	abs.f32 	%r282, %r275;
	.loc	1 123 55                        // sk05_mlp_gateup.py:123:55
	mov.b32 	{%rs61, %rs62}, %r26;
	cvt.f32.bf16 	%r283, %rs61;
	cvt.f32.bf16 	%r284, %rs62;
	mov.b32 	{%rs63, %rs64}, %r27;
	cvt.f32.bf16 	%r285, %rs63;
	cvt.f32.bf16 	%r286, %rs64;
	.loc	1 124 56                        // sk05_mlp_gateup.py:124:56
	mul.f32 	%r287, %r170, %r286;
	mul.f32 	%r288, %r169, %r285;
	mul.f32 	%r289, %r168, %r284;
	mul.f32 	%r290, %r167, %r283;
	.loc	1 125 36                        // sk05_mlp_gateup.py:125:36
	abs.f32 	%r291, %r290;
	abs.f32 	%r292, %r289;
	abs.f32 	%r293, %r288;
	abs.f32 	%r294, %r287;
$L__tmp24:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:125:29 ] ]
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
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:125:29 ]
	shfl.sync.bfly.b32 	%r311, %r310, 16, 31, -1;
$L__tmp26:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:125:29 ] ]
	max.f32 	%r312, %r310, %r311;
$L__tmp27:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:125:29 ]
	shfl.sync.bfly.b32 	%r313, %r312, 8, 31, -1;
$L__tmp28:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:125:29 ] ]
	max.f32 	%r314, %r312, %r313;
$L__tmp29:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:125:29 ]
	shfl.sync.bfly.b32 	%r315, %r314, 4, 31, -1;
$L__tmp30:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:125:29 ] ]
	max.f32 	%r316, %r314, %r315;
$L__tmp31:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:125:29 ]
	shfl.sync.bfly.b32 	%r317, %r316, 2, 31, -1;
$L__tmp32:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:125:29 ] ]
	max.f32 	%r318, %r316, %r317;
$L__tmp33:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:125:29 ]
	shfl.sync.bfly.b32 	%r319, %r318, 1, 31, -1;
$L__tmp34:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:125:29 ] ]
	max.f32 	%r39, %r318, %r319;
$L__tmp35:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:125:29 ]
	// begin inline asm
	@%p3 st.shared.b32 [ %r34 + 0 ], %r39;
	// end inline asm
	bar.sync 	0;
	// begin inline asm
	@%p4 ld.shared.b32 %r40, [ %r37 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r320, %r40, 4, 31, -1;
$L__tmp36:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:125:29 ] ]
	max.f32 	%r321, %r40, %r320;
$L__tmp37:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:125:29 ]
	shfl.sync.bfly.b32 	%r322, %r321, 2, 31, -1;
$L__tmp38:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:125:29 ] ]
	max.f32 	%r323, %r321, %r322;
$L__tmp39:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:125:29 ]
	shfl.sync.bfly.b32 	%r324, %r323, 1, 31, -1;
$L__tmp40:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk05_mlp_gateup.py:125:29 ] ]
	max.f32 	%r41, %r323, %r324;
$L__tmp41:
	.loc	2 191 40                        // standard.py:191:40 @[ sk05_mlp_gateup.py:125:29 ]
	// begin inline asm
	@%p5 st.shared.b32 [ %r37 + 0 ], %r41;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r325, [global_smem];
$L__tmp42:
	.loc	1 125 49                        // sk05_mlp_gateup.py:125:49
	max.f32 	%r326, %r325, 0f0DA24260;
	mov.b32 	%r327, 0f42FE0000;
	.loc	1 126 22                        // sk05_mlp_gateup.py:126:22
	div.full.f32 	%r328, %r327, %r326;
	.loc	1 126 14                        // sk05_mlp_gateup.py:126:14
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
	.loc	1 127 29                        // sk05_mlp_gateup.py:127:29
	.loc	1 127 39                        // sk05_mlp_gateup.py:127:39
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
	.loc	1 127 14                        // sk05_mlp_gateup.py:127:14
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
	.loc	1 127 49                        // sk05_mlp_gateup.py:127:49
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
	.loc	1 128 70                        // sk05_mlp_gateup.py:128:70
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
	.loc	1 128 77                        // sk05_mlp_gateup.py:128:77
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
	.loc	1 128 85                        // sk05_mlp_gateup.py:128:85
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
	.loc	1 128 45                        // sk05_mlp_gateup.py:128:45
	// begin inline asm
	@%p1 st.global.v4.b32 [ %rd9 + 0 ], { %r42, %r43, %r44, %r45 };
	// end inline asm
	// begin inline asm
	@%p2 st.global.v4.b32 [ %rd10 + 0 ], { %r46, %r47, %r48, %r49 };
	// end inline asm
	.loc	1 129 21                        // sk05_mlp_gateup.py:129:21
	mad.wide.u32 	%rd11, %r51, 4, %rd15;
	.loc	1 129 34                        // sk05_mlp_gateup.py:129:34
	mul.f32 	%r50, %r326, 0f3C010204;
	.loc	1 129 26                        // sk05_mlp_gateup.py:129:26
	or.b32 	%r537, %r54, %r56;
	setp.eq.b32 	%p6, %r537, 0;
	// begin inline asm
	@%p6 st.global.b32 [ %rd11 + 0 ], { %r50 };
	// end inline asm
	.loc	1 129 4                         // sk05_mlp_gateup.py:129:4
	ret;
$L__tmp43:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk05_mlp_gateup.py"
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
.b8 11                                  // DW_FORM_data1
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
.b32 219                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0xd4 DW_TAG_compile_unit
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
.b8 114
.b8 101
.b8 112
.b8 111
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
.b8 2                                   // Abbrev [2] 0x48:0x1d DW_TAG_subprogram
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
.b8 3                                   // Abbrev [3] 0x65:0x79 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 72                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x7a:0x32 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp19                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 124                                 // DW_AT_call_line
.b8 28                                  // DW_AT_call_column
.b8 5                                   // Abbrev [5] 0x92:0x19 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp18                          // DW_AT_high_pc
.b8 2                                   // DW_AT_call_file
.b8 37                                  // DW_AT_call_line
.b8 1
.b8 36                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 4                                   // Abbrev [4] 0xac:0x31 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
.b64 $L__tmp20                          // DW_AT_low_pc
.b64 $L__tmp42                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 125                                 // DW_AT_call_line
.b8 29                                  // DW_AT_call_column
.b8 6                                   // Abbrev [6] 0xc4:0x18 DW_TAG_inlined_subroutine
.b32 72                                 // DW_AT_abstract_origin
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

_QVAR0 = _Nativo(
    "sk05_mlp_gateup/_sk05_rmsnorm_quant_kernel",
    _QPTX0, "_sk05_rmsnorm_quant_kernel",
    warps=8, shared=32,
    abi=[0, 1, 2, 3, 4, 5, 6],
    horneado={7: 8192, 8: 1e-06},
    div16=[4, 5, 6],
)


def _q0_impl(grid: list[int], x_ptr: torch.Tensor, w_ptr: torch.Tensor, q_ptr: torch.Tensor, s_ptr: torch.Tensor, K: int, stride_xm: int, stride_qm: int, BLOCK: int, EPS: float) -> None:
    """Cuerpo del custom op: lanza el PTX embebido."""
    _QVAR0(tuple(grid), x_ptr, w_ptr, q_ptr, s_ptr, K, stride_xm, stride_qm, BLOCK, EPS)


def _q0_fake(grid: list[int], x_ptr: torch.Tensor, w_ptr: torch.Tensor, q_ptr: torch.Tensor, s_ptr: torch.Tensor, K: int, stride_xm: int, stride_qm: int, BLOCK: int, EPS: float) -> None:
    """Meta impl: no toca la GPU; las salidas se mutan in-place."""
    return None


_q0_registrado = False


def _q0_registrar() -> None:
    """Registra el op una sola vez, fuera de la region compilada."""
    global _q0_registrado
    if _q0_registrado:
        return
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op(
        op_name="genesis_sk05_mlp_gateup_q0",
        op_func=_q0_impl,
        mutates_args=['q_ptr', 's_ptr'],
        fake_impl=_q0_fake,
    )
    _q0_registrado = True


def _lanzar_quant0(grid, *args):
    """PTX embebido via custom op; cae al Triton con el kill-switch.

    ``GENESIS_PTQ_NATIVO=0`` vuelve al kernel Triton, que queda en este
    archivo como referencia para los tests.
    """
    if habilitado():
        _q0_registrar()
        g = [grid] if isinstance(grid, int) else list(grid)
        return torch.ops.vllm.genesis_sk05_mlp_gateup_q0(g, *args)
    ce = ['BLOCK', 'EPS']
    nom = ['x_ptr', 'w_ptr', 'q_ptr', 's_ptr', 'K', 'stride_xm', 'stride_qm', 'BLOCK', 'EPS']
    return _sk05_rmsnorm_quant_kernel[grid](*args[:len(nom) - len(ce)],
                    **{n: args[nom.index(n)] for n in ce},
                    num_warps=8, num_stages=1)
