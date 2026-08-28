# SPDX-License-Identifier: Apache-2.0
"""SK-03 — FA_QKV_FUSED_INT8_DIADIC — capa completa Full-Attention QKV en GPU.

Capa completa (Qwen3.5-27B, 16 capas Full, TP=2):
    hidden bf16 [M, 5120]
      -> RMSNorm(input_layernorm) + quant per-token amax/127  (kernel 1)
      -> GEMM INT8 diádico Q|gate|K|V fusionado               (kernel 2)
      -> split Q/gate/K/V (views, coste cero)

Geometría: N global 14336, per-rank TP=2 N=7168 = Q 3072 | gate 3072 | K 512 |
V 512. K=5120. Todo múltiplo de 128.

Diseño (sm_86, mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32):
  * RMSNorm y quant salen del GEMM y viven en su propio kernel de una pasada.
    La versión anterior recalculaba ``sum_sq`` y ``amax`` **dentro** de cada
    programa ``pid_n``: con N=7168 y BLOCK_N=64 eran 112 relecturas completas
    de la activación, más otra pasada para cuantizar. Ahora la fila entra una
    sola vez a registros y se reusa para varianza, amax y quant.
  * ``BLOCK_K == SHIFT_BLOCK == 128`` -> un ``tl.dot`` por bloque diádico.
  * Shift por columna como ``acc += int_acc.to(f32) * exp2(shift)``.
  * Acumulador **fp32**, no bf16: la versión anterior acumulaba 40 parciales en
    bf16 (8 bits de mantisa).
  * Escalas fuera del bucle k, punteros que avanzan, grid 1-D con swizzle L2.

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



SK03_N_GLOBAL: int = 14336
SK03_N_PER_RANK: int = 7168
SK03_K: int = 5120
SK03_Q_PER_RANK: int = 3072
SK03_GATE_PER_RANK: int = 3072
SK03_K_PER_RANK: int = 512
SK03_V_PER_RANK: int = 512
SK03_Q_HEADS: int = 24
SK03_KV_HEADS: int = 4
SK03_HEAD_DIM: int = 256
SK03_HIDDEN: int = 5120
SK03_NUM_LAYERS: int = 16

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
    """Escalar cero por device, usado como epílogo neutro con strides 0."""
    z = _ZERO.get(device)
    if z is None:
        z = torch.zeros((), dtype=torch.bfloat16, device=device)
        _ZERO[device] = z
    return z


_HAS_SHIFT: dict[int, bool] = {}


def _has_shift(shifts: torch.Tensor) -> bool:
    """True si el tensor de shifts tiene algun valor distinto de cero.

    Cacheado por ``data_ptr``: el tensor se construye una vez al cargar y no
    cambia. Sin el cache habria que hacer ``shifts.any().item()`` por forward,
    que es una sincronizacion GPU->CPU — a 60 us de GEMM en decode eso cuesta
    mas que el kernel entero (medido: 0.058 -> 0.084 ms, 0.66x).
    """
    k = shifts.data_ptr()
    v = _HAS_SHIFT.get(k)
    if v is None:
        v = bool(shifts.any().item())
        _HAS_SHIFT[k] = v
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
def _sk03_rmsnorm_quant_kernel(
    x_ptr, w_ptr, q_ptr, s_ptr,
    K, stride_xm, stride_qm,
    BLOCK: tl.constexpr, EPS: tl.constexpr,
):
    """RMSNorm + quant per-token en una pasada, fila por programa.

    La fila entra una sola vez a registros y se reusa para varianza, amax y
    cuantización: cero relecturas de memoria global.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < K
    x = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * tl.rsqrt(tl.sum(x * x, axis=0) / K + EPS) * w
    amax = tl.maximum(tl.max(tl.abs(y), axis=0), 1e-30)
    inv = 127.0 / amax
    yq = y * inv
    q = (yq + tl.where(yq >= 0.0, 0.5, -0.5)).to(tl.int32)
    q = tl.minimum(tl.maximum(q, -127), 127)
    tl.store(q_ptr + row * stride_qm + offs, q.to(tl.int8), mask=mask)
    tl.store(s_ptr + row, amax * (1.0 / 127.0))



@triton.jit
def _sk03_fa_qkv_kernel(
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
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for kb in range(0, K // BLOCK_K):
        d = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), out_dtype=tl.int32)
        if HAS_SHIFT:
            d = d << tl.load(sh_ptrs + kb * stride_shift_k).to(tl.int32)[None, :]
        acc += d
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    out = acc.to(tl.bfloat16) * tl.load(a_scale_ptr + offs_m).to(tl.bfloat16)[:, None]
    out = out * tl.load(b_scale_ptr + offs_n).to(tl.bfloat16)[None, :]
    out += tl.load(resid_ptr + offs_m[:, None] * stride_res_m + offs_n[None, :] * stride_res_n).to(tl.bfloat16)

    out_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(
        out_ptr + out_m[:, None] * stride_out_m + out_n[None, :] * stride_out_n,
        out.to(out_ptr.dtype.element_ty),
        mask=(out_m[:, None] < M) & (out_n[None, :] < N),
    )


def sk03_fa_qkv_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    shifts: torch.Tensor,
    residual: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """GEMM INT8 diádico QKV+gate fusionado. ``a`` int8 [M,K], ``b`` int8 [K,N]."""
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


def sk03_rmsnorm_quant(
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


def sk03_fa_qkv_forward(
    hidden: torch.Tensor,
    ln_weight: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_scales: torch.Tensor,
    qkv_shifts: torch.Tensor,
    eps: float = 1e-6,
    out_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Capa completa FA qkv: RMSNorm+quant -> GEMM diádico -> split Q/gate/K/V.

    Split per-rank sobre ``N=7168``: ``Q=3072``, ``gate=3072`` (attn_output_gate
    de Qwen3.5), ``K=512``, ``V=512`` (4 KV heads x 128 por rank).
    """
    a, a_scales = sk03_rmsnorm_quant(hidden, ln_weight, eps)
    qkv = sk03_fa_qkv_gemm(a, qkv_weight, a_scales, qkv_scales, qkv_shifts, None, out_dtype)
    n = qkv.shape[1]
    q_n = (n * 3) // 7
    kv_n = n // 14
    o = q_n
    q = qkv[:, 0:q_n]
    gate = qkv[:, o:o + q_n]
    o += q_n
    k_t = qkv[:, o:o + kv_n]
    v_t = qkv[:, o + kv_n:o + 2 * kv_n]
    return q, gate, k_t, v_t, qkv


FA_QKV_FUSED_INT8_DIADIC = sk03_fa_qkv_forward
fa_qkv_fused = sk03_fa_qkv_forward
qkv_forward = sk03_fa_qkv_forward

__all__ = [
    "SK03_N_GLOBAL", "SK03_N_PER_RANK", "SK03_K", "SK03_Q_PER_RANK",
    "SK03_GATE_PER_RANK", "SK03_K_PER_RANK", "SK03_V_PER_RANK",
    "SHIFT_BLOCK", "BLOCK_M", "BLOCK_N", "BLOCK_K",
    "sk03_rmsnorm_quant", "sk03_fa_qkv_gemm", "sk03_fa_qkv_forward",
    "FA_QKV_FUSED_INT8_DIADIC", "fa_qkv_fused", "qkv_forward",
]


# --- variantes PTX embebidas ---

_PTX_0 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk03_fa_qkv_kernel     // -- Begin function _sk03_fa_qkv_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk03_fa_qkv_kernel
.visible .entry _sk03_fa_qkv_kernel(
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_6,
	.param .u32 _sk03_fa_qkv_kernel_param_7,
	.param .u32 _sk03_fa_qkv_kernel_param_8,
	.param .u32 _sk03_fa_qkv_kernel_param_9,
	.param .u32 _sk03_fa_qkv_kernel_param_10,
	.param .u32 _sk03_fa_qkv_kernel_param_11,
	.param .u32 _sk03_fa_qkv_kernel_param_12,
	.param .u32 _sk03_fa_qkv_kernel_param_13,
	.param .u32 _sk03_fa_qkv_kernel_param_14,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_16
)
.reqntid 256
{
	.reg .pred 	%p<11>;
	.reg .b16 	%rs<40>;
	.reg .b32 	%r<254>;
	.reg .b64 	%rd<55>;
	.loc	1 145 0                         // sk03_fa_qkv.py:145:0
$L__func_begin0:
	.loc	1 145 0                         // sk03_fa_qkv.py:145:0

// %bb.0:
	ld.param.b32 	%r23, [_sk03_fa_qkv_kernel_param_13];
	ld.param.b32 	%r22, [_sk03_fa_qkv_kernel_param_12];
	ld.param.b32 	%r21, [_sk03_fa_qkv_kernel_param_8];
	ld.param.b32 	%r20, [_sk03_fa_qkv_kernel_param_7];
	ld.param.b64 	%rd18, [_sk03_fa_qkv_kernel_param_5];
	ld.param.b64 	%rd17, [_sk03_fa_qkv_kernel_param_4];
	ld.param.b64 	%rd16, [_sk03_fa_qkv_kernel_param_3];
	ld.param.b64 	%rd15, [_sk03_fa_qkv_kernel_param_2];
	ld.param.b64 	%rd14, [_sk03_fa_qkv_kernel_param_1];
	ld.param.b64 	%rd13, [_sk03_fa_qkv_kernel_param_0];
$L__tmp0:
	.loc	1 154 24                        // sk03_fa_qkv.py:154:24
	mov.u32 	%r38, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk03_fa_qkv.py:155:27 ]
	add.s32 	%r39, %r20, 15;
	.loc	2 43 30                         // standard.py:43:30 @[ sk03_fa_qkv.py:155:27 ]
	shr.s32 	%r40, %r39, 31;
	shr.u32 	%r41, %r40, 28;
	add.s32 	%r42, %r39, %r41;
	shr.s32 	%r43, %r42, 4;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk03_fa_qkv.py:156:27 ]
	add.s32 	%r44, %r21, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk03_fa_qkv.py:156:27 ]
	shr.s32 	%r45, %r44, 31;
	shr.u32 	%r46, %r45, 25;
	add.s32 	%r47, %r44, %r46;
	shr.s32 	%r48, %r47, 7;
$L__tmp3:
	.loc	1 157 29                        // sk03_fa_qkv.py:157:29
	shl.b32 	%r49, %r48, 3;
	.loc	1 158 22                        // sk03_fa_qkv.py:158:22
	div.s32 	%r50, %r38, %r49;
	.loc	1 158 38                        // sk03_fa_qkv.py:158:38
	shl.b32 	%r51, %r50, 3;
	ld.param.b32 	%r52, [_sk03_fa_qkv_kernel_param_9];
	.loc	1 159 30                        // sk03_fa_qkv.py:159:30
	sub.s32 	%r53, %r43, %r51;
	ld.param.b32 	%r54, [_sk03_fa_qkv_kernel_param_10];
	.loc	1 159 39                        // sk03_fa_qkv.py:159:39
	min.s32 	%r55, %r53, 8;
	ld.param.b32 	%r56, [_sk03_fa_qkv_kernel_param_11];
	.loc	1 160 30                        // sk03_fa_qkv.py:160:30
	mul.lo.s32 	%r57, %r50, %r49;
	sub.s32 	%r58, %r38, %r57;
	.loc	1 161 36                        // sk03_fa_qkv.py:161:36
	div.s32 	%r59, %r58, %r55;
	.loc	1 160 46                        // sk03_fa_qkv.py:160:46
	mul.lo.s32 	%r60, %r59, %r55;
	sub.s32 	%r61, %r58, %r60;
	.loc	1 160 23                        // sk03_fa_qkv.py:160:23
	add.s32 	%r62, %r61, %r51;
	.loc	1 163 22                        // sk03_fa_qkv.py:163:22
	shl.b32 	%r1, %r62, 4;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	mov.u32 	%r2, %tid.x;
	and.b32 	%r3, %r2, 240;
	bfe.u32 	%r63, %r2, 4, 4;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r4, %r1, %r63;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r5, %r4, %r20;
	.loc	1 164 22                        // sk03_fa_qkv.py:164:22
	shl.b32 	%r6, %r59, 7;
	.loc	1 164 45                        // sk03_fa_qkv.py:164:45
	shr.u32 	%r64, %r2, 3;
	bfe.u32 	%r65, %r2, 3, 5;
	and.b32 	%r7, %r2, 224;
	and.b32 	%r8, %r2, 15;
	shl.b32 	%r66, %r8, 3;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r67, %r6, %r65;
	or.b32 	%r68, %r67, 32;
	or.b32 	%r69, %r67, 64;
	or.b32 	%r70, %r64, %r6;
	or.b32 	%r71, %r70, 96;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r72, %r67, %r21;
	rem.s32 	%r73, %r68, %r21;
	rem.s32 	%r74, %r69, %r21;
	rem.s32 	%r75, %r71, %r21;
	.loc	1 167 39                        // sk03_fa_qkv.py:167:39
	mul.lo.s32 	%r76, %r5, %r54;
	.loc	1 167 21                        // sk03_fa_qkv.py:167:21
	cvt.s64.s32 	%rd1, %r76;
	add.s64 	%rd29, %rd13, %rd1;
	.loc	1 167 51                        // sk03_fa_qkv.py:167:51
	cvt.u64.u32 	%rd2, %r66;
	add.s64 	%rd19, %rd29, %rd2;
	.loc	1 168 28                        // sk03_fa_qkv.py:168:28
	and.b32 	%r9, %r2, 7;
	shl.b32 	%r77, %r9, 4;
	.loc	1 168 21                        // sk03_fa_qkv.py:168:21
	cvt.u64.u32 	%rd3, %r77;
	add.s64 	%rd30, %rd14, %rd3;
	.loc	1 168 69                        // sk03_fa_qkv.py:168:69
	mul.lo.s32 	%r78, %r72, %r56;
	mul.lo.s32 	%r79, %r73, %r56;
	mul.lo.s32 	%r80, %r74, %r56;
	mul.lo.s32 	%r81, %r75, %r56;
	.loc	1 168 51                        // sk03_fa_qkv.py:168:51
	cvt.s64.s32 	%rd4, %r78;
	add.s64 	%rd20, %rd30, %rd4;
	cvt.s64.s32 	%rd5, %r79;
	add.s64 	%rd21, %rd30, %rd5;
	cvt.s64.s32 	%rd6, %r80;
	add.s64 	%rd22, %rd30, %rd6;
	cvt.s64.s32 	%rd7, %r81;
	add.s64 	%rd23, %rd30, %rd7;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	setp.lt.s32 	%p1, %r52, 128;
	setp.gt.s32 	%p2, %r52, 127;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
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
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
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
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	setp.gt.s32 	%p3, %r52, 255;
	.loc	1 183 18                        // sk03_fa_qkv.py:183:18
	add.s64 	%rd24, %rd19, 128;
	.loc	1 184 18                        // sk03_fa_qkv.py:184:18
	add.s64 	%rd25, %rd20, 128;
	add.s64 	%rd26, %rd21, 128;
	add.s64 	%rd27, %rd22, 128;
	add.s64 	%rd28, %rd23, 128;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
	bar.sync 	0;
	add.s32 	%r31, %r11, 34816;
	selp.b32 	%r32, 8, 0, %p3;
	// begin inline asm
	cp.async.ca.shared.global [ %r31 + 0 ], [ %rd24 + 0 ], 0x8, %r32;
	// end inline asm
	cp.async.commit_group;
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
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
	mov.b16 	%rs32, 0x0000;
	cvt.u32.u64 	%r242, %rd3;
	mov.b16 	%rs33, %rs32;
	mov.b16 	%rs34, %rs32;
	mov.b16 	%rs35, %rs32;
	mov.b16 	%rs36, %rs32;
	mov.b16 	%rs37, %rs32;
	mov.b16 	%rs38, %rs32;
	mov.b16 	%rs39, %rs32;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	@%p1 bra 	$L__BB0_4;
// %bb.1:                               // %.lr.ph
	.loc	1 0 23                          // sk03_fa_qkv.py:0:23
	shr.s32 	%r82, %r52, 31;
	shr.u32 	%r83, %r82, 25;
	add.s32 	%r84, %r52, %r83;
	shr.s32 	%r10, %r84, 7;
	add.s32 	%r13, %r10, -2;
	shl.b32 	%r93, %r8, 7;
	and.b32 	%r94, %r2, 16;
	xor.b32 	%r95, %r242, %r94;
	or.b32 	%r14, %r95, %r93;
	xor.b32 	%r15, %r14, 32;
	xor.b32 	%r16, %r14, 64;
	xor.b32 	%r17, %r14, 96;
	shl.b32 	%r96, %r9, 7;
	shl.b32 	%r97, %r7, 5;
	and.b32 	%r98, %r12, 48;
	or.b32 	%r99, %r96, %r97;
	xor.b32 	%r100, %r242, %r98;
	or.b32 	%r18, %r99, %r100;
	xor.b32 	%r19, %r18, 64;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
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
	mov.b32 	%r245, 0;
	mov.b32 	%r244, 1;
	mov.b32 	%r243, -1;
	mov.b64 	%rd54, 0;
	mov.b32 	%r246, %r245;
	mov.b32 	%r247, %r245;
	mov.b32 	%r248, %r245;
	mov.b32 	%r249, %r245;
	mov.b32 	%r250, %r245;
	mov.b32 	%r251, %r245;
	mov.b32 	%r252, %r245;
	mov.b32 	%r253, %r245;
$L__BB0_2:                              // =>This Inner Loop Header: Depth=1
	setp.lt.s32 	%p4, %r253, %r13;
	add.s32 	%r140, %r243, 1;
	setp.gt.s32 	%p5, %r140, 1;
	selp.b32 	%r243, 0, %r140, %p5;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
	cp.async.wait_group 	2;
	bar.sync 	0;
	shl.b32 	%r141, %r243, 11;
	add.s32 	%r142, %r89, %r141;
	add.s32 	%r143, %r142, %r14;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r101, %r102, %r103, %r104}, [%r143+32768];
	add.s32 	%r144, %r142, %r15;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r109, %r110, %r111, %r112}, [%r144+32768];
	add.s32 	%r145, %r142, %r16;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r117, %r118, %r119, %r120}, [%r145+32768];
	add.s32 	%r146, %r142, %r17;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r125, %r126, %r127, %r128}, [%r146+32768];
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	shl.b32 	%r147, %r243, 14;
	add.s32 	%r148, %r89, %r147;
	add.s32 	%r149, %r148, %r18;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r105, %r106, %r113, %r114}, [%r149];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r107, %r108, %r115, %r116}, [%r149+8192];
	add.s32 	%r150, %r148, %r19;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r121, %r122, %r129, %r130}, [%r150];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r123, %r124, %r131, %r132}, [%r150+8192];
	.loc	1 179 36                        // sk03_fa_qkv.py:179:36
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r245, %r246, %r247, %r248 }, { %r101, %r102, %r103, %r104 }, { %r105, %r106 }, { %r245, %r246, %r247, %r248 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r249, %r250, %r251, %r252 }, { %r101, %r102, %r103, %r104 }, { %r107, %r108 }, { %r249, %r250, %r251, %r252 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r245, %r246, %r247, %r248 }, { %r109, %r110, %r111, %r112 }, { %r113, %r114 }, { %r245, %r246, %r247, %r248 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r249, %r250, %r251, %r252 }, { %r109, %r110, %r111, %r112 }, { %r115, %r116 }, { %r249, %r250, %r251, %r252 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r245, %r246, %r247, %r248 }, { %r117, %r118, %r119, %r120 }, { %r121, %r122 }, { %r245, %r246, %r247, %r248 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r249, %r250, %r251, %r252 }, { %r117, %r118, %r119, %r120 }, { %r123, %r124 }, { %r249, %r250, %r251, %r252 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r245, %r246, %r247, %r248 }, { %r125, %r126, %r127, %r128 }, { %r129, %r130 }, { %r245, %r246, %r247, %r248 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r249, %r250, %r251, %r252 }, { %r125, %r126, %r127, %r128 }, { %r131, %r132 }, { %r249, %r250, %r251, %r252 };
	// end inline asm
	.loc	1 184 18                        // sk03_fa_qkv.py:184:18
	add.s64 	%rd41, %rd12, %rd54;
	add.s64 	%rd42, %rd11, %rd54;
	add.s64 	%rd43, %rd10, %rd54;
	add.s64 	%rd44, %rd9, %rd54;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	add.s64 	%rd45, %rd8, %rd54;
	add.s32 	%r151, %r244, 1;
	setp.gt.s32 	%p6, %r151, 1;
	selp.b32 	%r244, 0, %r151, %p6;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
	shl.b32 	%r152, %r244, 11;
	bar.sync 	0;
	add.s32 	%r153, %r11, %r152;
	add.s32 	%r133, %r153, 32768;
	selp.b32 	%r134, 8, 0, %p4;
	// begin inline asm
	cp.async.ca.shared.global [ %r133 + 0 ], [ %rd41 + 0 ], 0x8, %r134;
	// end inline asm
	cp.async.commit_group;
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	shl.b32 	%r154, %r244, 14;
	add.s32 	%r135, %r26, %r154;
	selp.b32 	%r136, 16, 0, %p4;
	// begin inline asm
	cp.async.cg.shared.global [ %r135 + 0 ], [ %rd42 + 0 ], 0x10, %r136;
	// end inline asm
	add.s32 	%r137, %r135, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r137 + 0 ], [ %rd43 + 0 ], 0x10, %r136;
	// end inline asm
	add.s32 	%r138, %r135, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r138 + 0 ], [ %rd44 + 0 ], 0x10, %r136;
	// end inline asm
	add.s32 	%r139, %r135, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r139 + 0 ], [ %rd45 + 0 ], 0x10, %r136;
	// end inline asm
	cp.async.commit_group;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	add.s32 	%r253, %r253, 1;
	add.s64 	%rd54, %rd54, 128;
	setp.ne.b32 	%p7, %r10, %r253;
	@%p7 bra 	$L__BB0_2;
// %bb.3:                               // %._crit_edge.loopexit
	.loc	1 186 17                        // sk03_fa_qkv.py:186:17
	cvt.rn.f32.s32 	%r155, %r245;
	cvt.rn.bf16.f32 	%rs32, %r155;
	cvt.rn.f32.s32 	%r156, %r246;
	cvt.rn.bf16.f32 	%rs33, %r156;
	cvt.rn.f32.s32 	%r157, %r247;
	cvt.rn.bf16.f32 	%rs34, %r157;
	cvt.rn.f32.s32 	%r158, %r248;
	cvt.rn.bf16.f32 	%rs35, %r158;
	cvt.rn.f32.s32 	%r159, %r249;
	cvt.rn.bf16.f32 	%rs36, %r159;
	cvt.rn.f32.s32 	%r160, %r250;
	cvt.rn.bf16.f32 	%rs37, %r160;
	cvt.rn.f32.s32 	%r161, %r251;
	cvt.rn.bf16.f32 	%rs38, %r161;
	cvt.rn.f32.s32 	%r162, %r252;
	cvt.rn.bf16.f32 	%rs39, %r162;
$L__BB0_4:                              // %._crit_edge
	.loc	1 0 17                          // sk03_fa_qkv.py:0:17
	cvt.u32.u64 	%r182, %rd2;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r183, %r6, %r182;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r184, %r183, %r21;
	.loc	1 164 45                        // sk03_fa_qkv.py:164:45
	and.b32 	%r185, %r12, 6;
	shr.u32 	%r186, %r7, 2;
	or.b32 	%r187, %r185, %r186;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r188, %r187, %r6;
	or.b32 	%r189, %r188, 64;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r190, %r189, %r21;
	rem.s32 	%r191, %r188, %r21;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	and.b32 	%r192, %r2, 28;
	bfe.u32 	%r193, %r2, 2, 3;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r194, %r193, %r1;
	or.b32 	%r195, %r194, 8;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r196, %r195, %r20;
	rem.s32 	%r197, %r194, %r20;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 186 54                        // sk03_fa_qkv.py:186:54
	mad.wide.s32 	%rd46, %r197, 4, %rd17;
	mad.wide.s32 	%rd47, %r196, 4, %rd17;
	.loc	1 186 40                        // sk03_fa_qkv.py:186:40
	// begin inline asm
	mov.u32 %r163, 0x0;
	ld.global.b32 { %r163 }, [ %rd46 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r164, 0x0;
	ld.global.b32 { %r164 }, [ %rd47 + 0 ];
	// end inline asm
	.loc	1 186 65                        // sk03_fa_qkv.py:186:65
	cvt.rn.bf16.f32 	%rs9, %r163;
	cvt.rn.bf16.f32 	%rs10, %r164;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	mov.b16 	%rs11, 0x8000;
	fma.rn.bf16 	%rs12, %rs32, %rs9, %rs11;
	fma.rn.bf16 	%rs13, %rs33, %rs9, %rs11;
	fma.rn.bf16 	%rs14, %rs34, %rs10, %rs11;
	fma.rn.bf16 	%rs15, %rs35, %rs10, %rs11;
	fma.rn.bf16 	%rs16, %rs36, %rs9, %rs11;
	fma.rn.bf16 	%rs17, %rs37, %rs9, %rs11;
	fma.rn.bf16 	%rs18, %rs38, %rs10, %rs11;
	fma.rn.bf16 	%rs19, %rs39, %rs10, %rs11;
	.loc	1 187 38                        // sk03_fa_qkv.py:187:38
	mad.wide.s32 	%rd48, %r191, 4, %rd18;
	mad.wide.s32 	%rd49, %r190, 4, %rd18;
	.loc	1 187 24                        // sk03_fa_qkv.py:187:24
	// begin inline asm
	mov.u32 %r165, 0x0;
	mov.u32 %r166, 0x0;
	ld.global.v2.b32 { %r165, %r166 }, [ %rd48 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r167, 0x0;
	mov.u32 %r168, 0x0;
	ld.global.v2.b32 { %r167, %r168 }, [ %rd49 + 0 ];
	// end inline asm
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs20, %r165;
	cvt.rn.bf16.f32 	%rs21, %r166;
	cvt.rn.bf16.f32 	%rs22, %r167;
	cvt.rn.bf16.f32 	%rs23, %r168;
	.loc	1 188 49                        // sk03_fa_qkv.py:188:49
	mul.lo.s32 	%r198, %r5, %r23;
	.loc	1 188 31                        // sk03_fa_qkv.py:188:31
	mad.wide.s32 	%rd52, %r198, 2, %rd16;
	.loc	1 188 64                        // sk03_fa_qkv.py:188:64
	mad.wide.s32 	%rd50, %r184, 2, %rd52;
	.loc	1 188 19                        // sk03_fa_qkv.py:188:19
	// begin inline asm
	mov.u32 %r170, 0x0;
	mov.u32 %r171, 0x0;
	mov.u32 %r172, 0x0;
	mov.u32 %r173, 0x0;
	ld.global.v4.b32 { %r170, %r171, %r172, %r173 }, [ %rd50 + 0 ];
	// end inline asm
	and.b32 	%r199, %r2, 120;
	shl.b32 	%r200, %r199, 5;
	or.b32 	%r201, %r200, %r242;
	xor.b32 	%r202, %r201, %r3;
	add.s32 	%r169, %r89, %r202;
	// begin inline asm
	st.shared.v4.b32 [ %r169 + 0 ], { %r170, %r171, %r172, %r173 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r203, %r9, 9;
	shl.b32 	%r204, %r2, 4;
	and.b32 	%r205, %r204, 496;
	shr.u32 	%r206, %r7, 1;
	xor.b32 	%r207, %r205, %r206;
	add.s32 	%r208, %r89, %r203;
	add.s32 	%r209, %r208, %r207;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r210, %r211, %r212, %r213}, [%r209];
	mov.b32 	{%rs24, %rs25}, %r210;
	mov.b32 	{%rs26, %rs27}, %r211;
	mov.b32 	{%rs28, %rs29}, %r212;
	mov.b32 	{%rs30, %rs31}, %r213;
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs1, %rs12, %rs20, %rs24;
	fma.rn.bf16 	%rs2, %rs13, %rs21, %rs25;
	fma.rn.bf16 	%rs3, %rs14, %rs20, %rs26;
	fma.rn.bf16 	%rs4, %rs15, %rs21, %rs27;
	fma.rn.bf16 	%rs5, %rs16, %rs22, %rs28;
	fma.rn.bf16 	%rs6, %rs17, %rs23, %rs29;
	fma.rn.bf16 	%rs7, %rs18, %rs22, %rs30;
	fma.rn.bf16 	%rs8, %rs19, %rs23, %rs31;
	bar.sync 	0;
	shl.b32 	%r214, %r2, 5;
	and.b32 	%r215, %r214, 768;
	shl.b32 	%r216, %r192, 1;
	and.b32 	%r217, %r2, 1;
	neg.s32 	%r218, %r217;
	and.b32 	%r219, %r218, 1088;
	bfe.s32 	%r220, %r2, 1, 1;
	and.b32 	%r221, %r220, 2052;
	or.b32 	%r222, %r215, %r216;
	or.b32 	%r223, %r219, %r222;
	xor.b32 	%r224, %r223, %r206;
	or.b32 	%r225, %r224, %r221;
	add.s32 	%r174, %r89, %r225;
	// begin inline asm
	st.shared.v2.b16 [ %r174 + 0 ], { %rs1, %rs2 };
	// end inline asm
	add.s32 	%r175, %r174, 128;
	// begin inline asm
	st.shared.v2.b16 [ %r175 + 0 ], { %rs3, %rs4 };
	// end inline asm
	xor.b32 	%r226, %r225, 4;
	add.s32 	%r176, %r89, %r226;
	// begin inline asm
	st.shared.v2.b16 [ %r176 + 0 ], { %rs5, %rs6 };
	// end inline asm
	add.s32 	%r177, %r176, 128;
	// begin inline asm
	st.shared.v2.b16 [ %r177 + 0 ], { %rs7, %rs8 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r227, %r2, 3;
	and.b32 	%r228, %r227, 768;
	shr.u32 	%r229, %r199, 1;
	and.b32 	%r230, %r2, 128;
	or.b32 	%r231, %r242, %r228;
	xor.b32 	%r232, %r231, %r229;
	or.b32 	%r233, %r232, %r230;
	add.s32 	%r234, %r89, %r233;
	ld.shared.b32 	%r178, [%r234];
	xor.b32 	%r235, %r233, 64;
	add.s32 	%r236, %r89, %r235;
	ld.shared.b32 	%r179, [%r236+1024];
	xor.b32 	%r237, %r233, 4;
	add.s32 	%r238, %r89, %r237;
	ld.shared.b32 	%r180, [%r238+2048];
	xor.b32 	%r239, %r233, 68;
	add.s32 	%r240, %r89, %r239;
	ld.shared.b32 	%r181, [%r240+3072];
	.loc	1 195 31                        // sk03_fa_qkv.py:195:31
	setp.lt.s32 	%p9, %r4, %r20;
	.loc	1 195 54                        // sk03_fa_qkv.py:195:54
	setp.lt.s32 	%p10, %r183, %r21;
	.loc	1 195 37                        // sk03_fa_qkv.py:195:37
	and.pred 	%p8, %p9, %p10;
	.loc	1 193 35                        // sk03_fa_qkv.py:193:35
	mul.lo.s32 	%r241, %r4, %r22;
	.loc	1 193 18                        // sk03_fa_qkv.py:193:18
	mad.wide.s32 	%rd53, %r241, 2, %rd15;
	.loc	1 193 50                        // sk03_fa_qkv.py:193:50
	mad.wide.s32 	%rd51, %r183, 2, %rd53;
	.loc	1 194 8                         // sk03_fa_qkv.py:194:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd51 + 0 ], { %r178, %r179, %r180, %r181 };
	// end inline asm
	.loc	1 192 4                         // sk03_fa_qkv.py:192:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk03_fa_qkv.py"
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
.b32 157                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x96 DW_TAG_compile_unit
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
.b8 51
.b8 95
.b8 102
.b8 97
.b8 95
.b8 113
.b8 107
.b8 118
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
.b8 2                                   // Abbrev [2] 0x44:0x16 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 51
.b8 95
.b8 102
.b8 97
.b8 95
.b8 113
.b8 107
.b8 118
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x5a:0x46 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 68                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x6f:0x18 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 155                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x87:0x18 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 156                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_0 = _Nativo(
    "sk03_fa_qkv/tile16x128x128_shift0_abi15",
    _PTX_0, "_sk03_fa_qkv_kernel",
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

	// .globl	_sk03_fa_qkv_kernel     // -- Begin function _sk03_fa_qkv_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk03_fa_qkv_kernel
.visible .entry _sk03_fa_qkv_kernel(
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_6,
	.param .u32 _sk03_fa_qkv_kernel_param_7,
	.param .u32 _sk03_fa_qkv_kernel_param_8,
	.param .u32 _sk03_fa_qkv_kernel_param_9,
	.param .u32 _sk03_fa_qkv_kernel_param_10,
	.param .u32 _sk03_fa_qkv_kernel_param_11,
	.param .u32 _sk03_fa_qkv_kernel_param_12,
	.param .u32 _sk03_fa_qkv_kernel_param_13,
	.param .u32 _sk03_fa_qkv_kernel_param_14,
	.param .u32 _sk03_fa_qkv_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_16,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_17
)
.reqntid 256
{
	.reg .pred 	%p<11>;
	.reg .b16 	%rs<48>;
	.reg .b32 	%r<277>;
	.reg .b64 	%rd<62>;
	.loc	1 145 0                         // sk03_fa_qkv.py:145:0
$L__func_begin0:
	.loc	1 145 0                         // sk03_fa_qkv.py:145:0

// %bb.0:
	ld.param.b32 	%r24, [_sk03_fa_qkv_kernel_param_14];
	ld.param.b32 	%r23, [_sk03_fa_qkv_kernel_param_13];
	ld.param.b32 	%r22, [_sk03_fa_qkv_kernel_param_12];
	ld.param.b32 	%r21, [_sk03_fa_qkv_kernel_param_8];
	ld.param.b32 	%r20, [_sk03_fa_qkv_kernel_param_7];
	ld.param.b64 	%rd18, [_sk03_fa_qkv_kernel_param_5];
	ld.param.b64 	%rd17, [_sk03_fa_qkv_kernel_param_4];
	ld.param.b64 	%rd16, [_sk03_fa_qkv_kernel_param_3];
	ld.param.b64 	%rd15, [_sk03_fa_qkv_kernel_param_2];
	ld.param.b64 	%rd14, [_sk03_fa_qkv_kernel_param_1];
	ld.param.b64 	%rd13, [_sk03_fa_qkv_kernel_param_0];
$L__tmp0:
	.loc	1 154 24                        // sk03_fa_qkv.py:154:24
	mov.u32 	%r39, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk03_fa_qkv.py:155:27 ]
	add.s32 	%r40, %r20, 15;
	.loc	2 43 30                         // standard.py:43:30 @[ sk03_fa_qkv.py:155:27 ]
	shr.s32 	%r41, %r40, 31;
	shr.u32 	%r42, %r41, 28;
	add.s32 	%r43, %r40, %r42;
	shr.s32 	%r44, %r43, 4;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk03_fa_qkv.py:156:27 ]
	add.s32 	%r45, %r21, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk03_fa_qkv.py:156:27 ]
	shr.s32 	%r46, %r45, 31;
	shr.u32 	%r47, %r46, 25;
	add.s32 	%r48, %r45, %r47;
	shr.s32 	%r49, %r48, 7;
$L__tmp3:
	.loc	1 157 29                        // sk03_fa_qkv.py:157:29
	shl.b32 	%r50, %r49, 3;
	.loc	1 158 22                        // sk03_fa_qkv.py:158:22
	div.s32 	%r51, %r39, %r50;
	.loc	1 158 38                        // sk03_fa_qkv.py:158:38
	shl.b32 	%r52, %r51, 3;
	ld.param.b32 	%r53, [_sk03_fa_qkv_kernel_param_9];
	.loc	1 159 30                        // sk03_fa_qkv.py:159:30
	sub.s32 	%r54, %r44, %r52;
	ld.param.b32 	%r55, [_sk03_fa_qkv_kernel_param_10];
	.loc	1 159 39                        // sk03_fa_qkv.py:159:39
	min.s32 	%r56, %r54, 8;
	ld.param.b32 	%r57, [_sk03_fa_qkv_kernel_param_11];
	.loc	1 160 30                        // sk03_fa_qkv.py:160:30
	mul.lo.s32 	%r58, %r51, %r50;
	sub.s32 	%r59, %r39, %r58;
	.loc	1 161 36                        // sk03_fa_qkv.py:161:36
	div.s32 	%r60, %r59, %r56;
	.loc	1 160 46                        // sk03_fa_qkv.py:160:46
	mul.lo.s32 	%r61, %r60, %r56;
	sub.s32 	%r62, %r59, %r61;
	.loc	1 160 23                        // sk03_fa_qkv.py:160:23
	add.s32 	%r63, %r62, %r52;
	.loc	1 163 22                        // sk03_fa_qkv.py:163:22
	shl.b32 	%r1, %r63, 4;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	mov.u32 	%r2, %tid.x;
	and.b32 	%r3, %r2, 240;
	bfe.u32 	%r64, %r2, 4, 4;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r4, %r1, %r64;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r5, %r4, %r20;
	.loc	1 164 22                        // sk03_fa_qkv.py:164:22
	shl.b32 	%r6, %r60, 7;
	.loc	1 164 45                        // sk03_fa_qkv.py:164:45
	shr.u32 	%r65, %r2, 3;
	bfe.u32 	%r66, %r2, 3, 5;
	and.b32 	%r7, %r2, 224;
	and.b32 	%r8, %r2, 15;
	shl.b32 	%r67, %r8, 3;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r68, %r6, %r66;
	or.b32 	%r69, %r68, 32;
	or.b32 	%r70, %r68, 64;
	or.b32 	%r71, %r65, %r6;
	or.b32 	%r72, %r71, 96;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r73, %r68, %r21;
	rem.s32 	%r74, %r69, %r21;
	rem.s32 	%r75, %r70, %r21;
	rem.s32 	%r76, %r72, %r21;
	.loc	1 167 39                        // sk03_fa_qkv.py:167:39
	mul.lo.s32 	%r77, %r5, %r55;
	.loc	1 167 21                        // sk03_fa_qkv.py:167:21
	cvt.s64.s32 	%rd1, %r77;
	add.s64 	%rd29, %rd13, %rd1;
	.loc	1 167 51                        // sk03_fa_qkv.py:167:51
	cvt.u64.u32 	%rd2, %r67;
	add.s64 	%rd19, %rd29, %rd2;
	.loc	1 168 28                        // sk03_fa_qkv.py:168:28
	and.b32 	%r9, %r2, 7;
	shl.b32 	%r78, %r9, 4;
	.loc	1 168 21                        // sk03_fa_qkv.py:168:21
	cvt.u64.u32 	%rd3, %r78;
	add.s64 	%rd30, %rd14, %rd3;
	.loc	1 168 69                        // sk03_fa_qkv.py:168:69
	mul.lo.s32 	%r79, %r73, %r57;
	mul.lo.s32 	%r80, %r74, %r57;
	mul.lo.s32 	%r81, %r75, %r57;
	mul.lo.s32 	%r82, %r76, %r57;
	.loc	1 168 51                        // sk03_fa_qkv.py:168:51
	cvt.s64.s32 	%rd4, %r79;
	add.s64 	%rd20, %rd30, %rd4;
	cvt.s64.s32 	%rd5, %r80;
	add.s64 	%rd21, %rd30, %rd5;
	cvt.s64.s32 	%rd6, %r81;
	add.s64 	%rd22, %rd30, %rd6;
	cvt.s64.s32 	%rd7, %r82;
	add.s64 	%rd23, %rd30, %rd7;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	setp.lt.s32 	%p1, %r53, 128;
	setp.gt.s32 	%p2, %r53, 127;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
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
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
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
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	setp.gt.s32 	%p3, %r53, 255;
	.loc	1 183 18                        // sk03_fa_qkv.py:183:18
	add.s64 	%rd24, %rd19, 128;
	.loc	1 184 18                        // sk03_fa_qkv.py:184:18
	add.s64 	%rd25, %rd20, 128;
	add.s64 	%rd26, %rd21, 128;
	add.s64 	%rd27, %rd22, 128;
	add.s64 	%rd28, %rd23, 128;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
	bar.sync 	0;
	add.s32 	%r32, %r11, 34816;
	selp.b32 	%r33, 8, 0, %p3;
	// begin inline asm
	cp.async.ca.shared.global [ %r32 + 0 ], [ %rd24 + 0 ], 0x8, %r33;
	// end inline asm
	cp.async.commit_group;
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
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
	mov.b16 	%rs40, 0x0000;
	cvt.u32.u64 	%r265, %rd3;
	mov.b16 	%rs41, %rs40;
	mov.b16 	%rs42, %rs40;
	mov.b16 	%rs43, %rs40;
	mov.b16 	%rs44, %rs40;
	mov.b16 	%rs45, %rs40;
	mov.b16 	%rs46, %rs40;
	mov.b16 	%rs47, %rs40;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	@%p1 bra 	$L__BB0_4;
// %bb.1:                               // %.lr.ph
	.loc	1 0 23                          // sk03_fa_qkv.py:0:23
	shr.s32 	%r83, %r53, 31;
	shr.u32 	%r84, %r83, 25;
	add.s32 	%r85, %r53, %r84;
	shr.s32 	%r10, %r85, 7;
	add.s32 	%r13, %r10, -2;
	shl.b32 	%r94, %r8, 7;
	and.b32 	%r95, %r2, 16;
	xor.b32 	%r96, %r265, %r95;
	or.b32 	%r14, %r96, %r94;
	xor.b32 	%r15, %r14, 32;
	xor.b32 	%r16, %r14, 64;
	xor.b32 	%r17, %r14, 96;
	shl.b32 	%r97, %r9, 7;
	shl.b32 	%r98, %r7, 5;
	and.b32 	%r99, %r12, 48;
	or.b32 	%r100, %r97, %r98;
	xor.b32 	%r101, %r265, %r99;
	or.b32 	%r18, %r100, %r101;
	xor.b32 	%r19, %r18, 64;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
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
	mov.b32 	%r268, 0;
	mov.b32 	%r267, 1;
	mov.b32 	%r266, -1;
	mov.b64 	%rd61, 0;
	mov.b32 	%r269, %r268;
	mov.b32 	%r270, %r268;
	mov.b32 	%r271, %r268;
	mov.b32 	%r272, %r268;
	mov.b32 	%r273, %r268;
	mov.b32 	%r274, %r268;
	mov.b32 	%r275, %r268;
	mov.b32 	%r276, %r268;
$L__BB0_2:                              // =>This Inner Loop Header: Depth=1
	setp.lt.s32 	%p4, %r276, %r13;
	add.s32 	%r141, %r266, 1;
	setp.gt.s32 	%p5, %r141, 1;
	selp.b32 	%r266, 0, %r141, %p5;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
	cp.async.wait_group 	2;
	bar.sync 	0;
	shl.b32 	%r142, %r266, 11;
	add.s32 	%r143, %r90, %r142;
	add.s32 	%r144, %r143, %r14;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r102, %r103, %r104, %r105}, [%r144+32768];
	add.s32 	%r145, %r143, %r15;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r110, %r111, %r112, %r113}, [%r145+32768];
	add.s32 	%r146, %r143, %r16;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r118, %r119, %r120, %r121}, [%r146+32768];
	add.s32 	%r147, %r143, %r17;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r126, %r127, %r128, %r129}, [%r147+32768];
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	shl.b32 	%r148, %r266, 14;
	add.s32 	%r149, %r90, %r148;
	add.s32 	%r150, %r149, %r18;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r106, %r107, %r114, %r115}, [%r150];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r108, %r109, %r116, %r117}, [%r150+8192];
	add.s32 	%r151, %r149, %r19;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r122, %r123, %r130, %r131}, [%r151];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r124, %r125, %r132, %r133}, [%r151+8192];
	.loc	1 179 36                        // sk03_fa_qkv.py:179:36
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r268, %r269, %r270, %r271 }, { %r102, %r103, %r104, %r105 }, { %r106, %r107 }, { %r268, %r269, %r270, %r271 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r272, %r273, %r274, %r275 }, { %r102, %r103, %r104, %r105 }, { %r108, %r109 }, { %r272, %r273, %r274, %r275 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r268, %r269, %r270, %r271 }, { %r110, %r111, %r112, %r113 }, { %r114, %r115 }, { %r268, %r269, %r270, %r271 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r272, %r273, %r274, %r275 }, { %r110, %r111, %r112, %r113 }, { %r116, %r117 }, { %r272, %r273, %r274, %r275 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r268, %r269, %r270, %r271 }, { %r118, %r119, %r120, %r121 }, { %r122, %r123 }, { %r268, %r269, %r270, %r271 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r272, %r273, %r274, %r275 }, { %r118, %r119, %r120, %r121 }, { %r124, %r125 }, { %r272, %r273, %r274, %r275 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r268, %r269, %r270, %r271 }, { %r126, %r127, %r128, %r129 }, { %r130, %r131 }, { %r268, %r269, %r270, %r271 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r272, %r273, %r274, %r275 }, { %r126, %r127, %r128, %r129 }, { %r132, %r133 }, { %r272, %r273, %r274, %r275 };
	// end inline asm
	.loc	1 184 18                        // sk03_fa_qkv.py:184:18
	add.s64 	%rd41, %rd12, %rd61;
	add.s64 	%rd42, %rd11, %rd61;
	add.s64 	%rd43, %rd10, %rd61;
	add.s64 	%rd44, %rd9, %rd61;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	add.s64 	%rd45, %rd8, %rd61;
	add.s32 	%r152, %r267, 1;
	setp.gt.s32 	%p6, %r152, 1;
	selp.b32 	%r267, 0, %r152, %p6;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
	shl.b32 	%r153, %r267, 11;
	bar.sync 	0;
	add.s32 	%r154, %r11, %r153;
	add.s32 	%r134, %r154, 32768;
	selp.b32 	%r135, 8, 0, %p4;
	// begin inline asm
	cp.async.ca.shared.global [ %r134 + 0 ], [ %rd41 + 0 ], 0x8, %r135;
	// end inline asm
	cp.async.commit_group;
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	shl.b32 	%r155, %r267, 14;
	add.s32 	%r136, %r27, %r155;
	selp.b32 	%r137, 16, 0, %p4;
	// begin inline asm
	cp.async.cg.shared.global [ %r136 + 0 ], [ %rd42 + 0 ], 0x10, %r137;
	// end inline asm
	add.s32 	%r138, %r136, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r138 + 0 ], [ %rd43 + 0 ], 0x10, %r137;
	// end inline asm
	add.s32 	%r139, %r136, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r139 + 0 ], [ %rd44 + 0 ], 0x10, %r137;
	// end inline asm
	add.s32 	%r140, %r136, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r140 + 0 ], [ %rd45 + 0 ], 0x10, %r137;
	// end inline asm
	cp.async.commit_group;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	add.s32 	%r276, %r276, 1;
	add.s64 	%rd61, %rd61, 128;
	setp.ne.b32 	%p7, %r10, %r276;
	@%p7 bra 	$L__BB0_2;
// %bb.3:                               // %._crit_edge.loopexit
	.loc	1 186 17                        // sk03_fa_qkv.py:186:17
	cvt.rn.f32.s32 	%r156, %r268;
	cvt.rn.bf16.f32 	%rs40, %r156;
	cvt.rn.f32.s32 	%r157, %r269;
	cvt.rn.bf16.f32 	%rs41, %r157;
	cvt.rn.f32.s32 	%r158, %r270;
	cvt.rn.bf16.f32 	%rs42, %r158;
	cvt.rn.f32.s32 	%r159, %r271;
	cvt.rn.bf16.f32 	%rs43, %r159;
	cvt.rn.f32.s32 	%r160, %r272;
	cvt.rn.bf16.f32 	%rs44, %r160;
	cvt.rn.f32.s32 	%r161, %r273;
	cvt.rn.bf16.f32 	%rs45, %r161;
	cvt.rn.f32.s32 	%r162, %r274;
	cvt.rn.bf16.f32 	%rs46, %r162;
	cvt.rn.f32.s32 	%r163, %r275;
	cvt.rn.bf16.f32 	%rs47, %r163;
$L__BB0_4:                              // %._crit_edge
	.loc	1 0 17                          // sk03_fa_qkv.py:0:17
	cvt.u32.u64 	%r183, %rd2;
	.loc	1 164 45                        // sk03_fa_qkv.py:164:45
	or.b32 	%r184, %r6, %r183;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r185, %r184, 7;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r186, %r185, %r21;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r187, %r184, 6;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r188, %r187, %r21;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r189, %r184, 5;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r190, %r189, %r21;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r191, %r184, 4;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r192, %r191, %r21;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r193, %r184, 3;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r194, %r193, %r21;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r195, %r184, 2;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r196, %r195, %r21;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r197, %r184, 1;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r198, %r197, %r21;
	rem.s32 	%r199, %r184, %r21;
	.loc	1 164 45                        // sk03_fa_qkv.py:164:45
	and.b32 	%r200, %r12, 6;
	shr.u32 	%r201, %r7, 2;
	or.b32 	%r202, %r200, %r201;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r203, %r202, %r6;
	or.b32 	%r204, %r203, 64;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r205, %r204, %r21;
	rem.s32 	%r206, %r203, %r21;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	and.b32 	%r207, %r2, 28;
	bfe.u32 	%r208, %r2, 2, 3;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r209, %r208, %r1;
	or.b32 	%r210, %r209, 8;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r211, %r210, %r20;
	rem.s32 	%r212, %r209, %r20;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 186 54                        // sk03_fa_qkv.py:186:54
	mad.wide.s32 	%rd46, %r212, 4, %rd17;
	mad.wide.s32 	%rd47, %r211, 4, %rd17;
	.loc	1 186 40                        // sk03_fa_qkv.py:186:40
	// begin inline asm
	mov.u32 %r164, 0x0;
	ld.global.b32 { %r164 }, [ %rd46 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r165, 0x0;
	ld.global.b32 { %r165 }, [ %rd47 + 0 ];
	// end inline asm
	.loc	1 186 65                        // sk03_fa_qkv.py:186:65
	cvt.rn.bf16.f32 	%rs17, %r164;
	cvt.rn.bf16.f32 	%rs18, %r165;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	mov.b16 	%rs19, 0x8000;
	fma.rn.bf16 	%rs20, %rs40, %rs17, %rs19;
	fma.rn.bf16 	%rs21, %rs41, %rs17, %rs19;
	fma.rn.bf16 	%rs22, %rs42, %rs18, %rs19;
	fma.rn.bf16 	%rs23, %rs43, %rs18, %rs19;
	fma.rn.bf16 	%rs24, %rs44, %rs17, %rs19;
	fma.rn.bf16 	%rs25, %rs45, %rs17, %rs19;
	fma.rn.bf16 	%rs26, %rs46, %rs18, %rs19;
	fma.rn.bf16 	%rs27, %rs47, %rs18, %rs19;
	.loc	1 187 38                        // sk03_fa_qkv.py:187:38
	mad.wide.s32 	%rd48, %r206, 4, %rd18;
	mad.wide.s32 	%rd49, %r205, 4, %rd18;
	.loc	1 187 24                        // sk03_fa_qkv.py:187:24
	// begin inline asm
	mov.u32 %r166, 0x0;
	mov.u32 %r167, 0x0;
	ld.global.v2.b32 { %r166, %r167 }, [ %rd48 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r168, 0x0;
	mov.u32 %r169, 0x0;
	ld.global.v2.b32 { %r168, %r169 }, [ %rd49 + 0 ];
	// end inline asm
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs28, %r166;
	cvt.rn.bf16.f32 	%rs29, %r167;
	cvt.rn.bf16.f32 	%rs30, %r168;
	cvt.rn.bf16.f32 	%rs31, %r169;
	.loc	1 188 49                        // sk03_fa_qkv.py:188:49
	mul.lo.s32 	%r213, %r5, %r23;
	.loc	1 188 31                        // sk03_fa_qkv.py:188:31
	mad.wide.s32 	%rd59, %r213, 2, %rd16;
	.loc	1 188 82                        // sk03_fa_qkv.py:188:82
	mul.lo.s32 	%r214, %r199, %r24;
	mul.lo.s32 	%r215, %r198, %r24;
	mul.lo.s32 	%r216, %r196, %r24;
	mul.lo.s32 	%r217, %r194, %r24;
	mul.lo.s32 	%r218, %r192, %r24;
	mul.lo.s32 	%r219, %r190, %r24;
	mul.lo.s32 	%r220, %r188, %r24;
	mul.lo.s32 	%r221, %r186, %r24;
	.loc	1 188 64                        // sk03_fa_qkv.py:188:64
	mad.wide.s32 	%rd50, %r214, 2, %rd59;
	mad.wide.s32 	%rd51, %r215, 2, %rd59;
	mad.wide.s32 	%rd52, %r216, 2, %rd59;
	mad.wide.s32 	%rd53, %r217, 2, %rd59;
	mad.wide.s32 	%rd54, %r218, 2, %rd59;
	mad.wide.s32 	%rd55, %r219, 2, %rd59;
	mad.wide.s32 	%rd56, %r220, 2, %rd59;
	mad.wide.s32 	%rd57, %r221, 2, %rd59;
	.loc	1 188 19                        // sk03_fa_qkv.py:188:19
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
	and.b32 	%r222, %r2, 120;
	shl.b32 	%r223, %r222, 5;
	or.b32 	%r224, %r223, %r265;
	xor.b32 	%r225, %r224, %r3;
	add.s32 	%r170, %r90, %r225;
	mov.b32 	%r171, {%rs1, %rs2};
	mov.b32 	%r172, {%rs3, %rs4};
	mov.b32 	%r173, {%rs5, %rs6};
	mov.b32 	%r174, {%rs7, %rs8};
	// begin inline asm
	st.shared.v4.b32 [ %r170 + 0 ], { %r171, %r172, %r173, %r174 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r226, %r9, 9;
	shl.b32 	%r227, %r2, 4;
	and.b32 	%r228, %r227, 496;
	shr.u32 	%r229, %r7, 1;
	xor.b32 	%r230, %r228, %r229;
	add.s32 	%r231, %r90, %r226;
	add.s32 	%r232, %r231, %r230;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r233, %r234, %r235, %r236}, [%r232];
	mov.b32 	{%rs32, %rs33}, %r233;
	mov.b32 	{%rs34, %rs35}, %r234;
	mov.b32 	{%rs36, %rs37}, %r235;
	mov.b32 	{%rs38, %rs39}, %r236;
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs9, %rs20, %rs28, %rs32;
	fma.rn.bf16 	%rs10, %rs21, %rs29, %rs33;
	fma.rn.bf16 	%rs11, %rs22, %rs28, %rs34;
	fma.rn.bf16 	%rs12, %rs23, %rs29, %rs35;
	fma.rn.bf16 	%rs13, %rs24, %rs30, %rs36;
	fma.rn.bf16 	%rs14, %rs25, %rs31, %rs37;
	fma.rn.bf16 	%rs15, %rs26, %rs30, %rs38;
	fma.rn.bf16 	%rs16, %rs27, %rs31, %rs39;
	bar.sync 	0;
	shl.b32 	%r237, %r2, 5;
	and.b32 	%r238, %r237, 768;
	shl.b32 	%r239, %r207, 1;
	and.b32 	%r240, %r2, 1;
	neg.s32 	%r241, %r240;
	and.b32 	%r242, %r241, 1088;
	bfe.s32 	%r243, %r2, 1, 1;
	and.b32 	%r244, %r243, 2052;
	or.b32 	%r245, %r238, %r239;
	or.b32 	%r246, %r242, %r245;
	xor.b32 	%r247, %r246, %r229;
	or.b32 	%r248, %r247, %r244;
	add.s32 	%r175, %r90, %r248;
	// begin inline asm
	st.shared.v2.b16 [ %r175 + 0 ], { %rs9, %rs10 };
	// end inline asm
	add.s32 	%r176, %r175, 128;
	// begin inline asm
	st.shared.v2.b16 [ %r176 + 0 ], { %rs11, %rs12 };
	// end inline asm
	xor.b32 	%r249, %r248, 4;
	add.s32 	%r177, %r90, %r249;
	// begin inline asm
	st.shared.v2.b16 [ %r177 + 0 ], { %rs13, %rs14 };
	// end inline asm
	add.s32 	%r178, %r177, 128;
	// begin inline asm
	st.shared.v2.b16 [ %r178 + 0 ], { %rs15, %rs16 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r250, %r2, 3;
	and.b32 	%r251, %r250, 768;
	shr.u32 	%r252, %r222, 1;
	and.b32 	%r253, %r2, 128;
	or.b32 	%r254, %r265, %r251;
	xor.b32 	%r255, %r254, %r252;
	or.b32 	%r256, %r255, %r253;
	add.s32 	%r257, %r90, %r256;
	ld.shared.b32 	%r179, [%r257];
	xor.b32 	%r258, %r256, 64;
	add.s32 	%r259, %r90, %r258;
	ld.shared.b32 	%r180, [%r259+1024];
	xor.b32 	%r260, %r256, 4;
	add.s32 	%r261, %r90, %r260;
	ld.shared.b32 	%r181, [%r261+2048];
	xor.b32 	%r262, %r256, 68;
	add.s32 	%r263, %r90, %r262;
	ld.shared.b32 	%r182, [%r263+3072];
	.loc	1 195 31                        // sk03_fa_qkv.py:195:31
	setp.lt.s32 	%p9, %r4, %r20;
	.loc	1 195 54                        // sk03_fa_qkv.py:195:54
	setp.lt.s32 	%p10, %r184, %r21;
	.loc	1 195 37                        // sk03_fa_qkv.py:195:37
	and.pred 	%p8, %p9, %p10;
	.loc	1 193 35                        // sk03_fa_qkv.py:193:35
	mul.lo.s32 	%r264, %r4, %r22;
	.loc	1 193 18                        // sk03_fa_qkv.py:193:18
	mad.wide.s32 	%rd60, %r264, 2, %rd15;
	.loc	1 193 50                        // sk03_fa_qkv.py:193:50
	mad.wide.s32 	%rd58, %r184, 2, %rd60;
	.loc	1 194 8                         // sk03_fa_qkv.py:194:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd58 + 0 ], { %r179, %r180, %r181, %r182 };
	// end inline asm
	.loc	1 192 4                         // sk03_fa_qkv.py:192:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk03_fa_qkv.py"
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
.b32 157                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x96 DW_TAG_compile_unit
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
.b8 51
.b8 95
.b8 102
.b8 97
.b8 95
.b8 113
.b8 107
.b8 118
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
.b8 2                                   // Abbrev [2] 0x44:0x16 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 51
.b8 95
.b8 102
.b8 97
.b8 95
.b8 113
.b8 107
.b8 118
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x5a:0x46 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 68                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x6f:0x18 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 155                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x87:0x18 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 156                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_1 = _Nativo(
    "sk03_fa_qkv/tile16x128x128_shift0_abi16",
    _PTX_1, "_sk03_fa_qkv_kernel",
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

	// .globl	_sk03_fa_qkv_kernel     // -- Begin function _sk03_fa_qkv_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk03_fa_qkv_kernel
.visible .entry _sk03_fa_qkv_kernel(
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_6,
	.param .u32 _sk03_fa_qkv_kernel_param_7,
	.param .u32 _sk03_fa_qkv_kernel_param_8,
	.param .u32 _sk03_fa_qkv_kernel_param_9,
	.param .u32 _sk03_fa_qkv_kernel_param_10,
	.param .u32 _sk03_fa_qkv_kernel_param_11,
	.param .u32 _sk03_fa_qkv_kernel_param_12,
	.param .u32 _sk03_fa_qkv_kernel_param_13,
	.param .u32 _sk03_fa_qkv_kernel_param_14,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_16
)
.reqntid 256
{
	.reg .pred 	%p<11>;
	.reg .b16 	%rs<44>;
	.reg .b32 	%r<302>;
	.reg .b64 	%rd<72>;
	.loc	1 145 0                         // sk03_fa_qkv.py:145:0
$L__func_begin0:
	.loc	1 145 0                         // sk03_fa_qkv.py:145:0

// %bb.0:
	ld.param.b32 	%r24, [_sk03_fa_qkv_kernel_param_13];
	ld.param.b32 	%r23, [_sk03_fa_qkv_kernel_param_12];
	ld.param.b32 	%r22, [_sk03_fa_qkv_kernel_param_9];
	ld.param.b32 	%r21, [_sk03_fa_qkv_kernel_param_8];
	ld.param.b32 	%r20, [_sk03_fa_qkv_kernel_param_7];
	ld.param.b64 	%rd24, [_sk03_fa_qkv_kernel_param_5];
	ld.param.b64 	%rd23, [_sk03_fa_qkv_kernel_param_4];
	ld.param.b64 	%rd22, [_sk03_fa_qkv_kernel_param_3];
	ld.param.b64 	%rd21, [_sk03_fa_qkv_kernel_param_2];
	ld.param.b64 	%rd20, [_sk03_fa_qkv_kernel_param_1];
	ld.param.b64 	%rd19, [_sk03_fa_qkv_kernel_param_0];
$L__tmp0:
	.loc	1 154 24                        // sk03_fa_qkv.py:154:24
	mov.u32 	%r40, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk03_fa_qkv.py:155:27 ]
	add.s32 	%r41, %r20, 15;
	.loc	2 43 30                         // standard.py:43:30 @[ sk03_fa_qkv.py:155:27 ]
	shr.s32 	%r42, %r41, 31;
	shr.u32 	%r43, %r42, 28;
	add.s32 	%r44, %r41, %r43;
	shr.s32 	%r45, %r44, 4;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk03_fa_qkv.py:156:27 ]
	add.s32 	%r46, %r21, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk03_fa_qkv.py:156:27 ]
	shr.s32 	%r47, %r46, 31;
	shr.u32 	%r48, %r47, 25;
	add.s32 	%r49, %r46, %r48;
	shr.s32 	%r50, %r49, 7;
$L__tmp3:
	.loc	1 157 29                        // sk03_fa_qkv.py:157:29
	shl.b32 	%r51, %r50, 3;
	.loc	1 158 22                        // sk03_fa_qkv.py:158:22
	div.s32 	%r52, %r40, %r51;
	.loc	1 158 38                        // sk03_fa_qkv.py:158:38
	shl.b32 	%r53, %r52, 3;
	.loc	1 159 30                        // sk03_fa_qkv.py:159:30
	sub.s32 	%r54, %r45, %r53;
	ld.param.b32 	%r55, [_sk03_fa_qkv_kernel_param_10];
	.loc	1 159 39                        // sk03_fa_qkv.py:159:39
	min.s32 	%r56, %r54, 8;
	ld.param.b32 	%r57, [_sk03_fa_qkv_kernel_param_11];
	.loc	1 160 30                        // sk03_fa_qkv.py:160:30
	mul.lo.s32 	%r58, %r52, %r51;
	sub.s32 	%r59, %r40, %r58;
	.loc	1 161 36                        // sk03_fa_qkv.py:161:36
	div.s32 	%r60, %r59, %r56;
	.loc	1 160 46                        // sk03_fa_qkv.py:160:46
	mul.lo.s32 	%r61, %r60, %r56;
	sub.s32 	%r62, %r59, %r61;
	.loc	1 160 23                        // sk03_fa_qkv.py:160:23
	add.s32 	%r63, %r62, %r53;
	.loc	1 163 22                        // sk03_fa_qkv.py:163:22
	shl.b32 	%r1, %r63, 4;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	mov.u32 	%r2, %tid.x;
	and.b32 	%r3, %r2, 240;
	bfe.u32 	%r64, %r2, 4, 4;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r4, %r1, %r64;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r5, %r4, %r20;
	.loc	1 164 22                        // sk03_fa_qkv.py:164:22
	shl.b32 	%r6, %r60, 7;
	.loc	1 164 45                        // sk03_fa_qkv.py:164:45
	shr.u32 	%r65, %r2, 3;
	bfe.u32 	%r66, %r2, 3, 5;
	shl.b32 	%r7, %r2, 1;
	and.b32 	%r67, %r7, 6;
	and.b32 	%r8, %r2, 224;
	shr.u32 	%r68, %r8, 2;
	or.b32 	%r69, %r67, %r68;
	and.b32 	%r9, %r2, 15;
	shl.b32 	%r70, %r9, 3;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r71, %r6, %r66;
	or.b32 	%r72, %r71, 32;
	or.b32 	%r73, %r71, 64;
	or.b32 	%r74, %r65, %r6;
	or.b32 	%r75, %r74, 96;
	or.b32 	%r76, %r6, %r69;
	or.b32 	%r78, %r76, 64;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r80, %r71, %r21;
	rem.s32 	%r81, %r72, %r21;
	rem.s32 	%r82, %r73, %r21;
	rem.s32 	%r83, %r75, %r21;
	rem.s32 	%r10, %r76, %r21;
	rem.s32 	%r11, %r78, %r21;
	.loc	1 167 39                        // sk03_fa_qkv.py:167:39
	mul.lo.s32 	%r86, %r5, %r55;
	.loc	1 167 21                        // sk03_fa_qkv.py:167:21
	cvt.s64.s32 	%rd1, %r86;
	add.s64 	%rd36, %rd19, %rd1;
	.loc	1 167 51                        // sk03_fa_qkv.py:167:51
	cvt.u64.u32 	%rd2, %r70;
	add.s64 	%rd25, %rd36, %rd2;
	.loc	1 168 28                        // sk03_fa_qkv.py:168:28
	and.b32 	%r12, %r2, 7;
	shl.b32 	%r87, %r12, 4;
	.loc	1 168 21                        // sk03_fa_qkv.py:168:21
	cvt.u64.u32 	%rd3, %r87;
	add.s64 	%rd37, %rd20, %rd3;
	.loc	1 168 69                        // sk03_fa_qkv.py:168:69
	mul.lo.s32 	%r88, %r80, %r57;
	mul.lo.s32 	%r89, %r81, %r57;
	mul.lo.s32 	%r90, %r82, %r57;
	mul.lo.s32 	%r91, %r83, %r57;
	.loc	1 168 51                        // sk03_fa_qkv.py:168:51
	cvt.s64.s32 	%rd4, %r88;
	add.s64 	%rd26, %rd37, %rd4;
	cvt.s64.s32 	%rd5, %r89;
	add.s64 	%rd27, %rd37, %rd5;
	cvt.s64.s32 	%rd6, %r90;
	add.s64 	%rd28, %rd37, %rd6;
	cvt.s64.s32 	%rd7, %r91;
	add.s64 	%rd29, %rd37, %rd7;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	setp.lt.s32 	%p1, %r22, 128;
	setp.gt.s32 	%p2, %r22, 127;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
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
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
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
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	setp.gt.s32 	%p3, %r22, 255;
	.loc	1 183 18                        // sk03_fa_qkv.py:183:18
	add.s64 	%rd30, %rd25, 128;
	.loc	1 184 18                        // sk03_fa_qkv.py:184:18
	add.s64 	%rd31, %rd26, 128;
	add.s64 	%rd32, %rd27, 128;
	add.s64 	%rd33, %rd28, 128;
	add.s64 	%rd34, %rd29, 128;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
	bar.sync 	0;
	add.s32 	%r33, %r13, 34816;
	selp.b32 	%r34, 8, 0, %p3;
	// begin inline asm
	cp.async.ca.shared.global [ %r33 + 0 ], [ %rd30 + 0 ], 0x8, %r34;
	// end inline asm
	cp.async.commit_group;
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
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
	mov.b32 	%r298, 0;
	cvt.u32.u64 	%r286, %rd3;
	mov.b32 	%r299, %r298;
	mov.b32 	%r300, %r298;
	mov.b32 	%r301, %r298;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	@%p1 bra 	$L__BB0_4;
// %bb.1:                               // %.lr.ph
	.loc	1 0 23                          // sk03_fa_qkv.py:0:23
	ld.param.b32 	%r25, [_sk03_fa_qkv_kernel_param_14];
	ld.param.b64 	%rd35, [_sk03_fa_qkv_kernel_param_6];
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
	.loc	1 178 28                        // sk03_fa_qkv.py:178:28
	shr.u32 	%r117, %r22, 7;
	add.s32 	%r118, %r117, -2;
	shl.b32 	%r119, %r9, 7;
	and.b32 	%r120, %r2, 16;
	xor.b32 	%r121, %r286, %r120;
	or.b32 	%r14, %r121, %r119;
	xor.b32 	%r15, %r14, 32;
	xor.b32 	%r16, %r14, 64;
	xor.b32 	%r17, %r14, 96;
	shl.b32 	%r122, %r12, 7;
	shl.b32 	%r123, %r8, 5;
	and.b32 	%r124, %r7, 48;
	or.b32 	%r125, %r122, %r123;
	xor.b32 	%r126, %r286, %r124;
	or.b32 	%r18, %r125, %r126;
	xor.b32 	%r19, %r18, 64;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
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
	mov.b32 	%r116, 0;
	mov.b32 	%r289, 1;
	mov.b32 	%r288, -1;
	mov.b64 	%rd70, 0;
	mov.b32 	%r287, %r116;
	mov.b64 	%rd71, %rd70;
	mov.b32 	%r290, %r116;
	mov.b32 	%r291, %r116;
	mov.b32 	%r292, %r116;
	mov.b32 	%r293, %r116;
	mov.b32 	%r294, %r116;
	mov.b32 	%r295, %r116;
	mov.b32 	%r296, %r116;
	mov.b32 	%r297, %r116;
$L__BB0_2:                              // =>This Inner Loop Header: Depth=1
	setp.lt.s64 	%p4, %rd71, %rd12;
	add.s32 	%r175, %r288, 1;
	setp.gt.s32 	%p5, %r175, 1;
	selp.b32 	%r288, 0, %r175, %p5;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
	cp.async.wait_group 	2;
	bar.sync 	0;
	shl.b32 	%r176, %r288, 11;
	add.s32 	%r177, %r112, %r176;
	add.s32 	%r178, %r177, %r14;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r128, %r129, %r130, %r131}, [%r178+32768];
	add.s32 	%r179, %r177, %r15;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r140, %r141, %r142, %r143}, [%r179+32768];
	add.s32 	%r180, %r177, %r16;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r152, %r153, %r154, %r155}, [%r180+32768];
	add.s32 	%r181, %r177, %r17;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r160, %r161, %r162, %r163}, [%r181+32768];
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	shl.b32 	%r182, %r288, 14;
	add.s32 	%r183, %r112, %r182;
	add.s32 	%r184, %r183, %r18;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r132, %r133, %r144, %r145}, [%r184];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r134, %r135, %r150, %r151}, [%r184+8192];
	add.s32 	%r185, %r183, %r19;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r156, %r157, %r164, %r165}, [%r185];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r158, %r159, %r166, %r167}, [%r185+8192];
	.loc	1 179 36                        // sk03_fa_qkv.py:179:36
	mov.b32 	%r136, %r116;
	mov.b32 	%r137, %r116;
	mov.b32 	%r138, %r116;
	mov.b32 	%r139, %r116;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r136, %r137, %r138, %r139 }, { %r128, %r129, %r130, %r131 }, { %r132, %r133 }, { %r136, %r137, %r138, %r139 };
	// end inline asm
	mov.b32 	%r146, %r116;
	mov.b32 	%r147, %r116;
	mov.b32 	%r148, %r116;
	mov.b32 	%r149, %r116;
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
	.loc	1 181 39                        // sk03_fa_qkv.py:181:39
	cvt.s64.s32 	%rd61, %r287;
	add.s64 	%rd52, %rd8, %rd61;
	add.s64 	%rd53, %rd9, %rd61;
	add.s64 	%rd54, %rd10, %rd61;
	add.s64 	%rd55, %rd11, %rd61;
	.loc	1 181 29                        // sk03_fa_qkv.py:181:29
	// begin inline asm
	mov.u16 %rs1, 0x0;
	ld.global.b8 { %rs1 }, [ %rd52 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs2, 0x0;
	ld.global.b8 { %rs2 }, [ %rd53 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs3, 0x0;
	ld.global.b8 { %rs3 }, [ %rd54 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs4, 0x0;
	ld.global.b8 { %rs4 }, [ %rd55 + 0 ];
	// end inline asm
	.loc	1 181 21                        // sk03_fa_qkv.py:181:21
	cvt.u32.u16 	%r186, %rs1;
	and.b32 	%r187, %r186, 255;
	cvt.u32.u16 	%r188, %rs2;
	and.b32 	%r189, %r188, 255;
	cvt.u32.u16 	%r190, %rs3;
	and.b32 	%r191, %r190, 255;
	cvt.u32.u16 	%r192, %rs4;
	and.b32 	%r193, %r192, 255;
	shl.b32 	%r194, %r147, %r193;
	shl.b32 	%r195, %r149, %r193;
	shl.b32 	%r196, %r146, %r191;
	shl.b32 	%r197, %r148, %r191;
	shl.b32 	%r198, %r137, %r189;
	shl.b32 	%r199, %r139, %r189;
	shl.b32 	%r200, %r136, %r187;
	shl.b32 	%r201, %r138, %r187;
	.loc	1 182 15                        // sk03_fa_qkv.py:182:15
	add.s32 	%r292, %r201, %r292;
	add.s32 	%r290, %r200, %r290;
	add.s32 	%r293, %r199, %r293;
	add.s32 	%r291, %r198, %r291;
	add.s32 	%r296, %r197, %r296;
	add.s32 	%r294, %r196, %r294;
	add.s32 	%r297, %r195, %r297;
	add.s32 	%r295, %r194, %r295;
	.loc	1 184 18                        // sk03_fa_qkv.py:184:18
	add.s64 	%rd56, %rd18, %rd70;
	add.s64 	%rd57, %rd17, %rd70;
	add.s64 	%rd58, %rd16, %rd70;
	add.s64 	%rd59, %rd15, %rd70;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	add.s64 	%rd60, %rd14, %rd70;
	add.s32 	%r202, %r289, 1;
	setp.gt.s32 	%p6, %r202, 1;
	selp.b32 	%r289, 0, %r202, %p6;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
	shl.b32 	%r203, %r289, 11;
	bar.sync 	0;
	add.s32 	%r204, %r13, %r203;
	add.s32 	%r168, %r204, 32768;
	selp.b32 	%r169, 8, 0, %p4;
	// begin inline asm
	cp.async.ca.shared.global [ %r168 + 0 ], [ %rd56 + 0 ], 0x8, %r169;
	// end inline asm
	cp.async.commit_group;
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	shl.b32 	%r205, %r289, 14;
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
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	add.s64 	%rd71, %rd71, 1;
	add.s64 	%rd70, %rd70, 128;
	add.s32 	%r287, %r287, %r25;
	setp.ne.b64 	%p7, %rd13, %rd70;
	@%p7 bra 	$L__BB0_2;
// %bb.3:                               // %._crit_edge.loopexit
	.loc	1 186 17                        // sk03_fa_qkv.py:186:17
	cvt.rn.f32.s32 	%r206, %r290;
	cvt.rn.f32.s32 	%r207, %r291;
	cvt.rn.bf16x2.f32 	%r298, %r207, %r206;
	cvt.rn.f32.s32 	%r208, %r292;
	cvt.rn.f32.s32 	%r209, %r293;
	cvt.rn.bf16x2.f32 	%r299, %r209, %r208;
	cvt.rn.f32.s32 	%r210, %r294;
	cvt.rn.f32.s32 	%r211, %r295;
	cvt.rn.bf16x2.f32 	%r300, %r211, %r210;
	cvt.rn.f32.s32 	%r212, %r296;
	cvt.rn.f32.s32 	%r213, %r297;
	cvt.rn.bf16x2.f32 	%r301, %r213, %r212;
$L__BB0_4:                              // %._crit_edge
	.loc	1 0 17                          // sk03_fa_qkv.py:0:17
	cvt.u32.u64 	%r233, %rd2;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r234, %r6, %r233;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r235, %r234, %r21;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	and.b32 	%r236, %r2, 28;
	bfe.u32 	%r237, %r2, 2, 3;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r238, %r237, %r1;
	or.b32 	%r239, %r238, 8;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r240, %r239, %r20;
	rem.s32 	%r241, %r238, %r20;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 186 54                        // sk03_fa_qkv.py:186:54
	mad.wide.s32 	%rd62, %r241, 4, %rd23;
	mad.wide.s32 	%rd63, %r240, 4, %rd23;
	.loc	1 186 40                        // sk03_fa_qkv.py:186:40
	// begin inline asm
	mov.u32 %r214, 0x0;
	ld.global.b32 { %r214 }, [ %rd62 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r215, 0x0;
	ld.global.b32 { %r215 }, [ %rd63 + 0 ];
	// end inline asm
	.loc	1 186 65                        // sk03_fa_qkv.py:186:65
	cvt.rn.bf16.f32 	%rs13, %r214;
	cvt.rn.bf16.f32 	%rs14, %r215;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	mov.b32 	{%rs15, %rs16}, %r298;
	mov.b16 	%rs17, 0x8000;
	fma.rn.bf16 	%rs18, %rs15, %rs13, %rs17;
	fma.rn.bf16 	%rs19, %rs16, %rs13, %rs17;
	mov.b32 	{%rs20, %rs21}, %r299;
	fma.rn.bf16 	%rs22, %rs20, %rs14, %rs17;
	fma.rn.bf16 	%rs23, %rs21, %rs14, %rs17;
	mov.b32 	{%rs24, %rs25}, %r300;
	fma.rn.bf16 	%rs26, %rs24, %rs13, %rs17;
	fma.rn.bf16 	%rs27, %rs25, %rs13, %rs17;
	mov.b32 	{%rs28, %rs29}, %r301;
	fma.rn.bf16 	%rs30, %rs28, %rs14, %rs17;
	fma.rn.bf16 	%rs31, %rs29, %rs14, %rs17;
	.loc	1 187 38                        // sk03_fa_qkv.py:187:38
	mad.wide.s32 	%rd64, %r10, 4, %rd24;
	mad.wide.s32 	%rd65, %r11, 4, %rd24;
	.loc	1 187 24                        // sk03_fa_qkv.py:187:24
	// begin inline asm
	mov.u32 %r216, 0x0;
	mov.u32 %r217, 0x0;
	ld.global.v2.b32 { %r216, %r217 }, [ %rd64 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r218, 0x0;
	mov.u32 %r219, 0x0;
	ld.global.v2.b32 { %r218, %r219 }, [ %rd65 + 0 ];
	// end inline asm
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs32, %r216;
	cvt.rn.bf16.f32 	%rs33, %r217;
	cvt.rn.bf16.f32 	%rs34, %r218;
	cvt.rn.bf16.f32 	%rs35, %r219;
	.loc	1 188 49                        // sk03_fa_qkv.py:188:49
	mul.lo.s32 	%r242, %r5, %r24;
	.loc	1 188 31                        // sk03_fa_qkv.py:188:31
	mad.wide.s32 	%rd68, %r242, 2, %rd22;
	.loc	1 188 64                        // sk03_fa_qkv.py:188:64
	mad.wide.s32 	%rd66, %r235, 2, %rd68;
	.loc	1 188 19                        // sk03_fa_qkv.py:188:19
	// begin inline asm
	mov.u32 %r221, 0x0;
	mov.u32 %r222, 0x0;
	mov.u32 %r223, 0x0;
	mov.u32 %r224, 0x0;
	ld.global.v4.b32 { %r221, %r222, %r223, %r224 }, [ %rd66 + 0 ];
	// end inline asm
	and.b32 	%r243, %r2, 120;
	shl.b32 	%r244, %r243, 5;
	or.b32 	%r245, %r244, %r286;
	xor.b32 	%r246, %r245, %r3;
	add.s32 	%r220, %r112, %r246;
	// begin inline asm
	st.shared.v4.b32 [ %r220 + 0 ], { %r221, %r222, %r223, %r224 };
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
	mov.b32 	{%rs36, %rs37}, %r254;
	mov.b32 	{%rs38, %rs39}, %r255;
	mov.b32 	{%rs40, %rs41}, %r256;
	mov.b32 	{%rs42, %rs43}, %r257;
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs5, %rs18, %rs32, %rs36;
	fma.rn.bf16 	%rs6, %rs19, %rs33, %rs37;
	fma.rn.bf16 	%rs7, %rs22, %rs32, %rs38;
	fma.rn.bf16 	%rs8, %rs23, %rs33, %rs39;
	fma.rn.bf16 	%rs9, %rs26, %rs34, %rs40;
	fma.rn.bf16 	%rs10, %rs27, %rs35, %rs41;
	fma.rn.bf16 	%rs11, %rs30, %rs34, %rs42;
	fma.rn.bf16 	%rs12, %rs31, %rs35, %rs43;
	bar.sync 	0;
	shl.b32 	%r258, %r2, 5;
	and.b32 	%r259, %r258, 768;
	shl.b32 	%r260, %r236, 1;
	and.b32 	%r261, %r2, 1;
	neg.s32 	%r262, %r261;
	and.b32 	%r263, %r262, 1088;
	bfe.s32 	%r264, %r2, 1, 1;
	and.b32 	%r265, %r264, 2052;
	or.b32 	%r266, %r259, %r260;
	or.b32 	%r267, %r263, %r266;
	xor.b32 	%r268, %r267, %r250;
	or.b32 	%r269, %r268, %r265;
	add.s32 	%r225, %r112, %r269;
	// begin inline asm
	st.shared.v2.b16 [ %r225 + 0 ], { %rs5, %rs6 };
	// end inline asm
	add.s32 	%r226, %r225, 128;
	// begin inline asm
	st.shared.v2.b16 [ %r226 + 0 ], { %rs7, %rs8 };
	// end inline asm
	xor.b32 	%r270, %r269, 4;
	add.s32 	%r227, %r112, %r270;
	// begin inline asm
	st.shared.v2.b16 [ %r227 + 0 ], { %rs9, %rs10 };
	// end inline asm
	add.s32 	%r228, %r227, 128;
	// begin inline asm
	st.shared.v2.b16 [ %r228 + 0 ], { %rs11, %rs12 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r271, %r2, 3;
	and.b32 	%r272, %r271, 768;
	shr.u32 	%r273, %r243, 1;
	and.b32 	%r274, %r2, 128;
	or.b32 	%r275, %r286, %r272;
	xor.b32 	%r276, %r275, %r273;
	or.b32 	%r277, %r276, %r274;
	add.s32 	%r278, %r112, %r277;
	ld.shared.b32 	%r229, [%r278];
	xor.b32 	%r279, %r277, 64;
	add.s32 	%r280, %r112, %r279;
	ld.shared.b32 	%r230, [%r280+1024];
	xor.b32 	%r281, %r277, 4;
	add.s32 	%r282, %r112, %r281;
	ld.shared.b32 	%r231, [%r282+2048];
	xor.b32 	%r283, %r277, 68;
	add.s32 	%r284, %r112, %r283;
	ld.shared.b32 	%r232, [%r284+3072];
	.loc	1 195 31                        // sk03_fa_qkv.py:195:31
	setp.lt.s32 	%p9, %r4, %r20;
	.loc	1 195 54                        // sk03_fa_qkv.py:195:54
	setp.lt.s32 	%p10, %r234, %r21;
	.loc	1 195 37                        // sk03_fa_qkv.py:195:37
	and.pred 	%p8, %p9, %p10;
	.loc	1 193 35                        // sk03_fa_qkv.py:193:35
	mul.lo.s32 	%r285, %r4, %r23;
	.loc	1 193 18                        // sk03_fa_qkv.py:193:18
	mad.wide.s32 	%rd69, %r285, 2, %rd21;
	.loc	1 193 50                        // sk03_fa_qkv.py:193:50
	mad.wide.s32 	%rd67, %r234, 2, %rd69;
	.loc	1 194 8                         // sk03_fa_qkv.py:194:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd67 + 0 ], { %r229, %r230, %r231, %r232 };
	// end inline asm
	.loc	1 192 4                         // sk03_fa_qkv.py:192:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk03_fa_qkv.py"
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
.b32 157                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x96 DW_TAG_compile_unit
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
.b8 51
.b8 95
.b8 102
.b8 97
.b8 95
.b8 113
.b8 107
.b8 118
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
.b8 2                                   // Abbrev [2] 0x44:0x16 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 51
.b8 95
.b8 102
.b8 97
.b8 95
.b8 113
.b8 107
.b8 118
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x5a:0x46 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 68                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x6f:0x18 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 155                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x87:0x18 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 156                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_2 = _Nativo(
    "sk03_fa_qkv/tile16x128x128_shift1_abi15",
    _PTX_2, "_sk03_fa_qkv_kernel",
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

	// .globl	_sk03_fa_qkv_kernel     // -- Begin function _sk03_fa_qkv_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk03_fa_qkv_kernel
.visible .entry _sk03_fa_qkv_kernel(
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_6,
	.param .u32 _sk03_fa_qkv_kernel_param_7,
	.param .u32 _sk03_fa_qkv_kernel_param_8,
	.param .u32 _sk03_fa_qkv_kernel_param_9,
	.param .u32 _sk03_fa_qkv_kernel_param_10,
	.param .u32 _sk03_fa_qkv_kernel_param_11,
	.param .u32 _sk03_fa_qkv_kernel_param_12,
	.param .u32 _sk03_fa_qkv_kernel_param_13,
	.param .u32 _sk03_fa_qkv_kernel_param_14,
	.param .u32 _sk03_fa_qkv_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_16,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_17
)
.reqntid 256
{
	.reg .pred 	%p<11>;
	.reg .b16 	%rs<52>;
	.reg .b32 	%r<325>;
	.reg .b64 	%rd<79>;
	.loc	1 145 0                         // sk03_fa_qkv.py:145:0
$L__func_begin0:
	.loc	1 145 0                         // sk03_fa_qkv.py:145:0

// %bb.0:
	ld.param.b32 	%r25, [_sk03_fa_qkv_kernel_param_14];
	ld.param.b32 	%r24, [_sk03_fa_qkv_kernel_param_13];
	ld.param.b32 	%r23, [_sk03_fa_qkv_kernel_param_12];
	ld.param.b32 	%r22, [_sk03_fa_qkv_kernel_param_9];
	ld.param.b32 	%r21, [_sk03_fa_qkv_kernel_param_8];
	ld.param.b32 	%r20, [_sk03_fa_qkv_kernel_param_7];
	ld.param.b64 	%rd24, [_sk03_fa_qkv_kernel_param_5];
	ld.param.b64 	%rd23, [_sk03_fa_qkv_kernel_param_4];
	ld.param.b64 	%rd22, [_sk03_fa_qkv_kernel_param_3];
	ld.param.b64 	%rd21, [_sk03_fa_qkv_kernel_param_2];
	ld.param.b64 	%rd20, [_sk03_fa_qkv_kernel_param_1];
	ld.param.b64 	%rd19, [_sk03_fa_qkv_kernel_param_0];
$L__tmp0:
	.loc	1 154 24                        // sk03_fa_qkv.py:154:24
	mov.u32 	%r41, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk03_fa_qkv.py:155:27 ]
	add.s32 	%r42, %r20, 15;
	.loc	2 43 30                         // standard.py:43:30 @[ sk03_fa_qkv.py:155:27 ]
	shr.s32 	%r43, %r42, 31;
	shr.u32 	%r44, %r43, 28;
	add.s32 	%r45, %r42, %r44;
	shr.s32 	%r46, %r45, 4;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk03_fa_qkv.py:156:27 ]
	add.s32 	%r47, %r21, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk03_fa_qkv.py:156:27 ]
	shr.s32 	%r48, %r47, 31;
	shr.u32 	%r49, %r48, 25;
	add.s32 	%r50, %r47, %r49;
	shr.s32 	%r51, %r50, 7;
$L__tmp3:
	.loc	1 157 29                        // sk03_fa_qkv.py:157:29
	shl.b32 	%r52, %r51, 3;
	.loc	1 158 22                        // sk03_fa_qkv.py:158:22
	div.s32 	%r53, %r41, %r52;
	.loc	1 158 38                        // sk03_fa_qkv.py:158:38
	shl.b32 	%r54, %r53, 3;
	.loc	1 159 30                        // sk03_fa_qkv.py:159:30
	sub.s32 	%r55, %r46, %r54;
	ld.param.b32 	%r56, [_sk03_fa_qkv_kernel_param_10];
	.loc	1 159 39                        // sk03_fa_qkv.py:159:39
	min.s32 	%r57, %r55, 8;
	ld.param.b32 	%r58, [_sk03_fa_qkv_kernel_param_11];
	.loc	1 160 30                        // sk03_fa_qkv.py:160:30
	mul.lo.s32 	%r59, %r53, %r52;
	sub.s32 	%r60, %r41, %r59;
	.loc	1 161 36                        // sk03_fa_qkv.py:161:36
	div.s32 	%r61, %r60, %r57;
	.loc	1 160 46                        // sk03_fa_qkv.py:160:46
	mul.lo.s32 	%r62, %r61, %r57;
	sub.s32 	%r63, %r60, %r62;
	.loc	1 160 23                        // sk03_fa_qkv.py:160:23
	add.s32 	%r64, %r63, %r54;
	.loc	1 163 22                        // sk03_fa_qkv.py:163:22
	shl.b32 	%r1, %r64, 4;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	mov.u32 	%r2, %tid.x;
	and.b32 	%r3, %r2, 240;
	bfe.u32 	%r65, %r2, 4, 4;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r4, %r1, %r65;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r5, %r4, %r20;
	.loc	1 164 22                        // sk03_fa_qkv.py:164:22
	shl.b32 	%r6, %r61, 7;
	.loc	1 164 45                        // sk03_fa_qkv.py:164:45
	shr.u32 	%r66, %r2, 3;
	bfe.u32 	%r67, %r2, 3, 5;
	shl.b32 	%r7, %r2, 1;
	and.b32 	%r68, %r7, 6;
	and.b32 	%r8, %r2, 224;
	shr.u32 	%r69, %r8, 2;
	or.b32 	%r70, %r68, %r69;
	and.b32 	%r9, %r2, 15;
	shl.b32 	%r71, %r9, 3;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r72, %r6, %r67;
	or.b32 	%r73, %r72, 32;
	or.b32 	%r74, %r72, 64;
	or.b32 	%r75, %r66, %r6;
	or.b32 	%r76, %r75, 96;
	or.b32 	%r77, %r6, %r70;
	or.b32 	%r79, %r77, 64;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r81, %r72, %r21;
	rem.s32 	%r82, %r73, %r21;
	rem.s32 	%r83, %r74, %r21;
	rem.s32 	%r84, %r76, %r21;
	rem.s32 	%r10, %r77, %r21;
	rem.s32 	%r11, %r79, %r21;
	.loc	1 167 39                        // sk03_fa_qkv.py:167:39
	mul.lo.s32 	%r87, %r5, %r56;
	.loc	1 167 21                        // sk03_fa_qkv.py:167:21
	cvt.s64.s32 	%rd1, %r87;
	add.s64 	%rd36, %rd19, %rd1;
	.loc	1 167 51                        // sk03_fa_qkv.py:167:51
	cvt.u64.u32 	%rd2, %r71;
	add.s64 	%rd25, %rd36, %rd2;
	.loc	1 168 28                        // sk03_fa_qkv.py:168:28
	and.b32 	%r12, %r2, 7;
	shl.b32 	%r88, %r12, 4;
	.loc	1 168 21                        // sk03_fa_qkv.py:168:21
	cvt.u64.u32 	%rd3, %r88;
	add.s64 	%rd37, %rd20, %rd3;
	.loc	1 168 69                        // sk03_fa_qkv.py:168:69
	mul.lo.s32 	%r89, %r81, %r58;
	mul.lo.s32 	%r90, %r82, %r58;
	mul.lo.s32 	%r91, %r83, %r58;
	mul.lo.s32 	%r92, %r84, %r58;
	.loc	1 168 51                        // sk03_fa_qkv.py:168:51
	cvt.s64.s32 	%rd4, %r89;
	add.s64 	%rd26, %rd37, %rd4;
	cvt.s64.s32 	%rd5, %r90;
	add.s64 	%rd27, %rd37, %rd5;
	cvt.s64.s32 	%rd6, %r91;
	add.s64 	%rd28, %rd37, %rd6;
	cvt.s64.s32 	%rd7, %r92;
	add.s64 	%rd29, %rd37, %rd7;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	setp.lt.s32 	%p1, %r22, 128;
	setp.gt.s32 	%p2, %r22, 127;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
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
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
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
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	setp.gt.s32 	%p3, %r22, 255;
	.loc	1 183 18                        // sk03_fa_qkv.py:183:18
	add.s64 	%rd30, %rd25, 128;
	.loc	1 184 18                        // sk03_fa_qkv.py:184:18
	add.s64 	%rd31, %rd26, 128;
	add.s64 	%rd32, %rd27, 128;
	add.s64 	%rd33, %rd28, 128;
	add.s64 	%rd34, %rd29, 128;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
	bar.sync 	0;
	add.s32 	%r34, %r13, 34816;
	selp.b32 	%r35, 8, 0, %p3;
	// begin inline asm
	cp.async.ca.shared.global [ %r34 + 0 ], [ %rd30 + 0 ], 0x8, %r35;
	// end inline asm
	cp.async.commit_group;
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
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
	mov.b32 	%r321, 0;
	cvt.u32.u64 	%r309, %rd3;
	mov.b32 	%r322, %r321;
	mov.b32 	%r323, %r321;
	mov.b32 	%r324, %r321;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	@%p1 bra 	$L__BB0_4;
// %bb.1:                               // %.lr.ph
	.loc	1 0 23                          // sk03_fa_qkv.py:0:23
	ld.param.b32 	%r26, [_sk03_fa_qkv_kernel_param_15];
	ld.param.b64 	%rd35, [_sk03_fa_qkv_kernel_param_6];
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
	.loc	1 178 28                        // sk03_fa_qkv.py:178:28
	shr.u32 	%r118, %r22, 7;
	add.s32 	%r119, %r118, -2;
	shl.b32 	%r120, %r9, 7;
	and.b32 	%r121, %r2, 16;
	xor.b32 	%r122, %r309, %r121;
	or.b32 	%r14, %r122, %r120;
	xor.b32 	%r15, %r14, 32;
	xor.b32 	%r16, %r14, 64;
	xor.b32 	%r17, %r14, 96;
	shl.b32 	%r123, %r12, 7;
	shl.b32 	%r124, %r8, 5;
	and.b32 	%r125, %r7, 48;
	or.b32 	%r126, %r123, %r124;
	xor.b32 	%r127, %r309, %r125;
	or.b32 	%r18, %r126, %r127;
	xor.b32 	%r19, %r18, 64;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	cvt.s64.s32 	%rd12, %r119;
	and.b32 	%r128, %r22, -128;
	cvt.u64.u32 	%rd13, %r128;
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
	mov.b32 	%r117, 0;
	mov.b32 	%r312, 1;
	mov.b32 	%r311, -1;
	mov.b64 	%rd77, 0;
	mov.b32 	%r310, %r117;
	mov.b64 	%rd78, %rd77;
	mov.b32 	%r313, %r117;
	mov.b32 	%r314, %r117;
	mov.b32 	%r315, %r117;
	mov.b32 	%r316, %r117;
	mov.b32 	%r317, %r117;
	mov.b32 	%r318, %r117;
	mov.b32 	%r319, %r117;
	mov.b32 	%r320, %r117;
$L__BB0_2:                              // =>This Inner Loop Header: Depth=1
	setp.lt.s64 	%p4, %rd78, %rd12;
	add.s32 	%r176, %r311, 1;
	setp.gt.s32 	%p5, %r176, 1;
	selp.b32 	%r311, 0, %r176, %p5;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
	cp.async.wait_group 	2;
	bar.sync 	0;
	shl.b32 	%r177, %r311, 11;
	add.s32 	%r178, %r113, %r177;
	add.s32 	%r179, %r178, %r14;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r129, %r130, %r131, %r132}, [%r179+32768];
	add.s32 	%r180, %r178, %r15;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r141, %r142, %r143, %r144}, [%r180+32768];
	add.s32 	%r181, %r178, %r16;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r153, %r154, %r155, %r156}, [%r181+32768];
	add.s32 	%r182, %r178, %r17;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r161, %r162, %r163, %r164}, [%r182+32768];
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	shl.b32 	%r183, %r311, 14;
	add.s32 	%r184, %r113, %r183;
	add.s32 	%r185, %r184, %r18;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r133, %r134, %r145, %r146}, [%r185];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r135, %r136, %r151, %r152}, [%r185+8192];
	add.s32 	%r186, %r184, %r19;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r157, %r158, %r165, %r166}, [%r186];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r159, %r160, %r167, %r168}, [%r186+8192];
	.loc	1 179 36                        // sk03_fa_qkv.py:179:36
	mov.b32 	%r137, %r117;
	mov.b32 	%r138, %r117;
	mov.b32 	%r139, %r117;
	mov.b32 	%r140, %r117;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r137, %r138, %r139, %r140 }, { %r129, %r130, %r131, %r132 }, { %r133, %r134 }, { %r137, %r138, %r139, %r140 };
	// end inline asm
	mov.b32 	%r147, %r117;
	mov.b32 	%r148, %r117;
	mov.b32 	%r149, %r117;
	mov.b32 	%r150, %r117;
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
	.loc	1 181 39                        // sk03_fa_qkv.py:181:39
	cvt.s64.s32 	%rd61, %r310;
	add.s64 	%rd52, %rd8, %rd61;
	add.s64 	%rd53, %rd9, %rd61;
	add.s64 	%rd54, %rd10, %rd61;
	add.s64 	%rd55, %rd11, %rd61;
	.loc	1 181 29                        // sk03_fa_qkv.py:181:29
	// begin inline asm
	mov.u16 %rs1, 0x0;
	ld.global.b8 { %rs1 }, [ %rd52 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs2, 0x0;
	ld.global.b8 { %rs2 }, [ %rd53 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs3, 0x0;
	ld.global.b8 { %rs3 }, [ %rd54 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs4, 0x0;
	ld.global.b8 { %rs4 }, [ %rd55 + 0 ];
	// end inline asm
	.loc	1 181 21                        // sk03_fa_qkv.py:181:21
	cvt.u32.u16 	%r187, %rs1;
	and.b32 	%r188, %r187, 255;
	cvt.u32.u16 	%r189, %rs2;
	and.b32 	%r190, %r189, 255;
	cvt.u32.u16 	%r191, %rs3;
	and.b32 	%r192, %r191, 255;
	cvt.u32.u16 	%r193, %rs4;
	and.b32 	%r194, %r193, 255;
	shl.b32 	%r195, %r148, %r194;
	shl.b32 	%r196, %r150, %r194;
	shl.b32 	%r197, %r147, %r192;
	shl.b32 	%r198, %r149, %r192;
	shl.b32 	%r199, %r138, %r190;
	shl.b32 	%r200, %r140, %r190;
	shl.b32 	%r201, %r137, %r188;
	shl.b32 	%r202, %r139, %r188;
	.loc	1 182 15                        // sk03_fa_qkv.py:182:15
	add.s32 	%r315, %r202, %r315;
	add.s32 	%r313, %r201, %r313;
	add.s32 	%r316, %r200, %r316;
	add.s32 	%r314, %r199, %r314;
	add.s32 	%r319, %r198, %r319;
	add.s32 	%r317, %r197, %r317;
	add.s32 	%r320, %r196, %r320;
	add.s32 	%r318, %r195, %r318;
	.loc	1 184 18                        // sk03_fa_qkv.py:184:18
	add.s64 	%rd56, %rd18, %rd77;
	add.s64 	%rd57, %rd17, %rd77;
	add.s64 	%rd58, %rd16, %rd77;
	add.s64 	%rd59, %rd15, %rd77;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	add.s64 	%rd60, %rd14, %rd77;
	add.s32 	%r203, %r312, 1;
	setp.gt.s32 	%p6, %r203, 1;
	selp.b32 	%r312, 0, %r203, %p6;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
	shl.b32 	%r204, %r312, 11;
	bar.sync 	0;
	add.s32 	%r205, %r13, %r204;
	add.s32 	%r169, %r205, 32768;
	selp.b32 	%r170, 8, 0, %p4;
	// begin inline asm
	cp.async.ca.shared.global [ %r169 + 0 ], [ %rd56 + 0 ], 0x8, %r170;
	// end inline asm
	cp.async.commit_group;
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	shl.b32 	%r206, %r312, 14;
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
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	add.s64 	%rd78, %rd78, 1;
	add.s64 	%rd77, %rd77, 128;
	add.s32 	%r310, %r310, %r26;
	setp.ne.b64 	%p7, %rd13, %rd77;
	@%p7 bra 	$L__BB0_2;
// %bb.3:                               // %._crit_edge.loopexit
	.loc	1 186 17                        // sk03_fa_qkv.py:186:17
	cvt.rn.f32.s32 	%r207, %r313;
	cvt.rn.f32.s32 	%r208, %r314;
	cvt.rn.bf16x2.f32 	%r321, %r208, %r207;
	cvt.rn.f32.s32 	%r209, %r315;
	cvt.rn.f32.s32 	%r210, %r316;
	cvt.rn.bf16x2.f32 	%r322, %r210, %r209;
	cvt.rn.f32.s32 	%r211, %r317;
	cvt.rn.f32.s32 	%r212, %r318;
	cvt.rn.bf16x2.f32 	%r323, %r212, %r211;
	cvt.rn.f32.s32 	%r213, %r319;
	cvt.rn.f32.s32 	%r214, %r320;
	cvt.rn.bf16x2.f32 	%r324, %r214, %r213;
$L__BB0_4:                              // %._crit_edge
	.loc	1 0 17                          // sk03_fa_qkv.py:0:17
	cvt.u32.u64 	%r234, %rd2;
	.loc	1 164 45                        // sk03_fa_qkv.py:164:45
	or.b32 	%r235, %r6, %r234;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r236, %r235, 7;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r237, %r236, %r21;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r238, %r235, 6;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r239, %r238, %r21;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r240, %r235, 5;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r241, %r240, %r21;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r242, %r235, 4;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r243, %r242, %r21;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r244, %r235, 3;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r245, %r244, %r21;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r246, %r235, 2;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r247, %r246, %r21;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r248, %r235, 1;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r249, %r248, %r21;
	rem.s32 	%r250, %r235, %r21;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	and.b32 	%r251, %r2, 28;
	bfe.u32 	%r252, %r2, 2, 3;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r253, %r252, %r1;
	or.b32 	%r254, %r253, 8;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r255, %r254, %r20;
	rem.s32 	%r256, %r253, %r20;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 186 54                        // sk03_fa_qkv.py:186:54
	mad.wide.s32 	%rd62, %r256, 4, %rd23;
	mad.wide.s32 	%rd63, %r255, 4, %rd23;
	.loc	1 186 40                        // sk03_fa_qkv.py:186:40
	// begin inline asm
	mov.u32 %r215, 0x0;
	ld.global.b32 { %r215 }, [ %rd62 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r216, 0x0;
	ld.global.b32 { %r216 }, [ %rd63 + 0 ];
	// end inline asm
	.loc	1 186 65                        // sk03_fa_qkv.py:186:65
	cvt.rn.bf16.f32 	%rs21, %r215;
	cvt.rn.bf16.f32 	%rs22, %r216;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	mov.b32 	{%rs23, %rs24}, %r321;
	mov.b16 	%rs25, 0x8000;
	fma.rn.bf16 	%rs26, %rs23, %rs21, %rs25;
	fma.rn.bf16 	%rs27, %rs24, %rs21, %rs25;
	mov.b32 	{%rs28, %rs29}, %r322;
	fma.rn.bf16 	%rs30, %rs28, %rs22, %rs25;
	fma.rn.bf16 	%rs31, %rs29, %rs22, %rs25;
	mov.b32 	{%rs32, %rs33}, %r323;
	fma.rn.bf16 	%rs34, %rs32, %rs21, %rs25;
	fma.rn.bf16 	%rs35, %rs33, %rs21, %rs25;
	mov.b32 	{%rs36, %rs37}, %r324;
	fma.rn.bf16 	%rs38, %rs36, %rs22, %rs25;
	fma.rn.bf16 	%rs39, %rs37, %rs22, %rs25;
	.loc	1 187 38                        // sk03_fa_qkv.py:187:38
	mad.wide.s32 	%rd64, %r10, 4, %rd24;
	mad.wide.s32 	%rd65, %r11, 4, %rd24;
	.loc	1 187 24                        // sk03_fa_qkv.py:187:24
	// begin inline asm
	mov.u32 %r217, 0x0;
	mov.u32 %r218, 0x0;
	ld.global.v2.b32 { %r217, %r218 }, [ %rd64 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r219, 0x0;
	mov.u32 %r220, 0x0;
	ld.global.v2.b32 { %r219, %r220 }, [ %rd65 + 0 ];
	// end inline asm
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs40, %r217;
	cvt.rn.bf16.f32 	%rs41, %r218;
	cvt.rn.bf16.f32 	%rs42, %r219;
	cvt.rn.bf16.f32 	%rs43, %r220;
	.loc	1 188 49                        // sk03_fa_qkv.py:188:49
	mul.lo.s32 	%r257, %r5, %r24;
	.loc	1 188 31                        // sk03_fa_qkv.py:188:31
	mad.wide.s32 	%rd75, %r257, 2, %rd22;
	.loc	1 188 82                        // sk03_fa_qkv.py:188:82
	mul.lo.s32 	%r258, %r250, %r25;
	mul.lo.s32 	%r259, %r249, %r25;
	mul.lo.s32 	%r260, %r247, %r25;
	mul.lo.s32 	%r261, %r245, %r25;
	mul.lo.s32 	%r262, %r243, %r25;
	mul.lo.s32 	%r263, %r241, %r25;
	mul.lo.s32 	%r264, %r239, %r25;
	mul.lo.s32 	%r265, %r237, %r25;
	.loc	1 188 64                        // sk03_fa_qkv.py:188:64
	mad.wide.s32 	%rd66, %r258, 2, %rd75;
	mad.wide.s32 	%rd67, %r259, 2, %rd75;
	mad.wide.s32 	%rd68, %r260, 2, %rd75;
	mad.wide.s32 	%rd69, %r261, 2, %rd75;
	mad.wide.s32 	%rd70, %r262, 2, %rd75;
	mad.wide.s32 	%rd71, %r263, 2, %rd75;
	mad.wide.s32 	%rd72, %r264, 2, %rd75;
	mad.wide.s32 	%rd73, %r265, 2, %rd75;
	.loc	1 188 19                        // sk03_fa_qkv.py:188:19
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
	// begin inline asm
	mov.u16 %rs9, 0x0;
	ld.global.b16 { %rs9 }, [ %rd70 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs10, 0x0;
	ld.global.b16 { %rs10 }, [ %rd71 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs11, 0x0;
	ld.global.b16 { %rs11 }, [ %rd72 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs12, 0x0;
	ld.global.b16 { %rs12 }, [ %rd73 + 0 ];
	// end inline asm
	and.b32 	%r266, %r2, 120;
	shl.b32 	%r267, %r266, 5;
	or.b32 	%r268, %r267, %r309;
	xor.b32 	%r269, %r268, %r3;
	add.s32 	%r221, %r113, %r269;
	mov.b32 	%r222, {%rs5, %rs6};
	mov.b32 	%r223, {%rs7, %rs8};
	mov.b32 	%r224, {%rs9, %rs10};
	mov.b32 	%r225, {%rs11, %rs12};
	// begin inline asm
	st.shared.v4.b32 [ %r221 + 0 ], { %r222, %r223, %r224, %r225 };
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
	mov.b32 	{%rs44, %rs45}, %r277;
	mov.b32 	{%rs46, %rs47}, %r278;
	mov.b32 	{%rs48, %rs49}, %r279;
	mov.b32 	{%rs50, %rs51}, %r280;
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs13, %rs26, %rs40, %rs44;
	fma.rn.bf16 	%rs14, %rs27, %rs41, %rs45;
	fma.rn.bf16 	%rs15, %rs30, %rs40, %rs46;
	fma.rn.bf16 	%rs16, %rs31, %rs41, %rs47;
	fma.rn.bf16 	%rs17, %rs34, %rs42, %rs48;
	fma.rn.bf16 	%rs18, %rs35, %rs43, %rs49;
	fma.rn.bf16 	%rs19, %rs38, %rs42, %rs50;
	fma.rn.bf16 	%rs20, %rs39, %rs43, %rs51;
	bar.sync 	0;
	shl.b32 	%r281, %r2, 5;
	and.b32 	%r282, %r281, 768;
	shl.b32 	%r283, %r251, 1;
	and.b32 	%r284, %r2, 1;
	neg.s32 	%r285, %r284;
	and.b32 	%r286, %r285, 1088;
	bfe.s32 	%r287, %r2, 1, 1;
	and.b32 	%r288, %r287, 2052;
	or.b32 	%r289, %r282, %r283;
	or.b32 	%r290, %r286, %r289;
	xor.b32 	%r291, %r290, %r273;
	or.b32 	%r292, %r291, %r288;
	add.s32 	%r226, %r113, %r292;
	// begin inline asm
	st.shared.v2.b16 [ %r226 + 0 ], { %rs13, %rs14 };
	// end inline asm
	add.s32 	%r227, %r226, 128;
	// begin inline asm
	st.shared.v2.b16 [ %r227 + 0 ], { %rs15, %rs16 };
	// end inline asm
	xor.b32 	%r293, %r292, 4;
	add.s32 	%r228, %r113, %r293;
	// begin inline asm
	st.shared.v2.b16 [ %r228 + 0 ], { %rs17, %rs18 };
	// end inline asm
	add.s32 	%r229, %r228, 128;
	// begin inline asm
	st.shared.v2.b16 [ %r229 + 0 ], { %rs19, %rs20 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r294, %r2, 3;
	and.b32 	%r295, %r294, 768;
	shr.u32 	%r296, %r266, 1;
	and.b32 	%r297, %r2, 128;
	or.b32 	%r298, %r309, %r295;
	xor.b32 	%r299, %r298, %r296;
	or.b32 	%r300, %r299, %r297;
	add.s32 	%r301, %r113, %r300;
	ld.shared.b32 	%r230, [%r301];
	xor.b32 	%r302, %r300, 64;
	add.s32 	%r303, %r113, %r302;
	ld.shared.b32 	%r231, [%r303+1024];
	xor.b32 	%r304, %r300, 4;
	add.s32 	%r305, %r113, %r304;
	ld.shared.b32 	%r232, [%r305+2048];
	xor.b32 	%r306, %r300, 68;
	add.s32 	%r307, %r113, %r306;
	ld.shared.b32 	%r233, [%r307+3072];
	.loc	1 195 31                        // sk03_fa_qkv.py:195:31
	setp.lt.s32 	%p9, %r4, %r20;
	.loc	1 195 54                        // sk03_fa_qkv.py:195:54
	setp.lt.s32 	%p10, %r235, %r21;
	.loc	1 195 37                        // sk03_fa_qkv.py:195:37
	and.pred 	%p8, %p9, %p10;
	.loc	1 193 35                        // sk03_fa_qkv.py:193:35
	mul.lo.s32 	%r308, %r4, %r23;
	.loc	1 193 18                        // sk03_fa_qkv.py:193:18
	mad.wide.s32 	%rd76, %r308, 2, %rd21;
	.loc	1 193 50                        // sk03_fa_qkv.py:193:50
	mad.wide.s32 	%rd74, %r235, 2, %rd76;
	.loc	1 194 8                         // sk03_fa_qkv.py:194:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd74 + 0 ], { %r230, %r231, %r232, %r233 };
	// end inline asm
	.loc	1 192 4                         // sk03_fa_qkv.py:192:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk03_fa_qkv.py"
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
.b32 157                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x96 DW_TAG_compile_unit
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
.b8 51
.b8 95
.b8 102
.b8 97
.b8 95
.b8 113
.b8 107
.b8 118
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
.b8 2                                   // Abbrev [2] 0x44:0x16 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 51
.b8 95
.b8 102
.b8 97
.b8 95
.b8 113
.b8 107
.b8 118
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x5a:0x46 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 68                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x6f:0x18 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 155                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x87:0x18 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 156                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_3 = _Nativo(
    "sk03_fa_qkv/tile16x128x128_shift1_abi16",
    _PTX_3, "_sk03_fa_qkv_kernel",
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

	// .globl	_sk03_fa_qkv_kernel     // -- Begin function _sk03_fa_qkv_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk03_fa_qkv_kernel
.visible .entry _sk03_fa_qkv_kernel(
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_6,
	.param .u32 _sk03_fa_qkv_kernel_param_7,
	.param .u32 _sk03_fa_qkv_kernel_param_8,
	.param .u32 _sk03_fa_qkv_kernel_param_9,
	.param .u32 _sk03_fa_qkv_kernel_param_10,
	.param .u32 _sk03_fa_qkv_kernel_param_11,
	.param .u32 _sk03_fa_qkv_kernel_param_12,
	.param .u32 _sk03_fa_qkv_kernel_param_13,
	.param .u32 _sk03_fa_qkv_kernel_param_14,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_16
)
.reqntid 256
{
	.reg .pred 	%p<25>;
	.reg .b16 	%rs<242>;
	.reg .b32 	%r<631>;
	.reg .b64 	%rd<104>;
	.loc	1 145 0                         // sk03_fa_qkv.py:145:0
$L__func_begin0:
	.loc	1 145 0                         // sk03_fa_qkv.py:145:0

// %bb.0:
	ld.param.b32 	%r25, [_sk03_fa_qkv_kernel_param_13];
	ld.param.b32 	%r24, [_sk03_fa_qkv_kernel_param_12];
	ld.param.b32 	%r23, [_sk03_fa_qkv_kernel_param_8];
	ld.param.b32 	%r22, [_sk03_fa_qkv_kernel_param_7];
	ld.param.b64 	%rd15, [_sk03_fa_qkv_kernel_param_5];
	ld.param.b64 	%rd14, [_sk03_fa_qkv_kernel_param_4];
	ld.param.b64 	%rd13, [_sk03_fa_qkv_kernel_param_3];
	ld.param.b64 	%rd12, [_sk03_fa_qkv_kernel_param_2];
	ld.param.b64 	%rd11, [_sk03_fa_qkv_kernel_param_1];
	ld.param.b64 	%rd10, [_sk03_fa_qkv_kernel_param_0];
$L__tmp0:
	.loc	1 154 24                        // sk03_fa_qkv.py:154:24
	mov.u32 	%r44, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk03_fa_qkv.py:155:27 ]
	add.s32 	%r45, %r22, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk03_fa_qkv.py:155:27 ]
	shr.s32 	%r46, %r45, 31;
	shr.u32 	%r47, %r46, 25;
	add.s32 	%r48, %r45, %r47;
	shr.s32 	%r49, %r48, 7;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk03_fa_qkv.py:156:27 ]
	add.s32 	%r50, %r23, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk03_fa_qkv.py:156:27 ]
	shr.s32 	%r51, %r50, 31;
	shr.u32 	%r52, %r51, 25;
	add.s32 	%r53, %r50, %r52;
	shr.s32 	%r54, %r53, 7;
$L__tmp3:
	.loc	1 157 29                        // sk03_fa_qkv.py:157:29
	shl.b32 	%r55, %r54, 3;
	.loc	1 158 22                        // sk03_fa_qkv.py:158:22
	div.s32 	%r56, %r44, %r55;
	.loc	1 158 38                        // sk03_fa_qkv.py:158:38
	shl.b32 	%r57, %r56, 3;
	ld.param.b32 	%r58, [_sk03_fa_qkv_kernel_param_9];
	.loc	1 159 30                        // sk03_fa_qkv.py:159:30
	sub.s32 	%r59, %r49, %r57;
	ld.param.b32 	%r60, [_sk03_fa_qkv_kernel_param_10];
	.loc	1 159 39                        // sk03_fa_qkv.py:159:39
	min.s32 	%r61, %r59, 8;
	ld.param.b32 	%r62, [_sk03_fa_qkv_kernel_param_11];
	.loc	1 160 30                        // sk03_fa_qkv.py:160:30
	mul.lo.s32 	%r63, %r56, %r55;
	sub.s32 	%r64, %r44, %r63;
	.loc	1 161 36                        // sk03_fa_qkv.py:161:36
	div.s32 	%r65, %r64, %r61;
	.loc	1 160 46                        // sk03_fa_qkv.py:160:46
	mul.lo.s32 	%r66, %r65, %r61;
	sub.s32 	%r67, %r64, %r66;
	.loc	1 160 23                        // sk03_fa_qkv.py:160:23
	add.s32 	%r68, %r67, %r57;
	.loc	1 163 22                        // sk03_fa_qkv.py:163:22
	shl.b32 	%r1, %r68, 7;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	mov.u32 	%r2, %tid.x;
	and.b32 	%r3, %r2, 248;
	bfe.u32 	%r69, %r2, 3, 5;
	or.b32 	%r70, %r69, 32;
	or.b32 	%r71, %r69, 64;
	or.b32 	%r72, %r69, 96;
	and.b32 	%r4, %r2, 96;
	and.b32 	%r5, %r2, 7;
	shl.b32 	%r73, %r5, 4;
	and.b32 	%r6, %r2, 15;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r74, %r1, %r69;
	or.b32 	%r75, %r1, %r70;
	or.b32 	%r76, %r1, %r71;
	or.b32 	%r77, %r1, %r72;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r7, %r74, %r22;
	rem.s32 	%r8, %r75, %r22;
	rem.s32 	%r9, %r76, %r22;
	rem.s32 	%r10, %r77, %r22;
	.loc	1 164 22                        // sk03_fa_qkv.py:164:22
	shl.b32 	%r11, %r65, 7;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r78, %r11, %r69;
	or.b32 	%r79, %r11, %r70;
	or.b32 	%r80, %r11, %r71;
	or.b32 	%r81, %r11, %r72;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r82, %r78, %r23;
	rem.s32 	%r83, %r79, %r23;
	rem.s32 	%r84, %r80, %r23;
	rem.s32 	%r85, %r81, %r23;
	.loc	1 167 39                        // sk03_fa_qkv.py:167:39
	mul.lo.s32 	%r86, %r7, %r60;
	mul.lo.s32 	%r87, %r8, %r60;
	mul.lo.s32 	%r88, %r9, %r60;
	mul.lo.s32 	%r89, %r10, %r60;
	.loc	1 167 21                        // sk03_fa_qkv.py:167:21
	cvt.s64.s32 	%rd1, %r86;
	add.s64 	%rd32, %rd10, %rd1;
	cvt.s64.s32 	%rd2, %r87;
	add.s64 	%rd33, %rd10, %rd2;
	cvt.s64.s32 	%rd3, %r88;
	add.s64 	%rd34, %rd10, %rd3;
	cvt.s64.s32 	%rd4, %r89;
	add.s64 	%rd35, %rd10, %rd4;
	.loc	1 167 51                        // sk03_fa_qkv.py:167:51
	cvt.u64.u32 	%rd5, %r73;
	add.s64 	%rd16, %rd32, %rd5;
	add.s64 	%rd17, %rd33, %rd5;
	add.s64 	%rd18, %rd34, %rd5;
	add.s64 	%rd19, %rd35, %rd5;
	.loc	1 168 21                        // sk03_fa_qkv.py:168:21
	add.s64 	%rd36, %rd11, %rd5;
	.loc	1 168 69                        // sk03_fa_qkv.py:168:69
	mul.lo.s32 	%r90, %r82, %r62;
	mul.lo.s32 	%r91, %r83, %r62;
	mul.lo.s32 	%r92, %r84, %r62;
	mul.lo.s32 	%r93, %r85, %r62;
	.loc	1 168 51                        // sk03_fa_qkv.py:168:51
	cvt.s64.s32 	%rd6, %r90;
	add.s64 	%rd20, %rd36, %rd6;
	cvt.s64.s32 	%rd7, %r91;
	add.s64 	%rd21, %rd36, %rd7;
	cvt.s64.s32 	%rd8, %r92;
	add.s64 	%rd22, %rd36, %rd8;
	cvt.s64.s32 	%rd9, %r93;
	add.s64 	%rd23, %rd36, %rd9;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	setp.gt.s32 	%p1, %r58, 127;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
	shl.b32 	%r13, %r2, 4;
	and.b32 	%r97, %r13, 4080;
	and.b32 	%r14, %r2, 56;
	shl.b32 	%r98, %r14, 1;
	xor.b32 	%r99, %r97, %r98;
	mov.b32 	%r100, global_smem;
	add.s32 	%r26, %r100, %r99;
	selp.b32 	%r27, 16, 0, %p1;
	// begin inline asm
	cp.async.cg.shared.global [ %r26 + 0 ], [ %rd16 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r28, %r26, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r28 + 0 ], [ %rd17 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r29, %r26, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r29 + 0 ], [ %rd18 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r30, %r26, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r30 + 0 ], [ %rd19 + 0 ], 0x10, %r27;
	// end inline asm
	cp.async.commit_group;
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	add.s32 	%r31, %r26, 32768;
	// begin inline asm
	cp.async.cg.shared.global [ %r31 + 0 ], [ %rd20 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r32, %r26, 36864;
	// begin inline asm
	cp.async.cg.shared.global [ %r32 + 0 ], [ %rd21 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r33, %r26, 40960;
	// begin inline asm
	cp.async.cg.shared.global [ %r33 + 0 ], [ %rd22 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r34, %r26, 45056;
	// begin inline asm
	cp.async.cg.shared.global [ %r34 + 0 ], [ %rd23 + 0 ], 0x10, %r27;
	// end inline asm
	cp.async.commit_group;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	setp.gt.s32 	%p2, %r58, 255;
	.loc	1 183 18                        // sk03_fa_qkv.py:183:18
	add.s64 	%rd24, %rd16, 128;
	add.s64 	%rd25, %rd17, 128;
	add.s64 	%rd26, %rd18, 128;
	add.s64 	%rd27, %rd19, 128;
	.loc	1 184 18                        // sk03_fa_qkv.py:184:18
	add.s64 	%rd28, %rd20, 128;
	add.s64 	%rd29, %rd21, 128;
	add.s64 	%rd30, %rd22, 128;
	add.s64 	%rd31, %rd23, 128;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
	bar.sync 	0;
	add.s32 	%r35, %r26, 16384;
	selp.b32 	%r36, 16, 0, %p2;
	// begin inline asm
	cp.async.cg.shared.global [ %r35 + 0 ], [ %rd24 + 0 ], 0x10, %r36;
	// end inline asm
	add.s32 	%r37, %r26, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r37 + 0 ], [ %rd25 + 0 ], 0x10, %r36;
	// end inline asm
	add.s32 	%r38, %r26, 24576;
	// begin inline asm
	cp.async.cg.shared.global [ %r38 + 0 ], [ %rd26 + 0 ], 0x10, %r36;
	// end inline asm
	add.s32 	%r39, %r26, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r39 + 0 ], [ %rd27 + 0 ], 0x10, %r36;
	// end inline asm
	cp.async.commit_group;
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	add.s32 	%r40, %r26, 49152;
	// begin inline asm
	cp.async.cg.shared.global [ %r40 + 0 ], [ %rd28 + 0 ], 0x10, %r36;
	// end inline asm
	add.s32 	%r41, %r26, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r41 + 0 ], [ %rd29 + 0 ], 0x10, %r36;
	// end inline asm
	add.s32 	%r42, %r26, 57344;
	// begin inline asm
	cp.async.cg.shared.global [ %r42 + 0 ], [ %rd30 + 0 ], 0x10, %r36;
	// end inline asm
	add.s32 	%r43, %r26, 61440;
	// begin inline asm
	cp.async.cg.shared.global [ %r43 + 0 ], [ %rd31 + 0 ], 0x10, %r36;
	// end inline asm
	cp.async.commit_group;
	cvt.u32.u64 	%r557, %rd5;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	@%p1 bra 	$L__BB0_2;
	bra.uni 	$L__BB0_1;
$L__BB0_2:                              // %.lr.ph
	.loc	1 0 23                          // sk03_fa_qkv.py:0:23
	shr.s32 	%r94, %r58, 31;
	shr.u32 	%r95, %r94, 25;
	add.s32 	%r96, %r58, %r95;
	shr.s32 	%r12, %r96, 7;
	add.s32 	%r15, %r12, -2;
	shl.b32 	%r101, %r6, 7;
	and.b32 	%r102, %r13, 2160;
	and.b32 	%r625, %r2, 16;
	or.b32 	%r103, %r101, %r102;
	xor.b32 	%r16, %r103, %r625;
	xor.b32 	%r17, %r16, 32;
	xor.b32 	%r18, %r16, 64;
	xor.b32 	%r19, %r16, 96;
	shl.b32 	%r104, %r5, 7;
	shl.b32 	%r105, %r4, 5;
	shl.b32 	%r626, %r2, 1;
	and.b32 	%r106, %r626, 48;
	or.b32 	%r107, %r104, %r105;
	xor.b32 	%r108, %r557, %r106;
	or.b32 	%r20, %r107, %r108;
	xor.b32 	%r21, %r20, 64;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	add.s64 	%rd37, %rd9, %rd11;
	add.s64 	%rd103, %rd37, 256;
	add.s64 	%rd38, %rd8, %rd11;
	add.s64 	%rd102, %rd38, 256;
	add.s64 	%rd39, %rd7, %rd11;
	add.s64 	%rd101, %rd39, 256;
	add.s64 	%rd40, %rd6, %rd11;
	add.s64 	%rd100, %rd40, 256;
	add.s64 	%rd41, %rd4, %rd10;
	add.s64 	%rd99, %rd41, 256;
	add.s64 	%rd42, %rd3, %rd10;
	add.s64 	%rd98, %rd42, 256;
	add.s64 	%rd43, %rd2, %rd10;
	add.s64 	%rd97, %rd43, 256;
	add.s64 	%rd44, %rd1, %rd10;
	add.s64 	%rd96, %rd44, 256;
	mov.b32 	%r560, 0;
	mov.b32 	%r559, 1;
	mov.b32 	%r558, -1;
	mov.b32 	%r561, %r560;
	mov.b32 	%r562, %r560;
	mov.b32 	%r563, %r560;
	mov.b32 	%r564, %r560;
	mov.b32 	%r565, %r560;
	mov.b32 	%r566, %r560;
	mov.b32 	%r567, %r560;
	mov.b32 	%r568, %r560;
	mov.b32 	%r569, %r560;
	mov.b32 	%r570, %r560;
	mov.b32 	%r571, %r560;
	mov.b32 	%r572, %r560;
	mov.b32 	%r573, %r560;
	mov.b32 	%r574, %r560;
	mov.b32 	%r575, %r560;
	mov.b32 	%r576, %r560;
	mov.b32 	%r577, %r560;
	mov.b32 	%r578, %r560;
	mov.b32 	%r579, %r560;
	mov.b32 	%r580, %r560;
	mov.b32 	%r581, %r560;
	mov.b32 	%r582, %r560;
	mov.b32 	%r583, %r560;
	mov.b32 	%r584, %r560;
	mov.b32 	%r585, %r560;
	mov.b32 	%r586, %r560;
	mov.b32 	%r587, %r560;
	mov.b32 	%r588, %r560;
	mov.b32 	%r589, %r560;
	mov.b32 	%r590, %r560;
	mov.b32 	%r591, %r560;
	mov.b32 	%r592, %r560;
	mov.b32 	%r593, %r560;
	mov.b32 	%r594, %r560;
	mov.b32 	%r595, %r560;
	mov.b32 	%r596, %r560;
	mov.b32 	%r597, %r560;
	mov.b32 	%r598, %r560;
	mov.b32 	%r599, %r560;
	mov.b32 	%r600, %r560;
	mov.b32 	%r601, %r560;
	mov.b32 	%r602, %r560;
	mov.b32 	%r603, %r560;
	mov.b32 	%r604, %r560;
	mov.b32 	%r605, %r560;
	mov.b32 	%r606, %r560;
	mov.b32 	%r607, %r560;
	mov.b32 	%r608, %r560;
	mov.b32 	%r609, %r560;
	mov.b32 	%r610, %r560;
	mov.b32 	%r611, %r560;
	mov.b32 	%r612, %r560;
	mov.b32 	%r613, %r560;
	mov.b32 	%r614, %r560;
	mov.b32 	%r615, %r560;
	mov.b32 	%r616, %r560;
	mov.b32 	%r617, %r560;
	mov.b32 	%r618, %r560;
	mov.b32 	%r619, %r560;
	mov.b32 	%r620, %r560;
	mov.b32 	%r621, %r560;
	mov.b32 	%r622, %r560;
	mov.b32 	%r623, %r560;
	mov.b32 	%r624, %r560;
$L__BB0_3:                              // =>This Inner Loop Header: Depth=1
	setp.lt.s32 	%p3, %r624, %r15;
	add.s32 	%r214, %r558, 1;
	setp.gt.s32 	%p4, %r214, 1;
	selp.b32 	%r558, 0, %r214, %p4;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
	cp.async.wait_group 	2;
	bar.sync 	0;
	shl.b32 	%r215, %r558, 14;
	add.s32 	%r216, %r100, %r215;
	add.s32 	%r217, %r216, %r16;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r109, %r110, %r111, %r112}, [%r217];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r121, %r122, %r123, %r124}, [%r217+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r125, %r126, %r127, %r128}, [%r217+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r129, %r130, %r131, %r132}, [%r217+12288];
	add.s32 	%r218, %r216, %r17;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r133, %r134, %r135, %r136}, [%r218];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r145, %r146, %r147, %r148}, [%r218+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r149, %r150, %r151, %r152}, [%r218+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r153, %r154, %r155, %r156}, [%r218+12288];
	add.s32 	%r219, %r216, %r18;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r157, %r158, %r159, %r160}, [%r219];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r169, %r170, %r171, %r172}, [%r219+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r173, %r174, %r175, %r176}, [%r219+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r177, %r178, %r179, %r180}, [%r219+12288];
	add.s32 	%r220, %r216, %r19;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r181, %r182, %r183, %r184}, [%r220];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r193, %r194, %r195, %r196}, [%r220+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r197, %r198, %r199, %r200}, [%r220+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r201, %r202, %r203, %r204}, [%r220+12288];
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	add.s32 	%r221, %r216, %r20;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r113, %r114, %r137, %r138}, [%r221+32768];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r115, %r116, %r139, %r140}, [%r221+36864];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r117, %r118, %r141, %r142}, [%r221+40960];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r119, %r120, %r143, %r144}, [%r221+45056];
	add.s32 	%r222, %r216, %r21;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r161, %r162, %r185, %r186}, [%r222+32768];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r163, %r164, %r187, %r188}, [%r222+36864];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r165, %r166, %r189, %r190}, [%r222+40960];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r167, %r168, %r191, %r192}, [%r222+45056];
	.loc	1 179 36                        // sk03_fa_qkv.py:179:36
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r560, %r561, %r562, %r563 }, { %r109, %r110, %r111, %r112 }, { %r113, %r114 }, { %r560, %r561, %r562, %r563 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r564, %r565, %r566, %r567 }, { %r109, %r110, %r111, %r112 }, { %r115, %r116 }, { %r564, %r565, %r566, %r567 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r568, %r569, %r570, %r571 }, { %r109, %r110, %r111, %r112 }, { %r117, %r118 }, { %r568, %r569, %r570, %r571 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r572, %r573, %r574, %r575 }, { %r109, %r110, %r111, %r112 }, { %r119, %r120 }, { %r572, %r573, %r574, %r575 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r576, %r577, %r578, %r579 }, { %r121, %r122, %r123, %r124 }, { %r113, %r114 }, { %r576, %r577, %r578, %r579 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r580, %r581, %r582, %r583 }, { %r121, %r122, %r123, %r124 }, { %r115, %r116 }, { %r580, %r581, %r582, %r583 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r584, %r585, %r586, %r587 }, { %r121, %r122, %r123, %r124 }, { %r117, %r118 }, { %r584, %r585, %r586, %r587 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r588, %r589, %r590, %r591 }, { %r121, %r122, %r123, %r124 }, { %r119, %r120 }, { %r588, %r589, %r590, %r591 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r592, %r593, %r594, %r595 }, { %r125, %r126, %r127, %r128 }, { %r113, %r114 }, { %r592, %r593, %r594, %r595 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r596, %r597, %r598, %r599 }, { %r125, %r126, %r127, %r128 }, { %r115, %r116 }, { %r596, %r597, %r598, %r599 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r600, %r601, %r602, %r603 }, { %r125, %r126, %r127, %r128 }, { %r117, %r118 }, { %r600, %r601, %r602, %r603 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r604, %r605, %r606, %r607 }, { %r125, %r126, %r127, %r128 }, { %r119, %r120 }, { %r604, %r605, %r606, %r607 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r608, %r609, %r610, %r611 }, { %r129, %r130, %r131, %r132 }, { %r113, %r114 }, { %r608, %r609, %r610, %r611 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r612, %r613, %r614, %r615 }, { %r129, %r130, %r131, %r132 }, { %r115, %r116 }, { %r612, %r613, %r614, %r615 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r616, %r617, %r618, %r619 }, { %r129, %r130, %r131, %r132 }, { %r117, %r118 }, { %r616, %r617, %r618, %r619 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r620, %r621, %r622, %r623 }, { %r129, %r130, %r131, %r132 }, { %r119, %r120 }, { %r620, %r621, %r622, %r623 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r560, %r561, %r562, %r563 }, { %r133, %r134, %r135, %r136 }, { %r137, %r138 }, { %r560, %r561, %r562, %r563 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r564, %r565, %r566, %r567 }, { %r133, %r134, %r135, %r136 }, { %r139, %r140 }, { %r564, %r565, %r566, %r567 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r568, %r569, %r570, %r571 }, { %r133, %r134, %r135, %r136 }, { %r141, %r142 }, { %r568, %r569, %r570, %r571 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r572, %r573, %r574, %r575 }, { %r133, %r134, %r135, %r136 }, { %r143, %r144 }, { %r572, %r573, %r574, %r575 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r576, %r577, %r578, %r579 }, { %r145, %r146, %r147, %r148 }, { %r137, %r138 }, { %r576, %r577, %r578, %r579 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r580, %r581, %r582, %r583 }, { %r145, %r146, %r147, %r148 }, { %r139, %r140 }, { %r580, %r581, %r582, %r583 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r584, %r585, %r586, %r587 }, { %r145, %r146, %r147, %r148 }, { %r141, %r142 }, { %r584, %r585, %r586, %r587 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r588, %r589, %r590, %r591 }, { %r145, %r146, %r147, %r148 }, { %r143, %r144 }, { %r588, %r589, %r590, %r591 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r592, %r593, %r594, %r595 }, { %r149, %r150, %r151, %r152 }, { %r137, %r138 }, { %r592, %r593, %r594, %r595 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r596, %r597, %r598, %r599 }, { %r149, %r150, %r151, %r152 }, { %r139, %r140 }, { %r596, %r597, %r598, %r599 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r600, %r601, %r602, %r603 }, { %r149, %r150, %r151, %r152 }, { %r141, %r142 }, { %r600, %r601, %r602, %r603 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r604, %r605, %r606, %r607 }, { %r149, %r150, %r151, %r152 }, { %r143, %r144 }, { %r604, %r605, %r606, %r607 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r608, %r609, %r610, %r611 }, { %r153, %r154, %r155, %r156 }, { %r137, %r138 }, { %r608, %r609, %r610, %r611 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r612, %r613, %r614, %r615 }, { %r153, %r154, %r155, %r156 }, { %r139, %r140 }, { %r612, %r613, %r614, %r615 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r616, %r617, %r618, %r619 }, { %r153, %r154, %r155, %r156 }, { %r141, %r142 }, { %r616, %r617, %r618, %r619 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r620, %r621, %r622, %r623 }, { %r153, %r154, %r155, %r156 }, { %r143, %r144 }, { %r620, %r621, %r622, %r623 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r560, %r561, %r562, %r563 }, { %r157, %r158, %r159, %r160 }, { %r161, %r162 }, { %r560, %r561, %r562, %r563 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r564, %r565, %r566, %r567 }, { %r157, %r158, %r159, %r160 }, { %r163, %r164 }, { %r564, %r565, %r566, %r567 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r568, %r569, %r570, %r571 }, { %r157, %r158, %r159, %r160 }, { %r165, %r166 }, { %r568, %r569, %r570, %r571 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r572, %r573, %r574, %r575 }, { %r157, %r158, %r159, %r160 }, { %r167, %r168 }, { %r572, %r573, %r574, %r575 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r576, %r577, %r578, %r579 }, { %r169, %r170, %r171, %r172 }, { %r161, %r162 }, { %r576, %r577, %r578, %r579 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r580, %r581, %r582, %r583 }, { %r169, %r170, %r171, %r172 }, { %r163, %r164 }, { %r580, %r581, %r582, %r583 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r584, %r585, %r586, %r587 }, { %r169, %r170, %r171, %r172 }, { %r165, %r166 }, { %r584, %r585, %r586, %r587 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r588, %r589, %r590, %r591 }, { %r169, %r170, %r171, %r172 }, { %r167, %r168 }, { %r588, %r589, %r590, %r591 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r592, %r593, %r594, %r595 }, { %r173, %r174, %r175, %r176 }, { %r161, %r162 }, { %r592, %r593, %r594, %r595 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r596, %r597, %r598, %r599 }, { %r173, %r174, %r175, %r176 }, { %r163, %r164 }, { %r596, %r597, %r598, %r599 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r600, %r601, %r602, %r603 }, { %r173, %r174, %r175, %r176 }, { %r165, %r166 }, { %r600, %r601, %r602, %r603 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r604, %r605, %r606, %r607 }, { %r173, %r174, %r175, %r176 }, { %r167, %r168 }, { %r604, %r605, %r606, %r607 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r608, %r609, %r610, %r611 }, { %r177, %r178, %r179, %r180 }, { %r161, %r162 }, { %r608, %r609, %r610, %r611 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r612, %r613, %r614, %r615 }, { %r177, %r178, %r179, %r180 }, { %r163, %r164 }, { %r612, %r613, %r614, %r615 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r616, %r617, %r618, %r619 }, { %r177, %r178, %r179, %r180 }, { %r165, %r166 }, { %r616, %r617, %r618, %r619 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r620, %r621, %r622, %r623 }, { %r177, %r178, %r179, %r180 }, { %r167, %r168 }, { %r620, %r621, %r622, %r623 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r560, %r561, %r562, %r563 }, { %r181, %r182, %r183, %r184 }, { %r185, %r186 }, { %r560, %r561, %r562, %r563 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r564, %r565, %r566, %r567 }, { %r181, %r182, %r183, %r184 }, { %r187, %r188 }, { %r564, %r565, %r566, %r567 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r568, %r569, %r570, %r571 }, { %r181, %r182, %r183, %r184 }, { %r189, %r190 }, { %r568, %r569, %r570, %r571 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r572, %r573, %r574, %r575 }, { %r181, %r182, %r183, %r184 }, { %r191, %r192 }, { %r572, %r573, %r574, %r575 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r576, %r577, %r578, %r579 }, { %r193, %r194, %r195, %r196 }, { %r185, %r186 }, { %r576, %r577, %r578, %r579 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r580, %r581, %r582, %r583 }, { %r193, %r194, %r195, %r196 }, { %r187, %r188 }, { %r580, %r581, %r582, %r583 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r584, %r585, %r586, %r587 }, { %r193, %r194, %r195, %r196 }, { %r189, %r190 }, { %r584, %r585, %r586, %r587 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r588, %r589, %r590, %r591 }, { %r193, %r194, %r195, %r196 }, { %r191, %r192 }, { %r588, %r589, %r590, %r591 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r592, %r593, %r594, %r595 }, { %r197, %r198, %r199, %r200 }, { %r185, %r186 }, { %r592, %r593, %r594, %r595 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r596, %r597, %r598, %r599 }, { %r197, %r198, %r199, %r200 }, { %r187, %r188 }, { %r596, %r597, %r598, %r599 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r600, %r601, %r602, %r603 }, { %r197, %r198, %r199, %r200 }, { %r189, %r190 }, { %r600, %r601, %r602, %r603 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r604, %r605, %r606, %r607 }, { %r197, %r198, %r199, %r200 }, { %r191, %r192 }, { %r604, %r605, %r606, %r607 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r608, %r609, %r610, %r611 }, { %r201, %r202, %r203, %r204 }, { %r185, %r186 }, { %r608, %r609, %r610, %r611 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r612, %r613, %r614, %r615 }, { %r201, %r202, %r203, %r204 }, { %r187, %r188 }, { %r612, %r613, %r614, %r615 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r616, %r617, %r618, %r619 }, { %r201, %r202, %r203, %r204 }, { %r189, %r190 }, { %r616, %r617, %r618, %r619 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r620, %r621, %r622, %r623 }, { %r201, %r202, %r203, %r204 }, { %r191, %r192 }, { %r620, %r621, %r622, %r623 };
	// end inline asm
	.loc	1 183 18                        // sk03_fa_qkv.py:183:18
	add.s64 	%rd45, %rd96, %rd5;
	add.s64 	%rd46, %rd97, %rd5;
	add.s64 	%rd47, %rd98, %rd5;
	.loc	1 184 18                        // sk03_fa_qkv.py:184:18
	add.s64 	%rd48, %rd99, %rd5;
	add.s64 	%rd49, %rd100, %rd5;
	add.s64 	%rd50, %rd101, %rd5;
	add.s64 	%rd51, %rd102, %rd5;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	add.s64 	%rd52, %rd103, %rd5;
	add.s32 	%r223, %r559, 1;
	setp.gt.s32 	%p5, %r223, 1;
	selp.b32 	%r559, 0, %r223, %p5;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
	shl.b32 	%r224, %r559, 14;
	bar.sync 	0;
	add.s32 	%r205, %r26, %r224;
	selp.b32 	%r206, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r205 + 0 ], [ %rd45 + 0 ], 0x10, %r206;
	// end inline asm
	add.s32 	%r207, %r205, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r207 + 0 ], [ %rd46 + 0 ], 0x10, %r206;
	// end inline asm
	add.s32 	%r208, %r205, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r208 + 0 ], [ %rd47 + 0 ], 0x10, %r206;
	// end inline asm
	add.s32 	%r209, %r205, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r209 + 0 ], [ %rd48 + 0 ], 0x10, %r206;
	// end inline asm
	cp.async.commit_group;
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	add.s32 	%r210, %r205, 32768;
	// begin inline asm
	cp.async.cg.shared.global [ %r210 + 0 ], [ %rd49 + 0 ], 0x10, %r206;
	// end inline asm
	add.s32 	%r211, %r205, 36864;
	// begin inline asm
	cp.async.cg.shared.global [ %r211 + 0 ], [ %rd50 + 0 ], 0x10, %r206;
	// end inline asm
	add.s32 	%r212, %r205, 40960;
	// begin inline asm
	cp.async.cg.shared.global [ %r212 + 0 ], [ %rd51 + 0 ], 0x10, %r206;
	// end inline asm
	add.s32 	%r213, %r205, 45056;
	// begin inline asm
	cp.async.cg.shared.global [ %r213 + 0 ], [ %rd52 + 0 ], 0x10, %r206;
	// end inline asm
	cp.async.commit_group;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	add.s32 	%r624, %r624, 1;
	add.s64 	%rd103, %rd103, 128;
	add.s64 	%rd102, %rd102, 128;
	add.s64 	%rd101, %rd101, 128;
	add.s64 	%rd100, %rd100, 128;
	add.s64 	%rd99, %rd99, 128;
	add.s64 	%rd98, %rd98, 128;
	add.s64 	%rd97, %rd97, 128;
	add.s64 	%rd96, %rd96, 128;
	setp.ne.b32 	%p6, %r12, %r624;
	@%p6 bra 	$L__BB0_3;
// %bb.4:                               // %._crit_edge.loopexit
	.loc	1 186 17                        // sk03_fa_qkv.py:186:17
	cvt.rn.f32.s32 	%r225, %r560;
	cvt.rn.bf16.f32 	%rs186, %r225;
	cvt.rn.f32.s32 	%r226, %r561;
	cvt.rn.bf16.f32 	%rs187, %r226;
	cvt.rn.f32.s32 	%r227, %r562;
	cvt.rn.bf16.f32 	%rs188, %r227;
	cvt.rn.f32.s32 	%r228, %r563;
	cvt.rn.bf16.f32 	%rs189, %r228;
	cvt.rn.f32.s32 	%r229, %r564;
	cvt.rn.bf16.f32 	%rs190, %r229;
	cvt.rn.f32.s32 	%r230, %r565;
	cvt.rn.bf16.f32 	%rs191, %r230;
	cvt.rn.f32.s32 	%r231, %r566;
	cvt.rn.bf16.f32 	%rs192, %r231;
	cvt.rn.f32.s32 	%r232, %r567;
	cvt.rn.bf16.f32 	%rs193, %r232;
	cvt.rn.f32.s32 	%r233, %r568;
	cvt.rn.bf16.f32 	%rs194, %r233;
	cvt.rn.f32.s32 	%r234, %r569;
	cvt.rn.bf16.f32 	%rs195, %r234;
	cvt.rn.f32.s32 	%r235, %r570;
	cvt.rn.bf16.f32 	%rs196, %r235;
	cvt.rn.f32.s32 	%r236, %r571;
	cvt.rn.bf16.f32 	%rs197, %r236;
	cvt.rn.f32.s32 	%r237, %r572;
	cvt.rn.bf16.f32 	%rs198, %r237;
	cvt.rn.f32.s32 	%r238, %r573;
	cvt.rn.bf16.f32 	%rs199, %r238;
	cvt.rn.f32.s32 	%r239, %r574;
	cvt.rn.bf16.f32 	%rs200, %r239;
	cvt.rn.f32.s32 	%r240, %r575;
	cvt.rn.bf16.f32 	%rs201, %r240;
	cvt.rn.f32.s32 	%r241, %r576;
	cvt.rn.bf16.f32 	%rs202, %r241;
	cvt.rn.f32.s32 	%r242, %r577;
	cvt.rn.bf16.f32 	%rs203, %r242;
	cvt.rn.f32.s32 	%r243, %r578;
	cvt.rn.bf16.f32 	%rs204, %r243;
	cvt.rn.f32.s32 	%r244, %r579;
	cvt.rn.bf16.f32 	%rs205, %r244;
	cvt.rn.f32.s32 	%r245, %r580;
	cvt.rn.bf16.f32 	%rs206, %r245;
	cvt.rn.f32.s32 	%r246, %r581;
	cvt.rn.bf16.f32 	%rs207, %r246;
	cvt.rn.f32.s32 	%r247, %r582;
	cvt.rn.bf16.f32 	%rs208, %r247;
	cvt.rn.f32.s32 	%r248, %r583;
	cvt.rn.bf16.f32 	%rs209, %r248;
	cvt.rn.f32.s32 	%r249, %r584;
	cvt.rn.bf16.f32 	%rs210, %r249;
	cvt.rn.f32.s32 	%r250, %r585;
	cvt.rn.bf16.f32 	%rs211, %r250;
	cvt.rn.f32.s32 	%r251, %r586;
	cvt.rn.bf16.f32 	%rs212, %r251;
	cvt.rn.f32.s32 	%r252, %r587;
	cvt.rn.bf16.f32 	%rs213, %r252;
	cvt.rn.f32.s32 	%r253, %r588;
	cvt.rn.bf16.f32 	%rs214, %r253;
	cvt.rn.f32.s32 	%r254, %r589;
	cvt.rn.bf16.f32 	%rs215, %r254;
	cvt.rn.f32.s32 	%r255, %r590;
	cvt.rn.bf16.f32 	%rs216, %r255;
	cvt.rn.f32.s32 	%r256, %r591;
	cvt.rn.bf16.f32 	%rs217, %r256;
	cvt.rn.f32.s32 	%r257, %r592;
	cvt.rn.bf16.f32 	%rs218, %r257;
	cvt.rn.f32.s32 	%r258, %r593;
	cvt.rn.bf16.f32 	%rs219, %r258;
	cvt.rn.f32.s32 	%r259, %r594;
	cvt.rn.bf16.f32 	%rs220, %r259;
	cvt.rn.f32.s32 	%r260, %r595;
	cvt.rn.bf16.f32 	%rs221, %r260;
	cvt.rn.f32.s32 	%r261, %r596;
	cvt.rn.bf16.f32 	%rs222, %r261;
	cvt.rn.f32.s32 	%r262, %r597;
	cvt.rn.bf16.f32 	%rs223, %r262;
	cvt.rn.f32.s32 	%r263, %r598;
	cvt.rn.bf16.f32 	%rs224, %r263;
	cvt.rn.f32.s32 	%r264, %r599;
	cvt.rn.bf16.f32 	%rs225, %r264;
	cvt.rn.f32.s32 	%r265, %r600;
	cvt.rn.bf16.f32 	%rs226, %r265;
	cvt.rn.f32.s32 	%r266, %r601;
	cvt.rn.bf16.f32 	%rs227, %r266;
	cvt.rn.f32.s32 	%r267, %r602;
	cvt.rn.bf16.f32 	%rs228, %r267;
	cvt.rn.f32.s32 	%r268, %r603;
	cvt.rn.bf16.f32 	%rs229, %r268;
	cvt.rn.f32.s32 	%r269, %r604;
	cvt.rn.bf16.f32 	%rs230, %r269;
	cvt.rn.f32.s32 	%r270, %r605;
	cvt.rn.bf16.f32 	%rs231, %r270;
	cvt.rn.f32.s32 	%r271, %r606;
	cvt.rn.bf16.f32 	%rs232, %r271;
	cvt.rn.f32.s32 	%r272, %r607;
	cvt.rn.bf16.f32 	%rs233, %r272;
	cvt.rn.f32.s32 	%r273, %r608;
	cvt.rn.f32.s32 	%r274, %r609;
	cvt.rn.bf16x2.f32 	%r627, %r274, %r273;
	cvt.rn.f32.s32 	%r275, %r610;
	cvt.rn.bf16.f32 	%rs234, %r275;
	cvt.rn.f32.s32 	%r276, %r611;
	cvt.rn.bf16.f32 	%rs235, %r276;
	cvt.rn.f32.s32 	%r277, %r612;
	cvt.rn.f32.s32 	%r278, %r613;
	cvt.rn.bf16x2.f32 	%r629, %r278, %r277;
	cvt.rn.f32.s32 	%r279, %r614;
	cvt.rn.bf16.f32 	%rs236, %r279;
	cvt.rn.f32.s32 	%r280, %r615;
	cvt.rn.bf16.f32 	%rs237, %r280;
	cvt.rn.f32.s32 	%r281, %r616;
	cvt.rn.f32.s32 	%r282, %r617;
	cvt.rn.bf16x2.f32 	%r628, %r282, %r281;
	cvt.rn.f32.s32 	%r283, %r618;
	cvt.rn.bf16.f32 	%rs238, %r283;
	cvt.rn.f32.s32 	%r284, %r619;
	cvt.rn.bf16.f32 	%rs239, %r284;
	cvt.rn.f32.s32 	%r285, %r620;
	cvt.rn.f32.s32 	%r286, %r621;
	cvt.rn.bf16x2.f32 	%r630, %r286, %r285;
	cvt.rn.f32.s32 	%r287, %r622;
	cvt.rn.bf16.f32 	%rs240, %r287;
	cvt.rn.f32.s32 	%r288, %r623;
	cvt.rn.bf16.f32 	%rs241, %r288;
	bra.uni 	$L__BB0_5;
$L__BB0_1:                              // %.._crit_edge_crit_edge
	.loc	1 188 19                        // sk03_fa_qkv.py:188:19
	shl.b32 	%r626, %r2, 1;
	.loc	1 194 8                         // sk03_fa_qkv.py:194:8
	and.b32 	%r625, %r2, 16;
	mov.b32 	%r627, 0;
	mov.b16 	%rs186, 0x0000;
	mov.b16 	%rs187, %rs186;
	mov.b16 	%rs188, %rs186;
	mov.b16 	%rs189, %rs186;
	mov.b16 	%rs190, %rs186;
	mov.b16 	%rs191, %rs186;
	mov.b16 	%rs192, %rs186;
	mov.b16 	%rs193, %rs186;
	mov.b16 	%rs194, %rs186;
	mov.b16 	%rs195, %rs186;
	mov.b16 	%rs196, %rs186;
	mov.b16 	%rs197, %rs186;
	mov.b16 	%rs198, %rs186;
	mov.b16 	%rs199, %rs186;
	mov.b16 	%rs200, %rs186;
	mov.b16 	%rs201, %rs186;
	mov.b16 	%rs202, %rs186;
	mov.b16 	%rs203, %rs186;
	mov.b16 	%rs204, %rs186;
	mov.b16 	%rs205, %rs186;
	mov.b16 	%rs206, %rs186;
	mov.b16 	%rs207, %rs186;
	mov.b16 	%rs208, %rs186;
	mov.b16 	%rs209, %rs186;
	mov.b16 	%rs210, %rs186;
	mov.b16 	%rs211, %rs186;
	mov.b16 	%rs212, %rs186;
	mov.b16 	%rs213, %rs186;
	mov.b16 	%rs214, %rs186;
	mov.b16 	%rs215, %rs186;
	mov.b16 	%rs216, %rs186;
	mov.b16 	%rs217, %rs186;
	mov.b16 	%rs218, %rs186;
	mov.b16 	%rs219, %rs186;
	mov.b16 	%rs220, %rs186;
	mov.b16 	%rs221, %rs186;
	mov.b16 	%rs222, %rs186;
	mov.b16 	%rs223, %rs186;
	mov.b16 	%rs224, %rs186;
	mov.b16 	%rs225, %rs186;
	mov.b16 	%rs226, %rs186;
	mov.b16 	%rs227, %rs186;
	mov.b16 	%rs228, %rs186;
	mov.b16 	%rs229, %rs186;
	mov.b16 	%rs230, %rs186;
	mov.b16 	%rs231, %rs186;
	mov.b16 	%rs232, %rs186;
	mov.b16 	%rs233, %rs186;
	mov.b16 	%rs234, %rs186;
	mov.b16 	%rs235, %rs186;
	mov.b16 	%rs236, %rs186;
	mov.b16 	%rs237, %rs186;
	mov.b16 	%rs238, %rs186;
	mov.b16 	%rs239, %rs186;
	mov.b16 	%rs240, %rs186;
	mov.b16 	%rs241, %rs186;
	mov.b32 	%r628, %r627;
	mov.b32 	%r629, %r627;
	mov.b32 	%r630, %r627;
$L__BB0_5:                              // %._crit_edge
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	or.b32 	%r411, %r11, %r557;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r412, %r411, 8;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r413, %r412, %r23;
	rem.s32 	%r414, %r411, %r23;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	and.b32 	%r415, %r2, 3;
	shl.b32 	%r416, %r415, 1;
	shr.u32 	%r417, %r4, 2;
	or.b32 	%r418, %r416, %r417;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r419, %r418, %r11;
	or.b32 	%r420, %r419, 96;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r421, %r420, %r23;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r422, %r419, 64;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r423, %r422, %r23;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r424, %r419, 32;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r425, %r424, %r23;
	rem.s32 	%r426, %r419, %r23;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	shl.b32 	%r427, %r6, 3;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r428, %r11, %r427;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	and.b32 	%r429, %r2, 128;
	shr.u32 	%r430, %r429, 3;
	shr.u32 	%r431, %r2, 2;
	bfe.u32 	%r432, %r2, 2, 3;
	or.b32 	%r433, %r430, %r432;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r434, %r433, %r1;
	or.b32 	%r435, %r434, 104;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r436, %r435, %r22;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r437, %r434, 96;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r438, %r437, %r22;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r439, %r434, 72;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r440, %r439, %r22;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r441, %r434, 64;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r442, %r441, %r22;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r443, %r434, 40;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r444, %r443, %r22;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r445, %r434, 32;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r446, %r445, %r22;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r447, %r434, 8;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r448, %r447, %r22;
	rem.s32 	%r449, %r434, %r22;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	shr.u32 	%r450, %r2, 4;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r451, %r450, %r1;
	or.b32 	%r452, %r451, 112;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	bfe.u32 	%r453, %r2, 4, 4;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r454, %r453, %r1;
	or.b32 	%r455, %r454, 96;
	or.b32 	%r456, %r454, 80;
	or.b32 	%r457, %r454, 64;
	or.b32 	%r458, %r451, 48;
	or.b32 	%r459, %r454, 32;
	or.b32 	%r460, %r454, 16;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 186 54                        // sk03_fa_qkv.py:186:54
	mad.wide.s32 	%rd53, %r449, 4, %rd14;
	mad.wide.s32 	%rd54, %r448, 4, %rd14;
	mad.wide.s32 	%rd55, %r446, 4, %rd14;
	mad.wide.s32 	%rd56, %r444, 4, %rd14;
	mad.wide.s32 	%rd57, %r442, 4, %rd14;
	mad.wide.s32 	%rd58, %r440, 4, %rd14;
	mad.wide.s32 	%rd59, %r438, 4, %rd14;
	mad.wide.s32 	%rd60, %r436, 4, %rd14;
	.loc	1 186 40                        // sk03_fa_qkv.py:186:40
	// begin inline asm
	mov.u32 %r289, 0x0;
	ld.global.b32 { %r289 }, [ %rd53 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r290, 0x0;
	ld.global.b32 { %r290 }, [ %rd54 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r291, 0x0;
	ld.global.b32 { %r291 }, [ %rd55 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r292, 0x0;
	ld.global.b32 { %r292 }, [ %rd56 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r293, 0x0;
	ld.global.b32 { %r293 }, [ %rd57 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r294, 0x0;
	ld.global.b32 { %r294 }, [ %rd58 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r295, 0x0;
	ld.global.b32 { %r295 }, [ %rd59 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r296, 0x0;
	ld.global.b32 { %r296 }, [ %rd60 + 0 ];
	// end inline asm
	.loc	1 186 65                        // sk03_fa_qkv.py:186:65
	cvt.rn.bf16.f32 	%rs1, %r289;
	cvt.rn.bf16.f32 	%rs2, %r290;
	cvt.rn.bf16.f32 	%rs3, %r291;
	cvt.rn.bf16.f32 	%rs4, %r292;
	cvt.rn.bf16.f32 	%rs5, %r293;
	cvt.rn.bf16.f32 	%rs6, %r294;
	cvt.rn.bf16.f32 	%rs7, %r295;
	cvt.rn.bf16.f32 	%rs8, %r296;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	mov.b16 	%rs9, 0x8000;
	fma.rn.bf16 	%rs10, %rs186, %rs1, %rs9;
	fma.rn.bf16 	%rs11, %rs187, %rs1, %rs9;
	fma.rn.bf16 	%rs12, %rs188, %rs2, %rs9;
	fma.rn.bf16 	%rs13, %rs189, %rs2, %rs9;
	fma.rn.bf16 	%rs14, %rs190, %rs1, %rs9;
	fma.rn.bf16 	%rs15, %rs191, %rs1, %rs9;
	fma.rn.bf16 	%rs16, %rs192, %rs2, %rs9;
	fma.rn.bf16 	%rs17, %rs193, %rs2, %rs9;
	fma.rn.bf16 	%rs18, %rs194, %rs1, %rs9;
	fma.rn.bf16 	%rs19, %rs195, %rs1, %rs9;
	fma.rn.bf16 	%rs20, %rs196, %rs2, %rs9;
	fma.rn.bf16 	%rs21, %rs197, %rs2, %rs9;
	fma.rn.bf16 	%rs22, %rs198, %rs1, %rs9;
	fma.rn.bf16 	%rs23, %rs199, %rs1, %rs9;
	fma.rn.bf16 	%rs24, %rs200, %rs2, %rs9;
	fma.rn.bf16 	%rs25, %rs201, %rs2, %rs9;
	fma.rn.bf16 	%rs26, %rs202, %rs3, %rs9;
	fma.rn.bf16 	%rs27, %rs203, %rs3, %rs9;
	fma.rn.bf16 	%rs28, %rs204, %rs4, %rs9;
	fma.rn.bf16 	%rs29, %rs205, %rs4, %rs9;
	fma.rn.bf16 	%rs30, %rs206, %rs3, %rs9;
	fma.rn.bf16 	%rs31, %rs207, %rs3, %rs9;
	fma.rn.bf16 	%rs32, %rs208, %rs4, %rs9;
	fma.rn.bf16 	%rs33, %rs209, %rs4, %rs9;
	fma.rn.bf16 	%rs34, %rs210, %rs3, %rs9;
	fma.rn.bf16 	%rs35, %rs211, %rs3, %rs9;
	fma.rn.bf16 	%rs36, %rs212, %rs4, %rs9;
	fma.rn.bf16 	%rs37, %rs213, %rs4, %rs9;
	fma.rn.bf16 	%rs38, %rs214, %rs3, %rs9;
	fma.rn.bf16 	%rs39, %rs215, %rs3, %rs9;
	fma.rn.bf16 	%rs40, %rs216, %rs4, %rs9;
	fma.rn.bf16 	%rs41, %rs217, %rs4, %rs9;
	fma.rn.bf16 	%rs42, %rs218, %rs5, %rs9;
	fma.rn.bf16 	%rs43, %rs219, %rs5, %rs9;
	fma.rn.bf16 	%rs44, %rs220, %rs6, %rs9;
	fma.rn.bf16 	%rs45, %rs221, %rs6, %rs9;
	fma.rn.bf16 	%rs46, %rs222, %rs5, %rs9;
	fma.rn.bf16 	%rs47, %rs223, %rs5, %rs9;
	fma.rn.bf16 	%rs48, %rs224, %rs6, %rs9;
	fma.rn.bf16 	%rs49, %rs225, %rs6, %rs9;
	fma.rn.bf16 	%rs50, %rs226, %rs5, %rs9;
	fma.rn.bf16 	%rs51, %rs227, %rs5, %rs9;
	fma.rn.bf16 	%rs52, %rs228, %rs6, %rs9;
	fma.rn.bf16 	%rs53, %rs229, %rs6, %rs9;
	fma.rn.bf16 	%rs54, %rs230, %rs5, %rs9;
	fma.rn.bf16 	%rs55, %rs231, %rs5, %rs9;
	fma.rn.bf16 	%rs56, %rs232, %rs6, %rs9;
	fma.rn.bf16 	%rs57, %rs233, %rs6, %rs9;
	fma.rn.bf16 	%rs58, %rs234, %rs8, %rs9;
	fma.rn.bf16 	%rs59, %rs235, %rs8, %rs9;
	fma.rn.bf16 	%rs60, %rs236, %rs8, %rs9;
	fma.rn.bf16 	%rs61, %rs237, %rs8, %rs9;
	fma.rn.bf16 	%rs62, %rs238, %rs8, %rs9;
	fma.rn.bf16 	%rs63, %rs239, %rs8, %rs9;
	fma.rn.bf16 	%rs64, %rs240, %rs8, %rs9;
	fma.rn.bf16 	%rs65, %rs241, %rs8, %rs9;
	.loc	1 187 38                        // sk03_fa_qkv.py:187:38
	mad.wide.s32 	%rd61, %r426, 4, %rd15;
	mad.wide.s32 	%rd62, %r425, 4, %rd15;
	mad.wide.s32 	%rd63, %r423, 4, %rd15;
	mad.wide.s32 	%rd64, %r421, 4, %rd15;
	.loc	1 187 24                        // sk03_fa_qkv.py:187:24
	// begin inline asm
	mov.u32 %r297, 0x0;
	mov.u32 %r298, 0x0;
	ld.global.v2.b32 { %r297, %r298 }, [ %rd61 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r299, 0x0;
	mov.u32 %r300, 0x0;
	ld.global.v2.b32 { %r299, %r300 }, [ %rd62 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r301, 0x0;
	mov.u32 %r302, 0x0;
	ld.global.v2.b32 { %r301, %r302 }, [ %rd63 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r303, 0x0;
	mov.u32 %r304, 0x0;
	ld.global.v2.b32 { %r303, %r304 }, [ %rd64 + 0 ];
	// end inline asm
	.loc	1 188 49                        // sk03_fa_qkv.py:188:49
	mul.lo.s32 	%r461, %r7, %r25;
	mul.lo.s32 	%r462, %r8, %r25;
	mul.lo.s32 	%r463, %r9, %r25;
	mul.lo.s32 	%r464, %r10, %r25;
	.loc	1 188 31                        // sk03_fa_qkv.py:188:31
	mad.wide.s32 	%rd81, %r461, 2, %rd13;
	mad.wide.s32 	%rd82, %r462, 2, %rd13;
	mad.wide.s32 	%rd83, %r463, 2, %rd13;
	mad.wide.s32 	%rd84, %r464, 2, %rd13;
	.loc	1 188 64                        // sk03_fa_qkv.py:188:64
	mul.wide.s32 	%rd85, %r414, 2;
	add.s64 	%rd65, %rd81, %rd85;
	mul.wide.s32 	%rd86, %r413, 2;
	add.s64 	%rd66, %rd81, %rd86;
	add.s64 	%rd67, %rd82, %rd85;
	add.s64 	%rd68, %rd82, %rd86;
	add.s64 	%rd69, %rd83, %rd85;
	add.s64 	%rd70, %rd83, %rd86;
	add.s64 	%rd71, %rd84, %rd85;
	add.s64 	%rd72, %rd84, %rd86;
	.loc	1 188 19                        // sk03_fa_qkv.py:188:19
	// begin inline asm
	mov.u32 %r306, 0x0;
	mov.u32 %r307, 0x0;
	mov.u32 %r308, 0x0;
	mov.u32 %r309, 0x0;
	ld.global.v4.b32 { %r306, %r307, %r308, %r309 }, [ %rd65 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r311, 0x0;
	mov.u32 %r312, 0x0;
	mov.u32 %r313, 0x0;
	mov.u32 %r314, 0x0;
	ld.global.v4.b32 { %r311, %r312, %r313, %r314 }, [ %rd66 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r315, 0x0;
	mov.u32 %r316, 0x0;
	mov.u32 %r317, 0x0;
	mov.u32 %r318, 0x0;
	ld.global.v4.b32 { %r315, %r316, %r317, %r318 }, [ %rd67 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r319, 0x0;
	mov.u32 %r320, 0x0;
	mov.u32 %r321, 0x0;
	mov.u32 %r322, 0x0;
	ld.global.v4.b32 { %r319, %r320, %r321, %r322 }, [ %rd68 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r323, 0x0;
	mov.u32 %r324, 0x0;
	mov.u32 %r325, 0x0;
	mov.u32 %r326, 0x0;
	ld.global.v4.b32 { %r323, %r324, %r325, %r326 }, [ %rd69 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r327, 0x0;
	mov.u32 %r328, 0x0;
	mov.u32 %r329, 0x0;
	mov.u32 %r330, 0x0;
	ld.global.v4.b32 { %r327, %r328, %r329, %r330 }, [ %rd70 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r331, 0x0;
	mov.u32 %r332, 0x0;
	mov.u32 %r333, 0x0;
	mov.u32 %r334, 0x0;
	ld.global.v4.b32 { %r331, %r332, %r333, %r334 }, [ %rd71 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r335, 0x0;
	mov.u32 %r336, 0x0;
	mov.u32 %r337, 0x0;
	mov.u32 %r338, 0x0;
	ld.global.v4.b32 { %r335, %r336, %r337, %r338 }, [ %rd72 + 0 ];
	// end inline asm
	shl.b32 	%r465, %r14, 7;
	shl.b32 	%r466, %r3, 1;
	or.b32 	%r467, %r465, %r557;
	xor.b32 	%r468, %r467, %r466;
	add.s32 	%r305, %r100, %r468;
	// begin inline asm
	st.shared.v4.b32 [ %r305 + 0 ], { %r306, %r307, %r308, %r309 };
	// end inline asm
	add.s32 	%r310, %r305, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r310 + 0 ], { %r311, %r312, %r313, %r314 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r469, %r5, 10;
	and.b32 	%r470, %r13, 752;
	and.b32 	%r471, %r626, 288;
	and.b32 	%r472, %r431, 16;
	xor.b32 	%r473, %r470, %r471;
	xor.b32 	%r474, %r473, %r472;
	or.b32 	%r475, %r474, %r469;
	add.s32 	%r476, %r100, %r475;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r477, %r478, %r479, %r480}, [%r476];
	mov.b32 	{%rs66, %rs67}, %r477;
	mov.b32 	{%rs68, %rs69}, %r478;
	mov.b32 	{%rs70, %rs71}, %r479;
	mov.b32 	{%rs72, %rs73}, %r480;
	xor.b32 	%r481, %r475, 64;
	add.s32 	%r482, %r100, %r481;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r483, %r484, %r485, %r486}, [%r482];
	mov.b32 	{%rs74, %rs75}, %r483;
	mov.b32 	{%rs76, %rs77}, %r484;
	mov.b32 	{%rs78, %rs79}, %r485;
	mov.b32 	{%rs80, %rs81}, %r486;
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r305 + 0 ], { %r315, %r316, %r317, %r318 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r310 + 0 ], { %r319, %r320, %r321, %r322 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r487, %r488, %r489, %r490}, [%r476];
	mov.b32 	{%rs82, %rs83}, %r487;
	mov.b32 	{%rs84, %rs85}, %r488;
	mov.b32 	{%rs86, %rs87}, %r489;
	mov.b32 	{%rs88, %rs89}, %r490;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r491, %r492, %r493, %r494}, [%r482];
	mov.b32 	{%rs90, %rs91}, %r491;
	mov.b32 	{%rs92, %rs93}, %r492;
	mov.b32 	{%rs94, %rs95}, %r493;
	mov.b32 	{%rs96, %rs97}, %r494;
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r305 + 0 ], { %r323, %r324, %r325, %r326 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r310 + 0 ], { %r327, %r328, %r329, %r330 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r495, %r496, %r497, %r498}, [%r476];
	mov.b32 	{%rs98, %rs99}, %r495;
	mov.b32 	{%rs100, %rs101}, %r496;
	mov.b32 	{%rs102, %rs103}, %r497;
	mov.b32 	{%rs104, %rs105}, %r498;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r499, %r500, %r501, %r502}, [%r482];
	mov.b32 	{%rs106, %rs107}, %r499;
	mov.b32 	{%rs108, %rs109}, %r500;
	mov.b32 	{%rs110, %rs111}, %r501;
	mov.b32 	{%rs112, %rs113}, %r502;
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r305 + 0 ], { %r331, %r332, %r333, %r334 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r310 + 0 ], { %r335, %r336, %r337, %r338 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r503, %r504, %r505, %r506}, [%r476];
	mov.b32 	{%rs114, %rs115}, %r504;
	mov.b32 	{%rs116, %rs117}, %r506;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r507, %r508, %r509, %r510}, [%r482];
	mov.b32 	{%rs118, %rs119}, %r508;
	mov.b32 	{%rs120, %rs121}, %r510;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	mov.b32 	%r511, {%rs7, %rs7};
	mov.b32 	%r512, -2147450880;
	fma.rn.bf16x2 	%r513, %r627, %r511, %r512;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs122, %r298;
	cvt.rn.bf16.f32 	%rs123, %r297;
	mov.b32 	%r514, {%rs123, %rs122};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs124, %rs10, %rs123, %rs66;
	fma.rn.bf16 	%rs125, %rs11, %rs122, %rs67;
	fma.rn.bf16 	%rs126, %rs12, %rs123, %rs68;
	fma.rn.bf16 	%rs127, %rs13, %rs122, %rs69;
	fma.rn.bf16 	%rs128, %rs26, %rs123, %rs82;
	fma.rn.bf16 	%rs129, %rs27, %rs122, %rs83;
	fma.rn.bf16 	%rs130, %rs28, %rs123, %rs84;
	fma.rn.bf16 	%rs131, %rs29, %rs122, %rs85;
	fma.rn.bf16 	%rs132, %rs42, %rs123, %rs98;
	fma.rn.bf16 	%rs133, %rs43, %rs122, %rs99;
	fma.rn.bf16 	%rs134, %rs44, %rs123, %rs100;
	fma.rn.bf16 	%rs135, %rs45, %rs122, %rs101;
	fma.rn.bf16x2 	%r343, %r513, %r514, %r503;
	fma.rn.bf16 	%rs136, %rs58, %rs123, %rs114;
	fma.rn.bf16 	%rs137, %rs59, %rs122, %rs115;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r515, %r629, %r511, %r512;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs138, %r300;
	cvt.rn.bf16.f32 	%rs139, %r299;
	mov.b32 	%r516, {%rs139, %rs138};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs140, %rs14, %rs139, %rs70;
	fma.rn.bf16 	%rs141, %rs15, %rs138, %rs71;
	fma.rn.bf16 	%rs142, %rs16, %rs139, %rs72;
	fma.rn.bf16 	%rs143, %rs17, %rs138, %rs73;
	fma.rn.bf16 	%rs144, %rs30, %rs139, %rs86;
	fma.rn.bf16 	%rs145, %rs31, %rs138, %rs87;
	fma.rn.bf16 	%rs146, %rs32, %rs139, %rs88;
	fma.rn.bf16 	%rs147, %rs33, %rs138, %rs89;
	fma.rn.bf16 	%rs148, %rs46, %rs139, %rs102;
	fma.rn.bf16 	%rs149, %rs47, %rs138, %rs103;
	fma.rn.bf16 	%rs150, %rs48, %rs139, %rs104;
	fma.rn.bf16 	%rs151, %rs49, %rs138, %rs105;
	fma.rn.bf16x2 	%r363, %r515, %r516, %r505;
	fma.rn.bf16 	%rs152, %rs60, %rs139, %rs116;
	fma.rn.bf16 	%rs153, %rs61, %rs138, %rs117;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r517, %r628, %r511, %r512;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs154, %r302;
	cvt.rn.bf16.f32 	%rs155, %r301;
	mov.b32 	%r518, {%rs155, %rs154};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs156, %rs18, %rs155, %rs74;
	fma.rn.bf16 	%rs157, %rs19, %rs154, %rs75;
	fma.rn.bf16 	%rs158, %rs20, %rs155, %rs76;
	fma.rn.bf16 	%rs159, %rs21, %rs154, %rs77;
	fma.rn.bf16 	%rs160, %rs34, %rs155, %rs90;
	fma.rn.bf16 	%rs161, %rs35, %rs154, %rs91;
	fma.rn.bf16 	%rs162, %rs36, %rs155, %rs92;
	fma.rn.bf16 	%rs163, %rs37, %rs154, %rs93;
	fma.rn.bf16 	%rs164, %rs50, %rs155, %rs106;
	fma.rn.bf16 	%rs165, %rs51, %rs154, %rs107;
	fma.rn.bf16 	%rs166, %rs52, %rs155, %rs108;
	fma.rn.bf16 	%rs167, %rs53, %rs154, %rs109;
	fma.rn.bf16x2 	%r353, %r517, %r518, %r507;
	fma.rn.bf16 	%rs168, %rs62, %rs155, %rs118;
	fma.rn.bf16 	%rs169, %rs63, %rs154, %rs119;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r519, %r630, %r511, %r512;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs170, %r304;
	cvt.rn.bf16.f32 	%rs171, %r303;
	mov.b32 	%r520, {%rs171, %rs170};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs172, %rs22, %rs171, %rs78;
	fma.rn.bf16 	%rs173, %rs23, %rs170, %rs79;
	fma.rn.bf16 	%rs174, %rs24, %rs171, %rs80;
	fma.rn.bf16 	%rs175, %rs25, %rs170, %rs81;
	fma.rn.bf16 	%rs176, %rs38, %rs171, %rs94;
	fma.rn.bf16 	%rs177, %rs39, %rs170, %rs95;
	fma.rn.bf16 	%rs178, %rs40, %rs171, %rs96;
	fma.rn.bf16 	%rs179, %rs41, %rs170, %rs97;
	fma.rn.bf16 	%rs180, %rs54, %rs171, %rs110;
	fma.rn.bf16 	%rs181, %rs55, %rs170, %rs111;
	fma.rn.bf16 	%rs182, %rs56, %rs171, %rs112;
	fma.rn.bf16 	%rs183, %rs57, %rs170, %rs113;
	fma.rn.bf16x2 	%r373, %r519, %r520, %r509;
	fma.rn.bf16 	%rs184, %rs64, %rs171, %rs120;
	fma.rn.bf16 	%rs185, %rs65, %rs170, %rs121;
	.loc	1 195 31                        // sk03_fa_qkv.py:195:31
	setp.lt.s32 	%p15, %r454, %r22;
	setp.lt.s32 	%p16, %r460, %r22;
	setp.lt.s32 	%p17, %r459, %r22;
	setp.lt.s32 	%p18, %r458, %r22;
	setp.lt.s32 	%p19, %r457, %r22;
	setp.lt.s32 	%p20, %r456, %r22;
	setp.lt.s32 	%p21, %r455, %r22;
	setp.lt.s32 	%p22, %r452, %r22;
	.loc	1 195 54                        // sk03_fa_qkv.py:195:54
	setp.lt.s32 	%p23, %r428, %r23;
	.loc	1 195 37                        // sk03_fa_qkv.py:195:37
	and.pred 	%p7, %p15, %p23;
	and.pred 	%p8, %p16, %p23;
	and.pred 	%p9, %p17, %p23;
	and.pred 	%p10, %p18, %p23;
	and.pred 	%p11, %p19, %p23;
	and.pred 	%p12, %p20, %p23;
	and.pred 	%p13, %p21, %p23;
	and.pred 	%p14, %p22, %p23;
	.loc	1 193 35                        // sk03_fa_qkv.py:193:35
	mul.lo.s32 	%r521, %r454, %r24;
	mul.lo.s32 	%r522, %r460, %r24;
	mul.lo.s32 	%r523, %r459, %r24;
	mul.lo.s32 	%r524, %r458, %r24;
	mul.lo.s32 	%r525, %r457, %r24;
	mul.lo.s32 	%r526, %r456, %r24;
	mul.lo.s32 	%r527, %r455, %r24;
	mul.lo.s32 	%r528, %r452, %r24;
	.loc	1 193 18                        // sk03_fa_qkv.py:193:18
	mad.wide.s32 	%rd87, %r521, 2, %rd12;
	mad.wide.s32 	%rd88, %r522, 2, %rd12;
	mad.wide.s32 	%rd89, %r523, 2, %rd12;
	mad.wide.s32 	%rd90, %r524, 2, %rd12;
	mad.wide.s32 	%rd91, %r525, 2, %rd12;
	mad.wide.s32 	%rd92, %r526, 2, %rd12;
	mad.wide.s32 	%rd93, %r527, 2, %rd12;
	mad.wide.s32 	%rd94, %r528, 2, %rd12;
	.loc	1 193 50                        // sk03_fa_qkv.py:193:50
	mul.wide.s32 	%rd95, %r428, 2;
	add.s64 	%rd73, %rd87, %rd95;
	add.s64 	%rd74, %rd88, %rd95;
	add.s64 	%rd75, %rd89, %rd95;
	add.s64 	%rd76, %rd90, %rd95;
	add.s64 	%rd77, %rd91, %rd95;
	add.s64 	%rd78, %rd92, %rd95;
	add.s64 	%rd79, %rd93, %rd95;
	add.s64 	%rd80, %rd94, %rd95;
	.loc	1 194 8                         // sk03_fa_qkv.py:194:8
	bar.sync 	0;
	shl.b32 	%r529, %r415, 13;
	shl.b32 	%r530, %r415, 5;
	and.b32 	%r531, %r13, 384;
	shr.u32 	%r532, %r4, 1;
	bfe.s32 	%r533, %r2, 2, 1;
	and.b32 	%r534, %r533, 4112;
	shl.b32 	%r535, %r429, 3;
	or.b32 	%r536, %r529, %r535;
	or.b32 	%r537, %r530, %r531;
	xor.b32 	%r538, %r534, %r532;
	xor.b32 	%r539, %r538, %r537;
	or.b32 	%r540, %r539, %r536;
	add.s32 	%r339, %r100, %r540;
	mov.b32 	%r340, {%rs124, %rs125};
	mov.b32 	%r341, {%rs128, %rs129};
	mov.b32 	%r342, {%rs132, %rs133};
	// begin inline asm
	st.shared.v4.b32 [ %r339 + 0 ], { %r340, %r341, %r342, %r343 };
	// end inline asm
	add.s32 	%r344, %r339, 512;
	mov.b32 	%r345, {%rs126, %rs127};
	mov.b32 	%r346, {%rs130, %rs131};
	mov.b32 	%r347, {%rs134, %rs135};
	mov.b32 	%r348, {%rs136, %rs137};
	// begin inline asm
	st.shared.v4.b32 [ %r344 + 0 ], { %r345, %r346, %r347, %r348 };
	// end inline asm
	add.s32 	%r349, %r339, 2048;
	mov.b32 	%r350, {%rs156, %rs157};
	mov.b32 	%r351, {%rs160, %rs161};
	mov.b32 	%r352, {%rs164, %rs165};
	// begin inline asm
	st.shared.v4.b32 [ %r349 + 0 ], { %r350, %r351, %r352, %r353 };
	// end inline asm
	add.s32 	%r354, %r339, 2560;
	mov.b32 	%r355, {%rs158, %rs159};
	mov.b32 	%r356, {%rs162, %rs163};
	mov.b32 	%r357, {%rs166, %rs167};
	mov.b32 	%r358, {%rs168, %rs169};
	// begin inline asm
	st.shared.v4.b32 [ %r354 + 0 ], { %r355, %r356, %r357, %r358 };
	// end inline asm
	xor.b32 	%r541, %r540, 64;
	add.s32 	%r359, %r100, %r541;
	mov.b32 	%r360, {%rs140, %rs141};
	mov.b32 	%r361, {%rs144, %rs145};
	mov.b32 	%r362, {%rs148, %rs149};
	// begin inline asm
	st.shared.v4.b32 [ %r359 + 0 ], { %r360, %r361, %r362, %r363 };
	// end inline asm
	add.s32 	%r364, %r359, 512;
	mov.b32 	%r365, {%rs142, %rs143};
	mov.b32 	%r366, {%rs146, %rs147};
	mov.b32 	%r367, {%rs150, %rs151};
	mov.b32 	%r368, {%rs152, %rs153};
	// begin inline asm
	st.shared.v4.b32 [ %r364 + 0 ], { %r365, %r366, %r367, %r368 };
	// end inline asm
	add.s32 	%r369, %r359, 2048;
	mov.b32 	%r370, {%rs172, %rs173};
	mov.b32 	%r371, {%rs176, %rs177};
	mov.b32 	%r372, {%rs180, %rs181};
	// begin inline asm
	st.shared.v4.b32 [ %r369 + 0 ], { %r370, %r371, %r372, %r373 };
	// end inline asm
	add.s32 	%r374, %r359, 2560;
	mov.b32 	%r375, {%rs174, %rs175};
	mov.b32 	%r376, {%rs178, %rs179};
	mov.b32 	%r377, {%rs182, %rs183};
	mov.b32 	%r378, {%rs184, %rs185};
	// begin inline asm
	st.shared.v4.b32 [ %r374 + 0 ], { %r375, %r376, %r377, %r378 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r542, %r2, 2;
	and.b32 	%r543, %r542, 896;
	shl.b32 	%r544, %r2, 8;
	and.b32 	%r545, %r544, 2048;
	setp.eq.b32 	%p24, %r625, 0;
	selp.b32 	%r546, 0, 4112, %p24;
	or.b32 	%r547, %r557, %r543;
	xor.b32 	%r548, %r547, %r546;
	or.b32 	%r549, %r548, %r545;
	add.s32 	%r550, %r100, %r549;
	ld.shared.v4.b32 	{%r379, %r387, %r395, %r403}, [%r550];
	ld.shared.v4.b32 	{%r383, %r391, %r399, %r407}, [%r550+1024];
	xor.b32 	%r551, %r549, 32;
	add.s32 	%r552, %r100, %r551;
	ld.shared.v4.b32 	{%r380, %r388, %r396, %r404}, [%r552+8192];
	ld.shared.v4.b32 	{%r384, %r392, %r400, %r408}, [%r552+9216];
	xor.b32 	%r553, %r549, 64;
	add.s32 	%r554, %r100, %r553;
	ld.shared.v4.b32 	{%r381, %r389, %r397, %r405}, [%r554+16384];
	ld.shared.v4.b32 	{%r385, %r393, %r401, %r409}, [%r554+17408];
	xor.b32 	%r555, %r549, 96;
	add.s32 	%r556, %r100, %r555;
	ld.shared.v4.b32 	{%r382, %r390, %r398, %r406}, [%r556+24576];
	ld.shared.v4.b32 	{%r386, %r394, %r402, %r410}, [%r556+25600];
	// begin inline asm
	@%p7 st.global.v4.b32 [ %rd73 + 0 ], { %r379, %r380, %r381, %r382 };
	// end inline asm
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd74 + 0 ], { %r383, %r384, %r385, %r386 };
	// end inline asm
	// begin inline asm
	@%p9 st.global.v4.b32 [ %rd75 + 0 ], { %r387, %r388, %r389, %r390 };
	// end inline asm
	// begin inline asm
	@%p10 st.global.v4.b32 [ %rd76 + 0 ], { %r391, %r392, %r393, %r394 };
	// end inline asm
	// begin inline asm
	@%p11 st.global.v4.b32 [ %rd77 + 0 ], { %r395, %r396, %r397, %r398 };
	// end inline asm
	// begin inline asm
	@%p12 st.global.v4.b32 [ %rd78 + 0 ], { %r399, %r400, %r401, %r402 };
	// end inline asm
	// begin inline asm
	@%p13 st.global.v4.b32 [ %rd79 + 0 ], { %r403, %r404, %r405, %r406 };
	// end inline asm
	// begin inline asm
	@%p14 st.global.v4.b32 [ %rd80 + 0 ], { %r407, %r408, %r409, %r410 };
	// end inline asm
	.loc	1 192 4                         // sk03_fa_qkv.py:192:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk03_fa_qkv.py"
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
.b32 157                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x96 DW_TAG_compile_unit
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
.b8 51
.b8 95
.b8 102
.b8 97
.b8 95
.b8 113
.b8 107
.b8 118
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
.b8 2                                   // Abbrev [2] 0x44:0x16 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 51
.b8 95
.b8 102
.b8 97
.b8 95
.b8 113
.b8 107
.b8 118
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x5a:0x46 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 68                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x6f:0x18 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 155                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x87:0x18 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 156                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_4 = _Nativo(
    "sk03_fa_qkv/tile128x128x128_shift0_abi15",
    _PTX_4, "_sk03_fa_qkv_kernel",
    warps=8, shared=65536,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 18],
    horneado={11: 1, 12: 1, 15: 1, 17: 1, 19: 1, 20: 128, 21: 128, 22: 128, 23: 8, 24: 128, 25: False},
    div16=[8, 9, 10, 13, 14, 16],
)

_PTX_5 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk03_fa_qkv_kernel     // -- Begin function _sk03_fa_qkv_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk03_fa_qkv_kernel
.visible .entry _sk03_fa_qkv_kernel(
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_6,
	.param .u32 _sk03_fa_qkv_kernel_param_7,
	.param .u32 _sk03_fa_qkv_kernel_param_8,
	.param .u32 _sk03_fa_qkv_kernel_param_9,
	.param .u32 _sk03_fa_qkv_kernel_param_10,
	.param .u32 _sk03_fa_qkv_kernel_param_11,
	.param .u32 _sk03_fa_qkv_kernel_param_12,
	.param .u32 _sk03_fa_qkv_kernel_param_13,
	.param .u32 _sk03_fa_qkv_kernel_param_14,
	.param .u32 _sk03_fa_qkv_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_16,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_17
)
.reqntid 256
{
	.reg .pred 	%p<25>;
	.reg .b16 	%rs<306>;
	.reg .b32 	%r<676>;
	.reg .b64 	%rd<174>;
	.loc	1 145 0                         // sk03_fa_qkv.py:145:0
$L__func_begin0:
	.loc	1 145 0                         // sk03_fa_qkv.py:145:0

// %bb.0:
	ld.param.b32 	%r26, [_sk03_fa_qkv_kernel_param_14];
	ld.param.b32 	%r25, [_sk03_fa_qkv_kernel_param_13];
	ld.param.b32 	%r24, [_sk03_fa_qkv_kernel_param_12];
	ld.param.b32 	%r23, [_sk03_fa_qkv_kernel_param_8];
	ld.param.b32 	%r22, [_sk03_fa_qkv_kernel_param_7];
	ld.param.b64 	%rd15, [_sk03_fa_qkv_kernel_param_5];
	ld.param.b64 	%rd14, [_sk03_fa_qkv_kernel_param_4];
	ld.param.b64 	%rd13, [_sk03_fa_qkv_kernel_param_3];
	ld.param.b64 	%rd12, [_sk03_fa_qkv_kernel_param_2];
	ld.param.b64 	%rd11, [_sk03_fa_qkv_kernel_param_1];
	ld.param.b64 	%rd10, [_sk03_fa_qkv_kernel_param_0];
$L__tmp0:
	.loc	1 154 24                        // sk03_fa_qkv.py:154:24
	mov.u32 	%r45, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk03_fa_qkv.py:155:27 ]
	add.s32 	%r46, %r22, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk03_fa_qkv.py:155:27 ]
	shr.s32 	%r47, %r46, 31;
	shr.u32 	%r48, %r47, 25;
	add.s32 	%r49, %r46, %r48;
	shr.s32 	%r50, %r49, 7;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk03_fa_qkv.py:156:27 ]
	add.s32 	%r51, %r23, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk03_fa_qkv.py:156:27 ]
	shr.s32 	%r52, %r51, 31;
	shr.u32 	%r53, %r52, 25;
	add.s32 	%r54, %r51, %r53;
	shr.s32 	%r55, %r54, 7;
$L__tmp3:
	.loc	1 157 29                        // sk03_fa_qkv.py:157:29
	shl.b32 	%r56, %r55, 3;
	.loc	1 158 22                        // sk03_fa_qkv.py:158:22
	div.s32 	%r57, %r45, %r56;
	.loc	1 158 38                        // sk03_fa_qkv.py:158:38
	shl.b32 	%r58, %r57, 3;
	ld.param.b32 	%r59, [_sk03_fa_qkv_kernel_param_9];
	.loc	1 159 30                        // sk03_fa_qkv.py:159:30
	sub.s32 	%r60, %r50, %r58;
	ld.param.b32 	%r61, [_sk03_fa_qkv_kernel_param_10];
	.loc	1 159 39                        // sk03_fa_qkv.py:159:39
	min.s32 	%r62, %r60, 8;
	ld.param.b32 	%r63, [_sk03_fa_qkv_kernel_param_11];
	.loc	1 160 30                        // sk03_fa_qkv.py:160:30
	mul.lo.s32 	%r64, %r57, %r56;
	sub.s32 	%r65, %r45, %r64;
	.loc	1 161 36                        // sk03_fa_qkv.py:161:36
	div.s32 	%r66, %r65, %r62;
	.loc	1 160 46                        // sk03_fa_qkv.py:160:46
	mul.lo.s32 	%r67, %r66, %r62;
	sub.s32 	%r68, %r65, %r67;
	.loc	1 160 23                        // sk03_fa_qkv.py:160:23
	add.s32 	%r69, %r68, %r58;
	.loc	1 163 22                        // sk03_fa_qkv.py:163:22
	shl.b32 	%r1, %r69, 7;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	mov.u32 	%r2, %tid.x;
	and.b32 	%r3, %r2, 248;
	bfe.u32 	%r70, %r2, 3, 5;
	or.b32 	%r71, %r70, 32;
	or.b32 	%r72, %r70, 64;
	or.b32 	%r73, %r70, 96;
	and.b32 	%r4, %r2, 96;
	and.b32 	%r5, %r2, 7;
	shl.b32 	%r74, %r5, 4;
	and.b32 	%r6, %r2, 15;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r75, %r1, %r70;
	or.b32 	%r76, %r1, %r71;
	or.b32 	%r77, %r1, %r72;
	or.b32 	%r78, %r1, %r73;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r7, %r75, %r22;
	rem.s32 	%r8, %r76, %r22;
	rem.s32 	%r9, %r77, %r22;
	rem.s32 	%r10, %r78, %r22;
	.loc	1 164 22                        // sk03_fa_qkv.py:164:22
	shl.b32 	%r11, %r66, 7;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r79, %r11, %r70;
	or.b32 	%r80, %r11, %r71;
	or.b32 	%r81, %r11, %r72;
	or.b32 	%r82, %r11, %r73;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r83, %r79, %r23;
	rem.s32 	%r84, %r80, %r23;
	rem.s32 	%r85, %r81, %r23;
	rem.s32 	%r86, %r82, %r23;
	.loc	1 167 39                        // sk03_fa_qkv.py:167:39
	mul.lo.s32 	%r87, %r7, %r61;
	mul.lo.s32 	%r88, %r8, %r61;
	mul.lo.s32 	%r89, %r9, %r61;
	mul.lo.s32 	%r90, %r10, %r61;
	.loc	1 167 21                        // sk03_fa_qkv.py:167:21
	cvt.s64.s32 	%rd1, %r87;
	add.s64 	%rd32, %rd10, %rd1;
	cvt.s64.s32 	%rd2, %r88;
	add.s64 	%rd33, %rd10, %rd2;
	cvt.s64.s32 	%rd3, %r89;
	add.s64 	%rd34, %rd10, %rd3;
	cvt.s64.s32 	%rd4, %r90;
	add.s64 	%rd35, %rd10, %rd4;
	.loc	1 167 51                        // sk03_fa_qkv.py:167:51
	cvt.u64.u32 	%rd5, %r74;
	add.s64 	%rd16, %rd32, %rd5;
	add.s64 	%rd17, %rd33, %rd5;
	add.s64 	%rd18, %rd34, %rd5;
	add.s64 	%rd19, %rd35, %rd5;
	.loc	1 168 21                        // sk03_fa_qkv.py:168:21
	add.s64 	%rd36, %rd11, %rd5;
	.loc	1 168 69                        // sk03_fa_qkv.py:168:69
	mul.lo.s32 	%r91, %r83, %r63;
	mul.lo.s32 	%r92, %r84, %r63;
	mul.lo.s32 	%r93, %r85, %r63;
	mul.lo.s32 	%r94, %r86, %r63;
	.loc	1 168 51                        // sk03_fa_qkv.py:168:51
	cvt.s64.s32 	%rd6, %r91;
	add.s64 	%rd20, %rd36, %rd6;
	cvt.s64.s32 	%rd7, %r92;
	add.s64 	%rd21, %rd36, %rd7;
	cvt.s64.s32 	%rd8, %r93;
	add.s64 	%rd22, %rd36, %rd8;
	cvt.s64.s32 	%rd9, %r94;
	add.s64 	%rd23, %rd36, %rd9;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	setp.gt.s32 	%p1, %r59, 127;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
	shl.b32 	%r13, %r2, 4;
	and.b32 	%r98, %r13, 4080;
	and.b32 	%r14, %r2, 56;
	shl.b32 	%r99, %r14, 1;
	xor.b32 	%r100, %r98, %r99;
	mov.b32 	%r101, global_smem;
	add.s32 	%r27, %r101, %r100;
	selp.b32 	%r28, 16, 0, %p1;
	// begin inline asm
	cp.async.cg.shared.global [ %r27 + 0 ], [ %rd16 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r29, %r27, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r29 + 0 ], [ %rd17 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r30, %r27, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r30 + 0 ], [ %rd18 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r31, %r27, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r31 + 0 ], [ %rd19 + 0 ], 0x10, %r28;
	// end inline asm
	cp.async.commit_group;
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	add.s32 	%r32, %r27, 32768;
	// begin inline asm
	cp.async.cg.shared.global [ %r32 + 0 ], [ %rd20 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r33, %r27, 36864;
	// begin inline asm
	cp.async.cg.shared.global [ %r33 + 0 ], [ %rd21 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r34, %r27, 40960;
	// begin inline asm
	cp.async.cg.shared.global [ %r34 + 0 ], [ %rd22 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r35, %r27, 45056;
	// begin inline asm
	cp.async.cg.shared.global [ %r35 + 0 ], [ %rd23 + 0 ], 0x10, %r28;
	// end inline asm
	cp.async.commit_group;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	setp.gt.s32 	%p2, %r59, 255;
	.loc	1 183 18                        // sk03_fa_qkv.py:183:18
	add.s64 	%rd24, %rd16, 128;
	add.s64 	%rd25, %rd17, 128;
	add.s64 	%rd26, %rd18, 128;
	add.s64 	%rd27, %rd19, 128;
	.loc	1 184 18                        // sk03_fa_qkv.py:184:18
	add.s64 	%rd28, %rd20, 128;
	add.s64 	%rd29, %rd21, 128;
	add.s64 	%rd30, %rd22, 128;
	add.s64 	%rd31, %rd23, 128;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
	bar.sync 	0;
	add.s32 	%r36, %r27, 16384;
	selp.b32 	%r37, 16, 0, %p2;
	// begin inline asm
	cp.async.cg.shared.global [ %r36 + 0 ], [ %rd24 + 0 ], 0x10, %r37;
	// end inline asm
	add.s32 	%r38, %r27, 20480;
	// begin inline asm
	cp.async.cg.shared.global [ %r38 + 0 ], [ %rd25 + 0 ], 0x10, %r37;
	// end inline asm
	add.s32 	%r39, %r27, 24576;
	// begin inline asm
	cp.async.cg.shared.global [ %r39 + 0 ], [ %rd26 + 0 ], 0x10, %r37;
	// end inline asm
	add.s32 	%r40, %r27, 28672;
	// begin inline asm
	cp.async.cg.shared.global [ %r40 + 0 ], [ %rd27 + 0 ], 0x10, %r37;
	// end inline asm
	cp.async.commit_group;
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	add.s32 	%r41, %r27, 49152;
	// begin inline asm
	cp.async.cg.shared.global [ %r41 + 0 ], [ %rd28 + 0 ], 0x10, %r37;
	// end inline asm
	add.s32 	%r42, %r27, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r42 + 0 ], [ %rd29 + 0 ], 0x10, %r37;
	// end inline asm
	add.s32 	%r43, %r27, 57344;
	// begin inline asm
	cp.async.cg.shared.global [ %r43 + 0 ], [ %rd30 + 0 ], 0x10, %r37;
	// end inline asm
	add.s32 	%r44, %r27, 61440;
	// begin inline asm
	cp.async.cg.shared.global [ %r44 + 0 ], [ %rd31 + 0 ], 0x10, %r37;
	// end inline asm
	cp.async.commit_group;
	cvt.u32.u64 	%r602, %rd5;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	@%p1 bra 	$L__BB0_2;
	bra.uni 	$L__BB0_1;
$L__BB0_2:                              // %.lr.ph
	.loc	1 0 23                          // sk03_fa_qkv.py:0:23
	shr.s32 	%r95, %r59, 31;
	shr.u32 	%r96, %r95, 25;
	add.s32 	%r97, %r59, %r96;
	shr.s32 	%r12, %r97, 7;
	add.s32 	%r15, %r12, -2;
	shl.b32 	%r102, %r6, 7;
	and.b32 	%r103, %r13, 2160;
	and.b32 	%r670, %r2, 16;
	or.b32 	%r104, %r102, %r103;
	xor.b32 	%r16, %r104, %r670;
	xor.b32 	%r17, %r16, 32;
	xor.b32 	%r18, %r16, 64;
	xor.b32 	%r19, %r16, 96;
	shl.b32 	%r105, %r5, 7;
	shl.b32 	%r106, %r4, 5;
	shl.b32 	%r671, %r2, 1;
	and.b32 	%r107, %r671, 48;
	or.b32 	%r108, %r105, %r106;
	xor.b32 	%r109, %r602, %r107;
	or.b32 	%r20, %r108, %r109;
	xor.b32 	%r21, %r20, 64;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	add.s64 	%rd37, %rd9, %rd11;
	add.s64 	%rd173, %rd37, 256;
	add.s64 	%rd38, %rd8, %rd11;
	add.s64 	%rd172, %rd38, 256;
	add.s64 	%rd39, %rd7, %rd11;
	add.s64 	%rd171, %rd39, 256;
	add.s64 	%rd40, %rd6, %rd11;
	add.s64 	%rd170, %rd40, 256;
	add.s64 	%rd41, %rd4, %rd10;
	add.s64 	%rd169, %rd41, 256;
	add.s64 	%rd42, %rd3, %rd10;
	add.s64 	%rd168, %rd42, 256;
	add.s64 	%rd43, %rd2, %rd10;
	add.s64 	%rd167, %rd43, 256;
	add.s64 	%rd44, %rd1, %rd10;
	add.s64 	%rd166, %rd44, 256;
	mov.b32 	%r605, 0;
	mov.b32 	%r604, 1;
	mov.b32 	%r603, -1;
	mov.b32 	%r606, %r605;
	mov.b32 	%r607, %r605;
	mov.b32 	%r608, %r605;
	mov.b32 	%r609, %r605;
	mov.b32 	%r610, %r605;
	mov.b32 	%r611, %r605;
	mov.b32 	%r612, %r605;
	mov.b32 	%r613, %r605;
	mov.b32 	%r614, %r605;
	mov.b32 	%r615, %r605;
	mov.b32 	%r616, %r605;
	mov.b32 	%r617, %r605;
	mov.b32 	%r618, %r605;
	mov.b32 	%r619, %r605;
	mov.b32 	%r620, %r605;
	mov.b32 	%r621, %r605;
	mov.b32 	%r622, %r605;
	mov.b32 	%r623, %r605;
	mov.b32 	%r624, %r605;
	mov.b32 	%r625, %r605;
	mov.b32 	%r626, %r605;
	mov.b32 	%r627, %r605;
	mov.b32 	%r628, %r605;
	mov.b32 	%r629, %r605;
	mov.b32 	%r630, %r605;
	mov.b32 	%r631, %r605;
	mov.b32 	%r632, %r605;
	mov.b32 	%r633, %r605;
	mov.b32 	%r634, %r605;
	mov.b32 	%r635, %r605;
	mov.b32 	%r636, %r605;
	mov.b32 	%r637, %r605;
	mov.b32 	%r638, %r605;
	mov.b32 	%r639, %r605;
	mov.b32 	%r640, %r605;
	mov.b32 	%r641, %r605;
	mov.b32 	%r642, %r605;
	mov.b32 	%r643, %r605;
	mov.b32 	%r644, %r605;
	mov.b32 	%r645, %r605;
	mov.b32 	%r646, %r605;
	mov.b32 	%r647, %r605;
	mov.b32 	%r648, %r605;
	mov.b32 	%r649, %r605;
	mov.b32 	%r650, %r605;
	mov.b32 	%r651, %r605;
	mov.b32 	%r652, %r605;
	mov.b32 	%r653, %r605;
	mov.b32 	%r654, %r605;
	mov.b32 	%r655, %r605;
	mov.b32 	%r656, %r605;
	mov.b32 	%r657, %r605;
	mov.b32 	%r658, %r605;
	mov.b32 	%r659, %r605;
	mov.b32 	%r660, %r605;
	mov.b32 	%r661, %r605;
	mov.b32 	%r662, %r605;
	mov.b32 	%r663, %r605;
	mov.b32 	%r664, %r605;
	mov.b32 	%r665, %r605;
	mov.b32 	%r666, %r605;
	mov.b32 	%r667, %r605;
	mov.b32 	%r668, %r605;
	mov.b32 	%r669, %r605;
$L__BB0_3:                              // =>This Inner Loop Header: Depth=1
	setp.lt.s32 	%p3, %r669, %r15;
	add.s32 	%r215, %r603, 1;
	setp.gt.s32 	%p4, %r215, 1;
	selp.b32 	%r603, 0, %r215, %p4;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
	cp.async.wait_group 	2;
	bar.sync 	0;
	shl.b32 	%r216, %r603, 14;
	add.s32 	%r217, %r101, %r216;
	add.s32 	%r218, %r217, %r16;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r110, %r111, %r112, %r113}, [%r218];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r122, %r123, %r124, %r125}, [%r218+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r126, %r127, %r128, %r129}, [%r218+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r130, %r131, %r132, %r133}, [%r218+12288];
	add.s32 	%r219, %r217, %r17;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r134, %r135, %r136, %r137}, [%r219];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r146, %r147, %r148, %r149}, [%r219+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r150, %r151, %r152, %r153}, [%r219+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r154, %r155, %r156, %r157}, [%r219+12288];
	add.s32 	%r220, %r217, %r18;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r158, %r159, %r160, %r161}, [%r220];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r170, %r171, %r172, %r173}, [%r220+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r174, %r175, %r176, %r177}, [%r220+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r178, %r179, %r180, %r181}, [%r220+12288];
	add.s32 	%r221, %r217, %r19;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r182, %r183, %r184, %r185}, [%r221];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r194, %r195, %r196, %r197}, [%r221+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r198, %r199, %r200, %r201}, [%r221+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r202, %r203, %r204, %r205}, [%r221+12288];
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	add.s32 	%r222, %r217, %r20;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r114, %r115, %r138, %r139}, [%r222+32768];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r116, %r117, %r140, %r141}, [%r222+36864];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r118, %r119, %r142, %r143}, [%r222+40960];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r120, %r121, %r144, %r145}, [%r222+45056];
	add.s32 	%r223, %r217, %r21;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r162, %r163, %r186, %r187}, [%r223+32768];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r164, %r165, %r188, %r189}, [%r223+36864];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r166, %r167, %r190, %r191}, [%r223+40960];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r168, %r169, %r192, %r193}, [%r223+45056];
	.loc	1 179 36                        // sk03_fa_qkv.py:179:36
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r605, %r606, %r607, %r608 }, { %r110, %r111, %r112, %r113 }, { %r114, %r115 }, { %r605, %r606, %r607, %r608 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r609, %r610, %r611, %r612 }, { %r110, %r111, %r112, %r113 }, { %r116, %r117 }, { %r609, %r610, %r611, %r612 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r613, %r614, %r615, %r616 }, { %r110, %r111, %r112, %r113 }, { %r118, %r119 }, { %r613, %r614, %r615, %r616 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r617, %r618, %r619, %r620 }, { %r110, %r111, %r112, %r113 }, { %r120, %r121 }, { %r617, %r618, %r619, %r620 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r621, %r622, %r623, %r624 }, { %r122, %r123, %r124, %r125 }, { %r114, %r115 }, { %r621, %r622, %r623, %r624 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r625, %r626, %r627, %r628 }, { %r122, %r123, %r124, %r125 }, { %r116, %r117 }, { %r625, %r626, %r627, %r628 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r629, %r630, %r631, %r632 }, { %r122, %r123, %r124, %r125 }, { %r118, %r119 }, { %r629, %r630, %r631, %r632 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r633, %r634, %r635, %r636 }, { %r122, %r123, %r124, %r125 }, { %r120, %r121 }, { %r633, %r634, %r635, %r636 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r637, %r638, %r639, %r640 }, { %r126, %r127, %r128, %r129 }, { %r114, %r115 }, { %r637, %r638, %r639, %r640 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r641, %r642, %r643, %r644 }, { %r126, %r127, %r128, %r129 }, { %r116, %r117 }, { %r641, %r642, %r643, %r644 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r645, %r646, %r647, %r648 }, { %r126, %r127, %r128, %r129 }, { %r118, %r119 }, { %r645, %r646, %r647, %r648 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r649, %r650, %r651, %r652 }, { %r126, %r127, %r128, %r129 }, { %r120, %r121 }, { %r649, %r650, %r651, %r652 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r653, %r654, %r655, %r656 }, { %r130, %r131, %r132, %r133 }, { %r114, %r115 }, { %r653, %r654, %r655, %r656 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r657, %r658, %r659, %r660 }, { %r130, %r131, %r132, %r133 }, { %r116, %r117 }, { %r657, %r658, %r659, %r660 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r661, %r662, %r663, %r664 }, { %r130, %r131, %r132, %r133 }, { %r118, %r119 }, { %r661, %r662, %r663, %r664 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r665, %r666, %r667, %r668 }, { %r130, %r131, %r132, %r133 }, { %r120, %r121 }, { %r665, %r666, %r667, %r668 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r605, %r606, %r607, %r608 }, { %r134, %r135, %r136, %r137 }, { %r138, %r139 }, { %r605, %r606, %r607, %r608 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r609, %r610, %r611, %r612 }, { %r134, %r135, %r136, %r137 }, { %r140, %r141 }, { %r609, %r610, %r611, %r612 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r613, %r614, %r615, %r616 }, { %r134, %r135, %r136, %r137 }, { %r142, %r143 }, { %r613, %r614, %r615, %r616 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r617, %r618, %r619, %r620 }, { %r134, %r135, %r136, %r137 }, { %r144, %r145 }, { %r617, %r618, %r619, %r620 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r621, %r622, %r623, %r624 }, { %r146, %r147, %r148, %r149 }, { %r138, %r139 }, { %r621, %r622, %r623, %r624 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r625, %r626, %r627, %r628 }, { %r146, %r147, %r148, %r149 }, { %r140, %r141 }, { %r625, %r626, %r627, %r628 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r629, %r630, %r631, %r632 }, { %r146, %r147, %r148, %r149 }, { %r142, %r143 }, { %r629, %r630, %r631, %r632 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r633, %r634, %r635, %r636 }, { %r146, %r147, %r148, %r149 }, { %r144, %r145 }, { %r633, %r634, %r635, %r636 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r637, %r638, %r639, %r640 }, { %r150, %r151, %r152, %r153 }, { %r138, %r139 }, { %r637, %r638, %r639, %r640 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r641, %r642, %r643, %r644 }, { %r150, %r151, %r152, %r153 }, { %r140, %r141 }, { %r641, %r642, %r643, %r644 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r645, %r646, %r647, %r648 }, { %r150, %r151, %r152, %r153 }, { %r142, %r143 }, { %r645, %r646, %r647, %r648 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r649, %r650, %r651, %r652 }, { %r150, %r151, %r152, %r153 }, { %r144, %r145 }, { %r649, %r650, %r651, %r652 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r653, %r654, %r655, %r656 }, { %r154, %r155, %r156, %r157 }, { %r138, %r139 }, { %r653, %r654, %r655, %r656 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r657, %r658, %r659, %r660 }, { %r154, %r155, %r156, %r157 }, { %r140, %r141 }, { %r657, %r658, %r659, %r660 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r661, %r662, %r663, %r664 }, { %r154, %r155, %r156, %r157 }, { %r142, %r143 }, { %r661, %r662, %r663, %r664 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r665, %r666, %r667, %r668 }, { %r154, %r155, %r156, %r157 }, { %r144, %r145 }, { %r665, %r666, %r667, %r668 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r605, %r606, %r607, %r608 }, { %r158, %r159, %r160, %r161 }, { %r162, %r163 }, { %r605, %r606, %r607, %r608 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r609, %r610, %r611, %r612 }, { %r158, %r159, %r160, %r161 }, { %r164, %r165 }, { %r609, %r610, %r611, %r612 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r613, %r614, %r615, %r616 }, { %r158, %r159, %r160, %r161 }, { %r166, %r167 }, { %r613, %r614, %r615, %r616 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r617, %r618, %r619, %r620 }, { %r158, %r159, %r160, %r161 }, { %r168, %r169 }, { %r617, %r618, %r619, %r620 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r621, %r622, %r623, %r624 }, { %r170, %r171, %r172, %r173 }, { %r162, %r163 }, { %r621, %r622, %r623, %r624 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r625, %r626, %r627, %r628 }, { %r170, %r171, %r172, %r173 }, { %r164, %r165 }, { %r625, %r626, %r627, %r628 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r629, %r630, %r631, %r632 }, { %r170, %r171, %r172, %r173 }, { %r166, %r167 }, { %r629, %r630, %r631, %r632 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r633, %r634, %r635, %r636 }, { %r170, %r171, %r172, %r173 }, { %r168, %r169 }, { %r633, %r634, %r635, %r636 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r637, %r638, %r639, %r640 }, { %r174, %r175, %r176, %r177 }, { %r162, %r163 }, { %r637, %r638, %r639, %r640 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r641, %r642, %r643, %r644 }, { %r174, %r175, %r176, %r177 }, { %r164, %r165 }, { %r641, %r642, %r643, %r644 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r645, %r646, %r647, %r648 }, { %r174, %r175, %r176, %r177 }, { %r166, %r167 }, { %r645, %r646, %r647, %r648 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r649, %r650, %r651, %r652 }, { %r174, %r175, %r176, %r177 }, { %r168, %r169 }, { %r649, %r650, %r651, %r652 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r653, %r654, %r655, %r656 }, { %r178, %r179, %r180, %r181 }, { %r162, %r163 }, { %r653, %r654, %r655, %r656 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r657, %r658, %r659, %r660 }, { %r178, %r179, %r180, %r181 }, { %r164, %r165 }, { %r657, %r658, %r659, %r660 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r661, %r662, %r663, %r664 }, { %r178, %r179, %r180, %r181 }, { %r166, %r167 }, { %r661, %r662, %r663, %r664 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r665, %r666, %r667, %r668 }, { %r178, %r179, %r180, %r181 }, { %r168, %r169 }, { %r665, %r666, %r667, %r668 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r605, %r606, %r607, %r608 }, { %r182, %r183, %r184, %r185 }, { %r186, %r187 }, { %r605, %r606, %r607, %r608 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r609, %r610, %r611, %r612 }, { %r182, %r183, %r184, %r185 }, { %r188, %r189 }, { %r609, %r610, %r611, %r612 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r613, %r614, %r615, %r616 }, { %r182, %r183, %r184, %r185 }, { %r190, %r191 }, { %r613, %r614, %r615, %r616 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r617, %r618, %r619, %r620 }, { %r182, %r183, %r184, %r185 }, { %r192, %r193 }, { %r617, %r618, %r619, %r620 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r621, %r622, %r623, %r624 }, { %r194, %r195, %r196, %r197 }, { %r186, %r187 }, { %r621, %r622, %r623, %r624 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r625, %r626, %r627, %r628 }, { %r194, %r195, %r196, %r197 }, { %r188, %r189 }, { %r625, %r626, %r627, %r628 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r629, %r630, %r631, %r632 }, { %r194, %r195, %r196, %r197 }, { %r190, %r191 }, { %r629, %r630, %r631, %r632 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r633, %r634, %r635, %r636 }, { %r194, %r195, %r196, %r197 }, { %r192, %r193 }, { %r633, %r634, %r635, %r636 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r637, %r638, %r639, %r640 }, { %r198, %r199, %r200, %r201 }, { %r186, %r187 }, { %r637, %r638, %r639, %r640 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r641, %r642, %r643, %r644 }, { %r198, %r199, %r200, %r201 }, { %r188, %r189 }, { %r641, %r642, %r643, %r644 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r645, %r646, %r647, %r648 }, { %r198, %r199, %r200, %r201 }, { %r190, %r191 }, { %r645, %r646, %r647, %r648 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r649, %r650, %r651, %r652 }, { %r198, %r199, %r200, %r201 }, { %r192, %r193 }, { %r649, %r650, %r651, %r652 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r653, %r654, %r655, %r656 }, { %r202, %r203, %r204, %r205 }, { %r186, %r187 }, { %r653, %r654, %r655, %r656 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r657, %r658, %r659, %r660 }, { %r202, %r203, %r204, %r205 }, { %r188, %r189 }, { %r657, %r658, %r659, %r660 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r661, %r662, %r663, %r664 }, { %r202, %r203, %r204, %r205 }, { %r190, %r191 }, { %r661, %r662, %r663, %r664 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r665, %r666, %r667, %r668 }, { %r202, %r203, %r204, %r205 }, { %r192, %r193 }, { %r665, %r666, %r667, %r668 };
	// end inline asm
	.loc	1 183 18                        // sk03_fa_qkv.py:183:18
	add.s64 	%rd45, %rd166, %rd5;
	add.s64 	%rd46, %rd167, %rd5;
	add.s64 	%rd47, %rd168, %rd5;
	.loc	1 184 18                        // sk03_fa_qkv.py:184:18
	add.s64 	%rd48, %rd169, %rd5;
	add.s64 	%rd49, %rd170, %rd5;
	add.s64 	%rd50, %rd171, %rd5;
	add.s64 	%rd51, %rd172, %rd5;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	add.s64 	%rd52, %rd173, %rd5;
	add.s32 	%r224, %r604, 1;
	setp.gt.s32 	%p5, %r224, 1;
	selp.b32 	%r604, 0, %r224, %p5;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
	shl.b32 	%r225, %r604, 14;
	bar.sync 	0;
	add.s32 	%r206, %r27, %r225;
	selp.b32 	%r207, 16, 0, %p3;
	// begin inline asm
	cp.async.cg.shared.global [ %r206 + 0 ], [ %rd45 + 0 ], 0x10, %r207;
	// end inline asm
	add.s32 	%r208, %r206, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r208 + 0 ], [ %rd46 + 0 ], 0x10, %r207;
	// end inline asm
	add.s32 	%r209, %r206, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r209 + 0 ], [ %rd47 + 0 ], 0x10, %r207;
	// end inline asm
	add.s32 	%r210, %r206, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r210 + 0 ], [ %rd48 + 0 ], 0x10, %r207;
	// end inline asm
	cp.async.commit_group;
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	add.s32 	%r211, %r206, 32768;
	// begin inline asm
	cp.async.cg.shared.global [ %r211 + 0 ], [ %rd49 + 0 ], 0x10, %r207;
	// end inline asm
	add.s32 	%r212, %r206, 36864;
	// begin inline asm
	cp.async.cg.shared.global [ %r212 + 0 ], [ %rd50 + 0 ], 0x10, %r207;
	// end inline asm
	add.s32 	%r213, %r206, 40960;
	// begin inline asm
	cp.async.cg.shared.global [ %r213 + 0 ], [ %rd51 + 0 ], 0x10, %r207;
	// end inline asm
	add.s32 	%r214, %r206, 45056;
	// begin inline asm
	cp.async.cg.shared.global [ %r214 + 0 ], [ %rd52 + 0 ], 0x10, %r207;
	// end inline asm
	cp.async.commit_group;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	add.s32 	%r669, %r669, 1;
	add.s64 	%rd173, %rd173, 128;
	add.s64 	%rd172, %rd172, 128;
	add.s64 	%rd171, %rd171, 128;
	add.s64 	%rd170, %rd170, 128;
	add.s64 	%rd169, %rd169, 128;
	add.s64 	%rd168, %rd168, 128;
	add.s64 	%rd167, %rd167, 128;
	add.s64 	%rd166, %rd166, 128;
	setp.ne.b32 	%p6, %r12, %r669;
	@%p6 bra 	$L__BB0_3;
// %bb.4:                               // %._crit_edge.loopexit
	.loc	1 186 17                        // sk03_fa_qkv.py:186:17
	cvt.rn.f32.s32 	%r226, %r605;
	cvt.rn.bf16.f32 	%rs250, %r226;
	cvt.rn.f32.s32 	%r227, %r606;
	cvt.rn.bf16.f32 	%rs251, %r227;
	cvt.rn.f32.s32 	%r228, %r607;
	cvt.rn.bf16.f32 	%rs252, %r228;
	cvt.rn.f32.s32 	%r229, %r608;
	cvt.rn.bf16.f32 	%rs253, %r229;
	cvt.rn.f32.s32 	%r230, %r609;
	cvt.rn.bf16.f32 	%rs254, %r230;
	cvt.rn.f32.s32 	%r231, %r610;
	cvt.rn.bf16.f32 	%rs255, %r231;
	cvt.rn.f32.s32 	%r232, %r611;
	cvt.rn.bf16.f32 	%rs256, %r232;
	cvt.rn.f32.s32 	%r233, %r612;
	cvt.rn.bf16.f32 	%rs257, %r233;
	cvt.rn.f32.s32 	%r234, %r613;
	cvt.rn.bf16.f32 	%rs258, %r234;
	cvt.rn.f32.s32 	%r235, %r614;
	cvt.rn.bf16.f32 	%rs259, %r235;
	cvt.rn.f32.s32 	%r236, %r615;
	cvt.rn.bf16.f32 	%rs260, %r236;
	cvt.rn.f32.s32 	%r237, %r616;
	cvt.rn.bf16.f32 	%rs261, %r237;
	cvt.rn.f32.s32 	%r238, %r617;
	cvt.rn.bf16.f32 	%rs262, %r238;
	cvt.rn.f32.s32 	%r239, %r618;
	cvt.rn.bf16.f32 	%rs263, %r239;
	cvt.rn.f32.s32 	%r240, %r619;
	cvt.rn.bf16.f32 	%rs264, %r240;
	cvt.rn.f32.s32 	%r241, %r620;
	cvt.rn.bf16.f32 	%rs265, %r241;
	cvt.rn.f32.s32 	%r242, %r621;
	cvt.rn.bf16.f32 	%rs266, %r242;
	cvt.rn.f32.s32 	%r243, %r622;
	cvt.rn.bf16.f32 	%rs267, %r243;
	cvt.rn.f32.s32 	%r244, %r623;
	cvt.rn.bf16.f32 	%rs268, %r244;
	cvt.rn.f32.s32 	%r245, %r624;
	cvt.rn.bf16.f32 	%rs269, %r245;
	cvt.rn.f32.s32 	%r246, %r625;
	cvt.rn.bf16.f32 	%rs270, %r246;
	cvt.rn.f32.s32 	%r247, %r626;
	cvt.rn.bf16.f32 	%rs271, %r247;
	cvt.rn.f32.s32 	%r248, %r627;
	cvt.rn.bf16.f32 	%rs272, %r248;
	cvt.rn.f32.s32 	%r249, %r628;
	cvt.rn.bf16.f32 	%rs273, %r249;
	cvt.rn.f32.s32 	%r250, %r629;
	cvt.rn.bf16.f32 	%rs274, %r250;
	cvt.rn.f32.s32 	%r251, %r630;
	cvt.rn.bf16.f32 	%rs275, %r251;
	cvt.rn.f32.s32 	%r252, %r631;
	cvt.rn.bf16.f32 	%rs276, %r252;
	cvt.rn.f32.s32 	%r253, %r632;
	cvt.rn.bf16.f32 	%rs277, %r253;
	cvt.rn.f32.s32 	%r254, %r633;
	cvt.rn.bf16.f32 	%rs278, %r254;
	cvt.rn.f32.s32 	%r255, %r634;
	cvt.rn.bf16.f32 	%rs279, %r255;
	cvt.rn.f32.s32 	%r256, %r635;
	cvt.rn.bf16.f32 	%rs280, %r256;
	cvt.rn.f32.s32 	%r257, %r636;
	cvt.rn.bf16.f32 	%rs281, %r257;
	cvt.rn.f32.s32 	%r258, %r637;
	cvt.rn.bf16.f32 	%rs282, %r258;
	cvt.rn.f32.s32 	%r259, %r638;
	cvt.rn.bf16.f32 	%rs283, %r259;
	cvt.rn.f32.s32 	%r260, %r639;
	cvt.rn.bf16.f32 	%rs284, %r260;
	cvt.rn.f32.s32 	%r261, %r640;
	cvt.rn.bf16.f32 	%rs285, %r261;
	cvt.rn.f32.s32 	%r262, %r641;
	cvt.rn.bf16.f32 	%rs286, %r262;
	cvt.rn.f32.s32 	%r263, %r642;
	cvt.rn.bf16.f32 	%rs287, %r263;
	cvt.rn.f32.s32 	%r264, %r643;
	cvt.rn.bf16.f32 	%rs288, %r264;
	cvt.rn.f32.s32 	%r265, %r644;
	cvt.rn.bf16.f32 	%rs289, %r265;
	cvt.rn.f32.s32 	%r266, %r645;
	cvt.rn.bf16.f32 	%rs290, %r266;
	cvt.rn.f32.s32 	%r267, %r646;
	cvt.rn.bf16.f32 	%rs291, %r267;
	cvt.rn.f32.s32 	%r268, %r647;
	cvt.rn.bf16.f32 	%rs292, %r268;
	cvt.rn.f32.s32 	%r269, %r648;
	cvt.rn.bf16.f32 	%rs293, %r269;
	cvt.rn.f32.s32 	%r270, %r649;
	cvt.rn.bf16.f32 	%rs294, %r270;
	cvt.rn.f32.s32 	%r271, %r650;
	cvt.rn.bf16.f32 	%rs295, %r271;
	cvt.rn.f32.s32 	%r272, %r651;
	cvt.rn.bf16.f32 	%rs296, %r272;
	cvt.rn.f32.s32 	%r273, %r652;
	cvt.rn.bf16.f32 	%rs297, %r273;
	cvt.rn.f32.s32 	%r274, %r653;
	cvt.rn.f32.s32 	%r275, %r654;
	cvt.rn.bf16x2.f32 	%r672, %r275, %r274;
	cvt.rn.f32.s32 	%r276, %r655;
	cvt.rn.bf16.f32 	%rs298, %r276;
	cvt.rn.f32.s32 	%r277, %r656;
	cvt.rn.bf16.f32 	%rs299, %r277;
	cvt.rn.f32.s32 	%r278, %r657;
	cvt.rn.f32.s32 	%r279, %r658;
	cvt.rn.bf16x2.f32 	%r674, %r279, %r278;
	cvt.rn.f32.s32 	%r280, %r659;
	cvt.rn.bf16.f32 	%rs300, %r280;
	cvt.rn.f32.s32 	%r281, %r660;
	cvt.rn.bf16.f32 	%rs301, %r281;
	cvt.rn.f32.s32 	%r282, %r661;
	cvt.rn.f32.s32 	%r283, %r662;
	cvt.rn.bf16x2.f32 	%r673, %r283, %r282;
	cvt.rn.f32.s32 	%r284, %r663;
	cvt.rn.bf16.f32 	%rs302, %r284;
	cvt.rn.f32.s32 	%r285, %r664;
	cvt.rn.bf16.f32 	%rs303, %r285;
	cvt.rn.f32.s32 	%r286, %r665;
	cvt.rn.f32.s32 	%r287, %r666;
	cvt.rn.bf16x2.f32 	%r675, %r287, %r286;
	cvt.rn.f32.s32 	%r288, %r667;
	cvt.rn.bf16.f32 	%rs304, %r288;
	cvt.rn.f32.s32 	%r289, %r668;
	cvt.rn.bf16.f32 	%rs305, %r289;
	bra.uni 	$L__BB0_5;
$L__BB0_1:                              // %.._crit_edge_crit_edge
	.loc	1 188 19                        // sk03_fa_qkv.py:188:19
	shl.b32 	%r671, %r2, 1;
	.loc	1 194 8                         // sk03_fa_qkv.py:194:8
	and.b32 	%r670, %r2, 16;
	mov.b32 	%r672, 0;
	mov.b16 	%rs250, 0x0000;
	mov.b16 	%rs251, %rs250;
	mov.b16 	%rs252, %rs250;
	mov.b16 	%rs253, %rs250;
	mov.b16 	%rs254, %rs250;
	mov.b16 	%rs255, %rs250;
	mov.b16 	%rs256, %rs250;
	mov.b16 	%rs257, %rs250;
	mov.b16 	%rs258, %rs250;
	mov.b16 	%rs259, %rs250;
	mov.b16 	%rs260, %rs250;
	mov.b16 	%rs261, %rs250;
	mov.b16 	%rs262, %rs250;
	mov.b16 	%rs263, %rs250;
	mov.b16 	%rs264, %rs250;
	mov.b16 	%rs265, %rs250;
	mov.b16 	%rs266, %rs250;
	mov.b16 	%rs267, %rs250;
	mov.b16 	%rs268, %rs250;
	mov.b16 	%rs269, %rs250;
	mov.b16 	%rs270, %rs250;
	mov.b16 	%rs271, %rs250;
	mov.b16 	%rs272, %rs250;
	mov.b16 	%rs273, %rs250;
	mov.b16 	%rs274, %rs250;
	mov.b16 	%rs275, %rs250;
	mov.b16 	%rs276, %rs250;
	mov.b16 	%rs277, %rs250;
	mov.b16 	%rs278, %rs250;
	mov.b16 	%rs279, %rs250;
	mov.b16 	%rs280, %rs250;
	mov.b16 	%rs281, %rs250;
	mov.b16 	%rs282, %rs250;
	mov.b16 	%rs283, %rs250;
	mov.b16 	%rs284, %rs250;
	mov.b16 	%rs285, %rs250;
	mov.b16 	%rs286, %rs250;
	mov.b16 	%rs287, %rs250;
	mov.b16 	%rs288, %rs250;
	mov.b16 	%rs289, %rs250;
	mov.b16 	%rs290, %rs250;
	mov.b16 	%rs291, %rs250;
	mov.b16 	%rs292, %rs250;
	mov.b16 	%rs293, %rs250;
	mov.b16 	%rs294, %rs250;
	mov.b16 	%rs295, %rs250;
	mov.b16 	%rs296, %rs250;
	mov.b16 	%rs297, %rs250;
	mov.b16 	%rs298, %rs250;
	mov.b16 	%rs299, %rs250;
	mov.b16 	%rs300, %rs250;
	mov.b16 	%rs301, %rs250;
	mov.b16 	%rs302, %rs250;
	mov.b16 	%rs303, %rs250;
	mov.b16 	%rs304, %rs250;
	mov.b16 	%rs305, %rs250;
	mov.b32 	%r673, %r672;
	mov.b32 	%r674, %r672;
	mov.b32 	%r675, %r672;
$L__BB0_5:                              // %._crit_edge
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	or.b32 	%r412, %r11, %r602;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r413, %r412, 15;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r414, %r413, %r23;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r415, %r412, 14;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r416, %r415, %r23;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r417, %r412, 13;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r418, %r417, %r23;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r419, %r412, 12;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r420, %r419, %r23;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r421, %r412, 11;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r422, %r421, %r23;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r423, %r412, 10;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r424, %r423, %r23;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r425, %r412, 9;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r426, %r425, %r23;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r427, %r412, 8;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r428, %r427, %r23;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r429, %r412, 7;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r430, %r429, %r23;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r431, %r412, 6;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r432, %r431, %r23;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r433, %r412, 5;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r434, %r433, %r23;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r435, %r412, 4;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r436, %r435, %r23;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r437, %r412, 3;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r438, %r437, %r23;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r439, %r412, 2;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r440, %r439, %r23;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r441, %r412, 1;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r442, %r441, %r23;
	rem.s32 	%r443, %r412, %r23;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	and.b32 	%r444, %r2, 3;
	shl.b32 	%r445, %r444, 1;
	shr.u32 	%r446, %r4, 2;
	or.b32 	%r447, %r445, %r446;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r448, %r447, %r11;
	or.b32 	%r449, %r448, 96;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r450, %r449, %r23;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r451, %r448, 64;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r452, %r451, %r23;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r453, %r448, 32;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r454, %r453, %r23;
	rem.s32 	%r455, %r448, %r23;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	shl.b32 	%r456, %r6, 3;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r457, %r11, %r456;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	and.b32 	%r458, %r2, 128;
	shr.u32 	%r459, %r458, 3;
	shr.u32 	%r460, %r2, 2;
	bfe.u32 	%r461, %r2, 2, 3;
	or.b32 	%r462, %r459, %r461;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r463, %r462, %r1;
	or.b32 	%r464, %r463, 104;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r465, %r464, %r22;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r466, %r463, 96;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r467, %r466, %r22;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r468, %r463, 72;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r469, %r468, %r22;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r470, %r463, 64;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r471, %r470, %r22;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r472, %r463, 40;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r473, %r472, %r22;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r474, %r463, 32;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r475, %r474, %r22;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r476, %r463, 8;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r477, %r476, %r22;
	rem.s32 	%r478, %r463, %r22;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	shr.u32 	%r479, %r2, 4;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r480, %r479, %r1;
	or.b32 	%r481, %r480, 112;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	bfe.u32 	%r482, %r2, 4, 4;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r483, %r482, %r1;
	or.b32 	%r484, %r483, 96;
	or.b32 	%r485, %r483, 80;
	or.b32 	%r486, %r483, 64;
	or.b32 	%r487, %r480, 48;
	or.b32 	%r488, %r483, 32;
	or.b32 	%r489, %r483, 16;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 186 54                        // sk03_fa_qkv.py:186:54
	mad.wide.s32 	%rd53, %r478, 4, %rd14;
	mad.wide.s32 	%rd54, %r477, 4, %rd14;
	mad.wide.s32 	%rd55, %r475, 4, %rd14;
	mad.wide.s32 	%rd56, %r473, 4, %rd14;
	mad.wide.s32 	%rd57, %r471, 4, %rd14;
	mad.wide.s32 	%rd58, %r469, 4, %rd14;
	mad.wide.s32 	%rd59, %r467, 4, %rd14;
	mad.wide.s32 	%rd60, %r465, 4, %rd14;
	.loc	1 186 40                        // sk03_fa_qkv.py:186:40
	// begin inline asm
	mov.u32 %r290, 0x0;
	ld.global.b32 { %r290 }, [ %rd53 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r291, 0x0;
	ld.global.b32 { %r291 }, [ %rd54 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r292, 0x0;
	ld.global.b32 { %r292 }, [ %rd55 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r293, 0x0;
	ld.global.b32 { %r293 }, [ %rd56 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r294, 0x0;
	ld.global.b32 { %r294 }, [ %rd57 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r295, 0x0;
	ld.global.b32 { %r295 }, [ %rd58 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r296, 0x0;
	ld.global.b32 { %r296 }, [ %rd59 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r297, 0x0;
	ld.global.b32 { %r297 }, [ %rd60 + 0 ];
	// end inline asm
	.loc	1 186 65                        // sk03_fa_qkv.py:186:65
	cvt.rn.bf16.f32 	%rs65, %r290;
	cvt.rn.bf16.f32 	%rs66, %r291;
	cvt.rn.bf16.f32 	%rs67, %r292;
	cvt.rn.bf16.f32 	%rs68, %r293;
	cvt.rn.bf16.f32 	%rs69, %r294;
	cvt.rn.bf16.f32 	%rs70, %r295;
	cvt.rn.bf16.f32 	%rs71, %r296;
	cvt.rn.bf16.f32 	%rs72, %r297;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	mov.b16 	%rs73, 0x8000;
	fma.rn.bf16 	%rs74, %rs250, %rs65, %rs73;
	fma.rn.bf16 	%rs75, %rs251, %rs65, %rs73;
	fma.rn.bf16 	%rs76, %rs252, %rs66, %rs73;
	fma.rn.bf16 	%rs77, %rs253, %rs66, %rs73;
	fma.rn.bf16 	%rs78, %rs254, %rs65, %rs73;
	fma.rn.bf16 	%rs79, %rs255, %rs65, %rs73;
	fma.rn.bf16 	%rs80, %rs256, %rs66, %rs73;
	fma.rn.bf16 	%rs81, %rs257, %rs66, %rs73;
	fma.rn.bf16 	%rs82, %rs258, %rs65, %rs73;
	fma.rn.bf16 	%rs83, %rs259, %rs65, %rs73;
	fma.rn.bf16 	%rs84, %rs260, %rs66, %rs73;
	fma.rn.bf16 	%rs85, %rs261, %rs66, %rs73;
	fma.rn.bf16 	%rs86, %rs262, %rs65, %rs73;
	fma.rn.bf16 	%rs87, %rs263, %rs65, %rs73;
	fma.rn.bf16 	%rs88, %rs264, %rs66, %rs73;
	fma.rn.bf16 	%rs89, %rs265, %rs66, %rs73;
	fma.rn.bf16 	%rs90, %rs266, %rs67, %rs73;
	fma.rn.bf16 	%rs91, %rs267, %rs67, %rs73;
	fma.rn.bf16 	%rs92, %rs268, %rs68, %rs73;
	fma.rn.bf16 	%rs93, %rs269, %rs68, %rs73;
	fma.rn.bf16 	%rs94, %rs270, %rs67, %rs73;
	fma.rn.bf16 	%rs95, %rs271, %rs67, %rs73;
	fma.rn.bf16 	%rs96, %rs272, %rs68, %rs73;
	fma.rn.bf16 	%rs97, %rs273, %rs68, %rs73;
	fma.rn.bf16 	%rs98, %rs274, %rs67, %rs73;
	fma.rn.bf16 	%rs99, %rs275, %rs67, %rs73;
	fma.rn.bf16 	%rs100, %rs276, %rs68, %rs73;
	fma.rn.bf16 	%rs101, %rs277, %rs68, %rs73;
	fma.rn.bf16 	%rs102, %rs278, %rs67, %rs73;
	fma.rn.bf16 	%rs103, %rs279, %rs67, %rs73;
	fma.rn.bf16 	%rs104, %rs280, %rs68, %rs73;
	fma.rn.bf16 	%rs105, %rs281, %rs68, %rs73;
	fma.rn.bf16 	%rs106, %rs282, %rs69, %rs73;
	fma.rn.bf16 	%rs107, %rs283, %rs69, %rs73;
	fma.rn.bf16 	%rs108, %rs284, %rs70, %rs73;
	fma.rn.bf16 	%rs109, %rs285, %rs70, %rs73;
	fma.rn.bf16 	%rs110, %rs286, %rs69, %rs73;
	fma.rn.bf16 	%rs111, %rs287, %rs69, %rs73;
	fma.rn.bf16 	%rs112, %rs288, %rs70, %rs73;
	fma.rn.bf16 	%rs113, %rs289, %rs70, %rs73;
	fma.rn.bf16 	%rs114, %rs290, %rs69, %rs73;
	fma.rn.bf16 	%rs115, %rs291, %rs69, %rs73;
	fma.rn.bf16 	%rs116, %rs292, %rs70, %rs73;
	fma.rn.bf16 	%rs117, %rs293, %rs70, %rs73;
	fma.rn.bf16 	%rs118, %rs294, %rs69, %rs73;
	fma.rn.bf16 	%rs119, %rs295, %rs69, %rs73;
	fma.rn.bf16 	%rs120, %rs296, %rs70, %rs73;
	fma.rn.bf16 	%rs121, %rs297, %rs70, %rs73;
	fma.rn.bf16 	%rs122, %rs298, %rs72, %rs73;
	fma.rn.bf16 	%rs123, %rs299, %rs72, %rs73;
	fma.rn.bf16 	%rs124, %rs300, %rs72, %rs73;
	fma.rn.bf16 	%rs125, %rs301, %rs72, %rs73;
	fma.rn.bf16 	%rs126, %rs302, %rs72, %rs73;
	fma.rn.bf16 	%rs127, %rs303, %rs72, %rs73;
	fma.rn.bf16 	%rs128, %rs304, %rs72, %rs73;
	fma.rn.bf16 	%rs129, %rs305, %rs72, %rs73;
	.loc	1 187 38                        // sk03_fa_qkv.py:187:38
	mad.wide.s32 	%rd61, %r455, 4, %rd15;
	mad.wide.s32 	%rd62, %r454, 4, %rd15;
	mad.wide.s32 	%rd63, %r452, 4, %rd15;
	mad.wide.s32 	%rd64, %r450, 4, %rd15;
	.loc	1 187 24                        // sk03_fa_qkv.py:187:24
	// begin inline asm
	mov.u32 %r298, 0x0;
	mov.u32 %r299, 0x0;
	ld.global.v2.b32 { %r298, %r299 }, [ %rd61 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r300, 0x0;
	mov.u32 %r301, 0x0;
	ld.global.v2.b32 { %r300, %r301 }, [ %rd62 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r302, 0x0;
	mov.u32 %r303, 0x0;
	ld.global.v2.b32 { %r302, %r303 }, [ %rd63 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r304, 0x0;
	mov.u32 %r305, 0x0;
	ld.global.v2.b32 { %r304, %r305 }, [ %rd64 + 0 ];
	// end inline asm
	.loc	1 188 49                        // sk03_fa_qkv.py:188:49
	mul.lo.s32 	%r490, %r7, %r25;
	mul.lo.s32 	%r491, %r8, %r25;
	mul.lo.s32 	%r492, %r9, %r25;
	mul.lo.s32 	%r493, %r10, %r25;
	.loc	1 188 31                        // sk03_fa_qkv.py:188:31
	mad.wide.s32 	%rd137, %r490, 2, %rd13;
	mad.wide.s32 	%rd138, %r491, 2, %rd13;
	mad.wide.s32 	%rd139, %r492, 2, %rd13;
	mad.wide.s32 	%rd140, %r493, 2, %rd13;
	.loc	1 188 82                        // sk03_fa_qkv.py:188:82
	mul.lo.s32 	%r494, %r443, %r26;
	mul.lo.s32 	%r495, %r442, %r26;
	mul.lo.s32 	%r496, %r440, %r26;
	mul.lo.s32 	%r497, %r438, %r26;
	mul.lo.s32 	%r498, %r436, %r26;
	mul.lo.s32 	%r499, %r434, %r26;
	mul.lo.s32 	%r500, %r432, %r26;
	mul.lo.s32 	%r501, %r430, %r26;
	mul.lo.s32 	%r502, %r428, %r26;
	mul.lo.s32 	%r503, %r426, %r26;
	mul.lo.s32 	%r504, %r424, %r26;
	mul.lo.s32 	%r505, %r422, %r26;
	mul.lo.s32 	%r506, %r420, %r26;
	mul.lo.s32 	%r507, %r418, %r26;
	mul.lo.s32 	%r508, %r416, %r26;
	mul.lo.s32 	%r509, %r414, %r26;
	.loc	1 188 64                        // sk03_fa_qkv.py:188:64
	mul.wide.s32 	%rd141, %r494, 2;
	add.s64 	%rd65, %rd137, %rd141;
	mul.wide.s32 	%rd142, %r495, 2;
	add.s64 	%rd66, %rd137, %rd142;
	mul.wide.s32 	%rd143, %r496, 2;
	add.s64 	%rd67, %rd137, %rd143;
	mul.wide.s32 	%rd144, %r497, 2;
	add.s64 	%rd68, %rd137, %rd144;
	mul.wide.s32 	%rd145, %r498, 2;
	add.s64 	%rd69, %rd137, %rd145;
	mul.wide.s32 	%rd146, %r499, 2;
	add.s64 	%rd70, %rd137, %rd146;
	mul.wide.s32 	%rd147, %r500, 2;
	add.s64 	%rd71, %rd137, %rd147;
	mul.wide.s32 	%rd148, %r501, 2;
	add.s64 	%rd72, %rd137, %rd148;
	mul.wide.s32 	%rd149, %r502, 2;
	add.s64 	%rd73, %rd137, %rd149;
	mul.wide.s32 	%rd150, %r503, 2;
	add.s64 	%rd74, %rd137, %rd150;
	mul.wide.s32 	%rd151, %r504, 2;
	add.s64 	%rd75, %rd137, %rd151;
	mul.wide.s32 	%rd152, %r505, 2;
	add.s64 	%rd76, %rd137, %rd152;
	mul.wide.s32 	%rd153, %r506, 2;
	add.s64 	%rd77, %rd137, %rd153;
	mul.wide.s32 	%rd154, %r507, 2;
	add.s64 	%rd78, %rd137, %rd154;
	mul.wide.s32 	%rd155, %r508, 2;
	add.s64 	%rd79, %rd137, %rd155;
	mul.wide.s32 	%rd156, %r509, 2;
	add.s64 	%rd80, %rd137, %rd156;
	add.s64 	%rd81, %rd138, %rd141;
	add.s64 	%rd82, %rd138, %rd142;
	add.s64 	%rd83, %rd138, %rd143;
	add.s64 	%rd84, %rd138, %rd144;
	add.s64 	%rd85, %rd138, %rd145;
	add.s64 	%rd86, %rd138, %rd146;
	add.s64 	%rd87, %rd138, %rd147;
	add.s64 	%rd88, %rd138, %rd148;
	add.s64 	%rd89, %rd138, %rd149;
	add.s64 	%rd90, %rd138, %rd150;
	add.s64 	%rd91, %rd138, %rd151;
	add.s64 	%rd92, %rd138, %rd152;
	add.s64 	%rd93, %rd138, %rd153;
	add.s64 	%rd94, %rd138, %rd154;
	add.s64 	%rd95, %rd138, %rd155;
	add.s64 	%rd96, %rd138, %rd156;
	add.s64 	%rd97, %rd139, %rd141;
	add.s64 	%rd98, %rd139, %rd142;
	add.s64 	%rd99, %rd139, %rd143;
	add.s64 	%rd100, %rd139, %rd144;
	add.s64 	%rd101, %rd139, %rd145;
	add.s64 	%rd102, %rd139, %rd146;
	add.s64 	%rd103, %rd139, %rd147;
	add.s64 	%rd104, %rd139, %rd148;
	add.s64 	%rd105, %rd139, %rd149;
	add.s64 	%rd106, %rd139, %rd150;
	add.s64 	%rd107, %rd139, %rd151;
	add.s64 	%rd108, %rd139, %rd152;
	add.s64 	%rd109, %rd139, %rd153;
	add.s64 	%rd110, %rd139, %rd154;
	add.s64 	%rd111, %rd139, %rd155;
	add.s64 	%rd112, %rd139, %rd156;
	add.s64 	%rd113, %rd140, %rd141;
	add.s64 	%rd114, %rd140, %rd142;
	add.s64 	%rd115, %rd140, %rd143;
	add.s64 	%rd116, %rd140, %rd144;
	add.s64 	%rd117, %rd140, %rd145;
	add.s64 	%rd118, %rd140, %rd146;
	add.s64 	%rd119, %rd140, %rd147;
	add.s64 	%rd120, %rd140, %rd148;
	add.s64 	%rd121, %rd140, %rd149;
	add.s64 	%rd122, %rd140, %rd150;
	add.s64 	%rd123, %rd140, %rd151;
	add.s64 	%rd124, %rd140, %rd152;
	add.s64 	%rd125, %rd140, %rd153;
	add.s64 	%rd126, %rd140, %rd154;
	add.s64 	%rd127, %rd140, %rd155;
	add.s64 	%rd128, %rd140, %rd156;
	.loc	1 188 19                        // sk03_fa_qkv.py:188:19
	// begin inline asm
	mov.u16 %rs1, 0x0;
	ld.global.b16 { %rs1 }, [ %rd65 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs2, 0x0;
	ld.global.b16 { %rs2 }, [ %rd66 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs3, 0x0;
	ld.global.b16 { %rs3 }, [ %rd67 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs4, 0x0;
	ld.global.b16 { %rs4 }, [ %rd68 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs5, 0x0;
	ld.global.b16 { %rs5 }, [ %rd69 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs6, 0x0;
	ld.global.b16 { %rs6 }, [ %rd70 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs7, 0x0;
	ld.global.b16 { %rs7 }, [ %rd71 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs8, 0x0;
	ld.global.b16 { %rs8 }, [ %rd72 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs9, 0x0;
	ld.global.b16 { %rs9 }, [ %rd73 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs10, 0x0;
	ld.global.b16 { %rs10 }, [ %rd74 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs11, 0x0;
	ld.global.b16 { %rs11 }, [ %rd75 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs12, 0x0;
	ld.global.b16 { %rs12 }, [ %rd76 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs13, 0x0;
	ld.global.b16 { %rs13 }, [ %rd77 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs14, 0x0;
	ld.global.b16 { %rs14 }, [ %rd78 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs15, 0x0;
	ld.global.b16 { %rs15 }, [ %rd79 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs16, 0x0;
	ld.global.b16 { %rs16 }, [ %rd80 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs17, 0x0;
	ld.global.b16 { %rs17 }, [ %rd81 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs18, 0x0;
	ld.global.b16 { %rs18 }, [ %rd82 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs19, 0x0;
	ld.global.b16 { %rs19 }, [ %rd83 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs20, 0x0;
	ld.global.b16 { %rs20 }, [ %rd84 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs21, 0x0;
	ld.global.b16 { %rs21 }, [ %rd85 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs22, 0x0;
	ld.global.b16 { %rs22 }, [ %rd86 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs23, 0x0;
	ld.global.b16 { %rs23 }, [ %rd87 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs24, 0x0;
	ld.global.b16 { %rs24 }, [ %rd88 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs25, 0x0;
	ld.global.b16 { %rs25 }, [ %rd89 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs26, 0x0;
	ld.global.b16 { %rs26 }, [ %rd90 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs27, 0x0;
	ld.global.b16 { %rs27 }, [ %rd91 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs28, 0x0;
	ld.global.b16 { %rs28 }, [ %rd92 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs29, 0x0;
	ld.global.b16 { %rs29 }, [ %rd93 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs30, 0x0;
	ld.global.b16 { %rs30 }, [ %rd94 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs31, 0x0;
	ld.global.b16 { %rs31 }, [ %rd95 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs32, 0x0;
	ld.global.b16 { %rs32 }, [ %rd96 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs33, 0x0;
	ld.global.b16 { %rs33 }, [ %rd97 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs34, 0x0;
	ld.global.b16 { %rs34 }, [ %rd98 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs35, 0x0;
	ld.global.b16 { %rs35 }, [ %rd99 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs36, 0x0;
	ld.global.b16 { %rs36 }, [ %rd100 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs37, 0x0;
	ld.global.b16 { %rs37 }, [ %rd101 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs38, 0x0;
	ld.global.b16 { %rs38 }, [ %rd102 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs39, 0x0;
	ld.global.b16 { %rs39 }, [ %rd103 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs40, 0x0;
	ld.global.b16 { %rs40 }, [ %rd104 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs41, 0x0;
	ld.global.b16 { %rs41 }, [ %rd105 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs42, 0x0;
	ld.global.b16 { %rs42 }, [ %rd106 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs43, 0x0;
	ld.global.b16 { %rs43 }, [ %rd107 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs44, 0x0;
	ld.global.b16 { %rs44 }, [ %rd108 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs45, 0x0;
	ld.global.b16 { %rs45 }, [ %rd109 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs46, 0x0;
	ld.global.b16 { %rs46 }, [ %rd110 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs47, 0x0;
	ld.global.b16 { %rs47 }, [ %rd111 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs48, 0x0;
	ld.global.b16 { %rs48 }, [ %rd112 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs49, 0x0;
	ld.global.b16 { %rs49 }, [ %rd113 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs50, 0x0;
	ld.global.b16 { %rs50 }, [ %rd114 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs51, 0x0;
	ld.global.b16 { %rs51 }, [ %rd115 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs52, 0x0;
	ld.global.b16 { %rs52 }, [ %rd116 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs53, 0x0;
	ld.global.b16 { %rs53 }, [ %rd117 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs54, 0x0;
	ld.global.b16 { %rs54 }, [ %rd118 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs55, 0x0;
	ld.global.b16 { %rs55 }, [ %rd119 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs56, 0x0;
	ld.global.b16 { %rs56 }, [ %rd120 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs57, 0x0;
	ld.global.b16 { %rs57 }, [ %rd121 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs58, 0x0;
	ld.global.b16 { %rs58 }, [ %rd122 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs59, 0x0;
	ld.global.b16 { %rs59 }, [ %rd123 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs60, 0x0;
	ld.global.b16 { %rs60 }, [ %rd124 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs61, 0x0;
	ld.global.b16 { %rs61 }, [ %rd125 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs62, 0x0;
	ld.global.b16 { %rs62 }, [ %rd126 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs63, 0x0;
	ld.global.b16 { %rs63 }, [ %rd127 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs64, 0x0;
	ld.global.b16 { %rs64 }, [ %rd128 + 0 ];
	// end inline asm
	shl.b32 	%r510, %r14, 7;
	shl.b32 	%r511, %r3, 1;
	or.b32 	%r512, %r510, %r602;
	xor.b32 	%r513, %r512, %r511;
	add.s32 	%r306, %r101, %r513;
	mov.b32 	%r307, {%rs1, %rs2};
	mov.b32 	%r308, {%rs3, %rs4};
	mov.b32 	%r309, {%rs5, %rs6};
	mov.b32 	%r310, {%rs7, %rs8};
	// begin inline asm
	st.shared.v4.b32 [ %r306 + 0 ], { %r307, %r308, %r309, %r310 };
	// end inline asm
	add.s32 	%r311, %r306, 512;
	mov.b32 	%r312, {%rs9, %rs10};
	mov.b32 	%r313, {%rs11, %rs12};
	mov.b32 	%r314, {%rs13, %rs14};
	mov.b32 	%r315, {%rs15, %rs16};
	// begin inline asm
	st.shared.v4.b32 [ %r311 + 0 ], { %r312, %r313, %r314, %r315 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r514, %r5, 10;
	and.b32 	%r515, %r13, 752;
	and.b32 	%r516, %r671, 288;
	and.b32 	%r517, %r460, 16;
	xor.b32 	%r518, %r515, %r516;
	xor.b32 	%r519, %r518, %r517;
	or.b32 	%r520, %r519, %r514;
	add.s32 	%r521, %r101, %r520;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r522, %r523, %r524, %r525}, [%r521];
	mov.b32 	{%rs130, %rs131}, %r522;
	mov.b32 	{%rs132, %rs133}, %r523;
	mov.b32 	{%rs134, %rs135}, %r524;
	mov.b32 	{%rs136, %rs137}, %r525;
	xor.b32 	%r526, %r520, 64;
	add.s32 	%r527, %r101, %r526;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r528, %r529, %r530, %r531}, [%r527];
	mov.b32 	{%rs138, %rs139}, %r528;
	mov.b32 	{%rs140, %rs141}, %r529;
	mov.b32 	{%rs142, %rs143}, %r530;
	mov.b32 	{%rs144, %rs145}, %r531;
	bar.sync 	0;
	mov.b32 	%r316, {%rs17, %rs18};
	mov.b32 	%r317, {%rs19, %rs20};
	mov.b32 	%r318, {%rs21, %rs22};
	mov.b32 	%r319, {%rs23, %rs24};
	// begin inline asm
	st.shared.v4.b32 [ %r306 + 0 ], { %r316, %r317, %r318, %r319 };
	// end inline asm
	mov.b32 	%r320, {%rs25, %rs26};
	mov.b32 	%r321, {%rs27, %rs28};
	mov.b32 	%r322, {%rs29, %rs30};
	mov.b32 	%r323, {%rs31, %rs32};
	// begin inline asm
	st.shared.v4.b32 [ %r311 + 0 ], { %r320, %r321, %r322, %r323 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r532, %r533, %r534, %r535}, [%r521];
	mov.b32 	{%rs146, %rs147}, %r532;
	mov.b32 	{%rs148, %rs149}, %r533;
	mov.b32 	{%rs150, %rs151}, %r534;
	mov.b32 	{%rs152, %rs153}, %r535;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r536, %r537, %r538, %r539}, [%r527];
	mov.b32 	{%rs154, %rs155}, %r536;
	mov.b32 	{%rs156, %rs157}, %r537;
	mov.b32 	{%rs158, %rs159}, %r538;
	mov.b32 	{%rs160, %rs161}, %r539;
	bar.sync 	0;
	mov.b32 	%r324, {%rs33, %rs34};
	mov.b32 	%r325, {%rs35, %rs36};
	mov.b32 	%r326, {%rs37, %rs38};
	mov.b32 	%r327, {%rs39, %rs40};
	// begin inline asm
	st.shared.v4.b32 [ %r306 + 0 ], { %r324, %r325, %r326, %r327 };
	// end inline asm
	mov.b32 	%r328, {%rs41, %rs42};
	mov.b32 	%r329, {%rs43, %rs44};
	mov.b32 	%r330, {%rs45, %rs46};
	mov.b32 	%r331, {%rs47, %rs48};
	// begin inline asm
	st.shared.v4.b32 [ %r311 + 0 ], { %r328, %r329, %r330, %r331 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r540, %r541, %r542, %r543}, [%r521];
	mov.b32 	{%rs162, %rs163}, %r540;
	mov.b32 	{%rs164, %rs165}, %r541;
	mov.b32 	{%rs166, %rs167}, %r542;
	mov.b32 	{%rs168, %rs169}, %r543;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r544, %r545, %r546, %r547}, [%r527];
	mov.b32 	{%rs170, %rs171}, %r544;
	mov.b32 	{%rs172, %rs173}, %r545;
	mov.b32 	{%rs174, %rs175}, %r546;
	mov.b32 	{%rs176, %rs177}, %r547;
	bar.sync 	0;
	mov.b32 	%r332, {%rs49, %rs50};
	mov.b32 	%r333, {%rs51, %rs52};
	mov.b32 	%r334, {%rs53, %rs54};
	mov.b32 	%r335, {%rs55, %rs56};
	// begin inline asm
	st.shared.v4.b32 [ %r306 + 0 ], { %r332, %r333, %r334, %r335 };
	// end inline asm
	mov.b32 	%r336, {%rs57, %rs58};
	mov.b32 	%r337, {%rs59, %rs60};
	mov.b32 	%r338, {%rs61, %rs62};
	mov.b32 	%r339, {%rs63, %rs64};
	// begin inline asm
	st.shared.v4.b32 [ %r311 + 0 ], { %r336, %r337, %r338, %r339 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r548, %r549, %r550, %r551}, [%r521];
	mov.b32 	{%rs178, %rs179}, %r549;
	mov.b32 	{%rs180, %rs181}, %r551;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r552, %r553, %r554, %r555}, [%r527];
	mov.b32 	{%rs182, %rs183}, %r553;
	mov.b32 	{%rs184, %rs185}, %r555;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	mov.b32 	%r556, {%rs71, %rs71};
	mov.b32 	%r557, -2147450880;
	fma.rn.bf16x2 	%r558, %r672, %r556, %r557;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs186, %r299;
	cvt.rn.bf16.f32 	%rs187, %r298;
	mov.b32 	%r559, {%rs187, %rs186};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs188, %rs74, %rs187, %rs130;
	fma.rn.bf16 	%rs189, %rs75, %rs186, %rs131;
	fma.rn.bf16 	%rs190, %rs76, %rs187, %rs132;
	fma.rn.bf16 	%rs191, %rs77, %rs186, %rs133;
	fma.rn.bf16 	%rs192, %rs90, %rs187, %rs146;
	fma.rn.bf16 	%rs193, %rs91, %rs186, %rs147;
	fma.rn.bf16 	%rs194, %rs92, %rs187, %rs148;
	fma.rn.bf16 	%rs195, %rs93, %rs186, %rs149;
	fma.rn.bf16 	%rs196, %rs106, %rs187, %rs162;
	fma.rn.bf16 	%rs197, %rs107, %rs186, %rs163;
	fma.rn.bf16 	%rs198, %rs108, %rs187, %rs164;
	fma.rn.bf16 	%rs199, %rs109, %rs186, %rs165;
	fma.rn.bf16x2 	%r344, %r558, %r559, %r548;
	fma.rn.bf16 	%rs200, %rs122, %rs187, %rs178;
	fma.rn.bf16 	%rs201, %rs123, %rs186, %rs179;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r560, %r674, %r556, %r557;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs202, %r301;
	cvt.rn.bf16.f32 	%rs203, %r300;
	mov.b32 	%r561, {%rs203, %rs202};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs204, %rs78, %rs203, %rs134;
	fma.rn.bf16 	%rs205, %rs79, %rs202, %rs135;
	fma.rn.bf16 	%rs206, %rs80, %rs203, %rs136;
	fma.rn.bf16 	%rs207, %rs81, %rs202, %rs137;
	fma.rn.bf16 	%rs208, %rs94, %rs203, %rs150;
	fma.rn.bf16 	%rs209, %rs95, %rs202, %rs151;
	fma.rn.bf16 	%rs210, %rs96, %rs203, %rs152;
	fma.rn.bf16 	%rs211, %rs97, %rs202, %rs153;
	fma.rn.bf16 	%rs212, %rs110, %rs203, %rs166;
	fma.rn.bf16 	%rs213, %rs111, %rs202, %rs167;
	fma.rn.bf16 	%rs214, %rs112, %rs203, %rs168;
	fma.rn.bf16 	%rs215, %rs113, %rs202, %rs169;
	fma.rn.bf16x2 	%r364, %r560, %r561, %r550;
	fma.rn.bf16 	%rs216, %rs124, %rs203, %rs180;
	fma.rn.bf16 	%rs217, %rs125, %rs202, %rs181;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r562, %r673, %r556, %r557;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs218, %r303;
	cvt.rn.bf16.f32 	%rs219, %r302;
	mov.b32 	%r563, {%rs219, %rs218};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs220, %rs82, %rs219, %rs138;
	fma.rn.bf16 	%rs221, %rs83, %rs218, %rs139;
	fma.rn.bf16 	%rs222, %rs84, %rs219, %rs140;
	fma.rn.bf16 	%rs223, %rs85, %rs218, %rs141;
	fma.rn.bf16 	%rs224, %rs98, %rs219, %rs154;
	fma.rn.bf16 	%rs225, %rs99, %rs218, %rs155;
	fma.rn.bf16 	%rs226, %rs100, %rs219, %rs156;
	fma.rn.bf16 	%rs227, %rs101, %rs218, %rs157;
	fma.rn.bf16 	%rs228, %rs114, %rs219, %rs170;
	fma.rn.bf16 	%rs229, %rs115, %rs218, %rs171;
	fma.rn.bf16 	%rs230, %rs116, %rs219, %rs172;
	fma.rn.bf16 	%rs231, %rs117, %rs218, %rs173;
	fma.rn.bf16x2 	%r354, %r562, %r563, %r552;
	fma.rn.bf16 	%rs232, %rs126, %rs219, %rs182;
	fma.rn.bf16 	%rs233, %rs127, %rs218, %rs183;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r564, %r675, %r556, %r557;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs234, %r305;
	cvt.rn.bf16.f32 	%rs235, %r304;
	mov.b32 	%r565, {%rs235, %rs234};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs236, %rs86, %rs235, %rs142;
	fma.rn.bf16 	%rs237, %rs87, %rs234, %rs143;
	fma.rn.bf16 	%rs238, %rs88, %rs235, %rs144;
	fma.rn.bf16 	%rs239, %rs89, %rs234, %rs145;
	fma.rn.bf16 	%rs240, %rs102, %rs235, %rs158;
	fma.rn.bf16 	%rs241, %rs103, %rs234, %rs159;
	fma.rn.bf16 	%rs242, %rs104, %rs235, %rs160;
	fma.rn.bf16 	%rs243, %rs105, %rs234, %rs161;
	fma.rn.bf16 	%rs244, %rs118, %rs235, %rs174;
	fma.rn.bf16 	%rs245, %rs119, %rs234, %rs175;
	fma.rn.bf16 	%rs246, %rs120, %rs235, %rs176;
	fma.rn.bf16 	%rs247, %rs121, %rs234, %rs177;
	fma.rn.bf16x2 	%r374, %r564, %r565, %r554;
	fma.rn.bf16 	%rs248, %rs128, %rs235, %rs184;
	fma.rn.bf16 	%rs249, %rs129, %rs234, %rs185;
	.loc	1 195 31                        // sk03_fa_qkv.py:195:31
	setp.lt.s32 	%p15, %r483, %r22;
	setp.lt.s32 	%p16, %r489, %r22;
	setp.lt.s32 	%p17, %r488, %r22;
	setp.lt.s32 	%p18, %r487, %r22;
	setp.lt.s32 	%p19, %r486, %r22;
	setp.lt.s32 	%p20, %r485, %r22;
	setp.lt.s32 	%p21, %r484, %r22;
	setp.lt.s32 	%p22, %r481, %r22;
	.loc	1 195 54                        // sk03_fa_qkv.py:195:54
	setp.lt.s32 	%p23, %r457, %r23;
	.loc	1 195 37                        // sk03_fa_qkv.py:195:37
	and.pred 	%p7, %p15, %p23;
	and.pred 	%p8, %p16, %p23;
	and.pred 	%p9, %p17, %p23;
	and.pred 	%p10, %p18, %p23;
	and.pred 	%p11, %p19, %p23;
	and.pred 	%p12, %p20, %p23;
	and.pred 	%p13, %p21, %p23;
	and.pred 	%p14, %p22, %p23;
	.loc	1 193 35                        // sk03_fa_qkv.py:193:35
	mul.lo.s32 	%r566, %r483, %r24;
	mul.lo.s32 	%r567, %r489, %r24;
	mul.lo.s32 	%r568, %r488, %r24;
	mul.lo.s32 	%r569, %r487, %r24;
	mul.lo.s32 	%r570, %r486, %r24;
	mul.lo.s32 	%r571, %r485, %r24;
	mul.lo.s32 	%r572, %r484, %r24;
	mul.lo.s32 	%r573, %r481, %r24;
	.loc	1 193 18                        // sk03_fa_qkv.py:193:18
	mad.wide.s32 	%rd157, %r566, 2, %rd12;
	mad.wide.s32 	%rd158, %r567, 2, %rd12;
	mad.wide.s32 	%rd159, %r568, 2, %rd12;
	mad.wide.s32 	%rd160, %r569, 2, %rd12;
	mad.wide.s32 	%rd161, %r570, 2, %rd12;
	mad.wide.s32 	%rd162, %r571, 2, %rd12;
	mad.wide.s32 	%rd163, %r572, 2, %rd12;
	mad.wide.s32 	%rd164, %r573, 2, %rd12;
	.loc	1 193 50                        // sk03_fa_qkv.py:193:50
	mul.wide.s32 	%rd165, %r457, 2;
	add.s64 	%rd129, %rd157, %rd165;
	add.s64 	%rd130, %rd158, %rd165;
	add.s64 	%rd131, %rd159, %rd165;
	add.s64 	%rd132, %rd160, %rd165;
	add.s64 	%rd133, %rd161, %rd165;
	add.s64 	%rd134, %rd162, %rd165;
	add.s64 	%rd135, %rd163, %rd165;
	add.s64 	%rd136, %rd164, %rd165;
	.loc	1 194 8                         // sk03_fa_qkv.py:194:8
	bar.sync 	0;
	shl.b32 	%r574, %r444, 13;
	shl.b32 	%r575, %r444, 5;
	and.b32 	%r576, %r13, 384;
	shr.u32 	%r577, %r4, 1;
	bfe.s32 	%r578, %r2, 2, 1;
	and.b32 	%r579, %r578, 4112;
	shl.b32 	%r580, %r458, 3;
	or.b32 	%r581, %r574, %r580;
	or.b32 	%r582, %r575, %r576;
	xor.b32 	%r583, %r579, %r577;
	xor.b32 	%r584, %r583, %r582;
	or.b32 	%r585, %r584, %r581;
	add.s32 	%r340, %r101, %r585;
	mov.b32 	%r341, {%rs188, %rs189};
	mov.b32 	%r342, {%rs192, %rs193};
	mov.b32 	%r343, {%rs196, %rs197};
	// begin inline asm
	st.shared.v4.b32 [ %r340 + 0 ], { %r341, %r342, %r343, %r344 };
	// end inline asm
	add.s32 	%r345, %r340, 512;
	mov.b32 	%r346, {%rs190, %rs191};
	mov.b32 	%r347, {%rs194, %rs195};
	mov.b32 	%r348, {%rs198, %rs199};
	mov.b32 	%r349, {%rs200, %rs201};
	// begin inline asm
	st.shared.v4.b32 [ %r345 + 0 ], { %r346, %r347, %r348, %r349 };
	// end inline asm
	add.s32 	%r350, %r340, 2048;
	mov.b32 	%r351, {%rs220, %rs221};
	mov.b32 	%r352, {%rs224, %rs225};
	mov.b32 	%r353, {%rs228, %rs229};
	// begin inline asm
	st.shared.v4.b32 [ %r350 + 0 ], { %r351, %r352, %r353, %r354 };
	// end inline asm
	add.s32 	%r355, %r340, 2560;
	mov.b32 	%r356, {%rs222, %rs223};
	mov.b32 	%r357, {%rs226, %rs227};
	mov.b32 	%r358, {%rs230, %rs231};
	mov.b32 	%r359, {%rs232, %rs233};
	// begin inline asm
	st.shared.v4.b32 [ %r355 + 0 ], { %r356, %r357, %r358, %r359 };
	// end inline asm
	xor.b32 	%r586, %r585, 64;
	add.s32 	%r360, %r101, %r586;
	mov.b32 	%r361, {%rs204, %rs205};
	mov.b32 	%r362, {%rs208, %rs209};
	mov.b32 	%r363, {%rs212, %rs213};
	// begin inline asm
	st.shared.v4.b32 [ %r360 + 0 ], { %r361, %r362, %r363, %r364 };
	// end inline asm
	add.s32 	%r365, %r360, 512;
	mov.b32 	%r366, {%rs206, %rs207};
	mov.b32 	%r367, {%rs210, %rs211};
	mov.b32 	%r368, {%rs214, %rs215};
	mov.b32 	%r369, {%rs216, %rs217};
	// begin inline asm
	st.shared.v4.b32 [ %r365 + 0 ], { %r366, %r367, %r368, %r369 };
	// end inline asm
	add.s32 	%r370, %r360, 2048;
	mov.b32 	%r371, {%rs236, %rs237};
	mov.b32 	%r372, {%rs240, %rs241};
	mov.b32 	%r373, {%rs244, %rs245};
	// begin inline asm
	st.shared.v4.b32 [ %r370 + 0 ], { %r371, %r372, %r373, %r374 };
	// end inline asm
	add.s32 	%r375, %r360, 2560;
	mov.b32 	%r376, {%rs238, %rs239};
	mov.b32 	%r377, {%rs242, %rs243};
	mov.b32 	%r378, {%rs246, %rs247};
	mov.b32 	%r379, {%rs248, %rs249};
	// begin inline asm
	st.shared.v4.b32 [ %r375 + 0 ], { %r376, %r377, %r378, %r379 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r587, %r2, 2;
	and.b32 	%r588, %r587, 896;
	shl.b32 	%r589, %r2, 8;
	and.b32 	%r590, %r589, 2048;
	setp.eq.b32 	%p24, %r670, 0;
	selp.b32 	%r591, 0, 4112, %p24;
	or.b32 	%r592, %r602, %r588;
	xor.b32 	%r593, %r592, %r591;
	or.b32 	%r594, %r593, %r590;
	add.s32 	%r595, %r101, %r594;
	ld.shared.v4.b32 	{%r380, %r388, %r396, %r404}, [%r595];
	ld.shared.v4.b32 	{%r384, %r392, %r400, %r408}, [%r595+1024];
	xor.b32 	%r596, %r594, 32;
	add.s32 	%r597, %r101, %r596;
	ld.shared.v4.b32 	{%r381, %r389, %r397, %r405}, [%r597+8192];
	ld.shared.v4.b32 	{%r385, %r393, %r401, %r409}, [%r597+9216];
	xor.b32 	%r598, %r594, 64;
	add.s32 	%r599, %r101, %r598;
	ld.shared.v4.b32 	{%r382, %r390, %r398, %r406}, [%r599+16384];
	ld.shared.v4.b32 	{%r386, %r394, %r402, %r410}, [%r599+17408];
	xor.b32 	%r600, %r594, 96;
	add.s32 	%r601, %r101, %r600;
	ld.shared.v4.b32 	{%r383, %r391, %r399, %r407}, [%r601+24576];
	ld.shared.v4.b32 	{%r387, %r395, %r403, %r411}, [%r601+25600];
	// begin inline asm
	@%p7 st.global.v4.b32 [ %rd129 + 0 ], { %r380, %r381, %r382, %r383 };
	// end inline asm
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd130 + 0 ], { %r384, %r385, %r386, %r387 };
	// end inline asm
	// begin inline asm
	@%p9 st.global.v4.b32 [ %rd131 + 0 ], { %r388, %r389, %r390, %r391 };
	// end inline asm
	// begin inline asm
	@%p10 st.global.v4.b32 [ %rd132 + 0 ], { %r392, %r393, %r394, %r395 };
	// end inline asm
	// begin inline asm
	@%p11 st.global.v4.b32 [ %rd133 + 0 ], { %r396, %r397, %r398, %r399 };
	// end inline asm
	// begin inline asm
	@%p12 st.global.v4.b32 [ %rd134 + 0 ], { %r400, %r401, %r402, %r403 };
	// end inline asm
	// begin inline asm
	@%p13 st.global.v4.b32 [ %rd135 + 0 ], { %r404, %r405, %r406, %r407 };
	// end inline asm
	// begin inline asm
	@%p14 st.global.v4.b32 [ %rd136 + 0 ], { %r408, %r409, %r410, %r411 };
	// end inline asm
	.loc	1 192 4                         // sk03_fa_qkv.py:192:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk03_fa_qkv.py"
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
.b32 157                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x96 DW_TAG_compile_unit
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
.b8 51
.b8 95
.b8 102
.b8 97
.b8 95
.b8 113
.b8 107
.b8 118
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
.b8 2                                   // Abbrev [2] 0x44:0x16 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 51
.b8 95
.b8 102
.b8 97
.b8 95
.b8 113
.b8 107
.b8 118
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x5a:0x46 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 68                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x6f:0x18 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 155                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x87:0x18 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 156                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_5 = _Nativo(
    "sk03_fa_qkv/tile128x128x128_shift0_abi16",
    _PTX_5, "_sk03_fa_qkv_kernel",
    warps=8, shared=65536,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 17, 18],
    horneado={11: 1, 12: 1, 15: 1, 19: 1, 20: 128, 21: 128, 22: 128, 23: 8, 24: 128, 25: False},
    div16=[8, 9, 10, 13, 14, 16, 17],
)

_PTX_6 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk03_fa_qkv_kernel     // -- Begin function _sk03_fa_qkv_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk03_fa_qkv_kernel
.visible .entry _sk03_fa_qkv_kernel(
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_6,
	.param .u32 _sk03_fa_qkv_kernel_param_7,
	.param .u32 _sk03_fa_qkv_kernel_param_8,
	.param .u32 _sk03_fa_qkv_kernel_param_9,
	.param .u32 _sk03_fa_qkv_kernel_param_10,
	.param .u32 _sk03_fa_qkv_kernel_param_11,
	.param .u32 _sk03_fa_qkv_kernel_param_12,
	.param .u32 _sk03_fa_qkv_kernel_param_13,
	.param .u32 _sk03_fa_qkv_kernel_param_14,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_16
)
.reqntid 256
{
	.reg .pred 	%p<25>;
	.reg .b16 	%rs<250>;
	.reg .b32 	%r<843>;
	.reg .b64 	%rd<142>;
	.loc	1 145 0                         // sk03_fa_qkv.py:145:0
$L__func_begin0:
	.loc	1 145 0                         // sk03_fa_qkv.py:145:0

// %bb.0:
	ld.param.b32 	%r29, [_sk03_fa_qkv_kernel_param_13];
	ld.param.b32 	%r28, [_sk03_fa_qkv_kernel_param_12];
	ld.param.b32 	%r27, [_sk03_fa_qkv_kernel_param_9];
	ld.param.b32 	%r26, [_sk03_fa_qkv_kernel_param_8];
	ld.param.b32 	%r25, [_sk03_fa_qkv_kernel_param_7];
	ld.param.b64 	%rd33, [_sk03_fa_qkv_kernel_param_5];
	ld.param.b64 	%rd32, [_sk03_fa_qkv_kernel_param_4];
	ld.param.b64 	%rd31, [_sk03_fa_qkv_kernel_param_3];
	ld.param.b64 	%rd30, [_sk03_fa_qkv_kernel_param_2];
	ld.param.b64 	%rd29, [_sk03_fa_qkv_kernel_param_1];
	ld.param.b64 	%rd28, [_sk03_fa_qkv_kernel_param_0];
$L__tmp0:
	.loc	1 154 24                        // sk03_fa_qkv.py:154:24
	mov.u32 	%r49, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk03_fa_qkv.py:155:27 ]
	add.s32 	%r50, %r25, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk03_fa_qkv.py:155:27 ]
	shr.s32 	%r51, %r50, 31;
	shr.u32 	%r52, %r51, 25;
	add.s32 	%r53, %r50, %r52;
	shr.s32 	%r54, %r53, 7;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk03_fa_qkv.py:156:27 ]
	add.s32 	%r55, %r26, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk03_fa_qkv.py:156:27 ]
	shr.s32 	%r56, %r55, 31;
	shr.u32 	%r57, %r56, 25;
	add.s32 	%r58, %r55, %r57;
	shr.s32 	%r59, %r58, 7;
$L__tmp3:
	.loc	1 157 29                        // sk03_fa_qkv.py:157:29
	shl.b32 	%r60, %r59, 3;
	.loc	1 158 22                        // sk03_fa_qkv.py:158:22
	div.s32 	%r61, %r49, %r60;
	.loc	1 158 38                        // sk03_fa_qkv.py:158:38
	shl.b32 	%r62, %r61, 3;
	.loc	1 159 30                        // sk03_fa_qkv.py:159:30
	sub.s32 	%r63, %r54, %r62;
	ld.param.b32 	%r64, [_sk03_fa_qkv_kernel_param_10];
	.loc	1 159 39                        // sk03_fa_qkv.py:159:39
	min.s32 	%r65, %r63, 8;
	ld.param.b32 	%r66, [_sk03_fa_qkv_kernel_param_11];
	.loc	1 160 30                        // sk03_fa_qkv.py:160:30
	mul.lo.s32 	%r67, %r61, %r60;
	sub.s32 	%r68, %r49, %r67;
	.loc	1 161 36                        // sk03_fa_qkv.py:161:36
	div.s32 	%r69, %r68, %r65;
	.loc	1 160 46                        // sk03_fa_qkv.py:160:46
	mul.lo.s32 	%r70, %r69, %r65;
	sub.s32 	%r71, %r68, %r70;
	.loc	1 160 23                        // sk03_fa_qkv.py:160:23
	add.s32 	%r72, %r71, %r62;
	.loc	1 163 22                        // sk03_fa_qkv.py:163:22
	shl.b32 	%r1, %r72, 7;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
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
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r81, %r1, %r73;
	or.b32 	%r82, %r1, %r74;
	or.b32 	%r83, %r1, %r75;
	or.b32 	%r84, %r1, %r76;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r8, %r81, %r25;
	rem.s32 	%r9, %r82, %r25;
	rem.s32 	%r10, %r83, %r25;
	rem.s32 	%r11, %r84, %r25;
	.loc	1 164 22                        // sk03_fa_qkv.py:164:22
	shl.b32 	%r12, %r69, 7;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r85, %r12, %r73;
	or.b32 	%r86, %r12, %r74;
	or.b32 	%r87, %r12, %r75;
	or.b32 	%r88, %r12, %r76;
	or.b32 	%r89, %r12, %r79;
	or.b32 	%r91, %r89, 32;
	or.b32 	%r93, %r89, 64;
	or.b32 	%r95, %r89, 96;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r97, %r85, %r26;
	rem.s32 	%r98, %r86, %r26;
	rem.s32 	%r99, %r87, %r26;
	rem.s32 	%r100, %r88, %r26;
	rem.s32 	%r13, %r89, %r26;
	rem.s32 	%r14, %r91, %r26;
	rem.s32 	%r15, %r93, %r26;
	rem.s32 	%r16, %r95, %r26;
	.loc	1 167 39                        // sk03_fa_qkv.py:167:39
	mul.lo.s32 	%r105, %r8, %r64;
	mul.lo.s32 	%r106, %r9, %r64;
	mul.lo.s32 	%r107, %r10, %r64;
	mul.lo.s32 	%r108, %r11, %r64;
	.loc	1 167 21                        // sk03_fa_qkv.py:167:21
	cvt.s64.s32 	%rd1, %r105;
	add.s64 	%rd51, %rd28, %rd1;
	cvt.s64.s32 	%rd2, %r106;
	add.s64 	%rd52, %rd28, %rd2;
	cvt.s64.s32 	%rd3, %r107;
	add.s64 	%rd53, %rd28, %rd3;
	cvt.s64.s32 	%rd4, %r108;
	add.s64 	%rd54, %rd28, %rd4;
	.loc	1 167 51                        // sk03_fa_qkv.py:167:51
	cvt.u64.u32 	%rd5, %r80;
	add.s64 	%rd34, %rd51, %rd5;
	add.s64 	%rd35, %rd52, %rd5;
	add.s64 	%rd36, %rd53, %rd5;
	add.s64 	%rd37, %rd54, %rd5;
	.loc	1 168 21                        // sk03_fa_qkv.py:168:21
	add.s64 	%rd55, %rd29, %rd5;
	.loc	1 168 69                        // sk03_fa_qkv.py:168:69
	mul.lo.s32 	%r109, %r97, %r66;
	mul.lo.s32 	%r110, %r98, %r66;
	mul.lo.s32 	%r111, %r99, %r66;
	mul.lo.s32 	%r112, %r100, %r66;
	.loc	1 168 51                        // sk03_fa_qkv.py:168:51
	cvt.s64.s32 	%rd6, %r109;
	add.s64 	%rd38, %rd55, %rd6;
	cvt.s64.s32 	%rd7, %r110;
	add.s64 	%rd39, %rd55, %rd7;
	cvt.s64.s32 	%rd8, %r111;
	add.s64 	%rd40, %rd55, %rd8;
	cvt.s64.s32 	%rd9, %r112;
	add.s64 	%rd41, %rd55, %rd9;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	setp.gt.s32 	%p1, %r27, 127;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
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
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
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
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	setp.gt.s32 	%p2, %r27, 255;
	.loc	1 183 18                        // sk03_fa_qkv.py:183:18
	add.s64 	%rd42, %rd34, 128;
	add.s64 	%rd43, %rd35, 128;
	add.s64 	%rd44, %rd36, 128;
	add.s64 	%rd45, %rd37, 128;
	.loc	1 184 18                        // sk03_fa_qkv.py:184:18
	add.s64 	%rd46, %rd38, 128;
	add.s64 	%rd47, %rd39, 128;
	add.s64 	%rd48, %rd40, 128;
	add.s64 	%rd49, %rd41, 128;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
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
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
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
	cvt.u32.u64 	%r741, %rd5;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	@%p1 bra 	$L__BB0_2;
	bra.uni 	$L__BB0_1;
$L__BB0_2:                              // %.lr.ph
	.loc	1 0 23                          // sk03_fa_qkv.py:0:23
	ld.param.b32 	%r30, [_sk03_fa_qkv_kernel_param_14];
	ld.param.b64 	%rd50, [_sk03_fa_qkv_kernel_param_6];
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
	.loc	1 178 28                        // sk03_fa_qkv.py:178:28
	shr.u32 	%r150, %r27, 7;
	add.s32 	%r151, %r150, -2;
	shl.b32 	%r152, %r7, 7;
	and.b32 	%r153, %r17, 2160;
	and.b32 	%r809, %r2, 16;
	or.b32 	%r154, %r152, %r153;
	xor.b32 	%r19, %r154, %r809;
	xor.b32 	%r20, %r19, 32;
	xor.b32 	%r21, %r19, 64;
	xor.b32 	%r22, %r19, 96;
	shl.b32 	%r155, %r6, 7;
	shl.b32 	%r156, %r5, 5;
	shl.b32 	%r810, %r2, 1;
	and.b32 	%r157, %r810, 48;
	or.b32 	%r158, %r155, %r156;
	xor.b32 	%r159, %r741, %r157;
	or.b32 	%r23, %r158, %r159;
	xor.b32 	%r24, %r23, 64;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
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
	mov.b32 	%r149, 0;
	mov.b32 	%r744, 1;
	mov.b32 	%r743, -1;
	mov.b64 	%rd140, 0;
	mov.b32 	%r742, %r149;
	mov.b64 	%rd141, %rd140;
	mov.b32 	%r745, %r149;
	mov.b32 	%r746, %r149;
	mov.b32 	%r747, %r149;
	mov.b32 	%r748, %r149;
	mov.b32 	%r749, %r149;
	mov.b32 	%r750, %r149;
	mov.b32 	%r751, %r149;
	mov.b32 	%r752, %r149;
	mov.b32 	%r753, %r149;
	mov.b32 	%r754, %r149;
	mov.b32 	%r755, %r149;
	mov.b32 	%r756, %r149;
	mov.b32 	%r757, %r149;
	mov.b32 	%r758, %r149;
	mov.b32 	%r759, %r149;
	mov.b32 	%r760, %r149;
	mov.b32 	%r761, %r149;
	mov.b32 	%r762, %r149;
	mov.b32 	%r763, %r149;
	mov.b32 	%r764, %r149;
	mov.b32 	%r765, %r149;
	mov.b32 	%r766, %r149;
	mov.b32 	%r767, %r149;
	mov.b32 	%r768, %r149;
	mov.b32 	%r769, %r149;
	mov.b32 	%r770, %r149;
	mov.b32 	%r771, %r149;
	mov.b32 	%r772, %r149;
	mov.b32 	%r773, %r149;
	mov.b32 	%r774, %r149;
	mov.b32 	%r775, %r149;
	mov.b32 	%r776, %r149;
	mov.b32 	%r777, %r149;
	mov.b32 	%r778, %r149;
	mov.b32 	%r779, %r149;
	mov.b32 	%r780, %r149;
	mov.b32 	%r781, %r149;
	mov.b32 	%r782, %r149;
	mov.b32 	%r783, %r149;
	mov.b32 	%r784, %r149;
	mov.b32 	%r785, %r149;
	mov.b32 	%r786, %r149;
	mov.b32 	%r787, %r149;
	mov.b32 	%r788, %r149;
	mov.b32 	%r789, %r149;
	mov.b32 	%r790, %r149;
	mov.b32 	%r791, %r149;
	mov.b32 	%r792, %r149;
	mov.b32 	%r793, %r149;
	mov.b32 	%r794, %r149;
	mov.b32 	%r795, %r149;
	mov.b32 	%r796, %r149;
	mov.b32 	%r797, %r149;
	mov.b32 	%r798, %r149;
	mov.b32 	%r799, %r149;
	mov.b32 	%r800, %r149;
	mov.b32 	%r801, %r149;
	mov.b32 	%r802, %r149;
	mov.b32 	%r803, %r149;
	mov.b32 	%r804, %r149;
	mov.b32 	%r805, %r149;
	mov.b32 	%r806, %r149;
	mov.b32 	%r807, %r149;
	mov.b32 	%r808, %r149;
$L__BB0_3:                              // =>This Inner Loop Header: Depth=1
	setp.lt.s64 	%p3, %rd141, %rd18;
	add.s32 	%r330, %r743, 1;
	setp.gt.s32 	%p4, %r330, 1;
	selp.b32 	%r743, 0, %r330, %p4;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
	cp.async.wait_group 	2;
	bar.sync 	0;
	shl.b32 	%r331, %r743, 14;
	add.s32 	%r332, %r148, %r331;
	add.s32 	%r333, %r332, %r19;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r161, %r162, %r163, %r164}, [%r333];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r173, %r174, %r175, %r176}, [%r333+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r177, %r178, %r179, %r180}, [%r333+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r181, %r182, %r183, %r184}, [%r333+12288];
	add.s32 	%r334, %r332, %r20;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r189, %r190, %r191, %r192}, [%r334];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r217, %r218, %r219, %r220}, [%r334+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r237, %r238, %r239, %r240}, [%r334+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r257, %r258, %r259, %r260}, [%r334+12288];
	add.s32 	%r335, %r332, %r21;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r273, %r274, %r275, %r276}, [%r335];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r285, %r286, %r287, %r288}, [%r335+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r289, %r290, %r291, %r292}, [%r335+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r293, %r294, %r295, %r296}, [%r335+12288];
	add.s32 	%r336, %r332, %r22;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r297, %r298, %r299, %r300}, [%r336];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r309, %r310, %r311, %r312}, [%r336+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r313, %r314, %r315, %r316}, [%r336+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r317, %r318, %r319, %r320}, [%r336+12288];
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	add.s32 	%r337, %r332, %r23;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r165, %r166, %r193, %r194}, [%r337+32768];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r167, %r168, %r199, %r200}, [%r337+36864];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r169, %r170, %r205, %r206}, [%r337+40960];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r171, %r172, %r211, %r212}, [%r337+45056];
	add.s32 	%r338, %r332, %r24;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r277, %r278, %r301, %r302}, [%r338+32768];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r279, %r280, %r303, %r304}, [%r338+36864];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r281, %r282, %r305, %r306}, [%r338+40960];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r283, %r284, %r307, %r308}, [%r338+45056];
	.loc	1 179 36                        // sk03_fa_qkv.py:179:36
	mov.b32 	%r185, %r149;
	mov.b32 	%r186, %r149;
	mov.b32 	%r187, %r149;
	mov.b32 	%r188, %r149;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r185, %r186, %r187, %r188 }, { %r161, %r162, %r163, %r164 }, { %r165, %r166 }, { %r185, %r186, %r187, %r188 };
	// end inline asm
	mov.b32 	%r195, %r149;
	mov.b32 	%r196, %r149;
	mov.b32 	%r197, %r149;
	mov.b32 	%r198, %r149;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r195, %r196, %r197, %r198 }, { %r161, %r162, %r163, %r164 }, { %r167, %r168 }, { %r195, %r196, %r197, %r198 };
	// end inline asm
	mov.b32 	%r201, %r149;
	mov.b32 	%r202, %r149;
	mov.b32 	%r203, %r149;
	mov.b32 	%r204, %r149;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r201, %r202, %r203, %r204 }, { %r161, %r162, %r163, %r164 }, { %r169, %r170 }, { %r201, %r202, %r203, %r204 };
	// end inline asm
	mov.b32 	%r207, %r149;
	mov.b32 	%r208, %r149;
	mov.b32 	%r209, %r149;
	mov.b32 	%r210, %r149;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r207, %r208, %r209, %r210 }, { %r161, %r162, %r163, %r164 }, { %r171, %r172 }, { %r207, %r208, %r209, %r210 };
	// end inline asm
	mov.b32 	%r213, %r149;
	mov.b32 	%r214, %r149;
	mov.b32 	%r215, %r149;
	mov.b32 	%r216, %r149;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r213, %r214, %r215, %r216 }, { %r173, %r174, %r175, %r176 }, { %r165, %r166 }, { %r213, %r214, %r215, %r216 };
	// end inline asm
	mov.b32 	%r221, %r149;
	mov.b32 	%r222, %r149;
	mov.b32 	%r223, %r149;
	mov.b32 	%r224, %r149;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r221, %r222, %r223, %r224 }, { %r173, %r174, %r175, %r176 }, { %r167, %r168 }, { %r221, %r222, %r223, %r224 };
	// end inline asm
	mov.b32 	%r225, %r149;
	mov.b32 	%r226, %r149;
	mov.b32 	%r227, %r149;
	mov.b32 	%r228, %r149;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r225, %r226, %r227, %r228 }, { %r173, %r174, %r175, %r176 }, { %r169, %r170 }, { %r225, %r226, %r227, %r228 };
	// end inline asm
	mov.b32 	%r229, %r149;
	mov.b32 	%r230, %r149;
	mov.b32 	%r231, %r149;
	mov.b32 	%r232, %r149;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r229, %r230, %r231, %r232 }, { %r173, %r174, %r175, %r176 }, { %r171, %r172 }, { %r229, %r230, %r231, %r232 };
	// end inline asm
	mov.b32 	%r233, %r149;
	mov.b32 	%r234, %r149;
	mov.b32 	%r235, %r149;
	mov.b32 	%r236, %r149;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r233, %r234, %r235, %r236 }, { %r177, %r178, %r179, %r180 }, { %r165, %r166 }, { %r233, %r234, %r235, %r236 };
	// end inline asm
	mov.b32 	%r241, %r149;
	mov.b32 	%r242, %r149;
	mov.b32 	%r243, %r149;
	mov.b32 	%r244, %r149;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r241, %r242, %r243, %r244 }, { %r177, %r178, %r179, %r180 }, { %r167, %r168 }, { %r241, %r242, %r243, %r244 };
	// end inline asm
	mov.b32 	%r245, %r149;
	mov.b32 	%r246, %r149;
	mov.b32 	%r247, %r149;
	mov.b32 	%r248, %r149;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r245, %r246, %r247, %r248 }, { %r177, %r178, %r179, %r180 }, { %r169, %r170 }, { %r245, %r246, %r247, %r248 };
	// end inline asm
	mov.b32 	%r249, %r149;
	mov.b32 	%r250, %r149;
	mov.b32 	%r251, %r149;
	mov.b32 	%r252, %r149;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r249, %r250, %r251, %r252 }, { %r177, %r178, %r179, %r180 }, { %r171, %r172 }, { %r249, %r250, %r251, %r252 };
	// end inline asm
	mov.b32 	%r253, %r149;
	mov.b32 	%r254, %r149;
	mov.b32 	%r255, %r149;
	mov.b32 	%r256, %r149;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r253, %r254, %r255, %r256 }, { %r181, %r182, %r183, %r184 }, { %r165, %r166 }, { %r253, %r254, %r255, %r256 };
	// end inline asm
	mov.b32 	%r261, %r149;
	mov.b32 	%r262, %r149;
	mov.b32 	%r263, %r149;
	mov.b32 	%r264, %r149;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r261, %r262, %r263, %r264 }, { %r181, %r182, %r183, %r184 }, { %r167, %r168 }, { %r261, %r262, %r263, %r264 };
	// end inline asm
	mov.b32 	%r265, %r149;
	mov.b32 	%r266, %r149;
	mov.b32 	%r267, %r149;
	mov.b32 	%r268, %r149;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r265, %r266, %r267, %r268 }, { %r181, %r182, %r183, %r184 }, { %r169, %r170 }, { %r265, %r266, %r267, %r268 };
	// end inline asm
	mov.b32 	%r272, %r149;
	mov.b32 	%r269, %r149;
	mov.b32 	%r270, %r149;
	mov.b32 	%r271, %r149;
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
	.loc	1 181 39                        // sk03_fa_qkv.py:181:39
	cvt.s64.s32 	%rd96, %r742;
	add.s64 	%rd80, %rd10, %rd96;
	add.s64 	%rd81, %rd11, %rd96;
	add.s64 	%rd82, %rd12, %rd96;
	add.s64 	%rd83, %rd13, %rd96;
	add.s64 	%rd84, %rd14, %rd96;
	add.s64 	%rd85, %rd15, %rd96;
	add.s64 	%rd86, %rd16, %rd96;
	add.s64 	%rd87, %rd17, %rd96;
	.loc	1 181 29                        // sk03_fa_qkv.py:181:29
	// begin inline asm
	mov.u16 %rs1, 0x0;
	ld.global.b8 { %rs1 }, [ %rd80 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs2, 0x0;
	ld.global.b8 { %rs2 }, [ %rd81 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs3, 0x0;
	ld.global.b8 { %rs3 }, [ %rd82 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs4, 0x0;
	ld.global.b8 { %rs4 }, [ %rd83 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs5, 0x0;
	ld.global.b8 { %rs5 }, [ %rd84 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs6, 0x0;
	ld.global.b8 { %rs6 }, [ %rd85 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs7, 0x0;
	ld.global.b8 { %rs7 }, [ %rd86 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs8, 0x0;
	ld.global.b8 { %rs8 }, [ %rd87 + 0 ];
	// end inline asm
	.loc	1 181 21                        // sk03_fa_qkv.py:181:21
	cvt.u32.u16 	%r339, %rs1;
	and.b32 	%r340, %r339, 255;
	cvt.u32.u16 	%r341, %rs2;
	and.b32 	%r342, %r341, 255;
	cvt.u32.u16 	%r343, %rs3;
	and.b32 	%r344, %r343, 255;
	cvt.u32.u16 	%r345, %rs4;
	and.b32 	%r346, %r345, 255;
	cvt.u32.u16 	%r347, %rs5;
	and.b32 	%r348, %r347, 255;
	cvt.u32.u16 	%r349, %rs6;
	and.b32 	%r350, %r349, 255;
	cvt.u32.u16 	%r351, %rs7;
	and.b32 	%r352, %r351, 255;
	cvt.u32.u16 	%r353, %rs8;
	and.b32 	%r354, %r353, 255;
	shl.b32 	%r355, %r208, %r354;
	shl.b32 	%r356, %r210, %r354;
	shl.b32 	%r357, %r230, %r354;
	shl.b32 	%r358, %r232, %r354;
	shl.b32 	%r359, %r250, %r354;
	shl.b32 	%r360, %r252, %r354;
	shl.b32 	%r361, %r270, %r354;
	shl.b32 	%r362, %r272, %r354;
	shl.b32 	%r363, %r207, %r352;
	shl.b32 	%r364, %r209, %r352;
	shl.b32 	%r365, %r229, %r352;
	shl.b32 	%r366, %r231, %r352;
	shl.b32 	%r367, %r249, %r352;
	shl.b32 	%r368, %r251, %r352;
	shl.b32 	%r369, %r269, %r352;
	shl.b32 	%r370, %r271, %r352;
	shl.b32 	%r371, %r202, %r350;
	shl.b32 	%r372, %r204, %r350;
	shl.b32 	%r373, %r226, %r350;
	shl.b32 	%r374, %r228, %r350;
	shl.b32 	%r375, %r246, %r350;
	shl.b32 	%r376, %r248, %r350;
	shl.b32 	%r377, %r266, %r350;
	shl.b32 	%r378, %r268, %r350;
	shl.b32 	%r379, %r201, %r348;
	shl.b32 	%r380, %r203, %r348;
	shl.b32 	%r381, %r225, %r348;
	shl.b32 	%r382, %r227, %r348;
	shl.b32 	%r383, %r245, %r348;
	shl.b32 	%r384, %r247, %r348;
	shl.b32 	%r385, %r265, %r348;
	shl.b32 	%r386, %r267, %r348;
	shl.b32 	%r387, %r196, %r346;
	shl.b32 	%r388, %r198, %r346;
	shl.b32 	%r389, %r222, %r346;
	shl.b32 	%r390, %r224, %r346;
	shl.b32 	%r391, %r242, %r346;
	shl.b32 	%r392, %r244, %r346;
	shl.b32 	%r393, %r262, %r346;
	shl.b32 	%r394, %r264, %r346;
	shl.b32 	%r395, %r195, %r344;
	shl.b32 	%r396, %r197, %r344;
	shl.b32 	%r397, %r221, %r344;
	shl.b32 	%r398, %r223, %r344;
	shl.b32 	%r399, %r241, %r344;
	shl.b32 	%r400, %r243, %r344;
	shl.b32 	%r401, %r261, %r344;
	shl.b32 	%r402, %r263, %r344;
	shl.b32 	%r403, %r186, %r342;
	shl.b32 	%r404, %r188, %r342;
	shl.b32 	%r405, %r214, %r342;
	shl.b32 	%r406, %r216, %r342;
	shl.b32 	%r407, %r234, %r342;
	shl.b32 	%r408, %r236, %r342;
	shl.b32 	%r409, %r254, %r342;
	shl.b32 	%r410, %r256, %r342;
	shl.b32 	%r411, %r185, %r340;
	shl.b32 	%r412, %r187, %r340;
	shl.b32 	%r413, %r213, %r340;
	shl.b32 	%r414, %r215, %r340;
	shl.b32 	%r415, %r233, %r340;
	shl.b32 	%r416, %r235, %r340;
	shl.b32 	%r417, %r253, %r340;
	shl.b32 	%r418, %r255, %r340;
	.loc	1 182 15                        // sk03_fa_qkv.py:182:15
	add.s32 	%r795, %r418, %r795;
	add.s32 	%r793, %r417, %r793;
	add.s32 	%r779, %r416, %r779;
	add.s32 	%r777, %r415, %r777;
	add.s32 	%r763, %r414, %r763;
	add.s32 	%r761, %r413, %r761;
	add.s32 	%r747, %r412, %r747;
	add.s32 	%r745, %r411, %r745;
	add.s32 	%r796, %r410, %r796;
	add.s32 	%r794, %r409, %r794;
	add.s32 	%r780, %r408, %r780;
	add.s32 	%r778, %r407, %r778;
	add.s32 	%r764, %r406, %r764;
	add.s32 	%r762, %r405, %r762;
	add.s32 	%r748, %r404, %r748;
	add.s32 	%r746, %r403, %r746;
	add.s32 	%r799, %r402, %r799;
	add.s32 	%r797, %r401, %r797;
	add.s32 	%r783, %r400, %r783;
	add.s32 	%r781, %r399, %r781;
	add.s32 	%r767, %r398, %r767;
	add.s32 	%r765, %r397, %r765;
	add.s32 	%r751, %r396, %r751;
	add.s32 	%r749, %r395, %r749;
	add.s32 	%r800, %r394, %r800;
	add.s32 	%r798, %r393, %r798;
	add.s32 	%r784, %r392, %r784;
	add.s32 	%r782, %r391, %r782;
	add.s32 	%r768, %r390, %r768;
	add.s32 	%r766, %r389, %r766;
	add.s32 	%r752, %r388, %r752;
	add.s32 	%r750, %r387, %r750;
	add.s32 	%r803, %r386, %r803;
	add.s32 	%r801, %r385, %r801;
	add.s32 	%r787, %r384, %r787;
	add.s32 	%r785, %r383, %r785;
	add.s32 	%r771, %r382, %r771;
	add.s32 	%r769, %r381, %r769;
	add.s32 	%r755, %r380, %r755;
	add.s32 	%r753, %r379, %r753;
	add.s32 	%r804, %r378, %r804;
	add.s32 	%r802, %r377, %r802;
	add.s32 	%r788, %r376, %r788;
	add.s32 	%r786, %r375, %r786;
	add.s32 	%r772, %r374, %r772;
	add.s32 	%r770, %r373, %r770;
	add.s32 	%r756, %r372, %r756;
	add.s32 	%r754, %r371, %r754;
	add.s32 	%r807, %r370, %r807;
	add.s32 	%r805, %r369, %r805;
	add.s32 	%r791, %r368, %r791;
	add.s32 	%r789, %r367, %r789;
	add.s32 	%r775, %r366, %r775;
	add.s32 	%r773, %r365, %r773;
	add.s32 	%r759, %r364, %r759;
	add.s32 	%r757, %r363, %r757;
	add.s32 	%r808, %r362, %r808;
	add.s32 	%r806, %r361, %r806;
	add.s32 	%r792, %r360, %r792;
	add.s32 	%r790, %r359, %r790;
	add.s32 	%r776, %r358, %r776;
	add.s32 	%r774, %r357, %r774;
	add.s32 	%r760, %r356, %r760;
	add.s32 	%r758, %r355, %r758;
	.loc	1 183 18                        // sk03_fa_qkv.py:183:18
	add.s64 	%rd88, %rd27, %rd140;
	add.s64 	%rd89, %rd26, %rd140;
	add.s64 	%rd90, %rd25, %rd140;
	.loc	1 184 18                        // sk03_fa_qkv.py:184:18
	add.s64 	%rd91, %rd24, %rd140;
	add.s64 	%rd92, %rd23, %rd140;
	add.s64 	%rd93, %rd22, %rd140;
	add.s64 	%rd94, %rd21, %rd140;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	add.s64 	%rd95, %rd20, %rd140;
	add.s32 	%r419, %r744, 1;
	setp.gt.s32 	%p5, %r419, 1;
	selp.b32 	%r744, 0, %r419, %p5;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
	shl.b32 	%r420, %r744, 14;
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
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
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
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	add.s64 	%rd141, %rd141, 1;
	add.s64 	%rd140, %rd140, 128;
	add.s32 	%r742, %r742, %r30;
	setp.ne.b64 	%p6, %rd19, %rd140;
	@%p6 bra 	$L__BB0_3;
// %bb.4:                               // %._crit_edge.loopexit
	.loc	1 186 17                        // sk03_fa_qkv.py:186:17
	cvt.rn.f32.s32 	%r421, %r745;
	cvt.rn.f32.s32 	%r422, %r746;
	cvt.rn.bf16x2.f32 	%r811, %r422, %r421;
	cvt.rn.f32.s32 	%r423, %r747;
	cvt.rn.f32.s32 	%r424, %r748;
	cvt.rn.bf16x2.f32 	%r812, %r424, %r423;
	cvt.rn.f32.s32 	%r425, %r749;
	cvt.rn.f32.s32 	%r426, %r750;
	cvt.rn.bf16x2.f32 	%r813, %r426, %r425;
	cvt.rn.f32.s32 	%r427, %r751;
	cvt.rn.f32.s32 	%r428, %r752;
	cvt.rn.bf16x2.f32 	%r814, %r428, %r427;
	cvt.rn.f32.s32 	%r429, %r753;
	cvt.rn.f32.s32 	%r430, %r754;
	cvt.rn.bf16x2.f32 	%r815, %r430, %r429;
	cvt.rn.f32.s32 	%r431, %r755;
	cvt.rn.f32.s32 	%r432, %r756;
	cvt.rn.bf16x2.f32 	%r816, %r432, %r431;
	cvt.rn.f32.s32 	%r433, %r757;
	cvt.rn.f32.s32 	%r434, %r758;
	cvt.rn.bf16x2.f32 	%r817, %r434, %r433;
	cvt.rn.f32.s32 	%r435, %r759;
	cvt.rn.f32.s32 	%r436, %r760;
	cvt.rn.bf16x2.f32 	%r818, %r436, %r435;
	cvt.rn.f32.s32 	%r437, %r761;
	cvt.rn.f32.s32 	%r438, %r762;
	cvt.rn.bf16x2.f32 	%r819, %r438, %r437;
	cvt.rn.f32.s32 	%r439, %r763;
	cvt.rn.f32.s32 	%r440, %r764;
	cvt.rn.bf16x2.f32 	%r820, %r440, %r439;
	cvt.rn.f32.s32 	%r441, %r765;
	cvt.rn.f32.s32 	%r442, %r766;
	cvt.rn.bf16x2.f32 	%r821, %r442, %r441;
	cvt.rn.f32.s32 	%r443, %r767;
	cvt.rn.f32.s32 	%r444, %r768;
	cvt.rn.bf16x2.f32 	%r822, %r444, %r443;
	cvt.rn.f32.s32 	%r445, %r769;
	cvt.rn.f32.s32 	%r446, %r770;
	cvt.rn.bf16x2.f32 	%r823, %r446, %r445;
	cvt.rn.f32.s32 	%r447, %r771;
	cvt.rn.f32.s32 	%r448, %r772;
	cvt.rn.bf16x2.f32 	%r824, %r448, %r447;
	cvt.rn.f32.s32 	%r449, %r773;
	cvt.rn.f32.s32 	%r450, %r774;
	cvt.rn.bf16x2.f32 	%r825, %r450, %r449;
	cvt.rn.f32.s32 	%r451, %r775;
	cvt.rn.f32.s32 	%r452, %r776;
	cvt.rn.bf16x2.f32 	%r826, %r452, %r451;
	cvt.rn.f32.s32 	%r453, %r777;
	cvt.rn.f32.s32 	%r454, %r778;
	cvt.rn.bf16x2.f32 	%r827, %r454, %r453;
	cvt.rn.f32.s32 	%r455, %r779;
	cvt.rn.f32.s32 	%r456, %r780;
	cvt.rn.bf16x2.f32 	%r828, %r456, %r455;
	cvt.rn.f32.s32 	%r457, %r781;
	cvt.rn.f32.s32 	%r458, %r782;
	cvt.rn.bf16x2.f32 	%r829, %r458, %r457;
	cvt.rn.f32.s32 	%r459, %r783;
	cvt.rn.f32.s32 	%r460, %r784;
	cvt.rn.bf16x2.f32 	%r830, %r460, %r459;
	cvt.rn.f32.s32 	%r461, %r785;
	cvt.rn.f32.s32 	%r462, %r786;
	cvt.rn.bf16x2.f32 	%r831, %r462, %r461;
	cvt.rn.f32.s32 	%r463, %r787;
	cvt.rn.f32.s32 	%r464, %r788;
	cvt.rn.bf16x2.f32 	%r832, %r464, %r463;
	cvt.rn.f32.s32 	%r465, %r789;
	cvt.rn.f32.s32 	%r466, %r790;
	cvt.rn.bf16x2.f32 	%r833, %r466, %r465;
	cvt.rn.f32.s32 	%r467, %r791;
	cvt.rn.f32.s32 	%r468, %r792;
	cvt.rn.bf16x2.f32 	%r834, %r468, %r467;
	cvt.rn.f32.s32 	%r469, %r793;
	cvt.rn.f32.s32 	%r470, %r794;
	cvt.rn.bf16x2.f32 	%r835, %r470, %r469;
	cvt.rn.f32.s32 	%r471, %r795;
	cvt.rn.f32.s32 	%r472, %r796;
	cvt.rn.bf16x2.f32 	%r836, %r472, %r471;
	cvt.rn.f32.s32 	%r473, %r797;
	cvt.rn.f32.s32 	%r474, %r798;
	cvt.rn.bf16x2.f32 	%r837, %r474, %r473;
	cvt.rn.f32.s32 	%r475, %r799;
	cvt.rn.f32.s32 	%r476, %r800;
	cvt.rn.bf16x2.f32 	%r838, %r476, %r475;
	cvt.rn.f32.s32 	%r477, %r801;
	cvt.rn.f32.s32 	%r478, %r802;
	cvt.rn.bf16x2.f32 	%r839, %r478, %r477;
	cvt.rn.f32.s32 	%r479, %r803;
	cvt.rn.f32.s32 	%r480, %r804;
	cvt.rn.bf16x2.f32 	%r840, %r480, %r479;
	cvt.rn.f32.s32 	%r481, %r805;
	cvt.rn.f32.s32 	%r482, %r806;
	cvt.rn.bf16x2.f32 	%r841, %r482, %r481;
	cvt.rn.f32.s32 	%r483, %r807;
	cvt.rn.f32.s32 	%r484, %r808;
	cvt.rn.bf16x2.f32 	%r842, %r484, %r483;
	bra.uni 	$L__BB0_5;
$L__BB0_1:                              // %.._crit_edge_crit_edge
	.loc	1 188 19                        // sk03_fa_qkv.py:188:19
	shl.b32 	%r810, %r2, 1;
	.loc	1 194 8                         // sk03_fa_qkv.py:194:8
	and.b32 	%r809, %r2, 16;
	mov.b32 	%r811, 0;
	mov.b32 	%r812, %r811;
	mov.b32 	%r813, %r811;
	mov.b32 	%r814, %r811;
	mov.b32 	%r815, %r811;
	mov.b32 	%r816, %r811;
	mov.b32 	%r817, %r811;
	mov.b32 	%r818, %r811;
	mov.b32 	%r819, %r811;
	mov.b32 	%r820, %r811;
	mov.b32 	%r821, %r811;
	mov.b32 	%r822, %r811;
	mov.b32 	%r823, %r811;
	mov.b32 	%r824, %r811;
	mov.b32 	%r825, %r811;
	mov.b32 	%r826, %r811;
	mov.b32 	%r827, %r811;
	mov.b32 	%r828, %r811;
	mov.b32 	%r829, %r811;
	mov.b32 	%r830, %r811;
	mov.b32 	%r831, %r811;
	mov.b32 	%r832, %r811;
	mov.b32 	%r833, %r811;
	mov.b32 	%r834, %r811;
	mov.b32 	%r835, %r811;
	mov.b32 	%r836, %r811;
	mov.b32 	%r837, %r811;
	mov.b32 	%r838, %r811;
	mov.b32 	%r839, %r811;
	mov.b32 	%r840, %r811;
	mov.b32 	%r841, %r811;
	mov.b32 	%r842, %r811;
$L__BB0_5:                              // %._crit_edge
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	or.b32 	%r607, %r12, %r741;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r608, %r607, 8;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r609, %r608, %r26;
	rem.s32 	%r610, %r607, %r26;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	shl.b32 	%r611, %r7, 3;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r612, %r12, %r611;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	and.b32 	%r613, %r2, 128;
	shr.u32 	%r614, %r613, 3;
	shr.u32 	%r615, %r2, 2;
	bfe.u32 	%r616, %r2, 2, 3;
	or.b32 	%r617, %r614, %r616;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r618, %r617, %r1;
	or.b32 	%r619, %r618, 104;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r620, %r619, %r25;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r621, %r618, 96;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r622, %r621, %r25;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r623, %r618, 72;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r624, %r623, %r25;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r625, %r618, 64;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r626, %r625, %r25;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r627, %r618, 40;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r628, %r627, %r25;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r629, %r618, 32;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r630, %r629, %r25;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r631, %r618, 8;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r632, %r631, %r25;
	rem.s32 	%r633, %r618, %r25;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	shr.u32 	%r634, %r2, 4;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r635, %r634, %r1;
	or.b32 	%r636, %r635, 112;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	bfe.u32 	%r637, %r2, 4, 4;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r638, %r637, %r1;
	or.b32 	%r639, %r638, 96;
	or.b32 	%r640, %r638, 80;
	or.b32 	%r641, %r638, 64;
	or.b32 	%r642, %r635, 48;
	or.b32 	%r643, %r638, 32;
	or.b32 	%r644, %r638, 16;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 186 54                        // sk03_fa_qkv.py:186:54
	mad.wide.s32 	%rd97, %r633, 4, %rd32;
	mad.wide.s32 	%rd98, %r632, 4, %rd32;
	mad.wide.s32 	%rd99, %r630, 4, %rd32;
	mad.wide.s32 	%rd100, %r628, 4, %rd32;
	mad.wide.s32 	%rd101, %r626, 4, %rd32;
	mad.wide.s32 	%rd102, %r624, 4, %rd32;
	mad.wide.s32 	%rd103, %r622, 4, %rd32;
	mad.wide.s32 	%rd104, %r620, 4, %rd32;
	.loc	1 186 40                        // sk03_fa_qkv.py:186:40
	// begin inline asm
	mov.u32 %r485, 0x0;
	ld.global.b32 { %r485 }, [ %rd97 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r486, 0x0;
	ld.global.b32 { %r486 }, [ %rd98 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r487, 0x0;
	ld.global.b32 { %r487 }, [ %rd99 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r488, 0x0;
	ld.global.b32 { %r488 }, [ %rd100 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r489, 0x0;
	ld.global.b32 { %r489 }, [ %rd101 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r490, 0x0;
	ld.global.b32 { %r490 }, [ %rd102 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r491, 0x0;
	ld.global.b32 { %r491 }, [ %rd103 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r492, 0x0;
	ld.global.b32 { %r492 }, [ %rd104 + 0 ];
	// end inline asm
	.loc	1 186 65                        // sk03_fa_qkv.py:186:65
	cvt.rn.bf16.f32 	%rs9, %r485;
	cvt.rn.bf16.f32 	%rs10, %r486;
	cvt.rn.bf16.f32 	%rs11, %r487;
	cvt.rn.bf16.f32 	%rs12, %r488;
	cvt.rn.bf16.f32 	%rs13, %r489;
	cvt.rn.bf16.f32 	%rs14, %r490;
	cvt.rn.bf16.f32 	%rs15, %r491;
	cvt.rn.bf16.f32 	%rs16, %r492;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	mov.b32 	{%rs17, %rs18}, %r811;
	mov.b16 	%rs19, 0x8000;
	fma.rn.bf16 	%rs20, %rs17, %rs9, %rs19;
	fma.rn.bf16 	%rs21, %rs18, %rs9, %rs19;
	mov.b32 	{%rs22, %rs23}, %r812;
	fma.rn.bf16 	%rs24, %rs22, %rs10, %rs19;
	fma.rn.bf16 	%rs25, %rs23, %rs10, %rs19;
	mov.b32 	{%rs26, %rs27}, %r813;
	fma.rn.bf16 	%rs28, %rs26, %rs9, %rs19;
	fma.rn.bf16 	%rs29, %rs27, %rs9, %rs19;
	mov.b32 	{%rs30, %rs31}, %r814;
	fma.rn.bf16 	%rs32, %rs30, %rs10, %rs19;
	fma.rn.bf16 	%rs33, %rs31, %rs10, %rs19;
	mov.b32 	{%rs34, %rs35}, %r815;
	fma.rn.bf16 	%rs36, %rs34, %rs9, %rs19;
	fma.rn.bf16 	%rs37, %rs35, %rs9, %rs19;
	mov.b32 	{%rs38, %rs39}, %r816;
	fma.rn.bf16 	%rs40, %rs38, %rs10, %rs19;
	fma.rn.bf16 	%rs41, %rs39, %rs10, %rs19;
	mov.b32 	{%rs42, %rs43}, %r817;
	fma.rn.bf16 	%rs44, %rs42, %rs9, %rs19;
	fma.rn.bf16 	%rs45, %rs43, %rs9, %rs19;
	mov.b32 	{%rs46, %rs47}, %r818;
	fma.rn.bf16 	%rs48, %rs46, %rs10, %rs19;
	fma.rn.bf16 	%rs49, %rs47, %rs10, %rs19;
	mov.b32 	{%rs50, %rs51}, %r819;
	fma.rn.bf16 	%rs52, %rs50, %rs11, %rs19;
	fma.rn.bf16 	%rs53, %rs51, %rs11, %rs19;
	mov.b32 	{%rs54, %rs55}, %r820;
	fma.rn.bf16 	%rs56, %rs54, %rs12, %rs19;
	fma.rn.bf16 	%rs57, %rs55, %rs12, %rs19;
	mov.b32 	{%rs58, %rs59}, %r821;
	fma.rn.bf16 	%rs60, %rs58, %rs11, %rs19;
	fma.rn.bf16 	%rs61, %rs59, %rs11, %rs19;
	mov.b32 	{%rs62, %rs63}, %r822;
	fma.rn.bf16 	%rs64, %rs62, %rs12, %rs19;
	fma.rn.bf16 	%rs65, %rs63, %rs12, %rs19;
	mov.b32 	{%rs66, %rs67}, %r823;
	fma.rn.bf16 	%rs68, %rs66, %rs11, %rs19;
	fma.rn.bf16 	%rs69, %rs67, %rs11, %rs19;
	mov.b32 	{%rs70, %rs71}, %r824;
	fma.rn.bf16 	%rs72, %rs70, %rs12, %rs19;
	fma.rn.bf16 	%rs73, %rs71, %rs12, %rs19;
	mov.b32 	{%rs74, %rs75}, %r825;
	fma.rn.bf16 	%rs76, %rs74, %rs11, %rs19;
	fma.rn.bf16 	%rs77, %rs75, %rs11, %rs19;
	mov.b32 	{%rs78, %rs79}, %r826;
	fma.rn.bf16 	%rs80, %rs78, %rs12, %rs19;
	fma.rn.bf16 	%rs81, %rs79, %rs12, %rs19;
	mov.b32 	{%rs82, %rs83}, %r827;
	fma.rn.bf16 	%rs84, %rs82, %rs13, %rs19;
	fma.rn.bf16 	%rs85, %rs83, %rs13, %rs19;
	mov.b32 	{%rs86, %rs87}, %r828;
	fma.rn.bf16 	%rs88, %rs86, %rs14, %rs19;
	fma.rn.bf16 	%rs89, %rs87, %rs14, %rs19;
	mov.b32 	{%rs90, %rs91}, %r829;
	fma.rn.bf16 	%rs92, %rs90, %rs13, %rs19;
	fma.rn.bf16 	%rs93, %rs91, %rs13, %rs19;
	mov.b32 	{%rs94, %rs95}, %r830;
	fma.rn.bf16 	%rs96, %rs94, %rs14, %rs19;
	fma.rn.bf16 	%rs97, %rs95, %rs14, %rs19;
	mov.b32 	{%rs98, %rs99}, %r831;
	fma.rn.bf16 	%rs100, %rs98, %rs13, %rs19;
	fma.rn.bf16 	%rs101, %rs99, %rs13, %rs19;
	mov.b32 	{%rs102, %rs103}, %r832;
	fma.rn.bf16 	%rs104, %rs102, %rs14, %rs19;
	fma.rn.bf16 	%rs105, %rs103, %rs14, %rs19;
	mov.b32 	{%rs106, %rs107}, %r833;
	fma.rn.bf16 	%rs108, %rs106, %rs13, %rs19;
	fma.rn.bf16 	%rs109, %rs107, %rs13, %rs19;
	mov.b32 	{%rs110, %rs111}, %r834;
	fma.rn.bf16 	%rs112, %rs110, %rs14, %rs19;
	fma.rn.bf16 	%rs113, %rs111, %rs14, %rs19;
	mov.b32 	{%rs114, %rs115}, %r836;
	fma.rn.bf16 	%rs116, %rs114, %rs16, %rs19;
	fma.rn.bf16 	%rs117, %rs115, %rs16, %rs19;
	mov.b32 	{%rs118, %rs119}, %r838;
	fma.rn.bf16 	%rs120, %rs118, %rs16, %rs19;
	fma.rn.bf16 	%rs121, %rs119, %rs16, %rs19;
	mov.b32 	{%rs122, %rs123}, %r840;
	fma.rn.bf16 	%rs124, %rs122, %rs16, %rs19;
	fma.rn.bf16 	%rs125, %rs123, %rs16, %rs19;
	mov.b32 	{%rs126, %rs127}, %r842;
	fma.rn.bf16 	%rs128, %rs126, %rs16, %rs19;
	fma.rn.bf16 	%rs129, %rs127, %rs16, %rs19;
	.loc	1 187 38                        // sk03_fa_qkv.py:187:38
	mad.wide.s32 	%rd105, %r13, 4, %rd33;
	mad.wide.s32 	%rd106, %r14, 4, %rd33;
	mad.wide.s32 	%rd107, %r15, 4, %rd33;
	mad.wide.s32 	%rd108, %r16, 4, %rd33;
	.loc	1 187 24                        // sk03_fa_qkv.py:187:24
	// begin inline asm
	mov.u32 %r493, 0x0;
	mov.u32 %r494, 0x0;
	ld.global.v2.b32 { %r493, %r494 }, [ %rd105 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r495, 0x0;
	mov.u32 %r496, 0x0;
	ld.global.v2.b32 { %r495, %r496 }, [ %rd106 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r497, 0x0;
	mov.u32 %r498, 0x0;
	ld.global.v2.b32 { %r497, %r498 }, [ %rd107 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r499, 0x0;
	mov.u32 %r500, 0x0;
	ld.global.v2.b32 { %r499, %r500 }, [ %rd108 + 0 ];
	// end inline asm
	.loc	1 188 49                        // sk03_fa_qkv.py:188:49
	mul.lo.s32 	%r645, %r8, %r29;
	mul.lo.s32 	%r646, %r9, %r29;
	mul.lo.s32 	%r647, %r10, %r29;
	mul.lo.s32 	%r648, %r11, %r29;
	.loc	1 188 31                        // sk03_fa_qkv.py:188:31
	mad.wide.s32 	%rd125, %r645, 2, %rd31;
	mad.wide.s32 	%rd126, %r646, 2, %rd31;
	mad.wide.s32 	%rd127, %r647, 2, %rd31;
	mad.wide.s32 	%rd128, %r648, 2, %rd31;
	.loc	1 188 64                        // sk03_fa_qkv.py:188:64
	mul.wide.s32 	%rd129, %r610, 2;
	add.s64 	%rd109, %rd125, %rd129;
	mul.wide.s32 	%rd130, %r609, 2;
	add.s64 	%rd110, %rd125, %rd130;
	add.s64 	%rd111, %rd126, %rd129;
	add.s64 	%rd112, %rd126, %rd130;
	add.s64 	%rd113, %rd127, %rd129;
	add.s64 	%rd114, %rd127, %rd130;
	add.s64 	%rd115, %rd128, %rd129;
	add.s64 	%rd116, %rd128, %rd130;
	.loc	1 188 19                        // sk03_fa_qkv.py:188:19
	// begin inline asm
	mov.u32 %r502, 0x0;
	mov.u32 %r503, 0x0;
	mov.u32 %r504, 0x0;
	mov.u32 %r505, 0x0;
	ld.global.v4.b32 { %r502, %r503, %r504, %r505 }, [ %rd109 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r507, 0x0;
	mov.u32 %r508, 0x0;
	mov.u32 %r509, 0x0;
	mov.u32 %r510, 0x0;
	ld.global.v4.b32 { %r507, %r508, %r509, %r510 }, [ %rd110 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r511, 0x0;
	mov.u32 %r512, 0x0;
	mov.u32 %r513, 0x0;
	mov.u32 %r514, 0x0;
	ld.global.v4.b32 { %r511, %r512, %r513, %r514 }, [ %rd111 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r515, 0x0;
	mov.u32 %r516, 0x0;
	mov.u32 %r517, 0x0;
	mov.u32 %r518, 0x0;
	ld.global.v4.b32 { %r515, %r516, %r517, %r518 }, [ %rd112 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r519, 0x0;
	mov.u32 %r520, 0x0;
	mov.u32 %r521, 0x0;
	mov.u32 %r522, 0x0;
	ld.global.v4.b32 { %r519, %r520, %r521, %r522 }, [ %rd113 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r523, 0x0;
	mov.u32 %r524, 0x0;
	mov.u32 %r525, 0x0;
	mov.u32 %r526, 0x0;
	ld.global.v4.b32 { %r523, %r524, %r525, %r526 }, [ %rd114 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r527, 0x0;
	mov.u32 %r528, 0x0;
	mov.u32 %r529, 0x0;
	mov.u32 %r530, 0x0;
	ld.global.v4.b32 { %r527, %r528, %r529, %r530 }, [ %rd115 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r531, 0x0;
	mov.u32 %r532, 0x0;
	mov.u32 %r533, 0x0;
	mov.u32 %r534, 0x0;
	ld.global.v4.b32 { %r531, %r532, %r533, %r534 }, [ %rd116 + 0 ];
	// end inline asm
	shl.b32 	%r649, %r18, 7;
	shl.b32 	%r650, %r3, 1;
	or.b32 	%r651, %r649, %r741;
	xor.b32 	%r652, %r651, %r650;
	add.s32 	%r501, %r148, %r652;
	// begin inline asm
	st.shared.v4.b32 [ %r501 + 0 ], { %r502, %r503, %r504, %r505 };
	// end inline asm
	add.s32 	%r506, %r501, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r506 + 0 ], { %r507, %r508, %r509, %r510 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r653, %r6, 10;
	and.b32 	%r654, %r17, 752;
	and.b32 	%r655, %r810, 288;
	and.b32 	%r656, %r615, 16;
	xor.b32 	%r657, %r654, %r655;
	xor.b32 	%r658, %r657, %r656;
	or.b32 	%r659, %r658, %r653;
	add.s32 	%r660, %r148, %r659;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r661, %r662, %r663, %r664}, [%r660];
	mov.b32 	{%rs130, %rs131}, %r661;
	mov.b32 	{%rs132, %rs133}, %r662;
	mov.b32 	{%rs134, %rs135}, %r663;
	mov.b32 	{%rs136, %rs137}, %r664;
	xor.b32 	%r665, %r659, 64;
	add.s32 	%r666, %r148, %r665;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r667, %r668, %r669, %r670}, [%r666];
	mov.b32 	{%rs138, %rs139}, %r667;
	mov.b32 	{%rs140, %rs141}, %r668;
	mov.b32 	{%rs142, %rs143}, %r669;
	mov.b32 	{%rs144, %rs145}, %r670;
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r501 + 0 ], { %r511, %r512, %r513, %r514 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r506 + 0 ], { %r515, %r516, %r517, %r518 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r671, %r672, %r673, %r674}, [%r660];
	mov.b32 	{%rs146, %rs147}, %r671;
	mov.b32 	{%rs148, %rs149}, %r672;
	mov.b32 	{%rs150, %rs151}, %r673;
	mov.b32 	{%rs152, %rs153}, %r674;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r675, %r676, %r677, %r678}, [%r666];
	mov.b32 	{%rs154, %rs155}, %r675;
	mov.b32 	{%rs156, %rs157}, %r676;
	mov.b32 	{%rs158, %rs159}, %r677;
	mov.b32 	{%rs160, %rs161}, %r678;
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r501 + 0 ], { %r519, %r520, %r521, %r522 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r506 + 0 ], { %r523, %r524, %r525, %r526 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r679, %r680, %r681, %r682}, [%r660];
	mov.b32 	{%rs162, %rs163}, %r679;
	mov.b32 	{%rs164, %rs165}, %r680;
	mov.b32 	{%rs166, %rs167}, %r681;
	mov.b32 	{%rs168, %rs169}, %r682;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r683, %r684, %r685, %r686}, [%r666];
	mov.b32 	{%rs170, %rs171}, %r683;
	mov.b32 	{%rs172, %rs173}, %r684;
	mov.b32 	{%rs174, %rs175}, %r685;
	mov.b32 	{%rs176, %rs177}, %r686;
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r501 + 0 ], { %r527, %r528, %r529, %r530 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r506 + 0 ], { %r531, %r532, %r533, %r534 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r687, %r688, %r689, %r690}, [%r660];
	mov.b32 	{%rs178, %rs179}, %r688;
	mov.b32 	{%rs180, %rs181}, %r690;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r691, %r692, %r693, %r694}, [%r666];
	mov.b32 	{%rs182, %rs183}, %r692;
	mov.b32 	{%rs184, %rs185}, %r694;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	mov.b32 	%r695, {%rs15, %rs15};
	mov.b32 	%r696, -2147450880;
	fma.rn.bf16x2 	%r697, %r835, %r695, %r696;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs186, %r494;
	cvt.rn.bf16.f32 	%rs187, %r493;
	mov.b32 	%r698, {%rs187, %rs186};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs188, %rs20, %rs187, %rs130;
	fma.rn.bf16 	%rs189, %rs21, %rs186, %rs131;
	fma.rn.bf16 	%rs190, %rs24, %rs187, %rs132;
	fma.rn.bf16 	%rs191, %rs25, %rs186, %rs133;
	fma.rn.bf16 	%rs192, %rs52, %rs187, %rs146;
	fma.rn.bf16 	%rs193, %rs53, %rs186, %rs147;
	fma.rn.bf16 	%rs194, %rs56, %rs187, %rs148;
	fma.rn.bf16 	%rs195, %rs57, %rs186, %rs149;
	fma.rn.bf16 	%rs196, %rs84, %rs187, %rs162;
	fma.rn.bf16 	%rs197, %rs85, %rs186, %rs163;
	fma.rn.bf16 	%rs198, %rs88, %rs187, %rs164;
	fma.rn.bf16 	%rs199, %rs89, %rs186, %rs165;
	fma.rn.bf16x2 	%r539, %r697, %r698, %r687;
	fma.rn.bf16 	%rs200, %rs116, %rs187, %rs178;
	fma.rn.bf16 	%rs201, %rs117, %rs186, %rs179;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r699, %r837, %r695, %r696;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs202, %r496;
	cvt.rn.bf16.f32 	%rs203, %r495;
	mov.b32 	%r700, {%rs203, %rs202};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs204, %rs28, %rs203, %rs134;
	fma.rn.bf16 	%rs205, %rs29, %rs202, %rs135;
	fma.rn.bf16 	%rs206, %rs32, %rs203, %rs136;
	fma.rn.bf16 	%rs207, %rs33, %rs202, %rs137;
	fma.rn.bf16 	%rs208, %rs60, %rs203, %rs150;
	fma.rn.bf16 	%rs209, %rs61, %rs202, %rs151;
	fma.rn.bf16 	%rs210, %rs64, %rs203, %rs152;
	fma.rn.bf16 	%rs211, %rs65, %rs202, %rs153;
	fma.rn.bf16 	%rs212, %rs92, %rs203, %rs166;
	fma.rn.bf16 	%rs213, %rs93, %rs202, %rs167;
	fma.rn.bf16 	%rs214, %rs96, %rs203, %rs168;
	fma.rn.bf16 	%rs215, %rs97, %rs202, %rs169;
	fma.rn.bf16x2 	%r559, %r699, %r700, %r689;
	fma.rn.bf16 	%rs216, %rs120, %rs203, %rs180;
	fma.rn.bf16 	%rs217, %rs121, %rs202, %rs181;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r701, %r839, %r695, %r696;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs218, %r498;
	cvt.rn.bf16.f32 	%rs219, %r497;
	mov.b32 	%r702, {%rs219, %rs218};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs220, %rs36, %rs219, %rs138;
	fma.rn.bf16 	%rs221, %rs37, %rs218, %rs139;
	fma.rn.bf16 	%rs222, %rs40, %rs219, %rs140;
	fma.rn.bf16 	%rs223, %rs41, %rs218, %rs141;
	fma.rn.bf16 	%rs224, %rs68, %rs219, %rs154;
	fma.rn.bf16 	%rs225, %rs69, %rs218, %rs155;
	fma.rn.bf16 	%rs226, %rs72, %rs219, %rs156;
	fma.rn.bf16 	%rs227, %rs73, %rs218, %rs157;
	fma.rn.bf16 	%rs228, %rs100, %rs219, %rs170;
	fma.rn.bf16 	%rs229, %rs101, %rs218, %rs171;
	fma.rn.bf16 	%rs230, %rs104, %rs219, %rs172;
	fma.rn.bf16 	%rs231, %rs105, %rs218, %rs173;
	fma.rn.bf16x2 	%r549, %r701, %r702, %r691;
	fma.rn.bf16 	%rs232, %rs124, %rs219, %rs182;
	fma.rn.bf16 	%rs233, %rs125, %rs218, %rs183;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r703, %r841, %r695, %r696;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs234, %r500;
	cvt.rn.bf16.f32 	%rs235, %r499;
	mov.b32 	%r704, {%rs235, %rs234};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs236, %rs44, %rs235, %rs142;
	fma.rn.bf16 	%rs237, %rs45, %rs234, %rs143;
	fma.rn.bf16 	%rs238, %rs48, %rs235, %rs144;
	fma.rn.bf16 	%rs239, %rs49, %rs234, %rs145;
	fma.rn.bf16 	%rs240, %rs76, %rs235, %rs158;
	fma.rn.bf16 	%rs241, %rs77, %rs234, %rs159;
	fma.rn.bf16 	%rs242, %rs80, %rs235, %rs160;
	fma.rn.bf16 	%rs243, %rs81, %rs234, %rs161;
	fma.rn.bf16 	%rs244, %rs108, %rs235, %rs174;
	fma.rn.bf16 	%rs245, %rs109, %rs234, %rs175;
	fma.rn.bf16 	%rs246, %rs112, %rs235, %rs176;
	fma.rn.bf16 	%rs247, %rs113, %rs234, %rs177;
	fma.rn.bf16x2 	%r569, %r703, %r704, %r693;
	fma.rn.bf16 	%rs248, %rs128, %rs235, %rs184;
	fma.rn.bf16 	%rs249, %rs129, %rs234, %rs185;
	.loc	1 195 31                        // sk03_fa_qkv.py:195:31
	setp.lt.s32 	%p15, %r638, %r25;
	setp.lt.s32 	%p16, %r644, %r25;
	setp.lt.s32 	%p17, %r643, %r25;
	setp.lt.s32 	%p18, %r642, %r25;
	setp.lt.s32 	%p19, %r641, %r25;
	setp.lt.s32 	%p20, %r640, %r25;
	setp.lt.s32 	%p21, %r639, %r25;
	setp.lt.s32 	%p22, %r636, %r25;
	.loc	1 195 54                        // sk03_fa_qkv.py:195:54
	setp.lt.s32 	%p23, %r612, %r26;
	.loc	1 195 37                        // sk03_fa_qkv.py:195:37
	and.pred 	%p7, %p15, %p23;
	and.pred 	%p8, %p16, %p23;
	and.pred 	%p9, %p17, %p23;
	and.pred 	%p10, %p18, %p23;
	and.pred 	%p11, %p19, %p23;
	and.pred 	%p12, %p20, %p23;
	and.pred 	%p13, %p21, %p23;
	and.pred 	%p14, %p22, %p23;
	.loc	1 193 35                        // sk03_fa_qkv.py:193:35
	mul.lo.s32 	%r705, %r638, %r28;
	mul.lo.s32 	%r706, %r644, %r28;
	mul.lo.s32 	%r707, %r643, %r28;
	mul.lo.s32 	%r708, %r642, %r28;
	mul.lo.s32 	%r709, %r641, %r28;
	mul.lo.s32 	%r710, %r640, %r28;
	mul.lo.s32 	%r711, %r639, %r28;
	mul.lo.s32 	%r712, %r636, %r28;
	.loc	1 193 18                        // sk03_fa_qkv.py:193:18
	mad.wide.s32 	%rd131, %r705, 2, %rd30;
	mad.wide.s32 	%rd132, %r706, 2, %rd30;
	mad.wide.s32 	%rd133, %r707, 2, %rd30;
	mad.wide.s32 	%rd134, %r708, 2, %rd30;
	mad.wide.s32 	%rd135, %r709, 2, %rd30;
	mad.wide.s32 	%rd136, %r710, 2, %rd30;
	mad.wide.s32 	%rd137, %r711, 2, %rd30;
	mad.wide.s32 	%rd138, %r712, 2, %rd30;
	.loc	1 193 50                        // sk03_fa_qkv.py:193:50
	mul.wide.s32 	%rd139, %r612, 2;
	add.s64 	%rd117, %rd131, %rd139;
	add.s64 	%rd118, %rd132, %rd139;
	add.s64 	%rd119, %rd133, %rd139;
	add.s64 	%rd120, %rd134, %rd139;
	add.s64 	%rd121, %rd135, %rd139;
	add.s64 	%rd122, %rd136, %rd139;
	add.s64 	%rd123, %rd137, %rd139;
	add.s64 	%rd124, %rd138, %rd139;
	.loc	1 194 8                         // sk03_fa_qkv.py:194:8
	bar.sync 	0;
	shl.b32 	%r713, %r4, 13;
	shl.b32 	%r714, %r4, 5;
	and.b32 	%r715, %r17, 384;
	shr.u32 	%r716, %r5, 1;
	bfe.s32 	%r717, %r2, 2, 1;
	and.b32 	%r718, %r717, 4112;
	shl.b32 	%r719, %r613, 3;
	or.b32 	%r720, %r713, %r719;
	or.b32 	%r721, %r714, %r715;
	xor.b32 	%r722, %r718, %r716;
	xor.b32 	%r723, %r722, %r721;
	or.b32 	%r724, %r723, %r720;
	add.s32 	%r535, %r148, %r724;
	mov.b32 	%r536, {%rs188, %rs189};
	mov.b32 	%r537, {%rs192, %rs193};
	mov.b32 	%r538, {%rs196, %rs197};
	// begin inline asm
	st.shared.v4.b32 [ %r535 + 0 ], { %r536, %r537, %r538, %r539 };
	// end inline asm
	add.s32 	%r540, %r535, 512;
	mov.b32 	%r541, {%rs190, %rs191};
	mov.b32 	%r542, {%rs194, %rs195};
	mov.b32 	%r543, {%rs198, %rs199};
	mov.b32 	%r544, {%rs200, %rs201};
	// begin inline asm
	st.shared.v4.b32 [ %r540 + 0 ], { %r541, %r542, %r543, %r544 };
	// end inline asm
	add.s32 	%r545, %r535, 2048;
	mov.b32 	%r546, {%rs220, %rs221};
	mov.b32 	%r547, {%rs224, %rs225};
	mov.b32 	%r548, {%rs228, %rs229};
	// begin inline asm
	st.shared.v4.b32 [ %r545 + 0 ], { %r546, %r547, %r548, %r549 };
	// end inline asm
	add.s32 	%r550, %r535, 2560;
	mov.b32 	%r551, {%rs222, %rs223};
	mov.b32 	%r552, {%rs226, %rs227};
	mov.b32 	%r553, {%rs230, %rs231};
	mov.b32 	%r554, {%rs232, %rs233};
	// begin inline asm
	st.shared.v4.b32 [ %r550 + 0 ], { %r551, %r552, %r553, %r554 };
	// end inline asm
	xor.b32 	%r725, %r724, 64;
	add.s32 	%r555, %r148, %r725;
	mov.b32 	%r556, {%rs204, %rs205};
	mov.b32 	%r557, {%rs208, %rs209};
	mov.b32 	%r558, {%rs212, %rs213};
	// begin inline asm
	st.shared.v4.b32 [ %r555 + 0 ], { %r556, %r557, %r558, %r559 };
	// end inline asm
	add.s32 	%r560, %r555, 512;
	mov.b32 	%r561, {%rs206, %rs207};
	mov.b32 	%r562, {%rs210, %rs211};
	mov.b32 	%r563, {%rs214, %rs215};
	mov.b32 	%r564, {%rs216, %rs217};
	// begin inline asm
	st.shared.v4.b32 [ %r560 + 0 ], { %r561, %r562, %r563, %r564 };
	// end inline asm
	add.s32 	%r565, %r555, 2048;
	mov.b32 	%r566, {%rs236, %rs237};
	mov.b32 	%r567, {%rs240, %rs241};
	mov.b32 	%r568, {%rs244, %rs245};
	// begin inline asm
	st.shared.v4.b32 [ %r565 + 0 ], { %r566, %r567, %r568, %r569 };
	// end inline asm
	add.s32 	%r570, %r555, 2560;
	mov.b32 	%r571, {%rs238, %rs239};
	mov.b32 	%r572, {%rs242, %rs243};
	mov.b32 	%r573, {%rs246, %rs247};
	mov.b32 	%r574, {%rs248, %rs249};
	// begin inline asm
	st.shared.v4.b32 [ %r570 + 0 ], { %r571, %r572, %r573, %r574 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r726, %r2, 2;
	and.b32 	%r727, %r726, 896;
	shl.b32 	%r728, %r2, 8;
	and.b32 	%r729, %r728, 2048;
	setp.eq.b32 	%p24, %r809, 0;
	selp.b32 	%r730, 0, 4112, %p24;
	or.b32 	%r731, %r741, %r727;
	xor.b32 	%r732, %r731, %r730;
	or.b32 	%r733, %r732, %r729;
	add.s32 	%r734, %r148, %r733;
	ld.shared.v4.b32 	{%r575, %r583, %r591, %r599}, [%r734];
	ld.shared.v4.b32 	{%r579, %r587, %r595, %r603}, [%r734+1024];
	xor.b32 	%r735, %r733, 32;
	add.s32 	%r736, %r148, %r735;
	ld.shared.v4.b32 	{%r576, %r584, %r592, %r600}, [%r736+8192];
	ld.shared.v4.b32 	{%r580, %r588, %r596, %r604}, [%r736+9216];
	xor.b32 	%r737, %r733, 64;
	add.s32 	%r738, %r148, %r737;
	ld.shared.v4.b32 	{%r577, %r585, %r593, %r601}, [%r738+16384];
	ld.shared.v4.b32 	{%r581, %r589, %r597, %r605}, [%r738+17408];
	xor.b32 	%r739, %r733, 96;
	add.s32 	%r740, %r148, %r739;
	ld.shared.v4.b32 	{%r578, %r586, %r594, %r602}, [%r740+24576];
	ld.shared.v4.b32 	{%r582, %r590, %r598, %r606}, [%r740+25600];
	// begin inline asm
	@%p7 st.global.v4.b32 [ %rd117 + 0 ], { %r575, %r576, %r577, %r578 };
	// end inline asm
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd118 + 0 ], { %r579, %r580, %r581, %r582 };
	// end inline asm
	// begin inline asm
	@%p9 st.global.v4.b32 [ %rd119 + 0 ], { %r583, %r584, %r585, %r586 };
	// end inline asm
	// begin inline asm
	@%p10 st.global.v4.b32 [ %rd120 + 0 ], { %r587, %r588, %r589, %r590 };
	// end inline asm
	// begin inline asm
	@%p11 st.global.v4.b32 [ %rd121 + 0 ], { %r591, %r592, %r593, %r594 };
	// end inline asm
	// begin inline asm
	@%p12 st.global.v4.b32 [ %rd122 + 0 ], { %r595, %r596, %r597, %r598 };
	// end inline asm
	// begin inline asm
	@%p13 st.global.v4.b32 [ %rd123 + 0 ], { %r599, %r600, %r601, %r602 };
	// end inline asm
	// begin inline asm
	@%p14 st.global.v4.b32 [ %rd124 + 0 ], { %r603, %r604, %r605, %r606 };
	// end inline asm
	.loc	1 192 4                         // sk03_fa_qkv.py:192:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk03_fa_qkv.py"
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
.b32 157                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x96 DW_TAG_compile_unit
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
.b8 51
.b8 95
.b8 102
.b8 97
.b8 95
.b8 113
.b8 107
.b8 118
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
.b8 2                                   // Abbrev [2] 0x44:0x16 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 51
.b8 95
.b8 102
.b8 97
.b8 95
.b8 113
.b8 107
.b8 118
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x5a:0x46 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 68                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x6f:0x18 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 155                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x87:0x18 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 156                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_6 = _Nativo(
    "sk03_fa_qkv/tile128x128x128_shift1_abi15",
    _PTX_6, "_sk03_fa_qkv_kernel",
    warps=8, shared=65536,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 18],
    horneado={11: 1, 12: 1, 15: 1, 17: 1, 19: 1, 20: 128, 21: 128, 22: 128, 23: 8, 24: 128, 25: True},
    div16=[8, 9, 10, 13, 14, 16],
)

_PTX_7 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk03_fa_qkv_kernel     // -- Begin function _sk03_fa_qkv_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk03_fa_qkv_kernel
.visible .entry _sk03_fa_qkv_kernel(
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_6,
	.param .u32 _sk03_fa_qkv_kernel_param_7,
	.param .u32 _sk03_fa_qkv_kernel_param_8,
	.param .u32 _sk03_fa_qkv_kernel_param_9,
	.param .u32 _sk03_fa_qkv_kernel_param_10,
	.param .u32 _sk03_fa_qkv_kernel_param_11,
	.param .u32 _sk03_fa_qkv_kernel_param_12,
	.param .u32 _sk03_fa_qkv_kernel_param_13,
	.param .u32 _sk03_fa_qkv_kernel_param_14,
	.param .u32 _sk03_fa_qkv_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_16,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_17
)
.reqntid 256
{
	.reg .pred 	%p<25>;
	.reg .b16 	%rs<314>;
	.reg .b32 	%r<888>;
	.reg .b64 	%rd<212>;
	.loc	1 145 0                         // sk03_fa_qkv.py:145:0
$L__func_begin0:
	.loc	1 145 0                         // sk03_fa_qkv.py:145:0

// %bb.0:
	ld.param.b32 	%r30, [_sk03_fa_qkv_kernel_param_14];
	ld.param.b32 	%r29, [_sk03_fa_qkv_kernel_param_13];
	ld.param.b32 	%r28, [_sk03_fa_qkv_kernel_param_12];
	ld.param.b32 	%r27, [_sk03_fa_qkv_kernel_param_9];
	ld.param.b32 	%r26, [_sk03_fa_qkv_kernel_param_8];
	ld.param.b32 	%r25, [_sk03_fa_qkv_kernel_param_7];
	ld.param.b64 	%rd33, [_sk03_fa_qkv_kernel_param_5];
	ld.param.b64 	%rd32, [_sk03_fa_qkv_kernel_param_4];
	ld.param.b64 	%rd31, [_sk03_fa_qkv_kernel_param_3];
	ld.param.b64 	%rd30, [_sk03_fa_qkv_kernel_param_2];
	ld.param.b64 	%rd29, [_sk03_fa_qkv_kernel_param_1];
	ld.param.b64 	%rd28, [_sk03_fa_qkv_kernel_param_0];
$L__tmp0:
	.loc	1 154 24                        // sk03_fa_qkv.py:154:24
	mov.u32 	%r50, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk03_fa_qkv.py:155:27 ]
	add.s32 	%r51, %r25, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk03_fa_qkv.py:155:27 ]
	shr.s32 	%r52, %r51, 31;
	shr.u32 	%r53, %r52, 25;
	add.s32 	%r54, %r51, %r53;
	shr.s32 	%r55, %r54, 7;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk03_fa_qkv.py:156:27 ]
	add.s32 	%r56, %r26, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk03_fa_qkv.py:156:27 ]
	shr.s32 	%r57, %r56, 31;
	shr.u32 	%r58, %r57, 25;
	add.s32 	%r59, %r56, %r58;
	shr.s32 	%r60, %r59, 7;
$L__tmp3:
	.loc	1 157 29                        // sk03_fa_qkv.py:157:29
	shl.b32 	%r61, %r60, 3;
	.loc	1 158 22                        // sk03_fa_qkv.py:158:22
	div.s32 	%r62, %r50, %r61;
	.loc	1 158 38                        // sk03_fa_qkv.py:158:38
	shl.b32 	%r63, %r62, 3;
	.loc	1 159 30                        // sk03_fa_qkv.py:159:30
	sub.s32 	%r64, %r55, %r63;
	ld.param.b32 	%r65, [_sk03_fa_qkv_kernel_param_10];
	.loc	1 159 39                        // sk03_fa_qkv.py:159:39
	min.s32 	%r66, %r64, 8;
	ld.param.b32 	%r67, [_sk03_fa_qkv_kernel_param_11];
	.loc	1 160 30                        // sk03_fa_qkv.py:160:30
	mul.lo.s32 	%r68, %r62, %r61;
	sub.s32 	%r69, %r50, %r68;
	.loc	1 161 36                        // sk03_fa_qkv.py:161:36
	div.s32 	%r70, %r69, %r66;
	.loc	1 160 46                        // sk03_fa_qkv.py:160:46
	mul.lo.s32 	%r71, %r70, %r66;
	sub.s32 	%r72, %r69, %r71;
	.loc	1 160 23                        // sk03_fa_qkv.py:160:23
	add.s32 	%r73, %r72, %r63;
	.loc	1 163 22                        // sk03_fa_qkv.py:163:22
	shl.b32 	%r1, %r73, 7;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
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
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r82, %r1, %r74;
	or.b32 	%r83, %r1, %r75;
	or.b32 	%r84, %r1, %r76;
	or.b32 	%r85, %r1, %r77;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r8, %r82, %r25;
	rem.s32 	%r9, %r83, %r25;
	rem.s32 	%r10, %r84, %r25;
	rem.s32 	%r11, %r85, %r25;
	.loc	1 164 22                        // sk03_fa_qkv.py:164:22
	shl.b32 	%r12, %r70, 7;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r86, %r12, %r74;
	or.b32 	%r87, %r12, %r75;
	or.b32 	%r88, %r12, %r76;
	or.b32 	%r89, %r12, %r77;
	or.b32 	%r90, %r12, %r80;
	or.b32 	%r92, %r90, 32;
	or.b32 	%r94, %r90, 64;
	or.b32 	%r96, %r90, 96;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r98, %r86, %r26;
	rem.s32 	%r99, %r87, %r26;
	rem.s32 	%r100, %r88, %r26;
	rem.s32 	%r101, %r89, %r26;
	rem.s32 	%r13, %r90, %r26;
	rem.s32 	%r14, %r92, %r26;
	rem.s32 	%r15, %r94, %r26;
	rem.s32 	%r16, %r96, %r26;
	.loc	1 167 39                        // sk03_fa_qkv.py:167:39
	mul.lo.s32 	%r106, %r8, %r65;
	mul.lo.s32 	%r107, %r9, %r65;
	mul.lo.s32 	%r108, %r10, %r65;
	mul.lo.s32 	%r109, %r11, %r65;
	.loc	1 167 21                        // sk03_fa_qkv.py:167:21
	cvt.s64.s32 	%rd1, %r106;
	add.s64 	%rd51, %rd28, %rd1;
	cvt.s64.s32 	%rd2, %r107;
	add.s64 	%rd52, %rd28, %rd2;
	cvt.s64.s32 	%rd3, %r108;
	add.s64 	%rd53, %rd28, %rd3;
	cvt.s64.s32 	%rd4, %r109;
	add.s64 	%rd54, %rd28, %rd4;
	.loc	1 167 51                        // sk03_fa_qkv.py:167:51
	cvt.u64.u32 	%rd5, %r81;
	add.s64 	%rd34, %rd51, %rd5;
	add.s64 	%rd35, %rd52, %rd5;
	add.s64 	%rd36, %rd53, %rd5;
	add.s64 	%rd37, %rd54, %rd5;
	.loc	1 168 21                        // sk03_fa_qkv.py:168:21
	add.s64 	%rd55, %rd29, %rd5;
	.loc	1 168 69                        // sk03_fa_qkv.py:168:69
	mul.lo.s32 	%r110, %r98, %r67;
	mul.lo.s32 	%r111, %r99, %r67;
	mul.lo.s32 	%r112, %r100, %r67;
	mul.lo.s32 	%r113, %r101, %r67;
	.loc	1 168 51                        // sk03_fa_qkv.py:168:51
	cvt.s64.s32 	%rd6, %r110;
	add.s64 	%rd38, %rd55, %rd6;
	cvt.s64.s32 	%rd7, %r111;
	add.s64 	%rd39, %rd55, %rd7;
	cvt.s64.s32 	%rd8, %r112;
	add.s64 	%rd40, %rd55, %rd8;
	cvt.s64.s32 	%rd9, %r113;
	add.s64 	%rd41, %rd55, %rd9;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	setp.gt.s32 	%p1, %r27, 127;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
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
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
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
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	setp.gt.s32 	%p2, %r27, 255;
	.loc	1 183 18                        // sk03_fa_qkv.py:183:18
	add.s64 	%rd42, %rd34, 128;
	add.s64 	%rd43, %rd35, 128;
	add.s64 	%rd44, %rd36, 128;
	add.s64 	%rd45, %rd37, 128;
	.loc	1 184 18                        // sk03_fa_qkv.py:184:18
	add.s64 	%rd46, %rd38, 128;
	add.s64 	%rd47, %rd39, 128;
	add.s64 	%rd48, %rd40, 128;
	add.s64 	%rd49, %rd41, 128;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
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
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
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
	cvt.u32.u64 	%r786, %rd5;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	@%p1 bra 	$L__BB0_2;
	bra.uni 	$L__BB0_1;
$L__BB0_2:                              // %.lr.ph
	.loc	1 0 23                          // sk03_fa_qkv.py:0:23
	ld.param.b32 	%r31, [_sk03_fa_qkv_kernel_param_15];
	ld.param.b64 	%rd50, [_sk03_fa_qkv_kernel_param_6];
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
	.loc	1 178 28                        // sk03_fa_qkv.py:178:28
	shr.u32 	%r151, %r27, 7;
	add.s32 	%r152, %r151, -2;
	shl.b32 	%r153, %r7, 7;
	and.b32 	%r154, %r17, 2160;
	and.b32 	%r854, %r2, 16;
	or.b32 	%r155, %r153, %r154;
	xor.b32 	%r19, %r155, %r854;
	xor.b32 	%r20, %r19, 32;
	xor.b32 	%r21, %r19, 64;
	xor.b32 	%r22, %r19, 96;
	shl.b32 	%r156, %r6, 7;
	shl.b32 	%r157, %r5, 5;
	shl.b32 	%r855, %r2, 1;
	and.b32 	%r158, %r855, 48;
	or.b32 	%r159, %r156, %r157;
	xor.b32 	%r160, %r786, %r158;
	or.b32 	%r23, %r159, %r160;
	xor.b32 	%r24, %r23, 64;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	cvt.s64.s32 	%rd18, %r152;
	and.b32 	%r161, %r27, -128;
	cvt.u64.u32 	%rd19, %r161;
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
	mov.b32 	%r150, 0;
	mov.b32 	%r789, 1;
	mov.b32 	%r788, -1;
	mov.b64 	%rd210, 0;
	mov.b32 	%r787, %r150;
	mov.b64 	%rd211, %rd210;
	mov.b32 	%r790, %r150;
	mov.b32 	%r791, %r150;
	mov.b32 	%r792, %r150;
	mov.b32 	%r793, %r150;
	mov.b32 	%r794, %r150;
	mov.b32 	%r795, %r150;
	mov.b32 	%r796, %r150;
	mov.b32 	%r797, %r150;
	mov.b32 	%r798, %r150;
	mov.b32 	%r799, %r150;
	mov.b32 	%r800, %r150;
	mov.b32 	%r801, %r150;
	mov.b32 	%r802, %r150;
	mov.b32 	%r803, %r150;
	mov.b32 	%r804, %r150;
	mov.b32 	%r805, %r150;
	mov.b32 	%r806, %r150;
	mov.b32 	%r807, %r150;
	mov.b32 	%r808, %r150;
	mov.b32 	%r809, %r150;
	mov.b32 	%r810, %r150;
	mov.b32 	%r811, %r150;
	mov.b32 	%r812, %r150;
	mov.b32 	%r813, %r150;
	mov.b32 	%r814, %r150;
	mov.b32 	%r815, %r150;
	mov.b32 	%r816, %r150;
	mov.b32 	%r817, %r150;
	mov.b32 	%r818, %r150;
	mov.b32 	%r819, %r150;
	mov.b32 	%r820, %r150;
	mov.b32 	%r821, %r150;
	mov.b32 	%r822, %r150;
	mov.b32 	%r823, %r150;
	mov.b32 	%r824, %r150;
	mov.b32 	%r825, %r150;
	mov.b32 	%r826, %r150;
	mov.b32 	%r827, %r150;
	mov.b32 	%r828, %r150;
	mov.b32 	%r829, %r150;
	mov.b32 	%r830, %r150;
	mov.b32 	%r831, %r150;
	mov.b32 	%r832, %r150;
	mov.b32 	%r833, %r150;
	mov.b32 	%r834, %r150;
	mov.b32 	%r835, %r150;
	mov.b32 	%r836, %r150;
	mov.b32 	%r837, %r150;
	mov.b32 	%r838, %r150;
	mov.b32 	%r839, %r150;
	mov.b32 	%r840, %r150;
	mov.b32 	%r841, %r150;
	mov.b32 	%r842, %r150;
	mov.b32 	%r843, %r150;
	mov.b32 	%r844, %r150;
	mov.b32 	%r845, %r150;
	mov.b32 	%r846, %r150;
	mov.b32 	%r847, %r150;
	mov.b32 	%r848, %r150;
	mov.b32 	%r849, %r150;
	mov.b32 	%r850, %r150;
	mov.b32 	%r851, %r150;
	mov.b32 	%r852, %r150;
	mov.b32 	%r853, %r150;
$L__BB0_3:                              // =>This Inner Loop Header: Depth=1
	setp.lt.s64 	%p3, %rd211, %rd18;
	add.s32 	%r331, %r788, 1;
	setp.gt.s32 	%p4, %r331, 1;
	selp.b32 	%r788, 0, %r331, %p4;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
	cp.async.wait_group 	2;
	bar.sync 	0;
	shl.b32 	%r332, %r788, 14;
	add.s32 	%r333, %r149, %r332;
	add.s32 	%r334, %r333, %r19;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r162, %r163, %r164, %r165}, [%r334];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r174, %r175, %r176, %r177}, [%r334+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r178, %r179, %r180, %r181}, [%r334+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r182, %r183, %r184, %r185}, [%r334+12288];
	add.s32 	%r335, %r333, %r20;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r190, %r191, %r192, %r193}, [%r335];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r218, %r219, %r220, %r221}, [%r335+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r238, %r239, %r240, %r241}, [%r335+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r258, %r259, %r260, %r261}, [%r335+12288];
	add.s32 	%r336, %r333, %r21;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r274, %r275, %r276, %r277}, [%r336];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r286, %r287, %r288, %r289}, [%r336+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r290, %r291, %r292, %r293}, [%r336+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r294, %r295, %r296, %r297}, [%r336+12288];
	add.s32 	%r337, %r333, %r22;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r298, %r299, %r300, %r301}, [%r337];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r310, %r311, %r312, %r313}, [%r337+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r314, %r315, %r316, %r317}, [%r337+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r318, %r319, %r320, %r321}, [%r337+12288];
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	add.s32 	%r338, %r333, %r23;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r166, %r167, %r194, %r195}, [%r338+32768];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r168, %r169, %r200, %r201}, [%r338+36864];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r170, %r171, %r206, %r207}, [%r338+40960];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r172, %r173, %r212, %r213}, [%r338+45056];
	add.s32 	%r339, %r333, %r24;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r278, %r279, %r302, %r303}, [%r339+32768];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r280, %r281, %r304, %r305}, [%r339+36864];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r282, %r283, %r306, %r307}, [%r339+40960];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r284, %r285, %r308, %r309}, [%r339+45056];
	.loc	1 179 36                        // sk03_fa_qkv.py:179:36
	mov.b32 	%r186, %r150;
	mov.b32 	%r187, %r150;
	mov.b32 	%r188, %r150;
	mov.b32 	%r189, %r150;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r186, %r187, %r188, %r189 }, { %r162, %r163, %r164, %r165 }, { %r166, %r167 }, { %r186, %r187, %r188, %r189 };
	// end inline asm
	mov.b32 	%r196, %r150;
	mov.b32 	%r197, %r150;
	mov.b32 	%r198, %r150;
	mov.b32 	%r199, %r150;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r196, %r197, %r198, %r199 }, { %r162, %r163, %r164, %r165 }, { %r168, %r169 }, { %r196, %r197, %r198, %r199 };
	// end inline asm
	mov.b32 	%r202, %r150;
	mov.b32 	%r203, %r150;
	mov.b32 	%r204, %r150;
	mov.b32 	%r205, %r150;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r202, %r203, %r204, %r205 }, { %r162, %r163, %r164, %r165 }, { %r170, %r171 }, { %r202, %r203, %r204, %r205 };
	// end inline asm
	mov.b32 	%r208, %r150;
	mov.b32 	%r209, %r150;
	mov.b32 	%r210, %r150;
	mov.b32 	%r211, %r150;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r208, %r209, %r210, %r211 }, { %r162, %r163, %r164, %r165 }, { %r172, %r173 }, { %r208, %r209, %r210, %r211 };
	// end inline asm
	mov.b32 	%r214, %r150;
	mov.b32 	%r215, %r150;
	mov.b32 	%r216, %r150;
	mov.b32 	%r217, %r150;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r214, %r215, %r216, %r217 }, { %r174, %r175, %r176, %r177 }, { %r166, %r167 }, { %r214, %r215, %r216, %r217 };
	// end inline asm
	mov.b32 	%r222, %r150;
	mov.b32 	%r223, %r150;
	mov.b32 	%r224, %r150;
	mov.b32 	%r225, %r150;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r222, %r223, %r224, %r225 }, { %r174, %r175, %r176, %r177 }, { %r168, %r169 }, { %r222, %r223, %r224, %r225 };
	// end inline asm
	mov.b32 	%r226, %r150;
	mov.b32 	%r227, %r150;
	mov.b32 	%r228, %r150;
	mov.b32 	%r229, %r150;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r226, %r227, %r228, %r229 }, { %r174, %r175, %r176, %r177 }, { %r170, %r171 }, { %r226, %r227, %r228, %r229 };
	// end inline asm
	mov.b32 	%r230, %r150;
	mov.b32 	%r231, %r150;
	mov.b32 	%r232, %r150;
	mov.b32 	%r233, %r150;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r230, %r231, %r232, %r233 }, { %r174, %r175, %r176, %r177 }, { %r172, %r173 }, { %r230, %r231, %r232, %r233 };
	// end inline asm
	mov.b32 	%r234, %r150;
	mov.b32 	%r235, %r150;
	mov.b32 	%r236, %r150;
	mov.b32 	%r237, %r150;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r234, %r235, %r236, %r237 }, { %r178, %r179, %r180, %r181 }, { %r166, %r167 }, { %r234, %r235, %r236, %r237 };
	// end inline asm
	mov.b32 	%r242, %r150;
	mov.b32 	%r243, %r150;
	mov.b32 	%r244, %r150;
	mov.b32 	%r245, %r150;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r242, %r243, %r244, %r245 }, { %r178, %r179, %r180, %r181 }, { %r168, %r169 }, { %r242, %r243, %r244, %r245 };
	// end inline asm
	mov.b32 	%r246, %r150;
	mov.b32 	%r247, %r150;
	mov.b32 	%r248, %r150;
	mov.b32 	%r249, %r150;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r246, %r247, %r248, %r249 }, { %r178, %r179, %r180, %r181 }, { %r170, %r171 }, { %r246, %r247, %r248, %r249 };
	// end inline asm
	mov.b32 	%r250, %r150;
	mov.b32 	%r251, %r150;
	mov.b32 	%r252, %r150;
	mov.b32 	%r253, %r150;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r250, %r251, %r252, %r253 }, { %r178, %r179, %r180, %r181 }, { %r172, %r173 }, { %r250, %r251, %r252, %r253 };
	// end inline asm
	mov.b32 	%r254, %r150;
	mov.b32 	%r255, %r150;
	mov.b32 	%r256, %r150;
	mov.b32 	%r257, %r150;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r254, %r255, %r256, %r257 }, { %r182, %r183, %r184, %r185 }, { %r166, %r167 }, { %r254, %r255, %r256, %r257 };
	// end inline asm
	mov.b32 	%r262, %r150;
	mov.b32 	%r263, %r150;
	mov.b32 	%r264, %r150;
	mov.b32 	%r265, %r150;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r262, %r263, %r264, %r265 }, { %r182, %r183, %r184, %r185 }, { %r168, %r169 }, { %r262, %r263, %r264, %r265 };
	// end inline asm
	mov.b32 	%r266, %r150;
	mov.b32 	%r267, %r150;
	mov.b32 	%r268, %r150;
	mov.b32 	%r269, %r150;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r266, %r267, %r268, %r269 }, { %r182, %r183, %r184, %r185 }, { %r170, %r171 }, { %r266, %r267, %r268, %r269 };
	// end inline asm
	mov.b32 	%r273, %r150;
	mov.b32 	%r270, %r150;
	mov.b32 	%r271, %r150;
	mov.b32 	%r272, %r150;
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
	.loc	1 181 39                        // sk03_fa_qkv.py:181:39
	cvt.s64.s32 	%rd96, %r787;
	add.s64 	%rd80, %rd10, %rd96;
	add.s64 	%rd81, %rd11, %rd96;
	add.s64 	%rd82, %rd12, %rd96;
	add.s64 	%rd83, %rd13, %rd96;
	add.s64 	%rd84, %rd14, %rd96;
	add.s64 	%rd85, %rd15, %rd96;
	add.s64 	%rd86, %rd16, %rd96;
	add.s64 	%rd87, %rd17, %rd96;
	.loc	1 181 29                        // sk03_fa_qkv.py:181:29
	// begin inline asm
	mov.u16 %rs1, 0x0;
	ld.global.b8 { %rs1 }, [ %rd80 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs2, 0x0;
	ld.global.b8 { %rs2 }, [ %rd81 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs3, 0x0;
	ld.global.b8 { %rs3 }, [ %rd82 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs4, 0x0;
	ld.global.b8 { %rs4 }, [ %rd83 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs5, 0x0;
	ld.global.b8 { %rs5 }, [ %rd84 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs6, 0x0;
	ld.global.b8 { %rs6 }, [ %rd85 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs7, 0x0;
	ld.global.b8 { %rs7 }, [ %rd86 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs8, 0x0;
	ld.global.b8 { %rs8 }, [ %rd87 + 0 ];
	// end inline asm
	.loc	1 181 21                        // sk03_fa_qkv.py:181:21
	cvt.u32.u16 	%r340, %rs1;
	and.b32 	%r341, %r340, 255;
	cvt.u32.u16 	%r342, %rs2;
	and.b32 	%r343, %r342, 255;
	cvt.u32.u16 	%r344, %rs3;
	and.b32 	%r345, %r344, 255;
	cvt.u32.u16 	%r346, %rs4;
	and.b32 	%r347, %r346, 255;
	cvt.u32.u16 	%r348, %rs5;
	and.b32 	%r349, %r348, 255;
	cvt.u32.u16 	%r350, %rs6;
	and.b32 	%r351, %r350, 255;
	cvt.u32.u16 	%r352, %rs7;
	and.b32 	%r353, %r352, 255;
	cvt.u32.u16 	%r354, %rs8;
	and.b32 	%r355, %r354, 255;
	shl.b32 	%r356, %r209, %r355;
	shl.b32 	%r357, %r211, %r355;
	shl.b32 	%r358, %r231, %r355;
	shl.b32 	%r359, %r233, %r355;
	shl.b32 	%r360, %r251, %r355;
	shl.b32 	%r361, %r253, %r355;
	shl.b32 	%r362, %r271, %r355;
	shl.b32 	%r363, %r273, %r355;
	shl.b32 	%r364, %r208, %r353;
	shl.b32 	%r365, %r210, %r353;
	shl.b32 	%r366, %r230, %r353;
	shl.b32 	%r367, %r232, %r353;
	shl.b32 	%r368, %r250, %r353;
	shl.b32 	%r369, %r252, %r353;
	shl.b32 	%r370, %r270, %r353;
	shl.b32 	%r371, %r272, %r353;
	shl.b32 	%r372, %r203, %r351;
	shl.b32 	%r373, %r205, %r351;
	shl.b32 	%r374, %r227, %r351;
	shl.b32 	%r375, %r229, %r351;
	shl.b32 	%r376, %r247, %r351;
	shl.b32 	%r377, %r249, %r351;
	shl.b32 	%r378, %r267, %r351;
	shl.b32 	%r379, %r269, %r351;
	shl.b32 	%r380, %r202, %r349;
	shl.b32 	%r381, %r204, %r349;
	shl.b32 	%r382, %r226, %r349;
	shl.b32 	%r383, %r228, %r349;
	shl.b32 	%r384, %r246, %r349;
	shl.b32 	%r385, %r248, %r349;
	shl.b32 	%r386, %r266, %r349;
	shl.b32 	%r387, %r268, %r349;
	shl.b32 	%r388, %r197, %r347;
	shl.b32 	%r389, %r199, %r347;
	shl.b32 	%r390, %r223, %r347;
	shl.b32 	%r391, %r225, %r347;
	shl.b32 	%r392, %r243, %r347;
	shl.b32 	%r393, %r245, %r347;
	shl.b32 	%r394, %r263, %r347;
	shl.b32 	%r395, %r265, %r347;
	shl.b32 	%r396, %r196, %r345;
	shl.b32 	%r397, %r198, %r345;
	shl.b32 	%r398, %r222, %r345;
	shl.b32 	%r399, %r224, %r345;
	shl.b32 	%r400, %r242, %r345;
	shl.b32 	%r401, %r244, %r345;
	shl.b32 	%r402, %r262, %r345;
	shl.b32 	%r403, %r264, %r345;
	shl.b32 	%r404, %r187, %r343;
	shl.b32 	%r405, %r189, %r343;
	shl.b32 	%r406, %r215, %r343;
	shl.b32 	%r407, %r217, %r343;
	shl.b32 	%r408, %r235, %r343;
	shl.b32 	%r409, %r237, %r343;
	shl.b32 	%r410, %r255, %r343;
	shl.b32 	%r411, %r257, %r343;
	shl.b32 	%r412, %r186, %r341;
	shl.b32 	%r413, %r188, %r341;
	shl.b32 	%r414, %r214, %r341;
	shl.b32 	%r415, %r216, %r341;
	shl.b32 	%r416, %r234, %r341;
	shl.b32 	%r417, %r236, %r341;
	shl.b32 	%r418, %r254, %r341;
	shl.b32 	%r419, %r256, %r341;
	.loc	1 182 15                        // sk03_fa_qkv.py:182:15
	add.s32 	%r840, %r419, %r840;
	add.s32 	%r838, %r418, %r838;
	add.s32 	%r824, %r417, %r824;
	add.s32 	%r822, %r416, %r822;
	add.s32 	%r808, %r415, %r808;
	add.s32 	%r806, %r414, %r806;
	add.s32 	%r792, %r413, %r792;
	add.s32 	%r790, %r412, %r790;
	add.s32 	%r841, %r411, %r841;
	add.s32 	%r839, %r410, %r839;
	add.s32 	%r825, %r409, %r825;
	add.s32 	%r823, %r408, %r823;
	add.s32 	%r809, %r407, %r809;
	add.s32 	%r807, %r406, %r807;
	add.s32 	%r793, %r405, %r793;
	add.s32 	%r791, %r404, %r791;
	add.s32 	%r844, %r403, %r844;
	add.s32 	%r842, %r402, %r842;
	add.s32 	%r828, %r401, %r828;
	add.s32 	%r826, %r400, %r826;
	add.s32 	%r812, %r399, %r812;
	add.s32 	%r810, %r398, %r810;
	add.s32 	%r796, %r397, %r796;
	add.s32 	%r794, %r396, %r794;
	add.s32 	%r845, %r395, %r845;
	add.s32 	%r843, %r394, %r843;
	add.s32 	%r829, %r393, %r829;
	add.s32 	%r827, %r392, %r827;
	add.s32 	%r813, %r391, %r813;
	add.s32 	%r811, %r390, %r811;
	add.s32 	%r797, %r389, %r797;
	add.s32 	%r795, %r388, %r795;
	add.s32 	%r848, %r387, %r848;
	add.s32 	%r846, %r386, %r846;
	add.s32 	%r832, %r385, %r832;
	add.s32 	%r830, %r384, %r830;
	add.s32 	%r816, %r383, %r816;
	add.s32 	%r814, %r382, %r814;
	add.s32 	%r800, %r381, %r800;
	add.s32 	%r798, %r380, %r798;
	add.s32 	%r849, %r379, %r849;
	add.s32 	%r847, %r378, %r847;
	add.s32 	%r833, %r377, %r833;
	add.s32 	%r831, %r376, %r831;
	add.s32 	%r817, %r375, %r817;
	add.s32 	%r815, %r374, %r815;
	add.s32 	%r801, %r373, %r801;
	add.s32 	%r799, %r372, %r799;
	add.s32 	%r852, %r371, %r852;
	add.s32 	%r850, %r370, %r850;
	add.s32 	%r836, %r369, %r836;
	add.s32 	%r834, %r368, %r834;
	add.s32 	%r820, %r367, %r820;
	add.s32 	%r818, %r366, %r818;
	add.s32 	%r804, %r365, %r804;
	add.s32 	%r802, %r364, %r802;
	add.s32 	%r853, %r363, %r853;
	add.s32 	%r851, %r362, %r851;
	add.s32 	%r837, %r361, %r837;
	add.s32 	%r835, %r360, %r835;
	add.s32 	%r821, %r359, %r821;
	add.s32 	%r819, %r358, %r819;
	add.s32 	%r805, %r357, %r805;
	add.s32 	%r803, %r356, %r803;
	.loc	1 183 18                        // sk03_fa_qkv.py:183:18
	add.s64 	%rd88, %rd27, %rd210;
	add.s64 	%rd89, %rd26, %rd210;
	add.s64 	%rd90, %rd25, %rd210;
	.loc	1 184 18                        // sk03_fa_qkv.py:184:18
	add.s64 	%rd91, %rd24, %rd210;
	add.s64 	%rd92, %rd23, %rd210;
	add.s64 	%rd93, %rd22, %rd210;
	add.s64 	%rd94, %rd21, %rd210;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	add.s64 	%rd95, %rd20, %rd210;
	add.s32 	%r420, %r789, 1;
	setp.gt.s32 	%p5, %r420, 1;
	selp.b32 	%r789, 0, %r420, %p5;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
	shl.b32 	%r421, %r789, 14;
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
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
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
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	add.s64 	%rd211, %rd211, 1;
	add.s64 	%rd210, %rd210, 128;
	add.s32 	%r787, %r787, %r31;
	setp.ne.b64 	%p6, %rd19, %rd210;
	@%p6 bra 	$L__BB0_3;
// %bb.4:                               // %._crit_edge.loopexit
	.loc	1 186 17                        // sk03_fa_qkv.py:186:17
	cvt.rn.f32.s32 	%r422, %r790;
	cvt.rn.f32.s32 	%r423, %r791;
	cvt.rn.bf16x2.f32 	%r856, %r423, %r422;
	cvt.rn.f32.s32 	%r424, %r792;
	cvt.rn.f32.s32 	%r425, %r793;
	cvt.rn.bf16x2.f32 	%r857, %r425, %r424;
	cvt.rn.f32.s32 	%r426, %r794;
	cvt.rn.f32.s32 	%r427, %r795;
	cvt.rn.bf16x2.f32 	%r858, %r427, %r426;
	cvt.rn.f32.s32 	%r428, %r796;
	cvt.rn.f32.s32 	%r429, %r797;
	cvt.rn.bf16x2.f32 	%r859, %r429, %r428;
	cvt.rn.f32.s32 	%r430, %r798;
	cvt.rn.f32.s32 	%r431, %r799;
	cvt.rn.bf16x2.f32 	%r860, %r431, %r430;
	cvt.rn.f32.s32 	%r432, %r800;
	cvt.rn.f32.s32 	%r433, %r801;
	cvt.rn.bf16x2.f32 	%r861, %r433, %r432;
	cvt.rn.f32.s32 	%r434, %r802;
	cvt.rn.f32.s32 	%r435, %r803;
	cvt.rn.bf16x2.f32 	%r862, %r435, %r434;
	cvt.rn.f32.s32 	%r436, %r804;
	cvt.rn.f32.s32 	%r437, %r805;
	cvt.rn.bf16x2.f32 	%r863, %r437, %r436;
	cvt.rn.f32.s32 	%r438, %r806;
	cvt.rn.f32.s32 	%r439, %r807;
	cvt.rn.bf16x2.f32 	%r864, %r439, %r438;
	cvt.rn.f32.s32 	%r440, %r808;
	cvt.rn.f32.s32 	%r441, %r809;
	cvt.rn.bf16x2.f32 	%r865, %r441, %r440;
	cvt.rn.f32.s32 	%r442, %r810;
	cvt.rn.f32.s32 	%r443, %r811;
	cvt.rn.bf16x2.f32 	%r866, %r443, %r442;
	cvt.rn.f32.s32 	%r444, %r812;
	cvt.rn.f32.s32 	%r445, %r813;
	cvt.rn.bf16x2.f32 	%r867, %r445, %r444;
	cvt.rn.f32.s32 	%r446, %r814;
	cvt.rn.f32.s32 	%r447, %r815;
	cvt.rn.bf16x2.f32 	%r868, %r447, %r446;
	cvt.rn.f32.s32 	%r448, %r816;
	cvt.rn.f32.s32 	%r449, %r817;
	cvt.rn.bf16x2.f32 	%r869, %r449, %r448;
	cvt.rn.f32.s32 	%r450, %r818;
	cvt.rn.f32.s32 	%r451, %r819;
	cvt.rn.bf16x2.f32 	%r870, %r451, %r450;
	cvt.rn.f32.s32 	%r452, %r820;
	cvt.rn.f32.s32 	%r453, %r821;
	cvt.rn.bf16x2.f32 	%r871, %r453, %r452;
	cvt.rn.f32.s32 	%r454, %r822;
	cvt.rn.f32.s32 	%r455, %r823;
	cvt.rn.bf16x2.f32 	%r872, %r455, %r454;
	cvt.rn.f32.s32 	%r456, %r824;
	cvt.rn.f32.s32 	%r457, %r825;
	cvt.rn.bf16x2.f32 	%r873, %r457, %r456;
	cvt.rn.f32.s32 	%r458, %r826;
	cvt.rn.f32.s32 	%r459, %r827;
	cvt.rn.bf16x2.f32 	%r874, %r459, %r458;
	cvt.rn.f32.s32 	%r460, %r828;
	cvt.rn.f32.s32 	%r461, %r829;
	cvt.rn.bf16x2.f32 	%r875, %r461, %r460;
	cvt.rn.f32.s32 	%r462, %r830;
	cvt.rn.f32.s32 	%r463, %r831;
	cvt.rn.bf16x2.f32 	%r876, %r463, %r462;
	cvt.rn.f32.s32 	%r464, %r832;
	cvt.rn.f32.s32 	%r465, %r833;
	cvt.rn.bf16x2.f32 	%r877, %r465, %r464;
	cvt.rn.f32.s32 	%r466, %r834;
	cvt.rn.f32.s32 	%r467, %r835;
	cvt.rn.bf16x2.f32 	%r878, %r467, %r466;
	cvt.rn.f32.s32 	%r468, %r836;
	cvt.rn.f32.s32 	%r469, %r837;
	cvt.rn.bf16x2.f32 	%r879, %r469, %r468;
	cvt.rn.f32.s32 	%r470, %r838;
	cvt.rn.f32.s32 	%r471, %r839;
	cvt.rn.bf16x2.f32 	%r880, %r471, %r470;
	cvt.rn.f32.s32 	%r472, %r840;
	cvt.rn.f32.s32 	%r473, %r841;
	cvt.rn.bf16x2.f32 	%r881, %r473, %r472;
	cvt.rn.f32.s32 	%r474, %r842;
	cvt.rn.f32.s32 	%r475, %r843;
	cvt.rn.bf16x2.f32 	%r882, %r475, %r474;
	cvt.rn.f32.s32 	%r476, %r844;
	cvt.rn.f32.s32 	%r477, %r845;
	cvt.rn.bf16x2.f32 	%r883, %r477, %r476;
	cvt.rn.f32.s32 	%r478, %r846;
	cvt.rn.f32.s32 	%r479, %r847;
	cvt.rn.bf16x2.f32 	%r884, %r479, %r478;
	cvt.rn.f32.s32 	%r480, %r848;
	cvt.rn.f32.s32 	%r481, %r849;
	cvt.rn.bf16x2.f32 	%r885, %r481, %r480;
	cvt.rn.f32.s32 	%r482, %r850;
	cvt.rn.f32.s32 	%r483, %r851;
	cvt.rn.bf16x2.f32 	%r886, %r483, %r482;
	cvt.rn.f32.s32 	%r484, %r852;
	cvt.rn.f32.s32 	%r485, %r853;
	cvt.rn.bf16x2.f32 	%r887, %r485, %r484;
	bra.uni 	$L__BB0_5;
$L__BB0_1:                              // %.._crit_edge_crit_edge
	.loc	1 188 19                        // sk03_fa_qkv.py:188:19
	shl.b32 	%r855, %r2, 1;
	.loc	1 194 8                         // sk03_fa_qkv.py:194:8
	and.b32 	%r854, %r2, 16;
	mov.b32 	%r856, 0;
	mov.b32 	%r857, %r856;
	mov.b32 	%r858, %r856;
	mov.b32 	%r859, %r856;
	mov.b32 	%r860, %r856;
	mov.b32 	%r861, %r856;
	mov.b32 	%r862, %r856;
	mov.b32 	%r863, %r856;
	mov.b32 	%r864, %r856;
	mov.b32 	%r865, %r856;
	mov.b32 	%r866, %r856;
	mov.b32 	%r867, %r856;
	mov.b32 	%r868, %r856;
	mov.b32 	%r869, %r856;
	mov.b32 	%r870, %r856;
	mov.b32 	%r871, %r856;
	mov.b32 	%r872, %r856;
	mov.b32 	%r873, %r856;
	mov.b32 	%r874, %r856;
	mov.b32 	%r875, %r856;
	mov.b32 	%r876, %r856;
	mov.b32 	%r877, %r856;
	mov.b32 	%r878, %r856;
	mov.b32 	%r879, %r856;
	mov.b32 	%r880, %r856;
	mov.b32 	%r881, %r856;
	mov.b32 	%r882, %r856;
	mov.b32 	%r883, %r856;
	mov.b32 	%r884, %r856;
	mov.b32 	%r885, %r856;
	mov.b32 	%r886, %r856;
	mov.b32 	%r887, %r856;
$L__BB0_5:                              // %._crit_edge
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	or.b32 	%r608, %r12, %r786;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r609, %r608, 15;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r610, %r609, %r26;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r611, %r608, 14;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r612, %r611, %r26;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r613, %r608, 13;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r614, %r613, %r26;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r615, %r608, 12;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r616, %r615, %r26;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r617, %r608, 11;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r618, %r617, %r26;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r619, %r608, 10;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r620, %r619, %r26;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r621, %r608, 9;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r622, %r621, %r26;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r623, %r608, 8;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r624, %r623, %r26;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r625, %r608, 7;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r626, %r625, %r26;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r627, %r608, 6;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r628, %r627, %r26;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r629, %r608, 5;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r630, %r629, %r26;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r631, %r608, 4;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r632, %r631, %r26;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r633, %r608, 3;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r634, %r633, %r26;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r635, %r608, 2;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r636, %r635, %r26;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r637, %r608, 1;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r638, %r637, %r26;
	rem.s32 	%r639, %r608, %r26;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	shl.b32 	%r640, %r7, 3;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r641, %r12, %r640;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	and.b32 	%r642, %r2, 128;
	shr.u32 	%r643, %r642, 3;
	shr.u32 	%r644, %r2, 2;
	bfe.u32 	%r645, %r2, 2, 3;
	or.b32 	%r646, %r643, %r645;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r647, %r646, %r1;
	or.b32 	%r648, %r647, 104;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r649, %r648, %r25;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r650, %r647, 96;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r651, %r650, %r25;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r652, %r647, 72;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r653, %r652, %r25;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r654, %r647, 64;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r655, %r654, %r25;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r656, %r647, 40;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r657, %r656, %r25;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r658, %r647, 32;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r659, %r658, %r25;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r660, %r647, 8;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r661, %r660, %r25;
	rem.s32 	%r662, %r647, %r25;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	shr.u32 	%r663, %r2, 4;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r664, %r663, %r1;
	or.b32 	%r665, %r664, 112;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	bfe.u32 	%r666, %r2, 4, 4;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r667, %r666, %r1;
	or.b32 	%r668, %r667, 96;
	or.b32 	%r669, %r667, 80;
	or.b32 	%r670, %r667, 64;
	or.b32 	%r671, %r664, 48;
	or.b32 	%r672, %r667, 32;
	or.b32 	%r673, %r667, 16;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 186 54                        // sk03_fa_qkv.py:186:54
	mad.wide.s32 	%rd97, %r662, 4, %rd32;
	mad.wide.s32 	%rd98, %r661, 4, %rd32;
	mad.wide.s32 	%rd99, %r659, 4, %rd32;
	mad.wide.s32 	%rd100, %r657, 4, %rd32;
	mad.wide.s32 	%rd101, %r655, 4, %rd32;
	mad.wide.s32 	%rd102, %r653, 4, %rd32;
	mad.wide.s32 	%rd103, %r651, 4, %rd32;
	mad.wide.s32 	%rd104, %r649, 4, %rd32;
	.loc	1 186 40                        // sk03_fa_qkv.py:186:40
	// begin inline asm
	mov.u32 %r486, 0x0;
	ld.global.b32 { %r486 }, [ %rd97 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r487, 0x0;
	ld.global.b32 { %r487 }, [ %rd98 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r488, 0x0;
	ld.global.b32 { %r488 }, [ %rd99 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r489, 0x0;
	ld.global.b32 { %r489 }, [ %rd100 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r490, 0x0;
	ld.global.b32 { %r490 }, [ %rd101 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r491, 0x0;
	ld.global.b32 { %r491 }, [ %rd102 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r492, 0x0;
	ld.global.b32 { %r492 }, [ %rd103 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r493, 0x0;
	ld.global.b32 { %r493 }, [ %rd104 + 0 ];
	// end inline asm
	.loc	1 186 65                        // sk03_fa_qkv.py:186:65
	cvt.rn.bf16.f32 	%rs73, %r486;
	cvt.rn.bf16.f32 	%rs74, %r487;
	cvt.rn.bf16.f32 	%rs75, %r488;
	cvt.rn.bf16.f32 	%rs76, %r489;
	cvt.rn.bf16.f32 	%rs77, %r490;
	cvt.rn.bf16.f32 	%rs78, %r491;
	cvt.rn.bf16.f32 	%rs79, %r492;
	cvt.rn.bf16.f32 	%rs80, %r493;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	mov.b32 	{%rs81, %rs82}, %r856;
	mov.b16 	%rs83, 0x8000;
	fma.rn.bf16 	%rs84, %rs81, %rs73, %rs83;
	fma.rn.bf16 	%rs85, %rs82, %rs73, %rs83;
	mov.b32 	{%rs86, %rs87}, %r857;
	fma.rn.bf16 	%rs88, %rs86, %rs74, %rs83;
	fma.rn.bf16 	%rs89, %rs87, %rs74, %rs83;
	mov.b32 	{%rs90, %rs91}, %r858;
	fma.rn.bf16 	%rs92, %rs90, %rs73, %rs83;
	fma.rn.bf16 	%rs93, %rs91, %rs73, %rs83;
	mov.b32 	{%rs94, %rs95}, %r859;
	fma.rn.bf16 	%rs96, %rs94, %rs74, %rs83;
	fma.rn.bf16 	%rs97, %rs95, %rs74, %rs83;
	mov.b32 	{%rs98, %rs99}, %r860;
	fma.rn.bf16 	%rs100, %rs98, %rs73, %rs83;
	fma.rn.bf16 	%rs101, %rs99, %rs73, %rs83;
	mov.b32 	{%rs102, %rs103}, %r861;
	fma.rn.bf16 	%rs104, %rs102, %rs74, %rs83;
	fma.rn.bf16 	%rs105, %rs103, %rs74, %rs83;
	mov.b32 	{%rs106, %rs107}, %r862;
	fma.rn.bf16 	%rs108, %rs106, %rs73, %rs83;
	fma.rn.bf16 	%rs109, %rs107, %rs73, %rs83;
	mov.b32 	{%rs110, %rs111}, %r863;
	fma.rn.bf16 	%rs112, %rs110, %rs74, %rs83;
	fma.rn.bf16 	%rs113, %rs111, %rs74, %rs83;
	mov.b32 	{%rs114, %rs115}, %r864;
	fma.rn.bf16 	%rs116, %rs114, %rs75, %rs83;
	fma.rn.bf16 	%rs117, %rs115, %rs75, %rs83;
	mov.b32 	{%rs118, %rs119}, %r865;
	fma.rn.bf16 	%rs120, %rs118, %rs76, %rs83;
	fma.rn.bf16 	%rs121, %rs119, %rs76, %rs83;
	mov.b32 	{%rs122, %rs123}, %r866;
	fma.rn.bf16 	%rs124, %rs122, %rs75, %rs83;
	fma.rn.bf16 	%rs125, %rs123, %rs75, %rs83;
	mov.b32 	{%rs126, %rs127}, %r867;
	fma.rn.bf16 	%rs128, %rs126, %rs76, %rs83;
	fma.rn.bf16 	%rs129, %rs127, %rs76, %rs83;
	mov.b32 	{%rs130, %rs131}, %r868;
	fma.rn.bf16 	%rs132, %rs130, %rs75, %rs83;
	fma.rn.bf16 	%rs133, %rs131, %rs75, %rs83;
	mov.b32 	{%rs134, %rs135}, %r869;
	fma.rn.bf16 	%rs136, %rs134, %rs76, %rs83;
	fma.rn.bf16 	%rs137, %rs135, %rs76, %rs83;
	mov.b32 	{%rs138, %rs139}, %r870;
	fma.rn.bf16 	%rs140, %rs138, %rs75, %rs83;
	fma.rn.bf16 	%rs141, %rs139, %rs75, %rs83;
	mov.b32 	{%rs142, %rs143}, %r871;
	fma.rn.bf16 	%rs144, %rs142, %rs76, %rs83;
	fma.rn.bf16 	%rs145, %rs143, %rs76, %rs83;
	mov.b32 	{%rs146, %rs147}, %r872;
	fma.rn.bf16 	%rs148, %rs146, %rs77, %rs83;
	fma.rn.bf16 	%rs149, %rs147, %rs77, %rs83;
	mov.b32 	{%rs150, %rs151}, %r873;
	fma.rn.bf16 	%rs152, %rs150, %rs78, %rs83;
	fma.rn.bf16 	%rs153, %rs151, %rs78, %rs83;
	mov.b32 	{%rs154, %rs155}, %r874;
	fma.rn.bf16 	%rs156, %rs154, %rs77, %rs83;
	fma.rn.bf16 	%rs157, %rs155, %rs77, %rs83;
	mov.b32 	{%rs158, %rs159}, %r875;
	fma.rn.bf16 	%rs160, %rs158, %rs78, %rs83;
	fma.rn.bf16 	%rs161, %rs159, %rs78, %rs83;
	mov.b32 	{%rs162, %rs163}, %r876;
	fma.rn.bf16 	%rs164, %rs162, %rs77, %rs83;
	fma.rn.bf16 	%rs165, %rs163, %rs77, %rs83;
	mov.b32 	{%rs166, %rs167}, %r877;
	fma.rn.bf16 	%rs168, %rs166, %rs78, %rs83;
	fma.rn.bf16 	%rs169, %rs167, %rs78, %rs83;
	mov.b32 	{%rs170, %rs171}, %r878;
	fma.rn.bf16 	%rs172, %rs170, %rs77, %rs83;
	fma.rn.bf16 	%rs173, %rs171, %rs77, %rs83;
	mov.b32 	{%rs174, %rs175}, %r879;
	fma.rn.bf16 	%rs176, %rs174, %rs78, %rs83;
	fma.rn.bf16 	%rs177, %rs175, %rs78, %rs83;
	mov.b32 	{%rs178, %rs179}, %r881;
	fma.rn.bf16 	%rs180, %rs178, %rs80, %rs83;
	fma.rn.bf16 	%rs181, %rs179, %rs80, %rs83;
	mov.b32 	{%rs182, %rs183}, %r883;
	fma.rn.bf16 	%rs184, %rs182, %rs80, %rs83;
	fma.rn.bf16 	%rs185, %rs183, %rs80, %rs83;
	mov.b32 	{%rs186, %rs187}, %r885;
	fma.rn.bf16 	%rs188, %rs186, %rs80, %rs83;
	fma.rn.bf16 	%rs189, %rs187, %rs80, %rs83;
	mov.b32 	{%rs190, %rs191}, %r887;
	fma.rn.bf16 	%rs192, %rs190, %rs80, %rs83;
	fma.rn.bf16 	%rs193, %rs191, %rs80, %rs83;
	.loc	1 187 38                        // sk03_fa_qkv.py:187:38
	mad.wide.s32 	%rd105, %r13, 4, %rd33;
	mad.wide.s32 	%rd106, %r14, 4, %rd33;
	mad.wide.s32 	%rd107, %r15, 4, %rd33;
	mad.wide.s32 	%rd108, %r16, 4, %rd33;
	.loc	1 187 24                        // sk03_fa_qkv.py:187:24
	// begin inline asm
	mov.u32 %r494, 0x0;
	mov.u32 %r495, 0x0;
	ld.global.v2.b32 { %r494, %r495 }, [ %rd105 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r496, 0x0;
	mov.u32 %r497, 0x0;
	ld.global.v2.b32 { %r496, %r497 }, [ %rd106 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r498, 0x0;
	mov.u32 %r499, 0x0;
	ld.global.v2.b32 { %r498, %r499 }, [ %rd107 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r500, 0x0;
	mov.u32 %r501, 0x0;
	ld.global.v2.b32 { %r500, %r501 }, [ %rd108 + 0 ];
	// end inline asm
	.loc	1 188 49                        // sk03_fa_qkv.py:188:49
	mul.lo.s32 	%r674, %r8, %r29;
	mul.lo.s32 	%r675, %r9, %r29;
	mul.lo.s32 	%r676, %r10, %r29;
	mul.lo.s32 	%r677, %r11, %r29;
	.loc	1 188 31                        // sk03_fa_qkv.py:188:31
	mad.wide.s32 	%rd181, %r674, 2, %rd31;
	mad.wide.s32 	%rd182, %r675, 2, %rd31;
	mad.wide.s32 	%rd183, %r676, 2, %rd31;
	mad.wide.s32 	%rd184, %r677, 2, %rd31;
	.loc	1 188 82                        // sk03_fa_qkv.py:188:82
	mul.lo.s32 	%r678, %r639, %r30;
	mul.lo.s32 	%r679, %r638, %r30;
	mul.lo.s32 	%r680, %r636, %r30;
	mul.lo.s32 	%r681, %r634, %r30;
	mul.lo.s32 	%r682, %r632, %r30;
	mul.lo.s32 	%r683, %r630, %r30;
	mul.lo.s32 	%r684, %r628, %r30;
	mul.lo.s32 	%r685, %r626, %r30;
	mul.lo.s32 	%r686, %r624, %r30;
	mul.lo.s32 	%r687, %r622, %r30;
	mul.lo.s32 	%r688, %r620, %r30;
	mul.lo.s32 	%r689, %r618, %r30;
	mul.lo.s32 	%r690, %r616, %r30;
	mul.lo.s32 	%r691, %r614, %r30;
	mul.lo.s32 	%r692, %r612, %r30;
	mul.lo.s32 	%r693, %r610, %r30;
	.loc	1 188 64                        // sk03_fa_qkv.py:188:64
	mul.wide.s32 	%rd185, %r678, 2;
	add.s64 	%rd109, %rd181, %rd185;
	mul.wide.s32 	%rd186, %r679, 2;
	add.s64 	%rd110, %rd181, %rd186;
	mul.wide.s32 	%rd187, %r680, 2;
	add.s64 	%rd111, %rd181, %rd187;
	mul.wide.s32 	%rd188, %r681, 2;
	add.s64 	%rd112, %rd181, %rd188;
	mul.wide.s32 	%rd189, %r682, 2;
	add.s64 	%rd113, %rd181, %rd189;
	mul.wide.s32 	%rd190, %r683, 2;
	add.s64 	%rd114, %rd181, %rd190;
	mul.wide.s32 	%rd191, %r684, 2;
	add.s64 	%rd115, %rd181, %rd191;
	mul.wide.s32 	%rd192, %r685, 2;
	add.s64 	%rd116, %rd181, %rd192;
	mul.wide.s32 	%rd193, %r686, 2;
	add.s64 	%rd117, %rd181, %rd193;
	mul.wide.s32 	%rd194, %r687, 2;
	add.s64 	%rd118, %rd181, %rd194;
	mul.wide.s32 	%rd195, %r688, 2;
	add.s64 	%rd119, %rd181, %rd195;
	mul.wide.s32 	%rd196, %r689, 2;
	add.s64 	%rd120, %rd181, %rd196;
	mul.wide.s32 	%rd197, %r690, 2;
	add.s64 	%rd121, %rd181, %rd197;
	mul.wide.s32 	%rd198, %r691, 2;
	add.s64 	%rd122, %rd181, %rd198;
	mul.wide.s32 	%rd199, %r692, 2;
	add.s64 	%rd123, %rd181, %rd199;
	mul.wide.s32 	%rd200, %r693, 2;
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
	.loc	1 188 19                        // sk03_fa_qkv.py:188:19
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
	// begin inline asm
	mov.u16 %rs65, 0x0;
	ld.global.b16 { %rs65 }, [ %rd165 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs66, 0x0;
	ld.global.b16 { %rs66 }, [ %rd166 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs67, 0x0;
	ld.global.b16 { %rs67 }, [ %rd167 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs68, 0x0;
	ld.global.b16 { %rs68 }, [ %rd168 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs69, 0x0;
	ld.global.b16 { %rs69 }, [ %rd169 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs70, 0x0;
	ld.global.b16 { %rs70 }, [ %rd170 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs71, 0x0;
	ld.global.b16 { %rs71 }, [ %rd171 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs72, 0x0;
	ld.global.b16 { %rs72 }, [ %rd172 + 0 ];
	// end inline asm
	shl.b32 	%r694, %r18, 7;
	shl.b32 	%r695, %r3, 1;
	or.b32 	%r696, %r694, %r786;
	xor.b32 	%r697, %r696, %r695;
	add.s32 	%r502, %r149, %r697;
	mov.b32 	%r503, {%rs9, %rs10};
	mov.b32 	%r504, {%rs11, %rs12};
	mov.b32 	%r505, {%rs13, %rs14};
	mov.b32 	%r506, {%rs15, %rs16};
	// begin inline asm
	st.shared.v4.b32 [ %r502 + 0 ], { %r503, %r504, %r505, %r506 };
	// end inline asm
	add.s32 	%r507, %r502, 512;
	mov.b32 	%r508, {%rs17, %rs18};
	mov.b32 	%r509, {%rs19, %rs20};
	mov.b32 	%r510, {%rs21, %rs22};
	mov.b32 	%r511, {%rs23, %rs24};
	// begin inline asm
	st.shared.v4.b32 [ %r507 + 0 ], { %r508, %r509, %r510, %r511 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r698, %r6, 10;
	and.b32 	%r699, %r17, 752;
	and.b32 	%r700, %r855, 288;
	and.b32 	%r701, %r644, 16;
	xor.b32 	%r702, %r699, %r700;
	xor.b32 	%r703, %r702, %r701;
	or.b32 	%r704, %r703, %r698;
	add.s32 	%r705, %r149, %r704;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r706, %r707, %r708, %r709}, [%r705];
	mov.b32 	{%rs194, %rs195}, %r706;
	mov.b32 	{%rs196, %rs197}, %r707;
	mov.b32 	{%rs198, %rs199}, %r708;
	mov.b32 	{%rs200, %rs201}, %r709;
	xor.b32 	%r710, %r704, 64;
	add.s32 	%r711, %r149, %r710;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r712, %r713, %r714, %r715}, [%r711];
	mov.b32 	{%rs202, %rs203}, %r712;
	mov.b32 	{%rs204, %rs205}, %r713;
	mov.b32 	{%rs206, %rs207}, %r714;
	mov.b32 	{%rs208, %rs209}, %r715;
	bar.sync 	0;
	mov.b32 	%r512, {%rs25, %rs26};
	mov.b32 	%r513, {%rs27, %rs28};
	mov.b32 	%r514, {%rs29, %rs30};
	mov.b32 	%r515, {%rs31, %rs32};
	// begin inline asm
	st.shared.v4.b32 [ %r502 + 0 ], { %r512, %r513, %r514, %r515 };
	// end inline asm
	mov.b32 	%r516, {%rs33, %rs34};
	mov.b32 	%r517, {%rs35, %rs36};
	mov.b32 	%r518, {%rs37, %rs38};
	mov.b32 	%r519, {%rs39, %rs40};
	// begin inline asm
	st.shared.v4.b32 [ %r507 + 0 ], { %r516, %r517, %r518, %r519 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r716, %r717, %r718, %r719}, [%r705];
	mov.b32 	{%rs210, %rs211}, %r716;
	mov.b32 	{%rs212, %rs213}, %r717;
	mov.b32 	{%rs214, %rs215}, %r718;
	mov.b32 	{%rs216, %rs217}, %r719;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r720, %r721, %r722, %r723}, [%r711];
	mov.b32 	{%rs218, %rs219}, %r720;
	mov.b32 	{%rs220, %rs221}, %r721;
	mov.b32 	{%rs222, %rs223}, %r722;
	mov.b32 	{%rs224, %rs225}, %r723;
	bar.sync 	0;
	mov.b32 	%r520, {%rs41, %rs42};
	mov.b32 	%r521, {%rs43, %rs44};
	mov.b32 	%r522, {%rs45, %rs46};
	mov.b32 	%r523, {%rs47, %rs48};
	// begin inline asm
	st.shared.v4.b32 [ %r502 + 0 ], { %r520, %r521, %r522, %r523 };
	// end inline asm
	mov.b32 	%r524, {%rs49, %rs50};
	mov.b32 	%r525, {%rs51, %rs52};
	mov.b32 	%r526, {%rs53, %rs54};
	mov.b32 	%r527, {%rs55, %rs56};
	// begin inline asm
	st.shared.v4.b32 [ %r507 + 0 ], { %r524, %r525, %r526, %r527 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r724, %r725, %r726, %r727}, [%r705];
	mov.b32 	{%rs226, %rs227}, %r724;
	mov.b32 	{%rs228, %rs229}, %r725;
	mov.b32 	{%rs230, %rs231}, %r726;
	mov.b32 	{%rs232, %rs233}, %r727;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r728, %r729, %r730, %r731}, [%r711];
	mov.b32 	{%rs234, %rs235}, %r728;
	mov.b32 	{%rs236, %rs237}, %r729;
	mov.b32 	{%rs238, %rs239}, %r730;
	mov.b32 	{%rs240, %rs241}, %r731;
	bar.sync 	0;
	mov.b32 	%r528, {%rs57, %rs58};
	mov.b32 	%r529, {%rs59, %rs60};
	mov.b32 	%r530, {%rs61, %rs62};
	mov.b32 	%r531, {%rs63, %rs64};
	// begin inline asm
	st.shared.v4.b32 [ %r502 + 0 ], { %r528, %r529, %r530, %r531 };
	// end inline asm
	mov.b32 	%r532, {%rs65, %rs66};
	mov.b32 	%r533, {%rs67, %rs68};
	mov.b32 	%r534, {%rs69, %rs70};
	mov.b32 	%r535, {%rs71, %rs72};
	// begin inline asm
	st.shared.v4.b32 [ %r507 + 0 ], { %r532, %r533, %r534, %r535 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r732, %r733, %r734, %r735}, [%r705];
	mov.b32 	{%rs242, %rs243}, %r733;
	mov.b32 	{%rs244, %rs245}, %r735;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r736, %r737, %r738, %r739}, [%r711];
	mov.b32 	{%rs246, %rs247}, %r737;
	mov.b32 	{%rs248, %rs249}, %r739;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	mov.b32 	%r740, {%rs79, %rs79};
	mov.b32 	%r741, -2147450880;
	fma.rn.bf16x2 	%r742, %r880, %r740, %r741;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs250, %r495;
	cvt.rn.bf16.f32 	%rs251, %r494;
	mov.b32 	%r743, {%rs251, %rs250};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs252, %rs84, %rs251, %rs194;
	fma.rn.bf16 	%rs253, %rs85, %rs250, %rs195;
	fma.rn.bf16 	%rs254, %rs88, %rs251, %rs196;
	fma.rn.bf16 	%rs255, %rs89, %rs250, %rs197;
	fma.rn.bf16 	%rs256, %rs116, %rs251, %rs210;
	fma.rn.bf16 	%rs257, %rs117, %rs250, %rs211;
	fma.rn.bf16 	%rs258, %rs120, %rs251, %rs212;
	fma.rn.bf16 	%rs259, %rs121, %rs250, %rs213;
	fma.rn.bf16 	%rs260, %rs148, %rs251, %rs226;
	fma.rn.bf16 	%rs261, %rs149, %rs250, %rs227;
	fma.rn.bf16 	%rs262, %rs152, %rs251, %rs228;
	fma.rn.bf16 	%rs263, %rs153, %rs250, %rs229;
	fma.rn.bf16x2 	%r540, %r742, %r743, %r732;
	fma.rn.bf16 	%rs264, %rs180, %rs251, %rs242;
	fma.rn.bf16 	%rs265, %rs181, %rs250, %rs243;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r744, %r882, %r740, %r741;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs266, %r497;
	cvt.rn.bf16.f32 	%rs267, %r496;
	mov.b32 	%r745, {%rs267, %rs266};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs268, %rs92, %rs267, %rs198;
	fma.rn.bf16 	%rs269, %rs93, %rs266, %rs199;
	fma.rn.bf16 	%rs270, %rs96, %rs267, %rs200;
	fma.rn.bf16 	%rs271, %rs97, %rs266, %rs201;
	fma.rn.bf16 	%rs272, %rs124, %rs267, %rs214;
	fma.rn.bf16 	%rs273, %rs125, %rs266, %rs215;
	fma.rn.bf16 	%rs274, %rs128, %rs267, %rs216;
	fma.rn.bf16 	%rs275, %rs129, %rs266, %rs217;
	fma.rn.bf16 	%rs276, %rs156, %rs267, %rs230;
	fma.rn.bf16 	%rs277, %rs157, %rs266, %rs231;
	fma.rn.bf16 	%rs278, %rs160, %rs267, %rs232;
	fma.rn.bf16 	%rs279, %rs161, %rs266, %rs233;
	fma.rn.bf16x2 	%r560, %r744, %r745, %r734;
	fma.rn.bf16 	%rs280, %rs184, %rs267, %rs244;
	fma.rn.bf16 	%rs281, %rs185, %rs266, %rs245;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r746, %r884, %r740, %r741;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs282, %r499;
	cvt.rn.bf16.f32 	%rs283, %r498;
	mov.b32 	%r747, {%rs283, %rs282};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs284, %rs100, %rs283, %rs202;
	fma.rn.bf16 	%rs285, %rs101, %rs282, %rs203;
	fma.rn.bf16 	%rs286, %rs104, %rs283, %rs204;
	fma.rn.bf16 	%rs287, %rs105, %rs282, %rs205;
	fma.rn.bf16 	%rs288, %rs132, %rs283, %rs218;
	fma.rn.bf16 	%rs289, %rs133, %rs282, %rs219;
	fma.rn.bf16 	%rs290, %rs136, %rs283, %rs220;
	fma.rn.bf16 	%rs291, %rs137, %rs282, %rs221;
	fma.rn.bf16 	%rs292, %rs164, %rs283, %rs234;
	fma.rn.bf16 	%rs293, %rs165, %rs282, %rs235;
	fma.rn.bf16 	%rs294, %rs168, %rs283, %rs236;
	fma.rn.bf16 	%rs295, %rs169, %rs282, %rs237;
	fma.rn.bf16x2 	%r550, %r746, %r747, %r736;
	fma.rn.bf16 	%rs296, %rs188, %rs283, %rs246;
	fma.rn.bf16 	%rs297, %rs189, %rs282, %rs247;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r748, %r886, %r740, %r741;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs298, %r501;
	cvt.rn.bf16.f32 	%rs299, %r500;
	mov.b32 	%r749, {%rs299, %rs298};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs300, %rs108, %rs299, %rs206;
	fma.rn.bf16 	%rs301, %rs109, %rs298, %rs207;
	fma.rn.bf16 	%rs302, %rs112, %rs299, %rs208;
	fma.rn.bf16 	%rs303, %rs113, %rs298, %rs209;
	fma.rn.bf16 	%rs304, %rs140, %rs299, %rs222;
	fma.rn.bf16 	%rs305, %rs141, %rs298, %rs223;
	fma.rn.bf16 	%rs306, %rs144, %rs299, %rs224;
	fma.rn.bf16 	%rs307, %rs145, %rs298, %rs225;
	fma.rn.bf16 	%rs308, %rs172, %rs299, %rs238;
	fma.rn.bf16 	%rs309, %rs173, %rs298, %rs239;
	fma.rn.bf16 	%rs310, %rs176, %rs299, %rs240;
	fma.rn.bf16 	%rs311, %rs177, %rs298, %rs241;
	fma.rn.bf16x2 	%r570, %r748, %r749, %r738;
	fma.rn.bf16 	%rs312, %rs192, %rs299, %rs248;
	fma.rn.bf16 	%rs313, %rs193, %rs298, %rs249;
	.loc	1 195 31                        // sk03_fa_qkv.py:195:31
	setp.lt.s32 	%p15, %r667, %r25;
	setp.lt.s32 	%p16, %r673, %r25;
	setp.lt.s32 	%p17, %r672, %r25;
	setp.lt.s32 	%p18, %r671, %r25;
	setp.lt.s32 	%p19, %r670, %r25;
	setp.lt.s32 	%p20, %r669, %r25;
	setp.lt.s32 	%p21, %r668, %r25;
	setp.lt.s32 	%p22, %r665, %r25;
	.loc	1 195 54                        // sk03_fa_qkv.py:195:54
	setp.lt.s32 	%p23, %r641, %r26;
	.loc	1 195 37                        // sk03_fa_qkv.py:195:37
	and.pred 	%p7, %p15, %p23;
	and.pred 	%p8, %p16, %p23;
	and.pred 	%p9, %p17, %p23;
	and.pred 	%p10, %p18, %p23;
	and.pred 	%p11, %p19, %p23;
	and.pred 	%p12, %p20, %p23;
	and.pred 	%p13, %p21, %p23;
	and.pred 	%p14, %p22, %p23;
	.loc	1 193 35                        // sk03_fa_qkv.py:193:35
	mul.lo.s32 	%r750, %r667, %r28;
	mul.lo.s32 	%r751, %r673, %r28;
	mul.lo.s32 	%r752, %r672, %r28;
	mul.lo.s32 	%r753, %r671, %r28;
	mul.lo.s32 	%r754, %r670, %r28;
	mul.lo.s32 	%r755, %r669, %r28;
	mul.lo.s32 	%r756, %r668, %r28;
	mul.lo.s32 	%r757, %r665, %r28;
	.loc	1 193 18                        // sk03_fa_qkv.py:193:18
	mad.wide.s32 	%rd201, %r750, 2, %rd30;
	mad.wide.s32 	%rd202, %r751, 2, %rd30;
	mad.wide.s32 	%rd203, %r752, 2, %rd30;
	mad.wide.s32 	%rd204, %r753, 2, %rd30;
	mad.wide.s32 	%rd205, %r754, 2, %rd30;
	mad.wide.s32 	%rd206, %r755, 2, %rd30;
	mad.wide.s32 	%rd207, %r756, 2, %rd30;
	mad.wide.s32 	%rd208, %r757, 2, %rd30;
	.loc	1 193 50                        // sk03_fa_qkv.py:193:50
	mul.wide.s32 	%rd209, %r641, 2;
	add.s64 	%rd173, %rd201, %rd209;
	add.s64 	%rd174, %rd202, %rd209;
	add.s64 	%rd175, %rd203, %rd209;
	add.s64 	%rd176, %rd204, %rd209;
	add.s64 	%rd177, %rd205, %rd209;
	add.s64 	%rd178, %rd206, %rd209;
	add.s64 	%rd179, %rd207, %rd209;
	add.s64 	%rd180, %rd208, %rd209;
	.loc	1 194 8                         // sk03_fa_qkv.py:194:8
	bar.sync 	0;
	shl.b32 	%r758, %r4, 13;
	shl.b32 	%r759, %r4, 5;
	and.b32 	%r760, %r17, 384;
	shr.u32 	%r761, %r5, 1;
	bfe.s32 	%r762, %r2, 2, 1;
	and.b32 	%r763, %r762, 4112;
	shl.b32 	%r764, %r642, 3;
	or.b32 	%r765, %r758, %r764;
	or.b32 	%r766, %r759, %r760;
	xor.b32 	%r767, %r763, %r761;
	xor.b32 	%r768, %r767, %r766;
	or.b32 	%r769, %r768, %r765;
	add.s32 	%r536, %r149, %r769;
	mov.b32 	%r537, {%rs252, %rs253};
	mov.b32 	%r538, {%rs256, %rs257};
	mov.b32 	%r539, {%rs260, %rs261};
	// begin inline asm
	st.shared.v4.b32 [ %r536 + 0 ], { %r537, %r538, %r539, %r540 };
	// end inline asm
	add.s32 	%r541, %r536, 512;
	mov.b32 	%r542, {%rs254, %rs255};
	mov.b32 	%r543, {%rs258, %rs259};
	mov.b32 	%r544, {%rs262, %rs263};
	mov.b32 	%r545, {%rs264, %rs265};
	// begin inline asm
	st.shared.v4.b32 [ %r541 + 0 ], { %r542, %r543, %r544, %r545 };
	// end inline asm
	add.s32 	%r546, %r536, 2048;
	mov.b32 	%r547, {%rs284, %rs285};
	mov.b32 	%r548, {%rs288, %rs289};
	mov.b32 	%r549, {%rs292, %rs293};
	// begin inline asm
	st.shared.v4.b32 [ %r546 + 0 ], { %r547, %r548, %r549, %r550 };
	// end inline asm
	add.s32 	%r551, %r536, 2560;
	mov.b32 	%r552, {%rs286, %rs287};
	mov.b32 	%r553, {%rs290, %rs291};
	mov.b32 	%r554, {%rs294, %rs295};
	mov.b32 	%r555, {%rs296, %rs297};
	// begin inline asm
	st.shared.v4.b32 [ %r551 + 0 ], { %r552, %r553, %r554, %r555 };
	// end inline asm
	xor.b32 	%r770, %r769, 64;
	add.s32 	%r556, %r149, %r770;
	mov.b32 	%r557, {%rs268, %rs269};
	mov.b32 	%r558, {%rs272, %rs273};
	mov.b32 	%r559, {%rs276, %rs277};
	// begin inline asm
	st.shared.v4.b32 [ %r556 + 0 ], { %r557, %r558, %r559, %r560 };
	// end inline asm
	add.s32 	%r561, %r556, 512;
	mov.b32 	%r562, {%rs270, %rs271};
	mov.b32 	%r563, {%rs274, %rs275};
	mov.b32 	%r564, {%rs278, %rs279};
	mov.b32 	%r565, {%rs280, %rs281};
	// begin inline asm
	st.shared.v4.b32 [ %r561 + 0 ], { %r562, %r563, %r564, %r565 };
	// end inline asm
	add.s32 	%r566, %r556, 2048;
	mov.b32 	%r567, {%rs300, %rs301};
	mov.b32 	%r568, {%rs304, %rs305};
	mov.b32 	%r569, {%rs308, %rs309};
	// begin inline asm
	st.shared.v4.b32 [ %r566 + 0 ], { %r567, %r568, %r569, %r570 };
	// end inline asm
	add.s32 	%r571, %r556, 2560;
	mov.b32 	%r572, {%rs302, %rs303};
	mov.b32 	%r573, {%rs306, %rs307};
	mov.b32 	%r574, {%rs310, %rs311};
	mov.b32 	%r575, {%rs312, %rs313};
	// begin inline asm
	st.shared.v4.b32 [ %r571 + 0 ], { %r572, %r573, %r574, %r575 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r771, %r2, 2;
	and.b32 	%r772, %r771, 896;
	shl.b32 	%r773, %r2, 8;
	and.b32 	%r774, %r773, 2048;
	setp.eq.b32 	%p24, %r854, 0;
	selp.b32 	%r775, 0, 4112, %p24;
	or.b32 	%r776, %r786, %r772;
	xor.b32 	%r777, %r776, %r775;
	or.b32 	%r778, %r777, %r774;
	add.s32 	%r779, %r149, %r778;
	ld.shared.v4.b32 	{%r576, %r584, %r592, %r600}, [%r779];
	ld.shared.v4.b32 	{%r580, %r588, %r596, %r604}, [%r779+1024];
	xor.b32 	%r780, %r778, 32;
	add.s32 	%r781, %r149, %r780;
	ld.shared.v4.b32 	{%r577, %r585, %r593, %r601}, [%r781+8192];
	ld.shared.v4.b32 	{%r581, %r589, %r597, %r605}, [%r781+9216];
	xor.b32 	%r782, %r778, 64;
	add.s32 	%r783, %r149, %r782;
	ld.shared.v4.b32 	{%r578, %r586, %r594, %r602}, [%r783+16384];
	ld.shared.v4.b32 	{%r582, %r590, %r598, %r606}, [%r783+17408];
	xor.b32 	%r784, %r778, 96;
	add.s32 	%r785, %r149, %r784;
	ld.shared.v4.b32 	{%r579, %r587, %r595, %r603}, [%r785+24576];
	ld.shared.v4.b32 	{%r583, %r591, %r599, %r607}, [%r785+25600];
	// begin inline asm
	@%p7 st.global.v4.b32 [ %rd173 + 0 ], { %r576, %r577, %r578, %r579 };
	// end inline asm
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd174 + 0 ], { %r580, %r581, %r582, %r583 };
	// end inline asm
	// begin inline asm
	@%p9 st.global.v4.b32 [ %rd175 + 0 ], { %r584, %r585, %r586, %r587 };
	// end inline asm
	// begin inline asm
	@%p10 st.global.v4.b32 [ %rd176 + 0 ], { %r588, %r589, %r590, %r591 };
	// end inline asm
	// begin inline asm
	@%p11 st.global.v4.b32 [ %rd177 + 0 ], { %r592, %r593, %r594, %r595 };
	// end inline asm
	// begin inline asm
	@%p12 st.global.v4.b32 [ %rd178 + 0 ], { %r596, %r597, %r598, %r599 };
	// end inline asm
	// begin inline asm
	@%p13 st.global.v4.b32 [ %rd179 + 0 ], { %r600, %r601, %r602, %r603 };
	// end inline asm
	// begin inline asm
	@%p14 st.global.v4.b32 [ %rd180 + 0 ], { %r604, %r605, %r606, %r607 };
	// end inline asm
	.loc	1 192 4                         // sk03_fa_qkv.py:192:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk03_fa_qkv.py"
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
.b32 157                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x96 DW_TAG_compile_unit
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
.b8 51
.b8 95
.b8 102
.b8 97
.b8 95
.b8 113
.b8 107
.b8 118
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
.b8 2                                   // Abbrev [2] 0x44:0x16 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 51
.b8 95
.b8 102
.b8 97
.b8 95
.b8 113
.b8 107
.b8 118
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x5a:0x46 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 68                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x6f:0x18 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 155                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x87:0x18 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 156                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_7 = _Nativo(
    "sk03_fa_qkv/tile128x128x128_shift1_abi16",
    _PTX_7, "_sk03_fa_qkv_kernel",
    warps=8, shared=65536,
    abi=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 17, 18],
    horneado={11: 1, 12: 1, 15: 1, 19: 1, 20: 128, 21: 128, 22: 128, 23: 8, 24: 128, 25: True},
    div16=[8, 9, 10, 13, 14, 16, 17],
)

_PTX_8 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk03_fa_qkv_kernel     // -- Begin function _sk03_fa_qkv_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk03_fa_qkv_kernel
.visible .entry _sk03_fa_qkv_kernel(
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_6,
	.param .u32 _sk03_fa_qkv_kernel_param_7,
	.param .u32 _sk03_fa_qkv_kernel_param_8,
	.param .u32 _sk03_fa_qkv_kernel_param_9,
	.param .u32 _sk03_fa_qkv_kernel_param_10,
	.param .u32 _sk03_fa_qkv_kernel_param_11,
	.param .u32 _sk03_fa_qkv_kernel_param_12,
	.param .u32 _sk03_fa_qkv_kernel_param_13,
	.param .u32 _sk03_fa_qkv_kernel_param_14,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_16
)
.reqntid 256
{
	.reg .pred 	%p<42>;
	.reg .b16 	%rs<475>;
	.reg .b32 	%r<918>;
	.reg .b64 	%rd<130>;
	.loc	1 145 0                         // sk03_fa_qkv.py:145:0
$L__func_begin0:
	.loc	1 145 0                         // sk03_fa_qkv.py:145:0

// %bb.0:
	ld.param.b32 	%r18, [_sk03_fa_qkv_kernel_param_13];
	ld.param.b32 	%r17, [_sk03_fa_qkv_kernel_param_12];
	ld.param.b32 	%r16, [_sk03_fa_qkv_kernel_param_8];
	ld.param.b32 	%r15, [_sk03_fa_qkv_kernel_param_7];
	ld.param.b64 	%rd13, [_sk03_fa_qkv_kernel_param_5];
	ld.param.b64 	%rd12, [_sk03_fa_qkv_kernel_param_4];
	ld.param.b64 	%rd11, [_sk03_fa_qkv_kernel_param_3];
	ld.param.b64 	%rd10, [_sk03_fa_qkv_kernel_param_2];
	ld.param.b64 	%rd9, [_sk03_fa_qkv_kernel_param_1];
	ld.param.b64 	%rd8, [_sk03_fa_qkv_kernel_param_0];
$L__tmp0:
	.loc	1 154 24                        // sk03_fa_qkv.py:154:24
	mov.u32 	%r40, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk03_fa_qkv.py:155:27 ]
	add.s32 	%r41, %r15, 255;
	.loc	2 43 30                         // standard.py:43:30 @[ sk03_fa_qkv.py:155:27 ]
	shr.s32 	%r42, %r41, 31;
	shr.u32 	%r43, %r42, 24;
	add.s32 	%r44, %r41, %r43;
	shr.s32 	%r45, %r44, 8;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk03_fa_qkv.py:156:27 ]
	add.s32 	%r46, %r16, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk03_fa_qkv.py:156:27 ]
	shr.s32 	%r47, %r46, 31;
	shr.u32 	%r48, %r47, 25;
	add.s32 	%r49, %r46, %r48;
	shr.s32 	%r50, %r49, 7;
$L__tmp3:
	.loc	1 157 29                        // sk03_fa_qkv.py:157:29
	shl.b32 	%r51, %r50, 3;
	.loc	1 158 22                        // sk03_fa_qkv.py:158:22
	div.s32 	%r52, %r40, %r51;
	.loc	1 158 38                        // sk03_fa_qkv.py:158:38
	shl.b32 	%r53, %r52, 3;
	ld.param.b32 	%r54, [_sk03_fa_qkv_kernel_param_9];
	.loc	1 159 30                        // sk03_fa_qkv.py:159:30
	sub.s32 	%r55, %r45, %r53;
	ld.param.b32 	%r56, [_sk03_fa_qkv_kernel_param_10];
	.loc	1 159 39                        // sk03_fa_qkv.py:159:39
	min.s32 	%r57, %r55, 8;
	ld.param.b32 	%r58, [_sk03_fa_qkv_kernel_param_11];
	.loc	1 160 30                        // sk03_fa_qkv.py:160:30
	mul.lo.s32 	%r59, %r52, %r51;
	sub.s32 	%r60, %r40, %r59;
	.loc	1 161 36                        // sk03_fa_qkv.py:161:36
	div.s32 	%r61, %r60, %r57;
	.loc	1 160 46                        // sk03_fa_qkv.py:160:46
	mul.lo.s32 	%r62, %r61, %r57;
	sub.s32 	%r63, %r60, %r62;
	.loc	1 160 23                        // sk03_fa_qkv.py:160:23
	add.s32 	%r64, %r63, %r53;
	.loc	1 163 22                        // sk03_fa_qkv.py:163:22
	shl.b32 	%r1, %r64, 8;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	mov.u32 	%r2, %tid.x;
	shr.u32 	%r65, %r2, 2;
	bfe.u32 	%r66, %r2, 2, 6;
	or.b32 	%r67, %r66, 64;
	and.b32 	%r3, %r2, 255;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r68, %r1, %r66;
	or.b32 	%r69, %r1, %r67;
	or.b32 	%r70, %r68, 128;
	or.b32 	%r71, %r1, %r65;
	or.b32 	%r72, %r71, 192;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r73, %r68, %r15;
	rem.s32 	%r74, %r69, %r15;
	rem.s32 	%r75, %r70, %r15;
	rem.s32 	%r76, %r72, %r15;
	.loc	1 164 22                        // sk03_fa_qkv.py:164:22
	shl.b32 	%r4, %r61, 7;
	.loc	1 164 45                        // sk03_fa_qkv.py:164:45
	and.b32 	%r5, %r2, 3;
	and.b32 	%r6, %r2, 32;
	and.b32 	%r7, %r2, 15;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r77, %r4, %r66;
	or.b32 	%r78, %r4, %r67;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r79, %r77, %r16;
	rem.s32 	%r80, %r78, %r16;
	.loc	1 167 39                        // sk03_fa_qkv.py:167:39
	mul.lo.s32 	%r81, %r73, %r56;
	mul.lo.s32 	%r82, %r74, %r56;
	mul.lo.s32 	%r83, %r75, %r56;
	mul.lo.s32 	%r84, %r76, %r56;
	.loc	1 167 21                        // sk03_fa_qkv.py:167:21
	cvt.s64.s32 	%rd1, %r81;
	add.s64 	%rd32, %rd8, %rd1;
	cvt.s64.s32 	%rd2, %r82;
	add.s64 	%rd33, %rd8, %rd2;
	cvt.s64.s32 	%rd3, %r83;
	add.s64 	%rd34, %rd8, %rd3;
	cvt.s64.s32 	%rd4, %r84;
	add.s64 	%rd35, %rd8, %rd4;
	.loc	1 167 58                        // sk03_fa_qkv.py:167:58
	shl.b32 	%r85, %r5, 4;
	.loc	1 167 51                        // sk03_fa_qkv.py:167:51
	cvt.u64.u32 	%rd5, %r85;
	add.s64 	%rd14, %rd32, %rd5;
	add.s64 	%rd15, %rd33, %rd5;
	add.s64 	%rd16, %rd34, %rd5;
	add.s64 	%rd17, %rd35, %rd5;
	.loc	1 168 21                        // sk03_fa_qkv.py:168:21
	add.s64 	%rd36, %rd9, %rd5;
	.loc	1 168 69                        // sk03_fa_qkv.py:168:69
	mul.lo.s32 	%r86, %r79, %r58;
	mul.lo.s32 	%r87, %r80, %r58;
	.loc	1 168 51                        // sk03_fa_qkv.py:168:51
	cvt.s64.s32 	%rd6, %r86;
	add.s64 	%rd18, %rd36, %rd6;
	cvt.s64.s32 	%rd7, %r87;
	add.s64 	%rd19, %rd36, %rd7;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	setp.gt.s32 	%p1, %r54, 63;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
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
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	add.s32 	%r24, %r19, 49152;
	// begin inline asm
	cp.async.cg.shared.global [ %r24 + 0 ], [ %rd18 + 0 ], 0x10, %r20;
	// end inline asm
	add.s32 	%r25, %r19, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r25 + 0 ], [ %rd19 + 0 ], 0x10, %r20;
	// end inline asm
	cp.async.commit_group;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	setp.gt.s32 	%p2, %r54, 127;
	.loc	1 183 18                        // sk03_fa_qkv.py:183:18
	add.s64 	%rd20, %rd14, 64;
	add.s64 	%rd21, %rd15, 64;
	add.s64 	%rd22, %rd16, 64;
	add.s64 	%rd23, %rd17, 64;
	.loc	1 184 18                        // sk03_fa_qkv.py:184:18
	add.s64 	%rd24, %rd18, 64;
	add.s64 	%rd25, %rd19, 64;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
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
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	add.s32 	%r31, %r19, 57344;
	// begin inline asm
	cp.async.cg.shared.global [ %r31 + 0 ], [ %rd24 + 0 ], 0x10, %r27;
	// end inline asm
	add.s32 	%r32, %r19, 61440;
	// begin inline asm
	cp.async.cg.shared.global [ %r32 + 0 ], [ %rd25 + 0 ], 0x10, %r27;
	// end inline asm
	cp.async.commit_group;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	setp.gt.s32 	%p3, %r54, 191;
	.loc	1 183 18                        // sk03_fa_qkv.py:183:18
	add.s64 	%rd26, %rd14, 128;
	add.s64 	%rd27, %rd15, 128;
	add.s64 	%rd28, %rd16, 128;
	add.s64 	%rd29, %rd17, 128;
	.loc	1 184 18                        // sk03_fa_qkv.py:184:18
	add.s64 	%rd30, %rd18, 128;
	add.s64 	%rd31, %rd19, 128;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
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
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	add.s32 	%r38, %r19, 65536;
	// begin inline asm
	cp.async.cg.shared.global [ %r38 + 0 ], [ %rd30 + 0 ], 0x10, %r34;
	// end inline asm
	add.s32 	%r39, %r19, 69632;
	// begin inline asm
	cp.async.cg.shared.global [ %r39 + 0 ], [ %rd31 + 0 ], 0x10, %r34;
	// end inline asm
	cp.async.commit_group;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	@%p1 bra 	$L__BB0_2;
	bra.uni 	$L__BB0_1;
$L__BB0_2:                              // %.lr.ph
	.loc	1 0 23                          // sk03_fa_qkv.py:0:23
	shr.s32 	%r88, %r54, 31;
	shr.u32 	%r89, %r88, 26;
	add.s32 	%r90, %r54, %r89;
	shr.s32 	%r8, %r90, 6;
	add.s32 	%r11, %r8, -3;
	shl.b32 	%r94, %r7, 6;
	shl.b32 	%r908, %r2, 4;
	and.b32 	%r95, %r908, 3072;
	shl.b32 	%r96, %r2, 3;
	and.b32 	%r97, %r96, 48;
	and.b32 	%r909, %r2, 16;
	or.b32 	%r98, %r94, %r95;
	xor.b32 	%r99, %r97, %r909;
	or.b32 	%r12, %r98, %r99;
	xor.b32 	%r13, %r12, 32;
	shl.b32 	%r100, %r2, 6;
	and.b32 	%r101, %r100, 448;
	shl.b32 	%r102, %r6, 4;
	or.b32 	%r103, %r101, %r97;
	xor.b32 	%r104, %r103, %r10;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
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
	mov.b32 	%r779, 0;
	mov.b32 	%r778, 2;
	mov.b32 	%r777, -1;
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
	mov.b32 	%r843, %r779;
	mov.b32 	%r844, %r779;
	mov.b32 	%r845, %r779;
	mov.b32 	%r846, %r779;
	mov.b32 	%r847, %r779;
	mov.b32 	%r848, %r779;
	mov.b32 	%r849, %r779;
	mov.b32 	%r850, %r779;
	mov.b32 	%r851, %r779;
	mov.b32 	%r852, %r779;
	mov.b32 	%r853, %r779;
	mov.b32 	%r854, %r779;
	mov.b32 	%r855, %r779;
	mov.b32 	%r856, %r779;
	mov.b32 	%r857, %r779;
	mov.b32 	%r858, %r779;
	mov.b32 	%r859, %r779;
	mov.b32 	%r860, %r779;
	mov.b32 	%r861, %r779;
	mov.b32 	%r862, %r779;
	mov.b32 	%r863, %r779;
	mov.b32 	%r864, %r779;
	mov.b32 	%r865, %r779;
	mov.b32 	%r866, %r779;
	mov.b32 	%r867, %r779;
	mov.b32 	%r868, %r779;
	mov.b32 	%r869, %r779;
	mov.b32 	%r870, %r779;
	mov.b32 	%r871, %r779;
	mov.b32 	%r872, %r779;
	mov.b32 	%r873, %r779;
	mov.b32 	%r874, %r779;
	mov.b32 	%r875, %r779;
	mov.b32 	%r876, %r779;
	mov.b32 	%r877, %r779;
	mov.b32 	%r878, %r779;
	mov.b32 	%r879, %r779;
	mov.b32 	%r880, %r779;
	mov.b32 	%r881, %r779;
	mov.b32 	%r882, %r779;
	mov.b32 	%r883, %r779;
	mov.b32 	%r884, %r779;
	mov.b32 	%r885, %r779;
	mov.b32 	%r886, %r779;
	mov.b32 	%r887, %r779;
	mov.b32 	%r888, %r779;
	mov.b32 	%r889, %r779;
	mov.b32 	%r890, %r779;
	mov.b32 	%r891, %r779;
	mov.b32 	%r892, %r779;
	mov.b32 	%r893, %r779;
	mov.b32 	%r894, %r779;
	mov.b32 	%r895, %r779;
	mov.b32 	%r896, %r779;
	mov.b32 	%r897, %r779;
	mov.b32 	%r898, %r779;
	mov.b32 	%r899, %r779;
	mov.b32 	%r900, %r779;
	mov.b32 	%r901, %r779;
	mov.b32 	%r902, %r779;
	mov.b32 	%r903, %r779;
	mov.b32 	%r904, %r779;
	mov.b32 	%r905, %r779;
	mov.b32 	%r906, %r779;
	mov.b32 	%r907, %r779;
$L__BB0_3:                              // =>This Inner Loop Header: Depth=1
	setp.lt.s32 	%p4, %r907, %r11;
	add.s32 	%r177, %r777, 1;
	setp.gt.s32 	%p5, %r177, 2;
	selp.b32 	%r777, 0, %r177, %p5;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
	cp.async.wait_group 	4;
	bar.sync 	0;
	shl.b32 	%r178, %r777, 14;
	add.s32 	%r179, %r93, %r178;
	add.s32 	%r180, %r179, %r12;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r106, %r107, %r108, %r109}, [%r180];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r126, %r127, %r128, %r129}, [%r180+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r130, %r131, %r132, %r133}, [%r180+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r134, %r135, %r136, %r137}, [%r180+12288];
	add.s32 	%r181, %r179, %r13;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r138, %r139, %r140, %r141}, [%r181];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r158, %r159, %r160, %r161}, [%r181+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r162, %r163, %r164, %r165}, [%r181+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r166, %r167, %r168, %r169}, [%r181+12288];
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	shl.b32 	%r182, %r777, 13;
	add.s32 	%r183, %r14, %r182;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r110, %r111, %r142, %r143}, [%r183+49152];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r112, %r113, %r144, %r145}, [%r183+50176];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r114, %r115, %r146, %r147}, [%r183+51200];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r116, %r117, %r148, %r149}, [%r183+52224];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r118, %r119, %r150, %r151}, [%r183+53248];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r120, %r121, %r152, %r153}, [%r183+54272];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r122, %r123, %r154, %r155}, [%r183+55296];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r124, %r125, %r156, %r157}, [%r183+56320];
	.loc	1 179 36                        // sk03_fa_qkv.py:179:36
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r779, %r780, %r781, %r782 }, { %r106, %r107, %r108, %r109 }, { %r110, %r111 }, { %r779, %r780, %r781, %r782 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r783, %r784, %r785, %r786 }, { %r106, %r107, %r108, %r109 }, { %r112, %r113 }, { %r783, %r784, %r785, %r786 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r787, %r788, %r789, %r790 }, { %r106, %r107, %r108, %r109 }, { %r114, %r115 }, { %r787, %r788, %r789, %r790 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r791, %r792, %r793, %r794 }, { %r106, %r107, %r108, %r109 }, { %r116, %r117 }, { %r791, %r792, %r793, %r794 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r795, %r796, %r797, %r798 }, { %r106, %r107, %r108, %r109 }, { %r118, %r119 }, { %r795, %r796, %r797, %r798 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r799, %r800, %r801, %r802 }, { %r106, %r107, %r108, %r109 }, { %r120, %r121 }, { %r799, %r800, %r801, %r802 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r803, %r804, %r805, %r806 }, { %r106, %r107, %r108, %r109 }, { %r122, %r123 }, { %r803, %r804, %r805, %r806 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r807, %r808, %r809, %r810 }, { %r106, %r107, %r108, %r109 }, { %r124, %r125 }, { %r807, %r808, %r809, %r810 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r811, %r812, %r813, %r814 }, { %r126, %r127, %r128, %r129 }, { %r110, %r111 }, { %r811, %r812, %r813, %r814 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r815, %r816, %r817, %r818 }, { %r126, %r127, %r128, %r129 }, { %r112, %r113 }, { %r815, %r816, %r817, %r818 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r819, %r820, %r821, %r822 }, { %r126, %r127, %r128, %r129 }, { %r114, %r115 }, { %r819, %r820, %r821, %r822 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r823, %r824, %r825, %r826 }, { %r126, %r127, %r128, %r129 }, { %r116, %r117 }, { %r823, %r824, %r825, %r826 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r827, %r828, %r829, %r830 }, { %r126, %r127, %r128, %r129 }, { %r118, %r119 }, { %r827, %r828, %r829, %r830 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r831, %r832, %r833, %r834 }, { %r126, %r127, %r128, %r129 }, { %r120, %r121 }, { %r831, %r832, %r833, %r834 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r835, %r836, %r837, %r838 }, { %r126, %r127, %r128, %r129 }, { %r122, %r123 }, { %r835, %r836, %r837, %r838 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r839, %r840, %r841, %r842 }, { %r126, %r127, %r128, %r129 }, { %r124, %r125 }, { %r839, %r840, %r841, %r842 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r843, %r844, %r845, %r846 }, { %r130, %r131, %r132, %r133 }, { %r110, %r111 }, { %r843, %r844, %r845, %r846 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r847, %r848, %r849, %r850 }, { %r130, %r131, %r132, %r133 }, { %r112, %r113 }, { %r847, %r848, %r849, %r850 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r851, %r852, %r853, %r854 }, { %r130, %r131, %r132, %r133 }, { %r114, %r115 }, { %r851, %r852, %r853, %r854 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r855, %r856, %r857, %r858 }, { %r130, %r131, %r132, %r133 }, { %r116, %r117 }, { %r855, %r856, %r857, %r858 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r859, %r860, %r861, %r862 }, { %r130, %r131, %r132, %r133 }, { %r118, %r119 }, { %r859, %r860, %r861, %r862 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r863, %r864, %r865, %r866 }, { %r130, %r131, %r132, %r133 }, { %r120, %r121 }, { %r863, %r864, %r865, %r866 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r867, %r868, %r869, %r870 }, { %r130, %r131, %r132, %r133 }, { %r122, %r123 }, { %r867, %r868, %r869, %r870 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r871, %r872, %r873, %r874 }, { %r130, %r131, %r132, %r133 }, { %r124, %r125 }, { %r871, %r872, %r873, %r874 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r875, %r876, %r877, %r878 }, { %r134, %r135, %r136, %r137 }, { %r110, %r111 }, { %r875, %r876, %r877, %r878 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r879, %r880, %r881, %r882 }, { %r134, %r135, %r136, %r137 }, { %r112, %r113 }, { %r879, %r880, %r881, %r882 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r883, %r884, %r885, %r886 }, { %r134, %r135, %r136, %r137 }, { %r114, %r115 }, { %r883, %r884, %r885, %r886 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r887, %r888, %r889, %r890 }, { %r134, %r135, %r136, %r137 }, { %r116, %r117 }, { %r887, %r888, %r889, %r890 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r891, %r892, %r893, %r894 }, { %r134, %r135, %r136, %r137 }, { %r118, %r119 }, { %r891, %r892, %r893, %r894 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r895, %r896, %r897, %r898 }, { %r134, %r135, %r136, %r137 }, { %r120, %r121 }, { %r895, %r896, %r897, %r898 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r899, %r900, %r901, %r902 }, { %r134, %r135, %r136, %r137 }, { %r122, %r123 }, { %r899, %r900, %r901, %r902 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r903, %r904, %r905, %r906 }, { %r134, %r135, %r136, %r137 }, { %r124, %r125 }, { %r903, %r904, %r905, %r906 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r779, %r780, %r781, %r782 }, { %r138, %r139, %r140, %r141 }, { %r142, %r143 }, { %r779, %r780, %r781, %r782 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r783, %r784, %r785, %r786 }, { %r138, %r139, %r140, %r141 }, { %r144, %r145 }, { %r783, %r784, %r785, %r786 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r787, %r788, %r789, %r790 }, { %r138, %r139, %r140, %r141 }, { %r146, %r147 }, { %r787, %r788, %r789, %r790 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r791, %r792, %r793, %r794 }, { %r138, %r139, %r140, %r141 }, { %r148, %r149 }, { %r791, %r792, %r793, %r794 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r795, %r796, %r797, %r798 }, { %r138, %r139, %r140, %r141 }, { %r150, %r151 }, { %r795, %r796, %r797, %r798 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r799, %r800, %r801, %r802 }, { %r138, %r139, %r140, %r141 }, { %r152, %r153 }, { %r799, %r800, %r801, %r802 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r803, %r804, %r805, %r806 }, { %r138, %r139, %r140, %r141 }, { %r154, %r155 }, { %r803, %r804, %r805, %r806 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r807, %r808, %r809, %r810 }, { %r138, %r139, %r140, %r141 }, { %r156, %r157 }, { %r807, %r808, %r809, %r810 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r811, %r812, %r813, %r814 }, { %r158, %r159, %r160, %r161 }, { %r142, %r143 }, { %r811, %r812, %r813, %r814 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r815, %r816, %r817, %r818 }, { %r158, %r159, %r160, %r161 }, { %r144, %r145 }, { %r815, %r816, %r817, %r818 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r819, %r820, %r821, %r822 }, { %r158, %r159, %r160, %r161 }, { %r146, %r147 }, { %r819, %r820, %r821, %r822 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r823, %r824, %r825, %r826 }, { %r158, %r159, %r160, %r161 }, { %r148, %r149 }, { %r823, %r824, %r825, %r826 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r827, %r828, %r829, %r830 }, { %r158, %r159, %r160, %r161 }, { %r150, %r151 }, { %r827, %r828, %r829, %r830 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r831, %r832, %r833, %r834 }, { %r158, %r159, %r160, %r161 }, { %r152, %r153 }, { %r831, %r832, %r833, %r834 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r835, %r836, %r837, %r838 }, { %r158, %r159, %r160, %r161 }, { %r154, %r155 }, { %r835, %r836, %r837, %r838 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r839, %r840, %r841, %r842 }, { %r158, %r159, %r160, %r161 }, { %r156, %r157 }, { %r839, %r840, %r841, %r842 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r843, %r844, %r845, %r846 }, { %r162, %r163, %r164, %r165 }, { %r142, %r143 }, { %r843, %r844, %r845, %r846 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r847, %r848, %r849, %r850 }, { %r162, %r163, %r164, %r165 }, { %r144, %r145 }, { %r847, %r848, %r849, %r850 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r851, %r852, %r853, %r854 }, { %r162, %r163, %r164, %r165 }, { %r146, %r147 }, { %r851, %r852, %r853, %r854 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r855, %r856, %r857, %r858 }, { %r162, %r163, %r164, %r165 }, { %r148, %r149 }, { %r855, %r856, %r857, %r858 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r859, %r860, %r861, %r862 }, { %r162, %r163, %r164, %r165 }, { %r150, %r151 }, { %r859, %r860, %r861, %r862 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r863, %r864, %r865, %r866 }, { %r162, %r163, %r164, %r165 }, { %r152, %r153 }, { %r863, %r864, %r865, %r866 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r867, %r868, %r869, %r870 }, { %r162, %r163, %r164, %r165 }, { %r154, %r155 }, { %r867, %r868, %r869, %r870 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r871, %r872, %r873, %r874 }, { %r162, %r163, %r164, %r165 }, { %r156, %r157 }, { %r871, %r872, %r873, %r874 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r875, %r876, %r877, %r878 }, { %r166, %r167, %r168, %r169 }, { %r142, %r143 }, { %r875, %r876, %r877, %r878 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r879, %r880, %r881, %r882 }, { %r166, %r167, %r168, %r169 }, { %r144, %r145 }, { %r879, %r880, %r881, %r882 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r883, %r884, %r885, %r886 }, { %r166, %r167, %r168, %r169 }, { %r146, %r147 }, { %r883, %r884, %r885, %r886 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r887, %r888, %r889, %r890 }, { %r166, %r167, %r168, %r169 }, { %r148, %r149 }, { %r887, %r888, %r889, %r890 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r891, %r892, %r893, %r894 }, { %r166, %r167, %r168, %r169 }, { %r150, %r151 }, { %r891, %r892, %r893, %r894 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r895, %r896, %r897, %r898 }, { %r166, %r167, %r168, %r169 }, { %r152, %r153 }, { %r895, %r896, %r897, %r898 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r899, %r900, %r901, %r902 }, { %r166, %r167, %r168, %r169 }, { %r154, %r155 }, { %r899, %r900, %r901, %r902 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r903, %r904, %r905, %r906 }, { %r166, %r167, %r168, %r169 }, { %r156, %r157 }, { %r903, %r904, %r905, %r906 };
	// end inline asm
	.loc	1 183 18                        // sk03_fa_qkv.py:183:18
	add.s64 	%rd43, %rd124, %rd5;
	add.s64 	%rd44, %rd125, %rd5;
	add.s64 	%rd45, %rd126, %rd5;
	.loc	1 184 18                        // sk03_fa_qkv.py:184:18
	add.s64 	%rd46, %rd127, %rd5;
	add.s64 	%rd47, %rd128, %rd5;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	add.s64 	%rd48, %rd129, %rd5;
	add.s32 	%r184, %r778, 1;
	setp.gt.s32 	%p6, %r184, 2;
	selp.b32 	%r778, 0, %r184, %p6;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
	shl.b32 	%r185, %r778, 14;
	bar.sync 	0;
	add.s32 	%r170, %r19, %r185;
	selp.b32 	%r171, 16, 0, %p4;
	// begin inline asm
	cp.async.cg.shared.global [ %r170 + 0 ], [ %rd43 + 0 ], 0x10, %r171;
	// end inline asm
	add.s32 	%r172, %r170, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r172 + 0 ], [ %rd44 + 0 ], 0x10, %r171;
	// end inline asm
	add.s32 	%r173, %r170, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r173 + 0 ], [ %rd45 + 0 ], 0x10, %r171;
	// end inline asm
	add.s32 	%r174, %r170, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r174 + 0 ], [ %rd46 + 0 ], 0x10, %r171;
	// end inline asm
	cp.async.commit_group;
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	shl.b32 	%r186, %r778, 13;
	add.s32 	%r187, %r19, %r186;
	add.s32 	%r175, %r187, 49152;
	// begin inline asm
	cp.async.cg.shared.global [ %r175 + 0 ], [ %rd47 + 0 ], 0x10, %r171;
	// end inline asm
	add.s32 	%r176, %r187, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r176 + 0 ], [ %rd48 + 0 ], 0x10, %r171;
	// end inline asm
	cp.async.commit_group;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	add.s32 	%r907, %r907, 1;
	add.s64 	%rd129, %rd129, 64;
	add.s64 	%rd128, %rd128, 64;
	add.s64 	%rd127, %rd127, 64;
	add.s64 	%rd126, %rd126, 64;
	add.s64 	%rd125, %rd125, 64;
	add.s64 	%rd124, %rd124, 64;
	setp.ne.b32 	%p7, %r8, %r907;
	@%p7 bra 	$L__BB0_3;
// %bb.4:                               // %._crit_edge.loopexit
	.loc	1 186 17                        // sk03_fa_qkv.py:186:17
	cvt.rn.f32.s32 	%r188, %r779;
	cvt.rn.bf16.f32 	%rs363, %r188;
	cvt.rn.f32.s32 	%r189, %r780;
	cvt.rn.bf16.f32 	%rs364, %r189;
	cvt.rn.f32.s32 	%r190, %r781;
	cvt.rn.bf16.f32 	%rs365, %r190;
	cvt.rn.f32.s32 	%r191, %r782;
	cvt.rn.bf16.f32 	%rs366, %r191;
	cvt.rn.f32.s32 	%r192, %r783;
	cvt.rn.bf16.f32 	%rs367, %r192;
	cvt.rn.f32.s32 	%r193, %r784;
	cvt.rn.bf16.f32 	%rs368, %r193;
	cvt.rn.f32.s32 	%r194, %r785;
	cvt.rn.bf16.f32 	%rs369, %r194;
	cvt.rn.f32.s32 	%r195, %r786;
	cvt.rn.bf16.f32 	%rs370, %r195;
	cvt.rn.f32.s32 	%r196, %r787;
	cvt.rn.bf16.f32 	%rs371, %r196;
	cvt.rn.f32.s32 	%r197, %r788;
	cvt.rn.bf16.f32 	%rs372, %r197;
	cvt.rn.f32.s32 	%r198, %r789;
	cvt.rn.bf16.f32 	%rs373, %r198;
	cvt.rn.f32.s32 	%r199, %r790;
	cvt.rn.bf16.f32 	%rs374, %r199;
	cvt.rn.f32.s32 	%r200, %r791;
	cvt.rn.bf16.f32 	%rs375, %r200;
	cvt.rn.f32.s32 	%r201, %r792;
	cvt.rn.bf16.f32 	%rs376, %r201;
	cvt.rn.f32.s32 	%r202, %r793;
	cvt.rn.bf16.f32 	%rs377, %r202;
	cvt.rn.f32.s32 	%r203, %r794;
	cvt.rn.bf16.f32 	%rs378, %r203;
	cvt.rn.f32.s32 	%r204, %r795;
	cvt.rn.bf16.f32 	%rs379, %r204;
	cvt.rn.f32.s32 	%r205, %r796;
	cvt.rn.bf16.f32 	%rs380, %r205;
	cvt.rn.f32.s32 	%r206, %r797;
	cvt.rn.bf16.f32 	%rs381, %r206;
	cvt.rn.f32.s32 	%r207, %r798;
	cvt.rn.bf16.f32 	%rs382, %r207;
	cvt.rn.f32.s32 	%r208, %r799;
	cvt.rn.bf16.f32 	%rs383, %r208;
	cvt.rn.f32.s32 	%r209, %r800;
	cvt.rn.bf16.f32 	%rs384, %r209;
	cvt.rn.f32.s32 	%r210, %r801;
	cvt.rn.bf16.f32 	%rs385, %r210;
	cvt.rn.f32.s32 	%r211, %r802;
	cvt.rn.bf16.f32 	%rs386, %r211;
	cvt.rn.f32.s32 	%r212, %r803;
	cvt.rn.bf16.f32 	%rs387, %r212;
	cvt.rn.f32.s32 	%r213, %r804;
	cvt.rn.bf16.f32 	%rs388, %r213;
	cvt.rn.f32.s32 	%r214, %r805;
	cvt.rn.bf16.f32 	%rs389, %r214;
	cvt.rn.f32.s32 	%r215, %r806;
	cvt.rn.bf16.f32 	%rs390, %r215;
	cvt.rn.f32.s32 	%r216, %r807;
	cvt.rn.bf16.f32 	%rs391, %r216;
	cvt.rn.f32.s32 	%r217, %r808;
	cvt.rn.bf16.f32 	%rs392, %r217;
	cvt.rn.f32.s32 	%r218, %r809;
	cvt.rn.bf16.f32 	%rs393, %r218;
	cvt.rn.f32.s32 	%r219, %r810;
	cvt.rn.bf16.f32 	%rs394, %r219;
	cvt.rn.f32.s32 	%r220, %r811;
	cvt.rn.bf16.f32 	%rs395, %r220;
	cvt.rn.f32.s32 	%r221, %r812;
	cvt.rn.bf16.f32 	%rs396, %r221;
	cvt.rn.f32.s32 	%r222, %r813;
	cvt.rn.bf16.f32 	%rs397, %r222;
	cvt.rn.f32.s32 	%r223, %r814;
	cvt.rn.bf16.f32 	%rs398, %r223;
	cvt.rn.f32.s32 	%r224, %r815;
	cvt.rn.bf16.f32 	%rs399, %r224;
	cvt.rn.f32.s32 	%r225, %r816;
	cvt.rn.bf16.f32 	%rs400, %r225;
	cvt.rn.f32.s32 	%r226, %r817;
	cvt.rn.bf16.f32 	%rs401, %r226;
	cvt.rn.f32.s32 	%r227, %r818;
	cvt.rn.bf16.f32 	%rs402, %r227;
	cvt.rn.f32.s32 	%r228, %r819;
	cvt.rn.bf16.f32 	%rs403, %r228;
	cvt.rn.f32.s32 	%r229, %r820;
	cvt.rn.bf16.f32 	%rs404, %r229;
	cvt.rn.f32.s32 	%r230, %r821;
	cvt.rn.bf16.f32 	%rs405, %r230;
	cvt.rn.f32.s32 	%r231, %r822;
	cvt.rn.bf16.f32 	%rs406, %r231;
	cvt.rn.f32.s32 	%r232, %r823;
	cvt.rn.bf16.f32 	%rs407, %r232;
	cvt.rn.f32.s32 	%r233, %r824;
	cvt.rn.bf16.f32 	%rs408, %r233;
	cvt.rn.f32.s32 	%r234, %r825;
	cvt.rn.bf16.f32 	%rs409, %r234;
	cvt.rn.f32.s32 	%r235, %r826;
	cvt.rn.bf16.f32 	%rs410, %r235;
	cvt.rn.f32.s32 	%r236, %r827;
	cvt.rn.bf16.f32 	%rs411, %r236;
	cvt.rn.f32.s32 	%r237, %r828;
	cvt.rn.bf16.f32 	%rs412, %r237;
	cvt.rn.f32.s32 	%r238, %r829;
	cvt.rn.bf16.f32 	%rs413, %r238;
	cvt.rn.f32.s32 	%r239, %r830;
	cvt.rn.bf16.f32 	%rs414, %r239;
	cvt.rn.f32.s32 	%r240, %r831;
	cvt.rn.bf16.f32 	%rs415, %r240;
	cvt.rn.f32.s32 	%r241, %r832;
	cvt.rn.bf16.f32 	%rs416, %r241;
	cvt.rn.f32.s32 	%r242, %r833;
	cvt.rn.bf16.f32 	%rs417, %r242;
	cvt.rn.f32.s32 	%r243, %r834;
	cvt.rn.bf16.f32 	%rs418, %r243;
	cvt.rn.f32.s32 	%r244, %r835;
	cvt.rn.bf16.f32 	%rs419, %r244;
	cvt.rn.f32.s32 	%r245, %r836;
	cvt.rn.bf16.f32 	%rs420, %r245;
	cvt.rn.f32.s32 	%r246, %r837;
	cvt.rn.bf16.f32 	%rs421, %r246;
	cvt.rn.f32.s32 	%r247, %r838;
	cvt.rn.bf16.f32 	%rs422, %r247;
	cvt.rn.f32.s32 	%r248, %r839;
	cvt.rn.bf16.f32 	%rs423, %r248;
	cvt.rn.f32.s32 	%r249, %r840;
	cvt.rn.bf16.f32 	%rs424, %r249;
	cvt.rn.f32.s32 	%r250, %r841;
	cvt.rn.bf16.f32 	%rs425, %r250;
	cvt.rn.f32.s32 	%r251, %r842;
	cvt.rn.bf16.f32 	%rs426, %r251;
	cvt.rn.f32.s32 	%r252, %r843;
	cvt.rn.bf16.f32 	%rs427, %r252;
	cvt.rn.f32.s32 	%r253, %r844;
	cvt.rn.bf16.f32 	%rs428, %r253;
	cvt.rn.f32.s32 	%r254, %r845;
	cvt.rn.bf16.f32 	%rs429, %r254;
	cvt.rn.f32.s32 	%r255, %r846;
	cvt.rn.bf16.f32 	%rs430, %r255;
	cvt.rn.f32.s32 	%r256, %r847;
	cvt.rn.bf16.f32 	%rs431, %r256;
	cvt.rn.f32.s32 	%r257, %r848;
	cvt.rn.bf16.f32 	%rs432, %r257;
	cvt.rn.f32.s32 	%r258, %r849;
	cvt.rn.bf16.f32 	%rs433, %r258;
	cvt.rn.f32.s32 	%r259, %r850;
	cvt.rn.bf16.f32 	%rs434, %r259;
	cvt.rn.f32.s32 	%r260, %r851;
	cvt.rn.bf16.f32 	%rs435, %r260;
	cvt.rn.f32.s32 	%r261, %r852;
	cvt.rn.bf16.f32 	%rs436, %r261;
	cvt.rn.f32.s32 	%r262, %r853;
	cvt.rn.bf16.f32 	%rs437, %r262;
	cvt.rn.f32.s32 	%r263, %r854;
	cvt.rn.bf16.f32 	%rs438, %r263;
	cvt.rn.f32.s32 	%r264, %r855;
	cvt.rn.bf16.f32 	%rs439, %r264;
	cvt.rn.f32.s32 	%r265, %r856;
	cvt.rn.bf16.f32 	%rs440, %r265;
	cvt.rn.f32.s32 	%r266, %r857;
	cvt.rn.bf16.f32 	%rs441, %r266;
	cvt.rn.f32.s32 	%r267, %r858;
	cvt.rn.bf16.f32 	%rs442, %r267;
	cvt.rn.f32.s32 	%r268, %r859;
	cvt.rn.bf16.f32 	%rs443, %r268;
	cvt.rn.f32.s32 	%r269, %r860;
	cvt.rn.bf16.f32 	%rs444, %r269;
	cvt.rn.f32.s32 	%r270, %r861;
	cvt.rn.bf16.f32 	%rs445, %r270;
	cvt.rn.f32.s32 	%r271, %r862;
	cvt.rn.bf16.f32 	%rs446, %r271;
	cvt.rn.f32.s32 	%r272, %r863;
	cvt.rn.bf16.f32 	%rs447, %r272;
	cvt.rn.f32.s32 	%r273, %r864;
	cvt.rn.bf16.f32 	%rs448, %r273;
	cvt.rn.f32.s32 	%r274, %r865;
	cvt.rn.bf16.f32 	%rs449, %r274;
	cvt.rn.f32.s32 	%r275, %r866;
	cvt.rn.bf16.f32 	%rs450, %r275;
	cvt.rn.f32.s32 	%r276, %r867;
	cvt.rn.bf16.f32 	%rs451, %r276;
	cvt.rn.f32.s32 	%r277, %r868;
	cvt.rn.bf16.f32 	%rs452, %r277;
	cvt.rn.f32.s32 	%r278, %r869;
	cvt.rn.bf16.f32 	%rs453, %r278;
	cvt.rn.f32.s32 	%r279, %r870;
	cvt.rn.bf16.f32 	%rs454, %r279;
	cvt.rn.f32.s32 	%r280, %r871;
	cvt.rn.bf16.f32 	%rs455, %r280;
	cvt.rn.f32.s32 	%r281, %r872;
	cvt.rn.bf16.f32 	%rs456, %r281;
	cvt.rn.f32.s32 	%r282, %r873;
	cvt.rn.bf16.f32 	%rs457, %r282;
	cvt.rn.f32.s32 	%r283, %r874;
	cvt.rn.bf16.f32 	%rs458, %r283;
	cvt.rn.f32.s32 	%r284, %r875;
	cvt.rn.f32.s32 	%r285, %r876;
	cvt.rn.bf16x2.f32 	%r910, %r285, %r284;
	cvt.rn.f32.s32 	%r286, %r877;
	cvt.rn.bf16.f32 	%rs459, %r286;
	cvt.rn.f32.s32 	%r287, %r878;
	cvt.rn.bf16.f32 	%rs460, %r287;
	cvt.rn.f32.s32 	%r288, %r879;
	cvt.rn.f32.s32 	%r289, %r880;
	cvt.rn.bf16x2.f32 	%r912, %r289, %r288;
	cvt.rn.f32.s32 	%r290, %r881;
	cvt.rn.bf16.f32 	%rs461, %r290;
	cvt.rn.f32.s32 	%r291, %r882;
	cvt.rn.bf16.f32 	%rs462, %r291;
	cvt.rn.f32.s32 	%r292, %r883;
	cvt.rn.f32.s32 	%r293, %r884;
	cvt.rn.bf16x2.f32 	%r914, %r293, %r292;
	cvt.rn.f32.s32 	%r294, %r885;
	cvt.rn.bf16.f32 	%rs463, %r294;
	cvt.rn.f32.s32 	%r295, %r886;
	cvt.rn.bf16.f32 	%rs464, %r295;
	cvt.rn.f32.s32 	%r296, %r887;
	cvt.rn.f32.s32 	%r297, %r888;
	cvt.rn.bf16x2.f32 	%r916, %r297, %r296;
	cvt.rn.f32.s32 	%r298, %r889;
	cvt.rn.bf16.f32 	%rs465, %r298;
	cvt.rn.f32.s32 	%r299, %r890;
	cvt.rn.bf16.f32 	%rs466, %r299;
	cvt.rn.f32.s32 	%r300, %r891;
	cvt.rn.f32.s32 	%r301, %r892;
	cvt.rn.bf16x2.f32 	%r911, %r301, %r300;
	cvt.rn.f32.s32 	%r302, %r893;
	cvt.rn.bf16.f32 	%rs467, %r302;
	cvt.rn.f32.s32 	%r303, %r894;
	cvt.rn.bf16.f32 	%rs468, %r303;
	cvt.rn.f32.s32 	%r304, %r895;
	cvt.rn.f32.s32 	%r305, %r896;
	cvt.rn.bf16x2.f32 	%r913, %r305, %r304;
	cvt.rn.f32.s32 	%r306, %r897;
	cvt.rn.bf16.f32 	%rs469, %r306;
	cvt.rn.f32.s32 	%r307, %r898;
	cvt.rn.bf16.f32 	%rs470, %r307;
	cvt.rn.f32.s32 	%r308, %r899;
	cvt.rn.f32.s32 	%r309, %r900;
	cvt.rn.bf16x2.f32 	%r915, %r309, %r308;
	cvt.rn.f32.s32 	%r310, %r901;
	cvt.rn.bf16.f32 	%rs471, %r310;
	cvt.rn.f32.s32 	%r311, %r902;
	cvt.rn.bf16.f32 	%rs472, %r311;
	cvt.rn.f32.s32 	%r312, %r903;
	cvt.rn.f32.s32 	%r313, %r904;
	cvt.rn.bf16x2.f32 	%r917, %r313, %r312;
	cvt.rn.f32.s32 	%r314, %r905;
	cvt.rn.bf16.f32 	%rs473, %r314;
	cvt.rn.f32.s32 	%r315, %r906;
	cvt.rn.bf16.f32 	%rs474, %r315;
	bra.uni 	$L__BB0_5;
$L__BB0_1:                              // %.._crit_edge_crit_edge
	.loc	1 188 19                        // sk03_fa_qkv.py:188:19
	and.b32 	%r909, %r2, 16;
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	shl.b32 	%r908, %r2, 4;
	mov.b32 	%r910, 0;
	mov.b16 	%rs363, 0x0000;
	mov.b16 	%rs364, %rs363;
	mov.b16 	%rs365, %rs363;
	mov.b16 	%rs366, %rs363;
	mov.b16 	%rs367, %rs363;
	mov.b16 	%rs368, %rs363;
	mov.b16 	%rs369, %rs363;
	mov.b16 	%rs370, %rs363;
	mov.b16 	%rs371, %rs363;
	mov.b16 	%rs372, %rs363;
	mov.b16 	%rs373, %rs363;
	mov.b16 	%rs374, %rs363;
	mov.b16 	%rs375, %rs363;
	mov.b16 	%rs376, %rs363;
	mov.b16 	%rs377, %rs363;
	mov.b16 	%rs378, %rs363;
	mov.b16 	%rs379, %rs363;
	mov.b16 	%rs380, %rs363;
	mov.b16 	%rs381, %rs363;
	mov.b16 	%rs382, %rs363;
	mov.b16 	%rs383, %rs363;
	mov.b16 	%rs384, %rs363;
	mov.b16 	%rs385, %rs363;
	mov.b16 	%rs386, %rs363;
	mov.b16 	%rs387, %rs363;
	mov.b16 	%rs388, %rs363;
	mov.b16 	%rs389, %rs363;
	mov.b16 	%rs390, %rs363;
	mov.b16 	%rs391, %rs363;
	mov.b16 	%rs392, %rs363;
	mov.b16 	%rs393, %rs363;
	mov.b16 	%rs394, %rs363;
	mov.b16 	%rs395, %rs363;
	mov.b16 	%rs396, %rs363;
	mov.b16 	%rs397, %rs363;
	mov.b16 	%rs398, %rs363;
	mov.b16 	%rs399, %rs363;
	mov.b16 	%rs400, %rs363;
	mov.b16 	%rs401, %rs363;
	mov.b16 	%rs402, %rs363;
	mov.b16 	%rs403, %rs363;
	mov.b16 	%rs404, %rs363;
	mov.b16 	%rs405, %rs363;
	mov.b16 	%rs406, %rs363;
	mov.b16 	%rs407, %rs363;
	mov.b16 	%rs408, %rs363;
	mov.b16 	%rs409, %rs363;
	mov.b16 	%rs410, %rs363;
	mov.b16 	%rs411, %rs363;
	mov.b16 	%rs412, %rs363;
	mov.b16 	%rs413, %rs363;
	mov.b16 	%rs414, %rs363;
	mov.b16 	%rs415, %rs363;
	mov.b16 	%rs416, %rs363;
	mov.b16 	%rs417, %rs363;
	mov.b16 	%rs418, %rs363;
	mov.b16 	%rs419, %rs363;
	mov.b16 	%rs420, %rs363;
	mov.b16 	%rs421, %rs363;
	mov.b16 	%rs422, %rs363;
	mov.b16 	%rs423, %rs363;
	mov.b16 	%rs424, %rs363;
	mov.b16 	%rs425, %rs363;
	mov.b16 	%rs426, %rs363;
	mov.b16 	%rs427, %rs363;
	mov.b16 	%rs428, %rs363;
	mov.b16 	%rs429, %rs363;
	mov.b16 	%rs430, %rs363;
	mov.b16 	%rs431, %rs363;
	mov.b16 	%rs432, %rs363;
	mov.b16 	%rs433, %rs363;
	mov.b16 	%rs434, %rs363;
	mov.b16 	%rs435, %rs363;
	mov.b16 	%rs436, %rs363;
	mov.b16 	%rs437, %rs363;
	mov.b16 	%rs438, %rs363;
	mov.b16 	%rs439, %rs363;
	mov.b16 	%rs440, %rs363;
	mov.b16 	%rs441, %rs363;
	mov.b16 	%rs442, %rs363;
	mov.b16 	%rs443, %rs363;
	mov.b16 	%rs444, %rs363;
	mov.b16 	%rs445, %rs363;
	mov.b16 	%rs446, %rs363;
	mov.b16 	%rs447, %rs363;
	mov.b16 	%rs448, %rs363;
	mov.b16 	%rs449, %rs363;
	mov.b16 	%rs450, %rs363;
	mov.b16 	%rs451, %rs363;
	mov.b16 	%rs452, %rs363;
	mov.b16 	%rs453, %rs363;
	mov.b16 	%rs454, %rs363;
	mov.b16 	%rs455, %rs363;
	mov.b16 	%rs456, %rs363;
	mov.b16 	%rs457, %rs363;
	mov.b16 	%rs458, %rs363;
	mov.b16 	%rs459, %rs363;
	mov.b16 	%rs460, %rs363;
	mov.b16 	%rs461, %rs363;
	mov.b16 	%rs462, %rs363;
	mov.b16 	%rs463, %rs363;
	mov.b16 	%rs464, %rs363;
	mov.b16 	%rs465, %rs363;
	mov.b16 	%rs466, %rs363;
	mov.b16 	%rs467, %rs363;
	mov.b16 	%rs468, %rs363;
	mov.b16 	%rs469, %rs363;
	mov.b16 	%rs470, %rs363;
	mov.b16 	%rs471, %rs363;
	mov.b16 	%rs472, %rs363;
	mov.b16 	%rs473, %rs363;
	mov.b16 	%rs474, %rs363;
	mov.b32 	%r911, %r910;
	mov.b32 	%r912, %r910;
	mov.b32 	%r913, %r910;
	mov.b32 	%r914, %r910;
	mov.b32 	%r915, %r910;
	mov.b32 	%r916, %r910;
	mov.b32 	%r917, %r910;
$L__BB0_5:                              // %._crit_edge
	.loc	1 164 45                        // sk03_fa_qkv.py:164:45
	shl.b32 	%r546, %r7, 3;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r547, %r4, %r546;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r548, %r547, %r16;
	.loc	1 164 45                        // sk03_fa_qkv.py:164:45
	shr.u32 	%r549, %r6, 2;
	shl.b32 	%r550, %r5, 1;
	or.b32 	%r551, %r549, %r550;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r552, %r551, %r4;
	or.b32 	%r553, %r552, 112;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r554, %r553, %r16;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r555, %r552, 96;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r556, %r555, %r16;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r557, %r552, 80;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r558, %r557, %r16;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r559, %r552, 64;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r560, %r559, %r16;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r561, %r552, 48;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r562, %r561, %r16;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r563, %r552, 32;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r564, %r563, %r16;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r565, %r552, 16;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r566, %r565, %r16;
	rem.s32 	%r567, %r552, %r16;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r568, %r1, %r3;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r569, %r568, %r15;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	and.b32 	%r570, %r2, 240;
	bfe.u32 	%r571, %r2, 4, 4;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r572, %r571, %r1;
	or.b32 	%r573, %r572, 240;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r574, %r573, %r15;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r575, %r572, 224;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r576, %r575, %r15;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r577, %r572, 208;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r578, %r577, %r15;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r579, %r572, 192;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r580, %r579, %r15;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r581, %r572, 176;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r582, %r581, %r15;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r583, %r572, 160;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r584, %r583, %r15;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r585, %r572, 144;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r586, %r585, %r15;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r587, %r572, 128;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r588, %r587, %r15;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r589, %r572, 112;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r590, %r589, %r15;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r591, %r572, 96;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r592, %r591, %r15;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r593, %r572, 80;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r594, %r593, %r15;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r595, %r572, 64;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r596, %r595, %r15;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r597, %r572, 48;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r598, %r597, %r15;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r599, %r572, 32;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r600, %r599, %r15;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r601, %r572, 16;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r602, %r601, %r15;
	rem.s32 	%r603, %r572, %r15;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 186 54                        // sk03_fa_qkv.py:186:54
	mad.wide.s32 	%rd49, %r569, 4, %rd12;
	.loc	1 186 40                        // sk03_fa_qkv.py:186:40
	// begin inline asm
	mov.u32 %r316, 0x0;
	ld.global.b32 { %r316 }, [ %rd49 + 0 ];
	// end inline asm
	.loc	1 186 65                        // sk03_fa_qkv.py:186:65
	cvt.rn.bf16.f32 	%rs1, %r316;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	and.b32 	%r604, %r2, 7;
	shl.b32 	%r605, %r604, 3;
	shl.b32 	%r606, %r2, 2;
	and.b32 	%r607, %r606, 192;
	and.b32 	%r608, %r2, 8;
	shr.u32 	%r609, %r608, 1;
	shr.u32 	%r610, %r2, 5;
	and.b32 	%r611, %r610, 2;
	and.b32 	%r612, %r9, 256;
	add.s32 	%r613, %r93, %r605;
	add.s32 	%r614, %r613, %r607;
	add.s32 	%r615, %r614, %r609;
	add.s32 	%r616, %r615, %r611;
	add.s32 	%r317, %r616, %r612;
	// begin inline asm
	st.shared.b16 [ %r317 + 0 ], %rs1;
	// end inline asm
	bar.sync 	0;
	and.b32 	%r617, %r9, 56;
	and.b32 	%r618, %r2, 192;
	add.s32 	%r619, %r93, %r617;
	add.s32 	%r620, %r619, %r618;
	ld.shared.v4.b16 	{%rs2, %rs3, %rs4, %rs5}, [%r620];
	ld.shared.v4.b16 	{%rs6, %rs7, %rs8, %rs9}, [%r620+256];
	mov.b16 	%rs10, 0x8000;
	fma.rn.bf16 	%rs11, %rs2, %rs363, %rs10;
	fma.rn.bf16 	%rs12, %rs2, %rs364, %rs10;
	fma.rn.bf16 	%rs13, %rs4, %rs365, %rs10;
	fma.rn.bf16 	%rs14, %rs4, %rs366, %rs10;
	fma.rn.bf16 	%rs15, %rs2, %rs367, %rs10;
	fma.rn.bf16 	%rs16, %rs2, %rs368, %rs10;
	fma.rn.bf16 	%rs17, %rs4, %rs369, %rs10;
	fma.rn.bf16 	%rs18, %rs4, %rs370, %rs10;
	fma.rn.bf16 	%rs19, %rs2, %rs371, %rs10;
	fma.rn.bf16 	%rs20, %rs2, %rs372, %rs10;
	fma.rn.bf16 	%rs21, %rs4, %rs373, %rs10;
	fma.rn.bf16 	%rs22, %rs4, %rs374, %rs10;
	fma.rn.bf16 	%rs23, %rs2, %rs375, %rs10;
	fma.rn.bf16 	%rs24, %rs2, %rs376, %rs10;
	fma.rn.bf16 	%rs25, %rs4, %rs377, %rs10;
	fma.rn.bf16 	%rs26, %rs4, %rs378, %rs10;
	fma.rn.bf16 	%rs27, %rs2, %rs379, %rs10;
	fma.rn.bf16 	%rs28, %rs2, %rs380, %rs10;
	fma.rn.bf16 	%rs29, %rs4, %rs381, %rs10;
	fma.rn.bf16 	%rs30, %rs4, %rs382, %rs10;
	fma.rn.bf16 	%rs31, %rs2, %rs383, %rs10;
	fma.rn.bf16 	%rs32, %rs2, %rs384, %rs10;
	fma.rn.bf16 	%rs33, %rs4, %rs385, %rs10;
	fma.rn.bf16 	%rs34, %rs4, %rs386, %rs10;
	fma.rn.bf16 	%rs35, %rs2, %rs387, %rs10;
	fma.rn.bf16 	%rs36, %rs2, %rs388, %rs10;
	fma.rn.bf16 	%rs37, %rs4, %rs389, %rs10;
	fma.rn.bf16 	%rs38, %rs4, %rs390, %rs10;
	fma.rn.bf16 	%rs39, %rs2, %rs391, %rs10;
	fma.rn.bf16 	%rs40, %rs2, %rs392, %rs10;
	fma.rn.bf16 	%rs41, %rs4, %rs393, %rs10;
	fma.rn.bf16 	%rs42, %rs4, %rs394, %rs10;
	fma.rn.bf16 	%rs43, %rs3, %rs395, %rs10;
	fma.rn.bf16 	%rs44, %rs3, %rs396, %rs10;
	fma.rn.bf16 	%rs45, %rs5, %rs397, %rs10;
	fma.rn.bf16 	%rs46, %rs5, %rs398, %rs10;
	fma.rn.bf16 	%rs47, %rs3, %rs399, %rs10;
	fma.rn.bf16 	%rs48, %rs3, %rs400, %rs10;
	fma.rn.bf16 	%rs49, %rs5, %rs401, %rs10;
	fma.rn.bf16 	%rs50, %rs5, %rs402, %rs10;
	fma.rn.bf16 	%rs51, %rs3, %rs403, %rs10;
	fma.rn.bf16 	%rs52, %rs3, %rs404, %rs10;
	fma.rn.bf16 	%rs53, %rs5, %rs405, %rs10;
	fma.rn.bf16 	%rs54, %rs5, %rs406, %rs10;
	fma.rn.bf16 	%rs55, %rs3, %rs407, %rs10;
	fma.rn.bf16 	%rs56, %rs3, %rs408, %rs10;
	fma.rn.bf16 	%rs57, %rs5, %rs409, %rs10;
	fma.rn.bf16 	%rs58, %rs5, %rs410, %rs10;
	fma.rn.bf16 	%rs59, %rs3, %rs411, %rs10;
	fma.rn.bf16 	%rs60, %rs3, %rs412, %rs10;
	fma.rn.bf16 	%rs61, %rs5, %rs413, %rs10;
	fma.rn.bf16 	%rs62, %rs5, %rs414, %rs10;
	fma.rn.bf16 	%rs63, %rs3, %rs415, %rs10;
	fma.rn.bf16 	%rs64, %rs3, %rs416, %rs10;
	fma.rn.bf16 	%rs65, %rs5, %rs417, %rs10;
	fma.rn.bf16 	%rs66, %rs5, %rs418, %rs10;
	fma.rn.bf16 	%rs67, %rs3, %rs419, %rs10;
	fma.rn.bf16 	%rs68, %rs3, %rs420, %rs10;
	fma.rn.bf16 	%rs69, %rs5, %rs421, %rs10;
	fma.rn.bf16 	%rs70, %rs5, %rs422, %rs10;
	fma.rn.bf16 	%rs71, %rs3, %rs423, %rs10;
	fma.rn.bf16 	%rs72, %rs3, %rs424, %rs10;
	fma.rn.bf16 	%rs73, %rs5, %rs425, %rs10;
	fma.rn.bf16 	%rs74, %rs5, %rs426, %rs10;
	fma.rn.bf16 	%rs75, %rs6, %rs427, %rs10;
	fma.rn.bf16 	%rs76, %rs6, %rs428, %rs10;
	fma.rn.bf16 	%rs77, %rs8, %rs429, %rs10;
	fma.rn.bf16 	%rs78, %rs8, %rs430, %rs10;
	fma.rn.bf16 	%rs79, %rs6, %rs431, %rs10;
	fma.rn.bf16 	%rs80, %rs6, %rs432, %rs10;
	fma.rn.bf16 	%rs81, %rs8, %rs433, %rs10;
	fma.rn.bf16 	%rs82, %rs8, %rs434, %rs10;
	fma.rn.bf16 	%rs83, %rs6, %rs435, %rs10;
	fma.rn.bf16 	%rs84, %rs6, %rs436, %rs10;
	fma.rn.bf16 	%rs85, %rs8, %rs437, %rs10;
	fma.rn.bf16 	%rs86, %rs8, %rs438, %rs10;
	fma.rn.bf16 	%rs87, %rs6, %rs439, %rs10;
	fma.rn.bf16 	%rs88, %rs6, %rs440, %rs10;
	fma.rn.bf16 	%rs89, %rs8, %rs441, %rs10;
	fma.rn.bf16 	%rs90, %rs8, %rs442, %rs10;
	fma.rn.bf16 	%rs91, %rs6, %rs443, %rs10;
	fma.rn.bf16 	%rs92, %rs6, %rs444, %rs10;
	fma.rn.bf16 	%rs93, %rs8, %rs445, %rs10;
	fma.rn.bf16 	%rs94, %rs8, %rs446, %rs10;
	fma.rn.bf16 	%rs95, %rs6, %rs447, %rs10;
	fma.rn.bf16 	%rs96, %rs6, %rs448, %rs10;
	fma.rn.bf16 	%rs97, %rs8, %rs449, %rs10;
	fma.rn.bf16 	%rs98, %rs8, %rs450, %rs10;
	fma.rn.bf16 	%rs99, %rs6, %rs451, %rs10;
	fma.rn.bf16 	%rs100, %rs6, %rs452, %rs10;
	fma.rn.bf16 	%rs101, %rs8, %rs453, %rs10;
	fma.rn.bf16 	%rs102, %rs8, %rs454, %rs10;
	fma.rn.bf16 	%rs103, %rs6, %rs455, %rs10;
	fma.rn.bf16 	%rs104, %rs6, %rs456, %rs10;
	fma.rn.bf16 	%rs105, %rs8, %rs457, %rs10;
	fma.rn.bf16 	%rs106, %rs8, %rs458, %rs10;
	fma.rn.bf16 	%rs107, %rs9, %rs459, %rs10;
	fma.rn.bf16 	%rs108, %rs9, %rs460, %rs10;
	fma.rn.bf16 	%rs109, %rs9, %rs461, %rs10;
	fma.rn.bf16 	%rs110, %rs9, %rs462, %rs10;
	fma.rn.bf16 	%rs111, %rs9, %rs463, %rs10;
	fma.rn.bf16 	%rs112, %rs9, %rs464, %rs10;
	fma.rn.bf16 	%rs113, %rs9, %rs465, %rs10;
	fma.rn.bf16 	%rs114, %rs9, %rs466, %rs10;
	fma.rn.bf16 	%rs115, %rs9, %rs467, %rs10;
	fma.rn.bf16 	%rs116, %rs9, %rs468, %rs10;
	fma.rn.bf16 	%rs117, %rs9, %rs469, %rs10;
	fma.rn.bf16 	%rs118, %rs9, %rs470, %rs10;
	fma.rn.bf16 	%rs119, %rs9, %rs471, %rs10;
	fma.rn.bf16 	%rs120, %rs9, %rs472, %rs10;
	fma.rn.bf16 	%rs121, %rs9, %rs473, %rs10;
	fma.rn.bf16 	%rs122, %rs9, %rs474, %rs10;
	.loc	1 187 38                        // sk03_fa_qkv.py:187:38
	mad.wide.s32 	%rd50, %r567, 4, %rd13;
	mad.wide.s32 	%rd51, %r566, 4, %rd13;
	mad.wide.s32 	%rd52, %r564, 4, %rd13;
	mad.wide.s32 	%rd53, %r562, 4, %rd13;
	mad.wide.s32 	%rd54, %r560, 4, %rd13;
	mad.wide.s32 	%rd55, %r558, 4, %rd13;
	mad.wide.s32 	%rd56, %r556, 4, %rd13;
	mad.wide.s32 	%rd57, %r554, 4, %rd13;
	.loc	1 187 24                        // sk03_fa_qkv.py:187:24
	// begin inline asm
	mov.u32 %r318, 0x0;
	mov.u32 %r319, 0x0;
	ld.global.v2.b32 { %r318, %r319 }, [ %rd50 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r320, 0x0;
	mov.u32 %r321, 0x0;
	ld.global.v2.b32 { %r320, %r321 }, [ %rd51 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r322, 0x0;
	mov.u32 %r323, 0x0;
	ld.global.v2.b32 { %r322, %r323 }, [ %rd52 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r324, 0x0;
	mov.u32 %r325, 0x0;
	ld.global.v2.b32 { %r324, %r325 }, [ %rd53 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r326, 0x0;
	mov.u32 %r327, 0x0;
	ld.global.v2.b32 { %r326, %r327 }, [ %rd54 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r328, 0x0;
	mov.u32 %r329, 0x0;
	ld.global.v2.b32 { %r328, %r329 }, [ %rd55 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r330, 0x0;
	mov.u32 %r331, 0x0;
	ld.global.v2.b32 { %r330, %r331 }, [ %rd56 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r332, 0x0;
	mov.u32 %r333, 0x0;
	ld.global.v2.b32 { %r332, %r333 }, [ %rd57 + 0 ];
	// end inline asm
	.loc	1 188 49                        // sk03_fa_qkv.py:188:49
	mul.lo.s32 	%r621, %r603, %r18;
	mul.lo.s32 	%r622, %r602, %r18;
	mul.lo.s32 	%r623, %r600, %r18;
	mul.lo.s32 	%r624, %r598, %r18;
	mul.lo.s32 	%r625, %r596, %r18;
	mul.lo.s32 	%r626, %r594, %r18;
	mul.lo.s32 	%r627, %r592, %r18;
	mul.lo.s32 	%r628, %r590, %r18;
	mul.lo.s32 	%r629, %r588, %r18;
	mul.lo.s32 	%r630, %r586, %r18;
	mul.lo.s32 	%r631, %r584, %r18;
	mul.lo.s32 	%r632, %r582, %r18;
	mul.lo.s32 	%r633, %r580, %r18;
	mul.lo.s32 	%r634, %r578, %r18;
	mul.lo.s32 	%r635, %r576, %r18;
	mul.lo.s32 	%r636, %r574, %r18;
	.loc	1 188 31                        // sk03_fa_qkv.py:188:31
	mad.wide.s32 	%rd90, %r621, 2, %rd11;
	mad.wide.s32 	%rd91, %r622, 2, %rd11;
	mad.wide.s32 	%rd92, %r623, 2, %rd11;
	mad.wide.s32 	%rd93, %r624, 2, %rd11;
	mad.wide.s32 	%rd94, %r625, 2, %rd11;
	mad.wide.s32 	%rd95, %r626, 2, %rd11;
	mad.wide.s32 	%rd96, %r627, 2, %rd11;
	mad.wide.s32 	%rd97, %r628, 2, %rd11;
	mad.wide.s32 	%rd98, %r629, 2, %rd11;
	mad.wide.s32 	%rd99, %r630, 2, %rd11;
	mad.wide.s32 	%rd100, %r631, 2, %rd11;
	mad.wide.s32 	%rd101, %r632, 2, %rd11;
	mad.wide.s32 	%rd102, %r633, 2, %rd11;
	mad.wide.s32 	%rd103, %r634, 2, %rd11;
	mad.wide.s32 	%rd104, %r635, 2, %rd11;
	mad.wide.s32 	%rd105, %r636, 2, %rd11;
	.loc	1 188 64                        // sk03_fa_qkv.py:188:64
	mul.wide.s32 	%rd106, %r548, 2;
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
	.loc	1 188 19                        // sk03_fa_qkv.py:188:19
	// begin inline asm
	mov.u32 %r335, 0x0;
	mov.u32 %r336, 0x0;
	mov.u32 %r337, 0x0;
	mov.u32 %r338, 0x0;
	ld.global.v4.b32 { %r335, %r336, %r337, %r338 }, [ %rd58 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r340, 0x0;
	mov.u32 %r341, 0x0;
	mov.u32 %r342, 0x0;
	mov.u32 %r343, 0x0;
	ld.global.v4.b32 { %r340, %r341, %r342, %r343 }, [ %rd59 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r345, 0x0;
	mov.u32 %r346, 0x0;
	mov.u32 %r347, 0x0;
	mov.u32 %r348, 0x0;
	ld.global.v4.b32 { %r345, %r346, %r347, %r348 }, [ %rd60 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r350, 0x0;
	mov.u32 %r351, 0x0;
	mov.u32 %r352, 0x0;
	mov.u32 %r353, 0x0;
	ld.global.v4.b32 { %r350, %r351, %r352, %r353 }, [ %rd61 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r354, 0x0;
	mov.u32 %r355, 0x0;
	mov.u32 %r356, 0x0;
	mov.u32 %r357, 0x0;
	ld.global.v4.b32 { %r354, %r355, %r356, %r357 }, [ %rd62 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r358, 0x0;
	mov.u32 %r359, 0x0;
	mov.u32 %r360, 0x0;
	mov.u32 %r361, 0x0;
	ld.global.v4.b32 { %r358, %r359, %r360, %r361 }, [ %rd63 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r362, 0x0;
	mov.u32 %r363, 0x0;
	mov.u32 %r364, 0x0;
	mov.u32 %r365, 0x0;
	ld.global.v4.b32 { %r362, %r363, %r364, %r365 }, [ %rd64 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r366, 0x0;
	mov.u32 %r367, 0x0;
	mov.u32 %r368, 0x0;
	mov.u32 %r369, 0x0;
	ld.global.v4.b32 { %r366, %r367, %r368, %r369 }, [ %rd65 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r370, 0x0;
	mov.u32 %r371, 0x0;
	mov.u32 %r372, 0x0;
	mov.u32 %r373, 0x0;
	ld.global.v4.b32 { %r370, %r371, %r372, %r373 }, [ %rd66 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r374, 0x0;
	mov.u32 %r375, 0x0;
	mov.u32 %r376, 0x0;
	mov.u32 %r377, 0x0;
	ld.global.v4.b32 { %r374, %r375, %r376, %r377 }, [ %rd67 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r378, 0x0;
	mov.u32 %r379, 0x0;
	mov.u32 %r380, 0x0;
	mov.u32 %r381, 0x0;
	ld.global.v4.b32 { %r378, %r379, %r380, %r381 }, [ %rd68 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r382, 0x0;
	mov.u32 %r383, 0x0;
	mov.u32 %r384, 0x0;
	mov.u32 %r385, 0x0;
	ld.global.v4.b32 { %r382, %r383, %r384, %r385 }, [ %rd69 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r386, 0x0;
	mov.u32 %r387, 0x0;
	mov.u32 %r388, 0x0;
	mov.u32 %r389, 0x0;
	ld.global.v4.b32 { %r386, %r387, %r388, %r389 }, [ %rd70 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r390, 0x0;
	mov.u32 %r391, 0x0;
	mov.u32 %r392, 0x0;
	mov.u32 %r393, 0x0;
	ld.global.v4.b32 { %r390, %r391, %r392, %r393 }, [ %rd71 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r394, 0x0;
	mov.u32 %r395, 0x0;
	mov.u32 %r396, 0x0;
	mov.u32 %r397, 0x0;
	ld.global.v4.b32 { %r394, %r395, %r396, %r397 }, [ %rd72 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r398, 0x0;
	mov.u32 %r399, 0x0;
	mov.u32 %r400, 0x0;
	mov.u32 %r401, 0x0;
	ld.global.v4.b32 { %r398, %r399, %r400, %r401 }, [ %rd73 + 0 ];
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r637, %r2, 7;
	and.b32 	%r638, %r637, 15360;
	shl.b32 	%r639, %r604, 4;
	or.b32 	%r640, %r638, %r639;
	xor.b32 	%r641, %r640, %r570;
	add.s32 	%r334, %r93, %r641;
	// begin inline asm
	st.shared.v4.b32 [ %r334 + 0 ], { %r335, %r336, %r337, %r338 };
	// end inline asm
	add.s32 	%r339, %r334, 256;
	// begin inline asm
	st.shared.v4.b32 [ %r339 + 0 ], { %r340, %r341, %r342, %r343 };
	// end inline asm
	add.s32 	%r344, %r334, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r344 + 0 ], { %r345, %r346, %r347, %r348 };
	// end inline asm
	add.s32 	%r349, %r334, 768;
	// begin inline asm
	st.shared.v4.b32 [ %r349 + 0 ], { %r350, %r351, %r352, %r353 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r642, %r604, 11;
	shl.b32 	%r643, %r7, 4;
	shl.b32 	%r644, %r618, 2;
	setp.eq.b32 	%p24, %r909, 0;
	shl.b32 	%r645, %r909, 1;
	shr.u32 	%r646, %r6, 1;
	or.b32 	%r647, %r643, %r644;
	or.b32 	%r648, %r645, %r646;
	xor.b32 	%r649, %r647, %r648;
	or.b32 	%r650, %r649, %r642;
	add.s32 	%r651, %r93, %r650;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r652, %r653, %r654, %r655}, [%r651];
	mov.b32 	{%rs123, %rs124}, %r652;
	mov.b32 	{%rs125, %rs126}, %r653;
	mov.b32 	{%rs127, %rs128}, %r654;
	mov.b32 	{%rs129, %rs130}, %r655;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r656, %r657, %r658, %r659}, [%r651+1024];
	mov.b32 	{%rs131, %rs132}, %r656;
	mov.b32 	{%rs133, %rs134}, %r657;
	mov.b32 	{%rs135, %rs136}, %r658;
	mov.b32 	{%rs137, %rs138}, %r659;
	xor.b32 	%r660, %r650, 64;
	add.s32 	%r661, %r93, %r660;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r662, %r663, %r664, %r665}, [%r661];
	mov.b32 	{%rs139, %rs140}, %r662;
	mov.b32 	{%rs141, %rs142}, %r663;
	mov.b32 	{%rs143, %rs144}, %r664;
	mov.b32 	{%rs145, %rs146}, %r665;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r666, %r667, %r668, %r669}, [%r661+1024];
	mov.b32 	{%rs147, %rs148}, %r666;
	mov.b32 	{%rs149, %rs150}, %r667;
	mov.b32 	{%rs151, %rs152}, %r668;
	mov.b32 	{%rs153, %rs154}, %r669;
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r334 + 0 ], { %r354, %r355, %r356, %r357 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r339 + 0 ], { %r358, %r359, %r360, %r361 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r344 + 0 ], { %r362, %r363, %r364, %r365 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r349 + 0 ], { %r366, %r367, %r368, %r369 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r670, %r671, %r672, %r673}, [%r651];
	mov.b32 	{%rs155, %rs156}, %r670;
	mov.b32 	{%rs157, %rs158}, %r671;
	mov.b32 	{%rs159, %rs160}, %r672;
	mov.b32 	{%rs161, %rs162}, %r673;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r674, %r675, %r676, %r677}, [%r651+1024];
	mov.b32 	{%rs163, %rs164}, %r674;
	mov.b32 	{%rs165, %rs166}, %r675;
	mov.b32 	{%rs167, %rs168}, %r676;
	mov.b32 	{%rs169, %rs170}, %r677;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r678, %r679, %r680, %r681}, [%r661];
	mov.b32 	{%rs171, %rs172}, %r678;
	mov.b32 	{%rs173, %rs174}, %r679;
	mov.b32 	{%rs175, %rs176}, %r680;
	mov.b32 	{%rs177, %rs178}, %r681;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r682, %r683, %r684, %r685}, [%r661+1024];
	mov.b32 	{%rs179, %rs180}, %r682;
	mov.b32 	{%rs181, %rs182}, %r683;
	mov.b32 	{%rs183, %rs184}, %r684;
	mov.b32 	{%rs185, %rs186}, %r685;
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r334 + 0 ], { %r370, %r371, %r372, %r373 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r339 + 0 ], { %r374, %r375, %r376, %r377 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r344 + 0 ], { %r378, %r379, %r380, %r381 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r349 + 0 ], { %r382, %r383, %r384, %r385 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r686, %r687, %r688, %r689}, [%r651];
	mov.b32 	{%rs187, %rs188}, %r686;
	mov.b32 	{%rs189, %rs190}, %r687;
	mov.b32 	{%rs191, %rs192}, %r688;
	mov.b32 	{%rs193, %rs194}, %r689;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r690, %r691, %r692, %r693}, [%r651+1024];
	mov.b32 	{%rs195, %rs196}, %r690;
	mov.b32 	{%rs197, %rs198}, %r691;
	mov.b32 	{%rs199, %rs200}, %r692;
	mov.b32 	{%rs201, %rs202}, %r693;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r694, %r695, %r696, %r697}, [%r661];
	mov.b32 	{%rs203, %rs204}, %r694;
	mov.b32 	{%rs205, %rs206}, %r695;
	mov.b32 	{%rs207, %rs208}, %r696;
	mov.b32 	{%rs209, %rs210}, %r697;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r698, %r699, %r700, %r701}, [%r661+1024];
	mov.b32 	{%rs211, %rs212}, %r698;
	mov.b32 	{%rs213, %rs214}, %r699;
	mov.b32 	{%rs215, %rs216}, %r700;
	mov.b32 	{%rs217, %rs218}, %r701;
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r334 + 0 ], { %r386, %r387, %r388, %r389 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r339 + 0 ], { %r390, %r391, %r392, %r393 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r344 + 0 ], { %r394, %r395, %r396, %r397 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r349 + 0 ], { %r398, %r399, %r400, %r401 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r702, %r703, %r704, %r705}, [%r651];
	mov.b32 	{%rs219, %rs220}, %r703;
	mov.b32 	{%rs221, %rs222}, %r705;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r706, %r707, %r708, %r709}, [%r651+1024];
	mov.b32 	{%rs223, %rs224}, %r707;
	mov.b32 	{%rs225, %rs226}, %r709;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r710, %r711, %r712, %r713}, [%r661];
	mov.b32 	{%rs227, %rs228}, %r711;
	mov.b32 	{%rs229, %rs230}, %r713;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r714, %r715, %r716, %r717}, [%r661+1024];
	mov.b32 	{%rs231, %rs232}, %r715;
	mov.b32 	{%rs233, %rs234}, %r717;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	mov.b32 	%r718, {%rs7, %rs7};
	mov.b32 	%r719, -2147450880;
	fma.rn.bf16x2 	%r720, %r718, %r910, %r719;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs235, %r319;
	cvt.rn.bf16.f32 	%rs236, %r318;
	mov.b32 	%r721, {%rs236, %rs235};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs237, %rs11, %rs236, %rs123;
	fma.rn.bf16 	%rs238, %rs12, %rs235, %rs124;
	fma.rn.bf16 	%rs239, %rs13, %rs236, %rs125;
	fma.rn.bf16 	%rs240, %rs14, %rs235, %rs126;
	fma.rn.bf16 	%rs241, %rs43, %rs236, %rs155;
	fma.rn.bf16 	%rs242, %rs44, %rs235, %rs156;
	fma.rn.bf16 	%rs243, %rs45, %rs236, %rs157;
	fma.rn.bf16 	%rs244, %rs46, %rs235, %rs158;
	fma.rn.bf16 	%rs245, %rs75, %rs236, %rs187;
	fma.rn.bf16 	%rs246, %rs76, %rs235, %rs188;
	fma.rn.bf16 	%rs247, %rs77, %rs236, %rs189;
	fma.rn.bf16 	%rs248, %rs78, %rs235, %rs190;
	fma.rn.bf16x2 	%r406, %r720, %r721, %r702;
	fma.rn.bf16 	%rs249, %rs107, %rs236, %rs219;
	fma.rn.bf16 	%rs250, %rs108, %rs235, %rs220;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r722, %r718, %r912, %r719;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs251, %r321;
	cvt.rn.bf16.f32 	%rs252, %r320;
	mov.b32 	%r723, {%rs252, %rs251};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs253, %rs15, %rs252, %rs127;
	fma.rn.bf16 	%rs254, %rs16, %rs251, %rs128;
	fma.rn.bf16 	%rs255, %rs17, %rs252, %rs129;
	fma.rn.bf16 	%rs256, %rs18, %rs251, %rs130;
	fma.rn.bf16 	%rs257, %rs47, %rs252, %rs159;
	fma.rn.bf16 	%rs258, %rs48, %rs251, %rs160;
	fma.rn.bf16 	%rs259, %rs49, %rs252, %rs161;
	fma.rn.bf16 	%rs260, %rs50, %rs251, %rs162;
	fma.rn.bf16 	%rs261, %rs79, %rs252, %rs191;
	fma.rn.bf16 	%rs262, %rs80, %rs251, %rs192;
	fma.rn.bf16 	%rs263, %rs81, %rs252, %rs193;
	fma.rn.bf16 	%rs264, %rs82, %rs251, %rs194;
	fma.rn.bf16x2 	%r426, %r722, %r723, %r704;
	fma.rn.bf16 	%rs265, %rs109, %rs252, %rs221;
	fma.rn.bf16 	%rs266, %rs110, %rs251, %rs222;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r724, %r718, %r914, %r719;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs267, %r323;
	cvt.rn.bf16.f32 	%rs268, %r322;
	mov.b32 	%r725, {%rs268, %rs267};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs269, %rs19, %rs268, %rs139;
	fma.rn.bf16 	%rs270, %rs20, %rs267, %rs140;
	fma.rn.bf16 	%rs271, %rs21, %rs268, %rs141;
	fma.rn.bf16 	%rs272, %rs22, %rs267, %rs142;
	fma.rn.bf16 	%rs273, %rs51, %rs268, %rs171;
	fma.rn.bf16 	%rs274, %rs52, %rs267, %rs172;
	fma.rn.bf16 	%rs275, %rs53, %rs268, %rs173;
	fma.rn.bf16 	%rs276, %rs54, %rs267, %rs174;
	fma.rn.bf16 	%rs277, %rs83, %rs268, %rs203;
	fma.rn.bf16 	%rs278, %rs84, %rs267, %rs204;
	fma.rn.bf16 	%rs279, %rs85, %rs268, %rs205;
	fma.rn.bf16 	%rs280, %rs86, %rs267, %rs206;
	fma.rn.bf16x2 	%r446, %r724, %r725, %r710;
	fma.rn.bf16 	%rs281, %rs111, %rs268, %rs227;
	fma.rn.bf16 	%rs282, %rs112, %rs267, %rs228;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r726, %r718, %r916, %r719;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs283, %r325;
	cvt.rn.bf16.f32 	%rs284, %r324;
	mov.b32 	%r727, {%rs284, %rs283};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs285, %rs23, %rs284, %rs143;
	fma.rn.bf16 	%rs286, %rs24, %rs283, %rs144;
	fma.rn.bf16 	%rs287, %rs25, %rs284, %rs145;
	fma.rn.bf16 	%rs288, %rs26, %rs283, %rs146;
	fma.rn.bf16 	%rs289, %rs55, %rs284, %rs175;
	fma.rn.bf16 	%rs290, %rs56, %rs283, %rs176;
	fma.rn.bf16 	%rs291, %rs57, %rs284, %rs177;
	fma.rn.bf16 	%rs292, %rs58, %rs283, %rs178;
	fma.rn.bf16 	%rs293, %rs87, %rs284, %rs207;
	fma.rn.bf16 	%rs294, %rs88, %rs283, %rs208;
	fma.rn.bf16 	%rs295, %rs89, %rs284, %rs209;
	fma.rn.bf16 	%rs296, %rs90, %rs283, %rs210;
	fma.rn.bf16x2 	%r466, %r726, %r727, %r712;
	fma.rn.bf16 	%rs297, %rs113, %rs284, %rs229;
	fma.rn.bf16 	%rs298, %rs114, %rs283, %rs230;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r728, %r718, %r911, %r719;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs299, %r327;
	cvt.rn.bf16.f32 	%rs300, %r326;
	mov.b32 	%r729, {%rs300, %rs299};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs301, %rs27, %rs300, %rs131;
	fma.rn.bf16 	%rs302, %rs28, %rs299, %rs132;
	fma.rn.bf16 	%rs303, %rs29, %rs300, %rs133;
	fma.rn.bf16 	%rs304, %rs30, %rs299, %rs134;
	fma.rn.bf16 	%rs305, %rs59, %rs300, %rs163;
	fma.rn.bf16 	%rs306, %rs60, %rs299, %rs164;
	fma.rn.bf16 	%rs307, %rs61, %rs300, %rs165;
	fma.rn.bf16 	%rs308, %rs62, %rs299, %rs166;
	fma.rn.bf16 	%rs309, %rs91, %rs300, %rs195;
	fma.rn.bf16 	%rs310, %rs92, %rs299, %rs196;
	fma.rn.bf16 	%rs311, %rs93, %rs300, %rs197;
	fma.rn.bf16 	%rs312, %rs94, %rs299, %rs198;
	fma.rn.bf16x2 	%r416, %r728, %r729, %r706;
	fma.rn.bf16 	%rs313, %rs115, %rs300, %rs223;
	fma.rn.bf16 	%rs314, %rs116, %rs299, %rs224;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r730, %r718, %r913, %r719;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs315, %r329;
	cvt.rn.bf16.f32 	%rs316, %r328;
	mov.b32 	%r731, {%rs316, %rs315};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs317, %rs31, %rs316, %rs135;
	fma.rn.bf16 	%rs318, %rs32, %rs315, %rs136;
	fma.rn.bf16 	%rs319, %rs33, %rs316, %rs137;
	fma.rn.bf16 	%rs320, %rs34, %rs315, %rs138;
	fma.rn.bf16 	%rs321, %rs63, %rs316, %rs167;
	fma.rn.bf16 	%rs322, %rs64, %rs315, %rs168;
	fma.rn.bf16 	%rs323, %rs65, %rs316, %rs169;
	fma.rn.bf16 	%rs324, %rs66, %rs315, %rs170;
	fma.rn.bf16 	%rs325, %rs95, %rs316, %rs199;
	fma.rn.bf16 	%rs326, %rs96, %rs315, %rs200;
	fma.rn.bf16 	%rs327, %rs97, %rs316, %rs201;
	fma.rn.bf16 	%rs328, %rs98, %rs315, %rs202;
	fma.rn.bf16x2 	%r436, %r730, %r731, %r708;
	fma.rn.bf16 	%rs329, %rs117, %rs316, %rs225;
	fma.rn.bf16 	%rs330, %rs118, %rs315, %rs226;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r732, %r718, %r915, %r719;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs331, %r331;
	cvt.rn.bf16.f32 	%rs332, %r330;
	mov.b32 	%r733, {%rs332, %rs331};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs333, %rs35, %rs332, %rs147;
	fma.rn.bf16 	%rs334, %rs36, %rs331, %rs148;
	fma.rn.bf16 	%rs335, %rs37, %rs332, %rs149;
	fma.rn.bf16 	%rs336, %rs38, %rs331, %rs150;
	fma.rn.bf16 	%rs337, %rs67, %rs332, %rs179;
	fma.rn.bf16 	%rs338, %rs68, %rs331, %rs180;
	fma.rn.bf16 	%rs339, %rs69, %rs332, %rs181;
	fma.rn.bf16 	%rs340, %rs70, %rs331, %rs182;
	fma.rn.bf16 	%rs341, %rs99, %rs332, %rs211;
	fma.rn.bf16 	%rs342, %rs100, %rs331, %rs212;
	fma.rn.bf16 	%rs343, %rs101, %rs332, %rs213;
	fma.rn.bf16 	%rs344, %rs102, %rs331, %rs214;
	fma.rn.bf16x2 	%r456, %r732, %r733, %r714;
	fma.rn.bf16 	%rs345, %rs119, %rs332, %rs231;
	fma.rn.bf16 	%rs346, %rs120, %rs331, %rs232;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r734, %r718, %r917, %r719;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs347, %r333;
	cvt.rn.bf16.f32 	%rs348, %r332;
	mov.b32 	%r735, {%rs348, %rs347};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs349, %rs39, %rs348, %rs151;
	fma.rn.bf16 	%rs350, %rs40, %rs347, %rs152;
	fma.rn.bf16 	%rs351, %rs41, %rs348, %rs153;
	fma.rn.bf16 	%rs352, %rs42, %rs347, %rs154;
	fma.rn.bf16 	%rs353, %rs71, %rs348, %rs183;
	fma.rn.bf16 	%rs354, %rs72, %rs347, %rs184;
	fma.rn.bf16 	%rs355, %rs73, %rs348, %rs185;
	fma.rn.bf16 	%rs356, %rs74, %rs347, %rs186;
	fma.rn.bf16 	%rs357, %rs103, %rs348, %rs215;
	fma.rn.bf16 	%rs358, %rs104, %rs347, %rs216;
	fma.rn.bf16 	%rs359, %rs105, %rs348, %rs217;
	fma.rn.bf16 	%rs360, %rs106, %rs347, %rs218;
	fma.rn.bf16x2 	%r476, %r734, %r735, %r716;
	fma.rn.bf16 	%rs361, %rs121, %rs348, %rs233;
	fma.rn.bf16 	%rs362, %rs122, %rs347, %rs234;
	bar.sync 	0;
	shl.b32 	%r736, %r5, 14;
	shl.b32 	%r737, %r5, 5;
	and.b32 	%r738, %r908, 3456;
	bfe.s32 	%r739, %r2, 2, 1;
	and.b32 	%r740, %r739, 8208;
	or.b32 	%r741, %r737, %r738;
	xor.b32 	%r742, %r740, %r646;
	or.b32 	%r743, %r742, %r741;
	or.b32 	%r744, %r743, %r736;
	add.s32 	%r402, %r93, %r744;
	mov.b32 	%r403, {%rs237, %rs238};
	mov.b32 	%r404, {%rs241, %rs242};
	mov.b32 	%r405, {%rs245, %rs246};
	// begin inline asm
	st.shared.v4.b32 [ %r402 + 0 ], { %r403, %r404, %r405, %r406 };
	// end inline asm
	add.s32 	%r407, %r402, 512;
	mov.b32 	%r408, {%rs239, %rs240};
	mov.b32 	%r409, {%rs243, %rs244};
	mov.b32 	%r410, {%rs247, %rs248};
	mov.b32 	%r411, {%rs249, %rs250};
	// begin inline asm
	st.shared.v4.b32 [ %r407 + 0 ], { %r408, %r409, %r410, %r411 };
	// end inline asm
	add.s32 	%r412, %r402, 4096;
	mov.b32 	%r413, {%rs301, %rs302};
	mov.b32 	%r414, {%rs305, %rs306};
	mov.b32 	%r415, {%rs309, %rs310};
	// begin inline asm
	st.shared.v4.b32 [ %r412 + 0 ], { %r413, %r414, %r415, %r416 };
	// end inline asm
	add.s32 	%r417, %r402, 4608;
	mov.b32 	%r418, {%rs303, %rs304};
	mov.b32 	%r419, {%rs307, %rs308};
	mov.b32 	%r420, {%rs311, %rs312};
	mov.b32 	%r421, {%rs313, %rs314};
	// begin inline asm
	st.shared.v4.b32 [ %r417 + 0 ], { %r418, %r419, %r420, %r421 };
	// end inline asm
	xor.b32 	%r745, %r744, 32;
	add.s32 	%r422, %r93, %r745;
	mov.b32 	%r423, {%rs253, %rs254};
	mov.b32 	%r424, {%rs257, %rs258};
	mov.b32 	%r425, {%rs261, %rs262};
	// begin inline asm
	st.shared.v4.b32 [ %r422 + 0 ], { %r423, %r424, %r425, %r426 };
	// end inline asm
	add.s32 	%r427, %r422, 512;
	mov.b32 	%r428, {%rs255, %rs256};
	mov.b32 	%r429, {%rs259, %rs260};
	mov.b32 	%r430, {%rs263, %rs264};
	mov.b32 	%r431, {%rs265, %rs266};
	// begin inline asm
	st.shared.v4.b32 [ %r427 + 0 ], { %r428, %r429, %r430, %r431 };
	// end inline asm
	add.s32 	%r432, %r422, 4096;
	mov.b32 	%r433, {%rs317, %rs318};
	mov.b32 	%r434, {%rs321, %rs322};
	mov.b32 	%r435, {%rs325, %rs326};
	// begin inline asm
	st.shared.v4.b32 [ %r432 + 0 ], { %r433, %r434, %r435, %r436 };
	// end inline asm
	add.s32 	%r437, %r422, 4608;
	mov.b32 	%r438, {%rs319, %rs320};
	mov.b32 	%r439, {%rs323, %rs324};
	mov.b32 	%r440, {%rs327, %rs328};
	mov.b32 	%r441, {%rs329, %rs330};
	// begin inline asm
	st.shared.v4.b32 [ %r437 + 0 ], { %r438, %r439, %r440, %r441 };
	// end inline asm
	xor.b32 	%r746, %r744, 64;
	add.s32 	%r442, %r93, %r746;
	mov.b32 	%r443, {%rs269, %rs270};
	mov.b32 	%r444, {%rs273, %rs274};
	mov.b32 	%r445, {%rs277, %rs278};
	// begin inline asm
	st.shared.v4.b32 [ %r442 + 0 ], { %r443, %r444, %r445, %r446 };
	// end inline asm
	add.s32 	%r447, %r442, 512;
	mov.b32 	%r448, {%rs271, %rs272};
	mov.b32 	%r449, {%rs275, %rs276};
	mov.b32 	%r450, {%rs279, %rs280};
	mov.b32 	%r451, {%rs281, %rs282};
	// begin inline asm
	st.shared.v4.b32 [ %r447 + 0 ], { %r448, %r449, %r450, %r451 };
	// end inline asm
	add.s32 	%r452, %r442, 4096;
	mov.b32 	%r453, {%rs333, %rs334};
	mov.b32 	%r454, {%rs337, %rs338};
	mov.b32 	%r455, {%rs341, %rs342};
	// begin inline asm
	st.shared.v4.b32 [ %r452 + 0 ], { %r453, %r454, %r455, %r456 };
	// end inline asm
	add.s32 	%r457, %r442, 4608;
	mov.b32 	%r458, {%rs335, %rs336};
	mov.b32 	%r459, {%rs339, %rs340};
	mov.b32 	%r460, {%rs343, %rs344};
	mov.b32 	%r461, {%rs345, %rs346};
	// begin inline asm
	st.shared.v4.b32 [ %r457 + 0 ], { %r458, %r459, %r460, %r461 };
	// end inline asm
	xor.b32 	%r747, %r744, 96;
	add.s32 	%r462, %r93, %r747;
	mov.b32 	%r463, {%rs285, %rs286};
	mov.b32 	%r464, {%rs289, %rs290};
	mov.b32 	%r465, {%rs293, %rs294};
	// begin inline asm
	st.shared.v4.b32 [ %r462 + 0 ], { %r463, %r464, %r465, %r466 };
	// end inline asm
	add.s32 	%r467, %r462, 512;
	mov.b32 	%r468, {%rs287, %rs288};
	mov.b32 	%r469, {%rs291, %rs292};
	mov.b32 	%r470, {%rs295, %rs296};
	mov.b32 	%r471, {%rs297, %rs298};
	// begin inline asm
	st.shared.v4.b32 [ %r467 + 0 ], { %r468, %r469, %r470, %r471 };
	// end inline asm
	add.s32 	%r472, %r462, 4096;
	mov.b32 	%r473, {%rs349, %rs350};
	mov.b32 	%r474, {%rs353, %rs354};
	mov.b32 	%r475, {%rs357, %rs358};
	// begin inline asm
	st.shared.v4.b32 [ %r472 + 0 ], { %r473, %r474, %r475, %r476 };
	// end inline asm
	add.s32 	%r477, %r462, 4608;
	mov.b32 	%r478, {%rs351, %rs352};
	mov.b32 	%r479, {%rs355, %rs356};
	mov.b32 	%r480, {%rs359, %rs360};
	mov.b32 	%r481, {%rs361, %rs362};
	// begin inline asm
	st.shared.v4.b32 [ %r477 + 0 ], { %r478, %r479, %r480, %r481 };
	// end inline asm
	bar.sync 	0;
	and.b32 	%r748, %r606, 896;
	shl.b32 	%r749, %r608, 9;
	selp.b32 	%r750, 0, 8208, %p24;
	or.b32 	%r751, %r639, %r748;
	xor.b32 	%r752, %r751, %r750;
	or.b32 	%r753, %r752, %r749;
	add.s32 	%r754, %r93, %r753;
	ld.shared.v4.b32 	{%r482, %r498, %r514, %r530}, [%r754];
	ld.shared.v4.b32 	{%r486, %r502, %r518, %r534}, [%r754+1024];
	ld.shared.v4.b32 	{%r490, %r506, %r522, %r538}, [%r754+2048];
	ld.shared.v4.b32 	{%r494, %r510, %r526, %r542}, [%r754+3072];
	xor.b32 	%r755, %r753, 32;
	add.s32 	%r756, %r93, %r755;
	ld.shared.v4.b32 	{%r483, %r499, %r515, %r531}, [%r756+16384];
	ld.shared.v4.b32 	{%r487, %r503, %r519, %r535}, [%r756+17408];
	ld.shared.v4.b32 	{%r491, %r507, %r523, %r539}, [%r756+18432];
	ld.shared.v4.b32 	{%r495, %r511, %r527, %r543}, [%r756+19456];
	xor.b32 	%r757, %r753, 64;
	add.s32 	%r758, %r93, %r757;
	ld.shared.v4.b32 	{%r484, %r500, %r516, %r532}, [%r758+32768];
	ld.shared.v4.b32 	{%r488, %r504, %r520, %r536}, [%r758+33792];
	ld.shared.v4.b32 	{%r492, %r508, %r524, %r540}, [%r758+34816];
	ld.shared.v4.b32 	{%r496, %r512, %r528, %r544}, [%r758+35840];
	xor.b32 	%r759, %r753, 96;
	add.s32 	%r760, %r93, %r759;
	ld.shared.v4.b32 	{%r485, %r501, %r517, %r533}, [%r760+49152];
	ld.shared.v4.b32 	{%r489, %r505, %r521, %r537}, [%r760+50176];
	ld.shared.v4.b32 	{%r493, %r509, %r525, %r541}, [%r760+51200];
	ld.shared.v4.b32 	{%r497, %r513, %r529, %r545}, [%r760+52224];
	.loc	1 195 31                        // sk03_fa_qkv.py:195:31
	setp.lt.s32 	%p25, %r572, %r15;
	setp.lt.s32 	%p26, %r601, %r15;
	setp.lt.s32 	%p27, %r599, %r15;
	setp.lt.s32 	%p28, %r597, %r15;
	setp.lt.s32 	%p29, %r595, %r15;
	setp.lt.s32 	%p30, %r593, %r15;
	setp.lt.s32 	%p31, %r591, %r15;
	setp.lt.s32 	%p32, %r589, %r15;
	setp.lt.s32 	%p33, %r587, %r15;
	setp.lt.s32 	%p34, %r585, %r15;
	setp.lt.s32 	%p35, %r583, %r15;
	setp.lt.s32 	%p36, %r581, %r15;
	setp.lt.s32 	%p37, %r579, %r15;
	setp.lt.s32 	%p38, %r577, %r15;
	setp.lt.s32 	%p39, %r575, %r15;
	setp.lt.s32 	%p40, %r573, %r15;
	.loc	1 195 54                        // sk03_fa_qkv.py:195:54
	setp.lt.s32 	%p41, %r547, %r16;
	.loc	1 195 37                        // sk03_fa_qkv.py:195:37
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
	.loc	1 193 35                        // sk03_fa_qkv.py:193:35
	mul.lo.s32 	%r761, %r572, %r17;
	mul.lo.s32 	%r762, %r601, %r17;
	mul.lo.s32 	%r763, %r599, %r17;
	mul.lo.s32 	%r764, %r597, %r17;
	mul.lo.s32 	%r765, %r595, %r17;
	mul.lo.s32 	%r766, %r593, %r17;
	mul.lo.s32 	%r767, %r591, %r17;
	mul.lo.s32 	%r768, %r589, %r17;
	mul.lo.s32 	%r769, %r587, %r17;
	mul.lo.s32 	%r770, %r585, %r17;
	mul.lo.s32 	%r771, %r583, %r17;
	mul.lo.s32 	%r772, %r581, %r17;
	mul.lo.s32 	%r773, %r579, %r17;
	mul.lo.s32 	%r774, %r577, %r17;
	mul.lo.s32 	%r775, %r575, %r17;
	mul.lo.s32 	%r776, %r573, %r17;
	.loc	1 193 18                        // sk03_fa_qkv.py:193:18
	mad.wide.s32 	%rd107, %r761, 2, %rd10;
	mad.wide.s32 	%rd108, %r762, 2, %rd10;
	mad.wide.s32 	%rd109, %r763, 2, %rd10;
	mad.wide.s32 	%rd110, %r764, 2, %rd10;
	mad.wide.s32 	%rd111, %r765, 2, %rd10;
	mad.wide.s32 	%rd112, %r766, 2, %rd10;
	mad.wide.s32 	%rd113, %r767, 2, %rd10;
	mad.wide.s32 	%rd114, %r768, 2, %rd10;
	mad.wide.s32 	%rd115, %r769, 2, %rd10;
	mad.wide.s32 	%rd116, %r770, 2, %rd10;
	mad.wide.s32 	%rd117, %r771, 2, %rd10;
	mad.wide.s32 	%rd118, %r772, 2, %rd10;
	mad.wide.s32 	%rd119, %r773, 2, %rd10;
	mad.wide.s32 	%rd120, %r774, 2, %rd10;
	mad.wide.s32 	%rd121, %r775, 2, %rd10;
	mad.wide.s32 	%rd122, %r776, 2, %rd10;
	.loc	1 193 50                        // sk03_fa_qkv.py:193:50
	mul.wide.s32 	%rd123, %r547, 2;
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
	.loc	1 194 8                         // sk03_fa_qkv.py:194:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd74 + 0 ], { %r482, %r483, %r484, %r485 };
	// end inline asm
	// begin inline asm
	@%p9 st.global.v4.b32 [ %rd75 + 0 ], { %r486, %r487, %r488, %r489 };
	// end inline asm
	// begin inline asm
	@%p10 st.global.v4.b32 [ %rd76 + 0 ], { %r490, %r491, %r492, %r493 };
	// end inline asm
	// begin inline asm
	@%p11 st.global.v4.b32 [ %rd77 + 0 ], { %r494, %r495, %r496, %r497 };
	// end inline asm
	// begin inline asm
	@%p12 st.global.v4.b32 [ %rd78 + 0 ], { %r498, %r499, %r500, %r501 };
	// end inline asm
	// begin inline asm
	@%p13 st.global.v4.b32 [ %rd79 + 0 ], { %r502, %r503, %r504, %r505 };
	// end inline asm
	// begin inline asm
	@%p14 st.global.v4.b32 [ %rd80 + 0 ], { %r506, %r507, %r508, %r509 };
	// end inline asm
	// begin inline asm
	@%p15 st.global.v4.b32 [ %rd81 + 0 ], { %r510, %r511, %r512, %r513 };
	// end inline asm
	// begin inline asm
	@%p16 st.global.v4.b32 [ %rd82 + 0 ], { %r514, %r515, %r516, %r517 };
	// end inline asm
	// begin inline asm
	@%p17 st.global.v4.b32 [ %rd83 + 0 ], { %r518, %r519, %r520, %r521 };
	// end inline asm
	// begin inline asm
	@%p18 st.global.v4.b32 [ %rd84 + 0 ], { %r522, %r523, %r524, %r525 };
	// end inline asm
	// begin inline asm
	@%p19 st.global.v4.b32 [ %rd85 + 0 ], { %r526, %r527, %r528, %r529 };
	// end inline asm
	// begin inline asm
	@%p20 st.global.v4.b32 [ %rd86 + 0 ], { %r530, %r531, %r532, %r533 };
	// end inline asm
	// begin inline asm
	@%p21 st.global.v4.b32 [ %rd87 + 0 ], { %r534, %r535, %r536, %r537 };
	// end inline asm
	// begin inline asm
	@%p22 st.global.v4.b32 [ %rd88 + 0 ], { %r538, %r539, %r540, %r541 };
	// end inline asm
	// begin inline asm
	@%p23 st.global.v4.b32 [ %rd89 + 0 ], { %r542, %r543, %r544, %r545 };
	// end inline asm
	.loc	1 192 4                         // sk03_fa_qkv.py:192:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk03_fa_qkv.py"
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
.b32 157                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x96 DW_TAG_compile_unit
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
.b8 51
.b8 95
.b8 102
.b8 97
.b8 95
.b8 113
.b8 107
.b8 118
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
.b8 2                                   // Abbrev [2] 0x44:0x16 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 51
.b8 95
.b8 102
.b8 97
.b8 95
.b8 113
.b8 107
.b8 118
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x5a:0x46 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 68                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x6f:0x18 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 155                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x87:0x18 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 156                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_8 = _Nativo(
    "sk03_fa_qkv/tile256x128x64_shift0_abi15",
    _PTX_8, "_sk03_fa_qkv_kernel",
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

	// .globl	_sk03_fa_qkv_kernel     // -- Begin function _sk03_fa_qkv_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk03_fa_qkv_kernel
.visible .entry _sk03_fa_qkv_kernel(
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_6,
	.param .u32 _sk03_fa_qkv_kernel_param_7,
	.param .u32 _sk03_fa_qkv_kernel_param_8,
	.param .u32 _sk03_fa_qkv_kernel_param_9,
	.param .u32 _sk03_fa_qkv_kernel_param_10,
	.param .u32 _sk03_fa_qkv_kernel_param_11,
	.param .u32 _sk03_fa_qkv_kernel_param_12,
	.param .u32 _sk03_fa_qkv_kernel_param_13,
	.param .u32 _sk03_fa_qkv_kernel_param_14,
	.param .u32 _sk03_fa_qkv_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_16,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_17
)
.reqntid 256
{
	.reg .pred 	%p<42>;
	.reg .b16 	%rs<603>;
	.reg .b32 	%r<941>;
	.reg .b64 	%rd<249>;
	.loc	1 145 0                         // sk03_fa_qkv.py:145:0
$L__func_begin0:
	.loc	1 145 0                         // sk03_fa_qkv.py:145:0

// %bb.0:
	ld.param.b32 	%r19, [_sk03_fa_qkv_kernel_param_14];
	ld.param.b32 	%r18, [_sk03_fa_qkv_kernel_param_13];
	ld.param.b32 	%r17, [_sk03_fa_qkv_kernel_param_12];
	ld.param.b32 	%r16, [_sk03_fa_qkv_kernel_param_8];
	ld.param.b32 	%r15, [_sk03_fa_qkv_kernel_param_7];
	ld.param.b64 	%rd13, [_sk03_fa_qkv_kernel_param_5];
	ld.param.b64 	%rd12, [_sk03_fa_qkv_kernel_param_4];
	ld.param.b64 	%rd11, [_sk03_fa_qkv_kernel_param_3];
	ld.param.b64 	%rd10, [_sk03_fa_qkv_kernel_param_2];
	ld.param.b64 	%rd9, [_sk03_fa_qkv_kernel_param_1];
	ld.param.b64 	%rd8, [_sk03_fa_qkv_kernel_param_0];
$L__tmp0:
	.loc	1 154 24                        // sk03_fa_qkv.py:154:24
	mov.u32 	%r41, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk03_fa_qkv.py:155:27 ]
	add.s32 	%r42, %r15, 255;
	.loc	2 43 30                         // standard.py:43:30 @[ sk03_fa_qkv.py:155:27 ]
	shr.s32 	%r43, %r42, 31;
	shr.u32 	%r44, %r43, 24;
	add.s32 	%r45, %r42, %r44;
	shr.s32 	%r46, %r45, 8;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk03_fa_qkv.py:156:27 ]
	add.s32 	%r47, %r16, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk03_fa_qkv.py:156:27 ]
	shr.s32 	%r48, %r47, 31;
	shr.u32 	%r49, %r48, 25;
	add.s32 	%r50, %r47, %r49;
	shr.s32 	%r51, %r50, 7;
$L__tmp3:
	.loc	1 157 29                        // sk03_fa_qkv.py:157:29
	shl.b32 	%r52, %r51, 3;
	.loc	1 158 22                        // sk03_fa_qkv.py:158:22
	div.s32 	%r53, %r41, %r52;
	.loc	1 158 38                        // sk03_fa_qkv.py:158:38
	shl.b32 	%r54, %r53, 3;
	ld.param.b32 	%r55, [_sk03_fa_qkv_kernel_param_9];
	.loc	1 159 30                        // sk03_fa_qkv.py:159:30
	sub.s32 	%r56, %r46, %r54;
	ld.param.b32 	%r57, [_sk03_fa_qkv_kernel_param_10];
	.loc	1 159 39                        // sk03_fa_qkv.py:159:39
	min.s32 	%r58, %r56, 8;
	ld.param.b32 	%r59, [_sk03_fa_qkv_kernel_param_11];
	.loc	1 160 30                        // sk03_fa_qkv.py:160:30
	mul.lo.s32 	%r60, %r53, %r52;
	sub.s32 	%r61, %r41, %r60;
	.loc	1 161 36                        // sk03_fa_qkv.py:161:36
	div.s32 	%r62, %r61, %r58;
	.loc	1 160 46                        // sk03_fa_qkv.py:160:46
	mul.lo.s32 	%r63, %r62, %r58;
	sub.s32 	%r64, %r61, %r63;
	.loc	1 160 23                        // sk03_fa_qkv.py:160:23
	add.s32 	%r65, %r64, %r54;
	.loc	1 163 22                        // sk03_fa_qkv.py:163:22
	shl.b32 	%r1, %r65, 8;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	mov.u32 	%r2, %tid.x;
	shr.u32 	%r66, %r2, 2;
	bfe.u32 	%r67, %r2, 2, 6;
	or.b32 	%r68, %r67, 64;
	and.b32 	%r3, %r2, 255;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r69, %r1, %r67;
	or.b32 	%r70, %r1, %r68;
	or.b32 	%r71, %r69, 128;
	or.b32 	%r72, %r1, %r66;
	or.b32 	%r73, %r72, 192;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r74, %r69, %r15;
	rem.s32 	%r75, %r70, %r15;
	rem.s32 	%r76, %r71, %r15;
	rem.s32 	%r77, %r73, %r15;
	.loc	1 164 22                        // sk03_fa_qkv.py:164:22
	shl.b32 	%r4, %r62, 7;
	.loc	1 164 45                        // sk03_fa_qkv.py:164:45
	and.b32 	%r5, %r2, 3;
	and.b32 	%r6, %r2, 32;
	and.b32 	%r7, %r2, 15;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r78, %r4, %r67;
	or.b32 	%r79, %r4, %r68;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r80, %r78, %r16;
	rem.s32 	%r81, %r79, %r16;
	.loc	1 167 39                        // sk03_fa_qkv.py:167:39
	mul.lo.s32 	%r82, %r74, %r57;
	mul.lo.s32 	%r83, %r75, %r57;
	mul.lo.s32 	%r84, %r76, %r57;
	mul.lo.s32 	%r85, %r77, %r57;
	.loc	1 167 21                        // sk03_fa_qkv.py:167:21
	cvt.s64.s32 	%rd1, %r82;
	add.s64 	%rd32, %rd8, %rd1;
	cvt.s64.s32 	%rd2, %r83;
	add.s64 	%rd33, %rd8, %rd2;
	cvt.s64.s32 	%rd3, %r84;
	add.s64 	%rd34, %rd8, %rd3;
	cvt.s64.s32 	%rd4, %r85;
	add.s64 	%rd35, %rd8, %rd4;
	.loc	1 167 58                        // sk03_fa_qkv.py:167:58
	shl.b32 	%r86, %r5, 4;
	.loc	1 167 51                        // sk03_fa_qkv.py:167:51
	cvt.u64.u32 	%rd5, %r86;
	add.s64 	%rd14, %rd32, %rd5;
	add.s64 	%rd15, %rd33, %rd5;
	add.s64 	%rd16, %rd34, %rd5;
	add.s64 	%rd17, %rd35, %rd5;
	.loc	1 168 21                        // sk03_fa_qkv.py:168:21
	add.s64 	%rd36, %rd9, %rd5;
	.loc	1 168 69                        // sk03_fa_qkv.py:168:69
	mul.lo.s32 	%r87, %r80, %r59;
	mul.lo.s32 	%r88, %r81, %r59;
	.loc	1 168 51                        // sk03_fa_qkv.py:168:51
	cvt.s64.s32 	%rd6, %r87;
	add.s64 	%rd18, %rd36, %rd6;
	cvt.s64.s32 	%rd7, %r88;
	add.s64 	%rd19, %rd36, %rd7;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	setp.gt.s32 	%p1, %r55, 63;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
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
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	add.s32 	%r25, %r20, 49152;
	// begin inline asm
	cp.async.cg.shared.global [ %r25 + 0 ], [ %rd18 + 0 ], 0x10, %r21;
	// end inline asm
	add.s32 	%r26, %r20, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r26 + 0 ], [ %rd19 + 0 ], 0x10, %r21;
	// end inline asm
	cp.async.commit_group;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	setp.gt.s32 	%p2, %r55, 127;
	.loc	1 183 18                        // sk03_fa_qkv.py:183:18
	add.s64 	%rd20, %rd14, 64;
	add.s64 	%rd21, %rd15, 64;
	add.s64 	%rd22, %rd16, 64;
	add.s64 	%rd23, %rd17, 64;
	.loc	1 184 18                        // sk03_fa_qkv.py:184:18
	add.s64 	%rd24, %rd18, 64;
	add.s64 	%rd25, %rd19, 64;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
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
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	add.s32 	%r32, %r20, 57344;
	// begin inline asm
	cp.async.cg.shared.global [ %r32 + 0 ], [ %rd24 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r33, %r20, 61440;
	// begin inline asm
	cp.async.cg.shared.global [ %r33 + 0 ], [ %rd25 + 0 ], 0x10, %r28;
	// end inline asm
	cp.async.commit_group;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	setp.gt.s32 	%p3, %r55, 191;
	.loc	1 183 18                        // sk03_fa_qkv.py:183:18
	add.s64 	%rd26, %rd14, 128;
	add.s64 	%rd27, %rd15, 128;
	add.s64 	%rd28, %rd16, 128;
	add.s64 	%rd29, %rd17, 128;
	.loc	1 184 18                        // sk03_fa_qkv.py:184:18
	add.s64 	%rd30, %rd18, 128;
	add.s64 	%rd31, %rd19, 128;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
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
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	add.s32 	%r39, %r20, 65536;
	// begin inline asm
	cp.async.cg.shared.global [ %r39 + 0 ], [ %rd30 + 0 ], 0x10, %r35;
	// end inline asm
	add.s32 	%r40, %r20, 69632;
	// begin inline asm
	cp.async.cg.shared.global [ %r40 + 0 ], [ %rd31 + 0 ], 0x10, %r35;
	// end inline asm
	cp.async.commit_group;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	@%p1 bra 	$L__BB0_2;
	bra.uni 	$L__BB0_1;
$L__BB0_2:                              // %.lr.ph
	.loc	1 0 23                          // sk03_fa_qkv.py:0:23
	shr.s32 	%r89, %r55, 31;
	shr.u32 	%r90, %r89, 26;
	add.s32 	%r91, %r55, %r90;
	shr.s32 	%r8, %r91, 6;
	add.s32 	%r11, %r8, -3;
	shl.b32 	%r95, %r7, 6;
	shl.b32 	%r931, %r2, 4;
	and.b32 	%r96, %r931, 3072;
	shl.b32 	%r97, %r2, 3;
	and.b32 	%r98, %r97, 48;
	and.b32 	%r932, %r2, 16;
	or.b32 	%r99, %r95, %r96;
	xor.b32 	%r100, %r98, %r932;
	or.b32 	%r12, %r99, %r100;
	xor.b32 	%r13, %r12, 32;
	shl.b32 	%r101, %r2, 6;
	and.b32 	%r102, %r101, 448;
	shl.b32 	%r103, %r6, 4;
	or.b32 	%r104, %r102, %r98;
	xor.b32 	%r105, %r104, %r10;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
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
	mov.b32 	%r802, 0;
	mov.b32 	%r801, 2;
	mov.b32 	%r800, -1;
	mov.b32 	%r803, %r802;
	mov.b32 	%r804, %r802;
	mov.b32 	%r805, %r802;
	mov.b32 	%r806, %r802;
	mov.b32 	%r807, %r802;
	mov.b32 	%r808, %r802;
	mov.b32 	%r809, %r802;
	mov.b32 	%r810, %r802;
	mov.b32 	%r811, %r802;
	mov.b32 	%r812, %r802;
	mov.b32 	%r813, %r802;
	mov.b32 	%r814, %r802;
	mov.b32 	%r815, %r802;
	mov.b32 	%r816, %r802;
	mov.b32 	%r817, %r802;
	mov.b32 	%r818, %r802;
	mov.b32 	%r819, %r802;
	mov.b32 	%r820, %r802;
	mov.b32 	%r821, %r802;
	mov.b32 	%r822, %r802;
	mov.b32 	%r823, %r802;
	mov.b32 	%r824, %r802;
	mov.b32 	%r825, %r802;
	mov.b32 	%r826, %r802;
	mov.b32 	%r827, %r802;
	mov.b32 	%r828, %r802;
	mov.b32 	%r829, %r802;
	mov.b32 	%r830, %r802;
	mov.b32 	%r831, %r802;
	mov.b32 	%r832, %r802;
	mov.b32 	%r833, %r802;
	mov.b32 	%r834, %r802;
	mov.b32 	%r835, %r802;
	mov.b32 	%r836, %r802;
	mov.b32 	%r837, %r802;
	mov.b32 	%r838, %r802;
	mov.b32 	%r839, %r802;
	mov.b32 	%r840, %r802;
	mov.b32 	%r841, %r802;
	mov.b32 	%r842, %r802;
	mov.b32 	%r843, %r802;
	mov.b32 	%r844, %r802;
	mov.b32 	%r845, %r802;
	mov.b32 	%r846, %r802;
	mov.b32 	%r847, %r802;
	mov.b32 	%r848, %r802;
	mov.b32 	%r849, %r802;
	mov.b32 	%r850, %r802;
	mov.b32 	%r851, %r802;
	mov.b32 	%r852, %r802;
	mov.b32 	%r853, %r802;
	mov.b32 	%r854, %r802;
	mov.b32 	%r855, %r802;
	mov.b32 	%r856, %r802;
	mov.b32 	%r857, %r802;
	mov.b32 	%r858, %r802;
	mov.b32 	%r859, %r802;
	mov.b32 	%r860, %r802;
	mov.b32 	%r861, %r802;
	mov.b32 	%r862, %r802;
	mov.b32 	%r863, %r802;
	mov.b32 	%r864, %r802;
	mov.b32 	%r865, %r802;
	mov.b32 	%r866, %r802;
	mov.b32 	%r867, %r802;
	mov.b32 	%r868, %r802;
	mov.b32 	%r869, %r802;
	mov.b32 	%r870, %r802;
	mov.b32 	%r871, %r802;
	mov.b32 	%r872, %r802;
	mov.b32 	%r873, %r802;
	mov.b32 	%r874, %r802;
	mov.b32 	%r875, %r802;
	mov.b32 	%r876, %r802;
	mov.b32 	%r877, %r802;
	mov.b32 	%r878, %r802;
	mov.b32 	%r879, %r802;
	mov.b32 	%r880, %r802;
	mov.b32 	%r881, %r802;
	mov.b32 	%r882, %r802;
	mov.b32 	%r883, %r802;
	mov.b32 	%r884, %r802;
	mov.b32 	%r885, %r802;
	mov.b32 	%r886, %r802;
	mov.b32 	%r887, %r802;
	mov.b32 	%r888, %r802;
	mov.b32 	%r889, %r802;
	mov.b32 	%r890, %r802;
	mov.b32 	%r891, %r802;
	mov.b32 	%r892, %r802;
	mov.b32 	%r893, %r802;
	mov.b32 	%r894, %r802;
	mov.b32 	%r895, %r802;
	mov.b32 	%r896, %r802;
	mov.b32 	%r897, %r802;
	mov.b32 	%r898, %r802;
	mov.b32 	%r899, %r802;
	mov.b32 	%r900, %r802;
	mov.b32 	%r901, %r802;
	mov.b32 	%r902, %r802;
	mov.b32 	%r903, %r802;
	mov.b32 	%r904, %r802;
	mov.b32 	%r905, %r802;
	mov.b32 	%r906, %r802;
	mov.b32 	%r907, %r802;
	mov.b32 	%r908, %r802;
	mov.b32 	%r909, %r802;
	mov.b32 	%r910, %r802;
	mov.b32 	%r911, %r802;
	mov.b32 	%r912, %r802;
	mov.b32 	%r913, %r802;
	mov.b32 	%r914, %r802;
	mov.b32 	%r915, %r802;
	mov.b32 	%r916, %r802;
	mov.b32 	%r917, %r802;
	mov.b32 	%r918, %r802;
	mov.b32 	%r919, %r802;
	mov.b32 	%r920, %r802;
	mov.b32 	%r921, %r802;
	mov.b32 	%r922, %r802;
	mov.b32 	%r923, %r802;
	mov.b32 	%r924, %r802;
	mov.b32 	%r925, %r802;
	mov.b32 	%r926, %r802;
	mov.b32 	%r927, %r802;
	mov.b32 	%r928, %r802;
	mov.b32 	%r929, %r802;
	mov.b32 	%r930, %r802;
$L__BB0_3:                              // =>This Inner Loop Header: Depth=1
	setp.lt.s32 	%p4, %r930, %r11;
	add.s32 	%r178, %r800, 1;
	setp.gt.s32 	%p5, %r178, 2;
	selp.b32 	%r800, 0, %r178, %p5;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
	cp.async.wait_group 	4;
	bar.sync 	0;
	shl.b32 	%r179, %r800, 14;
	add.s32 	%r180, %r94, %r179;
	add.s32 	%r181, %r180, %r12;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r107, %r108, %r109, %r110}, [%r181];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r127, %r128, %r129, %r130}, [%r181+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r131, %r132, %r133, %r134}, [%r181+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r135, %r136, %r137, %r138}, [%r181+12288];
	add.s32 	%r182, %r180, %r13;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r139, %r140, %r141, %r142}, [%r182];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r159, %r160, %r161, %r162}, [%r182+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r163, %r164, %r165, %r166}, [%r182+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r167, %r168, %r169, %r170}, [%r182+12288];
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	shl.b32 	%r183, %r800, 13;
	add.s32 	%r184, %r14, %r183;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r111, %r112, %r143, %r144}, [%r184+49152];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r113, %r114, %r145, %r146}, [%r184+50176];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r115, %r116, %r147, %r148}, [%r184+51200];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r117, %r118, %r149, %r150}, [%r184+52224];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r119, %r120, %r151, %r152}, [%r184+53248];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r121, %r122, %r153, %r154}, [%r184+54272];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r123, %r124, %r155, %r156}, [%r184+55296];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r125, %r126, %r157, %r158}, [%r184+56320];
	.loc	1 179 36                        // sk03_fa_qkv.py:179:36
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r802, %r803, %r804, %r805 }, { %r107, %r108, %r109, %r110 }, { %r111, %r112 }, { %r802, %r803, %r804, %r805 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r806, %r807, %r808, %r809 }, { %r107, %r108, %r109, %r110 }, { %r113, %r114 }, { %r806, %r807, %r808, %r809 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r810, %r811, %r812, %r813 }, { %r107, %r108, %r109, %r110 }, { %r115, %r116 }, { %r810, %r811, %r812, %r813 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r814, %r815, %r816, %r817 }, { %r107, %r108, %r109, %r110 }, { %r117, %r118 }, { %r814, %r815, %r816, %r817 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r818, %r819, %r820, %r821 }, { %r107, %r108, %r109, %r110 }, { %r119, %r120 }, { %r818, %r819, %r820, %r821 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r822, %r823, %r824, %r825 }, { %r107, %r108, %r109, %r110 }, { %r121, %r122 }, { %r822, %r823, %r824, %r825 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r826, %r827, %r828, %r829 }, { %r107, %r108, %r109, %r110 }, { %r123, %r124 }, { %r826, %r827, %r828, %r829 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r830, %r831, %r832, %r833 }, { %r107, %r108, %r109, %r110 }, { %r125, %r126 }, { %r830, %r831, %r832, %r833 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r834, %r835, %r836, %r837 }, { %r127, %r128, %r129, %r130 }, { %r111, %r112 }, { %r834, %r835, %r836, %r837 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r838, %r839, %r840, %r841 }, { %r127, %r128, %r129, %r130 }, { %r113, %r114 }, { %r838, %r839, %r840, %r841 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r842, %r843, %r844, %r845 }, { %r127, %r128, %r129, %r130 }, { %r115, %r116 }, { %r842, %r843, %r844, %r845 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r846, %r847, %r848, %r849 }, { %r127, %r128, %r129, %r130 }, { %r117, %r118 }, { %r846, %r847, %r848, %r849 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r850, %r851, %r852, %r853 }, { %r127, %r128, %r129, %r130 }, { %r119, %r120 }, { %r850, %r851, %r852, %r853 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r854, %r855, %r856, %r857 }, { %r127, %r128, %r129, %r130 }, { %r121, %r122 }, { %r854, %r855, %r856, %r857 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r858, %r859, %r860, %r861 }, { %r127, %r128, %r129, %r130 }, { %r123, %r124 }, { %r858, %r859, %r860, %r861 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r862, %r863, %r864, %r865 }, { %r127, %r128, %r129, %r130 }, { %r125, %r126 }, { %r862, %r863, %r864, %r865 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r866, %r867, %r868, %r869 }, { %r131, %r132, %r133, %r134 }, { %r111, %r112 }, { %r866, %r867, %r868, %r869 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r870, %r871, %r872, %r873 }, { %r131, %r132, %r133, %r134 }, { %r113, %r114 }, { %r870, %r871, %r872, %r873 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r874, %r875, %r876, %r877 }, { %r131, %r132, %r133, %r134 }, { %r115, %r116 }, { %r874, %r875, %r876, %r877 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r878, %r879, %r880, %r881 }, { %r131, %r132, %r133, %r134 }, { %r117, %r118 }, { %r878, %r879, %r880, %r881 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r882, %r883, %r884, %r885 }, { %r131, %r132, %r133, %r134 }, { %r119, %r120 }, { %r882, %r883, %r884, %r885 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r886, %r887, %r888, %r889 }, { %r131, %r132, %r133, %r134 }, { %r121, %r122 }, { %r886, %r887, %r888, %r889 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r890, %r891, %r892, %r893 }, { %r131, %r132, %r133, %r134 }, { %r123, %r124 }, { %r890, %r891, %r892, %r893 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r894, %r895, %r896, %r897 }, { %r131, %r132, %r133, %r134 }, { %r125, %r126 }, { %r894, %r895, %r896, %r897 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r898, %r899, %r900, %r901 }, { %r135, %r136, %r137, %r138 }, { %r111, %r112 }, { %r898, %r899, %r900, %r901 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r902, %r903, %r904, %r905 }, { %r135, %r136, %r137, %r138 }, { %r113, %r114 }, { %r902, %r903, %r904, %r905 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r906, %r907, %r908, %r909 }, { %r135, %r136, %r137, %r138 }, { %r115, %r116 }, { %r906, %r907, %r908, %r909 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r910, %r911, %r912, %r913 }, { %r135, %r136, %r137, %r138 }, { %r117, %r118 }, { %r910, %r911, %r912, %r913 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r914, %r915, %r916, %r917 }, { %r135, %r136, %r137, %r138 }, { %r119, %r120 }, { %r914, %r915, %r916, %r917 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r918, %r919, %r920, %r921 }, { %r135, %r136, %r137, %r138 }, { %r121, %r122 }, { %r918, %r919, %r920, %r921 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r922, %r923, %r924, %r925 }, { %r135, %r136, %r137, %r138 }, { %r123, %r124 }, { %r922, %r923, %r924, %r925 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r926, %r927, %r928, %r929 }, { %r135, %r136, %r137, %r138 }, { %r125, %r126 }, { %r926, %r927, %r928, %r929 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r802, %r803, %r804, %r805 }, { %r139, %r140, %r141, %r142 }, { %r143, %r144 }, { %r802, %r803, %r804, %r805 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r806, %r807, %r808, %r809 }, { %r139, %r140, %r141, %r142 }, { %r145, %r146 }, { %r806, %r807, %r808, %r809 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r810, %r811, %r812, %r813 }, { %r139, %r140, %r141, %r142 }, { %r147, %r148 }, { %r810, %r811, %r812, %r813 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r814, %r815, %r816, %r817 }, { %r139, %r140, %r141, %r142 }, { %r149, %r150 }, { %r814, %r815, %r816, %r817 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r818, %r819, %r820, %r821 }, { %r139, %r140, %r141, %r142 }, { %r151, %r152 }, { %r818, %r819, %r820, %r821 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r822, %r823, %r824, %r825 }, { %r139, %r140, %r141, %r142 }, { %r153, %r154 }, { %r822, %r823, %r824, %r825 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r826, %r827, %r828, %r829 }, { %r139, %r140, %r141, %r142 }, { %r155, %r156 }, { %r826, %r827, %r828, %r829 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r830, %r831, %r832, %r833 }, { %r139, %r140, %r141, %r142 }, { %r157, %r158 }, { %r830, %r831, %r832, %r833 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r834, %r835, %r836, %r837 }, { %r159, %r160, %r161, %r162 }, { %r143, %r144 }, { %r834, %r835, %r836, %r837 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r838, %r839, %r840, %r841 }, { %r159, %r160, %r161, %r162 }, { %r145, %r146 }, { %r838, %r839, %r840, %r841 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r842, %r843, %r844, %r845 }, { %r159, %r160, %r161, %r162 }, { %r147, %r148 }, { %r842, %r843, %r844, %r845 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r846, %r847, %r848, %r849 }, { %r159, %r160, %r161, %r162 }, { %r149, %r150 }, { %r846, %r847, %r848, %r849 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r850, %r851, %r852, %r853 }, { %r159, %r160, %r161, %r162 }, { %r151, %r152 }, { %r850, %r851, %r852, %r853 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r854, %r855, %r856, %r857 }, { %r159, %r160, %r161, %r162 }, { %r153, %r154 }, { %r854, %r855, %r856, %r857 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r858, %r859, %r860, %r861 }, { %r159, %r160, %r161, %r162 }, { %r155, %r156 }, { %r858, %r859, %r860, %r861 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r862, %r863, %r864, %r865 }, { %r159, %r160, %r161, %r162 }, { %r157, %r158 }, { %r862, %r863, %r864, %r865 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r866, %r867, %r868, %r869 }, { %r163, %r164, %r165, %r166 }, { %r143, %r144 }, { %r866, %r867, %r868, %r869 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r870, %r871, %r872, %r873 }, { %r163, %r164, %r165, %r166 }, { %r145, %r146 }, { %r870, %r871, %r872, %r873 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r874, %r875, %r876, %r877 }, { %r163, %r164, %r165, %r166 }, { %r147, %r148 }, { %r874, %r875, %r876, %r877 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r878, %r879, %r880, %r881 }, { %r163, %r164, %r165, %r166 }, { %r149, %r150 }, { %r878, %r879, %r880, %r881 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r882, %r883, %r884, %r885 }, { %r163, %r164, %r165, %r166 }, { %r151, %r152 }, { %r882, %r883, %r884, %r885 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r886, %r887, %r888, %r889 }, { %r163, %r164, %r165, %r166 }, { %r153, %r154 }, { %r886, %r887, %r888, %r889 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r890, %r891, %r892, %r893 }, { %r163, %r164, %r165, %r166 }, { %r155, %r156 }, { %r890, %r891, %r892, %r893 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r894, %r895, %r896, %r897 }, { %r163, %r164, %r165, %r166 }, { %r157, %r158 }, { %r894, %r895, %r896, %r897 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r898, %r899, %r900, %r901 }, { %r167, %r168, %r169, %r170 }, { %r143, %r144 }, { %r898, %r899, %r900, %r901 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r902, %r903, %r904, %r905 }, { %r167, %r168, %r169, %r170 }, { %r145, %r146 }, { %r902, %r903, %r904, %r905 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r906, %r907, %r908, %r909 }, { %r167, %r168, %r169, %r170 }, { %r147, %r148 }, { %r906, %r907, %r908, %r909 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r910, %r911, %r912, %r913 }, { %r167, %r168, %r169, %r170 }, { %r149, %r150 }, { %r910, %r911, %r912, %r913 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r914, %r915, %r916, %r917 }, { %r167, %r168, %r169, %r170 }, { %r151, %r152 }, { %r914, %r915, %r916, %r917 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r918, %r919, %r920, %r921 }, { %r167, %r168, %r169, %r170 }, { %r153, %r154 }, { %r918, %r919, %r920, %r921 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r922, %r923, %r924, %r925 }, { %r167, %r168, %r169, %r170 }, { %r155, %r156 }, { %r922, %r923, %r924, %r925 };
	// end inline asm
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r926, %r927, %r928, %r929 }, { %r167, %r168, %r169, %r170 }, { %r157, %r158 }, { %r926, %r927, %r928, %r929 };
	// end inline asm
	.loc	1 183 18                        // sk03_fa_qkv.py:183:18
	add.s64 	%rd43, %rd243, %rd5;
	add.s64 	%rd44, %rd244, %rd5;
	add.s64 	%rd45, %rd245, %rd5;
	.loc	1 184 18                        // sk03_fa_qkv.py:184:18
	add.s64 	%rd46, %rd246, %rd5;
	add.s64 	%rd47, %rd247, %rd5;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	add.s64 	%rd48, %rd248, %rd5;
	add.s32 	%r185, %r801, 1;
	setp.gt.s32 	%p6, %r185, 2;
	selp.b32 	%r801, 0, %r185, %p6;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
	shl.b32 	%r186, %r801, 14;
	bar.sync 	0;
	add.s32 	%r171, %r20, %r186;
	selp.b32 	%r172, 16, 0, %p4;
	// begin inline asm
	cp.async.cg.shared.global [ %r171 + 0 ], [ %rd43 + 0 ], 0x10, %r172;
	// end inline asm
	add.s32 	%r173, %r171, 4096;
	// begin inline asm
	cp.async.cg.shared.global [ %r173 + 0 ], [ %rd44 + 0 ], 0x10, %r172;
	// end inline asm
	add.s32 	%r174, %r171, 8192;
	// begin inline asm
	cp.async.cg.shared.global [ %r174 + 0 ], [ %rd45 + 0 ], 0x10, %r172;
	// end inline asm
	add.s32 	%r175, %r171, 12288;
	// begin inline asm
	cp.async.cg.shared.global [ %r175 + 0 ], [ %rd46 + 0 ], 0x10, %r172;
	// end inline asm
	cp.async.commit_group;
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	shl.b32 	%r187, %r801, 13;
	add.s32 	%r188, %r20, %r187;
	add.s32 	%r176, %r188, 49152;
	// begin inline asm
	cp.async.cg.shared.global [ %r176 + 0 ], [ %rd47 + 0 ], 0x10, %r172;
	// end inline asm
	add.s32 	%r177, %r188, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r177 + 0 ], [ %rd48 + 0 ], 0x10, %r172;
	// end inline asm
	cp.async.commit_group;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	add.s32 	%r930, %r930, 1;
	add.s64 	%rd248, %rd248, 64;
	add.s64 	%rd247, %rd247, 64;
	add.s64 	%rd246, %rd246, 64;
	add.s64 	%rd245, %rd245, 64;
	add.s64 	%rd244, %rd244, 64;
	add.s64 	%rd243, %rd243, 64;
	setp.ne.b32 	%p7, %r8, %r930;
	@%p7 bra 	$L__BB0_3;
// %bb.4:                               // %._crit_edge.loopexit
	.loc	1 186 17                        // sk03_fa_qkv.py:186:17
	cvt.rn.f32.s32 	%r189, %r802;
	cvt.rn.bf16.f32 	%rs491, %r189;
	cvt.rn.f32.s32 	%r190, %r803;
	cvt.rn.bf16.f32 	%rs492, %r190;
	cvt.rn.f32.s32 	%r191, %r804;
	cvt.rn.bf16.f32 	%rs493, %r191;
	cvt.rn.f32.s32 	%r192, %r805;
	cvt.rn.bf16.f32 	%rs494, %r192;
	cvt.rn.f32.s32 	%r193, %r806;
	cvt.rn.bf16.f32 	%rs495, %r193;
	cvt.rn.f32.s32 	%r194, %r807;
	cvt.rn.bf16.f32 	%rs496, %r194;
	cvt.rn.f32.s32 	%r195, %r808;
	cvt.rn.bf16.f32 	%rs497, %r195;
	cvt.rn.f32.s32 	%r196, %r809;
	cvt.rn.bf16.f32 	%rs498, %r196;
	cvt.rn.f32.s32 	%r197, %r810;
	cvt.rn.bf16.f32 	%rs499, %r197;
	cvt.rn.f32.s32 	%r198, %r811;
	cvt.rn.bf16.f32 	%rs500, %r198;
	cvt.rn.f32.s32 	%r199, %r812;
	cvt.rn.bf16.f32 	%rs501, %r199;
	cvt.rn.f32.s32 	%r200, %r813;
	cvt.rn.bf16.f32 	%rs502, %r200;
	cvt.rn.f32.s32 	%r201, %r814;
	cvt.rn.bf16.f32 	%rs503, %r201;
	cvt.rn.f32.s32 	%r202, %r815;
	cvt.rn.bf16.f32 	%rs504, %r202;
	cvt.rn.f32.s32 	%r203, %r816;
	cvt.rn.bf16.f32 	%rs505, %r203;
	cvt.rn.f32.s32 	%r204, %r817;
	cvt.rn.bf16.f32 	%rs506, %r204;
	cvt.rn.f32.s32 	%r205, %r818;
	cvt.rn.bf16.f32 	%rs507, %r205;
	cvt.rn.f32.s32 	%r206, %r819;
	cvt.rn.bf16.f32 	%rs508, %r206;
	cvt.rn.f32.s32 	%r207, %r820;
	cvt.rn.bf16.f32 	%rs509, %r207;
	cvt.rn.f32.s32 	%r208, %r821;
	cvt.rn.bf16.f32 	%rs510, %r208;
	cvt.rn.f32.s32 	%r209, %r822;
	cvt.rn.bf16.f32 	%rs511, %r209;
	cvt.rn.f32.s32 	%r210, %r823;
	cvt.rn.bf16.f32 	%rs512, %r210;
	cvt.rn.f32.s32 	%r211, %r824;
	cvt.rn.bf16.f32 	%rs513, %r211;
	cvt.rn.f32.s32 	%r212, %r825;
	cvt.rn.bf16.f32 	%rs514, %r212;
	cvt.rn.f32.s32 	%r213, %r826;
	cvt.rn.bf16.f32 	%rs515, %r213;
	cvt.rn.f32.s32 	%r214, %r827;
	cvt.rn.bf16.f32 	%rs516, %r214;
	cvt.rn.f32.s32 	%r215, %r828;
	cvt.rn.bf16.f32 	%rs517, %r215;
	cvt.rn.f32.s32 	%r216, %r829;
	cvt.rn.bf16.f32 	%rs518, %r216;
	cvt.rn.f32.s32 	%r217, %r830;
	cvt.rn.bf16.f32 	%rs519, %r217;
	cvt.rn.f32.s32 	%r218, %r831;
	cvt.rn.bf16.f32 	%rs520, %r218;
	cvt.rn.f32.s32 	%r219, %r832;
	cvt.rn.bf16.f32 	%rs521, %r219;
	cvt.rn.f32.s32 	%r220, %r833;
	cvt.rn.bf16.f32 	%rs522, %r220;
	cvt.rn.f32.s32 	%r221, %r834;
	cvt.rn.bf16.f32 	%rs523, %r221;
	cvt.rn.f32.s32 	%r222, %r835;
	cvt.rn.bf16.f32 	%rs524, %r222;
	cvt.rn.f32.s32 	%r223, %r836;
	cvt.rn.bf16.f32 	%rs525, %r223;
	cvt.rn.f32.s32 	%r224, %r837;
	cvt.rn.bf16.f32 	%rs526, %r224;
	cvt.rn.f32.s32 	%r225, %r838;
	cvt.rn.bf16.f32 	%rs527, %r225;
	cvt.rn.f32.s32 	%r226, %r839;
	cvt.rn.bf16.f32 	%rs528, %r226;
	cvt.rn.f32.s32 	%r227, %r840;
	cvt.rn.bf16.f32 	%rs529, %r227;
	cvt.rn.f32.s32 	%r228, %r841;
	cvt.rn.bf16.f32 	%rs530, %r228;
	cvt.rn.f32.s32 	%r229, %r842;
	cvt.rn.bf16.f32 	%rs531, %r229;
	cvt.rn.f32.s32 	%r230, %r843;
	cvt.rn.bf16.f32 	%rs532, %r230;
	cvt.rn.f32.s32 	%r231, %r844;
	cvt.rn.bf16.f32 	%rs533, %r231;
	cvt.rn.f32.s32 	%r232, %r845;
	cvt.rn.bf16.f32 	%rs534, %r232;
	cvt.rn.f32.s32 	%r233, %r846;
	cvt.rn.bf16.f32 	%rs535, %r233;
	cvt.rn.f32.s32 	%r234, %r847;
	cvt.rn.bf16.f32 	%rs536, %r234;
	cvt.rn.f32.s32 	%r235, %r848;
	cvt.rn.bf16.f32 	%rs537, %r235;
	cvt.rn.f32.s32 	%r236, %r849;
	cvt.rn.bf16.f32 	%rs538, %r236;
	cvt.rn.f32.s32 	%r237, %r850;
	cvt.rn.bf16.f32 	%rs539, %r237;
	cvt.rn.f32.s32 	%r238, %r851;
	cvt.rn.bf16.f32 	%rs540, %r238;
	cvt.rn.f32.s32 	%r239, %r852;
	cvt.rn.bf16.f32 	%rs541, %r239;
	cvt.rn.f32.s32 	%r240, %r853;
	cvt.rn.bf16.f32 	%rs542, %r240;
	cvt.rn.f32.s32 	%r241, %r854;
	cvt.rn.bf16.f32 	%rs543, %r241;
	cvt.rn.f32.s32 	%r242, %r855;
	cvt.rn.bf16.f32 	%rs544, %r242;
	cvt.rn.f32.s32 	%r243, %r856;
	cvt.rn.bf16.f32 	%rs545, %r243;
	cvt.rn.f32.s32 	%r244, %r857;
	cvt.rn.bf16.f32 	%rs546, %r244;
	cvt.rn.f32.s32 	%r245, %r858;
	cvt.rn.bf16.f32 	%rs547, %r245;
	cvt.rn.f32.s32 	%r246, %r859;
	cvt.rn.bf16.f32 	%rs548, %r246;
	cvt.rn.f32.s32 	%r247, %r860;
	cvt.rn.bf16.f32 	%rs549, %r247;
	cvt.rn.f32.s32 	%r248, %r861;
	cvt.rn.bf16.f32 	%rs550, %r248;
	cvt.rn.f32.s32 	%r249, %r862;
	cvt.rn.bf16.f32 	%rs551, %r249;
	cvt.rn.f32.s32 	%r250, %r863;
	cvt.rn.bf16.f32 	%rs552, %r250;
	cvt.rn.f32.s32 	%r251, %r864;
	cvt.rn.bf16.f32 	%rs553, %r251;
	cvt.rn.f32.s32 	%r252, %r865;
	cvt.rn.bf16.f32 	%rs554, %r252;
	cvt.rn.f32.s32 	%r253, %r866;
	cvt.rn.bf16.f32 	%rs555, %r253;
	cvt.rn.f32.s32 	%r254, %r867;
	cvt.rn.bf16.f32 	%rs556, %r254;
	cvt.rn.f32.s32 	%r255, %r868;
	cvt.rn.bf16.f32 	%rs557, %r255;
	cvt.rn.f32.s32 	%r256, %r869;
	cvt.rn.bf16.f32 	%rs558, %r256;
	cvt.rn.f32.s32 	%r257, %r870;
	cvt.rn.bf16.f32 	%rs559, %r257;
	cvt.rn.f32.s32 	%r258, %r871;
	cvt.rn.bf16.f32 	%rs560, %r258;
	cvt.rn.f32.s32 	%r259, %r872;
	cvt.rn.bf16.f32 	%rs561, %r259;
	cvt.rn.f32.s32 	%r260, %r873;
	cvt.rn.bf16.f32 	%rs562, %r260;
	cvt.rn.f32.s32 	%r261, %r874;
	cvt.rn.bf16.f32 	%rs563, %r261;
	cvt.rn.f32.s32 	%r262, %r875;
	cvt.rn.bf16.f32 	%rs564, %r262;
	cvt.rn.f32.s32 	%r263, %r876;
	cvt.rn.bf16.f32 	%rs565, %r263;
	cvt.rn.f32.s32 	%r264, %r877;
	cvt.rn.bf16.f32 	%rs566, %r264;
	cvt.rn.f32.s32 	%r265, %r878;
	cvt.rn.bf16.f32 	%rs567, %r265;
	cvt.rn.f32.s32 	%r266, %r879;
	cvt.rn.bf16.f32 	%rs568, %r266;
	cvt.rn.f32.s32 	%r267, %r880;
	cvt.rn.bf16.f32 	%rs569, %r267;
	cvt.rn.f32.s32 	%r268, %r881;
	cvt.rn.bf16.f32 	%rs570, %r268;
	cvt.rn.f32.s32 	%r269, %r882;
	cvt.rn.bf16.f32 	%rs571, %r269;
	cvt.rn.f32.s32 	%r270, %r883;
	cvt.rn.bf16.f32 	%rs572, %r270;
	cvt.rn.f32.s32 	%r271, %r884;
	cvt.rn.bf16.f32 	%rs573, %r271;
	cvt.rn.f32.s32 	%r272, %r885;
	cvt.rn.bf16.f32 	%rs574, %r272;
	cvt.rn.f32.s32 	%r273, %r886;
	cvt.rn.bf16.f32 	%rs575, %r273;
	cvt.rn.f32.s32 	%r274, %r887;
	cvt.rn.bf16.f32 	%rs576, %r274;
	cvt.rn.f32.s32 	%r275, %r888;
	cvt.rn.bf16.f32 	%rs577, %r275;
	cvt.rn.f32.s32 	%r276, %r889;
	cvt.rn.bf16.f32 	%rs578, %r276;
	cvt.rn.f32.s32 	%r277, %r890;
	cvt.rn.bf16.f32 	%rs579, %r277;
	cvt.rn.f32.s32 	%r278, %r891;
	cvt.rn.bf16.f32 	%rs580, %r278;
	cvt.rn.f32.s32 	%r279, %r892;
	cvt.rn.bf16.f32 	%rs581, %r279;
	cvt.rn.f32.s32 	%r280, %r893;
	cvt.rn.bf16.f32 	%rs582, %r280;
	cvt.rn.f32.s32 	%r281, %r894;
	cvt.rn.bf16.f32 	%rs583, %r281;
	cvt.rn.f32.s32 	%r282, %r895;
	cvt.rn.bf16.f32 	%rs584, %r282;
	cvt.rn.f32.s32 	%r283, %r896;
	cvt.rn.bf16.f32 	%rs585, %r283;
	cvt.rn.f32.s32 	%r284, %r897;
	cvt.rn.bf16.f32 	%rs586, %r284;
	cvt.rn.f32.s32 	%r285, %r898;
	cvt.rn.f32.s32 	%r286, %r899;
	cvt.rn.bf16x2.f32 	%r933, %r286, %r285;
	cvt.rn.f32.s32 	%r287, %r900;
	cvt.rn.bf16.f32 	%rs587, %r287;
	cvt.rn.f32.s32 	%r288, %r901;
	cvt.rn.bf16.f32 	%rs588, %r288;
	cvt.rn.f32.s32 	%r289, %r902;
	cvt.rn.f32.s32 	%r290, %r903;
	cvt.rn.bf16x2.f32 	%r935, %r290, %r289;
	cvt.rn.f32.s32 	%r291, %r904;
	cvt.rn.bf16.f32 	%rs589, %r291;
	cvt.rn.f32.s32 	%r292, %r905;
	cvt.rn.bf16.f32 	%rs590, %r292;
	cvt.rn.f32.s32 	%r293, %r906;
	cvt.rn.f32.s32 	%r294, %r907;
	cvt.rn.bf16x2.f32 	%r937, %r294, %r293;
	cvt.rn.f32.s32 	%r295, %r908;
	cvt.rn.bf16.f32 	%rs591, %r295;
	cvt.rn.f32.s32 	%r296, %r909;
	cvt.rn.bf16.f32 	%rs592, %r296;
	cvt.rn.f32.s32 	%r297, %r910;
	cvt.rn.f32.s32 	%r298, %r911;
	cvt.rn.bf16x2.f32 	%r939, %r298, %r297;
	cvt.rn.f32.s32 	%r299, %r912;
	cvt.rn.bf16.f32 	%rs593, %r299;
	cvt.rn.f32.s32 	%r300, %r913;
	cvt.rn.bf16.f32 	%rs594, %r300;
	cvt.rn.f32.s32 	%r301, %r914;
	cvt.rn.f32.s32 	%r302, %r915;
	cvt.rn.bf16x2.f32 	%r934, %r302, %r301;
	cvt.rn.f32.s32 	%r303, %r916;
	cvt.rn.bf16.f32 	%rs595, %r303;
	cvt.rn.f32.s32 	%r304, %r917;
	cvt.rn.bf16.f32 	%rs596, %r304;
	cvt.rn.f32.s32 	%r305, %r918;
	cvt.rn.f32.s32 	%r306, %r919;
	cvt.rn.bf16x2.f32 	%r936, %r306, %r305;
	cvt.rn.f32.s32 	%r307, %r920;
	cvt.rn.bf16.f32 	%rs597, %r307;
	cvt.rn.f32.s32 	%r308, %r921;
	cvt.rn.bf16.f32 	%rs598, %r308;
	cvt.rn.f32.s32 	%r309, %r922;
	cvt.rn.f32.s32 	%r310, %r923;
	cvt.rn.bf16x2.f32 	%r938, %r310, %r309;
	cvt.rn.f32.s32 	%r311, %r924;
	cvt.rn.bf16.f32 	%rs599, %r311;
	cvt.rn.f32.s32 	%r312, %r925;
	cvt.rn.bf16.f32 	%rs600, %r312;
	cvt.rn.f32.s32 	%r313, %r926;
	cvt.rn.f32.s32 	%r314, %r927;
	cvt.rn.bf16x2.f32 	%r940, %r314, %r313;
	cvt.rn.f32.s32 	%r315, %r928;
	cvt.rn.bf16.f32 	%rs601, %r315;
	cvt.rn.f32.s32 	%r316, %r929;
	cvt.rn.bf16.f32 	%rs602, %r316;
	bra.uni 	$L__BB0_5;
$L__BB0_1:                              // %.._crit_edge_crit_edge
	.loc	1 188 19                        // sk03_fa_qkv.py:188:19
	and.b32 	%r932, %r2, 16;
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	shl.b32 	%r931, %r2, 4;
	mov.b32 	%r933, 0;
	mov.b16 	%rs491, 0x0000;
	mov.b16 	%rs492, %rs491;
	mov.b16 	%rs493, %rs491;
	mov.b16 	%rs494, %rs491;
	mov.b16 	%rs495, %rs491;
	mov.b16 	%rs496, %rs491;
	mov.b16 	%rs497, %rs491;
	mov.b16 	%rs498, %rs491;
	mov.b16 	%rs499, %rs491;
	mov.b16 	%rs500, %rs491;
	mov.b16 	%rs501, %rs491;
	mov.b16 	%rs502, %rs491;
	mov.b16 	%rs503, %rs491;
	mov.b16 	%rs504, %rs491;
	mov.b16 	%rs505, %rs491;
	mov.b16 	%rs506, %rs491;
	mov.b16 	%rs507, %rs491;
	mov.b16 	%rs508, %rs491;
	mov.b16 	%rs509, %rs491;
	mov.b16 	%rs510, %rs491;
	mov.b16 	%rs511, %rs491;
	mov.b16 	%rs512, %rs491;
	mov.b16 	%rs513, %rs491;
	mov.b16 	%rs514, %rs491;
	mov.b16 	%rs515, %rs491;
	mov.b16 	%rs516, %rs491;
	mov.b16 	%rs517, %rs491;
	mov.b16 	%rs518, %rs491;
	mov.b16 	%rs519, %rs491;
	mov.b16 	%rs520, %rs491;
	mov.b16 	%rs521, %rs491;
	mov.b16 	%rs522, %rs491;
	mov.b16 	%rs523, %rs491;
	mov.b16 	%rs524, %rs491;
	mov.b16 	%rs525, %rs491;
	mov.b16 	%rs526, %rs491;
	mov.b16 	%rs527, %rs491;
	mov.b16 	%rs528, %rs491;
	mov.b16 	%rs529, %rs491;
	mov.b16 	%rs530, %rs491;
	mov.b16 	%rs531, %rs491;
	mov.b16 	%rs532, %rs491;
	mov.b16 	%rs533, %rs491;
	mov.b16 	%rs534, %rs491;
	mov.b16 	%rs535, %rs491;
	mov.b16 	%rs536, %rs491;
	mov.b16 	%rs537, %rs491;
	mov.b16 	%rs538, %rs491;
	mov.b16 	%rs539, %rs491;
	mov.b16 	%rs540, %rs491;
	mov.b16 	%rs541, %rs491;
	mov.b16 	%rs542, %rs491;
	mov.b16 	%rs543, %rs491;
	mov.b16 	%rs544, %rs491;
	mov.b16 	%rs545, %rs491;
	mov.b16 	%rs546, %rs491;
	mov.b16 	%rs547, %rs491;
	mov.b16 	%rs548, %rs491;
	mov.b16 	%rs549, %rs491;
	mov.b16 	%rs550, %rs491;
	mov.b16 	%rs551, %rs491;
	mov.b16 	%rs552, %rs491;
	mov.b16 	%rs553, %rs491;
	mov.b16 	%rs554, %rs491;
	mov.b16 	%rs555, %rs491;
	mov.b16 	%rs556, %rs491;
	mov.b16 	%rs557, %rs491;
	mov.b16 	%rs558, %rs491;
	mov.b16 	%rs559, %rs491;
	mov.b16 	%rs560, %rs491;
	mov.b16 	%rs561, %rs491;
	mov.b16 	%rs562, %rs491;
	mov.b16 	%rs563, %rs491;
	mov.b16 	%rs564, %rs491;
	mov.b16 	%rs565, %rs491;
	mov.b16 	%rs566, %rs491;
	mov.b16 	%rs567, %rs491;
	mov.b16 	%rs568, %rs491;
	mov.b16 	%rs569, %rs491;
	mov.b16 	%rs570, %rs491;
	mov.b16 	%rs571, %rs491;
	mov.b16 	%rs572, %rs491;
	mov.b16 	%rs573, %rs491;
	mov.b16 	%rs574, %rs491;
	mov.b16 	%rs575, %rs491;
	mov.b16 	%rs576, %rs491;
	mov.b16 	%rs577, %rs491;
	mov.b16 	%rs578, %rs491;
	mov.b16 	%rs579, %rs491;
	mov.b16 	%rs580, %rs491;
	mov.b16 	%rs581, %rs491;
	mov.b16 	%rs582, %rs491;
	mov.b16 	%rs583, %rs491;
	mov.b16 	%rs584, %rs491;
	mov.b16 	%rs585, %rs491;
	mov.b16 	%rs586, %rs491;
	mov.b16 	%rs587, %rs491;
	mov.b16 	%rs588, %rs491;
	mov.b16 	%rs589, %rs491;
	mov.b16 	%rs590, %rs491;
	mov.b16 	%rs591, %rs491;
	mov.b16 	%rs592, %rs491;
	mov.b16 	%rs593, %rs491;
	mov.b16 	%rs594, %rs491;
	mov.b16 	%rs595, %rs491;
	mov.b16 	%rs596, %rs491;
	mov.b16 	%rs597, %rs491;
	mov.b16 	%rs598, %rs491;
	mov.b16 	%rs599, %rs491;
	mov.b16 	%rs600, %rs491;
	mov.b16 	%rs601, %rs491;
	mov.b16 	%rs602, %rs491;
	mov.b32 	%r934, %r933;
	mov.b32 	%r935, %r933;
	mov.b32 	%r936, %r933;
	mov.b32 	%r937, %r933;
	mov.b32 	%r938, %r933;
	mov.b32 	%r939, %r933;
	mov.b32 	%r940, %r933;
$L__BB0_5:                              // %._crit_edge
	.loc	1 164 45                        // sk03_fa_qkv.py:164:45
	shl.b32 	%r547, %r7, 3;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r548, %r547, %r4;
	or.b32 	%r549, %r548, 7;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r550, %r549, %r16;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r551, %r548, 6;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r552, %r551, %r16;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r553, %r548, 5;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r554, %r553, %r16;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r555, %r548, 4;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r556, %r555, %r16;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r557, %r548, 3;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r558, %r557, %r16;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r559, %r548, 2;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r560, %r559, %r16;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r561, %r548, 1;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r562, %r561, %r16;
	rem.s32 	%r563, %r548, %r16;
	.loc	1 164 45                        // sk03_fa_qkv.py:164:45
	shr.u32 	%r564, %r6, 2;
	shl.b32 	%r565, %r5, 1;
	or.b32 	%r566, %r564, %r565;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r567, %r566, %r4;
	or.b32 	%r568, %r567, 112;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r569, %r568, %r16;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r570, %r567, 96;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r571, %r570, %r16;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r572, %r567, 80;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r573, %r572, %r16;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r574, %r567, 64;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r575, %r574, %r16;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r576, %r567, 48;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r577, %r576, %r16;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r578, %r567, 32;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r579, %r578, %r16;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r580, %r567, 16;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r581, %r580, %r16;
	rem.s32 	%r582, %r567, %r16;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r583, %r1, %r3;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r584, %r583, %r15;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	and.b32 	%r585, %r2, 240;
	bfe.u32 	%r586, %r2, 4, 4;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r587, %r586, %r1;
	or.b32 	%r588, %r587, 240;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r589, %r588, %r15;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r590, %r587, 224;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r591, %r590, %r15;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r592, %r587, 208;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r593, %r592, %r15;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r594, %r587, 192;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r595, %r594, %r15;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r596, %r587, 176;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r597, %r596, %r15;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r598, %r587, 160;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r599, %r598, %r15;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r600, %r587, 144;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r601, %r600, %r15;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r602, %r587, 128;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r603, %r602, %r15;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r604, %r587, 112;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r605, %r604, %r15;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r606, %r587, 96;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r607, %r606, %r15;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r608, %r587, 80;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r609, %r608, %r15;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r610, %r587, 64;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r611, %r610, %r15;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r612, %r587, 48;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r613, %r612, %r15;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r614, %r587, 32;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r615, %r614, %r15;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r616, %r587, 16;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r617, %r616, %r15;
	rem.s32 	%r618, %r587, %r15;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 186 54                        // sk03_fa_qkv.py:186:54
	mad.wide.s32 	%rd49, %r584, 4, %rd12;
	.loc	1 186 40                        // sk03_fa_qkv.py:186:40
	// begin inline asm
	mov.u32 %r317, 0x0;
	ld.global.b32 { %r317 }, [ %rd49 + 0 ];
	// end inline asm
	.loc	1 186 65                        // sk03_fa_qkv.py:186:65
	cvt.rn.bf16.f32 	%rs1, %r317;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	and.b32 	%r619, %r2, 7;
	shl.b32 	%r620, %r619, 3;
	shl.b32 	%r621, %r2, 2;
	and.b32 	%r622, %r621, 192;
	and.b32 	%r623, %r2, 8;
	shr.u32 	%r624, %r623, 1;
	shr.u32 	%r625, %r2, 5;
	and.b32 	%r626, %r625, 2;
	and.b32 	%r627, %r9, 256;
	add.s32 	%r628, %r94, %r620;
	add.s32 	%r629, %r628, %r622;
	add.s32 	%r630, %r629, %r624;
	add.s32 	%r631, %r630, %r626;
	add.s32 	%r318, %r631, %r627;
	// begin inline asm
	st.shared.b16 [ %r318 + 0 ], %rs1;
	// end inline asm
	bar.sync 	0;
	and.b32 	%r632, %r9, 56;
	and.b32 	%r633, %r2, 192;
	add.s32 	%r634, %r94, %r632;
	add.s32 	%r635, %r634, %r633;
	ld.shared.v4.b16 	{%rs130, %rs131, %rs132, %rs133}, [%r635];
	ld.shared.v4.b16 	{%rs134, %rs135, %rs136, %rs137}, [%r635+256];
	mov.b16 	%rs138, 0x8000;
	fma.rn.bf16 	%rs139, %rs130, %rs491, %rs138;
	fma.rn.bf16 	%rs140, %rs130, %rs492, %rs138;
	fma.rn.bf16 	%rs141, %rs132, %rs493, %rs138;
	fma.rn.bf16 	%rs142, %rs132, %rs494, %rs138;
	fma.rn.bf16 	%rs143, %rs130, %rs495, %rs138;
	fma.rn.bf16 	%rs144, %rs130, %rs496, %rs138;
	fma.rn.bf16 	%rs145, %rs132, %rs497, %rs138;
	fma.rn.bf16 	%rs146, %rs132, %rs498, %rs138;
	fma.rn.bf16 	%rs147, %rs130, %rs499, %rs138;
	fma.rn.bf16 	%rs148, %rs130, %rs500, %rs138;
	fma.rn.bf16 	%rs149, %rs132, %rs501, %rs138;
	fma.rn.bf16 	%rs150, %rs132, %rs502, %rs138;
	fma.rn.bf16 	%rs151, %rs130, %rs503, %rs138;
	fma.rn.bf16 	%rs152, %rs130, %rs504, %rs138;
	fma.rn.bf16 	%rs153, %rs132, %rs505, %rs138;
	fma.rn.bf16 	%rs154, %rs132, %rs506, %rs138;
	fma.rn.bf16 	%rs155, %rs130, %rs507, %rs138;
	fma.rn.bf16 	%rs156, %rs130, %rs508, %rs138;
	fma.rn.bf16 	%rs157, %rs132, %rs509, %rs138;
	fma.rn.bf16 	%rs158, %rs132, %rs510, %rs138;
	fma.rn.bf16 	%rs159, %rs130, %rs511, %rs138;
	fma.rn.bf16 	%rs160, %rs130, %rs512, %rs138;
	fma.rn.bf16 	%rs161, %rs132, %rs513, %rs138;
	fma.rn.bf16 	%rs162, %rs132, %rs514, %rs138;
	fma.rn.bf16 	%rs163, %rs130, %rs515, %rs138;
	fma.rn.bf16 	%rs164, %rs130, %rs516, %rs138;
	fma.rn.bf16 	%rs165, %rs132, %rs517, %rs138;
	fma.rn.bf16 	%rs166, %rs132, %rs518, %rs138;
	fma.rn.bf16 	%rs167, %rs130, %rs519, %rs138;
	fma.rn.bf16 	%rs168, %rs130, %rs520, %rs138;
	fma.rn.bf16 	%rs169, %rs132, %rs521, %rs138;
	fma.rn.bf16 	%rs170, %rs132, %rs522, %rs138;
	fma.rn.bf16 	%rs171, %rs131, %rs523, %rs138;
	fma.rn.bf16 	%rs172, %rs131, %rs524, %rs138;
	fma.rn.bf16 	%rs173, %rs133, %rs525, %rs138;
	fma.rn.bf16 	%rs174, %rs133, %rs526, %rs138;
	fma.rn.bf16 	%rs175, %rs131, %rs527, %rs138;
	fma.rn.bf16 	%rs176, %rs131, %rs528, %rs138;
	fma.rn.bf16 	%rs177, %rs133, %rs529, %rs138;
	fma.rn.bf16 	%rs178, %rs133, %rs530, %rs138;
	fma.rn.bf16 	%rs179, %rs131, %rs531, %rs138;
	fma.rn.bf16 	%rs180, %rs131, %rs532, %rs138;
	fma.rn.bf16 	%rs181, %rs133, %rs533, %rs138;
	fma.rn.bf16 	%rs182, %rs133, %rs534, %rs138;
	fma.rn.bf16 	%rs183, %rs131, %rs535, %rs138;
	fma.rn.bf16 	%rs184, %rs131, %rs536, %rs138;
	fma.rn.bf16 	%rs185, %rs133, %rs537, %rs138;
	fma.rn.bf16 	%rs186, %rs133, %rs538, %rs138;
	fma.rn.bf16 	%rs187, %rs131, %rs539, %rs138;
	fma.rn.bf16 	%rs188, %rs131, %rs540, %rs138;
	fma.rn.bf16 	%rs189, %rs133, %rs541, %rs138;
	fma.rn.bf16 	%rs190, %rs133, %rs542, %rs138;
	fma.rn.bf16 	%rs191, %rs131, %rs543, %rs138;
	fma.rn.bf16 	%rs192, %rs131, %rs544, %rs138;
	fma.rn.bf16 	%rs193, %rs133, %rs545, %rs138;
	fma.rn.bf16 	%rs194, %rs133, %rs546, %rs138;
	fma.rn.bf16 	%rs195, %rs131, %rs547, %rs138;
	fma.rn.bf16 	%rs196, %rs131, %rs548, %rs138;
	fma.rn.bf16 	%rs197, %rs133, %rs549, %rs138;
	fma.rn.bf16 	%rs198, %rs133, %rs550, %rs138;
	fma.rn.bf16 	%rs199, %rs131, %rs551, %rs138;
	fma.rn.bf16 	%rs200, %rs131, %rs552, %rs138;
	fma.rn.bf16 	%rs201, %rs133, %rs553, %rs138;
	fma.rn.bf16 	%rs202, %rs133, %rs554, %rs138;
	fma.rn.bf16 	%rs203, %rs134, %rs555, %rs138;
	fma.rn.bf16 	%rs204, %rs134, %rs556, %rs138;
	fma.rn.bf16 	%rs205, %rs136, %rs557, %rs138;
	fma.rn.bf16 	%rs206, %rs136, %rs558, %rs138;
	fma.rn.bf16 	%rs207, %rs134, %rs559, %rs138;
	fma.rn.bf16 	%rs208, %rs134, %rs560, %rs138;
	fma.rn.bf16 	%rs209, %rs136, %rs561, %rs138;
	fma.rn.bf16 	%rs210, %rs136, %rs562, %rs138;
	fma.rn.bf16 	%rs211, %rs134, %rs563, %rs138;
	fma.rn.bf16 	%rs212, %rs134, %rs564, %rs138;
	fma.rn.bf16 	%rs213, %rs136, %rs565, %rs138;
	fma.rn.bf16 	%rs214, %rs136, %rs566, %rs138;
	fma.rn.bf16 	%rs215, %rs134, %rs567, %rs138;
	fma.rn.bf16 	%rs216, %rs134, %rs568, %rs138;
	fma.rn.bf16 	%rs217, %rs136, %rs569, %rs138;
	fma.rn.bf16 	%rs218, %rs136, %rs570, %rs138;
	fma.rn.bf16 	%rs219, %rs134, %rs571, %rs138;
	fma.rn.bf16 	%rs220, %rs134, %rs572, %rs138;
	fma.rn.bf16 	%rs221, %rs136, %rs573, %rs138;
	fma.rn.bf16 	%rs222, %rs136, %rs574, %rs138;
	fma.rn.bf16 	%rs223, %rs134, %rs575, %rs138;
	fma.rn.bf16 	%rs224, %rs134, %rs576, %rs138;
	fma.rn.bf16 	%rs225, %rs136, %rs577, %rs138;
	fma.rn.bf16 	%rs226, %rs136, %rs578, %rs138;
	fma.rn.bf16 	%rs227, %rs134, %rs579, %rs138;
	fma.rn.bf16 	%rs228, %rs134, %rs580, %rs138;
	fma.rn.bf16 	%rs229, %rs136, %rs581, %rs138;
	fma.rn.bf16 	%rs230, %rs136, %rs582, %rs138;
	fma.rn.bf16 	%rs231, %rs134, %rs583, %rs138;
	fma.rn.bf16 	%rs232, %rs134, %rs584, %rs138;
	fma.rn.bf16 	%rs233, %rs136, %rs585, %rs138;
	fma.rn.bf16 	%rs234, %rs136, %rs586, %rs138;
	fma.rn.bf16 	%rs235, %rs137, %rs587, %rs138;
	fma.rn.bf16 	%rs236, %rs137, %rs588, %rs138;
	fma.rn.bf16 	%rs237, %rs137, %rs589, %rs138;
	fma.rn.bf16 	%rs238, %rs137, %rs590, %rs138;
	fma.rn.bf16 	%rs239, %rs137, %rs591, %rs138;
	fma.rn.bf16 	%rs240, %rs137, %rs592, %rs138;
	fma.rn.bf16 	%rs241, %rs137, %rs593, %rs138;
	fma.rn.bf16 	%rs242, %rs137, %rs594, %rs138;
	fma.rn.bf16 	%rs243, %rs137, %rs595, %rs138;
	fma.rn.bf16 	%rs244, %rs137, %rs596, %rs138;
	fma.rn.bf16 	%rs245, %rs137, %rs597, %rs138;
	fma.rn.bf16 	%rs246, %rs137, %rs598, %rs138;
	fma.rn.bf16 	%rs247, %rs137, %rs599, %rs138;
	fma.rn.bf16 	%rs248, %rs137, %rs600, %rs138;
	fma.rn.bf16 	%rs249, %rs137, %rs601, %rs138;
	fma.rn.bf16 	%rs250, %rs137, %rs602, %rs138;
	.loc	1 187 38                        // sk03_fa_qkv.py:187:38
	mad.wide.s32 	%rd50, %r582, 4, %rd13;
	mad.wide.s32 	%rd51, %r581, 4, %rd13;
	mad.wide.s32 	%rd52, %r579, 4, %rd13;
	mad.wide.s32 	%rd53, %r577, 4, %rd13;
	mad.wide.s32 	%rd54, %r575, 4, %rd13;
	mad.wide.s32 	%rd55, %r573, 4, %rd13;
	mad.wide.s32 	%rd56, %r571, 4, %rd13;
	mad.wide.s32 	%rd57, %r569, 4, %rd13;
	.loc	1 187 24                        // sk03_fa_qkv.py:187:24
	// begin inline asm
	mov.u32 %r319, 0x0;
	mov.u32 %r320, 0x0;
	ld.global.v2.b32 { %r319, %r320 }, [ %rd50 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r321, 0x0;
	mov.u32 %r322, 0x0;
	ld.global.v2.b32 { %r321, %r322 }, [ %rd51 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r323, 0x0;
	mov.u32 %r324, 0x0;
	ld.global.v2.b32 { %r323, %r324 }, [ %rd52 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r325, 0x0;
	mov.u32 %r326, 0x0;
	ld.global.v2.b32 { %r325, %r326 }, [ %rd53 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r327, 0x0;
	mov.u32 %r328, 0x0;
	ld.global.v2.b32 { %r327, %r328 }, [ %rd54 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r329, 0x0;
	mov.u32 %r330, 0x0;
	ld.global.v2.b32 { %r329, %r330 }, [ %rd55 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r331, 0x0;
	mov.u32 %r332, 0x0;
	ld.global.v2.b32 { %r331, %r332 }, [ %rd56 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r333, 0x0;
	mov.u32 %r334, 0x0;
	ld.global.v2.b32 { %r333, %r334 }, [ %rd57 + 0 ];
	// end inline asm
	.loc	1 188 49                        // sk03_fa_qkv.py:188:49
	mul.lo.s32 	%r636, %r618, %r18;
	mul.lo.s32 	%r637, %r617, %r18;
	mul.lo.s32 	%r638, %r615, %r18;
	mul.lo.s32 	%r639, %r613, %r18;
	mul.lo.s32 	%r640, %r611, %r18;
	mul.lo.s32 	%r641, %r609, %r18;
	mul.lo.s32 	%r642, %r607, %r18;
	mul.lo.s32 	%r643, %r605, %r18;
	mul.lo.s32 	%r644, %r603, %r18;
	mul.lo.s32 	%r645, %r601, %r18;
	mul.lo.s32 	%r646, %r599, %r18;
	mul.lo.s32 	%r647, %r597, %r18;
	mul.lo.s32 	%r648, %r595, %r18;
	mul.lo.s32 	%r649, %r593, %r18;
	mul.lo.s32 	%r650, %r591, %r18;
	mul.lo.s32 	%r651, %r589, %r18;
	.loc	1 188 31                        // sk03_fa_qkv.py:188:31
	mad.wide.s32 	%rd202, %r636, 2, %rd11;
	mad.wide.s32 	%rd203, %r637, 2, %rd11;
	mad.wide.s32 	%rd204, %r638, 2, %rd11;
	mad.wide.s32 	%rd205, %r639, 2, %rd11;
	mad.wide.s32 	%rd206, %r640, 2, %rd11;
	mad.wide.s32 	%rd207, %r641, 2, %rd11;
	mad.wide.s32 	%rd208, %r642, 2, %rd11;
	mad.wide.s32 	%rd209, %r643, 2, %rd11;
	mad.wide.s32 	%rd210, %r644, 2, %rd11;
	mad.wide.s32 	%rd211, %r645, 2, %rd11;
	mad.wide.s32 	%rd212, %r646, 2, %rd11;
	mad.wide.s32 	%rd213, %r647, 2, %rd11;
	mad.wide.s32 	%rd214, %r648, 2, %rd11;
	mad.wide.s32 	%rd215, %r649, 2, %rd11;
	mad.wide.s32 	%rd216, %r650, 2, %rd11;
	mad.wide.s32 	%rd217, %r651, 2, %rd11;
	.loc	1 188 82                        // sk03_fa_qkv.py:188:82
	mul.lo.s32 	%r652, %r563, %r19;
	mul.lo.s32 	%r653, %r562, %r19;
	mul.lo.s32 	%r654, %r560, %r19;
	mul.lo.s32 	%r655, %r558, %r19;
	mul.lo.s32 	%r656, %r556, %r19;
	mul.lo.s32 	%r657, %r554, %r19;
	mul.lo.s32 	%r658, %r552, %r19;
	mul.lo.s32 	%r659, %r550, %r19;
	.loc	1 188 64                        // sk03_fa_qkv.py:188:64
	mul.wide.s32 	%rd218, %r652, 2;
	add.s64 	%rd58, %rd202, %rd218;
	mul.wide.s32 	%rd219, %r653, 2;
	add.s64 	%rd59, %rd202, %rd219;
	mul.wide.s32 	%rd220, %r654, 2;
	add.s64 	%rd60, %rd202, %rd220;
	mul.wide.s32 	%rd221, %r655, 2;
	add.s64 	%rd61, %rd202, %rd221;
	mul.wide.s32 	%rd222, %r656, 2;
	add.s64 	%rd62, %rd202, %rd222;
	mul.wide.s32 	%rd223, %r657, 2;
	add.s64 	%rd63, %rd202, %rd223;
	mul.wide.s32 	%rd224, %r658, 2;
	add.s64 	%rd64, %rd202, %rd224;
	mul.wide.s32 	%rd225, %r659, 2;
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
	.loc	1 188 19                        // sk03_fa_qkv.py:188:19
	// begin inline asm
	mov.u16 %rs2, 0x0;
	ld.global.b16 { %rs2 }, [ %rd58 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs3, 0x0;
	ld.global.b16 { %rs3 }, [ %rd59 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs4, 0x0;
	ld.global.b16 { %rs4 }, [ %rd60 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs5, 0x0;
	ld.global.b16 { %rs5 }, [ %rd61 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs6, 0x0;
	ld.global.b16 { %rs6 }, [ %rd62 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs7, 0x0;
	ld.global.b16 { %rs7 }, [ %rd63 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs8, 0x0;
	ld.global.b16 { %rs8 }, [ %rd64 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs9, 0x0;
	ld.global.b16 { %rs9 }, [ %rd65 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs10, 0x0;
	ld.global.b16 { %rs10 }, [ %rd66 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs11, 0x0;
	ld.global.b16 { %rs11 }, [ %rd67 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs12, 0x0;
	ld.global.b16 { %rs12 }, [ %rd68 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs13, 0x0;
	ld.global.b16 { %rs13 }, [ %rd69 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs14, 0x0;
	ld.global.b16 { %rs14 }, [ %rd70 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs15, 0x0;
	ld.global.b16 { %rs15 }, [ %rd71 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs16, 0x0;
	ld.global.b16 { %rs16 }, [ %rd72 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs17, 0x0;
	ld.global.b16 { %rs17 }, [ %rd73 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs18, 0x0;
	ld.global.b16 { %rs18 }, [ %rd74 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs19, 0x0;
	ld.global.b16 { %rs19 }, [ %rd75 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs20, 0x0;
	ld.global.b16 { %rs20 }, [ %rd76 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs21, 0x0;
	ld.global.b16 { %rs21 }, [ %rd77 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs22, 0x0;
	ld.global.b16 { %rs22 }, [ %rd78 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs23, 0x0;
	ld.global.b16 { %rs23 }, [ %rd79 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs24, 0x0;
	ld.global.b16 { %rs24 }, [ %rd80 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs25, 0x0;
	ld.global.b16 { %rs25 }, [ %rd81 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs26, 0x0;
	ld.global.b16 { %rs26 }, [ %rd82 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs27, 0x0;
	ld.global.b16 { %rs27 }, [ %rd83 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs28, 0x0;
	ld.global.b16 { %rs28 }, [ %rd84 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs29, 0x0;
	ld.global.b16 { %rs29 }, [ %rd85 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs30, 0x0;
	ld.global.b16 { %rs30 }, [ %rd86 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs31, 0x0;
	ld.global.b16 { %rs31 }, [ %rd87 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs32, 0x0;
	ld.global.b16 { %rs32 }, [ %rd88 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs33, 0x0;
	ld.global.b16 { %rs33 }, [ %rd89 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs34, 0x0;
	ld.global.b16 { %rs34 }, [ %rd90 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs35, 0x0;
	ld.global.b16 { %rs35 }, [ %rd91 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs36, 0x0;
	ld.global.b16 { %rs36 }, [ %rd92 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs37, 0x0;
	ld.global.b16 { %rs37 }, [ %rd93 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs38, 0x0;
	ld.global.b16 { %rs38 }, [ %rd94 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs39, 0x0;
	ld.global.b16 { %rs39 }, [ %rd95 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs40, 0x0;
	ld.global.b16 { %rs40 }, [ %rd96 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs41, 0x0;
	ld.global.b16 { %rs41 }, [ %rd97 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs42, 0x0;
	ld.global.b16 { %rs42 }, [ %rd98 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs43, 0x0;
	ld.global.b16 { %rs43 }, [ %rd99 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs44, 0x0;
	ld.global.b16 { %rs44 }, [ %rd100 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs45, 0x0;
	ld.global.b16 { %rs45 }, [ %rd101 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs46, 0x0;
	ld.global.b16 { %rs46 }, [ %rd102 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs47, 0x0;
	ld.global.b16 { %rs47 }, [ %rd103 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs48, 0x0;
	ld.global.b16 { %rs48 }, [ %rd104 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs49, 0x0;
	ld.global.b16 { %rs49 }, [ %rd105 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs50, 0x0;
	ld.global.b16 { %rs50 }, [ %rd106 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs51, 0x0;
	ld.global.b16 { %rs51 }, [ %rd107 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs52, 0x0;
	ld.global.b16 { %rs52 }, [ %rd108 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs53, 0x0;
	ld.global.b16 { %rs53 }, [ %rd109 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs54, 0x0;
	ld.global.b16 { %rs54 }, [ %rd110 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs55, 0x0;
	ld.global.b16 { %rs55 }, [ %rd111 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs56, 0x0;
	ld.global.b16 { %rs56 }, [ %rd112 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs57, 0x0;
	ld.global.b16 { %rs57 }, [ %rd113 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs58, 0x0;
	ld.global.b16 { %rs58 }, [ %rd114 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs59, 0x0;
	ld.global.b16 { %rs59 }, [ %rd115 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs60, 0x0;
	ld.global.b16 { %rs60 }, [ %rd116 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs61, 0x0;
	ld.global.b16 { %rs61 }, [ %rd117 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs62, 0x0;
	ld.global.b16 { %rs62 }, [ %rd118 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs63, 0x0;
	ld.global.b16 { %rs63 }, [ %rd119 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs64, 0x0;
	ld.global.b16 { %rs64 }, [ %rd120 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs65, 0x0;
	ld.global.b16 { %rs65 }, [ %rd121 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs66, 0x0;
	ld.global.b16 { %rs66 }, [ %rd122 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs67, 0x0;
	ld.global.b16 { %rs67 }, [ %rd123 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs68, 0x0;
	ld.global.b16 { %rs68 }, [ %rd124 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs69, 0x0;
	ld.global.b16 { %rs69 }, [ %rd125 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs70, 0x0;
	ld.global.b16 { %rs70 }, [ %rd126 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs71, 0x0;
	ld.global.b16 { %rs71 }, [ %rd127 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs72, 0x0;
	ld.global.b16 { %rs72 }, [ %rd128 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs73, 0x0;
	ld.global.b16 { %rs73 }, [ %rd129 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs74, 0x0;
	ld.global.b16 { %rs74 }, [ %rd130 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs75, 0x0;
	ld.global.b16 { %rs75 }, [ %rd131 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs76, 0x0;
	ld.global.b16 { %rs76 }, [ %rd132 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs77, 0x0;
	ld.global.b16 { %rs77 }, [ %rd133 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs78, 0x0;
	ld.global.b16 { %rs78 }, [ %rd134 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs79, 0x0;
	ld.global.b16 { %rs79 }, [ %rd135 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs80, 0x0;
	ld.global.b16 { %rs80 }, [ %rd136 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs81, 0x0;
	ld.global.b16 { %rs81 }, [ %rd137 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs82, 0x0;
	ld.global.b16 { %rs82 }, [ %rd138 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs83, 0x0;
	ld.global.b16 { %rs83 }, [ %rd139 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs84, 0x0;
	ld.global.b16 { %rs84 }, [ %rd140 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs85, 0x0;
	ld.global.b16 { %rs85 }, [ %rd141 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs86, 0x0;
	ld.global.b16 { %rs86 }, [ %rd142 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs87, 0x0;
	ld.global.b16 { %rs87 }, [ %rd143 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs88, 0x0;
	ld.global.b16 { %rs88 }, [ %rd144 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs89, 0x0;
	ld.global.b16 { %rs89 }, [ %rd145 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs90, 0x0;
	ld.global.b16 { %rs90 }, [ %rd146 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs91, 0x0;
	ld.global.b16 { %rs91 }, [ %rd147 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs92, 0x0;
	ld.global.b16 { %rs92 }, [ %rd148 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs93, 0x0;
	ld.global.b16 { %rs93 }, [ %rd149 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs94, 0x0;
	ld.global.b16 { %rs94 }, [ %rd150 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs95, 0x0;
	ld.global.b16 { %rs95 }, [ %rd151 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs96, 0x0;
	ld.global.b16 { %rs96 }, [ %rd152 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs97, 0x0;
	ld.global.b16 { %rs97 }, [ %rd153 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs98, 0x0;
	ld.global.b16 { %rs98 }, [ %rd154 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs99, 0x0;
	ld.global.b16 { %rs99 }, [ %rd155 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs100, 0x0;
	ld.global.b16 { %rs100 }, [ %rd156 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs101, 0x0;
	ld.global.b16 { %rs101 }, [ %rd157 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs102, 0x0;
	ld.global.b16 { %rs102 }, [ %rd158 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs103, 0x0;
	ld.global.b16 { %rs103 }, [ %rd159 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs104, 0x0;
	ld.global.b16 { %rs104 }, [ %rd160 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs105, 0x0;
	ld.global.b16 { %rs105 }, [ %rd161 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs106, 0x0;
	ld.global.b16 { %rs106 }, [ %rd162 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs107, 0x0;
	ld.global.b16 { %rs107 }, [ %rd163 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs108, 0x0;
	ld.global.b16 { %rs108 }, [ %rd164 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs109, 0x0;
	ld.global.b16 { %rs109 }, [ %rd165 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs110, 0x0;
	ld.global.b16 { %rs110 }, [ %rd166 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs111, 0x0;
	ld.global.b16 { %rs111 }, [ %rd167 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs112, 0x0;
	ld.global.b16 { %rs112 }, [ %rd168 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs113, 0x0;
	ld.global.b16 { %rs113 }, [ %rd169 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs114, 0x0;
	ld.global.b16 { %rs114 }, [ %rd170 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs115, 0x0;
	ld.global.b16 { %rs115 }, [ %rd171 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs116, 0x0;
	ld.global.b16 { %rs116 }, [ %rd172 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs117, 0x0;
	ld.global.b16 { %rs117 }, [ %rd173 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs118, 0x0;
	ld.global.b16 { %rs118 }, [ %rd174 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs119, 0x0;
	ld.global.b16 { %rs119 }, [ %rd175 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs120, 0x0;
	ld.global.b16 { %rs120 }, [ %rd176 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs121, 0x0;
	ld.global.b16 { %rs121 }, [ %rd177 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs122, 0x0;
	ld.global.b16 { %rs122 }, [ %rd178 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs123, 0x0;
	ld.global.b16 { %rs123 }, [ %rd179 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs124, 0x0;
	ld.global.b16 { %rs124 }, [ %rd180 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs125, 0x0;
	ld.global.b16 { %rs125 }, [ %rd181 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs126, 0x0;
	ld.global.b16 { %rs126 }, [ %rd182 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs127, 0x0;
	ld.global.b16 { %rs127 }, [ %rd183 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs128, 0x0;
	ld.global.b16 { %rs128 }, [ %rd184 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs129, 0x0;
	ld.global.b16 { %rs129 }, [ %rd185 + 0 ];
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r660, %r2, 7;
	and.b32 	%r661, %r660, 15360;
	shl.b32 	%r662, %r619, 4;
	or.b32 	%r663, %r661, %r662;
	xor.b32 	%r664, %r663, %r585;
	add.s32 	%r335, %r94, %r664;
	mov.b32 	%r336, {%rs2, %rs3};
	mov.b32 	%r337, {%rs4, %rs5};
	mov.b32 	%r338, {%rs6, %rs7};
	mov.b32 	%r339, {%rs8, %rs9};
	// begin inline asm
	st.shared.v4.b32 [ %r335 + 0 ], { %r336, %r337, %r338, %r339 };
	// end inline asm
	add.s32 	%r340, %r335, 256;
	mov.b32 	%r341, {%rs10, %rs11};
	mov.b32 	%r342, {%rs12, %rs13};
	mov.b32 	%r343, {%rs14, %rs15};
	mov.b32 	%r344, {%rs16, %rs17};
	// begin inline asm
	st.shared.v4.b32 [ %r340 + 0 ], { %r341, %r342, %r343, %r344 };
	// end inline asm
	add.s32 	%r345, %r335, 512;
	mov.b32 	%r346, {%rs18, %rs19};
	mov.b32 	%r347, {%rs20, %rs21};
	mov.b32 	%r348, {%rs22, %rs23};
	mov.b32 	%r349, {%rs24, %rs25};
	// begin inline asm
	st.shared.v4.b32 [ %r345 + 0 ], { %r346, %r347, %r348, %r349 };
	// end inline asm
	add.s32 	%r350, %r335, 768;
	mov.b32 	%r351, {%rs26, %rs27};
	mov.b32 	%r352, {%rs28, %rs29};
	mov.b32 	%r353, {%rs30, %rs31};
	mov.b32 	%r354, {%rs32, %rs33};
	// begin inline asm
	st.shared.v4.b32 [ %r350 + 0 ], { %r351, %r352, %r353, %r354 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r665, %r619, 11;
	shl.b32 	%r666, %r7, 4;
	shl.b32 	%r667, %r633, 2;
	setp.eq.b32 	%p24, %r932, 0;
	shl.b32 	%r668, %r932, 1;
	shr.u32 	%r669, %r6, 1;
	or.b32 	%r670, %r666, %r667;
	or.b32 	%r671, %r668, %r669;
	xor.b32 	%r672, %r670, %r671;
	or.b32 	%r673, %r672, %r665;
	add.s32 	%r674, %r94, %r673;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r675, %r676, %r677, %r678}, [%r674];
	mov.b32 	{%rs251, %rs252}, %r675;
	mov.b32 	{%rs253, %rs254}, %r676;
	mov.b32 	{%rs255, %rs256}, %r677;
	mov.b32 	{%rs257, %rs258}, %r678;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r679, %r680, %r681, %r682}, [%r674+1024];
	mov.b32 	{%rs259, %rs260}, %r679;
	mov.b32 	{%rs261, %rs262}, %r680;
	mov.b32 	{%rs263, %rs264}, %r681;
	mov.b32 	{%rs265, %rs266}, %r682;
	xor.b32 	%r683, %r673, 64;
	add.s32 	%r684, %r94, %r683;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r685, %r686, %r687, %r688}, [%r684];
	mov.b32 	{%rs267, %rs268}, %r685;
	mov.b32 	{%rs269, %rs270}, %r686;
	mov.b32 	{%rs271, %rs272}, %r687;
	mov.b32 	{%rs273, %rs274}, %r688;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r689, %r690, %r691, %r692}, [%r684+1024];
	mov.b32 	{%rs275, %rs276}, %r689;
	mov.b32 	{%rs277, %rs278}, %r690;
	mov.b32 	{%rs279, %rs280}, %r691;
	mov.b32 	{%rs281, %rs282}, %r692;
	bar.sync 	0;
	mov.b32 	%r355, {%rs34, %rs35};
	mov.b32 	%r356, {%rs36, %rs37};
	mov.b32 	%r357, {%rs38, %rs39};
	mov.b32 	%r358, {%rs40, %rs41};
	// begin inline asm
	st.shared.v4.b32 [ %r335 + 0 ], { %r355, %r356, %r357, %r358 };
	// end inline asm
	mov.b32 	%r359, {%rs42, %rs43};
	mov.b32 	%r360, {%rs44, %rs45};
	mov.b32 	%r361, {%rs46, %rs47};
	mov.b32 	%r362, {%rs48, %rs49};
	// begin inline asm
	st.shared.v4.b32 [ %r340 + 0 ], { %r359, %r360, %r361, %r362 };
	// end inline asm
	mov.b32 	%r363, {%rs50, %rs51};
	mov.b32 	%r364, {%rs52, %rs53};
	mov.b32 	%r365, {%rs54, %rs55};
	mov.b32 	%r366, {%rs56, %rs57};
	// begin inline asm
	st.shared.v4.b32 [ %r345 + 0 ], { %r363, %r364, %r365, %r366 };
	// end inline asm
	mov.b32 	%r367, {%rs58, %rs59};
	mov.b32 	%r368, {%rs60, %rs61};
	mov.b32 	%r369, {%rs62, %rs63};
	mov.b32 	%r370, {%rs64, %rs65};
	// begin inline asm
	st.shared.v4.b32 [ %r350 + 0 ], { %r367, %r368, %r369, %r370 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r693, %r694, %r695, %r696}, [%r674];
	mov.b32 	{%rs283, %rs284}, %r693;
	mov.b32 	{%rs285, %rs286}, %r694;
	mov.b32 	{%rs287, %rs288}, %r695;
	mov.b32 	{%rs289, %rs290}, %r696;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r697, %r698, %r699, %r700}, [%r674+1024];
	mov.b32 	{%rs291, %rs292}, %r697;
	mov.b32 	{%rs293, %rs294}, %r698;
	mov.b32 	{%rs295, %rs296}, %r699;
	mov.b32 	{%rs297, %rs298}, %r700;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r701, %r702, %r703, %r704}, [%r684];
	mov.b32 	{%rs299, %rs300}, %r701;
	mov.b32 	{%rs301, %rs302}, %r702;
	mov.b32 	{%rs303, %rs304}, %r703;
	mov.b32 	{%rs305, %rs306}, %r704;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r705, %r706, %r707, %r708}, [%r684+1024];
	mov.b32 	{%rs307, %rs308}, %r705;
	mov.b32 	{%rs309, %rs310}, %r706;
	mov.b32 	{%rs311, %rs312}, %r707;
	mov.b32 	{%rs313, %rs314}, %r708;
	bar.sync 	0;
	mov.b32 	%r371, {%rs66, %rs67};
	mov.b32 	%r372, {%rs68, %rs69};
	mov.b32 	%r373, {%rs70, %rs71};
	mov.b32 	%r374, {%rs72, %rs73};
	// begin inline asm
	st.shared.v4.b32 [ %r335 + 0 ], { %r371, %r372, %r373, %r374 };
	// end inline asm
	mov.b32 	%r375, {%rs74, %rs75};
	mov.b32 	%r376, {%rs76, %rs77};
	mov.b32 	%r377, {%rs78, %rs79};
	mov.b32 	%r378, {%rs80, %rs81};
	// begin inline asm
	st.shared.v4.b32 [ %r340 + 0 ], { %r375, %r376, %r377, %r378 };
	// end inline asm
	mov.b32 	%r379, {%rs82, %rs83};
	mov.b32 	%r380, {%rs84, %rs85};
	mov.b32 	%r381, {%rs86, %rs87};
	mov.b32 	%r382, {%rs88, %rs89};
	// begin inline asm
	st.shared.v4.b32 [ %r345 + 0 ], { %r379, %r380, %r381, %r382 };
	// end inline asm
	mov.b32 	%r383, {%rs90, %rs91};
	mov.b32 	%r384, {%rs92, %rs93};
	mov.b32 	%r385, {%rs94, %rs95};
	mov.b32 	%r386, {%rs96, %rs97};
	// begin inline asm
	st.shared.v4.b32 [ %r350 + 0 ], { %r383, %r384, %r385, %r386 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r709, %r710, %r711, %r712}, [%r674];
	mov.b32 	{%rs315, %rs316}, %r709;
	mov.b32 	{%rs317, %rs318}, %r710;
	mov.b32 	{%rs319, %rs320}, %r711;
	mov.b32 	{%rs321, %rs322}, %r712;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r713, %r714, %r715, %r716}, [%r674+1024];
	mov.b32 	{%rs323, %rs324}, %r713;
	mov.b32 	{%rs325, %rs326}, %r714;
	mov.b32 	{%rs327, %rs328}, %r715;
	mov.b32 	{%rs329, %rs330}, %r716;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r717, %r718, %r719, %r720}, [%r684];
	mov.b32 	{%rs331, %rs332}, %r717;
	mov.b32 	{%rs333, %rs334}, %r718;
	mov.b32 	{%rs335, %rs336}, %r719;
	mov.b32 	{%rs337, %rs338}, %r720;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r721, %r722, %r723, %r724}, [%r684+1024];
	mov.b32 	{%rs339, %rs340}, %r721;
	mov.b32 	{%rs341, %rs342}, %r722;
	mov.b32 	{%rs343, %rs344}, %r723;
	mov.b32 	{%rs345, %rs346}, %r724;
	bar.sync 	0;
	mov.b32 	%r387, {%rs98, %rs99};
	mov.b32 	%r388, {%rs100, %rs101};
	mov.b32 	%r389, {%rs102, %rs103};
	mov.b32 	%r390, {%rs104, %rs105};
	// begin inline asm
	st.shared.v4.b32 [ %r335 + 0 ], { %r387, %r388, %r389, %r390 };
	// end inline asm
	mov.b32 	%r391, {%rs106, %rs107};
	mov.b32 	%r392, {%rs108, %rs109};
	mov.b32 	%r393, {%rs110, %rs111};
	mov.b32 	%r394, {%rs112, %rs113};
	// begin inline asm
	st.shared.v4.b32 [ %r340 + 0 ], { %r391, %r392, %r393, %r394 };
	// end inline asm
	mov.b32 	%r395, {%rs114, %rs115};
	mov.b32 	%r396, {%rs116, %rs117};
	mov.b32 	%r397, {%rs118, %rs119};
	mov.b32 	%r398, {%rs120, %rs121};
	// begin inline asm
	st.shared.v4.b32 [ %r345 + 0 ], { %r395, %r396, %r397, %r398 };
	// end inline asm
	mov.b32 	%r399, {%rs122, %rs123};
	mov.b32 	%r400, {%rs124, %rs125};
	mov.b32 	%r401, {%rs126, %rs127};
	mov.b32 	%r402, {%rs128, %rs129};
	// begin inline asm
	st.shared.v4.b32 [ %r350 + 0 ], { %r399, %r400, %r401, %r402 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r725, %r726, %r727, %r728}, [%r674];
	mov.b32 	{%rs347, %rs348}, %r726;
	mov.b32 	{%rs349, %rs350}, %r728;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r729, %r730, %r731, %r732}, [%r674+1024];
	mov.b32 	{%rs351, %rs352}, %r730;
	mov.b32 	{%rs353, %rs354}, %r732;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r733, %r734, %r735, %r736}, [%r684];
	mov.b32 	{%rs355, %rs356}, %r734;
	mov.b32 	{%rs357, %rs358}, %r736;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r737, %r738, %r739, %r740}, [%r684+1024];
	mov.b32 	{%rs359, %rs360}, %r738;
	mov.b32 	{%rs361, %rs362}, %r740;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	mov.b32 	%r741, {%rs135, %rs135};
	mov.b32 	%r742, -2147450880;
	fma.rn.bf16x2 	%r743, %r741, %r933, %r742;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs363, %r320;
	cvt.rn.bf16.f32 	%rs364, %r319;
	mov.b32 	%r744, {%rs364, %rs363};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs365, %rs139, %rs364, %rs251;
	fma.rn.bf16 	%rs366, %rs140, %rs363, %rs252;
	fma.rn.bf16 	%rs367, %rs141, %rs364, %rs253;
	fma.rn.bf16 	%rs368, %rs142, %rs363, %rs254;
	fma.rn.bf16 	%rs369, %rs171, %rs364, %rs283;
	fma.rn.bf16 	%rs370, %rs172, %rs363, %rs284;
	fma.rn.bf16 	%rs371, %rs173, %rs364, %rs285;
	fma.rn.bf16 	%rs372, %rs174, %rs363, %rs286;
	fma.rn.bf16 	%rs373, %rs203, %rs364, %rs315;
	fma.rn.bf16 	%rs374, %rs204, %rs363, %rs316;
	fma.rn.bf16 	%rs375, %rs205, %rs364, %rs317;
	fma.rn.bf16 	%rs376, %rs206, %rs363, %rs318;
	fma.rn.bf16x2 	%r407, %r743, %r744, %r725;
	fma.rn.bf16 	%rs377, %rs235, %rs364, %rs347;
	fma.rn.bf16 	%rs378, %rs236, %rs363, %rs348;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r745, %r741, %r935, %r742;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs379, %r322;
	cvt.rn.bf16.f32 	%rs380, %r321;
	mov.b32 	%r746, {%rs380, %rs379};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs381, %rs143, %rs380, %rs255;
	fma.rn.bf16 	%rs382, %rs144, %rs379, %rs256;
	fma.rn.bf16 	%rs383, %rs145, %rs380, %rs257;
	fma.rn.bf16 	%rs384, %rs146, %rs379, %rs258;
	fma.rn.bf16 	%rs385, %rs175, %rs380, %rs287;
	fma.rn.bf16 	%rs386, %rs176, %rs379, %rs288;
	fma.rn.bf16 	%rs387, %rs177, %rs380, %rs289;
	fma.rn.bf16 	%rs388, %rs178, %rs379, %rs290;
	fma.rn.bf16 	%rs389, %rs207, %rs380, %rs319;
	fma.rn.bf16 	%rs390, %rs208, %rs379, %rs320;
	fma.rn.bf16 	%rs391, %rs209, %rs380, %rs321;
	fma.rn.bf16 	%rs392, %rs210, %rs379, %rs322;
	fma.rn.bf16x2 	%r427, %r745, %r746, %r727;
	fma.rn.bf16 	%rs393, %rs237, %rs380, %rs349;
	fma.rn.bf16 	%rs394, %rs238, %rs379, %rs350;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r747, %r741, %r937, %r742;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs395, %r324;
	cvt.rn.bf16.f32 	%rs396, %r323;
	mov.b32 	%r748, {%rs396, %rs395};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs397, %rs147, %rs396, %rs267;
	fma.rn.bf16 	%rs398, %rs148, %rs395, %rs268;
	fma.rn.bf16 	%rs399, %rs149, %rs396, %rs269;
	fma.rn.bf16 	%rs400, %rs150, %rs395, %rs270;
	fma.rn.bf16 	%rs401, %rs179, %rs396, %rs299;
	fma.rn.bf16 	%rs402, %rs180, %rs395, %rs300;
	fma.rn.bf16 	%rs403, %rs181, %rs396, %rs301;
	fma.rn.bf16 	%rs404, %rs182, %rs395, %rs302;
	fma.rn.bf16 	%rs405, %rs211, %rs396, %rs331;
	fma.rn.bf16 	%rs406, %rs212, %rs395, %rs332;
	fma.rn.bf16 	%rs407, %rs213, %rs396, %rs333;
	fma.rn.bf16 	%rs408, %rs214, %rs395, %rs334;
	fma.rn.bf16x2 	%r447, %r747, %r748, %r733;
	fma.rn.bf16 	%rs409, %rs239, %rs396, %rs355;
	fma.rn.bf16 	%rs410, %rs240, %rs395, %rs356;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r749, %r741, %r939, %r742;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs411, %r326;
	cvt.rn.bf16.f32 	%rs412, %r325;
	mov.b32 	%r750, {%rs412, %rs411};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs413, %rs151, %rs412, %rs271;
	fma.rn.bf16 	%rs414, %rs152, %rs411, %rs272;
	fma.rn.bf16 	%rs415, %rs153, %rs412, %rs273;
	fma.rn.bf16 	%rs416, %rs154, %rs411, %rs274;
	fma.rn.bf16 	%rs417, %rs183, %rs412, %rs303;
	fma.rn.bf16 	%rs418, %rs184, %rs411, %rs304;
	fma.rn.bf16 	%rs419, %rs185, %rs412, %rs305;
	fma.rn.bf16 	%rs420, %rs186, %rs411, %rs306;
	fma.rn.bf16 	%rs421, %rs215, %rs412, %rs335;
	fma.rn.bf16 	%rs422, %rs216, %rs411, %rs336;
	fma.rn.bf16 	%rs423, %rs217, %rs412, %rs337;
	fma.rn.bf16 	%rs424, %rs218, %rs411, %rs338;
	fma.rn.bf16x2 	%r467, %r749, %r750, %r735;
	fma.rn.bf16 	%rs425, %rs241, %rs412, %rs357;
	fma.rn.bf16 	%rs426, %rs242, %rs411, %rs358;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r751, %r741, %r934, %r742;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs427, %r328;
	cvt.rn.bf16.f32 	%rs428, %r327;
	mov.b32 	%r752, {%rs428, %rs427};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs429, %rs155, %rs428, %rs259;
	fma.rn.bf16 	%rs430, %rs156, %rs427, %rs260;
	fma.rn.bf16 	%rs431, %rs157, %rs428, %rs261;
	fma.rn.bf16 	%rs432, %rs158, %rs427, %rs262;
	fma.rn.bf16 	%rs433, %rs187, %rs428, %rs291;
	fma.rn.bf16 	%rs434, %rs188, %rs427, %rs292;
	fma.rn.bf16 	%rs435, %rs189, %rs428, %rs293;
	fma.rn.bf16 	%rs436, %rs190, %rs427, %rs294;
	fma.rn.bf16 	%rs437, %rs219, %rs428, %rs323;
	fma.rn.bf16 	%rs438, %rs220, %rs427, %rs324;
	fma.rn.bf16 	%rs439, %rs221, %rs428, %rs325;
	fma.rn.bf16 	%rs440, %rs222, %rs427, %rs326;
	fma.rn.bf16x2 	%r417, %r751, %r752, %r729;
	fma.rn.bf16 	%rs441, %rs243, %rs428, %rs351;
	fma.rn.bf16 	%rs442, %rs244, %rs427, %rs352;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r753, %r741, %r936, %r742;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs443, %r330;
	cvt.rn.bf16.f32 	%rs444, %r329;
	mov.b32 	%r754, {%rs444, %rs443};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs445, %rs159, %rs444, %rs263;
	fma.rn.bf16 	%rs446, %rs160, %rs443, %rs264;
	fma.rn.bf16 	%rs447, %rs161, %rs444, %rs265;
	fma.rn.bf16 	%rs448, %rs162, %rs443, %rs266;
	fma.rn.bf16 	%rs449, %rs191, %rs444, %rs295;
	fma.rn.bf16 	%rs450, %rs192, %rs443, %rs296;
	fma.rn.bf16 	%rs451, %rs193, %rs444, %rs297;
	fma.rn.bf16 	%rs452, %rs194, %rs443, %rs298;
	fma.rn.bf16 	%rs453, %rs223, %rs444, %rs327;
	fma.rn.bf16 	%rs454, %rs224, %rs443, %rs328;
	fma.rn.bf16 	%rs455, %rs225, %rs444, %rs329;
	fma.rn.bf16 	%rs456, %rs226, %rs443, %rs330;
	fma.rn.bf16x2 	%r437, %r753, %r754, %r731;
	fma.rn.bf16 	%rs457, %rs245, %rs444, %rs353;
	fma.rn.bf16 	%rs458, %rs246, %rs443, %rs354;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r755, %r741, %r938, %r742;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs459, %r332;
	cvt.rn.bf16.f32 	%rs460, %r331;
	mov.b32 	%r756, {%rs460, %rs459};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs461, %rs163, %rs460, %rs275;
	fma.rn.bf16 	%rs462, %rs164, %rs459, %rs276;
	fma.rn.bf16 	%rs463, %rs165, %rs460, %rs277;
	fma.rn.bf16 	%rs464, %rs166, %rs459, %rs278;
	fma.rn.bf16 	%rs465, %rs195, %rs460, %rs307;
	fma.rn.bf16 	%rs466, %rs196, %rs459, %rs308;
	fma.rn.bf16 	%rs467, %rs197, %rs460, %rs309;
	fma.rn.bf16 	%rs468, %rs198, %rs459, %rs310;
	fma.rn.bf16 	%rs469, %rs227, %rs460, %rs339;
	fma.rn.bf16 	%rs470, %rs228, %rs459, %rs340;
	fma.rn.bf16 	%rs471, %rs229, %rs460, %rs341;
	fma.rn.bf16 	%rs472, %rs230, %rs459, %rs342;
	fma.rn.bf16x2 	%r457, %r755, %r756, %r737;
	fma.rn.bf16 	%rs473, %rs247, %rs460, %rs359;
	fma.rn.bf16 	%rs474, %rs248, %rs459, %rs360;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r757, %r741, %r940, %r742;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs475, %r334;
	cvt.rn.bf16.f32 	%rs476, %r333;
	mov.b32 	%r758, {%rs476, %rs475};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs477, %rs167, %rs476, %rs279;
	fma.rn.bf16 	%rs478, %rs168, %rs475, %rs280;
	fma.rn.bf16 	%rs479, %rs169, %rs476, %rs281;
	fma.rn.bf16 	%rs480, %rs170, %rs475, %rs282;
	fma.rn.bf16 	%rs481, %rs199, %rs476, %rs311;
	fma.rn.bf16 	%rs482, %rs200, %rs475, %rs312;
	fma.rn.bf16 	%rs483, %rs201, %rs476, %rs313;
	fma.rn.bf16 	%rs484, %rs202, %rs475, %rs314;
	fma.rn.bf16 	%rs485, %rs231, %rs476, %rs343;
	fma.rn.bf16 	%rs486, %rs232, %rs475, %rs344;
	fma.rn.bf16 	%rs487, %rs233, %rs476, %rs345;
	fma.rn.bf16 	%rs488, %rs234, %rs475, %rs346;
	fma.rn.bf16x2 	%r477, %r757, %r758, %r739;
	fma.rn.bf16 	%rs489, %rs249, %rs476, %rs361;
	fma.rn.bf16 	%rs490, %rs250, %rs475, %rs362;
	bar.sync 	0;
	shl.b32 	%r759, %r5, 14;
	shl.b32 	%r760, %r5, 5;
	and.b32 	%r761, %r931, 3456;
	bfe.s32 	%r762, %r2, 2, 1;
	and.b32 	%r763, %r762, 8208;
	or.b32 	%r764, %r760, %r761;
	xor.b32 	%r765, %r763, %r669;
	or.b32 	%r766, %r765, %r764;
	or.b32 	%r767, %r766, %r759;
	add.s32 	%r403, %r94, %r767;
	mov.b32 	%r404, {%rs365, %rs366};
	mov.b32 	%r405, {%rs369, %rs370};
	mov.b32 	%r406, {%rs373, %rs374};
	// begin inline asm
	st.shared.v4.b32 [ %r403 + 0 ], { %r404, %r405, %r406, %r407 };
	// end inline asm
	add.s32 	%r408, %r403, 512;
	mov.b32 	%r409, {%rs367, %rs368};
	mov.b32 	%r410, {%rs371, %rs372};
	mov.b32 	%r411, {%rs375, %rs376};
	mov.b32 	%r412, {%rs377, %rs378};
	// begin inline asm
	st.shared.v4.b32 [ %r408 + 0 ], { %r409, %r410, %r411, %r412 };
	// end inline asm
	add.s32 	%r413, %r403, 4096;
	mov.b32 	%r414, {%rs429, %rs430};
	mov.b32 	%r415, {%rs433, %rs434};
	mov.b32 	%r416, {%rs437, %rs438};
	// begin inline asm
	st.shared.v4.b32 [ %r413 + 0 ], { %r414, %r415, %r416, %r417 };
	// end inline asm
	add.s32 	%r418, %r403, 4608;
	mov.b32 	%r419, {%rs431, %rs432};
	mov.b32 	%r420, {%rs435, %rs436};
	mov.b32 	%r421, {%rs439, %rs440};
	mov.b32 	%r422, {%rs441, %rs442};
	// begin inline asm
	st.shared.v4.b32 [ %r418 + 0 ], { %r419, %r420, %r421, %r422 };
	// end inline asm
	xor.b32 	%r768, %r767, 32;
	add.s32 	%r423, %r94, %r768;
	mov.b32 	%r424, {%rs381, %rs382};
	mov.b32 	%r425, {%rs385, %rs386};
	mov.b32 	%r426, {%rs389, %rs390};
	// begin inline asm
	st.shared.v4.b32 [ %r423 + 0 ], { %r424, %r425, %r426, %r427 };
	// end inline asm
	add.s32 	%r428, %r423, 512;
	mov.b32 	%r429, {%rs383, %rs384};
	mov.b32 	%r430, {%rs387, %rs388};
	mov.b32 	%r431, {%rs391, %rs392};
	mov.b32 	%r432, {%rs393, %rs394};
	// begin inline asm
	st.shared.v4.b32 [ %r428 + 0 ], { %r429, %r430, %r431, %r432 };
	// end inline asm
	add.s32 	%r433, %r423, 4096;
	mov.b32 	%r434, {%rs445, %rs446};
	mov.b32 	%r435, {%rs449, %rs450};
	mov.b32 	%r436, {%rs453, %rs454};
	// begin inline asm
	st.shared.v4.b32 [ %r433 + 0 ], { %r434, %r435, %r436, %r437 };
	// end inline asm
	add.s32 	%r438, %r423, 4608;
	mov.b32 	%r439, {%rs447, %rs448};
	mov.b32 	%r440, {%rs451, %rs452};
	mov.b32 	%r441, {%rs455, %rs456};
	mov.b32 	%r442, {%rs457, %rs458};
	// begin inline asm
	st.shared.v4.b32 [ %r438 + 0 ], { %r439, %r440, %r441, %r442 };
	// end inline asm
	xor.b32 	%r769, %r767, 64;
	add.s32 	%r443, %r94, %r769;
	mov.b32 	%r444, {%rs397, %rs398};
	mov.b32 	%r445, {%rs401, %rs402};
	mov.b32 	%r446, {%rs405, %rs406};
	// begin inline asm
	st.shared.v4.b32 [ %r443 + 0 ], { %r444, %r445, %r446, %r447 };
	// end inline asm
	add.s32 	%r448, %r443, 512;
	mov.b32 	%r449, {%rs399, %rs400};
	mov.b32 	%r450, {%rs403, %rs404};
	mov.b32 	%r451, {%rs407, %rs408};
	mov.b32 	%r452, {%rs409, %rs410};
	// begin inline asm
	st.shared.v4.b32 [ %r448 + 0 ], { %r449, %r450, %r451, %r452 };
	// end inline asm
	add.s32 	%r453, %r443, 4096;
	mov.b32 	%r454, {%rs461, %rs462};
	mov.b32 	%r455, {%rs465, %rs466};
	mov.b32 	%r456, {%rs469, %rs470};
	// begin inline asm
	st.shared.v4.b32 [ %r453 + 0 ], { %r454, %r455, %r456, %r457 };
	// end inline asm
	add.s32 	%r458, %r443, 4608;
	mov.b32 	%r459, {%rs463, %rs464};
	mov.b32 	%r460, {%rs467, %rs468};
	mov.b32 	%r461, {%rs471, %rs472};
	mov.b32 	%r462, {%rs473, %rs474};
	// begin inline asm
	st.shared.v4.b32 [ %r458 + 0 ], { %r459, %r460, %r461, %r462 };
	// end inline asm
	xor.b32 	%r770, %r767, 96;
	add.s32 	%r463, %r94, %r770;
	mov.b32 	%r464, {%rs413, %rs414};
	mov.b32 	%r465, {%rs417, %rs418};
	mov.b32 	%r466, {%rs421, %rs422};
	// begin inline asm
	st.shared.v4.b32 [ %r463 + 0 ], { %r464, %r465, %r466, %r467 };
	// end inline asm
	add.s32 	%r468, %r463, 512;
	mov.b32 	%r469, {%rs415, %rs416};
	mov.b32 	%r470, {%rs419, %rs420};
	mov.b32 	%r471, {%rs423, %rs424};
	mov.b32 	%r472, {%rs425, %rs426};
	// begin inline asm
	st.shared.v4.b32 [ %r468 + 0 ], { %r469, %r470, %r471, %r472 };
	// end inline asm
	add.s32 	%r473, %r463, 4096;
	mov.b32 	%r474, {%rs477, %rs478};
	mov.b32 	%r475, {%rs481, %rs482};
	mov.b32 	%r476, {%rs485, %rs486};
	// begin inline asm
	st.shared.v4.b32 [ %r473 + 0 ], { %r474, %r475, %r476, %r477 };
	// end inline asm
	add.s32 	%r478, %r463, 4608;
	mov.b32 	%r479, {%rs479, %rs480};
	mov.b32 	%r480, {%rs483, %rs484};
	mov.b32 	%r481, {%rs487, %rs488};
	mov.b32 	%r482, {%rs489, %rs490};
	// begin inline asm
	st.shared.v4.b32 [ %r478 + 0 ], { %r479, %r480, %r481, %r482 };
	// end inline asm
	bar.sync 	0;
	and.b32 	%r771, %r621, 896;
	shl.b32 	%r772, %r623, 9;
	selp.b32 	%r773, 0, 8208, %p24;
	or.b32 	%r774, %r662, %r771;
	xor.b32 	%r775, %r774, %r773;
	or.b32 	%r776, %r775, %r772;
	add.s32 	%r777, %r94, %r776;
	ld.shared.v4.b32 	{%r483, %r499, %r515, %r531}, [%r777];
	ld.shared.v4.b32 	{%r487, %r503, %r519, %r535}, [%r777+1024];
	ld.shared.v4.b32 	{%r491, %r507, %r523, %r539}, [%r777+2048];
	ld.shared.v4.b32 	{%r495, %r511, %r527, %r543}, [%r777+3072];
	xor.b32 	%r778, %r776, 32;
	add.s32 	%r779, %r94, %r778;
	ld.shared.v4.b32 	{%r484, %r500, %r516, %r532}, [%r779+16384];
	ld.shared.v4.b32 	{%r488, %r504, %r520, %r536}, [%r779+17408];
	ld.shared.v4.b32 	{%r492, %r508, %r524, %r540}, [%r779+18432];
	ld.shared.v4.b32 	{%r496, %r512, %r528, %r544}, [%r779+19456];
	xor.b32 	%r780, %r776, 64;
	add.s32 	%r781, %r94, %r780;
	ld.shared.v4.b32 	{%r485, %r501, %r517, %r533}, [%r781+32768];
	ld.shared.v4.b32 	{%r489, %r505, %r521, %r537}, [%r781+33792];
	ld.shared.v4.b32 	{%r493, %r509, %r525, %r541}, [%r781+34816];
	ld.shared.v4.b32 	{%r497, %r513, %r529, %r545}, [%r781+35840];
	xor.b32 	%r782, %r776, 96;
	add.s32 	%r783, %r94, %r782;
	ld.shared.v4.b32 	{%r486, %r502, %r518, %r534}, [%r783+49152];
	ld.shared.v4.b32 	{%r490, %r506, %r522, %r538}, [%r783+50176];
	ld.shared.v4.b32 	{%r494, %r510, %r526, %r542}, [%r783+51200];
	ld.shared.v4.b32 	{%r498, %r514, %r530, %r546}, [%r783+52224];
	.loc	1 195 31                        // sk03_fa_qkv.py:195:31
	setp.lt.s32 	%p25, %r587, %r15;
	setp.lt.s32 	%p26, %r616, %r15;
	setp.lt.s32 	%p27, %r614, %r15;
	setp.lt.s32 	%p28, %r612, %r15;
	setp.lt.s32 	%p29, %r610, %r15;
	setp.lt.s32 	%p30, %r608, %r15;
	setp.lt.s32 	%p31, %r606, %r15;
	setp.lt.s32 	%p32, %r604, %r15;
	setp.lt.s32 	%p33, %r602, %r15;
	setp.lt.s32 	%p34, %r600, %r15;
	setp.lt.s32 	%p35, %r598, %r15;
	setp.lt.s32 	%p36, %r596, %r15;
	setp.lt.s32 	%p37, %r594, %r15;
	setp.lt.s32 	%p38, %r592, %r15;
	setp.lt.s32 	%p39, %r590, %r15;
	setp.lt.s32 	%p40, %r588, %r15;
	.loc	1 195 54                        // sk03_fa_qkv.py:195:54
	setp.lt.s32 	%p41, %r548, %r16;
	.loc	1 195 37                        // sk03_fa_qkv.py:195:37
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
	.loc	1 193 35                        // sk03_fa_qkv.py:193:35
	mul.lo.s32 	%r784, %r587, %r17;
	mul.lo.s32 	%r785, %r616, %r17;
	mul.lo.s32 	%r786, %r614, %r17;
	mul.lo.s32 	%r787, %r612, %r17;
	mul.lo.s32 	%r788, %r610, %r17;
	mul.lo.s32 	%r789, %r608, %r17;
	mul.lo.s32 	%r790, %r606, %r17;
	mul.lo.s32 	%r791, %r604, %r17;
	mul.lo.s32 	%r792, %r602, %r17;
	mul.lo.s32 	%r793, %r600, %r17;
	mul.lo.s32 	%r794, %r598, %r17;
	mul.lo.s32 	%r795, %r596, %r17;
	mul.lo.s32 	%r796, %r594, %r17;
	mul.lo.s32 	%r797, %r592, %r17;
	mul.lo.s32 	%r798, %r590, %r17;
	mul.lo.s32 	%r799, %r588, %r17;
	.loc	1 193 18                        // sk03_fa_qkv.py:193:18
	mad.wide.s32 	%rd226, %r784, 2, %rd10;
	mad.wide.s32 	%rd227, %r785, 2, %rd10;
	mad.wide.s32 	%rd228, %r786, 2, %rd10;
	mad.wide.s32 	%rd229, %r787, 2, %rd10;
	mad.wide.s32 	%rd230, %r788, 2, %rd10;
	mad.wide.s32 	%rd231, %r789, 2, %rd10;
	mad.wide.s32 	%rd232, %r790, 2, %rd10;
	mad.wide.s32 	%rd233, %r791, 2, %rd10;
	mad.wide.s32 	%rd234, %r792, 2, %rd10;
	mad.wide.s32 	%rd235, %r793, 2, %rd10;
	mad.wide.s32 	%rd236, %r794, 2, %rd10;
	mad.wide.s32 	%rd237, %r795, 2, %rd10;
	mad.wide.s32 	%rd238, %r796, 2, %rd10;
	mad.wide.s32 	%rd239, %r797, 2, %rd10;
	mad.wide.s32 	%rd240, %r798, 2, %rd10;
	mad.wide.s32 	%rd241, %r799, 2, %rd10;
	.loc	1 193 50                        // sk03_fa_qkv.py:193:50
	mul.wide.s32 	%rd242, %r548, 2;
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
	.loc	1 194 8                         // sk03_fa_qkv.py:194:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd186 + 0 ], { %r483, %r484, %r485, %r486 };
	// end inline asm
	// begin inline asm
	@%p9 st.global.v4.b32 [ %rd187 + 0 ], { %r487, %r488, %r489, %r490 };
	// end inline asm
	// begin inline asm
	@%p10 st.global.v4.b32 [ %rd188 + 0 ], { %r491, %r492, %r493, %r494 };
	// end inline asm
	// begin inline asm
	@%p11 st.global.v4.b32 [ %rd189 + 0 ], { %r495, %r496, %r497, %r498 };
	// end inline asm
	// begin inline asm
	@%p12 st.global.v4.b32 [ %rd190 + 0 ], { %r499, %r500, %r501, %r502 };
	// end inline asm
	// begin inline asm
	@%p13 st.global.v4.b32 [ %rd191 + 0 ], { %r503, %r504, %r505, %r506 };
	// end inline asm
	// begin inline asm
	@%p14 st.global.v4.b32 [ %rd192 + 0 ], { %r507, %r508, %r509, %r510 };
	// end inline asm
	// begin inline asm
	@%p15 st.global.v4.b32 [ %rd193 + 0 ], { %r511, %r512, %r513, %r514 };
	// end inline asm
	// begin inline asm
	@%p16 st.global.v4.b32 [ %rd194 + 0 ], { %r515, %r516, %r517, %r518 };
	// end inline asm
	// begin inline asm
	@%p17 st.global.v4.b32 [ %rd195 + 0 ], { %r519, %r520, %r521, %r522 };
	// end inline asm
	// begin inline asm
	@%p18 st.global.v4.b32 [ %rd196 + 0 ], { %r523, %r524, %r525, %r526 };
	// end inline asm
	// begin inline asm
	@%p19 st.global.v4.b32 [ %rd197 + 0 ], { %r527, %r528, %r529, %r530 };
	// end inline asm
	// begin inline asm
	@%p20 st.global.v4.b32 [ %rd198 + 0 ], { %r531, %r532, %r533, %r534 };
	// end inline asm
	// begin inline asm
	@%p21 st.global.v4.b32 [ %rd199 + 0 ], { %r535, %r536, %r537, %r538 };
	// end inline asm
	// begin inline asm
	@%p22 st.global.v4.b32 [ %rd200 + 0 ], { %r539, %r540, %r541, %r542 };
	// end inline asm
	// begin inline asm
	@%p23 st.global.v4.b32 [ %rd201 + 0 ], { %r543, %r544, %r545, %r546 };
	// end inline asm
	.loc	1 192 4                         // sk03_fa_qkv.py:192:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk03_fa_qkv.py"
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
.b32 157                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x96 DW_TAG_compile_unit
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
.b8 51
.b8 95
.b8 102
.b8 97
.b8 95
.b8 113
.b8 107
.b8 118
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
.b8 2                                   // Abbrev [2] 0x44:0x16 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 51
.b8 95
.b8 102
.b8 97
.b8 95
.b8 113
.b8 107
.b8 118
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x5a:0x46 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 68                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x6f:0x18 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 155                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x87:0x18 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 156                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_9 = _Nativo(
    "sk03_fa_qkv/tile256x128x64_shift0_abi16",
    _PTX_9, "_sk03_fa_qkv_kernel",
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

	// .globl	_sk03_fa_qkv_kernel     // -- Begin function _sk03_fa_qkv_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk03_fa_qkv_kernel
.visible .entry _sk03_fa_qkv_kernel(
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_6,
	.param .u32 _sk03_fa_qkv_kernel_param_7,
	.param .u32 _sk03_fa_qkv_kernel_param_8,
	.param .u32 _sk03_fa_qkv_kernel_param_9,
	.param .u32 _sk03_fa_qkv_kernel_param_10,
	.param .u32 _sk03_fa_qkv_kernel_param_11,
	.param .u32 _sk03_fa_qkv_kernel_param_12,
	.param .u32 _sk03_fa_qkv_kernel_param_13,
	.param .u32 _sk03_fa_qkv_kernel_param_14,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_16
)
.reqntid 256
{
	.reg .pred 	%p<42>;
	.reg .b16 	%rs<491>;
	.reg .b32 	%r<1342>;
	.reg .b64 	%rd<190>;
	.loc	1 145 0                         // sk03_fa_qkv.py:145:0
$L__func_begin0:
	.loc	1 145 0                         // sk03_fa_qkv.py:145:0

// %bb.0:
	ld.param.b32 	%r25, [_sk03_fa_qkv_kernel_param_13];
	ld.param.b32 	%r24, [_sk03_fa_qkv_kernel_param_12];
	ld.param.b32 	%r23, [_sk03_fa_qkv_kernel_param_9];
	ld.param.b32 	%r22, [_sk03_fa_qkv_kernel_param_8];
	ld.param.b32 	%r21, [_sk03_fa_qkv_kernel_param_7];
	ld.param.b64 	%rd37, [_sk03_fa_qkv_kernel_param_5];
	ld.param.b64 	%rd36, [_sk03_fa_qkv_kernel_param_4];
	ld.param.b64 	%rd35, [_sk03_fa_qkv_kernel_param_3];
	ld.param.b64 	%rd34, [_sk03_fa_qkv_kernel_param_2];
	ld.param.b64 	%rd33, [_sk03_fa_qkv_kernel_param_1];
	ld.param.b64 	%rd32, [_sk03_fa_qkv_kernel_param_0];
$L__tmp0:
	.loc	1 154 24                        // sk03_fa_qkv.py:154:24
	mov.u32 	%r48, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk03_fa_qkv.py:155:27 ]
	add.s32 	%r49, %r21, 255;
	.loc	2 43 30                         // standard.py:43:30 @[ sk03_fa_qkv.py:155:27 ]
	shr.s32 	%r50, %r49, 31;
	shr.u32 	%r51, %r50, 24;
	add.s32 	%r52, %r49, %r51;
	shr.s32 	%r53, %r52, 8;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk03_fa_qkv.py:156:27 ]
	add.s32 	%r54, %r22, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk03_fa_qkv.py:156:27 ]
	shr.s32 	%r55, %r54, 31;
	shr.u32 	%r56, %r55, 25;
	add.s32 	%r57, %r54, %r56;
	shr.s32 	%r58, %r57, 7;
$L__tmp3:
	.loc	1 157 29                        // sk03_fa_qkv.py:157:29
	shl.b32 	%r59, %r58, 3;
	.loc	1 158 22                        // sk03_fa_qkv.py:158:22
	div.s32 	%r60, %r48, %r59;
	.loc	1 158 38                        // sk03_fa_qkv.py:158:38
	shl.b32 	%r61, %r60, 3;
	.loc	1 159 30                        // sk03_fa_qkv.py:159:30
	sub.s32 	%r62, %r53, %r61;
	ld.param.b32 	%r63, [_sk03_fa_qkv_kernel_param_10];
	.loc	1 159 39                        // sk03_fa_qkv.py:159:39
	min.s32 	%r64, %r62, 8;
	ld.param.b32 	%r65, [_sk03_fa_qkv_kernel_param_11];
	.loc	1 160 30                        // sk03_fa_qkv.py:160:30
	mul.lo.s32 	%r66, %r60, %r59;
	sub.s32 	%r67, %r48, %r66;
	.loc	1 161 36                        // sk03_fa_qkv.py:161:36
	div.s32 	%r68, %r67, %r64;
	.loc	1 160 46                        // sk03_fa_qkv.py:160:46
	mul.lo.s32 	%r69, %r68, %r64;
	sub.s32 	%r70, %r67, %r69;
	.loc	1 160 23                        // sk03_fa_qkv.py:160:23
	add.s32 	%r71, %r70, %r61;
	.loc	1 163 22                        // sk03_fa_qkv.py:163:22
	shl.b32 	%r1, %r71, 8;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	mov.u32 	%r2, %tid.x;
	shr.u32 	%r72, %r2, 2;
	bfe.u32 	%r73, %r2, 2, 6;
	or.b32 	%r74, %r73, 64;
	and.b32 	%r3, %r2, 255;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r75, %r1, %r73;
	or.b32 	%r76, %r1, %r74;
	or.b32 	%r77, %r75, 128;
	or.b32 	%r78, %r1, %r72;
	or.b32 	%r79, %r78, 192;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r80, %r75, %r21;
	rem.s32 	%r81, %r76, %r21;
	rem.s32 	%r82, %r77, %r21;
	rem.s32 	%r83, %r79, %r21;
	.loc	1 164 22                        // sk03_fa_qkv.py:164:22
	shl.b32 	%r4, %r68, 7;
	.loc	1 164 45                        // sk03_fa_qkv.py:164:45
	and.b32 	%r5, %r2, 3;
	shl.b32 	%r84, %r5, 1;
	and.b32 	%r6, %r2, 32;
	shr.u32 	%r85, %r6, 2;
	or.b32 	%r86, %r85, %r84;
	and.b32 	%r7, %r2, 15;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
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
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
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
	.loc	1 167 39                        // sk03_fa_qkv.py:167:39
	mul.lo.s32 	%r115, %r80, %r63;
	mul.lo.s32 	%r116, %r81, %r63;
	mul.lo.s32 	%r117, %r82, %r63;
	mul.lo.s32 	%r118, %r83, %r63;
	.loc	1 167 21                        // sk03_fa_qkv.py:167:21
	cvt.s64.s32 	%rd1, %r115;
	add.s64 	%rd57, %rd32, %rd1;
	cvt.s64.s32 	%rd2, %r116;
	add.s64 	%rd58, %rd32, %rd2;
	cvt.s64.s32 	%rd3, %r117;
	add.s64 	%rd59, %rd32, %rd3;
	cvt.s64.s32 	%rd4, %r118;
	add.s64 	%rd60, %rd32, %rd4;
	.loc	1 167 58                        // sk03_fa_qkv.py:167:58
	shl.b32 	%r119, %r5, 4;
	.loc	1 167 51                        // sk03_fa_qkv.py:167:51
	cvt.u64.u32 	%rd5, %r119;
	add.s64 	%rd38, %rd57, %rd5;
	add.s64 	%rd39, %rd58, %rd5;
	add.s64 	%rd40, %rd59, %rd5;
	add.s64 	%rd41, %rd60, %rd5;
	.loc	1 168 21                        // sk03_fa_qkv.py:168:21
	add.s64 	%rd61, %rd33, %rd5;
	.loc	1 168 69                        // sk03_fa_qkv.py:168:69
	mul.lo.s32 	%r120, %r105, %r65;
	mul.lo.s32 	%r121, %r106, %r65;
	.loc	1 168 51                        // sk03_fa_qkv.py:168:51
	cvt.s64.s32 	%rd6, %r120;
	add.s64 	%rd42, %rd61, %rd6;
	cvt.s64.s32 	%rd7, %r121;
	add.s64 	%rd43, %rd61, %rd7;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	setp.gt.s32 	%p1, %r23, 63;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
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
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	add.s32 	%r32, %r27, 49152;
	// begin inline asm
	cp.async.cg.shared.global [ %r32 + 0 ], [ %rd42 + 0 ], 0x10, %r28;
	// end inline asm
	add.s32 	%r33, %r27, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r33 + 0 ], [ %rd43 + 0 ], 0x10, %r28;
	// end inline asm
	cp.async.commit_group;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	setp.gt.s32 	%p2, %r23, 127;
	.loc	1 183 18                        // sk03_fa_qkv.py:183:18
	add.s64 	%rd44, %rd38, 64;
	add.s64 	%rd45, %rd39, 64;
	add.s64 	%rd46, %rd40, 64;
	add.s64 	%rd47, %rd41, 64;
	.loc	1 184 18                        // sk03_fa_qkv.py:184:18
	add.s64 	%rd48, %rd42, 64;
	add.s64 	%rd49, %rd43, 64;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
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
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	add.s32 	%r39, %r27, 57344;
	// begin inline asm
	cp.async.cg.shared.global [ %r39 + 0 ], [ %rd48 + 0 ], 0x10, %r35;
	// end inline asm
	add.s32 	%r40, %r27, 61440;
	// begin inline asm
	cp.async.cg.shared.global [ %r40 + 0 ], [ %rd49 + 0 ], 0x10, %r35;
	// end inline asm
	cp.async.commit_group;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	setp.gt.s32 	%p3, %r23, 191;
	.loc	1 183 18                        // sk03_fa_qkv.py:183:18
	add.s64 	%rd50, %rd38, 128;
	add.s64 	%rd51, %rd39, 128;
	add.s64 	%rd52, %rd40, 128;
	add.s64 	%rd53, %rd41, 128;
	.loc	1 184 18                        // sk03_fa_qkv.py:184:18
	add.s64 	%rd54, %rd42, 128;
	add.s64 	%rd55, %rd43, 128;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
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
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	add.s32 	%r46, %r27, 65536;
	// begin inline asm
	cp.async.cg.shared.global [ %r46 + 0 ], [ %rd54 + 0 ], 0x10, %r42;
	// end inline asm
	add.s32 	%r47, %r27, 69632;
	// begin inline asm
	cp.async.cg.shared.global [ %r47 + 0 ], [ %rd55 + 0 ], 0x10, %r42;
	// end inline asm
	cp.async.commit_group;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	@%p1 bra 	$L__BB0_2;
	bra.uni 	$L__BB0_1;
$L__BB0_2:                              // %.lr.ph
	.loc	1 0 23                          // sk03_fa_qkv.py:0:23
	ld.param.b32 	%r26, [_sk03_fa_qkv_kernel_param_14];
	ld.param.b64 	%rd56, [_sk03_fa_qkv_kernel_param_6];
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
	.loc	1 178 28                        // sk03_fa_qkv.py:178:28
	shr.u32 	%r190, %r23, 6;
	add.s32 	%r191, %r190, -3;
	shl.b32 	%r192, %r7, 6;
	shl.b32 	%r1276, %r2, 4;
	and.b32 	%r193, %r1276, 3072;
	shl.b32 	%r194, %r2, 3;
	and.b32 	%r195, %r194, 48;
	and.b32 	%r1277, %r2, 16;
	or.b32 	%r196, %r192, %r193;
	xor.b32 	%r197, %r195, %r1277;
	or.b32 	%r18, %r196, %r197;
	xor.b32 	%r19, %r18, 32;
	shl.b32 	%r198, %r2, 6;
	and.b32 	%r199, %r198, 448;
	shl.b32 	%r200, %r6, 4;
	or.b32 	%r201, %r199, %r195;
	xor.b32 	%r202, %r201, %r17;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	add.s32 	%r203, %r188, %r200;
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
	mov.b32 	%r189, 0;
	mov.b32 	%r1147, 2;
	mov.b32 	%r1146, -1;
	mov.b64 	%rd188, 0;
	mov.b32 	%r1145, %r189;
	mov.b64 	%rd189, %rd188;
	mov.b32 	%r1148, %r189;
	mov.b32 	%r1149, %r189;
	mov.b32 	%r1150, %r189;
	mov.b32 	%r1151, %r189;
	mov.b32 	%r1152, %r189;
	mov.b32 	%r1153, %r189;
	mov.b32 	%r1154, %r189;
	mov.b32 	%r1155, %r189;
	mov.b32 	%r1156, %r189;
	mov.b32 	%r1157, %r189;
	mov.b32 	%r1158, %r189;
	mov.b32 	%r1159, %r189;
	mov.b32 	%r1160, %r189;
	mov.b32 	%r1161, %r189;
	mov.b32 	%r1162, %r189;
	mov.b32 	%r1163, %r189;
	mov.b32 	%r1164, %r189;
	mov.b32 	%r1165, %r189;
	mov.b32 	%r1166, %r189;
	mov.b32 	%r1167, %r189;
	mov.b32 	%r1168, %r189;
	mov.b32 	%r1169, %r189;
	mov.b32 	%r1170, %r189;
	mov.b32 	%r1171, %r189;
	mov.b32 	%r1172, %r189;
	mov.b32 	%r1173, %r189;
	mov.b32 	%r1174, %r189;
	mov.b32 	%r1175, %r189;
	mov.b32 	%r1176, %r189;
	mov.b32 	%r1177, %r189;
	mov.b32 	%r1178, %r189;
	mov.b32 	%r1179, %r189;
	mov.b32 	%r1180, %r189;
	mov.b32 	%r1181, %r189;
	mov.b32 	%r1182, %r189;
	mov.b32 	%r1183, %r189;
	mov.b32 	%r1184, %r189;
	mov.b32 	%r1185, %r189;
	mov.b32 	%r1186, %r189;
	mov.b32 	%r1187, %r189;
	mov.b32 	%r1188, %r189;
	mov.b32 	%r1189, %r189;
	mov.b32 	%r1190, %r189;
	mov.b32 	%r1191, %r189;
	mov.b32 	%r1192, %r189;
	mov.b32 	%r1193, %r189;
	mov.b32 	%r1194, %r189;
	mov.b32 	%r1195, %r189;
	mov.b32 	%r1196, %r189;
	mov.b32 	%r1197, %r189;
	mov.b32 	%r1198, %r189;
	mov.b32 	%r1199, %r189;
	mov.b32 	%r1200, %r189;
	mov.b32 	%r1201, %r189;
	mov.b32 	%r1202, %r189;
	mov.b32 	%r1203, %r189;
	mov.b32 	%r1204, %r189;
	mov.b32 	%r1205, %r189;
	mov.b32 	%r1206, %r189;
	mov.b32 	%r1207, %r189;
	mov.b32 	%r1208, %r189;
	mov.b32 	%r1209, %r189;
	mov.b32 	%r1210, %r189;
	mov.b32 	%r1211, %r189;
	mov.b32 	%r1212, %r189;
	mov.b32 	%r1213, %r189;
	mov.b32 	%r1214, %r189;
	mov.b32 	%r1215, %r189;
	mov.b32 	%r1216, %r189;
	mov.b32 	%r1217, %r189;
	mov.b32 	%r1218, %r189;
	mov.b32 	%r1219, %r189;
	mov.b32 	%r1220, %r189;
	mov.b32 	%r1221, %r189;
	mov.b32 	%r1222, %r189;
	mov.b32 	%r1223, %r189;
	mov.b32 	%r1224, %r189;
	mov.b32 	%r1225, %r189;
	mov.b32 	%r1226, %r189;
	mov.b32 	%r1227, %r189;
	mov.b32 	%r1228, %r189;
	mov.b32 	%r1229, %r189;
	mov.b32 	%r1230, %r189;
	mov.b32 	%r1231, %r189;
	mov.b32 	%r1232, %r189;
	mov.b32 	%r1233, %r189;
	mov.b32 	%r1234, %r189;
	mov.b32 	%r1235, %r189;
	mov.b32 	%r1236, %r189;
	mov.b32 	%r1237, %r189;
	mov.b32 	%r1238, %r189;
	mov.b32 	%r1239, %r189;
	mov.b32 	%r1240, %r189;
	mov.b32 	%r1241, %r189;
	mov.b32 	%r1242, %r189;
	mov.b32 	%r1243, %r189;
	mov.b32 	%r1244, %r189;
	mov.b32 	%r1245, %r189;
	mov.b32 	%r1246, %r189;
	mov.b32 	%r1247, %r189;
	mov.b32 	%r1248, %r189;
	mov.b32 	%r1249, %r189;
	mov.b32 	%r1250, %r189;
	mov.b32 	%r1251, %r189;
	mov.b32 	%r1252, %r189;
	mov.b32 	%r1253, %r189;
	mov.b32 	%r1254, %r189;
	mov.b32 	%r1255, %r189;
	mov.b32 	%r1256, %r189;
	mov.b32 	%r1257, %r189;
	mov.b32 	%r1258, %r189;
	mov.b32 	%r1259, %r189;
	mov.b32 	%r1260, %r189;
	mov.b32 	%r1261, %r189;
	mov.b32 	%r1262, %r189;
	mov.b32 	%r1263, %r189;
	mov.b32 	%r1264, %r189;
	mov.b32 	%r1265, %r189;
	mov.b32 	%r1266, %r189;
	mov.b32 	%r1267, %r189;
	mov.b32 	%r1268, %r189;
	mov.b32 	%r1269, %r189;
	mov.b32 	%r1270, %r189;
	mov.b32 	%r1271, %r189;
	mov.b32 	%r1272, %r189;
	mov.b32 	%r1273, %r189;
	mov.b32 	%r1274, %r189;
	mov.b32 	%r1275, %r189;
$L__BB0_3:                              // =>This Inner Loop Header: Depth=1
	setp.lt.s64 	%p4, %rd189, %rd24;
	add.s32 	%r404, %r1146, 1;
	setp.gt.s32 	%p5, %r404, 2;
	selp.b32 	%r1146, 0, %r404, %p5;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
	cp.async.wait_group 	4;
	bar.sync 	0;
	shl.b32 	%r405, %r1146, 14;
	add.s32 	%r406, %r188, %r405;
	add.s32 	%r407, %r406, %r18;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r205, %r206, %r207, %r208}, [%r407];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r225, %r226, %r227, %r228}, [%r407+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r229, %r230, %r231, %r232}, [%r407+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r233, %r234, %r235, %r236}, [%r407+12288];
	add.s32 	%r408, %r406, %r19;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r241, %r242, %r243, %r244}, [%r408];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r293, %r294, %r295, %r296}, [%r408+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r329, %r330, %r331, %r332}, [%r408+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r365, %r366, %r367, %r368}, [%r408+12288];
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	shl.b32 	%r409, %r1146, 13;
	add.s32 	%r410, %r20, %r409;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r209, %r210, %r245, %r246}, [%r410+49152];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r211, %r212, %r251, %r252}, [%r410+50176];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r213, %r214, %r257, %r258}, [%r410+51200];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r215, %r216, %r263, %r264}, [%r410+52224];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r217, %r218, %r269, %r270}, [%r410+53248];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r219, %r220, %r275, %r276}, [%r410+54272];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r221, %r222, %r281, %r282}, [%r410+55296];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r223, %r224, %r287, %r288}, [%r410+56320];
	.loc	1 179 36                        // sk03_fa_qkv.py:179:36
	mov.b32 	%r237, %r189;
	mov.b32 	%r238, %r189;
	mov.b32 	%r239, %r189;
	mov.b32 	%r240, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r237, %r238, %r239, %r240 }, { %r205, %r206, %r207, %r208 }, { %r209, %r210 }, { %r237, %r238, %r239, %r240 };
	// end inline asm
	mov.b32 	%r247, %r189;
	mov.b32 	%r248, %r189;
	mov.b32 	%r249, %r189;
	mov.b32 	%r250, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r247, %r248, %r249, %r250 }, { %r205, %r206, %r207, %r208 }, { %r211, %r212 }, { %r247, %r248, %r249, %r250 };
	// end inline asm
	mov.b32 	%r253, %r189;
	mov.b32 	%r254, %r189;
	mov.b32 	%r255, %r189;
	mov.b32 	%r256, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r253, %r254, %r255, %r256 }, { %r205, %r206, %r207, %r208 }, { %r213, %r214 }, { %r253, %r254, %r255, %r256 };
	// end inline asm
	mov.b32 	%r259, %r189;
	mov.b32 	%r260, %r189;
	mov.b32 	%r261, %r189;
	mov.b32 	%r262, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r259, %r260, %r261, %r262 }, { %r205, %r206, %r207, %r208 }, { %r215, %r216 }, { %r259, %r260, %r261, %r262 };
	// end inline asm
	mov.b32 	%r265, %r189;
	mov.b32 	%r266, %r189;
	mov.b32 	%r267, %r189;
	mov.b32 	%r268, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r265, %r266, %r267, %r268 }, { %r205, %r206, %r207, %r208 }, { %r217, %r218 }, { %r265, %r266, %r267, %r268 };
	// end inline asm
	mov.b32 	%r271, %r189;
	mov.b32 	%r272, %r189;
	mov.b32 	%r273, %r189;
	mov.b32 	%r274, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r271, %r272, %r273, %r274 }, { %r205, %r206, %r207, %r208 }, { %r219, %r220 }, { %r271, %r272, %r273, %r274 };
	// end inline asm
	mov.b32 	%r277, %r189;
	mov.b32 	%r278, %r189;
	mov.b32 	%r279, %r189;
	mov.b32 	%r280, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r277, %r278, %r279, %r280 }, { %r205, %r206, %r207, %r208 }, { %r221, %r222 }, { %r277, %r278, %r279, %r280 };
	// end inline asm
	mov.b32 	%r283, %r189;
	mov.b32 	%r284, %r189;
	mov.b32 	%r285, %r189;
	mov.b32 	%r286, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r283, %r284, %r285, %r286 }, { %r205, %r206, %r207, %r208 }, { %r223, %r224 }, { %r283, %r284, %r285, %r286 };
	// end inline asm
	mov.b32 	%r289, %r189;
	mov.b32 	%r290, %r189;
	mov.b32 	%r291, %r189;
	mov.b32 	%r292, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r289, %r290, %r291, %r292 }, { %r225, %r226, %r227, %r228 }, { %r209, %r210 }, { %r289, %r290, %r291, %r292 };
	// end inline asm
	mov.b32 	%r297, %r189;
	mov.b32 	%r298, %r189;
	mov.b32 	%r299, %r189;
	mov.b32 	%r300, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r297, %r298, %r299, %r300 }, { %r225, %r226, %r227, %r228 }, { %r211, %r212 }, { %r297, %r298, %r299, %r300 };
	// end inline asm
	mov.b32 	%r301, %r189;
	mov.b32 	%r302, %r189;
	mov.b32 	%r303, %r189;
	mov.b32 	%r304, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r301, %r302, %r303, %r304 }, { %r225, %r226, %r227, %r228 }, { %r213, %r214 }, { %r301, %r302, %r303, %r304 };
	// end inline asm
	mov.b32 	%r305, %r189;
	mov.b32 	%r306, %r189;
	mov.b32 	%r307, %r189;
	mov.b32 	%r308, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r305, %r306, %r307, %r308 }, { %r225, %r226, %r227, %r228 }, { %r215, %r216 }, { %r305, %r306, %r307, %r308 };
	// end inline asm
	mov.b32 	%r309, %r189;
	mov.b32 	%r310, %r189;
	mov.b32 	%r311, %r189;
	mov.b32 	%r312, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r309, %r310, %r311, %r312 }, { %r225, %r226, %r227, %r228 }, { %r217, %r218 }, { %r309, %r310, %r311, %r312 };
	// end inline asm
	mov.b32 	%r313, %r189;
	mov.b32 	%r314, %r189;
	mov.b32 	%r315, %r189;
	mov.b32 	%r316, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r313, %r314, %r315, %r316 }, { %r225, %r226, %r227, %r228 }, { %r219, %r220 }, { %r313, %r314, %r315, %r316 };
	// end inline asm
	mov.b32 	%r317, %r189;
	mov.b32 	%r318, %r189;
	mov.b32 	%r319, %r189;
	mov.b32 	%r320, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r317, %r318, %r319, %r320 }, { %r225, %r226, %r227, %r228 }, { %r221, %r222 }, { %r317, %r318, %r319, %r320 };
	// end inline asm
	mov.b32 	%r321, %r189;
	mov.b32 	%r322, %r189;
	mov.b32 	%r323, %r189;
	mov.b32 	%r324, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r321, %r322, %r323, %r324 }, { %r225, %r226, %r227, %r228 }, { %r223, %r224 }, { %r321, %r322, %r323, %r324 };
	// end inline asm
	mov.b32 	%r325, %r189;
	mov.b32 	%r326, %r189;
	mov.b32 	%r327, %r189;
	mov.b32 	%r328, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r325, %r326, %r327, %r328 }, { %r229, %r230, %r231, %r232 }, { %r209, %r210 }, { %r325, %r326, %r327, %r328 };
	// end inline asm
	mov.b32 	%r333, %r189;
	mov.b32 	%r334, %r189;
	mov.b32 	%r335, %r189;
	mov.b32 	%r336, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r333, %r334, %r335, %r336 }, { %r229, %r230, %r231, %r232 }, { %r211, %r212 }, { %r333, %r334, %r335, %r336 };
	// end inline asm
	mov.b32 	%r337, %r189;
	mov.b32 	%r338, %r189;
	mov.b32 	%r339, %r189;
	mov.b32 	%r340, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r337, %r338, %r339, %r340 }, { %r229, %r230, %r231, %r232 }, { %r213, %r214 }, { %r337, %r338, %r339, %r340 };
	// end inline asm
	mov.b32 	%r341, %r189;
	mov.b32 	%r342, %r189;
	mov.b32 	%r343, %r189;
	mov.b32 	%r344, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r341, %r342, %r343, %r344 }, { %r229, %r230, %r231, %r232 }, { %r215, %r216 }, { %r341, %r342, %r343, %r344 };
	// end inline asm
	mov.b32 	%r345, %r189;
	mov.b32 	%r346, %r189;
	mov.b32 	%r347, %r189;
	mov.b32 	%r348, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r345, %r346, %r347, %r348 }, { %r229, %r230, %r231, %r232 }, { %r217, %r218 }, { %r345, %r346, %r347, %r348 };
	// end inline asm
	mov.b32 	%r349, %r189;
	mov.b32 	%r350, %r189;
	mov.b32 	%r351, %r189;
	mov.b32 	%r352, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r349, %r350, %r351, %r352 }, { %r229, %r230, %r231, %r232 }, { %r219, %r220 }, { %r349, %r350, %r351, %r352 };
	// end inline asm
	mov.b32 	%r353, %r189;
	mov.b32 	%r354, %r189;
	mov.b32 	%r355, %r189;
	mov.b32 	%r356, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r353, %r354, %r355, %r356 }, { %r229, %r230, %r231, %r232 }, { %r221, %r222 }, { %r353, %r354, %r355, %r356 };
	// end inline asm
	mov.b32 	%r357, %r189;
	mov.b32 	%r358, %r189;
	mov.b32 	%r359, %r189;
	mov.b32 	%r360, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r357, %r358, %r359, %r360 }, { %r229, %r230, %r231, %r232 }, { %r223, %r224 }, { %r357, %r358, %r359, %r360 };
	// end inline asm
	mov.b32 	%r361, %r189;
	mov.b32 	%r362, %r189;
	mov.b32 	%r363, %r189;
	mov.b32 	%r364, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r361, %r362, %r363, %r364 }, { %r233, %r234, %r235, %r236 }, { %r209, %r210 }, { %r361, %r362, %r363, %r364 };
	// end inline asm
	mov.b32 	%r369, %r189;
	mov.b32 	%r370, %r189;
	mov.b32 	%r371, %r189;
	mov.b32 	%r372, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r369, %r370, %r371, %r372 }, { %r233, %r234, %r235, %r236 }, { %r211, %r212 }, { %r369, %r370, %r371, %r372 };
	// end inline asm
	mov.b32 	%r373, %r189;
	mov.b32 	%r374, %r189;
	mov.b32 	%r375, %r189;
	mov.b32 	%r376, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r373, %r374, %r375, %r376 }, { %r233, %r234, %r235, %r236 }, { %r213, %r214 }, { %r373, %r374, %r375, %r376 };
	// end inline asm
	mov.b32 	%r377, %r189;
	mov.b32 	%r378, %r189;
	mov.b32 	%r379, %r189;
	mov.b32 	%r380, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r377, %r378, %r379, %r380 }, { %r233, %r234, %r235, %r236 }, { %r215, %r216 }, { %r377, %r378, %r379, %r380 };
	// end inline asm
	mov.b32 	%r381, %r189;
	mov.b32 	%r382, %r189;
	mov.b32 	%r383, %r189;
	mov.b32 	%r384, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r381, %r382, %r383, %r384 }, { %r233, %r234, %r235, %r236 }, { %r217, %r218 }, { %r381, %r382, %r383, %r384 };
	// end inline asm
	mov.b32 	%r385, %r189;
	mov.b32 	%r386, %r189;
	mov.b32 	%r387, %r189;
	mov.b32 	%r388, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r385, %r386, %r387, %r388 }, { %r233, %r234, %r235, %r236 }, { %r219, %r220 }, { %r385, %r386, %r387, %r388 };
	// end inline asm
	mov.b32 	%r389, %r189;
	mov.b32 	%r390, %r189;
	mov.b32 	%r391, %r189;
	mov.b32 	%r392, %r189;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r389, %r390, %r391, %r392 }, { %r233, %r234, %r235, %r236 }, { %r221, %r222 }, { %r389, %r390, %r391, %r392 };
	// end inline asm
	mov.b32 	%r396, %r189;
	mov.b32 	%r393, %r189;
	mov.b32 	%r394, %r189;
	mov.b32 	%r395, %r189;
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
	.loc	1 181 39                        // sk03_fa_qkv.py:181:39
	cvt.s64.s32 	%rd112, %r1145;
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
	.loc	1 181 29                        // sk03_fa_qkv.py:181:29
	// begin inline asm
	mov.u16 %rs1, 0x0;
	ld.global.b8 { %rs1 }, [ %rd90 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs2, 0x0;
	ld.global.b8 { %rs2 }, [ %rd91 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs3, 0x0;
	ld.global.b8 { %rs3 }, [ %rd92 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs4, 0x0;
	ld.global.b8 { %rs4 }, [ %rd93 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs5, 0x0;
	ld.global.b8 { %rs5 }, [ %rd94 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs6, 0x0;
	ld.global.b8 { %rs6 }, [ %rd95 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs7, 0x0;
	ld.global.b8 { %rs7 }, [ %rd96 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs8, 0x0;
	ld.global.b8 { %rs8 }, [ %rd97 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs9, 0x0;
	ld.global.b8 { %rs9 }, [ %rd98 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs10, 0x0;
	ld.global.b8 { %rs10 }, [ %rd99 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs11, 0x0;
	ld.global.b8 { %rs11 }, [ %rd100 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs12, 0x0;
	ld.global.b8 { %rs12 }, [ %rd101 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs13, 0x0;
	ld.global.b8 { %rs13 }, [ %rd102 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs14, 0x0;
	ld.global.b8 { %rs14 }, [ %rd103 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs15, 0x0;
	ld.global.b8 { %rs15 }, [ %rd104 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs16, 0x0;
	ld.global.b8 { %rs16 }, [ %rd105 + 0 ];
	// end inline asm
	.loc	1 181 21                        // sk03_fa_qkv.py:181:21
	cvt.u32.u16 	%r411, %rs1;
	and.b32 	%r412, %r411, 255;
	cvt.u32.u16 	%r413, %rs2;
	and.b32 	%r414, %r413, 255;
	cvt.u32.u16 	%r415, %rs3;
	and.b32 	%r416, %r415, 255;
	cvt.u32.u16 	%r417, %rs4;
	and.b32 	%r418, %r417, 255;
	cvt.u32.u16 	%r419, %rs5;
	and.b32 	%r420, %r419, 255;
	cvt.u32.u16 	%r421, %rs6;
	and.b32 	%r422, %r421, 255;
	cvt.u32.u16 	%r423, %rs7;
	and.b32 	%r424, %r423, 255;
	cvt.u32.u16 	%r425, %rs8;
	and.b32 	%r426, %r425, 255;
	cvt.u32.u16 	%r427, %rs9;
	and.b32 	%r428, %r427, 255;
	cvt.u32.u16 	%r429, %rs10;
	and.b32 	%r430, %r429, 255;
	cvt.u32.u16 	%r431, %rs11;
	and.b32 	%r432, %r431, 255;
	cvt.u32.u16 	%r433, %rs12;
	and.b32 	%r434, %r433, 255;
	cvt.u32.u16 	%r435, %rs13;
	and.b32 	%r436, %r435, 255;
	cvt.u32.u16 	%r437, %rs14;
	and.b32 	%r438, %r437, 255;
	cvt.u32.u16 	%r439, %rs15;
	and.b32 	%r440, %r439, 255;
	cvt.u32.u16 	%r441, %rs16;
	and.b32 	%r442, %r441, 255;
	shl.b32 	%r443, %r396, %r442;
	shl.b32 	%r444, %r395, %r440;
	shl.b32 	%r445, %r394, %r442;
	shl.b32 	%r446, %r393, %r440;
	shl.b32 	%r447, %r392, %r438;
	shl.b32 	%r448, %r391, %r436;
	shl.b32 	%r449, %r390, %r438;
	shl.b32 	%r450, %r389, %r436;
	shl.b32 	%r451, %r388, %r434;
	shl.b32 	%r452, %r387, %r432;
	shl.b32 	%r453, %r386, %r434;
	shl.b32 	%r454, %r385, %r432;
	shl.b32 	%r455, %r384, %r430;
	shl.b32 	%r456, %r383, %r428;
	shl.b32 	%r457, %r382, %r430;
	shl.b32 	%r458, %r381, %r428;
	shl.b32 	%r459, %r380, %r426;
	shl.b32 	%r460, %r379, %r424;
	shl.b32 	%r461, %r378, %r426;
	shl.b32 	%r462, %r377, %r424;
	shl.b32 	%r463, %r376, %r422;
	shl.b32 	%r464, %r375, %r420;
	shl.b32 	%r465, %r284, %r442;
	shl.b32 	%r466, %r286, %r442;
	shl.b32 	%r467, %r322, %r442;
	shl.b32 	%r468, %r324, %r442;
	shl.b32 	%r469, %r358, %r442;
	shl.b32 	%r470, %r360, %r442;
	shl.b32 	%r471, %r374, %r422;
	shl.b32 	%r472, %r283, %r440;
	shl.b32 	%r473, %r285, %r440;
	shl.b32 	%r474, %r321, %r440;
	shl.b32 	%r475, %r323, %r440;
	shl.b32 	%r476, %r357, %r440;
	shl.b32 	%r477, %r359, %r440;
	shl.b32 	%r478, %r373, %r420;
	shl.b32 	%r479, %r278, %r438;
	shl.b32 	%r480, %r280, %r438;
	shl.b32 	%r481, %r318, %r438;
	shl.b32 	%r482, %r320, %r438;
	shl.b32 	%r483, %r354, %r438;
	shl.b32 	%r484, %r356, %r438;
	shl.b32 	%r485, %r372, %r418;
	shl.b32 	%r486, %r277, %r436;
	shl.b32 	%r487, %r279, %r436;
	shl.b32 	%r488, %r317, %r436;
	shl.b32 	%r489, %r319, %r436;
	shl.b32 	%r490, %r353, %r436;
	shl.b32 	%r491, %r355, %r436;
	shl.b32 	%r492, %r371, %r416;
	shl.b32 	%r493, %r272, %r434;
	shl.b32 	%r494, %r274, %r434;
	shl.b32 	%r495, %r314, %r434;
	shl.b32 	%r496, %r316, %r434;
	shl.b32 	%r497, %r350, %r434;
	shl.b32 	%r498, %r352, %r434;
	shl.b32 	%r499, %r370, %r418;
	shl.b32 	%r500, %r271, %r432;
	shl.b32 	%r501, %r273, %r432;
	shl.b32 	%r502, %r313, %r432;
	shl.b32 	%r503, %r315, %r432;
	shl.b32 	%r504, %r349, %r432;
	shl.b32 	%r505, %r351, %r432;
	shl.b32 	%r506, %r369, %r416;
	shl.b32 	%r507, %r266, %r430;
	shl.b32 	%r508, %r268, %r430;
	shl.b32 	%r509, %r310, %r430;
	shl.b32 	%r510, %r312, %r430;
	shl.b32 	%r511, %r346, %r430;
	shl.b32 	%r512, %r348, %r430;
	shl.b32 	%r513, %r364, %r414;
	shl.b32 	%r514, %r265, %r428;
	shl.b32 	%r515, %r267, %r428;
	shl.b32 	%r516, %r309, %r428;
	shl.b32 	%r517, %r311, %r428;
	shl.b32 	%r518, %r345, %r428;
	shl.b32 	%r519, %r347, %r428;
	shl.b32 	%r520, %r363, %r412;
	shl.b32 	%r521, %r260, %r426;
	shl.b32 	%r522, %r262, %r426;
	shl.b32 	%r523, %r306, %r426;
	shl.b32 	%r524, %r308, %r426;
	shl.b32 	%r525, %r342, %r426;
	shl.b32 	%r526, %r344, %r426;
	shl.b32 	%r527, %r362, %r414;
	shl.b32 	%r528, %r259, %r424;
	shl.b32 	%r529, %r261, %r424;
	shl.b32 	%r530, %r305, %r424;
	shl.b32 	%r531, %r307, %r424;
	shl.b32 	%r532, %r341, %r424;
	shl.b32 	%r533, %r343, %r424;
	shl.b32 	%r534, %r361, %r412;
	shl.b32 	%r535, %r254, %r422;
	shl.b32 	%r536, %r256, %r422;
	shl.b32 	%r537, %r302, %r422;
	shl.b32 	%r538, %r304, %r422;
	shl.b32 	%r539, %r338, %r422;
	shl.b32 	%r540, %r340, %r422;
	shl.b32 	%r541, %r253, %r420;
	shl.b32 	%r542, %r255, %r420;
	shl.b32 	%r543, %r301, %r420;
	shl.b32 	%r544, %r303, %r420;
	shl.b32 	%r545, %r337, %r420;
	shl.b32 	%r546, %r339, %r420;
	shl.b32 	%r547, %r248, %r418;
	shl.b32 	%r548, %r250, %r418;
	shl.b32 	%r549, %r298, %r418;
	shl.b32 	%r550, %r300, %r418;
	shl.b32 	%r551, %r334, %r418;
	shl.b32 	%r552, %r336, %r418;
	shl.b32 	%r553, %r247, %r416;
	shl.b32 	%r554, %r249, %r416;
	shl.b32 	%r555, %r297, %r416;
	shl.b32 	%r556, %r299, %r416;
	shl.b32 	%r557, %r333, %r416;
	shl.b32 	%r558, %r335, %r416;
	shl.b32 	%r559, %r238, %r414;
	shl.b32 	%r560, %r240, %r414;
	shl.b32 	%r561, %r290, %r414;
	shl.b32 	%r562, %r292, %r414;
	shl.b32 	%r563, %r326, %r414;
	shl.b32 	%r564, %r328, %r414;
	shl.b32 	%r565, %r237, %r412;
	shl.b32 	%r566, %r239, %r412;
	shl.b32 	%r567, %r289, %r412;
	shl.b32 	%r568, %r291, %r412;
	shl.b32 	%r569, %r325, %r412;
	shl.b32 	%r570, %r327, %r412;
	.loc	1 182 15                        // sk03_fa_qkv.py:182:15
	add.s32 	%r1214, %r570, %r1214;
	add.s32 	%r1212, %r569, %r1212;
	add.s32 	%r1182, %r568, %r1182;
	add.s32 	%r1180, %r567, %r1180;
	add.s32 	%r1150, %r566, %r1150;
	add.s32 	%r1148, %r565, %r1148;
	add.s32 	%r1215, %r564, %r1215;
	add.s32 	%r1213, %r563, %r1213;
	add.s32 	%r1183, %r562, %r1183;
	add.s32 	%r1181, %r561, %r1181;
	add.s32 	%r1151, %r560, %r1151;
	add.s32 	%r1149, %r559, %r1149;
	add.s32 	%r1218, %r558, %r1218;
	add.s32 	%r1216, %r557, %r1216;
	add.s32 	%r1186, %r556, %r1186;
	add.s32 	%r1184, %r555, %r1184;
	add.s32 	%r1154, %r554, %r1154;
	add.s32 	%r1152, %r553, %r1152;
	add.s32 	%r1219, %r552, %r1219;
	add.s32 	%r1217, %r551, %r1217;
	add.s32 	%r1187, %r550, %r1187;
	add.s32 	%r1185, %r549, %r1185;
	add.s32 	%r1155, %r548, %r1155;
	add.s32 	%r1153, %r547, %r1153;
	add.s32 	%r1222, %r546, %r1222;
	add.s32 	%r1220, %r545, %r1220;
	add.s32 	%r1190, %r544, %r1190;
	add.s32 	%r1188, %r543, %r1188;
	add.s32 	%r1158, %r542, %r1158;
	add.s32 	%r1156, %r541, %r1156;
	add.s32 	%r1223, %r540, %r1223;
	add.s32 	%r1221, %r539, %r1221;
	add.s32 	%r1191, %r538, %r1191;
	add.s32 	%r1189, %r537, %r1189;
	add.s32 	%r1159, %r536, %r1159;
	add.s32 	%r1157, %r535, %r1157;
	add.s32 	%r1244, %r534, %r1244;
	add.s32 	%r1226, %r533, %r1226;
	add.s32 	%r1224, %r532, %r1224;
	add.s32 	%r1194, %r531, %r1194;
	add.s32 	%r1192, %r530, %r1192;
	add.s32 	%r1162, %r529, %r1162;
	add.s32 	%r1160, %r528, %r1160;
	add.s32 	%r1245, %r527, %r1245;
	add.s32 	%r1227, %r526, %r1227;
	add.s32 	%r1225, %r525, %r1225;
	add.s32 	%r1195, %r524, %r1195;
	add.s32 	%r1193, %r523, %r1193;
	add.s32 	%r1163, %r522, %r1163;
	add.s32 	%r1161, %r521, %r1161;
	add.s32 	%r1246, %r520, %r1246;
	add.s32 	%r1230, %r519, %r1230;
	add.s32 	%r1228, %r518, %r1228;
	add.s32 	%r1198, %r517, %r1198;
	add.s32 	%r1196, %r516, %r1196;
	add.s32 	%r1166, %r515, %r1166;
	add.s32 	%r1164, %r514, %r1164;
	add.s32 	%r1247, %r513, %r1247;
	add.s32 	%r1231, %r512, %r1231;
	add.s32 	%r1229, %r511, %r1229;
	add.s32 	%r1199, %r510, %r1199;
	add.s32 	%r1197, %r509, %r1197;
	add.s32 	%r1167, %r508, %r1167;
	add.s32 	%r1165, %r507, %r1165;
	add.s32 	%r1248, %r506, %r1248;
	add.s32 	%r1234, %r505, %r1234;
	add.s32 	%r1232, %r504, %r1232;
	add.s32 	%r1202, %r503, %r1202;
	add.s32 	%r1200, %r502, %r1200;
	add.s32 	%r1170, %r501, %r1170;
	add.s32 	%r1168, %r500, %r1168;
	add.s32 	%r1249, %r499, %r1249;
	add.s32 	%r1235, %r498, %r1235;
	add.s32 	%r1233, %r497, %r1233;
	add.s32 	%r1203, %r496, %r1203;
	add.s32 	%r1201, %r495, %r1201;
	add.s32 	%r1171, %r494, %r1171;
	add.s32 	%r1169, %r493, %r1169;
	add.s32 	%r1250, %r492, %r1250;
	add.s32 	%r1238, %r491, %r1238;
	add.s32 	%r1236, %r490, %r1236;
	add.s32 	%r1206, %r489, %r1206;
	add.s32 	%r1204, %r488, %r1204;
	add.s32 	%r1174, %r487, %r1174;
	add.s32 	%r1172, %r486, %r1172;
	add.s32 	%r1251, %r485, %r1251;
	add.s32 	%r1239, %r484, %r1239;
	add.s32 	%r1237, %r483, %r1237;
	add.s32 	%r1207, %r482, %r1207;
	add.s32 	%r1205, %r481, %r1205;
	add.s32 	%r1175, %r480, %r1175;
	add.s32 	%r1173, %r479, %r1173;
	add.s32 	%r1252, %r478, %r1252;
	add.s32 	%r1242, %r477, %r1242;
	add.s32 	%r1240, %r476, %r1240;
	add.s32 	%r1210, %r475, %r1210;
	add.s32 	%r1208, %r474, %r1208;
	add.s32 	%r1178, %r473, %r1178;
	add.s32 	%r1176, %r472, %r1176;
	add.s32 	%r1253, %r471, %r1253;
	add.s32 	%r1243, %r470, %r1243;
	add.s32 	%r1241, %r469, %r1241;
	add.s32 	%r1211, %r468, %r1211;
	add.s32 	%r1209, %r467, %r1209;
	add.s32 	%r1179, %r466, %r1179;
	add.s32 	%r1177, %r465, %r1177;
	add.s32 	%r1254, %r464, %r1254;
	add.s32 	%r1255, %r463, %r1255;
	add.s32 	%r1256, %r462, %r1256;
	add.s32 	%r1257, %r461, %r1257;
	add.s32 	%r1258, %r460, %r1258;
	add.s32 	%r1259, %r459, %r1259;
	add.s32 	%r1260, %r458, %r1260;
	add.s32 	%r1261, %r457, %r1261;
	add.s32 	%r1262, %r456, %r1262;
	add.s32 	%r1263, %r455, %r1263;
	add.s32 	%r1264, %r454, %r1264;
	add.s32 	%r1265, %r453, %r1265;
	add.s32 	%r1266, %r452, %r1266;
	add.s32 	%r1267, %r451, %r1267;
	add.s32 	%r1268, %r450, %r1268;
	add.s32 	%r1269, %r449, %r1269;
	add.s32 	%r1270, %r448, %r1270;
	add.s32 	%r1271, %r447, %r1271;
	add.s32 	%r1272, %r446, %r1272;
	add.s32 	%r1273, %r445, %r1273;
	add.s32 	%r1274, %r444, %r1274;
	add.s32 	%r1275, %r443, %r1275;
	.loc	1 183 18                        // sk03_fa_qkv.py:183:18
	add.s64 	%rd106, %rd31, %rd188;
	add.s64 	%rd107, %rd30, %rd188;
	add.s64 	%rd108, %rd29, %rd188;
	.loc	1 184 18                        // sk03_fa_qkv.py:184:18
	add.s64 	%rd109, %rd28, %rd188;
	add.s64 	%rd110, %rd27, %rd188;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	add.s64 	%rd111, %rd26, %rd188;
	add.s32 	%r571, %r1147, 1;
	setp.gt.s32 	%p6, %r571, 2;
	selp.b32 	%r1147, 0, %r571, %p6;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
	shl.b32 	%r572, %r1147, 14;
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
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	shl.b32 	%r573, %r1147, 13;
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
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	add.s64 	%rd189, %rd189, 1;
	add.s64 	%rd188, %rd188, 64;
	add.s32 	%r1145, %r1145, %r26;
	setp.ne.b64 	%p7, %rd25, %rd188;
	@%p7 bra 	$L__BB0_3;
// %bb.4:                               // %._crit_edge.loopexit
	.loc	1 186 17                        // sk03_fa_qkv.py:186:17
	cvt.rn.f32.s32 	%r575, %r1148;
	cvt.rn.f32.s32 	%r576, %r1149;
	cvt.rn.bf16x2.f32 	%r1278, %r576, %r575;
	cvt.rn.f32.s32 	%r577, %r1150;
	cvt.rn.f32.s32 	%r578, %r1151;
	cvt.rn.bf16x2.f32 	%r1279, %r578, %r577;
	cvt.rn.f32.s32 	%r579, %r1152;
	cvt.rn.f32.s32 	%r580, %r1153;
	cvt.rn.bf16x2.f32 	%r1280, %r580, %r579;
	cvt.rn.f32.s32 	%r581, %r1154;
	cvt.rn.f32.s32 	%r582, %r1155;
	cvt.rn.bf16x2.f32 	%r1281, %r582, %r581;
	cvt.rn.f32.s32 	%r583, %r1156;
	cvt.rn.f32.s32 	%r584, %r1157;
	cvt.rn.bf16x2.f32 	%r1282, %r584, %r583;
	cvt.rn.f32.s32 	%r585, %r1158;
	cvt.rn.f32.s32 	%r586, %r1159;
	cvt.rn.bf16x2.f32 	%r1283, %r586, %r585;
	cvt.rn.f32.s32 	%r587, %r1160;
	cvt.rn.f32.s32 	%r588, %r1161;
	cvt.rn.bf16x2.f32 	%r1284, %r588, %r587;
	cvt.rn.f32.s32 	%r589, %r1162;
	cvt.rn.f32.s32 	%r590, %r1163;
	cvt.rn.bf16x2.f32 	%r1285, %r590, %r589;
	cvt.rn.f32.s32 	%r591, %r1164;
	cvt.rn.f32.s32 	%r592, %r1165;
	cvt.rn.bf16x2.f32 	%r1286, %r592, %r591;
	cvt.rn.f32.s32 	%r593, %r1166;
	cvt.rn.f32.s32 	%r594, %r1167;
	cvt.rn.bf16x2.f32 	%r1287, %r594, %r593;
	cvt.rn.f32.s32 	%r595, %r1168;
	cvt.rn.f32.s32 	%r596, %r1169;
	cvt.rn.bf16x2.f32 	%r1288, %r596, %r595;
	cvt.rn.f32.s32 	%r597, %r1170;
	cvt.rn.f32.s32 	%r598, %r1171;
	cvt.rn.bf16x2.f32 	%r1289, %r598, %r597;
	cvt.rn.f32.s32 	%r599, %r1172;
	cvt.rn.f32.s32 	%r600, %r1173;
	cvt.rn.bf16x2.f32 	%r1290, %r600, %r599;
	cvt.rn.f32.s32 	%r601, %r1174;
	cvt.rn.f32.s32 	%r602, %r1175;
	cvt.rn.bf16x2.f32 	%r1291, %r602, %r601;
	cvt.rn.f32.s32 	%r603, %r1176;
	cvt.rn.f32.s32 	%r604, %r1177;
	cvt.rn.bf16x2.f32 	%r1292, %r604, %r603;
	cvt.rn.f32.s32 	%r605, %r1178;
	cvt.rn.f32.s32 	%r606, %r1179;
	cvt.rn.bf16x2.f32 	%r1293, %r606, %r605;
	cvt.rn.f32.s32 	%r607, %r1180;
	cvt.rn.f32.s32 	%r608, %r1181;
	cvt.rn.bf16x2.f32 	%r1294, %r608, %r607;
	cvt.rn.f32.s32 	%r609, %r1182;
	cvt.rn.f32.s32 	%r610, %r1183;
	cvt.rn.bf16x2.f32 	%r1295, %r610, %r609;
	cvt.rn.f32.s32 	%r611, %r1184;
	cvt.rn.f32.s32 	%r612, %r1185;
	cvt.rn.bf16x2.f32 	%r1296, %r612, %r611;
	cvt.rn.f32.s32 	%r613, %r1186;
	cvt.rn.f32.s32 	%r614, %r1187;
	cvt.rn.bf16x2.f32 	%r1297, %r614, %r613;
	cvt.rn.f32.s32 	%r615, %r1188;
	cvt.rn.f32.s32 	%r616, %r1189;
	cvt.rn.bf16x2.f32 	%r1298, %r616, %r615;
	cvt.rn.f32.s32 	%r617, %r1190;
	cvt.rn.f32.s32 	%r618, %r1191;
	cvt.rn.bf16x2.f32 	%r1299, %r618, %r617;
	cvt.rn.f32.s32 	%r619, %r1192;
	cvt.rn.f32.s32 	%r620, %r1193;
	cvt.rn.bf16x2.f32 	%r1300, %r620, %r619;
	cvt.rn.f32.s32 	%r621, %r1194;
	cvt.rn.f32.s32 	%r622, %r1195;
	cvt.rn.bf16x2.f32 	%r1301, %r622, %r621;
	cvt.rn.f32.s32 	%r623, %r1196;
	cvt.rn.f32.s32 	%r624, %r1197;
	cvt.rn.bf16x2.f32 	%r1302, %r624, %r623;
	cvt.rn.f32.s32 	%r625, %r1198;
	cvt.rn.f32.s32 	%r626, %r1199;
	cvt.rn.bf16x2.f32 	%r1303, %r626, %r625;
	cvt.rn.f32.s32 	%r627, %r1200;
	cvt.rn.f32.s32 	%r628, %r1201;
	cvt.rn.bf16x2.f32 	%r1304, %r628, %r627;
	cvt.rn.f32.s32 	%r629, %r1202;
	cvt.rn.f32.s32 	%r630, %r1203;
	cvt.rn.bf16x2.f32 	%r1305, %r630, %r629;
	cvt.rn.f32.s32 	%r631, %r1204;
	cvt.rn.f32.s32 	%r632, %r1205;
	cvt.rn.bf16x2.f32 	%r1306, %r632, %r631;
	cvt.rn.f32.s32 	%r633, %r1206;
	cvt.rn.f32.s32 	%r634, %r1207;
	cvt.rn.bf16x2.f32 	%r1307, %r634, %r633;
	cvt.rn.f32.s32 	%r635, %r1208;
	cvt.rn.f32.s32 	%r636, %r1209;
	cvt.rn.bf16x2.f32 	%r1308, %r636, %r635;
	cvt.rn.f32.s32 	%r637, %r1210;
	cvt.rn.f32.s32 	%r638, %r1211;
	cvt.rn.bf16x2.f32 	%r1309, %r638, %r637;
	cvt.rn.f32.s32 	%r639, %r1212;
	cvt.rn.f32.s32 	%r640, %r1213;
	cvt.rn.bf16x2.f32 	%r1310, %r640, %r639;
	cvt.rn.f32.s32 	%r641, %r1214;
	cvt.rn.f32.s32 	%r642, %r1215;
	cvt.rn.bf16x2.f32 	%r1311, %r642, %r641;
	cvt.rn.f32.s32 	%r643, %r1216;
	cvt.rn.f32.s32 	%r644, %r1217;
	cvt.rn.bf16x2.f32 	%r1312, %r644, %r643;
	cvt.rn.f32.s32 	%r645, %r1218;
	cvt.rn.f32.s32 	%r646, %r1219;
	cvt.rn.bf16x2.f32 	%r1313, %r646, %r645;
	cvt.rn.f32.s32 	%r647, %r1220;
	cvt.rn.f32.s32 	%r648, %r1221;
	cvt.rn.bf16x2.f32 	%r1314, %r648, %r647;
	cvt.rn.f32.s32 	%r649, %r1222;
	cvt.rn.f32.s32 	%r650, %r1223;
	cvt.rn.bf16x2.f32 	%r1315, %r650, %r649;
	cvt.rn.f32.s32 	%r651, %r1224;
	cvt.rn.f32.s32 	%r652, %r1225;
	cvt.rn.bf16x2.f32 	%r1316, %r652, %r651;
	cvt.rn.f32.s32 	%r653, %r1226;
	cvt.rn.f32.s32 	%r654, %r1227;
	cvt.rn.bf16x2.f32 	%r1317, %r654, %r653;
	cvt.rn.f32.s32 	%r655, %r1228;
	cvt.rn.f32.s32 	%r656, %r1229;
	cvt.rn.bf16x2.f32 	%r1318, %r656, %r655;
	cvt.rn.f32.s32 	%r657, %r1230;
	cvt.rn.f32.s32 	%r658, %r1231;
	cvt.rn.bf16x2.f32 	%r1319, %r658, %r657;
	cvt.rn.f32.s32 	%r659, %r1232;
	cvt.rn.f32.s32 	%r660, %r1233;
	cvt.rn.bf16x2.f32 	%r1320, %r660, %r659;
	cvt.rn.f32.s32 	%r661, %r1234;
	cvt.rn.f32.s32 	%r662, %r1235;
	cvt.rn.bf16x2.f32 	%r1321, %r662, %r661;
	cvt.rn.f32.s32 	%r663, %r1236;
	cvt.rn.f32.s32 	%r664, %r1237;
	cvt.rn.bf16x2.f32 	%r1322, %r664, %r663;
	cvt.rn.f32.s32 	%r665, %r1238;
	cvt.rn.f32.s32 	%r666, %r1239;
	cvt.rn.bf16x2.f32 	%r1323, %r666, %r665;
	cvt.rn.f32.s32 	%r667, %r1240;
	cvt.rn.f32.s32 	%r668, %r1241;
	cvt.rn.bf16x2.f32 	%r1324, %r668, %r667;
	cvt.rn.f32.s32 	%r669, %r1242;
	cvt.rn.f32.s32 	%r670, %r1243;
	cvt.rn.bf16x2.f32 	%r1325, %r670, %r669;
	cvt.rn.f32.s32 	%r671, %r1244;
	cvt.rn.f32.s32 	%r672, %r1245;
	cvt.rn.bf16x2.f32 	%r1326, %r672, %r671;
	cvt.rn.f32.s32 	%r673, %r1246;
	cvt.rn.f32.s32 	%r674, %r1247;
	cvt.rn.bf16x2.f32 	%r1327, %r674, %r673;
	cvt.rn.f32.s32 	%r675, %r1248;
	cvt.rn.f32.s32 	%r676, %r1249;
	cvt.rn.bf16x2.f32 	%r1328, %r676, %r675;
	cvt.rn.f32.s32 	%r677, %r1250;
	cvt.rn.f32.s32 	%r678, %r1251;
	cvt.rn.bf16x2.f32 	%r1329, %r678, %r677;
	cvt.rn.f32.s32 	%r679, %r1252;
	cvt.rn.f32.s32 	%r680, %r1253;
	cvt.rn.bf16x2.f32 	%r1330, %r680, %r679;
	cvt.rn.f32.s32 	%r681, %r1254;
	cvt.rn.f32.s32 	%r682, %r1255;
	cvt.rn.bf16x2.f32 	%r1331, %r682, %r681;
	cvt.rn.f32.s32 	%r683, %r1256;
	cvt.rn.f32.s32 	%r684, %r1257;
	cvt.rn.bf16x2.f32 	%r1332, %r684, %r683;
	cvt.rn.f32.s32 	%r685, %r1258;
	cvt.rn.f32.s32 	%r686, %r1259;
	cvt.rn.bf16x2.f32 	%r1333, %r686, %r685;
	cvt.rn.f32.s32 	%r687, %r1260;
	cvt.rn.f32.s32 	%r688, %r1261;
	cvt.rn.bf16x2.f32 	%r1334, %r688, %r687;
	cvt.rn.f32.s32 	%r689, %r1262;
	cvt.rn.f32.s32 	%r690, %r1263;
	cvt.rn.bf16x2.f32 	%r1335, %r690, %r689;
	cvt.rn.f32.s32 	%r691, %r1264;
	cvt.rn.f32.s32 	%r692, %r1265;
	cvt.rn.bf16x2.f32 	%r1336, %r692, %r691;
	cvt.rn.f32.s32 	%r693, %r1266;
	cvt.rn.f32.s32 	%r694, %r1267;
	cvt.rn.bf16x2.f32 	%r1337, %r694, %r693;
	cvt.rn.f32.s32 	%r695, %r1268;
	cvt.rn.f32.s32 	%r696, %r1269;
	cvt.rn.bf16x2.f32 	%r1338, %r696, %r695;
	cvt.rn.f32.s32 	%r697, %r1270;
	cvt.rn.f32.s32 	%r698, %r1271;
	cvt.rn.bf16x2.f32 	%r1339, %r698, %r697;
	cvt.rn.f32.s32 	%r699, %r1272;
	cvt.rn.f32.s32 	%r700, %r1273;
	cvt.rn.bf16x2.f32 	%r1340, %r700, %r699;
	cvt.rn.f32.s32 	%r701, %r1274;
	cvt.rn.f32.s32 	%r702, %r1275;
	cvt.rn.bf16x2.f32 	%r1341, %r702, %r701;
	bra.uni 	$L__BB0_5;
$L__BB0_1:                              // %.._crit_edge_crit_edge
	.loc	1 188 19                        // sk03_fa_qkv.py:188:19
	and.b32 	%r1277, %r2, 16;
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	shl.b32 	%r1276, %r2, 4;
	mov.b32 	%r1278, 0;
	mov.b32 	%r1279, %r1278;
	mov.b32 	%r1280, %r1278;
	mov.b32 	%r1281, %r1278;
	mov.b32 	%r1282, %r1278;
	mov.b32 	%r1283, %r1278;
	mov.b32 	%r1284, %r1278;
	mov.b32 	%r1285, %r1278;
	mov.b32 	%r1286, %r1278;
	mov.b32 	%r1287, %r1278;
	mov.b32 	%r1288, %r1278;
	mov.b32 	%r1289, %r1278;
	mov.b32 	%r1290, %r1278;
	mov.b32 	%r1291, %r1278;
	mov.b32 	%r1292, %r1278;
	mov.b32 	%r1293, %r1278;
	mov.b32 	%r1294, %r1278;
	mov.b32 	%r1295, %r1278;
	mov.b32 	%r1296, %r1278;
	mov.b32 	%r1297, %r1278;
	mov.b32 	%r1298, %r1278;
	mov.b32 	%r1299, %r1278;
	mov.b32 	%r1300, %r1278;
	mov.b32 	%r1301, %r1278;
	mov.b32 	%r1302, %r1278;
	mov.b32 	%r1303, %r1278;
	mov.b32 	%r1304, %r1278;
	mov.b32 	%r1305, %r1278;
	mov.b32 	%r1306, %r1278;
	mov.b32 	%r1307, %r1278;
	mov.b32 	%r1308, %r1278;
	mov.b32 	%r1309, %r1278;
	mov.b32 	%r1310, %r1278;
	mov.b32 	%r1311, %r1278;
	mov.b32 	%r1312, %r1278;
	mov.b32 	%r1313, %r1278;
	mov.b32 	%r1314, %r1278;
	mov.b32 	%r1315, %r1278;
	mov.b32 	%r1316, %r1278;
	mov.b32 	%r1317, %r1278;
	mov.b32 	%r1318, %r1278;
	mov.b32 	%r1319, %r1278;
	mov.b32 	%r1320, %r1278;
	mov.b32 	%r1321, %r1278;
	mov.b32 	%r1322, %r1278;
	mov.b32 	%r1323, %r1278;
	mov.b32 	%r1324, %r1278;
	mov.b32 	%r1325, %r1278;
	mov.b32 	%r1326, %r1278;
	mov.b32 	%r1327, %r1278;
	mov.b32 	%r1328, %r1278;
	mov.b32 	%r1329, %r1278;
	mov.b32 	%r1330, %r1278;
	mov.b32 	%r1331, %r1278;
	mov.b32 	%r1332, %r1278;
	mov.b32 	%r1333, %r1278;
	mov.b32 	%r1334, %r1278;
	mov.b32 	%r1335, %r1278;
	mov.b32 	%r1336, %r1278;
	mov.b32 	%r1337, %r1278;
	mov.b32 	%r1338, %r1278;
	mov.b32 	%r1339, %r1278;
	mov.b32 	%r1340, %r1278;
	mov.b32 	%r1341, %r1278;
$L__BB0_5:                              // %._crit_edge
	.loc	1 164 45                        // sk03_fa_qkv.py:164:45
	shl.b32 	%r933, %r7, 3;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r934, %r4, %r933;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r935, %r934, %r22;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r936, %r1, %r3;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r937, %r936, %r21;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	and.b32 	%r938, %r2, 240;
	bfe.u32 	%r939, %r2, 4, 4;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r940, %r939, %r1;
	or.b32 	%r941, %r940, 240;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r942, %r941, %r21;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r943, %r940, 224;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r944, %r943, %r21;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r945, %r940, 208;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r946, %r945, %r21;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r947, %r940, 192;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r948, %r947, %r21;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r949, %r940, 176;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r950, %r949, %r21;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r951, %r940, 160;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r952, %r951, %r21;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r953, %r940, 144;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r954, %r953, %r21;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r955, %r940, 128;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r956, %r955, %r21;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r957, %r940, 112;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r958, %r957, %r21;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r959, %r940, 96;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r960, %r959, %r21;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r961, %r940, 80;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r962, %r961, %r21;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r963, %r940, 64;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r964, %r963, %r21;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r965, %r940, 48;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r966, %r965, %r21;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r967, %r940, 32;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r968, %r967, %r21;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r969, %r940, 16;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r970, %r969, %r21;
	rem.s32 	%r971, %r940, %r21;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 186 54                        // sk03_fa_qkv.py:186:54
	mad.wide.s32 	%rd113, %r937, 4, %rd36;
	.loc	1 186 40                        // sk03_fa_qkv.py:186:40
	// begin inline asm
	mov.u32 %r703, 0x0;
	ld.global.b32 { %r703 }, [ %rd113 + 0 ];
	// end inline asm
	.loc	1 186 65                        // sk03_fa_qkv.py:186:65
	cvt.rn.bf16.f32 	%rs17, %r703;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	and.b32 	%r972, %r2, 7;
	shl.b32 	%r973, %r972, 3;
	shl.b32 	%r974, %r2, 2;
	and.b32 	%r975, %r974, 192;
	and.b32 	%r976, %r2, 8;
	shr.u32 	%r977, %r976, 1;
	shr.u32 	%r978, %r2, 5;
	and.b32 	%r979, %r978, 2;
	and.b32 	%r980, %r16, 256;
	add.s32 	%r981, %r188, %r973;
	add.s32 	%r982, %r981, %r975;
	add.s32 	%r983, %r982, %r977;
	add.s32 	%r984, %r983, %r979;
	add.s32 	%r704, %r984, %r980;
	// begin inline asm
	st.shared.b16 [ %r704 + 0 ], %rs17;
	// end inline asm
	bar.sync 	0;
	and.b32 	%r985, %r16, 56;
	and.b32 	%r986, %r2, 192;
	add.s32 	%r987, %r188, %r985;
	add.s32 	%r988, %r987, %r986;
	ld.shared.v4.b16 	{%rs18, %rs19, %rs20, %rs21}, [%r988];
	ld.shared.v4.b16 	{%rs22, %rs23, %rs24, %rs25}, [%r988+256];
	mov.b32 	{%rs26, %rs27}, %r1278;
	mov.b16 	%rs28, 0x8000;
	fma.rn.bf16 	%rs29, %rs18, %rs26, %rs28;
	fma.rn.bf16 	%rs30, %rs18, %rs27, %rs28;
	mov.b32 	{%rs31, %rs32}, %r1279;
	fma.rn.bf16 	%rs33, %rs20, %rs31, %rs28;
	fma.rn.bf16 	%rs34, %rs20, %rs32, %rs28;
	mov.b32 	{%rs35, %rs36}, %r1280;
	fma.rn.bf16 	%rs37, %rs18, %rs35, %rs28;
	fma.rn.bf16 	%rs38, %rs18, %rs36, %rs28;
	mov.b32 	{%rs39, %rs40}, %r1281;
	fma.rn.bf16 	%rs41, %rs20, %rs39, %rs28;
	fma.rn.bf16 	%rs42, %rs20, %rs40, %rs28;
	mov.b32 	{%rs43, %rs44}, %r1282;
	fma.rn.bf16 	%rs45, %rs18, %rs43, %rs28;
	fma.rn.bf16 	%rs46, %rs18, %rs44, %rs28;
	mov.b32 	{%rs47, %rs48}, %r1283;
	fma.rn.bf16 	%rs49, %rs20, %rs47, %rs28;
	fma.rn.bf16 	%rs50, %rs20, %rs48, %rs28;
	mov.b32 	{%rs51, %rs52}, %r1284;
	fma.rn.bf16 	%rs53, %rs18, %rs51, %rs28;
	fma.rn.bf16 	%rs54, %rs18, %rs52, %rs28;
	mov.b32 	{%rs55, %rs56}, %r1285;
	fma.rn.bf16 	%rs57, %rs20, %rs55, %rs28;
	fma.rn.bf16 	%rs58, %rs20, %rs56, %rs28;
	mov.b32 	{%rs59, %rs60}, %r1286;
	fma.rn.bf16 	%rs61, %rs18, %rs59, %rs28;
	fma.rn.bf16 	%rs62, %rs18, %rs60, %rs28;
	mov.b32 	{%rs63, %rs64}, %r1287;
	fma.rn.bf16 	%rs65, %rs20, %rs63, %rs28;
	fma.rn.bf16 	%rs66, %rs20, %rs64, %rs28;
	mov.b32 	{%rs67, %rs68}, %r1288;
	fma.rn.bf16 	%rs69, %rs18, %rs67, %rs28;
	fma.rn.bf16 	%rs70, %rs18, %rs68, %rs28;
	mov.b32 	{%rs71, %rs72}, %r1289;
	fma.rn.bf16 	%rs73, %rs20, %rs71, %rs28;
	fma.rn.bf16 	%rs74, %rs20, %rs72, %rs28;
	mov.b32 	{%rs75, %rs76}, %r1290;
	fma.rn.bf16 	%rs77, %rs18, %rs75, %rs28;
	fma.rn.bf16 	%rs78, %rs18, %rs76, %rs28;
	mov.b32 	{%rs79, %rs80}, %r1291;
	fma.rn.bf16 	%rs81, %rs20, %rs79, %rs28;
	fma.rn.bf16 	%rs82, %rs20, %rs80, %rs28;
	mov.b32 	{%rs83, %rs84}, %r1292;
	fma.rn.bf16 	%rs85, %rs18, %rs83, %rs28;
	fma.rn.bf16 	%rs86, %rs18, %rs84, %rs28;
	mov.b32 	{%rs87, %rs88}, %r1293;
	fma.rn.bf16 	%rs89, %rs20, %rs87, %rs28;
	fma.rn.bf16 	%rs90, %rs20, %rs88, %rs28;
	mov.b32 	{%rs91, %rs92}, %r1294;
	fma.rn.bf16 	%rs93, %rs19, %rs91, %rs28;
	fma.rn.bf16 	%rs94, %rs19, %rs92, %rs28;
	mov.b32 	{%rs95, %rs96}, %r1295;
	fma.rn.bf16 	%rs97, %rs21, %rs95, %rs28;
	fma.rn.bf16 	%rs98, %rs21, %rs96, %rs28;
	mov.b32 	{%rs99, %rs100}, %r1296;
	fma.rn.bf16 	%rs101, %rs19, %rs99, %rs28;
	fma.rn.bf16 	%rs102, %rs19, %rs100, %rs28;
	mov.b32 	{%rs103, %rs104}, %r1297;
	fma.rn.bf16 	%rs105, %rs21, %rs103, %rs28;
	fma.rn.bf16 	%rs106, %rs21, %rs104, %rs28;
	mov.b32 	{%rs107, %rs108}, %r1298;
	fma.rn.bf16 	%rs109, %rs19, %rs107, %rs28;
	fma.rn.bf16 	%rs110, %rs19, %rs108, %rs28;
	mov.b32 	{%rs111, %rs112}, %r1299;
	fma.rn.bf16 	%rs113, %rs21, %rs111, %rs28;
	fma.rn.bf16 	%rs114, %rs21, %rs112, %rs28;
	mov.b32 	{%rs115, %rs116}, %r1300;
	fma.rn.bf16 	%rs117, %rs19, %rs115, %rs28;
	fma.rn.bf16 	%rs118, %rs19, %rs116, %rs28;
	mov.b32 	{%rs119, %rs120}, %r1301;
	fma.rn.bf16 	%rs121, %rs21, %rs119, %rs28;
	fma.rn.bf16 	%rs122, %rs21, %rs120, %rs28;
	mov.b32 	{%rs123, %rs124}, %r1302;
	fma.rn.bf16 	%rs125, %rs19, %rs123, %rs28;
	fma.rn.bf16 	%rs126, %rs19, %rs124, %rs28;
	mov.b32 	{%rs127, %rs128}, %r1303;
	fma.rn.bf16 	%rs129, %rs21, %rs127, %rs28;
	fma.rn.bf16 	%rs130, %rs21, %rs128, %rs28;
	mov.b32 	{%rs131, %rs132}, %r1304;
	fma.rn.bf16 	%rs133, %rs19, %rs131, %rs28;
	fma.rn.bf16 	%rs134, %rs19, %rs132, %rs28;
	mov.b32 	{%rs135, %rs136}, %r1305;
	fma.rn.bf16 	%rs137, %rs21, %rs135, %rs28;
	fma.rn.bf16 	%rs138, %rs21, %rs136, %rs28;
	mov.b32 	{%rs139, %rs140}, %r1306;
	fma.rn.bf16 	%rs141, %rs19, %rs139, %rs28;
	fma.rn.bf16 	%rs142, %rs19, %rs140, %rs28;
	mov.b32 	{%rs143, %rs144}, %r1307;
	fma.rn.bf16 	%rs145, %rs21, %rs143, %rs28;
	fma.rn.bf16 	%rs146, %rs21, %rs144, %rs28;
	mov.b32 	{%rs147, %rs148}, %r1308;
	fma.rn.bf16 	%rs149, %rs19, %rs147, %rs28;
	fma.rn.bf16 	%rs150, %rs19, %rs148, %rs28;
	mov.b32 	{%rs151, %rs152}, %r1309;
	fma.rn.bf16 	%rs153, %rs21, %rs151, %rs28;
	fma.rn.bf16 	%rs154, %rs21, %rs152, %rs28;
	mov.b32 	{%rs155, %rs156}, %r1310;
	fma.rn.bf16 	%rs157, %rs22, %rs155, %rs28;
	fma.rn.bf16 	%rs158, %rs22, %rs156, %rs28;
	mov.b32 	{%rs159, %rs160}, %r1311;
	fma.rn.bf16 	%rs161, %rs24, %rs159, %rs28;
	fma.rn.bf16 	%rs162, %rs24, %rs160, %rs28;
	mov.b32 	{%rs163, %rs164}, %r1312;
	fma.rn.bf16 	%rs165, %rs22, %rs163, %rs28;
	fma.rn.bf16 	%rs166, %rs22, %rs164, %rs28;
	mov.b32 	{%rs167, %rs168}, %r1313;
	fma.rn.bf16 	%rs169, %rs24, %rs167, %rs28;
	fma.rn.bf16 	%rs170, %rs24, %rs168, %rs28;
	mov.b32 	{%rs171, %rs172}, %r1314;
	fma.rn.bf16 	%rs173, %rs22, %rs171, %rs28;
	fma.rn.bf16 	%rs174, %rs22, %rs172, %rs28;
	mov.b32 	{%rs175, %rs176}, %r1315;
	fma.rn.bf16 	%rs177, %rs24, %rs175, %rs28;
	fma.rn.bf16 	%rs178, %rs24, %rs176, %rs28;
	mov.b32 	{%rs179, %rs180}, %r1316;
	fma.rn.bf16 	%rs181, %rs22, %rs179, %rs28;
	fma.rn.bf16 	%rs182, %rs22, %rs180, %rs28;
	mov.b32 	{%rs183, %rs184}, %r1317;
	fma.rn.bf16 	%rs185, %rs24, %rs183, %rs28;
	fma.rn.bf16 	%rs186, %rs24, %rs184, %rs28;
	mov.b32 	{%rs187, %rs188}, %r1318;
	fma.rn.bf16 	%rs189, %rs22, %rs187, %rs28;
	fma.rn.bf16 	%rs190, %rs22, %rs188, %rs28;
	mov.b32 	{%rs191, %rs192}, %r1319;
	fma.rn.bf16 	%rs193, %rs24, %rs191, %rs28;
	fma.rn.bf16 	%rs194, %rs24, %rs192, %rs28;
	mov.b32 	{%rs195, %rs196}, %r1320;
	fma.rn.bf16 	%rs197, %rs22, %rs195, %rs28;
	fma.rn.bf16 	%rs198, %rs22, %rs196, %rs28;
	mov.b32 	{%rs199, %rs200}, %r1321;
	fma.rn.bf16 	%rs201, %rs24, %rs199, %rs28;
	fma.rn.bf16 	%rs202, %rs24, %rs200, %rs28;
	mov.b32 	{%rs203, %rs204}, %r1322;
	fma.rn.bf16 	%rs205, %rs22, %rs203, %rs28;
	fma.rn.bf16 	%rs206, %rs22, %rs204, %rs28;
	mov.b32 	{%rs207, %rs208}, %r1323;
	fma.rn.bf16 	%rs209, %rs24, %rs207, %rs28;
	fma.rn.bf16 	%rs210, %rs24, %rs208, %rs28;
	mov.b32 	{%rs211, %rs212}, %r1324;
	fma.rn.bf16 	%rs213, %rs22, %rs211, %rs28;
	fma.rn.bf16 	%rs214, %rs22, %rs212, %rs28;
	mov.b32 	{%rs215, %rs216}, %r1325;
	fma.rn.bf16 	%rs217, %rs24, %rs215, %rs28;
	fma.rn.bf16 	%rs218, %rs24, %rs216, %rs28;
	mov.b32 	{%rs219, %rs220}, %r1327;
	fma.rn.bf16 	%rs221, %rs25, %rs219, %rs28;
	fma.rn.bf16 	%rs222, %rs25, %rs220, %rs28;
	mov.b32 	{%rs223, %rs224}, %r1329;
	fma.rn.bf16 	%rs225, %rs25, %rs223, %rs28;
	fma.rn.bf16 	%rs226, %rs25, %rs224, %rs28;
	mov.b32 	{%rs227, %rs228}, %r1331;
	fma.rn.bf16 	%rs229, %rs25, %rs227, %rs28;
	fma.rn.bf16 	%rs230, %rs25, %rs228, %rs28;
	mov.b32 	{%rs231, %rs232}, %r1333;
	fma.rn.bf16 	%rs233, %rs25, %rs231, %rs28;
	fma.rn.bf16 	%rs234, %rs25, %rs232, %rs28;
	mov.b32 	{%rs235, %rs236}, %r1335;
	fma.rn.bf16 	%rs237, %rs25, %rs235, %rs28;
	fma.rn.bf16 	%rs238, %rs25, %rs236, %rs28;
	mov.b32 	{%rs239, %rs240}, %r1337;
	fma.rn.bf16 	%rs241, %rs25, %rs239, %rs28;
	fma.rn.bf16 	%rs242, %rs25, %rs240, %rs28;
	mov.b32 	{%rs243, %rs244}, %r1339;
	fma.rn.bf16 	%rs245, %rs25, %rs243, %rs28;
	fma.rn.bf16 	%rs246, %rs25, %rs244, %rs28;
	mov.b32 	{%rs247, %rs248}, %r1341;
	fma.rn.bf16 	%rs249, %rs25, %rs247, %rs28;
	fma.rn.bf16 	%rs250, %rs25, %rs248, %rs28;
	.loc	1 187 38                        // sk03_fa_qkv.py:187:38
	mad.wide.s32 	%rd114, %r8, 4, %rd37;
	mad.wide.s32 	%rd115, %r9, 4, %rd37;
	mad.wide.s32 	%rd116, %r10, 4, %rd37;
	mad.wide.s32 	%rd117, %r11, 4, %rd37;
	mad.wide.s32 	%rd118, %r12, 4, %rd37;
	mad.wide.s32 	%rd119, %r13, 4, %rd37;
	mad.wide.s32 	%rd120, %r14, 4, %rd37;
	mad.wide.s32 	%rd121, %r15, 4, %rd37;
	.loc	1 187 24                        // sk03_fa_qkv.py:187:24
	// begin inline asm
	mov.u32 %r705, 0x0;
	mov.u32 %r706, 0x0;
	ld.global.v2.b32 { %r705, %r706 }, [ %rd114 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r707, 0x0;
	mov.u32 %r708, 0x0;
	ld.global.v2.b32 { %r707, %r708 }, [ %rd115 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r709, 0x0;
	mov.u32 %r710, 0x0;
	ld.global.v2.b32 { %r709, %r710 }, [ %rd116 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r711, 0x0;
	mov.u32 %r712, 0x0;
	ld.global.v2.b32 { %r711, %r712 }, [ %rd117 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r713, 0x0;
	mov.u32 %r714, 0x0;
	ld.global.v2.b32 { %r713, %r714 }, [ %rd118 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r715, 0x0;
	mov.u32 %r716, 0x0;
	ld.global.v2.b32 { %r715, %r716 }, [ %rd119 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r717, 0x0;
	mov.u32 %r718, 0x0;
	ld.global.v2.b32 { %r717, %r718 }, [ %rd120 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r719, 0x0;
	mov.u32 %r720, 0x0;
	ld.global.v2.b32 { %r719, %r720 }, [ %rd121 + 0 ];
	// end inline asm
	.loc	1 188 49                        // sk03_fa_qkv.py:188:49
	mul.lo.s32 	%r989, %r971, %r25;
	mul.lo.s32 	%r990, %r970, %r25;
	mul.lo.s32 	%r991, %r968, %r25;
	mul.lo.s32 	%r992, %r966, %r25;
	mul.lo.s32 	%r993, %r964, %r25;
	mul.lo.s32 	%r994, %r962, %r25;
	mul.lo.s32 	%r995, %r960, %r25;
	mul.lo.s32 	%r996, %r958, %r25;
	mul.lo.s32 	%r997, %r956, %r25;
	mul.lo.s32 	%r998, %r954, %r25;
	mul.lo.s32 	%r999, %r952, %r25;
	mul.lo.s32 	%r1000, %r950, %r25;
	mul.lo.s32 	%r1001, %r948, %r25;
	mul.lo.s32 	%r1002, %r946, %r25;
	mul.lo.s32 	%r1003, %r944, %r25;
	mul.lo.s32 	%r1004, %r942, %r25;
	.loc	1 188 31                        // sk03_fa_qkv.py:188:31
	mad.wide.s32 	%rd154, %r989, 2, %rd35;
	mad.wide.s32 	%rd155, %r990, 2, %rd35;
	mad.wide.s32 	%rd156, %r991, 2, %rd35;
	mad.wide.s32 	%rd157, %r992, 2, %rd35;
	mad.wide.s32 	%rd158, %r993, 2, %rd35;
	mad.wide.s32 	%rd159, %r994, 2, %rd35;
	mad.wide.s32 	%rd160, %r995, 2, %rd35;
	mad.wide.s32 	%rd161, %r996, 2, %rd35;
	mad.wide.s32 	%rd162, %r997, 2, %rd35;
	mad.wide.s32 	%rd163, %r998, 2, %rd35;
	mad.wide.s32 	%rd164, %r999, 2, %rd35;
	mad.wide.s32 	%rd165, %r1000, 2, %rd35;
	mad.wide.s32 	%rd166, %r1001, 2, %rd35;
	mad.wide.s32 	%rd167, %r1002, 2, %rd35;
	mad.wide.s32 	%rd168, %r1003, 2, %rd35;
	mad.wide.s32 	%rd169, %r1004, 2, %rd35;
	.loc	1 188 64                        // sk03_fa_qkv.py:188:64
	mul.wide.s32 	%rd170, %r935, 2;
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
	.loc	1 188 19                        // sk03_fa_qkv.py:188:19
	// begin inline asm
	mov.u32 %r722, 0x0;
	mov.u32 %r723, 0x0;
	mov.u32 %r724, 0x0;
	mov.u32 %r725, 0x0;
	ld.global.v4.b32 { %r722, %r723, %r724, %r725 }, [ %rd122 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r727, 0x0;
	mov.u32 %r728, 0x0;
	mov.u32 %r729, 0x0;
	mov.u32 %r730, 0x0;
	ld.global.v4.b32 { %r727, %r728, %r729, %r730 }, [ %rd123 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r732, 0x0;
	mov.u32 %r733, 0x0;
	mov.u32 %r734, 0x0;
	mov.u32 %r735, 0x0;
	ld.global.v4.b32 { %r732, %r733, %r734, %r735 }, [ %rd124 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r737, 0x0;
	mov.u32 %r738, 0x0;
	mov.u32 %r739, 0x0;
	mov.u32 %r740, 0x0;
	ld.global.v4.b32 { %r737, %r738, %r739, %r740 }, [ %rd125 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r741, 0x0;
	mov.u32 %r742, 0x0;
	mov.u32 %r743, 0x0;
	mov.u32 %r744, 0x0;
	ld.global.v4.b32 { %r741, %r742, %r743, %r744 }, [ %rd126 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r745, 0x0;
	mov.u32 %r746, 0x0;
	mov.u32 %r747, 0x0;
	mov.u32 %r748, 0x0;
	ld.global.v4.b32 { %r745, %r746, %r747, %r748 }, [ %rd127 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r749, 0x0;
	mov.u32 %r750, 0x0;
	mov.u32 %r751, 0x0;
	mov.u32 %r752, 0x0;
	ld.global.v4.b32 { %r749, %r750, %r751, %r752 }, [ %rd128 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r753, 0x0;
	mov.u32 %r754, 0x0;
	mov.u32 %r755, 0x0;
	mov.u32 %r756, 0x0;
	ld.global.v4.b32 { %r753, %r754, %r755, %r756 }, [ %rd129 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r757, 0x0;
	mov.u32 %r758, 0x0;
	mov.u32 %r759, 0x0;
	mov.u32 %r760, 0x0;
	ld.global.v4.b32 { %r757, %r758, %r759, %r760 }, [ %rd130 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r761, 0x0;
	mov.u32 %r762, 0x0;
	mov.u32 %r763, 0x0;
	mov.u32 %r764, 0x0;
	ld.global.v4.b32 { %r761, %r762, %r763, %r764 }, [ %rd131 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r765, 0x0;
	mov.u32 %r766, 0x0;
	mov.u32 %r767, 0x0;
	mov.u32 %r768, 0x0;
	ld.global.v4.b32 { %r765, %r766, %r767, %r768 }, [ %rd132 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r769, 0x0;
	mov.u32 %r770, 0x0;
	mov.u32 %r771, 0x0;
	mov.u32 %r772, 0x0;
	ld.global.v4.b32 { %r769, %r770, %r771, %r772 }, [ %rd133 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r773, 0x0;
	mov.u32 %r774, 0x0;
	mov.u32 %r775, 0x0;
	mov.u32 %r776, 0x0;
	ld.global.v4.b32 { %r773, %r774, %r775, %r776 }, [ %rd134 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r777, 0x0;
	mov.u32 %r778, 0x0;
	mov.u32 %r779, 0x0;
	mov.u32 %r780, 0x0;
	ld.global.v4.b32 { %r777, %r778, %r779, %r780 }, [ %rd135 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r781, 0x0;
	mov.u32 %r782, 0x0;
	mov.u32 %r783, 0x0;
	mov.u32 %r784, 0x0;
	ld.global.v4.b32 { %r781, %r782, %r783, %r784 }, [ %rd136 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r785, 0x0;
	mov.u32 %r786, 0x0;
	mov.u32 %r787, 0x0;
	mov.u32 %r788, 0x0;
	ld.global.v4.b32 { %r785, %r786, %r787, %r788 }, [ %rd137 + 0 ];
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r1005, %r2, 7;
	and.b32 	%r1006, %r1005, 15360;
	shl.b32 	%r1007, %r972, 4;
	or.b32 	%r1008, %r1006, %r1007;
	xor.b32 	%r1009, %r1008, %r938;
	add.s32 	%r721, %r188, %r1009;
	// begin inline asm
	st.shared.v4.b32 [ %r721 + 0 ], { %r722, %r723, %r724, %r725 };
	// end inline asm
	add.s32 	%r726, %r721, 256;
	// begin inline asm
	st.shared.v4.b32 [ %r726 + 0 ], { %r727, %r728, %r729, %r730 };
	// end inline asm
	add.s32 	%r731, %r721, 512;
	// begin inline asm
	st.shared.v4.b32 [ %r731 + 0 ], { %r732, %r733, %r734, %r735 };
	// end inline asm
	add.s32 	%r736, %r721, 768;
	// begin inline asm
	st.shared.v4.b32 [ %r736 + 0 ], { %r737, %r738, %r739, %r740 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r1010, %r972, 11;
	shl.b32 	%r1011, %r7, 4;
	shl.b32 	%r1012, %r986, 2;
	setp.eq.b32 	%p24, %r1277, 0;
	shl.b32 	%r1013, %r1277, 1;
	shr.u32 	%r1014, %r6, 1;
	or.b32 	%r1015, %r1011, %r1012;
	or.b32 	%r1016, %r1013, %r1014;
	xor.b32 	%r1017, %r1015, %r1016;
	or.b32 	%r1018, %r1017, %r1010;
	add.s32 	%r1019, %r188, %r1018;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1020, %r1021, %r1022, %r1023}, [%r1019];
	mov.b32 	{%rs251, %rs252}, %r1020;
	mov.b32 	{%rs253, %rs254}, %r1021;
	mov.b32 	{%rs255, %rs256}, %r1022;
	mov.b32 	{%rs257, %rs258}, %r1023;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1024, %r1025, %r1026, %r1027}, [%r1019+1024];
	mov.b32 	{%rs259, %rs260}, %r1024;
	mov.b32 	{%rs261, %rs262}, %r1025;
	mov.b32 	{%rs263, %rs264}, %r1026;
	mov.b32 	{%rs265, %rs266}, %r1027;
	xor.b32 	%r1028, %r1018, 64;
	add.s32 	%r1029, %r188, %r1028;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1030, %r1031, %r1032, %r1033}, [%r1029];
	mov.b32 	{%rs267, %rs268}, %r1030;
	mov.b32 	{%rs269, %rs270}, %r1031;
	mov.b32 	{%rs271, %rs272}, %r1032;
	mov.b32 	{%rs273, %rs274}, %r1033;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1034, %r1035, %r1036, %r1037}, [%r1029+1024];
	mov.b32 	{%rs275, %rs276}, %r1034;
	mov.b32 	{%rs277, %rs278}, %r1035;
	mov.b32 	{%rs279, %rs280}, %r1036;
	mov.b32 	{%rs281, %rs282}, %r1037;
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r721 + 0 ], { %r741, %r742, %r743, %r744 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r726 + 0 ], { %r745, %r746, %r747, %r748 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r731 + 0 ], { %r749, %r750, %r751, %r752 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r736 + 0 ], { %r753, %r754, %r755, %r756 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1038, %r1039, %r1040, %r1041}, [%r1019];
	mov.b32 	{%rs283, %rs284}, %r1038;
	mov.b32 	{%rs285, %rs286}, %r1039;
	mov.b32 	{%rs287, %rs288}, %r1040;
	mov.b32 	{%rs289, %rs290}, %r1041;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1042, %r1043, %r1044, %r1045}, [%r1019+1024];
	mov.b32 	{%rs291, %rs292}, %r1042;
	mov.b32 	{%rs293, %rs294}, %r1043;
	mov.b32 	{%rs295, %rs296}, %r1044;
	mov.b32 	{%rs297, %rs298}, %r1045;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1046, %r1047, %r1048, %r1049}, [%r1029];
	mov.b32 	{%rs299, %rs300}, %r1046;
	mov.b32 	{%rs301, %rs302}, %r1047;
	mov.b32 	{%rs303, %rs304}, %r1048;
	mov.b32 	{%rs305, %rs306}, %r1049;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1050, %r1051, %r1052, %r1053}, [%r1029+1024];
	mov.b32 	{%rs307, %rs308}, %r1050;
	mov.b32 	{%rs309, %rs310}, %r1051;
	mov.b32 	{%rs311, %rs312}, %r1052;
	mov.b32 	{%rs313, %rs314}, %r1053;
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r721 + 0 ], { %r757, %r758, %r759, %r760 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r726 + 0 ], { %r761, %r762, %r763, %r764 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r731 + 0 ], { %r765, %r766, %r767, %r768 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r736 + 0 ], { %r769, %r770, %r771, %r772 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1054, %r1055, %r1056, %r1057}, [%r1019];
	mov.b32 	{%rs315, %rs316}, %r1054;
	mov.b32 	{%rs317, %rs318}, %r1055;
	mov.b32 	{%rs319, %rs320}, %r1056;
	mov.b32 	{%rs321, %rs322}, %r1057;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1058, %r1059, %r1060, %r1061}, [%r1019+1024];
	mov.b32 	{%rs323, %rs324}, %r1058;
	mov.b32 	{%rs325, %rs326}, %r1059;
	mov.b32 	{%rs327, %rs328}, %r1060;
	mov.b32 	{%rs329, %rs330}, %r1061;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1062, %r1063, %r1064, %r1065}, [%r1029];
	mov.b32 	{%rs331, %rs332}, %r1062;
	mov.b32 	{%rs333, %rs334}, %r1063;
	mov.b32 	{%rs335, %rs336}, %r1064;
	mov.b32 	{%rs337, %rs338}, %r1065;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1066, %r1067, %r1068, %r1069}, [%r1029+1024];
	mov.b32 	{%rs339, %rs340}, %r1066;
	mov.b32 	{%rs341, %rs342}, %r1067;
	mov.b32 	{%rs343, %rs344}, %r1068;
	mov.b32 	{%rs345, %rs346}, %r1069;
	bar.sync 	0;
	// begin inline asm
	st.shared.v4.b32 [ %r721 + 0 ], { %r773, %r774, %r775, %r776 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r726 + 0 ], { %r777, %r778, %r779, %r780 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r731 + 0 ], { %r781, %r782, %r783, %r784 };
	// end inline asm
	// begin inline asm
	st.shared.v4.b32 [ %r736 + 0 ], { %r785, %r786, %r787, %r788 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1070, %r1071, %r1072, %r1073}, [%r1019];
	mov.b32 	{%rs347, %rs348}, %r1071;
	mov.b32 	{%rs349, %rs350}, %r1073;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1074, %r1075, %r1076, %r1077}, [%r1019+1024];
	mov.b32 	{%rs351, %rs352}, %r1075;
	mov.b32 	{%rs353, %rs354}, %r1077;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1078, %r1079, %r1080, %r1081}, [%r1029];
	mov.b32 	{%rs355, %rs356}, %r1079;
	mov.b32 	{%rs357, %rs358}, %r1081;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1082, %r1083, %r1084, %r1085}, [%r1029+1024];
	mov.b32 	{%rs359, %rs360}, %r1083;
	mov.b32 	{%rs361, %rs362}, %r1085;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	mov.b32 	%r1086, {%rs23, %rs23};
	mov.b32 	%r1087, -2147450880;
	fma.rn.bf16x2 	%r1088, %r1086, %r1326, %r1087;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs363, %r706;
	cvt.rn.bf16.f32 	%rs364, %r705;
	mov.b32 	%r1089, {%rs364, %rs363};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs365, %rs29, %rs364, %rs251;
	fma.rn.bf16 	%rs366, %rs30, %rs363, %rs252;
	fma.rn.bf16 	%rs367, %rs33, %rs364, %rs253;
	fma.rn.bf16 	%rs368, %rs34, %rs363, %rs254;
	fma.rn.bf16 	%rs369, %rs93, %rs364, %rs283;
	fma.rn.bf16 	%rs370, %rs94, %rs363, %rs284;
	fma.rn.bf16 	%rs371, %rs97, %rs364, %rs285;
	fma.rn.bf16 	%rs372, %rs98, %rs363, %rs286;
	fma.rn.bf16 	%rs373, %rs157, %rs364, %rs315;
	fma.rn.bf16 	%rs374, %rs158, %rs363, %rs316;
	fma.rn.bf16 	%rs375, %rs161, %rs364, %rs317;
	fma.rn.bf16 	%rs376, %rs162, %rs363, %rs318;
	fma.rn.bf16x2 	%r793, %r1088, %r1089, %r1070;
	fma.rn.bf16 	%rs377, %rs221, %rs364, %rs347;
	fma.rn.bf16 	%rs378, %rs222, %rs363, %rs348;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r1090, %r1086, %r1328, %r1087;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs379, %r708;
	cvt.rn.bf16.f32 	%rs380, %r707;
	mov.b32 	%r1091, {%rs380, %rs379};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs381, %rs37, %rs380, %rs255;
	fma.rn.bf16 	%rs382, %rs38, %rs379, %rs256;
	fma.rn.bf16 	%rs383, %rs41, %rs380, %rs257;
	fma.rn.bf16 	%rs384, %rs42, %rs379, %rs258;
	fma.rn.bf16 	%rs385, %rs101, %rs380, %rs287;
	fma.rn.bf16 	%rs386, %rs102, %rs379, %rs288;
	fma.rn.bf16 	%rs387, %rs105, %rs380, %rs289;
	fma.rn.bf16 	%rs388, %rs106, %rs379, %rs290;
	fma.rn.bf16 	%rs389, %rs165, %rs380, %rs319;
	fma.rn.bf16 	%rs390, %rs166, %rs379, %rs320;
	fma.rn.bf16 	%rs391, %rs169, %rs380, %rs321;
	fma.rn.bf16 	%rs392, %rs170, %rs379, %rs322;
	fma.rn.bf16x2 	%r813, %r1090, %r1091, %r1072;
	fma.rn.bf16 	%rs393, %rs225, %rs380, %rs349;
	fma.rn.bf16 	%rs394, %rs226, %rs379, %rs350;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r1092, %r1086, %r1330, %r1087;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs395, %r710;
	cvt.rn.bf16.f32 	%rs396, %r709;
	mov.b32 	%r1093, {%rs396, %rs395};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs397, %rs45, %rs396, %rs267;
	fma.rn.bf16 	%rs398, %rs46, %rs395, %rs268;
	fma.rn.bf16 	%rs399, %rs49, %rs396, %rs269;
	fma.rn.bf16 	%rs400, %rs50, %rs395, %rs270;
	fma.rn.bf16 	%rs401, %rs109, %rs396, %rs299;
	fma.rn.bf16 	%rs402, %rs110, %rs395, %rs300;
	fma.rn.bf16 	%rs403, %rs113, %rs396, %rs301;
	fma.rn.bf16 	%rs404, %rs114, %rs395, %rs302;
	fma.rn.bf16 	%rs405, %rs173, %rs396, %rs331;
	fma.rn.bf16 	%rs406, %rs174, %rs395, %rs332;
	fma.rn.bf16 	%rs407, %rs177, %rs396, %rs333;
	fma.rn.bf16 	%rs408, %rs178, %rs395, %rs334;
	fma.rn.bf16x2 	%r833, %r1092, %r1093, %r1078;
	fma.rn.bf16 	%rs409, %rs229, %rs396, %rs355;
	fma.rn.bf16 	%rs410, %rs230, %rs395, %rs356;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r1094, %r1086, %r1332, %r1087;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs411, %r712;
	cvt.rn.bf16.f32 	%rs412, %r711;
	mov.b32 	%r1095, {%rs412, %rs411};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs413, %rs53, %rs412, %rs271;
	fma.rn.bf16 	%rs414, %rs54, %rs411, %rs272;
	fma.rn.bf16 	%rs415, %rs57, %rs412, %rs273;
	fma.rn.bf16 	%rs416, %rs58, %rs411, %rs274;
	fma.rn.bf16 	%rs417, %rs117, %rs412, %rs303;
	fma.rn.bf16 	%rs418, %rs118, %rs411, %rs304;
	fma.rn.bf16 	%rs419, %rs121, %rs412, %rs305;
	fma.rn.bf16 	%rs420, %rs122, %rs411, %rs306;
	fma.rn.bf16 	%rs421, %rs181, %rs412, %rs335;
	fma.rn.bf16 	%rs422, %rs182, %rs411, %rs336;
	fma.rn.bf16 	%rs423, %rs185, %rs412, %rs337;
	fma.rn.bf16 	%rs424, %rs186, %rs411, %rs338;
	fma.rn.bf16x2 	%r853, %r1094, %r1095, %r1080;
	fma.rn.bf16 	%rs425, %rs233, %rs412, %rs357;
	fma.rn.bf16 	%rs426, %rs234, %rs411, %rs358;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r1096, %r1086, %r1334, %r1087;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs427, %r714;
	cvt.rn.bf16.f32 	%rs428, %r713;
	mov.b32 	%r1097, {%rs428, %rs427};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs429, %rs61, %rs428, %rs259;
	fma.rn.bf16 	%rs430, %rs62, %rs427, %rs260;
	fma.rn.bf16 	%rs431, %rs65, %rs428, %rs261;
	fma.rn.bf16 	%rs432, %rs66, %rs427, %rs262;
	fma.rn.bf16 	%rs433, %rs125, %rs428, %rs291;
	fma.rn.bf16 	%rs434, %rs126, %rs427, %rs292;
	fma.rn.bf16 	%rs435, %rs129, %rs428, %rs293;
	fma.rn.bf16 	%rs436, %rs130, %rs427, %rs294;
	fma.rn.bf16 	%rs437, %rs189, %rs428, %rs323;
	fma.rn.bf16 	%rs438, %rs190, %rs427, %rs324;
	fma.rn.bf16 	%rs439, %rs193, %rs428, %rs325;
	fma.rn.bf16 	%rs440, %rs194, %rs427, %rs326;
	fma.rn.bf16x2 	%r803, %r1096, %r1097, %r1074;
	fma.rn.bf16 	%rs441, %rs237, %rs428, %rs351;
	fma.rn.bf16 	%rs442, %rs238, %rs427, %rs352;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r1098, %r1086, %r1336, %r1087;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs443, %r716;
	cvt.rn.bf16.f32 	%rs444, %r715;
	mov.b32 	%r1099, {%rs444, %rs443};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs445, %rs69, %rs444, %rs263;
	fma.rn.bf16 	%rs446, %rs70, %rs443, %rs264;
	fma.rn.bf16 	%rs447, %rs73, %rs444, %rs265;
	fma.rn.bf16 	%rs448, %rs74, %rs443, %rs266;
	fma.rn.bf16 	%rs449, %rs133, %rs444, %rs295;
	fma.rn.bf16 	%rs450, %rs134, %rs443, %rs296;
	fma.rn.bf16 	%rs451, %rs137, %rs444, %rs297;
	fma.rn.bf16 	%rs452, %rs138, %rs443, %rs298;
	fma.rn.bf16 	%rs453, %rs197, %rs444, %rs327;
	fma.rn.bf16 	%rs454, %rs198, %rs443, %rs328;
	fma.rn.bf16 	%rs455, %rs201, %rs444, %rs329;
	fma.rn.bf16 	%rs456, %rs202, %rs443, %rs330;
	fma.rn.bf16x2 	%r823, %r1098, %r1099, %r1076;
	fma.rn.bf16 	%rs457, %rs241, %rs444, %rs353;
	fma.rn.bf16 	%rs458, %rs242, %rs443, %rs354;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r1100, %r1086, %r1338, %r1087;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs459, %r718;
	cvt.rn.bf16.f32 	%rs460, %r717;
	mov.b32 	%r1101, {%rs460, %rs459};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs461, %rs77, %rs460, %rs275;
	fma.rn.bf16 	%rs462, %rs78, %rs459, %rs276;
	fma.rn.bf16 	%rs463, %rs81, %rs460, %rs277;
	fma.rn.bf16 	%rs464, %rs82, %rs459, %rs278;
	fma.rn.bf16 	%rs465, %rs141, %rs460, %rs307;
	fma.rn.bf16 	%rs466, %rs142, %rs459, %rs308;
	fma.rn.bf16 	%rs467, %rs145, %rs460, %rs309;
	fma.rn.bf16 	%rs468, %rs146, %rs459, %rs310;
	fma.rn.bf16 	%rs469, %rs205, %rs460, %rs339;
	fma.rn.bf16 	%rs470, %rs206, %rs459, %rs340;
	fma.rn.bf16 	%rs471, %rs209, %rs460, %rs341;
	fma.rn.bf16 	%rs472, %rs210, %rs459, %rs342;
	fma.rn.bf16x2 	%r843, %r1100, %r1101, %r1082;
	fma.rn.bf16 	%rs473, %rs245, %rs460, %rs359;
	fma.rn.bf16 	%rs474, %rs246, %rs459, %rs360;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r1102, %r1086, %r1340, %r1087;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs475, %r720;
	cvt.rn.bf16.f32 	%rs476, %r719;
	mov.b32 	%r1103, {%rs476, %rs475};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs477, %rs85, %rs476, %rs279;
	fma.rn.bf16 	%rs478, %rs86, %rs475, %rs280;
	fma.rn.bf16 	%rs479, %rs89, %rs476, %rs281;
	fma.rn.bf16 	%rs480, %rs90, %rs475, %rs282;
	fma.rn.bf16 	%rs481, %rs149, %rs476, %rs311;
	fma.rn.bf16 	%rs482, %rs150, %rs475, %rs312;
	fma.rn.bf16 	%rs483, %rs153, %rs476, %rs313;
	fma.rn.bf16 	%rs484, %rs154, %rs475, %rs314;
	fma.rn.bf16 	%rs485, %rs213, %rs476, %rs343;
	fma.rn.bf16 	%rs486, %rs214, %rs475, %rs344;
	fma.rn.bf16 	%rs487, %rs217, %rs476, %rs345;
	fma.rn.bf16 	%rs488, %rs218, %rs475, %rs346;
	fma.rn.bf16x2 	%r863, %r1102, %r1103, %r1084;
	fma.rn.bf16 	%rs489, %rs249, %rs476, %rs361;
	fma.rn.bf16 	%rs490, %rs250, %rs475, %rs362;
	bar.sync 	0;
	shl.b32 	%r1104, %r5, 14;
	shl.b32 	%r1105, %r5, 5;
	and.b32 	%r1106, %r1276, 3456;
	bfe.s32 	%r1107, %r2, 2, 1;
	and.b32 	%r1108, %r1107, 8208;
	or.b32 	%r1109, %r1105, %r1106;
	xor.b32 	%r1110, %r1108, %r1014;
	or.b32 	%r1111, %r1110, %r1109;
	or.b32 	%r1112, %r1111, %r1104;
	add.s32 	%r789, %r188, %r1112;
	mov.b32 	%r790, {%rs365, %rs366};
	mov.b32 	%r791, {%rs369, %rs370};
	mov.b32 	%r792, {%rs373, %rs374};
	// begin inline asm
	st.shared.v4.b32 [ %r789 + 0 ], { %r790, %r791, %r792, %r793 };
	// end inline asm
	add.s32 	%r794, %r789, 512;
	mov.b32 	%r795, {%rs367, %rs368};
	mov.b32 	%r796, {%rs371, %rs372};
	mov.b32 	%r797, {%rs375, %rs376};
	mov.b32 	%r798, {%rs377, %rs378};
	// begin inline asm
	st.shared.v4.b32 [ %r794 + 0 ], { %r795, %r796, %r797, %r798 };
	// end inline asm
	add.s32 	%r799, %r789, 4096;
	mov.b32 	%r800, {%rs429, %rs430};
	mov.b32 	%r801, {%rs433, %rs434};
	mov.b32 	%r802, {%rs437, %rs438};
	// begin inline asm
	st.shared.v4.b32 [ %r799 + 0 ], { %r800, %r801, %r802, %r803 };
	// end inline asm
	add.s32 	%r804, %r789, 4608;
	mov.b32 	%r805, {%rs431, %rs432};
	mov.b32 	%r806, {%rs435, %rs436};
	mov.b32 	%r807, {%rs439, %rs440};
	mov.b32 	%r808, {%rs441, %rs442};
	// begin inline asm
	st.shared.v4.b32 [ %r804 + 0 ], { %r805, %r806, %r807, %r808 };
	// end inline asm
	xor.b32 	%r1113, %r1112, 32;
	add.s32 	%r809, %r188, %r1113;
	mov.b32 	%r810, {%rs381, %rs382};
	mov.b32 	%r811, {%rs385, %rs386};
	mov.b32 	%r812, {%rs389, %rs390};
	// begin inline asm
	st.shared.v4.b32 [ %r809 + 0 ], { %r810, %r811, %r812, %r813 };
	// end inline asm
	add.s32 	%r814, %r809, 512;
	mov.b32 	%r815, {%rs383, %rs384};
	mov.b32 	%r816, {%rs387, %rs388};
	mov.b32 	%r817, {%rs391, %rs392};
	mov.b32 	%r818, {%rs393, %rs394};
	// begin inline asm
	st.shared.v4.b32 [ %r814 + 0 ], { %r815, %r816, %r817, %r818 };
	// end inline asm
	add.s32 	%r819, %r809, 4096;
	mov.b32 	%r820, {%rs445, %rs446};
	mov.b32 	%r821, {%rs449, %rs450};
	mov.b32 	%r822, {%rs453, %rs454};
	// begin inline asm
	st.shared.v4.b32 [ %r819 + 0 ], { %r820, %r821, %r822, %r823 };
	// end inline asm
	add.s32 	%r824, %r809, 4608;
	mov.b32 	%r825, {%rs447, %rs448};
	mov.b32 	%r826, {%rs451, %rs452};
	mov.b32 	%r827, {%rs455, %rs456};
	mov.b32 	%r828, {%rs457, %rs458};
	// begin inline asm
	st.shared.v4.b32 [ %r824 + 0 ], { %r825, %r826, %r827, %r828 };
	// end inline asm
	xor.b32 	%r1114, %r1112, 64;
	add.s32 	%r829, %r188, %r1114;
	mov.b32 	%r830, {%rs397, %rs398};
	mov.b32 	%r831, {%rs401, %rs402};
	mov.b32 	%r832, {%rs405, %rs406};
	// begin inline asm
	st.shared.v4.b32 [ %r829 + 0 ], { %r830, %r831, %r832, %r833 };
	// end inline asm
	add.s32 	%r834, %r829, 512;
	mov.b32 	%r835, {%rs399, %rs400};
	mov.b32 	%r836, {%rs403, %rs404};
	mov.b32 	%r837, {%rs407, %rs408};
	mov.b32 	%r838, {%rs409, %rs410};
	// begin inline asm
	st.shared.v4.b32 [ %r834 + 0 ], { %r835, %r836, %r837, %r838 };
	// end inline asm
	add.s32 	%r839, %r829, 4096;
	mov.b32 	%r840, {%rs461, %rs462};
	mov.b32 	%r841, {%rs465, %rs466};
	mov.b32 	%r842, {%rs469, %rs470};
	// begin inline asm
	st.shared.v4.b32 [ %r839 + 0 ], { %r840, %r841, %r842, %r843 };
	// end inline asm
	add.s32 	%r844, %r829, 4608;
	mov.b32 	%r845, {%rs463, %rs464};
	mov.b32 	%r846, {%rs467, %rs468};
	mov.b32 	%r847, {%rs471, %rs472};
	mov.b32 	%r848, {%rs473, %rs474};
	// begin inline asm
	st.shared.v4.b32 [ %r844 + 0 ], { %r845, %r846, %r847, %r848 };
	// end inline asm
	xor.b32 	%r1115, %r1112, 96;
	add.s32 	%r849, %r188, %r1115;
	mov.b32 	%r850, {%rs413, %rs414};
	mov.b32 	%r851, {%rs417, %rs418};
	mov.b32 	%r852, {%rs421, %rs422};
	// begin inline asm
	st.shared.v4.b32 [ %r849 + 0 ], { %r850, %r851, %r852, %r853 };
	// end inline asm
	add.s32 	%r854, %r849, 512;
	mov.b32 	%r855, {%rs415, %rs416};
	mov.b32 	%r856, {%rs419, %rs420};
	mov.b32 	%r857, {%rs423, %rs424};
	mov.b32 	%r858, {%rs425, %rs426};
	// begin inline asm
	st.shared.v4.b32 [ %r854 + 0 ], { %r855, %r856, %r857, %r858 };
	// end inline asm
	add.s32 	%r859, %r849, 4096;
	mov.b32 	%r860, {%rs477, %rs478};
	mov.b32 	%r861, {%rs481, %rs482};
	mov.b32 	%r862, {%rs485, %rs486};
	// begin inline asm
	st.shared.v4.b32 [ %r859 + 0 ], { %r860, %r861, %r862, %r863 };
	// end inline asm
	add.s32 	%r864, %r849, 4608;
	mov.b32 	%r865, {%rs479, %rs480};
	mov.b32 	%r866, {%rs483, %rs484};
	mov.b32 	%r867, {%rs487, %rs488};
	mov.b32 	%r868, {%rs489, %rs490};
	// begin inline asm
	st.shared.v4.b32 [ %r864 + 0 ], { %r865, %r866, %r867, %r868 };
	// end inline asm
	bar.sync 	0;
	and.b32 	%r1116, %r974, 896;
	shl.b32 	%r1117, %r976, 9;
	selp.b32 	%r1118, 0, 8208, %p24;
	or.b32 	%r1119, %r1007, %r1116;
	xor.b32 	%r1120, %r1119, %r1118;
	or.b32 	%r1121, %r1120, %r1117;
	add.s32 	%r1122, %r188, %r1121;
	ld.shared.v4.b32 	{%r869, %r885, %r901, %r917}, [%r1122];
	ld.shared.v4.b32 	{%r873, %r889, %r905, %r921}, [%r1122+1024];
	ld.shared.v4.b32 	{%r877, %r893, %r909, %r925}, [%r1122+2048];
	ld.shared.v4.b32 	{%r881, %r897, %r913, %r929}, [%r1122+3072];
	xor.b32 	%r1123, %r1121, 32;
	add.s32 	%r1124, %r188, %r1123;
	ld.shared.v4.b32 	{%r870, %r886, %r902, %r918}, [%r1124+16384];
	ld.shared.v4.b32 	{%r874, %r890, %r906, %r922}, [%r1124+17408];
	ld.shared.v4.b32 	{%r878, %r894, %r910, %r926}, [%r1124+18432];
	ld.shared.v4.b32 	{%r882, %r898, %r914, %r930}, [%r1124+19456];
	xor.b32 	%r1125, %r1121, 64;
	add.s32 	%r1126, %r188, %r1125;
	ld.shared.v4.b32 	{%r871, %r887, %r903, %r919}, [%r1126+32768];
	ld.shared.v4.b32 	{%r875, %r891, %r907, %r923}, [%r1126+33792];
	ld.shared.v4.b32 	{%r879, %r895, %r911, %r927}, [%r1126+34816];
	ld.shared.v4.b32 	{%r883, %r899, %r915, %r931}, [%r1126+35840];
	xor.b32 	%r1127, %r1121, 96;
	add.s32 	%r1128, %r188, %r1127;
	ld.shared.v4.b32 	{%r872, %r888, %r904, %r920}, [%r1128+49152];
	ld.shared.v4.b32 	{%r876, %r892, %r908, %r924}, [%r1128+50176];
	ld.shared.v4.b32 	{%r880, %r896, %r912, %r928}, [%r1128+51200];
	ld.shared.v4.b32 	{%r884, %r900, %r916, %r932}, [%r1128+52224];
	.loc	1 195 31                        // sk03_fa_qkv.py:195:31
	setp.lt.s32 	%p25, %r940, %r21;
	setp.lt.s32 	%p26, %r969, %r21;
	setp.lt.s32 	%p27, %r967, %r21;
	setp.lt.s32 	%p28, %r965, %r21;
	setp.lt.s32 	%p29, %r963, %r21;
	setp.lt.s32 	%p30, %r961, %r21;
	setp.lt.s32 	%p31, %r959, %r21;
	setp.lt.s32 	%p32, %r957, %r21;
	setp.lt.s32 	%p33, %r955, %r21;
	setp.lt.s32 	%p34, %r953, %r21;
	setp.lt.s32 	%p35, %r951, %r21;
	setp.lt.s32 	%p36, %r949, %r21;
	setp.lt.s32 	%p37, %r947, %r21;
	setp.lt.s32 	%p38, %r945, %r21;
	setp.lt.s32 	%p39, %r943, %r21;
	setp.lt.s32 	%p40, %r941, %r21;
	.loc	1 195 54                        // sk03_fa_qkv.py:195:54
	setp.lt.s32 	%p41, %r934, %r22;
	.loc	1 195 37                        // sk03_fa_qkv.py:195:37
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
	.loc	1 193 35                        // sk03_fa_qkv.py:193:35
	mul.lo.s32 	%r1129, %r940, %r24;
	mul.lo.s32 	%r1130, %r969, %r24;
	mul.lo.s32 	%r1131, %r967, %r24;
	mul.lo.s32 	%r1132, %r965, %r24;
	mul.lo.s32 	%r1133, %r963, %r24;
	mul.lo.s32 	%r1134, %r961, %r24;
	mul.lo.s32 	%r1135, %r959, %r24;
	mul.lo.s32 	%r1136, %r957, %r24;
	mul.lo.s32 	%r1137, %r955, %r24;
	mul.lo.s32 	%r1138, %r953, %r24;
	mul.lo.s32 	%r1139, %r951, %r24;
	mul.lo.s32 	%r1140, %r949, %r24;
	mul.lo.s32 	%r1141, %r947, %r24;
	mul.lo.s32 	%r1142, %r945, %r24;
	mul.lo.s32 	%r1143, %r943, %r24;
	mul.lo.s32 	%r1144, %r941, %r24;
	.loc	1 193 18                        // sk03_fa_qkv.py:193:18
	mad.wide.s32 	%rd171, %r1129, 2, %rd34;
	mad.wide.s32 	%rd172, %r1130, 2, %rd34;
	mad.wide.s32 	%rd173, %r1131, 2, %rd34;
	mad.wide.s32 	%rd174, %r1132, 2, %rd34;
	mad.wide.s32 	%rd175, %r1133, 2, %rd34;
	mad.wide.s32 	%rd176, %r1134, 2, %rd34;
	mad.wide.s32 	%rd177, %r1135, 2, %rd34;
	mad.wide.s32 	%rd178, %r1136, 2, %rd34;
	mad.wide.s32 	%rd179, %r1137, 2, %rd34;
	mad.wide.s32 	%rd180, %r1138, 2, %rd34;
	mad.wide.s32 	%rd181, %r1139, 2, %rd34;
	mad.wide.s32 	%rd182, %r1140, 2, %rd34;
	mad.wide.s32 	%rd183, %r1141, 2, %rd34;
	mad.wide.s32 	%rd184, %r1142, 2, %rd34;
	mad.wide.s32 	%rd185, %r1143, 2, %rd34;
	mad.wide.s32 	%rd186, %r1144, 2, %rd34;
	.loc	1 193 50                        // sk03_fa_qkv.py:193:50
	mul.wide.s32 	%rd187, %r934, 2;
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
	.loc	1 194 8                         // sk03_fa_qkv.py:194:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd138 + 0 ], { %r869, %r870, %r871, %r872 };
	// end inline asm
	// begin inline asm
	@%p9 st.global.v4.b32 [ %rd139 + 0 ], { %r873, %r874, %r875, %r876 };
	// end inline asm
	// begin inline asm
	@%p10 st.global.v4.b32 [ %rd140 + 0 ], { %r877, %r878, %r879, %r880 };
	// end inline asm
	// begin inline asm
	@%p11 st.global.v4.b32 [ %rd141 + 0 ], { %r881, %r882, %r883, %r884 };
	// end inline asm
	// begin inline asm
	@%p12 st.global.v4.b32 [ %rd142 + 0 ], { %r885, %r886, %r887, %r888 };
	// end inline asm
	// begin inline asm
	@%p13 st.global.v4.b32 [ %rd143 + 0 ], { %r889, %r890, %r891, %r892 };
	// end inline asm
	// begin inline asm
	@%p14 st.global.v4.b32 [ %rd144 + 0 ], { %r893, %r894, %r895, %r896 };
	// end inline asm
	// begin inline asm
	@%p15 st.global.v4.b32 [ %rd145 + 0 ], { %r897, %r898, %r899, %r900 };
	// end inline asm
	// begin inline asm
	@%p16 st.global.v4.b32 [ %rd146 + 0 ], { %r901, %r902, %r903, %r904 };
	// end inline asm
	// begin inline asm
	@%p17 st.global.v4.b32 [ %rd147 + 0 ], { %r905, %r906, %r907, %r908 };
	// end inline asm
	// begin inline asm
	@%p18 st.global.v4.b32 [ %rd148 + 0 ], { %r909, %r910, %r911, %r912 };
	// end inline asm
	// begin inline asm
	@%p19 st.global.v4.b32 [ %rd149 + 0 ], { %r913, %r914, %r915, %r916 };
	// end inline asm
	// begin inline asm
	@%p20 st.global.v4.b32 [ %rd150 + 0 ], { %r917, %r918, %r919, %r920 };
	// end inline asm
	// begin inline asm
	@%p21 st.global.v4.b32 [ %rd151 + 0 ], { %r921, %r922, %r923, %r924 };
	// end inline asm
	// begin inline asm
	@%p22 st.global.v4.b32 [ %rd152 + 0 ], { %r925, %r926, %r927, %r928 };
	// end inline asm
	// begin inline asm
	@%p23 st.global.v4.b32 [ %rd153 + 0 ], { %r929, %r930, %r931, %r932 };
	// end inline asm
	.loc	1 192 4                         // sk03_fa_qkv.py:192:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk03_fa_qkv.py"
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
.b32 157                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x96 DW_TAG_compile_unit
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
.b8 51
.b8 95
.b8 102
.b8 97
.b8 95
.b8 113
.b8 107
.b8 118
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
.b8 2                                   // Abbrev [2] 0x44:0x16 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 51
.b8 95
.b8 102
.b8 97
.b8 95
.b8 113
.b8 107
.b8 118
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x5a:0x46 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 68                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x6f:0x18 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 155                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x87:0x18 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 156                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_10 = _Nativo(
    "sk03_fa_qkv/tile256x128x64_shift1_abi15",
    _PTX_10, "_sk03_fa_qkv_kernel",
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

	// .globl	_sk03_fa_qkv_kernel     // -- Begin function _sk03_fa_qkv_kernel
.extern .shared .align 16 .b8 global_smem[];
                                        // @_sk03_fa_qkv_kernel
.visible .entry _sk03_fa_qkv_kernel(
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_3,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_4,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_5,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_6,
	.param .u32 _sk03_fa_qkv_kernel_param_7,
	.param .u32 _sk03_fa_qkv_kernel_param_8,
	.param .u32 _sk03_fa_qkv_kernel_param_9,
	.param .u32 _sk03_fa_qkv_kernel_param_10,
	.param .u32 _sk03_fa_qkv_kernel_param_11,
	.param .u32 _sk03_fa_qkv_kernel_param_12,
	.param .u32 _sk03_fa_qkv_kernel_param_13,
	.param .u32 _sk03_fa_qkv_kernel_param_14,
	.param .u32 _sk03_fa_qkv_kernel_param_15,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_16,
	.param .u64 .ptr .global .align 1 _sk03_fa_qkv_kernel_param_17
)
.reqntid 256
{
	.reg .pred 	%p<42>;
	.reg .b16 	%rs<619>;
	.reg .b32 	%r<1365>;
	.reg .b64 	%rd<309>;
	.loc	1 145 0                         // sk03_fa_qkv.py:145:0
$L__func_begin0:
	.loc	1 145 0                         // sk03_fa_qkv.py:145:0

// %bb.0:
	ld.param.b32 	%r26, [_sk03_fa_qkv_kernel_param_14];
	ld.param.b32 	%r25, [_sk03_fa_qkv_kernel_param_13];
	ld.param.b32 	%r24, [_sk03_fa_qkv_kernel_param_12];
	ld.param.b32 	%r23, [_sk03_fa_qkv_kernel_param_9];
	ld.param.b32 	%r22, [_sk03_fa_qkv_kernel_param_8];
	ld.param.b32 	%r21, [_sk03_fa_qkv_kernel_param_7];
	ld.param.b64 	%rd37, [_sk03_fa_qkv_kernel_param_5];
	ld.param.b64 	%rd36, [_sk03_fa_qkv_kernel_param_4];
	ld.param.b64 	%rd35, [_sk03_fa_qkv_kernel_param_3];
	ld.param.b64 	%rd34, [_sk03_fa_qkv_kernel_param_2];
	ld.param.b64 	%rd33, [_sk03_fa_qkv_kernel_param_1];
	ld.param.b64 	%rd32, [_sk03_fa_qkv_kernel_param_0];
$L__tmp0:
	.loc	1 154 24                        // sk03_fa_qkv.py:154:24
	mov.u32 	%r49, %ctaid.x;
$L__tmp1:
	.loc	2 43 17                         // standard.py:43:17 @[ sk03_fa_qkv.py:155:27 ]
	add.s32 	%r50, %r21, 255;
	.loc	2 43 30                         // standard.py:43:30 @[ sk03_fa_qkv.py:155:27 ]
	shr.s32 	%r51, %r50, 31;
	shr.u32 	%r52, %r51, 24;
	add.s32 	%r53, %r50, %r52;
	shr.s32 	%r54, %r53, 8;
$L__tmp2:
	.loc	2 43 17                         // standard.py:43:17 @[ sk03_fa_qkv.py:156:27 ]
	add.s32 	%r55, %r22, 127;
	.loc	2 43 30                         // standard.py:43:30 @[ sk03_fa_qkv.py:156:27 ]
	shr.s32 	%r56, %r55, 31;
	shr.u32 	%r57, %r56, 25;
	add.s32 	%r58, %r55, %r57;
	shr.s32 	%r59, %r58, 7;
$L__tmp3:
	.loc	1 157 29                        // sk03_fa_qkv.py:157:29
	shl.b32 	%r60, %r59, 3;
	.loc	1 158 22                        // sk03_fa_qkv.py:158:22
	div.s32 	%r61, %r49, %r60;
	.loc	1 158 38                        // sk03_fa_qkv.py:158:38
	shl.b32 	%r62, %r61, 3;
	.loc	1 159 30                        // sk03_fa_qkv.py:159:30
	sub.s32 	%r63, %r54, %r62;
	ld.param.b32 	%r64, [_sk03_fa_qkv_kernel_param_10];
	.loc	1 159 39                        // sk03_fa_qkv.py:159:39
	min.s32 	%r65, %r63, 8;
	ld.param.b32 	%r66, [_sk03_fa_qkv_kernel_param_11];
	.loc	1 160 30                        // sk03_fa_qkv.py:160:30
	mul.lo.s32 	%r67, %r61, %r60;
	sub.s32 	%r68, %r49, %r67;
	.loc	1 161 36                        // sk03_fa_qkv.py:161:36
	div.s32 	%r69, %r68, %r65;
	.loc	1 160 46                        // sk03_fa_qkv.py:160:46
	mul.lo.s32 	%r70, %r69, %r65;
	sub.s32 	%r71, %r68, %r70;
	.loc	1 160 23                        // sk03_fa_qkv.py:160:23
	add.s32 	%r72, %r71, %r62;
	.loc	1 163 22                        // sk03_fa_qkv.py:163:22
	shl.b32 	%r1, %r72, 8;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	mov.u32 	%r2, %tid.x;
	shr.u32 	%r73, %r2, 2;
	bfe.u32 	%r74, %r2, 2, 6;
	or.b32 	%r75, %r74, 64;
	and.b32 	%r3, %r2, 255;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r76, %r1, %r74;
	or.b32 	%r77, %r1, %r75;
	or.b32 	%r78, %r76, 128;
	or.b32 	%r79, %r1, %r73;
	or.b32 	%r80, %r79, 192;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r81, %r76, %r21;
	rem.s32 	%r82, %r77, %r21;
	rem.s32 	%r83, %r78, %r21;
	rem.s32 	%r84, %r80, %r21;
	.loc	1 164 22                        // sk03_fa_qkv.py:164:22
	shl.b32 	%r4, %r69, 7;
	.loc	1 164 45                        // sk03_fa_qkv.py:164:45
	and.b32 	%r5, %r2, 3;
	shl.b32 	%r85, %r5, 1;
	and.b32 	%r6, %r2, 32;
	shr.u32 	%r86, %r6, 2;
	or.b32 	%r87, %r86, %r85;
	and.b32 	%r7, %r2, 15;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
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
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
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
	.loc	1 167 39                        // sk03_fa_qkv.py:167:39
	mul.lo.s32 	%r116, %r81, %r64;
	mul.lo.s32 	%r117, %r82, %r64;
	mul.lo.s32 	%r118, %r83, %r64;
	mul.lo.s32 	%r119, %r84, %r64;
	.loc	1 167 21                        // sk03_fa_qkv.py:167:21
	cvt.s64.s32 	%rd1, %r116;
	add.s64 	%rd57, %rd32, %rd1;
	cvt.s64.s32 	%rd2, %r117;
	add.s64 	%rd58, %rd32, %rd2;
	cvt.s64.s32 	%rd3, %r118;
	add.s64 	%rd59, %rd32, %rd3;
	cvt.s64.s32 	%rd4, %r119;
	add.s64 	%rd60, %rd32, %rd4;
	.loc	1 167 58                        // sk03_fa_qkv.py:167:58
	shl.b32 	%r120, %r5, 4;
	.loc	1 167 51                        // sk03_fa_qkv.py:167:51
	cvt.u64.u32 	%rd5, %r120;
	add.s64 	%rd38, %rd57, %rd5;
	add.s64 	%rd39, %rd58, %rd5;
	add.s64 	%rd40, %rd59, %rd5;
	add.s64 	%rd41, %rd60, %rd5;
	.loc	1 168 21                        // sk03_fa_qkv.py:168:21
	add.s64 	%rd61, %rd33, %rd5;
	.loc	1 168 69                        // sk03_fa_qkv.py:168:69
	mul.lo.s32 	%r121, %r106, %r66;
	mul.lo.s32 	%r122, %r107, %r66;
	.loc	1 168 51                        // sk03_fa_qkv.py:168:51
	cvt.s64.s32 	%rd6, %r121;
	add.s64 	%rd42, %rd61, %rd6;
	cvt.s64.s32 	%rd7, %r122;
	add.s64 	%rd43, %rd61, %rd7;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	setp.gt.s32 	%p1, %r23, 63;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
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
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	add.s32 	%r33, %r28, 49152;
	// begin inline asm
	cp.async.cg.shared.global [ %r33 + 0 ], [ %rd42 + 0 ], 0x10, %r29;
	// end inline asm
	add.s32 	%r34, %r28, 53248;
	// begin inline asm
	cp.async.cg.shared.global [ %r34 + 0 ], [ %rd43 + 0 ], 0x10, %r29;
	// end inline asm
	cp.async.commit_group;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	setp.gt.s32 	%p2, %r23, 127;
	.loc	1 183 18                        // sk03_fa_qkv.py:183:18
	add.s64 	%rd44, %rd38, 64;
	add.s64 	%rd45, %rd39, 64;
	add.s64 	%rd46, %rd40, 64;
	add.s64 	%rd47, %rd41, 64;
	.loc	1 184 18                        // sk03_fa_qkv.py:184:18
	add.s64 	%rd48, %rd42, 64;
	add.s64 	%rd49, %rd43, 64;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
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
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	add.s32 	%r40, %r28, 57344;
	// begin inline asm
	cp.async.cg.shared.global [ %r40 + 0 ], [ %rd48 + 0 ], 0x10, %r36;
	// end inline asm
	add.s32 	%r41, %r28, 61440;
	// begin inline asm
	cp.async.cg.shared.global [ %r41 + 0 ], [ %rd49 + 0 ], 0x10, %r36;
	// end inline asm
	cp.async.commit_group;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	setp.gt.s32 	%p3, %r23, 191;
	.loc	1 183 18                        // sk03_fa_qkv.py:183:18
	add.s64 	%rd50, %rd38, 128;
	add.s64 	%rd51, %rd39, 128;
	add.s64 	%rd52, %rd40, 128;
	add.s64 	%rd53, %rd41, 128;
	.loc	1 184 18                        // sk03_fa_qkv.py:184:18
	add.s64 	%rd54, %rd42, 128;
	add.s64 	%rd55, %rd43, 128;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
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
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	add.s32 	%r47, %r28, 65536;
	// begin inline asm
	cp.async.cg.shared.global [ %r47 + 0 ], [ %rd54 + 0 ], 0x10, %r43;
	// end inline asm
	add.s32 	%r48, %r28, 69632;
	// begin inline asm
	cp.async.cg.shared.global [ %r48 + 0 ], [ %rd55 + 0 ], 0x10, %r43;
	// end inline asm
	cp.async.commit_group;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	@%p1 bra 	$L__BB0_2;
	bra.uni 	$L__BB0_1;
$L__BB0_2:                              // %.lr.ph
	.loc	1 0 23                          // sk03_fa_qkv.py:0:23
	ld.param.b32 	%r27, [_sk03_fa_qkv_kernel_param_15];
	ld.param.b64 	%rd56, [_sk03_fa_qkv_kernel_param_6];
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
	.loc	1 178 28                        // sk03_fa_qkv.py:178:28
	shr.u32 	%r191, %r23, 6;
	add.s32 	%r192, %r191, -3;
	shl.b32 	%r193, %r7, 6;
	shl.b32 	%r1299, %r2, 4;
	and.b32 	%r194, %r1299, 3072;
	shl.b32 	%r195, %r2, 3;
	and.b32 	%r196, %r195, 48;
	and.b32 	%r1300, %r2, 16;
	or.b32 	%r197, %r193, %r194;
	xor.b32 	%r198, %r196, %r1300;
	or.b32 	%r18, %r197, %r198;
	xor.b32 	%r19, %r18, 32;
	shl.b32 	%r199, %r2, 6;
	and.b32 	%r200, %r199, 448;
	shl.b32 	%r201, %r6, 4;
	or.b32 	%r202, %r200, %r196;
	xor.b32 	%r203, %r202, %r17;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	add.s32 	%r204, %r189, %r201;
	add.s32 	%r20, %r204, %r203;
	cvt.s64.s32 	%rd24, %r192;
	and.b32 	%r205, %r23, -64;
	cvt.u64.u32 	%rd25, %r205;
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
	mov.b32 	%r190, 0;
	mov.b32 	%r1170, 2;
	mov.b32 	%r1169, -1;
	mov.b64 	%rd307, 0;
	mov.b32 	%r1168, %r190;
	mov.b64 	%rd308, %rd307;
	mov.b32 	%r1171, %r190;
	mov.b32 	%r1172, %r190;
	mov.b32 	%r1173, %r190;
	mov.b32 	%r1174, %r190;
	mov.b32 	%r1175, %r190;
	mov.b32 	%r1176, %r190;
	mov.b32 	%r1177, %r190;
	mov.b32 	%r1178, %r190;
	mov.b32 	%r1179, %r190;
	mov.b32 	%r1180, %r190;
	mov.b32 	%r1181, %r190;
	mov.b32 	%r1182, %r190;
	mov.b32 	%r1183, %r190;
	mov.b32 	%r1184, %r190;
	mov.b32 	%r1185, %r190;
	mov.b32 	%r1186, %r190;
	mov.b32 	%r1187, %r190;
	mov.b32 	%r1188, %r190;
	mov.b32 	%r1189, %r190;
	mov.b32 	%r1190, %r190;
	mov.b32 	%r1191, %r190;
	mov.b32 	%r1192, %r190;
	mov.b32 	%r1193, %r190;
	mov.b32 	%r1194, %r190;
	mov.b32 	%r1195, %r190;
	mov.b32 	%r1196, %r190;
	mov.b32 	%r1197, %r190;
	mov.b32 	%r1198, %r190;
	mov.b32 	%r1199, %r190;
	mov.b32 	%r1200, %r190;
	mov.b32 	%r1201, %r190;
	mov.b32 	%r1202, %r190;
	mov.b32 	%r1203, %r190;
	mov.b32 	%r1204, %r190;
	mov.b32 	%r1205, %r190;
	mov.b32 	%r1206, %r190;
	mov.b32 	%r1207, %r190;
	mov.b32 	%r1208, %r190;
	mov.b32 	%r1209, %r190;
	mov.b32 	%r1210, %r190;
	mov.b32 	%r1211, %r190;
	mov.b32 	%r1212, %r190;
	mov.b32 	%r1213, %r190;
	mov.b32 	%r1214, %r190;
	mov.b32 	%r1215, %r190;
	mov.b32 	%r1216, %r190;
	mov.b32 	%r1217, %r190;
	mov.b32 	%r1218, %r190;
	mov.b32 	%r1219, %r190;
	mov.b32 	%r1220, %r190;
	mov.b32 	%r1221, %r190;
	mov.b32 	%r1222, %r190;
	mov.b32 	%r1223, %r190;
	mov.b32 	%r1224, %r190;
	mov.b32 	%r1225, %r190;
	mov.b32 	%r1226, %r190;
	mov.b32 	%r1227, %r190;
	mov.b32 	%r1228, %r190;
	mov.b32 	%r1229, %r190;
	mov.b32 	%r1230, %r190;
	mov.b32 	%r1231, %r190;
	mov.b32 	%r1232, %r190;
	mov.b32 	%r1233, %r190;
	mov.b32 	%r1234, %r190;
	mov.b32 	%r1235, %r190;
	mov.b32 	%r1236, %r190;
	mov.b32 	%r1237, %r190;
	mov.b32 	%r1238, %r190;
	mov.b32 	%r1239, %r190;
	mov.b32 	%r1240, %r190;
	mov.b32 	%r1241, %r190;
	mov.b32 	%r1242, %r190;
	mov.b32 	%r1243, %r190;
	mov.b32 	%r1244, %r190;
	mov.b32 	%r1245, %r190;
	mov.b32 	%r1246, %r190;
	mov.b32 	%r1247, %r190;
	mov.b32 	%r1248, %r190;
	mov.b32 	%r1249, %r190;
	mov.b32 	%r1250, %r190;
	mov.b32 	%r1251, %r190;
	mov.b32 	%r1252, %r190;
	mov.b32 	%r1253, %r190;
	mov.b32 	%r1254, %r190;
	mov.b32 	%r1255, %r190;
	mov.b32 	%r1256, %r190;
	mov.b32 	%r1257, %r190;
	mov.b32 	%r1258, %r190;
	mov.b32 	%r1259, %r190;
	mov.b32 	%r1260, %r190;
	mov.b32 	%r1261, %r190;
	mov.b32 	%r1262, %r190;
	mov.b32 	%r1263, %r190;
	mov.b32 	%r1264, %r190;
	mov.b32 	%r1265, %r190;
	mov.b32 	%r1266, %r190;
	mov.b32 	%r1267, %r190;
	mov.b32 	%r1268, %r190;
	mov.b32 	%r1269, %r190;
	mov.b32 	%r1270, %r190;
	mov.b32 	%r1271, %r190;
	mov.b32 	%r1272, %r190;
	mov.b32 	%r1273, %r190;
	mov.b32 	%r1274, %r190;
	mov.b32 	%r1275, %r190;
	mov.b32 	%r1276, %r190;
	mov.b32 	%r1277, %r190;
	mov.b32 	%r1278, %r190;
	mov.b32 	%r1279, %r190;
	mov.b32 	%r1280, %r190;
	mov.b32 	%r1281, %r190;
	mov.b32 	%r1282, %r190;
	mov.b32 	%r1283, %r190;
	mov.b32 	%r1284, %r190;
	mov.b32 	%r1285, %r190;
	mov.b32 	%r1286, %r190;
	mov.b32 	%r1287, %r190;
	mov.b32 	%r1288, %r190;
	mov.b32 	%r1289, %r190;
	mov.b32 	%r1290, %r190;
	mov.b32 	%r1291, %r190;
	mov.b32 	%r1292, %r190;
	mov.b32 	%r1293, %r190;
	mov.b32 	%r1294, %r190;
	mov.b32 	%r1295, %r190;
	mov.b32 	%r1296, %r190;
	mov.b32 	%r1297, %r190;
	mov.b32 	%r1298, %r190;
$L__BB0_3:                              // =>This Inner Loop Header: Depth=1
	setp.lt.s64 	%p4, %rd308, %rd24;
	add.s32 	%r405, %r1169, 1;
	setp.gt.s32 	%p5, %r405, 2;
	selp.b32 	%r1169, 0, %r405, %p5;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
	cp.async.wait_group 	4;
	bar.sync 	0;
	shl.b32 	%r406, %r1169, 14;
	add.s32 	%r407, %r189, %r406;
	add.s32 	%r408, %r407, %r18;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r206, %r207, %r208, %r209}, [%r408];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r226, %r227, %r228, %r229}, [%r408+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r230, %r231, %r232, %r233}, [%r408+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r234, %r235, %r236, %r237}, [%r408+12288];
	add.s32 	%r409, %r407, %r19;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r242, %r243, %r244, %r245}, [%r409];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r294, %r295, %r296, %r297}, [%r409+4096];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r330, %r331, %r332, %r333}, [%r409+8192];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r366, %r367, %r368, %r369}, [%r409+12288];
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	shl.b32 	%r410, %r1169, 13;
	add.s32 	%r411, %r20, %r410;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r210, %r211, %r246, %r247}, [%r411+49152];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r212, %r213, %r252, %r253}, [%r411+50176];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r214, %r215, %r258, %r259}, [%r411+51200];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r216, %r217, %r264, %r265}, [%r411+52224];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r218, %r219, %r270, %r271}, [%r411+53248];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r220, %r221, %r276, %r277}, [%r411+54272];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r222, %r223, %r282, %r283}, [%r411+55296];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r224, %r225, %r288, %r289}, [%r411+56320];
	.loc	1 179 36                        // sk03_fa_qkv.py:179:36
	mov.b32 	%r238, %r190;
	mov.b32 	%r239, %r190;
	mov.b32 	%r240, %r190;
	mov.b32 	%r241, %r190;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r238, %r239, %r240, %r241 }, { %r206, %r207, %r208, %r209 }, { %r210, %r211 }, { %r238, %r239, %r240, %r241 };
	// end inline asm
	mov.b32 	%r248, %r190;
	mov.b32 	%r249, %r190;
	mov.b32 	%r250, %r190;
	mov.b32 	%r251, %r190;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r248, %r249, %r250, %r251 }, { %r206, %r207, %r208, %r209 }, { %r212, %r213 }, { %r248, %r249, %r250, %r251 };
	// end inline asm
	mov.b32 	%r254, %r190;
	mov.b32 	%r255, %r190;
	mov.b32 	%r256, %r190;
	mov.b32 	%r257, %r190;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r254, %r255, %r256, %r257 }, { %r206, %r207, %r208, %r209 }, { %r214, %r215 }, { %r254, %r255, %r256, %r257 };
	// end inline asm
	mov.b32 	%r260, %r190;
	mov.b32 	%r261, %r190;
	mov.b32 	%r262, %r190;
	mov.b32 	%r263, %r190;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r260, %r261, %r262, %r263 }, { %r206, %r207, %r208, %r209 }, { %r216, %r217 }, { %r260, %r261, %r262, %r263 };
	// end inline asm
	mov.b32 	%r266, %r190;
	mov.b32 	%r267, %r190;
	mov.b32 	%r268, %r190;
	mov.b32 	%r269, %r190;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r266, %r267, %r268, %r269 }, { %r206, %r207, %r208, %r209 }, { %r218, %r219 }, { %r266, %r267, %r268, %r269 };
	// end inline asm
	mov.b32 	%r272, %r190;
	mov.b32 	%r273, %r190;
	mov.b32 	%r274, %r190;
	mov.b32 	%r275, %r190;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r272, %r273, %r274, %r275 }, { %r206, %r207, %r208, %r209 }, { %r220, %r221 }, { %r272, %r273, %r274, %r275 };
	// end inline asm
	mov.b32 	%r278, %r190;
	mov.b32 	%r279, %r190;
	mov.b32 	%r280, %r190;
	mov.b32 	%r281, %r190;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r278, %r279, %r280, %r281 }, { %r206, %r207, %r208, %r209 }, { %r222, %r223 }, { %r278, %r279, %r280, %r281 };
	// end inline asm
	mov.b32 	%r284, %r190;
	mov.b32 	%r285, %r190;
	mov.b32 	%r286, %r190;
	mov.b32 	%r287, %r190;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r284, %r285, %r286, %r287 }, { %r206, %r207, %r208, %r209 }, { %r224, %r225 }, { %r284, %r285, %r286, %r287 };
	// end inline asm
	mov.b32 	%r290, %r190;
	mov.b32 	%r291, %r190;
	mov.b32 	%r292, %r190;
	mov.b32 	%r293, %r190;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r290, %r291, %r292, %r293 }, { %r226, %r227, %r228, %r229 }, { %r210, %r211 }, { %r290, %r291, %r292, %r293 };
	// end inline asm
	mov.b32 	%r298, %r190;
	mov.b32 	%r299, %r190;
	mov.b32 	%r300, %r190;
	mov.b32 	%r301, %r190;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r298, %r299, %r300, %r301 }, { %r226, %r227, %r228, %r229 }, { %r212, %r213 }, { %r298, %r299, %r300, %r301 };
	// end inline asm
	mov.b32 	%r302, %r190;
	mov.b32 	%r303, %r190;
	mov.b32 	%r304, %r190;
	mov.b32 	%r305, %r190;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r302, %r303, %r304, %r305 }, { %r226, %r227, %r228, %r229 }, { %r214, %r215 }, { %r302, %r303, %r304, %r305 };
	// end inline asm
	mov.b32 	%r306, %r190;
	mov.b32 	%r307, %r190;
	mov.b32 	%r308, %r190;
	mov.b32 	%r309, %r190;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r306, %r307, %r308, %r309 }, { %r226, %r227, %r228, %r229 }, { %r216, %r217 }, { %r306, %r307, %r308, %r309 };
	// end inline asm
	mov.b32 	%r310, %r190;
	mov.b32 	%r311, %r190;
	mov.b32 	%r312, %r190;
	mov.b32 	%r313, %r190;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r310, %r311, %r312, %r313 }, { %r226, %r227, %r228, %r229 }, { %r218, %r219 }, { %r310, %r311, %r312, %r313 };
	// end inline asm
	mov.b32 	%r314, %r190;
	mov.b32 	%r315, %r190;
	mov.b32 	%r316, %r190;
	mov.b32 	%r317, %r190;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r314, %r315, %r316, %r317 }, { %r226, %r227, %r228, %r229 }, { %r220, %r221 }, { %r314, %r315, %r316, %r317 };
	// end inline asm
	mov.b32 	%r318, %r190;
	mov.b32 	%r319, %r190;
	mov.b32 	%r320, %r190;
	mov.b32 	%r321, %r190;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r318, %r319, %r320, %r321 }, { %r226, %r227, %r228, %r229 }, { %r222, %r223 }, { %r318, %r319, %r320, %r321 };
	// end inline asm
	mov.b32 	%r322, %r190;
	mov.b32 	%r323, %r190;
	mov.b32 	%r324, %r190;
	mov.b32 	%r325, %r190;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r322, %r323, %r324, %r325 }, { %r226, %r227, %r228, %r229 }, { %r224, %r225 }, { %r322, %r323, %r324, %r325 };
	// end inline asm
	mov.b32 	%r326, %r190;
	mov.b32 	%r327, %r190;
	mov.b32 	%r328, %r190;
	mov.b32 	%r329, %r190;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r326, %r327, %r328, %r329 }, { %r230, %r231, %r232, %r233 }, { %r210, %r211 }, { %r326, %r327, %r328, %r329 };
	// end inline asm
	mov.b32 	%r334, %r190;
	mov.b32 	%r335, %r190;
	mov.b32 	%r336, %r190;
	mov.b32 	%r337, %r190;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r334, %r335, %r336, %r337 }, { %r230, %r231, %r232, %r233 }, { %r212, %r213 }, { %r334, %r335, %r336, %r337 };
	// end inline asm
	mov.b32 	%r338, %r190;
	mov.b32 	%r339, %r190;
	mov.b32 	%r340, %r190;
	mov.b32 	%r341, %r190;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r338, %r339, %r340, %r341 }, { %r230, %r231, %r232, %r233 }, { %r214, %r215 }, { %r338, %r339, %r340, %r341 };
	// end inline asm
	mov.b32 	%r342, %r190;
	mov.b32 	%r343, %r190;
	mov.b32 	%r344, %r190;
	mov.b32 	%r345, %r190;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r342, %r343, %r344, %r345 }, { %r230, %r231, %r232, %r233 }, { %r216, %r217 }, { %r342, %r343, %r344, %r345 };
	// end inline asm
	mov.b32 	%r346, %r190;
	mov.b32 	%r347, %r190;
	mov.b32 	%r348, %r190;
	mov.b32 	%r349, %r190;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r346, %r347, %r348, %r349 }, { %r230, %r231, %r232, %r233 }, { %r218, %r219 }, { %r346, %r347, %r348, %r349 };
	// end inline asm
	mov.b32 	%r350, %r190;
	mov.b32 	%r351, %r190;
	mov.b32 	%r352, %r190;
	mov.b32 	%r353, %r190;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r350, %r351, %r352, %r353 }, { %r230, %r231, %r232, %r233 }, { %r220, %r221 }, { %r350, %r351, %r352, %r353 };
	// end inline asm
	mov.b32 	%r354, %r190;
	mov.b32 	%r355, %r190;
	mov.b32 	%r356, %r190;
	mov.b32 	%r357, %r190;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r354, %r355, %r356, %r357 }, { %r230, %r231, %r232, %r233 }, { %r222, %r223 }, { %r354, %r355, %r356, %r357 };
	// end inline asm
	mov.b32 	%r358, %r190;
	mov.b32 	%r359, %r190;
	mov.b32 	%r360, %r190;
	mov.b32 	%r361, %r190;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r358, %r359, %r360, %r361 }, { %r230, %r231, %r232, %r233 }, { %r224, %r225 }, { %r358, %r359, %r360, %r361 };
	// end inline asm
	mov.b32 	%r362, %r190;
	mov.b32 	%r363, %r190;
	mov.b32 	%r364, %r190;
	mov.b32 	%r365, %r190;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r362, %r363, %r364, %r365 }, { %r234, %r235, %r236, %r237 }, { %r210, %r211 }, { %r362, %r363, %r364, %r365 };
	// end inline asm
	mov.b32 	%r370, %r190;
	mov.b32 	%r371, %r190;
	mov.b32 	%r372, %r190;
	mov.b32 	%r373, %r190;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r370, %r371, %r372, %r373 }, { %r234, %r235, %r236, %r237 }, { %r212, %r213 }, { %r370, %r371, %r372, %r373 };
	// end inline asm
	mov.b32 	%r374, %r190;
	mov.b32 	%r375, %r190;
	mov.b32 	%r376, %r190;
	mov.b32 	%r377, %r190;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r374, %r375, %r376, %r377 }, { %r234, %r235, %r236, %r237 }, { %r214, %r215 }, { %r374, %r375, %r376, %r377 };
	// end inline asm
	mov.b32 	%r378, %r190;
	mov.b32 	%r379, %r190;
	mov.b32 	%r380, %r190;
	mov.b32 	%r381, %r190;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r378, %r379, %r380, %r381 }, { %r234, %r235, %r236, %r237 }, { %r216, %r217 }, { %r378, %r379, %r380, %r381 };
	// end inline asm
	mov.b32 	%r382, %r190;
	mov.b32 	%r383, %r190;
	mov.b32 	%r384, %r190;
	mov.b32 	%r385, %r190;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r382, %r383, %r384, %r385 }, { %r234, %r235, %r236, %r237 }, { %r218, %r219 }, { %r382, %r383, %r384, %r385 };
	// end inline asm
	mov.b32 	%r386, %r190;
	mov.b32 	%r387, %r190;
	mov.b32 	%r388, %r190;
	mov.b32 	%r389, %r190;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r386, %r387, %r388, %r389 }, { %r234, %r235, %r236, %r237 }, { %r220, %r221 }, { %r386, %r387, %r388, %r389 };
	// end inline asm
	mov.b32 	%r390, %r190;
	mov.b32 	%r391, %r190;
	mov.b32 	%r392, %r190;
	mov.b32 	%r393, %r190;
	// begin inline asm
	mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 { %r390, %r391, %r392, %r393 }, { %r234, %r235, %r236, %r237 }, { %r222, %r223 }, { %r390, %r391, %r392, %r393 };
	// end inline asm
	mov.b32 	%r397, %r190;
	mov.b32 	%r394, %r190;
	mov.b32 	%r395, %r190;
	mov.b32 	%r396, %r190;
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
	.loc	1 181 39                        // sk03_fa_qkv.py:181:39
	cvt.s64.s32 	%rd112, %r1168;
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
	.loc	1 181 29                        // sk03_fa_qkv.py:181:29
	// begin inline asm
	mov.u16 %rs1, 0x0;
	ld.global.b8 { %rs1 }, [ %rd90 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs2, 0x0;
	ld.global.b8 { %rs2 }, [ %rd91 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs3, 0x0;
	ld.global.b8 { %rs3 }, [ %rd92 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs4, 0x0;
	ld.global.b8 { %rs4 }, [ %rd93 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs5, 0x0;
	ld.global.b8 { %rs5 }, [ %rd94 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs6, 0x0;
	ld.global.b8 { %rs6 }, [ %rd95 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs7, 0x0;
	ld.global.b8 { %rs7 }, [ %rd96 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs8, 0x0;
	ld.global.b8 { %rs8 }, [ %rd97 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs9, 0x0;
	ld.global.b8 { %rs9 }, [ %rd98 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs10, 0x0;
	ld.global.b8 { %rs10 }, [ %rd99 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs11, 0x0;
	ld.global.b8 { %rs11 }, [ %rd100 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs12, 0x0;
	ld.global.b8 { %rs12 }, [ %rd101 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs13, 0x0;
	ld.global.b8 { %rs13 }, [ %rd102 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs14, 0x0;
	ld.global.b8 { %rs14 }, [ %rd103 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs15, 0x0;
	ld.global.b8 { %rs15 }, [ %rd104 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs16, 0x0;
	ld.global.b8 { %rs16 }, [ %rd105 + 0 ];
	// end inline asm
	.loc	1 181 21                        // sk03_fa_qkv.py:181:21
	cvt.u32.u16 	%r412, %rs1;
	and.b32 	%r413, %r412, 255;
	cvt.u32.u16 	%r414, %rs2;
	and.b32 	%r415, %r414, 255;
	cvt.u32.u16 	%r416, %rs3;
	and.b32 	%r417, %r416, 255;
	cvt.u32.u16 	%r418, %rs4;
	and.b32 	%r419, %r418, 255;
	cvt.u32.u16 	%r420, %rs5;
	and.b32 	%r421, %r420, 255;
	cvt.u32.u16 	%r422, %rs6;
	and.b32 	%r423, %r422, 255;
	cvt.u32.u16 	%r424, %rs7;
	and.b32 	%r425, %r424, 255;
	cvt.u32.u16 	%r426, %rs8;
	and.b32 	%r427, %r426, 255;
	cvt.u32.u16 	%r428, %rs9;
	and.b32 	%r429, %r428, 255;
	cvt.u32.u16 	%r430, %rs10;
	and.b32 	%r431, %r430, 255;
	cvt.u32.u16 	%r432, %rs11;
	and.b32 	%r433, %r432, 255;
	cvt.u32.u16 	%r434, %rs12;
	and.b32 	%r435, %r434, 255;
	cvt.u32.u16 	%r436, %rs13;
	and.b32 	%r437, %r436, 255;
	cvt.u32.u16 	%r438, %rs14;
	and.b32 	%r439, %r438, 255;
	cvt.u32.u16 	%r440, %rs15;
	and.b32 	%r441, %r440, 255;
	cvt.u32.u16 	%r442, %rs16;
	and.b32 	%r443, %r442, 255;
	shl.b32 	%r444, %r397, %r443;
	shl.b32 	%r445, %r396, %r441;
	shl.b32 	%r446, %r395, %r443;
	shl.b32 	%r447, %r394, %r441;
	shl.b32 	%r448, %r393, %r439;
	shl.b32 	%r449, %r392, %r437;
	shl.b32 	%r450, %r391, %r439;
	shl.b32 	%r451, %r390, %r437;
	shl.b32 	%r452, %r389, %r435;
	shl.b32 	%r453, %r388, %r433;
	shl.b32 	%r454, %r387, %r435;
	shl.b32 	%r455, %r386, %r433;
	shl.b32 	%r456, %r385, %r431;
	shl.b32 	%r457, %r384, %r429;
	shl.b32 	%r458, %r383, %r431;
	shl.b32 	%r459, %r382, %r429;
	shl.b32 	%r460, %r381, %r427;
	shl.b32 	%r461, %r380, %r425;
	shl.b32 	%r462, %r379, %r427;
	shl.b32 	%r463, %r378, %r425;
	shl.b32 	%r464, %r377, %r423;
	shl.b32 	%r465, %r376, %r421;
	shl.b32 	%r466, %r285, %r443;
	shl.b32 	%r467, %r287, %r443;
	shl.b32 	%r468, %r323, %r443;
	shl.b32 	%r469, %r325, %r443;
	shl.b32 	%r470, %r359, %r443;
	shl.b32 	%r471, %r361, %r443;
	shl.b32 	%r472, %r375, %r423;
	shl.b32 	%r473, %r284, %r441;
	shl.b32 	%r474, %r286, %r441;
	shl.b32 	%r475, %r322, %r441;
	shl.b32 	%r476, %r324, %r441;
	shl.b32 	%r477, %r358, %r441;
	shl.b32 	%r478, %r360, %r441;
	shl.b32 	%r479, %r374, %r421;
	shl.b32 	%r480, %r279, %r439;
	shl.b32 	%r481, %r281, %r439;
	shl.b32 	%r482, %r319, %r439;
	shl.b32 	%r483, %r321, %r439;
	shl.b32 	%r484, %r355, %r439;
	shl.b32 	%r485, %r357, %r439;
	shl.b32 	%r486, %r373, %r419;
	shl.b32 	%r487, %r278, %r437;
	shl.b32 	%r488, %r280, %r437;
	shl.b32 	%r489, %r318, %r437;
	shl.b32 	%r490, %r320, %r437;
	shl.b32 	%r491, %r354, %r437;
	shl.b32 	%r492, %r356, %r437;
	shl.b32 	%r493, %r372, %r417;
	shl.b32 	%r494, %r273, %r435;
	shl.b32 	%r495, %r275, %r435;
	shl.b32 	%r496, %r315, %r435;
	shl.b32 	%r497, %r317, %r435;
	shl.b32 	%r498, %r351, %r435;
	shl.b32 	%r499, %r353, %r435;
	shl.b32 	%r500, %r371, %r419;
	shl.b32 	%r501, %r272, %r433;
	shl.b32 	%r502, %r274, %r433;
	shl.b32 	%r503, %r314, %r433;
	shl.b32 	%r504, %r316, %r433;
	shl.b32 	%r505, %r350, %r433;
	shl.b32 	%r506, %r352, %r433;
	shl.b32 	%r507, %r370, %r417;
	shl.b32 	%r508, %r267, %r431;
	shl.b32 	%r509, %r269, %r431;
	shl.b32 	%r510, %r311, %r431;
	shl.b32 	%r511, %r313, %r431;
	shl.b32 	%r512, %r347, %r431;
	shl.b32 	%r513, %r349, %r431;
	shl.b32 	%r514, %r365, %r415;
	shl.b32 	%r515, %r266, %r429;
	shl.b32 	%r516, %r268, %r429;
	shl.b32 	%r517, %r310, %r429;
	shl.b32 	%r518, %r312, %r429;
	shl.b32 	%r519, %r346, %r429;
	shl.b32 	%r520, %r348, %r429;
	shl.b32 	%r521, %r364, %r413;
	shl.b32 	%r522, %r261, %r427;
	shl.b32 	%r523, %r263, %r427;
	shl.b32 	%r524, %r307, %r427;
	shl.b32 	%r525, %r309, %r427;
	shl.b32 	%r526, %r343, %r427;
	shl.b32 	%r527, %r345, %r427;
	shl.b32 	%r528, %r363, %r415;
	shl.b32 	%r529, %r260, %r425;
	shl.b32 	%r530, %r262, %r425;
	shl.b32 	%r531, %r306, %r425;
	shl.b32 	%r532, %r308, %r425;
	shl.b32 	%r533, %r342, %r425;
	shl.b32 	%r534, %r344, %r425;
	shl.b32 	%r535, %r362, %r413;
	shl.b32 	%r536, %r255, %r423;
	shl.b32 	%r537, %r257, %r423;
	shl.b32 	%r538, %r303, %r423;
	shl.b32 	%r539, %r305, %r423;
	shl.b32 	%r540, %r339, %r423;
	shl.b32 	%r541, %r341, %r423;
	shl.b32 	%r542, %r254, %r421;
	shl.b32 	%r543, %r256, %r421;
	shl.b32 	%r544, %r302, %r421;
	shl.b32 	%r545, %r304, %r421;
	shl.b32 	%r546, %r338, %r421;
	shl.b32 	%r547, %r340, %r421;
	shl.b32 	%r548, %r249, %r419;
	shl.b32 	%r549, %r251, %r419;
	shl.b32 	%r550, %r299, %r419;
	shl.b32 	%r551, %r301, %r419;
	shl.b32 	%r552, %r335, %r419;
	shl.b32 	%r553, %r337, %r419;
	shl.b32 	%r554, %r248, %r417;
	shl.b32 	%r555, %r250, %r417;
	shl.b32 	%r556, %r298, %r417;
	shl.b32 	%r557, %r300, %r417;
	shl.b32 	%r558, %r334, %r417;
	shl.b32 	%r559, %r336, %r417;
	shl.b32 	%r560, %r239, %r415;
	shl.b32 	%r561, %r241, %r415;
	shl.b32 	%r562, %r291, %r415;
	shl.b32 	%r563, %r293, %r415;
	shl.b32 	%r564, %r327, %r415;
	shl.b32 	%r565, %r329, %r415;
	shl.b32 	%r566, %r238, %r413;
	shl.b32 	%r567, %r240, %r413;
	shl.b32 	%r568, %r290, %r413;
	shl.b32 	%r569, %r292, %r413;
	shl.b32 	%r570, %r326, %r413;
	shl.b32 	%r571, %r328, %r413;
	.loc	1 182 15                        // sk03_fa_qkv.py:182:15
	add.s32 	%r1237, %r571, %r1237;
	add.s32 	%r1235, %r570, %r1235;
	add.s32 	%r1205, %r569, %r1205;
	add.s32 	%r1203, %r568, %r1203;
	add.s32 	%r1173, %r567, %r1173;
	add.s32 	%r1171, %r566, %r1171;
	add.s32 	%r1238, %r565, %r1238;
	add.s32 	%r1236, %r564, %r1236;
	add.s32 	%r1206, %r563, %r1206;
	add.s32 	%r1204, %r562, %r1204;
	add.s32 	%r1174, %r561, %r1174;
	add.s32 	%r1172, %r560, %r1172;
	add.s32 	%r1241, %r559, %r1241;
	add.s32 	%r1239, %r558, %r1239;
	add.s32 	%r1209, %r557, %r1209;
	add.s32 	%r1207, %r556, %r1207;
	add.s32 	%r1177, %r555, %r1177;
	add.s32 	%r1175, %r554, %r1175;
	add.s32 	%r1242, %r553, %r1242;
	add.s32 	%r1240, %r552, %r1240;
	add.s32 	%r1210, %r551, %r1210;
	add.s32 	%r1208, %r550, %r1208;
	add.s32 	%r1178, %r549, %r1178;
	add.s32 	%r1176, %r548, %r1176;
	add.s32 	%r1245, %r547, %r1245;
	add.s32 	%r1243, %r546, %r1243;
	add.s32 	%r1213, %r545, %r1213;
	add.s32 	%r1211, %r544, %r1211;
	add.s32 	%r1181, %r543, %r1181;
	add.s32 	%r1179, %r542, %r1179;
	add.s32 	%r1246, %r541, %r1246;
	add.s32 	%r1244, %r540, %r1244;
	add.s32 	%r1214, %r539, %r1214;
	add.s32 	%r1212, %r538, %r1212;
	add.s32 	%r1182, %r537, %r1182;
	add.s32 	%r1180, %r536, %r1180;
	add.s32 	%r1267, %r535, %r1267;
	add.s32 	%r1249, %r534, %r1249;
	add.s32 	%r1247, %r533, %r1247;
	add.s32 	%r1217, %r532, %r1217;
	add.s32 	%r1215, %r531, %r1215;
	add.s32 	%r1185, %r530, %r1185;
	add.s32 	%r1183, %r529, %r1183;
	add.s32 	%r1268, %r528, %r1268;
	add.s32 	%r1250, %r527, %r1250;
	add.s32 	%r1248, %r526, %r1248;
	add.s32 	%r1218, %r525, %r1218;
	add.s32 	%r1216, %r524, %r1216;
	add.s32 	%r1186, %r523, %r1186;
	add.s32 	%r1184, %r522, %r1184;
	add.s32 	%r1269, %r521, %r1269;
	add.s32 	%r1253, %r520, %r1253;
	add.s32 	%r1251, %r519, %r1251;
	add.s32 	%r1221, %r518, %r1221;
	add.s32 	%r1219, %r517, %r1219;
	add.s32 	%r1189, %r516, %r1189;
	add.s32 	%r1187, %r515, %r1187;
	add.s32 	%r1270, %r514, %r1270;
	add.s32 	%r1254, %r513, %r1254;
	add.s32 	%r1252, %r512, %r1252;
	add.s32 	%r1222, %r511, %r1222;
	add.s32 	%r1220, %r510, %r1220;
	add.s32 	%r1190, %r509, %r1190;
	add.s32 	%r1188, %r508, %r1188;
	add.s32 	%r1271, %r507, %r1271;
	add.s32 	%r1257, %r506, %r1257;
	add.s32 	%r1255, %r505, %r1255;
	add.s32 	%r1225, %r504, %r1225;
	add.s32 	%r1223, %r503, %r1223;
	add.s32 	%r1193, %r502, %r1193;
	add.s32 	%r1191, %r501, %r1191;
	add.s32 	%r1272, %r500, %r1272;
	add.s32 	%r1258, %r499, %r1258;
	add.s32 	%r1256, %r498, %r1256;
	add.s32 	%r1226, %r497, %r1226;
	add.s32 	%r1224, %r496, %r1224;
	add.s32 	%r1194, %r495, %r1194;
	add.s32 	%r1192, %r494, %r1192;
	add.s32 	%r1273, %r493, %r1273;
	add.s32 	%r1261, %r492, %r1261;
	add.s32 	%r1259, %r491, %r1259;
	add.s32 	%r1229, %r490, %r1229;
	add.s32 	%r1227, %r489, %r1227;
	add.s32 	%r1197, %r488, %r1197;
	add.s32 	%r1195, %r487, %r1195;
	add.s32 	%r1274, %r486, %r1274;
	add.s32 	%r1262, %r485, %r1262;
	add.s32 	%r1260, %r484, %r1260;
	add.s32 	%r1230, %r483, %r1230;
	add.s32 	%r1228, %r482, %r1228;
	add.s32 	%r1198, %r481, %r1198;
	add.s32 	%r1196, %r480, %r1196;
	add.s32 	%r1275, %r479, %r1275;
	add.s32 	%r1265, %r478, %r1265;
	add.s32 	%r1263, %r477, %r1263;
	add.s32 	%r1233, %r476, %r1233;
	add.s32 	%r1231, %r475, %r1231;
	add.s32 	%r1201, %r474, %r1201;
	add.s32 	%r1199, %r473, %r1199;
	add.s32 	%r1276, %r472, %r1276;
	add.s32 	%r1266, %r471, %r1266;
	add.s32 	%r1264, %r470, %r1264;
	add.s32 	%r1234, %r469, %r1234;
	add.s32 	%r1232, %r468, %r1232;
	add.s32 	%r1202, %r467, %r1202;
	add.s32 	%r1200, %r466, %r1200;
	add.s32 	%r1277, %r465, %r1277;
	add.s32 	%r1278, %r464, %r1278;
	add.s32 	%r1279, %r463, %r1279;
	add.s32 	%r1280, %r462, %r1280;
	add.s32 	%r1281, %r461, %r1281;
	add.s32 	%r1282, %r460, %r1282;
	add.s32 	%r1283, %r459, %r1283;
	add.s32 	%r1284, %r458, %r1284;
	add.s32 	%r1285, %r457, %r1285;
	add.s32 	%r1286, %r456, %r1286;
	add.s32 	%r1287, %r455, %r1287;
	add.s32 	%r1288, %r454, %r1288;
	add.s32 	%r1289, %r453, %r1289;
	add.s32 	%r1290, %r452, %r1290;
	add.s32 	%r1291, %r451, %r1291;
	add.s32 	%r1292, %r450, %r1292;
	add.s32 	%r1293, %r449, %r1293;
	add.s32 	%r1294, %r448, %r1294;
	add.s32 	%r1295, %r447, %r1295;
	add.s32 	%r1296, %r446, %r1296;
	add.s32 	%r1297, %r445, %r1297;
	add.s32 	%r1298, %r444, %r1298;
	.loc	1 183 18                        // sk03_fa_qkv.py:183:18
	add.s64 	%rd106, %rd31, %rd307;
	add.s64 	%rd107, %rd30, %rd307;
	add.s64 	%rd108, %rd29, %rd307;
	.loc	1 184 18                        // sk03_fa_qkv.py:184:18
	add.s64 	%rd109, %rd28, %rd307;
	add.s64 	%rd110, %rd27, %rd307;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	add.s64 	%rd111, %rd26, %rd307;
	add.s32 	%r572, %r1170, 1;
	setp.gt.s32 	%p6, %r572, 2;
	selp.b32 	%r1170, 0, %r572, %p6;
	.loc	1 179 27                        // sk03_fa_qkv.py:179:27
	shl.b32 	%r573, %r1170, 14;
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
	.loc	1 179 44                        // sk03_fa_qkv.py:179:44
	shl.b32 	%r574, %r1170, 13;
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
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	add.s64 	%rd308, %rd308, 1;
	add.s64 	%rd307, %rd307, 64;
	add.s32 	%r1168, %r1168, %r27;
	setp.ne.b64 	%p7, %rd25, %rd307;
	@%p7 bra 	$L__BB0_3;
// %bb.4:                               // %._crit_edge.loopexit
	.loc	1 186 17                        // sk03_fa_qkv.py:186:17
	cvt.rn.f32.s32 	%r576, %r1171;
	cvt.rn.f32.s32 	%r577, %r1172;
	cvt.rn.bf16x2.f32 	%r1301, %r577, %r576;
	cvt.rn.f32.s32 	%r578, %r1173;
	cvt.rn.f32.s32 	%r579, %r1174;
	cvt.rn.bf16x2.f32 	%r1302, %r579, %r578;
	cvt.rn.f32.s32 	%r580, %r1175;
	cvt.rn.f32.s32 	%r581, %r1176;
	cvt.rn.bf16x2.f32 	%r1303, %r581, %r580;
	cvt.rn.f32.s32 	%r582, %r1177;
	cvt.rn.f32.s32 	%r583, %r1178;
	cvt.rn.bf16x2.f32 	%r1304, %r583, %r582;
	cvt.rn.f32.s32 	%r584, %r1179;
	cvt.rn.f32.s32 	%r585, %r1180;
	cvt.rn.bf16x2.f32 	%r1305, %r585, %r584;
	cvt.rn.f32.s32 	%r586, %r1181;
	cvt.rn.f32.s32 	%r587, %r1182;
	cvt.rn.bf16x2.f32 	%r1306, %r587, %r586;
	cvt.rn.f32.s32 	%r588, %r1183;
	cvt.rn.f32.s32 	%r589, %r1184;
	cvt.rn.bf16x2.f32 	%r1307, %r589, %r588;
	cvt.rn.f32.s32 	%r590, %r1185;
	cvt.rn.f32.s32 	%r591, %r1186;
	cvt.rn.bf16x2.f32 	%r1308, %r591, %r590;
	cvt.rn.f32.s32 	%r592, %r1187;
	cvt.rn.f32.s32 	%r593, %r1188;
	cvt.rn.bf16x2.f32 	%r1309, %r593, %r592;
	cvt.rn.f32.s32 	%r594, %r1189;
	cvt.rn.f32.s32 	%r595, %r1190;
	cvt.rn.bf16x2.f32 	%r1310, %r595, %r594;
	cvt.rn.f32.s32 	%r596, %r1191;
	cvt.rn.f32.s32 	%r597, %r1192;
	cvt.rn.bf16x2.f32 	%r1311, %r597, %r596;
	cvt.rn.f32.s32 	%r598, %r1193;
	cvt.rn.f32.s32 	%r599, %r1194;
	cvt.rn.bf16x2.f32 	%r1312, %r599, %r598;
	cvt.rn.f32.s32 	%r600, %r1195;
	cvt.rn.f32.s32 	%r601, %r1196;
	cvt.rn.bf16x2.f32 	%r1313, %r601, %r600;
	cvt.rn.f32.s32 	%r602, %r1197;
	cvt.rn.f32.s32 	%r603, %r1198;
	cvt.rn.bf16x2.f32 	%r1314, %r603, %r602;
	cvt.rn.f32.s32 	%r604, %r1199;
	cvt.rn.f32.s32 	%r605, %r1200;
	cvt.rn.bf16x2.f32 	%r1315, %r605, %r604;
	cvt.rn.f32.s32 	%r606, %r1201;
	cvt.rn.f32.s32 	%r607, %r1202;
	cvt.rn.bf16x2.f32 	%r1316, %r607, %r606;
	cvt.rn.f32.s32 	%r608, %r1203;
	cvt.rn.f32.s32 	%r609, %r1204;
	cvt.rn.bf16x2.f32 	%r1317, %r609, %r608;
	cvt.rn.f32.s32 	%r610, %r1205;
	cvt.rn.f32.s32 	%r611, %r1206;
	cvt.rn.bf16x2.f32 	%r1318, %r611, %r610;
	cvt.rn.f32.s32 	%r612, %r1207;
	cvt.rn.f32.s32 	%r613, %r1208;
	cvt.rn.bf16x2.f32 	%r1319, %r613, %r612;
	cvt.rn.f32.s32 	%r614, %r1209;
	cvt.rn.f32.s32 	%r615, %r1210;
	cvt.rn.bf16x2.f32 	%r1320, %r615, %r614;
	cvt.rn.f32.s32 	%r616, %r1211;
	cvt.rn.f32.s32 	%r617, %r1212;
	cvt.rn.bf16x2.f32 	%r1321, %r617, %r616;
	cvt.rn.f32.s32 	%r618, %r1213;
	cvt.rn.f32.s32 	%r619, %r1214;
	cvt.rn.bf16x2.f32 	%r1322, %r619, %r618;
	cvt.rn.f32.s32 	%r620, %r1215;
	cvt.rn.f32.s32 	%r621, %r1216;
	cvt.rn.bf16x2.f32 	%r1323, %r621, %r620;
	cvt.rn.f32.s32 	%r622, %r1217;
	cvt.rn.f32.s32 	%r623, %r1218;
	cvt.rn.bf16x2.f32 	%r1324, %r623, %r622;
	cvt.rn.f32.s32 	%r624, %r1219;
	cvt.rn.f32.s32 	%r625, %r1220;
	cvt.rn.bf16x2.f32 	%r1325, %r625, %r624;
	cvt.rn.f32.s32 	%r626, %r1221;
	cvt.rn.f32.s32 	%r627, %r1222;
	cvt.rn.bf16x2.f32 	%r1326, %r627, %r626;
	cvt.rn.f32.s32 	%r628, %r1223;
	cvt.rn.f32.s32 	%r629, %r1224;
	cvt.rn.bf16x2.f32 	%r1327, %r629, %r628;
	cvt.rn.f32.s32 	%r630, %r1225;
	cvt.rn.f32.s32 	%r631, %r1226;
	cvt.rn.bf16x2.f32 	%r1328, %r631, %r630;
	cvt.rn.f32.s32 	%r632, %r1227;
	cvt.rn.f32.s32 	%r633, %r1228;
	cvt.rn.bf16x2.f32 	%r1329, %r633, %r632;
	cvt.rn.f32.s32 	%r634, %r1229;
	cvt.rn.f32.s32 	%r635, %r1230;
	cvt.rn.bf16x2.f32 	%r1330, %r635, %r634;
	cvt.rn.f32.s32 	%r636, %r1231;
	cvt.rn.f32.s32 	%r637, %r1232;
	cvt.rn.bf16x2.f32 	%r1331, %r637, %r636;
	cvt.rn.f32.s32 	%r638, %r1233;
	cvt.rn.f32.s32 	%r639, %r1234;
	cvt.rn.bf16x2.f32 	%r1332, %r639, %r638;
	cvt.rn.f32.s32 	%r640, %r1235;
	cvt.rn.f32.s32 	%r641, %r1236;
	cvt.rn.bf16x2.f32 	%r1333, %r641, %r640;
	cvt.rn.f32.s32 	%r642, %r1237;
	cvt.rn.f32.s32 	%r643, %r1238;
	cvt.rn.bf16x2.f32 	%r1334, %r643, %r642;
	cvt.rn.f32.s32 	%r644, %r1239;
	cvt.rn.f32.s32 	%r645, %r1240;
	cvt.rn.bf16x2.f32 	%r1335, %r645, %r644;
	cvt.rn.f32.s32 	%r646, %r1241;
	cvt.rn.f32.s32 	%r647, %r1242;
	cvt.rn.bf16x2.f32 	%r1336, %r647, %r646;
	cvt.rn.f32.s32 	%r648, %r1243;
	cvt.rn.f32.s32 	%r649, %r1244;
	cvt.rn.bf16x2.f32 	%r1337, %r649, %r648;
	cvt.rn.f32.s32 	%r650, %r1245;
	cvt.rn.f32.s32 	%r651, %r1246;
	cvt.rn.bf16x2.f32 	%r1338, %r651, %r650;
	cvt.rn.f32.s32 	%r652, %r1247;
	cvt.rn.f32.s32 	%r653, %r1248;
	cvt.rn.bf16x2.f32 	%r1339, %r653, %r652;
	cvt.rn.f32.s32 	%r654, %r1249;
	cvt.rn.f32.s32 	%r655, %r1250;
	cvt.rn.bf16x2.f32 	%r1340, %r655, %r654;
	cvt.rn.f32.s32 	%r656, %r1251;
	cvt.rn.f32.s32 	%r657, %r1252;
	cvt.rn.bf16x2.f32 	%r1341, %r657, %r656;
	cvt.rn.f32.s32 	%r658, %r1253;
	cvt.rn.f32.s32 	%r659, %r1254;
	cvt.rn.bf16x2.f32 	%r1342, %r659, %r658;
	cvt.rn.f32.s32 	%r660, %r1255;
	cvt.rn.f32.s32 	%r661, %r1256;
	cvt.rn.bf16x2.f32 	%r1343, %r661, %r660;
	cvt.rn.f32.s32 	%r662, %r1257;
	cvt.rn.f32.s32 	%r663, %r1258;
	cvt.rn.bf16x2.f32 	%r1344, %r663, %r662;
	cvt.rn.f32.s32 	%r664, %r1259;
	cvt.rn.f32.s32 	%r665, %r1260;
	cvt.rn.bf16x2.f32 	%r1345, %r665, %r664;
	cvt.rn.f32.s32 	%r666, %r1261;
	cvt.rn.f32.s32 	%r667, %r1262;
	cvt.rn.bf16x2.f32 	%r1346, %r667, %r666;
	cvt.rn.f32.s32 	%r668, %r1263;
	cvt.rn.f32.s32 	%r669, %r1264;
	cvt.rn.bf16x2.f32 	%r1347, %r669, %r668;
	cvt.rn.f32.s32 	%r670, %r1265;
	cvt.rn.f32.s32 	%r671, %r1266;
	cvt.rn.bf16x2.f32 	%r1348, %r671, %r670;
	cvt.rn.f32.s32 	%r672, %r1267;
	cvt.rn.f32.s32 	%r673, %r1268;
	cvt.rn.bf16x2.f32 	%r1349, %r673, %r672;
	cvt.rn.f32.s32 	%r674, %r1269;
	cvt.rn.f32.s32 	%r675, %r1270;
	cvt.rn.bf16x2.f32 	%r1350, %r675, %r674;
	cvt.rn.f32.s32 	%r676, %r1271;
	cvt.rn.f32.s32 	%r677, %r1272;
	cvt.rn.bf16x2.f32 	%r1351, %r677, %r676;
	cvt.rn.f32.s32 	%r678, %r1273;
	cvt.rn.f32.s32 	%r679, %r1274;
	cvt.rn.bf16x2.f32 	%r1352, %r679, %r678;
	cvt.rn.f32.s32 	%r680, %r1275;
	cvt.rn.f32.s32 	%r681, %r1276;
	cvt.rn.bf16x2.f32 	%r1353, %r681, %r680;
	cvt.rn.f32.s32 	%r682, %r1277;
	cvt.rn.f32.s32 	%r683, %r1278;
	cvt.rn.bf16x2.f32 	%r1354, %r683, %r682;
	cvt.rn.f32.s32 	%r684, %r1279;
	cvt.rn.f32.s32 	%r685, %r1280;
	cvt.rn.bf16x2.f32 	%r1355, %r685, %r684;
	cvt.rn.f32.s32 	%r686, %r1281;
	cvt.rn.f32.s32 	%r687, %r1282;
	cvt.rn.bf16x2.f32 	%r1356, %r687, %r686;
	cvt.rn.f32.s32 	%r688, %r1283;
	cvt.rn.f32.s32 	%r689, %r1284;
	cvt.rn.bf16x2.f32 	%r1357, %r689, %r688;
	cvt.rn.f32.s32 	%r690, %r1285;
	cvt.rn.f32.s32 	%r691, %r1286;
	cvt.rn.bf16x2.f32 	%r1358, %r691, %r690;
	cvt.rn.f32.s32 	%r692, %r1287;
	cvt.rn.f32.s32 	%r693, %r1288;
	cvt.rn.bf16x2.f32 	%r1359, %r693, %r692;
	cvt.rn.f32.s32 	%r694, %r1289;
	cvt.rn.f32.s32 	%r695, %r1290;
	cvt.rn.bf16x2.f32 	%r1360, %r695, %r694;
	cvt.rn.f32.s32 	%r696, %r1291;
	cvt.rn.f32.s32 	%r697, %r1292;
	cvt.rn.bf16x2.f32 	%r1361, %r697, %r696;
	cvt.rn.f32.s32 	%r698, %r1293;
	cvt.rn.f32.s32 	%r699, %r1294;
	cvt.rn.bf16x2.f32 	%r1362, %r699, %r698;
	cvt.rn.f32.s32 	%r700, %r1295;
	cvt.rn.f32.s32 	%r701, %r1296;
	cvt.rn.bf16x2.f32 	%r1363, %r701, %r700;
	cvt.rn.f32.s32 	%r702, %r1297;
	cvt.rn.f32.s32 	%r703, %r1298;
	cvt.rn.bf16x2.f32 	%r1364, %r703, %r702;
	bra.uni 	$L__BB0_5;
$L__BB0_1:                              // %.._crit_edge_crit_edge
	.loc	1 188 19                        // sk03_fa_qkv.py:188:19
	and.b32 	%r1300, %r2, 16;
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	shl.b32 	%r1299, %r2, 4;
	mov.b32 	%r1301, 0;
	mov.b32 	%r1302, %r1301;
	mov.b32 	%r1303, %r1301;
	mov.b32 	%r1304, %r1301;
	mov.b32 	%r1305, %r1301;
	mov.b32 	%r1306, %r1301;
	mov.b32 	%r1307, %r1301;
	mov.b32 	%r1308, %r1301;
	mov.b32 	%r1309, %r1301;
	mov.b32 	%r1310, %r1301;
	mov.b32 	%r1311, %r1301;
	mov.b32 	%r1312, %r1301;
	mov.b32 	%r1313, %r1301;
	mov.b32 	%r1314, %r1301;
	mov.b32 	%r1315, %r1301;
	mov.b32 	%r1316, %r1301;
	mov.b32 	%r1317, %r1301;
	mov.b32 	%r1318, %r1301;
	mov.b32 	%r1319, %r1301;
	mov.b32 	%r1320, %r1301;
	mov.b32 	%r1321, %r1301;
	mov.b32 	%r1322, %r1301;
	mov.b32 	%r1323, %r1301;
	mov.b32 	%r1324, %r1301;
	mov.b32 	%r1325, %r1301;
	mov.b32 	%r1326, %r1301;
	mov.b32 	%r1327, %r1301;
	mov.b32 	%r1328, %r1301;
	mov.b32 	%r1329, %r1301;
	mov.b32 	%r1330, %r1301;
	mov.b32 	%r1331, %r1301;
	mov.b32 	%r1332, %r1301;
	mov.b32 	%r1333, %r1301;
	mov.b32 	%r1334, %r1301;
	mov.b32 	%r1335, %r1301;
	mov.b32 	%r1336, %r1301;
	mov.b32 	%r1337, %r1301;
	mov.b32 	%r1338, %r1301;
	mov.b32 	%r1339, %r1301;
	mov.b32 	%r1340, %r1301;
	mov.b32 	%r1341, %r1301;
	mov.b32 	%r1342, %r1301;
	mov.b32 	%r1343, %r1301;
	mov.b32 	%r1344, %r1301;
	mov.b32 	%r1345, %r1301;
	mov.b32 	%r1346, %r1301;
	mov.b32 	%r1347, %r1301;
	mov.b32 	%r1348, %r1301;
	mov.b32 	%r1349, %r1301;
	mov.b32 	%r1350, %r1301;
	mov.b32 	%r1351, %r1301;
	mov.b32 	%r1352, %r1301;
	mov.b32 	%r1353, %r1301;
	mov.b32 	%r1354, %r1301;
	mov.b32 	%r1355, %r1301;
	mov.b32 	%r1356, %r1301;
	mov.b32 	%r1357, %r1301;
	mov.b32 	%r1358, %r1301;
	mov.b32 	%r1359, %r1301;
	mov.b32 	%r1360, %r1301;
	mov.b32 	%r1361, %r1301;
	mov.b32 	%r1362, %r1301;
	mov.b32 	%r1363, %r1301;
	mov.b32 	%r1364, %r1301;
$L__BB0_5:                              // %._crit_edge
	.loc	1 164 45                        // sk03_fa_qkv.py:164:45
	shl.b32 	%r934, %r7, 3;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r935, %r934, %r4;
	or.b32 	%r936, %r935, 7;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r937, %r936, %r22;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r938, %r935, 6;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r939, %r938, %r22;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r940, %r935, 5;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r941, %r940, %r22;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r942, %r935, 4;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r943, %r942, %r22;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r944, %r935, 3;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r945, %r944, %r22;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r946, %r935, 2;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r947, %r946, %r22;
	.loc	1 164 32                        // sk03_fa_qkv.py:164:32
	or.b32 	%r948, %r935, 1;
	.loc	1 164 57                        // sk03_fa_qkv.py:164:57
	rem.s32 	%r949, %r948, %r22;
	rem.s32 	%r950, %r935, %r22;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r951, %r1, %r3;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r952, %r951, %r21;
	.loc	1 163 45                        // sk03_fa_qkv.py:163:45
	and.b32 	%r953, %r2, 240;
	bfe.u32 	%r954, %r2, 4, 4;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r955, %r954, %r1;
	or.b32 	%r956, %r955, 240;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r957, %r956, %r21;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r958, %r955, 224;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r959, %r958, %r21;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r960, %r955, 208;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r961, %r960, %r21;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r962, %r955, 192;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r963, %r962, %r21;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r964, %r955, 176;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r965, %r964, %r21;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r966, %r955, 160;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r967, %r966, %r21;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r968, %r955, 144;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r969, %r968, %r21;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r970, %r955, 128;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r971, %r970, %r21;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r972, %r955, 112;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r973, %r972, %r21;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r974, %r955, 96;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r975, %r974, %r21;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r976, %r955, 80;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r977, %r976, %r21;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r978, %r955, 64;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r979, %r978, %r21;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r980, %r955, 48;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r981, %r980, %r21;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r982, %r955, 32;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r983, %r982, %r21;
	.loc	1 163 32                        // sk03_fa_qkv.py:163:32
	or.b32 	%r984, %r955, 16;
	.loc	1 163 57                        // sk03_fa_qkv.py:163:57
	rem.s32 	%r985, %r984, %r21;
	rem.s32 	%r986, %r955, %r21;
	.loc	1 178 23                        // sk03_fa_qkv.py:178:23
	cp.async.wait_group 	0;
	bar.sync 	0;
	.loc	1 186 54                        // sk03_fa_qkv.py:186:54
	mad.wide.s32 	%rd113, %r952, 4, %rd36;
	.loc	1 186 40                        // sk03_fa_qkv.py:186:40
	// begin inline asm
	mov.u32 %r704, 0x0;
	ld.global.b32 { %r704 }, [ %rd113 + 0 ];
	// end inline asm
	.loc	1 186 65                        // sk03_fa_qkv.py:186:65
	cvt.rn.bf16.f32 	%rs17, %r704;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	and.b32 	%r987, %r2, 7;
	shl.b32 	%r988, %r987, 3;
	shl.b32 	%r989, %r2, 2;
	and.b32 	%r990, %r989, 192;
	and.b32 	%r991, %r2, 8;
	shr.u32 	%r992, %r991, 1;
	shr.u32 	%r993, %r2, 5;
	and.b32 	%r994, %r993, 2;
	and.b32 	%r995, %r16, 256;
	add.s32 	%r996, %r189, %r988;
	add.s32 	%r997, %r996, %r990;
	add.s32 	%r998, %r997, %r992;
	add.s32 	%r999, %r998, %r994;
	add.s32 	%r705, %r999, %r995;
	// begin inline asm
	st.shared.b16 [ %r705 + 0 ], %rs17;
	// end inline asm
	bar.sync 	0;
	and.b32 	%r1000, %r16, 56;
	and.b32 	%r1001, %r2, 192;
	add.s32 	%r1002, %r189, %r1000;
	add.s32 	%r1003, %r1002, %r1001;
	ld.shared.v4.b16 	{%rs146, %rs147, %rs148, %rs149}, [%r1003];
	ld.shared.v4.b16 	{%rs150, %rs151, %rs152, %rs153}, [%r1003+256];
	mov.b32 	{%rs154, %rs155}, %r1301;
	mov.b16 	%rs156, 0x8000;
	fma.rn.bf16 	%rs157, %rs146, %rs154, %rs156;
	fma.rn.bf16 	%rs158, %rs146, %rs155, %rs156;
	mov.b32 	{%rs159, %rs160}, %r1302;
	fma.rn.bf16 	%rs161, %rs148, %rs159, %rs156;
	fma.rn.bf16 	%rs162, %rs148, %rs160, %rs156;
	mov.b32 	{%rs163, %rs164}, %r1303;
	fma.rn.bf16 	%rs165, %rs146, %rs163, %rs156;
	fma.rn.bf16 	%rs166, %rs146, %rs164, %rs156;
	mov.b32 	{%rs167, %rs168}, %r1304;
	fma.rn.bf16 	%rs169, %rs148, %rs167, %rs156;
	fma.rn.bf16 	%rs170, %rs148, %rs168, %rs156;
	mov.b32 	{%rs171, %rs172}, %r1305;
	fma.rn.bf16 	%rs173, %rs146, %rs171, %rs156;
	fma.rn.bf16 	%rs174, %rs146, %rs172, %rs156;
	mov.b32 	{%rs175, %rs176}, %r1306;
	fma.rn.bf16 	%rs177, %rs148, %rs175, %rs156;
	fma.rn.bf16 	%rs178, %rs148, %rs176, %rs156;
	mov.b32 	{%rs179, %rs180}, %r1307;
	fma.rn.bf16 	%rs181, %rs146, %rs179, %rs156;
	fma.rn.bf16 	%rs182, %rs146, %rs180, %rs156;
	mov.b32 	{%rs183, %rs184}, %r1308;
	fma.rn.bf16 	%rs185, %rs148, %rs183, %rs156;
	fma.rn.bf16 	%rs186, %rs148, %rs184, %rs156;
	mov.b32 	{%rs187, %rs188}, %r1309;
	fma.rn.bf16 	%rs189, %rs146, %rs187, %rs156;
	fma.rn.bf16 	%rs190, %rs146, %rs188, %rs156;
	mov.b32 	{%rs191, %rs192}, %r1310;
	fma.rn.bf16 	%rs193, %rs148, %rs191, %rs156;
	fma.rn.bf16 	%rs194, %rs148, %rs192, %rs156;
	mov.b32 	{%rs195, %rs196}, %r1311;
	fma.rn.bf16 	%rs197, %rs146, %rs195, %rs156;
	fma.rn.bf16 	%rs198, %rs146, %rs196, %rs156;
	mov.b32 	{%rs199, %rs200}, %r1312;
	fma.rn.bf16 	%rs201, %rs148, %rs199, %rs156;
	fma.rn.bf16 	%rs202, %rs148, %rs200, %rs156;
	mov.b32 	{%rs203, %rs204}, %r1313;
	fma.rn.bf16 	%rs205, %rs146, %rs203, %rs156;
	fma.rn.bf16 	%rs206, %rs146, %rs204, %rs156;
	mov.b32 	{%rs207, %rs208}, %r1314;
	fma.rn.bf16 	%rs209, %rs148, %rs207, %rs156;
	fma.rn.bf16 	%rs210, %rs148, %rs208, %rs156;
	mov.b32 	{%rs211, %rs212}, %r1315;
	fma.rn.bf16 	%rs213, %rs146, %rs211, %rs156;
	fma.rn.bf16 	%rs214, %rs146, %rs212, %rs156;
	mov.b32 	{%rs215, %rs216}, %r1316;
	fma.rn.bf16 	%rs217, %rs148, %rs215, %rs156;
	fma.rn.bf16 	%rs218, %rs148, %rs216, %rs156;
	mov.b32 	{%rs219, %rs220}, %r1317;
	fma.rn.bf16 	%rs221, %rs147, %rs219, %rs156;
	fma.rn.bf16 	%rs222, %rs147, %rs220, %rs156;
	mov.b32 	{%rs223, %rs224}, %r1318;
	fma.rn.bf16 	%rs225, %rs149, %rs223, %rs156;
	fma.rn.bf16 	%rs226, %rs149, %rs224, %rs156;
	mov.b32 	{%rs227, %rs228}, %r1319;
	fma.rn.bf16 	%rs229, %rs147, %rs227, %rs156;
	fma.rn.bf16 	%rs230, %rs147, %rs228, %rs156;
	mov.b32 	{%rs231, %rs232}, %r1320;
	fma.rn.bf16 	%rs233, %rs149, %rs231, %rs156;
	fma.rn.bf16 	%rs234, %rs149, %rs232, %rs156;
	mov.b32 	{%rs235, %rs236}, %r1321;
	fma.rn.bf16 	%rs237, %rs147, %rs235, %rs156;
	fma.rn.bf16 	%rs238, %rs147, %rs236, %rs156;
	mov.b32 	{%rs239, %rs240}, %r1322;
	fma.rn.bf16 	%rs241, %rs149, %rs239, %rs156;
	fma.rn.bf16 	%rs242, %rs149, %rs240, %rs156;
	mov.b32 	{%rs243, %rs244}, %r1323;
	fma.rn.bf16 	%rs245, %rs147, %rs243, %rs156;
	fma.rn.bf16 	%rs246, %rs147, %rs244, %rs156;
	mov.b32 	{%rs247, %rs248}, %r1324;
	fma.rn.bf16 	%rs249, %rs149, %rs247, %rs156;
	fma.rn.bf16 	%rs250, %rs149, %rs248, %rs156;
	mov.b32 	{%rs251, %rs252}, %r1325;
	fma.rn.bf16 	%rs253, %rs147, %rs251, %rs156;
	fma.rn.bf16 	%rs254, %rs147, %rs252, %rs156;
	mov.b32 	{%rs255, %rs256}, %r1326;
	fma.rn.bf16 	%rs257, %rs149, %rs255, %rs156;
	fma.rn.bf16 	%rs258, %rs149, %rs256, %rs156;
	mov.b32 	{%rs259, %rs260}, %r1327;
	fma.rn.bf16 	%rs261, %rs147, %rs259, %rs156;
	fma.rn.bf16 	%rs262, %rs147, %rs260, %rs156;
	mov.b32 	{%rs263, %rs264}, %r1328;
	fma.rn.bf16 	%rs265, %rs149, %rs263, %rs156;
	fma.rn.bf16 	%rs266, %rs149, %rs264, %rs156;
	mov.b32 	{%rs267, %rs268}, %r1329;
	fma.rn.bf16 	%rs269, %rs147, %rs267, %rs156;
	fma.rn.bf16 	%rs270, %rs147, %rs268, %rs156;
	mov.b32 	{%rs271, %rs272}, %r1330;
	fma.rn.bf16 	%rs273, %rs149, %rs271, %rs156;
	fma.rn.bf16 	%rs274, %rs149, %rs272, %rs156;
	mov.b32 	{%rs275, %rs276}, %r1331;
	fma.rn.bf16 	%rs277, %rs147, %rs275, %rs156;
	fma.rn.bf16 	%rs278, %rs147, %rs276, %rs156;
	mov.b32 	{%rs279, %rs280}, %r1332;
	fma.rn.bf16 	%rs281, %rs149, %rs279, %rs156;
	fma.rn.bf16 	%rs282, %rs149, %rs280, %rs156;
	mov.b32 	{%rs283, %rs284}, %r1333;
	fma.rn.bf16 	%rs285, %rs150, %rs283, %rs156;
	fma.rn.bf16 	%rs286, %rs150, %rs284, %rs156;
	mov.b32 	{%rs287, %rs288}, %r1334;
	fma.rn.bf16 	%rs289, %rs152, %rs287, %rs156;
	fma.rn.bf16 	%rs290, %rs152, %rs288, %rs156;
	mov.b32 	{%rs291, %rs292}, %r1335;
	fma.rn.bf16 	%rs293, %rs150, %rs291, %rs156;
	fma.rn.bf16 	%rs294, %rs150, %rs292, %rs156;
	mov.b32 	{%rs295, %rs296}, %r1336;
	fma.rn.bf16 	%rs297, %rs152, %rs295, %rs156;
	fma.rn.bf16 	%rs298, %rs152, %rs296, %rs156;
	mov.b32 	{%rs299, %rs300}, %r1337;
	fma.rn.bf16 	%rs301, %rs150, %rs299, %rs156;
	fma.rn.bf16 	%rs302, %rs150, %rs300, %rs156;
	mov.b32 	{%rs303, %rs304}, %r1338;
	fma.rn.bf16 	%rs305, %rs152, %rs303, %rs156;
	fma.rn.bf16 	%rs306, %rs152, %rs304, %rs156;
	mov.b32 	{%rs307, %rs308}, %r1339;
	fma.rn.bf16 	%rs309, %rs150, %rs307, %rs156;
	fma.rn.bf16 	%rs310, %rs150, %rs308, %rs156;
	mov.b32 	{%rs311, %rs312}, %r1340;
	fma.rn.bf16 	%rs313, %rs152, %rs311, %rs156;
	fma.rn.bf16 	%rs314, %rs152, %rs312, %rs156;
	mov.b32 	{%rs315, %rs316}, %r1341;
	fma.rn.bf16 	%rs317, %rs150, %rs315, %rs156;
	fma.rn.bf16 	%rs318, %rs150, %rs316, %rs156;
	mov.b32 	{%rs319, %rs320}, %r1342;
	fma.rn.bf16 	%rs321, %rs152, %rs319, %rs156;
	fma.rn.bf16 	%rs322, %rs152, %rs320, %rs156;
	mov.b32 	{%rs323, %rs324}, %r1343;
	fma.rn.bf16 	%rs325, %rs150, %rs323, %rs156;
	fma.rn.bf16 	%rs326, %rs150, %rs324, %rs156;
	mov.b32 	{%rs327, %rs328}, %r1344;
	fma.rn.bf16 	%rs329, %rs152, %rs327, %rs156;
	fma.rn.bf16 	%rs330, %rs152, %rs328, %rs156;
	mov.b32 	{%rs331, %rs332}, %r1345;
	fma.rn.bf16 	%rs333, %rs150, %rs331, %rs156;
	fma.rn.bf16 	%rs334, %rs150, %rs332, %rs156;
	mov.b32 	{%rs335, %rs336}, %r1346;
	fma.rn.bf16 	%rs337, %rs152, %rs335, %rs156;
	fma.rn.bf16 	%rs338, %rs152, %rs336, %rs156;
	mov.b32 	{%rs339, %rs340}, %r1347;
	fma.rn.bf16 	%rs341, %rs150, %rs339, %rs156;
	fma.rn.bf16 	%rs342, %rs150, %rs340, %rs156;
	mov.b32 	{%rs343, %rs344}, %r1348;
	fma.rn.bf16 	%rs345, %rs152, %rs343, %rs156;
	fma.rn.bf16 	%rs346, %rs152, %rs344, %rs156;
	mov.b32 	{%rs347, %rs348}, %r1350;
	fma.rn.bf16 	%rs349, %rs153, %rs347, %rs156;
	fma.rn.bf16 	%rs350, %rs153, %rs348, %rs156;
	mov.b32 	{%rs351, %rs352}, %r1352;
	fma.rn.bf16 	%rs353, %rs153, %rs351, %rs156;
	fma.rn.bf16 	%rs354, %rs153, %rs352, %rs156;
	mov.b32 	{%rs355, %rs356}, %r1354;
	fma.rn.bf16 	%rs357, %rs153, %rs355, %rs156;
	fma.rn.bf16 	%rs358, %rs153, %rs356, %rs156;
	mov.b32 	{%rs359, %rs360}, %r1356;
	fma.rn.bf16 	%rs361, %rs153, %rs359, %rs156;
	fma.rn.bf16 	%rs362, %rs153, %rs360, %rs156;
	mov.b32 	{%rs363, %rs364}, %r1358;
	fma.rn.bf16 	%rs365, %rs153, %rs363, %rs156;
	fma.rn.bf16 	%rs366, %rs153, %rs364, %rs156;
	mov.b32 	{%rs367, %rs368}, %r1360;
	fma.rn.bf16 	%rs369, %rs153, %rs367, %rs156;
	fma.rn.bf16 	%rs370, %rs153, %rs368, %rs156;
	mov.b32 	{%rs371, %rs372}, %r1362;
	fma.rn.bf16 	%rs373, %rs153, %rs371, %rs156;
	fma.rn.bf16 	%rs374, %rs153, %rs372, %rs156;
	mov.b32 	{%rs375, %rs376}, %r1364;
	fma.rn.bf16 	%rs377, %rs153, %rs375, %rs156;
	fma.rn.bf16 	%rs378, %rs153, %rs376, %rs156;
	.loc	1 187 38                        // sk03_fa_qkv.py:187:38
	mad.wide.s32 	%rd114, %r8, 4, %rd37;
	mad.wide.s32 	%rd115, %r9, 4, %rd37;
	mad.wide.s32 	%rd116, %r10, 4, %rd37;
	mad.wide.s32 	%rd117, %r11, 4, %rd37;
	mad.wide.s32 	%rd118, %r12, 4, %rd37;
	mad.wide.s32 	%rd119, %r13, 4, %rd37;
	mad.wide.s32 	%rd120, %r14, 4, %rd37;
	mad.wide.s32 	%rd121, %r15, 4, %rd37;
	.loc	1 187 24                        // sk03_fa_qkv.py:187:24
	// begin inline asm
	mov.u32 %r706, 0x0;
	mov.u32 %r707, 0x0;
	ld.global.v2.b32 { %r706, %r707 }, [ %rd114 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r708, 0x0;
	mov.u32 %r709, 0x0;
	ld.global.v2.b32 { %r708, %r709 }, [ %rd115 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r710, 0x0;
	mov.u32 %r711, 0x0;
	ld.global.v2.b32 { %r710, %r711 }, [ %rd116 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r712, 0x0;
	mov.u32 %r713, 0x0;
	ld.global.v2.b32 { %r712, %r713 }, [ %rd117 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r714, 0x0;
	mov.u32 %r715, 0x0;
	ld.global.v2.b32 { %r714, %r715 }, [ %rd118 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r716, 0x0;
	mov.u32 %r717, 0x0;
	ld.global.v2.b32 { %r716, %r717 }, [ %rd119 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r718, 0x0;
	mov.u32 %r719, 0x0;
	ld.global.v2.b32 { %r718, %r719 }, [ %rd120 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u32 %r720, 0x0;
	mov.u32 %r721, 0x0;
	ld.global.v2.b32 { %r720, %r721 }, [ %rd121 + 0 ];
	// end inline asm
	.loc	1 188 49                        // sk03_fa_qkv.py:188:49
	mul.lo.s32 	%r1004, %r986, %r25;
	mul.lo.s32 	%r1005, %r985, %r25;
	mul.lo.s32 	%r1006, %r983, %r25;
	mul.lo.s32 	%r1007, %r981, %r25;
	mul.lo.s32 	%r1008, %r979, %r25;
	mul.lo.s32 	%r1009, %r977, %r25;
	mul.lo.s32 	%r1010, %r975, %r25;
	mul.lo.s32 	%r1011, %r973, %r25;
	mul.lo.s32 	%r1012, %r971, %r25;
	mul.lo.s32 	%r1013, %r969, %r25;
	mul.lo.s32 	%r1014, %r967, %r25;
	mul.lo.s32 	%r1015, %r965, %r25;
	mul.lo.s32 	%r1016, %r963, %r25;
	mul.lo.s32 	%r1017, %r961, %r25;
	mul.lo.s32 	%r1018, %r959, %r25;
	mul.lo.s32 	%r1019, %r957, %r25;
	.loc	1 188 31                        // sk03_fa_qkv.py:188:31
	mad.wide.s32 	%rd266, %r1004, 2, %rd35;
	mad.wide.s32 	%rd267, %r1005, 2, %rd35;
	mad.wide.s32 	%rd268, %r1006, 2, %rd35;
	mad.wide.s32 	%rd269, %r1007, 2, %rd35;
	mad.wide.s32 	%rd270, %r1008, 2, %rd35;
	mad.wide.s32 	%rd271, %r1009, 2, %rd35;
	mad.wide.s32 	%rd272, %r1010, 2, %rd35;
	mad.wide.s32 	%rd273, %r1011, 2, %rd35;
	mad.wide.s32 	%rd274, %r1012, 2, %rd35;
	mad.wide.s32 	%rd275, %r1013, 2, %rd35;
	mad.wide.s32 	%rd276, %r1014, 2, %rd35;
	mad.wide.s32 	%rd277, %r1015, 2, %rd35;
	mad.wide.s32 	%rd278, %r1016, 2, %rd35;
	mad.wide.s32 	%rd279, %r1017, 2, %rd35;
	mad.wide.s32 	%rd280, %r1018, 2, %rd35;
	mad.wide.s32 	%rd281, %r1019, 2, %rd35;
	.loc	1 188 82                        // sk03_fa_qkv.py:188:82
	mul.lo.s32 	%r1020, %r950, %r26;
	mul.lo.s32 	%r1021, %r949, %r26;
	mul.lo.s32 	%r1022, %r947, %r26;
	mul.lo.s32 	%r1023, %r945, %r26;
	mul.lo.s32 	%r1024, %r943, %r26;
	mul.lo.s32 	%r1025, %r941, %r26;
	mul.lo.s32 	%r1026, %r939, %r26;
	mul.lo.s32 	%r1027, %r937, %r26;
	.loc	1 188 64                        // sk03_fa_qkv.py:188:64
	mul.wide.s32 	%rd282, %r1020, 2;
	add.s64 	%rd122, %rd266, %rd282;
	mul.wide.s32 	%rd283, %r1021, 2;
	add.s64 	%rd123, %rd266, %rd283;
	mul.wide.s32 	%rd284, %r1022, 2;
	add.s64 	%rd124, %rd266, %rd284;
	mul.wide.s32 	%rd285, %r1023, 2;
	add.s64 	%rd125, %rd266, %rd285;
	mul.wide.s32 	%rd286, %r1024, 2;
	add.s64 	%rd126, %rd266, %rd286;
	mul.wide.s32 	%rd287, %r1025, 2;
	add.s64 	%rd127, %rd266, %rd287;
	mul.wide.s32 	%rd288, %r1026, 2;
	add.s64 	%rd128, %rd266, %rd288;
	mul.wide.s32 	%rd289, %r1027, 2;
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
	.loc	1 188 19                        // sk03_fa_qkv.py:188:19
	// begin inline asm
	mov.u16 %rs18, 0x0;
	ld.global.b16 { %rs18 }, [ %rd122 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs19, 0x0;
	ld.global.b16 { %rs19 }, [ %rd123 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs20, 0x0;
	ld.global.b16 { %rs20 }, [ %rd124 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs21, 0x0;
	ld.global.b16 { %rs21 }, [ %rd125 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs22, 0x0;
	ld.global.b16 { %rs22 }, [ %rd126 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs23, 0x0;
	ld.global.b16 { %rs23 }, [ %rd127 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs24, 0x0;
	ld.global.b16 { %rs24 }, [ %rd128 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs25, 0x0;
	ld.global.b16 { %rs25 }, [ %rd129 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs26, 0x0;
	ld.global.b16 { %rs26 }, [ %rd130 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs27, 0x0;
	ld.global.b16 { %rs27 }, [ %rd131 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs28, 0x0;
	ld.global.b16 { %rs28 }, [ %rd132 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs29, 0x0;
	ld.global.b16 { %rs29 }, [ %rd133 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs30, 0x0;
	ld.global.b16 { %rs30 }, [ %rd134 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs31, 0x0;
	ld.global.b16 { %rs31 }, [ %rd135 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs32, 0x0;
	ld.global.b16 { %rs32 }, [ %rd136 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs33, 0x0;
	ld.global.b16 { %rs33 }, [ %rd137 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs34, 0x0;
	ld.global.b16 { %rs34 }, [ %rd138 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs35, 0x0;
	ld.global.b16 { %rs35 }, [ %rd139 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs36, 0x0;
	ld.global.b16 { %rs36 }, [ %rd140 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs37, 0x0;
	ld.global.b16 { %rs37 }, [ %rd141 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs38, 0x0;
	ld.global.b16 { %rs38 }, [ %rd142 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs39, 0x0;
	ld.global.b16 { %rs39 }, [ %rd143 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs40, 0x0;
	ld.global.b16 { %rs40 }, [ %rd144 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs41, 0x0;
	ld.global.b16 { %rs41 }, [ %rd145 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs42, 0x0;
	ld.global.b16 { %rs42 }, [ %rd146 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs43, 0x0;
	ld.global.b16 { %rs43 }, [ %rd147 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs44, 0x0;
	ld.global.b16 { %rs44 }, [ %rd148 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs45, 0x0;
	ld.global.b16 { %rs45 }, [ %rd149 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs46, 0x0;
	ld.global.b16 { %rs46 }, [ %rd150 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs47, 0x0;
	ld.global.b16 { %rs47 }, [ %rd151 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs48, 0x0;
	ld.global.b16 { %rs48 }, [ %rd152 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs49, 0x0;
	ld.global.b16 { %rs49 }, [ %rd153 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs50, 0x0;
	ld.global.b16 { %rs50 }, [ %rd154 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs51, 0x0;
	ld.global.b16 { %rs51 }, [ %rd155 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs52, 0x0;
	ld.global.b16 { %rs52 }, [ %rd156 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs53, 0x0;
	ld.global.b16 { %rs53 }, [ %rd157 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs54, 0x0;
	ld.global.b16 { %rs54 }, [ %rd158 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs55, 0x0;
	ld.global.b16 { %rs55 }, [ %rd159 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs56, 0x0;
	ld.global.b16 { %rs56 }, [ %rd160 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs57, 0x0;
	ld.global.b16 { %rs57 }, [ %rd161 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs58, 0x0;
	ld.global.b16 { %rs58 }, [ %rd162 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs59, 0x0;
	ld.global.b16 { %rs59 }, [ %rd163 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs60, 0x0;
	ld.global.b16 { %rs60 }, [ %rd164 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs61, 0x0;
	ld.global.b16 { %rs61 }, [ %rd165 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs62, 0x0;
	ld.global.b16 { %rs62 }, [ %rd166 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs63, 0x0;
	ld.global.b16 { %rs63 }, [ %rd167 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs64, 0x0;
	ld.global.b16 { %rs64 }, [ %rd168 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs65, 0x0;
	ld.global.b16 { %rs65 }, [ %rd169 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs66, 0x0;
	ld.global.b16 { %rs66 }, [ %rd170 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs67, 0x0;
	ld.global.b16 { %rs67 }, [ %rd171 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs68, 0x0;
	ld.global.b16 { %rs68 }, [ %rd172 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs69, 0x0;
	ld.global.b16 { %rs69 }, [ %rd173 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs70, 0x0;
	ld.global.b16 { %rs70 }, [ %rd174 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs71, 0x0;
	ld.global.b16 { %rs71 }, [ %rd175 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs72, 0x0;
	ld.global.b16 { %rs72 }, [ %rd176 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs73, 0x0;
	ld.global.b16 { %rs73 }, [ %rd177 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs74, 0x0;
	ld.global.b16 { %rs74 }, [ %rd178 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs75, 0x0;
	ld.global.b16 { %rs75 }, [ %rd179 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs76, 0x0;
	ld.global.b16 { %rs76 }, [ %rd180 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs77, 0x0;
	ld.global.b16 { %rs77 }, [ %rd181 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs78, 0x0;
	ld.global.b16 { %rs78 }, [ %rd182 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs79, 0x0;
	ld.global.b16 { %rs79 }, [ %rd183 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs80, 0x0;
	ld.global.b16 { %rs80 }, [ %rd184 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs81, 0x0;
	ld.global.b16 { %rs81 }, [ %rd185 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs82, 0x0;
	ld.global.b16 { %rs82 }, [ %rd186 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs83, 0x0;
	ld.global.b16 { %rs83 }, [ %rd187 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs84, 0x0;
	ld.global.b16 { %rs84 }, [ %rd188 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs85, 0x0;
	ld.global.b16 { %rs85 }, [ %rd189 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs86, 0x0;
	ld.global.b16 { %rs86 }, [ %rd190 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs87, 0x0;
	ld.global.b16 { %rs87 }, [ %rd191 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs88, 0x0;
	ld.global.b16 { %rs88 }, [ %rd192 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs89, 0x0;
	ld.global.b16 { %rs89 }, [ %rd193 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs90, 0x0;
	ld.global.b16 { %rs90 }, [ %rd194 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs91, 0x0;
	ld.global.b16 { %rs91 }, [ %rd195 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs92, 0x0;
	ld.global.b16 { %rs92 }, [ %rd196 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs93, 0x0;
	ld.global.b16 { %rs93 }, [ %rd197 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs94, 0x0;
	ld.global.b16 { %rs94 }, [ %rd198 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs95, 0x0;
	ld.global.b16 { %rs95 }, [ %rd199 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs96, 0x0;
	ld.global.b16 { %rs96 }, [ %rd200 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs97, 0x0;
	ld.global.b16 { %rs97 }, [ %rd201 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs98, 0x0;
	ld.global.b16 { %rs98 }, [ %rd202 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs99, 0x0;
	ld.global.b16 { %rs99 }, [ %rd203 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs100, 0x0;
	ld.global.b16 { %rs100 }, [ %rd204 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs101, 0x0;
	ld.global.b16 { %rs101 }, [ %rd205 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs102, 0x0;
	ld.global.b16 { %rs102 }, [ %rd206 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs103, 0x0;
	ld.global.b16 { %rs103 }, [ %rd207 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs104, 0x0;
	ld.global.b16 { %rs104 }, [ %rd208 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs105, 0x0;
	ld.global.b16 { %rs105 }, [ %rd209 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs106, 0x0;
	ld.global.b16 { %rs106 }, [ %rd210 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs107, 0x0;
	ld.global.b16 { %rs107 }, [ %rd211 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs108, 0x0;
	ld.global.b16 { %rs108 }, [ %rd212 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs109, 0x0;
	ld.global.b16 { %rs109 }, [ %rd213 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs110, 0x0;
	ld.global.b16 { %rs110 }, [ %rd214 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs111, 0x0;
	ld.global.b16 { %rs111 }, [ %rd215 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs112, 0x0;
	ld.global.b16 { %rs112 }, [ %rd216 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs113, 0x0;
	ld.global.b16 { %rs113 }, [ %rd217 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs114, 0x0;
	ld.global.b16 { %rs114 }, [ %rd218 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs115, 0x0;
	ld.global.b16 { %rs115 }, [ %rd219 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs116, 0x0;
	ld.global.b16 { %rs116 }, [ %rd220 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs117, 0x0;
	ld.global.b16 { %rs117 }, [ %rd221 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs118, 0x0;
	ld.global.b16 { %rs118 }, [ %rd222 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs119, 0x0;
	ld.global.b16 { %rs119 }, [ %rd223 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs120, 0x0;
	ld.global.b16 { %rs120 }, [ %rd224 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs121, 0x0;
	ld.global.b16 { %rs121 }, [ %rd225 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs122, 0x0;
	ld.global.b16 { %rs122 }, [ %rd226 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs123, 0x0;
	ld.global.b16 { %rs123 }, [ %rd227 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs124, 0x0;
	ld.global.b16 { %rs124 }, [ %rd228 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs125, 0x0;
	ld.global.b16 { %rs125 }, [ %rd229 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs126, 0x0;
	ld.global.b16 { %rs126 }, [ %rd230 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs127, 0x0;
	ld.global.b16 { %rs127 }, [ %rd231 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs128, 0x0;
	ld.global.b16 { %rs128 }, [ %rd232 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs129, 0x0;
	ld.global.b16 { %rs129 }, [ %rd233 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs130, 0x0;
	ld.global.b16 { %rs130 }, [ %rd234 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs131, 0x0;
	ld.global.b16 { %rs131 }, [ %rd235 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs132, 0x0;
	ld.global.b16 { %rs132 }, [ %rd236 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs133, 0x0;
	ld.global.b16 { %rs133 }, [ %rd237 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs134, 0x0;
	ld.global.b16 { %rs134 }, [ %rd238 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs135, 0x0;
	ld.global.b16 { %rs135 }, [ %rd239 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs136, 0x0;
	ld.global.b16 { %rs136 }, [ %rd240 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs137, 0x0;
	ld.global.b16 { %rs137 }, [ %rd241 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs138, 0x0;
	ld.global.b16 { %rs138 }, [ %rd242 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs139, 0x0;
	ld.global.b16 { %rs139 }, [ %rd243 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs140, 0x0;
	ld.global.b16 { %rs140 }, [ %rd244 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs141, 0x0;
	ld.global.b16 { %rs141 }, [ %rd245 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs142, 0x0;
	ld.global.b16 { %rs142 }, [ %rd246 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs143, 0x0;
	ld.global.b16 { %rs143 }, [ %rd247 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs144, 0x0;
	ld.global.b16 { %rs144 }, [ %rd248 + 0 ];
	// end inline asm
	// begin inline asm
	mov.u16 %rs145, 0x0;
	ld.global.b16 { %rs145 }, [ %rd249 + 0 ];
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r1028, %r2, 7;
	and.b32 	%r1029, %r1028, 15360;
	shl.b32 	%r1030, %r987, 4;
	or.b32 	%r1031, %r1029, %r1030;
	xor.b32 	%r1032, %r1031, %r953;
	add.s32 	%r722, %r189, %r1032;
	mov.b32 	%r723, {%rs18, %rs19};
	mov.b32 	%r724, {%rs20, %rs21};
	mov.b32 	%r725, {%rs22, %rs23};
	mov.b32 	%r726, {%rs24, %rs25};
	// begin inline asm
	st.shared.v4.b32 [ %r722 + 0 ], { %r723, %r724, %r725, %r726 };
	// end inline asm
	add.s32 	%r727, %r722, 256;
	mov.b32 	%r728, {%rs26, %rs27};
	mov.b32 	%r729, {%rs28, %rs29};
	mov.b32 	%r730, {%rs30, %rs31};
	mov.b32 	%r731, {%rs32, %rs33};
	// begin inline asm
	st.shared.v4.b32 [ %r727 + 0 ], { %r728, %r729, %r730, %r731 };
	// end inline asm
	add.s32 	%r732, %r722, 512;
	mov.b32 	%r733, {%rs34, %rs35};
	mov.b32 	%r734, {%rs36, %rs37};
	mov.b32 	%r735, {%rs38, %rs39};
	mov.b32 	%r736, {%rs40, %rs41};
	// begin inline asm
	st.shared.v4.b32 [ %r732 + 0 ], { %r733, %r734, %r735, %r736 };
	// end inline asm
	add.s32 	%r737, %r722, 768;
	mov.b32 	%r738, {%rs42, %rs43};
	mov.b32 	%r739, {%rs44, %rs45};
	mov.b32 	%r740, {%rs46, %rs47};
	mov.b32 	%r741, {%rs48, %rs49};
	// begin inline asm
	st.shared.v4.b32 [ %r737 + 0 ], { %r738, %r739, %r740, %r741 };
	// end inline asm
	bar.sync 	0;
	shl.b32 	%r1033, %r987, 11;
	shl.b32 	%r1034, %r7, 4;
	shl.b32 	%r1035, %r1001, 2;
	setp.eq.b32 	%p24, %r1300, 0;
	shl.b32 	%r1036, %r1300, 1;
	shr.u32 	%r1037, %r6, 1;
	or.b32 	%r1038, %r1034, %r1035;
	or.b32 	%r1039, %r1036, %r1037;
	xor.b32 	%r1040, %r1038, %r1039;
	or.b32 	%r1041, %r1040, %r1033;
	add.s32 	%r1042, %r189, %r1041;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1043, %r1044, %r1045, %r1046}, [%r1042];
	mov.b32 	{%rs379, %rs380}, %r1043;
	mov.b32 	{%rs381, %rs382}, %r1044;
	mov.b32 	{%rs383, %rs384}, %r1045;
	mov.b32 	{%rs385, %rs386}, %r1046;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1047, %r1048, %r1049, %r1050}, [%r1042+1024];
	mov.b32 	{%rs387, %rs388}, %r1047;
	mov.b32 	{%rs389, %rs390}, %r1048;
	mov.b32 	{%rs391, %rs392}, %r1049;
	mov.b32 	{%rs393, %rs394}, %r1050;
	xor.b32 	%r1051, %r1041, 64;
	add.s32 	%r1052, %r189, %r1051;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1053, %r1054, %r1055, %r1056}, [%r1052];
	mov.b32 	{%rs395, %rs396}, %r1053;
	mov.b32 	{%rs397, %rs398}, %r1054;
	mov.b32 	{%rs399, %rs400}, %r1055;
	mov.b32 	{%rs401, %rs402}, %r1056;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1057, %r1058, %r1059, %r1060}, [%r1052+1024];
	mov.b32 	{%rs403, %rs404}, %r1057;
	mov.b32 	{%rs405, %rs406}, %r1058;
	mov.b32 	{%rs407, %rs408}, %r1059;
	mov.b32 	{%rs409, %rs410}, %r1060;
	bar.sync 	0;
	mov.b32 	%r742, {%rs50, %rs51};
	mov.b32 	%r743, {%rs52, %rs53};
	mov.b32 	%r744, {%rs54, %rs55};
	mov.b32 	%r745, {%rs56, %rs57};
	// begin inline asm
	st.shared.v4.b32 [ %r722 + 0 ], { %r742, %r743, %r744, %r745 };
	// end inline asm
	mov.b32 	%r746, {%rs58, %rs59};
	mov.b32 	%r747, {%rs60, %rs61};
	mov.b32 	%r748, {%rs62, %rs63};
	mov.b32 	%r749, {%rs64, %rs65};
	// begin inline asm
	st.shared.v4.b32 [ %r727 + 0 ], { %r746, %r747, %r748, %r749 };
	// end inline asm
	mov.b32 	%r750, {%rs66, %rs67};
	mov.b32 	%r751, {%rs68, %rs69};
	mov.b32 	%r752, {%rs70, %rs71};
	mov.b32 	%r753, {%rs72, %rs73};
	// begin inline asm
	st.shared.v4.b32 [ %r732 + 0 ], { %r750, %r751, %r752, %r753 };
	// end inline asm
	mov.b32 	%r754, {%rs74, %rs75};
	mov.b32 	%r755, {%rs76, %rs77};
	mov.b32 	%r756, {%rs78, %rs79};
	mov.b32 	%r757, {%rs80, %rs81};
	// begin inline asm
	st.shared.v4.b32 [ %r737 + 0 ], { %r754, %r755, %r756, %r757 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1061, %r1062, %r1063, %r1064}, [%r1042];
	mov.b32 	{%rs411, %rs412}, %r1061;
	mov.b32 	{%rs413, %rs414}, %r1062;
	mov.b32 	{%rs415, %rs416}, %r1063;
	mov.b32 	{%rs417, %rs418}, %r1064;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1065, %r1066, %r1067, %r1068}, [%r1042+1024];
	mov.b32 	{%rs419, %rs420}, %r1065;
	mov.b32 	{%rs421, %rs422}, %r1066;
	mov.b32 	{%rs423, %rs424}, %r1067;
	mov.b32 	{%rs425, %rs426}, %r1068;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1069, %r1070, %r1071, %r1072}, [%r1052];
	mov.b32 	{%rs427, %rs428}, %r1069;
	mov.b32 	{%rs429, %rs430}, %r1070;
	mov.b32 	{%rs431, %rs432}, %r1071;
	mov.b32 	{%rs433, %rs434}, %r1072;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1073, %r1074, %r1075, %r1076}, [%r1052+1024];
	mov.b32 	{%rs435, %rs436}, %r1073;
	mov.b32 	{%rs437, %rs438}, %r1074;
	mov.b32 	{%rs439, %rs440}, %r1075;
	mov.b32 	{%rs441, %rs442}, %r1076;
	bar.sync 	0;
	mov.b32 	%r758, {%rs82, %rs83};
	mov.b32 	%r759, {%rs84, %rs85};
	mov.b32 	%r760, {%rs86, %rs87};
	mov.b32 	%r761, {%rs88, %rs89};
	// begin inline asm
	st.shared.v4.b32 [ %r722 + 0 ], { %r758, %r759, %r760, %r761 };
	// end inline asm
	mov.b32 	%r762, {%rs90, %rs91};
	mov.b32 	%r763, {%rs92, %rs93};
	mov.b32 	%r764, {%rs94, %rs95};
	mov.b32 	%r765, {%rs96, %rs97};
	// begin inline asm
	st.shared.v4.b32 [ %r727 + 0 ], { %r762, %r763, %r764, %r765 };
	// end inline asm
	mov.b32 	%r766, {%rs98, %rs99};
	mov.b32 	%r767, {%rs100, %rs101};
	mov.b32 	%r768, {%rs102, %rs103};
	mov.b32 	%r769, {%rs104, %rs105};
	// begin inline asm
	st.shared.v4.b32 [ %r732 + 0 ], { %r766, %r767, %r768, %r769 };
	// end inline asm
	mov.b32 	%r770, {%rs106, %rs107};
	mov.b32 	%r771, {%rs108, %rs109};
	mov.b32 	%r772, {%rs110, %rs111};
	mov.b32 	%r773, {%rs112, %rs113};
	// begin inline asm
	st.shared.v4.b32 [ %r737 + 0 ], { %r770, %r771, %r772, %r773 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1077, %r1078, %r1079, %r1080}, [%r1042];
	mov.b32 	{%rs443, %rs444}, %r1077;
	mov.b32 	{%rs445, %rs446}, %r1078;
	mov.b32 	{%rs447, %rs448}, %r1079;
	mov.b32 	{%rs449, %rs450}, %r1080;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1081, %r1082, %r1083, %r1084}, [%r1042+1024];
	mov.b32 	{%rs451, %rs452}, %r1081;
	mov.b32 	{%rs453, %rs454}, %r1082;
	mov.b32 	{%rs455, %rs456}, %r1083;
	mov.b32 	{%rs457, %rs458}, %r1084;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1085, %r1086, %r1087, %r1088}, [%r1052];
	mov.b32 	{%rs459, %rs460}, %r1085;
	mov.b32 	{%rs461, %rs462}, %r1086;
	mov.b32 	{%rs463, %rs464}, %r1087;
	mov.b32 	{%rs465, %rs466}, %r1088;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1089, %r1090, %r1091, %r1092}, [%r1052+1024];
	mov.b32 	{%rs467, %rs468}, %r1089;
	mov.b32 	{%rs469, %rs470}, %r1090;
	mov.b32 	{%rs471, %rs472}, %r1091;
	mov.b32 	{%rs473, %rs474}, %r1092;
	bar.sync 	0;
	mov.b32 	%r774, {%rs114, %rs115};
	mov.b32 	%r775, {%rs116, %rs117};
	mov.b32 	%r776, {%rs118, %rs119};
	mov.b32 	%r777, {%rs120, %rs121};
	// begin inline asm
	st.shared.v4.b32 [ %r722 + 0 ], { %r774, %r775, %r776, %r777 };
	// end inline asm
	mov.b32 	%r778, {%rs122, %rs123};
	mov.b32 	%r779, {%rs124, %rs125};
	mov.b32 	%r780, {%rs126, %rs127};
	mov.b32 	%r781, {%rs128, %rs129};
	// begin inline asm
	st.shared.v4.b32 [ %r727 + 0 ], { %r778, %r779, %r780, %r781 };
	// end inline asm
	mov.b32 	%r782, {%rs130, %rs131};
	mov.b32 	%r783, {%rs132, %rs133};
	mov.b32 	%r784, {%rs134, %rs135};
	mov.b32 	%r785, {%rs136, %rs137};
	// begin inline asm
	st.shared.v4.b32 [ %r732 + 0 ], { %r782, %r783, %r784, %r785 };
	// end inline asm
	mov.b32 	%r786, {%rs138, %rs139};
	mov.b32 	%r787, {%rs140, %rs141};
	mov.b32 	%r788, {%rs142, %rs143};
	mov.b32 	%r789, {%rs144, %rs145};
	// begin inline asm
	st.shared.v4.b32 [ %r737 + 0 ], { %r786, %r787, %r788, %r789 };
	// end inline asm
	bar.sync 	0;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1093, %r1094, %r1095, %r1096}, [%r1042];
	mov.b32 	{%rs475, %rs476}, %r1094;
	mov.b32 	{%rs477, %rs478}, %r1096;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1097, %r1098, %r1099, %r1100}, [%r1042+1024];
	mov.b32 	{%rs479, %rs480}, %r1098;
	mov.b32 	{%rs481, %rs482}, %r1100;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1101, %r1102, %r1103, %r1104}, [%r1052];
	mov.b32 	{%rs483, %rs484}, %r1102;
	mov.b32 	{%rs485, %rs486}, %r1104;
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1105, %r1106, %r1107, %r1108}, [%r1052+1024];
	mov.b32 	{%rs487, %rs488}, %r1106;
	mov.b32 	{%rs489, %rs490}, %r1108;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	mov.b32 	%r1109, {%rs151, %rs151};
	mov.b32 	%r1110, -2147450880;
	fma.rn.bf16x2 	%r1111, %r1109, %r1349, %r1110;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs491, %r707;
	cvt.rn.bf16.f32 	%rs492, %r706;
	mov.b32 	%r1112, {%rs492, %rs491};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs493, %rs157, %rs492, %rs379;
	fma.rn.bf16 	%rs494, %rs158, %rs491, %rs380;
	fma.rn.bf16 	%rs495, %rs161, %rs492, %rs381;
	fma.rn.bf16 	%rs496, %rs162, %rs491, %rs382;
	fma.rn.bf16 	%rs497, %rs221, %rs492, %rs411;
	fma.rn.bf16 	%rs498, %rs222, %rs491, %rs412;
	fma.rn.bf16 	%rs499, %rs225, %rs492, %rs413;
	fma.rn.bf16 	%rs500, %rs226, %rs491, %rs414;
	fma.rn.bf16 	%rs501, %rs285, %rs492, %rs443;
	fma.rn.bf16 	%rs502, %rs286, %rs491, %rs444;
	fma.rn.bf16 	%rs503, %rs289, %rs492, %rs445;
	fma.rn.bf16 	%rs504, %rs290, %rs491, %rs446;
	fma.rn.bf16x2 	%r794, %r1111, %r1112, %r1093;
	fma.rn.bf16 	%rs505, %rs349, %rs492, %rs475;
	fma.rn.bf16 	%rs506, %rs350, %rs491, %rs476;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r1113, %r1109, %r1351, %r1110;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs507, %r709;
	cvt.rn.bf16.f32 	%rs508, %r708;
	mov.b32 	%r1114, {%rs508, %rs507};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs509, %rs165, %rs508, %rs383;
	fma.rn.bf16 	%rs510, %rs166, %rs507, %rs384;
	fma.rn.bf16 	%rs511, %rs169, %rs508, %rs385;
	fma.rn.bf16 	%rs512, %rs170, %rs507, %rs386;
	fma.rn.bf16 	%rs513, %rs229, %rs508, %rs415;
	fma.rn.bf16 	%rs514, %rs230, %rs507, %rs416;
	fma.rn.bf16 	%rs515, %rs233, %rs508, %rs417;
	fma.rn.bf16 	%rs516, %rs234, %rs507, %rs418;
	fma.rn.bf16 	%rs517, %rs293, %rs508, %rs447;
	fma.rn.bf16 	%rs518, %rs294, %rs507, %rs448;
	fma.rn.bf16 	%rs519, %rs297, %rs508, %rs449;
	fma.rn.bf16 	%rs520, %rs298, %rs507, %rs450;
	fma.rn.bf16x2 	%r814, %r1113, %r1114, %r1095;
	fma.rn.bf16 	%rs521, %rs353, %rs508, %rs477;
	fma.rn.bf16 	%rs522, %rs354, %rs507, %rs478;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r1115, %r1109, %r1353, %r1110;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs523, %r711;
	cvt.rn.bf16.f32 	%rs524, %r710;
	mov.b32 	%r1116, {%rs524, %rs523};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs525, %rs173, %rs524, %rs395;
	fma.rn.bf16 	%rs526, %rs174, %rs523, %rs396;
	fma.rn.bf16 	%rs527, %rs177, %rs524, %rs397;
	fma.rn.bf16 	%rs528, %rs178, %rs523, %rs398;
	fma.rn.bf16 	%rs529, %rs237, %rs524, %rs427;
	fma.rn.bf16 	%rs530, %rs238, %rs523, %rs428;
	fma.rn.bf16 	%rs531, %rs241, %rs524, %rs429;
	fma.rn.bf16 	%rs532, %rs242, %rs523, %rs430;
	fma.rn.bf16 	%rs533, %rs301, %rs524, %rs459;
	fma.rn.bf16 	%rs534, %rs302, %rs523, %rs460;
	fma.rn.bf16 	%rs535, %rs305, %rs524, %rs461;
	fma.rn.bf16 	%rs536, %rs306, %rs523, %rs462;
	fma.rn.bf16x2 	%r834, %r1115, %r1116, %r1101;
	fma.rn.bf16 	%rs537, %rs357, %rs524, %rs483;
	fma.rn.bf16 	%rs538, %rs358, %rs523, %rs484;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r1117, %r1109, %r1355, %r1110;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs539, %r713;
	cvt.rn.bf16.f32 	%rs540, %r712;
	mov.b32 	%r1118, {%rs540, %rs539};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs541, %rs181, %rs540, %rs399;
	fma.rn.bf16 	%rs542, %rs182, %rs539, %rs400;
	fma.rn.bf16 	%rs543, %rs185, %rs540, %rs401;
	fma.rn.bf16 	%rs544, %rs186, %rs539, %rs402;
	fma.rn.bf16 	%rs545, %rs245, %rs540, %rs431;
	fma.rn.bf16 	%rs546, %rs246, %rs539, %rs432;
	fma.rn.bf16 	%rs547, %rs249, %rs540, %rs433;
	fma.rn.bf16 	%rs548, %rs250, %rs539, %rs434;
	fma.rn.bf16 	%rs549, %rs309, %rs540, %rs463;
	fma.rn.bf16 	%rs550, %rs310, %rs539, %rs464;
	fma.rn.bf16 	%rs551, %rs313, %rs540, %rs465;
	fma.rn.bf16 	%rs552, %rs314, %rs539, %rs466;
	fma.rn.bf16x2 	%r854, %r1117, %r1118, %r1103;
	fma.rn.bf16 	%rs553, %rs361, %rs540, %rs485;
	fma.rn.bf16 	%rs554, %rs362, %rs539, %rs486;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r1119, %r1109, %r1357, %r1110;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs555, %r715;
	cvt.rn.bf16.f32 	%rs556, %r714;
	mov.b32 	%r1120, {%rs556, %rs555};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs557, %rs189, %rs556, %rs387;
	fma.rn.bf16 	%rs558, %rs190, %rs555, %rs388;
	fma.rn.bf16 	%rs559, %rs193, %rs556, %rs389;
	fma.rn.bf16 	%rs560, %rs194, %rs555, %rs390;
	fma.rn.bf16 	%rs561, %rs253, %rs556, %rs419;
	fma.rn.bf16 	%rs562, %rs254, %rs555, %rs420;
	fma.rn.bf16 	%rs563, %rs257, %rs556, %rs421;
	fma.rn.bf16 	%rs564, %rs258, %rs555, %rs422;
	fma.rn.bf16 	%rs565, %rs317, %rs556, %rs451;
	fma.rn.bf16 	%rs566, %rs318, %rs555, %rs452;
	fma.rn.bf16 	%rs567, %rs321, %rs556, %rs453;
	fma.rn.bf16 	%rs568, %rs322, %rs555, %rs454;
	fma.rn.bf16x2 	%r804, %r1119, %r1120, %r1097;
	fma.rn.bf16 	%rs569, %rs365, %rs556, %rs479;
	fma.rn.bf16 	%rs570, %rs366, %rs555, %rs480;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r1121, %r1109, %r1359, %r1110;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs571, %r717;
	cvt.rn.bf16.f32 	%rs572, %r716;
	mov.b32 	%r1122, {%rs572, %rs571};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs573, %rs197, %rs572, %rs391;
	fma.rn.bf16 	%rs574, %rs198, %rs571, %rs392;
	fma.rn.bf16 	%rs575, %rs201, %rs572, %rs393;
	fma.rn.bf16 	%rs576, %rs202, %rs571, %rs394;
	fma.rn.bf16 	%rs577, %rs261, %rs572, %rs423;
	fma.rn.bf16 	%rs578, %rs262, %rs571, %rs424;
	fma.rn.bf16 	%rs579, %rs265, %rs572, %rs425;
	fma.rn.bf16 	%rs580, %rs266, %rs571, %rs426;
	fma.rn.bf16 	%rs581, %rs325, %rs572, %rs455;
	fma.rn.bf16 	%rs582, %rs326, %rs571, %rs456;
	fma.rn.bf16 	%rs583, %rs329, %rs572, %rs457;
	fma.rn.bf16 	%rs584, %rs330, %rs571, %rs458;
	fma.rn.bf16x2 	%r824, %r1121, %r1122, %r1099;
	fma.rn.bf16 	%rs585, %rs369, %rs572, %rs481;
	fma.rn.bf16 	%rs586, %rs370, %rs571, %rs482;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r1123, %r1109, %r1361, %r1110;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs587, %r719;
	cvt.rn.bf16.f32 	%rs588, %r718;
	mov.b32 	%r1124, {%rs588, %rs587};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs589, %rs205, %rs588, %rs403;
	fma.rn.bf16 	%rs590, %rs206, %rs587, %rs404;
	fma.rn.bf16 	%rs591, %rs209, %rs588, %rs405;
	fma.rn.bf16 	%rs592, %rs210, %rs587, %rs406;
	fma.rn.bf16 	%rs593, %rs269, %rs588, %rs435;
	fma.rn.bf16 	%rs594, %rs270, %rs587, %rs436;
	fma.rn.bf16 	%rs595, %rs273, %rs588, %rs437;
	fma.rn.bf16 	%rs596, %rs274, %rs587, %rs438;
	fma.rn.bf16 	%rs597, %rs333, %rs588, %rs467;
	fma.rn.bf16 	%rs598, %rs334, %rs587, %rs468;
	fma.rn.bf16 	%rs599, %rs337, %rs588, %rs469;
	fma.rn.bf16 	%rs600, %rs338, %rs587, %rs470;
	fma.rn.bf16x2 	%r844, %r1123, %r1124, %r1105;
	fma.rn.bf16 	%rs601, %rs373, %rs588, %rs487;
	fma.rn.bf16 	%rs602, %rs374, %rs587, %rs488;
	.loc	1 186 32                        // sk03_fa_qkv.py:186:32
	fma.rn.bf16x2 	%r1125, %r1109, %r1363, %r1110;
	.loc	1 187 49                        // sk03_fa_qkv.py:187:49
	cvt.rn.bf16.f32 	%rs603, %r721;
	cvt.rn.bf16.f32 	%rs604, %r720;
	mov.b32 	%r1126, {%rs604, %rs603};
	.loc	1 188 11                        // sk03_fa_qkv.py:188:11
	fma.rn.bf16 	%rs605, %rs213, %rs604, %rs407;
	fma.rn.bf16 	%rs606, %rs214, %rs603, %rs408;
	fma.rn.bf16 	%rs607, %rs217, %rs604, %rs409;
	fma.rn.bf16 	%rs608, %rs218, %rs603, %rs410;
	fma.rn.bf16 	%rs609, %rs277, %rs604, %rs439;
	fma.rn.bf16 	%rs610, %rs278, %rs603, %rs440;
	fma.rn.bf16 	%rs611, %rs281, %rs604, %rs441;
	fma.rn.bf16 	%rs612, %rs282, %rs603, %rs442;
	fma.rn.bf16 	%rs613, %rs341, %rs604, %rs471;
	fma.rn.bf16 	%rs614, %rs342, %rs603, %rs472;
	fma.rn.bf16 	%rs615, %rs345, %rs604, %rs473;
	fma.rn.bf16 	%rs616, %rs346, %rs603, %rs474;
	fma.rn.bf16x2 	%r864, %r1125, %r1126, %r1107;
	fma.rn.bf16 	%rs617, %rs377, %rs604, %rs489;
	fma.rn.bf16 	%rs618, %rs378, %rs603, %rs490;
	bar.sync 	0;
	shl.b32 	%r1127, %r5, 14;
	shl.b32 	%r1128, %r5, 5;
	and.b32 	%r1129, %r1299, 3456;
	bfe.s32 	%r1130, %r2, 2, 1;
	and.b32 	%r1131, %r1130, 8208;
	or.b32 	%r1132, %r1128, %r1129;
	xor.b32 	%r1133, %r1131, %r1037;
	or.b32 	%r1134, %r1133, %r1132;
	or.b32 	%r1135, %r1134, %r1127;
	add.s32 	%r790, %r189, %r1135;
	mov.b32 	%r791, {%rs493, %rs494};
	mov.b32 	%r792, {%rs497, %rs498};
	mov.b32 	%r793, {%rs501, %rs502};
	// begin inline asm
	st.shared.v4.b32 [ %r790 + 0 ], { %r791, %r792, %r793, %r794 };
	// end inline asm
	add.s32 	%r795, %r790, 512;
	mov.b32 	%r796, {%rs495, %rs496};
	mov.b32 	%r797, {%rs499, %rs500};
	mov.b32 	%r798, {%rs503, %rs504};
	mov.b32 	%r799, {%rs505, %rs506};
	// begin inline asm
	st.shared.v4.b32 [ %r795 + 0 ], { %r796, %r797, %r798, %r799 };
	// end inline asm
	add.s32 	%r800, %r790, 4096;
	mov.b32 	%r801, {%rs557, %rs558};
	mov.b32 	%r802, {%rs561, %rs562};
	mov.b32 	%r803, {%rs565, %rs566};
	// begin inline asm
	st.shared.v4.b32 [ %r800 + 0 ], { %r801, %r802, %r803, %r804 };
	// end inline asm
	add.s32 	%r805, %r790, 4608;
	mov.b32 	%r806, {%rs559, %rs560};
	mov.b32 	%r807, {%rs563, %rs564};
	mov.b32 	%r808, {%rs567, %rs568};
	mov.b32 	%r809, {%rs569, %rs570};
	// begin inline asm
	st.shared.v4.b32 [ %r805 + 0 ], { %r806, %r807, %r808, %r809 };
	// end inline asm
	xor.b32 	%r1136, %r1135, 32;
	add.s32 	%r810, %r189, %r1136;
	mov.b32 	%r811, {%rs509, %rs510};
	mov.b32 	%r812, {%rs513, %rs514};
	mov.b32 	%r813, {%rs517, %rs518};
	// begin inline asm
	st.shared.v4.b32 [ %r810 + 0 ], { %r811, %r812, %r813, %r814 };
	// end inline asm
	add.s32 	%r815, %r810, 512;
	mov.b32 	%r816, {%rs511, %rs512};
	mov.b32 	%r817, {%rs515, %rs516};
	mov.b32 	%r818, {%rs519, %rs520};
	mov.b32 	%r819, {%rs521, %rs522};
	// begin inline asm
	st.shared.v4.b32 [ %r815 + 0 ], { %r816, %r817, %r818, %r819 };
	// end inline asm
	add.s32 	%r820, %r810, 4096;
	mov.b32 	%r821, {%rs573, %rs574};
	mov.b32 	%r822, {%rs577, %rs578};
	mov.b32 	%r823, {%rs581, %rs582};
	// begin inline asm
	st.shared.v4.b32 [ %r820 + 0 ], { %r821, %r822, %r823, %r824 };
	// end inline asm
	add.s32 	%r825, %r810, 4608;
	mov.b32 	%r826, {%rs575, %rs576};
	mov.b32 	%r827, {%rs579, %rs580};
	mov.b32 	%r828, {%rs583, %rs584};
	mov.b32 	%r829, {%rs585, %rs586};
	// begin inline asm
	st.shared.v4.b32 [ %r825 + 0 ], { %r826, %r827, %r828, %r829 };
	// end inline asm
	xor.b32 	%r1137, %r1135, 64;
	add.s32 	%r830, %r189, %r1137;
	mov.b32 	%r831, {%rs525, %rs526};
	mov.b32 	%r832, {%rs529, %rs530};
	mov.b32 	%r833, {%rs533, %rs534};
	// begin inline asm
	st.shared.v4.b32 [ %r830 + 0 ], { %r831, %r832, %r833, %r834 };
	// end inline asm
	add.s32 	%r835, %r830, 512;
	mov.b32 	%r836, {%rs527, %rs528};
	mov.b32 	%r837, {%rs531, %rs532};
	mov.b32 	%r838, {%rs535, %rs536};
	mov.b32 	%r839, {%rs537, %rs538};
	// begin inline asm
	st.shared.v4.b32 [ %r835 + 0 ], { %r836, %r837, %r838, %r839 };
	// end inline asm
	add.s32 	%r840, %r830, 4096;
	mov.b32 	%r841, {%rs589, %rs590};
	mov.b32 	%r842, {%rs593, %rs594};
	mov.b32 	%r843, {%rs597, %rs598};
	// begin inline asm
	st.shared.v4.b32 [ %r840 + 0 ], { %r841, %r842, %r843, %r844 };
	// end inline asm
	add.s32 	%r845, %r830, 4608;
	mov.b32 	%r846, {%rs591, %rs592};
	mov.b32 	%r847, {%rs595, %rs596};
	mov.b32 	%r848, {%rs599, %rs600};
	mov.b32 	%r849, {%rs601, %rs602};
	// begin inline asm
	st.shared.v4.b32 [ %r845 + 0 ], { %r846, %r847, %r848, %r849 };
	// end inline asm
	xor.b32 	%r1138, %r1135, 96;
	add.s32 	%r850, %r189, %r1138;
	mov.b32 	%r851, {%rs541, %rs542};
	mov.b32 	%r852, {%rs545, %rs546};
	mov.b32 	%r853, {%rs549, %rs550};
	// begin inline asm
	st.shared.v4.b32 [ %r850 + 0 ], { %r851, %r852, %r853, %r854 };
	// end inline asm
	add.s32 	%r855, %r850, 512;
	mov.b32 	%r856, {%rs543, %rs544};
	mov.b32 	%r857, {%rs547, %rs548};
	mov.b32 	%r858, {%rs551, %rs552};
	mov.b32 	%r859, {%rs553, %rs554};
	// begin inline asm
	st.shared.v4.b32 [ %r855 + 0 ], { %r856, %r857, %r858, %r859 };
	// end inline asm
	add.s32 	%r860, %r850, 4096;
	mov.b32 	%r861, {%rs605, %rs606};
	mov.b32 	%r862, {%rs609, %rs610};
	mov.b32 	%r863, {%rs613, %rs614};
	// begin inline asm
	st.shared.v4.b32 [ %r860 + 0 ], { %r861, %r862, %r863, %r864 };
	// end inline asm
	add.s32 	%r865, %r850, 4608;
	mov.b32 	%r866, {%rs607, %rs608};
	mov.b32 	%r867, {%rs611, %rs612};
	mov.b32 	%r868, {%rs615, %rs616};
	mov.b32 	%r869, {%rs617, %rs618};
	// begin inline asm
	st.shared.v4.b32 [ %r865 + 0 ], { %r866, %r867, %r868, %r869 };
	// end inline asm
	bar.sync 	0;
	and.b32 	%r1139, %r989, 896;
	shl.b32 	%r1140, %r991, 9;
	selp.b32 	%r1141, 0, 8208, %p24;
	or.b32 	%r1142, %r1030, %r1139;
	xor.b32 	%r1143, %r1142, %r1141;
	or.b32 	%r1144, %r1143, %r1140;
	add.s32 	%r1145, %r189, %r1144;
	ld.shared.v4.b32 	{%r870, %r886, %r902, %r918}, [%r1145];
	ld.shared.v4.b32 	{%r874, %r890, %r906, %r922}, [%r1145+1024];
	ld.shared.v4.b32 	{%r878, %r894, %r910, %r926}, [%r1145+2048];
	ld.shared.v4.b32 	{%r882, %r898, %r914, %r930}, [%r1145+3072];
	xor.b32 	%r1146, %r1144, 32;
	add.s32 	%r1147, %r189, %r1146;
	ld.shared.v4.b32 	{%r871, %r887, %r903, %r919}, [%r1147+16384];
	ld.shared.v4.b32 	{%r875, %r891, %r907, %r923}, [%r1147+17408];
	ld.shared.v4.b32 	{%r879, %r895, %r911, %r927}, [%r1147+18432];
	ld.shared.v4.b32 	{%r883, %r899, %r915, %r931}, [%r1147+19456];
	xor.b32 	%r1148, %r1144, 64;
	add.s32 	%r1149, %r189, %r1148;
	ld.shared.v4.b32 	{%r872, %r888, %r904, %r920}, [%r1149+32768];
	ld.shared.v4.b32 	{%r876, %r892, %r908, %r924}, [%r1149+33792];
	ld.shared.v4.b32 	{%r880, %r896, %r912, %r928}, [%r1149+34816];
	ld.shared.v4.b32 	{%r884, %r900, %r916, %r932}, [%r1149+35840];
	xor.b32 	%r1150, %r1144, 96;
	add.s32 	%r1151, %r189, %r1150;
	ld.shared.v4.b32 	{%r873, %r889, %r905, %r921}, [%r1151+49152];
	ld.shared.v4.b32 	{%r877, %r893, %r909, %r925}, [%r1151+50176];
	ld.shared.v4.b32 	{%r881, %r897, %r913, %r929}, [%r1151+51200];
	ld.shared.v4.b32 	{%r885, %r901, %r917, %r933}, [%r1151+52224];
	.loc	1 195 31                        // sk03_fa_qkv.py:195:31
	setp.lt.s32 	%p25, %r955, %r21;
	setp.lt.s32 	%p26, %r984, %r21;
	setp.lt.s32 	%p27, %r982, %r21;
	setp.lt.s32 	%p28, %r980, %r21;
	setp.lt.s32 	%p29, %r978, %r21;
	setp.lt.s32 	%p30, %r976, %r21;
	setp.lt.s32 	%p31, %r974, %r21;
	setp.lt.s32 	%p32, %r972, %r21;
	setp.lt.s32 	%p33, %r970, %r21;
	setp.lt.s32 	%p34, %r968, %r21;
	setp.lt.s32 	%p35, %r966, %r21;
	setp.lt.s32 	%p36, %r964, %r21;
	setp.lt.s32 	%p37, %r962, %r21;
	setp.lt.s32 	%p38, %r960, %r21;
	setp.lt.s32 	%p39, %r958, %r21;
	setp.lt.s32 	%p40, %r956, %r21;
	.loc	1 195 54                        // sk03_fa_qkv.py:195:54
	setp.lt.s32 	%p41, %r935, %r22;
	.loc	1 195 37                        // sk03_fa_qkv.py:195:37
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
	.loc	1 193 35                        // sk03_fa_qkv.py:193:35
	mul.lo.s32 	%r1152, %r955, %r24;
	mul.lo.s32 	%r1153, %r984, %r24;
	mul.lo.s32 	%r1154, %r982, %r24;
	mul.lo.s32 	%r1155, %r980, %r24;
	mul.lo.s32 	%r1156, %r978, %r24;
	mul.lo.s32 	%r1157, %r976, %r24;
	mul.lo.s32 	%r1158, %r974, %r24;
	mul.lo.s32 	%r1159, %r972, %r24;
	mul.lo.s32 	%r1160, %r970, %r24;
	mul.lo.s32 	%r1161, %r968, %r24;
	mul.lo.s32 	%r1162, %r966, %r24;
	mul.lo.s32 	%r1163, %r964, %r24;
	mul.lo.s32 	%r1164, %r962, %r24;
	mul.lo.s32 	%r1165, %r960, %r24;
	mul.lo.s32 	%r1166, %r958, %r24;
	mul.lo.s32 	%r1167, %r956, %r24;
	.loc	1 193 18                        // sk03_fa_qkv.py:193:18
	mad.wide.s32 	%rd290, %r1152, 2, %rd34;
	mad.wide.s32 	%rd291, %r1153, 2, %rd34;
	mad.wide.s32 	%rd292, %r1154, 2, %rd34;
	mad.wide.s32 	%rd293, %r1155, 2, %rd34;
	mad.wide.s32 	%rd294, %r1156, 2, %rd34;
	mad.wide.s32 	%rd295, %r1157, 2, %rd34;
	mad.wide.s32 	%rd296, %r1158, 2, %rd34;
	mad.wide.s32 	%rd297, %r1159, 2, %rd34;
	mad.wide.s32 	%rd298, %r1160, 2, %rd34;
	mad.wide.s32 	%rd299, %r1161, 2, %rd34;
	mad.wide.s32 	%rd300, %r1162, 2, %rd34;
	mad.wide.s32 	%rd301, %r1163, 2, %rd34;
	mad.wide.s32 	%rd302, %r1164, 2, %rd34;
	mad.wide.s32 	%rd303, %r1165, 2, %rd34;
	mad.wide.s32 	%rd304, %r1166, 2, %rd34;
	mad.wide.s32 	%rd305, %r1167, 2, %rd34;
	.loc	1 193 50                        // sk03_fa_qkv.py:193:50
	mul.wide.s32 	%rd306, %r935, 2;
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
	.loc	1 194 8                         // sk03_fa_qkv.py:194:8
	// begin inline asm
	@%p8 st.global.v4.b32 [ %rd250 + 0 ], { %r870, %r871, %r872, %r873 };
	// end inline asm
	// begin inline asm
	@%p9 st.global.v4.b32 [ %rd251 + 0 ], { %r874, %r875, %r876, %r877 };
	// end inline asm
	// begin inline asm
	@%p10 st.global.v4.b32 [ %rd252 + 0 ], { %r878, %r879, %r880, %r881 };
	// end inline asm
	// begin inline asm
	@%p11 st.global.v4.b32 [ %rd253 + 0 ], { %r882, %r883, %r884, %r885 };
	// end inline asm
	// begin inline asm
	@%p12 st.global.v4.b32 [ %rd254 + 0 ], { %r886, %r887, %r888, %r889 };
	// end inline asm
	// begin inline asm
	@%p13 st.global.v4.b32 [ %rd255 + 0 ], { %r890, %r891, %r892, %r893 };
	// end inline asm
	// begin inline asm
	@%p14 st.global.v4.b32 [ %rd256 + 0 ], { %r894, %r895, %r896, %r897 };
	// end inline asm
	// begin inline asm
	@%p15 st.global.v4.b32 [ %rd257 + 0 ], { %r898, %r899, %r900, %r901 };
	// end inline asm
	// begin inline asm
	@%p16 st.global.v4.b32 [ %rd258 + 0 ], { %r902, %r903, %r904, %r905 };
	// end inline asm
	// begin inline asm
	@%p17 st.global.v4.b32 [ %rd259 + 0 ], { %r906, %r907, %r908, %r909 };
	// end inline asm
	// begin inline asm
	@%p18 st.global.v4.b32 [ %rd260 + 0 ], { %r910, %r911, %r912, %r913 };
	// end inline asm
	// begin inline asm
	@%p19 st.global.v4.b32 [ %rd261 + 0 ], { %r914, %r915, %r916, %r917 };
	// end inline asm
	// begin inline asm
	@%p20 st.global.v4.b32 [ %rd262 + 0 ], { %r918, %r919, %r920, %r921 };
	// end inline asm
	// begin inline asm
	@%p21 st.global.v4.b32 [ %rd263 + 0 ], { %r922, %r923, %r924, %r925 };
	// end inline asm
	// begin inline asm
	@%p22 st.global.v4.b32 [ %rd264 + 0 ], { %r926, %r927, %r928, %r929 };
	// end inline asm
	// begin inline asm
	@%p23 st.global.v4.b32 [ %rd265 + 0 ], { %r930, %r931, %r932, %r933 };
	// end inline asm
	.loc	1 192 4                         // sk03_fa_qkv.py:192:4
	ret;
$L__tmp4:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk03_fa_qkv.py"
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
.b32 157                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0x96 DW_TAG_compile_unit
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
.b8 51
.b8 95
.b8 102
.b8 97
.b8 95
.b8 113
.b8 107
.b8 118
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
.b8 2                                   // Abbrev [2] 0x44:0x16 DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 51
.b8 95
.b8 102
.b8 97
.b8 95
.b8 113
.b8 107
.b8 118
.b8 95
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 0
.b8 1                                   // DW_AT_inline
.b8 3                                   // Abbrev [3] 0x5a:0x46 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 68                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x6f:0x18 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp2                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 155                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 4                                   // Abbrev [4] 0x87:0x18 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp2                           // DW_AT_low_pc
.b64 $L__tmp3                           // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 156                                 // DW_AT_call_line
.b8 27                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 0                                   // End Of Children Mark
	}
	.section	.debug_macinfo	{	}
"""

_VAR_11 = _Nativo(
    "sk03_fa_qkv/tile256x128x64_shift1_abi16",
    _PTX_11, "_sk03_fa_qkv_kernel",
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
_POR_CFG.setdefault(((128, 128, 128, 8, 8, 3), False), []).append(_VAR_4)
_POR_CFG.setdefault(((128, 128, 128, 8, 8, 3), False), []).append(_VAR_5)
_POR_CFG.setdefault(((128, 128, 128, 8, 8, 3), True), []).append(_VAR_6)
_POR_CFG.setdefault(((128, 128, 128, 8, 8, 3), True), []).append(_VAR_7)
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


# --- quant: PTX embebido (E1=32 lop3, E2=0 mul) ---

_QPTX0 = r"""//
// Generated by LLVM NVPTX Back-End
//

.version 8.7
.target sm_86
.address_size 64

	// .globl	_sk03_rmsnorm_quant_kernel // -- Begin function _sk03_rmsnorm_quant_kernel
.extern .shared .align 16 .b8 global_smem[];
.global .align 1 .b8 _$_str[11] = {95, 95, 67, 85, 68, 65, 95, 70, 84, 90};
                                        // @_sk03_rmsnorm_quant_kernel
.visible .entry _sk03_rmsnorm_quant_kernel(
	.param .u64 .ptr .global .align 1 _sk03_rmsnorm_quant_kernel_param_0,
	.param .u64 .ptr .global .align 1 _sk03_rmsnorm_quant_kernel_param_1,
	.param .u64 .ptr .global .align 1 _sk03_rmsnorm_quant_kernel_param_2,
	.param .u64 .ptr .global .align 1 _sk03_rmsnorm_quant_kernel_param_3,
	.param .u32 _sk03_rmsnorm_quant_kernel_param_4,
	.param .u32 _sk03_rmsnorm_quant_kernel_param_5,
	.param .u32 _sk03_rmsnorm_quant_kernel_param_6,
	.param .u64 .ptr .global .align 1 _sk03_rmsnorm_quant_kernel_param_7,
	.param .u64 .ptr .global .align 1 _sk03_rmsnorm_quant_kernel_param_8
)
.reqntid 256
{
	.reg .pred 	%p<40>;
	.reg .b16 	%rs<65>;
	.reg .b32 	%r<538>;
	.reg .b64 	%rd<21>;
	.loc	1 118 0                         // sk03_fa_qkv.py:118:0
$L__func_begin0:
	.loc	1 118 0                         // sk03_fa_qkv.py:118:0

// %bb.0:                               // %__nv_rsqrtf.exit
	ld.param.b64 	%rd12, [_sk03_rmsnorm_quant_kernel_param_0];
	ld.param.b64 	%rd13, [_sk03_rmsnorm_quant_kernel_param_1];
$L__tmp0:
	.loc	1 128 24                        // sk03_fa_qkv.py:128:24
	mov.u32 	%r51, %ctaid.x;
	ld.param.b64 	%rd14, [_sk03_rmsnorm_quant_kernel_param_2];
	.loc	1 129 24                        // sk03_fa_qkv.py:129:24
	mov.u32 	%r52, %tid.x;
	and.b32 	%r53, %r52, 255;
	ld.param.b64 	%rd15, [_sk03_rmsnorm_quant_kernel_param_3];
	and.b32 	%r54, %r52, 31;
	ld.param.b32 	%r55, [_sk03_rmsnorm_quant_kernel_param_4];
	shr.u32 	%r56, %r52, 5;
	ld.param.b32 	%r57, [_sk03_rmsnorm_quant_kernel_param_5];
	shl.b32 	%r58, %r52, 4;
	ld.param.b32 	%r59, [_sk03_rmsnorm_quant_kernel_param_6];
	and.b32 	%r60, %r58, 4080;
	or.b32 	%r61, %r60, 4096;
	.loc	1 130 18                        // sk03_fa_qkv.py:130:18
	setp.lt.s32 	%p1, %r60, %r55;
	setp.lt.s32 	%p2, %r61, %r55;
	.loc	1 131 30                        // sk03_fa_qkv.py:131:30
	mul.lo.s32 	%r62, %r57, %r51;
	.loc	1 131 24                        // sk03_fa_qkv.py:131:24
	mad.wide.s32 	%rd16, %r62, 2, %rd12;
	.loc	1 131 42                        // sk03_fa_qkv.py:131:42
	cvt.u64.u32 	%rd17, %r60;
	mul.wide.u32 	%rd18, %r60, 2;
	add.s64 	%rd1, %rd16, %rd18;
	add.s64 	%rd2, %rd1, 16;
	add.s64 	%rd3, %rd1, 8192;
	add.s64 	%rd4, %rd1, 8208;
	mov.b32 	%r5, 0;
	.loc	1 131 16                        // sk03_fa_qkv.py:131:16
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
	.loc	1 131 73                        // sk03_fa_qkv.py:131:73
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
	.loc	1 132 24                        // sk03_fa_qkv.py:132:24
	add.s64 	%rd5, %rd13, %rd18;
	add.s64 	%rd6, %rd5, 16;
	add.s64 	%rd7, %rd5, 8192;
	add.s64 	%rd8, %rd5, 8208;
	.loc	1 132 16                        // sk03_fa_qkv.py:132:16
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
	.loc	1 133 32                        // sk03_fa_qkv.py:133:32
	mul.f32 	%r95, %r66, %r66;
$L__tmp1:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk03_fa_qkv.py:133:28 ] ]
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
	.loc	2 293 36                        // standard.py:293:36 @[ sk03_fa_qkv.py:133:28 ]
	shfl.sync.bfly.b32 	%r127, %r126, 16, 31, -1;
$L__tmp3:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk03_fa_qkv.py:133:28 ] ]
	add.f32 	%r128, %r126, %r127;
$L__tmp4:
	.loc	2 293 36                        // standard.py:293:36 @[ sk03_fa_qkv.py:133:28 ]
	shfl.sync.bfly.b32 	%r129, %r128, 8, 31, -1;
$L__tmp5:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk03_fa_qkv.py:133:28 ] ]
	add.f32 	%r130, %r128, %r129;
$L__tmp6:
	.loc	2 293 36                        // standard.py:293:36 @[ sk03_fa_qkv.py:133:28 ]
	shfl.sync.bfly.b32 	%r131, %r130, 4, 31, -1;
$L__tmp7:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk03_fa_qkv.py:133:28 ] ]
	add.f32 	%r132, %r130, %r131;
$L__tmp8:
	.loc	2 293 36                        // standard.py:293:36 @[ sk03_fa_qkv.py:133:28 ]
	shfl.sync.bfly.b32 	%r133, %r132, 2, 31, -1;
$L__tmp9:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk03_fa_qkv.py:133:28 ] ]
	add.f32 	%r134, %r132, %r133;
$L__tmp10:
	.loc	2 293 36                        // standard.py:293:36 @[ sk03_fa_qkv.py:133:28 ]
	shfl.sync.bfly.b32 	%r135, %r134, 1, 31, -1;
$L__tmp11:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk03_fa_qkv.py:133:28 ] ]
	add.f32 	%r35, %r134, %r135;
$L__tmp12:
	.loc	2 293 36                        // standard.py:293:36 @[ sk03_fa_qkv.py:133:28 ]
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
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk03_fa_qkv.py:133:28 ] ]
	add.f32 	%r141, %r36, %r140;
$L__tmp14:
	.loc	2 293 36                        // standard.py:293:36 @[ sk03_fa_qkv.py:133:28 ]
	shfl.sync.bfly.b32 	%r142, %r141, 2, 31, -1;
$L__tmp15:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk03_fa_qkv.py:133:28 ] ]
	add.f32 	%r143, %r141, %r142;
$L__tmp16:
	.loc	2 293 36                        // standard.py:293:36 @[ sk03_fa_qkv.py:133:28 ]
	shfl.sync.bfly.b32 	%r144, %r143, 1, 31, -1;
$L__tmp17:
	.loc	2 263 15                        // standard.py:263:15 @[ standard.py:293:36 @[ sk03_fa_qkv.py:133:28 ] ]
	add.f32 	%r38, %r143, %r144;
$L__tmp18:
	.loc	2 293 36                        // standard.py:293:36 @[ sk03_fa_qkv.py:133:28 ]
	and.b32 	%r145, %r52, 7;
	setp.eq.b32 	%p7, %r145, 0;
	and.pred 	%p5, %p4, %p7;
	// begin inline asm
	@%p5 st.shared.b32 [ %r37 + 0 ], %r38;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r146, [global_smem];
$L__tmp19:
	.loc	1 133 45                        // sk03_fa_qkv.py:133:45
	cvt.rn.f32.s32 	%r147, %r55;
	div.full.f32 	%r148, %r146, %r147;
	.loc	1 133 49                        // sk03_fa_qkv.py:133:49
	add.f32 	%r149, %r148, 0f358637BD;
	.loc	1 133 21                        // sk03_fa_qkv.py:133:21
	rsqrt.approx.ftz.f32 	%r150, %r149;
	.loc	1 133 12                        // sk03_fa_qkv.py:133:12
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
	.loc	2 191 40                        // standard.py:191:40 @[ sk03_fa_qkv.py:134:29 ]
	bar.sync 	0;
$L__tmp21:
	.loc	1 139 27                        // sk03_fa_qkv.py:139:27
	mul.lo.s32 	%r183, %r59, %r51;
	.loc	1 139 21                        // sk03_fa_qkv.py:139:21
	cvt.s64.s32 	%rd19, %r183;
	add.s64 	%rd20, %rd14, %rd19;
	.loc	1 139 39                        // sk03_fa_qkv.py:139:39
	add.s64 	%rd9, %rd20, %rd17;
	add.s64 	%rd10, %rd9, 4096;
	.loc	1 132 55                        // sk03_fa_qkv.py:132:55
	mov.b32 	{%rs33, %rs34}, %r24;
	cvt.f32.bf16 	%r184, %rs33;
	cvt.f32.bf16 	%r185, %rs34;
	mov.b32 	{%rs35, %rs36}, %r25;
	cvt.f32.bf16 	%r186, %rs35;
	cvt.f32.bf16 	%r187, %rs36;
	.loc	1 133 56                        // sk03_fa_qkv.py:133:56
	mul.f32 	%r188, %r166, %r187;
	mul.f32 	%r189, %r165, %r186;
	mul.f32 	%r190, %r164, %r185;
	mul.f32 	%r191, %r163, %r184;
	.loc	1 134 36                        // sk03_fa_qkv.py:134:36
	abs.f32 	%r192, %r191;
	abs.f32 	%r193, %r190;
	abs.f32 	%r194, %r189;
	abs.f32 	%r195, %r188;
	.loc	1 132 55                        // sk03_fa_qkv.py:132:55
	mov.b32 	{%rs37, %rs38}, %r22;
	cvt.f32.bf16 	%r196, %rs37;
	cvt.f32.bf16 	%r197, %rs38;
	mov.b32 	{%rs39, %rs40}, %r23;
	cvt.f32.bf16 	%r198, %rs39;
	cvt.f32.bf16 	%r199, %rs40;
	.loc	1 133 56                        // sk03_fa_qkv.py:133:56
	mul.f32 	%r200, %r162, %r199;
	mul.f32 	%r201, %r161, %r198;
	mul.f32 	%r202, %r160, %r197;
	mul.f32 	%r203, %r159, %r196;
	.loc	1 134 36                        // sk03_fa_qkv.py:134:36
	abs.f32 	%r204, %r203;
	abs.f32 	%r205, %r202;
	abs.f32 	%r206, %r201;
	abs.f32 	%r207, %r200;
	.loc	1 132 55                        // sk03_fa_qkv.py:132:55
	mov.b32 	{%rs41, %rs42}, %r20;
	cvt.f32.bf16 	%r208, %rs41;
	cvt.f32.bf16 	%r209, %rs42;
	mov.b32 	{%rs43, %rs44}, %r21;
	cvt.f32.bf16 	%r210, %rs43;
	cvt.f32.bf16 	%r211, %rs44;
	.loc	1 133 56                        // sk03_fa_qkv.py:133:56
	mul.f32 	%r212, %r158, %r211;
	mul.f32 	%r213, %r157, %r210;
	mul.f32 	%r214, %r156, %r209;
	mul.f32 	%r215, %r155, %r208;
	.loc	1 134 36                        // sk03_fa_qkv.py:134:36
	abs.f32 	%r216, %r215;
	abs.f32 	%r217, %r214;
	abs.f32 	%r218, %r213;
	abs.f32 	%r219, %r212;
	.loc	1 132 55                        // sk03_fa_qkv.py:132:55
	mov.b32 	{%rs45, %rs46}, %r18;
	cvt.f32.bf16 	%r220, %rs45;
	cvt.f32.bf16 	%r221, %rs46;
	mov.b32 	{%rs47, %rs48}, %r19;
	cvt.f32.bf16 	%r222, %rs47;
	cvt.f32.bf16 	%r223, %rs48;
	.loc	1 133 56                        // sk03_fa_qkv.py:133:56
	mul.f32 	%r224, %r154, %r223;
	mul.f32 	%r225, %r153, %r222;
	mul.f32 	%r226, %r152, %r221;
	mul.f32 	%r227, %r151, %r220;
	.loc	1 134 36                        // sk03_fa_qkv.py:134:36
	abs.f32 	%r228, %r227;
	abs.f32 	%r229, %r226;
	abs.f32 	%r230, %r225;
	abs.f32 	%r231, %r224;
$L__tmp22:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk03_fa_qkv.py:134:29 ] ]
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
	.loc	1 132 55                        // sk03_fa_qkv.py:132:55
	mov.b32 	{%rs49, %rs50}, %r32;
	cvt.f32.bf16 	%r247, %rs49;
	cvt.f32.bf16 	%r248, %rs50;
	mov.b32 	{%rs51, %rs52}, %r33;
	cvt.f32.bf16 	%r249, %rs51;
	cvt.f32.bf16 	%r250, %rs52;
	.loc	1 133 56                        // sk03_fa_qkv.py:133:56
	mul.f32 	%r251, %r182, %r250;
	mul.f32 	%r252, %r181, %r249;
	mul.f32 	%r253, %r180, %r248;
	mul.f32 	%r254, %r179, %r247;
	.loc	1 134 36                        // sk03_fa_qkv.py:134:36
	abs.f32 	%r255, %r254;
	abs.f32 	%r256, %r253;
	abs.f32 	%r257, %r252;
	abs.f32 	%r258, %r251;
	.loc	1 132 55                        // sk03_fa_qkv.py:132:55
	mov.b32 	{%rs53, %rs54}, %r30;
	cvt.f32.bf16 	%r259, %rs53;
	cvt.f32.bf16 	%r260, %rs54;
	mov.b32 	{%rs55, %rs56}, %r31;
	cvt.f32.bf16 	%r261, %rs55;
	cvt.f32.bf16 	%r262, %rs56;
	.loc	1 133 56                        // sk03_fa_qkv.py:133:56
	mul.f32 	%r263, %r178, %r262;
	mul.f32 	%r264, %r177, %r261;
	mul.f32 	%r265, %r176, %r260;
	mul.f32 	%r266, %r175, %r259;
	.loc	1 134 36                        // sk03_fa_qkv.py:134:36
	abs.f32 	%r267, %r266;
	abs.f32 	%r268, %r265;
	abs.f32 	%r269, %r264;
	abs.f32 	%r270, %r263;
	.loc	1 132 55                        // sk03_fa_qkv.py:132:55
	mov.b32 	{%rs57, %rs58}, %r28;
	cvt.f32.bf16 	%r271, %rs57;
	cvt.f32.bf16 	%r272, %rs58;
	mov.b32 	{%rs59, %rs60}, %r29;
	cvt.f32.bf16 	%r273, %rs59;
	cvt.f32.bf16 	%r274, %rs60;
	.loc	1 133 56                        // sk03_fa_qkv.py:133:56
	mul.f32 	%r275, %r174, %r274;
	mul.f32 	%r276, %r173, %r273;
	mul.f32 	%r277, %r172, %r272;
	mul.f32 	%r278, %r171, %r271;
	.loc	1 134 36                        // sk03_fa_qkv.py:134:36
	abs.f32 	%r279, %r278;
	abs.f32 	%r280, %r277;
	abs.f32 	%r281, %r276;
	abs.f32 	%r282, %r275;
	.loc	1 132 55                        // sk03_fa_qkv.py:132:55
	mov.b32 	{%rs61, %rs62}, %r26;
	cvt.f32.bf16 	%r283, %rs61;
	cvt.f32.bf16 	%r284, %rs62;
	mov.b32 	{%rs63, %rs64}, %r27;
	cvt.f32.bf16 	%r285, %rs63;
	cvt.f32.bf16 	%r286, %rs64;
	.loc	1 133 56                        // sk03_fa_qkv.py:133:56
	mul.f32 	%r287, %r170, %r286;
	mul.f32 	%r288, %r169, %r285;
	mul.f32 	%r289, %r168, %r284;
	mul.f32 	%r290, %r167, %r283;
	.loc	1 134 36                        // sk03_fa_qkv.py:134:36
	abs.f32 	%r291, %r290;
	abs.f32 	%r292, %r289;
	abs.f32 	%r293, %r288;
	abs.f32 	%r294, %r287;
$L__tmp24:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk03_fa_qkv.py:134:29 ] ]
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
	.loc	2 191 40                        // standard.py:191:40 @[ sk03_fa_qkv.py:134:29 ]
	shfl.sync.bfly.b32 	%r311, %r310, 16, 31, -1;
$L__tmp26:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk03_fa_qkv.py:134:29 ] ]
	max.f32 	%r312, %r310, %r311;
$L__tmp27:
	.loc	2 191 40                        // standard.py:191:40 @[ sk03_fa_qkv.py:134:29 ]
	shfl.sync.bfly.b32 	%r313, %r312, 8, 31, -1;
$L__tmp28:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk03_fa_qkv.py:134:29 ] ]
	max.f32 	%r314, %r312, %r313;
$L__tmp29:
	.loc	2 191 40                        // standard.py:191:40 @[ sk03_fa_qkv.py:134:29 ]
	shfl.sync.bfly.b32 	%r315, %r314, 4, 31, -1;
$L__tmp30:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk03_fa_qkv.py:134:29 ] ]
	max.f32 	%r316, %r314, %r315;
$L__tmp31:
	.loc	2 191 40                        // standard.py:191:40 @[ sk03_fa_qkv.py:134:29 ]
	shfl.sync.bfly.b32 	%r317, %r316, 2, 31, -1;
$L__tmp32:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk03_fa_qkv.py:134:29 ] ]
	max.f32 	%r318, %r316, %r317;
$L__tmp33:
	.loc	2 191 40                        // standard.py:191:40 @[ sk03_fa_qkv.py:134:29 ]
	shfl.sync.bfly.b32 	%r319, %r318, 1, 31, -1;
$L__tmp34:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk03_fa_qkv.py:134:29 ] ]
	max.f32 	%r39, %r318, %r319;
$L__tmp35:
	.loc	2 191 40                        // standard.py:191:40 @[ sk03_fa_qkv.py:134:29 ]
	// begin inline asm
	@%p3 st.shared.b32 [ %r34 + 0 ], %r39;
	// end inline asm
	bar.sync 	0;
	// begin inline asm
	@%p4 ld.shared.b32 %r40, [ %r37 + 0 ];
	// end inline asm
	shfl.sync.bfly.b32 	%r320, %r40, 4, 31, -1;
$L__tmp36:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk03_fa_qkv.py:134:29 ] ]
	max.f32 	%r321, %r40, %r320;
$L__tmp37:
	.loc	2 191 40                        // standard.py:191:40 @[ sk03_fa_qkv.py:134:29 ]
	shfl.sync.bfly.b32 	%r322, %r321, 2, 31, -1;
$L__tmp38:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk03_fa_qkv.py:134:29 ] ]
	max.f32 	%r323, %r321, %r322;
$L__tmp39:
	.loc	2 191 40                        // standard.py:191:40 @[ sk03_fa_qkv.py:134:29 ]
	shfl.sync.bfly.b32 	%r324, %r323, 1, 31, -1;
$L__tmp40:
	.loc	2 170 27                        // standard.py:170:27 @[ standard.py:191:40 @[ sk03_fa_qkv.py:134:29 ] ]
	max.f32 	%r41, %r323, %r324;
$L__tmp41:
	.loc	2 191 40                        // standard.py:191:40 @[ sk03_fa_qkv.py:134:29 ]
	// begin inline asm
	@%p5 st.shared.b32 [ %r37 + 0 ], %r41;
	// end inline asm
	bar.sync 	0;
	ld.shared.b32 	%r325, [global_smem];
$L__tmp42:
	.loc	1 134 49                        // sk03_fa_qkv.py:134:49
	max.f32 	%r326, %r325, 0f0DA24260;
	mov.b32 	%r327, 0f42FE0000;
	.loc	1 135 18                        // sk03_fa_qkv.py:135:18
	div.full.f32 	%r328, %r327, %r326;
	.loc	1 136 13                        // sk03_fa_qkv.py:136:13
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
	.loc	1 137 29                        // sk03_fa_qkv.py:137:29
	.loc	1 137 39                        // sk03_fa_qkv.py:137:39
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
	.loc	1 137 14                        // sk03_fa_qkv.py:137:14
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
	.loc	1 137 49                        // sk03_fa_qkv.py:137:49
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
	.loc	1 138 33                        // sk03_fa_qkv.py:138:33
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
	.loc	1 138 40                        // sk03_fa_qkv.py:138:40
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
	.loc	1 139 50                        // sk03_fa_qkv.py:139:50
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
	.loc	1 139 45                        // sk03_fa_qkv.py:139:45
	// begin inline asm
	@%p1 st.global.v4.b32 [ %rd9 + 0 ], { %r42, %r43, %r44, %r45 };
	// end inline asm
	// begin inline asm
	@%p2 st.global.v4.b32 [ %rd10 + 0 ], { %r46, %r47, %r48, %r49 };
	// end inline asm
	.loc	1 140 21                        // sk03_fa_qkv.py:140:21
	mad.wide.u32 	%rd11, %r51, 4, %rd15;
	.loc	1 140 34                        // sk03_fa_qkv.py:140:34
	mul.f32 	%r50, %r326, 0f3C010204;
	.loc	1 140 26                        // sk03_fa_qkv.py:140:26
	or.b32 	%r537, %r54, %r56;
	setp.eq.b32 	%p6, %r537, 0;
	// begin inline asm
	@%p6 st.global.b32 [ %rd11 + 0 ], { %r50 };
	// end inline asm
	.loc	1 140 4                         // sk03_fa_qkv.py:140:4
	ret;
$L__tmp43:
$L__func_end0:
                                        // -- End function
}
	.file	1 "/repo/vllm/_genesis/kernels/sk03_fa_qkv.py"
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
.b32 215                                // Length of Unit
.b8 2                                   // DWARF version number
.b8 0
.b32 .debug_abbrev                      // Offset Into Abbrev. Section
.b8 8                                   // Address Size (in bytes)
.b8 1                                   // Abbrev [1] 0xb:0xd0 DW_TAG_compile_unit
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
.b8 51
.b8 95
.b8 102
.b8 97
.b8 95
.b8 113
.b8 107
.b8 118
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
.b8 2                                   // Abbrev [2] 0x44:0x1d DW_TAG_subprogram
.b8 95                                  // DW_AT_name
.b8 115
.b8 107
.b8 48
.b8 51
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
.b8 3                                   // Abbrev [3] 0x61:0x79 DW_TAG_subprogram
.b64 $L__func_begin0                    // DW_AT_low_pc
.b64 $L__func_end0                      // DW_AT_high_pc
.b32 68                                 // DW_AT_abstract_origin
.b8 4                                   // Abbrev [4] 0x76:0x32 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp19                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 133                                 // DW_AT_call_line
.b8 28                                  // DW_AT_call_column
.b8 5                                   // Abbrev [5] 0x8e:0x19 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp1                           // DW_AT_low_pc
.b64 $L__tmp18                          // DW_AT_high_pc
.b8 2                                   // DW_AT_call_file
.b8 37                                  // DW_AT_call_line
.b8 1
.b8 36                                  // DW_AT_call_column
.b8 0                                   // End Of Children Mark
.b8 4                                   // Abbrev [4] 0xa8:0x31 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
.b64 $L__tmp20                          // DW_AT_low_pc
.b64 $L__tmp42                          // DW_AT_high_pc
.b8 1                                   // DW_AT_call_file
.b8 134                                 // DW_AT_call_line
.b8 29                                  // DW_AT_call_column
.b8 6                                   // Abbrev [6] 0xc0:0x18 DW_TAG_inlined_subroutine
.b32 68                                 // DW_AT_abstract_origin
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
    "sk03_fa_qkv/_sk03_rmsnorm_quant_kernel",
    _QPTX0, "_sk03_rmsnorm_quant_kernel",
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
        op_name="genesis_sk03_fa_qkv_q0",
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
        return torch.ops.vllm.genesis_sk03_fa_qkv_q0(g, *args)
    ce = ['BLOCK', 'EPS']
    nom = ['x_ptr', 'w_ptr', 'q_ptr', 's_ptr', 'K', 'stride_xm', 'stride_qm', 'BLOCK', 'EPS']
    return _sk03_rmsnorm_quant_kernel[grid](*args[:len(nom) - len(ce)],
                    **{n: args[nom.index(n)] for n in ce},
                    num_warps=8, num_stages=1)
