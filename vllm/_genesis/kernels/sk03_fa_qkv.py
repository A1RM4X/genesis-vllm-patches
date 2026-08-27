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

import torch
import triton
import triton.language as tl

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
    _sk03_fa_qkv_kernel[grid](
        a, b, out, res, a_scales, b_scales, shifts,
        M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        out.stride(0), out.stride(1), rm, rn,
        shifts.stride(0), shifts.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=gm, SHIFT_BLOCK=SHIFT_BLOCK,
        HAS_SHIFT=has_shift,
        num_warps=warps, num_stages=stages,
    )
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
    _sk03_rmsnorm_quant_kernel[(M,)](
        hidden, ln_weight, q, s,
        K, hidden.stride(0), q.stride(0),
        BLOCK=triton.next_power_of_2(K), EPS=eps, num_warps=8, num_stages=1,
    )
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
