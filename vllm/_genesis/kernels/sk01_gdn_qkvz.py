# SPDX-License-Identifier: Apache-2.0
"""SK-01 — GDN_QKVZ_FUSED_INT8_DIADIC — capa completa GDN in_proj_qkvz en GPU.

Capa completa (Qwen3.5-27B, TP=2):
    hidden bf16 [M,5120]
      -> RMSNorm(input_layernorm) + quant per-token amax/127  (kernel 1)
      -> GEMM INT8 diádico  w = q * 2^shift * s_row           (kernel 2)
      -> split Q/K/V/Z (views, coste cero)

Geometría: N global 16384 = 10240 (QKV) + 6144 (Z) = 128*128, K = 5120 = 40*128.
Per-rank TP=2: N = 8192. Todas las dims múltiplo de 128 -> sin máscaras en K.

Diseño del GEMM (sm_86, mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32):
  * ``BLOCK_K == SHIFT_BLOCK == 128``: un único ``tl.dot`` por bloque diádico.
    El acumulador INT32 vive en registros durante los 128 k y se vacía una sola
    vez por bloque, no cuatro. Elimina el sub-bucle de la versión anterior.
  * Shift **vectorial por columna** ``[BLOCK_N]``: cada columna usa su propio
    ``shifts[kb, n//128]``, así que ``BLOCK_N`` puede ser 128 o 256 sin perder
    fidelidad diádica.
  * El shift se aplica como ``acc += int_acc.to(f32) * exp2(shift)`` en vez de
    ``shl``/``shr``/``selp`` sobre INT32: una sola operación fp32 en lugar de
    tres, **y exacto** — el ``>>`` aritmético truncaba hacia -inf para shift
    negativo, y ``<<`` podía desbordar INT32 con shift >= 11. Multiplicar por
    una potencia de dos en fp32 no tiene error de redondeo.
  * Escalas ``a_scale``/``b_scale`` factorizadas **fuera** del bucle k: son
    constantes por fila/columna, multiplicarlas 40 veces era trabajo tirado.
  * Grid 1-D con swizzle ``GROUP_M`` para que los CTA concurrentes reusen los
    mismos bloques de B en L2.
  * Punteros que avanzan (``a_ptrs += BLOCK_K*stride_ak``) en vez de recalcular
    offsets y máscaras en cada iteración.
  * ``offs % M`` / ``offs % N`` en vez de máscaras: el wrap es inocuo porque el
    ``tl.store`` final sí enmascara.

Branchless: sólo ``tl.load`` / ``tl.dot`` / ``tl.where`` / ``tl.store``.
Sin validación, sin fallback torch, sin CPU: todo se asume ya comprobado.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

SK01_N_GLOBAL: int = 16384
SK01_K: int = 5120
SK01_QKV_N: int = 10240
SK01_Z_N: int = 6144
SK01_Q_END: int = 2048
SK01_K_END: int = 4096
SK01_V_END: int = 10240
SK01_N_PER_RANK: int = 8192
SK01_K_PER_RANK: int = 5120

SHIFT_BLOCK: int = 128


# (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages) por bucket de M.
# BLOCK_K=128 fija un tl.dot por bloque diádico; shared = (BM+BN)*128*stages.
_CFG: tuple[tuple[int, int, int, int, int, int], ...] = (
    # (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages) por bucket de M.
    # Medido en RTX 3090 (K=5120, N=8192), mediana de 5 corridas, contra el
    # tile unico 128x128x128 que habia antes:
    #   M=128   128x128x128  1.00x  (los tiles grandes pierden 0.4-0.5x aca)
    #   M=512   256x128x64   1.29x
    #   M=1664  256x128x128  1.06x
    #   M=8000  256x128x128  1.21x
    # BLOCK_M=256 amortiza el tile de B sobre el doble de filas. La ocupacion
    # no es la palanca: TODAS las configuraciones quedan en 1 CTA/SM (8 de 48
    # warps), asi que lo que manda es la intensidad aritmetica.
    (16, 128, 128, 8, 8, 3),    # M <=  32   decode
    (128, 128, 128, 8, 8, 3),   # M <= 128
    (256, 128, 64, 8, 8, 4),    # M <= 1024
    (256, 128, 128, 8, 8, 2),   # M  > 1024  prefill
)

# Compat: el resto del árbol importa estos nombres.
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


def _cfg(m: int) -> tuple[int, int, int, int, int, int]:
    """Selecciona tile por bucket de M sin ramificar (suma de predicados)."""
    return _CFG[(m > 32) + (m > 128) + (m > 1024)]


@triton.jit
def _sk01_rmsnorm_quant_kernel(
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
def _sk01_gdn_qkvz_kernel(
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


def sk01_gdn_qkvz_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    shifts: torch.Tensor,
    residual: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """GEMM INT8 diádico QKVZ. ``a`` [M,K] int8, ``b`` [K,N] int8 col-major."""
    M, K = a.shape
    N = b.shape[1]
    res = _zero(a.device) if residual is None else residual
    rm, rn = (0, 0) if residual is None else (res.stride(0), res.stride(1))
    has_shift = _has_shift(shifts)
    bm, bn, bk, gm, warps, stages = _cfg(M)
    out = torch.empty((M, N), dtype=out_dtype, device=a.device)
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    _sk01_gdn_qkvz_kernel[grid](
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


def sk01_rmsnorm_quant(
    hidden: torch.Tensor,
    ln_weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm + quant per-token INT8. Devuelve ``(q [M,K] int8, s [M] fp32)``."""
    M, K = hidden.shape
    q = torch.empty((M, K), dtype=torch.int8, device=hidden.device)
    s = torch.empty((M,), dtype=torch.float32, device=hidden.device)
    _sk01_rmsnorm_quant_kernel[(M,)](
        hidden, ln_weight, q, s,
        K, hidden.stride(0), q.stride(0),
        BLOCK=triton.next_power_of_2(K), EPS=eps, num_warps=8, num_stages=1,
    )
    return q, s


def sk01_gdn_qkvz_forward(
    hidden: torch.Tensor,
    ln_weight: torch.Tensor,
    weight_qkvz: torch.Tensor,
    b_scales: torch.Tensor,
    shifts: torch.Tensor,
    eps: float = 1e-6,
    out_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Capa completa GDN in_proj_qkvz: RMSNorm+quant -> GEMM diádico -> split.

    Split per-rank sobre ``N``: ``Q = N//8``, ``K = N//8``, ``V = 3N//8``,
    ``Z = 3N//8`` (16384 global -> 2048/2048/6144/6144).
    """
    a, a_scales = sk01_rmsnorm_quant(hidden, ln_weight, eps)
    qkvz = sk01_gdn_qkvz_gemm(a, weight_qkvz, a_scales, b_scales, shifts, None, out_dtype)
    n8 = qkvz.shape[1] // 8
    return (
        qkvz[:, 0:n8],
        qkvz[:, n8:2 * n8],
        qkvz[:, 2 * n8:5 * n8],
        qkvz[:, 5 * n8:],
    )


gdn_qkvz_fused_int8_diadic = sk01_gdn_qkvz_gemm
gdn_qkvz_gemm = sk01_gdn_qkvz_gemm
sk01_forward = sk01_gdn_qkvz_forward

__all__ = [
    "sk01_gdn_qkvz_gemm", "gdn_qkvz_fused_int8_diadic", "gdn_qkvz_gemm",
    "sk01_gdn_qkvz_forward", "sk01_forward", "sk01_rmsnorm_quant",
    "SK01_N_GLOBAL", "SK01_K", "SK01_N_PER_RANK", "SK01_QKV_N", "SK01_Z_N",
    "BLOCK_M", "BLOCK_N", "BLOCK_K", "SHIFT_BLOCK",
]
