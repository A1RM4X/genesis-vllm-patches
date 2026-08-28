# SPDX-License-Identifier: Apache-2.0
"""SK-02 — GDN_OUT_INT8_SCALED — capa completa GDN out_proj (RowParallel) en GPU.

Capa completa (Qwen3.5-27B, 48 capas GDN, TP=2):
    hidden bf16 [M, 3072]            (salida del core GDN, sin norm delante)
      -> quant per-token amax/127                              (kernel 1)
      -> GEMM INT8 diádico + residual fusionado en el epílogo  (kernel 2)
      -> all_reduce NCCL                                       (colectivo GPU)

Geometría: global [5120, 6144] = [40*128, 48*128]; per-rank TP=2 K=3072, N=5120.

Diseño (sm_86, mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32):
  * ``BLOCK_K == SHIFT_BLOCK == 128`` -> un ``tl.dot`` por bloque diádico, el
    acumulador INT32 no se vacía cuatro veces por bloque.
  * Shift por columna ``[BLOCK_N]`` aplicado como ``acc += int_acc.to(f32) *
    exp2(shift)``: una operación fp32, exacta, sin truncar shift negativo ni
    desbordar INT32.
  * Escalas fuera del bucle k. Punteros que avanzan. Grid 1-D con swizzle L2.
  * **Residual fusionado**: se suma dentro del epílogo, sin pasada extra sobre
    M*N. Cuando no hay residual el llamador pasa un escalar cero con strides 0,
    así el ``tl.load`` es un broadcast (no cuesta ancho de banda) y no hace
    falta ninguna rama.
  * Salida bf16 directa desde el ``tl.store`` — la versión anterior escribía
    fp32 y hacía ``.to(bf16)`` después, una pasada extra sobre M*N.

RowParallel: el residual se suma **después** del all_reduce (si no, se sumaría
TP veces). ``sk02_gdn_out_proj`` respeta ese orden; con TP=1 el residual sí va
fusionado en el epílogo.

Los kernels son branchless (sólo ``tl.load`` / ``tl.dot`` / ``tl.where`` /
``tl.store``) y no validan nada: dtypes, layout y múltiplos de 128 se asumen
ya comprobados. Cero fallback torch, cero trabajo en CPU. La única rama en
Python decide el orden residual/all_reduce, que es topología, no validación.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

SK_ID = "SK-02"
SK_NAME = "GDN_OUT_INT8_SCALED"
GLOBAL_SHAPE = (5120, 6144)
RANK_SHAPE = (5120, 3072)
NUM_LAYERS = 48
ROW_PARALLEL = True

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
def _sk02_quant_kernel(x_ptr, q_ptr, s_ptr, K, stride_xm, stride_qm, BLOCK: tl.constexpr):
    """Quant per-token amax/127. Sin norm: out_proj no tiene norma delante."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < K
    x = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-30)
    xq = x * (127.0 / amax)
    q = (xq + tl.where(xq >= 0.0, 0.5, -0.5)).to(tl.int32)
    tl.store(q_ptr + row * stride_qm + offs, tl.minimum(tl.maximum(q, -127), 127).to(tl.int8), mask=mask)
    tl.store(s_ptr + row, amax * (1.0 / 127.0))



@triton.jit
def _sk02_gdn_out_int8_kernel(
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


def sk02_quant(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quant per-token INT8 -> ``(q [M,K] int8, s [M] fp32)``."""
    M, K = hidden.shape
    q = torch.empty((M, K), dtype=torch.int8, device=hidden.device)
    s = torch.empty((M,), dtype=torch.float32, device=hidden.device)
    _sk02_quant_kernel[(M,)](
        hidden, q, s, K, hidden.stride(0), q.stride(0),
        BLOCK=triton.next_power_of_2(K), num_warps=8, num_stages=1,
    )
    return q, s


def sk02_gemm_int8_scaled(
    hidden: torch.Tensor,
    weight_i8: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    shifts: torch.Tensor,
    residual: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """GEMM INT8 diádico + residual fusionado. ``hidden`` int8 [M,K], ``weight_i8`` [K,N]."""
    M, K = hidden.shape
    N = weight_i8.shape[1]
    res = _zero(hidden.device) if residual is None else residual
    rm, rn = (0, 0) if residual is None else (res.stride(0), res.stride(1))
    has_shift = _has_shift(shifts)
    bm, bn, bk, gm, warps, stages = _cfg(M, N)
    out = torch.empty((M, N), dtype=out_dtype, device=hidden.device)
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    _sk02_gdn_out_int8_kernel[grid](
        hidden, weight_i8, out, res, a_scales, b_scales, shifts,
        M, N, K,
        hidden.stride(0), hidden.stride(1), weight_i8.stride(0), weight_i8.stride(1),
        out.stride(0), out.stride(1), rm, rn,
        shifts.stride(0), shifts.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=gm, SHIFT_BLOCK=SHIFT_BLOCK,
        HAS_SHIFT=has_shift,
        num_warps=warps, num_stages=stages,
    )
    return out


def sk02_gdn_out_proj(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    shifts: torch.Tensor,
    residual: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    do_allreduce: bool = True,
) -> torch.Tensor:
    """Capa completa GDN out_proj: quant -> GEMM diádico -> all_reduce -> residual.

    RowParallel: el residual se fusiona en el epílogo del GEMM sólo cuando NO
    hay all_reduce; con TP>1 se suma después del colectivo para no replicarlo.
    """
    a, a_scales = sk02_quant(hidden)
    if not do_allreduce:
        # TP=1: el residual va fusionado en el epílogo del GEMM, sin pasada extra.
        return sk02_gemm_int8_scaled(a, weight, a_scales, weight_scale, shifts, residual, out_dtype)
    out = sk02_gemm_int8_scaled(a, weight, a_scales, weight_scale, shifts, None, out_dtype)
    torch.distributed.all_reduce(out)
    out.add_(residual)
    return out


sk02_gdn_out = sk02_gdn_out_proj
gdn_out_int8_scaled = sk02_gemm_int8_scaled
sk02_gemm_w8a16 = sk02_gemm_int8_scaled

__all__ = [
    "SK_ID", "SK_NAME", "GLOBAL_SHAPE", "RANK_SHAPE", "NUM_LAYERS", "ROW_PARALLEL",
    "SHIFT_BLOCK", "BLOCK_M", "BLOCK_N", "BLOCK_K",
    "sk02_quant", "sk02_gemm_int8_scaled", "sk02_gdn_out_proj", "sk02_gdn_out",
    "gdn_out_int8_scaled", "sk02_gemm_w8a16",
]
