# SPDX-License-Identifier: Apache-2.0
"""int8 hybrid GEMM per-bloque 128 — el camino NO-SK de PN110.

Contrato de ``shifts`` [K/128, N/128]. Hay dos diseños vivos que producen ese
tensor y **se distinguen por dtype**:

* **entero** (Diseño C, shift diádico): el factor del bloque es ``2**shift``.
* **flotante** (cuantización exacta por bloques 128x128): el tensor YA es el
  factor del bloque, y ``b_scales`` viene en unos.

Antes este kernel asumía siempre el diádico y hacía ``shifts.to(tl.int32)``.
Con el tensor fp32 del segundo diseño (valores ~1e-3) eso trunca a 0: el
factor del bloque desaparece y la salida sale ~1500x de escala. Es basura
directa, y como este es el camino de TODA capa sin super kernel ligado (MTP
incluido), envenena el modelo entero.

Unificación: ambos diseños se reducen a un multiplicador fp32 por bloque
resuelto ANTES del kernel. Es bit-exacto contra el corrimiento entero — el
acumulador de un bloque son 128 productos de int8, ``|acc| <= 128*127*127 =
2064512 < 2**24``, así que cabe entero en la mantisa de fp32 y multiplicar por
una potencia de 2 no pierde nada.

Estructura del bucle espejada de SK-06, que ya está medido: ``BLOCK_K ==
SHIFT_BLOCK == 128`` -> un ``tl.dot`` por bloque, un vaciado del acumulador por
bloque en vez de cuatro, y el factor del bloque cargado como escalar (uniforme
en el CTA porque ``BLOCK_N <= SHIFT_BLOCK``) en vez de una carga por lane.
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl

BLOCK_M = 32
BLOCK_N = 128
BLOCK_K = 128
SHIFT_BLOCK = 128


def is_triton_available() -> bool:
    return True


def is_available() -> bool:
    return torch.cuda.is_available()


@triton.jit
def _int8_hybrid_gemm_kernel(
    a_ptr, b_ptr, out_ptr, a_scale_ptr, b_scale_ptr, blk_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_out_m, stride_out_n,
    stride_blk_k, stride_blk_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SHIFT_BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    # BLOCK_N <= SHIFT_BLOCK: el tile entero cae en un solo bloque de columnas,
    # asi que el factor es un ESCALAR difundido, no una carga por lane.
    blk_col = (pid_n * BLOCK_N) // SHIFT_BLOCK

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, K // BLOCK_K):
        a_tile = tl.load(a_ptrs, mask=mask_m[:, None], other=0)
        b_tile = tl.load(b_ptrs, mask=mask_n[None, :], other=0)
        d = tl.dot(a_tile, b_tile, out_dtype=tl.float32)
        f = tl.load(blk_ptr + kb * stride_blk_k + blk_col * stride_blk_n).to(tl.float32)
        acc += d * f
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    acc = acc * tl.load(a_scale_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)[:, None]
    acc = acc * tl.load(b_scale_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty),
             mask=mask_m[:, None] & mask_n[None, :])


def _factores_por_bloque(shifts: torch.Tensor) -> torch.Tensor:
    """``shifts`` -> multiplicador fp32 por bloque, según el dtype.

    Entero: shift diádico, el factor es ``2**shift``. Flotante: el tensor ya
    ES el factor. Cacheado como atributo del tensor porque el peso no cambia
    entre forwards y ``2**shift`` sobre [K/128,N/128] es un kernel entero que
    no tiene por qué repetirse.
    """
    if shifts.is_floating_point():
        return shifts if shifts.dtype == torch.float32 else shifts.float()
    f = getattr(shifts, "_factores_fp32", None)
    if f is None:
        f = torch.exp2(shifts.to(torch.float32))
        try:
            shifts._factores_fp32 = f
        except Exception:
            pass
    return f


def int8_hybrid_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    shifts: torch.Tensor,
    out_dtype: torch.dtype | None = None,
    block_k: int = 128,
) -> torch.Tensor:
    if out_dtype is None:
        out_dtype = torch.float16
    assert a.dtype == torch.int8
    assert b.dtype == torch.int8
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    assert K % block_k == 0
    assert N % block_k == 0
    assert shifts.shape == (K // block_k, N // block_k)

    if a_scales.dim() == 2:
        a_vec = (a_scales.squeeze(1) if a_scales.shape[1] == 1
                 else a_scales.reshape(-1)[:M]).contiguous().to(torch.float32)
    else:
        a_vec = a_scales.contiguous().to(torch.float32)
        if a_vec.numel() > M:
            a_vec = a_vec[:M]
    if b_scales.dim() == 2:
        b_vec = (b_scales.squeeze(1) if b_scales.shape[0] == N
                 else b_scales.reshape(-1)[:N]).contiguous().to(torch.float32)
    else:
        b_vec = b_scales.contiguous().to(torch.float32)
        if b_vec.numel() > N:
            b_vec = b_vec[:N]

    blk = _factores_por_bloque(shifts)
    out = torch.empty((M, N), dtype=out_dtype, device=a.device)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _int8_hybrid_gemm_kernel[grid](
        a, b, out, a_vec, b_vec, blk, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        out.stride(0), out.stride(1), blk.stride(0), blk.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        SHIFT_BLOCK=SHIFT_BLOCK, num_warps=4, num_stages=3,
    )
    return out


hybrid_gemm = int8_hybrid_gemm
__all__ = ["int8_hybrid_gemm", "hybrid_gemm", "is_available", "is_triton_available"]
