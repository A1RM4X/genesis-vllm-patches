# SPDX-License-Identifier: Apache-2.0
"""SK-09 — NORM_EMBED — RMSNorm + quant fusionado y embed passthrough, en GPU.

RMSNorm + quant per-token INT8 en **una sola pasada**: la fila entra una vez a
registros y se reusa para la varianza, el ``amax`` y la cuantización. Es el
productor natural de los GEMM INT8 (SK-01/03/05/07/10): entrega ``(q, s)`` sin
que ninguno de ellos tenga que recalcular estadísticas de fila.

    y = x * rsqrt(mean(x^2) + eps) * ((w + GAMMA_OFFSET) * s_pow2)
    s = amax(|y|) / 127
    q = clamp(round(y / s), -127, 127)

Corrección respecto de la versión anterior
------------------------------------------
La versión anterior fijaba ``IS_GEMMA=1`` como ``constexpr`` en el wrapper, sin
ningún parámetro para desactivarlo, de modo que el kernel siempre calculaba
``w_eff = 1 + w`` (semántica Gemma). Qwen3.5 usa RMSNorm plana (``w``, sin el
``+1``): medido contra una RMSNorm estándar, la diferencia llegaba a **101
niveles INT8 de 127**. Ahora la semántica es el parámetro ``gamma_offset``
(0.0 plana, 1.0 Gemma) y por defecto es **0.0**.

También se fue todo el andamiaje que no aportaba: el memo de lanzamiento con
seguimiento de alineación de ``data_ptr``, el buffer único con la escala
empotrada a 16 B, el camino ``_wide`` con ``tl.dot(..., input_precision="ieee")``
para reproducir bit a bit el orden de reducción de ATen (una reducción de suma
de cuadrados no necesita Tensor Cores), y el fallback torch. El kernel resultante
es una sola pasada por fila.

``s_pow2`` es el vector SmoothQuant restringido a potencias de dos absorbido en
la norma. Cuando no hay, el llamador no pasa nada y el puntero apunta a un
escalar 1.0 con stride 0: broadcast gratis, sin rama.

Los kernels son branchless y no validan nada: se asume todo comprobado.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

SK_ID = "SK-09"
SK_NAME = "NORM_EMBED"
BLOCK_MAX: int = 16384

_ONE: dict[torch.device, torch.Tensor] = {}


def _one(device: torch.device) -> torch.Tensor:
    """Escalar 1.0 por device, usado como ``s_pow2`` neutro con stride 0."""
    o = _ONE.get(device)
    if o is None:
        o = torch.ones((), dtype=torch.float32, device=device)
        _ONE[device] = o
    return o


@triton.jit
def _sk09_rmsnorm_quant_kernel(
    x_ptr, w_ptr, s_pow2_ptr, q_ptr, s_ptr,
    K, stride_xm, stride_qm, stride_sp,
    BLOCK: tl.constexpr, EPS: tl.constexpr, GAMMA_OFFSET: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < K
    x = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32) + GAMMA_OFFSET
    w = w * tl.load(s_pow2_ptr + offs * stride_sp, mask=mask, other=1.0).to(tl.float32)
    y = x * tl.rsqrt(tl.sum(x * x, axis=0) / K + EPS) * w
    amax = tl.maximum(tl.max(tl.abs(y), axis=0), 1e-30)
    yq = y * (127.0 / amax)
    q = (yq + tl.where(yq >= 0.0, 0.5, -0.5)).to(tl.int32)
    tl.store(q_ptr + row * stride_qm + offs, tl.minimum(tl.maximum(q, -127), 127).to(tl.int8), mask=mask)
    tl.store(s_ptr + row, amax * (1.0 / 127.0))


@triton.jit
def _sk09_quant_kernel(x_ptr, q_ptr, s_ptr, K, stride_xm, stride_qm, BLOCK: tl.constexpr):
    """Quant per-token amax/127 sin norma delante, una fila por programa."""
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
def _sk09_rmsnorm_kernel(
    x_ptr, w_ptr, out_ptr, K, stride_xm, stride_om,
    BLOCK: tl.constexpr, EPS: tl.constexpr, GAMMA_OFFSET: tl.constexpr,
):
    """RMSNorm sola, salida bf16, para las normas que no alimentan un GEMM INT8."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < K
    x = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32) + GAMMA_OFFSET
    y = x * tl.rsqrt(tl.sum(x * x, axis=0) / K + EPS) * w
    tl.store(out_ptr + row * stride_om + offs, y.to(tl.bfloat16), mask=mask)


def rmsnorm_quant_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    s_pow2: torch.Tensor | None = None,
    eps: float = 1e-6,
    gamma_offset: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm + quant per-token -> ``(q [M,K] int8, s [M] fp32)``.

    ``gamma_offset``: 0.0 RMSNorm plana (Qwen3.5), 1.0 semántica Gemma.
    """
    k = x.shape[-1]
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    s = torch.empty((m,), dtype=torch.float32, device=x.device)
    sp = _one(x.device) if s_pow2 is None else s_pow2
    _sk09_rmsnorm_quant_kernel[(m,)](
        x2, weight, sp, q, s,
        k, x2.stride(0), q.stride(0), 0 if s_pow2 is None else sp.stride(0),
        BLOCK=triton.next_power_of_2(k), EPS=eps, GAMMA_OFFSET=gamma_offset,
        num_warps=8, num_stages=1,
    )
    return q, s


def quant_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quant per-token INT8 -> ``(q [..., K] int8, s [..., 1] fp32)``.

    Sin tope de K: ``BLOCK = next_power_of_2(K)``. ``fused_quant_triton`` tenía
    ``BLOCK_MAX = 8192`` y un ``assert n <= BLOCK_MAX``, pero ``down_proj`` de
    Qwen3.5-27B tiene **K = 8704** por rank (TP=2), así que ese assert hacía
    reventar el ``apply`` de PN110 en las 65 capas ``down_proj`` — y sin
    fallback, porque el peso Marlin ya había sido liberado.
    """
    k = x.shape[-1]
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    s = torch.empty((m,), dtype=torch.float32, device=x.device)
    _sk09_quant_kernel[(m,)](
        x2, q, s, k, x2.stride(0), q.stride(0),
        BLOCK=triton.next_power_of_2(k), num_warps=8, num_stages=1,
    )
    return q.view(x.shape), s.view(*x.shape[:-1], 1)


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    gamma_offset: float = 0.0,
) -> torch.Tensor:
    """RMSNorm sola -> bf16, misma forma que ``x``."""
    k = x.shape[-1]
    x2 = x.reshape(-1, k)
    out = torch.empty_like(x2, dtype=torch.bfloat16)
    _sk09_rmsnorm_kernel[(x2.shape[0],)](
        x2, weight, out, k, x2.stride(0), out.stride(0),
        BLOCK=triton.next_power_of_2(k), EPS=eps, GAMMA_OFFSET=gamma_offset,
        num_warps=8, num_stages=1,
    )
    return out.view(x.shape)


def embed_tokens_bf16_passthrough(embed_weight: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Embed lookup. Es un gather puro sobre bf16: ``index_select`` ya es óptimo."""
    return embed_weight.index_select(0, input_ids.reshape(-1)).view(*input_ids.shape, embed_weight.shape[1])


fused_rmsnorm_quant = rmsnorm_quant_fused
sk09_rmsnorm_quant = rmsnorm_quant_fused

__all__ = [
    "SK_ID", "SK_NAME", "BLOCK_MAX",
    "rmsnorm_quant_fused", "fused_rmsnorm_quant", "sk09_rmsnorm_quant",
    "rmsnorm", "quant_per_token", "embed_tokens_bf16_passthrough",
]
