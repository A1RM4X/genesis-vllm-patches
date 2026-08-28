# SPDX-License-Identifier: Apache-2.0
"""SK-11 — VISION_BF16 — GEMM bf16 del ViT con epílogo fusionado, en GPU.

El ViT de Qwen3.5 está en ``modules_to_not_convert``: sus pesos quedan bf16 y
no hay Tensor Core INT8 que aprovechar. En bf16 puro, cuBLAS ya corre a ~90%
del pico de la 3090 (medido: 64 TFLOPS de 71 nominales), así que **un GEMM
Triton no le va a ganar** — la versión anterior de este archivo era un GEMM
naive 32x64x32 que corría 1.0-1.4x *más lento* que ``torch.matmul``.

Lo que sí se puede ganar es el epílogo. cuBLAS no fusiona bias+GELU, así que
cada proyección del ViT paga una pasada extra de lectura+escritura sobre M*N.
SK-11 ofrece las dos cosas y es explícito sobre cuál usar:

  * ``sk11_vision_gemm``      -> delega en cuBLAS. Es el óptimo para el GEMM
                                 desnudo y no hay motivo para reimplementarlo.
  * ``sk11_linear_gelu``      -> GEMM Triton con **bias + GELU tanh fusionados**
                                 en el epílogo: ahorra la pasada extra.
  * ``sk11_bias_gelu``        -> sólo el epílogo, para encadenar tras cuBLAS.

Diseño del GEMM bf16 (sm_86, mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32):
  * Acumulador fp32, tiles grandes, punteros que avanzan, grid 1-D con
    swizzle ``GROUP_M`` para reuso de B en L2.
  * shared = (BLOCK_M + BLOCK_N) * BLOCK_K * 2 bytes * num_stages.

Los kernels son branchless y no validan nada: se asume todo comprobado.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

SK_ID = "SK-11"
SK_NAME = "VISION_BF16"
VISION_HIDDEN: int = 1152

_CFG: tuple[tuple[int, int, int, int, int, int], ...] = (
    (16, 128, 64, 8, 4, 4),
    (64, 128, 64, 8, 8, 4),
    (128, 128, 64, 8, 8, 4),
    (128, 256, 64, 8, 8, 3),
)

BLOCK_M: int = 128
BLOCK_N: int = 128
BLOCK_K: int = 64

_ZERO: dict[torch.device, torch.Tensor] = {}


def _zero(device: torch.device) -> torch.Tensor:
    """Escalar cero por device, usado como bias neutro con stride 0."""
    z = _ZERO.get(device)
    if z is None:
        z = torch.zeros((), dtype=torch.bfloat16, device=device)
        _ZERO[device] = z
    return z


def _cfg(m: int) -> tuple[int, int, int, int, int, int]:
    return _CFG[(m > 32) + (m > 128) + (m > 512)]


@triton.jit
def _sk11_linear_gelu_kernel(
    a_ptr, b_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_out_m, stride_out_n, stride_bias,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    """GEMM bf16 con bias + GELU tanh fusionados en el epílogo."""
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        acc += tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    x = acc + tl.load(bias_ptr + offs_n * stride_bias).to(tl.float32)[None, :]
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    y = 0.5 * x * (1.0 + (2.0 / (1.0 + tl.exp(-2.0 * inner)) - 1.0))

    out_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(
        out_ptr + out_m[:, None] * stride_out_m + out_n[None, :] * stride_out_n,
        y.to(out_ptr.dtype.element_ty),
        mask=(out_m[:, None] < M) & (out_n[None, :] < N),
    )


@triton.jit
def _sk11_bias_gelu_kernel(x_ptr, bias_ptr, out_ptr, N, stride_xm, stride_om, BLOCK: tl.constexpr):
    """bias + GELU tanh, una fila por programa."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    y = 0.5 * x * (1.0 + (2.0 / (1.0 + tl.exp(-2.0 * inner)) - 1.0))
    tl.store(out_ptr + row * stride_om + offs, y.to(tl.bfloat16), mask=mask)


def sk11_vision_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """GEMM bf16 desnudo. Delega en cuBLAS: es el óptimo en sm_86 para bf16."""
    return torch.matmul(a, b)


def sk11_linear_gelu(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """GEMM bf16 con bias + GELU tanh fusionados -> ``[M, N]`` bf16."""
    M, K = a.shape
    N = b.shape[1]
    bi = _zero(a.device) if bias is None else bias
    sb = 0 if bias is None else bi.stride(0)
    bm, bn, bk, gm, warps, stages = _cfg(M)
    out = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    _sk11_linear_gelu_kernel[grid](
        a, b, bi, out, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), out.stride(0), out.stride(1), sb,
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=gm, num_warps=warps, num_stages=stages,
    )
    return out


def sk11_bias_gelu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """bias + GELU tanh sobre ``[M, N]`` bf16, para encadenar tras cuBLAS."""
    M, N = x.shape
    out = torch.empty_like(x)
    _sk11_bias_gelu_kernel[(M,)](
        x, bias, out, N, x.stride(0), out.stride(0),
        BLOCK=triton.next_power_of_2(N), num_warps=8, num_stages=1,
    )
    return out


passthrough_bf16 = sk11_vision_gemm
sk11_passthrough = sk11_vision_gemm
sk11_forward = sk11_vision_gemm

__all__ = [
    "SK_ID", "SK_NAME", "VISION_HIDDEN", "BLOCK_M", "BLOCK_N", "BLOCK_K",
    "sk11_vision_gemm", "sk11_linear_gelu", "sk11_bias_gelu",
    "passthrough_bf16", "sk11_passthrough", "sk11_forward",
]
