# SPDX-License-Identifier: Apache-2.0
"""SK-08 — SSM_CONTROL — capa completa del control GDN (decode) en GPU.

Capa completa (Qwen3.5-27B, 48 capas GDN):
    mixed_qkv bf16 [B, 10240]      (salida de in_proj_qkv, ya con conv pendiente)
      -> conv1d causal depthwise width=4 + SiLU + shift de estado   (kernel 1)
      -> gated delta rule recurrente + escritura de ssm_state       (kernel 2)
      -> o [B, HV, V]

Geometría: H=16 cabezas K, HV=48 cabezas V, K=V=128,
``conv_dim = 2*H*K + HV*V = 2048 + 2048 + 6144 = 10240``.
``ssm_state [B, HV, V, K]``, ``conv_state [B, conv_dim, 3]``.

Corrección respecto de la versión anterior
------------------------------------------
La versión anterior **no implementaba la delta rule**. Hacía:

    hk_sum   = sum(b_h, axis=1) * 0.01
    b_v_corr = (v - hk_sum) * beta
    b_h     += b_v_corr[:, None] * 0.5
    o        = sum(b_h, axis=1) + q * 0.1

con constantes ``0.01``/``0.5``/``0.1`` sin origen, **sin el vector k en
ninguna parte** (no había producto externo ``k (x) v`` para actualizar el
estado ni proyección por ``q`` para la salida: usaba ``sum(b_h, axis=1)``, que
es sumar el estado, no proyectarlo), y con ``q = v = conv_acc``, es decir q, k
y v eran el mismo tensor. Producía números, no la recurrencia.

La recurrencia correcta, espejo de
``vllm/model_executor/layers/fla/ops/fused_recurrent.py``:

    b_q, b_k  <- L2-normalizados;  b_q <- b_q * scale
    g         = -exp(A_log) * softplus(a + dt_bias)
    beta      = sigmoid(b)
    S        *= exp(g)                       # decaimiento
    b_v      -= sum(S * k[None, :], axis=1)   # error de predicción: v - S k
    b_v      *= beta
    S        += b_v[:, None] * k[None, :]     # producto externo
    o         = sum(S * q[None, :], axis=1)   # proyección por q

El mapeo de cabezas también estaba mal: ``i_h = i_hv // (HV // H)`` (3 cabezas
V por cabeza K), y los offsets ``q_off = i_h*K``, ``k_off = H*K + i_h*K``,
``v_off = 2*H*K + i_hv*V`` sobre ``mixed_qkv``.

Por qué dos kernels y no uno
----------------------------
Los canales de q y k los comparten 3 cabezas V. Si el kernel de la recurrencia
hiciera también el shift del estado del conv, esas 3 cabezas escribirían el
mismo canal mientras las otras aún lo leen: carrera de lectura-escritura, con
resultado dependiente del orden de los CTA. Los canales de v sí son exclusivos
por cabeza, pero separar el conv completo es lo único correcto sin sincronizar
entre bloques. El conv es bandwidth-bound y trivialmente paralelo.

Los kernels son branchless y no validan nada: se asume todo comprobado.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

SK_ID = "SK-08"
SK_NAME = "SSM_CONTROL"
SK08_CONV_DIM: int = 10240
SK08_CONV_KERNEL_WIDTH: int = 4
SK08_NUM_GDN_LAYERS: int = 48
SK08_NUM_K_HEADS: int = 16
SK08_NUM_V_HEADS: int = 48
SK08_HEAD_DIM: int = 128
SK08_CONV1D_SHAPE: tuple[int, int, int] = (10240, 1, 4)
SK08_A_LOG_SHAPE: tuple[int, ...] = (48,)
SK08_DT_BIAS_SHAPE: tuple[int, ...] = (48,)
SOFTPLUS_THRESHOLD: float = 20.0


@triton.jit
def _sk08_conv1d_silu_kernel(
    x_ptr, w_ptr, bias_ptr, state_ptr, out_ptr,
    D,
    stride_x_b, stride_state_b, stride_state_d, stride_state_w, stride_out_b,
    WIDTH: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """conv1d causal depthwise width=4 + SiLU + shift del estado, un token.

    ``state[b, d, 0..2]`` son los 3 valores pasados; se desplaza a
    ``(s1, s2, x)`` tras calcular. Los pesos se leen como ``w[d*WIDTH + t]``.
    """
    b = tl.program_id(0)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d < D

    x = tl.load(x_ptr + b * stride_x_b + d, mask=mask, other=0.0).to(tl.float32)
    sb = state_ptr + b * stride_state_b + d * stride_state_d
    s0 = tl.load(sb + 0 * stride_state_w, mask=mask, other=0.0).to(tl.float32)
    s1 = tl.load(sb + 1 * stride_state_w, mask=mask, other=0.0).to(tl.float32)
    s2 = tl.load(sb + 2 * stride_state_w, mask=mask, other=0.0).to(tl.float32)

    wb = w_ptr + d * WIDTH
    y = s0 * tl.load(wb + 0, mask=mask, other=0.0).to(tl.float32)
    y += s1 * tl.load(wb + 1, mask=mask, other=0.0).to(tl.float32)
    y += s2 * tl.load(wb + 2, mask=mask, other=0.0).to(tl.float32)
    y += x * tl.load(wb + 3, mask=mask, other=0.0).to(tl.float32)
    y += tl.load(bias_ptr + d, mask=mask, other=0.0).to(tl.float32)
    y = y * tl.sigmoid(y)

    tl.store(sb + 0 * stride_state_w, s1, mask=mask)
    tl.store(sb + 1 * stride_state_w, s2, mask=mask)
    tl.store(sb + 2 * stride_state_w, x, mask=mask)
    tl.store(out_ptr + b * stride_out_b + d, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _sk08_gated_delta_rule_kernel(
    qkv_ptr, a_ptr, b_ptr, A_log_ptr, dt_bias_ptr, state_ptr, o_ptr,
    stride_qkv_b, stride_a_b, stride_state_b, stride_o_b,
    SCALE: tl.constexpr,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
):
    """Gated delta rule recurrente, una cabeza V por programa."""
    pid = tl.program_id(0)
    i_n = pid // HV
    i_hv = pid % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)

    p_state = state_ptr + i_n * stride_state_b + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(p_state).to(tl.float32)

    p_mixed = qkv_ptr + i_n * stride_qkv_b
    b_q = tl.load(p_mixed + i_h * K + o_k).to(tl.float32)
    b_k = tl.load(p_mixed + H * K + i_h * K + o_k).to(tl.float32)
    b_v = tl.load(p_mixed + 2 * H * K + i_hv * V + o_v).to(tl.float32)

    b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6) * SCALE
    b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)

    a_val = tl.load(a_ptr + i_n * stride_a_b + i_hv).to(tl.float32)
    b_val = tl.load(b_ptr + i_n * stride_a_b + i_hv).to(tl.float32)
    x = a_val + tl.load(dt_bias_ptr + i_hv).to(tl.float32)
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(tl.load(A_log_ptr + i_hv).to(tl.float32)) * softplus_x
    beta_val = tl.sigmoid(b_val)

    b_h *= tl.exp(g_val)
    b_v -= tl.sum(b_h * b_k[None, :], axis=1)
    b_v *= beta_val
    b_h += b_v[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], axis=1)

    tl.store(p_state, b_h.to(state_ptr.dtype.element_ty))
    tl.store(o_ptr + i_n * stride_o_b + i_hv * V + o_v, b_o.to(o_ptr.dtype.element_ty))


def sk08_conv1d_silu(
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    conv_state: torch.Tensor,
) -> torch.Tensor:
    """conv1d causal + SiLU + shift de estado. ``mixed_qkv`` [B, D] -> [B, D]."""
    B, D = mixed_qkv.shape
    out = torch.empty_like(mixed_qkv)
    _sk08_conv1d_silu_kernel[(B, triton.cdiv(D, 1024))](
        mixed_qkv, conv_weight, conv_bias, conv_state, out,
        D,
        mixed_qkv.stride(0), conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
        out.stride(0),
        WIDTH=SK08_CONV_KERNEL_WIDTH, BLOCK_D=1024, num_warps=8, num_stages=2,
    )
    return out


def sk08_gated_delta_rule(
    qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state: torch.Tensor,
    scale: float = 1.0,
    num_k_heads: int = SK08_NUM_K_HEADS,
    num_v_heads: int = SK08_NUM_V_HEADS,
    head_dim: int = SK08_HEAD_DIM,
) -> torch.Tensor:
    """Gated delta rule recurrente. Actualiza ``ssm_state`` in-place, devuelve ``o``."""
    B = qkv.shape[0]
    o = torch.empty((B, num_v_heads, head_dim), dtype=qkv.dtype, device=qkv.device)
    _sk08_gated_delta_rule_kernel[(B * num_v_heads,)](
        qkv, a, b, A_log, dt_bias, ssm_state, o,
        qkv.stride(0), a.stride(0), ssm_state.stride(0), o.stride(0),
        SCALE=scale,
        H=num_k_heads, HV=num_v_heads, K=head_dim, V=head_dim,
        SOFTPLUS_THRESHOLD=SOFTPLUS_THRESHOLD,
        num_warps=4, num_stages=1,
    )
    return o


def sk08_ssm_control_bf16_fused(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    scale: float = 1.0,
    num_k_heads: int = SK08_NUM_K_HEADS,
    num_v_heads: int = SK08_NUM_V_HEADS,
    head_dim: int = SK08_HEAD_DIM,
) -> torch.Tensor:
    """Capa completa del control GDN en decode: conv1d+SiLU -> gated delta rule."""
    qkv = sk08_conv1d_silu(mixed_qkv, conv_weight, conv_bias, conv_state)
    return sk08_gated_delta_rule(
        qkv, a, b, A_log, dt_bias, ssm_state, scale, num_k_heads, num_v_heads, head_dim
    )


sk08_ssm_control = sk08_ssm_control_bf16_fused

__all__ = [
    "SK_ID", "SK_NAME", "SK08_CONV_DIM", "SK08_CONV_KERNEL_WIDTH",
    "SK08_NUM_GDN_LAYERS", "SK08_NUM_K_HEADS", "SK08_NUM_V_HEADS", "SK08_HEAD_DIM",
    "SK08_CONV1D_SHAPE", "SK08_A_LOG_SHAPE", "SK08_DT_BIAS_SHAPE",
    "sk08_conv1d_silu", "sk08_gated_delta_rule",
    "sk08_ssm_control_bf16_fused", "sk08_ssm_control",
]
