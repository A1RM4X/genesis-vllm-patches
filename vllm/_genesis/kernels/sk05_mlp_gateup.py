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

import torch
import triton
import triton.language as tl

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
    _sk05_rmsnorm_quant_kernel[(M,)](
        hidden, ln_weight, q, s,
        K, hidden.stride(0), q.stride(0),
        BLOCK=triton.next_power_of_2(K), EPS=eps, num_warps=8, num_stages=1,
    )
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
    _sk05_mlp_gateup_kernel[grid](
        a, b, out, res, a_scales, b_scales, shifts,
        M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        out.stride(0), out.stride(1), rm, rn,
        shifts.stride(0), shifts.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=gm, SHIFT_BLOCK=SHIFT_BLOCK,
        num_warps=warps, num_stages=stages,
    )
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
