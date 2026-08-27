# SPDX-License-Identifier: Apache-2.0
"""warmup_all_kernels — precompilación de los super kernels SK-01..SK-11.

Único propósito: forzar la compilación JIT de Triton para cada kernel en cada
bucket de M antes del primer token, de modo que ningún request pague el stall
de compilación. No valida nada — dtypes, layouts y múltiplos de 128 se asumen
comprobados aguas arriba — y no atrapa excepciones: si un kernel no compila,
eso tiene que romper el arranque, no quedar enterrado.

La versión anterior envolvía cada kernel en ``try: ... except Exception: pass``.
Eso ocultaba fallos reales: el warmup de SK-02 le pasaba ``hidden`` bf16 a una
función que empieza con ``if hidden.dtype != torch.int8: raise TypeError``, así
que SK-02 **nunca se precalentaba** y nadie se enteraba. También pasaba siempre
``shifts = zeros``, con lo cual el camino diádico —la razón de ser de estos
kernels— jamás se ejercitaba.

Llamado una vez desde ``patch_PN110.install()`` tras el rebind — es decir
**en cada worker, antes de que vLLM haga su profiling de memoria**. Por eso los
buckets de M son chicos: lo que Triton compila depende de los ``tl.constexpr``
(el tile elegido por ``_cfg``), no del valor de M, así que alcanza con un M por
bucket. Precalentar con M=8000 compilaba exactamente lo mismo pero dejaba
gigabytes reservados en el allocator (la salida de ``lm_head`` a M=8000 son
1.99 GB por sí sola), y vLLM abortaba con "Free memory on device cuda:0 ... is
less than desired GPU memory utilization".
"""

from __future__ import annotations

import torch

from vllm._genesis.kernels import (
    sk01_gdn_qkvz,
    sk01_gdn_qkvz_w4a8,
    sk02_gdn_out,
    sk02_gdn_out_w4a8,
    sk03_fa_qkv,
    sk03_fa_qkv_w4a8,
    sk04_fa_o,
    sk04_fa_o_w4a8,
    sk05_mlp_gateup,
    sk05_mlp_gateup_w4a8,
    sk06_mlp_down,
    sk06_mlp_down_w4a8,
    sk07_lm_head,
    sk08_ssm_control,
    sk09_norm_embed,
    sk10_mtp_draft,
    sk10_mtp_draft_w4a8,
    sk11_vision,
)

# Un M por bucket de ``_cfg`` (m>32)+(m>128)+(m>512): 1 -> 0, 64 -> 1, 256 -> 2.
# Cubre las tres configuraciones distintas de tile sin alocar de más.
M_BUCKETS: tuple[int, ...] = (1, 64, 256)

# (K, N) per-rank TP=2 de cada super kernel.
SHAPES: tuple[tuple[str, int, int], ...] = (
    ("SK-01", 5120, 8192),    # GDN in_proj_qkvz
    ("SK-02", 3072, 5120),    # GDN out_proj (RowParallel)
    ("SK-03", 5120, 7168),    # FA qkv
    ("SK-04", 3072, 5120),    # FA o_proj (RowParallel)
    ("SK-05", 5120, 17408),   # MLP gate_up
    ("SK-06", 8704, 5120),    # MLP down_proj (RowParallel)
)

VOCAB_PER_RANK: int = 124160
HIDDEN: int = 5120
VISION_K: int = 1152
VISION_N: int = 4304


def _int8_operands(M: int, K: int, N: int, device: torch.device):
    """Activación int8 [M,K], peso int8 col-major [K,N], escalas y shifts no nulos."""
    a = torch.randint(-127, 128, (M, K), dtype=torch.int8, device=device)
    b = torch.empty_strided((K, N), (1, K), dtype=torch.int8, device=device)
    b.copy_(torch.randint(-127, 128, (N, K), dtype=torch.int8, device=device).t())
    a_scale = torch.rand(M, dtype=torch.float32, device=device) * 0.01 + 1e-3
    b_scale = torch.rand(N, dtype=torch.float32, device=device) * 0.01 + 1e-3
    # Shifts mixtos: ejercita el camino diádico de verdad, no sólo el neutro.
    shifts = torch.randint(-2, 3, (K // 128, N // 128), dtype=torch.int8, device=device)
    return a, b, a_scale, b_scale, shifts


def _w4a8_operands(M: int, K: int, N: int, device: torch.device):
    """Activación int8 [M,K], peso int4 empacado [K/2,N], escalas por grupo [K/128,N]."""
    a = torch.randint(-127, 128, (M, K), dtype=torch.int8, device=device)
    packed = torch.randint(-128, 128, (K // 2, N), dtype=torch.int8, device=device)
    a_scale = torch.rand(M, dtype=torch.float32, device=device) * 0.01 + 1e-3
    w_scale = torch.rand(K // 128, N, dtype=torch.float32, device=device) * 0.01 + 1e-3
    return a, packed, a_scale, w_scale


def warmup_all_kernels() -> int:
    """Precompila todos los super kernels. Devuelve el número de formas precalentadas."""
    device = torch.device("cuda")
    warmed = 0
    ln_w = torch.randn(HIDDEN, dtype=torch.bfloat16, device=device)

    for M in M_BUCKETS:
        # ── GEMM INT8 diádicos ────────────────────────────────────────────
        for _sk, K, N in SHAPES:
            a, b, a_s, b_s, sh = _int8_operands(M, K, N, device)
            resid = torch.randn(M, N, dtype=torch.bfloat16, device=device)
            sk01_gdn_qkvz.sk01_gdn_qkvz_gemm(a, b, a_s, b_s, sh, resid)
            sk02_gdn_out.sk02_gemm_int8_scaled(a, b, a_s, b_s, sh, resid)
            sk03_fa_qkv.sk03_fa_qkv_gemm(a, b, a_s, b_s, sh, resid)
            sk04_fa_o.fa_o_int8_scaled_gemm(a, b, a_s, b_s, sh, resid)
            sk05_mlp_gateup.sk05_gateup_gemm(a, b, a_s, b_s, sh, resid)
            sk06_mlp_down.mlp_down_gemm(a, b, a_s, b_s, sh, resid)
            sk10_mtp_draft.mtp_draft_fused_gemm(a, b, a_s, b_s, sh, resid)
            del a, b, a_s, b_s, sh, resid
            torch.cuda.empty_cache()
            warmed += 1

        # ── GEMM W4A8 ─────────────────────────────────────────────────────
        for _sk, K, N in SHAPES:
            a, packed, a_s, w_s = _w4a8_operands(M, K, N, device)
            resid = torch.randn(M, N, dtype=torch.bfloat16, device=device)
            sk01_gdn_qkvz_w4a8.sk01_gdn_qkvz_w4a8_gemm(a, packed, a_s, w_s)
            sk02_gdn_out_w4a8.sk02_gdn_out_w4a8_gemm(a, packed, a_s, w_s, resid)
            sk03_fa_qkv_w4a8.sk03_fa_qkv_w4a8_gemm(a, packed, a_s, w_s)
            sk04_fa_o_w4a8.sk04_fa_o_w4a8_gemm(a, packed, a_s, w_s, resid)
            sk05_mlp_gateup_w4a8.sk05_mlp_gateup_w4a8_gemm(a, packed, a_s, w_s)
            sk06_mlp_down_w4a8.sk06_mlp_down_w4a8_gemm(a, packed, a_s, w_s, resid)
            sk10_mtp_draft_w4a8.sk10_mtp_draft_w4a8_gemm(a, packed, a_s, w_s)
            del a, packed, a_s, w_s, resid
            torch.cuda.empty_cache()
            warmed += 1

        # ── Productores y epílogos: norm+quant, SiLU, lm_head, ViT ────────
        hidden = torch.randn(M, HIDDEN, dtype=torch.bfloat16, device=device)
        sk09_norm_embed.rmsnorm_quant_fused(hidden, ln_w)
        sk09_norm_embed.rmsnorm(hidden, ln_w)
        sk01_gdn_qkvz.sk01_rmsnorm_quant(hidden, ln_w)
        sk02_gdn_out.sk02_quant(hidden)
        _gu = torch.randn(M, 17408, dtype=torch.bfloat16, device=device)
        sk05_mlp_gateup.sk05_silu_mul_quant(_gu)
        sk05_mlp_gateup.sk05_silu_mul(_gu)
        del _gu

        # lm_head: sólo el camino con muestreo. El de vocabulario completo
        # compila el mismo kernel (mismos constexpr) pero materializaría
        # [M, 124160] de logits, que es lo que hacía reventar el arranque.
        lm_w = torch.empty_strided((HIDDEN, VOCAB_PER_RANK), (1, HIDDEN), dtype=torch.int8, device=device)
        lm_s = torch.rand(VOCAB_PER_RANK, dtype=torch.float32, device=device) * 0.01 + 1e-3
        lm_sh = torch.randint(-2, 3, (HIDDEN // 128, VOCAB_PER_RANK // 128), dtype=torch.int8, device=device)
        sk07_lm_head.lm_head_forward(
            hidden, lm_w, lm_s, lm_sh,
            torch.randint(0, VOCAB_PER_RANK, (1024,), dtype=torch.int32, device=device),
        )
        del lm_w, lm_s, lm_sh

        vis = torch.randn(M, VISION_K, dtype=torch.bfloat16, device=device)
        sk11_vision.sk11_linear_gelu(
            vis,
            torch.randn(VISION_K, VISION_N, dtype=torch.bfloat16, device=device),
            torch.randn(VISION_N, dtype=torch.bfloat16, device=device),
        )
        warmed += 1

        torch.cuda.empty_cache()

    # ── SK-08: control GDN, sólo decode (una fila por secuencia) ──────────
    cd = sk08_ssm_control.SK08_CONV_DIM
    hv = sk08_ssm_control.SK08_NUM_V_HEADS
    hd = sk08_ssm_control.SK08_HEAD_DIM
    for B in (1, 8, 32, 128):
        sk08_ssm_control.sk08_ssm_control_bf16_fused(
            torch.randn(B, cd, dtype=torch.bfloat16, device=device),
            torch.randn(B, hv, dtype=torch.bfloat16, device=device),
            torch.randn(B, hv, dtype=torch.bfloat16, device=device),
            torch.randn(hv, device=device),
            torch.randn(hv, device=device),
            torch.randn(cd, 4, dtype=torch.bfloat16, device=device),
            torch.randn(cd, dtype=torch.bfloat16, device=device),
            torch.randn(B, cd, 3, dtype=torch.bfloat16, device=device),
            torch.randn(B, hv, hd, hd, dtype=torch.float32, device=device),
            scale=hd ** -0.5,
        )
        warmed += 1

    torch.cuda.empty_cache()
    return warmed


__all__ = ["warmup_all_kernels", "M_BUCKETS", "SHAPES"]
