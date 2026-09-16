# SPDX-License-Identifier: Apache-2.0
"""PN116 — super kernel de MLP AISLADO al prefill (SK-05), con A/B incorporado.

Por que existe
--------------
El MLP es el **70,2% de los FLOPs de GEMM** del modelo (17,38 B de 24,76 B
totales, contados del indice de safetensors). Es el unico bloque que vale la
pena atacar con un super kernel, y aparece identico en las 64 capas.

Que lo diferencia de P113
-------------------------
P113 instala ``sk05_gateup_silu_gemm`` en **todos** los M. Su propio docstring
registra la medicion que lo condena en la config real:

    cutlass gana 1.55x a M=512 y 1.13x a M=1664; empatan a M=8000.

Con ``--long-prefill-token-threshold 2048`` el prefill corre siempre a M=2048,
o sea dentro del rango donde el super kernel PIERDE. P113 queda apagado no
porque el kernel sea malo sino porque se lo llama en el M equivocado.

PN116 dispatchea **solo por encima de un M minimo** (``GENESIS_PN116_M_MIN``,
default 4096). Debajo de eso el forward original queda intacto y corre por
Marlin/cutlass, que ahi es mas rapido. Decode (M = 4..40) nunca se toca.

Nota de operacion: medido el 2026-09-12, subir el chunk de prefill de 2048 a
8192 NO cuesta throughput (1.504 vs 1.466 tok/s agregados), asi que mudarse a
M>=4096 para habilitar este camino es gratis. Si el threshold configurado es
menor que ``M_MIN``, el kernel no se llama nunca y este parche avisa al aplicar
en vez de quedarse mudo.

A/B incorporado
---------------
Con ``GENESIS_PN116_AB=N`` las primeras N invocaciones elegibles corren los DOS
caminos, comparan numericamente (max abs diff y error relativo) y cronometran
ambos con eventos CUDA. Al terminar loguea el veredicto. Es el punto de la
palabra "aislado": se puede tunear el kernel y leer el delta sin desarmar nada.

Dependencias
------------
Necesita el estado INT8 por capa que deja PN110 (``sk_id == "SK-05"``,
``sk_perm_w``, ``sk_perm_bs``, ``sk_shifts``). Sin PN110 no hay nada que
dispatchear y el parche se salta con una razon explicita.

Mutuamente excluyente con P113: los dos envuelven el mismo ``forward``.

Ampere: el camino fusionado termina en ``mma.m16n8k32`` (IMMA int8) via
``sk05_gateup_silu_gemm``; el epilogo resuelve ``SiLU(gate)*up`` sin volver a
memoria, que es de donde sale la ganancia sobre ``cutlass_scaled_mm`` + un
kernel de activacion aparte.

Env
---
``GENESIS_ENABLE_PN116_PREFILL_MLP_SK=1``  activa (default 0)
``GENESIS_PN116_M_MIN=4096``               M minimo para dispatchear
``GENESIS_PN116_AB=0``                     N invocaciones a medir en A/B
"""

from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("genesis.wiring.pn116_prefill_mlp_superkernel")

ENV_FLAG = "GENESIS_ENABLE_PN116_PREFILL_MLP_SK"
_TRUTHY = ("1", "true", "yes", "on")
_MARKER = "_genesis_pn116_installed"
_P113_MARKER = "_genesis_p113_installed"
_LAYER_ATTR = "_genesis_sk_state"

DEFAULT_M_MIN = 4096


def _enabled() -> bool:
    return os.environ.get(ENV_FLAG, "0").strip().lower() in _TRUTHY


def _m_min() -> int:
    try:
        return int(os.environ.get("GENESIS_PN116_M_MIN", str(DEFAULT_M_MIN)))
    except ValueError:
        return DEFAULT_M_MIN


def _ab_budget() -> int:
    try:
        return int(os.environ.get("GENESIS_PN116_AB", "0"))
    except ValueError:
        return 0


class _Stats:
    """Contadores del dispatch. Sin esto el parche es invisible."""

    def __init__(self) -> None:
        self.calls = 0
        self.dispatched = 0
        self.skipped_small_m = 0
        self.skipped_no_state = 0
        self.ab_left = _ab_budget()
        self.ab_fused_ms = 0.0
        self.ab_ref_ms = 0.0
        self.ab_n = 0
        self.ab_max_abs = 0.0
        self.ab_max_rel = 0.0
        self.reported = False


STATS = _Stats()


def _fused_gate_up(gate_up_proj, x_2d: torch.Tensor):
    """``SiLU(gate)*up`` [M, N/2] por el super kernel, o None si no aplica."""
    state = getattr(gate_up_proj, _LAYER_ATTR, None)
    if state is None or state.get("sk_id") != "SK-05":
        return None
    if getattr(gate_up_proj, "gather_output", False):
        return None
    w = state.get("sk_perm_w")
    if w is None:
        return None

    from vllm._genesis.kernels.sk05_mlp_gateup import sk05_gateup_silu_gemm
    from vllm._genesis.kernels.sk09_norm_embed import quant_per_token

    out_dtype = (x_2d.dtype if x_2d.dtype in (torch.float16, torch.bfloat16)
                 else torch.bfloat16)
    a_i8, a_scales = quant_per_token(x_2d)
    return sk05_gateup_silu_gemm(
        a_i8, w, a_scales.reshape(-1), state["sk_perm_bs"],
        state["sk_shifts"], out_dtype,
    )


def _time_cuda(fn):
    """Cronometra en GPU. Devuelve (resultado, ms)."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return out, start.elapsed_time(end)


def _ab_compare(mlp, x_2d, fused):
    """Corre tambien el camino de referencia y acumula el delta."""
    s = STATS
    try:
        def _ref():
            gate_up, _ = mlp.gate_up_proj(x_2d)
            return mlp.act_fn(gate_up)

        ref, ref_ms = _time_cuda(_ref)
        _, fused_ms = _time_cuda(lambda: _fused_gate_up(mlp.gate_up_proj, x_2d))

        diff = (fused.float() - ref.float()).abs()
        denom = ref.float().abs().clamp_min(1e-3)
        s.ab_max_abs = max(s.ab_max_abs, float(diff.max()))
        s.ab_max_rel = max(s.ab_max_rel, float((diff / denom).max()))
        s.ab_fused_ms += fused_ms
        s.ab_ref_ms += ref_ms
        s.ab_n += 1
    except Exception as e:
        log.warning("[PN116] A/B fallo, se ignora: %s", e)
    finally:
        s.ab_left -= 1
        if s.ab_left <= 0 and not s.reported:
            s.reported = True
            if s.ab_n:
                log.warning(
                    "[PN116] A/B sobre %d invocaciones a M>=%d: "
                    "fusionado %.3f ms/llamada vs referencia %.3f ms/llamada "
                    "(%.2fx) | max_abs=%.4g max_rel=%.4g",
                    s.ab_n, _m_min(), s.ab_fused_ms / s.ab_n,
                    s.ab_ref_ms / s.ab_n,
                    (s.ab_ref_ms / s.ab_fused_ms) if s.ab_fused_ms else 0.0,
                    s.ab_max_abs, s.ab_max_rel,
                )
            else:
                log.warning("[PN116] A/B no junto ninguna muestra.")


def _make_forward(original):
    m_min = _m_min()

    def forward(self, x):
        s = STATS
        s.calls += 1
        if self.expert_gate is None:
            x_2d = x.reshape(-1, x.shape[-1])
            if x_2d.shape[0] >= m_min:
                act = _fused_gate_up(self.gate_up_proj, x_2d)
                if act is not None:
                    s.dispatched += 1
                    if s.ab_left > 0:
                        _ab_compare(self, x_2d, act)
                    out, _ = self.down_proj(act.reshape(*x.shape[:-1], -1))
                    return out
                s.skipped_no_state += 1
            else:
                s.skipped_small_m += 1
        return original(self, x)

    return forward


def _chunk_config_warning() -> str | None:
    """Avisa si el chunk de prefill hace que el kernel no se llame NUNCA.

    Es el modo de falla de P113: el kernel esta bien pero se lo invoca en un M
    donde pierde, o directamente no se lo invoca.
    """
    try:
        import sys
        argv = " ".join(sys.argv)
        import re
        m = re.search(r"--long-prefill-token-threshold[= ]+(\d+)", argv)
        if not m:
            m = re.search(r"--max-num-batched-tokens[= ]+(\d+)", argv)
        if m and int(m.group(1)) < _m_min():
            return (
                f"el chunk de prefill es {m.group(1)} y M_MIN es {_m_min()}: "
                f"el super kernel NO se va a llamar nunca. Subi "
                f"--long-prefill-token-threshold (medido: subirlo de 2048 a "
                f"8192 no cuesta throughput) o baja GENESIS_PN116_M_MIN."
            )
    except Exception:
        pass
    return None


def apply() -> tuple[str, str]:
    """Instala PN116 sobre ``Qwen2MoeMLP.forward`` (alias ``Qwen3NextMLP``)."""
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN116")
    log_decision("PN116", decision, reason)
    if not decision:
        return "skipped", reason

    try:
        from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
    except Exception as e:
        return "skipped", f"Qwen2MoeMLP no importable: {e}"

    if getattr(Qwen2MoeMLP, _P113_MARKER, False):
        return "skipped", (
            "P113 ya envolvio este forward — PN116 y P113 son mutuamente "
            "excluyentes. Apaga GENESIS_P113_MLP_FUSED_SILU."
        )
    if getattr(Qwen2MoeMLP, _MARKER, False):
        return "applied", "ya instalado"

    Qwen2MoeMLP.forward = _make_forward(Qwen2MoeMLP.forward)
    setattr(Qwen2MoeMLP, _MARKER, True)

    msg = (
        f"MLP gate_up+SiLU por SK-05 solo a M>={_m_min()} "
        f"(decode y prefill chico quedan en Marlin/cutlass)"
    )
    warn = _chunk_config_warning()
    if warn:
        log.warning("[PN116] %s", warn)
        msg += f" — OJO: {warn}"
    if _ab_budget() > 0:
        msg += f" | A/B activo sobre {_ab_budget()} invocaciones"
    return "applied", msg


def stats() -> dict:
    """Contadores del dispatch, para inspeccion desde fuera."""
    s = STATS
    return {
        "calls": s.calls,
        "dispatched": s.dispatched,
        "skipped_small_m": s.skipped_small_m,
        "skipped_no_state": s.skipped_no_state,
        "m_min": _m_min(),
        "ab_samples": s.ab_n,
        "ab_fused_ms_avg": (s.ab_fused_ms / s.ab_n) if s.ab_n else 0.0,
        "ab_ref_ms_avg": (s.ab_ref_ms / s.ab_n) if s.ab_n else 0.0,
        "ab_max_abs": s.ab_max_abs,
        "ab_max_rel": s.ab_max_rel,
    }


__all__ = ["apply", "stats", "ENV_FLAG", "STATS"]
