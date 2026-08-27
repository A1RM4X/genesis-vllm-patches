# SPDX-License-Identifier: Apache-2.0
"""P113 — MLP gate_up + SiLU fusionado en un solo GEMM (SK-05).

``Qwen2MoeMLP.forward`` (que ``qwen3_5.py`` importa como ``Qwen3NextMLP``) es::

    gate_up, _ = self.gate_up_proj(x)     # GEMM  -> [M, 17408] bf16
    out = self.act_fn(gate_up)            # SiLU  -> [M,  8704] bf16
    out, _ = self.down_proj(out)          # GEMM

Este parche reemplaza las dos primeras líneas por una sola llamada a
``sk05_gateup_silu_gemm``, que resuelve ``SiLU(gate)*up`` en el epílogo del
GEMM. ``down_proj`` queda intacto, así que toda la semántica de TensorParallel
(``reduce_results``, ``all_reduce``) sigue viviendo donde ya vivía.

Cómo se fusiona sin ampliar el tile
-----------------------------------
``gate`` y ``up`` están separados por N/2 columnas, así que un tile tendría que
cargar dos bloques de B a la vez; en sm_86 eso obliga a ``BLOCK_N=64`` para no
pasarse de los 99 KB de shared, y el tile angosto cuesta más de lo que ahorra
(medido: 10-17% peor). La solución es reordenar las columnas del peso **una vez
al cargar** a pares intercalados ``gate0,up0,gate1,up1,...``: entonces un tile
de ``BLOCK_N=128`` contiene 64 pares completos, el tile de B es tan ancho como
el de un GEMM normal y sólo la salida es la mitad.

Qué se gana (RTX 3090, K=5120, N=17408 per-rank)
------------------------------------------------
* **Velocidad: 1-6%.** M=1 0.108 vs 0.115 ms, M=512 0.552 vs 0.560,
  M=1664 2.384 vs 2.484, M=8000 11.614 vs 12.332. Es poco, y es lo esperable:
  el kernel de SiLU era el 2-4% de la capa. No esperes más de acá.
* **VRAM: 278 MB menos a M=8000.** El intermedio ``[M, 17408]`` bf16 nunca se
  materializa. En una placa de 24 GB esto vale más que el 6%.
* **Exactitud: 2× mejor.** ``gate`` y ``up`` llegan a la SiLU en fp32 en vez de
  pasar por bf16 y volver: error relativo mediano 1.1e-3 vs 2.4e-3.
* Un lanzamiento menos por capa (65 capas).

La permutación **reemplaza** al peso original para no duplicar 89 MB por capa
(5.7 GB sobre 64 capas). Si alguien llamara igual a ``gate_up_proj`` por fuera
de este parche, ``apply`` de PN110 devolvería las columnas permutadas; para que
eso no pueda dar un resultado silenciosamente incorrecto, el estado guarda la
permutación y PN110 deshace el orden en ese camino (que no debería ocurrir
nunca, y es el único ``if`` de todo esto).

Depende de PN110: sólo actúa sobre capas que tengan estado INT8 con
``sk_id == "SK-05"``. Si PN110 está apagado, el forward original queda intacto.

Por qué viene APAGADO por defecto
----------------------------------
Medido en RTX 3090 contra ``ops.cutlass_scaled_mm`` (gate_up, K=5120 N=17408):
``cutlass`` gana 1.55x a M=512 y 1.13x a M=1664; empatan a M=8000 (12.31 vs
11.99 ms). Como la permutación **reemplaza** al peso y el layout intercalado
sólo lo entiende ``sk05_gateup_silu_gemm``, una vez permutado ya no se puede
volver a cutlass para los M donde cutlass gana. O sea: P113 conviene sólo si el
prefill corre consistentemente en M >= 4096; con prefill chunked a 512-2048
tokens es una regresión.

Env: ``GENESIS_P113_MLP_FUSED_SILU=1`` lo activa (default 0).
"""

from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("genesis.wiring.p113_mlp_fused_silu")

ENV_FLAG = "GENESIS_P113_MLP_FUSED_SILU"
_TRUTHY = ("1", "true", "yes", "on")
_MARKER = "_genesis_p113_installed"
_LAYER_ATTR = "_genesis_pn110_int8"


def _enabled() -> bool:
    return os.environ.get(ENV_FLAG, "0").strip().lower() in _TRUTHY


def _fused_gate_up(gate_up_proj, x_2d: torch.Tensor) -> torch.Tensor | None:
    """Devuelve ``SiLU(gate)*up`` [M, N/2] o None si la capa no es elegible."""
    state = getattr(gate_up_proj, _LAYER_ATTR, None)
    if state is None or state.get("sk_id") != "SK-05":
        return None
    if getattr(gate_up_proj, "gather_output", False):
        return None

    from vllm._genesis.kernels.sk05_mlp_gateup import (
        sk05_gateup_silu_gemm,
        sk05_permute_gateup,
    )
    from vllm._genesis.kernels.sk09_norm_embed import quant_per_token

    w = state.get("sk_perm_w")
    if w is None:
        # Permutación una sola vez, al primer forward. Reemplaza al peso
        # original (no lo duplica) y deja registrado el orden para que el
        # camino no fusionado de PN110 pueda deshacerlo.
        w, bs = sk05_permute_gateup(state["b_col"], state["sk_bscales"])
        n2 = w.shape[1] // 2
        perm = torch.empty(w.shape[1], dtype=torch.long, device=w.device)
        idx = torch.arange(n2, dtype=torch.long, device=w.device)
        perm[0::2] = idx
        perm[1::2] = idx + n2
        state["sk_perm_w"] = w
        state["sk_perm_bs"] = bs
        state["sk_gateup_perm"] = perm
        state["b_col"] = w
        state["w_int8"] = w
        state["sk_bscales"] = bs
        torch.cuda.empty_cache()
        log.info("P113: gate_up permutado a pares intercalados (%s)", tuple(w.shape))

    out_dtype = x_2d.dtype if x_2d.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
    a_i8, a_scales = quant_per_token(x_2d)
    return sk05_gateup_silu_gemm(
        a_i8, w, a_scales.reshape(-1), state["sk_perm_bs"], state["sk_shifts"], out_dtype
    )


def _make_forward(original):
    def forward(self, x):
        if self.expert_gate is None:
            x_2d = x.reshape(-1, x.shape[-1])
            act = _fused_gate_up(self.gate_up_proj, x_2d)
            if act is not None:
                out, _ = self.down_proj(act.reshape(*x.shape[:-1], -1))
                return out
        return original(self, x)

    return forward


def apply() -> tuple[str, str]:
    """Aplica P113 sobre ``Qwen2MoeMLP`` (alias ``Qwen3NextMLP``)."""
    from vllm._genesis.dispatcher import log_decision

    if not _enabled():
        log_decision("P113", False, f"{ENV_FLAG}=0")
        return "skipped", f"{ENV_FLAG}=0"

    try:
        from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
    except Exception as e:
        log_decision("P113", False, f"import failed: {e}")
        return "skipped", f"Qwen2MoeMLP no importable: {e}"

    if getattr(Qwen2MoeMLP, _MARKER, False):
        return "applied", "ya instalado"

    Qwen2MoeMLP.forward = _make_forward(Qwen2MoeMLP.forward)
    setattr(Qwen2MoeMLP, _MARKER, True)
    log_decision("P113", True, "gate_up+SiLU fusionado (SK-05)")
    return "applied", "Qwen2MoeMLP.forward -> gate_up+SiLU fusionado (SK-05)"


__all__ = ["apply", "ENV_FLAG"]
