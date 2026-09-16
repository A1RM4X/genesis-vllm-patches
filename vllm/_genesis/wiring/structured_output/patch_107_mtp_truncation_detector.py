# SPDX-License-Identifier: Apache-2.0
"""Wiring for P107 — vllm#41467 backport: MTP truncation detector.

При MTP K≥1 + tools + reasoning parser возможна редкая (≈0.25% по
автору на Qwen3.6-27B-FP8) ситуация: модель производит EOS на
boundary reasoning→tool_call. finish_reason=stop + tools configured +
ни tool_calls, ни content не отдано → silent client-side error.

Решение: defensive guard в `chat_completion_stream_generator` которая
detect'ит этот combo и raise GenerationError (retryable) вместо
silent stop. Клиент получит SSE error event и retry.

Affects прямо наш PROD: 27B Lorbus + MTP K=3 + tools — exactly the
config which author reported. P107 — safety net на уже defended path
(P59/P60/P61/P62/P64/P68/P69 family).

Default OFF — defensive backport. Risk: low, добавляет один extra
branch в hot path, не аффектит happy path.

Author: Sandermage backport (ToastyTheBot, vllm#41467).
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)

log = logging.getLogger("genesis.wiring.p107_mtp_truncation_detector")

GENESIS_P107_MARKER = "Genesis P107 MTP truncation detector (vllm#41467) v2"


def _is_enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_P107_MTP_TRUNCATION_DETECTOR", ""
    ).strip().lower() in ("1", "true", "yes", "on")


# Re-anclado 2026-08-26 para pin v0.27.1: el bloque finish_reason del
# stream generator ya no evalúa auto_tools_called/use_harmony (quedó en
# `if tools_streamed[i] and not tool_choice_function_name:`). La condición
# `not auto_tools_called` se cubre con `not tools_streamed[i]`; el reasoner
# se consulta vía `parser.reasoning_parser` (ParserManager facade).
# GenerationError se movió a entrypoints/openai/engine/protocol.py.
ANCHOR_OLD = (
    "                        if tools_streamed[i] and not tool_choice_function_name:\n"
    "                            finish_reason_ = \"tool_calls\"\n"
    "                        else:\n"
    "                            finish_reason_ = (\n"
    "                                output.finish_reason if output.finish_reason else \"stop\"\n"
    "                            )\n"
    "                        choice_data = ChatCompletionResponseStreamChoice("
)

# V1 patch that unconditionally raised GenerationError
ANCHOR_V1 = (
    "                        if tools_streamed[i] and not tool_choice_function_name:\n"
    "                            finish_reason_ = \"tool_calls\"\n"
    "                        else:\n"
    "                            finish_reason_ = (\n"
    "                                output.finish_reason if output.finish_reason else \"stop\"\n"
    "                            )\n"
    "\n"
    "                        # [Genesis P107 vllm#41467] MTP truncation detector.\n"
    "                        # ~0.25% rate on Qwen3.6 27B-FP8 + MTP K=3 (per upstream\n"
    "                        # author): EOS at reasoning→tool_call boundary leaves\n"
    "                        # finish_reason=stop with no content/tool_calls. Raise\n"
    "                        # retryable error so client retries instead of seeing\n"
    "                        # silent empty response. Defensive AND-conditions,\n"
    "                        # no impact on happy path.\n"
    "                        if (\n"
    "                            finish_reason_ == \"stop\"\n"
    "                            and request.tools\n"
    "                            and not tools_streamed[i]\n"
    "                            and parser is not None\n"
    "                            and parser.reasoning_parser is not None\n"
    "                            and delta_message is not None\n"
    "                            and not delta_message.content\n"
    "                            and not delta_message.tool_calls\n"
    "                        ):\n"
    "                            from vllm.entrypoints.openai.engine.protocol import (\n"
    "                                GenerationError as _P107_GenError,\n"
    "                            )\n"
    "                            logger.warning(\n"
    "                                \"[Genesis P107] MTP truncation detected for request %s: \"\n"
    "                                \"finished with 'stop' but tools configured and only \"\n"
    "                                \"reasoning produced.\",\n"
    "                                request_id,\n"
    "                            )\n"
    "                            raise _P107_GenError(\n"
    "                                \"MTP speculative decoding truncated tool call \"\n"
    "                                \"generation. Please retry.\"\n"
    "                            )\n"
    "                        choice_data = ChatCompletionResponseStreamChoice("
)

ANCHOR_NEW = (
    "                        if tools_streamed[i] and not tool_choice_function_name:\n"
    "                            finish_reason_ = \"tool_calls\"\n"
    "                        else:\n"
    "                            finish_reason_ = (\n"
    "                                output.finish_reason if output.finish_reason else \"stop\"\n"
    "                            )\n"
    "\n"
    "                        # [Genesis P107 vllm#41467] MTP truncation detector.\n"
    "                        # ~0.25% rate on Qwen3.6 27B-FP8 + MTP K=3 (per upstream\n"
    "                        # author): EOS at reasoning→tool_call boundary leaves\n"
    "                        # finish_reason=stop with no content/tool_calls. Raise\n"
    "                        # retryable error only when GENESIS_P107_RAISE_ERROR=1,\n"
    "                        # otherwise complete cleanly to prevent infinite loops in\n"
    "                        # agentic clients (OpenCode/Cline).\n"
    "                        if (\n"
    "                            finish_reason_ == \"stop\"\n"
    "                            and request.tools\n"
    "                            and not tools_streamed[i]\n"
    "                            and parser is not None\n"
    "                            and parser.reasoning_parser is not None\n"
    "                            and delta_message is not None\n"
    "                            and not delta_message.content\n"
    "                            and not delta_message.tool_calls\n"
    "                        ):\n"
    "                            logger.warning(\n"
    "                                \"[Genesis P107] MTP truncation detected for request %s: \"\n"
    "                                \"finished with 'stop' but tools configured and only \"\n"
    "                                \"reasoning produced.\",\n"
    "                                request_id,\n"
    "                            )\n"
    "                            import os as _p107_os\n"
    "                            if _p107_os.environ.get(\"GENESIS_P107_RAISE_ERROR\", \"\").strip().lower() in (\"1\", \"true\", \"yes\"):\n"
    "                                from vllm.entrypoints.openai.engine.protocol import (\n"
    "                                    GenerationError as _P107_GenError,\n"
    "                                )\n"
    "                                raise _P107_GenError(\n"
    "                                    \"MTP speculative decoding truncated tool call \"\n"
    "                                    \"generation. Please retry.\"\n"
    "                                )\n"
    "                        choice_data = ChatCompletionResponseStreamChoice("
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(
        "entrypoints/openai/chat_completion/serving.py"
    )
    if target is None:
        return None
    # Los dos anchors son ALTERNATIVAS mutuamente excluyentes (texto virgen vs
    # texto ya parcheado por la v1), no piezas independientes. Dejarlos como dos
    # sub-patches required=False bajo un mismo marker es la trampa de aplicacion
    # parcial que prohibe A-19: el primero que calza escribe el marker y el otro
    # no se reintenta nunca. Se elige el anchor al construir y queda UNO solo,
    # required=True.
    anchor = ANCHOR_OLD
    name = "p107_mtp_truncation_v2"
    try:
        with open(target, encoding="utf-8") as f:
            content = f.read()
        if ANCHOR_V1 in content:
            anchor = ANCHOR_V1
            name = "p107_mtp_truncation_v1_upgrade"
    except OSError:
        pass

    return TextPatcher(
        patch_name="P107 MTP truncation detector (vllm#41467)",
        target_file=str(target),
        marker=GENESIS_P107_MARKER,
        sub_patches=[
            TextPatch(
                name=name,
                anchor=anchor,
                replacement=ANCHOR_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("P107")
    log_decision("P107", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "serving.py not found"
    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        import os as _os
        modo = ("error reintentable" if _os.environ.get(
            "GENESIS_P107_RAISE_ERROR", "").strip().lower() in ("1", "true", "yes")
            else "solo warning (GENESIS_P107_RAISE_ERROR=1 para el error)")
        return "applied", f"P107 aplicado: deteccion de truncamiento MTP — {modo}"
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied (idempotent)"
    if result == TextPatchResult.SKIPPED:
        msg = failure.reason if failure else "anchor not found"
        return "skipped", f"{msg} — likely upstream merged"
    return "failed", failure.reason if failure else "unknown failure"
