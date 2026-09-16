# SPDX-License-Identifier: Apache-2.0
"""KV Offload Activity Tracker & REST Endpoints for vLLM (Genesis PN89/PN91).

Provides in-memory ring-buffer tracking of recent chat completion requests,
exposes GET /v1/kv-offload/requests, and provides safe zero-downtime cache
and metrics resetting via POST /v1/kv-offload/reset with graceful client
termination notifications.
"""
from __future__ import annotations

import asyncio
import collections
import glob
import json
import logging
import os
import shutil
import threading
import time
from typing import Any, AsyncIterator, Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("vllm._genesis.kv_offload_tracker")

_MAX_REQUESTS = 300
_RING_BUFFER = collections.deque(maxlen=_MAX_REQUESTS)
_LOCK = threading.Lock()

# Set of abort events for active streaming connections
_ACTIVE_STREAMS: set[asyncio.Event] = set()

# Prometheus metrics for request-level execution statistics
try:
    from prometheus_client import Counter, Gauge

    PROM_LAST_PP_TPS = Gauge(
        "vllm:request_last_pp_tokens_per_second",
        "Genesis Prompt Processing (PP / prefill) tokens per second of the last completed request",
    )
    PROM_LAST_TG_TPS = Gauge(
        "vllm:request_last_tg_tokens_per_second",
        "Genesis Text Generation (TG / decode) tokens per second of the last completed request",
    )
    PROM_LAST_KV_HIT_RATE = Gauge(
        "vllm:request_last_kv_hit_rate_pct",
        "Genesis KV cache hit percentage of the last completed request",
    )
    PROM_REQS_BY_PRIORITY = Counter(
        "vllm:requests_by_priority_total",
        "Total completed requests partitioned by priority tier",
        ["priority"],
    )
    PROM_REQS_BY_AGENT = Counter(
        "vllm:requests_by_agent_total",
        "Total completed requests partitioned by agent name",
        ["agent"],
    )
except Exception as _prom_err:
    logger.debug("Prometheus metrics not initialized in kv_offload_tracker: %s", _prom_err)
    PROM_LAST_PP_TPS = None
    PROM_LAST_TG_TPS = None
    PROM_LAST_KV_HIT_RATE = None
    PROM_REQS_BY_PRIORITY = None
    PROM_REQS_BY_AGENT = None

router = APIRouter()


class RequestTrackerContext:
    __slots__ = ("t0", "timestamp", "agent", "name", "priority", "id", "ttft", "usage", "status", "prompt_chars", "stream_output_tokens")

    def __init__(self, agent: Optional[str] = None, name: Optional[str] = None, priority: int = 0, prompt_chars: int = 0):
        self.t0 = time.perf_counter()
        self.timestamp = time.time()
        self.agent = agent
        self.name = name or agent
        self.priority = priority
        self.prompt_chars = prompt_chars
        self.stream_output_tokens = 0
        self.id: Optional[str] = None
        self.ttft: Optional[float] = None
        self.usage: Optional[dict[str, Any]] = None
        self.status = 200


def create_tracker(req: Any) -> RequestTrackerContext:
    agent = None
    name = None
    priority = 0
    prompt_chars = 0
    try:
        if hasattr(req, "priority") and req.priority is not None:
            priority = int(req.priority)
        if hasattr(req, "user") and req.user:
            name = str(req.user)

        if hasattr(req, "messages") and req.messages:
            for m in req.messages:
                content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
                if isinstance(content, str):
                    prompt_chars += len(content)
                elif isinstance(content, list):
                    for part in content:
                        part_text = part.get("text") if isinstance(part, dict) else getattr(part, "text", "")
                        if isinstance(part_text, str):
                            prompt_chars += len(part_text)
        elif hasattr(req, "prompt") and req.prompt:
            if isinstance(req.prompt, str):
                prompt_chars = len(req.prompt)
            elif isinstance(req.prompt, list):
                prompt_chars = sum(len(str(p)) for p in req.prompt)

        if hasattr(req, "kv_transfer_params") and isinstance(req.kv_transfer_params, dict):
            agent = req.kv_transfer_params.get("genesis_agent") or req.kv_transfer_params.get("agent")
            if not priority and "priority" in req.kv_transfer_params:
                try:
                    priority = int(req.kv_transfer_params["priority"])
                except Exception:
                    pass
            if not name and "name" in req.kv_transfer_params:
                name = req.kv_transfer_params["name"]

        # Heuristic resolution if priority is 0 and agent is known
        if priority == 0 and agent:
            try:
                from vllm._genesis.dynamic_pid_gating import AGENT_PRIORITY_MAP
                priority = AGENT_PRIORITY_MAP.get(agent, 0)
            except Exception:
                pass

        if not name:
            name = agent or "direct"
    except Exception:
        pass
    return RequestTrackerContext(agent=agent, name=name, priority=priority, prompt_chars=prompt_chars)


def observe_chunk(ctx: RequestTrackerContext, chunk_data: Any) -> None:
    try:
        if isinstance(chunk_data, (str, bytes)):
            raw = chunk_data if isinstance(chunk_data, str) else chunk_data.decode("utf-8", "replace")
            if '"id"' in raw and ctx.id is None:
                idx = raw.find('"id":"')
                if idx != -1:
                    end_idx = raw.find('"', idx + 6)
                    if end_idx != -1:
                        ctx.id = raw[idx + 6:end_idx]

            has_content_piece = False
            if '"content":' in raw or '"reasoning_content":' in raw:
                # Count stream tokens and mark TTFT on first real token
                for line in raw.split("\n"):
                    if line.startswith("data:") and line.strip() != "data: [DONE]":
                        try:
                            obj = json.loads(line[5:].strip())
                            choices = obj.get("choices") or []
                            if choices:
                                delta = choices[0].get("delta") or {}
                                piece = delta.get("content") or delta.get("reasoning_content")
                                if piece:
                                    has_content_piece = True
                                    ctx.stream_output_tokens += 1
                        except Exception:
                            pass

            if has_content_piece and ctx.ttft is None:
                ctx.ttft = time.perf_counter() - ctx.t0

            if '"usage"' in raw and '"usage":null' not in raw.replace(" ", ""):
                try:
                    for line in raw.split("\n"):
                        if line.startswith("data:") and line.strip() != "data: [DONE]":
                            obj = json.loads(line[5:].strip())
                            u = obj.get("usage")
                            if u:
                                ctx.usage = u
                                if obj.get("id"):
                                    ctx.id = obj.get("id")
                except Exception:
                    pass
        elif ctx.ttft is None:
            ctx.ttft = time.perf_counter() - ctx.t0
    except Exception:
        pass


def finish_request(ctx: RequestTrackerContext, response_obj_or_dict: Any = None, status: int = 200) -> None:
    try:
        e2e_ms = round((time.perf_counter() - ctx.t0) * 1000.0, 1)
        ttft_ms = round(ctx.ttft * 1000.0, 1) if ctx.ttft is not None else None

        req_id = ctx.id
        prompt_tokens = None
        cached_tokens = None
        output_tokens = None

        if response_obj_or_dict is not None:
            if hasattr(response_obj_or_dict, "id"):
                req_id = response_obj_or_dict.id
            if hasattr(response_obj_or_dict, "usage") and response_obj_or_dict.usage:
                u = response_obj_or_dict.usage
                prompt_tokens = getattr(u, "prompt_tokens", None)
                output_tokens = getattr(u, "completion_tokens", None)
                det = getattr(u, "prompt_tokens_details", None)
                if det:
                    cached_tokens = getattr(det, "cached_tokens", None)
            elif isinstance(response_obj_or_dict, dict):
                req_id = response_obj_or_dict.get("id", req_id)
                u = response_obj_or_dict.get("usage") or {}
                prompt_tokens = u.get("prompt_tokens")
                output_tokens = u.get("completion_tokens")
                det = u.get("prompt_tokens_details") or {}
                cached_tokens = det.get("cached_tokens")
        elif ctx.usage:
            u = ctx.usage
            prompt_tokens = u.get("prompt_tokens")
            output_tokens = u.get("completion_tokens")
            det = u.get("prompt_tokens_details") or {}
            cached_tokens = det.get("cached_tokens")

        # Fallbacks for streaming responses without explicit usage object
        if output_tokens is None and ctx.stream_output_tokens > 0:
            output_tokens = ctx.stream_output_tokens
        if prompt_tokens is None:
            if ctx.prompt_chars > 0:
                prompt_tokens = max(1, int(ctx.prompt_chars / 3.5))
            elif ttft_ms is not None and ttft_ms > 0:
                prompt_tokens = max(1, int(ttft_ms * 1.5))
            else:
                prompt_tokens = 16
        if ttft_ms is None and e2e_ms is not None:
            ttft_ms = min(e2e_ms, 50.0)

        # Calculate PP TPS (Prompt Processing)
        pp_tps = None
        if ttft_ms is not None and ttft_ms > 0 and prompt_tokens:
            pp_tps = round(prompt_tokens / (ttft_ms / 1000.0), 1)

        # Calculate TG TPS (Text Generation / decode)
        tg_tps = None
        if output_tokens is not None and output_tokens > 0 and e2e_ms is not None:
            decode_ms = max(1.0, e2e_ms - (ttft_ms or 0.0))
            tg_tps = round(output_tokens / (decode_ms / 1000.0), 1)

        # Calculate KV Cache Hit Rate %
        kv_hit_rate_pct = None
        if prompt_tokens and prompt_tokens > 0 and cached_tokens is not None:
            kv_hit_rate_pct = round((cached_tokens / prompt_tokens) * 100.0, 1)

        record = {
            "id": req_id,
            "timestamp": round(ctx.timestamp, 3),
            "name": ctx.name,
            "agent": ctx.agent,
            "priority": ctx.priority,
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "output_tokens": output_tokens,
            "ttft_ms": ttft_ms,
            "e2e_ms": e2e_ms,
            "pp_tps": pp_tps,
            "tg_tps": tg_tps,
            "kv_hit_rate_pct": kv_hit_rate_pct,
            "status": status,
        }

        with _LOCK:
            _RING_BUFFER.append(record)

        # Update Prometheus metrics
        if pp_tps is not None and PROM_LAST_PP_TPS:
            PROM_LAST_PP_TPS.set(pp_tps)
        if tg_tps is not None and PROM_LAST_TG_TPS:
            PROM_LAST_TG_TPS.set(tg_tps)
        if kv_hit_rate_pct is not None and PROM_LAST_KV_HIT_RATE:
            PROM_LAST_KV_HIT_RATE.set(kv_hit_rate_pct)
        if PROM_REQS_BY_PRIORITY:
            prio_tier = "vip" if ctx.priority < -5 else ("high" if ctx.priority < 0 else ("normal" if ctx.priority == 0 else "low"))
            PROM_REQS_BY_PRIORITY.labels(priority=prio_tier).inc()
        if ctx.agent and PROM_REQS_BY_AGENT:
            PROM_REQS_BY_AGENT.labels(agent=ctx.agent).inc()
    except Exception as e:
        logger.debug("Error recording kv-offload request: %s", e)


async def wrap_streaming_generator(generator: AsyncIterator[str], ctx: RequestTrackerContext) -> AsyncIterator[str]:
    abort_event = asyncio.Event()
    with _LOCK:
        _ACTIVE_STREAMS.add(abort_event)

    try:
        async for chunk in generator:
            if abort_event.is_set():
                logger.info("Active stream for req %s received reset abort signal; closing cleanly", ctx.id)
                # Graceful termination message according to OpenAI SSE standard
                close_payload = {
                    "id": ctx.id or "chatcmpl-reset",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "qwen3.8",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "\n\n[KV Cache Reset: sesión finalizada ordenadamente por el servidor]"},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(close_payload)}\n\n"
                yield "data: [DONE]\n\n"
                finish_request(ctx, None, status=499)
                return

            observe_chunk(ctx, chunk)
            yield chunk
    except Exception as e:
        finish_request(ctx, None, status=500)
        raise
    else:
        finish_request(ctx, None, status=200)
    finally:
        with _LOCK:
            _ACTIVE_STREAMS.discard(abort_event)


def get_records() -> list[dict[str, Any]]:
    with _LOCK:
        return list(_RING_BUFFER)


@router.get("/v1/kv-offload/requests")
@router.get("/kv-offload/requests")
async def get_kv_offload_requests():
    return JSONResponse(content=get_records())


@router.post("/v1/kv-offload/reset")
@router.post("/kv-offload/reset")
@router.post("/reset_prefix_cache")
async def reset_kv_cache_and_metrics(
    raw_request: Request,
    force: bool = Query(default=True, description="Force preemption and graceful notification of active requests"),
    notify_clients: bool = Query(default=True, description="Send [DONE] termination signal to active SSE clients"),
    clear_l1: bool = Query(default=True, description="Clear GPU VRAM L1 prefix cache"),
    clear_l2: bool = Query(default=True, description="Clear Host RAM L2 ARC offload cache"),
    clear_l3: bool = Query(default=True, description="Clear NVMe SSD L3 persistent cache"),
    clear_metrics: bool = Query(default=True, description="Reset Prometheus metrics to 0"),
    clear_history: bool = Query(default=True, description="Reset recent request activity ring buffer"),
):
    """Safely reset KV Cache across L1, L2, L3 tiers, notify open connections, and zero Prometheus metrics in-place."""
    logger.info(
        "KV Offload Reset requested (force=%s, notify_clients=%s, L1=%s, L2=%s, L3=%s, metrics=%s, history=%s)",
        force,
        notify_clients,
        clear_l1,
        clear_l2,
        clear_l3,
        clear_metrics,
        clear_history,
    )

    # 1. Notify and terminate all active SSE streaming client connections gracefully
    notified_clients_count = 0
    if notify_clients:
        with _LOCK:
            notified_clients_count = len(_ACTIVE_STREAMS)
            for ev in _ACTIVE_STREAMS:
                ev.set()

    # Small async yield to allow active generators to flush their [DONE] chunks
    if notified_clients_count > 0:
        await asyncio.sleep(0.05)

    report = {
        "active_clients_notified": notified_clients_count,
        "l1_vram_prefix_cache": False,
        "l2_ram_arc_cache": False,
        "l3_disk_nvme_cache": False,
        "prometheus_metrics": False,
        "request_history": False,
    }

    # 2. Reset L1 VRAM & L2 RAM through vLLM Engine Client
    try:
        engine_client = getattr(raw_request.app.state, "engine_client", None)
        if engine_client is None:
            chat_serving = getattr(raw_request.app.state, "openai_serving_chat", None)
            if chat_serving is not None:
                engine_client = getattr(chat_serving, "engine_client", None)

        if not clear_l1 and not clear_l2:
            # Ni L1 ni L2: no hay nada que pedirle al engine. Antes se llamaba
            # igual y se reportaba exito segun `success`, ignorando los flags —
            # o sea que "resetear solo metricas" intentaba borrar la cache de la
            # GPU. En la practica fallaba sola porque ese boton manda
            # force=false, pero era una bomba con la mecha corta.
            logger.debug("Reset sin clear_l1 ni clear_l2: no se toca el engine")
        elif engine_client is not None and hasattr(engine_client, "reset_prefix_cache"):
            success = await engine_client.reset_prefix_cache(
                reset_running_requests=force,
                reset_connector=clear_l2,
            )
            # `reset_prefix_cache` resetea L1 SIEMPRE que se la llama; no acepta
            # un flag para saltearlo. Asi que solo se la invoca si se pidio al
            # menos uno de los dos, y se reporta lo que de verdad se toco.
            report["l1_vram_prefix_cache"] = bool(success)
            report["l2_ram_arc_cache"] = bool(success and clear_l2)
        else:
            logger.warning("engine_client not found on app.state for L1/L2 reset")
    except Exception as e:
        logger.error("Error resetting L1/L2 cache: %s", e)
        if not force:
            return JSONResponse(
                status_code=409,
                content={"error": f"Failed to reset prefix cache: {e}. Active requests may be running. Try with ?force=true."},
            )

    # 3. Reset L3 NVMe Disk
    if clear_l3:
        try:
            disk_dirs = glob.glob("/kv-offload/*")
            for d in disk_dirs:
                if os.path.isdir(d):
                    shutil.rmtree(d, ignore_errors=True)
                elif os.path.isfile(d):
                    os.remove(d)
            report["l3_disk_nvme_cache"] = True
        except Exception as e:
            logger.error("Error clearing L3 disk cache: %s", e)

    # 4. Reset Prometheus metrics in-place without destroying metric structures
    if clear_metrics:
        try:
            from prometheus_client import REGISTRY

            for collector in list(REGISTRY._collector_to_names):
                if hasattr(collector, "_metrics"):
                    try:
                        for child in list(collector._metrics.values()):
                            if hasattr(child, "_value"):
                                try:
                                    child._value.set(0.0)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                elif hasattr(collector, "_value"):
                    try:
                        collector._value.set(0.0)
                    except Exception:
                        pass

            try:
                from vllm._genesis import kv_tier_metrics as _g88
                _g88.sink().drain()
            except Exception:
                pass

            report["prometheus_metrics"] = True
        except Exception as e:
            logger.error("Error clearing prometheus metrics: %s", e)

    # 5. Reset Request Ring Buffer History
    if clear_history:
        try:
            with _LOCK:
                _RING_BUFFER.clear()
            report["request_history"] = True
        except Exception as e:
            logger.error("Error clearing ring buffer history: %s", e)

    return JSONResponse(
        content={
            "status": "success",
            "cleared": report,
            "timestamp": round(time.time(), 3),
        }
    )


@router.get("/v1/genesis/pid")
@router.get("/v1/kv-offload/pid")
async def get_pid_status_endpoint():
    """Returns current Genesis PN115 PID admission tuner status and metrics."""
    try:
        from vllm._genesis import dynamic_pid_gating as _g115
        return JSONResponse(content=_g115.get_pid_status())
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/v1/genesis/pid")
@router.post("/v1/kv-offload/pid")
async def update_pid_config_endpoint(
    raw_request: Request,
    enabled: Optional[bool] = Query(None),
    target_step_ms: Optional[float] = Query(None),
    max_kv_tokens: Optional[int] = Query(None),
    max_concurrency: Optional[int] = Query(None),
    min_concurrency: Optional[int] = Query(None),
):
    """Dynamically update Genesis PN115 PID admission tuner configuration in live execution."""
    updates: dict[str, Any] = {}
    if enabled is not None:
        updates["enabled"] = enabled
    if target_step_ms is not None:
        updates["target_step_ms"] = target_step_ms
    if max_kv_tokens is not None:
        updates["max_kv_tokens"] = max_kv_tokens
    if max_concurrency is not None:
        updates["max_concurrency"] = max_concurrency
    if min_concurrency is not None:
        updates["min_concurrency"] = min_concurrency

    try:
        body = await raw_request.json()
        if isinstance(body, dict):
            updates.update(body)
    except Exception:
        pass

    try:
        from vllm._genesis import dynamic_pid_gating as _g115
        res = _g115.set_pid_config(updates)
        return JSONResponse(content={"status": "success", "config": res})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

