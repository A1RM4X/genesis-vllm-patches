#!/usr/bin/env python3
"""
Genesis Dynamic PID Admission Evaluation Suite & Live Telemetry Engine.
Runs controlled workloads:
1. 10K Context Workload (Concurrency & Gating evaluation)
2. 60K Context Workload (KV Cache Saturation evaluation)
Collects high-frequency telemetry every ~150ms:
- Individual and aggregate TG tokens/sec
- Active running vs queued waiting requests
- Step latency vs target 24ms & EMA
- KV Cache token usage vs saturation limit
- GPU 0 & GPU 1 NVML utilization & VRAM
Generates an interactive HTML dashboard and serves it on the local network.
"""

from __future__ import annotations

import collections
import concurrent.futures
import ctypes
import http.server
import json
import os
import socket
import socketserver
import sys
import threading
import time
import urllib.parse
import urllib.request
from typing import Any, Optional

API_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8320/v1/chat/completions")
PID_URL = os.environ.get("PID_URL", "http://127.0.0.1:8320/v1/genesis/pid")
RESET_URL = os.environ.get("RESET_URL", "http://127.0.0.1:8320/v1/kv-offload/reset")
REQS_URL = os.environ.get("REQS_URL", "http://127.0.0.1:8320/v1/kv-offload/requests")
API_KEY = os.environ.get("VLLM_API_KEY", os.environ.get("VLLM_API_KEY", ""))
MODEL = os.environ.get("VLLM_MODEL", "qwen3.8")
SERVER_PORT = int(os.environ.get("REPORT_PORT", "8088"))

# ────────────────── NVML CTYPES INTERFACE (<0.2ms overhead) ──────────────────

class c_nvmlMemory_t(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]

class c_nvmlUtilization_t(ctypes.Structure):
    _fields_ = [
        ("gpu", ctypes.c_uint),
        ("memory", ctypes.c_uint),
    ]

class GPUCollector:
    def __init__(self):
        self.available = False
        try:
            self.nvml = ctypes.CDLL("libnvidia-ml.so.1")
            self.nvml.nvmlInit_v2()
            count = ctypes.c_uint()
            self.nvml.nvmlDeviceGetCount_v2(ctypes.byref(count))
            self.device_count = count.value
            self.handles = []
            for i in range(self.device_count):
                h = ctypes.c_void_p()
                self.nvml.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(i), ctypes.byref(h))
                self.handles.append(h)
            self.available = True
        except Exception as e:
            print(f"[NVML] Notice: NVML direct ctypes not available ({e}), falling back to 0s", file=sys.stderr)

    def read_metrics(self) -> dict[str, Any]:
        if not self.available:
            return {"gpu0_util": 0, "gpu0_mem_mb": 0, "gpu1_util": 0, "gpu1_mem_mb": 0}
        out = {}
        for i, h in enumerate(self.handles[:2]):
            try:
                mem = c_nvmlMemory_t()
                self.nvml.nvmlDeviceGetMemoryInfo(h, ctypes.byref(mem))
                util = c_nvmlUtilization_t()
                self.nvml.nvmlDeviceGetUtilizationRates(h, ctypes.byref(util))
                out[f"gpu{i}_util"] = util.gpu
                out[f"gpu{i}_mem_mb"] = round(mem.used / (1024 * 1024), 1)
                out[f"gpu{i}_mem_total_mb"] = round(mem.total / (1024 * 1024), 1)
            except Exception:
                out[f"gpu{i}_util"] = 0
                out[f"gpu{i}_mem_mb"] = 0
        return out


# ────────────────── HIGH FREQUENCY TELEMETRY SAMPLER ──────────────────

class TelemetrySampler:
    def __init__(self, sample_interval: float = 0.15):
        self.sample_interval = sample_interval
        self.gpu_collector = GPUCollector()
        self.samples: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.token_times = collections.deque()  # (monotonic_time, token_count)
        self.total_tokens_generated = 0
        self.active_running_reqs = 0
        self.active_queued_reqs = 0
        self.t0 = time.time()

    def record_token(self, count: int = 1):
        now = time.monotonic()
        with self.lock:
            self.token_times.append((now, count))
            self.total_tokens_generated += count

    def set_req_counts(self, running: int, queued: int):
        with self.lock:
            self.active_running_reqs = running
            self.active_queued_reqs = queued

    def add_event(self, event_type: str, message: str):
        now_rel = round(time.time() - self.t0, 2)
        with self.lock:
            self.events.append({
                "time_s": now_rel,
                "type": event_type,
                "message": message,
            })
        print(f"[{now_rel:6.2f}s] [EVENT] [{event_type}] {message}", flush=True)

    def _sample_loop(self):
        while self.running:
            t_now = time.time()
            m_now = time.monotonic()
            rel_t = round(t_now - self.t0, 3)

            # 1. Compute rolling instant TG tok/s over the last 0.5s window
            window_start = m_now - 0.5
            with self.lock:
                while self.token_times and self.token_times[0][0] < window_start:
                    self.token_times.popleft()
                toks_in_window = sum(item[1] for item in self.token_times)
                instant_tps = round(toks_in_window / 0.5, 1) if toks_in_window > 0 else 0.0
                cum_toks = self.total_tokens_generated
                running_reqs = self.active_running_reqs
                queued_reqs = self.active_queued_reqs

            # 2. Fetch PID state from vLLM SHM / API
            pid_info = {}
            try:
                req = urllib.request.Request(
                    PID_URL,
                    headers={"Authorization": f"Bearer {API_KEY}"}
                )
                with urllib.request.urlopen(req, timeout=0.1) as resp:
                    pid_info = json.loads(resp.read().decode("utf-8"))
            except Exception:
                pass

            # 3. Read NVML GPU utilization & memory
            gpu_info = self.gpu_collector.read_metrics()

            sample = {
                "t": rel_t,
                "instant_tg_tps": instant_tps,
                "cum_tokens": cum_toks,
                "running_reqs": running_reqs,
                "queued_reqs": queued_reqs,
                "pid_enabled": pid_info.get("enabled", True),
                "pid_status": pid_info.get("status", "active"),
                "concurrency_limit": pid_info.get("concurrency_limit", 10),
                "ema_step_ms": pid_info.get("ema_step_ms", 20.0),
                "target_step_ms": pid_info.get("target_step_ms", 24.0),
                "active_kv_tokens": pid_info.get("active_kv_tokens", 0),
                "max_kv_tokens": pid_info.get("max_kv_tokens", 350000),
                "gated_requests_total": pid_info.get("gated_requests_total", 0),
                "gpu0_util": gpu_info.get("gpu0_util", 0),
                "gpu0_mem_mb": gpu_info.get("gpu0_mem_mb", 0),
                "gpu1_util": gpu_info.get("gpu1_util", 0),
                "gpu1_mem_mb": gpu_info.get("gpu1_mem_mb", 0),
            }

            with self.lock:
                self.samples.append(sample)

            time.sleep(self.sample_interval)

    def start(self):
        self.running = True
        self.t0 = time.time()
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)


# ────────────────── WORKLOAD HELPERS & GENERATORS ──────────────────

def build_context(target_tokens: int) -> str:
    """Builds synthetic multi-module Python codebase context of exact token depth."""
    template = '''
class CacheNode:
    def __init__(self, key: str, val: Any, ttl: float):
        self.key = key
        self.val = val
        self.expire_at = time.time() + ttl
        self.prev = None
        self.next = None

class DistributedMemoryCoordinator:
    """Manages multi-tier KV caches and consensus coordination."""
    def __init__(self, node_id: str, peers: list[str]):
        self.node_id = node_id
        self.peers = peers
        self.lock = threading.RLock()
        self.state = {}
        self.commit_log = []

    def commit(self, tx_id: str, data: bytes) -> bool:
        with self.lock:
            entry = {"tx": tx_id, "data": data, "t": time.time()}
            self.commit_log.append(entry)
            self.state[tx_id] = entry
            return True
'''
    repeats = max(1, target_tokens // 100)
    lines = []
    for i in range(repeats):
        lines.append(f"# Subsystem block {i:04d} - Virtual Memory and Execution Context")
        lines.append(template.replace("CacheNode", f"CacheNode_{i:04d}").replace("DistributedMemoryCoordinator", f"DistributedCoordinator_{i:04d}"))
    return "\n".join(lines)


def update_pid_config(enabled: bool, max_kv_tokens: Optional[int] = None) -> dict[str, Any]:
    url = f"{PID_URL}?enabled={'true' if enabled else 'false'}"
    if max_kv_tokens is not None:
        url += f"&max_kv_tokens={max_kv_tokens}"
    req = urllib.request.Request(
        url,
        method="POST",
        headers={"Authorization": f"Bearer {API_KEY}"}
    )
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


def reset_server_kv_and_metrics():
    req = urllib.request.Request(
        RESET_URL,
        method="POST",
        headers={"Authorization": f"Bearer {API_KEY}"}
    )
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_streaming_client(
    req_idx: int,
    prompt: str,
    max_tokens: int,
    sampler: TelemetrySampler,
    name: str,
    priority: int = 0
) -> dict[str, Any]:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a senior software architect. Analyze and write comprehensive code."},
            {"role": "user", "content": f"Context analysis request:\n{prompt}\n\nPlease generate a thorough implementation based on this context."}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "user": name,
    }

    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        }
    )

    t_submit = time.perf_counter()
    t_first = None
    tokens_received = 0
    usage = {}

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for line in resp:
                line = line.decode("utf-8", "replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                data_str = line[6:]
                try:
                    data = json.loads(data_str)
                except Exception:
                    continue

                if "usage" in data and data["usage"]:
                    usage = data["usage"]

                choices = data.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content") or delta.get("reasoning_content")
                    if piece:
                        if t_first is None:
                            t_first = time.perf_counter()
                        tokens_received += 1
                        sampler.record_token(1)
    except Exception as e:
        return {
            "id": f"req-{req_idx}",
            "name": name,
            "error": str(e),
            "tokens": tokens_received,
            "status": "failed"
        }

    t_end = time.perf_counter()
    ttft = (t_first - t_submit) if t_first else (t_end - t_submit)
    decode_s = (t_end - t_first) if t_first else 0.001
    prompt_tokens = usage.get("prompt_tokens", len(prompt.split()))
    completion_tokens = usage.get("completion_tokens", tokens_received)
    pp_tps = round(prompt_tokens / ttft, 1) if ttft > 0 else 0.0
    tg_tps = round(completion_tokens / decode_s, 1) if decode_s > 0 else 0.0

    return {
        "id": f"req-{req_idx}",
        "name": name,
        "priority": priority,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_ms": round(ttft * 1000.0, 1),
        "decode_s": round(decode_s, 2),
        "e2e_s": round(t_end - t_submit, 2),
        "pp_tps": pp_tps,
        "tg_tps": tg_tps,
        "status": "completed"
    }


# ────────────────── TEST EXECUTION ORCHESTRATION ──────────────────

def run_evaluation_suite() -> dict[str, Any]:
    print("=================================================================", flush=True)
    print("  GENESIS DYNAMIC PID ADMISSION EVALUATION SUITE", flush=True)
    print("=================================================================", flush=True)

    sampler = TelemetrySampler(sample_interval=0.15)
    sampler.start()

    # Pre-warm server & reset caches
    print("[Suite] Initializing and resetting server KV cache...", flush=True)
    reset_server_kv_and_metrics()
    update_pid_config(enabled=True, max_kv_tokens=350000)
    sampler.add_event("SYSTEM", "Clean baseline: PID enabled, KV cache reset")
    time.sleep(2.0)

    results = {
        "scenario_10k_pid_on": [],
        "scenario_10k_pid_off": [],
        "scenario_60k_pid_on": [],
        "scenario_60k_pid_off": [],
    }

    def execute_batch(count: int, prompt: str, max_tokens: int, prefix: str, is_high_prio_first: bool = False):
        futures = []
        active_track = {"running": 0, "queued": count}
        sampler.set_req_counts(active_track["running"], active_track["queued"])

        with concurrent.futures.ThreadPoolExecutor(max_workers=count) as ex:
            def worker(idx):
                prio = -8 if (is_high_prio_first and idx == 0) else 0
                name = f"agent-vip-{idx}" if prio < 0 else f"{prefix}-{idx}"
                # Transition queued -> running
                with sampler.lock:
                    sampler.active_queued_reqs = max(0, sampler.active_queued_reqs - 1)
                    sampler.active_running_reqs += 1
                try:
                    res = run_streaming_client(idx, prompt, max_tokens, sampler, name, prio)
                finally:
                    with sampler.lock:
                        sampler.active_running_reqs = max(0, sampler.active_running_reqs - 1)
                return res

            futures = [ex.submit(worker, i) for i in range(count)]
            res_list = [f.result() for f in futures]
        sampler.set_req_counts(0, 0)
        return res_list

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: 10K CONTEXT WORKLOAD (Concurrency & Dynamic Gating)
    # ══════════════════════════════════════════════════════════════════
    prompt_10k = build_context(10000)
    print(f"\n[Phase 1] 10K Context Workload (~{len(prompt_10k)} characters)", flush=True)

    # Phase 1A: PID ON - Burst of 6 requests (target sweet spot 300-500 tok/s)
    sampler.add_event("PHASE_START", "Phase 1A: 6 parallel requests with 10K context [PID ENABLED]")
    update_pid_config(enabled=True, max_kv_tokens=350000)
    res_10k_on = execute_batch(6, prompt_10k, max_tokens=160, prefix="10k-on")
    results["scenario_10k_pid_on"] = res_10k_on
    avg_tg_on = sum(r.get('tg_tps',0) for r in res_10k_on)/len(res_10k_on) if res_10k_on else 0
    sampler.add_event("PHASE_DONE", f"Phase 1A completed: 6 reqs [PID ON], avg TG={avg_tg_on:.1f} tok/s")
    time.sleep(2.0)

    # Phase 1B: LIVE TOGGLE PID OFF mid-test & run with PID DISABLED
    sampler.add_event("PID_TOGGLE", "Toggling PID controller to DISABLED in vivo")
    update_pid_config(enabled=False)
    sampler.add_event("PHASE_START", "Phase 1B: 6 parallel requests with 10K context [PID DISABLED]")
    res_10k_off = execute_batch(6, prompt_10k, max_tokens=160, prefix="10k-off")
    results["scenario_10k_pid_off"] = res_10k_off
    avg_tg_off = sum(r.get('tg_tps',0) for r in res_10k_off)/len(res_10k_off) if res_10k_off else 0
    sampler.add_event("PHASE_DONE", f"Phase 1B completed: 6 reqs [PID OFF], avg TG={avg_tg_off:.1f} tok/s")
    time.sleep(2.0)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: 60K CONTEXT WORKLOAD (KV Cache Saturation & Bus Contention)
    # ══════════════════════════════════════════════════════════════════
    prompt_60k = build_context(60000)
    print(f"\n[Phase 2] 60K Context Workload (~{len(prompt_60k)} characters)", flush=True)

    # Phase 2A: PID ON - Saturating KV Cache with PID Protection (KV Budget = 150k)
    sampler.add_event("PID_TOGGLE", "Toggling PID controller back to ENABLED in vivo (KV threshold=150K)")
    update_pid_config(enabled=True, max_kv_tokens=150000)
    sampler.add_event("PHASE_START", "Phase 2A: 3 parallel requests with 60K context (180K tokens total) [PID ENABLED]")
    res_60k_on = execute_batch(3, prompt_60k, max_tokens=128, prefix="60k-on", is_high_prio_first=True)
    results["scenario_60k_pid_on"] = res_60k_on
    avg_60k_on = sum(r.get('tg_tps',0) for r in res_60k_on)/len(res_60k_on) if res_60k_on else 0
    sampler.add_event("PHASE_DONE", f"Phase 2A completed: 3 reqs [PID ON], avg TG={avg_60k_on:.1f} tok/s")
    time.sleep(2.0)

    # Phase 2B: PID OFF - Saturating KV Cache WITHOUT PID Admission Control
    sampler.add_event("PID_TOGGLE", "Toggling PID controller to DISABLED in vivo under 60K load")
    update_pid_config(enabled=False)
    sampler.add_event("PHASE_START", "Phase 2B: 3 parallel requests with 60K context [PID DISABLED]")
    res_60k_off = execute_batch(3, prompt_60k, max_tokens=128, prefix="60k-off")
    results["scenario_60k_pid_off"] = res_60k_off
    avg_60k_off = sum(r.get('tg_tps',0) for r in res_60k_off)/len(res_60k_off) if res_60k_off else 0
    sampler.add_event("PHASE_DONE", f"Phase 2B completed: 3 reqs [PID OFF], avg TG={avg_60k_off:.1f} tok/s")
    time.sleep(2.0)

    # Re-enable PID for clean standby state
    update_pid_config(enabled=True, max_kv_tokens=350000)
    sampler.add_event("SYSTEM", "Evaluation suite finished. Restored PID to ENABLED standby (350K limit).")
    sampler.stop()

    return {
        "results": results,
        "samples": sampler.samples,
        "events": sampler.events,
    }


# ────────────────── HTML DASHBOARD COMPILER ──────────────────

def generate_html_report(data: dict[str, Any]) -> str:
    samples = data["samples"]
    events = data["events"]
    results = data["results"]

    # Pre-process time-series data for Chart.js
    labels = [round(s["t"], 1) for s in samples]
    tg_tps_data = [s["instant_tg_tps"] for s in samples]
    running_reqs_data = [s["running_reqs"] for s in samples]
    queued_reqs_data = [s["queued_reqs"] for s in samples]
    concurrency_limit_data = [s["concurrency_limit"] for s in samples]
    step_ms_data = [s["ema_step_ms"] for s in samples]
    target_step_data = [s["target_step_ms"] for s in samples]
    active_kv_data = [s["active_kv_tokens"] for s in samples]
    max_kv_data = [s["max_kv_tokens"] for s in samples]
    gpu0_util_data = [s["gpu0_util"] for s in samples]
    gpu1_util_data = [s["gpu1_util"] for s in samples]
    gpu0_mem_data = [s["gpu0_mem_mb"] for s in samples]

    def calc_stats(req_list):
        if not req_list:
            return {"avg_tg": 0, "min_tg": 0, "max_tg": 0, "avg_ttft": 0, "total_toks": 0}
        tgs = [r.get("tg_tps", 0) for r in req_list if r.get("tg_tps")]
        ttfts = [r.get("ttft_ms", 0) for r in req_list if r.get("ttft_ms")]
        return {
            "avg_tg": round(sum(tgs) / len(tgs), 1) if tgs else 0,
            "min_tg": round(min(tgs), 1) if tgs else 0,
            "max_tg": round(max(tgs), 1) if tgs else 0,
            "avg_ttft": round(sum(ttfts) / len(ttfts), 1) if ttfts else 0,
            "total_toks": sum(r.get("completion_tokens", 0) for r in req_list),
        }

    s_10k_on = calc_stats(results["scenario_10k_pid_on"])
    s_10k_off = calc_stats(results["scenario_10k_pid_off"])
    s_60k_on = calc_stats(results["scenario_60k_pid_on"])
    s_60k_off = calc_stats(results["scenario_60k_pid_off"])

    all_reqs = (
        results["scenario_10k_pid_on"] +
        results["scenario_10k_pid_off"] +
        results["scenario_60k_pid_on"] +
        results["scenario_60k_pid_off"]
    )

    reqs_rows = ""
    for r in all_reqs:
        prio_badge = f'<span class="badge badge-vip">VIP ({r.get("priority",0)})</span>' if r.get("priority", 0) < 0 else f'<span class="badge badge-normal">Normal</span>'
        reqs_rows += f"""
        <tr>
            <td class="mono">{r.get('id', '-')}</td>
            <td><strong>{r.get('name', '-')}</strong></td>
            <td>{prio_badge}</td>
            <td class="mono">{r.get('prompt_tokens', 0):,}</td>
            <td class="mono"><strong>{r.get('completion_tokens', 0):,}</strong></td>
            <td class="mono">{r.get('ttft_ms', 0):.1f} ms</td>
            <td class="mono">{r.get('pp_tps', 0):.1f}</td>
            <td class="mono tg-cell"><strong>{r.get('tg_tps', 0):.1f} tok/s</strong></td>
            <td><span class="badge badge-ok">{r.get('status', 'OK')}</span></td>
        </tr>
        """

    events_rows = ""
    for e in events:
        badge_cls = "badge-info"
        if "TOGGLE" in e["type"]:
            badge_cls = "badge-warn"
        elif "DONE" in e["type"]:
            badge_cls = "badge-ok"
        events_rows += f"""
        <tr>
            <td class="mono">{e['time_s']:.2f}s</td>
            <td><span class="badge {badge_cls}">{e['type']}</span></td>
            <td>{e['message']}</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Genesis vLLM — Dynamic PID Admission & Throughput Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #0b0f19;
            --surface: #111827;
            --surface-card: #162032;
            --border: #1f293d;
            --border-highlight: #2d3f5e;
            --text: #f3f4f6;
            --text-dim: #9ca3af;
            --accent: #3b82f6;
            --accent-glow: rgba(59, 130, 246, 0.25);
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --purple: #8b5cf6;
            --cyan: #06b6d4;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            line-height: 1.5;
            padding: 24px;
            font-size: 14px;
        }}
        .container {{ max-width: 1440px; margin: 0 auto; }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 20px;
            margin-bottom: 24px;
        }}
        .logo-group h1 {{
            font-size: 24px;
            font-weight: 800;
            letter-spacing: -0.02em;
            background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .logo-group p {{ color: var(--text-dim); font-size: 13px; margin-top: 4px; }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .badge-ok {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }}
        .badge-warn {{ background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }}
        .badge-info {{ background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }}
        .badge-vip {{ background: rgba(139, 92, 246, 0.2); color: #c084fc; border: 1px solid rgba(139, 92, 246, 0.4); }}
        .badge-normal {{ background: rgba(156, 163, 175, 0.15); color: #d1d5db; border: 1px solid rgba(156, 163, 175, 0.3); }}

        .grid-scorecards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .card {{
            background: var(--surface-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 18px;
            transition: transform 0.15s ease, border-color 0.15s ease;
        }}
        .card:hover {{ border-color: var(--border-highlight); }}
        .card-header {{ font-size: 12px; font-weight: 600; color: var(--text-dim); text-transform: uppercase; margin-bottom: 8px; display: flex; justify-content: space-between; }}
        .metric-val {{ font-size: 28px; font-weight: 800; font-family: 'JetBrains Mono', monospace; }}
        .metric-sub {{ font-size: 12px; color: var(--text-dim); margin-top: 6px; }}
        .highlight-tg {{ color: #34d399; }}
        .highlight-warn {{ color: #fbbf24; }}
        .highlight-pid {{ color: #60a5fa; }}

        .charts-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 24px;
        }}
        .chart-box {{
            background: var(--surface-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 18px;
        }}
        .chart-box.full-width {{ grid-column: span 2; }}
        .chart-title {{
            font-size: 14px;
            font-weight: 700;
            margin-bottom: 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .chart-container {{ position: relative; height: 260px; width: 100%; }}
        .chart-container.tall {{ height: 320px; }}

        .table-box {{
            background: var(--surface-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 13px;
        }}
        th {{
            border-bottom: 1px solid var(--border);
            padding: 10px 14px;
            font-weight: 600;
            color: var(--text-dim);
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        td {{
            border-bottom: 1px solid rgba(31, 41, 61, 0.5);
            padding: 10px 14px;
        }}
        tr:hover td {{ background: rgba(255, 255, 255, 0.02); }}
        .mono {{ font-family: 'JetBrains Mono', monospace; }}
        .tg-cell {{ color: #34d399; }}

        footer {{
            border-top: 1px solid var(--border);
            padding-top: 16px;
            text-align: center;
            color: var(--text-dim);
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo-group">
                <h1>Genesis vLLM — Dynamic PID Throughput & Gating Evaluation</h1>
                <p>Live Telemetry (150ms sampling) • AWQ W4A16 ASYM TP=2 • Qwen3.8-27B Uncensored MTP</p>
            </div>
            <div>
                <span class="badge badge-ok">Telemetry Active</span>
                <span class="badge badge-info">{len(samples)} Samples Recorded</span>
            </div>
        </header>

        <!-- SCORECARDS -->
        <div class="grid-scorecards">
            <div class="card">
                <div class="card-header">10K Context (PID ON) <span>Sweet Spot Target</span></div>
                <div class="metric-val highlight-tg">{s_10k_on['avg_tg']} <span style="font-size:16px;">tok/s</span></div>
                <div class="metric-sub">Min: {s_10k_on['min_tg']} tok/s • Max: {s_10k_on['max_tg']} tok/s • Total: {s_10k_on['total_toks']} toks</div>
            </div>
            <div class="card">
                <div class="card-header">10K Context (PID OFF) <span>Unregulated</span></div>
                <div class="metric-val highlight-warn">{s_10k_off['avg_tg']} <span style="font-size:16px;">tok/s</span></div>
                <div class="metric-sub">Min: {s_10k_off['min_tg']} tok/s • Max: {s_10k_off['max_tg']} tok/s • Total: {s_10k_off['total_toks']} toks</div>
            </div>
            <div class="card">
                <div class="card-header">60K Context (PID ON) <span>KV Cache Protected</span></div>
                <div class="metric-val highlight-tg">{s_60k_on['avg_tg']} <span style="font-size:16px;">tok/s</span></div>
                <div class="metric-sub">VIP Bypass Admitted • TTFT Avg: {s_60k_on['avg_ttft']/1000:.2f}s</div>
            </div>
            <div class="card">
                <div class="card-header">60K Context (PID OFF) <span>KV Saturated</span></div>
                <div class="metric-val highlight-warn">{s_60k_off['avg_tg']} <span style="font-size:16px;">tok/s</span></div>
                <div class="metric-sub">Bus contention / thrashing zone • TTFT Avg: {s_60k_off['avg_ttft']/1000:.2f}s</div>
            </div>
        </div>

        <!-- CHARTS -->
        <div class="charts-grid">
            <div class="chart-box full-width">
                <div class="chart-title">
                    <span>Text Generation Throughput (Instantaneous tok/s across all active streams)</span>
                    <span class="badge badge-info">300 - 500 tok/s Target Zone</span>
                </div>
                <div class="chart-container tall">
                    <canvas id="chartTps"></canvas>
                </div>
            </div>

            <div class="chart-box">
                <div class="chart-title">
                    <span>Concurrency Admission: Running Requests vs Queued (Waiting)</span>
                    <span class="badge badge-ok">PID Gating</span>
                </div>
                <div class="chart-container">
                    <canvas id="chartConcurrency"></canvas>
                </div>
            </div>

            <div class="chart-box">
                <div class="chart-title">
                    <span>Forward Pass Latency: Step EMA (ms) vs Target (24.0ms)</span>
                    <span class="badge badge-warn">Bus Health</span>
                </div>
                <div class="chart-container">
                    <canvas id="chartLatency"></canvas>
                </div>
            </div>

            <div class="chart-box">
                <div class="chart-title">
                    <span>Active KV Cache Tokens vs Budget Limit</span>
                    <span class="badge badge-info">VRAM Capacity</span>
                </div>
                <div class="chart-container">
                    <canvas id="chartKV"></canvas>
                </div>
            </div>

            <div class="chart-box">
                <div class="chart-title">
                    <span>Dual GPU Utilization & Memory (RTX 3090 TP=2)</span>
                    <span class="badge badge-ok">NVML Telemetry</span>
                </div>
                <div class="chart-container">
                    <canvas id="chartGPU"></canvas>
                </div>
            </div>
        </div>

        <!-- REQUESTS DETAIL TABLE -->
        <div class="table-box">
            <div class="chart-title">Detailed Request Execution Metrics</div>
            <table>
                <thead>
                    <tr>
                        <th>Req ID</th>
                        <th>Client / Agent Name</th>
                        <th>Priority Tier</th>
                        <th>Prompt Tokens</th>
                        <th>Generated Tokens</th>
                        <th>TTFT</th>
                        <th>Prefill (PP TPS)</th>
                        <th>Text Gen (TG TPS)</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {reqs_rows}
                </tbody>
            </table>
        </div>

        <!-- EVENTS LOG -->
        <div class="table-box">
            <div class="chart-title">Live Dynamic Control & Toggle Events Timeline</div>
            <table>
                <thead>
                    <tr>
                        <th style="width: 100px;">Timeline</th>
                        <th style="width: 160px;">Event Class</th>
                        <th>Event Description / Status Change</th>
                    </tr>
                </thead>
                <tbody>
                    {events_rows}
                </tbody>
            </table>
        </div>

        <footer>
            Genesis vLLM Architecture & Performance Suite • Real-time Evaluation Report
        </footer>
    </div>

    <script>
        const labels = {json.dumps(labels)};
        const tpsData = {json.dumps(tg_tps_data)};
        const runningData = {json.dumps(running_reqs_data)};
        const queuedData = {json.dumps(queued_reqs_data)};
        const limitData = {json.dumps(concurrency_limit_data)};
        const stepMsData = {json.dumps(step_ms_data)};
        const targetStepData = {json.dumps(target_step_data)};
        const kvData = {json.dumps(active_kv_data)};
        const maxKvData = {json.dumps(max_kv_data)};
        const gpu0Util = {json.dumps(gpu0_util_data)};
        const gpu1Util = {json.dumps(gpu1_util_data)};

        const chartColors = {{
            blue: '#3b82f6',
            green: '#10b981',
            yellow: '#f59e0b',
            purple: '#8b5cf6',
            red: '#ef4444',
            cyan: '#06b6d4',
            grid: '#1f293d',
            text: '#9ca3af'
        }};

        Chart.defaults.color = chartColors.text;
        Chart.defaults.borderColor = chartColors.grid;
        Chart.defaults.font.family = "'Inter', sans-serif";

        // Chart 1: Instantaneous TPS
        new Chart(document.getElementById('chartTps'), {{
            type: 'line',
            data: {{
                labels: labels,
                datasets: [
                    {{
                        label: 'Instantaneous Text Generation (tok/s)',
                        data: tpsData,
                        borderColor: chartColors.green,
                        backgroundColor: 'rgba(16, 185, 129, 0.12)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.25,
                        pointRadius: 0
                    }},
                    {{
                        label: 'Target Sweet Spot Floor (300 tok/s)',
                        data: Array(labels.length).fill(300),
                        borderColor: 'rgba(59, 130, 246, 0.5)',
                        borderDash: [5, 5],
                        borderWidth: 1.5,
                        pointRadius: 0
                    }},
                    {{
                        label: 'Target Sweet Spot Ceiling (500 tok/s)',
                        data: Array(labels.length).fill(500),
                        borderColor: 'rgba(139, 92, 246, 0.5)',
                        borderDash: [5, 5],
                        borderWidth: 1.5,
                        pointRadius: 0
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{ title: {{ display: true, text: 'Elapsed Time (seconds)' }} }},
                    y: {{ title: {{ display: true, text: 'Tokens / Second' }}, beginAtZero: true }}
                }}
            }}
        }});

        // Chart 2: Concurrency & Queue
        new Chart(document.getElementById('chartConcurrency'), {{
            type: 'line',
            data: {{
                labels: labels,
                datasets: [
                    {{
                        label: 'Active Running Requests',
                        data: runningData,
                        borderColor: chartColors.blue,
                        backgroundColor: 'rgba(59, 130, 246, 0.2)',
                        fill: true,
                        tension: 0.1,
                        pointRadius: 0
                    }},
                    {{
                        label: 'Queued (Waiting Gated)',
                        data: queuedData,
                        borderColor: chartColors.yellow,
                        backgroundColor: 'rgba(245, 158, 11, 0.15)',
                        fill: true,
                        tension: 0.1,
                        pointRadius: 0
                    }},
                    {{
                        label: 'Dynamic Admission Limit',
                        data: limitData,
                        borderColor: chartColors.purple,
                        borderDash: [4, 4],
                        borderWidth: 1.5,
                        pointRadius: 0
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{ y: {{ beginAtZero: true, ticks: {{ stepSize: 1 }} }} }}
            }}
        }});

        // Chart 3: Latency & Target
        new Chart(document.getElementById('chartLatency'), {{
            type: 'line',
            data: {{
                labels: labels,
                datasets: [
                    {{
                        label: 'Step Latency EMA (ms)',
                        data: stepMsData,
                        borderColor: chartColors.yellow,
                        borderWidth: 2,
                        pointRadius: 0
                    }},
                    {{
                        label: 'Target Step Latency (24.0 ms)',
                        data: targetStepData,
                        borderColor: chartColors.red,
                        borderDash: [4, 4],
                        borderWidth: 1.5,
                        pointRadius: 0
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: 'Latency (ms)' }} }} }}
            }}
        }});

        // Chart 4: KV Cache Tokens
        new Chart(document.getElementById('chartKV'), {{
            type: 'line',
            data: {{
                labels: labels,
                datasets: [
                    {{
                        label: 'Active KV Tokens',
                        data: kvData,
                        borderColor: chartColors.cyan,
                        backgroundColor: 'rgba(6, 182, 212, 0.15)',
                        fill: true,
                        borderWidth: 2,
                        pointRadius: 0
                    }},
                    {{
                        label: 'Active KV Budget Limit',
                        data: maxKvData,
                        borderColor: chartColors.red,
                        borderDash: [5, 5],
                        borderWidth: 1.5,
                        pointRadius: 0
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: 'Tokens' }} }} }}
            }}
        }});

        // Chart 5: Dual GPU
        new Chart(document.getElementById('chartGPU'), {{
            type: 'line',
            data: {{
                labels: labels,
                datasets: [
                    {{
                        label: 'GPU 0 Compute Util (%)',
                        data: gpu0Util,
                        borderColor: chartColors.green,
                        borderWidth: 1.5,
                        pointRadius: 0
                    }},
                    {{
                        label: 'GPU 1 Compute Util (%)',
                        data: gpu1Util,
                        borderColor: chartColors.blue,
                        borderWidth: 1.5,
                        pointRadius: 0
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{ y: {{ beginAtZero: true, max: 100, title: {{ display: true, text: 'Utilization %' }} }} }}
            }}
        }});
    </script>
</body>
</html>
"""
    return html


# ────────────────── LOCAL NETWORK HTTP SERVER ──────────────────

class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

def start_http_server(html_content: str, port: int = 8088):
    html_bytes = html_content.encode("utf-8")

    class ReportHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(html_bytes)

        def log_message(self, format, *args):
            pass

    server = ReusableTCPServer(("0.0.0.0", port), ReportHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    return server


# ────────────────── ENTRYPOINT ──────────────────

if __name__ == "__main__":
    eval_data = run_evaluation_suite()
    html_output = generate_html_report(eval_data)

    out_file = "/tmp/genesis_pid_eval_report.html"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_output)

    print(f"\n[HTML] Report written to {out_file} ({len(html_output)} bytes)", flush=True)

    # Save JSON data as well
    json_file = "/tmp/genesis_pid_eval_data.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({
            "results": eval_data["results"],
            "samples_count": len(eval_data["samples"]),
            "events_count": len(eval_data["events"])
        }, f, indent=2)

    server = start_http_server(html_output, port=SERVER_PORT)
    print(f"[HTTP] Dashboard live and serving on:", flush=True)
    print(f"       http://192.168.1.20:{SERVER_PORT}/", flush=True)
    print(f"       http://127.0.0.1:{SERVER_PORT}/", flush=True)

    # Keep server open
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping server...")
