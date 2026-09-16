#!/usr/bin/env python3
"""
Genesis vLLM — Sustained 10-Minute Continuous Load & PID Evaluation Suite.
Duration: Exactly 600 seconds (10 minutes).
Workloads:
- Phase 1 (0:00 - 5:00): 10K Context Continuous Stream (PID ON for 2.5m -> PID OFF for 2.5m)
- Phase 2 (5:00 - 10:00): 60K Context Continuous Stream (PID ON for 2.5m -> PID OFF for 2.5m)
Telemetry:
- Sampled every 200ms (~3,000 samples)
- Instantaneous Text Generation Throughput (tok/s) across active decode streams
- Queue size (waiting in scheduler vs running)
- Step duration EMA (ms) vs Target 24ms
- KV Cache token usage vs saturation budget
- Dual RTX 3090 NVML utilization & VRAM
Live Web Dashboard:
- Served at http://192.168.1.20:8088/
- Real-time updates every 2 seconds during the 10-minute run.
"""

from __future__ import annotations

import collections
import concurrent.futures
import ctypes
import http.server
import json
import os
import socketserver
import sys
import threading
import time
import urllib.request
from typing import Any, Optional

API_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8320/v1/chat/completions")
PID_URL = os.environ.get("PID_URL", "http://127.0.0.1:8320/v1/genesis/pid")
RESET_URL = os.environ.get("RESET_URL", "http://127.0.0.1:8320/v1/kv-offload/reset")
REQS_URL = os.environ.get("REQS_URL", "http://127.0.0.1:8320/v1/kv-offload/requests")
API_KEY = os.environ.get("VLLM_API_KEY", "<REDACTADO: clave rotada 2026-09-19>")
MODEL = os.environ.get("VLLM_MODEL", "qwen3.8")
SERVER_PORT = int(os.environ.get("REPORT_PORT", "8088"))
TOTAL_TEST_DURATION_SEC = 600  # 10 minutes

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
            print(f"[NVML] Notice: {e}", file=sys.stderr)

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
            except Exception:
                out[f"gpu{i}_util"] = 0
                out[f"gpu{i}_mem_mb"] = 0
        return out


# ────────────────── HIGH FREQUENCY TELEMETRY SAMPLER ──────────────────

class TelemetrySampler:
    def __init__(self, sample_interval: float = 0.20):
        self.sample_interval = sample_interval
        self.gpu_collector = GPUCollector()
        self.samples: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.completed_requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.token_times = collections.deque()  # (monotonic_time, token_count)
        self.total_tokens_generated = 0
        self.active_running_reqs = 0
        self.active_queued_reqs = 0
        self.t0 = time.time()
        self.current_phase_name = "Initializing"

    def record_token(self, count: int = 1):
        now = time.monotonic()
        with self.lock:
            self.token_times.append((now, count))
            self.total_tokens_generated += count

    def set_req_counts(self, running: int, queued: int):
        with self.lock:
            self.active_running_reqs = running
            self.active_queued_reqs = queued

    def add_completed_req(self, req_data: dict[str, Any]):
        with self.lock:
            self.completed_requests.append(req_data)

    def add_event(self, event_type: str, message: str):
        now_rel = round(time.time() - self.t0, 2)
        with self.lock:
            self.events.append({
                "time_s": now_rel,
                "type": event_type,
                "message": message,
            })
            if "PHASE" in event_type or "PID" in event_type:
                self.current_phase_name = message
        print(f"[{now_rel:6.2f}s] [EVENT] [{event_type}] {message}", flush=True)

    def _sample_loop(self):
        while self.running:
            t_now = time.time()
            m_now = time.monotonic()
            rel_t = round(t_now - self.t0, 2)

            # 1. Rolling window of 1.0 second for stable text generation TPS
            window_start = m_now - 1.0
            with self.lock:
                while self.token_times and self.token_times[0][0] < window_start:
                    self.token_times.popleft()
                toks_in_window = sum(item[1] for item in self.token_times)
                instant_tps = round(toks_in_window / 1.0, 1)
                cum_toks = self.total_tokens_generated
                running_reqs = self.active_running_reqs
                queued_reqs = self.active_queued_reqs
                phase = self.current_phase_name

            # 2. Fetch PID state from vLLM API
            pid_info = {}
            try:
                req = urllib.request.Request(
                    PID_URL,
                    headers={"Authorization": f"Bearer {API_KEY}"}
                )
                with urllib.request.urlopen(req, timeout=0.15) as resp:
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
                "phase": phase,
                "pid_enabled": pid_info.get("enabled", True),
                "concurrency_limit": pid_info.get("concurrency_limit", 10),
                "ema_step_ms": pid_info.get("ema_step_ms", 20.0),
                "target_step_ms": pid_info.get("target_step_ms", 24.0),
                "active_kv_tokens": pid_info.get("active_kv_tokens", 0),
                "max_kv_tokens": pid_info.get("max_kv_tokens", 350000),
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

def build_context(target_tokens: int, seed_id: int) -> str:
    """Builds synthetic multi-module Python codebase context with unique entropy per request."""
    template = f'''
# Module Namespace: {seed_id:04d}
class CacheNode_{seed_id:04d}:
    def __init__(self, key: str, val: Any, ttl: float):
        self.key = f"k_{{{seed_id}}}_{{key}}"
        self.val = val
        self.expire_at = time.time() + ttl
        self.prev = None
        self.next = None

class DistributedMemoryCoordinator_{seed_id:04d}:
    """Manages multi-tier KV caches and consensus coordination for node {seed_id}."""
    def __init__(self, node_id: str, peers: list[str]):
        self.node_id = f"node_{{{seed_id}}}_{{node_id}}"
        self.peers = peers
        self.lock = threading.RLock()
        self.state = {{}}
        self.commit_log = []

    def commit(self, tx_id: str, data: bytes) -> bool:
        with self.lock:
            entry = {{"tx": tx_id, "data": data, "t": time.time(), "seed": {seed_id}}}
            self.commit_log.append(entry)
            self.state[tx_id] = entry
            return True
'''
    repeats = max(1, target_tokens // 100)
    lines = [f"# Subsystem namespace {seed_id} - Virtual Memory and Execution Context"]
    for i in range(repeats):
        lines.append(template.replace(f"_{seed_id:04d}", f"_{seed_id:04d}_{i:03d}"))
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
            {"role": "system", "content": "You are a senior software architect. Implement robust, complete algorithms with full docstrings and tests."},
            {"role": "user", "content": f"Context analysis request {req_idx}:\n{prompt}\n\nPlease generate a thorough Python implementation based on this context. Write extensive code."}
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

    record = {
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
    sampler.add_completed_req(record)
    return record


# ────────────────── HTML DASHBOARD COMPILER ──────────────────

def render_html(sampler: TelemetrySampler, is_finished: bool = False) -> str:
    with sampler.lock:
        samples = list(sampler.samples)
        events = list(sampler.events)
        completed = list(sampler.completed_requests)

    # Downsample for browser rendering if > 1500 samples
    step = max(1, len(samples) // 1200)
    display_samples = samples[::step] if step > 1 else samples

    labels = [s["t"] for s in display_samples]
    tps_data = [s["instant_tg_tps"] for s in display_samples]
    running_data = [s["running_reqs"] for s in display_samples]
    queued_data = [s["queued_reqs"] for s in display_samples]
    concurrency_limit_data = [s["concurrency_limit"] for s in display_samples]
    step_ms_data = [s["ema_step_ms"] for s in display_samples]
    target_step_data = [s["target_step_ms"] for s in display_samples]
    active_kv_data = [s["active_kv_tokens"] for s in display_samples]
    max_kv_data = [s["max_kv_tokens"] for s in display_samples]
    gpu0_util_data = [s["gpu0_util"] for s in display_samples]
    gpu1_util_data = [s["gpu1_util"] for s in display_samples]
    gpu0_mem_data = [s["gpu0_mem_mb"] for s in display_samples]

    elapsed = round(time.time() - sampler.t0, 1)
    status_tag = "COMPLETED (10 min benchmark)" if is_finished else f"RUNNING LIVE ({elapsed}s / 600s)"
    status_class = "badge-ok" if is_finished else "badge-live"

    total_toks = sampler.total_tokens_generated
    avg_tps = round(total_toks / max(1.0, elapsed), 1)
    peak_tps = max(tps_data) if tps_data else 0.0

    # Categorize completed requests
    reqs_p1_on = [r for r in completed if "10k-on" in r.get("name", "")]
    reqs_p1_off = [r for r in completed if "10k-off" in r.get("name", "")]
    reqs_p2_on = [r for r in completed if "60k-on" in r.get("name", "") or "vip" in r.get("name", "")]
    reqs_p2_off = [r for r in completed if "60k-off" in r.get("name", "")]

    def avg_tg(reqs):
        vals = [r["tg_tps"] for r in reqs if r.get("tg_tps")]
        return round(sum(vals) / len(vals), 1) if vals else 0.0

    p1_on_tg = avg_tg(reqs_p1_on)
    p1_off_tg = avg_tg(reqs_p1_off)
    p2_on_tg = avg_tg(reqs_p2_on)
    p2_off_tg = avg_tg(reqs_p2_off)

    req_rows = ""
    for r in completed[-30:]:  # Last 30 requests
        prio_badge = f'<span class="badge badge-vip">VIP</span>' if r.get("priority", 0) < 0 else f'<span class="badge badge-normal">Normal</span>'
        req_rows += f"""
        <tr>
            <td class="mono">{r.get('id', '-')}</td>
            <td><strong>{r.get('name', '-')}</strong></td>
            <td>{prio_badge}</td>
            <td class="mono">{r.get('prompt_tokens', 0):,}</td>
            <td class="mono"><strong>{r.get('completion_tokens', 0):,}</strong></td>
            <td class="mono">{r.get('ttft_ms', 0):.1f} ms</td>
            <td class="mono tg-cell"><strong>{r.get('tg_tps', 0):.1f} tok/s</strong></td>
            <td><span class="badge badge-ok">{r.get('status', 'OK')}</span></td>
        </tr>
        """

    event_rows = ""
    for e in events:
        b_class = "badge-warn" if "PID" in e["type"] else ("badge-info" if "PHASE" in e["type"] else "badge-ok")
        event_rows += f"""
        <tr>
            <td class="mono">{e['time_s']:.1f}s</td>
            <td><span class="badge {b_class}">{e['type']}</span></td>
            <td>{e['message']}</td>
        </tr>
        """

    refresh_meta = '<meta http-equiv="refresh" content="3">' if not is_finished else ''

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    {refresh_meta}
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Genesis vLLM — Sustained 10-Minute Dynamic PID Evaluation</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com">
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
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .badge-live {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); animation: pulse 1.5s infinite; }}
        .badge-ok {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }}
        .badge-warn {{ background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }}
        .badge-info {{ background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }}
        .badge-vip {{ background: rgba(139, 92, 246, 0.2); color: #c084fc; border: 1px solid rgba(139, 92, 246, 0.4); }}
        .badge-normal {{ background: rgba(156, 163, 175, 0.15); color: #d1d5db; border: 1px solid rgba(156, 163, 175, 0.3); }}
        @keyframes pulse {{ 0% {{ opacity: 0.7; }} 50% {{ opacity: 1.0; }} 100% {{ opacity: 0.7; }} }}

        .grid-scorecards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .card {{
            background: var(--surface-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 18px;
        }}
        .card-header {{ font-size: 11px; font-weight: 600; color: var(--text-dim); text-transform: uppercase; margin-bottom: 8px; display: flex; justify-content: space-between; }}
        .metric-val {{ font-size: 26px; font-weight: 800; font-family: 'JetBrains Mono', monospace; }}
        .metric-sub {{ font-size: 12px; color: var(--text-dim); margin-top: 6px; }}
        .highlight-tg {{ color: #34d399; }}
        .highlight-warn {{ color: #fbbf24; }}
        .highlight-cyan {{ color: #06b6d4; }}

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
        .chart-container.tall {{ height: 340px; }}

        .table-box {{
            background: var(--surface-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
            overflow-x: auto;
        }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 13px; }}
        th {{ border-bottom: 1px solid var(--border); padding: 10px 14px; font-weight: 600; color: var(--text-dim); font-size: 11px; text-transform: uppercase; }}
        td {{ border-bottom: 1px solid rgba(31, 41, 61, 0.5); padding: 10px 14px; }}
        .mono {{ font-family: 'JetBrains Mono', monospace; }}
        .tg-cell {{ color: #34d399; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo-group">
                <h1>Genesis vLLM — Sustained 10-Minute Dynamic PID Benchmark</h1>
                <p>AWQ W4A16 ASYM TP=2 • Qwen3.8-27B Uncensored MTP • Dual RTX 3090 NVLink</p>
            </div>
            <div>
                <span class="badge {status_class}">{status_tag}</span>
                <span class="badge badge-info">{len(samples)} Samples (200ms)</span>
            </div>
        </header>

        <!-- SCORECARDS -->
        <div class="grid-scorecards">
            <div class="card">
                <div class="card-header">10K Context (PID ON) <span>Sweet Spot</span></div>
                <div class="metric-val highlight-tg">{p1_on_tg} <span style="font-size:15px;">tok/s</span></div>
                <div class="metric-sub">{len(reqs_p1_on)} requests completed in zone</div>
            </div>
            <div class="card">
                <div class="card-header">10K Context (PID OFF) <span>Unregulated</span></div>
                <div class="metric-val highlight-warn">{p1_off_tg} <span style="font-size:15px;">tok/s</span></div>
                <div class="metric-sub">{len(reqs_p1_off)} requests completed in zone</div>
            </div>
            <div class="card">
                <div class="card-header">60K Context (PID ON) <span>KV Managed</span></div>
                <div class="metric-val highlight-tg">{p2_on_tg} <span style="font-size:15px;">tok/s</span></div>
                <div class="metric-sub">{len(reqs_p2_on)} requests completed in zone</div>
            </div>
            <div class="card">
                <div class="card-header">60K Context (PID OFF) <span>KV Saturated</span></div>
                <div class="metric-val highlight-warn">{p2_off_tg} <span style="font-size:15px;">tok/s</span></div>
                <div class="metric-sub">{len(reqs_p2_off)} requests completed in zone</div>
            </div>
            <div class="card">
                <div class="card-header">Aggregate Total Tokens <span>Cumulative</span></div>
                <div class="metric-val highlight-cyan">{total_toks:,}</div>
                <div class="metric-sub">Peak TPS: {peak_tps} tok/s • Avg: {avg_tps} tok/s</div>
            </div>
        </div>

        <!-- CHARTS -->
        <div class="charts-grid">
            <div class="chart-box full-width">
                <div class="chart-title">
                    <span>Continuous Instantaneous Text Generation Throughput (tok/s) across all active decode streams</span>
                    <span class="badge badge-info">Target 300 - 500 tok/s Zone</span>
                </div>
                <div class="chart-container tall">
                    <canvas id="chartTps"></canvas>
                </div>
            </div>

            <div class="chart-box">
                <div class="chart-title">
                    <span>Concurrency Admission: Active Running vs Queued (Waiting)</span>
                    <span class="badge badge-ok">Scheduler State</span>
                </div>
                <div class="chart-container">
                    <canvas id="chartConcurrency"></canvas>
                </div>
            </div>

            <div class="chart-box">
                <div class="chart-title">
                    <span>Step Latency EMA (ms) vs Target (24.0ms)</span>
                    <span class="badge badge-warn">Bus Health</span>
                </div>
                <div class="chart-container">
                    <canvas id="chartLatency"></canvas>
                </div>
            </div>

            <div class="chart-box">
                <div class="chart-title">
                    <span>Active KV Cache Tokens vs Saturation Limit</span>
                    <span class="badge badge-info">VRAM Capacity</span>
                </div>
                <div class="chart-container">
                    <canvas id="chartKV"></canvas>
                </div>
            </div>

            <div class="chart-box">
                <div class="chart-title">
                    <span>Dual GPU Utilization & Memory (RTX 3090 TP=2)</span>
                    <span class="badge badge-ok">Hardware NVML</span>
                </div>
                <div class="chart-container">
                    <canvas id="chartGPU"></canvas>
                </div>
            </div>
        </div>

        <!-- RECENT REQUESTS TABLE -->
        <div class="table-box">
            <div class="chart-title">Recent Completed Requests Log ({len(completed)} total)</div>
            <table>
                <thead>
                    <tr>
                        <th>Req ID</th>
                        <th>Client / Agent Name</th>
                        <th>Priority</th>
                        <th>Prompt Tokens</th>
                        <th>Generated</th>
                        <th>TTFT</th>
                        <th>TG Speed</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {req_rows}
                </tbody>
            </table>
        </div>

        <!-- TIMELINE EVENTS -->
        <div class="table-box">
            <div class="chart-title">Live 10-Minute Dynamic Control Events Timeline</div>
            <table>
                <thead>
                    <tr>
                        <th style="width: 110px;">Timeline</th>
                        <th style="width: 150px;">Event</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
                    {event_rows}
                </tbody>
            </table>
        </div>
    </div>

    <script>
        const labels = {json.dumps(labels)};
        const tpsData = {json.dumps(tps_data)};
        const runningData = {json.dumps(running_data)};
        const queuedData = {json.dumps(queued_data)};
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

        // Chart 1: Continuous TPS
        new Chart(document.getElementById('chartTps'), {{
            type: 'line',
            data: {{
                labels: labels,
                datasets: [
                    {{
                        label: 'Instantaneous Text Generation (tok/s)',
                        data: tpsData,
                        borderColor: chartColors.green,
                        backgroundColor: 'rgba(16, 185, 129, 0.10)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.2,
                        pointRadius: 0
                    }},
                    {{
                        label: 'Target Sweet Spot Floor (300 tok/s)',
                        data: Array(labels.length).fill(300),
                        borderColor: 'rgba(59, 130, 246, 0.4)',
                        borderDash: [5, 5],
                        borderWidth: 1.5,
                        pointRadius: 0
                    }},
                    {{
                        label: 'Target Sweet Spot Ceiling (500 tok/s)',
                        data: Array(labels.length).fill(500),
                        borderColor: 'rgba(139, 92, 246, 0.4)',
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
                        pointRadius: 0
                    }},
                    {{
                        label: 'Queued (Waiting Gated)',
                        data: queuedData,
                        borderColor: chartColors.yellow,
                        backgroundColor: 'rgba(245, 158, 11, 0.15)',
                        fill: true,
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


# ────────────────── LIVE HTTP SERVER ──────────────────

class ReportServer:
    def __init__(self, sampler: TelemetrySampler, port: int = 8088):
        self.sampler = sampler
        self.port = port
        self.is_finished = False

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(h_self):
                html = render_html(self.sampler, is_finished=self.is_finished)
                data = html.encode("utf-8")
                h_self.send_response(200)
                h_self.send_header("Content-Type", "text/html; charset=utf-8")
                h_self.send_header("Content-Length", str(len(data)))
                h_self.send_header("Access-Control-Allow-Origin", "*")
                h_self.end_headers()
                h_self.wfile.write(data)

            def log_message(h_self, format, *args):
                pass

        class TCPServer(socketserver.TCPServer):
            allow_reuse_address = True

        self.server = TCPServer(("0.0.0.0", port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def mark_finished(self):
        self.is_finished = True


# ────────────────── 10-MINUTE SUSTAINED BENCHMARK RUNNER ──────────────────

def run_10_minute_benchmark():
    print("=================================================================", flush=True)
    print("  GENESIS vLLM — SUSTAINED 10-MINUTE DYNAMIC PID EVALUATION", flush=True)
    print("=================================================================", flush=True)

    sampler = TelemetrySampler(sample_interval=0.20)
    sampler.start()

    server = ReportServer(sampler, port=SERVER_PORT)
    print(f"[HTTP] Live Dashboard accessible on:", flush=True)
    print(f"       http://192.168.1.20:{SERVER_PORT}/", flush=True)
    print(f"       http://127.0.0.1:{SERVER_PORT}/", flush=True)

    # Initial reset
    print("[Suite] Initializing baseline: reset KV cache and enable PID...", flush=True)
    reset_server_kv_and_metrics()
    update_pid_config(enabled=True, max_kv_tokens=350000)
    sampler.add_event("SYSTEM", "Clean baseline: PID enabled, KV cache reset")

    t_start = time.time()
    t_end = t_start + TOTAL_TEST_DURATION_SEC

    # Concurrency control
    CONCURRENT_WORKERS = 8
    active_workers = 0
    queued_workers = 0
    worker_lock = threading.Lock()
    req_counter = 0

    # Phase boundaries (in elapsed seconds)
    # Phase 1A: 0 to 150s (10K Context, PID ON)
    # Phase 1B: 150 to 300s (10K Context, PID OFF)
    # Phase 2A: 300 to 450s (60K Context, PID ON, KV limit=150K)
    # Phase 2B: 450 to 600s (60K Context, PID OFF, KV limit=150K)
    t_p1b = t_start + 150
    t_p2a = t_start + 300
    t_p2b = t_start + 450

    p1b_triggered = False
    p2a_triggered = False
    p2b_triggered = False

    sampler.add_event("PHASE_START", "Phase 1A (0:00 - 2:30): 10K Context Continuous Stream [PID ENABLED]")

    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
        futures = set()

        while time.time() < t_end:
            now = time.time()

            # Dynamic timeline transitions
            if now >= t_p1b and not p1b_triggered:
                p1b_triggered = True
                update_pid_config(enabled=False)
                sampler.add_event("PID_TOGGLE", "Phase 1B (2:30 - 5:00): Live Toggle PID DISABLED [10K Context]")

            elif now >= t_p2a and not p2a_triggered:
                p2a_triggered = True
                update_pid_config(enabled=True, max_kv_tokens=150000)
                sampler.add_event("PID_TOGGLE", "Phase 2A (5:00 - 7:30): Live Toggle PID ENABLED [60K Context, KV Budget=150K]")

            elif now >= t_p2b and not p2b_triggered:
                p2b_triggered = True
                update_pid_config(enabled=False)
                sampler.add_event("PID_TOGGLE", "Phase 2B (7:30 - 10:00): Live Toggle PID DISABLED under 60K KV Saturation")

            # Determine workload for new requests
            is_60k = now >= t_p2a
            token_depth = 60000 if is_60k else 10000
            max_tokens = 512

            # Clean completed futures
            done = {f for f in futures if f.done()}
            futures.difference_update(done)

            # Spawn new requests to keep exactly CONCURRENT_WORKERS active
            while len(futures) < CONCURRENT_WORKERS and time.time() < t_end:
                req_counter += 1
                r_id = req_counter
                is_vip = (r_id % 7 == 0)
                prio = -8 if is_vip else 0
                prefix = f"60k-{'on' if not p2b_triggered else 'off'}" if is_60k else f"10k-{'on' if not p1b_triggered else 'off'}"
                name = f"vip-agent-{r_id}" if is_vip else f"{prefix}-c{r_id}"

                prompt = build_context(token_depth, seed_id=r_id)

                def task(idx=r_id, p=prompt, n=name, pr=prio):
                    with worker_lock:
                        sampler.active_running_reqs += 1
                    try:
                        return run_streaming_client(idx, p, max_tokens, sampler, n, pr)
                    finally:
                        with worker_lock:
                            sampler.active_running_reqs = max(0, sampler.active_running_reqs - 1)

                with worker_lock:
                    sampler.active_queued_reqs = max(0, CONCURRENT_WORKERS - len(futures))

                f = executor.submit(task)
                futures.add(f)

            time.sleep(0.5)

        print("[Suite] 10 minutes reached! Draining remaining active requests...", flush=True)
        concurrent.futures.wait(futures, timeout=60.0)

    sampler.add_event("SYSTEM", "10-minute benchmark completed. Restored PID to ENABLED standby.")
    update_pid_config(enabled=True, max_kv_tokens=350000)
    server.mark_finished()
    sampler.stop()

    # Save final HTML and JSON
    final_html = render_html(sampler, is_finished=True)
    with open("/tmp/genesis_pid_10m_report.html", "w", encoding="utf-8") as f:
        f.write(final_html)

    with open("/tmp/genesis_pid_10m_data.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_tokens": sampler.total_tokens_generated,
            "samples_count": len(sampler.samples),
            "completed_count": len(sampler.completed_requests),
            "events": sampler.events,
            "completed_requests": sampler.completed_requests
        }, f, indent=2)

    print(f"\n[HTML] Final 10-minute dashboard written to /tmp/genesis_pid_10m_report.html", flush=True)
    print(f"[HTTP] Server is running and accessible indefinitely on:", flush=True)
    print(f"       http://192.168.1.20:{SERVER_PORT}/", flush=True)
    print(f"       http://127.0.0.1:{SERVER_PORT}/", flush=True)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting server...")


if __name__ == "__main__":
    run_10_minute_benchmark()
