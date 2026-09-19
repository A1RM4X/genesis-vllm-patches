#!/usr/bin/env python3
"""Benchmark and verification script: Testing 80% KV cache saturation and PID gating.

Sends concurrent streaming requests with deep context (~54,000 tokens each)
to achieve ~80% KV cache allocation in GPU VRAM, measuring:
1. Exact KV cache usage % from /metrics
2. Prefill (PP) tokens/sec & TTFT
3. Text generation (TG) tokens/sec across all streams under saturation
4. PID admission gating and emergency priority bypass behavior
"""

import asyncio
import json
import time
import urllib.request
import aiohttp
import os

BASE_URL = "http://127.0.0.1:8320"
API_KEY = os.environ.get("VLLM_API_KEY", "")

def generate_prompt(req_idx: int, shared_tokens: int = 15000, unique_tokens: int = 35000) -> str:
    # Shared system context (~15,000 tokens)
    shared_text = (
        "Genesis Engine Multi-Agent Cognitive Framework Architecture Document. "
        "Section Alpha: Sublinear attention routing and hierarchical KV cache tiering. "
        "Each agent evaluates state transitions using high-dimensional projective geometry. "
    )
    shared_repeats = max(1, int((shared_tokens * 7.2) / len(shared_text)))
    prefix = shared_text * shared_repeats

    # Unique request context (~35,000 tokens)
    unique_text = (
        f"[Agent Worker Thread #{req_idx}] Specialized task execution block. "
        f"Processing partition {req_idx} differential equations and stochastic gradient flows. "
        f"Entropy estimation for subspace {req_idx} demonstrates fast convergence. "
    )
    unique_repeats = max(1, int((unique_tokens * 7.2) / len(unique_text)))
    suffix = unique_text * unique_repeats

    return prefix + suffix

def fetch_pid_status():
    try:
        req = urllib.request.Request(
            f"{BASE_URL}/v1/genesis/pid",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}

def fetch_metrics():
    try:
        req = urllib.request.Request(
            f"{BASE_URL}/metrics",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            lines = resp.read().decode().splitlines()
            metrics = {}
            for line in lines:
                if line.startswith("vllm:kv_cache_usage_perc"):
                    metrics["kv_perc"] = float(line.split()[-1])
                elif line.startswith("vllm:num_requests_running"):
                    metrics["running"] = float(line.split()[-1])
                elif line.startswith("vllm:num_requests_waiting"):
                    metrics["waiting"] = float(line.split()[-1])
                elif line.startswith("vllm:pid_ema_step_time_ms"):
                    metrics["step_ms"] = float(line.split()[-1])
                elif line.startswith("vllm:pid_active_kv_tokens"):
                    metrics["kv_tokens"] = float(line.split()[-1])
                elif line.startswith("vllm:request_last_tg_tokens_per_second"):
                    metrics["last_tg_tps"] = float(line.split()[-1])
            return metrics
    except Exception as e:
        return {"error": str(e)}

async def run_single_request(
    session: aiohttp.ClientSession,
    req_idx: int,
    prompt: str,
    priority: int,
    agent_name: str,
    max_tokens: int = 150,
    results_list: list = None
):
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {
        "model": "qwen3.8",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "priority": priority,
        "kv_transfer_params": {"genesis_agent": agent_name}
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    t0 = time.perf_counter()
    ttft = None
    output_tokens = 0
    prompt_tokens = 0
    cached_tokens = 0
    status_code = 200

    try:
        async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=900)) as resp:
            status_code = resp.status
            async for line in resp.content:
                decoded = line.decode('utf-8', 'replace').strip()
                if not decoded.startswith("data:"):
                    continue
                data_str = decoded[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        if delta.get("content") or delta.get("reasoning_content"):
                            output_tokens += 1
                    usage = chunk.get("usage")
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                        output_tokens = usage.get("completion_tokens", output_tokens)
                        det = usage.get("prompt_tokens_details") or {}
                        cached_tokens = det.get("cached_tokens", cached_tokens)
                except Exception:
                    pass
    except Exception as e:
        print(f"[{agent_name} #{req_idx}] Exception: {e}")
        status_code = 500

    e2e = time.perf_counter() - t0
    ttft_ms = round(ttft * 1000.0, 1) if ttft else None
    e2e_ms = round(e2e * 1000.0, 1)
    decode_s = max(0.001, e2e - (ttft or 0.0))
    tg_tps = round(output_tokens / decode_s, 1) if output_tokens > 0 else 0.0
    pp_tps = round(prompt_tokens / (ttft if ttft and ttft > 0 else 1.0), 1) if prompt_tokens > 0 else 0.0

    record = {
        "req_idx": req_idx,
        "agent": agent_name,
        "priority": priority,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "ttft_ms": ttft_ms,
        "e2e_ms": e2e_ms,
        "pp_tps": pp_tps,
        "tg_tps": tg_tps,
        "status": status_code,
    }
    if results_list is not None:
        results_list.append(record)
    print(f"  [Req #{req_idx} - {agent_name} (Prio {priority})] Complete: {prompt_tokens} prompt toks, {output_tokens} gen toks, TTFT: {ttft_ms}ms, PP: {pp_tps} tok/s, TG: {tg_tps} tok/s, Status: {status_code}")
    return record

async def monitor_loop(stop_event: asyncio.Event, peak_metrics: dict):
    print("  [Monitor] Starting KV cache monitor loop...")
    while not stop_event.is_set():
        m = fetch_metrics()
        kv_p = m.get("kv_perc", 0.0) * 100.0
        running = m.get("running", 0)
        waiting = m.get("waiting", 0)
        step_ms = m.get("step_ms", 0.0)
        kv_toks = m.get("kv_tokens", 0)

        if kv_p > peak_metrics.get("max_kv_perc", 0.0):
            peak_metrics["max_kv_perc"] = kv_p
        if running > peak_metrics.get("max_running", 0):
            peak_metrics["max_running"] = running

        print(f"  [Telemetry] Active Running: {int(running)} | Waiting: {int(waiting)} | KV Cache: {kv_p:.1f}% ({int(kv_toks):,} toks) | Step: {step_ms:.1f}ms")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pass

async def main():
    print("=" * 70)
    print(" GENESIS — 80% KV CACHE STRESS TEST & PID ADMISSION VERIFICATION")
    print("=" * 70)

    pid_before = fetch_pid_status()
    print(f"PID Initial Status: {pid_before.get('status')} (enabled={pid_before.get('enabled')}, target={pid_before.get('target_step_ms')}ms, max_kv={pid_before.get('max_kv_tokens')})")

    # 10 requests with ~15k shared tokens + ~35k unique tokens = ~50k tokens each
    # Total unique KV tokens across 10 requests: 15k + (10 * 35k) = 365,000 - 500,000 tokens (~75-80% of KV)
    num_requests = 10
    shared_tokens = 15000
    unique_tokens = 35000
    max_tokens = 80

    print(f"Preparing {num_requests} requests with {shared_tokens} shared + {unique_tokens} unique tokens...")
    prompts = [generate_prompt(i, shared_tokens, unique_tokens) for i in range(num_requests)]
    print("Prompts generated successfully. Starting concurrent execution...")

    stop_event = asyncio.Event()
    peak_metrics = {"max_kv_perc": 0.0, "max_running": 0}
    monitor_task = asyncio.create_task(monitor_loop(stop_event, peak_metrics))

    results = []
    connector = aiohttp.TCPConnector(limit=20)
    t_start = time.perf_counter()
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for i in range(num_requests):
            # Mix priorities: coach (VIP -10), coder (High -5), utility (Normal 0)
            if i < 4:
                prio = -10
                agent = "coach"
            elif i < 8:
                prio = -5
                agent = "coder"
            else:
                prio = 0
                agent = "utility"
            tasks.append(run_single_request(session, i, prompts[i], prio, agent, max_tokens, results))
        
        await asyncio.gather(*tasks)
    t_end = time.perf_counter()

    stop_event.set()
    await monitor_task

    total_wall_s = t_end - t_start
    pid_after = fetch_pid_status()

    print("\n" + "=" * 70)
    print(" BENCHMARK REPORT: 80% KV CACHE TEST")
    print("=" * 70)
    print(f"Max KV Cache Allocation Observed: {peak_metrics['max_kv_perc']:.1f}%")
    print(f"Max Concurrent Running Requests:  {int(peak_metrics['max_running'])}")
    print(f"Total Wall-Clock Time:            {total_wall_s:.1f} s")

    total_prompt = sum(r.get("prompt_tokens", 0) for r in results)
    total_output = sum(r.get("output_tokens", 0) for r in results)
    total_cached = sum(r.get("cached_tokens", 0) for r in results)
    valid_pp = [r["pp_tps"] for r in results if r.get("pp_tps", 0) > 0]
    valid_tg = [r["tg_tps"] for r in results if r.get("tg_tps", 0) > 0]
    valid_ttft = [r["ttft_ms"] for r in results if r.get("ttft_ms") is not None]

    avg_pp_tps = sum(valid_pp) / len(valid_pp) if valid_pp else 0.0
    avg_tg_tps = sum(valid_tg) / len(valid_tg) if valid_tg else 0.0
    avg_ttft = sum(valid_ttft) / len(valid_ttft) if valid_ttft else 0.0

    # Aggregate TG TPS: total completion tokens produced across all streams / elapsed decode duration
    first_ttft_s = min([r["ttft_ms"] / 1000.0 for r in results if r.get("ttft_ms") is not None] or [0.0])
    effective_decode_s = max(0.1, total_wall_s - first_ttft_s)
    aggregate_tg_tps = round(total_output / effective_decode_s, 1)

    print(f"Total Prompt Tokens Processed:    {total_prompt:,}")
    print(f"Total Output Tokens Generated:   {total_output:,}")
    print(f"Total Tokens Cached (Prefix Hit): {total_cached:,}")
    print(f"Average TTFT:                    {avg_ttft:.1f} ms")
    print(f"Average Prefill (PP) Throughput: {avg_pp_tps:.1f} tok/s")
    print(f"Per-Stream Decode (TG) Average:  {avg_tg_tps:.1f} tok/s")
    print(f"AGGREGATE Generation Throughput: {aggregate_tg_tps:.1f} tok/s")
    print("-" * 70)
    print("PID Controller Telemetry:")
    print(f"  Gated Requests Total:      {pid_after.get('gated_requests_total', 0)}")
    print(f"  Priority Bypasses Total:   {pid_after.get('priority_bypass_total', 0)}")
    print(f"  Emergency Preemptions:     {pid_after.get('emergency_preemptions_total', 0)}")
    print(f"  Concurrency Limit Post-Run: {pid_after.get('concurrency_limit')}")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
