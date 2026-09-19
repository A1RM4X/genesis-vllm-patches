#!/usr/bin/env python3
"""
Verification script: 4 concurrent requests with deep context (~43k tokens each)
under Zero-Hardcoded Auto-Discovered Dynamic PID Gating.
"""

import concurrent.futures
import json
import os
import sys
import threading
import time
import urllib.request

BASE_URL = "http://127.0.0.1:8320"
API_KEY = os.environ.get("VLLM_API_KEY", "")
MODEL = "qwen3.8"


def generate_prompt(req_idx: int, repeats: int = 600) -> str:
    template = f'''
# Partition Module: {req_idx:03d}
class EngineWorkerPipeline_{req_idx:03d}:
    def __init__(self, stage_id: int):
        self.stage_id = stage_id
        self.active_buffer = bytearray(1024 * 64)
        self.entropy_table = [x * 1.618 for x in range(256)]
    
    def process_block(self, block_id: int, payload: bytes) -> bool:
        if len(payload) == 0:
            return False
        return True
'''
    lines = [f"# Genesis Deep Stress Context Task {req_idx}"]
    for i in range(repeats):
        lines.append(template.replace(f"_{req_idx:03d}", f"_{req_idx:03d}_{i:04d}"))
    return "\n".join(lines)


def run_streaming_request(req_idx: int, prompt: str, max_tokens: int, results: list):
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert compiler engineer."},
            {"role": "user", "content": f"Task #{req_idx}:\n{prompt}\n\nPlease generate a clean Python analysis with extensive code."}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "user": f"stress-req-{req_idx}"
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
    )

    t_start = time.perf_counter()
    t_first = None
    tokens_received = 0
    usage = {}

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for line in resp:
                line = line.decode("utf-8", "replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                try:
                    data = json.loads(line[6:])
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
                            print(f"  [Req #{req_idx}] First token received after {t_first - t_start:.2f}s (TTFT)", flush=True)
                        tokens_received += 1
    except Exception as e:
        print(f"  [Req #{req_idx}] Exception: {e}", flush=True)

    t_end = time.perf_counter()
    ttft = (t_first - t_start) if t_first else (t_end - t_start)
    decode_s = (t_end - t_first) if t_first else 0.001
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", tokens_received)
    pp_tps = round(prompt_tokens / ttft, 1) if ttft > 0 else 0.0
    tg_tps = round(completion_tokens / decode_s, 1) if decode_s > 0 else 0.0

    rec = {
        "req_idx": req_idx,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": round(ttft, 2),
        "decode_s": round(decode_s, 2),
        "pp_tps": pp_tps,
        "tg_tps": tg_tps,
    }
    results.append(rec)
    print(f"  [Req #{req_idx}] COMPLETED: Prompt={prompt_tokens:,}, Gen={completion_tokens:,}, TTFT={ttft:.1f}s, PP={pp_tps} tok/s, TG={tg_tps} tok/s", flush=True)
    return rec


def monitor_loop(stop_event: threading.Event):
    t0 = time.time()

    while not stop_event.is_set():
        now = time.time()
        elapsed = now - t0

        try:
            req = urllib.request.Request(f"{BASE_URL}/v1/genesis/pid", headers={"Authorization": f"Bearer {API_KEY}"})
            with urllib.request.urlopen(req, timeout=0.25) as resp:
                pid_d = json.loads(resp.read().decode())
                
            req2 = urllib.request.Request(f"{BASE_URL}/metrics")
            with urllib.request.urlopen(req2, timeout=0.25) as resp2:
                text = resp2.read().decode()
                kv = 0.0
                running = 0
                waiting = 0
                preempts = 0
                for line in text.splitlines():
                    if line.startswith("vllm:kv_cache_usage_perc"):
                        kv = float(line.split()[-1]) * 100.0
                    elif line.startswith("vllm:num_requests_running"):
                        running = int(float(line.split()[-1]))
                    elif line.startswith("vllm:num_requests_waiting"):
                        waiting = int(float(line.split()[-1]))
                    elif line.startswith("vllm:num_preemptions_total"):
                        preempts = int(float(line.split()[-1]))

                step_ms = pid_d.get("ema_step_ms", 0.0)
                gated = pid_d.get("gated_requests_total", 0)
                committed = pid_d.get("committed_kv_tokens", 0)

                print(f"[{elapsed:5.1f}s] Running: {running} | Waiting: {waiting} | GatedTotal: {gated} | KV: {kv:5.1f}% ({committed:,} comm) | Step: {step_ms:4.1f}ms | Preempts: {preempts}", flush=True)
        except Exception:
            pass

        time.sleep(1.0)


def main():
    print("=================================================================")
    print(" VERIFICATION: 4 CONCURRENT DEEP-CONTEXT REQUESTS UNDER DYNAMIC PID")
    print("=================================================================")
    
    # 1. Reset KV cache
    try:
        req = urllib.request.Request(
            f"{BASE_URL}/v1/kv-offload/reset",
            method="POST",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        urllib.request.urlopen(req, timeout=5)
        print("[Setup] Clean baseline: KV Cache reset clean.")
    except Exception as e:
        print(f"[Setup] Warning reset: {e}")

    # 2. Check auto-detected PID status
    try:
        req = urllib.request.Request(f"{BASE_URL}/v1/genesis/pid", headers={"Authorization": f"Bearer {API_KEY}"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            status = json.loads(resp.read().decode())
            print(f"[Setup] Dynamic PID Status: enabled={status.get('enabled')}, auto_cap={status.get('total_kv_capacity_tokens'):,} tok, safe_budget={status.get('max_kv_tokens'):,} tok, target={status.get('target_step_ms')} ms")
    except Exception as e:
        print(f"[Setup] Warning PID query: {e}")

    # 3. Start monitor
    stop_event = threading.Event()
    monitor_t = threading.Thread(target=monitor_loop, args=(stop_event,), daemon=True)
    monitor_t.start()

    # 4. Launch 4 concurrent requests (~43,200 tokens each)
    num_reqs = 4
    max_tokens = 256
    results = []

    print(f"\n[Workload] Generating {num_reqs} prompts (~43k tokens each)...")
    prompts = [generate_prompt(i, repeats=600) for i in range(num_reqs)]
    print(f"[Workload] Launching {num_reqs} concurrent requests...")

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_reqs) as executor:
        futures = [
            executor.submit(run_streaming_request, i, prompts[i], max_tokens, results)
            for i in range(num_reqs)
        ]
        concurrent.futures.wait(futures, timeout=300)
    total_time = time.time() - t0

    stop_event.set()
    monitor_t.join(timeout=2.0)

    print("\n" + "=" * 65)
    print(" VERIFICATION COMPLETE")
    print("=" * 65)
    print(f"Total Completed Requests: {len(results)} / {num_reqs}")
    print(f"Total Wall-Clock Time:    {total_time:.1f}s")
    for r in results:
        print(f"  Req #{r['req_idx']}: Prompt={r['prompt_tokens']:,}, Gen={r['completion_tokens']}, TTFT={r['ttft_s']}s, PP={r['pp_tps']} tok/s, TG={r['tg_tps']} tok/s")

    # Aggregate TG speed
    total_gen_tokens = sum(r['completion_tokens'] for r in results)
    print(f"\nAggregate Tokens Generated: {total_gen_tokens}")
    if results:
        avg_tg = sum(r['tg_tps'] for r in results) / len(results)
        print(f"Average Per-Stream TG Speed: {avg_tg:.1f} tok/s")
    print("=" * 65)


if __name__ == "__main__":
    main()
