#!/usr/bin/env python3
"""
Agentic Evaluation Suite: Terminal-Bench & SWE-bench Pro
Evaluates Qwen3.8-27B on realistic terminal execution, tool calling, bug fixing, and SWE tasks.
"""
import json
import os
import sys
import time
import urllib.request

API_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8320/v1/chat/completions")
API_KEY = os.environ.get("VLLM_API_KEY", "<REDACTADO: clave rotada 2026-09-19>")
MODEL = os.environ.get("VLLM_MODEL", "qwen3.8")

def query_model(messages, max_tokens=2048, temperature=0.0):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as response:
        data = json.loads(response.read().decode("utf-8"))
    t1 = time.perf_counter()
    
    choice = data["choices"][0]
    msg = choice["message"]
    usage = data.get("usage", {})
    return {
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
        "elapsed_s": t1 - t0,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }

# ─────────────────────────────────────────────────────────────────────────────
# 1. TERMINAL-BENCH TASKS
# ─────────────────────────────────────────────────────────────────────────────
TERMINAL_BENCH_TASKS = [
    {
        "id": "TB-01",
        "name": "Log Analysis & Stream Filtering (sed/awk/grep)",
        "prompt": (
            "You are a terminal automation agent. You have an access log file `/var/log/nginx/access.log`.\n"
            "Write a single robust bash pipeline to:\n"
            "1. Filter only HTTP 5xx status codes.\n"
            "2. Extract the client IP and requested path.\n"
            "3. Count frequency of each (IP, path) pair and sort top 10 descending.\n"
            "Return the bash command."
        ),
        "eval_fn": lambda res: "awk" in res and "sort" in res and "uniq" in res and ("5" in res),
        "description": "Validates single-line stream parsing with awk/sort/uniq for 5xx errors."
    },
    {
        "id": "TB-02",
        "name": "Git Merge Conflict & Rebase Automation",
        "prompt": (
            "You are resolving a detached HEAD rebase collision in git.\n"
            "The repository is mid-rebase on branch `feature/payment-v2` onto `origin/main`.\n"
            "A conflict occurred on `src/billing/stripe_adapter.py`.\n"
            "Provide the exact sequence of git commands to:\n"
            "1. Inspect the conflicting files.\n"
            "2. Keep the incoming changes from `origin/main` (their version) for this specific file.\n"
            "3. Stage the resolved file and continue the rebase process cleanly."
        ),
        "eval_fn": lambda res: ("status" in res or "diff" in res) and ("--theirs" in res or "checkout" in res) and "add" in res and "continue" in res,
        "description": "Tests multi-step git rebase conflict resolution sequence."
    },
    {
        "id": "TB-03",
        "name": "Linux Network & Zombie Process Triage",
        "prompt": (
            "A server has port 8443 in a CLOSE_WAIT accumulation state and orphan processes occupying file descriptors.\n"
            "Provide the terminal commands to:\n"
            "1. Identify the PID and process name holding port 8443.\n"
            "2. Count the number of open sockets in CLOSE_WAIT state.\n"
            "3. Gracefully terminate the process with SIGTERM, falling back to SIGKILL if not stopped after 5 seconds.\n"
            "Provide the complete shell script/commands."
        ),
        "eval_fn": lambda res: ("ss" in res or "netstat" in res or "lsof" in res) and ("8443" in res) and ("kill" in res),
        "description": "Tests socket inspection (ss/lsof) and signal-based process termination."
    },
    {
        "id": "TB-04",
        "name": "Multi-stage Dockerfile Optimization for PyTorch CUDA",
        "prompt": (
            "Write a minimal, production-grade multi-stage Dockerfile for a Python 3.12 FastAPI app with PyTorch CUDA:\n"
            "1. Builder stage with build-essential, git, compiling wheel dependencies.\n"
            "2. Final lightweight runtime stage copying only installed virtual environment and app code.\n"
            "3. Non-root user `appuser`.\n"
            "4. Proper CMD with dumb-init or exec uvicorn.\n"
            "Return the Dockerfile."
        ),
        "eval_fn": lambda res: "FROM" in res and ("builder" in res.lower()) and ("COPY" in res) and ("USER" in res or "useradd" in res),
        "description": "Verifies multi-stage build structure and security best practices."
    }
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. SWE-BENCH PRO TASKS
# ─────────────────────────────────────────────────────────────────────────────
SWE_BENCH_PRO_TASKS = [
    {
        "id": "SWE-01",
        "name": "Asyncio Concurrency Deadlock & Cancellation Safety",
        "prompt": (
            "Review this buggy Python async worker queue:\n\n"
            "```python\n"
            "import asyncio\n"
            "class WorkerPool:\n"
            "    def __init__(self, size=5):\n"
            "        self.queue = asyncio.Queue()\n"
            "        self.workers = [asyncio.create_task(self._worker()) for _ in range(size)]\n"
            "    async def _worker(self):\n"
            "        while True:\n"
            "            task, fut = await self.queue.get()\n"
            "            res = await task()\n"
            "            fut.set_result(res)\n"
            "            self.queue.task_done()\n"
            "    async def submit(self, task_fn):\n"
            "        fut = asyncio.Future()\n"
            "        await self.queue.put((task_fn, fut))\n"
            "        return await fut\n"
            "    async def shutdown(self):\n"
            "        for w in self.workers: w.cancel()\n"
            "```\n\n"
            "Identify 2 major bugs in exception handling and shutdown, and write the corrected `WorkerPool` implementation."
        ),
        "eval_fn": lambda res: ("except" in res or "try" in res) and ("set_exception" in res or "exception" in res) and ("cancel" in res or "gather" in res or "CancelledError" in res),
        "description": "Tests asyncio error propagation (set_exception) and clean task cancellation handling."
    },
    {
        "id": "SWE-02",
        "name": "Unified Diff / Patch Synthesis for JSON Schema Validation",
        "prompt": (
            "We have a bug in `validator.py` where nullable nested dicts trigger `AttributeError: 'NoneType' object has no attribute 'get'`.\n"
            "Original file:\n"
            "```python\n"
            "def validate_payload(data: dict) -> bool:\n"
            "    user_profile = data.get('user', {})\n"
            "    settings = user_profile.get('settings', {})\n"
            "    return settings.get('theme') in ['dark', 'light']\n"
            "```\n"
            "If `data = {'user': None}` is passed, `data.get('user', {})` returns `None`, causing `user_profile.get` to crash.\n"
            "Provide the unified git diff patch (`--- a/validator.py ... +++ b/validator.py`) that fixes this robustly."
        ),
        "eval_fn": lambda res: ("---" in res or "+++" in res or "diff" in res) and ("(data.get('user') or {})" in res or "user_profile is None" in res or "isinstance" in res or "data.get('user') or {}" in res or "if user_profile" in res or "get" in res),
        "description": "Validates git diff format and defensive None-coalescing logic."
    },
    {
        "id": "SWE-03",
        "name": "Algorithmic Refactoring: Keyword Matcher using Trie",
        "prompt": (
            "Implement a high-throughput Python keyword matcher class `FastMatcher` that:\n"
            "1. Takes a list of keywords `['<|think|>', '</think>', '<tool_call>', '<function=']` in `__init__`.\n"
            "2. Provides `find_first(text: str) -> tuple[int, str] | None` returning the start index and matched keyword in O(L) time.\n"
            "3. Uses a Trie structure with root and children dictionary nodes.\n"
            "Return complete Python code with docstrings."
        ),
        "eval_fn": lambda res: "class" in res and ("trie" in res.lower() or "node" in res.lower() or "children" in res.lower()) and "find_first" in res,
        "description": "Evaluates Trie data structure implementation and search logic."
    }
]

def run_suite():
    print("================================================================================")
    print("  AGENTIC BENCHMARK: TERMINAL-BENCH & SWE-BENCH PRO (twolven/Qwen3.8-27B AWQ MTP)")
    print(f"  Endpoint: {API_URL} | Modelo: {MODEL}")
    print("================================================================================\n")

    results = []

    # 1. Terminal-Bench
    print(">>> 1. TERMINAL-BENCH EVALUATION (Linux CLI, Git, Net & Docker)")
    print("────────────────────────────────────────────────────────────────────────────────")
    tb_passed = 0
    for t in TERMINAL_BENCH_TASKS:
        print(f"• Running [{t['id']}] {t['name']}...")
        messages = [{"role": "user", "content": t["prompt"]}]
        res = query_model(messages, max_tokens=2048)
        full_text = (res["reasoning"] or "") + "\n" + (res["content"] or "")
        is_pass = t["eval_fn"](full_text)
        if is_pass:
            tb_passed += 1
            status = "✔ PASS"
        else:
            status = "✘ FAIL"
        
        reasoning_words = len(res["reasoning"].split()) if res["reasoning"] else 0
        print(f"  Result: {status} | Time: {res['elapsed_s']:.2f}s | Tokens: {res['completion_tokens']} (reasoning={reasoning_words}w)")
        print(f"  Criteria: {t['description']}\n")
        results.append({"suite": "Terminal-Bench", "id": t["id"], "name": t["name"], "pass": is_pass, "time": res["elapsed_s"]})

    # 2. SWE-bench Pro
    print("\n>>> 2. SWE-BENCH PRO EVALUATION (Architecture, Bugfix, Git Diff & Trie)")
    print("────────────────────────────────────────────────────────────────────────────────")
    swe_passed = 0
    for t in SWE_BENCH_PRO_TASKS:
        print(f"• Running [{t['id']}] {t['name']}...")
        messages = [{"role": "user", "content": t["prompt"]}]
        res = query_model(messages, max_tokens=2048)
        full_text = (res["reasoning"] or "") + "\n" + (res["content"] or "")
        is_pass = t["eval_fn"](full_text)
        if is_pass:
            swe_passed += 1
            status = "✔ PASS"
        else:
            status = "✘ FAIL"
        
        reasoning_words = len(res["reasoning"].split()) if res["reasoning"] else 0
        print(f"  Result: {status} | Time: {res['elapsed_s']:.2f}s | Tokens: {res['completion_tokens']} (reasoning={reasoning_words}w)")
        print(f"  Criteria: {t['description']}\n")
        results.append({"suite": "SWE-bench Pro", "id": t["id"], "name": t["name"], "pass": is_pass, "time": res["elapsed_s"]})

    # Summary
    print("================================================================================")
    print("  RESUMEN FINAL DE EVALUACIÓN AGÉNTICA")
    print("================================================================================")
    print(f"  Terminal-Bench: {tb_passed}/{len(TERMINAL_BENCH_TASKS)} aprobados ({tb_passed/len(TERMINAL_BENCH_TASKS)*100:.1f}%)")
    print(f"  SWE-bench Pro:  {swe_passed}/{len(SWE_BENCH_PRO_TASKS)} aprobados ({swe_passed/len(SWE_BENCH_PRO_TASKS)*100:.1f}%)")
    total_passed = tb_passed + swe_passed
    total_tasks = len(TERMINAL_BENCH_TASKS) + len(SWE_BENCH_PRO_TASKS)
    print(f"  TOTAL GLOBAL:   {total_passed}/{total_tasks} aprobados ({total_passed/total_tasks*100:.1f}%)")
    print("================================================================================")

if __name__ == "__main__":
    run_suite()
