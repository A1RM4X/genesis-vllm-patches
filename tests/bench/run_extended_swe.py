#!/usr/bin/env python3
import json, os, time, urllib.request

API_URL = os.environ.get("VLLM_URL", "http://172.20.0.229:8398/v1/chat/completions")
API_KEY = os.environ.get("VLLM_API_KEY", "<REDACTADO: clave rotada 2026-09-19>")
MODEL = os.environ.get("VLLM_MODEL", "qwen3.8")

def query_model(messages, max_tokens=4096, temperature=0.0):
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
    with urllib.request.urlopen(req, timeout=300) as response:
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

TASKS = [
    {
        "id": "SWE-03",
        "name": "Algorithmic Trie Keyword Matcher",
        "prompt": "Implement a high-performance Python Trie keyword matcher class `FastMatcher` with `find_first(text: str) -> tuple[int, str] | None`. Include complete code and concise explanation.",
        "eval_fn": lambda res: "class" in res and ("trie" in res.lower() or "node" in res.lower() or "children" in res.lower()) and "find_first" in res
    },
    {
        "id": "SWE-04",
        "name": "FastAPI Pydantic V2 Migration & Custom Validator",
        "prompt": "Refactor this Pydantic V1 model to Pydantic V2:\n```python\nfrom pydantic import BaseModel, validator\nclass User(BaseModel):\n    email: str\n    @validator('email')\n    def validate_email(cls, v):\n        if '@' not in v: raise ValueError('invalid')\n        return v\n```\nUse `@field_validator(..., mode='after')` and `model_config` according to Pydantic V2 specifications.",
        "eval_fn": lambda res: "field_validator" in res and ("mode=" in res or "@field_validator" in res) and "User" in res
    },
    {
        "id": "SWE-05",
        "name": "PostgreSQL Composite Indexing for Range & Equality Queries",
        "prompt": "Given a PostgreSQL table `orders(id serial, tenant_id uuid, status varchar, created_at timestamptz, amount numeric)`.\nWrite the optimal composite B-tree index for this frequent query:\n`SELECT * FROM orders WHERE tenant_id = $1 AND status = 'COMPLETED' AND created_at >= $2 ORDER BY created_at DESC LIMIT 50;`\nExplain index column ordering rule (Equality first, Range/Order last).",
        "eval_fn": lambda res: "CREATE INDEX" in res and "tenant_id" in res and "status" in res and "created_at" in res
    }
]

print("Running Extended SWE-bench Pro Suite...")
results = []
for t in TASKS:
    print(f"• Executing {t['id']}: {t['name']}...")
    res = query_model([{"role": "user", "content": t["prompt"]}], max_tokens=3000)
    full_text = (res["reasoning"] or "") + "\n" + (res["content"] or "")
    passed = t["eval_fn"](full_text)
    print(f"  Result: {'✔ PASS' if passed else '✘ FAIL'} | Time: {res['elapsed_s']:.2f}s | Tokens: {res['completion_tokens']}")
    results.append({"id": t["id"], "name": t["name"], "passed": passed, "latency": res["elapsed_s"], "tokens": res["completion_tokens"]})

with open("tests/bench/extended_swe_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Finished. Saved to tests/bench/extended_swe_results.json")
