#!/usr/bin/env python3
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8320/v1/chat/completions")
API_KEY = os.environ.get("VLLM_API_KEY", os.environ.get("VLLM_API_KEY", ""))
MODEL = os.environ.get("VLLM_MODEL", "qwen3.8")

def query(prompt: str, max_tokens: int = 256, temperature: float = 0.0):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
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
    t_first_token = None
    chunks_received = 0
    prompt_tokens = 0
    completion_tokens = 0
    generated_text = ""

    with urllib.request.urlopen(req, timeout=300) as response:
        for line in response:
            line = line.decode("utf-8").strip()
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if "usage" in data and data["usage"]:
                usage = data["usage"]
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", completion_tokens)

            choices = data.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or ""
                token_piece = content or reasoning
                if token_piece:
                    if t_first_token is None:
                        t_first_token = time.perf_counter()
                    generated_text += token_piece
                    chunks_received += 1

    t_end = time.perf_counter()
    if t_first_token is None:
        t_first_token = t_end

    ttft = t_first_token - t0
    gen_time = t_end - t_first_token
    total_time = t_end - t0

    if completion_tokens == 0:
        completion_tokens = max(1, chunks_received)
    if prompt_tokens == 0:
        prompt_tokens = len(prompt.split()) * 4 // 3

    pp_speed = prompt_tokens / ttft if ttft > 0 else 0.0
    tg_speed = (completion_tokens - 1) / gen_time if gen_time > 0 and completion_tokens > 1 else (completion_tokens / total_time)
    tpot_ms = (gen_time / max(1, completion_tokens - 1)) * 1000 if completion_tokens > 1 else 0.0

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_ms": ttft * 1000,
        "gen_time_s": gen_time,
        "total_time_s": total_time,
        "pp_tok_s": pp_speed,
        "tg_tok_s": tg_speed,
        "tpot_ms": tpot_ms,
        "preview": generated_text[:80].replace("\n", " ") + "...",
    }

def main():
    print(f"================================================================")
    print(f"  BENCHMARK DETALLADO: PREFILL (PP) & GENERACIÓN (TG) - {MODEL}")
    print(f"  Hardware: Dual RTX 3090 (TP=2) | Quant: AWQ W4A16 | MTP: Active")
    print(f"================================================================\n")

    # 1. Sweep de Prefill
    print(">>> 1. PREFILL & TTFT SWEEP (Variación de tamaño de Prompt)")
    paragraphs = (
        "El desarrollo de sistemas de inteligencia artificial modernos requiere un equilibrio "
        "exhaustivo entre precisión computacional, eficiencia de memoria y rendimiento de throughput en inferencia. "
        "A nivel de arquitectura de hardware, los Tensor Cores permiten acelerar multiplicaciones matriciales densas "
        "mediante cuantizaciones int4, int8 y fp8, reduciendo el cuello de botella en el bus PCIe y memoria VRAM. "
    )

    for factor, label in [(1, "Corto (~50 tok)"), (15, "Medio (~500 tok)"), (60, "Largo (~2K tok)"), (150, "Muy Largo (~5K tok)")]:
        p = paragraphs * factor + "\n\nEscribe una breve conclusión técnica en 2 líneas."
        res = query(p, max_tokens=32)
        print(f"  [{label}]")
        print(f"    Prompt Tokens: {res['prompt_tokens']:>5} | TTFT: {res['ttft_ms']:>6.1f} ms | Prefill: {res['pp_tok_s']:>7.1f} tok/s | TG: {res['tg_tok_s']:>5.1f} tok/s")

    # 2. Sweep de Generación (TG / TPOT)
    print("\n>>> 2. GENERATION THROUGHPUT & TPOT SWEEP (Salidas largas)")
    for gen_tokens in [128, 256, 512, 1024]:
        p = "Genera una guía detallada paso a paso sobre implementación de arquitecturas de microservicios con Kubernetes, gRPC y monitoreo distribuido con Prometheus y Grafana."
        res = query(p, max_tokens=gen_tokens)
        print(f"  [Target: {gen_tokens} tokens]")
        print(f"    Gen Tokens: {res['completion_tokens']:>5} | Tiempo: {res['gen_time_s']:>5.2f}s | TG: {res['tg_tok_s']:>5.1f} tok/s | TPOT: {res['tpot_ms']:>5.1f} ms/tok")

    # 3. Concurrencia
    print("\n>>> 3. TEST DE CONCURRENCIA (Streams concurrentes)")
    for conc in [1, 2, 4]:
        prompt = "Explica los fundamentos del algoritmo Raft de consenso distribuido y sus 3 subproblemas."
        t_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=conc) as pool:
            futures = [pool.submit(query, prompt, 256) for _ in range(conc)]
            results = [f.result() for f in futures]
        t_total = time.perf_counter() - t_start
        total_gen_tokens = sum(r["completion_tokens"] for r in results)
        agg_throughput = total_gen_tokens / t_total
        avg_tg = sum(r["tg_tok_s"] for r in results) / len(results)
        print(f"  [Concurrencia c={conc}]")
        print(f"    Total Tokens: {total_gen_tokens} en {t_total:.2f}s | Throughput Agregado: {agg_throughput:.1f} tok/s | TG Promedio por Stream: {avg_tg:.1f} tok/s")

    print("\n================================================================")
    print("  TEST DE VELOCIDAD COMPLETADO")
    print("================================================================")

if __name__ == "__main__":
    main()
