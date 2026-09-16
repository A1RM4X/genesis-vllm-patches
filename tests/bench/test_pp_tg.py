#!/usr/bin/env python3
"""
Test de rendimiento: TTFT, Prefill Throughput (PP tok/s) y Generation Throughput (TG tok/s).
Usa el endpoint local OpenAI compatible de vLLM en http://127.0.0.1:8320/v1/chat/completions.
"""
import json
import os
import sys
import time
import urllib.request

API_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8320/v1/chat/completions")
API_KEY = os.environ.get("VLLM_API_KEY", "<REDACTADO: clave rotada 2026-09-19>")
MODEL = os.environ.get("VLLM_MODEL", "qwen3.8")


def run_benchmark_request(prompt: str, max_tokens: int = 256, temperature: float = 0.0):
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
    generated_text = ""
    prompt_tokens = 0
    completion_tokens = 0

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

    # Si usage no vino por streaming, estimar tokens generados con chunks
    if completion_tokens == 0:
        completion_tokens = max(1, chunks_received)
    if prompt_tokens == 0:
        # Estimación aproximada si no vino en usage
        prompt_tokens = len(prompt.split()) * 4 // 3

    pp_speed = prompt_tokens / ttft if ttft > 0 else 0.0
    tg_speed = (completion_tokens - 1) / gen_time if gen_time > 0 and completion_tokens > 1 else (completion_tokens / total_time)
    inter_token_latency = (gen_time / max(1, completion_tokens - 1)) * 1000 if completion_tokens > 1 else 0.0

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": ttft,
        "gen_time_s": gen_time,
        "total_time_s": total_time,
        "pp_tok_s": pp_speed,
        "tg_tok_s": tg_speed,
        "inter_token_ms": inter_token_latency,
        "text_preview": generated_text[:120].replace("\n", " ") + "...",
    }


def main():
    print(f"=== BENCHMARK VLLM PP & TG ===")
    print(f"Endpoint: {API_URL}")
    print(f"Modelo:   {MODEL}\n")

    test_cases = [
        ("Warmup", "Hola, respondé en una sola palabra: ¿estás listo?", 16),
        ("Prompt Corto (Gen Media)", "Explica en 3 párrafos qué es la computación cuántica y sus aplicaciones principales.", 256),
        ("Prompt Largo (Prefill Test)", ("El análisis numérico y la optimización en GPUs modernas requieren entender tanto la microarquitectura de hardware como las limitaciones de memoria de ancho de banda y latencia. " * 30) + "\n\nResume los puntos clave del texto en 5 viñetas concisas.", 200),
    ]

    for name, prompt, max_tok in test_cases:
        print(f"--- Ejecutando: {name} (max_tokens={max_tok}) ---")
        try:
            res = run_benchmark_request(prompt, max_tokens=max_tok)
            print(f"  Prompt tokens:       {res['prompt_tokens']}")
            print(f"  Completion tokens:   {res['completion_tokens']}")
            print(f"  TTFT:                {res['ttft_s']*1000:.1f} ms")
            print(f"  Prefill (PP):        {res['pp_tok_s']:.1f} tokens/s")
            print(f"  Generación (TG):     {res['tg_tok_s']:.1f} tokens/s")
            print(f"  Latencia inter-token:{res['inter_token_ms']:.1f} ms/token")
            print(f"  Preview:             {res['text_preview']}\n")
        except Exception as e:
            print(f"  [ERROR] Falló request: {e}\n")


if __name__ == "__main__":
    main()
