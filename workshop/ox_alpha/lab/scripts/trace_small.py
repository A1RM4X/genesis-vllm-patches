#!/usr/bin/env python3
"""Trazado end-to-end del circuito del token con un modelo pequeño.

Corre DENTRO del contenedor lab (imagen vllm/vllm-openai:v0.23.0). La
instrumentación real vive en patch/sitecustomize.py (inyectado por
PYTHONPATH): este script solo orquesta generación para que el runner pase
por warmup + ventana de profiling.

Fases:
  1. Warmup: 1 prompt corto x 8 tokens (pasos 1..N fuera de la ventana).
  2. Medición: 8 prompts x 48 tokens (decode-heavy, ignore_eos) — aquí el
     profiler del worker captura kernels + Memcpy con tamaños.
"""
import os
import time

MODEL = os.environ.get("LAB_MODEL", "Qwen/Qwen3-0.6B")

from vllm import LLM, SamplingParams  # noqa: E402


def main() -> None:
    kwargs = dict(
        model=MODEL,
        max_model_len=4096,
        gpu_memory_utilization=float(os.environ.get("LAB_GMU", "0.85")),
        enforce_eager=os.environ.get("LAB_EAGER") == "1",
        disable_log_stats=True,
    )
    print(f"[trace_small] loading {MODEL} with {kwargs}", flush=True)
    llm = LLM(**kwargs)

    prompts = ["El circuito de un token en vLLM comienza cuando"] * 8

    # Fase 1: warmup (consume pasos 1..~10 según scheduling)
    llm.generate(prompts[:1], SamplingParams(max_tokens=8, temperature=0.0))
    print("[trace_small] warmup done", flush=True)

    # Fase 2: ventana de profiling activa (steps 5..25 del runner)
    t0 = time.perf_counter()
    outs = llm.generate(
        prompts,
        SamplingParams(max_tokens=48, temperature=0.7, ignore_eos=True),
    )
    dt = time.perf_counter() - t0
    n_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"[trace_small] GEN_DONE requests={len(outs)} tokens={n_tok} "
          f"wall={dt:.2f}s tps={n_tok / dt:.1f}", flush=True)
    print("[trace_small] SAMPLE_TEXT:",
          repr(outs[0].outputs[0].text[:96]), flush=True)


if __name__ == "__main__":
    main()
