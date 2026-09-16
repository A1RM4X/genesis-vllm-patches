#!/usr/bin/env python3
"""CK-0.2 — Workload batch-10 para el barrido NCCL_PROTO/ALGO.

Igual config que trace_real.py (27B FP8 TP=2 MTP K=3) pero con 10 prompts
x 100 tokens para reproducir el régimen de PROD (max_num_seqs=10, ARs de
~400 KB por capa). La instrumentación de sitecustomize captura los tiempos
de kernel nccl en la ventana de profiling (pasos 5..25).
"""
import glob
import os


def main() -> None:
    snaps = sorted(glob.glob(
        "/models-cache/hub/models--orcarouter--Qwen3.8-27B-Uncensored-FP8"
        "/snapshots/*"))
    assert snaps, "modelo no encontrado en /models-cache"
    model = snaps[0]
    n_prompts = int(os.environ.get("NCCL_BENCH_PROMPTS", "10"))
    max_tokens = int(os.environ.get("NCCL_BENCH_TOKENS", "100"))
    print(f"[nccl] model={model} prompts={n_prompts} tokens={max_tokens} "
          f"PROTO={os.environ.get('NCCL_PROTO', '-')} "
          f"ALGO={os.environ.get('NCCL_ALGO', '-')}", flush=True)

    from vllm import LLM, SamplingParams  # noqa: E402

    kwargs = dict(
        model=model,
        tensor_parallel_size=2,
        max_model_len=16384,
        gpu_memory_utilization=0.80,
        dtype="float16",
        kv_cache_dtype="fp8_e4m3",
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=1664,
        max_num_seqs=10,
        disable_log_stats=True,
        speculative_config={"method": "mtp", "num_speculative_tokens": 3},
        cudagraph_capture_sizes=[4, 8, 16],
    )
    try:
        llm = LLM(async_scheduling=True, **kwargs)
    except TypeError:
        llm = LLM(**kwargs)

    prompts = ["Analiza el siguiente problema paso a paso:"] * n_prompts

    # Warmup fuera de ventana de profiling
    llm.generate(prompts[:1], SamplingParams(max_tokens=8, temperature=0.0))
    print("[nccl] warmup done", flush=True)

    import time
    t0 = time.perf_counter()
    outs = llm.generate(
        prompts,
        SamplingParams(max_tokens=max_tokens, temperature=0.7,
                       ignore_eos=True))
    dt = time.perf_counter() - t0
    n_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"[nccl] GEN_DONE requests={len(outs)} tokens={n_tok} "
          f"wall={dt:.2f}s tps={n_tok / dt:.1f}", flush=True)


if __name__ == "__main__":
    main()
