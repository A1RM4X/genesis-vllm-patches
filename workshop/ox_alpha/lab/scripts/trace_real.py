#!/usr/bin/env python3
"""Corrida REAL del laboratorio: Qwen3.8-27B-Uncensored-FP8, TP=2, MTP K=3.

Debe ser un ARCHIVO (no stdin) porque VLLM_WORKMPROC_METHOD=spawn re-importa
el módulo principal en cada worker; y todo el código va bajo el guard
``__main__`` para que los hijos no reconstruyan el engine.
"""
import glob
import os


def main() -> None:
    snaps = sorted(glob.glob(
        "/models-cache/hub/models--orcarouter--Qwen3.8-27B-Uncensored-FP8"
        "/snapshots/*"))
    assert snaps, "modelo no encontrado en /models-cache"
    model = snaps[0]
    attn_backend = os.environ.get("ATTN_BACKEND", "FLASHINFER")
    print(f"[trace_real] model = {model}", flush=True)
    print(f"[trace_real] attention backend = {attn_backend}", flush=True)

    from vllm import LLM, SamplingParams  # noqa: E402

    kwargs = dict(
        model=model,
        tensor_parallel_size=2,
        max_model_len=16384,
        gpu_memory_utilization=float(os.environ.get("LAB_GPU_MEMORY_UTILIZATION", "0.80")),
        dtype="float16",
        kv_cache_dtype="fp8_e4m3",
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=1664,
        max_num_seqs=10,
        disable_log_stats=os.environ.get("LAB_LOG_STATS", "0") != "1",
        speculative_config={"method": "mtp", "num_speculative_tokens": 3},
        cudagraph_capture_sizes=[4, 8, 16],
    )
    if attn_backend:
        kwargs["attention_backend"] = attn_backend
    try:
        llm = LLM(async_scheduling=True, **kwargs)
        print("[trace_real] async_scheduling=True accepted", flush=True)
    except TypeError as exc:
        print(f"[trace_real] async_scheduling rejected ({exc}); retrying",
              flush=True)
        llm = LLM(**kwargs)

    n_p = int(os.environ.get("LAB_PROMPTS", "4"))
    n_t = int(os.environ.get("LAB_TOKENS", "64"))
    # LAB_PROMPT_LEN: longitud objetivo del prompt en tokens (aprox; ~11-12
    # tokens por repetición de la frase de relleno). Con >0 el workload pasa
    # a ser PREFILL-dominado (mide la ganancia INT8 de PN110); con 0 se usa
    # el prompt corto clásico (decode-dominado).
    plen = int(os.environ.get("LAB_PROMPT_LEN", "0"))
    if plen > 0:
        filler = ("Analiza el siguiente problema paso a paso considerando "
                  "todas las restricciones y casos borde posibles. ")
        reps = max(1, plen * 12 // len(filler))
        base = filler * reps
        prompts = [base] * n_p
        print(f"[trace_real] prompt_len~{plen} tokens x{n_p} prompts "
              f"(chars={len(base)})", flush=True)
    else:
        prompts = ["Analiza el siguiente problema paso a paso:"] * n_p

    # Warmup fuera de la ventana de profiling
    llm.generate(prompts[:1], SamplingParams(max_tokens=8, temperature=0.0))
    print("[trace_real] warmup done", flush=True)

    # Ventana de profiling activa en los workers (steps 5..25)
    import time
    t0 = time.perf_counter()
    outs = llm.generate(prompts,
                        SamplingParams(max_tokens=n_t, temperature=0.7,
                                       ignore_eos=True))
    dt = time.perf_counter() - t0
    n_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"[trace_real] GEN_DONE requests={len(outs)} tokens={n_tok} "
          f"wall={dt:.2f}s tps={n_tok / dt:.1f}", flush=True)
    print("[trace_real] SAMPLE_TEXT:",
          repr(outs[0].outputs[0].text[:96]), flush=True)


if __name__ == "__main__":
    main()
