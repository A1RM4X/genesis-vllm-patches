#!/usr/bin/env python3
"""CK-1.1 — Sonda de estructura del draft MTP: ¿qué existe realmente?

Con VLLM_ENABLE_V1_MULTIPROCESSING=0 todo vive en-process, así que podemos
caminar el árbol del modelo tras la carga y volcar:
  - todos los módulos LMHead/ParallelLMHead (prefijo, dtype, forma)
  - los hijos de primer nivel bajo cualquier prefijo que contenga 'mtp'
  - bytes de parámetros bajo 'mtp'
"""
import glob
import os


def main() -> None:
    snaps = sorted(glob.glob(
        "/models-cache/hub/models--orcarouter--Qwen3.8-27B-Uncensored-FP8"
        "/snapshots/*"))
    model = snaps[0]
    print(f"[probe] model={model}", flush=True)

    from vllm import LLM  # noqa: E402

    kwargs = dict(
        model=model,
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.80,
        dtype="float16",
        kv_cache_dtype="fp8_e4m3",
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        disable_log_stats=True,
        speculative_config={"method": "mtp", "num_speculative_tokens": 3},
        cudagraph_capture_sizes=[4],
    )
    llm = LLM(**kwargs)
    print("[probe] engine up", flush=True)

    # Navegar al modelo real (in-process)
    ec = getattr(llm.llm_engine, "engine_core", None)
    inner = getattr(ec, "engine_core", None)
    me = getattr(inner, "model_executor", None)
    worker = getattr(me, "worker", None) or getattr(
        inner, "worker", None)
    mr = getattr(worker, "model_runner", None)
    model = getattr(mr, "model", None)
    if model is None:
        # uniprocs alternativos
        for attr in ("engine", "_engine"):
            pass
        print("[probe] WARN: no pude llegar al model; attrs:",
              [a for a in dir(inner) if "exec" in a.lower() or "worker" in
               a.lower()], flush=True)
        return

    import torch  # noqa: E402

    lms, mtp_bytes = [], 0
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        if "LMHead" in cls:
            w = getattr(mod, "weight", None)
            qm = getattr(mod, "quant_method", None)
            lms.append((name, cls,
                        str(tuple(w.shape)) if w is not None else "-",
                        str(w.dtype) if w is not None else "-",
                        type(qm).__name__ if qm is not None else "-"))
        if "mtp" in name.lower():
            for pn, p in mod.named_parameters(recurse=False):
                mtp_bytes += p.numel() * p.element_size()

    print("=== LMHead-like modules ===", flush=True)
    for row in lms:
        print("  ", row, flush=True)

    print(f"=== parámetros directos bajo prefijos 'mtp': "
          f"{mtp_bytes/1e6:.1f} MB ===", flush=True)

    # Desglose del subárbol mtp (primer nivel de grandes bloques)
    agg: dict[str, int] = {}
    for name, p in model.named_parameters():
        low = name.lower()
        if "mtp" in low or "draft" in low:
            key = "mtp/" + name.split(".")[1] if "." in name else name
            agg[key.split(".")[0]] = agg.get(key.split(".")[0], 0) + \
                p.numel() * p.element_size()
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1])[:12]:
        print(f"   {v/1e6:9.1f} MB  {k}", flush=True)


if __name__ == "__main__":
    main()
