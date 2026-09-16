#!/usr/bin/env python3
"""Benchmark del sampler con el vocabulario REAL del modelo (248,320).

Compara: argmax (greedy) · triton top-k/top-p · flashinfer top-k/top-p,
sobre logits [40, 248320] fp32 (decode MTP 10 seqs). Mide además el costo
del softmax fp32 del rejection sampler ([160, 248320]).
"""
import sys
import traceback

import torch

V = 248320
M_DECODE = 40
M_SPEC = 160  # 40 tokens × (K+1)


def timeit(fn, iters=50, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    print(f"=== bench_sampler: vocab={V} ===", flush=True)
    logits = torch.randn(M_DECODE, V, device="cuda", dtype=torch.float32)

    # greedy
    t = timeit(lambda: logits.argmax(dim=-1))
    print(f"argmax[{M_DECODE},{V}]: {t:.3f} ms", flush=True)

    # triton top-k/top-p de vLLM
    try:
        from vllm.v1.sample.ops import topk_topp_triton as tt
        cands = [n for n in dir(tt) if n.startswith("apply")]
        print("[sampler] candidatos triton:", cands, flush=True)
        fn = getattr(tt, cands[0])
        k = torch.full((M_DECODE,), 50, device="cuda", dtype=torch.int32)
        p = torch.full((M_DECODE,), 0.95, device="cuda", dtype=torch.float32)
        try:
            fn(logits.clone(), k, p)
            t = timeit(lambda: fn(logits.clone(), k, p))
            print(f"triton topk_topp: {t:.3f} ms", flush=True)
        except TypeError:
            # firma alternativa (logits, sampling_params-ish)
            t = timeit(lambda: fn(logits.clone(), 50, 0.95))
            print(f"triton topk_topp(alt-sig): {t:.3f} ms", flush=True)
    except Exception:
        traceback.print_exc()

    # flashinfer sampler
    try:
        import flashinfer.sampling as fis
        k = 50
        p = 0.95

        def run_fi():
            return fis.top_k_top_p_sampling_from_logits(logits, k, p)

        run_fi()
        t = timeit(run_fi)
        print(f"flashinfer topk_topp: {t:.3f} ms", flush=True)
    except Exception:
        traceback.print_exc()

    # softmax del rejection sampler (path aleatorio)
    try:
        big = torch.randn(M_SPEC, V, device="cuda", dtype=torch.float32)
        t = timeit(lambda: torch.softmax(big, dim=-1, dtype=torch.float32),
                   iters=20)
        print(f"softmax fp32[{M_SPEC},{V}] (rejection path): {t:.3f} ms",
              flush=True)
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    sys.exit(main())
