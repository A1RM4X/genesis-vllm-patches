#!/usr/bin/env python3
"""Benchmarks FLA/GDN en formas reales del híbrido (por rank TP=2).

H=24 value-heads/rank, K=V=128, BT=FLA_CHUNK_SIZE=64.
Prueba:
  1. solve_tril con FLA_TRIL_PRECISION=ieee vs tf32 (env antes de import).
  2. chunk_o con BKV_LIST ampliado (check_shared_mem→True) vs default.
  3. Pipeline completo chunk_gated_delta_rule prefill T=1664.
"""
import os
import sys
import traceback

MODE = os.environ.get("FLA_TRIL_PRECISION", "ieee")  # ieee | tf32 | tf32x3
EXPAND_BKV = os.environ.get("LAB_EXPAND_BKV", "0") == "1"

import torch  # noqa: E402

T = int(os.environ.get("LAB_T", "1664"))
B, H, DK, DV = 1, 24, 128, 128


def timeit(fn, iters=20, warmup=5):
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
    print(f"=== bench_fla: T={T} H={H} K={DK} V={DV} "
          f"precision={MODE} expand_bkv={EXPAND_BKV} ===", flush=True)
    from vllm.model_executor.layers.fla import ops as fla

    if EXPAND_BKV:
        # Fuerza el branch de shared-mem grande para que el autotuner pruebe
        # también BKV=[64,128] (GA102 queda fuera por ~1KB en el umbral).
        fla.chunk_o.check_shared_mem = lambda *a, **k: True
        print("[fla] check_shared_mem parcheado -> True", flush=True)

    q = torch.randn(B, T, H, DK, device="cuda", dtype=torch.float16)
    k = torch.randn(B, T, H, DK, device="cuda", dtype=torch.float16)
    v = torch.randn(B, T, H, DV, device="cuda", dtype=torch.float16)
    g = torch.randn(B, T, H, device="cuda", dtype=torch.float32)
    beta = torch.rand(B, T, H, device="cuda", dtype=torch.float32)
    h0 = torch.randn(B, H, DK, DV, device="cuda", dtype=torch.float32)

    # 1) solve_tril solo
    try:
        A = fla.chunk_scaled_dot_kkt_fwd(k=k, beta=beta, chunk_size=64)
        t = timeit(lambda: fla.solve_tril(A))
        print(f"solve_tril[{MODE}]: {t:.3f} ms", flush=True)
    except Exception:
        traceback.print_exc()

    # 2) chunk_o solo
    try:
        h = fla.chunk_gated_delta_rule_fwd_h(h=h0, k=k, v=v, g=g, beta=beta)[0]
        t_def = timeit(lambda: fla.chunk_o(q, k, v, h, g))
        print(f"chunk_o[BKV default]: {t_def:.3f} ms", flush=True)
        if EXPAND_BKV:
            t_exp = timeit(lambda: fla.chunk_o(q, k, v, h, g))
            print(f"chunk_o[BKV expandido]: {t_exp:.3f} ms", flush=True)
    except Exception:
        traceback.print_exc()

    # 3) pipeline completo de prefill
    try:
        def full():
            return fla.chunk_gated_delta_rule(
                q.clone(), k.clone(), v.clone(), g=g.clone(), beta=beta.clone(),
                initial_state=h0.clone(), output_final_state=True)

        full()
        t_full = timeit(full, iters=10)
        print(f"chunk_gated_delta_rule FULL prefill: {t_full:.3f} ms "
              f"({T/t_full*1000:.0f} tok/s)", flush=True)
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    sys.exit(main())
