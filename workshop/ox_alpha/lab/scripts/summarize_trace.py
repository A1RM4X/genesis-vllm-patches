#!/usr/bin/env python3
"""Contabilidad de Memcpy sobre trazas chrome de torch.profiler.

Uso: summarize_trace.py <trace1.json> [trace2.json ...]

Agrega por archivo y en total:
  - bytes y conteo de cada evento gpu_memcpy (DtoH / HtoD / DtoD)
  - split global DtoH vs HtoD
  - top kernels (marlin/flashinfer/triton/nccl) por tiempo total
"""
import collections
import glob
import json
import sys


def summarize(path: str) -> tuple[dict, dict]:
    with open(path) as f:
        data = json.load(f)
    memcpy_bytes: dict[str, int] = collections.Counter()
    memcpy_count: dict[str, int] = collections.Counter()
    kernel_time: dict[str, float] = collections.Counter()
    for e in data.get("traceEvents", []):
        name = str(e.get("name", ""))
        cat = str(e.get("cat", ""))
        if cat == "gpu_memcpy" or "Memcpy" in name:
            b = int((e.get("args") or {}).get("bytes", 0))
            memcpy_bytes[name] += b
            memcpy_count[name] += 1
        elif cat == "kernel":
            dur = float(e.get("dur", 0.0))
            for tag in ("marlin", "flashinfer", "nccl", "triton",
                        "gemm", "attention"):
                if tag in name.lower():
                    kernel_time[tag] += dur
                    break
            else:
                kernel_time["_other"] += dur
    return ({"bytes": dict(memcpy_bytes), "count": dict(memcpy_count)},
            dict(kernel_time))


def main() -> None:
    paths: list[str] = []
    for p in sys.argv[1:]:
        paths.extend(sorted(glob.glob(p)))
    if not paths:
        print("no traces found")
        return

    tot_d2h = 0
    tot_h2d = 0
    for path in paths:
        print(f"\n===== {path} =====")
        info, kern = summarize(path)
        d2h = h2d = d2d = 0
        for name, b in sorted(info["bytes"].items(),
                              key=lambda kv: -kv[1]):
            c = info["count"][name]
            print(f"  {b / 1e6:12.4f} MB  {c:7d}x  {name}")
            low = name.upper()
            if "DTOH" in low:
                d2h += b
            elif "HTOD" in low:
                h2d += b
            else:
                d2d += b
        print(f"  -- DtoH={d2h / 1e6:.4f} MB  HtoD={h2d / 1e6:.4f} MB  "
              f"DtoD={d2d / 1e6:.4f} MB")
        if kern:
            top = ", ".join(f"{k}={v / 1e3:.1f}ms"
                            for k, v in sorted(kern.items(),
                                               key=lambda kv: -kv[1])[:8])
            print(f"  -- kernels(top): {top}")
        tot_d2h += d2h
        tot_h2d += h2d

    print(f"\n===== TOTAL ({len(paths)} traces) =====")
    print(f"DtoH = {tot_d2h / 1e6:.4f} MB   HtoD = {tot_h2d / 1e6:.4f} MB")


if __name__ == "__main__":
    main()
