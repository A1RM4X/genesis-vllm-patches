#!/usr/bin/env python3
"""Benchmark de copias de estado mamba: batch_memcpy (Triton, V1 actual)
vs cuMemcpyBatchAsync (DMA batch vía ctypes, como el KV offload).

Payload realista: estado temporal GDN por capa = H24×128×128 fp16 ≈ 786 KB.
"""
import ctypes
import os
import sys
import traceback

import numpy as np
import torch

N_COPIES = int(os.environ.get("LAB_N_COPIES", "96"))
STATE_BYTES = 24 * 128 * 128 * 2  # 786,432 B


def timeit(fn, iters=30, warmup=5):
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
    print(f"=== bench_memcpy: {N_COPIES} copias x {STATE_BYTES} B "
          f"({N_COPIES * STATE_BYTES / 1e6:.1f} MB) ===", flush=True)
    src_pool = torch.randn(N_COPIES * STATE_BYTES // 2, device="cuda",
                           dtype=torch.float16)
    dst_pool = torch.empty_like(src_pool)

    # Punteros base alineados a cada "estado"
    src_ptrs_np = np.array(
        [src_pool.data_ptr() + i * STATE_BYTES for i in range(N_COPIES)],
        dtype=np.uint64)
    dst_ptrs_np = np.array(
        [dst_pool.data_ptr() + i * STATE_BYTES for i in range(N_COPIES)],
        dtype=np.uint64)
    sizes_np = np.full(N_COPIES, STATE_BYTES, dtype=np.uint64)

    # --- 1) Triton batch_memcpy del runner V1 (tensores GPU de punteros) ---
    try:
        from vllm.v1.worker.mamba_utils import batch_memcpy
        src_t = torch.from_numpy(src_ptrs_np.astype(np.int64)).cuda()
        dst_t = torch.from_numpy(dst_ptrs_np.astype(np.int64)).cuda()
        siz_t = torch.from_numpy(sizes_np.astype(np.int64)).cuda()

        def run_triton():
            batch_memcpy(src_t, dst_t, siz_t)

        run_triton()
        t_tri = timeit(run_triton)
        gb = N_COPIES * STATE_BYTES * 2 / 1e9
        print(f"Triton batch_memcpy: {t_tri:.3f} ms ({gb/t_tri*1000:.2f} GB/s)",
              flush=True)
    except Exception:
        traceback.print_exc()

    # --- 2) cuMemcpyBatchAsync directo (ctypes, patrón cuda_mem_ops) ---
    try:
        from vllm.v1.simple_kv_offload.cuda_mem_ops import (
            _resolve_batch_memcpy,)
        fn = _resolve_batch_memcpy()

        # attrs: un entry con srcAccessOrder=ANY (0) — igual que el connector.
        # CU_MEMCPY_SRC_ACCESS_ORDER_ANY = 0 (value), attr[0]=kind? El connector
        # arma: attrs=[(attr_kind, value)]; aquí replicamos lo mínimo válido:
        # usar numAttrs=0 también es legal según la API.
        dst_arr = np.frombuffer(dst_ptrs_np.tobytes(), dtype=np.uint64)
        src_arr = np.frombuffer(src_ptrs_np.tobytes(), dtype=np.uint64)
        siz_arr = sizes_np.copy()
        addr_d = dst_arr.ctypes.data_as(ctypes.c_void_p)
        addr_s = src_arr.ctypes.data_as(ctypes.c_void_p)
        addr_z = siz_arr.ctypes.data_as(ctypes.c_void_p)

        def run_cu():
            rc = fn(addr_d, addr_s, addr_z, ctypes.c_size_t(N_COPIES),
                    None, None, ctypes.c_size_t(0), None, None)
            if rc != 0:
                raise RuntimeError(f"cuMemcpyBatchAsync rc={rc}")

        run_cu()
        torch.cuda.synchronize()
        t_cu = timeit(run_cu)
        gb = N_COPIES * STATE_BYTES * 2 / 1e9
        print(f"cuMemcpyBatchAsync: {t_cu:.3f} ms ({gb/t_cu*1000:.2f} GB/s)",
              flush=True)
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    sys.exit(main())
