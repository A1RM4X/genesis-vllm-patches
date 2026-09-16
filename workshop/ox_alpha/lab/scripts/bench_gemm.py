#!/usr/bin/env python3
"""Shootout de GEMMs en las formas EXACTAS del Qwen3.8-27B por rank (TP=2).

Compara tres caminos sobre los mismos (M, N, K):
  1. cuBLAS fp16            — referencia (lo que haría un checkpoint sin cuantizar)
  2. Marlin W8A16 (FP8 blk) — lo que corre HOY en producción (sm_86)
  3. cutlass INT8 W8A8      — tensor cores INT8 de Ampere (requeriría requantizar)

Shapes por rank: qkv(N=4096,K=5120) · o(5120,3072) · gate_up(17408,5120) ·
down(5120,8704) · lm_head(124160,5120). M=40 (decode MTP 10seq×4) y M=1664
(prefill chunk).
"""
import sys
import traceback

import torch

SHAPES = [
    ("qkv",     4096,  5120),
    ("o",       5120,  3072),
    ("gate_up", 17408, 5120),
    ("down",    5120,  8704),
    ("lm_head", 124160, 5120),
]
MS = [40, 1664]
ITERS = {40: 50, 1664: 20}


def timeit(fn, iters, warmup=5):
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
    return s.elapsed_time(e) / iters  # ms


def bench_one(name, N, K, M):
    iters = ITERS[M]
    x16 = torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.05

    # --- cuBLAS fp16 ---
    w16 = torch.randn(N, K, device="cuda", dtype=torch.float16) * 0.02
    t_cublas = timeit(lambda: torch.nn.functional.linear(x16, w16), iters)

    # --- Marlin W8A16 vía Fp8LinearMethod (camino de producción) ---
    t_marlin = None
    try:
        from vllm.model_executor.layers.quantization.fp8 import (
            Fp8Config, Fp8LinearMethod)

        class Shim(torch.nn.Module):
            pass

        layer = Shim()
        method = Fp8LinearMethod(Fp8Config(
            weight_block_size=[128, 128], activation_scheme="dynamic",
            ignored_layers=None,
            is_checkpoint_fp8_serialized=True))
        method.create_weights(
            layer,
            input_size_per_partition=K,
            output_partition_sizes=[N],
            input_size=K,
            output_size=N,
            params_dtype=torch.float16,
        )
        layer = layer.to("cuda")
        with torch.no_grad():
            layer.weight.copy_(torch.clamp(
                torch.randn_like(layer.weight, dtype=torch.float32) * 0.05,
                -0.4, 0.4).to(torch.float8_e4m3fn))
            layer.weight_scale_inv.fill_(0.004)
        method.process_weights_after_loading(layer)
        out = method.apply(layer, x16, None)
        t_marlin = timeit(lambda: method.apply(layer, x16, None), iters)
        del layer, out
    except Exception:
        traceback.print_exc()

    # --- cutlass INT8 W8A8 ---
    t_int8 = None
    try:
        import vllm._custom_ops as ops
        wmax = w16.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        w_i8 = (w16 / wmax * 127).to(torch.int8)
        w_scales = (wmax / 127).to(torch.float16)  # [N,1]
        # b debe ser [K,N] column-major: desde w_i8 [N,K] row-major es
        # simplemente .t() (vista, sin copia) — verificado empíricamente.
        b_col = w_i8.t()
        a_i8 = (x16.float() / 127.0).to(torch.int8)
        # cutlass int8 en sm80 exige scales float32 (verificado empíricamente)
        a_scales = torch.ones(M, 1, device="cuda", dtype=torch.float32)
        b_scales = w_scales.t().contiguous().float()  # [1,N] fp32

        def run_i8():
            return ops.cutlass_scaled_mm(a_i8, b_col, a_scales, b_scales,
                                         torch.float16)

        run_i8()
        t_int8 = timeit(run_i8, iters)
    except Exception:
        traceback.print_exc()

    # bytes de pesos leídos por llamada (lo que domina en decode M chico)
    wb = {
        "cublas": K * N * 2,
        "marlin": K * N * 1,      # fp8 = 1 byte/peso
        "int8": K * N * 1,
    }
    row = f"M={M:5d} N={N:6d} K={K:5d} [{name:7s}] "
    row += f"cuBLAS={t_cublas:8.3f}ms ({wb['cublas']/t_cublas/1e6:7.1f}GB/s) "
    if t_marlin:
        row += f"| Marlin={t_marlin:7.3f}ms ({wb['marlin']/t_marlin/1e6:7.1f}GB/s) "
    else:
        row += "| Marlin=FAIL "
    if t_int8:
        row += f"| INT8={t_int8:7.3f}ms ({wb['int8']/t_int8/1e6:7.1f}GB/s)"
    else:
        row += "| INT8=FAIL"
    print(row, flush=True)


def init_single_tp():
    """Inicializa entorno mínimo para usar Fp8LinearMethod fuera del engine.

    Receta verificada empíricamente (ver KERNELS doc):
    1. bypass del global _current_vllm_config con stub (dtype/is_moe/is_quantized)
    2. init_distributed_environment + initialize_model_parallel (world_size=1)
    3. layer.to('cuda') ANTES de process_weights_after_loading (repack CUDA)
    """
    from types import SimpleNamespace

    import vllm.config.vllm as vcv
    from vllm.config import VllmConfig

    cfg = VllmConfig()
    cfg.model_config = SimpleNamespace(dtype=torch.float16, is_moe=False,
                                       is_quantized=True)
    vcv._current_vllm_config = cfg

    import vllm.distributed.parallel_state as ps
    ps.init_distributed_environment(
        world_size=1, rank=0, local_rank=0,
        distributed_init_method="tcp://127.0.0.1:29631",
        backend="gloo")
    ps.initialize_model_parallel(tensor_model_parallel_size=1)


def main():
    torch.manual_seed(0)
    init_single_tp()
    print("=== bench_gemm: cuBLAS fp16 vs Marlin W8A16(fp8blk) vs "
          "cutlass INT8 W8A8 ===", flush=True)
    for name, N, K in SHAPES:
        for M in MS:
            try:
                bench_one(name, N, K, M)
            except Exception:
                traceback.print_exc()


if __name__ == "__main__":
    sys.exit(main())
