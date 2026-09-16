#!/usr/bin/env python3
"""Perfilado FP8 vs Marlin — compara tiempo de capas FP8 con y sin Marlin.

Extiende bench_gemm.py añadiendo camino 'dequant + torch.mm' (FP8 sin Marlin)
para aislar el speedup de Marlin vs dequant naive.

Compara 4 caminos sobre los mismos (M,N,K) reales por rank TP=2:
  1. cuBLAS fp16            — checkpoint sin cuantizar (2 bytes/peso)
  2. Dequant FP8 -> fp16 + torch.mm — FP8 sin Marlin (costo dequant + GEMM fp16)
  3. Marlin W8A16 (FP8 block) — prod actual en sm_86 via Fp8LinearMethod
  4. cutlass INT8 W8A8      — INT8 tensor cores Ampere (si disponible)

Shapes: Qwen3-30B-A3B ~27B por rank: qkv 4096x5120, o 5120x3072, gate_up 17408x5120,
down 5120x8704, lm_head 124160x5120. M en {1,40,1664} (single decode, MTPx10, prefill chunk).
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
MS = [1, 40, 1664]
ITERS = {1: 100, 40: 50, 1664: 20}

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
    return s.elapsed_time(e) / iters

def bench_one(name, N, K, M):
    iters = ITERS[M]
    x16 = torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.05

    # --- 1. cuBLAS fp16 ---
    w16 = torch.randn(N, K, device="cuda", dtype=torch.float16) * 0.02
    t_cublas = timeit(lambda: torch.nn.functional.linear(x16, w16), iters)

    # --- 2. Dequant FP8 + torch.mm (sin Marlin) ---
    # Simula peso FP8 por-tensor (escala unica) + dequant a fp16 + GEMM.
    # Este es el baseline "FP8 sin kernel fusionado".
    t_dequant = None
    try:
        w_fp8 = torch.randint(-112, 112, (N, K), device="cuda", dtype=torch.int8).view(torch.float8_e4m3fn) if hasattr(torch, 'float8_e4m3fn') else None
        # fallback si no hay float8: usar int8 simulado
        if w_fp8 is None or w_fp8.dtype != torch.float8_e4m3fn:
            # construir via float8_e4m3fn si disponible, sino simular con fp16*scale
            w_f32 = torch.randn(N, K, device="cuda", dtype=torch.float32) * 0.02
            w_fp8 = w_f32.clamp(-0.4, 0.4).to(torch.float8_e4m3fn) if hasattr(torch, 'float8_e4m3fn') else w_f32.to(torch.float16)
        scale = 0.004  # ~ inverso de 250, tipico en checkpoints FP8 block 128
        def run_dequant():
            # dequant: cast fp8->fp16 * scale, luego linear
            w_deq = w_fp8.to(torch.float16) * scale if w_fp8.dtype == torch.float8_e4m3fn else w_fp8
            return torch.nn.functional.linear(x16, w_deq)
        run_dequant()
        t_dequant = timeit(run_dequant, iters)
    except Exception:
        traceback.print_exc()
        t_dequant = None

    # --- 3. Marlin W8A16 via Fp8LinearMethod ---
    t_marlin = None
    try:
        from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
        import vllm.config.vllm as vcv
        from vllm.config import VllmConfig
        from types import SimpleNamespace
        # ya inicializado en init_single_tp, pero no re-inicializar
        class Shim(torch.nn.Module):
            pass
        layer = Shim()
        # Intentar dos firmas distintas (v0.23 vs main)
        try:
            cfg = Fp8Config(weight_block_size=[128,128], activation_scheme="dynamic", ignored_layers=None, is_checkpoint_fp8_serialized=True)
        except TypeError:
            cfg = Fp8Config(is_checkpoint_fp8_serialized=True)
        method = Fp8LinearMethod(cfg)
        # create_weights firma variante
        try:
            method.create_weights(layer, input_size_per_partition=K, output_partition_sizes=[N], input_size=K, output_size=N, params_dtype=torch.float16)
        except TypeError as e:
            # firma nueva: (layer, input_size_per_partition, output_partition_sizes, params_dtype)
            method.create_weights(layer, input_size_per_partition=K, output_partition_sizes=[N], params_dtype=torch.float16)
        layer = layer.to("cuda")
        with torch.no_grad():
            # weight es float8_e4m3fn, rellenar
            if hasattr(layer, 'weight') and layer.weight.dtype == torch.float8_e4m3fn:
                layer.weight.copy_(torch.clamp(torch.randn_like(layer.weight, dtype=torch.float32)*0.05, -0.4, 0.4).to(torch.float8_e4m3fn))
            elif hasattr(layer, 'weight'):
                layer.weight.copy_(torch.randn_like(layer.weight)*0.02)
            if hasattr(layer, 'weight_scale_inv'):
                layer.weight_scale_inv.fill_(0.004)
            if hasattr(layer, 'weight_scale'):
                layer.weight_scale.fill_(0.004)
        method.process_weights_after_loading(layer)
        out = method.apply(layer, x16, None)
        t_marlin = timeit(lambda: method.apply(layer, x16, None), iters)
        del layer, out
    except Exception:
        traceback.print_exc()

    # --- 4. cutlass INT8 W8A8 ---
    t_int8 = None
    try:
        import vllm._custom_ops as ops
        # amax correcto: torch.amax o .abs().max().values
        wmax = w16.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) if hasattr(w16.abs(), 'amax') else w16.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
        w_i8 = (w16 / wmax * 127).to(torch.int8)
        w_scales = (wmax / 127).to(torch.float16)
        b_col = w_i8.t().contiguous()  # [K,N]
        # activar quant dinamica simple para A
        a_abs_max = x16.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) if hasattr(x16.abs(), 'amax') else x16.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
        a_scales_f32 = (a_abs_max / 127).float()  # [M,1] fp32
        a_i8 = (x16 / a_abs_max * 127).to(torch.int8)
        b_scales = w_scales.t().contiguous().float()  # [1,N] fp32
        # cutlass en sm80 exige scales fp32
        def run_i8():
            return ops.cutlass_scaled_mm(a_i8, b_col, a_scales_f32, b_scales, torch.float16)
        run_i8()
        t_int8 = timeit(run_i8, iters)
    except Exception:
        traceback.print_exc()

    wb = {"cublas": K*N*2, "dequant": K*N*1, "marlin": K*N*1, "int8": K*N*1}
    row = f"M={M:5d} N={N:6d} K={K:5d} [{name:7s}] cuBLAS={t_cublas:7.3f}ms"
    if t_dequant:
        row += f" | Dequant={t_dequant:7.3f}ms"
    else:
        row += f" | Dequant=FAIL"
    if t_marlin:
        row += f" | Marlin={t_marlin:7.3f}ms ({wb['marlin']/t_marlin/1e6:6.1f}GB/s)"
    else:
        row += f" | Marlin=FAIL"
    if t_int8:
        row += f" | INT8={t_int8:7.3f}ms"
    else:
        row += f" | INT8=FAIL"
    # speedup relativo
    if t_marlin and t_cublas:
        row += f" | Marlin/cuBLAS={t_marlin/t_cublas:.2f}x"
    if t_dequant and t_marlin:
        row += f" Marlin/dequant={t_marlin/t_dequant:.2f}x"
    print(row, flush=True)

def init_single_tp():
    from types import SimpleNamespace
    import vllm.config.vllm as vcv
    from vllm.config import VllmConfig
    try:
        cfg = VllmConfig()
        cfg.model_config = SimpleNamespace(dtype=torch.float16, is_moe=False, is_quantized=True)
        vcv._current_vllm_config = cfg
    except Exception:
        pass
    import vllm.distributed.parallel_state as ps
    try:
        ps.init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="tcp://127.0.0.1:29631", backend="gloo")
        ps.initialize_model_parallel(tensor_model_parallel_size=1)
    except Exception:
        pass

def main():
    torch.manual_seed(0)
    try:
        init_single_tp()
    except Exception:
        traceback.print_exc()
    print("=== profile_fp8_vs_marlin: cuBLAS vs Dequant(FP8->fp16) vs Marlin vs INT8 ===", flush=True)
    for name, N, K in SHAPES:
        for M in MS:
            try:
                bench_one(name, N, K, M)
            except Exception:
                traceback.print_exc()

if __name__ == "__main__":
    sys.exit(main())
