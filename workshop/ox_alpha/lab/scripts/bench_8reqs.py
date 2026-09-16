#!/usr/bin/env python3
"""
bench_8reqs — benchmark multi-request M=8 capa por capa y total, con y sin PN110.

Mide con M=8 (8 requests concurrentes, 1 token cada uno, batch 8) el tiempo por capa
para cada tipo de capa (q_proj, k_proj, v_proj, o_proj, in_proj_qkv, in_proj_z,
out_proj, gate_proj, up_proj, down_proj) con y sin PN110
  sin = torch.nn.functional.linear (bf16)
  con = cutlass_scaled_mm (W8A8 int8)  == PN110

Para cada capa mide tiempo por forward con M=8, y calcula tiempo total sumando
las 64 capas del modelo (48 GDN*3 +16 Full*4 +64 MLP*3 =400 lineares):
  total = sum_capa( t_capa * count_capa )  donde count por tipo:
    q/k/v/o =16, in_qkv/in_z/out =48, gate/up/down =64

Guarda en workshop/ox_alpha/results/bench_8reqs.csv con columnas:
  capa,K,N,M,tiempo_sin_ms,tiempo_con_ms,speedup,tiempo_total_80capas_ms
donde tiempo_total_80capas_ms es la contribución de ESE tipo al total
(tiempo_con * count). Al final imprime totales extrapolados y comparación con M=1.

Ejecución en GPU (timeout 60s):
  docker run --rm --gpus all -v $(pwd)/workshop/ox_alpha/lab/scripts/bench_8reqs.py:/tmp/bench_8reqs.py:ro \
    -v $(pwd)/workshop/ox_alpha/results:/tmp/out --entrypoint bash vllm/vllm-openai:v0.23.0 \
    -c "python3 /tmp/bench_8reqs.py 2>&1 | tee /tmp/out/bench_8reqs.log"
"""
import csv, sys, traceback
from pathlib import Path
import torch
import torch.nn.functional as F

# ── capas del modelo Qwen3.8-27B híbrido por rank TP=2 (igual que bench_por_capa_v2) ──
LAYERS = [
    ("q_proj",       12288, 5120),
    ("k_proj",        1024, 5120),
    ("v_proj",        1024, 5120),
    ("o_proj",        5120, 6144),
    ("in_proj_qkv",  10240, 5120),
    ("in_proj_z",     6144, 5120),
    ("out_proj",      5120, 6144),
    ("gate_proj",    17408, 5120),
    ("up_proj",      17408, 5120),
    ("down_proj",     5120, 17408),
]

# Counts reales para extrapolación total (400 lineares)
COUNTS = {
    "q_proj": 16,
    "k_proj": 16,
    "v_proj": 16,
    "o_proj": 16,
    "in_proj_qkv": 48,
    "in_proj_z": 48,
    "out_proj": 48,
    "gate_proj": 64,
    "up_proj": 64,
    "down_proj": 64,
}

M = 8  # 8 requests concurrentes, 1 token cada uno, batch 8
MS_COMPARE = [1, 8]  # para tabla comparativa M=1 vs M=8 (lee bench_por_capa.csv si existe)
WARMUP = 10
ITERS = 50
OUT_DTYPE = torch.bfloat16

def timeit(fn, warmup=WARMUP, iters=ITERS):
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

def bench_one(capa, N, K, M):
    # sin: F.linear bf16
    t_sin = float("nan")
    try:
        x_bf16 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.05
        w_bf16 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
        _ = F.linear(x_bf16, w_bf16)
        torch.cuda.synchronize()
        def run_sin(): return F.linear(x_bf16, w_bf16)
        t_sin = timeit(run_sin)
        del x_bf16, w_bf16
        torch.cuda.empty_cache()
    except Exception as ex:
        print(f"[{capa} M={M}] sin FAIL {ex}", flush=True)
        traceback.print_exc()
        t_sin = float("nan")
        try: del x_bf16, w_bf16
        except: pass
        torch.cuda.empty_cache()

    t_con = float("nan")
    try:
        import vllm._custom_ops as ops
        w_i8 = torch.randint(-127, 127, (N, K), device="cuda", dtype=torch.int8)
        b_col = w_i8.t()  # [K,N] vista column-major sin copy
        b_scales = torch.empty(1, N, device="cuda", dtype=torch.float32).uniform_(0.002, 0.008)
        a_i8 = torch.randint(-127, 127, (M, K), device="cuda", dtype=torch.int8)
        a_scales = torch.empty(M, 1, device="cuda", dtype=torch.float32).uniform_(0.004, 0.01)
        out = ops.cutlass_scaled_mm(a_i8, b_col, a_scales, b_scales, OUT_DTYPE)
        torch.cuda.synchronize()
        del out
        def run_con(): return ops.cutlass_scaled_mm(a_i8, b_col, a_scales, b_scales, OUT_DTYPE)
        t_con = timeit(run_con)
        del w_i8, b_col, b_scales, a_i8, a_scales
        torch.cuda.empty_cache()
    except Exception as ex:
        print(f"[{capa} M={M}] con FAIL {ex}", flush=True)
        traceback.print_exc()
        t_con = float("nan")
        torch.cuda.empty_cache()
    try:
        sp = t_sin / t_con if t_con == t_con and t_con > 0 and t_sin == t_sin else float("nan")
    except:
        sp = float("nan")
    return t_sin, t_con, sp

def main():
    print("=== bench_8reqs: M=8 capa por capa con y sin PN110 (cutlass_scaled_mm) ===", flush=True)
    print(f"torch {torch.__version__} cuda {torch.version.cuda} dev {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA'}", flush=True)
    if not torch.cuda.is_available():
        return 1
    free, total = torch.cuda.mem_get_info()
    print(f"GPU mem free {free/1e9:.2f} GiB total {total/1e9:.2f} GiB", flush=True)
    import os
    print(f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF','')}", flush=True)
    try:
        import vllm._custom_ops as ops
        print(f"cutlass_scaled_mm available: {hasattr(ops,'cutlass_scaled_mm')}", flush=True)
    except Exception as e:
        print(f"vllm._custom_ops error: {e}", flush=True)
        traceback.print_exc()
        return 1

    # candidatos de escritura (replica bench_por_capa multiplataforma)
    candidates = [
        Path("workshop/ox_alpha/results/bench_8reqs.csv"),
        Path("/workspace/workshop/ox_alpha/results/bench_8reqs.csv"),
        Path("/tmp/out/bench_8reqs.csv"),
        Path("/tmp/bench_8reqs.csv"),
        Path("/tmp/opencode/bench_8reqs.csv"),
        Path("bench_8reqs.csv"),
    ]
    write_paths = []
    for p in candidates:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.parent.exists():
                write_paths.append(p)
        except:
            pass
    uniq = []
    seen = set()
    for p in write_paths:
        s = str(p.resolve()) if p.exists() else str(p)
        if s not in seen:
            seen.add(s); uniq.append(p)
    write_paths = uniq
    print(f"write_paths {write_paths}", flush=True)

    header = ["capa","K","N","M","tiempo_sin_ms","tiempo_con_ms","speedup","tiempo_total_80capas_ms"]
    # tiempo_total_80capas_ms = contribución al total con PN110 (tiempo_con * count)
    # También guardamos totales sin en log; la columna es CON para comparar speedups.
    handles = []
    writers = []
    for p in write_paths:
        try:
            f = open(p, "w", newline="")
            w = csv.writer(f); w.writerow(header); handles.append(f); writers.append(w)
            print(f"[csv] abierto {p}", flush=True)
        except Exception as e:
            print(f"fail {p} {e}", flush=True)
    if not writers:
        print("ERROR: no se pudo abrir ningún csv", flush=True)
        return 1

    results = []
    for capa, N, K in LAYERS:
        print(f"\n--- {capa:12s} K={K:5d} N={N:5d} M={M:2d} ---", flush=True)
        free, total = torch.cuda.mem_get_info()
        print(f"  mem free {free/1e6:.0f} MB before", flush=True)
        t_sin, t_con, sp = bench_one(capa, N, K, M)
        cnt = COUNTS.get(capa, 1)
        # tiempo_total_80capas_ms como contribución con PN110
        t_total_con = t_con * cnt if t_con == t_con else float("nan")
        t_total_sin = t_sin * cnt if t_sin == t_sin else float("nan")
        ts = f"{t_sin:.4f}" if t_sin == t_sin else "nan"
        tc = f"{t_con:.4f}" if t_con == t_con else "nan"
        sps = f"{sp:.2f}x" if sp == sp else "nan"
        print(f"  sin={ts} ms con={tc} ms speedup={sps}  count={cnt}  total_con {t_total_con:.3f} ms  total_sin {t_total_sin:.3f} ms", flush=True)
        row = [capa, K, N, M, f"{t_sin:.6f}" if t_sin==t_sin else "", f"{t_con:.6f}" if t_con==t_con else "", f"{sp:.4f}" if sp==sp else "", f"{t_total_con:.6f}" if t_total_con==t_total_con else ""]
        results.append((capa, K, N, t_sin, t_con, sp, cnt, t_total_sin, t_total_con))
        for w in writers: w.writerow(row)
        for f in handles: f.flush()

    for f in handles: f.close()

    # ── totales extrapolados ──
    total_sin = sum(r[7] for r in results if r[7]==r[7])  # sum t_sin*cnt
    total_con = sum(r[8] for r in results if r[8]==r[8])
    total_speedup = total_sin/total_con if total_con>0 else float("nan")
    # también total si fuera 80 capas idénticas por tipo (80*t) para referencia solicitada literal
    total_sin_80 = sum(r[3]*80 if r[3]==r[3] else 0 for r in results)
    total_con_80 = sum(r[4]*80 if r[4]==r[4] else 0 for r in results)

    print("\n" + "="*110, flush=True)
    print(f"M={M} — Tabla por capa (speedup PN110) + contribución al total 400 lineares (64 capas reales)", flush=True)
    print(f"{'capa':12s} | {'K':5s} {'N':5s} | count | sin(ms) con(ms) speedup | total_sin(ms) total_con(ms)", flush=True)
    print("-"*110, flush=True)
    for capa, K, N, t_sin, t_con, sp, cnt, t_sin_tot, t_con_tot in results:
        ts = f"{t_sin:.4f}" if t_sin==t_sin else "nan"
        tc = f"{t_con:.4f}" if t_con==t_con else "nan"
        sps = f"{sp:.2f}x" if sp==sp else "nan"
        tot_s = f"{t_sin_tot:.2f}" if t_sin_tot==t_sin_tot else "nan"
        tot_c = f"{t_con_tot:.2f}" if t_con_tot==t_con_tot else "nan"
        print(f"{capa:12s} | {K:5d} {N:5d} | {cnt:3d}   | {ts:6s} {tc:6s} {sps:6s} | {tot_s:6s} {tot_c:6s}", flush=True)

    print("\n--- TOTAL extrapolado (400 lineares = 48*3 GDN +16*4 Full +64*3 MLP) ---", flush=True)
    print(f"  total SIN (bf16)  M={M}: {total_sin:.3f} ms por forward de 1 token batch {M}", flush=True)
    print(f"  total CON (int8)  M={M}: {total_con:.3f} ms por forward de 1 token batch {M}", flush=True)
    print(f"  speedup total 400 lineares: {total_speedup:.2f}x  (ahorro {(total_sin-total_con):.2f} ms = {(1-1/total_speedup)*100:.1f}%)", flush=True)
    print(f"  total SIN si 80*t por capa (10 tipos*80): {total_sin_80:.1f} ms  CON {total_con_80:.1f} ms  speedup {total_sin_80/total_con_80:.2f}x (ref simple 80)", flush=True)
    # por transformer layer promedio
    avg_per_layer_sin = total_sin/64
    avg_per_layer_con = total_con/64
    print(f"  per-transformer-layer (64 layers): sin {avg_per_layer_sin:.3f} ms  con {avg_per_layer_con:.3f} ms", flush=True)

    # comparación con M=1 si existe bench_por_capa.csv
    for p in [Path("workshop/ox_alpha/results/bench_por_capa.csv"), Path("/tmp/out/bench_por_capa.csv"), Path("bench_por_capa.csv")]:
        if p.exists():
            print(f"\n[compare] leyendo {p} para contraste M=1 vs M=8", flush=True)
            try:
                rows1 = {}
                with open(p) as f:
                    r = csv.DictReader(f)
                    for row in r:
                        if int(row["M"])==1:
                            rows1[row["capa"]] = row
                        elif int(row["M"])==8 and row["capa"] not in rows1:
                            pass
                print(f"{'capa':12s} | M=1 sin/con speedup | M=8 sin/con speedup | delta speedup", flush=True)
                for capa, K, N, ts8, tc8, sp8, cnt, _, _ in results:
                    r1 = rows1.get(capa)
                    if r1:
                        try:
                            s1 = float(r1["tiempo_sin_ms"]) if r1["tiempo_sin_ms"] else float("nan")
                            c1 = float(r1["tiempo_con_ms"]) if r1["tiempo_con_ms"] else float("nan")
                            sp1 = s1/c1 if c1 and c1==c1 else float("nan")
                            sp1s = f"{sp1:.2f}x" if sp1==sp1 else "nan"
                        except: sp1s="nan"
                    else: sp1s="nan"
                    sp8s = f"{sp8:.2f}x" if sp8==sp8 else "nan"
                    print(f"{capa:12s} | {sp1s:14s} | {sp8s:14s} |", flush=True)
                break
            except Exception as e:
                print(f"compare fail {e}", flush=True)
                traceback.print_exc()
            break

    for p in write_paths:
        try: print(f"\n[{p}]\n{Path(p).read_text()[:5000]}", flush=True)
        except: pass
    print(f"\n=== DONE {len(results)} filas M={M} ===", flush=True)
    print(f"=== TOTAL M={M} sin={total_sin:.3f}ms con={total_con:.3f}ms speedup={total_speedup:.2f}x ===", flush=True)
    return 0

if __name__=="__main__":
    sys.exit(main())
