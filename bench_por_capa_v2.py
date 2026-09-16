#!/usr/bin/env python3
"""
bench_por_capa_v2 — optimizado memoria para GPU con prod ocupando ~23GB.
- Usa generación directa int8 sin pasar por float32 intermedia.
- Separa sin y con para no tener ambos en memoria simultáneo.
- Usa expandable_segments y empty_cache agresivo.
"""
import csv, sys, traceback
from pathlib import Path
import torch
import torch.nn.functional as F

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
MS = [1,8,512,1664]
WARMUP=10
ITERS=50
OUT_DTYPE=torch.bfloat16

def timeit(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s=torch.cuda.Event(enable_timing=True)
    e=torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e)/iters

def bench_one(capa,N,K,M):
    # --- sin ---
    t_sin=float("nan")
    try:
        x_bf16=torch.randn(M,K, device="cuda", dtype=torch.bfloat16)*0.05
        w_bf16=torch.randn(N,K, device="cuda", dtype=torch.bfloat16)*0.02
        _=F.linear(x_bf16,w_bf16)
        torch.cuda.synchronize()
        def run_sin(): return F.linear(x_bf16,w_bf16)
        t_sin=timeit(run_sin)
        del x_bf16,w_bf16
        torch.cuda.empty_cache()
    except Exception as ex:
        print(f"[{capa} M={M}] sin FAIL {ex}", flush=True)
        traceback.print_exc()
        t_sin=float("nan")
        try: del x_bf16,w_bf16
        except: pass
        torch.cuda.empty_cache()

    # --- con ---
    t_con=float("nan")
    try:
        import vllm._custom_ops as ops
        # generar directamente int8 sin pasar por float duplicado
        # b_col pre-calculado: w_i8 [N,K] int8
        w_i8=torch.randint(-127,127,(N,K), device="cuda", dtype=torch.int8)
        b_col=w_i8.t()  # [K,N] vista
        b_scales=torch.empty(1,N, device="cuda", dtype=torch.float32).uniform_(0.002,0.008)
        a_i8=torch.randint(-127,127,(M,K), device="cuda", dtype=torch.int8)
        a_scales=torch.empty(M,1, device="cuda", dtype=torch.float32).uniform_(0.004,0.01)
        out=ops.cutlass_scaled_mm(a_i8,b_col,a_scales,b_scales,OUT_DTYPE)
        torch.cuda.synchronize()
        del out
        def run_con(): return ops.cutlass_scaled_mm(a_i8,b_col,a_scales,b_scales,OUT_DTYPE)
        t_con=timeit(run_con)
        del w_i8,b_col,b_scales,a_i8,a_scales
        torch.cuda.empty_cache()
    except Exception as ex:
        print(f"[{capa} M={M}] con FAIL {ex}", flush=True)
        traceback.print_exc()
        t_con=float("nan")
        torch.cuda.empty_cache()
    try:
        sp=t_sin/t_con if t_con==t_con and t_con>0 and t_sin==t_sin else float("nan")
    except: sp=float("nan")
    return t_sin,t_con,sp

def main():
    print("=== bench_por_capa v2 optimizado memoria ===", flush=True)
    print(f"torch {torch.__version__} cuda {torch.version.cuda} dev {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO'}", flush=True)
    if not torch.cuda.is_available():
        return 1
    # info memoria
    free,total=torch.cuda.mem_get_info()
    print(f"GPU mem free {free/1e9:.2f} GiB total {total/1e9:.2f} GiB", flush=True)
    import os
    print(f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF','')}", flush=True)
    candidates=[Path("workshop/ox_alpha/results/bench_por_capa.csv"),Path("/workspace/workshop/ox_alpha/results/bench_por_capa.csv"),Path("/tmp/bench_por_capa.csv"),Path("/tmp/opencode/bench_por_capa.csv")]
    write_paths=[]
    for p in candidates:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.parent.exists():
                write_paths.append(p)
        except: pass
    # dedup
    uniq=[]
    seen=set()
    for p in write_paths:
        s=str(p.resolve()) if p.exists() else str(p)
        if s not in seen:
            seen.add(s); uniq.append(p)
    write_paths=uniq
    print(f"write_paths {write_paths}", flush=True)
    header=["capa","K","N","M","tiempo_sin_ms","tiempo_con_ms","speedup"]
    handles=[]
    writers=[]
    for p in write_paths:
        try:
            f=open(p,"w",newline="")
            w=csv.writer(f); w.writerow(header); handles.append(f); writers.append(w)
            print(f"[csv] abierto {p}", flush=True)
        except Exception as e: print(f"fail {p} {e}", flush=True)
    if not writers: return 1
    results=[]
    for capa,N,K in LAYERS:
        for M in MS:
            print(f"\n--- {capa:12s} K={K:5d} N={N:5d} M={M:4d} ---", flush=True)
            free,total=torch.cuda.mem_get_info()
            print(f"  mem free {free/1e6:.0f} MB before", flush=True)
            t_sin,t_con,sp=bench_one(capa,N,K,M)
            ts=f"{t_sin:.4f}" if t_sin==t_sin else "nan"
            tc=f"{t_con:.4f}" if t_con==t_con else "nan"
            sps=f"{sp:.2f}x" if sp==sp else "nan"
            print(f"  sin={ts} ms con={tc} ms speedup={sps}", flush=True)
            row=[capa,K,N,M,f"{t_sin:.6f}" if t_sin==t_sin else "",f"{t_con:.6f}" if t_con==t_con else "",f"{sp:.4f}" if sp==sp else ""]
            results.append((capa,K,N,M,t_sin,t_con,sp))
            for w in writers: w.writerow(row)
            for f in handles: f.flush()
    for f in handles: f.close()
    print("\n"+"="*110, flush=True)
    for capa,N,K in LAYERS:
        rows=[r for r in results if r[0]==capa]
        rows.sort(key=lambda x: x[3])
        line=f"{capa:12s} | {K:5d} {N:5d} |"
        maxsp=0
        for _,K2,N2,M,ts,tc,sp in rows:
            if sp==sp and sp>maxsp: maxsp=sp
            if tc==tc and ts==ts and tc>0:
                line+=f" {sp:5.2f}x({ts:4.2f}/{tc:4.2f})"
            else: line+=f" {'nan':>14s}"
        line+=f" | {maxsp:.2f}x"
        print(line, flush=True)
    for p in write_paths:
        try: print(f"\n[{p}]\n{Path(p).read_text()[:3000]}", flush=True)
        except: pass
    print(f"\n=== DONE {len(results)} filas ===", flush=True)
    return 0
if __name__=="__main__": sys.exit(main())
