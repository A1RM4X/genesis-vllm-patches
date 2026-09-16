#!/usr/bin/env python3
"""secciones_traza.py — tiempo GPU por seccion del stack, lado a lado para N trazas.

Uso: secciones_traza.py <dir_a> <dir_b> ... [--otros]
Cada dir es el torch_profiler_dir de una corrida (se toma la traza del rank 0).
Normaliza por paso del engine (anotaciones execute_model) y reporta us/paso por seccion,
lanzamientos por paso, y los kernels mas pesados de cada lado.
"""
import collections, glob, gzip, json, os, sys

SECCIONES = [
    ("atencion: kernel", ("sk18h_batch", "sk18h_union", "sk18h_prep", "sk18h_salida", "batchprefill",
                          "batchdecode", "flashinfer", "unified_attention", "prefillwithkvcache",
                          "decodewithkvcache", "mergestates")),
    ("atencion: escritura KV", ("sk18h_escribir", "reshape_and_cache", "concat_and_cache")),
    ("GDN (recurrente+conv+cinta)", ("gated_delta", "fused_recurrent", "causal_conv1d", "conv1d", "fla_", "chunk_",
                               "l2norm", "gdn", "mamba", "ssm", "sigmoid", "wmma_tensorop_f16_s161616gemm",
                               "_k_escribir", "_k_spec", "_k_materializar")),
    ("lineales W4A8 (Marlin)", ("marlin",)),
    ("quant de activacion int8", ("per_token_quant", "scaled_int8", "quant")),
    ("activacion MLP (SiLU*mul)", ("act_and_mul", "silu")),
    ("GEMM cuBLAS/cutlass (lm_head, rotacion)", ("gemm", "cublas", "cutlass", "s16816", "ampere_fp16", "scaled_mm")),
    ("all-reduce NCCL", ("nccl", "allreduce", "all_reduce", "cross_device")),
    ("inductor fusionado (norm/rope/act)", ("triton_poi", "triton_red", "triton_per", "triton_tem")),
    ("norm/rope sueltos", ("rms_norm", "rmsnorm", "rotary", "layernorm")),
    ("muestreo / MTP verif.", ("sampl", "topk", "top_k", "argmax", "rejection", "softmax", "multinomial",
                               "gumbel", "exponential", "sort")),
    ("copias / elementwise / indices", ("copy", "elementwise", "index", "cat", "fill", "scatter", "gather",
                                        "memcpy", "vectorized", "reduce", "arange", "where")),
]


def seccion(nombre):
    l = nombre.lower()
    for s, claves in SECCIONES:
        if any(c in l for c in claves):
            return s
    return "otros"


def cargar(d):
    fs = sorted(glob.glob(os.path.join(d, "**", "*.json*"), recursive=True))
    fs0 = [f for f in fs if "rank0" in f or "rank_0" in f] or fs
    f = fs0[0]
    op = gzip.open if f.endswith(".gz") else open
    with op(f, "rt") as h:
        return json.load(h).get("traceEvents", []), f


def resumen(d):
    ev, f = cargar(d)
    ker = [e for e in ev if e.get("cat") in ("kernel", "gpu_user_annotation") and "dur" in e and e.get("cat") == "kernel"]
    # un eagle_prepare_inputs_padded_kernel por paso de decode con MTP; si no, anotaciones
    pasos = sum(1 for e in ker if "eagle_prepare_inputs_padded_kernel" in e["name"])
    if pasos == 0:
        pasos = sum(1 for e in ev if e.get("cat") == "gpu_user_annotation") // 2
    pasos = max(pasos, 1)
    por_s = collections.defaultdict(lambda: [0.0, 0])
    por_k = collections.defaultdict(lambda: [0.0, 0])
    for e in ker:
        s = seccion(e["name"])
        por_s[s][0] += e["dur"]; por_s[s][1] += 1
        por_k[(s, e["name"][:70])][0] += e["dur"]; por_k[(s, e["name"][:70])][1] += 1
    # tiempo de pared entre primer y ultimo kernel
    if ker:
        t0 = min(e["ts"] for e in ker); t1 = max(e["ts"] + e["dur"] for e in ker)
    else:
        t0 = t1 = 0
    return dict(archivo=f, pasos=pasos, s=por_s, k=por_k, total=sum(v[0] for v in por_s.values()),
                lanz=len(ker), pared=(t1 - t0))


def main():
    dirs = [a for a in sys.argv[1:] if not a.startswith("--")]
    R = [resumen(d) for d in dirs]
    nombres = [os.path.basename(d.rstrip("/")) for d in dirs]
    for n, r in zip(nombres, R):
        print(f"{n}: {r['archivo']} | {r['pasos']} pasos | {r['lanz']} lanzamientos | pared {r['pared']/1e3:.1f} ms")
    secs = [s for s, _ in SECCIONES] + ["otros"]
    ancho = 22
    print("\n| seccion | " + " | ".join(f"{n} us/paso (lanz/paso)" for n in nombres) + " |")
    print("|---|" + "---|" * len(nombres))
    for s in secs:
        fila = []
        for r in R:
            t, c = r["s"].get(s, [0.0, 0])
            fila.append(f"{t / r['pasos']:.0f} ({c / r['pasos']:.0f})")
        print(f"| {s} | " + " | ".join(fila) + " |")
    print("| **TOTAL GPU** | " + " | ".join(f"**{r['total'] / r['pasos']:.0f}** ({r['lanz'] / r['pasos']:.0f})" for r in R) + " |")
    print("| pared / paso | " + " | ".join(f"{r['pared'] / r['pasos']:.0f}" for r in R) + " |")
    for n, r in zip(nombres, R):
        print(f"\nTop kernels {n} (us/paso):")
        top = sorted(r["k"].items(), key=lambda kv: -kv[1][0])[:18]
        for (s, k), (t, c) in top:
            print(f"  {t / r['pasos']:8.0f}  x{c / r['pasos']:6.1f}  [{s}] {k}")
        if "--otros" in sys.argv:
            print("  -- otros:")
            for (s, k), (t, c) in sorted(((kk, v) for kk, v in r["k"].items() if kk[0] == "otros"), key=lambda kv: -kv[1][0])[:15]:
                print(f"  {t / r['pasos']:8.0f}  x{c / r['pasos']:6.1f}  {k}")


if __name__ == "__main__":
    main()
