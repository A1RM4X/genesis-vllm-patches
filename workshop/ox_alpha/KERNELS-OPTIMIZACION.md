# ⚡ Optimización de kernels del pipeline — análisis estático + benchmarks empíricos

> **Pregunta original**: dadas los avances en procesamiento CUDA de LLMs, ¿cómo
> se puede optimizar aún más el runner V1 que usamos? ¿Qué mejoras se pueden
> aplicar a todos los kernels del pipeline, uno por uno? ¿Alguno fusionable?
>
> | | |
> |---|---|
> | **Fecha** | 2026-08-24 |
> | **Método** | 2 exploraciones estáticas (alternativas por kernel + superficie de fusión) + **benchmarks empíricos en contenedor** con las formas EXACTAS del modelo (config.json del checkpoint) + A/B de backend de atención sobre el modelo real |
> | **Hardware** | 2× RTX 3090 (sm_86, Ampere: tensor cores INT8/FP16 sí, FP8/FP4 NO) |
> | **Complementa** | `CONTEXTO-INVESTIGACION.md`, `CIRCUITO-TOKEN.md`, `BACKPORT-V2.md` |

---

## 1. Formas reales del modelo (por rank, TP=2)

Del `config.json` del checkpoint: hidden 5120 · 64 capas (48 GDN + 16 full,
intervalo 4) · attn full 24Q/4KV × head_dim 256 → por rank 12Q/2KV ·
GDN 16K/48V × 128 → por rank 8K/24V · intermediate 17408 → 8704/rank ·
**vocab 248,320** (lm_head 124,160 × 5120 por rank ≈ 1.2 GB fp16).

| GEMM | N | K |
|:---|---:|---:|
| qkv_proj | 4096 | 5120 |
| o_proj | 5120 | 3072 |
| gate_up | 17408 | 5120 |
| down | 5120 | 8704 |
| lm_head | 124160 | 5120 |

M=40 (decode MTP 10 seqs × 4) y M=1664 (chunk prefill).

---

## 2. 🏆 Resultados empíricos del shootout GEMM

`lab/scripts/bench_gemm.py` — cuBLAS fp16 vs **Marlin W8A16 (lo que corre
hoy)** vs cutlass INT8 W8A8 (tensor cores INT8 de Ampere):

### Decode (M=40) — bandwidth-bound

| Capa | cuBLAS fp16 | Marlin (hoy) | INT8 W8A8 |
|:---|---:|---:|---:|
| qkv | 0.090 ms | 0.047 ms | **0.043 ms** |
| o | 0.064 ms | 0.039 ms | **0.035 ms** |
| gate_up | 0.323 ms | **0.147 ms** | 0.153 ms |
| down | 0.168 ms | 0.107 ms | **0.075 ms** |
| lm_head | 2.302 ms | **1.277 ms** | 1.366 ms |

### Prefill (M=1664) — compute-bound

| Capa | cuBLAS fp16 | Marlin (hoy) | INT8 W8A8 | INT8 vs Marlin |
|:---|---:|---:|---:|---:|
| qkv | 1.094 ms | 1.105 ms | **0.383 ms** | **2.9×** |
| o | 0.867 ms | 0.862 ms | **0.289 ms** | **3.0×** |
| gate_up | 4.954 ms | 5.529 ms | **1.613 ms** | **3.4×** |
| down | 2.539 ms | 2.427 ms | **0.751 ms** | **3.2×** |
| lm_head | 33.2 ms | 40.8 ms | **14.4 ms** | **2.8×** |

### Conclusiones

1. **Marlin ya es el óptimo en decode** (450–600 GB/s de pesos, al nivel del
   techo práctico de la 3090). No hay nada que ganar en decode por el lado
   GEMM: el límite es el ancho de banda, no el cómputo.
2. **El prefill es donde vive la fruta**: INT8 W8A8 con tensor cores Ampere es
   **~3× más rápido que Marlin** en todas las capas (~77 TFLOPS INT8 vs ~20
   de Marlin emulando 8→16 bit). Como el prefill de este stack está pegado al
   techo de cómputo (~770 tok/s medidos en el compose), un requant
   FP8→INT8 W8A8 (per-channel/per-block calibrado) implicaría **~2–3× más
   velocidad de prefill**. Costo: precisión a validar (el modelo ya es FP8;
   int8 per-channel típicamente cuesta <1% en benchmarks).
3. Marlin en prefill es incluso **peor que cuBLAS fp16** en capas grandes
   (lm_head: 40.8 vs 33.2 ms) — la emulación sin tensor cores INT8 cuesta cara
   cuando el GEMM es compute-bound.

---

## 3. Otros benchmarks empíricos

### Sampler (vocab 248,320 real)

| Kernel | Tiempo |
|:---|---:|
| argmax [40, 248320] (greedy) | 0.066 ms |
| triton top-k/top-p | 0.593 ms |
| flashinfer top-k/top-p | 0.618 ms |
| **softmax fp32 [160, 248320]** (path aleatorio del rejection sampler) | **0.749 ms** |

- flashinfer ≈ triton (empate): el env `VLLM_USE_FLASHINFER_SAMPLER=1` no da
  ganancia medible; es indiferente.
- **El softmax del rejection path cuesta 0.75 ms/paso** y se calcula para
  TODOS los tokens aunque solo se necesite en posiciones rechazadas —
  candidato a optimización (cómputo condicionado por máscara de rechazo).

### FLA/GDN (T=1664, H=24, K=V=128 por rank)

| Config | Pipeline prefill completo |
|:---|---:|
| ieee / BKV default | 0.436 ms |
| tf32 / BKV default | 0.443 ms |
| ieee / BKV ampliado | 0.434 ms |
| tf32 / BKV ampliado | 0.444 ms |

**Veredicto honesto: sin diferencias** (todo dentro del ruido ±2%). Las
hipótesis tempranas de `FLA_TRIL_PRECISION=tf32` y ampliación de `BKV_LIST`
para sm_86 **quedan descartadas** para estas formas: el pipeline GDN completo
es 0.44 ms × 48 capas ≈ 21 ms por chunk de prefill (~1.2% del paso) — no es
cuello de botella ni por asomo. El prefill está dominado por los GEMM (§2).

### Copias de estado mamba

- `batch_memcpy` Triton (lo que usa V1): **0.362 ms para 96 copias de 786 KB
  (75.5 MB) = 417 GB/s** — excelente, cerca del techo D2D.
- `cuMemcpyBatchAsync` (DMA batch): rc=1 en este driver con numAttrs=0; no se
  pudo comparar sin replicar el patrón exacto de attrs del connector. Dado el
  417 GB/s del Triton, la ganancia potencial es acotada — **prioridad baja**.

### A/B de backend de atención (modelo real completo)

| Backend | Resultado |
|:---|:---|
| FLASHINFER (producción) | 62.1 tok/s (4 reqs × 64 tok) — baseline estable |
| TRITON_ATTN | **CRASH** en `do_kv_cache_update` (:750) con spec-decode MTP — incompatible en v0.23 |

**TRITON_ATTN descartado** para esta config. FLASHINFER es la única
alternativa con KV fp8_e4m3 funcional en sm_86 + spec-decode.

**Hallazgo colateral del A/B**: el engine degrada automáticamente
`cudagraph_mode` a **PIECEWISE** con FlashInfer + spec-decode
(*"FULL_AND_PIECEWISE is not supported with spec-decode for attention
backend FlashInferBackend (support: UNIFORM_SINGLE_TOKEN_DECODE)"*). El P100
de Genesis ("CUDAGraphs FULL") no puede cumplir su promesa en esta
combinación — los FULL graphs no se capturan.

---

## 4. Inventario kernel-por-kernel con estado y acción

| # | Kernel (pipeline actual) | Estado | Acción recomendada |
|:--|:---|:---|:---|
| 1 | **Marlin W8A16** (todos los Linear FP8) | Decode óptimo; prefill 3× lejos del techo | **Requant FP8→INT8 W8A8** (proyecto mayor, ganancia ~2-3× prefill). Interino: nada — ya es lo mejor disponible para FP8 |
| 2 | **FlashInfer FA2** (atención full, KV fp8 in-kernel) | Única opción viable en sm_86+spec | Mantener. TRTLLM/cuDNN exigen SM90+/SM100+ |
| 3 | **FLA Triton pipeline** (GDN prefill) | 1.2% del paso, insensible a tuning | Nada que hacer (descartado empíricamente) |
| 4 | **fused_sigmoid_gating + causal_conv1d_update** (GDN decode) | Ya fusionados (gating+estado; conv+spec) | Nada |
| 5 | **cuBLAS fp16 lm_head** | 1.28-2.3 ms/paso (vocab gigante) | PN77 ya guarda fp8; el compute sigue fp16. Un lm_head Marlin/W8A16 ahorraría ~1 ms/paso en decode — candidato menor |
| 6 | **PYNCCL all-reduce** | 12.8-16.1 ms en la ventana perfilada | Barrido NCCL_ALGO/PROTO con P2P PIX+ReBAR (env, gratis); custom AR bloqueado por bug cumem×IPC (ver CONTEXTO §10) |
| 7 | **Sampler top-k/top-p** | Empate triton/flashinfer | Indiferente. Greedy (argmax) es 9× más barato — si los agentes usan temperature=0, ya va por ahí |
| 8 | **Rejection sampler** (MTP) | softmax fp32 0.75 ms/paso incondicional | **Candidato: softmax solo en posiciones rechazadas** (máscara) — parche Genesis factible |
| 9 | **batch_memcpy Triton** (estados mamba) | 417 GB/s | Nada (techo prácticamente) |
| 10 | **postprocess_mamba_fused_kernel** | Ya fusiona decisión+copia en GPU | Nada |
| 11 | **RMSNorm/SiluAndMul** | `native` (Inductor) / kernel CUDA único | Las fusiones norm+quant/act+quant del compilador NO aplican en W8A16 (no hay quant de activaciones). Con INT8 W8A8 sí se activarían `silu_and_mul_quant`-style (a validar) |

### Fusionables identificados (del análisis estático)

| Fusión | Estado |
|:---|:---|
| cumsum + kkt (GDN prefill) | FUSIONABLE — ahorraría 1 pase sobre [B,T,H]; impacto ~0 (GDN es 1.2% del paso) |
| conv1d + post_conv_prep (GDN) | FUSIONABLE — mismo veredicto: irrelevante en tiempo |
| all-reduce + RMSNorm | Existe (`AllReduceFusionPass` → flashinfer trtllm) pero exige SM90/100. En sm_86: solo vía custom AR (bloqueado) o SP forzado |
| norm+quant / act+quant | Solo con activaciones cuantizadas (W8A8) — se activarían solas tras un requant INT8 |

**Conclusión de fusión**: en el pipeline actual (W8A16) no queda NINGUNA
fusión de alto impacto disponible; las fusiones del compilador se activarían
como efecto colateral positivo del requant INT8.

---

## 5. Recomendaciones finales rankeadas

| # | Acción | Impacto | Esfuerzo | Tipo |
|:--|:---|:---|:---|:---|
| 1 | **Requant FP8→INT8 W8A8 del checkpoint** (calibrado, per-channel/block) — **DONE vía PN110 (2026-08-25): A/B prefill superado 5.56s vs 7.21s (-23%), tps 23.0 vs 17.8 (+29%)** | **~2-3× prefill** (medido -23% wall; swap INT8 `b_col` column-major 1:1, GDN excluido, chunked 512 per-channel) | Alto (requant + validación de calidad + boot con esquema compressed-tensors/quark int8) | **DONE** — Proyecto |
| 2 | **Barrido NCCL_ALGO/NCCL_PROTO** con P2P activo | Desconocido hasta medir (el AR era 31.8% en la era CNS) | Bajo (env vars) | Config |
| 3 | **Softmax del rejection sampler solo en rechazados** | ~0.7 ms/paso en path aleatorio | Medio (parche Triton) | Parche Genesis |
| 4 | **Backports V2** (ver BACKPORT-V2.md: PN-A/PN-B/PN-C...) | Latencia p99 / syncs ocultas | Medio | Parche Genesis |
| 5 | lm_head W8A16 (Marlin) además de PN77 | ~1 ms/paso decode | Bajo-medio | Parche Genesis |
| 6 | Cache disco del repack Marlin (boot) | Segundos por arranque | Bajo | Parche Genesis |

### Lo que NO vale la pena (descartado con evidencia)

- Tuning FLA (precisión/BKV): **cero efecto medible**, GDN es 1.2% del paso.
- TRITON_ATTN como backend: **crashea** con MTP en v0.23.
- Cambiar sampler (triton↔flashinfer): empate total.
- Optimizar batch_memcpy: ya a 417 GB/s.
- Cualquier cosa FP8-compute en sm_86: no existe tal hardware.

---

## 6. Detalles técnicos de los benchmarks (reproducibilidad)

- Contenedor: `vllm/vllm-openai:v0.23.0` (torch 2.11.0+cu130, triton 3.6.0,
  flashinfer 0.6.12+cubins, transformers 5.12.0).
- Timing: CUDA events, 5 warmup + 50 iters (M=40) / 20 iters (M=1664).
- Marlin vía `Fp8LinearMethod` real (create_weights → repack → apply) con
  pesos sintéticos fp8-bloque; receta para usarlo fuera del engine documentada
  en `bench_gemm.py::init_single_tp()` (bypass de `_current_vllm_config`,
  init distribuido gloo world_size=1, `layer.to('cuda')` antes del repack).
- cutlass INT8: `ops.cutlass_scaled_mm` con scales **float32** (exigencia
  sm80 verificada empíricamente) y `b = w.t()` (vista [K,N] column-major).
- Resultados crudos: `lab/results/benches/*.log`, A/B en
  `lab/results/attn_ab_*.log`.
