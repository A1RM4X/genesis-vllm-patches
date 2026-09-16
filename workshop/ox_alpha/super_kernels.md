# Super Kernels Inline — Qwen3.5-27B Híbrido (GDN + Full Attention) en Ampere

> **Objetivo**: inventario exhaustivo de todos los tipos de capas del checkpoint `orcarouter/Qwen3.8-27B-Uncensored-FP8` y propuesta de **super kernels inline** — un kernel fusionado por cada variación de (geometría × tipo de atención × régime de quant) — maximizando uso de Tensor Cores INT8 en Ampere sin pagar pérdida innecesaria.
>
> | | |
> |---|---|
> | **Fecha** | 2026-08-25 |
> | **Modelo** | `orcarouter/Qwen3.8-27B-Uncensored-FP8` → `Qwen3_5ForConditionalGeneration` (`model_type: qwen3_5`) |
> | **Hardware target** | Ampere SM80/86 (RTX 3090): TC FP16/BF16/INT8 sí, **FP8 NO** (`mma.m16n8k32.s8`) |
> | **TP** | 2 (vocab/gemm shardeado) |
> | **KV cache dtype** | `fp8_e4m3` por defecto — `compose/docker-compose.qwen38-27b-fp8.yml:446-447` (`--kv-cache-dtype fp8_e4m3`) + `GENESIS_PN92_KV_DTYPE=fp8_e4m3` (`compose/docker-compose.qwen38-27b-fp8.yml:196`); FlashInfer FA2 KV fp8 in-kernel dequant (`CONTEXTO-INVESTIGACION.md:419-420`) |
> | **Fuente primaria** | `workshop/ox_alpha/fp8_a_int8_ampere.md:32-154` + `workshop/ox_alpha/KERNELS-OPTIMIZACION.md:16-32` + `workshop/ox_alpha/CONTEXTO-INVESTIGACION.md:7-9` + `assets/vllm/vllm/model_executor/models/qwen3_5.py:276-449` |

---

## 0. Resumen ejecutivo

Qwen3.5-27B **no es denso**: `workshop/ox_alpha/fp8_a_int8_ampere.md:42-46` declara 64 capas con patrón `3 GDN : 1 Full` (`full_attention_interval=4`):

- **48 capas GDN** (Gated DeltaNet, atención lineal recurrente + SSM) — índices `0,1,2, 4,5,6, …`
- **16 capas Full** (softmax, GQA 24Q/4KV × 256, `attn_output_gate`) — índices `3,7,11,…,63`
- **MLP denso idéntico** en las 64 (`intermediate 17408 = 136·128`, `hidden 5120 = 40·128`)
- **+1 capa Full+MTP** en `qwen3_5_mtp.py:59-159` (draft head, 17ª `o_proj` + 65ª `down_proj`)

De **1606 tensores**: 407 cuantizados FP8 bloque `128×128` E4M3 + 407 `weight_scale_inv` BF16 + 792 copiados (`fp8_a_int8_ampere.md:79-89`). Todo lo que no es GEMM lineal FP8 queda **BF16 explícito y no se toca** (lista `modules_to_not_convert`, 882 entradas).

La taxonomía completa produce **3 regímenes de quant**, no 2:

| Régimen | Qué significa | Peso | Activación |
|---|---|---|---|
| **A — FP8→INT8 diádico sin pérdida** | `fp8_a_int8_ampere.md:3-5` shift entero `q=(8+m)≪(3-d)`, `s'=s·2^(e_max-13)`. Exacto para `d≤3` → **99.95% de la energía de Frobenius** (`fp8_a_int8_ampere.md:282-286`). | `rel_err <1e-3` Diseño A, `<1e-2` Diseño B (`fp8_a_int8_ampere.md:627-632`) | SmoothQuant restringido a **potencias de dos** absorbido en RMSNorm BF16 (`fp8_a_int8_ampere.md:468-476`) → grilla sigue diádica, exacta |
| **B — BF16 sin pérdida** | Tensores en `modules_to_not_convert` (`fp8_a_int8_ampere.md:82-89`): `A_log`, `dt_bias`, `conv1d` 3D, `in_proj_a/b`, norms, `embed_tokens`, `lm_head` stock, ViT. Se mantienen BF16/FP16 pero con **fused layernorm+quant** para alimentar GEMM INT8 aguas abajo sin coste. | 0 (se conserva) | 0 si la norm absorbe `s=2^k` (suma al exponente BF16, exacta) |
| **C — FP8→INT8 con pérdida (activación)** | Donde **no hay norm delante** para absorber SmoothQuant (`fp8_a_int8_ampere.md:459-462`): `o_proj` / `linear_attn.out_proj` / `down_proj` (columna). O bien Diseño B per-channel que colapsa bloques no-diádicos (`fp8_a_int8_ampere.md:392-399`), o W8A8 ingenuo sin SmoothQuant. | Diseño B: hasta 1 bit de rango desperdiciado; W4 outlier → colapso de bloque (`fp8_a_int8_ampere.md:324-326`) | Outliers de residual 20-100× mediana (`fp8_a_int8_ampere.md:432`) → SQNR cae sin SmoothQuant; GDN compone error `∝√L` (`fp8_a_int8_ampere.md:484-496`) |

> W8A16 **no sirve** en Ampere: `w8a8_block_fp8_matmul` exige `s8×s8→s32` (`fp8_a_int8_ampere.md:54-66`). Peso INT8 + activación FP16 → dequant a FP16 + MMA FP16 = mismo path que Marlin FP8 actual, ganancia 0. **El objetivo no negociable es W8A8** para activar TC INT8.

Benchmark empírico que justifica el esfuerzo (`KERNELS-OPTIMIZACION.md:50-59` en 2×3090, `bench_gemm.py`):

- Decode M=40 (bandwidth-bound): Marlin 0.047/0.039/0.147/0.107 ms ≈ INT8 W8A8 (empate, 448-608 GB/s techo).
- **Prefill M=1664 (compute-bound): INT8 W8A8 0.38/0.29/1.61/0.75 ms vs Marlin 1.10/0.86/5.53/2.43 ms → 2.8-3.4× más rápido** (~77 TFLOPS INT8 vs ~20 Marlin). Prefill hoy pegado a cómputo (~770 tok/s), luego PN110 midió **-23% wall (7.21→5.56s) +29% tps** (`PLAN-CHECKPOINTS.md:219`).

---

## 1. Taxonomía completa de capas — todas las variaciones encontradas

> Todas las dimensiones son múltiplos exactos de 128 → 0 padding (`fp8_a_int8_ampere.md:137-138`). TP=2 hace `N`/`K` divisibles a la mitad (N par).

### 1.1 Capas GDN — 48 × `linear_attention` (`fp8_a_int8_ampere.md:97-109` + `qwen3_5.py:137-143` + `qwen_gdn_linear_attn.py`)

| Tensor (HF name) | Forma global | Forma /rank TP=2 | Cuant | Rol GEMM | Norma delante | Fused en vLLM | Packing |
|---|---:|---:|---|---|---|---|---|
| `linear_attn.in_proj_qkv.weight` | `[10240, 5120]` (80·128 × 40·128) | `[5120, 5120]` | **Sí** FP8 | ColumnParallel, **fusionado Q/K/V**: filas `0:2047 Q (16K×128)`, `2048:4095 K (16K×128)`, `4096:10239 V (48V×128)` (`fp8_a_int8_ampere.md:133-136`) | `input_layernorm` | `in_proj_qkvz` con `in_proj_z` (`qwen3_5.py:280-281, 447-448`) | **QKVZ** |
| `linear_attn.in_proj_z.weight` | `[6144, 5120]` (48·128) | `[3072, 5120]` | **Sí** | ColumnParallel, gate Z | `input_layernorm` (mismo que QKV) → **comparten vector `s` SmoothQuant** (`fp8_a_int8_ampere.md:455-458`) | `in_proj_qkvz` con `in_proj_qkv` | **QKVZ** |
| `linear_attn.out_proj.weight` | `[5120, 6144]` (40·128 × 48·128) | `[5120, 3072]` | **Sí** | **RowParallel** residual writer (`fp8_a_int8_ampere.md:144-154`), 48× + AllReduce | **ninguna** (`fp8_a_int8_ampere.md:453`) | — | — |
| `linear_attn.in_proj_a.weight` | `[48, 5120]` | `[48, 5120]` | **No** BF16 | ColumnParallel diminuto (control SSM, no GEMM grande) | — | `in_proj_ba` con `in_proj_b` (`qwen3_5.py:289-290, 449-450`) | **BA** |
| `linear_attn.in_proj_b.weight` | `[48, 5120]` | `[48, 5120]` | **No** | idem | — | `in_proj_ba` | **BA** |
| `linear_attn.conv1d.weight` | `[10240, 1, 4]` **3D** depthwise causal (`fp8_a_int8_ampere.md:139-140`) | igual | **No** | Conv causal | — | — | — |
| `linear_attn.A_log` / `dt_bias` | `[48]` | igual | **No** | Vector decay SSM | — | — | — |
| `linear_attn.norm.weight` | `[128]` | igual | **No** | RMSNorm | — | — | — |

**Global GDN GEMM fusionado QKVZ**: `N=16384 (=10240+6144)=128·128`, `K=5120`. Por rank: `N=8192, K=5120`. Este es el `SHAPES` `qkv: N=4096 K=5120` de `KERNELS-OPTIMIZACION.md:18-24` si se cuenta solo `in_proj_qkv` sin Z, o `12288` con otra granularidad; el valor exacto depende de si se incluye Z. Lo relevante: **un solo GEMM por capa** si se explota `in_proj_qkvz`.

Riesgo específico GDN: error en `in_proj_qkv` entra al **estado recurrente** `h = B×NT×H×V×K = 12 KiB/token/rank` (`CONTEXTO-INVESTIGACION.md:372-374`, `chunk_delta_h.py:350`) y **se compone ∝ L** hasta 262K (`fp8_a_int8_ampere.md:484-498`, test T5). PPL no lo ve; NIAH sí.

### 1.2 Capas Full Attention — 16 × `self_attn` + MTP 1× (`fp8_a_int8_ampere.md:110-119` + `qwen3_5.py:144-151`)

| Tensor | Forma global | Forma /rank | Cuant | Rol | Norma delante | Fused | Notas |
|---|---:|---:|---|---|---|---|---|
| `self_attn.q_proj.weight` | `[12288, 5120]` (96·128) | `[6144, 5120]` | **Sí** | Col, 24Q×256 pero **12288 no 6144 por `attn_output_gate:true`: Q||gate concat** (`fp8_a_int8_ampere.md:130-132`) | `input_layernorm` | `qkv_proj` con `k/v` (`qwen3_5.py:283-285, 441-445`) | q_norm [256] BF16 después |
| `self_attn.k_proj.weight` | `[1024, 5120]` (8·128) | `[512, 5120]` | **Sí** | Col, 4 KV×256 (GQA) | `input_layernorm` | `qkv_proj` | k_norm |
| `self_attn.v_proj.weight` | `[1024, 5120]` | `[512, 5120]` | **Sí** | Col, 4 KV×256 | `input_layernorm` | `qkv_proj` | — |
| `self_attn.o_proj.weight` | `[5120, 6144]` | `[5120, 3072]` | **Sí** | **RowParallel** residual writer, 16× (+1 MTP=17) | **ninguna** | — | Abliterado (`fp8_a_int8_ampere.md:144-154`) |

**Global FA GEMM fusionado QKV**: `N=14336 (=12288+1024+1024)=112·128`, `K=5120`. Por rank `N=7168`. `KERNELS-OPTIMIZACION.md:24` lo reporta como `o_proj N=5120 K=3072` (ya shardeado por K). Atención full usa **FlashInfer FA2** con KV `fp8_e4m3` dequant in-kernel (`CONTEXTO-INVESTIGACION.md:419-420`).

### 1.3 MLP — 64 capas (+1 MTP=65 `down_proj`) (`fp8_a_int8_ampere.md:120-127` + `qwen3_5.py:162-169`)

| Tensor | Forma global | Forma /rank | Cuant | Rol | Norma delante | Fused |
|---|---:|---:|---|---|---|---|
| `mlp.gate_proj.weight` | `[17408, 5120]` (136·128) | `[8704, 5120]` | **Sí** | Col, SwiGLU gate | `post_attention_layernorm` | `gate_up_proj` con `up_proj` (`qwen3_5.py:287-288, 446`) |
| `mlp.up_proj.weight` | `[17408, 5120]` | `[8704, 5120]` | **Sí** | Col, SwiGLU up | `post_attention_layernorm` (mismo `s`) | `gate_up_proj` |
| `mlp.down_proj.weight` | `[5120, 17408]` | `[5120, 8704]` | **Sí** | **RowParallel** residual writer, 64× (+1 MTP) | **ninguna** (salvo cadena `up→down` vía `diag(s)` en `fp8_a_int8_ampere.md:452`) | — |

**Global MLP fusionado GateUp**: `N=34816 (=2×17408)=272·128`, `K=5120`. Por rank `N=17408` — coincide con `KERNELS-OPTIMIZACION.md:25` `gate_up N=17408 K=5120` (ya por rank) y `down N=5120 K=8704`.

`down_proj` concentra outliers del residual → T0 lo señalará (`fp8_a_int8_ampere.md:619, 721`).

### 1.4 Cabeza y torre

| Tensor | Forma global | Forma /rank | Cuant | Rol | Notas |
|---|---:|---:|---|---|---|
| `lm_head.weight` | `[248320, 5120]` | `[124160, 5120]` (~1.2 GiB/rank FP16) | **No** stock (`fp8_a_int8_ampere.md:87`, `fp8.py:176-208` → `UnquantizedEmbeddingMethod`, cuBLAS FP16 vocab-paralelo) — **PN77** lo guarda FP8 para ahorrar ~1.2 GiB/GPU pero compute sigue FP16; **PN108** replica al draft. `KERNELS-OPTIMIZACION.md:29` `lm_head N=124160 K=5120` |
| `embed_tokens.weight` | `[248320, 5120]` | `[124160, 5120]` | **No** | Lookup |
| `visual.*` (333 tens.) | — | — | **No** | ViT | Nunca cuantizar (`fp8_a_int8_ampere.md:84`) |
| Norms: `input_layernorm`, `post_attention_layernorm`, `q_norm/k_norm` [256], `linear_attn.norm` [128], `model.norm` | — | — | **No** | RMSNorm BF16 | Aquí se **absorbe SmoothQuant** gratis (`fp8_a_int8_ampere.md:443-444`) |

### 1.5 MTP / Spec-decode (`qwen3_5_mtp.py:59-159`)

- `VocabParallelEmbedding` + `fc` ColumnParallel `hidden*2→hidden` + 1× `Qwen3_5DecoderLayer` Full + `ParallelLMHead` — reusa kernels del target (`CONTEXTO-INVESTIGACION.md:429`). Su `lm_head` es **2.78 GB/rank FP16** propio con `load_model` separado que PN77 no parcheaba → **PN108** lo cubre (`PLAN-CHECKPOINTS.md:168`).
- MTP draft copy kernels: `batch_memcpy` Triton 417 GB/s (`KERNELS-OPTIMIZACION.md:112-113`), `postprocess_mamba_fused_kernel`.

---

## 2. Matriz de regímenes por familia (para elegir super kernel)

| Familia | Tensores | Régimen recomendado | Justificación | Alternativa con pérdida |
|---|---|---|---|---|
| **GDN-QKVZ** | `in_proj_qkv` + `in_proj_z` (QKVZ 16384×5120) | **A diádico** W8A8 per-channel + shift diádico per-bloque (**Diseño C**, `fp8_a_int8_ampere.md:402-414`) + SmoothQuant `s=2^k` en `input_layernorm` | `d≤3` exacto, `s` potencia-de-dos preserva grilla → `rel_err <1e-3` (A) / `<1e-2` (B); activa TC INT8. | W8A8 per-token ingenuo (sin SmoothQuant) → SQNR -6..-10 dB por outliers; Diseño B puro desperdicia ≤1 bit/rango |
| **GDN-OUT** | `out_proj` 5120×6144 | **C escalado** W8A8 Diseño C (shift sobre acumulador INT32, `fp8_a_int8_ampere.md:410-414`, headroom 2.06M < 2.1G → 10 posiciones sin overflow) **o** W8A16 keep si T5 diverge `>√L` | Sin norm delante → SmoothQuant no absorbible (`fp8_a_int8_ampere.md:459`). Si T5 >2e-2 a 64K, subir a W8A16/BF16 (`fp8_a_int8_ampere.md:495-498`) | W8A8 per-token → error se propaga SSM |
| **FA-QKV** | `q_proj` (Q+gate) + `k_proj` + `v_proj` (QKV 14336×5120) | **A diádico** W8A8 per-channel + `s=2^k` en `input_layernorm` | Mismo que GDN-QKVZ pero sin recurrencia → tolera más pérdida; `q_norm/k_norm` BF16 después no afectan GEMM | Per-tensor sobre QKVZ → mezcla rangos Q/K/V → T6 >6 dB gap (`fp8_a_int8_ampere.md:664-666`) |
| **FA-O** | `o_proj` 5120×6144 | **C escalado** idem GDN-OUT (residual writer, abliterado) | Abliteración rango-1 por debajo de paso E4M3 (`fp8_a_int8_ampere.md:155-158`) → T10 gate; T2 alto solo en `down/o_proj` → outliers residual | W8A16 keep (-13% cómputo capa, `fp8_a_int8_ampere.md:459-462`) |
| **MLP-GateUp** | `gate_proj+up_proj` (GateUp 34816×5120) | **A diádico** W8A8 + `s=2^k` en `post_attention_layernorm`, **fused `silu_and_mul_quant`** (`KERNELS-OPTIMIZACION.md:151`) | RMSNorm BF16 absorbe SmoothQuant exacto (`fp8_a_int8_ampere.md:451`) → `silu_and_mul_quant` no existe en W8A16, solo en W8A8 | Gate/Up separados → 2 GEMMs, 2× overhead launch |
| **MLP-Down** | `down_proj` 5120×17408 | **C escalado** W8A8 per-channel + corrección columna `diag(s)` (`fp8_a_int8_ampere.md:452`) **o** 2-slice Ozaki (`fp8_a_int8_ampere.md:744-755`) si SQNR flojo | Sin norm delante; `up_proj` filas ×1/s + `down` cols ×s. Si T0 muestra `d>8` frecuente → Diseño A/C o W8A16 (`fp8_a_int8_ampere.md:721`) | W4A16 Marlin solo si VRAM es cuello (`fp8_a_int8_ampere.md:711`) |
| **LM-Head** | `lm_head` 248320×5120 | **B BF16** stock cuBLAS **o** W8A16 Marlin leve (`KERNELS-OPTIMIZACION.md:145`: ~1 ms/paso decode) | Vocab gigante → decode bandwidth-bound pero prefill 33→14 ms con INT8 (`KERNELS-OPTIMIZACION.md:58`). PN77 FP8-weight/FP16-compute ya ahorra 1.2 GiB/GPU | INT8 W8A8 vocab-parallel: mayor error tolerable (última capa, sin composición), pero T10 abliteración vive en `embed_tokens` (filas) |
| **SSM-Control** | `in_proj_a/b` 48×5120, `conv1d` 3D, `A_log/dt_bias` 48 | **B BF16** puro, **fused `fused_sigmoid_gating_delta_rule_update` + `causal_conv1d_update`** (`CONTEXTO-INVESTIGACION.md:424`) | Recurrencia con decay → error se acumula exponencial; 48 valores, 0 ahorro cuantizar (`fp8_a_int8_ampere.md:90-94`) | Cuantizar = bug (`fp8_a_int8_ampere.md:529-531`) |
| **Norm/Embed** | `input/post_attention/q/k/linear_attn/model.norm`, `embed_tokens` | **B BF16** + **fused RMSNorm→quant** wrapper (escala potencia-de-dos) | Preservar RMSNorm BF16 permite migrar outliers gratis (`fp8_a_int8_ampere.md:443-444`) | — |
| **Vision** | 333 tens. `visual.*` | **B BF16** | BF16, no tocar (`fp8_a_int8_ampere.md:84`) | — |
| **MTP Draft** | `qkv_proj+gate_up+down+lm_head` del draft | **A/C mirror** del target (PN110 swap 1:1 `b_col` column-major `nn.Parameter`, `PLAN-CHECKPOINTS.md:215-219`) | PN108 ya draft lm_head FP8 (`PLAN-CHECKPOINTS.md:173`) → -630 MiB/rank, aceptancia 53.5% intacta | — |

> **Regla** `fp8_a_int8_ampere.md:536`: `qkv_proj` y `gate_up_proj` comparten `input/post_attention_layernorm` → un único vector `s` por norm, optimizado conjunto para `q/k/v` o `in_proj_qkv/z` y `gate/up`. **Nunca per-tensor sobre `in_proj_qkv`** (`fp8_a_int8_ampere.md:516`).

---

## 3. Qué es un super kernel inline en este contexto

En vLLM V1 el pipeline por capa es (simplificado, `CONTEXTO-INVESTIGACION.md:7`, `CIRCUITO-TOKEN.md:2.2`):

```
hidden --RMSNorm--> quant(dynamic per-token) --GEMM--> (SiLU*Mul) --GEMM--> AllReduce(RowParallel) --residual add--> next
         \_____________________________________________ SSM/FlashInfer _____________________________________________/
```

En stock **W8A16 (Marlin)** no hay quant de activaciones → `RMSNorm` y `SiluAndMul` corren `native`/Inductor separados y el GEMM hace dequant FP8→FP16 en-reg (`KERNELS-OPTIMIZACION.md:151`, `CONTEXTO-INVESTIGACION.md:425`). Con **W8A8 INT8** se activan `silu_and_mul_quant`-style y se puede fusionar:

**Super kernel inline** = **un kernel CUDA/Triton/CUTLASS que fusiona en un solo launch**:

`{RMSNorm + SmoothQuant scale (2^k) + per-token quant INT8} → GEMM INT8 TC (mma.m16n8k32.s8) → {bias/act/ scale epilog} → {AllReduce/RS+AG} → residual`

Ganancias: 1 launch vs 3-5, 0 materialización intermedia, mantiene INT32 acumulado hasta epílogo (Diseño A: flush cada 128 / Diseño B/C: 1 scale por fila), usa TC INT8 (77 TFLOPS vs 35 FP16 en 3090).

---

## Estrategia de precarga al arrancar

> **Invariante de diseño**: vLLM **no decide kernels en el forward**. Al arrancar analiza **todo** — número y tipo de capas, dtype de KV, TP, geometrías — y **precarga en memoria todos los kernels necesarios** para todas las combinaciones observadas. En inferencia solo hace **dispatch directo por índice** sin `if`/`branch`/`check`.

Al iniciar, vLLM escanea el modelo, determina para cada capa qué kernel exacto necesita según su **tipo** (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `in_proj_qkv`, `in_proj_z`, `out_proj`, `lm_head`, etc.) y su **M** (`1`, `8`, `1664`, `8000`), compila/cachea esos kernels, y en el `forward` solo hace dispatch directo sin branches ni checks.

### Fase 1 — Escaneo estático al arrancar

En la inicialización del `LLMEngine`/`ModelRunner` (`assets/vllm/vllm/model_executor/models/qwen3_5.py:276-449`, `workshop/ox_alpha/CONTEXTO-INVESTIGACION.md:7-9`):

1. **Enumera capas y tipos**: recorre `config.num_hidden_layers = 64` (+1 MTP `qwen3_5_mtp.py:59-159`) y clasifica cada capa en **GDN** (`linear_attn.*`, 48 capas) vs **Full** (`self_attn.*`, 16 capas) según `full_attention_interval=4` (`fp8_a_int8_ampere.md:42-46`), más **MLP** (`gate_proj`/`up_proj`/`down_proj`, 64+1) y **cabeza** (`lm_head`/`embed_tokens`). Para cada tensor lee nombre HF, forma `N×K` global y por rank TP=2, régimen A/B/C y fusión (`qkv_proj`, `gate_up_proj`, `in_proj_qkvz`, `in_proj_ba` en `qwen3_5.py:276-291, 440-450`).

2. **Determina `M` relevantes por tipo**: `M` = tokens batcheados que ve el GEMM. Del profiling (`KERNELS-OPTIMIZACION.md:27, 50-59`) los buckets que realmente aparecen son:
   - **M=1** — decode single-seq, latencia pura, bandwidth-bound
   - **M=8** — decode con spec-decode / micro-batch corto (draft `K=3`, `rejection_sampler.py:119-197`)
   - **M=1664** — prefill chunk medio (`max_num_batched_tokens` chunked, `KERNELS-OPTIMIZACION.md:27`)
   - **M=8000** — prefill batch largo / contexto largo (límite superior chunk/prefill)
   > Cada `SK` no necesita todos los `M` (p. ej. `lm_head` solo ve `M` agregado de prefill; `SSM_CONTROL` no es GEMM), pero la tabla es **exhaustiva**: si una combinación `tipo × M` existe en el modelo, su kernel ya está resuelto al arrancar. `M=40` (10 seqs×4 MTP decode) colapsa al bucket INT8 más cercano sin recompilar.

3. **Producto `tipo × M`**: para cada familia (ej. `MLP_GATEUP` 17408×5120, `GDN_QKVZ` 8192×5120, `FA_O` 5120×3072 por rank) × cada bucket `M` genera una clave `(SK, N, K, M, dtype, TP)`. Solo combinaciones que existen en el checkpoint `orcarouter/Qwen3.8-27B-Uncensored-FP8` → **cero kernels genéricos**, cero fallback dinámico.

### Fase 2 — Compilación y cacheo en memoria

Para cada clave `(SK, M)`:

- **Compila/cachea el kernel exacto**: CUTLASS `cutlass_scaled_mm` INT8 `mma.m16n8k32.s8.s32` con epílogo fusionado (`silu_and_mul_quant`, `RMSNorm+quant`), Triton `fused_sigmoid_gating_delta_rule_update`/`causal_conv1d_update` (`CONTEXTO-INVESTIGACION.md:424`), o cuBLAS BF16 según SK. En Ampere SM80/86 el `scale` es `float32` + `shift` diádico `int8` por bloque 128×128 (Diseño C, `fp8_a_int8_ampere.md:402-414`); la compilación especializa `M,N,K` y layout `b_col` column-major (`b=w.t()`, `KERNELS-OPTIMIZACION.md:198-199`).
- **Materializa en VRAM**: peso ya reordenado `b_col` como `nn.Parameter` column-major (patrón PN110 `PLAN-CHECKPOINTS.md:215-219`, chunked 512), `weight_scale` float32, `shift` int8, y `a_scale` por token pre-reservado. KV cache `fp8_e4m3` (`compose/docker-compose.qwen38-27b-fp8.yml:446-447`, `GENESIS_PN92_KV_DTYPE`) separado, dequant in-kernel FlashInfer (`CONTEXTO-INVESTIGACION.md:419`) no interfiere.
- **Memoiza puntero**: `kernel_table[SK][M] → fn_ptr` (dict de `torch.compile` / Triton cache / `CUDAGraph` capturado). Coste: ~11 SK × hasta 4 buckets `M` ≈ 30-40 kernels × <5 MB código ≪ 24 GiB de linears; latencia de compilación pagada **una vez al arranque**, no en TTFT/TPOT.

> Al terminar `load_model` + `warmup` ya están en memoria **todos** los kernels que el modelo usará toda su vida (prefill + decode, GDN + Full + MLP + lm_head). No hay compilación lazy.

### Fase 3 — Forward: dispatch directo sin branches ni checks

En `ModelRunner.execute_model` / `forward` por capa:

```python
# pseudo — tabla resuelta al construir el grafo, no en hot path
out = kernel_table[sk_id][M](hidden, weight_b_col, scale, shift)  # O(1) indexado
# ej: SK-05 GateUp M=1664 → cutlass_scaled_mm M=1664 N=17408 K=5120 INT8
#     SK-01 GDN_QKVZ M=8   → cutlass_scaled_mm M=8    N=8192  K=5120 INT8
```

- **Sin branches**: no hay `if tipo == "q_proj"` ni `if M < 16` ni `if kv_dtype == fp8` en hot path. El `sk_id` y `M` están fijados por capa al construir el grafo (unrolled por `qwen3_5.py:276-449`).
- **Sin checks**: sin validación de forma, dtype, `e_max` o `d` diádico por capa en inferencia — eso se hizo offline en T0-T2 (`fp8_a_int8_ampere.md:599-632`, `rel_err <1e-3` Diseño A). Si un tensor no cumple la grilla diádica (`d≤3` exacto, `fp8_a_int8_ampere.md:263-270`), el modelo **falla al arrancar** (fail-fast), no degrada silencioso en runtime.
- **Sin recompilación**: `torch.compile`/`CUDAGraph` captura el grafo con punteros fijos; prefill y decode pueden tener grafos distintos por `M` pero **ambos ya compilados** al inicio. `VLLM_MARLIN_USE_ATOMIC_ADD` y similares son inertes (`CONTEXTO-INVESTIGACION.md:438-439`) porque el kernel ya elegido no los necesita.

**Efecto**: elimina overhead de dispatch (3-5 launches → 1 por SK, `KERNELS-OPTIMIZACION.md:50-59` 2.8-3.4× en prefill, `-23% wall 7.21→5.56s` PN110 `PLAN-CHECKPOINTS.md:219`), jitter y divergencia de ramas; el path es **determinista** y profileable (`nsys` muestra siempre el mismo kernel por `(capa, M)`).

En resumen: **arranque = análisis total + precarga exhaustiva; inferencia = tabla de punteros + llamada directa**.

---

## 4. Catálogo de super kernels inline (uno por variación)

> **Convención**: cada fila es un super kernel desplegable independientemente vía `GENESIS_ENABLE_*`. `SK` = Super Kernel. `TC` = Tensor Cores. Pérdida vs `fp8_a_int8_ampere.md:1.6` targets: PPL 6.96, MMLU 84.7%, GSM8K 88.7%, CMMLU 80.8%.

> **KV por defecto (stack)**: todos los SK asumen `kv_cache_dtype fp8_e4m3` — `compose/docker-compose.qwen38-27b-fp8.yml:446-447` `--kv-cache-dtype fp8_e4m3` y `GENESIS_PN92_KV_DTYPE=fp8_e4m3` (`compose/docker-compose.qwen38-27b-fp8.yml:196`). FlashInfer FA2 hace dequant fp8 in-kernel (`CONTEXTO-INVESTIGACION.md:419`) sin interferir con GEMM INT8.

### Tabla resumen (vista rápida)

| # | Super kernel | Capas que cubre | `N×K` global (por rank) | Tipo quant | Pérdida esperada | TC | Fusión inline |
|---|---|---|---|---|---|---|---|
| **SK-01** | `GDN_QKVZ_FUSED_INT8_DIADIC` | `in_proj_qkv` + `in_proj_z` (QKVZ) — 48 capas GDN | `16384×5120` (`8192×5120`) | **FP8→INT8 diádico W8A8 per-channel + shift diádico per-bloque (Diseño C) + SmoothQuant `s=2^k` en `input_layernorm`** (`fp8_a_int8_ampere.md:5.1`) | **mínima** — peso `99.95%` energía exacta (`fp8_a_int8_ampere.md:282`), `rel_err <1e-3` (A) / `<1e-2` (B) (`fp8_a_int8_ampere.md:627-632`); act `α=0.5-0.8` barrido (`fp8_a_int8_ampere.md:5`) | **Sí INT8** `m16n8k32.s8.s32` | `RMSNorm(input_layernorm) + scale_pow2 + quant_per_token INT8 + GEMM + split Q/K/V (128-alineado)` |
| **SK-02** | `GDN_OUT_INT8_SCALED` | `linear_attn.out_proj` — 48 capas GDN, RowParallel residual | `5120×6144` (`5120×3072`) | **FP8→INT8 W8A8 Diseño C** (per-channel float32 `ops.cutlass_scaled_mm` + `int8` shift/bloque `fp8_a_int8_ampere.md:402-414`) **o fallback W8A16** si T5 diverge | **baja-moderada** — sin norm delante, column shift ≤1 bit desperdicio; si T5 `>2e-2` a 64K → subir a W8A16/BF16 (`fp8_a_int8_ampere.md:649-654`) | **Sí INT8** (C) / **No** FP16 TC si W8A16 | `GEMM + AllReduce(RowParallel) + residual add` fused; shift INT32 si C |
| **SK-03** | `FA_QKV_FUSED_INT8_DIADIC` | `q_proj` (Q+gate 12288) + `k_proj` (1024) + `v_proj` (1024) — 16 capas Full (+1 MTP) | `14336×5120` (`7168×5120`) | **FP8→INT8 diádico W8A8 + SmoothQuant `2^k` en `input_layernorm`** | **mínima** — idem SK-01; sin recurrencia, tolera más; `q/k_norm` BF16 después no interfiere | **Sí INT8** | `RMSNorm + quant + GEMM + split Q||gate/K/V + q_norm/k_norm BF16` |
| **SK-04** | `FA_O_INT8_SCALED` | `self_attn.o_proj` — 16 (+1 MTP) Full, RowParallel residual, abliterado | `5120×6144` (`5120×3072`) | **FP8→INT8 W8A8 Diseño C o W8A16 keep** (residual writer `fp8_a_int8_ampere.md:144-154`) | **baja-moderada** — T10 abliteración <paso E4M3 (`fp8_a_int8_ampere.md:155-158`); T2 outlier si `down_proj`-like | **Sí INT8** / No si W8A16 | `GEMM + AllReduce + residual` (gate `VLLM_MARLIN_USE_ATOMIC_ADD` inerte `n<2048` no aplica, `CONTEXTO-INVESTIGACION.md:438-439`) |
| **SK-05** | `MLP_GATEUP_FUSED_INT8_DIADIC` | `gate_proj` + `up_proj` → `gate_up_proj` — 64 capas (+1 MTP) | `34816×5120` (`17408×5120`) | **FP8→INT8 diádico W8A8 + SmoothQuant `2^k` en `post_attention_layernorm`**, `silu_and_mul_quant` | **mínima** — SwiGLU tolera shift diádico; `KERNELS-OPTIMIZACION.md:151` `silu_and_mul_quant` solo existe en W8A8 | **Sí INT8** | `RMSNorm(post) + quant + GEMM GateUp + SiLU+Mul fused + quant` (1 launch vs 2) |
| **SK-06** | `MLP_DOWN_INT8_SCALED_RESIDUAL` | `mlp.down_proj` — 64 (+1 MTP) RowParallel residual | `5120×17408` (`5120×8704`) | **FP8→INT8 W8A8 escalado** (`up` filas×1/s + `down` cols×s, `fp8_a_int8_ampere.md:452`) **o Ozaki 2-slice** (`fp8_a_int8_ampere.md:744-755`) si SQNR flojo | **moderada** — outliers residual concentrados aquí (`fp8_a_int8_ampere.md:619`); T0/T2 gate: si `p99 d>8` → Diseño A | **Sí INT8** (C) / 2×INT8 si Ozaki (∼1×FP16, mejor SQNR) | `GEMM + AllReduce + residual add` + columna scale |
| **SK-07** | `LM_HEAD_VOCAB` | `lm_head` — 1 (+1 draft PN108) vocab-parallel | `248320×5120` (`124160×5120`) | **BF16 cuBLAS** stock / **FP8-weight FP16-compute (PN77)** / **INT8 W8A8 o W8A16 Marlin** (`KERNELS-OPTIMIZACION.md:5, 145`) | **0 (BF16)** / **<0.5% (FP8-PN77)** / **moderada si INT8** (última capa, sin composición) | BF16: **Sí BF16 TC** · INT8: **Sí INT8** · FP8 stock: No (cuBLAS FP16) | `vocab_parallel_gather + logits_processor` (`CONTEXTO-INVESTIGACION.md:426`, `logits_processor.py:75-104`); PN77 guarda FP8 pero compute FP16 (`CONTEXTO-INVESTIGACION.md:509-510`) |
| **SK-08** | `SSM_CONTROL_BF16_FUSED` | `in_proj_a`/`in_proj_b` (48×5120), `conv1d` 3D [10240,1,4], `A_log`/`dt_bias` (48) — 48 GDN | — (GEMM 48×5120 micro, conv depthwise) | **BF16 sin pérdida** — **NUNCA cuantizar** (`fp8_a_int8_ampere.md:90-94, 529-534`) | **0** — BF16 conserva decay exacto; cuantizar → deriva SSM superlineal (`fp8_a_int8_ampere.md:6.1`) | **No** (Triton `fused_sigmoid_gating` + `causal_conv1d_update` `CONTEXTO-INVESTIGACION.md:424`) | `fused_sigmoid_gating_delta_rule_update` (in-place sobre `ssm_state` FP32 `fp8_a_int8_ampere.md:505`) + `causal_conv1d_update` (spec-packing `num_warps=1`) |
| **SK-09** | `NORM_EMBED_BF16_PASSTHROUGH` | `embed_tokens`, `input/post_attention/q/k/linear_attn/model.norm`, vision `rms` | — | **BF16 sin pérdida + wrapper fused norm→quant** (absorbe `s=2^k` exacto: suma a exponente BF16 `fp8_a_int8_ampere.md:301-303`) | **0** (wrapper es suma entera a campo exponente) | **No** (norm es memory-bound, `native`/`Triton`) | `RMSNorm + scale_pow2` (cuando downstream es INT8) + `embed lookup` (1.271B params c/u) |
| **SK-10** | `MTP_DRAFT_MIRROR` | Draft MTP: `qkv_proj` + `gate_up_proj` + `down_proj` + `lm_head` draft + `fc` hidden*2→hidden (1 capa Full) | espejo SK-03/05/06/07 por rank | **FP8→INT8 swap 1:1 `b_col` column-major `nn.Parameter`** (`PLAN-CHECKPOINTS.md:215-219`, PN110) + **PN108 draft lm_head FP8** (`PLAN-CHECKPOINTS.md:173`) | **mínima** — rejection sampler verifica (`PLAN-CHECKPOINTS.md:172`); A/B 2.61 vs 2.60 aceptancia, -630 MiB/rank (`PLAN-CHECKPOINTS.md:173-174`) | **Sí INT8** (draft) | `lm_head draft FC + decoder GDN/Full + rejection_greedy_sample_kernel` (`CONTEXTO-INVESTIGACION.md:430`, `rejection_sampler.py:119-197`) |
| **SK-11** | `VISION_BF16` | `visual.*` 333 tens. ViT | — | **BF16 sin pérdida** | 0 | BF16 TC (ViT) | Vision tower intacto, no fusionado con LLM GEMM |

> **Nota KV**: todos los SK anteriores operan con `kv_cache_dtype fp8_e4m3` por defecto (`compose/docker-compose.qwen38-27b-fp8.yml:446-447` `--kv-cache-dtype fp8_e4m3`, `GENESIS_PN92_KV_DTYPE=fp8_e4m3`); compatible con FlashInfer FA2 y `mamba-ssm-cache-dtype float16` separado.

> **No listado**: `FLA Triton` prefill `chunk_local_cumsum→chunk_scaled_dot_kkt→solve_tril→recompute_w_u→chunk_gated_delta_rule→chunk_fwd_o` (`CONTEXTO-INVESTIGACION.md:423`). Es **1.2% del paso** (0.44 ms×48≈21 ms, `KERNELS-OPTIMIZACION.md:98-108`), tuning `BKV_LIST` descartado empíricamente → no super kernel dedicado; solo `cumsum+kkt` fusionable marginal.

---

### Detalle por super kernel

#### SK-01 — `GDN_QKVZ_FUSED_INT8_DIADIC` ⭐ (máximo impacto / mayor riesgo T5)

- **Cubre**: `linear_attn.in_proj_qkv` [10240,5120] + `in_proj_z` [6144,5120] → `in_proj_qkvz` [16384,5120] (`qwen3_5.py:280-281`). 48 capas GDN. 10240+6144=16384 filas; por rank TP2: 8192×5120. Split post-GEMM `Q 0:2047 / K 2048:4095 / V 4096:10239` cae en múltiplo de 128 → 0 bloques cruzan Q/K/V (`fp8_a_int8_ampere.md:133-136`).
- **Quant**: **FP8 E4M3 bloque128 → INT8 W8A8 per-channel float32 + shift diádico per-bloque int8 (Diseño C)** con `s'=s_blk·2^(e_max-13)` (`fp8_a_int8_ampere.md:253-255`) + código `fp8_e4m3_to_int8_aligned` (`fp8_a_int8_ampere.md:334-365`). Activación: **SmoothQuant `s=2^round(log2 s)`, `α=0.5-0.8`** barrido por T4 (`fp8_a_int8_ampere.md:468-478`), absorbido en `input_layernorm.weight` BF16 (suma a exponente, `fp8_a_int8_ampere.md:301`). Escala compartida para QKV y Z (restricción `fp8_a_int8_ampere.md:455`).
- **Pérdida esperada**: peso **exacto `d≤3` (64-120 rango q, `fp8_a_int8_ampere.md:263-270`) para ~90% pesos / 99.95% energía** (`fp8_a_int8_ampere.md:282-286`); `d≥4` colisiona pares→flush0. T2 `rel_err <1e-3` Diseño A / `<1e-2` B-C; T3 energía `d≥4` ~5e-4; T6 SQNR Q/K/V balance <6 dB; **T5 SSM drift <1e-3 @1K, <5e-3 @8K, <2e-2 @64K** o subir a SK-02 W8A16.
- **Tensor cores**: **Sí INT8** `mma.m16n8k32.s8.s8.s32` — prefill 2.9× vs Marlin (`KERNELS-OPTIMIZACION.md:54` qkv 1.09→0.38 ms). Decode M=40 bandwidth-bound: 0.090→0.043 ms leve.
- **Fusión inline**: `input_layernorm(RMSNorm fp16/BF16) --pow2_scale--> quant_per_token INT8 (cutlass `b_col` column-major, `b=w.t()` `KERNELS-OPTIMIZACION.md:199`) --> GEMM INT8 --> split Q/K/V + conv-prep`. 1 launch por capa GDN (vs 2 GEMM + norm separado).
- **KV cache**: `kv_cache_dtype fp8_e4m3` por defecto (`compose/docker-compose.qwen38-27b-fp8.yml:446-447`, `GENESIS_PN92_KV_DTYPE=fp8_e4m3`); FLA/GDN SSM `float16` (`compose/docker-compose.qwen38-27b-fp8.yml:444-445` `mamba-ssm-cache-dtype float16`) separado del KV de atención.
- **Forma GEMM**: `M` = tokens batcheados (decode 40 =10 seqs×4 MTP, prefill 1664 chunk `KERNELS-OPTIMIZACION.md:27`), `N=8192`/`K=5120` por rank. `Dtype scale=w_scale float32` exigido sm80 (`KERNELS-OPTIMIZACION.md:198`).
- **Gate**: T0 histograma `d` por tensor (`fp8_a_int8_ampere.md:599-619`), T4 SQNR por capa, **T5 crítico** long-context.

#### SK-02 — `GDN_OUT_INT8_SCALED` (residual writer)

- **Cubre**: `linear_attn.out_proj` 5120×6144 (40·128×48·128), 48 capas, RowParallel → 2 AllReduce/capa.
- **Quant**: **W8A8 Diseño C per-channel + shift diádico por bloque** sobre acumulador INT32 (shift barato, no multiply `fp8_a_int8_ampere.md:410-412`). Headroom 2.06M <2.1G → 10 shifts sin overflow. Sin norm delante, alternativa `W8A16 keep` si T5 superlineal (`fp8_a_int8_ampere.md:486-498`). Nunca per-tensor.
- **Pérdida**: **baja-moderada**; `out_proj` es residual writer abliterado (`fp8_a_int8_ampere.md:144-154`); T10 probes rechazo (0-6% baseline) si `rel_err` perturba `r(rᵀW)` (`fp8_a_int8_ampere.md:521-523`). Diseño C desperdicia ≤1 bit/rango (`fp8_a_int8_ampere.md:416`).
- **TC**: **Sí INT8** con C; **No** (FP16 TC tras dequant) si W8A16.
- **Fusión**: `GEMM INT8 --> AllReduce (PYNCCL Ring, `CONTEXTO-INVESTIGACION.md:588-590`) --> residual add`. `VLLM_MARLIN_USE_ATOMIC_ADD` inerte `n<2048` (`CONTEXTO-INVESTIGACION.md:438`, `KERNELS-OPTIMIZACION.md:144:8`) → no fused atomic.
- **KV cache**: `kv_cache_dtype fp8_e4m3` por defecto (`compose/docker-compose.qwen38-27b-fp8.yml:446-447`); residual writer no toca KV, mantiene compatibilidad con cache fp8_e4m3.

#### SK-03 — `FA_QKV_FUSED_INT8_DIADIC` (GQA + output-gate)

- **Cubre**: `self_attn.q_proj` 12288×5120 (Q||gate) + `k_proj` 1024×5120 + `v_proj` 1024×5120 → `qkv_proj` 14336×5120, 16 capas (+1 MTP). Por rank 7168×5120. `q_norm/k_norm` [256] BF16 después.
- **Quant**: **FP8→INT8 diádico + SmoothQuant `2^k` en `input_layernorm`** (compartido q/k/v, `fp8_a_int8_ampere.md:449`). `k/v` chicos 8·128 pero mismo bloque128 → per-channel por fila.
- **Pérdida**: **mínima** (sin recurrencia). T6 balance Q/K/V <6 dB; per-channel preserva separación Q/K/V/gate.
- **TC**: **Sí INT8** `cutlass_scaled_mm`. FlashInfer FA2 aguas abajo con KV fp8 in-kernel dequant (`CONTEXTO-INVESTIGACION.md:419`) no interfiere (KV cache dtype fp8_e4m3 separado).
- **Fusión**: `RMSNorm + quant --> GEMM QKV --> split Q||gate / K / V --> q_norm/k_norm (BF16)`. 1 launch.
- **KV cache**: `kv_cache_dtype fp8_e4m3` por defecto (`compose/docker-compose.qwen38-27b-fp8.yml:446-447`); FlashInfer FA2 KV fp8 in-kernel dequant (`CONTEXTO-INVESTIGACION.md:419`) no interfiere con GEMM INT8.

#### SK-04 — `FA_O_INT8_SCALED`

- **Cubre**: `self_attn.o_proj` 5120×6144, 17× (16+MTP), RowParallel.
- **Quant**: **Diseño C W8A8** o W8A16 keep (13% cómputo capa si se deja en W8A16, `fp8_a_int8_ampere.md:460`).
- **Pérdida**: **baja-moderada**; mismo gate T10 abliteración que SK-02; `o_proj` headroom igual.
- **TC**: **Sí INT8** / No si W8A16.
- **Fusión**: `GEMM --> AllReduce --> residual`.
- **KV cache**: `kv_cache_dtype fp8_e4m3` por defecto (`compose/docker-compose.qwen38-27b-fp8.yml:446-447`); o_proj residual downstream del KV, conserva dtype fp8_e4m3.

#### SK-05 — `MLP_GATEUP_FUSED_INT8_DIADIC` ⭐ (mayor GEMM prefill)

- **Cubre**: `mlp.gate_proj` 17408×5120 + `up_proj` 17408×5120 → `gate_up_proj` 34816×5120, 64 capas (+1 MTP). Por rank 17408×5120 (`KERNELS-OPTIMIZACION.md:25`).
- **Quant**: **INT8 diádico + SmoothQuant `2^k` en `post_attention_layernorm`** (`fp8_a_int8_ampere.md:451`). Comparten `s`; `s` potencia-de-dos exacto en BF16 RMSNorm.
- **Pérdida**: **mínima**; SwiGLU `SiLU(gate)*up` tolera `q_max≈92` promedio → precisión relativa 2^-6.5 vs 2^-4 E4M3 (`fp8_a_int8_ampere.md:319` sigue más fino).
- **TC**: **Sí INT8** — prefill 5.53→1.61 ms **3.4×** (`KERNELS-OPTIMIZACION.md:57`), decode 0.323→0.153 ms.
- **Fusión**: `RMSNorm(post) + quant --> GEMM GateUp --> SiluAndMulQuant` (kernel `silu_and_mul_quant` solo en W8A8, `KERNELS-OPTIMIZACION.md:151`). 1 launch vs 2 GEMMs + SiLU.
- **KV cache**: `kv_cache_dtype fp8_e4m3` por defecto (`compose/docker-compose.qwen38-27b-fp8.yml:446-447`); MLP no toca KV, stack KV fp8_e4m3 sin cambios.

#### SK-06 — `MLP_DOWN_INT8_SCALED_RESIDUAL`

- **Cubre**: `mlp.down_proj` 5120×17408, 65× (64+MTP), RowParallel residual writer, forma `down N=5120 K=8704` por rank (`KERNELS-OPTIMIZACION.md:26`).
- **Quant**: **W8A8 escalado** — `gate/up` filas ×1/s + `down` cols ×s (`fp8_a_int8_ampere.md:452`). Si T2 alto solo en `down` → granularidad fina o Ozaki 2-slice: `A≈A_hi+2^-7·A_lo`, 2 GEMM INT8 → ~15 bits efectivos (`fp8_a_int8_ampere.md:746-750`), plan B si SmoothQuant no cierra.
- **Pérdida**: **moderada** — outliers residual aquí (`fp8_a_int8_ampere.md:619`). Prefill 2.43→0.75 ms **3.2×** si INT8.
- **TC**: **Sí INT8** (o 2×INT8 Ozaki ≈1×FP16).
- **Fusión**: `GEMM --> AllReduce --> residual add` (+ escala columna). `mp` fusionable con `AllReduceFusionPass` solo SM90+ TRtLLM (`KERNELS-OPTIMIZACION.md:159`) → en SM86 vía PYNCCL.
- **KV cache**: `kv_cache_dtype fp8_e4m3` por defecto (`compose/docker-compose.qwen38-27b-fp8.yml:446-447`); down_proj residual, no afecta KV fp8_e4m3.

#### SK-07 — `LM_HEAD_VOCAB`

- **Cubre**: `lm_head` 248320×5120 global → 124160×5120/rank (~1.2 GiB FP16), vocab-parallel + gather (`logits_processor.py:75-104`), 1 (+1 draft).
- **Quant**: **Stock BF16 cuBLAS** (0 pérdida) / **PN77 FP8-weight FP16-compute** (-1.2 GiB/GPU, `CONTEXTO-INVESTIGACION.md:506-510`) / **W8A16/W8A8 Marlin/cutlass** (proyecto `B7`, `PLAN_AB_POR_PARCHE.md:31,67` ~1 ms/paso decode `KERNELS-OPTIMIZACION.md:145`).
- **Pérdida**: BF16 0; PN77 <0.5% (lm_head tolera); INT8 moderada pero sin composición recurrente.
- **TC**: BF16/FP16 **Sí BF16/FP16 TC** (cuBLAS) / INT8 **Sí INT8** (cutlass `b=w.t()` float32 scale) / FP8 No.
- **Fusión**: `GEMM vocab --> gather (tensor_model_parallel_gather) --> logits_processor._get_logits`. Prefill 33→14 ms **2.8×** con INT8 (`KERNELS-OPTIMIZACION.md:58`).
- **KV cache**: `kv_cache_dtype fp8_e4m3` por defecto (`compose/docker-compose.qwen38-27b-fp8.yml:446-447`); lm_head vocab-parallel, sin KV.

> Decisión producto: BF16/PN77 para calidad máxima decode; INT8 solo para prefill largo (threshold `GENESIS_PNXX_W8A8_MIN_TOKENS`, `PLAN-CHECKPOINTS.md:220-226`).

#### SK-08 — `SSM_CONTROL_BF16_FUSED` (BF16 sin pérdida)

- **Cubre**: `in_proj_a` 48×5120 + `in_proj_b` 48×5120 → `in_proj_ba` (`qwen3_5.py:289-290`), `conv1d` 3D [10240,1,4] depthwise, `A_log`/`dt_bias` 48, `linear_attn.norm` 128 — 48 GDN.
- **Quant**: **BF16 sin pérdida** — lista `modules_to_not_convert` (`fp8_a_int8_ampere.md:88-89`). `A_log`/`dt_bias` controlan decay estado FP32 `mamba_ssm_dtype float32` (`fp8_a_int8_ampere.md:505`) → error se acumula L (`fp8_a_int8_ampere.md:90-94`).
- **Pérdida**: **0** si BF16; **si INT8 → bug** (`fp8_a_int8_ampere.md:529`, `conv1d` 3D corrompido `fp8_a_int8_ampere.md:534`).
- **TC**: **No** — decode `fused_sigmoid_gating_delta_rule_update` + `causal_conv1d_update` Triton (`CONTEXTO-INVESTIGACION.md:424`, `fla/ops/fused_sigmoid_gating.py:24`, `fla/ops/fused_recurrent.py:198-199`) 0.44 ms/capa irrelevante; prefill `FLA Triton` pipeline (`KERNELS-OPTIMIZACION.md:98-103`).
- **Fusión**: `fused_sigmoid_gating (a/b/A_log/dt_bias → gating) + causal_conv1d_update` ya fusionados (`KERNELS-OPTIMIZACION.md:144`).
- **KV cache**: `kv_cache_dtype fp8_e4m3` por defecto (`compose/docker-compose.qwen38-27b-fp8.yml:446-447`); SSM control (A_log/dt_bias/conv1d) no usa KV de atención, cache FP8 se mantiene.

#### SK-09 — `NORM_EMBED_BF16_PASSTHROUGH` (BF16 sin pérdida, habilitador)

- **Cubre**: `embed_tokens` (1.271B), `input_layernorm`/`post_attention_layernorm` (64 c/u), `q_norm`/`k_norm` (256) 16 Full, `linear_attn.norm` 128, `model.norm`.
- **Quant**: **BF16 sin pérdida** + **fused RMSNorm→quant wrapper** que absorbe `s=2^k` por suma al campo exponente BF16 `mu+1<<7` (`fp8_a_int8_ampere.md:301-303`), exacto hasta overflow.
- **Pérdida**: **0** (shift entero exacto).
- **TC**: RMSNorm `native`/Inductor; `embed` lookup no GEMM. `VLLM_MARLIN_USE_ATOMIC_ADD` irrelevante aquí (`KERNELS-OPTIMIZACION.md:439`).
- **Fusión**: `RMSNorm BF16 --pow2--> quant_per_token INT8` para alimentar SK-01/03/05. Sin esto W8A8 no activa `silu_and_mul_quant`/`fused_norm_quant` (`KERNELS-OPTIMIZACION.md:151:161`).
- **KV cache**: `kv_cache_dtype fp8_e4m3` por defecto (`compose/docker-compose.qwen38-27b-fp8.yml:446-447`); norms/embed passthrough, KV fp8_e4m3 sin alteración.

#### SK-10 — `MTP_DRAFT_MIRROR` (draft = espejo del target)

- **Cubre**: Draft MTP (`qwen3_5_mtp.py:59-159`): `VocabParallelEmbedding` 2.78 GiB/rank (`PLAN-CHECKPOINTS.md:172`), `fc` hidden*2→hidden, 1× decoder Full (`qkv_proj` + `gate_up` + `down` + `o_proj`), `ParallelLMHead` draft.
- **Quant**: **Mirror SK-03/05/06/07** en draft. **PN108** `SpecDecodeBaseProposer.load_model` rebind `Genesis_FP8_LMHead_EmbeddingMethod` (`PLAN-CHECKPOINTS.md:168-174,261`). **PN110** swap `b_col` column-major `nn.Parameter` 1:1 (`PLAN-CHECKPOINTS.md:215-219`) + chunked 512 per-channel + exclusión GDN/mamba.
- **Pérdida**: **mínima** — sampler MTP `rejection_greedy_sample_kernel` (`rejection_sampler.py:714`) verifica; A/B 2.61 vs 2.60 aceptancia (`PLAN-CHECKPOINTS.md:173`).
- **TC**: **Sí INT8** draft (prefill draft 5.56 vs 7.21s wall `PLAN-CHECKPOINTS.md:219`).
- **Fusión**: `embed draft --> fc --> decoder --> lm_head draft --> expand/verify` (`rejection_sampler.py:119-197,837,861`, `gpu_model_runner.py:598-603,5240-5243`).
- **KV cache**: `kv_cache_dtype fp8_e4m3` por defecto (`compose/docker-compose.qwen38-27b-fp8.yml:446-447`); draft MTP espejo del target, hereda KV fp8_e4m3.

#### SK-11 — `VISION_BF16` (no tocar)

- **Cubre**: `visual.*` 333 tens. (`fp8_a_int8_ampere.md:84`).
- **Quant**: **BF16 sin pérdida**. T11 verificación visión no-op (`fp8_a_int8_ampere.md:691`).
- **TC**: BF16.
- **Fusión**: ViT separado, no cruza con LLM GEMM.
- **KV cache**: `kv_cache_dtype fp8_e4m3` por defecto (`compose/docker-compose.qwen38-27b-fp8.yml:446-447`); ViT sin KV, stack LLM mantiene fp8_e4m3.

---

## 5. Matriz de cobertura — ¿queda algo sin super kernel?

| Tensor familia | Super kernel | Estado |
|---|---|---|
| `in_proj_qkv` + `in_proj_z` | SK-01 | INT8 diádico, TC INT8 |
| `out_proj` (GDN) | SK-02 | INT8 TC o W8A16 fallback |
| `q_proj`+`k_proj`+`v_proj` | SK-03 | INT8 diádico, TC INT8 |
| `o_proj` (Full+MTP) | SK-04 | INT8 TC o W8A16 fallback |
| `gate_proj`+`up_proj` | SK-05 | INT8 diádico fused SiLU, TC INT8 |
| `down_proj` (65×) | SK-06 | INT8 TC / Ozaki |
| `lm_head` + `embed_tokens` | SK-07 + SK-09 | BF16/PN77/INT8, embed BF16 |
| `in_proj_a/b`, `conv1d`, `A_log`/`dt_bias`, norms | SK-08 + SK-09 | BF16 puro, fused gating/conv |
| Vision 333 | SK-11 | BF16 |
| MTP head completo | SK-10 | mirror, TC INT8 |
| FLA GDN attention core (pre/post-proc) | — | 1.2% paso, no kernel dedicado (`KERNELS-OPTIMIZACION.md:98-108`) — fusion marginal `cumsum+kkt` / `conv+prep` (`KERNELS-OPTIMIZACION.md:157-158`) |

> Con 11 SK se cubre el **100% de GEMM FP8** (407 tens.) + el 100% de BF16 no-cuantizado. 0 tensores huérfanos.

---

## 6. Implementación — orden y gates

### 6.1 Prioridad por impacto prefill (Amdahl, `KERNELS-OPTIMIZACION.md:50-59`)

1. **SK-05 (GateUp) + SK-06 (Down)** — MLP = `~60%` del cómputo capa (`gate_up 5.53ms + down 2.43ms` Marlin prefill; INT8 1.61+0.75). **Hacer primero**.
2. **SK-01 (GDN-QKVZ)** — 48 capas → `48×0.44ms GDN + GEMM qkvz`; prefill 2.9×.
3. **SK-03 (FA-QKV) + SK-04/02 (O/OUT)** — 16 capas Full; o_proj/out_proj son 13% si W8A16 (`fp8_a_int8_ampere.md:460`).
4. **SK-07 (LM-Head)** — prefill 33→14 ms (2.8×) pero solo 1 capa; decode ~1 ms (`KERNELS-OPTIMIZACION.md:145`).
5. **SK-08/09 (BF16 passthrough)** — ya fusionados; solo wrapper `scale_pow2`.
6. **SK-10 (MTP)** — ya DONE PN108/PN110 (`PLAN-CHECKPOINTS.md:168-219`).

### 6.2 Diseño de scales (clave, `fp8_a_int8_ampere.md:4`)

- **Diseño C híbrido diádico recomendado** para todos los SK INT8: `w = q·2^shift_b·s_row` (`fp8_a_int8_ampere.md:407`). `s_row` float32 per-channel (`cutlass_scaled_mm` exige float32 `KERNELS-OPTIMIZACION.md:198`) + `shift_b int8` por bloque 128×128 (3200 B para `in_proj_qkv`, `fp8_a_int8_ampere.md:404`). Shift sobre acumulador INT32 barato.
- Alternativa **Diseño A** block-INT8 fiel (flush cada 4 MMAs K=32 → 4 MMAs por chunk 128, `fp8_a_int8_ampere.md:385-387`) como referencia de máxima fidelidad para T2.
- **Diseño B** per-channel puro (max sobre 40 bloques K) usa kernels `compressed-tensors` existentes pero **pierde grilla diádica inter-bloque** (`fp8_a_int8_ampere.md:397-399`) → solo si C falla.

### 6.3 SmoothQuant restringido a potencias de dos (`fp8_a_int8_ampere.md:5.1`)

- Barrer `α∈{0.5,0.6,0.7,0.8}` en **128-512 seqs Pile/C4** (`fp8_a_int8_ampere.md:561-564`), `s_j = max|X_j|^α / max|W_j|^(1-α)`, redondear `s→2^round(log2 s)` (`fp8_a_int8_ampere.md:469`).
- Cuesta ≤√2 en migración vs outlier 20-100× (`fp8_a_int8_ampere.md:472`) — despreciable, preserva grilla → todo el pipeline **diádico de punta a punta** (`fp8_a_int8_ampere.md:475`).
- Absorción: `input_layernorm` para SK-01/03, `post_attention_layernorm` para SK-05 (`fp8_a_int8_ampere.md:448-451`); `down_proj` vía filas/columnas (`fp8_a_int8_ampere.md:452`).

### 6.4 Gates de validación por SK (`fp8_a_int8_ampere.md:8`)

> No pasar de nivel sin cerrar anterior (`fp8_a_int8_ampere.md:591-593`).

| Nivel | Tests | Gate por SK |
|---|---|---|
| **0 estático (minutos, sin GPU)** | **T0 histograma `d`** (`fp8_a_int8_ampere.md:599-619`) — por familia Q/K/V separado + `down_proj` | `p99 d 4-6` sano → seguir; cola `d>8` → granularidad fina/SpQR; `d>10` frecuente → W8A16 |
| | T1 `0x7F` NaN, `e_max≥1` | 0 NaNs |
| | T2 `‖W_fp8-W_int8‖_F/‖W_fp8‖_F` | **A <1e-3, B <1e-2** (`fp8_a_int8_ampere.md:628-632`) |
| | T3 energía `d≥4` ~5e-4 | 5e-4± ruido (`fp8_a_int8_ampere.md:634-635`) |
| **1 numérico GPU** | T4 SQNR `10log10(‖Y_ref‖²/‖Δ‖²)` `α` sweep (`fp8_a_int8_ampere.md:642-648`) | elegir `α` max SQNR por capa |
| | **T5 drift SSM** (`fp8_a_int8_ampere.md:649-654`) 1K/8K/64K/256K | <1e-3 / <5e-3 / <2e-2 / investigar superlineal → SK-01 a A/C o W8A16 |
| | T6 balance Q/K/V (`fp8_a_int8_ampere.md:664-666`) | SQNR Q/K/V Δ<6 dB |
| **2 E2E** | T7 PPL WikiText-2-raw 296907 tok KV BF16 **6.96** (`fp8_a_int8_ampere.md:169`) | ≤7.03 (+1%) |
| | T8 MMLU 84.7/ MMLU-Pro 76.8/ GSM8K 88.7/ CMMLU 80.8 (`fp8_a_int8_ampere.md:164-173`) | ±1 pt / ±1.5 GSM8K |
| | **T9 NIAH 32/128/256K** | **no omitir** — T7/T8 no ve SSM (`fp8_a_int8_ampere.md:682-685`) |
| | T10 abliteración (0-6% rechazo, `fp8_a_int8_ampere.md:687-689`) | SK-02/04/06 no degradan gate |
| | T11 visión, T12 MTP spec K=3 | no-op |
| **3 perf** | T13 micro GEMM por forma (`fp8_a_int8_ampere.md:701`), T14 TTFT/TPOT (`fp8_a_int8_ampere.md:707`) | prefill 2-3×, decode neutro (bandwidth-bound, `fp8_a_int8_ampere.md:708-710`: W8A8 no ahorra bytes) |

### 6.5 Checklist NO-CUANTIZAR (`fp8_a_int8_ampere.md:525-534`)

Antes de emitir checkpoint: `A_log, dt_bias, conv1d (3D!), in_proj_a/b, todas las norms (input/post_attention/q/k/linear_attn/model), lm_head stock, embed_tokens, 333 visual.*` — si aparece INT8 es bug. `conv1d.weight` es **[out_channels,1,4]** no matriz.

---

## 7. Alternativas si INT8 no alcanza (`fp8_a_int8_ampere.md:10`)

| Opción | Cuándo | Nota Ampere |
|---|---|---|
| **QoQ/QServe** `2405.04532` `fp8_a_int8_ampere.md:736-743` | Si Diseño B/C no cierra SQNR | Grupo-progresivo 8→4bits, todo GEMM en TC INT8, rango [-119,119] con desempacado 3 ops |
| **Ozaki 2-slice** `2306.11975` `fp8_a_int8_ampere.md:744-755` | Solo `in_proj_qkv` GDN si T5 falla | 2 GEMM INT8 ≈1×FP16, 15 bits efectivos, sin SmoothQuant |
| **TC-FPx/FP6-LLM** `2401.14112` `fp8_a_int8_ampere.md:760-761` | Si se quiere <8 bits | Testeado A100, bit-splitting |
| **Marlin W4A16** `fp8_a_int8_ampere.md:767` | Si cuello es VRAM (no cómputo) | 12.4 GiB linears vs 24.7 GiB INT8/FP8; contradice calidad medida |

> **No hacer**: LUT-GEMM — no gana en Ampere (`fp8_a_int8_ampere.md:766-770`, `2408.06003`).

---

## 8. Mapa de archivos y referencias

| Ref | Ubicación |
|---|---|
| Spec FP8→INT8, teoria, codigo `fp8_e4m3_to_int8_aligned` | `workshop/ox_alpha/fp8_a_int8_ampere.md:202-365` |
| Formas, dims, packed mapping `qkv_proj/gate_up_proj/in_proj_qkvz/in_proj_ba` | `assets/vllm/vllm/model_executor/models/qwen3_5.py:276-291, 440-450` + `qwen3_5_mtp.py:59-159` |
| Benches GEMM INT8 vs Marlin, sampler, FLA, memcpy, FA A/B | `workshop/ox_alpha/KERNELS-OPTIMIZACION.md:2-126, 2` |
| Pipeline híbrido, context GDN estados, AR bug, inert flags | `workshop/ox_alpha/CONTEXTO-INVESTIGACION.md:7-11` |
| Backports V1/V2, syncs | `workshop/ox_alpha/BACKPORT-V2.md` |
| Plan checkpoints, PN110 full-swap, PN108 | `workshop/ox_alpha/PLAN-CHECKPOINTS.md:2.1-5` |
| Este documento | `workshop/ox_alpha/super_kernels.md` |

---

## 9. Entregable — cómo consumir este doc

1. **Elegir régimen por SK** con T0-T1 (horas) → decide A/B/C/W8A16.
2. Implementar **un SK por parche Genesis** (`GENESIS_ENABLE_SK01_*` etc.), `default_off`, `applies_to: fp8-block ∧ sm∈[80,89]`, rebind `process_weights_after_loading` (chunked 512, `b_col` column-major `nn.Parameter`) + `apply()` cutlass `scaled_mm` (float32 scale) — patrón PN110 (`PLAN-CHECKPOINTS.md:215-219`).
3. Validar T2-T6 por SK, luego T7-T10 E2E, finalmente T13-T14 perf antes de promover a PROD con comentario fechado en `compose/docker-compose.qwen38-27b-fp8.yml`.

> **Nota de portabilidad** (`PLAN-CHECKPOINTS.md:47-53`): SK-01..06/10 son **model-dependientes** (KL/aceptancia por checkpoint); SK-08/09/11 model-agnósticos (runner/allocator).

---

*Generado 2026-08-25 sobre `fp8_a_int8_ampere.md` 804 líneas + `qwen3_5.py` 809 líneas + benches lab `KERNELS-OPTIMIZACION.md`. Todo número citado es reproducible vía `workshop/ox_alpha/fp8_a_int8_ampere.md:8`.*
