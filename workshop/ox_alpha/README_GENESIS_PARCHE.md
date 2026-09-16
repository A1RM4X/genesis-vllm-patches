# Genesis — Parche vLLM v0.23 para Qwen3.8-27B en 2× RTX 3090

> **Versión del documento**: 2026-08-25  
> **Hardware canónico**: 2× RTX 3090 `sm_86` (Ampere GA102, TC `mma.m16n8k32.s8` sí, FP8 nativo **no**)  
> **Modelo canónico**: `orcarouter/Qwen3.8-27B-Uncensored-FP8` → `Qwen3_5ForConditionalGeneration` (64 capas: 48 GDN + 16 Full + 1 MTP)  
> **vLLM pin**: `vllm/vllm-openai:v0.23.0` (`compose/docker-compose.qwen38-27b-fp8.yml:45`)  
> **Autor**: ox-alpha / Genesis (Sandermage — Barzov Aleksandr)

---

## 1. Qué es el parche Genesis

Genesis **no es un fork** de vLLM sino un **conjunto de ~123 parches quirúrgicos** que se inyectan al arrancar el contenedor y convierten un vLLM stock en un servidor de producción afinado para `Qwen3.8-27B FP8` sobre hardware consumer.

Cada parche cumple el contrato defensivo de 5 capas (`vllm/_genesis/patches/apply_all.py:8-15`): existe el archivo → marker de idempotencia → no mergeado upstream → vendor/chip compatible → arquitectura de modelo compatible. Si cualquier capa falla, el parche queda en `skipped` sin romper el boot.

### 1.1 Qué optimiza para este deploy

| Problema en v0.23 stock | Parche Genesis que lo cierra | Métrica medida |
|---|---|---|
| Prefill FP8 por Marlin: 5.5 s (65% del step) | **PN110** INT8 W8A8 prefill 3× vs Marlin (`workshop/ox_alpha/KERNELS-OPTIMIZACION.md:54-58`) | wall 7.21 → 5.56 s (-23%), 17.8 → 23.0 tps |
| Custom all-reduce falla en `cuda_communicator:455` y degrada a PYNCCL | **B3** fast-path TP=2 + fallback robusto (`vllm/_genesis/wiring/communication/patch_B3_custom_ar.py:1-15`) | ~30% del tiempo GPU en AR con TP=2 |
| CUDAGraphs `FULL → FULL_DECODE_ONLY` en prefill largo (drafter pierde grafo) | **B2** fuerza `FULL` (`vllm/_genesis/wiring/cudagraph/patch_B2_full_cg.py:28-33`) | FULL capturado en `speculator.py:84` |
| KV cache crece sin cuota (`/kv-offload` 38 GB) | **PN81** quota + poda (`compose/docker-compose.qwen38-27b-fp8.yml:81-119`) | 0 OOMs con 5×30k paralelos a `gpu-util 0.76` |
| Rejection sampler lanza 5 allocs + H2D por paso | **B5** early-exit + cache LRU (`vllm/_genesis/dispatcher.py:283-289`) | -5-15 µs por batch trivial |
| `lm_head` 248k vocab ocupa 1.2 GiB/GPU sin compresión | **PN77/PN108 + B7** FP8/int8 + fused sampled (`vllm/_genesis/dispatcher.py:1121-1135`) | -630 MiB/rank draft |

El **catálogo completo** de capas cubiertas por los 11 super kernels (SK-01…SK-11) está en `workshop/ox_alpha/super_kernels.md:203-215` y la **matriz N-dimensional** que resuelve `arch × family × kv × quant × M_bucket × TP → KernelSpec` en `workshop/ox_alpha/matriz_kernels.md:64-83`.

### 1.2 Componentes

```
vllm/_genesis/
├── dispatcher.py          PATCH_REGISTRY — 123 entradas: env_flag, category, applies_to, conflicts_with
├── patches/apply_all.py   Orquestador — 5 capas defensivas + summary boot
├── wiring/                11 subdirs (hybrid/, quantization/, communication/, cudagraph/, …) — text-patch / rebind
├── kernels/               sk01…sk11 + fused_quant_gemm.py + int8_hybrid_gemm.py — Triton/CUTLASS kernels
├── guards.py              is_nvidia_cuda(), is_sm_at_least(), resolve_vllm_file()
└── competencia/compat     doctor, version_check, gpu_profile

workshop/ox_alpha/
├── matriz_kernels.md      Matriz N-dim + 3 fases de precarga al boot (§4)
├── super_kernels.md       Catálogo 11 SK + taxonomía 64 capas + gates T0-T13
├── KERNELS-OPTIMIZACION.md Benches 2×3090: INT8 vs Marlin vs cuBLAS por M
└── fp8_a_int8_ampere.md   Teoría FP8→INT8 diádica (Diseño A/B/C) + código conversión
```

---

## 2. Cómo se usa

### 2.1 Requisitos previos

```bash
cp compose/.env.example compose/.env
$EDITOR compose/.env   # mínimo: VLLM_API_KEY=$(openssl rand -hex 32)
# HF_TOKEN si el checkpoint es gated
# Ajustar /home/usuario/Proyectos/models-cache si tu path es distinto
```

`PYTHONHASHSEED=0` debe quedar fijo (`compose/docker-compose.qwen38-27b-fp8.yml:398`) — sin esto el tier de disco genera hashes aleatorios y todo `/kv-offload` queda inalcanzable tras reiniciar.

### 2.2 Boot canónico

```bash
# Desde la raíz del repo
docker compose -f compose/docker-compose.qwen38-27b-fp8.yml up -d

# Logs: debe aparecer "structured boot summary" al final del arranque (~60 s cold, ~10 s warm)
docker logs -f genesis-27b-qwen38-fp8 | grep -A 200 "structured boot summary"

# Verificación
curl -s http://localhost:8320/health | jq
curl -s http://localhost:8320/metrics -H "Authorization: Bearer $VLLM_API_KEY" | grep -E "vllm:kv_tier|process_start_time_seconds"
```

El `entrypoint` del compose (`compose/docker-compose.qwen38-27b-fp8.yml:409-415`) hace exactamente:

```bash
pip install pandas scipy xxhash -q
python3 -m vllm._genesis.patches.apply_all   # aplica todos los parches habilitados
exec vllm serve "$@"                         # arranca vLLM con los args del compose
```

Los volúmenes `vllm/_genesis` (`compose/docker-compose.qwen38-27b-fp8.yml:77`) y los caches `~/.cache/{vllm,torch,triton,torchinductor}` (`compose/docker-compose.qwen38-27b-fp8.yml:72-75`) son montajes persistentes — primer boot compila Triton/CUTLASS, siguientes boots hacen `HIT`.

### 2.3 Flags `GENESIS_ENABLE_*` — un parche, un flag

Todo patch del `PATCH_REGISTRY` (`vllm/_genesis/dispatcher.py:65`) expone un `env_flag`. El orquestador (`vllm/_genesis/patches/apply_all.py:182-191`) solo lo instala si el flag está en `ON` y si `dispatcher.should_apply(patch_id)` devuelve `True` (gates de `applies_to` + `conflicts_with` + `is_sm_at_least`).

**Compose `qwen38-27b-fp8` — flags activos hoy** (`compose/docker-compose.qwen38-27b-fp8.yml:78-371`):

| Flag | Patch | Estado en FP8 | Qué hace |
|---|---|---|---|
| `GENESIS_ENABLE_PN110_INT8_PHASE_DISPATCH` | **PN110** (`vllm/_genesis/dispatcher.py:1155`) | `1` | Requant FP8 bloque→INT8 per-channel chunked + swap 1:1 `b_col` column-major + invalidación Dynamo. Kill switch `GENESIS_DISABLE_PN110=1`. Tuning `GENESIS_PN110_W8A8_MIN_TOKENS`, `GENESIS_PN110_EXCLUDE_LAYERS` |
| `GENESIS_ENABLE_B3_CUSTOM_AR` | **B3** (`vllm/_genesis/dispatcher.py:2659`) | `1` | TP=2 fast-path (`<2 MiB` + 16-byte aligned) + fallback `torch.distributed.all_reduce` en `CudaCommunicator.all_reduce`/`CustomAllreduce.should_custom_ar` |
| `GENESIS_ENABLE_B2_FULL_CG` | **B2** (`vllm/_genesis/dispatcher.py:2668`) | `0` | Fuerza `CUDAGraphMode.FULL` en `speculator.py:84` (evita degradar a `FULL_DECODE_ONLY` con prefill largo). Requiere `B2=1` para test A/B |
| `GENESIS_ENABLE_B5_REJECTION_SAMPLER` | **B5** (`vllm/_genesis/dispatcher.py:283`) | `0` | Vectoriza `expand_batch_to_tokens` + early-exit si `num_tokens==0` + cache LRU 32 del patrón de expansión |
| `GENESIS_ENABLE_B7_LM_HEAD` | **B7** (`vllm/_genesis/dispatcher.py:1121`) | `0` | Fused sampled logits + tie-embed quant FP8 per-channel del lm_head `248k×5120` |
| `GENESIS_ENABLE_PN77_FP8_LM_HEAD` | **PN77** (`vllm/_genesis/dispatcher.py:1087`) | `1` | BF16→FP8 E4M3 del `lm_head` target (~606 MiB/rank) |
| `GENESIS_ENABLE_PN108_DRAFT_FP8_LM_HEAD` | **PN108** (`vllm/_genesis/dispatcher.py:1101`) | `1` | Extiende PN77 al drafter MTP (`SpecDecodeBaseProposer.load_model`) — -630 MiB/rank |
| `GENESIS_ENABLE_PN81_KV_DISK_QUOTA` | **PN81** | `1` | Cuota `30 GB` + poda por mtime cada `60 s` + purga orphans cada `3600 s` |
| `GENESIS_ENABLE_PN88_KV_TIER_METRICS` | **PN88** | `1` | `kv_tier_bytes_total/latency/errors/lookups` por `tier/direction/agent/group` |
| `GENESIS_ENABLE_PN93_SPARSE_GDN` | **PN93** (`vllm/_genesis/dispatcher.py:1170`) | `1` | 1 de cada `16` fronteras GDN (stride `16`) — 75% → 32% costo KV |
| `GENESIS_ENABLE_PN59_STREAMING_GDN` | **PN59** | `1` | Window-iterative `chunk_gated_delta_rule_fwd_h` — elimina Cliff 2b OOM |
| `GENESIS_ENABLE_PN50_GDN_FUSED_PROJ` | **PN50** | `1` | Triton fused `qkvzba_split_reshape_cat_contiguous` (SGLang #21019) |
| `GENESIS_ENABLE_PN57_TQ_CENTROIDS_DISK_CACHE` | **PN57** | `1` | Cache Lloyd-Max `~/.cache/genesis/turboquant_centroids.pkl` |
| `GENESIS_ENABLE_PN80_GDN_H_BUDGET_PROBE` | **PN80** | `0` | Sonda VRAM `h = B*NT*H*V*K` — proyección tokens/forward |
| `GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL` | **PN12** | `1` | Pool `SiluAndMul` — ataca OOM inductor `108 MiB = T×17408 fp16` |
| `GENESIS_ENABLE_PN25_SILU_INDUCTOR_SAFE` | **PN25** | `1` | Pool opaque-op para grafo inductor del FFN |
| `GENESIS_ENABLE_PN19_SCOPED_MAX_SPLIT` | **PN19** | `1` | `max_split_size_mb=20` solo durante carga → restaura al terminar |
| `GENESIS_ENABLE_P66_CUDAGRAPH_SIZE_FILTER` | **P66** | `1` | Filtra `--cudagraph-capture-sizes` a múltiplos de `K+1` (MTP=3) |
| `GENESIS_ENABLE_P5B` | **P5B** | `1` | `block_size 1600` (mamba align) — -34% VRAM por bloque KV |
| `GENESIS_ENABLE_P100` | **P100** | `1` | FULL cudagraphs spec-decode |
| `GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT` | **PN8** | `0` | No recomendado en este GPU (FP8 ya es estático) |
| `GENESIS_ENABLE_MTP_QUANT_CACHE` | **P112** | `0` | Disk cache INT8 draft — apagado (round-trip FP8→INT8→FP16 sin beneficio, ver `vllm/_genesis/mtp_cache.py`) |

> **Regla**: para un A/B limpio por parche (`workshop/ox_alpha/PLAN_AB_POR_PARCHE.md`), dejar **solo ese flag en `1`** y el resto como en baseline. Ejemplo PN110 aislado: `PN110=1, B2/B3/B5/B7=0` (`compose/docker-compose.qwen38-27b-fp8.yml:288-294`).

**Forzar o inhibir desde `dispatcher.py:2832`** (`should_apply`):

```bash
# Encender B2 + PN110 y forzar que B3 no entre aunque esté en 1 en el compose
GENESIS_ENABLE_B2_FULL_CG=1 GENESIS_ENABLE_PN110_INT8_PHASE_DISPATCH=1 GENESIS_DISABLE_B3=1 \
  docker compose -f compose/docker-compose.qwen38-27b-fp8.yml up -d
```

El orden de precedencia es: `GENESIS_DISABLE_*=1` > `should_apply()` > `env_flag=1` > `default_on`. El `dispatcher` escribe una línea por parche con el motivo del `skip` (`log_decision`).

---

## 3. Matriz de kernels — dónde vive y cómo se edita

La matriz es el contrato ` (arch, family, kv_dtype, quant, M_bucket, TP) → KernelSpec { import_path, backend, geometry, cache_key }` y sustituye la intuición 2D (`tipo×KV`) por una clave compuesta N-dimensional (`workshop/ox_alpha/matriz_kernels.md:64-83`).

```
E1 arch         sm_86 (3090) | sm_89 (4090) | sm_90 (H100) | sm_100 (B200)
E2 family/role  gdn_qkvz | gdn_out | fa_qkv | fa_o | mlp_gateup | mlp_down | lm_head | ssm_control | norm_embed | vision | mtp_draft_*
E3 quant        fp8_block128_e4m3 | int8_diadic_c | int8_per_channel_b | w8a16 | bf16 | w4a16_marlin
E4 kv_dtype     fp8_e4m3 (canon) | bf16 | fp16
E5 M_bucket     1 | 8 | 32 | 128 | 512 | 1664 | 8000   (sk03 ya codifica M_BUCKETS en vllm/_genesis/kernels/sk03_fa_qkv.py:81)
E6 TP+shard     1 | 2 (canon) | 4 | 8
E7 epilog/fusion  derivado (silu_and_mul, allreduce+residual, …)
E8 MTP_role     target | draft
```

### 3.1 Dónde editarla

| Artefacto | Ruta | Qué se hace ahí |
|---|---|---|
| **Diseño / fuente versionada** | `workshop/ox_alpha/matriz_kernels.md` | Editar §2.1 (ejes) y §6.1 (tabla plana por `family×M_bucket`). Es el **único** documento que la lista humana debe tocar. |
| **Registry runtime** | `vllm/_genesis/kernels/__init__.py` + `vllm/_genesis/kernels/kernel_registry.py` (o `matriz_kernels_registry.json` generado por `tools/gen_kernel_table.py` — `workshop/ox_alpha/matriz_kernels.md:164-166`) | Registrar la nueva fila `KernelKey → KernelSpec`. Contrato: `KernelSpec { import_path, triton_kernel, backend, N, K, scale_dtype, shifts, fusion, cache_key, requires }` (`matriz_kernels.md:120-138`) |
| **Kernel** | `vllm/_genesis/kernels/skNN_*.py` | Añadir la implementación (ver §4) |
| **Dispatcher gate** | `vllm/_genesis/dispatcher.py:65` `PATCH_REGISTRY` | Declarar `env_flag`, `category`, `applies_to: { is_hybrid, model_class, quant_format, sm_* }`, `conflicts_with`, `requires_patches` |
| **Orquestador** | `vllm/_genesis/patches/apply_all.py:185` `@register_patch` | Añadir `apply_patch_*()` que delega a `vllm._genesis.wiring.*:apply()` |
| **Wiring** | `vllm/_genesis/wiring/{category}/patch_*.py` | `TextPatcher` / `rebind` + `marker` idempotente + `apply() -> (status, reason)` |

Ejemplo mínimo para un nuevo `arch`:

```python
# vllm/_genesis/dispatcher.py — nueva fila (misma family, distinto arch y backend)
"SK05-mlp_gateup-sm90": {
    "title": "SK-05 GateUp fused FP8 blockScaled (Hopper wgmma)",
    "env_flag": "GENESIS_ENABLE_SK05_GATEUP_SM90",
    "category": "kernel_perf",
    "applies_to": {"is_hybrid": [True]},
}

# vllm/_genesis/kernels/kernel_registry.py — celda de la matriz
KernelKey = ("sm_90", "mlp_gateup", "fp8_e4m3", "fp8_blockScaled", 1664, 2)
table[KernelKey] = KernelSpec(
    key=KernelKey,
    import_path="vllm._genesis.kernels.sk05_mlp_gateup_hopper:mlp_gateup_fp8_blockscaled",
    triton_kernel=None,
    backend="cutlass",  # DeepGEMM / FlashInferFp8DeepGEMM en sm_90 (CONTEXTO-INVESTIGACION.md:8.1)
    N=17408, K=5120,
    scale_dtype="float32",
    shifts=None,  # FP8 nativo no necesita shift diádico
    fusion="RMSNorm(post)+quant -> GEMM GateUp -> SiLU*Mul",
    cache_key="sha256:…",
    requires="T0 d<=3, T2 <1e-3",
)
```

La tabla es **sparse** (`matriz_kernels.md:144-152`): solo existen las filas que el checkpoint (`orcarouter/...` 407 tensores FP8 + 407 `weight_scale_inv`, `super_kernels.md:25`) y la combinación `arch/kv/M/TP` realmente necesitan — ~30-40 filas para `(sm_86, fp8_e4m3, TP=2)`, no las 7600 teóricas del cartesiano.

---

## 4. Cómo agregar otras arquitecturas

### 4.1 Ejemplo `sm_89` (Ada, RTX 4090) y `sm_90` (Hopper, H100)

| `family` | `sm_86` canónico (3090) | `sm_89` (4090) | `sm_90` (H100) |
|---|---|---|---|
| `mlp_gateup` | `sk05:mlp_gateup_fused_int8_diadic` Triton `m16n8k32.s8` + `cutlass_scaled_mm` | `cutlass_fp8` o `triton_fp8` nativo (TC FP8 `mma` existe) → `w8a8_fp8` sin diádico | `DeepGEMM` / `FlashInferFp8DeepGEMM` blockScaled (`matriz_kernels.md:322`) |
| `lm_head` | `sk07:cublas_bf16` o `cutlass_int8` | `cublas_fp8` viable | `cutlass_fp8` vocab-parallel FP8 nativo |
| `ssm_control` | `sk08:_sk08_fused_decode_packed_kernel` WIDTH=4 | idem, `num_warps/BLOCK_D` retuneado AD102 | `TLX` fused más agresivo |

La matriz lo absorbe **sin rediseño**: se añade `arch="sm_89"` / `"sm_90"` con misma `family/kv/M/TP` pero distinto `import_path`/`backend`. El `sm_86` queda intacto — cero regresión (`matriz_kernels.md:372-376`).

**Pasos concretos para `sm_89`**:

1. **Matriz** — Añadir §6.2 / §6.1 fila con `arch=sm_89` en `workshop/ox_alpha/matriz_kernels.md:316-324`.
2. **Kernel** — Crear `vllm/_genesis/kernels/sk05_mlp_gateup_sm89.py` (o sufijo `_ada`) con el kernel FP8 nativo; mantener el Triton diádico solo para `sm_86`.
3. **Dispatcher** — `vllm/_genesis/dispatcher.py:65` añadir entrada con `env_flag=GENESIS_ENABLE_SK05_SM89` y `applies_to` que gatee por `is_sm_at_least(8,9)` (vía `vllm/_genesis/guards.py:1-40`) o etiqueta `sm_89` en `gpu_profile`.
4. **apply_all** — `vllm/_genesis/patches/apply_all.py:185` añadir `@register_patch("SK05 sm_89 …")` que delega a `vllm._genesis.wiring.kernels.patch_xxx.apply`.
5. **Wiring** — `vllm/_genesis/wiring/kernels/patch_sk05_sm89.py` con marker `Genesis SK05 sm_89 …` y `TextPatch`/`rebind` idempotente (patrón `patch_B2_full_cg.py:95-120`). `vllm/_genesis/wiring/` ya indexa recursivamente (`apply_all.py:218-239`), así que el subdir (`kernels/`, `quantization/`, `communication/`, `cudagraph/`) es libre.

> Nota de portabilidad (`super_kernels.md:431`): SK-01…06/10 son model-dependientes (KL/aceptancia por checkpoint); SK-08/09/11 son model-agnósticos (runner/allocator).

### 4.2 Agregar un nuevo tipo de capa

Si el checkpoint trae una familia GEMM no cubierta (ej. un nuevo `MoE gate` o un `vision_proj` distinto a SK-11):

1. Actualizar la **taxonomía** `workshop/ox_alpha/super_kernels.md:45-101` (tabla por tensor `HF name → forma global/por rank → cuant → rol → fusión`) y la **matriz de regímenes** `super_kernels.md:104-118` (elige A/B/C por `family`).
2. Crear la fila `family` nueva (E2) siguiendo el template de §3.1.
3. Implementar el kernel en `vllm/_genesis/kernels/` con el nombre `skNN_<family>.py` (siguiente SK disponible, hoy hasta `sk11_vision.py`) — debe exponer una función `forward(hidden, b_col, scales, shifts)` y opcionalmente un Triton `@triton.jit` interior (patrón `sk05_mlp_gateup.py:28-94`).
4. Repetir dispatcher → apply_all → wiring como en 4.1.

Referencia de formas reales por rank TP=2 (`KERNELS-OPTIMIZACION.md:16-32`, `super_kernels.md:50-101`):

```
hidden 5120, intermediate 17408, vocab 248320, heads GDN 8K/24V×128, Full 24Q/4KV×256
qkv_proj 7168×5120 | o_proj 5120×3072 | gate_up 17408×5120 | down 5120×8704 | lm_head 124160×5120
```

---

## 5. Cómo agregar un nuevo super kernel

> **Definición** (`super_kernels.md:3`): *un kernel CUDA/Triton/CUTLASS que fusiona en **un solo launch** `{RMSNorm + SmoothQuant 2^k + per-token quant INT8} → GEMM INT8 TC (mma.m16n8k32.s8) → {bias/act/scale epilog} → {AllReduce/RS+AG} → residual`* — 1 launch vs 3-5, 0 materialización intermedia, TC INT8 77 TFLOPS vs 35 FP16 en 3090.

### Paso 1 — Identificar la capa y su forma

1. **Enumera las capas** recorriendo `config.json` (`num_hidden_layers=64, hidden_size=5120, intermediate_size=17408, full_attention_interval=4, vocab=248320`) y el modelo `assets/vllm/vllm/model_executor/models/qwen3_5.py:276-291` (fusiones `qkv_proj`, `gate_up_proj`, `in_proj_qkvz`, `in_proj_ba`). Distingue 48 GDN (`linear_attn.*`) vs 16 Full (`self_attn.*`) — ver `super_kernels.md:16-23`.
2. **Lee el tensor HF**: nombre (`linear_attn.in_proj_qkv.weight`), forma global `N×K` y por rank TP=2, régimen A/B/C y si hay norma delante (`super_kernels.md:50-60`). Valida que `N,K % 128 == 0` y `BLOCK_N` viable (`sk01:98-102`, `sk02:80-82`). Si no lo son, el kernel no puede usar `SHIFT_BLOCK=128` sin padding.
3. **Determina `M_bucket`**: decode `1` (single-seq), `8` (spec K=3, `rejection_sampler.py:119-197`), `1664` (chunked prefill, `KERNELS-OPTIMIZACION.md:50-59`), `8000` (262K NIAH). Cada SK no necesita todos — la matriz registra solo los `M` que realmente aparecen para esa `family` (`matriz_kernels.md:288-315`).
4. **Elige régimen INT8** con `fp8_a_int8_ampere.md:32-154` + `KERNELS-OPTIMIZACION.md:50-59` (shootout GEMM): A (diádico sin pérdida `d≤3` → `rel_err <1e-3`), B (BF16 sin pérdida para `A_log/dt_bias/conv1d/norms`), C (W8A8 con pérdida donde no hay norm para absorber `s=2^k` — `o_proj`/`down_proj`).

### Paso 2 — Crear el kernel en `vllm/_genesis/kernels/`

Patrón canónico (`sk05_mlp_gateup.py:1-148`, `sk02_gdn_out.py:242-502`, `sk01_gdn_qkvz.py:299-636`):

```python
# vllm/_genesis/kernels/sk12_moe_gate.py
"""SK-12 MoE_GATE_FUSED_INT8_DIADIC — W8A8 diádico sobre MoE gate 8192×5120."""
from __future__ import annotations
import torch, triton, triton.language as tl

HIDDEN_SIZE = 5120
GATE_N_PER_RANK = 4096
BLOCK_M, BLOCK_N, BLOCK_K, SHIFT_BLOCK = 32, 64, 32, 128

@triton.jit
def _sk12_moe_gate_kernel(a_ptr, b_ptr, out_ptr, b_scale_ptr, shifts_ptr, M, N, K, ...):
    # Branchless: solo tl.load/dot/where/store. Validación en warmup, no en hot path
    ...

def moe_gate_fused_int8_diadic(hidden, gate_weight, gate_scale, gate_shifts, norm_weight=None, eps=1e-6, out_dtype=torch.bfloat16):
    # 1. RMSNorm bf16 --pow2--> quant per-token INT8 (branchless, sin if)
    # 2. tl.dot INT8 (mma.m16n8k32.s8.s32) sobre acc INT32
    # 3. shift sobre INT32 antes de multiplicar por scale float32 (Diseño C, fp8_a_int8_ampere.md:402-414)
    ...
    return out

moe_gate_fused = moe_gate_fused_int8_diadic
__all__ = ["moe_gate_fused_int8_diadic"]
```

Reglas (`super_kernels.md:322-330`, `fp8_a_int8_ampere.md:5.1`):

* Peso offline: `b_col = w.t()` column-major `nn.Parameter` chunked 512 (`sk02:498-502`, `PLAN-CHECKPOINTS.md:215-219`) + `b_scale [N] float32` (CUTLASS sm80 exige float32, `KERNELS-OPTIMIZACION.md:198`) + `shifts [K/128,N/128] int8` solo Diseño C.
* Sin branches Python en `forward` — toda validación (`K%128, N%128, dtype, contiguity`) en `warmup_all_kernels` de `patch_PN110_int8_phase_dispatch.py:134-160`.
* Excluir GDN/mamba incondicionalmente si toca (`patch_PN110:730-769` — `gdn`/`mamba` en nombre o tipo → `excluded`).
* Manejar `tie_word_embeddings=True` (storage compartido `embed`+`lm_head`) con canario `cos≥0.999` si afecta (B7, `dispatcher.py:1121`).

### Paso 3 — Registrarlo en la matriz

1. **Matriz humana** — `workshop/ox_alpha/matriz_kernels.md:288-315` añade fila en la tabla plana:

```
| SK-12 | moe_gate | moe.gate.weight 8192×5120 (4096×5120/rank) | 1/8/1664 | int8_diadic_c | vllm._genesis.kernels.sk12_moe_gate:moe_gate_fused_int8_diadic (_sk12_moe_gate_kernel) | Triton | GEMM→softmax→topk |
```

2. **Registry runtime** — añade clave en la dict plana (`matriz_kernels.md:109-138`):

```python
table[("sm_86","moe_gate","fp8_e4m3","int8_diadic_c",1664,2)] = KernelSpec(
    import_path="vllm._genesis.kernels.sk12_moe_gate:moe_gate_fused_int8_diadic",
    backend="triton", N=4096, K=5120, scale_dtype="float32",
    shifts="int8 [32,32]", fusion="RMSNorm+quant->GEMM gate",
    cache_key="sha256:…", requires="T0 d<=3, T2<1e-2",
)
```

3. **Ciclo de vida al boot** — la matriz se precarga en fase 1 (escaneo) + fase 2 (compilación/caching en `/root/.cache/{triton,torchinductor,vllm}`) + fase 3 (dispatch `O(1)` sin `if`) — `matriz_kernels.md:174-244`. Invalidación: cambia `arch`/`quant`/`BLOCK_SIZE`/pin → nuevo `cache_key = sha256(arch+triton_src+N,K,M+quant+torch.__version__)` (`matriz_kernels.md:403-404`).

### Paso 4 — Añadir tests

El repo exige **TDD rojo→verde** antes de merge (`vllm/_genesis/README.md:146-157`, `docs/PATCHES.md` asociado). Mínimo:

| Test | Archivo | Qué valida | Gate |
|---|---|---|---|
| **T0** histograma `d` | `workshop/ox_alpha/tests/test_skNN_t0_d_histogram.py` | `p99 d 4-6` sano; cola `d>8`→granularidad fina; `d>10` frecuente→W8A16 (`fp8_a_int8_ampere.md:599-619`) | Estático, sin GPU (minutos) |
| **T2** `‖W_fp8-W_int8‖/‖W_fp8‖` | `vllm/_genesis/tests/test_skNN_fp8_int8_relerr.py` | `A <1e-3`, `B/C <1e-2` (`fp8_a_int8_ampere.md:627-632`) | CPU ok, `torch.float8_e4m3fn` |
| **T4** SQNR sweep `α` | `vllm/_genesis/tests/test_skNN_sqnr_sweep.py` | Elige `α∈{0.5,0.6,0.7,0.8}` max SQNR por capa (`fp8_a_int8_ampere.md:642-648`) | GPU |
| **T5** drift SSM | `vllm/_genesis/tests/test_skNN_drift_ssm.py` | `<1e-3 @1K, <5e-3 @8K, <2e-2 @64K` (`fp8_a_int8_ampere.md:649-654`) — solo GDN | GPU 64K/256K NIAH |
| **T13** micro GEMM | `workshop/ox_alpha/lab/scripts/bench_skNN_gemm.py` | Prefill `2-3×` vs Marlin, decode neutro (bandwidth-bound `fp8_a_int8_ampere.md:708`) | `ops.cutlass_scaled_mm` + Triton |
| **Integración** | `vllm/_genesis/tests/test_pnXXX_skNN.py` | `compute_kl_per_layer` vs threshold `0.05` (`patch_PN110:100-103`) + `is_applied()` idempotencia + marker `__genesis_*_wrapped__` | CI `pytest` |
| **Checklist NO-CUANTIZAR** | `vllm/_genesis/tests/test_skNN_no_quant_checklist.py` | `A_log, dt_bias, conv1d 3D [10240,1,4], in_proj_a/b, norms, lm_head stock, embed_tokens, visual.*` nunca INT8 (`fp8_a_int8_ampere.md:525-534`) | Static |

Ejecutar:

```bash
# Sin GPU (T0-T2)
python -m pytest vllm/_genesis/tests/test_skNN_* -v

# Con GPU (T4-T5-T13)
python -m pytest workshop/ox_alpha/tests/test_skNN_* -v --cuda

# En contenedor 2×3090 — A/B real del parche (solo después de T0-T13 en verde)
# workshop/ox_alpha/bench_comparativo.py + PLAN_AB_POR_PARCHE.md CK-2.4
```

> **Orden obligatorio** (`fp8_a_int8_ampere.md:591-593`): no pasar de T0-T1 a T5/E2E sin cerrar T2, ni promover a PROD sin T7-T10 (PPL 6.96, MMLU 84.7%, GSM8K 88.7% ±1 pt, NIAH 32/128/256K — `super_kernels.md:385-390`). Un INT8 mal calibrado en `down_proj`/`o_proj` compone error `∝√L` en el estado GDN y T5 lo detecta a 64K aunque PPL esté ok.

---

## 6. Mapa de archivos — dónde mirar

| Ref | Ubicación |
|---|---|
| Compose PROD FP8 (canon) | `compose/docker-compose.qwen38-27b-fp8.yml` |
| Compose + panel demanda | `compose/docker-compose.yml` (profiles `demand-monitor`) |
| Env ejemplo | `compose/.env.example` |
| Dispatcher — PATCH_REGISTRY + `should_apply()` | `vllm/_genesis/dispatcher.py:65,2691` |
| Orquestador — 5 capas + `register_patch` | `vllm/_genesis/patches/apply_all.py:182-191` |
| Guards — vendor/chip/model gating | `vllm/_genesis/guards.py` ( `is_nvidia_cuda`, `is_sm_at_least` ) |
| Wiring — 11 categorías | `vllm/_genesis/wiring/{hybrid,quantization,communication,cudagraph,spec_decode,structured_output,perf_hotfix,kernels,…}/patch_*.py` |
| Kernels — SK-01…SK-11 + fused paths | `vllm/_genesis/kernels/sk*.py`, `fused_quant_gemm.py:1-30`, `int8_hybrid_gemm.py`, `warmup_all_kernels.py` |
| Super kernels — catálogo + taxonomía | `workshop/ox_alpha/super_kernels.md` |
| Matriz N-dim — ejes + tabla plana + lifecycle | `workshop/ox_alpha/matriz_kernels.md` |
| Teoría FP8→INT8 diádica + Diseño C | `workshop/ox_alpha/fp8_a_int8_ampere.md` |
| Benches GEMM por M + sampler + FLA | `workshop/ox_alpha/KERNELS-OPTIMIZACION.md:16-58` |
| Plan checkpoints PN110/PN108 + CK-2.4 gate | `workshop/ox_alpha/PLAN-CHECKPOINTS.md` |
| Plan A/B por parche | `workshop/ox_alpha/PLAN_AB_POR_PARCHE.md` |
| Circuito token + CONTEXTO investigación | `workshop/ox_alpha/CIRCUITO-TOKEN.md`, `workshop/ox_alpha/CONTEXTO-INVESTIGACION.md` |
| Engineering README del package | `vllm/_genesis/README.md` |
| Operator README + install one-liner | `README.md` (root) |

---

## 7. Troubleshooting rápido

| Síntoma | Causa | Fix |
|---|---|---|
| `sm.fill_(-1) -> CUDA error: invalid argument` al capturar graphs con TP=2 | `cudaHostRegister` sticky del segundo rank (`PN82`) o memoria grafo insuficiente | Verificar `GENESIS_ENABLE_PN82_HOST_REGISTER_STICKY=1` + bajar `gpu-util 0.745` o `--max-num-seqs 10` (`diagnóstico DIAGNOSTICO-OOM-qwen38-27b.md:372`) |
| `b_q_weight.size(0)=5120 Shape mismatch` tras activar PN110 | Cache Dynamo staled (grafo Marlin vs INT8) | `patch_PN110_int8_phase_dispatch.py:772-829` ya hace `torch._dynamo.reset()` + borra `/root/.cache/vllm/torch_compile_cache`; si persiste: `docker exec genesis-27b-qwen38-fp8 rm -rf /root/.cache/vllm/torch_compile_cache` + restart |
| `kv_tier_bytes_total` sube pero `kv_tier_disk_written_bytes_total` es 0 | `store_block` deduplica por `os.path.exists` — cada bloque único se cuenta pero no se re-escribe (ver `compose/docker-compose.qwen38-27b-fp8.yml:134-142`) | Medir desgaste con `rate(kv_tier_disk_written_bytes_total[5m])` no con `kv_tier_bytes_total` |
| Boot: muchos `SKIPPED (anchor not found)` | Pin drift (`v0.23.0 → v0.24.0`) — anclas movidas | Volver al pin `vllm/vllm-openai:v0.23.0` (`compose/docker-compose.qwen38-27b-fp8.yml:45`) o abrir issue con `vllm --version` |
| `/kv-offload` 30 GB lleno y 0 lookups de disco | Tabla vacía: `store_threshold=2` sin hits previos la dejó a `58 MB` (`compose/docker-compose.qwen38-27b-fp8.yml:623-638`) | `GENESIS_ENABLE_PN96_SCAN_RESISTANT_ADMISSION` + `store_threshold=1` (medido: `1` sí usa L2) |

---

*Slice `sm_86` generado sobre `super_kernels.md` 435 lín. + `fp8_a_int8_ampere.md` 804 lín. + `matriz_kernels.md` 431 lín. + 11× `sk*.py` (~5500 lín.) + benches `KERNELS-OPTIMIZACION.md:50-59` (2×3090). Toda celda apunta a un `import_path` verificable en `vllm/_genesis/kernels/` y a un `cache_key` en `/root/.cache/{vllm,triton,torchinductor}`.*

