# Matriz de Selección de Kernels — Arquitectura N-Dimensional

> **Objetivo**: definir la matriz que mapea `(arquitectura, capa, KV, quant, M, TP, …)` →
> `path del kernel a compilar/cachear al arrancar vLLM`. Sustituye la
> intuición inicial "2D = tipo de capa × KV" por una estructura N-dimensional
> con tuning por `sm_86` (3090) y precarga determinista al boot.

| | |
|---|---|
| **Fecha** | 2026-08-25 |
| **Hardware canon** | RTX 3090 `sm_86` (Ampere GA102, TC `mma.m16n8k32.s8` sí, FP8 NO) |
| **Modelo canon** | `orcarouter/Qwen3.8-27B-Uncensored-FP8` → `Qwen3_5ForConditionalGeneration`, 64 capas (48 GDN + 16 Full + 1 MTP), TP=2 |
| **KV canon** | `fp8_e4m3` (`compose/docker-compose.qwen38-27b-fp8.yml:446`, `GENESIS_PN92_KV_DTYPE`) + `mamba_ssm_cache_dtype=float16` separado |
| **Base** | `workshop/ox_alpha/super_kernels.md:1-435` (11 SK) + `vllm/_genesis/kernels/sk*.py` (SK-01…SK-11) |
| **Estado** | Diseño puro — **sin código** |

---

## 0. Resumen — contrato

```
Al arrancar, vLLM escanea TODO (capas, geometrías, KV, TP, arch) y
precarga en VRAM+cache TODOS los kernels que ese modelo usará en su vida
(prefill + decode, GDN + Full + MLP + lm_head).

En inferencia solo hace:

    out = KERNEL_TABLE[key][M_bucket](hidden, b_col, scales, shifts)   # O(1), sin if

Sin branches, sin checks, sin recompilación.
```

**Tabla** = diccionario `clave compuesta → KernelSpec { import_path, fn_ptr, cache_key }`.
La **célula** apunta al `path` del kernel a compilar (ej.
`vllm._genesis.kernels.sk01_gdn_qkvz:sk01_gdn_qkvz_gemm`), no a un
string genérico. Ese kernel se `torch.compile` / `triton.jit` / `CUDAGraph`
al boot y su código queda en `/root/.cache/vllm/torch_compile_cache` +
`/root/.cache/triton` + `/root/.cache/torchinductor` (montajes persistentes
del compose `super_kernels.md:144-172`).

---

## 1. Por qué 2D (`KV × tipo_capa`) es insuficiente

La intuición inicial (2 columnas porque "solo usamos FP8") colapsa en
cuanto se explota un tensor real:

| Eje oculto | Ejemplo concreto que rompe la 2D | Consecuencia |
|---|---|---|
| **Arquitectura `sm`** | Mismo `gate_proj` en `sm_86` (Ampere) requiere `mma.m16n8k32.s8` INT8 con dequant FP8→INT8 diádico (`fp8_a_int8_ampere.md:54-66`), en `sm_90` (Hopper) existe `wgmma` FP8 nativo y CUTLASS `cutlass_scaled_mm` FP8 blockScaled (`super_kernels.md:10`, `CONTEXTO-INVESTIGACION.md:8.1`). | El kernel óptimo cambia aun con **misma** capa y mismo KV → 2D elige mal o deja 3× de prefill sin explotar (`KERNELS-OPTIMIZACION.md:50-58`: INT8 2.8-3.4× vs Marlin en `sm_86`). |
| **`M` (tokens batcheados)** | Mismo `out_proj` `5120×3072` (por rank) en decode `M=1` es bandwidth-bound (~0.035 ms, `KERNELS-OPTIMIZACION.md:44`) y gana con Marlin/bf16; en prefill `M=1664` es compute-bound (0.29 ms INT8 vs 0.86 ms Marlin, 3×). El `BLOCK_M/BLOCK_N` y `num_warps` óptimos cambian. | Una sola celda `o_proj×fp8` no puede servir a los dos regímenes sin dejar perf o sin recompilar en hot path. |
| **`quant_method` (peso)** | Hoy checkpoint es `fp8_block128` E4M3, pero el SK decide **destino**: `int8_diadic_C` (Diseño C con `shift` sobre INT32, `sk01_gdn_qkvz.py:14-24`), `int8_per_channel` (Diseño B, `compressed-tensors`), `w8a16` keep (fallback T5, `sk02_gdn_out.py:242-243`), `bf16` (SK-08/09/11), `w4a16_marlin` (12.4 GiB, `fp8_a_int8_ampere.md:710`). SK-02 `out_proj` puede ser `int8_c` o `w8a16` según T5 `>2e-2 @64K` (`super_kernels.md:109`). | 2D no distingue `out_proj@fp8` diádico vs `out_proj@w8a16` — debe existir la tercera dimensión o la abliteración/SSM deriva (`fp8_a_int8_ampere.md:484-498`). |
| **`kv_cache_dtype`** | `fp8_e4m3` (FlashInfer FA2 dequant in-kernel, `CONTEXTO-INVESTIGACION.md:419`) vs `bf16` (PPL 6.96 target, `fp8_a_int8_ampere.md:169`) cambian el dtype del **epílogo de atención**, no del GEMM, pero el super kernel FA-QKV debe saber si insertar `q_norm/k_norm` BF16 después (`sk03_fa_qkv.py:250`) o no. | Mezclar KV en misma celda contamina validación T7/T9. |
| **`TP` / geometría shardeada** | `gate_up` global `34816×5120` vs por rank `17408×5120` (`sk05_mlp_gateup.py:22-28`); `lm_head` global `248320×5120` (2.46 GiB) vs por rank `124160×5120` (`sk07_lm_head.py:22-28`); `out_proj` `K=6144` vs `K=3072`. El kernel Triton exige `K,N % 128 == 0` y `BLOCK_N=64` óptimo depende de `N` (`sk01_gdn_qkvz.py:98-102`, `sk02_gdn_out.py:80-82`). | Sin TP la geometría no cuadra y el `shifts` `[K/128,N/128]` tiene shape distinto (`sk01` global `[40,128]` vs per-rank `[40,64]`). |
| **Fusión / epílogo** | `gate_up` con `silu_and_mul_quant` solo existe en W8A8 (`super_kernels.md:113`, `KERNELS-OPTIMIZACION.md:151`), no en W8A16; `norm→quant` fused solo si downstream es INT8 (`sk09_norm_embed.py:26-28`). | La 2D no captura si la celda debe ser 1 launch fused vs 2 launches. |
| **Draft / MTP** | MTP draft `gate_up` espeja SK-05 pero con `b_col` column-major ya transpuesto y `TP` independiente (`sk10_mtp_draft.py:2-10`). | Sin dimensión draft/target el draft cae en fallback `0x` y pierde -630 MiB (`super_kernels.md:317`). |

**Conclusión**: una matriz 2D cruza bien para un demo FP8, pero en cuanto se
introduce **un** eje real (arch, M, per-channel vs diádico, TP) el espacio
explota y la tabla deja de ser clave primaria. Necesitamos clave compuesta.

---

## 2. Ejes N-dimensionales propuestos

### 2.1 Ejes canónicos (6 obligatorios + 2 opcionales)

| # | Eje | Dominio (ejemplo) | Por qué existe | Cardinalidad efectiva (sparse) |
|---|---|---|---|---|
| **E1** | `arch` | `sm_80`, `sm_86` (3090), `sm_89` (4090), `sm_90` (H100), `sm_100` (B200) | TC disponibles: `mma.m16n8k32.s8` (Ampere) vs `wgmma` FP8/FP4 (Hopper/Blackwell) vs `mma.sp` INT4. Cambia `cutlass_scaled_mm` vs Marlin vs DeepGEMM (`CONTEXTO-INVESTIGACION.md:8.1`). Sin esto no hay tuning "super tuneado por arquitectura". | 2-3 activos por deploy (hoy solo `sm_86`) |
| **E2** | `family` / `role` | Ver catálogo SK §2.3 (17 valores con MTP) | La unidad mínima direccionable no es "capa 7" sino **familia de GEMM fusionada**: `gdn_qkvz`, `gdn_out`, `fa_qkv`, `fa_o`, `mlp_gateup`, `mlp_down`, `lm_head`, `ssm_control`, `norm_embed`, `vision`, `mtp_draft_{qkv,gateup,down,lm_head}` (`super_kernels.md:203-215`). | 11 SK → 17 con sharding MTP |
| **E3** | `quant` | `fp8_block128_e4m3` (origen), `int8_diadic_c` (Diseño C `fp8_a_int8_ampere.md:402-414`), `int8_per_channel_b`, `w8a16` (fallback T5), `bf16` (passthrough), `w4a16_marlin` | Destino de la conversión offline `fp8_e4m3_to_int8_aligned` (`sk01_gdn_qkvz.py:299-354`). Dos celdas con misma capa/KV pero distinto `quant` tienen pérdida y TC distintos (`super_kernels.md:28-34`: A/B/C). | 3-4 en INT8 track |
| **E4** | `kv_dtype` | `fp8_e4m3` (canon), `bf16`, `fp16`, `auto` | Afecta atención (FA2 dequant in-kernel vs BF16), no GEMM, pero condiciona el epílogo y PPL target (`super_kernels.md:11`). Se combina con `mamba_ssm_dtype=float32` (`fp8_a_int8_ampere.md:505`). | 1 hoy, 2 si se abre BF16 para PPL 6.96 |
| **E5** | `M_bucket` | `1`, `8`, `32`, `128`, `512`, `1664`, `8000` (o rangos `M<=8`, `8<M<=128`, …) | Régimen bandwidth vs compute (`KERNELS-OPTIMIZACION.md:27,50-59` y `super_kernels.md:155-160`). `sk03_fa_qkv.py:81` ya codifica `M_BUCKETS=(1,8,32,128,512,1664,8000)`. SK-01 a `M=1` vs `M=1664` pide launch configs distintas. | 4-7 buckets (no todos los SK necesitan todos) |
| **E6** | `TP` + `shard` | `1`, `2` (canon), `4`, `8` | Cambia `N,K` y el layout `b_col` column-major (`KERNELS-OPTIMIZACION.md:198-199`, `sk02_gdn_out.py:498-502`). Sin TP la clave no es única (`sk05` `17408` vs `34816`). | 1-2 activos (hoy `2`) |
| E7* | `epilog` / `fusion` | `none`, `rmsnorm+quant`, `silu_and_mul`, `allreduce+residual`, `gating+conv` | Derivado de E2+E3 (solo W8A8 activa `silu_and_mul_quant`, `super_kernels.md:208`), pero útil como índice para auditoría. | Derivado, no clave primaria |
| E8* | `MTP_role` | `target`, `draft` | Replica E2 para draft (`sk10_mtp_draft.py:65-77` espejo SK-03/05/06/07). | 2 |

> **No incluir como eje**: `hidden` (5120 fijo), `intermediate` (17408 fijo),
> `vocab` (248320 fijo) — son constantes del checkpoint `orcarouter/...`
> (`super_kernels.md:40-49`). Si mañana se sirve `Qwen3-7B` con `hidden=3584`,
> **sí** se añade `E_hidden` o se versiona la tabla por `model_id`.

### 2.2 Qué NO es un eje (derivados)

* `N,K` — se deriva de `(family, TP)` (`sk01: 16384×5120 → 8192×5120 per-rank`,
  `sk05: 34816×5120 → 17408×5120`, `super_kernels.md:50-86`).
* `BLOCK_M/N/K`, `SHIFT_BLOCK=128`, `num_warps/stages` — son tuning **dentro**
  del `KernelSpec`, no clave de búsqueda (residen en `sk*.py:98-102`).
* `dtype hidden` (`bf16/fp16`) — definido por `--dtype float16` del compose
  (`CONTEXTO-INVESTIGACION.md:5.6`), no por kernel.

---

## 3. Estructura propuesta de la matriz

### 3.1 Principio: tabla plana con clave compuesta + índices secundarios

Dos alternativas evaluadas (ver `super_kernels.md:143-192` "no decide en forward"):

| Opción | Forma | Pros | Contras | Veredicto |
|---|---|---|---|---|
| **A — Dict anidado** `arch → family → kv → quant → M → TP → spec` | `table["sm_86"]["mlp_gateup"]["fp8_e4m3"]["int8_diadic_c"][1664][2]` | Visual, agrupa por arch | Lookup `O(depth)`, difícil serializar/auditar, agujeros `None` profundos | ❌ No recomendado |
| **B — Tabla plana** `dict[tuple, KernelSpec]` | `table[(arch, family, kv, quant, M_bucket, TP)] = spec` | `O(1)`, hashable, filtrable, serializable a JSON/SQLite, `grep` directo | Menos "bonita" a ojo | ✅ **Recomendado** |
| C — Relacional (SQLite) | Tabla `kernels(arch, family, kv, quant, m_bucket, tp, path, cache_key)` | Query `WHERE arch='sm_86'` | Overhead | Útil solo si >1000 filas |

**Recomendación B** — tabla plana con índices secundarios en memoria:

```python
# Concepto (no código productivo) — clave compuesta canónica
KernelKey = tuple[
    str,  # arch:       "sm_86"
    str, # family:     "mlp_gateup" | "gdn_qkvz" | "fa_qkv" | "gdn_out" | ...
    str,  # kv_dtype:   "fp8_e4m3"
    str,  # quant:      "int8_diadic_c" | "int8_per_channel" | "w8a16" | "bf16"
    int,  # M_bucket:   1 | 8 | 32 | 128 | 512 | 1664 | 8000
    int,  # TP:         1 | 2
]

@dataclass(frozen=True)
class KernelSpec:
    key: KernelKey
    # Path del kernel a compilar (celda de la matriz)
    import_path: str          # ej. "vllm._genesis.kernels.sk05_mlp_gateup:mlp_gateup_fused_int8_diadic"
    triton_kernel: str | None # ej. "_sk05_fused_quant_gemm_silu_kernel" si aplica, None si CUTLASS/cuBLAS
    backend: str              # "triton" | "cutlass" | "cublas" | "passthrough"
    # Geometría resuelta (derivada, cacheada para validación)
    N: int; K: int            # por rank
    M_bucket: int
    # Layout de escalas
    scale_dtype: str          # "float32" (cutlass exige float32 en sm80, KERNELS-OPTIMIZACION.md:198)
    shifts: str | None        # "int8 [K/128,N/128]" si Diseño C, else None
    fusion: str               # "RMSNorm+quant+GEMM+SiLU" etc. (super_kernels.md:203-215)
    # Cache
    cache_key: str            # sha256(arch + import_path + triton_src + N,K,M,block,quant)
    # Gates de correctitud
    requires: str             # "T0 d<=3, T2 <1e-3, T5 <2e-2 @64K" (fp8_a_int8_ampere.md:599-654)
```

> La **célula** de la matriz es `import_path`. Todo lo demás es metadato
> para compilar, validar y cachear. Ejemplo real: `sk01_gdn_qkvz.py:335-636`
> expone `sk01_gdn_qkvz_gemm` + fallback ` _sk01_qkvz_fallback_torch`.

### 3.2 Sparse, no densa — solo filas que existen

Espacio cartesiano teórico `4 arch × 17 families × 2 kv × 4 quant × 7 M × 2 TP ≈ 7600`
celdas, pero **solo ~30-40 se materializan** para un `(modelo, arch, KV)` dado
(`super_kernels.md:170`: "11 SK × hasta 4 buckets M ≈ 30-40 kernels").
La tabla es **sparse**: si `(arch,family,kv,quant,M,TP)` no existe en el
checkpoint, no hay entrada → `KeyError` determinista al boot (fail-fast,
`super_kernels.md:186`), no fallback silencioso a cuBLAS.

Índices secundarios para depuración (construidos al cargar la tabla):

```python
by_arch:   dict[arch, list[KernelKey]]      # "sm_86" → todas sus celdas
by_family: dict[family, list[KernelKey]]    # "mlp_gateup" → buckets M
by_M:      dict[M_bucket, list[KernelKey]]  # 1664 → GEMMs compute-bound
```

### 3.3 Dónde vive la tabla

* **Fuente** (repo): `vllm/_genesis/kernels/kernel_registry.py` o
  `workshop/ox_alpha/matriz_kernels_registry.json` versionado en git.
* **Generada**: script offline `tools/gen_kernel_table.py --arch sm_86 --model orcarouter/... --kv fp8_e4m3 --tp 2`
  que lee `config.json` (`num_hidden_layers`, `hidden_size`, `intermediate_size`,
  `full_attention_interval`) y emite la tabla sparse.
* **Runtime** (VRAM): `dict[KernelKey, Callable]` con punteros ya compilados
  (`fn_ptr`), poblado en fase 2 del boot (ver §4).

---

## 4. Ciclo de vida en vLLM al arrancar — precarga y cache

Alineado con `super_kernels.md:143-192` "Estrategia de precarga al arrancar"
(3 fases: escaneo → compilación → dispatch) + `vllm/_genesis/analisis_arranque.py`
patrón PN83 de "analizar todo y cachear".

```
┌──────────────────────────────────────────────────────────────────────┐
│  vLLM boot  (antes de servir el primer token)                        │
│                                                                      │
│  Fase 1 — Escaneo estático  (LLMEngine / ModelRunner init)            │
│  ───────────────────────────────────────────────────────             │
│  ① Detecta arch: torch.cuda.get_device_capability() → (8,6) →       │
│     arch="sm_86" (CONTEXTO-INVESTIGACION.md:4, KERNELS-OPTIMIZACION.md:12) │
│  ② Lee config.json: num_hidden_layers=64, hidden=5120,               │
│     intermediate=17408, full_attention_interval=4, vocab=248320       │
│     + clasifica 48 GDN vs 16 Full (super_kernels.md:18-23)           │
│  ③ Lee CLI/env: --kv-cache-dtype fp8_e4m3, --dtype float16,         │
│     --tensor-parallel-size 2, GENESIS_PN92_KV_DTYPE,                 │
│     GENESIS_ENABLE_SK0*_ENV (sk04_fa_o.py:37-40, sk06_mlp_down.py:45-47, sk11_vision.py:65-70) │
│  ④ Enumera familias: para cada tensor HF lee nombre, forma N×K global│
│     y por rank TP=2, régimen A/B/C y fusión (qwen3_5.py:276-291)     │
│  ⑤ Determina M buckets relevantes: {1,8,1664,8000} (super_kernels.md:155-160,  │
│     KERNELS-OPTIMIZACION.md:27) + chunked-prefill buckets             │
│  ⑥ Producto sparse: genera set[KernelKey] solo para combinaciones que       │
│     existen en el checkpoint (407 tensores FP8 + 792 BF16, super_kernels.md:25)│
│                                                                      │
│  Fase 2 — Compilación y cacheo en memoria (load_model + warmup)      │
│  ───────────────────────────────────────────────────────────────     │
│  Para cada key en set[KernelKey]:                                    │
│    • Calcula cache_key = sha256( arch + triton_src + import_path     │
│      + N,K,M + block128 + quant + scale_dtype )                      │
│    • Probe cache en disco (ver §5):                                  │
│      /root/.cache/vllm/torch_compile_cache/<cache_key>.pt            │
│      /root/.cache/triton/<hash>/  +  /root/.cache/torchinductor/     │
│    • HIT  → carga binario, valida hash, instala fn_ptr               │
│    • MISS → compila:                                                  │
│      - Triton @triton.jit  (sk01: _sk01_gdn_qkvz_kernel,              │
│        sk02: _sk02_gdn_out_int8_kernel, sk05: _sk05_fused_*,         │
│        sk06: _sk06_mlp_down_kernel, sk09: _fused_rmsnorm_quant_kernel)│
│      - CUTLASS cutlass_scaled_mm (ops.cutlass_scaled_mm, exige       │
│        scales float32, KERNELS-OPTIMIZACION.md:198)                  │
│      - cuBLAS BF16 (lm_head stock, sk07_lm_head.py:240-252)          │
│      - Passthrough BF16 (sk08_ssm_control.py, sk09_norm_embed.py,    │
│        sk11_vision.py — sin kernel, solo marker nunca-cuantizar)    │
│    • Materializa pesos reordenados: b_col column-major nn.Parameter  │
│      (patrón PN110 PLAN-CHECKPOINTS.md:215-219, sk02_gdn_out.py:498-502)│
│      + b_scales [N] fp32 + shifts [K/128,N/128] int8 (Diseño C)      │
│    • Memoiza: kernel_table[key] = fn_ptr  (+ CUDAGraph capture si     │
│      aplica, KERNELS-OPTIMIZACION.md end: capture_sizes 4..40)       │
│    • Persiste binario a cache en disco para el próximo boot          │
│                                                                      │
│  Coste Fase 2: ~30-40 kernels × <5 MB código << 24 GiB linears;      │
│  latencia pagada UNA vez al arranque, no en TTFT/TPOT                │
│  (super_kernels.md:170). Falla aquí si T0/T2 gate no pasa            │
│  (fp8_a_int8_ampere.md:599-632, rel_err <1e-3 Diseño A).             │
│                                                                      │
│  Fase 3 — Forward: dispatch directo sin branches ni checks           │
│  ─────────────────────────────────────────────────────────           │
│  En ModelRunner.execute_model / forward por capa (qwen3_5.py:276-449):│
│                                                                      │
│    # tabla resuelta al construir el grafo, no en hot path            │
│    out = kernel_table[(arch, family, kv, quant, M_bucket, TP)](      │
│              hidden, weight_b_col, scales, shifts)  # O(1)            │
│    # ej: SK-05 GateUp M=1664 → cutlass_scaled_mm M=1664 N=17408 K=5120│
│    #     SK-01 GDN_QKVZ M=8   → _sk01_gdn_qkvz_kernel M=8 N=8192 K=5120│
│                                                                      │
│  Sin if tipo=="q_proj", sin if M<16, sin if kv_dtype==fp8 en hot     │
│  path (super_kernels.md:176-189). Si un tensor no cumple grilla       │
│  diádica d≤3 exacta, el modelo falla al arrancar (fail-fast), no     │
│  degrada silencioso.                                                 │
└──────────────────────────────────────────────────────────────────────┘
```

**Invariante**: `kernel_table` es **inmutable** tras Fase 2. Un deploy dado
sirve con una sola tabla; cambiar `arch` o `kv_dtype` invalida la tabla y
exige reboot + recompilación (hash distinto).

---

## 5. Cacheo — dónde y cómo queda cacheado

### 5.1 Capas de cache (3 niveles)

| Nivel | Qué cachea | Path en el compose PROD | Clave |
|---|---|---|---|
| **L1 Triton** | Código PTX/ cubin de cada `tl.dot`/`tl.store` (`_sk01_*`, `_sk05_*`, `_sk06_*`, `_fused_rmsnorm_quant_kernel`) | `/root/.cache/triton` → host `/home/usuario/.cache/triton/qwen38-27b-fp8` (`CONTEXTO-INVESTIGACION.md:5.3`) | `hash(triton_src + arch + BLOCK_M/N/K + SHIFT_BLOCK + dtype)` |
| **L2 TorchInductor / torch.compile** | Grafos `torch.compile` / `CUDAGraph` capturados (prefill vs decode, M bucket distinto → grafo distinto) | `/root/.cache/torchinductor` → host `/home/usuario/.cache/torchinductor/qwen38-27b-fp8` + `/root/.cache/vllm/torch_compile_cache` (`super_kernels.md:172`, `CONTEXTO-INVESTIGACION.md:5.3`) | `sha256(arch + import_path + N,K,M + quant + inductor_version + torch_version)` |
| **L3 Pesos reordenados** | `b_col` column-major + `b_scales` fp32 + `shifts` int8 materializados en VRAM como `nn.Parameter` | VRAM (no disco) tras `process_weights_after_loading` (`fp8.py:385-392`, `sk02_gdn_out.py:498-502`, `sk10_mtp_draft.py:101-125`) | `(tensor_name, arch, quant)` — el repack offline ya está en `weight_scale_inv` del safetensors (`fp8_a_int8_ampere.md:75-77`); opcionalmente cache disco del repack Marlin (`CONTEXTO-INVESTIGACION.md:12.1`) |

Los tres volúmenes son **persistentes** entre boots (bind mounts del compose).
Primer boot tras limpiar caches: ~30-60 s de compilación; siguientes boots:
`HIT` → carga en ~1-2 s + warmup GEMM (`super_kernels.md:171`).

### 5.2 Invalidación

* Cambia `arch` (`sm_86` → `sm_90`) → invalida L1+L2 (PTX distinto).
* Cambia `quant` (`int8_diadic_c` → `w8a16`) o `BLOCK_SIZE` → invalida L1+L2
  (triton_src distinto).
* Cambia `torch` / `triton` / `vllm` pin (`v0.23.0` → `v0.24.0`) → invalida L2
  (Inductor breaking change).
* Cambia `M_bucket` threshold (`1664` → `2048`) → solo compila buckets nuevos;
  `M=40` decode puede colapsar al bucket INT8 más cercano sin recompilar
  (`super_kernels.md:160`).

---

## 6. Ejemplo — matriz para RTX 3090 (`sm_86`) · FP8 `→` INT8 diádico · KV `fp8_e4m3` · TP=2

Fijamos `arch=sm_86`, `kv=fp8_e4m3`, `TP=2`, `quant=int8_diadic_c` donde
aplica (fallback `w8a16`/`bf16` donde no). Es el **slice** de la N-tabla que
corre hoy en `2×3090` PROD.

### 6.1 Tabla plana (vista por `family` × `M_bucket` — 17 familias × 4 buckets = 34 filas efectivas)

> `M_bucket` = tokens batcheados que ve el GEMM. Valores medidos en
> `KERNELS-OPTIMIZACION.md:27` (`M=40` decode MTP, `M=1664` prefill chunk) +
> `super_kernels.md:155-160` (`1,8,1664,8000`) + `sk03_fa_qkv.py:81` buckets extendidos.

| # | `family` (SK) | Tensores HF cubiertos | Geometría por rank `N×K` | `M_bucket` | `quant` efectivo | Kernel `import_path` (celda) | Backend | Fusión inline | Notas |
|---|---|---|---|---|---|---|---|
| **SK-01** | `gdn_qkvz` | `linear_attn.in_proj_qkv` [10240,5120] + `in_proj_z` [6144,5120] → `in_proj_qkvz` [16384,5120] (`super_kernels.md:51-53`) | `8192×5120` | 1 | `int8_diadic_c` | `vllm._genesis.kernels.sk01_gdn_qkvz:sk01_gdn_qkvz_gemm` (`_sk01_gdn_qkvz_kernel`) | Triton `mma.m16n8k32.s8.s32` + epílogo `s_row*shift` | `RMSNorm(input_layernorm)+pow2+quant→GEMM→split Q/K/V` | 48 capas idx `0,1,2,4,5,6…` (`fp8_a_int8_ampere.md:46`); `shifts [40,64]` int8; prefill 2.9× vs Marlin (`super_kernels.md:230`) |
|  |  |  |  | 8 | `int8_diadic_c` | idem | Triton | idem | Decode spec K=3 → `M=8` (`rejection_sampler.py:119-197`) |
|  |  |  |  | 1664 | `int8_diadic_c` | idem | Triton/CUTLASS | idem | Prefill chunk (`max_num_batched_tokens=1664`) |
|  |  |  |  | 8000 | `int8_diadic_c` | idem | CUTLASS `cutlass_scaled_mm` | idem | Prefill largo / 262K NIAH |
| **SK-02** | `gdn_out` | `linear_attn.out_proj` [5120,6144] (`super_kernels.md:54`) | `5120×3072` | 1/8 | `int8_diadic_c` (o `w8a16` si T5>2e-2@64K, `sk02_gdn_out.py:237-243`) | `vllm._genesis.kernels.sk02_gdn_out:sk02_gdn_out_proj` (`_sk02_gdn_out_int8_kernel`) | Triton (shift INT32 headroom 10, `sk02:83-84`) | `GEMM→AllReduce(RowParallel)→residual` | 48 GDN, **sin norm delante** → `C` no `A` (`fp8_a_int8_ampere.md:453`) |
|  |  |  |  | 1664 | `int8_diadic_c` | idem | Triton | idem |  |
| **SK-03** | `fa_qkv` | `q_proj` 12288 + `k_proj` 1024 + `v_proj` 1024 → `qkv_proj` 14336×5120 (`super_kernels.md:67-72`) | `7168×5120` | 1/8/1664/8000 | `int8_diadic_c` + SmoothQuant `2^k` en `input_layernorm` (`fp8_a_int8_ampere.md:5.1`) | `vllm._genesis.kernels.sk03_fa_qkv:sk03_fa_qkv_forward` (`_sk03_fused_rmsnorm_quant_gemm_kernel`) | Triton monolito 1 launch (auditoría `sk03_fa_qkv.py:2-15`) | `RMSNorm+quant→GEMM→split Q\|gate/K/V→q/k_norm BF16` | 16 Full (+1 MTP) `idx 3,7,11…63` (`sk03:78`); `M_BUCKETS=(1,8,32,128,512,1664,8000)` (`sk03:81`) |
| **SK-04** | `fa_o` | `self_attn.o_proj` [5120,6144] (`super_kernels.md:72`) | `5120×3072` | 1/8/1664 | `int8_diadic_c` (o `w8a16` keep, 13% cómputo capa si se deja, `fp8_a_int8_ampere.md:460`) | `vllm._genesis.kernels.sk04_fa_o:fa_o_forward` (`_sk04_fa_o_kernel` copia SK-02) | Triton / CUTLASS (hoist `_CUTLASS_OK/_HYBRID_OK`, `sk04:57-71`) | `GEMM→AllReduce→residual` | 17× (16+MTP), abliterado (`fp8_a_int8_ampere.md:144-158`) |
| **SK-05** | `mlp_gateup` | `gate_proj`+`up_proj` → `gate_up_proj` 34816×5120 (`super_kernels.md:78-84`) | `17408×5120` | 1 | `int8_diadic_c` + `scale_pow2` en `post_attention_layernorm` (`sk05:62-70`) | `vllm._genesis.kernels.sk05_mlp_gateup:mlp_gateup_fused_int8_diadic` (`_sk05_fused_quant_gemm_silu_kernel`) | Triton 1 launch (vs 3, `sk05:2-10`) + CUTLASS fallthrough | `RMSNorm(post)+quant→GEMM GateUp→silu_and_mul_quant` | 64 (+1 MTP) — **mayor GEMM prefill**: 5.53→1.61 ms 3.4× (`super_kernels.md:268-269`) |
|  |  |  |  | 8/1664/8000 | `int8_diadic_c` | idem | Triton/CUTLASS | idem |  |
| **SK-06** | `mlp_down` | `mlp.down_proj` [5120,17408] (`super_kernels.md:82`) | `5120×8704` | 1/8/1664/8000 | `int8_diadic_c` + columna `diag(s)` (`fp8_a_int8_ampere.md:452`) o Ozaki 2-slice (`sk06_mlp_down.py:210-259`) si SQNR flojo | `vllm._genesis.kernels.sk06_mlp_down:mlp_down_int8_scaled_residual` (`_sk06_mlp_down_kernel` `HAS_RESIDUAL/SHIFT_ENABLED`, `sk06:103-165`) | Triton unificado 1 kernel / Ozaki `sk06_ozaki.py` | `GEMM→AllReduce→residual` + col scale | 65× (64+MTP), concentra outliers residual (`fp8_a_int8_ampere.md:619`); `T0 d>8` frecuente → Diseño A |
| **SK-07** | `lm_head` | `lm_head` [248320,5120] vocab-parallel (`super_kernels.md:90-92`) | `124160×5120` | 1/8 | `bf16` (cuBLAS) / `fp8_weight_fp16_compute` (PN77, `sk07:32-33`, `-1.2 GiB/GPU`) / `w8a16_marlin` (~1 ms/paso, `KERNELS-OPTIMIZACION.md:145`) | `vllm._genesis.kernels.sk07_lm_head:lm_head_forward` / `lm_head_fused_sampled` (`_sk07_fused_sampled_kernel` `HAS_INT8`, `sk07:158-231`) | cuBLAS BF16 / CUTLASS INT8 / Triton sampled | `vocab_parallel_gather→logits_processor` (`logits_processor.py:75-104`) | Decode bandwidth-bound, prefill 33→14 ms 2.8× con INT8 (`KERNELS-OPTIMIZACION.md:58`) |
|  |  |  |  | 1664 | `bf16` / `int8` (threshold `GENESIS_PNXX_W8A8_MIN_TOKENS`, `super_kernels.md:290`) | idem | idem |  |  |
| **SK-08** | `ssm_control` | `in_proj_a/b` 48×5120, `conv1d` [10240,1,4] 3D, `A_log/dt_bias` [48] (`super_kernels.md:55-60`) | micro 48×5120 | 1/8 | `bf16` **nunca cuantizar** (`sk08_ssm_control.py:13`, `fp8_a_int8_ampere.md:90-94,529-534`) | `vllm._genesis.kernels.sk08_ssm_control:sk08_ssm_control_bf16_fused` (`_sk08_fused_decode_packed_kernel` WIDTH=4 HV=48, `sk08:81-168`) | Triton `num_warps=1,num_stages=3` | `fused_sigmoid_gating_delta_rule_update` + `causal_conv1d_update` | 48 GDN; estado `mamba_ssm_dtype=float32` persistente |
| **SK-09** | `norm_embed` | `embed_tokens` 248320×5120 + 8 normas + ViT rms (`sk09_norm_embed.py:86-95`) | — | 1/8/1664 | `bf16` + wrapper `s=2^k` exacto (`sk09:36-41`, suma `k<<7` a exponente BF16, `fp8_a_int8_ampere.md:301-303`) | `vllm._genesis.kernels.sk09_norm_embed:rmsnorm_quant_fused` (`_fused_rmsnorm_quant_kernel`, `sk09:457-530`) / `embed_tokens_bf16_passthrough` | Triton 1 pass (sin materializar `y_bf16`) / `native` | `RMSNorm bf16 --pow2--> quant_per_token INT8` | Habilitador de SK-01/03/05; sin él no hay `silu_and_mul_quant` (`KERNELS-OPTIMIZACION.md:151`) |
| **SK-10** | `mtp_draft` | Draft MTP: `qkv_proj+gate_up+down+lm_head` draft + `fc` hidden*2→hidden (`qwen3_5_mtp.py:59-159`) | espejo SK-03/05/06/07 por rank | 1/8 | `int8_diadic_c` mirror + PN108 draft `lm_head` FP8 (`sk10_mtp_draft.py:187-215`) | `vllm._genesis.kernels.sk10_mtp_draft:mtp_draft_fused_gemm` (re-export `sk05_mlp_gateup.mlp_gateup_fused_int8_diadic`, `sk10:66-77`) / `mtp_draft_linear` | CUTLASS 1 camino + fallback torch (pytest) | `embed draft→fc→decoder→lm_head draft→verify` | PN110 swap `b_col` column-major 1:1 (`super_kernels.md:313`, 5.56 vs 7.21s wall) |
| **SK-11** | `vision` | `visual.*` 333 tens. ViT (`sk11_vision.py:73-84`, `fp8_a_int8_ampere.md:84`) | — | — (1 imagen) | `bf16` passthrough (`sk11_vision.py:154-186`) | `vllm._genesis.kernels.sk11_vision:passthrough_bf16` (stub documentado, `is_available→True` siempre) | cuBLAS BF16 / FlashAttention ViT | Vision tower intacto, no fusionado con LLM GEMM | Nunca cruza LLM GEMM; `is_visual_tensor→True` (`sk11:102-117`) |

> `M_bucket` colapsa: si solo existen `M=1` y `M=1664` para `gdn_out`, la tabla
> registra 2 filas, no 4. `M=8` y `M=8000` restantes se mapean al bucket más
> cercano sin recompilar (`super_kernels.md:160`: `M=40` → bucket INT8 cercano).

### 6.2 Dimensión `arch` — por qué no basta con una tabla de `sm_86`

Ejemplo de diversificación futura (misma tabla plana, otra fila `arch`):

| `family` | `sm_86` (3090) canónico | `sm_89` (4090 Ada) | `sm_90` (H100 Hopper) |
|---|---|---|---|
| `mlp_gateup` | `sk05_mlp_gateup:mlp_gateup_fused_int8_diadic` Triton `m16n8k32.s8` + `cutlass_scaled_mm` (INT8) | `cutlass_fp8` o `triton_fp8` nativo (TC FP8 `mma` existe) → `w8a8_fp8` sin diádico | `DeepGEMM` / `FlashInferFp8DeepGEMM` blockScaled (`CONTEXTO-INVESTIGACION.md:8.1`, `_POSSIBLE_FP8_BLOCK_KERNELS` 1-2) |
| `lm_head` | `sk07:cublas_bf16` o `cutlass_int8` | `cublas_fp8` viable | `cutlass_fp8` vocab-parallel FP8 nativo |
| `ssm_control` | `sk08: _sk08_fused_decode_packed_kernel` (WIDTH=4 hardcode) | idem, pero `num_warps`/`BLOCK_D` puede retunear para AD102 | `sm_90` permite `TLX` fused más agresivo |

La tabla lo soporta sin rediseño: se añade `arch="sm_90"` con **mismo**
`family/kv/M/TP` pero distinto `import_path`/`backend`.

### 6.3 Vista JSON de 2 filas (cómo se serializa)

```json
{
  "table_version": "2026-08-25",
  "model": "orcarouter/Qwen3.8-27B-Uncensored-FP8",
  "arch": "sm_86",
  "kv_dtype": "fp8_e4m3",
  "TP": 2,
  "entries": [
    {
      "key": ["sm_86", "mlp_gateup", "fp8_e4m3", "int8_diadic_c", 1664, 2],
      "import_path": "vllm._genesis.kernels.sk05_mlp_gateup:mlp_gateup_fused_int8_diadic",
      "triton_kernel": "_sk05_fused_quant_gemm_silu_kernel",
      "backend": "triton",
      "geometry": {"N": 17408, "K": 5120, "M_bucket": 1664},
      "scale_dtype": "float32",
      "shifts": "int8 [40,136]",
      "fusion": "RMSNorm(post)+quant -> GEMM GateUp -> SiLU*Mul",
      "cache_key": "sha256:9f2c...a1",
      "requires": "T0 p99 d 4-6, T2 <1e-3, s=2^k en post_attention_layernorm"
    },
    {
      "key": ["sm_86", "gdn_out", "fp8_e4m3", "int8_diadic_c", 1, 2],
      "import_path": "vllm._genesis.kernels.sk02_gdn_out:sk02_gdn_out_proj",
      "triton_kernel": "_sk02_gdn_out_int8_kernel",
      "backend": "triton",
      "geometry": {"N": 5120, "K": 3072, "M_bucket": 1},
      "scale_dtype": "float32",
      "shifts": "int8 [24,40]",
      "fusion": "GEMM -> AllReduce(RowParallel) -> residual add",
      "cache_key": "sha256:4b7e...c3",
      "requires": "T5 <2e-2 @64K else fallback w8a16"
    }
  ]
}
```

---

## 7. Qué cambia si mañana aparece un `arch` nuevo, un `kv` nuevo o un `M` nuevo

* **Nuevo `arch` (`sm_90`)**: se añade la dimensión `arch` sin tocar `family`/`kv`.
  El `dispatcher.py:should_apply` ya gatea por `is_sm_at_least` (`CONTEXTO-INVESTIGACION.md:2.1`);
  la tabla añade filas `("sm_90", family, …)` con `import_path` hacia `DeepGEMM`/`CUTLASS FP8`.
  `sm_86` queda intacto — cero regresión.
* **Nuevo `kv_dtype` (`bf16`)**: se añade valor `kv="bf16"` en E4. Las filas
  `fa_qkv`/`gdn_qkvz` no cambian (GEMM W8A8 es ortogonal a KV), solo cambia el
  epílogo de atención y la métrica PPL target (`T7: 6.96`, `fp8_a_int8_ampere.md:169`).
* **Nuevo `M` (`M=4096` para batch 64)**: se añade bucket `4096` en E5.
  SKs compute-bound (`gate_up`, `down`) ganan sin recompilar los viejos; decode
  `M=1` intacto.
* **Nuevo `quant` (`w4a16` Marlin)**: si VRAM es cuello (W8A8 no ahorra bytes,
  `fp8_a_int8_ampere.md:708-710`), se añade `quant="w4a16"` solo para
  `family=mlp_gateup/mlp_down` y se reusa `Marlin W4A16` (`fp8_a_int8_ampere.md:710`).

La matriz N-dimensional **absorbe** la evolución sin reescribir la tabla:
se añaden filas, no columnas. Una 2D habría requerido una nueva tabla por
cada arch/kv/quant.

---

## 8. Decisiones de implementación (sin código)

1. **Tabla plana `dict[tuple→KernelSpec]`** (ver §3.1 B) en `vllm/_genesis/kernels/kernel_registry.py`.
2. **Generación offline** (`tools/gen_kernel_table.py`) que lee `config.json` +
   `quantization_config` y emite `matriz_kernels_registry.json` (hash de
   `weight_scale_inv` shapes para detectar checkpoint rotado).
3. **Resolución al boot**: `ModelRunner.__init__` llama `resolve_kernel_table(arch,kv,tp)`
   → `set[KernelKey]` sparse → `warmup_sk03_kernels` / `warmup_sk05` etc.
   (`sk03_fa_qkv.py:495-520` patrón warmup por `M`).
4. **Fail-fast**: si `T0` muestra `d>10` frecuente → esa familia se excluye de
   `int8_diadic_c` y se registra como `w8a16`/`bf16` (no hay fallback silencioso
   en runtime, `fp8_a_int8_ampere.md:599-632`).
5. **Cache en disco** (§5) con `cache_key = sha256(arch + triton_src + N,K,M + quant + torch.__version__ + triton.__version__)`
   y persistencia en los 3 mounts del compose (`/root/.cache/{vllm,triton,torchinductor}`).
6. **Dispatch**: `kernel_table: dict[KernelKey, Callable]` inyectado en el grafo
   `ModelRunner.execute_model` (indexado por `sk_id` + `M_bucket`, `super_kernels.md:178-182`);
   prefill y decode pueden tener grafos distintos por `M` pero **ambos ya compilados**
   al inicio (`super_kernels.md:187`).
7. **Auditoría**: `genesis doctor` lista `by_arch`/`by_family`/`by_M` + `cache_key`
   hit/miss + T2/T5 gates, reutilizando `analisis_arranque.py` patrón de informe
   humano (`vllm/_genesis/analisis_arranque.py:1-30`).

---

## 9. Mapa de archivos

| Ref | Ubicación |
|---|---|
| Catálogo 11 SK + matriz `tipo×M` + 3 fases de precarga | `workshop/ox_alpha/super_kernels.md:143-192,203-215` |
| Teoría FP8→INT8 diádica, Diseño A/B/C, T0-T13, `fp8_e4m3_to_int8_aligned` | `workshop/ox_alpha/fp8_a_int8_ampere.md:202-365,374-424,599-632` |
| Inventario GEMM por rank + shootout INT8 2.8-3.4× | `workshop/ox_alpha/KERNELS-OPTIMIZACION.md:16-32,50-59` |
| SK paths (celdas de la matriz) | `vllm/_genesis/kernels/sk01_gdn_qkvz.py:357-638`, `sk02_gdn_out.py:584-668`, `sk03_fa_qkv.py:495-520`, `sk04_fa_o.py:255-349`, `sk05_mlp_gateup.py:280-350`, `sk06_mlp_down.py:261-340`, `sk07_lm_head.py:340-417`, `sk08_ssm_control.py:261-290`, `sk09_norm_embed.py:575-637,704-737`, `sk10_mtp_draft.py:127-185`, `sk11_vision.py:140-194` |
| Dispatcher / should_apply / arch gate | `vllm/_genesis/dispatcher.py:2636` (`CONTEXTO-INVESTIGACION.md:2.1`) |
| `qwen3_5.py` fusión `in_proj_qkvz/gate_up_proj/in_proj_ba` | `assets/vllm/vllm/model_executor/models/qwen3_5.py:276-291,440-450` |
| Montajes cache persistentes | `compose/docker-compose.qwen38-27b-fp8.yml:248-252`, `CONTEXTO-INVESTIGACION.md:5.3` |
| Patrón informe humano de arranque | `vllm/_genesis/analisis_arranque.py:1-30,566-593` |
| Este documento | `workshop/ox_alpha/matriz_kernels.md` |

---

*Slice `sm_86` generado sobre `super_kernels.md` (435 lín.) + `fp8_a_int8_ampere.md` (804 lín.) + 11× `sk*.py` (~5500 lín.) + benches `KERNELS-OPTIMIZACION.md:50-59` (2×3090). Toda celda apunta a un `import_path` verificable y a un cache en `/root/.cache/vllm/torch_compile_cache`.*
