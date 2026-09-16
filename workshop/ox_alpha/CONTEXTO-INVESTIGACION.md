# 📌 Contexto de Investigación — genesis-vllm-patches

> **Archivo base de investigación.** Consolida todo el análisis realizado hasta la
> fecha sobre este repositorio, su contenedor de producción y los kernels CUDA que
> ejecuta. Es el documento de referencia para cualquier trabajo futuro sobre esta
> base: no hay que re-descubrir nada de lo que está acá.
>
> | | |
> |---|---|
> | **Fecha** | 2026-08-24 |
> | **Autor del análisis** | ox-alpha (agente de investigación, solo lectura) |
> | **Método** | Exploración estática de `assets/vllm` (vLLM v0.23.0) + lectura del compose PROD + inspección read-only del contenedor vivo. **No se modificó código ni se operó ningún contenedor.** |
> | **Estado** | Investigación abierta — ver §12 (oportunidades) y §13 (pendientes) |

---

## Índice

1. [Resumen ejecutivo](#1-resumen-ejecutivo)
2. [El proyecto Genesis](#2-el-proyecto-genesis)
3. [`assets/vllm` — árbol de referencia v0.23.0](#3-assetsvllm--árbol-de-referencia-v0230)
4. [Hardware: 2× RTX 3090 con P2P + ReBAR](#4-hardware-2-rtx-3090-con-p2p-rebar)
5. [El contenedor PROD (`compose/docker-compose.qwen38-27b-fp8.yml`)](#5-el-contenedor-prod)
6. [Estado del contenedor al momento del análisis](#6-estado-del-contenedor-al-momento-del-análisis)
7. [Mapa de kernels CUDA en ejecución](#7-mapa-de-kernels-cuda-en-ejecución)
8. [Camino FP8 detallado (GEMM en Ampere)](#8-camino-fp8-detallado-gemm-en-ampere)
9. [Camino atención + GDN detallado](#9-camino-atención--gdn-detallado)
10. [Comunicación TP=2 y el bug del custom all-reduce](#10-comunicación-tp2-y-el-bug-del-custom-all-reduce)
11. [Hallazgos que validan o corrigen el compose](#11-hallazgos-que-validan-o-corrijen-el-compose)
12. [Oportunidades de optimización (rankeadas)](#12-oportunidades-de-optimización-rankeadas)
13. [Inconsistencias detectadas y pendientes](#13-inconsistencias-detectadas-y-pendientes)
14. [Apéndice: referencias archivo:línea clave](#14-apéndice-referencias-archivolínea-clave)

---

## 1. Resumen ejecutivo

**Genesis** es un *drop-in patcher* para vLLM: fija un pin de vLLM y le aplica
~120 cambios quirúrgicos (text-edits con anclas, class-rebinds, middleware)
que convierten vLLM stock en un servidor de inferencia Qwen de grado
producción sobre hardware NVIDIA de consumo. No es un fork ni un cuantizador.

Este workspace tiene tres piezas:

| Pieza | Qué es | Estado |
|:---|:---|:---|
| Repo raíz | El patcher Genesis (dispatcher + wiring + tests + CLI + plugin) | Genesis local va **adelante** del README público (tiene parches PN88–PN100+ que el ledger v7.72 no documenta) |
| `assets/vllm` | Checkout completo de **vLLM v0.23.0**, commit `0fc695f` ("Cap fastapi < 0.137...") | **Limpio**: 0 marcadores Genesis aplicados. Es el espejo exacto de lo que corre en el contenedor |
| Contenedor PROD | `genesis-27b-qwen38-fp8` — imagen `vllm/vllm-openai:v0.23.0` | Corriendo (verificado Up ~1h al analizar). Sirve Qwen3.8-27B-Uncensored-FP8 en TP=2 |

**Conclusiones principales del análisis de kernels** (detalle en §7–§12):

1. En sm_86 (Ampere, sin tensor cores FP8), todos los Linear FP8-bloque corren
   por **Marlin W8A16** con dequant fusionado en-kernel — ya es el óptimo
   disponible; no existe camino W8A8 en este hardware.
2. La atención full usa **FlashInfer FA2 nativo** con KV fp8_e4m3 dequantizado
   in-kernel; el GDN corre por **kernels Triton FLA** autotuneados.
3. Tres flags del compose son **inertes** para este modelo denso:
   `--moe-backend flashinfer_trtllm`, `VLLM_USE_FUSED_MOE_GROUPED_TOPK`,
   `VLLM_MARLIN_USE_ATOMIC_ADD` (este último casi seguro), y
   `--enable-flashinfer-autotune` es **no-op en sm_86**.
4. El bug documentado del custom all-reduce tiene ahora una **hipótesis de causa
   raíz verificada mecánicamente**: `cudaIpcGetMemHandle` falla con
   `invalid argument` cuando el buffer de entrada de un AR capturado vive en
   memoria VMM del `cumem allocator` (que OffloadingConnector exige).
5. Las dos ideas de parche Genesis con mejor relación impacto/esfuerzo:
   **cache en disco del repack Marlin** (boot) y **expansión de `BKV_LIST`
   para sm_86 en `chunk_o`** (throughput prefill GDN).

---

## 2. El proyecto Genesis

### 2.1 Arquitectura general

Genesis engancha a vLLM en cuatro niveles:

1. **Dispatcher** — `vllm/_genesis/dispatcher.py` (3247 líneas).
   - `PATCH_REGISTRY: dict[str, dict]` (línea 65): single source of truth.
     Por parche: `title`, `env_flag`, `default_on`, `deprecated`, `category`,
     `credit`, `upstream_pr`, `conflicts_with`, `deprecation_note`.
   - `should_apply(patch_id) -> tuple[bool, str]` (línea 2636): gate unificado.
     Orden de decisión:
     1. Env flag truthy + parche opt-in → **aplica** (override del operador;
        `applies_to` solo se loguea como warning).
     2. Env flag ausente + `default_on=False` → skip ("opt-in only").
     3. `default_on=True` + mismatch de `applies_to` → hard-skip "MODEL-COMPAT".
     4. Si no, consulta `config_detect.recommend(patch_id)`:
        `"apply"/"neutral"` aplica; `"skip:"/"redundant:"/"deprecated:"` no.
   - `log_decision()` acumula en `_DECISIONS` para la matriz de arranque.
   - Ejecutable standalone: `python3 -m vllm._genesis.dispatcher` (tabla ASCII).

2. **Wiring** — `vllm/_genesis/wiring/`: 11 subcategorías (`compile_safety/`,
   `hybrid/`, `kernels/`, `kv_cache/`, `legacy/`, `loader/`, `memory/`,
   `middleware/`, `models/`, `perf_hotfix/`, `spec_decode/`,
   `structured_output/`). Cada parche es un módulo con `apply() -> (status, reason)`.

3. **Tres mecanismos de aplicación**:
   - **Text-patch** (`wiring/text_patch.py`, 529 líneas): reemplazo por ancla
     exacta (`TextPatch.anchor` → `replacement`), idempotente vía marker único
     escrito en el archivo (ej. `"Genesis P4 TQ hybrid v7.0"`). NUNCA lanza:
     enum `TextPatchResult = APPLIED | IDEMPOTENT | SKIPPED | FAILED`. Los
     `upstream_drift_markers` permiten jubilar un parche limpiamente cuando
     upstream absorbe el fix.
   - **Class/attribute-rebind** (`wiring/rebind.py`): `AttributeRebinder`
     guarda el original, verifica existencia e idempotencia, registra en
     `WiringRegistry`, con `revert()`.
   - **Middleware install**: FastAPI/Starlette sobre `build_app()` con marker
     en el app (ej. PN65 access log, `__pn65_installed__`).

4. **GPU profile** — `gpu_profile.py`: datasheet de ~28 placas
   (Ampere→Blackwell, consumo+datacenter) + predicados de recomendación.

### 2.2 Orquestador: `patches/apply_all.py` (5766 líneas)

- Entrada: `run(verbose=True, apply=False) -> PatchStats`; `main()` CLI
  standalone con exit codes 0/1/2. `apply=True` lo pasa el plugin o el
  entrypoint del contenedor.
- `PATCH_REGISTRY: list[(nombre, callable)]` (~100 entradas) con import tardío
  del wiring correspondiente.
- Guards de 5 capas antes de aplicar: (1) archivo existe (`resolve_vllm_file`),
  (2) marker de idempotencia, (3) upstream ya lo mergeó (`upstream_compat.py`),
  (4) vendor/chip (`is_nvidia_cuda`, `is_sm_at_least`), (5) modelo/backend.
- Antes del loop: **pin-gate** `assert_vllm_pin_allowed` contra
  `KNOWN_GOOD_VLLM_PINS` (exit 2 en modo strict).
- Salida: `PatchResult(name, status ∈ applied|skipped|failed, reason)`
  acumulado en `PatchStats` con detección de skips silenciosos
  (`partial_apply_warnings` vs lista `BENIGN`).

### 2.3 Plugin entry-point (`tools/genesis_vllm_plugin/`)

- Paquete `genesis-vllm-plugin` 7.0.0.dev0 sin dependencias runtime.
- Entry point `[project.entry-points."vllm.general_plugins"] genesis_v7 =
  "genesis_v7:register"` → vLLM lo invoca al inicio de **cada proceso**
  (main, engine-core, workers).
- `register()` (idempotente, non-fatal, rápido, sin imports de vllm a nivel
  módulo): (1) opt-out `GENESIS_DISABLE=1`; (2) P93: si
  `GENESIS_FORCE_MARLIN_W8A16=1` añade `AllSparkLinearKernel` a
  `VLLM_DISABLED_KERNELS` antes de que el engine lo lea; (3) ejecuta
  `apply_all.run(apply=True)` (desactivable con `GENESIS_WIRING_APPLY=0`).
- Script de consola: `genesis = vllm._genesis.compat.cli:main`.

### 2.4 Mecanismo de plugins de vLLM 0.23 (el lado receptor)

- Grupo de entry points `vllm.general_plugins` (otros: `vllm.platform_plugins`,
  `vllm.io_processor_plugins`, `vllm.stat_logger_plugins`).
- Carga: `load_plugins_by_group(group)` en `vllm/plugins/__init__.py:28`
  (filtra por env `VLLM_PLUGINS`; si es None carga todos);
  `load_general_plugins()` línea 69 con guard `plugins_loaded` por proceso.
- Invocado desde: `engine/arg_utils.py:745,2585`, `v1/engine/core.py:107`,
  `v1/worker/worker_base.py:245`, `model_executor/models/registry.py:1409`.
- Docs del mecanismo: `assets/vllm/docs/design/plugin_system.md`.

### 2.5 CLI `genesis` (18 subcomandos)

`doctor`, `explain`, `init`, `list-models`, `pull`, `lifecycle-audit`,
`validate-schema`, `categories`, `migrate`, `recipe`, `preset`, `plugins`,
`telemetry`, `update-channel`, `self-test`, `verify (--quick/--boot/--full)`,
`preflight`, `bench`. Dispatcher `_SUBCOMMAND_MAP` en `compat/cli.py:39`.

### 2.6 Tests

~127 archivos `test_*.py` en `vllm/_genesis/tests/` (+ conftest,
numerical_regression_helpers). Cubren por-parche (PN12…PN110), wiring
(text_patch, rebind, transacciones multi-archivo), dispatcher/validadores,
guards, regresión numérica y gates de sincronía
(`test_apply_all_dispatcher_sync`, `test_patches_md_sync`).

---

## 3. `assets/vllm` — árbol de referencia v0.23.0

| Dato | Valor |
|:---|:---|
| Tag git | `v0.23.0` |
| HEAD | `0fc695f` — "[Bugfix][Frontend] Cap fastapi < 0.137 to avoid prometheus-fastapi-instrumentator crash on serve startup" |
| Versión Python | Dinámica vía setuptools-scm (`pyproject.toml:36,50`); NO existe `vllm/_version.py` (se genera en build); `version.py` cae a `"dev"` sin build |
| Marcadores Genesis | **CERO** coincidencias de `_genesis`/`PN59`/`genesis_v7` en `assets/vllm/vllm/**/*.py` → árbol stock limpio |
| Uso previsto | Referencia para verificar anchors de text-patches contra el runtime real del contenedor (misma versión exacta) |

> ⚠️ Nota de coherencia: el README público del repo documenta pin
> `0.20.2rc1.dev9+g01d4d1ad3` y Genesis v7.72. La realidad local (compose +
> assets) está en **v0.23.0** con parches más nuevos (PN88–PN100+). El README
> quedó atrás; el header del compose lo confirma como desviación deliberada
> ("subir de versión mueve los anchors de todos los parches Genesis").

---

## 4. Hardware: 2× RTX 3090 con P2P + ReBAR

| Dato | Valor |
|:---|:---|
| GPUs | 2× NVIDIA RTX 3090 24 GB (GA102, **sm_86**, Ampere) |
| Limitación clave | **Sin tensor cores FP8/FP4 nativos** (eso empieza en Ada sm_89 / Hopper sm_90). Los pesos FP8 se procesan emulados a 16-bit dentro del kernel |
| Interconexión | PCIe Gen4 x8, topología **PIX** (un solo puente PCIe). SIN NVLink |
| P2P | ✅ Habilitado a nivel driver (**driver 610.57.04**): `nvidia-smi topo -p2p rw` OK en ambos sentidos |
| ReBAR | ✅ Al máximo: **BAR1 = 32 GB por placa** (> VRAM de 24 GB → toda la VRAM es CPU-mapeable) |
| Host | 30 GB RAM (~14 en uso), `/dev/shm` 16 GB — techo duro para el tier L2 RAM del KV offload |
| Historial | Antes del driver nuevo la topología era CNS (sin P2P); el 31,8% de GPU en all-reduce medido el 2026-08-17 corresponde a esa era → **cifra obsoleta, re-perfilar** |

Implicancias directas de P2P+ReBAR para este análisis:

- NCCL ya usa P2P directo (evita staging por host) → el all-reduce PYNCCL
  actual mejora respecto de la era CNS, pero NCCL elige algoritmo solo
  (típicamente Ring Simple/LL128 para los tamaños de este modelo).
- El custom all-reduce de vLLM **sigue fallando igual** con P2P activo
  (re-probado 2026-08-23 con PN94 aplicado) → el problema NO es topología.
  Ver §10 para la hipótesis de causa raíz.
- ReBAR no cambia el camino del KV offload (usa cudaMemcpy pinned sobre PCIe),
  pero habilita mapeos peer completos que NCCL/symm-mem sí aprovechan.

---

## 5. El contenedor PROD

Fuente: `compose/docker-compose.qwen38-27b-fp8.yml` (730 líneas, densamente
documentado con decisiones fechadas). Servicio `vllm-server`.

### 5.1 Identidad y ciclo de vida

| Aspecto | Valor |
|:---|:---|
| Container name | `genesis-27b-qwen38-fp8` |
| Imagen | `vllm/vllm-openai:v0.23.0` (pin fijo, coincide con `assets/vllm`) |
| Puerto | `8320:8320` (API OpenAI) |
| Red | externa `lanbridge`, IP estática `172.20.0.228` / `fd00:172:20::13` |
| Restart policy | **`no` a propósito** (2026-08-14): un OOM-crash debe quedar caído y visible, no relevantarse indistinguible. Detección de reinicio de proceso: `process_start_time_seconds` en `/metrics` |
| Healthcheck | No definido (se vigila desde afuera) |
| shm/ipc | `shm_size: 16gb`, `ipc: host` |
| GPU | `count: all` (las dos 3090) |

### 5.2 Modelo

| Aspecto | Valor |
|:---|:---|
| Checkpoint | `orcarouter/Qwen3.8-27B-Uncensored-FP8` (HF, montado desde `/home/usuario/Proyectos/models-cache`) |
| served-model-name | `qwen3.8` |
| Cuantización | **FP8 por bloques E4M3 128×128 sobre 407 de 1606 tensores** (`quantization="fp8"` sale del config.json — NO pasar `--quantization fp8`). Quedan BF16: vision tower, norms, mlp.gate, lm_head, embed_tokens y TODO linear_attn (48 capas GDN) |
| Peso | ~31 GB total → ~15,5 GiB por GPU con TP=2 |
| Calidad | Model card: MMLU 84,3→84,7 y GSM8K 90,0→88,7 vs base FP8; 99,9% códigos FP8 idénticos a oficiales Qwen |
| Arquitectura | Híbrido: ~16 capas atención full + **48 capas GDN** (GatedDeltaNet, atención lineal) + vision tower + MTP head abliterada (spec-decode sigue sirviendo). Resuelve a `Qwen3_5ForConditionalGeneration` (multimodal, denso) |
| Desviaciones de la model card | (1) Card pide vLLM v0.24.0; se corre v0.23.0 deliberadamente (mover versión mueve todos los anchors). (2) Resuelta: método spec-decode `"mtp"` (nombre correcto, evita warning de deprecación de `qwen3_next_mtp`) |

### 5.3 Volúmenes

```
/home/usuario/Proyectos/models-cache          → /root/.cache/huggingface   (pesos HF)
/home/usuario/Proyectos/kv-offload           → /kv-offload                (tier L3 NVMe del KV)
/home/usuario/.cache/vllm/qwen38-27b-fp8     → /root/.cache/vllm          (JIT vLLM)
/home/usuario/.cache/torch/qwen38-27b-fp8    → /root/.cache/torch
/home/usuario/.cache/triton/qwen38-27b-fp8   → /root/.cache/triton
/home/usuario/.cache/torchinductor/...       → /root/.cache/torchinductor
../vllm/_genesis → /usr/local/lib/python3.12/dist-packages/vllm/_genesis:ro  (patcher, READ-ONLY)
```

### 5.4 Entrypoint

```bash
pip install pandas scipy xxhash -q        # deps runtime de Genesis (¡en cada boot!)
python3 -m vllm._genesis.patches.apply_all # aplica text-patches sobre el vLLM de la imagen
exec vllm serve "$@"
```

Nota: aquí NO se usa el plugin entry-point sino invocación directa del
orquestador. Solo `_genesis` es ro; el vLLM de la imagen es escribible y ahí
operan los text-patches.

### 5.5 Env flags Genesis activos (los ~30 del bloque)

#### KV offloading multi-tier (el bloque más denso)

| Flag | Valor | Qué hace / decisión documentada |
|:---|:---|:---|
| `GENESIS_ENABLE_PN81_KV_DISK_QUOTA` | 1 | Cuota del tier disco. vLLM no trae cuota NI limpieza (38 GB en una sesión de test). Hook `on_schedule_end()` + purga de mmap huérfanos de /dev/shm al arrancar. **Cuota NO es por rank**: UN FileSystemTierManager en el proceso scheduler (offloading/scheduler.py:265); lo por-rank es el mmap RAM del worker |
| `GENESIS_KV_DISK_MAX_GB` | 30 | Techo de disco |
| `GENESIS_KV_DISK_CHECK_SECS` | 60 | Chequeo por TIEMPO, no por pasos (con engine ocioso el gate viejo de 2000 pasos nunca corría: 22 min / 61 requests sin llegar) |
| `GENESIS_KV_DISK_ORPHAN_DAYS` | 3 | Borra caches de OTROS modelos sin uso >N días (habían juntado 142 GiB, 54 de un modelo ya borrado) |
| `GENESIS_KV_DISK_ORPHAN_CHECK_SECS` | 3600 | Barrido periódico de huérfanos (antes solo al arranque: un dir huérfano POST-arranque no se revisaba nunca; caso medido _3ef784eef730_r0 con 25,47 GiB) |
| `GENESIS_KV_DISK_TARGET_RATIO` | 0.85 | Objetivo de ocupación |
| `GENESIS_ENABLE_PN88_KV_TIER_METRICS` | 1 | Telemetría Prometheus por tier/direction/agent/group (vLLM no publica NI UNA métrica del tier disco). ⚠️ Para desgaste SSD usar `kv_tier_disk_written_bytes_total` (no `kv_tier_bytes_total`, que sobreestima: cuenta re-escrituras como bytes y no sabe de compresión). Hit-rate del disco leerlo POR GRUPO (con PN93 los recurrentes cuentan miss; importa g3=atención) |
| `GENESIS_KV_AGENTS` | coach,primary,...,art | Allowlist CERRADA para label `agent` (viene del cliente vía kv_transfer_params.genesis_agent); lo no listado → `other`/`unknown` |
| `GENESIS_ENABLE_PN90_KV_DISK_WRITE_GATING` | 1 | L3 exclusiva: baja a disco solo al ser desalojado de L2 RAM y solo si el agente está en allowlist |
| `GENESIS_PN90_MAX_PENDING_MB` | 256 | Techo RAM de snapshots de democión pendientes (bloque pesa 27,6 MB; al llegar al tope NO se copia) |
| `GENESIS_ENABLE_PN91_KV_LAZY_STREAMING` | 1 | Tras N tiempo difiriendo, arranca con el prefijo ya listo (ataca TTFT de prefijos grandes; tráfico offloading = 1,0–1,9% wall time) |
| `GENESIS_PN91_MAX_DEFER_SECONDS` / `_STEPS` | 2.0 / 32 | Presupuesto en segundos (paso va de ~30ms ocioso a ~1,6s con prefill largo) + red de seguridad en pasos |
| `GENESIS_ENABLE_PN92_KV_FP4_COMPRESSION` | 1 | Instala hooks del codec 4-bit en fs/io.py (necesario para LEER bloques comprimidos) |
| `GENESIS_PN92_COMPRESS_ON_WRITE` | **0** | Compresión al escribir APAGADA: 12,4% error L2 relativo en grupo de atención (ratio 0,563); el KV ya corre fp8 y el híbrido no tolera menos; disco no es escaso (26/266 GB) |
| `GENESIS_PN92_KV_DTYPE` | fp8_e4m3 | Debe coincidir con --kv-cache-dtype o el codec se auto-desactiva ✓ |
| `GENESIS_PN92_CHUNK_ELEMS` | 1048576 | Bloque de trabajo del codec (antes materializaba bloque entero en float32: 28→116 MB × 16 hilos = 1,8 GB transitorio) |
| `GENESIS_ENABLE_PN93_SPARSE_GDN` | 1 | Guarda 1 de cada N fronteras de bloque en grupos recurrentes g0/g1/g2 (GDN). vLLM ya los lee con _sliding_window_lookup(window=1) — los intermedios jamás se leían. GDN = 102 de 136 KB/token (75%) |
| `GENESIS_PN93_GDN_CHECKPOINT_STRIDE` | **16** | Stride elegido en barrido 2026-08-23: OFF 21,32 GB al SSD / stride8 6,17 GB / stride16 4,92 GB, tiempo idéntico (~139s) y cero hit_truncated. Costo: recomputo ocasional al reanudar en medio de prefijo (ramas divergentes) |
| `GENESIS_ENABLE_PN99_GPU_COMPRESSED_L2` | 1 | KV baja a L2/L3 cuantizada a 4 bits con cuantización EN GPU: 1,78× más bloques (12 GiB: 444→790 bloques). ⚠️ Cambia layout de /dev/shm y /kv-offload: al prender/apagar BORRAR /kv-offload. PN92 debe seguir apagado (doble compresión = desastre) |
| `GENESIS_PN100_RING_BLOCKS` | 32 | Anillo de staging para agentes EFÍMEROS (coder/explorer sin persist_disk): K bloques reservados al final de L2 fuera del presupuesto del cache; el preámbulo compartido SÍ va al cache (hash de contenido ya visto). Cubre ~6 subagentes de 25k |

#### Estructura/streaming agéntico

| Flag | Valor | Qué hace |
|:---|:---|:---|
| `GENESIS_ENABLE_PN56_QWEN3CODER_XML_FALLBACK` | 1 | Fallback parseo XML qwen3_coder (evita filtrar `{}` placeholder a clientes OpenAI estrictos) |
| `GENESIS_ENABLE_P61B_STREAMING_OVERLAP` | 1 | Previene fuga de tags parciales en streaming |
| `GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING` | 1 | Timing de cierre de bloque think en decodificación especulativa |
| `GENESIS_ENABLE_PN51_QWEN3_STREAMING_THINKING_DISABLED` | 1 | Sin bloques think en streaming durante tool-calls |
| `GENESIS_ENABLE_PN66` | 1 | Fuga de `</think>` en multiturno (backport vllm#41696) |
| `GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER` | 1 | Inyecta recordatorio `<tool_call>` al final del último mensaje (>1000 chars); estilo `qwen3_coder` XML |

#### Memoria

| Flag | Valor | Qué hace |
|:---|:---|:---|
| `GENESIS_ENABLE_PN77_FP8_LM_HEAD` | 1 | Cuantización FP8 del LM head → ~1,2 GiB VRAM/GPU (en stock v0.23 el LM head es embedding no-cuantizado corriendo cuBLAS fp16 — ver §8.3) |
| `GENESIS_ENABLE_P5B` | 1 | Estrategia page-size KV para Qwen (~34% menos VRAM por bloque) |
| `GENESIS_ENABLE_PN25_SILU_INDUCTOR_SAFE` + `GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL` | 1 | Pools de buffers FFN ("Cliff 1"): atacan el OOM confirmado por traceback 2026-08-15 — el grafo inductor pedía 108 MiB = (T=3253 × intermediate_size 17408) fp16 |
| `GENESIS_ENABLE_PN19_SCOPED_MAX_SPLIT` | 1 | Durante la CARGA fija max_split_size_mb=20 (mínimo PyTorch) y restaura al terminar (fragmentación de carga, vllm#41268: 200-500 MiB inutilizables) |

#### Spec-decode / GDN / diagnóstico

| Flag | Valor | Qué hace |
|:---|:---|:---|
| `GENESIS_ENABLE_P100` | 1 | CUDAGraphs FULL para spec-decode (+5-10% latencia single-stream) |
| `GENESIS_ENABLE_P66_CUDAGRAPH_SIZE_FILTER` | 1 | Filtra capture-sizes a múltiplos de K+1 (hoy no-op: 4..40 ya son múltiplos de 4; red de seguridad para barridos de K — vllm#28015) |
| `GENESIS_ENABLE_P107_MTP_TRUNCATION_DETECTOR` | 1 | Detector de truncamiento MTP en frontera reasoning→tool_call |
| `GENESIS_ENABLE_PN50_GDN_FUSED_PROJ` / `PN59_STREAMING_GDN` / `PN54_GDN_CONTIGUOUS_DEDUP` | 1 | Familia GDN: proj fusionada, streaming-GDN (window-iterative, −142 MiB/GPU boot, −95% drift), dedup contiguo |
| `GENESIS_ENABLE_PN57_TQ_CENTROIDS_DISK_CACHE` | 1 | Cache disco de centroides TurboQuant (patrón de referencia para la idea de cache-Marlin, §12.1) |
| `GENESIS_ENABLE_PN80_GDN_H_BUDGET_PROBE` | **0** | Sonda de presupuesto VRAM GDN — apagada; era para tunear gpu-memory-utilization |
| `GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT` | **0** | Apagado 2026-08-19: no-op cuando el target ya está quantized estático (este caso) |
| `GENESIS_ENABLE_MTP_QUANT_CACHE` | **0** | Apagado tras auditoría (header de mtp_cache.py, 7 bugs): round-trip FP8→INT8→FP16 que dequantizaba de vuelta = toda la pérdida, cero beneficio; además corrompía weight_scale_inv |

#### Otros env

```bash
VLLM_FLOAT32_MATMUL_PRECISION=high      # torch.set_float32_matmul_precision("high") — irrelevante p/ Marlin (§8.5)
VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=1024
VLLM_MARLIN_USE_ATOMIC_ADD=1            # casi seguro inerte en este modelo (§8.2)
VLLM_USE_FUSED_MOE_GROUPED_TOPK=1       # INERTE: modelo denso (§7)
VLLM_USE_FLASHINFER_SAMPLER=1           # activo y válido en sm_86 (§7)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
PYTHONHASHSEED=0                        # FIJA la raíz de hashes de bloque: sin esto el tier disco
                                        # queda INALCANZABLE tras cada reinicio (kv_cache_utils.py:112
                                        # hace NONE_HASH=os.urandom(32)); los 57 GB de /kv-offload
                                        # no se borran, simplemente ningún lookup matchea
VLLM_WORKER_MULTIPROC_METHOD=spawn
OMP_NUM_THREADS=1
VLLM_API_KEY=${VLLM_API_KEY:-}          # viene de compose/.env (obligatorio)
TORCHINDUCTOR_CACHE_DIR / TRITON_CACHE_DIR → mounts persistentes
GENESIS_VERBOSE=0  GENESIS_LOG_LEVEL=INFO
```

### 5.6 Flags de `vllm serve`

| Grupo | Flags |
|:---|:---|
| API | `--disable-access-log-for-endpoints /metrics,/health` · `--enable-prompt-tokens-details` (cached_tokens por request; hoy suma hit local + KV externo, el split existe en PrefillStats pero output_processor.py:630 lo colapsa) · `--host 0.0.0.0 --port 8320 --api-key` |
| Backend | `--attention-backend FLASHINFER` · `--moe-backend flashinfer_trtllm` *(inerte, modelo denso)* · `--enable-flashinfer-autotune` *(no-op sm_86)* · `--trust-remote-code` · `--generation-config vllm` |
| Precisión | `--dtype float16` · `--kv-cache-dtype fp8_e4m3` · `--mamba-ssm-cache-dtype float16` |
| Paralelismo | `--tensor-parallel-size 2` · `--disable-custom-all-reduce` *(bug captura cudagraph, §10)* |
| Contexto/batch | `--max-model-len 262144` · `--max-num-seqs 10` *(barrido documentado: 12→OOM GDN; 10 validado con 2 cargas)* · `--max-num-batched-tokens 1664` · `--long-prefill-token-threshold 832` · `--enable-chunked-prefill` |
| VRAM | `--gpu-memory-utilization 0.745` *(bajó de 0.76→0.82-era offloading→0.745 el 08-23 por scratch de compresión PN99; barrido 0.7566/0.76 pasan 5×30k, 0.775 muere)* |
| Multimodal | `--limit-mm-per-prompt {"image":2,"video":0}` · `--mm-processor-kwargs {"max_pixels":2000000,"min_pixels":65536}` (techo de resolución: el profiler reserva según el ítem MÁS GRANDE; +5,4% KV medido) |
| Spec-decode | `--speculative-config {"method":"mtp","num_speculative_tokens":3}` · `--cudagraph-capture-sizes 4..40 paso 4` *(40 = 10 seqs × 4 tok/paso; extender a 48 costó +0,26 GiB y se revirtió)* · `--async-scheduling` |
| KV offload | `--enable-cumem-allocator` *(OBLIGATORIO para OffloadingConnector; efecto lateral: profiler SOBREESTIMA KV disponible)* · `--kv-transfer-config {"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","cpu_bytes_to_use":6442450944,"eviction_policy":"arc","store_threshold":1,"secondary_tiers":[{"type":"fs","root_dir":"/kv-offload"}]}}` |
| Cache | `--enable-prefix-caching` · `--performance-mode throughput` |
| Parsing/tools | `--reasoning-parser qwen3` · `--enable-auto-tool-choice` · `--tool-call-parser qwen3_coder` |
| Sampler | `VLLM_USE_FLASHINFER_SAMPLER=1` (env) |

### 5.7 Decisiones de tuning documentadas en los comentarios (no perder)

- **Dos sitios de OOM distintos** (diagnóstico 08-14/15): (a) max-num-seqs 12 →
  OOM en `chunk_gated_delta_rule_fwd_h` (GDN, h tensor); (b) prompts grandes →
  OOM en grafo inductor pidiendo 108 MiB (FFN, "Cliff 1" — NO Cliff 2).
- **Fórmula medida de h**: `h_bytes/GPU = tokens_forward × 12,05 KiB`
  (= NT×H×V×K×2, NT=T/64, H=24/rank, V=K=128, fp16). Se asigna por BATCH
  completo. **Verificada contra código en §9.2** (exactamente 12288 B/token).
- **KV no es el límite** (345k tokens libres): falta margen para transitorios
  que viven FUERA del pool perfilado por gpu-memory-utilization.
- **Throughput**: 1 seq → 88 tok/s | 5 seqs → 308 tok/s (3,5×). Validaciones:
  `tests/repro/carga_paralela.py` (10×40k prompt → 10/10, 643 MiB libres pico)
  y `carga_generacion.py` (10×4000 out → 509 tok/s agg., MTP 72,1% aceptación).
- **Quantum del scheduler**: el chunk de prefill se trunca a múltiplo de
  block_size en modo mamba align (`_mamba_block_aligned_split`). Barrido
  1x1600/2x3200/3x4800/4x6400: DOS regímenes (1x: agentes esperan 127s y luego
  corren a 51ms/tok; ≥2x: entran a 3,2s pero lockstep a 1615ms/tok). Para
  subagentes sin streaming gana 1x en TIEMPO TOTAL (132s vs 148s). Piso duro:
  <1600 el engine no arranca; debajo por threshold se cuelga SIN error.
- **Prefill pegado a techo de cómputo** (~770 tok/s ≈ 41,6 de 71 TFLOPS fp16
  de dos 3090; Ampere desempaqueta FP8→fp16: gana memoria, no cómputo).
  Subir max-num-batched-tokens NO acelera prefill (133,9/136,4/136,9/136,4s
  en las 4 configs) y cuesta KV.
- **store_threshold=2 probado y descartado** (08-23): lookup() solo corre
  cuando alguien busca el bloque; store ocurre ANTES → contador nunca llega a
  2 → desactiva L2 en vez de dar resistencia a escaneo.
- **PP=2 imposible**: `SupportsPP` solo en `Qwen3_5ForCausalLMBase` (texto);
  este checkpoint resuelve a `Qwen3_5ForConditionalGeneration` (vision) que no
  lo implementa. TP=2 no es elección, es lo único que hay.
- **Techo L2 RAM**: 6 GiB (máquina 30 GB, /dev/shm 16 GB; >8 GiB margen fino
  contra OOM killer). Capacidad efectiva: sin PN93 46k tokens; stride4 102k;
  stride8 123k. Un hilo de 126k necesita 6,1 GB con stride 8.

---

## 6. Estado del contenedor al momento del análisis

Inspección read-only (2026-08-24):

```
genesis-27b-qwen38-fp8 | vllm/vllm-openai:v0.23.0 | Up About an hour | 0.0.0.0:8320->8320/tcp
restart=no | started=2026-08-24T02:32:10Z
```

Coherente con el YAML: sin restart policy, puerto publicado, imagen pineada.

---

## 7. Mapa de kernels CUDA en ejecución

Síntesis del dispatch real de v0.23.0 para ESTE modelo y ESTE hardware.

| Componente | Kernel real | Referencia |
|:---|:---|:---|
| Atención full (~16 capas) — prefill | FlashInfer **FA2 FMHA** vía `BatchPrefillWithPagedKVCacheWrapper(backend="auto")`, causal, KV fp8_e4m3 con dequant in-kernel (escalares k_scale/v_scale), **Q en fp16** | `v1/attention/backends/flashinfer.py:766-785,1175-1193,1578-1585` |
| Atención full — decode | `BatchDecodeWithPagedKVCacheWrapper(use_tensor_cores=True, backend="auto")` + fast_plan para cudagraph | `flashinfer.py:787-825,1745-1752,1863-1953` |
| Ruta TRTLLM attention | **Nunca**: exige `is_device_capability_family(100)` (Blackwell) | `utils/flashinfer.py:350-363,388-469` |
| Cascade attention | Deshabilitada siempre en este backend (:1255-1262); además async-scheduling+spec la desactiva | `config/vllm.py:1017-1024` |
| GDN prefill (48 capas) | Pipeline **Triton FLA**: `chunk_local_cumsum` → `chunk_scaled_dot_kkt_fwd` → `solve_tril` → `recompute_w_u_fwd` → `chunk_gated_delta_rule_fwd_h` → `chunk_fwd_o` | `model_executor/layers/fla/ops/{cumsum,chunk_scaled_dot_kkt,solve_tril,wy_fast,chunk_delta_h,chunk_o}.py` |
| GDN decode | `fused_sigmoid_gating_delta_rule_update` (in-place sobre ssm_state) + `causal_conv1d_update`; variante packed con `num_warps=1, num_stages=3` FIJOS sin autotune | `fla/ops/fused_sigmoid_gating.py:24`, `fla/ops/fused_recurrent.py:198-199,438-439`, `causal_conv1d.py:1069` |
| Linear FP8 (qkv, o_proj, gate_up, down) | **Marlin W8A16** (`ops.marlin_gemm`, b_q_type=float8_e4m3fn): pesos fp8 empacados int32, dequant FUSIONADO en kernel, activaciones fp16, reduce fp32 | `quantization/utils/marlin_utils_fp8.py:42-89`, `csrc/quantization/marlin/marlin.cu:533` |
| LM head | **cuBLAS fp16** vocab-paralelo + gather (`UnquantizedEmbeddingMethod`; `get_quant_method` devuelve None para embeddings) | `vocab_parallel_embedding.py:67-75,270-274`, `fp8.py:176-208` |
| Sampler | FlashInfer rejection sampling sin sort (`top_k_top_p_sampling_from_logits`) — soportado sm_86 (rango 7.5–12.1) | `v1/sample/ops/topk_topp_sampler.py:21-67,460-497`, `flashinfer.py:407-410` |
| All-reduce TP | **PYNCCL** directo (`ncclAllReduce`, dtype f16; NCCL elige Ring Simple/LL128 internamente; vLLM no setea NCCL_ALGO/PROTO) | `distributed/device_communicators/cuda_communicator.py:246-303`, `pynccl.py:166-197` |
| MTP draft | **Reusa kernels del target**: propia `VocabParallelEmbedding` + `fc` ColumnParallelLinear(hidden*2→hidden) + 1× decoder layer full_attention + ParallelLMHead; propio cudagraph wrapper | `qwen3_5_mtp.py:59-159,377-388`, `gpu_model_runner.py:598-603,5240-5243` |
| Copias de estado mamba (align) | `MambaCopySpec` conv/temporal orquestadas en model runner | `mamba_utils.py:263-335`, `gpu_model_runner.py:1005-1008,1515,4159` |

### Flags inertes o no-ops detectados

| Flag | Motivo |
|:---|:---|
| `--moe-backend flashinfer_trtllm` | Modelo DENSO (`config.model_type` → `Qwen3NextMLP`); no existe capa `FusedMoE` en el grafo; el flag solo se lee al construir FusedMoE (`fused_moe/layer.py:326`) |
| `VLLM_USE_FUSED_MOE_GROUPED_TOPK=1` | Solo lo lee `grouped_topk_router.py:92-99` (y exigiría `e_score_correction_bias`, estilo DeepSeek) |
| `--enable-flashinfer-autotune` | Gate `has_device_capability(90)` en `kernel_warmup.py:67-74` → **se salta silencioso en sm_86**. Además su cache persistente está muerto (`_FLASHINFER_USE_PERSISTENT_CACHE=False`, :109-112) |
| `VLLM_MARLIN_USE_ATOMIC_ADD=1` | Requiere `n<2048 ∧ k≥2048` (`marlin_utils.py:469-489`); las capas grandes del 27B con TP=2 tienen N≥2048 → nunca dispara (auditar para confirmar 100%) |
| `VLLM_FLOAT32_MATMUL_PRECISION=high` | Solo toca matmuls fp32 (`gpu_worker.py:134-136`); Marlin tiene su propio reduce fp32 (`USE_FP32_REDUCE_DEFAULT=True`) y el LM head es fp16 → irrelevante |

### Coherencias verificadas contra código

- **cudagraph-capture-sizes 4..40 paso 4**: correcto por construcción —
  `adjust_cudagraph_sizes_for_spec_decode` redondea hacia arriba a múltiplos
  de K+1=4 (y múltiplo común con TP) (`config/compilation.py:1473-1518`);
  cada seq aporta `uniform_decode_query_len = 1+K = 4`
  (`gpu_model_runner.py:814`).
- **async-scheduling**: compatible con MTP (EagleModelTypes); NO cambia
  kernels — solapa copia GPU→CPU de tokens con stream+event separados
  (`gpu_model_runner.py:689-697,631-633`).
- **Modelo denso aunque haya MoE classes**: la decisión es por
  `config.model_type` en `Qwen3_5DecoderLayer.__init__`
  (`qwen3_5.py:155-171`): `qwen3_5_moe_text` → SparseMoeBlock;
  `qwen3_5_text` → MLP denso. El MTP head también usa DecoderLayer → también
  denso (`qwen3_5_mtp.py:100-106`).

---

## 8. Camino FP8 detallado (GEMM en Ampere)

### 8.1 Cadena de dispatch completa (por qué Marlin)

`Fp8LinearMethod.__init__` con `weight_block_size=[128,128]`
(`fp8.py:284-298`) → `create_weights` llama `init_fp8_linear_kernel`
(`fp8.py:374-383`) → lista CUDA `_POSSIBLE_FP8_BLOCK_KERNELS`
(`kernels/linear/__init__.py:308-329`):

```
1. FlashInferFp8DeepGEMMDynamicBlockScaledKernel  → exige is_device_capability(90)  ✗
2. DeepGemmFp8BlockScaledMMKernel                 → Hopper/Blackwell only           ✗
3. CutlassFp8BlockScaledMMKernel                  → cc≥90 (scaled_mm_entry.cu:160)  ✗
4. MarlinFP8ScaledMMLinearKernel                  → cc≥75 ∧ cc<89                   ✓ ELEGIDO
5. TritonFp8BlockScaledMMKernel                   → fallback final (nunca se llega)
```

Elección en `choose_scaled_mm_linear_kernel` (línea 450): primero que pase
`is_supported(cc)` + `can_implement`. En `apply()` el camino es directo a
`apply_weights` → `apply_fp8_marlin_linear` → `ops.marlin_gemm(...)`.
W8A8 explícitamente rechazado (`RuntimeError("Marlin W8A8 is not supported.")`)
→ **es W8A16: pesos fp8, activaciones fp16**. Warning esperable al boot:
*"Your GPU does not have native support for FP8 computation... leveraging the
Marlin kernel"* (`marlin_utils_fp8.py:97-102`).

Adaptación del block-quant: scales 2D (K/128, N/128) → `repeat_interleave`
a group-128 1D (K/128, N) (`marlin_utils_fp8.py:174-187`) +
`marlin_permute_scales(group_size=128)` + `fp8_fused_exponent_bias_into_scales`
(pliega el bias de exponente fp8→fp16 en el scale, líneas 28-39).

### 8.2 Repack y atomic-add

- **Repack una vez por arranque** en `process_weights_after_loading`
  (`fp8.py:385-392`): `pack_fp8_to_int32` + `ops.gptq_marlin_repack` (kernel
  CUDA) + permute/repeat scales. **Sin caché en disco** → costo de segundos a
  decenas de segundos para 27B en CADA boot. ← base de la oportunidad §12.1.
- **`VLLM_MARLIN_USE_ATOMIC_ADD`**: `should_use_atomic_add_reduce`
  (`marlin_utils.py:469-489`) exige n<2048 ∧ k≥2048 ∧ env=1 ∧ ¬(bf16∧sm<90).
  Efecto: con slices-K múltiples, atomicAdd directo al resultado en vez de
  parciales en C_tmp + segunda pasada (`marlin_template.h:1730,1903,1928,2001`;
  gating adicional `part_use_atomic_add` en `marlin.cu:511-512`). Útil para
  decode con N chico; probablemente inerte aquí (auditar formas reales).

### 8.3 LM head

`ParallelLMHead` hereda `VocabParallelEmbedding` (NO es `LinearBase`) →
`Fp8Config.get_quant_method` devuelve `None` → `UnquantizedEmbeddingMethod` →
GEMM cuBLAS en params_dtype (fp16) + `tensor_model_parallel_gather` en
`LogitsProcessor._get_logits` (`logits_processor.py:75-104`). Con PN77
(Génesis) los PESOS se guardan fp8 para ahorrar ~1,2 GiB/GPU; el compute path
stock sigue siendo fp16.

### 8.4 Machete y otros

`MacheteLinearKernel.get_min_capability()=90` y ni siquiera está en las listas
FP8 (vive en las WNA16). Descartado doblemente.

### 8.5 Conclusión de calidad

Marlin W8A16 con dequant fusionado y reduce fp32 **ya es el óptimo posible en
sm_86**. No existe camino W8A8 (requeriría tensor cores FP8). La única palanca
teórica superior sería W4A16 Marlin (mitad de ancho de banda de pesos), pero
exige requantizar el checkpoint y contradice la decisión de calidad documentada
(FP8 verificado 99,9% idéntico). **No recomendado.**

---

## 9. Camino atención + GDN detallado

### 9.1 Atención full (FlashInfer nativo)

- KV fp8_e4m3 es dtype soportado nativo (`supported_kv_cache_dtypes`,
  `flashinfer.py:327-335`); `plan()` pasa `kv_data_type=torch.float8_e4m3fn`,
  causal=True; `run()` recibe k_scale/v_scale → dequant DENTRO del kernel FA2.
- Q queda en dtype del modelo (fp16): la q-quantización fp8 solo existe en la
  ruta TRTLLM/SM100 (:644-662, :962-964).
- Los kernels FA2 viven en el paquete `flashinfer` (JIT/cubins), no en el
  árbol de vLLM. FA3 es SM90+.

### 9.2 GDN prefill (FLA Triton) — autotune y la constante de h

- Todos los kernels chunked usan `@triton.autotune` con `key=["H","K","V","BT"]`;
  `BT = FLA_CHUNK_SIZE = 64` fijo (`fla/ops/utils.py:31`).
- **Diferenciación por arch mínima**: `check_shared_mem()` (`utils.py:167-200`)
  compara shared mem reportada por Triton contra umbrales
  `{ADA:101376, AMPERE:166912, HOPPER:232448, DEFAULT:102400}`. GA102 reporta
  ~101376 B < 102400 → `chunk_o` usa `BKV_LIST=[32,64]` (pierde [64,128]) por
  un margen mínimo; `NUM_WARPS=[2,4,8]` (no-Hopper).
- `fwd_h` configs: BV∈{32,64} × warps{2,4} × stages{2,3,4}; grid
  `(cdiv(V,BV), N*H)`.
- **Constante h verificada**: `h = k.new_empty(B, NT, H, V, K)`
  (`chunk_delta_h.py:350`), NT=cdiv(T,64), dtype=k (fp16):
  `H·V·K·2/BT = 24·128·128·2/64 = 12288 B = 12 KiB/token` con H=24 v-heads/rank
  (48 heads / TP2) → **la fórmula del compose es exacta**. Estado final se
  devuelve fp32 y se castea al dtype del cache al escribir
  (`qwen_gdn_linear_attn.py:1532`).
- Warmup explícito del autotuner durante profiling
  (`_warmup_prefill_kernels`, `qwen_gdn_linear_attn.py:1068-1195`) para evitar
  OOM de autotune post-allocation.
- FlashInfer GDN prefill existe pero SOLO SM90/SM100 → en sm_86 siempre
  Triton (`_resolve_gdn_prefill_backend`, :150-211).

### 9.3 Estados recurrentes y page size

- Formas: conv_state `(conv_dim/tp, conv_kernel-1+num_spec)`;
  ssm_state `(num_v_heads/tp, head_v_dim, head_k_dim)`. ssm_state sigue
  `--mamba-ssm-cache-dtype` (fp16); conv_state sigue dtype del modelo
  (`mamba_utils.py:108-116,212-234`).
- Modo align: prefix-caching ON → "all"/"align"
  (`models/config.py:350-394`); exige chunked prefill.
- **El 1600/832 NO es una constante del código**: es `attn_block_size`
  calculado en `platforms/interface.py:610-699` a partir de
  `mamba_page_size`, `kernel_block_alignment` (FlashInfer ofrece [16,32,64])
  y el tamaño de página KV fp8. Por eso puede cambiar entre configs/modelos —
  explica la discrepancia interna del compose (§13.1).

---

## 10. Comunicación TP=2 y el bug del custom all-reduce

### 10.1 Cadena de dispatch de all-reduce (v0.23)

```
NCCL_SYMM_MEM → QUICK_REDUCE (ROCm) → FLASHINFER → CUSTOM → SYMM_MEM → PYNCCL → torch.distributed
```

(`cuda_communicator.py:246-303`; log de selección en `_log_all_reduce_allreduce_backend_selection`, :183-244.)

Con `--disable-custom-all-reduce`, `ca_comm` no se crea → todo va por
**PYNCCL** (`ncclAllReduce` directo, :166-197). Volumen: 2 AR/capa
(RowParallel o_proj/down_proj + GDN out_proj) × 64 capas; tamaño
`num_tokens × hidden × 2B` → ~90 MB por AR en prefill (1664 tokens), cientos
de KB en decode (batch 40).

### 10.2 El bug documentado y su causa raíz (hipótesis verificada mecánicamente)

**Síntoma** (documentado en el compose, reproducido 3× incluido con PN94):
habilitar custom AR → durante captura de cudagraphs:
`Failed: Cuda error csrc/custom_all_reduce.cuh:455 'invalid argument'` →
muere el engine.

**Trazado del código**:

1. Durante captura, `allreduce()` detecta stream capturing y apila el puntero
   de ENTRADA en `graph_unreg_buffers_` (`custom_all_reduce.cuh:542-546`).
2. Al salir de `capture()` → `register_graph_buffers()`
   (`custom_all_reduce.py:194-226`) → `ops.get_graph_buffer_ipc_meta(self._ptr)`
   (línea 210).
3. Eso ejecuta (`custom_all_reduce.cuh:442-460`): resuelve la BASE del rango
   con `cuPointerGetAttribute(CU_POINTER_ATTRIBUTE_RANGE_START_ADDR)` y llama
   **`cudaIpcGetMemHandle(base_ptr)` — línea 455**.
4. `CUDACHECK` imprime exactamente el formato del síntoma y hace
   `exit(EXIT_FAILURE)` (cuh:22-30). Sin fallback ni chequeo de tipo de
   asignación.

**Por qué falla**: `cudaIpcGetMemHandle` retorna `cudaErrorInvalidValue`
cuando el puntero base NO proviene de `cudaMalloc` (IPC-exportable). La
memoria VMM (`cuMemCreate`+`cuMemMap`) requiere
`cuMemExportToShareableHandle` — **no es exportable por IPC**.

**Conexión con este stack**: el contenedor usa `--enable-cumem-allocator`
(OBLIGATORIO para OffloadingConnector). El allocator cumem
(`csrc/cumem_allocator.cpp`: `cuMemCreate`:130, `cuMemMap`:145,
`cuMemSetAccess`:187, VA con `cuMemAddressReserve`:323) es VMM pura, instalado
como pluggable allocator acotado a MemPool contexts SOLO para pesos
(`load_model`, `gpu_worker.py:349-356`) y KV cache
(`initialize_from_config`, :577). La captura de graphs corre FUERA de esos
pools… pero `torch.cuda.empty_cache()` durante la captura
(`gpu_model_runner.py:6546`) y el reuso de rangos VA reservados pueden hacer
que el segmento base de algún buffer de entrada resuelva a memoria VMM.

Los buffers INTERNOS del custom AR son `cudaMalloc` puro (exportables por
construcción, `libtorch_stable/custom_all_reduce.cu:157-189`) — los
sospechosos son los buffers de entrada/salida de los ARs capturados.

**Estado**: hipótesis mecánicamente consistente con el síntoma exacto y con
que PN94 (pinneo) no cambiara nada. Pendiente de confirmación experimental
(p.ej. correr custom AR con `--enable-cumem-allocator` OFF en un motor sin
offloading → debería pasar; o instrumentar el tipo de asignación del base_ptr).

### 10.3 Alternativas evaluadas en el árbol

| Vía | Veredicto |
|:---|:---|
| NCCL symmetric memory (`VLLM_USE_NCCL_SYMM_MEM`) | **Bloqueada para TP=2**: `should_nccl_symm_mem_allreduce` exige `world_size ≥ min_world_size=4` (`all_reduce_utils.py:89,121`). Además requiere NCCL≥2.27.3, torch≥2.8, JIT del allocator. Parcheable (bajar a 2) para experimento |
| Torch symm-mem (`VLLM_ALLREDUCE_USE_SYMM_MEM`, default ON) | Deshabilitado en sm_86: `SYMM_MEM_ALL_REDUCE_MAX_SIZES` solo tiene claves "9.0"/"10.0"/"10.3" (`symm_mem.py:65-71`) y exige multicast (NVSwitch) |
| Sequence parallelism (compilation passes) | **EXISTE en v0.23**: `FirstAllReduceRMSNormPattern`/`MiddleAllReduceRMSNormPattern` reemplazan `all_reduce→rms_norm` por `reduce_scatter→rms_norm→all_gather` vía pynccl (`compilation/passes/fusion/sequence_parallelism.py:150-183`). Apagada en sm_86 porque `SP_MIN_HIDDEN_SIZE={90:8192,100:8192}` no tiene entrada 86 → threshold None. Forzable con `pass_config.enable_sp=True` + `sp_min_token_num` manual (`config/vllm.py:1170-1199`). Reduce a la mitad el volumen de comm en prefill y saca el tráfico del custom AR |

---

## 11. Hallazgos que validan o corrijen el compose

| # | Hallazgo | Tipo |
|:--|:---|:---|
| 1 | Fórmula h = 12 KiB/token exacta (12288 B con H=24/rank, BT=64, fp16) | ✅ Valida |
| 2 | capture-sizes 4..40 ya son múltiplos de K+1=4 (correcto por diseño del ajuste automático) | ✅ Valida |
| 3 | `PYTHONHASHSEED=0` necesario: `NONE_HASH=os.urandom(32)` en `kv_cache_utils.py:112` | ✅ Valida |
| 4 | `--moe-backend` + `VLLM_USE_FUSED_MOE_GROUPED_TOPK` inertes (modelo denso) | ⚠️ Corrige (higiene) |
| 5 | `--enable-flashinfer-autotune` no-op en sm_86 (gate SM≥9.0) | ⚠️ Corrige |
| 6 | `VLLM_MARLIN_USE_ATOMIC_ADD` casi seguro inerte (gate n<2048∧k≥2048) | ⚠️ Probable corrección (auditar) |
| 7 | `VLLM_FLOAT32_MATMUL_PRECISION` irrelevante para Marlin/LM-head | ⚠️ Corrige |
| 8 | Bug custom AR ≠ topología: interacción VMM(cumem) × cudaIpcGetMemHandle (§10.2) | 🔍 Nueva hipótesis de causa raíz |
| 9 | El "block 1600 vs 832" del compose: ninguno es constante; es `attn_block_size` calculado por plataforma (§9.3) | 🔍 Explica discrepancia |

---

## 12. Oportunidades de optimización (rankeadas)

### 12.1 Cache en disco del repack Marlin ⭐ (impacto: boot time; esfuerzo: bajo-medio)

**Problema**: el repack fp8→int32 + permute de scales corre en cada arranque
en memoria, sin caché (`prepare_fp8_layer_for_marlin`,
`marlin_utils_fp8.py:92-198`). Para 27B: segundos a decenas de segundos por
boot, multiplicado por cada reinicio (y esta operación NO tiene restart
policy a propósito → cada crash implica re-repack).

**Idea de parche Genesis (patrón PN57)**: persistir los tensores repackeados
keyed por hash de (pesos originales, forma, versión kernel) en el volume
persistente de cache; cargar directo si existe. Misma filosofía que
`GENESIS_ENABLE_PN57_TQ_CENTROIDS_DISK_CACHE`.

### 12.2 Autotune Triton GDN: ampliar BKV_LIST para sm_86 ⭐⭐ (impacto: throughput prefill; esfuerzo: bajo)

**Problema**: `check_shared_mem()` deja a GA102 fuera de las configs
`BKV_LIST=[64,128]` de `chunk_o` por 1 KB de margen (reporta ~101376 <
umbral DEFAULT 102400). El prefill GDN es parte del cuello (prefill pegado a
techo de cómputo).

**Idea**: parche que amplíe la lista de candidatos para sm_86 (el autotune
descarta solo lo que falla en runtime; riesgo bajo, medición directa).
Alternativa complementaria: revisar si `solve_tril`/`wy_fast` tienen
headroom similar.

### 12.3 All-reduce: barrido NCCL + re-perfil con P2P (impacto: desconocido hasta perfilar; esfuerzo: bajo)

- vLLM no setea `NCCL_ALGO`/`NCCL_PROTO` → barrido puro por env
  (`NCCL_ALGO=Ring,NVLS` × `NCCL_PROTO=LL,LL128,Simple`) con el P2P PIX+ReBAR
  nuevos.
- **Re-perfilar primero**: el 31,8% GPU en AR corresponde a la era sin P2P
  (CNS). Con PIX+ReBAR el número puede ser muy distinto.
- En decode los AR son chicos (cientos KB) → régimen ideal para one-shot
  custom AR, bloqueado por el bug §10.2. Si se confirma la causa raíz, un
  parche Genesis podría guardar el bug (fallback a PYNCCL solo para buffers
  VMM, o registrar via cuMemExportToShareableHandle) y recuperar el custom AR
  completo.

### 12.4 Sequence parallelism forzado (experimento) (impacto: volumen comm prefill ÷2; esfuerzo: medio)

Forzar `pass_config.enable_sp=True` + `sp_min_token_num` manual (la puerta
sm_86 no existe en `SP_MIN_HIDDEN_SIZE`). Convierte AR→RS+AG vía pynccl.
Requiere validar interacción con cudagraph capture sizes (hay helper
`update_sizes_for_sequence_parallelism`) y con spec-decode.

### 12.5 Higiene (sin ganancia de perf, reduce ruido)

Quitar `--moe-backend flashinfer_trtllm`, `VLLM_USE_FUSED_MOE_GROUPED_TOPK=1`,
`--enable-flashinfer-autotune` (no-op sm_86) y auditar/quitar
`VLLM_MARLIN_USE_ATOMIC_ADD`. Documentar en el compose POR QUÉ (evita que
alguien los "arregle" después).

### 12.6 Lo que NO conviene tocar

- **Marlin W8A16 ya es óptimo en Ampere** (Cutlass block-FP8 sm_89+, DeepGEMM/
  FlashInfer blockscale sm_90+, Machete sm_90+, W8A8 imposible sin tensor
  cores FP8).
- **KV fp8_e4m3 en FlashInfer FA2** ya corre nativo con dequant in-kernel.
- **W4A16** (requantizar): única palanca GEMM superior teórica; contradice la
  decisión de calidad del proyecto. Fuera de alcance.

---

## 13. Inconsistencias detectadas y pendientes

### 13.1 En el compose (documento, no código)

1. **Bloque 1600 vs 832 contradictorio**: la nota grande del quantum dice
   "EL QUANTUM ES block_size (1600)" con pisos en múltiplos de 1600; la nota
   del 19-08 justifica 1664 como "múltiplo exacto del bloque de 832" y usa
   `--long-prefill-token-threshold 832`. El código muestra que ese número es
   calculado (`attn_block_size`, §9.3) → uno de los dos comentarios quedó
   viejo. Aclarar cuál es el valor vigente y por qué cambió.
2. **Tabla de capacidad L2 desactualizada**: dice "stride 8 ← config actual
   (123k tokens)" pero el env real es `STRIDE=16` (elegido en el barrido del
   23-08, 4,92 GB al SSD).
3. **`version: '3.8'`** obsoleto en Compose v2 (warning inofensivo).
4. **`pip install pandas scipy xxhash` en cada boot**: dependencia de red y
   arranque no-reproducible (PyPI caído = contenedor no arranca). Hornear en
   imagen derivada o vendorizar wheels.
5. **Sin healthcheck** (vigilancia externa solamente — decisión consciente,
   pero documentarla como tal en un solo lugar).
6. **README público desfasado**: badge pin 0.20.2rc1.dev9 + ledger v7.72 vs
   realidad local v0.23.0 + parches PN88-PN100+.

### 13.2 Pendientes de investigación (abiertos)

- [ ] Confirmar experimentalmente la hipótesis del custom AR (§10.2): correr
      con cumem OFF + custom AR ON en engine sin offloading.
- [ ] Auditar formas reales (N,K) de cada Linear con TP=2 para cerrar el
      veredicto sobre `VLLM_MARLIN_USE_ATOMIC_ADD`.
- [ ] Re-perfil de all-reduce con P2P PIX+ReBAR (¿cuánto queda del 31,8%?).
- [ ] Medir el costo real del repack Marlin en boot (baseline para §12.1).
- [ ] Experimento controlado de BKV_LIST ampliado en `chunk_o` (§12.2).
- [ ] Verificar qué tensores exactamente quedan BF16 vs FP8 en el checkpoint
      (407/1606) y cruzar con `ignored_layers` de Fp8Config.
- [ ] PN77 (LM head fp8): confirmar cómo interactúa con el hecho de que
      stock v0.23 trata lm_head como embedding no-cuantizable (§8.3).

---

## 14. Apéndice: referencias archivo:línea clave

Todas relativas a `assets/vllm/` salvo indicación contraria.

```
Plugins vLLM
  vllm/plugins/__init__.py:14,28,69          grupos, carga, load_general_plugins
  docs/design/plugin_system.md               mecanismo entry points

Dispatch modelo
  model_executor/models/qwen3_5.py:81,155-171,433,529,554   denso vs MoE, clases
  model_executor/models/qwen3_5_mtp.py:59-159,377-388,455   MTP head
  model_executor/models/config.py:350-394                    mamba cache mode
  model_executor/models/registry.py:569-572,638-641,1409-1411

Atención FlashInfer
  v1/attention/backends/flashinfer.py:327-341,391-410,766-825,962-964,
    1175-1193,1255-1262,1578-1585,1745-1752,1863-1953
  utils/flashinfer.py:350-363,388-469,896                    TRTLLM gates

GDN / FLA Triton
  model_executor/layers/fla/ops/utils.py:25-31,151-163,167-200   CHUNK_SIZE, envs, shared_mem
  fla/ops/chunk_delta_h.py:32-41,43,318-380                      fwd_h + configs
  fla/ops/chunk_o.py:21-42                                       BKV_LIST
  fla/ops/{cumsum,chunk_scaled_dot_kkt,solve_tril,wy_fast}.py    pipeline prefill
  fla/ops/fused_sigmoid_gating.py:24,110-130                     decode in-place
  fla/ops/fused_recurrent.py:198-199,438-439                     warps/stages fijos
  ops/causal_conv1d.py:468,749,1069                              conv state
  models/qwen_gdn_linear_attn.py:150-211,290-416,1068-1195,1532  backend resolve, warmup, write state

FP8 → Marlin
  model_executor/layers/quantization/fp8.py:129-134,176-208,284-298,330-337,
    374-392,430-476
  model_executor/kernels/linear/__init__.py:308-329,347-357,450,522-568
  quantization/scaled_mm/marlin.py:36-60,70-102,104-123
  quantization/utils/marlin_utils_fp8.py:24-25,28-39,42-89,92-198,174-187,322-336
  quantization/utils/marlin_utils.py:31,36,167-199,420-429,469-489
  csrc/quantization/marlin/marlin.cu:511-520,533   marlin_template.h:1730,1901-1934,2001-2023
  csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_entry.cu:144-173
  quantization/mixed_precision/machete.py:26-36
  vocab_parallel_embedding.py:49-75,270-274,510    logits_processor.py:75-104

Sampler / warmup / scheduling
  v1/sample/ops/topk_topp_sampler.py:21-67,86-95,131-153,166-174,352-393,435-457,460-497
  model_executor/warmup/kernel_warmup.py:35-52,67-74,109-112,115-191
  config/speculative.py:40-58,550-557     config/compilation.py:1473-1518
  gpu_model_runner.py:598-603,631-633,689-697,814,4487,4974,5030,5240-5243,
    6525-6590,6807-6816,6860-6866         gpu/cudagraph_utils.py:134-137
  config/scheduler.py:146-170             config/vllm.py:805-845,933-997,1004-1024,
    1170-1199,1545,1735

Distribuidos / all-reduce / cumem
  distributed/device_communicators/cuda_communicator.py:100-108,183-244,246-303
  distributed/device_communicators/pynccl.py:166-197
  distributed/device_communicators/custom_all_reduce.py:104,147-154,171-192,
    194-226,253,262-278
  distributed/device_communicators/all_reduce_utils.py:52-71,89-95,100-134
  distributed/device_communicators/symm_mem.py:65-71,105-110
  distributed/device_communicators/pynccl_allocator.py:21-37,48-50,68-103,
    129-191,141,162-164
  csrc/custom_all_reduce.cuh:22-30,384-399,434,442-460,527-546
  csrc/libtorch_stable/custom_all_reduce.cu:135-155,157-189
  device_allocator/cumem.py:56-77,109,259-267,300-301
  csrc/cumem_allocator.cpp:95-193,130,145,187,299,323-328,430
  v1/worker/gpu_worker.py:134-136,202-223,349-356,577,592
  compilation/passes/fusion/sequence_parallelism.py:40-52,55-101,133-141,150-183
  envs.py:86,113,163,233,254,588-595,1097-1098,1360-1361,1760-1761,1867-1868

Mamba page size
  platforms/interface.py:610-699          v1/kv_cache_interface.py:616-625
  v1/worker/mamba_utils.py:108-116,212-234,263-335,536+

Repo (fuera de assets/)
  vllm/_genesis/dispatcher.py:65,2636,2733,3068,3138
  vllm/_genesis/patches/apply_all.py:5-11,182,5255,5713
  vllm/_genesis/wiring/text_patch.py (529 líneas) / rebind.py
  tools/genesis_vllm_plugin/pyproject.toml + genesis_v7/__init__.py:22
  compose/docker-compose.qwen38-27b-fp8.yml (730 líneas)
```

---

## Documentos relacionados

- **`CIRCUITO-TOKEN.md`** — trazado función-por-función del circuito del token
  (entrada→forward→salida) con verificación dinámica en laboratorio
  (`lab/`): inventario completo de cruces GPU⇄CPU/RAM, evidencia de profiler
  sobre el modelo real y veredicto de minimización.

*Fin del documento base. Próximos pasos sugeridos: §13.2. Este archivo debe
actualizarse (no reemplazarse) a medida que avance la investigación.*
