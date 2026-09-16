# Logros — Qwen3.8-27B en 2× RTX 3090 con vLLM + Genesis

Estado al **2026-09-14**. Todo lo que hay acá está medido en este equipo; donde algo es estimación, se dice.

Snapshot exacto del compose en producción: [`docs/logros-compose-produccion-2026-09-14.yml`](logros-compose-produccion-2026-09-14.yml).
Scripts de medición: [`tests/bench/medicion/`](../tests/bench/medicion/).

---

## 1. Resultado en producción

| Métrica | Valor | Cómo se mide |
|---|---|---|
| Prefill (PP), prompt frío de 42.790 tokens | **2.239-2.258 tok/s** | `pp.py <tag>` |
| Decode, prompts cortos (8 prompts mixtos) | **93-98 tok/s**, TAR 63-65% | `tar_multi.py <tag>` |
| Decode, código (TAR alto) | 101-115 tok/s | `tar_multi.py` / tareas de código |
| Decode con 57k de contexto | **122 tok/s** | `largo.py <tag>` |
| Recuperación en contexto largo (12 agujas en ~57k) | **12/12** | `largo.py` |
| Suite de calidad (10 tareas con respuesta única) | **10/10** | `suite_calidad.py` |
| Preguntas de control (6) | 6/6 en todos los arranques | `pocas.py <tag>` |
| Pool de KV en GPU | **353.324 tokens** | log `GPU KV cache size` |
| Bloques GDN por request (23k de contexto) | 3-4 por grupo (antes 5-7) | `bloques.py` (vía PN115) |
| Preempciones con carga real de opencode (41 min, 4 hilos, KV 97%) | **0** | `crudo.py` |
| Hit de prefix cache L1 con carga real | 83,5% | `crudo.py` |

Referencia de partida de la sesión: prefill ~2.052 tok/s.

---

## 2. Hardware y base

- 2× RTX 3090 24 GB (SM86, Ampere), **capadas a 220 W**, PCIe 4.0 x8, topología PIX, P2P habilitado.
- 30 GB de RAM. **Premisa del proyecto: no se compra hardware; se optimiza.**
- Imagen `vllm/vllm-openai:v0.27.1`, `cpuset: "0-7,16-23"`, `shm_size: 16gb`, `ipc: host`.
- Modelo: `Ar4ikov/Qwen3.8-27B-Uncensored-AWQ-W4A16-ASYM` (compressed-tensors, AWQ W4A16 asimétrico g128).
  - 64 capas: **48 GDN** (linear attention) + **16 de atención completa** (índices 3, 7, 11, …, 63).
  - Atención: 24 cabezas de query, 4 KV, head_dim 256 → por rank con TP=2: 12 q / 2 kv.
  - GDN: 16 k heads, 48 v heads, head 128, conv kernel 4.
  - MTP nativo (1 capa) usado con K=3.

---

## 3. Línea de comando de vLLM

```
vllm serve Ar4ikov/Qwen3.8-27B-Uncensored-AWQ-W4A16-ASYM
  --served-model-name qwen3.8 qwen3.6 qwen3.8-27b-uncensored Ar4ikov/Qwen3.8-27B-Uncensored-AWQ-W4A16-ASYM
  --trust-remote-code --quantization compressed-tensors --dtype float16
  --attention-backend FLASHINFER --enable-flashinfer-autotune
  --mamba-ssm-cache-dtype float16
  --kv-cache-dtype fp8_e4m3
  --tensor-parallel-size 2 --max-num-seqs 10
  --max-model-len 262144 --gpu-memory-utilization 0.90
  --enable-prefix-caching --enable-chunked-prefill
  --max-num-batched-tokens 8192 --long-prefill-token-threshold 8192
  --performance-mode throughput
  --cudagraph-capture-sizes 4 8 12 16 20 24 28 32 36 40
  --async-scheduling --scheduling-policy priority
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
      "kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","cpu_bytes_to_use":12884901888,
      "eviction_policy":"arc","store_threshold":1,
      "secondary_tiers":[{"type":"fs","root_dir":"/kv-offload","n_read_threads":4,"n_write_threads":4}]}}'
  --enable-cumem-allocator
  --chat-template /etc/qwen-froggeric-chat-template.jinja
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
  --generation-config vllm --moe-backend flashinfer_trtllm
  --limit-mm-per-prompt '{"image":2,"video":0}'
  --mm-processor-kwargs '{"max_pixels":2000000,"min_pixels":65536}'
  --enable-prompt-tokens-details --disable-access-log-for-endpoints "/metrics,/health"
  --host 0.0.0.0 --port 8320
```

Por qué algunos valores:
- **Bloque de atención de 832 tokens** (no se configura: lo fuerza vLLM para que la página de atención cubra la del estado GDN). Consecuencia: los chunks de prefill salen en múltiplos de 832 → con 8192 de presupuesto el chunk real es **7.488**.
- `fp8_e4m3` + FlashInfer: Triton no acepta fp8 en SM86 y FlashInfer es mucho más rápido en estas placas (ver §7.3).

---

## 4. Parches Genesis que hacen la diferencia (medidos en esta sesión)

| Parche | Qué hace | Medido |
|---|---|---|
| **PN118** INT8 MLP (+ `self_attn`) | Convierte AWQ int4 → INT8 per-canal en la carga; forward por `cutlass_scaled_mm` (instrucción INT8 nativa de Ampere) | 2,01x sobre Marlin W4A16 en el MLP. **Bug crítico arreglado hoy** (§6.1) |
| **PN119** SK-12 | gate_up + SiLU fusionado para prefill grande (M ≥ 5824) | kernel propio; gana por tráfico de memoria |
| **PN120** all-reduce INT8 | Parciales de TP en int8 con escala por grupo de 64, all_gather + suma fp16 | **+4,7-6,5% de prefill**; NCCL estaba al 96% del PCIe; decode intacto |
| **PN121** guard de cascada de preempción | Corta el bucle de preempción cuando los frees quedan diferidos (L2 + async) | Antes: 59 preempciones/min y KV 99%→14% en un paso. Después: **0 en 41 min** |
| **PN122** rollback del MTP en GDN con cinta | Un solo estado GDN + cinta (k norm, v, g, β) de los tokens 1..K en vez de K copias | **−9 bloques GDN por request**; TAR igual; −1,3% pasos/s |
| PN115 | Admisión por headroom de KV + serialización de prefills | + instrumento `bloques_por_request` en `/dev/shm/genesis_pid_status.json` |
| PN114 | Guardas de límites en copias `align` de mamba | necesario con MTP y concurrencia ≥ 4 |
| PN77 / PN108 | lm_head y lm_head del drafter en FP8 | ~1,2 GiB VRAM por GPU |
| PN99 | L2 comprimido a 4 bits en la GPU | 1,78x de capacidad de L2. **Asume KV fp8** (no usar con otros tipos de KV) |
| PN59 / PN50 / PN54 | GDN streaming, proyección fusionada, dedup contiguo | ya estaban |

Variables clave (lista completa en el snapshot del compose):

```
GENESIS_ENABLE_PN118_INT8_MLP=1   GENESIS_PN118_CAPAS=.mlp.,self_attn
GENESIS_ENABLE_PN119_SK12_MLP=1   GENESIS_PN119_M_MIN=5824
GENESIS_ENABLE_PN120_AR_INT8=1    GENESIS_PN120_M_MIN=512   GENESIS_PN120_GRUPO=64
GENESIS_ENABLE_PN121_PREEMPT_GUARD=1
GENESIS_ENABLE_PN122_GDN_CINTA=1
GENESIS_ENABLE_PN115_PID_GATING=1 GENESIS_PN115_KV_GATING=1 GENESIS_PN115_MAX_CONCURRENT_PREFILLS=1
GENESIS_ENABLE_PN99_GPU_COMPRESSED_L2=1
GENESIS_ENABLE_PN77_FP8_LM_HEAD=1 GENESIS_ENABLE_PN108_DRAFT_FP8_LM_HEAD=1
GENESIS_ENABLE_PN114_MAMBA_ALIGN_BOUNDS_GUARD=1
GENESIS_ENABLE_PN93_SPARSE_GDN=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
```

Diagnóstico disponible (apagado en producción):
- `GENESIS_PN122_DEBUG=1` + bits en `/dev/shm/pn122_sync` (8 log del builder, 32 verificación en sombra contra upstream, 512 kernel de upstream en caliente). `GENESIS_PN122_SIN_LIBERAR=1` conserva los bloques especulativos.
- `GENESIS_ENABLE_PN123_VOLCADO_QKV=1`: vuelca q/k/v reales de la atención a `/kv-offload/pn123_qkv/`.

---

## 5. Procedimiento operativo (lo que hay que hacer para llegar y mantenerse acá)

1. **Cualquier cambio en un parche que toque código del modelo** (qwen3_next, qwen_gdn_linear_attn, linear.py…) exige borrar la caché de torch.compile. Los archivos son de root:
   ```
   docker compose -f docker-compose.qwen38-27b-ar4ikov-awq.yml down
   docker run --rm -v /home/usuario/.cache/vllm:/c alpine rm -rf /c/qwen38-27b-ar4ikov-awq/torch_compile_cache
   docker compose -f docker-compose.qwen38-27b-ar4ikov-awq.yml up -d
   ```
   Con caché vieja el parche queda inerte o el arranque falla (`object has no attribute _genesis_int8_mlp`).
2. Reiniciar siempre con `--force-recreate` (con `up -d` solo, se reusa estado viejo).
3. **Calentar con un prompt largo** antes de cualquier request corto (`sanidad.py`): un primer request corto con MTP + CUDA graphs puede tumbar el engine.
4. **Validar en varios arranques** (≥3) cualquier A/B de calidad: hubo una falla que aparecía en ~1 de cada 2 arranques (§6.1).
5. Benchmarks contra el servidor compartido: `tar_multi.py` marca **CONTAMINADO** si otro request generó tokens en paralelo. Comparar **pasos/s** (tok/s ÷ tok/paso), no tok/s solo.
6. `/dev/shm` acumula segmentos `psm_*` de vLLMs muertos: limpiarlos (vía contenedor) cuando falte RAM, **no** justo antes de arrancar (dispara `Bad address` en el registro de L2).
7. Un custom op sin salida y con `mutates_args=[]` es código muerto para Inductor: se borra del grafo.

---

## 6. Bugs encontrados y arreglados

### 6.1 PN118 mezclaba los pesos de los dos ranks (el más grave)
- Síntoma: con temperatura 0, respuestas mal (847×23 → 19681, 7! → 5208) en ~50% de los arranques, idénticas dentro de cada arranque.
- Descartado con varios arranques cada uno: PN122, L2, SK-12, async scheduling, modo de CUDA graphs, INT8 en atención, especializaciones de Triton.
- Causa: la caché de conversión INT8 usaba `os.environ["RANK"]`, que los workers de vLLM no tienen → los dos ranks con `rank=0` y la misma clave (los fragmentos de TP tienen igual forma) → un rank cargaba el MLP del otro.
- Arreglo: `get_tensor_model_parallel_rank()` + huella del contenido en la clave. Verificado 4/4 arranques y 320 entradas en la caché.

### 6.2 Cascada de preempciones (PN121)
Con L2 + async, `defer_block_free` devuelve los bloques un paso después; el bucle de preempción seguía matando requests hasta vaciar la KV.

### 6.3 PN122 (tres bugs antes de llegar a producción)
- El conv spec usa `spec_state_indices.size(-1)` como `max_query_len` → con una columna procesaba mal (expandir a K+1).
- Slots de cinta que no se liberaban (el engine murió a los 21 requests) → liberar por `requests` vivos.
- Escribir la cinta dentro del kernel de recurrencia daba carrera entre programas que comparten cabeza k → lanzamiento aparte.

### 6.4 PN120 horneado en el grafo
El gate por tamaño dentro del forward trazado se horneaba (`evaluate_guards: False`) y ralentizaba el decode 12% → custom op opaco con helpers compilados aparte.

---

## 7. Investigaciones con resultado negativo (no repetir)

### 7.1 KV comprimida con diccionario / codebook
q/k/v reales (capas 3 y 35, 22k tokens de código), error en la **salida de la atención**:

| método | bits | error | top-8 |
|---|---|---|---|
| fp8 escala 1 (hoy) | 8 | 2,1-2,6% | 95-96% |
| int8 por token-cabeza | 8,1 | 0,5-0,6% | 99% |
| int4 + Hadamard | 4,1 | 6-8% | 87-90% |
| PQ / RVQ 4 bits | 4,1 | 7-8% | 87-90% |
| PQ / RVQ 2 bits | 2,1 | 22-26% | 65-73% |
| Disperso tipo Lexico | 1,9 / 3,8 | 32-44% / 15-22% | — |

Los vectores son casi isotrópicos: el diccionario no tiene estructura que explotar. Script: `tests/proto/kv_diccionario_eval.py`.

### 7.2 `int8_per_token_head` (Triton) contra fp8 (FlashInfer)
| | PP | decode@57k | agujas | suite |
|---|---|---|---|---|
| FlashInfer + fp8 | 2.255 | 122,4 | 12/12 | 10/10 |
| Triton + int8 por token-cabeza | 1.296 | 28,4 | 12/12 | 10/10 |
| Triton + fp16 | 1.361 | 45,6 | 12/12 | — |

### 7.3 Por qué Triton pierde en Ampere (base del próximo parche)
- **Prefill:** `BLOCK_Q = 16 // 6 = 2` queries por programa y tiles de 32; el afinado para head 256 es solo Blackwell y no entra en la SRAM de Ampere (155.648 > 101.376). `BLOCK_M=64, TILE=64, stages=1` → 29,1 ms vs 82,4 ms (2,84x), **igual que FlashAttention paginada (28,4 ms)**.
- **Decode con MTP:** el kernel 3D paralelo exige 1 query por request; con K=3 (4 queries) cae al 2D serial. Con 1 query @57k: **2D 3,40 ms vs 3D 0,31 ms (10,8x)**.
- Scripts: `tests/proto/triton_attn_bench.py`, `triton_attn_bench2.py`.

### 7.4 Otros
- Async TP + sequence parallelism: 3-4% peor. Fusión allreduce+RMSNorm de FlashInfer: sin cambio. Pipeline parallel: no soportado. DBO: solo MoE.
- TurboQuant: 10-68% más lento según el blog de vLLM; 3 bits pierde hasta ~20 puntos.

---

## 8. PN124 — Triton rápido en Ampere + KV int4 (listo, apagado en producción)

Parche `GENESIS_ENABLE_PN124_TRITON_AMPERE=1` (`vllm/_genesis/triton_attn_ampere.py`, wiring `wiring/hybrid/patch_PN124_triton_ampere.py`):
1. Prefill: parámetros de lanzamiento para SM8x + head 256. Kernel fp16/int8: `BLOCK_M=64, TILE=64, stages=1, warps=8` (2,84x). Kernel INT4 empaquetado: `BLOCK_M=32, TILE=32, stages=1, warps=4` (1,35x; 64/64 lo hace 2x más lento). Si un kernel no entra en la SRAM de Ampere, baja un escalón una sola vez.
2. Decode con MTP: aplana N requests × Q queries en N×Q pseudo-secuencias de 1 query para usar el kernel 3D paralelo. Verificado igual al 2D (error 3e-4 en fp16), 2,6-8,8x.
3. Tile de decode del kernel INT4 en 3D: 32 en vez de 16 (0,78 → 0,40 ms @57k).

Configuración para usarlo (en lugar de la de producción):
```
--attention-backend TRITON_ATTN
--kv-cache-dtype int4_per_token_head
GENESIS_ENABLE_PN124_TRITON_AMPERE=1
GENESIS_ENABLE_PN99_GPU_COMPRESSED_L2=0   # PN99 asume KV fp8
```
(borrar la caché de torch.compile al cambiar).

| | FlashInfer + fp8 (producción) | Triton + int8 sin parche | **Triton + int4 + PN124** |
|---|---|---|---|
| Pool de KV | 353.324 | 368.440 | **705.772 (2,0x)** |
| Bloque de atención | 832 | 816 | 1.616 |
| PP 42.790 tokens | **2.255** | 1.296 | 1.656 (−27%) |
| Decode @57k | **122,4** | 28,4 | 96,6 (−21%) |
| TAR / tok/s (8 prompts cortos) | 62,7% / 95 | 64,4% / 98 | 65,7% / 97 |
| Agujas @57k / suite | 12/12 · 10/10 | 12/12 · 10/10 | 12/12 · 10/10 |
| Decode de 3.000 tokens cruzando bordes | ok | — | ok |
| Prefix cache en turno 2 (3.681 tokens) | 2.496 | — | 1.616 |

Scripts: `tests/proto/pn124_aplanar_test.py`, `pn124_int4_barrido.py`, `pn124_int4_decode.py`.

---

## 9. Los 8 puntos de dispersión / cuantización con tensor cores (2026-09-14)

Métrica de calidad: 160 sondeos teacher-forced hasta 100k tokens (`tests/bench/medicion/sondeos.py`), contra fp16. fp16: logp del token real −1,261. KV fp8: 98,1% top-1, KL 0,0031.

| Variante (solo MLP, simulada en PN118) | top-1 | KL | logp real | suite |
|---|---|---|---|---|
| W4A4 g64 sin rotación | 93,1% | 0,0352 | −1,481 | 8/10 |
| W4A4 g64 + Hadamard por bloque | 94,4% | 0,0206 | −1,368 | 10/10 |
| **W4A4 g64 + WUSH** (calibrado 75k tokens) | 94,4% | **0,0165** | −1,369 | 10/10 |
| W4A4 + WUSH **solo gate_up** (down en W8A8) | 96,2% | 0,0118 | −1,264 | — |
| W4A4 + WUSH solo down_proj (gate_up en W8A8) | 95,0% | 0,0104 | −1,282 | — |
| solo A4 + Hadamard / solo W4 + Hadamard | 97,5% / 95,6% | 0,0114 / 0,0139 | −1,264 / −1,268 | — |
| P5: pesos 2:4 por magnitud (sin SparseGPT) | 86,2% | 0,1878 | −1,749 | 7/10 |
| P6: activaciones 8:16 estilo Amber | 93,1% | 0,0604 | −1,509 | 8/10 |
| P7: activaciones 2:4 | 88,8% | 0,1249 | −1,648 | 8/10 |

- **P2 permutación de canales:** exacta (1e-14) pero no ayuda: int4 g64 9,42→9,43%, magnitud retenida 2:4 74,98→75,16%.
- **P5 en kernel:** `ptxas` conoce "Sparse mma" (`.sp::ordered_metadata`) pero lo rechaza como *Illegal modifier* en sm_86, sm_89 y sm_120. No hay mma disperso nativo: el 2:4 no acelera en esta toolchain y además la calidad no alcanza.
- **WUSH (arXiv 2512.00956):** en q·k de atención (volcados PN123) baja la MSE de logits ~12% pero no la salida (int4 g64: 6,01→5,87% capa 3, 8,58→8,92% capa 35). En MLP W4A4 baja el KL 20% pero el logp no se mueve. La asignación es cruzada (el peso usa la T del momento de la activación). Código: `vllm/_genesis/wush_mlp.py` (`GENESIS_PN118_WUSH=captura:<dir>` con `--enforce-eager`, luego `aplicar:<dir>`), `tests/proto/wush_qk_eval.py`.
- Simulación: `GENESIS_PN118_FAKE=w4a4|w4|a4`, `GENESIS_PN118_FAKE_ROT=bloque`, `GENESIS_PN118_FAKE_DISP=w24|a816|a24`, `GENESIS_PN118_FAKE_SOLO=<subcadena>` (borrar la caché de torch.compile en cada cambio).

---

## 10. W4A8 con Marlin sobre noon-at-cgn (2026-09-14)

Checkpoint `noon-at-cgn/Qwen3.8-27B-Uncensored-W4A16-AutoRound` (base orcarouter, int4 simétrico g128, MTP conservado). Compose `compose/docker-compose.qwen38-27b-noon-w4a8.yml`: sin PN118/PN119, `VLLM_MARLIN_INPUT_DTYPE=int8`, `GENESIS_ENABLE_PN125_MARLIN_W4A8_ESCALAS=1`, KV fp8 FlashInfer. Resultados en `tests/bench/medicion/resultados_w4a8/`.

| | Ar4ikov producción (W8A8 PN118) | noon W4A16 | noon W4A8 sin PN125 | **noon W4A8 + PN125** |
|---|---|---|---|---|
| PP 42k (tok/s) | 2.127–2.236 | 1.550 | (basura) | **2.193** |
| Decode @57k | 121 | 148 | — | **146** |
| TAR / tok/s cortos | 63,5% / 95 | 67,7% / 129 | — | 64,3% / **119** |
| Pool KV | 352k | 625k | 622k | **617k** |
| pocas / suite / agujas | 6/6 · 10/10 · 12/12 | 6/6 · 10/10 · 12/12 | 0/6 · 0/10 · 0/12 | **6/6 · 10/10 · 12/12** |
| Sondeos | — | logp −1,239 | — | KL 0,016 contra W4A16 (98,1% top-1) |

**PN125:** Marlin W4A8-INT8 exige escalas ≥ 0 y AutoRound deja el 50,5% negativas → salida "!!!!" sin aviso. PN125 las corrige en la carga (s→|s|, q→16−q, escala óptima donde aparece q=0): error de peso 2,6%. Microtests: `tests/proto/marlin_w4a8_capa.py`, `marlin_w4a8_piezas.py`, `marlin_w4a8_fix.py`, `marlin_w4a8_pn125.py`.

---

## 11. Rotaciones, KV y kernels PTX W4A4 (2026-09-15)

**PN126 (rotación q/k después de RoPE, `GENESIS_ENABLE_PN126_ROT_QK=1`, `GENESIS_PN126_ROT=hadamard|captura:<dir>|wush:<dir>`):** exacta (0,0001%), gratis en velocidad con KV fp8. Offline baja el error de la KV cuantizada (fp8 capa 3: 1,88→1,33%; int4 capa 35: 8,3→6,1% con WUSH). En sondeos NO se distingue: el piso de ruido entre corridas idénticas es KL 0,017–0,019.

**noon W4A8 + PN125 según KV** (resultados en `tests/bench/medicion/resultados_rot/` y `resultados_noche/`):
| KV | PP 42k | Decode @57k | Pool KV | calidad |
|---|---|---|---|---|
| fp8 FlashInfer | 2.193 | 146 | 617k | 6/6 · 10/10 · 12/12 |
| int8 Triton+PN124 | 1.858 | 101 | 629k | igual |
| int4 Triton+PN124 | 1.633 | 113 | **1.207k** | igual (KL en el ruido) |

**Peso de la atención en el prefill (noon W4A8):** 10% a 16k, 18% a 32k, 30% a 64k, 40% a 100k.

**W4A4 gate_up (simulado, fp16 logp −1,261):** g256 Hadamard 256 KL 0,0187 · g1024 0,0212 · solo activaciones g256 0,0212 · por token/fila 0,029 (Hadamard 8192: 0,025, densa 5120: 0,036). Hace falta g256.

**Kernels PTX** (`vllm/_genesis/kernels/`), forma real M=7488 K=5120 N=8704, GPU libre:
| Kernel | Qué | Estado |
|---|---|---|
| SK-14 s4 | gate_up+SiLU W4A4 por token/fila | 4,4–6,5 ms (cutlass W8A8 9,6) — calidad insuficiente |
| SK-15 / SK-15b | ídem con escalas g256 | exacto; 85 → **42 ms** con escalas en shared. Techo: ~1,3 ms por volcado del acumulador (lineal) |
| SK-16 | GEMM fp16 (rotaciones) | exacto; Hadamard bloques 0,58 ms vs cuBLAS 0,40 |
| SK-17 + Q8b + Q4 | Hadamard INT8 exacta + int8→int4 | bit-exacto; 3,7 ms a M=7488 |

Conclusión: W4A4 por grupo en PTX no le gana a cutlass en sm_86. La mejora disponible es W4A8 Marlin + PN125; la siguiente palanca de PP es un kernel de atención INT8 (30–40% del prefill a 60–100k).

---

## 12. Pendientes conocidos

- `prompt_logprobs` devuelve basura en todas las posiciones del prompt (la generación está bien): algún parche rompe los logits de posiciones no sampleadas.
- PN115 no frena a los agentes de opencode con prioridad negativa (coach, primary, coder, planner, verifier).
- Prefix cache en prompts largos: de un turno de 3.681 tokens se reusaron 2.496.
- **Próximo:** parche de Triton (parámetros de prefill para SM86 + decode spec aplanado al kernel 3D) para habilitar KV `int4_per_token_head` (~1,95x) sin perder velocidad.
