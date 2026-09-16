# Plan A/B por parche — matriz inteligente PN110 + B2/B3/B5/B7

> **Objetivo**: medir ganancia aislada de cada parche que hoy está en PROD,
> apagando **uno por uno** y registrando `t/s`, **PP tok/s** (W1), VRAM y KV disponible.
> **W1 se reporta en PP tok/s = 31792 / wall (2 prompts × 15896 tok), no en wall.** `wall` y `tps` (gen) se mantienen solo como referencia histórica.
> Sin ejecutar tests — solo plan.

| | |
|---|---|
| **Creado** | 2026-08-25 |
| **Modelo activo** | `orcarouter/Qwen3.8-27B-Uncensored-FP8` (TP=2, 2×RTX 3090 sm86, MTP K=3) |
| **Compose PROD** | `compose/docker-compose.qwen38-27b-fp8.yml` |
| **Parches bajo test** | `PN110`, `B2`, `B3`, `B5`, `B7` (los 5 que cambian performance) |
| **Ignorado** | `PN109` (neutro verificado A/B 10/50 conc; `GENESIS_ENABLE_PN109_*=0`) |
| **Suite prefill** | `workshop/ox_alpha/lab/scripts/trace_real.py` (`LAB_PROMPTS=2 LAB_PROMPT_LEN=8000`) |
| **Suite concurrencia** | `workshop/ox_alpha/tests/suite_concurrencia.py` (`N=1` y `N=8`, `max_tokens=500`) |
| **Salida** | `workshop/ox_alpha/results/ab_por_parche/` (CSV + markdown) |
| **Docs hermanos** | `PLAN-CHECKPOINTS.md` §CK-2.x/CK-4.x · `KERNELS-OPTIMIZACION.md` §2-3 · `CONTEXTO-INVESTIGACION.md` §10 |

---

## 1. Estado PROD actual (baseline)

Todos activos (`=1`) en `compose/docker-compose.qwen38-27b-fp8.yml:286-293`:

```yaml
- GENESIS_ENABLE_PN110_INT8_PHASE_DISPATCH=1  # PN110: requant FP8→INT8 W8A8 swap 1:1, ~-23% wall prefill 2×8000 (7.21s→5.56s, 17.8→23.0 t/s en lab)
- GENESIS_ENABLE_B3_CUSTOM_AR=1               # B3: Custom all-reduce TP=2 fast path P2P PIX + fallback robusto (scope cumem×captura)
- GENESIS_ENABLE_B2_FULL_CG=1                 # B2: FULL cudagraphs para drafter MTP (fuerza FULL en prefill largo)
- GENESIS_ENABLE_B5_REJECTION_SAMPLER=1       # B5: softmax solo en posiciones rechazadas (rejection_sampler)
- GENESIS_ENABLE_B7_LM_HEAD=1                 # B7: lm_head W8A16 (extensión PN77, vocab 248k)
# dormidos (no tocar): PN109=0, PN106=0, P61B/P62/PN51/PN66=1 (corrección, no perf), PN8=0, PN92 compress_on_write=0
```

> Verificación baseline: `docker exec genesis-27b-qwen38-fp8 env | grep GENESIS_ENABLE_` +
> `docker logs --tail 80 genesis-27b-qwen38-fp8 | grep -E "PN110|B2|B3|B5|B7|Genesis.*applied"`

---

## 2. Orden inteligente (de menor a mayor impacto)

### 2.1 Orden prescrito — acumulativo

Cada paso **apaga un parche adicional** sobre el anterior. El `delta_tps` del
paso N aísla el parche quitado en ese paso (ganancia = `tps_{N-1} - tps_{N}`).

| Paso | Etiqueta | Qué cambia vs paso anterior | Env editado | Estado tras el paso |
|:--:|---|---|---|---|
| **0** | `baseline` | — (todo prendido) | — | PN110=1 B3=1 B2=1 B7=1 B5=1 |
| **1** | `sin-B5` | **quita B5** | `GENESIS_ENABLE_B5_REJECTION_SAMPLER=0` | B5=0 resto 1 |
| **2** | `sin-B5-B7` | **quita B7** | `GENESIS_ENABLE_B7_LM_HEAD=0` | B5=0 B7=0 resto 1 |
| **3** | `sin-B5-B7-B2` | **quita B2** | `GENESIS_ENABLE_B2_FULL_CG=0` | B5/B7/B2=0, PN110/B3=1 |
| **4** | `sin-B5-B7-B2-B3` | **quita B3** | `GENESIS_ENABLE_B3_CUSTOM_AR=0` | solo PN110=1 |
| **5** | `sin-todos` | **quita PN110** | `GENESIS_ENABLE_PN110_INT8_PHASE_DISPATCH=0` | todo 0 → **solo Marlin W8A16** (decoder PYNCCL, PIECEWISE CG, lm_head fp16 cuBLAS) |

Secuencia exacta solicitada: `baseline → B5 → B7 → B2 → B3 → PN110`.

### 2.2 Justificación: ¿por qué de menor a mayor?

Principio: **minimizar ruido cuando la señal es chica, maximizar señal cuando el drift es grande**.

Ordenado por impacto esperado (bench empírico `KERNELS-OPTIMIZACION.md` + A/B PN110):

| Parche | Ganancia esperada | Dónde duele si falla | Tamaño señal |
|---|---|---|---|
| **B5** | ~0.5–0.75 ms/paso solo en *rejected* tokens (`softmax fp32 [160,248k]` → condicionado) | `vllm/v1/sample/ops/rejection_sampler` | **Chica** (~1–2% wall, solo visible en `N=8` con MTP) |
| **B7** | ~1 ms/paso decode (lm_head vocab-paralelo 124k×5120/rank) | `vocab_parallel_embedding` / `ParallelLMHead` | **Pequeña-mediana** (~2–3% decode) |
| **B2** | FULL CG retenido en prefill largo → evita caída ~30% cuando B2 está OFF y hay prefill 8k en el batch | `speculator.py:84` + `llm_base_proposer.py:386` (`FULL→FULL_DECODE_ONLY`/`PIECEWISE`) | **Mediana** (5–10% single-stream, nula si no hay prefill) |
| **B3** | elimina fallback PYNCCL → latencia AR en TP=2 (PIX P2P). En era CNS era 31.8% del paso; con P2P+ReBAR es 2–5% pero aún el mayor tras GEMM | `cuda_communicator.py:246` + `custom_all_reduce.cuh:455` | **Mediana-grande** (comunicación) |
| **PN110** | **3× GEMM prefill** (cutlass INT8 vs Marlin): medido `-23% wall` (7.21→5.56s) `+29% tps` (17.8→23.0) en 2×8000 tok | `quantization/patch_PN110` (`b_col` column-major swap) | **Grande** (dominante prefill) |

Razones para ir **B5→B7→B2→B3→PN110** y no al revés:

1. **Sensibilidad estadística**: B5/B7 necesitan detectar deltas de 1–3% entre
   ruido inter-boot de ±10% (`ab_pn108` mostró 195→227 t/s solo por varianza).
   Se miden primero, con el servidor más frío y menos deriva térmica/VRAM
   fragmentation. Si se midieran después de quitar PN110 (-23% wall), su señal
   quedaría enterrada bajo el escalón grande y el error de reordenamiento.

2. **Aislamiento causal (ABL acumulativo)**: cada paso es *circuito por circuito*
   (sampler → lm_head → cudagraph → comm → GEMM), sin mezclar dominios. Si se
   empezara por PN110, los pasos siguientes medirían Marlin+PYNCCL con GEMM ya
   degradado — no se sabría si B2/B3 rinden solo con INT8 o también con Marlin.

3. **Riesgo de OOM ordenado**: PN110 es el único que toca VRAM de pesos
   (`b_col` swap 1:1, `torch.nn.Parameter`, `del state["b_col"]`). Apagarlo al
   final devuelve a Marlin estable; apagarlo primero haría que B3/B2 se midieran
   sobre un estado transitorio de `empty_cache` post-swap.

4. **Fail-fast invertido**: los parches chicos son los más baratos de revertir
   si cuelgan (B5/B7 no tocan allocator). Dejar los dos con riesgo de hang
   (`B3 custom AR cih:455 invalid argument` y `PN110 OOM`) al final permite
   abortar la matriz sin haber perdido la mitad de los datos.

> **Si se quisiera la señal máxima primero** (variante no elegida): `PN110→B3→B2→B7→B5`
> maximizaría el `delta_tps` del primer paso, pero sacrifica resolución de los
> parches chicos y contamina todos los pasos siguientes con el fallback
> Marlin+PYNCCL. No se usa.

---

## 3. Workload por A/B (dos harnesses complementarios)

Cada paso (0–5) corre **ambos** workloads, en este orden, sin reiniciar entre ellos.

### 3.1 W1 — Prefill pesado (`trace_real.py`) — mide **PP + TG + PN110**

Replica `lab/scripts/run_ab_pn110.sh` pero contra PROD (no lab). Prefill-dominado.

- **Comando**:
  ```bash
  # Desde el host, contra PROD en 8320 (si usas trace_real directo, lanzar dentro del contenedor lab con TP=2):
  LAB_PROMPTS=2 LAB_TOKENS=64 LAB_PROMPT_LEN=8000 \
  LAB_GPU_MEMORY_UTILIZATION=0.80 \
  python3 workshop/ox_alpha/lab/scripts/trace_real.py 2>&1 | tee /tmp/trace_real_${TAG}.log

  # Alternativa HTTP equivalente (produce el mismo PP largo, mide wall vía curl):
  # 2 prompts de ~8000 tokens cada uno → ~16000 tok prefill + 64 gen
  # El filler de trace_real.py es: "Analiza el siguiente problema..." repetido reps=plen*12/len(filler)
  ```

- **Qué registra** (W1 ahora en **PP tok/s**, `wall`/`tps` solo referencia):
  - `pp_tps` **(métrica primaria W1)** — `31792 / wall` (prompt tokens/s; 2 prompts × 15896 tok / wall). Ej: wall 20.67s → 1538.1 PP tok/s. Reportar siempre `pp_tps`; `wall` queda como diagnóstico.
  - `wall` (s) — `GEN_DONE wall=7.21s` (línea `GEN_DONE requests=2 tokens=128 wall=… tps=…`) — **referencia**, no métrica principal.
  - `tps` (gen) — `tokens / wall` (decode t/s; prefill está dentro de wall) — **referencia**, no comparable entre W1/W2. Para W1 el TG puro es `completion_tokens / (wall - TTFT)` si hay streaming; en `trace_real` TG ≈ `tps`.
  - Para separar PP/TG con streaming usar `bench_decode_tpot_clean_ab.py` como contraste (ver §3.3) o `prompt_tokens`/`completion_tokens` de `usage`.

- **Métrica VRAM/KV en esta pasada**: capturar en paralelo:
  ```bash
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits  # vram_used / vram_free (MiB)
  docker logs genesis-27b-qwen38-fp8 --since 2m | grep -i "Available KV cache memory" | tail -1  # kv_cache
  docker logs genesis-27b-qwen38-fp8 --since 2m | grep -i "Gpu cache usage" | tail -1            # gpu_memory_utilization efectiva
  ```

### 3.2 W2 — Concurrencia (`suite_concurrencia.py`) — mide **t/s + p50/p90**

Mide throughput agregado y colas con prompt tetris (no cacheable por nonce).

- **Comando**:
  ```bash
  python3 workshop/ox_alpha/tests/suite_concurrencia.py \
    --base-url http://127.0.0.1:8320 \
    --model qwen3.8 \
    --levels 1 8 \
    --max-tokens 500 \
    --timeout 60 2>&1 | tee /tmp/suite_concurrencia_${TAG}.log

  # W2a: N=1 (single-stream, baseline 88 t/s histórico)
  # W2b: N=8 (saturación, mide scaling; 5 seqs daban 308 t/s, 10 daban 509 agg)
  ```
  Payload por request: `NONCE_PREFIX + PROMPT_TETRIS` (≈ tetris 500 tok gen,
  temp 0.7, `enable_thinking: false`, `timeout 30–60s`).

- **Qué registra** (por nivel N, dos filas por paso):
  - `wall` (s) — pared del nivel (`wall=…s` en `print_level_report`)
  - `tps_agg` — `total_completion_tokens / wall` (agregado, primary metric; `suite_concurrencia.py:213`)
  - `p50` / `p90` (s) — latencia por-request (`percentile(lats_ok, 50/90)`, `suite_concurrencia.py:215-217`)
  - `hung` — `bool` si timeout/OOM
  - `total_prompt_tokens` / `total_completion_tokens` (para auditar)

### 3.3 Desglose PP vs TG (opcional pero recomendado)

`trace_real.py` no separa PP/TG con streaming. Para PP puro, añadir una pasada
de `tools/bench_decode_tpot_clean_ab.py` (metodología thc1006) que sí separa:

```bash
python3 tools/bench_decode_tpot_clean_ab.py --host 127.0.0.1 --port 8320 \
  --arm-name ${TAG} --runs 10 --max-tokens 64 --prompts standard --out /tmp/tpot_${TAG}.json
# Métricas: TTFT_ms (PP), decode_TPOT_ms (TG), wall_TPS
```

Si el tiempo no da, PP/TG se puede dejar como `wall` (PP-dominado) y `tps_agg`
(TG-dominado) de W1/W2 respectivamente.

### 3.4 Tabla de workloads por paso

| Workload | Prompts | Tokens por req | Concurrencia | Métricas primarias | Señal esperada por parche |
|---|---|---|---|---|---|
| W1 `trace_real` | 2 | 8000 prompt + 64 gen | 2 (secuencial en LLM) | **`pp_tps` (31792/wall)**, `wall` (ref), `tps` (ref), kv_cache | PN110 ≫ B2 > B3 > B7 > B5 |
| W2a `concurrencia N=1` | 1 | ~tetris → 500 gen | 1 | `wall`, `tps_agg`, `p50`, `p90` | B2 (FULL CG) > B3 > B7 > B5, PN110 ≈ neutro en decode |
| W2b `concurrencia N=8` | 8 | 500 gen | 8 | `wall`, `tps_agg`, `p50`, `p90` | B3 > B2 > B5 > B7 > PN110 |

Cada paso completo ≈ 2–3 min (W1 ~8–15s + W2 N=1 ~15s + W2 N=8 ~40–60s + overhead).

---

## 4. Formato de registro

### 4.1 Directorio

```
workshop/ox_alpha/results/ab_por_parche/
├── ab_por_parche.csv          # tabla maestra (una fila por paso×workload)
├── ab_por_parche.md           # tabla markdown legible (generada del CSV)
├── w1_trace_real/             # logs crudos W1 por TAG
│   ├── baseline.log
│   ├── sin-B5.log
│   ├── sin-B5-B7.log
│   ├── sin-B5-B7-B2.log
│   ├── sin-B5-B7-B2-B3.log
│   └── sin-todos.log
├── w2_concurrencia/           # logs crudos W2 por TAG
│   ├── baseline_N1.json
│   ├── baseline_N8.json
│   └── ...
└── vram_kv/                   # snapshots nvidia-smi + Available KV
    ├── baseline_vram.txt
    └── ...
```

### 4.2 CSV maestro — columnas exigidas

Una fila por **paso × workload** (3 filas por paso ⇒ 18 filas + header).
Columnas exactas solicitadas + `workload`/`n` para desambiguar:

```csv
paso,parche_quitado,workload,n,wall,tps,pp_tps,p50,p90,vram_used,vram_free,kv_cache,delta_tps,hung,notas
0,baseline,W1_trace_real,2,7.21,17.8,4409.0,,,18200,5800,344155,,false,PN110+B3+B2+B5+B7 todo ON; pp_tps=31792/wall
0,baseline,W2_concurrencia,1,12.3,85.2,,6.10,8.40,18200,5800,344155,,false,
0,baseline,W2_concurrencia,8,28.5,142.0,,22.1,27.8,18200,5800,344155,,false,
1,B5,W1_trace_real,2,7.25,17.6,4384.6,,,18205,5795,344155,-0.2,false,GENESIS_ENABLE_B5_REJECTION_SAMPLER=0; pp_tps=31792/wall
1,B5,W2_concurrencia,1,12.5,84.0,,6.20,8.55,18205,5795,344155,-1.2,false,
1,B5,W2_concurrencia,8,29.0,139.5,,22.8,28.5,18205,5795,344155,-2.5,false,
...
5,PN110,W1_trace_real,2,7.21,17.8,4409.0,,,18400,5600,344155,-5.2,false,GENESIS_ENABLE_PN110_INT8_PHASE_DISPATCH=0 → solo Marlin; pp_tps=31792/wall
```

> **W1 en PP tok/s**: `pp_tps = 31792 / wall` (2 prompts × 15896 tok; constante de `trace_real.py` con `LAB_PROMPT_LEN=8000`). `wall` y `tps` (gen) se conservan como referencia pero la métrica primaria de W1 es `pp_tps`. Para W2 `pp_tps` queda vacío (no aplica).

| Columna | Unidad | Origen | Descripción |
|---|---|---|---|
| `paso` | int 0–5 | plan | índice de la matriz |
| `parche_quitado` | str | plan | parche aislado en ese paso (o `baseline`) |
| `workload` | str | harness | `W1_trace_real` / `W2_concurrencia` |
| `n` | int | harness | concurrencia (`2` para W1, `1`/`8` para W2) |
| `wall` | s | `GEN_DONE wall` (W1) / `rep["wall"]` (W2) | tiempo de pared del workload (**referencia** en W1; ver `pp_tps`) |
| `tps` | tok/s | `tps` (W1) / `tps_agg` (W2) | throughput de generación (W2: agregado; en W1 es **referencia**, 128 gen tok / wall) |
| `pp_tps` | tok/s | `31792 / wall` (W1) | **prompt throughput W1** (2×15896 prompt tok / wall); vacío en W2 |
| `p50` | s | `rep["p50"]` (W2) | latencia mediana por request (vacío en W1) |
| `p90` | s | `rep["p90"]` (W2) | latencia p90 (vacío en W1) |
| `vram_used` | MiB | `nvidia-smi memory.used` | VRAM usada por GPU (tomar gpu 0; si difiere, promedio) |
| `vram_free` | MiB | `nvidia-smi memory.free` | VRAM libre |
| `kv_cache` | tokens | `docker logs \| grep "Available KV cache memory"` | KV disponible reportado por vLLM en boot (`344155` típico con `gpu_memory_utilization=0.745`, `max_num_seqs=10`) |
| `delta_tps` | tok/s | `tps_paso - tps_previo` (mismo workload/n) | ganancia aislada del parche quitado (negativo = el parche mejoraba) |
| `hung` | bool | `rep["hung"]` / timeout | `true` si timeout/OOM/cuelgue |
| `notas` | str | manual | env cambiado, `finish_reason`, anomalías |

> **Convención `delta_tps`**: `delta_tps = tps_{paso-1} - tps_{paso}` para el
> mismo `(workload, n)`. Si `delta_tps < -2` el parche **beneficiaba** (quitarlo
> empeora). `pct = delta_tps / tps_{paso-1} * 100`. Guardar también `delta_pct`
> en `notas` o columna extra si se quiere.

### 4.3 Markdown de resumen

Generado del CSV (no editar a mano). Ejemplo:

```markdown
| Paso | Parche quitado | Workload | N | wall (s) | tps (gen) | pp_tps | p50 (s) | p90 (s) | VRAM used/free (MiB) | KV cache | Δtps vs prev | Δ% | Estado |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | baseline | W1 | 2 | 7.21 | 17.8 | 4409.0 | — | — | 18200 / 5800 | 344155 | — | — | ✅ |
| 0 | baseline | W2 | 1 | 12.3 | 85.2 | — | 6.10 | 8.40 | 18200 / 5800 | 344155 | — | — | ✅ |
| 0 | baseline | W2 | 8 | 28.5 | 142.0 | — | 22.1 | 27.8 | 18200 / 5800 | 344155 | — | — | ✅ |
| 1 | B5 | W1 | 2 | 7.25 | 17.6 | 4384.6 | — | — | 18205 / 5795 | 344155 | -0.2 | -1.1% | ✅ |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 5 | PN110 | W1 | 2 | 9.40 | 12.6 | 3381.7 | — | — | 18400 / 5600 | 344155 | -5.2 | -29% | ✅ |
```

> **W1**: `pp_tps` es primaria (`31792/wall`); `wall` y `tps` (gen) son referencia. Ej. real baseline W1 `wall 20.67 → pp_tps 1538.1`.

Añadir al pie: `delta_tps` y `delta_pct` por workload, `baseline_tps` de
referencia (W1 17–23 t/s gen / ~1300–1550 PP tok/s, W2 N=1 80–95 t/s esperado, N=8 ~140–200 t/s).

---

## 5. Procedimiento para apagar un parche (paso a paso)

> **Regla de oro** (`PLAN-CHECKPOINTS.md:277`): *un cambio por A/B*. Nunca dos
> flags en el mismo reboot.

### 5.1 Editar el compose

Archivo: `compose/docker-compose.qwen38-27b-fp8.yml` (líneas 286–293).

Editar **in-place** con `edit` (no `bash` heredoc). Ejemplo para B5:

```diff
-      - GENESIS_ENABLE_B5_REJECTION_SAMPLER=1
+      - GENESIS_ENABLE_B5_REJECTION_SAMPLER=0
```

Flags por parche:

| Parche | Flag a poner en `0` |
|---|---|
| B5 | `GENESIS_ENABLE_B5_REJECTION_SAMPLER=0` |
| B7 | `GENESIS_ENABLE_B7_LM_HEAD=0` |
| B2 | `GENESIS_ENABLE_B2_FULL_CG=0` |
| B3 | `GENESIS_ENABLE_B3_CUSTOM_AR=0` |
| PN110 | `GENESIS_ENABLE_PN110_INT8_PHASE_DISPATCH=0` |

> **No tocar** `GENESIS_DISABLE_PN110`/`GENESIS_DISABLE_B3` (legacy). Usar
> siempre `GENESIS_ENABLE_*=0`. PN109 ya está `0`; no modificar.

Verificar con `read` que el archivo quedó con el valor `0` y que el resto de
flags siguen en `1`.

### 5.2 Recrear el contenedor

```bash
# Desde la raíz del repo
docker compose -f compose/docker-compose.qwen38-27b-fp8.yml up -d
# Equivale a: recreate genesis-27b-qwen38-fp8 con el nuevo env
```

`restart: no` a propósito (`compose:47-55`) — `up -d` es la forma de recrear;
`docker restart` **no** relee el YAML, solo reinicia el proceso viejo.

### 5.3 Esperar arranque

```bash
# 1) Seguir logs hasta "Application startup complete"
docker logs -f genesis-27b-qwen38-fp8 2>&1 | grep -E "Application startup complete|ERROR|Traceback|Available KV cache memory|Gpu cache usage"
# Señal OK: "Application startup complete." + "Available KV cache memory: 344155 tokens" (o similar)

# 2) Health poll (30s máx, sin sleeps largos)
for i in {1..15}; do
  curl -sf http://127.0.0.1:8320/health -H "Authorization: Bearer $VLLM_API_KEY" && break
  curl -sf http://127.0.0.1:8320/v1/models -H "Authorization: Bearer $VLLM_API_KEY" && break
  sleep 2
done

# 3) Verificar que el flag quedó aplicado (boot summary)
docker logs genesis-27b-qwen38-fp8 --since 2m | grep -E "GENESIS.*B5|GENESIS.*B7|GENESIS.*B2|GENESIS.*B3|GENESIS.*PN110|applied|skipped"
# Debe decir "skipped: B5 ... opt-in only" y "applied: PN110" según corresponda
```

Criterios de abort por paso:

| Síntoma en logs | Culpable | Acción |
|---|---|---|
| `OutOfMemoryError` / `OOM` / `_allocate_kv_cache` / `out of memory` | PN110 (swap) o VRAM | bajar `gpu_memory_utilization` o abortar matriz |
| `Cuda error csrc/custom_all_reduce.cuh:455 'invalid argument'` / `Engine core initialization failed` | B3 (cumem×IPC) | dejar B3=0 y anotar hang |
| `Traceback` en `rejection_sampler` / `cutlass_scaled_mm` / `int8` / `marlin` | B5 / PN110 | log + `hung=true` |
| `TIMEOUT 30s` en `suite_concurrencia` | saturación | `hung=true`, `docker logs --since 5m \| grep ERROR` |

### 5.4 Capturar VRAM y KV cache

```bash
# ANTES del workload (estado estable post-boot)
nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv,noheader,nounits | tee /tmp/vram_${TAG}_pre.txt
docker logs genesis-27b-qwen38-fp8 --since 2m | grep -i "Available KV cache memory" | tail -1 | tee /tmp/kv_${TAG}_pre.txt
# Disponible también vía métricas: curl -s http://127.0.0.1:8320/metrics | grep -E "kv_cache|gpu_cache"
```

Repetir **después** del workload para detectar leak/fragmentación.

### 5.5 Correr el workload y volcar métricas

```bash
TAG=sin-B5  # ejemplo paso 1
mkdir -p workshop/ox_alpha/results/ab_por_parche/w1_trace_real
mkdir -p workshop/ox_alpha/results/ab_por_parche/w2_concurrencia
mkdir -p workshop/ox_alpha/results/ab_por_parche/vram_kv

# W1
LAB_PROMPTS=2 LAB_TOKENS=64 LAB_PROMPT_LEN=8000 \
  python3 workshop/ox_alpha/lab/scripts/trace_real.py \
  2>&1 | tee workshop/ox_alpha/results/ab_por_parche/w1_trace_real/${TAG}.log
# Extraer: grep -E "GEN_DONE|prompt_len|wall|tps" w1_trace_real/${TAG}.log

# W2
python3 workshop/ox_alpha/tests/suite_concurrencia.py \
  --base-url http://127.0.0.1:8320 --model qwen3.8 --levels 1 8 --max-tokens 500 --timeout 60 \
  2>&1 | tee workshop/ox_alpha/results/ab_por_parche/w2_concurrencia/${TAG}.log
# Extraer: grep -E "N=|wall=|tps_agg|p50|p90|hung" w2_concurrencia/${TAG}.log

# VRAM/KV snapshot post
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits | tee workshop/ox_alpha/results/ab_por_parche/vram_kv/${TAG}_post.txt
docker logs genesis-27b-qwen38-fp8 --since 5m | grep -i "Available KV" | tail -1 | tee workshop/ox_alpha/results/ab_por_parche/vram_kv/${TAG}_kv.txt
```

### 5.6 Registrar en el CSV

Añadir 3 filas (W1, W2 N=1, W2 N=8) al `ab_por_parche.csv` con `wall/tps/pp_tps/p50/p90/vram_used/vram_free/kv_cache` y `delta_tps` vs paso anterior (mismo workload). Para W1 calcular `pp_tps = 31792 / wall` (2×15896) como métrica primaria; `wall`/`tps` quedan como referencia. En W2 `pp_tps` vacío.

Ejemplo de extracción para W2:

```bash
# wall/tps/p50/p90 están en print_level_report (suite_concurrencia.py:265-295)
# vram_used/vram_free vienen de nvidia-smi (col 1=cuda:0 used, col2=free)
# kv_cache: docker logs | grep -oP "Available KV cache memory:\s+\K[0-9]+"
```

### 5.7 Repetir para el siguiente paso

Volver a §5.1 con el siguiente flag en `0` (acumulativo). No re-prender parches
entre pasos; al final del paso 5 restaurar el compose a `=1` para volver a PROD:

```bash
# Restauración final (fuera de la matriz, no es un paso medido)
# Editar compose de vuelta a GENESIS_ENABLE_B5/B7/B2/B3/PN110=1, luego:
docker compose -f compose/docker-compose.qwen38-27b-fp8.yml up -d
# Verificar: docker logs ... | grep "Application startup complete"
```

---

## 6. Matriz completa — checklist operativo

| # | TAG | Comando edit | Up | Health | W1 (2×8000) | W2 (1/8×500) | VRAM/KV | CSV |
|---|---|---|---|---|---|---|---|---|
| 0 | `baseline` | verificar todo `=1` | `up -d` | `curl /health` | `trace_real` | `suite_concurrencia` | `nvidia-smi` + `Available KV` | fila 0 |
| 1 | `sin-B5` | `B5=0` | `up -d` | `health` | `trace_real` | `suite_concurrencia` | snapshot | fila 1 |
| 2 | `sin-B5-B7` | `B7=0` | `up -d` | `health` | `trace_real` | `suite_concurrencia` | snapshot | fila 2 |
| 3 | `sin-B5-B7-B2` | `B2=0` | `up -d` | `health` | `trace_real` | `suite_concurrencia` | snapshot | fila 3 |
| 4 | `sin-B5-B7-B2-B3` | `B3=0` | `up -d` | `health` | `trace_real` | `suite_concurrencia` | snapshot | fila 4 |
| 5 | `sin-todos` | `PN110=0` | `up -d` | `health` | `trace_real` | `suite_concurrencia` | snapshot | fila 5 |

Tiempo estimado matriz completa: **6 reboots × ~40s boot + 6×~2.5 min workload ≈ 25–30 min**.

---

## 7. Controles de validez (para minimizar ruido)

- **Temperatura y throttling**: `nvidia-smi -q -d TEMPERATURE,CLOCK` antes de cada paso; si >85 °C esperar 60s.
- **Cooldown entre workloads**: 10–15s entre W1 y W2 (dejar que se drenen `pending` de PN90).
- **Misma `gpu_memory_utilization`**: fijar `0.80` para W1 (como en `run_ab_pn110.sh`) si se usa `trace_real` directo; PROD usa `0.745` en HTTP — documentar cuál se usó en `notas`.
- **Mismo seed/temperatura**: `temperature 0.7`, `LAB_TOKENS=64` (W1) / `max_tokens=500` (W2), `seed` fijo si se usa `bench_decode_tpot_clean_ab.py`.
- **No mezclar métricas**: `tps` de W1 (prefill) no es comparable con `tps_agg` de W2 (decode concurrencia); nunca promediar entre workloads.
- **Varianza inter-boot**: repetir baseline al inicio y al final si el tiempo lo permite; reportar `±10%` como banda de ruido conocida (`ab_pn108` 195→227 t/s).
- **PN109 neutro**: verificar que sigue `0` en todos los pasos (`grep PN109` en logs debe ser `skipped`).

---

## 8. Plantilla de CSV inicial (vacía, lista para llenar)

Se deja `workshop/ox_alpha/results/ab_por_parche/ab_por_parche.csv` con header y sin filas (W1 en `pp_tps`):

```csv
paso,parche_quitado,workload,n,wall,tps,pp_tps,p50,p90,vram_used,vram_free,kv_cache,delta_tps,hung,notas
```

Y `ab_por_parche.md` con la tabla vacía y leyenda. El operador la llena
tras cada paso (ver §4.2). `W1` se registra con `pp_tps = 31792 / wall` como métrica principal.

---

## 9. Referencias

- `compose/docker-compose.qwen38-27b-fp8.yml:286-293` — flags activos PROD
- `workshop/ox_alpha/lab/scripts/trace_real.py:49-82` — `LAB_PROMPTS/LAB_TOKENS/LAB_PROMPT_LEN` y `GEN_DONE`
- `workshop/ox_alpha/lab/scripts/run_ab_pn110.sh:13-22` — workload 2×8000 prefill canónico PN110
- `workshop/ox_alpha/tests/suite_concurrencia.py:158-262` — `run_level` / `tps_agg` / `p50/p90` / `hung`
- `workshop/ox_alpha/KERNELS-OPTIMIZACION.md:50-59` — 3× prefill INT8 vs Marlin
- `workshop/ox_alpha/KERNELS-OPTIMIZACION.md:81-94` — sampler 0.75 ms, GDN 0.44 ms/capa
- `workshop/ox_alpha/CONTEXTO-INVESTIGACION.md:10` — bug custom AR `cuh:455 invalid argument` (B3)
- `workshop/ox_alpha/PLAN-CHECKPOINTS.md:232-253` — CK-4.1/4.2/4.3/4.4 y PN110 gate

---

## 10. Entregable de esta tarea

Este archivo (`workshop/ox_alpha/PLAN_AB_POR_PARCHE.md`) es el plan completo.
No se ejecutaron workloads ni se modificó el compose. Para ejecutar la matriz,
seguir §5 paso a paso y volcar en `workshop/ox_alpha/results/ab_por_parche/`.
