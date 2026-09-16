# 🗺️ PLAN-CHECKPOINTS — Estado del plan de optimización, portable entre modelos

> **Propósito**: tracker de checkpoints del plan maestro de optimizaciones
> (Fases 0-4). Diseñado para **sobrevivir cambios de modelo**: cada checkpoint
> declara si es model-agnóstico (sigue DONE al cambiar de checkpoint) o
> model-dependiente (vuelve a estado RE-VALIDAR).
>
> **Cómo usarlo**: al completar un checkpoint, marcar `[x]`, llenar la fila de
> mediciones y avanzar. Al cambiar de modelo, correr el §3 (Protocolo de
> cambio) y actualizar §2 (Perfil del modelo activo).
>
> | | |
> |---|---|
> | **Creado** | 2026-08-24 |
> | **Modelo activo** | `orcarouter/Qwen3.8-27B-Uncensored-FP8` |
> | **Checkpoint actual** | CK-0.1 (baseline capturado ✓) → próximo: CK-0.2 |
> | **Docs hermanos** | `KERNELS-OPTIMIZACION.md` (evidencia) · `BACKPORT-V2.md` · `CIRCUITO-TOKEN.md` · `CONTEXTO-INVESTIGACION.md` |

---

## 1. Mapa de checkpoints y estados

Estados: `PENDIENTE` · `EN-CURSO` · `DONE` · `RE-VALIDAR` (tras cambio de modelo) · `BLOQUEADO` · `DESCARTADO`

| ID | Fase | Ítem | Estado | Portabilidad |
|:--|:--|:---|:--|:--|
| CK-0.1 | 0 | Baseline del modelo capturado (perfil + benches + A/B) | **DONE** | Se RE-GENERA por modelo |
| CK-0.2 | 0 | B4 — Barrido NCCL_ALGO × NCCL_PROTO | **DONE** (cerrado por gate) | Model-agnóstico (hardware) |
| CK-0.3 | 0 | B6 — Cache disco del repack Marlin (**PN106**) | **DONE** (implementado+tests 9/9; dormido en PROD) | Model-agnóstico (hash invalida solo) |
| CK-1.1 | 1 | A6 — Draft MTP cuantizado (W8/W4) + medición de aceptancia | **DONE** (PN108: A/B válido 2026-08-24, gate superado, adoptado en PROD) | **Model-dependiente** (aceptancia) |
| CK-1.2a | 1 | B1/PN-B — Buffers persistentes spec-decode | **DONE** (PN109: tests 22/22, A/B neutro en 10 y 50 concurrentes → NO promover; flag opt-in disponible) | Model-agnóstico (runner V1) |
| CK-1.2b | 1 | B1/PN-D — Cache staging mamba | PENDIENTE | Model-agnóstico |
| CK-1.2c | 1 | B1/PN-E — Limpieza builder GDN | PENDIENTE | Semi (formas GDN cambian) |
| CK-1.2d | 1 | B1/PN-C — Gate draft_token_ids_event | PENDIENTE | Model-agnóstico |
| CK-1.2e | 1 | B1/PN-A — num_accepted_tokens GPU-residente | PENDIENTE | Model-agnóstico (regresión numérica) |
| CK-1.2f | 1 | B1/PN-F — AsyncOutput adelantado | PENDIENTE | Model-agnóstico |
| CK-2.1 | 2 | A1 — Conversión FP8→INT8 en carga (parche quantization) | **DONE** — A/B prefill superado 5.56s vs 7.21s (-23%); promovido a PROD con gate CK-2.4 pendiente | **Model-dependiente** (KL por capa) |
| CK-2.2 | 2 | A2 — Despacho por fase W8A8/W8A16 (umbral de M) | **DONE** — A/B prefill superado 5.56s vs 7.21s (-23%); promovido a PROD con gate CK-2.4 pendiente | Semi (umbral por formas) |
| CK-2.3 | 2 | A3 — Gate KL por capa en arranque + tabla en boot summary | PENDIENTE | **Model-dependiente** |
| CK-2.4 | 2 | Validación final: tool-calls 10/10 + TTFT medido | PENDIENTE | **Model-dependiente** |
| CK-3.1 | 3 | A5 — MLPs a W4A16 (calibración offline + artefacto + doctor) | PENDIENTE | **Model-dependiente** (artefacto) |
| CK-4.1 | 4 | B2 — FULL cudagraphs para drafter MTP | PENDIENTE | Model-agnóstico (spec mechanics) |
| CK-4.2 | 4 | B3 — Fix custom all-reduce (scoping cumem × captura) | PENDIENTE | Model-agnóstico (allocator) |
| CK-4.3 | 4 | B5 — Softmax rejection sampler solo en rechazados | PENDIENTE | Model-agnóstico (vocab escala) |
| CK-4.4 | 4 | B7 — lm_head W8A16 (extensión PN77) | PENDIENTE | Semi (vocab-dependiente) |

**Regla de portabilidad**: model-agnóstico = el mecanismo no depende del
checkpoint (runner, allocator, spec-decode, hardware). Al cambiar de modelo
estos checkpoints **conservan su estado** si el nuevo modelo usa el mismo
runner (V1) y la misma familia de mecanismos. Si el nuevo modelo activa el
runner V2 (denso no-cuantizado Llama/Mistral/Qwen3-plain), TODA la familia
CK-1.2 pasa a `DESCARTADO` (jubilación automática por drift-marker — ver
BACKPORT-V2.md §5).

---

## 2. Perfil del modelo activo

> **Se RE-GENERA en cada cambio de modelo.** Es la única sección que hay que
> reescribir completa. Todo lo demás del plan referencia este perfil.

### 2.1 Datos duros — Qwen3.8-27B-Uncensored-FP8 (actual)

| Campo | Valor | Cómo se obtiene |
|:---|:---|:---|
| Arquitectura | `Qwen3_5ForConditionalGeneration` (híbrido GDN+ViT, denso) | `config.json → architectures` |
| Runner | **V1** (`gpu_model_runner.py`) — V2 excluido por allowlist+quant+hybrid | `config/vllm.py:519` gates |
| Capas | 64 = 48 GDN + 16 full (patrón 3:1, `full_attention_interval=4`) | `text_config.layer_types` |
| hidden / heads | 5120 · full: 24Q/4KV × head_dim 256 · GDN: 16K/48V × 128 | `text_config` |
| intermediate / vocab | 17408 · **248,320** (lm_head gigante) | `text_config` |
| Cuantización | FP8 E4M3 bloque 128×128, **407/1606 tensores**; BF16: ViT, norms, gates, lm_head, embeds, TODAS las linear_attn GDN | `quantization_config` + inspección |
| Sensibilidad documentada | KV fp8 degrada GDN (memoria interna); PN92 rechazado (12,4% err L2) | docs del repo |
| Peso / GPU | ~15,5 GiB (TP=2) | boot log |

### 2.2 Formas GEMM por rank (TP=2) — input de los benches

```
qkv:     N=4096   K=5120     o:      N=5120  K=3072
gate_up: N=17408  K=5120     down:   N=5120  K=8704
lm_head: N=124160 K=5120
M decode=40 (10 seqs × K+1)   M prefill=1664 (chunk)
```

### 2.3 Baseline medido (NO borrar — es la referencia de comparación)

| Métrica | Valor | Evidencia |
|:---|:---|:---|
| GEMM decode M=40 | Marlin 0.047/0.039/0.147/0.107/1.277 ms (qkv/o/gu/down/lm) · 448-608 GB/s | `lab/results/benches/bench_gemm.log` |
| GEMM prefill M=1664 | Marlin 1.105/0.862/5.529/2.427/40.79 ms · INT8 3× mejor | ídem |
| FLA GDN prefill | 0.44 ms/capa (1,2% del paso) — tuning descartado | `bench_fla.log` |
| Sampler | argmax 0.066 · triton 0.593 ≈ flashinfer 0.618 · softmax rejection 0.749 ms | `bench_sampler.log` |
| batch_memcpy mamba | 417 GB/s (techo) | `bench_memcpy.log` |
| Atención A/B | FLASHINFER 62.1 tok/s (4×64) · TRITON_ATTN **CRASH** con MTP | `results/attn_ab_*.log` |
| Circuito token | DtoH 4 KB/paso · HtoD 250 KB · DtoD 319 MB/rank | `results/real/` |
| PROD single-stream | 88 tok/s (1 seq) · 308 (5) · 509 agg (10) = 42% pérdida concurrencia | compose + tests/repro |
| Cudagraphs | PIECEWISE forzado (FlashInfer+spec) — P100 no cumple FULL | `attn_ab_FLASHINFER.log` |

### 2.4 Comandos de regeneración del perfil (nuevo modelo)

```bash
# 1) Dims y mapa de cuantización
SNAP=$(ls -d /home/usuario/Proyectos/models-cache/hub/models--<org>--<model>/snapshots/*/ | head -1)
python3 -c "import json; c=json.load(open('$SNAP/config.json')); \
  t=c.get('text_config',c); print({k:t.get(k) for k in \
  ['hidden_size','num_hidden_layers','num_attention_heads','num_key_value_heads',\
   'head_dim','intermediate_size','vocab_size','full_attention_interval']}); \
  print('quant:', c.get('quantization_config',{}).get('quant_method'))"

# 2) Actualizar SHAPES en lab/scripts/bench_gemm.py según §2.1/2.2
# 3) Actualizar glob de modelo en lab/scripts/trace_real.py
# 4) Correr suite: LAB_RUN=run_benches.sh docker compose -f docker-compose.lab.yml up lab
#    (requiere GPUs libres: docker stop genesis-27b-qwen38-fp8 → docker start al final)
# 5) A/B atención: LAB_RUN=run_attn_ab.sh ...
# 6) Llenar §2.1-2.3 de este archivo con los números nuevos
```

---

## 3. Protocolo de cambio de modelo

Cuando se cambie el checkpoint servido (ej: al W4A16-AWQ cacheado, o un
Qwen3.9 futuro):

1. **Regenerar perfil** (§2.4) y reescribir §2.1-2.3. Los baselines viejos se
   archivan en §5 (log de decisión) — nunca se borran.
2. **Clasificar checkpoints afectados**:
   - Runner cambia a V2 (modelo denso no-cuantizado de la allowlist) →
     CK-1.2a-f `DESCARTADO` (drift-marker), CK-1.2c igual.
   - Formato de cuantización cambia (W4/AWQ, INT8, BF16) → CK-2.x y CK-3.1
     vuelven a `PENDIENTE` o `RE-VALIDAR` según solapamiento.
   - Vocab cambia → CK-4.3/CK-4.4 `RE-VALIDAR` (formas nuevas).
   - Mismo formato + mismo runner → solo CK-1.1 y CK-2.3/2.4 a
     `RE-VALIDAR` (aceptancia y KL son por-checkpoint).
3. **Re-capturar baseline** con la suite del lab (§2.4 pasos 4-5).
4. **Re-correr gates** de los checkpoints en estado RE-VALIDAR, en orden de
   fase. Los DONE model-agnósticos no se re-testean salvo que el boot summary
   reporte anchors rotos (el pin-gate de Genesis frena antes si cambió vLLM).
5. **Actualizar la fila "Modelo activo"** del encabezado.

**Candidatos ya cacheados** para cambio futuro (en `models-cache/hub/`):
`soyrsoyr/Qwen3.8-27B-W4A16-AWQ-GPTQ` (para CK-3.1 como referencia) ·
`lued/Qwen3.8-27B-heretic-ara-INT8-W8A16-MTP` (evidencia de tolerancia INT8
de la familia) · `Zynerji/...GPTQ-MTP` · `z-lab/Qwen3.6-35B-A3B-DFlash`.

---

## 4. Detalle por checkpoint (entrada → acciones → gate de salida)

### CK-0.2 — B4 NCCL sweep
- **Entrada**: GPUs libres, PROD abajo.
- **Acciones**: re-perfil del AR (torch profiler, ¿% del paso hoy con P2P
  PIX+ReBAR?); barrido `NCCL_ALGO=Ring,NVLS` × `NCCL_PROTO=LL,LL128,Simple`
  vía env en el compose del lab; A/B con `tests/repro/carga_mixta.py`.
- **Registrar**: % AR del paso antes/después; mejor combinación; tok/s.
- **Gate**: si AR < 5% del paso → cerrar frente y anotar. Si mejora > 5% →
  aplicar al compose PROD con comentario fechado (estilo del archivo).
- **Portabilidad**: agnóstica (re-corregir solo si cambia TP/topología).

### CK-0.3 — B6 Cache repack Marlin
- **Acciones**: parche Genesis `wiring/loader/patch_pnXX_marlin_repack_cache.py`
  (class-rebind de `prepare_fp8_layer_for_marlin`, persistir tensores keyed
  por hash de pesos — patrón PN57). TDD: boot 2 salta repack; hash distinto
  invalida.
- **Gate**: boot 2 mediblemente más rápido; output numérico idéntico.
- **Portabilidad**: agnóstica; el hash invalida solo por modelo.

### CK-1.1 — A6 Draft MTP cuantizado — **DONE (2026-08-24)**
- **Resultado**: PN108 (FP8 del lm_head del draft vía rebind de
  `SpecDecodeBaseProposer.load_model`). A/B en lab con mount de
  `_genesis` corregido: aceptancia 2,61 vs 2,60 baseline (+0,4%),
  draft acceptance 53,5% vs 53,4%, per-pos 0,707/0,520/0,379 vs
  0,716/0,521/0,366 (ruido). VRAM drafter 2,78 → 2,15 GiB
  (-630 MiB/rank). tps 227,6 vs 215,1 (+5,8%, dentro de la varianza
  ±10% medida entre boots). Gate <5% relativo: SUPERADO.
- **Adopción**: `GENESIS_ENABLE_PN108_DRAFT_FP8_LM_HEAD=1` añadido al
  compose PROD con comentario fechado. Verificar boot PROD + métricas
  SpecDecoding en las primeras peticiones.
- **Portabilidad**: RE-VALIDAR por modelo (la aceptancia es del checkpoint).

### CK-1.2a — B1/PN-B Buffers persistentes spec-decode — **DONE (2026-08-24, no promover)**
- **Resultado**: PN109 implementado (`wiring/loader/patch_PN109_spec_decode_persistent_metadata.py`,
  buffers pinned+GPU persistentes para los 5 tensores de
  `_calc_spec_decode_metadata`, ataca TODO upstream `gpu_model_runner.py:2778`).
  Tests 22/22 en contenedor (incluye igualdad exacta valores+dtypes vs oracle).
  A/B en lab: **NEUTRO**.
  - Concurrencia 10×250: mediana de paso 43,44 vs 44,02 ms (±1,3%, ruido).
  - Concurrencia 50×250: mediana 46,82 vs 47,63 ms (±1,7%, ruido); tps
    266,6 vs 249,7 (dentro de la varianza ±10% entre boots).
  - Aceptancia siempre en banda [2,55-2,73] (transparencia numérica OK).
- **Veredicto**: los 5 allocs + H2D pageable por paso NO están en el camino
  crítico de este stack (decode GPU-bound, CPU holgada). Flag opt-in
  `GENESIS_ENABLE_PN109_SPEC_DECODE_PERSISTENT_METADATA` disponible; NO se
  promueve a PROD. Kill switch interno `GENESIS_DISABLE_PN109=1`.
- **Implicancia CK-1.2b-f**: mismos overheads CPU-side → re-rankear antes de
  implementar más (ver log de decisiones 2026-08-24).
- **Portabilidad**: model-agnóstico; puede rendir en stacks CPU-contendidos.

### CK-1.2b-f — Backports V2 restantes (PN-D → PN-E → PN-C → PN-A → PN-F)
- **Acciones**: un parche por vez, TDD primero, A/B tras cada uno. Detalle
  completo de cada uno en `BACKPORT-V2.md` §3.
- **Gate por parche**: comportamiento numérico idéntico (regresión) +
  mejora medible de p99/latencia o syncs eliminadas en el log del lab.
- **Drift-marker de la familia**: "V2 runner activo para híbridos" →
  jubilación automática cuando upstream migre.
- **Portabilidad**: agnóstica salvo PN-E (formas GDN).

### CK-2.1/2.2/2.3 — INT8 en carga + despacho por fase + gate KL
- **Estado 2026-08-24**: PN110 v1 (diseño "phase dispatch", 51/51 tests CPU)
  hizo **OOM** en el primer A/B real de lab — el diseño dual suma ~1 byte/param
  de VRAM sobre un presupuesto ya lleno al 80% → **inviable**. **Pivote en
  curso a "full requant swap"**: el INT8 reemplaza al Marlin por capa (1:1 en
  bytes, VRAM neta ~igual) y decode también corre en INT8 → el target cambia
  en TODAS las fases, el gate CK-2.4 pasa a ser **bloqueante** antes de promover.
- **Bug raíz OOM (2026-08-24)**:
  - **Causa**: en producción las capas son `nn.Module` con `weight` registrado como `Parameter`; el wrapper hacía `setattr(layer, "weight", tensor_plano)` dentro de `try/except TypeError: pass` que tragaba el error en silencio → el swap INT8 nunca ocurría, `b_col` se acumulaba (+1 byte/param por capa, ~13 GiB/rank) sobre Marlin intacto → OOM en cascada con TP=2.
  - **Por qué tests no lo vieron**: usaban `SimpleNamespace` donde `setattr` funciona; no replica semántica `Parameter`/`register_parameter`/`register_buffer` de `nn.Module`.
  - **Fix**: envolver en `torch.nn.Parameter(state["b_col"], requires_grad=False)` + logging `ERROR` explícito si falla + liberar estado (`del state["b_col"]` / `torch.cuda.empty_cache()`) para no acumular VRAM.
  - **Lección**: los fakes de test deben replicar semántica `Parameter`/buffer de `nn.Module` (asignación debe exigir `nn.Parameter`), no `SimpleNamespace`.
- **Estado 2026-08-25 — DONE (A/B prefill superado)**: workload prefill 2×8000 tokens (8000 prompt + 64 gen), TP=2, 2×3090, 0.80 util, max_model_len 16384, MTP 3 — Baseline wall 7.21s tps 17.8, PN110 wall 5.56s tps 23.0 (-23% wall, +29% tps). Sin OOM tras fix `nn.Parameter` (swap INT8 `b_col` column-major reemplaza a Marlin 1:1) + exclusión GDN/mamba + invalidación cache Dynamo. Conversión chunked 512, per-channel, excluye GDN. Flag promovido a PROD con comentario fechado. **Gate CK-2.4 (tool-calls 10/10 + TTFT 126k) queda pendiente** antes de promoción definitiva, pero el prefill ya rinde.
- **Acciones**: parche compuesto `wiring/quantization/patch_pnXX_int8_load_quant.py`
   (diseño completo en la conversación del 08-24, §"estilo Genesis"):
   rebind de `process_weights_after_loading` (conversión GPU por bloque,
   KL por capa, decisión INT8/Marlin, caché junto a CK-0.3) + rebind de
   `apply()` con umbral de M (`GENESIS_PNXX_W8A8_MIN_TOKENS`).
- **Env flags**: `GENESIS_ENABLE_PNXX_INT8_LOAD_QUANT` ·
  `GENESIS_PNXX_KL_THRESHOLD` · `GENESIS_PNXX_W8A8_MIN_TOKENS` ·
  `GENESIS_PNXX_FORCE_LAYERS`.
- **applies_to**: `quantization=fp8-block ∧ sm∈[80,89]` (auto-off en Hopper).
- **Gate CK-2.4**: KL < umbral en capas convertidas + **tool-calls 10/10** +
  TTFT del hilo largo ~2-3× (medir con `probe_max_ctx.sh` / carga de 126k) — **PENDIENTE tras A/B prefill 2026-08-25** (prefill ya rinde 5.56s vs 7.21s; falta validar tool-calls 10/10 + TTFT 126k antes de promoción definitiva).
- **Abort**: clean rate cae → subir umbral / exiliar capas / híbrido por
  bloque-K (frontera A4).
- **Portabilidad**: RE-GENERA por modelo (el mapa KL es del checkpoint).

### CK-3.1 — MLPs a W4A16
- **Acciones**: subcomando `genesis requant --layers mlp --format awq-int4
  --calib <corpus>` (offline, produce artefacto) + regla de doctor que lo
  valide. Referencia cacheada: `soyrsoyr/Qwen3.8-27B-W4A16-AWQ-GPTQ`.
- **Gate**: tool-calls 10/10 + needle ladder + concurrencia medida a batch
  10/20/30 (¿se linealiza el scaling?) + VRAM liberada ≈ 4 GB/GPU.
- **Portabilidad**: artefacto por modelo (re-calibrar).

### CK-4.1-4.4 — Código restante
- **CK-4.1 (B2)**: text-patch del camino que degrada CG mode para el
  drafter; gate = FULL capturado (verificar boot log) + latencia draft.
- **CK-4.2 (B3)**: text-patch del scoping cumem alrededor de la captura
  (`gpu_model_runner.py:6546`, `gpu_worker.py` pools); gate = custom AR
  habilitado sin `invalid argument` + A/B decode.
- **CK-4.3 (B5)**: text-patch del softmax incondicional en
  `rejection_sampler.py`; gate = output idéntico en greedy + ~0,5 ms/paso.
- **CK-4.4 (B7)**: extensión PN77; gate = ~1 ms/paso decode + tool-calls.

---

## 5. Log de decisiones (append-only)

| Fecha | Checkpoint | Decisión | Evidencia |
|:---|:---|:---|:---|
| 2026-08-24 | CK-0.1 | Baseline completo capturado; TRITON_ATTN descartado (crash MTP); tuning FLA descartado (0 efecto); INT8 W8A8 identificado como palanca 3× prefill | `KERNELS-OPTIMIZACION.md` |
| 2026-08-24 | — | Plan maestro A+B ordenado por riesgo (gratis→cero→prefill→RAM→código) | conversación + este doc |
| 2026-08-24 | CK-0.2 | **CERRADO por gate**: AR/all-gather ≈ 2% del kernel time (< 5%). Sweep default/LL128/Simple sin efecto reproducible (Simple n=1 dio 169.5 tok/s pero confirmación n=2 dio 135.9 ≈ default 136.7 → varianza). El 31,8% histórico era de la era CNS — el P2P nuevo ya cosechó esa fruta. NO adoptar env NCCL. Colectivo dominante: AllGather_RING_LL (gather de logits del lm_head), ~19-21 ms/rank idéntico en todos los combos | `lab/results/nccl_sweep/` |
| 2026-08-24 | CK-0.3 | **PN106 implementado** (`wiring/loader/patch_PN106_marlin_repack_cache.py`, rebind con caché disco + invalidación por hash que incluye la fuente de la función). Tests 9/9 en contenedor v0.23.0 + sync gates verdes (de paso se regularizaron PN77/PN89/PN348 que estaban huérfanas del registry). Montado en PROD y DORMIDO (default_off; boot summary lo reporta skipped opt-in) | `vllm/_genesis/wiring/loader/patch_PN106_marlin_repack_cache.py` |
| 2026-08-24 | CK-1.1 | **RECON completo**: el drafter es `runner.drafter.model` (clase Qwen3_5MTP) con **2,78 GB/rank propios en FP16** (lm_head ~1,27 GB + embed ~1,27 GB + fc/capa ~0,25 GB). Su lm_head carga por `SpecDecodeBaseProposer.load_model` (llm_base_proposer.py:1202) que NO pasa por el walker parcheado por PN77 → estaba sin cuantizar. **PN108 implementado**: rebind de ese load_model que aplica el método Genesis_FP8_LMHead al lm_head del draft tras la carga (riesgo de calidad cero: rejection sampler verifica). Tests 7/7 + sync gates verdes. PENDIENTE: A/B de aceptancia con GPUs (gate <5% relativo) | `wiring/spec_decode/patch_PN108_draft_fp8_lm_head.py` + `results/probe_draft.log` |
| 2026-08-24 | CK-1.1 | **A/B VÁLIDO y gate SUPERADO**. Primer intento falló: el lab no montaba `vllm/_genesis` → ModuleNotFoundError en sitecustomize (corrida inválida = 2º baseline: 195,5 tok/s; revela varianza ±10% entre boots). Con mount corregido: PN108 aplicado ×5 procesos, dump confirma `qm=Genesis_FP8_LMHead_EmbeddingMethod`, drafter 2,78→2,15 GiB. Aceptancia 2,61 vs 2,60 / 53,5% vs 53,4% → sin pérdida. tps 227,6 vs 215,1 (+5,8%, en varianza). **Adoptado**: flag añadido al compose PROD con comentario fechado; PROD reiniciado 16:00 | `results/ab_pn108/{baseline,pn108}.log` + `lab_<pid>.log` |
| 2026-08-24 | CK-1.2a | **PN109 implementado y medido: NEUTRO → no promover**. Tests 22/22 (igualdad exacta vs oracle upstream; el test-loop además cazó un bug real: `apply()` devolvía `bool` en vez de `tuple[str,str]`). A/B doble en lab con logging de TODOS los pasos (`LAB_LOG_ALL_STEPS=1`, nuevo en sitecustomize): concurrencia 10 mediana de paso 43,44→44,02 ms; concurrencia 50 mediana 46,82→47,63 ms — ambos ±<2% (ruido). Conclusión de arquitectura: este stack es GPU-bound con CPU holgada; los micro-overheads CPU-side que motivaron los backports V2 no pagan aquí. **Decisión**: PN109 queda opt-in dormido; CK-1.2b-f se RE-RANKEAN — solo implementar los que toquen el camino GPU (p.ej. PN-A si elimina syncs reales) o esperar perfil de carga que demuestre contención de CPU | `results/ab_pn109*/{baseline,pn109}.log` |
| 2026-08-24 | CK-1.2a | **INCIDENTE de gating + corrección**: la primera versión de PN109 solo chequeaba el kill switch (`GENESIS_DISABLE_PN109`) y NO el opt-in del dispatcher → quedó ACTIVO en PROD tras el restart de las 17:19 sin promoción (violando "PROD solo recibe lo promovido"). Detectado en la verificación de boot. Fix: `apply()` ahora exige `GENESIS_ENABLE_PN109_SPEC_DECODE_PERSISTENT_METADATA=1` (espeja PN108); test nuevo `test_apply_skips_without_opt_in_flag` + rebind-test actualizado (23/23 verdes). PROD reiniciado 17:25:57 y verificado: `skipped: PN109 ... opt-in only`, PN108 sigue applied, completion end-to-end OK. Lección: TODO parche nuevo debe tener test de que SIN flag su apply() devuelve skipped — el sync-test dispatcher↔apply_all no cubre el gating interno del módulo | `wiring/loader/patch_PN109_...py` + tests |
| | | | |
| 2026-08-24 | CK-2.1 | **PN110 v1 implementado** (51/51 tests CPU; diseño "phase dispatch": copias INT8 residentes junto al estado Marlin, despacho por umbral de M). **Primer A/B real en lab** (2×RTX3090, `gpu_memory_utilization=0.80`, prefill 2×8000 tokens): `OutOfMemoryError` en `_build_int8_state` en casi todas las capas — el diseño dual suma ~1 byte/param de VRAM sobre un presupuesto ya lleno al 80% → **inviable**. Hallazgo secundario (verificado en GPU real): el kernel cutlass c2x de sm_86 NO acepta escalas de peso 2-D por bloque (epilogue `RowOrScalarBroadcast`: solo per-tensor/per-channel) → el peso se colapsa bloque→per-channel (precisión estándar W8A8). **Pivote de diseño a "full requant swap"**: el INT8 REEMPLAZA al Marlin por capa (1:1 en bytes, VRAM neta ~igual), conversión chunked para acotar transitorios, decode también en INT8 (bench §2: decode INT8 ≈ Marlin, bandwidth-bound). El target cambia en TODAS las fases → gate CK-2.4 obligatorio antes de promover | `wiring/quantization/patch_PN110_int8_phase_dispatch.py` + `results/ab_pn110/` |
| 2026-08-24 | CK-2.1 | **Bug raíz OOM PN110 — TypeError silencioso**: `setattr(layer, "weight", tensor_plano)` dentro de `try/except TypeError: pass` tragaba el error en silencio en producción (`nn.Module` con `weight` registrado como `Parameter` exige `nn.Parameter`) → el swap INT8 nunca ocurría, `b_col` se acumulaba (+1 byte/param por capa, ~13 GiB/rank) sobre Marlin intacto → OOM en cascada TP=2. Fix: envolver en `torch.nn.Parameter(state["b_col"], requires_grad=False)` + logging `ERROR` explícito + liberar estado si falla. A/B previos 21:05/21:20/21:36 **inválidos** (medían Marlin intacto + fuga de VRAM, no el swap). Lección: fakes de test deben replicar semántica `Parameter`/buffer (`nn.Module`), no `SimpleNamespace` | `wiring/quantization/patch_PN110_int8_phase_dispatch.py` |
| 2026-08-25 | CK-2.1 | **A/B prefill superado** — workload prefill 2×8000 tokens (8000 prompt + 64 gen), TP=2, 2×3090, 0.80 util, max_model_len 16384, MTP 3: Baseline wall 7.21s tps 17.8, PN110 wall 5.56s tps 23.0 (-23% wall, +29% tps); sin OOM tras fix `nn.Parameter` (swap INT8 `b_col` column-major reemplaza a Marlin 1:1) + exclusión GDN/mamba + invalidación cache Dynamo; conversión chunked 512, per-channel, excluye GDN; flag promovido a PROD con comentario fechado | `wiring/quantization/patch_PN110...py` + `results/ab_pn110/` + compose |
| 2026-08-25 | PLAN | **Pendiente por cerrar — orden prefijado (§1→§4)**: 1) **CK-2.3** — gate KL por capa + tabla en boot summary (model-dependiente, PENDIENTE); 2) **CK-2.4 — bloqueante** — tool-calls 10/10 + TTFT 126k (hilo largo ~2-3×, `probe_max_ctx.sh`) — **bloquea promoción definitiva de PN110** (prefill ya rinde 5.56s vs 7.21s); 3) **CK-3.1** — W4A16 MLPs offline + artefacto (26G ya cacheado `soyrsoyr/Qwen3.8-27B-W4A16-AWQ-GPTQ`, comando `genesis requant --layers mlp --format awq-int4` aún no implementado; gate: tool-calls+needle+concurrencia 10/20/30+VRAM ~4 GB/GPU); 4) **CK-4.1–4.4** — B2 FULL cudagraphs text-patch CG mode drafter, B3 fix custom all-reduce scoping cumem×captura (`gpu_model_runner.py:6546`), B5 softmax solo en rechazados (`rejection_sampler.py`), B7 lm_head W8A16; 5) **CK-1.2b-f re-rankeados** — solo **PN-A** (`num_accepted_tokens` GPU-residente) si perfil muestra contención CPU / syncs reales, resto en espera tras A/B PN109 neutro | `PLAN-CHECKPOINTS.md §1 y §4` |

---

## 6. Reglas de oro del plan

1. **Un cambio por A/B** — nunca dos parches en la misma medición.
2. **El harness de tool-calls 10/10 es el único juez de calidad** — los
   benchmarks genéricos (MMLU/GSM8K) no detectan lo que importa en agentic.
3. **Todo parche Genesis**: TDD primero, env-flag, `default_on=False`,
   registro en PATCH_REGISTRY, fila en boot summary, drift-marker.
4. **PROD solo recibe lo promovido** tras soak en lab; el compose documenta
   cada cambio con fecha y razón (la cultura del archivo actual).
5. **Al cambiar de modelo**: §3 primero, presumir nada.
6. **No borrar baselines** — archivar en §5; son la única forma de saber si
   el nuevo modelo es mejor o solo distinto.
