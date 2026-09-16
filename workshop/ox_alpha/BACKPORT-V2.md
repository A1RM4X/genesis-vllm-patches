# 🔄 Backports del Runner V2 al V1 como parches Genesis

> **Pregunta**: ¿hay optimizaciones visibles en el runner V2 de vLLM v0.23.0
> que valga la pena backportear al V1 (el que corre nuestro híbrido) mediante
> parches Genesis?
>
> **Método**: auditoría del hot-path CPU por paso del V1 (híbrido GDN + MTP
> K=3, TP=2, async-scheduling, 10 seqs) + inventario de las innovaciones del
> V2. Referencias sobre `assets/vllm`.
>
> | | |
> |---|---|
> | **Fecha** | 2026-08-24 |
> | **Complementa** | `CONTEXTO-INVESTIGACION.md`, `CIRCUITO-TOKEN.md` |

---

## 1. El hallazgo central: los 4 eventos de sincronización que V2 eliminó

El V1 sincroniza GPU→CPU **cuatro veces por paso** aunque el async-scheduling
esté activo. El V2 no tiene NINGUNA de las cuatro (grep en `vllm/v1/worker/gpu/`
= 0 coincidencias):

| Evento V1 | Sync cada paso | Sustituto V2 |
|:---|:---|:---|
| `num_accepted_tokens_event` (:870, record :1533, **sync :2021-2022**) | sí — espera el D2H del paso anterior antes de `_prepare_inputs` | tensor GPU persistente `num_accepted_tokens_gpu` + scatter Triton (`model_states/mamba_hybrid.py:68-70,145-175`) |
| `prepare_inputs_event` (:694, sync :3705-3716) | sí | buffers persistentes escritos con `copy_(non_blocking)`; el replay CG lee las mismas direcciones (`cudagraph_utils.py:1243-1247`) |
| `draft_token_ids_event` (:866, sync :4714-4739) | sí — incondicional vía `take_draft_token_ids` (`gpu_worker.py:896`) | drafts en tensor GPU `[max_reqs, K]`; D2H **solo con structured outputs** (`spec_decode/utils.py:26-43`) |
| `valid_sampled_token_count_event` (:862, sync :4745-4767) | sí (corrección diferida :1464) | `num_sampled`/`num_rejected` permanecen en GPU (kernel `input_batch.py:387-433`) |

Cada una de esas syncs es una parada potencial del pipeline cuando la GPU
todavía está drenando el paso anterior — exactamente el tipo de tail-latency
que duele en el patrón agéntico (ráfagas de subagentes).

## 2. Los otros dos pecados del V1 que V2 confesó arreglar

- **`_calc_spec_decode_metadata`** (`gpu_model_runner.py:2779-2793`):
  **5 allocaciones GPU + 5 H2D por paso**, con el propio upstream admitiendo
  `# TODO: Optimize the CPU -> GPU copy.` (:2778).
- **`preprocess_mamba` / `collect_mamba_copy_meta`**
  (`mamba_utils.py:640-703, 572-608`): loops Python req×grupo×capa×estado
  con `data_ptr()` y `state_copy_func` en Python — con el comentario
  *"TODO(Chen): we need to optimize this function a lot"* (:658). Escala con
  las ~48 capas GDN, no con num_seqs.

## 3. Candidatos a parche Genesis, rankeados

### PN-A — `num_accepted_tokens` residente en GPU ⭐⭐ (impacto alto, riesgo medio)

**Qué**: replicar `mamba_hybrid.py:68-70,145-175`: tensor GPU persistente
int32 `[max_num_seqs]` actualizado por scatter Triton tras el rejection
sampling; el builder GDN lo consume como tensor GPU y la máscara CPU se
deriva de `num_draft_tokens_per_req` (que el scheduler ya conoce).
**Elimina**: la única sync oculta dentro del camino crítico MTP+GDN (:2022).
**Cómo**: monkeypatch de `preprocess_mamba`/`_prepare_inputs` +
class-rebind del builder para aceptar el tensor GPU. Semántica a preservar:
valor neutro = 1 ("Mamba treats num_accepted_tokens=1 as the neutral
non-spec value", mamba_hybrid.py:149-152).

### PN-B — Buffers persistentes de metadatos spec-decode ⭐⭐ (impacto medio, riesgo bajo)

**Qué**: preasignar los 5 tensores de `_calc_spec_decode_metadata`
(`cu_num_draft_tokens`, `cu_num_sampled_tokens`, `logits_indices`,
`target_logits_indices`, `bonus_logits_indices`) una vez por shape máximo y
escribir por slice. Es el TODO explícito de upstream.
**Costo evitado**: 5 allocs + 5 H2D por paso → 5 escrituras en buffer pinned
reutilizado. Patrón idéntico al de `InputBuffers` del V2
(`gpu/input_batch.py:12-32`). Autocontenido, puro monkeypatch.

### PN-C — Gate del `draft_token_ids_event` ⭐ (impacto medio, riesgo bajo)

**Qué**: V1 sincroniza y `.tolist()`-ea los draft tokens **todos los pasos**
(:4714-4739 vía `take_draft_token_ids`) aunque nadie los consuma sin
structured-output/penalties. V2 demostró que se puede gates: copiar en
stream/event propios **solo si** hay grammar/bad-words/penalties
(`spec_decode/utils.py:26-30`).
**Verificación previa obligatoria**: confirmar que ningún otro consumidor lea
`draft_token_ids_cpu` en el camino común (core.py:480-482 es el único punto,
atado a correcciones diferidas).

### PN-D — Cache del staging mamba ⭐ (impacto medio, riesgo bajo)

**Qué**: `stage_postprocess_inputs_to_gpu` (`mamba_utils.py:807-900`) hace 2
loops Python por request + **4 H2D por paso** con valores que casi nunca
cambian. Cachear por clave `(req_id, num_computed_tokens)` y emitir H2D solo
si cambió. Complemento barato de PN-A dentro del mismo archivo.

### PN-E — Limpieza del builder GDN (impacto medio-bajo, riesgo bajo)

**Qué**: `gdn_attn.py:167-322` hace ~10 `.item()` sobre tensores **CPU**
(puro overhead de dispatch Python, no sync GPU) + 3 `torch.zeros` por paso en
el path spec-decode (:294-317) teniendo ya buffers persistentes para el caso
full-CG (:151-165). Reemplazar `.item()` por lecturas numpy y extender los
buffers persistentes al path mixed. Los comentarios del V2 validan la técnica
("Use CPU tensors to avoid CPU-GPU sync", :228).

### PN-F — Adelantar el lanzamiento del AsyncOutput (impacto bajo, esfuerzo trivial)

**Qué**: V2 lanza la D2H de sampled tokens **antes** del postprocess y del
`propose` del speculator para solapar más
(`gpu/model_runner.py:1376-1411`). En V1, `AsyncGPUModelRunnerOutput` se
construye después del bookkeeping. Adelantar la construcción = más solape
gratis con async-scheduling ya activo.

### ❌ Lo que NO conviene backportear

- **Espejo UVA del histórico + kernel `post_update` único**
  (`states.py:33-81`, `input_batch.py:436-535`): es el corazón del rediseño
  V2 — cambia el contrato de datos scheduler↔runner completo. Eso no es un
  parche, es portar el V2.
- **`prepare_inputs` sin loops**: depende del espejo anterior; mismo veredicto.
- **Penalties persistentes con skip numpy**: ganancia solo si hay penalties
  activas; en este stack son raras fuera de grammar. Bajo prioridad.

## 4. Expectativa honesta de ganancia

Con batch=10 y pasos de decode dominados por cómputo GPU (~50 ms/step en el
peor caso lockstep, ~13 ms típico), el trabajo CPU eliminado (items B/D/E:
~0,5–3 ms/paso) mueve poco el throughput. **La ganancia real está en las
syncs** (items A/C/F): cada `event.synchronize()` puede estancar el inicio
del paso siguiente esperando el drenaje del anterior — su costo aparece como
*p99/latencia bajo carga*, no como promedio. Estimación conservadora:
mejora de un dígito porcentual en latencia media de decode y mayor en colas;
throughput agregado casi intacto. Medir con el harness existente
(`tests/repro/carga_mixta.py`) antes/después, un parche por vez.

## 5. Orden de implementación sugerido

```
1. PN-B (bajo riesgo, patrón claro, valida el harness de medición)
2. PN-D + PN-E (mismo archivo familia, bajo riesgo)
3. PN-C (requiere verificación de consumidores)
4. PN-A (la joya; necesita test numérico de regresión GDN como PN59)
5. PN-F (trivial, último para no confundir mediciones)
```

Cada uno con su TDD según `docs/PLUGINS.md`, registro en `PATCH_REGISTRY`,
env-flag `GENESIS_ENABLE_PNxxx_*`, y A/B en PROD 2×3090 antes de promover a
default-ON.

## 6. Referencias cruzadas

- Auditoría V1: loops/allocs/syncs — `gpu_model_runner.py:1125-1493, 1700-1825,
  2730-2808, 3562-3701, 4714-4774`; `mamba_utils.py:572-703, 807-900`;
  `gdn_attn.py:167-322`; `step3p5.py:121-152, 274-459`.
- Innovaciones V2: `gpu/model_runner.py:816-979, 1437-1438`;
  `gpu/states.py:9-139`; `gpu/input_batch.py:12-32, 174-384, 387-535`;
  `gpu/buffer_utils.py:26-42, 92-233`; `gpu/model_states/mamba_hybrid.py:57-175`;
  `gpu/spec_decode/{speculator,autoregressive/speculator,utils}.py`;
  `gpu/sample/{penalties,states}.py`.
