# 🔬 Circuito del token en vLLM v0.23.0 — análisis función-por-función y veredicto de cruces CPU/RAM

> **Complemento de** `CONTEXTO-INVESTIGACION.md`. Pregunta a responder:
> *desde que entra un prompt hasta que sale el token generado, ¿algún dato
> pasa por CPU/RAM y vuelve a la GPU?* — con verificación estática
> (código v0.23.0) **y** dinámica (laboratorio en contenedor con
> instrumentación + `torch.profiler` sobre el modelo real de PROD).
>
> | | |
> |---|---|
> | **Fecha** | 2026-08-24 |
> | **Método** | 3 exploraciones estáticas del árbol + 2 corridas instrumentadas dentro de `vllm/vllm-openai:v0.23.0` (small: Qwen3-0.6B TP=1; real: Qwen3.8-27B-FP8 TP=2 MTP K=3) |
> | **Laboratorio** | `workshop/ox_alpha/lab/` (compose + `patch/sitecustomize.py` + scripts + resultados en `results/{small,real}/`) |

---

## 1. Veredicto ejecutivo

**Las activaciones nunca salen de la GPU.** Hidden states, logits completos,
estado GDN/mamba, KV cache y pesos viven y mueren en VRAM. Lo que sí cruza
por RAM, medido y clasificado:

| Cruce | Tamaño medido | Frecuencia | ¿Vuelve a la GPU? | Clasificación |
|:---|:---|:---|:---|:---|
| Token ids muestreados (D2H) | **16–64 B/paso** (`[num_reqs, K+1]` int32) | cada paso | **NO** (van a detokenizer/IPC) | Estructural al diseño multiproceso |
| Metadatos de entrada (H2D) | ~6 KB/paso (real, 4 reqs) | cada paso | SÍ (son los inputs) | Inevitable y mínimo |
| Round-trip del último token (síncrono) | 4 B/token | cada paso | **SÍ** — pero **eliminado** con `--async-scheduling` (scatter GPU→GPU) | Ya optimizado en esta config |
| KV blocks GPU⇄RAM⇄NVMe | MB–GB | bajo demanda | SÍ (es el feature) | Por diseño (OffloadingConnector), fuera del camino crítico |
| IPC entre procesos (ids por SHM/ZMQ-msgpack) | bytes | por paso | NO | Estructural (3 procesos: API/EngineCore/worker) |

En números del modelo real (21 pasos × 2 ranks, ventana de profiling):

```
DtoH TOTAL:  0.0040 MB  (ambos ranks)   ← solo token ids muestreados
HtoD TOTAL:  0.2493 MB                  ← input_ids + block tables + sampling metadata
DtoD:      318.88 MB/rank               ← KV writes + estados GDN (align) + graphs: TODO intra-GPU
```

---

## 2. El circuito completo, función por función

### 2.1 ENTRADA: HTTP → token ids en VRAM

| # | Función | Ref (assets/vllm) | Qué hace | CPU/GPU |
|:--|:---|:---|:---|:---|
| 1 | `create_chat_completion` | `entrypoints/openai/chat_completion/api_router.py:53` | recibe HTTP | CPU (proc API) |
| 2 | `render_chat` → `tokenize_prompts` | `renderers/base.py:982,1007` | chat template + encode; truncation `_token_truncation` (`params.py:396`) | CPU (tokenizer Rust) |
| 3 | `process_inputs` → `EngineCoreRequest` | `v1/engine/input_processor.py:242,370` | arma request con `prompt_token_ids: list[int]`; pixel_values MM van out-of-band por `TensorIpcSender` (SHM) | CPU |
| 4 | `MsgpackEncoder` + ZMQ ROUTER/DEALER | `core_client.py:590,846,1049`; `serial_utils.py:136,166` | serializa y envía al EngineCore (frames zero-copy >256 B) | CPU/RAM (IPC) |
| 5 | `EngineCore.add_request` → `scheduler.add_request` | `core.py:341,372` | cola interna | CPU |
| 6 | `Scheduler.schedule()` → `SchedulerOutput` | `core/sched/scheduler.py:340`; `sched/output.py:31,112,181` | decide qué tokens correr; **todo listas Python en CPU** | CPU |
| 7 | Broadcast a workers | `multiproc_executor.py:151,374` | MessageQueue sobre shared memory | CPU/RAM (IPC) |
| 8 | `GPUModelRunner._update_states` | `gpu_model_runner.py:4047` | agrega requests al `InputBatch`; histórico vive en `token_ids_cpu` numpy `(max_reqs, max_len)` **no pinned** (`gpu_input_batch.py:133-139,361`) — **no existe espejo GPU del histórico** | CPU |
| 9 | `_prepare_inputs` | `gpu_model_runner.py:4089,1891-1936` | gather del chunk agendado desde `token_ids_cpu`; block tables H2D primero (pinned, solapable) | CPU→prepara |
| 10 | **`_prepare_input_ids` → `CpuGpuBuffer.copy_to_gpu`** | `:1719,2108`; `v1/utils.py:138` | **EL H2D de input_ids**: buffer pinned + `non_blocking=True`, `total_scheduled × 4 B` por paso | **RAM→VRAM** |
| 11 | resto de H2D del paso | `:1990,2017,2079,2085,2088,2090,2117-2126`; `gpu_input_batch.py:855-903` | query_start_loc, req_indices, num_computed/scheduled_tokens, discard_mask, mrope positions, sampling metadata — todos pinned+async, ~KB | RAM→VRAM |
| 12 | (async-sched) scatter del último token | `:1785-1803` | `input_ids.gpu.scatter_(src=prev_sampled_token_ids)` — **el token muestreado del paso anterior se reutiliza EN GPU sin bajar** | VRAM→VRAM |
| 13 | `embed_input_ids` | `:3447,3492` | embedding lookup sobre `input_ids.gpu` | VRAM |

### 2.2 FORWARD: 100% VRAM

- Atención full: FlashInfer FA2 (KV fp8 in-kernel dequant) — ver `CONTEXTO-INVESTIGACION.md` §9.1.
- GDN: pipeline Triton FLA; estados recurrentes device-to-device
  (`postprocess_mamba_align_gpu` lee block tables desde buffers GPU;
  `mamba_utils.py:751-804` diseñado *"so the GPU kernel can perform state
  copies without CPU-GPU sync"*). Única salida: `num_accepted_tokens`
  D2H asíncrono diminuto (:802-804).
- Lineales FP8: Marlin W8A16 in-kernel dequant (§8 del contexto).
- All-reduce TP=2: PYNCCL `ncclAllReduce` con `data_ptr()` de device —
  **sin staging por host** (`pynccl.py:166-197`, assert de dispositivo :178).
- Spec-decode MTP: verificación accept/reject en kernels Triton GPU
  (`rejection_sampler.py:119-197`: `rejection_greedy_sample_kernel` :714,
  `rejection_random_sample_kernel` :768, `expand_kernel` :837,
  `sample_recovered_tokens_kernel` :861); drafts de la siguiente ronda
  permanecen en buffers GPU (`token_ids_gpu_tensor`,
  `gpu_model_runner.py:575-580`).
- CUDA graph replay: los `.item()` existentes leen buffers CPU-side
  pre-replay, no estancan el stream (`model_states/default.py:174,180`).

### 2.3 SALIDA: sampler → cliente

| # | Función | Ref | Qué hace | Cruce |
|:--|:---|:---|:---|:---|
| 1 | `Sampler.forward/sample` | `v1/sample/sampler.py:72-149,243-302` | logits fp32 in-place, processors in-place, top-k/top-p Triton/FlashInfer, argmax/gumbel | **TODO GPU** (comentario explícito `# These are GPU tensors.` :141) |
| 2 | `RejectionSampler.__call__` | `rejection_sampler.py:119-197` | verificación especulativa | GPU |
| 3a | Camino SÍNCRONO: `_bookkeeping_sync` → `_to_list` | `gpu_model_runner.py:3601-3635, 7472-7485` | pinned copy + `transfer_event.synchronize()` + `.tolist()` | **D2H bloqueante 4·num_reqs·(K+1) B** |
| 3b | Camino ASYNC (`--async-scheduling`): `AsyncGPUModelRunnerOutput.__init__` | `:4624,263-280` | copia en `async_output_copy_stream` (stream aparte): `.to("cpu", non_blocking=True)` + event | **D2H solapada**, sync diferido a `get_output()` (:288) en thread `async_output_busy_loop` (`multiproc_executor.py:951-967`) |
| 4 | `RejectionSampler.parse_output` | `rejection_sampler.py:249-283` (`output_token_ids.cpu().numpy()` :267) | filtra placeholders de rechazo | D2H pequeña post-copia |
| 5 | Worker→EngineCore | `multiproc_executor.py:938-939` | `ModelRunnerOutput` (list[list[int]]) por SHM broadcast | RAM (IPC) |
| 6 | `scheduler.update_from_output` → `EngineCoreOutput` | `core/sched/scheduler.py:1553-1568`; `engine/__init__.py:170-177` | new_token_ids como lista Python | CPU |
| 7 | EngineCore→frontend | `core_client.py:522,557,993-995`; MsgpackDecoder :591 | ZMQ PULL + msgspec.msgpack | RAM (IPC) |
| 8 | `OutputProcessor.process_outputs` → `IncrementalDetokenizer` | `output_processor.py:576-693,639`; `detokenizer.py:167,210,250` | decodificación incremental (DecodeStream Rust) + stop strings | CPU puro |
| 9 | SSE | `chat_completion/serving.py:498,553`; `api_router.py:74` | JSON chunks al socket | CPU/red |

### 2.4 OTROS cruces (fuera del camino directo)

- **KV OffloadingConnector**: store GPU→RAM **diferido al inicio del paso
  siguiente** en stream propio con copy-engine (`offloading/worker.py:252-257`;
  `cpu/gpu_worker.py:387-392,246-426`); load RAM→GPU en stream propio sin
  barrera por capa; tier NVMe **nunca toca GPU** (hilos CPU + mmap pinned,
  `tiering/fs/manager.py:44-160`). Bloqueo síncrono solo en preemption.
- **prompt_logprobs**: el D2H más grande del sistema —
  `(len_prompt−1)·(K+1)·8 B` por request + `_sync_device()`
  (`gpu_model_runner.py:5435-5499`). Solo si se piden.
- **logprobs por token**: `num_tokens·(K+1)·8 B` por paso si se piden
  (`outputs.py:62-80`).
- **draft tokens D2H**: solo con structured-output/penalties
  (`gpu_model_runner.py:4698-4729`).
- **sleep/wake cumem**: cudaMemcpy de backups (solo sleep mode).
- **profile/dummy runs**: datos sintéticos, connector desactivado — nada real.

---

## 3. Evidencia dinámica del laboratorio

Instrumentación: `patch/sitecustomize.py` inyectado por PYTHONPATH en todos
los procesos; envuelve `execute_model` (V1 y V2), `_to_list`,
`AsyncGPUModelRunnerOutput`, `AsyncOutput` (V2), `parse_output`,
`CpuGpuBuffer.copy_to_gpu`; abre `torch.profiler` en pasos 5..25.

### 3.1 Corrida SMALL (Qwen3-0.6B, TP=1, EngineCore in-process)

```
DtoH = 0.0010 MB (36 copias)   HtoD = 0.0194 MB (47)   DtoD = 0
kernels: gemm=29.1ms · triton=4.8ms
```

Hallazgo clave: v0.23 corrió el **runner V2** (`v1/worker/gpu/model_runner.py`)
para Qwen3 denso y activó **async-scheduling por defecto**
(`"Asynchronous scheduling is enabled"`). La D2H observada por paso:
`AsyncOutput.__init__` → numpy `(num_reqs,1)` int64.

### 3.2 Corrida REAL (Qwen3.8-27B-Uncensored-FP8, TP=2, MTP K=3, kv fp8_e4m3)

```
POR RANK (ventana 21 pasos):        DtoH = 0.0020 MB (89 copias)
                                    HtoD = 0.1246 MB (1209 copias, pinned+pageable)
                                    DtoD = 318.88 MB (1344 copias)
kernels: marlin=320-326ms · gemm=100-104ms · nccl=12.8-16.1ms
         · triton=13.6-14.2ms · flashinfer=4.4-4.9ms
```

Logs dirigidos (`results/real/lab_577.log`):
- `D2H_ASYNC V1 __init__ done bytes=16/32` → el tamaño REAL del D2H por paso:
  `[num_reqs, K+1]` int32 = 16 B (1 req) / 32 B (2 reqs).
- `SPEC parse_output` dispara 1×/paso (el `.cpu().numpy()` del spec-decode).
- `H2D copy_to_gpu` típicos: 44–1100 B (metadata), 56 B block tables delta.
- El híbrido corre el runner **V1** (V2 es solo para densos no-cuantizados).
- Boot log confirma: `"Setting attention block size to 1600 tokens"` → el
  1600 del compose es **calculado** (mamba page size), no constante.
- Throughput sanity: 256 tokens / 3.98 s = 64 tok/s agregados (4 reqs,
  enforce por defecto) — orden correcto para este hardware.

### 3.3 Bugs de instrumentación encontrados y corregidos (lecciones)

1. Runner V2 existe y es default para densos → envolver ambos.
2. `python - <<HEREDOC` + spawn multiprocessing = imposible re-importar main
   → siempre archivo con guard `__main__`.
3. `copy_to_gpu(num_elems=-1)` tiene default → wrapper debe respetarlo.
4. `parse_output` es `@staticmethod` → wrapper sin cls.

---

## 4. Respuesta formal a la pregunta

> ¿Nunca dato alguno pasa por CPU o RAM y vuelve a la GPU?

**Datos que SÍ hacen round-trip RAM⇄GPU en esta configuración:**

1. **El último token muestreado** — SOLO en modo síncrono (baja, pasa por
   `token_ids_cpu`, vuelve en el próximo `copy_to_gpu`). Con
   `--async-scheduling` (activo en PROD) ese round-trip **no existe**: el
   token se scatters GPU→GPU (`gpu_model_runner.py:1785-1803`).
2. **Bloques KV offloaded** — GPU→RAM(/NVMe)→GPU por diseño del feature
   (dos niveles de cache); medido fuera del camino crítico (streams propios,
   store diferido al paso siguiente, 1,0–1,9% del wall time según medición
   propia del compose).
3. **Metadatos de control por paso** (input_ids nuevos, block tables,
   sampling params): suben H2D una sola vez — no son round-trip.

**Lo que jamás cruza (verificado estática y dinámicamente):** activaciones,
hidden states, logits completos, pesos, estado conv/ssm GDN, KV residente en
VRAM, outputs de Marlin/FlashInfer/FLA. El DtoD de 319 MB/rank medido es
tráfico intra-GPU (KV writes + estados align + grafo), nunca toca RAM.

**Conclusión de minimización:** el circuito ya está en el mínimo estructural.
Los únicos residuos son (a) los token ids que DEBEN llegar a CPU para
detokenizar y salir por red — irreducibles sin romper la separación de
procesos, y (b) el offload KV, que es la funcionalidad pedida. No queda
ninguna transferencia grande ni redundante eliminable sin rediseñar la
arquitectura multiproceso (API/EngineCore/worker con IPC SHM+ZMQ).

---

## 5. Archivos del laboratorio

```
workshop/ox_alpha/lab/
├── docker-compose.lab.yml      # servicio GPU (imagen v0.23.0 ya presente)
├── patch/sitecustomize.py      # instrumentación (ver §3)
├── scripts/{trace_small.py,trace_real.py,run_small.sh,run_real.sh,
│            summarize_trace.py}
└── results/
    ├── small/   logs + worker_trace_140.json + memcpy_summary_small.txt
    └── real/    logs TP0/TP1 + worker_trace_{577,583}.json + memcpy_summary_real.txt
```

Para reproducir (requiere GPUs libres → `docker stop genesis-27b-qwen38-fp8`
antes y `docker start` después):

```bash
cd workshop/ox_alpha/lab
LAB_RUN=run_small.sh docker compose -f docker-compose.lab.yml up --abort-on-container-exit lab
LAB_RUN=run_real.sh  docker compose -f docker-compose.lab.yml up --abort-on-container-exit lab
```
