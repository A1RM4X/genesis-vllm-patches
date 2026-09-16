# Laboratorio ox-alpha — trazado del circuito del token en vLLM v0.23.0

Todo corre DENTRO del contenedor (`vllm/vllm-openai:v0.23.0`, ya presente en
el host). Nada se ejecuta localmente salvo controles de docker.

## Estructura

```
lab/
├── docker-compose.lab.yml   # servicio GPU con mounts ro de scripts/patch
├── patch/sitecustomize.py   # instrumentación inyectada en TODOS los procesos
├── scripts/
│   ├── trace_small.py       # orquestador de generación (modelo chico)
│   ├── run_small.sh         # corrida small: TP=1, EngineCore in-process
│   ├── run_real.sh          # corrida real: 27B FP8 TP=2 + MTP K=3
│   └── summarize_trace.py   # contabilidad de Memcpy desde trazas chrome
└── results/                 # logs + trazas + resúmenes (writable)
```

## Cómo funciona la instrumentación

`patch/sitecustomize.py` se activa con `LAB_TRACE=1` y viaja por PYTHONPATH a
todos los procesos que Python lance (spawn hereda env). Envuelve:

| Punto | Qué prueba |
|:---|:---|
| `GPUModelRunner.execute_model` + fases | tiempos por paso; abre ventana `torch.profiler` (pasos 5..25) que captura **todos** los Memcpy con bytes |
| `_to_list` | la D2H síncrona de `sampled_token_ids` (camino sin async-scheduling) |
| `AsyncGPUModelRunnerOutput.__init__/get_output` | la D2H async en copy stream (con async-scheduling) |
| `RejectionSampler.parse_output` | el `.cpu().numpy()` del spec-decode |
| `CpuGpuBuffer.copy_to_gpu` | cada H2D de buffers pinned (input_ids, metadata...) |

## Corridas

### Small (iteración rápida, ~5 min)

```bash
cd workshop/ox_alpha/lab
docker compose -f docker-compose.lab.yml up --abort-on-container-exit lab
# resultados: results/run_small_console.log, results/lab_*.log,
#             results/worker_trace_*.json, results/memcpy_summary_small.txt
```

Modelo: `Qwen/Qwen3-0.6B` descargado a `results/hf-cache` (no toca models-cache).

### Real (27B FP8 TP=2 MTP, réplica PROD sin offloading)

```bash
LAB_RUN=run_real.sh docker compose -f docker-compose.lab.yml up --abort-on-container-exit lab
```

Requiere las GPUs libres → **bajar antes el contenedor PROD**
(`docker stop genesis-27b-qwen38-fp8`) y **levantarlo después**
(`docker start genesis-27b-qwen38-fp8`).

## Qué demostrar

1. En steady-state, los únicos D2H del circuito son los token ids muestreados
   (~num_reqs×(K+1)×4 B/paso) — no activaciones.
2. Con `--async-scheduling` esa D2H se solapa en copy stream aparte.
3. Los H2D por paso son solo metadatos (input_ids, block tables, sampling
   metadata), todos pinned+non_blocking.
4. El estado GDN/mamba es device-to-device (sin paso por RAM).
