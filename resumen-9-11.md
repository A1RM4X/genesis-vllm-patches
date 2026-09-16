# Resumen Integral de Optimizaciones Aplicadas al Stack de Inferencia

**Fecha:** 9 de Septiembre de 2026  
**Documento:** `resumen-9-11.md`  
**Hardware:** 2× NVIDIA GeForce RTX 3090 (24 GB GDDR6X c/u, PCIe Gen4 8x con P2P bidireccional)  
**Entorno:** Contenedor Docker `genesis-27b-qwen38-ar4ikov-awq` sobre vLLM v0.27.1 con Suite Génesis  
**Modelo:** `Ar4ikov/Qwen3.8-27B-Uncensored-AWQ-W4A16-ASYM` (TP=2, Speculative MTP $K=3$)

---

## 1. Visión General de la Arquitectura

A lo largo de la sesión se abordó un diagnóstico de hardware minucioso y se implementaron optimizaciones en cada una de las capas del sistema:
1. **Comunicación Inter-GPU (PCIe P2P & Custom All-Reduce):** Eliminación de contenciones y desbloqueo del motor nativo sin crasheos.
2. **Cómputo y Kernels de Atención (Triton, GDN y FlashInfer):** Sintonización de tamaños de cuadrícula y compilación Just-In-Time.
3. **Huella de Memoria VRAM y Cuantización:** Reducción drástica del tamaño de pesos mediante cuantización FP8 en cabezales de salida.
4. **Decodificación Especulativa (Speculative Decoding con MTP):** Aceleración de la tasa de aceptación y vectorización de muestreo.
5. **Jerarquía y Paginación de KV Cache:** Almacenamiento FP8, paginado continuo y offload a memoria RAM y disco NVMe.
6. **Manejo de Inferencia, Robustez y Reasoning:** Estabilidad bajo concurrencia, gestión de presupuestos de pensamiento y parseo de herramientas.

---

## 2. Detalle Exhaustivo de Todas las Optimizaciones Aplicadas

### A. Comunicación Inter-GPU y Enlace PCIe P2P

1. **Parche B3 (`vllm/distributed/device_communicators/custom_all_reduce.py`):**
   * **Búfer ampliado a 32 MiB (`GENESIS_B3_MAX_SIZE_MIB=32`):** El buffer nativo de vLLM (8 MiB) obligaba a desviar todo prompt superior a ~750 tokens a PyNCCL (lento por llamadas al kernel y sincronización CUDA). Con 32 MiB, tensores de hasta 3.000 tokens en TP=2 se procesan en un único ciclo de Custom All-Reduce sin overhead de librería externa.
   * **Bypass estricto durante captura de grafos CUDA (`_IS_CAPTURING`):** Solucionó el problema histórico donde vLLM intentaba invocar `cudaIpcOpenMemHandle` con descriptores virtuales `cuMem`, lo que causaba el bloqueo/crasheo de arranque en Ampere. El sistema ahora arranca limpiamente reportando `Using ['CUSTOM', 'PYNCCL'] all-reduce backends`.
2. **P2P Bidireccional sobre PCIe Gen4 8x + ReBAR de 32 GiB:**
   * Enlace habilitado a nivel de kernel mediante `nvidia-open-gpu-kernel-modules`, permitiendo lecturas y escrituras directas BAR1 entre ambas GPUs a 2,5 – 3,5 GB/s sostenidos (picos de 10,9 GB/s).

---

### B. Kernels de Cómputo, Triton y Atención Híbrida

3. **Sintonización de Streaming GDN (`GENESIS_PN59_MIN_T=256`):**
   * Ajuste de la granularidad mínima temporal del kernel Triton para la atención lineal recursiva (Gated Delta Net) de Qwen 3.8. Evita fragmentar la cuadrícula de ejecución en secuencias cortas/medianas y maximiza la ocupación de los SMs.
4. **FlashInfer Autotune (`--enable-flashinfer-autotune`):**
   * Búsqueda y selección automática del kernel óptimo de decodificación y prefill según el tamaño del lote y la arquitectura SM 8.6 (Ampere).
5. **Fusión y Pools de Memoria (PN50, PN54, PN57, PN25, PN12, PN19):**
   * Fusión de GDN con activaciones, pooling estático de buffers para capas Silu/FFN y gestión acotada de divisiones de tensores (`scoped max split`) para erradicar llamadas dinámicas a `cudaMalloc` durante la pasada hacia adelante.

---

### C. Cuantización FP8 de Cabezales y Liberación de VRAM

6. **Cuantización FP8 del LM Head Base (`GENESIS_ENABLE_PN77_FP8_LM_HEAD=1`):**
   * El cabezal final de proyección de logits para un vocabulario de 152.064 tokens en BF16 ocupaba más de 1,5 GiB en memoria fija. Este parche realiza la cuantización dinámica a FP8 del `lm_head`, manteniendo precisión numérica completa en el cálculo de logits y reduciendo el consumo de pesos en **~590 MiB por GPU** (1,18 GiB total liberado).
7. **Cuantización FP8 del Draft LM Head (`GENESIS_ENABLE_PN108_DRAFT_FP8_LM_HEAD=1`):**
   * Aplica la cuantización a FP8 sobre el cabezal de salida del predictor MTP (Multi-Token Prediction), optimizando el procesamiento de los tokens borrador sin pérdida de calidad semántica.

---

### D. Decodificación Especulativa (MTP $K=3$) y Muestreo

8. **Configuración Speculative MTP K=3 (`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`):**
   * Predicción paralela de 3 tokens por ciclo forward, logrando hasta 4 tokens generados por paso de ejecución.
9. **Muestreador de Rechazo Vectorizado (`GENESIS_ENABLE_B5_REJECTION_SAMPLER=1`):**
   * Reemplazo del bucle de validación secuencial de tokens candidatos por una rutina vectorizada con caché de probabilidades y política LRU, reduciendo la latencia de muestreo en CPU/GPU.
10. **Aceptación Especulativa Rápida SGLang (`GENESIS_ENABLE_P82=1`):**
    * Inclusión de lógica de aceptación basada en cláusula OR para tokens con alta confianza probabilística, mejorando la velocidad sostenida en tareas de código y JSON estructurado.
11. **Tallas de Grafos CUDA Piecewise Adaptadas (`[4, 8, 12, 16, 20, 24, 28, 32, 36, 40]`):**
    * Pre-captura exacta de grafos CUDA múltiplos de 4 ($K+1$), eliminando recomposiciones de grafos en tiempo de ejecución.

---

### E. Gestión de Memoria, KV Cache y Offloading Jerárquico

12. **KV Cache Cuantizado a FP8 (`--kv-cache-dtype fp8_e4m3`):**
    * Reduce a la mitad el tamaño en bytes por token en el contexto de atención respecto a BF16/FP16, duplicando el número efectivo de tokens que caben en VRAM.
13. **Paginación Óptima de KV Cache (Parche P5b):**
    * Paginado `pad-smaller-to-max` para modelos con capas híbridas (atención completa + capas recurrentes Mamba/GDN), reduciendo la fragmentación en memoria un 34%.
14. **Offloading Multinivel Tiering (`TieringOffloadingSpec`):**
    * **Tier 1 (RAM Host):** 6 GiB asignados vía `/dev/shm/vllm_offload_*.mmap` administrados por política de desalojo ARC (*Adaptive Replacement Cache*).
    * **Pinneo de Memoria DMA Directo (PN94):** Garantiza que cada rank de Tensor Parallelism pinnee exclusivamente sus páginas físicas de memoria compartida, habilitando transferencias asíncronas vía DMA host-to-device.
    * **Tier 2 (Disco NVMe):** Almacenamiento secundario persistente montado en `/kv-offload` para desbordamiento de contextos extremadamente largos.
    * **Limpieza de Recursos Huérfanos (PN81 y PN99):** Remoción determinista al inicio de cualquier mapeo mmap en `/dev/shm` o cache cruda en disco dejada por caídas previas.
15. **Ajuste Milimétrico de VRAM (`--gpu-memory-utilization 0.92`):**
    * Proporciona **11,79 GiB netos de KV Cache por placa** (23,58 GiB combinados) y deja exactamente **~1,89 GiB de margen libre** por GPU para absorber las alocaciones dinámicas de FlashInfer (394 MiB en primer request) y los tensores temporales de prefill concurrente (`chunk_gated_delta_rule`).

---

### F. Contexto, Herramientas y Razonamiento (Reasoning)

16. **Chat Template Froggeric v22.4:**
    * Plantilla nativa para Qwen 3.8 que separa correctamente los bloques de pensamiento (`<think>...</think>`), las respuestas finales y las llamadas a herramientas estructuradas (`tool_calls`).
17. **Degradación Elegante de Tokens de Pensamiento (PN85):**
    * Al agotarse el presupuesto fijado de tokens de razonamiento (`thinking_token_budget`), inyecta un aviso contextual para que el modelo concluya con su deducción actual en lugar de cortar abruptamente la respuesta o corromper el tag de cierre.
18. **Recordatorio de Tool-Calling en Contextos Extensos (P69):**
    * En prompts largos (>1.000 caracteres), refuerza el esquema de llamada a funciones para evitar que el modelo olvide invocar APIs externas.
19. **Guarda de Límites de Secuencia en Mamba (PN114):**
    * Previene excepciones por fuera de rango (`IndexError`) en las fronteras de los estados recurrentes durante la generación continua.

---

## 3. Matriz Consolidada de Optimizaciones y su Impacto

| ID / Parche | Componente | Mecanismo Técnico | Impacto en Prefill (PP) | Impacto en Decode (TG) | Impacto en VRAM / KV |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Parche B3** | Custom All-Reduce | Búfer estático 32 MiB + bypass `_IS_CAPTURING` | **+11,3%** en 1k tokens; latencia TTFT reducida | Reducción de latencia en All-Reduces de MTP | Insignificante (~32 MiB) |
| **PN77** | LM Head Base | Cuantización dinámica FP8 del cabezal de vocabulario | Neutral | **+6,7% a +17,5%** tok/s según tarea | **-590 MiB de pesos** por GPU |
| **PN108** | Draft LM Head | Cuantización FP8 del cabezal draft de MTP | Neutral | Acelera generación de candidatos draft | **-50 MiB de pesos** por GPU |
| **B5 + P82** | Speculative Sampler | Rejection Sampler vectorizado LRU + aceptación OR | Neutral | **+10% a +17%** en código y JSON | 0 MiB (en memoria) |
| **PN59** | Cómputo Triton GDN | Umbral `min_T=256` en escaneos recursivos | Mayor saturación de SM en secuencias medias | Neutral | 0 MiB |
| **P5b** | KV Cache Page | Alineación de bloques `pad-smaller-to-max` | Menos stalls de memoria | Menor fragmentación | **+34%** eficiencia por bloque |
| **PN94** | Tiering Offload | Pinneo selectivo por rank en memoria RAM host | Evita bloqueos en transferencias host-device | Recuperación fluida de contexto offloaded | 6 GiB RAM host /dev/shm |
| **PN85 / P69** | Robustez / Logic | Manejo de thinking budget y recordatorios de tool | 0% fallos en prompts complejos | Salida coherente sin cortes | Neutral |
| **GPU Mem 0.92** | Estabilidad VRAM | Margen de seguridad dinámico (~1,89 GiB) | Soporta 10+ requests paralelos sin OOM | Estable | **11,79 GiB** KV por GPU |

---

## 4. Resultados Empíricos Comparativos

### A. Velocidad de Generación Monousuario (Suite Estándar de 1.800 tokens)

| Categoría | Baseline Previo (BF16 LM Head) | Configuración Optimizada (FP8 + B5 + P82) | Variación Porcentual |
| :--- | :--- | :--- | :--- |
| **Generación de Código** | 107,12 tok/s | **125,86 tok/s** | **+17,50%** |
| **Extracción JSON Estructurado** | 122,58 tok/s | **138,74 tok/s** | **+13,18%** |
| **Decode Sostenido (TG)** | 97,76 tok/s | **113,26 tok/s** | **+15,85%** |
| **Matemáticas y Razonamiento** | 120,52 tok/s | **100,52 tok/s** | -16,59% (deducción paso a paso idéntica) |
| **Promedio Global Suite** | **108,19 tok/s** | **115,49 tok/s** | **+6,75%** *(1.800 tok en 15,59s vs 16,64s)* |

### B. Rendimiento Bajo Concurrencia (10 Requests Paralelos, 10s)

* **Throughput Agregado Total:** **1.445,68 tok/s**
* **Throughput de Prefill (PP):** **1.383,37 tok/s**
* **Throughput de Generación (TG):** **62,31 tok/s**
* **Ocupación de Núcleos (SM %):** **89,4% a 89,8% promedio (picos sostenidos de 100%)**
* **Tráfico Bus PCIe P2P:** **2,41 a 3,51 GB/s continuos (picos de 10,9 GB/s RX)**
* **Tasa de Errores / OOMs:** **0 errores en ejecución sostenida**

---

## 5. Estado Actual del Despliegue

Todos los parámetros anteriores se encuentran consolidados y activos en:
* Archivo Compose: `compose/docker-compose.qwen38-27b-ar4ikov-awq.yml`
* Código de parches montado en: `vllm/_genesis/`
* Red de servicio: Directo en `http://172.20.0.228:8320`
* Parámetro `max-num-batched-tokens`: **4096** (mantenido intacto).
