# Catálogo de Optimizaciones PTX / Ensamblador de Bajo Nivel para Ampere (sm_86)

Este documento recopila las optimizaciones de hardware, ensamblador PTX y arquitectura de memoria para maximizar el throughput y reducir latencias en GPUs NVIDIA RTX 3090 (`sm_86`), bajo configuración Tensor Parallelism 2 (TP=2) y decodificación especulativa (MTP).

---

## 1. Concurrencia de Hardware y Solapamiento de Pipelines

### 1.1 Ejecución Concurrente Tensor Cores (INT8) + ALUs FP32 (Dual-Issue)
* **Mecanismo**: Cada Streaming Multiprocessor (SM) de Ampere cuenta con 4 Warp Schedulers y tuberías físicas independientes para los Tensor Cores de 3ra generación (`mma.sync.aligned.m16n8k32.row.col.s32.s8.s8`) y las ALUs de punto flotante de 32 bits (`fma.rn.f32`).
* **Aplicación**: Entramar (*interleave*) las instrucciones del MatMul en INT8 con el escalado en FP32 de los bloques $K$ previos. Mientras los Tensor Cores realizan la contracción de matrices, las ALUs FP32 escalan y acumulan resultados parciales en registros sin tiempos muertos.

### 1.2 Multi-Staging con Copia Asíncrona (`cp.async`)
* **Mecanismo**: Uso de DMA directo por hardware desde VRAM a Shared Memory sin pasar por registros de propósito general.
* **Instrucciones clave**:
  ```ptx
  cp.async.ca.shared.global [%r_smem], [%r_gmem], 16;
  cp.async.commit_group;
  cp.async.wait_group 2;
  ```
* **Configuración**: Triple Buffering (`num_stages = 3`) en decode y Quad Buffering (`num_stages = 4`) en prefill para ocultar completamente la latencia de memoria global.

---

## 2. Optimización de Memoria y Coalescencia

### 2.1 Cargas y Almacenamientos Vectorizados de 128 bits
* **Mecanismo**: Garantizar alineación estricta de 16 bytes en todas las direcciones de tensores de activación, pesos y escalas.
* **Instrucciones clave**:
  `ld.global.v4.u32` / `st.global.v4.u32` y `cp.async.ca 16B`.
* **Impacto**: Cada 4 hilos adyacentes saturan el bus de 384 bits de la GPU en una sola transacción en vez de 4 transacciones individuales.

### 2.2 Layouts con Swizzling XOR en Shared Memory (0 Bank Conflicts)
* **Mecanismo**: Evitar conflictos de bancos de 32 vías al acceder a columnas o matrices transpuestas en Shared Memory mediante permutación de índices XOR:
  $$\text{smem\_col} = \text{offs\_k} \oplus (\text{offs\_n} \bmod 32)$$
* **Impacto**: Ancho de banda de Shared Memory al 100% de la capacidad de hardware sostenida.

### 2.3 Ring Buffers de Shared Memory para Aumento de Ocupación
* **Mecanismo**: En lugar de alocar buffers estáticos para todas las etapas, implementar anillos circulares con punteros modulares sobre registros base.
* **Impacto**: Reduce la huella de Shared Memory por bloque de ~72 KB a ~36 KB, permitiendo 2 CTAs concurrentes por SM y duplicando la ocupación teórica.

### 2.4 Pistas de Caché L2 (`Cache Hints` y Políticas de Desalojo)
* **Mecanismo**:
  - Modificador `.nc` (*non-coherent / stream-once*) para activaciones efímeras del prefill (no contaminan L2).
  - Modificador `.ca` / `.cg` para pesos y escalas persistentes.
* **Impacto**: Mantiene los pesos más consultados (`lm_head`, capas iniciales) residentes en los 6 MB de caché L2.

---

## 3. Fusión de Operaciones a Nivel de Registros

### 3.1 Fusión Completa en el Epílogo de GEMM
* **Mecanismo**: Realizar el des-escalado (`acc_fp32 * a_scale * b_scale`), la suma de residuales / bias y el casteo final a `bfloat16` (`cvt.rn.bf16.f32`) directamente en los registros del hilo antes de escribir en memoria global con una única instrucción de escritura.

### 3.2 Fusión Inter-Kernel: SiLU + Mul + Quantización INT8 (SK-05 $\rightarrow$ SK-06)
* **Mecanismo**: El epílogo de `gate_up` (`SK-05`) calcula $\text{SiLU}(\text{gate}) \times \text{up}$, determina el `amax` por fila y emite la activación directamente en INT8 junto a su escala.
* **Impacto**: Elimina el buffer intermedio en VRAM y el kernel de cuantización intermedio previo a `down_proj` (`SK-06`).

### 3.3 Fusión de All-Reduce TP=2 en Epílogo (`RowParallelLinear`)
* **Mecanismo**: Con enlace PCIe P2P directo (topología PIX y ReBAR), el kernel de `down_proj` o `o_proj` escribe y acumula su suma parcial directamente en el buffer de la GPU vecina en el mismo epílogo.
* **Impacto**: Elimina dos lecturas y dos escrituras globales de VRAM por capa.

---

## 4. Instrucciones Especializadas a Nivel de Bits e Instrucción de Máquina

### 4.1 Operaciones Lógicas Ternarias y Permutaciones (`lop3.b32` y `prmt.b32`)
* **Mecanismo**:
  - `lop3.b32`: Resuelve cualquier función lógica de 3 entradas booleanas en 1 ciclo de reloj.
  - `prmt.b32`: Reordena, extrae y empaqueta 4 bytes en 1 ciclo.
* **Impacto**: Sustituye cadenas de 3 a 5 instrucciones `and`/`or`/`shr`/`shl` en rutinas de cuantización por una única instrucción de 1 ciclo.

### 4.2 Código 100% Predicado en Epílogos (Zero-Branching)
* **Mecanismo**: Uso sistemático de registros de predicado (`@%p0`, `@%p1`) para máscaras de bordes y residuales opcionales en vez de saltos condicionales (`bra`).
* **Impacto**: Cero penalización por fallos de predicción de saltos en el hardware.

### 4.3 Control de Presión de Registros (`launch_bounds`)
* **Mecanismo**: Acotar el uso máximo de registros por hilo a $\le 64$ o $\le 128$ registros mediante optimización de vida útil de variables temporales.
* **Impacto**: Duplica la cantidad de warps activos simultáneamente en el SM, permitiendo al Warp Scheduler ocultar totalmente las latencias de lectura.

---

## 5. Lanzamiento y Control de Runtime

### 5.1 Despacho con `cuLaunchKernelEx` y Programmatic Stream Serialization
* **Mecanismo**: Uso de flags avanzados de CUDA driver para encadenar las 64 capas directamente en la GPU sin interrupciones del hilo de CPU / Python.
* **Impacto**: Reduce la sobrecarga de despacho (*launch overhead*) a valores cercanos a cero.
