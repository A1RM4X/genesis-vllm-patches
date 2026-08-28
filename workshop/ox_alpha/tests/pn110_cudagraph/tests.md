# PN110 — perfilado de decode: método y resultados

Fecha: 2026-08-27. Modelo `orcarouter/Qwen3.8-27B-Uncensored-FP8`, TP=2 en 2×RTX 3090
(sm_86), MTP K=3, FlashInfer, CUDA graphs ON, vLLM v0.27.1.

Todo lo de acá está medido sobre trazas del profiler de vLLM en un server real.
Ningún número es estimado.

---

## 1. Cómo se obtienen las trazas

### 1.1 El profiler cambió de API en v0.27.1

`VLLM_TORCH_PROFILER_DIR` **fue eliminada**. Las rutas `/start_profile` y
`/stop_profile` las monta `attach_router()` en
`vllm/entrypoints/serve/profile/api_router.py`, y **sólo** si
`args.profiler_config.profiler is not None`. Con la variable vieja el endpoint
devuelve **404 y la traza nunca se escribe, sin ningún error visible**.

La configuración correcta es el flag CLI `--profiler-config`
(`vllm/config/profiler.py:ProfilerConfig`):

```
--profiler-config '{"profiler": "torch", "torch_profiler_dir": "/traces/<label>",
                    "torch_profiler_with_stack": false, "ignore_frontend": true,
                    "delay_iterations": 4, "max_iterations": 40}'
```

`delay_iterations` saltea el prefill y `max_iterations` acota a N pasos, así la
traza es **decode puro** en vez de mezclar las dos fases. No había equivalente
con la variable vieja.

Está cableado en `docker-compose.pn110-cg.yml` y `docker-compose.pn110-ab.yml`,
parametrizado por `PROF_KIND` / `PROF_DIR` / `PROF_DELAY` / `PROF_MAX`, e
**inerte por defecto**: sin `PROF_KIND` se renderiza `"profiler": null` y el
router no se monta, así que no mete overhead en los benchmarks normales.

Ojo con el default de `torch_profiler_dir`: tiene que ser cadena vacía. Con
`profiler: null` y un dir no vacío, pydantic rechaza la config y el server no
arranca (`profiler.py:146`).

### 1.2 El primer request tiene que ser LARGO

`flashinfer/prefill.py:2302-2311`: `_max_total_num_rows` **no es un límite
configurado, es un latch**. Se fija con el `total_num_rows` de la PRIMERA
llamada a `plan()` en modo cudagraph y queda congelado.

Si el primer request es `"hola"` (~2 filas de prefill) el latch queda en 2, y el
drafting de MTP K=3 —que necesita 1+3 = **4** filas— revienta:

```
ValueError: The total number of rows in qo_indptr 4 in cuda graph mode
cannot exceed the number of rows set during initialization
```

y mata el EngineCore. A/B sobre el mismo server, con PN110 apagado:

| orden | resultado |
|---|---|
| largo (~30 tok) → `"hola"` | ambos OK, engine vivo |
| `"hola"` → largo | muere en el primero |

Descartados como causa: el profiler (probado con y sin), el compose (pasa en los
dos) y el largo del prompt por sí solo. **Es el orden.**

Esto es un bug de producción, no del script: un server recién arrancado se cae
si el primer mensaje de un usuario es corto. `bench_pn110.py` nunca lo detectó
porque usa prompts de 24 tokens.

### 1.3 Comandos

```bash
./perfil_decode.sh <label> <PN110_ENABLE> <PN110_SK> [HYBRID] [SWAP_ONLY_SK]
./perfil_decode.sh off 0 0 0 1     # baseline Marlin
./perfil_decode.sh sk1 1 1 0 1     # PN110 con super kernels
COMPOSE_PERFIL=docker-compose.pn110-ab.yml ./perfil_decode.sh ...  # sin cumem
```

Análisis:

```bash
python3 analiza_traza.py  traces/<label>/rank0.*.pt.trace.json.gz --top 24
python3 compara_trazas.py traces/off/rank0.*.gz traces/sk1/rank0.*.gz
```

### 1.4 Normalización

Las corridas no ven la misma cantidad de iteraciones, así que **todo se
normaliza por forward**. El contador es la anotación `user_annotation` de nombre
`execute_context_0(0)_generation_1(4)` (1 request de generación, 4 filas de
query = 1 verificado + 3 borradores de MTP).

**No usar las duraciones de `gpu_user_annotation`**: se solapan (34 anotaciones
suman 818 ms en una ventana de 521 ms), son spans anidados.

---

## 2. Resultados

### 2.1 Perfil de Marlin (baseline)

51 forwards, ventana 521.6 ms, **GPU ocupada 87.9%** (o sea ~12% de huecos aun
con cudagraphs).

| familia | ms/fwd | % GPU | lanz./fwd |
|---|---:|---:|---:|
| **Marlin GEMM** | **6.614** | **73.6%** | 141.7 |
| NCCL all-reduce (TP=2) | 1.011 | 11.3% | 48.3 |
| GEMM cutlass/cuBLAS | 0.500 | 5.6% | 17.0 |
| elementwise/copias | 0.332 | 3.7% | 148.0 |
| atención/GDN/KV | 0.330 | 3.7% | 44.7 |
| inductor fusionado | 0.200 | 2.2% | 110.3 |
| **total** | **8.987** | | |

Kernel dominante de Marlin: **p50 56.1 µs**, p90 109.7, min 22.9, max 112.2 —
claramente bimodal, dos clases de forma en la misma instanciación del template.

> Nota: esta corrida es sobre el compose `cg`. Las tablas de 2.2 son sobre `ab`
> (sin cumem), que es donde se pudo correr la comparación con las dos
> configuraciones idénticas. Los ms/fwd difieren entre 2.1 y 2.2 porque el
> batching efectivo no es el mismo; **no mezclar las dos tablas**.

### 2.2 Marlin vs PN110, por forward de decode (compose `ab`, idéntico en ambos)

| | Marlin | PN110 | delta |
|---|---:|---:|---:|
| GEMM de las capas convertidas | **15.560 ms** | **16.772 ms** | +1.212 |
| quant de activación (SK-09) | 0 | 0.549 ms | +0.549 |
| **subtotal comparable** | **15.560** | **17.321** | **+1.761 (+11.3%)** |
| Marlin residual (idéntico en ambos) | 2.825 | 2.810 | −0.015 |
| lm_head / wmma | 1.429 | 1.406 | −0.023 |
| NCCL (TP=2) | 2.955 | 3.017 | +0.062 |
| resto | 2.789 | 2.751 | −0.038 |
| **TOTAL** | **25.559** | **27.306** | **+1.747** |

Por lanzamiento, mismas formas y misma cantidad de lanzamientos (253 vs 252):

| | p50 | media |
|---|---:|---:|
| Marlin | **56.0 µs** | 61.5 µs |
| cutlass-INT8 (`enable_sm80_to_sm89`) | 59.0 µs | 66.5 µs |

### 2.3 LOS SUPER KERNELS NO SE EJECUTAN

Confirmado en **dos trazas independientes** (con cumem y sin cumem): en toda la
traza de `sk1` el único kernel Genesis presente es `_sk09_quant_kernel`. Cero
`_sk01`, `_sk02`, `_sk03`, `_sk04`, `_sk05`, `_sk06`, `_sk07`, `_sk10`.

Lo que corre con `GENESIS_PN110_SK=1` es: SK-09 cuantiza la activación y después
se llama al GEMM INT8 genérico de vLLM (`cutlass::Kernel2<enable_sm80_to_sm89>`).
**Todo el trabajo de optimización de los SK está fuera del camino de ejecución**,
y por eso los microbenchmarks (SK 53 µs vs Marlin 59 µs) nunca se tradujeron en
nada end-to-end.

---

## 3. Por qué Marlin gana en decode (y por qué es estructural)

1. **No cuantiza la activación.** Marlin es W8A16: dequantiza el peso *dentro*
   del mainloop, elemento a elemento, escondido bajo el pipeline de memoria. La
   activación queda en fp16. PN110 es W8A8 y necesita una reducción por fila
   (amax) antes de cada GEMM: 252 lanzamientos extra por forward.

2. **Su GEMM está mejor afinado para las formas de decode**: 56.0 vs 59.0 µs.

La razón de fondo: **en decode, con M≈4, estos GEMM son 100% memory-bound
leyendo pesos.** El tiempo es `bytes_de_peso / ancho_de_banda`. Marlin lee 1
byte por peso; cutlass INT8 lee 1 byte por peso. Mismos bytes, mismo tiempo. Los
~284 TOPS de INT8 de la 3090 **no sirven a M=4** porque no hay cómputo que
saturar.

> **W8A8 no puede ganarle a W8A16 en decode por diseño.** Sólo puede empatar el
> GEMM y sumarle el costo del quant. La única forma de ganar en decode es leer
> *menos bytes de peso*: W4.

---

## 4. La brecha de tok/s NO está en la GPU

Medido: **+11.3% de tiempo de GPU por forward**. La diferencia reportada de
171.6 → 97.9 tok/s (−43%) **no está en el tiempo de GPU**.

Eso descarta que la brecha sea de kernels. Sospechoso principal: la **tasa de
aceptación de MTP** — si PN110 degrada el draft head, se aceptan menos tokens
por forward y los tok/s se derrumban sin que el tiempo de GPU cambie casi nada.
Se mide con `vllm:spec_decode_num_accepted_tokens` en `/metrics`. **Hipótesis,
no medido todavía.**

---

## 5. Problema de memoria de PN110 (arreglado)

`sk1` no arrancaba bajo cumem: `CUDA Error: out of memory at
cumem_allocator.cpp:163`.

**El traceback engañaba.** Decía `unmap_and_release`, pero la línea 163 es un
`cuMemMap` dentro de `create_and_map`: `error_code` es un global pegajoso
(`CUDA_CHECK`, línea 63), así que el mensaje se imprime donde falla y la
excepción salta donde se chequea después. Era un OOM **de asignación**, no de
liberación — por eso bajar `--gpu-memory-utilization` de 0.72 a 0.62 no cambió
nada.

**Causa:** `_warmup_for_layer(k, n)` se llamaba desde
`process_weights_after_loading`, o sea **por cada capa convertida dentro de
`load_model()`**, que corre dentro del pool `weights` de cumem. Por capa y por
cada M asignaba `b_col` [K,N] int8 y `tmp` [N,K] int8 (~89 MB cada uno para
5120×17408), más activaciones, salidas y un `torch.compile`. Dentro del pool de
cumem **la memoria liberada no se devuelve** hasta salir del contexto (ver el
comentario de vLLM en `device_allocator/cumem.py` sobre "online quantization"), y
el `torch.cuda.empty_cache()` que intentaba limpiarla **lanza** dentro de un
allocator pluggable (pytorch#145168) y quedaba tragado por el `except`.

**Además ese warmup no aportaba nada:** Triton compila por `tl.constexpr` (el
tile que elige `_cfg`), no por K y N, que son argumentos de runtime.

**Arreglo:** eliminada la llamada por capa. Verificado: `sk1` arranca con cumem
y el profiler activo, escenario que antes fallaba 3 de 3.

---

# 6. Por qué no se ejecutaba ni un super kernel (resuelto)

Tres bugs encadenados. Cada uno tapaba al siguiente, y el fallback silencioso a
`cutlass_scaled_mm` hacía que todo pareciera funcionar.

## 6.1 El gate por M quedaba horneado en el trazado

El camino caliente tenía:

```python
if state["sk_max_m"] < a_i8.shape[0] < state["sk_min_big_m"]:
    sk_fn = None                      # "Diseño B": cutlass en la franja media
```

`apply` corre **dentro** de la región que `torch.compile` traza y cudagraphs
captura. Se comprobó metiendo un log con side-effect ahí: la captura aborta con
*"Assigning / modifying buffers of nn.Module during forward pass"*. Una rama
Python sobre `a_i8.shape[0]` en ese contexto se evalúa **una vez, en el
trazado**, y su resultado queda horneado: no se re-decide por forward.

Síntoma: el bind ligaba 513 capas a SK-01..SK-06/SK-10 sin un solo fallo y aun
así el 100% del GEMM se iba por cutlass. **Eliminado.** Si alguna vez se quiere
elegir por M, la decisión va en el bind, no en el forward.

## 6.2 Triton especializa por valor y M es un símbolo sin backing

Al despachar apareció:

```
torch._dynamo.exc.UserError: Could not guard on data-dependent expression Eq(u0, 1)
Caused by: _sk01_gdn_qkvz_kernel[grid](...)
```

Triton hornea una variante por argumento entero según si vale 1 y si es múltiplo
de 16. `M` viene de una dimensión dinámica que dynamo traza como `u0`.

`@triton.jit(do_not_specialize=["M"])` **NO alcanza** — probado, el guard sigue.

**Solución:** registrar los GEMM como custom op (`kernels/sk_ops.py`), como hace
vLLM con sus propios kernels Triton. Dynamo no traza adentro, así que `M` llega
como `int` concreto. Esto además es lo que **habilita** elegir tile por M —o
sea kernels distintos para prefill y decode—, que sin el custom op era
imposible en el camino compilado.

## 6.3 Dos bugs de correctitud en los kernels, nunca ejecutados hasta ahora

**Indexado de shifts fuera de rango** (8 kernels, 10 sitios). El bucle hacía
`sh_ptrs + kb * stride_shift_k`, pero los shifts tienen una fila cada
`SHIFT_BLOCK`(=128) filas de K mientras `kb` cuenta pasos de `BLOCK_K`. Sólo era
correcto con `BLOCK_K == 128`; con `BLOCK_K=64` (buckets de M>128) `kb` llegaba
a `K/64-1` leyendo `K/128` filas **fuera del tensor** -> `illegal memory access`.
Corregido a `(kb * BLOCK_K // SHIFT_BLOCK)`.

**Corrimiento a izquierda por cantidad negativa** (6 kernels). El requantizador
produce shifts **<= 0** por construcción (`need = log2(amax/(127*s_row))`, el
código lo comenta). Los kernels hacían `d << shift`; en PTX `shl` toma los 5 bits
bajos, así que `-1` se vuelve `31`. **El camino diádico estaba roto en 6 de 8
kernels.** Corregido a corrimiento aritmético a derecha de `-shift`, con medio
LSB para redondear al más cercano en vez de truncar (el sesgo de truncar se
acumula sobre los ~40 bloques de K y siempre tira hacia cero). Todo entero.

> **PENDIENTE:** el redondeo entero es una aproximación distinta al `exp2` fp32
> que usan SK-05 y SK-07. Necesita validación de perplejidad end-to-end.

**Invariante nueva, documentada en los 8 kernels:** `BLOCK_K <= SHIFT_BLOCK`. Un
`tl.dot` cubre `BLOCK_K` filas con UN shift; con `BLOCK_K=256` abarcaría dos
bloques diádicos distintos y daría números mal **sin fallar**. Los tiles con
`BK=256` miden 1-4% mejor en el banco y no son válidos.

## 6.4 Verificación

`prueba_sk.py`: los 8 kernels contra referencia fp32 bloque a bloque, con shifts
**negativos** (el caso real) y M ∈ {1,4,40,256,2048}. **TODOS OK.** Antes de
estos arreglos, ninguno daba resultado correcto en el camino diádico.

## 6.5 Resultado

| configuración | ms/forward |
|---|---:|
| Marlin (baseline) | 25.559 |
| PN110 con cutlass (antes) | 27.306 |
| **PN110 con super kernels** | **25.568** |

Sólo el GEMM de las capas convertidas:

| | ms/fwd |
|---|---:|
| Marlin | 15.560 |
| cutlass INT8 | 16.772 |
| **super kernels** | **15.088** |

Los super kernels son **3% más rápidos que Marlin** en el GEMM. La brecha que
queda es exactamente el quant de activación (0.561 ms/fwd).

---

# 7. Hipótesis probadas y DESCARTADAS

Se anotan para no volver a probarlas.

**Las fusiones de inductor explican la brecha.** No: 0.200 ms/fwd, 2.2% del
total. Aunque PN110 las pagara enteras por separado, no alcanza.

**Bajar BLOCK_N sube los CTAs y por eso acelera el decode.** Mayormente no. A
M=4 varias capas quedan en 40 CTAs sobre 82 SMs, pero llenar los SMs **no**
mejora: la config actual (BN=128) es la mejor en 4 de 6 formas. La ocupación no
es el limitante. Única excepción medida: `down` (K=8704, N=5120), donde
`(16,64,128,8,4,4)` da 67.6us contra 70.7us = **4.4% mejor**. Aplicado sólo ahí.

**Más CTAs aceleran el quant de activación.** No: la variante 2-D es **peor**
(0.73-0.88x) porque lee la fila dos veces. El kernel actual (grid=(M,)) ya es
casi óptimo: 1.96-3.03us según K.

**`do_not_specialize=["M"]` arregla el guard de Triton.** No, ver 6.2.

# 8. Trampa de medición recurrente

Medir un kernel de ~2us en **eager** da ~11us para todo, porque el lanzamiento
domina, y todas las variantes parecen idénticas. Hay que medir **dentro de un
CUDA graph**. Ya causó un error de diagnóstico en esta sesión (se reportó 21% de
costo de quant donde el server real tiene 4%) y otro en el banco de quant.
`bench_quant.py` ya mide con grafo.

---

# 9. Tabla de tiles ajustada a la carga real (2026-08-27)

## 9.1 Cuál es el M real

Con MTP k=3 cada request aporta **4 filas** al GEMM (1 verificado + 3
borradores). El server corre **4-6 requests en paralelo de forma típica, 10 como
máximo**, así que:

| requests | M con MTP | M sin MTP |
|---:|---:|---:|
| 1 | 4 | 1 |
| 4 | 16 | 4 |
| 5 | 20 | 5 |
| 6 | 24 | 6 |
| 10 | 40 | 10 |

El rango que importa es **M = 16..40**, típico **16..24**. La tabla vieja tenía
un solo escalón en M=32 y mandaba todo 33..128 a `BLOCK_M=128`.

## 9.2 Medido (qkv K=5120 N=7168, relojes fijos 1500 MHz)

| requests | M | tabla vieja | tabla nueva | ganancia |
|---:|---:|---:|---:|---:|
| 1 | 4 | 59.4 us | 59.4 us | — |
| 4 | 16 | 70.7 us | **65.5 us** | 7% |
| 5 | 20 | 70.7 us | **67.6 us** | 5% |
| 6 | 24 | 72.2 us | **67.6 us** | 6% |
| 8 | 32 | 69.1 us | 67.6 us | 2% |
| 10 | 40 | 116.7 us | **77.8 us** | **1.50x** |

En `down` (K=8704, N=5120) a M=20: 95.2 -> 84.0 us (**13%**).

El cruce de `BLOCK_M` 16->32 está **entre M=16 y M=20**: a M=16 gana 16 por 6%,
a M=20 gana 32 por 8%.

## 9.3 Dos límites que NO son intuitivos

**El desperdicio de MMA es barato; releer B no.** Primero se calculó "desperdicio
de filas" y sugería `BLOCK_M=16` para todo el rango medio. Es al revés: con
`BLOCK_M=16` a M=128 hacen falta 8 tiles en M y son **8 pasadas completas sobre
el peso** — 210.9 us contra 116.7 us con `BLOCK_M=128`, que "desperdicia" filas.

**`BLOCK_M` debe ser potencia de dos.** `BM=48` (que cubriría M=40 sin
desperdicio) muere con `CompilationError`. Por eso M=40 usa `BM=64`; partirlo en
dos tiles de 32 obliga a releer B (107.5 us contra 77.8 us).

## 9.4 Estado del bucle tras las optimizaciones de PTX

| | registros | spills | instr/iter | MMA | útil |
|---|---:|---:|---:|---:|---:|
| SK05 decode antes | 73 | 0 | 112 | 8 | 7.1% |
| SK05 decode ahora | **62** | 0 | **95** | 8 | 8.4% |
| SK05 prefill antes | 255 | **16** | 612 | 64 | 10.5% |
| SK05 prefill ahora | 255 | **0** | **524** | 64 | 12.2% |
| SK06 decode antes | 79 | 0 | 116 | 8 | 6.9% |
| SK06 decode ahora | **64** | 0 | **95** | 8 | 8.4% |

Cambios: shift **escalar** en vez de vector (de 4 `ld.global.b8` por iteración a
1 — antes eran 160 viajes a global por kernel para leer 40 bytes constantes), y
SK-05 pasado al **acumulador entero** como el resto (se fueron 28 instrucciones
fp32 por iteración en decode y 384 en prefill, y con ellas los 16 spills).

## 9.5 Régimen: memoria o instrucciones

| | modelo emisión | techo DRAM | medido |
|---|---:|---:|---:|
| SK05 decode M=4 | 37.0 us | **95.2 us** | 105.0 us |
| SK06 decode M=4 | 15.3 us | **47.6 us** | 67.6 us |

A **M=4 el kernel está limitado por memoria** (90% del techo de DRAM): podar
instrucciones no cambia casi nada. A **M=20-40 estamos al 37-58% del techo**, o
sea que ahí las instrucciones sí pesan. Por eso la tabla de tiles se ajustó al
rango real y no a M=4.

En prefill el modelo da 2.74 ms contra 1.29 ms de techo de tensor cores: **2.1x**,
limitado por emisión. Ahí cada instrucción que salga del bucle se paga directa.

---

# 10. Estado al 2026-08-27 (fin de sesión larga)

## 10.1 El número que importa

Todo medido con el mismo método (5 requests concurrentes, 200 tokens, `vllm:iteration_tokens_total`):

| | Marlin | PN110 | |
|---|---:|---:|---:|
| **prefill** | 1438 tok/s | **1860** | **+29%** |
| decode sin MTP | **104.5** | 98.7 | −5.6% |
| decode con MTP | **169.0** | 157.9 | −6.6% |

**PN110 = +29% en prefill por −6% en decode.** Para carga agéntica (contextos
largos, subagentes que desalojan la caché) el prefill pesa más.

## 10.2 De dónde sale el −6% (traza del profiler)

| | ms/fwd |
|---|---:|
| GEMM Marlin | 16.726 |
| GEMM super kernels | 16.840 (+0.7%, **empate**) |
| quant de activación SK-09 | **0.648** |

Los GEMM empatan. **Todo el déficit es el quant**, que existe sólo porque el
esquema es W8A8. En decode no compra nada: a M=4..40 el kernel espera pesos.

## 10.3 Por qué no se puede arreglar barato

* **Phase dispatch con Marlin en decode: NO ENTRA EN VRAM.** Hacen falta las dos
  representaciones del peso (Marlin 8 bits + INT8 8 bits) = 24 GB sobre 24.
* **Reimplementar Marlin en Triton (kernel W8A16): MEDIDO, NO ALCANZA.** Ver 10.4.
* **La aceptación de MTP NO es el problema** (hipótesis descartada): 6.91 vs 7.32
  tokens/forward = 5.6%, o sea la misma brecha de decode propagada.

## 10.4 W8A16 decode — experimento completo (`sk_decode_w8a16.py`)

Peso INT8 dequantizado en el mainloop, activación fp16 sin cuantizar. Arco de
optimización guiado por PTX:

| paso | qkv M=4 |
|---|---:|
| primera versión | 89.3 us |
| activación fp16 sin convertir (bf16→fp32→fp16 costaba 96 instr/iter) | 62.6 us |
| `PRMT`+`LOP3` empaquetado | **52.8 us** |
| W8A8 de referencia | 53.1 us |

**Empata a M=4, pierde 1.1-1.8x a M=20-40** (el rango real). Razón: W8A16
dequantiza el tile de B en CADA iteración y ese costo no depende de M; W8A8
cuantiza A una sola vez. Al crecer M, W8A8 mejora y W8A16 no. **No se cableó.**

## 10.5 Herramientas de asm — las tres validadas

| | qué resuelve | estado |
|---|---|---|
| `tl.inline_asm_elementwise` | lo elementwise dentro del kernel | validado bit a bit |
| `ptx_launcher.guardar_ptx` | volcar el asm real (1153 líneas / GEMM) | OK |
| `ptx_launcher.KernelPTX` | cargar y lanzar asm editado | salida idéntica al JIT |

**int8 → fp16 (truco de Marlin), 1.5 instr/valor:**
```ptx
prmt.b32  lo, $2, 0, 0x4140;   prmt.b32  hi, $2, 0, 0x4342;
lop3.b32  $0, lo, 0x00800080, 0x64006400, 0xBE;   // (x^0x80)|0x6400
sub.f16x2 $0, $0, c;                              // c = 0x64806480 = 1152.0
```
fp16 tiene 10 bits de mantisa -> un int8 entra exacto. `0x6400|u` ES el fp16 de
`1024+u`. El XOR con 0x80 es el `+128`. LUT `0xBE` = `(a^b)|c`.

**nibbles W4A8, 4 instr para 8 pesos:**
```ptx
mov.b32 m, 0x0f0f0f0f;  lop3.b32 $0, $2, m, 0, 0xC0;
shr.b32 t, $2, 4;       lop3.b32 $1, t, m, 0, 0xC0;
```

Detalles de la API que costaron encontrar:
* `load_binary` devuelve **5** valores, y el device no puede ser `None`.
* `run(gridX,gridY,gridZ,stream,function,kernel_metadata,launch_metadata,enter,exit,*args)`
  es **posicional estricto**; `stream=` por keyword choca con los `*args`.
* La metadata es la **empaquetada** (`packed_metadata`), no `h.metadata`.
* **`vsub4` NO compila** en sm_86: `Internal Triton PTX codegen error`.

## 10.6 Lo que sigue, en orden de valor esperado

1. **Cablear W4A8.** Es lo ÚNICO que rompe el empate del GEMM en decode, porque
   lee la mitad de los bytes de peso. 7 kernels ya escritos y validados, sin
   cablear. El `LOP3` de 10.5 les aplica directo.
2. **Validar calidad end-to-end.** El camino diádico cambió de numérica
   (redondeo entero en vez de `exp2` fp32). Los kernels dan correcto contra
   referencia fp32, pero **no se midió perplejidad**.
3. **DFlash2 como parche Genesis.** ~690 líneas, ~505 en archivos nuevos. NO hay
   que tocar el `Literal` de pydantic (se elige por arquitectura del draft).
   v0.27.1 ya tiene `v1/worker/gpu/spec_decode/` con `dflash/` y `dspark/`.
   Cuidado con `gumbel.py`: toca muestreo compartido por TODOS los métodos.

## 10.7 NADA DE ESTO ESTÁ COMMITEADO

## 11. Analisis del PTX del bucle de SK-05 (2026-08-27)

Compilado con la geometria de produccion (K=5120, N=17408, M=20 -> tile
32x64x128, 4 warps, 4 stages). `n_regs=95`, `shared=36864`, 0 spills.

### 11.1 Mezcla del bucle interno (`$L__BB0_3`, 143 instrucciones)

| clase                      | n  | %   |
|----------------------------|---:|----:|
| `add.s32`                  | 34 | 24% |
| `shr.s32` + `shl.b32`      | 20 | 14% |
| `mov.b32`                  | 16 | 11% |
| `mma.sync…s8.s8.s32`       | **16** | **11%** |
| `cvt.rn.f32.s32`           | 16 | 11% |
| `ldmatrix.x4`              | 12 |  8% |
| `cp.async`                 |  9 |  6% |
| resto                      | 20 | 14% |

El acumulador es 32x64/128 hilos = 16 elementos por hilo, asi que casi todos
los grupos de 16 son "una operacion por elemento del acumulador".

### 11.2 Falsa pista descartada: layout de B

El primer volcado dio 64 `ld.shared.b8` + 48 `prmt.b32` (41% del bucle): Triton
leyendo un operando de shared byte por byte porque `mma…row.col` quiere los dos
operandos contiguos en K y `ldmatrix.trans` no existe en int8. Medido con B
transpuesto fisicamente: 206 -> 80 instrucciones de bucle y 1.22-1.37x.

**No aplica.** El banco pasaba `B [K,N]` contiguo (strides `(N,1)`), que no es
lo que corre. `requant_inplace.py:156` devuelve `bits.t()` con `bits` `[N,K]`
row-major, o sea `b_col` ya tiene strides `(1,K)` = contiguo en K. Con los
strides reales el bucle emite 12 `ldmatrix` y **cero** `ld.shared.b8` / `prmt`.
Produccion ya estaba en el layout bueno.

### 11.3 Costo del shift diadico

SK-05 real contra el mismo GEMM sin shift ni epilogo, mismos tiles y geometria:

| M  | SK-05 | GEMM pelado | costo |
|---:|------:|------------:|------:|
| 16 | 130.0 | 131.1 | ~0 |
| 20 | 131.6 | 119.3 | 10% |
| 32 | 128.0 | 118.8 |  8% |
| 40 | 158.7 | 138.2 | 15% |

### 11.4 CORREGIDO — no habia desperdicio en el bucle

Version anterior de esta seccion afirmaba que 16 `cvt.rn.f32.s32` estaban
hundidos en el bucle. **Era un error de medicion.** Yo tomaba "el bloque de
PTX mas grande entre dos labels" como cuerpo del bucle, y eso se traga el
cuerpo MAS el bloque de salida. El salto de retorno de SK-05 M=20 esta en la
linea 541 (`@%p7 bra $L__BB0_3`) y los `cvt` estan en 544-559, o sea DESPUES:
son la salida, se ejecutan una vez.

Regla para no repetirlo: **el bucle termina en el salto de retorno, no en el
label siguiente.** El extractor correcto esta en `scratchpad/mezcla.py`.

Con el limite bien puesto, el bucle de SK-05 a M=20 son 132 instrucciones
(no 143) y **cero** `cvt.rn` adentro.

Queda en pie el otro punto: el `ld.global.b8` del shift esta en el camino
critico (`ld -> cvt.s32.s8 -> neg.s32 ->` los 16 `shr`), sin pipeline, porque
Triton no mete cargas escalares en `cp.async`.

### 11.5 Lo que si se aplico: ocupacion por shared

`shared = (BM+BN)*BK*(stages-1)` = 36864 B -> 100 KB/36 KB = **2 CTAs/SM**
= 8 warps de 48 = **17% de ocupacion**, con el kernel a 72% del techo de DRAM.
Pocos warps en vuelo para tapar latencia. Con `num_stages=3` el shared baja a
24576 B -> 4 CTAs/SM.

Medido intercalado round-robin (mediana de 60), stages 4 -> 3:

| kernel            | M=16 | M=20 | M=32 | M=40 |
|-------------------|-----:|-----:|-----:|-----:|
| SK-05 (N=17408)   | **1.095** | **1.060** | **1.112** | 0.835 |
| SK-06 (N=5120)    | 1.000 | 1.031 | 0.987 | 1.031 |
| SK-03 (N=7168)    | 0.983 | 1.000 | 0.953 | 0.986 |

Aplicado **solo a SK-05, solo en los escalones M<=16 y M<=32**. La ganancia
depende de tener muchos CTAs (SK-05 = 272) y desaparece o se da vuelta en los
demas. El sweep en tandas daba numeros muy distintos (la misma config, 175.1 y
199.7 us en dos corridas): a M=40 hay que medir intercalado o no medir.

## 12. PTX de SK-03 y SK-06 (2026-08-27)

Medidos con el extractor corregido (cuerpo = header .. salto de retorno).
SK-03: K=5120, N=7168, 112 CTAs. SK-06: K=8704, N=5120, 80 CTAs.
Ambos con `HAS_SHIFT=False`, que es el default de produccion (Diseno B).

### 12.1 SK-03 y SK-06 tienen el bucle IDENTICO

| tile   | instr | mma | ldmatrix | ALU | cvt.rn |
|--------|------:|----:|---------:|----:|-------:|
| 16x64  |    65 |   8 |        8 |  31 |      0 |
| 32x64  |    79 |  16 |       12 |  32 |      0 |
| 64x64  |   105 |  32 |       16 |  36 |      0 |

La geometria (K, N) **no cambia el cuerpo del bucle**, solo el numero de
vueltas y de CTAs. El cuerpo depende unicamente del tile. Por eso alcanza con
analizar un tile por kernel, no un (kernel, geometria).

Cero `cvt.rn` en los tres: el epilogo `acc.to(tl.bfloat16)` queda entero en el
bloque de salida. No hay nada que rescatar ahi.

### 12.2 SK-05 paga 53 instrucciones mas por iteracion que SK-03/06

Mismo tile 32x64, mismas 16 `mma`:

| kernel | instr | mma      | ALU      |
|--------|------:|---------:|---------:|
| SK-03  |    79 | 16 (20%) | 32 (40%) |
| SK-06  |    79 | 16 (20%) | 32 (40%) |
| SK-05  |   132 | 16 (12%) | 64 (48%) |

La diferencia es exactamente el shift diadico: SK-03/06 lo compilan afuera con
`HAS_SHIFT` constexpr; **SK-05 y SK-07 no tienen esa rama** y hacen 32
operaciones enteras por iteracion para aplicar un shift que en produccion
vale cero.

### 12.3 Cuanto vale `HAS_SHIFT=False`, por geometria

Kernel de SK-03, mismo codigo, `HS=True` vs `HS=False`, intercalado, mediana 60:

| geometria              | M=16  | M=20  | M=32  | M=40  |
|------------------------|------:|------:|------:|------:|
| SK-05 (K=5120,N=17408) | 0.933 | 1.059 | 1.047 | 0.986 |
| SK-03 (K=5120,N=7168)  | 1.089 | 1.203 | 1.085 | 1.145 |
| SK-06 (K=8704,N=5120)  | 1.172 | 1.156 | 1.152 | 1.270 |

**La palanca es distinta segun el kernel.** SK-03 y SK-06 estan limitados por
instrucciones (37 y 45 MB de peso, 1.0-1.4 olas) y sacar el shift les da
1.09-1.27x. SK-05 lee 89 MB, corre a 72% del techo de DRAM y esta limitado por
memoria: sacarle instrucciones no compra nada, y a M=16 y M=40 hasta "pierde"
(o sea, esta dentro del ruido del plateau memory-bound).

**Decision: NO se le agrega `HAS_SHIFT` a SK-05.** No lo justifica la medicion.
Para SK-05 la palanca es la ocupacion (seccion 11.5), no las instrucciones.

### 12.4 Descartado: el kernel de SiLU no es el lastre

El docstring de sk05 dice que la version fp32 con `exp2` en el bucle daba "255
registros CON derrame". `_sk05_gateup_silu_kernel` sigue con acumulador fp32 y
`exp2` adentro, asi que la hipotesis era que el flojo +0.6% de P113 salia de
ahi. Medido, no:

| tile  | GEMM int32                  | SiLU fp32                    |
|-------|-----------------------------|------------------------------|
| 16x64 | 94 instr, 63 regs, 0 spill  | 100 instr, 56 regs, 0 spill  |
| 32x64 | 131 instr, 95 regs, 0 spill | 137 instr, 96 regs, 0 spill  |
| 64x64 | 204 instr, 164 regs, 0 spill| 209 instr, **128** regs, 0 spill |

~5% mas de instrucciones, sin derrame en ningun punto, y a 64x64 usa MENOS
registros que el GEMM entero. Los "255 con derrame" describen una version que
ya no existe. El +0.6% de P113 hay que explicarlo en otro lado.

## 13. PTX de SK-01 y SK-02, y un defecto del protocolo de medicion (2026-08-27)

### 13.1 SK-01, SK-02, SK-03 y SK-06 son EL MISMO kernel

Cuerpo del bucle, mismo tile -> mismas instrucciones, byte por byte:

| tile   | instr | mma | ALU | regs | shared |
|--------|------:|----:|----:|-----:|-------:|
| 16x64  |    65 |   8 |  31 |   56 |  30720 |
| 32x64  |    79 |  16 |  32 |   64 |  36864 |
| 64x64  |   105 |  32 |  36 |  103 |  49152 |

La geometria (K, N) no toca el cuerpo: solo cambia vueltas, CTAs y bytes.
**En el asm de SK-01/02/03/06 no queda nada especifico por optimizar.** El
unico eje es la eleccion de (tile, geometria).

| kernel | K    | N     | vueltas | CTAs | olas | MB   | x/fwd |
|--------|-----:|------:|--------:|-----:|-----:|-----:|------:|
| SK-01  | 5120 |  8192 |      40 |  128 | 1.56 | 41.9 |    48 |
| SK-02  | 3072 |  5120 |      24 |   80 | 0.98 | 15.7 |    48 |
| SK-03  | 5120 |  7168 |      40 |  112 | 1.37 | 36.7 |    16 |
| SK-06  | 8704 |  5120 |      68 |   80 | 0.98 | 44.6 |    64 |

### 13.2 EL DEFECTO: medir con un sync por lanzamiento infla hasta 44%

SK-02 parecia el patito feo: 33.8 us = 49.7% del techo de DRAM, contra 67-70%
de los otros tres. Y ningun tile lo movia — todo empataba en 32.8-33.8 us sin
importar BLOCK_M, BLOCK_N, warps ni stages.

La causa no era el kernel. El protocolo de todo el proyecto mide **cada
lanzamiento aislado, con `torch.cuda.synchronize()` alrededor**. Eso agrega
~10 us fijos por llamada:

| K     | vueltas | aislado | rafaga | GB/s rafaga | inflacion |
|------:|--------:|--------:|-------:|------------:|----------:|
|  3072 |      24 |    33.8 |   23.5 |         670 |    1.44x  |
|  6144 |      48 |    52.2 |   42.2 |         746 |    1.24x  |
| 12288 |      96 |    90.1 |   82.2 |         765 |    1.10x  |

SK-02 real: 23.5 us, 670 GB/s, **71% del techo — en linea con los otros tres**.
El 49.7% era artefacto. Y el costo fijo comprime toda razon A/B hacia 1, que es
exactamente por que "todo empataba".

En produccion estos GEMM corren back-to-back dentro de un cudagraph. El regimen
valido es la rafaga, no el lanzamiento aislado.

### 13.3 Pero la rafaga secuencial TAMBIEN miente

Rehecho el barrido de SK-02 en rafaga (7 rafagas de 300 por config, en tandas),
ganaba `BLOCK_N=128, stages=3` por 1.11-1.19x, con MENOS CTAs (40, media ola).
Confirmado en **rafagas ALTERNADAS** A,B,A,B: se da vuelta entero.

| M  | actual (32,64,S4) | propuesto (32,128,S3) | gana  |
|---:|------------------:|----------------------:|------:|
| 20 |          24.60 us |              26.29 us | 0.936 |
| 32 |          24.45 us |              26.85 us | 0.911 |
| 40 |          28.61 us |              35.81 us | 0.799 |

Corriendo las rafagas de cada config en tandas, la L2 queda caliente con el
patron de esa config. **Solo el A/B alternado sirve.**

**SK-02 queda como esta.** La tabla actual es correcta.

### 13.4 Descartado: split-K en SK-02

Con 0.98 olas la hipotesis era falta de paralelismo. Prototipado con
`tl.atomic_add`, empeora monotonamente: SPLIT=1 42.0 us, 2 -> 43.0, 4 -> 53.2,
8 -> 68.6. Y SPLIT=1 con BLOCK_M=16 (160 CTAs, 1.95 olas) da exactamente lo
mismo que con 80 CTAs. Mas CTAs, mas olas y mas warps no mueven nada: el limite
no es el paralelismo.

### 13.5 Re-validado: el `num_stages=3` de SK-05 vale MUCHO mas de lo medido

Re-medido en rafagas alternadas de 200 (mediana de 9):

| M  | S=4 (antes)          | S=3 (aplicado)       | gana  |
|---:|---------------------:|---------------------:|------:|
| 16 | 167.97 us (56.7%)    | 115.28 us (82.6%)    | **1.457** |
| 20 | 187.46 us (50.8%)    | 133.94 us (71.1%)    | **1.400** |
| 32 | 179.06 us (53.2%)    | 143.54 us (66.3%)    | **1.247** |

Aislado daba 1.06-1.11x. Real: **1.25-1.46x**. En rafaga la ocupacion pesa
mucho mas, porque la cola de un lanzamiento se solapa con la cabeza del
siguiente y con 2 CTAs/SM no hay con que solaparla. SK-05 corre 64 veces por
forward: es la ganancia mas grande encontrada hasta ahora.

**Pendiente: re-verificar en rafaga alternada TODA la tabla `_CFG` de los 8
kernels.** Fue tuneada con el protocolo aislado.

## 14. Ciclo Triton -> PTX -> editar -> correr, y los 8 kernels volcados

### 14.1 Faltaba el ensamblador

`ptx_launcher.py` cargaba desde `cubin`, o sea que solo podia relanzar lo que
ya habia producido el JIT: **el ciclo estaba cortado en el ultimo paso**, no
habia forma de correr un PTX editado a mano.

Agregado en `ptx_launcher.py`:

  * `compilar_ptx(ptx, arch=None, opt=3) -> bytes` — ensambla con el `ptxas`
    que trae Triton (`triton/backends/nvidia/bin/ptxas`), no el del sistema:
    Triton emite una version de ISA concreta y el ptxas del CUDA instalado
    puede no aceptarla. Con `opt=0` el asm sale tal cual se escribio, util para
    ver si una edicion a mano sobrevive; con `opt=3` ptxas reordena y puede
    deshacerla.
  * `desde_ptx(ptx, handle, opt=3) -> KernelPTX` — ensambla y devuelve
    lanzable. `handle` aporta nombre de funcion, shared y la plomeria del
    launcher: se reemplaza el CUERPO, no el contrato (firma y grid iguales).

Validado en SK-01 y SK-02:

  * roundtrip PTX -> ptxas -> cargar -> lanzar: **identico bit a bit** al JIT
    (`torch.equal`), mismos regs (64), 0 derrames.
  * edicion real `cvt.rn.bf16.f32` -> `cvt.rz`: cambia 135506/163840 (SK-01) y
    82956/102400 (SK-02) elementos. La edicion **llega efectivamente a la GPU**.

### 14.2 Volcador

`tools/volcar_ptx_sk.py` vuelca una variante por bucket de M para los 8
kernels. Salida en `assets/ptx_sk/` (1.3 MB, 32 archivos, regenerable — no hace
falta versionarlo). Nombre:
`<SK>_M<bucket>_<BM>x<BN>x<BK>_w<warps>_s<stages>.ptx`.

### 14.3 Seis de los ocho kernels tienen el bucle IDENTICO

Cuerpo del bucle, tile 32x64 (M=20) y 64x64 (M=40):

| kernel | 32x64 instr | 64x64 instr | mma | ALU | cvt en bucle | regs |
|--------|------------:|------------:|----:|----:|-------------:|-----:|
| SK-01  |          79 |         105 |  16 |  32 |            0 |   64 |
| SK-02  |          79 |         105 |  16 |  32 |            0 |   64 |
| SK-03  |          79 |         105 |  16 |  32 |            0 |   64 |
| SK-04  |          79 |         105 |  16 |  32 |            0 |   64 |
| SK-06  |          79 |         105 |  16 |  32 |            0 |   64 |
| SK-10  |          79 |         105 |  16 |  32 |            0 |   64 |
| SK-05  |     **131** |     **204** |  16 |  64 |            0 |   95 |
| SK-07  |     **148** |     **244** |  16 |  32 |    **20/40** |  114 |

Los seis primeros son literalmente el mismo bucle. Ninguno tiene `ld.shared.b8`
ni `prmt` (el layout de B es el bueno en todos), ni `cvt` adentro, ni derrames.
**En el asm de esos seis no queda nada por optimizar**: 12 `ldmatrix` +
16 `mma` + 32 enteras, que es el minimo para el tile.

### 14.4 SK-05 y SK-07 son los dos que quedaron atras

  * **SK-05**: tiene acumulador int32 pero **no** tiene `HAS_SHIFT`, asi que
    hace 32 operaciones enteras por iteracion para aplicar un shift que en
    produccion vale cero. Medido: no se justifica agregarselo (seccion 12.3).
  * **SK-07**: no tiene ninguno de los dos. Sigue con el diseno fp32 viejo,
    `acc` fp32 y `.to(tl.float32) * tl.exp2(shift)` dentro del bucle — el que
    el docstring de sk05 dice haber eliminado por costar 63% del bucle y 255
    registros con derrame. De ahi salen los 20/40 `cvt.rn` y los 114 regs.

Migrar SK-07 al patron int32 + `HAS_SHIFT`, A/B en rafagas alternadas, M=20:

| S (vocab) | CTAs | fp32 (actual) | int32 | gana  |
|----------:|-----:|--------------:|------:|------:|
|      8192 |  128 |      69.2 us  |  65.4 | 1.059 |
|     16384 |  256 |     149.4 us  | 144.3 | 1.035 |
|     32768 |  512 |     340.2 us  | 259.7 | 1.310 |
|    124160 | 1940 |    1137.4 us  |1094.3 | 1.039 |

**4-6%, con el 1.310 fuera de serie (ruido).** Las 87% de instrucciones de mas
no se traducen en tiempo: SK-07 mueve 42-636 MB y esta dominado por DRAM.
Misma leccion que SK-05. **Baja prioridad**, pero corresponde hacerlo por
consistencia (es el unico kernel con fp32 en el bucle).

## 15. Revision de lo hecho fuera de sesion (2026-08-27)

### 15.1 La conversion a asm NO esta en el codigo

`grep -c inline_asm_elementwise` en los 8 super kernels INT8 y los 7 `_w4a8`:
**cero en los quince**. Lo que matchea "mma.sync" / "ldmatrix" en esos archivos
son menciones en los docstrings, no asm.

Lo unico con asm real:

  * `fused_quant_ptx.py` — CUDA C++ con `asm volatile`, y solo el kernel de
    **cuantizacion**, no los GEMM. Ademas el asm envuelve operaciones triviales
    (`cvt`, `abs`, `max`, `mul`, `rcp`) que el compilador emite igual.
  * `sk_decode_w8a16.py` — `tl.inline_asm_elementwise` con `prmt`+`lop3`+
    `sub.f16x2`, sin cablear.

Y segun las secciones 12-14, en esos bucles **no hay nada que ganar**: seis de
los ocho kernels emiten 12 `ldmatrix` + 16 `mma` + 32 enteras, sin `ld.shared.b8`,
sin `prmt`, sin `cvt` y sin derrames. Es el minimo para el tile.

### 15.2 `bench_decode_tiles.py` esta roto: mide un kernel que corrompe memoria

Le pasa al kernel `(a, b, a_s, b_s, sh, epi, out)` cuando la firma es
`(a, b, out, resid, a_scale, b_scale, shifts)`. **Los siete punteros corridos**,
y los strides tambien.

Efecto medido: `out_ptr` apunta a `a_s` (20 floats = 80 bytes) con los strides
de `sh` = (56, 1), o sea indice maximo 8231. Escribe ~16 KB fuera de rango,
**no lanza excepcion**, pisa las escalas, deja `out` lleno de NaN — y el bench
cronometra eso y publica un ranking de tiles con "% del techo" y "olas".

**Cualquier tabla de tiles tuneada con este bench es invalida.** Quedo desfasado
respecto de la reescritura de firmas del commit c8d07af.

### 15.3 `bench_quant.py` es el que esta bien, y ya sabia lo de la seccion 13.2

Mide **dentro de un CUDA graph**, con N repeticiones por replay. Su docstring
nombra exactamente la trampa que yo redescubri por otro lado:

> "En eager el lanzamiento cuesta ~11us y tapa por completo un kernel de 2us:
> medir asi da 11.26us para todo y las variantes parecen identicas. Es la misma
> trampa que ya hizo dar 21% donde en el server real habia 4%."

Es el metodo correcto y ademas es el regimen de produccion. `bench_decode_tiles.py`,
en el mismo directorio, hace justo lo contrario: `record / corre / record /
synchronize` por repeticion. **Unificar sobre el metodo de `bench_quant.py`.**

### 15.4 `suite_parches.py`: los dos brazos del A/B son identicos

`_restart_with_env()` no aplica los overrides. Hace un `docker exec ... export`
que solo afecta al shell del exec, y despues un `docker restart` liso, que
restaura el env original del contenedor. El propio codigo lo dice
("docker exec NO persiste env en PROD; se requiere recrear via compose") pero
igual sigue y reporta los escenarios.

O sea que "PN110 deshabilitado" y "B3 deshabilitado" corren **con PN110 y B3
activos**: compara el baseline contra si mismo. Para que sirva hay que recrear
via compose con el env, como hace `run_ab.sh`.

### 15.5 Las tres suites no fijan seed y usan temperature 0.7

`suite_calidad` valida por substrings (`import curses`, `class Tetris`,
`rotate`, `attack`) sobre generacion no determinista, y compara longitud contra
un baseline con umbral de 70%. Es un smoke test, no una metrica de calidad: la
variacion entre corridas puede dar vuelta un hit sola. Y no mide nada numerico,
asi que **no detectaria** una perdida de precision como la de 15.6.

### 15.6 Correctitud de los 8 kernels: bien, pero el epilogo bf16 cuesta 2x

Verificados contra referencia exacta (fp32, bloque de K por bloque de K), con
shift cero y con shift real, M=20 y M=40: **todos correctos**.

Pero seis de los ocho (SK-01/02/03/04/06/10) hacen el epilogo entero en bf16:

    out = acc.to(bf16) * a_scale.to(bf16) * b_scale.to(bf16) + resid.to(bf16)

Tres redondeos encadenados. Error contra la referencia redondeada al dtype de
salida: **2.0x el piso**, contra 1.0x de SK-05, que lo hace en fp32.

A/B en SK-03 (rafagas alternadas de 200, mediana de 9), epilogo bf16 -> fp32:

| epilogo | error   | regs | shared | lineas PTX | tiempo   |
|---------|--------:|-----:|-------:|-----------:|---------:|
| bf16    | 2.0x    |   64 |  36864 |        931 | 63.59 us |
| fp32    | **1.0x**|   64 |  36864 |    **913** | **62.55 us** |

Mismos registros, mismo shared, **menos** instrucciones y **1.6% mas rapido**.
El epilogo esta fuera del bucle (una vez por CTA), asi que no cuesta nada.
El bf16 es estrictamente peor en las tres dimensiones. **Cambio recomendado.**
