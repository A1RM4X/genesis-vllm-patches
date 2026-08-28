# Traspaso — estado del trabajo de PTX monolítico

Documento de handoff. Fecha: 2026-08-28. Rama: `opt/super-kernels-int32`.
Último commit: `7195916`.

---

## 1. Qué se hizo

Los 18 archivos de super kernels de `vllm/_genesis/kernels/` pasaron de Triton
JIT a **PTX embebido como único camino ejecutable**. No hay fallback, no hay
kill-switch. El kernel Triton queda en cada archivo como función **privada que
sólo llaman los tests**.

Cada archivo se basta solo: la plomería de ensamblado (`ptxas`) y lanzamiento
(`cuModuleLoadData` / `cuLaunchKernel` por ctypes sobre `libcuda`) está
**duplicada dentro de cada uno** a propósito. Ninguno importa plomería
compartida. Son ~11.6 MB de PTX embebido en total.

Todo lo generó `tools/monolitizar.py`; nada se escribió a mano.

## 2. Estado actual: NO usable

**vLLM arranca con los kernels PTX, pero el modelo genera basura.** Ensalada de
tokens en varios idiomas, con `temperature=0`.

Verificado que la causa son los super kernels:

| configuración | resultado |
|---|---|
| `PN110_SK=0` (super kernels apagados) | **texto perfecto y coherente** |
| `PN110_SK=1` (default, super kernels activos) | **basura** |

El servidor quedó corriendo en el puerto 8390 con `PN110_SK=0`.

## 3. Lo que hay que hacer, por orden

### 3.1 PRIMERO: la basura con SK activos

Lo que ya se descartó, con medición:

* **La matemática del GEMM está bien.** Los 8 GEMM se contrastaron contra una
  referencia real (no contra Triton) en M = 1, 4, 20, 40, 105, 130, 512. El
  error es del orden del piso de bf16 (~2-3e-3). Hay tres casos marginales a
  ~4.5× el piso (sk02 y sk04 a M=40, sk06 a M=512) — sospechosos, pero eso
  degrada calidad, no destruye la salida.
* **El quant funciona**: con `PN110_SK=0` se usa el mismo `quant_per_token` de
  SK-09 y el texto sale bien.

Entonces la diferencia entre andar y no andar es **sólo el despacho del GEMM**:
`sk_gemm_op` → `_lanzar` → PTX, contra `cutlass_scaled_mm`.

**Hipótesis a probar, en este orden:**

1. La ruta SK recibe **escalas o shifts distintos** de los que usa
   `cutlass_scaled_mm`. Mirar qué pone `_build_int8_state` en
   `state["b_col"]`, `state["sk_bscales"]`, `state["sk_shifts"]` y comparar
   contra lo que consume la ruta que funciona.
2. El manejo del **epílogo/residual**. En la tabla `_SK_GEMM` el argumento
   `epi` mapea a `residual`. Verificar que no se esté sumando dos veces, o que
   no falte, sobre todo en las capas RowParallel (SK-02, SK-04, SK-06), que
   además llevan `all_reduce`.
3. El **layout de `b_col`**. Tiene que ser `[K,N]` con strides `(1,K)`, o sea
   contiguo en K. Lo produce `requant_inplace.py:156` como `bits.t()`.

**Cómo aislarlo rápido:** el dispatch por SK se puede apagar de a uno con
`GENESIS_PN110_EXCLUDE_LAYERS`. Si con una sola familia de SK activa ya sale
basura, ese es el culpable.

### 3.2 SK-05 no-finito en prefill

`sk05_gateup_gemm` devuelve NaN/Inf a **M ≥ 1200** con el tile (256,128,64).
**El Triton original hace lo mismo bit a bit**, así que es previo a la
conversión. Ese tile corre con 255 registros y 16 derrames, primer sospechoso.
Marcado `xfail` en el barrido. Detalle en `optimizacion_ptx.md` §1.

### 3.3 Los tres casos a ~4.5× el piso de bf16

sk02 y sk04 a M=40, sk06 a M=512.

### 3.4 Recién después: benchmark y optimización

Línea base a batir, medida antes de todo esto (5 requests concurrentes):

| | Marlin | PN110 |
|---|---:|---:|
| prefill | 1438 tok/s | 1860 (+29%) |
| decode sin MTP | 104.5 | 98.7 (−5.6%) |
| decode con MTP | 169.0 | 157.9 (−6.6%) |

## 4. Archivos de referencia

| archivo | qué tiene |
|---|---|
| **`optimizacion_ptx.md`** | **Plan de optimización.** Tabla de los 18 kernels con casillas para ir marcando. Qué atacar, qué NO volver a intentar (dos falsas pistas ya descartadas), techos reales medidos. |
| **`workshop/ox_alpha/tests/pn110_cudagraph/tests.md`** | 40 KB de metodología y resultados. Secciones 11-15: análisis del PTX línea por línea, el defecto del instrumento de medición, revisión del trabajo previo. |
| **`vllm/_genesis/tests/test_sk_ptx_formas.py`** | **El barrido de formas.** 201 pasan, 6 xfail, **16 segundos**. Es la red para optimizar sin levantar vLLM. |
| `tools/monolitizar.py` | El generador. Convierte un kernel de punta a punta y se autovalida. |
| `tools/plantillas/plomeria_ptx.py.tpl` | La plomería que se duplica dentro de cada kernel. |
| `tools/plantillas/ediciones_ptx.py` | Las ediciones E1 (copysign → `lop3` LUT 0xF8) y E2 (div por potencia de 2 → mul por recíproco), aplicadas por script. |

### Cómo correr el barrido

```
docker run --rm --gpus '"device=0"' --entrypoint bash \
  -v $PWD:/repo \
  -v $PWD/vllm/_genesis:/usr/local/lib/python3.12/dist-packages/vllm/_genesis:ro \
  -w /repo vllm/vllm-openai:v0.27.1 \
  -c "python3 -m pytest /usr/local/lib/python3.12/dist-packages/vllm/_genesis/tests/test_sk_ptx_formas.py -q"
```

### Cómo levantar el server

```
cd workshop/ox_alpha/tests/pn110_cudagraph
PN110_SK=0 docker compose -f docker-compose.pn110-cg.yml up   # sin SK: anda
docker compose -f docker-compose.pn110-cg.yml up              # con SK: basura
```

Puerto 8390. **El primer request tiene que ser con prompt largo**: con MTP y
cudagraphs, el primer `plan()` deja fijado `_max_total_num_rows` como un latch,
y si el primero es corto queda en 2 mientras MTP necesita 4 — el engine muere
con un error de `qo_indptr`.

## 5. Trampas ya pagadas, no volver a pisarlas

**Del arranque** (seis intentos hasta que levantó):

1. `with lock` no es trazable por dynamo.
2. `torch.compiler.disable` tampoco: vLLM compila con `fullgraph=True` y no
   admite cortes de grafo.
3. El **registro** del custom op también tiene que estar fuera del forward:
   `direct_register_custom_op` llama a `infer_schema`, que dynamo no traza.
4. Todo lanzamiento PTX dentro del forward va detrás de
   `direct_register_custom_op`. Hay un test que verifica ese invariante.
5. `BLOCK` es `next_power_of_2(K)` y es constexpr: hace falta **una variante
   por valor**, no una sola.
6. El índice del shift diádico es `(kb * BLOCK_K // SHIFT_BLOCK)`, no `kb`. Con
   `BLOCK_K=64` la versión mal indexada lee fuera de rango y da
   `CUDA_ERROR_ILLEGAL_ADDRESS`. Con `BLOCK_K=128` coinciden, y por eso el
   fallo aparece sólo en el tile de prefill.

**De medición:**

* Medir con un `synchronize()` por lanzamiento agrega ~10 µs fijos — 44% de
  inflación sobre un kernel de 23 µs — y comprime toda razón A/B hacia 1.
  **Usar ráfagas alternadas A,B,A,B.** Barrer configs en tandas tampoco sirve:
  la L2 queda caliente con el patrón de la config en curso.
* El techo real de esta forma de kernel es **710 GB/s**, no los 936 nominales.

**De herramientas:**

* `tools/monolitizar.py` **no es idempotente**: corre siempre desde el archivo
  original. Sobre uno ya convertido, duplica el bloque PTX.
* **No paralelizar `git`**: dos instancias tocando el índice chocan con
  `index.lock` y el `checkout` falla en silencio, dejando el archivo equivocado.
* **Commitear seguido.** Dos veces en esta sesión un `git checkout --` destruyó
  trabajo sin commitear, incluidos los arreglos del shift, que después costaron
  un arranque entero volver a encontrar.

## 6. Advertencia sobre el gate

El gate de `tools/monolitizar.py` compara **PTX contra el Triton del mismo
archivo**, bit a bit. Eso valida *la conversión*, no *la corrección del kernel
original*: si el Triton ya estaba mal, el PTX reproduce el error con fidelidad
perfecta. Ya pasó con el no-finito de SK-05, donde Triton y PTX coinciden bit a
bit en estar mal.

**Para corrección de verdad hay que contrastar contra una referencia
independiente**, como hace el test contra referencia fp32 descrito en §3.1.
