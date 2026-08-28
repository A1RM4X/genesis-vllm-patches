# Plan de optimización PTX

Estado del asm de los super kernels y qué falta optimizar. Se marca cada kernel
a medida que se optimiza.

**Regla de trabajo:** antes y después de tocar un asm, correr

```
docker run --rm --gpus '"device=0"' --entrypoint bash \
  -v $PWD:/repo \
  -v $PWD/vllm/_genesis/kernels:/usr/local/lib/python3.12/dist-packages/vllm/_genesis/kernels:ro \
  -w /repo vllm/vllm-openai:v0.27.1 \
  -c "python3 -m pytest /usr/local/lib/python3.12/dist-packages/vllm/_genesis/tests/test_sk_ptx_formas.py -q"
```

Son 16 segundos contra los minutos de levantar vLLM. Cubre el cruce de
`HAS_SHIFT` con todos los tiles, todos los buckets de M, residual sí/no, un
`BLOCK` por cada K real, y el invariante de que ningún lanzamiento PTX quede
fuera del custom op. Si se agrega una clase de fallo nueva, va ahí.

**Cómo medir.** Ráfagas alternadas A/B, nunca lanzamientos aislados. Medido: un
`torch.cuda.synchronize()` por llamada agrega ~10 µs fijos, que sobre un kernel
de 23 µs son 44% de inflación, y comprime toda razón A/B hacia 1. Barrer
configs en tandas tampoco sirve: la L2 queda caliente con el patrón de la
config en curso. Sólo A,B,A,B da números que se sostienen.

---

## Lista de kernels

| # | kernel | K×N per-rank | veces/fwd | estado | qué falta |
|---|--------|--------------|----------:|--------|-----------|
| 1 | SK-01 GDN qkvz | 5120×8192 | 48 | ⬜ | bucle en el mínimo; ver §3 |
| 2 | SK-02 GDN out | 3072×5120 | 48 | ⬜ | bucle en el mínimo; ver §3 |
| 3 | SK-03 FA qkv | 5120×7168 | 16 | ⬜ | bucle en el mínimo; ver §3 |
| 4 | SK-04 FA o | 3072×5120 | 16 | ⬜ | bucle en el mínimo; ver §3 |
| 5 | SK-05 MLP gate_up | 5120×17408 | 64 | ⬜ | **§1 bug** + §2 + `HAS_SHIFT` |
| 6 | SK-06 MLP down | 8704×5120 | 64 | ⬜ | bucle en el mínimo; ver §3 |
| 7 | SK-07 lm_head | 5120×124160 | 1 | ⬜ | §2 acumulador fp32 |
| 8 | SK-08 SSM control | — | 48 | ⬜ | sin analizar |
| 9 | SK-09 norm/embed | 5120 | ~1 | ⬜ | **el que más pesa**, ver §4 |
| 10 | SK-10 MTP draft | 5120×7168 | 1 | ⬜ | bucle en el mínimo; ver §3 |
| 11 | SK-11 vision | 1280×1280 | 0 | ⬜ | bf16, fuera del camino de texto |
| 12-18 | los 7 `_w4a8` | varias | 0 | ⬜ | no cableados en producción |

---

## §1. Bug abierto: SK-05 no-finito en prefill

`sk05_gateup_gemm` devuelve NaN/Inf a **M ≥ 1200** con el tile (256,128,64).
El kernel Triton original hace exactamente lo mismo, **bit a bit**, así que es
previo a la conversión a PTX y no lo introdujo ella.

```
M=130   tile=(256,128,64)  triton_fin=True   ptx_fin=True   iguales=True
M=1200  tile=(256,128,64)  triton_fin=False  ptx_fin=False  iguales=True
M=8192  tile=(256,128,64)  triton_fin=False  ptx_fin=False  iguales=True
```

Ese tile corre con **255 registros y 16 derrames**, que es el primer sospechoso.
Está marcado `xfail` en el barrido para que el resto siga sirviendo de red.

**Va primero, antes que cualquier optimización**: si PN110 produce basura en
prefill largo, medir rendimiento no significa nada.

## §2. Los dos kernels que quedaron con acumulador fp32

SK-05 (`_sk05_gateup_silu_kernel`) y SK-07 siguen con `acc` en fp32 y `tl.exp2`
dentro del bucle. Los otros seis usan acumulador int32 con corrimiento entero.

Medido en SK-07, migrar a int32 + `HAS_SHIFT`: **1.06× / 1.04× / 1.31× / 1.04×**
según el tamaño de vocabulario. Modesto, y el 1.31 se sale de la serie. Las 87%
de instrucciones de más no se traducen en tiempo porque SK-07 mueve 42-636 MB y
está dominado por DRAM.

Vale hacerlo por consistencia, no por velocidad. Y rompe el bit-exacto contra
la referencia actual, así que hay que rehacer la línea base del gate.

## §3. El bucle ya está en el mínimo

Seis de los ocho kernels emiten **el mismo cuerpo de bucle, byte por byte**:

| tile | instr | mma | ldmatrix | ALU | cvt |
|------|------:|----:|---------:|----:|----:|
| 16×64 | 65 | 8 | 8 | 31 | 0 |
| 32×64 | 79 | 16 | 12 | 32 | 0 |
| 64×64 | 105 | 32 | 16 | 36 | 0 |

Sin `ld.shared.b8`, sin `prmt`, sin `cvt` adentro, sin derrames. Es el mínimo
para el tile: 12 `ldmatrix` + 16 `mma` + las enteras del acumulador.

Dos falsas pistas ya descartadas con medición, **no volver a intentarlas**:

* **Transponer B.** Un volcado mostró 64 `ld.shared.b8` + 48 `prmt` (41% del
  bucle) y transponer daba 1.37×. No aplica: el banco pasaba `B [K,N]` contiguo,
  y `requant_inplace` ya devuelve strides `(1,K)`, o sea contiguo en K.
  Producción ya está en el layout bueno.
* **`cvt` hundidos en el bucle.** Están en el bloque de SALIDA, después del
  salto de retorno: se ejecutan una vez. El error fue tomar "el bloque más
  grande entre dos labels" como cuerpo del bucle. El bucle termina en el salto
  de retorno, no en el label siguiente.

## §4. Lo que de verdad conviene atacar

**El quant de SK-09.** En la traza de decode son **0.648 ms por forward** y ahí
está *todo* el déficit contra Marlin: los GEMM empatan (16.840 contra 16.726
ms/fwd). Optimizar GEMM no mueve la aguja; optimizar esto sí.

**La ocupación, no las instrucciones.** Los GEMM corren a 60-74% del techo de
DRAM, o sea limitados por memoria. `shared = (BM+BN)·BK·(stages-1)` = 36864 B
topea en 2 CTAs/SM = 8 warps de 48 = 17% de ocupación. Con `num_stages=3` baja
a 24576 B → 4 CTAs/SM. Medido en ráfagas alternadas sobre SK-05:

| M | S=4 | S=3 | gana |
|---|----:|----:|-----:|
| 16 | 167.97 µs (56.7%) | 115.28 µs (82.6%) | **1.457×** |
| 20 | 187.46 µs (50.8%) | 133.94 µs (71.1%) | **1.400×** |
| 32 | 179.06 µs (53.2%) | 143.54 µs (66.3%) | **1.247×** |

Depende de la geometría: SK-05 (N=17408, 272 CTAs) gana; SK-03 (N=7168, 112
CTAs) y SK-06 (N=5120, 80 CTAs) dan 0.95-1.03×. **Hay que medir por kernel.**

Ese cambio se aplicó en su momento y se perdió con un `git checkout`. Hay que
rehacerlo.

**El puntero escalar del shift.** Con `sh_ptrs` indexado por `offs_n` el PTX
emite cuatro `ld.global.b8` por iteración — una carga de un byte por grupo de
lanes — o sea 160 viajes a global por kernel para leer 40 bytes que no cambian
nunca. Como `BLOCK_N ≤ SHIFT_BLOCK` en todas las configs, el tile entero cae en
un solo bloque diádico y el shift es uniforme en el CTA: alcanza un escalar,
difundido a todos los lanes. Está implementado sólo en algunos kernels.

## §5. Techos, para no perseguir fantasmas

* El techo real de esta forma de kernel es **710 GB/s**, no los 936 nominales.
* El aparato del shift cuesta **8-15%** contra el GEMM pelado. Es el máximo
  recuperable tocando el epílogo.
* `HAS_SHIFT=False` vale 1.09-1.27× en SK-03 y SK-06 (limitados por
  instrucciones) y **nada** en SK-05 (limitado por memoria). SK-05 y SK-07 no
  tienen esa rama y hacen 32 operaciones enteras por iteración para aplicar un
  shift que en producción vale cero.
* El epílogo en bf16 de seis kernels da **2× el error** del piso del dtype. En
  fp32: mismos registros, mismo shared, **menos** instrucciones (913 contra 931
  líneas de PTX) y **1.6% más rápido**. Estrictamente mejor en las tres
  dimensiones. El epílogo está fuera del bucle, así que no cuesta nada.

## §6. Anatomía del PTX, para dimensionar

El bucle no es lo que ocupa. Para el tile de prefill:

| tile | total | `.loc` | vacías | prólogo | **bucle** | **epílogo** |
|------|------:|-------:|-------:|--------:|----------:|------------:|
| 16×64×128 | 788 | 88 | 89 | 205 | **100** | 331 |
| 256×128×64 | 2377 | 153 | 335 | 346 | **279** | **1404** |

El epílogo corre **una vez** y es lo más grande: PTX no tiene noción de tile, y
una línea de Triton sobre `BLOCK_M × BLOCK_N` valores se desenrolla a ~10
instrucciones por elemento de salida. A 256×128 con 8 warps son 128 elementos
por hilo. Un cuarto del archivo ni siquiera son instrucciones (`.loc`,
comentarios, vacías). Se dejan a propósito.

## §7. Deuda de infraestructura

* **Commitear seguido.** Dos veces en esta sesión un `git checkout --` destruyó
  trabajo sin commitear, incluidos los dos arreglos del shift que después
  costaron un arranque entero encontrar de nuevo.
* **`tools/monolitizar.py` no es idempotente**: corre siempre desde el archivo
  original. Correrlo sobre uno ya convertido duplica el bloque PTX.
* **No paralelizar `git`**: dos instancias tocando el índice chocan con
  `index.lock` y el `checkout` falla en silencio, dejando el archivo equivocado.
