# Informe — Super Kernels SK-01..SK-11 (PN110) en sm_86

> **Fecha:** 2026-08-26
> **Scope:** `vllm/_genesis/kernels/sk0*.py`, `sk1*.py` (+ variantes `_w4a8`), `warmup_all_kernels.py`,
> `vllm/_genesis/wiring/quantization/patch_PN110_int8_phase_dispatch.py`, `vllm/_genesis/tests/test_sk*_proper.py`
> **Hardware de medición:** 2× RTX 3090 (GA102, sm_86, 24 GB, 936 GB/s), torch 2.13.0+cu130, Triton 3.7.1
> **Complementa (y contradice en parte):** `workshop/ox_alpha/super_kernels_auditoria.md`, que auditó
> *estructura* (LoC, `if`, "monolito") pero nunca midió contra un baseline real.

---

## 0. Veredicto

**No cumplen.** Ni funcional ni de performance, y en dos ejes distintos:

1. **No están conectados al modelo.** Ninguno de los 11 SK se invoca desde el forward. El único
   consumidor es `warmup_all_kernels.py` (que además se traga todos los errores) y los tests.
   El camino de producción de PN110 es `int8_hybrid_gemm` / `cutlass_scaled_mm`. El `+29% prefill`
   del commit viene de ahí, **no** de los SK.
2. **Cuando se los mide, son más lentos que el baseline que dicen reemplazar.** SK-01 en su forma
   real (K=5120, N=8192, M=8000) corre a **33 TOPS INT8** cuando un GEMM Triton INT8 tuneado
   estándar en la misma GPU llega a **147 TOPS**. Y es **2× más lento que `F.linear` bf16 sin
   cuantizar**. Es decir: en prefill, el "super kernel INT8" es una regresión neta contra no
   cuantizar nada.

La premisa —"fusionar en uno lo que antes eran múltiples kernels"— tampoco se cumple: **8 de 11 no
fusionan nada** (son GEMMs con epílogo de escala, que es lo que ya hace `cutlass_scaled_mm`), y
SK-05 en concreto **añade** lanzamientos respecto al camino normal.

Lo bueno: **SK-09 sí es un kernel real y rápido** (2.4–6.5× vs la cadena torch), y la idea de SK-07
(GEMM de lm_head solo sobre el vocabulario muestreado) es algorítmicamente correcta.

---

## 1. Metodología

Tres pasadas:

1. **Lectura línea por línea** de los 11 kernels + 8 variantes W4A8, del selector de PN110 y del warmup.
2. **Trazado de llamadas**: `grep` exhaustivo de quién invoca cada símbolo exportado.
3. **Medición en 3090 real**, no rooflining. Scripts en el scratchpad de la sesión; configuración
   idéntica a la geometría declarada por cada SK (per-rank TP=2).

Todos los números de abajo son medidos, `torch.cuda.synchronize()` alrededor, 10 warmups + 20-30 iters.

---

## 2. Hallazgo crítico: el código está muerto

`_genesis_select_super_kernel()` (PN110 L276) **devuelve un string**:

```python
if "gate_proj" in name_l or "up_proj" in name_l or "gate_up" in name_l:
    return "SK-05 MLP_GATEUP_FUSED_INT8_DIADIC"
```

Sus 6 usos (L508, L537, L775, L2707, L2970 + warmup L87) son todos para **loguear una tabla de
arranque**. No hay dispatch. El `apply` de PN110 va a `int8_linear()` → `int8_hybrid_gemm` (Triton) o
`ops.cutlass_scaled_mm`.

`warmup_all_kernels()` tampoco valida nada: cada SK está envuelto en `try: ... except Exception: pass`.
Ejemplo concreto de que eso oculta fallos reales — el warmup de SK-02 pasa `hidden` **bf16**:

```python
hidden = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
sk02_gemm_int8_scaled(hidden, b_col.t().contiguous(), a_scale, b_scale, shifts)
```

y `sk02_gemm_int8_scaled` empieza con `if hidden.dtype != torch.int8: raise TypeError`. **SK-02 nunca
se precalienta**, y nadie se entera. Además el warmup pasa siempre `shifts = zeros`, con lo cual el
camino diádico —la razón de existir de estos kernels— jamás se ejercita.

---

## 3. Hallazgo: no fusionan

| SK | Qué promete el docstring | Qué hace realmente | ¿Fusión? |
|---|---|---|---|
| SK-01 GDN QKVZ | "monolito branchless diádico" | GEMM INT8 + escala. El "split QKVZ" son 4 slices (views, gratis) | **No** |
| SK-02 GDN out | "GDN_OUT_INT8_SCALED" | GEMM INT8. Escribe fp32 y después `.to(bf16)` → pasada extra sobre M×N | **No** |
| SK-03 FA QKV | RMSNorm+quant+GEMM | Sí fusiona, pero con doble lectura de `hidden` por cada bloque N | **Sí** (mal) |
| SK-04 FA O | GEMM + residual | GEMM INT8; el residual se suma en Python fuera | **No** |
| SK-05 MLP gate/up | "FUSED" | RMSNorm en **torch** (≈7 lanzamientos + copia fp32 de M×5120), GEMM, SiLU en **torch**. Más lanzamientos que el camino normal | **No** |
| SK-06 MLP down | GEMM + residual | GEMM INT8 + residual dentro del kernel | **Sí** (real, chica) |
| SK-07 lm_head | GEMM sobre vocab muestreado | Correcto algorítmicamente; el gather de columnas es escalar | **Sí** |
| SK-08 SSM control | conv1d + delta rule GDN | **No es la delta rule** (§5.1) | n/a |
| SK-09 norm+embed | RMSNorm + quant fusionado | Sí, y bien hecho | **Sí** |
| SK-10 MTP draft | GEMM draft | quant + GEMM | parcial |
| SK-11 vision | "passthrough" | GEMM bf16 naive; 1.0–1.4× **más lento** que cuBLAS | **No** |

El propio docstring de SK-05 lo admite: *"No vende RMSNorm+quant+GEMM+SiLU single launch"*. El código
es honesto; el mensaje de commit ("11 super kernels PTX inline branchless") no.

Sobre el **"PTX inline"**: solo SK-04, SK-05 y SK-06 usan `tl.inline_asm_elementwise` (y solo para un
`selp` de shift, ~3 instrucciones). SK-01 tiene un `_PTX_MONOLITH_DOC` que es un string con la
anotación explícita *"auditoría, no emisión directa"*: es un comentario, no código. El resto es Triton
normal. La etiqueta "PTX inline" es cosmética.

---

## 4. Hallazgo: performance medida

### 4.1 SK-01, geometría real (K=5120, N=8192 per-rank TP=2)

Tiempo por llamada, ms:

| M | `F.linear` bf16 (cuBLAS) | SK-01 INT8 | Triton INT8 tuneado | SK-01 vs mejor |
|---:|---:|---:|---:|---:|
| 1 | 0.106 | 0.191 | 0.079 | **2.4× peor** |
| 32 | 0.110 | 0.193 | 0.078 | **2.5× peor** |
| 512 | 0.784 | 0.543 | 0.254 | **2.1× peor** |
| 1664 | 2.516 | 3.135 | 1.197 | **2.6× peor** |
| 8000 | 10.444 | 20.206 | 4.576 | **4.4× peor** |

En TOPS efectivos (`2·M·N·K`), M=8000:

- cuBLAS bf16: **64.3 TFLOPS** (≈90% del pico bf16 de la 3090, sanity check OK)
- **SK-01 INT8: 33.2 TOPS**
- Triton INT8 tuneado: **146.7 TOPS** (≈pico INT8 denso de GA102)

**SK-01 alcanza el 23% de lo que la misma GPU, el mismo Triton y el mismo dtype pueden dar.** Y en
M=1664 y M=8000 pierde contra bf16 sin cuantizar, que es exactamente el régimen de prefill que PN110
dice acelerar.

### 4.2 Atribución del gap (ablación sobre el kernel de SK-01, mismo código, solo cambiando constexpr)

M=8000, ms:

| BLOCK_M | BLOCK_N | BLOCK_K | SHIFT_BLOCK | warps | stages | ms |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 64 | 32 | 128 | 4 | 2 | **18.9** ← configuración actual |
| 32 | 64 | 64 | 128 | 4 | 2 | 16.3 |
| 64 | 128 | 64 | 128 | 4 | 4 | 9.65 |
| 128 | 128 | 64 | 128 | 8 | 3 | **7.98** |
| 128 | 128 | 64 | 5120 | 8 | 3 | **7.54** ← sin flush por bloque de shift |

Es decir: **2.4× se recupera solo cambiando cinco números**. El resto hasta 4.58 ms lo aporta
reescribir el bucle (avance de punteros en vez de recalcular offsets+máscaras por iteración, y
swizzle de grid para reuso de L2).

### 4.3 SK-09 y SK-11

| Kernel | M | SK | torch | Resultado |
|---|---:|---:|---:|---|
| SK-09 rmsnorm+quant | 1 | 0.024 ms | 0.057 ms | **2.37× más rápido** |
| SK-09 rmsnorm+quant | 512 | 0.044 ms | 0.283 ms | **6.47× más rápido** |
| SK-09 rmsnorm+quant | 8000 | 0.571 ms | 3.718 ms | **6.51× más rápido** |
| SK-11 vision GEMM | 4096×1152×1152 | 0.290 ms | 0.214 ms | **1.4× más lento** |
| SK-11 vision GEMM | 4096×1152×4304 | 1.072 ms | 1.038 ms | empate |

SK-09 es el único que justifica su existencia. SK-11 no aporta nada sobre cuBLAS (aunque su docstring
al menos dice "passthrough").

---

## 5. Hallazgo: por qué son lentos en sm_86

### 5.1 Tile fijo, idéntico en los 19 archivos, sin autotune

`BLOCK_M=32, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=2` está hardcodeado en **todos** los SK,
para cualquier forma y cualquier M. No hay un solo `@triton.autotune` en todo el repo.

Intensidad aritmética del tile 32×64 con K=5120:

- MACs por tile: `32·64·5120 = 10.5 M`
- Bytes leídos: `(32+64)·5120 = 490 KB`
- → **21 MAC/byte**

Con 936 GB/s, 21 MAC/byte topea alrededor de 40 TOPS incluso con L2 perfecta — que es justo lo medido
(33 TOPS). Un tile 128×128 da `128·128·5120 / ((128+128)·5120) = 64 MAC/byte`, 3× más, y ahí sí el
Tensor Core INT8 puede saturarse.

Además:

- **`num_stages=2` desactiva el pipeline `cp.async` efectivo de Ampere.** Triton necesita ≥3 etapas
  para solapar carga global→shared con el `mma`. Con 2 el kernel queda serializado en latencia de
  memoria. La ablación lo confirma (stages 2→3/4 es la mitad de la ganancia).
- **Grid 2-D `(cdiv(M,32), cdiv(N,64))` sin swizzle.** A M=8000, N=8192 son 32 000 CTAs en orden
  row-major: los bloques concurrentes barren toda la fila de B antes de reusarla, con lo cual la L2 de
  6 MB no retiene nada. Un `GROUP_SIZE_M` (grouped ordering) es el patrón estándar y está ausente.
- **`BLOCK_K=32` es el mínimo de `mma.m16n8k32`.** Fuerza 160 iteraciones de bucle para K=5120, cada
  una con recálculo completo de `offs`, punteros y máscaras. `BLOCK_K=64` ya da 15% gratis.

### 5.2 El bucle de shift rompe el acumulador

Todos los SK diádicos (01, 02, 04, 06) tienen esta estructura:

```python
for kb in range(K // 128):          # 40 iteraciones
    int_acc = tl.zeros(..., tl.int32)   # <-- acumulador nuevo
    for sub in range(128 // 32):        # 4 iteraciones
        int_acc += tl.dot(a_tile, b_tile)
    acc += (int_acc << shift_val).to(f32) * a_scales * b_scales
```

Eso es **40 vaciados** del acumulador con conversión INT32→fp32 y dos multiplicaciones sobre un tile
32×64 por cada uno. En un GEMM normal el acumulador vive en registros durante todo K y el epílogo se
paga una vez. Coste medido: 5% a M=8000, pero **40% a M=512** (0.432 → 0.254 ms). Es el precio del
diseño diádico por bloque de K, y no está contabilizado en ningún lado.

Nota: SK-05 **no** hace esto (acumula todo K y shiftea una vez al final), lo cual es más rápido pero
significa que aplica un esquema de shift **distinto** (solo per-N) al de SK-01/02/04/06. Los 11
kernels no implementan la misma matemática.

### 5.3 Lecturas redundantes de la activación

SK-03, SK-05 y SK-07 hacen **dos pasadas completas sobre `hidden`** dentro del kernel (una para
`amax`/`sum_sq`, otra para cuantizar), y esto ocurre **en cada programa `pid_n`**. Para SK-05 con
N=17408 y BLOCK_N=64 son 272 bloques × 2 pasadas = **544 lecturas de la matriz de activación
completa**. La L2 amortigua parte, pero es tráfico que no debería existir: el patrón correcto es un
kernel de quantización previo (que es exactamente lo que ya hace SK-09) o un `amax` por bloque con
`atomic_max`.

### 5.4 W4A8: la descompresión anula la ganancia

En `sk01_gdn_qkvz_w4a8.py` y hermanos:

```python
packed_k = cur_k // 2
is_even  = (cur_k % 2) == 0
packed   = tl.load(w_packed_ptrs, ...)   # carga el byte entero
w_low = packed & 15; w_high = (packed >> 4) & 15
w_q = tl.where(is_even[:, None], w_low, w_high)   # tira la mitad
```

Cada byte empacado se lee **dos veces** (una para el nibble par, otra para el impar) y en cada lectura
se descarta la mitad. Eso es exactamente el mismo tráfico de memoria que INT8: **el beneficio de
ancho de banda de W4 desaparece por completo**, que es el único motivo por el que uno haría W4A8. El
patrón correcto es cargar `BLOCK_K/2` bytes una vez y desempacar ambos nibbles en registro.

---

## 6. Bugs de corrección

### 6.1 SK-08 no implementa la delta rule (bloqueante)

`sk08_ssm_control.py:112-120`:

```python
hk_sum = tl.sum(b_h, axis=1) * 0.01
b_v_corr = (v_val - hk_sum) * beta_val
b_h = b_h + b_v_corr[:, None] * 0.5
o_val = tl.sum(b_h, axis=1) + q_val * 0.1
```

Constantes mágicas `0.01`, `0.5`, `0.1` sin ninguna justificación. Y, más grave: **no aparece el
vector `k` en ningún lado** — no hay producto externo `k ⊗ v` para la actualización de estado, ni
producto `q · S` para la salida (se usa `sum(b_h, axis=1)`, que es sumar el estado, no proyectarlo por
`q`). Además `q_val = v_val = conv_acc`: q, k y v son el mismo tensor. Esto no es Gated DeltaNet; es
un placeholder que genera números. El commit `6a25ff7` dice "fix SK08 SSM" pero el fix no ocurrió.
La auditoría previa ya lo marcaba "Warn — SSM aún placeholder"; sigue igual.

### 6.2 SK-09 fuerza semántica Gemma (bloqueante para Qwen)

`sk09_norm_embed.py:403` fija `IS_GEMMA=1` como constexpr, sin parámetro para desactivarlo, y el
kernel hace `w_eff = 1 + w`. Qwen3.5 usa RMSNorm plana (`w`, sin el +1). Medido contra una RMSNorm
estándar: **diferencia máxima de 101 niveles INT8 de 127**. Es el único kernel rápido del set y
produciría basura en el modelo objetivo.

### 6.3 SK-04 `fa_o_forward` cuantiza con un cast de C

`sk04_fa_o.py:76-79`:

```python
if x.dtype != torch.int8:
    a = x.to(torch.int8)                       # truncamiento, no cuantización
    a_scale = torch.ones((x.shape[0],), ...)   # escala 1.0
```

`x.to(torch.int8)` sobre activaciones bf16 (rango típico ±3) trunca todo a {-3..3} y descarta la
escala. No es una cuantización; es destrucción de la señal, silenciosa.

### 6.4 SK-01 y SK-02 no manejan shift negativo

SK-01 (L83) y SK-02 (L49) hacen `int_acc << shift_val` incondicional. SK-03/04/05/06 sí implementan
`selp` para el caso negativo. Si el requantizador emite algún shift negativo, SK-01/02 dan un
resultado silenciosamente distinto (`<<` con operando negativo es UB en PTX).

### 6.5 Overflow de INT32 sin guarda

El docstring de `int8_linear` dice *"shift ≤10 cabe en INT32"*. La cuenta: peor caso por bloque de 128
es `128·127·127 = 2 064 512`; `<< 10` = `2 114 060 288`, contra `2^31-1 = 2 147 483 647`. Entra por un
1.5% de margen, y **con shift=11 desborda**. No hay ningún `assert` ni clamp en ningún kernel ni en el
requantizador que garantice `shift ≤ 10`.

### 6.6 Precisión gratuitamente perdida

- **SK-01** (L64): `a_scale_ptr` es fp32 y se carga como `.to(tl.bfloat16).to(tl.float32)` —
  truncamiento deliberado a 8 bits de mantisa. Medido: el error relativo sube de 4.5e-3 a 5.8e-3 sin
  ninguna contrapartida de velocidad.
- **SK-03, SK-04** y todas las variantes W4A8: `acc = tl.zeros(..., dtype=tl.bfloat16)`. Acumular 40
  (o 160, en W4A8) parciales en bf16 con 8 bits de mantisa. El acumulador de un GEMM va en fp32,
  siempre; en Ampere no cuesta nada porque el `mma` ya acumula en INT32.
- **SK-02**: escribe `out_fp32` y devuelve `.to(out_dtype)` — una pasada extra de lectura+escritura
  sobre M×N y el doble de memoria pico, cuando el `tl.store` podría emitir bf16 directo.

---

## 7. Hallazgo: los tests validan strings, no kernels

`test_sk*_proper.py` (≈9 100 líneas en total) se apoya en:

- **Chequeos de estándares por regex** sobre el fuente: que el archivo contenga `mma.sync`, que el
  cuerpo `@triton.jit` no contenga la cadena `float32`, que no haya `if`.
- **Chequeos funcionales** contra un fallback torch, `max_diff <= 1e-2`.
- **"Benchs"** que solo asertan `kernel_ms < 1.5 * fallback_torch_ms`.

Tres consecuencias:

1. El "bench" compara contra la **referencia torch lenta**, nunca contra cuBLAS ni contra
   `cutlass_scaled_mm`. Un kernel 4× peor que el baseline real pasa el test cómodamente.
2. La prohibición textual de `float32` empujó a workarounds cosméticos: `sk06_mlp_down.py:129` define
   `_F32 = tl.float32` con el comentario literal *"test_sk06 prohíbe float32 literal dentro de
   @triton.jit; alias evita falso positivo"*. El test se satisface, la semántica no cambia.
3. El requisito `mma.sync` se cumple **poniendo la cadena en un docstring** (`_PTX_MONOLITH_DOC` en
   SK-01, `_PTX_DOC` en SK-05). Eso es lo que permitió que "PTX inline" quedara como etiqueta sin
   código detrás.

El test mide adherencia a un estilo declarado, no que el kernel sea rápido ni correcto.

---

## 8. Qué haría falta, por prioridad

**P0 — decidir si esto vive o muere.** Hoy son ~5 500 líneas de kernel + ~9 100 de test que no
ejecutan en producción. Si la respuesta es "vive", el trabajo real es el de abajo. Si es "muere",
rescatar SK-09 (arreglando `IS_GEMMA`) y SK-07, y borrar el resto: el camino
`int8_hybrid_gemm`/`cutlass` es el que da el +29% medido.

**P1 — un solo GEMM INT8 bien hecho, no once.** SK-01/02/04/06 y sus variantes son el mismo GEMM con
distinta forma. Un kernel con `@triton.autotune` sobre `{BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M,
num_warps, num_stages}`, cacheado por `(M_bucket, K, N)`, avance de punteros, grid 1-D con swizzle y
acumulador INT32 sobre todo K. Ganancia medida disponible: **4.4×** en el caso de prefill.

**P2 — quantización fuera del GEMM.** SK-09 ya hace `rmsnorm+quant` a 6.5× vs torch. Que SK-03/05/07
consuman su salida en vez de recomputar `amax` dos veces por bloque N. Eso además convierte a SK-05 de
"RMSNorm en torch + GEMM + SiLU en torch" en dos lanzamientos reales.

**P3 — arreglar los bloqueantes de corrección** (§6.1 SK-08, §6.2 `IS_GEMMA`, §6.3 cast de SK-04,
§6.4 shift negativo, §6.5 guarda de overflow) antes de cablear nada.

**P4 — reescribir el bench de los tests** para comparar contra `cutlass_scaled_mm` y `F.linear` bf16,
con un umbral que falle si el SK no gana. Un kernel INT8 que pierde contra bf16 sin cuantizar tiene
que romper CI.

**P5 — W4A8:** desempacar `BLOCK_K/2` bytes una vez, o abandonar la vía. Como está, W4A8 tiene el
tráfico de W8 con la precisión de W4.
