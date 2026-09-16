onversión block-FP8 → INT8 W8A8 en Ampere: especificación técnica

**Documento de handoff.** Escrito para ser leído sin contexto previo por un modelo o
persona que vaya a implementar el código. Todo lo necesario está acá.

---

## 0. Resumen ejecutivo

Tenemos un checkpoint LLM cuantizado en **block-FP8 (E4M3, bloques 128×128)** que corre en
una GPU **Ampere (SM80/86)**. Ampere no tiene tensor cores FP8, así que vLLM lo emula:
dequantiza a FP16 en registros y ejecuta el MMA en FP16. No ganamos nada del formato de 8
bits salvo ancho de banda.

**Objetivo:** convertir a **INT8 W8A8** para usar los tensor cores INT8 nativos de Ampere
(~2× el throughput de FP16), perdiendo lo mínimo posible de calidad.

**Los tres resultados centrales del análisis:**

1. **Los pesos son gratis.** La conversión FP8 → INT8 puede ser *exacta* para el 99.95% de
   la energía de Frobenius de cada bloque, usando solo un shift entero. No es cuantización,
   es un cambio de representación. Sección 3.
2. **Las activaciones son el problema.** Ahí es donde E4M3 gana genuinamente sobre INT8, por
   rango dinámico. Necesita SmoothQuant. Sección 5.
3. **W8A16 no sirve.** Si las activaciones quedan en BF16, los tensor cores INT8 no se
   activan y volvemos al mismo path que ya tenemos. W8A8 o nada. Sección 1.2.

---

## 1. Contexto y restricciones

### 1.1 El modelo

`Qwen/Qwen3.8-27B` → build cuantizado `orcarouter/Qwen3.8-27B-Uncensored-FP8`.

Arquitectura `Qwen3_5ForConditionalGeneration` (`model_type: qwen3_5`). No es un
transformer denso estándar: es **híbrido**.

| Propiedad | Valor |
|---|---|
| Capas | 64 |
| `hidden_size` (d) | 5120 |
| `intermediate_size` (d_ff) | 17408 |
| Patrón de capas | 3 lineales : 1 atención completa (`full_attention_interval: 4`) |
| Capas de atención lineal | 48 (índices 0,1,2, 4,5,6, 8,9,10, …) |
| Capas de atención completa | 16 (índices 3, 7, 11, …, 63) |
| `vocab_size` | 248320 |
| `max_position_embeddings` | 262144 |
| Parámetros totales | 27.78 B |

Las 48 capas "lineales" son **Gated DeltaNet** (atención lineal con estado recurrente tipo
SSM), no atención softmax. Esto importa muchísimo para el análisis de error — ver 6.1.

### 1.2 El hardware, y por qué esto define todo

Ampere (SM80/86) soporta tensor cores para FP16, BF16, TF32, INT8 e INT4. **No soporta
FP8** — eso llega con Ada (SM89) y Hopper (SM90).

La instrucción relevante es `mma.m16n8k32.s8.s8.s32`: acumula en INT32 y **exige que los
dos operandos sean INT8**. No existe un MMA de "peso INT8 × activación FP16".

> **Consecuencia crítica que el implementador debe internalizar:** un esquema W8A16
> (pesos INT8, activaciones FP16) obliga a dequantizar los pesos a FP16 en registros y correr
> el MMA en FP16. Eso es *exactamente* lo que hace el path FP8-Marlin actual. Ganancia: cero.
>
> **El objetivo no negociable es W8A8.**

Ampere también tiene tensor cores INT4 (`mma.m16n8k64.s4`), pero están deprecados (removidos
en Hopper) y W4A4 destruye la calidad en LLMs. No los usamos.

### 1.3 El checkpoint de partida

Esquema de cuantización, leído de `config.json` → `quantization_config`:

- **Pesos:** FP8 `float8_e4m3fn`, bloques de **128×128**, con un tensor acompañante
  `weight_scale_inv` en BF16 (una escala por bloque, guardada como recíproco).
- **Activaciones:** FP8 dinámico por token, sin calibración.
- **Conteo de tensores:** 407 cuantizados + 407 escalas + 792 copiados = **1606**.
- **Tamaño:** 30.9 GB en 7 shards.

**Lo que NO está cuantizado** (lista `modules_to_not_convert`, 882 entradas) — *esto se
respeta tal cual, no se toca*:

- Toda la torre de visión (333 tensores `visual.*`)
- Todas las normalizaciones (`input_layernorm`, `post_attention_layernorm`, `q_norm`,
  `k_norm`, `linear_attn.norm`, `model.language_model.norm`)
- `lm_head` y `embed_tokens` (1.271 B parámetros cada uno)
- Las compuertas del linear attention: `A_log`, `conv1d`, `dt_bias`, `in_proj_a`, `in_proj_b`

**Razón de fondo para las compuertas del SSM:** `A_log` y `dt_bias` controlan el decay del
estado recurrente. Un error de cuantización ahí no afecta un token — se propaga y **se
acumula a lo largo de toda la secuencia**. Son 48 valores por capa; cuantizarlos no ahorra
nada y rompe el modelo en contexto largo.

### 1.4 Formas de los tensores

**Capa de atención lineal (48 de éstas):**

| Tensor | Forma | Cuantizado |
|---|---|---|
| `linear_attn.in_proj_qkv.weight` | [10240, 5120] | Sí |
| `linear_attn.in_proj_z.weight` | [6144, 5120] | Sí |
| `linear_attn.out_proj.weight` | [5120, 6144] | Sí |
| `linear_attn.in_proj_a.weight` | [48, 5120] | No |
| `linear_attn.in_proj_b.weight` | [48, 5120] | No |
| `linear_attn.conv1d.weight` | [10240, 1, 4] — **3D** | No |
| `linear_attn.A_log` / `dt_bias` | [48] | No |
| `linear_attn.norm.weight` | [128] | No |

**Capa de atención completa (16 de éstas):**

| Tensor | Forma | Cuantizado |
|---|---|---|
| `self_attn.q_proj.weight` | [12288, 5120] | Sí |
| `self_attn.k_proj.weight` | [1024, 5120] | Sí |
| `self_attn.v_proj.weight` | [1024, 5120] | Sí |
| `self_attn.o_proj.weight` | [5120, 6144] | Sí |
| `self_attn.q_norm` / `k_norm` | [256] | No |

**MLP (idéntico en ambos tipos, 64 capas + MTP):**

| Tensor | Forma | Cuantizado |
|---|---|---|
| `mlp.gate_proj.weight` | [17408, 5120] | Sí |
| `mlp.up_proj.weight` | [17408, 5120] | Sí |
| `mlp.down_proj.weight` | [5120, 17408] | Sí |

**Notas de layout que ahorran bugs:**

- `q_proj` tiene 12288 filas y no 6144 (= 24 cabezas × 256) porque `attn_output_gate: true`:
  la proyección emite query y compuerta concatenadas y se parten después. Verificado por
  conteo de bytes del checkpoint.
- `in_proj_qkv` es **fusionado**: filas 0–2047 = Q, 2048–4095 = K, 4096–10239 = V
  (16 key heads × 128, 16 × 128, 48 value heads × 128). **Los cortes caen en múltiplos de
  128**, así que ningún bloque de cuantización cruza dos de Q/K/V. Esto se preserva
  gratis. Nunca usar una escala per-tensor sobre este tensor.
- **Todas las dimensiones son múltiplos de 128** (5120=40·128, 6144=48·128, 10240=80·128,
  12288=96·128, 17408=136·128, 1024=8·128). No hace falta padding en ningún lado.
- `conv1d.weight` es **3D**: `[out_channels, in_channels/groups, kernel]` = [10240, 1, 4].
  Convolución causal depthwise. No es una matriz; no la trates como tal.

### 1.5 El modelo fue abliterado — leer antes de tocar los residual writers

Este build tiene la dirección de rechazo removida del residual stream mediante
`W' = W − r(rᵀW)` sobre **131 matrices**, que son exactamente las que escriben al residual
stream (identificables por tener 5120 filas):

| Matriz | Cantidad |
|---|---|
| `self_attn.o_proj` | 16 + 1 (MTP) = 17 |
| `linear_attn.out_proj` | 48 |
| `mlp.down_proj` | 64 + 1 (MTP) = 65 |
| `embed_tokens` (espacio de filas) | 1 |

**Implicancia técnica, no moral:** la magnitud de esa edición está *por debajo del escalón
de cuantización de E4M3* (el card reporta 99.9% de códigos FP8 idénticos al checkpoint
oficial no abliterado). Funciona porque `r` es sistemática y se acumula a través de 131
matrices y 64 capas, mientras el ruido de cuantización es aleatorio y se cancela.

**Si re-cuantizamos con escalas más gruesas, podemos perturbar la abliteración de forma
medible.** Perplejidad no lo detecta. Ver test T10.

### 1.6 Números de referencia publicados (nuestros targets)

Medidos sobre este mismo checkpoint FP8 servido con vLLM:

| Métrica | Valor | n |
|---|---|---|
| WikiText-2-raw PPL | **6.96** | 296,907 tokens, KV en BF16 |
| MMLU (0-shot letter) | **84.7%** | 300 |
| MMLU-Pro (CoT) | **76.8%** | 250 |
| GSM8K (CoT) | **88.7%** | 150 |
| CMMLU (0-shot, chino) | **80.8%** | 500 |

Tener targets reproducibles publicados es un lujo. Toda la validación se ancla acá.

---

## 2. Teoría, parte 1: por qué INT8 no es un downgrade para pesos

Intuición común y equivocada: "FP8 es punto flotante, INT8 es entero, entonces FP8 es más
preciso". Falso para pesos.

**E4M3** (1 signo, 4 exponente, 3 mantissa) tiene **error relativo constante** en todo el
rango: 4 bits significativos (3 explícitos + 1 implícito) → paso relativo de 2⁻⁴ = 6.25% en
el peor caso dentro de una binada.

**INT8 simétrico** con escala s tiene **error absoluto constante**: paso s = max/127. Error
relativo pequeño para valores grandes, grande para valores chicos.

Para una distribución aproximadamente gaussiana — que es lo que son los pesos de un LLM
dentro de un bloque — **INT8 tiene menor MSE que E4M3**, porque E4M3 gasta códigos en las
colas donde casi no hay masa de probabilidad. La ventaja de FP8 aparece con outliers
pesados, que es la situación de las *activaciones*, no la de los pesos.

Referencia: van Baalen et al., *FP8 versus INT8 for efficient deep learning inference*.

**Conclusión:** si re-derivamos las escalas correctamente, el paso FP8 → INT8 en pesos no
degrada. Y con el truco de la sección 3, ni siquiera introduce error.

---

## 3. Teoría, parte 2: el truco de alineación de grilla

Este es el núcleo del documento. No lo encontré publicado en la literatura — es específico
de la situación de partir desde un checkpoint que *ya está* en FP8.

### 3.1 La observación

Todo valor finito de E4M3 es **un múltiplo entero exacto de 2⁻⁹** (su subnormal mínimo).
Con campo de exponente `e` (4 bits) y mantissa `m` (3 bits):

```
valor = (8 + m) · 2^(e−10)          para e ≥ 1  (normales)
valor = m · 2^(−9)                  para e = 0  (subnormales)

valor / 2^(−9) = (8 + m) · 2^(e−1)  ← entero, siempre
```

**E4M3 ya es un entero.** Uno de 18 bits con signo (máximo 14·2¹⁴ = 229376).

Entonces la pregunta "¿cómo cuantizo FP8 a INT8?" **no es una pregunta de cuantización**.
Es: *¿qué ventana de 8 bits me quedo de ese entero de 18?* Y eso es un shift.

### 3.2 La derivación

Necesitamos que el máximo del bloque entre en 7 bits: `n_max >> k ≤ 127`.

Como `n = (8+m) << (e−1)` y `8+m ∈ [8,15]`, la longitud en bits de `n` es exactamente
`e+3`. Por lo tanto:

```
k = e_max − 4
```

donde **`e_max` es simplemente el mayor campo de exponente presente en el bloque.** Sin
max en flotante, sin división, sin log₂. Solo mirar 4 bits.

El shift no pierde bits mientras `e − 1 ≥ k`, es decir `e ≥ e_max − 3`: **cuatro
exponentes de exactitud.**

### 3.3 La forma final

Definí `d = e_max − e` (el déficit de exponente del valor respecto del tope del bloque).
Sustituyendo:

```
q = (8 + m) << (3 − d)      si d ≤ 3
q = (8 + m) >> (d − 3)      si d > 3
```

**Una resta y un shift.** La escala nueva es:

```
s' = s_blk · 2^(e_max − 13)
```

Verificación: `w = q · s'`. Para `e = e_max, m = 7`:
`q = 15 << 3 = 120`, `q·s' = 120 · 2^(e_max−13) = 15 · 2^(e_max−10)` ✓ (coincide con el
valor E4M3 original).

### 3.4 Tabla de comportamiento

| d | operación | rango de q | estado |
|---|---|---|---|
| 0 | `(8+m) << 3` | 64–120 | exacto |
| 1 | `(8+m) << 2` | 32–60 | exacto |
| 2 | `(8+m) << 1` | 16–30 | exacto |
| 3 | `(8+m)` | 8–15 | exacto |
| 4 | `(8+m) >> 1` | 4–7 | colisionan pares |
| 5 | `(8+m) >> 2` | 2–3 | colisionan cuartetos |
| 6 | `(8+m) >> 3` | 1 | colapsa |
| ≥7 | — | 0 | flush |

### 3.5 Cuantificación del error

El error existe solo para `d ≥ 4`: pesos por debajo de 1/16 de la binada superior del
bloque. Para un bloque gaussiano de 128×128 = 16384 muestras, el máximo cae en ≈ 4σ, así
que el umbral está en ≈ 0.125σ.

- Fracción de **pesos** afectados: `P(|Z| < 0.125) ≈ 10%`
- Fracción de **energía** (Frobenius) que vive ahí: `∫₋₀.₁₂₅^0.125 z²φ(z)dz ≈ 5·10⁻⁴`

> **La conversión es exacta para el ~99.95% de la energía de Frobenius del bloque.**
> El error sobre el 0.05% restante está acotado por medio paso de INT8 — no peor que
> INT8 estándar.

### 3.6 Detalles de implementación que evitan bugs

**Encontrar `e_max` sin decodificar nada.** En E4M3 el orden de bytes es monótono en la
magnitud (igual que IEEE-754): el campo de 7 bits `eeeemmm` crece con el valor. Entonces:

```python
e_max = (codes & 0x7F).max() >> 3
```

**Cuidado con `0x7F`**: en `float8_e4m3fn` ese patrón es NaN, no un valor. Enmascararlo
antes del max. En un checkpoint sano no debería existir — verificarlo es el test T1.

**Actualizar la escala también es entero.** Multiplicar un BF16 por 2^j equivale a sumar j
al campo de exponente, o sea sumar `j << 7` a los bits del uint16. Válido mientras no
desborde el exponente.

**Subnormales (e = 0):** la fórmula `8+m` no aplica (no hay bit implícito). Su magnitud es
≤ 7·2⁻⁹ ≈ 0.0137, que frente a cualquier `e_max` realista da `d` enorme. **Flush a cero.**
Es correcto en la práctica y evita una rama.

### 3.7 El desperdicio de rango que queda

`q_max = (8 + m_max) << 3`, que va de **64 a 120** según la mantissa del máximo del bloque.
Nunca llegamos a 127. Utilización promedio del rango: ~73%, o sea regalamos ~0.46 bits.

**¿Se puede recuperar?** No sin romper la exactitud. Preservar la grilla exige que `s'`
divida a todos los pasos de E4M3, y esos son potencias de dos. Cualquier multiplicador no
diádico rompe la alineación, y el único que entraría sin desbordar sería ≤ 127/120 = 1.058,
que no aporta nada.

**El intercambio sigue siendo bueno:** con `q_max ≈ 92` promedio tenemos precisión relativa
2⁻⁶·⁵ en el tope, contra 2⁻⁴ de E4M3. **Seguimos siendo más finos que el formato de origen.**

### 3.8 El modo de falla

Todo esto se rompe si un bloque tiene un **outlier aislado**: `e_max` queda fijado por el
outlier, todo lo demás cae en `d` grande, y se pierde el bloque entero.

Es la debilidad de cualquier escala por bloque, pero E4M3 la tolera mejor por su espaciado
logarítmico — así que acá sí perderíamos algo real respecto del checkpoint de partida.

**Es medible antes de escribir una línea de kernel.** Ver test T0, que es la primera cosa
que hay que correr.

### 3.9 Código de referencia (numpy, offline)

```python
import numpy as np

def fp8_e4m3_to_int8_aligned(codes_u8, s_blk):
    """
    codes_u8 : uint8 [..., 128, 128]  bytes crudos float8_e4m3fn de un bloque
    s_blk    : float32 escalar        escala del bloque (ya invertida si venía como _inv)
    returns  : (int8 [...,128,128], float32 escalar)
    """
    mag = codes_u8 & 0x7F
    assert not (mag == 0x7F).any(), "NaN en los códigos FP8"

    e_max = int(mag.max()) >> 3
    if e_max == 0:                      # bloque entero subnormal/cero
        return np.zeros_like(codes_u8, dtype=np.int8), np.float32(0.0)

    e = (codes_u8 >> 3) & 0x0F
    m = codes_u8 & 0x07
    mant = (8 + m).astype(np.int32)
    d = (e_max - e).astype(np.int32)

    up   = mant << np.clip(3 - d, 0, None)
    # +half-ulp antes del shift derecho = round-to-nearest en vez de truncar
    down = (mant + (1 << np.clip(d - 4, 0, None))) >> np.clip(d - 3, 0, None)

    q = np.where(d <= 3, up, down)
    q = np.where((e == 0) | (d > 6), 0, q)
    q = np.where(codes_u8 & 0x80, -q, q)

    s_new = np.float32(s_blk * 2.0 ** (e_max - 13))
    return q.astype(np.int8), s_new
```

El `+ (1 << (d-4))` es redondeo al más cercano en lugar de truncamiento. Es gratis y evita
un sesgo sistemático hacia cero en la cola de la distribución.

---

## 4. Decisión de diseño: qué hacer con las escalas de bloque

Acá hay una tensión real que el implementador tiene que resolver conscientemente.

El checkpoint tiene escalas **por bloque 128×128**, o sea que la escala cambia también a lo
largo de la dimensión K. Los kernels CUTLASS INT8 de Ampere quieren **escala por canal de
salida**, para aplicar un solo multiply sobre el acumulador INT32 en el epílogo.

### Diseño A — preservar los bloques (fiel)

Mantener escalas 128×128. Acumular INT32 sobre K=128, convertir a FP32, multiplicar por la
escala del bloque, acumular en FP32.

- **Preserva la exactitud de la sección 3 al 100%.**
- Con `m16n8k32` son 4 MMAs por chunk de K antes de cada flush. El flush cada 4 MMAs se
  come buena parte de la ventaja de INT8.
- Requiere kernel custom. vLLM ya tiene `w8a8_block_fp8_matmul` en Triton — adaptarlo a
  block-INT8 es un cambio moderado, no una reescritura.

### Diseño B — colapsar a per-channel (pragmático)

Una escala por fila de salida = max sobre los 40 bloques K de esa fila.

- **Usa kernels existentes y soportados**: checkpoint `compressed-tensors` W8A8, que vLLM
  sirve nativo en Ampere vía CUTLASS. Sin código de kernel.
- **Pierde la exactitud** de la sección 3: las escalas de bloque son BF16 arbitrarias, no
  diádicas, así que los productos `código × s_blk` de bloques distintos no caen en una
  grilla común. El truco de alineación aplica *dentro* del bloque, no a lo largo de la fila.

### Diseño C — híbrido diádico (el interesante)

Escala per-channel en float + **corrección per-bloque como potencia de dos** (un `int8` de
shift por bloque; para `in_proj_qkv` son 3200 bytes, ruido).

```
w = q · 2^(shift_b) · s_row
```

El shift es entero, así que la grilla sigue siendo diádica y exacta. Y el kernel aplica el
shift **sobre el acumulador INT32**, que es un shift entero barato, no un multiply flotante.

Chequeo de overflow: 128 productos de int8×int8 ≤ 127·127·128 = 2.06 M, con margen amplio
en INT32 (2.1 G). Se puede shiftear a la izquierda ~10 posiciones sin desbordar.

Costo: redondear la razón `s_blk/s_row` hacia abajo a potencia de dos desperdicia hasta 1
bit de rango en algunos bloques — pero es desperdicio, no error de redondeo.

### Recomendación

**Empezar por B**, medir contra los targets de 1.6, y solo pasar a C si B no cierra. A es
la referencia de máxima fidelidad para validar los otros dos, no necesariamente lo que se
despliega.

---

## 5. Teoría, parte 3: las activaciones son el problema real

Acá es donde se va a perder calidad, no en los pesos.

El residual stream de un LLM tiene **outliers por canal de 20–100× la mediana**. E4M3 los
absorbe con su rango dinámico logarítmico. INT8 con escala por token, no.

**Solución: SmoothQuant.** Migrar la dificultad de las activaciones a los pesos con un
escalado por canal de entrada:

```
Y = (X · diag(s)⁻¹) · (diag(s) · W)
```

**Ventaja específica de este modelo:** `input_layernorm.weight` y
`post_attention_layernorm.weight` están en **BF16 sin cuantizar**. Absorbemos `s` ahí gratis,
sin GEMM extra.

Qué se puede fusionar con qué:

| Tensor | Cómo absorber `s` |
|---|---|
| `q_proj`, `k_proj`, `v_proj` | en `input_layernorm.weight` |
| `in_proj_qkv`, `in_proj_z` | en `input_layernorm.weight` |
| `gate_proj`, `up_proj` | en `post_attention_layernorm.weight` |
| `down_proj` | escalar filas de `up_proj` por 1/s y columnas de `down_proj` por s |
| `o_proj`, `linear_attn.out_proj` | **no hay norm adelante** — ver abajo |

**Restricción importante:** los tensores que leen del *mismo* estado normalizado deben
compartir un único vector `s` (es la restricción estándar de SmoothQuant/AWQ). O sea, `s`
para `input_layernorm` se optimiza conjuntamente para `q/k/v` o para `in_proj_qkv/in_proj_z`.

**Para `o_proj` y `out_proj`:** o se dejan en W8A16 (pagando el no-uso de tensor cores INT8
en esos dos GEMMs, que son ~13% del cómputo de la capa), o se acepta el error. Empezar
dejándolos en W8A16 y medir si vale la pena bajarlos.

### 5.1 SmoothQuant restringido a potencias de dos

**Tensión a resolver:** SmoothQuant multiplica los pesos por floats arbitrarios, lo que
**destruye la grilla diádica** de la sección 3.

**Resolución elegante:** restringir `s` a potencias de dos, `s → 2^round(log2 s)`.

- Preserva la grilla diádica perfectamente → el truco de alineación sigue siendo exacto.
- Cuesta a lo sumo un factor √2 en la migración, despreciable frente a ratios de outlier
  de 20–100×.
- Absorberlo en el RMSNorm BF16 sigue siendo exacto (es sumar al campo de exponente).

**Esto hace que el pipeline entero sea diádico de punta a punta.** Recomendado.

α típico de SmoothQuant: 0.5–0.8. Barrer y elegir por T4.

---

## 6. Riesgos específicos de esta arquitectura

### 6.1 Las 48 capas lineales acumulan error a lo largo de la secuencia

**El riesgo más importante y el más fácil de pasar por alto.**

En una capa de atención softmax, un error de cuantización en `q_proj` afecta el cómputo de
ese token. En una capa Gated DeltaNet, el error en `in_proj_qkv` entra al **estado
recurrente**, que persiste. El error se **compone a lo largo de la secuencia**.

Con `max_position_embeddings = 262144`, esto puede ser catastrófico en contexto largo y
**completamente invisible** en una evaluación de perplejidad sobre WikiText (secuencias
cortas).

**Mitigación:** si hay que gastar bits extra en algún lado, es acá. Dejar `in_proj_qkv` de
las capas lineales con escalas por bloque reales (Diseño A o C) mientras el resto usa B.

**Test obligatorio: T5.**

### 6.2 KV cache: solo 16 de 64 capas lo tienen

Las 48 capas lineales guardan un **estado recurrente de tamaño fijo** (independiente de la
longitud de secuencia) más el buffer de la convolución causal. No tienen KV cache.

Consecuencias:
- `--kv-cache-dtype fp8` rinde ~1/4 de lo que rendiría en un transformer denso de 64 capas.
- Es también la razón por la que el modelo puede declarar 262K de contexto sin explotar en
  memoria.
- El estado del SSM está declarado en FP32 (`mamba_ssm_dtype: float32`). **No tocarlo.**

### 6.3 `in_proj_qkv` es fusionado

Q, K y V tienen rangos dinámicos distintos. Los cortes en filas 2048 y 4096 son múltiplos
de 128, así que las escalas por bloque ya los separan. Al colapsar a per-channel (Diseño B)
la separación se mantiene, porque per-channel es por fila.

**Nunca usar una escala per-tensor sobre este tensor.** Verificar con T6.

### 6.4 La abliteración vive en los residual writers

Ver 1.5. `down_proj`, `out_proj` y `o_proj` cargan una edición de rango 1 cuya magnitud
está por debajo del ruido de E4M3. Re-cuantizar puede perturbarla.

**No es un riesgo de calidad de lenguaje** (perplejidad no lo va a ver) sino de que el
comportamiento del modelo deje de coincidir con el que se midió. Test T10.

### 6.5 No cuantizar de más

Lista de verificación antes de escribir el checkpoint: `A_log`, `dt_bias`, `conv1d`,
`in_proj_a`, `in_proj_b`, todas las norms, `lm_head`, `embed_tokens`, los 333 tensores
`visual.*`. Si alguno de estos termina en INT8, es un bug.

`conv1d.weight` es **3D**. Un pipeline que asuma 2D lo va a corromper silenciosamente.

---

## 7. Pipeline de implementación

### Paso 0 — Diagnóstico previo (sin GPU, minutos)

Correr **T0 antes de escribir cualquier otra cosa.** Determina si el enfoque entero es
viable. Ver sección 8.

### Paso 1 — Verificar el baseline real

Mirar el log de arranque de vLLM y determinar qué kernel se está usando hoy:

- `fp8_marlin` → W8A16, repack a formato Marlin, MMA en FP16
- `w8a8_block_fp8_matmul` (Triton) → emulación con escalas de bloque
- Dequant completo a BF16 → no habría ahorro de memoria (56 GB, no entraría)

**Esto define contra qué se compara la ganancia.** Sin este número, cualquier benchmark
posterior no significa nada.

Dato útil: el `dequant.h` de `gptq_marlin` en vLLM ya implementa el camino rápido con `lop3`
para INT4/INT8/FP4/**FP8** → FP16/BF16. O sea, el baseline ya usa trucos de bits. La
ganancia no puede venir de dequantizar más rápido; tiene que venir de **no dequantizar**.

### Paso 2 — Calibración de SmoothQuant

- 128–512 secuencias de un corpus general (Pile, C4, o un mix con algo del dominio de uso).
- Recolectar el máximo absoluto por canal de entrada de cada GEMM cuantizado.
- Calcular `s_j = max|X_j|^α / max|W_j|^(1−α)`, barrer α ∈ {0.5, 0.6, 0.7, 0.8}.
- **Redondear `s` a la potencia de dos más cercana** (sección 5.1).
- Absorber en los RMSNorm BF16 correspondientes.

### Paso 3 — Repack de los pesos

Por cada tensor cuantizado, por cada bloque 128×128:

1. Leer códigos FP8 crudos + `weight_scale_inv` → invertir a `s_blk`.
2. Aplicar el escalado de SmoothQuant (potencia de dos → suma al exponente, exacto).
3. Aplicar `fp8_e4m3_to_int8_aligned` (sección 3.9).
4. Según el diseño elegido (sección 4), emitir escalas por bloque, per-channel, o
   per-channel + shift diádico.

Es una pasada offline en numpy/torch. No necesita GPU salvo por memoria.

### Paso 4 — Emitir el checkpoint

Formato `compressed-tensors` W8A8:
- `weight` en `int8`
- `weight_scale` según diseño
- Activaciones: cuantización **dinámica por token** (sin escalas estáticas de activación)
- Copiar sin tocar los 792 tensores no cuantizados + las norms modificadas por SmoothQuant
- Actualizar `config.json` → `quantization_config` y la lista `ignore`

### Paso 5 — Validación

Correr la batería de la sección 8 en orden. No pasar de un nivel al siguiente sin que el
anterior cierre.

---

## 8. Plan de pruebas

### Nivel 0 — Estático, sin GPU (minutos)

**T0 — Histograma de `d`. LA PRUEBA DECISIVA. Correr primero.**

Para cada bloque 128×128 de cada tensor cuantizado, computar `d = e_max − e` para todos sus
elementos y acumular el histograma. Reportar por familia de tensores separadamente:
`in_proj_qkv` (y por separado sus rangos Q / K / V), `in_proj_z`, `out_proj`, `q_proj`,
`k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.

Criterios de decisión:

| Resultado | Interpretación | Acción |
|---|---|---|
| p99 de `d` en 4–6 en la mayoría de bloques | distribución sana | seguir, el truco sale casi gratis |
| cola larga de bloques con `d > 8` | outliers aislados | granularidad más fina o extracción de outliers (estilo SpQR) en esos bloques |
| `d > 10` frecuente en un tensor entero | ese tensor no es apto | dejarlo en W8A16 |

**Predicción a falsar:** si la hipótesis gaussiana se sostiene, ~10% de los pesos deberían
tener `d ≥ 4` y la energía en esa región debería ser ~5·10⁻⁴. Si los números reales se
desvían mucho, la sección 3.5 no aplica a este modelo y hay que rehacer el análisis.

Mi apuesta: los tres proyectores fusionados se ven bien; si algo aparece feo va a ser
`down_proj`, que es donde se concentran los outliers del residual stream.

**T1 — Sanidad de códigos.** Verificar que no existe el patrón `0x7F` (NaN) en ningún
tensor. Verificar que `e_max ≥ 1` en todo bloque no nulo.

**T2 — Error de round-trip por tensor.** Dequantizar el original FP8 a FP32, dequantizar el
INT8 nuevo a FP32, comparar:

```
rel_err = ||W_fp8 − W_int8|| _F / ||W_fp8||_F
```

Target Diseño A: **< 1e-3**. Diseño B: < 1e-2. Si A da más de 1e-3, hay un bug en la
implementación del shift, no una limitación del método.

**T3 — Verificación de la predicción de energía.** Medir directamente la fracción de energía
de Frobenius que vive en `d ≥ 4`. Debería dar ~5·10⁻⁴. Es la validación cuantitativa de 3.5.

### Nivel 1 — Numérico por capa (GPU, sin servir)

**T4 — SQNR por capa con activaciones reales.** Capturar activaciones de entrada de un
forward pass en BF16. Correr cada GEMM cuantizado, comparar contra la referencia BF16:

```
SQNR = 10·log10( ||Y_ref||² / ||Y_ref − Y_quant||² )
```

Reportar por capa y por tensor. Buscar outliers. Es también el criterio para elegir α de
SmoothQuant.

**T5 — Deriva del estado del SSM. LA PRUEBA CRÍTICA DE ESTA ARQUITECTURA.**

Para las 48 capas lineales, correr la recurrencia sobre secuencias de longitud creciente y
medir la divergencia del estado contra la referencia BF16:

| Longitud | Métrica | Criterio |
|---|---|---|
| 1K | error relativo del estado | < 1e-3 |
| 8K | " | < 5e-3 |
| 64K | " | < 2e-2 |
| 256K | " | investigar si crece superlinealmente |

**Si el error crece más rápido que √L, hay un problema de acumulación** y hay que subir la
precisión de `in_proj_qkv`. Ninguna evaluación estándar de LLM detecta esto.

**T6 — Balance Q/K/V.** Dentro de `in_proj_qkv`, reportar SQNR por separado para las filas
0–2047 (Q), 2048–4095 (K), 4096–10239 (V). Si difieren en más de ~6 dB, la estrategia de
escalas está mezclando rangos dinámicos que debería separar.

### Nivel 2 — Calidad end-to-end (servir con vLLM)

**T7 — Perplejidad.** WikiText-2-raw, **exactamente 296,907 tokens, KV en BF16**, mismos
settings. Target **6.96**. Aceptar < +1% (o sea ≤ 7.03).

**T8 — Benchmarks de capacidad.** Mismos scripts y settings que el baseline:

| Benchmark | Target | n | Tolerancia |
|---|---|---|---|
| MMLU | 84.7% | 300 | ±1 pt |
| MMLU-Pro (CoT) | 76.8% | 250 | ±1 pt |
| GSM8K (CoT) | 88.7% | 150 | ±1.5 pt |
| CMMLU | 80.8% | 500 | ±1 pt |

**T9 — Contexto largo. No omitir.** T7 y T8 usan secuencias cortas y **no van a detectar la
degradación del SSM**. Correr needle-in-a-haystack o equivalente a 32K / 128K / 256K y
comparar contra el baseline FP8. Este test es el que justifica T5.

**T10 — Preservación de la abliteración.** Si importa que el comportamiento coincida con el
medido: correr un set de probes de rechazo y comparar la tasa contra la del baseline (el
card reporta 0–6% en modo sin thinking). Una desviación grande significa que la
re-cuantización perturbó la edición de rango 1.

**T11 — Visión.** La torre queda en BF16 sin tocar, así que debería ser un no-op. Verificar
igual con una imagen de prueba (descripción + OCR).

**T12 — MTP.** Confirmar que la decodificación especulativa sigue funcionando
(`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`). La cabeza MTP
contiene una capa de atención completa que también hay que convertir consistentemente.

### Nivel 3 — Performance (el punto de todo esto)

**T13 — Microbenchmark de kernels.** Por cada forma de GEMM del modelo
([10240,5120], [6144,5120], [5120,6144], [17408,5120], [5120,17408], [12288,5120],
[1024,5120]) a batch 1, 8, 32, 128. Comparar INT8 W8A8 contra el baseline medido en el
Paso 1.

**T14 — End-to-end.** TTFT y TPOT contra el baseline, a concurrencia 1, 8, 32.

**Expectativa realista:** ganancia significativa en prefill y batch grande
(compute-bound); poca o ninguna en decode a batch 1 (memory-bound, y **W8A8 no ahorra ni
un byte** respecto de FP8 — los ~24.7 GB de linears siguen siendo ~24.7 GB).

> Si el problema real es que no entra en VRAM, INT8 **no es la respuesta**. Hay que ir a W4
> (Marlin W4A16 o QServe W4A8), que baja los linears a ~12.4 GB y el total a ~18 GB.

---

## 9. Triage de fallas

| Síntoma | Causa probable | Acción |
|---|---|---|
| T2 > 1e-3 en Diseño A | bug en el shift o en `e_max` | revisar máscara `0x7F`, manejo de subnormales, dirección del shift |
| T2 alto solo en `down_proj` | outliers del residual stream | T0 lo debería haber anticipado; granularidad más fina en ese tensor |
| T4 con SQNR bajo y uniforme | activaciones, no pesos | subir α de SmoothQuant |
| T4 con SQNR bajo en capas puntuales | outlier de bloque | extracción de outliers en esas capas |
| T5 crece superlinealmente | error compuesto en el SSM | `in_proj_qkv` a Diseño A/C o W8A16 en las capas lineales |
| T7 bien, T9 mal | **exactamente el modo de falla previsto en 6.1** | ídem anterior |
| T7 y T8 bien, T10 mal | perturbación de la abliteración | subir precisión en los 131 residual writers |
| T13 bien, T14 sin ganancia | memory-bound | esperado a batch 1; la ganancia está en prefill/batch |
| Modelo produce basura | probablemente se cuantizó algo de la lista 6.5 | auditar la lista `ignore` del checkpoint emitido |

---

## 10. Alternativas si esto no alcanza

Ordenadas por costo de implementación.

**A. Portar la receta de QoQ / QServe** (arXiv 2405.04532). Cuantización progresiva por
grupos: primero a 8 bits con escalas per-channel, después esos intermedios a 4 bits, de modo
que **todos los GEMM caigan en tensor cores INT8**. Usa un rango protegido de [−119, 119]
que habilita paralelismo a nivel de registro en el dequant INT4→INT8, y un reordenamiento
de cada 32 pesos UINT4 como `w0, w16, w1, w17, …` que permite desempacarlos con tres
operaciones lógicas. Medido en A100. **Nuestro checkpoint ya está a mitad del nivel 1.**
Hay código liberado.

**B. Esquema de Ozaki de 2 slices sobre las activaciones** (arXiv 2306.11975, `ozIMMU`).
Partir el operando en slices enteros, hacer varios GEMM INT8, recombinar:

```
A ≈ A_hi + 2⁻⁷ · A_lo          (dos tensores INT8)
Y  = (A_hi·W)·s + (A_lo·W)·s·2⁻⁷
```

Da ~15 bits efectivos de activación, cubriendo el problema de outliers **sin necesitar
SmoothQuant**. Costo: 2 GEMM INT8 ≈ 1× FP16, o sea empata en velocidad con el baseline
emulado actual pero con mucha mejor numérica que un W8A8 ingenuo. **Plan B si SmoothQuant
no cierra en las capas lineales.** Aplicarlo solo a `in_proj_qkv` de las 48 capas lineales.

Nota de calibración: emular FP64 con este esquema requiere 45–108 GEMM INT8 y sale 3–5×
más lento que DGEMM en A100. Nosotros necesitamos **2**. Es un régimen completamente distinto.

**C. TC-FPx / FP6-LLM** (arXiv 2401.14112, USENIX ATC'24) si en algún momento se quiere
bajar de 8 bits sin pasar por potencias de dos. Bit-splitting para anchos irregulares,
soporta FP6_e3m2 y FP5_e2m2. **Testeado específicamente en A100.**

**D. Marlin W4A16** (arXiv 2408.11743) si el problema es memoria y no cómputo.

**Lo que NO hacer:** GEMM basado en lookup table de productos precalculados. El paper de LUT
Tensor Core (arXiv 2408.06003) encontró que una implementación LUT convencional no alcanza
las ganancias prometidas en hardware existente — por eso terminan proponiendo silicio nuevo.
En Ampere ninguna tabla le gana al MMA. La única LUT útil es la de 256 bytes para el repack
offline.

---

## 11. Referencias

| Tema | Referencia |
|---|---|
| Truco `lop3` para dequant | MARLIN, arXiv 2408.11743 + `gptq_marlin/dequant.h` en vLLM |
| W4A8 con todo en tensor cores INT8 | QServe / QoQ, arXiv 2405.04532 |
| Bit-splitting, anchos irregulares, A100 | FP6-LLM / TC-FPx, arXiv 2401.14112 |
| Emulación de precisión con slices INT8 | DGEMM on Integer Matrix Multiplication Unit, arXiv 2306.11975 |
| Por qué LUT-GEMM no gana en GPU | LUT Tensor Core, arXiv 2408.06003 |
| Alineación de bias de exponente entre formatos | ZeroQuant(4+2), arXiv 2312.08583 |
| INT8 vs FP8 para pesos | van Baalen et al., *FP8 versus INT8 for efficient deep learning inference* |
| Dirección de rechazo (contexto de la abliteración) | Arditi et al. 2024, *Refusal in Language Models Is Mediated by a Single Direction* |

Constantes reales del path rápido de vLLM, por si sirven de referencia:
`MASK = 0x000f000f`, `EX = 0x6400` (half) / `0x4300` (bf16), `SUB = 0x64006400`,
`MUL = 0x2c002c00`, `ADD = 0xd400d400`, aplicadas con `__hsub2` / `__hfma2`.

---

## 12. Orden de trabajo sugerido

1. **T0.** Si falla, nada de lo demás importa. (horas)
2. Paso 1: identificar el baseline real. (minutos)
3. Implementar `fp8_e4m3_to_int8_aligned` + T1, T2, T3. (1 día)
4. Diseño B sin SmoothQuant, T4. Establece el piso. (1 día)
5. Agregar SmoothQuant con `s` en potencias de dos, re-correr T4, barrer α. (2 días)
6. T5 y T6. Decidir si `in_proj_qkv` necesita Diseño A/C. (1 día)
7. Emitir checkpoint, T7 y T8. (1 día)
8. **T9.** No saltearlo. (medio día)
9. T13, T14. Recién acá se sabe si valió la pena. (1 día)
10. T10, T11, T12 antes de considerarlo terminado. (medio día)
