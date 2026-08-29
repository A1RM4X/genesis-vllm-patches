# Diagnóstico y control granular de PN110 / Super-Kernels

Variables de entorno para aislar, medir y depurar **PN110 (FP8→INT8 W8A8)** y las
familias de super kernels en `sm_86` (2× RTX 3090, TP=2).

Todo lo de acá está contrastado contra
`vllm/_genesis/wiring/quantization/patch_PN110_int8_phase_dispatch.py` y contra
`compose/docker-compose.qwen38-27b-fp8.yml`.

---

## 0. La trampa: leer siempre el resumen de carga

**`GENESIS_PN110_SK=0` con `SWAP_ONLY_SK=1` (el default) no convierte NINGUNA
capa.** No es "PN110 con otro kernel": es el baseline FP8/Marlin puro, con PN110
presente pero inerte. Es intencional y está explicado en el código (`SWAP_ONLY_SK`,
línea ~325): el swap ocurre en `process_weights_after_loading` y **libera el peso
Marlin**, así que si `apply` decidiera después no usar SK ya no habría a dónde
volver.

La consecuencia práctica es que una corrida con `SK=0` **no dice nada sobre el
camino INT8**. Antes de sacar cualquier conclusión, mirar esta línea del log:

```
PN110 resumen carga: convertidas=N excluidas=M fallback=F
```

* `convertidas=0` → se midió FP8. Nada de PN110 se ejecutó.
* `convertidas=N` → se midió el camino INT8 sobre N capas *por rank*.

El mismo chequeo vale para `SK_ONLY`, `MAX_LAYERS` y `LAYER_INDEX`: los tres
reducen `convertidas`, y es fácil creer que se está midiendo algo que no se está
midiendo.

---

## 1. Control de ejecución y aislamiento

`Default código` es el que aplica si la variable no está en el entorno.
`Default compose` es lo que pone `compose/docker-compose.qwen38-27b-fp8.yml`.
**No siempre coinciden** — `PN110_SK` es el caso.

| Variable | Var de compose | Default código | Default compose | Tipo | Qué hace |
|---|---|:---:|:---:|:---:|---|
| `GENESIS_ENABLE_PN110_INT8_PHASE_DISPATCH` | `PN110_ENABLE` | — (requerida) | `1` | bool | Opt-in del parche. Sin esto PN110 no se instala. |
| `GENESIS_DISABLE_PN110` | `PN110_DISABLE` | `0` | `0` | bool | Kill-switch, tiene precedencia sobre todo lo demás. |
| `GENESIS_PN110_SK` | `PN110_SK` | **`1`** | **`0`** | bool | Habilita el despacho a super kernels. Ver §0: con `SWAP_ONLY_SK=1`, ponerlo en 0 deja el modelo entero en FP8. |
| `GENESIS_PN110_SK_ONLY` | `PN110_SK_ONLY` | `""` | `""` | lista | Filtro de **inclusión** de familias: `SK-06` o `SK-01,SK-02`. Acepta `SK-05`, `sk05` o `5`. Vacío = sin filtro. |
| `GENESIS_PN110_SK_SKIP` | `PN110_SK_SKIP` | `""` | `""` | lista | Filtro de **exclusión**. `ONLY` tiene precedencia sobre `SKIP`. |
| `GENESIS_PN110_SWAP_ONLY_SK` | `PN110_SWAP_ONLY_SK` | `1` | `1` | bool | Sólo convierte a INT8 las capas que van a correr por super kernel. En `0` vuelve al comportamiento viejo (convierte todo y despacha a cutlass / `int8_hybrid_gemm`), que midió 84.1 tok/s contra 171.6 del baseline. |
| `GENESIS_PN110_MAX_LAYERS` | `PN110_MAX_LAYERS` | sin tope | sin tope | int | Tope de **conversiones por worker**. Ver los detalles en §1.1. |
| `GENESIS_PN110_LAYER_INDEX` | `PN110_LAYER_INDEX` | sin filtro | sin filtro | int | Sólo interviene capas cuyo nombre contenga `layers.<i>.`. Ver §1.1. |
| `GENESIS_PN110_EXCLUDE_LAYERS` | `PN110_EXCLUDE` | `""` | `mtp.layers` | lista | Substrings de nombre de capa a excluir del swap. |
| `GENESIS_PN110_QUANTIZE` | — | `fp8_int8` | — | lista | Qué dtype de origen se acepta: `fp8_int8`, `bf16_int8`. Lista por comas. Un valor no implementado loguea WARNING una vez y se ignora. |
| `GENESIS_PN110_HYBRID` | — | `0` | — | bool | Diseño C diádico (per-channel float + shift int8 por bloque). En `0` se usa la cuantización exacta por bloques de 128x128 en fp32. |
| `GENESIS_P113_MLP_FUSED_SILU` | `P113` | `0` | — | bool | Fusiona `gate_up` + SiLU en el epílogo de SK-05. La permutación del peso se hace en la CARGA; el camino no fusionado con ese layout **lanza a propósito**. |

### 1.1 Detalles de `MAX_LAYERS` y `LAYER_INDEX` que muerden

Los dos se evalúan en `process_weights_after_loading`, por capa, sobre `_usa_sk`.

* **`MAX_LAYERS` cuenta por worker, no por modelo.** El contador es
  `_pn110_summary["converted"]`, que es por proceso: con TP=2, `MAX_LAYERS=16`
  da 16 conversiones **en cada rank**, o sea 16 shards de capa por GPU.
* **`MAX_LAYERS` corta por orden de carga**, no por índice de capa. "Las
  primeras 16 que se hubieran convertido", que con `SK_ONLY=SK-06` son las 16
  primeras `down_proj` pero sin `SK_ONLY` son otras.
* **`LAYER_INDEX` compara por substring `layers.<i>.`**, así que `LAYER_INDEX=1`
  matchea `language_model.model.layers.1.` **y también** `mtp.layers.1.`. Con el
  `EXCLUDE_LAYERS=mtp.layers` del compose eso queda cubierto; sin él, no.
* **Vacío es inerte, no cero.** `PN110_LAYER_INDEX=` deja la variable definida
  con string vacío; `int("")` lanza y el `except` lo traga, así que el filtro no
  se aplica. Es el efecto deseado, pero no está de más saber por qué.
* Los dos filtros funcionan con `SWAP_ONLY_SK=0` también: apagan `_usa_sk`, y
  una capa con `_usa_sk` en falso no se convierte.

---

## 2. Diagnóstico numérico

> **Las dos leen el valor con `== "1"` exacto**, no con el conjunto de verdad
> (`true`/`yes`/`on`) que usan el resto de los flags de PN110. `DIAG_SK=true`
> **no** activa nada. Es una asimetría real del código, no un error de esta doc.

| Variable | Var de compose | Default | Cuándo corre | Qué mide |
|---|---|:---:|---|---|
| `GENESIS_PN110_DIAG_SK` | `PN110_DIAG_SK` | `0` | Carga | **Dos chequeos distintos.** |
| `GENESIS_PN110_DIAG_REF` | `PN110_DIAG_REF` | `0` | Primer forward de cada capa | Extremo a extremo con la activación real. |

### `GENESIS_PN110_DIAG_SK=1`

Emite dos líneas por forma/familia, ambas en la carga, sin tocar el forward:

* `DIAG_Q K=.. N=.. | cos(recon,ref)=.. cos(b_col,w_i8)=..` — desde
  `_build_int8_state`. Valida la **cuantización por bloques 128x128 y el layout**
  contra el peso FP8 de origen que entrega vLLM. Una vez por forma `(K,N)`.
  Sano: `cos(recon,ref)` ≥ 0.999 y `cos(b_col,w_i8)` = 1.000000 exacto.
* `DIAG_SK SK-0x .. | cos(SK,hibrido)=.. escala=..` — desde
  `_genesis_bind_super_kernel`. Corre el **GEMM del super kernel y el
  `int8_hybrid_gemm` sobre el mismo input** y compara. Una vez por familia.
  Sano: cos ≥ 0.9999 y escala ≈ 1.000.

Sirve para separar "el peso quedó mal cuantizado" de "el kernel calcula mal".

### `GENESIS_PN110_DIAG_REF=1`

En la primera pasada de `apply` de cada capa convertida, compara la salida del
camino INT8 completo (quant per-token + GEMM) contra `x @ dequant(peso fp8)`,
**con la activación real de producción**. Loguea:

```
DIAG_REF <capa> M=.. | cos(INT8,fp8ref)=.. escala=.. | x amax/mediana=.. | a_scales min=.. max=..
```

Es el único chequeo que ve lo que los tests offline no pueden ver, porque estos
usan gaussianas y las activaciones reales tienen outliers por canal. El
`amax/mediana` de la fila es justamente el indicador de cuánto castiga el quant
per-token.

**Dos costos.** Guarda el peso dequantizado en fp32 (~170 MB por capa): usar
siempre con `MAX_LAYERS` chico. Y tiene side-effect dentro de la región que
traza `torch.compile`, así que **aborta la captura de cudagraphs**: hay que
correrlo con `--enforce-eager`.

---

## 3. Otras variables que lee el código

Están implementadas pero no son de diagnóstico; se documentan para que la lista
sea completa y nadie las descubra por accidente.

| Variable | Default | Qué hace |
|---|:---:|---|
| `GENESIS_PN110_KL_THRESHOLD` | `0.05` | Gate CK-2.3: si la divergencia KL entre el peso FP8 y el INT8 supera el umbral, la capa **no** se convierte y queda en Marlin. Con escalas por bloque presentes el umbral efectivo se duplica. |
| `GENESIS_PN110_KL_CALIB_TOKENS` | `512` | Filas usadas para calcular esa KL. |
| `GENESIS_PN110_SK_MAX_M` | `64` | Sólo aplica **sin** escalas por bloque: SK se usa hasta este M y arriba de `SK_MIN_BIG_M`; en el medio gana `cutlass_scaled_mm`. Con escalas por bloque el despacho es SK siempre. |
| `GENESIS_PN110_SK_MIN_BIG_M` | `4096` | El otro extremo de esa ventana. |
| `GENESIS_PN110_W8A8_MIN_TOKENS` | `256` | **Muerta.** Se conserva por compatibilidad; el despacho ya no mira M. |

---

## 4. Familias de super kernel

Geometría **per-rank con TP=2**, en `K×N` (K = entrada, N = salida), tal como la
imprime el log de carga. Los conteos son capas por forward.

| Familia | Nombre en el código | Capa del modelo | K×N per-rank | Paralelismo | Capas |
|---|---|---|---|---|---:|
| `SK-01` | `GDN_QKVZ_FUSED_INT8_DIADIC` | `linear_attn.in_proj_qkv` + `in_proj_z`, fusionados | 5120×8192 | MergedColumn | 48 |
| `SK-02` | `GDN_OUT_INT8_SCALED` | `linear_attn.out_proj` | 3072×5120 | Row | 48 |
| `SK-03` | `FA_QKV_FUSED_INT8_DIADIC` | `self_attn.qkv_proj` | 5120×7168 | QKVColumn | 16 |
| `SK-04` | `FA_O_INT8_SCALED` | `self_attn.o_proj` | 3072×5120 | Row | 16 |
| `SK-05` | `MLP_GATEUP_FUSED_INT8_DIADIC` | `mlp.gate_up_proj` | 5120×17408 | MergedColumn | 64 |
| `SK-06` | `MLP_DOWN_INT8_SCALED_RESIDUAL` | `mlp.down_proj` | 8704×5120 | Row + residual fusionado | 64 |
| `SK-07` | `LM_HEAD_VOCAB` | `lm_head` | 5120×124160 global | — | **0 en producción**: el `lm_head` sigue yendo por Marlin |
| `SK-08` | `SSM_CONTROL` | control del GDN | — | — | no es un GEMM lineal |
| `SK-09` | `NORM_EMBED` | `quant_per_token`, el productor canónico de la activación INT8 | — | — | 256 lanzamientos/fwd |
| `SK-10` | `MTP_DRAFT_MIRROR` | `mtp.layers.*`; espeja SK-03 / SK-05 / SK-06 | 5120×7168 | — | excluido por defecto |
| `SK-11` | `VISION_BF16` | passthrough, fuera del camino de texto | 1280×1280 | — | 0 |

**Ojo con `SK-01` y `SK-03`**: SK-01 es el **GDN**, SK-03 es la **atención
completa**. Es fácil invertirlos porque los dos son un QKV fusionado de K=5120.
El discriminador es N: 8192 el GDN, 7168 la atención.

**`SK-10` no es un kernel propio**, es un espejo: liga las capas de la cabeza MTP
al GEMM de SK-03, SK-05 o SK-06 según la capa. En el log aparece como
`SK-05 ... (via SK-10)`.

---

## 5. Recetas

Las variables `PN110_*` son las del compose. Las que no tienen columna de
compose en las tablas de arriba (`QUANTIZE`, `HYBRID`, las de KL y las de
`SK_MAX_M`/`SK_MIN_BIG_M`) hay que ponerlas con su nombre `GENESIS_*` completo,
directo en el entorno del servicio.

```bash
CF=compose/docker-compose.qwen38-27b-fp8.yml

# Baseline FP8, PN110 inerte. Es contra esto que se compara todo.
PN110_DISABLE=1 docker compose -f $CF up -d

# Una sola familia, todo lo demas en FP8.
PN110_SK=1 PN110_SK_ONLY=SK-06 docker compose -f $CF up -d

# Una sola familia y ademas un tope de capas (16 POR RANK).
PN110_SK=1 PN110_SK_ONLY=SK-06 PN110_MAX_LAYERS=16 docker compose -f $CF up -d

# Una capa concreta del modelo.
PN110_SK=1 PN110_SK_ONLY=SK-06 PN110_LAYER_INDEX=1 docker compose -f $CF up -d

# Chequeo numerico de pesos y kernels en la carga (no toca el forward).
PN110_DIAG_SK=1 PN110_SK=1 docker compose -f $CF up -d

# Chequeo de extremo a extremo con activacion real. Pide --enforce-eager
# y un MAX_LAYERS chico: guarda ~170 MB de peso fp32 por capa.
PN110_DIAG_REF=1 PN110_SK=1 PN110_MAX_LAYERS=2 docker compose -f $CF up -d
```

Después de cada una, verificar qué se midió de verdad:

```bash
docker logs genesis-27b-qwen38-fp8 2>&1 | grep -E "resumen carga|resumen kernels|DIAG_"
```
