# Diagnóstico y Control Granular PN110 / Super-Kernels

Guía completa de variables de entorno para control granular, diagnóstico numérico y depuración capa por capa de **PN110 (FP8→INT8 W8A8)** y familias de **Super-Kernels (SK-01 a SK-10)** en Ampere (`sm_86` / Dual RTX 3090).

---

## 1. Variables de Control de Ejecución y Aislamiento

| Variable en Entorno / Compose | Default | Tipo | Descripción |
|---|:---:|:---:|---|
| `GENESIS_ENABLE_PN110_INT8_PHASE_DISPATCH`<br>`(${PN110_ENABLE})` | `1` | bool | Activa o desactiva la infraestructura de despacho por fases INT8 de PN110. |
| `GENESIS_DISABLE_PN110`<br>`(${PN110_DISABLE})` | `0` | bool | Kill-switch maestro. Si se pone en `1`, ninguna capa es intervenida y todo corre en Marlin FP8 stock. |
| `GENESIS_PN110_SK`<br>`(${PN110_SK})` | `0` | bool | Habilita el despacho a super-kernels (PTX/CUBIN embebidos). Si es `0`, PN110 utiliza los caminos nativos/CUTLASS. |
| `GENESIS_PN110_SK_ONLY`<br>`(${PN110_SK_ONLY})` | `""` | string | **Filtro de inclusión de familias**. Permite activar únicamente super-kernels específicos (ej. `SK-06`, o `SK-01,SK-02`). Las demás familias son ignoradas. |
| `GENESIS_PN110_SK_SKIP`<br>`(${PN110_SK_SKIP})` | `""` | string | **Filtro de exclusión de familias**. Omite super-kernels específicos (ej. `SK-05`), dejándolos en el camino stock. |
| `GENESIS_PN110_SWAP_ONLY_SK`<br>`(${PN110_SWAP_ONLY_SK})` | `1` | bool | Si es `1`, **solo convierte pesos a INT8 en capas que tengan super-kernel activo**. Si una capa no tiene super-kernel (o fue excluida por `SK_ONLY`), se conserva en su formato Marlin FP8 original sin degradar precisión ni memoria. |
| `GENESIS_PN110_MAX_LAYERS`<br>`(${PN110_MAX_LAYERS})` | `None` | int | **Tope de capas a convertir**. Si se define (ej. `1`, `4`, `8`), PN110 convierte a INT8 como máximo esa cantidad de capas; todas las capas subsiguientes quedan en FP8. Crucial para búsqueda binaria de acumulación de error. |
| `GENESIS_PN110_LAYER_INDEX`<br>`(${PN110_LAYER_INDEX})` | `None` | int | **Aislamiento de capa única**. Si se define (ej. `0`, `1`, `15`), PN110 solo interviene la capa cuyo nombre contenga `layers.<index>.`. Permite verificar cualquier capa individual del modelo en live inference. |
| `GENESIS_PN110_EXCLUDE_LAYERS`<br>`(${PN110_EXCLUDE})` | `mtp.layers` | string | Subcadena o patrón para excluir capas de la conversión (por defecto excluye la cabeza MTP/draft para no alterar spec-decode). |

---

## 2. Variables de Diagnóstico Numérico y Verificación

| Variable | Default | Momento | Descripción |
|---|:---:|:---:|---|
| `GENESIS_PN110_DIAG_SK` | `0` | Carga de modelo (Bind) | Valida en tiempo de carga (`_build_int8_state`) la cuantización exacta por bloques 128x128 y el layout de los pesos (`b_col`) contra el peso FP8 de vLLM. Loguea `cos(recon, ref)` y `cos(b_col, w_i8)`. |
| `GENESIS_PN110_DIAG_REF`<br>`(${PN110_DIAG_REF})` | `0` | Inferencia (Forward) | En la primera pasada de `apply()`, calcula el producto de referencia contra los pesos FP8 desquantizados (`x @ w_ref`) y lo compara con la salida del kernel INT8. Reporta `cos(INT8, fp8ref)`, ratio de escala y el ratio `amax/mediana` de las activaciones. Nota: aborta CUDA graphs por side-effect; usar con `--enforce-eager`. |

---

## 3. Ejemplo de Uso en Docker Compose

En `compose/docker-compose.qwen38-27b-fp8.yml`, los parámetros se pasan como variables de entorno al comando de lanzamiento:

```bash
# Probar exclusivamente la capa 1 con SK-06 (MLP down_proj):
PN110_SK=1 PN110_SK_ONLY=SK-06 PN110_LAYER_INDEX=1 docker compose -f compose/docker-compose.qwen38-27b-fp8.yml up -d

# Probar hasta 4 capas acumuladas con SK-06:
PN110_SK=1 PN110_SK_ONLY=SK-06 PN110_MAX_LAYERS=4 docker compose -f compose/docker-compose.qwen38-27b-fp8.yml up -d

# Probar SK-01 y SK-02 en todas las capas:
PN110_SK=1 PN110_SK_ONLY=SK-01,SK-02 docker compose -f compose/docker-compose.qwen38-27b-fp8.yml up -d

# Correr diagnóstico numérico de pesos en tiempo de carga:
GENESIS_PN110_DIAG_SK=1 PN110_SK=1 docker compose -f compose/docker-compose.qwen38-27b-fp8.yml up -d
```

---

## 4. Familias de Super-Kernels Disponibles en PN110

| Familia | Operación / Capa | Geometría Típica (TP=2) | Tipo de GEMM |
|---|---|---|---|
| `SK-01` | Attention QKV | $K=5120, N=4096 / 7168$ | ColumnParallel |
| `SK-02` | Attention Out / O-Proj | $K=3072, N=5120$ | RowParallel |
| `SK-03` | Attention QKVZ (GDN) | $K=5120, N=8192$ | ColumnParallel |
| `SK-04` | GDN Out-Proj | $K=3072, N=5120$ | RowParallel |
| `SK-05` | MLP Gate+Up (SwiGLU) | $K=5120, N=17408$ | ColumnParallel Fused |
| `SK-06` | MLP Down-Proj | $K=8704, N=5120$ | RowParallel Residual Fused |
| `SK-07` | MoE Router / Gate | $K=5120, N=2048$ | ColumnParallel |
| `SK-10` | Dispatcher unificado | Dinámico | Wrapper multifamilia |
